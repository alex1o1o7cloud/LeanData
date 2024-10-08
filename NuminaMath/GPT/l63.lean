import Mathlib

namespace train_speed_proof_l63_63129

theorem train_speed_proof
  (length_of_train : ℕ)
  (length_of_bridge : ℕ)
  (time_to_cross_bridge : ℕ)
  (h_train_length : length_of_train = 145)
  (h_bridge_length : length_of_bridge = 230)
  (h_time : time_to_cross_bridge = 30) :
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 18 / 5 = 45 :=
by
  sorry

end train_speed_proof_l63_63129


namespace molecular_weight_correct_l63_63945

-- Define the atomic weights
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01

-- Define the number of atoms of each element
def num_atoms_K : ℕ := 2
def num_atoms_Br : ℕ := 2
def num_atoms_O : ℕ := 4
def num_atoms_H : ℕ := 3
def num_atoms_N : ℕ := 1

-- Calculate the molecular weight
def molecular_weight : ℝ :=
  num_atoms_K * atomic_weight_K +
  num_atoms_Br * atomic_weight_Br +
  num_atoms_O * atomic_weight_O +
  num_atoms_H * atomic_weight_H +
  num_atoms_N * atomic_weight_N

-- Define the expected molecular weight
def expected_molecular_weight : ℝ := 319.04

-- The theorem stating that the calculated molecular weight matches the expected molecular weight
theorem molecular_weight_correct : molecular_weight = expected_molecular_weight :=
  by
  sorry -- Proof is skipped

end molecular_weight_correct_l63_63945


namespace correct_operation_l63_63096

theorem correct_operation (a : ℝ) : a^5 / a^2 = a^3 := by
  -- Proof steps will be supplied here
  sorry

end correct_operation_l63_63096


namespace art_club_students_l63_63703

theorem art_club_students 
    (students artworks_per_student_per_quarter quarters_per_year artworks_in_two_years : ℕ) 
    (h1 : artworks_per_student_per_quarter = 2)
    (h2 : quarters_per_year = 4) 
    (h3 : artworks_in_two_years = 240) 
    (h4 : students * (artworks_per_student_per_quarter * quarters_per_year) * 2 = artworks_in_two_years) :
    students = 15 := 
by
    -- Given conditions for the problem
    sorry

end art_club_students_l63_63703


namespace complex_division_identity_l63_63546

noncomputable def left_hand_side : ℂ := (-2 : ℂ) + (5 : ℂ) * Complex.I / (6 : ℂ) - (3 : ℂ) * Complex.I
noncomputable def right_hand_side : ℂ := - (9 : ℂ) / 15 + (8 : ℂ) / 15 * Complex.I

theorem complex_division_identity : left_hand_side = right_hand_side := 
by
  sorry

end complex_division_identity_l63_63546


namespace ratio_of_cats_to_dogs_sold_l63_63692

theorem ratio_of_cats_to_dogs_sold (cats dogs : ℕ) (h1 : cats = 16) (h2 : dogs = 8) :
  (cats : ℚ) / dogs = 2 / 1 :=
by
  sorry

end ratio_of_cats_to_dogs_sold_l63_63692


namespace mira_jogging_distance_l63_63666

def jogging_speed : ℝ := 5 -- speed in miles per hour
def jogging_hours_per_day : ℝ := 2 -- hours per day
def days_count : ℕ := 5 -- number of days

theorem mira_jogging_distance :
  (jogging_speed * jogging_hours_per_day * days_count : ℝ) = 50 :=
by
  sorry

end mira_jogging_distance_l63_63666


namespace possible_values_of_n_l63_63282

-- Conditions: Definition of equilateral triangles and squares with side length 1
def equilateral_triangle_side_length_1 : Prop := ∀ (a : ℕ), 
  ∃ (triangle : ℕ), triangle * 60 = 180 * (a - 2)

def square_side_length_1 : Prop := ∀ (b : ℕ), 
  ∃ (square : ℕ), square * 90 = 180 * (b - 2)

-- Definition of convex n-sided polygon formed using these pieces
def convex_polygon_formed (n : ℕ) : Prop := 
  ∃ (a b c d : ℕ), 
    a + b + c + d = n ∧ 
    60 * a + 90 * b + 120 * c + 150 * d = 180 * (n - 2)

-- Equivalent proof problem
theorem possible_values_of_n :
  ∃ (n : ℕ), (5 ≤ n ∧ n ≤ 12) ∧ convex_polygon_formed n :=
sorry

end possible_values_of_n_l63_63282


namespace work_problem_l63_63717

theorem work_problem (W : ℕ) (T_AB T_A T_B together_worked alone_worked remaining_work : ℕ)
  (h1 : T_AB = 30)
  (h2 : T_A = 60)
  (h3 : together_worked = 20)
  (h4 : T_B = 30)
  (h5 : remaining_work = W / 3)
  (h6 : alone_worked = 20)
  : alone_worked = 20 :=
by
  /- Proof is not required -/
  sorry

end work_problem_l63_63717


namespace distinct_real_roots_iff_l63_63652

noncomputable def operation (a b : ℝ) : ℝ := a * b^2 - b 

theorem distinct_real_roots_iff (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ operation 1 x1 = k ∧ operation 1 x2 = k) ↔ k > -1/4 :=
by
  sorry

end distinct_real_roots_iff_l63_63652


namespace actual_average_height_correct_l63_63391

theorem actual_average_height_correct : 
  (∃ (avg_height : ℚ), avg_height = 181 ) →
  (∃ (num_boys : ℕ), num_boys = 35) →
  (∃ (incorrect_height : ℚ), incorrect_height = 166) →
  (∃ (actual_height : ℚ), actual_height = 106) →
  (179.29 : ℚ) = 
    (round ((6315 + 106 : ℚ) / 35 * 100) / 100 ) :=
by
sorry

end actual_average_height_correct_l63_63391


namespace largest_difference_l63_63455

noncomputable def A := 3 * (1003 ^ 1004)
noncomputable def B := 1003 ^ 1004
noncomputable def C := 1002 * (1003 ^ 1003)
noncomputable def D := 3 * (1003 ^ 1003)
noncomputable def E := 1003 ^ 1003
noncomputable def F := 1003 ^ 1002

theorem largest_difference : 
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B := 
sorry

end largest_difference_l63_63455


namespace sum_of_constants_l63_63011

variable (a b c : ℝ)

theorem sum_of_constants (h :  2 * (a - 2)^2 + 3 * (b - 3)^2 + 4 * (c - 4)^2 = 0) :
  a + b + c = 9 := 
sorry

end sum_of_constants_l63_63011


namespace find_a_2016_l63_63249

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (n + 1) / n * a n

theorem find_a_2016 (a : ℕ → ℝ) (h : seq a) : a 2016 = 4032 :=
by
  sorry

end find_a_2016_l63_63249


namespace watched_movies_count_l63_63554

theorem watched_movies_count {M : ℕ} (total_books total_movies read_books : ℕ) 
  (h1 : total_books = 15) (h2 : total_movies = 14) (h3 : read_books = 11) 
  (h4 : read_books = M + 1) : M = 10 :=
by
  sorry

end watched_movies_count_l63_63554


namespace mike_total_games_l63_63427

-- Define the number of games Mike went to this year
def games_this_year : ℕ := 15

-- Define the number of games Mike went to last year
def games_last_year : ℕ := 39

-- Prove the total number of games Mike went to
theorem mike_total_games : games_this_year + games_last_year = 54 :=
by
  sorry

end mike_total_games_l63_63427


namespace eval_f_g_at_4_l63_63649

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 12 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem eval_f_g_at_4 : f (g 4) = (25 / 7) * Real.sqrt 21 := by
  sorry

end eval_f_g_at_4_l63_63649


namespace length_segment_pq_l63_63405

theorem length_segment_pq 
  (P Q R S T : ℝ)
  (h1 : (dist P Q + dist P R + dist P S + dist P T = 67))
  (h2 : (dist Q P + dist Q R + dist Q S + dist Q T = 34)) :
  dist P Q = 11 :=
sorry

end length_segment_pq_l63_63405


namespace smallest_solution_proof_l63_63371

noncomputable def smallest_solution (x : ℝ) : ℝ :=
  if x = (1 - Real.sqrt 65) / 4 then x else x

theorem smallest_solution_proof :
  ∃ x : ℝ, (2 * x / (x - 2) + (2 * x^2 - 24) / x = 11) ∧
           (∀ y : ℝ, 2 * y / (y - 2) + (2 * y^2 - 24) / y = 11 → y ≥ (1 - Real.sqrt 65) / 4) ∧
           x = (1 - Real.sqrt 65) /4 :=
sorry

end smallest_solution_proof_l63_63371


namespace find_other_package_size_l63_63509

variable (total_coffee : ℕ)
variable (total_5_ounce_packages : ℕ)
variable (num_other_packages : ℕ)
variable (other_package_size : ℕ)

theorem find_other_package_size
  (h1 : total_coffee = 85)
  (h2 : total_5_ounce_packages = num_other_packages + 2)
  (h3 : num_other_packages = 5)
  (h4 : 5 * total_5_ounce_packages + other_package_size * num_other_packages = total_coffee) :
  other_package_size = 10 :=
sorry

end find_other_package_size_l63_63509


namespace alix_has_15_more_chocolates_than_nick_l63_63198

-- Definitions based on the problem conditions
def nick_chocolates : ℕ := 10
def alix_initial_chocolates : ℕ := 3 * nick_chocolates
def chocolates_taken_by_mom : ℕ := 5
def alix_chocolates_after_mom_took_some : ℕ := alix_initial_chocolates - chocolates_taken_by_mom

-- Statement of the theorem to prove
theorem alix_has_15_more_chocolates_than_nick :
  alix_chocolates_after_mom_took_some - nick_chocolates = 15 :=
sorry

end alix_has_15_more_chocolates_than_nick_l63_63198


namespace s_at_1_l63_63526

def t (x : ℚ) := 5 * x - 12
def s (y : ℚ) := (y + 12) / 5 ^ 2 + 5 * ((y + 12) / 5) - 4

theorem s_at_1 : s 1 = 394 / 25 := by
  sorry

end s_at_1_l63_63526


namespace men_to_complete_work_l63_63266

theorem men_to_complete_work (x : ℕ) (h1 : 10 * 80 = x * 40) : x = 20 :=
by
  sorry

end men_to_complete_work_l63_63266


namespace Damien_jogs_miles_over_three_weeks_l63_63274

theorem Damien_jogs_miles_over_three_weeks :
  (5 * 5) * 3 = 75 :=
by sorry

end Damien_jogs_miles_over_three_weeks_l63_63274


namespace hexagon_six_legal_triangles_hexagon_ten_legal_triangles_hexagon_two_thousand_fourteen_legal_triangles_l63_63890

-- Define a hexagon with legal points and triangles

structure Hexagon :=
  (A B C D E F : ℝ)

-- Legal point occurs when certain conditions on intersection between diagonals hold
def legal_point (h : Hexagon) (x : ℝ) (y : ℝ) : Prop :=
  -- Placeholder, we need to define the exact condition based on problem constraints.
  sorry

-- Function to check if a division is legal based on defined rules
def legal_triangle_division (h : Hexagon) (n : ℕ) : Prop :=
  -- Placeholder, this requires a definition based on how points and triangles are formed
  sorry

-- Prove the specific cases
theorem hexagon_six_legal_triangles (h : Hexagon) : legal_triangle_division h 6 :=
  sorry

theorem hexagon_ten_legal_triangles (h : Hexagon) : legal_triangle_division h 10 :=
  sorry

theorem hexagon_two_thousand_fourteen_legal_triangles (h : Hexagon)  : legal_triangle_division h 2014 :=
  sorry

end hexagon_six_legal_triangles_hexagon_ten_legal_triangles_hexagon_two_thousand_fourteen_legal_triangles_l63_63890


namespace missing_digit_divisibility_by_nine_l63_63259

theorem missing_digit_divisibility_by_nine (x : ℕ) (h : 0 ≤ x ∧ x < 10) :
  9 ∣ (3 + 5 + 2 + 4 + x) → x = 4 :=
by
  sorry

end missing_digit_divisibility_by_nine_l63_63259


namespace johns_umbrellas_in_house_l63_63425

-- Definitions based on the conditions
def umbrella_cost : Nat := 8
def total_amount_paid : Nat := 24
def umbrella_in_car : Nat := 1

-- The goal is to prove that the number of umbrellas in John's house is 2
theorem johns_umbrellas_in_house : 
  (total_amount_paid / umbrella_cost) - umbrella_in_car = 2 :=
by sorry

end johns_umbrellas_in_house_l63_63425


namespace train_length_l63_63560

theorem train_length (v_kmph : ℝ) (t_s : ℝ) (L_p : ℝ) (L_t : ℝ) : 
  (v_kmph = 72) ∧ (t_s = 15) ∧ (L_p = 250) →
  L_t = 50 :=
by
  intro h
  sorry

end train_length_l63_63560


namespace sequence_50th_term_l63_63168

def sequence_term (n : ℕ) : ℕ × ℕ :=
  (5 + (n - 1), n - 1)

theorem sequence_50th_term :
  sequence_term 50 = (54, 49) :=
by
  sorry

end sequence_50th_term_l63_63168


namespace train_speed_kmh_l63_63192

-- Definitions based on the conditions
variables (L V : ℝ)
variable (h1 : L = 10 * V)
variable (h2 : L + 600 = 30 * V)

-- The proof statement, no solution steps, just the conclusion
theorem train_speed_kmh : (V * 3.6) = 108 :=
by
  sorry

end train_speed_kmh_l63_63192


namespace henry_games_total_l63_63144

theorem henry_games_total
    (wins : ℕ)
    (losses : ℕ)
    (draws : ℕ)
    (hw : wins = 2)
    (hl : losses = 2)
    (hd : draws = 10) :
  wins + losses + draws = 14 :=
by
  -- The proof is omitted.
  sorry

end henry_games_total_l63_63144


namespace num_clerks_l63_63996

def manager_daily_salary := 5
def clerk_daily_salary := 2
def num_managers := 2
def total_daily_salary := 16

theorem num_clerks (c : ℕ) (h1 : num_managers * manager_daily_salary + c * clerk_daily_salary = total_daily_salary) : c = 3 :=
by 
  sorry

end num_clerks_l63_63996


namespace evaluate_expression_l63_63827

noncomputable def x : ℚ := 4 / 8
noncomputable def y : ℚ := 5 / 6

theorem evaluate_expression : (8 * x + 6 * y) / (72 * x * y) = 3 / 10 :=
by
  sorry

end evaluate_expression_l63_63827


namespace expenditure_proof_l63_63816

namespace OreoCookieProblem

variables (O C : ℕ) (CO CC : ℕ → ℕ) (total_items cost_difference : ℤ)

def oreo_count_eq : Prop := O = (4 * (65 : ℤ) / 13)
def cookie_count_eq : Prop := C = (9 * (65 : ℤ) / 13)
def oreo_cost (o : ℕ) : ℕ := o * 2
def cookie_cost (c : ℕ) : ℕ := c * 3
def total_item_condition : Prop := O + C = 65
def ratio_condition : Prop := 9 * O = 4 * C
def cost_difference_condition (o_cost c_cost : ℕ) : Prop := cost_difference = (c_cost - o_cost)

theorem expenditure_proof :
  (O + C = 65) →
  (9 * O = 4 * C) →
  (O = 20) →
  (C = 45) →
  cost_difference = (45 * 3 - 20 * 2) →
  cost_difference = 95 :=
by sorry

end OreoCookieProblem

end expenditure_proof_l63_63816


namespace math_problem_l63_63773

noncomputable def x : ℝ := 24

theorem math_problem : ∀ (x : ℝ), x = 3/8 * x + 15 → x = 24 := 
by 
  intro x
  intro h
  sorry

end math_problem_l63_63773


namespace complement_of_M_l63_63746

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x^2 - 2*x > 0 }
def complement (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

theorem complement_of_M :
  complement U M = { x | 0 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end complement_of_M_l63_63746


namespace evaluate_nested_fraction_l63_63716

theorem evaluate_nested_fraction :
  (1 / (3 - (1 / (2 - (1 / (3 - (1 / (2 - (1 / 2))))))))) = 11 / 26 :=
by
  sorry

end evaluate_nested_fraction_l63_63716


namespace diameter_increase_l63_63210

theorem diameter_increase (A A' D D' : ℝ)
  (hA_increase: A' = 4 * A)
  (hA: A = π * (D / 2)^2)
  (hA': A' = π * (D' / 2)^2) :
  D' = 2 * D :=
by 
  sorry

end diameter_increase_l63_63210


namespace positive_slope_asymptote_l63_63843

def hyperbola (x y : ℝ) :=
  Real.sqrt ((x - 1) ^ 2 + (y + 2) ^ 2) - Real.sqrt ((x - 6) ^ 2 + (y + 2) ^ 2) = 4

theorem positive_slope_asymptote :
  ∃ (m : ℝ), m = 0.75 ∧ (∃ x y, hyperbola x y) :=
sorry

end positive_slope_asymptote_l63_63843


namespace meat_division_l63_63206

theorem meat_division (w1 w2 meat : ℕ) (h1 : w1 = 645) (h2 : w2 = 237) (h3 : meat = 1000) :
  ∃ (m1 m2 : ℕ), m1 = 296 ∧ m2 = 704 ∧ w1 + m1 = w2 + m2 := by
  sorry

end meat_division_l63_63206


namespace sin_double_angle_identity_l63_63347

variable (α : Real)

theorem sin_double_angle_identity (h : Real.sin (α + π / 6) = 1 / 3) : 
  Real.sin (2 * α + 5 * π / 6) = 7 / 9 :=
by
  sorry

end sin_double_angle_identity_l63_63347


namespace ellipse_parabola_intersection_l63_63840

theorem ellipse_parabola_intersection (c : ℝ) : 
  (∀ x y : ℝ, (x^2 + (y^2 / 4) = c^2 ∧ y = x^2 - 2 * c) → false) ↔ c > 1 := by
  sorry

end ellipse_parabola_intersection_l63_63840


namespace cylindrical_surface_area_increase_l63_63469

theorem cylindrical_surface_area_increase (x : ℝ) :
  (2 * Real.pi * (10 + x)^2 + 2 * Real.pi * (10 + x) * (5 + x) = 
   2 * Real.pi * 10^2 + 2 * Real.pi * 10 * (5 + x)) →
   (x = -10 + 5 * Real.sqrt 6 ∨ x = -10 - 5 * Real.sqrt 6) :=
by
  intro h
  sorry

end cylindrical_surface_area_increase_l63_63469


namespace steve_reading_pages_l63_63651

theorem steve_reading_pages (total_pages: ℕ) (weeks: ℕ) (reading_days_per_week: ℕ) 
  (reads_on_monday: ℕ) (reads_on_wednesday: ℕ) (reads_on_friday: ℕ) :
  total_pages = 2100 → weeks = 7 → reading_days_per_week = 3 → 
  (reads_on_monday = reads_on_wednesday ∧ reads_on_wednesday = reads_on_friday) → 
  ((weeks * reading_days_per_week) > 0) → 
  (total_pages / (weeks * reading_days_per_week)) = reads_on_monday :=
by
  intro h_total_pages h_weeks h_reading_days_per_week h_reads_on_days h_nonzero
  sorry

end steve_reading_pages_l63_63651


namespace second_and_fourth_rows_identical_l63_63850

def count_occurrences (lst : List ℕ) (a : ℕ) (i : ℕ) : ℕ :=
  (lst.take (i + 1)).count a

def fill_next_row (current_row : List ℕ) : List ℕ :=
  current_row.enum.map (λ ⟨i, a⟩ => count_occurrences current_row a i)

theorem second_and_fourth_rows_identical (first_row : List ℕ) :
  let second_row := fill_next_row first_row 
  let third_row := fill_next_row second_row 
  let fourth_row := fill_next_row third_row 
  second_row = fourth_row :=
by
  sorry

end second_and_fourth_rows_identical_l63_63850


namespace net_effect_transactions_l63_63262

theorem net_effect_transactions {a o : ℝ} (h1 : 3 * a / 4 = 15000) (h2 : 5 * o / 4 = 15000) :
  a + o - (2 * 15000) = 2000 :=
by
  sorry

end net_effect_transactions_l63_63262


namespace problem_statement_l63_63954

theorem problem_statement (a b : ℝ) (h : a + b = 1) : 
  ((∀ (a b : ℝ), a + b = 1 → ab ≤ 1/4) ∧ 
   (∀ (a b : ℝ), ¬(ab ≤ 1/4) → ¬(a + b = 1)) ∧ 
   ¬(∀ (a b : ℝ), ab ≤ 1/4 → a + b = 1) ∧ 
   ¬(∀ (a b : ℝ), ¬(a + b = 1) → ¬(ab ≤ 1/4))) := 
sorry

end problem_statement_l63_63954


namespace distinct_naturals_and_power_of_prime_l63_63205

theorem distinct_naturals_and_power_of_prime (a b : ℕ) (p k : ℕ) (h1 : a ≠ b) (h2 : a^2 + b ∣ b^2 + a) (h3 : ∃ (p : ℕ) (k : ℕ), b^2 + a = p^k) : (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2) :=
sorry

end distinct_naturals_and_power_of_prime_l63_63205


namespace students_voted_both_issues_l63_63660

-- Define the total number of students.
def total_students : ℕ := 150

-- Define the number of students who voted in favor of the first issue.
def voted_first_issue : ℕ := 110

-- Define the number of students who voted in favor of the second issue.
def voted_second_issue : ℕ := 95

-- Define the number of students who voted against both issues.
def voted_against_both : ℕ := 15

-- Theorem: Number of students who voted in favor of both issues is 70.
theorem students_voted_both_issues : 
  ((voted_first_issue + voted_second_issue) - (total_students - voted_against_both)) = 70 :=
by
  sorry

end students_voted_both_issues_l63_63660


namespace find_x_l63_63220

theorem find_x (U : Set ℕ) (A B : Set ℕ) (x : ℕ) 
  (hU : U = Set.univ)
  (hA : A = {1, 4, x})
  (hB : B = {1, x ^ 2})
  (h : compl A ⊂ compl B) :
  x = 0 ∨ x = 2 := 
by 
  sorry

end find_x_l63_63220


namespace sequence_general_formula_l63_63131

/--
A sequence a_n is defined such that the first term a_1 = 3 and the recursive formula 
a_{n+1} = (3 * a_n - 4) / (a_n - 2).

We aim to prove that the general term of the sequence is given by:
a_n = ( (-2)^(n+2) - 1 ) / ( (-2)^n - 1 )
-/
theorem sequence_general_formula (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 3)
  (hr : ∀ n, a (n + 1) = (3 * a n - 4) / (a n - 2)) :
  a n = ( (-2:ℝ)^(n+2) - 1 ) / ( (-2:ℝ)^n - 1) :=
sorry

end sequence_general_formula_l63_63131


namespace find_a_l63_63488

def A : Set ℝ := {-1, 0, 1}
noncomputable def B (a : ℝ) : Set ℝ := {a + 1, 2 * a}

theorem find_a (a : ℝ) : (A ∩ B a = {0}) → a = -1 := by
  sorry

end find_a_l63_63488


namespace opposite_reciprocal_of_neg_five_l63_63383

theorem opposite_reciprocal_of_neg_five : 
  ∀ x : ℝ, x = -5 → - (1 / x) = 1 / 5 :=
by
  sorry

end opposite_reciprocal_of_neg_five_l63_63383


namespace collinear_vectors_l63_63169

theorem collinear_vectors (x : ℝ) :
  (∃ k : ℝ, (2, 4) = (k * 2, k * 4) ∧ (k * 2 = x ∧ k * 4 = 6)) → x = 3 :=
by
  intros h
  sorry

end collinear_vectors_l63_63169


namespace num_values_sum_l63_63036

noncomputable def g : ℝ → ℝ :=
sorry

theorem num_values_sum (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - 2 * x * y - 2 * x + 2) :
  ∃ n s : ℕ, (n = 1 ∧ s = 3 ∧ n * s = 3) :=
sorry

end num_values_sum_l63_63036


namespace restaurant_cost_l63_63402

section Restaurant
variable (total_people kids adults : ℕ) 
variable (meal_cost : ℕ)
variable (total_cost : ℕ)

def calculate_adults (total_people kids : ℕ) : ℕ := 
  total_people - kids

def calculate_total_cost (adults meal_cost : ℕ) : ℕ :=
  adults * meal_cost

theorem restaurant_cost (total_people kids meal_cost : ℕ) :
  total_people = 13 →
  kids = 9 →
  meal_cost = 7 →
  calculate_adults total_people kids = 4 →
  calculate_total_cost 4 meal_cost = 28 :=
by
  intros
  simp [calculate_adults, calculate_total_cost]
  sorry -- Proof would be added here
end Restaurant

end restaurant_cost_l63_63402


namespace cos_double_angle_zero_l63_63503

theorem cos_double_angle_zero
  (θ : ℝ)
  (a : ℝ×ℝ := (1, -Real.cos θ))
  (b : ℝ×ℝ := (1, 2 * Real.cos θ))
  (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  Real.cos (2 * θ) = 0 :=
by sorry

end cos_double_angle_zero_l63_63503


namespace integer_values_not_satisfying_inequality_l63_63549

theorem integer_values_not_satisfying_inequality :
  (∃ x : ℤ, ¬(3 * x^2 + 17 * x + 28 > 25)) ∧ (∃ x1 x2 : ℤ, x1 = -2 ∧ x2 = -1) ∧
  ∀ x : ℤ, (x = -2 ∨ x = -1) -> ¬(3 * x^2 + 17 * x + 28 > 25) :=
by
  sorry

end integer_values_not_satisfying_inequality_l63_63549


namespace domain_of_function_l63_63459

theorem domain_of_function :
  ∀ x : ℝ, (x - 1 ≥ 0) ↔ (x ≥ 1) ∧ (x + 1 ≠ 0) :=
by
  sorry

end domain_of_function_l63_63459


namespace swan_count_l63_63187

theorem swan_count (total_birds : ℕ) (fraction_ducks : ℚ):
  fraction_ducks = 5 / 6 →
  total_birds = 108 →
  ∃ (num_swans : ℕ), num_swans = 18 :=
by
  intro h_fraction_ducks h_total_birds
  sorry

end swan_count_l63_63187


namespace worst_is_father_l63_63953

-- Definitions for players
inductive Player
| father
| sister
| daughter
| son
deriving DecidableEq

open Player

def opposite_sex (p1 p2 : Player) : Bool :=
match p1, p2 with
| father, sister => true
| father, daughter => true
| sister, father => true
| daughter, father => true
| son, sister => true
| son, daughter => true
| daughter, son => true
| sister, son => true
| _, _ => false 

-- Problem conditions
variables (worst best : Player)
variable (twins : Player → Player)
variable (worst_best_twins : twins worst = best)
variable (worst_twin_conditions : opposite_sex (twins worst) best)

-- Goal: Prove that the worst player is the father
theorem worst_is_father : worst = Player.father := by
  sorry

end worst_is_father_l63_63953


namespace valid_parameterizations_l63_63377

theorem valid_parameterizations (y x : ℝ) (t : ℝ) :
  let A := (⟨0, 4⟩ : ℝ × ℝ) + t • (⟨3, 1⟩ : ℝ × ℝ)
  let B := (⟨-4/3, 0⟩ : ℝ × ℝ) + t • (⟨-1, -3⟩ : ℝ × ℝ)
  let C := (⟨1, 7⟩ : ℝ × ℝ) + t • (⟨9, 3⟩ : ℝ × ℝ)
  let D := (⟨2, 10⟩ : ℝ × ℝ) + t • (⟨1/3, 1⟩ : ℝ × ℝ)
  let E := (⟨-4, -8⟩ : ℝ × ℝ) + t • (⟨1/9, 1/3⟩ : ℝ × ℝ)
  (B = (x, y) ∧ D = (x, y) ∧ E = (x, y)) ↔ y = 3 * x + 4 :=
sorry

end valid_parameterizations_l63_63377


namespace amount_cut_off_l63_63361

def initial_length : ℕ := 11
def final_length : ℕ := 7

theorem amount_cut_off : (initial_length - final_length) = 4 :=
by
  sorry

end amount_cut_off_l63_63361


namespace sin_cos_eq_one_l63_63125

open Real

theorem sin_cos_eq_one (x : ℝ) (h0 : 0 ≤ x) (h2 : x < 2 * π) (h : sin x + cos x = 1) :
  x = 0 ∨ x = π / 2 := 
by
  sorry

end sin_cos_eq_one_l63_63125


namespace blanket_rate_l63_63291

/-- 
A man purchased 4 blankets at Rs. 100 each, 
5 blankets at Rs. 150 each, 
and two blankets at an unknown rate x. 
If the average price of the blankets was Rs. 150, 
prove that the unknown rate x is 250. 
-/
theorem blanket_rate (x : ℝ) 
  (h1 : 4 * 100 + 5 * 150 + 2 * x = 11 * 150) : 
  x = 250 := 
sorry

end blanket_rate_l63_63291


namespace ram_work_rate_l63_63866

-- Definitions as given in the problem
variable (W : ℕ) -- Total work can be represented by some natural number W
variable (R M : ℕ) -- Raja's work rate and Ram's work rate, respectively

-- Given conditions
variable (combined_work_rate : R + M = W / 4)
variable (raja_work_rate : R = W / 12)

-- Theorem to be proven
theorem ram_work_rate (combined_work_rate : R + M = W / 4) (raja_work_rate : R = W / 12) : M = W / 6 := 
  sorry

end ram_work_rate_l63_63866


namespace notebook_cost_l63_63963

theorem notebook_cost
  (initial_amount : ℝ)
  (notebook_count : ℕ)
  (pen_count : ℕ)
  (pen_cost : ℝ)
  (remaining_amount : ℝ)
  (total_spent : ℝ)
  (notebook_cost : ℝ) :
  initial_amount = 15 →
  notebook_count = 2 →
  pen_count = 2 →
  pen_cost = 1.5 →
  remaining_amount = 4 →
  total_spent = initial_amount - remaining_amount →
  total_spent = notebook_count * notebook_cost + pen_count * pen_cost →
  notebook_cost = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end notebook_cost_l63_63963


namespace find_f_l63_63698

noncomputable def func_satisfies_eq (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = x * f x - y * f y

theorem find_f (f : ℝ → ℝ) (h : func_satisfies_eq f) : ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end find_f_l63_63698


namespace tangent_line_at_A_tangent_line_through_B_l63_63531

open Real

noncomputable def f (x : ℝ) : ℝ := 4 / x
noncomputable def f' (x : ℝ) : ℝ := -4 / (x^2)

theorem tangent_line_at_A : 
  ∃ m b, m = -1 ∧ b = 4 ∧ (∀ x, 1 ≤ x → (x + b = 4)) :=
sorry

theorem tangent_line_through_B :
  ∃ m b, m = 4 ∧ b = -8 ∧ (∀ x, 1 ≤ x → (4*x + b = 8)) :=
sorry

end tangent_line_at_A_tangent_line_through_B_l63_63531


namespace race_result_l63_63803

-- Defining competitors
inductive Sprinter
| A
| B
| C

open Sprinter

-- Conditions as definitions
def position_changes : Sprinter → Nat
| A => sorry
| B => 5
| C => 6

def finishes_before (s1 s2 : Sprinter) : Prop := sorry

-- Stating the problem as a theorem
theorem race_result :
  position_changes C = 6 →
  position_changes B = 5 →
  finishes_before B A →
  (finishes_before B A ∧ finishes_before A C ∧ finishes_before B C) :=
by
  intros hC hB hBA
  sorry

end race_result_l63_63803


namespace functions_are_computable_l63_63137

def f1 : ℕ → ℕ := λ n => 0
def f2 : ℕ → ℕ := λ n => n + 1
def f3 : ℕ → ℕ := λ n => max 0 (n - 1)
def f4 : ℕ → ℕ := λ n => n % 2
def f5 : ℕ → ℕ := λ n => n * 2
def f6 : ℕ × ℕ → ℕ := λ (m, n) => if m ≤ n then 1 else 0

theorem functions_are_computable :
  (Computable f1) ∧
  (Computable f2) ∧
  (Computable f3) ∧
  (Computable f4) ∧
  (Computable f5) ∧
  (Computable f6) := by
  sorry

end functions_are_computable_l63_63137


namespace proof_problem_l63_63979

variable {a1 a2 b1 b2 b3 : ℝ}

-- Condition: -2, a1, a2, -8 form an arithmetic sequence
def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
  a2 - a1 = -2 / 3 * (-2 - 8)

-- Condition: -2, b1, b2, b3, -8 form a geometric sequence
def geometric_sequence (b1 b2 b3 : ℝ) : Prop :=
  b2^2 = (-2) * (-8) ∧ b1^2 = (-2) * b2 ∧ b3^2 = b2 * (-8)

theorem proof_problem (h1 : arithmetic_sequence a1 a2) (h2 : geometric_sequence b1 b2 b3) : b2 * (a2 - a1) = 8 :=
by
  admit -- Convert to sorry to skip the proof

end proof_problem_l63_63979


namespace Chloe_second_round_points_l63_63014

-- Conditions
def firstRoundPoints : ℕ := 40
def lastRoundPointsLost : ℕ := 4
def totalPoints : ℕ := 86
def secondRoundPoints : ℕ := 50

-- Statement to prove: Chloe scored 50 points in the second round
theorem Chloe_second_round_points :
  firstRoundPoints + secondRoundPoints - lastRoundPointsLost = totalPoints :=
by {
  -- Proof (not required, skipping with sorry)
  sorry
}

end Chloe_second_round_points_l63_63014


namespace mixed_doubles_pairing_l63_63106

theorem mixed_doubles_pairing: 
  let males := 5
  let females := 4
  let choose_males := Nat.choose males 2
  let choose_females := Nat.choose females 2
  let arrangements := Nat.factorial 2
  choose_males * choose_females * arrangements = 120 := by
  sorry

end mixed_doubles_pairing_l63_63106


namespace final_hair_length_is_14_l63_63061

def initial_hair_length : ℕ := 24

def half_hair_cut (l : ℕ) : ℕ := l / 2

def hair_growth (l : ℕ) : ℕ := l + 4

def final_hair_cut (l : ℕ) : ℕ := l - 2

theorem final_hair_length_is_14 :
  final_hair_cut (hair_growth (half_hair_cut initial_hair_length)) = 14 := by
  sorry

end final_hair_length_is_14_l63_63061


namespace inequality_8xyz_leq_1_equality_cases_8xyz_eq_1_l63_63357

theorem inequality_8xyz_leq_1 (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z ≤ 1 :=
sorry

theorem equality_cases_8xyz_eq_1 (x y z : ℝ) (h1 : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z = 1 ↔ 
  (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨ 
  (x = -1/2 ∧ y = -1/2 ∧ z = 1/2) ∨ 
  (x = -1/2 ∧ y = 1/2 ∧ z = -1/2) ∨ 
  (x = 1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end inequality_8xyz_leq_1_equality_cases_8xyz_eq_1_l63_63357


namespace dispatch_plans_l63_63574

theorem dispatch_plans (students : Finset ℕ) (h : students.card = 6) :
  ∃ (plans : Finset (Finset ℕ)), plans.card = 180 :=
by
  sorry

end dispatch_plans_l63_63574


namespace length_of_segment_BD_is_sqrt_3_l63_63926

open Real

-- Define the triangle ABC and the point D according to the problem conditions
def triangle_ABC (A B C : ℝ × ℝ) :=
  B.1 = 0 ∧ B.2 = 0 ∧
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = 3 ∧
  (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = 7 ∧
  C.2 = 0 ∧ (A.1 - C.1) ^ 2 + A.2 ^ 2 = 10

def point_D (A B C D : ℝ × ℝ) :=
  ∃ BD DC : ℝ, BD + DC = sqrt 7 ∧
  BD / DC = sqrt 3 / sqrt 7 ∧
  D.1 = BD / sqrt 7 ∧ D.2 = 0

-- The theorem to prove
theorem length_of_segment_BD_is_sqrt_3 (A B C D : ℝ × ℝ)
  (h₁ : triangle_ABC A B C)
  (h₂ : point_D A B C D) :
  (sqrt ((D.1 - B.1) ^ 2 + (D.2 - B.2) ^ 2)) = sqrt 3 :=
sorry

end length_of_segment_BD_is_sqrt_3_l63_63926


namespace soap_box_length_l63_63627

def VolumeOfEachSoapBox (L : ℝ) := 30 * L
def VolumeOfCarton := 25 * 42 * 60
def MaximumSoapBoxes := 300

theorem soap_box_length :
  ∀ L : ℝ,
  MaximumSoapBoxes * VolumeOfEachSoapBox L = VolumeOfCarton → 
  L = 7 :=
by
  intros L h
  sorry

end soap_box_length_l63_63627


namespace num_cats_l63_63697

-- Definitions based on conditions
variables (C S K Cap : ℕ)
variable (heads : ℕ) (legs : ℕ)

-- Conditions as equations
axiom heads_eq : C + S + K + Cap = 16
axiom legs_eq : 4 * C + 2 * S + 2 * K + 1 * Cap = 41

-- Given values from the problem
axiom K_val : K = 1
axiom Cap_val : Cap = 1

-- The proof goal in terms of satisfying the number of cats
theorem num_cats : C = 5 :=
by
  sorry

end num_cats_l63_63697


namespace necessary_not_sufficient_condition_l63_63090

variable (a : ℝ) (D : Set ℝ)

def p : Prop := a ∈ D
def q : Prop := ∃ x₀ : ℝ, x₀^2 - a * x₀ - a ≤ -3

theorem necessary_not_sufficient_condition (h : p a D → q a) : D = {x : ℝ | x < -4 ∨ x > 0} :=
sorry

end necessary_not_sufficient_condition_l63_63090


namespace compensation_problem_l63_63517

namespace CompensationProof

variables (a b c : ℝ)

def geometric_seq_with_ratio_1_by_2 (a b c : ℝ) : Prop :=
  c = (1/2) * b ∧ b = (1/2) * a

def total_compensation_eq (a b c : ℝ) : Prop :=
  4 * c + 2 * b + a = 50

theorem compensation_problem :
  total_compensation_eq a b c ∧ geometric_seq_with_ratio_1_by_2 a b c → c = 50 / 7 :=
sorry

end CompensationProof

end compensation_problem_l63_63517


namespace cannot_form_3x3_square_l63_63419

def square_pieces (squares : ℕ) (rectangles : ℕ) (triangles : ℕ) := 
  squares = 4 ∧ rectangles = 1 ∧ triangles = 1

def area (squares : ℕ) (rectangles : ℕ) (triangles : ℕ) : ℕ := 
  squares * 1 * 1 + rectangles * 2 * 1 + triangles * (1 * 1 / 2)

theorem cannot_form_3x3_square : 
  ∀ squares rectangles triangles, 
  square_pieces squares rectangles triangles → 
  area squares rectangles triangles < 9 := by
  intros squares rectangles triangles h
  unfold square_pieces at h
  unfold area
  sorry

end cannot_form_3x3_square_l63_63419


namespace min_stamps_needed_l63_63610

theorem min_stamps_needed {c f : ℕ} (h : 3 * c + 4 * f = 33) : c + f = 9 :=
sorry

end min_stamps_needed_l63_63610


namespace value_of_a_l63_63229

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 :=
by
  sorry

end value_of_a_l63_63229


namespace pentagon_area_l63_63634

theorem pentagon_area 
  (edge_length : ℝ) 
  (triangle_height : ℝ) 
  (n_pentagons : ℕ) 
  (equal_convex_pentagons : ℕ) 
  (pentagon_area : ℝ) : 
  edge_length = 5 ∧ triangle_height = 2 ∧ n_pentagons = 5 ∧ equal_convex_pentagons = 5 → pentagon_area = 30 := 
by
  sorry

end pentagon_area_l63_63634


namespace probability_team_A_3_points_probability_team_A_1_point_probability_combined_l63_63017

namespace TeamProbabilities

noncomputable def P_team_A_3_points : ℚ :=
  (1 / 3) * (1 / 3) * (1 / 3)

noncomputable def P_team_A_1_point : ℚ :=
  (1 / 3) * (2 / 3) * (2 / 3) + (2 / 3) * (1 / 3) * (2 / 3) + (2 / 3) * (2 / 3) * (1 / 3)

noncomputable def P_team_A_2_points : ℚ :=
  (1 / 3) * (1 / 3) * (2 / 3) + (1 / 3) * (2 / 3) * (1 / 3) + (2 / 3) * (1 / 3) * (1 / 3)

noncomputable def P_team_B_1_point : ℚ :=
  (1 / 2) * (2 / 3) * (3 / 4) + (1 / 2) * (1 / 3) * (3 / 4) + (1 / 2) * (2 / 3) * (1 / 4) + (1 / 2) * (2 / 3) * (1 / 4) +
  (1 / 2) * (1 / 3) * (1 / 4) + (1 / 2) * (1 / 3) * (3 / 4) + (2 / 3) * (2 / 3) * (1 / 4) + (2 / 3) * (1 / 3) * (1 / 4)

noncomputable def combined_probability : ℚ :=
  P_team_A_2_points * P_team_B_1_point

theorem probability_team_A_3_points :
  P_team_A_3_points = 1 / 27 := by
  sorry

theorem probability_team_A_1_point :
  P_team_A_1_point = 4 / 9 := by
  sorry

theorem probability_combined :
  combined_probability = 11 / 108 := by
  sorry

end TeamProbabilities

end probability_team_A_3_points_probability_team_A_1_point_probability_combined_l63_63017


namespace cube_edge_length_and_volume_l63_63964

variable (edge_length : ℕ)

def cube_edge_total_length (edge_length : ℕ) : ℕ := edge_length * 12
def cube_volume (edge_length : ℕ) : ℕ := edge_length * edge_length * edge_length

theorem cube_edge_length_and_volume (h : cube_edge_total_length edge_length = 96) :
  edge_length = 8 ∧ cube_volume edge_length = 512 :=
by
  sorry

end cube_edge_length_and_volume_l63_63964


namespace mean_median_mode_relation_l63_63739

-- Defining the data set of the number of fish caught in twelve outings.
def fish_catches : List ℕ := [3, 0, 2, 2, 1, 5, 3, 0, 1, 4, 3, 3]

-- Proof statement to show the relationship among mean, median and mode.
theorem mean_median_mode_relation (hs : fish_catches = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5]) :
  let mean := (fish_catches.sum : ℚ) / fish_catches.length
  let median := (fish_catches.nthLe 5 sorry + fish_catches.nthLe 6 sorry : ℚ) / 2
  let mode := 3
  mean < median ∧ median < mode := by
  -- Placeholder for the proof. Details are skipped here.
  sorry

end mean_median_mode_relation_l63_63739


namespace decimal_to_binary_thirteen_l63_63156

theorem decimal_to_binary_thirteen : (13 : ℕ) = 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by
  sorry

end decimal_to_binary_thirteen_l63_63156


namespace miles_driven_l63_63035

theorem miles_driven (rental_fee charge_per_mile total_amount_paid : ℝ) (h₁ : rental_fee = 20.99) (h₂ : charge_per_mile = 0.25) (h₃ : total_amount_paid = 95.74) :
  (total_amount_paid - rental_fee) / charge_per_mile = 299 :=
by
  -- Placeholder for proof
  sorry

end miles_driven_l63_63035


namespace dimension_tolerance_l63_63223

theorem dimension_tolerance (base_dim : ℝ) (pos_tolerance : ℝ) (neg_tolerance : ℝ) 
  (max_dim : ℝ) (min_dim : ℝ) 
  (h_base : base_dim = 7) 
  (h_pos_tolerance : pos_tolerance = 0.05) 
  (h_neg_tolerance : neg_tolerance = 0.02) 
  (h_max_dim : max_dim = base_dim + pos_tolerance) 
  (h_min_dim : min_dim = base_dim - neg_tolerance) :
  max_dim = 7.05 ∧ min_dim = 6.98 :=
by
  sorry

end dimension_tolerance_l63_63223


namespace sports_club_total_members_l63_63257

theorem sports_club_total_members :
  ∀ (B T Both Neither Total : ℕ),
    B = 17 → T = 19 → Both = 10 → Neither = 2 → Total = B + T - Both + Neither → Total = 28 :=
by
  intros B T Both Neither Total hB hT hBoth hNeither hTotal
  rw [hB, hT, hBoth, hNeither] at hTotal
  exact hTotal

end sports_club_total_members_l63_63257


namespace rectangular_garden_width_l63_63234

variable (w : ℕ)

/-- The length of a rectangular garden is three times its width.
Given that the area of the rectangular garden is 768 square meters,
prove that the width of the garden is 16 meters. -/
theorem rectangular_garden_width
  (h1 : 768 = w * (3 * w)) :
  w = 16 := by
  sorry

end rectangular_garden_width_l63_63234


namespace ratio_equality_l63_63161

theorem ratio_equality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : (y + 1) / (x + z) = (x + y + 2) / (z + 1))
  (h8 : (x + 1) / y = (y + 1) / (x + z)) :
  (x + 1) / y = 1 :=
by
  sorry

end ratio_equality_l63_63161


namespace three_digit_number_is_382_l63_63654

theorem three_digit_number_is_382 
  (x : ℕ) 
  (h1 : x >= 100 ∧ x < 1000) 
  (h2 : 7000 + x - (10 * x + 7) = 3555) : 
  x = 382 :=
by 
  sorry

end three_digit_number_is_382_l63_63654


namespace coins_of_each_type_l63_63066

theorem coins_of_each_type (x : ℕ) (h : x + x / 2 + x / 4 = 70) : x = 40 :=
sorry

end coins_of_each_type_l63_63066


namespace intersection_correct_l63_63941

-- Define sets M and N
def M := {x : ℝ | x < 1}
def N := {x : ℝ | Real.log (2 * x + 1) > 0}

-- Define the intersection of M and N
def M_intersect_N := {x : ℝ | 0 < x ∧ x < 1}

-- Prove that M_intersect_N is the correct intersection
theorem intersection_correct : M ∩ N = M_intersect_N :=
by
  sorry

end intersection_correct_l63_63941


namespace min_positive_numbers_l63_63219

theorem min_positive_numbers (n : ℕ) (numbers : ℕ → ℤ) 
  (h_length : n = 103) 
  (h_consecutive : ∀ i : ℕ, i < n → (∃ (p1 p2 : ℕ), p1 < 5 ∧ p2 < 5 ∧ p1 ≠ p2 ∧ numbers (i + p1) > 0 ∧ numbers (i + p2) > 0)) :
  ∃ (min_positive : ℕ), min_positive = 42 :=
by
  sorry

end min_positive_numbers_l63_63219


namespace sample_size_l63_63967

theorem sample_size (T : ℕ) (f_C : ℚ) (samples_C : ℕ) (n : ℕ) 
    (hT : T = 260)
    (hfC : f_C = 3 / 13)
    (hsamples_C : samples_C = 3) : n = 13 :=
by
  -- Proof goes here
  sorry

end sample_size_l63_63967


namespace bridget_bought_17_apples_l63_63605

noncomputable def total_apples (x : ℕ) : Prop :=
  (2 * x / 3) - 5 = 6

theorem bridget_bought_17_apples : ∃ x : ℕ, total_apples x ∧ x = 17 :=
  sorry

end bridget_bought_17_apples_l63_63605


namespace administrative_staff_drawn_in_stratified_sampling_l63_63644

theorem administrative_staff_drawn_in_stratified_sampling
  (total_staff : ℕ)
  (full_time_teachers : ℕ)
  (administrative_staff : ℕ)
  (logistics_personnel : ℕ)
  (sample_size : ℕ)
  (h_total : total_staff = 320)
  (h_teachers : full_time_teachers = 248)
  (h_admin : administrative_staff = 48)
  (h_logistics : logistics_personnel = 24)
  (h_sample : sample_size = 40)
  : (administrative_staff * (sample_size / total_staff) = 6) :=
by
  -- mathematical proof goes here
  sorry

end administrative_staff_drawn_in_stratified_sampling_l63_63644


namespace determine_m_l63_63597

-- Define the conditions: the quadratic equation and the sum of roots
def quadratic_eq (x m : ℝ) : Prop :=
  x^2 + m * x + 2 = 0

def sum_of_roots (x1 x2 : ℝ) : ℝ := x1 + x2

-- Problem Statement: Prove that m = 4
theorem determine_m (x1 x2 m : ℝ) 
  (h1 : quadratic_eq x1 m) 
  (h2 : quadratic_eq x2 m)
  (h3 : sum_of_roots x1 x2 = -4) : 
  m = 4 :=
by
  sorry

end determine_m_l63_63597


namespace complex_numbers_not_comparable_l63_63820

-- Definitions based on conditions
def is_real (z : ℂ) : Prop := ∃ r : ℝ, z = r
def is_not_entirely_real (z : ℂ) : Prop := ¬ is_real z

-- Proof problem statement
theorem complex_numbers_not_comparable (z1 z2 : ℂ) (h1 : is_not_entirely_real z1) (h2 : is_not_entirely_real z2) : 
  ¬ (z1.re = z2.re ∧ z1.im = z2.im) :=
sorry

end complex_numbers_not_comparable_l63_63820


namespace chuck_distance_l63_63533

theorem chuck_distance
  (total_time : ℝ) (out_speed : ℝ) (return_speed : ℝ) (D : ℝ)
  (h1 : total_time = 3)
  (h2 : out_speed = 16)
  (h3 : return_speed = 24)
  (h4 : D / out_speed + D / return_speed = total_time) :
  D = 28.80 :=
by
  sorry

end chuck_distance_l63_63533


namespace integer_solution_for_x_l63_63193

theorem integer_solution_for_x (x : ℤ) : 
  (∃ y z : ℤ, x = 7 * y + 3 ∧ x = 5 * z + 2) ↔ 
  (∃ t : ℤ, x = 35 * t + 17) :=
by
  sorry

end integer_solution_for_x_l63_63193


namespace exists_decreasing_lcm_sequence_l63_63944

theorem exists_decreasing_lcm_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j, i < j → a i < a j) ∧ 
    (∀ i : Fin 99, Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))) :=
sorry

end exists_decreasing_lcm_sequence_l63_63944


namespace find_value_of_a_l63_63538

theorem find_value_of_a (b : ℤ) (q : ℚ) (a : ℤ) (h₁ : b = 2120) (h₂ : q = 0.5) (h₃ : (a : ℚ) / b = q) : a = 1060 :=
sorry

end find_value_of_a_l63_63538


namespace maria_must_earn_l63_63856

-- Define the given conditions
def retail_price : ℕ := 600
def maria_savings : ℕ := 120
def mother_contribution : ℕ := 250

-- Total amount Maria has from savings and her mother's contribution
def total_savings : ℕ := maria_savings + mother_contribution

-- Prove that Maria must earn $230 to be able to buy the bike
theorem maria_must_earn : 600 - total_savings = 230 :=
by sorry

end maria_must_earn_l63_63856


namespace find_interest_rate_l63_63971

theorem find_interest_rate (P : ℕ) (diff : ℕ) (T : ℕ) (I2_rate : ℕ) (r : ℚ) 
  (hP : P = 15000) (hdiff : diff = 900) (hT : T = 2) (hI2_rate : I2_rate = 12)
  (h : P * (r / 100) * T = P * (I2_rate / 100) * T + diff) :
  r = 15 :=
sorry

end find_interest_rate_l63_63971


namespace total_children_with_cats_l63_63497

variable (D C B : ℕ)
variable (h1 : D = 18)
variable (h2 : B = 6)
variable (h3 : D + C + B = 30)

theorem total_children_with_cats : C + B = 12 := by
  sorry

end total_children_with_cats_l63_63497


namespace max_value_of_reciprocal_powers_l63_63047

variable {R : Type*} [CommRing R]
variables (s q r₁ r₂ : R)

-- Condition: the roots of the polynomial
def is_roots_of_polynomial (s q r₁ r₂ : R) : Prop :=
  r₁ + r₂ = s ∧ r₁ * r₂ = q ∧ (r₁ + r₂ = r₁ ^ 2 + r₂ ^ 2) ∧ (r₁ + r₂ = r₁^10 + r₂^10)

-- The theorem that needs to be proven
theorem max_value_of_reciprocal_powers (s q r₁ r₂ : ℝ) (h : is_roots_of_polynomial s q r₁ r₂):
  (∃ r₁ r₂, r₁ + r₂ = s ∧ r₁ * r₂ = q ∧
             r₁ + r₂ = r₁^2 + r₂^2 ∧
             r₁ + r₂ = r₁^10 + r₂^10) →
  (r₁^ 11 ≠ 0 ∧ r₂^11 ≠ 0 ∧
  ((1 / r₁^11) + (1 / r₂^11) = 2)) :=
by
  sorry

end max_value_of_reciprocal_powers_l63_63047


namespace initial_toys_count_l63_63390

-- Definitions for the conditions
def initial_toys (X : ℕ) : ℕ := X
def lost_toys (X : ℕ) : ℕ := X - 6
def found_toys (X : ℕ) : ℕ := (lost_toys X) + 9
def borrowed_toys (X : ℕ) : ℕ := (found_toys X) + 5
def traded_toys (X : ℕ) : ℕ := (borrowed_toys X) - 3

-- Statement to prove
theorem initial_toys_count (X : ℕ) : traded_toys X = 43 → X = 38 :=
by
  -- Proof to be filled in
  sorry

end initial_toys_count_l63_63390


namespace last_two_digits_of_9_pow_2008_l63_63046

theorem last_two_digits_of_9_pow_2008 : (9 ^ 2008) % 100 = 21 := 
by
  sorry

end last_two_digits_of_9_pow_2008_l63_63046


namespace base_unit_digit_l63_63845

def unit_digit (n : ℕ) : ℕ := n % 10

theorem base_unit_digit (x : ℕ) :
  unit_digit ((x^41) * (41^14) * (14^87) * (87^76)) = 4 →
  unit_digit x = 1 :=
by
  sorry

end base_unit_digit_l63_63845


namespace function_increasing_on_interval_l63_63677

theorem function_increasing_on_interval :
  ∀ x : ℝ, (1 / 2 < x) → (x > 0) → (8 * x - 1 / (x^2)) > 0 :=
sorry

end function_increasing_on_interval_l63_63677


namespace not_divisible_by_1000_pow_m_minus_1_l63_63761

theorem not_divisible_by_1000_pow_m_minus_1 (m : ℕ) : ¬ (1000^m - 1 ∣ 1998^m - 1) :=
sorry

end not_divisible_by_1000_pow_m_minus_1_l63_63761


namespace percentage_discount_total_amount_paid_l63_63782

variable (P Q : ℝ)

theorem percentage_discount (h₁ : P > Q) (h₂ : Q > 0) :
  100 * ((P - Q) / P) = 100 * (P - Q) / P :=
sorry

theorem total_amount_paid (h₁ : P > Q) (h₂ : Q > 0) :
  10 * Q = 10 * Q :=
sorry

end percentage_discount_total_amount_paid_l63_63782


namespace no_positive_integer_n_satisfies_conditions_l63_63876

theorem no_positive_integer_n_satisfies_conditions :
  ¬ ∃ (n : ℕ), (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
by
  sorry

end no_positive_integer_n_satisfies_conditions_l63_63876


namespace arithmetic_sequence_line_l63_63033

theorem arithmetic_sequence_line (A B C x y : ℝ) :
  (2 * B = A + C) → (A * 1 + B * -2 + C = 0) :=
by
  intros h
  sorry

end arithmetic_sequence_line_l63_63033


namespace primary_college_employee_relation_l63_63364

theorem primary_college_employee_relation
  (P C N : ℕ)
  (hN : N = 20 + P + C)
  (h_illiterate_wages_before : 20 * 25 = 500)
  (h_illiterate_wages_after : 20 * 10 = 200)
  (h_primary_wages_before : P * 40 = P * 40)
  (h_primary_wages_after : P * 25 = P * 25)
  (h_college_wages_before : C * 50 = C * 50)
  (h_college_wages_after : C * 60 = C * 60)
  (h_avg_decrease : (500 + 40 * P + 50 * C) / N - (200 + 25 * P + 60 * C) / N = 10) :
  15 * P - 10 * C = 10 * N - 300 := 
by
  sorry

end primary_college_employee_relation_l63_63364


namespace eval_expression_l63_63022

theorem eval_expression (x y z : ℝ) 
  (h1 : z = y - 11) 
  (h2 : y = x + 3) 
  (h3 : x = 5)
  (h4 : x + 2 ≠ 0) 
  (h5 : y - 3 ≠ 0) 
  (h6 : z + 7 ≠ 0) : 
  ( (x + 3) / (x + 2) * (y - 1) / (y - 3) * (z + 9) / (z + 7) ) = 2.4 := 
by
  sorry

end eval_expression_l63_63022


namespace count_ways_to_complete_20160_l63_63899

noncomputable def waysToComplete : Nat :=
  let choices_for_last_digit := 5
  let choices_for_first_three_digits := 9^3
  choices_for_last_digit * choices_for_first_three_digits

theorem count_ways_to_complete_20160 (choices : Fin 9 → Fin 9) : waysToComplete = 3645 := by
  sorry

end count_ways_to_complete_20160_l63_63899


namespace inverse_of_A_cubed_l63_63914

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -2,  3],
    ![  0,  1]]

theorem inverse_of_A_cubed :
  (A_inv ^ 3) = ![![ -8,  9],
                    ![  0,  1]] :=
by sorry

end inverse_of_A_cubed_l63_63914


namespace peanuts_difference_is_correct_l63_63835

-- Define the number of peanuts Jose has
def Jose_peanuts : ℕ := 85

-- Define the number of peanuts Kenya has
def Kenya_peanuts : ℕ := 133

-- Define the difference in the number of peanuts between Kenya and Jose
def peanuts_difference : ℕ := Kenya_peanuts - Jose_peanuts

-- Prove that the number of peanuts Kenya has minus the number of peanuts Jose has is equal to 48
theorem peanuts_difference_is_correct : peanuts_difference = 48 := by
  sorry

end peanuts_difference_is_correct_l63_63835


namespace optimal_playground_dimensions_and_area_l63_63982

theorem optimal_playground_dimensions_and_area:
  ∃ (l w : ℝ), 2 * l + 2 * w = 380 ∧ l ≥ 100 ∧ w ≥ 60 ∧ l * w = 9000 :=
by
  sorry

end optimal_playground_dimensions_and_area_l63_63982


namespace groupDivisionWays_l63_63470

-- Definitions based on conditions
def numDogs : ℕ := 12
def group1Size : ℕ := 4
def group2Size : ℕ := 5
def group3Size : ℕ := 3
def fluffy : ℕ := 1 -- Fluffy's assigned position
def nipper : ℕ := 2 -- Nipper's assigned position

-- Function to compute binomial coefficients
def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom (n+1) k

-- Theorem to prove the number of ways to form the groups
theorem groupDivisionWays :
  (binom 10 3 * binom 7 4) = 4200 :=
by
  sorry

end groupDivisionWays_l63_63470


namespace employee_salary_proof_l63_63251

variable (x : ℝ) (M : ℝ) (P : ℝ)

theorem employee_salary_proof (h1 : x + 1.2 * x + 1.8 * x = 1500)
(h2 : M = 1.2 * x)
(h3 : P = 1.8 * x)
: x = 375 ∧ M = 450 ∧ P = 675 :=
sorry

end employee_salary_proof_l63_63251


namespace solve_for_x_and_y_l63_63164

theorem solve_for_x_and_y (x y : ℚ) (h : (1 / 6) + (6 / x) = (14 / x) + (1 / 14) + y) : x = 84 ∧ y = 0 :=
sorry

end solve_for_x_and_y_l63_63164


namespace coefficient_x2_expansion_l63_63536

theorem coefficient_x2_expansion : 
  let binomial_coeff (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  let expansion_coeff (a b : ℤ) (n k : ℕ) : ℤ := (b ^ k) * (binomial_coeff n k) * (a ^ (n - k))
  (expansion_coeff 1 (-2) 4 2) = 24 :=
by
  let binomial_coeff (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  let expansion_coeff (a b : ℤ) (n k : ℕ) : ℤ := (b ^ k) * (binomial_coeff n k) * (a ^ (n - k))
  have coeff : ℤ := expansion_coeff 1 (-2) 4 2
  sorry -- Proof goes here

end coefficient_x2_expansion_l63_63536


namespace find_p_l63_63446

theorem find_p (n : ℝ) (p : ℝ) (h1 : p = 4 * n * (1 / (2 ^ 2009)) ^ Real.log 1) (h2 : n = 9 / 4) : p = 9 :=
by
  sorry

end find_p_l63_63446


namespace matinee_receipts_l63_63308

theorem matinee_receipts :
  let child_ticket_cost := 4.50
  let adult_ticket_cost := 6.75
  let num_children := 48
  let num_adults := num_children - 20
  total_receipts = num_children * child_ticket_cost + num_adults * adult_ticket_cost :=
by 
  sorry

end matinee_receipts_l63_63308


namespace num_pairs_sold_l63_63950

theorem num_pairs_sold : 
  let total_amount : ℤ := 735
  let avg_price : ℝ := 9.8
  let num_pairs : ℝ := total_amount / avg_price
  num_pairs = 75 :=
by
  let total_amount : ℤ := 735
  let avg_price : ℝ := 9.8
  let num_pairs : ℝ := total_amount / avg_price
  exact sorry

end num_pairs_sold_l63_63950


namespace tan_alpha_value_l63_63303

variables (α β : ℝ)

theorem tan_alpha_value
  (h1 : Real.tan (3 * α - 2 * β) = 1 / 2)
  (h2 : Real.tan (5 * α - 4 * β) = 1 / 4) :
  Real.tan α = 13 / 16 :=
sorry

end tan_alpha_value_l63_63303


namespace remainder_check_l63_63555

theorem remainder_check (q : ℕ) (n : ℕ) (h1 : q = 3^19) (h2 : n = 1162261460) : q % n = 7 := by
  rw [h1, h2]
  -- Proof skipped
  sorry

end remainder_check_l63_63555


namespace factor_expression_l63_63265

theorem factor_expression:
  ∀ (x : ℝ), (10 * x^3 + 50 * x^2 - 4) - (3 * x^3 - 5 * x^2 + 2) = 7 * x^3 + 55 * x^2 - 6 :=
by
  sorry

end factor_expression_l63_63265


namespace similar_triangle_shortest_side_l63_63946

theorem similar_triangle_shortest_side (a b c: ℝ) (d e f: ℝ) :
  a = 21 ∧ b = 20 ∧ c = 29 ∧ d = 87 ∧ c^2 = a^2 + b^2 ∧ d / c = 3 → e = 60 :=
by
  sorry

end similar_triangle_shortest_side_l63_63946


namespace triangle_max_distance_product_l63_63969

open Real

noncomputable def max_product_of_distances
  (a b c : ℝ) (P : {p : ℝ × ℝ // True}) : ℝ :=
  let h_a := 1 -- placeholder for actual distance calculation
  let h_b := 1 -- placeholder for actual distance calculation
  let h_c := 1 -- placeholder for actual distance calculation
  h_a * h_b * h_c

theorem triangle_max_distance_product
  (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5)
  (P : {p : ℝ × ℝ // True}) :
  max_product_of_distances a b c P = (16/15 : ℝ) :=
sorry

end triangle_max_distance_product_l63_63969


namespace b_share_in_profit_l63_63948

theorem b_share_in_profit (A B C : ℝ) (p : ℝ := 4400) (x : ℝ)
  (h1 : A = 3 * B)
  (h2 : B = (2 / 3) * C)
  (h3 : C = x) :
  B / (A + B + C) * p = 800 :=
by
  sorry

end b_share_in_profit_l63_63948


namespace net_profit_start_year_better_investment_option_l63_63906

-- Question 1: From which year does the developer start to make a net profit?
def investment_cost : ℕ := 81 -- in 10,000 yuan
def first_year_renovation_cost : ℕ := 1 -- in 10,000 yuan
def renovation_cost_increase : ℕ := 2 -- in 10,000 yuan per year
def annual_rental_income : ℕ := 30 -- in 10,000 yuan per year

theorem net_profit_start_year : ∃ n : ℕ, n ≥ 4 ∧ ∀ m < 4, ¬ (annual_rental_income * m > investment_cost + m^2) :=
by sorry

-- Question 2: Which option is better: maximizing total profit or average annual profit?
def profit_function (n : ℕ) : ℤ := 30 * n - (81 + n^2)
def average_annual_profit (n : ℕ) : ℤ := (30 * n - (81 + n^2)) / n
def max_total_profit_year : ℕ := 15
def max_total_profit : ℤ := 144 -- in 10,000 yuan
def max_average_profit_year : ℕ := 9
def max_average_profit : ℤ := 12 -- in 10,000 yuan

theorem better_investment_option : (average_annual_profit max_average_profit_year) ≥ (profit_function max_total_profit_year) / max_total_profit_year :=
by sorry

end net_profit_start_year_better_investment_option_l63_63906


namespace find_value_of_n_l63_63861

theorem find_value_of_n (n : ℤ) : 
    n + (n + 1) + (n + 2) + (n + 3) = 22 → n = 4 :=
by 
  intro h
  sorry

end find_value_of_n_l63_63861


namespace total_amount_740_l63_63731

theorem total_amount_740 (x y z : ℝ) (hz : z = 200) (hy : y = 1.20 * z) (hx : x = 1.25 * y) : 
  x + y + z = 740 := by
  sorry

end total_amount_740_l63_63731


namespace painting_area_l63_63537

theorem painting_area
  (wall_height : ℝ) (wall_length : ℝ)
  (window_height : ℝ) (window_length : ℝ)
  (door_height : ℝ) (door_length : ℝ)
  (cond1 : wall_height = 10) (cond2 : wall_length = 15)
  (cond3 : window_height = 3) (cond4 : window_length = 5)
  (cond5 : door_height = 2) (cond6 : door_length = 7) :
  wall_height * wall_length - window_height * window_length - door_height * door_length = 121 := 
by
  simp [cond1, cond2, cond3, cond4, cond5, cond6]
  sorry

end painting_area_l63_63537


namespace gina_initial_money_l63_63561

variable (M : ℝ)
variable (kept : ℝ := 170)

theorem gina_initial_money (h1 : M * 1 / 4 + M * 1 / 8 + M * 1 / 5 + kept = M) : 
  M = 400 :=
by
  sorry

end gina_initial_money_l63_63561


namespace periodic_sequence_condition_l63_63988

theorem periodic_sequence_condition (m : ℕ) (a : ℕ) 
  (h_pos : 0 < m)
  (a_seq : ℕ → ℕ) (h_initial : a_seq 0 = a)
  (h_relation : ∀ n, a_seq (n + 1) = if a_seq n % 2 = 0 then a_seq n / 2 else a_seq n + m) :
  (∃ p, ∀ k, a_seq (k + p) = a_seq k) ↔ 
  (a ∈ ({n | 1 ≤ n ∧ n ≤ m} ∪ {n | ∃ k, n = m + 2 * k + 1 ∧ n < 2 * m + 1})) :=
sorry

end periodic_sequence_condition_l63_63988


namespace smallest_x_l63_63994

theorem smallest_x (x : ℝ) (h : 4 * x^2 + 6 * x + 1 = 5) : x = -2 :=
sorry

end smallest_x_l63_63994


namespace no_solution_outside_intervals_l63_63395

theorem no_solution_outside_intervals (x a : ℝ) :
  (a < 0 ∨ a > 10) → 3 * |x + 3 * a| + |x + a^2| + 2 * x ≠ a :=
by {
  sorry
}

end no_solution_outside_intervals_l63_63395


namespace find_misread_solution_l63_63081

theorem find_misread_solution:
  ∃ a b : ℝ, 
  a = 5 ∧ b = 2 ∧ 
    (a^2 - 2 * a * b + b^2 = 9) ∧ 
    (∀ x y : ℝ, (5 * x + 4 * y = 23) ∧ (3 * x - 2 * y = 5) → (x = 3) ∧ (y = 2)) := by
    sorry

end find_misread_solution_l63_63081


namespace solve_system_of_inequalities_l63_63927

theorem solve_system_of_inequalities (x : ℝ) :
  4*x^2 - 27*x + 18 > 0 ∧ x^2 + 4*x + 4 > 0 ↔ (x < 3/4 ∨ x > 6) ∧ x ≠ -2 :=
by
  sorry

end solve_system_of_inequalities_l63_63927


namespace larry_wins_probability_l63_63530

noncomputable def probability (n : ℕ) : ℝ :=
  if n % 2 = 1 then (1/2)^(n) else 0

noncomputable def inf_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

theorem larry_wins_probability :
  inf_geometric_sum (1/2) (1/4) = 2/3 :=
by
  sorry

end larry_wins_probability_l63_63530


namespace Eunji_total_wrong_questions_l63_63334

theorem Eunji_total_wrong_questions 
  (solved_A : ℕ) (solved_B : ℕ) (wrong_A : ℕ) (right_diff : ℕ) 
  (h1 : solved_A = 12) 
  (h2 : solved_B = 15) 
  (h3 : wrong_A = 4) 
  (h4 : right_diff = 2) :
  (solved_A - (solved_A - (solved_A - wrong_A) + right_diff) + (solved_A - wrong_A) + right_diff - solved_B - (solved_B - (solved_A - (solved_A - wrong_A) + right_diff))) = 9 :=
by {
  sorry
}

end Eunji_total_wrong_questions_l63_63334


namespace hoseok_has_least_papers_l63_63252

-- Definitions based on the conditions
def pieces_jungkook : ℕ := 10
def pieces_hoseok : ℕ := 7
def pieces_seokjin : ℕ := pieces_jungkook - 2

-- Theorem stating Hoseok has the least pieces of colored paper
theorem hoseok_has_least_papers : pieces_hoseok < pieces_jungkook ∧ pieces_hoseok < pieces_seokjin := by 
  sorry

end hoseok_has_least_papers_l63_63252


namespace find_f_at_7_l63_63473

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^7 + b * x^3 + c * x - 5

theorem find_f_at_7 (a b c : ℝ) (h : f a b c (-7) = 7) : f a b c 7 = -17 :=
by
  sorry

end find_f_at_7_l63_63473


namespace george_coin_distribution_l63_63938

theorem george_coin_distribution (a b c : ℕ) (h₁ : a = 1050) (h₂ : b = 1260) (h₃ : c = 210) :
  Nat.gcd (Nat.gcd a b) c = 210 :=
by
  sorry

end george_coin_distribution_l63_63938


namespace longest_side_of_enclosure_l63_63189

theorem longest_side_of_enclosure
  (l w : ℝ)
  (h1 : 2 * l + 2 * w = 180)
  (h2 : l * w = 1440) :
  l = 72 ∨ w = 72 :=
by {
  sorry
}

end longest_side_of_enclosure_l63_63189


namespace greatest_ratio_AB_CD_on_circle_l63_63791

/-- The statement proving the greatest possible value of the ratio AB/CD for points A, B, C, D lying on the 
circle x^2 + y^2 = 16 with integer coordinates and unequal distances AB and CD is sqrt 10 / 3. -/
theorem greatest_ratio_AB_CD_on_circle :
  ∀ (A B C D : ℤ × ℤ), A ≠ B → C ≠ D → 
  A.1^2 + A.2^2 = 16 → B.1^2 + B.2^2 = 16 → 
  C.1^2 + C.2^2 = 16 → D.1^2 + D.2^2 = 16 → 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let CD := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let ratio := AB / CD
  AB ≠ CD →
  ratio ≤ Real.sqrt 10 / 3 :=
sorry

end greatest_ratio_AB_CD_on_circle_l63_63791


namespace odd_pair_exists_k_l63_63766

theorem odd_pair_exists_k (a b : ℕ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) : 
  ∃ k : ℕ, (2^2018 ∣ b^k - a^2) ∨ (2^2018 ∣ a^k - b^2) := 
sorry

end odd_pair_exists_k_l63_63766


namespace train_length_l63_63580

/-- Given that the jogger runs at 2.5 m/s,
    the train runs at 12.5 m/s, 
    the jogger is initially 260 meters ahead, 
    and the train takes 38 seconds to pass the jogger,
    prove that the length of the train is 120 meters. -/
theorem train_length (speed_jogger speed_train : ℝ) (initial_distance time_passing : ℝ)
  (hjogger : speed_jogger = 2.5) (htrain : speed_train = 12.5)
  (hinitial : initial_distance = 260) (htime : time_passing = 38) :
  ∃ L : ℝ, L = 120 :=
by
  sorry

end train_length_l63_63580


namespace problem_statement_l63_63440

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^2021 + a^2022 = 2 := 
by
  sorry

end problem_statement_l63_63440


namespace total_surface_area_of_cuboid_l63_63177

variables (l w h : ℝ)
variables (lw_area wh_area lh_area : ℝ)

def box_conditions :=
  lw_area = l * w ∧
  wh_area = w * h ∧
  lh_area = l * h

theorem total_surface_area_of_cuboid (hc : box_conditions l w h 120 72 60) :
  2 * (120 + 72 + 60) = 504 :=
sorry

end total_surface_area_of_cuboid_l63_63177


namespace prove_ellipse_and_sum_constant_l63_63420

-- Define the ellipse properties
def ellipse_center_origin (a b : ℝ) : Prop :=
  a = 4 ∧ b^2 = 12

-- Standard equation of the ellipse
def ellipse_standard_eqn (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 12) = 1

-- Define the conditions for m and n given point M(1, 3)
def condition_m_n (m n : ℝ) (x0 : ℝ) : Prop :=
  (9 * m^2 + 96 * m + 48 - (13/4) * x0^2 = 0) ∧ (9 * n^2 + 96 * n + 48 - (13/4) * x0^2 = 0)

-- Prove the standard equation of the ellipse and m+n constant properties
theorem prove_ellipse_and_sum_constant (a b x y m n x0 : ℝ) 
  (h1 : ellipse_center_origin a b)
  (h2 : ellipse_standard_eqn x y)
  (h3 : condition_m_n m n x0) :
  m + n = -32/3 := 
sorry

end prove_ellipse_and_sum_constant_l63_63420


namespace Berry_read_pages_thursday_l63_63695

theorem Berry_read_pages_thursday :
  ∀ (pages_per_day : ℕ) (pages_sunday : ℕ) (pages_monday : ℕ) (pages_tuesday : ℕ) 
    (pages_wednesday : ℕ) (pages_friday : ℕ) (pages_saturday : ℕ),
    (pages_per_day = 50) →
    (pages_sunday = 43) →
    (pages_monday = 65) →
    (pages_tuesday = 28) →
    (pages_wednesday = 0) →
    (pages_friday = 56) →
    (pages_saturday = 88) →
    pages_sunday + pages_monday + pages_tuesday +
    pages_wednesday + pages_friday + pages_saturday + x = 350 →
    x = 70 := by
  sorry

end Berry_read_pages_thursday_l63_63695


namespace log_product_l63_63340

theorem log_product :
  (Real.log 100 / Real.log 10) * (Real.log (1 / 10) / Real.log 10) = -2 := by
  sorry

end log_product_l63_63340


namespace quad_sin_theorem_l63_63190

-- Define the necessary entities in Lean
structure Quadrilateral (A B C D : Type) :=
(angleB : ℝ)
(angleD : ℝ)
(angleA : ℝ)

-- Define the main theorem
theorem quad_sin_theorem {A B C D : Type} (quad : Quadrilateral A B C D) (AC AD : ℝ) (α : ℝ) :
  quad.angleB = 90 ∧ quad.angleD = 90 ∧ quad.angleA = α → AD = AC * Real.sin α := 
sorry

end quad_sin_theorem_l63_63190


namespace number_of_buses_in_month_l63_63792

-- Given conditions
def weekday_buses := 36
def saturday_buses := 24
def sunday_holiday_buses := 12
def num_weekdays := 18
def num_saturdays := 4
def num_sundays_holidays := 6

-- Statement to prove
theorem number_of_buses_in_month : 
  num_weekdays * weekday_buses + num_saturdays * saturday_buses + num_sundays_holidays * sunday_holiday_buses = 816 := 
by 
  sorry

end number_of_buses_in_month_l63_63792


namespace jenny_sold_192_packs_l63_63212

-- Define the conditions
def boxes_sold : ℝ := 24.0
def packs_per_box : ℝ := 8.0

-- The total number of packs sold
def total_packs_sold : ℝ := boxes_sold * packs_per_box

-- Proof statement that total packs sold equals 192.0
theorem jenny_sold_192_packs : total_packs_sold = 192.0 :=
by
  sorry

end jenny_sold_192_packs_l63_63212


namespace range_of_a_l63_63508

theorem range_of_a (a : ℝ) :
  (∀ x, (x - 2)/5 + 2 ≤ x - 4/5 ∨ x ≤ a) → a ≥ 3 :=
by
  sorry

end range_of_a_l63_63508


namespace common_difference_of_arithmetic_sequence_l63_63608

variable (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) (d : ℝ)
variable (h₁ : S_n 5 = -15) (h₂ : a_n 2 + a_n 5 = -2)

theorem common_difference_of_arithmetic_sequence :
  d = 4 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l63_63608


namespace quarters_in_school_year_l63_63819

variable (students : ℕ) (artworks_per_student_per_quarter : ℕ) (total_artworks : ℕ) (school_years : ℕ)

def number_of_quarters (students : ℕ) (artworks_per_student_per_quarter : ℕ) (total_artworks : ℕ) (school_years : ℕ) : ℕ :=
  (total_artworks / (students * artworks_per_student_per_quarter * school_years))

theorem quarters_in_school_year :
  number_of_quarters 15 2 240 2 = 4 :=
by sorry

end quarters_in_school_year_l63_63819


namespace find_x_l63_63836

theorem find_x (n x q p : ℕ) (h1 : n = q * x + 2) (h2 : 2 * n = p * x + 4) : x = 6 :=
sorry

end find_x_l63_63836


namespace combined_area_correct_l63_63389

-- Define the given dimensions and border width
def length : ℝ := 0.6
def width : ℝ := 0.35
def border_width : ℝ := 0.05

-- Define the area of the rectangle, the new dimensions with the border, 
-- and the combined area of the rectangle and the border
def rectangle_area : ℝ := length * width
def new_length : ℝ := length + 2 * border_width
def new_width : ℝ := width + 2 * border_width
def combined_area : ℝ := new_length * new_width

-- The statement we want to prove
theorem combined_area_correct : combined_area = 0.315 := by
  sorry

end combined_area_correct_l63_63389


namespace maria_baggies_l63_63764

-- Definitions of the conditions
def total_cookies (chocolate_chip : Nat) (oatmeal : Nat) : Nat :=
  chocolate_chip + oatmeal

def cookies_per_baggie : Nat :=
  3

def number_of_baggies (total_cookies : Nat) (cookies_per_baggie : Nat) : Nat :=
  total_cookies / cookies_per_baggie

-- Proof statement
theorem maria_baggies :
  number_of_baggies (total_cookies 2 16) cookies_per_baggie = 6 := 
sorry

end maria_baggies_l63_63764


namespace correctness_of_solution_set_l63_63641

-- Define the set of real numbers satisfying the inequality
def solution_set : Set ℝ := { x | 3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9 }

-- Define the expected solution set derived from the problem
def expected_solution_set : Set ℝ := { x | -1 < x ∧ x ≤ 1 } ∪ { x | 2.5 < x ∧ x < 4.5 }

-- The proof statement
theorem correctness_of_solution_set : solution_set = expected_solution_set :=
  sorry

end correctness_of_solution_set_l63_63641


namespace quadratic_root_other_l63_63959

theorem quadratic_root_other (a : ℝ) (h : (3 : ℝ)*3 - 2*3 + a = 0) : 
  ∃ (b : ℝ), b = -1 ∧ (b : ℝ)*b - 2*b + a = 0 :=
by
  sorry

end quadratic_root_other_l63_63959


namespace impossible_sequence_l63_63612

def letters_order : List ℕ := [1, 2, 3, 4, 5]

def is_typing_sequence (order : List ℕ) (seq : List ℕ) : Prop :=
  sorry -- This function will evaluate if a sequence is possible given the order

theorem impossible_sequence : ¬ is_typing_sequence letters_order [4, 5, 2, 3, 1] :=
  sorry

end impossible_sequence_l63_63612


namespace value_of_b_l63_63298

theorem value_of_b (x y b : ℝ) (h1: 7^(3 * x - 1) * b^(4 * y - 3) = 49^x * 27^y) (h2: x + y = 4) : b = 3 :=
by
  sorry

end value_of_b_l63_63298


namespace smallest_number_gt_sum_digits_1755_l63_63465

theorem smallest_number_gt_sum_digits_1755 :
  ∃ (n : ℕ) (a b c d : ℕ), a ≠ 0 ∧ n = 1000 * a + 100 * b + 10 * c + d ∧ n = (a + b + c + d) + 1755 ∧ n = 1770 :=
by {
  sorry
}

end smallest_number_gt_sum_digits_1755_l63_63465


namespace div_by_prime_power_l63_63898

theorem div_by_prime_power (p α x : ℕ) (hp : Nat.Prime p) (hpg : p > 2) (hα : α > 0) (t : ℤ) :
  (∃ k : ℤ, x^2 - 1 = k * p^α) ↔ (∃ t : ℤ, x = t * p^α + 1 ∨ x = t * p^α - 1) :=
sorry

end div_by_prime_power_l63_63898


namespace max_value_of_function_l63_63567

theorem max_value_of_function (x : ℝ) (h : x < 5 / 4) :
    (∀ y, y = 4 * x - 2 + 1 / (4 * x - 5) → y ≤ 1):=
sorry

end max_value_of_function_l63_63567


namespace gain_percentage_is_30_l63_63741

def sellingPrice : ℕ := 195
def gain : ℕ := 45
def costPrice : ℕ := sellingPrice - gain

def gainPercentage : ℚ := (gain : ℚ) / (costPrice : ℚ) * 100

theorem gain_percentage_is_30 :
  gainPercentage = 30 := 
sorry

end gain_percentage_is_30_l63_63741


namespace solve_inequality_l63_63968

noncomputable def solution_set (x : ℝ) : Prop :=
  (-(9/2) ≤ x ∧ x ≤ -2) ∨ ((1 - Real.sqrt 5) / 2 < x ∧ x < (1 + Real.sqrt 5) / 2)

theorem solve_inequality (x : ℝ) :
  (x ≠ -2 ∧ x ≠ 9/2) →
  ( (x + 1) / (x + 2) > (3 * x + 4) / (2 * x + 9) ) ↔ solution_set x :=
sorry

end solve_inequality_l63_63968


namespace star_addition_l63_63485

-- Definition of the binary operation "star"
def star (x y : ℤ) := 5 * x - 2 * y

-- Statement of the problem
theorem star_addition : star 3 4 + star 2 2 = 13 :=
by
  -- By calculation, we have:
  -- star 3 4 = 7 and star 2 2 = 6
  -- Thus, star 3 4 + star 2 2 = 7 + 6 = 13
  sorry

end star_addition_l63_63485


namespace total_volume_of_removed_pyramids_l63_63214

noncomputable def volume_of_removed_pyramids (edge_length : ℝ) : ℝ :=
  8 * (1 / 3 * (1 / 2 * (edge_length / 4) * (edge_length / 4)) * (edge_length / 4) / 6)

theorem total_volume_of_removed_pyramids :
  volume_of_removed_pyramids 1 = 1 / 48 :=
by
  sorry

end total_volume_of_removed_pyramids_l63_63214


namespace melanie_missed_games_l63_63050

-- Define the total number of soccer games played and the number attended by Melanie
def total_games : ℕ := 64
def attended_games : ℕ := 32

-- Statement to be proven
theorem melanie_missed_games : total_games - attended_games = 32 := by
  -- Placeholder for the proof
  sorry

end melanie_missed_games_l63_63050


namespace range_of_m_l63_63401

-- Defining the conditions p and q
def p (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) < 0
def q (x : ℝ) : Prop := 1/2 < x ∧ x < 2/3

-- Defining the main theorem
theorem range_of_m (m : ℝ) : (∀ x : ℝ, q x → p x m) ∧ ¬ (∀ x : ℝ, p x m → q x) ↔ (-1/3 ≤ m ∧ m ≤ 3/2) :=
sorry

end range_of_m_l63_63401


namespace trajectory_of_P_l63_63380

-- Define points F1 and F2
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Define the condition |PF2| - |PF1| = 4 for a moving point P
def condition (P : ℝ × ℝ) : Prop :=
  let PF1 := Real.sqrt ((P.1 + 4)^2 + P.2^2)
  let PF2 := Real.sqrt ((P.1 - 4)^2 + P.2^2)
  abs (PF2 - PF1) = 4

-- The target equation of the trajectory
def target_eq (P : ℝ × ℝ) : Prop :=
  P.1 * P.1 / 4 - P.2 * P.2 / 12 = 1 ∧ P.1 ≤ -2

theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, condition P → target_eq P := by
  sorry

end trajectory_of_P_l63_63380


namespace correct_operation_l63_63412

variable (a b : ℝ)

theorem correct_operation : (-2 * a ^ 2) ^ 2 = 4 * a ^ 4 := by
  sorry

end correct_operation_l63_63412


namespace number_of_cow_herds_l63_63568

theorem number_of_cow_herds 
    (total_cows : ℕ) 
    (cows_per_herd : ℕ) 
    (h1 : total_cows = 320)
    (h2 : cows_per_herd = 40) : 
    total_cows / cows_per_herd = 8 :=
by
  sorry

end number_of_cow_herds_l63_63568


namespace range_of_a_l63_63154

theorem range_of_a 
  (a : ℝ):
  (∀ x : ℝ, |x + 2| + |x - 1| > a^2 - 2 * a) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l63_63154


namespace find_angle_A_triangle_is_right_l63_63657

theorem find_angle_A (A : ℝ) (h : 2 * Real.cos (Real.pi + A) + Real.sin (Real.pi / 2 + 2 * A) + 3 / 2 = 0) :
  A = Real.pi / 3 := 
sorry

theorem triangle_is_right (a b c : ℝ) (A : ℝ) (ha : c - b = (Real.sqrt 3) / 3 * a) (hA : A = Real.pi / 3) :
  c^2 = a^2 + b^2 :=
sorry

end find_angle_A_triangle_is_right_l63_63657


namespace original_expression_equals_l63_63794

noncomputable def evaluate_expression (a : ℝ) : ℝ :=
  ( (a / (a + 2) + 1 / (a^2 - 4)) / ( (a - 1) / (a + 2) + 1 / (a - 2) ))

theorem original_expression_equals (a : ℝ) (h : a = 2 + Real.sqrt 2) :
  evaluate_expression a = (Real.sqrt 2 + 1) :=
sorry

end original_expression_equals_l63_63794


namespace minimize_wage_l63_63989

def totalWorkers : ℕ := 150
def wageA : ℕ := 2000
def wageB : ℕ := 3000

theorem minimize_wage : ∃ (a : ℕ), a = 50 ∧ (totalWorkers - a) ≥ 2 * a ∧ 
  (wageA * a + wageB * (totalWorkers - a) = 400000) := sorry

end minimize_wage_l63_63989


namespace solve_for_f_1988_l63_63374

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom functional_eq (m n : ℕ+) : f (f m + f n) = m + n

theorem solve_for_f_1988 : f 1988 = 1988 :=
sorry

end solve_for_f_1988_l63_63374


namespace jimin_has_most_candy_left_l63_63865

-- Definitions based on conditions
def fraction_jimin_ate := 1 / 9
def fraction_taehyung_ate := 1 / 3
def fraction_hoseok_ate := 1 / 6

-- The goal to prove
theorem jimin_has_most_candy_left : 
  (1 - fraction_jimin_ate) > (1 - fraction_taehyung_ate) ∧ (1 - fraction_jimin_ate) > (1 - fraction_hoseok_ate) :=
by
  -- The actual proof steps are omitted here.
  sorry

end jimin_has_most_candy_left_l63_63865


namespace meal_cost_l63_63375

theorem meal_cost :
  ∃ (s c p : ℝ),
  (5 * s + 8 * c + 2 * p = 5.40) ∧
  (3 * s + 11 * c + 2 * p = 4.95) ∧
  (s + c + p = 1.55) :=
sorry

end meal_cost_l63_63375


namespace total_questions_attempted_l63_63635

theorem total_questions_attempted 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) (total_marks : ℕ) (correct_answers : ℕ) 
  (total_questions : ℕ) (incorrect_answers : ℕ)
  (h_marks_per_correct : marks_per_correct = 4)
  (h_marks_lost_per_wrong : marks_lost_per_wrong = 1) 
  (h_total_marks : total_marks = 130) 
  (h_correct_answers : correct_answers = 36) 
  (h_score_eq : marks_per_correct * correct_answers - marks_lost_per_wrong * incorrect_answers = total_marks)
  (h_total_questions : total_questions = correct_answers + incorrect_answers) : 
  total_questions = 50 :=
by
  sorry

end total_questions_attempted_l63_63635


namespace find_x_l63_63575

theorem find_x {x : ℝ} :
  (10 + 30 + 50) / 3 = ((20 + 40 + x) / 3) + 8 → x = 6 :=
by
  intro h
  -- Solution steps would go here, but they are omitted.
  sorry

end find_x_l63_63575


namespace dance_team_members_l63_63458

theorem dance_team_members (a b c : ℕ)
  (h1 : a + b + c = 100)
  (h2 : b = 2 * a)
  (h3 : c = 2 * a + 10) :
  c = 46 := by
  sorry

end dance_team_members_l63_63458


namespace tangent_lines_parallel_to_4x_minus_1_l63_63559

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_lines_parallel_to_4x_minus_1 :
  ∃ (a b : ℝ), (f a = b ∧ 3 * a^2 + 1 = 4) → (b = 4 * a - 4 ∨ b = 4 * a) :=
by
  sorry

end tangent_lines_parallel_to_4x_minus_1_l63_63559


namespace simplify_fraction_l63_63294

variable {a b m : ℝ}

theorem simplify_fraction (h : a + b ≠ 0) : (ma/a + b) + (mb/a + b) = m :=
by
  sorry

end simplify_fraction_l63_63294


namespace rory_more_jellybeans_l63_63256

-- Definitions based on the conditions
def G : ℕ := 15 -- Gigi has 15 jellybeans
def LorelaiConsumed (R G : ℕ) : ℕ := 3 * (R + G) -- Lorelai has already eaten three times the total number of jellybeans

theorem rory_more_jellybeans {R : ℕ} (h1 : LorelaiConsumed R G = 180) : (R - G) = 30 :=
  by
    -- we can skip the proof here with sorry, as we are only interested in the statement for now
    sorry

end rory_more_jellybeans_l63_63256


namespace two_person_subcommittees_from_six_l63_63065

theorem two_person_subcommittees_from_six :
  (Nat.choose 6 2) = 15 := by
  sorry

end two_person_subcommittees_from_six_l63_63065


namespace find_P_l63_63804

theorem find_P (P : ℕ) (h : 4 * (P + 4 + 8 + 20) = 252) : P = 31 :=
by
  -- Assume this proof is nontrivial and required steps
  sorry

end find_P_l63_63804


namespace find_value_of_expression_l63_63326

variable (p q r s : ℝ)

def g (x : ℝ) : ℝ := p * x ^ 3 + q * x ^ 2 + r * x + s

-- We state the condition that g(1) = 1
axiom g_at_one : g p q r s 1 = 1

-- Now, we state the problem we need to prove:
theorem find_value_of_expression : 5 * p - 3 * q + 2 * r - s = 5 :=
by
  -- We skip the proof here
  exact sorry

end find_value_of_expression_l63_63326


namespace sample_size_l63_63609

theorem sample_size (f_c f_o N: ℕ) (h1: f_c = 8) (h2: f_c = 1 / 4 * f_o) (h3: f_c + f_o = N) : N = 40 :=
  sorry

end sample_size_l63_63609


namespace decreasing_function_l63_63738

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem decreasing_function (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₂ ≤ 1) : 
  f x₁ > f x₂ :=
by
  -- Proof goes here
  sorry

end decreasing_function_l63_63738


namespace arithmetic_sequence_problem_l63_63621

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 4 + a 6 + a 8 + a 10 + a 12 = 60)
  (h2 : ∀ n, a (n + 1) = a n + d) :
  a 7 - (1 / 3) * a 5 = 8 :=
by
  sorry

end arithmetic_sequence_problem_l63_63621


namespace cannot_finish_third_l63_63480

-- Definitions for the orders of runners
def order (a b : String) : Prop := a < b

-- The problem statement and conditions
def conditions (P Q R S T U : String) : Prop :=
  order P Q ∧ order P R ∧ order Q S ∧ order P U ∧ order U T ∧ order T Q

theorem cannot_finish_third (P Q R S T U : String) (h : conditions P Q R S T U) :
  (P = "third" → False) ∧ (S = "third" → False) :=
by
  sorry

end cannot_finish_third_l63_63480


namespace growth_operation_two_operations_growth_operation_four_operations_l63_63160

noncomputable def growth_operation_perimeter (initial_side_length : ℕ) (growth_operations : ℕ) := 
  initial_side_length * 3 * (4/3 : ℚ)^(growth_operations + 1)

theorem growth_operation_two_operations :
  growth_operation_perimeter 9 2 = 48 := by sorry

theorem growth_operation_four_operations :
  growth_operation_perimeter 9 4 = 256 / 3 := by sorry

end growth_operation_two_operations_growth_operation_four_operations_l63_63160


namespace derivative_at_0_l63_63868

-- Define the function
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 2

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := deriv f x

-- State the theorem
theorem derivative_at_0 : f' 0 = 4 :=
by {
  -- Inserting sorry to skip the proof
  sorry
}

end derivative_at_0_l63_63868


namespace m_gt_n_l63_63917

variable (m n : ℝ)

-- Definition of points A and B lying on the line y = -2x + 1
def point_A_on_line : Prop := m = -2 * (-1) + 1
def point_B_on_line : Prop := n = -2 * 3 + 1

-- Theorem stating that m > n given the conditions
theorem m_gt_n (hA : point_A_on_line m) (hB : point_B_on_line n) : m > n :=
by
  -- To avoid the proof part, which we skip as per instructions
  sorry

end m_gt_n_l63_63917


namespace combined_length_of_all_CDs_l63_63556

-- Define the lengths of each CD based on the conditions
def length_cd1 := 1.5
def length_cd2 := 1.5
def length_cd3 := 2 * length_cd1
def length_cd4 := length_cd2 / 2
def length_cd5 := length_cd1 + length_cd2

-- Define the combined length of all CDs
def combined_length := length_cd1 + length_cd2 + length_cd3 + length_cd4 + length_cd5

-- State the theorem
theorem combined_length_of_all_CDs : combined_length = 9.75 := by
  sorry

end combined_length_of_all_CDs_l63_63556


namespace value_of_a_l63_63661

theorem value_of_a {a : ℝ} 
  (h : ∀ x y : ℝ, ax - 2*y + 2 = 0 ↔ x + (a-3)*y + 1 = 0) : 
  a = 1 := 
by 
  sorry

end value_of_a_l63_63661


namespace arccos_cos_of_11_l63_63478

-- Define the initial conditions
def angle_in_radians (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 2 * Real.pi

def arccos_principal_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ Real.pi

-- Define the main theorem to be proved
theorem arccos_cos_of_11 :
  angle_in_radians 11 →
  arccos_principal_range (Real.arccos (Real.cos 11)) →
  Real.arccos (Real.cos 11) = 4.71682 :=
by
  -- Proof is not required
  sorry

end arccos_cos_of_11_l63_63478


namespace x_coordinate_incenter_eq_l63_63844

theorem x_coordinate_incenter_eq {x y : ℝ} :
  (y = 0 → x + y = 3 → x = 0) → 
  (y = x → y = -x + 3 → x = 3 / 2) :=
by
  sorry

end x_coordinate_incenter_eq_l63_63844


namespace accurate_to_ten_thousandth_l63_63155

/-- Define the original number --/
def original_number : ℕ := 580000

/-- Define the accuracy of the number represented by 5.8 * 10^5 --/
def is_accurate_to_ten_thousandth_place (n : ℕ) : Prop :=
  n = 5 * 100000 + 8 * 10000

/-- The statement to be proven --/
theorem accurate_to_ten_thousandth : is_accurate_to_ten_thousandth_place original_number :=
by
  sorry

end accurate_to_ten_thousandth_l63_63155


namespace girls_with_rulers_l63_63888

theorem girls_with_rulers 
  (total_students : ℕ) (students_with_rulers : ℕ) (boys_with_set_squares : ℕ) 
  (total_girls : ℕ) (student_count : total_students = 50) 
  (ruler_count : students_with_rulers = 28) 
  (boys_with_set_squares_count : boys_with_set_squares = 14) 
  (girl_count : total_girls = 31) 
  : total_girls - (total_students - students_with_rulers - boys_with_set_squares) = 23 := 
by
  sorry

end girls_with_rulers_l63_63888


namespace cost_apples_l63_63685

def total_cost := 42
def cost_bananas := 12
def cost_bread := 9
def cost_milk := 7

theorem cost_apples:
  total_cost - (cost_bananas + cost_bread + cost_milk) = 14 :=
by
  sorry

end cost_apples_l63_63685


namespace problem_proof_l63_63664

theorem problem_proof (a b c d m n : ℕ) (h1 : a^2 + b^2 + c^2 + d^2 = 1989) 
  (h2 : a + b + c + d = m^2) 
  (h3 : max (max a b) (max c d) = n^2) : 
  m = 9 ∧ n = 6 :=
by
  sorry

end problem_proof_l63_63664


namespace walter_zoo_time_l63_63808

def seals_time : ℕ := 13
def penguins_time : ℕ := 8 * seals_time
def elephants_time : ℕ := 13
def total_time_spent_at_zoo : ℕ := seals_time + penguins_time + elephants_time

theorem walter_zoo_time : total_time_spent_at_zoo = 130 := by
  -- Proof goes here
  sorry

end walter_zoo_time_l63_63808


namespace find_max_marks_l63_63790

variable (marks_scored : ℕ) -- 212
variable (shortfall : ℕ) -- 22
variable (pass_percentage : ℝ) -- 0.30

theorem find_max_marks (h_marks : marks_scored = 212) 
                       (h_short : shortfall = 22) 
                       (h_pass : pass_percentage = 0.30) : 
  ∃ M : ℝ, M = 780 :=
by {
  sorry
}

end find_max_marks_l63_63790


namespace unique_corresponding_point_l63_63043

-- Define the points for the squares
structure Point := (x : ℝ) (y : ℝ)

structure Square :=
  (a b c d : Point)

def contains (sq1 sq2: Square) : Prop :=
  sq2.a.x >= sq1.a.x ∧ sq2.a.y >= sq1.a.y ∧
  sq2.b.x <= sq1.b.x ∧ sq2.b.y >= sq1.b.y ∧
  sq2.c.x <= sq1.c.x ∧ sq2.c.y <= sq1.c.y ∧
  sq2.d.x >= sq1.d.x ∧ sq2.d.y <= sq1.d.y

theorem unique_corresponding_point
  (sq1 sq2 : Square)
  (h1 : contains sq1 sq2)
  (h2 : sq1.a.x - sq1.c.x = sq2.a.x - sq2.c.x ∧ sq1.a.y - sq1.c.y = sq2.a.y - sq2.c.y):
  ∃! (O : Point), ∃ O' : Point, contains sq1 sq2 ∧ 
  (O.x - sq1.a.x) / (sq1.b.x - sq1.a.x) = (O'.x - sq2.a.x) / (sq2.b.x - sq2.a.x) ∧ 
  (O.y - sq1.a.y) / (sq1.d.y - sq1.a.y) = (O'.y - sq2.a.y) / (sq2.d.y - sq2.a.y) := 
sorry

end unique_corresponding_point_l63_63043


namespace count_valid_three_digit_numbers_l63_63783

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, n = 720 ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 → 
    (∀ d ∈ [m / 100, (m / 10) % 10, m % 10], d ∉ [2, 5, 7, 9])) := 
sorry

end count_valid_three_digit_numbers_l63_63783


namespace sum_of_last_two_digits_l63_63934

theorem sum_of_last_two_digits (x y : ℕ) : 
  x = 8 → y = 12 → (x^25 + y^25) % 100 = 0 := 
by
  intros hx hy
  sorry

end sum_of_last_two_digits_l63_63934


namespace train_crossing_time_l63_63714

noncomputable def length_of_train : ℕ := 250
noncomputable def length_of_bridge : ℕ := 350
noncomputable def speed_of_train_kmph : ℕ := 72

noncomputable def speed_of_train_mps : ℕ := (speed_of_train_kmph * 1000) / 3600

noncomputable def total_distance : ℕ := length_of_train + length_of_bridge

theorem train_crossing_time : total_distance / speed_of_train_mps = 30 := by
  sorry

end train_crossing_time_l63_63714


namespace find_range_of_m_l63_63801

def equation1 (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 = 0 → x < 0

def equation2 (m : ℝ) : Prop :=
  ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 → false

theorem find_range_of_m (m : ℝ) (h1 : equation1 m → m > 2) (h2 : equation2 m → 1 < m ∧ m < 3) :
  (equation1 m ∨ equation2 m) ∧ ¬(equation1 m ∧ equation2 m) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  sorry

end find_range_of_m_l63_63801


namespace number_of_diagonals_of_nonagon_l63_63434

theorem number_of_diagonals_of_nonagon:
  (9 * (9 - 3)) / 2 = 27 := by
  sorry

end number_of_diagonals_of_nonagon_l63_63434


namespace probability_shots_result_l63_63321

open ProbabilityTheory

noncomputable def P_A := 3 / 4
noncomputable def P_B := 4 / 5
noncomputable def P_not_A := 1 - P_A
noncomputable def P_not_B := 1 - P_B

theorem probability_shots_result :
    (P_not_A * P_not_B * P_A) + (P_not_A * P_not_B * P_not_A * P_B) = 19 / 400 :=
    sorry

end probability_shots_result_l63_63321


namespace problem_statement_l63_63566

theorem problem_statement (a b c : ℝ) (h1 : a + 2 * b + 3 * c = 12) (h2 : a^2 + b^2 + c^2 = a * b + a * c + b * c) :
  a + b^2 + c^3 = 14 :=
sorry

end problem_statement_l63_63566


namespace liam_comic_books_l63_63647

theorem liam_comic_books (cost_per_book : ℚ) (total_money : ℚ) (n : ℚ) : cost_per_book = 1.25 ∧ total_money = 10 → n = 8 :=
by
  intros h
  cases h
  have h1 : 1.25 * n ≤ 10 := by sorry
  have h2 : n ≤ 10 / 1.25 := by sorry
  have h3 : n ≤ 8 := by sorry
  have h4 : n = 8 := by sorry
  exact h4

end liam_comic_books_l63_63647


namespace complete_the_square_l63_63974

theorem complete_the_square (x : ℝ) (h : x^2 - 8 * x - 1 = 0) : (x - 4)^2 = 17 :=
by
  -- proof steps would go here, but we use sorry for now
  sorry

end complete_the_square_l63_63974


namespace females_with_advanced_degrees_eq_90_l63_63260

-- define the given constants
def total_employees : ℕ := 360
def total_females : ℕ := 220
def total_males : ℕ := 140
def advanced_degrees : ℕ := 140
def college_degrees : ℕ := 160
def vocational_training : ℕ := 60
def males_with_college_only : ℕ := 55
def females_with_vocational_training : ℕ := 25

-- define the main theorem to prove the number of females with advanced degrees
theorem females_with_advanced_degrees_eq_90 :
  ∃ (females_with_advanced_degrees : ℕ), females_with_advanced_degrees = 90 :=
by
  sorry

end females_with_advanced_degrees_eq_90_l63_63260


namespace find_x_coordinate_l63_63195

theorem find_x_coordinate :
  ∃ x : ℝ, (∃ m b : ℝ, (∀ y x : ℝ, y = m * x + b) ∧ 
                     ((3 = m * 10 + b) ∧ 
                      (0 = m * 4 + b)
                     ) ∧ 
                     (-3 = m * x + b) ∧ 
                     (x = -2)) :=
sorry

end find_x_coordinate_l63_63195


namespace center_of_circle_l63_63280

theorem center_of_circle (x y : ℝ) (h : x^2 + y^2 - 10 * x + 4 * y = -40) : 
  x + y = 3 := 
sorry

end center_of_circle_l63_63280


namespace total_bouncy_balls_l63_63909

def red_packs := 4
def yellow_packs := 8
def green_packs := 4
def blue_packs := 6

def red_balls_per_pack := 12
def yellow_balls_per_pack := 10
def green_balls_per_pack := 14
def blue_balls_per_pack := 8

def total_red_balls := red_packs * red_balls_per_pack
def total_yellow_balls := yellow_packs * yellow_balls_per_pack
def total_green_balls := green_packs * green_balls_per_pack
def total_blue_balls := blue_packs * blue_balls_per_pack

def total_balls := total_red_balls + total_yellow_balls + total_green_balls + total_blue_balls

theorem total_bouncy_balls : total_balls = 232 :=
by
  -- calculation proof goes here
  sorry

end total_bouncy_balls_l63_63909


namespace determine_b_l63_63702

noncomputable def Q (x : ℝ) (b : ℝ) : ℝ := x^3 + 3 * x^2 + b * x + 20

theorem determine_b (b : ℝ) :
  (∃ x : ℝ, x = 4 ∧ Q x b = 0) → b = -33 :=
by
  intro h
  rcases h with ⟨_, rfl, hQ⟩
  sorry

end determine_b_l63_63702


namespace possible_values_of_y_l63_63324

theorem possible_values_of_y (x : ℝ) (hx : x^2 + 5 * (x / (x - 3)) ^ 2 = 50) :
  ∃ (y : ℝ), y = (x - 3)^2 * (x + 4) / (3 * x - 4) ∧ (y = 0 ∨ y = 15 ∨ y = 49) :=
sorry

end possible_values_of_y_l63_63324


namespace eval_polynomial_at_4_using_horners_method_l63_63418

noncomputable def polynomial : (x : ℝ) → ℝ :=
  λ x => 3 * x^5 - 2 * x^4 + 5 * x^3 - 2.5 * x^2 + 1.5 * x - 0.7

theorem eval_polynomial_at_4_using_horners_method :
  polynomial 4 = 2845.3 :=
by
  sorry

end eval_polynomial_at_4_using_horners_method_l63_63418


namespace chess_tournament_l63_63456

theorem chess_tournament (m p k n : ℕ) 
  (h1 : m * 9 = p * 6) 
  (h2 : m * n = k * 8) 
  (h3 : p * 2 = k * 6) : 
  n = 4 := 
by 
  sorry

end chess_tournament_l63_63456


namespace tangent_circles_l63_63403

theorem tangent_circles (a b c : ℝ) :
    (∀ x y : ℝ, x^2 + y^2 = a^2 → (x-b)^2 + (y-c)^2 = a^2) →
    ( (b^2 + c^2) / (a^2) = 4 ) :=
by
  intro h
  have h_dist : (b^2 + c^2) = (2 * a) ^ 2 := sorry
  have h_div : (b^2 + c^2) / (a^2) = 4 := sorry
  exact h_div

end tangent_circles_l63_63403


namespace pencils_per_student_l63_63693

theorem pencils_per_student (total_pencils : ℤ) (num_students : ℤ) (pencils_per_student : ℤ)
  (h1 : total_pencils = 195)
  (h2 : num_students = 65) :
  total_pencils / num_students = 3 :=
by
  sorry

end pencils_per_student_l63_63693


namespace length_of_crease_l63_63104

/-- 
  Given a rectangular piece of paper 8 inches wide that is folded such that one corner 
  touches the opposite side at an angle θ from the horizontal, and one edge of the paper 
  remains aligned with the base, 
  prove that the length of the crease L is given by L = 8 * tan θ / (1 + tan θ). 
--/
theorem length_of_crease (theta : ℝ) (h : 0 < theta ∧ theta < Real.pi / 2): 
  ∃ L : ℝ, L = 8 * Real.tan theta / (1 + Real.tan theta) :=
sorry

end length_of_crease_l63_63104


namespace seq_v13_eq_b_l63_63115

noncomputable def seq (v : ℕ → ℝ) (b : ℝ) : Prop :=
v 1 = b ∧ ∀ n ≥ 1, v (n + 1) = -1 / (v n + 2)

theorem seq_v13_eq_b (b : ℝ) (hb : 0 < b) (v : ℕ → ℝ) (hs : seq v b) : v 13 = b := by
  sorry

end seq_v13_eq_b_l63_63115


namespace cookies_per_bag_l63_63656

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (H1 : total_cookies = 703) (H2 : num_bags = 37) : total_cookies / num_bags = 19 := by
  sorry

end cookies_per_bag_l63_63656


namespace find_monthly_salary_l63_63553

variables (x h_1 h_2 h_3 : ℕ)

theorem find_monthly_salary 
    (half_salary_bank : h_1 = x / 2)
    (half_remaining_mortgage : h_2 = (h_1 - 300) / 2)
    (half_remaining_expenses : h_3 = (h_2 + 300) / 2)
    (remaining_salary : h_3 = 800) :
  x = 7600 :=
sorry

end find_monthly_salary_l63_63553


namespace cube_root_110592_l63_63543

theorem cube_root_110592 :
  (∃ x : ℕ, x^3 = 110592) ∧ 
  10^3 = 1000 ∧ 11^3 = 1331 ∧ 12^3 = 1728 ∧ 13^3 = 2197 ∧ 14^3 = 2744 ∧ 
  15^3 = 3375 ∧ 20^3 = 8000 ∧ 21^3 = 9261 ∧ 22^3 = 10648 ∧ 23^3 = 12167 ∧ 
  24^3 = 13824 ∧ 25^3 = 15625 → 48^3 = 110592 :=
by
  sorry

end cube_root_110592_l63_63543


namespace inequality_transformation_l63_63068

theorem inequality_transformation (x : ℝ) (n : ℕ) (hn : 0 < n) (hx : 0 < x) : 
  x + (n^n) / (x^n) ≥ n + 1 := 
sorry

end inequality_transformation_l63_63068


namespace new_number_formed_l63_63936

theorem new_number_formed (h t u : ℕ) (Hh : h < 10) (Ht : t < 10) (Hu : u < 10) :
  let original_number := 100 * h + 10 * t + u
  let new_number := 2000 + 10 * original_number
  new_number = 1000 * (h + 2) + 100 * t + 10 * u :=
by
  -- Proof would go here
  sorry

end new_number_formed_l63_63936


namespace find_m_l63_63620

theorem find_m (m : ℝ) (h : (1 : ℝ) ^ 2 - m * (1 : ℝ) + 2 = 0) : m = 3 :=
by
  sorry

end find_m_l63_63620


namespace chalk_boxes_needed_l63_63859

theorem chalk_boxes_needed (pieces_per_box : ℕ) (total_pieces : ℕ) (pieces_per_box_pos : pieces_per_box > 0) : 
  (total_pieces + pieces_per_box - 1) / pieces_per_box = 194 :=
by 
  let boxes_needed := (total_pieces + pieces_per_box - 1) / pieces_per_box
  have h: boxes_needed = 194 := sorry
  exact h

end chalk_boxes_needed_l63_63859


namespace range_of_a_l63_63182

noncomputable def A := { x : ℝ | 0 < x ∧ x < 2 }
noncomputable def B (a : ℝ) := { x : ℝ | 0 < x ∧ x < (2 / a) }

theorem range_of_a (a : ℝ) (h : 0 < a) : (A ∩ (B a)) = A → 0 < a ∧ a ≤ 1 := by
  sorry

end range_of_a_l63_63182


namespace find_a_l63_63184

noncomputable def g (x : ℝ) := 5 * x - 7

theorem find_a (a : ℝ) (h : g a = 0) : a = 7 / 5 :=
sorry

end find_a_l63_63184


namespace probability_two_points_one_unit_apart_l63_63275

theorem probability_two_points_one_unit_apart :
  let total_points := 10
  let total_ways := (total_points * (total_points - 1)) / 2
  let favorable_horizontal_pairs := 8
  let favorable_vertical_pairs := 5
  let favorable_pairs := favorable_horizontal_pairs + favorable_vertical_pairs
  let probability := (favorable_pairs : ℚ) / total_ways
  probability = 13 / 45 :=
by
  sorry

end probability_two_points_one_unit_apart_l63_63275


namespace kombucha_bottles_after_refund_l63_63073

noncomputable def bottles_per_month : ℕ := 15
noncomputable def cost_per_bottle : ℝ := 3.0
noncomputable def refund_per_bottle : ℝ := 0.10
noncomputable def months_in_year : ℕ := 12

theorem kombucha_bottles_after_refund :
  let bottles_per_year := bottles_per_month * months_in_year
  let total_refund := bottles_per_year * refund_per_bottle
  let bottles_bought_with_refund := total_refund / cost_per_bottle
  bottles_bought_with_refund = 6 := sorry

end kombucha_bottles_after_refund_l63_63073


namespace part1_solution_part2_solution_l63_63892

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |x - 1|

theorem part1_solution :
  {x : ℝ | f x > 4} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x} :=
by
  sorry

theorem part2_solution (x0 : ℝ) :
  (∃ x0 : ℝ, ∀ t : ℝ, f x0 < |(x0 + t)| + |(t - x0)|) →
  ∀ m : ℝ, (f x0 < |m + t| + |t - m|) ↔ m ≠ 0 ∧ (|m| > 5 / 4) :=
by
  sorry

end part1_solution_part2_solution_l63_63892


namespace John_spent_fraction_toy_store_l63_63592

variable (weekly_allowance arcade_money toy_store_money candy_store_money : ℝ)
variable (spend_fraction : ℝ)

-- John's conditions
def John_conditions : Prop :=
  weekly_allowance = 3.45 ∧
  arcade_money = 3 / 5 * weekly_allowance ∧
  candy_store_money = 0.92 ∧
  toy_store_money = weekly_allowance - arcade_money - candy_store_money

-- Theorem to prove the fraction spent at the toy store
theorem John_spent_fraction_toy_store :
  John_conditions weekly_allowance arcade_money toy_store_money candy_store_money →
  spend_fraction = toy_store_money / (weekly_allowance - arcade_money) →
  spend_fraction = 1 / 3 :=
by
  sorry

end John_spent_fraction_toy_store_l63_63592


namespace solve_trig_eq_l63_63319

open Real

theorem solve_trig_eq (k : ℤ) : 
  (∃ x : ℝ, 
    (|cos x| + cos (3 * x)) / (sin x * cos (2 * x)) = -2 * sqrt 3 
    ∧ (x = -π/6 + 2 * k * π ∨ x = 2 * π/3 + 2 * k * π ∨ x = 7 * π/6 + 2 * k * π)) :=
sorry

end solve_trig_eq_l63_63319


namespace sarah_bought_new_shirts_l63_63929

-- Define the given conditions
def original_shirts : ℕ := 9
def total_shirts : ℕ := 17

-- The proof statement: Prove that the number of new shirts is 8
theorem sarah_bought_new_shirts : total_shirts - original_shirts = 8 := by
  sorry

end sarah_bought_new_shirts_l63_63929


namespace turquoise_beads_count_l63_63830

-- Define the conditions
def num_beads_total : ℕ := 40
def num_amethyst : ℕ := 7
def num_amber : ℕ := 2 * num_amethyst

-- Define the main theorem to prove
theorem turquoise_beads_count :
  num_beads_total - (num_amethyst + num_amber) = 19 :=
by
  sorry

end turquoise_beads_count_l63_63830


namespace range_of_a_l63_63999

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (1/2)^x = 3 * a + 2 ∧ x < 0) ↔ (a > -1 / 3) :=
by
  sorry

end range_of_a_l63_63999


namespace five_power_l63_63454

theorem five_power (a : ℕ) (h : 5^a = 3125) : 5^(a - 3) = 25 := 
  sorry

end five_power_l63_63454


namespace repeating_decimal_sum_l63_63010

noncomputable def a : ℚ := 0.66666667 -- Repeating decimal 0.666... corresponds to 2/3
noncomputable def b : ℚ := 0.22222223 -- Repeating decimal 0.222... corresponds to 2/9
noncomputable def c : ℚ := 0.44444445 -- Repeating decimal 0.444... corresponds to 4/9
noncomputable def d : ℚ := 0.99999999 -- Repeating decimal 0.999... corresponds to 1

theorem repeating_decimal_sum : a + b - c + d = 13 / 9 := by
  sorry

end repeating_decimal_sum_l63_63010


namespace min_value_ab_min_value_a_plus_2b_l63_63003
open Nat

theorem min_value_ab (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_cond : a * b = 2 * a + b) : 8 ≤ a * b :=
by
  sorry

theorem min_value_a_plus_2b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_cond : a * b = 2 * a + b) : 9 ≤ a + 2 * b :=
by
  sorry

end min_value_ab_min_value_a_plus_2b_l63_63003


namespace find_m_for_integer_solution_l63_63744

theorem find_m_for_integer_solution :
  ∀ (m x : ℤ), (x^3 - m*x^2 + m*x - (m^2 + 1) = 0) → (m = -3 ∨ m = 0) :=
by
  sorry

end find_m_for_integer_solution_l63_63744


namespace john_profit_percentage_is_50_l63_63887

noncomputable def profit_percentage
  (P : ℝ)  -- The sum of money John paid for purchasing 30 pens
  (recovered_amount : ℝ)  -- The amount John recovered when he sold 20 pens
  (condition : recovered_amount = P) -- Condition that John recovered the full amount P when he sold 20 pens
  : ℝ := 
  ((P / 20) - (P / 30)) / (P / 30) * 100

theorem john_profit_percentage_is_50
  (P : ℝ)
  (recovered_amount : ℝ)
  (condition : recovered_amount = P) :
  profit_percentage P recovered_amount condition = 50 := 
  by 
  sorry

end john_profit_percentage_is_50_l63_63887


namespace sum_of_squares_bounds_l63_63466

-- Given quadrilateral vertices' distances from the nearest vertices of the square
variable (w x y z : ℝ)
-- The side length of the square
def side_length_square : ℝ := 1

-- Expression for the square of each side of the quadrilateral
def square_AB : ℝ := w^2 + x^2
def square_BC : ℝ := (side_length_square - x)^2 + y^2
def square_CD : ℝ := (side_length_square - y)^2 + z^2
def square_DA : ℝ := (side_length_square - z)^2 + (side_length_square - w)^2

-- Sum of the squares of the sides
def sum_of_squares := square_AB w x + square_BC x y + square_CD y z + square_DA z w

-- Proof that the sum of the squares is within the bounds [2, 4]
theorem sum_of_squares_bounds (hw : 0 ≤ w ∧ w ≤ side_length_square)
                              (hx : 0 ≤ x ∧ x ≤ side_length_square)
                              (hy : 0 ≤ y ∧ y ≤ side_length_square)
                              (hz : 0 ≤ z ∧ z ≤ side_length_square)
                              : 2 ≤ sum_of_squares w x y z ∧ sum_of_squares w x y z ≤ 4 := sorry

end sum_of_squares_bounds_l63_63466


namespace find_y_from_eqns_l63_63153

theorem find_y_from_eqns (x y : ℝ) (h1 : x^2 - x + 6 = y + 2) (h2 : x = -5) : y = 34 :=
by {
  sorry
}

end find_y_from_eqns_l63_63153


namespace locus_of_centers_l63_63579

-- The Lean 4 statement
theorem locus_of_centers (a b : ℝ) 
  (C1 : (x y : ℝ) → x^2 + y^2 = 1)
  (C2 : (x y : ℝ) → (x - 3)^2 + y^2 = 25) :
  4 * a^2 + 4 * b^2 - 52 * a - 169 = 0 :=
sorry

end locus_of_centers_l63_63579


namespace student_C_has_sweetest_water_l63_63729

-- Define concentrations for each student
def concentration_A : ℚ := 35 / 175 * 100
def concentration_B : ℚ := 45 / 175 * 100
def concentration_C : ℚ := 65 / 225 * 100

-- Prove that Student C has the highest concentration
theorem student_C_has_sweetest_water :
  concentration_C > concentration_B ∧ concentration_C > concentration_A :=
by
  -- By direct calculation from the provided conditions
  sorry

end student_C_has_sweetest_water_l63_63729


namespace race_cars_count_l63_63103

theorem race_cars_count:
  (1 / 7 + 1 / 3 + 1 / 5 = 0.6761904761904762) -> 
  (∀ N : ℕ, (1 / N = 1 / 7 ∨ 1 / N = 1 / 3 ∨ 1 / N = 1 / 5)) -> 
  (1 / 105 = 0.6761904761904762) :=
by
  intro h_sum_probs h_indiv_probs
  sorry

end race_cars_count_l63_63103


namespace conic_sections_of_equation_l63_63221

noncomputable def is_parabola (s : Set (ℝ × ℝ)) : Prop :=
∃ a b c : ℝ, ∀ x y : ℝ, (x, y) ∈ s ↔ y ≠ 0 ∧ y = a * x^3 + b * x + c

theorem conic_sections_of_equation :
  let eq := { p : ℝ × ℝ | p.2^6 - 9 * p.1^6 = 3 * p.2^3 - 1 }
  (is_parabola eq1) → (is_parabola eq2) → (eq = eq1 ∪ eq2) :=
by sorry

end conic_sections_of_equation_l63_63221


namespace problem_l63_63079

noncomputable def a : Real := 9^(1/3)
noncomputable def b : Real := 3^(2/5)
noncomputable def c : Real := 4^(1/5)

theorem problem (a := 9^(1/3)) (b := 3^(2/5)) (c := 4^(1/5)) : a > b ∧ b > c := by
  sorry

end problem_l63_63079


namespace find_x_l63_63191

def is_mean_twice_mode (l : List ℕ) (mean eq_mode : ℕ) : Prop :=
  l.sum / l.length = eq_mode * 2

theorem find_x (x : ℕ) (h1 : x > 0) (h2 : x ≤ 100)
  (h3 : is_mean_twice_mode [20, x, x, x, x] x (x * 2)) : x = 10 :=
sorry

end find_x_l63_63191


namespace tagged_fish_in_second_catch_l63_63399

theorem tagged_fish_in_second_catch
  (N : ℕ)
  (initial_catch tagged_returned : ℕ)
  (second_catch : ℕ)
  (approximate_pond_fish : ℕ)
  (condition_1 : initial_catch = 60)
  (condition_2 : tagged_returned = 60)
  (condition_3 : second_catch = 60)
  (condition_4 : approximate_pond_fish = 1800) :
  (tagged_returned * second_catch) / approximate_pond_fish = 2 :=
by
  sorry

end tagged_fish_in_second_catch_l63_63399


namespace area_of_QCA_l63_63872

noncomputable def area_of_triangle (x p : ℝ) (hx_pos : 0 < x) (hp_bounds : 0 < p ∧ p < 15) : ℝ :=
  1 / 2 * x * (15 - p)

theorem area_of_QCA (x : ℝ) (p : ℝ) (hx_pos : 0 < x) (hp_bounds : 0 < p ∧ p < 15) :
  area_of_triangle x p hx_pos hp_bounds = 1 / 2 * x * (15 - p) :=
sorry

end area_of_QCA_l63_63872


namespace annual_interest_rate_l63_63253

-- Define the initial conditions
def P : ℝ := 5600
def A : ℝ := 6384
def t : ℝ := 2
def n : ℝ := 1

-- The theorem statement:
theorem annual_interest_rate : ∃ (r : ℝ), A = P * (1 + r / n) ^ (n * t) ∧ r = 0.067 :=
by 
  sorry -- proof goes here

end annual_interest_rate_l63_63253


namespace find_asymptote_slope_l63_63174

theorem find_asymptote_slope :
  (∀ x y : ℝ, (x^2 / 144 - y^2 / 81 = 0) → (y = 3/4 * x ∨ y = -3/4 * x)) :=
by
  sorry

end find_asymptote_slope_l63_63174


namespace part1_part2_l63_63159

noncomputable def f (a x : ℝ) := a * x^2 - (a + 1) * x + 1

theorem part1 (a : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 2) ↔ (-3 - 2 * Real.sqrt 2 ≤ a ∧ a ≤ -3 + 2 * Real.sqrt 2) :=
sorry

theorem part2 (a : ℝ) (h1 : a ≠ 0) (x : ℝ) :
  (f a x < 0) ↔
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1 / a) ∨
     (a = 1 ∧ false) ∨
     (a > 1 ∧ 1 / a < x ∧ x < 1) ∨
     (a < 0 ∧ (x < 1 / a ∨ x > 1))) :=
sorry

end part1_part2_l63_63159


namespace markup_is_correct_l63_63805

noncomputable def profit (S : ℝ) : ℝ := 0.12 * S
noncomputable def expenses (S : ℝ) : ℝ := 0.10 * S
noncomputable def cost (S : ℝ) : ℝ := S - (profit S + expenses S)
noncomputable def markup (S : ℝ) : ℝ :=
  ((S - cost S) / (cost S)) * 100

theorem markup_is_correct:
  markup 10 = 28.21 :=
by
  sorry

end markup_is_correct_l63_63805


namespace find_prime_A_l63_63272

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_prime_A (A : ℕ) :
  is_prime A ∧ is_prime (A + 14) ∧ is_prime (A + 18) ∧ is_prime (A + 32) ∧ is_prime (A + 36) → A = 5 := by
  sorry

end find_prime_A_l63_63272


namespace nh3_oxidation_mass_l63_63639

theorem nh3_oxidation_mass
  (initial_volume : ℚ)
  (initial_cl2_percentage : ℚ)
  (initial_n2_percentage : ℚ)
  (escaped_volume : ℚ)
  (escaped_cl2_percentage : ℚ)
  (escaped_n2_percentage : ℚ)
  (molar_volume : ℚ)
  (cl2_molar_mass : ℚ)
  (nh3_molar_mass : ℚ) :
  initial_volume = 1.12 →
  initial_cl2_percentage = 0.9 →
  initial_n2_percentage = 0.1 →
  escaped_volume = 0.672 →
  escaped_cl2_percentage = 0.5 →
  escaped_n2_percentage = 0.5 →
  molar_volume = 22.4 →
  cl2_molar_mass = 71 →
  nh3_molar_mass = 17 →
  ∃ (mass_nh3_oxidized : ℚ),
    mass_nh3_oxidized = 0.34 := 
by {
  sorry
}

end nh3_oxidation_mass_l63_63639


namespace value_of_2_68_times_0_74_l63_63960

theorem value_of_2_68_times_0_74 : 
  (268 * 74 = 19732) → (2.68 * 0.74 = 1.9732) :=
by intro h1; sorry

end value_of_2_68_times_0_74_l63_63960


namespace find_k_l63_63505

-- Auxiliary function to calculate the product of the digits of a number
def productOfDigits (n : ℕ) : ℕ :=
  (n.digits 10).foldl (λ acc d => acc * d) 1

theorem find_k (k : ℕ) (h1 : 0 < k) (h2 : productOfDigits k = (25 * k) / 8 - 211) : 
  k = 72 ∨ k = 88 :=
by
  sorry

end find_k_l63_63505


namespace intersection_is_correct_l63_63460

namespace IntervalProofs

def setA := {x : ℝ | 3 * x^2 - 14 * x + 16 ≤ 0}
def setB := {x : ℝ | (3 * x - 7) / x > 0}

theorem intersection_is_correct :
  {x | 7 / 3 < x ∧ x ≤ 8 / 3} = setA ∩ setB :=
by
  sorry

end IntervalProofs

end intersection_is_correct_l63_63460


namespace number_of_3_letter_words_with_at_least_one_A_l63_63233

theorem number_of_3_letter_words_with_at_least_one_A :
  let all_words := 5^3
  let no_A_words := 4^3
  all_words - no_A_words = 61 :=
by
  sorry

end number_of_3_letter_words_with_at_least_one_A_l63_63233


namespace snow_probability_january_first_week_l63_63691

noncomputable def P_snow_at_least_once_first_week : ℚ :=
  1 - ((2 / 3) ^ 4 * (3 / 4) ^ 3)

theorem snow_probability_january_first_week :
  P_snow_at_least_once_first_week = 11 / 12 :=
by
  sorry

end snow_probability_january_first_week_l63_63691


namespace volume_frustum_l63_63423

noncomputable def volume_of_frustum (base_edge_original : ℝ) (altitude_original : ℝ) 
(base_edge_smaller : ℝ) (altitude_smaller : ℝ) : ℝ :=
let volume_original := (1 / 3) * (base_edge_original ^ 2) * altitude_original
let volume_smaller := (1 / 3) * (base_edge_smaller ^ 2) * altitude_smaller
(volume_original - volume_smaller)

theorem volume_frustum
  (base_edge_original : ℝ) (altitude_original : ℝ) 
  (base_edge_smaller : ℝ) (altitude_smaller : ℝ)
  (h_base_edge_original : base_edge_original = 10)
  (h_altitude_original : altitude_original = 10)
  (h_base_edge_smaller : base_edge_smaller = 5)
  (h_altitude_smaller : altitude_smaller = 5) :
  volume_of_frustum base_edge_original altitude_original base_edge_smaller altitude_smaller = (875 / 3) :=
by
  rw [h_base_edge_original, h_altitude_original, h_base_edge_smaller, h_altitude_smaller]
  simp [volume_of_frustum]
  sorry

end volume_frustum_l63_63423


namespace students_tried_out_l63_63737

theorem students_tried_out (x : ℕ) (h1 : 8 * (x - 17) = 384) : x = 65 := 
by
  sorry

end students_tried_out_l63_63737


namespace remaining_distance_is_one_l63_63951

def total_distance_to_grandma : ℕ := 78
def initial_distance_traveled : ℕ := 35
def bakery_detour : ℕ := 7
def pie_distance : ℕ := 18
def gift_detour : ℕ := 3
def next_travel_distance : ℕ := 12
def scenic_detour : ℕ := 2

def total_distance_traveled : ℕ :=
  initial_distance_traveled + bakery_detour + pie_distance + gift_detour + next_travel_distance + scenic_detour

theorem remaining_distance_is_one :
  total_distance_to_grandma - total_distance_traveled = 1 := by
  sorry

end remaining_distance_is_one_l63_63951


namespace determine_value_of_a_l63_63386

theorem determine_value_of_a :
  ∃ b, (∀ x : ℝ, (4 * x^2 + 12 * x + (b^2)) = (2 * x + b)^2) :=
sorry

end determine_value_of_a_l63_63386


namespace obtain_1_after_3_operations_obtain_1_after_4_operations_obtain_1_after_5_operations_l63_63400

def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 3

theorem obtain_1_after_3_operations:
  (operation (operation (operation 1)) = 1) ∨ 
  (operation (operation (operation 8)) = 1) := by
  sorry

theorem obtain_1_after_4_operations:
  (operation (operation (operation (operation 1))) = 1) ∨ 
  (operation (operation (operation (operation 5))) = 1) ∨ 
  (operation (operation (operation (operation 16))) = 1) := by
  sorry

theorem obtain_1_after_5_operations:
  (operation (operation (operation (operation (operation 4)))) = 1) ∨ 
  (operation (operation (operation (operation (operation 10)))) = 1) ∨ 
  (operation (operation (operation (operation (operation 13)))) = 1) := by
  sorry

end obtain_1_after_3_operations_obtain_1_after_4_operations_obtain_1_after_5_operations_l63_63400


namespace charlie_pennies_l63_63179

variable (a c : ℕ)

theorem charlie_pennies (h1 : c + 1 = 4 * (a - 1)) (h2 : c - 1 = 3 * (a + 1)) : c = 31 := 
by
  sorry

end charlie_pennies_l63_63179


namespace train_length_l63_63802

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def jogger_head_start_m : ℝ := 240
noncomputable def train_passing_time_s : ℝ := 35.99712023038157

noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * (1000 / 3600)
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps
noncomputable def distance_covered_by_train : ℝ := relative_speed_mps * train_passing_time_s

theorem train_length :
  distance_covered_by_train - jogger_head_start_m = 119.9712023038157 :=
by
  sorry

end train_length_l63_63802


namespace triangle_inequality_for_powers_l63_63578

theorem triangle_inequality_for_powers (a b c : ℝ) :
  (∀ n : ℕ, (a ^ n + b ^ n > c ^ n)) ↔ (a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b = c) :=
sorry

end triangle_inequality_for_powers_l63_63578


namespace find_a_decreasing_l63_63248

-- Define the given function
def f (a x : ℝ) : ℝ := (x - 1) ^ 2 + 2 * a * x + 1

-- State the condition
def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y ≤ f x

-- State the proposition
theorem find_a_decreasing :
  ∀ a : ℝ, is_decreasing_on (f a) (Set.Iio 4) → a ≤ -3 :=
by
  intro a
  intro h
  sorry

end find_a_decreasing_l63_63248


namespace sum_of_squares_l63_63961

theorem sum_of_squares (b j s : ℕ) 
  (h1 : 3 * b + 2 * j + 4 * s = 70)
  (h2 : 4 * b + 3 * j + 2 * s = 88) : 
  b^2 + j^2 + s^2 = 405 := 
sorry

end sum_of_squares_l63_63961


namespace factor_expression_l63_63306

theorem factor_expression (b : ℝ) :
  (8 * b^4 - 100 * b^3 + 14 * b^2) - (3 * b^4 - 10 * b^3 + 14 * b^2) = 5 * b^3 * (b - 18) :=
by
  sorry

end factor_expression_l63_63306


namespace pipe_A_fill_time_l63_63571

theorem pipe_A_fill_time (t : ℕ) : 
  (∀ x : ℕ, x = 40 → (1 * x) = 40) ∧
  (∀ y : ℕ, y = 30 → (15/40) + ((1/t) + (1/40)) * 15 = 1) ∧ t = 60 :=
sorry

end pipe_A_fill_time_l63_63571


namespace kevin_leap_day_2024_is_monday_l63_63475

def days_between_leap_birthdays (years: ℕ) (leap_year_count: ℕ) : ℕ :=
  (years - leap_year_count) * 365 + leap_year_count * 366

def day_of_week_after_days (start_day: ℕ) (days: ℕ) : ℕ :=
  (start_day + days) % 7

noncomputable def kevin_leap_day_weekday_2024 : ℕ :=
  let days := days_between_leap_birthdays 24 6
  let start_day := 2 -- Tuesday as 2 (assuming 0 = Sunday, 1 = Monday,..., 6 = Saturday)
  day_of_week_after_days start_day days

theorem kevin_leap_day_2024_is_monday :
  kevin_leap_day_weekday_2024 = 1 -- 1 represents Monday
  :=
by
  sorry

end kevin_leap_day_2024_is_monday_l63_63475


namespace time_for_10_strikes_l63_63447

-- Assume a clock takes 7 seconds to strike 7 times
def clock_time_for_N_strikes (N : ℕ) : ℕ :=
  if N = 7 then 7 else sorry  -- This would usually be a function, simplified here for the specific condition

-- Assume there are 6 intervals for 7 strikes
def intervals_between_strikes (N : ℕ) : ℕ :=
  if N = 7 then 6 else N - 1

-- Function to calculate total time for any number of strikes based on intervals and time per strike
def total_time_for_strikes (N : ℕ) : ℚ :=
  (intervals_between_strikes N) * (clock_time_for_N_strikes 7 / intervals_between_strikes 7 : ℚ)

theorem time_for_10_strikes : total_time_for_strikes 10 = 10.5 :=
by
  -- Insert proof here
  sorry

end time_for_10_strikes_l63_63447


namespace rotated_angle_new_measure_l63_63846

theorem rotated_angle_new_measure (initial_angle : ℝ) (rotation : ℝ) (final_angle : ℝ) :
  initial_angle = 60 ∧ rotation = 300 → final_angle = 120 :=
by
  intros h
  sorry

end rotated_angle_new_measure_l63_63846


namespace age_problem_l63_63841

theorem age_problem
    (D X : ℕ) 
    (h1 : D = 4 * X) 
    (h2 : D = X + 30) : D = 40 ∧ X = 10 := by
  sorry

end age_problem_l63_63841


namespace original_price_eq_36_l63_63786

-- Definitions for the conditions
def first_cup_price (x : ℕ) : ℕ := x
def second_cup_price (x : ℕ) : ℕ := x / 2
def third_cup_price : ℕ := 3
def total_cost (x : ℕ) : ℕ := x + (x / 2) + third_cup_price
def average_price (total : ℕ) : ℕ := total / 3

-- The proof statement
theorem original_price_eq_36 (x : ℕ) (h : total_cost x = 57) : x = 36 :=
  sorry

end original_price_eq_36_l63_63786


namespace cost_of_gasoline_l63_63696

def odometer_initial : ℝ := 85120
def odometer_final : ℝ := 85150
def fuel_efficiency : ℝ := 30
def price_per_gallon : ℝ := 4.25

theorem cost_of_gasoline : 
  ((odometer_final - odometer_initial) / fuel_efficiency) * price_per_gallon = 4.25 := 
by 
  sorry

end cost_of_gasoline_l63_63696


namespace minimum_number_of_colors_l63_63970

theorem minimum_number_of_colors (n : ℕ) (h_n : 2 ≤ n) :
  ∀ (f : (Fin n) → ℕ),
  (∀ i j : Fin n, i ≠ j → f i ≠ f j) →
  (∃ c : ℕ, c = n) :=
by sorry

end minimum_number_of_colors_l63_63970


namespace rectangle_length_twice_breadth_l63_63812

theorem rectangle_length_twice_breadth
  (b : ℝ) 
  (l : ℝ)
  (h1 : l = 2 * b)
  (h2 : (l - 5) * (b + 4) = l * b + 75) :
  l = 190 / 3 :=
sorry

end rectangle_length_twice_breadth_l63_63812


namespace a_share_calculation_l63_63245

noncomputable def investment_a : ℕ := 15000
noncomputable def investment_b : ℕ := 21000
noncomputable def investment_c : ℕ := 27000
noncomputable def total_investment : ℕ := investment_a + investment_b + investment_c -- 63000
noncomputable def b_share : ℕ := 1540
noncomputable def total_profit : ℕ := 4620  -- from the solution steps

theorem a_share_calculation :
  (investment_a * total_profit) / total_investment = 1100 := 
by
  sorry

end a_share_calculation_l63_63245


namespace sum_of_tangents_l63_63690

theorem sum_of_tangents (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
    (h_tan_α : Real.tan α = 2) (h_tan_β : Real.tan β = 3) : α + β = 3 * π / 4 :=
by
  sorry

end sum_of_tangents_l63_63690


namespace solution_set_16_sin_pi_x_cos_pi_x_l63_63749

theorem solution_set_16_sin_pi_x_cos_pi_x (x : ℝ) :
  (x = 1 / 4 ∨ x = -1 / 4) ↔ 16 * Real.sin (π * x) * Real.cos (π * x) = 16 * x + 1 / x :=
sorry

end solution_set_16_sin_pi_x_cos_pi_x_l63_63749


namespace graduation_graduates_l63_63286

theorem graduation_graduates :
  ∃ G : ℕ, (∀ (chairs_for_parents chairs_for_teachers chairs_for_admins : ℕ),
    chairs_for_parents = 2 * G ∧
    chairs_for_teachers = 20 ∧
    chairs_for_admins = 10 ∧
    G + chairs_for_parents + chairs_for_teachers + chairs_for_admins = 180) ↔ G = 50 :=
by
  sorry

end graduation_graduates_l63_63286


namespace number_to_add_l63_63747

theorem number_to_add (a b n : ℕ) (h_a : a = 425897) (h_b : b = 456) (h_n : n = 47) : 
  (a + n) % b = 0 :=
by
  rw [h_a, h_b, h_n]
  sorry

end number_to_add_l63_63747


namespace caps_eaten_correct_l63_63091

def initial_bottle_caps : ℕ := 34
def remaining_bottle_caps : ℕ := 26
def eaten_bottle_caps (k_i k_r : ℕ) : ℕ := k_i - k_r

theorem caps_eaten_correct :
  eaten_bottle_caps initial_bottle_caps remaining_bottle_caps = 8 :=
by
  sorry

end caps_eaten_correct_l63_63091


namespace ab_eq_one_l63_63633

theorem ab_eq_one (a b : ℝ) (h1 : a ≠ b) (h2 : abs (Real.log a) = abs (Real.log b)) : a * b = 1 := sorry

end ab_eq_one_l63_63633


namespace max_regions_11_l63_63063

noncomputable def max_regions (n : ℕ) : ℕ :=
  1 + n * (n + 1) / 2

theorem max_regions_11 : max_regions 11 = 67 := by
  unfold max_regions
  norm_num

end max_regions_11_l63_63063


namespace car_owners_without_motorcycles_l63_63630

theorem car_owners_without_motorcycles
  (total_adults : ℕ)
  (car_owners : ℕ)
  (motorcycle_owners : ℕ)
  (all_owners : total_adults = 400)
  (john_owns_cars : car_owners = 370)
  (john_owns_motorcycles : motorcycle_owners = 50)
  (all_adult_owners : total_adults = car_owners + motorcycle_owners - (car_owners - motorcycle_owners)) : 
  (car_owners - (car_owners + motorcycle_owners - total_adults) = 350) :=
by {
  sorry
}

end car_owners_without_motorcycles_l63_63630


namespace exam_paper_max_marks_l63_63093

/-- A candidate appearing for an examination has to secure 40% marks to pass paper i.
    The candidate secured 40 marks and failed by 20 marks.
    Prove that the maximum mark for paper i is 150. -/
theorem exam_paper_max_marks (p : ℝ) (s f : ℝ) (M : ℝ) (h1 : p = 0.40) (h2 : s = 40) (h3 : f = 20) (h4 : p * M = s + f) :
  M = 150 :=
sorry

end exam_paper_max_marks_l63_63093


namespace total_legs_in_park_l63_63281

theorem total_legs_in_park :
  let dogs := 109
  let cats := 37
  let birds := 52
  let spiders := 19
  let dog_legs := 4
  let cat_legs := 4
  let bird_legs := 2
  let spider_legs := 8
  dogs * dog_legs + cats * cat_legs + birds * bird_legs + spiders * spider_legs = 840 := by
  sorry

end total_legs_in_park_l63_63281


namespace range_of_k_l63_63539

theorem range_of_k (k : ℝ) :
  (∃ x y : ℝ, k^2 * x^2 + y^2 - 4 * k * x + 2 * k * y + k^2 - 1 = 0 ∧ (x, y) = (0, 0)) →
  0 < |k| ∧ |k| < 1 :=
by
  intros
  sorry

end range_of_k_l63_63539


namespace find_x_l63_63040

theorem find_x (x : ℕ) (h : (85 + 32 / x : ℝ) * x = 9637) : x = 113 :=
sorry

end find_x_l63_63040


namespace Henry_age_ratio_l63_63880

theorem Henry_age_ratio (A S H : ℕ)
  (hA : A = 15)
  (hS : S = 3 * A)
  (h_sum : A + S + H = 240) :
  H / S = 4 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end Henry_age_ratio_l63_63880


namespace parabola_position_l63_63142

-- Define the two parabolas as functions
def parabola1 (x : ℝ) : ℝ := x^2 - 2 * x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + 2 * x + 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (1, parabola1 1) -- (1, 2)
def vertex2 : ℝ × ℝ := (-1, parabola2 (-1)) -- (-1, 0)

-- Define the proof problem where we show relative positions
theorem parabola_position :
  (vertex1.1 > vertex2.1) ∧ (vertex1.2 > vertex2.2) :=
by
  sorry

end parabola_position_l63_63142


namespace line_plane_intersection_l63_63071

theorem line_plane_intersection :
  (∃ t : ℝ, (x, y, z) = (3 + t, 1 - t, -5) ∧ (3 + t) + 7 * (1 - t) + 3 * (-5) + 11 = 0) →
  (x, y, z) = (4, 0, -5) :=
sorry

end line_plane_intersection_l63_63071


namespace radius_moon_scientific_notation_l63_63018

def scientific_notation := 1738000 = 1.738 * 10^6

theorem radius_moon_scientific_notation : scientific_notation := 
sorry

end radius_moon_scientific_notation_l63_63018


namespace remaining_blocks_correct_l63_63026

-- Define the initial number of blocks
def initial_blocks : ℕ := 59

-- Define the number of blocks used
def used_blocks : ℕ := 36

-- Define the remaining blocks equation
def remaining_blocks : ℕ := initial_blocks - used_blocks

-- Prove that the number of remaining blocks is 23
theorem remaining_blocks_correct : remaining_blocks = 23 := by
  sorry

end remaining_blocks_correct_l63_63026


namespace basketball_games_won_difference_l63_63117

theorem basketball_games_won_difference :
  ∀ (total_games games_won games_lost difference_won_lost : ℕ),
  total_games = 62 →
  games_won = 45 →
  games_lost = 17 →
  difference_won_lost = games_won - games_lost →
  difference_won_lost = 28 :=
by
  intros total_games games_won games_lost difference_won_lost
  intros h_total h_won h_lost h_diff
  rw [h_won, h_lost] at h_diff
  exact h_diff

end basketball_games_won_difference_l63_63117


namespace new_person_weight_l63_63255

theorem new_person_weight (average_increase : ℝ) (num_persons : ℕ) (replaced_weight : ℝ) (new_weight : ℝ) 
  (h1 : num_persons = 10) 
  (h2 : average_increase = 3.2) 
  (h3 : replaced_weight = 65) : 
  new_weight = 97 :=
by
  sorry

end new_person_weight_l63_63255


namespace cyclic_quadrilateral_fourth_side_length_l63_63367

theorem cyclic_quadrilateral_fourth_side_length
  (r : ℝ) (a b c d : ℝ) (r_eq : r = 300 * Real.sqrt 2) (a_eq : a = 300) (b_eq : b = 400)
  (c_eq : c = 300) :
  d = 500 := 
by 
  sorry

end cyclic_quadrilateral_fourth_side_length_l63_63367


namespace set_intersection_eq_l63_63653

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }

def B : Set ℝ := { x | x < -2 ∨ x > 5 }

def C_U (B : Set ℝ) : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }

theorem set_intersection_eq : A ∩ (C_U B) = { x | -2 ≤ x ∧ x ≤ 3 } :=
  sorry

end set_intersection_eq_l63_63653


namespace apples_given_to_father_l63_63704

theorem apples_given_to_father
  (total_apples : ℤ) 
  (people_sharing : ℤ) 
  (apples_per_person : ℤ)
  (jack_and_friends : ℤ) :
  total_apples = 55 →
  people_sharing = 5 →
  apples_per_person = 9 →
  jack_and_friends = 4 →
  (total_apples - people_sharing * apples_per_person) = 10 :=
by 
  intros h1 h2 h3 h4
  sorry

end apples_given_to_father_l63_63704


namespace alicia_tax_deduction_l63_63100

theorem alicia_tax_deduction (earnings_per_hour_in_cents : ℕ) (tax_rate : ℚ) 
  (h1 : earnings_per_hour_in_cents = 2500) (h2 : tax_rate = 0.02) : 
  earnings_per_hour_in_cents * tax_rate = 50 := 
  sorry

end alicia_tax_deduction_l63_63100


namespace real_estate_profit_l63_63486

def purchase_price_first : ℝ := 350000
def purchase_price_second : ℝ := 450000
def purchase_price_third : ℝ := 600000

def gain_first : ℝ := 0.12
def loss_second : ℝ := 0.08
def gain_third : ℝ := 0.18

def selling_price_first : ℝ :=
  purchase_price_first + (purchase_price_first * gain_first)
def selling_price_second : ℝ :=
  purchase_price_second - (purchase_price_second * loss_second)
def selling_price_third : ℝ :=
  purchase_price_third + (purchase_price_third * gain_third)

def total_purchase_price : ℝ :=
  purchase_price_first + purchase_price_second + purchase_price_third
def total_selling_price : ℝ :=
  selling_price_first + selling_price_second + selling_price_third

def overall_gain : ℝ :=
  total_selling_price - total_purchase_price

theorem real_estate_profit :
  overall_gain = 114000 := by
  sorry

end real_estate_profit_l63_63486


namespace mod_equiv_l63_63884

theorem mod_equiv (a b c d e : ℤ) (n : ℤ) (h1 : a = 101)
                                    (h2 : b = 15)
                                    (h3 : c = 7)
                                    (h4 : d = 9)
                                    (h5 : e = 5)
                                    (h6 : n = 17) :
  (a * b - c * d + e) % n = 7 := by
  sorry

end mod_equiv_l63_63884


namespace bus_speed_excluding_stoppages_l63_63672

noncomputable def average_speed_excluding_stoppages
  (speed_including_stoppages : ℝ)
  (stoppage_time_ratio : ℝ) : ℝ :=
  (speed_including_stoppages * 1) / (1 - stoppage_time_ratio)

theorem bus_speed_excluding_stoppages :
  average_speed_excluding_stoppages 15 (3/4) = 60 := 
by
  sorry

end bus_speed_excluding_stoppages_l63_63672


namespace deck_cost_l63_63665

variable (rareCount : ℕ := 19)
variable (uncommonCount : ℕ := 11)
variable (commonCount : ℕ := 30)
variable (rareCost : ℝ := 1.0)
variable (uncommonCost : ℝ := 0.5)
variable (commonCost : ℝ := 0.25)

theorem deck_cost : rareCount * rareCost + uncommonCount * uncommonCost + commonCount * commonCost = 32 := by
  sorry

end deck_cost_l63_63665


namespace vector_addition_example_l63_63518

theorem vector_addition_example : 
  let v1 := (⟨-5, 3⟩ : ℝ × ℝ)
  let v2 := (⟨7, -6⟩ : ℝ × ℝ)
  v1 + v2 = (⟨2, -3⟩ : ℝ × ℝ) := 
by {
  sorry
}

end vector_addition_example_l63_63518


namespace smallest_value_of_3b_plus_2_l63_63615

theorem smallest_value_of_3b_plus_2 (b : ℝ) (h : 8 * b^2 + 7 * b + 6 = 5) : (∃ t : ℝ, t = 3 * b + 2 ∧ (∀ x : ℝ, 8 * x^2 + 7 * x + 6 = 5 → x = b → t ≤ 3 * x + 2)) :=
sorry

end smallest_value_of_3b_plus_2_l63_63615


namespace perfect_square_mod_3_l63_63196

theorem perfect_square_mod_3 (k : ℤ) (hk : ∃ m : ℤ, k = m^2) : k % 3 = 0 ∨ k % 3 = 1 :=
by
  sorry

end perfect_square_mod_3_l63_63196


namespace student_chose_121_l63_63450

theorem student_chose_121 (x : ℤ) (h : 2 * x - 140 = 102) : x = 121 := by
  sorry

end student_chose_121_l63_63450


namespace gcd_ab_is_22_l63_63188

def a : ℕ := 198
def b : ℕ := 308

theorem gcd_ab_is_22 : Nat.gcd a b = 22 := 
by { sorry }

end gcd_ab_is_22_l63_63188


namespace option_c_correct_l63_63032

theorem option_c_correct (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x < 2 → x^2 - a ≤ 0) : 4 < a :=
by
  sorry

end option_c_correct_l63_63032


namespace solution_set_ineq_min_value_sum_l63_63301

-- Part (1)
theorem solution_set_ineq (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1| + |x - 2|) :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 2} :=
sorry

-- Part (2)
theorem min_value_sum (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1| + |x - 2|)
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (hx : ∀ x, f x ≥ (1 / m) + (1 / n)) :
  m + n = 8 / 3 :=
sorry

end solution_set_ineq_min_value_sum_l63_63301


namespace minimize_quadratic_l63_63143

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l63_63143


namespace sqrt_expression_eq_36_l63_63581

theorem sqrt_expression_eq_36 : (Real.sqrt ((3^2 + 3^3)^2)) = 36 := 
by
  sorry

end sqrt_expression_eq_36_l63_63581


namespace distinct_sums_is_98_l63_63165

def arithmetic_sequence_distinct_sums (a_n : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) :=
  (∀ n : ℕ, S n = (n * (2 * a_n 0 + (n - 1) * d)) / 2) ∧
  S 5 = 0 ∧
  d ≠ 0 →
  (∃ distinct_count : ℕ, distinct_count = 98 ∧
   ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100 ∧ S i = S j → i = j)

theorem distinct_sums_is_98 (a_n : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h : arithmetic_sequence_distinct_sums a_n S d) :
  ∃ distinct_count : ℕ, distinct_count = 98 :=
sorry

end distinct_sums_is_98_l63_63165


namespace students_in_classroom_l63_63130

/-- There are some students in a classroom. Half of them have 5 notebooks each and the other half have 3 notebooks each. There are 112 notebooks in total in the classroom. Prove the number of students is 28. -/
theorem students_in_classroom (S : ℕ) (h1 : (S / 2) * 5 + (S / 2) * 3 = 112) : S = 28 := 
sorry

end students_in_classroom_l63_63130


namespace medicine_price_after_discount_l63_63930

theorem medicine_price_after_discount :
  ∀ (price : ℝ) (discount : ℝ), price = 120 → discount = 0.3 → 
  (price - price * discount) = 84 :=
by
  intros price discount h1 h2
  rw [h1, h2]
  sorry

end medicine_price_after_discount_l63_63930


namespace algebraic_expression_no_linear_term_l63_63594

theorem algebraic_expression_no_linear_term (a : ℝ) :
  (∀ x : ℝ, (x + a) * (x - 1/2) = x^2 - a/2 ↔ a = 1/2) :=
by
  sorry

end algebraic_expression_no_linear_term_l63_63594


namespace range_of_k_l63_63705

theorem range_of_k {x k : ℝ} :
  (∀ x, ((x - 2) * (x + 1) > 0) → ((2 * x + 7) * (x + k) < 0)) →
  (x = -3 ∨ x = -2) → 
  -3 ≤ k ∧ k < 2 :=
sorry

end range_of_k_l63_63705


namespace max_points_of_intersection_l63_63765

-- Definitions based on the conditions in a)
def intersects_circle (l : ℕ) : ℕ := 2 * l  -- Each line intersects the circle at most twice
def intersects_lines (n : ℕ) : ℕ := n * (n - 1) / 2  -- Number of intersection points between lines (combinatorial)

-- The main statement that needs to be proved
theorem max_points_of_intersection (lines circle : ℕ) (h_lines_distinct : lines = 3) (h_no_parallel : ∀ (i j : ℕ), i ≠ j → i < lines → j < lines → true) (h_no_common_point : ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬(true)) : (intersects_circle lines + intersects_lines lines = 9) := 
  by
    sorry

end max_points_of_intersection_l63_63765


namespace jasmine_additional_cans_needed_l63_63590

theorem jasmine_additional_cans_needed
  (n_initial : ℕ)
  (n_lost : ℕ)
  (n_remaining : ℕ)
  (additional_can_coverage : ℕ)
  (n_needed : ℕ) :
  n_initial = 50 →
  n_lost = 4 →
  n_remaining = 36 →
  additional_can_coverage = 2 →
  n_needed = 7 :=
by
  sorry

end jasmine_additional_cans_needed_l63_63590


namespace number_of_ways_to_choose_l63_63095

-- Define the teachers and classes
def teachers : ℕ := 5
def classes : ℕ := 4
def choices (t : ℕ) : ℕ := classes

-- Formalize the problem statement
theorem number_of_ways_to_choose : (choices teachers) ^ teachers = 1024 :=
by
  -- We denote the computation of (4^5)
  sorry

end number_of_ways_to_choose_l63_63095


namespace quadratic_inequality_l63_63864

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, x^2 + 2 * a * x + a > 0) : 0 < a ∧ a < 1 :=
sorry

end quadratic_inequality_l63_63864


namespace find_polynomial_q_l63_63490

theorem find_polynomial_q (q : ℝ → ℝ) :
  (∀ x : ℝ, q x + (x^6 + 4*x^4 + 8*x^2 + 7*x) = (12*x^4 + 30*x^3 + 40*x^2 + 10*x + 2)) →
  (∀ x : ℝ, q x = -x^6 + 8*x^4 + 30*x^3 + 32*x^2 + 3*x + 2) :=
by 
  sorry

end find_polynomial_q_l63_63490


namespace next_elements_l63_63857

-- Define the conditions and the question
def next_elements_in_sequence (n : ℕ) : String :=
  match n with
  | 1 => "О"  -- "Один"
  | 2 => "Д"  -- "Два"
  | 3 => "Т"  -- "Три"
  | 4 => "Ч"  -- "Четыре"
  | 5 => "П"  -- "Пять"
  | 6 => "Ш"  -- "Шесть"
  | 7 => "С"  -- "Семь"
  | 8 => "В"  -- "Восемь"
  | _ => "?"

theorem next_elements (n : ℕ) :
  next_elements_in_sequence 7 = "С" ∧ next_elements_in_sequence 8 = "В" := by
  sorry

end next_elements_l63_63857


namespace increase_80_by_150_percent_l63_63981

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l63_63981


namespace fraction_of_reciprocal_l63_63499

theorem fraction_of_reciprocal (x : ℝ) (hx : 0 < x) (h : (2/3) * x = y / x) (hx1 : x = 1) : y = 2/3 :=
by
  sorry

end fraction_of_reciprocal_l63_63499


namespace rectangle_area_l63_63098

theorem rectangle_area (L B r s : ℝ) (h1 : L = 5 * r)
                       (h2 : r = s)
                       (h3 : s^2 = 16)
                       (h4 : B = 11) :
  (L * B = 220) :=
by
  sorry

end rectangle_area_l63_63098


namespace fa_plus_fb_gt_zero_l63_63038

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- Define the conditions for a and b
variables (a b : ℝ)
axiom ab_pos : a + b > 0

-- State the theorem
theorem fa_plus_fb_gt_zero : f a + f b > 0 :=
sorry

end fa_plus_fb_gt_zero_l63_63038


namespace jacket_final_price_l63_63694

theorem jacket_final_price :
    let initial_price := 150
    let first_discount := 0.30
    let second_discount := 0.10
    let coupon := 10
    let tax := 0.05
    let price_after_first_discount := initial_price * (1 - first_discount)
    let price_after_second_discount := price_after_first_discount * (1 - second_discount)
    let price_after_coupon := price_after_second_discount - coupon
    let final_price := price_after_coupon * (1 + tax)
    final_price = 88.725 :=
by
  sorry

end jacket_final_price_l63_63694


namespace ratio_proof_l63_63057

-- Definitions and conditions
variables {A B C : ℕ}

-- Given condition: A : B : C = 3 : 2 : 5
def ratio_cond (A B C : ℕ) := 3 * B = 2 * A ∧ 5 * B = 2 * C

-- Theorem statement
theorem ratio_proof (h : ratio_cond A B C) : (2 * A + 3 * B) / (A + 5 * C) = 3 / 7 :=
by sorry

end ratio_proof_l63_63057


namespace scientific_notation_of_diameter_l63_63343

theorem scientific_notation_of_diameter :
  0.00000258 = 2.58 * 10^(-6) :=
by sorry

end scientific_notation_of_diameter_l63_63343


namespace inequality_solution_sets_l63_63167

variable (a x : ℝ)

theorem inequality_solution_sets:
    ({x | 12 * x^2 - a * x > a^2} =
        if a > 0 then {x | x < -a/4} ∪ {x | x > a/3}
        else if a = 0 then {x | x ≠ 0}
        else {x | x < a/3} ∪ {x | x > -a/4}) :=
by sorry

end inequality_solution_sets_l63_63167


namespace sufficient_but_not_necessary_condition_l63_63052

noncomputable def f (x a : ℝ) : ℝ := abs (x - a)

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≤ -2) ↔ (∀ x y : ℝ, (-1 ≤ x) → (x ≤ y) → (f x a ≤ f y a)) ∧ ¬ (∀ x y : ℝ, (-1 ≤ x) → (x ≤ y) → (f x a ≤ f y a) → (a ≤ -2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l63_63052


namespace product_of_numbers_l63_63394

theorem product_of_numbers (a b : ℕ) (hcf_val lcm_val : ℕ) 
  (h_hcf : Nat.gcd a b = hcf_val) 
  (h_lcm : Nat.lcm a b = lcm_val) 
  (hcf_eq : hcf_val = 33) 
  (lcm_eq : lcm_val = 2574) : 
  a * b = 84942 := 
by
  sorry

end product_of_numbers_l63_63394


namespace total_eggs_l63_63031

theorem total_eggs (eggs_today eggs_yesterday : ℕ) (h_today : eggs_today = 30) (h_yesterday : eggs_yesterday = 19) : eggs_today + eggs_yesterday = 49 :=
by
  sorry

end total_eggs_l63_63031


namespace distinct_real_roots_k_root_condition_k_l63_63316

-- Part (1) condition: The quadratic equation has two distinct real roots
theorem distinct_real_roots_k (k : ℝ) : (∃ x : ℝ, x^2 + 2*x + k = 0) ∧ (∀ x y : ℝ, x^2 + 2*x + k = 0 ∧ y^2 + 2*y + k = 0 → x ≠ y) → k < 1 := 
sorry

-- Part (2) condition: m is a root and satisfies m^2 + 2m = 2
theorem root_condition_k (m k : ℝ) : m^2 + 2*m = 2 → m^2 + 2*m + k = 0 → k = -2 := 
sorry

end distinct_real_roots_k_root_condition_k_l63_63316


namespace student_count_before_new_student_l63_63735

variable {W : ℝ} -- total weight of students before the new student joined
variable {n : ℕ} -- number of students before the new student joined
variable {W_new : ℝ} -- total weight including the new student
variable {n_new : ℕ} -- number of students including the new student

theorem student_count_before_new_student 
  (h1 : W = n * 28) 
  (h2 : W_new = W + 7) 
  (h3 : n_new = n + 1) 
  (h4 : W_new / n_new = 27.3) : n = 29 := 
by
  sorry

end student_count_before_new_student_l63_63735


namespace brick_height_l63_63237

theorem brick_height (H : ℝ) 
    (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
    (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℝ)
    (volume_wall: wall_length = 900 ∧ wall_width = 500 ∧ wall_height = 1850)
    (volume_brick: brick_length = 21 ∧ brick_width = 10)
    (num_bricks_value: num_bricks = 4955.357142857142) :
    (H = 0.8) :=
by {
  sorry
}

end brick_height_l63_63237


namespace binomial_coefficient_example_l63_63136

theorem binomial_coefficient_example :
  2 * (Nat.choose 7 4) = 70 := 
sorry

end binomial_coefficient_example_l63_63136


namespace total_profit_l63_63344

theorem total_profit (A_investment : ℝ) (B_investment : ℝ) (C_investment : ℝ) 
                     (A_months : ℝ) (B_months : ℝ) (C_months : ℝ)
                     (C_share : ℝ) (A_profit_percentage : ℝ) : ℝ :=
  let A_capital_months := A_investment * A_months
  let B_capital_months := B_investment * B_months
  let C_capital_months := C_investment * C_months
  let total_capital_months := A_capital_months + B_capital_months + C_capital_months
  let P := (C_share * total_capital_months) / (C_capital_months * (1 - A_profit_percentage))
  P

example : total_profit 6500 8400 10000 6 5 3 1900 0.05 = 24667 := by
  sorry

end total_profit_l63_63344


namespace Vasek_solved_18_problems_l63_63650

variables (m v z : ℕ)

theorem Vasek_solved_18_problems (h1 : m + v = 25) (h2 : z + v = 32) (h3 : z = 2 * m) : v = 18 := by 
  sorry

end Vasek_solved_18_problems_l63_63650


namespace seq_a_n_100th_term_l63_63315

theorem seq_a_n_100th_term :
  ∃ a : ℕ → ℤ, a 1 = 3 ∧ a 2 = 6 ∧ 
  (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) ∧ 
  a 100 = -3 := 
sorry

end seq_a_n_100th_term_l63_63315


namespace universal_proposition_example_l63_63330

theorem universal_proposition_example :
  (∀ n : ℕ, n % 2 = 0 → ∃ k : ℕ, n = 2 * k) :=
sorry

end universal_proposition_example_l63_63330


namespace jay_change_l63_63489

def cost_book : ℝ := 25
def cost_pen : ℝ := 4
def cost_ruler : ℝ := 1
def payment : ℝ := 50

theorem jay_change : (payment - (cost_book + cost_pen + cost_ruler) = 20) := sorry

end jay_change_l63_63489


namespace abs_eq_self_nonneg_l63_63506

theorem abs_eq_self_nonneg (x : ℝ) : abs x = x ↔ x ≥ 0 :=
sorry

end abs_eq_self_nonneg_l63_63506


namespace area_of_dodecagon_l63_63180

theorem area_of_dodecagon (r : ℝ) : 
  ∃ A : ℝ, (∃ n : ℕ, n = 12) ∧ (A = 3 * r^2) := 
by
  sorry

end area_of_dodecagon_l63_63180


namespace total_viewing_time_l63_63523

theorem total_viewing_time :
  let original_times := [4, 6, 7, 5, 9]
  let new_species_times := [3, 7, 8, 10]
  let total_breaks := 8
  let break_time_per_animal := 2
  let total_time := (original_times.sum + new_species_times.sum) + (total_breaks * break_time_per_animal)
  total_time = 75 :=
by
  sorry

end total_viewing_time_l63_63523


namespace distinct_real_roots_implies_positive_l63_63203

theorem distinct_real_roots_implies_positive (k : ℝ) (x1 x2 : ℝ) (h_distinct : x1 ≠ x2) 
  (h_root1 : x1^2 + 2*x1 - k = 0) 
  (h_root2 : x2^2 + 2*x2 - k = 0) : 
  x1^2 + x2^2 - 2 > 0 := 
sorry

end distinct_real_roots_implies_positive_l63_63203


namespace final_value_after_determinant_and_addition_l63_63477

theorem final_value_after_determinant_and_addition :
  let a := 5
  let b := 7
  let c := 3
  let d := 4
  let det := a * d - b * c
  det + 3 = 2 :=
by
  sorry

end final_value_after_determinant_and_addition_l63_63477


namespace max_min_of_f_l63_63769

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (2 * Real.pi + x) + 
  Real.sqrt 3 * Real.cos (2 * Real.pi - x) -
  Real.sin (2013 * Real.pi + Real.pi / 6)

theorem max_min_of_f : 
  - (Real.pi / 2) ≤ x ∧ x ≤ Real.pi / 2 →
  (-1 / 2) ≤ f x ∧ f x ≤ 5 / 2 :=
sorry

end max_min_of_f_l63_63769


namespace find_f_2017_l63_63932

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (x + 2) = -f x
axiom f_neg1 : f (-1) = -3

theorem find_f_2017 : f 2017 = 3 := 
by
  sorry

end find_f_2017_l63_63932


namespace great_dane_weight_l63_63595

theorem great_dane_weight : 
  ∀ (C P G : ℕ), 
    C + P + G = 439 ∧ P = 3 * C ∧ G = 3 * P + 10 → G = 307 := by
    sorry

end great_dane_weight_l63_63595


namespace sally_jolly_money_sum_l63_63341

/-- Prove the combined amount of money of Sally and Jolly is $150 given the conditions. -/
theorem sally_jolly_money_sum (S J x : ℝ) (h1 : S - x = 80) (h2 : J + 20 = 70) (h3 : S + J = 150) : S + J = 150 :=
by
  sorry

end sally_jolly_money_sum_l63_63341


namespace range_of_m_l63_63322

theorem range_of_m (m x : ℝ) :
  (m-1 < x ∧ x < m+1) → (2 < x ∧ x < 6) → (3 ≤ m ∧ m ≤ 5) :=
by
  intros hp hq
  sorry

end range_of_m_l63_63322


namespace annual_interest_correct_l63_63785

-- Define the conditions
def Rs_total : ℝ := 3400
def P1 : ℝ := 1300
def P2 : ℝ := Rs_total - P1
def Rate1 : ℝ := 0.03
def Rate2 : ℝ := 0.05

-- Define the interests
def Interest1 : ℝ := P1 * Rate1
def Interest2 : ℝ := P2 * Rate2

-- The total interest
def Total_Interest : ℝ := Interest1 + Interest2

-- The theorem to prove
theorem annual_interest_correct :
  Total_Interest = 144 :=
by
  sorry

end annual_interest_correct_l63_63785


namespace infinite_hexagons_exist_l63_63145

theorem infinite_hexagons_exist :
  ∃ (a1 a2 a3 a4 a5 a6 : ℤ), 
  (a1 + a2 + a3 + a4 + a5 + a6 = 20) ∧
  (a1 ≤ a2) ∧ (a1 + a2 ≤ a3) ∧ (a2 + a3 ≤ a4) ∧
  (a3 + a4 ≤ a5) ∧ (a4 + a5 ≤ a6) ∧ (a1 + a2 + a3 + a4 + a5 > a6) :=
sorry

end infinite_hexagons_exist_l63_63145


namespace evaluate_expression_l63_63139

-- Define the greatest power of 2 and 3 that are factors of 360
def a : ℕ := 3 -- 2^3 is the greatest power of 2 that is a factor of 360
def b : ℕ := 2 -- 3^2 is the greatest power of 3 that is a factor of 360

theorem evaluate_expression : (1 / 4)^(b - a) = 4 := 
by 
  have h1 : a = 3 := rfl
  have h2 : b = 2 := rfl
  rw [h1, h2]
  simp
  sorry

end evaluate_expression_l63_63139


namespace parabola_find_m_l63_63758

theorem parabola_find_m
  (p m : ℝ) (h_p_pos : p > 0) (h_point_on_parabola : (2 * p * m) = 8)
  (h_chord_length : (m + (2 / m))^2 - m^2 = 7) : m = (2 * Real.sqrt 3) / 3 :=
by sorry

end parabola_find_m_l63_63758


namespace circle_radius_given_circumference_l63_63172

theorem circle_radius_given_circumference (C : ℝ) (hC : C = 3.14) : ∃ r : ℝ, C = 2 * Real.pi * r ∧ r = 0.5 := 
by
  sorry

end circle_radius_given_circumference_l63_63172


namespace trig_identity_l63_63722

theorem trig_identity (α : ℝ) (h1 : Real.cos α = -4/5) (h2 : π/2 < α ∧ α < π) : 
  - (Real.sin (2 * α) / Real.cos α) = -6/5 :=
by
  sorry

end trig_identity_l63_63722


namespace sum_of_first_n_terms_l63_63293

-- Definitions for the sequences and the problem conditions.
def a (n : ℕ) : ℕ := 2 ^ n
def b (n : ℕ) : ℕ := 2 * n - 1
def c (n : ℕ) : ℕ := a n * b n
def T (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ (n + 1) + 6

-- The theorem statement
theorem sum_of_first_n_terms (n : ℕ) : (Finset.range n).sum c = T n :=
  sorry

end sum_of_first_n_terms_l63_63293


namespace find_K_l63_63056

theorem find_K (Z K : ℕ) (hZ1 : 1000 < Z) (hZ2 : Z < 8000) (hK : Z = K^3) : 11 ≤ K ∧ K ≤ 19 :=
sorry

end find_K_l63_63056


namespace teacher_age_l63_63824

theorem teacher_age {student_count : ℕ} (avg_age_students : ℕ) (avg_age_with_teacher : ℕ)
    (h1 : student_count = 25) (h2 : avg_age_students = 26) (h3 : avg_age_with_teacher = 27) :
    ∃ (teacher_age : ℕ), teacher_age = 52 :=
by
  sorry

end teacher_age_l63_63824


namespace minyoung_money_l63_63897

theorem minyoung_money (A M : ℕ) (h1 : M = 90 * A) (h2 : M = 60 * A + 270) : M = 810 :=
by 
  sorry

end minyoung_money_l63_63897


namespace g_of_36_l63_63227

theorem g_of_36 (g : ℕ → ℕ)
  (h1 : ∀ n, g (n + 1) > g n)
  (h2 : ∀ m n, g (m * n) = g m * g n)
  (h3 : ∀ m n, m ≠ n ∧ m ^ n = n ^ m → (g m = n ∨ g n = m))
  (h4 : ∀ n, g (n ^ 2) = g n * n) :
  g 36 = 36 :=
  sorry

end g_of_36_l63_63227


namespace customer_C_weight_l63_63416

def weights : List ℕ := [22, 25, 28, 31, 34, 36, 38, 40, 45]

-- Definitions for customer A and B such that customer A's total weight equals twice of customer B's total weight
variable {A B : List ℕ}

-- Condition on weights distribution
def valid_distribution (A B : List ℕ) : Prop :=
  (A.sum = 2 * B.sum) ∧ (A ++ B).sum + 38 = 299

-- Prove the weight of the bag received by customer C
theorem customer_C_weight :
  ∃ (C : ℕ), C ∈ weights ∧ C = 38 := by
  sorry

end customer_C_weight_l63_63416


namespace tan_bounds_l63_63368

theorem tan_bounds (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 1) :
    (2 / Real.pi) * (x / (1 - x)) ≤ Real.tan ((Real.pi * x) / 2) ∧
    Real.tan ((Real.pi * x) / 2) ≤ (Real.pi / 2) * (x / (1 - x)) :=
by
    sorry

end tan_bounds_l63_63368


namespace calculate_expression_l63_63483

theorem calculate_expression : (36 / (9 + 2 - 6)) * 4 = 28.8 := 
by
    sorry

end calculate_expression_l63_63483


namespace fish_count_l63_63669

theorem fish_count (total_fish blue_fish blue_spotted_fish : ℕ)
  (h1 : 1 / 3 * total_fish = blue_fish)
  (h2 : 1 / 2 * blue_fish = blue_spotted_fish)
  (h3 : blue_spotted_fish = 10) : total_fish = 60 :=
sorry

end fish_count_l63_63669


namespace verify_expressions_l63_63734

variable (x y : ℝ)
variable (h : x / y = 5 / 3)

theorem verify_expressions :
  (2 * x + y) / y = 13 / 3 ∧
  y / (y - 2 * x) = 3 / -7 ∧
  (x + y) / x = 8 / 5 ∧
  x / (3 * y) = 5 / 9 ∧
  (x - 2 * y) / y = -1 / 3 := by
sorry

end verify_expressions_l63_63734


namespace sequence1_sixth_seventh_terms_sequence2_sixth_term_sequence3_ninth_tenth_terms_l63_63975

def seq1 (n : ℕ) : ℕ := 2 * (n + 1)
def seq2 (n : ℕ) : ℕ := 3 * 2 ^ n
def seq3 (n : ℕ) : ℕ :=
  if n % 2 = 0 then 36 + n
  else 10 + n
  
theorem sequence1_sixth_seventh_terms :
  seq1 5 = 12 ∧ seq1 6 = 14 :=
by
  sorry

theorem sequence2_sixth_term :
  seq2 5 = 96 :=
by
  sorry

theorem sequence3_ninth_tenth_terms :
  seq3 8 = 44 ∧ seq3 9 = 19 :=
by
  sorry

end sequence1_sixth_seventh_terms_sequence2_sixth_term_sequence3_ninth_tenth_terms_l63_63975


namespace arithmetic_sequence_properties_l63_63586

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {n : ℕ}
variable {a1 : ℝ}

theorem arithmetic_sequence_properties 
  (h1 : ∀ n, a n = a1 + (n - 1) * d) 
  (h2 : a1 > 0) 
  (h3 : a 9 + a 10 = a 11) :
  (∀ m n, m < n → a m > a n) ∧ (∀ n, S n = n * (a1 + (d * (n - 1) / 2))) ∧ S 14 > 0 :=
by 
  sorry

end arithmetic_sequence_properties_l63_63586


namespace opponents_team_points_l63_63795

theorem opponents_team_points (M D V O : ℕ) (hM : M = 5) (hD : D = 3) 
    (hV : V = 2 * (M + D)) (hO : O = (M + D + V) + 16) : O = 40 := by
  sorry

end opponents_team_points_l63_63795


namespace maxvalue_on_ellipse_l63_63235

open Real

noncomputable def max_x_plus_y : ℝ := 343 / 88

theorem maxvalue_on_ellipse (x y : ℝ) :
  (x^2 + 3 * x * y + 2 * y^2 - 14 * x - 21 * y + 49 = 0) →
  x + y ≤ max_x_plus_y := 
sorry

end maxvalue_on_ellipse_l63_63235


namespace bus_commutes_three_times_a_week_l63_63935

-- Define the commuting times
def bike_time := 30
def bus_time := bike_time + 10
def friend_time := bike_time * (1 - (2/3))
def total_weekly_time := 160

-- Define the number of times taking the bus as a variable
variable (b : ℕ)

-- The equation for total commuting time
def commuting_time_eq := bike_time + bus_time * b + friend_time = total_weekly_time

-- The proof statement: b should be equal to 3
theorem bus_commutes_three_times_a_week (h : commuting_time_eq b) : b = 3 := sorry

end bus_commutes_three_times_a_week_l63_63935


namespace min_value_l63_63905

/-- Given x and y are positive real numbers such that x + 3y = 2,
    the minimum value of (2x + y) / (xy) is 1/2 * (7 + 2 * sqrt 6). -/
theorem min_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 3 * y = 2) :
  ∃ c : ℝ, c = (1/2) * (7 + 2 * Real.sqrt 6) ∧ ∀ (x y : ℝ), (0 < x) → (0 < y) → (x + 3 * y = 2) → ((2 * x + y) / (x * y)) ≥ c :=
sorry

end min_value_l63_63905


namespace non_neg_sequence_l63_63471

theorem non_neg_sequence (a : ℝ) (x : ℕ → ℝ) (h0 : x 0 = 0)
  (h1 : ∀ n, x (n + 1) = 1 - a * Real.exp (x n)) (ha : a ≤ 1) :
  ∀ n, x n ≥ 0 := 
  sorry

end non_neg_sequence_l63_63471


namespace cost_of_cookies_equal_3_l63_63529

def selling_price : ℝ := 1.5
def cost_price : ℝ := 1
def number_of_bracelets : ℕ := 12
def amount_left : ℝ := 3

theorem cost_of_cookies_equal_3 : 
  (selling_price - cost_price) * number_of_bracelets - amount_left = 3 := by
  sorry

end cost_of_cookies_equal_3_l63_63529


namespace square_side_length_l63_63021

theorem square_side_length (length width : ℕ) (h1 : length = 10) (h2 : width = 5) (cut_across_length : length % 2 = 0) :
  ∃ square_side : ℕ, square_side = 5 := by
  sorry

end square_side_length_l63_63021


namespace Alyssa_next_year_games_l63_63724

theorem Alyssa_next_year_games 
  (games_this_year : ℕ) 
  (games_last_year : ℕ) 
  (total_games : ℕ) 
  (games_up_to_this_year : ℕ)
  (total_up_to_next_year : ℕ) 
  (H1 : games_this_year = 11)
  (H2 : games_last_year = 13)
  (H3 : total_up_to_next_year = 39)
  (H4 : games_up_to_this_year = games_this_year + games_last_year) :
  total_up_to_next_year - games_up_to_this_year = 15 :=
by
  sorry

end Alyssa_next_year_games_l63_63724


namespace min_sum_ab_l63_63985

theorem min_sum_ab (a b : ℤ) (h : a * b = 196) : a + b = -197 :=
sorry

end min_sum_ab_l63_63985


namespace symmetric_circle_eq_l63_63426

theorem symmetric_circle_eq (C_1_eq : ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 1)
    (line_eq : ∀ x y : ℝ, x - y - 2 = 0) :
    ∀ x y : ℝ, (x - 1)^2 + y^2 = 1 :=
sorry

end symmetric_circle_eq_l63_63426


namespace num_triangles_square_even_num_triangles_rect_even_l63_63991

-- Problem (a): Proving that the number of triangles is even 
theorem num_triangles_square_even (a : ℕ) (n : ℕ) (h : a * a = n * (3 * 4 / 2)) : 
  n % 2 = 0 :=
sorry

-- Problem (b): Proving that the number of triangles is even
theorem num_triangles_rect_even (L W k : ℕ) (hL : L = k * 2) (hW : W = k * 1) (h : L * W = k * 1 * 2 / 2) :
  k % 2 = 0 :=
sorry

end num_triangles_square_even_num_triangles_rect_even_l63_63991


namespace intersection_of_sets_l63_63767

def M (x : ℝ) : Prop := (x - 2) / (x - 3) < 0
def N (x : ℝ) : Prop := Real.log (x - 2) / Real.log (1 / 2) ≥ 1 

theorem intersection_of_sets : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 2 < x ∧ x ≤ 5 / 2} := by
  sorry

end intersection_of_sets_l63_63767


namespace triangle_sides_and_angles_l63_63986

theorem triangle_sides_and_angles (a : Real) (α β : Real) :
  (a ≥ 0) →
  let sides := [a, a + 1, a + 2]
  let angles := [α, β, 2 * α]
  (∀ s, s ∈ sides) → (∀ θ, θ ∈ angles) →
  a = 4 ∧ a + 1 = 5 ∧ a + 2 = 6 := 
by {
  sorry
}

end triangle_sides_and_angles_l63_63986


namespace k_satisfies_triangle_condition_l63_63629

theorem k_satisfies_triangle_condition (k : ℤ) 
  (hk_pos : 0 < k) (a b c : ℝ) (ha_pos : 0 < a) 
  (hb_pos : 0 < b) (hc_pos : 0 < c) 
  (h_ineq : (k : ℝ) * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : k = 6 → 
  (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end k_satisfies_triangle_condition_l63_63629


namespace dave_tickets_l63_63548

-- Definitions based on given conditions
def initial_tickets : ℕ := 25
def spent_tickets : ℕ := 22
def additional_tickets : ℕ := 15

-- Proof statement to demonstrate Dave would have 18 tickets
theorem dave_tickets : initial_tickets - spent_tickets + additional_tickets = 18 := by
  sorry

end dave_tickets_l63_63548


namespace ducks_and_geese_difference_l63_63271

variable (d g d' l : ℕ)
variables (hd : d = 25)
variables (hg : g = 2 * d - 10)
variables (hd' : d' = d + 4)
variables (hl : l = 15 - 5)

theorem ducks_and_geese_difference :
  let geese_remain := g - l
  let ducks_remain := d'
  geese_remain - ducks_remain = 1 :=
by
  sorry

end ducks_and_geese_difference_l63_63271


namespace find_PQ_length_l63_63636

-- Define the lengths of the sides of the triangles and the angle
def PQ_length : ℝ := 9
def QR_length : ℝ := 20
def PR_length : ℝ := 15
def ST_length : ℝ := 4.5
def TU_length : ℝ := 7.5
def SU_length : ℝ := 15
def angle_PQR : ℝ := 135
def angle_STU : ℝ := 135

-- Define the similarity condition
def triangles_similar (PQ QR PR ST TU SU angle_PQR angle_STU : ℝ) : Prop :=
  angle_PQR = angle_STU ∧ PQ / QR = ST / TU

-- Theorem statement
theorem find_PQ_length (PQ QR PR ST TU SU angle_PQR angle_STU: ℝ) 
  (H : triangles_similar PQ QR PR ST TU SU angle_PQR angle_STU) : PQ = 20 :=
by
  sorry

end find_PQ_length_l63_63636


namespace reese_spending_l63_63445

-- Definitions used in Lean 4 statement
variable (S : ℝ := 11000)
variable (M : ℝ := 0.4 * S)
variable (A : ℝ := 1500)
variable (L : ℝ := 2900)

-- Lean 4 verification statement
theorem reese_spending :
  ∃ (P : ℝ), S - (P * S + M + A) = L ∧ P * 100 = 20 :=
by
  sorry

end reese_spending_l63_63445


namespace cos_150_eq_neg_half_l63_63039

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l63_63039


namespace area_of_circle_2pi_distance_AB_sqrt6_l63_63385

/- Definition of the circle in polar coordinates -/
def circle_polar := ∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

/- Definition of the line in polar coordinates -/
def line_polar := ∀ θ, ∃ ρ : ℝ, ρ * Real.cos θ - ρ * Real.sin θ + 1 = 0

/- The area of the circle -/
theorem area_of_circle_2pi : 
  (∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) → 
  ∃ A : ℝ, A = 2 * Real.pi :=
by
  intro h
  sorry

/- The distance between two intersection points A and B -/
theorem distance_AB_sqrt6 : 
  (∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) → 
  (∀ θ, ∃ ρ : ℝ, ρ * Real.cos θ - ρ * Real.sin θ + 1 = 0) → 
  ∃ d : ℝ, d = Real.sqrt 6 :=
by
  intros h1 h2
  sorry

end area_of_circle_2pi_distance_AB_sqrt6_l63_63385


namespace johns_new_weekly_earnings_l63_63670

-- Define the original weekly earnings and the percentage increase as given conditions:
def original_weekly_earnings : ℕ := 60
def percentage_increase : ℕ := 50

-- Prove that John's new weekly earnings after the raise is 90 dollars:
theorem johns_new_weekly_earnings : original_weekly_earnings + (percentage_increase * original_weekly_earnings / 100) = 90 := by
sorry

end johns_new_weekly_earnings_l63_63670


namespace mushroom_pickers_at_least_50_l63_63726

-- Given conditions
variables (a : Fin 7 → ℕ) -- Each picker collects a different number of mushrooms.
variables (distinct : ∀ i j, i ≠ j → a i ≠ a j)
variable (total_mushrooms : (Finset.univ.sum a) = 100)

-- The proof that at least three of the pickers collected at least 50 mushrooms together
theorem mushroom_pickers_at_least_50 (a : Fin 7 → ℕ) (distinct : ∀ i j, i ≠ j → a i ≠ a j)
    (total_mushrooms : (Finset.univ.sum a) = 100) :
    ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (a i + a j + a k) ≥ 50 :=
sorry

end mushroom_pickers_at_least_50_l63_63726


namespace pages_per_day_l63_63867

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (result : ℕ) :
  total_pages = 81 ∧ days = 3 → result = 27 :=
by
  sorry

end pages_per_day_l63_63867


namespace inverse_of_square_positive_is_negative_l63_63236

variable {x : ℝ}

-- Original proposition: ∀ x, x < 0 → x^2 > 0
def original_proposition : Prop :=
  ∀ x : ℝ, x < 0 → x^2 > 0

-- Inverse proposition to be proven: ∀ x, x^2 > 0 → x < 0
def inverse_proposition (x : ℝ) : Prop :=
  x^2 > 0 → x < 0

theorem inverse_of_square_positive_is_negative :
  (∀ x : ℝ, x < 0 → x^2 > 0) → (∀ x : ℝ, x^2 > 0 → x < 0) :=
  sorry

end inverse_of_square_positive_is_negative_l63_63236


namespace find_x_l63_63676

theorem find_x (x : ℝ) (h : 0.90 * 600 = 0.50 * x) : x = 1080 :=
sorry

end find_x_l63_63676


namespace sum_of_square_roots_l63_63760

theorem sum_of_square_roots :
  (Real.sqrt 1 + Real.sqrt (1 + 2) + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4)) = 
  (1 + Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 10) := 
sorry

end sum_of_square_roots_l63_63760


namespace marvin_number_is_correct_l63_63415

theorem marvin_number_is_correct (y : ℤ) (h : y - 5 = 95) : y + 5 = 105 := by
  sorry

end marvin_number_is_correct_l63_63415


namespace number_of_terms_in_sequence_l63_63064

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem number_of_terms_in_sequence : 
  ∃ n : ℕ, arithmetic_sequence (-3) 4 n = 53 ∧ n = 15 :=
by
  use 15
  constructor
  · unfold arithmetic_sequence
    norm_num
  · norm_num

end number_of_terms_in_sequence_l63_63064


namespace convex_polygon_obtuse_sum_l63_63411
open Int

def convex_polygon_sides (n : ℕ) (S : ℕ) : Prop :=
  180 * (n - 2) = 3000 + S ∧ (S = 60 ∨ S = 240)

theorem convex_polygon_obtuse_sum (n : ℕ) (hn : 3 ≤ n) :
  (∃ S, convex_polygon_sides n S) ↔ (n = 19 ∨ n = 20) :=
by
  sorry

end convex_polygon_obtuse_sum_l63_63411


namespace perimeter_of_specific_figure_l63_63755

-- Define the grid size and additional column properties as given in the problem
structure Figure :=
  (rows : ℕ)
  (cols : ℕ)
  (additionalCols : ℕ)
  (additionalRows : ℕ)

-- The specific figure properties from the problem statement
def specificFigure : Figure := {
  rows := 3,
  cols := 4,
  additionalCols := 1,
  additionalRows := 2
}

-- Define the perimeter computation
def computePerimeter (fig : Figure) : ℕ :=
  2 * (fig.rows + fig.cols + fig.additionalCols) + fig.additionalRows

theorem perimeter_of_specific_figure : computePerimeter specificFigure = 13 :=
by
  sorry

end perimeter_of_specific_figure_l63_63755


namespace sequence_is_geometric_l63_63107

def is_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → S n = 3 * a n - 3

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) (a₁ : ℝ) : Prop :=
∀ n : ℕ, a n = a₁ * r ^ n

theorem sequence_is_geometric (S : ℕ → ℝ) (a : ℕ → ℝ) :
  is_sequence_sum S a →
  (∃ a₁ : ℝ, ∃ r : ℝ, geometric_sequence a r a₁ ∧ a₁ = 3 / 2 ∧ r = 3 / 2) :=
by
  sorry

end sequence_is_geometric_l63_63107


namespace point_outside_circle_l63_63213

theorem point_outside_circle (a b : ℝ) (h_intersect : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ a*x + b*y = 1) : a^2 + b^2 > 1 :=
by
  sorry

end point_outside_circle_l63_63213


namespace smallest_number_is_correct_largest_number_is_correct_l63_63055

def initial_sequence := "123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960"

def remove_digits (n : ℕ) (s : String) : String := sorry  -- Placeholder function for removing n digits

noncomputable def smallest_number_after_removal (s : String) : String :=
  -- Function to find the smallest number possible after removing digits
  remove_digits 100 s

noncomputable def largest_number_after_removal (s : String) : String :=
  -- Function to find the largest number possible after removing digits
  remove_digits 100 s

theorem smallest_number_is_correct : smallest_number_after_removal initial_sequence = "123450" :=
  sorry

theorem largest_number_is_correct : largest_number_after_removal initial_sequence = "56758596049" :=
  sorry

end smallest_number_is_correct_largest_number_is_correct_l63_63055


namespace min_value_reciprocal_sum_l63_63342

theorem min_value_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) : 
  (1 / x) + (1 / y) ≥ 1 / 3 :=
by
  sorry

end min_value_reciprocal_sum_l63_63342


namespace original_number_is_1212_or_2121_l63_63768

theorem original_number_is_1212_or_2121 (x y z t : ℕ) (h₁ : t ≠ 0)
  (h₂ : 1000 * x + 100 * y + 10 * z + t + 1000 * t + 100 * x + 10 * y + z = 3333) : 
  (1000 * x + 100 * y + 10 * z + t = 1212) ∨ (1000 * x + 100 * y + 10 * z + t = 2121) :=
sorry

end original_number_is_1212_or_2121_l63_63768


namespace counterexample_exists_l63_63462

-- Define prime predicate
def is_prime (n : ℕ) : Prop :=
∀ m, m ∣ n → m = 1 ∨ m = n

def counterexample_to_statement (n : ℕ) : Prop :=
  is_prime n ∧ ¬ is_prime (n + 2)

theorem counterexample_exists : ∃ n ∈ [3, 5, 11, 17, 23], is_prime n ∧ ¬ is_prime (n + 2) :=
by
  sorry

end counterexample_exists_l63_63462


namespace kevin_sold_13_crates_of_grapes_l63_63822

-- Define the conditions
def total_crates : ℕ := 50
def crates_of_mangoes : ℕ := 20
def crates_of_passion_fruits : ℕ := 17

-- Define the question and expected answer
def crates_of_grapes : ℕ := total_crates - (crates_of_mangoes + crates_of_passion_fruits)

-- Prove that the crates of grapes equals to 13
theorem kevin_sold_13_crates_of_grapes :
  crates_of_grapes = 13 :=
by
  -- The proof steps are omitted as per instructions
  sorry

end kevin_sold_13_crates_of_grapes_l63_63822


namespace cost_of_lamp_and_flashlight_max_desk_lamps_l63_63551

-- Part 1: Cost of purchasing one desk lamp and one flashlight
theorem cost_of_lamp_and_flashlight (x : ℕ) (desk_lamp_cost flashlight_cost : ℕ) 
        (hx : desk_lamp_cost = x + 20)
        (hdesk : 400 = x / 2 * desk_lamp_cost)
        (hflash : 160 = x * flashlight_cost)
        (hnum : desk_lamp_cost = 2 * flashlight_cost) : 
        desk_lamp_cost = 25 ∧ flashlight_cost = 5 :=
sorry

-- Part 2: Maximum number of desk lamps Rongqing Company can purchase
theorem max_desk_lamps (a : ℕ) (desk_lamp_cost flashlight_cost : ℕ)
        (hc1 : desk_lamp_cost = 25)
        (hc2 : flashlight_cost = 5)
        (free_flashlight : ℕ := a) (required_flashlight : ℕ := 2 * a + 8) 
        (total_cost : ℕ := desk_lamp_cost * a + flashlight_cost * required_flashlight)
        (hcost : total_cost ≤ 670) :
        a ≤ 21 :=
sorry

end cost_of_lamp_and_flashlight_max_desk_lamps_l63_63551


namespace average_marks_l63_63813

variable (P C M : ℕ)

theorem average_marks :
  P = 140 →
  (P + M) / 2 = 90 →
  (P + C) / 2 = 70 →
  (P + C + M) / 3 = 60 :=
by
  intros hP hM hC
  sorry

end average_marks_l63_63813


namespace budget_allocations_and_percentage_changes_l63_63388

theorem budget_allocations_and_percentage_changes (X : ℝ) :
  (14 * X / 100, 24 * X / 100, 15 * X / 100, 19 * X / 100, 8 * X / 100, 20 * X / 100) = 
  (0.14 * X, 0.24 * X, 0.15 * X, 0.19 * X, 0.08 * X, 0.20 * X) ∧
  ((14 - 12) / 12 * 100 = 16.67 ∧
   (24 - 22) / 22 * 100 = 9.09 ∧
   (15 - 13) / 13 * 100 = 15.38 ∧
   (19 - 18) / 18 * 100 = 5.56 ∧
   (8 - 7) / 7 * 100 = 14.29 ∧
   ((20 - (100 - (12 + 22 + 13 + 18 + 7))) / (100 - (12 + 22 + 13 + 18 + 7)) * 100) = -28.57) := by
  sorry

end budget_allocations_and_percentage_changes_l63_63388


namespace stamps_difference_l63_63208

theorem stamps_difference (x : ℕ) (h1: 5 * x / 3 * x = 5 / 3)
(h2: (5 * x - 12) / (3 * x + 12) = 4 / 3) : 
(5 * x - 12) - (3 * x + 12) = 32 := by
sorry

end stamps_difference_l63_63208


namespace vegetable_planting_methods_l63_63871

theorem vegetable_planting_methods :
  let vegetables := ["cucumber", "cabbage", "rape", "lentils"]
  let cucumber := "cucumber"
  let other_vegetables := ["cabbage", "rape", "lentils"]
  let choose_2_out_of_3 := Nat.choose 3 2
  let arrangements := Nat.factorial 3
  total_methods = choose_2_out_of_3 * arrangements := by
  let total_methods := 3 * 6
  sorry

end vegetable_planting_methods_l63_63871


namespace train_length_calculation_l63_63616

theorem train_length_calculation
  (speed_kmph : ℝ)
  (time_seconds : ℝ)
  (train_length : ℝ)
  (h1 : speed_kmph = 80)
  (h2 : time_seconds = 8.999280057595392)
  (h3 : train_length = (80 * 1000) / 3600 * 8.999280057595392) :
  train_length = 200 := by
  sorry

end train_length_calculation_l63_63616


namespace moles_of_KI_formed_l63_63094

-- Define the given conditions
def moles_KOH : ℕ := 1
def moles_NH4I : ℕ := 1
def balanced_equation (KOH NH4I KI NH3 H2O : ℕ) : Prop :=
  (KOH = 1) ∧ (NH4I = 1) ∧ (KI = 1) ∧ (NH3 = 1) ∧ (H2O = 1)

-- The proof problem statement
theorem moles_of_KI_formed (h : balanced_equation moles_KOH moles_NH4I 1 1 1) : 
  1 = 1 :=
by sorry

end moles_of_KI_formed_l63_63094


namespace symmetric_point_coordinates_l63_63882

theorem symmetric_point_coordinates 
  (k : ℝ) 
  (P : ℝ × ℝ) 
  (h1 : ∀ k, k * (P.1) - P.2 + k - 2 = 0) 
  (P' : ℝ × ℝ) 
  (h2 : P'.1 + P'.2 = 3) 
  (h3 : 2 * P'.1^2 + 2 * P'.2^2 + 4 * P'.1 + 8 * P'.2 + 5 = 0) 
  (hP : P = (-1, -2)): 
  P' = (2, 1) := 
sorry

end symmetric_point_coordinates_l63_63882


namespace product_of_number_and_its_digits_sum_l63_63886

theorem product_of_number_and_its_digits_sum :
  ∃ (n : ℕ), (n = 24 ∧ (n % 10) = ((n / 10) % 10) + 2) ∧ (n * (n % 10 + (n / 10) % 10) = 144) :=
by
  sorry

end product_of_number_and_its_digits_sum_l63_63886


namespace sugar_used_in_two_minutes_l63_63048

-- Definitions according to conditions
def sugar_per_bar : ℝ := 1.5
def bars_per_minute : ℝ := 36
def minutes : ℝ := 2

-- Theorem statement
theorem sugar_used_in_two_minutes : bars_per_minute * sugar_per_bar * minutes = 108 :=
by
  -- We add sorry here to complete the proof later.
  sorry

end sugar_used_in_two_minutes_l63_63048


namespace arithmetic_sequence_m_l63_63218

theorem arithmetic_sequence_m (m : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, a n = 2 * n - 1) →
  (∀ n, S n = n * (2 * n - 1) / 2) →
  S m = (a m + a (m + 1)) / 2 →
  m = 2 :=
by
  sorry

end arithmetic_sequence_m_l63_63218


namespace work_completion_days_l63_63563

theorem work_completion_days (D_a : ℝ) (R_a R_b : ℝ)
  (h1 : R_a = 1 / D_a)
  (h2 : R_b = 1 / (1.5 * D_a))
  (h3 : R_a = 1.5 * R_b)
  (h4 : 1 / 18 = R_a + R_b) : D_a = 30 := 
by
  sorry

end work_completion_days_l63_63563


namespace find_a_l63_63541

theorem find_a (f : ℤ → ℤ) (h1 : ∀ (x : ℤ), f (2 * x + 1) = 3 * x + 2) (h2 : f a = 2) : a = 1 := by
sorry

end find_a_l63_63541


namespace sum_of_roots_l63_63547

variables {a b c : ℝ}

-- Conditions
-- The polynomial with roots a, b, c
def poly (x : ℝ) : ℝ := 24 * x^3 - 36 * x^2 + 14 * x - 1

-- The roots are in (0, 1)
def in_interval (x : ℝ) : Prop := 0 < x ∧ x < 1

-- All roots are distinct
def distinct (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

-- Main Theorem
theorem sum_of_roots :
  (∀ x, poly x = 0 → x = a ∨ x = b ∨ x = c) →
  in_interval a →
  in_interval b →
  in_interval c →
  distinct a b c →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2) :=
by
  intros
  sorry

end sum_of_roots_l63_63547


namespace side_length_irrational_l63_63781

theorem side_length_irrational (s : ℝ) (h : s^2 = 3) : ¬∃ (r : ℚ), s = r := by
  sorry

end side_length_irrational_l63_63781


namespace determine_h_l63_63284

theorem determine_h (x : ℝ) (h : ℝ → ℝ) :
  2 * x ^ 5 + 4 * x ^ 3 + h x = 7 * x ^ 3 - 5 * x ^ 2 + 9 * x + 3 →
  h x = -2 * x ^ 5 + 3 * x ^ 3 - 5 * x ^ 2 + 9 * x + 3 :=
by
  intro h_eq
  sorry

end determine_h_l63_63284


namespace total_cookies_eaten_l63_63815

theorem total_cookies_eaten :
  let charlie := 15
  let father := 10
  let mother := 5
  let grandmother := 12 / 2
  let dog := 3 * 0.75
  charlie + father + mother + grandmother + dog = 38.25 :=
by
  sorry

end total_cookies_eaten_l63_63815


namespace prism_width_l63_63956

/-- A rectangular prism with dimensions l, w, h such that the diagonal length is 13
    and given l = 3 and h = 12, has width w = 4. -/
theorem prism_width (w : ℕ) 
  (h : ℕ) (l : ℕ) 
  (diag_len : ℕ) 
  (hl : l = 3) 
  (hh : h = 12) 
  (hd : diag_len = 13) 
  (h_diag : diag_len = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := 
  sorry

end prism_width_l63_63956


namespace problem1_solution_problem2_solution_l63_63029

-- Conditions for Problem 1
def problem1_condition (x : ℝ) : Prop := 
  5 * (x - 20) + 2 * x = 600

-- Proof for Problem 1 Goal
theorem problem1_solution (x : ℝ) (h : problem1_condition x) : x = 100 := 
by sorry

-- Conditions for Problem 2
def problem2_condition (m : ℝ) : Prop :=
  (360 / m) + (540 / (1.2 * m)) = (900 / 100)

-- Proof for Problem 2 Goal
theorem problem2_solution (m : ℝ) (h : problem2_condition m) : m = 90 := 
by sorry

end problem1_solution_problem2_solution_l63_63029


namespace find_c_deg3_l63_63564

-- Define the polynomials f and g.
def f (x : ℚ) : ℚ := 2 - 10 * x + 4 * x^2 - 5 * x^3 + 7 * x^4
def g (x : ℚ) : ℚ := 5 - 3 * x - 8 * x^3 + 11 * x^4

-- The statement that needs proof.
theorem find_c_deg3 (c : ℚ) : (∀ x : ℚ, f x + c * g x ≠ 0 → f x + c * g x = 2 - 10 * x + 4 * x^2 - 5 * x^3 - c * 8 * x^3) ↔ c = -7 / 11 :=
sorry

end find_c_deg3_l63_63564


namespace remainder_of_sum_of_primes_mod_eighth_prime_l63_63952

def sum_first_seven_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13 + 17

def eighth_prime : ℕ := 19

theorem remainder_of_sum_of_primes_mod_eighth_prime : sum_first_seven_primes % eighth_prime = 1 := by
  sorry

end remainder_of_sum_of_primes_mod_eighth_prime_l63_63952


namespace number_of_persons_in_second_group_l63_63863

-- Definitions based on conditions
def total_man_hours_first_group : ℕ := 42 * 12 * 5

def total_man_hours_second_group (X : ℕ) : ℕ := X * 14 * 6

-- Theorem stating that the number of persons in the second group is 30, given the conditions
theorem number_of_persons_in_second_group (X : ℕ) : 
  total_man_hours_first_group = total_man_hours_second_group X → X = 30 :=
by
  sorry

end number_of_persons_in_second_group_l63_63863


namespace equivalent_operation_l63_63023

theorem equivalent_operation (x : ℚ) : 
  (x * (5 / 6) / (2 / 7)) = x * (35 / 12) :=
by
  sorry

end equivalent_operation_l63_63023


namespace inradius_inequality_l63_63787

/-- Given a point P inside the triangle ABC, where da, db, and dc are the distances from P to the sides BC, CA, and AB respectively,
 and r is the inradius of the triangle ABC, prove the inequality -/
theorem inradius_inequality (a b c da db dc : ℝ) (r : ℝ) 
  (h1 : 0 < da) (h2 : 0 < db) (h3 : 0 < dc)
  (h4 : r = (a * da + b * db + c * dc) / (a + b + c)) :
  2 / (1 / da + 1 / db + 1 / dc) < r ∧ r < (da + db + dc) / 2 :=
  sorry

end inradius_inequality_l63_63787


namespace exp_decreasing_range_l63_63204

theorem exp_decreasing_range (a : ℝ) :
  (∀ x : ℝ, (a-2) ^ x < (a-2) ^ (x - 1)) → 2 < a ∧ a < 3 :=
by
  sorry

end exp_decreasing_range_l63_63204


namespace toys_produced_each_day_l63_63521

theorem toys_produced_each_day (weekly_production : ℕ) (days_worked : ℕ) 
  (h1 : weekly_production = 5500) (h2 : days_worked = 4) : 
  (weekly_production / days_worked = 1375) :=
sorry

end toys_produced_each_day_l63_63521


namespace total_paintable_area_correct_l63_63602

-- Define the conditions
def warehouse_width := 12
def warehouse_length := 15
def warehouse_height := 7

def window_count_per_longer_wall := 3
def window_width := 2
def window_height := 3

-- Define areas for walls, ceiling, and floor
def area_wall_1 := warehouse_width * warehouse_height
def area_wall_2 := warehouse_length * warehouse_height
def window_area := window_width * window_height
def window_total_area := window_count_per_longer_wall * window_area
def area_wall_2_paintable := 2 * (area_wall_2 - window_total_area) -- both inside and outside
def area_ceiling := warehouse_width * warehouse_length
def area_floor := warehouse_width * warehouse_length

-- Total paintable area calculation
def total_paintable_area := 2 * area_wall_1 + area_wall_2_paintable + area_ceiling + area_floor

-- Final proof statement
theorem total_paintable_area_correct : total_paintable_area = 876 := by
  sorry

end total_paintable_area_correct_l63_63602


namespace speedster_convertibles_approx_l63_63500

-- Definitions corresponding to conditions
def total_inventory : ℕ := 120
def num_non_speedsters : ℕ := 40
def num_speedsters : ℕ := 2 * total_inventory / 3
def num_speedster_convertibles : ℕ := 64

-- Theorem statement
theorem speedster_convertibles_approx :
  2 * total_inventory / 3 - num_non_speedsters + num_speedster_convertibles = total_inventory :=
sorry

end speedster_convertibles_approx_l63_63500


namespace coordinates_of_point_P_l63_63428

theorem coordinates_of_point_P 
  (x y : ℝ)
  (h1 : y = x^3 - x)
  (h2 : (3 * x^2 - 1) = 2)
  (h3 : ∀ x y, x + 2 * y = 0 → ∃ m, -1/(m) = 2) :
  (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
by
  sorry

end coordinates_of_point_P_l63_63428


namespace min_distance_from_origin_l63_63721

-- Define the condition of the problem
def condition (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + 6 * y + 4 = 0

-- Statement of the problem in Lean 4
theorem min_distance_from_origin (x y : ℝ) (h : condition x y) : 
  ∃ m : ℝ, m = Real.sqrt (x^2 + y^2) ∧ m = Real.sqrt 13 - 3 := 
sorry

end min_distance_from_origin_l63_63721


namespace positive_difference_between_median_and_mode_l63_63058

-- Definition of the data as provided in the stem and leaf plot
def data : List ℕ := [
  21, 21, 21, 24, 25, 25,
  33, 33, 36, 37,
  40, 43, 44, 47, 49, 49,
  52, 56, 56, 58, 
  59, 59, 60, 63
]

-- Definition of mode and median calculations
def mode (l : List ℕ) : ℕ := 49  -- As determined, 49 is the mode
def median (l : List ℕ) : ℚ := (43 + 44) / 2  -- Median determined from the sorted list

-- The main theorem to prove
theorem positive_difference_between_median_and_mode (l : List ℕ) :
  abs (median l - mode l) = 5.5 := by
  sorry

end positive_difference_between_median_and_mode_l63_63058


namespace multiples_of_4_l63_63648

theorem multiples_of_4 (n : ℕ) (h : n + 23 * 4 = 112) : n = 20 :=
by
  sorry

end multiples_of_4_l63_63648


namespace base3_addition_l63_63118

theorem base3_addition :
  (2 + 1 * 3 + 2 * 9 + 1 * 27 + 2 * 81) + (1 + 1 * 3 + 2 * 9 + 2 * 27) + (2 * 9 + 1 * 27 + 0 * 81 + 2 * 243) + (1 + 1 * 3 + 1 * 9 + 2 * 27 + 2 * 81) = 
  2 + 1 * 3 + 1 * 9 + 2 * 27 + 2 * 81 + 1 * 243 + 1 * 729 := sorry

end base3_addition_l63_63118


namespace function_matches_table_values_l63_63296

variable (f : ℤ → ℤ)

theorem function_matches_table_values (h1 : f (-1) = -2) (h2 : f 0 = 0) (h3 : f 1 = 2) (h4 : f 2 = 4) : 
  ∀ x : ℤ, f x = 2 * x := 
by
  -- Prove that the function satisfying the given table values is f(x) = 2x
  sorry

end function_matches_table_values_l63_63296


namespace water_speed_l63_63443

theorem water_speed (swim_speed : ℝ) (time : ℝ) (distance : ℝ) (v : ℝ) 
  (h1: swim_speed = 10) (h2: time = 2) (h3: distance = 12) 
  (h4: distance = (swim_speed - v) * time) : 
  v = 4 :=
by
  sorry

end water_speed_l63_63443


namespace greatest_possible_positive_integer_difference_l63_63532

theorem greatest_possible_positive_integer_difference (x y : ℤ) (hx : 4 < x) (hx' : x < 6) (hy : 6 < y) (hy' : y < 10) :
  y - x = 4 :=
sorry

end greatest_possible_positive_integer_difference_l63_63532


namespace calculate_neg4_mul_three_div_two_l63_63874

theorem calculate_neg4_mul_three_div_two : (-4) * (3 / 2) = -6 := 
by
  sorry

end calculate_neg4_mul_three_div_two_l63_63874


namespace product_of_m_and_u_l63_63152

noncomputable def g : ℝ → ℝ := sorry

axiom g_conditions : (∀ x y : ℝ, g (x^2 - y^2) = (x - y) * ((g x) ^ 3 + (g y) ^ 3)) ∧ (g 1 = 1)

def m : ℕ := sorry
def u : ℝ := sorry

theorem product_of_m_and_u : m * u = 3 :=
by 
  -- all conditions about 'g' are assumed as axioms and not directly included in the proof steps
  exact sorry

end product_of_m_and_u_l63_63152


namespace standard_eq_minimal_circle_l63_63837

-- Definitions
variables {x y : ℝ}
variables (h₀ : 0 < x) (h₁ : 0 < y)
variables (h₂ : 3 / (2 + x) + 3 / (2 + y) = 1)

-- Theorem statement
theorem standard_eq_minimal_circle : (x - 4)^2 + (y - 4)^2 = 16^2 :=
sorry

end standard_eq_minimal_circle_l63_63837


namespace gcd_of_sum_and_fraction_l63_63424

theorem gcd_of_sum_and_fraction (p : ℕ) (a b : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1)
  (hcoprime : Nat.gcd a b = 1) : Nat.gcd (a + b) ((a^p + b^p) / (a + b)) = p := 
sorry

end gcd_of_sum_and_fraction_l63_63424


namespace cos_270_eq_zero_l63_63671

theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 := by
  sorry

end cos_270_eq_zero_l63_63671


namespace arithmetic_sequence_solution_l63_63618

theorem arithmetic_sequence_solution (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 + a 4 = 4)
  (h2 : a 2 * a 3 = 3)
  (hS : ∀ n, S n = n * (a 1 + a n) / 2):
  (a 1 = -1 ∧ (∀ n, a n = 2 * n - 3) ∧ (∀ n, S n = n^2 - 2 * n)) ∨ 
  (a 1 = 5 ∧ (∀ n, a n = 7 - 2 * n) ∧ (∀ n, S n = 6 * n - n^2)) :=
sorry

end arithmetic_sequence_solution_l63_63618


namespace second_car_distance_l63_63436

theorem second_car_distance (x : ℝ) : 
  let d_initial : ℝ := 150
  let d_first_car_initial : ℝ := 25
  let d_right_turn : ℝ := 15
  let d_left_turn : ℝ := 25
  let d_final_gap : ℝ := 65
  (d_initial - x = d_final_gap) → x = 85 := by
  sorry

end second_car_distance_l63_63436


namespace weavers_problem_l63_63351

theorem weavers_problem 
  (W : ℕ) 
  (H1 : 1 = W / 4) 
  (H2 : 3.5 = 49 / 14) :
  W = 4 :=
by
  sorry

end weavers_problem_l63_63351


namespace convert_base4_to_base10_l63_63826

-- Define a function to convert a base 4 number to base 10
def base4_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0

-- Assert the proof problem
theorem convert_base4_to_base10 : base4_to_base10 3201 = 225 :=
by
  -- The proof script goes here; for now, we use 'sorry' as a placeholder
  sorry

end convert_base4_to_base10_l63_63826


namespace trailing_zeros_30_factorial_l63_63511

-- Definitions directly from conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def trailing_zeros (n : ℕ) : ℕ :=
  let count_five_factors (k : ℕ) : ℕ :=
    k / 5 + k / 25 + k / 125 -- This generalizes for higher powers of 5 which is sufficient here.
  count_five_factors n

-- Mathematical proof problem statement
theorem trailing_zeros_30_factorial : trailing_zeros 30 = 7 := by
  sorry

end trailing_zeros_30_factorial_l63_63511


namespace amount_paid_l63_63993

theorem amount_paid (cost_price : ℝ) (percent_more : ℝ) (h1 : cost_price = 6525) (h2 : percent_more = 0.24) : 
  cost_price + percent_more * cost_price = 8091 :=
by 
  -- Proof here
  sorry

end amount_paid_l63_63993


namespace book_price_range_l63_63410

variable (x : ℝ) -- Assuming x is a real number

theorem book_price_range 
    (hA : ¬(x ≥ 20)) 
    (hB : ¬(x ≤ 15)) : 
    15 < x ∧ x < 20 := 
by
  sorry

end book_price_range_l63_63410


namespace find_x_l63_63384

theorem find_x (x : ℕ) : (x % 6 = 0) ∧ (x^2 > 200) ∧ (x < 30) → (x = 18 ∨ x = 24) :=
by
  intros
  sorry

end find_x_l63_63384


namespace rhombus_area_l63_63589

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 20) : 
  (d1 * d2) / 2 = 160 := by
sorry

end rhombus_area_l63_63589


namespace Angie_age_ratio_l63_63109

-- Define Angie's age as a variable
variables (A : ℕ)

-- Give the condition
def Angie_age_condition := A + 4 = 20

-- State the theorem to be proved
theorem Angie_age_ratio (h : Angie_age_condition A) : (A : ℚ) / (A + 4) = 4 / 5 := 
sorry

end Angie_age_ratio_l63_63109


namespace fraction_of_students_with_buddy_l63_63645

variables (f e : ℕ)
-- Given:
axiom H1 : e / 4 = f / 3

-- Prove:
theorem fraction_of_students_with_buddy : 
  (e / 4 + f / 3) / (e + f) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_buddy_l63_63645


namespace change_in_mean_l63_63940

theorem change_in_mean {a b c d : ℝ} 
  (h1 : (a + b + c + d) / 4 = 10)
  (h2 : (b + c + d) / 3 = 11)
  (h3 : (a + c + d) / 3 = 12)
  (h4 : (a + b + d) / 3 = 13) : 
  ((a + b + c) / 3) = 4 := by 
  sorry

end change_in_mean_l63_63940


namespace area_of_shaded_region_l63_63084

theorem area_of_shaded_region (r R : ℝ) (π : ℝ) (h1 : R = 3 * r) (h2 : 2 * r = 6) : 
  (π * R^2) - (π * r^2) = 72 * π :=
by
  sorry

end area_of_shaded_region_l63_63084


namespace units_digit_17_pow_2107_l63_63910

theorem units_digit_17_pow_2107 : (17 ^ 2107) % 10 = 3 := by
  -- Definitions derived from conditions:
  -- 1. Powers of 17 have the same units digit as the corresponding powers of 7.
  -- 2. Units digits of powers of 7 cycle: 7, 9, 3, 1.
  -- 3. 2107 modulo 4 gives remainder 3.
  sorry

end units_digit_17_pow_2107_l63_63910


namespace investment_amounts_proof_l63_63708

noncomputable def investment_proof_statement : Prop :=
  let p_investment_first_year := 52000
  let q_investment := (5/4) * p_investment_first_year
  let r_investment := (6/4) * p_investment_first_year;
  let p_investment_second_year := p_investment_first_year + (20/100) * p_investment_first_year;
  (q_investment = 65000) ∧ (r_investment = 78000) ∧ (q_investment = 65000) ∧ (r_investment = 78000)

theorem investment_amounts_proof : investment_proof_statement :=
  by
    sorry

end investment_amounts_proof_l63_63708


namespace acrobats_count_l63_63339

theorem acrobats_count
  (a e c : ℕ)
  (h1 : 2 * a + 4 * e + 2 * c = 58)
  (h2 : a + e + c = 25) :
  a = 11 :=
by
  -- Proof skipped
  sorry

end acrobats_count_l63_63339


namespace simplify_and_evaluate_expression_l63_63350

-- Definitions of the variables and their values
def x : ℤ := -2
def y : ℚ := 1 / 2

-- Theorem statement
theorem simplify_and_evaluate_expression : 
  2 * (x^2 * y + x * y^2) - 2 * (x^2 * y - 1) - 3 * x * y^2 - 2 = 
  (1 : ℚ) / 2 :=
by
  -- Proof goes here
  sorry

end simplify_and_evaluate_expression_l63_63350


namespace cube_volume_is_27_l63_63300

noncomputable def original_volume (s : ℝ) : ℝ := s^3
noncomputable def new_solid_volume (s : ℝ) : ℝ := (s + 2) * (s + 2) * (s - 2)

theorem cube_volume_is_27 (s : ℝ) (h : original_volume s - new_solid_volume s = 10) :
  original_volume s = 27 :=
by
  sorry

end cube_volume_is_27_l63_63300


namespace average_weight_14_children_l63_63832

theorem average_weight_14_children 
  (average_weight_boys : ℕ → ℤ → ℤ)
  (average_weight_girls : ℕ → ℤ → ℤ)
  (total_children : ℕ)
  (total_weight : ℤ)
  (total_average_weight : ℤ)
  (boys_count : ℕ)
  (girls_count : ℕ)
  (boys_average : ℤ)
  (girls_average : ℤ) :
  boys_count = 8 →
  girls_count = 6 →
  boys_average = 160 →
  girls_average = 130 →
  total_children = boys_count + girls_count →
  total_weight = average_weight_boys boys_count boys_average + average_weight_girls girls_count girls_average →
  average_weight_boys boys_count boys_average = boys_count * boys_average →
  average_weight_girls girls_count girls_average = girls_count * girls_average →
  total_average_weight = total_weight / total_children →
  total_average_weight = 147 :=
by
  sorry

end average_weight_14_children_l63_63832


namespace single_elimination_games_needed_l63_63552

theorem single_elimination_games_needed (teams : ℕ) (h : teams = 19) : 
∃ games, games = 18 ∧ (∀ (teams_left : ℕ), teams_left = teams - 1 → games = teams - 1) :=
by
  -- define the necessary parameters and properties here 
  sorry

end single_elimination_games_needed_l63_63552


namespace inequality_abc_l63_63965

open Real

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / sqrt (a^2 + 8 * b * c)) + (b / sqrt (b^2 + 8 * c * a)) + (c / sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_abc_l63_63965


namespace bernoulli_inequality_l63_63723

theorem bernoulli_inequality (n : ℕ) (h : 1 ≤ n) (x : ℝ) (h1 : x > -1) : (1 + x) ^ n ≥ 1 + n * x := 
sorry

end bernoulli_inequality_l63_63723


namespace total_visitors_l63_63304

noncomputable def visitors_questionnaire (V E U : ℕ) : Prop :=
  (130 ≠ E ∧ E ≠ U) ∧ 
  (E = U) ∧ 
  (3 * V = 4 * E) ∧ 
  (V = 130 + 3 / 4 * V)

theorem total_visitors (V : ℕ) : visitors_questionnaire V V V → V = 520 :=
by sorry

end total_visitors_l63_63304


namespace hiking_rate_up_the_hill_l63_63682

theorem hiking_rate_up_the_hill (r_down : ℝ) (t_total : ℝ) (t_up : ℝ) (r_up : ℝ) :
  r_down = 6 ∧ t_total = 3 ∧ t_up = 1.2 → r_up * t_up = 9 * t_up :=
by
  intro h
  let ⟨hrd, htt, htu⟩ := h
  sorry

end hiking_rate_up_the_hill_l63_63682


namespace scientific_notation_21500000_l63_63642

/-- Express the number 21500000 in scientific notation. -/
theorem scientific_notation_21500000 : 21500000 = 2.15 * 10^7 := 
sorry

end scientific_notation_21500000_l63_63642


namespace cricket_innings_l63_63925

theorem cricket_innings (n : ℕ) 
  (average_run : ℕ := 40) 
  (next_innings_run : ℕ := 84) 
  (new_average_run : ℕ := 44) :
  (40 * n + 84) / (n + 1) = 44 ↔ n = 10 := 
by
  sorry

end cricket_innings_l63_63925


namespace capsules_per_bottle_l63_63006

-- Translating conditions into Lean definitions
def days := 180
def daily_serving_size := 2
def total_bottles := 6
def total_capsules_required := days * daily_serving_size

-- The statement to prove
theorem capsules_per_bottle : total_capsules_required / total_bottles = 60 :=
by
  sorry

end capsules_per_bottle_l63_63006


namespace radius_of_intersection_l63_63789

noncomputable def sphere_radius := 2 * Real.sqrt 17

theorem radius_of_intersection (s : ℝ) 
  (h1 : (3:ℝ)=(3:ℝ)) (h2 : (5:ℝ)=(5:ℝ)) (h3 : (0-3:ℝ)^2 + (5-5:ℝ)^2 + (s-(-8+8))^2 = sphere_radius^2) :
  s = Real.sqrt 59 :=
by
  sorry

end radius_of_intersection_l63_63789


namespace reciprocal_of_2023_l63_63570

theorem reciprocal_of_2023 : 1 / 2023 = (1 : ℚ) / 2023 :=
by sorry

end reciprocal_of_2023_l63_63570


namespace seq_geq_4_l63_63893

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = (a n ^ 2 + 8 * a n + 16) / (4 * a n)

theorem seq_geq_4 (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, n ≥ 1 → a n ≥ 4 :=
sorry

end seq_geq_4_l63_63893


namespace initial_ratio_zinc_copper_l63_63583

theorem initial_ratio_zinc_copper (Z C : ℝ) 
  (h1 : Z + C = 6) 
  (h2 : Z + 8 = 3 * C) : 
  Z / C = 5 / 7 := 
sorry

end initial_ratio_zinc_copper_l63_63583


namespace friends_playing_video_game_l63_63699

def total_lives : ℕ := 64
def lives_per_player : ℕ := 8

theorem friends_playing_video_game (num_friends : ℕ) :
  num_friends = total_lives / lives_per_player :=
sorry

end friends_playing_video_game_l63_63699


namespace smallest_fraction_l63_63683

theorem smallest_fraction (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100) (eqn : y^2 - 1 = a^2 * (x^2 - 1)) :
  2 ≤ a / x :=
sorry

end smallest_fraction_l63_63683


namespace Albert_more_rocks_than_Joshua_l63_63110

-- Definitions based on the conditions
def Joshua_rocks : ℕ := 80
def Jose_rocks : ℕ := Joshua_rocks - 14
def Albert_rocks : ℕ := Jose_rocks + 20

-- Statement to prove
theorem Albert_more_rocks_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end Albert_more_rocks_than_Joshua_l63_63110


namespace regression_line_intercept_l63_63527

theorem regression_line_intercept
  (x : ℕ → ℝ)
  (y : ℕ → ℝ)
  (h_x_sum : x 1 + x 2 + x 3 + x 4 + x 5 + x 6 = 10)
  (h_y_sum : y 1 + y 2 + y 3 + y 4 + y 5 + y 6 = 4) :
  ∃ a : ℝ, (∀ i, y i = (1 / 4) * x i + a) → a = 1 / 4 :=
by
  sorry

end regression_line_intercept_l63_63527


namespace area_within_fence_is_328_l63_63114

-- Define the dimensions of the fenced area
def main_rectangle_length : ℝ := 20
def main_rectangle_width : ℝ := 18

-- Define the dimensions of the square cutouts
def cutout_length : ℝ := 4
def cutout_width : ℝ := 4

-- Calculate the areas
def main_rectangle_area : ℝ := main_rectangle_length * main_rectangle_width
def cutout_area : ℝ := cutout_length * cutout_width

-- Define the number of cutouts
def number_of_cutouts : ℝ := 2

-- Calculate the final area within the fence
def area_within_fence : ℝ := main_rectangle_area - number_of_cutouts * cutout_area

theorem area_within_fence_is_328 : area_within_fence = 328 := by
  -- This is a place holder for the proof, replace it with the actual proof
  sorry

end area_within_fence_is_328_l63_63114


namespace tank_fraction_before_gas_added_l63_63626

theorem tank_fraction_before_gas_added (capacity : ℝ) (added_gasoline : ℝ) (fraction_after : ℝ) (initial_fraction : ℝ) :
  capacity = 42 → added_gasoline = 7 → fraction_after = 9 / 10 → (initial_fraction * capacity + added_gasoline = fraction_after * capacity) → initial_fraction = 733 / 1000 :=
by
  intros h_capacity h_added_gasoline h_fraction_after h_equation
  sorry

end tank_fraction_before_gas_added_l63_63626


namespace min_value_quadratic_l63_63197

theorem min_value_quadratic :
  ∀ (x : ℝ), (2 * x^2 - 8 * x + 15) ≥ 7 :=
by
  -- We need to show that 2x^2 - 8x + 15 has a minimum value of 7
  sorry

end min_value_quadratic_l63_63197


namespace perimeter_is_correct_l63_63907

def side_length : ℕ := 2
def original_horizontal_segments : ℕ := 16
def original_vertical_segments : ℕ := 10

def horizontal_length : ℕ := original_horizontal_segments * side_length
def vertical_length : ℕ := original_vertical_segments * side_length

def perimeter : ℕ := horizontal_length + vertical_length

theorem perimeter_is_correct : perimeter = 52 :=
by 
  -- Proof goes here.
  sorry

end perimeter_is_correct_l63_63907


namespace coordinates_of_B_l63_63358

def pointA : Prod Int Int := (-3, 2)
def moveRight (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1 + units, p.2)
def moveDown (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1, p.2 - units)
def pointB : Prod Int Int := moveDown (moveRight pointA 1) 2

theorem coordinates_of_B :
  pointB = (-2, 0) :=
sorry

end coordinates_of_B_l63_63358


namespace relationship_among_abc_l63_63365

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 3) ^ 2
noncomputable def c : ℝ := Real.log (1 / 30) / Real.log (1 / 3)

theorem relationship_among_abc : c > a ∧ a > b :=
by
  sorry

end relationship_among_abc_l63_63365


namespace driver_total_miles_per_week_l63_63920

theorem driver_total_miles_per_week :
  let distance_monday_to_saturday := (30 * 3 + 25 * 4 + 40 * 2) * 6
  let distance_sunday := 35 * (5 - 1)
  distance_monday_to_saturday + distance_sunday = 1760 := by
  sorry

end driver_total_miles_per_week_l63_63920


namespace exists_four_digit_number_divisible_by_101_l63_63842

theorem exists_four_digit_number_divisible_by_101 :
  ∃ (a b c d : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
    b ≠ c ∧ b ≠ d ∧
    c ≠ d ∧
    (1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a) % 101 = 0 := 
by
  -- To be proven
  sorry

end exists_four_digit_number_divisible_by_101_l63_63842


namespace sqrt_200_eq_10_l63_63353

theorem sqrt_200_eq_10 (h : 200 = 2^2 * 5^2) : Real.sqrt 200 = 10 := 
by
  sorry

end sqrt_200_eq_10_l63_63353


namespace pradeep_pass_percentage_l63_63678

variable (marks_obtained : ℕ) (marks_short : ℕ) (max_marks : ℝ)

theorem pradeep_pass_percentage (h1 : marks_obtained = 150) (h2 : marks_short = 25) (h3 : max_marks = 500.00000000000006) :
  ((marks_obtained + marks_short) / max_marks) * 100 = 35 := 
by
  sorry

end pradeep_pass_percentage_l63_63678


namespace option_d_correct_l63_63297

theorem option_d_correct (m n : ℝ) : (m + n) * (m - 2 * n) = m^2 - m * n - 2 * n^2 :=
by
  sorry

end option_d_correct_l63_63297


namespace savings_equal_in_820_weeks_l63_63504

-- Definitions for the conditions
def sara_initial_savings : ℕ := 4100
def sara_weekly_savings : ℕ := 10
def jim_weekly_savings : ℕ := 15

-- The statement we want to prove
theorem savings_equal_in_820_weeks : 
  ∃ (w : ℕ), (sara_initial_savings + w * sara_weekly_savings) = (w * jim_weekly_savings) ∧ w = 820 :=
by
  sorry

end savings_equal_in_820_weeks_l63_63504


namespace tricia_age_is_5_l63_63838

theorem tricia_age_is_5 :
  (∀ Amilia Yorick Eugene Khloe Rupert Vincent : ℕ,
    Tricia = 5 ∧
    (3 * Tricia = Amilia) ∧
    (4 * Amilia = Yorick) ∧
    (2 * Eugene = Yorick) ∧
    (Eugene / 3 = Khloe) ∧
    (Khloe + 10 = Rupert) ∧
    (Vincent = 22)) → 
  Tricia = 5 :=
by
  sorry

end tricia_age_is_5_l63_63838


namespace intersection_point_l63_63101

variable (x y z t : ℝ)

-- Conditions
def line_parametric : Prop := 
  (x = 1 + 2 * t) ∧ 
  (y = 2) ∧ 
  (z = 4 + t)

def plane_equation : Prop :=
  x - 2 * y + 4 * z - 19 = 0

-- Problem statement
theorem intersection_point (h_line: line_parametric x y z t) (h_plane: plane_equation x y z):
  x = 3 ∧ y = 2 ∧ z = 5 :=
by
  sorry

end intersection_point_l63_63101


namespace monotonic_increasing_interval_l63_63924

noncomputable def log_base := (1 / 4 : ℝ)

def quad_expression (x : ℝ) : ℝ := -x^2 + 2*x + 3

def is_defined (x : ℝ) : Prop := quad_expression x > 0

theorem monotonic_increasing_interval : ∀ (x : ℝ), 
  is_defined x → 
  ∃ (a b : ℝ), 1 < a ∧ a ≤ x ∧ x < b ∧ b < 3 :=
by
  sorry

end monotonic_increasing_interval_l63_63924


namespace cylinder_surface_area_l63_63290

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches. -/
theorem cylinder_surface_area (h r : ℕ) (h_eq : h = 8) (r_eq : r = 3) :
  2 * Real.pi * r * h + 2 * Real.pi * r ^ 2 = 66 * Real.pi := by
  sorry

end cylinder_surface_area_l63_63290


namespace sum_invested_7000_l63_63202

-- Define the conditions
def interest_15 (P : ℝ) : ℝ := P * 0.15 * 2
def interest_12 (P : ℝ) : ℝ := P * 0.12 * 2

-- Main statement to prove
theorem sum_invested_7000 (P : ℝ) (h : interest_15 P - interest_12 P = 420) : P = 7000 := by
  sorry

end sum_invested_7000_l63_63202


namespace apples_given_by_Susan_l63_63148

theorem apples_given_by_Susan (x y final_apples : ℕ) (h1 : y = 9) (h2 : final_apples = 17) (h3: final_apples = y + x) : x = 8 := by
  sorry

end apples_given_by_Susan_l63_63148


namespace largest_unachievable_score_l63_63902

theorem largest_unachievable_score :
  ∀ (x y : ℕ), 3 * x + 7 * y ≠ 11 :=
by
  sorry

end largest_unachievable_score_l63_63902


namespace max_a_for_three_solutions_l63_63181

-- Define the equation as a Lean function
def equation (x a : ℝ) : ℝ :=
  (|x-2| + 2 * a)^2 - 3 * (|x-2| + 2 * a) + 4 * a * (3 - 4 * a)

-- Statement of the proof problem
theorem max_a_for_three_solutions :
  (∃ (a : ℝ), (∀ x : ℝ, equation x a = 0) ∧
  (∀ (b : ℝ), (∀ x : ℝ, equation x b = 0) → b ≤ 0.5)) :=
sorry

end max_a_for_three_solutions_l63_63181


namespace staff_discount_price_l63_63732

theorem staff_discount_price (d : ℝ) : (d - 0.15*d) * 0.90 = 0.765 * d :=
by
  have discount1 : d - 0.15 * d = d * 0.85 :=
    by ring
  have discount2 : (d * 0.85) * 0.90 = d * (0.85 * 0.90) :=
    by ring
  have final_price : d * (0.85 * 0.90) = d * 0.765 :=
    by norm_num
  rw [discount1, discount2, final_price]
  sorry

end staff_discount_price_l63_63732


namespace reduced_price_is_25_l63_63225

noncomputable def original_price (P : ℝ) := P
noncomputable def reduced_price (P : ℝ) := P * 0.85
noncomputable def amount_of_wheat_original (P : ℝ) := 500 / P
noncomputable def amount_of_wheat_reduced (P : ℝ) := 500 / (P * 0.85)

theorem reduced_price_is_25 : 
  ∃ (P : ℝ), reduced_price P = 25 ∧ (amount_of_wheat_reduced P = amount_of_wheat_original P + 3) :=
sorry

end reduced_price_is_25_l63_63225


namespace find_x_l63_63075

theorem find_x (x : ℝ) : |2 * x - 6| = 3 * x + 1 ↔ x = 1 := 
by 
  sorry

end find_x_l63_63075


namespace value_of_expression_l63_63860

theorem value_of_expression (x y : ℝ) (h₁ : x * y = 3) (h₂ : x + y = 4) : x ^ 2 + y ^ 2 - 3 * x * y = 1 := 
by
  sorry

end value_of_expression_l63_63860


namespace geometric_sequence_problem_l63_63072

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_problem (a : ℕ → ℝ) (ha : geometric_sequence a) (h : a 4 + a 8 = 1 / 2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 1 / 4 :=
sorry

end geometric_sequence_problem_l63_63072


namespace alices_favorite_number_l63_63588

theorem alices_favorite_number :
  ∃ n : ℕ, 80 < n ∧ n ≤ 130 ∧ n % 13 = 0 ∧ n % 3 ≠ 0 ∧ ((n / 100) + (n % 100 / 10) + (n % 10)) % 4 = 0 ∧ n = 130 :=
by
  sorry

end alices_favorite_number_l63_63588


namespace value_of_x2017_l63_63987

-- Definitions and conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f (x) < f (y)

def arithmetic_sequence (x : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, x (n + 1) = x n + d

variables (f : ℝ → ℝ) (x : ℕ → ℝ)
variables (d : ℝ)
variable (h_odd : is_odd_function f)
variable (h_increasing : is_increasing_function f)
variable (h_arithmetic : arithmetic_sequence x 2)
variable (h_condition : f (x 7) + f (x 8) = 0)

-- Define the proof goal
theorem value_of_x2017 : x 2017 = 4019 :=
by
  sorry

end value_of_x2017_l63_63987


namespace n_consecutive_even_sum_l63_63520

theorem n_consecutive_even_sum (n k : ℕ) (hn : n > 2) (hk : k > 2) : 
  ∃ (a : ℕ), (n * (n - 1)^(k - 1)) = (2 * a + (2 * a + 2 * (n - 1))) / 2 * n :=
by
  sorry

end n_consecutive_even_sum_l63_63520


namespace greatest_fourth_term_l63_63978

theorem greatest_fourth_term (a d : ℕ) (h1 : a > 0) (h2 : d > 0) 
  (h3 : 5 * a + 10 * d = 50) (h4 : a + 2 * d = 10) : 
  a + 3 * d = 14 :=
by {
  -- We introduced the given constraints and now need a proof
  sorry
}

end greatest_fourth_term_l63_63978


namespace Heather_heavier_than_Emily_l63_63992

def Heather_weight := 87
def Emily_weight := 9

theorem Heather_heavier_than_Emily : (Heather_weight - Emily_weight = 78) :=
by sorry

end Heather_heavier_than_Emily_l63_63992


namespace line_through_P_parallel_to_AB_circumcircle_of_triangle_OAB_l63_63915

-- Define the points A, B and P
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 2)
def P : ℝ × ℝ := (2, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the functions and theorems for the problem
theorem line_through_P_parallel_to_AB :
  ∃ k b : ℝ, ∀ x y : ℝ, ((y = k * x + b) ↔ (x + 2 * y - 8 = 0)) :=
sorry

theorem circumcircle_of_triangle_OAB :
  ∃ cx cy r : ℝ, (cx, cy) = (2, 1) ∧ r^2 = 5 ∧ ∀ x y : ℝ, ((x - cx)^2 + (y - cy)^2 = r^2) ↔ ((x - 2)^2 + (y - 1)^2 = 5) :=
sorry

end line_through_P_parallel_to_AB_circumcircle_of_triangle_OAB_l63_63915


namespace problem_solution_l63_63487

theorem problem_solution (x : ℕ) (h : 3^x + 3^x + 3^x + 3^x = 972) : (x + 2) * (x - 2) = 5 :=
by
  sorry

end problem_solution_l63_63487


namespace linear_valid_arrangements_circular_valid_arrangements_l63_63277

def word := "EFFERVESCES"
def multiplicities := [("E", 4), ("F", 2), ("S", 2), ("R", 1), ("V", 1), ("C", 1)]

-- Number of valid linear arrangements
def linear_arrangements_no_adj_e : ℕ := 88200

-- Number of valid circular arrangements
def circular_arrangements_no_adj_e : ℕ := 6300

theorem linear_valid_arrangements : 
  ∃ n, n = linear_arrangements_no_adj_e := 
  by
    sorry 

theorem circular_valid_arrangements :
  ∃ n, n = circular_arrangements_no_adj_e :=
  by
    sorry

end linear_valid_arrangements_circular_valid_arrangements_l63_63277


namespace tan_alpha_plus_cot_alpha_l63_63922

theorem tan_alpha_plus_cot_alpha (α : Real) (h : Real.sin (2 * α) = 3 / 4) : 
  Real.tan α + 1 / Real.tan α = 8 / 3 :=
  sorry

end tan_alpha_plus_cot_alpha_l63_63922


namespace final_speed_of_ball_l63_63640

/--
 A small rubber ball moves horizontally between two vertical walls. One wall is fixed, and the other wall moves away from it at a constant speed u.
 The ball's collisions are perfectly elastic. The initial speed of the ball is v₀. Prove that after 10 collisions with the moving wall, the ball's speed is 17 cm/s.
-/
theorem final_speed_of_ball
    (u : ℝ) (v₀ : ℝ) (n : ℕ)
    (u_val : u = 100) (v₀_val : v₀ = 2017) (n_val : n = 10) :
    v₀ - 2 * u * n = 17 := 
    by
    rw [u_val, v₀_val, n_val]
    sorry

end final_speed_of_ball_l63_63640


namespace ants_harvest_remaining_sugar_l63_63900

-- Define the initial conditions
def ants_removal_rate : ℕ := 4
def initial_sugar_amount : ℕ := 24
def hours_passed : ℕ := 3

-- Calculate the correct answer
def remaining_sugar (initial : ℕ) (rate : ℕ) (hours : ℕ) : ℕ :=
  initial - (rate * hours)

def additional_hours_needed (remaining_sugar : ℕ) (rate : ℕ) : ℕ :=
  remaining_sugar / rate

-- The specification of the proof problem
theorem ants_harvest_remaining_sugar :
  additional_hours_needed (remaining_sugar initial_sugar_amount ants_removal_rate hours_passed) ants_removal_rate = 3 :=
by
  -- Proof omitted
  sorry

end ants_harvest_remaining_sugar_l63_63900


namespace truck_travel_distance_l63_63126

theorem truck_travel_distance (b t : ℝ) (h1 : t > 0) :
  (300 * (b / 4) / t) / 3 = (25 * b) / t :=
by
  sorry

end truck_travel_distance_l63_63126


namespace solve_equation_l63_63673

theorem solve_equation : ∃ x : ℤ, 3 * x - 2 * x = 7 ∧ x = 7 :=
by
  sorry

end solve_equation_l63_63673


namespace gardener_area_l63_63222

-- The definition considers the placement of gardeners and the condition for attending flowers.
noncomputable def grid_assignment (gardener_position: (ℕ × ℕ)) (flower_position: (ℕ × ℕ)) : List (ℕ × ℕ) :=
  sorry

-- A theorem that states the equivalent proof.
theorem gardener_area (gardener_position: (ℕ × ℕ)) :
  ∀ flower_position: (ℕ × ℕ), (∃ g1 g2 g3, g1 ∈ grid_assignment gardener_position flower_position ∧
                                            g2 ∈ grid_assignment gardener_position flower_position ∧
                                            g3 ∈ grid_assignment gardener_position flower_position) →
  (gardener_position = g1 ∨ gardener_position = g2 ∨ gardener_position = g3) → true :=
by
  sorry

end gardener_area_l63_63222


namespace parabola_directrix_distance_l63_63243

theorem parabola_directrix_distance (m : ℝ) (h : |1 / (4 * m)| = 2) : m = 1/8 ∨ m = -1/8 :=
by { sorry }

end parabola_directrix_distance_l63_63243


namespace intersection_M_N_l63_63141

def M : Set ℝ := { y | ∃ x : ℝ, y = x^2 }
def N : Set ℝ := { y | ∃ x : ℝ, y = x }

theorem intersection_M_N : (M ∩ N) = { y : ℝ | 0 ≤ y } :=
by
  sorry

end intersection_M_N_l63_63141


namespace smallest_whole_number_larger_than_triangle_perimeter_l63_63238

theorem smallest_whole_number_larger_than_triangle_perimeter
  (s : ℝ) (h1 : 5 + 19 > s) (h2 : 5 + s > 19) (h3 : 19 + s > 5) :
  ∃ P : ℝ, P = 5 + 19 + s ∧ P < 48 ∧ ∀ n : ℤ, n > P → n = 48 :=
by
  sorry

end smallest_whole_number_larger_than_triangle_perimeter_l63_63238


namespace find_pairs_l63_63750

theorem find_pairs (n k : ℕ) (h_pos_n : 0 < n) (h_cond : n! + n = n ^ k) : 
  (n = 2 ∧ k = 2) ∨ (n = 3 ∧ k = 2) ∨ (n = 5 ∧ k = 3) := 
by 
  sorry

end find_pairs_l63_63750


namespace problem_1_problem_2_problem_3_l63_63393

open Set

-- Define the universal set U
def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 10}

-- Define sets A, B, and C
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

-- Problem Statements
theorem problem_1 : A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10} := by
  sorry

theorem problem_2 : (A ∩ B) ∩ C = ∅ := by
  sorry

theorem problem_3 : (U \ A) ∩ (U \ B) = {0, 3} := by
  sorry

end problem_1_problem_2_problem_3_l63_63393


namespace total_fencing_cost_is_correct_l63_63878

-- Define the fencing cost per side
def costPerSide : Nat := 69

-- Define the number of sides for a square
def sidesOfSquare : Nat := 4

-- Define the total cost calculation for fencing the square
def totalCostOfFencing (costPerSide : Nat) (sidesOfSquare : Nat) := costPerSide * sidesOfSquare

-- Prove that for a given cost per side and number of sides, the total cost of fencing the square is 276 dollars
theorem total_fencing_cost_is_correct : totalCostOfFencing 69 4 = 276 :=
by
    -- Proof goes here
    sorry

end total_fencing_cost_is_correct_l63_63878


namespace no_good_number_exists_l63_63329

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_good (n : ℕ) : Prop :=
  (n % sum_of_digits n = 0) ∧
  ((n + 1) % sum_of_digits (n + 1) = 0) ∧
  ((n + 2) % sum_of_digits (n + 2) = 0) ∧
  ((n + 3) % sum_of_digits (n + 3) = 0)

theorem no_good_number_exists : ¬ ∃ n : ℕ, is_good n :=
by sorry

end no_good_number_exists_l63_63329


namespace evaluate_at_3_l63_63409

def f (x : ℕ) : ℕ := x ^ 2

theorem evaluate_at_3 : f 3 = 9 :=
by
  sorry

end evaluate_at_3_l63_63409


namespace largest_m_for_game_with_2022_grids_l63_63502

variables (n : ℕ) (f : ℕ → ℕ)

/- Definitions using conditions given -/

/-- Definition of the game and the marking process -/
def game (n : ℕ) : ℕ := 
  if n % 4 = 0 then n / 2 + 1
  else if n % 4 = 2 then n / 2 + 1
  else 0

/-- Main theorem statement -/
theorem largest_m_for_game_with_2022_grids : game 2022 = 1011 :=
by sorry

end largest_m_for_game_with_2022_grids_l63_63502


namespace small_load_clothing_count_l63_63720

def initial_clothes : ℕ := 36
def first_load_clothes : ℕ := 18
def remaining_clothes := initial_clothes - first_load_clothes
def small_load_clothes := remaining_clothes / 2

theorem small_load_clothing_count : 
  small_load_clothes = 9 :=
by
  sorry

end small_load_clothing_count_l63_63720


namespace max_value_of_m_l63_63242

theorem max_value_of_m (x m : ℝ) (h1 : x^2 - 4*x - 5 > 0) (h2 : x^2 - 2*x + 1 - m^2 > 0) (hm : m > 0) 
(hsuff : ∀ (x : ℝ), (x < -1 ∨ x > 5) → (x > m + 1 ∨ x < 1 - m)) : m ≤ 2 :=
sorry

end max_value_of_m_l63_63242


namespace fraction_expression_l63_63494

theorem fraction_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + 3 * Real.sin α) = 3 / 8 := by
  sorry

end fraction_expression_l63_63494


namespace new_person_age_l63_63901

theorem new_person_age (T : ℕ) : 
  (T / 10) = ((T - 46 + A) / 10) + 3 → (A = 16) :=
by
  sorry

end new_person_age_l63_63901


namespace ratio_of_width_perimeter_is_3_16_l63_63512

-- We define the conditions
def length_of_room : ℕ := 25
def width_of_room : ℕ := 15

-- We define the calculation and verification of the ratio
theorem ratio_of_width_perimeter_is_3_16 :
  let P := 2 * (length_of_room + width_of_room)
  let ratio := width_of_room / P
  let a := 15 / Nat.gcd 15 80
  let b := 80 / Nat.gcd 15 80
  (a, b) = (3, 16) :=
by 
  -- The proof is skipped with sorry
  sorry

end ratio_of_width_perimeter_is_3_16_l63_63512


namespace weight_of_replaced_person_l63_63688

-- Define the conditions
variables (W : ℝ) (new_person_weight : ℝ) (avg_weight_increase : ℝ)
#check ℝ

def initial_group_size := 10

-- Define the conditions as hypothesis statements
axiom weight_increase_eq : avg_weight_increase = 3.5
axiom new_person_weight_eq : new_person_weight = 100

-- Define the result to be proved
theorem weight_of_replaced_person (W : ℝ) : 
  ∀ (avg_weight_increase : ℝ) (new_person_weight : ℝ),
    avg_weight_increase = 3.5 ∧ new_person_weight = 100 → 
    (new_person_weight - (avg_weight_increase * initial_group_size)) = 65 := 
by
  sorry

end weight_of_replaced_person_l63_63688


namespace perpendicular_length_GH_from_centroid_l63_63918

theorem perpendicular_length_GH_from_centroid
  (A B C D E F G : ℝ)
  -- Conditions for distances from vertices to the line RS
  (hAD : AD = 12)
  (hBE : BE = 12)
  (hCF : CF = 18)
  -- Define the coordinates based on the vertical distances to line RS
  (yA : A = 12)
  (yB : B = 12)
  (yC : C = 18)
  -- Define the centroid G of triangle ABC based on the average of the y-coordinates
  (yG : G = (A + B + C) / 3)
  : G = 14 :=
by
  sorry

end perpendicular_length_GH_from_centroid_l63_63918


namespace combination_10_3_eq_120_l63_63655

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l63_63655


namespace cost_per_bag_l63_63775

theorem cost_per_bag (C : ℝ)
  (total_bags : ℕ := 20)
  (price_per_bag_original : ℝ := 6)
  (sold_original : ℕ := 15)
  (price_per_bag_discounted : ℝ := 4)
  (sold_discounted : ℕ := 5)
  (net_profit : ℝ := 50) :
  sold_original * price_per_bag_original + sold_discounted * price_per_bag_discounted - net_profit = total_bags * C →
  C = 3 :=
by
  intros h
  sorry

end cost_per_bag_l63_63775


namespace hexagon_inequality_l63_63178

variable {Point : Type*} [MetricSpace Point]

-- Define points A1, A2, A3, A4, A5, A6 in a Metric Space
variables (A1 A2 A3 A4 A5 A6 O : Point)

-- Conditions
def angle_condition (O A1 A2 A3 A4 A5 A6 : Point) : Prop :=
  -- Points form a hexagon where each side is visible from O at 60 degrees
  -- We assume MetricSpace has a function measuring angles such as angle O x y = 60
  true -- A simplified condition; the actual angle measurement needs more geometry setup

def distance_condition_odd (O A1 A3 A5 : Point) : Prop := dist O A1 > dist O A3 ∧ dist O A3 > dist O A5
def distance_condition_even (O A2 A4 A6 : Point) : Prop := dist O A2 > dist O A4 ∧ dist O A4 > dist O A6

-- Question to prove
theorem hexagon_inequality 
  (hc : angle_condition O A1 A2 A3 A4 A5 A6) 
  (ho : distance_condition_odd O A1 A3 A5)
  (he : distance_condition_even O A2 A4 A6) : 
  dist A1 A2 + dist A3 A4 + dist A5 A6 < dist A2 A3 + dist A4 A5 + dist A6 A1 := 
sorry

end hexagon_inequality_l63_63178


namespace complex_number_quadrant_l63_63736

def imaginary_unit := Complex.I

def complex_simplification (z : Complex) : Complex :=
  z

theorem complex_number_quadrant :
  ∃ z : Complex, z = (5 * imaginary_unit) / (2 + imaginary_unit ^ 9) ∧ (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_number_quadrant_l63_63736


namespace overall_average_marks_l63_63686

theorem overall_average_marks 
  (n1 : ℕ) (m1 : ℕ) 
  (n2 : ℕ) (m2 : ℕ) 
  (n3 : ℕ) (m3 : ℕ) 
  (n4 : ℕ) (m4 : ℕ) 
  (h1 : n1 = 70) (h2 : m1 = 50) 
  (h3 : n2 = 35) (h4 : m2 = 60)
  (h5 : n3 = 45) (h6 : m3 = 55)
  (h7 : n4 = 42) (h8 : m4 = 45) :
  (n1 * m1 + n2 * m2 + n3 * m3 + n4 * m4) / (n1 + n2 + n3 + n4) = 9965 / 192 :=
by
  sorry

end overall_average_marks_l63_63686


namespace black_cards_remaining_proof_l63_63617

def initial_black_cards := 26
def black_cards_taken_out := 4
def black_cards_remaining := initial_black_cards - black_cards_taken_out

theorem black_cards_remaining_proof : black_cards_remaining = 22 := 
by sorry

end black_cards_remaining_proof_l63_63617


namespace expression_evaluation_l63_63372

theorem expression_evaluation : (16^3 + 3 * 16^2 + 3 * 16 + 1 = 4913) :=
by
  sorry

end expression_evaluation_l63_63372


namespace eliana_additional_steps_first_day_l63_63625

variables (x : ℝ)

def eliana_first_day_steps := 200 + x
def eliana_second_day_steps := 2 * eliana_first_day_steps
def eliana_third_day_steps := eliana_second_day_steps + 100
def eliana_total_steps := eliana_first_day_steps + eliana_second_day_steps + eliana_third_day_steps

theorem eliana_additional_steps_first_day : eliana_total_steps = 1600 → x = 100 :=
by {
  sorry
}

end eliana_additional_steps_first_day_l63_63625


namespace leak_empties_cistern_in_24_hours_l63_63244

noncomputable def cistern_fill_rate_without_leak : ℝ := 1 / 8
noncomputable def cistern_fill_rate_with_leak : ℝ := 1 / 12

theorem leak_empties_cistern_in_24_hours :
  (1 / (cistern_fill_rate_without_leak - cistern_fill_rate_with_leak)) = 24 :=
by
  sorry

end leak_empties_cistern_in_24_hours_l63_63244


namespace seat_arrangement_l63_63074

theorem seat_arrangement (seats : ℕ) (people : ℕ) (min_empty_between : ℕ) : 
  seats = 9 ∧ people = 3 ∧ min_empty_between = 2 → 
  ∃ ways : ℕ, ways = 60 :=
by
  intro h
  sorry

end seat_arrangement_l63_63074


namespace krystiana_earnings_l63_63931

def earning_building1_first_floor : ℝ := 5 * 15 * 0.8
def earning_building1_second_floor : ℝ := 6 * 25 * 0.75
def earning_building1_third_floor : ℝ := 9 * 30 * 0.5
def earning_building1_fourth_floor : ℝ := 4 * 60 * 0.85
def earnings_building1 : ℝ := earning_building1_first_floor + earning_building1_second_floor + earning_building1_third_floor + earning_building1_fourth_floor

def earning_building2_first_floor : ℝ := 7 * 20 * 0.9
def earning_building2_second_floor : ℝ := (25 + 30 + 35 + 40 + 45 + 50 + 55 + 60) * 0.7
def earning_building2_third_floor : ℝ := 6 * 60 * 0.6
def earnings_building2 : ℝ := earning_building2_first_floor + earning_building2_second_floor + earning_building2_third_floor

def total_earnings : ℝ := earnings_building1 + earnings_building2

theorem krystiana_earnings : total_earnings = 1091.5 := by
  sorry

end krystiana_earnings_l63_63931


namespace equal_tuesdays_and_fridays_l63_63111

theorem equal_tuesdays_and_fridays (days_in_month : ℕ) (days_of_week : ℕ) (extra_days : ℕ) (starting_days : Finset ℕ) :
  days_in_month = 30 → days_of_week = 7 → extra_days = 2 →
  starting_days = {0, 3, 6} →
  ∃ n : ℕ, n = 3 :=
by
  sorry

end equal_tuesdays_and_fridays_l63_63111


namespace quadrant_of_alpha_l63_63216

theorem quadrant_of_alpha (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
  (π / 2 < α ∧ α < π) := 
sorry

end quadrant_of_alpha_l63_63216


namespace john_finishes_fourth_task_at_12_18_PM_l63_63289

theorem john_finishes_fourth_task_at_12_18_PM :
  let start_time := 8 * 60 + 45 -- Start time in minutes from midnight
  let third_task_time := 11 * 60 + 25 -- End time of the third task in minutes from midnight
  let total_time_three_tasks := third_task_time - start_time -- Total time in minutes to complete three tasks
  let time_per_task := total_time_three_tasks / 3 -- Time per task in minutes
  let fourth_task_end_time := third_task_time + time_per_task -- End time of the fourth task in minutes from midnight
  fourth_task_end_time = 12 * 60 + 18 := -- Expected end time in minutes from midnight
  sorry

end john_finishes_fourth_task_at_12_18_PM_l63_63289


namespace pow_mod_sub_remainder_l63_63305

theorem pow_mod_sub_remainder :
  (10^23 - 7) % 6 = 3 :=
sorry

end pow_mod_sub_remainder_l63_63305


namespace units_digit_7_pow_75_plus_6_l63_63217

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_75_plus_6 : units_digit (7 ^ 75 + 6) = 9 := 
by
  sorry

end units_digit_7_pow_75_plus_6_l63_63217


namespace abc_sum_is_12_l63_63170

theorem abc_sum_is_12
  (a b c : ℕ)
  (h : 28 * a + 30 * b + 31 * c = 365) :
  a + b + c = 12 :=
by
  sorry

end abc_sum_is_12_l63_63170


namespace initially_tagged_fish_l63_63493

theorem initially_tagged_fish (second_catch_total : ℕ) (second_catch_tagged : ℕ)
  (total_fish_pond : ℕ) (approx_ratio : ℚ) 
  (h1 : second_catch_total = 50)
  (h2 : second_catch_tagged = 2)
  (h3 : total_fish_pond = 1750)
  (h4 : approx_ratio = (second_catch_tagged : ℚ) / second_catch_total) :
  ∃ T : ℕ, T = 70 :=
by
  sorry

end initially_tagged_fish_l63_63493


namespace problem_l63_63228

/-- Given a number d > 7, 
    digits A and B in base d such that the equation AB_d + AA_d = 172_d holds, 
    we want to prove that A_d - B_d = 5. --/

theorem problem (d A B : ℕ) (hd : 7 < d)
  (hAB : d * A + B + d * A + A = 1 * d^2 + 7 * d + 2) : A - B = 5 :=
sorry

end problem_l63_63228


namespace bottles_per_crate_l63_63604

theorem bottles_per_crate (num_bottles total_bottles bottles_not_placed num_crates : ℕ) 
    (h1 : total_bottles = 130)
    (h2 : bottles_not_placed = 10)
    (h3 : num_crates = 10) 
    (h4 : num_bottles = total_bottles - bottles_not_placed) :
    (num_bottles / num_crates) = 12 := 
by 
    sorry

end bottles_per_crate_l63_63604


namespace black_greater_than_gray_by_103_l63_63151

def a := 12
def b := 9
def c := 7
def d := 3

def area (side: ℕ) := side * side

def black_area_sum : ℕ := area a + area c
def gray_area_sum : ℕ := area b + area d

theorem black_greater_than_gray_by_103 :
  black_area_sum - gray_area_sum = 103 := by
  sorry

end black_greater_than_gray_by_103_l63_63151


namespace range_of_m_l63_63596

theorem range_of_m {a b c x0 y0 y1 y2 m : ℝ} (h1 : a ≠ 0)
    (A_on_parabola : y1 = a * m^2 + 4 * a * m + c)
    (B_on_parabola : y2 = a * (m + 2)^2 + 4 * a * (m + 2) + c)
    (C_on_parabola : y0 = a * (-2)^2 + 4 * a * (-2) + c)
    (C_is_vertex : x0 = -2)
    (y_relation : y0 ≥ y2 ∧ y2 > y1) :
    m < -3 := 
sorry

end range_of_m_l63_63596


namespace find_value_l63_63587

theorem find_value (number : ℕ) (h : number / 5 + 16 = 58) : number / 15 + 74 = 88 :=
sorry

end find_value_l63_63587


namespace heartbeats_during_race_l63_63335

-- Define the conditions as constants
def heart_rate := 150 -- beats per minute
def race_distance := 26 -- miles
def pace := 5 -- minutes per mile

-- Formulate the statement
theorem heartbeats_during_race :
  heart_rate * (race_distance * pace) = 19500 :=
by
  sorry

end heartbeats_during_race_l63_63335


namespace remainder_div_x_minus_2_l63_63707

noncomputable def q (x : ℝ) (A B C : ℝ) : ℝ := A * x^6 + B * x^4 + C * x^2 + 10

theorem remainder_div_x_minus_2 (A B C : ℝ) (h : q 2 A B C = 20) : q (-2) A B C = 20 :=
by sorry

end remainder_div_x_minus_2_l63_63707


namespace tom_seashells_l63_63007

theorem tom_seashells (days_at_beach : ℕ) (seashells_per_day : ℕ) (total_seashells : ℕ) 
  (h1 : days_at_beach = 5) (h2 : seashells_per_day = 7) : total_seashells = 35 := 
by 
  sorry

end tom_seashells_l63_63007


namespace ratio_AH_HD_triangle_l63_63472

theorem ratio_AH_HD_triangle (BC AC : ℝ) (angleC : ℝ) (H AD HD : ℝ) 
  (hBC : BC = 4) (hAC : AC = 3 * Real.sqrt 2) (hAngleC : angleC = 45) 
  (hAD : AD = 3) (hHD : HD = 1) : 
  (AH / HD) = 2 :=
by
  sorry

end ratio_AH_HD_triangle_l63_63472


namespace max_x2y_l63_63232

noncomputable def maximum_value_x_squared_y (x y : ℝ) : ℝ :=
  if x ∈ Set.Ici 0 ∧ y ∈ Set.Ici 0 ∧ x^3 + y^3 + 3*x*y = 1 then x^2 * y else 0

theorem max_x2y (x y : ℝ) (h1 : x ∈ Set.Ici 0) (h2 : y ∈ Set.Ici 0) (h3 : x^3 + y^3 + 3*x*y = 1) :
  maximum_value_x_squared_y x y = 4 / 27 :=
sorry

end max_x2y_l63_63232


namespace squirrel_cones_l63_63977

theorem squirrel_cones :
  ∃ (x y : ℕ), 
    x + y < 25 ∧ 
    2 * x > y + 26 ∧ 
    2 * y > x - 4 ∧
    x = 17 ∧ 
    y = 7 :=
by
  sorry

end squirrel_cones_l63_63977


namespace ratio_A_B_share_l63_63360

-- Define the capital contributions and time in months
def A_capital : ℕ := 3500
def B_capital : ℕ := 15750
def A_months: ℕ := 12
def B_months: ℕ := 4

-- Effective capital contributions
def A_contribution : ℕ := A_capital * A_months
def B_contribution : ℕ := B_capital * B_months

-- Declare the theorem to prove the ratio 2:3
theorem ratio_A_B_share : A_contribution / 21000 = 2 ∧ B_contribution / 21000 = 3 :=
by
  -- Calculate and simplify the ratios
  have hA : A_contribution = 42000 := rfl
  have hB : B_contribution = 63000 := rfl
  have hGCD : Nat.gcd 42000 63000 = 21000 := rfl
  sorry

end ratio_A_B_share_l63_63360


namespace survey_total_people_l63_63012

theorem survey_total_people (number_represented : ℕ) (percentage : ℝ) (h : number_represented = percentage * 200) : 
  (number_represented : ℝ) = 200 := 
by 
 sorry

end survey_total_people_l63_63012


namespace coverable_hook_l63_63200

def is_coverable (m n : ℕ) : Prop :=
  ∃ a b : ℕ, (m = 3 * a ∧ n = 4 * b) ∨ (m = 12 * a ∧ n = b ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 5)

theorem coverable_hook (m n : ℕ) : (∃ a b : ℕ, (m = 3 * a ∧ n = 4 * b) ∨ (m = 12 * a ∧ n = b ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 5))
  ↔ is_coverable m n :=
by
  sorry

end coverable_hook_l63_63200


namespace least_k_l63_63780

noncomputable def u : ℕ → ℝ
| 0 => 1 / 8
| (n + 1) => 3 * u n - 3 * (u n) ^ 2

theorem least_k :
  ∃ k : ℕ, |u k - (1 / 3)| ≤ 1 / 2 ^ 500 ∧ ∀ m < k, |u m - (1 / 3)| > 1 / 2 ^ 500 :=
by
  sorry

end least_k_l63_63780


namespace ab_value_l63_63102

theorem ab_value 
  (a b c : ℝ)
  (h1 : a - b = 5)
  (h2 : a^2 + b^2 = 34)
  (h3 : a^3 - b^3 = 30)
  (h4 : a^2 + b^2 - c^2 = 50)
  (h5 : c = 2 * a - b) : 
  a * b = 17 := 
by 
  sorry

end ab_value_l63_63102


namespace remainder_eq_four_l63_63862

theorem remainder_eq_four {x : ℤ} (h : x % 61 = 24) : x % 5 = 4 :=
sorry

end remainder_eq_four_l63_63862


namespace factor_expression_l63_63818

theorem factor_expression (x : ℝ) : 
  x^2 * (x + 3) + 3 * (x + 3) = (x^2 + 3) * (x + 3) :=
by
  sorry

end factor_expression_l63_63818


namespace range_of_a_l63_63199

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, a*x^2 - 3*x + 2 = 0) ∧ 
  (∀ x y : ℝ, a*x^2 - 3*x + 2 = 0 ∧ a*y^2 - 3*y + 2 = 0 → x = y) 
  ↔ (a = 0 ∨ a = 9 / 8) := by sorry

end range_of_a_l63_63199


namespace inequality_ab_gt_ac_l63_63542

theorem inequality_ab_gt_ac {a b c : ℝ} (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
sorry

end inequality_ab_gt_ac_l63_63542


namespace range_of_x_l63_63134

open Real

noncomputable def f (x : ℝ) : ℝ := log x - (x / (1 + 2 * x))

theorem range_of_x (x : ℝ) :
  f (x * (3 * x - 2)) < -1 / 3 ↔ (-(1 / 3) < x ∧ x < 0) ∨ ((2 / 3) < x ∧ x < 1) :=
by
  sorry

end range_of_x_l63_63134


namespace dawn_lemonade_price_l63_63777

theorem dawn_lemonade_price (x : ℕ) : 
  (10 * 25) = (8 * x) + 26 → x = 28 :=
by 
  sorry

end dawn_lemonade_price_l63_63777


namespace invitees_count_l63_63474

theorem invitees_count 
  (packages : ℕ) 
  (weight_per_package : ℕ) 
  (weight_per_burger : ℕ) 
  (total_people : ℕ)
  (H1 : packages = 4)
  (H2 : weight_per_package = 5)
  (H3 : weight_per_burger = 2)
  (H4 : total_people + 1 = (packages * weight_per_package) / weight_per_burger) :
  total_people = 9 := 
by
  sorry

end invitees_count_l63_63474


namespace percent_reduction_l63_63995

def original_price : ℕ := 500
def reduction_amount : ℕ := 400

theorem percent_reduction : (reduction_amount * 100) / original_price = 80 := by
  sorry

end percent_reduction_l63_63995


namespace vegetables_harvest_problem_l63_63349

theorem vegetables_harvest_problem
  (same_area : ∀ (a b : ℕ), a = b)
  (first_field_harvest : ℕ := 900)
  (second_field_harvest : ℕ := 1500)
  (less_harvest_per_acre : ∀ (x : ℕ), x - 300 = y) :
  x = y ->
  900 / x = 1500 / (x + 300) :=
by
  sorry

end vegetables_harvest_problem_l63_63349


namespace total_drink_volume_l63_63482

theorem total_drink_volume (oj wj gj : ℕ) (hoj : oj = 25) (hwj : wj = 40) (hgj : gj = 70) : (gj * 100) / (100 - oj - wj) = 200 :=
by
  -- Sorry is used to skip the proof
  sorry

end total_drink_volume_l63_63482


namespace min_value_of_quadratic_l63_63185

theorem min_value_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, y = x^2 - 8 * x + 15 → y ≥ -1) ∧ (∃ x₀ : ℝ, x₀ = 4 ∧ (x₀^2 - 8 * x₀ + 15 = -1)) :=
by
  sorry

end min_value_of_quadratic_l63_63185


namespace h_at_3_eq_3_l63_63847

-- Define the function h(x) based on the given condition
noncomputable def h (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * 
    (x^32 + 1) * (x^64 + 1) * (x^128 + 1) * (x^256 + 1) * (x^512 + 1) - 1) / 
  (x^(2^10 - 1) - 1)

-- State the required theorem
theorem h_at_3_eq_3 : h 3 = 3 := by
  sorry

end h_at_3_eq_3_l63_63847


namespace at_least_one_not_solved_l63_63311

theorem at_least_one_not_solved (p q : Prop) : (¬p ∨ ¬q) ↔ ¬(p ∧ q) :=
by sorry

end at_least_one_not_solved_l63_63311


namespace trader_gain_percentage_l63_63328

variable (x : ℝ) (cost_of_one_pen : ℝ := x) (selling_cost_90_pens : ℝ := 90 * x) (gain : ℝ := 30 * x)

theorem trader_gain_percentage :
  30 * cost_of_one_pen / (90 * cost_of_one_pen) * 100 = 33.33 := by
  sorry

end trader_gain_percentage_l63_63328


namespace baron_not_lying_l63_63949

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem baron_not_lying : 
  ∃ a b : ℕ, 
  (a ≠ b ∧ a ≥ 10^9 ∧ a < 10^10 ∧ b ≥ 10^9 ∧ b < 10^10 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a + sum_of_digits (a * a) = b + sum_of_digits (b * b))) :=
  sorry

end baron_not_lying_l63_63949


namespace solve_for_q_l63_63042

theorem solve_for_q (q : ℝ) (p : ℝ) (h : p = 15 * q^2 - 5) : p = 40 → q = Real.sqrt 3 :=
by
  sorry

end solve_for_q_l63_63042


namespace cos_alpha_value_l63_63807

variable (α : ℝ)
variable (x y r : ℝ)

-- Conditions
def point_condition : Prop := (x = 1 ∧ y = -Real.sqrt 3 ∧ r = 2 ∧ r = Real.sqrt (x^2 + y^2))

-- Question/Proof Statement
theorem cos_alpha_value (h : point_condition x y r) : Real.cos α = 1 / 2 :=
sorry

end cos_alpha_value_l63_63807


namespace initial_water_amount_l63_63356

open Real

theorem initial_water_amount (W : ℝ)
  (h1 : ∀ (d : ℝ), d = 0.03 * 20)
  (h2 : ∀ (W : ℝ) (d : ℝ), d = 0.06 * W) :
  W = 10 :=
by
  sorry

end initial_water_amount_l63_63356


namespace find_c_l63_63799

theorem find_c (x c : ℚ) (h1 : 3 * x + 5 = 1) (h2 : c * x + 8 = 6) : c = 3 / 2 := 
sorry

end find_c_l63_63799


namespace arcsin_one_eq_pi_div_two_l63_63246

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := by
  sorry

end arcsin_one_eq_pi_div_two_l63_63246


namespace crayons_end_of_school_year_l63_63751

-- Definitions based on conditions
def crayons_after_birthday : Float := 479.0
def total_crayons_now : Float := 613.0

-- The mathematically equivalent proof problem statement
theorem crayons_end_of_school_year : (total_crayons_now - crayons_after_birthday = 134.0) :=
by
  sorry

end crayons_end_of_school_year_l63_63751


namespace solid_is_cone_l63_63429

-- Definitions for the conditions
structure Solid where
  front_view : Type
  side_view : Type
  top_view : Type

def is_isosceles_triangle (shape : Type) : Prop := sorry
def is_circle (shape : Type) : Prop := sorry

-- Define the solid based on the given conditions
noncomputable def my_solid : Solid := {
  front_view := sorry,
  side_view := sorry,
  top_view := sorry
}

-- Prove that the solid is a cone given the provided conditions
theorem solid_is_cone (s : Solid) : 
  is_isosceles_triangle s.front_view → 
  is_isosceles_triangle s.side_view → 
  is_circle s.top_view → 
  s = my_solid :=
by
  sorry

end solid_is_cone_l63_63429


namespace smallest_product_not_factor_of_48_exists_l63_63831

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end smallest_product_not_factor_of_48_exists_l63_63831


namespace restaurant_june_production_l63_63140

-- Define the given conditions
def daily_hot_dogs := 60
def daily_pizzas := daily_hot_dogs + 40
def june_days := 30
def daily_total := daily_hot_dogs + daily_pizzas
def june_total := daily_total * june_days

-- The goal is to prove that the total number of pizzas and hot dogs made in June is 4800
theorem restaurant_june_production : june_total = 4800 := by
  -- Sorry to skip proof
  sorry

end restaurant_june_production_l63_63140


namespace real_part_of_solution_l63_63431

theorem real_part_of_solution (a b : ℝ) (z : ℂ) (h : z = a + b * Complex.I): 
  z * (z + Complex.I) * (z + 2 * Complex.I) = 1800 * Complex.I → a = 20.75 := by
  sorry

end real_part_of_solution_l63_63431


namespace part1_part2_part3_l63_63796

-- Definition of a companion point
structure Point where
  x : ℝ
  y : ℝ

def isCompanion (P Q : Point) : Prop :=
  Q.x = P.x + 2 ∧ Q.y = P.y - 4

-- Part (1) proof statement
theorem part1 (P Q : Point) (hPQ : isCompanion P Q) (hP : P = ⟨2, -1⟩) (hQ : Q.y = -20 / Q.x) : Q.x = 4 ∧ Q.y = -5 ∧ -20 / 4 = -5 :=
  sorry

-- Part (2) proof statement
theorem part2 (P Q : Point) (hPQ : isCompanion P Q) (hPLine : P.y = P.x - (-5)) (hQ : Q = ⟨-1, -2⟩) : P.x = -3 ∧ P.y = -3 - (-5) ∧ Q.x = -1 ∧ Q.y = -2 :=
  sorry

-- Part (3) proof statement
noncomputable def line2 (Q : Point) := 2*Q.x - 5

theorem part3 (P Q : Point) (hPQ : isCompanion P Q) (hP : P.y = 2*P.x + 3) (hQLine : Q.y = line2 Q) : line2 Q = 2*(P.x + 2) - 5 :=
  sorry

end part1_part2_part3_l63_63796


namespace initial_loss_percentage_l63_63121

theorem initial_loss_percentage 
  (C : ℝ) 
  (h1 : selling_price_one_pencil_20 = 1 / 20)
  (h2 : selling_price_one_pencil_10 = 1 / 10)
  (h3 : C = 1 / (10 * 1.30)) :
  (C - selling_price_one_pencil_20) / C * 100 = 35 :=
by
  sorry

end initial_loss_percentage_l63_63121


namespace smaller_of_two_digit_numbers_l63_63464

theorem smaller_of_two_digit_numbers (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4725) :
  min a b = 15 :=
sorry

end smaller_of_two_digit_numbers_l63_63464


namespace f_eight_l63_63895

noncomputable def f : ℝ → ℝ := sorry -- Defining the function without implementing it here

axiom f_x_neg {x : ℝ} (hx : x < 0) : f x = Real.log (-x) + x
axiom f_symmetric {x : ℝ} (hx : -Real.exp 1 ≤ x ∧ x ≤ Real.exp 1) : f (-x) = -f x
axiom f_periodic {x : ℝ} (hx : x > 1) : f (x + 2) = f x

theorem f_eight : f 8 = 2 - Real.log 2 := 
by
  sorry

end f_eight_l63_63895


namespace symmetry_center_2tan_2x_sub_pi_div_4_l63_63239

theorem symmetry_center_2tan_2x_sub_pi_div_4 (k : ℤ) :
  ∃ (x : ℝ), 2 * (x) - π / 4 = k * π / 2 ∧ x = k * π / 4 + π / 8 :=
by
  sorry

end symmetry_center_2tan_2x_sub_pi_div_4_l63_63239


namespace Creekview_science_fair_l63_63435

/-- Given the total number of students at Creekview High School is 1500,
    900 of these students participate in a science fair, where three-quarters
    of the girls participate and two-thirds of the boys participate,
    prove that 900 girls participate in the science fair. -/
theorem Creekview_science_fair
  (g b : ℕ)
  (h1 : g + b = 1500)
  (h2 : (3 / 4) * g + (2 / 3) * b = 900) :
  (3 / 4) * g = 900 := by
sorry

end Creekview_science_fair_l63_63435


namespace new_volume_correct_l63_63363

-- Define the conditions
def original_volume : ℝ := 60
def length_factor : ℝ := 3
def width_factor : ℝ := 2
def height_factor : ℝ := 1.20

-- Define the new volume as a result of the above factors
def new_volume : ℝ := original_volume * length_factor * width_factor * height_factor

-- Proof statement for the new volume being 432 cubic feet
theorem new_volume_correct : new_volume = 432 :=
by 
    -- Directly state the desired equality
    sorry

end new_volume_correct_l63_63363


namespace circle_shaded_region_perimeter_l63_63437

theorem circle_shaded_region_perimeter
  (O P Q : Type) [MetricSpace O]
  (r : ℝ) (OP OQ : ℝ) (arc_PQ : ℝ)
  (hOP : OP = 8)
  (hOQ : OQ = 8)
  (h_arc_PQ : arc_PQ = 8 * Real.pi) :
  (OP + OQ + arc_PQ = 16 + 8 * Real.pi) :=
by
  sorry

end circle_shaded_region_perimeter_l63_63437


namespace simplest_common_denominator_l63_63913

theorem simplest_common_denominator (x a : ℕ) :
  let d1 := 3 * x
  let d2 := 6 * x^2
  lcm d1 d2 = 6 * x^2 := 
by
  let d1 := 3 * x
  let d2 := 6 * x^2
  show lcm d1 d2 = 6 * x^2
  sorry

end simplest_common_denominator_l63_63913


namespace find_number_l63_63083

-- Define the condition
def is_number (x : ℝ) : Prop :=
  0.15 * x = 0.25 * 16 + 2

-- The theorem statement: proving the number is 40
theorem find_number (x : ℝ) (h : is_number x) : x = 40 :=
by
  -- We would insert the proof steps here
  sorry

end find_number_l63_63083


namespace molecularWeight_correct_l63_63919

noncomputable def molecularWeight (nC nH nO nN: ℤ) 
    (wC wH wO wN : ℚ) : ℚ := nC * wC + nH * wH + nO * wO + nN * wN

theorem molecularWeight_correct : 
    molecularWeight 5 12 3 1 12.01 1.008 16.00 14.01 = 134.156 := by
  sorry

end molecularWeight_correct_l63_63919


namespace initial_number_of_employees_l63_63733

variables (E : ℕ)
def hourly_rate : ℕ := 12
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 5
def weeks_per_month : ℕ := 4
def extra_employees : ℕ := 200
def total_payroll : ℕ := 1680000

-- Total hours worked by each employee per month
def monthly_hours_per_employee : ℕ := hours_per_day * days_per_week * weeks_per_month

-- Monthly salary per employee
def monthly_salary_per_employee : ℕ := monthly_hours_per_employee * hourly_rate

-- Condition expressing the constraint given in the problem
def payroll_equation : Prop :=
  (E + extra_employees) * monthly_salary_per_employee = total_payroll

-- The statement we are proving
theorem initial_number_of_employees :
  payroll_equation E → E = 500 :=
by
  -- Proof not required
  intros
  sorry

end initial_number_of_employees_l63_63733


namespace simplify_expression_l63_63877

variable (x y : ℤ) -- Assume x and y are integers for simplicity

theorem simplify_expression : (5 - 2 * x) - (8 - 6 * x + 3 * y) = -3 + 4 * x - 3 * y := by
  sorry

end simplify_expression_l63_63877


namespace miniature_tank_height_l63_63278

-- Given conditions
def actual_tank_height : ℝ := 50
def actual_tank_volume : ℝ := 200000
def model_tank_volume : ℝ := 0.2

-- Theorem: Calculate the height of the miniature water tank
theorem miniature_tank_height :
  (model_tank_volume / actual_tank_volume) ^ (1/3 : ℝ) * actual_tank_height = 0.5 :=
by
  sorry

end miniature_tank_height_l63_63278


namespace winning_percentage_is_62_l63_63135

-- Definitions based on given conditions
def candidate_winner_votes : ℕ := 992
def candidate_win_margin : ℕ := 384
def total_votes : ℕ := candidate_winner_votes + (candidate_winner_votes - candidate_win_margin)

-- The key proof statement
theorem winning_percentage_is_62 :
  ((candidate_winner_votes : ℚ) / total_votes) * 100 = 62 := 
sorry

end winning_percentage_is_62_l63_63135


namespace horizon_distance_ratio_l63_63034

def R : ℝ := 6000000
def h1 : ℝ := 1
def h2 : ℝ := 2

noncomputable def distance_to_horizon (R h : ℝ) : ℝ :=
  Real.sqrt (2 * R * h)

noncomputable def d1 : ℝ := distance_to_horizon R h1
noncomputable def d2 : ℝ := distance_to_horizon R h2

theorem horizon_distance_ratio : d2 / d1 = Real.sqrt 2 :=
  sorry

end horizon_distance_ratio_l63_63034


namespace parabola_vertex_shift_l63_63908

theorem parabola_vertex_shift
  (vertex_initial : ℝ × ℝ)
  (h₀ : vertex_initial = (0, 0))
  (move_left : ℝ)
  (move_up : ℝ)
  (h₁ : move_left = -2)
  (h₂ : move_up = 3):
  (vertex_initial.1 + move_left, vertex_initial.2 + move_up) = (-2, 3) :=
by
  sorry

end parabola_vertex_shift_l63_63908


namespace average_buns_per_student_l63_63742

theorem average_buns_per_student (packages_class1 packages_class2 packages_class3 packages_class4 : ℕ)
    (buns_per_package students_per_class stale_buns uneaten_buns : ℕ)
    (h1 : packages_class1 = 20)
    (h2 : packages_class2 = 25)
    (h3 : packages_class3 = 30)
    (h4 : packages_class4 = 35)
    (h5 : buns_per_package = 8)
    (h6 : students_per_class = 30)
    (h7 : stale_buns = 16)
    (h8 : uneaten_buns = 20) :
  let total_buns_class1 := packages_class1 * buns_per_package
  let total_buns_class2 := packages_class2 * buns_per_package
  let total_buns_class3 := packages_class3 * buns_per_package
  let total_buns_class4 := packages_class4 * buns_per_package
  let total_uneaten_buns := stale_buns + uneaten_buns
  let uneaten_buns_per_class := total_uneaten_buns / 4
  let remaining_buns_class1 := total_buns_class1 - uneaten_buns_per_class
  let remaining_buns_class2 := total_buns_class2 - uneaten_buns_per_class
  let remaining_buns_class3 := total_buns_class3 - uneaten_buns_per_class
  let remaining_buns_class4 := total_buns_class4 - uneaten_buns_per_class
  let avg_buns_class1 := remaining_buns_class1 / students_per_class
  let avg_buns_class2 := remaining_buns_class2 / students_per_class
  let avg_buns_class3 := remaining_buns_class3 / students_per_class
  let avg_buns_class4 := remaining_buns_class4 / students_per_class
  avg_buns_class1 = 5 ∧ avg_buns_class2 = 6 ∧ avg_buns_class3 = 7 ∧ avg_buns_class4 = 9 :=
by
  sorry

end average_buns_per_student_l63_63742


namespace min_odd_integers_is_zero_l63_63557

noncomputable def minOddIntegers (a b c d e f : ℤ) : ℕ :=
  if h₁ : a + b = 22 ∧ a + b + c + d = 36 ∧ a + b + c + d + e + f = 50 then
    0
  else
    6 -- default, just to match type expectations

theorem min_odd_integers_is_zero (a b c d e f : ℤ)
  (h₁ : a + b = 22)
  (h₂ : a + b + c + d = 36)
  (h₃ : a + b + c + d + e + f = 50) :
  minOddIntegers a b c d e f = 0 :=
  sorry

end min_odd_integers_is_zero_l63_63557


namespace middle_term_in_expansion_sum_of_odd_coefficients_weighted_sum_of_coefficients_l63_63085

noncomputable def term_in_expansion (n k : ℕ) : ℚ :=
  (Nat.choose n k) * ((-1/2) ^ k)

theorem middle_term_in_expansion :
  term_in_expansion 8 4 = 35 / 8 := by
  sorry

theorem sum_of_odd_coefficients :
  (term_in_expansion 8 1 + term_in_expansion 8 3 + term_in_expansion 8 5 + term_in_expansion 8 7) = -(205 / 16) := by
  sorry

theorem weighted_sum_of_coefficients :
  ((1 * term_in_expansion 8 1) + (2 * term_in_expansion 8 2) + (3 * term_in_expansion 8 3) + (4 * term_in_expansion 8 4) +
  (5 * term_in_expansion 8 5) + (6 * term_in_expansion 8 6) + (7 * term_in_expansion 8 7) + (8 * term_in_expansion 8 8)) =
  -(1 / 32) := by
  sorry

end middle_term_in_expansion_sum_of_odd_coefficients_weighted_sum_of_coefficients_l63_63085


namespace exists_student_not_wet_l63_63124

theorem exists_student_not_wet (n : ℕ) (students : Fin (2 * n + 1) → ℝ) (distinct_distances : ∀ i j : Fin (2 * n + 1), i ≠ j → students i ≠ students j) : 
  ∃ i : Fin (2 * n + 1), ∀ j : Fin (2 * n + 1), (j ≠ i → students j ≠ students i) :=
  sorry

end exists_student_not_wet_l63_63124


namespace cost_of_each_taco_l63_63668

variables (T E : ℝ)

-- Conditions
axiom condition1 : 2 * T + 3 * E = 7.80
axiom condition2 : 3 * T + 5 * E = 12.70

-- Question to prove
theorem cost_of_each_taco : T = 0.90 :=
by
  sorry

end cost_of_each_taco_l63_63668


namespace polynomial_solutions_l63_63772

-- Define the type of the polynomials and statement of the problem
def P1 (x : ℝ) : ℝ := x
def P2 (x : ℝ) : ℝ := x^2 + 1
def P3 (x : ℝ) : ℝ := x^4 + 2*x^2 + 2

theorem polynomial_solutions :
  (∀ x : ℝ, P1 (x^2 + 1) = P1 x^2 + 1) ∧
  (∀ x : ℝ, P2 (x^2 + 1) = P2 x^2 + 1) ∧
  (∀ x : ℝ, P3 (x^2 + 1) = P3 x^2 + 1) :=
by
  -- Proof will go here
  sorry

end polynomial_solutions_l63_63772


namespace terry_age_proof_l63_63680

-- Condition 1: In 10 years, Terry will be 4 times the age that Nora is currently.
-- Condition 2: Nora is currently 10 years old.
-- We need to prove that Terry's current age is 30 years old.

variable (Terry_now Terry_in_10 Nora_now : ℕ)

theorem terry_age_proof (h1: Terry_in_10 = 4 * Nora_now) (h2: Nora_now = 10) (h3: Terry_in_10 = Terry_now + 10) : Terry_now = 30 := 
by
  sorry

end terry_age_proof_l63_63680


namespace remainder_is_neg_x_plus_60_l63_63976

theorem remainder_is_neg_x_plus_60 (R : Polynomial ℝ) :
  (R.eval 10 = 50) ∧ (R.eval 50 = 10) → 
  ∃ Q : Polynomial ℝ, R = (Polynomial.X - 10) * (Polynomial.X - 50) * Q + (- Polynomial.X + 60) :=
by
  sorry

end remainder_is_neg_x_plus_60_l63_63976


namespace compute_g_ggg2_l63_63283

def g (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 1
  else if n < 5 then 2 * n + 2
  else 4 * n - 3

theorem compute_g_ggg2 : g (g (g 2)) = 65 :=
by
  sorry

end compute_g_ggg2_l63_63283


namespace sum_three_digit_integers_from_200_to_900_l63_63973

theorem sum_three_digit_integers_from_200_to_900 : 
  let a := 200
  let l := 900
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S = 385550 := by
    let a := 200
    let l := 900
    let d := 1
    let n := (l - a) / d + 1
    let S := n / 2 * (a + l)
    sorry

end sum_three_digit_integers_from_200_to_900_l63_63973


namespace older_brother_catches_up_in_half_hour_l63_63310

-- Defining the parameters according to the conditions
def speed_younger_brother := 4 -- kilometers per hour
def speed_older_brother := 20 -- kilometers per hour
def initial_distance := 8 -- kilometers

-- Calculate the relative speed difference
def speed_difference := speed_older_brother - speed_younger_brother

theorem older_brother_catches_up_in_half_hour:
  ∃ t : ℝ, initial_distance = speed_difference * t ∧ t = 0.5 := by
  use 0.5
  sorry

end older_brother_catches_up_in_half_hour_l63_63310


namespace alternating_draws_probability_l63_63051

noncomputable def probability_alternating_draws : ℚ :=
  let total_draws := 11
  let white_balls := 5
  let black_balls := 6
  let successful_sequences := 1
  let total_sequences := @Nat.choose total_draws black_balls
  successful_sequences / total_sequences

theorem alternating_draws_probability :
  probability_alternating_draws = 1 / 462 := by
  sorry

end alternating_draws_probability_l63_63051


namespace carla_total_time_l63_63009

def total_time_spent (knife_time : ℕ) (peeling_time_multiplier : ℕ) : ℕ :=
  knife_time + peeling_time_multiplier * knife_time

theorem carla_total_time :
  total_time_spent 10 3 = 40 :=
by
  sorry

end carla_total_time_l63_63009


namespace price_reduction_eq_l63_63080

theorem price_reduction_eq (x : ℝ) (price_original price_final : ℝ) 
    (h1 : price_original = 400) 
    (h2 : price_final = 200) 
    (h3 : price_final = price_original * (1 - x) * (1 - x)) :
  400 * (1 - x)^2 = 200 :=
by
  sorry

end price_reduction_eq_l63_63080


namespace least_positive_t_l63_63889

theorem least_positive_t
  (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (ht : ∃ t, 0 < t ∧ (∃ r, (Real.arcsin (Real.sin α) * r = Real.arcsin (Real.sin (3 * α)) ∧ 
                            Real.arcsin (Real.sin (3 * α)) * r = Real.arcsin (Real.sin (5 * α)) ∧
                            Real.arcsin (Real.sin (5 * α)) * r = Real.arcsin (Real.sin (t * α))))) :
  t = 6 :=
sorry

end least_positive_t_l63_63889


namespace union_of_sets_l63_63916

theorem union_of_sets (A B : Set α) : A ∪ B = { x | x ∈ A ∨ x ∈ B } :=
by
  sorry

end union_of_sets_l63_63916


namespace maximum_value_of_function_l63_63362

theorem maximum_value_of_function (a : ℕ) (ha : 0 < a) : 
  ∃ x : ℝ, x + Real.sqrt (13 - 2 * a * x) = 7 :=
by
  sorry

end maximum_value_of_function_l63_63362


namespace potential_values_of_k_l63_63045

theorem potential_values_of_k :
  ∃ k : ℚ, ∀ (a b : ℕ), 
  (10 * a + b = k * (a + b)) ∧ (10 * b + a = (13 - k) * (a + b)) → k = 11/2 :=
by
  sorry

end potential_values_of_k_l63_63045


namespace root_sum_reciprocal_l63_63839

theorem root_sum_reciprocal (p q r s : ℂ)
  (h1 : (∀ x : ℂ, x^4 - 6*x^3 + 11*x^2 - 6*x + 3 = 0 → x = p ∨ x = q ∨ x = r ∨ x = s))
  (h2 : p*q*r*s = 3) 
  (h3 : p*q + p*r + p*s + q*r + q*s + r*s = 11) :
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s)) = 11/3 :=
by
  sorry

end root_sum_reciprocal_l63_63839


namespace percent_receiving_speeding_tickets_l63_63869

theorem percent_receiving_speeding_tickets
  (total_motorists : ℕ)
  (percent_exceeding_limit percent_exceeding_limit_without_ticket : ℚ)
  (h_exceeding_limit : percent_exceeding_limit = 0.5)
  (h_exceeding_limit_without_ticket : percent_exceeding_limit_without_ticket = 0.2) :
  let exceeding_limit := percent_exceeding_limit * total_motorists
  let without_tickets := percent_exceeding_limit_without_ticket * exceeding_limit
  let with_tickets := exceeding_limit - without_tickets
  (with_tickets / total_motorists) * 100 = 40 :=
by
  sorry

end percent_receiving_speeding_tickets_l63_63869


namespace katie_five_dollar_bills_l63_63019

theorem katie_five_dollar_bills (x y : ℕ) (h1 : x + y = 12) (h2 : 5 * x + 10 * y = 80) : x = 8 :=
by
  sorry

end katie_five_dollar_bills_l63_63019


namespace entire_hike_length_l63_63730

-- Definitions directly from the conditions in part a)
def tripp_backpack_weight : ℕ := 25
def charlotte_backpack_weight : ℕ := tripp_backpack_weight - 7
def miles_hiked_first_day : ℕ := 9
def miles_left_to_hike : ℕ := 27

-- Theorem proving the entire hike length
theorem entire_hike_length :
  miles_hiked_first_day + miles_left_to_hike = 36 :=
by
  sorry

end entire_hike_length_l63_63730


namespace paint_cans_used_l63_63025

theorem paint_cans_used (initial_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) 
    (h1 : initial_rooms = 50) (h2 : lost_cans = 5) (h3 : remaining_rooms = 40) : 
    (remaining_rooms / (initial_rooms - remaining_rooms) / lost_cans) = 20 :=
by
  sorry

end paint_cans_used_l63_63025


namespace smallest_of_three_consecutive_even_numbers_l63_63020

def sum_of_three_consecutive_even_numbers (n : ℕ) : Prop :=
  n + (n + 2) + (n + 4) = 162

theorem smallest_of_three_consecutive_even_numbers (n : ℕ) (h : sum_of_three_consecutive_even_numbers n) : n = 52 :=
by
  sorry

end smallest_of_three_consecutive_even_numbers_l63_63020


namespace smallest_pos_integer_l63_63983

-- Definitions based on the given conditions
def arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
def sum_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

-- Given conditions
def condition1 (a1 d : ℤ) : Prop := arithmetic_seq a1 d 11 - arithmetic_seq a1 d 8 = 3
def condition2 (a1 d : ℤ) : Prop := sum_seq a1 d 11 - sum_seq a1 d 8 = 3

-- The claim we want to prove
theorem smallest_pos_integer 
  (n : ℕ) (a1 d : ℤ) 
  (h1 : condition1 a1 d) 
  (h2 : condition2 a1 d) : n = 10 :=
by
  sorry

end smallest_pos_integer_l63_63983


namespace closest_fraction_l63_63763

theorem closest_fraction :
  let won_france := (23 : ℝ) / 120
  let fractions := [ (1 : ℝ) / 4, (1 : ℝ) / 5, (1 : ℝ) / 6, (1 : ℝ) / 7, (1 : ℝ) / 8 ]
  ∃ closest : ℝ, closest ∈ fractions ∧ ∀ f ∈ fractions, abs (won_france - closest) ≤ abs (won_france - f)  :=
  sorry

end closest_fraction_l63_63763


namespace employee_discount_percentage_l63_63883

def wholesale_cost : ℝ := 200
def retail_markup : ℝ := 0.20
def employee_paid_price : ℝ := 228

theorem employee_discount_percentage :
  let retail_price := wholesale_cost * (1 + retail_markup)
  let discount := retail_price - employee_paid_price
  (discount / retail_price) * 100 = 5 := by
  sorry

end employee_discount_percentage_l63_63883


namespace total_area_correct_l63_63937

-- Define the given conditions
def dust_covered_area : ℕ := 64535
def untouched_area : ℕ := 522

-- Define the total area of prairie by summing covered and untouched areas
def total_prairie_area : ℕ := dust_covered_area + untouched_area

-- State the theorem we need to prove
theorem total_area_correct : total_prairie_area = 65057 := by
  sorry

end total_area_correct_l63_63937


namespace difference_of_interchanged_digits_l63_63700

theorem difference_of_interchanged_digits {x y : ℕ} (h : x - y = 4) :
  (10 * x + y) - (10 * y + x) = 36 :=
by sorry

end difference_of_interchanged_digits_l63_63700


namespace law_of_sines_l63_63715

theorem law_of_sines (a b c : ℝ) (A B C : ℝ) (R : ℝ) 
  (hA : a = 2 * R * Real.sin A)
  (hEquilateral1 : b = 2 * R * Real.sin B)
  (hEquilateral2 : c = 2 * R * Real.sin C):
  (a / Real.sin A) = (b / Real.sin B) ∧ 
  (b / Real.sin B) = (c / Real.sin C) ∧ 
  (c / Real.sin C) = 2 * R :=
by
  sorry

end law_of_sines_l63_63715


namespace simplify_expression_l63_63269

theorem simplify_expression (x y : ℝ) : 
  (x - y) * (x + y) + (x - y) ^ 2 = 2 * x ^ 2 - 2 * x * y :=
sorry

end simplify_expression_l63_63269


namespace cost_of_watermelon_and_grapes_l63_63833

variable (x y z f : ℕ)

theorem cost_of_watermelon_and_grapes (h1 : x + y + z + f = 45) 
                                    (h2 : f = 3 * x) 
                                    (h3 : z = x + y) :
    y + z = 9 := by
  sorry

end cost_of_watermelon_and_grapes_l63_63833


namespace cds_unique_to_either_l63_63049

-- Declare the variables for the given problem
variables (total_alice_shared : ℕ) (total_alice : ℕ) (unique_bob : ℕ)

-- The given conditions in the problem
def condition_alice : Prop := total_alice_shared + unique_bob + (total_alice - total_alice_shared) = total_alice

-- The theorem to prove: number of CDs in either Alice's or Bob's collection but not both is 19
theorem cds_unique_to_either (h1 : total_alice = 23) 
                             (h2 : total_alice_shared = 12) 
                             (h3 : unique_bob = 8) : 
                             (total_alice - total_alice_shared) + unique_bob = 19 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end cds_unique_to_either_l63_63049


namespace isosceles_triangle_angles_l63_63336

theorem isosceles_triangle_angles 
  (α r R : ℝ)
  (isosceles : α ∈ {β : ℝ | β = α})
  (circumference_relation : R = 3 * r) :
  (α = Real.arccos (1 / 2 + 1 / (2 * Real.sqrt 3)) ∨ 
   α = Real.arccos (1 / 2 - 1 / (2 * Real.sqrt 3))) ∧ 
  (
    180 = 2 * (Real.arccos (1 / 2 + 1 / (2 * Real.sqrt 3))) + 2 * α ∨
    180 = 2 * (Real.arccos (1 / 2 - 1 / (2 * Real.sqrt 3))) + 2 * α 
  ) :=
by sorry

end isosceles_triangle_angles_l63_63336


namespace cubic_identity_l63_63762

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + ac + bc = 40) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 1575 := 
by
  sorry

end cubic_identity_l63_63762


namespace abc_sum_eq_11sqrt6_l63_63582

theorem abc_sum_eq_11sqrt6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 :=
sorry

end abc_sum_eq_11sqrt6_l63_63582


namespace Laura_running_speed_l63_63573

noncomputable def running_speed (x : ℝ) : Prop :=
  (15 / (3 * x + 2)) + (4 / x) = 1.5 ∧ x > 0

theorem Laura_running_speed : ∃ (x : ℝ), running_speed x ∧ abs (x - 5.64) < 0.01 :=
by
  sorry

end Laura_running_speed_l63_63573


namespace worker_bees_hive_empty_l63_63127

theorem worker_bees_hive_empty:
  ∀ (initial_worker: ℕ) (leave_nectar: ℕ) (reassign_guard: ℕ) (return_trip: ℕ) (multiplier: ℕ),
  initial_worker = 400 →
  leave_nectar = 28 →
  reassign_guard = 30 →
  return_trip = 15 →
  multiplier = 5 →
  ((initial_worker - leave_nectar - reassign_guard + return_trip) * (1 - multiplier)) = 0 :=
by
  intros initial_worker leave_nectar reassign_guard return_trip multiplier
  sorry

end worker_bees_hive_empty_l63_63127


namespace plane_division_99_lines_l63_63433

theorem plane_division_99_lines (m : ℕ) (n : ℕ) : 
  m = 99 ∧ n < 199 → (n = 100 ∨ n = 198) :=
by 
  sorry

end plane_division_99_lines_l63_63433


namespace first_step_induction_l63_63333

theorem first_step_induction (n : ℕ) (h : 1 < n) : 1 + 1/2 + 1/3 < 2 :=
by
  sorry

end first_step_induction_l63_63333


namespace shopkeeper_profit_percentage_l63_63116

theorem shopkeeper_profit_percentage
  (cost_price : ℝ)
  (goods_lost_pct : ℝ)
  (loss_pct : ℝ)
  (remaining_goods : ℝ)
  (selling_price : ℝ)
  (profit_pct : ℝ)
  (h1 : cost_price = 100)
  (h2 : goods_lost_pct = 0.20)
  (h3 : loss_pct = 0.12)
  (h4 : remaining_goods = cost_price * (1 - goods_lost_pct))
  (h5 : selling_price = cost_price * (1 - loss_pct))
  (h6 : profit_pct = ((selling_price - remaining_goods) / remaining_goods) * 100) : 
  profit_pct = 10 := 
sorry

end shopkeeper_profit_percentage_l63_63116


namespace quadratic_difference_sum_l63_63519

theorem quadratic_difference_sum :
  let a := 2
  let b := -10
  let c := 3
  let Δ := b * b - 4 * a * c
  let root1 := (10 + Real.sqrt Δ) / (2 * a)
  let root2 := (10 - Real.sqrt Δ) / (2 * a)
  let diff := root1 - root2
  let m := 19  -- from the difference calculation
  let n := 1   -- from the simplified form
  m + n = 20 :=
by
  -- Placeholders for calculation and proof steps.
  sorry

end quadratic_difference_sum_l63_63519


namespace nate_total_time_l63_63957

/-- Definitions for the conditions -/
def sectionG : ℕ := 18 * 12
def sectionH : ℕ := 25 * 10
def sectionI : ℕ := 17 * 11
def sectionJ : ℕ := 20 * 9
def sectionK : ℕ := 15 * 13

def speedGH : ℕ := 8
def speedIJ : ℕ := 10
def speedK : ℕ := 6

/-- Compute the time spent in each section, rounding up where necessary -/
def timeG : ℕ := (sectionG + speedGH - 1) / speedGH
def timeH : ℕ := (sectionH + speedGH - 1) / speedGH
def timeI : ℕ := (sectionI + speedIJ - 1) / speedIJ
def timeJ : ℕ := (sectionJ + speedIJ - 1) / speedIJ
def timeK : ℕ := (sectionK + speedK - 1) / speedK

/-- Compute the total time spent -/
def totalTime : ℕ := timeG + timeH + timeI + timeJ + timeK

/-- The proof statement -/
theorem nate_total_time : totalTime = 129 := by
  -- the proof goes here
  sorry

end nate_total_time_l63_63957


namespace quadratic_zeros_interval_l63_63585

theorem quadratic_zeros_interval (a : ℝ) :
  (5 - 2 * a > 0) ∧ (4 * a^2 - 16 > 0) ∧ (a > 1) ↔ (2 < a ∧ a < 5 / 2) :=
by
  sorry

end quadratic_zeros_interval_l63_63585


namespace inequality_of_transformed_division_l63_63452

theorem inequality_of_transformed_division (A B : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (h : A * 5 = B * 4) : A ≤ B := by
  sorry

end inequality_of_transformed_division_l63_63452


namespace find_x_plus_y_l63_63451

theorem find_x_plus_y (x y : ℚ) (h1 : 5 * x - 3 * y = 27) (h2 : 3 * x + 5 * y = 1) : x + y = 31 / 17 :=
by
  sorry

end find_x_plus_y_l63_63451


namespace total_sold_l63_63631

theorem total_sold (D C : ℝ) (h1 : D = 1.6 * C) (h2 : D = 168) : D + C = 273 :=
by
  sorry

end total_sold_l63_63631


namespace min_initial_bags_l63_63314

theorem min_initial_bags :
  ∃ x : ℕ, (∃ y : ℕ, (y + 90 = 2 * (x - 90) ∧ x + (11 * x - 1620) / 7 = 6 * (2 * x - 270 - (11 * x - 1620) / 7))
             ∧ x = 153) :=
by { sorry }

end min_initial_bags_l63_63314


namespace difference_of_squares_example_l63_63972

theorem difference_of_squares_example (a b : ℕ) (h₁ : a = 650) (h₂ : b = 350) :
  a^2 - b^2 = 300000 :=
by
  sorry

end difference_of_squares_example_l63_63972


namespace equation_of_line_l63_63622

theorem equation_of_line (θ : ℝ) (b : ℝ) :
  θ = 135 ∧ b = -1 → (∀ x y : ℝ, x + y + 1 = 0) :=
by
  sorry

end equation_of_line_l63_63622


namespace gift_exchange_equation_l63_63005

theorem gift_exchange_equation (x : ℕ) (h : x * (x - 1) = 40) : 
  x * (x - 1) = 40 :=
by
  exact h

end gift_exchange_equation_l63_63005


namespace clara_current_age_l63_63448

theorem clara_current_age (a c : ℕ) (h1 : a = 54) (h2 : (c - 41) = 3 * (a - 41)) : c = 80 :=
by
  -- This is where the proof would be constructed.
  sorry

end clara_current_age_l63_63448


namespace ratio_of_ages_l63_63481

-- Necessary conditions as definitions in Lean
def combined_age (S D : ℕ) : Prop := S + D = 54
def sam_is_18 (S : ℕ) : Prop := S = 18

-- The statement that we need to prove
theorem ratio_of_ages (S D : ℕ) (h1 : combined_age S D) (h2 : sam_is_18 S) : S / D = 1 / 2 := by
  sorry

end ratio_of_ages_l63_63481


namespace area_AKM_less_than_area_ABC_l63_63186

-- Define the rectangle ABCD
structure Rectangle :=
(A B C D : ℝ) -- Four vertices of the rectangle
(side_AB : ℝ) (side_BC : ℝ) (side_CD : ℝ) (side_DA : ℝ)

-- Define the arbitrary points K and M on sides BC and CD respectively
variables (B C D K M : ℝ)

-- Define the area of triangle function and area of rectangle function
def area_triangle (A B C : ℝ) : ℝ := sorry -- Assuming a function calculating area of triangle given 3 vertices
def area_rectangle (A B C D : ℝ) : ℝ := sorry -- Assuming a function calculating area of rectangle given 4 vertices

-- Assuming the conditions given in the problem statement
variables (A : ℝ) (rect : Rectangle)

-- Prove that the area of triangle AKM is less than the area of triangle ABC
theorem area_AKM_less_than_area_ABC : 
  ∀ (K M : ℝ), K ∈ [B,C] → M ∈ [C,D] →
    area_triangle A K M < area_triangle A B C := sorry

end area_AKM_less_than_area_ABC_l63_63186


namespace isosceles_triangle_perimeter_l63_63911

theorem isosceles_triangle_perimeter (x y : ℝ) (h : |x - 4| + (y - 8)^2 = 0) :
  4 + 8 + 8 = 20 :=
by
  sorry

end isosceles_triangle_perimeter_l63_63911


namespace union_example_l63_63754

open Set

variable (A B : Set ℤ)
variable (AB : Set ℤ)

theorem union_example (hA : A = {-3, 1, 2})
                      (hB : B = {0, 1, 2, 3}) :
                      A ∪ B = {-3, 0, 1, 2, 3} :=
by
  rw [hA, hB]
  ext
  simp
  sorry

end union_example_l63_63754


namespace books_sold_wednesday_l63_63598

-- Define the conditions of the problem
def total_books : Nat := 1200
def sold_monday : Nat := 75
def sold_tuesday : Nat := 50
def sold_thursday : Nat := 78
def sold_friday : Nat := 135
def percentage_not_sold : Real := 66.5

-- Define the statement to be proved
theorem books_sold_wednesday : 
  let books_sold := total_books * (1 - percentage_not_sold / 100)
  let known_sales := sold_monday + sold_tuesday + sold_thursday + sold_friday
  books_sold - known_sales = 64 :=
by
  sorry

end books_sold_wednesday_l63_63598


namespace min_value_2xy_minus_2x_minus_y_l63_63955

theorem min_value_2xy_minus_2x_minus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 2/y = 1) :
  2 * x * y - 2 * x - y ≥ 8 :=
sorry

end min_value_2xy_minus_2x_minus_y_l63_63955


namespace work_rate_c_l63_63522

variables (rate_a rate_b rate_c : ℚ)

-- Given conditions
axiom h1 : rate_a + rate_b = 1 / 15
axiom h2 : rate_a + rate_b + rate_c = 1 / 6

theorem work_rate_c : rate_c = 1 / 10 :=
by sorry

end work_rate_c_l63_63522


namespace find_abc_sum_l63_63779

theorem find_abc_sum {U : Type} 
  (a b c : ℕ)
  (ha : a = 26)
  (hb : b = 1)
  (hc : c = 32)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) :
  a + b + c = 59 :=
by
  sorry

end find_abc_sum_l63_63779


namespace circumference_of_circle_l63_63719

def speed_cyclist1 : ℝ := 7
def speed_cyclist2 : ℝ := 8
def meeting_time : ℝ := 42
def circumference : ℝ := 630

theorem circumference_of_circle :
  (speed_cyclist1 * meeting_time + speed_cyclist2 * meeting_time = circumference) :=
by
  sorry

end circumference_of_circle_l63_63719


namespace part1_part2_l63_63267

variable (A B C : ℝ) (a b c : ℝ)
variable (h1 : a = 5) (h2 : c = 6) (h3 : Real.sin B = 3 / 5) (h4 : b < a)

-- Part 1: Prove b = sqrt(13) and sin A = (3 * sqrt(13)) / 13
theorem part1 : b = Real.sqrt 13 ∧ Real.sin A = (3 * Real.sqrt 13) / 13 := sorry

-- Part 2: Prove sin (2A + π / 4) = 7 * sqrt(2) / 26
theorem part2 (h5 : b = Real.sqrt 13) (h6 : Real.sin A = (3 * Real.sqrt 13) / 13) : 
  Real.sin (2 * A + Real.pi / 4) = (7 * Real.sqrt 2) / 26 := sorry

end part1_part2_l63_63267


namespace truck_speed_in_mph_l63_63392

-- Definitions based on the conditions
def truck_length : ℝ := 66  -- Truck length in feet
def tunnel_length : ℝ := 330  -- Tunnel length in feet
def exit_time : ℝ := 6  -- Exit time in seconds
def feet_to_miles : ℝ := 5280  -- Feet per mile

-- Problem statement
theorem truck_speed_in_mph :
  ((tunnel_length + truck_length) / exit_time) * (3600 / feet_to_miles) = 45 := 
sorry

end truck_speed_in_mph_l63_63392


namespace min_rice_pounds_l63_63921

variable {o r : ℝ}

theorem min_rice_pounds (h1 : o ≥ 8 + r / 3) (h2 : o ≤ 2 * r) : r ≥ 5 :=
sorry

end min_rice_pounds_l63_63921


namespace largest_subset_no_multiples_l63_63442

theorem largest_subset_no_multiples : ∀ (S : Finset ℕ), (S = Finset.range 101) → 
  ∃ (A : Finset ℕ), A ⊆ S ∧ (∀ x ∈ A, ∀ y ∈ A, x ≠ y → ¬(x ∣ y) ∧ ¬(y ∣ x)) ∧ A.card = 50 :=
by
  sorry

end largest_subset_no_multiples_l63_63442


namespace mass_percentage_H3BO3_l63_63814

theorem mass_percentage_H3BO3 :
  ∃ (element : String) (mass_percent : ℝ), 
    element ∈ ["H", "B", "O"] ∧ 
    mass_percent = 4.84 ∧ 
    mass_percent = 4.84 :=
sorry

end mass_percentage_H3BO3_l63_63814


namespace largest_difference_l63_63855

def U : ℕ := 2 * 1002 ^ 1003
def V : ℕ := 1002 ^ 1003
def W : ℕ := 1001 * 1002 ^ 1002
def X : ℕ := 2 * 1002 ^ 1002
def Y : ℕ := 1002 ^ 1002
def Z : ℕ := 1002 ^ 1001

theorem largest_difference : (U - V) = 1002 ^ 1003 ∧ 
  (V - W) = 1002 ^ 1002 ∧ 
  (W - X) = 999 * 1002 ^ 1002 ∧ 
  (X - Y) = 1002 ^ 1002 ∧ 
  (Y - Z) = 1001 * 1002 ^ 1001 ∧ 
  (1002 ^ 1003 > 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 999 * 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 1001 * 1002 ^ 1001) :=
by {
  sorry
}

end largest_difference_l63_63855


namespace sum_of_two_numbers_l63_63638

theorem sum_of_two_numbers (x y : ℤ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end sum_of_two_numbers_l63_63638


namespace mrs_bil_earnings_percentage_in_may_l63_63545

theorem mrs_bil_earnings_percentage_in_may
  (M F : ℝ)
  (h₁ : 1.10 * M / (1.10 * M + F) = 0.7196) :
  M / (M + F) = 0.70 :=
sorry

end mrs_bil_earnings_percentage_in_may_l63_63545


namespace shaded_region_perimeter_l63_63828

theorem shaded_region_perimeter :
  let side_length := 1
  let diagonal_length := Real.sqrt 2 * side_length
  let arc_TRU_length := (1 / 4) * (2 * Real.pi * diagonal_length)
  let arc_VPW_length := (1 / 4) * (2 * Real.pi * side_length)
  let arc_UV_length := (1 / 4) * (2 * Real.pi * (Real.sqrt 2 - side_length))
  let arc_WT_length := (1 / 4) * (2 * Real.pi * (Real.sqrt 2 - side_length))
  (arc_TRU_length + arc_VPW_length + arc_UV_length + arc_WT_length) = (2 * Real.sqrt 2 - 1) * Real.pi :=
by
  sorry

end shaded_region_perimeter_l63_63828


namespace solution_set_quadratic_inequality_l63_63133

def quadratic_inequality_solution (x : ℝ) : Prop := x^2 + x - 2 > 0

theorem solution_set_quadratic_inequality :
  {x : ℝ | quadratic_inequality_solution x} = {x : ℝ | x < -2 ∨ x > 1} :=
by
  sorry

end solution_set_quadratic_inequality_l63_63133


namespace somu_present_age_l63_63287

variable (S F : ℕ)

-- Conditions from the problem
def condition1 : Prop := S = F / 3
def condition2 : Prop := S - 10 = (F - 10) / 5

-- The statement we need to prove
theorem somu_present_age (h1 : condition1 S F) (h2 : condition2 S F) : S = 20 := 
by sorry

end somu_present_age_l63_63287


namespace stewart_farm_horse_food_l63_63514

theorem stewart_farm_horse_food 
  (ratio : ℚ) (food_per_horse : ℤ) (num_sheep : ℤ) (num_horses : ℤ)
  (h1 : ratio = 5 / 7)
  (h2 : food_per_horse = 230)
  (h3 : num_sheep = 40)
  (h4 : ratio * num_horses = num_sheep) : 
  (num_horses * food_per_horse = 12880) := 
sorry

end stewart_farm_horse_food_l63_63514


namespace no_bounded_sequences_at_least_one_gt_20_l63_63710

variable (x y z : ℕ → ℝ)
variable (x1 y1 z1 : ℝ)
variable (h0 : x1 > 0) (h1 : y1 > 0) (h2 : z1 > 0)
variable (h3 : ∀ n, x (n + 1) = y n + (1 / z n))
variable (h4 : ∀ n, y (n + 1) = z n + (1 / x n))
variable (h5 : ∀ n, z (n + 1) = x n + (1 / y n))

-- Part (a)
theorem no_bounded_sequences : (∀ n, x n > 0) ∧ (∀ n, y n > 0) ∧ (∀ n, z n > 0) → ¬ (∃ M, ∀ n, x n < M ∧ y n < M ∧ z n < M) :=
sorry

-- Part (b)
theorem at_least_one_gt_20 : x 1 = x1 ∧ y 1 = y1 ∧ z 1 = z1 → x 200 > 20 ∨ y 200 > 20 ∨ z 200 > 20 :=
sorry

end no_bounded_sequences_at_least_one_gt_20_l63_63710


namespace min_games_needed_l63_63495

theorem min_games_needed (N : ℕ) : 
  (2 + N) * 10 ≥ 9 * (5 + N) ↔ N ≥ 25 := 
by {
  sorry
}

end min_games_needed_l63_63495


namespace time_for_one_kid_to_wash_six_whiteboards_l63_63438

-- Define the conditions as a function
def time_taken (k : ℕ) (w : ℕ) : ℕ := 20 * 4 * w / k

theorem time_for_one_kid_to_wash_six_whiteboards :
  time_taken 1 6 = 160 := by
-- Proof omitted
sorry

end time_for_one_kid_to_wash_six_whiteboards_l63_63438


namespace pictures_left_l63_63147

def initial_zoo_pics : ℕ := 49
def initial_museum_pics : ℕ := 8
def deleted_pics : ℕ := 38

theorem pictures_left (total_pics : ℕ) :
  total_pics = initial_zoo_pics + initial_museum_pics →
  total_pics - deleted_pics = 19 :=
by
  intro h1
  rw [h1]
  sorry

end pictures_left_l63_63147


namespace percentage_problem_l63_63689

theorem percentage_problem 
  (number : ℕ)
  (h1 : number = 6400)
  (h2 : 5 * number / 100 = 20 * 650 / 100 + 190) : 
  20 = 20 :=
by 
  sorry

end percentage_problem_l63_63689


namespace jackson_paintable_area_l63_63024

namespace PaintWallCalculation

def length := 14
def width := 11
def height := 9
def windowArea := 70
def bedrooms := 4

def area_one_bedroom : ℕ :=
  2 * (length * height) + 2 * (width * height)

def paintable_area_one_bedroom : ℕ :=
  area_one_bedroom - windowArea

def total_paintable_area : ℕ :=
  bedrooms * paintable_area_one_bedroom

theorem jackson_paintable_area :
  total_paintable_area = 1520 :=
sorry

end PaintWallCalculation

end jackson_paintable_area_l63_63024


namespace negation_of_p_l63_63060

variable {x : ℝ}

def p := ∀ x : ℝ, x^3 - x^2 + 1 < 0

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 := by
  sorry

end negation_of_p_l63_63060


namespace thelma_tomato_count_l63_63501

-- Definitions and conditions
def slices_per_tomato : ℕ := 8
def slices_per_meal_per_person : ℕ := 20
def family_members : ℕ := 8
def total_slices_needed : ℕ := slices_per_meal_per_person * family_members
def tomatoes_needed : ℕ := total_slices_needed / slices_per_tomato

-- Statement of the theorem to be proved
theorem thelma_tomato_count :
  tomatoes_needed = 20 := by
  sorry

end thelma_tomato_count_l63_63501


namespace find_a_l63_63138

theorem find_a (a : ℝ) (t : ℝ) :
  (4 = 1 + 3 * t) ∧ (3 = a * t^2 + 2) → a = 1 :=
by
  sorry

end find_a_l63_63138


namespace angle_A_and_shape_of_triangle_l63_63078

theorem angle_A_and_shape_of_triangle 
  (a b c : ℝ)
  (h1 : a^2 - c^2 = a * c - b * c)
  (h2 : ∃ r : ℝ, a = b * r ∧ c = b / r)
  (h3 : ∃ B C : Type, B = A ∧ C ≠ A ) :
  ∃ (A : ℝ), A = 60 ∧ a = b ∧ b = c := 
sorry

end angle_A_and_shape_of_triangle_l63_63078


namespace quadratic_always_real_roots_rhombus_area_when_m_minus_7_l63_63132

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : ℝ := 2 * x^2 + (m - 2) * x - m

-- Statement 1: For any real number m, the quadratic equation always has real roots.
theorem quadratic_always_real_roots (m : ℝ) : ∃ x1 x2 : ℝ, quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 :=
by {
  -- Proof omitted
  sorry
}

-- Statement 2: When m = -7, the area of the rhombus whose diagonals are the roots of the quadratic equation is 7/4.
theorem rhombus_area_when_m_minus_7 : (∃ x1 x2 : ℝ, quadratic_eq (-7) x1 = 0 ∧ quadratic_eq (-7) x2 = 0 ∧ (1 / 2) * x1 * x2 = 7 / 4) :=
by {
  -- Proof omitted
  sorry
}

end quadratic_always_real_roots_rhombus_area_when_m_minus_7_l63_63132


namespace simplify_expr1_simplify_and_evaluate_l63_63439

-- First problem: simplify and prove equality.
theorem simplify_expr1 (a : ℝ) :
  -2 * a^2 + 3 - (3 * a^2 - 6 * a + 1) + 3 = -5 * a^2 + 6 * a + 2 :=
by sorry

-- Second problem: simplify and evaluate under given conditions.
theorem simplify_and_evaluate (x y : ℝ) (h_x : x = -2) (h_y : y = -3) :
  (1 / 2) * x - 2 * (x - (1 / 3) * y^2) + (-(3 / 2) * x + (1 / 3) * y^2) = 15 :=
by sorry

end simplify_expr1_simplify_and_evaluate_l63_63439


namespace minimum_value_of_function_l63_63712

-- Define the function y = 2x + 1/(x - 1) with the constraint x > 1
noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / (x - 1)

-- Prove that the minimum value of the function for x > 1 is 2√2 + 2
theorem minimum_value_of_function : 
  ∃ x : ℝ, x > 1 ∧ ∀ y : ℝ, (y = f x) → y ≥ 2 * Real.sqrt 2 + 2 := 
  sorry

end minimum_value_of_function_l63_63712


namespace unit_circle_sector_arc_length_l63_63183

theorem unit_circle_sector_arc_length (r S l : ℝ) (h1 : r = 1) (h2 : S = 1) (h3 : S = 1 / 2 * l * r) : l = 2 :=
by
  sorry

end unit_circle_sector_arc_length_l63_63183


namespace verify_distinct_outcomes_l63_63990

def i : ℂ := Complex.I

theorem verify_distinct_outcomes :
  ∃! S, ∀ n : ℤ, n % 8 = n → S = i^n + i^(-n)
  := sorry

end verify_distinct_outcomes_l63_63990


namespace combined_salaries_BCDE_l63_63028

-- Define the given conditions
def salary_A : ℕ := 10000
def average_salary : ℕ := 8400
def num_individuals : ℕ := 5

-- Define the total salary of all individuals
def total_salary_all : ℕ := average_salary * num_individuals

-- Define the proof problem
theorem combined_salaries_BCDE : (total_salary_all - salary_A) = 32000 := by
  sorry

end combined_salaries_BCDE_l63_63028


namespace abs_eq_condition_l63_63658

theorem abs_eq_condition (a b : ℝ) : |a - b| = |a - 1| + |b - 1| ↔ (a - 1) * (b - 1) ≤ 0 :=
sorry

end abs_eq_condition_l63_63658


namespace binomial_square_correct_k_l63_63016

theorem binomial_square_correct_k (k : ℚ) : (∃ t u : ℚ, k = t^2 ∧ 28 = 2 * t * u ∧ 9 = u^2) → k = 196 / 9 :=
by
  sorry

end binomial_square_correct_k_l63_63016


namespace rational_roots_of_quadratic_l63_63809

theorem rational_roots_of_quadratic (r : ℚ) :
  (∃ a b : ℤ, a ≠ b ∧ (r * a^2 + (r + 1) * a + r = 1 ∧ r * b^2 + (r + 1) * b + r = 1)) ↔ (r = 1 ∨ r = -1 / 7) :=
by
  sorry

end rational_roots_of_quadratic_l63_63809


namespace larger_box_cost_l63_63681

-- Definitions based on the conditions

def ounces_large : ℕ := 30
def ounces_small : ℕ := 20
def cost_small : ℝ := 3.40
def price_per_ounce_better_value : ℝ := 0.16

-- The statement to prove
theorem larger_box_cost :
  30 * price_per_ounce_better_value = 4.80 :=
by sorry

end larger_box_cost_l63_63681


namespace triangle_obtuse_l63_63053

variable {a b c : ℝ}

theorem triangle_obtuse (h : 2 * c^2 = 2 * a^2 + 2 * b^2 + a * b) :
  ∃ C : ℝ, 0 ≤ C ∧ C ≤ π ∧ Real.cos C = -1/4 ∧ C > Real.pi / 2 :=
by
  sorry

end triangle_obtuse_l63_63053


namespace expected_yield_correct_l63_63112

-- Conditions
def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def step_length_ft : ℝ := 2.5
def yield_per_sqft_pounds : ℝ := 0.75

-- Related quantities
def garden_length_ft : ℝ := garden_length_steps * step_length_ft
def garden_width_ft : ℝ := garden_width_steps * step_length_ft
def garden_area_sqft : ℝ := garden_length_ft * garden_width_ft
def expected_yield_pounds : ℝ := garden_area_sqft * yield_per_sqft_pounds

-- Statement to prove
theorem expected_yield_correct : expected_yield_pounds = 2109.375 := by
  sorry

end expected_yield_correct_l63_63112


namespace remainder_7531_mod_11_is_5_l63_63881

theorem remainder_7531_mod_11_is_5 :
  let n := 7531
  let m := 7 + 5 + 3 + 1
  n % 11 = 5 ∧ m % 11 = 5 :=
by
  let n := 7531
  let m := 7 + 5 + 3 + 1
  have h : n % 11 = m % 11 := sorry  -- by property of digits sum mod
  have hm : m % 11 = 5 := sorry      -- calculation
  exact ⟨h, hm⟩

end remainder_7531_mod_11_is_5_l63_63881


namespace excircle_tangent_segment_length_l63_63207

theorem excircle_tangent_segment_length (A B C M : ℝ) 
  (h1 : A + B + C = 1) 
  (h2 : M = (1 / 2)) : 
  M = 1 / 2 := 
  by
    -- This is where the proof would go
    sorry

end excircle_tangent_segment_length_l63_63207


namespace map_distance_l63_63270

theorem map_distance (scale_cm : ℝ) (scale_km : ℝ) (actual_distance_km : ℝ) 
  (h1 : scale_cm = 0.4) (h2 : scale_km = 5.3) (h3 : actual_distance_km = 848) :
  actual_distance_km / (scale_km / scale_cm) = 64 :=
by
  rw [h1, h2, h3]
  -- Further steps would follow here, but to ensure code compiles
  -- and there is no assumption directly from solution steps, we use sorry.
  sorry

end map_distance_l63_63270


namespace arithmetic_sequence_sum_l63_63870

theorem arithmetic_sequence_sum {S : ℕ → ℤ} (m : ℕ) (hm : 0 < m)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_sum_l63_63870


namespace domain_of_expression_l63_63728

theorem domain_of_expression (x : ℝ) : 
  x + 3 ≥ 0 → 7 - x > 0 → (x ∈ Set.Icc (-3) 7) :=
by 
  intros h1 h2
  sorry

end domain_of_expression_l63_63728


namespace triangles_with_equal_angles_and_two_equal_sides_not_necessarily_congruent_l63_63355

-- Define the triangle type
structure Triangle :=
(SideA : ℝ)
(SideB : ℝ)
(SideC : ℝ)
(AngleA : ℝ)
(AngleB : ℝ)
(AngleC : ℝ)
(h1 : SideA > 0)
(h2 : SideB > 0)
(h3 : SideC > 0)
(h4 : AngleA + AngleB + AngleC = 180)

-- Define what it means for two triangles to have three equal angles
def have_equal_angles (T1 T2 : Triangle) : Prop :=
(T1.AngleA = T2.AngleA ∧ T1.AngleB = T2.AngleB ∧ T1.AngleC = T2.AngleC)

-- Define what it means for two triangles to have two equal sides
def have_two_equal_sides (T1 T2 : Triangle) : Prop :=
(T1.SideA = T2.SideA ∧ T1.SideB = T2.SideB) ∨
(T1.SideA = T2.SideA ∧ T1.SideC = T2.SideC) ∨
(T1.SideB = T2.SideB ∧ T1.SideC = T2.SideC)

-- Define what it means for two triangles to be congruent
def congruent (T1 T2 : Triangle) : Prop :=
(T1.SideA = T2.SideA ∧ T1.SideB = T2.SideB ∧ T1.SideC = T2.SideC ∧
 T1.AngleA = T2.AngleA ∧ T1.AngleB = T2.AngleB ∧ T1.AngleC = T2.AngleC)

-- The final theorem
theorem triangles_with_equal_angles_and_two_equal_sides_not_necessarily_congruent 
  (T1 T2 : Triangle) 
  (h_angles : have_equal_angles T1 T2)
  (h_sides : have_two_equal_sides T1 T2) : ¬ congruent T1 T2 :=
sorry

end triangles_with_equal_angles_and_two_equal_sides_not_necessarily_congruent_l63_63355


namespace unique_positive_integers_abc_l63_63230

def coprime (a b : ℕ) := Nat.gcd a b = 1

def allPrimeDivisorsNotCongruentTo1Mod7 (n : ℕ) := 
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p % 7 ≠ 1

theorem unique_positive_integers_abc :
  ∀ a b c : ℕ,
    (1 ≤ a) →
    (1 ≤ b) →
    (1 ≤ c) →
    coprime a b →
    coprime b c →
    coprime c a →
    (a * a + b) ∣ (b * b + c) →
    (b * b + c) ∣ (c * c + a) →
    allPrimeDivisorsNotCongruentTo1Mod7 (a * a + b) →
  a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end unique_positive_integers_abc_l63_63230


namespace intervals_of_monotonicity_when_a_eq_2_no_increasing_intervals_on_1_3_implies_a_ge_19_over_6_l63_63299

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + x^2 - 2 * a * x + a^2

-- Question Ⅰ
theorem intervals_of_monotonicity_when_a_eq_2 :
  (∀ x : ℝ, 0 < x ∧ x < (2 - Real.sqrt 2) / 2 → f x 2 > 0) ∧
  (∀ x : ℝ, (2 - Real.sqrt 2) / 2 < x ∧ x < (2 + Real.sqrt 2) / 2 → f x 2 < 0) ∧
  (∀ x : ℝ, (2 + Real.sqrt 2) / 2 < x → f x 2 > 0) := sorry

-- Question Ⅱ
theorem no_increasing_intervals_on_1_3_implies_a_ge_19_over_6 (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 0) → a ≥ (19 / 6) := sorry

end intervals_of_monotonicity_when_a_eq_2_no_increasing_intervals_on_1_3_implies_a_ge_19_over_6_l63_63299


namespace gcd_max_value_l63_63295

theorem gcd_max_value (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 1005) : ∃ d, d = Int.gcd a b ∧ d = 335 :=
by {
  sorry
}

end gcd_max_value_l63_63295


namespace no_four_distinct_real_roots_l63_63674

theorem no_four_distinct_real_roots (a b : ℝ) :
  ¬ ∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
  (x1^4 - 4*x1^3 + 6*x1^2 + a*x1 + b = 0) ∧
  (x2^4 - 4*x2^3 + 6*x2^2 + a*x2 + b = 0) ∧
  (x3^4 - 4*x3^3 + 6*x3^2 + a*x3 + b = 0) ∧
  (x4^4 - 4*x4^3 + 6*x4^2 + a*x4 + b = 0) := 
by {
  sorry
}

end no_four_distinct_real_roots_l63_63674


namespace unique_seating_arrangements_l63_63725

/--
There are five couples including Charlie and his wife. The five men sit on the 
inner circle and each man's wife sits directly opposite him on the outer circle.
Prove that the number of unique seating arrangements where each man has another 
man seated directly to his right on the inner circle, counting all seat 
rotations as the same but not considering inner to outer flips as different, is 30.
-/
theorem unique_seating_arrangements : 
  ∃ (n : ℕ), n = 30 := 
sorry

end unique_seating_arrangements_l63_63725


namespace range_of_a4_l63_63176

noncomputable def geometric_sequence (a1 a2 a3 : ℝ) (q : ℝ) (a4 : ℝ) : Prop :=
  ∃ (a1 q : ℝ), 0 < a1 ∧ a1 < 1 ∧ 
                1 < a1 * q ∧ a1 * q < 2 ∧ 
                2 < a1 * q^2 ∧ a1 * q^2 < 4 ∧ 
                a4 = (a1 * q^2) * q ∧ 
                2 * Real.sqrt 2 < a4 ∧ a4 < 16

theorem range_of_a4 (a1 a2 a3 a4 : ℝ) (q : ℝ) (h1 : 0 < a1) (h2 : a1 < 1) 
  (h3 : 1 < a2) (h4 : a2 < 2) (h5 : a2 = a1 * q)
  (h6 : 2 < a3) (h7 : a3 < 4) (h8 : a3 = a1 * q^2) :
  2 * Real.sqrt 2 < a4 ∧ a4 < 16 :=
by
  have hq1 : 2 * q^2 < 1 := sorry    -- Placeholder for necessary inequalities
  have hq2: 1 < q ∧ q < 4 := sorry   -- Placeholder for necessary inequalities
  sorry

end range_of_a4_l63_63176


namespace arithmetic_sequence_sum_property_l63_63912

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ)  -- sequence terms are real numbers
  (d : ℝ)      -- common difference
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum_condition : a 4 + a 8 = 16) :
  a 2 + a 10 = 16 :=
sorry

end arithmetic_sequence_sum_property_l63_63912


namespace solve_for_wood_length_l63_63637

theorem solve_for_wood_length (y x : ℝ) (h1 : y - x = 4.5) (h2 : x - (1/2) * y = 1) :
  ∃! (x y : ℝ), (y - x = 4.5) ∧ (x - (1/2) * y = 1) :=
by
  -- The content of the proof is omitted
  sorry

end solve_for_wood_length_l63_63637


namespace value_of_D_l63_63784

theorem value_of_D (E F D : ℕ) (cond1 : E + F + D = 15) (cond2 : F + E = 11) : D = 4 := 
by
  sorry

end value_of_D_l63_63784


namespace exists_two_people_with_property_l63_63302

theorem exists_two_people_with_property (n : ℕ) (P : Fin (2 * n + 2) → Fin (2 * n + 2) → Prop) :
  ∃ A B : Fin (2 * n + 2), 
    A ≠ B ∧
    (∃ S : Finset (Fin (2 * n + 2)), 
      S.card = n ∧
      ∀ C ∈ S, (P C A ∧ P C B) ∨ (¬P C A ∧ ¬P C B)) :=
sorry

end exists_two_people_with_property_l63_63302


namespace original_selling_price_is_440_l63_63444

variable (P : ℝ)

-- Condition: Bill made a profit of 10% by selling a product.
def original_selling_price := 1.10 * P

-- Condition: He had purchased the product for 10% less.
def new_purchase_price := 0.90 * P

-- Condition: With a 30% profit on the new purchase price, the new selling price.
def new_selling_price := 1.17 * P

-- Condition: The new selling price is $28 more than the original selling price.
def price_difference_condition : Prop := new_selling_price P = original_selling_price P + 28

-- Conclusion: The original selling price was \$440
theorem original_selling_price_is_440
    (h : price_difference_condition P) : original_selling_price P = 440 :=
sorry

end original_selling_price_is_440_l63_63444


namespace remaining_yards_correct_l63_63600

-- Define the conversion constant
def yards_per_mile: ℕ := 1760

-- Define the conditions
def marathon_in_miles: ℕ := 26
def marathon_in_yards: ℕ := 395
def total_marathons: ℕ := 15

-- Define the function to calculate the remaining yards after conversion
def calculate_remaining_yards (marathon_in_miles marathon_in_yards total_marathons yards_per_mile: ℕ): ℕ :=
  let total_yards := total_marathons * marathon_in_yards
  total_yards % yards_per_mile

-- Statement to prove
theorem remaining_yards_correct :
  calculate_remaining_yards marathon_in_miles marathon_in_yards total_marathons yards_per_mile = 645 :=
  sorry

end remaining_yards_correct_l63_63600


namespace area_increase_cost_increase_l63_63614

-- Given definitions based only on the conditions from part a
def original_length := 60
def original_width := 20
def original_fence_cost_per_foot := 15
def original_perimeter := 2 * (original_length + original_width)
def original_fencing_cost := original_perimeter * original_fence_cost_per_foot

def new_fence_cost_per_foot := 20
def new_square_side := original_perimeter / 4
def new_square_area := new_square_side * new_square_side
def new_fencing_cost := original_perimeter * new_fence_cost_per_foot

-- Proof statements using the conditions and correct answers from part b
theorem area_increase : new_square_area - (original_length * original_width) = 400 := by
  sorry

theorem cost_increase : new_fencing_cost - original_fencing_cost = 800 := by
  sorry

end area_increase_cost_increase_l63_63614


namespace Tim_is_65_l63_63675

def James_age : Nat := 23
def John_age : Nat := 35
def Tim_age : Nat := 2 * John_age - 5

theorem Tim_is_65 : Tim_age = 65 := by
  sorry

end Tim_is_65_l63_63675


namespace original_proposition_inverse_proposition_converse_proposition_contrapositive_proposition_l63_63345

variable (a_n : ℕ → ℝ) (n : ℕ+)

-- To prove the original proposition
theorem original_proposition : (a_n n + a_n (n + 1)) / 2 < a_n n → (∀ m, a_n m ≥ a_n (m + 1)) := 
sorry

-- To prove the inverse proposition
theorem inverse_proposition : ((a_n n + a_n (n + 1)) / 2 ≥ a_n n → ¬ ∀ m, a_n m ≥ a_n (m + 1)) := 
sorry

-- To prove the converse proposition
theorem converse_proposition : (∀ m, a_n m ≥ a_n (m + 1)) → (a_n n + a_n (n + 1)) / 2 < a_n n := 
sorry

-- To prove the contrapositive proposition
theorem contrapositive_proposition : (¬ ∀ m, a_n m ≥ a_n (m + 1)) → (a_n n + a_n (n + 1)) / 2 ≥ a_n n :=
sorry

end original_proposition_inverse_proposition_converse_proposition_contrapositive_proposition_l63_63345


namespace scientific_notation_of_53_96_billion_l63_63331

theorem scientific_notation_of_53_96_billion :
  (53.96 * 10^9) = (5.396 * 10^10) :=
sorry

end scientific_notation_of_53_96_billion_l63_63331


namespace tax_difference_is_correct_l63_63753

-- Define the original price and discount rate as constants
def original_price : ℝ := 50
def discount_rate : ℝ := 0.10

-- Define the state and local sales tax rates as constants
def state_sales_tax_rate : ℝ := 0.075
def local_sales_tax_rate : ℝ := 0.07

-- Calculate the discounted price
def discounted_price : ℝ := original_price * (1 - discount_rate)

-- Calculate state and local sales taxes after discount
def state_sales_tax : ℝ := discounted_price * state_sales_tax_rate
def local_sales_tax : ℝ := discounted_price * local_sales_tax_rate

-- Calculate the difference between state and local sales taxes
def tax_difference : ℝ := state_sales_tax - local_sales_tax

-- The proof to show that the difference is 0.225
theorem tax_difference_is_correct : tax_difference = 0.225 := by
  sorry

end tax_difference_is_correct_l63_63753


namespace processing_time_600_parts_l63_63727

theorem processing_time_600_parts :
  ∀ (x: ℕ), x = 600 → (∃ y : ℝ, y = 0.01 * x + 0.5 ∧ y = 6.5) :=
by
  sorry

end processing_time_600_parts_l63_63727


namespace find_YZ_l63_63829

noncomputable def triangle_YZ (angle_Y : ℝ) (XY : ℝ) (XZ : ℝ) : ℝ :=
  if angle_Y = 45 ∧ XY = 100 ∧ XZ = 50 * Real.sqrt 2 then
    50 * Real.sqrt 6
  else
    0

theorem find_YZ :
  triangle_YZ 45 100 (50 * Real.sqrt 2) = 50 * Real.sqrt 6 :=
by
  sorry

end find_YZ_l63_63829


namespace quadratic_equation_solution_l63_63258

-- Define the problem statement and the conditions: the equation being quadratic.
theorem quadratic_equation_solution (m : ℤ) :
  (∃ (a : ℤ), a ≠ 0 ∧ (a*x^2 - x - 2 = 0)) →
  m = -1 :=
by
  sorry

end quadratic_equation_solution_l63_63258


namespace Larry_wins_game_probability_l63_63659

noncomputable def winning_probability_Larry : ℚ :=
  ∑' n : ℕ, if n % 3 = 0 then (2 / 3) ^ (n / 3 * 3) * (1 / 3) else 0

theorem Larry_wins_game_probability : winning_probability_Larry = 9 / 19 :=
by
  sorry

end Larry_wins_game_probability_l63_63659


namespace local_min_f_at_2_implies_a_eq_2_l63_63875

theorem local_min_f_at_2_implies_a_eq_2 (a : ℝ) : 
  (∃ f : ℝ → ℝ, 
     (∀ x : ℝ, f x = x * (x - a)^2) ∧ 
     (∀ f' : ℝ → ℝ, 
       (∀ x : ℝ, f' x = 3 * x^2 - 4 * a * x + a^2) ∧ 
       f' 2 = 0 ∧ 
       (∀ f'' : ℝ → ℝ, 
         (∀ x : ℝ, f'' x = 6 * x - 4 * a) ∧ 
         f'' 2 > 0
       )
     )
  ) → a = 2 :=
sorry

end local_min_f_at_2_implies_a_eq_2_l63_63875


namespace pyramid_volume_l63_63062

noncomputable def volume_of_pyramid (l : ℝ) : ℝ :=
  (l^3 / 24) * (Real.sqrt (Real.sqrt 2 + 1))

theorem pyramid_volume (l : ℝ) (α β : ℝ)
  (hα : α = π / 8)
  (hβ : β = π / 4)
  (hl : l = 6) :
  volume_of_pyramid l = 9 * Real.sqrt (Real.sqrt 2 + 1) := by
  sorry

end pyramid_volume_l63_63062


namespace rectangle_perimeter_l63_63479

noncomputable def perimeter (a b c : ℕ) : ℕ :=
  2 * (a + b)

theorem rectangle_perimeter (p q: ℕ) (rel_prime: Nat.gcd p q = 1) :
  ∃ (a b c: ℕ), p = 2 * (a + b) ∧ p + q = 52 ∧ a = 5 ∧ b = 12 ∧ c = 7 :=
by
  sorry

end rectangle_perimeter_l63_63479


namespace determine_pairs_of_positive_integers_l63_63378

open Nat

theorem determine_pairs_of_positive_integers (n p : ℕ) (hp : Nat.Prime p) (hn_le_2p : n ≤ 2 * p)
    (hdiv : (p - 1)^n + 1 ∣ n^(p - 1)) : (n = 1) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
  sorry

end determine_pairs_of_positive_integers_l63_63378


namespace total_dots_not_visible_l63_63565

-- Define the conditions and variables
def total_dots_one_die : Nat := 1 + 2 + 3 + 4 + 5 + 6
def number_of_dice : Nat := 4
def total_dots_all_dice : Nat := number_of_dice * total_dots_one_die
def visible_numbers : List Nat := [6, 6, 4, 4, 3, 2, 1]

-- The question can be formalized as proving that the total number of dots not visible is 58
theorem total_dots_not_visible :
  total_dots_all_dice - visible_numbers.sum = 58 :=
by
  -- Statement only, proof skipped
  sorry

end total_dots_not_visible_l63_63565


namespace base_7_to_base_10_equiv_l63_63711

theorem base_7_to_base_10_equiv (digits : List ℕ) 
  (h : digits = [5, 4, 3, 2, 1]) : 
  (5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0) = 13539 := 
by 
  sorry

end base_7_to_base_10_equiv_l63_63711


namespace no_such_p_l63_63468

theorem no_such_p : ¬ ∃ p : ℕ, p > 0 ∧ (∃ k : ℤ, 4 * p + 35 = k * (3 * p - 7)) :=
by
  sorry

end no_such_p_l63_63468


namespace transform_to_zero_set_l63_63453

def S (p : ℕ) : Finset ℕ := Finset.range p

def P (p : ℕ) (x : ℕ) : ℕ := 3 * x ^ ((2 * p - 1) / 3) + x ^ ((p + 1) / 3) + x + 1

def remainder (n p : ℕ) : ℕ := n % p

theorem transform_to_zero_set (p k : ℕ) (hp : Nat.Prime p) (h_cong : p % 3 = 2) (hk : 0 < k) :
  (∃ n : ℕ, ∀ i ∈ S p, remainder (P p i) p = n) ∨ (∃ n : ℕ, ∀ i ∈ S p, remainder (i ^ k) p = n) ↔
  Nat.gcd k (p - 1) > 1 :=
sorry

end transform_to_zero_set_l63_63453


namespace problem2_l63_63398

theorem problem2 (x y : ℝ) (h1 : x^2 + x * y = 3) (h2 : x * y + y^2 = -2) : 
  2 * x^2 - x * y - 3 * y^2 = 12 := 
by 
  sorry

end problem2_l63_63398


namespace sum_of_fractions_l63_63099

theorem sum_of_fractions :
  (3 / 9) + (6 / 12) = 5 / 6 := by
  sorry

end sum_of_fractions_l63_63099


namespace correct_blanks_l63_63558

def fill_in_blanks (category : String) (plural_noun : String) : String :=
  "For many, winning remains " ++ category ++ " dream, but they continue trying their luck as there're always " ++ plural_noun ++ " chances that they might succeed."

theorem correct_blanks :
  fill_in_blanks "a" "" = "For many, winning remains a dream, but they continue trying their luck as there're always chances that they might succeed." :=
sorry

end correct_blanks_l63_63558


namespace sum_of_integers_with_product_neg13_l63_63825

theorem sum_of_integers_with_product_neg13 (a b c : ℤ) (h : a * b * c = -13) : 
  a + b + c = 13 ∨ a + b + c = -11 := 
sorry

end sum_of_integers_with_product_neg13_l63_63825


namespace jane_earnings_l63_63894

def earnings_per_bulb : ℝ := 0.50
def tulip_bulbs : ℕ := 20
def iris_bulbs : ℕ := tulip_bulbs / 2
def daffodil_bulbs : ℕ := 30
def crocus_bulbs : ℕ := daffodil_bulbs * 3
def total_earnings : ℝ := (tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs) * earnings_per_bulb

theorem jane_earnings : total_earnings = 75.0 := by
  sorry

end jane_earnings_l63_63894


namespace weights_difference_l63_63001

-- Definitions based on conditions
def A : ℕ := 36
def ratio_part : ℕ := A / 4
def B : ℕ := 5 * ratio_part
def C : ℕ := 6 * ratio_part

-- Theorem to prove
theorem weights_difference :
  (A + C) - B = 45 := by
  sorry

end weights_difference_l63_63001


namespace center_digit_is_two_l63_63947

theorem center_digit_is_two :
  ∃ (a b : ℕ), (a^2 < 1000 ∧ b^2 < 1000 ∧ (a^2 ≠ b^2) ∧
  (∀ d, d ∈ [a^2 / 100, (a^2 / 10) % 10, a^2 % 10] → d ∈ [2, 3, 4, 5, 6]) ∧
  (∀ d, d ∈ [b^2 / 100, (b^2 / 10) % 10, b^2 % 10] → d ∈ [2, 3, 4, 5, 6])) ∧
  (∀ d, (d ∈ [2, 3, 4, 5, 6]) → (d ∈ [a^2 / 100, (a^2 / 10) % 10, a^2 % 10] ∨ d ∈ [b^2 / 100, (b^2 / 10) % 10, b^2 % 10])) ∧
  2 = (a^2 / 10) % 10 ∨ 2 = (b^2 / 10) % 10 :=
sorry -- no proof needed, just the statement

end center_digit_is_two_l63_63947


namespace ratio_geometric_sequence_of_arithmetic_l63_63313

variable {d : ℤ}
variable {a : ℕ → ℤ}

-- definition of an arithmetic sequence with common difference d
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- definition of a geometric sequence for a_5, a_9, a_{15}
def geometric_sequence (a : ℕ → ℤ) : Prop :=
  a 9 * a 9 = a 5 * a 15

theorem ratio_geometric_sequence_of_arithmetic
  (h_arith : arithmetic_sequence a d) (h_nonzero : d ≠ 0) (h_geom : geometric_sequence a) :
  a 15 / a 9 = 3 / 2 :=
by
  sorry

end ratio_geometric_sequence_of_arithmetic_l63_63313


namespace breadth_of_rectangular_plot_l63_63008

theorem breadth_of_rectangular_plot (b : ℝ) (h1 : ∃ l : ℝ, l = 3 * b) (h2 : b * 3 * b = 675) : b = 15 :=
by
  sorry

end breadth_of_rectangular_plot_l63_63008


namespace solve_equation1_solve_equation2_l63_63806

open Real

theorem solve_equation1 (x : ℝ) : (x - 2)^2 = 9 → (x = 5 ∨ x = -1) :=
by
  intro h
  sorry -- Proof would go here

theorem solve_equation2 (x : ℝ) : (2 * x^2 - 3 * x - 1 = 0) → (x = (3 + sqrt 17) / 4 ∨ x = (3 - sqrt 17) / 4) :=
by
  intro h
  sorry -- Proof would go here

end solve_equation1_solve_equation2_l63_63806


namespace side_length_square_correct_l63_63943

noncomputable def side_length_square (time_seconds : ℕ) (speed_kmph : ℕ) : ℕ := sorry

theorem side_length_square_correct (time_seconds : ℕ) (speed_kmph : ℕ) (h_time : time_seconds = 24) 
  (h_speed : speed_kmph = 12) : side_length_square time_seconds speed_kmph = 20 :=
sorry

end side_length_square_correct_l63_63943


namespace percentage_of_cars_on_monday_compared_to_tuesday_l63_63752

theorem percentage_of_cars_on_monday_compared_to_tuesday : 
  ∀ (cars_mon cars_tue cars_wed cars_thu cars_fri cars_sat cars_sun : ℕ),
    cars_mon + cars_tue + cars_wed + cars_thu + cars_fri + cars_sat + cars_sun = 97 →
    cars_tue = 25 →
    cars_wed = cars_mon + 2 →
    cars_thu = 10 →
    cars_fri = 10 →
    cars_sat = 5 →
    cars_sun = 5 →
    (cars_mon * 100 / cars_tue = 80) :=
by
  intros cars_mon cars_tue cars_wed cars_thu cars_fri cars_sat cars_sun
  intro h_total
  intro h_tue
  intro h_wed
  intro h_thu
  intro h_fri
  intro h_sat
  intro h_sun
  sorry

end percentage_of_cars_on_monday_compared_to_tuesday_l63_63752


namespace factorization_correct_l63_63811

theorem factorization_correct (x : ℝ) : 2 * x ^ 2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end factorization_correct_l63_63811


namespace p_is_necessary_but_not_sufficient_for_q_l63_63496

-- Conditions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x - a ≥ 0
def q (a : ℝ) : Prop := -1 < a ∧ a < 0

-- Proof target
theorem p_is_necessary_but_not_sufficient_for_q : 
  (∀ a : ℝ, p a → q a) ∧ ¬(∀ a : ℝ, q a → p a) :=
sorry

end p_is_necessary_but_not_sufficient_for_q_l63_63496


namespace denis_sum_of_numbers_l63_63332

theorem denis_sum_of_numbers :
  ∃ a b c d : ℕ, a < b ∧ b < c ∧ c < d ∧ a*d = 32 ∧ b*c = 14 ∧ a + b + c + d = 42 :=
sorry

end denis_sum_of_numbers_l63_63332


namespace sum_of_distinct_integers_l63_63312

theorem sum_of_distinct_integers (a b c d : ℤ) (h : (a - 1) * (b - 1) * (c - 1) * (d - 1) = 25) (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : a + b + c + d = 4 :=
by
    sorry

end sum_of_distinct_integers_l63_63312


namespace inverse_f_neg_3_l63_63201

def f (x : ℝ) : ℝ := 5 - 2 * x

theorem inverse_f_neg_3 : (∃ x : ℝ, f x = -3) ∧ (f 4 = -3) :=
by
  sorry

end inverse_f_neg_3_l63_63201


namespace divisibility_of_n_squared_plus_n_plus_two_l63_63713

-- Definition: n is a natural number.
def n (n : ℕ) : Prop := True

-- Theorem: For any natural number n, n^2 + n + 2 is always divisible by 2, but not necessarily divisible by 5.
theorem divisibility_of_n_squared_plus_n_plus_two (n : ℕ) : 
  (∃ k : ℕ, n^2 + n + 2 = 2 * k) ∧ (¬ ∃ m : ℕ, n^2 + n + 2 = 5 * m) :=
by
  sorry

end divisibility_of_n_squared_plus_n_plus_two_l63_63713


namespace ben_paints_150_square_feet_l63_63123

-- Define the given conditions
def ratio_allen_ben : ℕ := 3
def ratio_ben_allen : ℕ := 5
def total_work : ℕ := 240

-- Define the total amount of parts
def total_parts : ℕ := ratio_allen_ben + ratio_ben_allen

-- Define the work per part
def work_per_part : ℕ := total_work / total_parts

-- Define the work done by Ben
def ben_parts : ℕ := ratio_ben_allen
def ben_work : ℕ := work_per_part * ben_parts

-- The statement to be proved
theorem ben_paints_150_square_feet : ben_work = 150 :=
by
  sorry

end ben_paints_150_square_feet_l63_63123


namespace work_done_is_halved_l63_63896

theorem work_done_is_halved
  (A₁₂ A₃₄ : ℝ)
  (isothermal_process : ∀ (p V₁₂ V₃₄ : ℝ), V₁₂ = 2 * V₃₄ → p * V₁₂ = A₁₂ → p * V₃₄ = A₃₄) :
  A₃₄ = (1 / 2) * A₁₂ :=
sorry

end work_done_is_halved_l63_63896


namespace power_subtraction_divisibility_l63_63599

theorem power_subtraction_divisibility (N : ℕ) (h : N > 1) : 
  ∃ k : ℕ, (N^2)^2014 - (N^11)^106 = k * (N^6 + N^3 + 1) :=
by
  sorry

end power_subtraction_divisibility_l63_63599


namespace sixth_power_sum_l63_63254

/-- Given:
     (1) a + b = 1
     (2) a^2 + b^2 = 3
     (3) a^3 + b^3 = 4
     (4) a^4 + b^4 = 7
     (5) a^5 + b^5 = 11
    Prove:
     a^6 + b^6 = 18 -/
theorem sixth_power_sum (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 :=
sorry

end sixth_power_sum_l63_63254


namespace intersection_point_unique_l63_63382

theorem intersection_point_unique (k : ℝ) :
  (∃ y : ℝ, k = -2 * y^2 - 3 * y + 5) ∧ (∀ y₁ y₂ : ℝ, y₁ ≠ y₂ → -2 * y₁^2 - 3 * y₁ + 5 ≠ k ∨ -2 * y₂^2 - 3 * y₂ + 5 ≠ k)
  ↔ k = 49 / 8 := 
by sorry

end intersection_point_unique_l63_63382


namespace find_original_radius_l63_63577

theorem find_original_radius (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 2) / 2 :=
by
  sorry

end find_original_radius_l63_63577


namespace base_eight_to_ten_l63_63175

theorem base_eight_to_ten (n : Nat) (h : n = 52) : 8 * 5 + 2 = 42 :=
by
  -- Proof will be written here.
  sorry

end base_eight_to_ten_l63_63175


namespace greatest_n_divides_l63_63105

theorem greatest_n_divides (m : ℕ) (hm : 0 < m) : 
  ∃ n : ℕ, (n = m^4 - m^2 + m) ∧ (m^2 + n) ∣ (n^2 + m) := 
by {
  sorry
}

end greatest_n_divides_l63_63105


namespace shaded_area_calculation_l63_63484

noncomputable section

-- Definition of the total area of the grid
def total_area (rows columns : ℕ) : ℝ :=
  rows * columns

-- Definition of the area of a right triangle
def triangle_area (base height : ℕ) : ℝ :=
  1 / 2 * base * height

-- Definition of the shaded area in the grid
def shaded_area (total_area triangle_area : ℝ) : ℝ :=
  total_area - triangle_area

-- Theorem stating the shaded area
theorem shaded_area_calculation :
  let rows := 4
  let columns := 13
  let height := 3
  shaded_area (total_area rows columns) (triangle_area columns height) = 32.5 :=
  sorry

end shaded_area_calculation_l63_63484


namespace evaluate_expression_is_sixth_l63_63004

noncomputable def evaluate_expression := (1 / Real.log 3000^4 / Real.log 8) + (4 / Real.log 3000^4 / Real.log 9)

theorem evaluate_expression_is_sixth:
  evaluate_expression = 1 / 6 :=
  by
  sorry

end evaluate_expression_is_sixth_l63_63004


namespace slope_correct_l63_63601

-- Coordinates of the vertices of the polygon
def vertex_A := (0, 0)
def vertex_B := (0, 4)
def vertex_C := (4, 4)
def vertex_D := (4, 2)
def vertex_E := (6, 2)
def vertex_F := (6, 0)

-- Define the total area of the polygon
def total_area : ℝ := 20

-- Define the slope of the line through the origin dividing the area in half
def slope_line_dividing_area (slope : ℝ) : Prop :=
  ∃ l : ℝ, l = 5 / 3 ∧
  ∃ area_divided : ℝ, area_divided = total_area / 2

-- Prove the slope is 5/3
theorem slope_correct :
  slope_line_dividing_area (5 / 3) :=
by
  sorry

end slope_correct_l63_63601


namespace charts_per_associate_professor_l63_63928

theorem charts_per_associate_professor (A B C : ℕ) 
  (h1 : A + B = 6) 
  (h2 : 2 * A + B = 10) 
  (h3 : C * A + 2 * B = 8) : 
  C = 1 :=
by
  sorry

end charts_per_associate_professor_l63_63928


namespace ratio_3_7_not_possible_l63_63163

theorem ratio_3_7_not_possible (n : ℕ) (h : 30 < n ∧ n < 40) :
  ¬ (∃ k : ℕ, n = 10 * k) :=
by {
  sorry
}

end ratio_3_7_not_possible_l63_63163


namespace product_of_sisters_and_brothers_l63_63613

-- Lucy's family structure
def lucy_sisters : ℕ := 4
def lucy_brothers : ℕ := 6

-- Liam's siblings count
def liam_sisters : ℕ := lucy_sisters + 1  -- Including Lucy herself
def liam_brothers : ℕ := lucy_brothers    -- Excluding himself

-- Prove the product of Liam's sisters and brothers is 25
theorem product_of_sisters_and_brothers : liam_sisters * (liam_brothers - 1) = 25 :=
by
  sorry

end product_of_sisters_and_brothers_l63_63613


namespace sequence_term_general_sequence_sum_term_general_l63_63879

theorem sequence_term_general (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S (n + 1) = 2 * S n + 1) →
  a 1 = 1 →
  (∀ n ≥ 1, a n = 2^(n-1)) :=
  sorry

theorem sequence_sum_term_general (na : ℕ → ℕ) (T : ℕ → ℕ) :
  (∀ k, na k = k * 2^(k-1)) →
  (∀ n, T n = (n - 1) * 2^n + 1) :=
  sorry

end sequence_term_general_sequence_sum_term_general_l63_63879


namespace perpendicular_lines_have_given_slope_l63_63089

theorem perpendicular_lines_have_given_slope (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 3 = 0 → x + (2 * k + 3) * y - 2 = 0) →
  k = -3 :=
by
  sorry

end perpendicular_lines_have_given_slope_l63_63089


namespace remainder_is_6910_l63_63800

def polynomial (x : ℝ) : ℝ := 5 * x^7 - 3 * x^6 - 8 * x^5 + 3 * x^3 + 5 * x^2 - 20

def divisor (x : ℝ) : ℝ := 3 * x - 9

theorem remainder_is_6910 : polynomial 3 = 6910 := by
  sorry

end remainder_is_6910_l63_63800


namespace alice_bob_coffee_shop_spending_l63_63544

theorem alice_bob_coffee_shop_spending (A B : ℝ) (h1 : B = 0.5 * A) (h2 : A = B + 15) : A + B = 45 :=
by
  sorry

end alice_bob_coffee_shop_spending_l63_63544


namespace no_matching_formula_l63_63122

def formula_A (x : ℕ) : ℕ := 4 * x - 2
def formula_B (x : ℕ) : ℕ := x^3 - x^2 + 2 * x
def formula_C (x : ℕ) : ℕ := 2 * x^2
def formula_D (x : ℕ) : ℕ := x^2 + 2 * x + 1

theorem no_matching_formula :
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_A x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_B x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_C x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_D x)
  :=
by
  sorry

end no_matching_formula_l63_63122


namespace find_triplets_l63_63325

theorem find_triplets (u v w : ℝ):
  (u + v * w = 12) ∧ 
  (v + w * u = 12) ∧ 
  (w + u * v = 12) ↔ 
  (u = 3 ∧ v = 3 ∧ w = 3) ∨ 
  (u = -4 ∧ v = -4 ∧ w = -4) ∨ 
  (u = 1 ∧ v = 1 ∧ w = 11) ∨ 
  (u = 11 ∧ v = 1 ∧ w = 1) ∨ 
  (u = 1 ∧ v = 11 ∧ w = 1) := 
sorry

end find_triplets_l63_63325


namespace problem1_problem2_l63_63414

theorem problem1 : -20 + 3 + 5 - 7 = -19 := by
  sorry

theorem problem2 : (-3)^2 * 5 + (-2)^3 / 4 - |-3| = 40 := by
  sorry

end problem1_problem2_l63_63414


namespace sqrt_400_div_2_l63_63498

theorem sqrt_400_div_2 : (Nat.sqrt 400) / 2 = 10 := by
  sorry

end sqrt_400_div_2_l63_63498


namespace airline_passenger_capacity_l63_63852

def seats_per_row : Nat := 7
def rows_per_airplane : Nat := 20
def airplanes_owned : Nat := 5
def flights_per_day_per_airplane : Nat := 2

def seats_per_airplane : Nat := rows_per_airplane * seats_per_row
def total_seats : Nat := airplanes_owned * seats_per_airplane
def total_flights_per_day : Nat := airplanes_owned * flights_per_day_per_airplane
def total_passengers_per_day : Nat := total_flights_per_day * total_seats

theorem airline_passenger_capacity :
  total_passengers_per_day = 7000 := sorry

end airline_passenger_capacity_l63_63852


namespace max_value_seq_l63_63404

noncomputable def a_n (n : ℕ) : ℝ := n / (n^2 + 90)

theorem max_value_seq : ∃ n : ℕ, a_n n = 1 / 19 :=
by
  sorry

end max_value_seq_l63_63404


namespace number_of_values_f3_sum_of_values_f3_product_of_n_and_s_l63_63984

def S := { x : ℝ // x ≠ 0 }

def f (x : S) : S := sorry

lemma functional_equation (x y : S) (h : (x.val + y.val) ≠ 0) :
  (f x).val + (f y).val = (f ⟨(x.val * y.val) / (x.val + y.val) * (f ⟨x.val + y.val, sorry⟩).val, sorry⟩).val := sorry

-- Prove that the number of possible values of f(3) is 1

theorem number_of_values_f3 : ∃ n : ℕ, n = 1 := sorry

-- Prove that the sum of all possible values of f(3) is 1/3

theorem sum_of_values_f3 : ∃ s : ℚ, s = 1/3 := sorry

-- Prove that n * s = 1/3

theorem product_of_n_and_s (n : ℕ) (s : ℚ) (hn : n = 1) (hs : s = 1/3) : n * s = 1/3 := by
  rw [hn, hs]
  norm_num

end number_of_values_f3_sum_of_values_f3_product_of_n_and_s_l63_63984


namespace shopper_total_payment_l63_63273

theorem shopper_total_payment :
  let original_price := 150
  let discount_rate := 0.25
  let coupon_discount := 10
  let sales_tax_rate := 0.10
  let discounted_price := original_price * (1 - discount_rate)
  let price_after_coupon := discounted_price - coupon_discount
  let final_price := price_after_coupon * (1 + sales_tax_rate)
  final_price = 112.75 := by
{
  sorry
}

end shopper_total_payment_l63_63273


namespace total_notes_proof_l63_63097

variable (x : Nat)

def total_money := 10350
def fifty_notes_count := 17
def fifty_notes_value := 850  -- 17 * 50
def five_hundred_notes_value := 500 * x
def total_value_proposition := fifty_notes_value + five_hundred_notes_value = total_money

theorem total_notes_proof :
  total_value_proposition -> (fifty_notes_count + x) = 36 :=
by
  intros h
  -- The proof steps would go here, but we use sorry for now.
  sorry

end total_notes_proof_l63_63097


namespace total_games_equal_684_l63_63215

-- Define the number of players
def n : Nat := 19

-- Define the formula to calculate the total number of games played
def total_games (n : Nat) : Nat := n * (n - 1) * 2

-- The proposition asserting the total number of games equals 684
theorem total_games_equal_684 : total_games n = 684 :=
by
  sorry

end total_games_equal_684_l63_63215


namespace harrys_fish_count_l63_63569

theorem harrys_fish_count : 
  let sam_fish := 7
  let joe_fish := 8 * sam_fish
  let harry_fish := 4 * joe_fish
  harry_fish = 224 :=
by
  sorry

end harrys_fish_count_l63_63569


namespace total_percentage_of_failed_candidates_l63_63276

theorem total_percentage_of_failed_candidates :
  ∀ (total_candidates girls boys : ℕ) (passed_boys passed_girls : ℝ),
    total_candidates = 2000 →
    girls = 900 →
    boys = total_candidates - girls →
    passed_boys = 0.34 * boys →
    passed_girls = 0.32 * girls →
    (total_candidates - (passed_boys + passed_girls)) / total_candidates * 100 = 66.9 :=
by
  intros total_candidates girls boys passed_boys passed_girls
  intro h_total_candidates
  intro h_girls
  intro h_boys
  intro h_passed_boys
  intro h_passed_girls
  sorry

end total_percentage_of_failed_candidates_l63_63276


namespace total_paved_1120_l63_63379

-- Definitions based on given problem conditions
def workers_paved_april : ℕ := 480
def less_than_march : ℕ := 160
def workers_paved_march : ℕ := workers_paved_april + less_than_march
def total_paved : ℕ := workers_paved_april + workers_paved_march

-- The statement to prove
theorem total_paved_1120 : total_paved = 1120 := by
  sorry

end total_paved_1120_l63_63379


namespace triangle_exists_l63_63261

theorem triangle_exists (x : ℕ) (hx : x > 0) :
  (3 * x + 10 > x * x) ∧ (x * x + 10 > 3 * x) ∧ (x * x + 3 * x > 10) ↔ (x = 3 ∨ x = 4) :=
by
  sorry

end triangle_exists_l63_63261


namespace dealer_overall_gain_l63_63492

noncomputable def dealer_gain_percentage (weight1 weight2 : ℕ) (cost_price : ℕ) : ℚ :=
  let actual_weight_sold := weight1 + weight2
  let supposed_weight_sold := 1000 + 1000
  let gain_item1 := cost_price - (weight1 / 1000) * cost_price
  let gain_item2 := cost_price - (weight2 / 1000) * cost_price
  let total_gain := gain_item1 + gain_item2
  let total_actual_cost := (actual_weight_sold / 1000) * cost_price
  (total_gain / total_actual_cost) * 100

theorem dealer_overall_gain :
  dealer_gain_percentage 900 850 100 = 14.29 := 
sorry

end dealer_overall_gain_l63_63492


namespace minimum_common_perimeter_l63_63241

noncomputable def is_integer (x: ℝ) : Prop := ∃ (n: ℤ), x = n

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ is_triangle a b c

theorem minimum_common_perimeter :
  ∃ (a b : ℝ),
    is_integer a ∧ is_integer b ∧
    4 * a = 5 * b - 18 ∧
    is_isosceles_triangle a a (2 * a - 12) ∧
    is_isosceles_triangle b b (3 * b - 30) ∧
    (2 * a + (2 * a - 12) = 2 * b + (3 * b - 30)) ∧
    (2 * a + (2 * a - 12) = 228) := sorry

end minimum_common_perimeter_l63_63241


namespace value_of_k_range_of_k_l63_63528

noncomputable def quadratic_eq_has_real_roots (k : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ^ 2 + (2 - 2 * k) * x₁ + k ^ 2 = 0 ∧
    x₂ ^ 2 + (2 - 2 * k) * x₂ + k ^ 2 = 0

def roots_condition (x₁ x₂ : ℝ) : Prop :=
  |(x₁ + x₂)| + 1 = x₁ * x₂

theorem value_of_k (k : ℝ) :
  quadratic_eq_has_real_roots k →
  (∀ (x₁ x₂ : ℝ), roots_condition x₁ x₂ → x₁ ^ 2 + (2 - 2 * k) * x₁ + k ^ 2 = 0 →
                    x₂ ^ 2 + (2 - 2 * k) * x₂ + k ^ 2 = 0 → k = -3) :=
by sorry

theorem range_of_k :
  ∃ (k : ℝ), quadratic_eq_has_real_roots k → k ≤ 1 :=
by sorry

end value_of_k_range_of_k_l63_63528


namespace ratio_f_l63_63381

variable (f : ℝ → ℝ)

-- Hypothesis: For all x in ℝ^+, f'(x) = 3/x * f(x)
axiom hyp1 : ∀ x : ℝ, x > 0 → deriv f x = (3 / x) * f x

-- Hypothesis: f(2^2016) ≠ 0
axiom hyp2 : f (2^2016) ≠ 0

-- Prove that f(2^2017) / f(2^2016) = 8
theorem ratio_f : f (2^2017) / f (2^2016) = 8 :=
sorry

end ratio_f_l63_63381


namespace relationship_among_three_numbers_l63_63150

theorem relationship_among_three_numbers :
  let a := -0.3
  let b := 0.3^2
  let c := 2^0.3
  b < a ∧ a < c :=
by
  let a := -0.3
  let b := 0.3^2
  let c := 2^0.3
  sorry

end relationship_among_three_numbers_l63_63150


namespace max_length_shortest_arc_l63_63718

theorem max_length_shortest_arc (C : ℝ) (hC : C = 84) : 
  ∃ shortest_arc_length : ℝ, shortest_arc_length = 2 :=
by
  -- now prove it
  sorry

end max_length_shortest_arc_l63_63718


namespace initial_bird_families_l63_63128

/- Definitions: -/
def birds_away_africa : ℕ := 23
def birds_away_asia : ℕ := 37
def birds_left_mountain : ℕ := 25

/- Theorem (Question and Correct Answer): -/
theorem initial_bird_families : birds_away_africa + birds_away_asia + birds_left_mountain = 85 := by
  sorry

end initial_bird_families_l63_63128


namespace frost_cakes_total_l63_63624

-- Conditions
def Cagney_time := 60 -- seconds per cake
def Lacey_time := 40  -- seconds per cake
def total_time := 10 * 60 -- 10 minutes in seconds

-- The theorem to prove
theorem frost_cakes_total (Cagney_time Lacey_time total_time : ℕ) (h1 : Cagney_time = 60) (h2 : Lacey_time = 40) (h3 : total_time = 600):
  (total_time / (Cagney_time * Lacey_time / (Cagney_time + Lacey_time))) = 25 :=
by
  -- Proof to be filled in
  sorry

end frost_cakes_total_l63_63624


namespace solve_quadratic_eq_l63_63998

theorem solve_quadratic_eq (a c : ℝ) (h1 : a + c = 31) (h2 : a < c) (h3 : (24:ℝ)^2 - 4 * a * c = 0) : a = 9 ∧ c = 22 :=
by {
  sorry
}

end solve_quadratic_eq_l63_63998


namespace five_fourths_of_x_over_3_l63_63748

theorem five_fourths_of_x_over_3 (x : ℚ) : (5/4) * (x/3) = 5 * x / 12 :=
by
  sorry

end five_fourths_of_x_over_3_l63_63748


namespace required_fencing_l63_63491

-- Given definitions and conditions
def area (L W : ℕ) : ℕ := L * W

def fencing (W L : ℕ) : ℕ := 2 * W + L

theorem required_fencing
  (L W : ℕ)
  (hL : L = 10)
  (hA : area L W = 600) :
  fencing W L = 130 := by
  sorry

end required_fencing_l63_63491


namespace part1_part2_l63_63706

def p (a : ℝ) : Prop := a^2 - 5*a - 6 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 = 0 → x < 0

theorem part1 (a : ℝ) (hp : p a) : a ∈ Set.Iio (-1) ∪ Set.Ioi 6 :=
sorry

theorem part2 (a : ℝ) (h_or : p a ∨ q a) (h_and : ¬ (p a ∧ q a)) : a ∈ Set.Iio (-1) ∪ Set.Ioc 2 6 :=
sorry

end part1_part2_l63_63706


namespace tan_theta_perpendicular_vectors_l63_63572

theorem tan_theta_perpendicular_vectors (θ : ℝ) (h : Real.sqrt 3 * Real.cos θ + Real.sin θ = 0) : Real.tan θ = - Real.sqrt 3 :=
sorry

end tan_theta_perpendicular_vectors_l63_63572


namespace find_a_l63_63997

theorem find_a (a : ℝ) : (∀ x : ℝ, (x + 1) * (x - 3) = x^2 + a * x - 3) → a = -2 :=
  by
    sorry

end find_a_l63_63997


namespace gary_money_left_l63_63810

theorem gary_money_left (initial_amount spent_amount remaining_amount : ℕ)
  (h1 : initial_amount = 73)
  (h2 : spent_amount = 55)
  (h3 : remaining_amount = 18) : initial_amount - spent_amount = remaining_amount := 
by 
  sorry

end gary_money_left_l63_63810


namespace solution_l63_63323

def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem solution (x : ℝ) : g (g x) = g x ↔ x = 0 ∨ x = 4 ∨ x = 5 ∨ x = -1 :=
by
  sorry

end solution_l63_63323


namespace find_line_equation_l63_63346

variable (x y : ℝ)

theorem find_line_equation (hx : x = -5) (hy : y = 2)
  (line_through_point : ∃ a b c : ℝ, a * x + b * y + c = 0)
  (x_intercept_twice_y_intercept : ∀ a b c : ℝ, c ≠ 0 → b ≠ 0 → (a / c) = 2 * (c / b)) :
  ∃ a b c : ℝ, (a * x + b * y + c = 0 ∧ (a = 2 ∧ b = 5 ∧ c = 0) ∨ (a = 1 ∧ b = 2 ∧ c = 1)) :=
sorry

end find_line_equation_l63_63346


namespace sum_of_roots_l63_63076

theorem sum_of_roots: (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a ≠ b ∧ a + b = 6) :=
by
  sorry

end sum_of_roots_l63_63076


namespace divisor_is_three_l63_63279

noncomputable def find_divisor (n : ℕ) (reduction : ℕ) (result : ℕ) : ℕ :=
  n / result

theorem divisor_is_three (x : ℝ) : 
  (original : ℝ) → (reduction : ℝ) → (new_result : ℝ) → 
  original = 45 → new_result = 45 - 30 → (original / x = new_result) → 
  x = 3 := by 
  intros original reduction new_result h1 h2 h3
  sorry

end divisor_is_three_l63_63279


namespace original_price_eq_600_l63_63067

theorem original_price_eq_600 (P : ℝ) (h1 : 300 = P * 0.5) : 
  P = 600 :=
sorry

end original_price_eq_600_l63_63067


namespace geometric_sequence_seventh_term_l63_63853

theorem geometric_sequence_seventh_term (r : ℕ) (r_pos : 0 < r) 
  (h1 : 3 * r^4 = 243) : 
  3 * r^6 = 2187 :=
by
  sorry

end geometric_sequence_seventh_term_l63_63853


namespace solve_ineqs_l63_63873

theorem solve_ineqs (a x : ℝ) (h1 : |x - 2 * a| ≤ 3) (h2 : 0 < x + a ∧ x + a ≤ 4) 
  (ha : a = 3) (hx : x = 1) : 
  (|x - 2 * a| ≤ 3) ∧ (0 < x + a ∧ x + a ≤ 4) :=
by
  sorry

end solve_ineqs_l63_63873


namespace factorization_correct_l63_63108

theorem factorization_correct : 
  ∀ x : ℝ, (x^2 + 1) * (x^3 - x^2 + x - 1) = (x^2 + 1)^2 * (x - 1) :=
by
  intros
  sorry

end factorization_correct_l63_63108


namespace total_routes_A_to_B_l63_63263

-- Define the conditions
def routes_A_to_C : ℕ := 4
def routes_C_to_B : ℕ := 2

-- Statement to prove
theorem total_routes_A_to_B : (routes_A_to_C * routes_C_to_B = 8) :=
by
  -- Omitting the proof, but stating that there is a total of 8 routes from A to B
  sorry

end total_routes_A_to_B_l63_63263


namespace sum_nth_beginning_end_l63_63745

theorem sum_nth_beginning_end (n : ℕ) (F L : ℤ) (M : ℤ) 
  (consecutive : ℤ → ℤ) (median : M = 60) 
  (median_formula : M = (F + L) / 2) :
  n = n → F + L = 120 :=
by
  sorry

end sum_nth_beginning_end_l63_63745


namespace find_g_of_one_fifth_l63_63942

variable {g : ℝ → ℝ}

theorem find_g_of_one_fifth (h₀ : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1)
    (h₁ : g 0 = 0)
    (h₂ : ∀ {x y}, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y)
    (h₃ : ∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x)
    (h₄ : ∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2) :
  g (1 / 5) = 1 / 4 :=
by
  sorry

end find_g_of_one_fifth_l63_63942


namespace range_of_a_for_no_extreme_points_l63_63059

theorem range_of_a_for_no_extreme_points :
  ∀ (a : ℝ), (∀ x : ℝ, x * (x - 2 * a) * x + 1 ≠ 0) ↔ -1 ≤ a ∧ a ≤ 1 := sorry

end range_of_a_for_no_extreme_points_l63_63059


namespace problem1_f_x_linear_problem2_f_x_l63_63231

-- Problem 1 statement: Prove f(x) = 2x + 7 given conditions
theorem problem1_f_x_linear (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * x + 7)
  (h2 : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) : 
  ∀ x, f x = 2 * x + 7 :=
by sorry

-- Problem 2 statement: Prove f(x) = 2x - 1/x given conditions
theorem problem2_f_x (f : ℝ → ℝ) 
  (h1 : ∀ x, 2 * f x + f (1 / x) = 3 * x) : 
  ∀ x, f x = 2 * x - 1 / x :=
by sorry

end problem1_f_x_linear_problem2_f_x_l63_63231


namespace smallest_number_to_add_quotient_of_resulting_number_l63_63851

theorem smallest_number_to_add (k : ℕ) : 456 ∣ (897326 + k) → k = 242 := 
sorry

theorem quotient_of_resulting_number : (897326 + 242) / 456 = 1968 := 
sorry

end smallest_number_to_add_quotient_of_resulting_number_l63_63851


namespace albums_in_either_but_not_both_l63_63854

-- Defining the conditions
def shared_albums : ℕ := 9
def total_albums_andrew : ℕ := 17
def unique_albums_john : ℕ := 6

-- Stating the theorem to prove
theorem albums_in_either_but_not_both :
  (total_albums_andrew - shared_albums) + unique_albums_john = 14 :=
sorry

end albums_in_either_but_not_both_l63_63854


namespace part1_part2_l63_63793

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |2 * x - 1|

theorem part1 : {x : ℝ | f x ≤ 5} = {x : ℝ | -7 / 4 ≤ x ∧ x ≤ 3 / 4} :=
sorry

theorem part2 (h : ∃ x : ℝ, f x < |m - 2|) : m > 6 ∨ m < -2 :=
sorry

end part1_part2_l63_63793


namespace accommodation_ways_l63_63848

-- Definition of the problem
def triple_room_count : ℕ := 1
def double_room_count : ℕ := 2
def adults_count : ℕ := 3
def children_count : ℕ := 2
def total_ways : ℕ := 60

-- Main statement to be proved
theorem accommodation_ways :
  (triple_room_count = 1) →
  (double_room_count = 2) →
  (adults_count = 3) →
  (children_count = 2) →
  -- Children must be accompanied by adults, and not all rooms need to be occupied.
  -- We are to prove that the number of valid ways to assign the rooms is 60
  total_ways = 60 :=
by sorry

end accommodation_ways_l63_63848


namespace revenue_increase_l63_63623

theorem revenue_increase (R : ℕ) (r2000 r2003 r2005 : ℝ) (h1 : r2003 = r2000 * 1.50) (h2 : r2005 = r2000 * 1.80) :
  ((r2005 - r2003) / r2003) * 100 = 20 :=
by sorry

end revenue_increase_l63_63623


namespace points_on_line_l63_63510

theorem points_on_line : 
    ∀ (P : ℝ × ℝ),
      (P = (1, 2) ∨ P = (0, 0) ∨ P = (2, 4) ∨ P = (5, 10) ∨ P = (-1, -2))
      → (∃ m b, m = 2 ∧ b = 0 ∧ P.2 = m * P.1 + b) :=
by
  sorry

end points_on_line_l63_63510


namespace chocolate_bar_min_breaks_l63_63606

theorem chocolate_bar_min_breaks (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  ∃ k, k = m * n - 1 := by
  sorry

end chocolate_bar_min_breaks_l63_63606


namespace binom_prod_l63_63224

theorem binom_prod : (Nat.choose 10 3) * (Nat.choose 8 3) * 2 = 13440 := by
  sorry

end binom_prod_l63_63224


namespace fourth_power_sum_l63_63070

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a^2 + b^2 + c^2 = 3) 
  (h3 : a^3 + b^3 + c^3 = 6) : 
  a^4 + b^4 + c^4 = 4.5 :=
by
  sorry

end fourth_power_sum_l63_63070


namespace problem_1_problem_2_problem_3_l63_63980

-- Condition: x1 and x2 are the roots of the quadratic equation x^2 - 2(m+2)x + m^2 = 0
variables {x1 x2 m : ℝ}
axiom roots_quadratic_equation : x1^2 - 2*(m+2) * x1 + m^2 = 0 ∧ x2^2 - 2*(m+2) * x2 + m^2 = 0

-- 1. When m = 0, the roots of the equation are 0 and 4
theorem problem_1 (h : m = 0) : x1 = 0 ∧ x2 = 4 :=
by 
  sorry

-- 2. If (x1 - 2)(x2 - 2) = 41, then m = 9
theorem problem_2 (h : (x1 - 2) * (x2 - 2) = 41) : m = 9 :=
by
  sorry

-- 3. Given an isosceles triangle ABC with one side length 9, if x1 and x2 are the lengths of the other two sides, 
--    prove that the perimeter is 19.
theorem problem_3 (h1 : x1 + x2 > 9) (h2 : 9 + x1 > x2) (h3 : 9 + x2 > x1) : x1 = 1 ∧ x2 = 9 ∧ (x1 + x2 + 9) = 19 :=
by 
  sorry

end problem_1_problem_2_problem_3_l63_63980


namespace find_m_and_equation_of_l2_l63_63250

theorem find_m_and_equation_of_l2 (a : ℝ) (M: ℝ × ℝ) (m : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (hM : M = (-5, 1)) 
  (hl1 : ∀ {x y : ℝ}, 2 * x - y + 2 = 0) 
  (hl : ∀ {x y : ℝ}, x + y + m = 0) 
  (hl2 : ∀ {x y : ℝ}, (∃ p : ℝ × ℝ, p = M → x - 2 * y + 7 = 0)) : 
  m = -5 ∧ ∀ {x y : ℝ}, x - 2 * y + 7 = 0 :=
by
  sorry

end find_m_and_equation_of_l2_l63_63250


namespace value_of_x_l63_63788

theorem value_of_x (x : ℕ) : (1 / 16) * (2 ^ 20) = 4 ^ x → x = 8 := by
  sorry

end value_of_x_l63_63788


namespace tins_per_case_is_24_l63_63054

def total_cases : ℕ := 15
def damaged_percentage : ℝ := 0.05
def remaining_tins : ℕ := 342

theorem tins_per_case_is_24 (x : ℕ) (h : (1 - damaged_percentage) * (total_cases * x) = remaining_tins) : x = 24 :=
  sorry

end tins_per_case_is_24_l63_63054


namespace sin_cos_inequality_l63_63317

theorem sin_cos_inequality (α : ℝ) 
  (h1 : 0 ≤ α) (h2 : α < 2 * Real.pi) 
  (h3 : Real.sin α > Real.sqrt 3 * Real.cos α) : 
  (Real.pi / 3 < α ∧ α < 4 * Real.pi / 3) :=
sorry

end sin_cos_inequality_l63_63317


namespace rectangle_side_length_l63_63740

theorem rectangle_side_length (a c : ℝ) (h_ratio : a / c = 3 / 4) (hc : c = 4) : a = 3 :=
by
  sorry

end rectangle_side_length_l63_63740


namespace interior_angle_of_regular_nonagon_l63_63540

theorem interior_angle_of_regular_nonagon : 
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  (sum_of_interior_angles / n) = 140 := 
by
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  show sum_of_interior_angles / n = 140
  sorry

end interior_angle_of_regular_nonagon_l63_63540


namespace number_of_paths_from_C_to_D_l63_63770

-- Define the grid and positions
def C := (0,0)  -- Bottom-left corner
def D := (7,3)  -- Top-right corner
def gridWidth : ℕ := 7
def gridHeight : ℕ := 3

-- Define the binomial coefficient function
-- Note: Lean already has binomial coefficient defined in Mathlib, use Nat.choose for that

-- The statement to prove
theorem number_of_paths_from_C_to_D : Nat.choose (gridWidth + gridHeight) gridHeight = 120 :=
by
  sorry

end number_of_paths_from_C_to_D_l63_63770


namespace leftover_value_correct_l63_63584

noncomputable def leftover_value (nickels_per_roll pennies_per_roll : ℕ) (sarah_nickels sarah_pennies tom_nickels tom_pennies : ℕ) : ℚ :=
  let total_nickels := sarah_nickels + tom_nickels
  let total_pennies := sarah_pennies + tom_pennies
  let leftover_nickels := total_nickels % nickels_per_roll
  let leftover_pennies := total_pennies % pennies_per_roll
  (leftover_nickels * 5 + leftover_pennies) / 100

theorem leftover_value_correct :
  leftover_value 40 50 132 245 98 203 = 1.98 := 
by
  sorry

end leftover_value_correct_l63_63584


namespace time_after_hours_l63_63817

-- Definitions based on conditions
def current_time : ℕ := 3
def hours_later : ℕ := 2517
def clock_cycle : ℕ := 12

-- Statement to prove
theorem time_after_hours :
  (current_time + hours_later) % clock_cycle = 12 := 
sorry

end time_after_hours_l63_63817


namespace prime_addition_equality_l63_63417

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_addition_equality (x y : ℕ)
  (hx : is_prime x)
  (hy : is_prime y)
  (hxy : x < y)
  (hsum : x + y = 36) : 4 * x + y = 51 :=
sorry

end prime_addition_equality_l63_63417


namespace roommates_condition_l63_63535

def f (x : ℝ) := 3 * x ^ 2 + 5 * x - 1
def g (x : ℝ) := 2 * x ^ 2 - 3 * x + 5

theorem roommates_condition : f 3 = 2 * g 3 + 5 := 
by {
  sorry
}

end roommates_condition_l63_63535


namespace negation_universal_proposition_l63_63701

theorem negation_universal_proposition {x : ℝ} : 
  (¬ ∀ x : ℝ, x^2 ≥ 2) ↔ (∃ x : ℝ, x^2 < 2) := 
sorry

end negation_universal_proposition_l63_63701


namespace sum_of_7_and_2_terms_l63_63962

open Nat

variable {α : Type*} [Field α]

-- Definitions
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d
  
def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∀ m n k : ℕ, m < n → n < k → a n * a n = a m * a k
  
def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (n * (a 0 + a n)) / 2

-- Given Conditions
variable (a : ℕ → α) 
variable (d : α)

-- Checked. Arithmetic sequence with non-zero common difference
axiom h1 : is_arithmetic_sequence a d

-- Known values provided in the problem statement
axiom h2 : a 1 = 6

-- Terms forming a geometric sequence
axiom h3 : is_geometric_sequence a

-- The goal is to find the sum of the first 7 terms and the first 2 terms
theorem sum_of_7_and_2_terms : sum_first_n_terms a 7 + sum_first_n_terms a 2 = 80 := 
by {
  -- Proof will be here
  sorry
}

end sum_of_7_and_2_terms_l63_63962


namespace Morse_code_distinct_symbols_count_l63_63157

theorem Morse_code_distinct_symbols_count :
  let count (n : ℕ) := 2 ^ n
  count 1 + count 2 + count 3 + count 4 + count 5 = 62 :=
by
  sorry

end Morse_code_distinct_symbols_count_l63_63157


namespace evaluate_expression_l63_63366

theorem evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 2) : ((a^b)^a + (b^a)^b = 793) := by
  -- The following lines skip the proof but outline the structure:
  sorry

end evaluate_expression_l63_63366


namespace triangle_inequality_inequality_l63_63194

-- Define a helper function to describe the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

-- Define the main statement
theorem triangle_inequality_inequality (a b c : ℝ) (h_triangle : triangle_inequality a b c):
  a * (b - c) ^ 2 + b * (c - a) ^ 2 + c * (a - b) ^ 2 + 4 * a * b * c > a ^ 3 + b ^ 3 + c ^ 3 :=
sorry

end triangle_inequality_inequality_l63_63194


namespace sugar_needed_for_partial_recipe_l63_63904

theorem sugar_needed_for_partial_recipe :
  let initial_sugar := 5 + 3/4
  let part := 3/4
  let needed_sugar := 4 + 5/16
  initial_sugar * part = needed_sugar := 
by 
  sorry

end sugar_needed_for_partial_recipe_l63_63904


namespace base_length_of_prism_l63_63798

theorem base_length_of_prism (V : ℝ) (hV : V = 36 * Real.pi) : ∃ (AB : ℝ), AB = 3 * Real.sqrt 3 :=
by
  sorry

end base_length_of_prism_l63_63798


namespace max_value_expression_l63_63359

theorem max_value_expression (r : ℝ) : ∃ r : ℝ, -5 * r^2 + 40 * r - 12 = 68 ∧ (∀ s : ℝ, -5 * s^2 + 40 * s - 12 ≤ 68) :=
sorry

end max_value_expression_l63_63359


namespace polynomial_root_solution_l63_63619

theorem polynomial_root_solution (a b c : ℝ) (h1 : (2:ℝ)^5 + 4*(2:ℝ)^4 + a*(2:ℝ)^2 = b*(2:ℝ) + 4*c) 
  (h2 : (-2:ℝ)^5 + 4*(-2:ℝ)^4 + a*(-2:ℝ)^2 = b*(-2:ℝ) + 4*c) :
  a = -48 ∧ b = 16 ∧ c = -32 :=
sorry

end polynomial_root_solution_l63_63619


namespace sum_of_multiples_is_even_l63_63684

theorem sum_of_multiples_is_even (a b : ℤ) (h1 : ∃ m : ℤ, a = 4 * m) (h2 : ∃ n : ℤ, b = 6 * n) : Even (a + b) :=
sorry

end sum_of_multiples_is_even_l63_63684


namespace decreasing_condition_l63_63015

variable (m : ℝ)

def quadratic_fn (x : ℝ) : ℝ := x^2 + m * x + 1

theorem decreasing_condition (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → (deriv (quadratic_fn m) x ≤ 0)) :
    m ≤ -10 := 
by
  -- Proof omitted
  sorry

end decreasing_condition_l63_63015


namespace distance_to_y_axis_parabola_midpoint_l63_63923

noncomputable def distance_from_midpoint_to_y_axis (x1 x2 : ℝ) : ℝ :=
  (x1 + x2) / 2

theorem distance_to_y_axis_parabola_midpoint :
  ∀ (x1 y1 x2 y2 : ℝ), y1^2 = x1 → y2^2 = x2 → 
  abs (x1 + 1 / 4) + abs (x2 + 1 / 4) = 3 →
  abs (distance_from_midpoint_to_y_axis x1 x2) = 5 / 4 :=
by
  intros x1 y1 x2 y2 h1 h2 h3
  sorry

end distance_to_y_axis_parabola_midpoint_l63_63923


namespace distinct_solutions_subtraction_l63_63457

theorem distinct_solutions_subtraction (r s : ℝ) (h_eq : ∀ x ≠ 3, (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3) 
  (h_r : (6 * r - 18) / (r^2 + 4 * r - 21) = r + 3) 
  (h_s : (6 * s - 18) / (s^2 + 4 * s - 21) = s + 3) 
  (h_distinct : r ≠ s) 
  (h_order : r > s) : 
  r - s = 10 := 
by 
  sorry

end distinct_solutions_subtraction_l63_63457


namespace triangle_angle_sum_l63_63406

theorem triangle_angle_sum (P Q R : ℝ) (h1 : P + Q = 60) (h2 : P + Q + R = 180) : R = 120 := by
  sorry

end triangle_angle_sum_l63_63406


namespace f_1001_value_l63_63027

noncomputable def f : ℕ → ℝ := sorry

theorem f_1001_value :
  (∀ a b n : ℕ, a + b = 2^n → f a + f b = n^2) →
  f 1 = 1 →
  f 1001 = 83 :=
by
  intro h₁ h₂
  sorry

end f_1001_value_l63_63027


namespace keith_spent_on_tires_l63_63591

noncomputable def money_spent_on_speakers : ℝ := 136.01
noncomputable def money_spent_on_cd_player : ℝ := 139.38
noncomputable def total_expenditure : ℝ := 387.85
noncomputable def total_spent_on_speakers_and_cd_player : ℝ := money_spent_on_speakers + money_spent_on_cd_player
noncomputable def money_spent_on_new_tires : ℝ := total_expenditure - total_spent_on_speakers_and_cd_player

theorem keith_spent_on_tires :
  money_spent_on_new_tires = 112.46 :=
by
  sorry

end keith_spent_on_tires_l63_63591


namespace min_value_four_over_a_plus_nine_over_b_l63_63422

theorem min_value_four_over_a_plus_nine_over_b :
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → (∀ x y, x > 0 → y > 0 → x + y ≥ 2 * Real.sqrt (x * y)) →
  (∃ (min_val : ℝ), min_val = (4 / a + 9 / b) ∧ min_val = 25) :=
by
  intros a b ha hb hab am_gm
  sorry

end min_value_four_over_a_plus_nine_over_b_l63_63422


namespace sequence_formula_l63_63268

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, 0 < n →  1 / a (n + 1) = 1 / a n + 1) :
  ∀ n : ℕ, 0 < n → a n = 1 / n :=
by {
  sorry
}

end sequence_formula_l63_63268


namespace total_gold_coins_l63_63264

/--
An old man distributed all the gold coins he had to his two sons into 
two different numbers such that the difference between the squares 
of the two numbers is 49 times the difference between the two numbers. 
Prove that the total number of gold coins the old man had is 49.
-/
theorem total_gold_coins (x y : ℕ) (h : x ≠ y) (h1 : x^2 - y^2 = 49 * (x - y)) : x + y = 49 :=
sorry

end total_gold_coins_l63_63264


namespace positive_correlation_not_proportional_l63_63939

/-- Two quantities x and y depend on each other, and when one increases, the other also increases.
    This general relationship is denoted as a function g such that for any x₁, x₂,
    if x₁ < x₂ then g(x₁) < g(x₂). This implies a positive correlation but not necessarily proportionality. 
    We will prove that this does not imply a proportional relationship (y = kx). -/
theorem positive_correlation_not_proportional (g : ℝ → ℝ) 
(h_increasing: ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂) :
¬ ∃ k : ℝ, ∀ x : ℝ, g x = k * x :=
sorry

end positive_correlation_not_proportional_l63_63939


namespace find_m_and_n_l63_63632

namespace BinomialProof

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n m : ℕ) : Prop :=
  binom (n+1) (m+1) = binom (n+1) m

def condition2 (n m : ℕ) : Prop :=
  binom (n+1) m / binom (n+1) (m-1) = 5 / 3

-- Problem statement
theorem find_m_and_n : ∃ (m n : ℕ), 
  (condition1 n m) ∧ 
  (condition2 n m) ∧ 
  m = 3 ∧ n = 6 := sorry

end BinomialProof

end find_m_and_n_l63_63632


namespace negation_statement_contrapositive_statement_l63_63759

variable (x y : ℝ)

theorem negation_statement :
  (¬ ((x-1) * (y+2) ≠ 0 → x ≠ 1 ∧ y ≠ -2)) ↔ ((x-1) * (y+2) = 0 → x = 1 ∨ y = -2) :=
by sorry

theorem contrapositive_statement :
  (x = 1 ∨ y = -2) → ((x-1) * (y+2) = 0) :=
by sorry

end negation_statement_contrapositive_statement_l63_63759


namespace horse_revolutions_l63_63476

theorem horse_revolutions (r1 r2 : ℝ) (rev1 rev2 : ℕ) (h1 : r1 = 30) (h2 : rev1 = 25) (h3 : r2 = 10) : 
  rev2 = 75 :=
by 
  sorry

end horse_revolutions_l63_63476


namespace TV_cost_difference_l63_63149

def cost_per_square_inch_difference :=
  let first_TV_width := 24
  let first_TV_height := 16
  let first_TV_original_cost_euros := 840
  let first_TV_discount_percent := 0.10
  let first_TV_tax_percent := 0.05
  let exchange_rate_first := 1.20
  let first_TV_area := first_TV_width * first_TV_height

  let discounted_price_first_TV := first_TV_original_cost_euros * (1 - first_TV_discount_percent)
  let total_cost_euros_first_TV := discounted_price_first_TV * (1 + first_TV_tax_percent)
  let total_cost_dollars_first_TV := total_cost_euros_first_TV * exchange_rate_first
  let cost_per_square_inch_first_TV := total_cost_dollars_first_TV / first_TV_area

  let new_TV_width := 48
  let new_TV_height := 32
  let new_TV_original_cost_dollars := 1800
  let new_TV_first_discount_percent := 0.20
  let new_TV_second_discount_percent := 0.15
  let new_TV_tax_percent := 0.08
  let new_TV_area := new_TV_width * new_TV_height

  let price_after_first_discount := new_TV_original_cost_dollars * (1 - new_TV_first_discount_percent)
  let price_after_second_discount := price_after_first_discount * (1 - new_TV_second_discount_percent)
  let total_cost_dollars_new_TV := price_after_second_discount * (1 + new_TV_tax_percent)
  let cost_per_square_inch_new_TV := total_cost_dollars_new_TV / new_TV_area

  let cost_difference_per_square_inch := cost_per_square_inch_first_TV - cost_per_square_inch_new_TV
  cost_difference_per_square_inch

theorem TV_cost_difference :
  cost_per_square_inch_difference = 1.62 := by
  sorry

end TV_cost_difference_l63_63149


namespace max_time_for_taxiing_is_15_l63_63013

-- Declare the function representing the distance traveled by the plane with respect to time
def distance (t : ℝ) : ℝ := 60 * t - 2 * t ^ 2

-- The main theorem stating the maximum time s the plane uses for taxiing
theorem max_time_for_taxiing_is_15 : ∃ s, ∀ t, distance t ≤ distance s ∧ s = 15 :=
by
  sorry

end max_time_for_taxiing_is_15_l63_63013


namespace bacon_percentage_l63_63327

theorem bacon_percentage (total_calories : ℕ) (bacon_calories : ℕ) (strips_of_bacon : ℕ) :
  total_calories = 1250 →
  bacon_calories = 125 →
  strips_of_bacon = 2 →
  (strips_of_bacon * bacon_calories * 100 / total_calories) = 20 :=
by sorry

end bacon_percentage_l63_63327


namespace baker_initial_cakes_l63_63550

theorem baker_initial_cakes (sold : ℕ) (left : ℕ) (initial : ℕ) 
  (h_sold : sold = 41) (h_left : left = 13) : 
  sold + left = initial → initial = 54 :=
by
  intros
  exact sorry

end baker_initial_cakes_l63_63550


namespace common_number_l63_63515

theorem common_number (a b c d e u v w : ℝ) (h1 : (a + b + c + d + e) / 5 = 7) 
                                            (h2 : (u + v + w) / 3 = 10) 
                                            (h3 : (a + b + c + d + e + u + v + w) / 8 = 8) 
                                            (h4 : a + b + c + d + e = 35) 
                                            (h5 : u + v + w = 30) 
                                            (h6 : a + b + c + d + e + u + v + w = 64) 
                                            (h7 : 35 + 30 = 65):
  d = u := 
by
  sorry

end common_number_l63_63515


namespace positive_difference_between_two_numbers_l63_63821

theorem positive_difference_between_two_numbers :
  ∃ (x y : ℚ), x + y = 40 ∧ 3 * y - 4 * x = 10 ∧ abs (y - x) = 60 / 7 :=
sorry

end positive_difference_between_two_numbers_l63_63821


namespace number_of_students_l63_63593

theorem number_of_students (n : ℕ) (bow_cost : ℕ) (vinegar_cost : ℕ) (baking_soda_cost : ℕ) (total_cost : ℕ) :
  bow_cost = 5 → vinegar_cost = 2 → baking_soda_cost = 1 → total_cost = 184 → 8 * n = total_cost → n = 23 :=
by
  intros h_bow h_vinegar h_baking_soda h_total_cost h_equation
  sorry

end number_of_students_l63_63593


namespace divisible_by_6_l63_63823

theorem divisible_by_6 {n : ℕ} (h2 : 2 ∣ n) (h3 : 3 ∣ n) : 6 ∣ n :=
sorry

end divisible_by_6_l63_63823


namespace rectangle_k_value_l63_63463

theorem rectangle_k_value (x d : ℝ)
  (h_ratio : ∃ x, ∀ l w, l = 5 * x ∧ w = 4 * x)
  (h_diagonal : ∀ l w, l = 5 * x ∧ w = 4 * x → d^2 = (5 * x)^2 + (4 * x)^2)
  (h_area_written : ∃ k, ∀ A, A = (5 * x) * (4 * x) → A = k * d^2) :
  ∃ k, k = 20 / 41 := sorry

end rectangle_k_value_l63_63463


namespace area_of_triangle_PQS_l63_63226

-- Define a structure to capture the conditions of the trapezoid and its properties.
structure Trapezoid (P Q R S : Type) :=
(area : ℝ)
(PQ : ℝ)
(RS : ℝ)
(area_PQS : ℝ)
(condition1 : area = 18)
(condition2 : RS = 3 * PQ)

-- Here's the theorem we want to prove, stating the conclusion based on the given conditions.
theorem area_of_triangle_PQS {P Q R S : Type} (T : Trapezoid P Q R S) : T.area_PQS = 4.5 :=
by
  -- Proof will go here, but for now we use sorry.
  sorry

end area_of_triangle_PQS_l63_63226


namespace icing_two_sides_on_Jack_cake_l63_63158

noncomputable def Jack_cake_icing_two_sides (cake_size : ℕ) : ℕ :=
  let side_cubes := 4 * (cake_size - 2) * 3
  let vertical_edge_cubes := 4 * (cake_size - 2)
  side_cubes + vertical_edge_cubes

-- The statement to be proven
theorem icing_two_sides_on_Jack_cake : Jack_cake_icing_two_sides 5 = 96 :=
by
  sorry

end icing_two_sides_on_Jack_cake_l63_63158


namespace propositions_are_3_and_4_l63_63516

-- Conditions
def stmt_1 := "Is it fun to study math?"
def stmt_2 := "Do your homework well and strive to pass the math test next time;"
def stmt_3 := "2 is not a prime number"
def stmt_4 := "0 is a natural number"

-- Representation of a propositional statement
def isPropositional (stmt : String) : Bool :=
  stmt ≠ stmt_1 ∧ stmt ≠ stmt_2

-- The theorem proving the question given the conditions
theorem propositions_are_3_and_4 :
  isPropositional stmt_3 ∧ isPropositional stmt_4 :=
by
  -- Proof to be filled in later
  sorry

end propositions_are_3_and_4_l63_63516


namespace ratio_a2_a3_l63_63524

namespace SequenceProof

def a (n : ℕ) : ℤ := 3 - 2^n

theorem ratio_a2_a3 : a 2 / a 3 = 1 / 5 := by
  sorry

end SequenceProof

end ratio_a2_a3_l63_63524


namespace max_unique_dance_counts_l63_63776

theorem max_unique_dance_counts (boys girls : ℕ) (positive_boys : boys = 29) (positive_girls : girls = 15) 
  (dances : ∀ b g, b ≤ boys → g ≤ girls → ℕ) :
  ∃ num_dances, num_dances = 29 := 
by
  sorry

end max_unique_dance_counts_l63_63776


namespace find_x_l63_63687

def star (a b c d : ℤ) : ℤ × ℤ := (a + c, b - d)

theorem find_x 
  (x y : ℤ) 
  (h_star1 : star 5 4 2 2 = (7, 2)) 
  (h_eq : star x y 3 3 = (7, 2)) : 
  x = 4 := 
sorry

end find_x_l63_63687


namespace fraction_power_four_l63_63000

theorem fraction_power_four :
  (5 / 6) ^ 4 = 625 / 1296 :=
by sorry

end fraction_power_four_l63_63000


namespace number_of_girls_l63_63354

variable (total_children : ℕ) (boys : ℕ)

theorem number_of_girls (h1 : total_children = 117) (h2 : boys = 40) : 
  total_children - boys = 77 := by
  sorry

end number_of_girls_l63_63354


namespace parabola_normal_intersect_l63_63432

theorem parabola_normal_intersect {x y : ℝ} (h₁ : y = x^2) (A : ℝ × ℝ) (hA : A = (-1, 1)) :
  ∃ B : ℝ × ℝ, B = (1.5, 2.25) ∧ ∀ x : ℝ, (y - 1) = 1/2 * (x + 1) →
  ∀ x : ℝ, y = x^2 ∧ B = (1.5, 2.25) :=
sorry

end parabola_normal_intersect_l63_63432


namespace cody_spent_tickets_l63_63002

theorem cody_spent_tickets (initial_tickets lost_tickets remaining_tickets : ℝ) (h1 : initial_tickets = 49.0) (h2 : lost_tickets = 6.0) (h3 : remaining_tickets = 18.0) :
  initial_tickets - lost_tickets - remaining_tickets = 25.0 :=
by
  sorry

end cody_spent_tickets_l63_63002


namespace probability_merlin_dismissed_l63_63891

variable (p : ℝ) (q : ℝ) (coin_flip : ℝ)

axiom h₁ : q = 1 - p
axiom h₂ : 0 ≤ p ∧ p ≤ 1
axiom h₃ : 0 ≤ q ∧ q ≤ 1
axiom h₄ : coin_flip = 0.5

theorem probability_merlin_dismissed : coin_flip = 0.5 := by
  sorry

end probability_merlin_dismissed_l63_63891


namespace exists_nat_number_divisible_by_2019_and_its_digit_sum_also_divisible_by_2019_l63_63885

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_nat_number_divisible_by_2019_and_its_digit_sum_also_divisible_by_2019 :
  ∃ N : ℕ, (N % 2019 = 0) ∧ ((sum_of_digits N) % 2019 = 0) :=
by 
  sorry

end exists_nat_number_divisible_by_2019_and_its_digit_sum_also_divisible_by_2019_l63_63885


namespace shekar_marks_in_math_l63_63797

theorem shekar_marks_in_math (M : ℕ) : 
  (65 + 82 + 67 + 75 + M) / 5 = 73 → M = 76 :=
by
  intros h
  sorry

end shekar_marks_in_math_l63_63797


namespace cubic_has_real_root_l63_63507

open Real

-- Define the conditions
variables (a0 a1 a2 a3 : ℝ) (h : a0 ≠ 0)

-- Define the cubic polynomial function
def cubic (x : ℝ) : ℝ :=
  a0 * x^3 + a1 * x^2 + a2 * x + a3

-- State the theorem
theorem cubic_has_real_root : ∃ x : ℝ, cubic a0 a1 a2 a3 x = 0 :=
by
  sorry

end cubic_has_real_root_l63_63507


namespace find_s_l63_63421

noncomputable def is_monic (p : Polynomial ℝ) : Prop :=
  p.leadingCoeff = 1

variables (f g : Polynomial ℝ) (s : ℝ)
variables (r1 r2 r3 r4 r5 r6 : ℝ)

-- Conditions
def conditions : Prop :=
  is_monic f ∧ is_monic g ∧
  (f.roots = [s + 2, s + 8, r1] ∨ f.roots = [s + 8, s + 2, r1] ∨ f.roots = [s + 2, r1, s + 8] ∨
   f.roots = [r1, s + 2, s + 8] ∨ f.roots = [r1, s + 8, s + 2]) ∧
  (g.roots = [s + 4, s + 10, r2] ∨ g.roots = [s + 10, s + 4, r2] ∨ g.roots = [s + 4, r2, s + 10] ∨
   g.roots = [r2, s + 4, s + 10] ∨ g.roots = [r2, s + 10, s + 4]) ∧
  ∀ (x : ℝ), f.eval x - g.eval x = 2 * s

-- Theorem statement

theorem find_s (h : conditions f g r1 r2 s) : s = 288 / 14 :=
sorry

end find_s_l63_63421


namespace ratio_closest_to_10_l63_63607

theorem ratio_closest_to_10 :
  (⌊(10^3000 + 10^3004 : ℝ) / (10^3001 + 10^3003) + 0.5⌋ : ℝ) = 10 :=
sorry

end ratio_closest_to_10_l63_63607


namespace find_a_value_l63_63430

noncomputable def f (x a : ℝ) : ℝ := (x^2 + a) / (x + 1)

def slope_of_tangent_line (a : ℝ) : Prop :=
  (deriv (fun x => f x a) 1) = -1

theorem find_a_value : ∃ a : ℝ, slope_of_tangent_line a ∧ a = 7 := by
  sorry

end find_a_value_l63_63430


namespace probability_correct_l63_63467

structure Bag :=
  (blue : ℕ)
  (green : ℕ)
  (yellow : ℕ)

def marbles_drawn_sequence (bag : Bag) : ℚ :=
  let total_marbles := bag.blue + bag.green + bag.yellow
  let prob_blue_first := ↑bag.blue / total_marbles
  let prob_green_second := ↑bag.green / (total_marbles - 1)
  let prob_yellow_third := ↑bag.yellow / (total_marbles - 2)
  prob_blue_first * prob_green_second * prob_yellow_third

theorem probability_correct (bag : Bag) (h : bag = ⟨4, 6, 5⟩) : 
  marbles_drawn_sequence bag = 20 / 455 :=
by
  sorry

end probability_correct_l63_63467


namespace solve_inequality_l63_63679

def f (a x : ℝ) : ℝ := a * x * (x + 1) + 1

theorem solve_inequality (a x : ℝ) (h : f a x < 0) : x < (1 / a) ∨ (x > 1 ∧ a ≠ 0) := by
  sorry

end solve_inequality_l63_63679


namespace initial_cakes_l63_63292

variable (friend_bought : Nat) (baker_has : Nat)

theorem initial_cakes (h1 : friend_bought = 140) (h2 : baker_has = 15) : 
  (friend_bought + baker_has = 155) := 
by
  sorry

end initial_cakes_l63_63292


namespace cannot_determine_total_inhabitants_without_additional_info_l63_63513

variable (T : ℝ) (M F : ℝ)

axiom inhabitants_are_males_females : M + F = 1
axiom twenty_percent_of_males_are_literate : M * 0.20 * T = 0.20 * M * T
axiom twenty_five_percent_of_all_literates : 0.25 = 0.25 * T / T
axiom thirty_two_five_percent_of_females_are_literate : F = 1 - M ∧ F * 0.325 * T = 0.325 * (1 - M) * T

theorem cannot_determine_total_inhabitants_without_additional_info :
  ∃ (T : ℝ), True ↔ False := by
  sorry

end cannot_determine_total_inhabitants_without_additional_info_l63_63513


namespace product_is_correct_l63_63834

theorem product_is_correct :
  50 * 29.96 * 2.996 * 500 = 2244004 :=
by
  sorry

end product_is_correct_l63_63834


namespace vishal_investment_more_than_trishul_l63_63441

theorem vishal_investment_more_than_trishul:
  ∀ (V T R : ℝ),
  R = 2100 →
  T = 0.90 * R →
  V + T + R = 6069 →
  ((V - T) / T) * 100 = 10 :=
by
  intros V T R hR hT hSum
  sorry

end vishal_investment_more_than_trishul_l63_63441


namespace cottage_cheese_quantity_l63_63352

theorem cottage_cheese_quantity (x : ℝ) 
    (milk_fat : ℝ := 0.05) 
    (curd_fat : ℝ := 0.155) 
    (whey_fat : ℝ := 0.005) 
    (milk_mass : ℝ := 1) 
    (h : (curd_fat * x + whey_fat * (milk_mass - x)) = milk_fat * milk_mass) : 
    x = 0.3 :=
    sorry

end cottage_cheese_quantity_l63_63352


namespace area_of_triangle_is_correct_l63_63757

def line_1 (x y : ℝ) : Prop := y - 5 * x = -4
def line_2 (x y : ℝ) : Prop := 4 * y + 2 * x = 16

def y_axis (x y : ℝ) : Prop := x = 0

def satisfies_y_intercepts (f : ℝ → ℝ) : Prop :=
f 0 = -4 ∧ f 0 = 4

noncomputable def area_of_triangle (height base : ℝ) : ℝ :=
(1 / 2) * base * height

theorem area_of_triangle_is_correct :
  ∃ (x y : ℝ), line_1 x y ∧ line_2 x y ∧ y_axis 0 8 ∧ area_of_triangle (16 / 11) 8 = (64 / 11) := 
sorry

end area_of_triangle_is_correct_l63_63757


namespace diff_of_squares_535_465_l63_63211

theorem diff_of_squares_535_465 : (535^2 - 465^2) = 70000 :=
sorry

end diff_of_squares_535_465_l63_63211


namespace chapatis_order_count_l63_63667

theorem chapatis_order_count (chapati_cost rice_cost veg_cost total_paid chapati_count : ℕ) 
  (rice_plates veg_plates : ℕ)
  (H1 : chapati_cost = 6)
  (H2 : rice_cost = 45)
  (H3 : veg_cost = 70)
  (H4 : total_paid = 1111)
  (H5 : rice_plates = 5)
  (H6 : veg_plates = 7)
  (H7 : chapati_count = (total_paid - (rice_plates * rice_cost + veg_plates * veg_cost)) / chapati_cost) :
  chapati_count = 66 :=
by
  sorry

end chapatis_order_count_l63_63667


namespace sheets_in_stack_l63_63092

theorem sheets_in_stack (h : 200 * t = 2.5) (h_pos : t > 0) : (5 / t) = 400 :=
by
  sorry

end sheets_in_stack_l63_63092


namespace probability_of_negative_l63_63611

def set_of_numbers : Set ℤ := {-2, 1, 4, -3, 0}
def negative_numbers : Set ℤ := {-2, -3}
def total_numbers : ℕ := 5
def total_negative_numbers : ℕ := 2

theorem probability_of_negative :
  (total_negative_numbers : ℚ) / (total_numbers : ℚ) = 2 / 5 := 
by 
  sorry

end probability_of_negative_l63_63611


namespace diver_descend_rate_l63_63562

theorem diver_descend_rate (depth : ℕ) (time : ℕ) (rate : ℕ) 
  (h1 : depth = 6400) (h2 : time = 200) : rate = 32 :=
by
  sorry

end diver_descend_rate_l63_63562


namespace parabola_focus_value_of_a_l63_63285

theorem parabola_focus_value_of_a :
  (∀ a : ℝ, (∃ y : ℝ, y = a * (0^2) ∧ (0, y) = (0, 3 / 8)) → a = 2 / 3) := by
sorry

end parabola_focus_value_of_a_l63_63285


namespace smallest_m_l63_63771

theorem smallest_m (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  ∃ m, (∀ (a b c : ℝ), a + b + c = 1 → 0 < a → 0 < b → 0 < c → m * (a ^ 3 + b ^ 3 + c ^ 3) ≥ 6 * (a ^ 2 + b ^ 2 + c ^ 2) + 1) ↔ m = 27 :=
by
  sorry

end smallest_m_l63_63771


namespace fractional_equation_solution_l63_63166

theorem fractional_equation_solution (m : ℝ) (x : ℝ) :
  (m + 3) / (x - 1) = 1 → x > 0 → m > -4 ∧ m ≠ -3 :=
by
  sorry

end fractional_equation_solution_l63_63166


namespace average_distance_per_day_l63_63646

def miles_monday : ℕ := 12
def miles_tuesday : ℕ := 18
def miles_wednesday : ℕ := 21
def total_days : ℕ := 3

def total_distance : ℕ := miles_monday + miles_tuesday + miles_wednesday

theorem average_distance_per_day : total_distance / total_days = 17 := by
  sorry

end average_distance_per_day_l63_63646


namespace log_one_plus_xsq_lt_xsq_over_one_plus_xsq_l63_63743

theorem log_one_plus_xsq_lt_xsq_over_one_plus_xsq (x : ℝ) (hx : 0 < x) : 
  Real.log (1 + x^2) < x^2 / (1 + x^2) :=
sorry

end log_one_plus_xsq_lt_xsq_over_one_plus_xsq_l63_63743


namespace contrapositive_necessary_condition_l63_63903

theorem contrapositive_necessary_condition (a b : Prop) (h : a → b) : ¬b → ¬a :=
by
  sorry

end contrapositive_necessary_condition_l63_63903


namespace fermats_little_theorem_analogue_l63_63576

theorem fermats_little_theorem_analogue 
  (a : ℤ) (h1 : Int.gcd a 561 = 1) : a ^ 560 ≡ 1 [ZMOD 561] := 
sorry

end fermats_little_theorem_analogue_l63_63576


namespace find_first_prime_l63_63628

theorem find_first_prime (p1 p2 z : ℕ) 
  (prime_p1 : Nat.Prime p1)
  (prime_p2 : Nat.Prime p2)
  (z_eq : z = p1 * p2)
  (z_val : z = 33)
  (p2_range : 8 < p2 ∧ p2 < 24)
  : p1 = 3 := 
sorry

end find_first_prime_l63_63628


namespace find_m_value_l63_63407

variable (m a0 a1 a2 a3 a4 a5 : ℚ)

-- Defining the conditions given in the problem
def poly_expansion_condition : Prop := (m * 1 - 1)^5 = a5 * 1^5 + a4 * 1^4 + a3 * 1^3 + a2 * 1^2 + a1 * 1 + a0
def a1_a2_a3_a4_a5_condition : Prop := a1 + a2 + a3 + a4 + a5 = 33

-- We are required to prove that given these conditions, m = 3.
theorem find_m_value (h1 : a0 = -1) (h2 : poly_expansion_condition m a0 a1 a2 a3 a4 a5) 
(h3 : a1_a2_a3_a4_a5_condition a1 a2 a3 a4 a5) : m = 3 := by
  sorry

end find_m_value_l63_63407


namespace solve_equation_l63_63069

theorem solve_equation (x : ℝ) (h : (3*x^2 + 2*x + 1) / (x - 1) = 3*x + 1) : x = -1/2 :=
sorry

end solve_equation_l63_63069


namespace point_on_inverse_proportion_l63_63044

theorem point_on_inverse_proportion :
  ∀ (k x y : ℝ), 
    (∀ (x y: ℝ), (x = -2 ∧ y = 6) → y = k / x) →
    k = -12 →
    y = k / x →
    (x = 1 ∧ y = -12) :=
by
  sorry

end point_on_inverse_proportion_l63_63044


namespace magazines_per_bookshelf_l63_63337

noncomputable def total_books : ℕ := 23
noncomputable def total_books_and_magazines : ℕ := 2436
noncomputable def total_bookshelves : ℕ := 29

theorem magazines_per_bookshelf : (total_books_and_magazines - total_books) / total_bookshelves = 83 :=
by
  sorry

end magazines_per_bookshelf_l63_63337


namespace alcohol_percentage_after_additions_l63_63307

/-
Problem statement:
A 40-liter solution of alcohol and water is 5% alcohol. If 4.5 liters of alcohol and 5.5 liters of water are added to this solution, what percent of the solution produced is alcohol?

Conditions:
1. Initial solution volume = 40 liters
2. Initial percentage of alcohol = 5%
3. Volume of alcohol added = 4.5 liters
4. Volume of water added = 5.5 liters

Correct answer:
The percent of the solution that is alcohol after the additions is 13%.
-/

theorem alcohol_percentage_after_additions (initial_volume : ℝ) (initial_percentage : ℝ) 
  (alcohol_added : ℝ) (water_added : ℝ) :
  initial_volume = 40 ∧ initial_percentage = 5 ∧ alcohol_added = 4.5 ∧ water_added = 5.5 →
  ((initial_percentage / 100 * initial_volume + alcohol_added) / (initial_volume + alcohol_added + water_added) * 100) = 13 :=
by simp; sorry

end alcohol_percentage_after_additions_l63_63307


namespace sum_binomial_coefficients_l63_63525

theorem sum_binomial_coefficients :
  let a := 1
  let b := 1
  let binomial := (2 * a + 2 * b)
  (binomial)^7 = 16384 := by
  -- Proof omitted
  sorry

end sum_binomial_coefficients_l63_63525


namespace find_value_of_N_l63_63461

theorem find_value_of_N (N : ℝ) : 
  2 * ((3.6 * N * 2.50) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002 → 
  N = 0.4800000000000001 :=
by
  sorry

end find_value_of_N_l63_63461


namespace domain_of_g_l63_63663

theorem domain_of_g :
  {x : ℝ | -6*x^2 - 7*x + 8 >= 0} = 
  {x : ℝ | (7 - Real.sqrt 241) / 12 ≤ x ∧ x ≤ (7 + Real.sqrt 241) / 12} :=
by
  sorry

end domain_of_g_l63_63663


namespace nolan_total_savings_l63_63643

-- Define the conditions given in the problem
def monthly_savings : ℕ := 3000
def number_of_months : ℕ := 12

-- State the equivalent proof problem in Lean 4
theorem nolan_total_savings : (monthly_savings * number_of_months) = 36000 := by
  -- Proof is omitted
  sorry

end nolan_total_savings_l63_63643


namespace molecular_weight_8_moles_N2O_l63_63396

-- Definitions for atomic weights and the number of moles
def atomic_weight_N : Float := 14.01
def atomic_weight_O : Float := 16.00
def moles_N2O : Float := 8.0

-- Definition for molecular weight of N2O
def molecular_weight_N2O : Float := 
  (2 * atomic_weight_N) + (1 * atomic_weight_O)

-- Target statement to prove
theorem molecular_weight_8_moles_N2O :
  moles_N2O * molecular_weight_N2O = 352.16 :=
by
  sorry

end molecular_weight_8_moles_N2O_l63_63396


namespace evaluate_expression_l63_63288

theorem evaluate_expression : (120 / 6 * 2 / 3 = (40 / 3)) := 
by sorry

end evaluate_expression_l63_63288


namespace smallest_number_sum_of_three_squares_distinct_ways_l63_63338

theorem smallest_number_sum_of_three_squares_distinct_ways :
  ∃ n : ℤ, n = 30 ∧
  (∃ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℤ),
    a1^2 + b1^2 + c1^2 = n ∧
    a2^2 + b2^2 + c2^2 = n ∧
    a3^2 + b3^2 + c3^2 = n ∧
    (a1, b1, c1) ≠ (a2, b2, c2) ∧
    (a1, b1, c1) ≠ (a3, b3, c3) ∧
    (a2, b2, c2) ≠ (a3, b3, c3)) := sorry

end smallest_number_sum_of_three_squares_distinct_ways_l63_63338


namespace smallest_c_l63_63958

theorem smallest_c {a b c : ℤ} (h1 : a < b) (h2 : b < c) 
  (h3 : 2 * b = a + c)
  (h4 : a^2 = c * b) : c = 4 :=
by
  -- We state the theorem here without proof. 
  -- The actual proof steps are omitted and replaced by sorry.
  sorry

end smallest_c_l63_63958


namespace box_volume_max_l63_63240

noncomputable def volume (a x : ℝ) : ℝ :=
  (a - 2 * x) ^ 2 * x

theorem box_volume_max (a : ℝ) (h : 0 < a) :
  ∃ x, 0 < x ∧ x < a / 2 ∧ volume a x = volume a (a / 6) ∧ volume a (a / 6) = (2 * a^3) / 27 :=
by
  sorry

end box_volume_max_l63_63240


namespace smallest_value_of_Q_l63_63146

noncomputable def Q (x : ℝ) : ℝ := x^4 - 4*x^3 + 7*x^2 - 2*x + 10

theorem smallest_value_of_Q :
  min (Q 1) (min (10 : ℝ) (min (4 : ℝ) (min (1 - 4 + 7 - 2 + 10 : ℝ) (2.5 : ℝ)))) = 2.5 :=
by sorry

end smallest_value_of_Q_l63_63146


namespace total_cost_first_3_years_l63_63171

def monthly_fee : ℕ := 12
def down_payment : ℕ := 50
def years : ℕ := 3

theorem total_cost_first_3_years :
  (years * 12 * monthly_fee + down_payment) = 482 :=
by
  sorry

end total_cost_first_3_years_l63_63171


namespace curry_draymond_ratio_l63_63113

theorem curry_draymond_ratio :
  ∃ (curry draymond kelly durant klay : ℕ),
    draymond = 12 ∧
    kelly = 9 ∧
    durant = 2 * kelly ∧
    klay = draymond / 2 ∧
    curry + draymond + kelly + durant + klay = 69 ∧
    curry = 24 ∧ -- Curry's points calculated in the solution
    draymond = 12 → -- Draymond's points reaffirmed
    curry / draymond = 2 :=
by
  sorry

end curry_draymond_ratio_l63_63113


namespace arithmetic_sequence_k_l63_63247

theorem arithmetic_sequence_k (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (ha : ∀ n, S (n + 1) = S n + a (n + 1))
  (hS3_S8 : S 3 = S 8) 
  (hS7_Sk : ∃ k, S 7 = S k)
  : ∃ k, k = 4 :=
by
  sorry

end arithmetic_sequence_k_l63_63247


namespace find_value_of_y_l63_63858

theorem find_value_of_y (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := 
by {
  sorry
}

end find_value_of_y_l63_63858


namespace no_solution_a_squared_plus_b_squared_eq_2023_l63_63309

theorem no_solution_a_squared_plus_b_squared_eq_2023 :
  ∀ (a b : ℤ), a^2 + b^2 ≠ 2023 := 
by
  sorry

end no_solution_a_squared_plus_b_squared_eq_2023_l63_63309


namespace determine_all_cards_l63_63413

noncomputable def min_cards_to_determine_positions : ℕ :=
  2

theorem determine_all_cards {k : ℕ} (h : k = min_cards_to_determine_positions) :
  ∀ (placed_cards : ℕ → ℕ × ℕ),
  (∀ n, 1 ≤ n ∧ n ≤ 300 → placed_cards n = placed_cards (n + 1) ∨ placed_cards n + (1, 0) = placed_cards (n + 1) ∨ placed_cards n + (0, 1) = placed_cards (n + 1))
  → k = 2 :=
by
  sorry

end determine_all_cards_l63_63413


namespace rachel_points_product_l63_63370

-- Define the scores in the first 10 games
def scores_first_10_games := [9, 5, 7, 4, 8, 6, 2, 3, 5, 6]

-- Define the conditions as given in the problem
def total_score_first_10_games := scores_first_10_games.sum = 55
def points_scored_in_game_11 (P₁₁ : ℕ) : Prop := P₁₁ < 10 ∧ (55 + P₁₁) % 11 = 0
def points_scored_in_game_12 (P₁₁ P₁₂ : ℕ) : Prop := P₁₂ < 10 ∧ (55 + P₁₁ + P₁₂) % 12 = 0

-- Prove the product of the points scored in eleventh and twelfth games
theorem rachel_points_product : ∃ P₁₁ P₁₂ : ℕ, total_score_first_10_games ∧ points_scored_in_game_11 P₁₁ ∧ points_scored_in_game_12 P₁₁ P₁₂ ∧ P₁₁ * P₁₂ = 0 :=
by 
  sorry -- proof not required

end rachel_points_product_l63_63370


namespace min_value_expression_l63_63077

theorem min_value_expression (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : x * y * z = 2 / 3) :
  x^2 + 6 * x * y + 18 * y^2 + 12 * y * z + 4 * z^2 ≥ 18 :=
by
  sorry

end min_value_expression_l63_63077


namespace stable_number_divisible_by_11_l63_63933

/-- Definition of a stable number as a three-digit number (cen, ten, uni) where
    each digit is non-zero, and the sum of any two digits is greater than the remaining digit.
-/
def is_stable_number (cen ten uni : ℕ) : Prop :=
cen ≠ 0 ∧ ten ≠ 0 ∧ uni ≠ 0 ∧
(cen + ten > uni) ∧ (cen + uni > ten) ∧ (ten + uni > cen)

/-- Function F defined for a stable number (cen ten uni). -/
def F (cen ten uni : ℕ) : ℕ := 10 * ten + cen + uni

/-- Function Q defined for a stable number (cen ten uni). -/
def Q (cen ten uni : ℕ) : ℕ := 10 * cen + ten + uni

/-- Statement to prove: Given a stable number s = 100a + 101b + 30 where 1 ≤ a ≤ 5 and 1 ≤ b ≤ 4,
    the expression 5 * F(s) + 2 * Q(s) is divisible by 11.
-/
theorem stable_number_divisible_by_11 (a b cen ten uni : ℕ)
  (h_a : 1 ≤ a ∧ a ≤ 5)
  (h_b : 1 ≤ b ∧ b ≤ 4)
  (h_s : 100 * a + 101 * b + 30 = 100 * cen + 10 * ten + uni)
  (h_stable : is_stable_number cen ten uni) :
  (5 * F cen ten uni + 2 * Q cen ten uni) % 11 = 0 :=
sorry

end stable_number_divisible_by_11_l63_63933


namespace find_theta_l63_63162

theorem find_theta
  (θ : ℝ)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (ha : ∃ k, (2 * Real.cos θ, 2 * Real.sin θ) = (k * 3, k * Real.sqrt 3)) :
  θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6 :=
by
  sorry

end find_theta_l63_63162


namespace find_a_range_l63_63041

variable (a k : ℝ)
variable (x : ℝ) (hx : x > 0)

def p := ∀ x > 0, x + a / x ≥ 2
def q := ∀ k : ℝ, ∃ x y : ℝ, k * x - y + 2 = 0 ∧ x^2 + y^2 / a^2 = 1

theorem find_a_range :
  (a > 0) →
  ((p a) ∨ (q a)) ∧ ¬ ((p a) ∧ (q a)) ↔ 1 ≤ a ∧ a < 2 :=
sorry

end find_a_range_l63_63041


namespace intersecting_lines_l63_63348

theorem intersecting_lines (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 3 * x * y ↔ (x = 0 ∨ y = 0) := by
  sorry

end intersecting_lines_l63_63348


namespace prove_min_max_A_l63_63966

theorem prove_min_max_A : 
  ∃ (A_max A_min : ℕ), 
  (∃ B : ℕ, 
    A_max = 10^8 * (B % 10) + (B / 10) ∧ 
    B.gcd 24 = 1 ∧ 
    B > 666666666 ∧
    B % 10 = 9) ∧ 
  (∃ B : ℕ, 
    A_min = 10^8 * (B % 10) + (B / 10) ∧ 
    B.gcd 24 = 1 ∧ 
    B > 666666666 ∧ 
    B % 10 = 1) ∧ 
  A_max = 999999998 ∧ 
  A_min = 166666667 := sorry

end prove_min_max_A_l63_63966


namespace photo_album_slots_l63_63320

def photos_from_cristina : Nat := 7
def photos_from_john : Nat := 10
def photos_from_sarah : Nat := 9
def photos_from_clarissa : Nat := 14

theorem photo_album_slots :
  photos_from_cristina + photos_from_john + photos_from_sarah + photos_from_clarissa = 40 :=
by
  sorry

end photo_album_slots_l63_63320


namespace brick_wall_problem_l63_63120

theorem brick_wall_problem : 
  ∀ (B1 B2 B3 B4 B5 : ℕ) (d : ℕ),
  B1 = 38 →
  B1 + B2 + B3 + B4 + B5 = 200 →
  B2 = B1 - d →
  B3 = B1 - 2 * d →
  B4 = B1 - 3 * d →
  B5 = B1 - 4 * d →
  d = 1 :=
by
  intros B1 B2 B3 B4 B5 d h1 h2 h3 h4 h5 h6
  rw [h1] at h2
  sorry

end brick_wall_problem_l63_63120


namespace max_students_distributing_items_l63_63086

-- Define the given conditions
def pens : Nat := 1001
def pencils : Nat := 910

-- Define the statement
theorem max_students_distributing_items :
  Nat.gcd pens pencils = 91 :=
by
  sorry

end max_students_distributing_items_l63_63086


namespace find_k_l63_63082

theorem find_k (k : ℝ) (x₁ x₂ : ℝ) (h_distinct_roots : (2*k + 3)^2 - 4*k^2 > 0)
  (h_roots : ∀ (x : ℝ), x^2 + (2*k + 3)*x + k^2 = 0 ↔ x = x₁ ∨ x = x₂)
  (h_reciprocal_sum : 1/x₁ + 1/x₂ = -1) : k = 3 :=
by
  sorry

end find_k_l63_63082


namespace solve_system_of_equations_l63_63534

variable (a b c : Real)

def K : Real := a * b * c + a^2 * c + c^2 * b + b^2 * a

theorem solve_system_of_equations 
    (h₁ : (a + b) * (a - b) * (b + c) * (b - c) * (c + a) * (c - a) ≠ 0)
    (h₂ : K a b c ≠ 0) :
    ∃ (x y z : Real), 
    x = b^2 - c^2 ∧
    y = c^2 - a^2 ∧
    z = a^2 - b^2 ∧
    (x / (b + c) + y / (c - a) = a + b) ∧
    (y / (c + a) + z / (a - b) = b + c) ∧
    (z / (a + b) + x / (b - c) = c + a) :=
by
  sorry

end solve_system_of_equations_l63_63534


namespace vector_norm_sq_sum_l63_63376

theorem vector_norm_sq_sum (a b : ℝ × ℝ) (m : ℝ × ℝ) (h_m : m = (4, 6))
  (h_midpoint : m = ((2 * a.1 + 2 * b.1) / 2, (2 * a.2 + 2 * b.2) / 2))
  (h_dot : a.1 * b.1 + a.2 * b.2 = 10) :
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 32 :=
by 
  sorry

end vector_norm_sq_sum_l63_63376


namespace white_pawn_on_white_square_l63_63209

theorem white_pawn_on_white_square (w b N_b N_w : ℕ) (h1 : w > b) (h2 : N_b < N_w) : ∃ k : ℕ, k > 0 :=
by 
  -- Let's assume a contradiction
  -- The proof steps would be written here
  sorry

end white_pawn_on_white_square_l63_63209


namespace count_oddly_powerful_integers_l63_63849

def is_oddly_powerful (m : ℕ) : Prop :=
  ∃ (c d : ℕ), d > 1 ∧ d % 2 = 1 ∧ c^d = m

theorem count_oddly_powerful_integers :
  ∃ (S : Finset ℕ), 
  (∀ m, m ∈ S ↔ (m < 1500 ∧ is_oddly_powerful m)) ∧ S.card = 13 :=
by
  sorry

end count_oddly_powerful_integers_l63_63849


namespace interest_rate_eq_ten_l63_63778

theorem interest_rate_eq_ten (R : ℝ) (P : ℝ) (SI CI : ℝ) :
  P = 1400 ∧
  SI = 14 * R ∧
  CI = 1400 * ((1 + R / 200) ^ 2 - 1) ∧
  CI - SI = 3.50 → 
  R = 10 :=
by
  sorry

end interest_rate_eq_ten_l63_63778


namespace two_pow_n_plus_one_divisible_by_three_l63_63662

theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h1 : n > 0) :
  (2 ^ n + 1) % 3 = 0 ↔ n % 2 = 1 := 
sorry

end two_pow_n_plus_one_divisible_by_three_l63_63662


namespace intersection_M_N_l63_63603

def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} :=
sorry

end intersection_M_N_l63_63603


namespace longest_chord_in_circle_l63_63756

theorem longest_chord_in_circle {O : Type} (radius : ℝ) (h_radius : radius = 3) :
  ∃ d, d = 6 ∧ ∀ chord, chord <= d :=
by sorry

end longest_chord_in_circle_l63_63756


namespace sum_symmetric_prob_43_l63_63030

def prob_symmetric_sum_43_with_20 : Prop :=
  let n_dice := 9
  let min_sum := n_dice * 1
  let max_sum := n_dice * 6
  let midpoint := (min_sum + max_sum) / 2
  let symmetric_sum := 2 * midpoint - 20
  symmetric_sum = 43

theorem sum_symmetric_prob_43 (n_dice : ℕ) (h₁ : n_dice = 9) (h₂ : ∀ i : ℕ, i ≥ 1 ∧ i ≤ 6) :
  prob_symmetric_sum_43_with_20 :=
by
  sorry

end sum_symmetric_prob_43_l63_63030


namespace value_of_2a_minus_1_l63_63318

theorem value_of_2a_minus_1 (a : ℝ) (h : ∀ x : ℝ, (x = 2 → (3 / 2) * x - 2 * a = 0)) : 2 * a - 1 = 2 :=
sorry

end value_of_2a_minus_1_l63_63318


namespace difference_of_roots_l63_63387

noncomputable def r_and_s (r s : ℝ) : Prop :=
(∃ (r s : ℝ), (r, s) ≠ (s, r) ∧ r > s ∧ (5 * r - 15) / (r ^ 2 + 3 * r - 18) = r + 3
  ∧ (5 * s - 15) / (s ^ 2 + 3 * s - 18) = s + 3)

theorem difference_of_roots (r s : ℝ) (h : r_and_s r s) : r - s = Real.sqrt 29 := by
  sorry

end difference_of_roots_l63_63387


namespace tank_fraction_full_l63_63119

theorem tank_fraction_full 
  (initial_fraction : ℚ)
  (full_capacity : ℚ)
  (added_water : ℚ)
  (initial_fraction_eq : initial_fraction = 3/4)
  (full_capacity_eq : full_capacity = 40)
  (added_water_eq : added_water = 5) :
  ((initial_fraction * full_capacity + added_water) / full_capacity) = 7/8 :=
by 
  sorry

end tank_fraction_full_l63_63119


namespace rhombus_area_correct_l63_63449

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

theorem rhombus_area_correct :
  rhombus_area 30 12 = 180 :=
by
  sorry

end rhombus_area_correct_l63_63449


namespace assignment_statement_meaning_l63_63397

-- Define the meaning of the assignment statement
def is_assignment_statement (s: String) : Prop := s = "Variable = Expression"

-- Define the specific assignment statement we are considering
def assignment_statement : String := "i = i + 1"

-- Define the meaning of the specific assignment statement
def assignment_meaning (s: String) : Prop := s = "Add 1 to the original value of i and then assign it back to i, the value of i increases by 1"

-- The proof statement
theorem assignment_statement_meaning :
  is_assignment_statement "Variable = Expression" → assignment_meaning "i = i + 1" :=
by
  intros
  sorry

end assignment_statement_meaning_l63_63397


namespace min_value_of_f_l63_63037

noncomputable def f (x y z : ℝ) : ℝ :=
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)

theorem min_value_of_f (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : c * y + b * z = a) (h2 : a * z + c * x = b) (h3 : b * x + a * y = c) :
  ∃ x y z : ℝ, f x y z = 1 / 2 := sorry

end min_value_of_f_l63_63037


namespace number_of_common_divisors_l63_63408

theorem number_of_common_divisors :
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  let divisors_count := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  gcd_ab = 420 ∧ divisors_count = 24 :=
by
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  have h1 : gcd_ab = 420 := sorry
  have h2 : (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 24 := by norm_num
  exact ⟨h1, h2⟩

end number_of_common_divisors_l63_63408


namespace inequality_proof_l63_63709

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1)
                        (hb : 0 ≤ b) (hb1 : b ≤ 1)
                        (hc : 0 ≤ c) (hc1 : c ≤ 1) :
    (a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1) :=
by 
  sorry

end inequality_proof_l63_63709


namespace square_area_l63_63369

theorem square_area (x1 x2 : ℝ) (hx1 : x1^2 + 4 * x1 + 3 = 8) (hx2 : x2^2 + 4 * x2 + 3 = 8) (h_eq : y = 8) : 
  (|x1 - x2|) ^ 2 = 36 :=
sorry

end square_area_l63_63369


namespace beverage_price_function_l63_63087

theorem beverage_price_function (box_price : ℕ) (bottles_per_box : ℕ) (bottles_purchased : ℕ) (y : ℕ) :
  box_price = 55 →
  bottles_per_box = 6 →
  y = (55 * bottles_purchased) / 6 := 
sorry

end beverage_price_function_l63_63087


namespace simplify_and_evaluate_expr_l63_63774

theorem simplify_and_evaluate_expr (a b : ℤ) (h₁ : a = -1) (h₂ : b = 2) :
  (2 * a + b - 2 * (3 * a - 2 * b)) = 14 := by
  rw [h₁, h₂]
  sorry

end simplify_and_evaluate_expr_l63_63774


namespace initial_shares_bought_l63_63088

variable (x : ℕ) -- x is the number of shares Tom initially bought

-- Conditions:
def initial_cost_per_share : ℕ := 3
def num_shares_sold : ℕ := 10
def selling_price_per_share : ℕ := 4
def doubled_value_per_remaining_share : ℕ := 2 * initial_cost_per_share
def total_profit : ℤ := 40

-- Proving the number of shares initially bought
theorem initial_shares_bought (h : num_shares_sold * selling_price_per_share - x * initial_cost_per_share = total_profit) :
  x = 10 := by sorry

end initial_shares_bought_l63_63088


namespace opposite_face_number_l63_63373

theorem opposite_face_number (sum_faces : ℕ → ℕ → ℕ) (face_number : ℕ → ℕ) :
  (face_number 1 = 6) ∧ (face_number 2 = 7) ∧ (face_number 3 = 8) ∧ 
  (face_number 4 = 9) ∧ (face_number 5 = 10) ∧ (face_number 6 = 11) →
  (sum_faces 1 2 + sum_faces 3 4 + sum_faces 5 6 = 33 + 18) →
  (sum_faces 1 2 + sum_faces 3 4 + sum_faces 5 6 = 35 + 16) →
  (face_number 2 ≠ 9 ∨ face_number 2 ≠ 11) → 
  face_number 2 = 9 ∨ face_number 2 = 11 :=
by
  intros hface_numbers hsum1 hsum2 hnot_possible
  sorry

end opposite_face_number_l63_63373


namespace ball_price_equation_l63_63173

structure BallPrices where
  (x : Real) -- price of each soccer ball in yuan
  (condition1 : ∀ (x : Real), (1500 / (x + 20) - 800 / x = 5))

/-- Prove that the equation follows from the given conditions. -/
theorem ball_price_equation (b : BallPrices) : 1500 / (b.x + 20) - 800 / b.x = 5 := 
by sorry

end ball_price_equation_l63_63173
