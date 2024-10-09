import Mathlib

namespace shade_half_grid_additional_squares_l1905_190524

/-- A 4x5 grid consists of 20 squares, of which 3 are already shaded. 
Prove that the number of additional 1x1 squares needed to shade half the grid is 7. -/
theorem shade_half_grid_additional_squares (total_squares shaded_squares remaining_squares: ℕ) 
  (h1 : total_squares = 4 * 5)
  (h2 : shaded_squares = 3)
  (h3 : remaining_squares = total_squares / 2 - shaded_squares) :
  remaining_squares = 7 :=
by
  -- Proof not required.
  sorry

end shade_half_grid_additional_squares_l1905_190524


namespace months_in_season_l1905_190597

/-- Definitions for conditions in the problem --/
def total_games_per_month : ℝ := 323.0
def total_games_season : ℝ := 5491.0

/-- The statement to be proven: The number of months in the season --/
theorem months_in_season (x : ℝ) (h : x = total_games_season / total_games_per_month) : x = 17.0 := by
  sorry

end months_in_season_l1905_190597


namespace max_distance_between_circle_centers_l1905_190507

theorem max_distance_between_circle_centers :
  let rect_width := 20
  let rect_height := 16
  let circle_diameter := 8
  let horiz_distance := rect_width - circle_diameter
  let vert_distance := rect_height - circle_diameter
  let max_distance := Real.sqrt (horiz_distance ^ 2 + vert_distance ^ 2)
  max_distance = 4 * Real.sqrt 13 :=
by
  sorry

end max_distance_between_circle_centers_l1905_190507


namespace second_derivative_l1905_190536

noncomputable def y (x : ℝ) : ℝ := x^3 + Real.log x / Real.log 2 + Real.exp (-x)

theorem second_derivative (x : ℝ) : (deriv^[2] y x) = 3 * x^2 + (1 / (x * Real.log 2)) - Real.exp (-x) :=
by
  sorry

end second_derivative_l1905_190536


namespace factor_expression_l1905_190528

theorem factor_expression (x y z : ℝ) : 
  ( (x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3 ) / 
  ( (x - y)^3 + (y - z)^3 + (z - x)^3 ) = 
  (x + y) * (y + z) * (z + x) := 
by
  sorry

end factor_expression_l1905_190528


namespace right_triangle_expression_l1905_190523

theorem right_triangle_expression (a c b : ℝ) (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : 
  b^2 = 4 * (a + 1) :=
by
  sorry

end right_triangle_expression_l1905_190523


namespace distance_in_scientific_notation_l1905_190590

-- Definition for the number to be expressed in scientific notation
def distance : ℝ := 55000000

-- Expressing the number in scientific notation
def scientific_notation : ℝ := 5.5 * (10 ^ 7)

-- Theorem statement asserting the equality
theorem distance_in_scientific_notation : distance = scientific_notation :=
  by
  -- Proof not required here, so we leave it as sorry
  sorry

end distance_in_scientific_notation_l1905_190590


namespace marble_count_l1905_190522

theorem marble_count (x : ℝ) (h1 : 0.5 * x = x / 2) (h2 : 0.25 * x = x / 4) (h3 : (27 + 14) = 41) (h4 : 0.25 * x = 27 + 14)
  : x = 164 :=
by
  sorry

end marble_count_l1905_190522


namespace angle_terminal_side_equiv_l1905_190542

theorem angle_terminal_side_equiv (k : ℤ) : 
  ∀ θ α : ℝ, θ = - (π / 3) → α = 5 * π / 3 → α = θ + 2 * k * π := by
  intro θ α hθ hα
  sorry

end angle_terminal_side_equiv_l1905_190542


namespace problem_statement_l1905_190511

theorem problem_statement (x y : ℝ) (h : x / y = 2) : (x - y) / x = 1 / 2 :=
by
  sorry

end problem_statement_l1905_190511


namespace negation_of_universal_l1905_190549

theorem negation_of_universal :
  ¬(∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end negation_of_universal_l1905_190549


namespace equivalent_single_discount_calculation_l1905_190535

-- Definitions for the successive discounts
def discount10 (x : ℝ) : ℝ := 0.90 * x
def discount15 (x : ℝ) : ℝ := 0.85 * x
def discount25 (x : ℝ) : ℝ := 0.75 * x

-- Final price after applying all discounts
def final_price (x : ℝ) : ℝ := discount25 (discount15 (discount10 x))

-- Equivalent single discount fraction
def equivalent_discount (x : ℝ) : ℝ := 0.57375 * x

theorem equivalent_single_discount_calculation (x : ℝ) : 
  final_price x = equivalent_discount x :=
sorry

end equivalent_single_discount_calculation_l1905_190535


namespace salt_solution_mixture_l1905_190537

/-- Let's define the conditions and hypotheses required for our proof. -/
def ounces_of_salt_solution 
  (percent_salt : ℝ) (amount : ℝ) : ℝ := percent_salt * amount

def final_amount (x : ℝ) : ℝ := x + 70
def final_salt_content (x : ℝ) : ℝ := 0.40 * (x + 70)

theorem salt_solution_mixture (x : ℝ) :
  0.60 * x + 0.20 * 70 = 0.40 * (x + 70) ↔ x = 70 :=
by {
  sorry
}

end salt_solution_mixture_l1905_190537


namespace crow_distance_l1905_190541

theorem crow_distance (trips: ℕ) (hours: ℝ) (speed: ℝ) (distance: ℝ) :
  trips = 15 → hours = 1.5 → speed = 4 → (trips * 2 * distance) = (speed * hours) → distance = 200 / 1000 :=
by
  intros h_trips h_hours h_speed h_eq
  sorry

end crow_distance_l1905_190541


namespace P_subset_Q_l1905_190568

-- Define the set P
def P := {x : ℝ | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 1}

-- Define the set Q
def Q := {x : ℝ | x ≤ 2}

-- Prove P ⊆ Q
theorem P_subset_Q : P ⊆ Q :=
by
  sorry

end P_subset_Q_l1905_190568


namespace greatest_divisor_condition_l1905_190559

-- Define conditions
def leaves_remainder (a b k : ℕ) : Prop := ∃ q : ℕ, a = b * q + k

-- Define the greatest common divisor property
def gcd_of (a b k: ℕ) (g : ℕ) : Prop :=
  leaves_remainder a k g ∧ leaves_remainder b k g ∧ ∀ d : ℕ, (leaves_remainder a k d ∧ leaves_remainder b k d) → d ≤ g

theorem greatest_divisor_condition 
  (N : ℕ) (h1 : leaves_remainder 1657 N 6) (h2 : leaves_remainder 2037 N 5) :
  N = 127 :=
sorry

end greatest_divisor_condition_l1905_190559


namespace find_N_l1905_190527

theorem find_N (N : ℕ) : (4^5)^2 * (2^5)^4 = 2^N → N = 30 :=
by
  intros h
  -- Sorry to skip the proof.
  sorry

end find_N_l1905_190527


namespace number_of_10_people_rows_l1905_190504

theorem number_of_10_people_rows (x r : ℕ) (h1 : r = 54) (h2 : ∀ i : ℕ, i * 9 + x * 10 = 54) : x = 0 :=
by
  sorry

end number_of_10_people_rows_l1905_190504


namespace molecular_weight_CaSO4_2H2O_l1905_190546

def Ca := 40.08
def S := 32.07
def O := 16.00
def H := 1.008

def Ca_weight := 1 * Ca
def S_weight := 1 * S
def O_in_sulfate_weight := 4 * O
def O_in_water_weight := 4 * O
def H_in_water_weight := 4 * H

def total_weight := Ca_weight + S_weight + O_in_sulfate_weight + O_in_water_weight + H_in_water_weight

theorem molecular_weight_CaSO4_2H2O : total_weight = 204.182 := 
by {
  sorry
}

end molecular_weight_CaSO4_2H2O_l1905_190546


namespace smallest_n_circle_l1905_190550

theorem smallest_n_circle (n : ℕ) 
    (h1 : ∀ i j : ℕ, i < j → j - i = 3 ∨ j - i = 4 ∨ j - i = 5) :
    n = 7 :=
sorry

end smallest_n_circle_l1905_190550


namespace one_of_a_b_c_is_one_l1905_190591

theorem one_of_a_b_c_is_one (a b c : ℝ) (h1 : a * b * c = 1) (h2 : a + b + c = (1 / a) + (1 / b) + (1 / c)) :
  a = 1 ∨ b = 1 ∨ c = 1 :=
by
  sorry -- proof to be filled in

end one_of_a_b_c_is_one_l1905_190591


namespace rosa_parks_food_drive_l1905_190573

theorem rosa_parks_food_drive :
  ∀ (total_students students_collected_12_cans students_collected_none students_remaining total_cans cans_collected_first_group total_cans_first_group total_cans_last_group cans_per_student_last_group : ℕ),
    total_students = 30 →
    students_collected_12_cans = 15 →
    students_collected_none = 2 →
    students_remaining = total_students - students_collected_12_cans - students_collected_none →
    total_cans = 232 →
    cans_collected_first_group = 12 →
    total_cans_first_group = students_collected_12_cans * cans_collected_first_group →
    total_cans_last_group = total_cans - total_cans_first_group →
    cans_per_student_last_group = total_cans_last_group / students_remaining →
    cans_per_student_last_group = 4 :=
by
  intros total_students students_collected_12_cans students_collected_none students_remaining total_cans cans_collected_first_group total_cans_first_group total_cans_last_group cans_per_student_last_group
  sorry

end rosa_parks_food_drive_l1905_190573


namespace fraction_is_irreducible_l1905_190599

theorem fraction_is_irreducible :
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16 : ℚ) / 
   (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 :=
by 
  sorry

end fraction_is_irreducible_l1905_190599


namespace dig_time_comparison_l1905_190575

open Nat

theorem dig_time_comparison :
  (3 * 420 / 9) - (5 * 40 / 2) = 40 :=
by
  sorry

end dig_time_comparison_l1905_190575


namespace solve_x_l1905_190510

theorem solve_x (x : ℝ) (A B : ℝ × ℝ) (a : ℝ × ℝ) 
  (hA : A = (1, 3)) (hB : B = (2, 4))
  (ha : a = (2 * x - 1, x ^ 2 + 3 * x - 3))
  (hab : a = (B.1 - A.1, B.2 - A.2)) : x = 1 :=
by {
  sorry
}

end solve_x_l1905_190510


namespace concentration_time_within_bounds_l1905_190547

-- Define the time bounds for the highest concentration of the drug in the blood
def highest_concentration_time_lower (base : ℝ) (tolerance : ℝ) : ℝ := base - tolerance
def highest_concentration_time_upper (base : ℝ) (tolerance : ℝ) : ℝ := base + tolerance

-- Define the base and tolerance values
def base_time : ℝ := 0.65
def tolerance_time : ℝ := 0.15

-- Define the specific time we want to prove is within the bounds
def specific_time : ℝ := 0.8

-- Theorem statement
theorem concentration_time_within_bounds : 
  highest_concentration_time_lower base_time tolerance_time ≤ specific_time ∧ 
  specific_time ≤ highest_concentration_time_upper base_time tolerance_time :=
by sorry

end concentration_time_within_bounds_l1905_190547


namespace am_gm_inequality_l1905_190526

theorem am_gm_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) :
  (1 - x) * (1 - y) * (1 - z) ≥ 8 * x * y * z :=
by sorry

end am_gm_inequality_l1905_190526


namespace surveyed_individuals_not_working_percentage_l1905_190560

theorem surveyed_individuals_not_working_percentage :
  (55 / 100 * 0 + 35 / 100 * (1 / 8) + 10 / 100 * (1 / 4)) = 6.875 / 100 :=
by
  sorry

end surveyed_individuals_not_working_percentage_l1905_190560


namespace quotient_of_polynomial_l1905_190574

theorem quotient_of_polynomial (x : ℤ) :
  (x^6 + 8) = (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) + 9 :=
by { sorry }

end quotient_of_polynomial_l1905_190574


namespace price_difference_is_7_42_l1905_190592

def total_cost : ℝ := 80.34
def shirt_price : ℝ := 36.46
def sweater_price : ℝ := total_cost - shirt_price
def price_difference : ℝ := sweater_price - shirt_price

theorem price_difference_is_7_42 : price_difference = 7.42 :=
  by
    sorry

end price_difference_is_7_42_l1905_190592


namespace students_agreed_total_l1905_190586

theorem students_agreed_total :
  let third_grade_agreed : ℕ := 154
  let fourth_grade_agreed : ℕ := 237
  third_grade_agreed + fourth_grade_agreed = 391 := 
by
  let third_grade_agreed : ℕ := 154
  let fourth_grade_agreed : ℕ := 237
  show third_grade_agreed + fourth_grade_agreed = 391
  sorry

end students_agreed_total_l1905_190586


namespace magnitude_of_T_l1905_190594

theorem magnitude_of_T : 
  let i := Complex.I
  let T := 3 * ((1 + i) ^ 15 - (1 - i) ^ 15)
  Complex.abs T = 768 := by
  sorry

end magnitude_of_T_l1905_190594


namespace children_ticket_cost_is_8_l1905_190581

-- Defining the costs of different tickets
def adult_ticket_cost : ℕ := 11
def senior_ticket_cost : ℕ := 9
def total_tickets_cost : ℕ := 64

-- Number of tickets needed
def number_of_adult_tickets : ℕ := 2
def number_of_senior_tickets : ℕ := 2
def number_of_children_tickets : ℕ := 3

-- Defining the total cost equation using the price of children's tickets (C)
def total_cost (children_ticket_cost : ℕ) : ℕ :=
  number_of_adult_tickets * adult_ticket_cost +
  number_of_senior_tickets * senior_ticket_cost +
  number_of_children_tickets * children_ticket_cost

-- Statement to prove that the children's ticket cost is $8
theorem children_ticket_cost_is_8 : (C : ℕ) → total_cost C = total_tickets_cost → C = 8 :=
by
  intro C h
  sorry

end children_ticket_cost_is_8_l1905_190581


namespace sufficient_not_necessary_condition_l1905_190579

variables (a b : Line) (α β : Plane)

def Line : Type := sorry
def Plane : Type := sorry

-- Conditions: a and b are different lines, α and β are different planes
axiom diff_lines : a ≠ b
axiom diff_planes : α ≠ β

-- Perpendicular and parallel definitions
def perp (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry

-- Sufficient but not necessary condition
theorem sufficient_not_necessary_condition
  (h1 : perp a β)
  (h2 : parallel α β) :
  perp a α :=
sorry

end sufficient_not_necessary_condition_l1905_190579


namespace mod_pow_eq_l1905_190552

theorem mod_pow_eq (m : ℕ) (h1 : 13^4 % 11 = m) (h2 : 0 ≤ m ∧ m < 11) : m = 5 := by
  sorry

end mod_pow_eq_l1905_190552


namespace range_of_a_l1905_190572

theorem range_of_a 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : r > 0) 
  (cos_le_zero : (3 * a - 9) / r ≤ 0) 
  (sin_gt_zero : (a + 2) / r > 0) : 
  -2 < a ∧ a ≤ 3 :=
sorry

end range_of_a_l1905_190572


namespace total_apples_eaten_l1905_190505

def simone_consumption (days: ℕ) (consumption_per_day: ℚ) : ℚ := days * consumption_per_day
def lauri_consumption (days: ℕ) (consumption_per_day: ℚ) : ℚ := days * consumption_per_day

theorem total_apples_eaten :
  simone_consumption 16 (1/2) + lauri_consumption 15 (1/3) = 13 := by
  sorry

end total_apples_eaten_l1905_190505


namespace loaned_books_during_month_l1905_190551

-- Definitions corresponding to the conditions
def initial_books : ℕ := 75
def returned_percent : ℚ := 0.65
def end_books : ℕ := 68

-- Proof statement
theorem loaned_books_during_month (x : ℕ) 
  (h1 : returned_percent = 0.65)
  (h2 : initial_books = 75)
  (h3 : end_books = 68) :
  (0.35 * x : ℚ) = (initial_books - end_books) :=
sorry

end loaned_books_during_month_l1905_190551


namespace alice_height_after_growth_l1905_190501

/-- Conditions: Bob and Alice were initially the same height. Bob has grown by 25%, Alice 
has grown by one third as many inches as Bob, and Bob is now 75 inches tall. --/
theorem alice_height_after_growth (initial_height : ℕ)
  (bob_growth_rate : ℚ)
  (alice_growth_ratio : ℚ)
  (bob_final_height : ℕ) :
  bob_growth_rate = 0.25 →
  alice_growth_ratio = 1 / 3 →
  bob_final_height = 75 →
  initial_height + (bob_final_height - initial_height) / 3 = 65 :=
by
  sorry

end alice_height_after_growth_l1905_190501


namespace find_nat_pair_l1905_190582

theorem find_nat_pair (a b : ℕ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a = 2^155) (h₄ : b = 3^65) : a^13 * b^31 = 6^2015 :=
by {
  sorry
}

end find_nat_pair_l1905_190582


namespace find_C_l1905_190553

-- Define the sum of interior angles of a triangle
def sum_of_triangle_angles := 180

-- Define the total angles sum in a closed figure formed by multiple triangles
def total_internal_angles := 1080

-- Define the value to prove
def C := total_internal_angles - sum_of_triangle_angles

theorem find_C:
  C = 900 := by
  sorry

end find_C_l1905_190553


namespace f_at_pos_eq_l1905_190563

noncomputable def f (x : ℝ) : ℝ :=
  if h : x < 0 then x * (x - 1)
  else if h : x > 0 then -x * (x + 1)
  else 0

theorem f_at_pos_eq (x : ℝ) (hx : 0 < x) : f x = -x * (x + 1) :=
by
  -- Assume f is an odd function
  have h_odd : ∀ x : ℝ, f (-x) = -f x := sorry
  
  -- Given for x in (-∞, 0), f(x) = x * (x - 1)
  have h_neg : ∀ x : ℝ, x < 0 → f x = x * (x - 1) := sorry
  
  -- Prove for x > 0, f(x) = -x * (x + 1)
  sorry

end f_at_pos_eq_l1905_190563


namespace right_triangle_third_side_l1905_190514

theorem right_triangle_third_side (x : ℝ) : 
  (∃ (a b c : ℝ), (a = 3 ∧ b = 4 ∧ (a^2 + b^2 = c^2 ∧ (c = x ∨ x^2 + a^2 = b^2)))) → (x = 5 ∨ x = Real.sqrt 7) :=
by 
  sorry

end right_triangle_third_side_l1905_190514


namespace original_price_l1905_190548

theorem original_price (saving : ℝ) (percentage : ℝ) (h_saving : saving = 10) (h_percentage : percentage = 0.10) :
  ∃ OP : ℝ, OP = 100 :=
by
  sorry

end original_price_l1905_190548


namespace fifth_inequality_proof_l1905_190566

theorem fifth_inequality_proof : 
  1 + (1 / (2:ℝ)^2) + (1 / (3:ℝ)^2) + (1 / (4:ℝ)^2) + (1 / (5:ℝ)^2) + (1 / (6:ℝ)^2) < (11 / 6) :=
by {
  sorry
}

end fifth_inequality_proof_l1905_190566


namespace no_solution_system_of_equations_l1905_190515

theorem no_solution_system_of_equations :
  ¬ ∃ (x y : ℝ), (2 * x - 3 * y = 6) ∧ (4 * x - 6 * y = 8) ∧ (5 * x - 5 * y = 15) :=
by {
  sorry
}

end no_solution_system_of_equations_l1905_190515


namespace totalHighlighters_l1905_190520

-- Define the number of each type of highlighter
def pinkHighlighters : ℕ := 10
def yellowHighlighters : ℕ := 15
def blueHighlighters : ℕ := 8

-- State the theorem to prove
theorem totalHighlighters :
  pinkHighlighters + yellowHighlighters + blueHighlighters = 33 :=
by
  -- Proof to be filled
  sorry

end totalHighlighters_l1905_190520


namespace max_bio_homework_time_l1905_190558

-- Define our variables as non-negative real numbers
variables (B H G : ℝ)

-- Given conditions
axiom h1 : H = 2 * B
axiom h2 : G = 6 * B
axiom h3 : B + H + G = 180

-- We need to prove that B = 20
theorem max_bio_homework_time : B = 20 :=
by
  sorry

end max_bio_homework_time_l1905_190558


namespace cd_total_l1905_190576

theorem cd_total :
  ∀ (Kristine Dawn Mark Alice : ℕ),
  Dawn = 10 →
  Kristine = Dawn + 7 →
  Mark = 2 * Kristine →
  Alice = (Kristine + Mark) - 5 →
  (Dawn + Kristine + Mark + Alice) = 107 :=
by
  intros Kristine Dawn Mark Alice hDawn hKristine hMark hAlice
  rw [hDawn, hKristine, hMark, hAlice]
  sorry

end cd_total_l1905_190576


namespace combined_distance_l1905_190588

-- Definitions based on the conditions
def JulienDailyDistance := 50
def SarahDailyDistance := 2 * JulienDailyDistance
def JamirDailyDistance := SarahDailyDistance + 20
def Days := 7

-- Combined weekly distances
def JulienWeeklyDistance := JulienDailyDistance * Days
def SarahWeeklyDistance := SarahDailyDistance * Days
def JamirWeeklyDistance := JamirDailyDistance * Days

-- Theorem statement with the combined distance
theorem combined_distance :
  JulienWeeklyDistance + SarahWeeklyDistance + JamirWeeklyDistance = 1890 := by
  sorry

end combined_distance_l1905_190588


namespace zara_goats_l1905_190556

noncomputable def total_animals_per_group := 48
noncomputable def total_groups := 3
noncomputable def total_cows := 24
noncomputable def total_sheep := 7

theorem zara_goats : 
  (total_groups * total_animals_per_group = 144) ∧ 
  (144 = total_cows + total_sheep + 113) →
  113 = 144 - total_cows - total_sheep := 
by sorry

end zara_goats_l1905_190556


namespace find_mark_age_l1905_190540

-- Define Mark and Aaron's ages
variables (M A : ℕ)

-- The conditions
def condition1 : Prop := M - 3 = 3 * (A - 3) + 1
def condition2 : Prop := M + 4 = 2 * (A + 4) + 2

-- The proof statement
theorem find_mark_age (h1 : condition1 M A) (h2 : condition2 M A) : M = 28 :=
by sorry

end find_mark_age_l1905_190540


namespace integer_solution_count_l1905_190567

theorem integer_solution_count :
  ∃ n : ℕ, n = 10 ∧
  ∃ (x1 x2 x3 : ℕ), x1 + x2 + x3 = 15 ∧ (0 ≤ x1 ∧ x1 ≤ 5) ∧ (0 ≤ x2 ∧ x2 ≤ 6) ∧ (0 ≤ x3 ∧ x3 ≤ 7) := 
sorry

end integer_solution_count_l1905_190567


namespace largest_reciprocal_l1905_190518

-- Definitions for the given numbers
def a := 1/4
def b := 3/7
def c := 2
def d := 10
def e := 2023

-- Statement to prove the problem
theorem largest_reciprocal :
  (1/a) > (1/b) ∧ (1/a) > (1/c) ∧ (1/a) > (1/d) ∧ (1/a) > (1/e) :=
by
  sorry

end largest_reciprocal_l1905_190518


namespace find_value_l1905_190565

theorem find_value (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a^2006 + (a + b)^2007 = 2 := 
by
  sorry

end find_value_l1905_190565


namespace markers_multiple_of_4_l1905_190589

-- Definitions corresponding to conditions
def Lisa_has_12_coloring_books := 12
def Lisa_has_36_crayons := 36
def greatest_number_baskets := 4

-- Theorem statement
theorem markers_multiple_of_4
    (h1 : Lisa_has_12_coloring_books = 12)
    (h2 : Lisa_has_36_crayons = 36)
    (h3 : greatest_number_baskets = 4) :
    ∃ (M : ℕ), M % 4 = 0 :=
by
  sorry

end markers_multiple_of_4_l1905_190589


namespace sin_gamma_isosceles_l1905_190538

theorem sin_gamma_isosceles (a c m_a m_c s_1 s_2 : ℝ) (γ : ℝ) 
  (h1 : a + m_c = s_1) (h2 : c + m_a = s_2) :
  Real.sin γ = (s_2 / (2 * s_1)) * Real.sqrt ((4 * s_1^2) - s_2^2) :=
sorry

end sin_gamma_isosceles_l1905_190538


namespace A_runs_faster_l1905_190517

variable (v_A v_B : ℝ)  -- Speed of A and B
variable (k : ℝ)       -- Factor by which A is faster than B

-- Conditions as definitions in Lean:
def speed_relation (k : ℝ) (v_A v_B : ℝ) : Prop := v_A = k * v_B
def start_difference : ℝ := 60
def race_course_length : ℝ := 80
def reach_finish_same_time (v_A v_B : ℝ) : Prop := (80 / v_A) = ((80 - start_difference) / v_B)

theorem A_runs_faster
  (h1 : speed_relation k v_A v_B)
  (h2 : reach_finish_same_time v_A v_B) : k = 4 :=
by
  sorry

end A_runs_faster_l1905_190517


namespace solution_of_inequality_l1905_190531

theorem solution_of_inequality (a x : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  (x - a) * (x - a⁻¹) < 0 ↔ a < x ∧ x < a⁻¹ :=
by sorry

end solution_of_inequality_l1905_190531


namespace card_collection_average_l1905_190554

theorem card_collection_average (n : ℕ) (h : (2 * n + 1) / 3 = 2017) : n = 3025 :=
by
  sorry

end card_collection_average_l1905_190554


namespace exists_initial_segment_of_power_of_2_l1905_190521

theorem exists_initial_segment_of_power_of_2 (m : ℕ) : ∃ n : ℕ, ∃ k : ℕ, k ≥ m ∧ 2^n = 10^k * m ∨ 2^n = 10^k * (m+1) := 
by
  sorry

end exists_initial_segment_of_power_of_2_l1905_190521


namespace three_digit_number_proof_l1905_190596

noncomputable def is_prime (n : ℕ) : Prop := (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem three_digit_number_proof (H T U : ℕ) (h1 : H = 2 * T)
  (h2 : U = 2 * T^3)
  (h3 : is_prime (H + T + U))
  (h_digits : H < 10 ∧ T < 10 ∧ U < 10)
  (h_nonzero : T > 0) : H * 100 + T * 10 + U = 212 := 
by
  sorry

end three_digit_number_proof_l1905_190596


namespace binary_representation_88_l1905_190585

def binary_representation (n : Nat) : String := sorry

theorem binary_representation_88 : binary_representation 88 = "1011000" := sorry

end binary_representation_88_l1905_190585


namespace problem1_l1905_190598

theorem problem1 (a : ℝ) (m n : ℕ) (h1 : a^m = 10) (h2 : a^n = 2) : a^(m - 2 * n) = 2.5 := by
  sorry

end problem1_l1905_190598


namespace root_of_equation_l1905_190561

theorem root_of_equation : 
  ∀ x : ℝ, x ≠ 3 → x ≠ -3 → (18 / (x^2 - 9) - 3 / (x - 3) = 2) → (x = -4.5) :=
by sorry

end root_of_equation_l1905_190561


namespace work_completion_days_l1905_190509

structure WorkProblem :=
  (total_work : ℝ := 1) -- Assume total work to be 1 unit
  (days_A : ℝ := 30)
  (days_B : ℝ := 15)
  (days_together : ℝ := 5)

noncomputable def total_days_taken (wp : WorkProblem) : ℝ :=
  let work_per_day_A := 1 / wp.days_A
  let work_per_day_B := 1 / wp.days_B
  let work_per_day_together := work_per_day_A + work_per_day_B
  let work_done_together := wp.days_together * work_per_day_together
  let remaining_work := wp.total_work - work_done_together
  let days_for_A := remaining_work / work_per_day_A
  wp.days_together + days_for_A

theorem work_completion_days (wp : WorkProblem) : total_days_taken wp = 20 :=
by
  sorry

end work_completion_days_l1905_190509


namespace trail_length_is_20_km_l1905_190555

-- Define the conditions and the question
def length_of_trail (L : ℝ) (hiked_percentage remaining_distance : ℝ) : Prop :=
  hiked_percentage = 0.60 ∧ remaining_distance = 8 ∧ 0.40 * L = remaining_distance

-- The statement: given the conditions, prove that length of trail is 20 km
theorem trail_length_is_20_km : ∃ L : ℝ, length_of_trail L 0.60 8 ∧ L = 20 := by
  -- Proof goes here
  sorry

end trail_length_is_20_km_l1905_190555


namespace range_of_a_l1905_190569

theorem range_of_a (a : ℝ) :
  let A := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
  let B := {x : ℝ | 5 < x}
  (A ∩ B = ∅) ↔ a ∈ {a : ℝ | a ≤ 2 ∨ a > 3} :=
by
  sorry

end range_of_a_l1905_190569


namespace proof_problem_l1905_190516

def work_problem :=
  ∃ (B : ℝ),
  (1 / 6) + (1 / B) + (1 / 24) = (1 / 3) ∧ B = 8

theorem proof_problem : work_problem :=
by
  sorry

end proof_problem_l1905_190516


namespace bags_sold_in_first_week_l1905_190539

def total_bags_sold : ℕ := 100
def bags_sold_week1 (X : ℕ) : ℕ := X
def bags_sold_week2 (X : ℕ) : ℕ := 3 * X
def bags_sold_week3_4 : ℕ := 40

theorem bags_sold_in_first_week (X : ℕ) (h : total_bags_sold = bags_sold_week1 X + bags_sold_week2 X + bags_sold_week3_4) : X = 15 :=
by
  sorry

end bags_sold_in_first_week_l1905_190539


namespace trigonometric_inequality_l1905_190525

theorem trigonometric_inequality (x : ℝ) : 0 ≤ 5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ∧ 
                                            5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ≤ 18 :=
by
  sorry

end trigonometric_inequality_l1905_190525


namespace modulo_sum_remainder_l1905_190577

theorem modulo_sum_remainder (a b: ℤ) (k j: ℤ) 
  (h1 : a = 84 * k + 77) 
  (h2 : b = 120 * j + 113) :
  (a + b) % 42 = 22 := by
  sorry

end modulo_sum_remainder_l1905_190577


namespace coprime_n_minus_2_n_squared_minus_n_minus_1_l1905_190502

theorem coprime_n_minus_2_n_squared_minus_n_minus_1 (n : ℕ) : n - 2 ∣ n^2 - n - 1 → False :=
by
-- proof omitted as per instructions
sorry

end coprime_n_minus_2_n_squared_minus_n_minus_1_l1905_190502


namespace solve_for_n_l1905_190529

theorem solve_for_n (n x y : ℤ) (h : n * (x + y) + 17 = n * (-x + y) - 21) (hx : x = 1) : n = -19 :=
by
  sorry

end solve_for_n_l1905_190529


namespace non_degenerate_ellipse_l1905_190544

theorem non_degenerate_ellipse (k : ℝ) : (∃ a, a = -21) ↔ (k > -21) := by
  sorry

end non_degenerate_ellipse_l1905_190544


namespace number_of_solutions_eq_one_l1905_190508

theorem number_of_solutions_eq_one :
  (∃! y : ℝ, (y ≠ 0) ∧ (y ≠ 3) ∧ ((3 * y^2 - 15 * y) / (y^2 - 3 * y) = y + 1)) :=
  sorry

end number_of_solutions_eq_one_l1905_190508


namespace binomial_identity_l1905_190571

theorem binomial_identity :
  (Nat.choose 16 6 = 8008) → (Nat.choose 16 7 = 11440) → (Nat.choose 16 8 = 12870) →
  Nat.choose 18 8 = 43758 :=
by
  intros h1 h2 h3
  sorry

end binomial_identity_l1905_190571


namespace prob1_prob2_prob3_l1905_190562

-- Problem 1
theorem prob1 (f : ℝ → ℝ) (m : ℝ) (f_def : ∀ x, f x = Real.log x + (1/2) * m * x^2)
  (tangent_line_slope : ℝ) (perpendicular_line_eq : ℝ) :
  (tangent_line_slope = 1 + m) →
  (perpendicular_line_eq = -1/2) →
  (tangent_line_slope * perpendicular_line_eq = -1) →
  m = 1 := sorry

-- Problem 2
theorem prob2 (f : ℝ → ℝ) (m : ℝ) (f_def : ∀ x, f x = Real.log x + (1/2) * m * x^2) :
  (∀ x, f x ≤ m * x^2 + (m - 1) * x - 1) →
  ∃ (m_ : ℤ), m_ ≥ 2 := sorry

-- Problem 3
theorem prob3 (f : ℝ → ℝ) (F : ℝ → ℝ) (x1 x2 : ℝ) (m : ℝ) 
  (f_def : ∀ x, f x = Real.log x + (1/2) * x^2)
  (F_def : ∀ x, F x = f x + x)
  (hx1 : 0 < x1) (hx2: 0 < x2) :
  m = 1 →
  F x1 = -F x2 →
  x1 + x2 ≥ Real.sqrt 3 - 1 := sorry

end prob1_prob2_prob3_l1905_190562


namespace measure_of_angle_Q_in_hexagon_l1905_190583

theorem measure_of_angle_Q_in_hexagon :
  ∀ (Q : ℝ),
    (∃ (angles : List ℝ),
      angles = [134, 108, 122, 99, 87] ∧ angles.sum = 550) →
    180 * (6 - 2) - (134 + 108 + 122 + 99 + 87) = 170 → Q = 170 := by
  sorry

end measure_of_angle_Q_in_hexagon_l1905_190583


namespace silverware_probability_l1905_190584

-- Defining the number of each type of silverware
def num_forks : ℕ := 8
def num_spoons : ℕ := 10
def num_knives : ℕ := 4
def total_silverware : ℕ := num_forks + num_spoons + num_knives
def num_remove : ℕ := 4

-- Proving the probability calculation
theorem silverware_probability :
  -- Calculation of the total number of ways to choose 4 pieces from 22
  let total_ways := Nat.choose total_silverware num_remove
  -- Calculation of ways to choose 2 forks from 8
  let ways_to_choose_forks := Nat.choose num_forks 2
  -- Calculation of ways to choose 1 spoon from 10
  let ways_to_choose_spoon := Nat.choose num_spoons 1
  -- Calculation of ways to choose 1 knife from 4
  let ways_to_choose_knife := Nat.choose num_knives 1
  -- Calculation of the number of favorable outcomes
  let favorable_outcomes := ways_to_choose_forks * ways_to_choose_spoon * ways_to_choose_knife
  -- Probability in simplified form
  let probability := (favorable_outcomes : ℚ) / total_ways
  probability = (32 : ℚ) / 209 :=
by
  sorry

end silverware_probability_l1905_190584


namespace find_n_l1905_190532

-- We need a definition for permutations counting A_n^2 = n(n-1)
def permutations_squared (n : ℕ) : ℕ := n * (n - 1)

theorem find_n (n : ℕ) (h : permutations_squared n = 56) : n = 8 :=
by {
  sorry -- proof omitted as instructed
}

end find_n_l1905_190532


namespace smallest_positive_t_l1905_190545

theorem smallest_positive_t (x_1 x_2 x_3 x_4 x_5 t : ℝ) :
  (x_1 + x_3 = 2 * t * x_2) →
  (x_2 + x_4 = 2 * t * x_3) →
  (x_3 + x_5 = 2 * t * x_4) →
  (0 ≤ x_1) →
  (0 ≤ x_2) →
  (0 ≤ x_3) →
  (0 ≤ x_4) →
  (0 ≤ x_5) →
  (x_1 ≠ 0 ∨ x_2 ≠ 0 ∨ x_3 ≠ 0 ∨ x_4 ≠ 0 ∨ x_5 ≠ 0) →
  t = 1 / Real.sqrt 2 → 
  ∃ t, (0 < t) ∧ (x_1 + x_3 = 2 * t * x_2) ∧ (x_2 + x_4 = 2 * t * x_3) ∧ (x_3 + x_5 = 2 * t * x_4)
:=
sorry

end smallest_positive_t_l1905_190545


namespace no_m_for_necessary_and_sufficient_condition_m_geq_3_for_necessary_condition_l1905_190580

def P (x : ℝ) : Prop := x ^ 2 - 8 * x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem no_m_for_necessary_and_sufficient_condition :
  ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S x m :=
by sorry

theorem m_geq_3_for_necessary_condition :
  ∃ m : ℝ, (m ≥ 3) ∧ ∀ x : ℝ, S x m → P x :=
by sorry

end no_m_for_necessary_and_sufficient_condition_m_geq_3_for_necessary_condition_l1905_190580


namespace smaller_angle_parallelogram_l1905_190519

theorem smaller_angle_parallelogram (x : ℕ) (h1 : ∀ a b : ℕ, a ≠ b ∧ a + b = 180) (h2 : ∃ y : ℕ, y = x + 70) : x = 55 :=
by
  sorry

end smaller_angle_parallelogram_l1905_190519


namespace speed_of_faster_train_l1905_190593

-- Definitions based on the conditions.
def length_train_1 : ℝ := 180
def length_train_2 : ℝ := 360
def time_to_cross : ℝ := 21.598272138228943
def speed_slow_train_kmph : ℝ := 30
def speed_fast_train_kmph : ℝ := 60

-- The theorem that needs to be proven.
theorem speed_of_faster_train :
  (length_train_1 + length_train_2) / time_to_cross * 3.6 = speed_slow_train_kmph + speed_fast_train_kmph :=
sorry

end speed_of_faster_train_l1905_190593


namespace positive_number_property_l1905_190534

theorem positive_number_property (x : ℝ) (h_pos : 0 < x) (h_property : (x^2 / 100) = 9) : x = 30 :=
by
  sorry

end positive_number_property_l1905_190534


namespace proof_problem_l1905_190557

theorem proof_problem
  (a b : ℝ)
  (h1 : a = -(-3))
  (h2 : b = - (- (1 / 2))⁻¹)
  (m n : ℝ) :
  (|m - a| + |n + b| = 0) → (a = 3 ∧ b = -2 ∧ m = 3 ∧ n = -2) :=
by {
  sorry
}

end proof_problem_l1905_190557


namespace compute_expression_l1905_190506

theorem compute_expression (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end compute_expression_l1905_190506


namespace geometric_progression_exists_l1905_190564

theorem geometric_progression_exists :
  ∃ (b1 b2 b3 b4: ℤ) (q: ℤ), 
    b2 = b1 * q ∧ 
    b3 = b1 * q^2 ∧ 
    b4 = b1 * q^3 ∧  
    b3 - b1 = 9 ∧ 
    b2 - b4 = 18 ∧ 
    b1 = 3 ∧ b2 = -6 ∧ b3 = 12 ∧ b4 = -24 :=
sorry

end geometric_progression_exists_l1905_190564


namespace max_min_x1_x2_squared_l1905_190512

theorem max_min_x1_x2_squared (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 - (k-2)*x1 + (k^2 + 3*k + 5) = 0)
  (h2 : x2^2 - (k-2)*x2 + (k^2 + 3*k + 5) = 0)
  (h3 : -4 ≤ k ∧ k ≤ -4/3) : 
  (∃ (k_max k_min : ℝ), 
    k = -4 → x1^2 + x2^2 = 18 ∧ k = -4/3 → x1^2 + x2^2 = 50/9) :=
sorry

end max_min_x1_x2_squared_l1905_190512


namespace eval_expression_l1905_190500

theorem eval_expression (m : ℝ) (h : m^2 + 2*m - 1 = 0) : 2*m^2 + 4*m - 3 = -1 :=
by
  sorry

end eval_expression_l1905_190500


namespace problem_equivalent_l1905_190503

theorem problem_equivalent
  (x : ℚ)
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10) :
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 289 / 8 := 
by
  sorry

end problem_equivalent_l1905_190503


namespace product_of_roots_eq_neg35_l1905_190595

theorem product_of_roots_eq_neg35 (x : ℝ) : 
  (x + 3) * (x - 5) = 20 → ∃ a b c : ℝ, a ≠ 0 ∧ a * x^2 + b * x + c = 0 ∧ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1 * x2 = -35 := 
by
  sorry

end product_of_roots_eq_neg35_l1905_190595


namespace value_of_M_correct_l1905_190530

noncomputable def value_of_M : ℤ :=
  let d1 := 4        -- First column difference
  let d2 := -7       -- Row difference
  let d3 := 1        -- Second column difference
  let a1 := 25       -- First number in the row
  let a2 := 16 - d1  -- First number in the first column
  let a3 := a1 - d2 * 6  -- Last number in the row
  a3 + d3

theorem value_of_M_correct : value_of_M = -16 :=
  by
    let d1 := 4       -- First column difference
    let d2 := -7      -- Row difference
    let d3 := 1       -- Second column difference
    let a1 := 25      -- First number in the row
    let a2 := 16 - d1 -- First number in the first column
    let a3 := a1 - d2 * 6 -- Last number in the row
    have : a3 + d3 = -16
    · sorry
    exact this

end value_of_M_correct_l1905_190530


namespace more_candidates_selected_l1905_190578

theorem more_candidates_selected (n : ℕ) (pA pB : ℝ) 
  (hA : pA = 0.06) (hB : pB = 0.07) (hN : n = 8200) :
  (pB * n - pA * n) = 82 :=
by
  sorry

end more_candidates_selected_l1905_190578


namespace male_athletes_drawn_l1905_190543

theorem male_athletes_drawn (total_males : ℕ) (total_females : ℕ) (total_sample : ℕ)
  (h_males : total_males = 20) (h_females : total_females = 10) (h_sample : total_sample = 6) :
  (total_sample * total_males) / (total_males + total_females) = 4 := 
  by
  sorry

end male_athletes_drawn_l1905_190543


namespace max_base_angle_is_7_l1905_190533

-- Define the conditions and the problem statement
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def isosceles_triangle (x : ℕ) : Prop :=
  is_prime x ∧ ∃ y : ℕ, 2 * x + y = 180 ∧ is_prime y

theorem max_base_angle_is_7 :
  ∃ (x : ℕ), isosceles_triangle x ∧ x = 7 :=
by
  sorry

end max_base_angle_is_7_l1905_190533


namespace find_y_l1905_190513

theorem find_y (y : ℚ) (h : 6 * y + 3 * y + 4 * y + 2 * y + 1 * y + 5 * y = 360) : y = 120 / 7 := 
sorry

end find_y_l1905_190513


namespace perimeter_regular_polygon_l1905_190570

-- Definitions of the conditions
def side_length : ℕ := 8
def exterior_angle : ℕ := 72
def sum_of_exterior_angles : ℕ := 360

-- Number of sides calculation
def num_sides : ℕ := sum_of_exterior_angles / exterior_angle

-- Perimeter calculation
def perimeter (n : ℕ) (l : ℕ) : ℕ := n * l

-- Theorem statement
theorem perimeter_regular_polygon : perimeter num_sides side_length = 40 :=
by
  sorry

end perimeter_regular_polygon_l1905_190570


namespace max_cables_cut_l1905_190587

theorem max_cables_cut (computers cables clusters : ℕ) (h_computers : computers = 200) (h_cables : cables = 345) (h_clusters : clusters = 8) :
  ∃ k : ℕ, k = cables - (computers - clusters + 1) ∧ k = 153 :=
by
  sorry

end max_cables_cut_l1905_190587
