import Mathlib

namespace find_number_l1033_103315

noncomputable def solve_N (x : ℝ) (N : ℝ) : Prop :=
  ((N / x) / (3.6 * 0.2) = 2)

theorem find_number (x : ℝ) (N : ℝ) (h1 : x = 12) (h2 : solve_N x N) : N = 17.28 :=
  by
  sorry

end find_number_l1033_103315


namespace contestant_wins_probability_l1033_103365

-- Define the basic parameters: number of questions and number of choices
def num_questions : ℕ := 4
def num_choices : ℕ := 3

-- Define the probability of getting a single question right
def prob_right : ℚ := 1 / num_choices

-- Define the probability of guessing all questions right
def prob_all_right : ℚ := prob_right ^ num_questions

-- Define the probability of guessing exactly three questions right (one wrong)
def prob_one_wrong : ℚ := (prob_right ^ 3) * (2 / num_choices)

-- Calculate the total probability of winning
def total_prob_winning : ℚ := prob_all_right + 4 * prob_one_wrong

-- The final statement to prove
theorem contestant_wins_probability :
  total_prob_winning = 1 / 9 := 
sorry

end contestant_wins_probability_l1033_103365


namespace tangent_eq_inequality_not_monotonic_l1033_103368

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.log x) / (x + a)

theorem tangent_eq (a : ℝ) (h : 0 < a) : 
  ∃ k : ℝ, (k, f 1 a) ∈ {
    p : ℝ × ℝ | p.1 - (a + 1) * p.2 - 1 = 0 
  } :=
  sorry

theorem inequality (x : ℝ) (h : 1 ≤ x) : f x 1 ≤ (x - 1) / 2 := 
  sorry

theorem not_monotonic (a : ℝ) (h : 0 < a) : 
  ¬(∀ x y : ℝ, x < y → f x a ≤ f y a ∨ x < y → f x a ≥ f y a) := 
  sorry

end tangent_eq_inequality_not_monotonic_l1033_103368


namespace intersection_complement_eq_l1033_103358

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def P : Finset ℕ := {1, 2, 3, 4}
def Q : Finset ℕ := {3, 4, 5}
def U_complement_Q : Finset ℕ := U \ Q

theorem intersection_complement_eq : P ∩ U_complement_Q = {1, 2} :=
by {
  sorry
}

end intersection_complement_eq_l1033_103358


namespace third_height_of_triangle_l1033_103353

theorem third_height_of_triangle 
  (a b c ha hb hc : ℝ)
  (h_abc_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_heights : ∃ (h1 h2 h3 : ℕ), h1 = 3 ∧ h2 = 10 ∧ h3 ≠ h1 ∧ h3 ≠ h2) :
  ∃ (h3 : ℕ), h3 = 4 :=
by
  sorry

end third_height_of_triangle_l1033_103353


namespace find_prime_between_20_and_35_with_remainder_7_l1033_103318

theorem find_prime_between_20_and_35_with_remainder_7 : 
  ∃ p : ℕ, Nat.Prime p ∧ 20 ≤ p ∧ p ≤ 35 ∧ p % 11 = 7 ∧ p = 29 := 
by 
  sorry

end find_prime_between_20_and_35_with_remainder_7_l1033_103318


namespace largest_x_value_l1033_103345

theorem largest_x_value (x : ℝ) :
  (x ≠ 9) ∧ (x ≠ -4) ∧ ((x ^ 2 - x - 72) / (x - 9) = 5 / (x + 4)) → x = -3 :=
sorry

end largest_x_value_l1033_103345


namespace jake_has_one_more_balloon_than_allan_l1033_103363

def balloons_allan : ℕ := 6
def balloons_jake_initial : ℕ := 3
def balloons_jake_additional : ℕ := 4

theorem jake_has_one_more_balloon_than_allan :
  (balloons_jake_initial + balloons_jake_additional - balloons_allan) = 1 :=
by
  sorry

end jake_has_one_more_balloon_than_allan_l1033_103363


namespace translate_statement_to_inequality_l1033_103348

theorem translate_statement_to_inequality (y : ℝ) : (1/2) * y + 5 > 0 ↔ True := 
sorry

end translate_statement_to_inequality_l1033_103348


namespace sequence_fraction_l1033_103314

-- Definitions for arithmetic and geometric sequences
def isArithmeticSeq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def isGeometricSeq (a b c : ℝ) :=
  b^2 = a * c

-- Given conditions
variables {a : ℕ → ℝ} {d : ℝ}

-- a is an arithmetic sequence with common difference d ≠ 0
axiom h1 : isArithmeticSeq a d
axiom h2 : d ≠ 0

-- a_2, a_3, a_9 form a geometric sequence
axiom h3 : isGeometricSeq (a 2) (a 3) (a 9)

-- Goal: prove the value of the given expression
theorem sequence_fraction {a : ℕ → ℝ} {d : ℝ} (h1 : isArithmeticSeq a d) (h2 : d ≠ 0) (h3 : isGeometricSeq (a 2) (a 3) (a 9)) :
  (a 2 + a 3 + a 4) / (a 4 + a 5 + a 6) = 3 / 8 :=
by
  sorry

end sequence_fraction_l1033_103314


namespace symmetric_points_origin_l1033_103341

theorem symmetric_points_origin (a b : ℝ)
  (h1 : (-2 : ℝ) = -a)
  (h2 : (b : ℝ) = -3) : a - b = 5 :=
by
  sorry

end symmetric_points_origin_l1033_103341


namespace asymptotes_of_hyperbola_l1033_103373

variable {a : ℝ}

/-- Given that the length of the real axis of the hyperbola x^2/a^2 - y^2 = 1 (a > 0) is 1,
    we want to prove that the equation of its asymptotes is y = ± 2x. -/
theorem asymptotes_of_hyperbola (ha : a > 0) (h_len : 2 * a = 1) :
  ∀ x y : ℝ, (y = 2 * x) ∨ (y = -2 * x) :=
by {
  sorry
}

end asymptotes_of_hyperbola_l1033_103373


namespace proof_problem_l1033_103333

variables {a b c d : ℝ} (h1 : a ≠ -2) (h2 : b ≠ -2) (h3 : c ≠ -2) (h4 : d ≠ -2)
variable (ω : ℂ) (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
variable (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω)

theorem proof_problem : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 :=
sorry

end proof_problem_l1033_103333


namespace pyramid_surface_area_l1033_103393

noncomputable def total_surface_area_of_pyramid (a b : ℝ) (theta : ℝ) (height : ℝ) : ℝ :=
  let base_area := a * b * Real.sin theta
  let slant_height := Real.sqrt (height ^ 2 + (a / 2) ^ 2)
  let lateral_area := 4 * (1 / 2 * a * slant_height)
  base_area + lateral_area

theorem pyramid_surface_area :
  total_surface_area_of_pyramid 12 14 (Real.pi / 3) 15 = 168 * Real.sqrt 3 + 216 * Real.sqrt 29 :=
by sorry

end pyramid_surface_area_l1033_103393


namespace simultaneous_equations_solution_l1033_103359

theorem simultaneous_equations_solution (x y : ℚ) :
  3 * x^2 + x * y - 2 * y^2 = -5 ∧ x^2 + 2 * x * y + y^2 = 1 ↔ 
  (x = 3/5 ∧ y = -8/5) ∨ (x = -3/5 ∧ y = 8/5) :=
by
  sorry

end simultaneous_equations_solution_l1033_103359


namespace rate_of_painting_per_sq_m_l1033_103337

def length_of_floor : ℝ := 18.9999683334125
def total_cost : ℝ := 361
def ratio_of_length_to_breadth : ℝ := 3

theorem rate_of_painting_per_sq_m :
  ∃ (rate : ℝ), rate = 3 :=
by
  let B := length_of_floor / ratio_of_length_to_breadth
  let A := length_of_floor * B
  let rate := total_cost / A
  use rate
  sorry  -- Skipping proof as instructed

end rate_of_painting_per_sq_m_l1033_103337


namespace simplify_expr1_simplify_expr2_l1033_103395

theorem simplify_expr1 : (-4)^2023 * (-0.25)^2024 = -0.25 :=
by 
  sorry

theorem simplify_expr2 : 23 * (-4 / 11) + (-5 / 11) * 23 - 23 * (2 / 11) = -23 :=
by 
  sorry

end simplify_expr1_simplify_expr2_l1033_103395


namespace f_is_neither_odd_nor_even_l1033_103335

-- Defining the function f(x)
def f (x : ℝ) : ℝ := x^2 + 6 * x

-- Defining the concept of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

-- Defining the concept of an even function
def is_even (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

-- The goal is to prove that f is neither odd nor even
theorem f_is_neither_odd_nor_even : ¬ is_odd f ∧ ¬ is_even f :=
by
  sorry

end f_is_neither_odd_nor_even_l1033_103335


namespace combination_10_3_l1033_103311

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l1033_103311


namespace necessary_but_not_sufficient_for_gt_zero_l1033_103387

theorem necessary_but_not_sufficient_for_gt_zero (x : ℝ) : 
  x ≠ 0 → (¬ (x ≤ 0)) := by 
  sorry

end necessary_but_not_sufficient_for_gt_zero_l1033_103387


namespace tennis_balls_in_each_container_l1033_103300

theorem tennis_balls_in_each_container :
  let total_balls := 100
  let given_away := total_balls / 2
  let remaining := total_balls - given_away
  let containers := 5
  remaining / containers = 10 := 
by
  sorry

end tennis_balls_in_each_container_l1033_103300


namespace pie_remaining_portion_l1033_103397

theorem pie_remaining_portion (carlos_portion maria_portion remaining_portion : ℝ)
  (h1 : carlos_portion = 0.6) 
  (h2 : remaining_portion = 1 - carlos_portion)
  (h3 : maria_portion = 0.5 * remaining_portion) :
  remaining_portion - maria_portion = 0.2 := 
by
  sorry

end pie_remaining_portion_l1033_103397


namespace total_kids_played_with_l1033_103371

-- Define the conditions as separate constants
def kidsMonday : Nat := 12
def kidsTuesday : Nat := 7

-- Prove the total number of kids Julia played with
theorem total_kids_played_with : kidsMonday + kidsTuesday = 19 := 
by
  sorry

end total_kids_played_with_l1033_103371


namespace symmetric_angles_y_axis_l1033_103339

theorem symmetric_angles_y_axis (α β : ℝ) (k : ℤ)
  (h : ∃ k : ℤ, β = 2 * k * π + (π - α)) :
  α + β = (2 * k + 1) * π ∨ α = -β + (2 * k + 1) * π :=
by sorry

end symmetric_angles_y_axis_l1033_103339


namespace mary_principal_amount_l1033_103331

theorem mary_principal_amount (t1 t2 t3 t4:ℕ) (P R:ℕ) :
  (t1 = 2) →
  (t2 = 260) →
  (t3 = 5) →
  (t4 = 350) →
  (P + 2 * P * R = t2) →
  (P + 5 * P * R = t4) →
  P = 200 :=
by
  intros
  sorry

end mary_principal_amount_l1033_103331


namespace min_value_expression_l1033_103375

theorem min_value_expression (x y z : ℝ) (h1 : -1/2 < x ∧ x < 1/2) (h2 : -1/2 < y ∧ y < 1/2) (h3 : -1/2 < z ∧ z < 1/2) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) + 1 / 2) ≥ 2.5 :=
by {
  sorry
}

end min_value_expression_l1033_103375


namespace process_terminates_with_one_element_in_each_list_final_elements_are_different_l1033_103376

-- Define the initial lists
def List1 := [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96]
def List2 := [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]

-- Predicate to state the termination of the process with exactly one element in each list
theorem process_terminates_with_one_element_in_each_list (List1 List2 : List ℕ):
  ∃ n m, List.length List1 = n ∧ List.length List2 = m ∧ (n = 1 ∧ m = 1) :=
sorry

-- Predicate to state that the final elements in the lists are different
theorem final_elements_are_different (List1 List2 : List ℕ) :
  ∀ a b, a ∈ List1 → b ∈ List2 → (a % 5 = 1 ∧ b % 5 = 4) → a ≠ b :=
sorry

end process_terminates_with_one_element_in_each_list_final_elements_are_different_l1033_103376


namespace number_of_insects_l1033_103394

-- Conditions
def total_legs : ℕ := 30
def legs_per_insect : ℕ := 6

-- Theorem statement
theorem number_of_insects (total_legs legs_per_insect : ℕ) : 
  total_legs / legs_per_insect = 5 := 
by
  sorry

end number_of_insects_l1033_103394


namespace randy_piggy_bank_balance_l1033_103354

def initial_amount : ℕ := 200
def store_trip_cost : ℕ := 2
def trips_per_month : ℕ := 4
def extra_cost_trip : ℕ := 1
def extra_trip_interval : ℕ := 3
def months_in_year : ℕ := 12
def weekly_income : ℕ := 15
def internet_bill_per_month : ℕ := 20
def birthday_gift : ℕ := 100
def weeks_in_year : ℕ := 52

-- To be proved
theorem randy_piggy_bank_balance : 
  initial_amount 
  + (weekly_income * weeks_in_year) 
  + birthday_gift 
  - ((store_trip_cost * trips_per_month * months_in_year)
  + (months_in_year / extra_trip_interval) * extra_cost_trip
  + (internet_bill_per_month * months_in_year))
  = 740 :=
by
  sorry

end randy_piggy_bank_balance_l1033_103354


namespace lcm_5_7_10_14_l1033_103391

theorem lcm_5_7_10_14 : Nat.lcm (Nat.lcm 5 7) (Nat.lcm 10 14) = 70 := by
  sorry

end lcm_5_7_10_14_l1033_103391


namespace base_conversion_zero_l1033_103357

theorem base_conversion_zero (A B : ℕ) (hA : A < 8) (hB : B < 6) (h : 8 * A + B = 6 * B + A) : 8 * A + B = 0 :=
by
  sorry

end base_conversion_zero_l1033_103357


namespace factorize_a_cube_minus_nine_a_l1033_103388

theorem factorize_a_cube_minus_nine_a (a : ℝ) : a^3 - 9 * a = a * (a + 3) * (a - 3) :=
by sorry

end factorize_a_cube_minus_nine_a_l1033_103388


namespace probability_x_lt_y_l1033_103309

noncomputable def rectangle_vertices := [(0, 0), (4, 0), (4, 3), (0, 3)]

theorem probability_x_lt_y :
  let area_triangle := (1 / 2) * 3 * 3
  let area_rectangle := 4 * 3
  let probability := area_triangle / area_rectangle
  probability = (3 / 8) := 
by
  sorry

end probability_x_lt_y_l1033_103309


namespace no_solution_exists_l1033_103327

theorem no_solution_exists (a b : ℤ) : ∃ c : ℤ, ∀ m n : ℤ, m^2 + a * m + b ≠ 2 * n^2 + 2 * n + c :=
by {
  -- Insert correct proof here
  sorry
}

end no_solution_exists_l1033_103327


namespace scientific_notation_correct_l1033_103389

def big_number : ℕ := 274000000

noncomputable def scientific_notation : ℝ := 2.74 * 10^8

theorem scientific_notation_correct : (big_number : ℝ) = scientific_notation :=
by sorry

end scientific_notation_correct_l1033_103389


namespace number_of_solutions_l1033_103374

theorem number_of_solutions :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℕ,
    (x < 10^2006) ∧ ((x * (x - 1)) % 10^2006 = 0) → x ≤ n :=
sorry

end number_of_solutions_l1033_103374


namespace initial_animal_types_l1033_103349

theorem initial_animal_types (x : ℕ) (h1 : 6 * (x + 4) = 54) : x = 5 := 
sorry

end initial_animal_types_l1033_103349


namespace distinguishable_balls_boxes_l1033_103384

theorem distinguishable_balls_boxes : (3^6 = 729) :=
by {
  sorry
}

end distinguishable_balls_boxes_l1033_103384


namespace new_mean_after_adding_constant_l1033_103356

theorem new_mean_after_adding_constant (S : ℝ) (average : ℝ) (n : ℕ) (a : ℝ) :
  n = 15 → average = 40 → a = 15 → S = n * average → (S + n * a) / n = 55 :=
by
  intros hn haverage ha hS
  sorry

end new_mean_after_adding_constant_l1033_103356


namespace sum_of_primes_l1033_103385

theorem sum_of_primes (a b c : ℕ) (h₁ : Nat.Prime a) (h₂ : Nat.Prime b) (h₃ : Nat.Prime c) (h₄ : b + c = 13) (h₅ : c^2 - a^2 = 72) :
  a + b + c = 20 := 
sorry

end sum_of_primes_l1033_103385


namespace cos_double_angle_sin_double_angle_l1033_103383

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (2 * θ) = -1/2 :=
by sorry

theorem sin_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.sin (2 * θ) = (Real.sqrt 3) / 2 :=
by sorry

end cos_double_angle_sin_double_angle_l1033_103383


namespace unique_solution_set_l1033_103369

theorem unique_solution_set :
  {a : ℝ | ∃ x : ℝ, (x+a)/(x^2-1) = 1 ∧ 
                    (∀ y : ℝ, (y+a)/(y^2-1) = 1 → y = x)} 
  = {-1, 1, -5/4} :=
sorry

end unique_solution_set_l1033_103369


namespace find_alpha_l1033_103307

noncomputable def angle_in_interval (α : ℝ) : Prop :=
  370 < α ∧ α < 520 

theorem find_alpha (α : ℝ) (h_cos : Real.cos α = 1 / 2) (h_interval: angle_in_interval α) : α = 420 :=
sorry

end find_alpha_l1033_103307


namespace paul_money_duration_l1033_103325

theorem paul_money_duration (earn1 earn2 spend : ℕ) (h1 : earn1 = 3) (h2 : earn2 = 3) (h_spend : spend = 3) : 
  (earn1 + earn2) / spend = 2 :=
by
  sorry

end paul_money_duration_l1033_103325


namespace Events_B_and_C_mutex_l1033_103336

-- Definitions of events based on scores
def EventA (score : ℕ) := score ≥ 1 ∧ score ≤ 10
def EventB (score : ℕ) := score > 5 ∧ score ≤ 10
def EventC (score : ℕ) := score > 1 ∧ score < 6
def EventD (score : ℕ) := score > 0 ∧ score < 6

-- Mutually exclusive definition:
def mutually_exclusive (P Q : ℕ → Prop) := ∀ (x : ℕ), ¬ (P x ∧ Q x)

-- The proof statement:
theorem Events_B_and_C_mutex : mutually_exclusive EventB EventC :=
by
  sorry

end Events_B_and_C_mutex_l1033_103336


namespace ball_reaches_top_left_pocket_l1033_103340

-- Definitions based on the given problem
def table_width : ℕ := 26
def table_height : ℕ := 1965
def pocket_start : (ℕ × ℕ) := (0, 0)
def pocket_end : (ℕ × ℕ) := (0, table_height)
def angle_of_release : ℝ := 45

-- The goal is to prove that the ball will reach the top left pocket after reflections
theorem ball_reaches_top_left_pocket :
  ∃ reflections : ℕ, (reflections * table_width, reflections * table_height) = pocket_end :=
sorry

end ball_reaches_top_left_pocket_l1033_103340


namespace frequency_of_group_l1033_103379

-- Definitions based on conditions in the problem
def sampleCapacity : ℕ := 32
def frequencyRate : ℝ := 0.25

-- Lean statement representing the proof
theorem frequency_of_group : (frequencyRate * sampleCapacity : ℝ) = 8 := 
by 
  sorry -- Proof placeholder

end frequency_of_group_l1033_103379


namespace leaves_blew_away_l1033_103352

theorem leaves_blew_away (initial_leaves : ℕ) (leaves_left : ℕ) (blew_away : ℕ) 
  (h1 : initial_leaves = 356) (h2 : leaves_left = 112) (h3 : blew_away = initial_leaves - leaves_left) :
  blew_away = 244 :=
by
  sorry

end leaves_blew_away_l1033_103352


namespace orig_polygon_sides_l1033_103313

theorem orig_polygon_sides (n : ℕ) (S : ℕ) :
  (n - 1 > 2) ∧ S = 1620 → (n = 10 ∨ n = 11 ∨ n = 12) :=
by
  sorry

end orig_polygon_sides_l1033_103313


namespace scheme2_saves_money_for_80_participants_l1033_103390

-- Define the variables and conditions
def total_charge_scheme1 (x : ℕ) (hx : x > 50) : ℕ :=
  1500 + 240 * x

def total_charge_scheme2 (x : ℕ) (hx : x > 50) : ℕ :=
  270 * (x - 5)

-- Define the theorem
theorem scheme2_saves_money_for_80_participants :
  total_charge_scheme2 80 (by decide) < total_charge_scheme1 80 (by decide) :=
sorry

end scheme2_saves_money_for_80_participants_l1033_103390


namespace parallelogram_area_15_l1033_103361

def point := (ℝ × ℝ)

def base_length (p1 p2 : point) : ℝ :=
  abs (p2.1 - p1.1)

def height_length (p3 p4 : point) : ℝ :=
  abs (p3.2 - p4.2)

def parallelogram_area (p1 p2 p3 p4 : point) : ℝ :=
  base_length p1 p2 * height_length p1 p3

theorem parallelogram_area_15 :
  parallelogram_area (0, 0) (3, 0) (1, 5) (4, 5) = 15 := by
  sorry

end parallelogram_area_15_l1033_103361


namespace mapping_f_correct_l1033_103308

theorem mapping_f_correct (a1 a2 a3 a4 b1 b2 b3 b4 : ℤ) :
  (∀ (x : ℤ), x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4 = (x + 1)^4 + b1 * (x + 1)^3 + b2 * (x + 1)^2 + b3 * (x + 1) + b4) →
  a1 = 4 → a2 = 3 → a3 = 2 → a4 = 1 →
  b1 = 0 → b1 + b2 + b3 + b4 = 0 →
  (b1, b2, b3, b4) = (0, -3, 4, -1) :=
by
  intros
  sorry

end mapping_f_correct_l1033_103308


namespace smallest_non_lucky_multiple_of_8_correct_l1033_103305

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def smallest_non_lucky_multiple_of_8 := 16

theorem smallest_non_lucky_multiple_of_8_correct :
  smallest_non_lucky_multiple_of_8 = 16 ∧
  is_lucky smallest_non_lucky_multiple_of_8 = false :=
by
  sorry

end smallest_non_lucky_multiple_of_8_correct_l1033_103305


namespace average_snack_sales_per_ticket_l1033_103343

theorem average_snack_sales_per_ticket :
  let cracker_price := (3 : ℝ) * 2.25
  let beverage_price := (4 : ℝ) * 1.50
  let chocolate_price := (4 : ℝ) * 1.00
  let total_sales := cracker_price + beverage_price + chocolate_price
  let movie_tickets := 6
  (total_sales / movie_tickets = 2.79) :=
by
  let cracker_price := (3 : ℝ) * 2.25
  let beverage_price := (4 : ℝ) * 1.50
  let chocolate_price := (4 : ℝ) * 1.00
  let total_sales := cracker_price + beverage_price + chocolate_price
  let movie_tickets := 6
  show total_sales / movie_tickets = 2.79
  sorry

end average_snack_sales_per_ticket_l1033_103343


namespace largest_common_divisor_l1033_103317

theorem largest_common_divisor (d h m s : ℕ) : 
  40 ∣ (1000000 * d + 10000 * h + 100 * m + s - (86400 * d + 3600 * h + 60 * m + s)) :=
by
  sorry

end largest_common_divisor_l1033_103317


namespace sets_produced_and_sold_is_500_l1033_103320

-- Define the initial conditions as constants
def initial_outlay : ℕ := 10000
def manufacturing_cost_per_set : ℕ := 20
def selling_price_per_set : ℕ := 50
def total_profit : ℕ := 5000

-- The proof goal
theorem sets_produced_and_sold_is_500 (x : ℕ) : 
  (total_profit = selling_price_per_set * x - (initial_outlay + manufacturing_cost_per_set * x)) → 
  x = 500 :=
by 
  sorry

end sets_produced_and_sold_is_500_l1033_103320


namespace total_amount_is_33_l1033_103362

variable (n : ℕ) (c t : ℝ)

def total_amount_paid (n : ℕ) (c t : ℝ) : ℝ :=
  let cost_before_tax := n * c
  let tax := t * cost_before_tax
  cost_before_tax + tax

theorem total_amount_is_33
  (h1 : n = 5)
  (h2 : c = 6)
  (h3 : t = 0.10) :
  total_amount_paid n c t = 33 :=
by
  rw [h1, h2, h3]
  sorry

end total_amount_is_33_l1033_103362


namespace probability_no_self_draws_l1033_103381

theorem probability_no_self_draws :
  let total_outcomes := 6
  let favorable_outcomes := 2
  let probability := favorable_outcomes / total_outcomes
  probability = 1 / 3 :=
by
  sorry

end probability_no_self_draws_l1033_103381


namespace compute_expression_l1033_103370

theorem compute_expression : 2 + ((4 * 3 - 2) / 2 * 3) + 5 = 22 :=
by
  -- Place the solution steps if needed
  sorry

end compute_expression_l1033_103370


namespace correct_equation_l1033_103303

theorem correct_equation (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by
  sorry

end correct_equation_l1033_103303


namespace total_cost_sandwiches_sodas_l1033_103310

theorem total_cost_sandwiches_sodas (cost_per_sandwich cost_per_soda : ℝ) 
  (num_sandwiches num_sodas : ℕ) (discount_rate : ℝ) (total_items : ℕ) :
  cost_per_sandwich = 4 → 
  cost_per_soda = 3 → 
  num_sandwiches = 6 → 
  num_sodas = 7 → 
  discount_rate = 0.10 → 
  total_items = num_sandwiches + num_sodas → 
  total_items > 10 → 
  (num_sandwiches * cost_per_sandwich + num_sodas * cost_per_soda) * (1 - discount_rate) = 40.5 :=
by
  intros
  sorry

end total_cost_sandwiches_sodas_l1033_103310


namespace sum_not_divisible_by_three_times_any_number_l1033_103398

theorem sum_not_divisible_by_three_times_any_number (n : ℕ) (a : Fin n → ℕ) (h : n ≥ 3) (distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j) :
  ∃ (i j : Fin n), i ≠ j ∧ (∀ k : Fin n, ¬ (a i + a j) ∣ (3 * a k)) :=
sorry

end sum_not_divisible_by_three_times_any_number_l1033_103398


namespace field_width_l1033_103377

theorem field_width (W L : ℝ) (h1 : L = (7 / 5) * W) (h2 : 2 * L + 2 * W = 288) : W = 60 :=
by
  sorry

end field_width_l1033_103377


namespace age_of_b_l1033_103346

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 32) : b = 12 :=
by sorry

end age_of_b_l1033_103346


namespace find_cos_A_l1033_103386

noncomputable def cos_A_of_third_quadrant : Real :=
-3 / 5

theorem find_cos_A (A : Real) (h1 : A ∈ Set.Icc (π) (3 * π / 2)) 
  (h2 : Real.sin A = 4 / 5) : Real.cos A = -3 / 5 := 
sorry

end find_cos_A_l1033_103386


namespace right_triangle_perimeter_5_shortest_altitude_1_l1033_103355

-- Definition of a right-angled triangle's sides with given perimeter and altitude
def right_angled_triangle (a b c : ℚ) : Prop :=
a^2 + b^2 = c^2 ∧ a + b + c = 5 ∧ a * b = c

-- Statement of the theorem to prove the side lengths of the triangle
theorem right_triangle_perimeter_5_shortest_altitude_1 :
  ∃ (a b c : ℚ), right_angled_triangle a b c ∧ (a = 5 / 3 ∧ b = 5 / 4 ∧ c = 25 / 12) ∨ (a = 5 / 4 ∧ b = 5 / 3 ∧ c = 25 / 12) :=
by
  sorry

end right_triangle_perimeter_5_shortest_altitude_1_l1033_103355


namespace find_x_l1033_103392

noncomputable def vector_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_x (x : ℝ) :
  let a := (1, 2*x + 1)
  let b := (2, 3)
  (vector_parallel a b) → x = 1 / 4 :=
by
  intro h
  have h_eq := h
  sorry  -- proof is not needed as per instruction

end find_x_l1033_103392


namespace solve_equation_l1033_103351

theorem solve_equation (x y : ℝ) (k : ℤ) :
  x^2 - 2 * x * Real.sin (x * y) + 1 = 0 ↔ (x = 1 ∧ y = (Real.pi / 2) + 2 * Real.pi * k) ∨ (x = -1 ∧ y = (Real.pi / 2) + 2 * Real.pi * k) :=
by
  -- Logical content will be filled here, sorry is used because proof steps are not required.
  sorry

end solve_equation_l1033_103351


namespace frequency_of_middle_group_l1033_103328

theorem frequency_of_middle_group
    (num_rectangles : ℕ)
    (middle_area : ℝ)
    (other_areas_sum : ℝ)
    (sample_size : ℕ)
    (total_area_norm : ℝ)
    (h1 : num_rectangles = 11)
    (h2 : middle_area = other_areas_sum)
    (h3 : sample_size = 160)
    (h4 : middle_area + other_areas_sum = total_area_norm)
    (h5 : total_area_norm = 1):
    160 * (middle_area / total_area_norm) = 80 :=
by
  sorry

end frequency_of_middle_group_l1033_103328


namespace part1_inequality_part2_inequality_l1033_103304

-- Problem Part 1
def f (x : ℝ) : ℝ := abs (x - 2) - abs (x + 1)

theorem part1_inequality (x : ℝ) : f x ≤ 1 ↔ 0 ≤ x :=
by sorry

-- Problem Part 2
def max_f_value : ℝ := 3
def a : ℝ := sorry  -- Define in context
def b : ℝ := sorry  -- Define in context
def c : ℝ := sorry  -- Define in context

-- Prove √a + √b + √c ≤ 3 given a + b + c = 3
theorem part2_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = max_f_value) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 3 :=
by sorry

end part1_inequality_part2_inequality_l1033_103304


namespace original_garden_length_l1033_103382

theorem original_garden_length (x : ℝ) (area : ℝ) (reduced_length : ℝ) (width : ℝ) (length_condition : x - reduced_length = width) (area_condition : x * width = area) (given_area : area = 120) (given_reduced_length : reduced_length = 2) : x = 12 := 
by
  sorry

end original_garden_length_l1033_103382


namespace geometric_sequence_solution_l1033_103372

theorem geometric_sequence_solution:
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (q a1 : ℝ),
    a 2 = 6 → 6 * a1 + a 3 = 30 → q > 2 →
    (∀ n, a n = 2 * 3 ^ (n - 1)) ∧
    (∀ n, S n = (3 ^ n - 1) / 2) :=
by
  intros a S q a1 h1 h2 h3
  sorry

end geometric_sequence_solution_l1033_103372


namespace number_of_pieces_l1033_103332

def pan_length : ℕ := 24
def pan_width : ℕ := 30
def brownie_length : ℕ := 3
def brownie_width : ℕ := 4

def area (length : ℕ) (width : ℕ) : ℕ := length * width

theorem number_of_pieces :
  (area pan_length pan_width) / (area brownie_length brownie_width) = 60 := by
  sorry

end number_of_pieces_l1033_103332


namespace boat_distance_against_stream_in_one_hour_l1033_103323

-- Define the conditions
def speed_in_still_water : ℝ := 4 -- speed of the boat in still water (km/hr)
def downstream_distance_in_one_hour : ℝ := 6 -- distance traveled along the stream in one hour (km)

-- Define the function to compute the speed of the stream
def speed_of_stream (downstream_distance : ℝ) (boat_speed_still_water : ℝ) : ℝ :=
  downstream_distance - boat_speed_still_water

-- Define the effective speed against the stream
def effective_speed_against_stream (boat_speed_still_water : ℝ) (stream_speed : ℝ) : ℝ :=
  boat_speed_still_water - stream_speed

-- Prove that the boat travels 2 km against the stream in one hour given the conditions
theorem boat_distance_against_stream_in_one_hour :
  effective_speed_against_stream speed_in_still_water (speed_of_stream downstream_distance_in_one_hour speed_in_still_water) * 1 = 2 := 
by
  sorry

end boat_distance_against_stream_in_one_hour_l1033_103323


namespace balls_in_boxes_l1033_103319

theorem balls_in_boxes :
  ∃ (f : Fin 5 → Fin 3), 
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ b : Fin 3, ∃ i, f i = b) ∧
    f 0 ≠ f 1 :=
  sorry

end balls_in_boxes_l1033_103319


namespace find_x_l1033_103301

variable (A B x : ℝ)
variable (h1 : A > 0) (h2 : B > 0)
variable (h3 : A = (x / 100) * B)

theorem find_x : x = 100 * (A / B) :=
by
  sorry

end find_x_l1033_103301


namespace power_of_3_l1033_103330

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l1033_103330


namespace remainder_problem_l1033_103360

theorem remainder_problem {x y z : ℤ} (h1 : x % 102 = 56) (h2 : y % 154 = 79) (h3 : z % 297 = 183) :
  x % 19 = 18 ∧ y % 22 = 13 ∧ z % 33 = 18 :=
by
  sorry

end remainder_problem_l1033_103360


namespace B_months_grazing_eq_five_l1033_103367

-- Define the conditions in the problem
def A_oxen : ℕ := 10
def A_months : ℕ := 7
def B_oxen : ℕ := 12
def C_oxen : ℕ := 15
def C_months : ℕ := 3
def total_rent : ℝ := 175
def C_rent_share : ℝ := 45

-- Total ox-units function
def total_ox_units (x : ℕ) : ℕ :=
  A_oxen * A_months + B_oxen * x + C_oxen * C_months

-- Prove that the number of months B's oxen grazed is 5
theorem B_months_grazing_eq_five (x : ℕ) :
  total_ox_units x = 70 + 12 * x + 45 →
  (C_rent_share / total_rent = 45 / total_ox_units x) →
  x = 5 :=
by
  intros h1 h2
  sorry

end B_months_grazing_eq_five_l1033_103367


namespace largest_whole_number_less_than_100_l1033_103334

theorem largest_whole_number_less_than_100 (x : ℕ) (h1 : 7 * x < 100) (h_max : ∀ y : ℕ, 7 * y < 100 → y ≤ x) :
  x = 14 := 
sorry

end largest_whole_number_less_than_100_l1033_103334


namespace find_x_of_equation_l1033_103329

theorem find_x_of_equation (x : ℝ) (hx : x ≠ 0) : (7 * x)^4 = (14 * x)^3 → x = 8 / 7 :=
by
  intro h
  sorry

end find_x_of_equation_l1033_103329


namespace max_daily_sales_l1033_103321

def f (t : ℕ) : ℝ := -2 * t + 200
def g (t : ℕ) : ℝ :=
  if t ≤ 30 then 12 * t + 30
  else 45

def S (t : ℕ) : ℝ := f t * g t

theorem max_daily_sales : ∃ t, 1 ≤ t ∧ t ≤ 50 ∧ S t = 54600 := 
  sorry

end max_daily_sales_l1033_103321


namespace candy_bar_cost_l1033_103364

def cost_soft_drink : ℕ := 2
def num_candy_bars : ℕ := 5
def total_spent : ℕ := 27
def cost_per_candy_bar (C : ℕ) : Prop := cost_soft_drink + num_candy_bars * C = total_spent

-- The theorem we want to prove
theorem candy_bar_cost (C : ℕ) (h : cost_per_candy_bar C) : C = 5 :=
by sorry

end candy_bar_cost_l1033_103364


namespace batches_of_muffins_l1033_103312

-- Definitions of the costs and savings
def cost_blueberries_6oz : ℝ := 5
def cost_raspberries_12oz : ℝ := 3
def ounces_per_batch : ℝ := 12
def total_savings : ℝ := 22

-- The proof problem is to show the number of batches Bill plans to make
theorem batches_of_muffins : (total_savings / (2 * cost_blueberries_6oz - cost_raspberries_12oz)) = 3 := 
by 
  sorry  -- Proof goes here

end batches_of_muffins_l1033_103312


namespace smallest_three_digit_divisible_by_3_and_6_l1033_103324

theorem smallest_three_digit_divisible_by_3_and_6 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999 ∧ n % 3 = 0 ∧ n % 6 = 0) ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 3 = 0 ∧ m % 6 = 0 → n ≤ m) ∧ n = 102 := 
by {sorry}

end smallest_three_digit_divisible_by_3_and_6_l1033_103324


namespace suresh_work_hours_l1033_103380

theorem suresh_work_hours (x : ℝ) (h : x / 15 + 8 / 20 = 1) : x = 9 :=
by 
    sorry

end suresh_work_hours_l1033_103380


namespace B_investment_amount_l1033_103347

-- Definitions based on given conditions
variable (A_investment : ℕ := 300) -- A's investment in dollars
variable (B_investment : ℕ)        -- B's investment in dollars
variable (A_time : ℕ := 12)        -- Time A's investment was in the business in months
variable (B_time : ℕ := 6)         -- Time B's investment was in the business in months
variable (profit : ℕ := 100)       -- Total profit in dollars
variable (A_share : ℕ := 75)       -- A's share of the profit in dollars

-- The mathematically equivalent proof problem to prove that B invested $200
theorem B_investment_amount (h : A_share * (A_investment * A_time + B_investment * B_time) / profit = A_investment * A_time) : 
  B_investment = 200 := by
  sorry

end B_investment_amount_l1033_103347


namespace smoothie_combinations_l1033_103338

theorem smoothie_combinations :
  let flavors := 5
  let supplements := 8
  (flavors * Nat.choose supplements 3) = 280 :=
by
  sorry

end smoothie_combinations_l1033_103338


namespace right_triangle_area_l1033_103322

theorem right_triangle_area (a b c p S : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : a^2 + b^2 = c^2)
  (h4 : p = (a + b + c) / 2) (h5 : S = a * b / 2) :
  p * (p - c) = S ∧ (p - a) * (p - b) = S :=
sorry

end right_triangle_area_l1033_103322


namespace number_of_girls_l1033_103342

theorem number_of_girls
  (B G : ℕ)
  (h1 : B = (8 * G) / 5)
  (h2 : B + G = 351) :
  G = 135 :=
sorry

end number_of_girls_l1033_103342


namespace quadrilateral_smallest_angle_l1033_103326

theorem quadrilateral_smallest_angle
  (a d : ℝ)
  (h1 : a + (a + 2 * d) = 160)
  (h2 : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 360) :
  a = 60 :=
by
  sorry

end quadrilateral_smallest_angle_l1033_103326


namespace inequality_holds_l1033_103344

theorem inequality_holds (a b : ℝ) (h : a ≠ b) : a^4 + 6 * a^2 * b^2 + b^4 > 4 * a * b * (a^2 + b^2) := 
by
  sorry

end inequality_holds_l1033_103344


namespace children_sit_in_same_row_twice_l1033_103396

theorem children_sit_in_same_row_twice
  (rows : ℕ) (seats_per_row : ℕ) (children : ℕ)
  (h_rows : rows = 7) (h_seats_per_row : seats_per_row = 10) (h_children : children = 50) :
  ∃ (morning_evening_pair : ℕ × ℕ), 
  (morning_evening_pair.1 < rows ∧ morning_evening_pair.2 < rows) ∧ 
  morning_evening_pair.1 = morning_evening_pair.2 :=
by
  sorry

end children_sit_in_same_row_twice_l1033_103396


namespace find_common_difference_l1033_103316

def arithmetic_sequence (S_n : ℕ → ℝ) (d : ℝ) :=
  ∀ n, S_n n = (n / 2) * (2 * (S_n 1 / 1) + (n - 1) * d)

theorem find_common_difference (S_n : ℕ → ℝ) (d : ℝ) (h : ∀n, S_n n = (n / 2) * (2 * (S_n 1 / 1) + (n - 1) * d)) 
    (h_condition : S_n 3 / 3 - S_n 2 / 2 = 1) :
  d = 2 :=
sorry

end find_common_difference_l1033_103316


namespace entertainment_team_count_l1033_103366

theorem entertainment_team_count 
  (total_members : ℕ)
  (singers : ℕ) 
  (dancers : ℕ) 
  (prob_both_sing_dance_gt_0 : ℚ)
  (sing_count : singers = 2)
  (dance_count : dancers = 5)
  (prob_condition : prob_both_sing_dance_gt_0 = 7/10) :
  total_members = 5 := 
by 
  sorry

end entertainment_team_count_l1033_103366


namespace mirasol_balance_l1033_103350

/-- Given Mirasol initially had $50, spends $10 on coffee beans, and $30 on a tumbler,
    prove that the remaining balance in her account is $10. -/
theorem mirasol_balance (initial_balance spent_coffee spent_tumbler remaining_balance : ℕ)
  (h1 : initial_balance = 50)
  (h2 : spent_coffee = 10)
  (h3 : spent_tumbler = 30)
  (h4 : remaining_balance = initial_balance - (spent_coffee + spent_tumbler)) :
  remaining_balance = 10 :=
sorry

end mirasol_balance_l1033_103350


namespace number_of_valid_integers_l1033_103378

def count_valid_numbers : Nat :=
  let one_digit_count : Nat := 6
  let two_digit_count : Nat := 6 * 6
  let three_digit_count : Nat := 6 * 6 * 6
  one_digit_count + two_digit_count + three_digit_count

theorem number_of_valid_integers :
  count_valid_numbers = 258 :=
sorry

end number_of_valid_integers_l1033_103378


namespace gray_region_area_l1033_103302

theorem gray_region_area (r R : ℝ) (hR : R = 3 * r) (h_diff : R - r = 3) :
  π * (R^2 - r^2) = 18 * π :=
by
  -- The proof goes here
  sorry

end gray_region_area_l1033_103302


namespace cricket_target_run_rate_cricket_wicket_partnership_score_l1033_103399

noncomputable def remaining_runs_needed (initial_runs : ℕ) (target_runs : ℕ) : ℕ :=
  target_runs - initial_runs

noncomputable def required_run_rate (remaining_runs : ℕ) (remaining_overs : ℕ) : ℚ :=
  (remaining_runs : ℚ) / remaining_overs

theorem cricket_target_run_rate (initial_runs : ℕ) (target_runs : ℕ) (remaining_overs : ℕ)
  (initial_wickets : ℕ) :
  initial_runs = 32 → target_runs = 282 → remaining_overs = 40 → initial_wickets = 3 →
  required_run_rate (remaining_runs_needed initial_runs target_runs) remaining_overs = 6.25 :=
by
  sorry


theorem cricket_wicket_partnership_score (initial_runs : ℕ) (target_runs : ℕ)
  (initial_wickets : ℕ) :
  initial_runs = 32 → target_runs = 282 → initial_wickets = 3 →
  remaining_runs_needed initial_runs target_runs = 250 :=
by
  sorry

end cricket_target_run_rate_cricket_wicket_partnership_score_l1033_103399


namespace algebraic_expression_value_l1033_103306

theorem algebraic_expression_value (m n : ℤ) (h : n - m = 2):
  (m^2 - n^2) / m * (2 * m / (m + n)) = -4 :=
sorry

end algebraic_expression_value_l1033_103306
