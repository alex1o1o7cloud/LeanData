import Mathlib

namespace find_angle_A_find_tan_C_l2084_208439

-- Import necessary trigonometric identities and basic Lean setup
open Real

-- First statement: Given the dot product condition, find angle A
theorem find_angle_A (A : ℝ) (h1 : cos A + sqrt 3 * sin A = 1) :
  A = 2 * π / 3 := 
sorry

-- Second statement: Given the trigonometric condition, find tan C
theorem find_tan_C (B C : ℝ)
  (h1 : 1 + sin (2 * B) = 2 * (cos B ^ 2 - sin B ^ 2))
  (h2 : B + C = π) :
  tan C = (5 * sqrt 3 - 6) / 3 := 
sorry

end find_angle_A_find_tan_C_l2084_208439


namespace proof_problem_l2084_208423

def sequence : Nat → Rat
| 0 => 2000000
| (n + 1) => sequence n / 2

theorem proof_problem :
  (∀ n, ((sequence n).den = 1) → n < 7) ∧ 
  (sequence 7 = 15625) ∧ 
  (sequence 7 - 3 = 15622) :=
by
  sorry

end proof_problem_l2084_208423


namespace runners_meet_l2084_208438

theorem runners_meet (T : ℕ) 
  (h1 : T > 4) 
  (h2 : Nat.lcm 2 (Nat.lcm 4 T) = 44) : 
  T = 11 := 
sorry

end runners_meet_l2084_208438


namespace third_smallest_four_digit_in_pascals_triangle_l2084_208418

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l2084_208418


namespace intersection_point_l2084_208496

theorem intersection_point (x y : ℚ) 
  (h1 : 3 * y = -2 * x + 6) 
  (h2 : 2 * y = 7 * x - 4) :
  x = 24 / 25 ∧ y = 34 / 25 :=
sorry

end intersection_point_l2084_208496


namespace fraction_of_B_l2084_208421

theorem fraction_of_B (A B C : ℝ) 
  (h1 : A = (1/3) * (B + C)) 
  (h2 : A = B + 20) 
  (h3 : A + B + C = 720) : 
  B / (A + C) = 2 / 7 :=
  by 
  sorry

end fraction_of_B_l2084_208421


namespace tom_steps_l2084_208441

theorem tom_steps (matt_rate : ℕ) (tom_extra_rate : ℕ) (matt_steps : ℕ) (tom_rate : ℕ := matt_rate + tom_extra_rate) (time : ℕ := matt_steps / matt_rate)
(H_matt_rate : matt_rate = 20)
(H_tom_extra_rate : tom_extra_rate = 5)
(H_matt_steps : matt_steps = 220) :
  tom_rate * time = 275 :=
by
  -- We start the proof here, but leave it as sorry.
  sorry

end tom_steps_l2084_208441


namespace proposition_C_is_correct_l2084_208492

theorem proposition_C_is_correct :
  ∃ a b : ℝ, (a > 2 ∧ b > 2) → (a * b > 4) :=
by
  sorry

end proposition_C_is_correct_l2084_208492


namespace range_of_x_minus_cos_y_l2084_208425

theorem range_of_x_minus_cos_y
  (x y : ℝ)
  (h : x^2 + 2 * Real.cos y = 1) :
  ∃ (A : Set ℝ), A = {z | -1 ≤ z ∧ z ≤ 1 + Real.sqrt 3} ∧ x - Real.cos y ∈ A :=
by
  sorry

end range_of_x_minus_cos_y_l2084_208425


namespace distance_traveled_by_center_of_ball_l2084_208400

noncomputable def ball_diameter : ℝ := 6
noncomputable def ball_radius : ℝ := ball_diameter / 2
noncomputable def R1 : ℝ := 100
noncomputable def R2 : ℝ := 60
noncomputable def R3 : ℝ := 80
noncomputable def R4 : ℝ := 40

noncomputable def effective_radius_inner (R : ℝ) (r : ℝ) : ℝ := R - r
noncomputable def effective_radius_outer (R : ℝ) (r : ℝ) : ℝ := R + r

noncomputable def dist_travel_on_arc (R : ℝ) : ℝ := R * Real.pi

theorem distance_traveled_by_center_of_ball :
  dist_travel_on_arc (effective_radius_inner R1 ball_radius) +
  dist_travel_on_arc (effective_radius_outer R2 ball_radius) +
  dist_travel_on_arc (effective_radius_inner R3 ball_radius) +
  dist_travel_on_arc (effective_radius_outer R4 ball_radius) = 280 * Real.pi :=
by 
  -- Calculation steps can be filled in here but let's skip
  sorry

end distance_traveled_by_center_of_ball_l2084_208400


namespace inequality_solution_range_of_a_l2084_208465

noncomputable def f (x : ℝ) : ℝ := |1 - 2 * x| - |1 + x| 

theorem inequality_solution (x : ℝ) : f x ≥ 4 ↔ x ≤ -2 ∨ x ≥ 6 := 
by sorry

theorem range_of_a (a x : ℝ) (h : a^2 + 2 * a + |1 + x| < f x) : -3 < a ∧ a < 1 :=
by sorry

end inequality_solution_range_of_a_l2084_208465


namespace ratio_Ford_to_Toyota_l2084_208484

-- Definitions based on the conditions
variables (Ford Dodge Toyota VW : ℕ)

axiom h1 : Ford = (1/3 : ℚ) * Dodge
axiom h2 : VW = (1/2 : ℚ) * Toyota
axiom h3 : VW = 5
axiom h4 : Dodge = 60

-- Theorem statement to be proven
theorem ratio_Ford_to_Toyota : Ford / Toyota = 2 :=
by {
  sorry
}

end ratio_Ford_to_Toyota_l2084_208484


namespace sign_of_b_l2084_208447

variable (a b : ℝ)

theorem sign_of_b (h1 : (a + b > 0 ∨ a - b > 0) ∧ (a + b < 0 ∨ a - b < 0)) 
                  (h2 : (ab > 0 ∨ a / b > 0) ∧ (ab < 0 ∨ a / b < 0))
                  (h3 : (ab > 0 → a > 0 ∧ b > 0) ∨ (ab < 0 → (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0))) :
  b < 0 :=
sorry

end sign_of_b_l2084_208447


namespace scientific_notation_of_distance_l2084_208427

theorem scientific_notation_of_distance :
  ∃ (n : ℝ), n = 384000 ∧ 384000 = n * 10^5 :=
sorry

end scientific_notation_of_distance_l2084_208427


namespace positive_difference_l2084_208440

noncomputable def calculate_diff : ℕ :=
  let first_term := (8^2 - 8^2) / 8
  let second_term := (8^2 * 8^2) / 8
  second_term - first_term

theorem positive_difference : calculate_diff = 512 := by
  sorry

end positive_difference_l2084_208440


namespace positive_number_square_roots_l2084_208471

theorem positive_number_square_roots (m : ℝ) 
  (h : (2 * m - 1) + (2 - m) = 0) :
  (2 - m)^2 = 9 :=
by
  sorry

end positive_number_square_roots_l2084_208471


namespace largest_value_among_expressions_l2084_208482

theorem largest_value_among_expressions 
  (a1 a2 b1 b2 : ℝ)
  (h1 : 0 < a1) (h2 : a1 < a2) (h3 : a2 < 1)
  (h4 : 0 < b1) (h5 : b1 < b2) (h6 : b2 < 1)
  (ha : a1 + a2 = 1) (hb : b1 + b2 = 1) :
  a1 * b1 + a2 * b2 > a1 * a2 + b1 * b2 ∧ 
  a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 := 
sorry

end largest_value_among_expressions_l2084_208482


namespace max_value_of_expression_l2084_208416

theorem max_value_of_expression 
  (x y : ℝ) 
  (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  x^2 + y^2 + 2 * x ≤ 15 := sorry

end max_value_of_expression_l2084_208416


namespace sequence_solution_l2084_208433

theorem sequence_solution (a : ℕ → ℝ) (h1 : a 1 = 1/2)
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 1 / (n^2 + n)) : ∀ n : ℕ, n ≥ 1 → a n = 3/2 - 1/n :=
by
  intros n hn
  sorry

end sequence_solution_l2084_208433


namespace atleast_one_genuine_l2084_208452

noncomputable def products : ℕ := 12
noncomputable def genuine : ℕ := 10
noncomputable def defective : ℕ := 2
noncomputable def selected : ℕ := 3

theorem atleast_one_genuine :
  (selected = 3) →
  (genuine + defective = 12) →
  (genuine ≥ 3) →
  (selected ≥ 1) →
  ∃ g d : ℕ, g + d = 3 ∧ g > 0 ∧ d ≤ 2 :=
by
  -- Proof will go here.
  sorry

end atleast_one_genuine_l2084_208452


namespace students_in_classroom_l2084_208406

theorem students_in_classroom (n : ℕ) :
  n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3 → n = 21 ∨ n = 45 :=
by
  sorry

end students_in_classroom_l2084_208406


namespace chickens_bought_l2084_208456

theorem chickens_bought (total_spent : ℤ) (egg_count : ℤ) (egg_price : ℤ) (chicken_price : ℤ) (egg_cost : ℤ := egg_count * egg_price) (chicken_spent : ℤ := total_spent - egg_cost) : total_spent = 88 → egg_count = 20 → egg_price = 2 → chicken_price = 8 → chicken_spent / chicken_price = 6 :=
by
  intros
  sorry

end chickens_bought_l2084_208456


namespace max_members_choir_l2084_208413

variable (m k n : ℕ)

theorem max_members_choir :
  (∃ k, m = k^2 + 6) ∧ (∃ n, m = n * (n + 6)) → m = 294 :=
by
  sorry

end max_members_choir_l2084_208413


namespace circle_area_radius_8_l2084_208488

variable (r : ℝ) (π : ℝ)

theorem circle_area_radius_8 : r = 8 → (π * r^2) = 64 * π :=
by
  sorry

end circle_area_radius_8_l2084_208488


namespace brick_length_is_50_l2084_208477

theorem brick_length_is_50
  (x : ℝ)
  (brick_volume_eq : x * 11.25 * 6 * 3200 = 800 * 600 * 22.5) :
  x = 50 :=
by
  sorry

end brick_length_is_50_l2084_208477


namespace coffee_cost_per_week_l2084_208411

theorem coffee_cost_per_week 
  (number_people : ℕ) 
  (cups_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (total_cost_per_week : ℝ) 
  (h₁ : number_people = 4)
  (h₂ : cups_per_person_per_day = 2)
  (h₃ : ounces_per_cup = 0.5)
  (h₄ : cost_per_ounce = 1.25)
  (h₅ : total_cost_per_week = 35) : 
  number_people * cups_per_person_per_day * ounces_per_cup * cost_per_ounce * 7 = total_cost_per_week :=
by
  sorry

end coffee_cost_per_week_l2084_208411


namespace remainder_T2015_mod_12_eq_8_l2084_208445

-- Define sequences of length n consisting of the letters A and B,
-- with no more than two A's in a row and no more than two B's in a row
def T : ℕ → ℕ :=
  sorry  -- Definition for T(n) must follow the given rules

-- Theorem to prove that T(2015) modulo 12 equals 8
theorem remainder_T2015_mod_12_eq_8 :
  (T 2015) % 12 = 8 :=
  sorry

end remainder_T2015_mod_12_eq_8_l2084_208445


namespace gcd_32_48_l2084_208443

/--
The greatest common factor of 32 and 48 is 16.
-/
theorem gcd_32_48 : Int.gcd 32 48 = 16 :=
by
  sorry

end gcd_32_48_l2084_208443


namespace product_of_four_integers_l2084_208415

theorem product_of_four_integers 
  (w x y z : ℕ) 
  (h1 : x * y * z = 280)
  (h2 : w * y * z = 168)
  (h3 : w * x * z = 105)
  (h4 : w * x * y = 120) :
  w * x * y * z = 840 :=
by {
sorry
}

end product_of_four_integers_l2084_208415


namespace percentage_waiting_for_parts_l2084_208472

def totalComputers : ℕ := 20
def unfixableComputers : ℕ := (20 * 20) / 100
def fixedRightAway : ℕ := 8
def waitingForParts : ℕ := totalComputers - (unfixableComputers + fixedRightAway)

theorem percentage_waiting_for_parts : (waitingForParts : ℝ) / totalComputers * 100 = 40 := 
by 
  have : waitingForParts = 8 := sorry
  have : (8 / 20 : ℝ) * 100 = 40 := sorry
  exact sorry

end percentage_waiting_for_parts_l2084_208472


namespace exists_pos_int_such_sqrt_not_int_l2084_208454

theorem exists_pos_int_such_sqrt_not_int (a b c : ℤ) : ∃ n : ℕ, 0 < n ∧ ¬∃ k : ℤ, k * k = n^3 + a * n^2 + b * n + c :=
by
  sorry

end exists_pos_int_such_sqrt_not_int_l2084_208454


namespace value_of_expression_l2084_208431

theorem value_of_expression (a b c : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 3) :
  (2 * a - (3 * b - 4 * c)) - ((2 * a - 3 * b) - 4 * c) = 24 := by
  sorry

end value_of_expression_l2084_208431


namespace find_n_l2084_208467

noncomputable def parabola_focus : ℝ × ℝ :=
  (2, 0)

noncomputable def hyperbola_focus (n : ℝ) : ℝ × ℝ :=
  (Real.sqrt (3 + n), 0)

theorem find_n (n : ℝ) : hyperbola_focus n = parabola_focus → n = 1 :=
by
  sorry

end find_n_l2084_208467


namespace initial_interest_rate_l2084_208497

theorem initial_interest_rate
    (P R : ℝ) 
    (h1 : P * R = 10120) 
    (h2 : P * (R + 6) = 12144) : 
    R = 30 :=
sorry

end initial_interest_rate_l2084_208497


namespace percent_students_prefer_golf_l2084_208462

theorem percent_students_prefer_golf (students_north : ℕ) (students_south : ℕ)
  (percent_golf_north : ℚ) (percent_golf_south : ℚ) :
  students_north = 1800 →
  students_south = 2200 →
  percent_golf_north = 15 →
  percent_golf_south = 25 →
  (820 / 4000 : ℚ) = 20.5 :=
by
  intros h_north h_south h_percent_north h_percent_south
  sorry

end percent_students_prefer_golf_l2084_208462


namespace max_a_satisfies_no_lattice_points_l2084_208473

-- Define the conditions
def no_lattice_points (m : ℚ) (x_upper : ℕ) :=
  ∀ x : ℕ, 0 < x ∧ x ≤ x_upper → ¬∃ y : ℤ, y = m * x + 3

-- Final statement we need to prove
theorem max_a_satisfies_no_lattice_points :
  ∃ a : ℚ, a = 51 / 151 ∧ ∀ m : ℚ, 1 / 3 < m → m < a → no_lattice_points m 150 :=
sorry

end max_a_satisfies_no_lattice_points_l2084_208473


namespace calculate_10_odot_5_l2084_208405

def odot (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

theorem calculate_10_odot_5 : odot 10 5 = 38 / 3 := by
  sorry

end calculate_10_odot_5_l2084_208405


namespace max_value_of_f_l2084_208476

noncomputable def f (x : ℝ) : ℝ := 10 * x - 2 * x^2

theorem max_value_of_f : ∃ x : ℝ, f x = 12.5 :=
by
  sorry

end max_value_of_f_l2084_208476


namespace parabola_shift_l2084_208429

theorem parabola_shift (x : ℝ) : 
  let y := -2 * x^2 
  let y1 := -2 * (x + 1)^2 
  let y2 := y1 - 3 
  y2 = -2 * x^2 - 4 * x - 5 := 
by 
  sorry

end parabola_shift_l2084_208429


namespace future_tech_high_absentee_percentage_l2084_208446

theorem future_tech_high_absentee_percentage :
  let total_students := 180
  let boys := 100
  let girls := 80
  let absent_boys_fraction := 1 / 5
  let absent_girls_fraction := 1 / 4
  let absent_boys := absent_boys_fraction * boys
  let absent_girls := absent_girls_fraction * girls
  let total_absent_students := absent_boys + absent_girls
  let absent_percentage := (total_absent_students / total_students) * 100
  (absent_percentage = 22.22) := 
by
  sorry

end future_tech_high_absentee_percentage_l2084_208446


namespace possible_values_of_quadratic_l2084_208407

theorem possible_values_of_quadratic (x : ℝ) (hx : x^2 - 7 * x + 12 < 0) :
  1.75 ≤ x^2 - 7 * x + 14 ∧ x^2 - 7 * x + 14 ≤ 2 := by
  sorry

end possible_values_of_quadratic_l2084_208407


namespace initial_bottles_l2084_208495

-- Define the conditions
def drank_bottles : ℕ := 144
def left_bottles : ℕ := 157

-- Define the total_bottles function
def total_bottles : ℕ := drank_bottles + left_bottles

-- State the theorem to be proven
theorem initial_bottles : total_bottles = 301 :=
by
  sorry

end initial_bottles_l2084_208495


namespace marbles_problem_l2084_208469

theorem marbles_problem (p : ℕ) (m n r : ℕ) 
(hp : Nat.Prime p) 
(h1 : p = 2017)
(h2 : N = p^m * n)
(h3 : ¬ p ∣ n)
(h4 : r = n % p) 
(h N : ∀ (N : ℕ), N = 3 * p * 632 - 1)
: p * m + r = 3913 := 
sorry

end marbles_problem_l2084_208469


namespace double_angle_second_quadrant_l2084_208468

theorem double_angle_second_quadrant (α : ℝ) (h : π/2 < α ∧ α < π) : 
  ¬((0 ≤ 2*α ∧ 2*α < π/2) ∨ (3*π/2 < 2*α ∧ 2*α < 2*π)) :=
sorry

end double_angle_second_quadrant_l2084_208468


namespace fewest_colored_paper_l2084_208460
   
   /-- Jungkook, Hoseok, and Seokjin shared colored paper. 
       Jungkook took 10 cards, Hoseok took 7, and Seokjin took 2 less than Jungkook. 
       Prove that Hoseok took the fewest pieces of colored paper. -/
   theorem fewest_colored_paper 
       (Jungkook Hoseok Seokjin : ℕ)
       (hj : Jungkook = 10)
       (hh : Hoseok = 7)
       (hs : Seokjin = Jungkook - 2) :
       Hoseok < Jungkook ∧ Hoseok < Seokjin :=
   by
     sorry
   
end fewest_colored_paper_l2084_208460


namespace max_sin_sin2x_l2084_208480

open Real

theorem max_sin_sin2x (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  ∃ x : ℝ, (0 < x ∧ x < π / 2) ∧ (sin x * sin (2 * x) = 4 * sqrt 3 / 9) := 
sorry

end max_sin_sin2x_l2084_208480


namespace eval_expression_l2084_208410

theorem eval_expression : (2 ^ (-1 : ℤ)) + (Real.sin (Real.pi / 6)) - (Real.pi - 3.14) ^ (0 : ℤ) + abs (-3) - Real.sqrt 9 = 0 := by
  sorry

end eval_expression_l2084_208410


namespace rectangle_original_area_l2084_208466

theorem rectangle_original_area (L L' A : ℝ) 
  (h1: A = L * 10)
  (h2: L' * 10 = (4 / 3) * A)
  (h3: 2 * L' + 2 * 10 = 60) : A = 150 :=
by 
  sorry

end rectangle_original_area_l2084_208466


namespace real_roots_of_quadratic_l2084_208464

theorem real_roots_of_quadratic (k : ℝ) : (k ≤ 0 ∨ 1 ≤ k) →
  ∃ x : ℝ, x^2 + 2 * k * x + k = 0 :=
by
  intro h
  sorry

end real_roots_of_quadratic_l2084_208464


namespace measure_of_angle_B_l2084_208479

noncomputable def angle_opposite_side (a b c : ℝ) (h : (c^2)/(a+b) + (a^2)/(b+c) = b) : ℝ :=
  if h : (c^2)/(a+b) + (a^2)/(b+c) = b then 60 else 0

theorem measure_of_angle_B {a b c : ℝ} (h : (c^2)/(a+b) + (a^2)/(b+c) = b) : 
  angle_opposite_side a b c h = 60 :=
by
  sorry

end measure_of_angle_B_l2084_208479


namespace thre_digit_num_condition_l2084_208491

theorem thre_digit_num_condition (n : ℕ) (h : n = 735) :
  (n % 35 = 0) ∧ (Nat.digits 10 n).sum = 15 := by
  sorry

end thre_digit_num_condition_l2084_208491


namespace four_people_complete_task_in_18_days_l2084_208424

theorem four_people_complete_task_in_18_days :
  (forall r : ℝ, (3 * 24 * r = 1) → (4 * 18 * r = 1)) :=
by
  intro r
  intro h
  sorry

end four_people_complete_task_in_18_days_l2084_208424


namespace landscape_avoid_repetition_l2084_208430

theorem landscape_avoid_repetition :
  let frames : ℕ := 5
  let days_per_month : ℕ := 30
  (Nat.factorial frames) / days_per_month = 4 := by
  sorry

end landscape_avoid_repetition_l2084_208430


namespace projectile_height_l2084_208498

theorem projectile_height (t : ℝ) (h : (-16 * t^2 + 80 * t = 100)) : t = 2.5 :=
sorry

end projectile_height_l2084_208498


namespace number_of_yellow_parrots_l2084_208403

-- Given conditions
def fraction_red : ℚ := 5 / 8
def total_parrots : ℕ := 120

-- Proof statement
theorem number_of_yellow_parrots : 
    (total_parrots : ℚ) * (1 - fraction_red) = 45 :=
by 
    sorry

end number_of_yellow_parrots_l2084_208403


namespace contrapositive_false_1_negation_false_1_l2084_208442

theorem contrapositive_false_1 (m : ℝ) : ¬ (¬ (∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

theorem negation_false_1 (m : ℝ) : ¬ ((m > 0) → ¬ (∃ x : ℝ, x^2 + x - m = 0)) :=
sorry

end contrapositive_false_1_negation_false_1_l2084_208442


namespace product_of_roots_of_t_squared_equals_49_l2084_208458

theorem product_of_roots_of_t_squared_equals_49 : 
  ∃ t : ℝ, (t^2 = 49) ∧ (t = 7 ∨ t = -7) ∧ (t * (7 + -7)) = -49 := 
by
  sorry

end product_of_roots_of_t_squared_equals_49_l2084_208458


namespace converse_of_proposition_inverse_of_proposition_contrapositive_of_proposition_l2084_208459

variable (a b : ℝ)

theorem converse_of_proposition :
  (ab > 0 → a > 0 ∧ b > 0) = false := sorry

theorem inverse_of_proposition :
  (a ≤ 0 ∨ b ≤ 0 → ab ≤ 0) = false := sorry

theorem contrapositive_of_proposition :
  (ab ≤ 0 → a ≤ 0 ∨ b ≤ 0) = true := sorry

end converse_of_proposition_inverse_of_proposition_contrapositive_of_proposition_l2084_208459


namespace opposite_signs_abs_larger_l2084_208432

theorem opposite_signs_abs_larger (a b : ℝ) (h1 : a + b < 0) (h2 : a * b < 0) :
  (a < 0 ∧ b > 0 ∧ |a| > |b|) ∨ (a > 0 ∧ b < 0 ∧ |b| > |a|) :=
sorry

end opposite_signs_abs_larger_l2084_208432


namespace find_remainder_l2084_208417

-- Main statement with necessary definitions and conditions
theorem find_remainder (x : ℤ) (h : (x + 11) % 31 = 18) :
  x % 62 = 7 :=
sorry

end find_remainder_l2084_208417


namespace root_interval_l2084_208490

noncomputable def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_interval (h1 : f 1 < 0) (h2 : f 1.5 > 0) (h3 : f 1.25 < 0) :
  ∃ c, 1.25 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  -- Proof by the Intermediate Value Theorem
  sorry

end root_interval_l2084_208490


namespace prob_pass_kth_intersection_l2084_208436

variable {n k : ℕ}

-- Definitions based on problem conditions
def prob_approach_highway (n : ℕ) : ℚ := 1 / n
def prob_exit_highway (n : ℕ) : ℚ := 1 / n

-- Theorem stating the required probability
theorem prob_pass_kth_intersection (h_n : n > 0) (h_k : k > 0) (h_k_le_n : k ≤ n) :
  (prob_approach_highway n) * (prob_exit_highway n * n) * (2 * k - 1) / n ^ 2 = 
  (2 * k * n - 2 * k ^ 2 + 2 * k - 1) / n ^ 2 := sorry

end prob_pass_kth_intersection_l2084_208436


namespace equal_charges_at_x_l2084_208457

theorem equal_charges_at_x (x : ℝ) : 
  (2.75 * x + 125 = 1.50 * x + 140) → (x = 12) := 
by
  sorry

end equal_charges_at_x_l2084_208457


namespace part_I_part_II_l2084_208455

-- Part I: Inequality solution
theorem part_I (x : ℝ) : 
  (abs (x - 1) ≥ 4 - abs (x - 3)) ↔ (x ≤ 0 ∨ x ≥ 4) := 
sorry

-- Part II: Minimum value of mn
theorem part_II (m n : ℕ) (h1 : (1:ℝ)/m + (1:ℝ)/(2*n) = 1) (hm : 0 < m) (hn : 0 < n) :
  (mn : ℕ) = 2 :=
sorry

end part_I_part_II_l2084_208455


namespace max_pairs_300_grid_l2084_208426

noncomputable def max_pairs (n : ℕ) (k : ℕ) (remaining_squares : ℕ) [Fintype (Fin n × Fin n)] : ℕ :=
  sorry

theorem max_pairs_300_grid :
  max_pairs 300 100 50000 = 49998 :=
by
  -- problem conditions
  let grid_size := 300
  let corner_size := 100
  let remaining_squares := 50000
  let no_checkerboard (squares : Fin grid_size × Fin grid_size → Prop) : Prop :=
    ∀ i j, ¬(squares (i, j) ∧ squares (i + 1, j) ∧ squares (i, j + 1) ∧ squares (i + 1, j + 1))
  -- statement of the bound
  have max_pairs := max_pairs grid_size corner_size remaining_squares
  exact sorry

end max_pairs_300_grid_l2084_208426


namespace fraction_simplest_form_l2084_208494

def fracA (a b : ℤ) : ℤ × ℤ := (|2 * a|, 5 * a^2 * b)
def fracB (a : ℤ) : ℤ × ℤ := (a, a^2 - 2 * a)
def fracC (a b : ℤ) : ℤ × ℤ := (3 * a + b, a + b)
def fracD (a b : ℤ) : ℤ × ℤ := (a^2 - a * b, a^2 - b^2)

theorem fraction_simplest_form (a b : ℤ) : (fracC a b).1 / (fracC a b).2 = (3 * a + b) / (a + b) :=
by sorry

end fraction_simplest_form_l2084_208494


namespace unknown_number_eq_0_5_l2084_208402

theorem unknown_number_eq_0_5 : 
  ∃ x : ℝ, x + ((2 / 3) * (3 / 8) + 4) - (8 / 16) = 4.25 ∧ x = 0.5 :=
by
  use 0.5
  sorry

end unknown_number_eq_0_5_l2084_208402


namespace find_m_l2084_208451

theorem find_m
  (m : ℝ)
  (A B : ℝ × ℝ × ℝ)
  (hA : A = (m, 2, 3))
  (hB : B = (1, -1, 1))
  (h_dist : (Real.sqrt ((m - 1) ^ 2 + (2 - (-1)) ^ 2 + (3 - 1) ^ 2) = Real.sqrt 13)) :
  m = 1 := 
sorry

end find_m_l2084_208451


namespace Amy_gets_fewest_cookies_l2084_208408

theorem Amy_gets_fewest_cookies:
  let area_Amy := 4 * Real.pi
  let area_Ben := 9
  let area_Carl := 8
  let area_Dana := (9 / 2) * Real.pi
  let num_cookies_Amy := 1 / area_Amy
  let num_cookies_Ben := 1 / area_Ben
  let num_cookies_Carl := 1 / area_Carl
  let num_cookies_Dana := 1 / area_Dana
  num_cookies_Amy < num_cookies_Ben ∧ num_cookies_Amy < num_cookies_Carl ∧ num_cookies_Amy < num_cookies_Dana :=
by
  sorry

end Amy_gets_fewest_cookies_l2084_208408


namespace blue_whale_tongue_weight_l2084_208401

theorem blue_whale_tongue_weight (ton_in_pounds : ℕ) (tons : ℕ) (blue_whale_tongue_weight : ℕ) :
  ton_in_pounds = 2000 → tons = 3 → blue_whale_tongue_weight = tons * ton_in_pounds → blue_whale_tongue_weight = 6000 :=
  by
  intros h1 h2 h3
  rw [h2] at h3
  rw [h1] at h3
  exact h3

end blue_whale_tongue_weight_l2084_208401


namespace balls_into_boxes_l2084_208422

-- Define the conditions
def balls : ℕ := 7
def boxes : ℕ := 4

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Prove the equivalent proof problem
theorem balls_into_boxes :
    (binom (balls - 1) (boxes - 1) = 20) ∧ (binom (balls + (boxes - 1)) (boxes - 1) = 120) := by
  sorry

end balls_into_boxes_l2084_208422


namespace hotdogs_needed_l2084_208450

theorem hotdogs_needed 
  (ella_hotdogs : ℕ) (emma_hotdogs : ℕ)
  (luke_multiple : ℕ) (hunter_multiple : ℚ)
  (h_ella : ella_hotdogs = 2)
  (h_emma : emma_hotdogs = 2)
  (h_luke : luke_multiple = 2)
  (h_hunter : hunter_multiple = (3/2)) :
  ella_hotdogs + emma_hotdogs + luke_multiple * (ella_hotdogs + emma_hotdogs) + hunter_multiple * (ella_hotdogs + emma_hotdogs) = 18 := by
    sorry

end hotdogs_needed_l2084_208450


namespace geometric_sequence_a3_a5_l2084_208420

variable {a : ℕ → ℝ}

theorem geometric_sequence_a3_a5 (h₀ : a 1 > 0) 
                                (h₁ : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 16) : 
                                a 3 + a 5 = 4 := 
sorry

end geometric_sequence_a3_a5_l2084_208420


namespace inequality_a_b_cubed_l2084_208409

theorem inequality_a_b_cubed (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^3 < b^3 :=
sorry

end inequality_a_b_cubed_l2084_208409


namespace percentage_books_not_sold_l2084_208437

theorem percentage_books_not_sold :
    let initial_stock := 700
    let books_sold_mon := 50
    let books_sold_tue := 82
    let books_sold_wed := 60
    let books_sold_thu := 48
    let books_sold_fri := 40
    let total_books_sold := books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri 
    let books_not_sold := initial_stock - total_books_sold
    let percentage_not_sold := (books_not_sold * 100) / initial_stock
    percentage_not_sold = 60 :=
by
  -- definitions
  let initial_stock := 700
  let books_sold_mon := 50
  let books_sold_tue := 82
  let books_sold_wed := 60
  let books_sold_thu := 48
  let books_sold_fri := 40
  let total_books_sold := books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri
  let books_not_sold := initial_stock - total_books_sold
  let percentage_not_sold := (books_not_sold * 100) / initial_stock
  have : percentage_not_sold = 60 := sorry
  exact this

end percentage_books_not_sold_l2084_208437


namespace mary_needs_more_apples_l2084_208485

theorem mary_needs_more_apples (total_pies : ℕ) (apples_per_pie : ℕ) (harvested_apples : ℕ) (y : ℕ) :
  total_pies = 10 → apples_per_pie = 8 → harvested_apples = 50 → y = 30 :=
by
  intro h1 h2 h3
  have total_apples_needed := total_pies * apples_per_pie
  have apples_needed_to_buy := total_apples_needed - harvested_apples
  have proof_needed : apples_needed_to_buy = y := sorry
  have proof_given : y = 30 := sorry
  have apples_needed := total_pies * apples_per_pie - harvested_apples
  exact proof_given

end mary_needs_more_apples_l2084_208485


namespace last_two_digits_of_quotient_l2084_208486

noncomputable def greatest_integer_not_exceeding (x : ℝ) : ℤ := ⌊x⌋

theorem last_two_digits_of_quotient :
  let a : ℤ := 10 ^ 93
  let b : ℤ := 10 ^ 31 + 3
  let x : ℤ := greatest_integer_not_exceeding (a / b : ℝ)
  (x % 100) = 8 :=
by
  sorry

end last_two_digits_of_quotient_l2084_208486


namespace triangle_XDE_area_l2084_208483

theorem triangle_XDE_area 
  (XY YZ XZ : ℝ) (hXY : XY = 8) (hYZ : YZ = 12) (hXZ : XZ = 14)
  (D E : ℝ → ℝ) (XD XE : ℝ) (hXD : XD = 3) (hXE : XE = 9) :
  ∃ (A : ℝ), A = 1/2 * XD * XE * (15 * Real.sqrt 17 / 56) ∧ A = 405 * Real.sqrt 17 / 112 :=
  sorry

end triangle_XDE_area_l2084_208483


namespace point_on_circle_l2084_208412

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def circle_radius := 5

def A : Point := {x := 2, y := -3}
def M : Point := {x := 5, y := -7}

theorem point_on_circle :
  distance A.x A.y M.x M.y = circle_radius :=
by
  sorry

end point_on_circle_l2084_208412


namespace single_dog_barks_per_minute_l2084_208444

theorem single_dog_barks_per_minute (x : ℕ) (h : 10 * 2 * x = 600) : x = 30 :=
by
  sorry

end single_dog_barks_per_minute_l2084_208444


namespace division_exponentiation_addition_l2084_208478

theorem division_exponentiation_addition :
  6 / -3 + 2^2 * (1 - 4) = -14 := by
sorry

end division_exponentiation_addition_l2084_208478


namespace triangle_perimeter_inequality_l2084_208493

theorem triangle_perimeter_inequality (x : ℕ) (h₁ : 15 + 24 > x) (h₂ : 15 + x > 24) (h₃ : 24 + x > 15) 
    (h₄ : ∃ x : ℕ, x > 9 ∧ x < 39) : 15 + 24 + x = 49 :=
by { sorry }

end triangle_perimeter_inequality_l2084_208493


namespace find_digit_property_l2084_208474

theorem find_digit_property (a x : ℕ) (h : 10 * a + x = a + x + a * x) : x = 9 :=
sorry

end find_digit_property_l2084_208474


namespace AB_passes_fixed_point_locus_of_N_l2084_208461

-- Definition of the parabola
def parabola (x y : ℝ) := y^2 = 4 * x

-- Definition of the point M which is the right-angle vertex
def M : ℝ × ℝ := (1, 2)

-- Statement for Part 1: Prove line AB passes through a fixed point
theorem AB_passes_fixed_point 
    (A B : ℝ × ℝ)
    (hA : parabola A.1 A.2)
    (hB : parabola B.1 B.2)
    (hM : M = (1, 2))
    (hRightAngle : (A.1 - 1) * (B.1 - 1) + (A.2 - 2) * (B.2 - 2) = 0) :
    ∃ P : ℝ × ℝ, P = (5, -2) := sorry

-- Statement for Part 2: Find the locus of point N
theorem locus_of_N 
    (A B : ℝ × ℝ)
    (hA : parabola A.1 A.2)
    (hB : parabola B.1 B.2)
    (hM : M = (1, 2))
    (hRightAngle : (A.1 - 1) * (B.1 - 1) + (A.2 - 2) * (B.2 - 2) = 0) 
    (N : ℝ × ℝ)
    (hN : ∃ t : ℝ, N = (t, -(t - 3))) :
    (N.1 - 3)^2 + N.2^2 = 8 ∧ N.1 ≠ 1 := sorry

end AB_passes_fixed_point_locus_of_N_l2084_208461


namespace find_x_l2084_208434

theorem find_x (x : ℤ) (h : x + -27 = 30) : x = 57 :=
sorry

end find_x_l2084_208434


namespace necessary_but_not_sufficient_l2084_208435

open Set

variable {α : Type*}

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient : 
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ b, b ∈ M ∧ b ∉ N) := 
by 
  sorry

end necessary_but_not_sufficient_l2084_208435


namespace find_m_l2084_208487

theorem find_m {x : ℝ} (m : ℝ) (h : ∀ x, (0 < x ∧ x < 2) ↔ (-1/2 * x^2 + 2 * x > m * x)) : m = 1 :=
sorry

end find_m_l2084_208487


namespace price_of_one_shirt_l2084_208449

variable (P : ℝ)

-- Conditions
def cost_two_shirts := 1.5 * P
def cost_three_shirts := 1.9 * P 
def full_price_three_shirts := 3 * P
def savings := full_price_three_shirts - cost_three_shirts

-- Correct answer
theorem price_of_one_shirt (hs : savings = 11) : P = 10 :=
by
  sorry

end price_of_one_shirt_l2084_208449


namespace greatest_three_digit_multiple_of_thirteen_l2084_208428

theorem greatest_three_digit_multiple_of_thirteen : 
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (13 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ (13 ∣ m) → m ≤ n) ∧ n = 988 :=
  sorry

end greatest_three_digit_multiple_of_thirteen_l2084_208428


namespace geometric_seq_arithmetic_condition_l2084_208499

open Real

noncomputable def common_ratio (q : ℝ) := (q > 0) ∧ (q^2 - q - 1 = 0)

def arithmetic_seq_condition (a1 a2 a3 : ℝ) := (a2 = (a1 + a3) / 2)

theorem geometric_seq_arithmetic_condition (a1 a2 a3 a4 a5 : ℝ) (q : ℝ)
  (h1 : 0 < q)
  (h2 : q^2 - q - 1 = 0)
  (h3 : a2 = q * a1)
  (h4 : a3 = q * a2)
  (h5 : a4 = q * a3)
  (h6 : a5 = q * a4)
  (h7 : arithmetic_seq_condition a1 a2 a3) :
  (a4 + a5) / (a3 + a4) = (1 + sqrt 5) / 2 := 
sorry

end geometric_seq_arithmetic_condition_l2084_208499


namespace seventh_term_geometric_sequence_l2084_208489

theorem seventh_term_geometric_sequence (a : ℝ) (a3 : ℝ) (r : ℝ) (n : ℕ) (term : ℕ → ℝ)
    (h_a : a = 3)
    (h_a3 : a3 = 3 / 64)
    (h_term : ∀ n, term n = a * r ^ (n - 1))
    (h_r : r = 1 / 8) :
    term 7 = 3 / 262144 :=
by
  sorry

end seventh_term_geometric_sequence_l2084_208489


namespace total_growth_of_trees_l2084_208414

theorem total_growth_of_trees :
  let t1_growth_rate := 1 -- first tree grows 1 meter/day
  let t2_growth_rate := 2 -- second tree grows 2 meters/day
  let t3_growth_rate := 2 -- third tree grows 2 meters/day
  let t4_growth_rate := 3 -- fourth tree grows 3 meters/day
  let days := 4
  t1_growth_rate * days + t2_growth_rate * days + t3_growth_rate * days + t4_growth_rate * days = 32 :=
by
  let t1_growth_rate := 1
  let t2_growth_rate := 2
  let t3_growth_rate := 2
  let t4_growth_rate := 3
  let days := 4
  sorry

end total_growth_of_trees_l2084_208414


namespace sequence_product_l2084_208475

theorem sequence_product {n : ℕ} (h : 1 < n) (a : ℕ → ℕ) (h₀ : ∀ n, a n = 2^n) : 
  a (n-1) * a (n+1) = 4^n :=
by sorry

end sequence_product_l2084_208475


namespace general_formula_sequence_sum_first_n_terms_l2084_208419

-- Define the axioms or conditions of the arithmetic sequence
axiom a3_eq_7 : ∃ a1 d : ℝ, a1 + 2 * d = 7
axiom a5_plus_a7_eq_26 : ∃ a1 d : ℝ, (a1 + 4 * d) + (a1 + 6 * d) = 26

-- State the theorem for the general formula of the arithmetic sequence
theorem general_formula_sequence (a1 d : ℝ) (h3 : a1 + 2 * d = 7) (h5_7 : (a1 + 4 * d) + (a1 + 6 * d) = 26) :
  ∀ n : ℕ, a1 + (n - 1) * d = 2 * n + 1 :=
sorry

-- State the theorem for the sum of the first n terms of the arithmetic sequence
theorem sum_first_n_terms (a1 d : ℝ) (h3 : a1 + 2 * d = 7) (h5_7 : (a1 + 4 * d) + (a1 + 6 * d) = 26) :
  ∀ n : ℕ, n * (a1 + (n - 1) * d + a1) / 2 = (n^2 + 2 * n) :=
sorry

end general_formula_sequence_sum_first_n_terms_l2084_208419


namespace residue_of_neg_2035_mod_47_l2084_208463

theorem residue_of_neg_2035_mod_47 : (-2035 : ℤ) % 47 = 33 := 
by
  sorry

end residue_of_neg_2035_mod_47_l2084_208463


namespace initial_average_score_l2084_208481

theorem initial_average_score (A : ℝ) :
  (∃ (A : ℝ), (16 * A = 15 * 64 + 24)) → A = 61.5 := 
by 
  sorry 

end initial_average_score_l2084_208481


namespace day_of_week_proof_l2084_208448

def day_of_week_17th_2003 := "Wednesday"
def day_of_week_305th_2003 := "Thursday"

theorem day_of_week_proof (d17 : day_of_week_17th_2003 = "Wednesday") : day_of_week_305th_2003 = "Thursday" := 
sorry

end day_of_week_proof_l2084_208448


namespace parabola_translation_correct_l2084_208453

variable (x : ℝ)

def original_parabola : ℝ := 5 * x^2

def translated_parabola : ℝ := 5 * (x - 2)^2 + 3

theorem parabola_translation_correct :
  translated_parabola x = 5 * (x - 2)^2 + 3 :=
by
  sorry

end parabola_translation_correct_l2084_208453


namespace middle_term_is_35_l2084_208470

-- Define the arithmetic sequence condition
def arithmetic_sequence (a b c d e f : ℤ) : Prop :=
  b - a = c - b ∧ c - b = d - c ∧ d - c = e - d ∧ e - d = f - e

-- Given sequence values
def seq1 := 23
def seq6 := 47

-- Theorem stating that the middle term y in the sequence is 35
theorem middle_term_is_35 (x y z w : ℤ) :
  arithmetic_sequence seq1 x y z w seq6 → y = 35 :=
by
  sorry

end middle_term_is_35_l2084_208470


namespace exradii_product_eq_area_squared_l2084_208404

variable (a b c : ℝ) (t : ℝ)
variable (s := (a + b + c) / 2)
variable (exradius_a exradius_b exradius_c : ℝ)

-- Define the conditions
axiom Heron : t^2 = s * (s - a) * (s - b) * (s - c)
axiom exradius_definitions : exradius_a = t / (s - a) ∧ exradius_b = t / (s - b) ∧ exradius_c = t / (s - c)

-- The theorem we want to prove
theorem exradii_product_eq_area_squared : exradius_a * exradius_b * exradius_c = t^2 := sorry

end exradii_product_eq_area_squared_l2084_208404
