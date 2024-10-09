import Mathlib

namespace area_of_one_cookie_l744_74467

theorem area_of_one_cookie (L W : ℝ)
    (W_eq_15 : W = 15)
    (circumference_condition : 4 * L + 2 * W = 70) :
    L * W = 150 :=
by
  sorry

end area_of_one_cookie_l744_74467


namespace train_travel_distance_l744_74488

theorem train_travel_distance
  (coal_per_mile_lb : ℝ)
  (remaining_coal_lb : ℝ)
  (travel_distance_per_unit_mile : ℝ)
  (units_per_unit_lb : ℝ)
  (remaining_units : ℝ)
  (total_distance : ℝ) :
  coal_per_mile_lb = 2 →
  remaining_coal_lb = 160 →
  travel_distance_per_unit_mile = 5 →
  units_per_unit_lb = remaining_coal_lb / coal_per_mile_lb →
  remaining_units = units_per_unit_lb →
  total_distance = remaining_units * travel_distance_per_unit_mile →
  total_distance = 400 :=
by
  sorry

end train_travel_distance_l744_74488


namespace fraction_four_or_older_l744_74423

theorem fraction_four_or_older (total_students : ℕ) (under_three : ℕ) (not_between_three_and_four : ℕ)
  (h_total : total_students = 300) (h_under_three : under_three = 20) (h_not_between_three_and_four : not_between_three_and_four = 50) :
  (not_between_three_and_four - under_three) / total_students = 1 / 10 :=
by
  sorry

end fraction_four_or_older_l744_74423


namespace impossible_to_form_triangle_l744_74457

theorem impossible_to_form_triangle 
  (a b c : ℝ)
  (h1 : a = 9) 
  (h2 : b = 4) 
  (h3 : c = 3) 
  : ¬(a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  rw [h1, h2, h3]
  simp
  sorry

end impossible_to_form_triangle_l744_74457


namespace total_plates_l744_74485

-- define the variables for the number of plates
def plates_lobster_rolls : Nat := 25
def plates_spicy_hot_noodles : Nat := 14
def plates_seafood_noodles : Nat := 16

-- state the problem as a theorem
theorem total_plates :
  plates_lobster_rolls + plates_spicy_hot_noodles + plates_seafood_noodles = 55 := by
  sorry

end total_plates_l744_74485


namespace factor_expression_eq_l744_74472

-- Define the given expression
def given_expression (x : ℝ) : ℝ :=
  (12 * x^3 + 90 * x - 6) - (-3 * x^3 + 5 * x - 6)

-- Define the correct factored form
def factored_expression (x : ℝ) : ℝ :=
  5 * x * (3 * x^2 + 17)

-- The theorem stating the equality of the given expression and its factored form
theorem factor_expression_eq (x : ℝ) : given_expression x = factored_expression x :=
  by
  sorry

end factor_expression_eq_l744_74472


namespace circle_radius_l744_74489

theorem circle_radius (M N r : ℝ) (h1 : M = Real.pi * r^2) (h2 : N = 2 * Real.pi * r) (h3 : M / N = 25) : r = 50 :=
by
  sorry

end circle_radius_l744_74489


namespace train_pass_jogger_in_40_seconds_l744_74414

noncomputable def time_to_pass_jogger (jogger_speed_kmh : ℝ) (train_speed_kmh : ℝ) (initial_distance_m : ℝ) (train_length_m : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh - jogger_speed_kmh
  let relative_speed_ms := relative_speed_kmh * (5 / 18)  -- Conversion from km/hr to m/s
  let total_distance_m := initial_distance_m + train_length_m
  total_distance_m / relative_speed_ms

theorem train_pass_jogger_in_40_seconds :
  time_to_pass_jogger 9 45 280 120 = 40 := by
  sorry

end train_pass_jogger_in_40_seconds_l744_74414


namespace percentage_of_boys_l744_74495

theorem percentage_of_boys (total_students boys_per_group girls_per_group : ℕ)
  (ratio_condition : boys_per_group + girls_per_group = 7)
  (total_condition : total_students = 42)
  (ratio_b_condition : boys_per_group = 3)
  (ratio_g_condition : girls_per_group = 4) :
  (boys_per_group : ℚ) / (boys_per_group + girls_per_group : ℚ) * 100 = 42.86 :=
by sorry

end percentage_of_boys_l744_74495


namespace trigonometric_identity_l744_74436

theorem trigonometric_identity (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 1) :
  Real.cos (2 * x) + Real.cos (2 * y) + 2 * Real.cos (2 * z) = 2 :=
by
  sorry

end trigonometric_identity_l744_74436


namespace solve_rational_numbers_l744_74434

theorem solve_rational_numbers:
  ∃ (a b c d : ℚ),
    8 * a^2 - 3 * b^2 + 5 * c^2 + 16 * d^2 - 10 * a * b + 42 * c * d + 18 * a + 22 * b - 2 * c - 54 * d = 42 ∧
    15 * a^2 - 3 * b^2 + 21 * c^2 - 5 * d^2 + 4 * a * b + 32 * c * d - 28 * a + 14 * b - 54 * c - 52 * d = -22 ∧
    a = 4 / 7 ∧ b = 19 / 7 ∧ c = 29 / 19 ∧ d = -6 / 19 :=
  sorry

end solve_rational_numbers_l744_74434


namespace proof_problem_l744_74440

noncomputable def f (x : ℝ) : ℝ :=
  Real.log ((1 + Real.sqrt x) / (1 - Real.sqrt x))

theorem proof_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) :
  f ( (5 * x + 2 * x^2) / (1 + 5 * x + 3 * x^2) ) = Real.sqrt 5 * f x :=
by
  sorry

end proof_problem_l744_74440


namespace no_valid_k_exists_l744_74454

theorem no_valid_k_exists {k : ℕ} : ¬(∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p + q = 41 ∧ p * q = k) :=
by
  sorry

end no_valid_k_exists_l744_74454


namespace find_b_l744_74459

def passesThrough (b c : ℝ) (P : ℝ × ℝ) : Prop :=
  P.2 = P.1^2 + b * P.1 + c

theorem find_b (b c : ℝ)
  (H1 : passesThrough b c (1, 2))
  (H2 : passesThrough b c (5, 2)) :
  b = -6 :=
by
  sorry

end find_b_l744_74459


namespace bananas_to_oranges_l744_74417

theorem bananas_to_oranges :
  (3 / 4) * 12 * b = 9 * o →
  ((3 / 5) * 15 * b) = 9 * o := 
by
  sorry

end bananas_to_oranges_l744_74417


namespace emily_total_points_l744_74413

def score_round_1 : ℤ := 16
def score_round_2 : ℤ := 33
def score_round_3 : ℤ := -25
def score_round_4 : ℤ := 46
def score_round_5 : ℤ := 12
def score_round_6 : ℤ := 30 - (2 * score_round_5 / 3)

def total_score : ℤ :=
  score_round_1 + score_round_2 + score_round_3 + score_round_4 + score_round_5 + score_round_6

theorem emily_total_points : total_score = 104 := by
  sorry

end emily_total_points_l744_74413


namespace max_marks_is_400_l744_74471

theorem max_marks_is_400 :
  ∃ M : ℝ, (0.30 * M = 120) ∧ (M = 400) := 
by 
  sorry

end max_marks_is_400_l744_74471


namespace boys_running_speed_l744_74402
-- Import the necessary libraries

-- Define the input conditions:
def side_length : ℝ := 50
def time_seconds : ℝ := 80
def conversion_factor_meters_to_kilometers : ℝ := 1000
def conversion_factor_seconds_to_hours : ℝ := 3600

-- Define the theorem:
theorem boys_running_speed :
  let perimeter := 4 * side_length
  let distance_kilometers := perimeter / conversion_factor_meters_to_kilometers
  let time_hours := time_seconds / conversion_factor_seconds_to_hours
  distance_kilometers / time_hours = 9 :=
by
  sorry

end boys_running_speed_l744_74402


namespace polygon_sides_l744_74497

-- Define the given condition formally
def sum_of_internal_and_external_angle (n : ℕ) : ℕ :=
  (n - 2) * 180 + (1) -- This represents the sum of internal angles plus an external angle

theorem polygon_sides (n : ℕ) : 
  sum_of_internal_and_external_angle n = 1350 → n = 9 :=
by
  sorry

end polygon_sides_l744_74497


namespace lines_symmetric_about_y_axis_l744_74400

theorem lines_symmetric_about_y_axis (m n p : ℝ) :
  (∀ x y : ℝ, x + m * y + 5 = 0 ↔ x + n * y + p = 0)
  ↔ (m = -n ∧ p = -5) :=
sorry

end lines_symmetric_about_y_axis_l744_74400


namespace total_students_in_middle_school_l744_74470

/-- Given that 20% of the students are in the band and there are 168 students in the band,
    prove that the total number of students in the middle school is 840. -/
theorem total_students_in_middle_school (total_students : ℕ) (band_students : ℕ) 
  (h1 : 20 ≤ 100)
  (h2 : band_students = 168)
  (h3 : band_students = 20 * total_students / 100) 
  : total_students = 840 :=
sorry

end total_students_in_middle_school_l744_74470


namespace ratio_of_incomes_l744_74483

theorem ratio_of_incomes 
  (E1 E2 I1 I2 : ℕ)
  (h1 : E1 / E2 = 3 / 2)
  (h2 : E1 = I1 - 1200)
  (h3 : E2 = I2 - 1200)
  (h4 : I1 = 3000) :
  I1 / I2 = 5 / 4 :=
sorry

end ratio_of_incomes_l744_74483


namespace sum_of_100th_row_l744_74458

def triangularArraySum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2^(n+1) - 3*n

theorem sum_of_100th_row :
  triangularArraySum 100 = 2^100 - 297 :=
by
  sorry

end sum_of_100th_row_l744_74458


namespace percentage_singing_l744_74491

def total_rehearsal_time : ℕ := 75
def warmup_time : ℕ := 6
def notes_time : ℕ := 30
def words_time (t : ℕ) : ℕ := t
def singing_time (t : ℕ) : ℕ := total_rehearsal_time - warmup_time - notes_time - words_time t
def singing_percentage (t : ℕ) : ℕ := (singing_time t * 100) / total_rehearsal_time

theorem percentage_singing (t : ℕ) : (singing_percentage t) = (4 * (39 - t)) / 3 :=
by
  sorry

end percentage_singing_l744_74491


namespace range_of_a_l744_74465

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then x^2 + 2 * a else -x

theorem range_of_a (a : ℝ) (h : a < 0) (hf : f a (1 - a) ≥ f a (1 + a)) : -2 ≤ a ∧ a ≤ -1 :=
  sorry

end range_of_a_l744_74465


namespace solution_set_of_f_greater_than_one_l744_74433

theorem solution_set_of_f_greater_than_one (f : ℝ → ℝ) (h_inv : ∀ x, f (x / (x + 3)) = x) :
  {x | f x > 1} = {x | 1 / 4 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_f_greater_than_one_l744_74433


namespace count_divisors_divisible_exactly_2007_l744_74430

-- Definitions and conditions
def prime_factors_2006 : List Nat := [2, 17, 59]

def prime_factors_2006_pow_2006 : List (Nat × Nat) := [(2, 2006), (17, 2006), (59, 2006)]

def number_of_divisors (n : Nat) : Nat :=
  prime_factors_2006_pow_2006.foldl (λ acc ⟨p, exp⟩ => acc * (exp + 1)) 1

theorem count_divisors_divisible_exactly_2007 : 
  (number_of_divisors (2^2006 * 17^2006 * 59^2006) = 3) :=
  sorry

end count_divisors_divisible_exactly_2007_l744_74430


namespace both_shots_hit_both_shots_missed_exactly_one_shot_hit_at_least_one_shot_hit_l744_74443

variables (p1 p2 : Prop)

theorem both_shots_hit (p1 p2 : Prop) : (p1 ∧ p2) ↔ (p1 ∧ p2) :=
by sorry

theorem both_shots_missed (p1 p2 : Prop) : (¬p1 ∧ ¬p2) ↔ (¬p1 ∧ ¬p2) :=
by sorry

theorem exactly_one_shot_hit (p1 p2 : Prop) : ((p1 ∧ ¬p2) ∨ (p2 ∧ ¬p1)) ↔ ((p1 ∧ ¬p2) ∨ (p2 ∧ ¬p1)) :=
by sorry

theorem at_least_one_shot_hit (p1 p2 : Prop) : (p1 ∨ p2) ↔ (p1 ∨ p2) :=
by sorry

end both_shots_hit_both_shots_missed_exactly_one_shot_hit_at_least_one_shot_hit_l744_74443


namespace hyperbola_eccentricity_l744_74412

theorem hyperbola_eccentricity (a : ℝ) (e : ℝ) :
  (∀ x y : ℝ, y = (1 / 8) * x^2 → x^2 = 8 * y) →
  (∀ y x : ℝ, y^2 / a - x^2 = 1 → a + 1 = 4) →
  e^2 = 4 / 3 →
  e = 2 * Real.sqrt 3 / 3 :=
by
  intros h1 h2 h3
  sorry

end hyperbola_eccentricity_l744_74412


namespace pole_length_is_5_l744_74486

theorem pole_length_is_5 (x : ℝ) (gate_width gate_height : ℝ) 
  (h_gate_wide : gate_width = 3) 
  (h_pole_taller : gate_height = x - 1) 
  (h_diagonal : x^2 = gate_height^2 + gate_width^2) : 
  x = 5 :=
by
  sorry

end pole_length_is_5_l744_74486


namespace triangle_inequality_l744_74439

theorem triangle_inequality (A B C : ℝ) :
  ∀ (a b c : ℝ), (a = 2 * Real.sin (A / 2) * Real.cos (A / 2)) ∧
                 (b = 2 * Real.sin (B / 2) * Real.cos (B / 2)) ∧
                 (c = Real.cos ((A + B) / 2)) ∧
                 (x = Real.sqrt (Real.tan (A / 2) * Real.tan (B / 2)))
                 → (Real.sqrt (a * b) / Real.sin (C / 2) ≥ 3 * Real.sqrt 3 * Real.tan (A / 2) * Real.tan (B / 2)) := by {
  sorry
}

end triangle_inequality_l744_74439


namespace determine_values_of_a_and_b_l744_74474

namespace MathProofProblem

variables (a b : ℤ)

theorem determine_values_of_a_and_b :
  (b + 1 = 2) ∧ (a - 1 ≠ -3) ∧ (a - 1 = -3) ∧ (b + 1 ≠ 2) ∧ (a - 1 = 2) ∧ (b + 1 = -3) →
  a = 3 ∧ b = -4 := by
  sorry

end MathProofProblem

end determine_values_of_a_and_b_l744_74474


namespace ratio_a_c_l744_74487

theorem ratio_a_c (a b c : ℕ) (h1 : a / b = 5 / 3) (h2 : b / c = 1 / 5) : a / c = 1 / 3 :=
sorry

end ratio_a_c_l744_74487


namespace geometric_sum_n_equals_4_l744_74438

def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def S (n : ℕ) : ℚ := a * ((1 - r^n) / (1 - r))
def sum_value : ℚ := 26 / 81

theorem geometric_sum_n_equals_4 (n : ℕ) (h : S n = sum_value) : n = 4 :=
by sorry

end geometric_sum_n_equals_4_l744_74438


namespace C_finishes_job_in_days_l744_74477

theorem C_finishes_job_in_days :
  ∀ (A B C : ℚ),
    (A + B = 1 / 15) →
    (A + B + C = 1 / 3) →
    1 / C = 3.75 :=
by
  intros A B C hab habc
  sorry

end C_finishes_job_in_days_l744_74477


namespace ordered_triples_2022_l744_74482

theorem ordered_triples_2022 :
  ∃ n : ℕ, n = 13 ∧ (∃ a c : ℕ, a ≤ c ∧ (a * c = 2022^2)) := by
  sorry

end ordered_triples_2022_l744_74482


namespace negation_of_universal_proposition_l744_74420

theorem negation_of_universal_proposition :
  (∃ x : ℤ, x % 5 = 0 ∧ ¬ (x % 2 = 1)) ↔ ¬ (∀ x : ℤ, x % 5 = 0 → (x % 2 = 1)) :=
by sorry

end negation_of_universal_proposition_l744_74420


namespace compute_expression_l744_74496

theorem compute_expression : (-3) * 2 + 4 = -2 := 
by
  sorry

end compute_expression_l744_74496


namespace find_x_l744_74437

noncomputable def h (x : ℝ) : ℝ := (2 * x^2 + 3 * x + 1)^(1 / 3) / 5^(1/3)

theorem find_x (x : ℝ) :
  h (3 * x) = 3 * h x ↔ x = -1 + (10^(1/2)) / 3 ∨ x = -1 - (10^(1/2)) / 3 := by
  sorry

end find_x_l744_74437


namespace simplify_fraction_l744_74426

theorem simplify_fraction : (140 / 9800) * 35 = 1 / 70 := 
by
  -- Proof steps would go here.
  sorry

end simplify_fraction_l744_74426


namespace minimum_boys_needed_l744_74484

theorem minimum_boys_needed (k n m : ℕ) (hn : n > 0) (hm : m > 0) (h : 100 * n + m * k = 10 * k) : n + m = 6 :=
by
  sorry

end minimum_boys_needed_l744_74484


namespace empty_plane_speed_l744_74409

variable (V : ℝ)

def speed_first_plane (V : ℝ) : ℝ := V - 2 * 50
def speed_second_plane (V : ℝ) : ℝ := V - 2 * 60
def speed_third_plane (V : ℝ) : ℝ := V - 2 * 40

theorem empty_plane_speed (V : ℝ) (h : (speed_first_plane V + speed_second_plane V + speed_third_plane V) / 3 = 500) : V = 600 :=
by 
  sorry

end empty_plane_speed_l744_74409


namespace customers_left_l744_74466

theorem customers_left (initial_customers : ℝ) (first_left : ℝ) (second_left : ℝ) : initial_customers = 36.0 ∧ first_left = 19.0 ∧ second_left = 14.0 → initial_customers - first_left - second_left = 3.0 :=
by
  intros h
  sorry

end customers_left_l744_74466


namespace congruent_rectangle_perimeter_l744_74431

theorem congruent_rectangle_perimeter (x y w l P : ℝ) 
  (h1 : x + 2 * w = 2 * y) 
  (h2 : x + 2 * l = y) 
  (hP : P = 2 * l + 2 * w) : 
  P = 3 * y - 2 * x :=
by sorry

end congruent_rectangle_perimeter_l744_74431


namespace maximize_profit_l744_74405

noncomputable def profit (x : ℕ) : ℝ :=
  if x ≤ 200 then
    (0.40 - 0.24) * 30 * x
  else if x ≤ 300 then
    (0.40 - 0.24) * 10 * 200 + (0.40 - 0.24) * 20 * x - (0.24 - 0.08) * 10 * (x - 200)
  else
    (0.40 - 0.24) * 10 * 200 + (0.40 - 0.24) * 20 * 300 - (0.24 - 0.08) * 10 * (x - 200) - (0.24 - 0.08) * 20 * (x - 300)

theorem maximize_profit : ∀ x : ℕ, 
  profit 300 = 1120 ∧ (∀ y : ℕ, profit y ≤ 1120) :=
by
  sorry

end maximize_profit_l744_74405


namespace diamond_eight_five_l744_74416

def diamond (a b : ℕ) : ℕ := (a + b) * ((a - b) * (a - b))

theorem diamond_eight_five : diamond 8 5 = 117 := by
  sorry

end diamond_eight_five_l744_74416


namespace range_of_a_sqrt10_e_bounds_l744_74478

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f a x ≤ g x) ↔ a ≤ 1 :=
by
  sorry

theorem sqrt10_e_bounds : 
  (1095 / 1000 : ℝ) < Real.exp (1/10 : ℝ) ∧ Real.exp (1/10 : ℝ) < (2000 / 1791 : ℝ) :=
by
  sorry

end range_of_a_sqrt10_e_bounds_l744_74478


namespace invalid_perimeters_l744_74468

theorem invalid_perimeters (x : ℕ) (h1 : 18 < x) (h2 : x < 42) :
  (42 + x ≠ 58) ∧ (42 + x ≠ 85) :=
by
  sorry

end invalid_perimeters_l744_74468


namespace x_is_integer_l744_74481

theorem x_is_integer 
  (x : ℝ)
  (h1 : ∃ k1 : ℤ, x^2 - x = k1)
  (h2 : ∃ (n : ℕ) (_ : n > 2) (k2 : ℤ), x^n - x = k2) : 
  ∃ (m : ℤ), x = m := 
sorry

end x_is_integer_l744_74481


namespace sequence_linear_constant_l744_74403

open Nat

theorem sequence_linear_constant (a : ℕ → ℕ) 
  (h1 : ∀ n, 1 < a 1 ∧ a (n + 1) > a n)
  (h2 : ∀ n, a (n + a n) = 2 * a n) :
  ∃ c : ℕ, ∀ n, a n = n + c := 
sorry

end sequence_linear_constant_l744_74403


namespace sum_of_cubes_consecutive_divisible_by_9_l744_74435

theorem sum_of_cubes_consecutive_divisible_by_9 (n : ℤ) : 9 ∣ (n-1)^3 + n^3 + (n+1)^3 :=
  sorry

end sum_of_cubes_consecutive_divisible_by_9_l744_74435


namespace repair_cost_is_5000_l744_74408

-- Define the initial cost of the machine
def initial_cost : ℝ := 9000

-- Define the transportation charges
def transportation_charges : ℝ := 1000

-- Define the selling price
def selling_price : ℝ := 22500

-- Define the profit percentage as a decimal
def profit_percentage : ℝ := 0.5

-- Define the total cost including repairs
def total_cost (repair_cost : ℝ) : ℝ :=
  initial_cost + transportation_charges + repair_cost

-- Define the equation for selling price with 50% profit
def selling_price_equation (repair_cost : ℝ) : Prop :=
  selling_price = (1 + profit_percentage) * total_cost repair_cost

-- State the proof problem in Lean
theorem repair_cost_is_5000 : selling_price_equation 5000 :=
by 
  sorry

end repair_cost_is_5000_l744_74408


namespace volume_of_set_l744_74463

theorem volume_of_set (m n p : ℕ) (h_rel_prime : Nat.gcd n p = 1) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_pos_p : 0 < p) 
  (h_volume : (m + n * Real.pi) / p = (324 + 37 * Real.pi) / 3) : 
  m + n + p = 364 := 
  sorry

end volume_of_set_l744_74463


namespace age_difference_l744_74476

theorem age_difference (A B : ℕ) (h1 : B = 34) (h2 : A + 10 = 2 * (B - 10)) : A - B = 4 :=
by
  sorry

end age_difference_l744_74476


namespace sum_and_product_of_conjugates_l744_74410

theorem sum_and_product_of_conjugates (c d : ℚ) 
  (h1 : 2 * c = 6)
  (h2 : c^2 - 4 * d = 4) :
  c + d = 17 / 4 :=
by
  sorry

end sum_and_product_of_conjugates_l744_74410


namespace hyperbola_asymptote_slope_l744_74401

theorem hyperbola_asymptote_slope :
  (∀ x y : ℝ, (x^2 / 100 - y^2 / 64 = 1) → y = (4/5) * x ∨ y = -(4/5) * x) :=
by
  sorry

end hyperbola_asymptote_slope_l744_74401


namespace quadratic_min_n_l744_74475

theorem quadratic_min_n (m n : ℝ) : 
  (∃ x : ℝ, (x^2 + (m - 2023) * x + (n - 1)) = 0) ∧ 
  (m - 2023)^2 - 4 * (n - 1) = 0 → 
  n = 1 := 
sorry

end quadratic_min_n_l744_74475


namespace cost_of_toaster_l744_74404

-- Definitions based on the conditions
def initial_spending : ℕ := 3000
def tv_return : ℕ := 700
def returned_bike_cost : ℕ := 500
def sold_bike_cost : ℕ := returned_bike_cost + (returned_bike_cost / 5)
def selling_price : ℕ := (4 * sold_bike_cost) / 5
def total_out_of_pocket : ℕ := 2020

-- Proving the cost of the toaster
theorem cost_of_toaster : initial_spending - (tv_return + returned_bike_cost) + selling_price - total_out_of_pocket = 260 := by
  sorry

end cost_of_toaster_l744_74404


namespace problem_equivalent_l744_74406

-- Define the problem conditions
def an (n : ℕ) : ℤ := -4 * n + 2

-- Arithmetic sequence: given conditions
axiom arith_seq_cond1 : an 2 + an 7 = -32
axiom arith_seq_cond2 : an 3 + an 8 = -40

-- Suppose the sequence {an + bn} is geometric with first term 1 and common ratio 2
def geom_seq (n : ℕ) : ℤ := 2 ^ (n - 1)
def bn (n : ℕ) : ℤ := geom_seq n - an n

-- To prove: sum of the first n terms of {bn}, denoted as Sn
def Sn (n : ℕ) : ℤ := (n * (2 + 4 * n - 2)) / 2 + (1 - 2 ^ n) / (1 - 2)

theorem problem_equivalent (n : ℕ) :
  an 2 + an 7 = -32 ∧
  an 3 + an 8 = -40 ∧
  (∀ n : ℕ, an n + bn n = geom_seq n) →
  Sn n = 2 * n ^ 2 + 2 ^ n - 1 :=
by
  intros h
  sorry

end problem_equivalent_l744_74406


namespace factorial_plus_one_div_prime_l744_74449

theorem factorial_plus_one_div_prime (n : ℕ) (h : (n! + 1) % (n + 1) = 0) : Nat.Prime (n + 1) := 
sorry

end factorial_plus_one_div_prime_l744_74449


namespace gas_and_maintenance_money_l744_74446

theorem gas_and_maintenance_money
  (income : ℝ := 3200)
  (rent : ℝ := 1250)
  (utilities : ℝ := 150)
  (retirement_savings : ℝ := 400)
  (groceries : ℝ := 300)
  (insurance : ℝ := 200)
  (miscellaneous_expenses : ℝ := 200)
  (car_payment : ℝ := 350) :
  income - (rent + utilities + retirement_savings + groceries + insurance + miscellaneous_expenses + car_payment) = 350 :=
by
  sorry

end gas_and_maintenance_money_l744_74446


namespace factorial_comparison_l744_74492

theorem factorial_comparison :
  (Nat.factorial (Nat.factorial 100)) <
  (Nat.factorial 99)^(Nat.factorial 100) * (Nat.factorial 100)^(Nat.factorial 99) :=
  sorry

end factorial_comparison_l744_74492


namespace min_guesses_correct_l744_74407

def min_guesses (n k : ℕ) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h : n > k) :
  (min_guesses n k = 2 ↔ n = 2 * k) ∧ (min_guesses n k = 1 ↔ n ≠ 2 * k) := by
  sorry

end min_guesses_correct_l744_74407


namespace diamond_value_l744_74448

def diamond (a b : ℕ) : ℕ := 4 * a - 2 * b

theorem diamond_value : diamond 6 3 = 18 :=
by
  sorry

end diamond_value_l744_74448


namespace find_m_l744_74498

theorem find_m 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ)
  (h_f : ∀ x, f x = x^2 - 4*x + m)
  (h_g : ∀ x, g x = x^2 - 2*x + 2*m)
  (h_cond : 3 * f 3 = g 3)
  : m = 12 := 
sorry

end find_m_l744_74498


namespace mouse_seed_hiding_l744_74464

theorem mouse_seed_hiding : 
  ∀ (h_m h_r x : ℕ), 
  4 * h_m = x →
  7 * h_r = x →
  h_m = h_r + 3 →
  x = 28 :=
by
  intros h_m h_r x H1 H2 H3
  sorry

end mouse_seed_hiding_l744_74464


namespace sum_of_digits_of_number_of_rows_l744_74421

theorem sum_of_digits_of_number_of_rows :
  ∃ N, (3 * (N * (N + 1) / 2) = 1575) ∧ (Nat.digits 10 N).sum = 8 :=
by
  sorry

end sum_of_digits_of_number_of_rows_l744_74421


namespace number_of_numbers_is_11_l744_74415

noncomputable def total_number_of_numbers 
  (avg_all : ℝ) (avg_first_6 : ℝ) (avg_last_6 : ℝ) (num_6th : ℝ) : ℝ :=
if h : avg_all = 60 ∧ avg_first_6 = 58 ∧ avg_last_6 = 65 ∧ num_6th = 78 
then 11 else 0 

-- The theorem statement assuming the problem conditions
theorem number_of_numbers_is_11
  {n S : ℝ}
  (avg_all : ℝ) (avg_first_6 : ℝ) (avg_last_6 : ℝ) (num_6th : ℝ) 
  (h1 : avg_all = 60) 
  (h2 : avg_first_6 = 58)
  (h3 : avg_last_6 = 65)
  (h4 : num_6th = 78) 
  (h5 : S = 6 * avg_first_6 + 6 * avg_last_6 - num_6th)
  (h6 : S = avg_all * n) : 
  n = 11 := sorry

end number_of_numbers_is_11_l744_74415


namespace sculpture_cost_in_INR_l744_74451

def USD_per_NAD := 1 / 5
def INR_per_USD := 8
def cost_in_NAD := 200
noncomputable def cost_in_INR := (cost_in_NAD * USD_per_NAD) * INR_per_USD

theorem sculpture_cost_in_INR :
  cost_in_INR = 320 := by
  sorry

end sculpture_cost_in_INR_l744_74451


namespace not_lengths_of_external_diagonals_l744_74444

theorem not_lengths_of_external_diagonals (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) :
  (¬ (a = 5 ∧ b = 6 ∧ c = 9)) :=
by
  sorry

end not_lengths_of_external_diagonals_l744_74444


namespace standard_eq_of_parabola_l744_74441

-- Conditions:
-- The point (1, -2) lies on the parabola.
def point_on_parabola : Prop := ∃ p : ℝ, (1, -2).2^2 = 2 * p * (1, -2).1 ∨ (1, -2).1^2 = 2 * p * (1, -2).2

-- Question to be proved:
-- The standard equation of the parabola passing through the point (1, -2) is y^2 = 4x or x^2 = - (1/2) y.
theorem standard_eq_of_parabola : point_on_parabola → (y^2 = 4*x ∨ x^2 = -(1/(2:ℝ)) * y) :=
by
  sorry -- proof to be provided

end standard_eq_of_parabola_l744_74441


namespace books_total_l744_74469

theorem books_total (Tim_books Sam_books : ℕ) (h1 : Tim_books = 44) (h2 : Sam_books = 52) : Tim_books + Sam_books = 96 := 
by
  sorry

end books_total_l744_74469


namespace evaporate_water_l744_74445

theorem evaporate_water (M : ℝ) (W_i W_f x : ℝ) (d : ℝ)
  (h_initial_mass : M = 500)
  (h_initial_water_content : W_i = 0.85 * M)
  (h_final_water_content : W_f = 0.75 * (M - x))
  (h_desired_fraction : d = 0.75) :
  x = 200 := 
  sorry

end evaporate_water_l744_74445


namespace stratified_sampling_correct_l744_74456

-- Define the total number of employees
def total_employees : ℕ := 100

-- Define the number of employees in each age group
def under_30 : ℕ := 20
def between_30_and_40 : ℕ := 60
def over_40 : ℕ := 20

-- Define the number of people to be drawn
def total_drawn : ℕ := 20

-- Function to calculate number of people to be drawn from each group
def stratified_draw (group_size : ℕ) (total_size : ℕ) (drawn : ℕ) : ℕ :=
  (group_size * drawn) / total_size

-- The proof problem statement
theorem stratified_sampling_correct :
  stratified_draw under_30 total_employees total_drawn = 4 ∧
  stratified_draw between_30_and_40 total_employees total_drawn = 12 ∧
  stratified_draw over_40 total_employees total_drawn = 4 := by
  sorry

end stratified_sampling_correct_l744_74456


namespace frank_problems_each_type_l744_74494

theorem frank_problems_each_type (bill_total : ℕ) (ryan_ratio bill_total_ratio : ℕ) (frank_ratio ryan_total : ℕ) (types : ℕ)
  (h1 : bill_total = 20)
  (h2 : ryan_ratio = 2)
  (h3 : bill_total_ratio = bill_total * ryan_ratio)
  (h4 : ryan_total = bill_total_ratio)
  (h5 : frank_ratio = 3)
  (h6 : ryan_total * frank_ratio = ryan_total) :
  (ryan_total * frank_ratio) / types = 30 :=
by
  sorry

end frank_problems_each_type_l744_74494


namespace car_travel_l744_74460

namespace DistanceTravel

/- Define the conditions -/
def distance_initial : ℕ := 120
def car_speed : ℕ := 80

/- Define the relationship between y and x -/
def y (x : ℝ) : ℝ := distance_initial - car_speed * x

/- Prove that y is a linear function and verify the value of y at x = 0.8 -/
theorem car_travel (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1.5) : 
  (y x = distance_initial - car_speed * x) ∧ 
  (y x = 120 - 80 * x) ∧ 
  (x = 0.8 → y x = 56) :=
sorry

end DistanceTravel

end car_travel_l744_74460


namespace time_to_cross_first_platform_l744_74411

noncomputable section

def train_length : ℝ := 310
def platform_1_length : ℝ := 110
def platform_2_length : ℝ := 250
def crossing_time_platform_2 : ℝ := 20

def total_distance_2 (train_length platform_2_length : ℝ) : ℝ :=
  train_length + platform_2_length

def train_speed (total_distance_2 crossing_time_platform_2 : ℝ) : ℝ :=
  total_distance_2 / crossing_time_platform_2

def total_distance_1 (train_length platform_1_length : ℝ) : ℝ :=
  train_length + platform_1_length

def crossing_time_platform_1 (total_distance_1 train_speed : ℝ) : ℝ :=
  total_distance_1 / train_speed

theorem time_to_cross_first_platform :
  crossing_time_platform_1 (total_distance_1 train_length platform_1_length)
                           (train_speed (total_distance_2 train_length platform_2_length)
                                        crossing_time_platform_2) 
  = 15 :=
by
  -- We would prove this in a detailed proof which is omitted here.
  sorry

end time_to_cross_first_platform_l744_74411


namespace rank_siblings_l744_74490

variable (Person : Type) (Dan Elena Finn : Person)

variable (height : Person → ℝ)

-- Conditions
axiom different_heights : height Dan ≠ height Elena ∧ height Elena ≠ height Finn ∧ height Finn ≠ height Dan
axiom one_true_statement : (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn)) 
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))

theorem rank_siblings : height Finn > height Elena ∧ height Elena > height Dan := by
  sorry

end rank_siblings_l744_74490


namespace find_n_l744_74429

theorem find_n (k : ℤ) : 
  ∃ n : ℤ, (n = 35 * k + 24) ∧ (5 ∣ (3 * n - 2)) ∧ (7 ∣ (2 * n + 1)) :=
by
  -- Proof goes here
  sorry

end find_n_l744_74429


namespace football_match_goals_even_likely_l744_74422

noncomputable def probability_even_goals (p_1 : ℝ) (q_1 : ℝ) : Prop :=
  let p := p_1^2 + q_1^2
  let q := 2 * p_1 * q_1
  p >= q

theorem football_match_goals_even_likely (p_1 : ℝ) (h : p_1 >= 0 ∧ p_1 <= 1) : probability_even_goals p_1 (1 - p_1) :=
by sorry

end football_match_goals_even_likely_l744_74422


namespace quadrilateral_perimeter_l744_74447

theorem quadrilateral_perimeter
  (EF FG HG : ℝ)
  (h1 : EF = 7)
  (h2 : FG = 15)
  (h3 : HG = 3)
  (perp1 : EF * FG = 0)
  (perp2 : HG * FG = 0) :
  EF + FG + HG + Real.sqrt (4^2 + 15^2) = 25 + Real.sqrt 241 :=
by
  sorry

end quadrilateral_perimeter_l744_74447


namespace simplify_and_evaluate_l744_74432

noncomputable def expression (x : ℤ) : ℤ :=
  ( (-2 * x^3 - 6 * x) / (-2 * x) - 2 * (3 * x + 1) * (3 * x - 1) + 7 * x * (x - 1) )

theorem simplify_and_evaluate : 
  (expression (-3) = -64) := by
  sorry

end simplify_and_evaluate_l744_74432


namespace Alyssa_spent_on_marbles_l744_74442

def total_spent_on_toys : ℝ := 12.30
def cost_of_football : ℝ := 5.71
def amount_spent_on_marbles : ℝ := 12.30 - 5.71

theorem Alyssa_spent_on_marbles :
  total_spent_on_toys - cost_of_football = amount_spent_on_marbles :=
by
  sorry

end Alyssa_spent_on_marbles_l744_74442


namespace grandfather_age_l744_74455

variable (F S G : ℕ)

theorem grandfather_age (h1 : F = 58) (h2 : F - S = S) (h3 : S - 5 = (1 / 2) * G) : G = 48 := by
  sorry

end grandfather_age_l744_74455


namespace quadratic_roots_eq1_quadratic_roots_eq2_l744_74428

theorem quadratic_roots_eq1 :
  ∀ x : ℝ, (x^2 + 3 * x - 1 = 0) ↔ (x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) :=
by
  intros x
  sorry

theorem quadratic_roots_eq2 :
  ∀ x : ℝ, ((x + 2)^2 = (x + 2)) ↔ (x = -2 ∨ x = -1) :=
by
  intros x
  sorry

end quadratic_roots_eq1_quadratic_roots_eq2_l744_74428


namespace diamond_value_l744_74493

def diamond (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem diamond_value : diamond 7 3 = 22 :=
by
  -- Proof skipped
  sorry

end diamond_value_l744_74493


namespace xinjiang_arable_land_increase_reason_l744_74461

theorem xinjiang_arable_land_increase_reason
  (global_climate_warm: Prop)
  (annual_rainfall_increase: Prop)
  (reserve_arable_land_development: Prop)
  (national_land_policies_adjustment: Prop)
  (arable_land_increased: Prop) :
  (arable_land_increased → reserve_arable_land_development) :=
sorry

end xinjiang_arable_land_increase_reason_l744_74461


namespace price_of_each_lemon_square_l744_74450

-- Given
def brownies_sold : Nat := 4
def price_per_brownie : Nat := 3
def lemon_squares_sold : Nat := 5
def goal_amount : Nat := 50
def cookies_sold : Nat := 7
def price_per_cookie : Nat := 4

-- Prove
theorem price_of_each_lemon_square :
  (brownies_sold * price_per_brownie + lemon_squares_sold * L + cookies_sold * price_per_cookie = goal_amount) →
  L = 2 :=
by
  sorry

end price_of_each_lemon_square_l744_74450


namespace max_oranges_donated_l744_74452

theorem max_oranges_donated (N : ℕ) : ∃ n : ℕ, n < 7 ∧ (N % 7 = n) ∧ n = 6 :=
by
  sorry

end max_oranges_donated_l744_74452


namespace div_fact_l744_74427

-- Conditions
def fact_10 : ℕ := 3628800
def fact_4 : ℕ := 4 * 3 * 2 * 1

-- Question and Correct Answer
theorem div_fact (h : fact_10 = 3628800) : fact_10 / fact_4 = 151200 :=
by
  sorry

end div_fact_l744_74427


namespace min_xy_min_a_b_l744_74419

-- Problem 1 Lean Statement
theorem min_xy {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 1 / (4 * y) = 1) : xy ≥ 2 := sorry

-- Problem 2 Lean Statement
theorem min_a_b {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : ab = a + 2 * b + 4) : a + b ≥ 3 + 2 * Real.sqrt 6 := sorry

end min_xy_min_a_b_l744_74419


namespace find_m_l744_74480

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}

def C_UA : Set ℕ := {1, 2}

theorem find_m (m : ℝ) (hA : A m = {0, 3}) (hCUA : U \ A m = C_UA) : m = -3 := 
  sorry

end find_m_l744_74480


namespace betty_cookies_and_brownies_difference_l744_74479

-- Definitions based on the conditions
def initial_cookies : ℕ := 60
def initial_brownies : ℕ := 10
def cookies_per_day : ℕ := 3
def brownies_per_day : ℕ := 1
def days : ℕ := 7

-- The proof statement
theorem betty_cookies_and_brownies_difference :
  initial_cookies - (cookies_per_day * days) - (initial_brownies - (brownies_per_day * days)) = 36 :=
by
  sorry

end betty_cookies_and_brownies_difference_l744_74479


namespace extreme_points_sum_gt_l744_74473

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a * x^2 - Real.log x

theorem extreme_points_sum_gt (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1 / 8)
    {x₁ x₂ : ℝ} (h₂ : f x₁ a = 0) (h₃ : f x₂ a = 0) (h₄ : x₁ < x₂)
    (h₅ : 0 < x₁) (h₆ : 0 < x₂) : f x₁ a + f x₂ a > 3 - 2 * Real.log 2 := sorry

end extreme_points_sum_gt_l744_74473


namespace area_and_cost_of_path_l744_74462

-- Define the dimensions of the rectangular grass field
def length_field : ℝ := 75
def width_field : ℝ := 55

-- Define the width of the path around the field
def path_width : ℝ := 2.8

-- Define the cost per square meter for constructing the path
def cost_per_sq_m : ℝ := 2

-- Define the total length and width including the path
def total_length : ℝ := length_field + 2 * path_width
def total_width : ℝ := width_field + 2 * path_width

-- Define the area of the entire field including the path
def area_total : ℝ := total_length * total_width

-- Define the area of the grass field alone
def area_field : ℝ := length_field * width_field

-- Define the area of the path alone
def area_path : ℝ := area_total - area_field

-- Define the cost of constructing the path
def cost_path : ℝ := area_path * cost_per_sq_m

-- The statement to be proved
theorem area_and_cost_of_path :
  area_path = 759.36 ∧ cost_path = 1518.72 := by
  sorry

end area_and_cost_of_path_l744_74462


namespace two_people_paint_time_l744_74499

theorem two_people_paint_time (h : 5 * 7 = 35) :
  ∃ t : ℝ, 2 * t = 35 ∧ t = 17.5 := 
sorry

end two_people_paint_time_l744_74499


namespace opposite_number_of_neg_two_reciprocal_of_three_abs_val_three_eq_l744_74425

theorem opposite_number_of_neg_two (a : Int) (h : a = -2) :
  -a = 2 := by
  sorry

theorem reciprocal_of_three (x y : Real) (hx : x = 3) (hy : y = 1 / 3) : 
  x * y = 1 := by
  sorry

theorem abs_val_three_eq (x : Real) (hx : abs x = 3) :
  x = -3 ∨ x = 3 := by
  sorry

end opposite_number_of_neg_two_reciprocal_of_three_abs_val_three_eq_l744_74425


namespace exists_y_square_divisible_by_five_btw_50_and_120_l744_74453

theorem exists_y_square_divisible_by_five_btw_50_and_120 : ∃ y : ℕ, (∃ k : ℕ, y = k^2) ∧ (y % 5 = 0) ∧ (50 ≤ y ∧ y ≤ 120) ∧ y = 100 :=
by
  sorry

end exists_y_square_divisible_by_five_btw_50_and_120_l744_74453


namespace correct_choice_C_l744_74424

def geometric_sequence (n : ℕ) : ℕ := 
  2^(n - 1)

def sum_geometric_sequence (n : ℕ) : ℕ := 
  2^n - 1

theorem correct_choice_C (n : ℕ) (h : 0 < n) : sum_geometric_sequence n < geometric_sequence (n + 1) := by
  sorry

end correct_choice_C_l744_74424


namespace fish_lifespan_proof_l744_74418

def hamster_lifespan : ℝ := 2.5

def dog_lifespan : ℝ := 4 * hamster_lifespan

def fish_lifespan : ℝ := dog_lifespan + 2

theorem fish_lifespan_proof :
  fish_lifespan = 12 := 
  by
  sorry

end fish_lifespan_proof_l744_74418
