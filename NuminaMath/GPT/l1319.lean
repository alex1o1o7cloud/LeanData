import Mathlib

namespace problem_I_problem_II_l1319_131939

-- Problem (I): Proving the inequality solution set
theorem problem_I (x : ℝ) : |x - 5| + |x + 6| ≤ 12 ↔ -13/2 ≤ x ∧ x ≤ 11/2 :=
by
  sorry

-- Problem (II): Proving the range of m
theorem problem_II (m : ℝ) : (∀ x : ℝ, |x - m| + |x + 6| ≥ 7) ↔ (m ≤ -13 ∨ m ≥ 1) :=
by
  sorry

end problem_I_problem_II_l1319_131939


namespace part1_store_a_cost_part1_store_b_cost_part2_cost_comparison_part3_cost_effective_plan_l1319_131933

-- Defining the conditions
def racket_price : ℕ := 50
def ball_price : ℕ := 20
def num_rackets : ℕ := 10

-- Store A cost function
def store_A_cost (x : ℕ) : ℕ := 20 * x + 300

-- Store B cost function
def store_B_cost (x : ℕ) : ℕ := 16 * x + 400

-- Part (1): Express the costs in algebraic form
theorem part1_store_a_cost (x : ℕ) (hx : 10 < x) : store_A_cost x = 20 * x + 300 := by
  sorry

theorem part1_store_b_cost (x : ℕ) (hx : 10 < x) : store_B_cost x = 16 * x + 400 := by
  sorry

-- Part (2): Cost for x = 40
theorem part2_cost_comparison : store_A_cost 40 > store_B_cost 40 := by
  sorry

-- Part (3): Most cost-effective purchasing plan
def store_a_cost_rackets : ℕ := racket_price * num_rackets
def store_a_free_balls : ℕ := num_rackets
def remaining_balls (total_balls : ℕ) : ℕ := total_balls - store_a_free_balls
def store_b_cost_remaining_balls (remaining_balls : ℕ) : ℕ := remaining_balls * ball_price * 4 / 5

theorem part3_cost_effective_plan : store_a_cost_rackets + store_b_cost_remaining_balls (remaining_balls 40) = 980 := by
  sorry

end part1_store_a_cost_part1_store_b_cost_part2_cost_comparison_part3_cost_effective_plan_l1319_131933


namespace maximum_area_of_triangle_ABC_l1319_131926

noncomputable def max_area_triangle_ABC (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem maximum_area_of_triangle_ABC (a b c A B C : ℝ) 
  (h1: a = 4) 
  (h2: (4 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) :
  max_area_triangle_ABC a b c A B C = 4 * Real.sqrt 3 := 
sorry

end maximum_area_of_triangle_ABC_l1319_131926


namespace intersection_is_A_l1319_131911

-- Define the set M based on the given condition
def M : Set ℝ := {x | x / (x - 1) ≥ 0}

-- Define the set N based on the given condition
def N : Set ℝ := {x | ∃ y, y = 3 * x^2 + 1}

-- Define the set A as the intersection of M and N
def A : Set ℝ := {x | x > 1}

-- Prove that the intersection of M and N is equal to the set A
theorem intersection_is_A : (M ∩ N = A) :=
by {
  sorry
}

end intersection_is_A_l1319_131911


namespace alice_paid_percentage_l1319_131930

theorem alice_paid_percentage {P : ℝ} (hP : P > 0)
  (hMP : ∀ P, MP = 0.60 * P)
  (hPrice_Alice_Paid : ∀ MP, Price_Alice_Paid = 0.40 * MP) :
  (Price_Alice_Paid / P) * 100 = 24 := by
  sorry

end alice_paid_percentage_l1319_131930


namespace cost_of_four_dozen_bananas_l1319_131940

/-- Given that five dozen bananas cost $24.00,
    prove that the cost for four dozen bananas is $19.20. -/
theorem cost_of_four_dozen_bananas 
  (cost_five_dozen: ℝ)
  (rate: cost_five_dozen = 24) : 
  ∃ (cost_four_dozen: ℝ), cost_four_dozen = 19.2 := by
  sorry

end cost_of_four_dozen_bananas_l1319_131940


namespace unique_triple_l1319_131918

theorem unique_triple (x y p : ℕ) (hx : 0 < x) (hy : 0 < y) (hp : Nat.Prime p) (h1 : p = x^2 + 1) (h2 : 2 * p^2 = y^2 + 1) :
  (x, y, p) = (2, 7, 5) :=
sorry

end unique_triple_l1319_131918


namespace math_problem_l1319_131977

noncomputable def a : ℕ := 1265
noncomputable def b : ℕ := 168
noncomputable def c : ℕ := 21
noncomputable def d : ℕ := 6
noncomputable def e : ℕ := 3

theorem math_problem : 
  ( ( b / 100 : ℚ ) * (a ^ 2 / c) / (d - e ^ 2) : ℚ ) = -42646.27 :=
by sorry

end math_problem_l1319_131977


namespace ceil_sum_sqrt_eval_l1319_131960

theorem ceil_sum_sqrt_eval : 
  (⌈Real.sqrt 2⌉ + ⌈Real.sqrt 22⌉ + ⌈Real.sqrt 222⌉) = 22 := 
by
  sorry

end ceil_sum_sqrt_eval_l1319_131960


namespace ashok_total_subjects_l1319_131902

/-- Ashok secured an average of 78 marks in some subjects. If the average of marks in 5 subjects 
is 74, and he secured 98 marks in the last subject, how many subjects are there in total? -/
theorem ashok_total_subjects (n : ℕ) 
  (avg_all : 78 * n = 74 * (n - 1) + 98) : n = 6 :=
sorry

end ashok_total_subjects_l1319_131902


namespace find_x_l1319_131951

-- Definitions corresponding to conditions a)
def rectangle (AB CD BC AD x : ℝ) := AB = 2 ∧ CD = 2 ∧ BC = 1 ∧ AD = 1 ∧ x = 0

-- Define the main statement to be proven
theorem find_x (AB CD BC AD x k m: ℝ) (h: rectangle AB CD BC AD x) : 
  x = (0 : ℝ) ∧ k = 0 ∧ m = 0 ∧ x = (Real.sqrt k - m) ∧ k + m = 0 :=
by
  cases h
  sorry

end find_x_l1319_131951


namespace batting_average_drop_l1319_131995

theorem batting_average_drop 
    (avg : ℕ)
    (innings : ℕ)
    (high : ℕ)
    (high_low_diff : ℕ)
    (low : ℕ)
    (total_runs : ℕ)
    (new_avg : ℕ)

    (h1 : avg = 50)
    (h2 : innings = 40)
    (h3 : high = 174)
    (h4 : high = low + 172)
    (h5 : total_runs = avg * innings)
    (h6 : new_avg = (total_runs - high - low) / (innings - 2)) :

  avg - new_avg = 2 :=
by
  sorry

end batting_average_drop_l1319_131995


namespace solve_inequality_l1319_131969

def satisfies_inequality (x : ℝ) : Prop :=
  (3 * x - 4) * (x + 1) / x ≥ 0

theorem solve_inequality :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | -1 ≤ x ∧ x < 0 ∨ x ≥ 4 / 3} :=
by
  sorry

end solve_inequality_l1319_131969


namespace necessary_but_not_sufficient_l1319_131916

theorem necessary_but_not_sufficient (p q : Prop) : 
  (p ∨ q) → (p ∧ q) → False :=
by
  sorry

end necessary_but_not_sufficient_l1319_131916


namespace part_I_part_II_l1319_131964

def f (x a : ℝ) : ℝ := |2 * x + 1| + |2 * x - a| + a

theorem part_I (x : ℝ) (h₁ : f x 3 > 7) : sorry := sorry

theorem part_II (a : ℝ) (h₂ : ∀ (x : ℝ), f x a ≥ 3) : sorry := sorry

end part_I_part_II_l1319_131964


namespace team_A_more_points_than_team_B_l1319_131934

theorem team_A_more_points_than_team_B :
  let number_of_teams := 8
  let number_of_remaining_games := 6
  let win_probability_each_game := (1 : ℚ) / 2
  let team_A_beats_team_B_initial : Prop := True -- Corresponding to the condition team A wins the first game
  let probability_A_wins := 1087 / 2048
  team_A_beats_team_B_initial → win_probability_each_game = 1 / 2 → number_of_teams = 8 → 
    let A_more_points_than_B := team_A_beats_team_B_initial ∧ win_probability_each_game ^ number_of_remaining_games = probability_A_wins
    A_more_points_than_B :=
  sorry

end team_A_more_points_than_team_B_l1319_131934


namespace number_of_men_in_club_l1319_131935

variables (M W : ℕ)

theorem number_of_men_in_club 
  (h1 : M + W = 30) 
  (h2 : (1 / 3 : ℝ) * W + M = 18) : 
  M = 12 := 
sorry

end number_of_men_in_club_l1319_131935


namespace largest_repeating_number_l1319_131925

theorem largest_repeating_number :
  ∃ n, n * 365 = 273863 * 365 := sorry

end largest_repeating_number_l1319_131925


namespace intersection_of_M_and_N_l1319_131998

noncomputable def M : Set ℕ := { x | 1 < x ∧ x < 7 }
noncomputable def N : Set ℕ := { x | x % 3 ≠ 0 }

theorem intersection_of_M_and_N :
  M ∩ N = {2, 4, 5} := sorry

end intersection_of_M_and_N_l1319_131998


namespace arithmetic_sequence_sum_l1319_131997

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ) (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 := 
sorry

end arithmetic_sequence_sum_l1319_131997


namespace johns_payment_l1319_131906

-- Define the value of the camera
def camera_value : ℕ := 5000

-- Define the rental fee rate per week as a percentage
def rental_fee_rate : ℝ := 0.1

-- Define the rental period in weeks
def rental_period : ℕ := 4

-- Define the friend's contribution rate as a percentage
def friend_contribution_rate : ℝ := 0.4

-- Theorem: Calculate how much John pays for the camera rental
theorem johns_payment :
  let weekly_rental_fee := camera_value * rental_fee_rate
  let total_rental_fee := weekly_rental_fee * rental_period
  let friends_contribution := total_rental_fee * friend_contribution_rate
  let johns_payment := total_rental_fee - friends_contribution
  johns_payment = 1200 :=
by
  sorry

end johns_payment_l1319_131906


namespace solve_for_y_l1319_131988

noncomputable def determinant3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

noncomputable def determinant2x2 (a b c d : ℝ) : ℝ := 
  a*d - b*c

theorem solve_for_y (b y : ℝ) (h : b ≠ 0) :
  determinant3x3 (y + 2 * b) y y y (y + 2 * b) y y y (y + 2 * b) = 0 → 
  y = -b / 2 :=
by
  sorry

end solve_for_y_l1319_131988


namespace animal_shelter_dogs_l1319_131993

theorem animal_shelter_dogs (D C R : ℕ) 
  (h₁ : 15 * C = 7 * D)
  (h₂ : 15 * R = 4 * D)
  (h₃ : 15 * (C + 20) = 11 * D)
  (h₄ : 15 * (R + 10) = 6 * D) : 
  D = 75 :=
by
  -- Proof part is omitted
  sorry

end animal_shelter_dogs_l1319_131993


namespace find_d_l1319_131966

theorem find_d (d : ℤ) :
  (∀ x : ℤ, (4 * x^3 + 13 * x^2 + d * x + 18 = 0 ↔ x = -3)) →
  d = 9 :=
by
  sorry

end find_d_l1319_131966


namespace scientific_notation_correct_l1319_131983

/-- Given the weight of the "人" shaped gate of the Three Gorges ship lock -/
def weight_kg : ℝ := 867000

/-- The scientific notation representation of the given weight -/
def scientific_notation_weight_kg : ℝ := 8.67 * 10^5

theorem scientific_notation_correct :
  weight_kg = scientific_notation_weight_kg :=
sorry

end scientific_notation_correct_l1319_131983


namespace tangents_equal_l1319_131913

theorem tangents_equal (α β γ : ℝ) (h1 : Real.sin α + Real.sin β + Real.sin γ = 0) (h2 : Real.cos α + Real.cos β + Real.cos γ = 0) :
  Real.tan (3 * α) = Real.tan (3 * β) ∧ Real.tan (3 * β) = Real.tan (3 * γ) := 
sorry

end tangents_equal_l1319_131913


namespace marathon_fraction_l1319_131989

theorem marathon_fraction :
  ∃ (f : ℚ), (2 * 7) = (6 + (6 + 6 * f)) ∧ f = 1 / 3 :=
by 
  sorry

end marathon_fraction_l1319_131989


namespace decimal_equivalent_l1319_131971

theorem decimal_equivalent (x : ℚ) (h : x = 16 / 50) : x = 32 / 100 :=
by
  sorry

end decimal_equivalent_l1319_131971


namespace cube_and_fourth_power_remainders_l1319_131909

theorem cube_and_fourth_power_remainders (
  b : Fin 2018 → ℕ) 
  (h1 : StrictMono b) 
  (h2 : (Finset.univ.sum b) = 2018^3) :
  ((Finset.univ.sum (λ i => b i ^ 3)) % 5 = 3) ∧
  ((Finset.univ.sum (λ i => b i ^ 4)) % 5 = 1) := 
sorry

end cube_and_fourth_power_remainders_l1319_131909


namespace daffodil_stamps_count_l1319_131994

theorem daffodil_stamps_count (r d : ℕ) (h1 : r = 2) (h2 : r = d) : d = 2 := by
  sorry

end daffodil_stamps_count_l1319_131994


namespace college_students_freshmen_psych_majors_l1319_131948

variable (T : ℕ)
variable (hT : T > 0)

def freshmen (T : ℕ) : ℕ := 40 * T / 100
def lib_arts (F : ℕ) : ℕ := 50 * F / 100
def psych_majors (L : ℕ) : ℕ := 50 * L / 100
def percent_freshmen_psych_majors (P : ℕ) (T : ℕ) : ℕ := 100 * P / T

theorem college_students_freshmen_psych_majors :
  percent_freshmen_psych_majors (psych_majors (lib_arts (freshmen T))) T = 10 := by
  sorry

end college_students_freshmen_psych_majors_l1319_131948


namespace max_license_plates_l1319_131970

noncomputable def max_distinct_plates (m n : ℕ) : ℕ :=
  m ^ (n - 1)

theorem max_license_plates :
  max_distinct_plates 10 6 = 100000 := by
  sorry

end max_license_plates_l1319_131970


namespace percentage_greater_than_88_l1319_131944

theorem percentage_greater_than_88 (x : ℝ) (percentage : ℝ) (h1 : x = 110) (h2 : x = 88 + (percentage * 88)) : percentage = 0.25 :=
by
  sorry

end percentage_greater_than_88_l1319_131944


namespace c_S_power_of_2_l1319_131985

variables (m : ℕ) (S : String)

-- condition: m > 1
def is_valid_m (m : ℕ) : Prop := m > 1

-- function c(S)
def c (S : String) : ℕ := sorry  -- actual implementation is skipped

-- function to check if a number represented by a string is divisible by m
def is_divisible_by (n m : ℕ) : Prop := n % m = 0

-- Property that c(S) can take only powers of 2
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem c_S_power_of_2 (m : ℕ) (S : String) (h1 : is_valid_m m) :
  is_power_of_two (c S) :=
sorry

end c_S_power_of_2_l1319_131985


namespace perimeter_of_rectangle_l1319_131975

theorem perimeter_of_rectangle (area : ℝ) (num_squares : ℕ) (square_side : ℝ) (width : ℝ) (height : ℝ) 
  (h1 : area = 216) (h2 : num_squares = 6) (h3 : area / num_squares = square_side^2)
  (h4 : width = 3 * square_side) (h5 : height = 2 * square_side) : 
  2 * (width + height) = 60 :=
by
  sorry

end perimeter_of_rectangle_l1319_131975


namespace optimal_bicycle_point_l1319_131973

noncomputable def distance_A_B : ℝ := 30  -- Distance between A and B is 30 km
noncomputable def midpoint_distance : ℝ := distance_A_B / 2  -- Distance between midpoint C to both A and B is 15 km
noncomputable def walking_speed : ℝ := 5  -- Walking speed is 5 km/h
noncomputable def biking_speed : ℝ := 20  -- Biking speed is 20 km/h

theorem optimal_bicycle_point : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ (30 - x + 4 * x = 60 - 3 * x) → x = 5 :=
by sorry

end optimal_bicycle_point_l1319_131973


namespace problem_l1319_131978

noncomputable def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
noncomputable def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

variable (f g : ℝ → ℝ)
variable (h₁ : is_odd f)
variable (h₂ : is_even g)
variable (h₃ : ∀ x, f x - g x = 2 * x^3 + x^2 + 3)

theorem problem : f 2 + g 2 = 9 :=
by sorry

end problem_l1319_131978


namespace slope_of_line_l1319_131928

theorem slope_of_line (x y : ℝ) : 
  3 * y + 9 = -6 * x - 15 → 
  ∃ m b, y = m * x + b ∧ m = -2 := 
by {
  sorry
}

end slope_of_line_l1319_131928


namespace fraction_of_males_l1319_131904

theorem fraction_of_males (M F : ℝ) 
  (h1 : M + F = 1)
  (h2 : (7 / 8) * M + (4 / 5) * F = 0.845) :
  M = 0.6 :=
by
  sorry

end fraction_of_males_l1319_131904


namespace triangle_area_l1319_131999

theorem triangle_area (base height : ℕ) (h_base : base = 35) (h_height : height = 12) :
  (1 / 2 : ℚ) * base * height = 210 := by
  sorry

end triangle_area_l1319_131999


namespace curve_three_lines_intersect_at_origin_l1319_131958

theorem curve_three_lines_intersect_at_origin (a : ℝ) :
  ((∀ x y : ℝ, (x + 2 * y + a) * (x^2 - y^2) = 0 → 
    ((y = x ∨ y = -x ∨ y = - (1/2) * x - a/2) ∧ 
     (x = 0 ∧ y = 0)))) ↔ a = 0 :=
sorry

end curve_three_lines_intersect_at_origin_l1319_131958


namespace glasses_needed_l1319_131980

theorem glasses_needed (total_juice : ℕ) (juice_per_glass : ℕ) : Prop :=
  total_juice = 153 ∧ juice_per_glass = 30 → (total_juice + juice_per_glass - 1) / juice_per_glass = 6

-- This will state our theorem but we include sorry to omit the proof.

end glasses_needed_l1319_131980


namespace A_finishes_job_in_12_days_l1319_131908

variable (A B : ℝ)

noncomputable def work_rate_A_and_B := (1 / 40)
noncomputable def work_rate_A := (1 / A)
noncomputable def work_rate_B := (1 / B)

theorem A_finishes_job_in_12_days
  (h1 : work_rate_A + work_rate_B = work_rate_A_and_B)
  (h2 : 10 * work_rate_A_and_B = 1 / 4)
  (h3 : 9 * work_rate_A = 3 / 4) :
  A = 12 :=
  sorry

end A_finishes_job_in_12_days_l1319_131908


namespace f_g_minus_g_f_l1319_131910

-- Defining the functions f and g
def f (x : ℝ) : ℝ := x^2 + 3
def g (x : ℝ) : ℝ := 3 * x^2 + 5

-- Proving the given math problem
theorem f_g_minus_g_f :
  f (g 2) - g (f 2) = 140 := by
sorry

end f_g_minus_g_f_l1319_131910


namespace triangle_area_l1319_131955

theorem triangle_area (P : ℝ × ℝ)
  (Q : ℝ × ℝ) (R : ℝ × ℝ)
  (P_eq : P = (3, 2))
  (Q_eq : ∃ b, Q = (7/3, 0) ∧ 2 = 3 * 3 + b ∧ 0 = 3 * (7/3) + b)
  (R_eq : ∃ b, R = (4, 0) ∧ 2 = -2 * 3 + b ∧ 0 = -2 * 4 + b) :
  (1/2) * abs (Q.1 - R.1) * abs (P.2) = 5/3 :=
by
  sorry

end triangle_area_l1319_131955


namespace cos_double_angle_l1319_131912

theorem cos_double_angle (α : ℝ) (h : Real.sin (π + α) = 2 / 3) : Real.cos (2 * α) = 1 / 9 := 
by
  sorry

end cos_double_angle_l1319_131912


namespace gcd_pow_minus_one_l1319_131996

theorem gcd_pow_minus_one {m n : ℕ} (hm : 0 < m) (hn : 0 < n) :
  Nat.gcd (2^m - 1) (2^n - 1) = 2^Nat.gcd m n - 1 :=
sorry

end gcd_pow_minus_one_l1319_131996


namespace problem_remainder_6_pow_83_add_8_pow_83_mod_49_l1319_131945

-- Definitions based on the conditions.
def euler_totient_49 : ℕ := 42

theorem problem_remainder_6_pow_83_add_8_pow_83_mod_49 
  (h1 : 6 ^ euler_totient_49 ≡ 1 [MOD 49])
  (h2 : 8 ^ euler_totient_49 ≡ 1 [MOD 49]) :
  (6 ^ 83 + 8 ^ 83) % 49 = 35 :=
by
  sorry

end problem_remainder_6_pow_83_add_8_pow_83_mod_49_l1319_131945


namespace toaster_total_cost_l1319_131929

theorem toaster_total_cost :
  let MSRP := 30
  let insurance_rate := 0.20
  let premium_upgrade := 7
  let recycling_fee := 5
  let tax_rate := 0.50

  -- Calculate costs
  let insurance_cost := insurance_rate * MSRP
  let total_insurance_cost := insurance_cost + premium_upgrade
  let cost_before_tax := MSRP + total_insurance_cost + recycling_fee
  let state_tax := tax_rate * cost_before_tax
  let total_cost := cost_before_tax + state_tax

  -- Total cost Jon must pay
  total_cost = 72 :=
by
  sorry

end toaster_total_cost_l1319_131929


namespace proof_problem_l1319_131947

-- Define the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℕ := (n^2 + n) / 2

-- Define the arithmetic sequence a_n based on S_n
def a (n : ℕ) : ℕ := if n = 1 then 1 else S n - S (n - 1)

-- Define the geometric sequence b_n with initial conditions
def b (n : ℕ) : ℕ :=
  if n = 1 then a 1 + 1
  else if n = 2 then a 2 + 2
  else 2^n

-- Define the sum of the first n terms of the geometric sequence b_n
def T (n : ℕ) : ℕ := 2 * (2^n - 1)

-- Main theorem to prove
theorem proof_problem :
  (∀ n, a n = n) ∧
  (∀ n, n ≥ 1 → b n = 2^n) ∧
  (∃ n, T n + a n > 300 ∧ ∀ m < n, T m + a m ≤ 300) :=
by {
  sorry
}

end proof_problem_l1319_131947


namespace find_numbers_l1319_131923

theorem find_numbers (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (geom_mean_cond : Real.sqrt (a * b) = Real.sqrt 5)
  (harm_mean_cond : 2 / ((1 / a) + (1 / b)) = 2) :
  (a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2) ∨
  (a = (5 - Real.sqrt 5) / 2 ∧ b = (5 + Real.sqrt 5) / 2) :=
by
  sorry

end find_numbers_l1319_131923


namespace determine_x_l1319_131976

theorem determine_x (x y : ℤ) (h1 : x + 2 * y = 20) (h2 : y = 5) : x = 10 := 
by 
  sorry

end determine_x_l1319_131976


namespace gcd_5280_12155_l1319_131953

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end gcd_5280_12155_l1319_131953


namespace age_difference_l1319_131990

variables (X Y Z : ℕ)

theorem age_difference (h : X + Y = Y + Z + 12) : X - Z = 12 :=
sorry

end age_difference_l1319_131990


namespace GODOT_value_l1319_131920

theorem GODOT_value (G O D I T : ℕ) (h1 : G ≠ 0) (h2 : D ≠ 0) 
  (eq1 : 1000 * G + 100 * O + 10 * G + O + 1000 * D + 100 * I + 10 * D + I = 10000 * G + 1000 * O + 100 * D + 10 * O + T) : 
  10000 * G + 1000 * O + 100 * D + 10 * O + T = 10908 :=
by {
  sorry
}

end GODOT_value_l1319_131920


namespace trigonometric_identity_proof_l1319_131915

theorem trigonometric_identity_proof 
  (α β γ : ℝ) (a b c : ℝ)
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < β ∧ β < π)
  (h3 : 0 < γ ∧ γ < π)
  (hc : 0 < c)
  (hb : b = (c * (Real.cos α + Real.cos β * Real.cos γ)) / (Real.sin γ)^2)
  (ha : a = (c * (Real.cos β + Real.cos α * Real.cos γ)) / (Real.sin γ)^2) :
  1 - (Real.cos α)^2 - (Real.cos β)^2 - (Real.cos γ)^2 - 2 * Real.cos α * Real.cos β * Real.cos γ = 0 :=
by
  sorry

end trigonometric_identity_proof_l1319_131915


namespace nearest_integer_to_expansion_l1319_131961

theorem nearest_integer_to_expansion : 
  let a := (3 + 2 * Real.sqrt 2)
  let b := (3 - 2 * Real.sqrt 2)
  abs (a^4 - 1090) < 1 :=
by
  let a := (3 + 2 * Real.sqrt 2)
  let b := (3 - 2 * Real.sqrt 2)
  sorry

end nearest_integer_to_expansion_l1319_131961


namespace sin_identity_l1319_131986

theorem sin_identity (α : ℝ) (h : Real.sin (π * α) = 4 / 5) : 
  Real.sin (π / 2 + 2 * α) = -24 / 25 :=
by
  sorry

end sin_identity_l1319_131986


namespace find_age_l1319_131991

variable (x : ℤ)

def age_4_years_hence := x + 4
def age_4_years_ago := x - 4
def brothers_age := x - 6

theorem find_age (hx : x = 4 * (x + 4) - 4 * (x - 4) + 1/2 * (x - 6)) : x = 58 :=
sorry

end find_age_l1319_131991


namespace liquid_mixture_ratio_l1319_131965

theorem liquid_mixture_ratio (m1 m2 m3 : ℝ) (ρ1 ρ2 ρ3 : ℝ) (k : ℝ)
  (hρ1 : ρ1 = 6 * k) (hρ2 : ρ2 = 3 * k) (hρ3 : ρ3 = 2 * k)
  (h_condition : m1 ≥ 3.5 * m2)
  (h_arith_mean : (m1 + m2 + m3) / (m1 / ρ1 + m2 / ρ2 + m3 / ρ3) = (ρ1 + ρ2 + ρ3) / 3) :
    ∃ x y : ℝ, x ≤ 2/7 ∧ (4 * x + 15 * y = 7) := sorry

end liquid_mixture_ratio_l1319_131965


namespace cats_to_dogs_ratio_l1319_131937

noncomputable def num_dogs : ℕ := 18
noncomputable def num_cats : ℕ := num_dogs - 6
noncomputable def ratio (a b : ℕ) : ℚ := a / b

theorem cats_to_dogs_ratio (h1 : num_dogs = 18) (h2 : num_cats = num_dogs - 6) : ratio num_cats num_dogs = 2 / 3 :=
by
  sorry

end cats_to_dogs_ratio_l1319_131937


namespace number_of_chickens_l1319_131967

variables (C G Ch : ℕ)

theorem number_of_chickens (h1 : C = 9) (h2 : G = 4 * C) (h3 : G = 2 * Ch) : Ch = 18 :=
by
  sorry

end number_of_chickens_l1319_131967


namespace marbles_steve_now_l1319_131946
-- Import necessary libraries

-- Define the initial conditions as given in a)
def initial_conditions (sam steve sally : ℕ) := sam = 2 * steve ∧ sally = sam - 5 ∧ sam - 6 = 8

-- Define the proof problem statement
theorem marbles_steve_now (sam steve sally : ℕ) (h : initial_conditions sam steve sally) : steve + 3 = 10 :=
sorry

end marbles_steve_now_l1319_131946


namespace square_of_1023_l1319_131914

def square_1023_eq_1046529 : Prop :=
  let x := 1023
  x * x = 1046529

theorem square_of_1023 : square_1023_eq_1046529 :=
by
  sorry

end square_of_1023_l1319_131914


namespace hyperbola_foci_l1319_131979

theorem hyperbola_foci :
  (∀ x y : ℝ, x^2 - 2 * y^2 = 1) →
  (∃ c : ℝ, c = (Real.sqrt 6) / 2 ∧ (x = c ∨ x = -c) ∧ y = 0) :=
by
  sorry

end hyperbola_foci_l1319_131979


namespace power_multiplication_same_base_l1319_131962

theorem power_multiplication_same_base :
  (10 ^ 655 * 10 ^ 650 = 10 ^ 1305) :=
by {
  sorry
}

end power_multiplication_same_base_l1319_131962


namespace number_of_valid_m_values_l1319_131931

/--
In the coordinate plane, construct a right triangle with its legs parallel to the x and y axes, and with the medians on its legs lying on the lines y = 3x + 1 and y = mx + 2. 
Prove that the number of values for the constant m such that this triangle exists is 2.
-/
theorem number_of_valid_m_values : 
  ∃ (m : ℝ), 
    (∃ (a b : ℝ), 
      (∀ D E : ℝ × ℝ, D = (a / 2, 0) ∧ E = (0, b / 2) →
      D.2 = 3 * D.1 + 1 ∧ 
      E.2 = m * E.1 + 2)) → 
    (number_of_solutions_for_m = 2) 
  :=
sorry

end number_of_valid_m_values_l1319_131931


namespace trueConverseB_l1319_131900

noncomputable def conditionA : Prop :=
  ∀ (x y : ℝ), -- "Vertical angles are equal"
  sorry -- Placeholder for vertical angles equality

noncomputable def conditionB : Prop :=
  ∀ (l₁ l₂ : ℝ), -- "If the consecutive interior angles are supplementary, then the two lines are parallel."
  sorry -- Placeholder for supplementary angles imply parallel lines

noncomputable def conditionC : Prop :=
  ∀ (a b : ℝ), -- "If \(a = b\), then \(a^2 = b^2\)"
  a = b → a^2 = b^2

noncomputable def conditionD : Prop :=
  ∀ (a b : ℝ), -- "If \(a > 0\) and \(b > 0\), then \(a^2 + b^2 > 0\)"
  a > 0 ∧ b > 0 → a^2 + b^2 > 0

theorem trueConverseB (hB: conditionB) : -- Proposition (B) has a true converse
  ∀ (l₁ l₂ : ℝ), 
  (∃ (a1 a2 : ℝ), -- Placeholder for angles
  sorry) → (l₁ = l₂) := -- Placeholder for consecutive interior angles are supplementary
  sorry

end trueConverseB_l1319_131900


namespace find_a_odd_function_l1319_131954

noncomputable def f (a x : ℝ) := Real.log (Real.sqrt (x^2 + 1) - a * x)

theorem find_a_odd_function :
  ∀ a : ℝ, (∀ x : ℝ, f a (-x) + f a x = 0) ↔ (a = 1 ∨ a = -1) := by
  sorry

end find_a_odd_function_l1319_131954


namespace last_digit_base_4_of_77_l1319_131950

theorem last_digit_base_4_of_77 : (77 % 4) = 1 :=
by
  sorry

end last_digit_base_4_of_77_l1319_131950


namespace positive_integer_solution_l1319_131919

theorem positive_integer_solution (n : ℕ) (h1 : n + 2009 ∣ n^2 + 2009) (h2 : n + 2010 ∣ n^2 + 2010) : n = 1 := 
by
  -- The proof would go here.
  sorry

end positive_integer_solution_l1319_131919


namespace average_age_of_new_students_l1319_131968

theorem average_age_of_new_students :
  ∀ (initial_group_avg_age new_group_avg_age : ℝ) (initial_students new_students total_students : ℕ),
  initial_group_avg_age = 14 →
  initial_students = 10 →
  new_group_avg_age = 15 →
  new_students = 5 →
  total_students = initial_students + new_students →
  (new_group_avg_age * total_students - initial_group_avg_age * initial_students) / new_students = 17 :=
by
  intros initial_group_avg_age new_group_avg_age initial_students new_students total_students
  sorry

end average_age_of_new_students_l1319_131968


namespace solve_for_exponent_l1319_131936

theorem solve_for_exponent (K : ℕ) (h1 : 32 = 2 ^ 5) (h2 : 64 = 2 ^ 6) 
    (h3 : 32 ^ 5 * 64 ^ 2 = 2 ^ K) : K = 37 := 
by 
    sorry

end solve_for_exponent_l1319_131936


namespace find_xy_l1319_131922

theorem find_xy (x y : ℝ) (h : (x - 13)^2 + (y - 14)^2 + (x - y)^2 = 1/3) : 
  x = 40/3 ∧ y = 41/3 :=
sorry

end find_xy_l1319_131922


namespace find_S_l1319_131905

theorem find_S (R S T : ℝ) (c : ℝ)
  (h1 : R = c * (S / T))
  (h2 : R = 2) (h3 : S = 1/2) (h4 : T = 4/3) (h_c : c = 16/3)
  (h_R : R = Real.sqrt 75) (h_T : T = Real.sqrt 32) :
  S = 45/4 := by
  sorry

end find_S_l1319_131905


namespace abcd_product_l1319_131917

theorem abcd_product :
  let A := (Real.sqrt 3003 + Real.sqrt 3004)
  let B := (-Real.sqrt 3003 - Real.sqrt 3004)
  let C := (Real.sqrt 3003 - Real.sqrt 3004)
  let D := (Real.sqrt 3004 - Real.sqrt 3003)
  A * B * C * D = 1 := 
by
  sorry

end abcd_product_l1319_131917


namespace pair_divisibility_l1319_131927

theorem pair_divisibility (m n : ℕ) : 
  (m * n ∣ m ^ 2019 + n) ↔ ((m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 2 ^ 2019)) := sorry

end pair_divisibility_l1319_131927


namespace gcd_pow_sub_one_l1319_131924

theorem gcd_pow_sub_one (a b : ℕ) 
  (h_a : a = 2^2004 - 1) 
  (h_b : b = 2^1995 - 1) : 
  Int.gcd a b = 511 :=
by
  sorry

end gcd_pow_sub_one_l1319_131924


namespace kiddie_scoop_cost_is_three_l1319_131949

-- Define the parameters for the costs of different scoops and total payment
variable (k : ℕ)  -- cost of kiddie scoop
def cost_regular : ℕ := 4
def cost_double : ℕ := 6
def total_payment : ℕ := 32

-- Conditions: Mr. and Mrs. Martin each get a regular scoop
def regular_cost : ℕ := 2 * cost_regular

-- Their three teenage children each get double scoops
def double_cost : ℕ := 3 * cost_double

-- Total cost of regular and double scoops
def combined_cost : ℕ := regular_cost + double_cost

-- Total payment includes two kiddie scoops
def kiddie_total_cost : ℕ := total_payment - combined_cost

-- The cost of one kiddie scoop
def kiddie_cost : ℕ := kiddie_total_cost / 2

theorem kiddie_scoop_cost_is_three : kiddie_cost = 3 := by
  sorry

end kiddie_scoop_cost_is_three_l1319_131949


namespace line_through_point_with_opposite_intercepts_l1319_131972

theorem line_through_point_with_opposite_intercepts :
  (∃ m : ℝ, (∀ x y : ℝ, y = m * x → (2,3) = (x, y)) ∧ ((∀ a : ℝ, a ≠ 0 → (x / a + y / (-a) = 1) → (2 - 3 = a ∧ a = -1)))) →
  ((∀ x y : ℝ, 3 * x - 2 * y = 0) ∨ (∀ x y : ℝ, x - y + 1 = 0)) :=
by
  sorry

end line_through_point_with_opposite_intercepts_l1319_131972


namespace sin_double_angle_identity_l1319_131981

theorem sin_double_angle_identity (x : ℝ) (h : Real.sin (x + π/4) = -3/5) : Real.sin (2 * x) = -7/25 := 
by 
  sorry

end sin_double_angle_identity_l1319_131981


namespace arithmetic_sequence_sum_l1319_131932

theorem arithmetic_sequence_sum : 
  ∃ x y, (∃ d, 
  d = 12 - 5 ∧ 
  19 + d = x ∧ 
  x + d = y ∧ 
  y + d = 40 ∧ 
  x + y = 59) :=
by {
  sorry
}

end arithmetic_sequence_sum_l1319_131932


namespace area_PST_is_5_l1319_131943

noncomputable def area_of_triangle_PST 
  (P Q R S T : Point)
  (PQ QR PR : ℝ) 
  (PS PT : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 9)
  (hPR : PR = 10)
  (hPS : PS = 3)
  (hPT : PT = 5)
  : ℝ := 
  5

theorem area_PST_is_5 
  (P Q R S T : Point)
  (PQ QR PR : ℝ) 
  (PS PT : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 9)
  (hPR : PR = 10)
  (hPS : PS = 3)
  (hPT : PT = 5)
  : area_of_triangle_PST P Q R S T PQ QR PR PS PT hPQ hQR hPR hPS hPT = 5 :=
sorry

end area_PST_is_5_l1319_131943


namespace total_profit_is_18900_l1319_131984

-- Defining the conditions
variable (x : ℕ)  -- A's initial investment
variable (A_share : ℕ := 6300)  -- A's share in rupees

-- Total profit calculation
def total_annual_gain : ℕ :=
  (x * 12) + (2 * x * 6) + (3 * x * 4)

-- The main statement
theorem total_profit_is_18900 (x : ℕ) (A_share : ℕ := 6300) :
  3 * A_share = total_annual_gain x :=
by sorry

end total_profit_is_18900_l1319_131984


namespace mural_lunch_break_duration_l1319_131942

variable (a t L : ℝ)

theorem mural_lunch_break_duration
  (h1 : (8 - L) * (a + t) = 0.6)
  (h2 : (6.5 - L) * t = 0.3)
  (h3 : (11 - L) * a = 0.1) :
  L = 40 :=
by
  sorry

end mural_lunch_break_duration_l1319_131942


namespace misread_signs_in_front_of_6_terms_l1319_131957

/-- Define the polynomial function --/
def poly (x : ℝ) : ℝ :=
  10 * x ^ 9 + 9 * x ^ 8 + 8 * x ^ 7 + 7 * x ^ 6 + 6 * x ^ 5 + 5 * x ^ 4 + 4 * x ^ 3 + 3 * x ^ 2 + 2 * x + 1

/-- Xiao Ming's mistaken result --/
def mistaken_result : ℝ := 7

/-- Correct value of the expression at x = -1 --/
def correct_value : ℝ := poly (-1)

/-- The difference due to misreading signs --/
def difference : ℝ := mistaken_result - correct_value

/-- Prove that Xiao Ming misread the signs in front of 6 terms --/
theorem misread_signs_in_front_of_6_terms :
  difference / 2 = 6 :=
by
  simp [difference, correct_value, poly]
  -- the proof steps would go here
  sorry

#eval poly (-1)  -- to validate the correct value
#eval mistaken_result - poly (-1)  -- to validate the difference

end misread_signs_in_front_of_6_terms_l1319_131957


namespace fraction_simplification_l1319_131903

theorem fraction_simplification : 
  (1877^2 - 1862^2) / (1880^2 - 1859^2) = 5 / 7 := 
by 
  sorry

end fraction_simplification_l1319_131903


namespace expected_value_of_winning_is_2550_l1319_131987

-- Definitions based on the conditions
def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℚ := 1 / 8
def winnings (n : ℕ) : ℕ := n^2

-- Expected value calculation based on the conditions
noncomputable def expected_value : ℚ :=
  (outcomes.map (λ n => probability n * winnings n)).sum

-- Proposition stating that the expected value is 25.50
theorem expected_value_of_winning_is_2550 : expected_value = 25.50 :=
by
  sorry

end expected_value_of_winning_is_2550_l1319_131987


namespace haley_spent_32_dollars_l1319_131921

noncomputable def total_spending (ticket_price : ℕ) (tickets_bought_self_friends : ℕ) (extra_tickets : ℕ) : ℕ :=
  ticket_price * (tickets_bought_self_friends + extra_tickets)

theorem haley_spent_32_dollars :
  total_spending 4 3 5 = 32 :=
by
  sorry

end haley_spent_32_dollars_l1319_131921


namespace matt_new_average_commission_l1319_131982

noncomputable def new_average_commission (x : ℝ) : ℝ :=
  (5 * x + 1000) / 6

theorem matt_new_average_commission
  (x : ℝ)
  (h1 : (5 * x + 1000) / 6 = x + 150)
  (h2 : x = 100) :
  new_average_commission x = 250 :=
by
  sorry

end matt_new_average_commission_l1319_131982


namespace arithmetic_sequence_a8_l1319_131907

theorem arithmetic_sequence_a8 (a : ℕ → ℤ) (d : ℤ) :
  a 2 = 4 → a 4 = 2 → a 8 = -2 :=
by intros ha2 ha4
   sorry

end arithmetic_sequence_a8_l1319_131907


namespace binary1011_eq_11_l1319_131992

-- Define a function to convert a binary number represented as a list of bits to a decimal number.
def binaryToDecimal (bits : List (Fin 2)) : Nat :=
  bits.foldr (λ (bit : Fin 2) (acc : Nat) => acc * 2 + bit.val) 0

-- The binary number 1011 represented as a list of bits.
def binary1011 : List (Fin 2) := [1, 0, 1, 1]

-- The theorem stating that the decimal equivalent of binary 1011 is 11.
theorem binary1011_eq_11 : binaryToDecimal binary1011 = 11 :=
by
  sorry

end binary1011_eq_11_l1319_131992


namespace cos_and_sin_double_angle_l1319_131901

variables (θ : ℝ)

-- Conditions
def is_in_fourth_quadrant (θ : ℝ) : Prop :=
  θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi

def sin_theta (θ : ℝ) : Prop :=
  Real.sin θ = -1 / 3

-- Problem statement
theorem cos_and_sin_double_angle (h1 : is_in_fourth_quadrant θ) (h2 : sin_theta θ) :
  Real.cos θ = 2 * Real.sqrt 2 / 3 ∧ Real.sin (2 * θ) = -(4 * Real.sqrt 2 / 9) :=
sorry

end cos_and_sin_double_angle_l1319_131901


namespace randy_biscuits_left_l1319_131974

-- Define the function biscuits_left
def biscuits_left (initial: ℚ) (father_gift: ℚ) (mother_gift: ℚ) (brother_eat_percent: ℚ) : ℚ :=
  let total_before_eat := initial + father_gift + mother_gift
  let brother_ate := brother_eat_percent * total_before_eat
  total_before_eat - brother_ate

-- Given conditions
def initial_biscuits : ℚ := 32
def father_gift : ℚ := 2 / 3
def mother_gift : ℚ := 15
def brother_eat_percent : ℚ := 0.3

-- Correct answer as an approximation since we're dealing with real-world numbers
def approx (x y : ℚ) := abs (x - y) < 0.01

-- The proof problem statement in Lean 4
theorem randy_biscuits_left :
  approx (biscuits_left initial_biscuits father_gift mother_gift brother_eat_percent) 33.37 :=
by
  sorry

end randy_biscuits_left_l1319_131974


namespace parabola_equation_l1319_131959

theorem parabola_equation (p : ℝ) (h1 : 0 < p) (h2 : p / 2 = 2) : ∀ y x : ℝ, y^2 = -8 * x :=
by
  sorry

end parabola_equation_l1319_131959


namespace dad_steps_l1319_131941

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l1319_131941


namespace problem_1_problem_2_l1319_131952

-- First Proof Problem
theorem problem_1 (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x^2 + 1) : 
  f x = 2 * x^2 - 4 * x + 3 :=
sorry

-- Second Proof Problem
theorem problem_2 {a b : ℝ} (f : ℝ → ℝ) (hf : ∀ x, f x = x / (a * x + b))
  (h1 : f 2 = 1) (h2 : ∃! x, f x = x) : 
  f x = 2 * x / (x + 2) :=
sorry

end problem_1_problem_2_l1319_131952


namespace inequality_solution_set_non_empty_l1319_131938

theorem inequality_solution_set_non_empty (a : ℝ) :
  (∃ x : ℝ, a * x > -1 ∧ x + a > 0) ↔ a > -1 :=
sorry

end inequality_solution_set_non_empty_l1319_131938


namespace linear_dependence_k_l1319_131956

theorem linear_dependence_k :
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
    (a * (2 : ℝ) + b * (5 : ℝ) = 0) ∧ 
    (a * (3 : ℝ) + b * k = 0) →
  k = 15 / 2 := by
  sorry

end linear_dependence_k_l1319_131956


namespace binomial_expansion_const_term_l1319_131963

theorem binomial_expansion_const_term (a : ℝ) (h : a > 0) 
  (A : ℝ) (B : ℝ) :
  (A = (15 * a ^ 4)) ∧ (B = 15 * a ^ 2) ∧ (A = 4 * B) → B = 60 := 
by 
  -- The actual proof is omitted
  sorry

end binomial_expansion_const_term_l1319_131963
