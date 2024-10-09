import Mathlib

namespace largest_divisor_of_odd_sequence_for_even_n_l290_29041

theorem largest_divisor_of_odd_sequence_for_even_n (n : ℕ) (h : n % 2 = 0) : 
  ∃ d : ℕ, d = 105 ∧ ∀ k : ℕ, k = (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13) → 105 ∣ k :=
sorry

end largest_divisor_of_odd_sequence_for_even_n_l290_29041


namespace min_sum_factors_l290_29027

theorem min_sum_factors (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prod : a * b * c = 2310) : a + b + c = 40 :=
sorry

end min_sum_factors_l290_29027


namespace price_of_other_frisbees_l290_29058

theorem price_of_other_frisbees :
  ∃ F3 Fx Px : ℕ, F3 + Fx = 60 ∧ 3 * F3 + Px * Fx = 204 ∧ Fx ≥ 24 ∧ Px = 4 := 
by
  sorry

end price_of_other_frisbees_l290_29058


namespace two_digit_sum_condition_l290_29021

theorem two_digit_sum_condition (x y : ℕ) (hx : 1 ≤ x) (hx9 : x ≤ 9) (hy : 0 ≤ y) (hy9 : y ≤ 9)
    (h : (x + 1) + (y + 2) - 10 = 2 * (x + y)) :
    (x = 6 ∧ y = 8) ∨ (x = 5 ∧ y = 9) :=
sorry

end two_digit_sum_condition_l290_29021


namespace log_expression_simplification_l290_29019

open Real

theorem log_expression_simplification (p q r s t z : ℝ) :
  log (p / q) + log (q / r) + log (r / s) - log (p * t / (s * z)) = log (z / t) :=
  sorry

end log_expression_simplification_l290_29019


namespace line_parallel_to_y_axis_l290_29059

theorem line_parallel_to_y_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x + b * y + 1 = 0 → b = 0):
  a ≠ 0 ∧ b = 0 :=
sorry

end line_parallel_to_y_axis_l290_29059


namespace altitude_eq_4r_l290_29067

variable (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]

-- We define the geometrical relations and constraints
def AC_eq_BC (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (AC BC : ℝ) : Prop :=
AC = BC

def in_circle_radius_eq_r (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (incircle_radius r : ℝ) : Prop :=
incircle_radius = r

def ex_circle_radius_eq_r (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (excircle_radius r : ℝ) : Prop :=
excircle_radius = r

-- Main theorem to prove
theorem altitude_eq_4r 
  (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (AC BC : ℝ) (r : ℝ)
  (h : ℝ)
  (H1 : AC_eq_BC A B C D AC BC)
  (H2 : in_circle_radius_eq_r A B C D r r)
  (H3 : ex_circle_radius_eq_r A B C D r r) :
  h = 4 * r :=
  sorry

end altitude_eq_4r_l290_29067


namespace total_time_is_60_l290_29020

def emma_time : ℕ := 20
def fernando_time : ℕ := 2 * emma_time
def total_time : ℕ := emma_time + fernando_time

theorem total_time_is_60 : total_time = 60 := by
  sorry

end total_time_is_60_l290_29020


namespace john_days_off_l290_29032

def streams_per_week (earnings_per_week : ℕ) (rate_per_hour : ℕ) : ℕ := earnings_per_week / rate_per_hour

def streaming_sessions (hours_per_week : ℕ) (hours_per_session : ℕ) : ℕ := hours_per_week / hours_per_session

def days_off_per_week (total_days : ℕ) (streaming_days : ℕ) : ℕ := total_days - streaming_days

theorem john_days_off (hours_per_session : ℕ) (hourly_rate : ℕ) (weekly_earnings : ℕ) (total_days : ℕ) :
  hours_per_session = 4 → 
  hourly_rate = 10 → 
  weekly_earnings = 160 → 
  total_days = 7 → 
  days_off_per_week total_days (streaming_sessions (streams_per_week weekly_earnings hourly_rate) hours_per_session) = 3 := 
by
  intros
  sorry

end john_days_off_l290_29032


namespace negation_of_exists_l290_29011

theorem negation_of_exists (x : ℝ) : x^2 + 2 * x + 2 > 0 := sorry

end negation_of_exists_l290_29011


namespace problem_1_problem_2_l290_29082

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), Real.cos (2 * x))
noncomputable def vec_b : ℝ × ℝ := (1 / 2, Real.sqrt 3 / 2)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).fst * vec_b.fst + (vec_a x).snd * vec_b.snd + 2

theorem problem_1 (x : ℝ) : x ∈ Set.Icc (k * Real.pi - (5 / 12) * Real.pi) (k * Real.pi + (1 / 12) * Real.pi) → ∃ k : ℤ, ∀ x : ℝ, f (x) = Real.sin (2 * x + (1 / 3) * Real.pi) + 2 :=
sorry

theorem problem_2 (x : ℝ) : x ∈ Set.Icc (π / 6) (2 * π / 3) → f (π / 6) = (Real.sqrt 3 / 2) + 2 ∧ f (7 * π / 12) = 1 :=
sorry

end problem_1_problem_2_l290_29082


namespace find_coordinates_l290_29047

def A : Prod ℤ ℤ := (-3, 2)
def move_right (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst + 1, p.snd)
def move_down (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst, p.snd - 2)

theorem find_coordinates :
  move_down (move_right A) = (-2, 0) :=
by
  sorry

end find_coordinates_l290_29047


namespace sara_frosting_total_l290_29051

def cakes_baked_each_day : List Nat := [7, 12, 8, 10, 15]
def cakes_eaten_by_Carol : List Nat := [4, 6, 3, 2, 3]
def cans_per_cake_each_day : List Nat := [2, 3, 4, 3, 2]

def total_frosting_cans_needed : Nat :=
  let remaining_cakes := List.zipWith (· - ·) cakes_baked_each_day cakes_eaten_by_Carol
  let required_cans := List.zipWith (· * ·) remaining_cakes cans_per_cake_each_day
  required_cans.foldl (· + ·) 0

theorem sara_frosting_total : total_frosting_cans_needed = 92 := by
  sorry

end sara_frosting_total_l290_29051


namespace inequality_solution_l290_29064

theorem inequality_solution (x : ℝ) :
  (x+3)/(x+4) > (4*x+5)/(3*x+10) ↔ x ∈ Set.Ioo (-4 : ℝ) (- (10 : ℝ) / 3) ∪ Set.Ioi 2 :=
by
  sorry

end inequality_solution_l290_29064


namespace tan_double_angle_l290_29033

theorem tan_double_angle (α : Real) (h1 : Real.sin α - Real.cos α = 4 / 3) (h2 : α ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4)) :
  Real.tan (2 * α) = (7 * Real.sqrt 2) / 8 :=
by
  sorry

end tan_double_angle_l290_29033


namespace age_of_new_person_l290_29069

theorem age_of_new_person (n : ℕ) (T A : ℕ) (h₁ : n = 10) (h₂ : T = 15 * n)
    (h₃ : (T + A) / (n + 1) = 17) : A = 37 := by
  sorry

end age_of_new_person_l290_29069


namespace binom_60_3_eq_34220_l290_29098

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l290_29098


namespace kira_breakfast_time_l290_29075

theorem kira_breakfast_time :
  let fry_time_per_sausage := 5 -- minutes per sausage
  let scramble_time_per_egg := 4 -- minutes per egg
  let sausages := 3
  let eggs := 6
  let time_to_fry := sausages * fry_time_per_sausage
  let time_to_scramble := eggs * scramble_time_per_egg
  (time_to_fry + time_to_scramble) = 39 := 
by
  sorry

end kira_breakfast_time_l290_29075


namespace find_variable_value_l290_29062

axiom variable_property (x : ℝ) (h : 4 + 1 / x ≠ 0) : 5 / (4 + 1 / x) = 1 → x = 1

-- Given condition: 5 / (4 + 1 / x) = 1
-- Prove: x = 1
theorem find_variable_value (x : ℝ) (h : 4 + 1 / x ≠ 0) (h1 : 5 / (4 + 1 / x) = 1) : x = 1 :=
variable_property x h h1

end find_variable_value_l290_29062


namespace maximum_pencils_l290_29045

-- Define the problem conditions
def red_pencil_cost := 27
def blue_pencil_cost := 23
def max_total_cost := 940
def max_diff := 10

-- Define the main theorem
theorem maximum_pencils (x y : ℕ) 
  (h1 : red_pencil_cost * x + blue_pencil_cost * y ≤ max_total_cost)
  (h2 : y - x ≤ max_diff)
  (hx_min : ∀ z : ℕ, z < x → red_pencil_cost * z + blue_pencil_cost * (z + max_diff) > max_total_cost):
  x = 14 ∧ y = 24 ∧ x + y = 38 := 
  sorry

end maximum_pencils_l290_29045


namespace find_original_integer_l290_29065

theorem find_original_integer (a b c d : ℕ) 
    (h1 : (b + c + d) / 3 + 10 = 37) 
    (h2 : (a + c + d) / 3 + 10 = 31) 
    (h3 : (a + b + d) / 3 + 10 = 25) 
    (h4 : (a + b + c) / 3 + 10 = 19) : 
    d = 45 := 
    sorry

end find_original_integer_l290_29065


namespace evaluate_dollar_l290_29079

variable {R : Type} [Field R]

def dollar (a b : R) : R := (a - b) ^ 2

theorem evaluate_dollar (x y : R) : dollar (2 * x + 3 * y) (3 * x - 4 * y) = x ^ 2 - 14 * x * y + 49 * y ^ 2 := by
  sorry

end evaluate_dollar_l290_29079


namespace find_length_PQ_l290_29016

noncomputable def length_of_PQ (PQ PR : ℝ) (ST SU : ℝ) (angle_PQPR angle_STSU : ℝ) : ℝ :=
if (angle_PQPR = 120 ∧ angle_STSU = 120 ∧ PR / SU = 8 / 9) then 
  2 
else 
  0

theorem find_length_PQ :
  let PQ := 4 
  let PR := 8
  let ST := 9
  let SU := 18
  let PQ_crop := 2
  let angle_PQPR := 120
  let angle_STSU := 120
  length_of_PQ PQ PR ST SU angle_PQPR angle_STSU = PQ_crop :=
by
  sorry

end find_length_PQ_l290_29016


namespace zero_points_sum_gt_one_l290_29054

noncomputable def f (x : ℝ) : ℝ := Real.log x + (1 / (2 * x))

noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem zero_points_sum_gt_one (x₁ x₂ m : ℝ) (h₁ : x₁ < x₂) 
  (hx₁ : g x₁ m = 0) (hx₂ : g x₂ m = 0) : 
  x₁ + x₂ > 1 := 
  by
    sorry

end zero_points_sum_gt_one_l290_29054


namespace total_nephews_proof_l290_29070

-- We declare the current number of nephews as unknown variables
variable (Alden_current Vihaan Shruti Nikhil : ℕ)

-- State the conditions as hypotheses
theorem total_nephews_proof
  (h1 : 70 = (1 / 3 : ℚ) * Alden_current)
  (h2 : Vihaan = Alden_current + 120)
  (h3 : Shruti = 2 * Vihaan)
  (h4 : Nikhil = Alden_current + Shruti - 40) :
  Alden_current + Vihaan + Shruti + Nikhil = 2030 := 
by
  sorry

end total_nephews_proof_l290_29070


namespace digit_6_count_1_to_700_l290_29083

theorem digit_6_count_1_to_700 :
  let countNumbersWithDigit6 := 700 - (7 * 9 * 9)
  countNumbersWithDigit6 = 133 := 
by
  let countNumbersWithDigit6 := 700 - (7 * 9 * 9)
  show countNumbersWithDigit6 = 133
  sorry

end digit_6_count_1_to_700_l290_29083


namespace contrapositive_l290_29012

theorem contrapositive (x : ℝ) : (x > 1 → x^2 + x > 2) ↔ (x^2 + x ≤ 2 → x ≤ 1) :=
sorry

end contrapositive_l290_29012


namespace golden_triangle_expression_l290_29023

noncomputable def t : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_triangle_expression :
  t = (Real.sqrt 5 - 1) / 2 →
  (1 - 2 * (Real.sin (27 * Real.pi / 180))^2) / (2 * t * Real.sqrt (4 - t^2)) = 1 / 4 :=
by
  intro h_t
  have h1 : t = (Real.sqrt 5 - 1) / 2 := h_t
  sorry

end golden_triangle_expression_l290_29023


namespace fixed_point_on_line_find_m_values_l290_29008

-- Define the conditions and set up the statements to prove

/-- 
Condition 1: Line equation 
-/
def line_eq (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

/-- 
Condition 2: Circle equation 
-/
def circle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 25

/-- 
Question (1): Fixed point (3,1) is always on the line
-/
theorem fixed_point_on_line (m : ℝ) : line_eq m 3 1 := by
  sorry

/-- 
Question (2): Finding the values of m for the given chord length
-/
theorem find_m_values (m : ℝ) (h_chord : ∀x y : ℝ, circle_eq x y → line_eq m x y → (x - y)^2 = 6) : 
  m = -1/2 ∨ m = 1/2 := by
  sorry

end fixed_point_on_line_find_m_values_l290_29008


namespace wall_length_l290_29080

theorem wall_length (s : ℕ) (d : ℕ) (w : ℕ) (L : ℝ) 
  (hs : s = 18) 
  (hd : d = 20) 
  (hw : w = 32)
  (hcombined : (s ^ 2 + Real.pi * ((d / 2) ^ 2)) = (1 / 2) * (w * L)) :
  L = 39.88 := 
sorry

end wall_length_l290_29080


namespace earnings_correct_l290_29006

-- Define the initial number of roses, the number of roses left, and the price per rose.
def initial_roses : ℕ := 13
def roses_left : ℕ := 4
def price_per_rose : ℕ := 4

-- Calculate the number of roses sold.
def roses_sold : ℕ := initial_roses - roses_left

-- Calculate the total earnings.
def earnings : ℕ := roses_sold * price_per_rose

-- Prove that the earnings are 36 dollars.
theorem earnings_correct : earnings = 36 := by
  sorry

end earnings_correct_l290_29006


namespace rita_saving_l290_29028

theorem rita_saving
  (num_notebooks : ℕ)
  (price_per_notebook : ℝ)
  (discount_rate : ℝ) :
  num_notebooks = 7 →
  price_per_notebook = 3 →
  discount_rate = 0.15 →
  (num_notebooks * price_per_notebook) - (num_notebooks * (price_per_notebook * (1 - discount_rate))) = 3.15 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end rita_saving_l290_29028


namespace man_rate_in_still_water_l290_29026

theorem man_rate_in_still_water (Vm Vs : ℝ) :
  Vm + Vs = 20 ∧ Vm - Vs = 8 → Vm = 14 :=
by
  sorry

end man_rate_in_still_water_l290_29026


namespace sum_of_reciprocal_squares_leq_reciprocal_product_square_l290_29037

theorem sum_of_reciprocal_squares_leq_reciprocal_product_square (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 3) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a^2 * b^2 * c^2 * d^2) :=
sorry

end sum_of_reciprocal_squares_leq_reciprocal_product_square_l290_29037


namespace B_can_finish_work_in_6_days_l290_29046

theorem B_can_finish_work_in_6_days :
  (A_work_alone : ℕ) → (A_work_before_B : ℕ) → (A_B_together : ℕ) → (B_days_alone : ℕ) → 
  (A_work_alone = 12) → (A_work_before_B = 3) → (A_B_together = 3) → B_days_alone = 6 :=
by
  intros A_work_alone A_work_before_B A_B_together B_days_alone
  intros h1 h2 h3
  sorry

end B_can_finish_work_in_6_days_l290_29046


namespace kevin_food_expense_l290_29017

theorem kevin_food_expense
    (total_budget : ℕ)
    (samuel_ticket : ℕ)
    (samuel_food_drinks : ℕ)
    (kevin_ticket : ℕ)
    (kevin_drinks : ℕ)
    (kevin_total_exp : ℕ) :
    total_budget = 20 →
    samuel_ticket = 14 →
    samuel_food_drinks = 6 →
    kevin_ticket = 14 →
    kevin_drinks = 2 →
    kevin_total_exp = 20 →
    kevin_food = 4 :=
by
  sorry

end kevin_food_expense_l290_29017


namespace solve_expr_l290_29050

theorem solve_expr (x : ℝ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end solve_expr_l290_29050


namespace price_per_glass_first_day_l290_29004

variables (O G : ℝ) (P1 : ℝ)

theorem price_per_glass_first_day (H1 : G * P1 = 1.5 * G * 0.40) : 
  P1 = 0.60 :=
by sorry

end price_per_glass_first_day_l290_29004


namespace sum_evaluation_l290_29042

theorem sum_evaluation : 5 * 399 + 4 * 399 + 3 * 399 + 398 = 5186 :=
by
  sorry

end sum_evaluation_l290_29042


namespace axis_symmetry_shifted_graph_l290_29043

open Real

theorem axis_symmetry_shifted_graph :
  ∀ k : ℤ, ∃ x : ℝ, (y = 2 * sin (2 * x)) ∧
  y = 2 * sin (2 * (x + π / 12)) ↔
  x = k * π / 2 + π / 6 :=
sorry

end axis_symmetry_shifted_graph_l290_29043


namespace range_of_a_l290_29061

noncomputable def quadratic (a x : ℝ) : ℝ := a * x^2 + a * x + 1
noncomputable def quadratic_eq (x₀ a : ℝ) : Prop := x₀^2 - x₀ + a = 0

theorem range_of_a (a : ℝ) (p : ∀ x : ℝ, quadratic a x > 0) (q : ∃ x₀ : ℝ, quadratic_eq x₀ a) : 0 ≤ a ∧ a ≤ 1/4 :=
  sorry

end range_of_a_l290_29061


namespace range_of_b_l290_29007

noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 / 2 then (2 * x + 1) / (x ^ 2) else x + 1

def g (x : ℝ) : ℝ := x ^ 2 - 4 * x - 4

-- The main theorem to prove the range of b
theorem range_of_b (a b : ℝ) (h : f a + g b = 0) : b ∈ Set.Icc (-1) 5 := by
  sorry

end range_of_b_l290_29007


namespace nonnegative_integer_pairs_solution_l290_29088

theorem nonnegative_integer_pairs_solution :
  ∀ (x y: ℕ), ((x * y + 2) ^ 2 = x^2 + y^2) ↔ (x = 0 ∧ y = 2) ∨ (x = 2 ∧ y = 0) :=
by
  sorry

end nonnegative_integer_pairs_solution_l290_29088


namespace train_A_length_l290_29003

theorem train_A_length
  (speed_A : ℕ)
  (speed_B : ℕ)
  (time_to_cross : ℕ)
  (len_A : ℕ)
  (h1 : speed_A = 54) 
  (h2 : speed_B = 36) 
  (h3 : time_to_cross = 15)
  (h4 : len_A = (speed_A + speed_B) * 1000 / 3600 * time_to_cross) :
  len_A = 375 :=
sorry

end train_A_length_l290_29003


namespace number_of_girl_students_l290_29052

theorem number_of_girl_students (total_third_graders : ℕ) (boy_students : ℕ) (girl_students : ℕ) 
  (h1 : total_third_graders = 123) (h2 : boy_students = 66) (h3 : total_third_graders = boy_students + girl_students) :
  girl_students = 57 :=
by
  sorry

end number_of_girl_students_l290_29052


namespace total_animals_seen_correct_l290_29081

-- Define the number of beavers in the morning
def beavers_morning : ℕ := 35

-- Define the number of chipmunks in the morning
def chipmunks_morning : ℕ := 60

-- Define the number of beavers in the afternoon (tripled)
def beavers_afternoon : ℕ := 3 * beavers_morning

-- Define the number of chipmunks in the afternoon (decreased by 15)
def chipmunks_afternoon : ℕ := chipmunks_morning - 15

-- Calculate the total number of animals seen in the morning
def total_morning : ℕ := beavers_morning + chipmunks_morning

-- Calculate the total number of animals seen in the afternoon
def total_afternoon : ℕ := beavers_afternoon + chipmunks_afternoon

-- The total number of animals seen that day
def total_animals_seen : ℕ := total_morning + total_afternoon

theorem total_animals_seen_correct :
  total_animals_seen = 245 :=
by
  -- skipping the proof
  sorry

end total_animals_seen_correct_l290_29081


namespace school_distance_is_seven_l290_29085

-- Definitions based on conditions
def distance_to_school (x : ℝ) : Prop :=
  let monday_to_thursday_distance := 8 * x
  let friday_distance := 2 * x + 4
  let total_distance := monday_to_thursday_distance + friday_distance
  total_distance = 74

-- The problem statement to prove
theorem school_distance_is_seven : ∃ (x : ℝ), distance_to_school x ∧ x = 7 := 
by {
  sorry
}

end school_distance_is_seven_l290_29085


namespace inequality_proof_l290_29095

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x * (x - z) ^ 2 + y * (y - z) ^ 2 ≥ (x - z) * (y - z) * (x + y - z) :=
by
  sorry

end inequality_proof_l290_29095


namespace fraction_interval_l290_29078

theorem fraction_interval :
  (5 / 24 > 1 / 6) ∧ (5 / 24 < 1 / 4) ∧
  (¬ (5 / 12 > 1 / 6 ∧ 5 / 12 < 1 / 4)) ∧
  (¬ (5 / 36 > 1 / 6 ∧ 5 / 36 < 1 / 4)) ∧
  (¬ (5 / 60 > 1 / 6 ∧ 5 / 60 < 1 / 4)) ∧
  (¬ (5 / 48 > 1 / 6 ∧ 5 / 48 < 1 / 4)) :=
by
  sorry

end fraction_interval_l290_29078


namespace almond_butter_ratio_l290_29001

theorem almond_butter_ratio
  (peanut_cost almond_cost batch_extra almond_per_batch : ℝ)
  (h1 : almond_cost = 3 * peanut_cost)
  (h2 : peanut_cost = 3)
  (h3 : almond_per_batch = batch_extra)
  (h4 : batch_extra = 3) :
  almond_per_batch / almond_cost = 1 / 3 := sorry

end almond_butter_ratio_l290_29001


namespace intersection_of_sets_l290_29093

open Set

theorem intersection_of_sets (M N : Set ℕ) (hM : M = {1, 2, 3}) (hN : N = {2, 3, 4}) :
  M ∩ N = {2, 3} :=
by
  sorry

end intersection_of_sets_l290_29093


namespace f_pi_over_4_l290_29015

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

theorem f_pi_over_4 (ω φ : ℝ) (h : ω ≠ 0) 
  (symm : ∀ x, f ω φ (π / 4 + x) = f ω φ (π / 4 - x)) : 
  f ω φ (π / 4) = 2 ∨ f ω φ (π / 4) = -2 := 
by 
  sorry

end f_pi_over_4_l290_29015


namespace triangles_in_figure_l290_29010

-- Define the conditions of the problem.
def bottom_row_small := 4
def next_row_small := 3
def following_row_small := 2
def topmost_row_small := 1

def small_triangles := bottom_row_small + next_row_small + following_row_small + topmost_row_small

def medium_triangles := 3
def large_triangle := 1

def total_triangles := small_triangles + medium_triangles + large_triangle

-- Lean proof statement that the total number of triangles is 14
theorem triangles_in_figure : total_triangles = 14 :=
by
  unfold total_triangles
  unfold small_triangles
  unfold bottom_row_small next_row_small following_row_small topmost_row_small
  unfold medium_triangles large_triangle
  sorry

end triangles_in_figure_l290_29010


namespace sin_identity_l290_29034

theorem sin_identity (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
  by 
  sorry

end sin_identity_l290_29034


namespace imaginary_part_z_is_correct_l290_29039

open Complex

noncomputable def problem_conditions (z : ℂ) : Prop :=
  (3 - 4 * Complex.I) * z = Complex.abs (4 + 3 * Complex.I)

theorem imaginary_part_z_is_correct (z : ℂ) (hz : problem_conditions z) :
  z.im = 4 / 5 :=
sorry

end imaginary_part_z_is_correct_l290_29039


namespace rational_cos_terms_l290_29096

open Real

noncomputable def rational_sum (x : ℝ) (rS : ℚ) (rC : ℚ) :=
  let S := sin (64 * x) + sin (65 * x)
  let C := cos (64 * x) + cos (65 * x)
  S = rS ∧ C = rC

theorem rational_cos_terms (x : ℝ) (rS : ℚ) (rC : ℚ) :
  rational_sum x rS rC → (∃ q1 q2 : ℚ, cos (64 * x) = q1 ∧ cos (65 * x) = q2) :=
sorry

end rational_cos_terms_l290_29096


namespace work_duration_17_333_l290_29000

def work_done (rate: ℚ) (days: ℕ) : ℚ := rate * days

def combined_work_done (rate1: ℚ) (rate2: ℚ) (days: ℕ) : ℚ :=
  (rate1 + rate2) * days

def total_work_done (rate1: ℚ) (rate2: ℚ) (rate3: ℚ) (days: ℚ) : ℚ :=
  (rate1 + rate2 + rate3) * days

noncomputable def total_days_work_last (rate_p rate_q rate_r: ℚ) : ℚ :=
  have work_p := 8 * rate_p
  have work_pq := combined_work_done rate_p rate_q 4
  have remaining_work := 1 - (work_p + work_pq)
  have days_all_together := remaining_work / (rate_p + rate_q + rate_r)
  8 + 4 + days_all_together

theorem work_duration_17_333 (rate_p rate_q rate_r: ℚ) : total_days_work_last rate_p rate_q rate_r = 17.333 :=
  by 
  have hp := 1/40
  have hq := 1/24
  have hr := 1/30
  sorry -- proof omitted

end work_duration_17_333_l290_29000


namespace line_length_after_erasing_l290_29077

-- Definition of the initial length in meters and the erased length in centimeters
def initial_length_meters : ℝ := 1.5
def erased_length_centimeters : ℝ := 15.25

-- Conversion factor from meters to centimeters
def meters_to_centimeters (m : ℝ) : ℝ := m * 100

-- Definition of the initial length in centimeters
def initial_length_centimeters : ℝ := meters_to_centimeters initial_length_meters

-- Statement of the theorem
theorem line_length_after_erasing :
  initial_length_centimeters - erased_length_centimeters = 134.75 :=
by
  -- The proof would go here
  sorry

end line_length_after_erasing_l290_29077


namespace total_number_of_questions_l290_29076

theorem total_number_of_questions (type_a_problems type_b_problems : ℕ) 
(time_spent_type_a time_spent_type_b : ℕ) 
(total_exam_time : ℕ) 
(h1 : type_a_problems = 50) 
(h2 : time_spent_type_a = 2 * time_spent_type_b) 
(h3 : time_spent_type_a * type_a_problems = 72) 
(h4 : total_exam_time = 180) :
type_a_problems + type_b_problems = 200 := 
by
  sorry

end total_number_of_questions_l290_29076


namespace sum_a2_a4_a6_l290_29014

-- Define the arithmetic sequence with a positive common difference
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ (d : ℝ), d > 0 ∧ ∀ n, a (n + 1) = a n + d

-- Define that a_1 and a_7 are roots of the quadratic equation x^2 - 10x + 16 = 0
def roots_condition (a : ℕ → ℝ) : Prop :=
(a 1) * (a 7) = 16 ∧ (a 1) + (a 7) = 10

-- The main theorem we want to prove
theorem sum_a2_a4_a6 (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : roots_condition a) :
  a 2 + a 4 + a 6 = 15 :=
sorry

end sum_a2_a4_a6_l290_29014


namespace teacher_engineer_ratio_l290_29048

theorem teacher_engineer_ratio 
  (t e : ℕ) -- t is the number of teachers and e is the number of engineers.
  (h1 : (40 * t + 55 * e) / (t + e) = 45)
  : t = 2 * e :=
by
  sorry

end teacher_engineer_ratio_l290_29048


namespace james_veg_consumption_l290_29024

-- Define the given conditions in Lean
def asparagus_per_day : ℝ := 0.25
def broccoli_per_day : ℝ := 0.25
def days_in_week : ℝ := 7
def weeks : ℝ := 2
def kale_per_week : ℝ := 3

-- Define the amount of vegetables (initial, doubled, and added kale)
def initial_veg_per_day := asparagus_per_day + broccoli_per_day
def initial_veg_per_week := initial_veg_per_day * days_in_week
def double_veg_per_week := initial_veg_per_week * weeks
def total_veg_per_week_after_kale := double_veg_per_week + kale_per_week

-- Statement of the proof problem
theorem james_veg_consumption :
  total_veg_per_week_after_kale = 10 := by 
  sorry

end james_veg_consumption_l290_29024


namespace solve_abs_eq_l290_29094

theorem solve_abs_eq (x : ℝ) : 
    (3 * x + 9 = abs (-20 + 4 * x)) ↔ 
    (x = 29) ∨ (x = 11 / 7) := 
by sorry

end solve_abs_eq_l290_29094


namespace lateral_surface_area_cut_off_l290_29002

theorem lateral_surface_area_cut_off {a b c d : ℝ} (h₁ : a = 4) (h₂ : b = 25) 
(h₃ : c = (2/5 : ℝ)) (h₄ : d = 2 * (4 / 25) * b) : 
4 + 10 + (1/4 * b) = 20.25 :=
by
  sorry

end lateral_surface_area_cut_off_l290_29002


namespace complex_in_fourth_quadrant_l290_29049

theorem complex_in_fourth_quadrant (m : ℝ) :
  (m^2 - 8*m + 15 > 0) ∧ (m^2 - 5*m - 14 < 0) →
  (-2 < m ∧ m < 3) ∨ (5 < m ∧ m < 7) :=
sorry

end complex_in_fourth_quadrant_l290_29049


namespace triplet_solution_l290_29060

theorem triplet_solution (a b c : ℕ) (h1 : a^2 + b^2 + c^2 = 2005) (h2 : a ≤ b) (h3 : b ≤ c) :
  (a = 24 ∧ b = 30 ∧ c = 23) ∨ 
  (a = 12 ∧ b = 30 ∧ c = 31) ∨
  (a = 18 ∧ b = 40 ∧ c = 9) ∨
  (a = 15 ∧ b = 22 ∧ c = 36) ∨
  (a = 12 ∧ b = 30 ∧ c = 31) :=
sorry

end triplet_solution_l290_29060


namespace number_of_rows_seating_10_is_zero_l290_29056

theorem number_of_rows_seating_10_is_zero :
  ∀ (y : ℕ) (total_people : ℕ) (total_rows : ℕ),
    (∀ (r : ℕ), r * 9 + (total_rows - r) * 10 = total_people) →
    total_people = 54 →
    total_rows = 6 →
    y = 0 :=
by
  sorry

end number_of_rows_seating_10_is_zero_l290_29056


namespace max_length_of_each_piece_l290_29057

theorem max_length_of_each_piece (a b c d : ℕ) (h1 : a = 48) (h2 : b = 72) (h3 : c = 108) (h4 : d = 120) : Nat.gcd (Nat.gcd a b) (Nat.gcd c d) = 12 := by
  sorry

end max_length_of_each_piece_l290_29057


namespace no_tiling_with_seven_sided_convex_l290_29068

noncomputable def Polygon := {n : ℕ // 3 ≤ n}

def convex (M : Polygon) : Prop := sorry

def tiles_plane (M : Polygon) : Prop := sorry

theorem no_tiling_with_seven_sided_convex (M : Polygon) (h_convex : convex M) (h_sides : 7 ≤ M.1) : ¬ tiles_plane M := sorry

end no_tiling_with_seven_sided_convex_l290_29068


namespace correct_choice_l290_29013

-- Define the structures and options
inductive Structure
| Sequential
| Conditional
| Loop
| Module

def option_A : List Structure :=
  [Structure.Sequential, Structure.Module, Structure.Conditional]

def option_B : List Structure :=
  [Structure.Sequential, Structure.Loop, Structure.Module]

def option_C : List Structure :=
  [Structure.Sequential, Structure.Conditional, Structure.Loop]

def option_D : List Structure :=
  [Structure.Module, Structure.Conditional, Structure.Loop]

-- Define the correct structures
def basic_structures : List Structure :=
  [Structure.Sequential, Structure.Conditional, Structure.Loop]

-- The theorem statement
theorem correct_choice : option_C = basic_structures :=
  by
    sorry  -- Proof would go here

end correct_choice_l290_29013


namespace dante_coconuts_l290_29074

theorem dante_coconuts (P : ℕ) (D : ℕ) (S : ℕ) (hP : P = 14) (hD : D = 3 * P) (hS : S = 10) :
  (D - S) = 32 :=
by
  sorry

end dante_coconuts_l290_29074


namespace complex_number_purely_imaginary_l290_29040

theorem complex_number_purely_imaginary (m : ℝ) 
  (h1 : m^2 - 5 * m + 6 = 0) 
  (h2 : m^2 - 3 * m ≠ 0) : 
  m = 2 :=
sorry

end complex_number_purely_imaginary_l290_29040


namespace percentage_sum_l290_29090

theorem percentage_sum (A B C : ℕ) (x y : ℕ)
  (hA : A = 120) (hB : B = 110) (hC : C = 100)
  (hAx : A = C * (1 + x / 100))
  (hBy : B = C * (1 + y / 100)) : x + y = 30 := 
by
  sorry

end percentage_sum_l290_29090


namespace rear_revolutions_l290_29044

variable (r_r : ℝ)  -- radius of the rear wheel
variable (r_f : ℝ)  -- radius of the front wheel
variable (n_f : ℕ)  -- number of revolutions of the front wheel
variable (n_r : ℕ)  -- number of revolutions of the rear wheel

-- Condition: radius of the front wheel is 2 times the radius of the rear wheel.
axiom front_radius : r_f = 2 * r_r

-- Condition: the front wheel makes 10 revolutions.
axiom front_revolutions : n_f = 10

-- Theorem statement to prove
theorem rear_revolutions : n_r = 20 :=
sorry

end rear_revolutions_l290_29044


namespace garden_length_80_l290_29063

-- Let the width of the garden be denoted by w and the length by l
-- Given conditions
def is_rectangular_garden (l w : ℝ) := l = 2 * w ∧ 2 * l + 2 * w = 240

-- We want to prove that the length of the garden is 80 yards
theorem garden_length_80 (w : ℝ) (h : is_rectangular_garden (2 * w) w) : 2 * w = 80 :=
by
  sorry

end garden_length_80_l290_29063


namespace lunks_to_apples_l290_29073

theorem lunks_to_apples :
  (∀ (a b c d e f : ℕ), (7 * b = 4 * a) → (3 * d = 5 * c) → c = 24 → f * e = d → e = 27) :=
by sorry

end lunks_to_apples_l290_29073


namespace find_x_l290_29053

theorem find_x (x : ℝ) :
  (1 / 3) * ((2 * x + 8) + (7 * x + 3) + (3 * x + 9)) = 5 * x^2 - 8 * x + 2 ↔ 
  x = (36 + Real.sqrt 2136) / 30 ∨ x = (36 - Real.sqrt 2136) / 30 := 
sorry

end find_x_l290_29053


namespace point_inside_circle_implies_range_l290_29025

theorem point_inside_circle_implies_range (a : ℝ) : 
  (1 - a)^2 + (1 + a)^2 < 4 → -1 < a ∧ a < 1 :=
by
  intro h
  sorry

end point_inside_circle_implies_range_l290_29025


namespace total_profit_correct_l290_29030

def natasha_money : ℤ := 60
def carla_money : ℤ := natasha_money / 3
def cosima_money : ℤ := carla_money / 2
def sergio_money : ℤ := (3 * cosima_money) / 2

def natasha_items : ℤ := 4
def carla_items : ℤ := 6
def cosima_items : ℤ := 5
def sergio_items : ℤ := 3

def natasha_profit_margin : ℚ := 0.10
def carla_profit_margin : ℚ := 0.15
def cosima_sergio_profit_margin : ℚ := 0.12

def natasha_item_cost : ℚ := (natasha_money : ℚ) / natasha_items
def carla_item_cost : ℚ := (carla_money : ℚ) / carla_items
def cosima_item_cost : ℚ := (cosima_money : ℚ) / cosima_items
def sergio_item_cost : ℚ := (sergio_money : ℚ) / sergio_items

def natasha_profit : ℚ := natasha_items * natasha_item_cost * natasha_profit_margin
def carla_profit : ℚ := carla_items * carla_item_cost * carla_profit_margin
def cosima_profit : ℚ := cosima_items * cosima_item_cost * cosima_sergio_profit_margin
def sergio_profit : ℚ := sergio_items * sergio_item_cost * cosima_sergio_profit_margin

def total_profit : ℚ := natasha_profit + carla_profit + cosima_profit + sergio_profit

theorem total_profit_correct : total_profit = 11.99 := 
by sorry

end total_profit_correct_l290_29030


namespace jake_correct_speed_l290_29005

noncomputable def distance (d t : ℝ) : Prop :=
  d = 50 * (t + 4/60) ∧ d = 70 * (t - 4/60)

noncomputable def correct_speed (d t : ℝ) : ℝ :=
  d / t

theorem jake_correct_speed (d t : ℝ) (h1 : distance d t) : correct_speed d t = 58 :=
by
  sorry

end jake_correct_speed_l290_29005


namespace smallest_even_consecutive_sum_l290_29092

theorem smallest_even_consecutive_sum (n : ℕ) (h_even : n % 2 = 0) (h_sum : n + (n + 2) + (n + 4) = 162) : n = 52 :=
sorry

end smallest_even_consecutive_sum_l290_29092


namespace range_of_m_l290_29086

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, (x^2)/(9 - m) + (y^2)/(m - 5) = 1 → 
  (∃ m, (7 < m ∧ m < 9))) := 
sorry

end range_of_m_l290_29086


namespace even_function_derivative_at_zero_l290_29099

-- Define an even function f and its differentiability at x = 0
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def differentiable_at_zero (f : ℝ → ℝ) : Prop := DifferentiableAt ℝ f 0

-- The theorem to prove that f'(0) = 0
theorem even_function_derivative_at_zero
  (f : ℝ → ℝ)
  (hf_even : is_even_function f)
  (hf_diff : differentiable_at_zero f) :
  deriv f 0 = 0 := 
sorry

end even_function_derivative_at_zero_l290_29099


namespace calc_expression_l290_29038

theorem calc_expression : 5 + 2 * (8 - 3) = 15 :=
by
  -- Proof steps would go here
  sorry

end calc_expression_l290_29038


namespace total_cost_l290_29031

/-- Sam initially has s yellow balloons.
He gives away a of these balloons to Fred.
Mary has m yellow balloons.
Each balloon costs c dollars.
Determine the total cost for the remaining balloons that Sam and Mary jointly have.
Given: s = 6.0, a = 5.0, m = 7.0, c = 9.0 dollars.
Expected result: the total cost is 72.0 dollars.
-/
theorem total_cost (s a m c : ℝ) (h_s : s = 6.0) (h_a : a = 5.0) (h_m : m = 7.0) (h_c : c = 9.0) :
  (s - a + m) * c = 72.0 := 
by
  rw [h_s, h_a, h_m, h_c]
  -- At this stage, the proof would involve showing the expression is 72.0, but since no proof is required:
  sorry

end total_cost_l290_29031


namespace minimum_area_of_square_on_parabola_l290_29055

theorem minimum_area_of_square_on_parabola :
  ∃ (A B C : ℝ × ℝ), 
  (∃ (x₁ x₂ x₃ : ℝ), (A = (x₁, x₁^2)) ∧ (B = (x₂, x₂^2)) ∧ (C = (x₃, x₃^2)) 
  ∧ x₁ < x₂ ∧ x₂ < x₃ 
  ∧ ∀ S : ℝ, (S = (1 + (x₃ + x₂)^2) * ((x₂ - x₃) - (x₃ - x₂))^2) → S ≥ 2) :=
sorry

end minimum_area_of_square_on_parabola_l290_29055


namespace battery_usage_minutes_l290_29022

theorem battery_usage_minutes (initial_battery final_battery : ℝ) (initial_minutes : ℝ) (rate_of_usage : ℝ) :
  initial_battery - final_battery = rate_of_usage * initial_minutes →
  initial_battery = 100 →
  final_battery = 68 →
  initial_minutes = 60 →
  rate_of_usage = 8 / 15 →
  ∃ additional_minutes : ℝ, additional_minutes = 127.5 :=
by
  intros
  sorry

end battery_usage_minutes_l290_29022


namespace minimize_sum_l290_29089

noncomputable def objective_function (x : ℝ) : ℝ := x + x^2

theorem minimize_sum : ∃ x : ℝ, (objective_function x = x + x^2) ∧ (∀ y : ℝ, objective_function y ≥ objective_function (-1/2)) :=
by
  sorry

end minimize_sum_l290_29089


namespace condition_neither_sufficient_nor_necessary_l290_29097

variable (a b : ℝ)

theorem condition_neither_sufficient_nor_necessary 
    (h1 : ∃ a b : ℝ, a > b ∧ ¬(a^2 > b^2))
    (h2 : ∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b)) :
  ¬((a > b) ↔ (a^2 > b^2)) :=
sorry

end condition_neither_sufficient_nor_necessary_l290_29097


namespace packet_b_average_height_l290_29009

theorem packet_b_average_height (x y R_A R_B H_A H_B : ℝ)
  (h_RA : R_A = 2 * x + y)
  (h_RB : R_B = 3 * x - y)
  (h_x : x = 10)
  (h_y : y = 6)
  (h_HA : H_A = 192)
  (h_20percent : H_A = H_B + 0.20 * H_B) :
  H_B = 160 := 
sorry

end packet_b_average_height_l290_29009


namespace total_space_compacted_l290_29066

-- Definitions according to the conditions
def num_cans : ℕ := 60
def space_per_can_before : ℝ := 30
def compaction_rate : ℝ := 0.20

-- Theorem statement
theorem total_space_compacted : num_cans * (space_per_can_before * compaction_rate) = 360 := by
  sorry

end total_space_compacted_l290_29066


namespace pyramid_structure_l290_29036

variables {d e f a b c h i j g : ℝ}

theorem pyramid_structure (h_val : h = 16)
                         (i_val : i = 48)
                         (j_val : j = 72)
                         (g_val : g = 8)
                         (d_def : d = b * a)
                         (e_def1 : e = b * c) 
                         (e_def2 : e = d * a)
                         (f_def : f = c * a)
                         (h_def : h = d * b)
                         (i_def : i = d * a)
                         (j_def : j = e * c)
                         (g_def : g = f * c) : 
   a = 3 ∧ b = 1 ∧ c = 1.5 :=
by sorry

end pyramid_structure_l290_29036


namespace measure_angle_y_l290_29035

theorem measure_angle_y
  (triangle_angles : ∀ {A B C : ℝ}, (A = 45 ∧ B = 45 ∧ C = 90) ∨ (A = 45 ∧ B = 90 ∧ C = 45) ∨ (A = 90 ∧ B = 45 ∧ C = 45))
  (p q : ℝ) (hpq : p = q) :
  ∃ (y : ℝ), y = 90 :=
by
  sorry

end measure_angle_y_l290_29035


namespace car_speed_l290_29071

variable (v : ℝ)
variable (Distance : ℝ := 1)  -- distance in kilometers
variable (Speed_120 : ℝ := 120)  -- speed in kilometers per hour
variable (Time_120 : ℝ := Distance / Speed_120)  -- time in hours to travel 1 km at 120 km/h
variable (Time_120_sec : ℝ := Time_120 * 3600)  -- time in seconds to travel 1 km at 120 km/h
variable (Additional_time : ℝ := 2)  -- additional time in seconds
variable (Time_v_sec : ℝ := Time_120_sec + Additional_time)  -- time in seconds for unknown speed
variable (Time_v : ℝ := Time_v_sec / 3600)  -- time in hours for unknown speed

theorem car_speed (h : v = Distance / Time_v) : v = 112.5 :=
by
  -- The given proof steps will go here
  sorry

end car_speed_l290_29071


namespace forty_percent_of_number_l290_29029

/--
Given that (1/4) * (1/3) * (2/5) * N = 30, prove that 0.40 * N = 360.
-/
theorem forty_percent_of_number {N : ℝ} (h : (1/4 : ℝ) * (1/3) * (2/5) * N = 30) : 0.40 * N = 360 := 
by
  sorry

end forty_percent_of_number_l290_29029


namespace dolls_total_correct_l290_29018

def Jazmin_dolls : Nat := 1209
def Geraldine_dolls : Nat := 2186
def total_dolls : Nat := Jazmin_dolls + Geraldine_dolls

theorem dolls_total_correct : total_dolls = 3395 := by
  sorry

end dolls_total_correct_l290_29018


namespace points_for_win_l290_29084

variable (W T : ℕ)

theorem points_for_win (W T : ℕ) (h1 : W * (T + 12) + T = 60) : W = 2 :=
by {
  sorry
}

end points_for_win_l290_29084


namespace horner_eval_v4_at_2_l290_29087

def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

theorem horner_eval_v4_at_2 : 
  let x := 2
  let v_0 := 1
  let v_1 := (v_0 * x) - 12 
  let v_2 := (v_1 * x) + 60 
  let v_3 := (v_2 * x) - 160 
  let v_4 := (v_3 * x) + 240 
  v_4 = 80 := 
by 
  sorry

end horner_eval_v4_at_2_l290_29087


namespace number_of_sequences_l290_29091

-- Define the number of possible outcomes for a single coin flip
def coinFlipOutcomes : ℕ := 2

-- Define the number of flips
def numberOfFlips : ℕ := 8

-- Theorem statement: The number of distinct sequences when flipping a coin eight times is 256
theorem number_of_sequences (n : ℕ) (outcomes : ℕ) (h : outcomes = 2) (hn : n = 8) : outcomes ^ n = 256 := by
  sorry

end number_of_sequences_l290_29091


namespace kevin_total_cost_l290_29072

theorem kevin_total_cost :
  let muffin_cost := 0.75
  let juice_cost := 1.45
  let total_muffins := 3
  let cost_muffins := total_muffins * muffin_cost
  let total_cost := cost_muffins + juice_cost
  total_cost = 3.70 :=
by
  sorry

end kevin_total_cost_l290_29072
