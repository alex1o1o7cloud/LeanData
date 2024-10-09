import Mathlib

namespace difference_30th_28th_triangular_l1793_179385

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_30th_28th_triangular :
  triangular_number 30 - triangular_number 28 = 59 :=
by
  sorry

end difference_30th_28th_triangular_l1793_179385


namespace number_of_girls_l1793_179326

theorem number_of_girls (B G: ℕ) 
  (ratio : 8 * G = 5 * B) 
  (total : B + G = 780) :
  G = 300 := 
sorry

end number_of_girls_l1793_179326


namespace remainder_of_1234567_div_123_l1793_179304

theorem remainder_of_1234567_div_123 : 1234567 % 123 = 129 :=
by
  sorry

end remainder_of_1234567_div_123_l1793_179304


namespace pentagon_coloring_l1793_179332

theorem pentagon_coloring (convex : Prop) (unequal_sides : Prop)
  (colors : Prop) (adjacent_diff_color : Prop) :
  ∃ n : ℕ, n = 30 := by
  -- Definitions for conditions (in practical terms, these might need to be more elaborate)
  let convex := true           -- Simplified representation
  let unequal_sides := true    -- Simplified representation
  let colors := true           -- Simplified representation
  let adjacent_diff_color := true -- Simplified representation
  
  -- Proof that the number of coloring methods is 30
  existsi 30
  sorry

end pentagon_coloring_l1793_179332


namespace complex_pow_difference_l1793_179335

theorem complex_pow_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 12 - (1 - i) ^ 12 = 0 :=
  sorry

end complex_pow_difference_l1793_179335


namespace dig_days_l1793_179345

theorem dig_days (m1 m2 : ℕ) (d1 d2 : ℚ) (k : ℚ) 
  (h1 : m1 * d1 = k) (h2 : m2 * d2 = k) : 
  m1 = 30 ∧ d1 = 6 ∧ m2 = 40 → d2 = 4.5 := 
by sorry

end dig_days_l1793_179345


namespace lily_distance_from_start_l1793_179390

open Real

def north_south_net := 40 - 10 -- 30 meters south
def east_west_net := 30 - 15 -- 15 meters east

theorem lily_distance_from_start : 
  ∀ (north_south : ℝ) (east_west : ℝ), 
    north_south = north_south_net → 
    east_west = east_west_net → 
    distance = Real.sqrt ((north_south * north_south) + (east_west * east_west)) → 
    distance = 15 * Real.sqrt 5 :=
by
  intros
  sorry

end lily_distance_from_start_l1793_179390


namespace length_of_segment_P_to_P_l1793_179300

/-- Point P is given as (-4, 3) and P' is the reflection of P over the x-axis. 
    We need to prove that the length of the segment connecting P to P' is 6. -/
theorem length_of_segment_P_to_P' :
  let P := (-4, 3)
  let P' := (-4, -3)
  dist P P' = 6 :=
by
  sorry

end length_of_segment_P_to_P_l1793_179300


namespace square_of_radius_l1793_179379

-- Definitions based on conditions
def ER := 24
def RF := 31
def GS := 40
def SH := 29

-- The goal is to find square of radius r such that r^2 = 841
theorem square_of_radius (r : ℝ) :
  let R := ER
  let F := RF
  let G := GS
  let S := SH
  (∀ r : ℝ, (R + F) * (G + S) = r^2) → r^2 = 841 :=
sorry

end square_of_radius_l1793_179379


namespace solve_equation_l1793_179383

theorem solve_equation : ∀ x : ℝ, (x - (x + 2) / 2 = (2 * x - 1) / 3 - 1) → (x = 2) :=
by
  intros x h
  sorry

end solve_equation_l1793_179383


namespace gcd_1617_1225_gcd_2023_111_gcd_589_6479_l1793_179358

theorem gcd_1617_1225 : Nat.gcd 1617 1225 = 49 :=
by
  sorry

theorem gcd_2023_111 : Nat.gcd 2023 111 = 1 :=
by
  sorry

theorem gcd_589_6479 : Nat.gcd 589 6479 = 589 :=
by
  sorry

end gcd_1617_1225_gcd_2023_111_gcd_589_6479_l1793_179358


namespace correct_operation_l1793_179320

variable (a b : ℝ)

theorem correct_operation : 2 * (a - 1) = 2 * a - 2 :=
sorry

end correct_operation_l1793_179320


namespace min_major_axis_l1793_179370

theorem min_major_axis (a b c : ℝ) (h1 : b * c = 1) (h2 : a = Real.sqrt (b^2 + c^2)) : 2 * a ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_major_axis_l1793_179370


namespace sin_225_eq_neg_sqrt_two_div_two_l1793_179328

theorem sin_225_eq_neg_sqrt_two_div_two :
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt_two_div_two_l1793_179328


namespace book_set_cost_l1793_179308

theorem book_set_cost (charge_per_sqft : ℝ) (lawn_length lawn_width : ℝ) (num_lawns : ℝ) (additional_area : ℝ) (total_cost : ℝ) :
  charge_per_sqft = 0.10 ∧ lawn_length = 20 ∧ lawn_width = 15 ∧ num_lawns = 3 ∧ additional_area = 600 ∧ total_cost = 150 →
  (num_lawns * (lawn_length * lawn_width) * charge_per_sqft + additional_area * charge_per_sqft = total_cost) :=
by
  sorry

end book_set_cost_l1793_179308


namespace parabola_directrix_l1793_179337

theorem parabola_directrix (y : ℝ) (x : ℝ) (h : y = 8 * x^2) : 
  y = -1 / 32 :=
sorry

end parabola_directrix_l1793_179337


namespace pool_filling_water_amount_l1793_179322

theorem pool_filling_water_amount (Tina_pail Tommy_pail Timmy_pail Trudy_pail : ℕ) 
  (h1 : Tina_pail = 4)
  (h2 : Tommy_pail = Tina_pail + 2)
  (h3 : Timmy_pail = 2 * Tommy_pail)
  (h4 : Trudy_pail = (3 * Timmy_pail) / 2)
  (Timmy_trips Trudy_trips Tommy_trips Tina_trips: ℕ)
  (h5 : Timmy_trips = 4)
  (h6 : Trudy_trips = 4)
  (h7 : Tommy_trips = 6)
  (h8 : Tina_trips = 6) :
  Timmy_trips * Timmy_pail + Trudy_trips * Trudy_pail + Tommy_trips * Tommy_pail + Tina_trips * Tina_pail = 180 := by
  sorry

end pool_filling_water_amount_l1793_179322


namespace rectangle_width_squared_l1793_179317

theorem rectangle_width_squared (w l : ℝ) (h1 : w^2 + l^2 = 400) (h2 : 4 * w^2 + l^2 = 484) : w^2 = 28 := 
by
  sorry

end rectangle_width_squared_l1793_179317


namespace area_of_right_triangle_l1793_179302

variable (a b : ℝ)

theorem area_of_right_triangle (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ (S : ℝ), S = a * b :=
sorry

end area_of_right_triangle_l1793_179302


namespace relationship_of_inequalities_l1793_179386

theorem relationship_of_inequalities (a b : ℝ) : 
  ¬ (∀ a b : ℝ, (a > b) → (a^2 > b^2)) ∧ 
  ¬ (∀ a b : ℝ, (a^2 > b^2) → (a > b)) := 
by 
  sorry

end relationship_of_inequalities_l1793_179386


namespace evaluate_expression_l1793_179362

theorem evaluate_expression (x y : ℚ) (hx : x = 4 / 3) (hy : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 13 / 40 :=
by
  rw [hx, hy]
  sorry

end evaluate_expression_l1793_179362


namespace Erik_ate_pie_l1793_179350

theorem Erik_ate_pie (Frank_ate Erik_ate more_than: ℝ) (h1: Frank_ate = 0.3333333333333333)
(h2: more_than = 0.3333333333333333)
(h3: Erik_ate = Frank_ate + more_than) : Erik_ate = 0.6666666666666666 :=
by
  sorry

end Erik_ate_pie_l1793_179350


namespace probability_all_red_is_correct_l1793_179366

def total_marbles (R W B : Nat) : Nat := R + W + B

def first_red_probability (R W B : Nat) : Rat := R / total_marbles R W B
def second_red_probability (R W B : Nat) : Rat := (R - 1) / (total_marbles R W B - 1)
def third_red_probability (R W B : Nat) : Rat := (R - 2) / (total_marbles R W B - 2)

def all_red_probability (R W B : Nat) : Rat := 
  first_red_probability R W B * 
  second_red_probability R W B * 
  third_red_probability R W B

theorem probability_all_red_is_correct 
  (R W B : Nat) (hR : R = 5) (hW : W = 6) (hB : B = 7) :
  all_red_probability R W B = 5 / 408 := by
  sorry

end probability_all_red_is_correct_l1793_179366


namespace smallest_integer_l1793_179303

theorem smallest_integer (k : ℕ) : 
  (∀ (n : ℕ), n = 2^2 * 3^1 * 11^1 → 
  (∀ (f : ℕ), (f = 2^4 ∨ f = 3^3 ∨ f = 13^3) → f ∣ (n * k))) → 
  k = 79092 :=
  sorry

end smallest_integer_l1793_179303


namespace an_expression_l1793_179348

-- Given conditions
def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - n

-- The statement to be proved
theorem an_expression (a : ℕ → ℕ) (n : ℕ) (h_Sn : ∀ n, Sn a n = 2 * a n - n) :
  a n = 2^n - 1 :=
sorry

end an_expression_l1793_179348


namespace sum_remainder_l1793_179319

theorem sum_remainder (n : ℕ) (h : n = 102) :
  ((n * (n + 1) / 2) % 5250) = 3 :=
by
  sorry

end sum_remainder_l1793_179319


namespace remainder_y_div_13_l1793_179334

def x (k : ℤ) : ℤ := 159 * k + 37
def y (x : ℤ) : ℤ := 5 * x^2 + 18 * x + 22

theorem remainder_y_div_13 (k : ℤ) : (y (x k)) % 13 = 8 := by
  sorry

end remainder_y_div_13_l1793_179334


namespace sum_modified_midpoint_coordinates_l1793_179342

theorem sum_modified_midpoint_coordinates :
  let p1 : (ℝ × ℝ) := (10, 3)
  let p2 : (ℝ × ℝ) := (-4, 7)
  let midpoint : (ℝ × ℝ) := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let modified_x := 2 * midpoint.1 
  (modified_x + midpoint.2) = 11 := by
  sorry

end sum_modified_midpoint_coordinates_l1793_179342


namespace problem_l1793_179359

variable (a : Int)
variable (h : -a = 1)

theorem problem : 3 * a - 2 = -5 :=
by
  -- Proof will go here
  sorry

end problem_l1793_179359


namespace factorize_difference_of_squares_factorize_cubic_l1793_179339

-- Problem 1: Prove that 4x^2 - 36 = 4(x + 3)(x - 3)
theorem factorize_difference_of_squares (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) := 
  sorry

-- Problem 2: Prove that x^3 - 2x^2y + xy^2 = x(x - y)^2
theorem factorize_cubic (x y : ℝ) : x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2 := 
  sorry

end factorize_difference_of_squares_factorize_cubic_l1793_179339


namespace center_of_circle_eq_l1793_179306

theorem center_of_circle_eq {x y : ℝ} : (x - 2)^2 + (y - 3)^2 = 1 → (x, y) = (2, 3) :=
by
  intro h
  sorry

end center_of_circle_eq_l1793_179306


namespace find_piles_l1793_179340

theorem find_piles :
  ∃ N : ℕ, 
  (1000 < N ∧ N < 2000) ∧ 
  (N % 2 = 1) ∧ (N % 3 = 1) ∧ (N % 4 = 1) ∧ 
  (N % 5 = 1) ∧ (N % 6 = 1) ∧ (N % 7 = 1) ∧ (N % 8 = 1) ∧ 
  (∃ p : ℕ, p = 41 ∧ p > 1 ∧ p < N ∧ N % p = 0) :=
sorry

end find_piles_l1793_179340


namespace number_of_mowers_l1793_179327

noncomputable section

def area_larger_meadow (A : ℝ) : ℝ := 2 * A

def team_half_day_work (K a : ℝ) : ℝ := (K * a) / 2

def team_remaining_larger_meadow (K a : ℝ) : ℝ := (K * a) / 2

def half_team_half_day_work (K a : ℝ) : ℝ := (K * a) / 4

def larger_meadow_area_leq_sum (K a A : ℝ) : Prop :=
  team_half_day_work K a + team_remaining_larger_meadow K a = 2 * A

def smaller_meadow_area_left (K a A : ℝ) : ℝ :=
  A - half_team_half_day_work K a

def one_mower_one_day_work_rate (K a : ℝ) : ℝ := (K * a) / 4

def eq_total_mowed_by_team (K a A : ℝ) : Prop :=
  larger_meadow_area_leq_sum K a A ∧ smaller_meadow_area_left K a A = (K * a) / 4

theorem number_of_mowers
  (K a A b : ℝ)
  (h1 : larger_meadow_area_leq_sum K a A)
  (h2 : smaller_meadow_area_left K a A = one_mower_one_day_work_rate K a)
  (h3 : one_mower_one_day_work_rate K a = b)
  (h4 : K * a = 2 * A)
  (h5 : 2 * A = 4 * b)
  : K = 8 :=
  sorry

end number_of_mowers_l1793_179327


namespace exists_integer_in_seq_l1793_179323

noncomputable def x_seq (x : ℕ → ℚ) := ∀ n : ℕ, x (n + 1) = x n + 1 / ⌊x n⌋

theorem exists_integer_in_seq {x : ℕ → ℚ} (h1 : 1 < x 1) (h2 : x_seq x) : 
  ∃ n : ℕ, ∃ k : ℤ, x n = k :=
sorry

end exists_integer_in_seq_l1793_179323


namespace john_has_22_dimes_l1793_179318

theorem john_has_22_dimes (d q : ℕ) (h1 : d = q + 4) (h2 : 10 * d + 25 * q = 680) : d = 22 :=
by
sorry

end john_has_22_dimes_l1793_179318


namespace reciprocal_of_sum_l1793_179381

theorem reciprocal_of_sum :
  (1 / ((3 : ℚ) / 4 + (5 : ℚ) / 6)) = (12 / 19) :=
by
  sorry

end reciprocal_of_sum_l1793_179381


namespace product_evaluation_l1793_179343

theorem product_evaluation (a b c : ℕ) (h : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) :
  6 * 15 * 2 = 4 := by
  sorry

end product_evaluation_l1793_179343


namespace find_speed_of_man_in_still_water_l1793_179352

def speed_of_man_in_still_water (t1 t2 d1 d2: ℝ) (v_m v_s: ℝ) : Prop :=
  d1 / t1 = v_m + v_s ∧ d2 / t2 = v_m - v_s

theorem find_speed_of_man_in_still_water :
  ∃ v_m : ℝ, ∃ v_s : ℝ, speed_of_man_in_still_water 2 2 16 10 v_m v_s ∧ v_m = 6.5 :=
by
  sorry

end find_speed_of_man_in_still_water_l1793_179352


namespace males_in_sample_l1793_179365

theorem males_in_sample (total_employees female_employees sample_size : ℕ) 
  (h1 : total_employees = 300)
  (h2 : female_employees = 160)
  (h3 : sample_size = 15)
  (h4 : (female_employees * sample_size) / total_employees = 8) :
  sample_size - ((female_employees * sample_size) / total_employees) = 7 :=
by
  sorry

end males_in_sample_l1793_179365


namespace parallelogram_area_l1793_179301

noncomputable def angle_ABC : ℝ := 30
noncomputable def AX : ℝ := 20
noncomputable def CY : ℝ := 22

theorem parallelogram_area (angle_ABC_eq : angle_ABC = 30)
    (AX_eq : AX = 20)
    (CY_eq : CY = 22)
    : ∃ (BC : ℝ), (BC * AX = 880) := sorry

end parallelogram_area_l1793_179301


namespace ordering_of_xyz_l1793_179367

theorem ordering_of_xyz :
  let x := Real.sqrt 3
  let y := Real.log 2 / Real.log 3
  let z := Real.cos 2
  z < y ∧ y < x :=
by
  let x := Real.sqrt 3
  let y := Real.log 2 / Real.log 3
  let z := Real.cos 2
  sorry

end ordering_of_xyz_l1793_179367


namespace number_of_ways_to_write_528_as_sum_of_consecutive_integers_l1793_179341

theorem number_of_ways_to_write_528_as_sum_of_consecutive_integers : 
  ∃ (n : ℕ), (2 ≤ n ∧ ∃ k : ℕ, n * (2 * k + n - 1) = 1056) ∧ n = 15 :=
by
  sorry

end number_of_ways_to_write_528_as_sum_of_consecutive_integers_l1793_179341


namespace research_question_correct_survey_method_correct_l1793_179368

-- Define the conditions.
def total_students : Nat := 400
def sampled_students : Nat := 80

-- Define the research question.
def research_question : String := "To understand the vision conditions of 400 eighth-grade students in a certain school."

-- Define the survey method.
def survey_method : String := "A sampling survey method was used."

-- Prove the research_question matches the expected question given the conditions.
theorem research_question_correct :
  research_question = "To understand the vision conditions of 400 eighth-grade students in a certain school" := by
  sorry

-- Prove the survey method used matches the expected method given the conditions.
theorem survey_method_correct :
  survey_method = "A sampling survey method was used" := by
  sorry

end research_question_correct_survey_method_correct_l1793_179368


namespace cantor_length_formula_l1793_179376

noncomputable def cantor_length : ℕ → ℚ
| 0 => 1
| (n+1) => 2/3 * cantor_length n

theorem cantor_length_formula (n : ℕ) : cantor_length n = (2/3 : ℚ)^(n-1) :=
  sorry

end cantor_length_formula_l1793_179376


namespace articles_produced_l1793_179391

theorem articles_produced (x y z w : ℕ) :
  (x ≠ 0) → (y ≠ 0) → (z ≠ 0) → (w ≠ 0) →
  ((x * x * x * (1 / x^2) = x) →
  y * z * w * (1 / x^2) = y * z * w / x^2) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end articles_produced_l1793_179391


namespace units_digit_of_17_pow_28_l1793_179333

theorem units_digit_of_17_pow_28 : ((17:ℕ) ^ 28) % 10 = 1 := by
  sorry

end units_digit_of_17_pow_28_l1793_179333


namespace catering_service_comparison_l1793_179313

theorem catering_service_comparison :
  ∃ (x : ℕ), 150 + 18 * x > 250 + 15 * x ∧ (∀ y : ℕ, y < x -> (150 + 18 * y ≤ 250 + 15 * y)) ∧ x = 34 :=
sorry

end catering_service_comparison_l1793_179313


namespace find_number_l1793_179330

def problem (x : ℝ) : Prop :=
  0.25 * x = 130 + 190

theorem find_number (x : ℝ) (h : problem x) : x = 1280 :=
by 
  sorry

end find_number_l1793_179330


namespace blue_marbles_in_bag_l1793_179347

theorem blue_marbles_in_bag
  (total_marbles : ℕ)
  (red_marbles : ℕ)
  (prob_red_white : ℚ)
  (number_red_marbles: red_marbles = 9) 
  (total_marbles_eq: total_marbles = 30) 
  (prob_red_white_eq: prob_red_white = 5/6): 
  ∃ (blue_marbles : ℕ), blue_marbles = 5 :=
by
  have W := 16        -- This is from (9 + W)/30 = 5/6 which gives W = 16
  let B := total_marbles - red_marbles - W
  use B
  have h : B = 30 - 9 - 16 := by
    -- Remaining calculations
    sorry
  exact h

end blue_marbles_in_bag_l1793_179347


namespace coefficients_divisible_by_seven_l1793_179354

theorem coefficients_divisible_by_seven {a b c d e : ℤ}
  (h : ∀ x : ℤ, (a * x^4 + b * x^3 + c * x^2 + d * x + e) % 7 = 0) :
  a % 7 = 0 ∧ b % 7 = 0 ∧ c % 7 = 0 ∧ d % 7 = 0 ∧ e % 7 = 0 := 
  sorry

end coefficients_divisible_by_seven_l1793_179354


namespace sharon_trip_distance_l1793_179312

noncomputable def usual_speed (x : ℝ) : ℝ := x / 180
noncomputable def reduced_speed (x : ℝ) : ℝ := usual_speed x - 25 / 60
noncomputable def increased_speed (x : ℝ) : ℝ := usual_speed x + 10 / 60
noncomputable def pre_storm_time : ℝ := 60
noncomputable def total_time : ℝ := 300

theorem sharon_trip_distance : 
  ∀ (x : ℝ), 
  60 + (x / 3) / reduced_speed x + (x / 3) / increased_speed x = 240 → 
  x = 135 :=
sorry

end sharon_trip_distance_l1793_179312


namespace max_students_distribute_eq_pens_pencils_l1793_179353

theorem max_students_distribute_eq_pens_pencils (n_pens n_pencils n : ℕ) (h_pens : n_pens = 890) (h_pencils : n_pencils = 630) :
  (∀ k : ℕ, k > n → (n_pens % k ≠ 0 ∨ n_pencils % k ≠ 0)) → (n = Nat.gcd n_pens n_pencils) := by
  sorry

end max_students_distribute_eq_pens_pencils_l1793_179353


namespace fraction_expression_eq_l1793_179357

theorem fraction_expression_eq (x y : ℕ) (hx : x = 4) (hy : y = 5) : 
  ((1 / y) + (1 / x)) / (1 / x) = 9 / 5 :=
by
  rw [hx, hy]
  sorry

end fraction_expression_eq_l1793_179357


namespace range_of_a_l1793_179325

variable (f : ℝ → ℝ)

-- f is an odd function
def odd_function : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition 1: f is an odd function
axiom h_odd : odd_function f

-- Condition 2: f(x) + f(x + 3 / 2) = 0 for any real number x
axiom h_periodicity : ∀ x : ℝ, f x + f (x + 3 / 2) = 0

-- Condition 3: f(1) > 1
axiom h_f1 : f 1 > 1

-- Condition 4: f(2) = a for some real number a
variable (a : ℝ)
axiom h_f2 : f 2 = a

-- Goal: Prove that a < -1
theorem range_of_a : a < -1 :=
  sorry

end range_of_a_l1793_179325


namespace find_integers_l1793_179310

theorem find_integers (a b c : ℤ) (h1 : ∃ x : ℤ, a = 2 * x ∧ b = 5 * x ∧ c = 8 * x)
  (h2 : a + 6 = b / 3)
  (h3 : c - 10 = 5 * a / 4) :
  a = 36 ∧ b = 90 ∧ c = 144 :=
by
  sorry

end find_integers_l1793_179310


namespace tens_digit_of_7_pow_35_l1793_179364

theorem tens_digit_of_7_pow_35 : 
  (7 ^ 35) % 100 / 10 % 10 = 4 :=
by
  sorry

end tens_digit_of_7_pow_35_l1793_179364


namespace pow_fraction_eq_l1793_179338

theorem pow_fraction_eq : (4:ℕ) = 2^2 ∧ (8:ℕ) = 2^3 → (4^800 / 8^400 = 2^400) :=
by
  -- proof steps should go here, but they are omitted as per the instruction
  sorry

end pow_fraction_eq_l1793_179338


namespace initial_amount_of_liquid_A_l1793_179371

theorem initial_amount_of_liquid_A (A B : ℕ) (x : ℕ) (h1 : 4 * x = A) (h2 : x = B) (h3 : 4 * x + x = 5 * x)
    (h4 : 4 * x - 8 = 3 * (x + 8) / 2) : A = 16 :=
  by
  sorry

end initial_amount_of_liquid_A_l1793_179371


namespace four_digit_swap_square_l1793_179398

theorem four_digit_swap_square (a b : ℤ) (N M : ℤ) : 
  N = 1111 * a + 123 ∧ 
  M = 1111 * a + 1023 ∧ 
  M = b ^ 2 → 
  N = 3456 := 
by sorry

end four_digit_swap_square_l1793_179398


namespace price_of_each_toy_l1793_179399

variables (T : ℝ)

-- Given conditions
def total_cost (T : ℝ) : ℝ := 3 * T + 2 * 5 + 5 * 6

theorem price_of_each_toy :
  total_cost T = 70 → T = 10 :=
sorry

end price_of_each_toy_l1793_179399


namespace ordered_pair_A_B_l1793_179384

noncomputable def cubic_function (x : ℝ) : ℝ := x^3 - 2 * x^2 - 3 * x + 6
noncomputable def linear_function (x : ℝ) : ℝ := -2 / 3 * x + 2

noncomputable def points_intersect (x1 x2 x3 y1 y2 y3 : ℝ) : Prop :=
  cubic_function x1 = y1 ∧ cubic_function x2 = y2 ∧ cubic_function x3 = y3 ∧
  2 * x1 + 3 * y1 = 6 ∧ 2 * x2 + 3 * y2 = 6 ∧ 2 * x3 + 3 * y3 = 6

theorem ordered_pair_A_B (x1 x2 x3 y1 y2 y3 A B : ℝ)
  (h_intersect : points_intersect x1 x2 x3 y1 y2 y3) 
  (h_sum_x : x1 + x2 + x3 = A)
  (h_sum_y : y1 + y2 + y3 = B) :
  (A, B) = (2, 14 / 3) :=
by {
  sorry
}

end ordered_pair_A_B_l1793_179384


namespace pugs_working_together_l1793_179356

theorem pugs_working_together (P : ℕ) (H1 : P * 45 = 15 * 12) : P = 4 :=
by {
  sorry
}

end pugs_working_together_l1793_179356


namespace average_salary_l1793_179387

theorem average_salary (total_workers technicians other_workers technicians_avg_salary other_workers_avg_salary total_salary : ℝ)
  (h_workers : total_workers = 21)
  (h_technicians : technicians = 7)
  (h_other_workers : other_workers = total_workers - technicians)
  (h_technicians_avg_salary : technicians_avg_salary = 12000)
  (h_other_workers_avg_salary : other_workers_avg_salary = 6000)
  (h_total_technicians_salary : total_salary = (technicians * technicians_avg_salary + other_workers * other_workers_avg_salary))
  (h_total_other_salary : total_salary = 168000) :
  total_salary / total_workers = 8000 := by
    sorry

end average_salary_l1793_179387


namespace plane_divides_space_into_two_parts_l1793_179316

def divides_space : Prop :=
  ∀ (P : ℝ → ℝ → ℝ → Prop), (∀ x y z, P x y z → P x y z) →
  (∃ region1 region2 : ℝ → ℝ → ℝ → Prop,
    (∀ x y z, P x y z → (region1 x y z ∨ region2 x y z)) ∧
    (∀ x y z, region1 x y z → ¬region2 x y z) ∧
    (∃ x1 y1 z1 x2 y2 z2, region1 x1 y1 z1 ∧ region2 x2 y2 z2))

theorem plane_divides_space_into_two_parts (P : ℝ → ℝ → ℝ → Prop) (hP : ∀ x y z, P x y z → P x y z) : 
  divides_space :=
  sorry

end plane_divides_space_into_two_parts_l1793_179316


namespace term_is_18_minimum_value_l1793_179311

-- Define the sequence a_n
def a_n (n : ℕ) : ℤ := n^2 - 5 * n + 4

-- Prove that a_n = 18 implies n = 7
theorem term_is_18 (n : ℕ) (h : a_n n = 18) : n = 7 := 
by 
  sorry

-- Prove that the minimum value of a_n is -2 and it occurs at n = 2 or n = 3
theorem minimum_value (n : ℕ) : n = 2 ∨ n = 3 ∧ a_n n = -2 :=
by 
  sorry

end term_is_18_minimum_value_l1793_179311


namespace find_a_l1793_179346

theorem find_a (a: ℕ) : (2000 + 100 * a + 17) % 19 = 0 ↔ a = 7 :=
by
  sorry

end find_a_l1793_179346


namespace grims_groks_zeets_l1793_179397

variable {T : Type}
variable (Groks Zeets Grims Snarks : Set T)

-- Given conditions as definitions in Lean 4
variable (h1 : Groks ⊆ Zeets)
variable (h2 : Grims ⊆ Zeets)
variable (h3 : Snarks ⊆ Groks)
variable (h4 : Grims ⊆ Snarks)

-- The statement to be proved
theorem grims_groks_zeets : Grims ⊆ Groks ∧ Grims ⊆ Zeets := by
  sorry

end grims_groks_zeets_l1793_179397


namespace fencing_rate_l1793_179389

noncomputable def rate_per_meter (d : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := Real.pi * d
  total_cost / circumference

theorem fencing_rate (diameter cost : ℝ) (h₀ : diameter = 34) (h₁ : cost = 213.63) :
  rate_per_meter diameter cost = 2 := by
  sorry

end fencing_rate_l1793_179389


namespace initial_mixtureA_amount_l1793_179378

-- Condition 1: Mixture A is 20% oil and 80% material B by weight.
def oil_content (x : ℝ) : ℝ := 0.20 * x
def materialB_content (x : ℝ) : ℝ := 0.80 * x

-- Condition 2: 2 more kilograms of oil are added to a certain amount of mixture A
def oil_added := 2

-- Condition 3: 6 kilograms of mixture A must be added to make a 70% material B in the new mixture.
def mixture_added := 6

-- The total weight of the new mixture
def total_weight (x : ℝ) : ℝ := x + mixture_added + oil_added

-- The total amount of material B in the new mixture
def total_materialB (x : ℝ) : ℝ := 0.80 * x + 0.80 * mixture_added

-- The new mixture is supposed to be 70% material B.
def is_70_percent_materialB (x : ℝ) : Prop := total_materialB x = 0.70 * total_weight x

-- Proving x == 8 given the conditions
theorem initial_mixtureA_amount : ∃ x : ℝ, is_70_percent_materialB x ∧ x = 8 :=
by
  sorry

end initial_mixtureA_amount_l1793_179378


namespace find_m_l1793_179336

theorem find_m (m : ℕ) (h₁ : 256 = 4^4) : (256 : ℝ)^(1/4) = (4 : ℝ)^m ↔ m = 1 :=
by
  sorry

end find_m_l1793_179336


namespace divisibility_1989_l1793_179344

theorem divisibility_1989 (n : ℕ) (h1 : n ≥ 3) :
  1989 ∣ n^(n^(n^n)) - n^(n^n) :=
sorry

end divisibility_1989_l1793_179344


namespace smallest_c_minus_a_l1793_179388

theorem smallest_c_minus_a (a b c : ℕ) (h1 : a * b * c = 720) (h2 : a < b) (h3 : b < c) : c - a ≥ 24 :=
sorry

end smallest_c_minus_a_l1793_179388


namespace cos_in_third_quadrant_l1793_179307

theorem cos_in_third_quadrant (B : ℝ) (h_sin_B : Real.sin B = -5/13) (h_quadrant : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 :=
by
  sorry

end cos_in_third_quadrant_l1793_179307


namespace find_smallest_n_l1793_179324

-- Define costs and relationships
def cost_red (r : ℕ) : ℕ := 10 * r
def cost_green (g : ℕ) : ℕ := 18 * g
def cost_blue (b : ℕ) : ℕ := 20 * b
def cost_purple (n : ℕ) : ℕ := 24 * n

-- Define the mathematical problem
theorem find_smallest_n (r g b : ℕ) :
  ∃ n : ℕ, 24 * n = Nat.lcm (cost_red r) (Nat.lcm (cost_green g) (cost_blue b)) ∧ n = 15 :=
by
  sorry

end find_smallest_n_l1793_179324


namespace total_spent_l1793_179363

theorem total_spent (cost_per_deck : ℕ) (decks_frank : ℕ) (decks_friend : ℕ) (total : ℕ) : 
  cost_per_deck = 7 → 
  decks_frank = 3 → 
  decks_friend = 2 → 
  total = (decks_frank * cost_per_deck) + (decks_friend * cost_per_deck) → 
  total = 35 :=
by
  sorry

end total_spent_l1793_179363


namespace find_multiple_of_number_l1793_179369

theorem find_multiple_of_number (n : ℝ) (m : ℝ) (h1 : n ≠ 0) (h2 : n = 9) (h3 : (n + n^2) / 2 = m * n) : m = 5 :=
sorry

end find_multiple_of_number_l1793_179369


namespace isosceles_trapezoid_height_l1793_179382

theorem isosceles_trapezoid_height (S h : ℝ) (h_nonneg : 0 ≤ h) 
  (diag_perpendicular : S = (1 / 2) * h^2) : h = Real.sqrt S :=
by
  sorry

end isosceles_trapezoid_height_l1793_179382


namespace integer_satisfying_conditions_l1793_179373

theorem integer_satisfying_conditions :
  {a : ℤ | 1 ≤ a ∧ a ≤ 105 ∧ 35 ∣ (a^3 - 1)} = {1, 11, 16, 36, 46, 51, 71, 81, 86} :=
by
  sorry

end integer_satisfying_conditions_l1793_179373


namespace sine_thirteen_pi_over_six_l1793_179360

theorem sine_thirteen_pi_over_six : Real.sin ((13 * Real.pi) / 6) = 1 / 2 := by
  sorry

end sine_thirteen_pi_over_six_l1793_179360


namespace loan_to_scholarship_ratio_l1793_179331

noncomputable def tuition := 22000
noncomputable def parents_contribution := tuition / 2
noncomputable def scholarship := 3000
noncomputable def wage_per_hour := 10
noncomputable def working_hours := 200
noncomputable def earnings := wage_per_hour * working_hours
noncomputable def total_scholarship_and_work := scholarship + earnings
noncomputable def remaining_tuition := tuition - parents_contribution - total_scholarship_and_work
noncomputable def student_loan := remaining_tuition

theorem loan_to_scholarship_ratio :
  (student_loan / scholarship) = 2 := 
by
  sorry

end loan_to_scholarship_ratio_l1793_179331


namespace stock_percentage_l1793_179351

theorem stock_percentage (investment income : ℝ) (investment total : ℝ) (P : ℝ) : 
  (income = 3800) → (total = 15200) → (income = (total * P) / 100) → P = 25 :=
by
  intros h1 h2 h3
  sorry

end stock_percentage_l1793_179351


namespace find_C_line_MN_l1793_179380

def point := (ℝ × ℝ)

-- Given points A and B
def A : point := (5, -2)
def B : point := (7, 3)

-- Conditions: M is the midpoint of AC and is on the y-axis
def M_on_y_axis (M : point) (A C : point) : Prop :=
  M.1 = 0 ∧ M.2 = (A.2 + C.2) / 2

-- Conditions: N is the midpoint of BC and is on the x-axis
def N_on_x_axis (N : point) (B C : point) : Prop :=
  N.1 = (B.1 + C.1) / 2 ∧ N.2 = 0

-- Coordinates of point C
theorem find_C (C : point)
  (M : point) (N : point)
  (hM : M_on_y_axis M A C)
  (hN : N_on_x_axis N B C) : C = (-5, -8) := sorry

-- Equation of line MN
theorem line_MN (M N : point)
  (MN_eq : M_on_y_axis M A (-5, -8) ∧ N_on_x_axis N B (-5, -8)) :
   ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ ((y = M.2) ∧ (x = M.1)) ∨ ((y = N.2) ∧ (x = N.1))) ∧ m = (3/2) ∧ b = 0 := sorry

end find_C_line_MN_l1793_179380


namespace expected_sectors_pizza_l1793_179321

/-- Let N be the total number of pizza slices and M be the number of slices taken randomly.
    Given N = 16 and M = 5, the expected number of sectors formed is 11/3. -/
theorem expected_sectors_pizza (N M : ℕ) (hN : N = 16) (hM : M = 5) :
  (N - M) * M / (N - 1) = 11 / 3 :=
  sorry

end expected_sectors_pizza_l1793_179321


namespace carl_candy_bars_l1793_179396

/-- 
Carl earns $0.75 every week for taking out his neighbor's trash. 
Carl buys a candy bar every time he earns $0.50. 
After four weeks, Carl will be able to buy 6 candy bars.
-/
theorem carl_candy_bars :
  (0.75 * 4) / 0.50 = 6 := 
  by
    sorry

end carl_candy_bars_l1793_179396


namespace find_P_l1793_179314

variable (P : ℕ) 

-- Conditions
def cost_samosas : ℕ := 3 * 2
def cost_mango_lassi : ℕ := 2
def cost_per_pakora : ℕ := 3
def total_cost : ℕ := 25
def tip_rate : ℚ := 0.25

-- Total cost before tip
def total_cost_before_tip (P : ℕ) : ℕ := cost_samosas + cost_mango_lassi + cost_per_pakora * P

-- Total cost with tip
def total_cost_with_tip (P : ℕ) : ℚ := 
  (total_cost_before_tip P : ℚ) + (tip_rate * total_cost_before_tip P : ℚ)

-- Proof Goal
theorem find_P (h : total_cost_with_tip P = total_cost) : P = 4 :=
by
  sorry

end find_P_l1793_179314


namespace count_shapes_in_figure_l1793_179372

-- Definitions based on the conditions
def firstLayerTriangles : Nat := 3
def secondLayerSquares : Nat := 2
def thirdLayerLargeTriangle : Nat := 1
def totalSmallTriangles := firstLayerTriangles
def totalLargeTriangles := thirdLayerLargeTriangle
def totalTriangles := totalSmallTriangles + totalLargeTriangles
def totalSquares := secondLayerSquares

-- Lean 4 statement to prove the problem
theorem count_shapes_in_figure : totalTriangles = 4 ∧ totalSquares = 2 :=
by {
  -- The proof is not required, so we use sorry to skip it.
  sorry
}

end count_shapes_in_figure_l1793_179372


namespace find_xz_l1793_179329

theorem find_xz (x y z : ℝ) (h1 : 2 * x + z = 15) (h2 : x - 2 * y = 8) : x + z = 15 :=
sorry

end find_xz_l1793_179329


namespace problem_solutions_l1793_179375

theorem problem_solutions (a b c : ℝ) (h : ∀ x, ax^2 + bx + c ≤ 0 ↔ x ≤ -4 ∨ x ≥ 3) :
  (a + b + c > 0) ∧ (∀ x, bx + c > 0 ↔ x < 12) :=
by
  -- The following proof steps are not needed as per the instructions provided
  sorry

end problem_solutions_l1793_179375


namespace problem_l1793_179349

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 5}
def C : Set ℕ := {1, 3}

theorem problem : A ∩ (U \ B) = C := by
  sorry

end problem_l1793_179349


namespace Jason_cards_l1793_179355

theorem Jason_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 3) (h2 : cards_bought = 2) : remaining_cards = 1 :=
by
  sorry

end Jason_cards_l1793_179355


namespace probability_of_blue_or_yellow_l1793_179309

def num_red : ℕ := 6
def num_green : ℕ := 7
def num_yellow : ℕ := 8
def num_blue : ℕ := 9

def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue
def total_blue_or_yellow : ℕ := num_yellow + num_blue

theorem probability_of_blue_or_yellow (h : total_jelly_beans ≠ 0) : 
  (total_blue_or_yellow : ℚ) / (total_jelly_beans : ℚ) = 17 / 30 :=
by
  sorry

end probability_of_blue_or_yellow_l1793_179309


namespace arithmetic_sequence_example_l1793_179305

theorem arithmetic_sequence_example 
    (a : ℕ → ℤ) 
    (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) 
    (h2 : a 1 + a 4 + a 7 = 45) 
    (h3 : a 2 + a 5 + a 8 = 39) :
    a 3 + a 6 + a 9 = 33 :=
sorry

end arithmetic_sequence_example_l1793_179305


namespace waiter_tables_l1793_179393

/-
Problem:
A waiter had 22 customers in his section.
14 of them left.
The remaining customers were seated at tables with 4 people per table.
Prove the number of tables is 2.
-/

theorem waiter_tables:
  ∃ (tables : ℤ), 
    (∀ (customers_initial customers_remaining people_per_table tables_calculated : ℤ), 
      customers_initial = 22 →
      customers_remaining = customers_initial - 14 →
      people_per_table = 4 →
      tables_calculated = customers_remaining / people_per_table →
      tables = tables_calculated) →
    tables = 2 :=
by
  sorry

end waiter_tables_l1793_179393


namespace find_x_minus_y_l1793_179377

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end find_x_minus_y_l1793_179377


namespace multiplication_addition_l1793_179392

theorem multiplication_addition :
  23 * 37 + 16 = 867 :=
by
  sorry

end multiplication_addition_l1793_179392


namespace minimum_value_expr_l1793_179361

theorem minimum_value_expr (x y : ℝ) : 
  ∃ (a b : ℝ), 2 * x^2 + 3 * y^2 - 12 * x + 6 * y + 25 = 2 * (a - 3)^2 + 3 * (b + 1)^2 + 4 ∧ 
  2 * (a - 3)^2 + 3 * (b + 1)^2 + 4 ≥ 4 :=
by 
  sorry

end minimum_value_expr_l1793_179361


namespace geometric_sequence_sum_l1793_179374

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 = 3)
  (h_sum : a 1 + a 3 + a 5 = 21) : 
  a 3 + a 5 + a 7 = 42 :=
sorry

end geometric_sequence_sum_l1793_179374


namespace isosceles_triangle_smallest_angle_l1793_179395

def is_isosceles (angle_A angle_B angle_C : ℝ) : Prop := 
(angle_A = angle_B) ∨ (angle_B = angle_C) ∨ (angle_C = angle_A)

theorem isosceles_triangle_smallest_angle
  (angle_A angle_B angle_C : ℝ)
  (h_isosceles : is_isosceles angle_A angle_B angle_C)
  (h_angle_162 : angle_A = 162) :
  angle_B = 9 ∧ angle_C = 9 ∨ angle_A = 9 ∧ (angle_B = 9 ∨ angle_C = 9) :=
by
  sorry

end isosceles_triangle_smallest_angle_l1793_179395


namespace sum_of_roots_quadratic_l1793_179394

theorem sum_of_roots_quadratic :
  ∀ (a b : ℝ), (a^2 - a - 2 = 0) → (b^2 - b - 2 = 0) → (a + b = 1) :=
by
  intro a b
  intros
  sorry

end sum_of_roots_quadratic_l1793_179394


namespace route_down_distance_l1793_179315

theorem route_down_distance :
  ∀ (rate_up rate_down time_up time_down distance_up distance_down : ℝ),
    -- Conditions
    rate_down = 1.5 * rate_up →
    time_up = time_down →
    rate_up = 6 →
    time_up = 2 →
    distance_up = rate_up * time_up →
    distance_down = rate_down * time_down →
    -- Question: Prove the correct answer
    distance_down = 18 :=
by
  intros rate_up rate_down time_up time_down distance_up distance_down h1 h2 h3 h4 h5 h6
  sorry

end route_down_distance_l1793_179315
