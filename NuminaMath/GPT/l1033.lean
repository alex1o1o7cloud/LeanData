import Mathlib

namespace NUMINAMATH_GPT_first_machine_defect_probability_l1033_103358

/-- Probability that a randomly selected defective item was made by the first machine is 0.5 
given certain conditions. -/
theorem first_machine_defect_probability :
  let PFirstMachine := 0.4
  let PSecondMachine := 0.6
  let DefectRateFirstMachine := 0.03
  let DefectRateSecondMachine := 0.02
  let TotalDefectProbability := PFirstMachine * DefectRateFirstMachine + PSecondMachine * DefectRateSecondMachine
  let PDefectGivenFirstMachine := PFirstMachine * DefectRateFirstMachine / TotalDefectProbability
  PDefectGivenFirstMachine = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_first_machine_defect_probability_l1033_103358


namespace NUMINAMATH_GPT_solve_speeds_ratio_l1033_103376

noncomputable def speeds_ratio (v_A v_B : ℝ) : Prop :=
  v_A / v_B = 1 / 3

theorem solve_speeds_ratio (v_A v_B : ℝ) (h1 : ∃ t : ℝ, t = 1 ∧ v_A = 300 - v_B ∧ v_A = v_B ∧ v_B = 300) 
  (h2 : ∃ t : ℝ, t = 7 ∧ 7 * v_A = 300 - 7 * v_B ∧ 7 * v_A = 300 - v_B ∧ 7 * v_B = v_A): 
    speeds_ratio v_A v_B :=
sorry

end NUMINAMATH_GPT_solve_speeds_ratio_l1033_103376


namespace NUMINAMATH_GPT_man_speed_in_still_water_l1033_103379

theorem man_speed_in_still_water (c_speed : ℝ) (distance_m : ℝ) (time_sec : ℝ) (downstream_distance_km : ℝ) (downstream_time_hr : ℝ) :
    c_speed = 3 →
    distance_m = 15 →
    time_sec = 2.9997600191984644 →
    downstream_distance_km = distance_m / 1000 →
    downstream_time_hr = time_sec / 3600 →
    (downstream_distance_km / downstream_time_hr) - c_speed = 15 :=
by
  intros hc hd ht hdownstream_distance hdownstream_time 
  sorry

end NUMINAMATH_GPT_man_speed_in_still_water_l1033_103379


namespace NUMINAMATH_GPT_reduce_consumption_percentage_l1033_103392

theorem reduce_consumption_percentage :
  ∀ (current_rate old_rate : ℝ), 
  current_rate = 20 → 
  old_rate = 16 → 
  ((current_rate - old_rate) / old_rate * 100) = 25 :=
by
  intros current_rate old_rate h_current h_old
  sorry

end NUMINAMATH_GPT_reduce_consumption_percentage_l1033_103392


namespace NUMINAMATH_GPT_car_tank_capacity_l1033_103316

theorem car_tank_capacity
  (speed : ℝ) (usage_rate : ℝ) (time : ℝ) (used_fraction : ℝ) (distance : ℝ := speed * time) (gallons_used : ℝ := distance / usage_rate) 
  (fuel_used : ℝ := 10) (tank_capacity : ℝ := fuel_used / used_fraction)
  (h1 : speed = 60) (h2 : usage_rate = 30) (h3 : time = 5) (h4 : used_fraction = 0.8333333333333334) : 
  tank_capacity = 12 :=
by
  sorry

end NUMINAMATH_GPT_car_tank_capacity_l1033_103316


namespace NUMINAMATH_GPT_alloy_chromium_l1033_103352

variable (x : ℝ)

theorem alloy_chromium (h : 0.15 * 15 + 0.08 * x = 0.101 * (15 + x)) : x = 35 := by
  sorry

end NUMINAMATH_GPT_alloy_chromium_l1033_103352


namespace NUMINAMATH_GPT_fraction_simplification_l1033_103342

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  (2 * x - 5) / (x ^ 2 - 1) + 3 / (1 - x) = - (x + 8) / (x ^ 2 - 1) :=
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1033_103342


namespace NUMINAMATH_GPT_fraction_used_first_day_l1033_103363

theorem fraction_used_first_day (x : ℝ) :
  let initial_supplies := 400
  let supplies_remaining_after_first_day := initial_supplies * (1 - x)
  let supplies_remaining_after_three_days := (2/5 : ℝ) * supplies_remaining_after_first_day
  supplies_remaining_after_three_days = 96 → 
  x = (2/5 : ℝ) :=
by
  intros
  sorry

end NUMINAMATH_GPT_fraction_used_first_day_l1033_103363


namespace NUMINAMATH_GPT_sqrt_product_l1033_103308

theorem sqrt_product (h54 : Real.sqrt 54 = 3 * Real.sqrt 6)
                     (h32 : Real.sqrt 32 = 4 * Real.sqrt 2)
                     (h6 : Real.sqrt 6 = Real.sqrt 6) :
    Real.sqrt 54 * Real.sqrt 32 * Real.sqrt 6 = 72 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_sqrt_product_l1033_103308


namespace NUMINAMATH_GPT_complex_conjugate_x_l1033_103341

theorem complex_conjugate_x (x : ℝ) (h : x^2 + x - 2 + (x^2 - 3 * x + 2 : ℂ) * Complex.I = 4 + 20 * Complex.I) : x = -3 := sorry

end NUMINAMATH_GPT_complex_conjugate_x_l1033_103341


namespace NUMINAMATH_GPT_square_diagonal_l1033_103329

theorem square_diagonal (p : ℤ) (h : p = 28) : ∃ d : ℝ, d = 7 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_square_diagonal_l1033_103329


namespace NUMINAMATH_GPT_ellie_oil_needs_l1033_103387

def oil_per_wheel : ℕ := 10
def number_of_wheels : ℕ := 2
def oil_for_rest : ℕ := 5
def total_oil_needed : ℕ := oil_per_wheel * number_of_wheels + oil_for_rest

theorem ellie_oil_needs : total_oil_needed = 25 := by
  sorry

end NUMINAMATH_GPT_ellie_oil_needs_l1033_103387


namespace NUMINAMATH_GPT_truncated_cone_sphere_radius_l1033_103326

noncomputable def radius_of_sphere (r1 r2 h : ℝ) : ℝ := 
  (Real.sqrt (h^2 + (r1 - r2)^2)) / 2

theorem truncated_cone_sphere_radius : 
  ∀ (r1 r2 h : ℝ), r1 = 20 → r2 = 6 → h = 15 → radius_of_sphere r1 r2 h = Real.sqrt 421 / 2 :=
by
  intros r1 r2 h h1 h2 h3
  simp [radius_of_sphere]
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_truncated_cone_sphere_radius_l1033_103326


namespace NUMINAMATH_GPT_tan_beta_is_neg3_l1033_103383

theorem tan_beta_is_neg3 (α β : ℝ) (h1 : Real.tan α = -2) (h2 : Real.tan (α + β) = 1) : Real.tan β = -3 := 
sorry

end NUMINAMATH_GPT_tan_beta_is_neg3_l1033_103383


namespace NUMINAMATH_GPT_drinking_problem_solution_l1033_103337

def drinking_rate (name : String) (hours : ℕ) (total_liters : ℕ) : ℚ :=
  total_liters / hours

def total_wine_consumed_in_x_hours (x : ℚ) :=
  x * (
  drinking_rate "assistant1" 12 40 +
  drinking_rate "assistant2" 10 40 +
  drinking_rate "assistant3" 8 40
  )

theorem drinking_problem_solution : 
  (∃ x : ℚ, total_wine_consumed_in_x_hours x = 40) →
  ∃ x : ℚ, x = 120 / 37 :=
by 
  sorry

end NUMINAMATH_GPT_drinking_problem_solution_l1033_103337


namespace NUMINAMATH_GPT_reciprocal_roots_l1033_103381

theorem reciprocal_roots (a b : ℝ) (h : a ≠ 0) :
  ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + a = 0) ∧ (a * x2^2 + b * x2 + a = 0) → x1 = 1 / x2 ∧ x2 = 1 / x1 :=
by
  intros x1 x2 hroots
  have hsum : x1 + x2 = -b / a := by sorry
  have hprod : x1 * x2 = 1 := by sorry
  sorry

end NUMINAMATH_GPT_reciprocal_roots_l1033_103381


namespace NUMINAMATH_GPT_china_math_olympiad_34_2023_l1033_103370

-- Defining the problem conditions and verifying the minimum and maximum values of S.
theorem china_math_olympiad_34_2023 {a b c d e : ℝ}
  (h1 : a ≥ -1)
  (h2 : b ≥ -1)
  (h3 : c ≥ -1)
  (h4 : d ≥ -1)
  (h5 : e ≥ -1)
  (h6 : a + b + c + d + e = 5) :
  (-512 ≤ (a + b) * (b + c) * (c + d) * (d + e) * (e + a)) ∧
  ((a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≤ 288) :=
sorry

end NUMINAMATH_GPT_china_math_olympiad_34_2023_l1033_103370


namespace NUMINAMATH_GPT_student_count_estimate_l1033_103344

theorem student_count_estimate 
  (n : Nat) 
  (h1 : 80 ≤ n) 
  (h2 : 100 ≤ n) 
  (h3 : 20 * n = 8000) : 
  n = 400 := 
by 
  sorry

end NUMINAMATH_GPT_student_count_estimate_l1033_103344


namespace NUMINAMATH_GPT_function_symmetric_about_point_l1033_103345

theorem function_symmetric_about_point :
  ∃ x₀ y₀, (x₀, y₀) = (Real.pi / 3, 0) ∧ ∀ x y, y = Real.sin (2 * x + Real.pi / 3) →
    (Real.sin (2 * (2 * x₀ - x) + Real.pi / 3) = y) :=
sorry

end NUMINAMATH_GPT_function_symmetric_about_point_l1033_103345


namespace NUMINAMATH_GPT_compute_trig_expression_l1033_103361

theorem compute_trig_expression : 
  (1 - 1 / (Real.cos (37 * Real.pi / 180))) *
  (1 + 1 / (Real.sin (53 * Real.pi / 180))) *
  (1 - 1 / (Real.sin (37 * Real.pi / 180))) *
  (1 + 1 / (Real.cos (53 * Real.pi / 180))) = 1 :=
sorry

end NUMINAMATH_GPT_compute_trig_expression_l1033_103361


namespace NUMINAMATH_GPT_find_y_l1033_103368

theorem find_y 
  (x y : ℝ) 
  (h1 : (6 : ℝ) = (1/2 : ℝ) * x) 
  (h2 : y = (1/2 : ℝ) * 10) 
  (h3 : x * y = 60) 
: y = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_y_l1033_103368


namespace NUMINAMATH_GPT_election_winner_margin_l1033_103351

theorem election_winner_margin (V : ℝ) 
    (hV: V = 3744 / 0.52) 
    (w_votes: ℝ := 3744) 
    (l_votes: ℝ := 0.48 * V) :
    w_votes - l_votes = 288 := by
  sorry

end NUMINAMATH_GPT_election_winner_margin_l1033_103351


namespace NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l1033_103332

theorem solve_quadratic_1 (x : ℝ) : 2 * x^2 - 7 * x - 1 = 0 ↔ 
  (x = (7 + Real.sqrt 57) / 4 ∨ x = (7 - Real.sqrt 57) / 4) := 
by 
  sorry

theorem solve_quadratic_2 (x : ℝ) : (2 * x - 3)^2 = 10 * x - 15 ↔ 
  (x = 3 / 2 ∨ x = 4) := 
by 
  sorry

end NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l1033_103332


namespace NUMINAMATH_GPT_maximize_S_n_decreasing_arithmetic_sequence_l1033_103349

theorem maximize_S_n_decreasing_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d < 0)
  (h3 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2)
  (h4 : S 5 = S 10) :
  S 7 = S 8 :=
sorry

end NUMINAMATH_GPT_maximize_S_n_decreasing_arithmetic_sequence_l1033_103349


namespace NUMINAMATH_GPT_dodecahedron_equilateral_triangles_l1033_103348

-- Definitions reflecting the conditions
def vertices_of_dodecahedron := 20
def faces_of_dodecahedron := 12
def vertices_per_face := 5
def equilateral_triangles_per_face := 5

theorem dodecahedron_equilateral_triangles :
  (faces_of_dodecahedron * equilateral_triangles_per_face) = 60 := by
  sorry

end NUMINAMATH_GPT_dodecahedron_equilateral_triangles_l1033_103348


namespace NUMINAMATH_GPT_typing_time_together_l1033_103303

def meso_typing_rate : ℕ := 3 -- pages per minute
def tyler_typing_rate : ℕ := 5 -- pages per minute
def pages_to_type : ℕ := 40 -- pages

theorem typing_time_together :
  (meso_typing_rate + tyler_typing_rate) * 5 = pages_to_type :=
by
  sorry

end NUMINAMATH_GPT_typing_time_together_l1033_103303


namespace NUMINAMATH_GPT_rank_from_left_l1033_103389

theorem rank_from_left (total_students rank_from_right rank_from_left : ℕ) 
  (h_total : total_students = 31) (h_right : rank_from_right = 21) : 
  rank_from_left = 11 := by
  sorry

end NUMINAMATH_GPT_rank_from_left_l1033_103389


namespace NUMINAMATH_GPT_floor_sqrt_18_squared_eq_16_l1033_103325

theorem floor_sqrt_18_squared_eq_16 : (Int.floor (Real.sqrt 18)) ^ 2 = 16 := 
by 
  sorry

end NUMINAMATH_GPT_floor_sqrt_18_squared_eq_16_l1033_103325


namespace NUMINAMATH_GPT_area_of_annulus_l1033_103347

variable {b c h : ℝ}
variable (hb : b > c)
variable (h2 : h^2 = b^2 - 2 * c^2)

theorem area_of_annulus (hb : b > c) (h2 : h^2 = b^2 - 2 * c^2) :
    π * (b^2 - c^2) = π * h^2 := by
  sorry

end NUMINAMATH_GPT_area_of_annulus_l1033_103347


namespace NUMINAMATH_GPT_nth_term_150_l1033_103365

-- Conditions
def a : ℕ := 2
def d : ℕ := 5
def arithmetic_sequence (n : ℕ) : ℕ := a + (n - 1) * d

-- Question and corresponding answer proof
theorem nth_term_150 : arithmetic_sequence 150 = 747 := by
  sorry

end NUMINAMATH_GPT_nth_term_150_l1033_103365


namespace NUMINAMATH_GPT_intersection_locus_l1033_103302

theorem intersection_locus
  (a b : ℝ) (a_gt_b : a > b) (b_gt_zero : b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1) :
  ∃ (x y : ℝ), (x^2)/(a^2) - (y^2)/(b^2) = 1 :=
sorry

end NUMINAMATH_GPT_intersection_locus_l1033_103302


namespace NUMINAMATH_GPT_right_triangle_sum_of_squares_l1033_103396

   theorem right_triangle_sum_of_squares {AB AC BC : ℝ} (h_right: AB^2 + AC^2 = BC^2) (h_hypotenuse: BC = 1) :
     AB^2 + AC^2 + BC^2 = 2 :=
   by
     sorry
   
end NUMINAMATH_GPT_right_triangle_sum_of_squares_l1033_103396


namespace NUMINAMATH_GPT_successive_product_l1033_103335

theorem successive_product (n : ℤ) (h : n * (n + 1) = 4160) : n = 64 :=
sorry

end NUMINAMATH_GPT_successive_product_l1033_103335


namespace NUMINAMATH_GPT_algebraic_expression_value_l1033_103353

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 6 - Real.sqrt 2) : 2 * x^2 + 4 * Real.sqrt 2 * x = 8 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1033_103353


namespace NUMINAMATH_GPT_mary_has_10_blue_marbles_l1033_103311

-- Define the number of blue marbles Dan has
def dan_marbles : ℕ := 5

-- Define the factor by which Mary has more blue marbles than Dan
def factor : ℕ := 2

-- Define the number of blue marbles Mary has
def mary_marbles : ℕ := factor * dan_marbles

-- The theorem statement: Mary has 10 blue marbles
theorem mary_has_10_blue_marbles : mary_marbles = 10 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_mary_has_10_blue_marbles_l1033_103311


namespace NUMINAMATH_GPT_evaluate_expression_l1033_103318

theorem evaluate_expression (a b : ℚ) (h1 : a + b = 4) (h2 : a - b = 2) :
  ( (a^2 - 6 * a * b + 9 * b^2) / (a^2 - 2 * a * b) / ((5 * b^2 / (a - 2 * b)) - (a + 2 * b)) - 1 / a ) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1033_103318


namespace NUMINAMATH_GPT_factory_produces_6500_toys_per_week_l1033_103323

theorem factory_produces_6500_toys_per_week
    (days_per_week : ℕ)
    (toys_per_day : ℕ)
    (h1 : days_per_week = 5)
    (h2 : toys_per_day = 1300) :
    days_per_week * toys_per_day = 6500 := 
by 
  sorry

end NUMINAMATH_GPT_factory_produces_6500_toys_per_week_l1033_103323


namespace NUMINAMATH_GPT_candy_distribution_problem_l1033_103331

theorem candy_distribution_problem (n : ℕ) :
  (n - 1) * (n - 2) / 2 - 3 * (n/2 - 1) / 6 = n + 1 → n = 18 :=
sorry

end NUMINAMATH_GPT_candy_distribution_problem_l1033_103331


namespace NUMINAMATH_GPT_subtracting_five_equals_thirtyfive_l1033_103394

variable (x : ℕ)

theorem subtracting_five_equals_thirtyfive (h : x - 5 = 35) : x / 5 = 8 :=
sorry

end NUMINAMATH_GPT_subtracting_five_equals_thirtyfive_l1033_103394


namespace NUMINAMATH_GPT_slope_of_perpendicular_line_l1033_103367

-- Define what it means to be the slope of a line in a certain form
def slope_of_line (a b c : ℝ) (m : ℝ) : Prop :=
  b ≠ 0 ∧ m = -a / b

-- Define what it means for two slopes to be perpendicular
def are_perpendicular_slopes (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Given conditions
def given_line : Prop := slope_of_line 4 5 20 (-4 / 5)

-- The theorem to be proved
theorem slope_of_perpendicular_line : ∃ m : ℝ, given_line ∧ are_perpendicular_slopes (-4 / 5) m ∧ m = 5 / 4 :=
  sorry

end NUMINAMATH_GPT_slope_of_perpendicular_line_l1033_103367


namespace NUMINAMATH_GPT_value_set_for_a_non_empty_proper_subsets_l1033_103393

def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 6 = 0}

theorem value_set_for_a (M : Set ℝ) : 
  (∀ (a : ℝ), B a ⊆ A → a ∈ M) :=
sorry

theorem non_empty_proper_subsets (M : Set ℝ) :
  M = {0, 3, -3} →
  (∃ S : Set (Set ℝ), S = {{0}, {3}, {-3}, {0, 3}, {0, -3}, {3, -3}}) :=
sorry

end NUMINAMATH_GPT_value_set_for_a_non_empty_proper_subsets_l1033_103393


namespace NUMINAMATH_GPT_mans_rate_in_still_water_l1033_103350

-- Definitions from the conditions
def speed_with_stream : ℝ := 10
def speed_against_stream : ℝ := 6

-- The statement to prove the man's rate in still water is as expected.
theorem mans_rate_in_still_water : (speed_with_stream + speed_against_stream) / 2 = 8 := by
  sorry

end NUMINAMATH_GPT_mans_rate_in_still_water_l1033_103350


namespace NUMINAMATH_GPT_fifteen_percent_of_x_is_ninety_l1033_103312

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end NUMINAMATH_GPT_fifteen_percent_of_x_is_ninety_l1033_103312


namespace NUMINAMATH_GPT_q_value_l1033_103384

noncomputable def prove_q (a b m p q : Real) :=
  (a * b = 5) → 
  (b + 1/a) * (a + 1/b) = q →
  q = 36/5

theorem q_value (a b : ℝ) (h_roots : a * b = 5) : (b + 1/a) * (a + 1/b) = 36 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_q_value_l1033_103384


namespace NUMINAMATH_GPT_remainder_3203_4507_9929_mod_75_l1033_103386

theorem remainder_3203_4507_9929_mod_75 :
  (3203 * 4507 * 9929) % 75 = 34 :=
by
  have h1 : 3203 % 75 = 53 := sorry
  have h2 : 4507 % 75 = 32 := sorry
  have h3 : 9929 % 75 = 29 := sorry
  -- complete the proof using modular arithmetic rules.
  sorry

end NUMINAMATH_GPT_remainder_3203_4507_9929_mod_75_l1033_103386


namespace NUMINAMATH_GPT_minimize_expression_l1033_103378

theorem minimize_expression : 
  let a := -1
  let b := -0.5
  (a + b) ≤ (a - b) ∧ (a + b) ≤ (a * b) ∧ (a + b) ≤ (a / b) := by
  let a := -1
  let b := -0.5
  sorry

end NUMINAMATH_GPT_minimize_expression_l1033_103378


namespace NUMINAMATH_GPT_find_a5_l1033_103330

variables {a : ℕ → ℝ}  -- represent the arithmetic sequence

-- Definition of arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
axiom a3_a8_sum : a 3 + a 8 = 22
axiom a6_value : a 6 = 8
axiom arithmetic : is_arithmetic_sequence a

-- Target proof statement
theorem find_a5 (a : ℕ → ℝ) (arithmetic : is_arithmetic_sequence a) (a3_a8_sum : a 3 + a 8 = 22) (a6_value : a 6 = 8) : a 5 = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a5_l1033_103330


namespace NUMINAMATH_GPT_find_number_l1033_103375

theorem find_number : 
  (15^2 * 9^2) / x = 51.193820224719104 → x = 356 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1033_103375


namespace NUMINAMATH_GPT_exists_n_gt_1958_l1033_103391

noncomputable def polyline_path (n : ℕ) : ℝ := sorry
noncomputable def distance_to_origin (n : ℕ) : ℝ := sorry 
noncomputable def sum_lengths (n : ℕ) : ℝ := sorry

theorem exists_n_gt_1958 :
  ∃ (n : ℕ), n > 1958 ∧ (sum_lengths n) / (distance_to_origin n) > 1958 := 
sorry

end NUMINAMATH_GPT_exists_n_gt_1958_l1033_103391


namespace NUMINAMATH_GPT_campaign_funds_total_l1033_103321

variable (X : ℝ)

def campaign_funds (friends family remaining : ℝ) : Prop :=
  friends = 0.40 * X ∧
  family = 0.30 * (X - friends) ∧
  remaining = X - (friends + family) ∧
  remaining = 4200

theorem campaign_funds_total (X_val : ℝ) (friends family remaining : ℝ)
    (h : campaign_funds X friends family remaining) : X = 10000 :=
by
  have h_friends : friends = 0.40 * X := h.1
  have h_family : family = 0.30 * (X - friends) := h.2.1
  have h_remaining : remaining = X - (friends + family) := h.2.2.1
  have h_remaining_amount : remaining = 4200 := h.2.2.2
  sorry

end NUMINAMATH_GPT_campaign_funds_total_l1033_103321


namespace NUMINAMATH_GPT_find_a11_l1033_103343

variable (a : ℕ → ℝ)

axiom geometric_seq (a : ℕ → ℝ) (r : ℝ) : ∀ n, a (n + 1) = a n * r

variable (r : ℝ)
variable (h3 : a 3 = 4)
variable (h7 : a 7 = 12)

theorem find_a11 : a 11 = 36 := by
  sorry

end NUMINAMATH_GPT_find_a11_l1033_103343


namespace NUMINAMATH_GPT_evaluate_expression_l1033_103390

theorem evaluate_expression : (4 * 4 + 4) / (2 * 2 - 2) = 10 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1033_103390


namespace NUMINAMATH_GPT_evaluate_expression_l1033_103371

theorem evaluate_expression :
  3000 * (3000 ^ 1500 + 3000 ^ 1500) = 2 * 3000 ^ 1501 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1033_103371


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_twice_side_area_l1033_103327

noncomputable def triangle_side_length (s : ℝ) :=
  s * s * Real.sqrt 3 / 4 = 2 * s

noncomputable def triangle_perimeter (s : ℝ) := 3 * s

theorem equilateral_triangle_perimeter_twice_side_area (s : ℝ) (h : triangle_side_length s) : 
  triangle_perimeter s = 8 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_perimeter_twice_side_area_l1033_103327


namespace NUMINAMATH_GPT_math_problem_solution_l1033_103355

theorem math_problem_solution (pA : ℚ) (pB : ℚ)
  (hA : pA = 1/2) (hB : pB = 1/3) :
  let pNoSolve := (1 - pA) * (1 - pB)
  let pSolve := 1 - pNoSolve
  pNoSolve = 1/3 ∧ pSolve = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_solution_l1033_103355


namespace NUMINAMATH_GPT_total_volume_of_four_boxes_l1033_103317

theorem total_volume_of_four_boxes :
  (∃ (V : ℕ), (∀ (edge_length : ℕ) (num_boxes : ℕ), edge_length = 6 → num_boxes = 4 → V = (edge_length ^ 3) * num_boxes)) :=
by
  let edge_length := 6
  let num_boxes := 4
  let volume := (edge_length ^ 3) * num_boxes
  use volume
  sorry

end NUMINAMATH_GPT_total_volume_of_four_boxes_l1033_103317


namespace NUMINAMATH_GPT_tomatoes_price_per_pound_l1033_103322

noncomputable def price_per_pound (cost_per_pound : ℝ) (loss_percent : ℝ) (profit_percent : ℝ) : ℝ :=
  let remaining_percent := 1 - loss_percent / 100
  let desired_total := (1 + profit_percent / 100) * cost_per_pound
  desired_total / remaining_percent

theorem tomatoes_price_per_pound :
  price_per_pound 0.80 15 8 = 1.02 :=
by
  sorry

end NUMINAMATH_GPT_tomatoes_price_per_pound_l1033_103322


namespace NUMINAMATH_GPT_width_of_shop_l1033_103300

theorem width_of_shop 
  (monthly_rent : ℕ) 
  (length : ℕ) 
  (annual_rent_per_sqft : ℕ) 
  (h1 : monthly_rent = 3600) 
  (h2 : length = 18) 
  (h3 : annual_rent_per_sqft = 120) :
  ∃ width : ℕ, width = 20 :=
by
  sorry

end NUMINAMATH_GPT_width_of_shop_l1033_103300


namespace NUMINAMATH_GPT_minimize_expression_l1033_103315

theorem minimize_expression : ∃ c : ℝ, (∀ x : ℝ, (1/3 * x^2 + 7*x - 4) ≥ (1/3 * c^2 + 7*c - 4)) ∧ (c = -21/2) :=
sorry

end NUMINAMATH_GPT_minimize_expression_l1033_103315


namespace NUMINAMATH_GPT_residues_exponent_residues_divides_p_minus_one_primitive_roots_phi_l1033_103366

noncomputable def phi (n : ℕ) : ℕ := Nat.totient n

theorem residues_exponent (p : ℕ) (d : ℕ) [hp : Fact (Nat.Prime p)] (hd : d ∣ p - 1) : 
  ∃ (S : Finset ℕ), S.card = phi d ∧ ∀ x ∈ S, x^d % p = 1 :=
by sorry

theorem residues_divides_p_minus_one (p : ℕ) (d : ℕ) [hp : Fact (Nat.Prime p)] (hd : d ∣ p - 1) : 
  ∃ (S : Finset ℕ), S.card = phi d :=
by sorry
  
theorem primitive_roots_phi (p : ℕ) [hp : Fact (Nat.Prime p)] : 
  ∃ (S : Finset ℕ), S.card = phi (p-1) ∧ ∀ g ∈ S, IsPrimitiveRoot g p :=
by sorry

end NUMINAMATH_GPT_residues_exponent_residues_divides_p_minus_one_primitive_roots_phi_l1033_103366


namespace NUMINAMATH_GPT_expand_expression_l1033_103354

theorem expand_expression (y : ℝ) : (7 * y + 12) * 3 * y = 21 * y ^ 2 + 36 * y := by
  sorry

end NUMINAMATH_GPT_expand_expression_l1033_103354


namespace NUMINAMATH_GPT_basketball_surface_area_l1033_103307

theorem basketball_surface_area (C : ℝ) (r : ℝ) (A : ℝ) (π : ℝ) 
  (h1 : C = 30) 
  (h2 : C = 2 * π * r) 
  (h3 : A = 4 * π * r^2) 
  : A = 900 / π := by
  sorry

end NUMINAMATH_GPT_basketball_surface_area_l1033_103307


namespace NUMINAMATH_GPT_area_of_square_ABCD_l1033_103324

theorem area_of_square_ABCD :
  (∃ (x y : ℝ), 2 * x + 2 * y = 40) →
  ∃ (s : ℝ), s = 20 ∧ s * s = 400 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_ABCD_l1033_103324


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1033_103382

-- Define the quadratic function
def f (x t : ℝ) : ℝ := x^2 + t * x - t

-- The proof statement about the condition for roots
theorem sufficient_not_necessary_condition (t : ℝ) :
  (t ≥ 0 → ∃ x : ℝ, f x t = 0) ∧ (∃ x : ℝ, f x t = 0 → t ≥ 0 ∨ t ≤ -4) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1033_103382


namespace NUMINAMATH_GPT_product_of_digits_l1033_103340

theorem product_of_digits (n A B : ℕ) (h1 : n % 6 = 0) (h2 : A + B = 12) (h3 : n = 10 * A + B) : 
  (A * B = 32 ∨ A * B = 36) :=
by 
  sorry

end NUMINAMATH_GPT_product_of_digits_l1033_103340


namespace NUMINAMATH_GPT_Wayne_blocks_count_l1033_103320

-- Statement of the proof problem
theorem Wayne_blocks_count (initial_blocks additional_blocks total_blocks : ℕ) 
  (h1 : initial_blocks = 9) 
  (h2 : additional_blocks = 6) 
  (h3 : total_blocks = initial_blocks + additional_blocks) : 
  total_blocks = 15 := 
by 
  -- proof would go here, but we will use sorry for now
  sorry

end NUMINAMATH_GPT_Wayne_blocks_count_l1033_103320


namespace NUMINAMATH_GPT_overlapping_region_area_l1033_103338

noncomputable def radius : ℝ := 15
noncomputable def central_angle_radians : ℝ := Real.pi / 2
noncomputable def area_of_sector : ℝ := (1 / 4) * Real.pi * (radius^2)
noncomputable def side_length_equilateral_triangle : ℝ := radius
noncomputable def area_of_equilateral_triangle : ℝ := (Real.sqrt 3 / 4) * (side_length_equilateral_triangle^2)
noncomputable def overlapping_area : ℝ := 2 * area_of_sector - area_of_equilateral_triangle

theorem overlapping_region_area :
  overlapping_area = 112.5 * Real.pi - 56.25 * Real.sqrt 3 :=
by
  sorry
 
end NUMINAMATH_GPT_overlapping_region_area_l1033_103338


namespace NUMINAMATH_GPT_good_carrots_l1033_103334

theorem good_carrots (haley_picked : ℕ) (mom_picked : ℕ) (bad_carrots : ℕ) :
  haley_picked = 39 → mom_picked = 38 → bad_carrots = 13 →
  (haley_picked + mom_picked - bad_carrots) = 64 :=
by
  sorry  -- Proof is omitted.

end NUMINAMATH_GPT_good_carrots_l1033_103334


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1033_103310

theorem sufficient_but_not_necessary_condition : ∀ (y : ℝ), (y = 2 → y^2 = 4) ∧ (y^2 = 4 → (y = 2 ∨ y = -2)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1033_103310


namespace NUMINAMATH_GPT_find_k_from_direction_vector_l1033_103395

/-- Given points p1 and p2, the direction vector's k component
    is -3 when the x component is 3. -/
theorem find_k_from_direction_vector
  (p1 : ℤ × ℤ) (p2 : ℤ × ℤ)
  (h1 : p1 = (2, -1))
  (h2 : p2 = (-4, 5))
  (dv_x : ℤ) (dv_k : ℤ)
  (h3 : (dv_x, dv_k) = (3, -3)) :
  True :=
by
  sorry

end NUMINAMATH_GPT_find_k_from_direction_vector_l1033_103395


namespace NUMINAMATH_GPT_blue_face_area_factor_l1033_103314

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end NUMINAMATH_GPT_blue_face_area_factor_l1033_103314


namespace NUMINAMATH_GPT_tan_two_x_is_odd_l1033_103369

noncomputable def tan_two_x (x : ℝ) : ℝ := Real.tan (2 * x)

theorem tan_two_x_is_odd :
  ∀ x : ℝ,
  (∀ k : ℤ, x ≠ (k * Real.pi / 2) + (Real.pi / 4)) →
  tan_two_x (-x) = -tan_two_x x :=
by
  sorry

end NUMINAMATH_GPT_tan_two_x_is_odd_l1033_103369


namespace NUMINAMATH_GPT_expected_coin_worth_is_two_l1033_103305

-- Define the conditions
def p_heads : ℚ := 4 / 5
def p_tails : ℚ := 1 / 5
def gain_heads : ℚ := 5
def loss_tails : ℚ := -10

-- Expected worth calculation
def expected_worth : ℚ := (p_heads * gain_heads) + (p_tails * loss_tails)

-- Lean 4 statement to prove
theorem expected_coin_worth_is_two : expected_worth = 2 := by
  sorry

end NUMINAMATH_GPT_expected_coin_worth_is_two_l1033_103305


namespace NUMINAMATH_GPT_problem_l1033_103373

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 4

theorem problem (a b : ℝ) (h : f a b (-2) = 2) : f a b 2 = -10 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1033_103373


namespace NUMINAMATH_GPT_sum_of_three_integers_l1033_103398

theorem sum_of_three_integers (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) (h7 : a * b * c = 5^3) : a + b + c = 31 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_integers_l1033_103398


namespace NUMINAMATH_GPT_number_of_boys_l1033_103301

variables (total_girls total_teachers total_people : ℕ)
variables (total_girls_eq : total_girls = 315) (total_teachers_eq : total_teachers = 772) (total_people_eq : total_people = 1396)

theorem number_of_boys (total_boys : ℕ) : total_boys = total_people - total_girls - total_teachers :=
by sorry

end NUMINAMATH_GPT_number_of_boys_l1033_103301


namespace NUMINAMATH_GPT_robert_more_photos_than_claire_l1033_103304

theorem robert_more_photos_than_claire
  (claire_photos : ℕ)
  (Lisa_photos : ℕ)
  (Robert_photos : ℕ)
  (Claire_takes_photos : claire_photos = 12)
  (Lisa_takes_photos : Lisa_photos = 3 * claire_photos)
  (Lisa_and_Robert_same_photos : Lisa_photos = Robert_photos) :
  Robert_photos - claire_photos = 24 := by
    sorry

end NUMINAMATH_GPT_robert_more_photos_than_claire_l1033_103304


namespace NUMINAMATH_GPT_amoeba_count_after_week_l1033_103374

-- Definition of the initial conditions
def amoeba_splits_daily (n : ℕ) : ℕ := 2^n

-- Theorem statement translating the problem to Lean
theorem amoeba_count_after_week : amoeba_splits_daily 7 = 128 :=
by
  sorry

end NUMINAMATH_GPT_amoeba_count_after_week_l1033_103374


namespace NUMINAMATH_GPT_tank_a_height_l1033_103309

theorem tank_a_height (h_B : ℝ) (C_A C_B : ℝ) (V_A : ℝ → ℝ) (V_B : ℝ) :
  C_A = 4 ∧ C_B = 10 ∧ h_B = 8 ∧ (∀ h_A : ℝ, V_A h_A = 0.10000000000000002 * V_B) →
  ∃ h_A : ℝ, h_A = 5 :=
by sorry

end NUMINAMATH_GPT_tank_a_height_l1033_103309


namespace NUMINAMATH_GPT_max_angle_B_l1033_103328

-- We define the necessary terms to state our problem
variables {A B C : Real} -- The angles of triangle ABC
variables {cot_A cot_B cot_C : Real} -- The cotangents of angles A, B, and C

-- The main theorem stating that given the conditions the maximum value of angle B is pi/3
theorem max_angle_B (h1 : cot_B = (cot_A + cot_C) / 2) (h2 : A + B + C = Real.pi) :
  B ≤ Real.pi / 3 := by
  sorry

end NUMINAMATH_GPT_max_angle_B_l1033_103328


namespace NUMINAMATH_GPT_find_positive_integer_l1033_103388

theorem find_positive_integer (n : ℕ) (h1 : n % 14 = 0) (h2 : 676 ≤ n ∧ n ≤ 702) : n = 700 :=
sorry

end NUMINAMATH_GPT_find_positive_integer_l1033_103388


namespace NUMINAMATH_GPT_misha_card_numbers_l1033_103397

-- Define the context for digits
def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

-- Define conditions
def proper_fraction (a b : ℕ) : Prop := is_digit a ∧ is_digit b ∧ a < b

-- Original problem statement rewritten for Lean
theorem misha_card_numbers (L O M N S B : ℕ) :
  is_digit L → is_digit O → is_digit M → is_digit N → is_digit S → is_digit B →
  proper_fraction O M → proper_fraction O S →
  L + O / M + O + N + O / S = 10 + B :=
sorry

end NUMINAMATH_GPT_misha_card_numbers_l1033_103397


namespace NUMINAMATH_GPT_correction_amount_l1033_103399

variable (x : ℕ)

def half_dollar := 50
def quarter := 25
def nickel := 5
def dime := 10

theorem correction_amount : 
  ∀ x, (x * (half_dollar - quarter)) - (x * (dime - nickel)) = 20 * x := by
  intros x 
  sorry

end NUMINAMATH_GPT_correction_amount_l1033_103399


namespace NUMINAMATH_GPT_students_not_enrolled_l1033_103364

-- Declare the conditions
def total_students : Nat := 79
def students_french : Nat := 41
def students_german : Nat := 22
def students_both : Nat := 9

-- Define the problem statement
theorem students_not_enrolled : total_students - (students_french + students_german - students_both) = 25 := by
  sorry

end NUMINAMATH_GPT_students_not_enrolled_l1033_103364


namespace NUMINAMATH_GPT_maximum_enclosed_area_l1033_103346

theorem maximum_enclosed_area (P : ℝ) (A : ℝ) : 
  P = 100 → (∃ l w : ℝ, P = 2 * l + 2 * w ∧ A = l * w) → A ≤ 625 :=
by
  sorry

end NUMINAMATH_GPT_maximum_enclosed_area_l1033_103346


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l1033_103360

noncomputable def a : ℝ := 0.99 ^ (1.01 : ℝ)
noncomputable def b : ℝ := 1.01 ^ (0.99 : ℝ)
noncomputable def c : ℝ := Real.log 0.99 / Real.log 1.01

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l1033_103360


namespace NUMINAMATH_GPT_triangle_property_l1033_103362

theorem triangle_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (h_perimeter : a + b + c = 12) (h_inradius : 2 * (a + b + c) = 24) :
    ¬((a^2 + b^2 = c^2) ∨ (a^2 + b^2 > c^2) ∨ (c^2 > a^2 + b^2)) := 
sorry

end NUMINAMATH_GPT_triangle_property_l1033_103362


namespace NUMINAMATH_GPT_second_divisor_l1033_103380

theorem second_divisor (x : ℕ) (k q : ℤ) : 
  (197 % 13 = 2) → 
  (x > 13) → 
  (197 % x = 5) → 
  x = 16 :=
by sorry

end NUMINAMATH_GPT_second_divisor_l1033_103380


namespace NUMINAMATH_GPT_a_plus_b_is_18_over_5_l1033_103377

noncomputable def a_b_sum (a b : ℚ) : Prop :=
  (∃ (x y : ℚ), x = 2 ∧ y = 3 ∧ x = (1 / 3) * y + a ∧ y = (1 / 5) * x + b) → a + b = (18 / 5)

-- No proof provided, just the statement.
theorem a_plus_b_is_18_over_5 (a b : ℚ) : a_b_sum a b :=
sorry

end NUMINAMATH_GPT_a_plus_b_is_18_over_5_l1033_103377


namespace NUMINAMATH_GPT_percentage_second_division_l1033_103357

theorem percentage_second_division (total_students : ℕ) 
                                  (first_division_percentage : ℝ) 
                                  (just_passed : ℕ) 
                                  (all_students_passed : total_students = 300) 
                                  (percentage_first_division : first_division_percentage = 26) 
                                  (students_just_passed : just_passed = 60) : 
  (26 / 100 * 300 + (total_students - (26 / 100 * 300 + 60)) + 60) = 300 → 
  ((total_students - (26 / 100 * 300 + 60)) / total_students * 100) = 54 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_second_division_l1033_103357


namespace NUMINAMATH_GPT_complex_expression_eq_l1033_103359

open Real

theorem complex_expression_eq (p q : ℝ) (hpq : p ≠ q) :
  (sqrt ((p^4 + q^4)/(p^4 - p^2 * q^2) + (2 * q^2)/(p^2 - q^2)) * (p^3 - p * q^2) - 2 * q * sqrt p) /
  (sqrt (p / (p - q) - q / (p + q) - 2 * p * q / (p^2 - q^2)) * (p - q)) = 
  sqrt (p^2 - q^2) / sqrt p := 
sorry

end NUMINAMATH_GPT_complex_expression_eq_l1033_103359


namespace NUMINAMATH_GPT_prime_solution_exists_l1033_103336

theorem prime_solution_exists :
  ∃ (p q r : ℕ), p.Prime ∧ q.Prime ∧ r.Prime ∧ (p + q^2 = r^4) ∧ (p = 7) ∧ (q = 3) ∧ (r = 2) := 
by
  sorry

end NUMINAMATH_GPT_prime_solution_exists_l1033_103336


namespace NUMINAMATH_GPT_mark_owes_linda_l1033_103385

-- Define the payment per room and the number of rooms painted
def payment_per_room := (13 : ℚ) / 3
def rooms_painted := (8 : ℚ) / 5

-- State the theorem and the proof
theorem mark_owes_linda : (payment_per_room * rooms_painted) = (104 : ℚ) / 15 := by
  sorry

end NUMINAMATH_GPT_mark_owes_linda_l1033_103385


namespace NUMINAMATH_GPT_area_of_shaded_region_l1033_103333

theorem area_of_shaded_region :
  let v1 := (0, 0)
  let v2 := (15, 0)
  let v3 := (45, 30)
  let v4 := (45, 45)
  let v5 := (30, 45)
  let v6 := (0, 15)
  let area_large_rectangle := 45 * 45
  let area_triangle1 := 1 / 2 * 15 * 15
  let area_triangle2 := 1 / 2 * 15 * 15
  let shaded_area := area_large_rectangle - (area_triangle1 + area_triangle2)
  shaded_area = 1800 :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l1033_103333


namespace NUMINAMATH_GPT_value_of_a_l1033_103339

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem value_of_a (a : ℝ) :
  (1 / log_base 2 a) + (1 / log_base 3 a) + (1 / log_base 4 a) + (1 / log_base 5 a) = 7 / 4 ↔
  a = 120 ^ (4 / 7) :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1033_103339


namespace NUMINAMATH_GPT_factorization_correct_l1033_103356

theorem factorization_correct : 
  ¬(∃ x : ℝ, -x^2 + 4 * x = -x * (x + 4)) ∧
  ¬(∃ x y: ℝ, x^2 + x * y + x = x * (x + y)) ∧
  (∀ x y: ℝ, x * (x - y) + y * (y - x) = (x - y)^2) ∧
  ¬(∃ x : ℝ, x^2 - 4 * x + 4 = (x + 2) * (x - 2)) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1033_103356


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1033_103313

variable (f : ℝ → ℝ)
variable (h_inc : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)

theorem solution_set_of_inequality :
  {x | 0 < x ∧ f x > f (2 * x - 4)} = {x | 2 < x ∧ x < 4} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1033_103313


namespace NUMINAMATH_GPT_inequality_amgm_l1033_103372

theorem inequality_amgm (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : a^3 + b^3 + a + b ≥ 4 * a * b :=
sorry

end NUMINAMATH_GPT_inequality_amgm_l1033_103372


namespace NUMINAMATH_GPT_digits_base_d_l1033_103319

theorem digits_base_d (d A B : ℕ) (h₀ : d > 7) (h₁ : A < d) (h₂ : B < d) 
  (h₃ : A * d + B + B * d + A = 2 * d^2 + 2) : A - B = 2 :=
by
  sorry

end NUMINAMATH_GPT_digits_base_d_l1033_103319


namespace NUMINAMATH_GPT_loss_equals_cost_price_of_some_balls_l1033_103306

-- Conditions
def cost_price_per_ball := 60
def selling_price_for_17_balls := 720
def number_of_balls := 17

-- Calculations
def total_cost_price := number_of_balls * cost_price_per_ball
def loss := total_cost_price - selling_price_for_17_balls

-- Proof statement
theorem loss_equals_cost_price_of_some_balls : (loss / cost_price_per_ball) = 5 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_loss_equals_cost_price_of_some_balls_l1033_103306
