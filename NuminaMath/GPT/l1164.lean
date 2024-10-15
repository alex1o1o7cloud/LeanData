import Mathlib

namespace NUMINAMATH_GPT_find_value_of_f_at_1_l1164_116419

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem find_value_of_f_at_1 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, 2 * f x - f (- x) = 3 * x + 1) : f 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_f_at_1_l1164_116419


namespace NUMINAMATH_GPT_D_score_l1164_116486

noncomputable def score_A : ℕ := 94

variables (A B C D E : ℕ)

-- Conditions
def A_scored : A = score_A := sorry
def B_highest : B > A := sorry
def C_average_AD : (C * 2) = A + D := sorry
def D_average_five : (D * 5) = A + B + C + D + E := sorry
def E_score_C2 : E = C + 2 := sorry

-- Question
theorem D_score : D = 96 :=
by {
  sorry
}

end NUMINAMATH_GPT_D_score_l1164_116486


namespace NUMINAMATH_GPT_min_omega_value_l1164_116412

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) := 2 * Real.sin (ω * x + φ)

theorem min_omega_value
  (ω : ℝ) (φ : ℝ)
  (hω : ω > 0)
  (h1 : f (π / 3) ω φ = 0)
  (h2 : f (π / 2) ω φ = 2) :
  ω = 3 :=
sorry

end NUMINAMATH_GPT_min_omega_value_l1164_116412


namespace NUMINAMATH_GPT_total_area_painted_is_correct_l1164_116417

noncomputable def barn_area_painted (width length height : ℝ) : ℝ :=
  let walls_area := 2 * (width * height + length * height) * 2
  let ceiling_and_roof_area := 2 * (width * length)
  walls_area + ceiling_and_roof_area

theorem total_area_painted_is_correct 
  (width length height : ℝ) 
  (h_w : width = 12) 
  (h_l : length = 15) 
  (h_h : height = 6) 
  : barn_area_painted width length height = 1008 :=
  by
  rw [h_w, h_l, h_h]
  -- Simplify steps omitted
  sorry

end NUMINAMATH_GPT_total_area_painted_is_correct_l1164_116417


namespace NUMINAMATH_GPT_problem_statement_l1164_116411

open Real

noncomputable def f (x : ℝ) : ℝ := 10^x

theorem problem_statement : f (log 2) * f (log 5) = 10 :=
by {
  -- Note: Proof is omitted as indicated in the procedure.
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1164_116411


namespace NUMINAMATH_GPT_calculate_adults_in_play_l1164_116404

theorem calculate_adults_in_play :
  ∃ A : ℕ, (11 * A = 49 + 50) := sorry

end NUMINAMATH_GPT_calculate_adults_in_play_l1164_116404


namespace NUMINAMATH_GPT_min_cubes_required_l1164_116485

def volume_of_box (L W H : ℕ) : ℕ := L * W * H
def volume_of_cube (v_cube : ℕ) : ℕ := v_cube
def minimum_number_of_cubes (V_box V_cube : ℕ) : ℕ := V_box / V_cube

theorem min_cubes_required :
  minimum_number_of_cubes (volume_of_box 12 16 6) (volume_of_cube 3) = 384 :=
by sorry

end NUMINAMATH_GPT_min_cubes_required_l1164_116485


namespace NUMINAMATH_GPT_psychiatrist_problem_l1164_116413

theorem psychiatrist_problem 
  (x : ℕ)
  (h_total : 4 * 8 + x + (x + 5) = 25)
  : x = 2 := by
  sorry

end NUMINAMATH_GPT_psychiatrist_problem_l1164_116413


namespace NUMINAMATH_GPT_sequence_formula_l1164_116491

theorem sequence_formula (a : ℕ → ℚ) 
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) = a n / (2 * a n + 1)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l1164_116491


namespace NUMINAMATH_GPT_not_all_perfect_squares_l1164_116424

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem not_all_perfect_squares (x : ℕ) (hx : x > 0) :
  ¬ (is_perfect_square (2 * x - 1) ∧ is_perfect_square (5 * x - 1) ∧ is_perfect_square (13 * x - 1)) :=
by
  sorry

end NUMINAMATH_GPT_not_all_perfect_squares_l1164_116424


namespace NUMINAMATH_GPT_solve_system_of_equations_l1164_116428

theorem solve_system_of_equations (x y z t : ℤ) :
  (3 * x - 2 * y + 4 * z + 2 * t = 19) ∧ (5 * x + 6 * y - 2 * z + 3 * t = 23) →
  (x = 16 * z - 18 * y - 11) ∧ (t = 28 * y - 26 * z + 26) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_system_of_equations_l1164_116428


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1164_116444

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 := by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1164_116444


namespace NUMINAMATH_GPT_coefficient_of_x7_in_expansion_eq_15_l1164_116479

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := n.choose k

theorem coefficient_of_x7_in_expansion_eq_15 (a : ℝ) (hbinom : binomial 10 3 * (-a) ^ 3 = 15) : a = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_coefficient_of_x7_in_expansion_eq_15_l1164_116479


namespace NUMINAMATH_GPT_remainder_of_3_pow_20_mod_7_l1164_116469

theorem remainder_of_3_pow_20_mod_7 : (3^20) % 7 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_20_mod_7_l1164_116469


namespace NUMINAMATH_GPT_determine_x_l1164_116421

variable (n p : ℝ)

-- Definitions based on conditions
def x (n : ℝ) : ℝ := 4 * n
def percentage_condition (n p : ℝ) : Prop := 2 * n + 3 = (p / 100) * 25

-- Statement to be proven
theorem determine_x (h : percentage_condition n p) : x n = 4 * n := by
  sorry

end NUMINAMATH_GPT_determine_x_l1164_116421


namespace NUMINAMATH_GPT_solve_for_A_l1164_116496

theorem solve_for_A (A : ℚ) : 80 - (5 - (6 + A * (7 - 8 - 5))) = 89 → A = -4/3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_A_l1164_116496


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1164_116407

theorem geometric_sequence_problem
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h5 : a 5 * a 6 = 3)
  (h9 : a 9 * a 10 = 9) :
  a 7 * a 8 = 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1164_116407


namespace NUMINAMATH_GPT_total_books_l1164_116464

def books_per_shelf_mystery : ℕ := 7
def books_per_shelf_picture : ℕ := 5
def books_per_shelf_sci_fi : ℕ := 8
def books_per_shelf_biography : ℕ := 6

def shelves_mystery : ℕ := 8
def shelves_picture : ℕ := 2
def shelves_sci_fi : ℕ := 3
def shelves_biography : ℕ := 4

theorem total_books :
  (books_per_shelf_mystery * shelves_mystery) + 
  (books_per_shelf_picture * shelves_picture) + 
  (books_per_shelf_sci_fi * shelves_sci_fi) + 
  (books_per_shelf_biography * shelves_biography) = 114 :=
by
  sorry

end NUMINAMATH_GPT_total_books_l1164_116464


namespace NUMINAMATH_GPT_fraction_value_l1164_116440

theorem fraction_value :
  (0.02 ^ 2 + 0.52 ^ 2 + 0.035 ^ 2) / (0.002 ^ 2 + 0.052 ^ 2 + 0.0035 ^ 2) = 100 := by
    sorry

end NUMINAMATH_GPT_fraction_value_l1164_116440


namespace NUMINAMATH_GPT_matthew_total_time_on_failure_day_l1164_116450

-- Define the conditions as variables
def assembly_time : ℝ := 1 -- hours
def usual_baking_time : ℝ := 1.5 -- hours
def decoration_time : ℝ := 1 -- hours
def baking_factor : ℝ := 2 -- Factor by which baking time increased on that day

-- Prove that the total time taken is 5 hours
theorem matthew_total_time_on_failure_day : 
  (assembly_time + (usual_baking_time * baking_factor) + decoration_time) = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_matthew_total_time_on_failure_day_l1164_116450


namespace NUMINAMATH_GPT_train_crosses_pole_in_9_seconds_l1164_116484

theorem train_crosses_pole_in_9_seconds
  (speed_kmh : ℝ) (train_length_m : ℝ) (time_s : ℝ) 
  (h1 : speed_kmh = 58) 
  (h2 : train_length_m = 145) 
  (h3 : time_s = train_length_m / (speed_kmh * 1000 / 3600)) :
  time_s = 9 :=
by
  sorry

end NUMINAMATH_GPT_train_crosses_pole_in_9_seconds_l1164_116484


namespace NUMINAMATH_GPT_number_of_planting_methods_l1164_116442

noncomputable def num_planting_methods : ℕ :=
  -- Six different types of crops
  let crops := ['A', 'B', 'C', 'D', 'E', 'F']
  -- Six trial fields arranged in a row, numbered 1 through 6
  -- Condition: Crop A cannot be planted in the first two fields
  -- Condition: Crop B must not be adjacent to crop A
  -- Answer: 240 different planting methods
  240

theorem number_of_planting_methods :
  num_planting_methods = 240 :=
  by
    -- Proof omitted
    sorry

end NUMINAMATH_GPT_number_of_planting_methods_l1164_116442


namespace NUMINAMATH_GPT_area_parallelogram_proof_l1164_116434

/-- We are given a rectangle with a length of 10 cm and a width of 8 cm.
    We transform it into a parallelogram with a height of 9 cm.
    We need to prove that the area of the parallelogram is 72 square centimeters. -/
def area_of_parallelogram_from_rectangle (length width height : ℝ) : ℝ :=
  width * height

theorem area_parallelogram_proof
  (length width height : ℝ)
  (h_length : length = 10)
  (h_width : width = 8)
  (h_height : height = 9) :
  area_of_parallelogram_from_rectangle length width height = 72 :=
by
  sorry

end NUMINAMATH_GPT_area_parallelogram_proof_l1164_116434


namespace NUMINAMATH_GPT_altitude_angle_bisector_inequality_l1164_116445

theorem altitude_angle_bisector_inequality
  (h l R r : ℝ) 
  (triangle_condition : ∀ (h l : ℝ) (R r : ℝ), (h > 0 ∧ l > 0 ∧ R > 0 ∧ r > 0)) :
  h / l ≥ Real.sqrt (2 * r / R) :=
by
  sorry

end NUMINAMATH_GPT_altitude_angle_bisector_inequality_l1164_116445


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1164_116457

-- Let's define the variables and conditions first
variables (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
variable (h_asymptote : b = a)

-- We need to prove the eccentricity
theorem hyperbola_eccentricity : eccentricity = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1164_116457


namespace NUMINAMATH_GPT_determine_A_value_l1164_116422

noncomputable def solve_for_A (A B C : ℝ) : Prop :=
  (A = 1/16) ↔ 
  (∀ x : ℝ, (1 / ((x + 5) * (x - 3) * (x + 3))) = (A / (x + 5)) + (B / (x - 3)) + (C / (x + 3)))

theorem determine_A_value :
  solve_for_A (1/16) B C :=
by
  sorry

end NUMINAMATH_GPT_determine_A_value_l1164_116422


namespace NUMINAMATH_GPT_int_modulo_l1164_116416

theorem int_modulo (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 17) (h3 : 38574 ≡ n [ZMOD 17]) : n = 1 :=
by
  sorry

end NUMINAMATH_GPT_int_modulo_l1164_116416


namespace NUMINAMATH_GPT_graph_of_equation_is_line_and_hyperbola_l1164_116470

theorem graph_of_equation_is_line_and_hyperbola :
  ∀ (x y : ℝ), ((x^2 - 1) * (x + y) = y^2 * (x + y)) ↔ (y = -x) ∨ ((x + y) * (x - y) = 1) := by
  intro x y
  sorry

end NUMINAMATH_GPT_graph_of_equation_is_line_and_hyperbola_l1164_116470


namespace NUMINAMATH_GPT_coin_probability_l1164_116463

theorem coin_probability (a r : ℝ) (h : r < a / 2) :
  let favorable_cells := 3
  let larger_cell_area := 9 * a^2
  let favorable_area_per_cell := (a - 2 * r)^2
  let favorable_area := favorable_cells * favorable_area_per_cell
  let probability := favorable_area / larger_cell_area
  probability = (a - 2 * r)^2 / (3 * a^2) :=
by
  sorry

end NUMINAMATH_GPT_coin_probability_l1164_116463


namespace NUMINAMATH_GPT_relationship_among_f_values_l1164_116414

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0)

theorem relationship_among_f_values (h₀ : 0 < 2) (h₁ : 2 < 3) :
  f 0 > f (-2) ∧ f (-2) > f 3 :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_f_values_l1164_116414


namespace NUMINAMATH_GPT_appropriate_sampling_methods_l1164_116420
-- Import the entire Mathlib library for broader functionality

-- Define the conditions
def community_high_income_families : ℕ := 125
def community_middle_income_families : ℕ := 280
def community_low_income_families : ℕ := 95
def community_total_households : ℕ := community_high_income_families + community_middle_income_families + community_low_income_families

def student_count : ℕ := 12

-- Define the theorem to be proven
theorem appropriate_sampling_methods :
  (community_total_households = 500 → stratified_sampling) ∧
  (student_count = 12 → random_sampling) :=
by sorry

end NUMINAMATH_GPT_appropriate_sampling_methods_l1164_116420


namespace NUMINAMATH_GPT_unique_12_tuple_l1164_116429

theorem unique_12_tuple : 
  ∃! (x : Fin 12 → ℝ), 
    ((1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + 
    (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 +
    (x 7 - x 8)^2 + (x 8 - x 9)^2 + (x 9 - x 10)^2 + (x 10 - x 11)^2 + 
    (x 11)^2 = 1 / 13) ∧ (x 0 + x 11 = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_12_tuple_l1164_116429


namespace NUMINAMATH_GPT_sum_of_floors_of_square_roots_l1164_116471

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end NUMINAMATH_GPT_sum_of_floors_of_square_roots_l1164_116471


namespace NUMINAMATH_GPT_no_snow_five_days_l1164_116435

noncomputable def prob_snow_each_day : ℚ := 2 / 3

noncomputable def prob_no_snow_one_day : ℚ := 1 - prob_snow_each_day

noncomputable def prob_no_snow_five_days : ℚ := prob_no_snow_one_day ^ 5

theorem no_snow_five_days:
  prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end NUMINAMATH_GPT_no_snow_five_days_l1164_116435


namespace NUMINAMATH_GPT_problem_rewrite_expression_l1164_116433

theorem problem_rewrite_expression (j : ℝ) : 
  ∃ (c p q : ℝ), (8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ (q / p = -77) :=
sorry

end NUMINAMATH_GPT_problem_rewrite_expression_l1164_116433


namespace NUMINAMATH_GPT_value_of_f_5_l1164_116443

variable (f : ℕ → ℕ) (x y : ℕ)

theorem value_of_f_5 (h1 : f 2 = 50) (h2 : ∀ x, f x = 2 * x ^ 2 + y) : f 5 = 92 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_5_l1164_116443


namespace NUMINAMATH_GPT_expression_equals_answer_l1164_116430

noncomputable def evaluate_expression : ℚ :=
  (2011^2 * 2012 - 2013) / Nat.factorial 2012 +
  (2013^2 * 2014 - 2015) / Nat.factorial 2014

theorem expression_equals_answer :
  evaluate_expression = 
  1 / Nat.factorial 2009 + 
  1 / Nat.factorial 2010 - 
  1 / Nat.factorial 2013 - 
  1 / Nat.factorial 2014 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_answer_l1164_116430


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1164_116406

theorem quadratic_inequality_solution_set (a b c : ℝ) (Δ : ℝ) (hΔ : Δ = b^2 - 4*a*c) :
  (∀ x : ℝ, a*x^2 + b*x + c > 0) ↔ (a > 0 ∧ Δ < 0) := by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1164_116406


namespace NUMINAMATH_GPT_new_average_height_is_184_l1164_116480

-- Define the initial conditions
def original_num_students : ℕ := 35
def original_avg_height : ℕ := 180
def left_num_students : ℕ := 7
def left_avg_height : ℕ := 120
def joined_num_students : ℕ := 7
def joined_avg_height : ℕ := 140

-- Calculate the initial total height
def original_total_height := original_avg_height * original_num_students

-- Calculate the total height of the students who left
def left_total_height := left_avg_height * left_num_students

-- Calculate the new total height after the students left
def new_total_height1 := original_total_height - left_total_height

-- Calculate the total height of the new students who joined
def joined_total_height := joined_avg_height * joined_num_students

-- Calculate the new total height after the new students joined
def new_total_height2 := new_total_height1 + joined_total_height

-- Calculate the new average height
def new_avg_height := new_total_height2 / original_num_students

-- The theorem stating the result
theorem new_average_height_is_184 : new_avg_height = 184 := by
  sorry

end NUMINAMATH_GPT_new_average_height_is_184_l1164_116480


namespace NUMINAMATH_GPT_journey_time_difference_l1164_116474

theorem journey_time_difference :
  let speed := 40  -- mph
  let distance1 := 360  -- miles
  let distance2 := 320  -- miles
  (distance1 / speed - distance2 / speed) * 60 = 60 := 
by
  sorry

end NUMINAMATH_GPT_journey_time_difference_l1164_116474


namespace NUMINAMATH_GPT_matrix_det_evaluation_l1164_116461

noncomputable def matrix_det (x y z : ℝ) : ℝ :=
  Matrix.det ![
    ![1,   x,     y,     z],
    ![1, x + y,   y,     z],
    ![1,   x, x + y,     z],
    ![1,   x,     y, x + y + z]
  ]

theorem matrix_det_evaluation (x y z : ℝ) :
  matrix_det x y z = y * x * x + y * y * x :=
by sorry

end NUMINAMATH_GPT_matrix_det_evaluation_l1164_116461


namespace NUMINAMATH_GPT_harmonic_power_identity_l1164_116408

open Real

theorem harmonic_power_identity (a b c : ℝ) (n : ℕ) (hn : n % 2 = 1) 
(h : (1 / a + 1 / b + 1 / c) = 1 / (a + b + c)) :
  (1 / (a ^ n) + 1 / (b ^ n) + 1 / (c ^ n) = 1 / (a ^ n + b ^ n + c ^ n)) :=
sorry

end NUMINAMATH_GPT_harmonic_power_identity_l1164_116408


namespace NUMINAMATH_GPT_minimum_questions_to_identify_white_ball_l1164_116497

theorem minimum_questions_to_identify_white_ball (n : ℕ) (even_white : ℕ) 
  (h₁ : n = 2004) 
  (h₂ : even_white % 2 = 0) 
  (h₃ : 1 ≤ even_white ∧ even_white ≤ n) :
  ∃ m : ℕ, m = 2003 := 
sorry

end NUMINAMATH_GPT_minimum_questions_to_identify_white_ball_l1164_116497


namespace NUMINAMATH_GPT_prove_ln10_order_l1164_116438

def ln10_order_proof : Prop :=
  let a := Real.log 10
  let b := Real.log 100
  let c := (Real.log 10) ^ 2
  c > b ∧ b > a

theorem prove_ln10_order : ln10_order_proof := 
sorry

end NUMINAMATH_GPT_prove_ln10_order_l1164_116438


namespace NUMINAMATH_GPT_total_cost_verification_l1164_116498

def sandwich_cost : ℝ := 2.45
def soda_cost : ℝ := 0.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4
def total_cost : ℝ := 8.38

theorem total_cost_verification 
  (sc : sandwich_cost = 2.45)
  (sd : soda_cost = 0.87)
  (ns : num_sandwiches = 2)
  (nd : num_sodas = 4) :
  num_sandwiches * sandwich_cost + num_sodas * soda_cost = total_cost := 
sorry

end NUMINAMATH_GPT_total_cost_verification_l1164_116498


namespace NUMINAMATH_GPT_cost_per_person_trip_trips_rental_cost_l1164_116467

-- Define the initial conditions
def ticket_price_per_person := 60
def total_employees := 70
def small_car_seats := 4
def large_car_seats := 11
def extra_cost_small_car_per_person := 5
def extra_revenue_large_car := 50
def max_total_cost := 5000

-- Define the costs per person per trip for small and large cars
def large_car_cost_per_person := 10
def small_car_cost_per_person := large_car_cost_per_person + extra_cost_small_car_per_person

-- Define the number of trips for four-seater and eleven-seater cars
def four_seater_trips := 1
def eleven_seater_trips := 6

-- Prove the lean statements
theorem cost_per_person_trip : 
  (11 * large_car_cost_per_person) - (small_car_seats * small_car_cost_per_person) = extra_revenue_large_car := 
sorry

theorem trips_rental_cost (x y : ℕ) : 
  (small_car_seats * x + large_car_seats * y = total_employees) ∧
  ((total_employees * ticket_price_per_person) + (small_car_cost_per_person * small_car_seats * x) + (large_car_cost_per_person * large_car_seats * y) ≤ max_total_cost) :=
sorry

end NUMINAMATH_GPT_cost_per_person_trip_trips_rental_cost_l1164_116467


namespace NUMINAMATH_GPT_find_fraction_l1164_116478

noncomputable def some_fraction_of_number_is (N f : ℝ) : Prop :=
  1 + f * N = 0.75 * N

theorem find_fraction (N : ℝ) (hN : N = 12.0) :
  ∃ f : ℝ, some_fraction_of_number_is N f ∧ f = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l1164_116478


namespace NUMINAMATH_GPT_triangle_properties_l1164_116492

theorem triangle_properties (A B C a b c : ℝ) (h1 : a * Real.tan C = 2 * c * Real.sin A)
  (h2 : C > 0 ∧ C < Real.pi)
  (h3 : a / Real.sin A = c / Real.sin C) :
  C = Real.pi / 3 ∧ (1 / 2 < Real.sin (A + Real.pi / 6) ∧ Real.sin (A + Real.pi / 6) ≤ 1) →
  (Real.sqrt 3 / 2 < Real.sin A + Real.sin B ∧ Real.sin A + Real.sin B ≤ Real.sqrt 3) :=
by
  intro h4
  sorry

end NUMINAMATH_GPT_triangle_properties_l1164_116492


namespace NUMINAMATH_GPT_ratio_of_side_lengths_l1164_116487

theorem ratio_of_side_lengths (a b c : ℕ) (h : a * a * b * b = 18 * c * c * 50 * c * c) :
  (12 = 1800000) ->  (15 = 1500) -> (10 > 0):=
by
  sorry

end NUMINAMATH_GPT_ratio_of_side_lengths_l1164_116487


namespace NUMINAMATH_GPT_max_difference_is_correct_l1164_116446

noncomputable def max_y_difference : ℝ := 
  let x1 := Real.sqrt (2 / 3)
  let y1 := 2 + (x1 ^ 2) + (x1 ^ 3)
  let x2 := -x1
  let y2 := 2 + (x2 ^ 2) + (x2 ^ 3)
  abs (y1 - y2)

theorem max_difference_is_correct : max_y_difference = 4 * Real.sqrt 2 / 9 := 
  sorry -- Proof is omitted

end NUMINAMATH_GPT_max_difference_is_correct_l1164_116446


namespace NUMINAMATH_GPT_square_area_inscribed_in_parabola_l1164_116415

-- Declare the parabola equation
def parabola (x : ℝ) : ℝ := x^2 - 10 * x + 20

-- Declare the condition that we have a square inscribed to this parabola.
def is_inscribed_square (side_length : ℝ) : Prop :=
∀ (x : ℝ), (x = 5 - side_length/2 ∨ x = 5 + side_length/2) → (parabola x = 0)

-- Proof goal
theorem square_area_inscribed_in_parabola : ∃ (side_length : ℝ), is_inscribed_square side_length ∧ side_length^2 = 400 :=
by
  sorry

end NUMINAMATH_GPT_square_area_inscribed_in_parabola_l1164_116415


namespace NUMINAMATH_GPT_cut_half_meter_from_two_thirds_l1164_116432

theorem cut_half_meter_from_two_thirds (L : ℝ) (hL : L = 2 / 3) : L - 1 / 6 = 1 / 2 :=
by
  rw [hL]
  norm_num

end NUMINAMATH_GPT_cut_half_meter_from_two_thirds_l1164_116432


namespace NUMINAMATH_GPT_find_f_neg5_l1164_116472

theorem find_f_neg5 (a b : ℝ) (Sin : ℝ → ℝ) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x + b * (Sin x) ^ 3 + 1)
  (h_f5 : f 5 = 7) :
  f (-5) = -5 := 
by
  sorry

end NUMINAMATH_GPT_find_f_neg5_l1164_116472


namespace NUMINAMATH_GPT_circle_relationship_l1164_116465

noncomputable def f : ℝ × ℝ → ℝ := sorry

variables {x y x₁ y₁ x₂ y₂ : ℝ}
variables (h₁ : f (x₁, y₁) = 0) (h₂ : f (x₂, y₂) ≠ 0)

theorem circle_relationship :
  f (x, y) - f (x₁, y₁) - f (x₂, y₂) = 0 ↔ f (x, y) = f (x₂, y₂) :=
sorry

end NUMINAMATH_GPT_circle_relationship_l1164_116465


namespace NUMINAMATH_GPT_total_amount_is_20_yuan_60_cents_l1164_116409

-- Conditions
def ten_yuan_note : ℕ := 10
def five_yuan_notes : ℕ := 2 * 5
def twenty_cent_coins : ℕ := 3 * 20

-- Total amount calculation
def total_yuan : ℕ := ten_yuan_note + five_yuan_notes
def total_cents : ℕ := twenty_cent_coins

-- Conversion rates
def yuan_per_cent : ℕ := 100
def total_cents_in_yuan : ℕ := total_cents / yuan_per_cent
def remaining_cents : ℕ := total_cents % yuan_per_cent

-- Proof statement
theorem total_amount_is_20_yuan_60_cents : total_yuan = 20 ∧ total_cents_in_yuan = 0 ∧ remaining_cents = 60 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_is_20_yuan_60_cents_l1164_116409


namespace NUMINAMATH_GPT_minimal_withdrawals_proof_l1164_116449

-- Defining the conditions
def red_marbles : ℕ := 200
def blue_marbles : ℕ := 300
def green_marbles : ℕ := 400

def max_red_withdrawal_per_time : ℕ := 1
def max_blue_withdrawal_per_time : ℕ := 2
def max_total_withdrawal_per_time : ℕ := 5

-- The target minimal number of withdrawals
def minimal_withdrawals : ℕ := 200

-- Lean statement of the proof problem
theorem minimal_withdrawals_proof :
  ∃ (w : ℕ), w = minimal_withdrawals ∧ 
    (∀ n, n ≤ w →
      (n = 200 ∧ 
       (∀ r b g, r ≤ max_red_withdrawal_per_time ∧ b ≤ max_blue_withdrawal_per_time ∧ (r + b + g) ≤ max_total_withdrawal_per_time))) :=
sorry

end NUMINAMATH_GPT_minimal_withdrawals_proof_l1164_116449


namespace NUMINAMATH_GPT_result_of_subtraction_l1164_116455

theorem result_of_subtraction (N : ℝ) (h1 : N = 100) : 0.80 * N - 20 = 60 :=
by
  sorry

end NUMINAMATH_GPT_result_of_subtraction_l1164_116455


namespace NUMINAMATH_GPT_max_value_expr_l1164_116400

open Real

theorem max_value_expr {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 9) : 
  (x / y + y / z + z / x) * (y / x + z / y + x / z) = 81 / 4 :=
sorry

end NUMINAMATH_GPT_max_value_expr_l1164_116400


namespace NUMINAMATH_GPT_fixed_point_of_line_range_of_a_to_avoid_second_quadrant_l1164_116456

theorem fixed_point_of_line (a : ℝ) (A : ℝ × ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - a - 5 = 0 -> A = (1, 6)) :=
sorry

theorem range_of_a_to_avoid_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - a - 5 = 0 -> x * y < 0 -> a ≤ -5) :=
sorry

end NUMINAMATH_GPT_fixed_point_of_line_range_of_a_to_avoid_second_quadrant_l1164_116456


namespace NUMINAMATH_GPT_problem_statement_l1164_116475

theorem problem_statement (x y : ℝ) (h₁ : |x| = 3) (h₂ : |y| = 4) (h₃ : x > y) : 2 * x - y = 10 := 
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1164_116475


namespace NUMINAMATH_GPT_rachel_fathers_age_when_rachel_is_25_l1164_116410

theorem rachel_fathers_age_when_rachel_is_25 (R G M F Y : ℕ) 
  (h1 : R = 12)
  (h2 : G = 7 * R)
  (h3 : M = G / 2)
  (h4 : F = M + 5)
  (h5 : Y = 25 - R) : 
  F + Y = 60 :=
by sorry

end NUMINAMATH_GPT_rachel_fathers_age_when_rachel_is_25_l1164_116410


namespace NUMINAMATH_GPT_problem_statement_l1164_116458

variables (p1 p2 p3 p4 : Prop)

theorem problem_statement (h_p1 : p1 = True)
                         (h_p2 : p2 = False)
                         (h_p3 : p3 = False)
                         (h_p4 : p4 = True) :
  (p1 ∧ p4) = True ∧
  (p1 ∧ p2) = False ∧
  (¬p2 ∨ p3) = True ∧
  (¬p3 ∨ ¬p4) = True :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1164_116458


namespace NUMINAMATH_GPT_time_for_A_to_complete_work_l1164_116495

theorem time_for_A_to_complete_work (W : ℝ) (A B C : ℝ) (W_pos : 0 < W) (B_work : B = W / 40) (C_work : C = W / 20) : 
  (10 * (W / A) + 10 * (W / B) + 10 * (W / C) = W) → A = W / 40 :=
by 
  sorry

end NUMINAMATH_GPT_time_for_A_to_complete_work_l1164_116495


namespace NUMINAMATH_GPT_tom_overall_profit_l1164_116405

def initial_purchase_cost : ℝ := 20 * 3 + 30 * 5 + 15 * 10
def purchase_commission : ℝ := 0.02 * initial_purchase_cost
def total_initial_cost : ℝ := initial_purchase_cost + purchase_commission

def sale_revenue_before_commission : ℝ := 10 * 4 + 20 * 7 + 5 * 12
def sales_commission : ℝ := 0.02 * sale_revenue_before_commission
def total_sales_revenue : ℝ := sale_revenue_before_commission - sales_commission

def remaining_stock_a_value : ℝ := 10 * (3 * 2)
def remaining_stock_b_value : ℝ := 10 * (5 * 1.20)
def remaining_stock_c_value : ℝ := 10 * (10 * 0.90)
def total_remaining_value : ℝ := remaining_stock_a_value + remaining_stock_b_value + remaining_stock_c_value

def overall_profit_or_loss : ℝ := total_sales_revenue + total_remaining_value - total_initial_cost

theorem tom_overall_profit : overall_profit_or_loss = 78 := by
  sorry

end NUMINAMATH_GPT_tom_overall_profit_l1164_116405


namespace NUMINAMATH_GPT_area_of_transformed_region_l1164_116452

theorem area_of_transformed_region : 
  let T : ℝ := 15
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![6, -2]]
  (abs (Matrix.det A) * T = 450) := 
  sorry

end NUMINAMATH_GPT_area_of_transformed_region_l1164_116452


namespace NUMINAMATH_GPT_distance_between_foci_l1164_116436

theorem distance_between_foci :
  let x := ℝ
  let y := ℝ
  ∀ (x y : ℝ), 9*x^2 + 36*x + 4*y^2 - 8*y + 1 = 0 →
  ∃ (d : ℝ), d = (Real.sqrt 351) / 3 :=
sorry

end NUMINAMATH_GPT_distance_between_foci_l1164_116436


namespace NUMINAMATH_GPT_sum_of_values_satisfying_eq_l1164_116490

theorem sum_of_values_satisfying_eq (x : ℝ) :
  (x^2 - 5 * x + 5 = 16) → ∀ r s : ℝ, (r + s = 5) :=
by
  sorry  -- Proof is omitted, looking to verify the structure only.

end NUMINAMATH_GPT_sum_of_values_satisfying_eq_l1164_116490


namespace NUMINAMATH_GPT_clock_rings_in_a_day_l1164_116448

-- Define the conditions
def rings_every_3_hours : ℕ := 3
def first_ring : ℕ := 1 -- This is 1 A.M. in our problem
def total_hours_in_day : ℕ := 24

-- Define the theorem
theorem clock_rings_in_a_day (n_rings : ℕ) : 
  (∀ n : ℕ, n_rings = total_hours_in_day / rings_every_3_hours + 1) :=
by
  -- use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_clock_rings_in_a_day_l1164_116448


namespace NUMINAMATH_GPT_larger_number_is_391_l1164_116403

theorem larger_number_is_391 (A B : ℕ) 
  (hcf : ∀ n : ℕ, n ∣ A ∧ n ∣ B ↔ n = 23)
  (lcm_factors : ∃ C D : ℕ, lcm A B = 23 * 13 * 17 ∧ C = 13 ∧ D = 17) :
  max A B = 391 :=
sorry

end NUMINAMATH_GPT_larger_number_is_391_l1164_116403


namespace NUMINAMATH_GPT_Aunt_Zhang_expenditure_is_negative_l1164_116468

-- Define variables for the problem
def income_yuan : ℤ := 5
def expenditure_yuan : ℤ := 3

-- The theorem stating Aunt Zhang's expenditure in financial terms
theorem Aunt_Zhang_expenditure_is_negative :
  (- expenditure_yuan) = -3 :=
by
  sorry

end NUMINAMATH_GPT_Aunt_Zhang_expenditure_is_negative_l1164_116468


namespace NUMINAMATH_GPT_alice_additional_cookies_proof_l1164_116493

variable (alice_initial_cookies : ℕ)
variable (bob_initial_cookies : ℕ)
variable (cookies_thrown_away : ℕ)
variable (bob_additional_cookies : ℕ)
variable (total_edible_cookies : ℕ)

theorem alice_additional_cookies_proof 
    (h1 : alice_initial_cookies = 74)
    (h2 : bob_initial_cookies = 7)
    (h3 : cookies_thrown_away = 29)
    (h4 : bob_additional_cookies = 36)
    (h5 : total_edible_cookies = 93) :
  alice_initial_cookies + bob_initial_cookies - cookies_thrown_away + bob_additional_cookies + (93 - (74 + 7 - 29 + 36)) = total_edible_cookies :=
by
  sorry

end NUMINAMATH_GPT_alice_additional_cookies_proof_l1164_116493


namespace NUMINAMATH_GPT_largest_divisor_of_n_l1164_116473

-- Definitions and conditions from the problem
def is_positive_integer (n : ℕ) := n > 0
def is_divisible_by (a b : ℕ) := ∃ k : ℕ, a = k * b

-- Lean 4 statement encapsulating the problem
theorem largest_divisor_of_n (n : ℕ) (h1 : is_positive_integer n) (h2 : is_divisible_by (n * n) 72) : 
  ∃ v : ℕ, v = 12 ∧ is_divisible_by n v := 
sorry

end NUMINAMATH_GPT_largest_divisor_of_n_l1164_116473


namespace NUMINAMATH_GPT_value_of_expression_l1164_116462

theorem value_of_expression (x : ℝ) (hx : 23 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1164_116462


namespace NUMINAMATH_GPT_find_S9_l1164_116447

-- Setting up basic definitions for arithmetic sequence and the sum of its terms
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d
def sum_arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := n * (a + arithmetic_seq a d n) / 2

-- Given conditions
variables (a d : ℤ)
axiom h : 2 * arithmetic_seq a d 3 = 3 + a

-- Theorem to prove
theorem find_S9 : sum_arithmetic_seq a d 9 = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_S9_l1164_116447


namespace NUMINAMATH_GPT_greatest_integer_l1164_116459

theorem greatest_integer (m : ℕ) (h1 : 0 < m) (h2 : m < 150)
  (h3 : ∃ a : ℤ, m = 9 * a - 2) (h4 : ∃ b : ℤ, m = 5 * b + 4) :
  m = 124 := 
sorry

end NUMINAMATH_GPT_greatest_integer_l1164_116459


namespace NUMINAMATH_GPT_length_of_first_train_l1164_116489

theorem length_of_first_train
  (speed1_kmph : ℝ) (speed2_kmph : ℝ)
  (time_s : ℝ) (length2_m : ℝ)
  (relative_speed_mps : ℝ := (speed1_kmph + speed2_kmph) * 1000 / 3600)
  (total_distance_m : ℝ := relative_speed_mps * time_s)
  (length1_m : ℝ := total_distance_m - length2_m) :
  speed1_kmph = 80 →
  speed2_kmph = 65 →
  time_s = 7.199424046076314 →
  length2_m = 180 →
  length1_m = 110 :=
by
  sorry

end NUMINAMATH_GPT_length_of_first_train_l1164_116489


namespace NUMINAMATH_GPT_q_poly_correct_l1164_116477

open Polynomial

noncomputable def q : Polynomial ℚ := 
  -(C 1) * X^6 + C 4 * X^4 + C 21 * X^3 + C 15 * X^2 + C 14 * X + C 3

theorem q_poly_correct : 
  ∀ x : Polynomial ℚ,
  q + (X^6 + 4 * X^4 + 5 * X^3 + 12 * X) = 
  (8 * X^4 + 26 * X^3 + 15 * X^2 + 26 * X + C 3) := by sorry

end NUMINAMATH_GPT_q_poly_correct_l1164_116477


namespace NUMINAMATH_GPT_bacon_suggestions_count_l1164_116427

def mashed_potatoes_suggestions : ℕ := 324
def tomatoes_suggestions : ℕ := 128
def total_suggestions : ℕ := 826

theorem bacon_suggestions_count :
  total_suggestions - (mashed_potatoes_suggestions + tomatoes_suggestions) = 374 :=
by
  sorry

end NUMINAMATH_GPT_bacon_suggestions_count_l1164_116427


namespace NUMINAMATH_GPT_circle_sine_intersection_l1164_116437

theorem circle_sine_intersection (h k r : ℝ) (hr : r > 0) :
  ∃ (n : ℕ), n > 16 ∧
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, (x - h)^2 + (2 * Real.sin x - k)^2 = r^2) ∧ xs.card = n :=
by
  sorry

end NUMINAMATH_GPT_circle_sine_intersection_l1164_116437


namespace NUMINAMATH_GPT_inequality_addition_l1164_116439

theorem inequality_addition (a b : ℝ) (h : a > b) : a + 3 > b + 3 := by
  sorry

end NUMINAMATH_GPT_inequality_addition_l1164_116439


namespace NUMINAMATH_GPT_balcony_more_than_orchestra_l1164_116441

theorem balcony_more_than_orchestra (x y : ℕ) 
  (h1 : x + y = 340) 
  (h2 : 12 * x + 8 * y = 3320) : 
  y - x = 40 := 
sorry

end NUMINAMATH_GPT_balcony_more_than_orchestra_l1164_116441


namespace NUMINAMATH_GPT_y_in_terms_of_x_l1164_116431

theorem y_in_terms_of_x (x y : ℝ) (h : 2 * x + y = 5) : y = -2 * x + 5 :=
sorry

end NUMINAMATH_GPT_y_in_terms_of_x_l1164_116431


namespace NUMINAMATH_GPT_find_number_l1164_116425

theorem find_number (x : ℝ) :
  (7 * (x + 10) / 5) - 5 = 44 → x = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1164_116425


namespace NUMINAMATH_GPT_combinatorics_sum_l1164_116481

theorem combinatorics_sum :
  (Nat.choose 20 6 + Nat.choose 20 5 = 62016) :=
by
  sorry

end NUMINAMATH_GPT_combinatorics_sum_l1164_116481


namespace NUMINAMATH_GPT_complex_magnitude_l1164_116466

open Complex

theorem complex_magnitude {x y : ℝ} (h : (1 + Complex.I) * x = 1 + y * Complex.I) : abs (x + y * Complex.I) = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_complex_magnitude_l1164_116466


namespace NUMINAMATH_GPT_larger_number_l1164_116402

theorem larger_number (x y : ℕ) (h1 : x + y = 28) (h2 : x - y = 4) : max x y = 16 := by
  sorry

end NUMINAMATH_GPT_larger_number_l1164_116402


namespace NUMINAMATH_GPT_choir_members_max_l1164_116451

theorem choir_members_max (s x : ℕ) (h1 : s * x < 147) (h2 : s * x + 3 = (s - 3) * (x + 2)) : s * x = 84 :=
sorry

end NUMINAMATH_GPT_choir_members_max_l1164_116451


namespace NUMINAMATH_GPT_probability_failed_both_tests_eq_l1164_116460

variable (total_students pass_test1 pass_test2 pass_both : ℕ)

def students_failed_both_tests (total pass1 pass2 both : ℕ) : ℕ :=
  total - (pass1 + pass2 - both)

theorem probability_failed_both_tests_eq 
  (h_total : total_students = 100)
  (h_pass1 : pass_test1 = 60)
  (h_pass2 : pass_test2 = 40)
  (h_pass_both : pass_both = 20) :
  students_failed_both_tests total_students pass_test1 pass_test2 pass_both / (total_students : ℚ) = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_probability_failed_both_tests_eq_l1164_116460


namespace NUMINAMATH_GPT_jump_difference_l1164_116494

-- Definitions based on conditions
def grasshopper_jump : ℕ := 13
def frog_jump : ℕ := 11

-- Proof statement
theorem jump_difference : grasshopper_jump - frog_jump = 2 := by
  sorry

end NUMINAMATH_GPT_jump_difference_l1164_116494


namespace NUMINAMATH_GPT_lowest_possible_price_l1164_116454

def typeADiscountedPrice (msrp : ℕ) : ℕ :=
  let regularDiscount := msrp * 15 / 100
  let discountedPrice := msrp - regularDiscount
  let additionalDiscount := discountedPrice * 20 / 100
  discountedPrice - additionalDiscount

def typeBDiscountedPrice (msrp : ℕ) : ℕ :=
  let regularDiscount := msrp * 25 / 100
  let discountedPrice := msrp - regularDiscount
  let additionalDiscount := discountedPrice * 15 / 100
  discountedPrice - additionalDiscount

def typeCDiscountedPrice (msrp : ℕ) : ℕ :=
  let regularDiscount := msrp * 30 / 100
  let discountedPrice := msrp - regularDiscount
  let additionalDiscount := discountedPrice * 10 / 100
  discountedPrice - additionalDiscount

def finalPrice (discountedPrice : ℕ) : ℕ :=
  let tax := discountedPrice * 7 / 100
  discountedPrice + tax

theorem lowest_possible_price : 
  min (finalPrice (typeADiscountedPrice 4500)) 
      (min (finalPrice (typeBDiscountedPrice 5500)) 
           (finalPrice (typeCDiscountedPrice 5000))) = 3274 :=
by {
  sorry
}

end NUMINAMATH_GPT_lowest_possible_price_l1164_116454


namespace NUMINAMATH_GPT_find_x_value_l1164_116418

theorem find_x_value (A B C x : ℝ) (hA : A = 40) (hB : B = 3 * x) (hC : C = 2 * x) (hSum : A + B + C = 180) : x = 28 :=
by
  sorry

end NUMINAMATH_GPT_find_x_value_l1164_116418


namespace NUMINAMATH_GPT_mod_equiv_n_l1164_116483

theorem mod_equiv_n (n : ℤ) : 0 ≤ n ∧ n < 9 ∧ -1234 % 9 = n := 
by
  sorry

end NUMINAMATH_GPT_mod_equiv_n_l1164_116483


namespace NUMINAMATH_GPT_sum_of_80th_equation_l1164_116488

theorem sum_of_80th_equation : (2 * 80 + 1) + (5 * 80 - 1) = 560 := by
  sorry

end NUMINAMATH_GPT_sum_of_80th_equation_l1164_116488


namespace NUMINAMATH_GPT_percentage_y_less_than_x_l1164_116401

variable (x y : ℝ)
variable (h : x = 12 * y)

theorem percentage_y_less_than_x :
  (11 / 12) * 100 = 91.67 := by
  sorry

end NUMINAMATH_GPT_percentage_y_less_than_x_l1164_116401


namespace NUMINAMATH_GPT_sum_of_distances_l1164_116426

theorem sum_of_distances (P : ℤ × ℤ) (hP : P = (-1, -2)) :
  abs P.1 + abs P.2 = 3 :=
sorry

end NUMINAMATH_GPT_sum_of_distances_l1164_116426


namespace NUMINAMATH_GPT_find_k_l1164_116482

theorem find_k (x y k : ℝ) 
  (h1 : 4 * x + 2 * y = 5 * k - 4) 
  (h2 : 2 * x + 4 * y = -1) 
  (h3 : x - y = 1) : 
  k = 1 := 
by sorry

end NUMINAMATH_GPT_find_k_l1164_116482


namespace NUMINAMATH_GPT_problem_statement_l1164_116499

theorem problem_statement (x : ℝ) (h : 8 * x - 6 = 10) : 200 * (1 / x) = 100 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1164_116499


namespace NUMINAMATH_GPT_stamens_in_bouquet_l1164_116453

-- Define the number of pistils, leaves, stamens for black roses and crimson flowers
def pistils_black_rose : ℕ := 4
def stamens_black_rose : ℕ := 4
def leaves_black_rose : ℕ := 2

def pistils_crimson_flower : ℕ := 8
def stamens_crimson_flower : ℕ := 10
def leaves_crimson_flower : ℕ := 3

-- Define the number of black roses and crimson flowers (as variables x and y)
variables (x y : ℕ)

-- Define the total number of pistils and leaves in the bouquet
def total_pistils : ℕ := pistils_black_rose * x + pistils_crimson_flower * y
def total_leaves : ℕ := leaves_black_rose * x + leaves_crimson_flower * y

-- Condition: There are 108 fewer leaves than pistils
axiom leaves_pistils_relation : total_leaves = total_pistils - 108

-- Calculate the total number of stamens in the bouquet
def total_stamens : ℕ := stamens_black_rose * x + stamens_crimson_flower * y

-- The theorem to be proved
theorem stamens_in_bouquet : total_stamens = 216 :=
by
  sorry

end NUMINAMATH_GPT_stamens_in_bouquet_l1164_116453


namespace NUMINAMATH_GPT_small_circle_area_l1164_116423

theorem small_circle_area (r R : ℝ) (n : ℕ)
  (h_n : n = 6)
  (h_area_large : π * R^2 = 120)
  (h_relation : r = R / 2) :
  π * r^2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_small_circle_area_l1164_116423


namespace NUMINAMATH_GPT_hats_cost_l1164_116476

variables {week_days : ℕ} {weeks : ℕ} {cost_per_hat : ℕ}

-- Conditions
def num_hats (week_days : ℕ) (weeks : ℕ) : ℕ := week_days * weeks
def total_cost (num_hats : ℕ) (cost_per_hat : ℕ) : ℕ := num_hats * cost_per_hat

-- Proof problem
theorem hats_cost (h1 : week_days = 7) (h2 : weeks = 2) (h3 : cost_per_hat = 50) : 
  total_cost (num_hats week_days weeks) cost_per_hat = 700 :=
by 
  sorry

end NUMINAMATH_GPT_hats_cost_l1164_116476
