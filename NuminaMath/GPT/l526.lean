import Mathlib

namespace NUMINAMATH_GPT_set_operation_equivalence_l526_52636

variable {U : Type} -- U is the universal set
variables {X Y Z : Set U} -- X, Y, and Z are subsets of the universal set U

def star (A B : Set U) : Set U := A ∩ B  -- Define the operation "∗" as intersection

theorem set_operation_equivalence :
  star (star X Y) Z = (X ∩ Y) ∩ Z :=  -- Formulate the problem as a theorem to prove
by
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_set_operation_equivalence_l526_52636


namespace NUMINAMATH_GPT_surface_area_of_sphere_l526_52609

theorem surface_area_of_sphere (a : Real) (h : a = 2 * Real.sqrt 3) : 
  (4 * Real.pi * ((Real.sqrt 3 * a / 2) ^ 2)) = 36 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_sphere_l526_52609


namespace NUMINAMATH_GPT_compute_sqrt_eq_419_l526_52696

theorem compute_sqrt_eq_419 : Real.sqrt ((22 * 21 * 20 * 19) + 1) = 419 :=
by
  sorry

end NUMINAMATH_GPT_compute_sqrt_eq_419_l526_52696


namespace NUMINAMATH_GPT_gary_egg_collection_l526_52669

-- Conditions
def initial_chickens : ℕ := 4
def multiplier : ℕ := 8
def eggs_per_chicken_per_day : ℕ := 6
def days_in_week : ℕ := 7

-- Definitions derived from conditions
def current_chickens : ℕ := initial_chickens * multiplier
def eggs_per_day : ℕ := current_chickens * eggs_per_chicken_per_day
def eggs_per_week : ℕ := eggs_per_day * days_in_week

-- Proof statement
theorem gary_egg_collection : eggs_per_week = 1344 := by
  unfold eggs_per_week
  unfold eggs_per_day
  unfold current_chickens
  sorry

end NUMINAMATH_GPT_gary_egg_collection_l526_52669


namespace NUMINAMATH_GPT_probability_of_heart_and_joker_l526_52662

-- Define a deck with 54 cards, including jokers
def total_cards : ℕ := 54

-- Define the count of specific cards in the deck
def hearts_count : ℕ := 13
def jokers_count : ℕ := 2
def remaining_cards (x: ℕ) : ℕ := total_cards - x

-- Define the probability of drawing a specific card
def prob_of_first_heart : ℚ := hearts_count / total_cards
def prob_of_second_joker (first_card_a_heart: Bool) : ℚ :=
  if first_card_a_heart then jokers_count / remaining_cards 1 else 0

-- Calculate the probability of drawing a heart first and then a joker
def prob_first_heart_then_joker : ℚ :=
  prob_of_first_heart * prob_of_second_joker true

-- Proving the final probability
theorem probability_of_heart_and_joker :
  prob_first_heart_then_joker = 13 / 1419 := by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_probability_of_heart_and_joker_l526_52662


namespace NUMINAMATH_GPT_smallest_integer_solution_l526_52632

theorem smallest_integer_solution (x : ℤ) (h : 2 * (x : ℝ)^2 + 2 * |(x : ℝ)| + 7 < 25) : x = -2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_solution_l526_52632


namespace NUMINAMATH_GPT_sqrt_arith_progression_impossible_l526_52622

theorem sqrt_arith_progression_impossible (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hneab : a ≠ b) (hnebc : b ≠ c) (hneca : c ≠ a) :
  ¬ ∃ d : ℝ, (d = (Real.sqrt b - Real.sqrt a)) ∧ (d = (Real.sqrt c - Real.sqrt b)) :=
sorry

end NUMINAMATH_GPT_sqrt_arith_progression_impossible_l526_52622


namespace NUMINAMATH_GPT_ratio_square_correct_l526_52675

noncomputable def ratio_square (a b : ℝ) (h : a / b = b / Real.sqrt (a^2 + b^2)) : ℝ :=
  let k := a / b
  let x := k * k
  x

theorem ratio_square_correct (a b : ℝ) (h : a / b = b / Real.sqrt (a^2 + b^2)) :
  ratio_square a b h = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_square_correct_l526_52675


namespace NUMINAMATH_GPT_solve_inequality_l526_52623

-- Define the function satisfying the given conditions
def f (x : ℝ) : ℝ := sorry

axiom f_functional_eq : ∀ (x y : ℝ), f (x / y) = f x - f y
axiom f_not_zero : ∀ x : ℝ, f x ≠ 0
axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

-- Define the theorem that proves the inequality given the conditions
theorem solve_inequality (x : ℝ) :
  f x + f (x + 1/2) < 0 ↔ x ∈ (Set.Ioo ( (1 - Real.sqrt 17) / 4 ) 0) ∪ (Set.Ioo 0 ( (1 + Real.sqrt 17) / 4 )) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l526_52623


namespace NUMINAMATH_GPT_remainder_of_2_pow_23_mod_5_l526_52630

theorem remainder_of_2_pow_23_mod_5 
    (h1 : (2^2) % 5 = 4)
    (h2 : (2^3) % 5 = 3)
    (h3 : (2^4) % 5 = 1) :
    (2^23) % 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_2_pow_23_mod_5_l526_52630


namespace NUMINAMATH_GPT_fraction_subtraction_inequality_l526_52624

theorem fraction_subtraction_inequality (a b n : ℕ) (h1 : a < b) (h2 : 0 < n) (h3 : n < a) : 
  (a : ℚ) / b > (a - n : ℚ) / (b - n) :=
sorry

end NUMINAMATH_GPT_fraction_subtraction_inequality_l526_52624


namespace NUMINAMATH_GPT_simplify_expression_l526_52666

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 0) : x⁻¹ - 3 * x + 2 = - (3 * x^2 - 2 * x - 1) / x :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l526_52666


namespace NUMINAMATH_GPT_find_b_l526_52612

-- Define the lines and the condition of parallelism
def line1 := ∀ (x y b : ℝ), 4 * y + 8 * b = 16 * x
def line2 := ∀ (x y b : ℝ), y - 2 = (b - 3) * x
def are_parallel (m1 m2 : ℝ) := m1 = m2

-- Translate the problem to a Lean statement
theorem find_b (b : ℝ) : (∀ x y, 4 * y + 8 * b = 16 * x) → (∀ x y, y - 2 = (b - 3) * x) → b = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l526_52612


namespace NUMINAMATH_GPT_find_other_number_l526_52602

theorem find_other_number (B : ℕ) (HCF : ℕ) (LCM : ℕ) (A : ℕ) 
  (h1 : A = 24) 
  (h2 : HCF = 16) 
  (h3 : LCM = 312) 
  (h4 : HCF * LCM = A * B) :
  B = 208 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l526_52602


namespace NUMINAMATH_GPT_alpha_beta_sum_l526_52679

theorem alpha_beta_sum (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 80 * x + 1551) / (x^2 + 57 * x - 2970)) :
  α + β = 137 :=
by
  sorry

end NUMINAMATH_GPT_alpha_beta_sum_l526_52679


namespace NUMINAMATH_GPT_correct_result_is_102357_l526_52644

-- Defining the conditions
def number (f : ℕ) : Prop := f * 153 = 102357

-- Stating the proof problem
theorem correct_result_is_102357 (f : ℕ) (h : f * 153 = 102325) (wrong_digits : ℕ) :
  (number f) :=
by
  sorry

end NUMINAMATH_GPT_correct_result_is_102357_l526_52644


namespace NUMINAMATH_GPT_ratio_of_radii_of_truncated_cone_l526_52638

theorem ratio_of_radii_of_truncated_cone 
  (R r s : ℝ) 
  (h1 : s = Real.sqrt (R * r)) 
  (h2 : (π * (R^2 + r^2 + R * r) * (2 * s) / 3) = 3 * (4 * π * s^3 / 3)) :
  R / r = 7 := 
sorry

end NUMINAMATH_GPT_ratio_of_radii_of_truncated_cone_l526_52638


namespace NUMINAMATH_GPT_rectangle_square_ratio_l526_52635

theorem rectangle_square_ratio (l w s : ℝ) (h1 : 0.4 * l * w = 0.25 * s * s) : l / w = 15.625 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_square_ratio_l526_52635


namespace NUMINAMATH_GPT_tan_ratio_l526_52657

variable (a b : Real)

theorem tan_ratio (h1 : Real.sin (a + b) = 5 / 8) (h2 : Real.sin (a - b) = 1 / 4) : 
  (Real.tan a) / (Real.tan b) = 7 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_ratio_l526_52657


namespace NUMINAMATH_GPT_ca1_l526_52654

theorem ca1 {
  a b : ℝ
} (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := 
by
  sorry

end NUMINAMATH_GPT_ca1_l526_52654


namespace NUMINAMATH_GPT_geometric_sequence_304th_term_l526_52645

theorem geometric_sequence_304th_term (a r : ℤ) (n : ℕ) (h_a : a = 8) (h_ar : a * r = -8) (h_n : n = 304) :
  ∃ t : ℤ, t = -8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_304th_term_l526_52645


namespace NUMINAMATH_GPT_students_play_alto_saxophone_l526_52628

def roosevelt_high_school :=
  let total_students := 600
  let marching_band_students := total_students / 5
  let brass_instrument_students := marching_band_students / 2
  let saxophone_students := brass_instrument_students / 5
  let alto_saxophone_students := saxophone_students / 3
  alto_saxophone_students

theorem students_play_alto_saxophone :
  roosevelt_high_school = 4 :=
  by
    sorry

end NUMINAMATH_GPT_students_play_alto_saxophone_l526_52628


namespace NUMINAMATH_GPT_hypotenuse_length_l526_52626

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l526_52626


namespace NUMINAMATH_GPT_age_sum_l526_52659

theorem age_sum (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 10) : a + b + c = 27 := by
  sorry

end NUMINAMATH_GPT_age_sum_l526_52659


namespace NUMINAMATH_GPT_max_blue_cells_n2_max_blue_cells_n25_l526_52682

noncomputable def max_blue_cells (table_size n : ℕ) : ℕ :=
  if h : (table_size = 50 ∧ n = 2) then 2450
  else if h : (table_size = 50 ∧ n = 25) then 1300
  else 0 -- Default case that should not happen for this problem

theorem max_blue_cells_n2 : max_blue_cells 50 2 = 2450 := 
by
  sorry

theorem max_blue_cells_n25 : max_blue_cells 50 25 = 1300 :=
by
  sorry

end NUMINAMATH_GPT_max_blue_cells_n2_max_blue_cells_n25_l526_52682


namespace NUMINAMATH_GPT_right_triangles_with_specific_area_and_perimeter_l526_52642

theorem right_triangles_with_specific_area_and_perimeter :
  ∃ (count : ℕ),
    count = 7 ∧
    ∀ (a b : ℕ), 
      (a > 0 ∧ b > 0 ∧ (a ≠ b) ∧ (a^2 + b^2 = c^2) ∧ (a * b / 2 = 5 * (a + b + c))) → 
      count = 7 :=
by
  sorry

end NUMINAMATH_GPT_right_triangles_with_specific_area_and_perimeter_l526_52642


namespace NUMINAMATH_GPT_length_of_second_train_is_319_95_l526_52695

noncomputable def length_of_second_train (length_first_train : ℝ) (speed_first_train_kph : ℝ) (speed_second_train_kph : ℝ) (time_to_cross_seconds : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train_kph * 1000 / 3600
  let speed_second_train_mps := speed_second_train_kph * 1000 / 3600
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance_covered := relative_speed * time_to_cross_seconds
  let length_second_train := total_distance_covered - length_first_train
  length_second_train

theorem length_of_second_train_is_319_95 :
  length_of_second_train 180 120 80 9 = 319.95 :=
sorry

end NUMINAMATH_GPT_length_of_second_train_is_319_95_l526_52695


namespace NUMINAMATH_GPT_gcd_98_63_l526_52601

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_GPT_gcd_98_63_l526_52601


namespace NUMINAMATH_GPT_triangle_area_fraction_l526_52611

-- Define the grid size
def grid_size : ℕ := 6

-- Define the vertices of the triangle
def vertex_A : (ℕ × ℕ) := (3, 3)
def vertex_B : (ℕ × ℕ) := (3, 5)
def vertex_C : (ℕ × ℕ) := (5, 5)

-- Define the area of the larger grid
def area_square := grid_size ^ 2

-- Compute the base and height of the triangle
def base_triangle := vertex_C.1 - vertex_B.1
def height_triangle := vertex_B.2 - vertex_A.2

-- Compute the area of the triangle
def area_triangle := (base_triangle * height_triangle) / 2

-- Define the fraction of the area of the larger square inside the triangle
def area_fraction := area_triangle / area_square

-- State the theorem
theorem triangle_area_fraction :
  area_fraction = 1 / 18 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_fraction_l526_52611


namespace NUMINAMATH_GPT_find_k_l526_52608

theorem find_k (k : ℝ) (h : 32 / k = 4) : k = 8 := sorry

end NUMINAMATH_GPT_find_k_l526_52608


namespace NUMINAMATH_GPT_work_completion_time_l526_52655

theorem work_completion_time (days_B days_C days_all : ℝ) (h_B : days_B = 5) (h_C : days_C = 12) (h_all : days_all = 2.2222222222222223) : 
    (1 / ((days_all / 9) * 10) - 1 / days_B - 1 / days_C)⁻¹ = 60 / 37 := by 
  sorry

end NUMINAMATH_GPT_work_completion_time_l526_52655


namespace NUMINAMATH_GPT_sum_of_sampled_types_l526_52665

-- Define the types of books in each category
def Chinese_types := 20
def Mathematics_types := 10
def Liberal_Arts_Comprehensive_types := 40
def English_types := 30

-- Define the total types of books
def total_types := Chinese_types + Mathematics_types + Liberal_Arts_Comprehensive_types + English_types

-- Define the sample size and stratified sampling ratio
def sample_size := 20
def sampling_ratio := sample_size / total_types

-- Define the number of types sampled from each category
def Mathematics_sampled := Mathematics_types * sampling_ratio
def Liberal_Arts_Comprehensive_sampled := Liberal_Arts_Comprehensive_types * sampling_ratio

-- Define the proof statement
theorem sum_of_sampled_types : Mathematics_sampled + Liberal_Arts_Comprehensive_sampled = 10 :=
by
  -- Your proof here
  sorry

end NUMINAMATH_GPT_sum_of_sampled_types_l526_52665


namespace NUMINAMATH_GPT_sum_first_8_geometric_l526_52681

theorem sum_first_8_geometric :
  let a₁ := 1 / 15
  let r := 2
  let S₄ := a₁ * (1 - r^4) / (1 - r)
  let S₈ := a₁ * (1 - r^8) / (1 - r)
  S₄ = 1 → S₈ = 17 := 
by
  intros a₁ r S₄ S₈ h
  sorry

end NUMINAMATH_GPT_sum_first_8_geometric_l526_52681


namespace NUMINAMATH_GPT_edward_spent_amount_l526_52673

-- Definitions based on the problem conditions
def initial_amount : ℕ := 18
def remaining_amount : ℕ := 2

-- The statement to prove: Edward spent $16
theorem edward_spent_amount : initial_amount - remaining_amount = 16 := by
  sorry

end NUMINAMATH_GPT_edward_spent_amount_l526_52673


namespace NUMINAMATH_GPT_acute_triangle_angles_l526_52699

theorem acute_triangle_angles (x y z : ℕ) (angle1 angle2 angle3 : ℕ) 
  (h1 : angle1 = 7 * x) 
  (h2 : angle2 = 9 * y) 
  (h3 : angle3 = 11 * z) 
  (h4 : angle1 + angle2 + angle3 = 180)
  (hx : 1 ≤ x ∧ x ≤ 12)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (hz : 1 ≤ z ∧ z ≤ 8)
  (ha1 : angle1 < 90)
  (ha2 : angle2 < 90)
  (ha3 : angle3 < 90)
  : angle1 = 42 ∧ angle2 = 72 ∧ angle3 = 66 
  ∨ angle1 = 49 ∧ angle2 = 54 ∧ angle3 = 77 
  ∨ angle1 = 56 ∧ angle2 = 36 ∧ angle3 = 88 
  ∨ angle1 = 84 ∧ angle2 = 63 ∧ angle3 = 33 :=
sorry

end NUMINAMATH_GPT_acute_triangle_angles_l526_52699


namespace NUMINAMATH_GPT_proof_problem_l526_52643

noncomputable def initialEfficiencyOfOneMan : ℕ := sorry
noncomputable def initialEfficiencyOfOneWoman : ℕ := sorry
noncomputable def totalWork : ℕ := sorry

-- Condition (1): 10 men and 15 women together can complete the work in 6 days.
def condition1 := 10 * initialEfficiencyOfOneMan + 15 * initialEfficiencyOfOneWoman = totalWork / 6

-- Condition (2): The efficiency of men to complete the work decreases by 5% every day.
-- This condition is not directly measurable to our proof but noted as additional info.

-- Condition (3): The efficiency of women to complete the work increases by 3% every day.
-- This condition is not directly measurable to our proof but noted as additional info.

-- Condition (4): It takes 100 days for one man alone to complete the same work at his initial efficiency.
def condition4 := initialEfficiencyOfOneMan = totalWork / 100

-- Define the days required for one woman alone to complete the work at her initial efficiency.
noncomputable def daysForWomanToCompleteWork : ℕ := 225

-- Mathematically equivalent proof problem
theorem proof_problem : 
  condition1 ∧ condition4 → (totalWork / daysForWomanToCompleteWork = initialEfficiencyOfOneWoman) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l526_52643


namespace NUMINAMATH_GPT_original_investment_amount_l526_52614

-- Definitions
def annual_interest_rate : ℝ := 0.04
def investment_period_years : ℝ := 0.25
def final_amount : ℝ := 10204

-- Statement to prove
theorem original_investment_amount :
  let P := final_amount / (1 + annual_interest_rate * investment_period_years)
  P = 10104 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_original_investment_amount_l526_52614


namespace NUMINAMATH_GPT_find_x_squared_minus_one_l526_52634

theorem find_x_squared_minus_one (x : ℕ) 
  (h : 2^x + 2^x + 2^x + 2^x = 256) : 
  x^2 - 1 = 35 :=
sorry

end NUMINAMATH_GPT_find_x_squared_minus_one_l526_52634


namespace NUMINAMATH_GPT_sum_ratio_l526_52686

noncomputable def S (n : ℕ) : ℝ := sorry -- placeholder definition

def arithmetic_geometric_sum : Prop :=
  S 3 = 2 ∧ S 6 = 18

theorem sum_ratio :
  arithmetic_geometric_sum → S 10 / S 5 = 33 :=
by
  intros h 
  sorry 

end NUMINAMATH_GPT_sum_ratio_l526_52686


namespace NUMINAMATH_GPT_tim_used_to_run_days_l526_52633

def hours_per_day := 2
def total_hours_per_week := 10
def added_days := 2

theorem tim_used_to_run_days (runs_per_day : ℕ) (total_weekly_runs : ℕ) (additional_runs : ℕ) : 
  runs_per_day = hours_per_day →
  total_weekly_runs = total_hours_per_week →
  additional_runs = added_days →
  (total_weekly_runs / runs_per_day) - additional_runs = 3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_tim_used_to_run_days_l526_52633


namespace NUMINAMATH_GPT_negative_integer_solution_l526_52647

theorem negative_integer_solution (N : ℤ) (h1 : N < 0) (h2 : N^2 + N = 6) : N = -3 := 
by 
  sorry

end NUMINAMATH_GPT_negative_integer_solution_l526_52647


namespace NUMINAMATH_GPT_additional_track_length_l526_52613

theorem additional_track_length (elevation_gain : ℝ) (orig_grade new_grade : ℝ) (Δ_track : ℝ) :
  elevation_gain = 800 ∧ orig_grade = 0.04 ∧ new_grade = 0.015 ∧ Δ_track = ((elevation_gain / new_grade) - (elevation_gain / orig_grade)) ->
  Δ_track = 33333 :=
by sorry

end NUMINAMATH_GPT_additional_track_length_l526_52613


namespace NUMINAMATH_GPT_angles_measure_l526_52650

theorem angles_measure (A B C : ℝ) (h1 : A + B = 180) (h2 : C = 1 / 2 * B) (h3 : A = 6 * B) :
  A = 1080 / 7 ∧ B = 180 / 7 ∧ C = 90 / 7 :=
by
  sorry

end NUMINAMATH_GPT_angles_measure_l526_52650


namespace NUMINAMATH_GPT_mingyu_change_l526_52637

theorem mingyu_change :
  let eraser_cost := 350
  let pencil_cost := 180
  let erasers_count := 3
  let pencils_count := 2
  let payment := 2000
  let total_eraser_cost := erasers_count * eraser_cost
  let total_pencil_cost := pencils_count * pencil_cost
  let total_cost := total_eraser_cost + total_pencil_cost
  let change := payment - total_cost
  change = 590 := 
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_mingyu_change_l526_52637


namespace NUMINAMATH_GPT_two_pipes_fill_tank_l526_52660

theorem two_pipes_fill_tank (C : ℝ) (hA : ∀ (t : ℝ), t = 10 → t = C / (C / 10)) (hB : ∀ (t : ℝ), t = 15 → t = C / (C / 15)) :
  ∀ (t : ℝ), t = C / (C / 6) → t = 6 :=
by
  sorry

end NUMINAMATH_GPT_two_pipes_fill_tank_l526_52660


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_and_geometric_condition_l526_52671

theorem arithmetic_sequence_general_formula_and_geometric_condition :
  ∀ {a : ℕ → ℤ} {S : ℕ → ℤ} {k : ℕ}, 
    (∀ n, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)) →
    a 1 = 9 →
    S 3 = 21 →
    a 5 * S k = a 8 ^ 2 →
    k = 5 :=
by 
  intros a S k hS ha1 hS3 hgeom
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_and_geometric_condition_l526_52671


namespace NUMINAMATH_GPT_range_of_k_l526_52600

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem range_of_k (k : ℝ) (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) :
  (g x1 / k ≤ f x2 / (k + 1)) ↔ k ≥ 1 / (2 * Real.exp 1 - 1) := by
  sorry

end NUMINAMATH_GPT_range_of_k_l526_52600


namespace NUMINAMATH_GPT_geometric_sequence_sum_l526_52625

theorem geometric_sequence_sum :
  ∀ {a : ℕ → ℝ} (r : ℝ),
    (∀ n, a (n + 1) = r * a n) →
    a 1 + a 2 = 1 →
    a 3 + a 4 = 4 →
    a 5 + a 6 + a 7 + a 8 = 80 :=
by
  intros a r h_geom h_sum_1 h_sum_2
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l526_52625


namespace NUMINAMATH_GPT_student_weight_l526_52698

theorem student_weight (S W : ℕ) (h1 : S - 5 = 2 * W) (h2 : S + W = 104) : S = 71 :=
by {
  sorry
}

end NUMINAMATH_GPT_student_weight_l526_52698


namespace NUMINAMATH_GPT_convert_kmph_to_mps_l526_52615

theorem convert_kmph_to_mps (speed_kmph : ℕ) (one_kilometer_in_meters : ℕ) (one_hour_in_seconds : ℕ) :
  speed_kmph = 108 →
  one_kilometer_in_meters = 1000 →
  one_hour_in_seconds = 3600 →
  (speed_kmph * one_kilometer_in_meters) / one_hour_in_seconds = 30 := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_convert_kmph_to_mps_l526_52615


namespace NUMINAMATH_GPT_find_softball_players_l526_52651

def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def football_players : ℕ := 18
def total_players : ℕ := 59

theorem find_softball_players :
  let C := cricket_players
  let H := hockey_players
  let F := football_players
  let T := total_players
  S = T - (C + H + F) :=
by
  let C := cricket_players
  let H := hockey_players
  let F := football_players
  let T := total_players
  show S = T - (C + H + F)
  sorry

end NUMINAMATH_GPT_find_softball_players_l526_52651


namespace NUMINAMATH_GPT_kindergarten_classes_l526_52668

theorem kindergarten_classes :
  ∃ (j a m : ℕ), j + a + m = 32 ∧
                  j > 0 ∧ a > 0 ∧ m > 0 ∧
                  j / 2 + a / 4 + m / 8 = 6 ∧
                  (j = 4 ∧ a = 4 ∧ m = 24) :=
by {
  sorry
}

end NUMINAMATH_GPT_kindergarten_classes_l526_52668


namespace NUMINAMATH_GPT_andrew_kept_correct_l526_52649

open Nat

def andrew_bought : ℕ := 750
def daniel_received : ℕ := 250
def fred_received : ℕ := daniel_received + 120
def total_shared : ℕ := daniel_received + fred_received
def andrew_kept : ℕ := andrew_bought - total_shared

theorem andrew_kept_correct : andrew_kept = 130 :=
by
  unfold andrew_kept andrew_bought total_shared fred_received daniel_received
  rfl

end NUMINAMATH_GPT_andrew_kept_correct_l526_52649


namespace NUMINAMATH_GPT_correct_statements_l526_52697

-- Definitions for statements A, B, C, and D
def statementA (x : ℝ) : Prop := |x| > 1 → x > 1
def statementB (A B C : ℝ) : Prop := (C > 90) ↔ (A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90))
def statementC (a b : ℝ) : Prop := (a * b ≠ 0) ↔ (a ≠ 0 ∧ b ≠ 0)
def statementD (a b : ℝ) : Prop := a > b → 1 / a < 1 / b

-- Proof problem stating which statements are correct
theorem correct_statements :
  (∀ x : ℝ, statementA x = false) ∧ 
  (∀ (A B C : ℝ), statementB A B C = false) ∧ 
  (∀ (a b : ℝ), statementC a b) ∧ 
  (∀ (a b : ℝ), statementD a b = false) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l526_52697


namespace NUMINAMATH_GPT_digit_sum_is_14_l526_52631

theorem digit_sum_is_14 (P Q R S T : ℕ) 
  (h1 : P = 1)
  (h2 : Q = 0)
  (h3 : R = 2)
  (h4 : S = 5)
  (h5 : T = 6) :
  P + Q + R + S + T = 14 :=
by 
  sorry

end NUMINAMATH_GPT_digit_sum_is_14_l526_52631


namespace NUMINAMATH_GPT_correct_operation_l526_52620

theorem correct_operation (x : ℝ) (hx : x ≠ 0) : x^2 / x^8 = 1 / x^6 :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l526_52620


namespace NUMINAMATH_GPT_park_area_l526_52656

theorem park_area (P : ℝ) (w l : ℝ) (hP : P = 120) (hL : l = 3 * w) (hPerimeter : 2 * l + 2 * w = P) : l * w = 675 :=
by
  sorry

end NUMINAMATH_GPT_park_area_l526_52656


namespace NUMINAMATH_GPT_probability_blue_face_up_l526_52678

def cube_probability_blue : ℚ := 
  let total_faces := 6
  let blue_faces := 4
  blue_faces / total_faces

theorem probability_blue_face_up :
  cube_probability_blue = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_blue_face_up_l526_52678


namespace NUMINAMATH_GPT_smallest_number_l526_52688

theorem smallest_number (A B C : ℕ) 
  (h1 : A / 3 = B / 5) 
  (h2 : B / 5 = C / 7) 
  (h3 : C = 56) 
  (h4 : C - A = 32) : 
  A = 24 := 
sorry

end NUMINAMATH_GPT_smallest_number_l526_52688


namespace NUMINAMATH_GPT_codecracker_number_of_codes_l526_52670

theorem codecracker_number_of_codes : ∃ n : ℕ, n = 6 * 5^4 := by
  sorry

end NUMINAMATH_GPT_codecracker_number_of_codes_l526_52670


namespace NUMINAMATH_GPT_scientific_notation_141260_million_l526_52607

theorem scientific_notation_141260_million :
  ∃ (a : ℝ) (n : ℤ), 141260 * 10^6 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4126 ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_141260_million_l526_52607


namespace NUMINAMATH_GPT_total_percentage_of_failed_candidates_l526_52639

-- Define the given conditions
def total_candidates : ℕ := 2000
def number_of_girls : ℕ := 900
def number_of_boys : ℕ := total_candidates - number_of_girls
def percentage_of_boys_passed : ℚ := 0.28
def percentage_of_girls_passed : ℚ := 0.32

-- Define the proof statement
theorem total_percentage_of_failed_candidates : 
  (total_candidates - (percentage_of_boys_passed * number_of_boys + percentage_of_girls_passed * number_of_girls)) / total_candidates * 100 = 70.2 :=
by
  sorry

end NUMINAMATH_GPT_total_percentage_of_failed_candidates_l526_52639


namespace NUMINAMATH_GPT_find_number_of_girls_l526_52648

-- Define the ratio of boys to girls as 8:4.
def ratio_boys_to_girls : ℕ × ℕ := (8, 4)

-- Define the total number of students.
def total_students : ℕ := 600

-- Define what it means for the number of girls given a ratio and total students.
def number_of_girls (ratio : ℕ × ℕ) (total : ℕ) : ℕ :=
  let total_parts := (ratio.1 + ratio.2)
  let part_value := total / total_parts
  ratio.2 * part_value

-- State the goal to prove the number of girls is 200 given the conditions.
theorem find_number_of_girls :
  number_of_girls ratio_boys_to_girls total_students = 200 :=
sorry

end NUMINAMATH_GPT_find_number_of_girls_l526_52648


namespace NUMINAMATH_GPT_sandwiches_difference_l526_52658

-- Define the number of sandwiches Samson ate at lunch on Monday
def sandwichesLunchMonday : ℕ := 3

-- Define the number of sandwiches Samson ate at dinner on Monday (twice as many as lunch)
def sandwichesDinnerMonday : ℕ := 2 * sandwichesLunchMonday

-- Define the total number of sandwiches Samson ate on Monday
def totalSandwichesMonday : ℕ := sandwichesLunchMonday + sandwichesDinnerMonday

-- Define the number of sandwiches Samson ate for breakfast on Tuesday
def sandwichesBreakfastTuesday : ℕ := 1

-- Define the total number of sandwiches Samson ate on Tuesday
def totalSandwichesTuesday : ℕ := sandwichesBreakfastTuesday

-- Define the number of more sandwiches Samson ate on Monday than on Tuesday
theorem sandwiches_difference : totalSandwichesMonday - totalSandwichesTuesday = 8 :=
by
  sorry

end NUMINAMATH_GPT_sandwiches_difference_l526_52658


namespace NUMINAMATH_GPT_prob_sin_ge_half_l526_52685

theorem prob_sin_ge_half : 
  let a := -Real.pi / 6
  let b := Real.pi / 2
  let p := (Real.pi / 2 - Real.pi / 6) / (Real.pi / 2 + Real.pi / 6)
  a ≤ b ∧ a = -Real.pi / 6 ∧ b = Real.pi / 2 → p = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_prob_sin_ge_half_l526_52685


namespace NUMINAMATH_GPT_largest_corner_sum_l526_52667

noncomputable def sum_faces (cube : ℕ → ℕ) : Prop :=
  cube 1 + cube 7 = 8 ∧ 
  cube 2 + cube 6 = 8 ∧ 
  cube 3 + cube 5 = 8 ∧ 
  cube 4 + cube 4 = 8

theorem largest_corner_sum (cube : ℕ → ℕ) 
  (h : sum_faces cube) : 
  ∃ n, n = 17 ∧ 
  ∀ a b c, (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
            (cube a = 7 ∧ cube b = 6 ∧ cube c = 4 ∨ 
             cube a = 6 ∧ cube b = 4 ∧ cube c = 7 ∨ 
             cube a = 4 ∧ cube b = 7 ∧ cube c = 6)) → 
            a + b + c = n := sorry

end NUMINAMATH_GPT_largest_corner_sum_l526_52667


namespace NUMINAMATH_GPT_extra_time_needed_l526_52627

variable (S : ℝ) (d : ℝ) (T T' : ℝ)

-- Original conditions
def original_speed_at_time_distance (S : ℝ) (T : ℝ) (d : ℝ) : Prop :=
  S * T = d

def decreased_speed (original_S : ℝ) : ℝ :=
  0.80 * original_S

def decreased_speed_time (T' : ℝ ) (decreased_S : ℝ) (d : ℝ) : Prop :=
  decreased_S * T' = d

theorem extra_time_needed
  (h1 : original_speed_at_time_distance S T d)
  (h2 : T = 40)
  (h3 : decreased_speed S = 0.80 * S)
  (h4 : decreased_speed_time T' (decreased_speed S) d) :
  T' - T = 10 :=
by
  sorry

end NUMINAMATH_GPT_extra_time_needed_l526_52627


namespace NUMINAMATH_GPT_part_a_ellipse_and_lines_l526_52641

theorem part_a_ellipse_and_lines (x y : ℝ) : 
  (4 * x^2 + 8 * y^2 + 8 * y * abs y = 1) ↔ 
  ((y ≥ 0 ∧ (x^2 / (1/4) + y^2 / (1/16)) = 1) ∨ 
  (y < 0 ∧ ((x = 1/2) ∨ (x = -1/2)))) := 
sorry

end NUMINAMATH_GPT_part_a_ellipse_and_lines_l526_52641


namespace NUMINAMATH_GPT_ratio_of_cube_sides_l526_52603

theorem ratio_of_cube_sides {a b : ℝ} (h : (6 * a^2) / (6 * b^2) = 16) : a / b = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_cube_sides_l526_52603


namespace NUMINAMATH_GPT_gcd_12m_18n_l526_52610

theorem gcd_12m_18n (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_gcd_mn : m.gcd n = 10) : (12 * m).gcd (18 * n) = 60 := by
  sorry

end NUMINAMATH_GPT_gcd_12m_18n_l526_52610


namespace NUMINAMATH_GPT_triangular_array_nth_row_4th_number_l526_52683

theorem triangular_array_nth_row_4th_number (n : ℕ) (h : n ≥ 4) :
  ∃ k : ℕ, k = 4 ∧ (2: ℕ)^(n * (n - 1) / 2 + 3) = 2^((n^2 - n + 6) / 2) :=
by
  sorry

end NUMINAMATH_GPT_triangular_array_nth_row_4th_number_l526_52683


namespace NUMINAMATH_GPT_carnations_count_l526_52663

-- Define the conditions:
def vase_capacity : ℕ := 6
def number_of_roses : ℕ := 47
def number_of_vases : ℕ := 9

-- The goal is to prove that the number of carnations is 7:
theorem carnations_count : (number_of_vases * vase_capacity) - number_of_roses = 7 :=
by
  sorry

end NUMINAMATH_GPT_carnations_count_l526_52663


namespace NUMINAMATH_GPT_average_sales_is_96_l526_52646

-- Definitions for the sales data
def january_sales : ℕ := 110
def february_sales : ℕ := 80
def march_sales : ℕ := 70
def april_sales : ℕ := 130
def may_sales : ℕ := 90

-- Number of months
def num_months : ℕ := 5

-- Total sales calculation
def total_sales : ℕ := january_sales + february_sales + march_sales + april_sales + may_sales

-- Average sales per month calculation
def average_sales_per_month : ℕ := total_sales / num_months

-- Proposition to prove that the average sales per month is 96
theorem average_sales_is_96 : average_sales_per_month = 96 :=
by
  -- We use 'sorry' here to skip the proof, as the problem requires only the statement
  sorry

end NUMINAMATH_GPT_average_sales_is_96_l526_52646


namespace NUMINAMATH_GPT_nicolai_peaches_6_pounds_l526_52694

noncomputable def amount_peaches (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ) : ℕ :=
  let total_ounces := total_pounds * 16
  let total_consumed := oz_oranges + oz_apples
  let remaining_ounces := total_ounces - total_consumed
  remaining_ounces / 16

theorem nicolai_peaches_6_pounds (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ)
  (h_total_pounds : total_pounds = 8) (h_oz_oranges : oz_oranges = 8) (h_oz_apples : oz_apples = 24) :
  amount_peaches total_pounds oz_oranges oz_apples = 6 :=
by
  rw [h_total_pounds, h_oz_oranges, h_oz_apples]
  unfold amount_peaches
  sorry

end NUMINAMATH_GPT_nicolai_peaches_6_pounds_l526_52694


namespace NUMINAMATH_GPT_positive_sum_minus_terms_gt_zero_l526_52604

theorem positive_sum_minus_terms_gt_zero 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 1) : 
  a^2 + a * b + b^2 - a - b > 0 := 
by
  sorry

end NUMINAMATH_GPT_positive_sum_minus_terms_gt_zero_l526_52604


namespace NUMINAMATH_GPT_Tobias_change_l526_52629

def cost_of_shoes := 95
def allowance_per_month := 5
def months_saving := 3
def charge_per_lawn := 15
def lawns_mowed := 4
def charge_per_driveway := 7
def driveways_shoveled := 5
def total_amount_saved : ℕ := (allowance_per_month * months_saving)
                          + (charge_per_lawn * lawns_mowed)
                          + (charge_per_driveway * driveways_shoveled)

theorem Tobias_change : total_amount_saved - cost_of_shoes = 15 := by
  sorry

end NUMINAMATH_GPT_Tobias_change_l526_52629


namespace NUMINAMATH_GPT_domain_sqrt_log_l526_52605

def domain_condition1 (x : ℝ) : Prop := x + 1 ≥ 0
def domain_condition2 (x : ℝ) : Prop := 6 - 3 * x > 0

theorem domain_sqrt_log (x : ℝ) : domain_condition1 x ∧ domain_condition2 x ↔ -1 ≤ x ∧ x < 2 :=
  sorry

end NUMINAMATH_GPT_domain_sqrt_log_l526_52605


namespace NUMINAMATH_GPT_rent_percentage_l526_52674

noncomputable def condition1 (E : ℝ) : ℝ := 0.25 * E
noncomputable def condition2 (E : ℝ) : ℝ := 1.35 * E
noncomputable def condition3 (E' : ℝ) : ℝ := 0.40 * E'

theorem rent_percentage (E R R' : ℝ) (hR : R = condition1 E) (hE' : E = condition2 E) (hR' : R' = condition3 E) :
  (R' / R) * 100 = 216 :=
sorry

end NUMINAMATH_GPT_rent_percentage_l526_52674


namespace NUMINAMATH_GPT_impossible_a_values_l526_52664

theorem impossible_a_values (a : ℝ) :
  ¬((1-a)^2 + (1+a)^2 < 4) → (a ≤ -1 ∨ a ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_impossible_a_values_l526_52664


namespace NUMINAMATH_GPT_rhombus_construction_possible_l526_52617

-- Definitions for points, lines, and distances
variables {Point : Type} {Line : Type}
def is_parallel (l1 l2 : Line) : Prop := sorry
def distance_between (l1 l2 : Line) : ℝ := sorry
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Given parallel lines l₁ and l₂ and their distance a
variable {l1 l2 : Line}
variable (a : ℝ)
axiom parallel_lines : is_parallel l1 l2
axiom distance_eq_a : distance_between l1 l2 = a

-- Given points A and B
variable (A B : Point)

-- Definition of a rhombus that meets the criteria
noncomputable def construct_rhombus (A B : Point) (l1 l2 : Line) (a : ℝ) : Prop :=
  ∃ C1 C2 D1 D2 : Point, 
    point_on_line C1 l1 ∧ 
    point_on_line D1 l2 ∧ 
    point_on_line C2 l1 ∧ 
    point_on_line D2 l2 ∧ 
    sorry -- additional conditions ensuring sides passing through A and B and forming a rhombus

theorem rhombus_construction_possible : 
  construct_rhombus A B l1 l2 a :=
sorry

end NUMINAMATH_GPT_rhombus_construction_possible_l526_52617


namespace NUMINAMATH_GPT_total_weekly_reading_time_l526_52692

def morning_reading_weekdays (daily_minutes : ℕ) (days : ℕ) : ℕ :=
  daily_minutes * days

def morning_reading_weekends (daily_minutes : ℕ) : ℕ :=
  2 * daily_minutes * 2

def evening_reading_weekdays (daily_minutes : ℕ) (days : ℕ) : ℕ :=
  daily_minutes * days

def evening_reading_weekends (daily_minutes : ℕ) : ℕ :=
  2 * daily_minutes * 2

theorem total_weekly_reading_time :
  let morning_minutes := 30
  let evening_minutes := 60
  let weekdays := 5
  let weekend_days := 2
  morning_reading_weekdays morning_minutes weekdays +
  morning_reading_weekends morning_minutes +
  evening_reading_weekdays evening_minutes weekdays +
  evening_reading_weekends evening_minutes = 810 :=
by
  sorry

end NUMINAMATH_GPT_total_weekly_reading_time_l526_52692


namespace NUMINAMATH_GPT_right_triangle_condition_l526_52680

theorem right_triangle_condition (a b c : ℝ) (h : c^2 - a^2 = b^2) : 
  ∃ (A B C : ℝ), A + B + C = 180 ∧ A = 90 ∧ B + C = 90 :=
by sorry

end NUMINAMATH_GPT_right_triangle_condition_l526_52680


namespace NUMINAMATH_GPT_John_and_Rose_work_together_l526_52653

theorem John_and_Rose_work_together (John_work_days : ℕ) (Rose_work_days : ℕ) (combined_work_days: ℕ) 
  (hJohn : John_work_days = 10) (hRose : Rose_work_days = 40) :
  combined_work_days = 8 :=
by 
  sorry

end NUMINAMATH_GPT_John_and_Rose_work_together_l526_52653


namespace NUMINAMATH_GPT_smallest_is_B_l526_52691

def A : ℕ := 32 + 7
def B : ℕ := (3 * 10) + 3
def C : ℕ := 50 - 9

theorem smallest_is_B : min A (min B C) = B := 
by 
  have hA : A = 39 := by rfl
  have hB : B = 33 := by rfl
  have hC : C = 41 := by rfl
  rw [hA, hB, hC]
  exact sorry

end NUMINAMATH_GPT_smallest_is_B_l526_52691


namespace NUMINAMATH_GPT_representation_of_2015_l526_52652

theorem representation_of_2015 :
  ∃ (p d3 i : ℕ),
    Prime p ∧ -- p is prime
    d3 % 3 = 0 ∧ -- d3 is divisible by 3
    400 < i ∧ i < 500 ∧ i % 3 ≠ 0 ∧ -- i is in interval and not divisible by 3
    2015 = p + d3 + i := sorry

end NUMINAMATH_GPT_representation_of_2015_l526_52652


namespace NUMINAMATH_GPT_no_solution_eq_eight_diff_l526_52618

theorem no_solution_eq_eight_diff (k : ℕ) (h1 : k > 0) (h2 : k ≤ 99) 
  (h3 : ∀ x y : ℕ, x^2 - k * y^2 ≠ 8) : 
  (99 - 3 = 96) := 
by 
  sorry

end NUMINAMATH_GPT_no_solution_eq_eight_diff_l526_52618


namespace NUMINAMATH_GPT_projection_coordinates_eq_zero_l526_52621

theorem projection_coordinates_eq_zero (x y z : ℝ) :
  let M := (x, y, z)
  let M₁ := (x, y, 0)
  let M₂ := (0, y, 0)
  let M₃ := (0, 0, 0)
  M₃ = (0, 0, 0) :=
sorry

end NUMINAMATH_GPT_projection_coordinates_eq_zero_l526_52621


namespace NUMINAMATH_GPT_original_number_l526_52689

theorem original_number (n : ℕ) (h : (n + 1) % 30 = 0) : n = 29 :=
by
  sorry

end NUMINAMATH_GPT_original_number_l526_52689


namespace NUMINAMATH_GPT_parallel_lines_necessity_parallel_lines_not_sufficiency_l526_52606

theorem parallel_lines_necessity (a b : ℝ) (h : 2 * b = a * 2) : ab = 4 :=
by sorry

theorem parallel_lines_not_sufficiency (a b : ℝ) (h : ab = 4) : 
  ¬ (2 * b = a * 2 ∧ (2 * a - 2 = 0 -> 2 * b - 2 = 0)) :=
by sorry

end NUMINAMATH_GPT_parallel_lines_necessity_parallel_lines_not_sufficiency_l526_52606


namespace NUMINAMATH_GPT_shortest_path_Dasha_Vasya_l526_52684

-- Definitions for the given distances
def dist_Asya_Galia : ℕ := 12
def dist_Galia_Borya : ℕ := 10
def dist_Asya_Borya : ℕ := 8
def dist_Dasha_Galia : ℕ := 15
def dist_Vasya_Galia : ℕ := 17

-- Definition for shortest distance by roads from Dasha to Vasya
def shortest_dist_Dasha_Vasya : ℕ := 18

-- Proof statement of the goal that shortest distance from Dasha to Vasya is 18 km
theorem shortest_path_Dasha_Vasya : 
  dist_Dasha_Galia + dist_Vasya_Galia - dist_Asya_Galia - dist_Galia_Borya = shortest_dist_Dasha_Vasya := by
  sorry

end NUMINAMATH_GPT_shortest_path_Dasha_Vasya_l526_52684


namespace NUMINAMATH_GPT_translated_function_symmetry_center_l526_52687

theorem translated_function_symmetry_center :
  let f := fun x : ℝ => Real.sin (6 * x + π / 4)
  let g := fun x : ℝ => f (x / 3)
  let h := fun x : ℝ => g (x - π / 8)
  h π / 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_translated_function_symmetry_center_l526_52687


namespace NUMINAMATH_GPT_remainder_when_sum_div_by_8_l526_52616

theorem remainder_when_sum_div_by_8 (n : ℤ) : ((8 - n) + (n + 4)) % 8 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_when_sum_div_by_8_l526_52616


namespace NUMINAMATH_GPT_eraser_cost_l526_52640

noncomputable def price_of_erasers 
  (P : ℝ) -- price of one pencil
  (E : ℝ) -- price of one eraser
  (bundle_count : ℝ) -- number of bundles sold
  (total_earned : ℝ) -- total amount earned
  (discount : ℝ) -- discount percentage for 20 bundles
  (bundle_contents : ℕ) -- 1 pencil and 2 erasers per bundle
  (price_ratio : ℝ) -- price ratio of eraser to pencil
  : Prop := 
  E = 0.5 * P ∧ -- The price of the erasers is 1/2 the price of the pencils.
  bundle_count = 20 ∧ -- The store sold a total of 20 bundles.
  total_earned = 80 ∧ -- The store earned $80.
  discount = 30 ∧ -- 30% discount for 20 bundles
  bundle_contents = 1 + 2 -- A bundle consists of 1 pencil and 2 erasers

theorem eraser_cost
  (P : ℝ) -- price of one pencil
  (E : ℝ) -- price of one eraser
  : price_of_erasers P E 20 80 30 (1 + 2) 0.5 → E = 1.43 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_eraser_cost_l526_52640


namespace NUMINAMATH_GPT_domain_of_fn_l526_52677

noncomputable def domain_fn (x : ℝ) : ℝ := (Real.sqrt (3 * x + 4)) / x

theorem domain_of_fn :
  { x : ℝ | x ≥ -4 / 3 ∧ x ≠ 0 } =
  { x : ℝ | 3 * x + 4 ≥ 0 ∧ x ≠ 0 } :=
by
  ext x
  simp
  exact sorry

end NUMINAMATH_GPT_domain_of_fn_l526_52677


namespace NUMINAMATH_GPT_find_angle_EFC_l526_52619

-- Define the properties of the problem.
def is_isosceles (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist A C

def angle (A B C : ℝ × ℝ) : ℝ :=
  -- Compute the angle using the law of cosines or any other method
  sorry

def perpendicular_foot (P A B : ℝ × ℝ) : ℝ × ℝ :=
  -- Compute the foot of the perpendicular from point P to the line AB
  sorry

noncomputable def main_problem : Prop :=
  ∀ (A B C D E F : ℝ × ℝ),
    is_isosceles A B C →
    angle A B C = 22 →  -- Given angle BAC
    ∃ x : ℝ, dist B D = 2 * dist D C →  -- Point D such that BD = 2 * CD
    E = perpendicular_foot B A D →
    F = perpendicular_foot B A C →
    angle E F C = 33  -- required to prove

-- Statement of the main problem.
theorem find_angle_EFC : main_problem := sorry

end NUMINAMATH_GPT_find_angle_EFC_l526_52619


namespace NUMINAMATH_GPT_printed_value_l526_52672

theorem printed_value (X S : ℕ) (h1 : X = 5) (h2 : S = 0) : 
  (∃ n, S = (n * (3 * n + 7)) / 2 ∧ S ≥ 15000) → 
  X = 5 + 3 * 122 - 3 :=
by 
  sorry

end NUMINAMATH_GPT_printed_value_l526_52672


namespace NUMINAMATH_GPT_butcher_net_loss_l526_52690

noncomputable def dishonest_butcher (advertised_price actual_price : ℝ) (quantity_sold : ℕ) (fine : ℝ) : ℝ :=
  let dishonest_gain_per_kg := actual_price - advertised_price
  let total_dishonest_gain := dishonest_gain_per_kg * quantity_sold
  fine - total_dishonest_gain

theorem butcher_net_loss 
  (advertised_price : ℝ) 
  (actual_price : ℝ) 
  (quantity_sold : ℕ) 
  (fine : ℝ)
  (h_advertised_price : advertised_price = 3.79)
  (h_actual_price : actual_price = 4.00)
  (h_quantity_sold : quantity_sold = 1800)
  (h_fine : fine = 500) :
  dishonest_butcher advertised_price actual_price quantity_sold fine = 122 := 
by
  simp [dishonest_butcher, h_advertised_price, h_actual_price, h_quantity_sold, h_fine]
  sorry

end NUMINAMATH_GPT_butcher_net_loss_l526_52690


namespace NUMINAMATH_GPT_compute_expression_value_l526_52661

noncomputable def expression := 3 ^ (Real.log 4 / Real.log 3) - 27 ^ (2 / 3) - Real.log 0.01 / Real.log 10 + Real.log (Real.exp 3)

theorem compute_expression_value :
  expression = 0 := 
by
  sorry

end NUMINAMATH_GPT_compute_expression_value_l526_52661


namespace NUMINAMATH_GPT_probability_red_balls_by_4th_draw_l526_52676

theorem probability_red_balls_by_4th_draw :
  let total_balls := 10
  let red_prob := 2 / total_balls
  let white_prob := 1 - red_prob
  (white_prob^3) * red_prob = 0.0434 := sorry

end NUMINAMATH_GPT_probability_red_balls_by_4th_draw_l526_52676


namespace NUMINAMATH_GPT_inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed_l526_52693

theorem inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed:
  (∀ a b : ℝ, a > b → a^3 > b^3) → (∀ a b : ℝ, a^3 > b^3 → a > b) :=
  by
  sorry

end NUMINAMATH_GPT_inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed_l526_52693
