import Mathlib

namespace NUMINAMATH_GPT_max_value_x_1_minus_3x_is_1_over_12_l117_11729

open Real

noncomputable def max_value_of_x_1_minus_3x (x : ℝ) : ℝ :=
  x * (1 - 3 * x)

theorem max_value_x_1_minus_3x_is_1_over_12 :
  ∀ x : ℝ, 0 < x ∧ x < 1 / 3 → max_value_of_x_1_minus_3x x ≤ 1 / 12 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_max_value_x_1_minus_3x_is_1_over_12_l117_11729


namespace NUMINAMATH_GPT_factorization_1_min_value_l117_11764

-- Problem 1: Prove that m² - 4mn + 3n² = (m - 3n)(m - n)
theorem factorization_1 (m n : ℤ) : m^2 - 4*m*n + 3*n^2 = (m - 3*n)*(m - n) :=
by
  sorry

-- Problem 2: Prove that the minimum value of m² - 3m + 2015 is 2012 3/4
theorem min_value (m : ℝ) : ∃ x : ℝ, x = m^2 - 3*m + 2015 ∧ x = 2012 + 3/4 :=
by
  sorry

end NUMINAMATH_GPT_factorization_1_min_value_l117_11764


namespace NUMINAMATH_GPT_insurance_compensation_correct_l117_11714

def actual_damage : ℝ := 300000
def deductible_percent : ℝ := 0.01
def deductible_amount : ℝ := deductible_percent * actual_damage
def insurance_compensation : ℝ := actual_damage - deductible_amount

theorem insurance_compensation_correct : insurance_compensation = 297000 :=
by
  -- To be proved
  sorry

end NUMINAMATH_GPT_insurance_compensation_correct_l117_11714


namespace NUMINAMATH_GPT_parity_of_expression_l117_11724

theorem parity_of_expression {a b c : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : 0 < c) :
  ∃ k : ℕ, 3 ^ a + (b - 1) ^ 2 * c = 2 * k + 1 :=
by
  sorry

end NUMINAMATH_GPT_parity_of_expression_l117_11724


namespace NUMINAMATH_GPT_hours_on_task2_l117_11782

theorem hours_on_task2
    (total_hours_per_week : ℕ) 
    (work_days_per_week : ℕ) 
    (hours_per_day_task1 : ℕ) 
    (hours_reduction_task1 : ℕ)
    (h_total_hours : total_hours_per_week = 40)
    (h_work_days : work_days_per_week = 5)
    (h_hours_task1 : hours_per_day_task1 = 5)
    (h_hours_reduction : hours_reduction_task1 = 5)
    : (total_hours_per_week / 2 / work_days_per_week) = 4 :=
by
  -- Skipping proof with sorry
  sorry

end NUMINAMATH_GPT_hours_on_task2_l117_11782


namespace NUMINAMATH_GPT_volume_of_each_hemisphere_container_is_correct_l117_11783

-- Define the given conditions
def Total_volume : ℕ := 10936
def Number_containers : ℕ := 2734

-- Define the volume of each hemisphere container
def Volume_each_container : ℕ := Total_volume / Number_containers

-- The theorem to prove, asserting the volume is correct
theorem volume_of_each_hemisphere_container_is_correct :
  Volume_each_container  = 4 := by
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_volume_of_each_hemisphere_container_is_correct_l117_11783


namespace NUMINAMATH_GPT_trigonometric_comparison_l117_11790

noncomputable def a : ℝ := Real.sin (3 * Real.pi / 5)
noncomputable def b : ℝ := Real.cos (2 * Real.pi / 5)
noncomputable def c : ℝ := Real.tan (2 * Real.pi / 5)

theorem trigonometric_comparison :
  b < a ∧ a < c :=
by {
  -- Use necessary steps to demonstrate b < a and a < c
  sorry
}

end NUMINAMATH_GPT_trigonometric_comparison_l117_11790


namespace NUMINAMATH_GPT_gcd_of_expression_l117_11700

theorem gcd_of_expression 
  (a b c d : ℕ) :
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (a - b) (c - d)) (a - c)) (b - d)) (a - d)) (b - c) = 12 :=
sorry

end NUMINAMATH_GPT_gcd_of_expression_l117_11700


namespace NUMINAMATH_GPT_interest_years_eq_three_l117_11744

theorem interest_years_eq_three :
  ∀ (x y : ℝ),
    (x + 1720 = 2795) →
    (x * (3 / 100) * 8 = 1720 * (5 / 100) * y) →
    y = 3 :=
by
  intros x y hsum heq
  sorry

end NUMINAMATH_GPT_interest_years_eq_three_l117_11744


namespace NUMINAMATH_GPT_length_of_QR_l117_11793

theorem length_of_QR {P Q R N : Type} 
  (PQ PR QR : ℝ) (QN NR PN : ℝ)
  (h1 : PQ = 5)
  (h2 : PR = 10)
  (h3 : QN = 3 * NR)
  (h4 : PN = 6)
  (h5 : QR = QN + NR) :
  QR = 724 / 3 :=
by sorry

end NUMINAMATH_GPT_length_of_QR_l117_11793


namespace NUMINAMATH_GPT_fred_earnings_over_weekend_l117_11751

-- Fred's earning from delivering newspapers
def earnings_from_newspapers : ℕ := 16

-- Fred's earning from washing cars
def earnings_from_cars : ℕ := 74

-- Fred's total earnings over the weekend
def total_earnings : ℕ := earnings_from_newspapers + earnings_from_cars

-- Proof that total earnings is 90
theorem fred_earnings_over_weekend : total_earnings = 90 :=
by 
  -- sorry statement to skip the proof steps
  sorry

end NUMINAMATH_GPT_fred_earnings_over_weekend_l117_11751


namespace NUMINAMATH_GPT_student_community_arrangement_l117_11718

theorem student_community_arrangement :
  let students := 4
  let communities := 3
  (students.choose 2) * (communities.factorial / (communities - (students - 1)).factorial) = 36 :=
by
  have students := 4
  have communities := 3
  sorry

end NUMINAMATH_GPT_student_community_arrangement_l117_11718


namespace NUMINAMATH_GPT_factorize_expression_l117_11785

theorem factorize_expression (x y : ℝ) : x^2 + x * y + x = x * (x + y + 1) := 
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l117_11785


namespace NUMINAMATH_GPT_quadratic_solution_l117_11757

def quadratic_rewrite (x b c : ℝ) : ℝ := (x + b) * (x + b) + c

theorem quadratic_solution (b c : ℝ)
  (h1 : ∀ x, x^2 + 2100 * x + 4200 = quadratic_rewrite x b c)
  (h2 : c = -b^2 + 4200) :
  c / b = -1034 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l117_11757


namespace NUMINAMATH_GPT_problem_1_solution_problem_2_solution_problem_3_solution_problem_4_solution_l117_11772

noncomputable def problem_1 : Int :=
  (-3) + 5 - (-3)

theorem problem_1_solution : problem_1 = 5 := by
  sorry

noncomputable def problem_2 : ℚ :=
  (-1/3 - 3/4 + 5/6) * (-24)

theorem problem_2_solution : problem_2 = 6 := by
  sorry

noncomputable def problem_3 : ℚ :=
  1 - (1/9) * (-1/2 - 2^2)

theorem problem_3_solution : problem_3 = 3/2 := by
  sorry

noncomputable def problem_4 : ℚ :=
  ((-1)^2023) * (18 - (-2) * 3) / (15 - 3^3)

theorem problem_4_solution : problem_4 = 2 := by
  sorry

end NUMINAMATH_GPT_problem_1_solution_problem_2_solution_problem_3_solution_problem_4_solution_l117_11772


namespace NUMINAMATH_GPT_number_of_paperback_books_l117_11760

variables (P H : ℕ)

theorem number_of_paperback_books (h1 : H = 4) (h2 : P / 3 + 2 * H = 10) : P = 6 := 
by
  sorry

end NUMINAMATH_GPT_number_of_paperback_books_l117_11760


namespace NUMINAMATH_GPT_evaluate_Q_at_2_and_neg2_l117_11770

-- Define the polynomial Q and the conditions
variable {Q : ℤ → ℤ}
variable {m : ℤ}

-- The given conditions
axiom cond1 : Q 0 = m
axiom cond2 : Q 1 = 3 * m
axiom cond3 : Q (-1) = 4 * m

-- The proof goal
theorem evaluate_Q_at_2_and_neg2 : Q 2 + Q (-2) = 22 * m :=
sorry

end NUMINAMATH_GPT_evaluate_Q_at_2_and_neg2_l117_11770


namespace NUMINAMATH_GPT_probability_non_smokers_getting_lung_cancer_l117_11756

theorem probability_non_smokers_getting_lung_cancer 
  (overall_lung_cancer : ℝ)
  (smokers_fraction : ℝ)
  (smokers_lung_cancer : ℝ)
  (non_smokers_lung_cancer : ℝ)
  (H1 : overall_lung_cancer = 0.001)
  (H2 : smokers_fraction = 0.2)
  (H3 : smokers_lung_cancer = 0.004)
  (H4 : overall_lung_cancer = smokers_fraction * smokers_lung_cancer + (1 - smokers_fraction) * non_smokers_lung_cancer) :
  non_smokers_lung_cancer = 0.00025 := by
  sorry

end NUMINAMATH_GPT_probability_non_smokers_getting_lung_cancer_l117_11756


namespace NUMINAMATH_GPT_find_a6_l117_11703

variable {a : ℕ → ℤ} -- Assume we have a sequence of integers
variable (d : ℤ) -- Common difference of the arithmetic sequence

-- Conditions
axiom h1 : a 3 = 7
axiom h2 : a 5 = a 2 + 6

-- Define arithmetic sequence property
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n + d

-- Theorem to prove
theorem find_a6 (h1 : a 3 = 7) (h2 : a 5 = a 2 + 6) (h3 : arithmetic_seq a d) : a 6 = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_a6_l117_11703


namespace NUMINAMATH_GPT_discount_percentage_l117_11741

theorem discount_percentage (shirts : ℕ) (total_cost : ℕ) (price_after_discount : ℕ) 
  (h1 : shirts = 3) (h2 : total_cost = 60) (h3 : price_after_discount = 12) : 
  ∃ discount_percentage : ℕ, discount_percentage = 40 := 
by 
  sorry

end NUMINAMATH_GPT_discount_percentage_l117_11741


namespace NUMINAMATH_GPT_theta_in_fourth_quadrant_l117_11762

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin (2 * θ) < 0) : 
  (∃ k : ℤ, θ = 2 * π * k + 7 * π / 4 ∨ θ = 2 * π * k + π / 4) ∧ θ = 2 * π * k + 7 * π / 4 :=
sorry

end NUMINAMATH_GPT_theta_in_fourth_quadrant_l117_11762


namespace NUMINAMATH_GPT_time_taken_by_Arun_to_cross_train_B_l117_11747

structure Train :=
  (length : ℕ)
  (speed_kmh : ℕ)

def to_m_per_s (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 1000) / 3600

def relative_speed (trainA trainB : Train) : ℕ :=
  to_m_per_s trainA.speed_kmh + to_m_per_s trainB.speed_kmh

def total_length (trainA trainB : Train) : ℕ :=
  trainA.length + trainB.length

def time_to_cross (trainA trainB : Train) : ℕ :=
  total_length trainA trainB / relative_speed trainA trainB

theorem time_taken_by_Arun_to_cross_train_B :
  time_to_cross (Train.mk 175 54) (Train.mk 150 36) = 13 :=
by
  sorry

end NUMINAMATH_GPT_time_taken_by_Arun_to_cross_train_B_l117_11747


namespace NUMINAMATH_GPT_carol_first_round_points_l117_11723

theorem carol_first_round_points (P : ℤ) (h1 : P + 6 - 16 = 7) : P = 17 :=
by
  sorry

end NUMINAMATH_GPT_carol_first_round_points_l117_11723


namespace NUMINAMATH_GPT_proof_statement_l117_11727

-- Assume 5 * 3^x = 243
def condition (x : ℝ) : Prop := 5 * (3:ℝ)^x = 243

-- Define the log base 3 for use in the statement
noncomputable def log_base_3 (y : ℝ) : ℝ := Real.log y / Real.log 3

-- State that if the condition holds, then (x + 2)(x - 2) = 21 - 10 * log_base_3 5 + (log_base_3 5)^2
theorem proof_statement (x : ℝ) (h : condition x) : (x + 2) * (x - 2) = 21 - 10 * log_base_3 5 + (log_base_3 5)^2 := sorry

end NUMINAMATH_GPT_proof_statement_l117_11727


namespace NUMINAMATH_GPT_balloons_floated_away_l117_11712

theorem balloons_floated_away (starting_balloons given_away grabbed_balloons final_balloons flattened_balloons : ℕ)
  (h1 : starting_balloons = 50)
  (h2 : given_away = 10)
  (h3 : grabbed_balloons = 11)
  (h4 : final_balloons = 39)
  : flattened_balloons = starting_balloons - given_away + grabbed_balloons - final_balloons → flattened_balloons = 12 :=
by
  sorry

end NUMINAMATH_GPT_balloons_floated_away_l117_11712


namespace NUMINAMATH_GPT_olympic_triathlon_total_distance_l117_11754

theorem olympic_triathlon_total_distance (x : ℝ) (L S : ℝ)
  (hL : L = 4 * x)
  (hS : S = (3 / 80) * x)
  (h_diff : L - S = 8.5) :
  x + L + S = 51.5 := by
  sorry

end NUMINAMATH_GPT_olympic_triathlon_total_distance_l117_11754


namespace NUMINAMATH_GPT_fraction_simplification_l117_11776

theorem fraction_simplification :
  (1/2 * 1/3 * 1/4 * 1/5 + 3/2 * 3/4 * 3/5) / (1/2 * 2/3 * 2/5) = 41/8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l117_11776


namespace NUMINAMATH_GPT_find_ellipse_focus_l117_11788

theorem find_ellipse_focus :
  ∀ (a b : ℝ), a^2 = 5 → b^2 = 4 → 
  (∀ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1) →
  ((∃ c : ℝ, c^2 = a^2 - b^2) ∧ (∃ x y, x = 0 ∧ (y = 1 ∨ y = -1))) :=
by
  sorry

end NUMINAMATH_GPT_find_ellipse_focus_l117_11788


namespace NUMINAMATH_GPT_loss_percentage_is_75_l117_11745

-- Given conditions
def cost_price_one_book (C : ℝ) : Prop := C > 0
def selling_price_one_book (S : ℝ) : Prop := S > 0
def cost_price_5_equals_selling_price_20 (C S : ℝ) : Prop := 5 * C = 20 * S

-- Proof goal
theorem loss_percentage_is_75 (C S : ℝ) (h1 : cost_price_one_book C) (h2 : selling_price_one_book S) (h3 : cost_price_5_equals_selling_price_20 C S) : 
  ((C - S) / C) * 100 = 75 :=
by
  sorry

end NUMINAMATH_GPT_loss_percentage_is_75_l117_11745


namespace NUMINAMATH_GPT_wire_length_l117_11709

theorem wire_length (r_sphere r_cylinder : ℝ) (V_sphere_eq_V_cylinder : (4/3) * π * r_sphere^3 = π * r_cylinder^2 * 144) :
  r_sphere = 12 → r_cylinder = 4 → 144 = 144 := sorry

end NUMINAMATH_GPT_wire_length_l117_11709


namespace NUMINAMATH_GPT_seunghwa_express_bus_distance_per_min_l117_11742

noncomputable def distance_per_min_on_express_bus (total_distance : ℝ) (total_time : ℝ) (time_on_general : ℝ) (gasoline_general : ℝ) (distance_per_gallon : ℝ) (gasoline_used : ℝ) : ℝ :=
  let distance_general := (gasoline_used * distance_per_gallon) / gasoline_general
  let distance_express := total_distance - distance_general
  let time_express := total_time - time_on_general
  (distance_express / time_express)

theorem seunghwa_express_bus_distance_per_min :
  distance_per_min_on_express_bus 120 110 (70) 6 (40.8) 14 = 0.62 :=
by
  sorry

end NUMINAMATH_GPT_seunghwa_express_bus_distance_per_min_l117_11742


namespace NUMINAMATH_GPT_repeating_decimal_multiplication_l117_11707

theorem repeating_decimal_multiplication :
  (0.0808080808 : ℝ) * (0.3333333333 : ℝ) = (8 / 297) := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_multiplication_l117_11707


namespace NUMINAMATH_GPT_mary_initial_borrowed_books_l117_11759

-- We first define the initial number of books B.
variable (B : ℕ)

-- Next, we encode the conditions into a final condition of having 12 books.
def final_books (B : ℕ) : ℕ := (B - 3 + 5) - 2 + 7

-- The proof problem is to show that B must be 5.
theorem mary_initial_borrowed_books (B : ℕ) (h : final_books B = 12) : B = 5 :=
by
  sorry

end NUMINAMATH_GPT_mary_initial_borrowed_books_l117_11759


namespace NUMINAMATH_GPT_find_a_l117_11739

theorem find_a (a x_0 : ℝ) (h_tangent: (ax_0^3 + 1 = x_0) ∧ (3 * a * x_0^2 = 1)) : a = 4 / 27 :=
sorry

end NUMINAMATH_GPT_find_a_l117_11739


namespace NUMINAMATH_GPT_determine_students_and_benches_l117_11720

theorem determine_students_and_benches (a b s : ℕ) :
  (s = a * b + 5) ∧ (s = 8 * b - 4) →
  ((a = 7 ∧ b = 9 ∧ s = 68) ∨ (a = 5 ∧ b = 3 ∧ s = 20)) :=
by
  sorry

end NUMINAMATH_GPT_determine_students_and_benches_l117_11720


namespace NUMINAMATH_GPT_initial_markers_l117_11732

variable (markers_given : ℕ) (total_markers : ℕ)

theorem initial_markers (h_given : markers_given = 109) (h_total : total_markers = 326) :
  total_markers - markers_given = 217 :=
by
  sorry

end NUMINAMATH_GPT_initial_markers_l117_11732


namespace NUMINAMATH_GPT_staircase_problem_l117_11726

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem staircase_problem (total_steps required_steps : ℕ) (num_two_steps : ℕ) :
  total_steps = 11 ∧ required_steps = 7 ∧ num_two_steps = 4 →
  C 7 4 = 35 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_staircase_problem_l117_11726


namespace NUMINAMATH_GPT_proof_q_is_true_l117_11778

variable (p q : Prop)

-- Assuming the conditions
axiom h1 : p ∨ q   -- p or q is true
axiom h2 : ¬ p     -- not p is true

-- Theorem statement to prove q is true
theorem proof_q_is_true : q :=
by
  sorry

end NUMINAMATH_GPT_proof_q_is_true_l117_11778


namespace NUMINAMATH_GPT_percentage_reduction_l117_11799

theorem percentage_reduction (original reduced : ℝ) (h_original : original = 253.25) (h_reduced : reduced = 195) : 
  ((original - reduced) / original) * 100 = 22.99 :=
by
  sorry

end NUMINAMATH_GPT_percentage_reduction_l117_11799


namespace NUMINAMATH_GPT_bell_peppers_needed_l117_11789

-- Definitions based on the conditions
def large_slices_per_bell_pepper : ℕ := 20
def small_pieces_from_half_slices : ℕ := (20 / 2) * 3
def total_slices_and_pieces_per_bell_pepper : ℕ := large_slices_per_bell_pepper / 2 + small_pieces_from_half_slices
def desired_total_slices_and_pieces : ℕ := 200

-- Proving the number of bell peppers needed
theorem bell_peppers_needed : 
  desired_total_slices_and_pieces / total_slices_and_pieces_per_bell_pepper = 5 := 
by 
  -- Add the proof steps here
  sorry

end NUMINAMATH_GPT_bell_peppers_needed_l117_11789


namespace NUMINAMATH_GPT_largest_integer_x_l117_11794

theorem largest_integer_x (x : ℤ) : (8:ℚ)/11 > (x:ℚ)/15 → x ≤ 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_largest_integer_x_l117_11794


namespace NUMINAMATH_GPT_find_length_of_second_movie_l117_11777

noncomputable def length_of_second_movie := 1.5

theorem find_length_of_second_movie
  (total_free_time : ℝ)
  (first_movie_duration : ℝ)
  (words_read : ℝ)
  (reading_rate : ℝ) : 
  first_movie_duration = 3.5 → 
  total_free_time = 8 → 
  words_read = 1800 → 
  reading_rate = 10 → 
  length_of_second_movie = 1.5 := 
by
  intros h1 h2 h3 h4
  -- Here should be the proof steps, which are abstracted away.
  sorry

end NUMINAMATH_GPT_find_length_of_second_movie_l117_11777


namespace NUMINAMATH_GPT_cost_of_one_dozen_pens_l117_11743

noncomputable def cost_of_one_pen_and_one_pencil_ratio := 5

theorem cost_of_one_dozen_pens
  (cost_pencil : ℝ)
  (cost_3_pens_5_pencils : 3 * (cost_of_one_pen_and_one_pencil_ratio * cost_pencil) + 5 * cost_pencil = 200) :
  12 * (cost_of_one_pen_and_one_pencil_ratio * cost_pencil) = 600 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_dozen_pens_l117_11743


namespace NUMINAMATH_GPT_depth_of_sand_l117_11731

theorem depth_of_sand (h : ℝ) (fraction_above_sand : ℝ) :
  h = 9000 → fraction_above_sand = 1/9 → depth = 342 :=
by
  -- height of the pyramid
  let height := 9000
  -- ratio of submerged height to the total height
  let ratio := (8 / 9)^(1 / 3)
  -- height of the submerged part
  let submerged_height := height * ratio
  -- depth of the sand
  let depth := height - submerged_height
  sorry

end NUMINAMATH_GPT_depth_of_sand_l117_11731


namespace NUMINAMATH_GPT_trapezium_perimeters_l117_11786

theorem trapezium_perimeters (AB BC AD AF : ℝ)
  (h1 : AB = 30) (h2 : BC = 30) (h3 : AD = 25) (h4 : AF = 24) :
  ∃ p : ℝ, (p = 90 ∨ p = 104) :=
by
  sorry

end NUMINAMATH_GPT_trapezium_perimeters_l117_11786


namespace NUMINAMATH_GPT_time_for_first_half_is_15_l117_11725

-- Definitions of the conditions in Lean
def floors := 20
def time_per_floor_next_5 := 5
def time_per_floor_final_5 := 16
def total_time := 120

-- Theorem statement
theorem time_for_first_half_is_15 :
  ∃ T, (T + (5 * time_per_floor_next_5) + (5 * time_per_floor_final_5) = total_time) ∧ (T = 15) :=
by
  sorry

end NUMINAMATH_GPT_time_for_first_half_is_15_l117_11725


namespace NUMINAMATH_GPT_mrs_sheridan_total_cats_l117_11722

-- Definitions from the conditions
def original_cats : Nat := 17
def additional_cats : Nat := 14

-- The total number of cats is the sum of the original and additional cats
def total_cats : Nat := original_cats + additional_cats

-- Statement to prove
theorem mrs_sheridan_total_cats : total_cats = 31 := by
  sorry

end NUMINAMATH_GPT_mrs_sheridan_total_cats_l117_11722


namespace NUMINAMATH_GPT_tennis_tournament_rounds_needed_l117_11738

theorem tennis_tournament_rounds_needed (n : ℕ) (total_participants : ℕ) (win_points loss_points : ℕ) (get_point_no_pair : ℕ) (elimination_loss : ℕ) :
  total_participants = 1152 →
  win_points = 1 →
  loss_points = 0 →
  get_point_no_pair = 1 →
  elimination_loss = 2 →
  n = 14 :=
by
  sorry

end NUMINAMATH_GPT_tennis_tournament_rounds_needed_l117_11738


namespace NUMINAMATH_GPT_find_p_q_r_divisibility_l117_11773

theorem find_p_q_r_divisibility 
  (p q r : ℝ)
  (h_div : ∀ x, (x^4 + 4*x^3 + 6*p*x^2 + 4*q*x + r) % (x^3 + 3*x^2 + 9*x + 3) = 0)
  : (p + q) * r = 15 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_find_p_q_r_divisibility_l117_11773


namespace NUMINAMATH_GPT_prob_two_blue_balls_l117_11796

-- Ball and Urn Definitions
def total_balls : ℕ := 10
def blue_balls_initial : ℕ := 6
def red_balls_initial : ℕ := 4

-- Probabilities
def prob_blue_first_draw : ℚ := blue_balls_initial / total_balls
def prob_blue_second_draw_given_first_blue : ℚ :=
  (blue_balls_initial - 1) / (total_balls - 1)

-- Resulting Probability
def prob_both_blue : ℚ := prob_blue_first_draw * prob_blue_second_draw_given_first_blue

-- Statement to Prove
theorem prob_two_blue_balls :
  prob_both_blue = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_prob_two_blue_balls_l117_11796


namespace NUMINAMATH_GPT_geometric_seq_relation_l117_11702

variables {α : Type*} [Field α]

-- Conditions for the arithmetic sequence (for reference)
def arithmetic_seq_sum (S : ℕ → α) (d : α) : Prop :=
∀ m n : ℕ, S (m + n) = S m + S n + (m * n) * d

-- Conditions for the geometric sequence
def geometric_seq_prod (T : ℕ → α) (q : α) : Prop :=
∀ m n : ℕ, T (m + n) = T m * T n * (q ^ (m * n))

-- Proving the desired relationship
theorem geometric_seq_relation {T : ℕ → α} {q : α} (h : geometric_seq_prod T q) (m n : ℕ) :
  T (m + n) = T m * T n * (q ^ (m * n)) :=
by
  apply h m n

end NUMINAMATH_GPT_geometric_seq_relation_l117_11702


namespace NUMINAMATH_GPT_second_term_arithmetic_seq_l117_11779

variable (a d : ℝ)

theorem second_term_arithmetic_seq (h : a + (a + 2 * d) = 8) : a + d = 4 := by
  sorry

end NUMINAMATH_GPT_second_term_arithmetic_seq_l117_11779


namespace NUMINAMATH_GPT_find_intersection_points_l117_11775

def intersection_points (t α : ℝ) : Prop :=
∃ t α : ℝ,
  (2 + t, -1 - t) = (3 * Real.cos α, 3 * Real.sin α) ∧
  ((2 + t = (1 + Real.sqrt 17) / 2 ∧ -1 - t = (1 - Real.sqrt 17) / 2) ∨
   (2 + t = (1 - Real.sqrt 17) / 2 ∧ -1 - t = (1 + Real.sqrt 17) / 2))

theorem find_intersection_points : intersection_points t α :=
sorry

end NUMINAMATH_GPT_find_intersection_points_l117_11775


namespace NUMINAMATH_GPT_two_crows_problem_l117_11768

def Bird := { P | P = "parrot" ∨ P = "crow"} -- Define possible bird species.

-- Define birds and their statements
def Adam_statement (Adam Carl : Bird) : Prop := Carl = Adam
def Bob_statement (Adam : Bird) : Prop := Adam = "crow"
def Carl_statement (Dave : Bird) : Prop := Dave = "crow"
def Dave_statement (Adam Bob Carl Dave: Bird) : Prop := 
  (if Adam = "parrot" then 1 else 0) + 
  (if Bob = "parrot" then 1 else 0) + 
  (if Carl = "parrot" then 1 else 0) + 
  (if Dave = "parrot" then 1 else 0) ≥ 3

-- The main proposition to prove
def main_statement : Prop :=
  ∃ (Adam Bob Carl Dave : Bird), 
    (Adam_statement Adam Carl) ∧ 
    (Bob_statement Adam) ∧ 
    (Carl_statement Dave) ∧ 
    (Dave_statement Adam Bob Carl Dave) ∧ 
    (if Adam = "crow" then 1 else 0) + 
    (if Bob = "crow" then 1 else 0) + 
    (if Carl = "crow" then 1 else 0) + 
    (if Dave = "crow" then 1 else 0) = 2

-- Proof statement to be filled
theorem two_crows_problem : main_statement :=
by {
  sorry
}

end NUMINAMATH_GPT_two_crows_problem_l117_11768


namespace NUMINAMATH_GPT_geom_mean_does_not_exist_l117_11713

theorem geom_mean_does_not_exist (a b : Real) (h1 : a = 2) (h2 : b = -2) : ¬ ∃ g : Real, g^2 = a * b := 
by
  sorry

end NUMINAMATH_GPT_geom_mean_does_not_exist_l117_11713


namespace NUMINAMATH_GPT_solve_exponential_problem_l117_11765

noncomputable def satisfies_condition (a : ℝ) : Prop :=
  let max_value := if a > 1 then a^2 else a
  let min_value := if a > 1 then a else a^2
  max_value - min_value = a / 2

theorem solve_exponential_problem (a : ℝ) (hpos : a > 0) (hne1 : a ≠ 1) :
  satisfies_condition a ↔ (a = 1 / 2 ∨ a = 3 / 2) :=
sorry

end NUMINAMATH_GPT_solve_exponential_problem_l117_11765


namespace NUMINAMATH_GPT_fixed_point_and_max_distance_eqn_l117_11705

-- Define line l1
def l1 (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y - 8 = 0

-- Define line l2 parallel to l1 passing through origin
def l2 (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y = 0

-- Define line y = x
def line_y_eq_x (x y : ℝ) : Prop :=
  y = x

-- Define line x + y = 0
def line_x_plus_y_eq_0 (x y : ℝ) : Prop :=
  x + y = 0

theorem fixed_point_and_max_distance_eqn :
  (∀ m : ℝ, l1 m 2 2) ∧ (∀ m : ℝ, (l2 m 2 2 → false)) →
  (∃ x y : ℝ, l2 m x y ∧ line_x_plus_y_eq_0 x y) :=
by sorry

end NUMINAMATH_GPT_fixed_point_and_max_distance_eqn_l117_11705


namespace NUMINAMATH_GPT_josie_initial_amount_is_correct_l117_11746

def cost_of_milk := 4.00 / 2
def cost_of_bread := 3.50
def cost_of_detergent_after_coupon := 10.25 - 1.25
def cost_of_bananas := 2 * 0.75
def total_cost := cost_of_milk + cost_of_bread + cost_of_detergent_after_coupon + cost_of_bananas
def leftover := 4.00
def initial_amount := total_cost + leftover

theorem josie_initial_amount_is_correct :
  initial_amount = 20.00 := by
  sorry

end NUMINAMATH_GPT_josie_initial_amount_is_correct_l117_11746


namespace NUMINAMATH_GPT_diagonal_intersection_probability_decagon_l117_11737

noncomputable def probability_diagonal_intersection_in_decagon : ℚ :=
  let vertices := 10
  let total_diagonals := (vertices * (vertices - 3)) / 2
  let total_pairs_of_diagonals := total_diagonals * (total_diagonals - 1) / 2
  let total_intersecting_pairs := (vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24
  total_intersecting_pairs / total_pairs_of_diagonals

theorem diagonal_intersection_probability_decagon (h : probability_diagonal_intersection_in_decagon = 42 / 119) : 
  probability_diagonal_intersection_in_decagon = 42 / 119 :=
sorry

end NUMINAMATH_GPT_diagonal_intersection_probability_decagon_l117_11737


namespace NUMINAMATH_GPT_scheme_choice_l117_11704

variable (x y₁ y₂ : ℕ)

def cost_scheme_1 (x : ℕ) : ℕ := 12 * x + 40

def cost_scheme_2 (x : ℕ) : ℕ := 16 * x

theorem scheme_choice :
  ∀ (x : ℕ), 5 ≤ x → x ≤ 20 →
  (if x < 10 then cost_scheme_2 x < cost_scheme_1 x else
   if x = 10 then cost_scheme_2 x = cost_scheme_1 x else
   cost_scheme_1 x < cost_scheme_2 x) :=
by
  sorry

end NUMINAMATH_GPT_scheme_choice_l117_11704


namespace NUMINAMATH_GPT_average_stamps_collected_per_day_l117_11752

open Nat

-- Define an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + d * (n - 1)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions
def a := 10
def d := 10
def n := 7

-- Prove that the average number of stamps collected over 7 days is 40
theorem average_stamps_collected_per_day : 
  sum_arithmetic_sequence a d n / n = 40 := 
by
  sorry

end NUMINAMATH_GPT_average_stamps_collected_per_day_l117_11752


namespace NUMINAMATH_GPT_option_b_does_not_represent_5x_l117_11761

theorem option_b_does_not_represent_5x (x : ℝ) : 
  (∀ a, a = 5 * x ↔ a = x + x + x + x + x) →
  (¬ (5 * x = x * x * x * x * x)) :=
by
  intro h
  -- Using sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_option_b_does_not_represent_5x_l117_11761


namespace NUMINAMATH_GPT_max_digit_d_of_form_7d733e_multiple_of_33_l117_11730

theorem max_digit_d_of_form_7d733e_multiple_of_33 
  (d e : ℕ) (d_digit : d < 10) (e_digit : e < 10) 
  (multiple_of_33: ∃ k : ℕ, 7 * 10^5 + d * 10^4 + 7 * 10^3 + 33 * 10 + e = 33 * k) 
  : d ≤ 6 := 
sorry

end NUMINAMATH_GPT_max_digit_d_of_form_7d733e_multiple_of_33_l117_11730


namespace NUMINAMATH_GPT_total_businesses_l117_11750

theorem total_businesses (B : ℕ) (h1 : B / 2 + B / 3 + 12 = B) : B = 72 :=
sorry

end NUMINAMATH_GPT_total_businesses_l117_11750


namespace NUMINAMATH_GPT_find_m_2n_3k_l117_11749

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_m_2n_3k (m n k : ℕ) (h1 : m + n = 2021) (h2 : is_prime (m - 3 * k)) (h3 : is_prime (n + k)) :
  m + 2 * n + 3 * k = 2025 ∨ m + 2 * n + 3 * k = 4040 := by
  sorry

end NUMINAMATH_GPT_find_m_2n_3k_l117_11749


namespace NUMINAMATH_GPT_integer_solutions_eq_400_l117_11784

theorem integer_solutions_eq_400 : 
  ∃ (s : Finset (ℤ × ℤ)), (∀ x y, (x, y) ∈ s ↔ |3 * x + 2 * y| + |2 * x + y| = 100) ∧ s.card = 400 :=
sorry

end NUMINAMATH_GPT_integer_solutions_eq_400_l117_11784


namespace NUMINAMATH_GPT_max_product_two_integers_l117_11792

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end NUMINAMATH_GPT_max_product_two_integers_l117_11792


namespace NUMINAMATH_GPT_units_digit_sum_is_9_l117_11708

-- Define the units function
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions
def x := 42 ^ 2
def y := 25 ^ 3

-- Define variables for the units digits of x and y
def units_digit_x := units_digit x
def units_digit_y := units_digit y

-- Define the problem statement to be proven
theorem units_digit_sum_is_9 : units_digit (x + y) = 9 :=
by sorry

end NUMINAMATH_GPT_units_digit_sum_is_9_l117_11708


namespace NUMINAMATH_GPT_cos2_a_plus_sin2_b_eq_one_l117_11740

variable {a b c : ℝ}

theorem cos2_a_plus_sin2_b_eq_one
  (h1 : Real.sin a = Real.cos b)
  (h2 : Real.sin b = Real.cos c)
  (h3 : Real.sin c = Real.cos a) :
  Real.cos a ^ 2 + Real.sin b ^ 2 = 1 := 
  sorry

end NUMINAMATH_GPT_cos2_a_plus_sin2_b_eq_one_l117_11740


namespace NUMINAMATH_GPT_center_of_circle_polar_coords_l117_11758

theorem center_of_circle_polar_coords :
  ∀ (θ : ℝ), ∃ (ρ : ℝ), (ρ, θ) = (2, Real.pi) ∧ ρ = - 4 * Real.cos θ := 
sorry

end NUMINAMATH_GPT_center_of_circle_polar_coords_l117_11758


namespace NUMINAMATH_GPT_ending_number_of_second_range_l117_11715

theorem ending_number_of_second_range :
  let avg100_400 := (100 + 400) / 2
  let avg_50_n := (50 + n) / 2
  avg100_400 = avg_50_n + 100 → n = 250 :=
by
  sorry

end NUMINAMATH_GPT_ending_number_of_second_range_l117_11715


namespace NUMINAMATH_GPT_general_formula_minimum_n_l117_11701

-- Definitions based on given conditions
def arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d
def sum_arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- Conditions of the problem
def a2 : ℤ := -5
def S5 : ℤ := -20

-- Proving the general formula of the sequence
theorem general_formula :
  ∃ a₁ d, arith_seq a₁ d 2 = a2 ∧ sum_arith_seq a₁ d 5 = S5 ∧ (∀ n, arith_seq a₁ d n = n - 7) :=
by
  sorry

-- Proving the minimum value of n for which Sn > an
theorem minimum_n :
  ∃ n : ℕ, (n > 14) ∧ sum_arith_seq (-6) 1 n > arith_seq (-6) 1 n :=
by
  sorry

end NUMINAMATH_GPT_general_formula_minimum_n_l117_11701


namespace NUMINAMATH_GPT_find_f_prime_at_two_l117_11719

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_find_f_prime_at_two_l117_11719


namespace NUMINAMATH_GPT_reflex_angle_at_T_l117_11710

-- Assume points P, Q, R, and S are aligned
def aligned (P Q R S : ℝ × ℝ) : Prop :=
  ∃ a b, ∀ x, x = 0 * a + b + (P.1, Q.1, R.1, S.1)

-- Angles given in the problem
def PQT_angle : ℝ := 150
def RTS_angle : ℝ := 70

-- definition of the reflex angle at T
def reflex_angle (angle : ℝ) : ℝ := 360 - angle

theorem reflex_angle_at_T (P Q R S T : ℝ × ℝ) :
  aligned P Q R S → PQT_angle = 150 → RTS_angle = 70 →
  reflex_angle 40 = 320 :=
by
  sorry

end NUMINAMATH_GPT_reflex_angle_at_T_l117_11710


namespace NUMINAMATH_GPT_fiona_probability_correct_l117_11735

def probability_to_reach_pad14 :=
  (1 / 27) + (1 / 3) = 13 / 27 ∧
  (13 / 27) * (1 / 3) = 13 / 81 ∧
  (13 / 81) * (1 / 3) = 13 / 243 ∧
  (13 / 243) * (1 / 3) = 13 / 729 ∧
  (1 / 81) + (1 / 27) + (1 / 27) = 4 / 81 ∧
  (13 / 729) * (4 / 81) = 52 / 59049

theorem fiona_probability_correct :
  (probability_to_reach_pad14 : Prop) := by
  sorry

end NUMINAMATH_GPT_fiona_probability_correct_l117_11735


namespace NUMINAMATH_GPT_difference_max_min_is_7_l117_11733

-- Define the number of times Kale mowed his lawn during each season
def timesSpring : ℕ := 8
def timesSummer : ℕ := 5
def timesFall : ℕ := 12

-- Statement to prove
theorem difference_max_min_is_7 : 
  (max timesSpring (max timesSummer timesFall)) - (min timesSpring (min timesSummer timesFall)) = 7 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_difference_max_min_is_7_l117_11733


namespace NUMINAMATH_GPT_custom_op_4_8_l117_11769

-- Definition of the custom operation
def custom_op (a b : ℕ) : ℕ := b + b / a

-- Theorem stating the desired equality
theorem custom_op_4_8 : custom_op 4 8 = 10 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_custom_op_4_8_l117_11769


namespace NUMINAMATH_GPT_sales_tax_difference_l117_11797

theorem sales_tax_difference : 
  let price : Float := 50
  let tax1 : Float := 0.0725
  let tax2 : Float := 0.07
  let sales_tax1 := price * tax1
  let sales_tax2 := price * tax2
  sales_tax1 - sales_tax2 = 0.125 := 
by
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l117_11797


namespace NUMINAMATH_GPT_total_pieces_of_candy_l117_11721

-- Define the given conditions
def students : ℕ := 43
def pieces_per_student : ℕ := 8

-- Define the goal, which is proving the total number of pieces of candy is 344
theorem total_pieces_of_candy : students * pieces_per_student = 344 :=
by
  sorry

end NUMINAMATH_GPT_total_pieces_of_candy_l117_11721


namespace NUMINAMATH_GPT_initial_savings_correct_l117_11791

-- Define the constants for ticket prices and number of tickets.
def vip_ticket_price : ℕ := 100
def vip_tickets : ℕ := 2
def regular_ticket_price : ℕ := 50
def regular_tickets : ℕ := 3
def leftover_savings : ℕ := 150

-- Define the total cost of tickets.
def total_cost : ℕ := (vip_ticket_price * vip_tickets) + (regular_ticket_price * regular_tickets)

-- Define the initial savings calculation.
def initial_savings : ℕ := total_cost + leftover_savings

-- Theorem stating the initial savings should be $500.
theorem initial_savings_correct : initial_savings = 500 :=
by
  -- Proof steps can be added here.
  sorry

end NUMINAMATH_GPT_initial_savings_correct_l117_11791


namespace NUMINAMATH_GPT_original_price_petrol_in_euros_l117_11736

theorem original_price_petrol_in_euros
  (P : ℝ) -- The original price of petrol in USD per gallon
  (h1 : 0.865 * P * 7.25 + 0.135 * 325 = 325) -- Condition derived from price reduction and additional gallons
  (h2 : P > 0) -- Ensure original price is positive
  (exchange_rate : ℝ) (h3 : exchange_rate = 1.15) : 
  P / exchange_rate = 38.98 :=
by 
  let price_in_euros := P / exchange_rate 
  have h4 : price_in_euros = 38.98 := sorry
  exact h4

end NUMINAMATH_GPT_original_price_petrol_in_euros_l117_11736


namespace NUMINAMATH_GPT_prime_gt_three_square_minus_one_divisible_by_twentyfour_l117_11798

theorem prime_gt_three_square_minus_one_divisible_by_twentyfour (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) : 24 ∣ (p^2 - 1) :=
sorry

end NUMINAMATH_GPT_prime_gt_three_square_minus_one_divisible_by_twentyfour_l117_11798


namespace NUMINAMATH_GPT_solve_equation_l117_11755

theorem solve_equation (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) (h : a^2 = b * (b + 7)) : 
  (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
by sorry

end NUMINAMATH_GPT_solve_equation_l117_11755


namespace NUMINAMATH_GPT_friends_cant_go_to_movies_l117_11795

theorem friends_cant_go_to_movies (total_friends : ℕ) (friends_can_go : ℕ) (H1 : total_friends = 15) (H2 : friends_can_go = 8) : (total_friends - friends_can_go) = 7 :=
by
  sorry

end NUMINAMATH_GPT_friends_cant_go_to_movies_l117_11795


namespace NUMINAMATH_GPT_xy_power_l117_11717

def x : ℚ := 3/4
def y : ℚ := 4/3

theorem xy_power : x^7 * y^8 = 4/3 := by
  sorry

end NUMINAMATH_GPT_xy_power_l117_11717


namespace NUMINAMATH_GPT_novice_experienced_parts_l117_11734

variables (x y : ℕ)

theorem novice_experienced_parts :
  (y - x = 30) ∧ (x + 2 * y = 180) :=
sorry

end NUMINAMATH_GPT_novice_experienced_parts_l117_11734


namespace NUMINAMATH_GPT_second_number_is_three_l117_11774

theorem second_number_is_three (x y : ℝ) (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) : y = 3 :=
by
  -- To be proved: sorry for now
  sorry

end NUMINAMATH_GPT_second_number_is_three_l117_11774


namespace NUMINAMATH_GPT_cyclist_speed_l117_11787

theorem cyclist_speed 
  (course_length : ℝ)
  (second_cyclist_speed : ℝ)
  (meeting_time : ℝ)
  (total_distance : ℝ)
  (condition1 : course_length = 45)
  (condition2 : second_cyclist_speed = 16)
  (condition3 : meeting_time = 1.5)
  (condition4 : total_distance = meeting_time * (second_cyclist_speed + 14))
  : (meeting_time * 14 + meeting_time * second_cyclist_speed = course_length) :=
by
  sorry

end NUMINAMATH_GPT_cyclist_speed_l117_11787


namespace NUMINAMATH_GPT_compute_expression_l117_11711

theorem compute_expression : 12 * (1 / 26) * 52 * 4 = 96 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l117_11711


namespace NUMINAMATH_GPT_range_of_d_l117_11763

variable {S : ℕ → ℝ} -- S is the sum of the series
variable {a : ℕ → ℝ} -- a is the arithmetic sequence

theorem range_of_d (d : ℝ) (h1 : a 3 = 12) (h2 : S 12 > 0) (h3 : S 13 < 0) :
  -24 / 7 < d ∧ d < -3 := sorry

end NUMINAMATH_GPT_range_of_d_l117_11763


namespace NUMINAMATH_GPT_andrew_age_l117_11753

variables (a g : ℝ)

theorem andrew_age (h1 : g = 15 * a) (h2 : g - a = 60) : a = 30 / 7 :=
by sorry

end NUMINAMATH_GPT_andrew_age_l117_11753


namespace NUMINAMATH_GPT_part_I_part_II_l117_11780

noncomputable def f (x a : ℝ) : ℝ := |x + 1| - |x - a|

theorem part_I (x : ℝ) : (∃ a : ℝ, a = 1 ∧ f x a < 1) ↔ x < (1/2) :=
sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≤ 6) ↔ (a = 5 ∨ a = -7) :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l117_11780


namespace NUMINAMATH_GPT_trig_identity_l117_11716

theorem trig_identity :
  (4 * (1 / 2) - Real.sqrt 2 * (Real.sqrt 2 / 2) - Real.sqrt 3 * (Real.sqrt 3 / 3) + 2 * (Real.sqrt 3 / 2)) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_trig_identity_l117_11716


namespace NUMINAMATH_GPT_parabola_directrix_l117_11766

theorem parabola_directrix (y x : ℝ) (h : y^2 = -4 * x) : x = 1 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l117_11766


namespace NUMINAMATH_GPT_problem_remainders_l117_11767

open Int

theorem problem_remainders (x : ℤ) :
  (x + 2) % 45 = 7 →
  ((x + 2) % 20 = 7 ∧ x % 19 = 5) :=
by
  sorry

end NUMINAMATH_GPT_problem_remainders_l117_11767


namespace NUMINAMATH_GPT_total_wheels_l117_11748

def regular_bikes := 7
def children_bikes := 11
def tandem_bikes_4_wheels := 5
def tandem_bikes_6_wheels := 3
def unicycles := 4
def tricycles := 6
def bikes_with_training_wheels := 8

def wheels_regular := 2
def wheels_children := 4
def wheels_tandem_4 := 4
def wheels_tandem_6 := 6
def wheel_unicycle := 1
def wheels_tricycle := 3
def wheels_training := 4

theorem total_wheels : 
  (regular_bikes * wheels_regular) +
  (children_bikes * wheels_children) + 
  (tandem_bikes_4_wheels * wheels_tandem_4) + 
  (tandem_bikes_6_wheels * wheels_tandem_6) + 
  (unicycles * wheel_unicycle) + 
  (tricycles * wheels_tricycle) + 
  (bikes_with_training_wheels * wheels_training) 
  = 150 := by
  sorry

end NUMINAMATH_GPT_total_wheels_l117_11748


namespace NUMINAMATH_GPT_fraction_decomposition_l117_11781

theorem fraction_decomposition :
  ∀ (A B : ℚ), (∀ x : ℚ, x ≠ -2 → x ≠ 4/3 → 
  (7 * x - 15) / ((3 * x - 4) * (x + 2)) = A / (x + 2) + B / (3 * x - 4)) →
  A = 29 / 10 ∧ B = -17 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_decomposition_l117_11781


namespace NUMINAMATH_GPT_g_of_g_of_g_of_20_l117_11706

def g (x : ℕ) : ℕ :=
  if x < 10 then x^2 - 9 else x - 15

theorem g_of_g_of_g_of_20 : g (g (g 20)) = 1 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_g_of_g_of_g_of_20_l117_11706


namespace NUMINAMATH_GPT_fraction_addition_simplified_form_l117_11771

theorem fraction_addition_simplified_form :
  (7 / 8) + (3 / 5) = 59 / 40 := 
by sorry

end NUMINAMATH_GPT_fraction_addition_simplified_form_l117_11771


namespace NUMINAMATH_GPT_find_m_max_value_l117_11728

noncomputable def f (x : ℝ) := |x - 1|

theorem find_m (m : ℝ) :
  (∀ x, f (x + 5) ≤ 3 * m) ∧ m > 0 ∧ (∀ x, -7 ≤ x ∧ x ≤ -1 → f (x + 5) ≤ 3 * m) →
  m = 1 :=
by
  sorry

theorem max_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h2 : 2 * a ^ 2 + b ^ 2 = 3) :
  ∃ x, (∀ a b, 2 * a * Real.sqrt (1 + b ^ 2) ≤ x) ∧ x = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_max_value_l117_11728
