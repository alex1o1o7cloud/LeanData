import Mathlib

namespace NUMINAMATH_GPT_set_intersection_complement_l470_47094

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem set_intersection_complement :
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l470_47094


namespace NUMINAMATH_GPT_minimum_value_expression_l470_47028

theorem minimum_value_expression (a : ℝ) (h : a > 0) : 
  a + (a + 4) / a ≥ 5 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l470_47028


namespace NUMINAMATH_GPT_six_n_digit_remains_divisible_by_7_l470_47050

-- Given the conditions
def is_6n_digit_number (N : ℕ) (n : ℕ) : Prop :=
  N < 10^(6*n) ∧ N ≥ 10^(6*(n-1))

def is_divisible_by_7 (N : ℕ) : Prop :=
  N % 7 = 0

-- Define new number M formed by moving the unit digit to the beginning
def new_number (N : ℕ) (n : ℕ) : ℕ :=
  let a_0 := N % 10
  let rest := N / 10
  a_0 * 10^(6*n - 1) + rest

-- The theorem statement
theorem six_n_digit_remains_divisible_by_7 (N : ℕ) (n : ℕ)
  (hN : is_6n_digit_number N n)
  (hDiv7 : is_divisible_by_7 N) : is_divisible_by_7 (new_number N n) :=
sorry

end NUMINAMATH_GPT_six_n_digit_remains_divisible_by_7_l470_47050


namespace NUMINAMATH_GPT_sum_of_first_9_terms_l470_47041

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)
variable (a1 : ℤ)
variable (d : ℤ)

-- Given is that the sequence is arithmetic.
-- Given a1 is the first term, and d is the common difference, we can define properties based on the conditions.
def is_arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a1 + (n - 1) * d

def sum_first_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given condition: 2a_1 + a_13 = -9.
def given_condition (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ) : Prop :=
  2 * a1 + (a1 + 12 * d) = -9

theorem sum_of_first_9_terms (a : ℕ → ℤ) (S : ℕ → ℤ) (a1 d : ℤ)
  (h_arith : is_arithmetic_sequence a a1 d)
  (h_sum : sum_first_n_terms S a)
  (h_cond : given_condition a a1 d) :
  S 9 = -27 :=
sorry

end NUMINAMATH_GPT_sum_of_first_9_terms_l470_47041


namespace NUMINAMATH_GPT_combined_width_approximately_8_l470_47035

noncomputable def C1 := 352 / 7
noncomputable def C2 := 528 / 7
noncomputable def C3 := 704 / 7

noncomputable def r1 := C1 / (2 * Real.pi)
noncomputable def r2 := C2 / (2 * Real.pi)
noncomputable def r3 := C3 / (2 * Real.pi)

noncomputable def W1 := r2 - r1
noncomputable def W2 := r3 - r2

noncomputable def combined_width := W1 + W2

theorem combined_width_approximately_8 :
  |combined_width - 8| < 1 :=
by
  sorry

end NUMINAMATH_GPT_combined_width_approximately_8_l470_47035


namespace NUMINAMATH_GPT_sum_of_edges_96_l470_47046

noncomputable def volume (a r : ℝ) : ℝ := 
  (a / r) * a * (a * r)

noncomputable def surface_area (a r : ℝ) : ℝ := 
  2 * ((a^2) / r + a^2 + a^2 * r)

noncomputable def sum_of_edges (a r : ℝ) : ℝ := 
  4 * ((a / r) + a + (a * r))

theorem sum_of_edges_96 :
  (∃ (a r : ℝ), volume a r = 512 ∧ surface_area a r = 384 ∧ sum_of_edges a r = 96) :=
by
  have a := 8
  have r := 1
  have h_volume : volume a r = 512 := sorry
  have h_surface_area : surface_area a r = 384 := sorry
  have h_sum_of_edges : sum_of_edges a r = 96 := sorry
  exact ⟨a, r, h_volume, h_surface_area, h_sum_of_edges⟩

end NUMINAMATH_GPT_sum_of_edges_96_l470_47046


namespace NUMINAMATH_GPT_incorrect_reciprocal_quotient_l470_47032

-- Definitions based on problem conditions
def identity_property (x : ℚ) : x * 1 = x := by sorry
def division_property (a b : ℚ) (h : b ≠ 0) : a / b = 0 → a = 0 := by sorry
def additive_inverse_property (x : ℚ) : x * (-1) = -x := by sorry

-- Statement that needs to be proved
theorem incorrect_reciprocal_quotient (a b : ℚ) (h1 : a ≠ 0) (h2 : b = 1 / a) : a / b ≠ 1 :=
by sorry

end NUMINAMATH_GPT_incorrect_reciprocal_quotient_l470_47032


namespace NUMINAMATH_GPT_time_difference_leak_l470_47023

/-- 
The machine usually fills one barrel in 3 minutes. 
However, with a leak, it takes 5 minutes to fill one barrel. 
Given that it takes 24 minutes longer to fill 12 barrels with the leak, prove that it will take 2n minutes longer to fill n barrels with the leak.
-/
theorem time_difference_leak (n : ℕ) : 
  (3 * 12 + 24 = 5 * 12) →
  (5 * n) - (3 * n) = 2 * n :=
by
  intros h
  sorry

end NUMINAMATH_GPT_time_difference_leak_l470_47023


namespace NUMINAMATH_GPT_horner_evaluation_at_3_l470_47075

def f (x : ℤ) : ℤ := x^5 + 2 * x^3 + 3 * x^2 + x + 1

theorem horner_evaluation_at_3 : f 3 = 328 := by
  sorry

end NUMINAMATH_GPT_horner_evaluation_at_3_l470_47075


namespace NUMINAMATH_GPT_train_passes_man_in_approximately_18_seconds_l470_47052

noncomputable def length_of_train : ℝ := 330 -- meters
noncomputable def speed_of_train : ℝ := 60 -- kmph
noncomputable def speed_of_man : ℝ := 6 -- kmph

noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * (5/18)

noncomputable def relative_speed_mps : ℝ :=
  kmph_to_mps (speed_of_train + speed_of_man)

noncomputable def time_to_pass (length : ℝ) (speed : ℝ) : ℝ := length / speed

theorem train_passes_man_in_approximately_18_seconds :
  abs (time_to_pass length_of_train relative_speed_mps - 18) < 1 :=
by
  sorry

end NUMINAMATH_GPT_train_passes_man_in_approximately_18_seconds_l470_47052


namespace NUMINAMATH_GPT_average_test_score_fifty_percent_l470_47042

-- Given conditions
def percent1 : ℝ := 15
def avg1 : ℝ := 100
def percent2 : ℝ := 50
def avg3 : ℝ := 63
def overall_average : ℝ := 76.05

-- Intermediate calculations based on given conditions
def total_percent : ℝ := 100
def percent3: ℝ := total_percent - percent1 - percent2
def sum_of_weights: ℝ := overall_average * total_percent

-- Expected average of the group that is 50% of the class
theorem average_test_score_fifty_percent (X: ℝ) :
  sum_of_weights = percent1 * avg1 + percent2 * X + percent3 * avg3 → X = 78 := by
  sorry

end NUMINAMATH_GPT_average_test_score_fifty_percent_l470_47042


namespace NUMINAMATH_GPT_dima_age_l470_47092

variable (x : ℕ)

-- Dima's age is x years
def age_of_dima := x

-- Dima's age is twice his brother's age
def age_of_brother := x / 2

-- Dima's age is three times his sister's age
def age_of_sister := x / 3

-- The average age of Dima, his sister, and his brother is 11 years
def average_age := (x + age_of_brother x + age_of_sister x) / 3 = 11

theorem dima_age (h1 : age_of_brother x = x / 2) 
                 (h2 : age_of_sister x = x / 3) 
                 (h3 : average_age x) : x = 18 := 
by sorry

end NUMINAMATH_GPT_dima_age_l470_47092


namespace NUMINAMATH_GPT_decompose_fraction1_decompose_fraction2_l470_47045

-- Define the first problem as a theorem
theorem decompose_fraction1 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 / (x^2 - 1)) = (1 / (x - 1)) - (1 / (x + 1)) :=
sorry  -- Proof required

-- Define the second problem as a theorem
theorem decompose_fraction2 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 * x / (x^2 - 1)) = (1 / (x - 1)) + (1 / (x + 1)) :=
sorry  -- Proof required

end NUMINAMATH_GPT_decompose_fraction1_decompose_fraction2_l470_47045


namespace NUMINAMATH_GPT_bob_more_than_ken_l470_47087

def ken_situps : ℕ := 20

def nathan_situps : ℕ := 2 * ken_situps

def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

theorem bob_more_than_ken : bob_situps - ken_situps = 10 := 
sorry

end NUMINAMATH_GPT_bob_more_than_ken_l470_47087


namespace NUMINAMATH_GPT_selected_40th_is_795_l470_47003

-- Definitions of constants based on the problem conditions
def total_participants : ℕ := 1000
def selections : ℕ := 50
def equal_spacing : ℕ := total_participants / selections
def first_selected_number : ℕ := 15
def nth_selected_number (n : ℕ) : ℕ := (n - 1) * equal_spacing + first_selected_number

-- The theorem to prove the 40th selected number is 795
theorem selected_40th_is_795 : nth_selected_number 40 = 795 := 
by 
  -- Skipping the detailed proof
  sorry

end NUMINAMATH_GPT_selected_40th_is_795_l470_47003


namespace NUMINAMATH_GPT_positive_integers_divide_n_plus_7_l470_47013

theorem positive_integers_divide_n_plus_7 (n : ℕ) (hn_pos : 0 < n) : n ∣ n + 7 ↔ n = 1 ∨ n = 7 :=
by 
  sorry

end NUMINAMATH_GPT_positive_integers_divide_n_plus_7_l470_47013


namespace NUMINAMATH_GPT_range_of_sqrt_meaningful_real_l470_47049

theorem range_of_sqrt_meaningful_real (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_sqrt_meaningful_real_l470_47049


namespace NUMINAMATH_GPT_brownies_left_is_zero_l470_47006

-- Definitions of the conditions
def total_brownies : ℝ := 24
def tina_lunch : ℝ := 1.5 * 5
def tina_dinner : ℝ := 0.5 * 5
def tina_total : ℝ := tina_lunch + tina_dinner
def husband_total : ℝ := 0.75 * 5
def guests_total : ℝ := 2.5 * 2
def daughter_total : ℝ := 2 * 3

-- Formulate the proof statement
theorem brownies_left_is_zero :
    total_brownies - (tina_total + husband_total + guests_total + daughter_total) = 0 := by
  sorry

end NUMINAMATH_GPT_brownies_left_is_zero_l470_47006


namespace NUMINAMATH_GPT_point_d_lies_on_graph_l470_47059

theorem point_d_lies_on_graph : (-1 : ℝ) = -2 * (1 : ℝ) + 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_point_d_lies_on_graph_l470_47059


namespace NUMINAMATH_GPT_tan_arithmetic_seq_value_l470_47029

variable {a : ℕ → ℝ}
variable (d : ℝ)

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a n = a 0 + n * d

-- Given conditions and the final proof goal
theorem tan_arithmetic_seq_value (h_arith : arithmetic_seq a d)
    (h_sum : a 0 + a 6 + a 12 = Real.pi) :
    Real.tan (a 1 + a 11) = -Real.sqrt 3 := sorry

end NUMINAMATH_GPT_tan_arithmetic_seq_value_l470_47029


namespace NUMINAMATH_GPT_sequence_increasing_l470_47076

theorem sequence_increasing (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ∀ n : ℕ, a^n / n^b < a^(n+1) / (n+1)^b :=
by sorry

end NUMINAMATH_GPT_sequence_increasing_l470_47076


namespace NUMINAMATH_GPT_intersection_M_N_l470_47084

-- Defining set M
def M : Set ℕ := {1, 2, 3, 4}

-- Defining the set N based on the condition
def N : Set ℕ := {x | ∃ n ∈ M, x = n^2}

-- Lean statement to prove the intersection
theorem intersection_M_N : M ∩ N = {1, 4} := 
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l470_47084


namespace NUMINAMATH_GPT_sanjay_homework_fraction_l470_47099

theorem sanjay_homework_fraction (x : ℚ) :
  (2 * x + 1) / 3 + 4 / 15 = 1 ↔ x = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sanjay_homework_fraction_l470_47099


namespace NUMINAMATH_GPT_abs_neg_three_l470_47090

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end NUMINAMATH_GPT_abs_neg_three_l470_47090


namespace NUMINAMATH_GPT_florist_sold_16_roses_l470_47015

-- Definitions for initial and final states
def initial_roses : ℕ := 37
def picked_roses : ℕ := 19
def final_roses : ℕ := 40

-- Defining the variable for number of roses sold
variable (x : ℕ)

-- The statement to prove
theorem florist_sold_16_roses
  (h : initial_roses - x + picked_roses = final_roses) : x = 16 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_florist_sold_16_roses_l470_47015


namespace NUMINAMATH_GPT_robin_gum_total_l470_47061

theorem robin_gum_total :
  let original_gum := 18.0
  let given_gum := 44.0
  original_gum + given_gum = 62.0 := by
  sorry

end NUMINAMATH_GPT_robin_gum_total_l470_47061


namespace NUMINAMATH_GPT_solution_to_g_inv_2_l470_47014

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := 1 / (c * x + d)

theorem solution_to_g_inv_2 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
    ∃ x : ℝ, g x c d = 2 ↔ x = (1 - 2 * d) / (2 * c) :=
by
  sorry

end NUMINAMATH_GPT_solution_to_g_inv_2_l470_47014


namespace NUMINAMATH_GPT_prob_triangle_includes_G_l470_47025

-- Definitions based on conditions in the problem
def total_triangles : ℕ := 6
def triangles_including_G : ℕ := 4

-- The theorem statement proving the probability
theorem prob_triangle_includes_G : (triangles_including_G : ℚ) / total_triangles = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_prob_triangle_includes_G_l470_47025


namespace NUMINAMATH_GPT_regression_prediction_l470_47012

theorem regression_prediction
  (slope : ℝ) (centroid_x centroid_y : ℝ) (b : ℝ)
  (h_slope : slope = 1.23)
  (h_centroid : centroid_x = 4 ∧ centroid_y = 5)
  (h_intercept : centroid_y = slope * centroid_x + b)
  (x : ℝ) (h_x : x = 10) :
  centroid_y = 5 →
  slope = 1.23 →
  x = 10 →
  b = 5 - 1.23 * 4 →
  (slope * x + b) = 12.38 :=
by
  intros
  sorry

end NUMINAMATH_GPT_regression_prediction_l470_47012


namespace NUMINAMATH_GPT_problem_l470_47027

theorem problem (r : ℝ) (h : (r + 1/r)^4 = 17) : r^6 + 1/r^6 = 1 * Real.sqrt 17 - 6 :=
sorry

end NUMINAMATH_GPT_problem_l470_47027


namespace NUMINAMATH_GPT_negation_of_prop_l470_47031

def prop (x : ℝ) := x^2 ≥ 0

theorem negation_of_prop:
  ¬ ∀ x : ℝ, prop x ↔ ∃ x : ℝ, x^2 < 0 := by
    sorry

end NUMINAMATH_GPT_negation_of_prop_l470_47031


namespace NUMINAMATH_GPT_MEMOrable_rectangle_count_l470_47011

section MEMOrable_rectangles

variables (K L : ℕ) (hK : K > 0) (hL : L > 0) 

/-- In a 2K x 2L board, if the ant starts at (1,1) and ends at (2K, 2L),
    and some squares may remain unvisited forming a MEMOrable rectangle,
    then the number of such MEMOrable rectangles is (K(K+1)L(L+1))/2. -/
theorem MEMOrable_rectangle_count :
  ∃ (n : ℕ), n = K * (K + 1) * L * (L + 1) / 2 :=
by
  sorry

end MEMOrable_rectangles

end NUMINAMATH_GPT_MEMOrable_rectangle_count_l470_47011


namespace NUMINAMATH_GPT_consecutive_numbers_count_l470_47017

-- Definitions and conditions
variables (n : ℕ) (x : ℕ)
axiom avg_condition : (2 * 33 = 2 * x + n - 1)
axiom highest_num_condition : (x + (n - 1) = 36)

-- Thm statement
theorem consecutive_numbers_count : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_numbers_count_l470_47017


namespace NUMINAMATH_GPT_turtle_hare_race_headstart_l470_47063

noncomputable def hare_time_muddy (distance speed_reduction hare_speed : ℝ) : ℝ :=
  distance / (hare_speed * speed_reduction)

noncomputable def hare_time_sandy (distance hare_speed : ℝ) : ℝ :=
  distance / hare_speed

noncomputable def hare_time_regular (distance hare_speed : ℝ) : ℝ :=
  distance / hare_speed

noncomputable def turtle_time_muddy (distance turtle_speed : ℝ) : ℝ :=
  distance / turtle_speed

noncomputable def turtle_time_sandy (distance speed_increase turtle_speed : ℝ) : ℝ :=
  distance / (turtle_speed * speed_increase)

noncomputable def turtle_time_regular (distance turtle_speed : ℝ) : ℝ :=
  distance / turtle_speed

noncomputable def hare_total_time (hare_speed : ℝ) : ℝ :=
  hare_time_muddy 20 0.5 hare_speed + hare_time_sandy 10 hare_speed + hare_time_regular 20 hare_speed

noncomputable def turtle_total_time (turtle_speed : ℝ) : ℝ :=
  turtle_time_muddy 20 turtle_speed + turtle_time_sandy 10 1.5 turtle_speed + turtle_time_regular 20 turtle_speed

theorem turtle_hare_race_headstart (hare_speed turtle_speed : ℝ) (t_hs : ℝ) :
  hare_speed = 10 →
  turtle_speed = 1 →
  t_hs = 39.67 →
  hare_total_time hare_speed + t_hs = turtle_total_time turtle_speed :=
by
  intros 
  sorry

end NUMINAMATH_GPT_turtle_hare_race_headstart_l470_47063


namespace NUMINAMATH_GPT_part_I_part_II_l470_47065

def f (x a : ℝ) : ℝ := abs (x - a) + abs (2 * x + 1)

-- Part (I)
theorem part_I (x : ℝ) : f x 1 ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

-- Part (II)
theorem part_II (a : ℝ) : (∃ x ∈ Set.Ici a, f x a ≤ 2 * a + x) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l470_47065


namespace NUMINAMATH_GPT_xiao_ming_actual_sleep_time_l470_47066

def required_sleep_time : ℝ := 9
def recorded_excess_sleep_time : ℝ := 0.4
def actual_sleep_time (required : ℝ) (excess : ℝ) : ℝ := required + excess

theorem xiao_ming_actual_sleep_time :
  actual_sleep_time required_sleep_time recorded_excess_sleep_time = 9.4 := 
by
  sorry

end NUMINAMATH_GPT_xiao_ming_actual_sleep_time_l470_47066


namespace NUMINAMATH_GPT_cone_volume_in_liters_l470_47067

theorem cone_volume_in_liters (d h : ℝ) (pi : ℝ) (liters_conversion : ℝ) :
  d = 12 → h = 10 → liters_conversion = 1000 → (1/3) * pi * (d/2)^2 * h * (1 / liters_conversion) = 0.12 * pi :=
by
  intros hd hh hc
  sorry

end NUMINAMATH_GPT_cone_volume_in_liters_l470_47067


namespace NUMINAMATH_GPT_minimum_perimeter_triangle_l470_47005

noncomputable def minimum_perimeter (a b c : ℝ) (cos_C : ℝ) (ha : a + b = 10) (hroot : 2 * cos_C^2 - 3 * cos_C - 2 = 0) 
  : ℝ :=
  a + b + c

theorem minimum_perimeter_triangle (a b c : ℝ) (cos_C : ℝ)
  (ha : a + b = 10)
  (hroot : 2 * cos_C^2 - 3 * cos_C - 2 = 0)
  (cos_C_valid : cos_C = -1/2) :
  (minimum_perimeter a b c cos_C ha hroot) = 10 + 5 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_minimum_perimeter_triangle_l470_47005


namespace NUMINAMATH_GPT_min_performances_l470_47072

theorem min_performances (n_pairs_per_show m n_singers : ℕ) (h1 : n_singers = 8) (h2 : n_pairs_per_show = 6) 
  (condition : 6 * m = 28 * 3) : m = 14 :=
by
  -- Use the assumptions to prove the statement
  sorry

end NUMINAMATH_GPT_min_performances_l470_47072


namespace NUMINAMATH_GPT_cuts_needed_l470_47020

-- Define the length of the wood in centimeters
def wood_length_cm : ℕ := 400

-- Define the length of each stake in centimeters
def stake_length_cm : ℕ := 50

-- Define the expected number of cuts needed
def expected_cuts : ℕ := 7

-- The main theorem stating the equivalence
theorem cuts_needed (wood_length stake_length : ℕ) (h1 : wood_length = 400) (h2 : stake_length = 50) :
  (wood_length / stake_length) - 1 = expected_cuts :=
sorry

end NUMINAMATH_GPT_cuts_needed_l470_47020


namespace NUMINAMATH_GPT_sin_alpha_cos_2beta_l470_47088

theorem sin_alpha_cos_2beta :
  ∀ α β : ℝ, 3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2 →
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 :=
by
  intros α β h
  sorry

end NUMINAMATH_GPT_sin_alpha_cos_2beta_l470_47088


namespace NUMINAMATH_GPT_octopus_legs_l470_47058

-- Definitions of octopus behavior based on the number of legs
def tells_truth (legs: ℕ) : Prop := legs = 6 ∨ legs = 8
def lies (legs: ℕ) : Prop := legs = 7

-- Statements made by the octopuses
def blue_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 28
def green_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 27
def yellow_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 26
def red_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 25

noncomputable def legs_b := 7
noncomputable def legs_g := 6
noncomputable def legs_y := 7
noncomputable def legs_r := 7

-- Main theorem
theorem octopus_legs : 
  (tells_truth legs_g) ∧ 
  (lies legs_b) ∧ 
  (lies legs_y) ∧ 
  (lies legs_r) ∧ 
  blue_statement legs_b legs_g legs_y legs_r ∧ 
  green_statement legs_b legs_g legs_y legs_r ∧ 
  yellow_statement legs_b legs_g legs_y legs_r ∧ 
  red_statement legs_b legs_g legs_y legs_r := 
by 
  sorry

end NUMINAMATH_GPT_octopus_legs_l470_47058


namespace NUMINAMATH_GPT_f_zero_f_increasing_on_negative_l470_47019

noncomputable def f : ℝ → ℝ := sorry
variable {x : ℝ}

-- Assume f is an odd function
axiom odd_f : ∀ x, f (-x) = -f x

-- Assume f is increasing on (0, +∞)
axiom increasing_f_on_positive :
  ∀ ⦃x₁ x₂⦄, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂

-- Prove that f(0) = 0
theorem f_zero : f 0 = 0 := sorry

-- Prove that f is increasing on (-∞, 0)
theorem f_increasing_on_negative :
  ∀ ⦃x₁ x₂⦄, x₁ < x₂ → x₂ < 0 → f x₁ < f x₂ := sorry

end NUMINAMATH_GPT_f_zero_f_increasing_on_negative_l470_47019


namespace NUMINAMATH_GPT_solve_quadratic_eqn_l470_47089

theorem solve_quadratic_eqn:
  (∃ x: ℝ, (x + 10)^2 = (4 * x + 6) * (x + 8)) ↔ 
  (∀ x: ℝ, x = 2.131 ∨ x = -8.131) := 
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eqn_l470_47089


namespace NUMINAMATH_GPT_matrix_no_solution_neg_two_l470_47007

-- Define the matrix and vector equation
def matrix_equation (a x y : ℝ) : Prop :=
  (a * x + 2 * y = a + 2) ∧ (2 * x + a * y = 2 * a)

-- Define the condition for no solution
def no_solution_condition (a : ℝ) : Prop :=
  (a/2 = 2/a) ∧ (a/2 ≠ (a + 2) / (2 * a))

-- Theorem stating that a = -2 is the necessary condition for no solution
theorem matrix_no_solution_neg_two (a : ℝ) : no_solution_condition a → a = -2 := by
  sorry

end NUMINAMATH_GPT_matrix_no_solution_neg_two_l470_47007


namespace NUMINAMATH_GPT_probability_non_defective_second_draw_l470_47009

theorem probability_non_defective_second_draw 
  (total_products : ℕ)
  (defective_products : ℕ)
  (first_draw_defective : Bool)
  (second_draw_non_defective_probability : ℚ) : 
  total_products = 100 → 
  defective_products = 3 → 
  first_draw_defective = true → 
  second_draw_non_defective_probability = 97 / 99 :=
by
  intros h_total h_defective h_first_draw
  subst h_total
  subst h_defective
  subst h_first_draw
  sorry

end NUMINAMATH_GPT_probability_non_defective_second_draw_l470_47009


namespace NUMINAMATH_GPT_upper_limit_of_x_l470_47030

theorem upper_limit_of_x :
  ∀ x : ℤ, (0 < x ∧ x < 7) ∧ (0 < x ∧ x < some_upper_limit) ∧ (5 > x ∧ x > -1) ∧ (3 > x ∧ x > 0) ∧ (x + 2 < 4) →
  some_upper_limit = 2 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_upper_limit_of_x_l470_47030


namespace NUMINAMATH_GPT_yellow_marbles_problem_l470_47016

variable (Y B R : ℕ)

theorem yellow_marbles_problem
  (h1 : Y + B + R = 19)
  (h2 : B = (3 * R) / 4)
  (h3 : R = Y + 3) :
  Y = 5 :=
by
  sorry

end NUMINAMATH_GPT_yellow_marbles_problem_l470_47016


namespace NUMINAMATH_GPT_part_one_costs_part_two_feasible_values_part_three_min_cost_l470_47078

noncomputable def cost_of_stationery (a b : ℕ) (cost_A_and_B₁ : 2 * a + b = 35) (cost_A_and_B₂ : a + 3 * b = 30): ℕ × ℕ :=
(a, b)

theorem part_one_costs (a b : ℕ) (h₁ : 2 * a + b = 35) (h₂ : a + 3 * b = 30): cost_of_stationery a b h₁ h₂ = (15, 5) :=
sorry

theorem part_two_feasible_values (x : ℕ) (h₁ : x + (120 - x) = 120) (h₂ : 975 ≤ 15 * x + 5 * (120 - x)) (h₃ : 15 * x + 5 * (120 - x) ≤ 1000):
  x = 38 ∨ x = 39 ∨ x = 40 :=
sorry

theorem part_three_min_cost (x : ℕ) (h₁ : x = 38 ∨ x = 39 ∨ x = 40):
  ∃ min_cost, (min_cost = 10 * 38 + 600 ∧ min_cost ≤ 10 * x + 600) :=
sorry

end NUMINAMATH_GPT_part_one_costs_part_two_feasible_values_part_three_min_cost_l470_47078


namespace NUMINAMATH_GPT_general_term_formula_minimum_sum_value_l470_47021

variable {a : ℕ → ℚ} -- The arithmetic sequence
variable {S : ℕ → ℚ} -- Sum of the first n terms of the sequence

-- Conditions
axiom a_seq_cond1 : a 2 + a 6 = 6
axiom S_sum_cond5 : S 5 = 35 / 3

-- Definitions
def a_n (n : ℕ) : ℚ := (2 / 3) * n + 1 / 3
def S_n (n : ℕ) : ℚ := (1 / 3) * (n^2 + 2 * n)

-- Hypotheses
axiom seq_def : ∀ n, a n = a_n n
axiom sum_def : ∀ n, S n = S_n n

-- Theorems to be proved
theorem general_term_formula : ∀ n, a n = (2 / 3 * n) + 1 / 3 := by sorry
theorem minimum_sum_value : ∀ n, S 1 ≤ S n := by sorry

end NUMINAMATH_GPT_general_term_formula_minimum_sum_value_l470_47021


namespace NUMINAMATH_GPT_candy_proof_l470_47000

variable (x s t : ℤ)

theorem candy_proof (H1 : 4 * x - 15 * s = 23)
                    (H2 : 5 * x - 23 * t = 15) :
  x = 302 := by
  sorry

end NUMINAMATH_GPT_candy_proof_l470_47000


namespace NUMINAMATH_GPT_problem_solution_l470_47056

def seq (a : ℕ → ℝ) (a1 : a 1 = 0) (rec : ∀ n, a (n + 1) = (a n - Real.sqrt 3) / (1 + Real.sqrt 3 * a n)) : Prop :=
  a 6 = Real.sqrt 3

theorem problem_solution (a : ℕ → ℝ) (h1 : a 1 = 0) (hrec : ∀ n, a (n + 1) = (a n - Real.sqrt 3) / (1 + Real.sqrt 3 * a n)) : 
  seq a h1 hrec :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l470_47056


namespace NUMINAMATH_GPT_heptagon_angle_in_arithmetic_progression_l470_47098

theorem heptagon_angle_in_arithmetic_progression (a d : ℝ) :
  a + 3 * d = 128.57 → 
  (7 * a + 21 * d = 900) → 
  ∃ angle : ℝ, angle = 128.57 :=
by
  sorry

end NUMINAMATH_GPT_heptagon_angle_in_arithmetic_progression_l470_47098


namespace NUMINAMATH_GPT_al_sandwiches_correct_l470_47071

-- Definitions based on the given conditions
def num_breads := 5
def num_meats := 7
def num_cheeses := 6
def total_combinations := num_breads * num_meats * num_cheeses

def turkey_swiss := num_breads -- disallowed turkey/Swiss cheese combinations
def multigrain_turkey := num_cheeses -- disallowed multi-grain bread/turkey combinations

def al_sandwiches := total_combinations - turkey_swiss - multigrain_turkey

-- The theorem to prove
theorem al_sandwiches_correct : al_sandwiches = 199 := 
by sorry

end NUMINAMATH_GPT_al_sandwiches_correct_l470_47071


namespace NUMINAMATH_GPT_incorrect_statements_count_l470_47097

-- Definitions of the statements
def statement1 : Prop := "The diameter perpendicular to the chord bisects the chord" = "incorrect"

def statement2 : Prop := "A circle is a symmetrical figure, and any diameter is its axis of symmetry" = "incorrect"

def statement3 : Prop := "Two arcs of equal length are congruent" = "incorrect"

-- Theorem stating that the number of incorrect statements is 3
theorem incorrect_statements_count : 
  (statement1 → False) → (statement2 → False) → (statement3 → False) → 3 = 3 :=
by sorry

end NUMINAMATH_GPT_incorrect_statements_count_l470_47097


namespace NUMINAMATH_GPT_total_number_recruits_l470_47060

theorem total_number_recruits 
  (x y z : ℕ)
  (h1 : x = 50)
  (h2 : y = 100)
  (h3 : z = 170)
  (h4 : x = 4 * (y - 50) ∨ y = 4 * (z - 170) ∨ x = 4 * (z - 170)) : 
  171 + (z - 170) = 211 :=
by
  sorry

end NUMINAMATH_GPT_total_number_recruits_l470_47060


namespace NUMINAMATH_GPT_mod_11_residue_l470_47073

theorem mod_11_residue : 
  ((312 - 3 * 52 + 9 * 165 + 6 * 22) % 11) = 2 :=
by
  sorry

end NUMINAMATH_GPT_mod_11_residue_l470_47073


namespace NUMINAMATH_GPT_series_sum_eq_one_sixth_l470_47069

noncomputable def series_sum := 
  ∑' n : ℕ, (3^n) / ((7^ (2^n)) + 1)

theorem series_sum_eq_one_sixth : series_sum = 1 / 6 := 
  sorry

end NUMINAMATH_GPT_series_sum_eq_one_sixth_l470_47069


namespace NUMINAMATH_GPT_option_B_is_one_variable_quadratic_l470_47043

theorem option_B_is_one_variable_quadratic :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x : ℝ, 2 * (x - x^2) - 1 = a * x^2 + b * x + c) :=
by
  sorry

end NUMINAMATH_GPT_option_B_is_one_variable_quadratic_l470_47043


namespace NUMINAMATH_GPT_cannot_form_set_of_good_friends_of_wang_ming_l470_47004

def is_well_defined_set (description : String) : Prop := sorry  -- Placeholder for the formal definition.

theorem cannot_form_set_of_good_friends_of_wang_ming :
  ¬ is_well_defined_set "Good friends of Wang Ming" :=
sorry

end NUMINAMATH_GPT_cannot_form_set_of_good_friends_of_wang_ming_l470_47004


namespace NUMINAMATH_GPT_range_of_m_l470_47080

noncomputable def f (m x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
noncomputable def g (m x : ℝ) : ℝ := m * x

theorem range_of_m :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) → 0 < m ∧ m < 8 :=
sorry

end NUMINAMATH_GPT_range_of_m_l470_47080


namespace NUMINAMATH_GPT_uncovered_side_length_l470_47039

theorem uncovered_side_length {L W : ℕ} (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end NUMINAMATH_GPT_uncovered_side_length_l470_47039


namespace NUMINAMATH_GPT_cycling_route_length_l470_47038

-- Conditions (segment lengths)
def segment1 : ℝ := 4
def segment2 : ℝ := 7
def segment3 : ℝ := 2
def segment4 : ℝ := 6
def segment5 : ℝ := 7

-- Specify the total length calculation
noncomputable def total_length : ℝ :=
  2 * (segment1 + segment2 + segment3) + 2 * (segment4 + segment5)

-- The theorem we want to prove
theorem cycling_route_length :
  total_length = 52 :=
by
  sorry

end NUMINAMATH_GPT_cycling_route_length_l470_47038


namespace NUMINAMATH_GPT_jaylen_charge_per_yard_l470_47095

def total_cost : ℝ := 250
def number_of_yards : ℝ := 6
def charge_per_yard : ℝ := 41.67

theorem jaylen_charge_per_yard :
  total_cost / number_of_yards = charge_per_yard :=
sorry

end NUMINAMATH_GPT_jaylen_charge_per_yard_l470_47095


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l470_47024

def setA (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1
def setB (x : ℝ) : Prop := 0 < x ∧ x ≤ 2
def setIntersection (x : ℝ) : Prop := 0 < x ∧ x ≤ 1

theorem intersection_of_A_and_B :
  ∀ x, (setA x ∧ setB x) ↔ setIntersection x := 
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l470_47024


namespace NUMINAMATH_GPT_initial_discount_percentage_l470_47068

-- Statement of the problem
theorem initial_discount_percentage (d : ℝ) (x : ℝ)
  (h₁ : d > 0)
  (h_staff_price : d * ((100 - x) / 100) * 0.5 = 0.225 * d) :
  x = 55 := 
sorry

end NUMINAMATH_GPT_initial_discount_percentage_l470_47068


namespace NUMINAMATH_GPT_least_value_x_y_z_l470_47074

theorem least_value_x_y_z (x y z : ℕ) (hx : x = 4 * y) (hy : y = 7 * z) (hz : 0 < z) : x - y - z = 19 :=
by
  -- placeholder for actual proof
  sorry

end NUMINAMATH_GPT_least_value_x_y_z_l470_47074


namespace NUMINAMATH_GPT_work_days_l470_47079

theorem work_days (A B C : ℝ) (h₁ : A + B = 1 / 15) (h₂ : C = 1 / 7.5) : 1 / (A + B + C) = 5 :=
by
  sorry

end NUMINAMATH_GPT_work_days_l470_47079


namespace NUMINAMATH_GPT_proof_total_distance_l470_47057

-- Define the total distance
def total_distance (D : ℕ) :=
  let by_foot := (1 : ℚ) / 6
  let by_bicycle := (1 : ℚ) / 4
  let by_bus := (1 : ℚ) / 3
  let by_car := 10
  let by_train := (1 : ℚ) / 12
  D - (by_foot + by_bicycle + by_bus + by_train) * D = by_car

-- Given proof problem
theorem proof_total_distance : ∃ D : ℕ, total_distance D ∧ D = 60 :=
sorry

end NUMINAMATH_GPT_proof_total_distance_l470_47057


namespace NUMINAMATH_GPT_simplify_proof_l470_47037

noncomputable def simplify_expression (a b c d x y : ℝ) (h : c * x ≠ d * y) : ℝ :=
  (c * x * (b^2 * x^2 - 4 * b^2 * y^2 + a^2 * y^2) 
  - d * y * (b^2 * x^2 - 2 * a^2 * x^2 - 3 * a^2 * y^2)) / (c * x - d * y)

theorem simplify_proof (a b c d x y : ℝ) (h : c * x ≠ d * y) :
  simplify_expression a b c d x y h = b^2 * x^2 + a^2 * y^2 :=
by sorry

end NUMINAMATH_GPT_simplify_proof_l470_47037


namespace NUMINAMATH_GPT_xy_yz_zx_equal_zero_l470_47001

noncomputable def side1 (x y z : ℝ) : ℝ := 1 / abs (x^2 + 2 * y * z)
noncomputable def side2 (x y z : ℝ) : ℝ := 1 / abs (y^2 + 2 * z * x)
noncomputable def side3 (x y z : ℝ) : ℝ := 1 / abs (z^2 + 2 * x * y)

def non_degenerate_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem xy_yz_zx_equal_zero
  (x y z : ℝ)
  (h1 : non_degenerate_triangle (side1 x y z) (side2 x y z) (side3 x y z)) :
  xy + yz + zx = 0 := sorry

end NUMINAMATH_GPT_xy_yz_zx_equal_zero_l470_47001


namespace NUMINAMATH_GPT_usual_time_to_catch_bus_l470_47062

theorem usual_time_to_catch_bus (S T : ℝ) (h : S / (4 / 5 * S) = (T + 3) / T) : T = 12 :=
by 
  sorry

end NUMINAMATH_GPT_usual_time_to_catch_bus_l470_47062


namespace NUMINAMATH_GPT_length_PQ_is_5_l470_47054

/-
Given:
- Point P with coordinates (3, 4, 5)
- Point Q is the projection of P onto the xOy plane

Show:
- The length of the segment PQ is 5
-/

def P : ℝ × ℝ × ℝ := (3, 4, 5)
def Q : ℝ × ℝ × ℝ := (3, 4, 0)

theorem length_PQ_is_5 : dist P Q = 5 := by
  sorry

end NUMINAMATH_GPT_length_PQ_is_5_l470_47054


namespace NUMINAMATH_GPT_operation_two_three_l470_47086

def operation (a b : ℕ) : ℤ := 4 * a ^ 2 - 4 * b ^ 2

theorem operation_two_three : operation 2 3 = -20 :=
by
  sorry

end NUMINAMATH_GPT_operation_two_three_l470_47086


namespace NUMINAMATH_GPT_daily_production_l470_47033

-- Define the conditions
def bottles_per_case : ℕ := 9
def num_cases : ℕ := 8000

-- State the theorem with the question and the calculated answer
theorem daily_production : bottles_per_case * num_cases = 72000 :=
by
  sorry

end NUMINAMATH_GPT_daily_production_l470_47033


namespace NUMINAMATH_GPT_annual_interest_rate_l470_47083

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

theorem annual_interest_rate :
  compound_interest_rate 150 181.50 2 1 (0.2 : ℝ) :=
by
  unfold compound_interest_rate
  sorry

end NUMINAMATH_GPT_annual_interest_rate_l470_47083


namespace NUMINAMATH_GPT_quadratic_equation_solution_l470_47091

noncomputable def findOrderPair (b d : ℝ) : Prop :=
  (b + d = 7) ∧ (b < d) ∧ (36 - 4 * b * d = 0)

theorem quadratic_equation_solution :
  ∃ b d : ℝ, findOrderPair b d ∧ (b, d) = ( (7 - Real.sqrt 13) / 2, (7 + Real.sqrt 13) / 2 ) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_solution_l470_47091


namespace NUMINAMATH_GPT_initial_concentration_l470_47053

theorem initial_concentration (C : ℝ) 
  (hC : (C * 0.2222222222222221) + (0.25 * 0.7777777777777779) = 0.35) :
  C = 0.7 :=
sorry

end NUMINAMATH_GPT_initial_concentration_l470_47053


namespace NUMINAMATH_GPT_square_perimeter_ratio_l470_47034

theorem square_perimeter_ratio (a₁ a₂ s₁ s₂ : ℝ) 
  (h₁ : a₁ / a₂ = 16 / 25)
  (h₂ : a₁ = s₁^2)
  (h₃ : a₂ = s₂^2) :
  (4 : ℝ) / 5 = s₁ / s₂ :=
by sorry

end NUMINAMATH_GPT_square_perimeter_ratio_l470_47034


namespace NUMINAMATH_GPT_flour_masses_l470_47040

theorem flour_masses (x : ℝ) (h: 
    (x * (1 + x / 100) + (x + 10) * (1 + (x + 10) / 100) = 112.5)) :
    x = 35 ∧ (x + 10) = 45 :=
by 
  sorry

end NUMINAMATH_GPT_flour_masses_l470_47040


namespace NUMINAMATH_GPT_hiker_speed_calculation_l470_47093

theorem hiker_speed_calculation :
  ∃ (h_speed : ℝ),
    let c_speed := 10
    let c_time := 5.0 / 60.0
    let c_wait := 7.5 / 60.0
    let c_distance := c_speed * c_time
    let h_distance := c_distance
    h_distance = h_speed * c_wait ∧ h_speed = 10 * (5 / 7.5) := by
  sorry

end NUMINAMATH_GPT_hiker_speed_calculation_l470_47093


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l470_47055

theorem problem_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  a + b + c ≥ 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c := 
sorry

theorem problem_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt a + Real.sqrt b + Real.sqrt c :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l470_47055


namespace NUMINAMATH_GPT_problem_is_happy_number_512_l470_47044

/-- A number is a "happy number" if it is the square difference of two consecutive odd numbers. -/
def is_happy_number (x : ℕ) : Prop :=
  ∃ n : ℤ, x = 8 * n

/-- The number 512 is a "happy number". -/
theorem problem_is_happy_number_512 : is_happy_number 512 :=
  sorry

end NUMINAMATH_GPT_problem_is_happy_number_512_l470_47044


namespace NUMINAMATH_GPT_triangle_product_l470_47082

theorem triangle_product (a b c: ℕ) (p: ℕ)
    (h1: ∃ k1 k2 k3: ℕ, a * k1 * k2 = p ∧ k2 * k3 * b = p ∧ k3 * c * a = p) 
    : (1 ≤ c ∧ c ≤ 336) :=
by
  sorry

end NUMINAMATH_GPT_triangle_product_l470_47082


namespace NUMINAMATH_GPT_contemporaries_probability_l470_47048

open Real

noncomputable def probability_of_contemporaries
  (born_within : ℝ) (lifespan : ℝ) : ℝ :=
  let total_area := born_within * born_within
  let side := born_within - lifespan
  let non_overlap_area := 2 * (1/2 * side * side)
  let overlap_area := total_area - non_overlap_area
  overlap_area / total_area

theorem contemporaries_probability :
  probability_of_contemporaries 300 80 = 104 / 225 := 
by
  sorry

end NUMINAMATH_GPT_contemporaries_probability_l470_47048


namespace NUMINAMATH_GPT_solution_set_of_inequality_l470_47047

theorem solution_set_of_inequality (x : ℝ) : x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l470_47047


namespace NUMINAMATH_GPT_number_of_seats_in_classroom_l470_47077

theorem number_of_seats_in_classroom 
    (seats_per_row_condition : 7 + 13 = 19) 
    (rows_condition : 8 + 14 = 21) : 
    19 * 21 = 399 := 
by 
    sorry

end NUMINAMATH_GPT_number_of_seats_in_classroom_l470_47077


namespace NUMINAMATH_GPT_minimize_sum_of_reciprocals_l470_47026

def dataset : List ℝ := [2, 4, 6, 8]

def mean : ℝ := 5
def variance: ℝ := 5

theorem minimize_sum_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : mean * a + variance * b = 1) : 
  (1 / a + 1 / b) = 20 :=
sorry

end NUMINAMATH_GPT_minimize_sum_of_reciprocals_l470_47026


namespace NUMINAMATH_GPT_math_competition_l470_47070

theorem math_competition (a b c d e f g : ℕ) (h1 : a + b + c + d + e + f + g = 25)
    (h2 : b = 2 * c + f) (h3 : a = d + e + g + 1) (h4 : a = b + c) :
    b = 6 :=
by
  -- The proof is omitted as the problem requests the statement only.
  sorry

end NUMINAMATH_GPT_math_competition_l470_47070


namespace NUMINAMATH_GPT_max_unbounded_xy_sum_l470_47018

theorem max_unbounded_xy_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ∃ M : ℝ, ∀ z : ℝ, z > 0 → ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ (xy + 1)^2 + (x - y)^2 > z := 
  sorry

end NUMINAMATH_GPT_max_unbounded_xy_sum_l470_47018


namespace NUMINAMATH_GPT_tangent_line_right_triangle_l470_47064

theorem tangent_line_right_triangle {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (tangent_condition : a^2 + b^2 = c^2) : 
  (abs c)^2 = (abs a)^2 + (abs b)^2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_right_triangle_l470_47064


namespace NUMINAMATH_GPT_total_bills_is_126_l470_47002

noncomputable def F : ℕ := 84  -- number of 5-dollar bills
noncomputable def T : ℕ := (840 - 5 * F) / 10  -- derive T based on the total value and F
noncomputable def total_bills : ℕ := F + T

theorem total_bills_is_126 : total_bills = 126 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_bills_is_126_l470_47002


namespace NUMINAMATH_GPT_domain_of_sqrt_function_l470_47036

theorem domain_of_sqrt_function (x : ℝ) :
  (x + 4 ≥ 0) ∧ (1 - x ≥ 0) ∧ (x ≠ 0) ↔ (-4 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1) := 
sorry

end NUMINAMATH_GPT_domain_of_sqrt_function_l470_47036


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l470_47022

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l470_47022


namespace NUMINAMATH_GPT_set_equality_l470_47081

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {1, 4})
variable (hB : B = {3, 4})

theorem set_equality : ({2, 5} : Set ℕ) = U \ (A ∪ B) :=
by
  sorry

end NUMINAMATH_GPT_set_equality_l470_47081


namespace NUMINAMATH_GPT_train_distance_after_braking_l470_47096

theorem train_distance_after_braking : 
  (∃ t : ℝ, (27 * t - 0.45 * t^2 = 0) ∧ (∀ s : ℝ, s = 27 * t - 0.45 * t^2) ∧ s = 405) :=
sorry

end NUMINAMATH_GPT_train_distance_after_braking_l470_47096


namespace NUMINAMATH_GPT_find_largest_beta_l470_47085

theorem find_largest_beta (α : ℝ) (r : ℕ → ℝ) (C : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < 1)
  (h3 : ∀ n, ∀ m ≠ n, dist (r n) (r m) ≥ (r n) ^ α)
  (h4 : ∀ n, r n ≤ r (n + 1)) 
  (h5 : ∀ n, r n ≥ C * n ^ (1 / (2 * (1 - α)))) :
  ∀ β, (∃ C > 0, ∀ n, r n ≥ C * n ^ β) → β ≤ 1 / (2 * (1 - α)) :=
sorry

end NUMINAMATH_GPT_find_largest_beta_l470_47085


namespace NUMINAMATH_GPT_distance_between_intersections_l470_47008

theorem distance_between_intersections :
  let a := 3
  let b := 2
  let c := -7
  let x1 := (-1 + Real.sqrt 22) / 3
  let x2 := (-1 - Real.sqrt 22) / 3
  let distance := abs (x1 - x2)
  let p := 88  -- 2^2 * 22 = 88
  let q := 9   -- 3^2 = 9
  distance = 2 * Real.sqrt 22 / 3 →
  p - q = 79 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_intersections_l470_47008


namespace NUMINAMATH_GPT_linear_eq_substitution_l470_47010

theorem linear_eq_substitution (x y : ℝ) (h1 : 3 * x - 4 * y = 2) (h2 : x = 2 * y - 1) :
  3 * (2 * y - 1) - 4 * y = 2 :=
by
  sorry

end NUMINAMATH_GPT_linear_eq_substitution_l470_47010


namespace NUMINAMATH_GPT_pradeep_failed_marks_l470_47051

theorem pradeep_failed_marks
    (total_marks : ℕ)
    (obtained_marks : ℕ)
    (pass_percentage : ℕ)
    (pass_marks : ℕ)
    (fail_marks : ℕ)
    (total_marks_eq : total_marks = 2075)
    (obtained_marks_eq : obtained_marks = 390)
    (pass_percentage_eq : pass_percentage = 20)
    (pass_marks_eq : pass_marks = (pass_percentage * total_marks) / 100)
    (fail_marks_eq : fail_marks = pass_marks - obtained_marks) :
    fail_marks = 25 :=
by
  rw [total_marks_eq, obtained_marks_eq, pass_percentage_eq] at *
  sorry

end NUMINAMATH_GPT_pradeep_failed_marks_l470_47051
