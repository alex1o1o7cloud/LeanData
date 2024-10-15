import Mathlib

namespace NUMINAMATH_GPT_Roselyn_initial_books_correct_l318_31882

variables (Roselyn_initial_books Mara_books Rebecca_books : ℕ)

-- Conditions
axiom A1 : Rebecca_books = 40
axiom A2 : Mara_books = 3 * Rebecca_books
axiom A3 : Roselyn_initial_books - (Rebecca_books + Mara_books) = 60

-- Proof statement
theorem Roselyn_initial_books_correct : Roselyn_initial_books = 220 :=
sorry

end NUMINAMATH_GPT_Roselyn_initial_books_correct_l318_31882


namespace NUMINAMATH_GPT_number_of_gigs_played_l318_31842

/-- Given earnings per gig for each band member and the total earnings, prove the total number of gigs played -/

def lead_singer_earnings : ℕ := 30
def guitarist_earnings : ℕ := 25
def bassist_earnings : ℕ := 20
def drummer_earnings : ℕ := 25
def keyboardist_earnings : ℕ := 20
def backup_singer1_earnings : ℕ := 15
def backup_singer2_earnings : ℕ := 18
def backup_singer3_earnings : ℕ := 12
def total_earnings : ℕ := 3465

def total_earnings_per_gig : ℕ :=
  lead_singer_earnings +
  guitarist_earnings +
  bassist_earnings +
  drummer_earnings +
  keyboardist_earnings +
  backup_singer1_earnings +
  backup_singer2_earnings +
  backup_singer3_earnings

theorem number_of_gigs_played : (total_earnings / total_earnings_per_gig) = 21 := by
  sorry

end NUMINAMATH_GPT_number_of_gigs_played_l318_31842


namespace NUMINAMATH_GPT_filled_sacks_count_l318_31864

-- Definitions from the problem conditions
def pieces_per_sack := 20
def total_pieces := 80

theorem filled_sacks_count : total_pieces / pieces_per_sack = 4 := 
by sorry

end NUMINAMATH_GPT_filled_sacks_count_l318_31864


namespace NUMINAMATH_GPT_total_goals_l318_31801

-- Definitions
def louie_goals_last_match := 4
def louie_previous_goals := 40
def brother_multiplier := 2
def seasons := 3
def games_per_season := 50

-- Total number of goals scored by Louie and his brother
theorem total_goals : (louie_previous_goals + louie_goals_last_match) 
                      + (brother_multiplier * louie_goals_last_match * seasons * games_per_season) 
                      = 1244 :=
by sorry

end NUMINAMATH_GPT_total_goals_l318_31801


namespace NUMINAMATH_GPT_train_pass_jogger_in_36_sec_l318_31880

noncomputable def time_to_pass_jogger (speed_jogger speed_train : ℝ) (lead_jogger len_train : ℝ) : ℝ :=
  let speed_jogger_mps := speed_jogger * (1000 / 3600)
  let speed_train_mps := speed_train * (1000 / 3600)
  let relative_speed := speed_train_mps - speed_jogger_mps
  let total_distance := lead_jogger + len_train
  total_distance / relative_speed

theorem train_pass_jogger_in_36_sec :
  time_to_pass_jogger 9 45 240 120 = 36 := by
  sorry

end NUMINAMATH_GPT_train_pass_jogger_in_36_sec_l318_31880


namespace NUMINAMATH_GPT_dvd_blu_ratio_l318_31853

theorem dvd_blu_ratio (D B : ℕ) (h1 : D + B = 378) (h2 : (D : ℚ) / (B - 4 : ℚ) = 9 / 2) :
  D / Nat.gcd D B = 51 ∧ B / Nat.gcd D B = 12 :=
by
  sorry

end NUMINAMATH_GPT_dvd_blu_ratio_l318_31853


namespace NUMINAMATH_GPT_fourth_power_sum_l318_31897

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a^2 + b^2 + c^2 = 3) 
  (h3 : a^3 + b^3 + c^3 = 6) : 
  a^4 + b^4 + c^4 = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_fourth_power_sum_l318_31897


namespace NUMINAMATH_GPT_john_more_needed_l318_31845

def john_needs : ℝ := 2.5
def john_has : ℝ := 0.75
def john_needs_more : ℝ := 1.75

theorem john_more_needed : (john_needs - john_has) = john_needs_more :=
by
  sorry

end NUMINAMATH_GPT_john_more_needed_l318_31845


namespace NUMINAMATH_GPT_mikes_remaining_cards_l318_31849

variable (original_number_of_cards : ℕ)
variable (sam_bought : ℤ)
variable (alex_bought : ℤ)

theorem mikes_remaining_cards :
  original_number_of_cards = 87 →
  sam_bought = 8 →
  alex_bought = 13 →
  original_number_of_cards - (sam_bought + alex_bought) = 66 :=
by
  intros h_original h_sam h_alex
  rw [h_original, h_sam, h_alex]
  norm_num

end NUMINAMATH_GPT_mikes_remaining_cards_l318_31849


namespace NUMINAMATH_GPT_applicant_overall_score_l318_31815

-- Definitions for the conditions
def writtenTestScore : ℝ := 80
def interviewScore : ℝ := 60
def weightWrittenTest : ℝ := 0.6
def weightInterview : ℝ := 0.4

-- Theorem statement
theorem applicant_overall_score : 
  (writtenTestScore * weightWrittenTest) + (interviewScore * weightInterview) = 72 := 
by
  sorry

end NUMINAMATH_GPT_applicant_overall_score_l318_31815


namespace NUMINAMATH_GPT_volume_relation_l318_31848

noncomputable def A (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3
noncomputable def M (r : ℝ) : ℝ := 2 * Real.pi * r^3
noncomputable def C (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_relation (r : ℝ) : A r - M r + C r = 0 :=
by
  sorry

end NUMINAMATH_GPT_volume_relation_l318_31848


namespace NUMINAMATH_GPT_eight_x_plus_y_l318_31891

theorem eight_x_plus_y (x y z : ℝ) (h1 : x + 2 * y - 3 * z = 7) (h2 : 2 * x - y + 2 * z = 6) : 
  8 * x + y = 32 :=
sorry

end NUMINAMATH_GPT_eight_x_plus_y_l318_31891


namespace NUMINAMATH_GPT_increase_percentage_when_selfcheckout_broken_l318_31819

-- The problem conditions as variable definitions and declarations
def normal_complaints : ℕ := 120
def short_staffed_increase : ℚ := 1 / 3
def short_staffed_complaints : ℕ := normal_complaints + (normal_complaints / 3)
def total_complaints_three_days : ℕ := 576
def days : ℕ := 3
def both_conditions_complaints : ℕ := total_complaints_three_days / days

-- The theorem that we need to prove
theorem increase_percentage_when_selfcheckout_broken : 
  (both_conditions_complaints - short_staffed_complaints) * 100 / short_staffed_complaints = 20 := 
by
  -- This line sets up that the conclusion is true
  sorry

end NUMINAMATH_GPT_increase_percentage_when_selfcheckout_broken_l318_31819


namespace NUMINAMATH_GPT_more_girls_than_boys_l318_31843

def ratio_boys_girls (B G : ℕ) : Prop := B = (3/5 : ℚ) * G

def total_students (B G : ℕ) : Prop := B + G = 16

theorem more_girls_than_boys (B G : ℕ) (h1 : ratio_boys_girls B G) (h2 : total_students B G) : G - B = 4 :=
by
  sorry

end NUMINAMATH_GPT_more_girls_than_boys_l318_31843


namespace NUMINAMATH_GPT_sqrt_expression_simplification_l318_31835

theorem sqrt_expression_simplification : 
  (Real.sqrt 48 - Real.sqrt 2 * Real.sqrt 6 - Real.sqrt 15 / Real.sqrt 5) = Real.sqrt 3 := 
  by
    sorry

end NUMINAMATH_GPT_sqrt_expression_simplification_l318_31835


namespace NUMINAMATH_GPT_simplify_expression_l318_31863

-- Define general term for y
variable (y : ℤ)

-- Statement representing the given proof problem
theorem simplify_expression :
  4 * y + 5 * y + 6 * y + 2 = 15 * y + 2 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l318_31863


namespace NUMINAMATH_GPT_total_students_in_school_l318_31866

variable (TotalStudents : ℕ)
variable (num_students_8_years_old : ℕ := 48)
variable (percent_students_below_8 : ℝ := 0.20)
variable (num_students_above_8 : ℕ := (2 / 3) * num_students_8_years_old)

theorem total_students_in_school :
  percent_students_below_8 * TotalStudents + (num_students_8_years_old + num_students_above_8) = TotalStudents :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_school_l318_31866


namespace NUMINAMATH_GPT_other_divisor_l318_31885

theorem other_divisor (x : ℕ) (h₁ : 261 % 7 = 2) (h₂ : 261 % x = 2) : x = 259 :=
sorry

end NUMINAMATH_GPT_other_divisor_l318_31885


namespace NUMINAMATH_GPT_find_other_intersection_point_l318_31876

-- Definitions
def parabola_eq (x : ℝ) : ℝ := x^2 - 2 * x - 3
def intersection_point1 : Prop := parabola_eq (-1) = 0
def intersection_point2 : Prop := parabola_eq 3 = 0

-- Proof problem
theorem find_other_intersection_point :
  intersection_point1 → intersection_point2 := by
  sorry

end NUMINAMATH_GPT_find_other_intersection_point_l318_31876


namespace NUMINAMATH_GPT_analytic_expression_of_f_range_of_k_l318_31837

noncomputable def quadratic_function_minimum (a b : ℝ) : ℝ :=
a * (-1) ^ 2 + b * (-1) + 1

theorem analytic_expression_of_f (a b : ℝ) (ha : quadratic_function_minimum a b = 0)
  (hmin: -1 = -b / (2 * a)) : a = 1 ∧ b = 2 :=
by sorry

theorem range_of_k (k : ℝ) : ∃ k : ℝ, (k ∈ Set.Ici 3 ∨ k = 13 / 4) :=
by sorry

end NUMINAMATH_GPT_analytic_expression_of_f_range_of_k_l318_31837


namespace NUMINAMATH_GPT_max_value_l318_31809

variable (x y : ℝ)

def condition : Prop := 2 * x ^ 2 + x * y - y ^ 2 = 1

noncomputable def expression : ℝ := (x - 2 * y) / (5 * x ^ 2 - 2 * x * y + 2 * y ^ 2)

theorem max_value : ∀ x y : ℝ, condition x y → expression x y ≤ (Real.sqrt 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_value_l318_31809


namespace NUMINAMATH_GPT_at_most_n_pairs_with_distance_d_l318_31856

theorem at_most_n_pairs_with_distance_d
  (n : ℕ) (hn : n ≥ 3)
  (points : Fin n → ℝ × ℝ)
  (d : ℝ)
  (hd : ∀ i j, i ≠ j → dist (points i) (points j) ≤ d)
  (dmax : ∃ i j, i ≠ j ∧ dist (points i) (points j) = d) :
  ∃ (pairs : Finset (Fin n × Fin n)), ∀ p ∈ pairs, dist (points p.1) (points p.2) = d ∧ pairs.card ≤ n := 
sorry

end NUMINAMATH_GPT_at_most_n_pairs_with_distance_d_l318_31856


namespace NUMINAMATH_GPT_max_value_5x_minus_25x_l318_31800

noncomputable def max_value_of_expression : ℝ :=
  (1 / 4 : ℝ)

theorem max_value_5x_minus_25x :
  ∃ x : ℝ, ∀ y : ℝ, y = 5^x → (5^y - 25^y) ≤ max_value_of_expression :=
sorry

end NUMINAMATH_GPT_max_value_5x_minus_25x_l318_31800


namespace NUMINAMATH_GPT_stream_speed_l318_31889

theorem stream_speed (u v : ℝ) (h1 : 27 = 9 * (u - v)) (h2 : 81 = 9 * (u + v)) : v = 3 :=
by
  sorry

end NUMINAMATH_GPT_stream_speed_l318_31889


namespace NUMINAMATH_GPT_side_length_of_square_l318_31875

-- Define the conditions
def area_rectangle (length width : ℝ) : ℝ := length * width
def area_square (side : ℝ) : ℝ := side * side

-- Given conditions
def rect_length : ℝ := 2
def rect_width : ℝ := 8
def area_of_rectangle : ℝ := area_rectangle rect_length rect_width
def area_of_square : ℝ := area_of_rectangle

-- Main statement to prove
theorem side_length_of_square : ∃ (s : ℝ), s^2 = 16 ∧ s = 4 :=
by {
  -- use the conditions here
  sorry
}

end NUMINAMATH_GPT_side_length_of_square_l318_31875


namespace NUMINAMATH_GPT_min_value_expression_l318_31817

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 5) :
  ∃ (min_val : ℝ), min_val = ( (x + 1) * (2 * y + 1) ) / (Real.sqrt (x * y)) ∧ min_val = 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l318_31817


namespace NUMINAMATH_GPT_positive_difference_between_median_and_mode_l318_31894

-- Definition of the data as provided in the stem and leaf plot
def data : List ℕ := [
  21, 21, 21, 24, 25, 25,
  33, 33, 36, 37,
  40, 43, 44, 47, 49, 49,
  52, 56, 56, 58, 
  59, 59, 60, 63
]

-- Definition of mode and median calculations
def mode (l : List ℕ) : ℕ := 49  -- As determined, 49 is the mode
def median (l : List ℕ) : ℚ := (43 + 44) / 2  -- Median determined from the sorted list

-- The main theorem to prove
theorem positive_difference_between_median_and_mode (l : List ℕ) :
  abs (median l - mode l) = 5.5 := by
  sorry

end NUMINAMATH_GPT_positive_difference_between_median_and_mode_l318_31894


namespace NUMINAMATH_GPT_can_combine_with_sqrt2_l318_31852

theorem can_combine_with_sqrt2 :
  (∃ (x : ℝ), x = 2 * Real.sqrt 6 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 2 * Real.sqrt 3 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 2 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 3 * Real.sqrt 2 ∧ ∃ (y : ℝ), y = Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_can_combine_with_sqrt2_l318_31852


namespace NUMINAMATH_GPT_union_sets_l318_31838

-- Define the sets A and B based on their conditions
def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 < x ∧ x < 9 }

-- Statement of the proof problem
theorem union_sets (x : ℝ) : (x ∈ A ∪ B) ↔ (x ∈ { x | -1 ≤ x ∧ x < 9 }) := sorry

end NUMINAMATH_GPT_union_sets_l318_31838


namespace NUMINAMATH_GPT_sum_of_roots_eq_9_div_4_l318_31861

-- Define the values for the coefficients
def a : ℝ := -48
def b : ℝ := 108
def c : ℝ := -27

-- Define the quadratic equation and the function that represents the sum of the roots
def quadratic_eq (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Statement of the problem: Prove the sum of the roots of the quadratic equation equals 9/4
theorem sum_of_roots_eq_9_div_4 : 
  (∀ x y : ℝ, quadratic_eq x = 0 → quadratic_eq y = 0 → x ≠ y → x + y = - (b/a)) → - (b / a) = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_eq_9_div_4_l318_31861


namespace NUMINAMATH_GPT_minimum_value_of_objective_function_l318_31808

theorem minimum_value_of_objective_function :
  ∃ (x y : ℝ), x - y + 2 ≥ 0 ∧ 2 * x + 3 * y - 6 ≥ 0 ∧ 3 * x + 2 * y - 9 ≤ 0 ∧ (∀ (x' y' : ℝ), x' - y' + 2 ≥ 0 ∧ 2 * x' + 3 * y' - 6 ≥ 0 ∧ 3 * x' + 2 * y' - 9 ≤ 0 → 2 * x + 5 * y ≤ 2 * x' + 5 * y') ∧ 2 * x + 5 * y = 6 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_objective_function_l318_31808


namespace NUMINAMATH_GPT_unrealistic_data_l318_31813

theorem unrealistic_data :
  let A := 1000
  let A1 := 265
  let A2 := 51
  let A3 := 803
  let A1U2 := 287
  let A2U3 := 843
  let A1U3 := 919
  let A1I2 := A1 + A2 - A1U2
  let A2I3 := A2 + A3 - A2U3
  let A3I1 := A3 + A1 - A1U3
  let U := A1 + A2 + A3 - A1I2 - A2I3 - A3I1
  let A1I2I3 := A - U
  A1I2I3 > A2 :=
by
   sorry

end NUMINAMATH_GPT_unrealistic_data_l318_31813


namespace NUMINAMATH_GPT_find_R_plus_S_l318_31820

theorem find_R_plus_S (d e R S : ℝ) 
  (h1 : d + 3 = 0)
  (h2 : 7 * d + 3 * e = 0)
  (h3 : R = 3 * d + e + 7)
  (h4 : S = 7 * e) :
  R + S = 54 :=
by
  sorry

end NUMINAMATH_GPT_find_R_plus_S_l318_31820


namespace NUMINAMATH_GPT_wendy_adds_18_gallons_l318_31804

-- Definitions based on the problem
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_tank_fraction_full : ℚ := 1 / 2
def car_tank_fraction_full : ℚ := 1 / 3

-- Conditions on the amount of gallons currently in the tanks
def truck_current_gallons : ℚ := truck_tank_capacity * truck_tank_fraction_full
def car_current_gallons : ℚ := car_tank_capacity * car_tank_fraction_full

-- Amount of gallons needed to fill up each tank
def truck_gallons_to_add : ℚ := truck_tank_capacity - truck_current_gallons
def car_gallons_to_add : ℚ := car_tank_capacity - car_current_gallons

-- Total gallons needed to fill both tanks
def total_gallons_to_add : ℚ := truck_gallons_to_add + car_gallons_to_add

-- Theorem statement
theorem wendy_adds_18_gallons :
  total_gallons_to_add = 18 := sorry

end NUMINAMATH_GPT_wendy_adds_18_gallons_l318_31804


namespace NUMINAMATH_GPT_symmetric_line_eq_l318_31822

theorem symmetric_line_eq (x y : ℝ) (h₁ : y = 3 * x + 4) : y = x → y = (1 / 3) * x - (4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_eq_l318_31822


namespace NUMINAMATH_GPT_part1_part2_part3_l318_31881

noncomputable def functional_relationship (x : ℝ) : ℝ := -x + 26

theorem part1 (x y : ℝ) (hx6 : x = 6 ∧ y = 20) (hx8 : x = 8 ∧ y = 18) (hx10 : x = 10 ∧ y = 16) :
  ∀ (x : ℝ), functional_relationship x = -x + 26 := 
by
  sorry

theorem part2 (x : ℝ) (h_price_range : 6 ≤ x ∧ x ≤ 12) : 
  14 ≤ functional_relationship x ∧ functional_relationship x ≤ 20 :=
by
  sorry

noncomputable def gross_profit (x : ℝ) : ℝ := x * (functional_relationship x - 4)

theorem part3 (hx : 1 ≤ x) (hy : functional_relationship x ≤ 10):
  gross_profit (16 : ℝ) = 120 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l318_31881


namespace NUMINAMATH_GPT_abs_abc_eq_one_l318_31839

theorem abs_abc_eq_one 
  (a b c : ℝ)
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0)
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hca : c ≠ a)
  (h_eq : a + 1/b^2 = b + 1/c^2 ∧ b + 1/c^2 = c + 1/a^2) : 
  |a * b * c| = 1 := 
sorry

end NUMINAMATH_GPT_abs_abc_eq_one_l318_31839


namespace NUMINAMATH_GPT_rectangular_prism_height_eq_17_l318_31865

-- Defining the lengths of the edges of the cubes and rectangular prism
def side_length_cube1 := 10
def edges_cube := 12
def length_rect_prism := 8
def width_rect_prism := 5

-- The total length of the wire used for each shape must be equal
def wire_length_cube1 := edges_cube * side_length_cube1
def wire_length_rect_prism (h : ℕ) := 4 * length_rect_prism + 4 * width_rect_prism + 4 * h

theorem rectangular_prism_height_eq_17 (h : ℕ) :
  wire_length_cube1 = wire_length_rect_prism h → h = 17 := 
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_rectangular_prism_height_eq_17_l318_31865


namespace NUMINAMATH_GPT_trig_expression_value_l318_31805

theorem trig_expression_value (θ : Real) (h1 : θ > Real.pi) (h2 : θ < 3 * Real.pi / 2) (h3 : Real.tan (2 * θ) = 3 / 4) :
  (2 * Real.cos (θ / 2) ^ 2 + Real.sin θ - 1) / (Real.sqrt 2 * Real.cos (θ + Real.pi / 4)) = 2 := by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l318_31805


namespace NUMINAMATH_GPT_line_plane_intersection_l318_31895

theorem line_plane_intersection :
  (∃ t : ℝ, (x, y, z) = (3 + t, 1 - t, -5) ∧ (3 + t) + 7 * (1 - t) + 3 * (-5) + 11 = 0) →
  (x, y, z) = (4, 0, -5) :=
sorry

end NUMINAMATH_GPT_line_plane_intersection_l318_31895


namespace NUMINAMATH_GPT_main_theorem_l318_31831

-- Declare nonzero complex numbers
variables {x y z : ℂ} 

-- State the conditions
def conditions (x y z : ℂ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  x + y + z = 30 ∧
  (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z

-- Prove the main statement given the conditions
theorem main_theorem (h : conditions x y z) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 33 :=
by
  sorry

end NUMINAMATH_GPT_main_theorem_l318_31831


namespace NUMINAMATH_GPT_derivative_at_pi_div_3_l318_31824

noncomputable def f (x : ℝ) : ℝ := (1 + Real.sqrt 2) * Real.sin x - Real.cos x

theorem derivative_at_pi_div_3 :
  deriv f (π / 3) = (1 / 2) * (1 + Real.sqrt 2 + Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_pi_div_3_l318_31824


namespace NUMINAMATH_GPT_pushups_count_l318_31829

theorem pushups_count :
  ∀ (David Zachary Hailey : ℕ),
    David = 44 ∧ (David = Zachary + 9) ∧ (Zachary = 2 * Hailey) ∧ (Hailey = 27) →
      (David = 63 ∧ Zachary = 54 ∧ Hailey = 27) :=
by
  intros David Zachary Hailey
  intro conditions
  obtain ⟨hDavid44, hDavid9Zachary, hZachary2Hailey, hHailey27⟩ := conditions
  sorry

end NUMINAMATH_GPT_pushups_count_l318_31829


namespace NUMINAMATH_GPT_solve_inequality_l318_31803

theorem solve_inequality (x : ℝ) : 
  -2 < (x^2 - 18*x + 35) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 18*x + 35) / (x^2 - 4*x + 8) < 2 ↔ 
  3 < x ∧ x < 17 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l318_31803


namespace NUMINAMATH_GPT_smallest_x_l318_31855

theorem smallest_x (x : ℕ) (h : 450 * x % 648 = 0) : x = 36 := 
sorry

end NUMINAMATH_GPT_smallest_x_l318_31855


namespace NUMINAMATH_GPT_article_final_price_l318_31890

theorem article_final_price (list_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) : 
  first_discount = 0.1 → 
  second_discount = 0.01999999999999997 → 
  list_price = 70 → 
  ∃ final_price, final_price = 61.74 := 
by {
  sorry
}

end NUMINAMATH_GPT_article_final_price_l318_31890


namespace NUMINAMATH_GPT_work_completion_days_l318_31878

theorem work_completion_days (x : ℕ) 
  (h1 : (1 : ℚ) / x + 1 / 9 = 1 / 6) :
  x = 18 := 
sorry

end NUMINAMATH_GPT_work_completion_days_l318_31878


namespace NUMINAMATH_GPT_trajectory_equation_l318_31888

theorem trajectory_equation (x y : ℝ) : x^2 + y^2 = 2 * |x| + 2 * |y| → x^2 + y^2 = 2 * |x| + 2 * |y| :=
by
  sorry

end NUMINAMATH_GPT_trajectory_equation_l318_31888


namespace NUMINAMATH_GPT_eval_expression_l318_31874

theorem eval_expression :
  6 - 9 * (1 / 2 - 3^3) * 2 = 483 := 
sorry

end NUMINAMATH_GPT_eval_expression_l318_31874


namespace NUMINAMATH_GPT_coins_of_each_type_l318_31899

theorem coins_of_each_type (x : ℕ) (h : x + x / 2 + x / 4 = 70) : x = 40 :=
sorry

end NUMINAMATH_GPT_coins_of_each_type_l318_31899


namespace NUMINAMATH_GPT_carol_first_six_l318_31834

-- A formalization of the probabilities involved when Alice, Bob, Carol,
-- and Dave take turns rolling a die, and the process repeats.
def probability_carol_first_six (prob_rolling_six : ℚ) : ℚ := sorry

theorem carol_first_six (prob_rolling_six : ℚ) (h : prob_rolling_six = 1/6) :
  probability_carol_first_six prob_rolling_six = 25 / 91 :=
sorry

end NUMINAMATH_GPT_carol_first_six_l318_31834


namespace NUMINAMATH_GPT_selection_probability_correct_l318_31870

def percentage_women : ℝ := 0.55
def percentage_men : ℝ := 0.45

def women_below_35 : ℝ := 0.20
def women_35_to_50 : ℝ := 0.35
def women_above_50 : ℝ := 0.45

def men_below_35 : ℝ := 0.30
def men_35_to_50 : ℝ := 0.40
def men_above_50 : ℝ := 0.30

def women_below_35_lawyers : ℝ := 0.35
def women_below_35_doctors : ℝ := 0.45
def women_below_35_engineers : ℝ := 0.20

def women_35_to_50_lawyers : ℝ := 0.25
def women_35_to_50_doctors : ℝ := 0.50
def women_35_to_50_engineers : ℝ := 0.25

def women_above_50_lawyers : ℝ := 0.20
def women_above_50_doctors : ℝ := 0.30
def women_above_50_engineers : ℝ := 0.50

def men_below_35_lawyers : ℝ := 0.40
def men_below_35_doctors : ℝ := 0.30
def men_below_35_engineers : ℝ := 0.30

def men_35_to_50_lawyers : ℝ := 0.45
def men_35_to_50_doctors : ℝ := 0.25
def men_35_to_50_engineers : ℝ := 0.30

def men_above_50_lawyers : ℝ := 0.30
def men_above_50_doctors : ℝ := 0.40
def men_above_50_engineers : ℝ := 0.30

theorem selection_probability_correct :
  (percentage_women * women_below_35 * women_below_35_lawyers +
   percentage_men * men_above_50 * men_above_50_engineers +
   percentage_women * women_35_to_50 * women_35_to_50_doctors +
   percentage_men * men_35_to_50 * men_35_to_50_doctors) = 0.22025 :=
by
  sorry

end NUMINAMATH_GPT_selection_probability_correct_l318_31870


namespace NUMINAMATH_GPT_complement_of_M_wrt_U_l318_31821

-- Definitions of the sets U and M as given in the problem
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}

-- The goal is to show the complement of M w.r.t. U is {2, 4, 6}
theorem complement_of_M_wrt_U :
  (U \ M) = {2, 4, 6} := 
by
  sorry

end NUMINAMATH_GPT_complement_of_M_wrt_U_l318_31821


namespace NUMINAMATH_GPT_sugar_used_in_two_minutes_l318_31893

-- Definitions according to conditions
def sugar_per_bar : ℝ := 1.5
def bars_per_minute : ℝ := 36
def minutes : ℝ := 2

-- Theorem statement
theorem sugar_used_in_two_minutes : bars_per_minute * sugar_per_bar * minutes = 108 :=
by
  -- We add sorry here to complete the proof later.
  sorry

end NUMINAMATH_GPT_sugar_used_in_two_minutes_l318_31893


namespace NUMINAMATH_GPT_max_area_circle_center_l318_31872

theorem max_area_circle_center (k : ℝ) :
  (∃ (x y : ℝ), (x + k / 2)^2 + (y + 1)^2 = 1 - 3 / 4 * k^2 ∧ k = 0) →
  x = 0 ∧ y = -1 :=
sorry

end NUMINAMATH_GPT_max_area_circle_center_l318_31872


namespace NUMINAMATH_GPT_limit_of_an_l318_31862

theorem limit_of_an (a_n : ℕ → ℝ) (a : ℝ) : 
  (∀ n, a_n n = (4 * n - 3) / (2 * n + 1)) → 
  a = 2 → 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) :=
by
  intros ha hA ε hε
  sorry

end NUMINAMATH_GPT_limit_of_an_l318_31862


namespace NUMINAMATH_GPT_vector_subtraction_l318_31858

def a : Real × Real := (2, -1)
def b : Real × Real := (-2, 3)

theorem vector_subtraction :
  a.1 - 2 * b.1 = 6 ∧ a.2 - 2 * b.2 = -7 := by
  sorry

end NUMINAMATH_GPT_vector_subtraction_l318_31858


namespace NUMINAMATH_GPT_john_needs_392_tanks_l318_31850

/- Variables representing the conditions -/
def small_balloons : ℕ := 5000
def medium_balloons : ℕ := 5000
def large_balloons : ℕ := 5000

def small_balloon_volume : ℕ := 20
def medium_balloon_volume : ℕ := 30
def large_balloon_volume : ℕ := 50

def helium_tank_capacity : ℕ := 1000
def hydrogen_tank_capacity : ℕ := 1200
def mixture_tank_capacity : ℕ := 1500

/- Mathematical calculations -/
def helium_volume : ℕ := small_balloons * small_balloon_volume
def hydrogen_volume : ℕ := medium_balloons * medium_balloon_volume
def mixture_volume : ℕ := large_balloons * large_balloon_volume

def helium_tanks : ℕ := (helium_volume + helium_tank_capacity - 1) / helium_tank_capacity
def hydrogen_tanks : ℕ := (hydrogen_volume + hydrogen_tank_capacity - 1) / hydrogen_tank_capacity
def mixture_tanks : ℕ := (mixture_volume + mixture_tank_capacity - 1) / mixture_tank_capacity

def total_tanks : ℕ := helium_tanks + hydrogen_tanks + mixture_tanks

theorem john_needs_392_tanks : total_tanks = 392 :=
by {
  -- calculation proof goes here
  sorry
}

end NUMINAMATH_GPT_john_needs_392_tanks_l318_31850


namespace NUMINAMATH_GPT_product_of_digits_l318_31833

theorem product_of_digits (A B : ℕ) (h1 : A + B = 12) (h2 : 8 ∣ (10 * A + B)) : A * B = 32 :=
sorry

end NUMINAMATH_GPT_product_of_digits_l318_31833


namespace NUMINAMATH_GPT_find_number_l318_31898

-- Define the condition
def is_number (x : ℝ) : Prop :=
  0.15 * x = 0.25 * 16 + 2

-- The theorem statement: proving the number is 40
theorem find_number (x : ℝ) (h : is_number x) : x = 40 :=
by
  -- We would insert the proof steps here
  sorry

end NUMINAMATH_GPT_find_number_l318_31898


namespace NUMINAMATH_GPT_monotonic_intervals_of_f_g_minus_f_less_than_3_l318_31851

noncomputable def f (x : ℝ) : ℝ := -x * Real.log (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - x

theorem monotonic_intervals_of_f :
  ∀ x : ℝ, x < -1 / Real.exp 1 → f x < f (-1 / Real.exp 1) ∧ x > -1 / Real.exp 1 → f x > f (-1 / Real.exp 1) := sorry

theorem g_minus_f_less_than_3 :
  ∀ x : ℝ, x < 0 → g x - f x < 3 := sorry

end NUMINAMATH_GPT_monotonic_intervals_of_f_g_minus_f_less_than_3_l318_31851


namespace NUMINAMATH_GPT_distribute_diamonds_among_two_safes_l318_31844

theorem distribute_diamonds_among_two_safes (N : ℕ) :
  ∀ banker : ℕ, banker < 777 → ∃ s1 s2 : ℕ, s1 ≠ s2 ∧ s1 + s2 = N := sorry

end NUMINAMATH_GPT_distribute_diamonds_among_two_safes_l318_31844


namespace NUMINAMATH_GPT_logarithm_equation_l318_31807

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem logarithm_equation (a : ℝ) : 
  (1 / log_base 2 a + 1 / log_base 3 a + 1 / log_base 4 a = 1) → a = 24 :=
by
  sorry

end NUMINAMATH_GPT_logarithm_equation_l318_31807


namespace NUMINAMATH_GPT_pattyCoinsValue_l318_31802

def totalCoins (q d : ℕ) : Prop := q + d = 30
def originalValue (q d : ℕ) : ℝ := 0.25 * q + 0.10 * d
def swappedValue (q d : ℕ) : ℝ := 0.10 * q + 0.25 * d
def valueIncrease (q : ℕ) : Prop := swappedValue q (30 - q) - originalValue q (30 - q) = 1.20

theorem pattyCoinsValue (q d : ℕ) (h1 : totalCoins q d) (h2 : valueIncrease q) : originalValue q d = 4.65 := 
by
  sorry

end NUMINAMATH_GPT_pattyCoinsValue_l318_31802


namespace NUMINAMATH_GPT_xiaoLiangComprehensiveScore_l318_31871

-- Define the scores for the three aspects
def contentScore : ℝ := 88
def deliveryAbilityScore : ℝ := 95
def effectivenessScore : ℝ := 90

-- Define the weights for the three aspects
def contentWeight : ℝ := 0.5
def deliveryAbilityWeight : ℝ := 0.4
def effectivenessWeight : ℝ := 0.1

-- Define the comprehensive score
def comprehensiveScore : ℝ :=
  (contentScore * contentWeight) +
  (deliveryAbilityScore * deliveryAbilityWeight) +
  (effectivenessScore * effectivenessWeight)

-- The theorem stating that the comprehensive score equals 91
theorem xiaoLiangComprehensiveScore : comprehensiveScore = 91 := by
  -- proof here (omitted)
  sorry

end NUMINAMATH_GPT_xiaoLiangComprehensiveScore_l318_31871


namespace NUMINAMATH_GPT_cindys_correct_result_l318_31810

-- Explicitly stating the conditions as definitions
def incorrect_operation_result := 260
def x := (incorrect_operation_result / 5) - 7

theorem cindys_correct_result : 5 * x + 7 = 232 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_cindys_correct_result_l318_31810


namespace NUMINAMATH_GPT_remainder_of_sum_of_primes_l318_31884

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_primes_l318_31884


namespace NUMINAMATH_GPT_max_value_of_reciprocal_powers_l318_31892

variable {R : Type*} [CommRing R]
variables (s q r₁ r₂ : R)

-- Condition: the roots of the polynomial
def is_roots_of_polynomial (s q r₁ r₂ : R) : Prop :=
  r₁ + r₂ = s ∧ r₁ * r₂ = q ∧ (r₁ + r₂ = r₁ ^ 2 + r₂ ^ 2) ∧ (r₁ + r₂ = r₁^10 + r₂^10)

-- The theorem that needs to be proven
theorem max_value_of_reciprocal_powers (s q r₁ r₂ : ℝ) (h : is_roots_of_polynomial s q r₁ r₂):
  (∃ r₁ r₂, r₁ + r₂ = s ∧ r₁ * r₂ = q ∧
             r₁ + r₂ = r₁^2 + r₂^2 ∧
             r₁ + r₂ = r₁^10 + r₂^10) →
  (r₁^ 11 ≠ 0 ∧ r₂^11 ≠ 0 ∧
  ((1 / r₁^11) + (1 / r₂^11) = 2)) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_reciprocal_powers_l318_31892


namespace NUMINAMATH_GPT_find_angle_C_max_area_l318_31814

-- Define the conditions as hypotheses
variable (a b c : ℝ) (A B C : ℝ)
variable (h1 : c = 2 * Real.sqrt 3)
variable (h2 : c * Real.cos B + (b - 2 * a) * Real.cos C = 0)

-- Problem (1): Prove that angle C is π/3
theorem find_angle_C : C = Real.pi / 3 :=
by
  sorry

-- Problem (2): Prove that the maximum area of triangle ABC is 3√3
theorem max_area : (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_C_max_area_l318_31814


namespace NUMINAMATH_GPT_distinct_triangles_in_regular_ngon_l318_31825

theorem distinct_triangles_in_regular_ngon (n : ℕ) (h : n ≥ 3) :
  ∃ t : ℕ, t = n * (n-1) * (n-2) / 6 := 
sorry

end NUMINAMATH_GPT_distinct_triangles_in_regular_ngon_l318_31825


namespace NUMINAMATH_GPT_trader_allows_discount_l318_31827

-- Definitions for cost price, marked price, and selling price
variable (cp : ℝ)
def mp := cp + 0.12 * cp
def sp := cp - 0.01 * cp

-- The statement to prove
theorem trader_allows_discount :
  mp cp - sp cp = 13 :=
sorry

end NUMINAMATH_GPT_trader_allows_discount_l318_31827


namespace NUMINAMATH_GPT_child_ticket_cost_l318_31857

-- Define the conditions
def adult_ticket_cost : ℕ := 11
def total_people : ℕ := 23
def total_revenue : ℕ := 246
def children_count : ℕ := 7
def adults_count := total_people - children_count

-- Define the target to prove that the child ticket cost is 10
theorem child_ticket_cost (child_ticket_cost : ℕ) :
  16 * adult_ticket_cost + 7 * child_ticket_cost = total_revenue → 
  child_ticket_cost = 10 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_child_ticket_cost_l318_31857


namespace NUMINAMATH_GPT_time_via_route_B_l318_31879

-- Given conditions
def time_via_route_A : ℕ := 5
def time_saved_round_trip : ℕ := 6

-- Defining the proof problem
theorem time_via_route_B : time_via_route_A - (time_saved_round_trip / 2) = 2 :=
by
  -- Expected proof here
  sorry

end NUMINAMATH_GPT_time_via_route_B_l318_31879


namespace NUMINAMATH_GPT_train_cross_time_l318_31873

-- Define the given conditions
def train_length : ℕ := 100
def train_speed_kmph : ℕ := 45
def total_length : ℕ := 275
def seconds_in_hour : ℕ := 3600
def meters_in_km : ℕ := 1000

-- Convert the speed from km/hr to m/s
noncomputable def train_speed_mps : ℚ := (train_speed_kmph * meters_in_km) / seconds_in_hour

-- The time to cross the bridge
noncomputable def time_to_cross (train_length total_length : ℕ) (train_speed_mps : ℚ) : ℚ :=
  total_length / train_speed_mps

-- The statement we want to prove
theorem train_cross_time : time_to_cross train_length total_length train_speed_mps = 30 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_time_l318_31873


namespace NUMINAMATH_GPT_expand_product_l318_31836

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 :=
by 
  sorry

end NUMINAMATH_GPT_expand_product_l318_31836


namespace NUMINAMATH_GPT_yan_distance_ratio_l318_31828

theorem yan_distance_ratio 
  (w x y : ℝ)
  (h1 : y / w = x / w + (x + y) / (10 * w)) :
  x / y = 9 / 11 :=
by
  sorry

end NUMINAMATH_GPT_yan_distance_ratio_l318_31828


namespace NUMINAMATH_GPT_coefficient_of_neg2ab_is_neg2_l318_31869

-- Define the term -2ab
def term : ℤ := -2

-- Define the function to get the coefficient from term -2ab
def coefficient (t : ℤ) : ℤ := t

-- The theorem stating the coefficient of -2ab is -2
theorem coefficient_of_neg2ab_is_neg2 : coefficient term = -2 :=
by
  -- Proof can be filled later
  sorry

end NUMINAMATH_GPT_coefficient_of_neg2ab_is_neg2_l318_31869


namespace NUMINAMATH_GPT_circle_area_l318_31812

theorem circle_area (r : ℝ) (h : 2 * (1 / (2 * π * r)) = r / 2) : π * r^2 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_circle_area_l318_31812


namespace NUMINAMATH_GPT_third_term_arithmetic_sequence_l318_31868

theorem third_term_arithmetic_sequence (a x : ℝ) 
  (h : 2 * a + 2 * x = 10) : a + x = 5 := 
by
  sorry

end NUMINAMATH_GPT_third_term_arithmetic_sequence_l318_31868


namespace NUMINAMATH_GPT_gridiron_football_club_members_count_l318_31886

theorem gridiron_football_club_members_count :
  let sock_price := 6
  let tshirt_price := sock_price + 7
  let helmet_price := 2 * tshirt_price
  let total_cost_per_member := sock_price + tshirt_price + helmet_price
  let total_expenditure := 4680
  total_expenditure / total_cost_per_member = 104 :=
by
  let sock_price := 6
  let tshirt_price := sock_price + 7
  let helmet_price := 2 * tshirt_price
  let total_cost_per_member := sock_price + tshirt_price + helmet_price
  let total_expenditure := 4680
  sorry

end NUMINAMATH_GPT_gridiron_football_club_members_count_l318_31886


namespace NUMINAMATH_GPT_remainder_of_3_pow_19_mod_5_l318_31832

theorem remainder_of_3_pow_19_mod_5 : (3 ^ 19) % 5 = 2 := by
  have h : 3 ^ 4 % 5 = 1 := by sorry
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_19_mod_5_l318_31832


namespace NUMINAMATH_GPT_find_n_l318_31877

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 103) (h3 : 100 * n ≡ 85 [MOD 103]) : n = 6 := 
sorry

end NUMINAMATH_GPT_find_n_l318_31877


namespace NUMINAMATH_GPT_find_divisor_l318_31841

-- Define the given conditions
def dividend : ℕ := 122
def quotient : ℕ := 6
def remainder : ℕ := 2

-- Define the proof problem to find the divisor
theorem find_divisor : 
  ∃ D : ℕ, dividend = (D * quotient) + remainder ∧ D = 20 :=
by sorry

end NUMINAMATH_GPT_find_divisor_l318_31841


namespace NUMINAMATH_GPT_restore_supply_function_determine_tax_revenue_determine_optimal_tax_rate_determine_maximum_tax_revenue_l318_31867

-- Defining the given conditions
noncomputable def market_demand (P : ℝ) : ℝ := 688 - 4 * P
noncomputable def post_tax_producer_price : ℝ := 64
noncomputable def per_unit_tax : ℝ := 90
noncomputable def elasticity_supply_no_tax (P_e : ℝ) (Q_e : ℝ) : ℝ :=
  1.5 * (-(4 * P_e / Q_e))

-- Supply function to be proven
noncomputable def supply_function (P : ℝ) : ℝ := 6 * P - 312

-- Total tax revenue to be proven
noncomputable def total_tax_revenue : ℝ := 6480

-- Optimal tax rate to be proven
noncomputable def optimal_tax_rate : ℝ := 60

-- Maximum tax revenue to be proven
noncomputable def maximum_tax_revenue : ℝ := 8640

-- Theorem statements that need to be proven
theorem restore_supply_function (P : ℝ) : 
  supply_function P = 6 * P - 312 := sorry

theorem determine_tax_revenue : 
  total_tax_revenue = 6480 := sorry

theorem determine_optimal_tax_rate : 
  optimal_tax_rate = 60 := sorry

theorem determine_maximum_tax_revenue : 
  maximum_tax_revenue = 8640 := sorry

end NUMINAMATH_GPT_restore_supply_function_determine_tax_revenue_determine_optimal_tax_rate_determine_maximum_tax_revenue_l318_31867


namespace NUMINAMATH_GPT_joann_lollipops_l318_31840

theorem joann_lollipops : 
  ∃ (a : ℚ), 
  (7 * a  + 3 * (1 + 2 + 3 + 4 + 5 + 6) = 150) ∧ 
  (a_4 = a + 9) ∧ 
  (a_4 = 150 / 7) :=
by
  sorry

end NUMINAMATH_GPT_joann_lollipops_l318_31840


namespace NUMINAMATH_GPT_find_x_l318_31811

theorem find_x (x y : ℤ) (h1 : x + 2 * y = 100) (h2 : y = 25) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l318_31811


namespace NUMINAMATH_GPT_find_w_l318_31830

variables {x y : ℚ}

def w : ℚ × ℚ := (-48433 / 975, 2058 / 325)

def vec1 : ℚ × ℚ := (3, 2)
def vec2 : ℚ × ℚ := (3, 4)

def proj (u v : ℚ × ℚ) : ℚ × ℚ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

def p1 : ℚ × ℚ := (47 / 13, 31 / 13)
def p2 : ℚ × ℚ := (85 / 25, 113 / 25)

theorem find_w (hw : w = (x, y)) :
  proj ⟨x, y⟩ vec1 = p1 ∧
  proj ⟨x, y⟩ vec2 = p2 :=
sorry

end NUMINAMATH_GPT_find_w_l318_31830


namespace NUMINAMATH_GPT_find_P_and_Q_l318_31818

variables {x P Q b c : ℝ}

theorem find_P_and_Q :
  (∃ b c : ℝ, (x^2 + 3 * x + 7) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) →
  (b + 3 = 0) →
  (3 * b + c + 7 = P) →
  (7 * b + 3 * c = 0) →
  (7 * c = Q) →
  P + Q = 54 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_find_P_and_Q_l318_31818


namespace NUMINAMATH_GPT_sixth_day_is_wednesday_l318_31846

noncomputable def day_of_week : Type := 
  { d // d ∈ ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"] }

def five_fridays_sum_correct (x : ℤ) : Prop :=
  x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 75

def first_is_friday (x : ℤ) : Prop :=
  x = 1

def day_of_6th_is_wednesday (d : day_of_week) : Prop :=
  d.1 = "Wednesday"

theorem sixth_day_is_wednesday (x : ℤ) (d : day_of_week) :
  five_fridays_sum_correct x → first_is_friday x → day_of_6th_is_wednesday d :=
by
  sorry

end NUMINAMATH_GPT_sixth_day_is_wednesday_l318_31846


namespace NUMINAMATH_GPT_village_population_equal_in_years_l318_31860

theorem village_population_equal_in_years :
  ∀ (n : ℕ), (70000 - 1200 * n = 42000 + 800 * n) ↔ n = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_village_population_equal_in_years_l318_31860


namespace NUMINAMATH_GPT_problem1_problem2_l318_31826

-- Statement for Problem ①
theorem problem1 
: ( (-1 / 12 - 1 / 36 + 1 / 6) * (-36) = -2) := by
  sorry

-- Statement for Problem ②
theorem problem2
: ((-99 - 11 / 12) * 24 = -2398) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l318_31826


namespace NUMINAMATH_GPT_hash_difference_l318_31806

def hash (x y : ℕ) : ℤ := x * y - 3 * x + y

theorem hash_difference :
  (hash 8 5) - (hash 5 8) = -12 :=
by
  sorry

end NUMINAMATH_GPT_hash_difference_l318_31806


namespace NUMINAMATH_GPT_relationship_y1_y2_l318_31887

theorem relationship_y1_y2 (x1 x2 y1 y2 : ℝ) 
  (h1: x1 > 0) 
  (h2: 0 > x2) 
  (h3: y1 = 2 / x1)
  (h4: y2 = 2 / x2) : 
  y1 > y2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_l318_31887


namespace NUMINAMATH_GPT_smallest_irreducible_l318_31816

def is_irreducible (n : ℕ) : Prop :=
  ∀ k : ℕ, 19 ≤ k ∧ k ≤ 91 → Nat.gcd k (n + k + 2) = 1

theorem smallest_irreducible : ∃ n : ℕ, is_irreducible n ∧ ∀ m : ℕ, m < n → ¬ is_irreducible m :=
  by
  exists 95
  sorry

end NUMINAMATH_GPT_smallest_irreducible_l318_31816


namespace NUMINAMATH_GPT_x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero_l318_31823

theorem x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero :
  ∃ (x : ℝ), (x = 1) → (x^2 + x - 2 = 0) ∧ (¬ (∀ (y : ℝ), y^2 + y - 2 = 0 → y = 1)) := by
  sorry

end NUMINAMATH_GPT_x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero_l318_31823


namespace NUMINAMATH_GPT_angle_A_and_shape_of_triangle_l318_31896

theorem angle_A_and_shape_of_triangle 
  (a b c : ℝ)
  (h1 : a^2 - c^2 = a * c - b * c)
  (h2 : ∃ r : ℝ, a = b * r ∧ c = b / r)
  (h3 : ∃ B C : Type, B = A ∧ C ≠ A ) :
  ∃ (A : ℝ), A = 60 ∧ a = b ∧ b = c := 
sorry

end NUMINAMATH_GPT_angle_A_and_shape_of_triangle_l318_31896


namespace NUMINAMATH_GPT_num_photos_to_include_l318_31847

-- Define the conditions
def num_preselected_photos : ℕ := 7
def total_choices : ℕ := 56

-- Define the statement to prove
theorem num_photos_to_include : total_choices / num_preselected_photos = 8 :=
by sorry

end NUMINAMATH_GPT_num_photos_to_include_l318_31847


namespace NUMINAMATH_GPT_rational_product_nonpositive_l318_31883

open Classical

theorem rational_product_nonpositive (a b : ℚ) (ha : |a| = a) (hb : |b| ≠ b) : a * b ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_rational_product_nonpositive_l318_31883


namespace NUMINAMATH_GPT_subset_N_M_l318_31854

-- Define the sets M and N
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | x^2 - x < 0 }

-- The proof goal
theorem subset_N_M : N ⊆ M := by
  sorry

end NUMINAMATH_GPT_subset_N_M_l318_31854


namespace NUMINAMATH_GPT_length_of_second_train_correct_l318_31859

noncomputable def length_of_second_train : ℝ :=
  let speed_first_train := 60 / 3.6
  let speed_second_train := 90 / 3.6
  let relative_speed := speed_first_train + speed_second_train
  let time_to_clear := 6.623470122390208
  let total_distance := relative_speed * time_to_clear
  let length_first_train := 111
  total_distance - length_first_train

theorem length_of_second_train_correct :
  length_of_second_train = 164.978 :=
by
  unfold length_of_second_train
  sorry

end NUMINAMATH_GPT_length_of_second_train_correct_l318_31859
