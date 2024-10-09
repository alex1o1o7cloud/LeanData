import Mathlib

namespace total_combined_rainfall_l821_82155

theorem total_combined_rainfall :
  let monday_hours := 5
  let monday_rate := 1
  let tuesday_hours := 3
  let tuesday_rate := 1.5
  let wednesday_hours := 4
  let wednesday_rate := 2 * monday_rate
  let thursday_hours := 6
  let thursday_rate := tuesday_rate / 2
  let friday_hours := 2
  let friday_rate := 1.5 * wednesday_rate
  let monday_rain := monday_hours * monday_rate
  let tuesday_rain := tuesday_hours * tuesday_rate
  let wednesday_rain := wednesday_hours * wednesday_rate
  let thursday_rain := thursday_hours * thursday_rate
  let friday_rain := friday_hours * friday_rate
  monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain = 28 := by
  sorry

end total_combined_rainfall_l821_82155


namespace larger_number_eq_1599_l821_82146

theorem larger_number_eq_1599 (L S : ℕ) (h1 : L - S = 1335) (h2 : L = 6 * S + 15) : L = 1599 :=
by 
  sorry

end larger_number_eq_1599_l821_82146


namespace factor_expression_l821_82125

theorem factor_expression (x : ℝ) : 6 * x ^ 3 - 54 * x = 6 * x * (x + 3) * (x - 3) :=
by {
  sorry
}

end factor_expression_l821_82125


namespace C_plus_D_l821_82160

theorem C_plus_D (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) : 
  C + D = -10 := by
  sorry

end C_plus_D_l821_82160


namespace fido_leash_problem_l821_82151

theorem fido_leash_problem
  (r : ℝ) 
  (octagon_area : ℝ := 2 * r^2 * Real.sqrt 2)
  (circle_area : ℝ := Real.pi * r^2)
  (explore_fraction : ℝ := circle_area / octagon_area)
  (a b : ℝ) 
  (h_simplest_form : explore_fraction = (Real.sqrt a) / b * Real.pi)
  (h_a : a = 2)
  (h_b : b = 2) : a * b = 4 :=
by sorry

end fido_leash_problem_l821_82151


namespace wheel_stop_probability_l821_82153

theorem wheel_stop_probability 
  (pD pE pG pF : ℚ) 
  (h1 : pD = 1 / 4) 
  (h2 : pE = 1 / 3) 
  (h3 : pG = 1 / 6) 
  (h4 : pD + pE + pG + pF = 1) : 
  pF = 1 / 4 := 
by 
  sorry

end wheel_stop_probability_l821_82153


namespace hammerhead_teeth_fraction_l821_82168

theorem hammerhead_teeth_fraction (f : ℚ) : 
  let t := 180 
  let h := f * t
  let w := 2 * (t + h)
  w = 420 → f = (1 : ℚ) / 6 := by
  intros _ 
  sorry

end hammerhead_teeth_fraction_l821_82168


namespace luke_played_rounds_l821_82183

theorem luke_played_rounds (total_points : ℕ) (points_per_round : ℕ) (result : ℕ)
  (h1 : total_points = 154)
  (h2 : points_per_round = 11)
  (h3 : result = total_points / points_per_round) :
  result = 14 :=
by
  rw [h1, h2] at h3
  exact h3

end luke_played_rounds_l821_82183


namespace distinct_flags_count_l821_82101

theorem distinct_flags_count : 
  ∃ n, n = 36 ∧ (∀ c1 c2 c3 : Fin 4, c1 ≠ c2 ∧ c2 ≠ c3 → n = 4 * 3 * 3) := 
sorry

end distinct_flags_count_l821_82101


namespace lines_perpendicular_l821_82103

/-- Given two lines l1: 3x + 4y + 1 = 0 and l2: 4x - 3y + 2 = 0, 
    prove that the lines are perpendicular. -/
theorem lines_perpendicular :
  ∀ (x y : ℝ), (3 * x + 4 * y + 1 = 0) → (4 * x - 3 * y + 2 = 0) → (- (3 / 4) * (4 / 3) = -1) :=
by
  intro x y h₁ h₂
  sorry

end lines_perpendicular_l821_82103


namespace six_digit_start_5_no_12_digit_perfect_square_l821_82172

theorem six_digit_start_5_no_12_digit_perfect_square :
  ∀ (n : ℕ), (500000 ≤ n ∧ n < 600000) → 
  (∀ (m : ℕ), n * 10^6 + m ≠ k^2) :=
by
  sorry

end six_digit_start_5_no_12_digit_perfect_square_l821_82172


namespace fraction_value_l821_82174

variable (x y : ℝ)

theorem fraction_value (h : 1/x - 1/y = 3) : (2 * x + 3 * x * y - 2 * y) / (x - 2 * x * y - y) = 3 / 5 := 
by sorry

end fraction_value_l821_82174


namespace abs_neg_five_halves_l821_82169

theorem abs_neg_five_halves : abs (-5 / 2) = 5 / 2 := by
  sorry

end abs_neg_five_halves_l821_82169


namespace length_of_train_75_l821_82149

variable (L : ℝ) -- Length of the train in meters

-- Condition 1: The train crosses a bridge of length 150 m in 7.5 seconds
def crosses_bridge (L: ℝ) : Prop := (L + 150) / 7.5 = L / 2.5

-- Condition 2: The train crosses a lamp post in 2.5 seconds
def crosses_lamp (L: ℝ) : Prop := L / 2.5 = L / 2.5

theorem length_of_train_75 (L : ℝ) (h1 : crosses_bridge L) (h2 : crosses_lamp L) : L = 75 := 
by 
  sorry

end length_of_train_75_l821_82149


namespace reciprocal_neg_5_l821_82112

theorem reciprocal_neg_5 : ∃ x : ℚ, -5 * x = 1 ∧ x = -1/5 :=
by
  sorry

end reciprocal_neg_5_l821_82112


namespace hyperbola_condition_l821_82198

noncomputable def a_b_sum (a b : ℝ) : ℝ :=
  a + b

theorem hyperbola_condition
  (a b : ℝ)
  (h1 : a^2 - b^2 = 1)
  (h2 : abs (a - b) = 2)
  (h3 : a > b) :
  a_b_sum a b = 1/2 :=
sorry

end hyperbola_condition_l821_82198


namespace power_modulo_l821_82121

theorem power_modulo (h : 3 ^ 4 ≡ 1 [MOD 10]) : 3 ^ 2023 ≡ 7 [MOD 10] :=
by
  sorry

end power_modulo_l821_82121


namespace simplify_and_evaluate_l821_82147

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end simplify_and_evaluate_l821_82147


namespace function_increasing_value_of_a_function_decreasing_value_of_a_l821_82133

-- Part 1: Prove that if \( f(x) = x^3 - ax - 1 \) is increasing on the interval \( (1, +\infty) \), then \( a \leq 3 \)
theorem function_increasing_value_of_a (a : ℝ) :
  (∀ x > 1, 3 * x^2 - a ≥ 0) → a ≤ 3 := by
  sorry

-- Part 2: Prove that if the decreasing interval of \( f(x) = x^3 - ax - 1 \) is \( (-1, 1) \), then \( a = 3 \)
theorem function_decreasing_value_of_a (a : ℝ) :
  (∀ x, -1 < x ∧ x < 1 → 3 * x^2 - a < 0) ∧ (3 * (-1)^2 - a = 0 ∧ 3 * (1)^2 - a = 0) → a = 3 := by
  sorry

end function_increasing_value_of_a_function_decreasing_value_of_a_l821_82133


namespace number_with_specific_places_l821_82154

theorem number_with_specific_places :
  ∃ (n : Real), 
    (n / 10 % 10 = 6) ∧ -- tens place
    (n / 1 % 10 = 0) ∧  -- ones place
    (n * 10 % 10 = 0) ∧  -- tenths place
    (n * 100 % 10 = 6) →  -- hundredths place
    n = 60.06 :=
by
  sorry

end number_with_specific_places_l821_82154


namespace line_through_A1_slope_neg4_over_3_line_through_A2_l821_82105

-- (1) The line passing through point (1, 3) with a slope -4/3
theorem line_through_A1_slope_neg4_over_3 : 
    ∃ (a b c : ℝ), a * 1 + b * 3 + c = 0 ∧ ∃ m : ℝ, m = -4 / 3 ∧ a * m + b = 0 ∧ b ≠ 0 ∧ c = -13 := by
sorry

-- (2) The line passing through point (-5, 2) with x-intercept twice the y-intercept
theorem line_through_A2 : 
    ∃ (a b c : ℝ), (a * -5 + b * 2 + c = 0) ∧ ((∃ m : ℝ, m = 2 ∧ a * m + b = 0 ∧ b = -a) ∨ ((b = -2 / 5 * a) ∧ (a * 2 + b = 0))) := by
sorry

end line_through_A1_slope_neg4_over_3_line_through_A2_l821_82105


namespace certain_number_value_l821_82171

theorem certain_number_value :
  ∃ n : ℚ, 9 - (4 / 6) = 7 + (n / 6) ∧ n = 8 := by
sorry

end certain_number_value_l821_82171


namespace animals_total_l821_82184

-- Given definitions and conditions
def ducks : ℕ := 25
def rabbits : ℕ := 8
def chickens := 4 * ducks

-- Proof statement
theorem animals_total (h1 : chickens = 4 * ducks)
                     (h2 : ducks - 17 = rabbits)
                     (h3 : rabbits = 8) :
  chickens + ducks + rabbits = 133 := by
  sorry

end animals_total_l821_82184


namespace boxes_to_eliminate_l821_82106

noncomputable def total_boxes : ℕ := 26
noncomputable def high_value_boxes : ℕ := 6
noncomputable def threshold_probability : ℚ := 1 / 2

-- Define the condition for having the minimum number of boxes
def min_boxes_needed_for_probability (total high_value : ℕ) (prob : ℚ) : ℕ :=
  total - high_value - ((total - high_value) / 2)

theorem boxes_to_eliminate :
  min_boxes_needed_for_probability total_boxes high_value_boxes threshold_probability = 15 :=
by
  sorry

end boxes_to_eliminate_l821_82106


namespace smallest_digit_never_in_units_place_of_odd_numbers_l821_82109

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l821_82109


namespace length_OD1_l821_82136

-- Define the hypothesis of the problem
noncomputable def sphere_center : Point := sorry -- center O of the sphere
noncomputable def radius_sphere : ℝ := 10 -- radius of the sphere

-- Define face intersection properties
noncomputable def face_AA1D1D_radius : ℝ := 1
noncomputable def face_A1B1C1D1_radius : ℝ := 1
noncomputable def face_CDD1C1_radius : ℝ := 3

-- Define the coordinates of D1 (or in abstract form, we'll assume it is a known point)
noncomputable def segment_OD1 : ℝ := sorry -- Length of OD1 segment to be calculated

-- The main theorem to prove
theorem length_OD1 : 
  -- Given conditions
  (face_AA1D1D_radius = 1) ∧ 
  (face_A1B1C1D1_radius = 1) ∧ 
  (face_CDD1C1_radius = 3) ∧ 
  (radius_sphere = 10) →
  -- Prove the length of segment OD1 is 17
  segment_OD1 = 17 :=
by
  sorry

end length_OD1_l821_82136


namespace arithmetic_sequence_common_difference_and_m_l821_82185

theorem arithmetic_sequence_common_difference_and_m (S : ℕ → ℤ) (a : ℕ → ℤ) (m d : ℕ) 
(h1 : S (m-1) = -2) (h2 : S m = 0) (h3 : S (m+1) = 3) :
  d = 1 ∧ m = 5 :=
by sorry

end arithmetic_sequence_common_difference_and_m_l821_82185


namespace fraction_taken_by_kiley_l821_82114

-- Define the constants and conditions
def total_crayons : ℕ := 48
def remaining_crayons_after_joe : ℕ := 18

-- Define the main statement to be proven
theorem fraction_taken_by_kiley (f : ℚ) : 
  (48 - (48 * f)) / 2 = 18 → f = 1 / 4 :=
by 
  intro h
  sorry

end fraction_taken_by_kiley_l821_82114


namespace small_cube_edge_length_l821_82196

theorem small_cube_edge_length 
  (m n : ℕ)
  (h1 : 12 % m = 0) 
  (h2 : n = 12 / m) 
  (h3 : 6 * (n - 2)^2 = 12 * (n - 2)) 
  : m = 3 :=
by 
  sorry

end small_cube_edge_length_l821_82196


namespace correct_time_fraction_l821_82199

theorem correct_time_fraction :
  let hours := 24
  let correct_hours := 10
  let minutes_per_hour := 60
  let correct_minutes_per_hour := 20
  (correct_hours * correct_minutes_per_hour : ℝ) / (hours * minutes_per_hour) = (5 / 36 : ℝ) :=
by
  let hours := 24
  let correct_hours := 10
  let minutes_per_hour := 60
  let correct_minutes_per_hour := 20
  sorry

end correct_time_fraction_l821_82199


namespace toaster_sales_promotion_l821_82117

theorem toaster_sales_promotion :
  ∀ (p : ℕ) (c₁ c₂ : ℕ) (k : ℕ), 
    (c₁ = 600 ∧ p = 15 ∧ k = p * c₁) ∧ 
    (c₂ = 450 ∧ (p * c₂ = k) ) ∧ 
    (p' = p * 11 / 10) →
    p' = 22 :=
by 
  sorry

end toaster_sales_promotion_l821_82117


namespace min_value_Px_Py_l821_82193

def P (τ : ℝ) : ℝ := (τ + 1)^3

theorem min_value_Px_Py (x y : ℝ) (h : x + y = 0) : P x + P y = 2 :=
sorry

end min_value_Px_Py_l821_82193


namespace find_interest_rate_l821_82122

theorem find_interest_rate 
    (P : ℝ) (T : ℝ) (known_rate : ℝ) (diff : ℝ) (R : ℝ) :
    P = 7000 → T = 2 → known_rate = 0.18 → diff = 840 → (P * known_rate * T - (P * (R/100) * T) = diff) → R = 12 :=
by
  intros P_eq T_eq kr_eq diff_eq interest_eq
  simp only [P_eq, T_eq, kr_eq, diff_eq] at interest_eq
-- Solving equation is not required
  sorry

end find_interest_rate_l821_82122


namespace range_of_a_iff_l821_82116

def cubic_inequality (x : ℝ) : Prop := x^3 + 3 * x^2 - x - 3 > 0

def quadratic_inequality (x a : ℝ) : Prop := x^2 - 2 * a * x - 1 ≤ 0

def integer_solution_condition (x : ℤ) (a : ℝ) : Prop := 
  x^3 + 3 * x^2 - x - 3 > 0 ∧ x^2 - 2 * a * x - 1 ≤ 0

def range_of_a (a : ℝ) : Prop := (3 / 4 : ℝ) ≤ a ∧ a < (4 / 3 : ℝ)

theorem range_of_a_iff : 
  (∃ x : ℤ, integer_solution_condition x a) ↔ range_of_a a := 
sorry

end range_of_a_iff_l821_82116


namespace domain_sqrt_product_domain_log_fraction_l821_82108

theorem domain_sqrt_product (x : ℝ) (h1 : x - 2 ≥ 0) (h2 : x + 2 ≥ 0) : 
  2 ≤ x :=
by sorry

theorem domain_log_fraction (x : ℝ) (h1 : x + 1 > 0) (h2 : -x^2 - 3 * x + 4 > 0) : 
  -1 < x ∧ x < 1 :=
by sorry

end domain_sqrt_product_domain_log_fraction_l821_82108


namespace area_of_rectangle_l821_82102

-- Define the conditions
def width : ℕ := 6
def perimeter : ℕ := 28

-- Define the theorem statement
theorem area_of_rectangle (w : ℕ) (p : ℕ) (h_width : w = width) (h_perimeter : p = perimeter) :
  ∃ l : ℕ, (2 * (l + w) = p) → (l * w = 48) :=
by
  use 8
  intro h
  simp only [h_width, h_perimeter] at h
  sorry

end area_of_rectangle_l821_82102


namespace tangent_line_count_l821_82141

noncomputable def circles_tangent_lines (r1 r2 d : ℝ) : ℕ :=
if d = |r1 - r2| then 1 else 0 -- Define the function based on the problem statement

theorem tangent_line_count :
  circles_tangent_lines 4 5 3 = 1 := 
by
  -- Placeholder for the proof, which we are skipping as per instructions
  sorry

end tangent_line_count_l821_82141


namespace infinite_solutions_to_congruence_l821_82137

theorem infinite_solutions_to_congruence :
  ∃ᶠ n in atTop, 3^((n-2)^(n-1)-1) ≡ 1 [MOD 17 * n^2] :=
by
  sorry

end infinite_solutions_to_congruence_l821_82137


namespace smallest_tax_amount_is_professional_income_tax_l821_82130

def total_income : ℝ := 50000.00
def professional_deductions : ℝ := 35000.00

def tax_rate_ndfl : ℝ := 0.13
def tax_rate_simplified_income : ℝ := 0.06
def tax_rate_simplified_income_minus_expenditure : ℝ := 0.15
def tax_rate_professional_income : ℝ := 0.04

def ndfl_tax : ℝ := (total_income - professional_deductions) * tax_rate_ndfl
def simplified_tax_income : ℝ := total_income * tax_rate_simplified_income
def simplified_tax_income_minus_expenditure : ℝ := (total_income - professional_deductions) * tax_rate_simplified_income_minus_expenditure
def professional_income_tax : ℝ := total_income * tax_rate_professional_income

theorem smallest_tax_amount_is_professional_income_tax : 
  min (min ndfl_tax (min simplified_tax_income simplified_tax_income_minus_expenditure)) professional_income_tax = professional_income_tax := 
sorry

end smallest_tax_amount_is_professional_income_tax_l821_82130


namespace afternoon_emails_l821_82123

theorem afternoon_emails (A : ℕ) (five_morning_emails : ℕ) (two_more : five_morning_emails + 2 = A) : A = 7 :=
by
  sorry

end afternoon_emails_l821_82123


namespace find_b_value_l821_82191

theorem find_b_value (x y b : ℝ) (h1 : (7 * x + b * y) / (x - 2 * y) = 29) (h2 : x / (2 * y) = 3 / 2) : b = 8 :=
by
  sorry

end find_b_value_l821_82191


namespace incorrect_statements_l821_82104

-- Defining the first condition
def condition1 : Prop :=
  let a_sq := 169
  let b_sq := 144
  let c_sq := a_sq - b_sq
  let c_ := Real.sqrt c_sq
  let focal_points := [(0, c_), (0, -c_)]
  ¬((c_, 0) ∈ focal_points) ∧ ¬((-c_, 0) ∈ focal_points)

-- Defining the second condition
def condition2 : Prop :=
  let m := 1  -- Example choice since m is unspecified
  let a_sq := m^2 + 1
  let b_sq := m^2
  let c_sq := a_sq - b_sq
  let c_ := Real.sqrt c_sq
  let focal_points := [(0, c_), (0, -c_)]
  (0, 1) ∈ focal_points ∧ (0, -1) ∈ focal_points

-- Defining the third condition
def condition3 : Prop :=
  let a1_sq := 16
  let b1_sq := 7
  let c1_sq := a1_sq - b1_sq
  let c1_ := Real.sqrt c1_sq
  let focal_points1 := [(c1_, 0), (-c1_, 0)]
  
  let m := 10  -- Example choice since m > 0 is unspecified
  let a2_sq := m - 5
  let b2_sq := m + 4
  let c2_sq := a2_sq - b2_sq
  let focal_points2 := [(0, Real.sqrt c2_sq), (0, -Real.sqrt c2_sq)]
  
  ¬ (focal_points1 = focal_points2)

-- Defining the fourth condition
def condition4 : Prop :=
  let B := (-3, 0)
  let C := (3, 0)
  let BC := (C.1 - B.1, C.2 - B.2)
  let BC_dist := Real.sqrt (BC.1^2 + BC.2^2)
  let A_locus_eq := ∀ (x y : ℝ), x^2 / 36 + y^2 / 27 = 1
  2 * BC_dist = 12

-- Proof verification
theorem incorrect_statements : Prop :=
  condition1 ∧ condition3

end incorrect_statements_l821_82104


namespace average_of_remaining_two_l821_82166

theorem average_of_remaining_two (S S3 : ℚ) (h1 : S / 5 = 6) (h2 : S3 / 3 = 4) : (S - S3) / 2 = 9 :=
by
  sorry

end average_of_remaining_two_l821_82166


namespace inequality_holds_l821_82162

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
    a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
by
  sorry

end inequality_holds_l821_82162


namespace range_of_function_l821_82178

theorem range_of_function : 
  (∀ x, (Real.pi / 4) ≤ x ∧ x ≤ (Real.pi / 2) → 
   1 ≤ (Real.sin x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x ∧ 
    (Real.sin x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x ≤ 3 / 2) :=
sorry

end range_of_function_l821_82178


namespace range_of_a_maximum_of_z_l821_82158

-- Problem 1
theorem range_of_a (a b : ℝ) (h1 : a + 2 * b = 9) (h2 : |9 - 2 * b| + |a + 1| < 3) :
  -2 < a ∧ a < 1 :=
sorry

-- Problem 2
theorem maximum_of_z (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 9) :
  ∃ z, z = a * b^2 ∧ z ≤ 27 :=
sorry


end range_of_a_maximum_of_z_l821_82158


namespace no_solution_abs_eq_l821_82170

theorem no_solution_abs_eq : ∀ y : ℝ, |y - 2| ≠ |y - 1| + |y - 4| :=
by
  intros y
  sorry

end no_solution_abs_eq_l821_82170


namespace average_pages_per_hour_l821_82177

theorem average_pages_per_hour 
  (P : ℕ) (H : ℕ) (hP : P = 30000) (hH : H = 150) : 
  P / H = 200 := 
by 
  sorry

end average_pages_per_hour_l821_82177


namespace land_tax_calculation_l821_82139

theorem land_tax_calculation
  (area : ℝ)
  (value_per_acre : ℝ)
  (tax_rate : ℝ)
  (total_cadastral_value : ℝ := area * value_per_acre)
  (land_tax : ℝ := total_cadastral_value * tax_rate) :
  area = 15 → value_per_acre = 100000 → tax_rate = 0.003 → land_tax = 4500 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end land_tax_calculation_l821_82139


namespace total_clothes_l821_82100

-- Defining the conditions
def shirts := 12
def pants := 5 * shirts
def shorts := (1 / 4) * pants

-- Theorem to prove the total number of pieces of clothes
theorem total_clothes : shirts + pants + shorts = 87 := by
  -- using sorry to skip the proof
  sorry

end total_clothes_l821_82100


namespace midpoint_polar_coordinates_l821_82144

noncomputable def polar_midpoint :=
  let A := (10, 7 * Real.pi / 6)
  let B := (10, 11 * Real.pi / 6)
  let A_cartesian := (10 * Real.cos (7 * Real.pi / 6), 10 * Real.sin (7 * Real.pi / 6))
  let B_cartesian := (10 * Real.cos (11 * Real.pi / 6), 10 * Real.sin (11 * Real.pi / 6))
  let midpoint_cartesian := ((A_cartesian.1 + B_cartesian.1) / 2, (A_cartesian.2 + B_cartesian.2) / 2)
  let r := Real.sqrt (midpoint_cartesian.1 ^ 2 + midpoint_cartesian.2 ^ 2)
  let θ := if midpoint_cartesian.1 = 0 then 0 else Real.arctan (midpoint_cartesian.2 / midpoint_cartesian.1)
  (r, θ)

theorem midpoint_polar_coordinates :
  polar_midpoint = (5 * Real.sqrt 3, Real.pi) := by
  sorry

end midpoint_polar_coordinates_l821_82144


namespace relationship_among_a_ab_ab2_l821_82194

theorem relationship_among_a_ab_ab2 (a b : ℝ) (h_a : a < 0) (h_b1 : -1 < b) (h_b2 : b < 0) :
  a < a * b ∧ a * b < a * b^2 :=
by
  sorry

end relationship_among_a_ab_ab2_l821_82194


namespace problem1_problem2_l821_82175

theorem problem1 :
  ( (1/2) ^ (-2) - 0.01 ^ (-1) + (-(1 + 1/7)) ^ (0)) = -95 := by
  sorry

theorem problem2 (x : ℝ) :
  (x - 2) * (x + 1) - (x - 1) ^ 2 = x - 3 := by
  sorry

end problem1_problem2_l821_82175


namespace incorrect_statement_B_l821_82165

variable (a : Nat → Int) (S : Nat → Int)
variable (d : Int)

-- Given conditions
axiom S_5_lt_S_6 : S 5 < S 6
axiom S_6_eq_S_7 : S 6 = S 7
axiom S_7_gt_S_8 : S 7 > S 8
axiom S_n : ∀ n, S n = n * a n

-- Question to prove statement B is incorrect 
theorem incorrect_statement_B : ∃ (d : Int), (S 9 < S 5) :=
by 
  -- Proof goes here
  sorry

end incorrect_statement_B_l821_82165


namespace members_in_both_sets_are_23_l821_82176

variable (U A B : Finset ℕ)
variable (count_U count_A count_B count_neither count_both : ℕ)

theorem members_in_both_sets_are_23 (hU : count_U = 192)
    (hA : count_A = 107) (hB : count_B = 49) (hNeither : count_neither = 59) :
    count_both = 23 :=
by
  sorry

end members_in_both_sets_are_23_l821_82176


namespace find_a6_l821_82134

variable {a : ℕ → ℝ}

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
def given_condition (a : ℕ → ℝ) (d : ℝ) : Prop :=
  2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36

theorem find_a6 (d : ℝ) :
  is_arithmetic_sequence a d →
  given_condition a d →
  a 6 = 3 :=
by
  -- The proof would go here
  sorry

end find_a6_l821_82134


namespace cost_price_of_article_l821_82182

theorem cost_price_of_article (C : ℝ) (SP : ℝ) (C_new : ℝ) (SP_new : ℝ) :
  SP = 1.05 * C →
  C_new = 0.95 * C →
  SP_new = SP - 3 →
  SP_new = 1.045 * C →
  C = 600 :=
by
  intro h1 h2 h3 h4
  -- statement to be proved
  sorry

end cost_price_of_article_l821_82182


namespace single_bacteria_colony_days_to_limit_l821_82140

theorem single_bacteria_colony_days_to_limit (n : ℕ) (h : ∀ t : ℕ, t ≤ 21 → (2 ^ t = 2 * 2 ^ (t - 1))) : n = 22 :=
by
  sorry

end single_bacteria_colony_days_to_limit_l821_82140


namespace no_solution_for_conditions_l821_82152

theorem no_solution_for_conditions :
  ∀ (x y : ℝ), 0 < x → 0 < y → x * y = 2^15 → (Real.log x / Real.log 2) * (Real.log y / Real.log 2) = 60 → False :=
by
  intro x y x_pos y_pos h1 h2
  sorry

end no_solution_for_conditions_l821_82152


namespace train_speeds_l821_82120

noncomputable def c1 : ℝ := sorry  -- speed of the passenger train in km/min
noncomputable def c2 : ℝ := sorry  -- speed of the freight train in km/min
noncomputable def c3 : ℝ := sorry  -- speed of the express train in km/min

def conditions : Prop :=
  (5 / c1 + 5 / c2 = 15) ∧
  (5 / c2 + 5 / c3 = 11) ∧
  (c2 ≤ c1) ∧
  (c3 ≤ 2.5)

-- The theorem to be proved
theorem train_speeds :
  conditions →
  (40 / 60 ≤ c1 ∧ c1 ≤ 50 / 60) ∧ 
  (100 / 3 / 60 ≤ c2 ∧ c2 ≤ 40 / 60) ∧ 
  (600 / 7 / 60 ≤ c3 ∧ c3 ≤ 150 / 60) :=
sorry

end train_speeds_l821_82120


namespace fraction_of_male_gerbils_is_correct_l821_82181

def total_pets := 90
def total_gerbils := 66
def total_hamsters := total_pets - total_gerbils
def fraction_hamsters_male := 1/3
def total_males := 25
def male_hamsters := fraction_hamsters_male * total_hamsters
def male_gerbils := total_males - male_hamsters
def fraction_gerbils_male := male_gerbils / total_gerbils

theorem fraction_of_male_gerbils_is_correct : fraction_gerbils_male = 17 / 66 := by
  sorry

end fraction_of_male_gerbils_is_correct_l821_82181


namespace common_ratio_of_geometric_series_l821_82131

theorem common_ratio_of_geometric_series (a₁ q : ℝ) 
  (S_3 : ℝ) (S_2 : ℝ) 
  (hS3 : S_3 = a₁ * (1 - q^3) / (1 - q)) 
  (hS2 : S_2 = a₁ * (1 - q^2) / (1 - q)) 
  (h_ratio : S_3 / S_2 = 3 / 2) :
  q = 1 ∨ q = -1/2 :=
by
  -- Proof goes here.
  sorry

end common_ratio_of_geometric_series_l821_82131


namespace cindy_correct_answer_l821_82126

-- Define the conditions given in the problem
def x : ℤ := 272 -- Cindy's miscalculated number

-- The outcome of Cindy's incorrect operation
def cindy_incorrect (x : ℤ) : Prop := (x - 7) = 53 * 5

-- The outcome of Cindy's correct operation
def cindy_correct (x : ℤ) : ℤ := (x - 5) / 7

-- The main theorem to prove
theorem cindy_correct_answer : cindy_incorrect x → cindy_correct x = 38 :=
by
  sorry

end cindy_correct_answer_l821_82126


namespace number_2120_in_33rd_group_l821_82164

def last_number_in_group (n : ℕ) := 2 * n * (n + 1)

theorem number_2120_in_33rd_group :
  ∃ n, n = 33 ∧ (last_number_in_group (n - 1) < 2120) ∧ (2120 <= last_number_in_group n) :=
sorry

end number_2120_in_33rd_group_l821_82164


namespace exists_rectangle_with_perimeter_divisible_by_4_l821_82127

-- Define the problem conditions in Lean
def square_length : ℕ := 2015

-- Define what it means to cut the square into rectangles with integer sides
def is_rectangle (a b : ℕ) := 1 ≤ a ∧ a ≤ square_length ∧ 1 ≤ b ∧ b ≤ square_length

-- Define the perimeter condition
def perimeter_divisible_by_4 (a b : ℕ) := (2 * a + 2 * b) % 4 = 0

-- Final theorem statement
theorem exists_rectangle_with_perimeter_divisible_by_4 :
  ∃ (a b : ℕ), is_rectangle a b ∧ perimeter_divisible_by_4 a b :=
by {
  sorry -- The proof itself will be filled in to establish the theorem
}

end exists_rectangle_with_perimeter_divisible_by_4_l821_82127


namespace fraction_of_field_planted_l821_82159

theorem fraction_of_field_planted (AB AC : ℕ) (x : ℕ) (shortest_dist : ℕ) (hypotenuse : ℕ)
  (S : ℕ) (total_area : ℕ) (planted_area : ℕ) :
  AB = 5 ∧ AC = 12 ∧ hypotenuse = 13 ∧ shortest_dist = 2 ∧ x * x = S ∧ 
  total_area = 30 ∧ planted_area = total_area - S →
  (planted_area / total_area : ℚ) = 2951 / 3000 :=
by
  sorry

end fraction_of_field_planted_l821_82159


namespace minimum_value_frac_l821_82132

theorem minimum_value_frac (a b : ℝ) (h₁ : 2 * a - b + 2 * 0 = 0) 
  (h₂ : a > 0) (h₃ : b > 0) (h₄ : a + b = 1) : 
  (1 / a) + (1 / b) = 4 :=
sorry

end minimum_value_frac_l821_82132


namespace divides_iff_l821_82150

open Int

theorem divides_iff (n m : ℤ) : (9 ∣ (2 * n + 5 * m)) ↔ (9 ∣ (5 * n + 8 * m)) := 
sorry

end divides_iff_l821_82150


namespace smallest_positive_integer_x_for_cube_l821_82119

theorem smallest_positive_integer_x_for_cube (x : ℕ) (h1 : 1512 = 2^3 * 3^3 * 7) (h2 : ∀ n : ℕ, n > 0 → ∃ k : ℕ, 1512 * n = k^3) : x = 49 :=
sorry

end smallest_positive_integer_x_for_cube_l821_82119


namespace longest_segment_CD_l821_82156

theorem longest_segment_CD
  (ABD_angle : ℝ) (ADB_angle : ℝ) (BDC_angle : ℝ) (CBD_angle : ℝ)
  (angle_proof_ABD : ABD_angle = 50)
  (angle_proof_ADB : ADB_angle = 40)
  (angle_proof_BDC : BDC_angle = 35)
  (angle_proof_CBD : CBD_angle = 70) :
  true := 
by
  sorry

end longest_segment_CD_l821_82156


namespace solution_set_of_inequality_range_of_a_for_gx_zero_l821_82179

-- Define f(x) and g(x)
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 1) + abs (x + a)

def g (x : ℝ) (a : ℝ) : ℝ := f x a - abs (3 + a)

-- The first Lean statement
theorem solution_set_of_inequality (a : ℝ) (h : a = 3) :
  ∀ x : ℝ, f x a > 6 ↔ x < -4 ∨ (-3 < x ∧ x < 1) ∨ 2 < x := by
  sorry

-- The second Lean statement
theorem range_of_a_for_gx_zero (a : ℝ) :
  (∃ x : ℝ, g x a = 0) ↔ a ≥ -2 := by
  sorry

end solution_set_of_inequality_range_of_a_for_gx_zero_l821_82179


namespace johns_initial_money_l821_82142

theorem johns_initial_money (X : ℝ) 
  (h₁ : (1 / 2) * X + (1 / 3) * X + (1 / 10) * X + 10 = X) : X = 150 :=
sorry

end johns_initial_money_l821_82142


namespace tiles_needed_l821_82135

theorem tiles_needed (S : ℕ) (n : ℕ) (k : ℕ) (N : ℕ) (H1 : S = 18144) 
  (H2 : n * k^2 = S) (H3 : n = (N * (N + 1)) / 2) : n = 2016 := 
sorry

end tiles_needed_l821_82135


namespace cos_diff_to_product_l821_82124

open Real

theorem cos_diff_to_product (a b : ℝ) : 
  cos (a + b) - cos (a - b) = -2 * sin a * sin b := 
  sorry

end cos_diff_to_product_l821_82124


namespace probability_top_card_is_joker_l821_82189

def deck_size : ℕ := 54
def joker_count : ℕ := 2

theorem probability_top_card_is_joker :
  (joker_count : ℝ) / (deck_size : ℝ) = 1 / 27 :=
by
  sorry

end probability_top_card_is_joker_l821_82189


namespace commutative_binary_op_no_identity_element_associative_binary_op_l821_82115

def binary_op (x y : ℤ) : ℤ :=
  2 * (x + 2) * (y + 2) - 3

theorem commutative_binary_op (x y : ℤ) : binary_op x y = binary_op y x := by
  sorry

theorem no_identity_element (x e : ℤ) : ¬ (∀ x, binary_op x e = x) := by
  sorry

theorem associative_binary_op (x y z : ℤ) : (binary_op (binary_op x y) z = binary_op x (binary_op y z)) ∨ ¬ (binary_op (binary_op x y) z = binary_op x (binary_op y z)) := by
  sorry

end commutative_binary_op_no_identity_element_associative_binary_op_l821_82115


namespace cos_36_is_correct_l821_82192

noncomputable def cos_36_eq : Prop :=
  let b := Real.cos (Real.pi * 36 / 180)
  let a := Real.cos (Real.pi * 72 / 180)
  (a = 2 * b^2 - 1) ∧ (b = (1 + Real.sqrt 5) / 4)

theorem cos_36_is_correct : cos_36_eq :=
by sorry

end cos_36_is_correct_l821_82192


namespace interest_rate_l821_82173

noncomputable def simple_interest (P r t : ℝ) : ℝ := P * r * t / 100

noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r/100)^t - P

theorem interest_rate (P t : ℝ) (diff : ℝ) (r : ℝ) (h : P = 1000) (t_eq : t = 4) 
  (diff_eq : diff = 64.10) : 
  compound_interest P r t - simple_interest P r t = diff → r = 10 :=
by
  sorry

end interest_rate_l821_82173


namespace trapezoid_shorter_base_l821_82111

theorem trapezoid_shorter_base (a b : ℕ) (mid_segment : ℕ) (longer_base : ℕ) 
    (h1 : mid_segment = 5) (h2 : longer_base = 105) 
    (h3 : mid_segment = (longer_base - a) / 2) : 
  a = 95 := 
by
  sorry

end trapezoid_shorter_base_l821_82111


namespace decimalToFrac_l821_82161

theorem decimalToFrac : (145 / 100 : ℚ) = 29 / 20 := by
  sorry

end decimalToFrac_l821_82161


namespace mr_smith_children_l821_82180

noncomputable def gender_probability (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let equal_gender_ways := Nat.choose n (n / 2)
  let favourable_outcomes := total_outcomes - equal_gender_ways
  favourable_outcomes / total_outcomes

theorem mr_smith_children (n : ℕ) (h : n = 8) : 
  gender_probability n = 93 / 128 :=
by
  rw [h]
  sorry

end mr_smith_children_l821_82180


namespace jasmine_laps_per_afternoon_l821_82163

-- Defining the conditions
def swims_each_week (days_per_week : ℕ) := days_per_week = 5
def total_weeks := 5
def total_laps := 300

-- Main proof statement
theorem jasmine_laps_per_afternoon (d : ℕ) (l : ℕ) :
  swims_each_week d →
  total_weeks * d = 25 →
  total_laps = 300 →
  300 / 25 = l →
  l = 12 :=
by
  intros
  -- Skipping the proof
  sorry

end jasmine_laps_per_afternoon_l821_82163


namespace ratio_of_boys_l821_82138

theorem ratio_of_boys (p : ℚ) (h : p = (3/5) * (1 - p)) : p = 3 / 8 := by
  sorry

end ratio_of_boys_l821_82138


namespace area_ratio_of_octagon_l821_82129

theorem area_ratio_of_octagon (A : ℝ) (hA : 0 < A) :
  let triangle_ABJ_area := A / 8
  let triangle_ACE_area := A / 2
  triangle_ABJ_area / triangle_ACE_area = 1 / 4 := by
  sorry

end area_ratio_of_octagon_l821_82129


namespace rachel_plants_lamps_l821_82195

-- Define the conditions as types
def plants : Type := { fern1 : Prop // true } × { fern2 : Prop // true } × { cactus : Prop // true }
def lamps : Type := { yellow1 : Prop // true } × { yellow2 : Prop // true } × { blue1 : Prop // true } × { blue2 : Prop // true }

-- A function that counts the distribution of plants under lamps
noncomputable def count_ways (p : plants) (l : lamps) : ℕ :=
  -- Here we should define the function that counts the number of configurations, 
  -- but since we are only defining the problem here we'll skip this part.
  sorry

-- The statement to prove
theorem rachel_plants_lamps :
  ∀ (p : plants) (l : lamps), count_ways p l = 14 :=
by
  sorry

end rachel_plants_lamps_l821_82195


namespace magnitude_of_a_l821_82190

variable (a b : EuclideanSpace ℝ (Fin 2))
variable (theta : ℝ)
variable (hθ : theta = π / 3)
variable (hb : ‖b‖ = 1)
variable (hab : ‖a + 2 • b‖ = 2 * sqrt 3)

theorem magnitude_of_a :
  ‖a‖ = 2 :=
by
  sorry

end magnitude_of_a_l821_82190


namespace stream_speed_l821_82110

def upstream_time : ℝ := 4  -- time in hours
def downstream_time : ℝ := 4  -- time in hours
def upstream_distance : ℝ := 32  -- distance in km
def downstream_distance : ℝ := 72  -- distance in km

-- Speed equations based on given conditions
def effective_speed_upstream (vj vs : ℝ) : Prop := vj - vs = upstream_distance / upstream_time
def effective_speed_downstream (vj vs : ℝ) : Prop := vj + vs = downstream_distance / downstream_time

theorem stream_speed (vj vs : ℝ)  
  (h1 : effective_speed_upstream vj vs)
  (h2 : effective_speed_downstream vj vs) : 
  vs = 5 := sorry

end stream_speed_l821_82110


namespace positive_number_condition_l821_82145

theorem positive_number_condition (y : ℝ) (h: 0.04 * y = 16): y = 400 := 
by sorry

end positive_number_condition_l821_82145


namespace two_legged_birds_count_l821_82113

def count_birds (b m i : ℕ) : Prop :=
  b + m + i = 300 ∧ 2 * b + 4 * m + 6 * i = 680 → b = 280

theorem two_legged_birds_count : ∃ b m i : ℕ, count_birds b m i :=
by
  have h1 : count_birds 280 0 20 := sorry
  exact ⟨280, 0, 20, h1⟩

end two_legged_birds_count_l821_82113


namespace eq_factorial_sum_l821_82188

theorem eq_factorial_sum (k l m n : ℕ) (hk : k > 0) (hl : l > 0) (hm : m > 0) (hn : n > 0) :
  (1 / (Nat.factorial k : ℝ) + 1 / (Nat.factorial l : ℝ) + 1 / (Nat.factorial m : ℝ) = 1 / (Nat.factorial n : ℝ))
  ↔ (k = 3 ∧ l = 3 ∧ m = 3 ∧ n = 2) :=
by
  sorry

end eq_factorial_sum_l821_82188


namespace length_of_ST_l821_82143

theorem length_of_ST (PQ PS : ℝ) (ST : ℝ) (hPQ : PQ = 8) (hPS : PS = 7) 
  (h_area_eq : (1 / 2) * PQ * (PS * (1 / PS) * 8) = PQ * PS) : 
  ST = 2 * Real.sqrt 65 := 
by
  -- proof steps (to be written)
  sorry

end length_of_ST_l821_82143


namespace count_integers_satisfying_sqrt_condition_l821_82148

theorem count_integers_satisfying_sqrt_condition :
  ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℕ), (3 < Real.sqrt x ∧ Real.sqrt x < 5) → (9 < x ∧ x < 25) :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l821_82148


namespace cost_of_socks_l821_82157

theorem cost_of_socks (cost_shirt_no_discount cost_pants_no_discount cost_shirt_discounted cost_pants_discounted cost_socks_discounted total_savings team_size socks_cost_no_discount : ℝ) 
    (h1 : cost_shirt_no_discount = 7.5)
    (h2 : cost_pants_no_discount = 15)
    (h3 : cost_shirt_discounted = 6.75)
    (h4 : cost_pants_discounted = 13.5)
    (h5 : cost_socks_discounted = 3.75)
    (h6 : total_savings = 36)
    (h7 : team_size = 12)
    (h8 : 12 * (7.5 + 15 + socks_cost_no_discount) - 12 * (6.75 + 13.5 + 3.75) = 36)
    : socks_cost_no_discount = 4.5 :=
by
  sorry

end cost_of_socks_l821_82157


namespace function_value_corresponds_to_multiple_independent_variables_l821_82128

theorem function_value_corresponds_to_multiple_independent_variables
  {α β : Type*} (f : α → β) :
  ∃ (b : β), ∃ (a1 a2 : α), a1 ≠ a2 ∧ f a1 = b ∧ f a2 = b :=
sorry

end function_value_corresponds_to_multiple_independent_variables_l821_82128


namespace num_combinations_L_shape_l821_82167

theorem num_combinations_L_shape (n : ℕ) (k : ℕ) (grid_size : ℕ) (L_shape_blocks : ℕ) 
  (h1 : n = 6) (h2 : k = 4) (h3 : grid_size = 36) (h4 : L_shape_blocks = 4) : 
  ∃ (total_combinations : ℕ), total_combinations = 1800 := by
  sorry

end num_combinations_L_shape_l821_82167


namespace harmonic_mean_pairs_l821_82107

theorem harmonic_mean_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) 
    (hmean : (2 * x * y) / (x + y) = 2^30) :
    (∃! n, n = 29) :=
by
  sorry

end harmonic_mean_pairs_l821_82107


namespace overall_gain_percentage_l821_82187

theorem overall_gain_percentage (cost_A cost_B cost_C sp_A sp_B sp_C : ℕ)
  (hA : cost_A = 1000)
  (hB : cost_B = 3000)
  (hC : cost_C = 6000)
  (hsA : sp_A = 2000)
  (hsB : sp_B = 4500)
  (hsC : sp_C = 8000) :
  ((sp_A + sp_B + sp_C - (cost_A + cost_B + cost_C) : ℝ) / (cost_A + cost_B + cost_C) * 100) = 45 :=
by sorry

end overall_gain_percentage_l821_82187


namespace evaluate_g_at_neg2_l821_82197

-- Definition of the polynomial g
def g (x : ℝ) : ℝ := 3 * x^5 - 20 * x^4 + 40 * x^3 - 25 * x^2 - 75 * x + 90

-- Statement to prove using the condition
theorem evaluate_g_at_neg2 : g (-2) = -596 := 
by 
   sorry

end evaluate_g_at_neg2_l821_82197


namespace combined_operation_l821_82186

def f (x : ℚ) := (3 / 4) * x
def g (x : ℚ) := (5 / 3) * x

theorem combined_operation (x : ℚ) : g (f x) = (5 / 4) * x :=
by
    unfold f g
    sorry

end combined_operation_l821_82186


namespace sphere_volume_l821_82118

theorem sphere_volume (A : ℝ) (d : ℝ) (V : ℝ) : 
    (A = 2 * Real.pi) →  -- Cross-sectional area is 2π cm²
    (d = 1) →            -- Distance from center to cross-section is 1 cm
    (V = 4 * Real.sqrt 3 * Real.pi) :=  -- Volume of sphere is 4√3 π cm³
by 
  intros hA hd
  sorry

end sphere_volume_l821_82118
