import Mathlib

namespace car_distance_in_13_hours_l2677_267779

/-- Represents the driving characteristics of the car -/
structure Car where
  speed : ℕ            -- Speed in miles per hour
  drive_time : ℕ       -- Continuous driving time in hours
  cool_time : ℕ        -- Cooling time in hours

/-- Calculates the total distance a car can travel in a given time -/
def total_distance (c : Car) (total_time : ℕ) : ℕ :=
  sorry

/-- Theorem stating the total distance the car can travel in 13 hours -/
theorem car_distance_in_13_hours (c : Car) 
  (h1 : c.speed = 8)
  (h2 : c.drive_time = 5)
  (h3 : c.cool_time = 1) :
  total_distance c 13 = 88 :=
sorry

end car_distance_in_13_hours_l2677_267779


namespace senior_class_college_attendance_l2677_267711

theorem senior_class_college_attendance 
  (total_boys : ℕ) 
  (total_girls : ℕ) 
  (boys_not_attended_percent : ℚ) 
  (total_attended_percent : ℚ) :
  total_boys = 300 →
  total_girls = 240 →
  boys_not_attended_percent = 30 / 100 →
  total_attended_percent = 70 / 100 →
  (total_girls - (total_attended_percent * (total_boys + total_girls) - 
    (1 - boys_not_attended_percent) * total_boys)) / total_girls = 30 / 100 := by
  sorry

end senior_class_college_attendance_l2677_267711


namespace fourth_person_height_l2677_267766

/-- Given four people with heights in increasing order, prove that the fourth person is 82 inches tall -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- Heights are in increasing order
  h₂ - h₁ = 2 →                 -- Difference between 1st and 2nd person
  h₃ - h₂ = 2 →                 -- Difference between 2nd and 3rd person
  h₄ - h₃ = 6 →                 -- Difference between 3rd and 4th person
  (h₁ + h₂ + h₃ + h₄) / 4 = 76  -- Average height is 76 inches
  → h₄ = 82 := by               -- The fourth person is 82 inches tall
sorry

end fourth_person_height_l2677_267766


namespace unique_coefficient_exists_l2677_267749

theorem unique_coefficient_exists (x y : ℝ) 
  (eq1 : 4 * x + y = 8) 
  (eq2 : 3 * x - 4 * y = 5) : 
  ∃! a : ℝ, a * x - 3 * y = 23 := by
sorry

end unique_coefficient_exists_l2677_267749


namespace family_reunion_weight_gain_l2677_267771

/-- The total weight gain of three family members at a reunion -/
def total_weight_gain (orlando_gain jose_gain fernando_gain : ℝ) : ℝ :=
  orlando_gain + jose_gain + fernando_gain

theorem family_reunion_weight_gain :
  ∃ (orlando_gain jose_gain fernando_gain : ℝ),
    orlando_gain = 5 ∧
    jose_gain = 2 * orlando_gain + 2 ∧
    fernando_gain = (1/2 : ℝ) * jose_gain - 3 ∧
    total_weight_gain orlando_gain jose_gain fernando_gain = 20 := by
  sorry

end family_reunion_weight_gain_l2677_267771


namespace tan_two_beta_l2677_267783

theorem tan_two_beta (α β : Real) 
  (h1 : Real.tan (α + β) = 1) 
  (h2 : Real.tan (α - β) = 7) : 
  Real.tan (2 * β) = -3/4 := by sorry

end tan_two_beta_l2677_267783


namespace walking_problem_l2677_267761

/-- The correct system of equations for the walking problem -/
theorem walking_problem (x y : ℝ) : 
  (∃ (total_distance : ℝ) (meeting_time : ℝ) (additional_time : ℝ),
    total_distance = 3 ∧ 
    meeting_time = 20/60 ∧ 
    additional_time = 10/60 ∧ 
    meeting_time * (x + y) = total_distance ∧
    (total_distance - (meeting_time + additional_time) * x) = 2 * (total_distance - (meeting_time + additional_time) * y)) ↔ 
  ((20/60 * x + 20/60 * y = 3) ∧ (3 - 30/60 * x = 2 * (3 - 30/60 * y))) :=
by sorry

end walking_problem_l2677_267761


namespace alice_second_test_study_time_l2677_267770

/-- Represents the relationship between study time and test score -/
def study_score_product (study_time : ℝ) (score : ℝ) : ℝ := study_time * score

/-- Alice's first test data -/
def first_test_time : ℝ := 2
def first_test_score : ℝ := 60

/-- Target average score -/
def target_average : ℝ := 75

/-- Theorem: Alice needs to study 4/3 hours for her second test -/
theorem alice_second_test_study_time :
  ∃ (second_test_time : ℝ),
    second_test_time > 0 ∧
    study_score_product first_test_time first_test_score = study_score_product second_test_time ((target_average * 2) - first_test_score) ∧
    second_test_time = 4/3 :=
by sorry

end alice_second_test_study_time_l2677_267770


namespace arithmetic_sequence_30th_term_l2677_267724

/-- An arithmetic sequence with first term 3 and ninth term 27 has its thirtieth term equal to 90 -/
theorem arithmetic_sequence_30th_term : ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 3 →                            -- first term is 3
  a 9 = 27 →                           -- ninth term is 27
  a 30 = 90 :=                         -- thirtieth term is 90
by sorry

end arithmetic_sequence_30th_term_l2677_267724


namespace complex_product_real_l2677_267700

theorem complex_product_real (b : ℝ) : 
  let z₁ : ℂ := 1 + I
  let z₂ : ℂ := 2 + b * I
  (z₁ * z₂).im = 0 → b = -2 := by
  sorry

end complex_product_real_l2677_267700


namespace units_digit_sum_of_powers_l2677_267717

theorem units_digit_sum_of_powers : ∃ n : ℕ, n < 10 ∧ (35^87 + 93^49) % 10 = n :=
  by sorry

end units_digit_sum_of_powers_l2677_267717


namespace imaginary_unit_power_l2677_267728

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2013 = i := by
  sorry

end imaginary_unit_power_l2677_267728


namespace continuity_at_two_l2677_267748

noncomputable def f (x : ℝ) : ℝ := (x^4 - 16) / (x^3 - 2*x^2)

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |f x - 8| < ε :=
sorry

end continuity_at_two_l2677_267748


namespace expression_simplification_l2677_267725

theorem expression_simplification :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 10) / 4) = 13 := by
  sorry

end expression_simplification_l2677_267725


namespace ashas_initial_savings_l2677_267746

def borrowed_money : ℕ := 20 + 40 + 30
def gift_money : ℕ := 70
def remaining_money : ℕ := 65
def spending_fraction : ℚ := 3 / 4

theorem ashas_initial_savings :
  ∃ (initial_savings : ℕ),
    let total_money := initial_savings + borrowed_money + gift_money
    (total_money : ℚ) * (1 - spending_fraction) = remaining_money ∧
    initial_savings = 100 := by
  sorry

end ashas_initial_savings_l2677_267746


namespace yellow_ball_probability_l2677_267709

def container_X : ℕ × ℕ := (7, 3)  -- (blue balls, yellow balls)
def container_Y : ℕ × ℕ := (5, 5)
def container_Z : ℕ × ℕ := (8, 2)

def total_balls (c : ℕ × ℕ) : ℕ := c.1 + c.2

def prob_yellow (c : ℕ × ℕ) : ℚ := c.2 / (total_balls c)

def prob_container : ℚ := 1 / 3

theorem yellow_ball_probability :
  prob_container * prob_yellow container_X +
  prob_container * prob_yellow container_Y +
  prob_container * prob_yellow container_Z = 1 / 3 := by
  sorry

end yellow_ball_probability_l2677_267709


namespace train_platform_passing_time_l2677_267760

/-- Calculates the time for a train to pass a platform -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (tree_passing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1100) 
  (h2 : tree_passing_time = 110) 
  (h3 : platform_length = 700) : 
  (train_length + platform_length) / (train_length / tree_passing_time) = 180 :=
sorry

end train_platform_passing_time_l2677_267760


namespace correct_number_of_vans_l2677_267753

/-- The number of vans taken on a field trip -/
def number_of_vans : ℕ := 2

/-- The total number of people on the field trip -/
def total_people : ℕ := 76

/-- The number of buses taken on the field trip -/
def number_of_buses : ℕ := 3

/-- The number of people each bus can hold -/
def people_per_bus : ℕ := 20

/-- The number of people each van can hold -/
def people_per_van : ℕ := 8

/-- Theorem stating that the number of vans is correct given the conditions -/
theorem correct_number_of_vans : 
  number_of_vans * people_per_van + number_of_buses * people_per_bus = total_people :=
by sorry

end correct_number_of_vans_l2677_267753


namespace no_positive_solutions_l2677_267712

theorem no_positive_solutions :
  ¬∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x^3 + y^3 + z^3 = x + y + z ∧
  x^2 + y^2 + z^2 = x*y*z :=
by sorry

end no_positive_solutions_l2677_267712


namespace derivative_x_exp_x_l2677_267743

theorem derivative_x_exp_x (x : ℝ) : deriv (fun x => x * Real.exp x) x = (1 + x) * Real.exp x := by
  sorry

end derivative_x_exp_x_l2677_267743


namespace coastline_scientific_notation_l2677_267756

theorem coastline_scientific_notation : 
  37515000 = 3.7515 * (10 : ℝ)^7 := by
  sorry

end coastline_scientific_notation_l2677_267756


namespace sine_of_intersection_angle_l2677_267730

/-- The sine of the angle formed by a point on y = 3x and x^2 + y^2 = 1 in the first quadrant -/
theorem sine_of_intersection_angle (x y : ℝ) (h1 : y = 3 * x) (h2 : x^2 + y^2 = 1) 
  (h3 : x > 0) (h4 : y > 0) : 
  Real.sin (Real.arctan (y / x)) = 3 * Real.sqrt 10 / 10 := by
  sorry

end sine_of_intersection_angle_l2677_267730


namespace problem_solution_l2677_267714

theorem problem_solution :
  ∀ n : ℤ, 3 ≤ n ∧ n ≤ 9 ∧ n ≡ 6557 [ZMOD 7] → n = 5 := by
  sorry

end problem_solution_l2677_267714


namespace corner_cut_pentagon_area_corner_cut_pentagon_area_is_correct_l2677_267793

/-- Represents a pentagon formed by cutting a triangular corner from a rectangle -/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {17, 23, 26, 29, 35}

/-- The area of a CornerCutPentagon is 895 -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : ℕ :=
  895

/-- The area of a CornerCutPentagon is correct -/
theorem corner_cut_pentagon_area_is_correct (p : CornerCutPentagon) :
  corner_cut_pentagon_area p = 895 := by
  sorry

end corner_cut_pentagon_area_corner_cut_pentagon_area_is_correct_l2677_267793


namespace triangle_side_length_l2677_267747

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  c = 3 →
  C = π / 3 →
  a = 2 * b →
  b = Real.sqrt 3 :=
by
  sorry

end triangle_side_length_l2677_267747


namespace new_girl_weight_l2677_267740

/-- The weight of the new girl given the conditions of the problem -/
def weight_of_new_girl (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * weight_increase

/-- Theorem stating the weight of the new girl under the given conditions -/
theorem new_girl_weight :
  weight_of_new_girl 8 3 70 = 94 := by
  sorry

end new_girl_weight_l2677_267740


namespace angle_around_point_l2677_267794

/-- Given a point in a plane with four angles around it, where three of the angles are equal (x°) and the fourth is 140°, prove that x = 220/3. -/
theorem angle_around_point (x : ℚ) : 
  (3 * x + 140 = 360) → x = 220 / 3 := by
  sorry

end angle_around_point_l2677_267794


namespace corrected_mean_l2677_267713

theorem corrected_mean (n : ℕ) (incorrect_mean : ℝ) (incorrect_value : ℝ) (correct_value : ℝ) :
  n = 50 →
  incorrect_mean = 36 →
  incorrect_value = 21 →
  correct_value = 48 →
  (n : ℝ) * incorrect_mean - incorrect_value + correct_value = 36.54 * n :=
by sorry

end corrected_mean_l2677_267713


namespace set_equality_proof_l2677_267782

theorem set_equality_proof (A B : Set α) (h : A ∩ B = A) : A ∪ B = B := by
  sorry

end set_equality_proof_l2677_267782


namespace f_2015_5_l2677_267707

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_2015_5 (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 4)
  (h_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)) :
  f 2015.5 = -1/2 := by
  sorry

end f_2015_5_l2677_267707


namespace A_and_D_independent_l2677_267764

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define event A
def A : Set Ω := {ω | ω.1 = 0}

-- Define event D
def D : Set Ω := {ω | ω.1.val + ω.2.val + 2 = 7}

-- Theorem statement
theorem A_and_D_independent :
  P (A ∩ D) = P A * P D := by sorry

end A_and_D_independent_l2677_267764


namespace ruble_exchange_impossible_l2677_267752

theorem ruble_exchange_impossible : ¬ ∃ (x y z : ℕ), 
  x + y + z = 10 ∧ x + 3*y + 5*z = 25 := by sorry

end ruble_exchange_impossible_l2677_267752


namespace triangle_area_l2677_267701

theorem triangle_area (p A B : Real) (h_positive : p > 0) (h_angles : 0 < A ∧ 0 < B ∧ A + B < π) : 
  let C := π - A - B
  let S := (2 * p^2 * Real.sin A * Real.sin B * Real.sin C) / (Real.sin A + Real.sin B + Real.sin C)^2
  S > 0 ∧ S < p^2 := by sorry

end triangle_area_l2677_267701


namespace special_form_not_perfect_square_l2677_267704

/-- A function that returns true if the input number has at least three digits,
    all digits except the first and last are zeros, and the first and last digits are non-zeros -/
def has_special_form (n : ℕ) : Prop :=
  n ≥ 100 ∧
  ∃ (d b : ℕ) (k : ℕ), 
    d ≠ 0 ∧ b ≠ 0 ∧ 
    n = d * 10^k + b ∧
    k ≥ 1

theorem special_form_not_perfect_square (n : ℕ) :
  has_special_form n → ¬ ∃ (m : ℕ), n = m^2 :=
by sorry

end special_form_not_perfect_square_l2677_267704


namespace system_solution_l2677_267787

theorem system_solution (x y z w : ℝ) : 
  (x - y + z - w = 2 ∧
   x^2 - y^2 + z^2 - w^2 = 6 ∧
   x^3 - y^3 + z^3 - w^3 = 20 ∧
   x^4 - y^4 + z^4 - w^4 = 60) ↔ 
  ((x = 1 ∧ y = 2 ∧ z = 3 ∧ w = 0) ∨
   (x = 1 ∧ y = 0 ∧ z = 3 ∧ w = 2) ∨
   (x = 3 ∧ y = 2 ∧ z = 1 ∧ w = 0) ∨
   (x = 3 ∧ y = 0 ∧ z = 1 ∧ w = 2)) :=
by sorry

end system_solution_l2677_267787


namespace circumscribing_circle_diameter_l2677_267796

/-- The diameter of a circle circumscribing six equal, mutually tangent circles -/
theorem circumscribing_circle_diameter (r : ℝ) (h : r = 4) : 
  let small_circle_radius : ℝ := r
  let small_circles_count : ℕ := 6
  let large_circle_diameter : ℝ := 2 * (2 * small_circle_radius + small_circle_radius)
  large_circle_diameter = 24 := by sorry

end circumscribing_circle_diameter_l2677_267796


namespace track_length_is_300_l2677_267759

-- Define the track length
def track_length : ℝ := sorry

-- Define Brenda's distance to first meeting
def brenda_first_meeting : ℝ := 120

-- Define Sally's additional distance to second meeting
def sally_additional : ℝ := 180

-- Theorem statement
theorem track_length_is_300 :
  -- Conditions
  (brenda_first_meeting + (track_length - brenda_first_meeting) = track_length) ∧
  (brenda_first_meeting + brenda_first_meeting = 
   track_length - brenda_first_meeting + sally_additional) →
  -- Conclusion
  track_length = 300 := by
  sorry

end track_length_is_300_l2677_267759


namespace number_difference_l2677_267780

theorem number_difference (a b : ℕ) : 
  a + b = 26832 → 
  b % 10 = 0 → 
  a = b / 10 + 4 → 
  b - a = 21938 := by
sorry

end number_difference_l2677_267780


namespace lcm_of_ratio_numbers_l2677_267774

theorem lcm_of_ratio_numbers (a b : ℕ) (h1 : a = 20) (h2 : 5 * b = 4 * a) : 
  Nat.lcm a b = 80 := by
  sorry

end lcm_of_ratio_numbers_l2677_267774


namespace polynomial_integral_theorem_l2677_267744

/-- A polynomial of degree at most 2 -/
def Polynomial2 := ℝ → ℝ

/-- The definite integral of a polynomial from a to b -/
noncomputable def integral (f : Polynomial2) (a b : ℝ) : ℝ := sorry

/-- The condition that the integrals sum to zero -/
def integralCondition (f : Polynomial2) (p q r : ℝ) : Prop :=
  integral f (-1) p - integral f p q + integral f q r - integral f r 1 = 0

theorem polynomial_integral_theorem :
  ∃! (p q r : ℝ), 
    -1 < p ∧ p < q ∧ q < r ∧ r < 1 ∧
    (∀ f : Polynomial2, integralCondition f p q r) ∧
    p = 1 / Real.sqrt 2 ∧ q = 0 ∧ r = -1 / Real.sqrt 2 := by sorry

end polynomial_integral_theorem_l2677_267744


namespace repeating_decimal_fraction_l2677_267734

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  whole : ℤ
  repeating : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def to_rat (d : RepeatingDecimal) : ℚ :=
  d.whole + (d.repeating : ℚ) / (999 : ℚ)

/-- The repeating decimal 0.428428... -/
def d1 : RepeatingDecimal := ⟨0, 428⟩

/-- The repeating decimal 2.857857... -/
def d2 : RepeatingDecimal := ⟨2, 857⟩

theorem repeating_decimal_fraction :
  (to_rat d1) / (to_rat d2) = 1 / 6 := by sorry

end repeating_decimal_fraction_l2677_267734


namespace sqrt_nested_square_l2677_267745

theorem sqrt_nested_square : (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^2 = 4 := by
  sorry

end sqrt_nested_square_l2677_267745


namespace solution_sum_l2677_267732

theorem solution_sum (P q : ℝ) : 
  (2^2 - P*2 + 6 = 0) → (2^2 + 6*2 - q = 0) → P + q = 21 := by
  sorry

end solution_sum_l2677_267732


namespace fraction_ratio_equality_l2677_267755

theorem fraction_ratio_equality : ∃ x : ℚ, (x / (2/6) = (3/4) / (1/2)) ∧ (x = 2/9) := by
  sorry

end fraction_ratio_equality_l2677_267755


namespace diophantine_equation_solutions_l2677_267777

theorem diophantine_equation_solutions (a b c : ℤ) :
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1 ↔
  ((a = 3 ∧ b = 3 ∧ c = 3) ∨
   (a = 2 ∧ b = 3 ∧ c = 6) ∨
   (a = 2 ∧ b = 4 ∧ c = 4) ∨
   (∃ t : ℤ, a = 1 ∧ b = t ∧ c = -t)) :=
by sorry

end diophantine_equation_solutions_l2677_267777


namespace cos_pi_minus_alpha_l2677_267718

theorem cos_pi_minus_alpha (α : Real) (h : Real.sin (π / 2 + α) = 1 / 7) :
  Real.cos (π - α) = -1 / 7 := by
  sorry

end cos_pi_minus_alpha_l2677_267718


namespace johns_next_birthday_l2677_267754

/-- Represents the ages of John, Emily, and Lucas -/
structure Ages where
  john : ℝ
  emily : ℝ
  lucas : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.john = 1.25 * ages.emily ∧
  ages.emily = 0.7 * ages.lucas ∧
  ages.john + ages.emily + ages.lucas = 32

/-- The main theorem -/
theorem johns_next_birthday (ages : Ages) 
  (h : satisfies_conditions ages) : 
  ⌈ages.john⌉ = 11 := by
  sorry


end johns_next_birthday_l2677_267754


namespace tan_two_simplification_l2677_267729

theorem tan_two_simplification (x : ℝ) (h : Real.tan x = 2) :
  (2 * Real.sin x + Real.cos x) / (2 * Real.sin x - Real.cos x) = 5/3 := by
  sorry

end tan_two_simplification_l2677_267729


namespace number_added_to_x_l2677_267786

theorem number_added_to_x (x : ℝ) (some_number : ℝ) : 
  x + some_number = 2 → x = 1 → some_number = 1 := by
sorry

end number_added_to_x_l2677_267786


namespace max_tiles_on_floor_l2677_267742

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit on the floor -/
def maxTiles (floor : Dimensions) (tile : Dimensions) : ℕ :=
  let horizontal := (floor.length / tile.length) * (floor.width / tile.width)
  let vertical := (floor.length / tile.width) * (floor.width / tile.length)
  max horizontal vertical

/-- Theorem stating the maximum number of tiles that can be accommodated on the floor -/
theorem max_tiles_on_floor :
  let floor : Dimensions := ⟨390, 150⟩
  let tile : Dimensions := ⟨65, 25⟩
  maxTiles floor tile = 36 := by
  sorry

#eval maxTiles ⟨390, 150⟩ ⟨65, 25⟩

end max_tiles_on_floor_l2677_267742


namespace matrix_inverse_proof_l2677_267735

/-- Given a 2x2 matrix M, prove that its inverse is correct. -/
theorem matrix_inverse_proof (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M = ![![1, 0], ![1, 1]]) : 
  M⁻¹ = ![![1, 0], ![-1, 1]] := by
  sorry

end matrix_inverse_proof_l2677_267735


namespace morning_snowfall_l2677_267710

/-- Given the total snowfall and afternoon snowfall in Yardley, 
    prove that the morning snowfall is the difference between them. -/
theorem morning_snowfall (total : ℝ) (afternoon : ℝ) 
  (h1 : total = 0.625) (h2 : afternoon = 0.5) : 
  total - afternoon = 0.125 := by
  sorry

end morning_snowfall_l2677_267710


namespace sophie_total_spend_l2677_267723

def cupcake_quantity : ℕ := 5
def cupcake_price : ℚ := 2

def doughnut_quantity : ℕ := 6
def doughnut_price : ℚ := 1

def apple_pie_quantity : ℕ := 4
def apple_pie_price : ℚ := 2

def cookie_quantity : ℕ := 15
def cookie_price : ℚ := 0.6

theorem sophie_total_spend :
  (cupcake_quantity : ℚ) * cupcake_price +
  (doughnut_quantity : ℚ) * doughnut_price +
  (apple_pie_quantity : ℚ) * apple_pie_price +
  (cookie_quantity : ℚ) * cookie_price = 33 := by
  sorry

end sophie_total_spend_l2677_267723


namespace max_ratio_ab_l2677_267731

theorem max_ratio_ab (a b : ℕ+) (h : (a : ℚ) / ((a : ℚ) - 2) = ((b : ℚ) + 2021) / ((b : ℚ) + 2008)) :
  (a : ℚ) / (b : ℚ) ≤ 312 / 7 := by
sorry

end max_ratio_ab_l2677_267731


namespace divisors_of_3b_plus_18_l2677_267795

theorem divisors_of_3b_plus_18 (a b : ℤ) (h : 4 * b = 10 - 2 * a) :
  (∀ d : ℤ, d ∈ ({1, 2, 3, 6} : Set ℤ) → d ∣ (3 * b + 18)) ∧
  (∃ a b : ℤ, 4 * b = 10 - 2 * a ∧ (¬(4 ∣ (3 * b + 18)) ∨ ¬(5 ∣ (3 * b + 18)) ∨
                                   ¬(7 ∣ (3 * b + 18)) ∨ ¬(8 ∣ (3 * b + 18)))) :=
by sorry

end divisors_of_3b_plus_18_l2677_267795


namespace field_trip_capacity_l2677_267769

theorem field_trip_capacity (seats_per_bus : ℕ) (num_buses : ℕ) : 
  let max_students := seats_per_bus * num_buses
  seats_per_bus = 60 → num_buses = 3 → max_students = 180 := by
sorry

end field_trip_capacity_l2677_267769


namespace min_value_of_expression_l2677_267775

theorem min_value_of_expression (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z) ≥ 36 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + c = 1 ∧ (1 / a + 4 / b + 9 / c = 36) :=
by sorry

end min_value_of_expression_l2677_267775


namespace fixed_point_of_line_l2677_267767

/-- The fixed point of the line (2k-1)x-(k+3)y-(k-11)=0 for all real k -/
theorem fixed_point_of_line (k : ℝ) : (2*k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end fixed_point_of_line_l2677_267767


namespace f_min_max_l2677_267720

-- Define the function f
def f (x y z : ℝ) : ℝ := x * y + y * z + z * x - 3 * x * y * z

-- State the theorem
theorem f_min_max :
  ∀ x y z : ℝ,
  x ≥ 0 → y ≥ 0 → z ≥ 0 →
  x + y + z = 1 →
  (0 ≤ f x y z) ∧ (f x y z ≤ 1/4) ∧
  (∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ f a b c = 0) ∧
  (∃ d e g : ℝ, d ≥ 0 ∧ e ≥ 0 ∧ g ≥ 0 ∧ d + e + g = 1 ∧ f d e g = 1/4) :=
by sorry

end f_min_max_l2677_267720


namespace existence_of_special_multiple_l2677_267726

theorem existence_of_special_multiple (n : ℕ+) : 
  ∃ m : ℕ+, (m.val % n.val = 0) ∧ 
             (m.val ≤ n.val^2) ∧ 
             (∃ d : Fin 10, ∀ k : ℕ, (m.val / 10^k % 10) ≠ d.val) := by
  sorry

end existence_of_special_multiple_l2677_267726


namespace min_jugs_to_fill_container_l2677_267708

/-- The capacity of a regular water jug in milliliters -/
def regular_jug_capacity : ℕ := 300

/-- The capacity of a giant water container in milliliters -/
def giant_container_capacity : ℕ := 1800

/-- The minimum number of regular jugs needed to fill a giant container -/
def min_jugs_needed : ℕ := giant_container_capacity / regular_jug_capacity

theorem min_jugs_to_fill_container : min_jugs_needed = 6 := by
  sorry

end min_jugs_to_fill_container_l2677_267708


namespace land_plot_side_length_l2677_267762

theorem land_plot_side_length (area : ℝ) (h : area = Real.sqrt 1600) :
  Real.sqrt area = 40 := by
  sorry

end land_plot_side_length_l2677_267762


namespace reservoir_refill_rate_l2677_267705

theorem reservoir_refill_rate 
  (V : ℝ) (R : ℝ) 
  (h1 : V - 90 * (40000 - R) = 0) 
  (h2 : V - 60 * (32000 - R) = 0) : 
  R = 56000 := by
sorry

end reservoir_refill_rate_l2677_267705


namespace quadratic_max_min_difference_l2677_267776

/-- Given a quadratic function f(x) = -x^2 + 10x + 9 defined on the interval [2, a/9],
    where a/9 ≥ 8, the difference between its maximum and minimum values is 9. -/
theorem quadratic_max_min_difference (a : ℝ) (h : a / 9 ≥ 8) :
  let f : ℝ → ℝ := λ x ↦ -x^2 + 10*x + 9
  let max_val := (⨆ x ∈ Set.Icc 2 (a / 9), f x)
  let min_val := (⨅ x ∈ Set.Icc 2 (a / 9), f x)
  max_val - min_val = 9 := by
  sorry

end quadratic_max_min_difference_l2677_267776


namespace isosceles_triangle_height_decreases_as_base_increases_l2677_267773

/-- Given an isosceles triangle with fixed side length and variable base length,
    the height is a decreasing function of the base length. -/
theorem isosceles_triangle_height_decreases_as_base_increases 
  (a : ℝ) (b : ℝ → ℝ) (h : ℝ → ℝ) :
  (∀ x, a > 0 ∧ b x > 0 ∧ h x > 0) →  -- Positive lengths
  (∀ x, a^2 = (h x)^2 + (b x)^2) →   -- Pythagorean theorem
  (∀ x, h x = Real.sqrt (a^2 - (b x)^2)) →  -- Height formula
  (∀ x y, x < y → b x < b y) →  -- b is increasing
  (∀ x y, x < y → h x > h y) :=  -- h is decreasing
by sorry


end isosceles_triangle_height_decreases_as_base_increases_l2677_267773


namespace total_cost_european_stamps_50s_60s_l2677_267738

-- Define the cost of stamps
def italy_stamp_cost : ℚ := 0.07
def germany_stamp_cost : ℚ := 0.03

-- Define the number of stamps collected
def italy_stamps_50s_60s : ℕ := 9
def germany_stamps_50s_60s : ℕ := 15

-- Theorem statement
theorem total_cost_european_stamps_50s_60s : 
  (italy_stamp_cost * italy_stamps_50s_60s + germany_stamp_cost * germany_stamps_50s_60s : ℚ) = 1.08 := by
  sorry

end total_cost_european_stamps_50s_60s_l2677_267738


namespace art_interest_group_end_time_l2677_267721

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.minutes + m
  let newHours := (t.hours + totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  ⟨newHours, newMinutes, by sorry, by sorry⟩

theorem art_interest_group_end_time :
  let start_time : Time := ⟨15, 20, by sorry, by sorry⟩
  let duration : Nat := 50
  addMinutes start_time duration = ⟨16, 10, by sorry, by sorry⟩ := by
  sorry

end art_interest_group_end_time_l2677_267721


namespace election_loss_calculation_l2677_267722

theorem election_loss_calculation (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 8000 →
  candidate_percentage = 1/4 →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 4000 := by
  sorry

end election_loss_calculation_l2677_267722


namespace problem_statement_l2677_267788

open Real

theorem problem_statement : 
  let p := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
  let q := ∀ x : ℝ, 2^x > x^2
  (¬p) ∨ q := by sorry

end problem_statement_l2677_267788


namespace vector_expression_simplification_l2677_267799

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_expression_simplification (a b : V) :
  (1 / 2 : ℝ) • ((2 : ℝ) • a - (4 : ℝ) • b) + (2 : ℝ) • b = a := by sorry

end vector_expression_simplification_l2677_267799


namespace max_value_of_roots_sum_l2677_267757

/-- Given a quadratic polynomial x^2 - sx + q with roots r₁ and r₂ satisfying
    certain conditions, the maximum value of 1/r₁¹¹ + 1/r₂¹¹ is 2. -/
theorem max_value_of_roots_sum (s q r₁ r₂ : ℝ) : 
  r₁ + r₂ = s ∧ r₁ * r₂ = q ∧ 
  r₁ + r₂ = r₁^2 + r₂^2 ∧ 
  r₁ + r₂ = r₁^10 + r₂^10 →
  ∃ (M : ℝ), M = 2 ∧ ∀ (s' q' r₁' r₂' : ℝ), 
    (r₁' + r₂' = s' ∧ r₁' * r₂' = q' ∧ 
     r₁' + r₂' = r₁'^2 + r₂'^2 ∧ 
     r₁' + r₂' = r₁'^10 + r₂'^10) →
    1 / r₁'^11 + 1 / r₂'^11 ≤ M :=
by sorry

end max_value_of_roots_sum_l2677_267757


namespace circumcircle_point_values_l2677_267727

-- Define the points A, B, and C
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (5, 3)
def C : ℝ × ℝ := (3, -1)

-- Define the equation of a circle
def circle_equation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

-- Define the circumcircle of triangle ABC
def circumcircle (x y : ℝ) : Prop :=
  ∃ D E F : ℝ,
    circle_equation x y D E F ∧
    circle_equation A.1 A.2 D E F ∧
    circle_equation B.1 B.2 D E F ∧
    circle_equation C.1 C.2 D E F

-- Theorem statement
theorem circumcircle_point_values :
  ∀ a : ℝ, circumcircle a 2 → a = 2 ∨ a = 6 :=
by sorry

end circumcircle_point_values_l2677_267727


namespace smallest_n_for_inequality_l2677_267703

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ 
  (∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n*(x^4 + y^4 + z^4 + w^4)) ∧ 
  (∀ (m : ℕ), m < n → ∃ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 > m*(x^4 + y^4 + z^4 + w^4)) :=
by sorry

end smallest_n_for_inequality_l2677_267703


namespace circle_and_line_proof_l2677_267733

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 3)^2 + (y + 2)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y + 5 = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 1)

-- Define point B
def point_B : ℝ × ℝ := (2, -2)

-- Define point D
def point_D : ℝ × ℝ := (-1, -1)

-- Define the line m (both possible equations)
def line_m (x y : ℝ) : Prop := x = -1 ∨ 3*x + 4*y + 7 = 0

theorem circle_and_line_proof :
  ∀ (x y : ℝ),
  (circle_C x y ↔ (x - point_A.1)^2 + (y - point_A.2)^2 = 25 ∧ 
                  (x - point_B.1)^2 + (y - point_B.2)^2 = 25) ∧
  (∃ (cx cy : ℝ), line_l cx cy ∧ circle_C cx cy) ∧
  (∃ (mx my : ℝ), line_m mx my ∧ point_D = (mx, my) ∧
    ∃ (x1 y1 x2 y2 : ℝ),
      circle_C x1 y1 ∧ circle_C x2 y2 ∧
      line_m x1 y1 ∧ line_m x2 y2 ∧
      (x1 - x2)^2 + (y1 - y2)^2 = 4 * 21) :=
by sorry

end circle_and_line_proof_l2677_267733


namespace elberta_has_45_dollars_l2677_267702

-- Define the amounts for each person
def granny_smith_amount : ℕ := 100
def anjou_amount : ℕ := (2 * granny_smith_amount) / 5
def elberta_amount : ℕ := anjou_amount + 5

-- Theorem to prove
theorem elberta_has_45_dollars : elberta_amount = 45 := by
  sorry

end elberta_has_45_dollars_l2677_267702


namespace max_r_value_exists_max_r_unique_max_r_l2677_267785

open Set Real

/-- The set T parameterized by r -/
def T (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 7)^2 ≤ r^2}

/-- The set S -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∀ θ : ℝ, cos (2 * θ) + p.1 * cos θ + p.2 ≥ 0}

/-- The main theorem stating the maximum value of r -/
theorem max_r_value (r : ℝ) (h_pos : r > 0) (h_subset : T r ⊆ S) : r ≤ 4 * sqrt 2 := by
  sorry

/-- The existence of the maximum r value -/
theorem exists_max_r : ∃ r : ℝ, r > 0 ∧ T r ⊆ S ∧ ∀ s : ℝ, s > 0 ∧ T s ⊆ S → s ≤ r := by
  sorry

/-- The uniqueness of the maximum r value -/
theorem unique_max_r (r s : ℝ) (hr : r > 0) (hs : s > 0)
    (h_max_r : T r ⊆ S ∧ ∀ t : ℝ, t > 0 ∧ T t ⊆ S → t ≤ r)
    (h_max_s : T s ⊆ S ∧ ∀ t : ℝ, t > 0 ∧ T t ⊆ S → t ≤ s) : r = s := by
  sorry

end max_r_value_exists_max_r_unique_max_r_l2677_267785


namespace school_student_count_l2677_267741

theorem school_student_count 
  (total : ℕ) 
  (transferred : ℕ) 
  (difference : ℕ) 
  (h1 : total = 432) 
  (h2 : transferred = 16) 
  (h3 : difference = 24) :
  ∃ (a b : ℕ), 
    a + b = total ∧ 
    (a - transferred) = (b + transferred + difference) ∧
    a = 244 ∧ 
    b = 188 := by
  sorry

end school_student_count_l2677_267741


namespace green_balls_count_l2677_267792

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 ∧
  white = 50 ∧
  yellow = 10 ∧
  red = 7 ∧
  purple = 3 ∧
  prob_not_red_purple = 9/10 →
  ∃ green : ℕ, green = 30 ∧ total = white + green + yellow + red + purple :=
by sorry

end green_balls_count_l2677_267792


namespace max_sum_given_constraint_l2677_267784

theorem max_sum_given_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 + (x+y)^3 + 36*x*y = 3456) : 
  x + y ≤ 12 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^3 + y₀^3 + (x₀+y₀)^3 + 36*x₀*y₀ = 3456 ∧ x₀ + y₀ = 12 :=
by sorry

end max_sum_given_constraint_l2677_267784


namespace odd_sum_even_equivalence_l2677_267751

theorem odd_sum_even_equivalence (x y : ℤ) :
  (Odd x ∧ Odd y → Even (x + y)) ↔ (¬Even (x + y) → ¬(Odd x ∧ Odd y)) := by sorry

end odd_sum_even_equivalence_l2677_267751


namespace dividend_proof_l2677_267781

theorem dividend_proof (divisor quotient remainder dividend : ℕ) : 
  divisor = 17 →
  remainder = 6 →
  dividend = divisor * quotient + remainder →
  dividend = 159 :=
by
  sorry

end dividend_proof_l2677_267781


namespace pie_shop_revenue_l2677_267736

/-- Represents the price of a single slice of pie in dollars -/
def slice_price : ℕ := 5

/-- Represents the number of slices in a whole pie -/
def slices_per_pie : ℕ := 4

/-- Represents the number of pies sold -/
def pies_sold : ℕ := 9

/-- Calculates the total revenue from selling pies -/
def total_revenue : ℕ := pies_sold * slices_per_pie * slice_price

theorem pie_shop_revenue :
  total_revenue = 180 :=
by sorry

end pie_shop_revenue_l2677_267736


namespace extended_pattern_ratio_l2677_267706

/-- Represents a square tile pattern -/
structure TilePattern where
  side : ℕ
  black_tiles : ℕ
  white_tiles : ℕ

/-- Creates an extended pattern by adding a border of black tiles -/
def extend_pattern (p : TilePattern) : TilePattern :=
  { side := p.side + 2,
    black_tiles := p.black_tiles + 4 * p.side + 4,
    white_tiles := p.white_tiles }

/-- The theorem to be proved -/
theorem extended_pattern_ratio (p : TilePattern)
  (h1 : p.side * p.side = p.black_tiles + p.white_tiles)
  (h2 : p.black_tiles = 12)
  (h3 : p.white_tiles = 24) :
  let extended := extend_pattern p
  (extended.black_tiles : ℚ) / extended.white_tiles = 3 / 2 := by
  sorry

end extended_pattern_ratio_l2677_267706


namespace correct_verbs_for_sentence_l2677_267789

-- Define the structure of a sentence with two blanks
structure SentenceWithBlanks where
  first_blank : String
  second_blank : String

-- Define the concept of subject-verb agreement
def subjectVerbAgrees (subject : String) (verb : String) : Prop := sorry

-- Define the specific sentence structure
def remoteAreasNeed : String := "Remote areas need"
def childrenNeed : String := "children need"

-- Theorem to prove
theorem correct_verbs_for_sentence :
  ∃ (s : SentenceWithBlanks),
    subjectVerbAgrees remoteAreasNeed s.first_blank ∧
    subjectVerbAgrees childrenNeed s.second_blank ∧
    s.first_blank = "is" ∧
    s.second_blank = "are" := by sorry

end correct_verbs_for_sentence_l2677_267789


namespace average_price_reduction_l2677_267790

theorem average_price_reduction (original_price final_price : ℝ) 
  (h1 : original_price = 60) 
  (h2 : final_price = 48.6) : 
  ∃ (x : ℝ), x = 0.1 ∧ original_price * (1 - x)^2 = final_price :=
sorry

end average_price_reduction_l2677_267790


namespace common_divisors_9240_10800_l2677_267719

theorem common_divisors_9240_10800 : 
  (Finset.filter (fun d => d ∣ 9240 ∧ d ∣ 10800) (Finset.range 10801)).card = 16 := by
  sorry

end common_divisors_9240_10800_l2677_267719


namespace least_positive_t_for_geometric_progression_l2677_267750

theorem least_positive_t_for_geometric_progression :
  ∃ (t : ℝ) (α : ℝ),
    0 < α ∧ α < Real.pi / 2 ∧
    (∃ (r : ℝ),
      Real.arcsin (Real.sin α) * r = Real.arcsin (Real.sin (3 * α)) ∧
      Real.arcsin (Real.sin (3 * α)) * r = Real.arcsin (Real.sin (5 * α)) ∧
      Real.arcsin (Real.sin (5 * α)) * r = Real.arcsin (Real.sin (t * α))) ∧
    (∀ (t' : ℝ) (α' : ℝ),
      0 < α' ∧ α' < Real.pi / 2 →
      (∃ (r' : ℝ),
        Real.arcsin (Real.sin α') * r' = Real.arcsin (Real.sin (3 * α')) ∧
        Real.arcsin (Real.sin (3 * α')) * r' = Real.arcsin (Real.sin (5 * α')) ∧
        Real.arcsin (Real.sin (5 * α')) * r' = Real.arcsin (Real.sin (t' * α'))) →
      t ≤ t') ∧
    t = 27 := by
  sorry

end least_positive_t_for_geometric_progression_l2677_267750


namespace poached_percentage_less_than_sold_l2677_267797

def total_pears : ℕ := 42
def sold_pears : ℕ := 20

def canned_pears (poached : ℕ) : ℕ := poached + poached / 5

theorem poached_percentage_less_than_sold :
  ∃ (poached : ℕ),
    poached > 0 ∧
    poached < sold_pears ∧
    total_pears = sold_pears + canned_pears poached + poached ∧
    (sold_pears - poached) * 100 / sold_pears = 50 := by
  sorry

end poached_percentage_less_than_sold_l2677_267797


namespace hostel_expenditure_increase_l2677_267778

/-- Calculates the increase in total expenditure for a hostel after accommodating more students. -/
theorem hostel_expenditure_increase
  (initial_students : ℕ)
  (additional_students : ℕ)
  (average_decrease : ℚ)
  (new_total_expenditure : ℚ)
  (h1 : initial_students = 100)
  (h2 : additional_students = 20)
  (h3 : average_decrease = 5)
  (h4 : new_total_expenditure = 5400) :
  let total_students := initial_students + additional_students
  let new_average := new_total_expenditure / total_students
  let original_average := new_average + average_decrease
  let original_total_expenditure := original_average * initial_students
  new_total_expenditure - original_total_expenditure = 400 :=
by sorry

end hostel_expenditure_increase_l2677_267778


namespace alcohol_solution_proof_l2677_267715

/-- Proves that adding 14.285714285714286 liters of pure alcohol to a 100-liter solution
    results in a 30% alcohol solution if and only if the initial alcohol percentage was 20% -/
theorem alcohol_solution_proof (initial_percentage : ℝ) : 
  (initial_percentage / 100) * 100 + 14.285714285714286 = 0.30 * (100 + 14.285714285714286) ↔ 
  initial_percentage = 20 := by sorry

end alcohol_solution_proof_l2677_267715


namespace average_speed_two_hours_l2677_267737

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 90 → speed2 = 55 → (speed1 + speed2) / 2 = 72.5 := by
  sorry

end average_speed_two_hours_l2677_267737


namespace fib_recurrence_l2677_267768

/-- Fibonacci sequence defined as the number of ways to represent n as an ordered sum of ones and twos -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: The Fibonacci sequence satisfies the recurrence relation F_n = F_{n-1} + F_{n-2} for n ≥ 2 -/
theorem fib_recurrence (n : ℕ) (h : n ≥ 2) : fib n = fib (n - 1) + fib (n - 2) := by
  sorry

#check fib_recurrence

end fib_recurrence_l2677_267768


namespace different_course_selections_eq_30_l2677_267758

/-- The number of ways two people can choose 2 courses each from 4 courses with at least one course different -/
def different_course_selections : ℕ :=
  Nat.choose 4 2 * Nat.choose 2 2 + Nat.choose 4 1 * Nat.choose 3 1 * Nat.choose 2 1

/-- Theorem stating that the number of different course selections is 30 -/
theorem different_course_selections_eq_30 : different_course_selections = 30 := by
  sorry

end different_course_selections_eq_30_l2677_267758


namespace allocation_schemes_eq_36_l2677_267765

/-- The number of ways to assign 4 intern teachers to 3 classes, with at least 1 teacher in each class -/
def allocation_schemes : ℕ :=
  -- We define the number of allocation schemes here
  -- The actual calculation is not provided, as we're only writing the statement
  36

/-- Theorem stating that the number of allocation schemes is 36 -/
theorem allocation_schemes_eq_36 : allocation_schemes = 36 := by
  sorry


end allocation_schemes_eq_36_l2677_267765


namespace daughters_age_in_three_years_l2677_267798

/-- Given that 5 years ago, a mother was twice as old as her daughter, and the mother is 41 years old now,
    prove that the daughter will be 26 years old in 3 years. -/
theorem daughters_age_in_three_years 
  (mother_age_now : ℕ) 
  (mother_daughter_age_relation : ℕ → ℕ → Prop) 
  (h1 : mother_age_now = 41)
  (h2 : mother_daughter_age_relation (mother_age_now - 5) ((mother_age_now - 5) / 2)) :
  ((mother_age_now - 5) / 2) + 8 = 26 := by
  sorry

#check daughters_age_in_three_years

end daughters_age_in_three_years_l2677_267798


namespace common_area_rectangle_ellipse_l2677_267763

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents an ellipse with semi-major and semi-minor axis lengths -/
structure Ellipse where
  semiMajor : ℝ
  semiMinor : ℝ

/-- Calculates the area of the region common to a rectangle and an ellipse that share the same center -/
def commonArea (r : Rectangle) (e : Ellipse) : ℝ := sorry

theorem common_area_rectangle_ellipse :
  let r := Rectangle.mk 10 4
  let e := Ellipse.mk 3 2
  commonArea r e = 6 * Real.pi := by sorry

end common_area_rectangle_ellipse_l2677_267763


namespace cm_to_m_sq_dm_to_sq_m_min_to_hour_g_to_kg_seven_cm_to_m_thirtyfive_sq_dm_to_sq_m_fortyfive_min_to_hour_twothousandfivehundred_g_to_kg_l2677_267716

-- Define conversion rates
def cm_per_m : ℚ := 100
def sq_dm_per_sq_m : ℚ := 100
def min_per_hour : ℚ := 60
def g_per_kg : ℚ := 1000

-- Theorems to prove
theorem cm_to_m (x : ℚ) : x / cm_per_m = x / 100 := by sorry

theorem sq_dm_to_sq_m (x : ℚ) : x / sq_dm_per_sq_m = x / 100 := by sorry

theorem min_to_hour (x : ℚ) : x / min_per_hour = x / 60 := by sorry

theorem g_to_kg (x : ℚ) : x / g_per_kg = x / 1000 := by sorry

-- Specific conversions
theorem seven_cm_to_m : 7 / cm_per_m = 7 / 100 := by sorry

theorem thirtyfive_sq_dm_to_sq_m : 35 / sq_dm_per_sq_m = 7 / 20 := by sorry

theorem fortyfive_min_to_hour : 45 / min_per_hour = 3 / 4 := by sorry

theorem twothousandfivehundred_g_to_kg : 2500 / g_per_kg = 5 / 2 := by sorry

end cm_to_m_sq_dm_to_sq_m_min_to_hour_g_to_kg_seven_cm_to_m_thirtyfive_sq_dm_to_sq_m_fortyfive_min_to_hour_twothousandfivehundred_g_to_kg_l2677_267716


namespace cost_increase_l2677_267739

/-- Cost function -/
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

theorem cost_increase (t : ℝ) (b₀ b₁ : ℝ) (h : t > 0) :
  cost t b₁ = 16 * cost t b₀ → b₁ = 2 * b₀ := by
  sorry

end cost_increase_l2677_267739


namespace roof_weight_capacity_is_500_l2677_267772

/-- The number of leaves that fall on Bill's roof each day -/
def leaves_per_day : ℕ := 100

/-- The number of leaves that weigh one pound -/
def leaves_per_pound : ℕ := 1000

/-- The number of days it takes for Bill's roof to collapse -/
def days_to_collapse : ℕ := 5000

/-- The weight Bill's roof can bear in pounds -/
def roof_weight_capacity : ℚ :=
  (leaves_per_day : ℚ) / (leaves_per_pound : ℚ) * days_to_collapse

theorem roof_weight_capacity_is_500 :
  roof_weight_capacity = 500 := by sorry

end roof_weight_capacity_is_500_l2677_267772


namespace arithmetic_sequence_common_difference_l2677_267791

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: If S_7/7 - S_4/4 = 3 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : seq.S 7 / 7 - seq.S 4 / 4 = 3) :
  seq.d = 2 := by
  sorry

end arithmetic_sequence_common_difference_l2677_267791
