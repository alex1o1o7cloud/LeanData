import Mathlib

namespace num_winning_scores_l2451_245100

/-- Represents a cross country meet with 3 teams of 4 runners each -/
structure CrossCountryMeet where
  numTeams : Nat
  runnersPerTeam : Nat
  totalRunners : Nat
  (team_count : numTeams = 3)
  (runner_count : runnersPerTeam = 4)
  (total_runners : totalRunners = numTeams * runnersPerTeam)

/-- Calculates the total score of all runners -/
def totalScore (meet : CrossCountryMeet) : Nat :=
  meet.totalRunners * (meet.totalRunners + 1) / 2

/-- Calculates the minimum possible winning score -/
def minWinningScore (meet : CrossCountryMeet) : Nat :=
  meet.runnersPerTeam * (meet.runnersPerTeam + 1) / 2

/-- Calculates the maximum possible winning score -/
def maxWinningScore (meet : CrossCountryMeet) : Nat :=
  totalScore meet / meet.numTeams

/-- Theorem stating the number of different winning scores possible -/
theorem num_winning_scores (meet : CrossCountryMeet) :
  (maxWinningScore meet - minWinningScore meet + 1) = 17 := by
  sorry


end num_winning_scores_l2451_245100


namespace intersection_of_M_and_N_l2451_245169

def M : Set ℝ := {y | ∃ x, y = Real.sin x}
def N : Set ℝ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end intersection_of_M_and_N_l2451_245169


namespace series_convergence_l2451_245106

def series_term (n : ℕ) : ℚ :=
  (4 * n + 3) / ((4 * n - 2)^2 * (4 * n + 2)^2)

def series_sum : ℚ := 1 / 128

theorem series_convergence : 
  (∑' n, series_term n) = series_sum :=
sorry

end series_convergence_l2451_245106


namespace budget_is_seven_seventy_l2451_245199

/-- The budget for bulbs given the number of crocus bulbs and their cost -/
def budget_for_bulbs (num_crocus : ℕ) (cost_per_crocus : ℚ) : ℚ :=
  num_crocus * cost_per_crocus

/-- Theorem stating that the budget for bulbs is $7.70 -/
theorem budget_is_seven_seventy :
  budget_for_bulbs 22 (35/100) = 77/10 := by
  sorry

end budget_is_seven_seventy_l2451_245199


namespace closest_integer_to_cube_root_100_l2451_245136

theorem closest_integer_to_cube_root_100 :
  ∃ n : ℤ, ∀ m : ℤ, |n ^ 3 - 100| ≤ |m ^ 3 - 100| ∧ n = 5 :=
by sorry

end closest_integer_to_cube_root_100_l2451_245136


namespace solution_set_part1_solution_range_part2_l2451_245109

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |2*x + 1| + |a*x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 2 ≥ 3} = {x : ℝ | x ≤ -3/4 ∨ x ≥ 3/4} := by sorry

-- Part 2
theorem solution_range_part2 :
  ∀ a : ℝ, a > 0 → (∃ x : ℝ, f x a < a/2 + 1) ↔ a > 2 := by sorry

end solution_set_part1_solution_range_part2_l2451_245109


namespace closest_integer_to_cube_root_l2451_245117

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |m - (9^3 + 7^3)^(1/3)| ≥ |n - (9^3 + 7^3)^(1/3)| :=
sorry

end closest_integer_to_cube_root_l2451_245117


namespace max_value_trig_expression_l2451_245153

theorem max_value_trig_expression (x : ℝ) : 11 - 8 * Real.cos x - 2 * (Real.sin x)^2 ≤ 19 := by
  sorry

end max_value_trig_expression_l2451_245153


namespace least_sum_of_exponents_l2451_245147

/-- The sum of distinct powers of 2 that equals 700 -/
def sum_of_powers (powers : List ℕ) : Prop :=
  (powers.map (λ x => 2^x)).sum = 700 ∧ powers.Nodup

/-- The proposition that 30 is the least possible sum of exponents -/
theorem least_sum_of_exponents :
  ∀ powers : List ℕ,
    sum_of_powers powers →
    powers.length ≥ 3 →
    powers.sum ≥ 30 ∧
    ∃ optimal_powers : List ℕ,
      sum_of_powers optimal_powers ∧
      optimal_powers.length ≥ 3 ∧
      optimal_powers.sum = 30 :=
sorry

end least_sum_of_exponents_l2451_245147


namespace complex_sum_equality_l2451_245114

/-- Given complex numbers B, Q, R, and T, prove that their sum is equal to 1 + 9i -/
theorem complex_sum_equality (B Q R T : ℂ) 
  (hB : B = 3 + 2*I)
  (hQ : Q = -5)
  (hR : R = 2*I)
  (hT : T = 3 + 5*I) :
  B - Q + R + T = 1 + 9*I :=
by sorry

end complex_sum_equality_l2451_245114


namespace square_area_error_percentage_l2451_245171

theorem square_area_error_percentage (s : ℝ) (h : s > 0) :
  let measured_side := 1.06 * s
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let error_percentage := (area_error / actual_area) * 100
  error_percentage = 12.36 := by
sorry

end square_area_error_percentage_l2451_245171


namespace lines_equal_angles_with_plane_l2451_245135

-- Define a plane in 3D space
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define a line in 3D space
def Line : Type := ℝ → ℝ × ℝ × ℝ

-- Define the angle between a line and a plane
def angle_line_plane (l : Line) (p : Plane) : ℝ := sorry

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define intersecting lines
def intersecting (l1 l2 : Line) : Prop := sorry

-- Define skew lines
def skew (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem lines_equal_angles_with_plane (l1 l2 : Line) (p : Plane) 
  (h_distinct : l1 ≠ l2) 
  (h_equal_angles : angle_line_plane l1 p = angle_line_plane l2 p) :
  parallel l1 l2 ∨ intersecting l1 l2 ∨ skew l1 l2 := by sorry

end lines_equal_angles_with_plane_l2451_245135


namespace arithmetic_expression_equality_l2451_245138

theorem arithmetic_expression_equality : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 - (-9) = 8 := by
  sorry

end arithmetic_expression_equality_l2451_245138


namespace sum_of_roots_l2451_245176

theorem sum_of_roots (x : ℝ) : x + 16 / x = 12 → ∃ y : ℝ, y + 16 / y = 12 ∧ x + y = 12 :=
  sorry

end sum_of_roots_l2451_245176


namespace state_tax_deduction_l2451_245121

theorem state_tax_deduction (hourly_wage : ℝ) (tax_rate : ℝ) : 
  hourly_wage = 25 → tax_rate = 0.024 → hourly_wage * tax_rate * 100 = 60 := by
  sorry

end state_tax_deduction_l2451_245121


namespace album_slots_equal_sum_of_photos_l2451_245170

/-- The number of photos brought by each person --/
def cristina_photos : ℕ := 7
def john_photos : ℕ := 10
def sarah_photos : ℕ := 9
def clarissa_photos : ℕ := 14

/-- The total number of slots in the photo album --/
def album_slots : ℕ := cristina_photos + john_photos + sarah_photos + clarissa_photos

/-- Theorem stating that the number of slots in the photo album
    is equal to the sum of photos brought by all four people --/
theorem album_slots_equal_sum_of_photos :
  album_slots = cristina_photos + john_photos + sarah_photos + clarissa_photos :=
by sorry

end album_slots_equal_sum_of_photos_l2451_245170


namespace sum_of_specific_values_is_zero_l2451_245187

-- Define an odd function f on ℝ
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property f(x+1) = -f(x-1)
def hasFunctionalProperty (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = -f (x - 1)

-- Theorem statement
theorem sum_of_specific_values_is_zero
  (f : ℝ → ℝ)
  (h1 : isOddFunction f)
  (h2 : hasFunctionalProperty f) :
  f 0 + f 1 + f 2 + f 3 + f 4 = 0 :=
sorry

end sum_of_specific_values_is_zero_l2451_245187


namespace point_coordinates_sum_of_coordinates_l2451_245184

/-- Given three points X, Y, and Z in the plane satisfying certain ratios,
    prove that X has specific coordinates. -/
theorem point_coordinates (X Y Z : ℝ × ℝ) : 
  Y = (2, 3) →
  Z = (5, 1) →
  (dist X Z) / (dist X Y) = 1/3 →
  (dist Z Y) / (dist X Y) = 2/3 →
  X = (6.5, 0) :=
by sorry

/-- The sum of coordinates of point X -/
def sum_coordinates (X : ℝ × ℝ) : ℝ :=
  X.1 + X.2

/-- Prove that the sum of coordinates of X is 6.5 -/
theorem sum_of_coordinates (X : ℝ × ℝ) :
  X = (6.5, 0) →
  sum_coordinates X = 6.5 :=
by sorry

end point_coordinates_sum_of_coordinates_l2451_245184


namespace min_product_of_reciprocal_sum_l2451_245168

theorem min_product_of_reciprocal_sum (a b : ℕ+) : 
  (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6 → 
  ∀ c d : ℕ+, (1 : ℚ) / c + (1 : ℚ) / (3 * d) = (1 : ℚ) / 6 → 
  a * b ≤ c * d ∧ a * b = 98 :=
by sorry

end min_product_of_reciprocal_sum_l2451_245168


namespace min_value_of_f_l2451_245137

open Real

noncomputable def f (x y : ℝ) : ℝ :=
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) / 
  (4 - x^2 - 10 * x * y - 25 * y^2)^(7/2)

theorem min_value_of_f :
  ∃ (min : ℝ), min = 5/32 ∧ ∀ (x y : ℝ), f x y ≥ min :=
by sorry

end min_value_of_f_l2451_245137


namespace wages_theorem_l2451_245164

/-- Given a sum of money that can pay B's wages for 12 days and C's wages for 24 days,
    prove that it can pay both B and C's wages together for 8 days -/
theorem wages_theorem (S : ℝ) (W_B W_C : ℝ) (h1 : S = 12 * W_B) (h2 : S = 24 * W_C) :
  S = 8 * (W_B + W_C) := by
  sorry

end wages_theorem_l2451_245164


namespace pillar_base_side_length_l2451_245193

theorem pillar_base_side_length (string_length : ℝ) (side_length : ℝ) : 
  string_length = 78 → 
  string_length = 3 * side_length → 
  side_length = 26 := by
  sorry

#check pillar_base_side_length

end pillar_base_side_length_l2451_245193


namespace proportion_solution_l2451_245119

theorem proportion_solution (x : ℝ) : 
  (1.25 / x = 15 / 26.5) → x = 33.125 / 15 := by
  sorry

end proportion_solution_l2451_245119


namespace two_numbers_with_difference_and_quotient_l2451_245105

theorem two_numbers_with_difference_and_quotient :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a - b = 157 ∧ a / b = 2 ∧ a = 314 ∧ b = 157 := by
  sorry

end two_numbers_with_difference_and_quotient_l2451_245105


namespace cos_2alpha_plus_4pi_over_3_l2451_245155

theorem cos_2alpha_plus_4pi_over_3 (α : Real) 
  (h : Real.sqrt 3 * Real.sin α + Real.cos α = 1/2) : 
  Real.cos (2 * α + 4 * π / 3) = -7/8 := by
  sorry

end cos_2alpha_plus_4pi_over_3_l2451_245155


namespace car_fuel_consumption_l2451_245123

/-- Represents the distance a car can travel with a given amount of fuel -/
def distance_traveled (fuel_fraction : ℚ) (distance : ℚ) : ℚ := distance / fuel_fraction

/-- Represents the remaining distance a car can travel -/
def remaining_distance (total_distance : ℚ) (traveled_distance : ℚ) : ℚ :=
  total_distance - traveled_distance

theorem car_fuel_consumption 
  (initial_distance : ℚ) 
  (initial_fuel_fraction : ℚ) 
  (h1 : initial_distance = 165) 
  (h2 : initial_fuel_fraction = 3/8) : 
  remaining_distance (distance_traveled 1 initial_fuel_fraction) initial_distance = 275 := by
  sorry

#eval remaining_distance (distance_traveled 1 (3/8)) 165

end car_fuel_consumption_l2451_245123


namespace rectangle_width_l2451_245156

/-- Given a rectangle with specific properties, prove its width is 6 meters -/
theorem rectangle_width (area perimeter length width : ℝ) 
  (h_area : area = 50)
  (h_perimeter : perimeter = 30)
  (h_ratio : length = (3/2) * width)
  (h_area_def : area = length * width)
  (h_perimeter_def : perimeter = 2 * (length + width)) :
  width = 6 := by sorry

end rectangle_width_l2451_245156


namespace seven_digit_multiple_of_each_l2451_245149

/-- A function that returns the set of digits of a positive integer -/
def digits (n : ℕ+) : Finset ℕ :=
  sorry

/-- The theorem statement -/
theorem seven_digit_multiple_of_each : ∃ (n : ℕ+),
  (digits n).card = 7 ∧
  ∀ d ∈ digits n, d > 0 ∧ n % d = 0 →
  digits n = {1, 2, 3, 6, 7, 8, 9} :=
sorry

end seven_digit_multiple_of_each_l2451_245149


namespace function_properties_l2451_245190

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem function_properties :
  ∀ (a b c : ℝ),
  (f' a b 1 = 3) →  -- Tangent line condition
  (f a b c 1 = 2) →  -- Point condition
  (f' a b (-2) = 0) →  -- Extreme value condition
  (a = 2 ∧ b = -4 ∧ c = 5) ∧  -- Correct values of a, b, c
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f' 2 (-4) x ≥ 0)  -- Monotonically increasing condition
  :=
by sorry

end function_properties_l2451_245190


namespace unique_modular_residue_l2451_245110

theorem unique_modular_residue :
  ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ -150 ≡ n [ZMOD 17] := by sorry

end unique_modular_residue_l2451_245110


namespace local_max_implies_a_gt_half_l2451_245180

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + (2*a - 1) * x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2*a*x + 2*a

theorem local_max_implies_a_gt_half (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) →
  a > 1/2 :=
sorry

end local_max_implies_a_gt_half_l2451_245180


namespace line_up_count_distribution_count_l2451_245133

/-- Represents a student --/
inductive Student : Type
| A
| B
| C
| D
| E

/-- Represents a line-up of students --/
def LineUp := List Student

/-- Represents a distribution of students into classes --/
def Distribution := List (List Student)

/-- Checks if two students are adjacent in a line-up --/
def areAdjacent (s1 s2 : Student) (lineup : LineUp) : Prop := sorry

/-- Checks if a distribution is valid (three non-empty classes) --/
def isValidDistribution (d : Distribution) : Prop := sorry

/-- Counts the number of valid line-ups --/
def countValidLineUps : Nat := sorry

/-- Counts the number of valid distributions --/
def countValidDistributions : Nat := sorry

theorem line_up_count :
  countValidLineUps = 12 := by sorry

theorem distribution_count :
  countValidDistributions = 150 := by sorry

end line_up_count_distribution_count_l2451_245133


namespace probability_no_shaded_square_correct_l2451_245175

/-- The probability of a randomly chosen rectangle not including a shaded square
    in a 2 by 2001 rectangle with the middle unit square of each row shaded. -/
def probability_no_shaded_square : ℚ :=
  1001 / 2001

/-- The number of columns in the rectangle. -/
def num_columns : ℕ := 2001

/-- The number of rows in the rectangle. -/
def num_rows : ℕ := 2

/-- The total number of rectangles that can be formed in a single row. -/
def rectangles_per_row : ℕ := (num_columns + 1).choose 2

/-- The number of rectangles in a single row that include the shaded square. -/
def shaded_rectangles_per_row : ℕ := (num_columns + 1) / 2 * (num_columns / 2)

theorem probability_no_shaded_square_correct :
  probability_no_shaded_square = 1 - (3 * shaded_rectangles_per_row) / (3 * rectangles_per_row) :=
sorry

end probability_no_shaded_square_correct_l2451_245175


namespace becky_necklaces_l2451_245141

theorem becky_necklaces (initial : ℕ) : 
  initial - 3 + 5 - 15 = 37 → initial = 50 := by
  sorry

#check becky_necklaces

end becky_necklaces_l2451_245141


namespace problem_solution_l2451_245173

noncomputable section

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem problem_solution :
  (∀ a : ℝ, A ∩ B a = A ∪ B a → a = 1) ∧
  (∀ a : ℝ, A ∩ B a = B a → a ≤ -1 ∨ a = 1) :=
sorry

end

end problem_solution_l2451_245173


namespace function_value_at_ln_half_l2451_245179

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (5^x) / (5^x + 1)

theorem function_value_at_ln_half (a : ℝ) :
  (f a (Real.log 2) = 4) → (f a (Real.log (1/2)) = -3) := by
  sorry

end function_value_at_ln_half_l2451_245179


namespace percentage_equality_l2451_245194

-- Define variables
variable (j k l m : ℝ)

-- Define the conditions
def condition1 : Prop := 1.25 * j = 0.25 * k
def condition2 : Prop := 1.5 * k = 0.5 * l
def condition3 : Prop := 0.2 * m = 7 * j

-- Theorem statement
theorem percentage_equality 
  (h1 : condition1 j k) 
  (h2 : condition2 k l) 
  (h3 : condition3 j m) : 
  1.75 * l = 0.75 * m := by sorry

end percentage_equality_l2451_245194


namespace total_vehicles_is_282_l2451_245102

/-- The number of vehicles Kendra saw during her road trip -/
def total_vehicles : ℕ :=
  let morning_minivans := 20
  let morning_sedans := 17
  let morning_suvs := 12
  let morning_trucks := 8
  let morning_motorcycles := 5

  let afternoon_minivans := 22
  let afternoon_sedans := 13
  let afternoon_suvs := 15
  let afternoon_trucks := 10
  let afternoon_motorcycles := 7

  let evening_minivans := 15
  let evening_sedans := 19
  let evening_suvs := 18
  let evening_trucks := 14
  let evening_motorcycles := 10

  let night_minivans := 10
  let night_sedans := 12
  let night_suvs := 20
  let night_trucks := 20
  let night_motorcycles := 15

  let total_minivans := morning_minivans + afternoon_minivans + evening_minivans + night_minivans
  let total_sedans := morning_sedans + afternoon_sedans + evening_sedans + night_sedans
  let total_suvs := morning_suvs + afternoon_suvs + evening_suvs + night_suvs
  let total_trucks := morning_trucks + afternoon_trucks + evening_trucks + night_trucks
  let total_motorcycles := morning_motorcycles + afternoon_motorcycles + evening_motorcycles + night_motorcycles

  total_minivans + total_sedans + total_suvs + total_trucks + total_motorcycles

theorem total_vehicles_is_282 : total_vehicles = 282 := by
  sorry

end total_vehicles_is_282_l2451_245102


namespace exists_determining_question_l2451_245148

-- Define the types of guests
inductive GuestType
| Human
| Vampire

-- Define the possible answers
inductive Answer
| Bal
| Da

-- Define a question as a function that takes a GuestType and returns an Answer
def Question := GuestType → Answer

-- Define a function to determine the guest type based on the answer
def determineGuestType (q : Question) (a : Answer) : GuestType := 
  match a with
  | Answer.Bal => GuestType.Human
  | Answer.Da => GuestType.Vampire

-- Theorem statement
theorem exists_determining_question : 
  ∃ (q : Question), 
    (∀ (g : GuestType), (determineGuestType q (q g)) = g) :=
sorry

end exists_determining_question_l2451_245148


namespace non_negative_integer_solutions_l2451_245181

def is_solution (x y : ℕ) : Prop := 2 * x + y = 5

theorem non_negative_integer_solutions :
  {p : ℕ × ℕ | is_solution p.1 p.2} = {(0, 5), (1, 3), (2, 1)} := by
  sorry

end non_negative_integer_solutions_l2451_245181


namespace tan_to_sin_cos_ratio_l2451_245165

theorem tan_to_sin_cos_ratio (α : Real) (h : Real.tan α = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/3 := by
  sorry

end tan_to_sin_cos_ratio_l2451_245165


namespace property_1_property_2_property_3_f_satisfies_all_properties_l2451_245150

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Property 1: f(xy) = f(x)f(y)
theorem property_1 : ∀ x y : ℝ, f (x * y) = f x * f y := by sorry

-- Property 2: f'(x) is an even function
theorem property_2 : ∀ x : ℝ, (deriv f) (-x) = (deriv f) x := by sorry

-- Property 3: f(x) is monotonically increasing on (0, +∞)
theorem property_3 : ∀ x y : ℝ, 0 < x → x < y → f x < f y := by sorry

-- Main theorem: f(x) = x^3 satisfies all three properties
theorem f_satisfies_all_properties :
  (∀ x y : ℝ, f (x * y) = f x * f y) ∧
  (∀ x : ℝ, (deriv f) (-x) = (deriv f) x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by sorry

end property_1_property_2_property_3_f_satisfies_all_properties_l2451_245150


namespace complex_multiplication_l2451_245118

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2*i := by sorry

end complex_multiplication_l2451_245118


namespace distance_to_larger_section_l2451_245146

/-- Right pentagonal pyramid with two parallel cross sections -/
structure RightPentagonalPyramid where
  /-- Area of smaller cross section in square feet -/
  area_small : ℝ
  /-- Area of larger cross section in square feet -/
  area_large : ℝ
  /-- Distance between the two cross sections in feet -/
  distance_between : ℝ

/-- Theorem: Distance from apex to larger cross section -/
theorem distance_to_larger_section (pyramid : RightPentagonalPyramid) 
  (h_area_small : pyramid.area_small = 100 * Real.sqrt 3)
  (h_area_large : pyramid.area_large = 225 * Real.sqrt 3)
  (h_distance : pyramid.distance_between = 5) :
  ∃ (d : ℝ), d = 15 ∧ d * d * pyramid.area_small = (d - 5) * (d - 5) * pyramid.area_large :=
by sorry

end distance_to_larger_section_l2451_245146


namespace wages_comparison_l2451_245172

theorem wages_comparison (E R C : ℝ) 
  (hC_E : C = E * 1.7)
  (hC_R : C = R * 1.3076923076923077) :
  R = E * 1.3 :=
by sorry

end wages_comparison_l2451_245172


namespace cookies_per_bag_l2451_245161

theorem cookies_per_bag (chocolate_chip : ℕ) (oatmeal : ℕ) (baggies : ℕ) 
  (h1 : chocolate_chip = 23)
  (h2 : oatmeal = 25)
  (h3 : baggies = 8) :
  (chocolate_chip + oatmeal) / baggies = 6 := by
  sorry

end cookies_per_bag_l2451_245161


namespace other_workers_count_l2451_245182

def total_workers : ℕ := 5
def chosen_workers : ℕ := 2
def probability_jack_and_jill : ℚ := 1/10

theorem other_workers_count :
  let other_workers := total_workers - 2
  probability_jack_and_jill = 1 / (total_workers.choose chosen_workers) →
  other_workers = 3 := by
sorry

end other_workers_count_l2451_245182


namespace missing_number_is_seven_l2451_245196

def known_numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14]

theorem missing_number_is_seven (x : ℕ) :
  (known_numbers.sum + x) / 12 = 12 →
  x = 7 := by sorry

end missing_number_is_seven_l2451_245196


namespace ninth_ninety_ninth_digit_sum_l2451_245142

def decimal_expansion (n : ℕ) (d : ℕ) : ℚ := n / d

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem ninth_ninety_ninth_digit_sum (n : ℕ) : 
  nth_digit_after_decimal (decimal_expansion 2 9 + decimal_expansion 3 11 + decimal_expansion 5 13) 999 = 8 := by
  sorry

end ninth_ninety_ninth_digit_sum_l2451_245142


namespace expression_evaluation_l2451_245159

theorem expression_evaluation :
  (2^2003 * 3^2002 * 5) / 6^2003 = 5/3 := by
  sorry

end expression_evaluation_l2451_245159


namespace log_inequality_solution_set_l2451_245113

-- Define the logarithm function with base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the solution set
def solution_set : Set ℝ := { x | x > -1 ∧ x ≤ 0 }

-- Theorem statement
theorem log_inequality_solution_set :
  { x : ℝ | x > -1 ∧ log10 (x + 1) ≤ 0 } = solution_set :=
sorry

end log_inequality_solution_set_l2451_245113


namespace cos_seven_pi_sixths_l2451_245116

theorem cos_seven_pi_sixths : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 := by
  sorry

end cos_seven_pi_sixths_l2451_245116


namespace collinear_points_b_value_l2451_245128

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points_b_value :
  ∃ b : ℝ, collinear 4 (-6) (b + 3) 4 (3*b - 2) 3 ∧ b = 17/7 := by
sorry

end collinear_points_b_value_l2451_245128


namespace intersection_complement_M_and_N_l2451_245101

def U : Set Int := {-2, -1, 0, 1, 2}

def M : Set Int := {y | ∃ x, y = 2^x}

def N : Set Int := {x | x^2 - x - 2 = 0}

theorem intersection_complement_M_and_N :
  (U \ M) ∩ N = {-1} := by sorry

end intersection_complement_M_and_N_l2451_245101


namespace total_people_in_program_l2451_245166

theorem total_people_in_program (parents pupils teachers staff volunteers : ℕ) 
  (h1 : parents = 105)
  (h2 : pupils = 698)
  (h3 : teachers = 35)
  (h4 : staff = 20)
  (h5 : volunteers = 50) :
  parents + pupils + teachers + staff + volunteers = 908 := by
  sorry

end total_people_in_program_l2451_245166


namespace abc_problem_l2451_245115

theorem abc_problem (a b c : ℝ) 
  (h1 : 2011 * (a + b + c) = 1) 
  (h2 : a * b + a * c + b * c = 2011 * a * b * c) : 
  a^2011 * b^2011 + c^2011 = (1 : ℝ) / 2011^2011 := by
  sorry

end abc_problem_l2451_245115


namespace log_base_8_equals_3_l2451_245154

theorem log_base_8_equals_3 (y : ℝ) (h : Real.log y / Real.log 8 = 3) : y = 512 := by
  sorry

end log_base_8_equals_3_l2451_245154


namespace incenter_coords_l2451_245125

/-- Triangle ABC with incenter I -/
structure TriangleWithIncenter where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side BC -/
  BC : ℝ
  /-- Length of side CA -/
  CA : ℝ
  /-- Incenter I of the triangle -/
  I : ℝ × ℝ
  /-- Coordinates of incenter I as (x, y, z) where x⃗A + y⃗B + z⃗C = ⃗I -/
  coords : ℝ × ℝ × ℝ

/-- The theorem stating that the coordinates of the incenter are (2/9, 1/3, 4/9) -/
theorem incenter_coords (t : TriangleWithIncenter) 
  (h1 : t.AB = 6)
  (h2 : t.BC = 8)
  (h3 : t.CA = 4)
  (h4 : t.coords.1 + t.coords.2.1 + t.coords.2.2 = 1) :
  t.coords = (2/9, 1/3, 4/9) := by
  sorry

end incenter_coords_l2451_245125


namespace legos_set_cost_l2451_245112

def total_earnings : ℕ := 45
def car_price : ℕ := 5
def num_cars : ℕ := 3

theorem legos_set_cost : total_earnings - (car_price * num_cars) = 30 := by
  sorry

end legos_set_cost_l2451_245112


namespace galaxy_distance_in_miles_l2451_245144

/-- The number of miles in one light-year -/
def miles_per_light_year : ℝ := 6 * 10^12

/-- The distance to the observed galaxy in thousand million light-years -/
def galaxy_distance_thousand_million_light_years : ℝ := 13.4

/-- Conversion factor from thousand million to billion -/
def thousand_million_to_billion : ℝ := 1

theorem galaxy_distance_in_miles :
  let distance_light_years := galaxy_distance_thousand_million_light_years * thousand_million_to_billion * 10^9
  let distance_miles := distance_light_years * miles_per_light_year
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |distance_miles - 8 * 10^22| < ε * (8 * 10^22) :=
sorry

end galaxy_distance_in_miles_l2451_245144


namespace factorization_x_squared_minus_2x_l2451_245107

theorem factorization_x_squared_minus_2x (x : ℝ) : x^2 - 2*x = x*(x - 2) := by
  sorry

end factorization_x_squared_minus_2x_l2451_245107


namespace enrollment_analysis_l2451_245108

def summit_ridge : ℕ := 1560
def pine_hills : ℕ := 1150
def oak_valley : ℕ := 1950
def maple_town : ℕ := 1840

def enrollments : List ℕ := [summit_ridge, pine_hills, oak_valley, maple_town]

theorem enrollment_analysis :
  (List.maximum enrollments).get! - (List.minimum enrollments).get! = 800 ∧
  (List.sum enrollments) / enrollments.length = 1625 := by
  sorry

end enrollment_analysis_l2451_245108


namespace solution_exists_iff_a_in_interval_l2451_245127

/-- The system of equations has a solution within the specified square if and only if
    a is in the given interval for some integer k. -/
theorem solution_exists_iff_a_in_interval :
  ∀ (a : ℝ), ∃ (x y : ℝ),
    (x * Real.sin a - y * Real.cos a = 2 * Real.sin a - Real.cos a) ∧
    (x - 3 * y + 13 = 0) ∧
    (5 ≤ x ∧ x ≤ 9) ∧
    (3 ≤ y ∧ y ≤ 7)
  ↔
    ∃ (k : ℤ), π/4 + k * π ≤ a ∧ a ≤ Real.arctan (5/3) + k * π :=
by sorry

end solution_exists_iff_a_in_interval_l2451_245127


namespace teal_more_blue_l2451_245140

/-- The number of people surveyed -/
def total_surveyed : ℕ := 150

/-- The number of people who believe teal is "more green" -/
def more_green : ℕ := 90

/-- The number of people who believe teal is both "more green" and "more blue" -/
def both : ℕ := 35

/-- The number of people who believe teal is neither "more green" nor "more blue" -/
def neither : ℕ := 25

/-- The theorem stating that 70 people believe teal is "more blue" -/
theorem teal_more_blue : 
  ∃ (more_blue : ℕ), more_blue = 70 ∧ 
  more_blue + (more_green - both) + both + neither = total_surveyed :=
sorry

end teal_more_blue_l2451_245140


namespace tim_score_is_38000_l2451_245162

/-- The value of a single line in points -/
def single_line_value : ℕ := 1000

/-- The value of a tetris in points -/
def tetris_value : ℕ := 8 * single_line_value

/-- The number of singles Tim scored -/
def tim_singles : ℕ := 6

/-- The number of tetrises Tim scored -/
def tim_tetrises : ℕ := 4

/-- Tim's total score -/
def tim_total_score : ℕ := tim_singles * single_line_value + tim_tetrises * tetris_value

theorem tim_score_is_38000 : tim_total_score = 38000 := by
  sorry

end tim_score_is_38000_l2451_245162


namespace choir_arrangement_min_choir_members_l2451_245132

theorem choir_arrangement (n : ℕ) : 
  (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) → n ≥ 990 :=
by sorry

theorem min_choir_members : 
  ∃ (n : ℕ), n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n = 990 :=
by sorry

end choir_arrangement_min_choir_members_l2451_245132


namespace tan_sum_equals_one_l2451_245186

theorem tan_sum_equals_one (α β : Real) 
  (h1 : Real.tan (α + π/6) = 1/2) 
  (h2 : Real.tan (β - π/6) = 1/3) : 
  Real.tan (α + β) = 1 := by
sorry

end tan_sum_equals_one_l2451_245186


namespace bonnets_theorem_l2451_245167

def bonnets_problem (monday thursday friday : ℕ) : Prop :=
  let tuesday_wednesday := 2 * monday
  let total_mon_to_thu := monday + tuesday_wednesday + thursday
  let total_sent := 11 * 5
  thursday = monday + 5 ∧
  total_sent = total_mon_to_thu + friday ∧
  thursday - friday = 5

theorem bonnets_theorem : 
  ∃ (monday thursday friday : ℕ), 
    monday = 10 ∧ 
    bonnets_problem monday thursday friday :=
sorry

end bonnets_theorem_l2451_245167


namespace shadows_parallel_l2451_245134

-- Define a structure for a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a structure for a point on a projection plane
structure ProjectedPoint where
  x : ℝ
  y : ℝ

-- Define a structure for a light source (parallel lighting)
structure ParallelLight where
  direction : Point3D

-- Define a function to project a 3D point onto a plane
def project (p : Point3D) (plane : ℝ) (light : ParallelLight) : ProjectedPoint :=
  sorry

-- Define a function to check if two line segments are parallel
def areParallel (p1 p2 q1 q2 : ProjectedPoint) : Prop :=
  sorry

-- Theorem statement
theorem shadows_parallel 
  (A B C : Point3D) 
  (plane1 plane2 : ℝ) 
  (light : ParallelLight) :
  let A1 := project A plane1 light
  let A2 := project A plane2 light
  let B1 := project B plane1 light
  let B2 := project B plane2 light
  let C1 := project C plane1 light
  let C2 := project C plane2 light
  areParallel A1 A2 B1 B2 ∧ areParallel B1 B2 C1 C2 :=
sorry

end shadows_parallel_l2451_245134


namespace systematic_sampling_theorem_l2451_245183

/-- Represents the systematic sampling problem -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  sampling_interval : ℕ
  first_random_number : ℕ

/-- Calculates the number of selected students within a given range -/
def selected_students_in_range (s : SystematicSampling) (lower : ℕ) (upper : ℕ) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.total_students = 1000)
  (h2 : s.sample_size = 50)
  (h3 : s.sampling_interval = 20)
  (h4 : s.first_random_number = 15) :
  selected_students_in_range s 601 785 = 9 := by
  sorry

end systematic_sampling_theorem_l2451_245183


namespace symmetry_probability_l2451_245157

/-- Represents a point on the grid -/
structure GridPoint where
  x : Fin 11
  y : Fin 11

/-- The center point of the grid -/
def centerPoint : GridPoint := ⟨5, 5⟩

/-- Checks if a point is on a line of symmetry -/
def isOnLineOfSymmetry (p : GridPoint) : Bool :=
  p.x = 5 ∨ p.y = 5 ∨ p.x = p.y ∨ p.x + p.y = 10

/-- The total number of points in the grid -/
def totalPoints : Nat := 121

/-- The number of points on lines of symmetry, excluding the center -/
def symmetryPoints : Nat := 40

/-- Theorem stating the probability of selecting a point on a line of symmetry -/
theorem symmetry_probability :
  (symmetryPoints : ℚ) / (totalPoints - 1 : ℚ) = 1 / 3 := by
  sorry


end symmetry_probability_l2451_245157


namespace fraction_simplification_l2451_245104

theorem fraction_simplification (x : ℝ) : (3*x - 4)/4 + (5 - 2*x)/3 = (x + 8)/12 := by
  sorry

end fraction_simplification_l2451_245104


namespace quadratic_perfect_square_l2451_245177

theorem quadratic_perfect_square (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 200*x + c = (x + a)^2) → c = 10000 := by
  sorry

end quadratic_perfect_square_l2451_245177


namespace min_rain_day4_overflow_l2451_245111

/-- Represents the rainstorm scenario -/
structure RainstormScenario where
  capacity : ℝ  -- capacity in feet
  drain_rate : ℝ  -- drain rate in inches per day
  day1_rain : ℝ  -- rain on day 1 in inches
  days : ℕ  -- number of days
  overflow_day : ℕ  -- day when overflow occurs

/-- Calculates the minimum amount of rain on the last day to cause overflow -/
def min_rain_to_overflow (scenario : RainstormScenario) : ℝ :=
  sorry

/-- Theorem stating the minimum amount of rain on day 4 to cause overflow -/
theorem min_rain_day4_overflow (scenario : RainstormScenario) 
  (h1 : scenario.capacity = 6)
  (h2 : scenario.drain_rate = 3)
  (h3 : scenario.day1_rain = 10)
  (h4 : scenario.days = 4)
  (h5 : scenario.overflow_day = 4) :
  min_rain_to_overflow scenario = 4 :=
  sorry

end min_rain_day4_overflow_l2451_245111


namespace train_distance_problem_l2451_245103

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 16) (h2 : v2 = 21) (h3 : d = 60) :
  let t := d / (v2 - v1)
  let d1 := v1 * t
  let d2 := v2 * t
  d1 + d2 = 444 := by sorry

end train_distance_problem_l2451_245103


namespace carpet_length_l2451_245131

/-- Given a rectangular carpet with width 4 feet covering an entire room floor of area 60 square feet, 
    prove that the length of the carpet is 15 feet. -/
theorem carpet_length (carpet_width : ℝ) (room_area : ℝ) (h1 : carpet_width = 4) (h2 : room_area = 60) :
  room_area / carpet_width = 15 := by
  sorry

end carpet_length_l2451_245131


namespace hyperbola_equation_l2451_245188

/-- Given a hyperbola with the general equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    one asymptote passing through the point (2, √3),
    and one focus lying on the directrix of the parabola y² = 4√7x,
    prove that the specific equation of the hyperbola is x²/4 - y²/3 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote_condition : b / a = Real.sqrt 3 / 2)
  (focus_condition : ∃ (x y : ℝ), x = -Real.sqrt 7 ∧ x^2 / a^2 - y^2 / b^2 = 1) :
  a^2 = 4 ∧ b^2 = 3 := by sorry

end hyperbola_equation_l2451_245188


namespace geometric_sequence_common_ratio_l2451_245129

/-- Given a geometric sequence with first term a₁ and common ratio q,
    if S₁, S₃, and 2a₃ form an arithmetic sequence, then q = -1/2 -/
theorem geometric_sequence_common_ratio
  (a₁ : ℝ) (q : ℝ) (S₁ S₃ : ℝ)
  (h₁ : S₁ = a₁)
  (h₂ : S₃ = a₁ + a₁ * q + a₁ * q^2)
  (h₃ : 2 * S₃ = S₁ + 2 * a₁ * q^2)
  (h₄ : a₁ ≠ 0) :
  q = -1/2 := by sorry

end geometric_sequence_common_ratio_l2451_245129


namespace chris_initial_money_l2451_245195

def chris_money_problem (initial_money : ℕ) : Prop :=
  let grandmother_gift : ℕ := 25
  let aunt_uncle_gift : ℕ := 20
  let parents_gift : ℕ := 75
  let total_after_gifts : ℕ := 279
  initial_money + grandmother_gift + aunt_uncle_gift + parents_gift = total_after_gifts

theorem chris_initial_money :
  ∃ (initial_money : ℕ), chris_money_problem initial_money ∧ initial_money = 159 :=
by
  sorry

end chris_initial_money_l2451_245195


namespace ab_is_zero_l2451_245158

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Given complex equation -/
def complex_equation (a b : ℝ) : Prop :=
  (1 + i) / (1 - i) = (a : ℂ) + b * i

/-- Theorem stating that if the complex equation holds, then ab = 0 -/
theorem ab_is_zero (a b : ℝ) (h : complex_equation a b) : a * b = 0 := by
  sorry

end ab_is_zero_l2451_245158


namespace arctan_tan_equation_solution_l2451_245185

theorem arctan_tan_equation_solution :
  ∃! x : ℝ, -2*π/3 ≤ x ∧ x ≤ 2*π/3 ∧ Real.arctan (Real.tan x) = 3*x/4 ∧ x = 0 := by
  sorry

end arctan_tan_equation_solution_l2451_245185


namespace zeros_before_first_nonzero_digit_l2451_245126

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : d > 0) :
  let decimal := (n : ℚ) / d
  let whole_part := (decimal.floor : ℤ)
  let fractional_part := decimal - whole_part
  let expanded := fractional_part * (10 ^ 10 : ℚ)  -- Multiply by a large power of 10 to see digits
  ∃ k, 0 < k ∧ k ≤ 10 ∧ (expanded.floor : ℤ) % (10 ^ k) ≠ 0 ∧
      ∀ j, 0 < j ∧ j < k → (expanded.floor : ℤ) % (10 ^ j) = 0 →
  (n = 5 ∧ d = 3125) → k - 1 = 2 :=
by sorry

end zeros_before_first_nonzero_digit_l2451_245126


namespace solution_to_equation_l2451_245189

theorem solution_to_equation (x y : ℝ) :
  4 * x^2 * y^2 = 4 * x * y + 3 ↔ y = 3 / (2 * x) ∨ y = -1 / (2 * x) :=
sorry

end solution_to_equation_l2451_245189


namespace sum_of_ages_l2451_245198

/-- Given that Ann is 5 years older than Susan and Ann is 16 years old, 
    prove that the sum of their ages is 27 years. -/
theorem sum_of_ages (ann_age susan_age : ℕ) : 
  ann_age = 16 → 
  ann_age = susan_age + 5 → 
  ann_age + susan_age = 27 := by
sorry

end sum_of_ages_l2451_245198


namespace unique_integer_satisfying_conditions_l2451_245174

theorem unique_integer_satisfying_conditions (x : ℤ) 
  (h1 : 1 < x ∧ x < 9)
  (h2 : 2 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 7)
  (h4 : 0 < x ∧ x < 4)
  (h5 : x + 1 < 5) : 
  x = 3 := by
sorry

end unique_integer_satisfying_conditions_l2451_245174


namespace daragh_favorite_bears_l2451_245122

/-- The number of stuffed bears Daragh had initially -/
def initial_bears : ℕ := 20

/-- The number of sisters Daragh divided bears among -/
def num_sisters : ℕ := 3

/-- The number of bears Eden had before receiving more -/
def eden_initial_bears : ℕ := 10

/-- The number of bears Eden has after receiving more -/
def eden_final_bears : ℕ := 14

/-- The number of favorite stuffed bears Daragh took out -/
def favorite_bears : ℕ := initial_bears - (eden_final_bears - eden_initial_bears) * num_sisters

theorem daragh_favorite_bears :
  favorite_bears = 8 :=
by sorry

end daragh_favorite_bears_l2451_245122


namespace marthas_cat_rats_l2451_245163

theorem marthas_cat_rats (R : ℕ) : 
  (5 * (R + 7) - 3 = 47) → R = 3 := by
  sorry

end marthas_cat_rats_l2451_245163


namespace tangent_line_at_negative_one_unique_a_for_inequality_l2451_245152

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x

-- Part I
theorem tangent_line_at_negative_one (h : f 1 (-1) = 0) :
  ∃ (m : ℝ), ∀ (x : ℝ), f 1 x = m * (x + 1) + f 1 (-1) ∧ m = -2 :=
sorry

-- Part II
theorem unique_a_for_inequality (h : ∀ x, x ∈ Set.Icc 0 1 → (1/4 * x - 1/4 ≤ f 1 x ∧ f 1 x ≤ 1/4 * x + 1/4)) :
  ∀ a > 0, (∀ x, x ∈ Set.Icc 0 1 → (1/4 * x - 1/4 ≤ f a x ∧ f a x ≤ 1/4 * x + 1/4)) ↔ a = 1 :=
sorry

end tangent_line_at_negative_one_unique_a_for_inequality_l2451_245152


namespace tabithas_final_amount_l2451_245145

/-- Calculates Tabitha's remaining money after various transactions --/
def tabithas_remaining_money (initial_amount : ℚ) (given_to_mom : ℚ) (num_items : ℕ) (item_cost : ℚ) : ℚ :=
  let after_mom := initial_amount - given_to_mom
  let after_investment := after_mom / 2
  let spent_on_items := num_items * item_cost
  after_investment - spent_on_items

/-- Theorem stating that Tabitha's remaining money is 6 dollars --/
theorem tabithas_final_amount :
  tabithas_remaining_money 25 8 5 (1/2) = 6 := by
  sorry


end tabithas_final_amount_l2451_245145


namespace differential_of_exponential_trig_function_l2451_245191

/-- The differential of y = e^x(cos 2x + 2sin 2x) is dy = 5 e^x cos 2x · dx -/
theorem differential_of_exponential_trig_function (x : ℝ) :
  let y : ℝ → ℝ := λ x => Real.exp x * (Real.cos (2 * x) + 2 * Real.sin (2 * x))
  (deriv y) x = 5 * Real.exp x * Real.cos (2 * x) := by
  sorry

end differential_of_exponential_trig_function_l2451_245191


namespace johns_run_l2451_245120

/-- Theorem: John's total distance traveled is 5 miles -/
theorem johns_run (solo_speed : ℝ) (dog_speed : ℝ) (total_time : ℝ) (dog_time : ℝ) :
  solo_speed = 4 →
  dog_speed = 6 →
  total_time = 1 →
  dog_time = 0.5 →
  dog_speed * dog_time + solo_speed * (total_time - dog_time) = 5 := by
  sorry

#check johns_run

end johns_run_l2451_245120


namespace equal_numbers_product_l2451_245178

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 17.6 →
  a = 15 →
  b = 20 →
  c = 22 →
  d = e →
  d * e = 240.25 := by
sorry

end equal_numbers_product_l2451_245178


namespace function_composition_inverse_l2451_245197

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -5 * x + 3
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem function_composition_inverse (a b : ℝ) :
  (∀ x, h a b x = x - 9) → (a - b = 41 / 5) := by
  sorry

end function_composition_inverse_l2451_245197


namespace sarah_and_matt_age_sum_l2451_245139

/-- Given the age relationship between Sarah and Matt, prove that the sum of their current ages is 41 years. -/
theorem sarah_and_matt_age_sum :
  ∀ (sarah_age matt_age : ℝ),
  sarah_age = matt_age + 8 →
  sarah_age + 10 = 3 * (matt_age - 5) →
  sarah_age + matt_age = 41 :=
by
  sorry

end sarah_and_matt_age_sum_l2451_245139


namespace billy_experiment_result_l2451_245151

/-- Represents the mouse population dynamics in Billy's experiment --/
structure MousePopulation where
  initial_mice : ℕ
  pups_per_mouse : ℕ
  final_population : ℕ

/-- Calculates the number of pups eaten per adult mouse --/
def pups_eaten_per_adult (pop : MousePopulation) : ℕ :=
  let first_gen_total := pop.initial_mice + pop.initial_mice * pop.pups_per_mouse
  let second_gen_total := first_gen_total + first_gen_total * pop.pups_per_mouse
  let total_eaten := second_gen_total - pop.final_population
  total_eaten / first_gen_total

/-- Theorem stating that in Billy's experiment, each adult mouse ate 2 pups --/
theorem billy_experiment_result :
  let pop : MousePopulation := {
    initial_mice := 8,
    pups_per_mouse := 6,
    final_population := 280
  }
  pups_eaten_per_adult pop = 2 := by
  sorry


end billy_experiment_result_l2451_245151


namespace reverse_increase_l2451_245143

def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d3 * 100 + d2 * 10 + d1

theorem reverse_increase (n : ℕ) : 
  n = 253 → 
  (n / 100 + (n / 10) % 10 + n % 10 = 10) → 
  ((n / 10) % 10 = (n / 100 + n % 10)) → 
  reverse_number n - n = 99 :=
by sorry

end reverse_increase_l2451_245143


namespace hexagon_area_division_l2451_245124

/-- A hexagon constructed from unit squares -/
structure Hexagon :=
  (area : ℝ)
  (line_PQ : ℝ → ℝ)
  (area_below : ℝ)
  (area_above : ℝ)
  (XQ : ℝ)
  (QY : ℝ)

/-- The theorem statement -/
theorem hexagon_area_division (h : Hexagon) :
  h.area = 8 ∧
  h.area_below = h.area_above ∧
  h.area_below = 1 + (1/2 * 4 * (3/2)) ∧
  h.XQ + h.QY = 4 →
  h.XQ / h.QY = 2/3 :=
by sorry

end hexagon_area_division_l2451_245124


namespace residue_16_pow_3030_mod_23_l2451_245192

theorem residue_16_pow_3030_mod_23 : 16^3030 ≡ 1 [ZMOD 23] := by
  sorry

end residue_16_pow_3030_mod_23_l2451_245192


namespace olivia_remaining_money_l2451_245130

/-- Given an initial amount of money and an amount spent, 
    calculate the remaining amount. -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem stating that given an initial amount of 78 dollars 
    and a spent amount of 15 dollars, the remaining amount is 63 dollars. -/
theorem olivia_remaining_money :
  remaining_money 78 15 = 63 := by
  sorry

end olivia_remaining_money_l2451_245130


namespace radio_show_ad_break_duration_l2451_245160

theorem radio_show_ad_break_duration 
  (total_show_time : ℕ) 
  (talking_segment_duration : ℕ) 
  (num_talking_segments : ℕ) 
  (num_ad_breaks : ℕ) 
  (song_duration : ℕ) 
  (h1 : total_show_time = 3 * 60) 
  (h2 : talking_segment_duration = 10)
  (h3 : num_talking_segments = 3)
  (h4 : num_ad_breaks = 5)
  (h5 : song_duration = 125) : 
  (total_show_time - (num_talking_segments * talking_segment_duration) - song_duration) / num_ad_breaks = 5 := by
sorry

end radio_show_ad_break_duration_l2451_245160
