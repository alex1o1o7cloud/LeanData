import Mathlib

namespace coin_toss_probability_l2240_224069

theorem coin_toss_probability (p_heads : ℚ) (h1 : p_heads = 1/4) :
  1 - p_heads = 3/4 := by
  sorry

end coin_toss_probability_l2240_224069


namespace flight_750_male_first_class_fraction_l2240_224011

theorem flight_750_male_first_class_fraction 
  (total_passengers : ℕ) 
  (female_percentage : ℚ) 
  (first_class_percentage : ℚ) 
  (female_coach : ℕ) :
  total_passengers = 120 →
  female_percentage = 45/100 →
  first_class_percentage = 10/100 →
  female_coach = 46 →
  (total_passengers * first_class_percentage * (1 - female_percentage / (1 - first_class_percentage)) / 
   (total_passengers * first_class_percentage) : ℚ) = 1/3 := by
sorry

end flight_750_male_first_class_fraction_l2240_224011


namespace max_value_sqrt_sum_l2240_224006

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end max_value_sqrt_sum_l2240_224006


namespace unique_valid_set_l2240_224080

/-- The sum of n consecutive integers starting from a -/
def consecutiveSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A predicate that checks if a set of n consecutive integers starting from a sums to 30 -/
def isValidSet (a n : ℕ) : Prop :=
  a ≥ 3 ∧ n ≥ 2 ∧ consecutiveSum a n = 30

/-- The main theorem stating there is exactly one valid set -/
theorem unique_valid_set : ∃! p : ℕ × ℕ, isValidSet p.1 p.2 := by sorry

end unique_valid_set_l2240_224080


namespace line_perpendicular_to_plane_l2240_224090

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between planes
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α β : Plane) 
  (h1 : perpendicular_planes α β)
  (h2 : intersection α β = m)
  (h3 : subset n α) :
  perpendicular_line_plane n β ↔ perpendicular_lines n m :=
sorry

end line_perpendicular_to_plane_l2240_224090


namespace sqrt_one_third_equals_sqrt_three_over_three_l2240_224034

theorem sqrt_one_third_equals_sqrt_three_over_three :
  Real.sqrt (1 / 3) = Real.sqrt 3 / 3 := by sorry

end sqrt_one_third_equals_sqrt_three_over_three_l2240_224034


namespace min_value_zero_l2240_224038

/-- The expression for which we want to find the minimum value -/
def f (k x y : ℝ) : ℝ := 9*x^2 - 12*k*x*y + (4*k^2 + 3)*y^2 - 6*x - 6*y + 9

/-- The theorem stating the condition for the minimum value of f to be 0 -/
theorem min_value_zero (k : ℝ) :
  (∀ x y : ℝ, f k x y ≥ 0) ∧ (∃ x y : ℝ, f k x y = 0) ↔ k = 4/3 := by sorry

end min_value_zero_l2240_224038


namespace sum_of_permutations_divisible_by_digit_sum_l2240_224073

/-- A type representing a digit from 1 to 9 -/
def Digit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- A function to calculate the sum of all permutations of a five-digit number -/
def sumOfPermutations (a b c d e : Digit) : ℕ :=
  24 * 11111 * (a.val + b.val + c.val + d.val + e.val)

/-- The theorem statement -/
theorem sum_of_permutations_divisible_by_digit_sum 
  (a b c d e : Digit) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
                b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
                c ≠ d ∧ c ≠ e ∧ 
                d ≠ e) : 
  (sumOfPermutations a b c d e) % (a.val + b.val + c.val + d.val + e.val) = 0 := by
  sorry

end sum_of_permutations_divisible_by_digit_sum_l2240_224073


namespace double_root_condition_l2240_224071

/-- The equation has a double root when k is either 3 or 1/3 -/
theorem double_root_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 1) / (x^2 - 1) - 1 / (x - 1) = k / (x + 1) ∧ 
   ∀ y : ℝ, (k - 1) / (y^2 - 1) - 1 / (y - 1) = k / (y + 1) → y = x) ↔ 
  (k = 3 ∨ k = 1/3) :=
sorry

end double_root_condition_l2240_224071


namespace largest_divisor_of_n_squared_divisible_by_72_l2240_224044

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) :
  ∃ q : ℕ, q > 0 ∧ q ∣ n ∧ ∀ m : ℕ, m > 0 ∧ m ∣ n → m ≤ q ∧ q = 12 :=
by sorry

end largest_divisor_of_n_squared_divisible_by_72_l2240_224044


namespace conditional_probability_B_given_A_l2240_224078

-- Define the sample space
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define events A and B
def A : Finset Nat := {2, 3, 5}
def B : Finset Nat := {1, 2, 4, 5, 6}

-- Define the probability measure
noncomputable def P (S : Finset Nat) : ℝ := (S.card : ℝ) / (Ω.card : ℝ)

-- State the theorem
theorem conditional_probability_B_given_A : P (A ∩ B) / P A = 2/3 := by
  sorry

end conditional_probability_B_given_A_l2240_224078


namespace miss_adamson_paper_usage_l2240_224016

/-- Calculates the total number of sheets of paper used by a teacher for all students --/
def total_sheets_of_paper (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) : ℕ :=
  num_classes * students_per_class * sheets_per_student

/-- Proves that Miss Adamson will use 400 sheets of paper for all her students --/
theorem miss_adamson_paper_usage :
  total_sheets_of_paper 4 20 5 = 400 := by
  sorry

#eval total_sheets_of_paper 4 20 5

end miss_adamson_paper_usage_l2240_224016


namespace first_number_is_1841_l2240_224028

/-- Represents one operation of replacing the first number with the average of the other two -/
def operation (x y z : ℤ) : ℤ × ℤ × ℤ := (y, z, (y + z) / 2)

/-- Applies the operation n times -/
def apply_operations (n : ℕ) (x y z : ℤ) : ℤ × ℤ × ℤ :=
  match n with
  | 0 => (x, y, z)
  | n + 1 => 
    let (a, b, c) := apply_operations n x y z
    operation a b c

theorem first_number_is_1841 (a b c : ℤ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ -- all numbers are positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ -- all numbers are different
  a + b + c = 2013 ∧ -- initial sum
  (let (x, y, z) := apply_operations 7 a b c; x + y + z = 195) → -- sum after 7 operations
  a = 1841 := by sorry

end first_number_is_1841_l2240_224028


namespace final_daisy_count_l2240_224020

/-- Represents the number of flowers in Laura's garden -/
structure GardenFlowers where
  daisies : ℕ
  tulips : ℕ

/-- Represents the ratio of daisies to tulips -/
structure FlowerRatio where
  daisy : ℕ
  tulip : ℕ

/-- Theorem stating the final number of daisies after adding tulips while maintaining the ratio -/
theorem final_daisy_count 
  (initial : GardenFlowers) 
  (ratio : FlowerRatio) 
  (added_tulips : ℕ) : 
  (ratio.daisy : ℚ) / (ratio.tulip : ℚ) = (initial.daisies : ℚ) / (initial.tulips : ℚ) →
  initial.tulips = 32 →
  added_tulips = 24 →
  ratio.daisy = 3 →
  ratio.tulip = 4 →
  let final_tulips := initial.tulips + added_tulips
  let final_daisies := (ratio.daisy : ℚ) / (ratio.tulip : ℚ) * final_tulips
  final_daisies = 42 := by
  sorry


end final_daisy_count_l2240_224020


namespace quadratic_function_general_form_l2240_224076

/-- A quadratic function with the same shape as y = 5x² and vertex at (3, 7) -/
def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, 
    (a = 5 ∨ a = -5) ∧
    (∀ x : ℝ, f x = a * (x - 3)^2 + 7)

theorem quadratic_function_general_form (f : ℝ → ℝ) 
  (h : quadratic_function f) :
  (∀ x : ℝ, f x = 5 * x^2 - 30 * x + 52) ∨
  (∀ x : ℝ, f x = -5 * x^2 + 30 * x - 38) :=
sorry

end quadratic_function_general_form_l2240_224076


namespace least_common_period_is_36_l2240_224019

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least common positive period for all functions satisfying the equation -/
def LeastCommonPeriod (p : ℝ) : Prop :=
  p > 0 ∧
  (∀ f : ℝ → ℝ, SatisfiesEquation f → IsPeriod f p) ∧
  (∀ q : ℝ, q > 0 → (∀ f : ℝ → ℝ, SatisfiesEquation f → IsPeriod f q) → p ≤ q)

theorem least_common_period_is_36 :
  LeastCommonPeriod 36 := by sorry

end least_common_period_is_36_l2240_224019


namespace students_taking_both_music_and_art_l2240_224025

theorem students_taking_both_music_and_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (neither : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 50) 
  (h3 : art = 20) 
  (h4 : neither = 440) : 
  total - neither - (music + art - (total - neither)) = 10 := by
  sorry

end students_taking_both_music_and_art_l2240_224025


namespace tax_center_revenue_l2240_224097

/-- Calculates the total revenue for a tax center based on the number and types of returns sold --/
theorem tax_center_revenue (federal_price state_price quarterly_price : ℕ)
                           (federal_sold state_sold quarterly_sold : ℕ) :
  federal_price = 50 →
  state_price = 30 →
  quarterly_price = 80 →
  federal_sold = 60 →
  state_sold = 20 →
  quarterly_sold = 10 →
  federal_price * federal_sold + state_price * state_sold + quarterly_price * quarterly_sold = 4400 :=
by sorry

end tax_center_revenue_l2240_224097


namespace stratified_sampling_probability_l2240_224070

theorem stratified_sampling_probability (students teachers support_staff sample_size : ℕ) 
  (h1 : students = 2500)
  (h2 : teachers = 350)
  (h3 : support_staff = 150)
  (h4 : sample_size = 300) :
  (sample_size * students) / ((students + teachers + support_staff) * students) = 1 / 10 := by
sorry

end stratified_sampling_probability_l2240_224070


namespace construction_costs_l2240_224096

/-- Calculate the total construction costs for a house project. -/
theorem construction_costs
  (land_cost_per_sqm : ℝ)
  (brick_cost_per_1000 : ℝ)
  (tile_cost_per_tile : ℝ)
  (land_area : ℝ)
  (brick_count : ℝ)
  (tile_count : ℝ)
  (h1 : land_cost_per_sqm = 50)
  (h2 : brick_cost_per_1000 = 100)
  (h3 : tile_cost_per_tile = 10)
  (h4 : land_area = 2000)
  (h5 : brick_count = 10000)
  (h6 : tile_count = 500) :
  land_cost_per_sqm * land_area +
  brick_cost_per_1000 * (brick_count / 1000) +
  tile_cost_per_tile * tile_count = 106000 := by
  sorry


end construction_costs_l2240_224096


namespace discount_comparison_l2240_224000

def original_price : ℝ := 15000

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def scheme1_discounts : List ℝ := [0.25, 0.15, 0.10]
def scheme2_discounts : List ℝ := [0.30, 0.10, 0.05]

def apply_scheme (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem discount_comparison :
  apply_scheme original_price scheme1_discounts - apply_scheme original_price scheme2_discounts = 371.25 := by
  sorry

end discount_comparison_l2240_224000


namespace simplify_expression_find_a_value_independence_condition_l2240_224051

-- Define A and B as functions of a and b
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b - 1

-- Theorem 1: Simplification of 4A - (3A - 2B)
theorem simplify_expression (a b : ℝ) :
  4 * A a b - (3 * A a b - 2 * B a b) = 5 * a * b - 2 * a - 3 := by sorry

-- Theorem 2: Value of a when b = 1 and 4A - (3A - 2B) = b - 2a
theorem find_a_value (a : ℝ) :
  (4 * A a 1 - (3 * A a 1 - 2 * B a 1) = 1 - 2 * a) → a = 4/5 := by sorry

-- Theorem 3: A + 2B is independent of a iff b = 2/5
theorem independence_condition (b : ℝ) :
  (∀ a₁ a₂ : ℝ, A a₁ b + 2 * B a₁ b = A a₂ b + 2 * B a₂ b) ↔ b = 2/5 := by sorry

end simplify_expression_find_a_value_independence_condition_l2240_224051


namespace sum_of_fractions_l2240_224018

theorem sum_of_fractions : (1/2 : ℚ) + 2/4 + 4/8 + 8/16 = 2 := by
  sorry

end sum_of_fractions_l2240_224018


namespace max_triangles_is_eleven_l2240_224092

/-- Represents an equilateral triangle with a line segment connecting the midpoints of two sides -/
structure EquilateralTriangleWithMidline where
  side_length : ℝ
  midline_position : ℝ

/-- Represents the configuration of two overlapping equilateral triangles -/
structure OverlappingTriangles where
  triangle_a : EquilateralTriangleWithMidline
  triangle_b : EquilateralTriangleWithMidline
  overlap_distance : ℝ

/-- Counts the number of triangles formed in a given configuration -/
def count_triangles (config : OverlappingTriangles) : ℕ :=
  sorry

/-- Finds the maximum number of triangles formed during the overlap process -/
def max_triangles (triangle : EquilateralTriangleWithMidline) : ℕ :=
  sorry

/-- Main theorem: The maximum number of triangles formed is 11 -/
theorem max_triangles_is_eleven (triangle : EquilateralTriangleWithMidline) :
  max_triangles triangle = 11 :=
sorry

end max_triangles_is_eleven_l2240_224092


namespace new_supervisor_salary_is_960_l2240_224043

/-- Represents the monthly salary structure of a factory -/
structure FactorySalary where
  initial_avg : ℝ
  old_supervisor_salary : ℝ
  old_supervisor_bonus_rate : ℝ
  worker_increment_rate : ℝ
  old_supervisor_increment_rate : ℝ
  new_avg : ℝ
  new_supervisor_bonus_rate : ℝ
  new_supervisor_increment_rate : ℝ

/-- Calculates the new supervisor's monthly salary -/
def calculate_new_supervisor_salary (fs : FactorySalary) : ℝ :=
  sorry

/-- Theorem stating that given the factory salary conditions, 
    the new supervisor's monthly salary is $960 -/
theorem new_supervisor_salary_is_960 (fs : FactorySalary) 
  (h1 : fs.initial_avg = 430)
  (h2 : fs.old_supervisor_salary = 870)
  (h3 : fs.old_supervisor_bonus_rate = 0.05)
  (h4 : fs.worker_increment_rate = 0.03)
  (h5 : fs.old_supervisor_increment_rate = 0.04)
  (h6 : fs.new_avg = 450)
  (h7 : fs.new_supervisor_bonus_rate = 0.03)
  (h8 : fs.new_supervisor_increment_rate = 0.035) :
  calculate_new_supervisor_salary fs = 960 :=
sorry

end new_supervisor_salary_is_960_l2240_224043


namespace paul_lives_on_fifth_story_l2240_224084

/-- The number of stories in Paul's apartment building -/
def S : ℕ := sorry

/-- The number of trips Paul makes each day -/
def trips_per_day : ℕ := 3

/-- The height of each story in feet -/
def story_height : ℕ := 10

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total vertical distance Paul travels in a week in feet -/
def total_vertical_distance : ℕ := 2100

theorem paul_lives_on_fifth_story :
  S * story_height * trips_per_day * 2 * days_in_week = total_vertical_distance →
  S = 5 := by
  sorry

end paul_lives_on_fifth_story_l2240_224084


namespace maria_josh_age_sum_l2240_224059

/-- Proves that given the conditions about Maria and Josh's ages, the sum of their current ages is 31 years -/
theorem maria_josh_age_sum : 
  ∀ (maria josh : ℝ), 
  (maria = josh + 8) → 
  (maria + 6 = 3 * (josh - 3)) → 
  (maria + josh = 31) := by
sorry

end maria_josh_age_sum_l2240_224059


namespace seating_arrangements_count_l2240_224055

/-- Represents a theater with two rows of seats. -/
structure Theater :=
  (front_seats : Nat)
  (back_seats : Nat)

/-- Calculates the number of valid seating arrangements for two people in a theater. -/
def validArrangements (t : Theater) (middle_seats : Nat) : Nat :=
  sorry

/-- The theorem stating that the number of valid seating arrangements is 114. -/
theorem seating_arrangements_count (t : Theater) :
  t.front_seats = 9 ∧ t.back_seats = 8 →
  validArrangements t 3 = 114 :=
by sorry

end seating_arrangements_count_l2240_224055


namespace stamp_revenue_calculation_l2240_224074

/-- The total revenue generated from stamp sales --/
theorem stamp_revenue_calculation : 
  let color_price : ℚ := 15/100
  let bw_price : ℚ := 10/100
  let color_sold : ℕ := 578833
  let bw_sold : ℕ := 523776
  let total_revenue := (color_price * color_sold) + (bw_price * bw_sold)
  total_revenue = 139202551/10000 := by
  sorry

end stamp_revenue_calculation_l2240_224074


namespace max_m_inequality_l2240_224052

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2 / a + 1 / b = 1 / 4) : 
  (∃ (m : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → 2 / x + 1 / y = 1 / 4 → 2*x + y ≥ 4*m) ∧ 
               (∀ (ε : ℝ), ε > 0 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 / x + 1 / y = 1 / 4 ∧ 2*x + y < 4*(m + ε))) ∧
  (∀ (n : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → 2 / x + 1 / y = 1 / 4 → 2*x + y ≥ 4*n) → n ≤ m) ∧
  m = 9 :=
sorry

end max_m_inequality_l2240_224052


namespace construction_material_order_l2240_224058

theorem construction_material_order (concrete bricks stone total : ℝ) : 
  concrete = 0.17 →
  bricks = 0.17 →
  stone = 0.5 →
  total = concrete + bricks + stone →
  total = 0.84 := by
sorry

end construction_material_order_l2240_224058


namespace sufficient_not_necessary_condition_l2240_224031

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x = -1 → x^2 - 5*x - 6 = 0) ∧ 
  ¬(x^2 - 5*x - 6 = 0 → x = -1) :=
by sorry

end sufficient_not_necessary_condition_l2240_224031


namespace sum_1423_9_and_711_9_in_base3_l2240_224023

/-- Converts a number from base 9 to base 10 -/
def base9To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 3 -/
def base10To3 (n : ℕ) : ℕ := sorry

/-- The sum of 1423 in base 9 and 711 in base 9, converted to base 3 -/
def sumInBase3 : ℕ := base10To3 (base9To10 1423 + base9To10 711)

theorem sum_1423_9_and_711_9_in_base3 :
  sumInBase3 = 2001011 := by sorry

end sum_1423_9_and_711_9_in_base3_l2240_224023


namespace candy_distribution_l2240_224036

theorem candy_distribution (initial_candies : ℕ) (friends : ℕ) (additional_candies : ℕ) :
  initial_candies = 20 →
  friends = 6 →
  additional_candies = 4 →
  (initial_candies + additional_candies) / friends = 4 := by
sorry

end candy_distribution_l2240_224036


namespace arun_weight_upper_limit_l2240_224099

theorem arun_weight_upper_limit (arun_lower : ℝ) (arun_upper : ℝ) 
  (brother_lower : ℝ) (brother_upper : ℝ) (mother_upper : ℝ) (average : ℝ) :
  arun_lower = 61 →
  arun_upper = 72 →
  brother_lower = 60 →
  brother_upper = 70 →
  average = 63 →
  arun_lower < mother_upper →
  mother_upper ≤ brother_upper →
  (arun_lower + mother_upper) / 2 = average →
  mother_upper = 65 := by
sorry

end arun_weight_upper_limit_l2240_224099


namespace more_girls_than_boys_l2240_224027

theorem more_girls_than_boys (girls boys : ℝ) (h1 : girls = 542.0) (h2 : boys = 387.0) :
  girls - boys = 155.0 := by
  sorry

end more_girls_than_boys_l2240_224027


namespace tank_filling_time_l2240_224086

theorem tank_filling_time (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a = 60 → (15 / b + 15 * (1 / 60 + 1 / b) = 1) → b = 40 := by sorry

end tank_filling_time_l2240_224086


namespace supplement_of_complement_of_63_degree_l2240_224072

def complement (α : ℝ) : ℝ := 90 - α

def supplement (β : ℝ) : ℝ := 180 - β

theorem supplement_of_complement_of_63_degree :
  supplement (complement 63) = 153 := by sorry

end supplement_of_complement_of_63_degree_l2240_224072


namespace mean_of_seven_numbers_l2240_224049

theorem mean_of_seven_numbers (x y : ℝ) :
  (6 + 14 + x + 17 + 9 + y + 10) / 7 = 13 → x + y = 35 := by
sorry

end mean_of_seven_numbers_l2240_224049


namespace onion_harvest_bags_per_trip_l2240_224041

/-- Calculates the number of bags carried per trip given the total harvest weight,
    weight per bag, and number of trips. -/
def bagsPerTrip (totalHarvest : ℕ) (weightPerBag : ℕ) (numTrips : ℕ) : ℕ :=
  (totalHarvest / weightPerBag) / numTrips

/-- Theorem stating that given the specific conditions of Titan's father's onion harvest,
    the number of bags carried per trip is 10. -/
theorem onion_harvest_bags_per_trip :
  bagsPerTrip 10000 50 20 = 10 := by
  sorry

#eval bagsPerTrip 10000 50 20

end onion_harvest_bags_per_trip_l2240_224041


namespace back_seat_tickets_sold_l2240_224024

/-- Proves the number of back seat tickets sold at a concert --/
theorem back_seat_tickets_sold (total_seats : ℕ) (main_price back_price : ℕ) (total_revenue : ℕ) :
  total_seats = 20000 →
  main_price = 55 →
  back_price = 45 →
  total_revenue = 955000 →
  ∃ (main_seats back_seats : ℕ),
    main_seats + back_seats = total_seats ∧
    main_price * main_seats + back_price * back_seats = total_revenue ∧
    back_seats = 14500 := by
  sorry

#check back_seat_tickets_sold

end back_seat_tickets_sold_l2240_224024


namespace inequality_not_always_hold_l2240_224015

theorem inequality_not_always_hold (a b : ℝ) (h : a > b ∧ b > 0) : 
  ¬ ∀ c : ℝ, a * c > b * c :=
sorry

end inequality_not_always_hold_l2240_224015


namespace sequence_formula_correct_l2240_224061

def a (n : ℕ) : ℤ := (-1)^n * (4*n - 1)

theorem sequence_formula_correct :
  (a 1 = -3) ∧ (a 2 = 7) ∧ (a 3 = -11) ∧ (a 4 = 15) :=
by sorry

end sequence_formula_correct_l2240_224061


namespace toothpicks_200th_stage_l2240_224033

def toothpicks (n : ℕ) : ℕ :=
  if n ≤ 49 then
    4 + 4 * (n - 1)
  else if n ≤ 99 then
    toothpicks 49 + 5 * (n - 49)
  else if n ≤ 149 then
    toothpicks 99 + 6 * (n - 99)
  else
    toothpicks 149 + 7 * (n - 149)

theorem toothpicks_200th_stage :
  toothpicks 200 = 1082 := by sorry

end toothpicks_200th_stage_l2240_224033


namespace positive_integers_satisfying_condition_l2240_224032

theorem positive_integers_satisfying_condition :
  ∀ n : ℕ+, (25 - 3 * n.val ≥ 4) ↔ n.val ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) := by
  sorry

end positive_integers_satisfying_condition_l2240_224032


namespace solution_set_quadratic_inequality_l2240_224075

theorem solution_set_quadratic_inequality :
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x < -1 ∨ x > 3/2} := by
  sorry

end solution_set_quadratic_inequality_l2240_224075


namespace difference_c_minus_a_l2240_224093

theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) : 
  c - a = 90 := by
sorry

end difference_c_minus_a_l2240_224093


namespace hyperbola_parabola_focus_l2240_224046

/-- The value of 'a' for a hyperbola with equation x^2 - y^2 = a^2 (a > 0) 
    whose right focus coincides with the focus of the parabola y^2 = 4x -/
theorem hyperbola_parabola_focus (a : ℝ) : a > 0 → 
  (∃ (x y : ℝ), x^2 - y^2 = a^2 ∧ y^2 = 4*x ∧ (x, y) = (1, 0)) → 
  a = Real.sqrt 2 / 2 := by
  sorry

end hyperbola_parabola_focus_l2240_224046


namespace min_value_p_l2240_224065

theorem min_value_p (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20)
  (prod_sq_eq : p^2 * q^2 * r^2 * s^2 = 16) :
  ∃ (min_p : ℝ), min_p = 2 ∧ p ≥ min_p := by
  sorry

end min_value_p_l2240_224065


namespace range_of_a_l2240_224067

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > a → x > 2) ∧ (∃ x : ℝ, x > 2 ∧ x ≤ a) ↔ a > 2 :=
by sorry

end range_of_a_l2240_224067


namespace chlorine_discount_is_20_percent_l2240_224056

def original_chlorine_price : ℝ := 10
def original_soap_price : ℝ := 16
def soap_discount : ℝ := 0.25
def chlorine_quantity : ℕ := 3
def soap_quantity : ℕ := 5
def total_savings : ℝ := 26

theorem chlorine_discount_is_20_percent :
  ∃ (chlorine_discount : ℝ),
    chlorine_discount = 0.20 ∧
    (chlorine_quantity : ℝ) * original_chlorine_price * (1 - chlorine_discount) +
    soap_quantity * original_soap_price * (1 - soap_discount) =
    chlorine_quantity * original_chlorine_price +
    soap_quantity * original_soap_price - total_savings :=
by sorry

end chlorine_discount_is_20_percent_l2240_224056


namespace semicircle_radius_l2240_224062

theorem semicircle_radius (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  (1/2) * Real.pi * (a/2)^2 = 8 * Real.pi →
  Real.pi * (b/2) = 8.5 * Real.pi →
  c/2 = 7.5 := by
sorry

end semicircle_radius_l2240_224062


namespace solution_set_inequality_l2240_224095

theorem solution_set_inequality (x : ℝ) : 
  (2*x - 1) / (3*x + 1) > 1 ↔ -2 < x ∧ x < -1/3 := by
  sorry

end solution_set_inequality_l2240_224095


namespace evaluate_expression_l2240_224013

theorem evaluate_expression (x : ℝ) (h : x = -3) :
  (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
sorry

end evaluate_expression_l2240_224013


namespace hash_example_l2240_224088

def hash (a b c d : ℝ) : ℝ := d * b^2 - 4 * a * c

theorem hash_example : hash 2 3 1 (1/2) = -3.5 := by
  sorry

end hash_example_l2240_224088


namespace tan_five_pi_fourths_l2240_224008

theorem tan_five_pi_fourths : Real.tan (5 * π / 4) = 1 := by
  sorry

end tan_five_pi_fourths_l2240_224008


namespace total_cards_proof_l2240_224022

/-- The number of people who have baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person has -/
def cards_per_person : ℕ := 6

/-- The total number of baseball cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem total_cards_proof : total_cards = 24 := by
  sorry

end total_cards_proof_l2240_224022


namespace lucky_number_property_l2240_224039

/-- A number is lucky if the sum of its digits is 7 -/
def IsLucky (n : ℕ) : Prop :=
  (n.digits 10).sum = 7

/-- The sequence of lucky numbers in ascending order -/
def LuckySequence : ℕ → ℕ :=
  sorry

theorem lucky_number_property (n : ℕ) :
  LuckySequence n = 2005 → LuckySequence (5 * n) = 30301 :=
by
  sorry

end lucky_number_property_l2240_224039


namespace magicians_marbles_l2240_224045

/-- The number of marbles left after the magician's trick --/
def marbles_left (red_initial blue_initial green_initial yellow_initial : ℕ) : ℕ :=
  let red_removed := red_initial / 4
  let blue_removed := 3 * (green_initial / 5)
  let green_removed := (green_initial * 3) / 10  -- 30% rounded down
  let yellow_removed := 25

  let red_left := red_initial - red_removed
  let blue_left := blue_initial - blue_removed
  let green_left := green_initial - green_removed
  let yellow_left := yellow_initial - yellow_removed

  red_left + blue_left + green_left + yellow_left

/-- Theorem stating that given the initial conditions, the number of marbles left is 213 --/
theorem magicians_marbles :
  marbles_left 80 120 75 50 = 213 :=
by sorry

end magicians_marbles_l2240_224045


namespace product_of_reciprocal_differences_l2240_224048

theorem product_of_reciprocal_differences (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end product_of_reciprocal_differences_l2240_224048


namespace problem_1_problem_2_l2240_224047

theorem problem_1 : (1) - 1/2 / 3 * (3 - (-3)^2) = 1 := by sorry

theorem problem_2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) : 
  2*x / (x^2 - 4) - 1 / (x - 2) = 1 / (x + 2) := by sorry

end problem_1_problem_2_l2240_224047


namespace equation_one_solutions_equation_two_solution_l2240_224005

-- Problem 1
theorem equation_one_solutions (x : ℝ) :
  4 * x^2 - 16 = 0 ↔ x = 2 ∨ x = -2 :=
sorry

-- Problem 2
theorem equation_two_solution (x : ℝ) :
  (2*x - 1)^3 + 64 = 0 ↔ x = -3/2 :=
sorry

end equation_one_solutions_equation_two_solution_l2240_224005


namespace triangle_squares_area_sum_l2240_224014

/-- Given a right triangle EAB with BE = 12 and another right triangle EAH with AH = 5,
    the sum of the areas of squares ABCD, AEFG, and AHIJ is equal to 169 square units. -/
theorem triangle_squares_area_sum : 
  ∀ (A B C D E F G H I J : ℝ × ℝ),
  let ab := dist A B
  let ae := dist A E
  let ah := dist A H
  let be := dist B E
  -- Angle EAB is a right angle
  (ab ^ 2 + ae ^ 2 = be ^ 2) →
  -- BE = 12 units
  (be = 12) →
  -- Triangle EAH is a right triangle
  (ae ^ 2 + ah ^ 2 = (dist E H) ^ 2) →
  -- AH = 5 units
  (ah = 5) →
  -- The sum of the areas of squares ABCD, AEFG, and AHIJ is 169
  (ab ^ 2 + ae ^ 2 + (dist E H) ^ 2 = 169) := by
  sorry


end triangle_squares_area_sum_l2240_224014


namespace boris_candy_problem_l2240_224001

theorem boris_candy_problem (initial_candy : ℕ) : 
  let daughter_eats : ℕ := 8
  let num_bowls : ℕ := 4
  let boris_takes_per_bowl : ℕ := 3
  let candy_left_in_one_bowl : ℕ := 20
  (initial_candy - daughter_eats) / num_bowls - boris_takes_per_bowl = candy_left_in_one_bowl →
  initial_candy = 100 :=
by
  sorry

end boris_candy_problem_l2240_224001


namespace perpendicular_vector_scalar_l2240_224003

theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (x : ℝ) :
  a = (3, 4) →
  b = (2, -1) →
  (a.1 + x * b.1, a.2 + x * b.2) • b = 0 →
  x = -2/5 := by
  sorry

end perpendicular_vector_scalar_l2240_224003


namespace honor_roll_fraction_l2240_224035

theorem honor_roll_fraction (female_honor : Rat) (male_honor : Rat) (female_ratio : Rat) : 
  female_honor = 7/12 →
  male_honor = 11/15 →
  female_ratio = 13/27 →
  (female_ratio * female_honor) + ((1 - female_ratio) * male_honor) = 1071/1620 := by
sorry

end honor_roll_fraction_l2240_224035


namespace mias_test_score_l2240_224079

theorem mias_test_score (total_students : ℕ) (initial_average : ℚ) (average_after_ethan : ℚ) (final_average : ℚ) :
  total_students = 20 →
  initial_average = 84 →
  average_after_ethan = 85 →
  final_average = 86 →
  (total_students * final_average - (total_students - 1) * average_after_ethan : ℚ) = 105 := by
  sorry

end mias_test_score_l2240_224079


namespace triangle_max_perimeter_l2240_224089

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x - Real.sqrt 3 / 2

theorem triangle_max_perimeter 
  (A : ℝ) 
  (h_acute : 0 < A ∧ A < π / 2)
  (h_f_A : f A = Real.sqrt 3 / 2)
  (h_a : ∀ (a b c : ℝ), a = 2 → a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) :
  ∃ (b c : ℝ), 2 + b + c ≤ 6 ∧ 
    ∀ (b' c' : ℝ), 2 + b' + c' ≤ 2 + b + c := by
  sorry

end triangle_max_perimeter_l2240_224089


namespace number_proportion_l2240_224060

theorem number_proportion (x : ℚ) : 
  (x / 5 = 30 / (10 * 60)) → x = 1/4 := by
  sorry

end number_proportion_l2240_224060


namespace max_k_value_l2240_224081

theorem max_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 17 = 0 ∧ y^2 + k*y + 17 = 0 ∧ |x - y| = Real.sqrt 85) →
  k ≤ Real.sqrt 153 :=
sorry

end max_k_value_l2240_224081


namespace positive_A_value_l2240_224064

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 290) : A = Real.sqrt 241 := by
  sorry

end positive_A_value_l2240_224064


namespace rectangular_solid_surface_area_l2240_224057

/-- The surface area of a rectangular solid. -/
def surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: The surface area of a rectangular solid with length 6 meters, width 5 meters, 
    and depth 2 meters is 104 square meters. -/
theorem rectangular_solid_surface_area :
  surface_area 6 5 2 = 104 := by
  sorry

end rectangular_solid_surface_area_l2240_224057


namespace other_root_of_quadratic_l2240_224066

theorem other_root_of_quadratic (b : ℝ) : 
  ((-1 : ℝ)^2 + b * (-1) - 5 = 0) → 
  (∃ x : ℝ, x ≠ -1 ∧ x^2 + b*x - 5 = 0 ∧ x = 5) :=
by sorry

end other_root_of_quadratic_l2240_224066


namespace combined_efficiency_approx_38_l2240_224010

-- Define the fuel efficiencies and distance
def jane_efficiency : ℚ := 30
def mike_efficiency : ℚ := 15
def carl_efficiency : ℚ := 20
def distance : ℚ := 100

-- Define the combined fuel efficiency function
def combined_efficiency (e1 e2 e3 d : ℚ) : ℚ :=
  (3 * d) / (d / e1 + d / e2 + d / e3)

-- State the theorem
theorem combined_efficiency_approx_38 :
  ∃ ε > 0, abs (combined_efficiency jane_efficiency mike_efficiency carl_efficiency distance - 38) < ε :=
by sorry

end combined_efficiency_approx_38_l2240_224010


namespace vector_triangle_rule_l2240_224068

-- Define a triangle ABC in a vector space
variable {V : Type*} [AddCommGroup V]
variable (A B C : V)

-- State the theorem
theorem vector_triangle_rule :
  (C - A) - (B - A) + (B - C) = (0 : V) := by
  sorry

end vector_triangle_rule_l2240_224068


namespace max_value_expression_l2240_224094

theorem max_value_expression : 
  ∃ (M : ℝ), M = 27 ∧ 
  ∀ (x y : ℝ), 
    (Real.sqrt (36 - 4 * Real.sqrt 5) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 2) * 
    (3 + 2 * Real.sqrt (10 - Real.sqrt 5) * Real.cos y - Real.cos (2 * y)) ≤ M :=
by sorry

end max_value_expression_l2240_224094


namespace ticket_sales_proof_l2240_224017

theorem ticket_sales_proof (total_tickets : ℕ) (reduced_price_tickets : ℕ) (full_price_ratio : ℕ) :
  total_tickets = 25200 →
  reduced_price_tickets = 5400 →
  full_price_ratio = 5 →
  reduced_price_tickets + full_price_ratio * reduced_price_tickets = total_tickets →
  full_price_ratio * reduced_price_tickets = 21000 :=
by
  sorry

end ticket_sales_proof_l2240_224017


namespace owen_sleep_time_l2240_224007

theorem owen_sleep_time (total_hours work_hours chore_hours sleep_hours : ℕ) :
  total_hours = 24 ∧ work_hours = 6 ∧ chore_hours = 7 ∧ sleep_hours = total_hours - (work_hours + chore_hours) →
  sleep_hours = 11 := by
  sorry

end owen_sleep_time_l2240_224007


namespace kitten_growth_theorem_l2240_224002

/-- Represents the length of a kitten at different stages of growth -/
structure KittenGrowth where
  initial_length : ℝ
  first_double : ℝ
  second_double : ℝ

/-- Theorem stating that if a kitten's length doubles twice and ends at 16 inches, its initial length was 4 inches -/
theorem kitten_growth_theorem (k : KittenGrowth) :
  k.second_double = 16 ∧ 
  k.first_double = 2 * k.initial_length ∧ 
  k.second_double = 2 * k.first_double →
  k.initial_length = 4 :=
by
  sorry


end kitten_growth_theorem_l2240_224002


namespace total_people_l2240_224082

/-- Calculates the total number of people in two tribes of soldiers -/
theorem total_people (cannoneers : ℕ) : 
  cannoneers = 63 → 
  (let women := 2 * cannoneers
   let men := cannoneers + 2 * women
   women + men) = 441 := by
sorry

end total_people_l2240_224082


namespace triangle_perimeter_range_l2240_224026

/-- Given a triangle with sides a, b, and x, where a > b, 
    prove that the perimeter m satisfies 2a < m < 2(a+b) -/
theorem triangle_perimeter_range 
  (a b x : ℝ) 
  (h1 : a > b) 
  (h2 : a - b < x) 
  (h3 : x < a + b) : 
  2 * a < a + b + x ∧ a + b + x < 2 * (a + b) := by
  sorry

end triangle_perimeter_range_l2240_224026


namespace typist_salary_problem_l2240_224063

/-- Proves that if a salary is increased by 10% and then decreased by 5%, 
    resulting in 1045, the original salary was 1000. -/
theorem typist_salary_problem (original_salary : ℝ) : 
  (original_salary * 1.1 * 0.95 = 1045) → original_salary = 1000 := by
  sorry

end typist_salary_problem_l2240_224063


namespace opposite_sides_line_range_l2240_224054

/-- Given two points on opposite sides of a line, prove the range of the line's constant term -/
theorem opposite_sides_line_range (m : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ = 3 ∧ y₁ = 1 ∧ x₂ = -4 ∧ y₂ = 6) ∧ 
    ((3 * x₁ - 2 * y₁ + m) * (3 * x₂ - 2 * y₂ + m) < 0)) →
  (-7 < m ∧ m < 24) :=
by sorry

end opposite_sides_line_range_l2240_224054


namespace remaining_bird_families_l2240_224050

/-- The number of bird families left near the mountain after some flew away -/
def bird_families_left (initial : ℕ) (flew_away : ℕ) : ℕ :=
  initial - flew_away

/-- Theorem stating that 237 bird families were left near the mountain -/
theorem remaining_bird_families :
  bird_families_left 709 472 = 237 := by
  sorry

end remaining_bird_families_l2240_224050


namespace min_value_of_sum_of_squares_l2240_224021

theorem min_value_of_sum_of_squares (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + 2*x^2 + b*x + 1 = 0) → a^2 + b^2 ≥ 8 := by
  sorry

end min_value_of_sum_of_squares_l2240_224021


namespace average_salary_l2240_224091

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 16000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

theorem average_salary :
  (total_salary : ℚ) / num_people = 8800 := by sorry

end average_salary_l2240_224091


namespace stratified_sampling_probability_l2240_224037

theorem stratified_sampling_probability 
  (total_students : ℕ) 
  (first_year : ℕ) 
  (second_year : ℕ) 
  (third_year : ℕ) 
  (selected : ℕ) 
  (h1 : total_students = first_year + second_year + third_year)
  (h2 : total_students = 600)
  (h3 : first_year = 100)
  (h4 : second_year = 200)
  (h5 : third_year = 300)
  (h6 : selected = 30) :
  (selected : ℚ) / (total_students : ℚ) = 1 / 20 := by
sorry

end stratified_sampling_probability_l2240_224037


namespace min_value_expression_l2240_224042

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_prod : x * y * z = 108) : 
  x^2 + 9*x*y + 9*y^2 + 3*z^2 ≥ 324 := by
  sorry

end min_value_expression_l2240_224042


namespace ellipse_equation_for_given_conditions_l2240_224077

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Semi-focal distance

/-- The standard equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation_for_given_conditions :
  ∀ e : Ellipse,
  e.a = 6 →                  -- Major axis is 12 (2a = 12)
  e.c / e.a = 1 / 3 →        -- Eccentricity is 1/3
  e.c = 2 →                  -- Derived from eccentricity and semi-major axis
  e.b^2 = e.a^2 - e.c^2 →    -- Relationship between a, b, and c
  ∀ x y : ℝ,
  e.equation x y ↔ x^2 / 36 + y^2 / 32 = 1 :=
by sorry

end ellipse_equation_for_given_conditions_l2240_224077


namespace sum_of_fractions_l2240_224053

theorem sum_of_fractions : (2 : ℚ) / 5 + 3 / 8 + 1 / 4 = 41 / 40 := by
  sorry

end sum_of_fractions_l2240_224053


namespace sqrt_equation_solution_l2240_224083

theorem sqrt_equation_solution (y : ℝ) : Real.sqrt (y + 5) = 7 → y = 44 := by
  sorry

end sqrt_equation_solution_l2240_224083


namespace cyclic_quadrilateral_inequality_l2240_224087

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
def CyclicQuadrilateral (A B C D : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ) (r : ℝ), 
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r

/-- The theorem states that for a cyclic quadrilateral ABCD, 
    the sum of the absolute differences between opposite sides 
    is greater than or equal to twice the absolute difference between the diagonals. -/
theorem cyclic_quadrilateral_inequality 
  (A B C D : ℝ × ℝ) 
  (h : CyclicQuadrilateral A B C D) : 
  |dist A B - dist C D| + |dist A D - dist B C| ≥ 2 * |dist A C - dist B D| :=
sorry

end cyclic_quadrilateral_inequality_l2240_224087


namespace total_weight_of_balls_l2240_224004

theorem total_weight_of_balls (blue_weight brown_weight green_weight : ℝ) 
  (h1 : blue_weight = 6)
  (h2 : brown_weight = 3.12)
  (h3 : green_weight = 4.5) :
  blue_weight + brown_weight + green_weight = 13.62 := by
  sorry

end total_weight_of_balls_l2240_224004


namespace floor_equation_solutions_l2240_224009

theorem floor_equation_solutions (a : ℝ) (n : ℕ) (h1 : a > 1) (h2 : n ≥ 2) :
  (∃ (S : Finset ℝ), S.card = n ∧ (∀ x ∈ S, ⌊a * x⌋ = x)) ↔ 
  (1 + 1 / n : ℝ) ≤ a ∧ a < 1 + 1 / (n - 1) :=
sorry

end floor_equation_solutions_l2240_224009


namespace largest_difference_theorem_l2240_224030

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digit_constraints (a b c d e f g h i : ℕ) : Prop :=
  a ∈ ({3, 5, 9} : Set ℕ) ∧
  b ∈ ({2, 3, 7} : Set ℕ) ∧
  c ∈ ({3, 4, 8, 9} : Set ℕ) ∧
  d ∈ ({2, 3, 7} : Set ℕ) ∧
  e ∈ ({3, 5, 9} : Set ℕ) ∧
  f ∈ ({1, 4, 7} : Set ℕ) ∧
  g ∈ ({4, 5, 9} : Set ℕ) ∧
  h = 2 ∧
  i ∈ ({4, 5, 9} : Set ℕ)

def number_from_digits (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem largest_difference_theorem (a b c d e f g h i : ℕ) :
  digit_constraints a b c d e f g h i →
  is_three_digit (number_from_digits a b c) →
  is_three_digit (number_from_digits d e f) →
  is_three_digit (number_from_digits g h i) →
  number_from_digits a b c - number_from_digits d e f = number_from_digits g h i →
  ∀ (x y z u v w : ℕ),
    digit_constraints x y z u v w g h i →
    is_three_digit (number_from_digits x y z) →
    is_three_digit (number_from_digits u v w) →
    number_from_digits x y z - number_from_digits u v w = number_from_digits g h i →
    number_from_digits g h i ≤ 529 →
  (a = 9 ∧ b = 2 ∧ c = 3 ∧ d = 3 ∧ e = 9 ∧ f = 4) :=
sorry

end largest_difference_theorem_l2240_224030


namespace coin_difference_l2240_224012

/-- Represents the denominations of coins available -/
inductive Coin
  | fiveCent
  | twentyCent
  | fiftyCent

/-- The value of each coin in cents -/
def coinValue : Coin → Nat
  | Coin.fiveCent => 5
  | Coin.twentyCent => 20
  | Coin.fiftyCent => 50

/-- The amount to be paid in cents -/
def amountToPay : Nat := 50

/-- A function that calculates the minimum number of coins needed -/
def minCoins : Nat := sorry

/-- A function that calculates the maximum number of coins needed -/
def maxCoins : Nat := sorry

/-- Theorem stating the difference between max and min number of coins -/
theorem coin_difference : maxCoins - minCoins = 9 := by sorry

end coin_difference_l2240_224012


namespace fraction_equality_l2240_224029

/-- Given two amounts a and b, prove that the fraction of b that equals 2/3 of a is 2/3 -/
theorem fraction_equality (a b : ℚ) (h1 : a + b = 1210) (h2 : b = 484) : 
  ∃ x : ℚ, x * b = 2/3 * a ∧ x = 2/3 := by
sorry

end fraction_equality_l2240_224029


namespace m_intersect_n_l2240_224098

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}

theorem m_intersect_n : M ∩ N = {0, 1} := by sorry

end m_intersect_n_l2240_224098


namespace y_equals_seven_l2240_224040

/-- A shape composed entirely of right angles with specific side lengths -/
structure RightAngledShape where
  /-- Length of one side -/
  side1 : ℝ
  /-- Length of another side -/
  side2 : ℝ
  /-- Length of another side -/
  side3 : ℝ
  /-- Length of another side -/
  side4 : ℝ
  /-- Unknown length to be calculated -/
  Y : ℝ
  /-- The total horizontal lengths on the top and bottom sides are equal -/
  total_length_eq : side1 + side3 + Y + side2 = side4 + side2 + side3 + 5

/-- The theorem stating that Y equals 7 for the given shape -/
theorem y_equals_seven (shape : RightAngledShape) 
  (h1 : shape.side1 = 2) 
  (h2 : shape.side2 = 3) 
  (h3 : shape.side3 = 1) 
  (h4 : shape.side4 = 4) : 
  shape.Y = 7 := by
  sorry

end y_equals_seven_l2240_224040


namespace smallest_binary_divisible_by_product_l2240_224085

def is_binary_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

def product_of_first_six : ℕ := (List.range 6).map (· + 1) |>.prod

theorem smallest_binary_divisible_by_product :
  let n : ℕ := 1111111110000
  (is_binary_number n) ∧
  (n % product_of_first_six = 0) ∧
  (∀ m : ℕ, m < n → is_binary_number m → m % product_of_first_six ≠ 0) := by
  sorry

end smallest_binary_divisible_by_product_l2240_224085
