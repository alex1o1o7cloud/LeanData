import Mathlib

namespace NUMINAMATH_CALUDE_dons_walking_speed_l3158_315859

/-- Proof of Don's walking speed given the conditions of Cara and Don's walk --/
theorem dons_walking_speed
  (total_distance : ℝ)
  (caras_speed : ℝ)
  (caras_distance : ℝ)
  (don_delay : ℝ)
  (h1 : total_distance = 45)
  (h2 : caras_speed = 6)
  (h3 : caras_distance = 30)
  (h4 : don_delay = 2) :
  ∃ (dons_speed : ℝ), dons_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_dons_walking_speed_l3158_315859


namespace NUMINAMATH_CALUDE_product_local_abs_value_4_in_564823_l3158_315895

/-- The local value of a digit in a number -/
def local_value (n : ℕ) (d : ℕ) (p : ℕ) : ℕ := d * (10 ^ p)

/-- The absolute value of a natural number -/
def abs_nat (n : ℕ) : ℕ := n

theorem product_local_abs_value_4_in_564823 :
  let n : ℕ := 564823
  let d : ℕ := 4
  let p : ℕ := 4  -- position of 4 in 564823 (0-indexed from right)
  (local_value n d p) * (abs_nat d) = 160000 := by sorry

end NUMINAMATH_CALUDE_product_local_abs_value_4_in_564823_l3158_315895


namespace NUMINAMATH_CALUDE_a_7_equals_63_l3158_315862

def sequence_a : ℕ → ℚ
  | 0 => 0  -- We define a₀ and a₁ arbitrarily as they are not used
  | 1 => 0
  | 2 => 1
  | 3 => 3
  | (n + 1) => (sequence_a n ^ 2 - sequence_a (n - 1) + 2 * sequence_a n) / (sequence_a (n - 1) + 1)

theorem a_7_equals_63 : sequence_a 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_a_7_equals_63_l3158_315862


namespace NUMINAMATH_CALUDE_scavenger_hunt_items_l3158_315827

theorem scavenger_hunt_items (lewis samantha tanya : ℕ) : 
  lewis = samantha + 4 →
  samantha = 4 * tanya →
  lewis = 20 →
  tanya = 4 := by sorry

end NUMINAMATH_CALUDE_scavenger_hunt_items_l3158_315827


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3158_315842

/-- Given a hyperbola C with equation x²/m - y² = 1 (m > 0) and asymptote √3x + my = 0,
    prove that the focal length of C is 4. -/
theorem hyperbola_focal_length (m : ℝ) (hm : m > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / m - y^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | Real.sqrt 3 * x + m * y = 0}
  ∃ (a b c : ℝ), a^2 = m ∧ b^2 = m ∧ c^2 = a^2 + b^2 ∧ 2 * c = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3158_315842


namespace NUMINAMATH_CALUDE_existence_of_polynomials_l3158_315854

-- Define the function f
def f (x y z : ℝ) : ℝ := x^2 + y^2 + z^2 + x*y*z

-- Define the theorem
theorem existence_of_polynomials :
  ∃ (a b c : ℝ → ℝ → ℝ → ℝ),
    (∀ x y z, f (a x y z) (b x y z) (c x y z) = f x y z) ∧
    (∃ x y z, (a x y z, b x y z, c x y z) ≠ (x, y, z) ∧
              (a x y z, b x y z, c x y z) ≠ (x, y, -z) ∧
              (a x y z, b x y z, c x y z) ≠ (x, -y, z) ∧
              (a x y z, b x y z, c x y z) ≠ (-x, y, z) ∧
              (a x y z, b x y z, c x y z) ≠ (x, -y, -z) ∧
              (a x y z, b x y z, c x y z) ≠ (-x, y, -z) ∧
              (a x y z, b x y z, c x y z) ≠ (-x, -y, z)) :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_polynomials_l3158_315854


namespace NUMINAMATH_CALUDE_order_of_expressions_l3158_315808

theorem order_of_expressions : 
  let a : ℝ := (0.2 : ℝ)^2
  let b : ℝ := 2^(0.3 : ℝ)
  let c : ℝ := Real.log 2 / Real.log 0.2
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l3158_315808


namespace NUMINAMATH_CALUDE_power_17_2023_mod_26_l3158_315874

theorem power_17_2023_mod_26 : 17^2023 % 26 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_17_2023_mod_26_l3158_315874


namespace NUMINAMATH_CALUDE_square_nonnegative_l3158_315802

theorem square_nonnegative (x : ℝ) : x ^ 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_nonnegative_l3158_315802


namespace NUMINAMATH_CALUDE_earring_percentage_l3158_315838

theorem earring_percentage :
  ∀ (bella_earrings monica_earrings rachel_earrings : ℕ),
    bella_earrings = 10 →
    monica_earrings = 2 * rachel_earrings →
    bella_earrings + monica_earrings + rachel_earrings = 70 →
    (bella_earrings : ℚ) / (monica_earrings : ℚ) * 100 = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_earring_percentage_l3158_315838


namespace NUMINAMATH_CALUDE_range_of_m_l3158_315882

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h_eq : 2/x + 1/y = 1) 
  (h_ineq : x + 2*y > m^2 - 2*m) : 
  -2 < m ∧ m < 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3158_315882


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l3158_315880

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem largest_digit_divisible_by_6 :
  ∀ N : ℕ, N ≤ 9 →
    (is_divisible_by_6 (45670 + N) → N ≤ 8) ∧
    is_divisible_by_6 (45670 + 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l3158_315880


namespace NUMINAMATH_CALUDE_prob_three_red_is_one_fifty_fifth_l3158_315879

/-- The number of red balls in the bag -/
def red_balls : ℕ := 4

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 5

/-- The number of green balls in the bag -/
def green_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := red_balls + blue_balls + green_balls

/-- The number of balls to be picked -/
def picked_balls : ℕ := 3

/-- The probability of picking 3 red balls when randomly selecting 3 balls without replacement -/
def prob_three_red : ℚ := (red_balls * (red_balls - 1) * (red_balls - 2)) / 
  (total_balls * (total_balls - 1) * (total_balls - 2))

theorem prob_three_red_is_one_fifty_fifth : prob_three_red = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_red_is_one_fifty_fifth_l3158_315879


namespace NUMINAMATH_CALUDE_graph_equation_two_lines_l3158_315889

/-- The set of points (x, y) satisfying the equation (x-y)^2 = x^2 + y^2 is equivalent to the union of the lines x = 0 and y = 0 -/
theorem graph_equation_two_lines (x y : ℝ) :
  (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_graph_equation_two_lines_l3158_315889


namespace NUMINAMATH_CALUDE_most_probable_occurrences_l3158_315892

theorem most_probable_occurrences (p : ℝ) (k₀ : ℕ) (h_p : p = 0.4) (h_k₀ : k₀ = 25) :
  ∃ n : ℕ, 62 ≤ n ∧ n ≤ 64 ∧
  (∀ m : ℕ, (m * p - (1 - p) ≤ k₀ ∧ k₀ < m * p + p) → m = n) :=
by sorry

end NUMINAMATH_CALUDE_most_probable_occurrences_l3158_315892


namespace NUMINAMATH_CALUDE_oven_temperature_increase_l3158_315835

/-- Given an oven with a current temperature and a required temperature,
    calculate the temperature increase needed. -/
def temperature_increase_needed (current_temp required_temp : ℕ) : ℕ :=
  required_temp - current_temp

/-- Theorem stating that for an oven at 150 degrees that needs to reach 546 degrees,
    the temperature increase needed is 396 degrees. -/
theorem oven_temperature_increase :
  temperature_increase_needed 150 546 = 396 := by
  sorry

end NUMINAMATH_CALUDE_oven_temperature_increase_l3158_315835


namespace NUMINAMATH_CALUDE_quadratic_roots_preservation_l3158_315853

theorem quadratic_roots_preservation
  (a b : ℝ) (k : ℝ)
  (h_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*a*x₁ + b = 0 ∧ x₂^2 + 2*a*x₂ + b = 0)
  (h_k_pos : k > 0) :
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧
    (y₁^2 + 2*a*y₁ + b) + k*(y₁ + a)^2 = 0 ∧
    (y₂^2 + 2*a*y₂ + b) + k*(y₂ + a)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_preservation_l3158_315853


namespace NUMINAMATH_CALUDE_sqrt_2x_plus_3_eq_x_solution_l3158_315849

theorem sqrt_2x_plus_3_eq_x_solution :
  ∃! x : ℝ, Real.sqrt (2 * x + 3) = x :=
by
  -- The unique solution is x = 3
  use 3
  constructor
  · -- Prove that x = 3 satisfies the equation
    sorry
  · -- Prove that any solution must be equal to 3
    sorry

#check sqrt_2x_plus_3_eq_x_solution

end NUMINAMATH_CALUDE_sqrt_2x_plus_3_eq_x_solution_l3158_315849


namespace NUMINAMATH_CALUDE_count_special_numbers_is_792_l3158_315831

/-- A function that counts the number of 5-digit numbers beginning with 2 
    and having exactly three identical digits -/
def count_special_numbers : ℕ :=
  let count_with_three_twos := 6 * 9 * 8
  let count_with_three_non_twos := 5 * 8 * 9
  count_with_three_twos + count_with_three_non_twos

/-- Theorem stating that the count of special numbers is 792 -/
theorem count_special_numbers_is_792 : count_special_numbers = 792 := by
  sorry

#eval count_special_numbers

end NUMINAMATH_CALUDE_count_special_numbers_is_792_l3158_315831


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_squares_product_l3158_315805

theorem quadratic_equation_sum_squares_product (k : ℚ) : 
  (∃ a b : ℚ, 3 * a^2 + 7 * a + k = 0 ∧ 3 * b^2 + 7 * b + k = 0 ∧ a ≠ b) →
  (∀ a b : ℚ, 3 * a^2 + 7 * a + k = 0 → 3 * b^2 + 7 * b + k = 0 → a^2 + b^2 = 3 * a * b) ↔
  k = 49 / 15 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_squares_product_l3158_315805


namespace NUMINAMATH_CALUDE_jane_rejection_percentage_l3158_315820

theorem jane_rejection_percentage 
  (john_rejection_rate : Real) 
  (total_rejection_rate : Real) 
  (jane_inspection_ratio : Real) :
  john_rejection_rate = 0.005 →
  total_rejection_rate = 0.0075 →
  jane_inspection_ratio = 1.25 →
  ∃ jane_rejection_rate : Real,
    jane_rejection_rate = 0.0095 ∧
    john_rejection_rate * 1 + jane_rejection_rate * jane_inspection_ratio = 
      total_rejection_rate * (1 + jane_inspection_ratio) :=
by sorry

end NUMINAMATH_CALUDE_jane_rejection_percentage_l3158_315820


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l3158_315885

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter 
  (total : Rectangle)
  (small : Rectangle)
  (shaded_area : ℝ)
  (h1 : total.width = 12)
  (h2 : total.height = 10)
  (h3 : small.width = 4)
  (h4 : small.height = 3)
  (h5 : shaded_area = 120) :
  perimeter { width := total.width - (total.width - small.width),
              height := total.height - small.height } = 23 := by
sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l3158_315885


namespace NUMINAMATH_CALUDE_f_g_f_3_equals_101_l3158_315896

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 5 * x + 4

-- State the theorem
theorem f_g_f_3_equals_101 : f (g (f 3)) = 101 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_3_equals_101_l3158_315896


namespace NUMINAMATH_CALUDE_exam_score_problem_l3158_315816

theorem exam_score_problem (total_questions : ℕ) (correct_score wrong_score total_score : ℤ) 
  (h1 : total_questions = 60)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 140) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 40 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l3158_315816


namespace NUMINAMATH_CALUDE_cabbage_area_is_one_square_foot_l3158_315822

/-- Represents the cabbage garden --/
structure CabbageGarden where
  side_length : ℝ
  num_cabbages : ℕ

/-- The increase in cabbages from last year to this year --/
def cabbage_increase : ℕ := 211

/-- The number of cabbages grown this year --/
def cabbages_this_year : ℕ := 11236

/-- Calculates the area of a square garden --/
def garden_area (g : CabbageGarden) : ℝ := g.side_length ^ 2

theorem cabbage_area_is_one_square_foot 
  (last_year : CabbageGarden) 
  (this_year : CabbageGarden) 
  (h1 : this_year.num_cabbages = cabbages_this_year)
  (h2 : this_year.num_cabbages = last_year.num_cabbages + cabbage_increase)
  (h3 : garden_area this_year - garden_area last_year = cabbage_increase) :
  (garden_area this_year - garden_area last_year) / cabbage_increase = 1 := by
  sorry

#check cabbage_area_is_one_square_foot

end NUMINAMATH_CALUDE_cabbage_area_is_one_square_foot_l3158_315822


namespace NUMINAMATH_CALUDE_equation_system_solution_l3158_315833

theorem equation_system_solution :
  ∃ (x y z : ℝ),
    (x / 6) * 12 = 10 ∧
    (y / 4) * 8 = x ∧
    (z / 3) * 5 + y = 20 ∧
    x = 5 ∧
    y = 5 / 2 ∧
    z = 21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3158_315833


namespace NUMINAMATH_CALUDE_sandys_number_l3158_315873

theorem sandys_number : ∃! x : ℝ, (3 * x + 20)^2 = 2500 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sandys_number_l3158_315873


namespace NUMINAMATH_CALUDE_field_area_in_acres_l3158_315807

-- Define the field dimensions
def field_length : ℕ := 30
def width_plus_diagonal : ℕ := 50

-- Define the conversion rate
def square_steps_per_acre : ℕ := 240

-- Theorem statement
theorem field_area_in_acres :
  ∃ (width : ℕ),
    width^2 + field_length^2 = (width_plus_diagonal - width)^2 ∧
    (field_length * width) / square_steps_per_acre = 2 :=
by sorry

end NUMINAMATH_CALUDE_field_area_in_acres_l3158_315807


namespace NUMINAMATH_CALUDE_fliers_remaining_l3158_315864

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ)
  (h1 : total = 1000)
  (h2 : morning_fraction = 1 / 5)
  (h3 : afternoon_fraction = 1 / 4) :
  total - (morning_fraction * total).num - (afternoon_fraction * (total - (morning_fraction * total).num)).num = 600 :=
by sorry

end NUMINAMATH_CALUDE_fliers_remaining_l3158_315864


namespace NUMINAMATH_CALUDE_min_value_on_circle_l3158_315843

theorem min_value_on_circle (x y : ℝ) : 
  (x + 5)^2 + (y - 12)^2 = 14^2 → 
  ∃ (min : ℝ), (∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≤ a^2 + b^2) ∧ min = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l3158_315843


namespace NUMINAMATH_CALUDE_late_fee_is_124_l3158_315887

/-- Calculates the late fee per month for the second bill given the total amount owed and details of three bills. -/
def calculate_late_fee (total_owed : ℚ) (bill1_amount : ℚ) (bill1_interest_rate : ℚ) (bill1_months : ℕ)
  (bill2_amount : ℚ) (bill2_months : ℕ) (bill3_fee1 : ℚ) (bill3_fee2 : ℚ) : ℚ :=
  let bill1_total := bill1_amount + bill1_amount * bill1_interest_rate * bill1_months
  let bill3_total := bill3_fee1 + bill3_fee2
  let bill2_total := total_owed - bill1_total - bill3_total
  (bill2_total - bill2_amount) / bill2_months

/-- Theorem stating that the late fee per month for the second bill is $124. -/
theorem late_fee_is_124 :
  calculate_late_fee 1234 200 (1/10) 2 130 6 40 80 = 124 := by
  sorry

end NUMINAMATH_CALUDE_late_fee_is_124_l3158_315887


namespace NUMINAMATH_CALUDE_maize_stolen_l3158_315886

def months_in_year : ℕ := 12
def years : ℕ := 2
def maize_per_month : ℕ := 1
def donation : ℕ := 8
def final_amount : ℕ := 27

theorem maize_stolen : 
  (months_in_year * years * maize_per_month + donation) - final_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_maize_stolen_l3158_315886


namespace NUMINAMATH_CALUDE_specific_courses_not_consecutive_l3158_315803

-- Define the number of courses
def n : ℕ := 6

-- Define the number of specific courses we're interested in
def k : ℕ := 3

-- Theorem statement
theorem specific_courses_not_consecutive :
  (n.factorial : ℕ) - (n - k + 1).factorial * k.factorial = 576 := by
  sorry

end NUMINAMATH_CALUDE_specific_courses_not_consecutive_l3158_315803


namespace NUMINAMATH_CALUDE_james_van_capacity_l3158_315893

/-- Proves that the total capacity of James' vans is 57600 gallons --/
theorem james_van_capacity :
  let total_vans : ℕ := 6
  let large_van_capacity : ℕ := 8000
  let large_van_count : ℕ := 2
  let medium_van_capacity : ℕ := large_van_capacity * 7 / 10  -- 30% less than 8000
  let medium_van_count : ℕ := 1
  let small_van_count : ℕ := total_vans - large_van_count - medium_van_count
  let total_capacity : ℕ := 57600
  let remaining_capacity : ℕ := total_capacity - (large_van_capacity * large_van_count + medium_van_capacity * medium_van_count)
  let small_van_capacity : ℕ := remaining_capacity / small_van_count

  (large_van_capacity * large_van_count + 
   medium_van_capacity * medium_van_count + 
   small_van_capacity * small_van_count) = total_capacity := by
  sorry

end NUMINAMATH_CALUDE_james_van_capacity_l3158_315893


namespace NUMINAMATH_CALUDE_differential_of_y_l3158_315899

noncomputable def y (x : ℝ) : ℝ := 2 * x + Real.log (|Real.sin x + 2 * Real.cos x|)

theorem differential_of_y (x : ℝ) :
  deriv y x = (5 * Real.cos x) / (Real.sin x + 2 * Real.cos x) :=
by sorry

end NUMINAMATH_CALUDE_differential_of_y_l3158_315899


namespace NUMINAMATH_CALUDE_railway_optimization_l3158_315888

/-- The number of round trips per day as a function of the number of carriages -/
def t (n : ℕ) : ℤ := -2 * n + 24

/-- The number of passengers per day as a function of the number of carriages -/
def y (n : ℕ) : ℤ := t n * n * 110 * 2

theorem railway_optimization :
  (t 4 = 16 ∧ t 7 = 10) ∧ 
  (∀ n : ℕ, 1 ≤ n → n < 12 → y n ≤ y 6) ∧
  y 6 = 15840 := by
  sorry

#eval t 4  -- Expected: 16
#eval t 7  -- Expected: 10
#eval y 6  -- Expected: 15840

end NUMINAMATH_CALUDE_railway_optimization_l3158_315888


namespace NUMINAMATH_CALUDE_charity_race_dropouts_l3158_315865

/-- The number of people who dropped out of a bicycle charity race --/
def dropouts (initial_racers : ℕ) (joined_racers : ℕ) (finishers : ℕ) : ℕ :=
  (initial_racers + joined_racers) * 2 - finishers

theorem charity_race_dropouts : dropouts 50 30 130 = 30 := by
  sorry

end NUMINAMATH_CALUDE_charity_race_dropouts_l3158_315865


namespace NUMINAMATH_CALUDE_divisibility_problem_l3158_315814

theorem divisibility_problem (a b c : ℤ) (h : 18 ∣ (a^3 + b^3 + c^3)) : 6 ∣ (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3158_315814


namespace NUMINAMATH_CALUDE_pictures_per_album_l3158_315845

theorem pictures_per_album 
  (phone_pics : ℕ) 
  (camera_pics : ℕ) 
  (num_albums : ℕ) 
  (h1 : phone_pics = 23) 
  (h2 : camera_pics = 7) 
  (h3 : num_albums = 5) 
  (h4 : num_albums > 0) :
  (phone_pics + camera_pics) / num_albums = 6 := by
  sorry

end NUMINAMATH_CALUDE_pictures_per_album_l3158_315845


namespace NUMINAMATH_CALUDE_min_rental_cost_l3158_315813

/-- Represents the number of buses of type A -/
def x : ℕ := sorry

/-- Represents the number of buses of type B -/
def y : ℕ := sorry

/-- The total number of passengers -/
def total_passengers : ℕ := 900

/-- The capacity of a type A bus -/
def capacity_A : ℕ := 36

/-- The capacity of a type B bus -/
def capacity_B : ℕ := 60

/-- The rental cost of a type A bus -/
def cost_A : ℕ := 1600

/-- The rental cost of a type B bus -/
def cost_B : ℕ := 2400

/-- The maximum total number of buses allowed -/
def max_buses : ℕ := 21

theorem min_rental_cost :
  (∃ x y : ℕ,
    x * capacity_A + y * capacity_B ≥ total_passengers ∧
    x + y ≤ max_buses ∧
    y ≤ x + 7 ∧
    ∀ a b : ℕ,
      (a * capacity_A + b * capacity_B ≥ total_passengers ∧
       a + b ≤ max_buses ∧
       b ≤ a + 7) →
      x * cost_A + y * cost_B ≤ a * cost_A + b * cost_B) ∧
  (∀ x y : ℕ,
    x * capacity_A + y * capacity_B ≥ total_passengers ∧
    x + y ≤ max_buses ∧
    y ≤ x + 7 →
    x * cost_A + y * cost_B ≥ 36800) :=
by sorry

end NUMINAMATH_CALUDE_min_rental_cost_l3158_315813


namespace NUMINAMATH_CALUDE_intersection_and_distance_l3158_315848

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the parameter a
def a : ℝ := -3

-- Define the line equations
def line1 (x y : ℝ) : Prop := y = 2 * x
def line2 (x y : ℝ) : Prop := x + y + a = 0
def line3 (x y : ℝ) : Prop := a * x + 2 * y + 3 = 0

-- State the theorem
theorem intersection_and_distance :
  (line1 P.1 P.2 ∧ line2 P.1 P.2) →
  (a = -3 ∧ P.2 = 2 ∧
   (|a * P.1 + 2 * P.2 + 3| / Real.sqrt (a^2 + 2^2) = 4 * Real.sqrt 13 / 13)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_distance_l3158_315848


namespace NUMINAMATH_CALUDE_female_democrat_ratio_l3158_315840

theorem female_democrat_ratio (total_participants male_participants female_participants female_democrats : ℕ) 
  (h1 : total_participants = 720)
  (h2 : male_participants + female_participants = total_participants)
  (h3 : male_participants / 4 = total_participants / 3 - female_democrats)
  (h4 : total_participants / 3 = 240)
  (h5 : female_democrats = 120) :
  female_democrats / female_participants = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_female_democrat_ratio_l3158_315840


namespace NUMINAMATH_CALUDE_max_profit_theorem_l3158_315825

def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

def profit_B (x : ℝ) : ℝ := 2 * x

def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (15 - x)

theorem max_profit_theorem :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 15 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 15 → total_profit x ≥ total_profit y ∧
  total_profit x = 45.6 :=
sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l3158_315825


namespace NUMINAMATH_CALUDE_two_distinct_roots_iff_a_in_A_l3158_315863

/-- The equation has exactly two distinct roots -/
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (x₁ - a)^2 - 1 = 2 * (x₁ + |x₁|) ∧
  (x₂ - a)^2 - 1 = 2 * (x₂ + |x₂|) ∧
  ∀ x : ℝ, (x - a)^2 - 1 = 2 * (x + |x|) → x = x₁ ∨ x = x₂

/-- The set of values for a -/
def A : Set ℝ := Set.Ioi 1 ∪ Set.Ioo (-1) 1 ∪ Set.Iic (-5/4)

theorem two_distinct_roots_iff_a_in_A :
  ∀ a : ℝ, has_two_distinct_roots a ↔ a ∈ A :=
by sorry

end NUMINAMATH_CALUDE_two_distinct_roots_iff_a_in_A_l3158_315863


namespace NUMINAMATH_CALUDE_chessboard_coloring_l3158_315817

/-- A coloring of an n × n chessboard is valid if for every i ∈ {1,2,...,n}, 
    the 2n-1 cells on i-th row and i-th column have all different colors. -/
def ValidColoring (n : ℕ) (k : ℕ) : Prop :=
  ∃ (coloring : Fin n → Fin n → Fin k),
    ∀ i : Fin n, (∀ j j' : Fin n, j ≠ j' → coloring i j ≠ coloring i j') ∧
                 (∀ i' : Fin n, i ≠ i' → coloring i i' ≠ coloring i' i)

theorem chessboard_coloring :
  (¬ ValidColoring 2001 4001) ∧
  (∀ m : ℕ, ValidColoring (2^m - 1) (2^(m+1) - 1)) :=
sorry

end NUMINAMATH_CALUDE_chessboard_coloring_l3158_315817


namespace NUMINAMATH_CALUDE_shooting_events_contradictory_l3158_315860

-- Define the sample space
def Ω : Type := List Bool

-- Define the events
def at_least_one_hit (ω : Ω) : Prop := ω.any id
def three_consecutive_misses (ω : Ω) : Prop := ω = [false, false, false]

-- Define the property of being contradictory events
def contradictory (A B : Ω → Prop) : Prop :=
  (∀ ω : Ω, A ω → ¬B ω) ∧ (∀ ω : Ω, B ω → ¬A ω)

-- Theorem statement
theorem shooting_events_contradictory :
  contradictory at_least_one_hit three_consecutive_misses :=
by sorry

end NUMINAMATH_CALUDE_shooting_events_contradictory_l3158_315860


namespace NUMINAMATH_CALUDE_max_cube_sum_on_circle_l3158_315846

theorem max_cube_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 2) :
  x^3 + y^3 ≤ 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_cube_sum_on_circle_l3158_315846


namespace NUMINAMATH_CALUDE_inequality_proof_l3158_315867

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3158_315867


namespace NUMINAMATH_CALUDE_unique_four_digit_square_l3158_315871

/-- Reverses a four-digit number -/
def reverse (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The main theorem -/
theorem unique_four_digit_square : ∃! n : ℕ, 
  1000 ≤ n ∧ n < 10000 ∧ 
  is_perfect_square n ∧
  is_perfect_square (reverse n) ∧
  is_perfect_square (n / reverse n) ∧
  n = 9801 := by
sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_l3158_315871


namespace NUMINAMATH_CALUDE_total_precious_stones_l3158_315881

theorem total_precious_stones (agate olivine sapphire diamond amethyst ruby : ℕ) : 
  agate = 25 →
  olivine = agate + 5 →
  sapphire = 2 * olivine →
  diamond = olivine + 11 →
  amethyst = sapphire + diamond →
  ruby = diamond + 7 →
  agate + olivine + sapphire + diamond + amethyst + ruby = 305 := by
  sorry

end NUMINAMATH_CALUDE_total_precious_stones_l3158_315881


namespace NUMINAMATH_CALUDE_seth_candy_bars_l3158_315884

theorem seth_candy_bars (max_candy_bars : ℕ) (seth_candy_bars : ℕ) : 
  max_candy_bars = 24 →
  seth_candy_bars = 3 * max_candy_bars + 6 →
  seth_candy_bars = 78 :=
by sorry

end NUMINAMATH_CALUDE_seth_candy_bars_l3158_315884


namespace NUMINAMATH_CALUDE_range_of_a_l3158_315837

def p (x : ℝ) : Prop := |4 * x - 3| ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

theorem range_of_a :
  (∃ x, ¬(p x) ∧ q x a) ∧
  (∀ x, ¬(q x a) → ¬(p x)) →
  ∃ S : Set ℝ, S = {a : ℝ | 0 ≤ a ∧ a ≤ 1/2} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3158_315837


namespace NUMINAMATH_CALUDE_f_monotone_increasing_iff_l3158_315897

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1) + 1

theorem f_monotone_increasing_iff (x : ℝ) :
  StrictMono (fun y => f y) ↔ x ∈ Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_iff_l3158_315897


namespace NUMINAMATH_CALUDE_probability_at_least_one_one_is_correct_l3158_315851

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The probability of at least one die showing a 1 when two fair 6-sided dice are rolled -/
def probability_at_least_one_one : ℚ := 11 / 36

/-- Theorem stating that the probability of at least one die showing a 1 
    when two fair 6-sided dice are rolled is 11/36 -/
theorem probability_at_least_one_one_is_correct : 
  probability_at_least_one_one = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_one_is_correct_l3158_315851


namespace NUMINAMATH_CALUDE_average_customers_per_table_l3158_315811

/-- Given a restaurant scenario with tables, women, and men, calculate the average number of customers per table. -/
theorem average_customers_per_table 
  (tables : ℝ) 
  (women : ℝ) 
  (men : ℝ) 
  (h_tables : tables = 9.0) 
  (h_women : women = 7.0) 
  (h_men : men = 3.0) : 
  (women + men) / tables = 10.0 / 9.0 := by
  sorry

end NUMINAMATH_CALUDE_average_customers_per_table_l3158_315811


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_nine_l3158_315890

theorem smallest_n_multiple_of_nine (x y a : ℤ) : 
  (∃ k₁ k₂ : ℤ, x - a = 9 * k₁ ∧ y + a = 9 * k₂) →
  (∃ n : ℕ, n > 0 ∧ ∃ k : ℤ, x^2 + x*y + y^2 + n = 9 * k ∧
    ∀ m : ℕ, m > 0 → (∃ l : ℤ, x^2 + x*y + y^2 + m = 9 * l) → m ≥ n) →
  (∃ k : ℤ, x^2 + x*y + y^2 + 6 = 9 * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_nine_l3158_315890


namespace NUMINAMATH_CALUDE_exactly_one_positive_integer_satisfies_condition_l3158_315834

theorem exactly_one_positive_integer_satisfies_condition : 
  ∃! (n : ℕ), n > 0 ∧ 20 - 5 * n > 12 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_positive_integer_satisfies_condition_l3158_315834


namespace NUMINAMATH_CALUDE_product_not_exceeding_sum_l3158_315877

theorem product_not_exceeding_sum (x y : ℕ) (h : x * y ≤ x + y) :
  (x = 1 ∧ y ≥ 1) ∨ (x = 2 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_product_not_exceeding_sum_l3158_315877


namespace NUMINAMATH_CALUDE_complex_additive_inverse_l3158_315869

theorem complex_additive_inverse (m : ℝ) : 
  let z : ℂ := (1 - m * I) / (1 - 2 * I)
  (∃ (a : ℝ), z = a - a * I) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_additive_inverse_l3158_315869


namespace NUMINAMATH_CALUDE_sara_movie_tickets_l3158_315883

/-- The number of movie theater tickets Sara bought -/
def num_tickets : ℕ := 2

/-- The cost of each movie theater ticket in cents -/
def ticket_cost : ℕ := 1062

/-- The cost of renting a movie in cents -/
def rental_cost : ℕ := 159

/-- The cost of buying a movie in cents -/
def purchase_cost : ℕ := 1395

/-- The total amount Sara spent in cents -/
def total_spent : ℕ := 3678

/-- Theorem stating that the number of tickets Sara bought is correct -/
theorem sara_movie_tickets : 
  num_tickets * ticket_cost + rental_cost + purchase_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_sara_movie_tickets_l3158_315883


namespace NUMINAMATH_CALUDE_princess_count_proof_l3158_315852

/-- Represents the number of princesses at the ball -/
def num_princesses : ℕ := 8

/-- Represents the number of knights at the ball -/
def num_knights : ℕ := 22 - num_princesses

/-- Represents the total number of people at the ball -/
def total_people : ℕ := 22

/-- Function to calculate the number of knights a princess dances with -/
def knights_danced_with (princess_index : ℕ) : ℕ := 6 + princess_index

theorem princess_count_proof :
  (num_princesses + num_knights = total_people) ∧ 
  (knights_danced_with num_princesses = num_knights) ∧
  (∀ i, i ≥ 1 → i ≤ num_princesses → knights_danced_with i ≤ num_knights) :=
sorry

end NUMINAMATH_CALUDE_princess_count_proof_l3158_315852


namespace NUMINAMATH_CALUDE_complex_number_problem_l3158_315828

/-- Given a complex number z satisfying z = i(2-z), prove that z = 1 + i and |z-(2-i)| = √5 -/
theorem complex_number_problem (z : ℂ) (h : z = Complex.I * (2 - z)) : 
  z = 1 + Complex.I ∧ Complex.abs (z - (2 - Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3158_315828


namespace NUMINAMATH_CALUDE_combined_swim_time_l3158_315856

def freestyle_time : ℕ := 48

def backstroke_time : ℕ := freestyle_time + 4

def butterfly_time : ℕ := backstroke_time + 3

def breaststroke_time : ℕ := butterfly_time + 2

def total_time : ℕ := freestyle_time + backstroke_time + butterfly_time + breaststroke_time

theorem combined_swim_time : total_time = 212 := by
  sorry

end NUMINAMATH_CALUDE_combined_swim_time_l3158_315856


namespace NUMINAMATH_CALUDE_mixed_div_frac_example_l3158_315855

-- Define the division operation for mixed numbers and fractions
def mixedDivFrac (whole : ℤ) (num : ℕ) (den : ℕ) (frac_num : ℕ) (frac_den : ℕ) : ℚ :=
  (whole : ℚ) + (num : ℚ) / (den : ℚ) / ((frac_num : ℚ) / (frac_den : ℚ))

-- State the theorem
theorem mixed_div_frac_example : mixedDivFrac 2 1 4 3 5 = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_mixed_div_frac_example_l3158_315855


namespace NUMINAMATH_CALUDE_henan_population_scientific_notation_l3158_315876

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem henan_population_scientific_notation :
  toScientificNotation (98.83 * 1000000) = ScientificNotation.mk 9.883 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_henan_population_scientific_notation_l3158_315876


namespace NUMINAMATH_CALUDE_symmetry_line_of_circles_l3158_315858

/-- Given two circles O and C that are symmetric with respect to a line l, 
    prove that l has the equation x - y + 2 = 0 -/
theorem symmetry_line_of_circles (x y : ℝ) : 
  (x^2 + y^2 = 4) →  -- equation of circle O
  (x^2 + y^2 + 4*x - 4*y + 4 = 0) →  -- equation of circle C
  ∃ (l : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ l ↔ p.1 - p.2 + 2 = 0) ∧  -- equation of line l
    (∀ (p : ℝ × ℝ), p ∈ l ↔ 
      ∃ (q r : ℝ × ℝ), 
        (q.1^2 + q.2^2 = 4) ∧  -- q is on circle O
        (r.1^2 + r.2^2 + 4*r.1 - 4*r.2 + 4 = 0) ∧  -- r is on circle C
        (p = ((q.1 + r.1)/2, (q.2 + r.2)/2)) ∧  -- p is midpoint of qr
        ((r.1 - q.1) * (p.1 - q.1) + (r.2 - q.2) * (p.2 - q.2) = 0))  -- qr ⊥ l
  := by sorry

end NUMINAMATH_CALUDE_symmetry_line_of_circles_l3158_315858


namespace NUMINAMATH_CALUDE_circular_arrangements_count_l3158_315824

/-- The number of ways to arrange n people in a circle with r people between A and B -/
def circularArrangements (n : ℕ) (r : ℕ) : ℕ :=
  2 * Nat.factorial (n - 2)

/-- Theorem: The number of circular arrangements with r people between A and B -/
theorem circular_arrangements_count (n : ℕ) (r : ℕ) 
  (h₁ : n ≥ 3) 
  (h₂ : r < n / 2 - 1) : 
  circularArrangements n r = 2 * Nat.factorial (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangements_count_l3158_315824


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l3158_315866

def club_members : ℕ := 30
def boys : ℕ := 18
def girls : ℕ := 12

theorem president_vice_president_selection :
  (boys * girls) + (girls * boys) = 432 :=
by sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l3158_315866


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l3158_315875

theorem no_prime_roots_for_quadratic : 
  ¬∃ (k : ℤ), ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    p ≠ q ∧
    (p : ℤ) + q = 72 ∧ 
    (p : ℤ) * q = k ∧
    ∀ (x : ℤ), x^2 - 72*x + k = 0 ↔ x = p ∨ x = q :=
by sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l3158_315875


namespace NUMINAMATH_CALUDE_basketball_handshakes_l3158_315857

/-- Represents the number of handshakes in a basketball game --/
def total_handshakes (players_per_team : ℕ) (num_teams : ℕ) (num_referees : ℕ) : ℕ :=
  let total_players := players_per_team * num_teams
  let player_handshakes := players_per_team * players_per_team
  let referee_handshakes := total_players * num_referees
  player_handshakes + referee_handshakes

/-- Theorem stating the total number of handshakes in the given scenario --/
theorem basketball_handshakes :
  total_handshakes 6 2 3 = 72 := by
  sorry

#eval total_handshakes 6 2 3

end NUMINAMATH_CALUDE_basketball_handshakes_l3158_315857


namespace NUMINAMATH_CALUDE_samirs_age_in_five_years_l3158_315815

/-- Given that Samir's age is half of Hania's age 10 years ago,
    and Hania will be 45 years old in 5 years,
    prove that Samir will be 20 years old in 5 years. -/
theorem samirs_age_in_five_years
  (samir_current_age : ℕ)
  (hania_current_age : ℕ)
  (samir_age_condition : samir_current_age = (hania_current_age - 10) / 2)
  (hania_future_age_condition : hania_current_age + 5 = 45) :
  samir_current_age + 5 = 20 := by
  sorry


end NUMINAMATH_CALUDE_samirs_age_in_five_years_l3158_315815


namespace NUMINAMATH_CALUDE_F15_triangles_l3158_315829

/-- The number of triangles in figure n of the sequence -/
def T (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else T (n - 1) + 3 * n + 3

/-- The sequence of figures satisfies the given construction rules -/
axiom construction_rule (n : ℕ) : n ≥ 2 → T n = T (n - 1) + 3 * n + 3

/-- F₂ has 7 triangles -/
axiom F2_triangles : T 2 = 7

/-- The number of triangles in F₁₅ is 400 -/
theorem F15_triangles : T 15 = 400 := by sorry

end NUMINAMATH_CALUDE_F15_triangles_l3158_315829


namespace NUMINAMATH_CALUDE_odd_function_zero_value_l3158_315872

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_zero_value (f : ℝ → ℝ) (h : OddFunction f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_zero_value_l3158_315872


namespace NUMINAMATH_CALUDE_lending_rate_calculation_l3158_315823

def borrowed_amount : ℝ := 7000
def borrowed_time : ℝ := 2
def borrowed_rate : ℝ := 4
def gain_per_year : ℝ := 140

theorem lending_rate_calculation :
  let borrowed_interest := borrowed_amount * borrowed_rate * borrowed_time / 100
  let total_gain := gain_per_year * borrowed_time
  let total_interest_earned := borrowed_interest + total_gain
  let lending_rate := (total_interest_earned * 100) / (borrowed_amount * borrowed_time)
  lending_rate = 6 := by sorry

end NUMINAMATH_CALUDE_lending_rate_calculation_l3158_315823


namespace NUMINAMATH_CALUDE_product_of_roots_l3158_315850

theorem product_of_roots (y₁ y₂ : ℝ) : 
  y₁ + 16 / y₁ = 12 → 
  y₂ + 16 / y₂ = 12 → 
  y₁ * y₂ = 16 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l3158_315850


namespace NUMINAMATH_CALUDE_mary_max_earnings_l3158_315812

/-- Mary's maximum weekly earnings at the restaurant --/
theorem mary_max_earnings (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℚ) 
  (overtime_rate_increase : ℚ) (bonus_hours : ℕ) (bonus_amount : ℚ) :
  max_hours = 80 →
  regular_hours = 20 →
  regular_rate = 8 →
  overtime_rate_increase = 1/4 →
  bonus_hours = 5 →
  bonus_amount = 20 →
  let overtime_hours := max_hours - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  let regular_earnings := regular_hours * regular_rate
  let overtime_earnings := overtime_hours * overtime_rate
  let bonus_count := overtime_hours / bonus_hours
  let total_bonus := bonus_count * bonus_amount
  regular_earnings + overtime_earnings + total_bonus = 1000 := by
sorry

end NUMINAMATH_CALUDE_mary_max_earnings_l3158_315812


namespace NUMINAMATH_CALUDE_least_integer_with_specific_divisibility_l3158_315894

theorem least_integer_with_specific_divisibility : ∃ n : ℕ+,
  (∀ k : ℕ, k ≤ 28 → k ∣ n) ∧
  (31 ∣ n) ∧
  ¬(29 ∣ n) ∧
  ¬(30 ∣ n) ∧
  (∀ m : ℕ+, m < n →
    ¬((∀ k : ℕ, k ≤ 28 → k ∣ m) ∧
      (31 ∣ m) ∧
      ¬(29 ∣ m) ∧
      ¬(30 ∣ m))) ∧
  n = 477638700 := by
sorry

end NUMINAMATH_CALUDE_least_integer_with_specific_divisibility_l3158_315894


namespace NUMINAMATH_CALUDE_penalty_kick_test_l3158_315841

/-- The probability of scoring a single penalty kick -/
def p_score : ℚ := 2/3

/-- The probability of missing a single penalty kick -/
def p_miss : ℚ := 1 - p_score

/-- The probability of being admitted in the penalty kick test -/
def p_admitted : ℚ := 
  p_score * p_score + 
  p_miss * p_score * p_score + 
  p_miss * p_miss * p_score * p_score + 
  p_score * p_miss * p_score * p_score

/-- The expected number of goals scored in the penalty kick test -/
def expected_goals : ℚ := 
  0 * (p_miss * p_miss * p_miss) + 
  1 * (2 * p_score * p_miss * p_miss + p_miss * p_miss * p_score * p_miss) + 
  2 * (p_score * p_score + p_miss * p_score * p_score + p_miss * p_miss * p_score * p_score + p_score * p_miss * p_score * p_miss) + 
  3 * (p_score * p_miss * p_score * p_score)

theorem penalty_kick_test :
  p_admitted = 20/27 ∧ expected_goals = 50/27 := by
  sorry

end NUMINAMATH_CALUDE_penalty_kick_test_l3158_315841


namespace NUMINAMATH_CALUDE_black_population_in_south_percentage_l3158_315898

/-- Represents the population data for a specific ethnic group across regions -/
structure PopulationData :=
  (ne : ℕ)
  (mw : ℕ)
  (central : ℕ)
  (south : ℕ)
  (west : ℕ)

/-- The demographic data for the nation in 2020 -/
def demographicData : List PopulationData :=
  [
    ⟨50, 60, 40, 70, 45⟩,  -- White
    ⟨6, 7, 3, 23, 5⟩,      -- Black
    ⟨2, 2, 1, 2, 6⟩,       -- Asian
    ⟨2, 2, 1, 4, 5⟩        -- Other
  ]

/-- Calculates the total population for a given PopulationData -/
def totalPopulation (data : PopulationData) : ℕ :=
  data.ne + data.mw + data.central + data.south + data.west

/-- Rounds a rational number to the nearest integer -/
def roundToNearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem black_population_in_south_percentage :
  let blackData := demographicData[1]
  let totalBlack := totalPopulation blackData
  let blackInSouth := blackData.south
  roundToNearest ((blackInSouth : ℚ) / totalBlack * 100) = 52 := by
  sorry

end NUMINAMATH_CALUDE_black_population_in_south_percentage_l3158_315898


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3158_315878

/-- The lateral surface area of a cylinder with base radius 2 and generatrix length 3 is 12π. -/
theorem cylinder_lateral_surface_area :
  ∀ (r g : ℝ), r = 2 → g = 3 → 2 * π * r * g = 12 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3158_315878


namespace NUMINAMATH_CALUDE_jenny_sold_192_packs_l3158_315891

/-- The number of boxes Jenny sold -/
def boxes_sold : ℝ := 24.0

/-- The number of packs per box -/
def packs_per_box : ℝ := 8.0

/-- The total number of packs Jenny sold -/
def total_packs : ℝ := boxes_sold * packs_per_box

theorem jenny_sold_192_packs : total_packs = 192.0 := by
  sorry

end NUMINAMATH_CALUDE_jenny_sold_192_packs_l3158_315891


namespace NUMINAMATH_CALUDE_sorting_inequality_l3158_315836

theorem sorting_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a^3 + b^3 + c^3) ≥ a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sorting_inequality_l3158_315836


namespace NUMINAMATH_CALUDE_determine_hidden_numbers_l3158_315868

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Given two sums S1 and S2, it is possible to determine the original numbers a, b, and c -/
theorem determine_hidden_numbers (a b c : ℕ) :
  let k := num_digits (a + b + c)
  let S1 := a + b + c
  let S2 := a + b * 10^k + c * 10^(2*k)
  ∃! (a' b' c' : ℕ), S1 = a' + b' + c' ∧ S2 = a' + b' * 10^k + c' * 10^(2*k) ∧ a' = a ∧ b' = b ∧ c' = c :=
by sorry

end NUMINAMATH_CALUDE_determine_hidden_numbers_l3158_315868


namespace NUMINAMATH_CALUDE_triangle_theorem_l3158_315870

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition -/
def triangleCondition (t : Triangle) : Prop :=
  t.b * (Real.sin (t.C / 2))^2 + t.c * (Real.sin (t.B / 2))^2 = t.a / 2

theorem triangle_theorem (t : Triangle) (h : triangleCondition t) :
  (t.b + t.c = 2 * t.a) ∧ (t.A ≤ Real.pi / 3) := by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3158_315870


namespace NUMINAMATH_CALUDE_archie_antibiotic_cost_l3158_315861

/-- The total cost of antibiotics for Archie -/
def total_cost (doses_per_day : ℕ) (days : ℕ) (cost_per_dose : ℕ) : ℕ :=
  doses_per_day * days * cost_per_dose

/-- Proof that the total cost of antibiotics for Archie is $63 -/
theorem archie_antibiotic_cost :
  total_cost 3 7 3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_archie_antibiotic_cost_l3158_315861


namespace NUMINAMATH_CALUDE_apple_pie_servings_l3158_315801

theorem apple_pie_servings 
  (guests : ℕ) 
  (apples_per_guest : ℝ) 
  (num_pies : ℕ) 
  (apples_per_serving : ℝ) 
  (h1 : guests = 12) 
  (h2 : apples_per_guest = 3) 
  (h3 : num_pies = 3) 
  (h4 : apples_per_serving = 1.5) : 
  (guests * apples_per_guest) / (num_pies * apples_per_serving) = 8 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_servings_l3158_315801


namespace NUMINAMATH_CALUDE_smartphone_customers_l3158_315826

/-- Represents the relationship between number of customers and smartphone price -/
def inversely_proportional (p c : ℝ) := ∃ k : ℝ, p * c = k

theorem smartphone_customers : 
  ∀ (p₁ p₂ c₁ c₂ : ℝ),
  inversely_proportional p₁ c₁ →
  inversely_proportional p₂ c₂ →
  p₁ = 20 →
  c₁ = 200 →
  c₂ = 400 →
  p₂ = 10 :=
by sorry

end NUMINAMATH_CALUDE_smartphone_customers_l3158_315826


namespace NUMINAMATH_CALUDE_rectangle_area_solution_l3158_315839

/-- A rectangle with dimensions (3x - 4) and (4x + 6) has area 12x^2 + 2x - 24 -/
def rectangle_area (x : ℝ) : ℝ := (3*x - 4) * (4*x + 6)

/-- The solution set for x -/
def solution_set : Set ℝ := {x | x > 4/3}

theorem rectangle_area_solution :
  ∀ x : ℝ, x ∈ solution_set ↔ 
    (rectangle_area x = 12*x^2 + 2*x - 24 ∧ 
     3*x - 4 > 0 ∧ 
     4*x + 6 > 0) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_solution_l3158_315839


namespace NUMINAMATH_CALUDE_arrangements_count_l3158_315847

/-- The number of departments in the unit -/
def num_departments : ℕ := 3

/-- The number of people selected from each department for training -/
def people_per_department : ℕ := 2

/-- The total number of people trained -/
def total_trained : ℕ := num_departments * people_per_department

/-- The number of people returning to the unit after training -/
def returning_people : ℕ := 2

/-- Function to calculate the number of arrangements -/
def calculate_arrangements : ℕ := 
  let same_dept := num_departments * (returning_people * (returning_people - 1))
  let diff_dept := (num_departments * (num_departments - 1) / 2) * (returning_people * returning_people)
  same_dept + diff_dept

/-- Theorem stating that the number of different arrangements is 42 -/
theorem arrangements_count : calculate_arrangements = 42 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l3158_315847


namespace NUMINAMATH_CALUDE_overlap_area_is_two_l3158_315832

-- Define the 3x3 grid
def Grid := Fin 3 × Fin 3

-- Define the two quadrilaterals
def quad1 : List Grid := [(0,1), (1,2), (2,1), (1,0)]
def quad2 : List Grid := [(0,0), (2,2), (2,0), (0,2)]

-- Define the function to calculate the area of overlap
def overlapArea (q1 q2 : List Grid) : ℝ :=
  sorry  -- The actual calculation would go here

-- State the theorem
theorem overlap_area_is_two :
  overlapArea quad1 quad2 = 2 :=
sorry

end NUMINAMATH_CALUDE_overlap_area_is_two_l3158_315832


namespace NUMINAMATH_CALUDE_b_value_l3158_315830

def p (x : ℝ) : ℝ := 3 * x - 8

def q (x b : ℝ) : ℝ := 4 * x - b

theorem b_value (b : ℝ) : p (q 3 b) = 10 → b = 6 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l3158_315830


namespace NUMINAMATH_CALUDE_pyramid_edge_ratio_l3158_315821

/-- Represents a pyramid with a cross-section parallel to its base -/
structure Pyramid where
  base_area : ℝ
  cross_section_area : ℝ
  upper_edge_length : ℝ
  lower_edge_length : ℝ
  parallel_cross_section : cross_section_area > 0
  area_ratio : cross_section_area / base_area = 4 / 9

/-- 
Theorem: In a pyramid with a cross-section parallel to its base, 
if the ratio of the cross-sectional area to the base area is 4:9, 
then the ratio of the lengths of the upper and lower parts of the lateral edge is 2:3.
-/
theorem pyramid_edge_ratio (p : Pyramid) : 
  p.upper_edge_length / p.lower_edge_length = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_edge_ratio_l3158_315821


namespace NUMINAMATH_CALUDE_carols_rectangle_width_l3158_315809

theorem carols_rectangle_width (carol_length jordan_length jordan_width : ℝ) 
  (h1 : carol_length = 15)
  (h2 : jordan_length = 6)
  (h3 : jordan_width = 50)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  carol_width = 20 := by
  sorry

end NUMINAMATH_CALUDE_carols_rectangle_width_l3158_315809


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l3158_315810

theorem sum_of_fractions_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l3158_315810


namespace NUMINAMATH_CALUDE_problem_solution_l3158_315844

theorem problem_solution : ∃ m : ℚ, 
  (∃ x₁ x₂ : ℚ, 5*m + 3*x₁ = 1 + x₁ ∧ 2*x₂ + m = 3*m ∧ x₁ = x₂ + 2) →
  7*m^2 - 1 = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3158_315844


namespace NUMINAMATH_CALUDE_valid_parameterizations_l3158_315800

def is_valid_parameterization (p₀ : ℝ × ℝ) (d : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, let (x, y) := p₀ + t • d
           y = x - 1

theorem valid_parameterizations :
  (is_valid_parameterization (1, 0) (1, 1)) ∧
  (is_valid_parameterization (0, -1) (-1, -1)) ∧
  (is_valid_parameterization (2, 1) (0.5, 0.5)) :=
sorry

end NUMINAMATH_CALUDE_valid_parameterizations_l3158_315800


namespace NUMINAMATH_CALUDE_bean_ratio_l3158_315819

/-- Proves that the ratio of green beans to remaining beans after removing red and white beans is 1:1 --/
theorem bean_ratio (total : ℕ) (green : ℕ) : 
  total = 572 →
  green = 143 →
  (total - total / 4 - (total - total / 4) / 3 - green) = green :=
by
  sorry

end NUMINAMATH_CALUDE_bean_ratio_l3158_315819


namespace NUMINAMATH_CALUDE_value_of_x_l3158_315804

theorem value_of_x (x y z : ℝ) 
  (h1 : x = y / 3) 
  (h2 : y = z / 6) 
  (h3 : z = 72) : 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3158_315804


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_one_l3158_315818

theorem sum_reciprocals_equals_one 
  (a b c d : ℝ) 
  (ω : ℂ) 
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1) 
  (hd : d ≠ -1) 
  (hω1 : ω^4 = 1) 
  (hω2 : ω ≠ 1) 
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / (ω + 1)) : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_one_l3158_315818


namespace NUMINAMATH_CALUDE_solution_difference_l3158_315806

theorem solution_difference (r s : ℝ) : 
  r ≠ s →
  (r - 5) * (r + 5) = 25 * r - 125 →
  (s - 5) * (s + 5) = 25 * s - 125 →
  r > s →
  r - s = 15 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l3158_315806
