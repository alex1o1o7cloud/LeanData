import Mathlib

namespace fraction_zero_value_l1961_196183

theorem fraction_zero_value (x : ℝ) : 
  (|x| - 3) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 :=
by sorry

end fraction_zero_value_l1961_196183


namespace conditional_probability_good_air_quality_l1961_196161

-- Define the probability of good air quality on any given day
def p_good_day : ℝ := 0.75

-- Define the probability of good air quality for two consecutive days
def p_two_good_days : ℝ := 0.6

-- State the theorem
theorem conditional_probability_good_air_quality :
  (p_two_good_days / p_good_day : ℝ) = 0.8 := by
  sorry

end conditional_probability_good_air_quality_l1961_196161


namespace prime_sum_product_l1961_196138

theorem prime_sum_product (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p ≠ q → p + q = 10 → p * q = 21 := by
  sorry

end prime_sum_product_l1961_196138


namespace alexas_weight_l1961_196145

/-- Given the combined weight of two people and the weight of one person, 
    calculate the weight of the other person. -/
theorem alexas_weight (total_weight katerina_weight : ℕ) 
  (h1 : total_weight = 95)
  (h2 : katerina_weight = 49) :
  total_weight - katerina_weight = 46 := by
  sorry

end alexas_weight_l1961_196145


namespace ceiling_minus_value_l1961_196106

theorem ceiling_minus_value (x : ℝ) (h : ⌈(2 * x)⌉ - ⌊(2 * x)⌋ = 0) : 
  ⌈(2 * x)⌉ - (2 * x) = 0 := by
  sorry

end ceiling_minus_value_l1961_196106


namespace similar_triangles_proportion_l1961_196184

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
def SimilarTriangles (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

theorem similar_triangles_proportion 
  (P Q R X Y Z : ℝ × ℝ) 
  (h_similar : SimilarTriangles {P, Q, R} {X, Y, Z})
  (h_PQ : dist P Q = 8)
  (h_QR : dist Q R = 16)
  (h_ZY : dist Z Y = 32) :
  dist X Y = 16 := by sorry

end similar_triangles_proportion_l1961_196184


namespace least_possible_z_l1961_196151

theorem least_possible_z (x y z : ℤ) : 
  (∃ k : ℤ, x = 2 * k) →  -- x is even
  (∃ m n : ℤ, y = 2 * m + 1 ∧ z = 2 * n + 1) →  -- y and z are odd
  y - x > 5 →
  (∀ w : ℤ, w - x ≥ 9 → z ≤ w) →  -- least possible value of z - x is 9
  z ≥ 11 ∧ (∀ v : ℤ, v ≥ 11 → z ≤ v) :=  -- z is at least 11 and is the least such value
by sorry

end least_possible_z_l1961_196151


namespace evaluate_expression_l1961_196179

theorem evaluate_expression : (24^18) / (72^9) = 8^9 := by sorry

end evaluate_expression_l1961_196179


namespace range_of_a_l1961_196152

def p (x : ℝ) : Prop := |4 - x| ≤ 6

def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≥ 0

def sufficient_not_necessary (P Q : ℝ → Prop) : Prop :=
  (∀ x, ¬(P x) → Q x) ∧ ∃ x, Q x ∧ P x

theorem range_of_a :
  ∀ a : ℝ, 
    (a > 0 ∧ 
     sufficient_not_necessary p (q · a)) →
    (0 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l1961_196152


namespace fourth_root_equation_solution_l1961_196149

theorem fourth_root_equation_solution :
  let f (x : ℝ) := (Real.rpow (61 - 3*x) (1/4) + Real.rpow (17 + 3*x) (1/4))
  ∀ x : ℝ, f x = 6 ↔ x = 7 ∨ x = -23 := by
sorry

end fourth_root_equation_solution_l1961_196149


namespace max_sleep_duration_l1961_196111

/-- A time represented by hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : hours < 24 ∧ minutes < 60

/-- Checks if a given time is a happy moment -/
def is_happy_moment (t : Time) : Prop :=
  (t.hours = 4 * t.minutes) ∨ (t.minutes = 4 * t.hours)

/-- List of all happy moments in a day -/
def happy_moments : List Time :=
  sorry

/-- Calculates the time difference between two times in minutes -/
def time_difference (t1 t2 : Time) : ℕ :=
  sorry

/-- Theorem stating the maximum sleep duration -/
theorem max_sleep_duration :
  ∃ (t1 t2 : Time),
    t1 ∈ happy_moments ∧
    t2 ∈ happy_moments ∧
    time_difference t1 t2 = 239 ∧
    ∀ (t3 t4 : Time),
      t3 ∈ happy_moments →
      t4 ∈ happy_moments →
      time_difference t3 t4 ≤ 239 :=
  sorry

end max_sleep_duration_l1961_196111


namespace school_attendance_l1961_196109

/-- Represents the attendance schedule for a group of students -/
inductive AttendanceSchedule
  | A -- Attends Mondays and Wednesdays
  | B -- Attends Tuesdays and Thursdays
  | C -- Attends Fridays

/-- Represents a day of the week -/
inductive WeekDay
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- The school attendance problem -/
theorem school_attendance
  (total_students : Nat)
  (home_learning_percentage : Rat)
  (group_schedules : List AttendanceSchedule)
  (h1 : total_students = 1000)
  (h2 : home_learning_percentage = 60 / 100)
  (h3 : group_schedules = [AttendanceSchedule.A, AttendanceSchedule.B, AttendanceSchedule.C]) :
  ∃ (attendance : WeekDay → Nat),
    attendance WeekDay.Monday = 133 ∧
    attendance WeekDay.Tuesday = 133 ∧
    attendance WeekDay.Wednesday = 133 ∧
    attendance WeekDay.Thursday = 133 ∧
    attendance WeekDay.Friday = 134 := by
  sorry

end school_attendance_l1961_196109


namespace fuel_purchase_calculation_l1961_196133

/-- Given the cost of fuel per gallon, the fuel consumption rate per hour,
    and the total time to consume all fuel, calculate the number of gallons purchased. -/
theorem fuel_purchase_calculation 
  (cost_per_gallon : ℝ) 
  (consumption_rate_per_hour : ℝ) 
  (total_hours : ℝ) 
  (h1 : cost_per_gallon = 0.70)
  (h2 : consumption_rate_per_hour = 0.40)
  (h3 : total_hours = 175) :
  (consumption_rate_per_hour * total_hours) / cost_per_gallon = 100 := by
  sorry

end fuel_purchase_calculation_l1961_196133


namespace point_coordinates_proof_l1961_196129

/-- Given two points M and N, and a point P such that MP = 1/2 * MN, 
    prove that P has specific coordinates. -/
theorem point_coordinates_proof (M N P : ℝ × ℝ) : 
  M = (3, 2) → 
  N = (-5, -5) → 
  P - M = (1 / 2 : ℝ) • (N - M) → 
  P = (-1, -3/2) := by
  sorry

end point_coordinates_proof_l1961_196129


namespace weight_difference_l1961_196100

/-- Given that Antoinette and Rupert have a combined weight of 98 kg,
    and Antoinette weighs 63 kg, prove that Antoinette weighs 7 kg less
    than twice Rupert's weight. -/
theorem weight_difference (antoinette_weight rupert_weight : ℝ) : 
  antoinette_weight = 63 →
  antoinette_weight + rupert_weight = 98 →
  2 * rupert_weight - antoinette_weight = 7 :=
by sorry

end weight_difference_l1961_196100


namespace brown_mice_count_l1961_196124

theorem brown_mice_count (total white brown : ℕ) : 
  total = white + brown →
  (2 : ℚ) / 3 * total = white →
  white = 14 →
  brown = 7 := by
sorry

end brown_mice_count_l1961_196124


namespace lathe_processing_time_l1961_196117

/-- Given that 3 lathes can process 180 parts in 4 hours,
    prove that 5 lathes will process 600 parts in 8 hours. -/
theorem lathe_processing_time
  (initial_lathes : ℕ)
  (initial_parts : ℕ)
  (initial_hours : ℕ)
  (target_lathes : ℕ)
  (target_parts : ℕ)
  (h1 : initial_lathes = 3)
  (h2 : initial_parts = 180)
  (h3 : initial_hours = 4)
  (h4 : target_lathes = 5)
  (h5 : target_parts = 600)
  : (target_parts : ℚ) / (target_lathes : ℚ) * (initial_lathes : ℚ) / (initial_parts : ℚ) * (initial_hours : ℚ) = 8 := by
  sorry


end lathe_processing_time_l1961_196117


namespace expression_simplification_and_evaluation_l1961_196168

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 4) :
  (((2 * x + 2) / (x^2 - 1) + 1) / ((x + 1) / (x^2 - 2*x + 1))) = 3 :=
by sorry

end expression_simplification_and_evaluation_l1961_196168


namespace weekend_to_weekday_practice_ratio_l1961_196169

/-- Given Daniel's basketball practice schedule, prove the ratio of weekend to weekday practice time -/
theorem weekend_to_weekday_practice_ratio :
  let weekday_daily_practice : ℕ := 15
  let weekday_count : ℕ := 5
  let total_weekly_practice : ℕ := 135
  let weekday_practice := weekday_daily_practice * weekday_count
  let weekend_practice := total_weekly_practice - weekday_practice
  (weekend_practice : ℚ) / weekday_practice = 4 / 5 := by
sorry


end weekend_to_weekday_practice_ratio_l1961_196169


namespace ellipse_max_y_coordinate_l1961_196143

theorem ellipse_max_y_coordinate :
  let ellipse := {(x, y) : ℝ × ℝ | (x^2 / 49) + ((y - 3)^2 / 25) = 1}
  ∃ (y_max : ℝ), y_max = 8 ∧ ∀ (x y : ℝ), (x, y) ∈ ellipse → y ≤ y_max :=
by sorry

end ellipse_max_y_coordinate_l1961_196143


namespace bat_survey_result_l1961_196177

theorem bat_survey_result :
  ∀ (total : ℕ) 
    (blind_believers : ℕ) 
    (ebola_believers : ℕ),
  (blind_believers : ℚ) = 0.750 * total →
  (ebola_believers : ℚ) = 0.523 * blind_believers →
  ebola_believers = 49 →
  total = 125 :=
by
  sorry

end bat_survey_result_l1961_196177


namespace simplified_expression_and_evaluation_l1961_196110

theorem simplified_expression_and_evaluation (x : ℝ) 
  (h1 : x ≠ -3) (h2 : x ≠ 3) :
  (3 / (x - 3) - 3 * x / (x^2 - 9)) / ((3 * x - 9) / (x^2 - 6 * x + 9)) = 3 / (x + 3) ∧
  (3 / (1 + 3) = 3 / 4) := by
  sorry

end simplified_expression_and_evaluation_l1961_196110


namespace trapezoid_area_l1961_196193

-- Define the trapezoid
def trapezoid := {(x, y) : ℝ × ℝ | 0 ≤ x ∧ x ≤ 15 ∧ 10 ≤ y ∧ y ≤ 15 ∧ (y = x ∨ y = 10 ∨ y = 15)}

-- Define the area function
def area (T : Set (ℝ × ℝ)) : ℝ := 62.5

-- Theorem statement
theorem trapezoid_area : area trapezoid = 62.5 := by
  sorry

end trapezoid_area_l1961_196193


namespace point_in_plane_region_l1961_196167

def plane_region (x y : ℝ) : Prop := 2 * x + y - 6 < 0

theorem point_in_plane_region :
  plane_region 0 1 ∧
  ¬ plane_region 5 0 ∧
  ¬ plane_region 0 7 ∧
  ¬ plane_region 2 3 :=
by sorry

end point_in_plane_region_l1961_196167


namespace inverse_function_implies_a_value_l1961_196195

def f (a : ℝ) (x : ℝ) : ℝ := a - 2 * x

theorem inverse_function_implies_a_value (a : ℝ) :
  (∃ g : ℝ → ℝ, Function.LeftInverse g (f a) ∧ Function.RightInverse g (f a) ∧ g (-3) = 3) →
  a = 3 := by
  sorry

end inverse_function_implies_a_value_l1961_196195


namespace triangle_circle_area_ratio_l1961_196103

theorem triangle_circle_area_ratio : 
  let a : ℝ := 8
  let b : ℝ := 15
  let c : ℝ := 17
  let s : ℝ := (a + b + c) / 2
  let triangle_area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let circle_radius : ℝ := c / 2
  let circle_area : ℝ := π * circle_radius^2
  let semicircle_area : ℝ := circle_area / 2
  let outside_triangle_area : ℝ := semicircle_area - triangle_area
  abs ((outside_triangle_area / semicircle_area) - 0.471) < 0.001 := by
sorry

end triangle_circle_area_ratio_l1961_196103


namespace fraction_simplification_l1961_196108

theorem fraction_simplification :
  (1 / 20 : ℚ) - (1 / 21 : ℚ) + (1 / (20 * 21) : ℚ) = (1 / 210 : ℚ) := by
  sorry

end fraction_simplification_l1961_196108


namespace alfred_christmas_shopping_goal_l1961_196188

def christmas_shopping_goal (initial_amount : ℕ) (monthly_savings : ℕ) (months : ℕ) : ℕ :=
  initial_amount + monthly_savings * months

theorem alfred_christmas_shopping_goal :
  christmas_shopping_goal 100 75 12 = 1000 := by
  sorry

end alfred_christmas_shopping_goal_l1961_196188


namespace smallest_positive_multiple_of_45_l1961_196144

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by sorry

end smallest_positive_multiple_of_45_l1961_196144


namespace sum_of_m_and_n_is_zero_l1961_196102

theorem sum_of_m_and_n_is_zero 
  (h1 : ∃ p : ℝ, m * n + p^2 + 4 = 0) 
  (h2 : m - n = 4) : 
  m + n = 0 := by sorry

end sum_of_m_and_n_is_zero_l1961_196102


namespace lansing_elementary_students_l1961_196191

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := 247

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := num_schools * students_per_school

theorem lansing_elementary_students :
  total_students = 6175 :=
by sorry

end lansing_elementary_students_l1961_196191


namespace weight_replacement_l1961_196112

theorem weight_replacement (initial_count : ℕ) (weight_increase : ℝ) (new_person_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 6 →
  new_person_weight = 88 →
  ∃ (replaced_weight : ℝ),
    replaced_weight = 40 ∧
    (initial_count : ℝ) * weight_increase = new_person_weight - replaced_weight :=
by sorry

end weight_replacement_l1961_196112


namespace polynomial_roots_sum_l1961_196104

theorem polynomial_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 3001*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 118 := by
  sorry

end polynomial_roots_sum_l1961_196104


namespace norbs_age_l1961_196156

def guesses : List Nat := [25, 29, 33, 35, 37, 39, 42, 45, 48, 50]

def is_prime (n : Nat) : Prop := Nat.Prime n

def half_guesses_too_low (age : Nat) : Prop :=
  (guesses.filter (· < age)).length = guesses.length / 2

def two_guesses_off_by_one (age : Nat) : Prop :=
  (guesses.filter (λ g => g = age - 1 ∨ g = age + 1)).length = 2

theorem norbs_age : 
  ∃ (age : Nat), age = 47 ∧ 
    is_prime age ∧ 
    half_guesses_too_low age ∧ 
    two_guesses_off_by_one age ∧
    ∀ (n : Nat), n ≠ 47 → 
      ¬(is_prime n ∧ half_guesses_too_low n ∧ two_guesses_off_by_one n) :=
by sorry

end norbs_age_l1961_196156


namespace common_difference_not_three_l1961_196114

def is_valid_sequence (d : ℕ+) : Prop :=
  ∃ (n : ℕ+), 1 + (n - 1) * d = 81

theorem common_difference_not_three :
  ¬(is_valid_sequence 3) := by
  sorry

end common_difference_not_three_l1961_196114


namespace paper_stack_height_l1961_196159

/-- Given a ream of paper with known thickness and sheet count, 
    calculate the number of sheets in a stack of a different height -/
theorem paper_stack_height (ream_sheets : ℕ) (ream_thickness : ℝ) (stack_height : ℝ) :
  ream_sheets > 0 →
  ream_thickness > 0 →
  stack_height > 0 →
  ream_sheets * (stack_height / ream_thickness) = 900 :=
by
  -- Assuming ream_sheets = 400, ream_thickness = 4, and stack_height = 9
  sorry

#check paper_stack_height 400 4 9

end paper_stack_height_l1961_196159


namespace consecutive_prime_product_l1961_196162

-- Define the first four consecutive prime numbers
def first_four_primes : List Nat := [2, 3, 5, 7]

-- Define the product of these primes
def product_of_primes : Nat := first_four_primes.prod

theorem consecutive_prime_product :
  (product_of_primes = 210) ∧
  (product_of_primes % 10 = 0) :=
by sorry

end consecutive_prime_product_l1961_196162


namespace arithmetic_mean_problem_l1961_196122

theorem arithmetic_mean_problem :
  let numbers : Finset ℚ := {7/8, 9/10, 4/5, 17/20}
  17/20 ∈ numbers ∧ 
  9/10 ∈ numbers ∧ 
  4/5 ∈ numbers ∧
  (9/10 + 4/5) / 2 = 17/20 := by sorry

end arithmetic_mean_problem_l1961_196122


namespace seating_arrangements_count_l1961_196198

/-- The number of ways to arrange n distinct objects into k distinct positions --/
def arrangements (n k : ℕ) : ℕ := (k.factorial) / ((k - n).factorial)

/-- The number of seating arrangements for three people in a row of eight chairs
    with an empty seat on either side of each person --/
def seatingArrangements : ℕ :=
  let totalChairs : ℕ := 8
  let peopleToSeat : ℕ := 3
  let availablePositions : ℕ := totalChairs - 2 - (peopleToSeat - 1)
  arrangements peopleToSeat availablePositions

theorem seating_arrangements_count :
  seatingArrangements = 24 := by sorry

end seating_arrangements_count_l1961_196198


namespace carries_strawberry_harvest_l1961_196185

/-- Represents a rectangular garden with strawberry plants -/
structure StrawberryGarden where
  length : ℝ
  width : ℝ
  plants_per_sqft : ℝ
  strawberries_per_plant : ℝ

/-- Calculates the expected total number of strawberries in the garden -/
def total_strawberries (garden : StrawberryGarden) : ℝ :=
  garden.length * garden.width * garden.plants_per_sqft * garden.strawberries_per_plant

/-- Theorem stating the expected number of strawberries in Carrie's garden -/
theorem carries_strawberry_harvest :
  let garden : StrawberryGarden := {
    length := 10,
    width := 15,
    plants_per_sqft := 5,
    strawberries_per_plant := 12
  }
  total_strawberries garden = 9000 := by
  sorry

end carries_strawberry_harvest_l1961_196185


namespace original_index_is_12_l1961_196160

/-- Given an original sequence and a new sequence formed by inserting 3 numbers
    between every two adjacent terms of the original sequence, 
    this function returns the index in the original sequence that corresponds
    to the 49th term in the new sequence. -/
def original_index_of_49th_new_term : ℕ :=
  let x := (49 - 1) / 4
  x + 1

theorem original_index_is_12 : original_index_of_49th_new_term = 12 := by
  sorry

end original_index_is_12_l1961_196160


namespace fraction_addition_l1961_196121

theorem fraction_addition : (18 : ℚ) / 42 + 2 / 9 = 41 / 63 := by
  sorry

end fraction_addition_l1961_196121


namespace circle_area_ratio_l1961_196130

/-- Given three circles S, R, and T, where R's diameter is 20% of S's diameter,
    and T's diameter is 40% of R's diameter, prove that the combined area of
    R and T is 4.64% of the area of S. -/
theorem circle_area_ratio (S R T : ℝ) (hR : R = 0.2 * S) (hT : T = 0.4 * R) :
  (π * (R / 2)^2 + π * (T / 2)^2) / (π * (S / 2)^2) = 0.0464 := by
  sorry

end circle_area_ratio_l1961_196130


namespace triangle_otimes_calculation_l1961_196142

def triangle (a b : ℝ) : ℝ := a + b + a * b - 1

def otimes (a b : ℝ) : ℝ := a^2 - a * b + b^2

theorem triangle_otimes_calculation : triangle 3 (otimes 2 4) = 50 := by
  sorry

end triangle_otimes_calculation_l1961_196142


namespace quadratic_roots_sum_product_l1961_196105

theorem quadratic_roots_sum_product (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ + 3 = 0 → x₂^2 - 4*x₂ + 3 = 0 → x₁ + x₂ - 2*x₁*x₂ = -2 :=
by
  sorry

end quadratic_roots_sum_product_l1961_196105


namespace dentist_age_problem_l1961_196189

/-- Proves that given a dentist's current age of 32 years, if one-sixth of his age 8 years ago
    equals one-tenth of his age at a certain time in the future, then that future time is 8 years from now. -/
theorem dentist_age_problem (future_years : ℕ) : 
  (1/6 : ℚ) * (32 - 8) = (1/10 : ℚ) * (32 + future_years) → future_years = 8 := by
  sorry

end dentist_age_problem_l1961_196189


namespace no_real_roots_condition_l1961_196134

theorem no_real_roots_condition (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 ≠ 0) ↔ k < -1 :=
by sorry

end no_real_roots_condition_l1961_196134


namespace green_ball_theorem_l1961_196176

/-- Represents the price and quantity information for green balls --/
structure GreenBallInfo where
  saltyCost : ℚ
  saltyQuantity : ℕ
  duckCost : ℚ
  duckQuantity : ℕ

/-- Represents a purchase plan --/
structure PurchasePlan where
  saltyQuantity : ℕ
  duckQuantity : ℕ

/-- Represents an exchange method --/
structure ExchangeMethod where
  coupons : ℕ
  saltyCoupons : ℕ
  duckCoupons : ℕ

/-- Main theorem about green ball prices, purchase plans, and exchange methods --/
theorem green_ball_theorem (info : GreenBallInfo) 
  (h1 : info.duckCost = 2 * info.saltyCost)
  (h2 : info.duckCost * info.duckQuantity = 40)
  (h3 : info.saltyCost * info.saltyQuantity = 30)
  (h4 : info.saltyQuantity = info.duckQuantity + 4)
  (h5 : ∀ plan : PurchasePlan, 
    plan.saltyQuantity ≥ 20 ∧ 
    plan.duckQuantity ≥ 20 ∧ 
    plan.saltyQuantity % 10 = 0 ∧
    info.saltyCost * plan.saltyQuantity + info.duckCost * plan.duckQuantity = 200)
  (h6 : ∀ method : ExchangeMethod,
    1 < method.coupons ∧ 
    method.coupons < 10 ∧
    method.saltyCoupons + method.duckCoupons = method.coupons) :
  (info.saltyCost = 5/2 ∧ info.duckCost = 5) ∧
  (∃ (plans : List PurchasePlan), plans = 
    [(PurchasePlan.mk 20 30), (PurchasePlan.mk 30 25), (PurchasePlan.mk 40 20)]) ∧
  (∃ (methods : List ExchangeMethod), methods = 
    [(ExchangeMethod.mk 5 5 0), (ExchangeMethod.mk 5 0 5), 
     (ExchangeMethod.mk 8 6 2), (ExchangeMethod.mk 8 1 7)]) :=
by sorry

end green_ball_theorem_l1961_196176


namespace system_solution_unique_l1961_196199

theorem system_solution_unique : 
  ∃! (x y : ℚ), (6 * x = -9 - 3 * y) ∧ (4 * x = 5 * y - 34) := by
  sorry

end system_solution_unique_l1961_196199


namespace doctors_distribution_l1961_196174

def distribute_doctors (n : ℕ) (k : ℕ) : Prop :=
  ∃ (ways : ℕ),
    n = 7 ∧
    k = 3 ∧
    ways = (Nat.choose 2 1) * (Nat.choose 5 2) * (Nat.choose 3 1) +
           (Nat.choose 5 3) * (Nat.choose 2 1) ∧
    ways = 80

theorem doctors_distribution :
  ∀ (n k : ℕ), distribute_doctors n k :=
sorry

end doctors_distribution_l1961_196174


namespace two_digit_odd_integers_count_l1961_196181

def odd_digits : Finset Nat := {1, 3, 5, 7, 9}

theorem two_digit_odd_integers_count :
  (Finset.filter
    (fun n => n ≥ 10 ∧ n < 100 ∧ n % 2 = 1 ∧
      (n / 10) ∈ odd_digits ∧ (n % 10) ∈ odd_digits ∧
      (n / 10) ≠ (n % 10))
    (Finset.range 100)).card = 20 :=
by sorry

end two_digit_odd_integers_count_l1961_196181


namespace midpoint_sum_zero_l1961_196170

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (8, 10) and (-4, -14) is 0. -/
theorem midpoint_sum_zero : 
  let x1 : ℝ := 8
  let y1 : ℝ := 10
  let x2 : ℝ := -4
  let y2 : ℝ := -14
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 0 := by
sorry

end midpoint_sum_zero_l1961_196170


namespace min_value_sqrt_sum_min_value_sqrt_sum_attained_l1961_196135

theorem min_value_sqrt_sum (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((1 - x)^2 + (-1 + x)^2) ≥ Real.sqrt 10 :=
sorry

theorem min_value_sqrt_sum_attained : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((1 - x)^2 + (-1 + x)^2) = Real.sqrt 10 :=
sorry

end min_value_sqrt_sum_min_value_sqrt_sum_attained_l1961_196135


namespace function_identity_l1961_196113

open Real

theorem function_identity (f : ℝ → ℝ) (h : ∀ x, f (1 - cos x) = sin x ^ 2) :
  ∀ x, f x = 2 * x - x ^ 2 := by
  sorry

end function_identity_l1961_196113


namespace cereal_eating_time_l1961_196157

def mr_fat_rate : ℚ := 1 / 20
def mr_thin_rate : ℚ := 1 / 25
def total_cereal : ℚ := 4

def combined_rate : ℚ := mr_fat_rate + mr_thin_rate

theorem cereal_eating_time :
  (total_cereal / combined_rate) = 400 / 9 := by sorry

end cereal_eating_time_l1961_196157


namespace max_ratio_of_two_digit_integers_with_mean_45_l1961_196123

theorem max_ratio_of_two_digit_integers_with_mean_45 :
  ∀ x y : ℕ,
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  (x + y : ℚ) / 2 = 45 →
  ∀ z : ℚ,
  (z : ℚ) = x / y →
  z ≤ 8 :=
by sorry

end max_ratio_of_two_digit_integers_with_mean_45_l1961_196123


namespace benny_spent_34_dollars_l1961_196197

/-- Calculates the amount spent on baseball gear given the initial amount and the amount left over. -/
def amount_spent (initial : ℕ) (left_over : ℕ) : ℕ :=
  initial - left_over

/-- Proves that Benny spent 34 dollars on baseball gear. -/
theorem benny_spent_34_dollars (initial : ℕ) (left_over : ℕ) 
    (h1 : initial = 67) (h2 : left_over = 33) : 
    amount_spent initial left_over = 34 := by
  sorry

#eval amount_spent 67 33

end benny_spent_34_dollars_l1961_196197


namespace power_three_times_three_l1961_196166

theorem power_three_times_three (x : ℝ) : x^3 * x^3 = x^6 := by
  sorry

end power_three_times_three_l1961_196166


namespace donation_amount_l1961_196136

def barbara_stuffed_animals : ℕ := 9
def trish_stuffed_animals : ℕ := 2 * barbara_stuffed_animals
def sam_stuffed_animals : ℕ := barbara_stuffed_animals + 5
def linda_stuffed_animals : ℕ := sam_stuffed_animals - 7

def barbara_price : ℚ := 2
def trish_price : ℚ := (3:ℚ)/2
def sam_price : ℚ := (5:ℚ)/2
def linda_price : ℚ := 3

def discount_rate : ℚ := (1:ℚ)/10

theorem donation_amount (barbara_stuffed_animals : ℕ) (trish_stuffed_animals : ℕ) 
  (sam_stuffed_animals : ℕ) (linda_stuffed_animals : ℕ) (barbara_price : ℚ) 
  (trish_price : ℚ) (sam_price : ℚ) (linda_price : ℚ) (discount_rate : ℚ) :
  trish_stuffed_animals = 2 * barbara_stuffed_animals →
  sam_stuffed_animals = barbara_stuffed_animals + 5 →
  linda_stuffed_animals = sam_stuffed_animals - 7 →
  barbara_price = 2 →
  trish_price = (3:ℚ)/2 →
  sam_price = (5:ℚ)/2 →
  linda_price = 3 →
  discount_rate = (1:ℚ)/10 →
  (1 - discount_rate) * (barbara_stuffed_animals * barbara_price + 
    trish_stuffed_animals * trish_price + sam_stuffed_animals * sam_price + 
    linda_stuffed_animals * linda_price) = (909:ℚ)/10 := by
  sorry

end donation_amount_l1961_196136


namespace process_time_600_parts_l1961_196137

/-- Linear regression equation for processing time -/
def process_time (x : ℝ) : ℝ := 0.01 * x + 0.5

/-- Theorem: The time required to process 600 parts is 6.5 hours -/
theorem process_time_600_parts : process_time 600 = 6.5 := by
  sorry

#check process_time_600_parts

end process_time_600_parts_l1961_196137


namespace triangle_inequality_sign_l1961_196165

/-- Given a triangle ABC with sides a, b, c (a ≤ b ≤ c), circumradius R, and inradius r,
    the sign of a + b - 2R - 2r depends on angle C as follows:
    1. If π/3 ≤ C < π/2, then a + b - 2R - 2r > 0
    2. If C = π/2, then a + b - 2R - 2r = 0
    3. If π/2 < C < π, then a + b - 2R - 2r < 0 -/
theorem triangle_inequality_sign (a b c R r : ℝ) (C : ℝ) :
  a ≤ b ∧ b ≤ c ∧ 0 < a ∧ 0 < R ∧ 0 < r ∧ 0 < C ∧ C < π →
  (π/3 ≤ C ∧ C < π/2 → a + b - 2*R - 2*r > 0) ∧
  (C = π/2 → a + b - 2*R - 2*r = 0) ∧
  (π/2 < C ∧ C < π → a + b - 2*R - 2*r < 0) := by
  sorry


end triangle_inequality_sign_l1961_196165


namespace right_triangle_consecutive_even_sides_l1961_196132

/-- A triangle with sides 2a, 2a+2, and 2a+4 is a right triangle if and only if a = 3 -/
theorem right_triangle_consecutive_even_sides (a : ℕ) : 
  (2*a)^2 + (2*a+2)^2 = (2*a+4)^2 ↔ a = 3 := by
sorry

end right_triangle_consecutive_even_sides_l1961_196132


namespace apartment_households_l1961_196119

/-- Represents the position and structure of an apartment building --/
structure ApartmentBuilding where
  houses_per_row : ℕ
  floors : ℕ
  households_per_house : ℕ

/-- Represents the position of Mijoo's house in the apartment building --/
structure MijooHousePosition where
  from_left : ℕ
  from_right : ℕ
  from_top : ℕ
  from_bottom : ℕ

/-- Calculates the total number of households in the apartment building --/
def total_households (building : ApartmentBuilding) : ℕ :=
  building.houses_per_row * building.floors * building.households_per_house

/-- Theorem stating the total number of households in the apartment building --/
theorem apartment_households 
  (building : ApartmentBuilding)
  (mijoo_position : MijooHousePosition)
  (h1 : mijoo_position.from_left = 1)
  (h2 : mijoo_position.from_right = 7)
  (h3 : mijoo_position.from_top = 2)
  (h4 : mijoo_position.from_bottom = 4)
  (h5 : building.houses_per_row = mijoo_position.from_left + mijoo_position.from_right - 1)
  (h6 : building.floors = mijoo_position.from_top + mijoo_position.from_bottom - 1)
  (h7 : building.households_per_house = 3) :
  total_households building = 105 := by
  sorry

#eval total_households { houses_per_row := 7, floors := 5, households_per_house := 3 }

end apartment_households_l1961_196119


namespace peanut_cost_per_pound_l1961_196101

/-- The cost per pound of peanuts at Peanut Emporium -/
def cost_per_pound : ℝ := 3

/-- The minimum purchase amount in pounds -/
def minimum_purchase : ℝ := 15

/-- The amount purchased over the minimum in pounds -/
def over_minimum : ℝ := 20

/-- The total cost of the purchase in dollars -/
def total_cost : ℝ := 105

/-- Proof that the cost per pound of peanuts is $3 -/
theorem peanut_cost_per_pound :
  cost_per_pound = total_cost / (minimum_purchase + over_minimum) := by
  sorry

end peanut_cost_per_pound_l1961_196101


namespace product_inequality_l1961_196147

theorem product_inequality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) (h₄ : a₄ > 1) (h₅ : a₅ > 1) : 
  (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) ≤ 16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) := by
  sorry

end product_inequality_l1961_196147


namespace correct_proposition_l1961_196118

-- Define the parallel relation
def parallel (x y : Type) : Prop := sorry

-- Define the intersection of two planes
def intersection (α β : Type) : Type := sorry

-- Define proposition p
def p : Prop :=
  ∀ (a α β : Type), parallel a β ∧ parallel a α → parallel a β

-- Define proposition q
def q : Prop :=
  ∀ (a α β b : Type), parallel a α ∧ parallel a β ∧ intersection α β = b → parallel a b

-- Theorem to prove
theorem correct_proposition : (¬p) ∧ q := by
  sorry

end correct_proposition_l1961_196118


namespace fraction_simplification_l1961_196150

theorem fraction_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (1 - 1 / (x + 2)) / ((x^2 - 1) / (x + 2)) = Real.sqrt 3 / 3 := by
  sorry

end fraction_simplification_l1961_196150


namespace exist_good_numbers_not_preserving_sum_of_digits_l1961_196178

/-- A natural number is "good" if its decimal representation contains only zeros and ones. -/
def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The sum of digits of a natural number in base 10. -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem stating that there exist good numbers whose product is good,
    but the sum of digits property doesn't hold. -/
theorem exist_good_numbers_not_preserving_sum_of_digits :
  ∃ (A B : ℕ), is_good A ∧ is_good B ∧ is_good (A * B) ∧
    sum_of_digits (A * B) ≠ sum_of_digits A * sum_of_digits B := by
  sorry

end exist_good_numbers_not_preserving_sum_of_digits_l1961_196178


namespace quadratic_roots_product_l1961_196172

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p ^ 2 + 9 * p - 21 = 0) → 
  (3 * q ^ 2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = 122 := by
sorry

end quadratic_roots_product_l1961_196172


namespace sin_870_degrees_l1961_196194

theorem sin_870_degrees : Real.sin (870 * π / 180) = 1 / 2 := by
  sorry

end sin_870_degrees_l1961_196194


namespace line_intersects_AB_CD_l1961_196192

/-- Given points A, B, C, D, prove that the line x = 8t, y = 2t, z = 11t passes through
    the origin and intersects both lines AB and CD. -/
theorem line_intersects_AB_CD :
  let A : ℝ × ℝ × ℝ := (1, 0, 1)
  let B : ℝ × ℝ × ℝ := (-2, 2, 1)
  let C : ℝ × ℝ × ℝ := (2, 0, 3)
  let D : ℝ × ℝ × ℝ := (0, 4, -2)
  let line (t : ℝ) : ℝ × ℝ × ℝ := (8*t, 2*t, 11*t)
  (∃ t : ℝ, line t = (0, 0, 0)) ∧ 
  (∃ t₁ s₁ : ℝ, line t₁ = (1-3*s₁, 2*s₁, 1)) ∧
  (∃ t₂ s₂ : ℝ, line t₂ = (2-2*s₂, 4*s₂, 3+5*s₂)) :=
by
  sorry


end line_intersects_AB_CD_l1961_196192


namespace storks_and_birds_l1961_196148

theorem storks_and_birds (initial_birds initial_storks joining_storks : ℕ) :
  initial_birds = 4 →
  initial_storks = 3 →
  joining_storks = 6 →
  (initial_storks + joining_storks) - initial_birds = 5 := by
  sorry

end storks_and_birds_l1961_196148


namespace river_flow_speed_l1961_196153

/-- Proves that the speed of river flow is 2 km/hr given the conditions of the boat journey --/
theorem river_flow_speed 
  (distance : ℝ) 
  (boat_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : distance = 32) 
  (h2 : boat_speed = 6) 
  (h3 : total_time = 12) : 
  ∃ (v : ℝ), v = 2 ∧ 
    (distance / (boat_speed - v) + distance / (boat_speed + v) = total_time) :=
by sorry

end river_flow_speed_l1961_196153


namespace max_bc_value_l1961_196163

theorem max_bc_value (a b c : ℂ) 
  (h : ∀ z : ℂ, Complex.abs z ≤ 1 → Complex.abs (a * z^2 + b * z + c) ≤ 1) : 
  Complex.abs (b * c) ≤ (3 * Real.sqrt 3) / 16 := by
  sorry

end max_bc_value_l1961_196163


namespace range_of_f_l1961_196171

def f (x : ℝ) := x^2 - 2*x + 3

theorem range_of_f :
  ∀ y ∈ Set.Icc 2 6, ∃ x ∈ Set.Icc 0 3, f x = y ∧
  ∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc 2 6 :=
sorry

end range_of_f_l1961_196171


namespace no_solution_fractional_equation_l1961_196126

theorem no_solution_fractional_equation :
  ¬∃ (x : ℝ), (3 / x) + (6 / (x - 1)) - ((x + 5) / (x^2 - x)) = 0 := by
  sorry

end no_solution_fractional_equation_l1961_196126


namespace base8_to_base5_conversion_l1961_196154

-- Define a function to convert from base 8 to base 10
def base8_to_base10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Define a function to convert from base 10 to base 5
def base10_to_base5 (n : Nat) : Nat :=
  let thousands := n / 625
  let hundreds := (n % 625) / 125
  let tens := ((n % 625) % 125) / 25
  let ones := (((n % 625) % 125) % 25) / 5
  thousands * 1000 + hundreds * 100 + tens * 10 + ones

theorem base8_to_base5_conversion :
  base10_to_base5 (base8_to_base10 653) = 3202 := by
  sorry

end base8_to_base5_conversion_l1961_196154


namespace apple_bags_theorem_l1961_196158

theorem apple_bags_theorem (A B C : ℕ) 
  (h1 : A + B = 11) 
  (h2 : B + C = 18) 
  (h3 : A + C = 19) : 
  A + B + C = 24 := by
sorry

end apple_bags_theorem_l1961_196158


namespace amy_garden_seeds_l1961_196182

theorem amy_garden_seeds (initial_seeds : ℕ) (big_garden_seeds : ℕ) (small_gardens : ℕ) 
  (h1 : initial_seeds = 101)
  (h2 : big_garden_seeds = 47)
  (h3 : small_gardens = 9) :
  (initial_seeds - big_garden_seeds) / small_gardens = 6 :=
by
  sorry

end amy_garden_seeds_l1961_196182


namespace three_heads_probability_l1961_196180

/-- A fair coin has a probability of 1/2 for heads on a single flip -/
def fair_coin_prob : ℚ := 1 / 2

/-- The probability of getting three heads in three independent flips of a fair coin -/
def three_heads_prob : ℚ := fair_coin_prob * fair_coin_prob * fair_coin_prob

/-- Theorem: The probability of getting three heads in three independent flips of a fair coin is 1/8 -/
theorem three_heads_probability : three_heads_prob = 1 / 8 := by sorry

end three_heads_probability_l1961_196180


namespace hyperbola_intersection_theorem_l1961_196173

/-- Represents a hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- A line that intersects the hyperbola -/
def intersecting_line (m : ℝ) (x y : ℝ) : Prop :=
  x - y + m = 0

/-- Two points are perpendicular from the origin -/
def perpendicular_from_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem hyperbola_intersection_theorem (h : Hyperbola)
  (h_eccentricity : h.a / Real.sqrt (h.a^2 + h.b^2) = Real.sqrt 3 / 3)
  (h_imaginary_axis : h.b = Real.sqrt 2)
  (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h_on_hyperbola₁ : hyperbola_equation h x₁ y₁)
  (h_on_hyperbola₂ : hyperbola_equation h x₂ y₂)
  (h_on_line₁ : intersecting_line m x₁ y₁)
  (h_on_line₂ : intersecting_line m x₂ y₂)
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂))
  (h_perpendicular : perpendicular_from_origin x₁ y₁ x₂ y₂) :
  m = 2 ∨ m = -2 := by
  sorry

end hyperbola_intersection_theorem_l1961_196173


namespace yellow_balls_count_l1961_196128

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 60 ∧
  white = 22 ∧
  green = 18 ∧
  red = 3 ∧
  purple = 1 ∧
  prob = 95 / 100 ∧
  (white + green + (total - white - green - red - purple)) / total = prob →
  total - white - green - red - purple = 17 := by
    sorry

end yellow_balls_count_l1961_196128


namespace distance_between_homes_l1961_196115

def uphill_speed : ℝ := 3
def downhill_speed : ℝ := 6
def time_vasya_to_petya : ℝ := 2.5
def time_petya_to_vasya : ℝ := 3.5

theorem distance_between_homes : ℝ := by
  -- Define the distance between homes
  let distance : ℝ := 12

  -- Prove that the distance satisfies the given conditions
  have h1 : distance / uphill_speed + distance / downhill_speed = time_vasya_to_petya := by sorry
  have h2 : distance / downhill_speed + distance / uphill_speed = time_petya_to_vasya := by sorry

  -- Conclude that the distance is 12 km
  exact distance

end distance_between_homes_l1961_196115


namespace five_integer_solutions_l1961_196127

theorem five_integer_solutions (x : ℤ) : 
  (∃ (S : Finset ℤ), (∀ y ∈ S, 5*y^2 + 19*y + 16 ≤ 20) ∧ 
                     (∀ z : ℤ, 5*z^2 + 19*z + 16 ≤ 20 → z ∈ S) ∧
                     S.card = 5) := by
  sorry

end five_integer_solutions_l1961_196127


namespace exists_n_sum_of_digits_square_eq_2002_l1961_196120

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a positive integer n such that the sum of the digits of n^2 is 2002 -/
theorem exists_n_sum_of_digits_square_eq_2002 : ∃ n : ℕ+, sumOfDigits (n^2) = 2002 := by sorry

end exists_n_sum_of_digits_square_eq_2002_l1961_196120


namespace fraction_simplification_l1961_196107

theorem fraction_simplification (x y z : ℝ) :
  (16 * x^4 * z^4 - x^4 * y^16 - 64 * x^4 * y^2 * z^4 + 4 * x^4 * y^18 + 32 * x^2 * y * z^4 - 2 * x^2 * y^17 + 16 * y^2 * z^4 - y^18) /
  ((2 * x^2 * y - x^2 - y) * (8 * z^3 + 2 * y^8 * z + 4 * y^4 * z^2 + y^12) * (2 * z - y^4)) =
  -(2 * x^2 * y + x^2 + y) :=
by sorry

end fraction_simplification_l1961_196107


namespace middle_term_is_36_l1961_196187

/-- An arithmetic sequence with 7 terms -/
structure ArithmeticSequence :=
  (a : Fin 7 → ℝ)
  (is_arithmetic : ∀ i j k : Fin 7, i.val + 1 = j.val ∧ j.val + 1 = k.val →
    a j - a i = a k - a j)

/-- The theorem stating that the middle term of the arithmetic sequence is 36 -/
theorem middle_term_is_36 (seq : ArithmeticSequence)
  (h1 : seq.a 0 = 11)
  (h2 : seq.a 6 = 61) :
  seq.a 3 = 36 := by
  sorry

end middle_term_is_36_l1961_196187


namespace trees_died_l1961_196146

/-- Proof that 15 trees died in the park --/
theorem trees_died (initial : ℕ) (cut : ℕ) (remaining : ℕ) (died : ℕ) : 
  initial = 86 → cut = 23 → remaining = 48 → died = initial - cut - remaining → died = 15 := by
  sorry

end trees_died_l1961_196146


namespace consecutive_points_distance_l1961_196139

/-- Given 5 consecutive points on a straight line, prove that ae = 22 -/
theorem consecutive_points_distance (a b c d e : ℝ) : 
  (c - b = 2 * (d - c)) →   -- bc = 2 cd
  (e - d = 8) →             -- de = 8
  (b - a = 5) →             -- ab = 5
  (c - a = 11) →            -- ac = 11
  (e - a = 22) :=           -- ae = 22
by sorry

end consecutive_points_distance_l1961_196139


namespace cryptarithm_solution_l1961_196190

def is_valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def is_unique_assignment (K P O C S R T : ℕ) : Prop :=
  is_valid_digit K ∧ is_valid_digit P ∧ is_valid_digit O ∧ 
  is_valid_digit C ∧ is_valid_digit S ∧ is_valid_digit R ∧
  is_valid_digit T ∧
  K ≠ P ∧ K ≠ O ∧ K ≠ C ∧ K ≠ S ∧ K ≠ R ∧ K ≠ T ∧
  P ≠ O ∧ P ≠ C ∧ P ≠ S ∧ P ≠ R ∧ P ≠ T ∧
  O ≠ C ∧ O ≠ S ∧ O ≠ R ∧ O ≠ T ∧
  C ≠ S ∧ C ≠ R ∧ C ≠ T ∧
  S ≠ R ∧ S ≠ T ∧
  R ≠ T

def satisfies_equation (K P O C S R T : ℕ) : Prop :=
  10000 * K + 1000 * P + 100 * O + 10 * C + C +
  10000 * K + 1000 * P + 100 * O + 10 * C + C =
  10000 * S + 1000 * P + 100 * O + 10 * R + T

theorem cryptarithm_solution :
  ∃! (K P O C S R T : ℕ),
    is_unique_assignment K P O C S R T ∧
    satisfies_equation K P O C S R T ∧
    K = 3 ∧ P = 5 ∧ O = 9 ∧ C = 7 ∧ S = 7 ∧ R = 5 ∧ T = 4 :=
sorry

end cryptarithm_solution_l1961_196190


namespace bread_products_wasted_l1961_196116

/-- Calculates the pounds of bread products wasted in a food fight scenario -/
theorem bread_products_wasted (minimum_wage hours_worked meat_pounds meat_price 
  fruit_veg_pounds fruit_veg_price bread_price janitor_hours janitor_normal_pay : ℝ) 
  (h1 : minimum_wage = 8)
  (h2 : hours_worked = 50)
  (h3 : meat_pounds = 20)
  (h4 : meat_price = 5)
  (h5 : fruit_veg_pounds = 15)
  (h6 : fruit_veg_price = 4)
  (h7 : bread_price = 1.5)
  (h8 : janitor_hours = 10)
  (h9 : janitor_normal_pay = 10) : 
  (minimum_wage * hours_worked - 
   (meat_pounds * meat_price + 
    fruit_veg_pounds * fruit_veg_price + 
    janitor_hours * janitor_normal_pay * 1.5)) / bread_price = 60 := by
  sorry

end bread_products_wasted_l1961_196116


namespace logarithm_simplification_l1961_196186

theorem logarithm_simplification 
  (p q r s y z : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (hy : y > 0) (hz : z > 0) : 
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * z / (s * y)) = Real.log (y / z) :=
sorry

end logarithm_simplification_l1961_196186


namespace fourth_person_height_l1961_196196

theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℕ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- Heights are in increasing order
  h₂ = h₁ + 2 →                 -- Difference between 1st and 2nd is 2 inches
  h₃ = h₂ + 2 →                 -- Difference between 2nd and 3rd is 2 inches
  h₄ = h₃ + 6 →                 -- Difference between 3rd and 4th is 6 inches
  (h₁ + h₂ + h₃ + h₄) / 4 = 78  -- Average height is 78 inches
  → h₄ = 84 :=                  -- Fourth person's height is 84 inches
by sorry

end fourth_person_height_l1961_196196


namespace fraction_difference_simplification_l1961_196131

theorem fraction_difference_simplification :
  (2^2 + 4^2 + 6^2) / (1^2 + 3^2 + 5^2) - (1^2 + 3^2 + 5^2) / (2^2 + 4^2 + 6^2) = 1911 / 1960 := by
  sorry

end fraction_difference_simplification_l1961_196131


namespace angle_through_point_neg_pi_fourth_l1961_196141

/-- If the terminal side of angle α passes through the point (1, -1), 
    then α = -π/4 + 2kπ for some k ∈ ℤ, and specifically α = -π/4 when k = 0. -/
theorem angle_through_point_neg_pi_fourth (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 1 ∧ r * Real.sin α = -1) →
  (∃ (k : ℤ), α = -π/4 + 2 * k * π) ∧ 
  (α = -π/4 ∨ α = -π/4 + 2 * π ∨ α = -π/4 - 2 * π) :=
sorry

end angle_through_point_neg_pi_fourth_l1961_196141


namespace point_on_x_axis_l1961_196140

def on_x_axis (p : ℝ × ℝ × ℝ) : Prop :=
  p.2 = 0 ∧ p.2.2 = 0

theorem point_on_x_axis : on_x_axis (5, 0, 0) := by
  sorry

end point_on_x_axis_l1961_196140


namespace third_square_perimeter_l1961_196125

/-- Given two squares with perimeters 60 cm and 48 cm, prove that a third square
    whose area is equal to the difference of the areas of the first two squares
    has a perimeter of 36 cm. -/
theorem third_square_perimeter (square1 square2 square3 : ℝ → ℝ) :
  (∀ s, square1 s = s^2) →
  (∀ s, square2 s = s^2) →
  (∀ s, square3 s = s^2) →
  (4 * Real.sqrt (square1 (60 / 4))) = 60 →
  (4 * Real.sqrt (square2 (48 / 4))) = 48 →
  square3 (Real.sqrt (square1 (60 / 4) - square2 (48 / 4))) =
    square1 (60 / 4) - square2 (48 / 4) →
  (4 * Real.sqrt (square3 (Real.sqrt (square1 (60 / 4) - square2 (48 / 4))))) = 36 :=
by sorry

end third_square_perimeter_l1961_196125


namespace three_heads_in_eight_tosses_l1961_196164

/-- A fair coin is tossed eight times. -/
def coin_tosses : ℕ := 8

/-- The coin is fair, meaning the probability of heads is 1/2. -/
def fair_coin_prob : ℚ := 1/2

/-- The number of heads we're looking for. -/
def target_heads : ℕ := 3

/-- The probability of getting exactly three heads in eight tosses of a fair coin. -/
theorem three_heads_in_eight_tosses : 
  (Nat.choose coin_tosses target_heads : ℚ) * fair_coin_prob^target_heads * (1 - fair_coin_prob)^(coin_tosses - target_heads) = 7/32 := by
  sorry

end three_heads_in_eight_tosses_l1961_196164


namespace min_value_theorem_min_value_is_five_min_value_attained_l1961_196175

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + 3*b = 5*a*b → 3*x + 4*y ≤ 3*a + 4*b :=
by sorry

theorem min_value_is_five (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  3*x + 4*y ≥ 5 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 3*y = 5*x*y ∧ 3*x + 4*y < 5 + ε :=
by sorry

end min_value_theorem_min_value_is_five_min_value_attained_l1961_196175


namespace union_necessary_not_sufficient_for_intersection_l1961_196155

def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

theorem union_necessary_not_sufficient_for_intersection :
  (∀ x, x ∈ M ∩ P → x ∈ M ∪ P) ∧
  (∃ x, x ∈ M ∪ P ∧ x ∉ M ∩ P) :=
sorry

end union_necessary_not_sufficient_for_intersection_l1961_196155
