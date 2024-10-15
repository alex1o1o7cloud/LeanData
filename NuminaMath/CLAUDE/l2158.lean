import Mathlib

namespace NUMINAMATH_CALUDE_cylinder_volume_from_sheet_l2158_215841

/-- The volume of a cylinder formed by a rectangular sheet as its lateral surface -/
theorem cylinder_volume_from_sheet (length width : ℝ) (h : length = 12 ∧ width = 8) :
  ∃ (volume : ℝ), (volume = 192 / Real.pi ∨ volume = 288 / Real.pi) ∧
  ∃ (radius height : ℝ), 
    (2 * Real.pi * radius = width ∧ height = length ∧ volume = Real.pi * radius^2 * height) ∨
    (2 * Real.pi * radius = length ∧ height = width ∧ volume = Real.pi * radius^2 * height) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_sheet_l2158_215841


namespace NUMINAMATH_CALUDE_factorial_square_root_product_l2158_215828

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_square_root_product : (Real.sqrt (factorial 5 * factorial 4))^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_product_l2158_215828


namespace NUMINAMATH_CALUDE_bucket_fills_theorem_l2158_215810

/-- Calculates the number of times a bucket is filled to reach the top of a bathtub. -/
def bucket_fills_to_top (bucket_capacity : ℕ) (buckets_removed : ℕ) (weekly_usage : ℕ) (days_per_week : ℕ) : ℕ :=
  let daily_usage := weekly_usage / days_per_week
  let removed_water := buckets_removed * bucket_capacity
  let full_tub_water := daily_usage + removed_water
  full_tub_water / bucket_capacity

/-- Theorem stating that under given conditions, the bucket is filled 14 times to reach the top. -/
theorem bucket_fills_theorem :
  bucket_fills_to_top 120 3 9240 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_bucket_fills_theorem_l2158_215810


namespace NUMINAMATH_CALUDE_javier_has_four_children_l2158_215886

/-- The number of children Javier has -/
def num_children : ℕ :=
  let total_legs : ℕ := 22
  let num_dogs : ℕ := 2
  let num_cats : ℕ := 1
  let javier_legs : ℕ := 2
  let dog_legs : ℕ := 4
  let cat_legs : ℕ := 4
  (total_legs - (num_dogs * dog_legs + num_cats * cat_legs + javier_legs)) / 2

theorem javier_has_four_children : num_children = 4 := by
  sorry

end NUMINAMATH_CALUDE_javier_has_four_children_l2158_215886


namespace NUMINAMATH_CALUDE_distance_to_plane_l2158_215819

/-- The distance from a point to a plane in 3D space -/
def distance_point_to_plane (P : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_to_plane :
  let P : ℝ × ℝ × ℝ := (-1, 3, 2)
  let n : ℝ × ℝ × ℝ := (2, -2, 1)
  distance_point_to_plane P n = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_plane_l2158_215819


namespace NUMINAMATH_CALUDE_percentage_problem_l2158_215862

theorem percentage_problem (P : ℝ) : P = 50 → 30 = (P / 100) * 40 + 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2158_215862


namespace NUMINAMATH_CALUDE_boat_speed_problem_l2158_215801

/-- Proves that given a boat traveling 45 miles upstream in 5 hours and having a speed of 12 mph in still water, the speed of the current is 3 mph. -/
theorem boat_speed_problem (distance : ℝ) (time : ℝ) (still_water_speed : ℝ) 
  (h1 : distance = 45) 
  (h2 : time = 5) 
  (h3 : still_water_speed = 12) : 
  still_water_speed - (distance / time) = 3 := by
  sorry

#check boat_speed_problem

end NUMINAMATH_CALUDE_boat_speed_problem_l2158_215801


namespace NUMINAMATH_CALUDE_proposition_equivalence_l2158_215834

theorem proposition_equivalence (x : ℝ) :
  (x^2 + 3*x - 4 = 0 → x = -4 ∨ x = 1) ↔ (x ≠ -4 ∧ x ≠ 1 → x^2 + 3*x - 4 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l2158_215834


namespace NUMINAMATH_CALUDE_tan_expression_equals_neg_sqrt_three_l2158_215882

/-- A sequence is a geometric progression -/
def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence is an arithmetic progression -/
def is_arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- Main theorem -/
theorem tan_expression_equals_neg_sqrt_three
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_geom : is_geometric_progression a)
  (h_arith : is_arithmetic_progression b)
  (h_prod : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
  (h_sum : b 1 + b 6 + b 11 = 7 * Real.pi) :
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_tan_expression_equals_neg_sqrt_three_l2158_215882


namespace NUMINAMATH_CALUDE_class_mean_calculation_l2158_215820

theorem class_mean_calculation (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_mean : ℚ) (group2_mean : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 32 →
  group2_students = 8 →
  group1_mean = 68 / 100 →
  group2_mean = 82 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 708 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_class_mean_calculation_l2158_215820


namespace NUMINAMATH_CALUDE_xy_addition_identity_l2158_215864

theorem xy_addition_identity (x y : ℝ) : -x*y - x*y = -2*(x*y) := by
  sorry

end NUMINAMATH_CALUDE_xy_addition_identity_l2158_215864


namespace NUMINAMATH_CALUDE_bar_chart_best_for_rainfall_l2158_215803

-- Define the characteristics of the data
structure RainfallData where
  area : String
  seasons : Fin 4 → Float
  isRainfall : Bool

-- Define the types of charts
inductive ChartType
  | Bar
  | Line
  | Pie

-- Define a function to determine the best chart type
def bestChartType (data : RainfallData) : ChartType :=
  ChartType.Bar

-- Theorem stating that bar chart is the best choice for rainfall data
theorem bar_chart_best_for_rainfall (data : RainfallData) :
  data.isRainfall = true → bestChartType data = ChartType.Bar :=
by
  sorry

#check bar_chart_best_for_rainfall

end NUMINAMATH_CALUDE_bar_chart_best_for_rainfall_l2158_215803


namespace NUMINAMATH_CALUDE_tank_full_time_l2158_215816

/-- Represents the time it takes to fill a tank with given parameters -/
def fill_time (tank_capacity : ℕ) (pipe_a_rate : ℕ) (pipe_b_rate : ℕ) (pipe_c_rate : ℕ) : ℕ :=
  let cycle_net_fill := pipe_a_rate + pipe_b_rate - pipe_c_rate
  let cycles := tank_capacity / cycle_net_fill
  let total_minutes := cycles * 3
  total_minutes - 1

/-- Theorem stating that the tank will be full after 50 minutes -/
theorem tank_full_time :
  fill_time 850 40 30 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_tank_full_time_l2158_215816


namespace NUMINAMATH_CALUDE_original_workers_count_l2158_215865

/-- Given a work that can be completed by an unknown number of workers in 45 days,
    and that adding 10 workers allows the work to be completed in 35 days,
    prove that the original number of workers is 35. -/
theorem original_workers_count (work : ℝ) (h1 : work > 0) : ∃ (workers : ℕ),
  (workers : ℝ) * 45 = work ∧
  (workers + 10 : ℝ) * 35 = work ∧
  workers = 35 := by
sorry

end NUMINAMATH_CALUDE_original_workers_count_l2158_215865


namespace NUMINAMATH_CALUDE_smallest_perfect_cube_multiplier_l2158_215826

def y : ℕ := 2^3 * 3^4 * 4^3 * 5^4 * 6^6 * 7^7 * 8^8 * 9^9

theorem smallest_perfect_cube_multiplier (n : ℕ) :
  (∀ m : ℕ, m < 29400 → ¬ ∃ k : ℕ, m * y = k^3) ∧
  ∃ k : ℕ, 29400 * y = k^3 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_cube_multiplier_l2158_215826


namespace NUMINAMATH_CALUDE_product_equals_square_l2158_215880

theorem product_equals_square : 
  200 * 39.96 * 3.996 * 500 = (3996 : ℝ)^2 := by sorry

end NUMINAMATH_CALUDE_product_equals_square_l2158_215880


namespace NUMINAMATH_CALUDE_intersection_of_3n_and_2m_plus_1_l2158_215804

theorem intersection_of_3n_and_2m_plus_1 :
  {x : ℤ | ∃ n : ℤ, x = 3 * n} ∩ {x : ℤ | ∃ m : ℤ, x = 2 * m + 1} =
  {x : ℤ | ∃ k : ℤ, x = 12 * k + 1 ∨ x = 12 * k + 5} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_3n_and_2m_plus_1_l2158_215804


namespace NUMINAMATH_CALUDE_segment_length_l2158_215838

/-- A rectangle with side lengths 4 and 6, divided into four equal parts by two segments emanating from one vertex -/
structure DividedRectangle where
  /-- The length of the rectangle -/
  length : ℝ
  /-- The width of the rectangle -/
  width : ℝ
  /-- The length of the first dividing segment -/
  segment1 : ℝ
  /-- The length of the second dividing segment -/
  segment2 : ℝ
  /-- The rectangle has side lengths 4 and 6 -/
  dim_constraint : length = 4 ∧ width = 6
  /-- The two segments divide the rectangle into four equal parts -/
  division_constraint : ∃ (a b c d : ℝ), a + b + c + d = length * width ∧ 
                        a = b ∧ b = c ∧ c = d

/-- The theorem stating that one of the dividing segments has length √18.25 -/
theorem segment_length (r : DividedRectangle) : r.segment1 = Real.sqrt 18.25 ∨ r.segment2 = Real.sqrt 18.25 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_l2158_215838


namespace NUMINAMATH_CALUDE_bubble_radius_l2158_215832

/-- The radius of a sphere with volume equal to the sum of volumes of a hemisphere and a cylinder --/
theorem bubble_radius (hemisphere_radius cylinder_radius cylinder_height : ℝ) 
  (hr : hemisphere_radius = 5)
  (hcr : cylinder_radius = 2)
  (hch : cylinder_height = hemisphere_radius) : 
  ∃ R : ℝ, R^3 = 77.5 ∧ 
  (4/3 * Real.pi * R^3 = 2/3 * Real.pi * hemisphere_radius^3 + Real.pi * cylinder_radius^2 * cylinder_height) :=
by sorry

end NUMINAMATH_CALUDE_bubble_radius_l2158_215832


namespace NUMINAMATH_CALUDE_susan_chairs_l2158_215855

def chairs_problem (red_chairs : ℕ) (yellow_multiplier : ℕ) (blue_difference : ℕ) : Prop :=
  let yellow_chairs := red_chairs * yellow_multiplier
  let blue_chairs := yellow_chairs - blue_difference
  red_chairs + yellow_chairs + blue_chairs = 43

theorem susan_chairs : chairs_problem 5 4 2 := by
  sorry

end NUMINAMATH_CALUDE_susan_chairs_l2158_215855


namespace NUMINAMATH_CALUDE_tims_age_l2158_215840

/-- Given that Tom's age is 6 years more than 200% of Tim's age, 
    and Tom is 22 years old, Tim's age is 8 years. -/
theorem tims_age (tom_age tim_age : ℕ) 
  (h1 : tom_age = 2 * tim_age + 6)  -- Tom's age relation to Tim's
  (h2 : tom_age = 22)               -- Tom's actual age
  : tim_age = 8 := by
  sorry

#check tims_age

end NUMINAMATH_CALUDE_tims_age_l2158_215840


namespace NUMINAMATH_CALUDE_alex_max_correct_answers_l2158_215824

/-- Represents a math contest with multiple-choice questions. -/
structure MathContest where
  total_questions : ℕ
  correct_points : ℤ
  blank_points : ℤ
  incorrect_points : ℤ

/-- Represents a student's performance in the math contest. -/
structure StudentPerformance where
  contest : MathContest
  total_score : ℤ

/-- Calculates the maximum number of correct answers for a given student performance. -/
def max_correct_answers (perf : StudentPerformance) : ℕ :=
  sorry

/-- The theorem stating the maximum number of correct answers for Alex's performance. -/
theorem alex_max_correct_answers :
  let contest : MathContest := {
    total_questions := 80,
    correct_points := 5,
    blank_points := 0,
    incorrect_points := -2
  }
  let performance : StudentPerformance := {
    contest := contest,
    total_score := 150
  }
  max_correct_answers performance = 44 := by
  sorry

end NUMINAMATH_CALUDE_alex_max_correct_answers_l2158_215824


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l2158_215814

theorem complex_equation_solutions (c p q r s : ℂ) : 
  (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) →
  (∀ z : ℂ, (z - p) * (z - q) * (z - r) * (z - s) = 
             (z - c*p) * (z - c*q) * (z - c*r) * (z - c*s)) →
  (∃ (solutions : Finset ℂ), solutions.card = 4 ∧ c ∈ solutions ∧
    ∀ x ∈ solutions, x^4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l2158_215814


namespace NUMINAMATH_CALUDE_range_of_m_l2158_215827

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0

def q (m : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m + 2)*x₁ - 1 < (m + 2)*x₂ - 1

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (m ≤ -2 ∨ m ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2158_215827


namespace NUMINAMATH_CALUDE_officer_assignment_count_l2158_215837

-- Define the set of people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carol : Person
| Dave : Person

-- Define the set of officer positions
inductive Position : Type
| President : Position
| Secretary : Position
| Treasurer : Position

-- Define a function to check if a person is qualified for a position
def isQualified (p : Person) (pos : Position) : Prop :=
  match pos with
  | Position.President => p = Person.Dave
  | _ => True

-- Define an assignment of officers
def OfficerAssignment := Position → Person

-- Define a valid assignment
def validAssignment (assignment : OfficerAssignment) : Prop :=
  (∀ pos, isQualified (assignment pos) pos) ∧
  (∀ pos1 pos2, pos1 ≠ pos2 → assignment pos1 ≠ assignment pos2)

-- State the theorem
theorem officer_assignment_count :
  ∃ (assignments : Finset OfficerAssignment),
    (∀ a ∈ assignments, validAssignment a) ∧
    assignments.card = 6 :=
sorry

end NUMINAMATH_CALUDE_officer_assignment_count_l2158_215837


namespace NUMINAMATH_CALUDE_food_bank_remaining_lyanna_food_bank_remaining_l2158_215809

/-- Given a food bank with donations over two weeks and a distribution in the third week,
    calculate the remaining food. -/
theorem food_bank_remaining (first_week : ℝ) (second_week_multiplier : ℝ) (distribution_percentage : ℝ) : ℝ :=
  let second_week := first_week * second_week_multiplier
  let total_donated := first_week + second_week
  let distributed := total_donated * distribution_percentage
  let remaining := total_donated - distributed
  remaining

/-- The amount of food remaining in Lyanna's food bank after two weeks of donations
    and a distribution in the third week. -/
theorem lyanna_food_bank_remaining : food_bank_remaining 40 2 0.7 = 36 := by
  sorry

end NUMINAMATH_CALUDE_food_bank_remaining_lyanna_food_bank_remaining_l2158_215809


namespace NUMINAMATH_CALUDE_smallest_a_for_equation_l2158_215892

theorem smallest_a_for_equation : 
  (∀ a : ℝ, a < -8 → ¬∃ b : ℝ, a^4 + 2*a^2*b + 2*a*b + b^2 = 960) ∧ 
  (∃ b : ℝ, (-8)^4 + 2*(-8)^2*b + 2*(-8)*b + b^2 = 960) := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_for_equation_l2158_215892


namespace NUMINAMATH_CALUDE_monomial_coefficient_and_degree_l2158_215872

/-- Represents a monomial with coefficient and variables -/
structure Monomial where
  coeff : ℚ
  vars : List (Char × ℕ)

/-- Calculate the degree of a monomial -/
def monomialDegree (m : Monomial) : ℕ :=
  m.vars.foldl (fun acc (_, exp) => acc + exp) 0

/-- The monomial -2/3 * a * b^2 -/
def mono : Monomial :=
  { coeff := -2/3
  , vars := [('a', 1), ('b', 2)] }

theorem monomial_coefficient_and_degree :
  mono.coeff = -2/3 ∧ monomialDegree mono = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_coefficient_and_degree_l2158_215872


namespace NUMINAMATH_CALUDE_most_stable_scores_l2158_215811

theorem most_stable_scores (S_A S_B S_C : ℝ) 
  (h1 : S_A = 38) (h2 : S_B = 10) (h3 : S_C = 26) :
  S_B < S_A ∧ S_B < S_C := by
  sorry

end NUMINAMATH_CALUDE_most_stable_scores_l2158_215811


namespace NUMINAMATH_CALUDE_parallel_to_y_axis_coordinates_l2158_215881

/-- Given two points M(a-3, a+4) and N(√5, 9) in a Cartesian coordinate system,
    if the line MN is parallel to the y-axis, then M has coordinates (√5, 7 + √5) -/
theorem parallel_to_y_axis_coordinates (a : ℝ) :
  let M : ℝ × ℝ := (a - 3, a + 4)
  let N : ℝ × ℝ := (Real.sqrt 5, 9)
  (M.1 = N.1) →  -- MN is parallel to y-axis iff x-coordinates are equal
  M = (Real.sqrt 5, 7 + Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_parallel_to_y_axis_coordinates_l2158_215881


namespace NUMINAMATH_CALUDE_line_separation_parameter_range_l2158_215848

/-- Given a line 2x - y + a = 0 where the origin (0, 0) and the point (1, 1) 
    are on opposite sides of this line, prove that -1 < a < 0 -/
theorem line_separation_parameter_range :
  ∀ a : ℝ, 
  (∀ x y : ℝ, 2*x - y + a = 0 → 
    ((0 : ℝ) < 2*0 - 0 + a) ≠ ((0 : ℝ) < 2*1 - 1 + a)) →
  -1 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_line_separation_parameter_range_l2158_215848


namespace NUMINAMATH_CALUDE_triangle_inequality_with_constant_l2158_215874

theorem triangle_inequality_with_constant (k : ℕ) : 
  (k > 0) → 
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  k = 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_constant_l2158_215874


namespace NUMINAMATH_CALUDE_x_squared_gt_y_squared_necessary_not_sufficient_l2158_215856

theorem x_squared_gt_y_squared_necessary_not_sufficient (x y : ℝ) :
  (∀ x y, x < y ∧ y < 0 → x^2 > y^2) ∧
  (∃ x y, x^2 > y^2 ∧ ¬(x < y ∧ y < 0)) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_gt_y_squared_necessary_not_sufficient_l2158_215856


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2158_215808

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x, x^2 + a*x - 3 ≤ 0 ↔ -1 ≤ x ∧ x ≤ 3) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2158_215808


namespace NUMINAMATH_CALUDE_stella_restocks_six_bathrooms_l2158_215846

/-- The number of bathrooms Stella restocks -/
def num_bathrooms : ℕ :=
  let rolls_per_day : ℕ := 1
  let days_per_week : ℕ := 7
  let num_weeks : ℕ := 4
  let rolls_per_pack : ℕ := 12
  let packs_bought : ℕ := 14
  let rolls_per_bathroom : ℕ := rolls_per_day * days_per_week * num_weeks
  let total_rolls_bought : ℕ := packs_bought * rolls_per_pack
  total_rolls_bought / rolls_per_bathroom

theorem stella_restocks_six_bathrooms : num_bathrooms = 6 := by
  sorry

end NUMINAMATH_CALUDE_stella_restocks_six_bathrooms_l2158_215846


namespace NUMINAMATH_CALUDE_sin_cos_power_six_sum_one_l2158_215842

theorem sin_cos_power_six_sum_one (α : Real) (h : Real.sin α + Real.cos α = 1) :
  Real.sin α ^ 6 + Real.cos α ^ 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_power_six_sum_one_l2158_215842


namespace NUMINAMATH_CALUDE_point_C_coordinates_l2158_215839

-- Define points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 5)

-- Define vector AB
def vecAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define vector AC in terms of AB
def vecAC : ℝ × ℝ := (2 * vecAB.1, 2 * vecAB.2)

-- Define point C
def C : ℝ × ℝ := (A.1 + vecAC.1, A.2 + vecAC.2)

-- Theorem to prove
theorem point_C_coordinates : C = (-3, 9) := by
  sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l2158_215839


namespace NUMINAMATH_CALUDE_triangle_properties_l2158_215878

/-- Given a triangle ABC with angle A = π/3 and perimeter 6, 
    prove the relation between sides and find the maximum area -/
theorem triangle_properties (b c : ℝ) (h_perimeter : b + c ≤ 6) : 
  b * c + 12 = 4 * (b + c) ∧ 
  (∀ (b' c' : ℝ), b' + c' ≤ 6 → 
    (1/2 : ℝ) * b' * c' * Real.sqrt 3 ≤ Real.sqrt 3) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l2158_215878


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2158_215877

theorem complex_modulus_problem (m : ℝ) :
  (Complex.I : ℂ) * Complex.I = -1 →
  (↑1 + m * Complex.I) * (↑3 + Complex.I) = Complex.I * (Complex.im ((↑1 + m * Complex.I) * (↑3 + Complex.I))) →
  Complex.abs ((↑m + ↑3 * Complex.I) / (↑1 - Complex.I)) = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2158_215877


namespace NUMINAMATH_CALUDE_fraction_equality_l2158_215866

theorem fraction_equality (a b : ℝ) (h : (1/a + 1/b)/(1/a - 1/b) = 2023) : (a + b)/(a - b) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2158_215866


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l2158_215845

def birthday_money (grandmother aunt uncle cousin brother : ℕ) : ℕ :=
  grandmother + aunt + uncle + cousin + brother

def total_in_wallet : ℕ := 185

def game_costs (game1 game2 game3 game4 game5 : ℕ) : ℕ :=
  game1 + game1 + game2 + game3 + game4 + game5

theorem money_left_after_purchase 
  (grandmother aunt uncle cousin brother : ℕ)
  (game1 game2 game3 game4 game5 : ℕ)
  (h1 : grandmother = 30)
  (h2 : aunt = 35)
  (h3 : uncle = 40)
  (h4 : cousin = 25)
  (h5 : brother = 20)
  (h6 : game1 = 30)
  (h7 : game2 = 40)
  (h8 : game3 = 35)
  (h9 : game4 = 25)
  (h10 : game5 = 0)  -- We use 0 for the fifth game as it's already counted in game1
  : total_in_wallet - (birthday_money grandmother aunt uncle cousin brother + game_costs game1 game2 game3 game4 game5) = 25 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l2158_215845


namespace NUMINAMATH_CALUDE_iesha_book_count_l2158_215853

/-- Represents the number of books Iesha has -/
structure IeshasBooks where
  school : ℕ
  sports : ℕ

/-- The total number of books Iesha has -/
def total_books (b : IeshasBooks) : ℕ := b.school + b.sports

theorem iesha_book_count : 
  ∀ (b : IeshasBooks), b.school = 19 → b.sports = 39 → total_books b = 58 := by
  sorry

end NUMINAMATH_CALUDE_iesha_book_count_l2158_215853


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2158_215858

theorem modulus_of_complex_fraction : 
  Complex.abs ((3 - 4 * Complex.I) / Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2158_215858


namespace NUMINAMATH_CALUDE_f_min_at_neg_seven_l2158_215869

/-- The quadratic function f(x) = x^2 + 14x + 24 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 24

/-- Theorem: The function f(x) = x^2 + 14x + 24 attains its minimum value when x = -7 -/
theorem f_min_at_neg_seven :
  ∀ x : ℝ, f x ≥ f (-7) :=
by sorry

end NUMINAMATH_CALUDE_f_min_at_neg_seven_l2158_215869


namespace NUMINAMATH_CALUDE_max_n_with_special_divisors_l2158_215851

theorem max_n_with_special_divisors (N : ℕ) : 
  (∃ (d : ℕ), d ∣ N ∧ d ≠ 1 ∧ d ≠ N ∧
   (∃ (a b : ℕ), a ∣ N ∧ b ∣ N ∧ a < b ∧
    (∀ (x : ℕ), x ∣ N → x < a ∨ x > b) ∧
    b = 21 * d)) →
  N ≤ 441 :=
sorry

end NUMINAMATH_CALUDE_max_n_with_special_divisors_l2158_215851


namespace NUMINAMATH_CALUDE_bhanu_house_rent_expenditure_l2158_215813

/-- Calculates Bhanu's expenditure on house rent based on his spending pattern -/
theorem bhanu_house_rent_expenditure (total_income : ℝ) 
  (h1 : 0.30 * total_income = 300) 
  (h2 : total_income > 0) : 
  0.14 * (total_income - 0.30 * total_income) = 98 := by
  sorry

end NUMINAMATH_CALUDE_bhanu_house_rent_expenditure_l2158_215813


namespace NUMINAMATH_CALUDE_system_solution_range_l2158_215863

theorem system_solution_range (x y a : ℝ) : 
  x + 3*y = 2 + a → 
  3*x + y = -4*a → 
  x + y > 2 → 
  a < -2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_range_l2158_215863


namespace NUMINAMATH_CALUDE_circle_through_point_touching_lines_l2158_215883

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space (ax + by + c = 0)
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Function to check if a circle touches a line
def touchesLine (c : Circle) (l : Line) : Prop := sorry

-- Function to check if a point lies on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop := sorry

-- Function to check if two lines are parallel
def areParallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem circle_through_point_touching_lines 
  (p : Point) (l1 l2 : Line) : 
  ∃ (c1 c2 : Circle), 
    (touchesLine c1 l1 ∧ touchesLine c1 l2 ∧ pointOnCircle p c1) ∧
    (touchesLine c2 l1 ∧ touchesLine c2 l2 ∧ pointOnCircle p c2) :=
by sorry

end NUMINAMATH_CALUDE_circle_through_point_touching_lines_l2158_215883


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l2158_215890

/-- Calculate simple interest -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Proof of simple interest calculation -/
theorem simple_interest_calculation :
  let principal : ℚ := 15000
  let rate : ℚ := 6
  let time : ℚ := 3
  simple_interest principal rate time = 2700 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l2158_215890


namespace NUMINAMATH_CALUDE_remainder_double_n_l2158_215800

theorem remainder_double_n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_double_n_l2158_215800


namespace NUMINAMATH_CALUDE_no_double_reverse_number_l2158_215843

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ+) : ℕ+ :=
  sorry

/-- Theorem: There does not exist a positive integer N such that 
    when its digits are reversed, the resulting number is exactly twice N -/
theorem no_double_reverse_number : ¬ ∃ (N : ℕ+), reverseDigits N = 2 * N := by
  sorry

end NUMINAMATH_CALUDE_no_double_reverse_number_l2158_215843


namespace NUMINAMATH_CALUDE_line_symmetry_l2158_215898

/-- The original line -/
def original_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- The line of symmetry -/
def symmetry_line (x : ℝ) : Prop := x = 1

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to the symmetry_line -/
theorem line_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  original_line x₁ y₁ →
  symmetric_line x₂ y₂ →
  ∃ (x_sym : ℝ),
    symmetry_line x_sym ∧
    x_sym - x₁ = x₂ - x_sym ∧
    y₁ = y₂ :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l2158_215898


namespace NUMINAMATH_CALUDE_area_of_smaller_triangle_l2158_215899

/-- Given an outer equilateral triangle with area 36 square units and an inner equilateral triangle
    with area 4 square units, if the space between these triangles is divided into four congruent
    triangles, then the area of each of these smaller triangles is 8 square units. -/
theorem area_of_smaller_triangle (outer_area inner_area : ℝ) (h1 : outer_area = 36)
    (h2 : inner_area = 4) (h3 : outer_area > inner_area) :
  (outer_area - inner_area) / 4 = 8 :=
by sorry

end NUMINAMATH_CALUDE_area_of_smaller_triangle_l2158_215899


namespace NUMINAMATH_CALUDE_cube_sum_eq_prime_product_solution_l2158_215889

theorem cube_sum_eq_prime_product_solution :
  ∀ (x y p : ℕ+), 
    x^3 + y^3 = p * (x * y + p) ∧ Nat.Prime p.val →
    ((x = 8 ∧ y = 1 ∧ p = 19) ∨
     (x = 1 ∧ y = 8 ∧ p = 19) ∨
     (x = 7 ∧ y = 2 ∧ p = 13) ∨
     (x = 2 ∧ y = 7 ∧ p = 13) ∨
     (x = 5 ∧ y = 4 ∧ p = 7) ∨
     (x = 4 ∧ y = 5 ∧ p = 7)) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_eq_prime_product_solution_l2158_215889


namespace NUMINAMATH_CALUDE_correct_electric_bicycle_volumes_l2158_215879

/-- Represents the parking data for a day --/
structure ParkingData where
  totalVolume : ℕ
  regularFeeBefore : ℚ
  electricFeeBefore : ℚ
  regularFeeAfter : ℚ
  electricFeeAfter : ℚ
  regularVolumeBefore : ℕ
  regularVolumeAfter : ℕ
  incomeFactor : ℚ

/-- Theorem stating the correct parking volumes for electric bicycles --/
theorem correct_electric_bicycle_volumes (data : ParkingData)
  (h1 : data.totalVolume = 6882)
  (h2 : data.regularFeeBefore = 1/5)
  (h3 : data.electricFeeBefore = 1/2)
  (h4 : data.regularFeeAfter = 2/5)
  (h5 : data.electricFeeAfter = 1)
  (h6 : data.regularVolumeBefore = 5180)
  (h7 : data.regularVolumeAfter = 335)
  (h8 : data.incomeFactor = 3/2) :
  ∃ (x y : ℕ),
    x + y = data.totalVolume - data.regularVolumeBefore - data.regularVolumeAfter ∧
    data.regularFeeBefore * data.regularVolumeBefore +
    data.regularFeeAfter * data.regularVolumeAfter +
    data.electricFeeBefore * x + data.electricFeeAfter * y =
    data.incomeFactor * (data.electricFeeBefore * x + data.electricFeeAfter * y) ∧
    x = 1174 ∧ y = 193 := by
  sorry


end NUMINAMATH_CALUDE_correct_electric_bicycle_volumes_l2158_215879


namespace NUMINAMATH_CALUDE_block_weight_difference_l2158_215860

/-- Given two blocks with different weights, prove the difference between their weights. -/
theorem block_weight_difference (yellow_weight green_weight : ℝ)
  (h1 : yellow_weight = 0.6)
  (h2 : green_weight = 0.4) :
  yellow_weight - green_weight = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_block_weight_difference_l2158_215860


namespace NUMINAMATH_CALUDE_poorly_chosen_character_Lobster_poorly_chosen_l2158_215876

/-- Represents a character in "Alice's Adventures in Wonderland" --/
structure Character where
  name : String
  is_active : Bool
  appears_in_poem : Bool

/-- Defines what it means for a character to be poorly chosen --/
def is_poorly_chosen (c : Character) : Prop :=
  c.appears_in_poem ∧ ¬c.is_active

/-- Theorem stating that a character is poorly chosen if it only appears in a poem and is not active --/
theorem poorly_chosen_character (c : Character) :
  c.appears_in_poem ∧ ¬c.is_active → is_poorly_chosen c := by
  sorry

/-- The Lobster character --/
def Lobster : Character :=
  { name := "Lobster",
    is_active := false,
    appears_in_poem := true }

/-- Theorem specifically about the Lobster being poorly chosen --/
theorem Lobster_poorly_chosen : is_poorly_chosen Lobster := by
  sorry

end NUMINAMATH_CALUDE_poorly_chosen_character_Lobster_poorly_chosen_l2158_215876


namespace NUMINAMATH_CALUDE_penalty_kicks_count_l2158_215897

theorem penalty_kicks_count (total_players : ℕ) (goalies : ℕ) : 
  total_players = 25 → goalies = 4 → total_players * goalies - goalies^2 = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_penalty_kicks_count_l2158_215897


namespace NUMINAMATH_CALUDE_lottery_expected_profit_l2158_215817

/-- The expected profit for buying one lottery ticket -/
theorem lottery_expected_profit :
  let ticket_cost : ℝ := 10
  let win_probability : ℝ := 0.02
  let prize : ℝ := 300
  let expected_profit := (prize - ticket_cost) * win_probability + (-ticket_cost) * (1 - win_probability)
  expected_profit = -4 := by sorry

end NUMINAMATH_CALUDE_lottery_expected_profit_l2158_215817


namespace NUMINAMATH_CALUDE_pascal_triangle_p_row_zeros_l2158_215859

theorem pascal_triangle_p_row_zeros (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : 1 ≤ k ∧ k ≤ p - 1) : 
  Nat.choose p k ≡ 0 [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_p_row_zeros_l2158_215859


namespace NUMINAMATH_CALUDE_class_size_proof_l2158_215875

theorem class_size_proof (boys_avg : ℝ) (girls_avg : ℝ) (class_avg : ℝ) (boys_girls_diff : ℕ) :
  boys_avg = 73 →
  girls_avg = 77 →
  class_avg = 74 →
  boys_girls_diff = 22 →
  ∃ (total_students : ℕ), total_students = 44 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l2158_215875


namespace NUMINAMATH_CALUDE_pentomino_circumscribing_rectangle_ratio_l2158_215835

/-- A pentomino is a planar geometric figure formed by joining five equal squares edge to edge. -/
structure Pentomino where
  -- Add necessary fields to represent a pentomino
  -- This is a placeholder and may need to be expanded based on specific requirements

/-- A rectangle that circumscribes a pentomino. -/
structure CircumscribingRectangle (p : Pentomino) where
  width : ℝ
  height : ℝ
  -- Add necessary fields to represent the relationship between the pentomino and the rectangle
  -- This is a placeholder and may need to be expanded based on specific requirements

/-- The theorem stating that for any pentomino inscribed in a rectangle, 
    the ratio of the shorter side to the longer side of the rectangle is 1:2. -/
theorem pentomino_circumscribing_rectangle_ratio (p : Pentomino) 
  (r : CircumscribingRectangle p) : 
  min r.width r.height / max r.width r.height = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pentomino_circumscribing_rectangle_ratio_l2158_215835


namespace NUMINAMATH_CALUDE_project_completion_time_l2158_215812

theorem project_completion_time
  (days_A : ℝ)
  (days_B : ℝ)
  (break_days : ℝ)
  (h1 : days_A = 18)
  (h2 : days_B = 15)
  (h3 : break_days = 4) :
  let efficiency_A := 1 / days_A
  let efficiency_B := 1 / days_B
  let combined_efficiency := efficiency_A + efficiency_B
  let work_during_break := efficiency_B * break_days
  (1 - work_during_break) / combined_efficiency + break_days = 10 :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l2158_215812


namespace NUMINAMATH_CALUDE_smallest_integer_in_special_set_l2158_215829

theorem smallest_integer_in_special_set : ∀ n : ℤ,
  (n + 6 > 2 * ((7 * n + 21) / 7)) →
  (∀ m : ℤ, m < n → m + 6 ≤ 2 * ((7 * m + 21) / 7)) →
  n = -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_special_set_l2158_215829


namespace NUMINAMATH_CALUDE_sqrt_neg_five_squared_l2158_215818

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_five_squared_l2158_215818


namespace NUMINAMATH_CALUDE_abs_inequality_necessary_not_sufficient_l2158_215895

theorem abs_inequality_necessary_not_sufficient (x : ℝ) :
  (x * (x - 2) < 0 → abs (x - 1) < 2) ∧
  ¬(abs (x - 1) < 2 → x * (x - 2) < 0) := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_necessary_not_sufficient_l2158_215895


namespace NUMINAMATH_CALUDE_calculate_Y_l2158_215836

theorem calculate_Y : 
  let P : ℚ := 208 / 4
  let Q : ℚ := P / 2
  let Y : ℚ := P - Q * (10 / 100)
  Y = 49.4 := by sorry

end NUMINAMATH_CALUDE_calculate_Y_l2158_215836


namespace NUMINAMATH_CALUDE_hawk_breeding_theorem_l2158_215850

/-- Given information about hawk breeding --/
structure HawkBreeding where
  num_kettles : ℕ
  pregnancies_per_kettle : ℕ
  survival_rate : ℚ
  expected_babies : ℕ

/-- Calculate the number of babies yielded per batch before loss --/
def babies_per_batch (h : HawkBreeding) : ℚ :=
  (h.expected_babies : ℚ) / h.survival_rate / ((h.num_kettles * h.pregnancies_per_kettle) : ℚ)

/-- Theorem stating the number of babies yielded per batch --/
theorem hawk_breeding_theorem (h : HawkBreeding) 
  (h_kettles : h.num_kettles = 6)
  (h_pregnancies : h.pregnancies_per_kettle = 15)
  (h_survival : h.survival_rate = 3/4)
  (h_expected : h.expected_babies = 270) :
  babies_per_batch h = 4 := by
  sorry


end NUMINAMATH_CALUDE_hawk_breeding_theorem_l2158_215850


namespace NUMINAMATH_CALUDE_mom_bought_51_shirts_l2158_215844

/-- The number of t-shirts in a package -/
def shirts_per_package : ℕ := 3

/-- The number of packages if t-shirts were purchased in packages -/
def num_packages : ℕ := 17

/-- The total number of t-shirts Mom bought -/
def total_shirts : ℕ := shirts_per_package * num_packages

theorem mom_bought_51_shirts : total_shirts = 51 := by
  sorry

end NUMINAMATH_CALUDE_mom_bought_51_shirts_l2158_215844


namespace NUMINAMATH_CALUDE_heather_walking_distance_l2158_215896

/-- The total distance Heather walked at the county fair -/
theorem heather_walking_distance :
  let car_to_entrance : ℚ := 0.3333333333333333
  let entrance_to_rides : ℚ := 0.3333333333333333
  let rides_to_car : ℚ := 0.08333333333333333
  car_to_entrance + entrance_to_rides + rides_to_car = 0.75
:= by sorry

end NUMINAMATH_CALUDE_heather_walking_distance_l2158_215896


namespace NUMINAMATH_CALUDE_distance_at_time_l2158_215861

/-- Represents a right-angled triangle with given hypotenuse and leg lengths -/
structure RightTriangle where
  hypotenuse : ℝ
  leg : ℝ

/-- Represents a moving point with a given speed -/
structure MovingPoint where
  speed : ℝ

theorem distance_at_time (triangle : RightTriangle) (point1 point2 : MovingPoint) :
  triangle.hypotenuse = 85 →
  triangle.leg = 75 →
  point1.speed = 8.5 →
  point2.speed = 5 →
  ∃ t : ℝ, t = 4 ∧ 
    let d1 := triangle.hypotenuse - point1.speed * t
    let d2 := triangle.leg - point2.speed * t
    d1 * d1 + d2 * d2 = 26 * 26 :=
by sorry

end NUMINAMATH_CALUDE_distance_at_time_l2158_215861


namespace NUMINAMATH_CALUDE_special_triangle_property_l2158_215831

noncomputable section

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the angles of the triangle
def angle_A (t : Triangle) : ℝ := Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))
def angle_B (t : Triangle) : ℝ := Real.arccos ((t.c^2 + t.a^2 - t.b^2) / (2 * t.c * t.a))
def angle_C (t : Triangle) : ℝ := Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

-- The main theorem
theorem special_triangle_property (t : Triangle) 
  (h : t.b * (t.a + t.b) * (t.b + t.c) = t.a^3 + t.b * (t.a^2 + t.c^2) + t.c^3) :
  1 / (Real.sqrt (angle_A t) + Real.sqrt (angle_B t)) + 
  1 / (Real.sqrt (angle_B t) + Real.sqrt (angle_C t)) = 
  2 / (Real.sqrt (angle_C t) + Real.sqrt (angle_A t)) :=
sorry

end

end NUMINAMATH_CALUDE_special_triangle_property_l2158_215831


namespace NUMINAMATH_CALUDE_people_left_at_table_l2158_215893

theorem people_left_at_table (initial_people : ℕ) (people_who_left : ℕ) : 
  initial_people = 11 → people_who_left = 6 → initial_people - people_who_left = 5 :=
by sorry

end NUMINAMATH_CALUDE_people_left_at_table_l2158_215893


namespace NUMINAMATH_CALUDE_regression_analysis_conclusions_l2158_215857

-- Define the regression model
structure RegressionModel where
  R_squared : ℝ
  sum_of_squares_residuals : ℝ
  residual_plot : Set (ℝ × ℝ)

-- Define the concept of model fit
def better_fit (model1 model2 : RegressionModel) : Prop := sorry

-- Define the concept of evenly scattered residuals
def evenly_scattered_residuals (plot : Set (ℝ × ℝ)) : Prop := sorry

-- Define the concept of horizontal band
def horizontal_band (plot : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem stating the correct conclusions
theorem regression_analysis_conclusions 
  (model1 model2 : RegressionModel) (ε : ℝ) (hε : ε > 0) :
  -- Higher R² indicates better fit
  (model1.R_squared > model2.R_squared + ε → better_fit model1 model2) ∧ 
  -- Smaller sum of squares of residuals indicates better fit
  (model1.sum_of_squares_residuals < model2.sum_of_squares_residuals - ε → 
    better_fit model1 model2) ∧
  -- Evenly scattered residuals around a horizontal band indicate appropriate model
  (evenly_scattered_residuals model1.residual_plot ∧ 
   horizontal_band model1.residual_plot → 
   better_fit model1 model2) := by sorry


end NUMINAMATH_CALUDE_regression_analysis_conclusions_l2158_215857


namespace NUMINAMATH_CALUDE_original_deck_size_l2158_215849

/-- Represents a deck of cards with red and black cards -/
structure Deck where
  red : ℕ
  black : ℕ

/-- The probability of selecting a red card from the deck -/
def redProbability (d : Deck) : ℚ :=
  d.red / (d.red + d.black)

theorem original_deck_size :
  ∃ d : Deck,
    redProbability d = 2/5 ∧
    redProbability {red := d.red + 3, black := d.black} = 1/2 ∧
    d.red + d.black = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_deck_size_l2158_215849


namespace NUMINAMATH_CALUDE_scientific_notation_16907_l2158_215802

theorem scientific_notation_16907 :
  16907 = 1.6907 * (10 : ℝ)^4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_16907_l2158_215802


namespace NUMINAMATH_CALUDE_real_axis_length_l2158_215867

/-- A hyperbola with equation x²/a² - y²/b² = 1/4 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The standard hyperbola with equation x²/9 - y²/16 = 1 -/
def standard_hyperbola : Hyperbola where
  a := 3
  b := 4
  h_positive := by norm_num

theorem real_axis_length
  (C : Hyperbola)
  (h_asymptotes : C.a / C.b = standard_hyperbola.a / standard_hyperbola.b)
  (h_point : C.a^2 * 9 - C.b^2 * 12 = C.a^2 * C.b^2) :
  2 * C.a = 3 := by
  sorry

#check real_axis_length

end NUMINAMATH_CALUDE_real_axis_length_l2158_215867


namespace NUMINAMATH_CALUDE_jeff_pickup_cost_l2158_215825

/-- The cost of last year's costume in dollars -/
def last_year_cost : ℝ := 250

/-- The percentage increase in cost compared to last year -/
def cost_increase_percent : ℝ := 0.4

/-- The deposit percentage -/
def deposit_percent : ℝ := 0.1

/-- The total cost of this year's costume -/
def total_cost : ℝ := last_year_cost * (1 + cost_increase_percent)

/-- The amount of the deposit -/
def deposit : ℝ := total_cost * deposit_percent

/-- The amount Jeff paid when picking up the costume -/
def pickup_cost : ℝ := total_cost - deposit

theorem jeff_pickup_cost : pickup_cost = 315 := by
  sorry

end NUMINAMATH_CALUDE_jeff_pickup_cost_l2158_215825


namespace NUMINAMATH_CALUDE_last_digit_of_sum_l2158_215821

/-- Given a = 25 and b = -3, the last digit of a^1999 + b^2002 is 4 -/
theorem last_digit_of_sum (a b : ℤ) : a = 25 ∧ b = -3 → (a^1999 + b^2002) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_sum_l2158_215821


namespace NUMINAMATH_CALUDE_triangle_properties_l2158_215887

open Real

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle conditions
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a / sin A = b / sin B ∧
  b / sin B = c / sin C →
  -- Part 1
  (b = a * cos C + (1/2) * c → A = π/3) ∧
  -- Part 2
  (b * cos C + c * cos B = Real.sqrt 7 ∧ b = 2 → c = 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2158_215887


namespace NUMINAMATH_CALUDE_marble_collection_l2158_215822

theorem marble_collection (total : ℕ) (friend_total : ℕ) : 
  (40 : ℚ) / 100 * total + (20 : ℚ) / 100 * total + (40 : ℚ) / 100 * total = total →
  (40 : ℚ) / 100 * friend_total = 2 →
  friend_total = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_collection_l2158_215822


namespace NUMINAMATH_CALUDE_inequality_solution_l2158_215894

theorem inequality_solution (x : ℝ) : 
  (x + 1 ≠ 0) → ((2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2158_215894


namespace NUMINAMATH_CALUDE_white_sox_wins_l2158_215815

theorem white_sox_wins (total_games : ℕ) (games_lost : ℕ) (win_difference : ℕ) : 
  total_games = 162 →
  games_lost = 63 →
  win_difference = 36 →
  total_games = games_lost + (games_lost + win_difference) →
  games_lost + win_difference = 99 := by
sorry

end NUMINAMATH_CALUDE_white_sox_wins_l2158_215815


namespace NUMINAMATH_CALUDE_fair_spending_theorem_l2158_215833

/-- Calculates the remaining amount after spending at the fair -/
def remaining_amount (initial : ℕ) (snacks : ℕ) (rides_multiplier : ℕ) (games : ℕ) : ℕ :=
  initial - (snacks + rides_multiplier * snacks + games)

/-- Theorem stating that the remaining amount is 10 dollars -/
theorem fair_spending_theorem (initial : ℕ) (snacks : ℕ) (rides_multiplier : ℕ) (games : ℕ)
  (h1 : initial = 80)
  (h2 : snacks = 15)
  (h3 : rides_multiplier = 3)
  (h4 : games = 10) :
  remaining_amount initial snacks rides_multiplier games = 10 := by
  sorry

end NUMINAMATH_CALUDE_fair_spending_theorem_l2158_215833


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2158_215871

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x + 100 > 0) ↔ (0 ≤ m ∧ m < 400) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2158_215871


namespace NUMINAMATH_CALUDE_digit_before_y_l2158_215870

/-- Given a number of the form xy86038 where x and y are single digits,
    if y = 3 and the number is divisible by 11,
    then x = 6 -/
theorem digit_before_y (x y : ℕ) : 
  y = 3 →
  x < 10 →
  y < 10 →
  (x * 1000000 + y * 100000 + 86038) % 11 = 0 →
  (∀ z < y, (x * 1000000 + z * 100000 + 86038) % 11 ≠ 0) →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_digit_before_y_l2158_215870


namespace NUMINAMATH_CALUDE_difference_of_squares_l2158_215807

theorem difference_of_squares (a b : ℝ) : (a - b) * (-a - b) = b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2158_215807


namespace NUMINAMATH_CALUDE_f_properties_l2158_215847

noncomputable section

def f (x : ℝ) : ℝ := x^2 * Real.log x - x + 1

theorem f_properties :
  (∀ x > 0, f x = x^2 * Real.log x - x + 1) →
  f (Real.exp 1) = Real.exp 2 - Real.exp 1 + 1 ∧
  (deriv f) 1 = 0 ∧
  (∀ x ≥ 1, f x ≥ (x - 1)^2) ∧
  (∀ m > 3/2, ∃ x ≥ 1, f x < m * (x - 1)^2) ∧
  (∀ m ≤ 3/2, ∀ x ≥ 1, f x ≥ m * (x - 1)^2) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l2158_215847


namespace NUMINAMATH_CALUDE_pizza_expense_proof_l2158_215873

/-- Proves that given a total expense of $465 on pizzas in May (31 days),
    and assuming equal daily consumption, the daily expense on pizzas is $15. -/
theorem pizza_expense_proof (total_expense : ℕ) (days_in_may : ℕ) (daily_expense : ℕ) :
  total_expense = 465 →
  days_in_may = 31 →
  daily_expense * days_in_may = total_expense →
  daily_expense = 15 := by
sorry

end NUMINAMATH_CALUDE_pizza_expense_proof_l2158_215873


namespace NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l2158_215806

theorem units_digit_of_n_squared_plus_two_to_n (n : ℕ) : 
  n = 1234^2 + 2^1234 → (n^2 + 2^n) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l2158_215806


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2158_215884

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x > -1}

-- Define set B
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | -1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2158_215884


namespace NUMINAMATH_CALUDE_total_travel_time_l2158_215885

theorem total_travel_time (total_distance : ℝ) (initial_time : ℝ) (lunch_time : ℝ) 
  (h1 : total_distance = 200)
  (h2 : initial_time = 1)
  (h3 : lunch_time = 1)
  (h4 : initial_time * 4 * total_distance / 4 = total_distance) :
  initial_time + lunch_time + (total_distance - total_distance / 4) / (total_distance / 4 / initial_time) = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_l2158_215885


namespace NUMINAMATH_CALUDE_arctan_sum_equation_l2158_215888

theorem arctan_sum_equation (n : ℕ+) : 
  (Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/6) + Real.arctan (1/(n : ℝ)) = π/4) ↔ n = 57 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_l2158_215888


namespace NUMINAMATH_CALUDE_equation_solution_l2158_215805

theorem equation_solution :
  ∀ x : ℚ, (25 - 7 : ℚ) = 5/2 + x → x = 31/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2158_215805


namespace NUMINAMATH_CALUDE_distorted_polygon_sides_l2158_215854

/-- A regular polygon with a distorted exterior angle -/
structure DistortedPolygon where
  -- The apparent exterior angle in degrees
  apparent_angle : ℝ
  -- The distortion factor
  distortion_factor : ℝ
  -- The number of sides
  sides : ℕ

/-- The theorem stating the number of sides for the given conditions -/
theorem distorted_polygon_sides (p : DistortedPolygon) 
  (h1 : p.apparent_angle = 18)
  (h2 : p.distortion_factor = 1.5)
  (h3 : p.apparent_angle * p.sides = 360 * p.distortion_factor) : 
  p.sides = 30 := by
  sorry

end NUMINAMATH_CALUDE_distorted_polygon_sides_l2158_215854


namespace NUMINAMATH_CALUDE_gumble_words_count_l2158_215852

def alphabet_size : ℕ := 25
def max_word_length : ℕ := 5

def words_with_b (n : ℕ) : ℕ :=
  alphabet_size^n - (alphabet_size - 1)^n

def total_words : ℕ :=
  words_with_b 1 + words_with_b 2 + words_with_b 3 + words_with_b 4 + words_with_b 5

theorem gumble_words_count :
  total_words = 1863701 :=
by sorry

end NUMINAMATH_CALUDE_gumble_words_count_l2158_215852


namespace NUMINAMATH_CALUDE_inequality_proof_l2158_215891

theorem inequality_proof (a b : ℝ) (h : a < b) : 1 - a > 1 - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2158_215891


namespace NUMINAMATH_CALUDE_major_axis_length_l2158_215830

/-- Represents an ellipse formed by the intersection of a plane and a right circular cylinder. -/
structure IntersectionEllipse where
  cylinder_radius : ℝ
  major_axis : ℝ
  minor_axis : ℝ

/-- The theorem stating the length of the major axis given the conditions. -/
theorem major_axis_length 
  (e : IntersectionEllipse) 
  (h1 : e.cylinder_radius = 2)
  (h2 : e.minor_axis = 2 * e.cylinder_radius)
  (h3 : e.major_axis = e.minor_axis * (1 + 0.75)) :
  e.major_axis = 7 := by
  sorry


end NUMINAMATH_CALUDE_major_axis_length_l2158_215830


namespace NUMINAMATH_CALUDE_bookshop_inventory_l2158_215823

/-- Bookshop inventory problem -/
theorem bookshop_inventory (initial_books : ℕ) (saturday_instore : ℕ) (saturday_online : ℕ) 
  (sunday_instore : ℕ) (shipment : ℕ) (final_books : ℕ) 
  (h1 : initial_books = 743)
  (h2 : saturday_instore = 37)
  (h3 : saturday_online = 128)
  (h4 : sunday_instore = 2 * saturday_instore)
  (h5 : shipment = 160)
  (h6 : final_books = 502) :
  ∃ (sunday_online : ℕ), 
    final_books = initial_books - (saturday_instore + saturday_online + sunday_instore + sunday_online) + shipment ∧ 
    sunday_online = saturday_online + 34 :=
by sorry

end NUMINAMATH_CALUDE_bookshop_inventory_l2158_215823


namespace NUMINAMATH_CALUDE_show_receipts_l2158_215868

/-- Calculates the total receipts for a show given ticket prices and attendance. -/
def totalReceipts (adultPrice childPrice : ℚ) (numAdults : ℕ) : ℚ :=
  let numChildren := numAdults / 2
  adultPrice * numAdults + childPrice * numChildren

/-- Theorem stating that the total receipts for the show are 1026 dollars. -/
theorem show_receipts :
  totalReceipts (5.5) (2.5) 152 = 1026 := by
  sorry

#eval totalReceipts (5.5) (2.5) 152

end NUMINAMATH_CALUDE_show_receipts_l2158_215868
