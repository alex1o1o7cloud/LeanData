import Mathlib

namespace different_color_sock_pairs_l38_3854

theorem different_color_sock_pairs (white : ℕ) (brown : ℕ) (blue : ℕ) : 
  white = 5 → brown = 4 → blue = 3 → 
  (white * brown + brown * blue + white * blue = 47) :=
by
  sorry

end different_color_sock_pairs_l38_3854


namespace solution_set_f_greater_than_4_range_of_a_l38_3834

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 3| + |x - 1|

-- Theorem for part (I)
theorem solution_set_f_greater_than_4 :
  {x : ℝ | f x > 4} = {x : ℝ | x < -2 ∨ x > 0} := by sorry

-- Theorem for part (II)
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc (-3/2) 1, a + 1 > f x) → a > 3/2 := by sorry

end solution_set_f_greater_than_4_range_of_a_l38_3834


namespace aunts_gift_amount_l38_3878

def shirts_cost : ℕ := 5
def shirts_price : ℕ := 5
def pants_price : ℕ := 26
def remaining_money : ℕ := 20

theorem aunts_gift_amount : 
  shirts_cost * shirts_price + pants_price + remaining_money = 71 := by
  sorry

end aunts_gift_amount_l38_3878


namespace exponential_inequality_l38_3896

theorem exponential_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (3 : ℝ)^b < (3 : ℝ)^a ∧ (3 : ℝ)^a < (4 : ℝ)^a :=
by sorry

end exponential_inequality_l38_3896


namespace mistaken_subtraction_l38_3872

theorem mistaken_subtraction (x : ℤ) : x - 64 = 122 → x - 46 = 140 := by
  sorry

end mistaken_subtraction_l38_3872


namespace sin_cos_sum_fifteen_seventyfive_degrees_l38_3876

theorem sin_cos_sum_fifteen_seventyfive_degrees : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (75 * π / 180) = 1 := by
  sorry

end sin_cos_sum_fifteen_seventyfive_degrees_l38_3876


namespace garden_length_l38_3809

/-- Proves that a rectangular garden with length twice its width and perimeter 180 yards has a length of 60 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- Length is twice the width
  2 * width + 2 * length = 180 →  -- Perimeter is 180 yards
  length = 60 := by
sorry


end garden_length_l38_3809


namespace largest_divisor_of_n_squared_div_7200_l38_3812

theorem largest_divisor_of_n_squared_div_7200 (n : ℕ) (h1 : n > 0) (h2 : 7200 ∣ n^2) :
  (60 ∣ n) ∧ ∀ k : ℕ, k ∣ n → k ≤ 60 :=
by sorry

end largest_divisor_of_n_squared_div_7200_l38_3812


namespace no_odd_4digit_div5_no05_l38_3825

theorem no_odd_4digit_div5_no05 : 
  ¬ ∃ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧  -- 4-digit
    n % 2 = 1 ∧             -- odd
    n % 5 = 0 ∧             -- divisible by 5
    (∀ d : ℕ, d < 4 → (n / 10^d) % 10 ≠ 0 ∧ (n / 10^d) % 10 ≠ 5) -- no 0 or 5 digits
    := by sorry

end no_odd_4digit_div5_no05_l38_3825


namespace same_color_combination_probability_l38_3814

def total_candies : ℕ := 12 + 8 + 5

theorem same_color_combination_probability :
  let red : ℕ := 12
  let blue : ℕ := 8
  let green : ℕ := 5
  let total : ℕ := total_candies
  
  -- Probability of picking two red candies
  let p_red : ℚ := (red * (red - 1)) / (total * (total - 1)) *
                   ((red - 2) * (red - 3)) / ((total - 2) * (total - 3))
  
  -- Probability of picking two blue candies
  let p_blue : ℚ := (blue * (blue - 1)) / (total * (total - 1)) *
                    ((blue - 2) * (blue - 3)) / ((total - 2) * (total - 3))
  
  -- Probability of picking two green candies
  let p_green : ℚ := (green * (green - 1)) / (total * (total - 1)) *
                     ((green - 2) * (green - 3)) / ((total - 2) * (total - 3))
  
  -- Total probability of picking the same color combination
  p_red + p_blue + p_green = 11 / 77 :=
by sorry

end same_color_combination_probability_l38_3814


namespace greatest_four_digit_divisible_by_63_and_9_l38_3867

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a four-digit integer -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_63_and_9 :
  ∃ (p : ℕ), 
    isFourDigit p ∧ 
    isFourDigit (reverseDigits p) ∧ 
    p % 63 = 0 ∧ 
    (reverseDigits p) % 63 = 0 ∧ 
    p % 9 = 0 ∧
    ∀ (x : ℕ), 
      isFourDigit x ∧ 
      isFourDigit (reverseDigits x) ∧ 
      x % 63 = 0 ∧ 
      (reverseDigits x) % 63 = 0 ∧ 
      x % 9 = 0 → 
      x ≤ p ∧
    p = 9507 := by
  sorry

end greatest_four_digit_divisible_by_63_and_9_l38_3867


namespace toy_bear_production_efficiency_l38_3802

theorem toy_bear_production_efficiency (B H : ℝ) (H' : ℝ) : 
  B > 0 → H > 0 →
  (1.8 * B = 2 * (B / H) * H') →
  (H - H') / H * 100 = 10 :=
by sorry

end toy_bear_production_efficiency_l38_3802


namespace paul_crayons_left_l38_3837

/-- The number of crayons Paul had at the end of the school year -/
def crayons_left (initial_crayons lost_crayons : ℕ) : ℕ :=
  initial_crayons - lost_crayons

/-- Theorem: Paul had 291 crayons left at the end of the school year -/
theorem paul_crayons_left : crayons_left 606 315 = 291 := by
  sorry

end paul_crayons_left_l38_3837


namespace expression_equals_zero_l38_3831

theorem expression_equals_zero (x y z : ℝ) (h : x*y + y*z + z*x = 0) :
  3*x*y*z + x^2*(y + z) + y^2*(z + x) + z^2*(x + y) = 0 := by sorry

end expression_equals_zero_l38_3831


namespace clock_90_degree_times_l38_3870

/-- The angle between the hour hand and minute hand at time t minutes after 12:00 -/
def angle_between (t : ℝ) : ℝ :=
  |6 * t - 0.5 * t|

/-- The times when the hour hand and minute hand form a 90° angle after 12:00 -/
theorem clock_90_degree_times :
  ∃ (t₁ t₂ : ℝ), t₁ < t₂ ∧
  angle_between t₁ = 90 ∧
  angle_between t₂ = 90 ∧
  t₁ = 180 / 11 ∧
  t₂ = 540 / 11 :=
sorry

end clock_90_degree_times_l38_3870


namespace equation_solution_l38_3873

theorem equation_solution :
  ∃ x : ℝ, (4 * x + 6 * x = 360 - 9 * (x - 4)) ∧ (x = 396 / 19) := by
  sorry

end equation_solution_l38_3873


namespace fifth_grade_class_size_is_correct_l38_3828

/-- Represents the number of students in each fifth grade class -/
def fifth_grade_class_size : ℕ := 27

/-- Represents the total number of third grade classes -/
def third_grade_classes : ℕ := 5

/-- Represents the number of students in each third grade class -/
def third_grade_class_size : ℕ := 30

/-- Represents the total number of fourth grade classes -/
def fourth_grade_classes : ℕ := 4

/-- Represents the number of students in each fourth grade class -/
def fourth_grade_class_size : ℕ := 28

/-- Represents the total number of fifth grade classes -/
def fifth_grade_classes : ℕ := 4

/-- Represents the cost of a hamburger in cents -/
def hamburger_cost : ℕ := 210

/-- Represents the cost of carrots in cents -/
def carrots_cost : ℕ := 50

/-- Represents the cost of a cookie in cents -/
def cookie_cost : ℕ := 20

/-- Represents the total cost of all students' lunches in cents -/
def total_lunch_cost : ℕ := 103600

theorem fifth_grade_class_size_is_correct : 
  fifth_grade_class_size * fifth_grade_classes * (hamburger_cost + carrots_cost + cookie_cost) + 
  third_grade_classes * third_grade_class_size * (hamburger_cost + carrots_cost + cookie_cost) + 
  fourth_grade_classes * fourth_grade_class_size * (hamburger_cost + carrots_cost + cookie_cost) = 
  total_lunch_cost :=
by sorry

end fifth_grade_class_size_is_correct_l38_3828


namespace tesseract_triangles_l38_3845

/-- The number of vertices in a tesseract -/
def tesseract_vertices : ℕ := 16

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles in a tesseract -/
def distinct_triangles : ℕ := Nat.choose tesseract_vertices triangle_vertices

theorem tesseract_triangles : distinct_triangles = 560 := by
  sorry

end tesseract_triangles_l38_3845


namespace number_operation_result_l38_3823

theorem number_operation_result (x : ℝ) : (x - 5) / 7 = 7 → (x - 24) / 10 = 3 := by
  sorry

end number_operation_result_l38_3823


namespace marked_price_calculation_jobber_pricing_l38_3836

theorem marked_price_calculation (original_price : ℝ) (discount_percent : ℝ) 
  (gain_percent : ℝ) (final_discount_percent : ℝ) : ℝ :=
  let purchase_price := original_price * (1 - discount_percent / 100)
  let selling_price := purchase_price * (1 + gain_percent / 100)
  let marked_price := selling_price / (1 - final_discount_percent / 100)
  marked_price

theorem jobber_pricing : 
  marked_price_calculation 30 15 50 25 = 51 := by
  sorry

end marked_price_calculation_jobber_pricing_l38_3836


namespace volunteer_schedule_lcm_l38_3857

theorem volunteer_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 9 10))) = 360 := by
  sorry

end volunteer_schedule_lcm_l38_3857


namespace age_of_other_man_l38_3886

/-- Proves that the age of the other replaced man is 20 years old given the problem conditions -/
theorem age_of_other_man (n : ℕ) (avg_increase : ℝ) (age_one_man : ℕ) (avg_age_women : ℝ) : 
  n = 8 ∧ 
  avg_increase = 2 ∧ 
  age_one_man = 22 ∧ 
  avg_age_women = 29 → 
  ∃ (age_other_man : ℕ), 
    age_other_man = 20 ∧ 
    2 * avg_age_women - (age_one_man + age_other_man) = n * avg_increase :=
by sorry

end age_of_other_man_l38_3886


namespace trapezoid_circles_problem_l38_3869

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Check if four points are concyclic -/
def are_concyclic (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Check if a circle is tangent to a line -/
def is_tangent (c : Circle) (l : Line) : Prop := sorry

/-- Check if two line segments are parallel -/
def are_parallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculate the ratio of two line segments -/
def segment_ratio (p1 p2 p3 : Point) : ℝ := sorry

theorem trapezoid_circles_problem 
  (A B C D E : Point) 
  (circle1 circle2 : Circle) 
  (line_CD : Line) :
  are_parallel A D B C →
  E.x > B.x ∧ E.x < C.x →
  are_concyclic A C D E →
  circle2.center = circle1.center →
  is_tangent circle2 line_CD →
  distance A B = 12 →
  segment_ratio B E C = 4/5 →
  distance B C = 36 ∧ 
  2/3 < circle1.radius / circle2.radius ∧ 
  circle1.radius / circle2.radius < 4/3 := by sorry

end trapezoid_circles_problem_l38_3869


namespace wilson_family_ages_l38_3889

theorem wilson_family_ages : ∃ (w e j h t d : ℕ),
  (w > 0) ∧ (e > 0) ∧ (j > 0) ∧ (h > 0) ∧ (t > 0) ∧ (d > 0) ∧
  (w / 2 = e + j + h) ∧
  (w + 5 = (e + 5) + (j + 5) + (h + 5) + 0) ∧
  (e + j + h + t + d = 2 * w) ∧
  (w = e + j) ∧
  (e = t + d) := by
  sorry

end wilson_family_ages_l38_3889


namespace robin_water_bottles_l38_3800

/-- The number of additional bottles needed on the last day -/
def additional_bottles (total_bottles : ℕ) (daily_consumption : ℕ) : ℕ :=
  daily_consumption - (total_bottles % daily_consumption)

/-- Theorem stating that given 617 bottles and a daily consumption of 9 bottles, 
    4 additional bottles are needed on the last day -/
theorem robin_water_bottles : additional_bottles 617 9 = 4 := by
  sorry

end robin_water_bottles_l38_3800


namespace third_train_speed_l38_3840

/-- Calculates the speed of the third train given the conditions of the problem -/
theorem third_train_speed
  (man_train_speed : ℝ)
  (goods_train_length : ℝ)
  (third_train_length : ℝ)
  (goods_train_pass_time : ℝ)
  (third_train_pass_time : ℝ)
  (h_man_train_speed : man_train_speed = 45)
  (h_goods_train_length : goods_train_length = 340)
  (h_third_train_length : third_train_length = 480)
  (h_goods_train_pass_time : goods_train_pass_time = 8)
  (h_third_train_pass_time : third_train_pass_time = 12) :
  ∃ (third_train_speed : ℝ), third_train_speed = 99 := by
  sorry


end third_train_speed_l38_3840


namespace opinion_change_percentage_l38_3898

theorem opinion_change_percentage
  (physics_initial_enjoy : ℝ)
  (physics_initial_dislike : ℝ)
  (physics_final_enjoy : ℝ)
  (physics_final_dislike : ℝ)
  (chem_initial_enjoy : ℝ)
  (chem_initial_dislike : ℝ)
  (chem_final_enjoy : ℝ)
  (chem_final_dislike : ℝ)
  (h1 : physics_initial_enjoy = 40)
  (h2 : physics_initial_dislike = 60)
  (h3 : physics_final_enjoy = 75)
  (h4 : physics_final_dislike = 25)
  (h5 : chem_initial_enjoy = 30)
  (h6 : chem_initial_dislike = 70)
  (h7 : chem_final_enjoy = 65)
  (h8 : chem_final_dislike = 35)
  (h9 : physics_initial_enjoy + physics_initial_dislike = 100)
  (h10 : physics_final_enjoy + physics_final_dislike = 100)
  (h11 : chem_initial_enjoy + chem_initial_dislike = 100)
  (h12 : chem_final_enjoy + chem_final_dislike = 100) :
  ∃ (min_change max_change : ℝ),
    min_change = 70 ∧
    max_change = 70 ∧
    (∀ (actual_change : ℝ),
      actual_change ≥ min_change ∧
      actual_change ≤ max_change) :=
by sorry

end opinion_change_percentage_l38_3898


namespace final_cell_count_l38_3874

/-- Calculates the number of cells after a given number of days, 
    given an initial population and a tripling period. -/
def cell_population (initial_cells : ℕ) (tripling_period : ℕ) (total_days : ℕ) : ℕ :=
  initial_cells * (3 ^ (total_days / tripling_period))

/-- Theorem stating that given the specific conditions of the problem, 
    the final cell population after 9 days is 45. -/
theorem final_cell_count : cell_population 5 3 9 = 45 := by
  sorry

end final_cell_count_l38_3874


namespace unique_correct_expression_l38_3811

theorem unique_correct_expression :
  ((-3 - 1 = -2) = False) ∧
  ((-2 * (-1/2) = 1) = True) ∧
  ((16 / (-4/3) = 12) = False) ∧
  ((-3^2 / 4 = 9/4) = False) := by
  sorry

end unique_correct_expression_l38_3811


namespace square_properties_l38_3887

/-- Given a square with perimeter 48 feet, prove its side length and area. -/
theorem square_properties (perimeter : ℝ) (h : perimeter = 48) :
  ∃ (side_length area : ℝ),
    side_length = 12 ∧
    area = 144 ∧
    perimeter = 4 * side_length ∧
    area = side_length * side_length := by
  sorry

end square_properties_l38_3887


namespace cloth_sale_gain_percentage_l38_3891

/-- Calculates the gain percentage given the profit amount and total amount sold -/
def gainPercentage (profitAmount : ℕ) (totalAmount : ℕ) : ℚ :=
  (profitAmount : ℚ) / (totalAmount : ℚ) * 100

/-- Theorem: The gain percentage is 40% when the profit is 10 and the total amount sold is 25 -/
theorem cloth_sale_gain_percentage :
  gainPercentage 10 25 = 40 := by
  sorry

end cloth_sale_gain_percentage_l38_3891


namespace fraction_equality_l38_3864

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) 
  (h3 : a/b + (a+5*b)/(b+5*a) = 2) : a/b = 3/5 := by
  sorry

end fraction_equality_l38_3864


namespace homework_reduction_equation_l38_3819

theorem homework_reduction_equation 
  (initial_duration : ℝ) 
  (final_duration : ℝ) 
  (x : ℝ) 
  (h1 : initial_duration = 90) 
  (h2 : final_duration = 60) 
  (h3 : 0 ≤ x ∧ x < 1) : 
  initial_duration * (1 - x)^2 = final_duration := by
sorry

end homework_reduction_equation_l38_3819


namespace isosceles_triangle_angle_measure_l38_3838

/-- In an isosceles triangle DEF where angle D is congruent to angle E, 
    and the measure of angle E is three times the measure of angle F, 
    the measure of angle D is 540/7 degrees. -/
theorem isosceles_triangle_angle_measure (D E F : ℝ) : 
  D = E →                         -- Angle D is congruent to angle E
  E = 3 * F →                     -- Measure of angle E is three times the measure of angle F
  D + E + F = 180 →               -- Sum of angles in a triangle is 180 degrees
  D = 540 / 7 := by sorry         -- Measure of angle D is 540/7 degrees

end isosceles_triangle_angle_measure_l38_3838


namespace arithmetic_sequence_middle_term_l38_3803

theorem arithmetic_sequence_middle_term (z : ℤ) :
  (∃ (a d : ℤ), 3^2 = a ∧ z = a + d ∧ 3^3 = a + 2*d) → z = 18 := by
  sorry

end arithmetic_sequence_middle_term_l38_3803


namespace circle_pencil_theorem_l38_3818

/-- Definition of a circle in 2D plane -/
structure Circle where
  a : ℝ
  b : ℝ
  R : ℝ

/-- Left-hand side of circle equation -/
def K (C : Circle) (x y : ℝ) : ℝ :=
  (x - C.a)^2 + (y - C.b)^2 - C.R^2

/-- Type of circle pencil -/
inductive PencilType
  | Elliptic
  | Parabolic
  | Hyperbolic

/-- Theorem about circle pencils -/
theorem circle_pencil_theorem (C₁ C₂ : Circle) :
  ∃ (radical_axis : Set (ℝ × ℝ)) (pencil_type : PencilType),
    (∀ (t : ℝ), ∃ (C : Circle), ∀ (x y : ℝ),
      K C₁ x y + t * K C₂ x y = 0 ↔ K C x y = 0) ∧
    (∀ (C : Circle), (∀ (x y : ℝ), K C x y = 0 → (x, y) ∈ radical_axis) →
      ∃ (t : ℝ), ∀ (x y : ℝ), K C₁ x y + t * K C₂ x y = 0 ↔ K C x y = 0) ∧
    (pencil_type = PencilType.Elliptic →
      ∃ (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ K C₁ p₁.1 p₁.2 = 0 ∧ K C₂ p₁.1 p₁.2 = 0 ∧
                          K C₁ p₂.1 p₂.2 = 0 ∧ K C₂ p₂.1 p₂.2 = 0) ∧
    (pencil_type = PencilType.Parabolic →
      ∃ (p : ℝ × ℝ), K C₁ p.1 p.2 = 0 ∧ K C₂ p.1 p.2 = 0 ∧
        ∀ (ε : ℝ), ε > 0 → ∃ (q : ℝ × ℝ), q ≠ p ∧ 
          abs (K C₁ q.1 q.2) < ε ∧ abs (K C₂ q.1 q.2) < ε) ∧
    (pencil_type = PencilType.Hyperbolic →
      ∀ (x y : ℝ), K C₁ x y = 0 → K C₂ x y ≠ 0) :=
by sorry

end circle_pencil_theorem_l38_3818


namespace green_shirt_pairs_l38_3853

theorem green_shirt_pairs (red_students : ℕ) (green_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (red_red_pairs : ℕ) :
  red_students = 70 →
  green_students = 58 →
  total_students = 128 →
  total_pairs = 64 →
  red_red_pairs = 34 →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 28 :=
by
  sorry

end green_shirt_pairs_l38_3853


namespace log_inequality_l38_3850

theorem log_inequality (n : ℕ+) (k : ℕ) (h : k = (Nat.factors n).card) :
  Real.log n ≥ k * Real.log 2 := by
  sorry

end log_inequality_l38_3850


namespace absent_laborers_count_l38_3821

/-- Represents the number of laborers originally employed -/
def total_laborers : ℕ := 20

/-- Represents the original number of days planned to complete the work -/
def original_days : ℕ := 15

/-- Represents the actual number of days taken to complete the work -/
def actual_days : ℕ := 20

/-- Represents the total amount of work in laborer-days -/
def total_work : ℕ := total_laborers * original_days

/-- Calculates the number of absent laborers -/
def absent_laborers : ℕ := total_laborers - (total_work / actual_days)

theorem absent_laborers_count : absent_laborers = 5 := by
  sorry

end absent_laborers_count_l38_3821


namespace cats_to_dogs_ratio_l38_3890

theorem cats_to_dogs_ratio (cats : ℕ) (dogs : ℕ) : 
  cats = 16 → dogs = 8 → (cats : ℚ) / dogs = 2 := by
  sorry

end cats_to_dogs_ratio_l38_3890


namespace german_shepherd_vs_golden_retriever_pups_l38_3880

/-- The number of pups each breed has -/
structure DogBreedPups where
  husky : Nat
  golden_retriever : Nat
  pitbull : Nat
  german_shepherd : Nat

/-- The number of dogs James has for each breed -/
structure DogCounts where
  huskies : Nat
  golden_retrievers : Nat
  pitbulls : Nat
  german_shepherds : Nat

/-- Calculate the difference in total pups between German shepherds and golden retrievers -/
def pup_difference (breed_pups : DogBreedPups) (counts : DogCounts) : Int :=
  (breed_pups.german_shepherd * counts.german_shepherds) - 
  (breed_pups.golden_retriever * counts.golden_retrievers)

theorem german_shepherd_vs_golden_retriever_pups : 
  ∀ (breed_pups : DogBreedPups) (counts : DogCounts),
  counts.huskies = 5 →
  counts.pitbulls = 2 →
  counts.golden_retrievers = 4 →
  counts.german_shepherds = 3 →
  breed_pups.husky = 4 →
  breed_pups.golden_retriever = breed_pups.husky + 2 →
  breed_pups.pitbull = 3 →
  breed_pups.german_shepherd = breed_pups.pitbull + 3 →
  pup_difference breed_pups counts = -6 :=
by
  sorry

end german_shepherd_vs_golden_retriever_pups_l38_3880


namespace min_value_geometric_sequence_l38_3897

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence (a : ℕ → ℝ) 
    (h_geo : is_geometric_sequence a)
    (h_pos : ∀ n, a n > 0)
    (h_2018 : a 2018 = Real.sqrt 2 / 2) :
    (1 / a 2017 + 2 / a 2019) ≥ 4 ∧ 
    ∃ a, is_geometric_sequence a ∧ (∀ n, a n > 0) ∧ 
         a 2018 = Real.sqrt 2 / 2 ∧ 1 / a 2017 + 2 / a 2019 = 4 :=
by sorry

end min_value_geometric_sequence_l38_3897


namespace perpendicular_vector_implies_y_coord_l38_3805

/-- Given two points A and B, and a vector a, if AB is perpendicular to a, 
    then the y-coordinate of B is -4. -/
theorem perpendicular_vector_implies_y_coord (A B : ℝ × ℝ) (a : ℝ × ℝ) : 
  A = (-1, 2) → 
  B.1 = 2 → 
  a = (2, 1) → 
  (B.1 - A.1, B.2 - A.2) • a = 0 → 
  B.2 = -4 := by
sorry

end perpendicular_vector_implies_y_coord_l38_3805


namespace locus_is_circle_l38_3895

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

/-- Calculates the sum of squares of distances from a point to the vertices of an isosceles triangle -/
def sumOfSquaredDistances (p : Point) (t : IsoscelesTriangle) : ℝ :=
  3 * p.x^2 + 4 * p.y^2 - 2 * t.height * p.y + t.height^2 + t.base^2

/-- Theorem: The locus of points with constant sum of squared distances to the vertices of an isosceles triangle is a circle iff the sum exceeds h^2 + b^2 -/
theorem locus_is_circle (t : IsoscelesTriangle) (a : ℝ) :
  (∃ (center : Point) (radius : ℝ), ∀ (p : Point), 
    sumOfSquaredDistances p t = a ↔ (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2) ↔ 
  a > t.height^2 + t.base^2 := by
  sorry

end locus_is_circle_l38_3895


namespace rational_cube_equality_l38_3892

theorem rational_cube_equality (a b c : ℚ) 
  (eq1 : (a^2 + 1)^3 = b + 1)
  (eq2 : (b^2 + 1)^3 = c + 1)
  (eq3 : (c^2 + 1)^3 = a + 1) :
  a = 0 ∧ b = 0 ∧ c = 0 := by
sorry

end rational_cube_equality_l38_3892


namespace sequence_equality_l38_3806

theorem sequence_equality (a : Fin 100 → ℝ)
  (h1 : ∀ n : Fin 98, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := by
  sorry

end sequence_equality_l38_3806


namespace shoes_sold_main_theorem_l38_3894

/-- Represents the inventory of a shoe shop -/
structure ShoeInventory where
  large_boots : Nat
  medium_sandals : Nat
  small_sneakers : Nat
  large_sandals : Nat
  medium_boots : Nat
  small_boots : Nat

/-- Calculates the total number of shoes in the inventory -/
def total_shoes (inventory : ShoeInventory) : Nat :=
  inventory.large_boots + inventory.medium_sandals + inventory.small_sneakers +
  inventory.large_sandals + inventory.medium_boots + inventory.small_boots

/-- Theorem: The shop sold 106 pairs of shoes -/
theorem shoes_sold (initial_inventory : ShoeInventory) (pairs_left : Nat) : Nat :=
  let initial_total := total_shoes initial_inventory
  initial_total - pairs_left

/-- Main theorem: The shop sold 106 pairs of shoes -/
theorem main_theorem : shoes_sold
  { large_boots := 22
    medium_sandals := 32
    small_sneakers := 24
    large_sandals := 45
    medium_boots := 35
    small_boots := 26 }
  78 = 106 := by
  sorry

end shoes_sold_main_theorem_l38_3894


namespace negation_of_universal_proposition_l38_3832

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 + x < 0) :=
by sorry

end negation_of_universal_proposition_l38_3832


namespace tens_digit_of_9_pow_2023_l38_3843

theorem tens_digit_of_9_pow_2023 : ∃ n : ℕ, n ≥ 10 ∧ n < 100 ∧ 9^2023 % 100 = n ∧ (n / 10) % 10 = 2 := by
  sorry

end tens_digit_of_9_pow_2023_l38_3843


namespace minimize_plates_l38_3826

/-- Represents the number of units of each product produced by a single plate of each type -/
def PlateProduction := Fin 2 → Fin 2 → ℕ

/-- The required production amounts for products A and B -/
def RequiredProduction := Fin 2 → ℕ

/-- The solution represented as the number of plates of each type used -/
def Solution := Fin 2 → ℕ

/-- Checks if a solution satisfies the production requirements -/
def satisfiesRequirements (plate_prod : PlateProduction) (req_prod : RequiredProduction) (sol : Solution) : Prop :=
  ∀ i, (sol 0 * plate_prod 0 i + sol 1 * plate_prod 1 i) = req_prod i

/-- Calculates the total number of plates used in a solution -/
def totalPlates (sol : Solution) : ℕ :=
  sol 0 + sol 1

theorem minimize_plates (plate_prod : PlateProduction) (req_prod : RequiredProduction) :
  let solution : Solution := ![6, 2]
  satisfiesRequirements plate_prod req_prod solution ∧
  (∀ other : Solution, satisfiesRequirements plate_prod req_prod other →
    totalPlates solution ≤ totalPlates other) :=
by
  sorry

end minimize_plates_l38_3826


namespace inequality_solution_set_l38_3881

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x + 2 > 0) ↔ (0 ≤ m ∧ m < 8) :=
sorry

end inequality_solution_set_l38_3881


namespace lantern_tower_top_count_l38_3813

/-- Represents a tower with geometric progression of lanterns -/
structure LanternTower where
  levels : ℕ
  ratio : ℕ
  total : ℕ
  top : ℕ

/-- Calculates the sum of a geometric sequence -/
def geometricSum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

/-- Theorem: In a 7-level tower where the number of lanterns doubles at each level
    from top to bottom, and the total number of lanterns is 381,
    the number of lanterns at the top level is 3. -/
theorem lantern_tower_top_count (tower : LanternTower)
    (h1 : tower.levels = 7)
    (h2 : tower.ratio = 2)
    (h3 : tower.total = 381)
    : tower.top = 3 := by
  sorry

#check lantern_tower_top_count

end lantern_tower_top_count_l38_3813


namespace reeya_fourth_subject_score_l38_3849

theorem reeya_fourth_subject_score 
  (score1 score2 score3 : ℕ) 
  (average : ℚ) 
  (h1 : score1 = 55)
  (h2 : score2 = 67)
  (h3 : score3 = 76)
  (h4 : average = 67)
  (h5 : ∀ s : ℕ, s ≤ 100) -- Assuming all scores are out of 100
  : ∃ score4 : ℕ, 
    (score1 + score2 + score3 + score4 : ℚ) / 4 = average ∧ 
    score4 = 70 := by
  sorry

end reeya_fourth_subject_score_l38_3849


namespace max_even_distribution_l38_3865

/-- Represents the pencil distribution problem --/
def PencilDistribution (initial_pencils : ℕ) (initial_containers : ℕ) (first_addition : ℕ) (second_addition : ℕ) (final_containers : ℕ) : Prop :=
  let total_pencils : ℕ := initial_pencils + first_addition + second_addition
  let even_distribution : ℕ := total_pencils / final_containers
  even_distribution * final_containers ≤ total_pencils ∧
  (even_distribution + 1) * final_containers > total_pencils

/-- Theorem stating the maximum even distribution of pencils --/
theorem max_even_distribution :
  PencilDistribution 150 5 30 47 6 →
  ∃ (n : ℕ), n = 37 ∧ PencilDistribution 150 5 30 47 6 := by
  sorry

#check max_even_distribution

end max_even_distribution_l38_3865


namespace braking_distance_properties_l38_3860

/-- Represents the braking distance in meters -/
def braking_distance (v : ℝ) : ℝ := 0.25 * v

/-- The maximum legal speed on highways in km/h -/
def max_legal_speed : ℝ := 120

theorem braking_distance_properties :
  (braking_distance 60 = 15) ∧
  (braking_distance 128 = 32) ∧
  (128 > max_legal_speed) := by
  sorry

end braking_distance_properties_l38_3860


namespace exam_max_marks_l38_3824

theorem exam_max_marks (victor_score : ℝ) (victor_percentage : ℝ) (max_marks : ℝ) : 
  victor_score = 184 → 
  victor_percentage = 0.92 → 
  victor_score = victor_percentage * max_marks → 
  max_marks = 200 := by
sorry

end exam_max_marks_l38_3824


namespace total_spent_on_toys_l38_3877

-- Define the cost of toy cars
def toy_cars_cost : ℚ := 14.88

-- Define the cost of toy trucks
def toy_trucks_cost : ℚ := 5.86

-- Define the total cost of toys
def total_toys_cost : ℚ := toy_cars_cost + toy_trucks_cost

-- Theorem to prove
theorem total_spent_on_toys :
  total_toys_cost = 20.74 :=
by sorry

end total_spent_on_toys_l38_3877


namespace math_competition_average_score_l38_3817

theorem math_competition_average_score 
  (total_people : ℕ) 
  (group_average : ℚ) 
  (xiaoming_score : ℚ) 
  (h1 : total_people = 10)
  (h2 : group_average = 84)
  (h3 : xiaoming_score = 93) :
  let remaining_people := total_people - 1
  let total_score := group_average * total_people
  let remaining_score := total_score - xiaoming_score
  remaining_score / remaining_people = 83 := by
sorry

end math_competition_average_score_l38_3817


namespace original_number_is_45_l38_3844

theorem original_number_is_45 (x : ℝ) : x - 30 = x / 3 → x = 45 := by sorry

end original_number_is_45_l38_3844


namespace tan_beta_calculation_l38_3804

open Real

theorem tan_beta_calculation (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : sin α = 4/5) (h4 : tan (α - β) = 2/3) : tan β = 6/17 := by
  sorry

end tan_beta_calculation_l38_3804


namespace homothety_transforms_circles_l38_3863

/-- Two circles in a plane -/
structure TangentCircles where
  S₁ : Set (ℝ × ℝ)
  S₂ : Set (ℝ × ℝ)
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  r : ℝ
  R : ℝ
  K : ℝ × ℝ
  h_circle₁ : ∀ p ∈ S₁, dist p O₁ = r
  h_circle₂ : ∀ p ∈ S₂, dist p O₂ = R
  h_tangent : K ∈ S₁ ∧ K ∈ S₂
  h_external : dist O₁ O₂ = r + R

/-- Homothety transformation -/
def homothety (center : ℝ × ℝ) (k : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (center.1 + k * (p.1 - center.1), center.2 + k * (p.2 - center.2))

/-- Main theorem: Homothety transforms one circle into another -/
theorem homothety_transforms_circles (tc : TangentCircles) :
  ∃ h : Set (ℝ × ℝ) → Set (ℝ × ℝ),
    h tc.S₁ = tc.S₂ ∧
    ∀ p ∈ tc.S₁, h {p} = {homothety tc.K (tc.R / tc.r) p} :=
  sorry

end homothety_transforms_circles_l38_3863


namespace arithmetic_sequence_properties_l38_3829

/-- Given an arithmetic sequence {aₙ} with sum Sₙ of the first n terms,
    if S₆ > S₇ > S₅, then the common difference d < 0 and |a₆| > |a₇| -/
theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (d : ℝ)      -- The common difference
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)  -- Definition of Sₙ
  (h2 : ∀ n, a (n + 1) = a n + d)        -- Definition of arithmetic sequence
  (h3 : S 6 > S 7)                       -- Given condition
  (h4 : S 7 > S 5)                       -- Given condition
  : d < 0 ∧ |a 6| > |a 7| := by
  sorry

end arithmetic_sequence_properties_l38_3829


namespace no_nontrivial_integer_solutions_l38_3868

theorem no_nontrivial_integer_solutions :
  ∀ (x y z : ℤ), x^3 + 2*y^3 + 4*z^3 - 6*x*y*z = 0 → x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end no_nontrivial_integer_solutions_l38_3868


namespace cistern_fill_time_l38_3899

/-- Time to fill a cistern with two pipes -/
theorem cistern_fill_time 
  (fill_time_A : ℝ) 
  (empty_time_B : ℝ) 
  (h1 : fill_time_A = 16) 
  (h2 : empty_time_B = 20) : 
  (fill_time_A * empty_time_B) / (empty_time_B - fill_time_A) = 80 :=
by sorry

end cistern_fill_time_l38_3899


namespace direct_proportion_through_point_one_two_l38_3839

/-- The equation of a direct proportion function passing through (1, 2) -/
theorem direct_proportion_through_point_one_two :
  ∀ (k : ℝ), (∃ f : ℝ → ℝ, (∀ x, f x = k * x) ∧ f 1 = 2) → 
  (∀ x, k * x = 2 * x) :=
sorry

end direct_proportion_through_point_one_two_l38_3839


namespace investment_balance_l38_3808

/-- Proves that given an initial investment of 1800 at 7% interest, an additional investment of 1800 at 10% interest will result in a total annual income equal to 8.5% of the entire investment. -/
theorem investment_balance (initial_investment : ℝ) (additional_investment : ℝ) 
  (initial_rate : ℝ) (additional_rate : ℝ) (total_rate : ℝ) : 
  initial_investment = 1800 →
  additional_investment = 1800 →
  initial_rate = 0.07 →
  additional_rate = 0.10 →
  total_rate = 0.085 →
  initial_rate * initial_investment + additional_rate * additional_investment = 
    total_rate * (initial_investment + additional_investment) :=
by sorry

end investment_balance_l38_3808


namespace number_to_add_for_divisibility_l38_3885

theorem number_to_add_for_divisibility (a b : ℕ) (h : b > 0) : 
  ∃ n : ℕ, (a + n) % b = 0 ∧ n = if a % b = 0 then 0 else b - a % b :=
sorry

end number_to_add_for_divisibility_l38_3885


namespace tangent_line_equation_l38_3859

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y + 1 = 0

/-- A point on the line -/
def point : ℝ × ℝ := (-2, 5)

/-- Possible equations of the tangent line -/
def tangent_line_eq1 (x : ℝ) : Prop := x = -2
def tangent_line_eq2 (x y : ℝ) : Prop := 15*x + 8*y - 10 = 0

/-- The main theorem -/
theorem tangent_line_equation :
  ∃ (x y : ℝ), (x = point.1 ∧ y = point.2) ∧
  (∀ (x' y' : ℝ), circle_equation x' y' →
    (tangent_line_eq1 x ∨ tangent_line_eq2 x y) ∧
    (x = x' ∧ y = y' → ¬circle_equation x y)) :=
sorry

end tangent_line_equation_l38_3859


namespace no_real_roots_quadratic_l38_3855

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - x + k ≠ 0) → k > 1/4 := by
  sorry

end no_real_roots_quadratic_l38_3855


namespace prob_monochromatic_triangle_l38_3862

/-- A regular hexagon with colored edges -/
structure ColoredHexagon where
  /-- The probability of an edge being colored red -/
  p : ℝ
  /-- Assumption that p is between 0 and 1 -/
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The number of edges (sides and diagonals) in a regular hexagon -/
def num_edges : ℕ := 15

/-- The number of triangles in a regular hexagon -/
def num_triangles : ℕ := 20

/-- The probability of a specific triangle not being monochromatic -/
def prob_not_monochromatic (h : ColoredHexagon) : ℝ :=
  3 * h.p^2 * (1 - h.p) + 3 * (1 - h.p)^2 * h.p

/-- The main theorem: probability of at least one monochromatic triangle -/
theorem prob_monochromatic_triangle (h : ColoredHexagon) :
  h.p = 1/2 → 1 - (prob_not_monochromatic h)^num_triangles = 1 - (3/4)^20 := by
  sorry

end prob_monochromatic_triangle_l38_3862


namespace triangle_property_l38_3861

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  b = 5 →
  c = 7 →
  (a + c) / b = (Real.sin B + Real.sin A) / (Real.sin C - Real.sin A) →
  C = 2 * Real.pi / 3 ∧
  (1 / 2) * a * b * Real.sin C = 15 * Real.sqrt 3 / 4 :=
by sorry

end triangle_property_l38_3861


namespace triangle_property_l38_3852

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_property (t : Triangle) 
  (h1 : t.b * (1 + Real.cos t.C) = t.c * (2 - Real.cos t.B))
  (h2 : t.C = π / 3)
  (h3 : (1 / 2) * t.a * t.b * Real.sin t.C = 4 * Real.sqrt 3) :
  (t.a + t.b = 2 * t.c) ∧ (t.c = 4) := by
  sorry


end triangle_property_l38_3852


namespace power_sum_equality_l38_3888

theorem power_sum_equality : (-1 : ℤ) ^ 47 + 2 ^ (3^3 + 4^2 - 6^2) = 127 := by
  sorry

end power_sum_equality_l38_3888


namespace inverse_variation_problem_l38_3858

/-- Given that x varies inversely as the square of y, prove that x = 2.25 when y = 2,
    given that y = 3 when x = 1. -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) : 
  (∀ (x y : ℝ), x = k / (y^2)) →  -- x varies inversely as the square of y
  (1 = k / (3^2)) →               -- y = 3 when x = 1
  (2.25 = k / (2^2))              -- x = 2.25 when y = 2
  := by sorry

end inverse_variation_problem_l38_3858


namespace potato_difference_l38_3879

/-- The number of potato wedges -/
def x : ℕ := 8 * 13

/-- The number of potatoes used for french fries or potato chips -/
def k : ℕ := (67 - 13) / 2

/-- The number of potato chips -/
def z : ℕ := 20 * k

/-- The difference between the number of potato chips and potato wedges -/
def d : ℤ := z - x

theorem potato_difference : d = 436 := by
  sorry

end potato_difference_l38_3879


namespace jason_has_36_seashells_l38_3830

/-- The number of seashells Jason has now, given his initial count and the number he gave away. -/
def jasonsSeashells (initialCount gaveAway : ℕ) : ℕ :=
  initialCount - gaveAway

/-- Theorem stating that Jason has 36 seashells after giving some away. -/
theorem jason_has_36_seashells : jasonsSeashells 49 13 = 36 := by
  sorry

end jason_has_36_seashells_l38_3830


namespace visitors_not_enjoy_not_understand_l38_3848

-- Define the total number of visitors
def V : ℕ := 560

-- Define the number of visitors who enjoyed the painting
def E : ℕ := (3 * V) / 4

-- Define the number of visitors who understood the painting
def U : ℕ := E

-- Theorem to prove
theorem visitors_not_enjoy_not_understand : V - E = 140 := by
  sorry

end visitors_not_enjoy_not_understand_l38_3848


namespace equal_costs_at_twenty_l38_3815

/-- Represents the cost function for company A -/
def cost_A (x : ℝ) : ℝ := 450 * x + 1000

/-- Represents the cost function for company B -/
def cost_B (x : ℝ) : ℝ := 500 * x

/-- Theorem stating that the costs are equal when 20 desks are purchased -/
theorem equal_costs_at_twenty :
  ∃ (x : ℝ), x = 20 ∧ cost_A x = cost_B x :=
sorry

end equal_costs_at_twenty_l38_3815


namespace one_third_square_coloring_l38_3820

theorem one_third_square_coloring (n : ℕ) (k : ℕ) : n = 18 ∧ k = 6 → Nat.choose n k = 18564 := by
  sorry

end one_third_square_coloring_l38_3820


namespace flea_jump_angle_rational_l38_3883

/-- A flea jumping between two intersecting lines --/
structure FleaJump where
  α : ℝ  -- Angle between the lines in radians
  jumpLength : ℝ  -- Length of each jump
  returnsToStart : Prop  -- Flea eventually returns to starting point
  noPreviousPosition : Prop  -- Flea never returns to previous position

/-- Theorem stating that if a flea jumps as described, the angle is rational --/
theorem flea_jump_angle_rational (jump : FleaJump) 
  (h1 : jump.jumpLength = 1)
  (h2 : jump.returnsToStart)
  (h3 : jump.noPreviousPosition) :
  ∃ (p q : ℤ), jump.α = (p / q) * (π / 180) :=
sorry

end flea_jump_angle_rational_l38_3883


namespace thirtieth_triangular_number_properties_l38_3841

/-- Calculate the nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculate the number of dots in the perimeter of the nth triangular figure -/
def perimeter_dots (n : ℕ) : ℕ := n + 2 * (n - 1)

theorem thirtieth_triangular_number_properties :
  (triangular_number 30 = 465) ∧ (perimeter_dots 30 = 88) := by
  sorry

end thirtieth_triangular_number_properties_l38_3841


namespace symmetric_point_y_axis_l38_3875

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem symmetric_point_y_axis :
  let P : Point := { x := 2, y := 1 }
  let P' : Point := reflect_y_axis P
  P'.x = -2 ∧ P'.y = 1 := by sorry

end symmetric_point_y_axis_l38_3875


namespace quadratic_vertex_l38_3842

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + (8 - m)*x + 12

-- Define the derivative of the function
def f' (m : ℝ) (x : ℝ) : ℝ := -2*x + (8 - m)

-- Theorem statement
theorem quadratic_vertex (m : ℝ) :
  (∀ x > 2, (f' m x < 0)) ∧ 
  (∀ x < 2, (f' m x > 0)) →
  m = 4 := by
  sorry

end quadratic_vertex_l38_3842


namespace hash_three_two_l38_3807

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * (b + 1) + a * b + b^2

-- Theorem statement
theorem hash_three_two : hash 3 2 = 19 := by
  sorry

end hash_three_two_l38_3807


namespace paper_folding_thickness_l38_3816

def initial_thickness : ℝ := 0.1
def num_folds : ℕ := 4

theorem paper_folding_thickness :
  initial_thickness * (2 ^ num_folds) = 1.6 := by
  sorry

end paper_folding_thickness_l38_3816


namespace range_of_m_l38_3801

-- Define propositions P and Q
def P (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*m*x + m ≠ 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 ≥ 0

-- Define the condition that either P or Q is true, and both P and Q are false
def condition (m : ℝ) : Prop := 
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m)

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, condition m ↔ ((-2 ≤ m ∧ m ≤ 0) ∨ (1 ≤ m ∧ m ≤ 2)) :=
by sorry

end range_of_m_l38_3801


namespace mean_equality_implies_z_value_l38_3871

theorem mean_equality_implies_z_value :
  let mean1 := (5 + 8 + 17) / 3
  let mean2 := (15 + z) / 2
  mean1 = mean2 → z = 5 := by
sorry

end mean_equality_implies_z_value_l38_3871


namespace min_value_complex_expression_l38_3851

theorem min_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (min : ℝ), min = 0 ∧ ∀ w : ℂ, Complex.abs w = 2 → Complex.abs ((w - 2)^2 * (w + 2)) ≥ min :=
by sorry

end min_value_complex_expression_l38_3851


namespace inequality_system_solution_l38_3882

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 2 < x ∧ (1/3) * x < -2) ↔ x < -6 := by
  sorry

end inequality_system_solution_l38_3882


namespace pencil_count_original_pencils_count_l38_3893

/-- The number of pencils originally in the drawer -/
def original_pencils : ℕ := sorry

/-- The number of pencils Tim added to the drawer -/
def added_pencils : ℕ := 3

/-- The total number of pencils in the drawer after Tim added some -/
def total_pencils : ℕ := 5

/-- Theorem stating that the original number of pencils plus the added pencils equals the total pencils -/
theorem pencil_count : original_pencils + added_pencils = total_pencils := by sorry

/-- Theorem proving that the original number of pencils in the drawer was 2 -/
theorem original_pencils_count : original_pencils = 2 := by sorry

end pencil_count_original_pencils_count_l38_3893


namespace alex_friends_cookout_l38_3847

theorem alex_friends_cookout (burgers_per_guest : ℕ) (buns_per_pack : ℕ) (packs_of_buns : ℕ) 
  (h1 : burgers_per_guest = 3)
  (h2 : buns_per_pack = 8)
  (h3 : packs_of_buns = 3) :
  ∃ (friends : ℕ), friends = 9 ∧ 
    (packs_of_buns * buns_per_pack) / burgers_per_guest + 1 = friends :=
by
  sorry

end alex_friends_cookout_l38_3847


namespace circle_equation_m_range_l38_3833

/-- Given an equation x^2 + y^2 - 2x - 4y + m = 0 that represents a circle, prove that m < 5 -/
theorem circle_equation_m_range (m : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0 ↔ (x - 1)^2 + (y - 2)^2 = r^2) →
  m < 5 := by
sorry

end circle_equation_m_range_l38_3833


namespace min_value_of_function_l38_3866

theorem min_value_of_function :
  let f : ℝ → ℝ := λ x => 5/4 - Real.sin x^2 - 3 * Real.cos x
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = -7/4 := by
  sorry

end min_value_of_function_l38_3866


namespace constant_pace_run_time_l38_3884

/-- Represents the time taken to run a certain distance at a constant pace -/
structure RunTime where
  distance : ℝ
  time : ℝ

/-- Calculates the time taken to run a given distance at a constant pace -/
def calculateTime (pace : ℝ) (distance : ℝ) : ℝ :=
  pace * distance

theorem constant_pace_run_time 
  (store_run : RunTime) 
  (friend_house_distance : ℝ) 
  (h1 : store_run.distance = 3) 
  (h2 : store_run.time = 24) 
  (h3 : friend_house_distance = 1.5) :
  calculateTime (store_run.time / store_run.distance) friend_house_distance = 12 := by
  sorry

#check constant_pace_run_time

end constant_pace_run_time_l38_3884


namespace no_multiple_of_five_2C4_l38_3846

theorem no_multiple_of_five_2C4 : 
  ¬ ∃ (C : ℕ), 
    (100 ≤ 200 + 10 * C + 4) ∧ 
    (200 + 10 * C + 4 < 1000) ∧ 
    (C < 10) ∧ 
    ((200 + 10 * C + 4) % 5 = 0) :=
by sorry

end no_multiple_of_five_2C4_l38_3846


namespace john_driving_equation_l38_3822

def speed_before_lunch : ℝ := 60
def speed_after_lunch : ℝ := 90
def total_distance : ℝ := 300
def total_time : ℝ := 4
def lunch_break : ℝ := 0.5

theorem john_driving_equation (t : ℝ) : 
  speed_before_lunch * t + speed_after_lunch * (total_time - lunch_break - t) = total_distance :=
sorry

end john_driving_equation_l38_3822


namespace co_captains_probability_l38_3856

def team_sizes : List Nat := [6, 8, 9, 10]
def num_teams : Nat := 4
def co_captains_per_team : Nat := 3

def probability_co_captains (n : Nat) : Rat :=
  6 / (n * (n - 1) * (n - 2))

theorem co_captains_probability : 
  (1 / num_teams) * (team_sizes.map probability_co_captains).sum = 37 / 1680 := by
  sorry

end co_captains_probability_l38_3856


namespace quadratic_factorization_l38_3827

theorem quadratic_factorization (a : ℝ) : 
  (∃ m n : ℝ, ∀ x y : ℝ, 
    x^2 + 7*x*y + a*y^2 - 5*x - 45*y - 24 = (x - 8 + m*y) * (x + 3 + n*y)) → 
  a = 6 := by
sorry

end quadratic_factorization_l38_3827


namespace unique_solution_for_equation_l38_3835

theorem unique_solution_for_equation :
  ∃! (x y : ℝ), (x - 10)^2 + (y - 11)^2 + (x - y)^2 = 1/3 ∧ 
  x = 10 + 1/3 ∧ y = 10 + 2/3 := by
  sorry

end unique_solution_for_equation_l38_3835


namespace least_number_of_cubes_l38_3810

def block_length : ℕ := 15
def block_width : ℕ := 30
def block_height : ℕ := 75

theorem least_number_of_cubes :
  let gcd := Nat.gcd (Nat.gcd block_length block_width) block_height
  let cube_side := gcd
  let num_cubes := (block_length * block_width * block_height) / (cube_side * cube_side * cube_side)
  num_cubes = 10 := by sorry

end least_number_of_cubes_l38_3810
