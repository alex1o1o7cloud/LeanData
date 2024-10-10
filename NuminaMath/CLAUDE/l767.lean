import Mathlib

namespace most_colored_pencils_l767_76775

theorem most_colored_pencils (total : ℕ) (red blue yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow :=
by sorry

end most_colored_pencils_l767_76775


namespace B_power_100_l767_76753

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_100 : B ^ 100 = B := by sorry

end B_power_100_l767_76753


namespace right_triangle_coordinate_l767_76786

/-- Given a right triangle ABC with vertices A(0, 0), B(0, 4a - 2), and C(x, 4a - 2),
    if the area of the triangle is 63, then the x-coordinate of point C is 126 / (4a - 2). -/
theorem right_triangle_coordinate (a : ℝ) (x : ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 4 * a - 2)
  let C : ℝ × ℝ := (x, 4 * a - 2)
  (4 * a - 2 ≠ 0) →
  (1 / 2 : ℝ) * x * (4 * a - 2) = 63 →
  x = 126 / (4 * a - 2) :=
by sorry

end right_triangle_coordinate_l767_76786


namespace total_celestial_bodies_l767_76797

-- Define the number of planets
def num_planets : ℕ := 20

-- Define the ratio of solar systems to planets
def solar_system_ratio : ℕ := 8

-- Theorem: The total number of solar systems and planets is 180
theorem total_celestial_bodies : 
  num_planets * (solar_system_ratio + 1) = 180 := by
  sorry

end total_celestial_bodies_l767_76797


namespace line_segment_endpoint_l767_76771

/-- Given a line segment in the Cartesian plane with midpoint (2020, 11), 
    one endpoint at (a, 0), and the other endpoint on the line y = x, 
    prove that a = 4018 -/
theorem line_segment_endpoint (a : ℝ) : 
  (∃ t : ℝ, (a + t) / 2 = 2020 ∧ t / 2 = 11 ∧ t = t) → a = 4018 := by
  sorry

end line_segment_endpoint_l767_76771


namespace weight_of_new_person_l767_76784

/-- Theorem: Weight of the new person in a group replacement scenario -/
theorem weight_of_new_person
  (n : ℕ) -- Number of people in the group
  (w : ℝ) -- Total weight of the original group
  (r : ℝ) -- Weight of the person being replaced
  (i : ℝ) -- Increase in average weight
  (h1 : n = 15) -- There are 15 people initially
  (h2 : r = 75) -- The replaced person weighs 75 kg
  (h3 : i = 3.2) -- The average weight increases by 3.2 kg
  (h4 : (w - r + (w / n + n * i)) / n = w / n + i) -- Equation for the new average weight
  : w / n + n * i - r = 123 := by
  sorry

#check weight_of_new_person

end weight_of_new_person_l767_76784


namespace min_value_x_plus_y_l767_76747

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 19 / x + 98 / y = 1) : 
  x + y ≥ 117 + 14 * Real.sqrt 38 := by
  sorry

end min_value_x_plus_y_l767_76747


namespace total_towels_calculation_l767_76768

/-- The number of loads of laundry washed -/
def loads : ℕ := 6

/-- The number of towels in each load -/
def towels_per_load : ℕ := 7

/-- The total number of towels washed -/
def total_towels : ℕ := loads * towels_per_load

theorem total_towels_calculation : total_towels = 42 := by
  sorry

end total_towels_calculation_l767_76768


namespace solve_equation_l767_76755

theorem solve_equation (x : ℝ) : 3 * x + 15 = (1 / 3) * (6 * x + 45) → x = 0 := by
  sorry

end solve_equation_l767_76755


namespace ten_lines_intersection_points_l767_76743

/-- The number of intersection points of n lines in a plane, where no lines are parallel
    and exactly two lines pass through each intersection point. -/
def intersection_points (n : ℕ) : ℕ :=
  Nat.choose n 2

/-- Given 10 lines in a plane where no lines are parallel and exactly two lines pass through
    each intersection point, the number of intersection points is 45. -/
theorem ten_lines_intersection_points :
  intersection_points 10 = 45 := by
  sorry

end ten_lines_intersection_points_l767_76743


namespace fraction_inequality_l767_76770

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  a / (a + c) > b / (b + c) := by
  sorry

end fraction_inequality_l767_76770


namespace sunday_to_weekday_ratio_is_correct_l767_76731

/-- The weight ratio of Sunday papers to Monday-Saturday papers --/
def sunday_to_weekday_ratio : ℚ :=
  let weekday_paper_weight : ℚ := 8  -- ounces
  let papers_per_day : ℕ := 250
  let weeks : ℕ := 10
  let weekdays_per_week : ℕ := 6
  let recycling_rate : ℚ := 100 / 2000  -- $/pound

  let total_weekday_papers : ℕ := papers_per_day * weekdays_per_week * weeks
  let total_weekday_weight : ℚ := weekday_paper_weight * total_weekday_papers
  
  let total_sunday_papers : ℕ := papers_per_day * weeks
  let total_sunday_weight : ℚ := 2000 * 16  -- 1 ton in ounces
  
  let sunday_paper_weight : ℚ := total_sunday_weight / total_sunday_papers
  
  sunday_paper_weight / weekday_paper_weight

theorem sunday_to_weekday_ratio_is_correct : sunday_to_weekday_ratio = 8/5 := by
  sorry

end sunday_to_weekday_ratio_is_correct_l767_76731


namespace bills_equal_at_100_minutes_l767_76723

/-- United Telephone's base rate in dollars -/
def united_base : ℚ := 7

/-- United Telephone's per-minute charge in dollars -/
def united_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℚ := 12

/-- Atlantic Call's per-minute charge in dollars -/
def atlantic_per_minute : ℚ := 1/5

/-- The number of minutes at which the bills are equal -/
def equal_minutes : ℚ := 100

theorem bills_equal_at_100_minutes :
  united_base + united_per_minute * equal_minutes =
  atlantic_base + atlantic_per_minute * equal_minutes :=
sorry

end bills_equal_at_100_minutes_l767_76723


namespace factor_expression_l767_76741

theorem factor_expression (y : ℝ) : 3*y*(y-4) + 8*(y-4) - 2*(y-4) = 3*(y+2)*(y-4) := by
  sorry

end factor_expression_l767_76741


namespace number_at_2002_2003_l767_76788

/-- Represents the number at position (row, col) in the arrangement -/
def number_at_position (row : ℕ) (col : ℕ) : ℕ :=
  (col - 1)^2 + 1 + (row - 1)

/-- The theorem to be proved -/
theorem number_at_2002_2003 :
  number_at_position 2002 2003 = 2002 * 2003 := by
  sorry

#check number_at_2002_2003

end number_at_2002_2003_l767_76788


namespace beaus_age_proof_l767_76716

/-- Represents Beau's age today -/
def beaus_age_today : ℕ := 42

/-- Represents the age of Beau's sons today -/
def sons_age_today : ℕ := 16

/-- The number of Beau's sons (triplets) -/
def number_of_sons : ℕ := 3

/-- The number of years ago when the sum of sons' ages equaled Beau's age -/
def years_ago : ℕ := 3

theorem beaus_age_proof :
  (sons_age_today - years_ago) * number_of_sons + years_ago = beaus_age_today :=
by sorry

end beaus_age_proof_l767_76716


namespace inscribed_circle_radius_l767_76766

/-- An isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  /-- The length of the equal sides of the isosceles triangle -/
  a : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The area of the triangle -/
  area : ℝ
  /-- The ratio of AN to AB, where N is the point where a line parallel to BC 
      and tangent to the inscribed circle intersects AB -/
  an_ratio : ℝ
  /-- Condition that the triangle is isosceles -/
  isosceles : a > 0
  /-- Condition that AN = 3/8 * AB -/
  an_condition : an_ratio = 3/8
  /-- Condition that the area of the triangle is 12 -/
  area_condition : area = 12

/-- Theorem: If the conditions are met, the radius of the inscribed circle is 3/2 -/
theorem inscribed_circle_radius 
  (t : IsoscelesTriangleWithInscribedCircle) : t.r = 3/2 := by
  sorry

end inscribed_circle_radius_l767_76766


namespace quadratic_minimum_quadratic_achieves_minimum_l767_76776

theorem quadratic_minimum (x : ℝ) (h : x > 0) : x^2 - 2*x + 3 ≥ 2 := by
  sorry

theorem quadratic_achieves_minimum : ∃ (x : ℝ), x > 0 ∧ x^2 - 2*x + 3 = 2 := by
  sorry

end quadratic_minimum_quadratic_achieves_minimum_l767_76776


namespace greatest_three_digit_multiple_of_17_l767_76751

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 986 ∧ 
  n % 17 = 0 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  ∀ m : ℕ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
sorry

end greatest_three_digit_multiple_of_17_l767_76751


namespace sqrt_64_equals_8_l767_76752

theorem sqrt_64_equals_8 : Real.sqrt 64 = 8 := by
  sorry

end sqrt_64_equals_8_l767_76752


namespace factorization_xy_squared_l767_76765

theorem factorization_xy_squared (x y : ℝ) : x^2*y + x*y^2 = x*y*(x + y) := by
  sorry

end factorization_xy_squared_l767_76765


namespace french_toast_weekends_l767_76754

/-- Represents the number of slices used per weekend -/
def slices_per_weekend : ℚ := 3

/-- Represents the number of slices in a loaf of bread -/
def slices_per_loaf : ℕ := 12

/-- Represents the number of loaves of bread used -/
def loaves_used : ℕ := 26

/-- Theorem stating that 26 loaves of bread cover 104 weekends of french toast making -/
theorem french_toast_weekends : 
  (loaves_used : ℚ) * (slices_per_loaf : ℚ) / slices_per_weekend = 104 := by
  sorry

end french_toast_weekends_l767_76754


namespace brittany_age_is_32_l767_76707

/-- Brittany's age when she returns from vacation -/
def brittany_age_after_vacation (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ) : ℕ :=
  rebecca_age + age_difference + vacation_duration

/-- Theorem: Brittany's age when she returns from vacation is 32 -/
theorem brittany_age_is_32 : brittany_age_after_vacation 25 3 4 = 32 := by
  sorry

end brittany_age_is_32_l767_76707


namespace max_pairs_proof_max_pairs_achievable_l767_76720

/-- The maximum number of pairs that can be chosen from the set {1, 2, ..., 2017}
    such that a_i < b_i, no two pairs share a common element, and all sums a_i + b_i
    are distinct and less than or equal to 2017. -/
def max_pairs : ℕ := 806

theorem max_pairs_proof :
  ∀ (k : ℕ) (a b : Fin k → ℕ),
  (∀ i : Fin k, a i < b i) →
  (∀ i : Fin k, b i ≤ 2017) →
  (∀ i j : Fin k, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j) →
  (∀ i j : Fin k, i ≠ j → a i + b i ≠ a j + b j) →
  (∀ i : Fin k, a i + b i ≤ 2017) →
  k ≤ max_pairs :=
by sorry

theorem max_pairs_achievable :
  ∃ (k : ℕ) (a b : Fin k → ℕ),
  k = max_pairs ∧
  (∀ i : Fin k, a i < b i) ∧
  (∀ i : Fin k, b i ≤ 2017) ∧
  (∀ i j : Fin k, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j) ∧
  (∀ i j : Fin k, i ≠ j → a i + b i ≠ a j + b j) ∧
  (∀ i : Fin k, a i + b i ≤ 2017) :=
by sorry

end max_pairs_proof_max_pairs_achievable_l767_76720


namespace sum_of_fraction_parts_of_2_52_l767_76750

def decimal_to_fraction (d : ℚ) : ℤ × ℤ :=
  let n := d.num
  let d := d.den
  let g := n.gcd d
  (n / g, d / g)

theorem sum_of_fraction_parts_of_2_52 :
  let (n, d) := decimal_to_fraction (252 / 100)
  n + d = 88 := by sorry

end sum_of_fraction_parts_of_2_52_l767_76750


namespace factory_solution_l767_76762

def factory_problem (total_employees : ℕ) : Prop :=
  ∃ (employees_17 : ℕ),
    -- 200 employees earn $12/hour
    -- 40 employees earn $14/hour
    -- The rest earn $17/hour
    total_employees = 200 + 40 + employees_17 ∧
    -- The cost for one 8-hour shift is $31840
    31840 = (200 * 12 + 40 * 14 + employees_17 * 17) * 8

theorem factory_solution : ∃ (total_employees : ℕ), factory_problem total_employees ∧ total_employees = 300 := by
  sorry

end factory_solution_l767_76762


namespace sphere_volume_circumscribing_cube_l767_76727

/-- The volume of a sphere circumscribing a cube with edge length 2 is 4√3π -/
theorem sphere_volume_circumscribing_cube (cube_edge : ℝ) (sphere_volume : ℝ) : 
  cube_edge = 2 →
  sphere_volume = (4 / 3) * Real.pi * (Real.sqrt 3) ^ 3 →
  sphere_volume = 4 * Real.sqrt 3 * Real.pi :=
by sorry

end sphere_volume_circumscribing_cube_l767_76727


namespace contrapositive_equivalence_l767_76795

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔
  (∀ x : ℝ, x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) :=
by sorry

end contrapositive_equivalence_l767_76795


namespace dress_pocket_ratio_l767_76749

/-- Proves that the ratio of dresses with 2 pockets to the total number of dresses with pockets is 1:3 --/
theorem dress_pocket_ratio :
  ∀ (x y : ℕ),
  -- Total number of dresses
  24 = x + y + (24 / 2) →
  -- Total number of pockets
  2 * x + 3 * y = 32 →
  -- Ratio of dresses with 2 pockets to total dresses with pockets
  x / (x + y) = 1 / 3 :=
by
  sorry

end dress_pocket_ratio_l767_76749


namespace fraction_sum_equality_l767_76710

theorem fraction_sum_equality (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 =
  1 / (b - c) + 1 / (c - a) + 1 / (a - b) := by
  sorry

end fraction_sum_equality_l767_76710


namespace equation_proof_l767_76730

theorem equation_proof : Real.sqrt ((5568 / 87) ^ (1/3) + Real.sqrt (72 * 2)) = 4 := by
  sorry

end equation_proof_l767_76730


namespace complex_fraction_simplification_l767_76778

theorem complex_fraction_simplification (N : ℕ) (h : N = 2^16) :
  (65533^3 + 65534^3 + 65535^3 + 65536^3 + 65537^3 + 65538^3 + 65539^3) / 
  (32765 * 32766 + 32767 * 32768 + 32768 * 32769 + 32770 * 32771 : ℕ) = 7 * N :=
by sorry

end complex_fraction_simplification_l767_76778


namespace probability_of_black_piece_l767_76779

/-- Given a set of items with two types, this function calculates the probability of selecting an item of a specific type. -/
def probability_of_selection (total : ℕ) (type_a : ℕ) : ℚ :=
  type_a / total

/-- The probability of selecting a black piece from a set of Go pieces -/
theorem probability_of_black_piece : probability_of_selection 7 4 = 4 / 7 := by
  sorry

#eval probability_of_selection 7 4

end probability_of_black_piece_l767_76779


namespace cube_sum_from_sum_and_square_sum_l767_76733

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
sorry

end cube_sum_from_sum_and_square_sum_l767_76733


namespace book_reading_percentage_l767_76774

theorem book_reading_percentage (total_pages : ℕ) (second_night_percent : ℝ) 
  (third_night_percent : ℝ) (pages_left : ℕ) : ℝ :=
by
  have h1 : total_pages = 500 := by sorry
  have h2 : second_night_percent = 20 := by sorry
  have h3 : third_night_percent = 30 := by sorry
  have h4 : pages_left = 150 := by sorry
  
  -- Define the first night percentage
  let first_night_percent : ℝ := 20

  -- Prove that the first night percentage is correct
  have h5 : first_night_percent / 100 * total_pages + 
            second_night_percent / 100 * total_pages + 
            third_night_percent / 100 * total_pages = 
            total_pages - pages_left := by sorry

  exact first_night_percent

end book_reading_percentage_l767_76774


namespace A_equals_B_l767_76764

def A (a : ℕ) : Set ℕ :=
  {k : ℕ | ∃ x y : ℤ, x > Real.sqrt a ∧ k = (x^2 - a) / (x^2 - y^2)}

def B (a : ℕ) : Set ℕ :=
  {k : ℕ | ∃ x y : ℤ, 0 ≤ x ∧ x < Real.sqrt a ∧ k = (x^2 - a) / (x^2 - y^2)}

theorem A_equals_B (a : ℕ) (h : ¬ ∃ n : ℕ, n^2 = a) : A a = B a := by
  sorry

end A_equals_B_l767_76764


namespace concession_stand_sales_l767_76756

/-- Calculates the total number of items sold given the prices, total revenue, and number of hot dogs sold. -/
theorem concession_stand_sales
  (hot_dog_price : ℚ)
  (soda_price : ℚ)
  (total_revenue : ℚ)
  (hot_dogs_sold : ℕ)
  (h1 : hot_dog_price = 3/2)
  (h2 : soda_price = 1/2)
  (h3 : total_revenue = 157/2)
  (h4 : hot_dogs_sold = 35) :
  ∃ (sodas_sold : ℕ), hot_dogs_sold + sodas_sold = 87 :=
by sorry

end concession_stand_sales_l767_76756


namespace project_hours_difference_l767_76700

theorem project_hours_difference (total_pay : ℝ) (wage_p wage_q : ℝ) :
  total_pay = 420 ∧ 
  wage_p = wage_q * 1.5 ∧ 
  wage_p = wage_q + 7 →
  (total_pay / wage_q) - (total_pay / wage_p) = 10 := by
sorry

end project_hours_difference_l767_76700


namespace consecutive_even_numbers_sum_l767_76719

theorem consecutive_even_numbers_sum (n : ℕ) (sum : ℕ) (start : ℕ) : 
  (sum = (n / 2) * (2 * start + (n - 1) * 2)) →
  (start = 32) →
  (sum = 140) →
  (n = 4) :=
by
  sorry

end consecutive_even_numbers_sum_l767_76719


namespace notebooks_per_student_in_second_half_l767_76712

/-- Given a classroom with students and notebooks, prove that each student
    in the second half has 3 notebooks. -/
theorem notebooks_per_student_in_second_half
  (total_students : ℕ)
  (total_notebooks : ℕ)
  (notebooks_per_first_half_student : ℕ)
  (h1 : total_students = 28)
  (h2 : total_notebooks = 112)
  (h3 : notebooks_per_first_half_student = 5)
  (h4 : 2 ∣ total_students) :
  (total_notebooks - (total_students / 2 * notebooks_per_first_half_student)) / (total_students / 2) = 3 :=
by sorry

end notebooks_per_student_in_second_half_l767_76712


namespace final_ratio_is_16_to_9_l767_76745

/-- Represents the contents of a bin with peanuts and raisins -/
structure BinContents where
  peanuts : ℚ
  raisins : ℚ

/-- Removes an amount from the bin proportionally -/
def removeProportionally (bin : BinContents) (amount : ℚ) : BinContents :=
  let total := bin.peanuts + bin.raisins
  let peanutsProportion := bin.peanuts / total
  let raisinsProportion := bin.raisins / total
  { peanuts := bin.peanuts - (peanutsProportion * amount)
  , raisins := bin.raisins - (raisinsProportion * amount) }

/-- Adds an amount of raisins to the bin -/
def addRaisins (bin : BinContents) (amount : ℚ) : BinContents :=
  { peanuts := bin.peanuts, raisins := bin.raisins + amount }

/-- Theorem stating the final ratio of peanuts to raisins -/
theorem final_ratio_is_16_to_9 :
  let initial_bin : BinContents := { peanuts := 10, raisins := 0 }
  let after_first_operation := addRaisins { peanuts := initial_bin.peanuts - 2, raisins := 0 } 2
  let after_second_operation := addRaisins (removeProportionally after_first_operation 2) 2
  (after_second_operation.peanuts * 9 = after_second_operation.raisins * 16) := by
  sorry

end final_ratio_is_16_to_9_l767_76745


namespace school_year_days_is_180_l767_76792

/-- The number of days in a school year. -/
def school_year_days : ℕ := 180

/-- The maximum percentage of days that can be missed without taking exams. -/
def max_missed_percentage : ℚ := 5 / 100

/-- The number of days Hazel has already missed. -/
def days_already_missed : ℕ := 6

/-- The additional number of days Hazel can miss without taking exams. -/
def additional_days_can_miss : ℕ := 3

/-- Theorem stating that the number of days in the school year is 180. -/
theorem school_year_days_is_180 :
  (days_already_missed + additional_days_can_miss : ℚ) / school_year_days = max_missed_percentage :=
by sorry

end school_year_days_is_180_l767_76792


namespace sqrt_equation_solution_l767_76780

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 3) = 7 → x = 46 := by
  sorry

end sqrt_equation_solution_l767_76780


namespace part_one_simplification_part_two_simplification_l767_76761

-- Part 1
theorem part_one_simplification :
  (1 / 2)⁻¹ - (Real.sqrt 2019 - 1)^0 = 1 := by sorry

-- Part 2
theorem part_two_simplification (x y : ℝ) :
  (x - y)^2 - (x + 2*y) * (x - 2*y) = -2*x*y + 5*y^2 := by sorry

end part_one_simplification_part_two_simplification_l767_76761


namespace dogwood_trees_planted_l767_76773

/-- The number of dogwood trees planted in a park --/
theorem dogwood_trees_planted (initial_trees final_trees : ℕ) 
  (h1 : initial_trees = 34)
  (h2 : final_trees = 83) :
  final_trees - initial_trees = 49 := by
  sorry

end dogwood_trees_planted_l767_76773


namespace vending_machine_probability_l767_76758

def num_toys : ℕ := 10
def min_cost : ℚ := 1/2
def max_cost : ℚ := 5
def cost_difference : ℚ := 1/2
def initial_half_dollars : ℕ := 10
def favorite_toy_cost : ℚ := 9/2

theorem vending_machine_probability :
  let total_permutations : ℕ := num_toys.factorial
  let favorable_outcomes : ℕ := (num_toys - 1).factorial + (num_toys - 2).factorial
  (1 : ℚ) - (favorable_outcomes : ℚ) / (total_permutations : ℚ) = 8/9 := by
  sorry

end vending_machine_probability_l767_76758


namespace hyperbola_proof_l767_76701

/-- Given hyperbola -/
def given_hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- Hyperbola to prove -/
def target_hyperbola (x y : ℝ) : Prop := x^2/3 - y^2/12 = 1

/-- Point that the target hyperbola passes through -/
def point : ℝ × ℝ := (2, 2)

theorem hyperbola_proof :
  (∀ x y : ℝ, given_hyperbola x y ↔ ∃ k : ℝ, x^2 - y^2/4 = k) ∧
  target_hyperbola point.1 point.2 ∧
  (∀ x y : ℝ, given_hyperbola x y ↔ target_hyperbola x y) :=
sorry

end hyperbola_proof_l767_76701


namespace interest_rate_problem_l767_76706

/-- Calculates the simple interest rate given the principal, time, and interest amount. -/
def calculate_interest_rate (principal : ℕ) (time : ℕ) (interest : ℕ) : ℚ :=
  (interest : ℚ) * 100 / ((principal : ℚ) * (time : ℚ))

theorem interest_rate_problem (principal time interest_difference : ℕ) 
  (h1 : principal = 3000)
  (h2 : time = 5)
  (h3 : interest_difference = 2400)
  (h4 : principal - interest_difference > 0) :
  calculate_interest_rate principal time (principal - interest_difference) = 4 := by
  sorry

#eval calculate_interest_rate 3000 5 600

end interest_rate_problem_l767_76706


namespace intersection_characterization_l767_76783

-- Define sets A and B
def A : Set ℝ := {x | x^2 - x + 1 ≥ 0}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

-- Define the intersection of A and B
def A_inter_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_characterization : 
  A_inter_B = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

end intersection_characterization_l767_76783


namespace log_equation_solution_l767_76782

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 4 + 2 * (Real.log x / Real.log 8) = 7 → x = 64 := by
  sorry

end log_equation_solution_l767_76782


namespace point_P_properties_l767_76713

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (-3*a - 4, 2 + a)

-- Define the point Q
def Q : ℝ × ℝ := (5, 8)

theorem point_P_properties (a : ℝ) :
  -- Case 1: P lies on x-axis
  (P a).1 = 2 ∧ (P a).2 = 0 → a = -2
  ∧
  -- Case 2: PQ is parallel to y-axis
  (P a).1 = Q.1 → a = -3
  ∧
  -- Case 3: P is in second quadrant and equidistant from axes
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).1| = |(P a).2| → 
    a = -1 ∧ (-1 : ℝ)^2023 + 2023 = 2022 :=
by sorry

end point_P_properties_l767_76713


namespace petrol_expenses_l767_76702

def monthly_salary : ℕ := 23000
def savings_percentage : ℚ := 1/10
def savings : ℕ := 2300
def known_expenses : ℕ := 18700

theorem petrol_expenses : 
  monthly_salary * savings_percentage = savings →
  monthly_salary - savings - known_expenses = 2000 := by
sorry

end petrol_expenses_l767_76702


namespace solve_linear_equation_l767_76793

theorem solve_linear_equation :
  ∃ x : ℝ, x + 1 = 3 ∧ x = 2 := by
sorry

end solve_linear_equation_l767_76793


namespace smallest_number_with_remainders_l767_76744

theorem smallest_number_with_remainders : ∃ (n : ℕ), n > 0 ∧ 
  n % 2 = 1 ∧ n % 3 = 2 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 → n ≤ m :=
by sorry

end smallest_number_with_remainders_l767_76744


namespace sum_of_digits_3125_base6_l767_76705

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem: The sum of digits of 3125 in base 6 equals 15 -/
theorem sum_of_digits_3125_base6 : sumDigits (toBase6 3125) = 15 := by
  sorry

end sum_of_digits_3125_base6_l767_76705


namespace sqrt_equation_solution_l767_76790

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 12) = 10 → x = 88 := by
  sorry

end sqrt_equation_solution_l767_76790


namespace exam_students_count_l767_76725

theorem exam_students_count :
  ∀ (n : ℕ) (T : ℝ),
    n > 0 →
    T = n * 90 →
    T - 120 = (n - 3) * 95 →
    n = 33 :=
by
  sorry

end exam_students_count_l767_76725


namespace number_equation_l767_76718

theorem number_equation (x : ℚ) (N : ℚ) : x = 9 → (N - 5 / x = 4 + 4 / x ↔ N = 5) := by
  sorry

end number_equation_l767_76718


namespace expression_simplification_l767_76787

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 2) (h2 : x ≠ 4) (h3 : x ≠ 3) (h4 : x ≠ 5) :
  (x^2 - 2*x + 1) / (x^2 - 6*x + 8) / ((x^2 - 4*x + 3) / (x^2 - 8*x + 15)) = (x - 5) / (x - 2) := by
  sorry

end expression_simplification_l767_76787


namespace transportation_theorem_l767_76777

/-- Represents the capacity and cost of a truck type -/
structure TruckType where
  capacity : ℕ
  cost : ℕ

/-- Represents a transportation plan -/
structure TransportPlan where
  typeA : ℕ
  typeB : ℕ

/-- Solves the transportation problem -/
def solve_transportation_problem (typeA typeB : TruckType) (total_goods : ℕ) : 
  (TruckType × TruckType × TransportPlan) := sorry

theorem transportation_theorem 
  (typeA typeB : TruckType) (total_goods : ℕ) 
  (h1 : 3 * typeA.capacity + 2 * typeB.capacity = 90)
  (h2 : 5 * typeA.capacity + 4 * typeB.capacity = 160)
  (h3 : typeA.cost = 500)
  (h4 : typeB.cost = 400)
  (h5 : total_goods = 190) :
  let (solvedA, solvedB, optimal_plan) := solve_transportation_problem typeA typeB total_goods
  solvedA.capacity = 20 ∧ 
  solvedB.capacity = 15 ∧ 
  optimal_plan.typeA = 8 ∧ 
  optimal_plan.typeB = 2 := by sorry

end transportation_theorem_l767_76777


namespace variance_of_X_l767_76757

/-- The probability of Person A hitting the target -/
def prob_A : ℚ := 2/3

/-- The probability of Person B hitting the target -/
def prob_B : ℚ := 4/5

/-- The random variable X representing the number of people hitting the target -/
def X : ℕ → ℚ
| 0 => (1 - prob_A) * (1 - prob_B)
| 1 => prob_A * (1 - prob_B) + (1 - prob_A) * prob_B
| 2 => prob_A * prob_B
| _ => 0

/-- The expected value of X -/
def E_X : ℚ := 0 * X 0 + 1 * X 1 + 2 * X 2

/-- The variance of X -/
def Var_X : ℚ := (0 - E_X)^2 * X 0 + (1 - E_X)^2 * X 1 + (2 - E_X)^2 * X 2

theorem variance_of_X : Var_X = 86/225 := by sorry

end variance_of_X_l767_76757


namespace cos_alpha_plus_7pi_12_l767_76748

theorem cos_alpha_plus_7pi_12 (α : ℝ) (h : Real.sin (α + π/12) = 1/3) :
  Real.cos (α + 7*π/12) = -(1 + Real.sqrt 24) / 6 := by
  sorry

end cos_alpha_plus_7pi_12_l767_76748


namespace man_son_age_ratio_l767_76763

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1 -/
theorem man_son_age_ratio :
  ∀ (man_age son_age : ℕ),
    man_age = son_age + 32 →
    son_age = 30 →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end man_son_age_ratio_l767_76763


namespace half_power_inequality_l767_76746

theorem half_power_inequality (m n : ℝ) (h : m > n) : (1/2 : ℝ)^m < (1/2 : ℝ)^n := by
  sorry

end half_power_inequality_l767_76746


namespace min_value_geometric_sequence_l767_76738

/-- Given a geometric sequence with first term a₁ = 2, 
    the minimum value of 3a₂ + 6a₃ is -3/2. -/
theorem min_value_geometric_sequence (r : ℝ) : 
  let a₁ : ℝ := 2
  let a₂ : ℝ := a₁ * r
  let a₃ : ℝ := a₂ * r
  3 * a₂ + 6 * a₃ ≥ -3/2 :=
sorry

end min_value_geometric_sequence_l767_76738


namespace fifteenth_student_age_l767_76740

theorem fifteenth_student_age (total_students : Nat) (avg_age : Nat) (group1_size : Nat) (group1_avg : Nat) (group2_size : Nat) (group2_avg : Nat) :
  total_students = 15 →
  avg_age = 15 →
  group1_size = 3 →
  group1_avg = 14 →
  group2_size = 11 →
  group2_avg = 16 →
  (total_students * avg_age) - (group1_size * group1_avg + group2_size * group2_avg) = 7 := by
  sorry

end fifteenth_student_age_l767_76740


namespace smallest_cube_box_volume_l767_76704

def cone_height : ℝ := 20
def cone_base_diameter : ℝ := 18

theorem smallest_cube_box_volume (h : cone_height ≥ cone_base_diameter) :
  let box_side := max cone_height cone_base_diameter
  box_side ^ 3 = 8000 := by sorry

end smallest_cube_box_volume_l767_76704


namespace reciprocal_equals_self_l767_76759

theorem reciprocal_equals_self (x : ℝ) : x ≠ 0 → (x = 1/x ↔ x = 1 ∨ x = -1) := by sorry

end reciprocal_equals_self_l767_76759


namespace root_reciprocal_sum_l767_76796

theorem root_reciprocal_sum (a b c : ℂ) : 
  (a^3 - 2*a^2 - a + 2 = 0) → 
  (b^3 - 2*b^2 - b + 2 = 0) → 
  (c^3 - 2*c^2 - c + 2 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  1/(a-1) + 1/(b-1) + 1/(c-1) = -1 := by
sorry

end root_reciprocal_sum_l767_76796


namespace rectangular_prism_sum_l767_76742

/-- A rectangular prism is a three-dimensional shape with six faces. -/
structure RectangularPrism where
  faces : Fin 6 → Rectangle

/-- The number of edges in a rectangular prism -/
def edges (p : RectangularPrism) : ℕ := 12

/-- The number of corners in a rectangular prism -/
def corners (p : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def faces (p : RectangularPrism) : ℕ := 6

/-- The sum of edges, corners, and faces in a rectangular prism is 26 -/
theorem rectangular_prism_sum (p : RectangularPrism) : 
  edges p + corners p + faces p = 26 := by
  sorry

end rectangular_prism_sum_l767_76742


namespace range_of_a_for_monotonic_f_l767_76711

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a + 3) * Real.exp (a * x)

theorem range_of_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ∨ (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (-2) 0 ∧ a ≠ 0 :=
sorry

end range_of_a_for_monotonic_f_l767_76711


namespace euclidean_continued_fraction_connection_l767_76715

/-- Euclidean algorithm steps -/
def euclidean_steps (m n : ℕ) : List (ℕ × ℕ) :=
  sorry

/-- Continued fraction representation -/
def continued_fraction (as : List ℕ) : ℚ :=
  sorry

/-- Theorem connecting Euclidean algorithm and continued fractions -/
theorem euclidean_continued_fraction_connection (m n : ℕ) (h : m < n) :
  let steps := euclidean_steps m n
  let as := steps.map Prod.fst
  ∀ k, k ≤ steps.length →
    continued_fraction (as.drop k) =
      (steps.get! k).snd / (steps.get! (k - 1)).snd :=
by sorry

end euclidean_continued_fraction_connection_l767_76715


namespace quadratic_function_b_value_l767_76799

/-- Given a quadratic function f(x) = ax² + bx + c, if f(2) - f(-2) = 8, then b = 2 -/
theorem quadratic_function_b_value (a b c : ℝ) (y₁ y₂ : ℝ) : 
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = 8 →
  b = 2 := by sorry

end quadratic_function_b_value_l767_76799


namespace geometric_sum_seven_terms_l767_76717

theorem geometric_sum_seven_terms : 
  let a : ℚ := 1/4  -- first term
  let r : ℚ := 1/4  -- common ratio
  let n : ℕ := 7    -- number of terms
  let S := a * (1 - r^n) / (1 - r)  -- formula for sum of geometric series
  S = 16383/49152 := by sorry

end geometric_sum_seven_terms_l767_76717


namespace pirate_treasure_distribution_l767_76772

/-- Represents the number of coins in the final distribution step -/
def x : ℕ := 13

/-- The sum of squares from 1 to n -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Pete's coins after the distribution -/
def pete_coins : ℕ := 5 * x^2

/-- Paul's coins after the distribution -/
def paul_coins : ℕ := x^2

/-- The total number of coins -/
def total_coins : ℕ := pete_coins + paul_coins

theorem pirate_treasure_distribution :
  (sum_of_squares x = pete_coins) ∧
  (total_coins = 1014) := by
  sorry

end pirate_treasure_distribution_l767_76772


namespace valid_purchase_options_l767_76732

/-- Represents the price of an item in kopecks -/
def ItemPrice : ℕ → Prop := λ p => ∃ (a : ℕ), p = 100 * a + 99

/-- The total cost of the purchase in kopecks -/
def TotalCost : ℕ := 20083

/-- Proposition that n is a valid number of items purchased -/
def ValidPurchase (n : ℕ) : Prop :=
  ∃ (p : ℕ), ItemPrice p ∧ n * p = TotalCost

theorem valid_purchase_options :
  ∀ n : ℕ, ValidPurchase n ↔ (n = 17 ∨ n = 117) :=
sorry

end valid_purchase_options_l767_76732


namespace f_sum_zero_a_geq_2_sufficient_not_necessary_l767_76729

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.log ((2 / (x + 1)) - 1) / Real.log 10

-- Define the domain A of function f
def A : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the function g (a is a parameter)
def g (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (1 - a^2 - 2*a*x - x^2)

-- Define the domain B of function g
def B (a : ℝ) : Set ℝ := {x | 1 - a^2 - 2*a*x - x^2 ≥ 0}

-- Statement 1: f(1/2013) + f(-1/2013) = 0
theorem f_sum_zero : f (1/2013) + f (-1/2013) = 0 := by sorry

-- Statement 2: a ≥ 2 is sufficient but not necessary for A ∩ B = ∅
theorem a_geq_2_sufficient_not_necessary :
  (∀ a : ℝ, a ≥ 2 → A ∩ B a = ∅) ∧
  ¬(∀ a : ℝ, A ∩ B a = ∅ → a ≥ 2) := by sorry

end

end f_sum_zero_a_geq_2_sufficient_not_necessary_l767_76729


namespace complex_number_location_l767_76739

theorem complex_number_location :
  let z : ℂ := (2 - I) / I
  (z.re < 0) ∧ (z.im < 0) := by sorry

end complex_number_location_l767_76739


namespace polynomial_division_theorem_l767_76769

theorem polynomial_division_theorem (x : ℝ) : 
  2*x^4 - 3*x^3 + x^2 + 5*x - 7 = (x + 1)*(2*x^3 - 5*x^2 + 6*x - 1) + (-6) := by
  sorry

end polynomial_division_theorem_l767_76769


namespace mn_value_l767_76760

theorem mn_value (m n : ℕ+) (h : m.val^2 + n.val^2 + 4*m.val - 46 = 0) :
  m.val * n.val = 5 ∨ m.val * n.val = 15 := by
sorry

end mn_value_l767_76760


namespace max_stores_visited_l767_76734

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) (double_visitors : ℕ) 
  (h1 : total_stores = 8)
  (h2 : total_visits = 23)
  (h3 : total_shoppers = 12)
  (h4 : double_visitors = 8)
  (h5 : double_visitors ≤ total_shoppers)
  (h6 : double_visitors * 2 ≤ total_visits)
  (h7 : ∀ n : ℕ, n ≤ total_shoppers → n > 0) :
  ∃ max_visits : ℕ, max_visits ≤ total_stores ∧ 
    max_visits * 1 + (total_shoppers - 1) * 1 + double_visitors * 1 = total_visits ∧
    ∀ n : ℕ, n ≤ total_shoppers → n * total_stores ≥ total_visits → n ≥ total_shoppers - 1 :=
by sorry

end max_stores_visited_l767_76734


namespace middle_card_is_six_l767_76798

theorem middle_card_is_six (a b c : ℕ) : 
  0 < a → 0 < b → 0 < c →
  a < b → b < c →
  a + b + c = 15 →
  (∀ x y z, x < y ∧ y < z ∧ x + y + z = 15 → x ≠ 3 ∨ (y ≠ 4 ∧ y ≠ 5)) →
  (∀ x y z, x < y ∧ y < z ∧ x + y + z = 15 → z ≠ 12 ∧ z ≠ 11 ∧ z ≠ 7) →
  (∃ p q, p < b ∧ b < q ∧ p + b + q = 15 ∧ (p ≠ a ∨ q ≠ c)) →
  b = 6 := by
sorry

end middle_card_is_six_l767_76798


namespace tetrahedron_volume_and_surface_area_l767_76721

/-- Given an equilateral cone with volume V and a tetrahedron circumscribed around it 
    with an equilateral triangle base, this theorem proves the volume and surface area 
    of the tetrahedron. -/
theorem tetrahedron_volume_and_surface_area 
  (V : ℝ) -- Volume of the equilateral cone
  (h : V > 0) -- Assumption that volume is positive
  : 
  ∃ (K F : ℝ), 
    K = (3 * V * Real.sqrt 3) / Real.pi ∧ 
    F = 9 * Real.sqrt 3 * (((3 * V ^ 2) / Real.pi ^ 2) ^ (1/3 : ℝ)) ∧
    K > 0 ∧ 
    F > 0
  := by sorry

end tetrahedron_volume_and_surface_area_l767_76721


namespace deposit_time_problem_l767_76728

/-- Proves that given the conditions of the problem, the deposit time is 3 years -/
theorem deposit_time_problem (initial_deposit : ℝ) (final_amount : ℝ) (final_amount_higher_rate : ℝ) 
  (h1 : initial_deposit = 8000)
  (h2 : final_amount = 10200)
  (h3 : final_amount_higher_rate = 10680) :
  ∃ (r : ℝ), 
    final_amount = initial_deposit + initial_deposit * (r / 100) * 3 ∧
    final_amount_higher_rate = initial_deposit + initial_deposit * ((r + 2) / 100) * 3 :=
sorry

end deposit_time_problem_l767_76728


namespace sequence_problem_l767_76767

/-- Given a sequence where each term is obtained by doubling the previous term and adding 4,
    if the third term is 52, then the first term is 10. -/
theorem sequence_problem (x : ℝ) : 
  let second_term := 2 * x + 4
  let third_term := 2 * second_term + 4
  third_term = 52 → x = 10 := by
sorry

end sequence_problem_l767_76767


namespace largest_r_for_sequence_convergence_r_two_satisfies_condition_l767_76722

theorem largest_r_for_sequence_convergence (r : ℝ) :
  r > 2 →
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, 0 < a n) ∧
    (∀ n : ℕ, a n ≤ a (n + 2) ∧ a (n + 2) ≤ Real.sqrt (a n ^ 2 + r * a (n + 1))) ∧
    (¬ ∃ M : ℕ, ∀ n ≥ M, a (n + 2) = a n) :=
by sorry

theorem r_two_satisfies_condition :
  ∀ (a : ℕ → ℕ), (∀ n : ℕ, 0 < a n) →
    (∀ n : ℕ, a n ≤ a (n + 2) ∧ a (n + 2) ≤ Real.sqrt (a n ^ 2 + 2 * a (n + 1))) →
    ∃ M : ℕ, ∀ n ≥ M, a (n + 2) = a n :=
by sorry

end largest_r_for_sequence_convergence_r_two_satisfies_condition_l767_76722


namespace coin_value_equality_l767_76781

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of quarters in the first group -/
def quarters_1 : ℕ := 15

/-- The number of dimes in the first group -/
def dimes_1 : ℕ := 10

/-- The number of quarters in the second group -/
def quarters_2 : ℕ := 25

theorem coin_value_equality (n : ℕ) : 
  quarters_1 * quarter_value + dimes_1 * dime_value = 
  quarters_2 * quarter_value + n * dime_value → n = 35 := by
sorry

end coin_value_equality_l767_76781


namespace false_or_false_is_false_l767_76726

theorem false_or_false_is_false (p q : Prop) (hp : ¬p) (hq : ¬q) : ¬(p ∨ q) := by
  sorry

end false_or_false_is_false_l767_76726


namespace simplify_fraction_l767_76714

theorem simplify_fraction : (140 : ℚ) / 210 = 2 / 3 := by
  sorry

end simplify_fraction_l767_76714


namespace celebrity_photo_matching_probability_l767_76736

theorem celebrity_photo_matching_probability :
  ∀ (n : ℕ) (k : ℕ),
    n = 5 →
    k = 2 →
    (Nat.choose n k * k.factorial : ℚ)⁻¹ = 1 / 20 :=
by sorry

end celebrity_photo_matching_probability_l767_76736


namespace square_sum_and_product_l767_76791

theorem square_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 49) : 
  x^2 + y^2 = 25 ∧ x * y = -12 := by
sorry

end square_sum_and_product_l767_76791


namespace no_integer_solutions_l767_76724

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 10*m^2 + 11*m + 2 = 81*n^3 + 27*n^2 + 3*n - 8 := by
  sorry

end no_integer_solutions_l767_76724


namespace subsets_containing_six_l767_76703

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem subsets_containing_six (A : Finset ℕ) (h : A ⊆ S) (h6 : 6 ∈ A) :
  (Finset.filter (fun A => 6 ∈ A) (Finset.powerset S)).card = 32 := by
  sorry

end subsets_containing_six_l767_76703


namespace expression_simplification_l767_76735

theorem expression_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a + b ≠ 0) :
  (3 * a^2 + 3 * a * b + 3 * b^2) / (4 * a + 4 * b) *
  (2 * a^2 - 2 * b^2) / (9 * a^3 - 9 * b^3) = 1/6 := by
  sorry

end expression_simplification_l767_76735


namespace equal_area_rectangles_l767_76709

/-- Given two rectangles with equal area, where one rectangle measures 12 inches by 15 inches
    and the other has a length of 9 inches, prove that the width of the second rectangle is 20 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length jordan_width : ℝ)
    (h1 : carol_length = 12)
    (h2 : carol_width = 15)
    (h3 : jordan_length = 9)
    (h4 : carol_length * carol_width = jordan_length * jordan_width) :
    jordan_width = 20 := by
  sorry

end equal_area_rectangles_l767_76709


namespace arithmetic_calculation_l767_76785

theorem arithmetic_calculation : 1375 + 150 / 50 * 3 - 275 = 1109 := by
  sorry

end arithmetic_calculation_l767_76785


namespace percentage_increase_l767_76794

theorem percentage_increase (x : ℝ) (h1 : x = 90.4) (h2 : ∃ p, x = 80 * (1 + p / 100)) : 
  ∃ p, x = 80 * (1 + p / 100) ∧ p = 13 := by
sorry

end percentage_increase_l767_76794


namespace solution_set_of_even_increasing_function_l767_76789

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := (x - 2) * (a * x + b)

-- State the theorem
theorem solution_set_of_even_increasing_function 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x)) 
  (h_increasing : ∀ x y, 0 < x → x < y → f a b x < f a b y) :
  {x : ℝ | f a b (2 - x) > 0} = {x : ℝ | x > 4 ∨ x < 0} := by
sorry

end solution_set_of_even_increasing_function_l767_76789


namespace p_minus_q_equals_two_l767_76708

-- Define an invertible function g
variable (g : ℝ → ℝ)
variable (hg : Function.Injective g)

-- Define p and q based on the given conditions
variable (p q : ℝ)
variable (hp : g p = 3)
variable (hq : g q = 5)

-- State the theorem
theorem p_minus_q_equals_two : p - q = 2 := by
  sorry

end p_minus_q_equals_two_l767_76708


namespace game_result_l767_76737

def score (n : Nat) : Nat :=
  if n % 3 = 0 then 5
  else if n % 2 = 0 then 3
  else 0

def allieRolls : List Nat := [3, 5, 6, 2, 4]
def bettyRolls : List Nat := [3, 2, 1, 6, 4]

def totalScore (rolls : List Nat) : Nat :=
  (rolls.map score).sum

theorem game_result : totalScore allieRolls * totalScore bettyRolls = 256 := by
  sorry

end game_result_l767_76737
