import Mathlib

namespace recipe_flour_amount_l1079_107937

theorem recipe_flour_amount (flour_added : ℕ) (flour_needed : ℕ) : 
  flour_added = 2 → flour_needed = 6 → flour_added + flour_needed = 8 :=
by
  sorry

end recipe_flour_amount_l1079_107937


namespace matthew_lollipops_l1079_107901

theorem matthew_lollipops (total_lollipops : ℕ) (friends : ℕ) (h1 : total_lollipops = 500) (h2 : friends = 15) :
  total_lollipops % friends = 5 := by
  sorry

end matthew_lollipops_l1079_107901


namespace prime_square_mod_180_l1079_107960

theorem prime_square_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) :
  ∃ r : Fin 180, p^2 % 180 = r.val ∧ (r.val = 1 ∨ r.val = 145) := by
  sorry

end prime_square_mod_180_l1079_107960


namespace least_multiple_of_primes_l1079_107942

theorem least_multiple_of_primes : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → n ≤ m) ∧ 
  3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ n = 105 := by
  sorry

end least_multiple_of_primes_l1079_107942


namespace line_through_point_parallel_to_given_l1079_107920

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

def Line.passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.b

def Line.parallel_to (l1 l2 : Line) : Prop :=
  l1.m = l2.m

theorem line_through_point_parallel_to_given : 
  let P : Point := ⟨1, 2⟩
  let given_line : Line := ⟨2, 3⟩
  let parallel_line : Line := ⟨2, 0⟩
  parallel_line.passes_through P ∧ parallel_line.parallel_to given_line := by
  sorry


end line_through_point_parallel_to_given_l1079_107920


namespace sin_240_degrees_l1079_107928

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l1079_107928


namespace polynomial_simplification_l1079_107980

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 6 * x - 5) - (2 * x^3 + 2 * x^2 + 3 * x - 8) = x^3 + 2 * x^2 + 3 * x + 3 := by
  sorry

end polynomial_simplification_l1079_107980


namespace job_completion_time_l1079_107979

/-- Represents the workforce and time required to complete a job -/
structure JobInfo where
  initialWorkforce : ℕ
  initialDays : ℕ
  extraWorkers : ℕ
  joinInterval : ℕ

/-- Calculates the total time required to complete the job given the job information -/
def calculateTotalTime (job : JobInfo) : ℕ :=
  sorry

/-- Theorem stating that for the given job information, the total time is 12 days -/
theorem job_completion_time (job : JobInfo) 
  (h1 : job.initialWorkforce = 20)
  (h2 : job.initialDays = 15)
  (h3 : job.extraWorkers = 10)
  (h4 : job.joinInterval = 5) : 
  calculateTotalTime job = 12 :=
  sorry

end job_completion_time_l1079_107979


namespace cuboid_volume_l1079_107990

/-- The volume of a cuboid with edges 2, 5, and 8 is 80 -/
theorem cuboid_volume : 
  let edge1 : ℝ := 2
  let edge2 : ℝ := 5
  let edge3 : ℝ := 8
  edge1 * edge2 * edge3 = 80 := by sorry

end cuboid_volume_l1079_107990


namespace square_equation_solutions_cubic_equation_solution_l1079_107950

-- Part 1
theorem square_equation_solutions (x : ℝ) :
  (x - 2)^2 = 9 ↔ x = 5 ∨ x = -1 := by sorry

-- Part 2
theorem cubic_equation_solution (x : ℝ) :
  27 * (x + 1)^3 + 8 = 0 ↔ x = -5/3 := by sorry

end square_equation_solutions_cubic_equation_solution_l1079_107950


namespace work_completion_time_l1079_107993

theorem work_completion_time (x : ℝ) : 
  (x > 0) →  -- A's completion time is positive
  (4 * (1 / x + 1 / 30) = 1 / 3) →  -- Equation from working together
  (x = 20) :=
by sorry

end work_completion_time_l1079_107993


namespace evaluate_expression_l1079_107954

theorem evaluate_expression : (2^3)^2 - (3^2)^3 = -665 := by
  sorry

end evaluate_expression_l1079_107954


namespace arithmetic_geometric_mean_problem_l1079_107956

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h1 : (x + y) / 2 = 20) 
  (h2 : Real.sqrt (x * y) = Real.sqrt 132) : 
  x^2 + y^2 = 1336 := by
sorry

end arithmetic_geometric_mean_problem_l1079_107956


namespace larger_solution_of_quadratic_l1079_107944

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x - 48 = 0 → 
  (x = 16 ∨ x = -3) → 
  ∃ y : ℝ, y^2 - 13*y - 48 = 0 ∧ y ≠ x ∧ x ≤ y → x = 16 :=
by sorry

end larger_solution_of_quadratic_l1079_107944


namespace min_value_problem_l1079_107970

theorem min_value_problem (a b m n : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n)
  (hab : a + b = 1) (hmn : m * n = 2) :
  (a * m + b * n) * (b * m + a * n) ≥ 3/2 := by
  sorry

end min_value_problem_l1079_107970


namespace ann_has_eight_bags_l1079_107945

/-- Represents the number of apples in one of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- Represents the total number of apples Pam has -/
def pams_total_apples : ℕ := 1800

/-- Represents the total number of apples Ann has -/
def anns_total_apples : ℕ := 1800

/-- Represents the number of apples in one of Pam's bags -/
def pams_bag_count : ℕ := 3 * geralds_bag_count

/-- Represents the number of apples in one of Ann's bags -/
def anns_bag_count : ℕ := 2 * pams_bag_count

/-- Theorem stating that Ann has 8 bags of apples -/
theorem ann_has_eight_bags : 
  anns_total_apples / anns_bag_count = 8 ∧ 
  anns_total_apples % anns_bag_count = 0 :=
by sorry

end ann_has_eight_bags_l1079_107945


namespace tangent_line_y_intercept_l1079_107900

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 11

-- Define the point of tangency
def P : ℝ × ℝ := (1, 12)

-- Theorem statement
theorem tangent_line_y_intercept :
  let df (x : ℝ) := 3 * x^2  -- Derivative of f
  let m : ℝ := df P.1        -- Slope of the tangent line
  let b : ℝ := P.2 - m * P.1 -- y-intercept of the tangent line
  b = 9 := by sorry

end tangent_line_y_intercept_l1079_107900


namespace coloring_count_l1079_107958

/-- The number of colors available -/
def num_colors : ℕ := 5

/-- The number of parts to be colored -/
def num_parts : ℕ := 3

/-- A function that calculates the number of coloring possibilities -/
def count_colorings : ℕ := num_colors * (num_colors - 1) * (num_colors - 2)

/-- Theorem stating that the number of valid colorings is 60 -/
theorem coloring_count : count_colorings = 60 := by
  sorry

end coloring_count_l1079_107958


namespace kitchen_tiling_l1079_107974

def kitchen_length : ℕ := 20
def kitchen_width : ℕ := 15
def border_width : ℕ := 2
def border_tile_length : ℕ := 2
def border_tile_width : ℕ := 1
def inner_tile_size : ℕ := 3

def border_tiles_count : ℕ := 
  2 * (kitchen_length - 2 * border_width) / border_tile_length +
  2 * (kitchen_width - 2 * border_width) / border_tile_length

def inner_area : ℕ := (kitchen_length - 2 * border_width) * (kitchen_width - 2 * border_width)

def inner_tiles_count : ℕ := (inner_area + inner_tile_size^2 - 1) / inner_tile_size^2

def total_tiles : ℕ := border_tiles_count + inner_tiles_count

theorem kitchen_tiling :
  total_tiles = 48 :=
sorry

end kitchen_tiling_l1079_107974


namespace race_length_is_18_l1079_107953

/-- The length of a cross-country relay race with 5 members -/
def race_length : ℕ :=
  let other_members : ℕ := 4
  let other_distance : ℕ := 3
  let ralph_multiplier : ℕ := 2
  (other_members * other_distance) + (ralph_multiplier * other_distance)

/-- Theorem: The length of the race is 18 km -/
theorem race_length_is_18 : race_length = 18 := by
  sorry

end race_length_is_18_l1079_107953


namespace same_quotient_remainder_numbers_l1079_107966

theorem same_quotient_remainder_numbers : 
  {a : ℕ | ∃ q : ℕ, 0 < q ∧ q < 6 ∧ a = 7 * q} = {7, 14, 21, 28, 35} := by
  sorry

end same_quotient_remainder_numbers_l1079_107966


namespace right_pyramid_base_side_length_l1079_107935

/-- Represents a right pyramid with an equilateral triangular base -/
structure RightPyramid where
  base_side_length : ℝ
  slant_height : ℝ
  lateral_face_area : ℝ

/-- Theorem: If the area of one lateral face is 90 square meters and the slant height is 20 meters,
    then the side length of the base is 9 meters -/
theorem right_pyramid_base_side_length 
  (pyramid : RightPyramid) 
  (h1 : pyramid.lateral_face_area = 90) 
  (h2 : pyramid.slant_height = 20) : 
  pyramid.base_side_length = 9 := by
  sorry

end right_pyramid_base_side_length_l1079_107935


namespace roberts_birth_year_l1079_107951

theorem roberts_birth_year (n : ℕ) : 
  (n + 1)^2 - n^2 = 89 → n^2 = 1936 := by
  sorry

end roberts_birth_year_l1079_107951


namespace ninth_term_of_sequence_l1079_107903

theorem ninth_term_of_sequence (n : ℕ) : 
  let a : ℕ → ℝ := fun n => Real.sqrt (3 * (2 * n - 1))
  a 14 = 9 := by
sorry

end ninth_term_of_sequence_l1079_107903


namespace right_triangle_sets_set_a_not_right_triangle_l1079_107936

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that set A cannot form a right triangle while others can -/
theorem right_triangle_sets :
  ¬(is_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5)) ∧
  (is_right_triangle 1 (Real.sqrt 2) (Real.sqrt 3)) ∧
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle 3 4 5) := by
  sorry

/-- The specific theorem for set A -/
theorem set_a_not_right_triangle :
  ¬(is_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5)) := by
  sorry

end right_triangle_sets_set_a_not_right_triangle_l1079_107936


namespace angle_measure_from_cosine_l1079_107962

theorem angle_measure_from_cosine (A : Real) : 
  0 < A → A < Real.pi / 2 → -- A is acute
  Real.cos A = Real.sqrt 3 / 2 → -- cos A = √3/2
  A = Real.pi / 6 -- A = 30° (π/6 radians)
:= by sorry

end angle_measure_from_cosine_l1079_107962


namespace square_bound_values_l1079_107995

theorem square_bound_values (k : ℤ) : 
  (∃ (s : Finset ℤ), (∀ x ∈ s, 121 < x^2 ∧ x^2 < 225) ∧ s.card ≤ 3 ∧ 
   (∀ y : ℤ, 121 < y^2 ∧ y^2 < 225 → y ∈ s)) :=
by sorry

end square_bound_values_l1079_107995


namespace y_intercept_of_specific_line_l1079_107932

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ := sorry

/-- Given a line with slope 3 and x-intercept (-3, 0), its y-intercept is (0, 9). -/
theorem y_intercept_of_specific_line :
  let l : Line := { slope := 3, x_intercept := (-3, 0) }
  y_intercept l = (0, 9) := by sorry

end y_intercept_of_specific_line_l1079_107932


namespace registered_number_scientific_notation_l1079_107973

/-- The number of people registered for the national college entrance examination in 2023 -/
def registered_number : ℝ := 12910000

/-- The scientific notation representation of the registered number -/
def scientific_notation : ℝ := 1.291 * (10 ^ 7)

/-- Theorem stating that the registered number is equal to its scientific notation representation -/
theorem registered_number_scientific_notation : registered_number = scientific_notation := by
  sorry

end registered_number_scientific_notation_l1079_107973


namespace video_game_sales_earnings_l1079_107940

theorem video_game_sales_earnings 
  (total_games : ℕ) 
  (non_working_games : ℕ) 
  (price_per_game : ℕ) : 
  total_games = 16 → 
  non_working_games = 8 → 
  price_per_game = 7 → 
  (total_games - non_working_games) * price_per_game = 56 := by
sorry

end video_game_sales_earnings_l1079_107940


namespace base8_digit_product_l1079_107964

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers --/
def productList (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 8 representation of 7254 (base 10) is 72 --/
theorem base8_digit_product :
  productList (toBase8 7254) = 72 := by
  sorry

end base8_digit_product_l1079_107964


namespace quadratic_function_property_l1079_107908

def f (b c x : ℝ) : ℝ := -x^2 + b*x + c

theorem quadratic_function_property (b c : ℝ) :
  f b c 2 + f b c 4 = 12138 →
  3*b + c = 6079 →
  f b c 3 = 6070 := by
sorry

end quadratic_function_property_l1079_107908


namespace min_apples_in_basket_l1079_107955

theorem min_apples_in_basket (N : ℕ) : 
  N ≥ 67 ∧ 
  N % 3 = 1 ∧ 
  N % 4 = 3 ∧ 
  N % 5 = 2 ∧
  (∀ m : ℕ, m < N → ¬(m % 3 = 1 ∧ m % 4 = 3 ∧ m % 5 = 2)) := by
  sorry

end min_apples_in_basket_l1079_107955


namespace locus_is_ellipse_l1079_107939

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 91 = 0

-- Define the property of being externally tangent
def externally_tangent (x y : ℝ) : Prop :=
  ∃ (R : ℝ), R > 0 ∧ (x + 3)^2 + y^2 = (R + 2)^2

-- Define the property of being internally tangent
def internally_tangent (x y : ℝ) : Prop :=
  ∃ (R : ℝ), R > 0 ∧ (x - 3)^2 + y^2 = (10 - R)^2

-- Define the locus of points
def locus (x y : ℝ) : Prop :=
  externally_tangent x y ∧ internally_tangent x y

-- Theorem stating that the locus forms an ellipse
theorem locus_is_ellipse :
  ∀ (x y : ℝ), locus x y → (x + 3)^2 / 36 + y^2 / 27 = 1 :=
sorry

end locus_is_ellipse_l1079_107939


namespace cube_sum_rational_l1079_107902

theorem cube_sum_rational (a b c : ℚ) 
  (h1 : a - b + c = 3) 
  (h2 : a^2 + b^2 + c^2 = 3) : 
  a^3 + b^3 + c^3 = 1 := by
sorry

end cube_sum_rational_l1079_107902


namespace nine_digit_divisible_by_101_l1079_107931

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ | 100 ≤ n ∧ n < 1000 }

/-- Converts a three-digit number to a nine-digit number by repeating it three times -/
def toNineDigitNumber (n : ThreeDigitNumber) : ℕ :=
  1000000 * n + 1000 * n + n

/-- Theorem: Any nine-digit number formed by repeating a three-digit number three times is divisible by 101 -/
theorem nine_digit_divisible_by_101 (n : ThreeDigitNumber) :
  ∃ k : ℕ, toNineDigitNumber n = 101 * k := by
  sorry

end nine_digit_divisible_by_101_l1079_107931


namespace tray_height_l1079_107981

theorem tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) : 
  side_length = 120 →
  cut_distance = Real.sqrt 20 →
  cut_angle = π / 4 →
  ∃ (height : ℝ), height = Real.sqrt 10 ∧ 
    height = (cut_distance * Real.sqrt 2) / 2 :=
sorry

end tray_height_l1079_107981


namespace inverse_function_symmetry_l1079_107987

-- Define a function and its inverse
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Define symmetry about the line x - y = 0
def symmetric_about_x_eq_y (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Theorem statement
theorem inverse_function_symmetry :
  symmetric_about_x_eq_y f f_inv :=
sorry

end inverse_function_symmetry_l1079_107987


namespace existence_of_linear_bound_l1079_107978

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 2 * sin x) * (2^(-x) + 1)

theorem existence_of_linear_bound :
  ∃ (a b m : ℝ), ∀ x > 0, |f x - a * x - b| ≤ m :=
sorry

end existence_of_linear_bound_l1079_107978


namespace sum_smallest_largest_fourdigit_l1079_107948

/-- A function that generates all four-digit numbers using the digits 0, 3, 4, and 8 -/
def fourDigitNumbers : List Nat := sorry

/-- The smallest four-digit number formed using 0, 3, 4, and 8 -/
def smallestNumber : Nat := sorry

/-- The largest four-digit number formed using 0, 3, 4, and 8 -/
def largestNumber : Nat := sorry

/-- Theorem stating that the sum of the smallest and largest four-digit numbers
    formed using 0, 3, 4, and 8 is 11478 -/
theorem sum_smallest_largest_fourdigit :
  smallestNumber + largestNumber = 11478 := by sorry

end sum_smallest_largest_fourdigit_l1079_107948


namespace max_candies_one_student_l1079_107976

/-- Given a class of students, proves the maximum number of candies one student could have taken -/
theorem max_candies_one_student 
  (n : ℕ) -- number of students
  (mean : ℕ) -- mean number of candies per student
  (min_candies : ℕ) -- minimum number of candies per student
  (h1 : n = 25) -- there are 25 students
  (h2 : mean = 6) -- the mean number of candies is 6
  (h3 : min_candies = 2) -- each student takes at least 2 candies
  : ∃ (max_candies : ℕ), max_candies = 102 ∧ 
    max_candies = n * mean - (n - 1) * min_candies :=
by sorry

end max_candies_one_student_l1079_107976


namespace paint_mixture_problem_l1079_107982

/-- Given a paint mixture with ratio 7:2:1:1 for blue, red, white, and green,
    prove that if 140 oz of blue paint is used and the total mixture should not exceed 220 oz,
    then 20 oz of white paint is required. -/
theorem paint_mixture_problem (blue red white green : ℕ) 
  (ratio : blue = 7 ∧ red = 2 ∧ white = 1 ∧ green = 1) 
  (blue_amount : ℕ) (total_limit : ℕ)
  (h_blue_amount : blue_amount = 140)
  (h_total_limit : total_limit = 220) :
  let total_parts := blue + red + white + green
  let ounces_per_part := blue_amount / blue
  let white_amount := ounces_per_part * white
  white_amount = 20 ∧ white_amount ≤ total_limit - blue_amount :=
by sorry

end paint_mixture_problem_l1079_107982


namespace system_1_solution_system_2_solution_l1079_107968

-- System 1
theorem system_1_solution :
  ∃ (x y : ℝ), y = 2 * x ∧ x + y = 12 ∧ x = 4 ∧ y = 8 := by sorry

-- System 2
theorem system_2_solution :
  ∃ (x y : ℝ), 3 * x + 5 * y = 21 ∧ 2 * x - 5 * y = -11 ∧ x = 2 ∧ y = 3 := by sorry

end system_1_solution_system_2_solution_l1079_107968


namespace robin_water_bottles_l1079_107923

theorem robin_water_bottles (morning_bottles : ℕ) (afternoon_bottles : ℕ) 
  (h1 : morning_bottles = 7) 
  (h2 : afternoon_bottles = 7) : 
  morning_bottles + afternoon_bottles = 14 := by
  sorry

end robin_water_bottles_l1079_107923


namespace prob_green_marble_l1079_107934

/-- The probability of drawing a green marble from a box of 90 marbles -/
theorem prob_green_marble (total_marbles : ℕ) (prob_white : ℝ) (prob_red_or_blue : ℝ) :
  total_marbles = 90 →
  prob_white = 1 / 6 →
  prob_red_or_blue = 0.6333333333333333 →
  ∃ (prob_green : ℝ), prob_green = 0.2 ∧ prob_white + prob_red_or_blue + prob_green = 1 :=
by sorry

end prob_green_marble_l1079_107934


namespace right_triangles_shared_hypotenuse_l1079_107933

/-- Given two right triangles ABC and ABD sharing hypotenuse AB, 
    where BC = 3, AC = a, and AD = 4, prove that BD = √(a² - 7) -/
theorem right_triangles_shared_hypotenuse 
  (a : ℝ) 
  (h : a ≥ Real.sqrt 7) : 
  ∃ (AB BC AC AD BD : ℝ),
    BC = 3 ∧ 
    AC = a ∧ 
    AD = 4 ∧
    AB ^ 2 = AC ^ 2 + BC ^ 2 ∧ 
    AB ^ 2 = AD ^ 2 + BD ^ 2 ∧
    BD = Real.sqrt (a ^ 2 - 7) := by
  sorry


end right_triangles_shared_hypotenuse_l1079_107933


namespace linear_function_quadrant_l1079_107925

theorem linear_function_quadrant (m : ℤ) : 
  (∀ x y : ℝ, y = (m + 4) * x + (m + 2) → ¬(x < 0 ∧ y > 0)) →
  (m = -3 ∨ m = -2) :=
sorry

end linear_function_quadrant_l1079_107925


namespace fifteen_power_division_l1079_107988

theorem fifteen_power_division : (15 : ℕ) ^ 11 / (15 : ℕ) ^ 8 = 3375 := by
  sorry

end fifteen_power_division_l1079_107988


namespace solve_equation_l1079_107986

theorem solve_equation : ∃ x : ℚ, 3 * x + 15 = (1/3) * (8 * x - 24) ∧ x = -69 := by
  sorry

end solve_equation_l1079_107986


namespace range_of_a_minus_b_l1079_107994

theorem range_of_a_minus_b (a b : ℝ) (ha : -1 < a ∧ a < 2) (hb : -2 < b ∧ b < 1) :
  ∃ x, -2 < x ∧ x < 4 ∧ ∃ a' b', -1 < a' ∧ a' < 2 ∧ -2 < b' ∧ b' < 1 ∧ x = a' - b' :=
by sorry

end range_of_a_minus_b_l1079_107994


namespace system_solution_range_l1079_107946

theorem system_solution_range (x y k : ℝ) : 
  (2 * x - 3 * y = 5) → 
  (2 * x - y = k) → 
  (x > y) → 
  (k > -5) :=
sorry

end system_solution_range_l1079_107946


namespace minimum_donut_cost_minimum_donut_cost_proof_l1079_107914

/-- The minimum cost to buy at least 550 donuts, given that they are sold in dozens at $7.49 per dozen -/
theorem minimum_donut_cost : ℝ → Prop :=
  fun cost =>
    ∀ n : ℕ,
      (12 * n ≥ 550) →
      (cost ≤ n * 7.49) ∧
      (∃ m : ℕ, (12 * m ≥ 550) ∧ (cost = m * 7.49)) →
      cost = 344.54

/-- Proof of the minimum_donut_cost theorem -/
theorem minimum_donut_cost_proof : minimum_donut_cost 344.54 := by
  sorry

end minimum_donut_cost_minimum_donut_cost_proof_l1079_107914


namespace isosceles_triangle_quadratic_roots_l1079_107963

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 - 8*x + m = 0

-- Define an isosceles triangle
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

-- Define the triangle inequality
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

-- Main theorem
theorem isosceles_triangle_quadratic_roots (m : ℝ) : 
  (∃ x y : ℝ, 
    quadratic_equation x m ∧ 
    quadratic_equation y m ∧ 
    x ≠ y ∧
    is_isosceles_triangle 6 x y ∧
    satisfies_triangle_inequality 6 x y) ↔ 
  (m = 12 ∨ m = 16) :=
sorry

end isosceles_triangle_quadratic_roots_l1079_107963


namespace prove_d_value_l1079_107910

def floor_d : ℤ := -9

def frac_d : ℚ := 2/5

theorem prove_d_value :
  let d : ℚ := floor_d + frac_d
  (3 * floor_d^2 + 14 * floor_d - 45 = 0) ∧
  (5 * frac_d^2 - 18 * frac_d + 8 = 0) ∧
  (0 ≤ frac_d ∧ frac_d < 1) →
  d = -43/5 := by sorry

end prove_d_value_l1079_107910


namespace test_scores_theorem_l1079_107957

-- Define the total number of tests
def total_tests : ℕ := 13

-- Define the number of tests with scores exceeding 90
def high_score_tests : ℕ := 4

-- Define the number of tests taken by A and B
def A_tests : ℕ := 6
def B_tests : ℕ := 7

-- Define the number of excellent scores for A and B
def A_excellent : ℕ := 3
def B_excellent : ℕ := 4

-- Define the number of tests selected from A and B
def A_selected : ℕ := 4
def B_selected : ℕ := 3

-- Define the probability of selecting a test with score > 90
def prob_high_score : ℚ := high_score_tests / total_tests

-- Define the expected value of X (excellent scores when selecting 4 out of A's 6 tests)
def E_X : ℚ := 2

-- Define the expected value of Y (excellent scores when selecting 3 out of B's 7 tests)
def E_Y : ℚ := 12 / 7

theorem test_scores_theorem :
  (prob_high_score = 4 / 13) ∧
  (E_X = 2) ∧
  (E_Y = 12 / 7) ∧
  (E_X > E_Y) := by
  sorry

end test_scores_theorem_l1079_107957


namespace perpendicular_line_equation_l1079_107917

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem -/
theorem perpendicular_line_equation (given_line : Line) (p : Point) :
  given_line.a = 2 ∧ given_line.b = -3 ∧ given_line.c = 4 ∧
  p.x = -1 ∧ p.y = 2 →
  ∃ (l : Line), l.contains p ∧ l.perpendicular given_line ∧
  l.a = 3 ∧ l.b = 2 ∧ l.c = -1 := by
  sorry


end perpendicular_line_equation_l1079_107917


namespace restaurant_bill_l1079_107991

theorem restaurant_bill (total_friends : ℕ) (contributing_friends : ℕ) (extra_payment : ℚ) (total_bill : ℚ) :
  total_friends = 10 →
  contributing_friends = 9 →
  extra_payment = 3 →
  total_bill = (contributing_friends * (total_bill / total_friends + extra_payment)) →
  total_bill = 270 :=
by sorry

end restaurant_bill_l1079_107991


namespace f_odd_f_increasing_f_odd_and_increasing_l1079_107904

/-- The function f(x) = x|x| -/
def f (x : ℝ) : ℝ := x * abs x

/-- f is an odd function -/
theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

/-- f is an increasing function -/
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

/-- f is both odd and increasing -/
theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) := by sorry

end f_odd_f_increasing_f_odd_and_increasing_l1079_107904


namespace arrangements_with_A_B_at_ends_arrangements_with_A_B_not_adjacent_adjustment_methods_l1079_107926

-- Define the number of instructors and students
def num_instructors : ℕ := 3
def num_students : ℕ := 7

-- Define the theorem for part (1)
theorem arrangements_with_A_B_at_ends :
  (2 * Nat.factorial 5 * Nat.factorial num_instructors : ℕ) = 1440 := by sorry

-- Define the theorem for part (2)
theorem arrangements_with_A_B_not_adjacent :
  (Nat.factorial 5 * Nat.choose 6 2 * 2 * Nat.factorial num_instructors : ℕ) = 21600 := by sorry

-- Define the theorem for part (3)
theorem adjustment_methods :
  (Nat.choose num_students 2 * (Nat.factorial 5 / Nat.factorial 3) : ℕ) = 420 := by sorry

end arrangements_with_A_B_at_ends_arrangements_with_A_B_not_adjacent_adjustment_methods_l1079_107926


namespace product_consecutive_integers_even_l1079_107969

theorem product_consecutive_integers_even (n : ℤ) : ∃ k : ℤ, n * (n + 1) = 2 * k := by
  sorry

end product_consecutive_integers_even_l1079_107969


namespace max_profit_at_max_price_l1079_107938

/-- Represents the monthly sales and profit of eye-protection lamps --/
structure LampSales where
  cost_price : ℝ
  selling_price : ℝ
  monthly_sales : ℝ
  profit : ℝ

/-- The conditions and constraints of the lamp sales problem --/
def lamp_sales_constraints (s : LampSales) : Prop :=
  s.cost_price = 40 ∧
  s.selling_price ≥ s.cost_price ∧
  s.selling_price ≤ 2 * s.cost_price ∧
  s.monthly_sales = -s.selling_price + 140 ∧
  s.profit = (s.selling_price - s.cost_price) * s.monthly_sales

/-- Theorem stating that the maximum monthly profit is achieved at the highest allowed selling price --/
theorem max_profit_at_max_price (s : LampSales) :
  lamp_sales_constraints s →
  ∃ (max_s : LampSales),
    lamp_sales_constraints max_s ∧
    max_s.selling_price = 80 ∧
    max_s.profit = 2400 ∧
    ∀ (other_s : LampSales), lamp_sales_constraints other_s → other_s.profit ≤ max_s.profit :=
by sorry

end max_profit_at_max_price_l1079_107938


namespace inequality_proof_equality_condition_l1079_107915

theorem inequality_proof (a b c d : ℝ) 
  (ha : 1 ≤ a ∧ a ≤ 2) 
  (hb : 1 ≤ b ∧ b ≤ 2) 
  (hc : 1 ≤ c ∧ c ≤ 2) 
  (hd : 1 ≤ d ∧ d ≤ 2) : 
  (a + b) / (b + c) + (c + d) / (d + a) ≤ 4 * (a + c) / (b + d) :=
sorry

theorem equality_condition (a b c d : ℝ) 
  (ha : 1 ≤ a ∧ a ≤ 2) 
  (hb : 1 ≤ b ∧ b ≤ 2) 
  (hc : 1 ≤ c ∧ c ≤ 2) 
  (hd : 1 ≤ d ∧ d ≤ 2) :
  (a + b) / (b + c) + (c + d) / (d + a) = 4 * (a + c) / (b + d) ↔ a = b ∧ b = c ∧ c = d :=
sorry

end inequality_proof_equality_condition_l1079_107915


namespace sports_conference_games_l1079_107983

/-- Calculates the total number of games in a sports conference season -/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  (total_teams * (intra_division_games * (teams_per_division - 1) + 
  inter_division_games * teams_per_division)) / 2

/-- Theorem stating the total number of games in the given sports conference -/
theorem sports_conference_games : 
  total_games 16 8 2 1 = 176 := by
  sorry

end sports_conference_games_l1079_107983


namespace expression_value_l1079_107985

theorem expression_value (x y z : ℝ) 
  (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 5) :
  ((x - 2) / (3 - z) * (y - 3) / (5 - x) * (z - 5) / (2 - y))^2 = 1 := by
  sorry

end expression_value_l1079_107985


namespace complex_number_location_l1079_107941

theorem complex_number_location :
  let z : ℂ := (1 + Complex.I) * (1 - 2 * Complex.I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end complex_number_location_l1079_107941


namespace percentage_relation_l1079_107907

theorem percentage_relation (T S F : ℝ) 
  (h1 : F = 0.06 * T) 
  (h2 : F = (1/3) * S) : 
  S = 0.18 * T := by
sorry

end percentage_relation_l1079_107907


namespace complement_of_union_l1079_107999

def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3, 4}
def B : Set ℕ := {3, 5}

theorem complement_of_union :
  (U \ (A ∪ B)) = {2} := by
  sorry

end complement_of_union_l1079_107999


namespace floor_ceil_sum_l1079_107984

theorem floor_ceil_sum : ⌊(-1.001 : ℝ)⌋ + ⌈(3.999 : ℝ)⌉ + ⌊(0.998 : ℝ)⌋ = 2 := by
  sorry

end floor_ceil_sum_l1079_107984


namespace square_difference_identity_l1079_107967

theorem square_difference_identity (x : ℝ) (c : ℝ) (hc : c > 0) :
  (x^2 + c)^2 - (x^2 - c)^2 = 4*x^2*c := by
  sorry

end square_difference_identity_l1079_107967


namespace empty_set_condition_single_element_condition_single_element_values_l1079_107911

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

-- Theorem for the empty set condition
theorem empty_set_condition (a : ℝ) : A a = ∅ ↔ a > 9/8 := by sorry

-- Theorem for the single element condition
theorem single_element_condition (a : ℝ) : 
  (∃! x, x ∈ A a) ↔ (a = 0 ∨ a = 9/8) := by sorry

-- Theorem for the specific elements when A has a single element
theorem single_element_values (a : ℝ) :
  (∃! x, x ∈ A a) → 
  ((a = 0 → A a = {2/3}) ∧ (a = 9/8 → A a = {4/3})) := by sorry

end empty_set_condition_single_element_condition_single_element_values_l1079_107911


namespace seventh_oblong_number_l1079_107922

/-- Definition of an oblong number -/
def oblong_number (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem: The 7th oblong number is 56 -/
theorem seventh_oblong_number : oblong_number 7 = 56 := by
  sorry

end seventh_oblong_number_l1079_107922


namespace quartic_root_ratio_l1079_107924

theorem quartic_root_ratio (a b c d e : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) →
  d / e = -25 / 12 := by
sorry

end quartic_root_ratio_l1079_107924


namespace cuboid_surface_area_l1079_107905

/-- A cuboid with three distinct side areas -/
structure Cuboid where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ

/-- The total surface area of a cuboid -/
def surface_area (c : Cuboid) : ℝ := 2 * (c.area1 + c.area2 + c.area3)

/-- Theorem: The surface area of a cuboid with side areas 4, 3, and 6 is 26 -/
theorem cuboid_surface_area :
  let c : Cuboid := { area1 := 4, area2 := 3, area3 := 6 }
  surface_area c = 26 := by
  sorry

#check cuboid_surface_area

end cuboid_surface_area_l1079_107905


namespace single_intersection_l1079_107992

/-- The quadratic function representing the first graph -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * x + 3

/-- The linear function representing the second graph -/
def g (x : ℝ) : ℝ := 2 * x + 5

/-- The theorem stating the condition for a single intersection point -/
theorem single_intersection (k : ℝ) : 
  (∃! x, f k x = g x) ↔ k = -1/2 := by sorry

end single_intersection_l1079_107992


namespace odd_prime_square_root_l1079_107997

theorem odd_prime_square_root (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (k : ℕ), k > 0 ∧ ∃ (n : ℕ), n > 0 ∧ k - p * k = n^2 ∧ k = (p + 1)^2 / 4 := by
  sorry

end odd_prime_square_root_l1079_107997


namespace tax_rate_ratio_l1079_107998

theorem tax_rate_ratio (mork_rate mindy_rate combined_rate : ℚ) 
  (h1 : mork_rate = 45/100)
  (h2 : mindy_rate = 15/100)
  (h3 : combined_rate = 21/100) :
  ∃ (m k : ℚ), m > 0 ∧ k > 0 ∧ 
    mindy_rate * m + mork_rate * k = combined_rate * (m + k) ∧
    m / k = 4 := by
  sorry

end tax_rate_ratio_l1079_107998


namespace second_patient_hours_l1079_107929

/-- Represents the psychologist's pricing model and patient charges -/
structure TherapyPricing where
  firstHourCost : ℕ
  additionalHourCost : ℕ
  firstPatientHours : ℕ
  firstPatientCharge : ℕ
  secondPatientCharge : ℕ

/-- 
Given a psychologist's pricing model where:
- The first hour costs $30 more than each additional hour
- A 5-hour therapy session costs $400
- Another therapy session costs $252

This theorem proves that the second therapy session lasted 3 hours.
-/
theorem second_patient_hours (tp : TherapyPricing) 
  (h1 : tp.firstHourCost = tp.additionalHourCost + 30)
  (h2 : tp.firstPatientHours = 5)
  (h3 : tp.firstPatientCharge = 400)
  (h4 : tp.firstPatientCharge = tp.firstHourCost + (tp.firstPatientHours - 1) * tp.additionalHourCost)
  (h5 : tp.secondPatientCharge = 252) : 
  ∃ (h : ℕ), tp.secondPatientCharge = tp.firstHourCost + (h - 1) * tp.additionalHourCost ∧ h = 3 := by
  sorry

end second_patient_hours_l1079_107929


namespace largest_triangle_perimeter_l1079_107912

theorem largest_triangle_perimeter (x : ℤ) : 
  (7 : ℝ) + 11 > (x : ℝ) → 
  (7 : ℝ) + (x : ℝ) > 11 → 
  11 + (x : ℝ) > 7 → 
  (∃ (y : ℤ), (7 : ℝ) + 11 + (y : ℝ) ≥ 7 + 11 + (x : ℝ)) ∧ 
  (7 : ℝ) + 11 + (y : ℝ) ≤ 35 :=
by sorry

end largest_triangle_perimeter_l1079_107912


namespace edward_rides_l1079_107930

def max_rides (initial_tickets : ℕ) (spent_tickets : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  (initial_tickets - spent_tickets) / tickets_per_ride

theorem edward_rides : max_rides 325 115 13 = 16 := by
  sorry

end edward_rides_l1079_107930


namespace prime_sum_squares_l1079_107921

theorem prime_sum_squares (p q m : ℕ) : 
  p.Prime → q.Prime → p ≠ q →
  p^2 - 2001*p + m = 0 →
  q^2 - 2001*q + m = 0 →
  p^2 + q^2 = 3996005 := by
sorry

end prime_sum_squares_l1079_107921


namespace x_value_from_ratios_l1079_107959

theorem x_value_from_ratios (x y z a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : x * y / (x + y) = a)
  (h2 : x * z / (x + z) = b)
  (h3 : y * z / (y + z) = c) :
  x = 2 * a * b * c / (a * c + b * c - a * b) := by
sorry

end x_value_from_ratios_l1079_107959


namespace expansion_properties_l1079_107947

-- Define the expansion of (x-m)^7
def expansion (x m : ℝ) (a : Fin 8 → ℝ) : Prop :=
  (x - m)^7 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7

-- State the theorem
theorem expansion_properties {m : ℝ} {a : Fin 8 → ℝ} 
  (h_expansion : ∀ x, expansion x m a)
  (h_coeff : a 4 = -35) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 1) ∧
  (a 1 + a 3 + a 5 + a 7 = 26) := by
  sorry

end expansion_properties_l1079_107947


namespace micrometer_conversion_l1079_107906

-- Define the conversion factor from micrometers to meters
def micrometer_to_meter : ℝ := 1e-6

-- State the theorem
theorem micrometer_conversion :
  0.01 * micrometer_to_meter = 1e-8 := by
  sorry

end micrometer_conversion_l1079_107906


namespace divisors_of_72_l1079_107949

theorem divisors_of_72 : Finset.card ((Finset.range 73).filter (λ x => 72 % x = 0)) * 2 = 24 := by
  sorry

end divisors_of_72_l1079_107949


namespace mL_to_L_conversion_l1079_107927

-- Define the conversion rate
def mL_per_L : ℝ := 1000

-- Define the volume in milliliters
def volume_mL : ℝ := 27

-- Theorem to prove the conversion
theorem mL_to_L_conversion :
  volume_mL / mL_per_L = 0.027 := by
  sorry

end mL_to_L_conversion_l1079_107927


namespace shop_width_calculation_l1079_107989

/-- Given a shop with specified rent and dimensions, calculate its width -/
theorem shop_width_calculation (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_sqft : ℕ) : 
  monthly_rent = 2244 →
  length = 22 →
  annual_rent_per_sqft = 68 →
  (monthly_rent * 12) / annual_rent_per_sqft / length = 18 := by
  sorry

#check shop_width_calculation

end shop_width_calculation_l1079_107989


namespace inscribed_circle_radius_l1079_107913

theorem inscribed_circle_radius 
  (R : ℝ) 
  (r : ℝ) 
  (h1 : R = 18) 
  (h2 : r = 9) 
  (h3 : r = R / 2) : 
  ∃ x : ℝ, x = 8 ∧ 
    (R - x)^2 - x^2 = (r + x)^2 - x^2 ∧ 
    x > 0 ∧ 
    x < R ∧ 
    x < r := by
  sorry

end inscribed_circle_radius_l1079_107913


namespace min_value_theorem_l1079_107916

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2/a + 1/b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y = 2 → 3*x + y ≥ (7 + 2*Real.sqrt 6)/2) ∧
  ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2/a₀ + 1/b₀ = 2 ∧ 3*a₀ + b₀ = (7 + 2*Real.sqrt 6)/2 :=
by sorry

end min_value_theorem_l1079_107916


namespace decagon_interior_intersections_l1079_107971

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def interior_intersection_points (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- A regular decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_interior_intersections :
  interior_intersection_points decagon_sides = 210 := by
  sorry

end decagon_interior_intersections_l1079_107971


namespace karls_clothing_store_l1079_107918

/-- Karl's clothing store problem -/
theorem karls_clothing_store (tshirt_price : ℝ) (pants_price : ℝ) (skirt_price : ℝ) :
  tshirt_price = 5 →
  pants_price = 4 →
  (2 * tshirt_price + pants_price + 4 * skirt_price + 6 * (tshirt_price / 2) = 53) →
  skirt_price = 6 := by
sorry

end karls_clothing_store_l1079_107918


namespace diagonal_cut_result_l1079_107909

/-- Represents a scarf with areas of different colors -/
structure Scarf where
  white : ℚ
  gray : ℚ
  black : ℚ

/-- The original square scarf -/
def original_scarf : Scarf where
  white := 1/2
  gray := 1/3
  black := 1/6

/-- The first triangular scarf after cutting -/
def first_triangular_scarf : Scarf where
  white := 3/4
  gray := 2/9
  black := 1/36

/-- The second triangular scarf after cutting -/
def second_triangular_scarf : Scarf where
  white := 1/4
  gray := 4/9
  black := 11/36

/-- Theorem stating that cutting the original square scarf diagonally 
    results in the two specified triangular scarves -/
theorem diagonal_cut_result : 
  (original_scarf.white + original_scarf.gray + original_scarf.black = 1) →
  (first_triangular_scarf.white + first_triangular_scarf.gray + first_triangular_scarf.black = 1) ∧
  (second_triangular_scarf.white + second_triangular_scarf.gray + second_triangular_scarf.black = 1) ∧
  (first_triangular_scarf.white = 3/4) ∧
  (first_triangular_scarf.gray = 2/9) ∧
  (first_triangular_scarf.black = 1/36) ∧
  (second_triangular_scarf.white = 1/4) ∧
  (second_triangular_scarf.gray = 4/9) ∧
  (second_triangular_scarf.black = 11/36) := by
  sorry

end diagonal_cut_result_l1079_107909


namespace malcolm_facebook_followers_l1079_107961

/-- Represents the number of followers on different social media platforms -/
structure Followers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ

/-- Calculates the total number of followers across all platforms -/
def totalFollowers (f : Followers) : ℕ :=
  f.instagram + f.facebook + f.twitter + f.tiktok + f.youtube

/-- Theorem stating that given the conditions, Malcolm has 375 followers on Facebook -/
theorem malcolm_facebook_followers :
  ∃ (f : Followers),
    f.instagram = 240 ∧
    f.twitter = (f.instagram + f.facebook) / 2 ∧
    f.tiktok = 3 * f.twitter ∧
    f.youtube = f.tiktok + 510 ∧
    totalFollowers f = 3840 →
    f.facebook = 375 := by
  sorry

end malcolm_facebook_followers_l1079_107961


namespace max_a_value_l1079_107943

theorem max_a_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = a * x^2 - a * x + 1) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) →
  a ≤ 8 ∧ ∃ b : ℝ, b > 8 ∧ ∃ y : ℝ, 0 ≤ y ∧ y ≤ 1 ∧ |b * y^2 - b * y + 1| > 1 :=
by sorry

end max_a_value_l1079_107943


namespace range_of_f_l1079_107965

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (Real.cos x) + Real.arccos (Real.sin x)

theorem range_of_f :
  Set.range f = Set.Icc 0 Real.pi :=
sorry

end range_of_f_l1079_107965


namespace symmetry_wrt_x_axis_l1079_107952

/-- Given a point A(-2, 3) in a Cartesian coordinate system, 
    its symmetrical point with respect to the x-axis has coordinates (-2, -3). -/
theorem symmetry_wrt_x_axis : 
  let A : ℝ × ℝ := (-2, 3)
  let symmetrical_point : ℝ × ℝ := (-2, -3)
  (∀ (x y : ℝ), (x, y) = A → (x, -y) = symmetrical_point) :=
by sorry

end symmetry_wrt_x_axis_l1079_107952


namespace parallel_lines_k_value_l1079_107977

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes (m₁ m₂ : ℝ) : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, (y = m₁ * x + b₁ ↔ y = m₂ * x + b₂)) ↔ m₁ = m₂

/-- The problem statement -/
theorem parallel_lines_k_value :
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, (y = 5 * x - 3 ↔ y = 3 * k * x + 7)) → k = 5 / 3 :=
by sorry

end parallel_lines_k_value_l1079_107977


namespace oranges_per_child_l1079_107996

/-- Given 4 children and 12 oranges in total, prove that each child has 3 oranges. -/
theorem oranges_per_child (num_children : ℕ) (total_oranges : ℕ) 
  (h1 : num_children = 4) (h2 : total_oranges = 12) : 
  total_oranges / num_children = 3 := by
  sorry

end oranges_per_child_l1079_107996


namespace pool_volume_l1079_107972

/-- The volume of a circular pool with linearly varying depth -/
theorem pool_volume (diameter : ℝ) (min_depth max_depth : ℝ) :
  diameter = 20 →
  min_depth = 3 →
  max_depth = 6 →
  let radius := diameter / 2
  let avg_depth := (min_depth + max_depth) / 2
  let volume := π * radius^2 * avg_depth
  volume = 450 * π := by
  sorry

end pool_volume_l1079_107972


namespace max_value_zero_l1079_107975

theorem max_value_zero (a : Real) (h1 : 0 ≤ a) (h2 : a ≤ 1) :
  (∀ x : Real, x ≤ Real.sqrt (a * 1 * 0) + Real.sqrt ((1 - a) * (1 - 1) * (1 - 0))) →
  (Real.sqrt (a * 1 * 0) + Real.sqrt ((1 - a) * (1 - 1) * (1 - 0))) = 0 :=
by sorry

end max_value_zero_l1079_107975


namespace classroom_setup_l1079_107919

/-- Represents the number of desks in a classroom setup for an exam. -/
def num_desks : ℕ := 33

/-- Represents the number of chairs per desk. -/
def chairs_per_desk : ℕ := 4

/-- Represents the number of legs per chair. -/
def legs_per_chair : ℕ := 4

/-- Represents the number of legs per desk. -/
def legs_per_desk : ℕ := 6

/-- Represents the total number of legs from all desks and chairs. -/
def total_legs : ℕ := 728

theorem classroom_setup :
  num_desks * chairs_per_desk * legs_per_chair + num_desks * legs_per_desk = total_legs :=
by sorry

end classroom_setup_l1079_107919
