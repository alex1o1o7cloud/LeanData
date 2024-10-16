import Mathlib

namespace NUMINAMATH_CALUDE_carwash_problem_l2959_295926

theorem carwash_problem (car_price truck_price suv_price : ℕ) 
                        (total_raised : ℕ) 
                        (num_cars num_trucks : ℕ) : 
  car_price = 5 →
  truck_price = 6 →
  suv_price = 7 →
  total_raised = 100 →
  num_cars = 7 →
  num_trucks = 5 →
  ∃ (num_suvs : ℕ), 
    num_suvs * suv_price + num_cars * car_price + num_trucks * truck_price = total_raised ∧
    num_suvs = 5 := by
  sorry

end NUMINAMATH_CALUDE_carwash_problem_l2959_295926


namespace NUMINAMATH_CALUDE_multiple_with_binary_digits_l2959_295912

theorem multiple_with_binary_digits (n : ℕ) : ∃ m : ℕ, 
  (m % n = 0) ∧ 
  (Nat.digits 2 m).length = n ∧ 
  (∀ d ∈ Nat.digits 2 m, d = 0 ∨ d = 1) :=
sorry

end NUMINAMATH_CALUDE_multiple_with_binary_digits_l2959_295912


namespace NUMINAMATH_CALUDE_isosceles_triangle_construction_uniqueness_l2959_295917

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  radius : ℝ
  altitude : ℝ
  orthocenter : ℝ
  is_positive : base > 0 ∧ radius > 0 ∧ altitude > 0
  bisects_altitude : orthocenter = altitude / 2

/-- Theorem stating that an isosceles triangle can be uniquely constructed given the base, radius, and orthocenter condition -/
theorem isosceles_triangle_construction_uniqueness 
  (b r : ℝ) 
  (hb : b > 0) 
  (hr : r > 0) : 
  ∃! t : IsoscelesTriangle, t.base = b ∧ t.radius = r :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_construction_uniqueness_l2959_295917


namespace NUMINAMATH_CALUDE_eighty_seventh_odd_integer_l2959_295976

theorem eighty_seventh_odd_integer : ∀ n : ℕ, n > 0 → (2 * n - 1) = 173 ↔ n = 87 := by
  sorry

end NUMINAMATH_CALUDE_eighty_seventh_odd_integer_l2959_295976


namespace NUMINAMATH_CALUDE_fraction_increase_l2959_295980

theorem fraction_increase (x y : ℝ) (h : x + y ≠ 0) :
  (2 * (2 * x) * (2 * y)) / (2 * x + 2 * y) = 2 * ((2 * x * y) / (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_increase_l2959_295980


namespace NUMINAMATH_CALUDE_pet_store_combinations_l2959_295903

theorem pet_store_combinations (puppies kittens hamsters : ℕ) 
  (h1 : puppies = 20) (h2 : kittens = 9) (h3 : hamsters = 12) :
  (puppies * kittens * hamsters) * 6 = 12960 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l2959_295903


namespace NUMINAMATH_CALUDE_dog_tether_area_l2959_295982

/-- The area outside a regular hexagon that can be reached by a tethered point -/
theorem dog_tether_area (side_length : Real) (rope_length : Real) :
  side_length = 1 ∧ rope_length = 3 →
  let hexagon_area := 3 * Real.sqrt 3 / 2 * side_length^2
  let tether_area := 2 * Real.pi * rope_length^2 / 3 + Real.pi * (rope_length - side_length)^2 / 3
  tether_area - hexagon_area = 22 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_dog_tether_area_l2959_295982


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l2959_295956

/-- Given vectors a and b in ℝ², if a + t*b is perpendicular to a, then t = -5/8 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (t : ℝ) 
  (h1 : a = (1, 2))
  (h2 : b = (2, 3))
  (h3 : (a.1 + t * b.1, a.2 + t * b.2) • a = 0) :
  t = -5/8 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l2959_295956


namespace NUMINAMATH_CALUDE_max_threshold_price_l2959_295945

/-- Represents a company with a product line -/
structure Company where
  num_products : ℕ
  avg_price : ℝ
  min_price : ℝ
  max_price : ℝ
  num_below_threshold : ℕ

/-- The threshold price for a given company -/
def threshold_price (c : Company) : ℝ := sorry

theorem max_threshold_price (c : Company) :
  c.num_products = 25 →
  c.avg_price = 1200 →
  c.min_price = 400 →
  c.max_price = 13200 →
  c.num_below_threshold = 12 →
  threshold_price c ≤ 700 ∧
  ∀ t, t > 700 → ¬(threshold_price c = t) := by
  sorry

#check max_threshold_price

end NUMINAMATH_CALUDE_max_threshold_price_l2959_295945


namespace NUMINAMATH_CALUDE_janets_group_children_count_l2959_295919

theorem janets_group_children_count 
  (total_people : Nat) 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (discount_rate : ℚ) 
  (soda_price : ℚ) 
  (total_paid : ℚ) :
  total_people = 10 ∧ 
  adult_price = 30 ∧ 
  child_price = 15 ∧ 
  discount_rate = 0.8 ∧ 
  soda_price = 5 ∧ 
  total_paid = 197 →
  ∃ (children : Nat),
    children ≤ total_people ∧
    (total_paid - soda_price) = 
      ((adult_price * (total_people - children) + child_price * children) * discount_rate) ∧
    children = 4 := by
  sorry

end NUMINAMATH_CALUDE_janets_group_children_count_l2959_295919


namespace NUMINAMATH_CALUDE_probability_at_least_one_red_l2959_295952

def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def selected_balls : ℕ := 2

theorem probability_at_least_one_red :
  (1 : ℚ) - (Nat.choose white_balls selected_balls : ℚ) / (Nat.choose total_balls selected_balls : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_red_l2959_295952


namespace NUMINAMATH_CALUDE_common_chord_length_is_2_sqrt_5_l2959_295953

/-- Two circles in a 2D plane -/
structure TwoCircles where
  /-- Equation of the first circle: x^2 + y^2 - 2x + 10y - 24 = 0 -/
  circle1 : ℝ → ℝ → Prop
  /-- Equation of the second circle: x^2 + y^2 + 2x + 2y - 8 = 0 -/
  circle2 : ℝ → ℝ → Prop
  /-- The circles intersect -/
  intersect : ∃ x y, circle1 x y ∧ circle2 x y

/-- Definition of the specific two circles from the problem -/
def specificCircles : TwoCircles where
  circle1 := fun x y => x^2 + y^2 - 2*x + 10*y - 24 = 0
  circle2 := fun x y => x^2 + y^2 + 2*x + 2*y - 8 = 0
  intersect := sorry -- We assume the circles intersect as given in the problem

/-- The length of the common chord of two intersecting circles -/
def commonChordLength (c : TwoCircles) : ℝ := sorry

/-- Theorem stating that the length of the common chord is 2√5 -/
theorem common_chord_length_is_2_sqrt_5 :
  commonChordLength specificCircles = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_is_2_sqrt_5_l2959_295953


namespace NUMINAMATH_CALUDE_angle_problem_l2959_295914

theorem angle_problem (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π)  -- θ is in the second quadrant
  (h2 : Real.tan (θ + π/4) = 1/2) :
  Real.tan θ = -1/3 ∧ 
  Real.sin (π/2 - 2*θ) + Real.sin (π + 2*θ) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_problem_l2959_295914


namespace NUMINAMATH_CALUDE_solution_set_circle_plus_l2959_295960

/-- Custom operation ⊕ -/
def circle_plus (a b : ℝ) : ℝ := -2 * a + b

/-- Theorem stating the solution set of x ⊕ 4 > 0 -/
theorem solution_set_circle_plus (x : ℝ) :
  circle_plus x 4 > 0 ↔ x < 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_circle_plus_l2959_295960


namespace NUMINAMATH_CALUDE_f_greater_than_g_l2959_295923

def f (x : ℝ) : ℝ := 3 * x^2 - x + 1

def g (x : ℝ) : ℝ := 2 * x^2 + x - 1

theorem f_greater_than_g : ∀ x : ℝ, f x > g x := by
  sorry

end NUMINAMATH_CALUDE_f_greater_than_g_l2959_295923


namespace NUMINAMATH_CALUDE_parking_spaces_available_l2959_295997

theorem parking_spaces_available (front_spaces back_spaces total_parked : ℕ) 
  (h1 : front_spaces = 52)
  (h2 : back_spaces = 38)
  (h3 : total_parked = 39)
  (h4 : total_parked = front_spaces + back_spaces / 2) : 
  front_spaces + back_spaces - total_parked = 51 := by
  sorry

end NUMINAMATH_CALUDE_parking_spaces_available_l2959_295997


namespace NUMINAMATH_CALUDE_product_less_than_square_l2959_295924

theorem product_less_than_square : 1234567 * 1234569 < 1234568^2 := by
  sorry

end NUMINAMATH_CALUDE_product_less_than_square_l2959_295924


namespace NUMINAMATH_CALUDE_tan_four_greater_than_tan_three_l2959_295966

theorem tan_four_greater_than_tan_three :
  π / 2 < 3 ∧ 3 < π ∧ π < 4 ∧ 4 < 3 * π / 2 →
  Real.tan 4 > Real.tan 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_four_greater_than_tan_three_l2959_295966


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2959_295900

/-- Represents a school population with teachers and students. -/
structure SchoolPopulation where
  teachers : ℕ
  maleStudents : ℕ
  femaleStudents : ℕ

/-- Represents a stratified sample from the school population. -/
structure StratifiedSample where
  totalSize : ℕ
  femalesSampled : ℕ

/-- Theorem: Given the school population and number of females sampled, 
    the total sample size is 192. -/
theorem stratified_sample_size 
  (school : SchoolPopulation)
  (sample : StratifiedSample)
  (h1 : school.teachers = 200)
  (h2 : school.maleStudents = 1200)
  (h3 : school.femaleStudents = 1000)
  (h4 : sample.femalesSampled = 80) :
  sample.totalSize = 192 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l2959_295900


namespace NUMINAMATH_CALUDE_free_throw_probabilities_l2959_295949

/-- The probability of player A scoring a free throw -/
def prob_A : ℚ := 1/2

/-- The probability of player B scoring a free throw -/
def prob_B : ℚ := 2/5

/-- The probability of both A and B scoring their free throws -/
def prob_both_score : ℚ := prob_A * prob_B

/-- The probability of at least one of A or B scoring their free throw -/
def prob_at_least_one_scores : ℚ := 1 - (1 - prob_A) * (1 - prob_B)

theorem free_throw_probabilities :
  (prob_both_score = 1/5) ∧ (prob_at_least_one_scores = 7/10) := by
  sorry

end NUMINAMATH_CALUDE_free_throw_probabilities_l2959_295949


namespace NUMINAMATH_CALUDE_equation_solutions_l2959_295967

theorem equation_solutions :
  (∀ x y z w : ℕ+, x.val + y.val + z.val + w.val = 60 →
    ¬(y.val = x.val + 1 ∧ z.val = y.val + 1 ∧ w.val = z.val + 1)) ∧
  (∃ x y z w : ℕ+, x.val + y.val + z.val + w.val = 60 ∧
    y.val = x.val + 2 ∧ z.val = y.val + 2 ∧ w.val = z.val + 2 ∧
    Even x.val ∧ Even y.val ∧ Even z.val ∧ Even w.val) ∧
  (∀ x y z w : ℕ+, x.val + y.val + z.val + w.val = 60 →
    ¬(y.val = x.val + 2 ∧ z.val = y.val + 2 ∧ w.val = z.val + 2 ∧
      Odd x.val ∧ Odd y.val ∧ Odd z.val ∧ Odd w.val)) ∧
  (∃ x y z w : ℕ+, x.val + y.val + z.val + w.val = 60 ∧
    y.val = x.val + 2 ∧ z.val = y.val + 2 ∧ w.val = z.val + 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2959_295967


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l2959_295965

/-- Given a geometric sequence {aₙ} with a₁ = 3 and a₅ = 75, prove that a₃ = 15 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (h1 : a 1 = 3) (h5 : a 5 = 75) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) : 
  a 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l2959_295965


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l2959_295983

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l2959_295983


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2959_295905

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - (k + 1) * x + 2 * k = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - (k + 1) * y + 2 * k = 0 → y = x) ↔ 
  k = 11 + 10 * Real.sqrt 6 ∨ k = 11 - 10 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2959_295905


namespace NUMINAMATH_CALUDE_first_valid_year_is_2913_l2959_295941

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2100 ∧ sum_of_digits year = 15

theorem first_valid_year_is_2913 :
  (∀ y, 2100 < y ∧ y < 2913 → sum_of_digits y ≠ 15) ∧
  is_valid_year 2913 := by
  sorry

end NUMINAMATH_CALUDE_first_valid_year_is_2913_l2959_295941


namespace NUMINAMATH_CALUDE_lcm_5_7_10_21_l2959_295921

theorem lcm_5_7_10_21 : Nat.lcm 5 (Nat.lcm 7 (Nat.lcm 10 21)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_lcm_5_7_10_21_l2959_295921


namespace NUMINAMATH_CALUDE_bianca_coloring_books_l2959_295916

/-- The number of coloring books Bianca initially had -/
def initial_books : ℕ := 45

/-- The number of books Bianca gave away -/
def books_given_away : ℕ := 6

/-- The number of books Bianca bought -/
def books_bought : ℕ := 20

/-- The total number of books Bianca has after the transactions -/
def final_books : ℕ := 59

/-- Theorem stating that the initial number of books is correct -/
theorem bianca_coloring_books : 
  initial_books - books_given_away + books_bought = final_books := by
  sorry

#check bianca_coloring_books

end NUMINAMATH_CALUDE_bianca_coloring_books_l2959_295916


namespace NUMINAMATH_CALUDE_dice_puzzle_l2959_295913

/-- Given five dice with 21 dots each and 43 visible dots, prove that 62 dots are not visible -/
theorem dice_puzzle (num_dice : ℕ) (dots_per_die : ℕ) (visible_dots : ℕ) : 
  num_dice = 5 → dots_per_die = 21 → visible_dots = 43 → 
  num_dice * dots_per_die - visible_dots = 62 := by
  sorry

end NUMINAMATH_CALUDE_dice_puzzle_l2959_295913


namespace NUMINAMATH_CALUDE_complement_intersection_l2959_295968

def A : Set ℕ := {2, 3, 4}
def B (a : ℕ) : Set ℕ := {a + 2, a}

theorem complement_intersection (a : ℕ) (h : A ∩ B a = B a) : (Aᶜ ∩ B a) = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_l2959_295968


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2959_295978

/-- The line 5x + 12y + a = 0 is tangent to the circle (x-1)^2 + y^2 = 1 if and only if a = -18 or a = 8 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, 5*x + 12*y + a = 0 → (x-1)^2 + y^2 = 1) ↔ (a = -18 ∨ a = 8) := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2959_295978


namespace NUMINAMATH_CALUDE_mans_running_speed_l2959_295998

/-- Proves that given a man who walks at 8 kmph for 4 hours and 45 minutes,
    and runs the same distance in 120 minutes, his running speed is 19 kmph. -/
theorem mans_running_speed
  (walking_speed : ℝ)
  (walking_time_hours : ℝ)
  (walking_time_minutes : ℝ)
  (running_time_minutes : ℝ)
  (h1 : walking_speed = 8)
  (h2 : walking_time_hours = 4)
  (h3 : walking_time_minutes = 45)
  (h4 : running_time_minutes = 120)
  : (walking_speed * (walking_time_hours + walking_time_minutes / 60)) /
    (running_time_minutes / 60) = 19 := by
  sorry


end NUMINAMATH_CALUDE_mans_running_speed_l2959_295998


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_example_l2959_295904

/-- The point symmetric to (x, y) with respect to the x-axis is (x, -y) -/
def symmetricPointXAxis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The statement that the point symmetric to (-1, 2) with respect to the x-axis is (-1, -2) -/
theorem symmetric_point_x_axis_example : symmetricPointXAxis (-1, 2) = (-1, -2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_example_l2959_295904


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2959_295946

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2959_295946


namespace NUMINAMATH_CALUDE_family_weight_ratio_l2959_295925

/-- Given the weights of three generations in a family, prove the ratio of the child's weight to the grandmother's weight -/
theorem family_weight_ratio :
  ∀ (grandmother daughter child : ℝ),
  grandmother + daughter + child = 110 →
  daughter + child = 60 →
  daughter = 50 →
  child / grandmother = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_family_weight_ratio_l2959_295925


namespace NUMINAMATH_CALUDE_mike_weekly_pullups_l2959_295907

/-- The number of pull-ups Mike does each time he enters his office -/
def pullups_per_entry : ℕ := 2

/-- The number of times Mike enters his office per day -/
def office_entries_per_day : ℕ := 5

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pull-ups Mike does in a week -/
def total_pullups_per_week : ℕ := pullups_per_entry * office_entries_per_day * days_in_week

/-- Theorem stating that Mike does 70 pull-ups in a week -/
theorem mike_weekly_pullups : total_pullups_per_week = 70 := by
  sorry

end NUMINAMATH_CALUDE_mike_weekly_pullups_l2959_295907


namespace NUMINAMATH_CALUDE_delta_value_l2959_295975

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ - 3 → Δ = -9 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l2959_295975


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2959_295950

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 11 / 17) 
  (h2 : x - y = 1 / 143) : 
  x^2 - y^2 = 11 / 2431 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2959_295950


namespace NUMINAMATH_CALUDE_greatest_b_value_l2959_295981

theorem greatest_b_value (a b : ℤ) (h : a * b + 7 * a + 6 * b = -6) : 
  ∀ c : ℤ, (∃ d : ℤ, d * c + 7 * d + 6 * c = -6) → c ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_greatest_b_value_l2959_295981


namespace NUMINAMATH_CALUDE_coconut_ratio_l2959_295959

theorem coconut_ratio (paolo_coconuts : ℕ) (dante_sold : ℕ) (dante_remaining : ℕ) :
  paolo_coconuts = 14 →
  dante_sold = 10 →
  dante_remaining = 32 →
  (dante_remaining : ℚ) / paolo_coconuts = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_coconut_ratio_l2959_295959


namespace NUMINAMATH_CALUDE_a2023_coordinates_l2959_295947

def companion_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2) + 1, p.1 + 1)

def sequence_point (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0 => (2, 4)
  | n + 1 => companion_point (sequence_point n)

theorem a2023_coordinates :
  sequence_point 2022 = (-2, -2) :=
sorry

end NUMINAMATH_CALUDE_a2023_coordinates_l2959_295947


namespace NUMINAMATH_CALUDE_distribute_nine_to_three_l2959_295974

/-- The number of ways to distribute n distinct objects into k distinct containers,
    with each container receiving at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 9 distinct objects into 3 distinct containers,
    with each container receiving at least one object, is 504 -/
theorem distribute_nine_to_three : distribute 9 3 = 504 := by sorry

end NUMINAMATH_CALUDE_distribute_nine_to_three_l2959_295974


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2959_295971

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 1/a + 1/b + 1/c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2959_295971


namespace NUMINAMATH_CALUDE_speed_ratio_eddy_freddy_l2959_295940

/-- Proves that the ratio of Eddy's average speed to Freddy's average speed is 34:15 -/
theorem speed_ratio_eddy_freddy :
  let eddy_distance : ℝ := 510  -- km
  let eddy_time : ℝ := 3        -- hours
  let freddy_distance : ℝ := 300  -- km
  let freddy_time : ℝ := 4        -- hours
  let eddy_speed : ℝ := eddy_distance / eddy_time
  let freddy_speed : ℝ := freddy_distance / freddy_time
  (eddy_speed / freddy_speed) = 34 / 15 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_eddy_freddy_l2959_295940


namespace NUMINAMATH_CALUDE_unique_symmetry_center_l2959_295986

/-- A point is symmetric to another point with respect to a center -/
def isSymmetric (A B O : ℝ × ℝ) : Prop :=
  A.1 + B.1 = 2 * O.1 ∧ A.2 + B.2 = 2 * O.2

/-- A point is a symmetry center of a set of points -/
def isSymmetryCenter (O : ℝ × ℝ) (H : Set (ℝ × ℝ)) : Prop :=
  ∀ A ∈ H, ∃ B ∈ H, isSymmetric A B O

theorem unique_symmetry_center (H : Set (ℝ × ℝ)) (hfin : Set.Finite H) :
  ∀ O O' : ℝ × ℝ, isSymmetryCenter O H → isSymmetryCenter O' H → O = O' := by
  sorry

#check unique_symmetry_center

end NUMINAMATH_CALUDE_unique_symmetry_center_l2959_295986


namespace NUMINAMATH_CALUDE_range_of_sum_l2959_295920

theorem range_of_sum (a b : ℝ) :
  (∀ x : ℝ, |x - a| + |x + b| ≥ 3) →
  a + b ∈ Set.Iic (-3) ∪ Set.Ioi 3 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l2959_295920


namespace NUMINAMATH_CALUDE_window_purchase_savings_l2959_295934

/-- Calculates the cost of purchasing windows with a discount after the first five -/
def calculateCost (regularPrice : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  if quantity ≤ 5 then
    regularPrice * quantity
  else
    regularPrice * 5 + (regularPrice - discount) * (quantity - 5)

theorem window_purchase_savings :
  let regularPrice : ℕ := 120
  let discount : ℕ := 20
  let daveWindows : ℕ := 10
  let dougWindows : ℕ := 13
  let daveCost := calculateCost regularPrice discount daveWindows
  let dougCost := calculateCost regularPrice discount dougWindows
  let jointCost := calculateCost regularPrice discount (daveWindows + dougWindows)
  daveCost + dougCost - jointCost = 100 := by sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l2959_295934


namespace NUMINAMATH_CALUDE_weight_of_substance_a_l2959_295942

/-- Given a mixture of substances a and b in the ratio 9:11 with a total weight,
    calculate the weight of substance a in the mixture. -/
theorem weight_of_substance_a (total_weight : ℝ) : 
  total_weight = 58.00000000000001 →
  (9 : ℝ) / (9 + 11) * total_weight = 26.1 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_substance_a_l2959_295942


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2959_295985

/-- The slope of the original line -/
def m₁ : ℚ := 4 / 3

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -3 / 4

/-- The area of the triangle formed by the line and the coordinate axes -/
def A : ℚ := 6

/-- The x-intercept of the perpendicular line -/
def x_intercept : Set ℚ := {4, -4}

theorem perpendicular_line_x_intercept :
  ∀ (C : ℚ), (3 * C / 4) * (C / 3) / 2 = A → C ∈ x_intercept :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2959_295985


namespace NUMINAMATH_CALUDE_pythagorean_sum_and_difference_squares_l2959_295987

theorem pythagorean_sum_and_difference_squares (a b c : ℕ+) 
  (h : c^2 = a^2 + b^2) : 
  ∃ (x y z w : ℕ+), c^2 + a*b = x^2 + y^2 ∧ c^2 - a*b = z^2 + w^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_sum_and_difference_squares_l2959_295987


namespace NUMINAMATH_CALUDE_units_digit_of_F_F8_l2959_295977

def modifiedFibonacci : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | (n + 2) => modifiedFibonacci (n + 1) + modifiedFibonacci n

theorem units_digit_of_F_F8 : 
  (modifiedFibonacci (modifiedFibonacci 8)) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_F_F8_l2959_295977


namespace NUMINAMATH_CALUDE_number_equation_solution_l2959_295937

theorem number_equation_solution : 
  ∃ (N : ℝ), (16/100) * (40/100) * N = 5 * (8/100) * N ∧ N = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2959_295937


namespace NUMINAMATH_CALUDE_intersection_with_complement_of_B_l2959_295992

variable (U A B : Finset ℕ)

theorem intersection_with_complement_of_B (hU : U = {1, 2, 3, 4, 5, 6, 7})
  (hA : A = {3, 4, 5}) (hB : B = {1, 3, 6}) :
  A ∩ (U \ B) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_of_B_l2959_295992


namespace NUMINAMATH_CALUDE_amy_music_files_l2959_295915

theorem amy_music_files :
  ∀ (initial_music_files : ℕ),
    initial_music_files + 21 - 23 = 2 →
    initial_music_files = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_music_files_l2959_295915


namespace NUMINAMATH_CALUDE_girls_count_l2959_295918

/-- The number of boys in the school -/
def num_boys : ℕ := 337

/-- The difference between the number of girls and boys -/
def girl_boy_difference : ℕ := 402

/-- The number of girls in the school -/
def num_girls : ℕ := num_boys + girl_boy_difference

theorem girls_count : num_girls = 739 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_l2959_295918


namespace NUMINAMATH_CALUDE_election_votes_l2959_295994

theorem election_votes (V : ℝ) 
  (h1 : V > 0) -- Ensure total votes is positive
  (h2 : ∃ (x : ℝ), x = 0.25 * V ∧ x + 4000 = V - x) : V = 8000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l2959_295994


namespace NUMINAMATH_CALUDE_permutation_product_difference_divisibility_l2959_295944

theorem permutation_product_difference_divisibility :
  ∀ (a b : Fin 2016 → Fin 2016),
  (Function.Bijective a ∧ Function.Bijective b) →
  ∃ (i j : Fin 2016), i ≠ j ∧ (2017 : ℕ) ∣ (a i * b i - a j * b j) := by
  sorry

end NUMINAMATH_CALUDE_permutation_product_difference_divisibility_l2959_295944


namespace NUMINAMATH_CALUDE_trigonometric_sum_divisibility_l2959_295909

theorem trigonometric_sum_divisibility (n : ℕ) :
  ∃ k : ℤ, (2 * Real.sin (π / 7 : ℝ))^(2*n) + 
           (2 * Real.sin (2*π / 7 : ℝ))^(2*n) + 
           (2 * Real.sin (3*π / 7 : ℝ))^(2*n) = 
           k * (7 : ℝ)^(Int.floor (n / 3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_divisibility_l2959_295909


namespace NUMINAMATH_CALUDE_unique_solution_l2959_295929

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ :=
  x - (floor x : ℝ)

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  2 * (floor x : ℝ) * frac x = x^2 - 3/2 * x - 11/16

theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 9/4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2959_295929


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2959_295932

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 13

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | polynomial b₂ b₁ x = 0} = {-13, -1, 1, 13} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2959_295932


namespace NUMINAMATH_CALUDE_merchant_discount_l2959_295962

theorem merchant_discount (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * 1.2
  let final_price := increased_price * 0.8
  let actual_discount := (original_price - final_price) / original_price
  actual_discount = 0.04 := by
sorry

end NUMINAMATH_CALUDE_merchant_discount_l2959_295962


namespace NUMINAMATH_CALUDE_max_geometric_mean_of_sequence_l2959_295939

theorem max_geometric_mean_of_sequence (A : ℝ) (a : Fin 6 → ℝ) :
  (∃ i, a i = 1) →
  (∀ i, i < 4 → (a i + a (i + 1) + a (i + 2)) / 3 = (a (i + 1) + a (i + 2) + a (i + 3)) / 3) →
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5) / 6 = A →
  (∃ i, i < 4 → 
    ∀ j, j < 4 → 
      (a j * a (j + 1) * a (j + 2)) ^ (1/3 : ℝ) ≤ ((3 * A - 1) ^ 2 / 4) ^ (1/3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_max_geometric_mean_of_sequence_l2959_295939


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2959_295906

theorem arithmetic_expression_equality : 76 + (144 / 12) + (15 * 19)^2 - 350 - (270 / 6) = 80918 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2959_295906


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l2959_295943

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁ ∧ a₁ ≠ 0 ∧ a₂ ≠ 0

/-- Definition of line l₁ -/
def line_l₁ (m : ℝ) (x y : ℝ) : Prop :=
  x + (1 + m) * y = 2 - m

/-- Definition of line l₂ -/
def line_l₂ (m : ℝ) (x y : ℝ) : Prop :=
  2 * m * x + 4 * y = -16

theorem parallel_lines_m_value :
  ∀ m : ℝ, (parallel_lines 1 (1 + m) (m - 2) (2 * m) 4 16) → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l2959_295943


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l2959_295989

/-- Calculates the total cost for a group at a restaurant given specific pricing and group composition. -/
theorem restaurant_bill_calculation 
  (adult_meal_cost : ℚ)
  (kid_meal_cost : ℚ)
  (adult_drink_cost : ℚ)
  (kid_drink_cost : ℚ)
  (dessert_cost : ℚ)
  (total_people : ℕ)
  (num_adults : ℕ)
  (num_children : ℕ)
  (h1 : adult_meal_cost = 12)
  (h2 : kid_meal_cost = 0)
  (h3 : adult_drink_cost = 5/2)
  (h4 : kid_drink_cost = 3/2)
  (h5 : dessert_cost = 4)
  (h6 : total_people = 11)
  (h7 : num_adults = 7)
  (h8 : num_children = 4)
  (h9 : total_people = num_adults + num_children) :
  (num_adults * adult_meal_cost) +
  (num_adults * adult_drink_cost) +
  (num_children * kid_drink_cost) +
  (total_people * dessert_cost) = 151.5 := by
    sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l2959_295989


namespace NUMINAMATH_CALUDE_largest_even_digit_number_with_four_proof_largest_even_digit_number_with_four_l2959_295938

def is_even_digit (d : ℕ) : Prop := d % 2 = 0 ∧ d < 10

def all_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

def contains_digit (n d : ℕ) : Prop := d ∈ n.digits 10

theorem largest_even_digit_number_with_four (n : ℕ) : Prop :=
  n = 5408 ∧
  all_even_digits n ∧
  contains_digit n 4 ∧
  n < 6000 ∧
  n % 8 = 0 ∧
  ∀ m : ℕ, m ≠ n →
    (all_even_digits m ∧ contains_digit m 4 ∧ m < 6000 ∧ m % 8 = 0) →
    m < n

theorem proof_largest_even_digit_number_with_four :
  ∃ n : ℕ, largest_even_digit_number_with_four n :=
sorry

end NUMINAMATH_CALUDE_largest_even_digit_number_with_four_proof_largest_even_digit_number_with_four_l2959_295938


namespace NUMINAMATH_CALUDE_circle_equation_l2959_295922

-- Define the circle C
def circle_C (a r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = r^2}

-- Define the line 2x - y = 0
def line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - p.2 = 0}

theorem circle_equation : 
  ∃ (a r : ℝ), 
    a > 0 ∧ 
    (0, Real.sqrt 5) ∈ circle_C a r ∧ 
    (abs (2 * a) / Real.sqrt 5 = 4 * Real.sqrt 5 / 5) ∧
    circle_C a r = circle_C 2 3 :=
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2959_295922


namespace NUMINAMATH_CALUDE_arccos_of_one_eq_zero_l2959_295972

theorem arccos_of_one_eq_zero : Real.arccos 1 = 0 := by sorry

end NUMINAMATH_CALUDE_arccos_of_one_eq_zero_l2959_295972


namespace NUMINAMATH_CALUDE_function_domain_l2959_295984

/-- The domain of the function y = ln(x+1) / sqrt(-x^2 - 3x + 4) -/
theorem function_domain (x : ℝ) : 
  (x + 1 > 0 ∧ -x^2 - 3*x + 4 > 0) ↔ -1 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_function_domain_l2959_295984


namespace NUMINAMATH_CALUDE_fraction_equality_l2959_295901

theorem fraction_equality : (1^4 + 2009^4 + 2010^4) / (1^2 + 2009^2 + 2010^2) = 4038091 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2959_295901


namespace NUMINAMATH_CALUDE_cube_root_problem_l2959_295902

theorem cube_root_problem :
  ∀ (a b : ℤ) (c : ℚ),
  (5 * a - 2 : ℚ) = -27 →
  b = ⌊Real.sqrt 22⌋ →
  c = -(4 / 25 : ℚ).sqrt →
  a = -5 ∧
  b = 4 ∧
  c = -2/5 ∧
  Real.sqrt (4 * (a : ℚ) * c + 7 * (b : ℚ)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_problem_l2959_295902


namespace NUMINAMATH_CALUDE_sin_cos_sum_20_40_l2959_295930

theorem sin_cos_sum_20_40 :
  Real.sin (20 * π / 180) * Real.cos (40 * π / 180) +
  Real.cos (20 * π / 180) * Real.sin (40 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_20_40_l2959_295930


namespace NUMINAMATH_CALUDE_cos_675_degrees_l2959_295948

theorem cos_675_degrees : Real.cos (675 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_675_degrees_l2959_295948


namespace NUMINAMATH_CALUDE_smallest_egg_count_l2959_295928

/-- Represents the number of eggs in a container -/
def ContainerCapacity : ℕ := 15

/-- Represents the number of partially filled containers -/
def PartialContainers : ℕ := 3

/-- Represents the number of eggs in a partially filled container -/
def PartialContainerContent : ℕ := 14

/-- Calculates the number of eggs given the number of containers -/
def eggCount (containers : ℕ) : ℕ :=
  containers * ContainerCapacity - PartialContainers * (ContainerCapacity - PartialContainerContent)

theorem smallest_egg_count :
  ∃ (n : ℕ), (∀ m : ℕ, eggCount m > 200 → n ≤ m) ∧ eggCount n = 207 := by
  sorry

end NUMINAMATH_CALUDE_smallest_egg_count_l2959_295928


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2959_295970

/-- A line with slope 4 passing through (2, -1) has m + b = -5 -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
    (m = 4) →  -- Given slope
    (-1 = 4 * 2 + b) →  -- Line passes through (2, -1)
    (m + b = -5) :=  -- Conclusion to prove
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2959_295970


namespace NUMINAMATH_CALUDE_spatial_relations_l2959_295973

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular parallel subset : ∀ {T U : Type}, T → U → Prop)

-- Define the given conditions
variable (α β γ : Plane)
variable (m n : Line)
variable (h1 : m ≠ n)

-- Define the main theorem
theorem spatial_relations :
  (∀ α β γ : Plane, perpendicular α β → parallel α γ ∧ perpendicular α γ) →
  ((parallel m n ∧ subset n α) → parallel m α) ∧
  ((perpendicular m α ∧ parallel n α) → perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_spatial_relations_l2959_295973


namespace NUMINAMATH_CALUDE_c_invests_after_eight_months_l2959_295964

/-- Represents the investment scenario of three partners A, B, and C -/
structure Investment where
  /-- A's initial investment amount -/
  a_amount : ℝ
  /-- Number of months after which C invests -/
  c_invest_time : ℝ
  /-- Total annual gain -/
  total_gain : ℝ
  /-- A's share of the profit -/
  a_share : ℝ
  /-- B invests double A's amount after 6 months -/
  b_amount_eq : a_amount * 2 = a_amount
  /-- C invests triple A's amount -/
  c_amount_eq : a_amount * 3 = a_amount
  /-- Total annual gain is Rs. 18600 -/
  total_gain_eq : total_gain = 18600
  /-- A's share is Rs. 6200 -/
  a_share_eq : a_share = 6200
  /-- Profit share is proportional to investment and time -/
  profit_share_prop : a_share / total_gain = 
    (a_amount * 12) / (a_amount * 12 + a_amount * 2 * 6 + a_amount * 3 * (12 - c_invest_time))

/-- Theorem stating that C invests after 8 months -/
theorem c_invests_after_eight_months (i : Investment) : i.c_invest_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_c_invests_after_eight_months_l2959_295964


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2959_295911

theorem sum_of_squares_and_square_of_sum : (3 + 9)^2 + (3^2 + 9^2) = 234 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2959_295911


namespace NUMINAMATH_CALUDE_percentage_in_70to79_is_25_percent_l2959_295961

/-- Represents the score ranges in Ms. Hernandez's biology class -/
inductive ScoreRange
  | Above90
  | Range80to89
  | Range70to79
  | Range60to69
  | Below60

/-- The frequency of students in each score range -/
def frequency (range : ScoreRange) : ℕ :=
  match range with
  | ScoreRange.Above90 => 5
  | ScoreRange.Range80to89 => 9
  | ScoreRange.Range70to79 => 7
  | ScoreRange.Range60to69 => 4
  | ScoreRange.Below60 => 3

/-- The total number of students in the class -/
def totalStudents : ℕ := 
  frequency ScoreRange.Above90 +
  frequency ScoreRange.Range80to89 +
  frequency ScoreRange.Range70to79 +
  frequency ScoreRange.Range60to69 +
  frequency ScoreRange.Below60

/-- The percentage of students who scored in the 70%-79% range -/
def percentageIn70to79Range : ℚ :=
  (frequency ScoreRange.Range70to79 : ℚ) / (totalStudents : ℚ) * 100

theorem percentage_in_70to79_is_25_percent :
  percentageIn70to79Range = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_in_70to79_is_25_percent_l2959_295961


namespace NUMINAMATH_CALUDE_min_value_cyclic_fraction_min_value_cyclic_fraction_achievable_l2959_295936

theorem min_value_cyclic_fraction (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a / b + b / c + c / d + d / a ≥ 4 :=
by sorry

theorem min_value_cyclic_fraction_achievable :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a / b + b / c + c / d + d / a = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_cyclic_fraction_min_value_cyclic_fraction_achievable_l2959_295936


namespace NUMINAMATH_CALUDE_legs_exceed_twice_heads_by_30_l2959_295951

/-- Represents the number of ducks in the group -/
def num_ducks : ℕ := sorry

/-- Represents the number of cows in the group -/
def num_cows : ℕ := 15

/-- Calculates the total number of legs in the group -/
def total_legs : ℕ := 2 * num_ducks + 4 * num_cows

/-- Calculates the total number of heads in the group -/
def total_heads : ℕ := num_ducks + num_cows

/-- Theorem stating that the number of legs exceeds twice the number of heads by 30 -/
theorem legs_exceed_twice_heads_by_30 : total_legs = 2 * total_heads + 30 := by
  sorry

end NUMINAMATH_CALUDE_legs_exceed_twice_heads_by_30_l2959_295951


namespace NUMINAMATH_CALUDE_prob_same_color_is_45_128_l2959_295963

def blue_chips : ℕ := 7
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4
def total_chips : ℕ := blue_chips + red_chips + yellow_chips

def prob_same_color : ℚ :=
  (blue_chips^2 + red_chips^2 + yellow_chips^2) / total_chips^2

theorem prob_same_color_is_45_128 : prob_same_color = 45 / 128 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_45_128_l2959_295963


namespace NUMINAMATH_CALUDE_group_size_calculation_l2959_295979

theorem group_size_calculation (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 2.5 ∧ old_weight = 65 ∧ new_weight = 90 →
  ∃ n : ℕ, n = 10 ∧ n * average_increase = new_weight - old_weight :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l2959_295979


namespace NUMINAMATH_CALUDE_largest_integer_with_four_digit_square_base8_l2959_295954

/-- The largest integer whose square has exactly 4 digits in base 8 -/
def M : ℕ := 31

/-- Conversion of M to base 8 -/
def M_base8 : ℕ := 37

theorem largest_integer_with_four_digit_square_base8 :
  (∀ n : ℕ, n > M → ¬(8^3 ≤ n^2 ∧ n^2 < 8^4)) ∧
  (8^3 ≤ M^2 ∧ M^2 < 8^4) ∧
  M_base8 = M := by sorry

end NUMINAMATH_CALUDE_largest_integer_with_four_digit_square_base8_l2959_295954


namespace NUMINAMATH_CALUDE_inequality_proof_l2959_295969

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z) / (x^3 + y^3 + x * y * z) +
  (x * y * z) / (y^3 + z^3 + x * y * z) +
  (x * y * z) / (z^3 + x^3 + x * y * z) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2959_295969


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2959_295990

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 4 = 24 →
  a 6 = 38 →
  a 3 + a 5 = 48 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2959_295990


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2959_295955

/-- Given two arithmetic sequences {a_n} and {b_n} with S_n and T_n as the sum of their first n terms respectively -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2 ∧ T n = (n * (b 1 + b n)) / 2

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ) (S T : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequences a b S T)
  (h_ratio : ∀ n, S n / T n = (7 * n + 1) / (n + 3)) :
  (a 2 + a 5 + a 17 + a 22) / (b 8 + b 10 + b 12 + b 16) = 31 / 5 ∧
  a 5 / b 5 = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2959_295955


namespace NUMINAMATH_CALUDE_min_value_theorem_l2959_295991

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' = 2 → 1 / x' - y' ≥ 2 * Real.sqrt 2 - 2) ∧
  (1 / (Real.sqrt 2 / 2) - (2 - Real.sqrt 2) = 2 * Real.sqrt 2 - 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2959_295991


namespace NUMINAMATH_CALUDE_rectangle_grid_ratio_l2959_295935

/-- Given a 3x2 grid of identical rectangles with height h and width w,
    and a line segment PQ intersecting the grid as described,
    prove that h/w = 3/8 -/
theorem rectangle_grid_ratio (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0) : 
  let grid_width := 3 * w
  let grid_height := 2 * h
  ∃ (X Y Z : ℝ × ℝ),
    X.1 ∈ Set.Icc 0 grid_width ∧
    X.2 ∈ Set.Icc 0 grid_height ∧
    Z.1 ∈ Set.Icc 0 grid_width ∧
    Z.2 ∈ Set.Icc 0 grid_height ∧
    Y.1 = X.1 ∧
    Y.2 = Z.2 ∧
    (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = (Y.1 - X.1)^2 + (Y.2 - X.2)^2 + (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 ∧
    (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 = 4 * ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) →
  h / w = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_grid_ratio_l2959_295935


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2959_295993

theorem imaginary_part_of_z (m : ℝ) (z : ℂ) : 
  z = 1 - m * I ∧ z = -2 * I → z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2959_295993


namespace NUMINAMATH_CALUDE_august_calculator_problem_l2959_295958

theorem august_calculator_problem (a b c : ℕ) : 
  a = 600 →
  b = 2 * a →
  c = a + b - 400 →
  a + b + c = 3200 :=
by sorry

end NUMINAMATH_CALUDE_august_calculator_problem_l2959_295958


namespace NUMINAMATH_CALUDE_briannas_books_l2959_295999

/-- Brianna's book reading problem -/
theorem briannas_books :
  let books_per_year : ℕ := 24
  let gift_books : ℕ := 6
  let old_books : ℕ := 4
  let bought_books : ℕ := x
  let borrowed_books : ℕ := x - 2

  gift_books + bought_books + borrowed_books + old_books = books_per_year →
  bought_books = 8 := by
  sorry

end NUMINAMATH_CALUDE_briannas_books_l2959_295999


namespace NUMINAMATH_CALUDE_rabbit_weight_l2959_295933

/-- Given the weights of a rabbit and two guinea pigs satisfying certain conditions,
    prove that the rabbit weighs 5 pounds. -/
theorem rabbit_weight (a b c : ℝ) 
  (total_weight : a + b + c = 30)
  (larger_smaller : a + c = 2 * b)
  (rabbit_smaller : a + b = c) : 
  a = 5 := by sorry

end NUMINAMATH_CALUDE_rabbit_weight_l2959_295933


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l2959_295910

/-- Given a quadrilateral with one diagonal of 50 cm, one offset of 8 cm, and an area of 450 cm²,
    the length of the other offset is 10 cm. -/
theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 50 ∧ offset1 = 8 ∧ area = 450 →
  ∃ offset2 : ℝ, offset2 = 10 ∧ area = (diagonal * (offset1 + offset2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_offset_l2959_295910


namespace NUMINAMATH_CALUDE_expression_evaluation_l2959_295927

theorem expression_evaluation : 
  (2019^3 - 3 * 2019^2 * 2020 + 3 * 2019 * 2020^2 - 2020^3 + 6) / (2019 * 2020) = 5 / (2019 * 2020) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2959_295927


namespace NUMINAMATH_CALUDE_book_sales_calculation_l2959_295996

/-- Calculates the total book sales over three days given specific sales patterns. -/
theorem book_sales_calculation (day1_sales : ℕ) : 
  day1_sales = 15 →
  (day1_sales + 3 * day1_sales + (3 * day1_sales) / 5 : ℕ) = 69 := by
  sorry

end NUMINAMATH_CALUDE_book_sales_calculation_l2959_295996


namespace NUMINAMATH_CALUDE_smallest_multiple_of_5_and_24_l2959_295908

theorem smallest_multiple_of_5_and_24 : ∃ n : ℕ, n > 0 ∧ n % 5 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, m > 0 → m % 5 = 0 → m % 24 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_5_and_24_l2959_295908


namespace NUMINAMATH_CALUDE_second_worker_de_time_l2959_295988

/-- Represents a worker paving a path -/
structure Worker where
  speed : ℝ
  distance : ℝ

/-- Represents the paving scenario -/
structure PavingScenario where
  worker1 : Worker
  worker2 : Worker
  totalTime : ℝ

/-- The theorem statement -/
theorem second_worker_de_time (scenario : PavingScenario) : 
  scenario.worker1.speed > 0 ∧ 
  scenario.worker2.speed = 1.2 * scenario.worker1.speed ∧
  scenario.totalTime = 9 ∧
  scenario.worker1.distance * scenario.worker2.speed = scenario.worker2.distance * scenario.worker1.speed →
  ∃ (de_time : ℝ), de_time = 45 ∧ de_time = (scenario.totalTime * 60) / 12 :=
by sorry

end NUMINAMATH_CALUDE_second_worker_de_time_l2959_295988


namespace NUMINAMATH_CALUDE_parabola_vertex_equation_l2959_295931

/-- A parabola with vertex coordinates (-2, 0) is represented by the equation y = (x+2)^2 -/
theorem parabola_vertex_equation :
  ∀ (x y : ℝ), (∃ (a : ℝ), y = a * (x + 2)^2) ↔ 
  (y = (x + 2)^2 ∧ (∀ (x₀ y₀ : ℝ), y₀ = (x₀ + 2)^2 → y₀ ≥ 0 ∧ (y₀ = 0 → x₀ = -2))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_equation_l2959_295931


namespace NUMINAMATH_CALUDE_airplane_seats_total_l2959_295995

/-- Represents the number of seats in an airplane -/
def AirplaneSeats (total : ℝ) : Prop :=
  let first_class : ℝ := 36
  let business_class : ℝ := 0.3 * total
  let economy : ℝ := 0.6 * total
  let premium_economy : ℝ := total - first_class - business_class - economy
  (first_class + business_class + economy + premium_economy = total) ∧
  (premium_economy ≥ 0)

/-- The total number of seats in the airplane is 360 -/
theorem airplane_seats_total : ∃ (total : ℝ), AirplaneSeats total ∧ total = 360 := by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_total_l2959_295995


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2959_295957

/-- A geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ := fun n ↦ a₁ * q^(n - 1)

/-- The sequence is increasing -/
def is_increasing (a : ℕ → ℝ) : Prop := ∀ n, a n < a (n + 1)

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geom : ∃ a₁ q, a = geometric_sequence a₁ q) 
  (h_a₁ : a 1 = -2)
  (h_inc : is_increasing a)
  (h_eq : ∀ n, 3 * (a n + a (n + 2)) = 10 * a (n + 1)) :
  ∃ q, a = geometric_sequence (-2) q ∧ q = 1/3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2959_295957
