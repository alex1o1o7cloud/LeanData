import Mathlib

namespace NUMINAMATH_CALUDE_table_tennis_tournament_l1078_107864

theorem table_tennis_tournament (x : ℕ) :
  let sixth_graders := 2 * x
  let seventh_graders := x
  let total_participants := sixth_graders + seventh_graders
  let total_matches := total_participants * (total_participants - 1) / 2
  let matches_between_grades := sixth_graders * seventh_graders
  let matches_among_sixth := sixth_graders * (sixth_graders - 1) / 2
  let matches_among_seventh := seventh_graders * (seventh_graders - 1) / 2
  let matches_won_by_sixth := matches_among_sixth + matches_between_grades / 2
  let matches_won_by_seventh := matches_among_seventh + matches_between_grades / 2
  matches_won_by_seventh = (matches_won_by_sixth * 14) / 10 →
  total_participants = 9 :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_tournament_l1078_107864


namespace NUMINAMATH_CALUDE_max_value_with_remainder_l1078_107820

theorem max_value_with_remainder (A B : ℕ) (h1 : A ≠ B) (h2 : A = 17 * 25 + B) : 
  (∀ C : ℕ, C < 17 → B ≤ C) → A = 441 :=
by sorry

end NUMINAMATH_CALUDE_max_value_with_remainder_l1078_107820


namespace NUMINAMATH_CALUDE_valid_course_combinations_l1078_107890

/-- The number of courses available to choose from -/
def total_courses : ℕ := 6

/-- The number of courses to be chosen -/
def courses_to_choose : ℕ := 2

/-- The number of pairs of courses that cannot be chosen together -/
def restricted_pairs : ℕ := 2

/-- Calculates the number of combinations of n choose k -/
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The theorem stating the number of valid course combinations -/
theorem valid_course_combinations :
  combinations total_courses courses_to_choose - restricted_pairs = 13 := by
  sorry

end NUMINAMATH_CALUDE_valid_course_combinations_l1078_107890


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l1078_107897

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l1078_107897


namespace NUMINAMATH_CALUDE_ellipse_properties_l1078_107816

-- Define the ellipse
def ellipse (x y m : ℝ) : Prop := x^2 / 25 + y^2 / m^2 = 1

-- Define the focus
def left_focus (x y : ℝ) : Prop := x = -4 ∧ y = 0

-- Define eccentricity
def eccentricity (e : ℝ) (a c : ℝ) : Prop := e = c / a

theorem ellipse_properties (m : ℝ) (h : m > 0) :
  (∃ x y, ellipse x y m ∧ left_focus x y) →
  m = 3 ∧ ∃ e, eccentricity e 5 4 ∧ e = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1078_107816


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1078_107842

/-- Quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-value of a quadratic function at a given x -/
def evaluate (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex_x (f : QuadraticFunction) : ℚ :=
  -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
def vertex_y (f : QuadraticFunction) : ℚ :=
  evaluate f (vertex_x f)

theorem quadratic_coefficient (f : QuadraticFunction) 
  (h1 : vertex_x f = 2)
  (h2 : vertex_y f = 5)
  (h3 : evaluate f 3 = 4) :
  f.a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1078_107842


namespace NUMINAMATH_CALUDE_painted_cubes_count_l1078_107854

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  red_faces : Nat
  blue_faces : Nat

/-- Calculates the number of painted unit cubes in a PaintedCube -/
def num_painted_cubes (cube : PaintedCube) : Nat :=
  cube.size ^ 3 - (cube.size - 2) ^ 3

/-- Theorem: In a 5x5x5 cube with 2 red faces and 4 blue faces, 101 unit cubes are painted -/
theorem painted_cubes_count (cube : PaintedCube) 
  (h_size : cube.size = 5)
  (h_red : cube.red_faces = 2)
  (h_blue : cube.blue_faces = 4) :
  num_painted_cubes cube = 101 := by
  sorry

#check painted_cubes_count

end NUMINAMATH_CALUDE_painted_cubes_count_l1078_107854


namespace NUMINAMATH_CALUDE_remainder_problem_l1078_107835

theorem remainder_problem (divisor : ℕ) (a b : ℕ) (rem_a rem_sum : ℕ) 
  (h_divisor : divisor = 13)
  (h_a : a = 242)
  (h_b : b = 698)
  (h_rem_a : a % divisor = rem_a)
  (h_rem_a_val : rem_a = 8)
  (h_rem_sum : (a + b) % divisor = rem_sum)
  (h_rem_sum_val : rem_sum = 4) :
  b % divisor = 9 := by
  sorry


end NUMINAMATH_CALUDE_remainder_problem_l1078_107835


namespace NUMINAMATH_CALUDE_fred_change_is_correct_l1078_107824

/-- The change Fred received after paying for movie tickets and borrowing a movie --/
def fred_change : ℝ :=
  let ticket_price : ℝ := 5.92
  let num_tickets : ℕ := 2
  let borrowed_movie_cost : ℝ := 6.79
  let payment : ℝ := 20
  let total_cost : ℝ := ticket_price * num_tickets + borrowed_movie_cost
  payment - total_cost

/-- Theorem stating that Fred's change is $1.37 --/
theorem fred_change_is_correct : fred_change = 1.37 := by
  sorry

end NUMINAMATH_CALUDE_fred_change_is_correct_l1078_107824


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1078_107804

theorem fraction_equation_solution (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  (2 / x) - (1 / y) = (3 / z) → z = (2 * y - x) / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1078_107804


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1078_107858

theorem rectangle_dimensions (area perimeter long_side : ℝ) 
  (h_area : area = 300)
  (h_perimeter : perimeter = 70)
  (h_long_side : long_side = 20) : 
  ∃ (width : ℝ), 
    area = long_side * width ∧ 
    perimeter = 2 * (long_side + width) ∧
    width = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1078_107858


namespace NUMINAMATH_CALUDE_equation_solutions_l1078_107847

theorem equation_solutions :
  (∃ (s₁ s₂ : Set ℝ),
    s₁ = {x : ℝ | (5 - 2*x)^2 - 16 = 0} ∧
    s₂ = {x : ℝ | 2*(x - 3) = x^2 - 9} ∧
    s₁ = {1/2, 9/2} ∧
    s₂ = {3, -1}) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1078_107847


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1078_107819

theorem quadratic_factorization (k : ℝ) : 
  (∀ x, x^2 - 8*x + 3 = 0 ↔ (x - 4)^2 = k) → k = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1078_107819


namespace NUMINAMATH_CALUDE_scooter_price_calculation_l1078_107802

/-- Calculates the selling price of a scooter given its purchase price, repair costs, and gain percentage. -/
def scooter_selling_price (purchase_price repair_costs : ℚ) (gain_percent : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_costs
  let gain := gain_percent / 100
  let profit := total_cost * gain
  total_cost + profit

/-- Theorem stating that the selling price of the scooter is $5800 given the specified conditions. -/
theorem scooter_price_calculation :
  scooter_selling_price 4700 800 (5454545454545454 / 100000000000000) = 5800 := by
  sorry

end NUMINAMATH_CALUDE_scooter_price_calculation_l1078_107802


namespace NUMINAMATH_CALUDE_vanya_age_l1078_107807

/-- Represents the ages of Vanya, his father, and Seryozha -/
structure Ages where
  vanya : ℕ
  father : ℕ
  seryozha : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.father = 3 * ages.vanya ∧
  ages.vanya = 3 * ages.seryozha ∧
  ages.father = ages.seryozha + 40

/-- Theorem stating that if the conditions are satisfied, Vanya's age is 15 -/
theorem vanya_age (ages : Ages) : satisfiesConditions ages → ages.vanya = 15 := by
  sorry

#check vanya_age

end NUMINAMATH_CALUDE_vanya_age_l1078_107807


namespace NUMINAMATH_CALUDE_farm_sheep_count_l1078_107836

/-- Given a farm with sheep and horses, prove that the number of sheep is 16 -/
theorem farm_sheep_count (sheep horses : ℕ) (horse_food_per_day total_horse_food : ℕ) : 
  (sheep : ℚ) / horses = 2 / 7 →
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  horses * horse_food_per_day = total_horse_food →
  sheep = 16 := by
sorry

end NUMINAMATH_CALUDE_farm_sheep_count_l1078_107836


namespace NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l1078_107845

theorem square_difference_divided (a b : ℕ) (h : a > b) : 
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution : (725^2 - 675^2) / 25 = 2800 :=
by 
  have h : 725 > 675 := by sorry
  have key := square_difference_divided 725 675 h
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l1078_107845


namespace NUMINAMATH_CALUDE_largest_non_factor_product_of_100_l1078_107811

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem largest_non_factor_product_of_100 :
  ∀ x y : ℕ,
    x ≠ y →
    x > 0 →
    y > 0 →
    is_factor x 100 →
    is_factor y 100 →
    ¬ is_factor (x * y) 100 →
    x * y ≤ 40 :=
by sorry

end NUMINAMATH_CALUDE_largest_non_factor_product_of_100_l1078_107811


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_quadratic_inequality_parameter_range_l1078_107822

-- Problem 1
theorem quadratic_inequality_solution_sets (a c : ℝ) :
  (∀ x : ℝ, ax^2 + 2*x + c > 0 ↔ -1/3 < x ∧ x < 1/2) →
  (∀ x : ℝ, c*x^2 - 2*x + a < 0 ↔ -2 < x ∧ x < 3) :=
sorry

-- Problem 2
theorem quadratic_inequality_parameter_range (m : ℝ) :
  (∀ x : ℝ, x > 0 → x^2 - m*x + 4 > 0) ↔ m < 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_quadratic_inequality_parameter_range_l1078_107822


namespace NUMINAMATH_CALUDE_race_distance_l1078_107851

theorem race_distance (a : ℝ) (r : ℝ) (S_n : ℝ) (n : ℕ) :
  a = 10 ∧ r = 2 ∧ S_n = 310 ∧ S_n = a * (r^n - 1) / (r - 1) →
  2^n = 32 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l1078_107851


namespace NUMINAMATH_CALUDE_apple_cost_price_l1078_107803

theorem apple_cost_price (selling_price : ℚ) (loss_fraction : ℚ) : 
  selling_price = 16 → loss_fraction = 1/6 → 
  ∃ cost_price : ℚ, 
    selling_price = cost_price - loss_fraction * cost_price ∧ 
    cost_price = 19.2 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_price_l1078_107803


namespace NUMINAMATH_CALUDE_donut_selection_count_l1078_107817

theorem donut_selection_count : Nat := by
  -- Define the number of donuts and types
  let n : Nat := 5  -- number of donuts to select
  let k : Nat := 4  -- number of types of donuts

  -- Define the function to calculate combinations
  let combinations (n m : Nat) : Nat := (Nat.factorial n) / ((Nat.factorial m) * (Nat.factorial (n - m)))

  -- Calculate the number of selections using the stars and bars theorem
  let result := combinations (n + k - 1) (k - 1)

  -- Assert that the result is 56
  have h : result = 56 := by sorry

  -- Return the result
  exact 56

end NUMINAMATH_CALUDE_donut_selection_count_l1078_107817


namespace NUMINAMATH_CALUDE_max_value_of_x_l1078_107829

theorem max_value_of_x (x : ℕ) : 
  x > 0 ∧ 
  ∃ k : ℕ, x = 4 * k ∧ 
  x^3 < 1728 →
  x ≤ 8 ∧ ∃ y : ℕ, y > 0 ∧ ∃ m : ℕ, y = 4 * m ∧ y^3 < 1728 ∧ y = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_l1078_107829


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1078_107889

theorem inequality_and_equality_conditions (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) :
  ((1 + a)^2 / (1 + b) ≤ 1 + a^2 / b ↔ b < -1 ∨ b > 0) ∧
  ((1 + a)^2 / (1 + b) = 1 + a^2 / b ↔ a = b ∧ b ≠ -1 ∧ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1078_107889


namespace NUMINAMATH_CALUDE_bakery_sugar_amount_l1078_107808

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount (sugar flour baking_soda : ℚ) 
  (h1 : sugar / flour = 5 / 2)
  (h2 : flour / baking_soda = 10 / 1)
  (h3 : flour / (baking_soda + 60) = 8 / 1) :
  sugar = 6000 := by
  sorry

end NUMINAMATH_CALUDE_bakery_sugar_amount_l1078_107808


namespace NUMINAMATH_CALUDE_circle_center_and_sum_l1078_107846

/-- The equation of a circle in the form x² + y² = ax + by + c -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

def circle_equation : CircleEquation :=
  { a := 6, b := -10, c := 9 }

theorem circle_center_and_sum (eq : CircleEquation) :
  ∃ (center : CircleCenter), 
    center.x = eq.a / 2 ∧
    center.y = -eq.b / 2 ∧
    center.x + center.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_sum_l1078_107846


namespace NUMINAMATH_CALUDE_total_cookies_l1078_107893

theorem total_cookies (num_bags : ℕ) (cookies_per_bag : ℕ) : 
  num_bags = 7 → cookies_per_bag = 2 → num_bags * cookies_per_bag = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l1078_107893


namespace NUMINAMATH_CALUDE_correct_operation_l1078_107812

theorem correct_operation (x y : ℝ) : 3 * x * y^2 - 4 * x * y^2 = -x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1078_107812


namespace NUMINAMATH_CALUDE_triangle_altitude_l1078_107865

theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) :
  area = 960 ∧ base = 48 ∧ area = (1/2) * base * altitude →
  altitude = 40 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_l1078_107865


namespace NUMINAMATH_CALUDE_park_area_is_525_l1078_107850

/-- Represents a rectangular park with given perimeter and length-width relationship. -/
structure RectangularPark where
  perimeter : ℝ
  length : ℝ
  width : ℝ
  perimeter_eq : perimeter = 2 * (length + width)
  length_eq : length = 3 * width - 10

/-- Calculates the area of a rectangular park. -/
def parkArea (park : RectangularPark) : ℝ := park.length * park.width

/-- Theorem stating that a rectangular park with perimeter 100 meters and length equal to
    three times the width minus 10 meters has an area of 525 square meters. -/
theorem park_area_is_525 (park : RectangularPark) 
    (h_perimeter : park.perimeter = 100) : parkArea park = 525 := by
  sorry

end NUMINAMATH_CALUDE_park_area_is_525_l1078_107850


namespace NUMINAMATH_CALUDE_math_city_intersections_l1078_107833

/-- Represents a city with streets and intersections -/
structure City where
  num_streets : ℕ
  num_non_intersections : ℕ

/-- Calculates the number of intersections in a city -/
def num_intersections (c : City) : ℕ :=
  Nat.choose c.num_streets 2 - c.num_non_intersections

/-- Theorem: A city with 10 streets and 2 non-intersections has 43 intersections -/
theorem math_city_intersections :
  let c : City := { num_streets := 10, num_non_intersections := 2 }
  num_intersections c = 43 := by sorry

end NUMINAMATH_CALUDE_math_city_intersections_l1078_107833


namespace NUMINAMATH_CALUDE_percentage_increase_relation_l1078_107849

theorem percentage_increase_relation (A B k x : ℝ) : 
  A > 0 → B > 0 → k > 1 → A = k * B → A = B * (1 + x / 100) → k = 1 + x / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_relation_l1078_107849


namespace NUMINAMATH_CALUDE_line_circle_separation_l1078_107827

/-- If a point (a,b) is inside the unit circle, then the line ax + by = 1 is separated from the circle -/
theorem line_circle_separation (a b : ℝ) (h : a^2 + b^2 < 1) :
  ∃ (d : ℝ), d > 1 ∧ d = 1 / Real.sqrt (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_separation_l1078_107827


namespace NUMINAMATH_CALUDE_remainder_proof_l1078_107878

theorem remainder_proof (n : ℕ) (h1 : n = 129) (h2 : 1428 % n = 9) : 2206 % n = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l1078_107878


namespace NUMINAMATH_CALUDE_cube_product_divided_l1078_107884

theorem cube_product_divided : (12 : ℝ)^3 * 6^3 / 432 = 864 := by
  sorry

end NUMINAMATH_CALUDE_cube_product_divided_l1078_107884


namespace NUMINAMATH_CALUDE_point_of_tangency_parabolas_l1078_107859

/-- The point of tangency of two parabolas -/
theorem point_of_tangency_parabolas :
  let f (x : ℝ) := x^2 + 8*x + 15
  let g (y : ℝ) := y^2 + 16*y + 63
  let point : ℝ × ℝ := (-7/2, -15/2)
  (f (point.1) = point.2 ∧ g (point.2) = point.1) ∧
  ∀ x y : ℝ, (f x = y ∧ g y = x) → (x, y) = point :=
by sorry


end NUMINAMATH_CALUDE_point_of_tangency_parabolas_l1078_107859


namespace NUMINAMATH_CALUDE_a_range_for_two_positive_zeros_l1078_107866

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

/-- The condition for f to have two positive zeros -/
def has_two_positive_zeros (a : ℝ) : Prop :=
  ∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0

/-- Theorem stating the range of a for f to have two positive zeros -/
theorem a_range_for_two_positive_zeros :
  ∀ a : ℝ, has_two_positive_zeros a ↔ a > 3 :=
sorry

end NUMINAMATH_CALUDE_a_range_for_two_positive_zeros_l1078_107866


namespace NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l1078_107875

theorem students_taking_neither_music_nor_art 
  (total_students : ℕ) 
  (music_students : ℕ) 
  (art_students : ℕ) 
  (both_students : ℕ) 
  (h1 : total_students = 500)
  (h2 : music_students = 40)
  (h3 : art_students = 20)
  (h4 : both_students = 10)
  : total_students - (music_students + art_students - both_students) = 450 :=
by
  sorry

#check students_taking_neither_music_nor_art

end NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l1078_107875


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l1078_107876

theorem quadratic_equivalence : ∀ x : ℝ, x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l1078_107876


namespace NUMINAMATH_CALUDE_set_operations_and_range_l1078_107830

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

theorem set_operations_and_range :
  ∀ a : ℝ, (B ∪ C a = C a) →
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 4}) ∧
  (A ∪ B = {x : ℝ | x ≥ 2}) ∧
  ((U \ (A ∪ B)) = {x : ℝ | 0 < x ∧ x < 2}) ∧
  ((U \ A) ∩ B = {x : ℝ | x ≥ 4}) ∧
  (∀ x : ℝ, x > -6 ↔ ∃ y : ℝ, y ∈ C x ∧ y ∉ B) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l1078_107830


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1078_107888

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + 3*x₁ - 2 = 0) ∧ (x₂^2 + 3*x₂ - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1078_107888


namespace NUMINAMATH_CALUDE_money_sharing_calculation_l1078_107809

/-- Proves that given a money sharing scenario with a specific ratio and known amount,
    the total shared amount can be calculated. -/
theorem money_sharing_calculation (mark_ratio nina_ratio oliver_ratio : ℕ)
                                  (nina_amount : ℕ) :
  mark_ratio = 2 →
  nina_ratio = 3 →
  oliver_ratio = 9 →
  nina_amount = 60 →
  ∃ (total : ℕ), total = 280 ∧ 
    nina_amount * (mark_ratio + nina_ratio + oliver_ratio) = total * nina_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_money_sharing_calculation_l1078_107809


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_divisibility_l1078_107862

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Theorem statement -/
theorem infinitely_many_pairs_divisibility :
  ∀ n : ℕ, 
    (fib (2*n+1) ∣ fib (2*n-1)^2 + 1) ∧ 
    (fib (2*n-1) ∣ fib (2*n+1)^2 + 1) := by
  sorry

#check infinitely_many_pairs_divisibility

end NUMINAMATH_CALUDE_infinitely_many_pairs_divisibility_l1078_107862


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1078_107828

theorem arithmetic_mean_problem (x y : ℝ) : 
  ((x + 10) + 18 + 3*x + 12 + (3*x + 6) + y) / 6 = 26 → x = 90/7 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1078_107828


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l1078_107867

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  a = 2 →
  c = 3 →
  B = 2 * π / 3 →  -- 120° in radians
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B →  -- Law of Cosines
  b = Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l1078_107867


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1078_107880

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), n = 15 ∧ 
  (∀ (m : ℕ), m < n → ¬(23 ∣ (78721 - m))) ∧ (23 ∣ (78721 - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1078_107880


namespace NUMINAMATH_CALUDE_smallest_n_value_l1078_107882

def is_not_divisible_by_ten (m : ℕ) : Prop := ∀ k : ℕ, m ≠ 10 * k

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a ≥ b → b ≥ c →
  a + b + c = 2010 →
  a * b * c = m * (10 ^ n) →
  is_not_divisible_by_ten m →
  (∀ k : ℕ, k < n → ∃ (m' : ℕ), a * b * c = m' * (10 ^ k) → ¬(is_not_divisible_by_ten m')) →
  n = 500 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_value_l1078_107882


namespace NUMINAMATH_CALUDE_unique_friendship_configs_l1078_107813

/-- Represents a friendship configuration in a group of 8 people --/
structure FriendshipConfig :=
  (num_friends : Nat)
  (valid : num_friends = 0 ∨ num_friends = 1 ∨ num_friends = 6)

/-- Counts the number of unique friendship configurations --/
def count_unique_configs : Nat :=
  sorry

/-- Theorem stating that the number of unique friendship configurations is 37 --/
theorem unique_friendship_configs :
  count_unique_configs = 37 :=
sorry

end NUMINAMATH_CALUDE_unique_friendship_configs_l1078_107813


namespace NUMINAMATH_CALUDE_train_speed_train_speed_problem_l1078_107887

/-- The speed of a train given its length, time to pass a person, and the person's speed in the opposite direction. -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_mps := relative_speed - man_speed_mps
  let train_speed_kmph := train_speed_mps * (3600 / 1000)
  train_speed_kmph

/-- Proof that a train 165 meters long passing a man in 9 seconds, with the man running at 6 kmph in the opposite direction, has a speed of 60 kmph. -/
theorem train_speed_problem : train_speed 165 9 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_problem_l1078_107887


namespace NUMINAMATH_CALUDE_f_properties_l1078_107848

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/a + 2/x

theorem f_properties (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 0 → f a x₁ < f a x₂) ∧
  (a < 0 → ∀ x : ℝ, x > 0 → f a x > 0) ∧
  (a > 0 → ∀ x : ℝ, 0 < x ∧ x < 2*a ↔ f a x > 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1078_107848


namespace NUMINAMATH_CALUDE_inequality_solution_l1078_107861

theorem inequality_solution (x : ℝ) : 3 * x + 2 < 10 - 2 * x → x < 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1078_107861


namespace NUMINAMATH_CALUDE_sandwich_combinations_l1078_107879

theorem sandwich_combinations (meat_types : ℕ) (cheese_types : ℕ) : meat_types = 12 → cheese_types = 8 → (meat_types.choose 2) * cheese_types = 528 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1078_107879


namespace NUMINAMATH_CALUDE_ratio_equality_l1078_107840

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 25)
  (sum_xyz : x^2 + y^2 + z^2 = 36)
  (sum_axbycz : a*x + b*y + c*z = 30) :
  (a + b + c) / (x + y + z) = 5/6 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1078_107840


namespace NUMINAMATH_CALUDE_circle_through_points_with_center_on_y_axis_l1078_107868

/-- The circle passing through points A (-1, 4) and B (3, 2) with its center on the y-axis -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 10

/-- Point A coordinates -/
def point_A : ℝ × ℝ := (-1, 4)

/-- Point B coordinates -/
def point_B : ℝ × ℝ := (3, 2)

/-- The center of the circle is on the y-axis -/
def center_on_y_axis (h k : ℝ) : Prop :=
  h = 0

theorem circle_through_points_with_center_on_y_axis :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  ∃ k, center_on_y_axis 0 k ∧
    ∀ x y, circle_equation x y ↔ (x - 0)^2 + (y - k)^2 = (0 - point_A.1)^2 + (k - point_A.2)^2 :=
sorry

end NUMINAMATH_CALUDE_circle_through_points_with_center_on_y_axis_l1078_107868


namespace NUMINAMATH_CALUDE_sixth_power_sum_of_roots_l1078_107891

theorem sixth_power_sum_of_roots (r s : ℝ) : 
  r^2 - 3 * r * Real.sqrt 2 + 4 = 0 →
  s^2 - 3 * s * Real.sqrt 2 + 4 = 0 →
  r^6 + s^6 = 648 := by
sorry

end NUMINAMATH_CALUDE_sixth_power_sum_of_roots_l1078_107891


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1078_107899

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1078_107899


namespace NUMINAMATH_CALUDE_system_solution_l1078_107870

theorem system_solution :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (x * y = 500 ∧ x^(Real.log y) = 25) ↔
  ((x = 100 ∧ y = 5) ∨ (x = 5 ∧ y = 100)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1078_107870


namespace NUMINAMATH_CALUDE_clothing_store_problem_l1078_107837

theorem clothing_store_problem (cost_A B : ℕ) (profit_A B : ℕ) :
  3 * cost_A + 2 * cost_B = 450 →
  cost_A + cost_B = 175 →
  profit_A = 30 →
  profit_B = 20 →
  (∀ m : ℕ, m ≤ 100 → profit_A * m + profit_B * (100 - m) ≥ 2400 →
    ∃ n : ℕ, n ≥ m ∧ n ≥ 40) →
  ∃ m : ℕ, m ≥ 40 ∧ ∀ n : ℕ, n < m → profit_A * n + profit_B * (100 - n) < 2400 :=
by
  sorry

end NUMINAMATH_CALUDE_clothing_store_problem_l1078_107837


namespace NUMINAMATH_CALUDE_lower_price_proof_l1078_107895

/-- Given a book with cost C and two selling prices P and H, where H yields 5% more gain than P, 
    this function calculates the lower selling price P. -/
def calculate_lower_price (C H : ℚ) : ℚ :=
  H / (1 + 0.05)

theorem lower_price_proof (C H : ℚ) (hC : C = 200) (hH : H = 350) :
  let P := calculate_lower_price C H
  ∃ ε > 0, |P - 368.42| < ε := by sorry

end NUMINAMATH_CALUDE_lower_price_proof_l1078_107895


namespace NUMINAMATH_CALUDE_energy_usage_is_219_l1078_107881

/-- Calculates the total energy usage given the energy consumption and duration of each light. -/
def total_energy_usage (bedroom_watts_per_hour : ℝ) 
                       (bedroom_hours : ℝ)
                       (office_multiplier : ℝ)
                       (office_hours : ℝ)
                       (living_room_multiplier : ℝ)
                       (living_room_hours : ℝ)
                       (kitchen_multiplier : ℝ)
                       (kitchen_hours : ℝ)
                       (bathroom_multiplier : ℝ)
                       (bathroom_hours : ℝ) : ℝ :=
  bedroom_watts_per_hour * bedroom_hours +
  (office_multiplier * bedroom_watts_per_hour) * office_hours +
  (living_room_multiplier * bedroom_watts_per_hour) * living_room_hours +
  (kitchen_multiplier * bedroom_watts_per_hour) * kitchen_hours +
  (bathroom_multiplier * bedroom_watts_per_hour) * bathroom_hours

/-- Theorem stating that the total energy usage is 219 watts given the specified conditions. -/
theorem energy_usage_is_219 :
  total_energy_usage 6 2 3 3 4 4 2 1 5 1.5 = 219 := by
  sorry

end NUMINAMATH_CALUDE_energy_usage_is_219_l1078_107881


namespace NUMINAMATH_CALUDE_initial_number_proof_l1078_107877

theorem initial_number_proof (N : ℕ) : 
  (∃ k : ℕ, N - 7 = 15 * k) ∧ 
  (∀ m : ℕ, m < 7 → ¬∃ j : ℕ, N - m = 15 * j) → 
  N = 22 := by
sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1078_107877


namespace NUMINAMATH_CALUDE_terez_cows_l1078_107860

theorem terez_cows (total : ℕ) (females : ℕ) (pregnant : ℕ) : 
  2 * females = total → 
  2 * pregnant = females → 
  pregnant = 11 → 
  total = 44 := by
sorry

end NUMINAMATH_CALUDE_terez_cows_l1078_107860


namespace NUMINAMATH_CALUDE_negative_square_cubed_l1078_107898

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l1078_107898


namespace NUMINAMATH_CALUDE_engine_problem_solution_l1078_107832

/-- Represents the fuel consumption and operation time of two engines -/
structure EnginePair where
  first_consumption : ℝ
  second_consumption : ℝ
  time_difference : ℝ
  consumption_difference : ℝ

/-- Determines if the given fuel consumption rates satisfy the conditions for the two engines -/
def is_valid_solution (pair : EnginePair) (first_rate second_rate : ℝ) : Prop :=
  first_rate > 0 ∧
  second_rate > 0 ∧
  first_rate = second_rate + pair.consumption_difference ∧
  pair.first_consumption / first_rate - pair.second_consumption / second_rate = pair.time_difference

/-- Theorem stating that the given solution satisfies the engine problem conditions -/
theorem engine_problem_solution (pair : EnginePair) 
    (h1 : pair.first_consumption = 300)
    (h2 : pair.second_consumption = 192)
    (h3 : pair.time_difference = 2)
    (h4 : pair.consumption_difference = 6) :
    is_valid_solution pair 30 24 := by
  sorry


end NUMINAMATH_CALUDE_engine_problem_solution_l1078_107832


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_and_inradius_l1078_107821

theorem right_triangle_arithmetic_progression_and_inradius (d : ℝ) (h : d > 0) :
  ∃ (a b c : ℝ),
    a^2 + b^2 = c^2 ∧  -- Pythagorean theorem
    a = 3*d ∧ 
    b = 4*d ∧ 
    c = 5*d ∧ 
    (a + b - c) / 2 = d  -- Inradius formula
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_and_inradius_l1078_107821


namespace NUMINAMATH_CALUDE_floor_of_4_7_l1078_107805

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l1078_107805


namespace NUMINAMATH_CALUDE_cylindrical_to_rectangular_conversion_l1078_107844

/-- Conversion from cylindrical coordinates to rectangular coordinates -/
theorem cylindrical_to_rectangular_conversion 
  (r θ z : ℝ) 
  (hr : r = 7) 
  (hθ : θ = π / 3) 
  (hz : z = -3) :
  ∃ (x y : ℝ), 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ ∧ 
    x = 3.5 ∧ 
    y = 7 * Real.sqrt 3 / 2 ∧ 
    z = -3 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_to_rectangular_conversion_l1078_107844


namespace NUMINAMATH_CALUDE_octal_calculation_l1078_107838

/-- Converts a number from base 8 to base 10 --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 8 --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- Multiplies two numbers in base 8 --/
def octal_multiply (a b : ℕ) : ℕ := 
  decimal_to_octal (octal_to_decimal a * octal_to_decimal b)

/-- Subtracts two numbers in base 8 --/
def octal_subtract (a b : ℕ) : ℕ := 
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

theorem octal_calculation : 
  octal_subtract (octal_multiply 245 5) 107 = 1356 := by sorry

end NUMINAMATH_CALUDE_octal_calculation_l1078_107838


namespace NUMINAMATH_CALUDE_carls_weight_l1078_107871

theorem carls_weight (al ben carl ed : ℕ) 
  (h1 : al = ben + 25)
  (h2 : ben + 16 = carl)
  (h3 : ed = 146)
  (h4 : al = ed + 38) :
  carl = 175 := by
  sorry

end NUMINAMATH_CALUDE_carls_weight_l1078_107871


namespace NUMINAMATH_CALUDE_total_pay_calculation_l1078_107892

/-- Calculate the total pay for a worker given their regular and overtime hours -/
theorem total_pay_calculation (regular_rate : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ) :
  let overtime_rate := 2 * regular_rate
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  let total_pay := regular_pay + overtime_pay
  (regular_rate = 3 ∧ regular_hours = 40 ∧ overtime_hours = 10) →
  total_pay = 180 := by
sorry


end NUMINAMATH_CALUDE_total_pay_calculation_l1078_107892


namespace NUMINAMATH_CALUDE_sum_geq_three_cube_root_three_l1078_107815

theorem sum_geq_three_cube_root_three
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (h : a^3 + b^3 + c^3 = a^2 * b^2 * c^2) :
  a + b + c ≥ 3 * (3 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_sum_geq_three_cube_root_three_l1078_107815


namespace NUMINAMATH_CALUDE_trigonometric_sum_equals_three_halves_l1078_107856

theorem trigonometric_sum_equals_three_halves :
  Real.sin (π / 24) ^ 4 + Real.cos (5 * π / 24) ^ 4 + 
  Real.sin (19 * π / 24) ^ 4 + Real.cos (23 * π / 24) ^ 4 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equals_three_halves_l1078_107856


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1078_107843

theorem complex_fraction_sum (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I) / (1 + Complex.I) = Complex.mk a b →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1078_107843


namespace NUMINAMATH_CALUDE_exactly_two_partitions_l1078_107857

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- A function that represents a valid partition of 100 into three distinct positive perfect squares -/
def valid_partition (a b c : ℕ) : Prop :=
  a + b + c = 100 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c

/-- The main theorem stating that there are exactly 2 valid partitions -/
theorem exactly_two_partitions :
  ∃! (s : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ s ↔ valid_partition a b c) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_partitions_l1078_107857


namespace NUMINAMATH_CALUDE_equation_solutions_l1078_107810

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 - 16 = 0 ↔ x = 6 ∨ x = -2) ∧
  (∀ x : ℝ, (x + 3)^3 = -27 ↔ x = -6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1078_107810


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1078_107869

/-- A line is tangent to a circle if the distance from the center of the circle to the line is equal to the radius of the circle. -/
def is_tangent_line (a b c : ℝ) (r : ℝ) : Prop :=
  |c| / Real.sqrt (a^2 + b^2) = r

theorem tangent_line_to_circle (b : ℝ) :
  is_tangent_line 2 (-1) b (Real.sqrt 5) ↔ b = 5 ∨ b = -5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1078_107869


namespace NUMINAMATH_CALUDE_pie_price_is_seven_l1078_107873

def number_of_cakes : ℕ := 453
def price_per_cake : ℕ := 12
def number_of_pies : ℕ := 126
def total_earnings : ℕ := 6318

theorem pie_price_is_seven :
  ∃ (price_per_pie : ℕ),
    price_per_pie = 7 ∧
    price_per_pie * number_of_pies + price_per_cake * number_of_cakes = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_pie_price_is_seven_l1078_107873


namespace NUMINAMATH_CALUDE_trevor_dropped_eggs_l1078_107894

/-- The number of eggs Trevor collected from each chicken and the number left after dropping some -/
structure EggCollection where
  gertrude : Nat
  blanche : Nat
  nancy : Nat
  martha : Nat
  left : Nat

/-- The total number of eggs collected -/
def total_eggs (e : EggCollection) : Nat :=
  e.gertrude + e.blanche + e.nancy + e.martha

/-- The number of eggs Trevor dropped -/
def dropped_eggs (e : EggCollection) : Nat :=
  total_eggs e - e.left

theorem trevor_dropped_eggs (e : EggCollection) 
  (h1 : e.gertrude = 4)
  (h2 : e.blanche = 3)
  (h3 : e.nancy = 2)
  (h4 : e.martha = 2)
  (h5 : e.left = 9) :
  dropped_eggs e = 2 := by
  sorry

#check trevor_dropped_eggs

end NUMINAMATH_CALUDE_trevor_dropped_eggs_l1078_107894


namespace NUMINAMATH_CALUDE_total_birds_in_store_l1078_107874

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 6

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 7

/-- Theorem: The total number of birds in the pet store is 54 -/
theorem total_birds_in_store : 
  num_cages * (parrots_per_cage + parakeets_per_cage) = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_in_store_l1078_107874


namespace NUMINAMATH_CALUDE_white_bread_served_l1078_107872

/-- Given that a restaurant served 0.5 loaf of wheat bread and a total of 0.9 loaves,
    prove that 0.4 loaves of white bread were served. -/
theorem white_bread_served (wheat_bread : ℝ) (total_bread : ℝ) (white_bread : ℝ)
    (h1 : wheat_bread = 0.5)
    (h2 : total_bread = 0.9)
    (h3 : white_bread = total_bread - wheat_bread) :
    white_bread = 0.4 := by
  sorry

#check white_bread_served

end NUMINAMATH_CALUDE_white_bread_served_l1078_107872


namespace NUMINAMATH_CALUDE_shaded_region_correct_l1078_107831

-- Define the universal set U and subsets A and B
variable (U : Type) (A B : Set U)

-- Define the shaded region
def shaded_region (U : Type) (A B : Set U) : Set U :=
  (Aᶜ) ∩ (Bᶜ)

-- Theorem statement
theorem shaded_region_correct (U : Type) (A B : Set U) :
  shaded_region U A B = (Aᶜ) ∩ (Bᶜ) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_shaded_region_correct_l1078_107831


namespace NUMINAMATH_CALUDE_game_ends_in_45_rounds_l1078_107885

/-- Represents the state of the game with token counts for each player -/
structure GameState where
  playerA : ℕ
  playerB : ℕ
  playerC : ℕ

/-- Applies one round of the game rules to the current state -/
def applyRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (any player has 0 tokens) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends -/
def countRounds (initialState : GameState) : ℕ :=
  sorry

theorem game_ends_in_45_rounds :
  let initialState : GameState := ⟨18, 16, 15⟩
  countRounds initialState = 45 := by
  sorry

end NUMINAMATH_CALUDE_game_ends_in_45_rounds_l1078_107885


namespace NUMINAMATH_CALUDE_vending_machine_probability_l1078_107826

/-- Represents a vending machine with toys and their prices -/
structure VendingMachine :=
  (num_toys : ℕ)
  (min_price : ℚ)
  (price_increment : ℚ)

/-- Represents Peter's initial money -/
structure InitialMoney :=
  (quarters : ℕ)
  (bill : ℚ)

/-- The main theorem statement -/
theorem vending_machine_probability
  (vm : VendingMachine)
  (money : InitialMoney)
  (favorite_toy_price : ℚ) :
  vm.num_toys = 10 →
  vm.min_price = 25/100 →
  vm.price_increment = 25/100 →
  money.quarters = 10 →
  money.bill = 20 →
  favorite_toy_price = 2 →
  (probability_need_break_bill : ℚ) →
  probability_need_break_bill = 9/10 :=
by sorry

end NUMINAMATH_CALUDE_vending_machine_probability_l1078_107826


namespace NUMINAMATH_CALUDE_trajectory_and_PQ_length_l1078_107839

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the point A on C₁
def A_on_C₁ (x₀ y₀ : ℝ) : Prop := C₁ x₀ y₀

-- Define the perpendicular condition for AN
def AN_perp_x (x₀ y₀ : ℝ) : Prop := ∃ (N : ℝ × ℝ), N.1 = x₀ ∧ N.2 = 0

-- Define the condition for point M
def M_condition (x y x₀ y₀ : ℝ) : Prop :=
  ∃ (N : ℝ × ℝ), N.1 = x₀ ∧ N.2 = 0 ∧
  (x, y) + 2 * (x - x₀, y - y₀) = (2 * Real.sqrt 2 - 2) • (x₀, 0)

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the intersection of line l with curve C
def l_intersects_C (P Q : ℝ × ℝ) : Prop :=
  C P.1 P.2 ∧ C Q.1 Q.2 ∧ P ≠ Q

-- Define the condition for circle PQ passing through O
def circle_PQ_through_O (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = 0

theorem trajectory_and_PQ_length :
  ∀ (x y x₀ y₀ : ℝ) (P Q : ℝ × ℝ),
  A_on_C₁ x₀ y₀ →
  AN_perp_x x₀ y₀ →
  M_condition x y x₀ y₀ →
  l_intersects_C P Q →
  circle_PQ_through_O P Q →
  (C x y ∧ 
   (4 * Real.sqrt 6 / 3)^2 ≤ ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ∧
   ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ (2 * Real.sqrt 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_PQ_length_l1078_107839


namespace NUMINAMATH_CALUDE_zero_at_neg_one_one_zero_in_interval_l1078_107825

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 2 - a

-- Theorem 1: When a = -1, the function has a zero at x = 1
theorem zero_at_neg_one :
  f (-1) 1 = 0 := by sorry

-- Theorem 2: The function has exactly one zero in (0, 1] iff -1 ≤ a ≤ 0 or a ≤ -2
theorem one_zero_in_interval (a : ℝ) :
  (∃! x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a x = 0) ↔ (-1 ≤ a ∧ a ≤ 0) ∨ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_zero_at_neg_one_one_zero_in_interval_l1078_107825


namespace NUMINAMATH_CALUDE_min_value_expression_l1078_107863

/-- Given that x₁ and x₂ are the roots of the equations x + exp x = 3 and x + log x = 3 respectively,
    and x₁ + x₂ = a + b where a and b are positive real numbers,
    prove that the minimum value of (7b² + 1) / (ab) is 2. -/
theorem min_value_expression (x₁ x₂ a b : ℝ) : 
  (∃ (x : ℝ), x + Real.exp x = 3 ∧ x = x₁) →
  (∃ (x : ℝ), x + Real.log x = 3 ∧ x = x₂) →
  x₁ + x₂ = a + b →
  a > 0 →
  b > 0 →
  (∀ c d : ℝ, c > 0 → d > 0 → c + d = a + b → (7 * d^2 + 1) / (c * d) ≥ 2) ∧
  (∃ e f : ℝ, e > 0 ∧ f > 0 ∧ e + f = a + b ∧ (7 * f^2 + 1) / (e * f) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1078_107863


namespace NUMINAMATH_CALUDE_two_circles_common_tangents_l1078_107852

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Number of common tangent lines between two circles -/
def commonTangentLines (c1 c2 : Circle) : ℕ :=
  sorry

/-- The main theorem -/
theorem two_circles_common_tangents :
  let c1 : Circle := { center := (1, 2), radius := 1 }
  let c2 : Circle := { center := (2, 5), radius := 3 }
  commonTangentLines c1 c2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_circles_common_tangents_l1078_107852


namespace NUMINAMATH_CALUDE_unique_420_sequence_l1078_107823

-- Define the sum of consecutive integers
def sum_consecutive (n : ℕ) (k : ℕ) : ℕ := k * n + k * (k - 1) / 2

-- Define a predicate for valid sequences
def valid_sequence (n : ℕ) (k : ℕ) : Prop :=
  k ≥ 2 ∧ (k % 2 = 0 ∨ k = 3) ∧ sum_consecutive n k = 420

-- The main theorem
theorem unique_420_sequence :
  ∃! (n k : ℕ), valid_sequence n k :=
sorry

end NUMINAMATH_CALUDE_unique_420_sequence_l1078_107823


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l1078_107806

/-- Given integers a, b, c, d, e, and f satisfying the polynomial identity
    729x^3 + 64 = (ax^2 + bx + c)(dx^2 + ex + f) for all x,
    prove that a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 -/
theorem polynomial_identity_sum_of_squares (a b c d e f : ℤ) 
  (h : ∀ x : ℝ, 729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l1078_107806


namespace NUMINAMATH_CALUDE_min_value_on_interval_l1078_107883

/-- The function f(x) = x^2 - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The closed interval [2, 3] -/
def I : Set ℝ := Set.Icc 2 3

theorem min_value_on_interval :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ x ∈ I, f x ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l1078_107883


namespace NUMINAMATH_CALUDE_range_of_m_l1078_107841

def f (x : ℝ) : ℝ := x^2 - 4*x - 2

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-6) (-2)) ∧
  (∀ y ∈ Set.Icc (-6) (-2), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1078_107841


namespace NUMINAMATH_CALUDE_number_of_divisors_of_45_l1078_107814

theorem number_of_divisors_of_45 : Nat.card {d : ℕ | d ∣ 45} = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_45_l1078_107814


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l1078_107818

theorem consecutive_even_integers_sum (n : ℤ) : 
  (∃ k : ℤ, n = k^2) → 
  (n - 2) + (n + 2) = 162 → 
  (n - 2) + n + (n + 2) = 243 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l1078_107818


namespace NUMINAMATH_CALUDE_power_sum_fifth_l1078_107886

theorem power_sum_fifth (a b x y : ℝ) 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_fifth_l1078_107886


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l1078_107855

theorem unique_solution_power_equation :
  ∃! (x y m n : ℕ), x > y ∧ y > 0 ∧ m > 1 ∧ n > 1 ∧ (x + y)^n = x^m + y^m :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l1078_107855


namespace NUMINAMATH_CALUDE_shorter_segment_length_l1078_107801

-- Define the triangle
def triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the altitude and segments
def altitude_segment (a b c x h : ℝ) : Prop :=
  triangle a b c ∧
  x > 0 ∧ h > 0 ∧
  x + (c - x) = c ∧
  a^2 = x^2 + h^2 ∧
  b^2 = (c - x)^2 + h^2

-- Theorem statement
theorem shorter_segment_length :
  ∀ (x h : ℝ),
  altitude_segment 40 50 90 x h →
  x = 40 :=
sorry

end NUMINAMATH_CALUDE_shorter_segment_length_l1078_107801


namespace NUMINAMATH_CALUDE_invalid_assignment_l1078_107800

-- Define what constitutes a valid assignment statement
def is_valid_assignment (lhs : String) (rhs : String) : Prop :=
  lhs.length = 1 ∧ lhs.all Char.isAlpha

-- Define the statement in question
def statement : String × String := ("x*y", "a")

-- Theorem to prove
theorem invalid_assignment :
  ¬(is_valid_assignment statement.1 statement.2) :=
sorry

end NUMINAMATH_CALUDE_invalid_assignment_l1078_107800


namespace NUMINAMATH_CALUDE_original_number_is_perfect_square_l1078_107853

theorem original_number_is_perfect_square :
  ∃ (n : ℕ), n^2 = 1296 ∧ ∃ (m : ℕ), (1296 + 148) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_perfect_square_l1078_107853


namespace NUMINAMATH_CALUDE_abs_m_eq_abs_half_implies_m_eq_plus_minus_half_l1078_107834

theorem abs_m_eq_abs_half_implies_m_eq_plus_minus_half (m : ℝ) : 
  |(-m)| = |(-1/2)| → m = -1/2 ∨ m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_abs_m_eq_abs_half_implies_m_eq_plus_minus_half_l1078_107834


namespace NUMINAMATH_CALUDE_circle_rolling_inside_square_l1078_107896

/-- The distance traveled by the center of a circle rolling inside a square -/
theorem circle_rolling_inside_square
  (circle_radius : ℝ)
  (square_side : ℝ)
  (h1 : circle_radius = 1)
  (h2 : square_side = 5) :
  (square_side - 2 * circle_radius) * 4 = 12 :=
by sorry

end NUMINAMATH_CALUDE_circle_rolling_inside_square_l1078_107896
