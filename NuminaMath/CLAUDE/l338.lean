import Mathlib

namespace NUMINAMATH_CALUDE_one_basket_total_peaches_l338_33868

/-- Given a basket of peaches with red and green peaches, calculate the total number of peaches -/
def total_peaches (red_peaches green_peaches : ℕ) : ℕ :=
  red_peaches + green_peaches

/-- Theorem: The total number of peaches in 1 basket is 7 -/
theorem one_basket_total_peaches :
  total_peaches 4 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_one_basket_total_peaches_l338_33868


namespace NUMINAMATH_CALUDE_dereks_lowest_score_l338_33840

theorem dereks_lowest_score (test1 test2 test3 test4 : ℕ) : 
  test1 = 85 →
  test2 = 78 →
  test1 ≤ 100 →
  test2 ≤ 100 →
  test3 ≤ 100 →
  test4 ≤ 100 →
  test3 ≥ 60 →
  test4 ≥ 60 →
  (test1 + test2 + test3 + test4) / 4 = 84 →
  (min test3 test4 = 73 ∨ min test3 test4 > 73) :=
by sorry

end NUMINAMATH_CALUDE_dereks_lowest_score_l338_33840


namespace NUMINAMATH_CALUDE_qinJiushao_V₁_for_f_10_l338_33846

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qinJiushao (f : ℝ → ℝ) (x : ℝ) : ℝ := sorry

/-- The polynomial f(x) = 3x⁴ + 2x² + x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

/-- V₁ in Qin Jiushao's algorithm for f(10) -/
def V₁ : ℝ := 3 * 10 + 2

theorem qinJiushao_V₁_for_f_10 : 
  V₁ = 32 := by sorry

end NUMINAMATH_CALUDE_qinJiushao_V₁_for_f_10_l338_33846


namespace NUMINAMATH_CALUDE_sausages_theorem_l338_33805

def sausages_left (initial : ℕ) : ℕ :=
  let after_monday := initial - (2 * initial / 5)
  let after_tuesday := after_monday - (after_monday / 2)
  let after_wednesday := after_tuesday - (after_tuesday / 4)
  let after_thursday := after_wednesday - (after_wednesday / 3)
  let after_sharing := after_thursday - (after_thursday / 5)
  after_sharing - ((3 * after_sharing) / 5)

theorem sausages_theorem :
  sausages_left 1200 = 58 := by
  sorry

end NUMINAMATH_CALUDE_sausages_theorem_l338_33805


namespace NUMINAMATH_CALUDE_det_equals_nine_l338_33861

-- Define the determinant for a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem det_equals_nine (x : ℝ) (h : x^2 - 2*x - 5 = 0) : 
  det2x2 (x + 1) x (4 - x) (x - 1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_det_equals_nine_l338_33861


namespace NUMINAMATH_CALUDE_distance_to_tangent_point_l338_33881

/-- Two externally tangent circles with a common external tangent -/
structure TangentCircles where
  /-- Radius of the larger circle -/
  r₁ : ℝ
  /-- Radius of the smaller circle -/
  r₂ : ℝ
  /-- The circles are externally tangent -/
  tangent : r₁ > 0 ∧ r₂ > 0
  /-- The common external tangent exists -/
  common_tangent_exists : True

/-- The distance from the center of the larger circle to the point where 
    the common external tangent touches the smaller circle -/
theorem distance_to_tangent_point (c : TangentCircles) (h₁ : c.r₁ = 10) (h₂ : c.r₂ = 5) :
  ∃ d : ℝ, d = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_tangent_point_l338_33881


namespace NUMINAMATH_CALUDE_expression_evaluation_l338_33890

theorem expression_evaluation : 
  |-2| + (1/4 : ℝ) - 1 - 4 * Real.cos (π/4) + Real.sqrt 8 = 5/4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l338_33890


namespace NUMINAMATH_CALUDE_shifted_function_equals_g_l338_33832

-- Define the original function
def f (x : ℝ) : ℝ := -3 * x + 2

-- Define the shifted function
def g (x : ℝ) : ℝ := -3 * x - 1

-- Define the vertical shift
def shift : ℝ := 3

-- Theorem statement
theorem shifted_function_equals_g :
  ∀ x : ℝ, f x - shift = g x :=
by
  sorry

end NUMINAMATH_CALUDE_shifted_function_equals_g_l338_33832


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l338_33817

theorem solution_to_linear_equation :
  ∃ (x y : ℝ), x + 2 * y = 6 ∧ x = 2 ∧ y = 2 := by
sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l338_33817


namespace NUMINAMATH_CALUDE_find_number_l338_33867

theorem find_number : ∃ x : ℕ, x * 9999 = 724777430 ∧ x = 72483 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l338_33867


namespace NUMINAMATH_CALUDE_clara_stickers_l338_33811

def stickers_left (initial : ℕ) (given_to_boy : ℕ) : ℕ := 
  (initial - given_to_boy) / 2

theorem clara_stickers : stickers_left 100 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_clara_stickers_l338_33811


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l338_33806

/-- A line passing through points (4, -5) and (k, 23) is parallel to the line 3x - 4y = 12 -/
theorem parallel_line_k_value (k : ℚ) : 
  (∃ (m b : ℚ), (m * 4 + b = -5) ∧ (m * k + b = 23) ∧ (m = 3/4)) → k = 124/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l338_33806


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l338_33816

-- Problem 1
theorem problem_1 : 12 - (-18) - |(-7)| + 15 = 38 := by sorry

-- Problem 2
theorem problem_2 : -24 / (-3/2) + 6 * (-1/3) = 14 := by sorry

-- Problem 3
theorem problem_3 : (-7/9 + 5/6 - 1/4) * (-36) = 7 := by sorry

-- Problem 4
theorem problem_4 : -1^2 + 1/4 * (-2)^3 + (-3)^2 = 6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l338_33816


namespace NUMINAMATH_CALUDE_equation_solution_l338_33883

theorem equation_solution (n : ℝ) (h : n = 3) :
  n^4 - 20*n + 1 = 22 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l338_33883


namespace NUMINAMATH_CALUDE_susan_cats_proof_l338_33842

/-- The number of cats Bob has -/
def bob_cats : ℕ := 3

/-- The number of cats Susan gives away -/
def cats_given_away : ℕ := 4

/-- The difference in cats between Susan and Bob after Susan gives some away -/
def cat_difference : ℕ := 14

/-- Susan's initial number of cats -/
def susan_initial_cats : ℕ := 25

theorem susan_cats_proof :
  susan_initial_cats = bob_cats + cats_given_away + cat_difference := by
  sorry

end NUMINAMATH_CALUDE_susan_cats_proof_l338_33842


namespace NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l338_33829

theorem smallest_third_term_geometric_progression (a b c : ℝ) : 
  a = 5 ∧ 
  b - a = c - b ∧ 
  (5 * (c + 27) = (b + 9)^2) →
  c + 27 ≥ 16 - 4 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l338_33829


namespace NUMINAMATH_CALUDE_f_extrema_l338_33896

/-- A cubic function f(x) = x³ - px² - qx that is tangent to the x-axis at (1,0) -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^3 - p*x^2 - q*x

/-- The condition that f(x) is tangent to the x-axis at (1,0) -/
def is_tangent (p q : ℝ) : Prop :=
  f p q 1 = 0 ∧ (p + q = 1) ∧ (p^2 + 4*q = 0)

theorem f_extrema (p q : ℝ) (h : is_tangent p q) :
  (∃ x, f p q x = 4/27) ∧ (∀ x, f p q x ≥ 0) ∧ (∃ x, f p q x = 0) :=
sorry

end NUMINAMATH_CALUDE_f_extrema_l338_33896


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l338_33888

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : balls_in_boxes 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l338_33888


namespace NUMINAMATH_CALUDE_subjective_collection_not_set_l338_33835

-- Define a type for objects in a textbook
structure TextbookObject where
  id : Nat

-- Define a property that determines if an object belongs to a collection
def belongsToCollection (P : TextbookObject → Prop) (obj : TextbookObject) : Prop :=
  P obj

-- Define what it means for a collection to have a clear, objective criterion
def hasClearCriterion (P : TextbookObject → Prop) : Prop :=
  ∀ (obj1 obj2 : TextbookObject), obj1 = obj2 → (P obj1 ↔ P obj2)

-- Define what it means for a collection to be subjective
def isSubjective (P : TextbookObject → Prop) : Prop :=
  ∃ (obj1 obj2 : TextbookObject), obj1 = obj2 ∧ (P obj1 ↔ ¬P obj2)

-- Theorem: A collection with subjective criteria cannot form a well-defined set
theorem subjective_collection_not_set (P : TextbookObject → Prop) :
  isSubjective P → ¬(hasClearCriterion P) :=
by
  sorry

#check subjective_collection_not_set

end NUMINAMATH_CALUDE_subjective_collection_not_set_l338_33835


namespace NUMINAMATH_CALUDE_expression_simplification_l338_33869

theorem expression_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) (h3 : x ≠ -2) :
  (x - 1 - 3 / (x + 1)) / ((x^2 - 4) / (x^2 + 2*x + 1)) = x + 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l338_33869


namespace NUMINAMATH_CALUDE_rectangle_circle_chord_length_l338_33856

theorem rectangle_circle_chord_length :
  ∀ (rectangle : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) (P Q : ℝ × ℝ),
    -- Rectangle properties
    (∀ (x y : ℝ), (x, y) ∈ rectangle ↔ 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 2) →
    -- Circle properties
    (∃ (cx cy : ℝ), ∀ (x y : ℝ), (x, y) ∈ circle ↔ (x - cx)^2 + (y - cy)^2 = 1) →
    -- Circle touches three sides of the rectangle
    (∃ (x : ℝ), (x, 0) ∈ circle ∧ 0 < x ∧ x < 4) →
    (∃ (y : ℝ), (0, y) ∈ circle ∧ 0 < y ∧ y < 2) →
    (∃ (x : ℝ), (x, 2) ∈ circle ∧ 0 < x ∧ x < 4) →
    -- P and Q are on the circle and the diagonal
    P ∈ circle → Q ∈ circle →
    (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ P = (4*t, 2*t) ∧ Q = (4*(1-t), 2*(1-t))) →
    -- Conclusion: length of PQ is 4/√5
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 4 / Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_chord_length_l338_33856


namespace NUMINAMATH_CALUDE_complete_square_form_l338_33801

theorem complete_square_form (x : ℝ) : 
  (∃ a b : ℝ, (-x + 1) * (1 - x) = (a + b)^2) ∧ 
  (¬∃ a b : ℝ, (1 + x) * (1 - x) = (a + b)^2) ∧ 
  (¬∃ a b : ℝ, (-x - 1) * (-1 + x) = (a + b)^2) ∧ 
  (¬∃ a b : ℝ, (x - 1) * (1 + x) = (a + b)^2) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_form_l338_33801


namespace NUMINAMATH_CALUDE_perfect_numbers_mn_value_S_is_perfect_min_sum_value_l338_33864

/-- Definition of a perfect number -/
def is_perfect_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

/-- Statement 1: 29 and 13 are perfect numbers -/
theorem perfect_numbers : is_perfect_number 29 ∧ is_perfect_number 13 := by sorry

/-- Statement 2: Given equation has mn = ±4 -/
theorem mn_value (m n : ℝ) : 
  (∀ a : ℝ, a^2 - 4*a + 8 = (a - m)^2 + n^2) → m*n = 4 ∨ m*n = -4 := by sorry

/-- Statement 3: S is a perfect number when k = 36 -/
theorem S_is_perfect (a b : ℤ) :
  let S := a^2 + 4*a*b + 5*b^2 - 12*b + 36
  ∃ x y : ℤ, S = x^2 + y^2 := by sorry

/-- Statement 4: Minimum value of a + b is 3 -/
theorem min_sum_value (a b : ℝ) :
  -a^2 + 5*a + b - 7 = 0 → a + b ≥ 3 := by sorry

end NUMINAMATH_CALUDE_perfect_numbers_mn_value_S_is_perfect_min_sum_value_l338_33864


namespace NUMINAMATH_CALUDE_williams_tickets_l338_33865

/-- William's ticket problem -/
theorem williams_tickets : 
  ∀ (initial_tickets additional_tickets : ℕ),
  initial_tickets = 15 → 
  additional_tickets = 3 → 
  initial_tickets + additional_tickets = 18 := by
sorry

end NUMINAMATH_CALUDE_williams_tickets_l338_33865


namespace NUMINAMATH_CALUDE_game_configurations_l338_33863

/-- The number of rows in the grid -/
def m : ℕ := 5

/-- The number of columns in the grid -/
def n : ℕ := 7

/-- The total number of steps needed to reach from bottom-left to top-right -/
def total_steps : ℕ := m + n

/-- The number of unique paths from bottom-left to top-right of an m × n grid -/
def num_paths : ℕ := Nat.choose total_steps n

theorem game_configurations : num_paths = 792 := by sorry

end NUMINAMATH_CALUDE_game_configurations_l338_33863


namespace NUMINAMATH_CALUDE_books_from_first_shop_l338_33889

theorem books_from_first_shop :
  ∀ (x : ℕ),
    (1000 : ℝ) + 800 = 20 * (x + 40) →
    x = 50 := by
  sorry

end NUMINAMATH_CALUDE_books_from_first_shop_l338_33889


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l338_33882

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r = 3) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l338_33882


namespace NUMINAMATH_CALUDE_student_average_age_l338_33836

theorem student_average_age (num_students : ℕ) (teacher_age : ℕ) (avg_increase : ℕ) :
  num_students = 15 →
  teacher_age = 26 →
  avg_increase = 1 →
  (num_students * 10 + teacher_age) / (num_students + 1) = 10 + avg_increase :=
by sorry

end NUMINAMATH_CALUDE_student_average_age_l338_33836


namespace NUMINAMATH_CALUDE_norma_cards_l338_33800

theorem norma_cards (initial_cards lost_cards remaining_cards : ℕ) :
  lost_cards = 70 →
  remaining_cards = 18 →
  initial_cards = lost_cards + remaining_cards →
  initial_cards = 88 := by
sorry

end NUMINAMATH_CALUDE_norma_cards_l338_33800


namespace NUMINAMATH_CALUDE_factorization_equality_l338_33871

theorem factorization_equality (a b : ℝ) :
  a * b^2 - 2 * a^2 * b + a^2 = a * (b - a)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l338_33871


namespace NUMINAMATH_CALUDE_lcm_gcd_sum_reciprocal_sum_l338_33862

theorem lcm_gcd_sum_reciprocal_sum (m n : ℕ+) 
  (h_lcm : Nat.lcm m n = 210)
  (h_gcd : Nat.gcd m n = 6)
  (h_sum : m + n = 72) :
  1 / (m : ℚ) + 1 / (n : ℚ) = 2 / 35 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_sum_reciprocal_sum_l338_33862


namespace NUMINAMATH_CALUDE_range_of_negative_values_l338_33838

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y, x < y ∧ y ≤ 0 → f x > f y

-- State the theorem
theorem range_of_negative_values (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_decreasing : decreasing_on_neg f) 
  (h_zero : f 3 = 0) : 
  {x : ℝ | f x < 0} = Set.Ioo (-3) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_negative_values_l338_33838


namespace NUMINAMATH_CALUDE_school_boys_count_l338_33878

theorem school_boys_count (muslim_percent : ℝ) (hindu_percent : ℝ) (sikh_percent : ℝ) (other_count : ℕ) :
  muslim_percent = 44 →
  hindu_percent = 28 →
  sikh_percent = 10 →
  other_count = 54 →
  ∃ (total : ℕ), 
    (muslim_percent + hindu_percent + sikh_percent + (other_count : ℝ) / (total : ℝ) * 100 = 100) ∧
    total = 300 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l338_33878


namespace NUMINAMATH_CALUDE_longest_side_is_80_l338_33818

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 120
  area_eq : length * width = 2400

/-- The longest side of a SpecialRectangle is 80 -/
theorem longest_side_is_80 (rect : SpecialRectangle) : 
  max rect.length rect.width = 80 := by
  sorry

#check longest_side_is_80

end NUMINAMATH_CALUDE_longest_side_is_80_l338_33818


namespace NUMINAMATH_CALUDE_juice_boxes_for_school_year_l338_33847

/-- Calculate the total number of juice boxes needed for a school year -/
def total_juice_boxes (num_children : ℕ) (school_days_per_week : ℕ) (weeks_in_school_year : ℕ) : ℕ :=
  num_children * school_days_per_week * weeks_in_school_year

/-- Theorem: Given the specific conditions, the total number of juice boxes needed is 375 -/
theorem juice_boxes_for_school_year :
  let num_children : ℕ := 3
  let school_days_per_week : ℕ := 5
  let weeks_in_school_year : ℕ := 25
  total_juice_boxes num_children school_days_per_week weeks_in_school_year = 375 := by
  sorry


end NUMINAMATH_CALUDE_juice_boxes_for_school_year_l338_33847


namespace NUMINAMATH_CALUDE_gcd_sum_problem_l338_33839

def is_valid (a b c : ℕ+) : Prop :=
  Nat.gcd a.val (Nat.gcd b.val c.val) = 1 ∧
  Nat.gcd a.val (b.val + c.val) > 1 ∧
  Nat.gcd b.val (c.val + a.val) > 1 ∧
  Nat.gcd c.val (a.val + b.val) > 1

theorem gcd_sum_problem :
  (∃ a b c : ℕ+, is_valid a b c ∧ a.val + b.val + c.val = 2015) ∧
  (∀ a b c : ℕ+, is_valid a b c → a.val + b.val + c.val ≥ 30) ∧
  (∃ a b c : ℕ+, is_valid a b c ∧ a.val + b.val + c.val = 30) := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_problem_l338_33839


namespace NUMINAMATH_CALUDE_trip_distance_l338_33899

theorem trip_distance (total : ℝ) 
  (h1 : total > 0)
  (h2 : total / 4 + 30 + total / 10 + (total - (total / 4 + 30 + total / 10)) = total) : 
  total = 60 := by
sorry

end NUMINAMATH_CALUDE_trip_distance_l338_33899


namespace NUMINAMATH_CALUDE_two_digit_integer_problem_l338_33866

theorem two_digit_integer_problem (m n : ℕ) : 
  10 ≤ m ∧ m < 100 ∧ 
  10 ≤ n ∧ n < 100 ∧ 
  m ≠ n ∧
  (m + n) / 2 = (m : ℚ) + n / 100 →
  min m n = 32 := by
sorry

end NUMINAMATH_CALUDE_two_digit_integer_problem_l338_33866


namespace NUMINAMATH_CALUDE_distance_to_focus_is_4_l338_33887

/-- The distance from a point on the parabola y^2 = 4x with x-coordinate 3 to its focus -/
def distance_to_focus (y : ℝ) : ℝ :=
  4

/-- A point P lies on the parabola y^2 = 4x -/
def on_parabola (x y : ℝ) : Prop :=
  y^2 = 4*x

theorem distance_to_focus_is_4 :
  ∀ y : ℝ, on_parabola 3 y → distance_to_focus y = 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_is_4_l338_33887


namespace NUMINAMATH_CALUDE_jo_bob_max_height_l338_33848

/-- Represents the state of a hot air balloon ride -/
structure BalloonRide where
  ascent_rate : ℝ
  descent_rate : ℝ
  first_pull_time : ℝ
  release_time : ℝ
  second_pull_time : ℝ

/-- Calculates the maximum height reached during a balloon ride -/
def max_height (ride : BalloonRide) : ℝ :=
  let first_ascent := ride.ascent_rate * ride.first_pull_time
  let descent := ride.descent_rate * ride.release_time
  let second_ascent := ride.ascent_rate * ride.second_pull_time
  first_ascent - descent + second_ascent

/-- Theorem stating the maximum height reached in Jo-Bob's balloon ride -/
theorem jo_bob_max_height :
  let ride : BalloonRide := {
    ascent_rate := 50,
    descent_rate := 10,
    first_pull_time := 15,
    release_time := 10,
    second_pull_time := 15
  }
  max_height ride = 1400 := by
  sorry


end NUMINAMATH_CALUDE_jo_bob_max_height_l338_33848


namespace NUMINAMATH_CALUDE_f_composition_equals_14_l338_33813

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x + 2

-- State the theorem
theorem f_composition_equals_14 : f (1 + g 3) = 14 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_14_l338_33813


namespace NUMINAMATH_CALUDE_number_problem_l338_33897

theorem number_problem (x : ℝ) : (1 / 5 * x - 5 = 5) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l338_33897


namespace NUMINAMATH_CALUDE_average_income_proof_l338_33873

def cab_driver_income : List ℝ := [200, 150, 750, 400, 500]

theorem average_income_proof :
  (List.sum cab_driver_income) / (List.length cab_driver_income) = 400 := by
  sorry

end NUMINAMATH_CALUDE_average_income_proof_l338_33873


namespace NUMINAMATH_CALUDE_football_games_per_month_l338_33828

theorem football_games_per_month 
  (total_games : ℕ) 
  (num_months : ℕ) 
  (h1 : total_games = 323) 
  (h2 : num_months = 17) 
  (h3 : total_games % num_months = 0) : 
  total_games / num_months = 19 := by
sorry


end NUMINAMATH_CALUDE_football_games_per_month_l338_33828


namespace NUMINAMATH_CALUDE_bread_cost_l338_33843

/-- Proves that the cost of a loaf of bread is $2 given the specified conditions --/
theorem bread_cost (total_budget : ℝ) (candy_cost : ℝ) (turkey_proportion : ℝ) (money_left : ℝ)
  (h1 : total_budget = 32)
  (h2 : candy_cost = 2)
  (h3 : turkey_proportion = 1/3)
  (h4 : money_left = 18)
  : ∃ (bread_cost : ℝ),
    bread_cost = 2 ∧
    money_left = total_budget - candy_cost - turkey_proportion * (total_budget - candy_cost) - bread_cost :=
by
  sorry

end NUMINAMATH_CALUDE_bread_cost_l338_33843


namespace NUMINAMATH_CALUDE_disjoint_sets_imply_a_values_l338_33894

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | (p.2 - 3) / (p.1 - 1) = 2 ∧ p.1 ≠ 1}
def B (a : ℝ) : Set (ℝ × ℝ) := {p | 4 * p.1 + a * p.2 = 16}

-- State the theorem
theorem disjoint_sets_imply_a_values (a : ℝ) :
  A ∩ B a = ∅ → a = -2 ∨ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_sets_imply_a_values_l338_33894


namespace NUMINAMATH_CALUDE_bob_cannot_win_bob_must_choose_nine_l338_33891

/-- Represents the possible game numbers -/
inductive GameNumber
| nineteen : GameNumber
| twenty : GameNumber

/-- Represents the possible starting numbers -/
inductive StartNumber
| nine : StartNumber
| ten : StartNumber

/-- Represents a player in the game -/
inductive Player
| alice : Player
| bob : Player

/-- Represents the state of the game after each turn -/
structure GameState where
  current_sum : ℕ
  current_player : Player

/-- Represents the outcome of the game -/
inductive GameOutcome
| alice_wins : GameOutcome
| bob_wins : GameOutcome
| draw : GameOutcome

/-- Simulates a single turn of the game -/
def play_turn (state : GameState) (alice_number : GameNumber) (bob_number : GameNumber) : GameState :=
  sorry

/-- Simulates the entire game until completion -/
def play_game (start : StartNumber) (alice_number : GameNumber) (bob_number : GameNumber) : GameOutcome :=
  sorry

/-- Theorem stating that Bob cannot win -/
theorem bob_cannot_win :
  ∀ (start : StartNumber) (alice_number bob_number : GameNumber),
    play_game start alice_number bob_number ≠ GameOutcome.bob_wins :=
  sorry

/-- Theorem stating that Bob must choose 9 to prevent Alice from winning -/
theorem bob_must_choose_nine :
  (∀ (alice_number bob_number : GameNumber),
    play_game StartNumber.nine alice_number bob_number ≠ GameOutcome.alice_wins) ∧
  (∃ (alice_number bob_number : GameNumber),
    play_game StartNumber.ten alice_number bob_number = GameOutcome.alice_wins) :=
  sorry

end NUMINAMATH_CALUDE_bob_cannot_win_bob_must_choose_nine_l338_33891


namespace NUMINAMATH_CALUDE_shopping_money_theorem_l338_33850

theorem shopping_money_theorem (initial_money : ℚ) : 
  (initial_money - 3/7 * initial_money - 2/5 * initial_money - 1/4 * initial_money = 24) →
  (initial_money - 1/2 * initial_money - 1/3 * initial_money = 36) →
  (initial_money + initial_money) / 2 = 458.18 := by
sorry

end NUMINAMATH_CALUDE_shopping_money_theorem_l338_33850


namespace NUMINAMATH_CALUDE_range_of_a_for_P_and_Q_l338_33833

theorem range_of_a_for_P_and_Q (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) ↔ 
  a ≤ -2 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_P_and_Q_l338_33833


namespace NUMINAMATH_CALUDE_race_participants_l338_33852

/-- Represents a bicycle race with participants. -/
structure BicycleRace where
  participants : ℕ
  petya_position : ℕ
  vasya_position : ℕ
  vasya_position_from_end : ℕ

/-- The bicycle race satisfies the given conditions. -/
def valid_race (race : BicycleRace) : Prop :=
  race.petya_position = 10 ∧
  race.vasya_position = race.petya_position - 1 ∧
  race.vasya_position_from_end = 15

theorem race_participants (race : BicycleRace) :
  valid_race race → race.participants = 23 := by
  sorry

#check race_participants

end NUMINAMATH_CALUDE_race_participants_l338_33852


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l338_33853

theorem complex_magnitude_problem (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l338_33853


namespace NUMINAMATH_CALUDE_equation_solution_l338_33827

theorem equation_solution :
  ∀ x : ℝ, x + 36 / (x - 4) = -9 ↔ x = 0 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l338_33827


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l338_33822

theorem ratio_equation_solution (a b : ℚ) 
  (h1 : b / a = 4) 
  (h2 : b = 20 - 7 * a) : 
  a = 20 / 11 := by
sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l338_33822


namespace NUMINAMATH_CALUDE_keegan_class_count_l338_33880

/-- Calculates the number of classes Keegan is taking given his school schedule --/
theorem keegan_class_count :
  ∀ (total_school_time : ℝ) 
    (history_chem_time : ℝ) 
    (avg_other_class_time : ℝ),
  total_school_time = 7.5 →
  history_chem_time = 1.5 →
  avg_other_class_time = 72 / 60 →
  (total_school_time - history_chem_time) / avg_other_class_time + 2 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_keegan_class_count_l338_33880


namespace NUMINAMATH_CALUDE_compaction_percentage_is_twenty_l338_33831

/-- Represents the compaction problem with cans -/
structure CanCompaction where
  num_cans : ℕ
  space_before : ℕ
  total_space_after : ℕ

/-- Calculates the percentage of original space each can takes up after compaction -/
def compaction_percentage (c : CanCompaction) : ℚ :=
  (c.total_space_after : ℚ) / ((c.num_cans * c.space_before) : ℚ) * 100

/-- Theorem stating that for the given conditions, the compaction percentage is 20% -/
theorem compaction_percentage_is_twenty (c : CanCompaction) 
  (h1 : c.num_cans = 60)
  (h2 : c.space_before = 30)
  (h3 : c.total_space_after = 360) : 
  compaction_percentage c = 20 := by
  sorry

end NUMINAMATH_CALUDE_compaction_percentage_is_twenty_l338_33831


namespace NUMINAMATH_CALUDE_class_average_problem_l338_33855

theorem class_average_problem (total_students : ℕ) (high_scorers : ℕ) (zero_scorers : ℕ) 
  (high_score : ℕ) (class_average : ℚ) :
  total_students = 28 →
  high_scorers = 4 →
  zero_scorers = 3 →
  high_score = 95 →
  class_average = 47.32142857142857 →
  let remaining_students := total_students - high_scorers - zero_scorers
  let total_score := total_students * class_average
  let high_score_total := high_scorers * high_score
  let remaining_score := total_score - high_score_total
  remaining_score / remaining_students = 45 := by
    sorry

#eval (28 : ℚ) * 47.32142857142857 -- To verify the total score

end NUMINAMATH_CALUDE_class_average_problem_l338_33855


namespace NUMINAMATH_CALUDE_sally_carl_owe_amount_l338_33819

def total_promised : ℝ := 400
def amount_received : ℝ := 285
def amy_owes : ℝ := 30

theorem sally_carl_owe_amount :
  ∃ (s : ℝ), 
    s > 0 ∧
    2 * s + amy_owes + amy_owes / 2 = total_promised - amount_received ∧
    s = 35 := by sorry

end NUMINAMATH_CALUDE_sally_carl_owe_amount_l338_33819


namespace NUMINAMATH_CALUDE_min_cubes_in_block_l338_33859

theorem min_cubes_in_block (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 252 → 
  l * m * n ≥ 392 ∧ 
  (∃ (l' m' n' : ℕ), (l' - 1) * (m' - 1) * (n' - 1) = 252 ∧ l' * m' * n' = 392) :=
by sorry

end NUMINAMATH_CALUDE_min_cubes_in_block_l338_33859


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l338_33830

/-- In triangle ABC, prove that given specific conditions, angle A and the area of the triangle can be determined. -/
theorem triangle_ABC_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given condition
  2 * Real.cos A * (b * Real.cos C + c * Real.cos B) = a →
  -- Additional conditions
  a = Real.sqrt 7 →
  b + c = 5 →
  -- Conclusions
  A = π / 3 ∧ 
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l338_33830


namespace NUMINAMATH_CALUDE_mary_cake_flour_l338_33809

/-- Given a recipe that requires a certain amount of flour and the amount already added,
    calculate the remaining amount of flour needed. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- The problem statement -/
theorem mary_cake_flour : remaining_flour 9 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mary_cake_flour_l338_33809


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_l338_33815

theorem ac_squared_gt_bc_squared (a b c : ℝ) : a > b → a * c^2 > b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_l338_33815


namespace NUMINAMATH_CALUDE_problem_statement_l338_33810

theorem problem_statement : (-1)^2023 - Real.tan (π/3) + (Real.sqrt 5 - 1)^0 + |-(Real.sqrt 3)| = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l338_33810


namespace NUMINAMATH_CALUDE_monotonic_function_a_range_l338_33823

/-- Given that f(x) = ln x + a/x is monotonically increasing on [2, +∞), 
    prove that the range of values for a is (-∞, 2] -/
theorem monotonic_function_a_range (a : ℝ) :
  (∀ x ≥ 2, Monotone (fun x => Real.log x + a / x)) ↔ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_monotonic_function_a_range_l338_33823


namespace NUMINAMATH_CALUDE_no_valid_flippy_numbers_l338_33898

/-- A five-digit flippy number is a number of the form ababa or babab where a and b are distinct digits -/
def is_flippy_number (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ a < 10 ∧ b < 10 ∧
  ((n = a * 10000 + b * 1000 + a * 100 + b * 10 + a) ∨
   (n = b * 10000 + a * 1000 + b * 100 + a * 10 + b))

/-- The sum of digits of a five-digit flippy number -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

/-- There are no five-digit flippy numbers that are divisible by 11 and have a sum of digits divisible by 6 -/
theorem no_valid_flippy_numbers :
  ¬ ∃ n : ℕ, is_flippy_number n ∧ n % 11 = 0 ∧ (sum_of_digits n) % 6 = 0 := by
  sorry

#check no_valid_flippy_numbers

end NUMINAMATH_CALUDE_no_valid_flippy_numbers_l338_33898


namespace NUMINAMATH_CALUDE_cube_root_of_square_l338_33857

theorem cube_root_of_square (x : ℝ) : x > 0 → (x^2)^(1/3) = x^(2/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_of_square_l338_33857


namespace NUMINAMATH_CALUDE_area_triangle_EYH_l338_33872

/-- Represents a trapezoid with bases and diagonals -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Theorem: Area of triangle EYH in trapezoid EFGH -/
theorem area_triangle_EYH (EFGH : Trapezoid) (h1 : EFGH.base1 = 15) (h2 : EFGH.base2 = 35) (h3 : EFGH.area = 400) :
  ∃ (area_EYH : ℝ), area_EYH = 84 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_EYH_l338_33872


namespace NUMINAMATH_CALUDE_sport_water_amount_l338_33854

/-- Represents the ratios in a flavored drink formulation -/
structure DrinkFormulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the drink -/
def standard : DrinkFormulation := ⟨1, 12, 30⟩

/-- The sport formulation of the drink -/
def sport : DrinkFormulation :=
  ⟨standard.flavoring, standard.corn_syrup / 3, standard.water * 2⟩

theorem sport_water_amount (corn_syrup_amount : ℚ) :
  corn_syrup_amount = 1 →
  (corn_syrup_amount * sport.water / sport.corn_syrup) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sport_water_amount_l338_33854


namespace NUMINAMATH_CALUDE_spinner_probability_l338_33845

theorem spinner_probability (p_A p_B p_C p_DE : ℚ) : 
  p_A = 1/3 →
  p_B = 1/6 →
  p_C = p_DE →
  p_A + p_B + p_C + p_DE = 1 →
  p_C = 1/4 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l338_33845


namespace NUMINAMATH_CALUDE_quadratic_polynomial_proof_l338_33814

theorem quadratic_polynomial_proof :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = (1/3) * (2*x^2 - 4*x + 9)) ∧
    q (-2) = 8 ∧
    q 1 = 2 ∧
    q 3 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_proof_l338_33814


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_line_l_equation_l338_33885

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + m * y - 6 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + y - 3 = 0

-- Define perpendicularity of lines
def perpendicular (m : ℝ) : Prop := 
  (m = 0) ∨ (m ≠ 0 ∧ (m + 2) / m * m = -1)

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (1, 2 * m)

-- Define the line l
def l (k : ℝ) (x y : ℝ) : Prop := y - 2 = k * (x - 1)

-- Define the intercept condition
def intercept_condition (k : ℝ) : Prop :=
  (k - 2) / k = 2 * (2 - k)

theorem perpendicular_lines_m (m : ℝ) : 
  perpendicular m → m = -3 ∨ m = 0 :=
sorry

theorem line_l_equation (m : ℝ) :
  l₂ m 1 (2 * m) →
  (∃ k, l k 1 (2 * m) ∧ intercept_condition k) →
  (∀ x y, l 2 x y ∨ l (-1/2) x y) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_line_l_equation_l338_33885


namespace NUMINAMATH_CALUDE_franks_reading_time_l338_33849

/-- Represents the problem of calculating Frank's effective reading time --/
theorem franks_reading_time (total_pages : ℕ) (reading_speed : ℕ) (total_days : ℕ) :
  total_pages = 2345 →
  reading_speed = 50 →
  total_days = 34 →
  ∃ (effective_time : ℚ),
    effective_time > 2.03 ∧
    effective_time < 2.05 ∧
    effective_time = (total_pages : ℚ) / reading_speed / ((2 * total_days : ℚ) / 3) :=
by sorry

end NUMINAMATH_CALUDE_franks_reading_time_l338_33849


namespace NUMINAMATH_CALUDE_first_battery_was_voltaic_pile_l338_33886

/-- Represents a battery -/
structure Battery where
  year : Nat
  creator : String
  components : List String

/-- The first recognized battery in the world -/
def first_battery : Battery :=
  { year := 1800,
    creator := "Alessandro Volta",
    components := ["different metals", "electrolyte"] }

/-- Theorem stating that the first recognized battery was the Voltaic pile -/
theorem first_battery_was_voltaic_pile :
  first_battery.year = 1800 ∧
  first_battery.creator = "Alessandro Volta" ∧
  first_battery.components = ["different metals", "electrolyte"] :=
by sorry

#check first_battery_was_voltaic_pile

end NUMINAMATH_CALUDE_first_battery_was_voltaic_pile_l338_33886


namespace NUMINAMATH_CALUDE_total_cost_is_51_l338_33802

/-- The cost of a single shirt in dollars -/
def shirt_cost : ℕ := 5

/-- The cost of a single hat in dollars -/
def hat_cost : ℕ := 4

/-- The cost of a single pair of jeans in dollars -/
def jeans_cost : ℕ := 10

/-- The number of shirts to be purchased -/
def num_shirts : ℕ := 3

/-- The number of hats to be purchased -/
def num_hats : ℕ := 4

/-- The number of pairs of jeans to be purchased -/
def num_jeans : ℕ := 2

/-- Theorem stating that the total cost of the purchase is $51 -/
theorem total_cost_is_51 : 
  num_shirts * shirt_cost + num_hats * hat_cost + num_jeans * jeans_cost = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_51_l338_33802


namespace NUMINAMATH_CALUDE_max_sum_of_digits_watch_l338_33824

-- Define the type for hours and minutes
def Hour := Fin 12
def Minute := Fin 60

-- Function to calculate the sum of digits
def sumOfDigits (n : Nat) : Nat :=
  let digits := n.repr.toList.map (λc => c.toNat - '0'.toNat)
  digits.sum

-- Define the theorem
theorem max_sum_of_digits_watch :
  ∃ (h : Hour) (m : Minute),
    ∀ (h' : Hour) (m' : Minute),
      sumOfDigits (h.val + 1) + sumOfDigits m.val ≥ 
      sumOfDigits (h'.val + 1) + sumOfDigits m'.val ∧
      sumOfDigits (h.val + 1) + sumOfDigits m.val = 23 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_watch_l338_33824


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l338_33851

/-- For a quadratic equation x^2 + 2x - k = 0 to have two distinct real roots, k must be greater than -1. -/
theorem quadratic_distinct_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x - k = 0 ∧ y^2 + 2*y - k = 0) ↔ k > -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l338_33851


namespace NUMINAMATH_CALUDE_handshake_count_l338_33841

theorem handshake_count (num_gremlins num_imps : ℕ) (h1 : num_gremlins = 30) (h2 : num_imps = 20) :
  let gremlin_handshakes := num_gremlins.choose 2
  let gremlin_imp_handshakes := num_gremlins * num_imps
  gremlin_handshakes + gremlin_imp_handshakes = 1035 := by
  sorry

#check handshake_count

end NUMINAMATH_CALUDE_handshake_count_l338_33841


namespace NUMINAMATH_CALUDE_team_combinations_l338_33895

theorem team_combinations (n m : ℕ) (h1 : n = 7) (h2 : m = 4) : 
  Nat.choose n m = 35 := by
  sorry

end NUMINAMATH_CALUDE_team_combinations_l338_33895


namespace NUMINAMATH_CALUDE_complex_square_condition_l338_33834

theorem complex_square_condition (a b : ℝ) : 
  (∃ a b : ℝ, (Complex.I : ℂ)^2 = -1 ∧ (a + b * Complex.I)^2 = 2 * Complex.I ∧ ¬(a = 1 ∧ b = 1)) ∧
  ((a = 1 ∧ b = 1) → (a + b * Complex.I)^2 = 2 * Complex.I) :=
sorry

end NUMINAMATH_CALUDE_complex_square_condition_l338_33834


namespace NUMINAMATH_CALUDE_phil_change_is_seven_l338_33858

/-- The change Phil received after buying apples -/
def change_received : ℚ :=
  let number_of_apples : ℕ := 4
  let cost_per_apple : ℚ := 75 / 100
  let amount_paid : ℚ := 10
  amount_paid - (number_of_apples * cost_per_apple)

/-- Proof that Phil received $7.00 in change -/
theorem phil_change_is_seven : change_received = 7 := by
  sorry

end NUMINAMATH_CALUDE_phil_change_is_seven_l338_33858


namespace NUMINAMATH_CALUDE_fixed_point_on_parabola_l338_33876

theorem fixed_point_on_parabola (a b c : ℝ) 
  (h1 : |a| ≥ |b - c|) 
  (h2 : |b| ≥ |a + c|) 
  (h3 : |c| ≥ |a - b|) : 
  a * (-1)^2 + b * (-1) + c = 0 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_on_parabola_l338_33876


namespace NUMINAMATH_CALUDE_six_eight_ten_pythagorean_triple_l338_33826

/-- A Pythagorean triple is a set of three positive integers (a, b, c) where a² + b² = c² --/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The set (6, 8, 10) is a Pythagorean triple --/
theorem six_eight_ten_pythagorean_triple : is_pythagorean_triple 6 8 10 := by
  sorry

end NUMINAMATH_CALUDE_six_eight_ten_pythagorean_triple_l338_33826


namespace NUMINAMATH_CALUDE_barbie_gave_four_pairs_l338_33803

/-- The number of pairs of earrings Barbie bought -/
def total_earrings : ℕ := 12

/-- The number of pairs of earrings Barbie gave to Alissa -/
def earrings_given : ℕ := 4

/-- Alissa's total collection after receiving earrings from Barbie -/
def alissa_total (x : ℕ) : ℕ := 3 * x

theorem barbie_gave_four_pairs :
  earrings_given = 4 ∧
  alissa_total earrings_given + earrings_given = total_earrings :=
by sorry

end NUMINAMATH_CALUDE_barbie_gave_four_pairs_l338_33803


namespace NUMINAMATH_CALUDE_sum_specific_terms_l338_33893

/-- Given a sequence {a_n} where S_n = n^2 - 1 for n ∈ ℕ+, prove a_1 + a_3 + a_5 + a_7 + a_9 = 44 -/
theorem sum_specific_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ+, S n = n^2 - 1) → 
  (∀ n : ℕ+, S n - S (n-1) = a n) → 
  a 1 + a 3 + a 5 + a 7 + a 9 = 44 := by
sorry

end NUMINAMATH_CALUDE_sum_specific_terms_l338_33893


namespace NUMINAMATH_CALUDE_inequality_solution_set_l338_33837

theorem inequality_solution_set (x : ℝ) :
  (3 - 2*x - x^2 ≤ 0) ↔ (x ≤ -3 ∨ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l338_33837


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_times_two_l338_33825

theorem sin_sixty_degrees_times_two : 2 * Real.sin (π / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_times_two_l338_33825


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_rational_inequality_solution_l338_33808

-- Part 1
theorem quadratic_inequality_solution (b c : ℝ) :
  (∀ x, 5 * x^2 - b * x + c < 0 ↔ -1 < x ∧ x < 3) →
  b + c = -5 :=
sorry

-- Part 2
theorem rational_inequality_solution :
  {x : ℝ | (2 * x - 5) / (x + 4) ≥ 0} = {x : ℝ | x ≥ 5/2 ∨ x < -4} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_rational_inequality_solution_l338_33808


namespace NUMINAMATH_CALUDE_x_equals_five_l338_33884

/-- A composite rectangular figure with specific segment lengths -/
structure CompositeRectangle where
  top_left : ℝ
  top_middle : ℝ
  top_right : ℝ
  bottom_left : ℝ
  bottom_middle : ℝ
  bottom_right : ℝ

/-- The theorem stating that X equals 5 in the given composite rectangle -/
theorem x_equals_five (r : CompositeRectangle) 
  (h1 : r.top_left = 3)
  (h2 : r.top_right = 4)
  (h3 : r.bottom_left = 5)
  (h4 : r.bottom_middle = 7)
  (h5 : r.top_middle = r.bottom_right)
  (h6 : r.top_left + r.top_middle + r.top_right = r.bottom_left + r.bottom_middle + r.bottom_right) :
  r.top_middle = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_five_l338_33884


namespace NUMINAMATH_CALUDE_bridge_crossing_time_l338_33807

/-- Proves that a man walking at 5 km/hr takes 15 minutes to cross a 1250-meter bridge -/
theorem bridge_crossing_time :
  let walking_speed : ℝ := 5  -- km/hr
  let bridge_length : ℝ := 1250  -- meters
  let crossing_time : ℝ := 15  -- minutes
  
  walking_speed * 1000 / 60 * crossing_time = bridge_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_crossing_time_l338_33807


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l338_33892

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℝ),
    (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 →
      5 * x^2 / ((x - 4) * (x - 2)^3) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^3) ∧
    P = 10 ∧ Q = -10 ∧ R = -10 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l338_33892


namespace NUMINAMATH_CALUDE_ratio_percentage_difference_l338_33877

theorem ratio_percentage_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_ratio : x / 8 = y / 7) : (y - x) / x = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_percentage_difference_l338_33877


namespace NUMINAMATH_CALUDE_least_common_denominator_l338_33870

theorem least_common_denominator : Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l338_33870


namespace NUMINAMATH_CALUDE_probability_distance_sqrt2_over_2_l338_33875

/-- A point on a unit square, either a vertex or the center -/
inductive SquarePoint
  | vertex : Fin 4 → SquarePoint
  | center : SquarePoint

/-- The distance between two points on a unit square -/
def distance (p q : SquarePoint) : ℝ :=
  sorry

/-- The set of all possible pairs of points -/
def allPairs : Finset (SquarePoint × SquarePoint) :=
  sorry

/-- The set of pairs of points with distance √2/2 -/
def pairsWithDistance : Finset (SquarePoint × SquarePoint) :=
  sorry

theorem probability_distance_sqrt2_over_2 :
  (Finset.card pairsWithDistance : ℚ) / (Finset.card allPairs : ℚ) = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_distance_sqrt2_over_2_l338_33875


namespace NUMINAMATH_CALUDE_male_athletes_to_sample_l338_33844

theorem male_athletes_to_sample (total_athletes : ℕ) (female_athletes : ℕ) (selection_prob : ℚ) :
  total_athletes = 98 →
  female_athletes = 42 →
  selection_prob = 2 / 7 →
  (total_athletes - female_athletes) * selection_prob = 16 := by
  sorry

end NUMINAMATH_CALUDE_male_athletes_to_sample_l338_33844


namespace NUMINAMATH_CALUDE_vector_dot_product_l338_33820

/-- Given two 2D vectors a and b, prove that their dot product is -18. -/
theorem vector_dot_product (a b : ℝ × ℝ) : 
  a = (1, -3) → b = (3, 7) → a.1 * b.1 + a.2 * b.2 = -18 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l338_33820


namespace NUMINAMATH_CALUDE_negative_390_same_terminal_side_as_330_l338_33821

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem negative_390_same_terminal_side_as_330 :
  same_terminal_side (-390) 330 :=
sorry

end NUMINAMATH_CALUDE_negative_390_same_terminal_side_as_330_l338_33821


namespace NUMINAMATH_CALUDE_spade_operation_result_l338_33804

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_operation_result : spade 1.5 (spade 2.5 (spade 4.5 6)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_spade_operation_result_l338_33804


namespace NUMINAMATH_CALUDE_train_speed_calculation_l338_33874

/-- Given two trains A and B moving towards each other, calculate the speed of train B. -/
theorem train_speed_calculation (length_A length_B : ℝ) (speed_A : ℝ) (crossing_time : ℝ) 
  (h1 : length_A = 225) 
  (h2 : length_B = 150) 
  (h3 : speed_A = 54) 
  (h4 : crossing_time = 15) : 
  (((length_A + length_B) / crossing_time) * (3600 / 1000) - speed_A) = 36 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l338_33874


namespace NUMINAMATH_CALUDE_checkerboard_square_count_l338_33860

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  topLeftRow : Nat
  topLeftCol : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 10

/-- Checks if a square contains at least 5 black squares -/
def containsAtLeast5Black (s : Square) : Bool :=
  sorry

/-- Counts the number of valid squares of a given size -/
def countValidSquares (size : Nat) : Nat :=
  sorry

/-- Counts the total number of squares containing at least 5 black squares -/
def totalValidSquares : Nat :=
  sorry

/-- Main theorem: The number of distinct squares containing at least 5 black squares is 172 -/
theorem checkerboard_square_count : totalValidSquares = 172 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_square_count_l338_33860


namespace NUMINAMATH_CALUDE_square_inequality_negative_l338_33812

theorem square_inequality_negative (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_negative_l338_33812


namespace NUMINAMATH_CALUDE_power_inequality_l338_33879

theorem power_inequality (n : ℕ) (h : n > 1) : n ^ n > (n + 1) ^ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l338_33879
