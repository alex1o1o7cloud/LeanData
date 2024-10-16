import Mathlib

namespace NUMINAMATH_CALUDE_multiply_82519_by_9999_l1974_197430

theorem multiply_82519_by_9999 : 82519 * 9999 = 825117481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_82519_by_9999_l1974_197430


namespace NUMINAMATH_CALUDE_figure_area_l1974_197486

theorem figure_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ) 
  (h1 : rect1_width = 7 ∧ rect1_height = 7)
  (h2 : rect2_width = 3 ∧ rect2_height = 2)
  (h3 : rect3_width = 4 ∧ rect3_height = 4) :
  rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height = 71 := by
sorry

end NUMINAMATH_CALUDE_figure_area_l1974_197486


namespace NUMINAMATH_CALUDE_john_mileage_conversion_l1974_197438

/-- Converts a base-8 number represented as a list of digits to its base-10 equivalent -/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

/-- The base-8 representation of John's mileage -/
def johnMileageBase8 : List Nat := [3, 4, 5, 2]

/-- Theorem: John's mileage in base-10 is 1834 miles -/
theorem john_mileage_conversion :
  base8ToBase10 johnMileageBase8 = 1834 := by
  sorry

end NUMINAMATH_CALUDE_john_mileage_conversion_l1974_197438


namespace NUMINAMATH_CALUDE_jennas_driving_speed_l1974_197471

/-- Proves that Jenna's driving speed is 50 miles per hour given the road trip conditions -/
theorem jennas_driving_speed 
  (total_distance : ℝ) 
  (jenna_distance : ℝ) 
  (friend_distance : ℝ)
  (total_time : ℝ) 
  (break_time : ℝ) 
  (friend_speed : ℝ) 
  (h1 : total_distance = jenna_distance + friend_distance)
  (h2 : total_distance = 300)
  (h3 : jenna_distance = 200)
  (h4 : friend_distance = 100)
  (h5 : total_time = 10)
  (h6 : break_time = 1)
  (h7 : friend_speed = 20) : 
  jenna_distance / (total_time - break_time - friend_distance / friend_speed) = 50 := by
  sorry

#check jennas_driving_speed

end NUMINAMATH_CALUDE_jennas_driving_speed_l1974_197471


namespace NUMINAMATH_CALUDE_equality_from_quadratic_equation_l1974_197453

theorem equality_from_quadratic_equation 
  (m n p : ℝ) 
  (h_nonzero : m ≠ 0 ∧ n ≠ 0 ∧ p ≠ 0) 
  (h_eq : (1/4) * (m - n)^2 = (p - n) * (m - p)) : 
  2 * p = m + n := by
sorry

end NUMINAMATH_CALUDE_equality_from_quadratic_equation_l1974_197453


namespace NUMINAMATH_CALUDE_auction_bidding_l1974_197475

theorem auction_bidding (price_increase : ℕ) (start_price : ℕ) (end_price : ℕ) (num_bidders : ℕ) :
  price_increase = 5 →
  start_price = 15 →
  end_price = 65 →
  num_bidders = 2 →
  (end_price - start_price) / price_increase / num_bidders = 5 :=
by sorry

end NUMINAMATH_CALUDE_auction_bidding_l1974_197475


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1974_197442

theorem consecutive_integers_sum (n : ℤ) : 
  (n - 1) * n * (n + 1) = -336 ∧ n < 0 → (n - 1) + n + (n + 1) = -21 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1974_197442


namespace NUMINAMATH_CALUDE_remaining_bottle_caps_l1974_197424

-- Define the initial number of bottle caps
def initial_caps : ℕ := 34

-- Define the number of bottle caps eaten
def eaten_caps : ℕ := 8

-- Theorem to prove
theorem remaining_bottle_caps : initial_caps - eaten_caps = 26 := by
  sorry

end NUMINAMATH_CALUDE_remaining_bottle_caps_l1974_197424


namespace NUMINAMATH_CALUDE_different_set_l1974_197477

def set_A : Set ℝ := {x | x = 1}
def set_B : Set ℝ := {x | x^2 = 1}
def set_C : Set ℝ := {1}
def set_D : Set ℝ := {y | (y - 1)^2 = 0}

theorem different_set :
  (set_A = set_C) ∧ (set_A = set_D) ∧ (set_C = set_D) ∧ (set_B ≠ set_A) ∧ (set_B ≠ set_C) ∧ (set_B ≠ set_D) :=
sorry

end NUMINAMATH_CALUDE_different_set_l1974_197477


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l1974_197456

theorem cubic_polynomial_root (d e : ℚ) :
  (3 - Real.sqrt 5 : ℂ) ^ 3 + d * (3 - Real.sqrt 5 : ℂ) + e = 0 →
  (-6 : ℂ) ^ 3 + d * (-6 : ℂ) + e = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l1974_197456


namespace NUMINAMATH_CALUDE_triangle_theorem_l1974_197418

/-- Theorem about a triangle ABC with specific conditions -/
theorem triangle_theorem (a b c A B C : ℝ) : 
  -- Given conditions
  (2 * b * Real.cos C = 2 * a - c) →  -- Condition from the problem
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area condition
  (b = 2) →  -- Given value of b
  -- Conclusions to prove
  (B = Real.pi / 3) ∧  -- 60 degrees in radians
  (a = 2) ∧ 
  (c = 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1974_197418


namespace NUMINAMATH_CALUDE_best_fit_model_l1974_197432

/-- Represents a regression model with its R² value -/
structure RegressionModel where
  r_squared : ℝ

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

theorem best_fit_model (models : List RegressionModel) 
  (h1 : RegressionModel.mk 0.95 ∈ models)
  (h2 : RegressionModel.mk 0.70 ∈ models)
  (h3 : RegressionModel.mk 0.55 ∈ models)
  (h4 : RegressionModel.mk 0.30 ∈ models)
  (h5 : models.length = 4) :
  has_best_fit (RegressionModel.mk 0.95) models :=
sorry

end NUMINAMATH_CALUDE_best_fit_model_l1974_197432


namespace NUMINAMATH_CALUDE_color_films_count_l1974_197406

theorem color_films_count (x y : ℝ) (h : x > 0) :
  let total_bw := 40 * x
  let selected_bw := 2 * y / 5
  let fraction_color := 0.9615384615384615
  let color_films := (fraction_color * (selected_bw + color_films)) / (1 - fraction_color)
  color_films = 10 * y := by
  sorry

end NUMINAMATH_CALUDE_color_films_count_l1974_197406


namespace NUMINAMATH_CALUDE_bom_watermelon_seeds_l1974_197427

/-- Given the number of watermelon seeds for Bom, Gwi, and Yeon, prove that Bom has 300 seeds. -/
theorem bom_watermelon_seeds :
  ∀ (bom gwi yeon : ℕ),
  gwi = bom + 40 →
  yeon = 3 * gwi →
  bom + gwi + yeon = 1660 →
  bom = 300 := by
sorry

end NUMINAMATH_CALUDE_bom_watermelon_seeds_l1974_197427


namespace NUMINAMATH_CALUDE_cone_rolling_ratio_l1974_197414

/-- 
Theorem: For a right circular cone with base radius r and height h, 
if the cone makes 23 complete rotations when rolled on its side, 
then h/r = 4√33.
-/
theorem cone_rolling_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (2 * Real.pi * Real.sqrt (r^2 + h^2) = 46 * Real.pi * r) → 
  (h / r = 4 * Real.sqrt 33) := by
sorry

end NUMINAMATH_CALUDE_cone_rolling_ratio_l1974_197414


namespace NUMINAMATH_CALUDE_index_cards_per_student_l1974_197492

theorem index_cards_per_student 
  (num_classes : ℕ) 
  (students_per_class : ℕ) 
  (total_packs : ℕ) 
  (h1 : num_classes = 6) 
  (h2 : students_per_class = 30) 
  (h3 : total_packs = 360) : 
  total_packs / (num_classes * students_per_class) = 2 := by
  sorry

end NUMINAMATH_CALUDE_index_cards_per_student_l1974_197492


namespace NUMINAMATH_CALUDE_sharon_trip_distance_l1974_197479

def normal_time : ℝ := 200
def reduced_speed_time : ℝ := 310
def speed_reduction : ℝ := 30

def trip_distance : ℝ := 220

theorem sharon_trip_distance :
  let normal_speed := trip_distance / normal_time
  let reduced_speed := normal_speed - speed_reduction / 60
  (trip_distance / 3) / normal_speed + (2 * trip_distance / 3) / reduced_speed = reduced_speed_time :=
by sorry

end NUMINAMATH_CALUDE_sharon_trip_distance_l1974_197479


namespace NUMINAMATH_CALUDE_square_circle_union_area_l1974_197443

/-- The area of the union of a square and a circle with specific properties -/
theorem square_circle_union_area :
  let square_side : ℝ := 12
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let overlap_area : ℝ := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 144 + 108 * π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l1974_197443


namespace NUMINAMATH_CALUDE_water_added_fourth_hour_l1974_197450

-- Define the water tank scenario
def water_tank_scenario (initial_water : ℝ) (loss_rate : ℝ) (added_third_hour : ℝ) (added_fourth_hour : ℝ) : ℝ :=
  initial_water - 4 * loss_rate + added_third_hour + added_fourth_hour

-- Theorem statement
theorem water_added_fourth_hour :
  ∃ (added_fourth_hour : ℝ),
    water_tank_scenario 40 2 1 added_fourth_hour = 36 ∧
    added_fourth_hour = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_water_added_fourth_hour_l1974_197450


namespace NUMINAMATH_CALUDE_marble_difference_l1974_197473

theorem marble_difference (total : ℕ) (yellow : ℕ) (blue_ratio : ℕ) (red_ratio : ℕ)
  (h_total : total = 19)
  (h_yellow : yellow = 5)
  (h_blue_ratio : blue_ratio = 3)
  (h_red_ratio : red_ratio = 4) :
  let remaining : ℕ := total - yellow
  let share : ℕ := remaining / (blue_ratio + red_ratio)
  let red : ℕ := red_ratio * share
  red - yellow = 3 := by sorry

end NUMINAMATH_CALUDE_marble_difference_l1974_197473


namespace NUMINAMATH_CALUDE_f_simplification_f_value_in_second_quadrant_l1974_197483

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (7 * Real.pi - α) * Real.cos (α + 3 * Real.pi / 2) * Real.cos (3 * Real.pi + α)) /
  (Real.sin (α - 3 * Real.pi / 2) * Real.cos (α + 5 * Real.pi / 2) * Real.tan (α - 5 * Real.pi))

theorem f_simplification (α : ℝ) : f α = Real.cos α := by sorry

theorem f_value_in_second_quadrant (α : ℝ) 
  (h1 : π < α ∧ α < 3 * π / 2) 
  (h2 : Real.cos (3 * Real.pi / 2 + α) = 1 / 7) : 
  f α = -4 * Real.sqrt 3 / 7 := by sorry

end NUMINAMATH_CALUDE_f_simplification_f_value_in_second_quadrant_l1974_197483


namespace NUMINAMATH_CALUDE_dot_product_of_vectors_l1974_197461

theorem dot_product_of_vectors (a b : ℝ × ℝ) 
  (h1 : a + b = (1, -3)) 
  (h2 : a - b = (3, 7)) : 
  a.1 * b.1 + a.2 * b.2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_vectors_l1974_197461


namespace NUMINAMATH_CALUDE_expression_evaluation_l1974_197403

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x) * ((y^3 + 2) / y) + ((x^3 - 2) / y) * ((y^3 - 2) / x) = 
  2 * x * y * (x^2 * y^2) + 8 / (x * y) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1974_197403


namespace NUMINAMATH_CALUDE_percentage_relationship_l1974_197449

theorem percentage_relationship (j p t m n x y : ℕ) (r : ℚ) : 
  j > 0 ∧ p > 0 ∧ t > 0 ∧ m > 0 ∧ n > 0 ∧ x > 0 ∧ y > 0 →
  j = (3 / 4 : ℚ) * p →
  j = (4 / 5 : ℚ) * t →
  t = p - (r / 100) * p →
  m = (11 / 10 : ℚ) * p →
  n = (7 / 10 : ℚ) * m →
  j + p + t = m * n →
  x = (23 / 20 : ℚ) * j →
  y = (4 / 5 : ℚ) * n →
  x * y = (j + p + t)^2 →
  r = (25 / 4 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l1974_197449


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1974_197446

theorem arithmetic_calculations :
  ((-24) - (-15) + (-1) + (-15) = -25) ∧
  ((-27) / (3/2) * (2/3) = -12) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1974_197446


namespace NUMINAMATH_CALUDE_scaling_2_3_to_3_2_l1974_197482

/-- A scaling transformation in 2D space -/
structure ScalingTransformation where
  x_scale : ℝ
  y_scale : ℝ

/-- Apply a scaling transformation to a point -/
def apply_scaling (t : ScalingTransformation) (p : ℝ × ℝ) : ℝ × ℝ :=
  (t.x_scale * p.1, t.y_scale * p.2)

/-- The scaling transformation that changes (2, 3) to (3, 2) -/
theorem scaling_2_3_to_3_2 : 
  ∃ (t : ScalingTransformation), apply_scaling t (2, 3) = (3, 2) ∧ 
    t.x_scale = 3/2 ∧ t.y_scale = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_scaling_2_3_to_3_2_l1974_197482


namespace NUMINAMATH_CALUDE_restaurant_spirits_profit_l1974_197411

/-- Calculates the profit made by a restaurant on a bottle of spirits -/
theorem restaurant_spirits_profit
  (bottle_cost : ℝ)
  (servings_per_bottle : ℕ)
  (price_per_serving : ℝ)
  (h1 : bottle_cost = 30)
  (h2 : servings_per_bottle = 16)
  (h3 : price_per_serving = 8) :
  servings_per_bottle * price_per_serving - bottle_cost = 98 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_spirits_profit_l1974_197411


namespace NUMINAMATH_CALUDE_coprime_20172019_l1974_197489

theorem coprime_20172019 :
  (Nat.gcd 20172019 20172017 = 1) ∧
  (Nat.gcd 20172019 20172018 = 1) ∧
  (Nat.gcd 20172019 20172020 = 1) ∧
  (Nat.gcd 20172019 20172021 = 1) :=
by sorry

end NUMINAMATH_CALUDE_coprime_20172019_l1974_197489


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1974_197472

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Property of geometric sequences: if m + n = p + q, then a_m * a_n = a_p * a_q -/
axiom geometric_property {a : ℕ → ℝ} (h : GeometricSequence a) :
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_problem (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_sum : a 4 + a 8 = -3) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1974_197472


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l1974_197465

/-- Given a square divided into four congruent rectangles, if the perimeter of each rectangle is 32 inches, then the perimeter of the square is 51.2 inches. -/
theorem square_perimeter_from_rectangle_perimeter (s : ℝ) 
  (h1 : s > 0) 
  (h2 : 2 * s + 2 * (s / 4) = 32) : 
  4 * s = 51.2 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l1974_197465


namespace NUMINAMATH_CALUDE_probability_multiple_of_three_l1974_197451

theorem probability_multiple_of_three (n : ℕ) (h : n = 21) :
  (Finset.filter (fun x => x % 3 = 0) (Finset.range n.succ)).card / n = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_of_three_l1974_197451


namespace NUMINAMATH_CALUDE_systemC_is_linear_l1974_197448

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants. -/
def IsLinearEquationInTwoVars (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- A system of two equations in two variables -/
structure SystemOfTwoEquations :=
  (eq1 : ℝ → ℝ → ℝ)
  (eq2 : ℝ → ℝ → ℝ)

/-- Definition of the specific system of equations given in Option C -/
def systemC : SystemOfTwoEquations :=
  { eq1 := λ x y => x - 3,
    eq2 := λ x y => 2 * x - y - 7 }

/-- Theorem stating that the given system is a system of linear equations in two variables -/
theorem systemC_is_linear : 
  IsLinearEquationInTwoVars systemC.eq1 ∧ IsLinearEquationInTwoVars systemC.eq2 :=
sorry

end NUMINAMATH_CALUDE_systemC_is_linear_l1974_197448


namespace NUMINAMATH_CALUDE_nicole_fish_tanks_l1974_197493

/-- Represents the number of fish tanks Nicole has -/
def num_tanks : ℕ := 4

/-- Represents the amount of water (in gallons) needed for each of the first two tanks -/
def water_first_two : ℕ := 8

/-- Represents the amount of water (in gallons) needed for each of the other two tanks -/
def water_other_two : ℕ := water_first_two - 2

/-- Represents the total amount of water (in gallons) needed for all tanks in one week -/
def total_water_per_week : ℕ := 2 * water_first_two + 2 * water_other_two

/-- Represents the number of weeks -/
def num_weeks : ℕ := 4

/-- Represents the total amount of water (in gallons) needed for all tanks in four weeks -/
def total_water_four_weeks : ℕ := 112

theorem nicole_fish_tanks :
  num_tanks = 4 ∧
  water_first_two = 8 ∧
  water_other_two = water_first_two - 2 ∧
  total_water_per_week = 2 * water_first_two + 2 * water_other_two ∧
  total_water_four_weeks = num_weeks * total_water_per_week :=
by sorry

end NUMINAMATH_CALUDE_nicole_fish_tanks_l1974_197493


namespace NUMINAMATH_CALUDE_michaels_earnings_l1974_197409

/-- Calculates earnings based on hours worked and pay rates -/
def calculate_earnings (regular_hours : ℝ) (overtime_hours : ℝ) (regular_rate : ℝ) : ℝ :=
  regular_hours * regular_rate + overtime_hours * (2 * regular_rate)

theorem michaels_earnings :
  let total_hours : ℝ := 42.857142857142854
  let regular_hours : ℝ := 40
  let overtime_hours : ℝ := total_hours - regular_hours
  let regular_rate : ℝ := 7
  calculate_earnings regular_hours overtime_hours regular_rate = 320 := by
sorry

end NUMINAMATH_CALUDE_michaels_earnings_l1974_197409


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1974_197462

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + m) - a n = m * (a (n + 1) - a n)

theorem arithmetic_sequence_middle_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 5 = 16) : 
  a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1974_197462


namespace NUMINAMATH_CALUDE_additional_courses_is_two_l1974_197433

/-- Represents the wall construction problem --/
structure WallProblem where
  initial_courses : ℕ
  bricks_per_course : ℕ
  total_bricks : ℕ

/-- Calculates the number of additional courses added to the wall --/
def additional_courses (w : WallProblem) : ℕ :=
  let initial_bricks := w.initial_courses * w.bricks_per_course
  let remaining_bricks := w.total_bricks - initial_bricks + (w.bricks_per_course / 2)
  remaining_bricks / w.bricks_per_course

/-- Theorem stating that the number of additional courses is 2 --/
theorem additional_courses_is_two (w : WallProblem) 
    (h1 : w.initial_courses = 3)
    (h2 : w.bricks_per_course = 400)
    (h3 : w.total_bricks = 1800) : 
  additional_courses w = 2 := by
  sorry

#eval additional_courses { initial_courses := 3, bricks_per_course := 400, total_bricks := 1800 }

end NUMINAMATH_CALUDE_additional_courses_is_two_l1974_197433


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_diff_l1974_197401

theorem sqrt_fraction_sum_diff (x : ℝ) : 
  x = Real.sqrt ((1 : ℝ) / 25 + (1 : ℝ) / 36 - (1 : ℝ) / 100) → x = (Real.sqrt 13) / 15 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_diff_l1974_197401


namespace NUMINAMATH_CALUDE_toms_ribbon_length_l1974_197467

theorem toms_ribbon_length 
  (num_gifts : ℕ) 
  (ribbon_per_gift : ℝ) 
  (remaining_ribbon : ℝ) 
  (h1 : num_gifts = 8)
  (h2 : ribbon_per_gift = 1.5)
  (h3 : remaining_ribbon = 3) :
  (num_gifts : ℝ) * ribbon_per_gift + remaining_ribbon = 15 := by
  sorry

end NUMINAMATH_CALUDE_toms_ribbon_length_l1974_197467


namespace NUMINAMATH_CALUDE_new_homes_theorem_l1974_197437

/-- The number of original trailer homes -/
def original_homes : ℕ := 30

/-- The initial average age of original trailer homes 5 years ago -/
def initial_avg_age : ℚ := 15

/-- The current average age of all trailer homes -/
def current_avg_age : ℚ := 12

/-- The number of years that have passed -/
def years_passed : ℕ := 5

/-- Function to calculate the number of new trailer homes added -/
def new_homes_added : ℚ :=
  (original_homes * (initial_avg_age + years_passed) - original_homes * current_avg_age) /
  (current_avg_age - years_passed)

theorem new_homes_theorem :
  new_homes_added = 240 / 7 :=
sorry

end NUMINAMATH_CALUDE_new_homes_theorem_l1974_197437


namespace NUMINAMATH_CALUDE_expression_evaluation_l1974_197460

theorem expression_evaluation (x y : ℚ) (hx : x = -1/2) (hy : y = 2022) :
  ((2*x - y)^2 - (2*x + y)*(2*x - y)) / (2*y) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1974_197460


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1974_197417

theorem sqrt_sum_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (Real.sqrt x + Real.sqrt y)^8 ≥ 64 * x * y * (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1974_197417


namespace NUMINAMATH_CALUDE_polynomial_factorization_and_sum_power_l1974_197420

theorem polynomial_factorization_and_sum_power (a b : ℤ) : 
  (∀ x : ℝ, x^2 + x - 6 = (x + a) * (x + b)) → (a + b)^2023 = 1 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_and_sum_power_l1974_197420


namespace NUMINAMATH_CALUDE_same_color_sock_pairs_l1974_197447

/-- The number of ways to choose a pair of socks of the same color from a drawer -/
theorem same_color_sock_pairs (white brown blue red : ℕ) 
  (h_white : white = 5)
  (h_brown : brown = 6)
  (h_blue : blue = 3)
  (h_red : red = 2) : 
  Nat.choose white 2 + Nat.choose brown 2 + Nat.choose blue 2 + Nat.choose red 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_same_color_sock_pairs_l1974_197447


namespace NUMINAMATH_CALUDE_deer_cheetah_time_difference_l1974_197445

/-- Proves the time difference between a deer and cheetah passing a point, given their speeds and catch-up time. -/
theorem deer_cheetah_time_difference 
  (deer_speed : ℝ) 
  (cheetah_speed : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : deer_speed = 50) 
  (h2 : cheetah_speed = 60) 
  (h3 : catch_up_time = 1) : 
  ∃ (time_difference : ℝ), 
    time_difference = 4 ∧ 
    deer_speed * (catch_up_time + time_difference) = cheetah_speed * catch_up_time :=
by sorry

end NUMINAMATH_CALUDE_deer_cheetah_time_difference_l1974_197445


namespace NUMINAMATH_CALUDE_gcd_of_180_and_270_l1974_197497

theorem gcd_of_180_and_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_180_and_270_l1974_197497


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_63_with_digit_sum_63_l1974_197496

def digit_sum (n : ℕ) : ℕ := sorry

def is_divisible_by (n m : ℕ) : Prop := sorry

theorem smallest_number_divisible_by_63_with_digit_sum_63 :
  ∃ (n : ℕ),
    is_divisible_by n 63 ∧
    digit_sum n = 63 ∧
    (∀ m : ℕ, m < n → ¬(is_divisible_by m 63 ∧ digit_sum m = 63)) ∧
    n = 63999999 :=
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_63_with_digit_sum_63_l1974_197496


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1974_197422

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 10 feet is 52 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1974_197422


namespace NUMINAMATH_CALUDE_compute_expression_l1974_197468

theorem compute_expression : 15 * (1 / 17) * 34 - 1 / 2 = 59 / 2 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1974_197468


namespace NUMINAMATH_CALUDE_excluded_age_is_nine_l1974_197458

/-- A 5-digit number with distinct, consecutive digits in increasing order -/
def ConsecutiveDigitNumber := { n : ℕ | 
  12345 ≤ n ∧ n ≤ 98765 ∧ 
  ∃ (a b c d e : ℕ), n = 10000*a + 1000*b + 100*c + 10*d + e ∧
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e }

/-- The set of ages of Mrs. Smith's children -/
def ChildrenAges := { n : ℕ | 5 ≤ n ∧ n ≤ 13 }

theorem excluded_age_is_nine :
  ∃ (n : ConsecutiveDigitNumber),
    ∀ (k : ℕ), k ∈ ChildrenAges → k ≠ 9 → n % k = 0 :=
by sorry

end NUMINAMATH_CALUDE_excluded_age_is_nine_l1974_197458


namespace NUMINAMATH_CALUDE_may_to_june_increase_l1974_197499

-- Define the percentage changes
def march_to_april_increase : ℝ := 0.10
def april_to_may_decrease : ℝ := 0.20
def overall_increase : ℝ := 0.3200000000000003

-- Define the function to calculate the final value after percentage changes
def final_value (initial : ℝ) (increase1 : ℝ) (decrease : ℝ) (increase2 : ℝ) : ℝ :=
  initial * (1 + increase1) * (1 - decrease) * (1 + increase2)

-- Theorem to prove
theorem may_to_june_increase (initial : ℝ) (initial_pos : initial > 0) :
  ∃ (may_to_june : ℝ), 
    final_value initial march_to_april_increase april_to_may_decrease may_to_june = 
    initial * (1 + overall_increase) ∧ 
    may_to_june = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_may_to_june_increase_l1974_197499


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1974_197469

/-- A rhombus with given diagonal lengths has the specified perimeter. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side_length := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side_length = 52 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1974_197469


namespace NUMINAMATH_CALUDE_jihoons_class_size_l1974_197470

theorem jihoons_class_size :
  ∃! n : ℕ, 35 < n ∧ n < 70 ∧ n % 6 = 3 ∧ n % 8 = 1 ∧ n = 57 := by
  sorry

end NUMINAMATH_CALUDE_jihoons_class_size_l1974_197470


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1974_197463

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, x - 1 = 0 → (x - 1) * (x + 2) = 0) ∧
  (∃ x : ℝ, (x - 1) * (x + 2) = 0 ∧ x - 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1974_197463


namespace NUMINAMATH_CALUDE_yellow_marbles_total_l1974_197435

/-- The total number of yellow marbles after redistribution -/
def total_marbles_after_redistribution : ℕ → ℕ → ℕ → ℕ → ℕ
  | mary_initial, joan, john, mary_to_tim =>
    (mary_initial - mary_to_tim) + joan + john + mary_to_tim

/-- Theorem stating the total number of yellow marbles after redistribution -/
theorem yellow_marbles_total
  (mary_initial : ℕ)
  (joan : ℕ)
  (john : ℕ)
  (mary_to_tim : ℕ)
  (h1 : mary_initial = 9)
  (h2 : joan = 3)
  (h3 : john = 7)
  (h4 : mary_to_tim = 4)
  (h5 : mary_initial ≥ mary_to_tim) :
  total_marbles_after_redistribution mary_initial joan john mary_to_tim = 19 :=
by sorry

end NUMINAMATH_CALUDE_yellow_marbles_total_l1974_197435


namespace NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l1974_197498

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  radius : ℝ
  side_positive : 0 < side
  base_positive : 0 < base
  radius_positive : 0 < radius

-- Theorem statement
theorem isosceles_triangle_not_unique (r : ℝ) (hr : 0 < r) :
  ∃ (t1 t2 : IsoscelesTriangle), t1.radius = r ∧ t2.radius = r ∧ t1 ≠ t2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l1974_197498


namespace NUMINAMATH_CALUDE_expression_value_l1974_197419

theorem expression_value (x : ℝ) : x = 2 → 3 * x^2 - 4 * x + 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1974_197419


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l1974_197429

theorem triangle_area_from_squares (a b c : ℝ) (h1 : a^2 = 64) (h2 : b^2 = 121) (h3 : c^2 = 169)
  (h4 : a^2 + b^2 = c^2) : (1/2) * a * b = 44 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l1974_197429


namespace NUMINAMATH_CALUDE_x_plus_2y_inequality_l1974_197415

theorem x_plus_2y_inequality (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y > m^2 + 2*m ↔ m > -4 ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_2y_inequality_l1974_197415


namespace NUMINAMATH_CALUDE_multiple_of_reciprocal_l1974_197457

theorem multiple_of_reciprocal (x : ℝ) (k : ℝ) (h1 : x > 0) (h2 : x = 3) (h3 : x + 17 = k * (1 / x)) : k = 60 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_reciprocal_l1974_197457


namespace NUMINAMATH_CALUDE_inverse_expression_equals_one_fifth_l1974_197426

theorem inverse_expression_equals_one_fifth :
  (3 - 4 * (3 - 5)⁻¹)⁻¹ = (1 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_expression_equals_one_fifth_l1974_197426


namespace NUMINAMATH_CALUDE_number_is_forty_l1974_197410

theorem number_is_forty (N : ℝ) (P : ℝ) : 
  (P / 100) * N = 0.25 * 16 + 2 → N = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_is_forty_l1974_197410


namespace NUMINAMATH_CALUDE_dimitri_weekly_calorie_intake_l1974_197459

/-- Represents the daily calorie intake from burgers -/
def daily_calorie_intake (burger_a_calories burger_b_calories burger_c_calories : ℕ) 
  (burger_a_count burger_b_count burger_c_count : ℕ) : ℕ :=
  burger_a_calories * burger_a_count + 
  burger_b_calories * burger_b_count + 
  burger_c_calories * burger_c_count

/-- Calculates the weekly calorie intake based on daily intake -/
def weekly_calorie_intake (daily_intake : ℕ) (days_in_week : ℕ) : ℕ :=
  daily_intake * days_in_week

/-- Theorem stating Dimitri's weekly calorie intake from burgers -/
theorem dimitri_weekly_calorie_intake : 
  weekly_calorie_intake 
    (daily_calorie_intake 350 450 550 2 1 3) 
    7 = 19600 := by
  sorry


end NUMINAMATH_CALUDE_dimitri_weekly_calorie_intake_l1974_197459


namespace NUMINAMATH_CALUDE_cannot_form_square_l1974_197454

/-- Represents the number of sticks of each length -/
structure StickCounts where
  one_cm : Nat
  two_cm : Nat
  three_cm : Nat
  four_cm : Nat

/-- Calculates the total perimeter from the given stick counts -/
def totalPerimeter (counts : StickCounts) : Nat :=
  counts.one_cm * 1 + counts.two_cm * 2 + counts.three_cm * 3 + counts.four_cm * 4

/-- Checks if it's possible to form a square with the given stick counts -/
def canFormSquare (counts : StickCounts) : Prop :=
  ∃ (side : Nat), side > 0 ∧ 4 * side = totalPerimeter counts

/-- The given stick counts -/
def givenSticks : StickCounts :=
  { one_cm := 6
  , two_cm := 3
  , three_cm := 6
  , four_cm := 5
  }

/-- Theorem stating it's impossible to form a square with the given sticks -/
theorem cannot_form_square : ¬ canFormSquare givenSticks := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_square_l1974_197454


namespace NUMINAMATH_CALUDE_composite_form_l1974_197491

theorem composite_form (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (111111 + 9 * 10^n = a * b) := by
  sorry

end NUMINAMATH_CALUDE_composite_form_l1974_197491


namespace NUMINAMATH_CALUDE_equation_solution_l1974_197455

theorem equation_solution : ∃ n : ℚ, (6 / n) - (6 - 3) / 6 = 1 ∧ n = 4 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1974_197455


namespace NUMINAMATH_CALUDE_divisibility_conditions_l1974_197478

theorem divisibility_conditions (n : ℕ) (hn : n ≥ 1) :
  (n ∣ 2^n - 1 ↔ n = 1) ∧
  (n % 2 = 1 ∧ n ∣ 3^n + 1 ↔ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_conditions_l1974_197478


namespace NUMINAMATH_CALUDE_prime_triplet_divisiblity_l1974_197402

theorem prime_triplet_divisiblity (p q r : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) →
  ((p = 2 ∧ q = 5 ∧ r = 3) ∨ 
   (p = 5 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 5)) :=
sorry

end NUMINAMATH_CALUDE_prime_triplet_divisiblity_l1974_197402


namespace NUMINAMATH_CALUDE_circles_internally_tangent_with_common_tangent_l1974_197407

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 4 = 0
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 12*y + 4 = 0

-- Define the common tangent line
def common_tangent (x y : ℝ) : Prop := 3*x + 4*y = 0

-- Theorem statement
theorem circles_internally_tangent_with_common_tangent :
  ∃ (x₀ y₀ : ℝ),
    (circle_M x₀ y₀ ∧ circle_N x₀ y₀) ∧  -- Circles are internally tangent
    (∀ x y, circle_M x y ∧ circle_N x y → x = x₀ ∧ y = y₀) ∧  -- Only one intersection point
    (∀ x y, common_tangent x y →  -- Common tangent is tangent to both circles
      (∃ ε > 0, ∀ δ ∈ Set.Ioo (-ε) ε,
        ¬(circle_M (x + δ) (y - 3*δ/4) ∧ circle_N (x + δ) (y - 3*δ/4)))) :=
by sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_with_common_tangent_l1974_197407


namespace NUMINAMATH_CALUDE_cos_equality_solutions_l1974_197452

theorem cos_equality_solutions (n : ℤ) : 
  0 ≤ n ∧ n ≤ 360 ∧ Real.cos (n * π / 180) = Real.cos (340 * π / 180) → n = 20 ∨ n = 340 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_solutions_l1974_197452


namespace NUMINAMATH_CALUDE_no_real_solution_for_equation_l1974_197423

theorem no_real_solution_for_equation :
  ¬ ∃ (x : ℝ), x ≠ 0 ∧ (2 / x - (3 / x) * (6 / x) = 0.5) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_equation_l1974_197423


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1974_197428

theorem inscribed_square_area (R : ℝ) (h : R > 0) :
  (R^2 * (π - 2) / 4 = 2*π - 4) →
  (2 * R)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1974_197428


namespace NUMINAMATH_CALUDE_quadratic_above_x_axis_l1974_197405

/-- Given a quadratic function f(x) = ax^2 + x + 5, if f(x) > 0 for all real x, then a > 1/20 -/
theorem quadratic_above_x_axis (a : ℝ) :
  (∀ x : ℝ, a * x^2 + x + 5 > 0) → a > 1/20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_above_x_axis_l1974_197405


namespace NUMINAMATH_CALUDE_family_eye_count_l1974_197481

-- Define the family members and their eye counts
def mother_eyes : ℕ := 1
def father_eyes : ℕ := 3
def num_children : ℕ := 3
def eyes_per_child : ℕ := 4

-- Theorem statement
theorem family_eye_count :
  mother_eyes + father_eyes + num_children * eyes_per_child = 16 :=
by sorry

end NUMINAMATH_CALUDE_family_eye_count_l1974_197481


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1974_197421

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 * a 6 = 6 →
  a 8 * a 10 * a 12 = 24 →
  a 5 * a 7 * a 9 = 12 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1974_197421


namespace NUMINAMATH_CALUDE_orange_difference_l1974_197404

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges : ℕ := 44

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges : ℕ := 25

/-- The difference between the number of sacks of ripe and unripe oranges harvested per day -/
theorem orange_difference : ripe_oranges - unripe_oranges = 19 := by
  sorry

end NUMINAMATH_CALUDE_orange_difference_l1974_197404


namespace NUMINAMATH_CALUDE_ellipse_properties_l1974_197412

/-- Definition of the ellipse M -/
def ellipse_M (x y : ℝ) (a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1 ∧ a > 0

/-- One focus of the ellipse is at (-1, 0) -/
def focus_F : ℝ × ℝ := (-1, 0)

/-- A line l passing through F -/
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

/-- Theorem stating the main results -/
theorem ellipse_properties :
  ∃ (a : ℝ),
    -- 1. The equation of the ellipse
    (∀ x y : ℝ, ellipse_M x y a ↔ x^2 / 4 + y^2 / 3 = 1) ∧
    -- 2. Length of CD when l has a 45° angle
    (∃ C D : ℝ × ℝ,
      C.1 ≠ D.1 ∧
      ellipse_M C.1 C.2 a ∧
      ellipse_M D.1 D.2 a ∧
      C.2 = line_l 1 C.1 ∧
      D.2 = line_l 1 D.1 ∧
      Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 24 / 7) ∧
    -- 3. Maximum value of |S₁ - S₂|
    (∃ S_max : ℝ,
      S_max = Real.sqrt 3 ∧
      ∀ k : ℝ,
        ∃ C D : ℝ × ℝ,
          C.1 ≠ D.1 ∧
          ellipse_M C.1 C.2 a ∧
          ellipse_M D.1 D.2 a ∧
          C.2 = line_l k C.1 ∧
          D.2 = line_l k D.1 ∧
          |C.2 - D.2| ≤ S_max) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1974_197412


namespace NUMINAMATH_CALUDE_marks_total_spent_l1974_197485

/-- Represents the purchase of a fruit with its quantity and price per pound -/
structure FruitPurchase where
  quantity : ℝ
  price_per_pound : ℝ

/-- Calculates the total cost of a fruit purchase -/
def total_cost (purchase : FruitPurchase) : ℝ :=
  purchase.quantity * purchase.price_per_pound

/-- Represents Mark's shopping list -/
structure ShoppingList where
  tomatoes : FruitPurchase
  apples : FruitPurchase
  oranges : FruitPurchase

/-- Calculates the total cost of all items in the shopping list -/
def total_spent (list : ShoppingList) : ℝ :=
  total_cost list.tomatoes + total_cost list.apples + total_cost list.oranges

/-- Mark's actual shopping list -/
def marks_shopping : ShoppingList :=
  { tomatoes := { quantity := 3, price_per_pound := 4.5 }
  , apples := { quantity := 7, price_per_pound := 3.25 }
  , oranges := { quantity := 4, price_per_pound := 2.75 }
  }

/-- Theorem: The total amount Mark spent is $47.25 -/
theorem marks_total_spent :
  total_spent marks_shopping = 47.25 := by
  sorry


end NUMINAMATH_CALUDE_marks_total_spent_l1974_197485


namespace NUMINAMATH_CALUDE_us_stripes_count_l1974_197436

/-- The number of stars on the US flag -/
def us_stars : ℕ := 50

/-- The number of circles on Pete's flag -/
def pete_circles : ℕ := us_stars / 2 - 3

/-- The number of squares on Pete's flag as a function of US flag stripes -/
def pete_squares (s : ℕ) : ℕ := 2 * s + 6

/-- The total number of shapes on Pete's flag -/
def pete_total_shapes : ℕ := 54

/-- Theorem: The number of stripes on the US flag is 13 -/
theorem us_stripes_count : 
  ∃ (s : ℕ), s = 13 ∧ pete_circles + pete_squares s = pete_total_shapes :=
sorry

end NUMINAMATH_CALUDE_us_stripes_count_l1974_197436


namespace NUMINAMATH_CALUDE_blue_markers_count_l1974_197474

theorem blue_markers_count (total : ℝ) (red : ℝ) (blue : ℝ) : 
  total = 64.0 → red = 41.0 → blue = total - red → blue = 23.0 := by
  sorry

end NUMINAMATH_CALUDE_blue_markers_count_l1974_197474


namespace NUMINAMATH_CALUDE_train_crossing_time_l1974_197494

/-- Given a train traveling at a certain speed that crosses a platform of known length in a specific time,
    calculate the time it takes for the train to cross a man standing on the platform. -/
theorem train_crossing_time
  (train_speed_kmph : ℝ)
  (train_speed_mps : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_speed_kmph = 72)
  (h2 : train_speed_mps = 20)
  (h3 : platform_length = 220)
  (h4 : platform_crossing_time = 30)
  (h5 : train_speed_mps = train_speed_kmph * (1000 / 3600)) :
  (train_speed_mps * platform_crossing_time - platform_length) / train_speed_mps = 19 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1974_197494


namespace NUMINAMATH_CALUDE_spadesuit_example_l1974_197431

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := |a^2 - b^2|

-- Theorem statement
theorem spadesuit_example : spadesuit 3 (spadesuit 5 2) = 432 := by
  sorry

end NUMINAMATH_CALUDE_spadesuit_example_l1974_197431


namespace NUMINAMATH_CALUDE_function_composition_problem_l1974_197484

theorem function_composition_problem (a : ℝ) : 
  let f (x : ℝ) := x / 4 + 2
  let g (x : ℝ) := 5 - x
  f (g a) = 4 → a = -3 := by
sorry

end NUMINAMATH_CALUDE_function_composition_problem_l1974_197484


namespace NUMINAMATH_CALUDE_snacks_sold_l1974_197495

/-- Given the initial number of snacks and ramens in a market, and the final total after some transactions, 
    prove that the number of snacks sold is 599. -/
theorem snacks_sold (initial_snacks : ℕ) (initial_ramens : ℕ) (ramens_bought : ℕ) (final_total : ℕ) :
  initial_snacks = 1238 →
  initial_ramens = initial_snacks + 374 →
  ramens_bought = 276 →
  final_total = 2527 →
  (initial_snacks - (initial_snacks - (initial_ramens + ramens_bought - final_total))) = 599 := by
  sorry

end NUMINAMATH_CALUDE_snacks_sold_l1974_197495


namespace NUMINAMATH_CALUDE_student_A_more_stable_l1974_197425

/-- Represents a student's jumping rope performance -/
structure JumpRopePerformance where
  average_score : ℝ
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines when one performance is more stable than another -/
def more_stable (a b : JumpRopePerformance) : Prop :=
  a.variance < b.variance

theorem student_A_more_stable (
  student_A student_B : JumpRopePerformance
) (h1 : student_A.average_score = student_B.average_score)
  (h2 : student_A.variance = 0.06)
  (h3 : student_B.variance = 0.35) :
  more_stable student_A student_B :=
sorry

end NUMINAMATH_CALUDE_student_A_more_stable_l1974_197425


namespace NUMINAMATH_CALUDE_b_finish_days_l1974_197408

/-- The number of days A needs to complete the entire work -/
def a_total_days : ℝ := 20

/-- The number of days B needs to complete the entire work -/
def b_total_days : ℝ := 30

/-- The number of days A worked before leaving -/
def a_worked_days : ℝ := 10

/-- Theorem: Given the conditions, B can finish the remaining work in 15 days -/
theorem b_finish_days : 
  ∃ (b_days : ℝ), 
    (1 / b_total_days) * b_days = 1 - (a_worked_days / a_total_days) ∧ 
    b_days = 15 := by
  sorry

end NUMINAMATH_CALUDE_b_finish_days_l1974_197408


namespace NUMINAMATH_CALUDE_crow_probability_l1974_197413

theorem crow_probability (a b c d : ℕ) : 
  a + b = 50 →  -- Total crows on birch
  c + d = 50 →  -- Total crows on oak
  b ≥ a →       -- Black crows ≥ White crows on birch
  d ≥ c - 1 →   -- Black crows ≥ White crows - 1 on oak
  (b * (d + 1) + a * (c + 1)) / (50 * 51 : ℚ) > (b * c + a * d) / (50 * 51 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_crow_probability_l1974_197413


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1974_197490

theorem pure_imaginary_complex_number (a : ℝ) :
  (((2 : ℂ) - a * Complex.I) / (1 + Complex.I)).re = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1974_197490


namespace NUMINAMATH_CALUDE_count_valid_integers_l1974_197487

/-- A function that returns true if a natural number is a four-digit positive integer -/
def isFourDigitPositive (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- A function that returns true if a natural number is divisible by 25 -/
def isDivisibleBy25 (n : ℕ) : Prop :=
  n % 25 = 0

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- A function that returns true if the sum of digits of a natural number is divisible by 3 -/
def sumOfDigitsDivisibleBy3 (n : ℕ) : Prop :=
  (sumOfDigits n) % 3 = 0

/-- The count of positive four-digit integers divisible by 25 with sum of digits divisible by 3 -/
def countValidIntegers : ℕ :=
  sorry

/-- Theorem stating that the count of valid integers satisfies all conditions -/
theorem count_valid_integers :
  ∃ (n : ℕ), n = countValidIntegers ∧
  ∀ (m : ℕ), (isFourDigitPositive m ∧ isDivisibleBy25 m ∧ sumOfDigitsDivisibleBy3 m) →
  (m ∈ Finset.range n) :=
  sorry

end NUMINAMATH_CALUDE_count_valid_integers_l1974_197487


namespace NUMINAMATH_CALUDE_nonreal_roots_product_l1974_197488

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 6*x^3 + 15*x^2 - 20*x = 396) →
  (∃ z w : ℂ, z ≠ w ∧ z.im ≠ 0 ∧ w.im ≠ 0 ∧ 
   (x^4 - 6*x^3 + 15*x^2 - 20*x - 396 = 0 → x = z ∨ x = w) ∧
   z * w = 4 + Real.sqrt 412) :=
by sorry

end NUMINAMATH_CALUDE_nonreal_roots_product_l1974_197488


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1974_197464

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 7) / (x - 4) = (x - 5) / (x + 2) ∧ x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1974_197464


namespace NUMINAMATH_CALUDE_max_difference_averages_l1974_197466

theorem max_difference_averages (x y : ℝ) (hx : 4 ≤ x ∧ x ≤ 100) (hy : 4 ≤ y ∧ y ≤ 100) :
  ∃ (z : ℝ), z = |((x + y) / 2) - ((x + 2 * y) / 3)| ∧
  z ≤ 16 ∧
  ∃ (a b : ℝ), (4 ≤ a ∧ a ≤ 100) ∧ (4 ≤ b ∧ b ≤ 100) ∧
    |((a + b) / 2) - ((a + 2 * b) / 3)| = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_averages_l1974_197466


namespace NUMINAMATH_CALUDE_regions_bound_l1974_197480

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields to define a plane

/-- The number of regions formed by three planes in 3D space -/
def num_regions (p1 p2 p3 : Plane3D) : ℕ :=
  sorry

theorem regions_bound (p1 p2 p3 : Plane3D) :
  4 ≤ num_regions p1 p2 p3 ∧ num_regions p1 p2 p3 ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_regions_bound_l1974_197480


namespace NUMINAMATH_CALUDE_magic_square_sum_l1974_197444

theorem magic_square_sum (b c d e g h : ℕ) : 
  b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ g > 0 ∧ h > 0 →
  30 * b * c = d * e * 3 →
  30 * b * c = g * h * 3 →
  30 * b * c = 30 * e * 3 →
  30 * b * c = b * e * h →
  30 * b * c = c * 3 * 3 →
  30 * b * c = 30 * e * g →
  30 * b * c = c * e * 3 →
  (∃ g₁ g₂ : ℕ, g = g₁ ∨ g = g₂) →
  g₁ + g₂ = 25 :=
by sorry

end NUMINAMATH_CALUDE_magic_square_sum_l1974_197444


namespace NUMINAMATH_CALUDE_probability_of_losing_is_one_third_l1974_197441

/-- A game where a single standard die is rolled once -/
structure DieGame where
  /-- The set of all possible outcomes when rolling a standard die -/
  outcomes : Finset Nat
  /-- The set of losing outcomes -/
  losing_outcomes : Finset Nat
  /-- Assumption that outcomes are the numbers 1 to 6 -/
  outcomes_def : outcomes = Finset.range 6
  /-- Assumption that losing outcomes are 5 and 6 -/
  losing_def : losing_outcomes = {5, 6}

/-- The probability of losing in the die game -/
def probability_of_losing (game : DieGame) : ℚ :=
  (game.losing_outcomes.card : ℚ) / (game.outcomes.card : ℚ)

/-- Theorem stating that the probability of losing is 1/3 -/
theorem probability_of_losing_is_one_third (game : DieGame) :
    probability_of_losing game = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_losing_is_one_third_l1974_197441


namespace NUMINAMATH_CALUDE_comparison_inequality_l1974_197440

theorem comparison_inequality (a b : ℝ) (h1 : a ≠ b) (h2 : a < 0) :
  a < 2 * b - b^2 / a := by
sorry

end NUMINAMATH_CALUDE_comparison_inequality_l1974_197440


namespace NUMINAMATH_CALUDE_parallel_line_slope_l1974_197434

/-- Given a line with equation 2x + 4y = -17, prove that its slope (and the slope of any parallel line) is -1/2 -/
theorem parallel_line_slope (x y : ℝ) (h : 2 * x + 4 * y = -17) :
  ∃ m b : ℝ, m = -1/2 ∧ y = m * x + b := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l1974_197434


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l1974_197439

/-- Given a complex number z satisfying z + |z| = 2 + 8i, prove that |z|² = 289 -/
theorem complex_modulus_squared (z : ℂ) (h : z + Complex.abs z = 2 + 8 * Complex.I) : 
  Complex.abs z ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l1974_197439


namespace NUMINAMATH_CALUDE_speeding_proof_l1974_197400

theorem speeding_proof (distance : ℝ) (time : ℝ) (speed_limit : ℝ)
  (h1 : distance = 165)
  (h2 : time = 2)
  (h3 : speed_limit = 80)
  : ∃ t : ℝ, 0 ≤ t ∧ t ≤ time ∧ (distance / time > speed_limit) :=
by
  sorry

#check speeding_proof

end NUMINAMATH_CALUDE_speeding_proof_l1974_197400


namespace NUMINAMATH_CALUDE_triangle_area_l1974_197416

/-- Given a triangle with side lengths 9, 12, and 15 units, its area is 54 square units. -/
theorem triangle_area : ∀ (a b c : ℝ), 
  a = 9 ∧ b = 12 ∧ c = 15 →
  (∃ (A : ℝ), A = (1/2) * a * b ∧ A = 54) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1974_197416


namespace NUMINAMATH_CALUDE_chess_tournament_players_l1974_197476

theorem chess_tournament_players (total_games : ℕ) (h : total_games = 380) :
  ∃ n : ℕ, n > 0 ∧ 2 * n * (n - 1) = total_games :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l1974_197476
