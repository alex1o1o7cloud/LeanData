import Mathlib

namespace NUMINAMATH_CALUDE_find_n_l1898_189887

theorem find_n : ∃ n : ℝ, n + (n + 1) + (n + 2) + (n + 3) = 20 ∧ n = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1898_189887


namespace NUMINAMATH_CALUDE_library_visitors_sunday_visitors_proof_l1898_189814

/-- Calculates the average number of visitors on Sundays in a library -/
theorem library_visitors (non_sunday_visitors : ℕ) (total_average : ℕ) : ℕ :=
  let total_days : ℕ := 30
  let sundays : ℕ := 5
  let non_sundays : ℕ := total_days - sundays
  let sunday_visitors : ℕ := 
    (total_average * total_days - non_sunday_visitors * non_sundays) / sundays
  sunday_visitors

/-- Proves that the average number of Sunday visitors is 510 given the conditions -/
theorem sunday_visitors_proof (h1 : library_visitors 240 285 = 510) : 
  library_visitors 240 285 = 510 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_sunday_visitors_proof_l1898_189814


namespace NUMINAMATH_CALUDE_max_intersections_for_given_points_l1898_189836

/-- The maximum number of intersection points in the first quadrant -/
def max_intersection_points (x_points y_points : ℕ) : ℕ :=
  (x_points * y_points * (x_points - 1) * (y_points - 1)) / 4

/-- Theorem stating the maximum number of intersection points for the given conditions -/
theorem max_intersections_for_given_points :
  max_intersection_points 5 3 = 30 := by sorry

end NUMINAMATH_CALUDE_max_intersections_for_given_points_l1898_189836


namespace NUMINAMATH_CALUDE_taehyungs_mother_age_l1898_189821

/-- Given the age differences and the younger brother's age, prove Taehyung's mother's age --/
theorem taehyungs_mother_age :
  ∀ (taehyung_age mother_age brother_age : ℕ),
    mother_age - taehyung_age = 31 →
    taehyung_age - brother_age = 5 →
    brother_age = 7 →
    mother_age = 43 := by
  sorry

end NUMINAMATH_CALUDE_taehyungs_mother_age_l1898_189821


namespace NUMINAMATH_CALUDE_angle_P_measure_l1898_189812

-- Define the triangle PQR
structure Triangle :=
  (P Q R : Real)

-- Define the properties of the triangle
def valid_triangle (t : Triangle) : Prop :=
  t.P > 0 ∧ t.Q > 0 ∧ t.R > 0 ∧ t.P + t.Q + t.R = 180

-- Define the theorem
theorem angle_P_measure (t : Triangle) 
  (h1 : valid_triangle t) 
  (h2 : t.Q = 3 * t.R) 
  (h3 : t.R = 18) : 
  t.P = 108 := by
  sorry

end NUMINAMATH_CALUDE_angle_P_measure_l1898_189812


namespace NUMINAMATH_CALUDE_evaluate_expression_l1898_189856

theorem evaluate_expression : 5 * (9 - 3) + 8 = 38 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1898_189856


namespace NUMINAMATH_CALUDE_prime_power_sum_l1898_189811

theorem prime_power_sum (w x y z : ℕ) :
  2^w * 3^x * 5^y * 7^z = 882 →
  2*w + 3*x + 5*y + 7*z = 22 := by
sorry

end NUMINAMATH_CALUDE_prime_power_sum_l1898_189811


namespace NUMINAMATH_CALUDE_treats_ratio_wanda_to_jane_l1898_189833

/-- Proves that the ratio of treats Wanda brings compared to Jane is 1:2 -/
theorem treats_ratio_wanda_to_jane :
  ∀ (jane_treats jane_bread wanda_treats wanda_bread : ℕ),
    jane_bread = (75 * jane_treats) / 100 →
    wanda_bread = 3 * wanda_treats →
    wanda_bread = 90 →
    jane_treats + jane_bread + wanda_treats + wanda_bread = 225 →
    wanda_treats * 2 = jane_treats :=
by
  sorry

#check treats_ratio_wanda_to_jane

end NUMINAMATH_CALUDE_treats_ratio_wanda_to_jane_l1898_189833


namespace NUMINAMATH_CALUDE_probability_of_rolling_six_l1898_189871

/-- The probability of rolling a total of 6 with two fair dice -/
theorem probability_of_rolling_six (dice : ℕ) (faces : ℕ) (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  dice = 2 →
  faces = 6 →
  total_outcomes = faces * faces →
  favorable_outcomes = 5 →
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_rolling_six_l1898_189871


namespace NUMINAMATH_CALUDE_bridge_length_l1898_189842

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 205 ∧
    bridge_length = train_speed_kmh * (1000 / 3600) * crossing_time - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1898_189842


namespace NUMINAMATH_CALUDE_cubic_root_implies_coefficients_l1898_189895

theorem cubic_root_implies_coefficients 
  (a b : ℝ) 
  (h : (2 - 3*Complex.I)^3 + a*(2 - 3*Complex.I)^2 - 2*(2 - 3*Complex.I) + b = 0) : 
  a = -1/4 ∧ b = 195/4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_implies_coefficients_l1898_189895


namespace NUMINAMATH_CALUDE_fraction_simplification_l1898_189878

theorem fraction_simplification :
  (1 / 5 + 1 / 7) / ((2 / 3 - 1 / 4) * 2 / 5) = 72 / 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1898_189878


namespace NUMINAMATH_CALUDE_smallest_positive_angle_l1898_189827

def angle_equation (x : ℝ) : Prop :=
  12 * (Real.sin x)^3 * (Real.cos x)^2 - 12 * (Real.sin x)^2 * (Real.cos x)^3 = 3/2

theorem smallest_positive_angle :
  ∃ (x : ℝ), x > 0 ∧ x < π/2 ∧ angle_equation x ∧
  ∀ (y : ℝ), y > 0 ∧ y < x → ¬(angle_equation y) ∧
  x = 7.5 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_l1898_189827


namespace NUMINAMATH_CALUDE_sunflower_seed_distribution_l1898_189838

theorem sunflower_seed_distribution (total_seeds : ℕ) (num_cans : ℕ) (seeds_per_can : ℕ) 
  (h1 : total_seeds = 54)
  (h2 : num_cans = 9)
  (h3 : total_seeds = num_cans * seeds_per_can) :
  seeds_per_can = 6 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_seed_distribution_l1898_189838


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1898_189815

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_solution_set : ∀ x, f a b c x > 0 ↔ -1/2 < x ∧ x < 3) :
  c > 0 ∧ 4*a + 2*b + c > 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1898_189815


namespace NUMINAMATH_CALUDE_reach_floor_pushups_l1898_189872

/-- Represents the number of push-up variations -/
def numVariations : ℕ := 5

/-- Represents the number of training days per week -/
def trainingDaysPerWeek : ℕ := 6

/-- Represents the number of reps added per day -/
def repsAddedPerDay : ℕ := 1

/-- Represents the target number of reps to progress to the next variation -/
def targetReps : ℕ := 25

/-- Calculates the number of weeks needed to progress through one variation -/
def weeksPerVariation : ℕ := 
  (targetReps + trainingDaysPerWeek - 1) / trainingDaysPerWeek

/-- The total number of weeks needed to reach floor push-ups -/
def totalWeeks : ℕ := numVariations * weeksPerVariation

theorem reach_floor_pushups : totalWeeks = 20 := by
  sorry

end NUMINAMATH_CALUDE_reach_floor_pushups_l1898_189872


namespace NUMINAMATH_CALUDE_trapezoid_angle_sequence_l1898_189861

theorem trapezoid_angle_sequence (a d : ℝ) : 
  (a > 0) →
  (d > 0) →
  (a + 2*d = 105) →
  (4*a + 6*d = 360) →
  (a + d = 85) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_angle_sequence_l1898_189861


namespace NUMINAMATH_CALUDE_negative_four_star_two_simplify_a_minus_b_cubed_specific_values_l1898_189845

-- Define the * operation
def star (x y : ℚ) : ℚ := x^2 - 3*y + 3

-- Theorem 1
theorem negative_four_star_two : star (-4) 2 = 13 := by sorry

-- Theorem 2
theorem simplify_a_minus_b_cubed (a b : ℚ) : 
  star (a - b) ((a - b)^2) = -2*a^2 - 2*b^2 + 4*a*b + 3 := by sorry

-- Theorem 3
theorem specific_values : 
  star (-2 - (1/2)) ((-2 - (1/2))^2) = -13/2 := by sorry

end NUMINAMATH_CALUDE_negative_four_star_two_simplify_a_minus_b_cubed_specific_values_l1898_189845


namespace NUMINAMATH_CALUDE_sandys_books_l1898_189810

theorem sandys_books (total_spent : ℕ) (books_second_shop : ℕ) (avg_price : ℕ) :
  total_spent = 1920 →
  books_second_shop = 55 →
  avg_price = 16 →
  ∃ (books_first_shop : ℕ), 
    books_first_shop = 65 ∧
    avg_price * (books_first_shop + books_second_shop) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_sandys_books_l1898_189810


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l1898_189865

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 0 < x ∧ x < 2)
  (h2 : 0 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + 2 < 4)
  (h6 : x = 1) :
  ∀ y : ℝ, (∀ z : ℤ, (0 < z ∧ z < y ∧ 
                       0 < z ∧ z < 15 ∧
                       -1 < z ∧ z < 5 ∧
                       0 < z ∧ z < 3 ∧
                       z + 2 < 4 ∧
                       z = 1) → z ≤ x) → 
  y ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l1898_189865


namespace NUMINAMATH_CALUDE_deleted_items_count_l1898_189839

/-- Calculates the total number of deleted items given initial and final counts of apps and files, and the number of files transferred. -/
def totalDeletedItems (initialApps initialFiles finalApps finalFiles transferredFiles : ℕ) : ℕ :=
  (initialApps - finalApps) + (initialFiles - (finalFiles + transferredFiles))

/-- Theorem stating that the total number of deleted items is 24 given the problem conditions. -/
theorem deleted_items_count :
  totalDeletedItems 17 21 3 7 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_deleted_items_count_l1898_189839


namespace NUMINAMATH_CALUDE_solutions_for_twenty_initial_conditions_arithmetic_progression_l1898_189851

/-- The number of integer solutions for |x| + |y| = n -/
def numSolutions (n : ℕ) : ℕ := 4 * n

theorem solutions_for_twenty :
  numSolutions 20 = 80 :=
by sorry

/-- Verifies that the first three terms match the given conditions -/
theorem initial_conditions :
  numSolutions 1 = 4 ∧ numSolutions 2 = 8 ∧ numSolutions 3 = 12 :=
by sorry

/-- The sequence of solutions forms an arithmetic progression -/
theorem arithmetic_progression (n : ℕ) :
  numSolutions (n + 1) - numSolutions n = 4 :=
by sorry

end NUMINAMATH_CALUDE_solutions_for_twenty_initial_conditions_arithmetic_progression_l1898_189851


namespace NUMINAMATH_CALUDE_unused_edge_exists_l1898_189896

/-- Represents a token on a vertex of the 2n-gon -/
structure Token (n : ℕ) where
  position : Fin (2 * n)

/-- Represents a move (swapping tokens on an edge) -/
structure Move (n : ℕ) where
  edge : Fin (2 * n) × Fin (2 * n)

/-- Represents the state of the 2n-gon after some moves -/
structure GameState (n : ℕ) where
  tokens : Fin (2 * n) → Token n
  moves : List (Move n)

/-- Predicate to check if two tokens have been swapped -/
def haveBeenSwapped (n : ℕ) (t1 t2 : Token n) (moves : List (Move n)) : Prop :=
  sorry

/-- Predicate to check if an edge has been used for swapping -/
def edgeUsed (n : ℕ) (edge : Fin (2 * n) × Fin (2 * n)) (moves : List (Move n)) : Prop :=
  sorry

/-- The main theorem -/
theorem unused_edge_exists (n : ℕ) (finalState : GameState n) :
  (∀ t1 t2 : Token n, t1 ≠ t2 → haveBeenSwapped n t1 t2 finalState.moves) →
  ∃ edge : Fin (2 * n) × Fin (2 * n), ¬edgeUsed n edge finalState.moves :=
sorry

end NUMINAMATH_CALUDE_unused_edge_exists_l1898_189896


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l1898_189894

/-- A linear function y = (1-2m)x + m + 1 passes through the first, second, and third quadrants
    if and only if -1 < m < 1/2 -/
theorem linear_function_quadrants (m : ℝ) :
  (∀ x y : ℝ, y = (1 - 2*m)*x + m + 1 →
    (∃ x₁ y₁, x₁ > 0 ∧ y₁ > 0 ∧ y₁ = (1 - 2*m)*x₁ + m + 1) ∧
    (∃ x₂ y₂, x₂ < 0 ∧ y₂ > 0 ∧ y₂ = (1 - 2*m)*x₂ + m + 1) ∧
    (∃ x₃ y₃, x₃ < 0 ∧ y₃ < 0 ∧ y₃ = (1 - 2*m)*x₃ + m + 1)) ↔
  -1 < m ∧ m < 1/2 :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l1898_189894


namespace NUMINAMATH_CALUDE_max_value_ab_squared_l1898_189846

theorem max_value_ab_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∃ (max : ℝ), max = (4 * Real.sqrt 6) / 9 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 2 → x * y^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_ab_squared_l1898_189846


namespace NUMINAMATH_CALUDE_one_fourth_between_fifths_l1898_189860

/-- The weighted average of two rational numbers -/
def weightedAverage (x₁ x₂ : ℚ) (w₁ w₂ : ℚ) : ℚ :=
  (w₁ * x₁ + w₂ * x₂) / (w₁ + w₂)

/-- The number one fourth of the way from 1/5 to 4/5 is 7/20 -/
theorem one_fourth_between_fifths :
  weightedAverage (1/5 : ℚ) (4/5 : ℚ) 3 1 = 7/20 := by
  sorry

#check one_fourth_between_fifths

end NUMINAMATH_CALUDE_one_fourth_between_fifths_l1898_189860


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l1898_189855

theorem tan_double_angle_special_case (θ : ℝ) :
  2 * Real.cos (θ - π / 3) = 3 * Real.cos θ →
  Real.tan (2 * θ) = -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l1898_189855


namespace NUMINAMATH_CALUDE_inequality_properties_l1898_189822

theorem inequality_properties (x y : ℝ) (h : x > y) :
  (x - 3 > y - 3) ∧
  (x / 3 > y / 3) ∧
  (x + 3 > y + 3) ∧
  (1 - 3*x < 1 - 3*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l1898_189822


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l1898_189899

theorem arithmetic_evaluation : 6 - 8 * (9 - 4^2) * 3 = 174 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l1898_189899


namespace NUMINAMATH_CALUDE_different_arrangements_count_l1898_189864

def num_red_balls : ℕ := 6
def num_green_balls : ℕ := 3
def num_selected_balls : ℕ := 4

def num_arrangements : ℕ := 15

theorem different_arrangements_count :
  (num_red_balls = 6) →
  (num_green_balls = 3) →
  (num_selected_balls = 4) →
  num_arrangements = 15 := by
  sorry

end NUMINAMATH_CALUDE_different_arrangements_count_l1898_189864


namespace NUMINAMATH_CALUDE_fifth_reading_calculation_l1898_189807

theorem fifth_reading_calculation (r1 r2 r3 r4 : ℝ) (mean : ℝ) (h1 : r1 = 2) (h2 : r2 = 2.1) (h3 : r3 = 2) (h4 : r4 = 2.2) (h_mean : mean = 2) :
  ∃ r5 : ℝ, (r1 + r2 + r3 + r4 + r5) / 5 = mean ∧ r5 = 1.7 :=
by sorry

end NUMINAMATH_CALUDE_fifth_reading_calculation_l1898_189807


namespace NUMINAMATH_CALUDE_area_of_tangent_square_l1898_189849

/-- Given a 6 by 6 square with four semicircles on its sides, and another square EFGH
    with sides parallel and tangent to the semicircles, the area of EFGH is 144. -/
theorem area_of_tangent_square (original_side_length : ℝ) (EFGH_side_length : ℝ) : 
  original_side_length = 6 →
  EFGH_side_length = original_side_length + 2 * (original_side_length / 2) →
  EFGH_side_length ^ 2 = 144 := by
sorry

end NUMINAMATH_CALUDE_area_of_tangent_square_l1898_189849


namespace NUMINAMATH_CALUDE_line_representation_slope_nonexistence_x_intercept_angle_of_inclination_l1898_189820

/-- Represents the equation ((m^2 - 2m - 3)x + (2m^2 + m - 1)y + 6 - 2m = 0) -/
def equation (m x y : ℝ) : Prop :=
  (m^2 - 2*m - 3)*x + (2*m^2 + m - 1)*y + 6 - 2*m = 0

/-- The equation represents a line if and only if m ≠ -1 -/
theorem line_representation (m : ℝ) : 
  (∃ x y, equation m x y) ↔ m ≠ -1 := by sorry

/-- The slope of the line does not exist when m = 1/2 -/
theorem slope_nonexistence (m : ℝ) : 
  (∀ x y, equation m x y → x = 4/3) ↔ m = 1/2 := by sorry

/-- When the x-intercept is -3, m = -5/3 -/
theorem x_intercept (m : ℝ) : 
  (∃ y, equation m (-3) y) ↔ m = -5/3 := by sorry

/-- When the angle of inclination is 45°, m = 4/3 -/
theorem angle_of_inclination (m : ℝ) : 
  (∀ x₁ y₁ x₂ y₂, equation m x₁ y₁ ∧ equation m x₂ y₂ ∧ x₁ ≠ x₂ → 
    (y₂ - y₁) / (x₂ - x₁) = 1) ↔ m = 4/3 := by sorry

end NUMINAMATH_CALUDE_line_representation_slope_nonexistence_x_intercept_angle_of_inclination_l1898_189820


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l1898_189893

theorem consecutive_pages_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 285 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l1898_189893


namespace NUMINAMATH_CALUDE_north_pond_duck_count_l1898_189844

/-- The number of ducks in Lake Michigan -/
def lake_michigan_ducks : ℕ := 100

/-- The number of ducks in North Pond -/
def north_pond_ducks : ℕ := 2 * lake_michigan_ducks + 6

/-- Theorem stating that North Pond has 206 ducks -/
theorem north_pond_duck_count : north_pond_ducks = 206 := by
  sorry

end NUMINAMATH_CALUDE_north_pond_duck_count_l1898_189844


namespace NUMINAMATH_CALUDE_x_squared_less_than_abs_x_plus_two_l1898_189886

theorem x_squared_less_than_abs_x_plus_two (x : ℝ) :
  x^2 < |x| + 2 ↔ -2 < x ∧ x < 2 :=
by sorry

end NUMINAMATH_CALUDE_x_squared_less_than_abs_x_plus_two_l1898_189886


namespace NUMINAMATH_CALUDE_min_shapes_for_square_l1898_189848

/-- The area of each shape in square units -/
def shape_area : ℕ := 3

/-- The side length of the smallest possible square that can be formed -/
def square_side : ℕ := 6

/-- The theorem stating the minimum number of shapes required -/
theorem min_shapes_for_square :
  let total_area : ℕ := square_side * square_side
  let num_shapes : ℕ := total_area / shape_area
  (∀ n : ℕ, n < num_shapes → n * shape_area < square_side * square_side) ∧
  (num_shapes * shape_area = square_side * square_side) ∧
  (∃ (arrangement : ℕ → ℕ → ℕ),
    (∀ i j : ℕ, i < square_side ∧ j < square_side →
      ∃ k : ℕ, k < num_shapes ∧ arrangement i j = k)) :=
by sorry

end NUMINAMATH_CALUDE_min_shapes_for_square_l1898_189848


namespace NUMINAMATH_CALUDE_no_number_with_digit_product_1560_l1898_189829

/-- The product of the decimal digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- Theorem stating that no natural number has a digit product of 1560 -/
theorem no_number_with_digit_product_1560 : 
  ¬ ∃ (n : ℕ), digit_product n = 1560 := by sorry

end NUMINAMATH_CALUDE_no_number_with_digit_product_1560_l1898_189829


namespace NUMINAMATH_CALUDE_mod_eleven_fifth_power_l1898_189809

theorem mod_eleven_fifth_power (n : ℕ) : 
  11^5 ≡ n [ZMOD 9] → 0 ≤ n → n < 9 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_eleven_fifth_power_l1898_189809


namespace NUMINAMATH_CALUDE_triangle_area_is_71_l1898_189858

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by slope and a point -/
structure Line where
  slope : ℝ
  point : Point

/-- Vertical line represented by x-coordinate -/
structure VerticalLine where
  x : ℝ

theorem triangle_area_is_71 
  (l1 : Line) 
  (l2 : Line) 
  (l3 : VerticalLine)
  (h1 : l1.slope = 3)
  (h2 : l2.slope = -1/3)
  (h3 : l1.point = l2.point)
  (h4 : l1.point = ⟨1, 1⟩)
  (h5 : l3.x + (3 * l3.x - 2) = 12) : 
  ∃ (A B C : Point), 
    (A.x + A.y = 12 ∧ A.y = 3 * A.x - 2) ∧
    (B.x + B.y = 12 ∧ B.y = -1/3 * B.x + 4/3) ∧
    (C = l1.point) ∧
    abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2) = 71 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_71_l1898_189858


namespace NUMINAMATH_CALUDE_total_students_is_fifteen_l1898_189806

/-- The number of students originally in Class 1 -/
def n1 : ℕ := 8

/-- The number of students originally in Class 2 -/
def n2 : ℕ := 5

/-- Lei Lei's height in cm -/
def lei_lei_height : ℕ := 158

/-- Rong Rong's height in cm -/
def rong_rong_height : ℕ := 140

/-- The change in average height of Class 1 after the swap (in cm) -/
def class1_avg_change : ℚ := 2

/-- The change in average height of Class 2 after the swap (in cm) -/
def class2_avg_change : ℚ := 3

/-- The total number of students in both classes -/
def total_students : ℕ := n1 + n2 + 2

theorem total_students_is_fifteen :
  (lei_lei_height - rong_rong_height : ℚ) / (n1 + 1) = class1_avg_change ∧
  (lei_lei_height - rong_rong_height : ℚ) / (n2 + 1) = class2_avg_change →
  total_students = 15 := by sorry

end NUMINAMATH_CALUDE_total_students_is_fifteen_l1898_189806


namespace NUMINAMATH_CALUDE_valid_a_values_l1898_189834

theorem valid_a_values :
  ∀ a : ℚ, (∃ m : ℤ, a = m + 1/2 ∨ a = m + 1/3 ∨ a = m - 1/3) ↔
  ((∃ m : ℤ, a = m + 1/2) ∨ (∃ m : ℤ, a = m + 1/3) ∨ (∃ m : ℤ, a = m - 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_valid_a_values_l1898_189834


namespace NUMINAMATH_CALUDE_odd_function_extension_l1898_189826

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_extension 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_nonneg : ∀ x ≥ 0, f x = x * (1 + 3 * x)) :
  ∀ x < 0, f x = x * (1 - 3 * x) := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1898_189826


namespace NUMINAMATH_CALUDE_sock_pairs_problem_l1898_189841

theorem sock_pairs_problem (n : ℕ) : 
  (2 * n * (2 * n - 1)) / 2 = 6 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_problem_l1898_189841


namespace NUMINAMATH_CALUDE_average_daily_temp_range_l1898_189819

def high_temps : List ℝ := [49, 62, 58, 57, 46, 60, 55]
def low_temps : List ℝ := [40, 47, 45, 41, 39, 42, 44]

def daily_range (high low : List ℝ) : List ℝ :=
  List.zipWith (·-·) high low

theorem average_daily_temp_range :
  let ranges := daily_range high_temps low_temps
  (ranges.sum / ranges.length : ℝ) = 89 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_temp_range_l1898_189819


namespace NUMINAMATH_CALUDE_f_always_positive_implies_m_greater_than_e_range_of_y_when_f_has_two_zeros_l1898_189888

noncomputable section

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * Real.exp x - x - 2

-- Theorem 1: If f(x) > 0 for all x in ℝ, then m > e
theorem f_always_positive_implies_m_greater_than_e (m : ℝ) :
  (∀ x : ℝ, f m x > 0) → m > Real.exp 1 := by sorry

-- Theorem 2: Range of y when f has two zeros
theorem range_of_y_when_f_has_two_zeros (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  f m x₁ = 0 →
  f m x₂ = 0 →
  let y := (Real.exp x₂ - Real.exp x₁) * (1 / (Real.exp x₂ + Real.exp x₁) - m)
  ∀ z : ℝ, z < 0 ∧ ∃ (t : ℝ), y = t := by sorry

end

end NUMINAMATH_CALUDE_f_always_positive_implies_m_greater_than_e_range_of_y_when_f_has_two_zeros_l1898_189888


namespace NUMINAMATH_CALUDE_binomial_18_choose_4_l1898_189863

theorem binomial_18_choose_4 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_choose_4_l1898_189863


namespace NUMINAMATH_CALUDE_eight_b_value_l1898_189835

theorem eight_b_value (a b : ℚ) (h1 : 7 * a + 3 * b = 0) (h2 : a = b - 3) : 8 * b = 84/5 := by
  sorry

end NUMINAMATH_CALUDE_eight_b_value_l1898_189835


namespace NUMINAMATH_CALUDE_trajectory_and_max_area_l1898_189874

noncomputable section

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the ellipse
def on_ellipse (p : ℝ × ℝ) : Prop := p.1^2 / 2 + p.2^2 = 1

-- Define the relation between P and M
def P_relation (P M : ℝ × ℝ) : Prop := P.1 = 2 * M.1 ∧ P.2 = 2 * M.2

-- Define the trajectory C
def on_trajectory (p : ℝ × ℝ) : Prop := p.1^2 / 8 + p.2^2 / 4 = 1

-- Define the line l
def on_line (p : ℝ × ℝ) (m : ℝ) : Prop := p.2 = p.1 + m

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2)) / 2

theorem trajectory_and_max_area 
  (M : ℝ × ℝ) (P : ℝ × ℝ) (m : ℝ) (A B : ℝ × ℝ) :
  on_ellipse M → 
  P_relation P M → 
  m ≠ 0 →
  on_line A m →
  on_line B m →
  on_trajectory A →
  on_trajectory B →
  A ≠ B →
  (∀ P, P_relation P M → on_trajectory P) ∧
  (∀ X Y, on_trajectory X → on_trajectory Y → on_line X m → on_line Y m → 
    triangle_area O X Y ≤ 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_max_area_l1898_189874


namespace NUMINAMATH_CALUDE_multiple_of_9_digit_sum_possible_digits_for_multiple_of_9_l1898_189883

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.repr.data.map (λ c => c.toNat - '0'.toNat)
  digits.sum

theorem multiple_of_9_digit_sum (n : ℕ) : is_multiple_of_9 n ↔ is_multiple_of_9 (digit_sum n) := by sorry

theorem possible_digits_for_multiple_of_9 :
  ∀ d : ℕ, d < 10 →
    (is_multiple_of_9 (86300 + d * 10 + 7) ↔ d = 3 ∨ d = 9) := by sorry

end NUMINAMATH_CALUDE_multiple_of_9_digit_sum_possible_digits_for_multiple_of_9_l1898_189883


namespace NUMINAMATH_CALUDE_gage_skating_problem_l1898_189873

theorem gage_skating_problem (days_75min : ℕ) (days_90min : ℕ) (total_days : ℕ) (avg_minutes : ℕ) :
  days_75min = 5 →
  days_90min = 3 →
  total_days = days_75min + days_90min + 1 →
  avg_minutes = 85 →
  (days_75min * 75 + days_90min * 90 + (total_days * avg_minutes - (days_75min * 75 + days_90min * 90))) / total_days = avg_minutes :=
by sorry

end NUMINAMATH_CALUDE_gage_skating_problem_l1898_189873


namespace NUMINAMATH_CALUDE_amys_pencils_l1898_189828

/-- Amy's pencil counting problem -/
theorem amys_pencils (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 3 → bought = 7 → total = initial + bought → total = 10 := by
  sorry

end NUMINAMATH_CALUDE_amys_pencils_l1898_189828


namespace NUMINAMATH_CALUDE_min_teams_for_athletes_l1898_189898

theorem min_teams_for_athletes (total_athletes : ℕ) (max_per_team : ℕ) (h1 : total_athletes = 30) (h2 : max_per_team = 9) :
  ∃ (num_teams : ℕ) (athletes_per_team : ℕ),
    num_teams * athletes_per_team = total_athletes ∧
    athletes_per_team ≤ max_per_team ∧
    num_teams = 5 ∧
    ∀ (other_num_teams : ℕ) (other_athletes_per_team : ℕ),
      other_num_teams * other_athletes_per_team = total_athletes →
      other_athletes_per_team ≤ max_per_team →
      other_num_teams ≥ num_teams :=
by sorry

end NUMINAMATH_CALUDE_min_teams_for_athletes_l1898_189898


namespace NUMINAMATH_CALUDE_perimeter_of_fourth_figure_l1898_189840

/-- Given four planar figures composed of identical triangles, prove that the perimeter of the fourth figure is 10 cm. -/
theorem perimeter_of_fourth_figure
  (p₁ : ℝ) (p₂ : ℝ) (p₃ : ℝ) (p₄ : ℝ)
  (h₁ : p₁ = 8)
  (h₂ : p₂ = 11.4)
  (h₃ : p₃ = 14.7)
  (h_relation : p₁ + p₂ + p₄ = 2 * p₃) :
  p₄ = 10 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_fourth_figure_l1898_189840


namespace NUMINAMATH_CALUDE_inequality_proofs_l1898_189824

theorem inequality_proofs :
  (∀ x : ℝ, |x - 1| < 1 - 2*x ↔ x ∈ Set.Ioo 0 1) ∧
  (∀ x : ℝ, |x - 1| - |x + 1| > x ↔ x ∈ Set.Ioi (-1) ∪ Set.Ico (-1) 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l1898_189824


namespace NUMINAMATH_CALUDE_brick_width_l1898_189859

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: Given a rectangular prism with length 10, height 3, and surface area 164, the width must be 4 -/
theorem brick_width (l h : ℝ) (w : ℝ) 
  (h1 : l = 10) 
  (h2 : h = 3) 
  (h3 : surface_area l w h = 164) : w = 4 := by
  sorry

end NUMINAMATH_CALUDE_brick_width_l1898_189859


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1898_189830

/-- Given two hyperbolas with equations x²/9 - y²/16 = 1 and y²/25 - x²/M = 1,
    prove that if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/25 - x^2/M = 1) →
  (∀ k : ℝ, (∃ x y : ℝ, y = k*x ∧ x^2/9 - y^2/16 = 1) ↔
            (∃ x y : ℝ, y = k*x ∧ y^2/25 - x^2/M = 1)) →
  M = 225/16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1898_189830


namespace NUMINAMATH_CALUDE_set_of_integers_between_10_and_16_l1898_189881

def S : Set ℤ := {n | 10 < n ∧ n < 16}

theorem set_of_integers_between_10_and_16 : S = {11, 12, 13, 14, 15} := by
  sorry

end NUMINAMATH_CALUDE_set_of_integers_between_10_and_16_l1898_189881


namespace NUMINAMATH_CALUDE_sector_radius_l1898_189877

/-- Given a circular sector with area 11.25 cm² and arc length 4.5 cm, 
    the radius of the circle is 5 cm. -/
theorem sector_radius (area : ℝ) (arc_length : ℝ) (radius : ℝ) : 
  area = 11.25 → arc_length = 4.5 → area = (1/2) * radius * arc_length → radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l1898_189877


namespace NUMINAMATH_CALUDE_inverse_proportion_k_range_l1898_189867

/-- Prove that for an inverse proportion function y = (4-k)/x with points A(x₁, y₁) and B(x₂, y₂) 
    on its graph, where x₁ < 0 < x₂ and y₁ < y₂, the range of values for k is k < 4. -/
theorem inverse_proportion_k_range (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁ < 0) (h2 : 0 < x₂) (h3 : y₁ < y₂)
  (h4 : y₁ = (4 - k) / x₁) (h5 : y₂ = (4 - k) / x₂) :
  k < 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_range_l1898_189867


namespace NUMINAMATH_CALUDE_work_rate_problem_l1898_189805

/-- Given three workers with work rates satisfying certain conditions,
    prove that two of them together have a specific work rate. -/
theorem work_rate_problem (A B C : ℚ) 
  (h1 : A + B = 1/8)
  (h2 : A + B + C = 1/6)
  (h3 : A + C = 1/8) :
  B + C = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_work_rate_problem_l1898_189805


namespace NUMINAMATH_CALUDE_trig_identity_l1898_189802

theorem trig_identity (θ : Real) (h : Real.tan (θ - Real.pi) = 2) :
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1898_189802


namespace NUMINAMATH_CALUDE_inequality_proof_l1898_189869

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = a^2 + b^2 + c^2) : 
  a^2 / (a^2 + b*c) + b^2 / (b^2 + c*a) + c^2 / (c^2 + a*b) ≥ (a + b + c) / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1898_189869


namespace NUMINAMATH_CALUDE_special_school_student_count_l1898_189825

/-- Represents a school for deaf and blind students -/
structure School where
  deaf_students : ℕ
  blind_students : ℕ

/-- The total number of students in the school -/
def total_students (s : School) : ℕ := s.deaf_students + s.blind_students

/-- Theorem: Given a school where the deaf student population is three times 
    the size of the blind student population, and the number of deaf students 
    is 180, the total number of students is 240. -/
theorem special_school_student_count :
  ∀ (s : School),
  s.deaf_students = 180 →
  s.deaf_students = 3 * s.blind_students →
  total_students s = 240 := by
  sorry

end NUMINAMATH_CALUDE_special_school_student_count_l1898_189825


namespace NUMINAMATH_CALUDE_bracelet_selling_price_l1898_189880

theorem bracelet_selling_price 
  (total_bracelets : ℕ)
  (given_away : ℕ)
  (material_cost : ℚ)
  (profit : ℚ)
  (h1 : total_bracelets = 52)
  (h2 : given_away = 8)
  (h3 : material_cost = 3)
  (h4 : profit = 8) :
  let sold_bracelets := total_bracelets - given_away
  let total_sales := profit + material_cost
  let price_per_bracelet := total_sales / sold_bracelets
  price_per_bracelet = 1/4 := by
sorry

end NUMINAMATH_CALUDE_bracelet_selling_price_l1898_189880


namespace NUMINAMATH_CALUDE_pythagorean_triples_example_l1898_189875

-- Define a Pythagorean triple
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Define the two sets of triples
def triple1 : (ℕ × ℕ × ℕ) := (3, 4, 5)
def triple2 : (ℕ × ℕ × ℕ) := (6, 8, 10)

-- Theorem stating that both triples are Pythagorean triples
theorem pythagorean_triples_example :
  (is_pythagorean_triple triple1.1 triple1.2.1 triple1.2.2) ∧
  (is_pythagorean_triple triple2.1 triple2.2.1 triple2.2.2) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triples_example_l1898_189875


namespace NUMINAMATH_CALUDE_oranges_per_box_l1898_189884

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (h1 : total_oranges = 35) (h2 : num_boxes = 7) :
  total_oranges / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l1898_189884


namespace NUMINAMATH_CALUDE_water_distribution_l1898_189847

/-- A water distribution problem for four neighborhoods. -/
theorem water_distribution (total : ℕ) (left_for_fourth : ℕ) : 
  total = 1200 → 
  left_for_fourth = 350 → 
  ∃ (first second third fourth : ℕ),
    first + second + third + fourth = total ∧
    second = 2 * first ∧
    third = second + 100 ∧
    fourth = left_for_fourth ∧
    first = 150 := by
  sorry

end NUMINAMATH_CALUDE_water_distribution_l1898_189847


namespace NUMINAMATH_CALUDE_net_population_change_l1898_189879

def population_change (initial : ℝ) : ℝ :=
  initial * (1.2 * 0.9 * 1.3 * 0.85)

theorem net_population_change :
  ∀ initial : ℝ, initial > 0 →
  let final := population_change initial
  let percent_change := (final - initial) / initial * 100
  round percent_change = 51 := by
  sorry

#check net_population_change

end NUMINAMATH_CALUDE_net_population_change_l1898_189879


namespace NUMINAMATH_CALUDE_solve_for_q_l1898_189813

theorem solve_for_q (k l q : ℚ) : 
  (2/3 : ℚ) = k/45 ∧ (2/3 : ℚ) = (k+l)/75 ∧ (2/3 : ℚ) = (q-l)/105 → q = 90 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_q_l1898_189813


namespace NUMINAMATH_CALUDE_least_lcm_a_c_l1898_189852

theorem least_lcm_a_c (a b c : ℕ) 
  (h1 : Nat.lcm a b = 18) 
  (h2 : Nat.lcm b c = 20) : 
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 90 ∧ 
    (∀ (x y : ℕ), Nat.lcm x b = 18 → Nat.lcm b y = 20 → 
      Nat.lcm x y ≥ Nat.lcm a' c') := by
  sorry

end NUMINAMATH_CALUDE_least_lcm_a_c_l1898_189852


namespace NUMINAMATH_CALUDE_cafeteria_pies_l1898_189843

/-- Given a cafeteria with initial apples, apples handed out, and apples needed per pie,
    calculate the number of pies that can be made. -/
def calculate_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

/-- Theorem stating that with 62 initial apples, 8 apples handed out, and 9 apples per pie,
    the cafeteria can make 6 pies. -/
theorem cafeteria_pies :
  calculate_pies 62 8 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l1898_189843


namespace NUMINAMATH_CALUDE_class_size_l1898_189823

theorem class_size (n : ℕ) (best_rank : ℕ) (worst_rank : ℕ) 
  (h1 : best_rank = 30) 
  (h2 : worst_rank = 25) 
  (h3 : n = (best_rank - 1) + (worst_rank - 1) + 1) : 
  n = 54 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1898_189823


namespace NUMINAMATH_CALUDE_louis_oranges_l1898_189857

/-- Given the fruit distribution among Louis, Samantha, and Marley, prove that Louis has 5 oranges. -/
theorem louis_oranges :
  ∀ (louis_oranges louis_apples samantha_oranges samantha_apples marley_oranges marley_apples : ℕ),
  louis_apples = 3 →
  samantha_oranges = 8 →
  samantha_apples = 7 →
  marley_oranges = 2 * louis_oranges →
  marley_apples = 3 * samantha_apples →
  marley_oranges + marley_apples = 31 →
  louis_oranges = 5 := by
sorry


end NUMINAMATH_CALUDE_louis_oranges_l1898_189857


namespace NUMINAMATH_CALUDE_solution_set_correct_l1898_189892

-- Define the solution set
def solution_set : Set ℝ := {x | x < 0 ∨ x ≥ 1}

-- Define the inequality
def inequality (x : ℝ) : Prop := (1 - x) / x ≤ 0

-- Theorem stating that the solution set is correct
theorem solution_set_correct :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l1898_189892


namespace NUMINAMATH_CALUDE_latest_start_time_l1898_189832

def movie_start_time : ℕ := 20 -- 8 pm in 24-hour format
def home_time : ℕ := 17 -- 5 pm in 24-hour format
def dinner_duration : ℕ := 45
def homework_duration : ℕ := 30
def clean_room_duration : ℕ := 30
def trash_duration : ℕ := 5
def dishwasher_duration : ℕ := 10

def total_task_duration : ℕ := 
  dinner_duration + homework_duration + clean_room_duration + trash_duration + dishwasher_duration

theorem latest_start_time (start_time : ℕ) :
  start_time + total_task_duration / 60 = movie_start_time →
  start_time ≥ home_time →
  start_time = 18 := by sorry

end NUMINAMATH_CALUDE_latest_start_time_l1898_189832


namespace NUMINAMATH_CALUDE_modulus_of_z_l1898_189800

theorem modulus_of_z (z : ℂ) (h : (3 + 4 * Complex.I) * z = 1) : Complex.abs z = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1898_189800


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l1898_189889

theorem max_product_under_constraint :
  ∀ x y : ℕ, 27 * x + 35 * y ≤ 1000 →
  x * y ≤ 252 ∧ ∃ a b : ℕ, 27 * a + 35 * b ≤ 1000 ∧ a * b = 252 := by
sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l1898_189889


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_12n_integer_l1898_189803

theorem smallest_n_for_sqrt_12n_integer :
  ∀ n : ℕ+, (∃ k : ℕ+, (12 * n : ℕ) = k ^ 2) → n ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_12n_integer_l1898_189803


namespace NUMINAMATH_CALUDE_banana_price_is_five_l1898_189870

/-- Represents the market problem with Peter's purchases --/
def market_problem (banana_price : ℝ) : Prop :=
  let initial_money : ℝ := 500
  let potato_kilos : ℝ := 6
  let potato_price : ℝ := 2
  let tomato_kilos : ℝ := 9
  let tomato_price : ℝ := 3
  let cucumber_kilos : ℝ := 5
  let cucumber_price : ℝ := 4
  let banana_kilos : ℝ := 3
  let remaining_money : ℝ := 426
  initial_money - (potato_kilos * potato_price + tomato_kilos * tomato_price + 
    cucumber_kilos * cucumber_price + banana_kilos * banana_price) = remaining_money

/-- Theorem stating that the price per kilo of bananas is $5 --/
theorem banana_price_is_five : 
  ∃ (banana_price : ℝ), market_problem banana_price ∧ banana_price = 5 :=
sorry

end NUMINAMATH_CALUDE_banana_price_is_five_l1898_189870


namespace NUMINAMATH_CALUDE_largest_certain_divisor_l1898_189897

def is_valid_roll (roll : Finset Nat) : Prop :=
  roll.card = 7 ∧ roll ⊆ Finset.range 9 \ {0}

def product_of_roll (roll : Finset Nat) : Nat :=
  roll.prod id

theorem largest_certain_divisor :
  ∃ (n : Nat), n = 192 ∧
  (∀ (roll : Finset Nat), is_valid_roll roll → n ∣ product_of_roll roll) ∧
  (∀ (m : Nat), m > n →
    ∃ (roll : Finset Nat), is_valid_roll roll ∧ ¬(m ∣ product_of_roll roll)) :=
sorry

end NUMINAMATH_CALUDE_largest_certain_divisor_l1898_189897


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_l1898_189876

theorem a_gt_one_sufficient_not_necessary (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ ∃ b : ℝ, 1/b < 1 ∧ ¬(b > 1) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_l1898_189876


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l1898_189831

theorem cube_volume_surface_area (x : ℝ) :
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l1898_189831


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l1898_189891

theorem arithmetic_simplification : (4 + 4 + 6) / 2 - 2 / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l1898_189891


namespace NUMINAMATH_CALUDE_wilfred_carrots_l1898_189817

/-- The number of carrots Wilfred ate on Tuesday, Wednesday, and Thursday -/
def carrots_tuesday : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun tuesday wednesday thursday total =>
    tuesday + wednesday + thursday = total

theorem wilfred_carrots :
  ∃ (tuesday : ℕ),
    carrots_tuesday tuesday 6 5 15 ∧ tuesday = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_wilfred_carrots_l1898_189817


namespace NUMINAMATH_CALUDE_evan_future_books_l1898_189868

/-- Calculates the number of books Evan will have in 5 years -/
def books_in_five_years (books_two_years_ago : ℕ) : ℕ :=
  let current_books := books_two_years_ago - 40
  5 * current_books + 60

/-- Proves that Evan will have 860 books in 5 years -/
theorem evan_future_books :
  books_in_five_years 200 = 860 := by
  sorry

#eval books_in_five_years 200

end NUMINAMATH_CALUDE_evan_future_books_l1898_189868


namespace NUMINAMATH_CALUDE_hiram_age_is_40_l1898_189862

/-- Hiram's age in years -/
def hiram_age : ℕ := sorry

/-- Allyson's age in years -/
def allyson_age : ℕ := 28

/-- Theorem stating Hiram's age based on the given conditions -/
theorem hiram_age_is_40 :
  (hiram_age + 12 = 2 * allyson_age - 4) → hiram_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_hiram_age_is_40_l1898_189862


namespace NUMINAMATH_CALUDE_special_triangle_properties_l1898_189804

open Real

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  acute_A : 0 < A ∧ A < π/2
  acute_B : 0 < B ∧ B < π/2
  acute_C : 0 < C ∧ C < π/2
  angle_sum : A + B + C = π

-- Define the specific conditions of the triangle
def SpecialTriangle (t : AcuteTriangle) : Prop :=
  t.B = 2 * t.A ∧ sin t.A ≠ 0 ∧ cos t.A ≠ 0

-- State the theorems to be proved
theorem special_triangle_properties (t : AcuteTriangle) (h : SpecialTriangle t) :
  ∃ (AC : ℝ), 
    AC / cos t.A = 2 ∧ 
    sqrt 2 < AC ∧ 
    AC < sqrt 3 := by sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l1898_189804


namespace NUMINAMATH_CALUDE_sine_function_inequality_l1898_189818

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x) * Real.cos φ + Real.cos (2 * x) * Real.sin φ

theorem sine_function_inequality 
  (φ : ℝ) 
  (h : ∀ x : ℝ, f x φ ≤ f (2 * Real.pi / 9) φ) : 
  f (2 * Real.pi / 3) φ < f (5 * Real.pi / 6) φ ∧ 
  f (5 * Real.pi / 6) φ < f (7 * Real.pi / 6) φ :=
sorry

end NUMINAMATH_CALUDE_sine_function_inequality_l1898_189818


namespace NUMINAMATH_CALUDE_distribution_count_is_18_l1898_189801

/-- The number of ways to distribute 6 numbered balls into 3 boxes -/
def distributionCount : ℕ :=
  let totalBalls : ℕ := 6
  let numBoxes : ℕ := 3
  let ballsPerBox : ℕ := 2
  let fixedPair : Fin totalBalls := 2  -- Represents balls 1 and 2 as a fixed pair
  18

/-- Theorem stating that the number of distributions is 18 -/
theorem distribution_count_is_18 : distributionCount = 18 := by
  sorry

end NUMINAMATH_CALUDE_distribution_count_is_18_l1898_189801


namespace NUMINAMATH_CALUDE_tank_capacity_l1898_189837

theorem tank_capacity (initial_fullness final_fullness : ℚ) (added_water : ℕ) : 
  initial_fullness = 1/4 →
  final_fullness = 3/4 →
  added_water = 208 →
  (final_fullness - initial_fullness) * (added_water / (final_fullness - initial_fullness)) = 416 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l1898_189837


namespace NUMINAMATH_CALUDE_target_state_reachable_l1898_189816

/-- Represents the state of the urn -/
structure UrnState where
  black : ℕ
  white : ℕ

/-- Represents the possible operations on the urn -/
inductive Operation
  | replaceBlacks
  | replaceBlackWhite
  | replaceWhiteBlack
  | replaceWhites

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.replaceBlacks => 
      if state.black ≥ 3 then UrnState.mk (state.black - 1) state.white
      else state
  | Operation.replaceBlackWhite => 
      if state.black ≥ 2 && state.white ≥ 1 then UrnState.mk (state.black - 2) (state.white + 1)
      else state
  | Operation.replaceWhiteBlack => 
      if state.black ≥ 1 && state.white ≥ 2 then UrnState.mk state.black (state.white - 1)
      else state
  | Operation.replaceWhites => 
      if state.white ≥ 3 then UrnState.mk (state.black + 1) (state.white - 3)
      else state

/-- Checks if the target state is reachable from the initial state -/
def isReachable (initial : UrnState) (target : UrnState) : Prop :=
  ∃ (sequence : List Operation), 
    List.foldl applyOperation initial sequence = target

/-- The main theorem stating that the target state is reachable -/
theorem target_state_reachable : 
  isReachable (UrnState.mk 80 120) (UrnState.mk 1 2) := by
  sorry

end NUMINAMATH_CALUDE_target_state_reachable_l1898_189816


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l1898_189885

/-- The number of distinct arrangements of n distinct beads on a bracelet,
    considering rotational and reflectional symmetries --/
def braceletArrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of distinct arrangements of 8 distinct beads
    on a bracelet, considering rotational and reflectional symmetries, is 2520 --/
theorem eight_bead_bracelet_arrangements :
  braceletArrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l1898_189885


namespace NUMINAMATH_CALUDE_minimum_total_cost_l1898_189850

-- Define the ticket prices
def price_cheap : ℕ := 60
def price_expensive : ℕ := 100

-- Define the total number of tickets
def total_tickets : ℕ := 140

-- Define the function to calculate the total cost
def total_cost (cheap_tickets expensive_tickets : ℕ) : ℕ :=
  cheap_tickets * price_cheap + expensive_tickets * price_expensive

-- State the theorem
theorem minimum_total_cost :
  ∃ (cheap_tickets expensive_tickets : ℕ),
    cheap_tickets + expensive_tickets = total_tickets ∧
    expensive_tickets ≥ 2 * cheap_tickets ∧
    ∀ (c e : ℕ),
      c + e = total_tickets →
      e ≥ 2 * c →
      total_cost cheap_tickets expensive_tickets ≤ total_cost c e ∧
      total_cost cheap_tickets expensive_tickets = 12160 :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_total_cost_l1898_189850


namespace NUMINAMATH_CALUDE_route_length_l1898_189854

/-- Proves that the length of a route is 125 miles given the conditions of two trains meeting. -/
theorem route_length (time_A time_B meeting_distance : ℝ) 
  (h1 : time_A = 12)
  (h2 : time_B = 8)
  (h3 : meeting_distance = 50)
  (h4 : time_A > 0)
  (h5 : time_B > 0)
  (h6 : meeting_distance > 0) :
  ∃ (route_length : ℝ),
    route_length = 125 ∧
    route_length / time_A * (meeting_distance * time_A / route_length) = meeting_distance ∧
    route_length / time_B * (meeting_distance * time_A / route_length) = route_length - meeting_distance :=
by
  sorry


end NUMINAMATH_CALUDE_route_length_l1898_189854


namespace NUMINAMATH_CALUDE_junior_prom_dancer_ratio_l1898_189882

theorem junior_prom_dancer_ratio :
  let total_kids : ℕ := 140
  let slow_dancers : ℕ := 25
  let non_slow_dancers : ℕ := 10
  let total_dancers : ℕ := slow_dancers + non_slow_dancers
  (total_dancers : ℚ) / total_kids = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_junior_prom_dancer_ratio_l1898_189882


namespace NUMINAMATH_CALUDE_arrangement_count_l1898_189808

/-- The number of people in the row -/
def n : ℕ := 5

/-- The number of arrangements where A and B are adjacent -/
def adjacent_arrangements : ℕ := 48

/-- The total number of arrangements of n people -/
def total_arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of arrangements where A and B are not adjacent -/
def non_adjacent_arrangements (n : ℕ) : ℕ := total_arrangements n - adjacent_arrangements

/-- The number of arrangements where A and B are not adjacent and A is to the left of B -/
def target_arrangements (n : ℕ) : ℕ := non_adjacent_arrangements n / 2

theorem arrangement_count : target_arrangements n = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l1898_189808


namespace NUMINAMATH_CALUDE_no_prime_pair_with_odd_difference_quotient_l1898_189866

theorem no_prime_pair_with_odd_difference_quotient :
  ¬ ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p > q ∧ (∃ (k : ℕ), 2 * k + 1 = (p^2 - q^2) / 4) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_pair_with_odd_difference_quotient_l1898_189866


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1898_189853

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1898_189853


namespace NUMINAMATH_CALUDE_bonus_remainder_l1898_189890

theorem bonus_remainder (X : ℕ) (h : X % 5 = 2) : (3 * X) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_bonus_remainder_l1898_189890
