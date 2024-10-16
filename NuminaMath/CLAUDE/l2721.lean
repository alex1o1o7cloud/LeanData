import Mathlib

namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2721_272134

-- Define the line
def line (x : ℝ) : ℝ := -x + 1

-- Define the third quadrant
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem statement
theorem line_not_in_third_quadrant :
  ∀ x : ℝ, ¬(third_quadrant x (line x)) :=
by
  sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2721_272134


namespace NUMINAMATH_CALUDE_integer_sum_problem_l2721_272138

theorem integer_sum_problem (x y : ℤ) : 
  (x = 15 ∨ y = 15) → (4 * x + 3 * y = 150) → (x = 30 ∨ y = 30) := by
sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l2721_272138


namespace NUMINAMATH_CALUDE_third_one_is_13th_a_2015_is_31_l2721_272150

-- Define the sequence a_n
def a : ℕ → ℚ
| 0 => 0  -- We start from index 1, so define 0 as a placeholder
| n => 
  let k := (n.sqrt + 1) / 2  -- Calculate which group the term belongs to
  let m := n - (k - 1) * k   -- Calculate position within the group
  m.succ / (k + 1 - m)       -- Return the fraction

-- Third term equal to 1
theorem third_one_is_13th : ∃ n₁ n₂ : ℕ, n₁ < n₂ ∧ n₂ < 13 ∧ a n₁ = 1 ∧ a n₂ = 1 ∧ a 13 = 1 :=
sorry

-- 2015th term
theorem a_2015_is_31 : a 2015 = 31 :=
sorry

end NUMINAMATH_CALUDE_third_one_is_13th_a_2015_is_31_l2721_272150


namespace NUMINAMATH_CALUDE_no_rational_sqrt_sin_cos_l2721_272101

theorem no_rational_sqrt_sin_cos : 
  ¬ ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ 
    ∃ (a b c d : ℕ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∧
    (Real.sqrt (Real.sin θ) = a / b) ∧ 
    (Real.sqrt (Real.cos θ) = c / d) :=
by sorry

end NUMINAMATH_CALUDE_no_rational_sqrt_sin_cos_l2721_272101


namespace NUMINAMATH_CALUDE_s_range_l2721_272104

theorem s_range (a b c : ℝ) 
  (ha : 1/2 ≤ a ∧ a ≤ 1) 
  (hb : 1/2 ≤ b ∧ b ≤ 1) 
  (hc : 1/2 ≤ c ∧ c ≤ 1) : 
  let s := (a + b) / (1 + c) + (b + c) / (1 + a) + (c + a) / (1 + b)
  2 ≤ s ∧ s ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_s_range_l2721_272104


namespace NUMINAMATH_CALUDE_solve_for_z_l2721_272198

theorem solve_for_z (m : ℕ) (z : ℝ) 
  (h1 : ((1 ^ (m + 1)) / (5 ^ (m + 1))) * ((1 ^ 18) / (z ^ 18)) = 1 / (2 * (10 ^ 35)))
  (h2 : m = 34) : 
  z = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_for_z_l2721_272198


namespace NUMINAMATH_CALUDE_bicycle_trip_average_speed_l2721_272175

/-- Calculates the average speed of a bicycle trip with varying conditions -/
theorem bicycle_trip_average_speed :
  let total_distance : ℝ := 500
  let flat_road_distance : ℝ := 100
  let flat_road_speed : ℝ := 20
  let uphill_distance : ℝ := 50
  let uphill_speed : ℝ := 10
  let flat_terrain_distance : ℝ := 200
  let flat_terrain_speed : ℝ := 15
  let headwind_distance : ℝ := 150
  let headwind_speed : ℝ := 12
  let rest_time_1 : ℝ := 0.5  -- 30 minutes in hours
  let rest_time_2 : ℝ := 1/3  -- 20 minutes in hours
  let rest_time_3 : ℝ := 2/3  -- 40 minutes in hours
  
  let total_time : ℝ := 
    flat_road_distance / flat_road_speed +
    uphill_distance / uphill_speed +
    flat_terrain_distance / flat_terrain_speed +
    headwind_distance / headwind_speed +
    rest_time_1 + rest_time_2 + rest_time_3
  
  let average_speed : ℝ := total_distance / total_time
  
  ∃ ε > 0, |average_speed - 13.4| < ε :=
by sorry

end NUMINAMATH_CALUDE_bicycle_trip_average_speed_l2721_272175


namespace NUMINAMATH_CALUDE_circle_area_increase_l2721_272137

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_increase_l2721_272137


namespace NUMINAMATH_CALUDE_minimum_additional_amount_l2721_272196

def current_order : ℝ := 49.90
def discount_rate : ℝ := 0.10
def target_amount : ℝ := 50.00

theorem minimum_additional_amount :
  ∃ (x : ℝ), x ≥ 0 ∧
  (current_order + x) * (1 - discount_rate) = target_amount ∧
  ∀ (y : ℝ), y ≥ 0 → (current_order + y) * (1 - discount_rate) ≥ target_amount → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_minimum_additional_amount_l2721_272196


namespace NUMINAMATH_CALUDE_journey_remaining_distance_l2721_272147

/-- The remaining distance to be driven in a journey. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Proof that the remaining distance is 3610 miles. -/
theorem journey_remaining_distance :
  remaining_distance 9475 5865 = 3610 := by
  sorry

end NUMINAMATH_CALUDE_journey_remaining_distance_l2721_272147


namespace NUMINAMATH_CALUDE_pages_read_l2721_272115

/-- Given a book with a total number of pages and the number of pages left to read,
    calculate the number of pages already read. -/
theorem pages_read (total_pages left_to_read : ℕ) : 
  total_pages = 17 → left_to_read = 6 → total_pages - left_to_read = 11 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_l2721_272115


namespace NUMINAMATH_CALUDE_missing_consonants_fraction_l2721_272103

theorem missing_consonants_fraction 
  (total_letters : ℕ) 
  (total_vowels : ℕ) 
  (total_missing : ℕ) 
  (missing_vowels : ℕ) 
  (h1 : total_letters = 26) 
  (h2 : total_vowels = 5) 
  (h3 : total_missing = 5) 
  (h4 : missing_vowels = 2) :
  (total_missing - missing_vowels) / (total_letters - total_vowels) = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_missing_consonants_fraction_l2721_272103


namespace NUMINAMATH_CALUDE_painted_subcubes_count_l2721_272131

def cube_size : ℕ := 4

-- Define a function to calculate the number of subcubes with at least two painted faces
def subcubes_with_two_or_more_painted_faces (n : ℕ) : ℕ :=
  8 + 12 * (n - 2)

-- Theorem statement
theorem painted_subcubes_count :
  subcubes_with_two_or_more_painted_faces cube_size = 32 := by
  sorry

end NUMINAMATH_CALUDE_painted_subcubes_count_l2721_272131


namespace NUMINAMATH_CALUDE_halloween_cleanup_time_halloween_cleanup_time_specific_l2721_272146

/-- Calculates the total cleaning time for Halloween vandalism -/
theorem halloween_cleanup_time 
  (egg_cleanup_time : ℕ) 
  (tp_cleanup_time : ℕ) 
  (num_eggs : ℕ) 
  (num_tp : ℕ) : ℕ :=
  let egg_time_seconds := egg_cleanup_time * num_eggs
  let egg_time_minutes := egg_time_seconds / 60
  let tp_time_minutes := tp_cleanup_time * num_tp
  egg_time_minutes + tp_time_minutes

/-- Proves that the total cleaning time for 60 eggs and 7 rolls of toilet paper is 225 minutes -/
theorem halloween_cleanup_time_specific : 
  halloween_cleanup_time 15 30 60 7 = 225 := by
  sorry

end NUMINAMATH_CALUDE_halloween_cleanup_time_halloween_cleanup_time_specific_l2721_272146


namespace NUMINAMATH_CALUDE_expression_simplification_l2721_272165

theorem expression_simplification (y : ℝ) :
  3 * y - 2 * y^2 + 4 - (5 - 3 * y + 2 * y^2 - y^3) = y^3 + 6 * y - 4 * y^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2721_272165


namespace NUMINAMATH_CALUDE_problem_solution_l2721_272160

theorem problem_solution (x y : ℝ) 
  (sum_eq : x + y = 360)
  (ratio_eq : x / y = 3 / 5) : 
  y - x = 90 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2721_272160


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l2721_272120

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate its speed in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 20)
  (h2 : average_speed = 40) : 
  ∃ (speed_second_hour : ℝ), speed_second_hour = 60 := by
  sorry

#check car_speed_second_hour

end NUMINAMATH_CALUDE_car_speed_second_hour_l2721_272120


namespace NUMINAMATH_CALUDE_min_value_on_line_l2721_272187

/-- Given a point A(m,n) on the line x + 2y = 1 where m > 0 and n > 0,
    the minimum value of 2/m + 1/n is 8 -/
theorem min_value_on_line (m n : ℝ) (h1 : m + 2*n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∀ (x y : ℝ), x + 2*y = 1 → x > 0 → y > 0 → 2/m + 1/n ≤ 2/x + 1/y :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_line_l2721_272187


namespace NUMINAMATH_CALUDE_complex_parts_of_z_l2721_272139

theorem complex_parts_of_z : ∃ (z : ℂ), z = 2 - 3 * I ∧ z.re = 2 ∧ z.im = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_parts_of_z_l2721_272139


namespace NUMINAMATH_CALUDE_triangle_side_length_l2721_272100

-- Define the triangle ABC
def triangle_ABC (a : ℕ) : Prop :=
  ∃ (A B C : ℝ × ℝ),
    let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
    let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
    let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
    AB = 1 ∧ BC = 2007 ∧ AC = a

-- Theorem statement
theorem triangle_side_length :
  ∀ a : ℕ, triangle_ABC a → a = 2007 :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l2721_272100


namespace NUMINAMATH_CALUDE_function_property_l2721_272185

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 2) + f x = 3) 
  (h2 : f 1 = 0) : 
  f 2023 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2721_272185


namespace NUMINAMATH_CALUDE_londolozi_lion_cubs_per_month_l2721_272178

/-- The number of lion cubs born per month in Londolozi -/
def lion_cubs_per_month (initial_population final_population : ℕ) (months death_rate : ℕ) : ℕ :=
  (final_population - initial_population + months * death_rate) / months

/-- Theorem stating the number of lion cubs born per month in Londolozi -/
theorem londolozi_lion_cubs_per_month :
  lion_cubs_per_month 100 148 12 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_londolozi_lion_cubs_per_month_l2721_272178


namespace NUMINAMATH_CALUDE_certain_number_problem_l2721_272112

theorem certain_number_problem (x : ℝ) : 0.7 * x = 0.6 * 80 + 22 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2721_272112


namespace NUMINAMATH_CALUDE_units_digit_17_pow_2023_l2721_272181

theorem units_digit_17_pow_2023 :
  ∃ (n : ℕ), n < 10 ∧ 17^2023 ≡ n [ZMOD 10] ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_2023_l2721_272181


namespace NUMINAMATH_CALUDE_min_value_of_f_l2721_272130

def f (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem min_value_of_f :
  ∃ (x_min : ℝ), (∀ x, f x ≥ f x_min) ∧ (x_min = -1) ∧ (f x_min = 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2721_272130


namespace NUMINAMATH_CALUDE_sams_carrots_l2721_272143

theorem sams_carrots (sandy_carrots : ℕ) (total_carrots : ℕ) (h1 : sandy_carrots = 6) (h2 : total_carrots = 9) : 
  total_carrots - sandy_carrots = 3 := by
  sorry

end NUMINAMATH_CALUDE_sams_carrots_l2721_272143


namespace NUMINAMATH_CALUDE_ninth_grade_students_l2721_272188

/-- Proves that given a total of 50 students from three grades, with the seventh grade having 2x - 1 students and the eighth grade having x students, the number of students in the ninth grade is 51 - 3x. -/
theorem ninth_grade_students (x : ℕ) : 
  (50 : ℕ) = (2 * x - 1) + x + (51 - 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_students_l2721_272188


namespace NUMINAMATH_CALUDE_sliced_meat_variety_pack_l2721_272133

theorem sliced_meat_variety_pack :
  let base_cost : ℚ := 40
  let rush_delivery_rate : ℚ := 0.3
  let cost_per_type : ℚ := 13
  let total_cost : ℚ := base_cost * (1 + rush_delivery_rate)
  (total_cost / cost_per_type : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sliced_meat_variety_pack_l2721_272133


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_property_l2721_272171

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  is_isosceles : (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

/-- Distance squared between two points -/
def dist_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

/-- The theorem to be proved -/
theorem isosceles_right_triangle_property (triangle : IsoscelesRightTriangle) :
  ∀ P : ℝ × ℝ, (P.2 = triangle.A.2 ∧ P.2 = triangle.B.2) →
    dist_squared P triangle.A + dist_squared P triangle.B = 2 * dist_squared P triangle.C :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_property_l2721_272171


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2721_272199

theorem complex_expression_equality : ∀ (a b : ℂ), 
  a = 3 - 2*I ∧ b = -2 + 3*I → 3*a + 4*b = 1 + 6*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2721_272199


namespace NUMINAMATH_CALUDE_second_bus_students_l2721_272168

theorem second_bus_students (first_bus : ℕ) (second_bus : ℕ) : 
  first_bus = 38 →
  second_bus - 4 = (first_bus + 4) + 2 →
  second_bus = 44 := by
sorry

end NUMINAMATH_CALUDE_second_bus_students_l2721_272168


namespace NUMINAMATH_CALUDE_total_sandwiches_l2721_272189

/-- The number of sandwiches made by each person and the total -/
def sandwiches : ℕ → ℕ
| 0 => 49  -- Billy
| 1 => 49 + (49 * 3 / 10)  -- Katelyn
| 2 => (sandwiches 1 * 3) / 5  -- Chloe
| 3 => 25  -- Emma
| 4 => 25 * 2  -- Stella
| _ => 0

/-- The theorem stating the total number of sandwiches made -/
theorem total_sandwiches : 
  sandwiches 0 + sandwiches 1 + sandwiches 2 + sandwiches 3 + sandwiches 4 = 226 := by
  sorry


end NUMINAMATH_CALUDE_total_sandwiches_l2721_272189


namespace NUMINAMATH_CALUDE_expand_expression_l2721_272145

theorem expand_expression (x : ℝ) : (3*x^2 + 2*x - 4)*(x - 3) = 3*x^3 - 7*x^2 - 10*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2721_272145


namespace NUMINAMATH_CALUDE_inscribed_squares_perimeter_ratio_l2721_272158

theorem inscribed_squares_perimeter_ratio :
  let r : ℝ := 5
  let s₁ : ℝ := Real.sqrt ((2 * r^2) / 5)  -- side length of square in semicircle
  let s₂ : ℝ := r * Real.sqrt 2           -- side length of square in circle
  (4 * s₁) / (4 * s₂) = Real.sqrt 10 / 5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_perimeter_ratio_l2721_272158


namespace NUMINAMATH_CALUDE_price_increase_x_l2721_272172

/-- The annual price increase of commodity x -/
def annual_increase_x : ℚ := 30 / 100

/-- The annual price increase of commodity y -/
def annual_increase_y : ℚ := 20 / 100

/-- The price of commodity x in 2001 -/
def price_x_2001 : ℚ := 420 / 100

/-- The price of commodity y in 2001 -/
def price_y_2001 : ℚ := 440 / 100

/-- The number of years between 2001 and 2012 -/
def years : ℕ := 11

/-- The difference in price between commodities x and y in 2012 -/
def price_difference_2012 : ℚ := 90 / 100

theorem price_increase_x : 
  annual_increase_x * years + price_x_2001 = 
  annual_increase_y * years + price_y_2001 + price_difference_2012 :=
sorry

end NUMINAMATH_CALUDE_price_increase_x_l2721_272172


namespace NUMINAMATH_CALUDE_lincoln_county_houses_l2721_272119

/-- The number of houses in Lincoln County after a housing boom -/
def houses_after_boom (original : ℕ) (new_built : ℕ) : ℕ :=
  original + new_built

/-- Theorem: The number of houses in Lincoln County after the housing boom is 118558 -/
theorem lincoln_county_houses :
  houses_after_boom 20817 97741 = 118558 := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_houses_l2721_272119


namespace NUMINAMATH_CALUDE_line_slope_l2721_272174

/-- The slope of a line given by the equation (x/2) + (y/3) = 1 is -3/2 -/
theorem line_slope : 
  let line_eq : ℝ → ℝ → Prop := λ x y ↦ (x / 2 + y / 3 = 1)
  ∃ m b : ℝ, (∀ x y, line_eq x y ↔ y = m * x + b) ∧ m = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l2721_272174


namespace NUMINAMATH_CALUDE_puzzle_solution_l2721_272127

theorem puzzle_solution :
  ∀ (F I V T E N : ℕ),
    F = 8 →
    N % 2 = 1 →
    F ≠ I ∧ F ≠ V ∧ F ≠ T ∧ F ≠ E ∧ F ≠ N ∧
    I ≠ V ∧ I ≠ T ∧ I ≠ E ∧ I ≠ N ∧
    V ≠ T ∧ V ≠ E ∧ V ≠ N ∧
    T ≠ E ∧ T ≠ N ∧
    E ≠ N →
    F < 10 ∧ I < 10 ∧ V < 10 ∧ T < 10 ∧ E < 10 ∧ N < 10 →
    100 * F + 10 * I + V + 100 * F + 10 * I + V = 1000 * T + 100 * E + 10 * N →
    I = 4 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2721_272127


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_100_l2721_272121

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_greater_than_100 : 
  (¬ ∃ n : ℕ, 2^n > 100) ↔ (∀ n : ℕ, 2^n ≤ 100) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_100_l2721_272121


namespace NUMINAMATH_CALUDE_units_digit_of_k_cubed_plus_five_to_k_l2721_272170

theorem units_digit_of_k_cubed_plus_five_to_k (k : ℕ) : 
  k = 2024^2 + 3^2024 → (k^3 + 5^k) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_cubed_plus_five_to_k_l2721_272170


namespace NUMINAMATH_CALUDE_local_max_at_two_l2721_272176

/-- The function f(x) = x(x-c)² has a local maximum at x=2 if and only if c = 6 -/
theorem local_max_at_two (c : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), x * (x - c)^2 ≤ 2 * (2 - c)^2) ↔ c = 6 := by
  sorry

end NUMINAMATH_CALUDE_local_max_at_two_l2721_272176


namespace NUMINAMATH_CALUDE_investment_growth_l2721_272111

/-- Proves that an initial investment of $400, when compounded annually at 12% interest for 5 years, results in a final value of $704.98. -/
theorem investment_growth (initial_investment : ℝ) (interest_rate : ℝ) (years : ℕ) (final_value : ℝ) :
  initial_investment = 400 →
  interest_rate = 0.12 →
  years = 5 →
  final_value = 704.98 →
  final_value = initial_investment * (1 + interest_rate) ^ years := by
  sorry


end NUMINAMATH_CALUDE_investment_growth_l2721_272111


namespace NUMINAMATH_CALUDE_five_students_three_companies_l2721_272117

/-- The number of ways to assign n students to k companies, where each company must receive at least one student. -/
def assignment_count (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2)

/-- Theorem stating that the number of ways to assign 5 students to 3 companies, where each company must receive at least one student, is 150. -/
theorem five_students_three_companies : assignment_count 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_students_three_companies_l2721_272117


namespace NUMINAMATH_CALUDE_unique_solution_l2721_272116

/-- Represents the ages of two sons given specific conditions --/
structure SonsAges where
  elder : ℕ
  younger : ℕ
  doubled_elder_exceeds_sum : 2 * elder = elder + younger + 18
  younger_less_than_difference : younger = elder - younger - 6

/-- The unique solution to the SonsAges problem --/
def solution : SonsAges := { 
  elder := 30,
  younger := 12,
  doubled_elder_exceeds_sum := by sorry,
  younger_less_than_difference := by sorry
}

/-- Proves that the solution is unique --/
theorem unique_solution (s : SonsAges) : s = solution := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2721_272116


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2721_272182

/-- An isosceles triangle with congruent sides of 8 cm and perimeter of 25 cm has a base of 9 cm -/
theorem isosceles_triangle_base_length : 
  ∀ (base congruent_side : ℝ),
  congruent_side = 8 →
  base + 2 * congruent_side = 25 →
  base = 9 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2721_272182


namespace NUMINAMATH_CALUDE_lego_airplane_model_l2721_272184

theorem lego_airplane_model (total_legos : ℕ) (additional_legos : ℕ) (num_models : ℕ) :
  total_legos = 400 →
  additional_legos = 80 →
  num_models = 2 →
  (total_legos + additional_legos) / num_models = 240 :=
by sorry

end NUMINAMATH_CALUDE_lego_airplane_model_l2721_272184


namespace NUMINAMATH_CALUDE_chromatic_flow_duality_l2721_272149

/-- A planar multigraph -/
structure PlanarMultigraph where
  -- Add necessary fields

/-- The dual of a planar multigraph -/
def dual (G : PlanarMultigraph) : PlanarMultigraph :=
  sorry

/-- The chromatic number of a planar multigraph -/
def chromaticNumber (G : PlanarMultigraph) : ℕ :=
  sorry

/-- The flow number of a planar multigraph -/
def flowNumber (G : PlanarMultigraph) : ℕ :=
  sorry

/-- Theorem: The chromatic number of a planar multigraph equals the flow number of its dual -/
theorem chromatic_flow_duality (G : PlanarMultigraph) :
    chromaticNumber G = flowNumber (dual G) :=
  sorry

end NUMINAMATH_CALUDE_chromatic_flow_duality_l2721_272149


namespace NUMINAMATH_CALUDE_sum_of_ages_l2721_272122

/-- Given the ages of Masc and Sam, prove that the sum of their ages is 27. -/
theorem sum_of_ages (Masc_age Sam_age : ℕ) 
  (h1 : Masc_age = Sam_age + 7)
  (h2 : Masc_age = 17)
  (h3 : Sam_age = 10) : 
  Masc_age + Sam_age = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2721_272122


namespace NUMINAMATH_CALUDE_candy_bar_consumption_l2721_272126

theorem candy_bar_consumption (calories_per_bar : ℕ) (total_calories : ℕ) (num_bars : ℕ) : 
  calories_per_bar = 8 → total_calories = 24 → num_bars = total_calories / calories_per_bar → num_bars = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_bar_consumption_l2721_272126


namespace NUMINAMATH_CALUDE_quadratic_function_form_l2721_272169

/-- A quadratic function with two equal real roots and derivative 2x + 2 -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
  equal_roots : ∃ (r : ℝ), (∀ x, f x = 0 ↔ x = r)
  derivative : ∀ x, deriv f x = 2 * x + 2

/-- The quadratic function with the given properties is x^2 + 2x + 1 -/
theorem quadratic_function_form (qf : QuadraticFunction) : 
  ∀ x, qf.f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_form_l2721_272169


namespace NUMINAMATH_CALUDE_quadratic_inequality_sum_l2721_272106

/-- Given a quadratic inequality ax^2 - 5x + b > 0 with solution set {x | -3 < x < 2}, prove that a + b = 25 -/
theorem quadratic_inequality_sum (a b : ℝ) : 
  (∀ x, ax^2 - 5*x + b > 0 ↔ -3 < x ∧ x < 2) → 
  a + b = 25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_sum_l2721_272106


namespace NUMINAMATH_CALUDE_sufficient_questions_sufficient_questions_10n_l2721_272135

/-- Represents the origin of a scientist -/
inductive Origin
| Piripocs
| Nekeresd

/-- Represents a scientist at the congress -/
structure Scientist where
  origin : Origin

/-- Represents the congress of scientists -/
structure Congress where
  n : ℕ
  scientists : Fin n → Scientist
  more_piripocs : ∃ (p : ℕ), 2 * p > n ∧ (∀ i : Fin n, (scientists i).origin = Origin.Piripocs → i.val < p)

/-- Function to ask a question about a scientist's origin -/
def ask_question (c : Congress) (asker : Fin c.n) (about : Fin c.n) : Origin :=
  match (c.scientists asker).origin with
  | Origin.Piripocs => (c.scientists about).origin
  | Origin.Nekeresd => sorry  -- This can be either true or false

/-- Theorem stating that n^2 / 2 questions are sufficient -/
theorem sufficient_questions (c : Congress) :
  ∃ (strategy : (Fin c.n → Fin c.n → Origin) → Fin c.n → Origin),
    (∀ f : Fin c.n → Fin c.n → Origin, 
      (∀ i j : Fin c.n, f i j = ask_question c i j) → 
      (∀ i : Fin c.n, strategy f i = (c.scientists i).origin)) ∧
    (∃ m : ℕ, 2 * m ≤ c.n * c.n ∧ 
      ∀ f : Fin c.n → Fin c.n → Origin, 
        (∃ s : Finset (Fin c.n × Fin c.n), s.card ≤ m ∧ 
          ∀ i j : Fin c.n, f i j = ask_question c i j → (i, j) ∈ s)) :=
sorry

/-- Theorem stating that 10n questions are also sufficient -/
theorem sufficient_questions_10n (c : Congress) :
  ∃ (strategy : (Fin c.n → Fin c.n → Origin) → Fin c.n → Origin),
    (∀ f : Fin c.n → Fin c.n → Origin, 
      (∀ i j : Fin c.n, f i j = ask_question c i j) → 
      (∀ i : Fin c.n, strategy f i = (c.scientists i).origin)) ∧
    (∃ s : Finset (Fin c.n × Fin c.n), s.card ≤ 10 * c.n ∧ 
      ∀ f : Fin c.n → Fin c.n → Origin, 
        (∀ i j : Fin c.n, f i j = ask_question c i j → (i, j) ∈ s)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_questions_sufficient_questions_10n_l2721_272135


namespace NUMINAMATH_CALUDE_train_speed_without_stoppages_l2721_272157

/-- The average speed of a train without stoppages, given its speed with stoppages and stoppage time. -/
theorem train_speed_without_stoppages
  (distance : ℝ) -- The distance traveled by the train
  (speed_with_stoppages : ℝ) -- The average speed of the train with stoppages
  (stoppage_time : ℝ) -- The time the train stops per hour
  (h1 : speed_with_stoppages = 360) -- The given speed with stoppages
  (h2 : stoppage_time = 6) -- The given stoppage time in minutes
  (h3 : distance > 0) -- Ensure the distance is positive
  : ∃ (speed_without_stoppages : ℝ),
    speed_without_stoppages = 400 ∧
    distance = speed_with_stoppages * 1 ∧
    distance = speed_without_stoppages * (1 - stoppage_time / 60) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_without_stoppages_l2721_272157


namespace NUMINAMATH_CALUDE_ilya_incorrect_l2721_272113

theorem ilya_incorrect : ¬∃ (s t : ℝ), s + t = s * t ∧ s + t = s / t := by
  sorry

end NUMINAMATH_CALUDE_ilya_incorrect_l2721_272113


namespace NUMINAMATH_CALUDE_constant_value_proof_l2721_272132

theorem constant_value_proof (f : ℝ → ℝ) (c : ℝ) 
  (h1 : ∀ x, f x + 3 * f (c - x) = x) 
  (h2 : f 2 = 2) : 
  c = 8 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_proof_l2721_272132


namespace NUMINAMATH_CALUDE_ellipse_angle_ratio_l2721_272190

noncomputable section

variables (a b : ℝ) (x y : ℝ)

-- Define the ellipse
def is_on_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define the angles α and β
def alpha (x y a : ℝ) : ℝ := Real.arctan (y / (x + a))
def beta (x y a : ℝ) : ℝ := Real.arctan (y / (x - a))

theorem ellipse_angle_ratio 
  (h1 : a > b) (h2 : b > 0)
  (h3 : is_on_ellipse x y a b)
  (h4 : eccentricity a b = Real.sqrt 3 / 2)
  (h5 : x ≠ a ∧ x ≠ -a) :
  (Real.cos (alpha x y a - beta x y a)) / 
  (Real.cos (alpha x y a + beta x y a)) = 3/5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_angle_ratio_l2721_272190


namespace NUMINAMATH_CALUDE_prove_a_minus_b_l2721_272166

-- Define the equation
def equation (a b c x : ℝ) : Prop :=
  (2*x - 3)^2 = a*x^2 + b*x + c

-- Theorem statement
theorem prove_a_minus_b (a b c : ℝ) 
  (h : ∀ x : ℝ, equation a b c x) : a - b = 16 := by
  sorry

end NUMINAMATH_CALUDE_prove_a_minus_b_l2721_272166


namespace NUMINAMATH_CALUDE_moon_speed_km_per_hour_l2721_272109

/-- The speed of the moon in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.05

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: The moon's speed in kilometers per hour -/
theorem moon_speed_km_per_hour :
  moon_speed_km_per_sec * seconds_per_hour = 3780 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_km_per_hour_l2721_272109


namespace NUMINAMATH_CALUDE_average_of_first_n_naturals_l2721_272177

theorem average_of_first_n_naturals (n : ℕ) : 
  (n * (n + 1)) / (2 * n) = 10 → n = 19 := by
  sorry

end NUMINAMATH_CALUDE_average_of_first_n_naturals_l2721_272177


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_primes_l2721_272125

/-- A function that returns true if a number is composite, false otherwise -/
def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that returns true if a number has no prime factors less than 20, false otherwise -/
def no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_primes : 
  (is_composite 529 ∧ no_small_prime_factors 529) ∧ 
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_primes_l2721_272125


namespace NUMINAMATH_CALUDE_average_speed_two_segments_l2721_272105

/-- Given a 100-mile trip where the first 50 miles are traveled at 20 mph
    and the remaining 50 miles at 50 mph, prove that the average speed
    for the entire trip is 100 / (50/20 + 50/50) miles per hour. -/
theorem average_speed_two_segments (total_distance : ℝ) (first_segment : ℝ) (second_segment : ℝ)
  (first_speed : ℝ) (second_speed : ℝ)
  (h1 : total_distance = 100)
  (h2 : first_segment = 50)
  (h3 : second_segment = 50)
  (h4 : first_speed = 20)
  (h5 : second_speed = 50)
  (h6 : total_distance = first_segment + second_segment) :
  (total_distance / (first_segment / first_speed + second_segment / second_speed)) =
  100 / (50 / 20 + 50 / 50) :=
by sorry

end NUMINAMATH_CALUDE_average_speed_two_segments_l2721_272105


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l2721_272186

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Plane → Plane → Prop)
variable (perpLine : Line → Plane → Prop)
variable (perpLines : Line → Line → Prop)

-- Define the property of being different planes
variable (different : Plane → Plane → Prop)

-- Define the property of being non-coincident lines
variable (nonCoincident : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (m n : Line)
  (h1 : different α β)
  (h2 : nonCoincident m n)
  (h3 : perpLine m α)
  (h4 : perpLine n β)
  (h5 : perp α β) :
  perpLines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l2721_272186


namespace NUMINAMATH_CALUDE_stating_min_additional_games_is_ten_l2721_272192

/-- Represents the number of games initially played -/
def initial_games : ℕ := 5

/-- Represents the number of games initially won by the Wolves -/
def initial_wolves_wins : ℕ := 2

/-- Represents the minimum winning percentage required for the Wolves -/
def min_winning_percentage : ℚ := 4/5

/-- 
Determines if a given number of additional games results in the Wolves
winning at least the minimum required percentage of all games
-/
def meets_winning_percentage (additional_games : ℕ) : Prop :=
  (initial_wolves_wins + additional_games : ℚ) / (initial_games + additional_games) ≥ min_winning_percentage

/-- 
Theorem stating that 10 is the minimum number of additional games
needed for the Wolves to meet the minimum winning percentage
-/
theorem min_additional_games_is_ten :
  (∀ n < 10, ¬(meets_winning_percentage n)) ∧ meets_winning_percentage 10 :=
sorry

end NUMINAMATH_CALUDE_stating_min_additional_games_is_ten_l2721_272192


namespace NUMINAMATH_CALUDE_last_digit_of_fraction_l2721_272180

/-- The last digit of the decimal expansion of 1 / (3^15 * 2^5) is 5 -/
theorem last_digit_of_fraction : ∃ (n : ℕ), (1 : ℚ) / (3^15 * 2^5) = n / 10 + 5 / 10^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_fraction_l2721_272180


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2721_272148

theorem arithmetic_calculations :
  ((-20) + (-14) - (-18) - 13 = -29) ∧
  ((-6) * (-2) / (1/8) = 96) ∧
  ((-24) * ((-3/4) - (5/6) + (7/8)) = 17) ∧
  (-(1^4) - (1 - 0.5) * (1/3) * ((-3)^2) = -5/2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2721_272148


namespace NUMINAMATH_CALUDE_trapezoid_area_l2721_272154

/-- Trapezoid ABCD with diagonals AC and BD, and midpoints P and S of AD and BC respectively -/
structure Trapezoid :=
  (A B C D P S : ℝ × ℝ)
  (is_trapezoid : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1))
  (diag_AC_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 8)
  (diag_BD_length : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 6)
  (P_midpoint : P = ((A.1 + D.1) / 2, (A.2 + D.2) / 2))
  (S_midpoint : S = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (PS_length : Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2) = 5)

/-- The area of the trapezoid ABCD is 24 -/
theorem trapezoid_area (t : Trapezoid) : 
  (1 / 2) * Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) * 
  Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2721_272154


namespace NUMINAMATH_CALUDE_remainder_101_pow_37_mod_100_l2721_272124

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_pow_37_mod_100_l2721_272124


namespace NUMINAMATH_CALUDE_odd_prime_square_root_l2721_272153

theorem odd_prime_square_root (p : ℕ) (k : ℕ) : 
  Prime p → 
  Odd p → 
  k > 0 → 
  ∃ n : ℕ, n > 0 ∧ n^2 = k^2 - p*k → 
  k = (p + 1)^2 / 4 := by
sorry

end NUMINAMATH_CALUDE_odd_prime_square_root_l2721_272153


namespace NUMINAMATH_CALUDE_parabola_midpoint_trajectory_l2721_272193

theorem parabola_midpoint_trajectory (x y : ℝ) : 
  let parabola := {(x, y) : ℝ × ℝ | x^2 = 4*y}
  let focus := (0, 1)
  ∀ (p : ℝ × ℝ), p ∈ parabola → 
    let midpoint := ((p.1 + focus.1)/2, (p.2 + focus.2)/2)
    midpoint.1^2 = 2*midpoint.2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_trajectory_l2721_272193


namespace NUMINAMATH_CALUDE_exponent_equality_l2721_272142

theorem exponent_equality (y x : ℕ) (h1 : 9^y = 3^x) (h2 : y = 7) : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l2721_272142


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2721_272183

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : a + b + c = 45) 
  (h2 : a^2 + b^2 + c^2 = 625) : 
  2 * (a * b + b * c + c * a) = 1400 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2721_272183


namespace NUMINAMATH_CALUDE_equation_solutions_count_l2721_272162

theorem equation_solutions_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 - 7)^2 = 49) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l2721_272162


namespace NUMINAMATH_CALUDE_island_marriage_proportion_l2721_272129

theorem island_marriage_proportion (men women : ℕ) (h1 : 2 * men = 3 * women) :
  (2 * men + 2 * women : ℚ) / (3 * men + 5 * women : ℚ) = 12 / 19 := by
  sorry

end NUMINAMATH_CALUDE_island_marriage_proportion_l2721_272129


namespace NUMINAMATH_CALUDE_farey_sequence_properties_l2721_272140

/-- Farey sequence of order n -/
def farey_sequence (n : ℕ) : List (ℚ) := sorry

/-- Sum of numerators in a Farey sequence -/
def sum_numerators (seq : List ℚ) : ℚ := sorry

/-- Sum of denominators in a Farey sequence -/
def sum_denominators (seq : List ℚ) : ℚ := sorry

/-- Sum of fractions in a Farey sequence -/
def sum_fractions (seq : List ℚ) : ℚ := sorry

theorem farey_sequence_properties (n : ℕ) :
  let seq := farey_sequence n
  (sum_denominators seq = 2 * sum_numerators seq) ∧
  (sum_fractions seq = (seq.length : ℚ) / 2) := by sorry

end NUMINAMATH_CALUDE_farey_sequence_properties_l2721_272140


namespace NUMINAMATH_CALUDE_inequality_problem_l2721_272195

theorem inequality_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y ≤ 4) :
  1 / (x * y) ≥ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l2721_272195


namespace NUMINAMATH_CALUDE_base_2_representation_of_123_l2721_272163

theorem base_2_representation_of_123 :
  ∃ (a b c d e f g : ℕ),
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1) ∧
    123 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_123_l2721_272163


namespace NUMINAMATH_CALUDE_horizontal_chord_iff_l2721_272108

/-- A cubic function with a horizontal chord -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The existence of a horizontal chord of length a for the function f -/
def has_horizontal_chord (a : ℝ) : Prop :=
  ∃ x : ℝ, f (a + x) = f x

/-- Theorem: The cubic function f has a horizontal chord of length a iff 0 < a ≤ 2 -/
theorem horizontal_chord_iff (a : ℝ) :
  has_horizontal_chord a ↔ (0 < a ∧ a ≤ 2) := by sorry

end NUMINAMATH_CALUDE_horizontal_chord_iff_l2721_272108


namespace NUMINAMATH_CALUDE_remainder_theorem_l2721_272161

theorem remainder_theorem (x : Int) (h : x % 285 = 31) :
  (x % 17 = 14) ∧ (x % 23 = 8) ∧ (x % 19 = 12) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2721_272161


namespace NUMINAMATH_CALUDE_manuscript_cost_theorem_l2721_272114

def manuscript_typing_cost (total_pages : ℕ) (initial_cost : ℕ) (revision_cost : ℕ) 
  (pages_revised_once : ℕ) (pages_revised_twice : ℕ) : ℕ :=
  (total_pages * initial_cost) + 
  (pages_revised_once * revision_cost) + 
  (pages_revised_twice * revision_cost * 2)

theorem manuscript_cost_theorem :
  manuscript_typing_cost 100 5 4 30 20 = 780 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_theorem_l2721_272114


namespace NUMINAMATH_CALUDE_fixed_point_on_all_parabolas_l2721_272179

/-- A parabola of the form y = 4x^2 + 2tx - 3t, where t is a real parameter -/
def parabola (t : ℝ) (x : ℝ) : ℝ := 4 * x^2 + 2 * t * x - 3 * t

/-- The fixed point through which all parabolas pass -/
def fixed_point : ℝ × ℝ := (1, 4)

/-- Theorem stating that the fixed point lies on all parabolas -/
theorem fixed_point_on_all_parabolas :
  ∀ t : ℝ, parabola t (fixed_point.1) = fixed_point.2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_all_parabolas_l2721_272179


namespace NUMINAMATH_CALUDE_polynomial_correction_l2721_272191

/-- If a polynomial P(x) satisfies P(x) - 3x² = x² - 2x + 1, 
    then -3x² * P(x) = -12x⁴ + 6x³ - 3x² -/
theorem polynomial_correction (x : ℝ) (P : ℝ → ℝ) 
  (h : P x - 3 * x^2 = x^2 - 2*x + 1) : 
  -3 * x^2 * P x = -12 * x^4 + 6 * x^3 - 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_correction_l2721_272191


namespace NUMINAMATH_CALUDE_luncheon_invitees_l2721_272128

/-- The number of people who didn't show up -/
def no_shows : ℕ := 7

/-- The number of people each table can hold -/
def people_per_table : ℕ := 5

/-- The number of tables needed -/
def tables_needed : ℕ := 8

/-- The original number of invited people -/
def original_invitees : ℕ := (tables_needed * people_per_table) + no_shows

theorem luncheon_invitees : original_invitees = 47 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_invitees_l2721_272128


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2721_272159

def complex_number (z : ℂ) : Prop :=
  z = (3 + Complex.I) / (1 - Complex.I)

theorem z_in_first_quadrant (z : ℂ) (h : complex_number z) :
  Complex.re z > 0 ∧ Complex.im z > 0 :=
sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2721_272159


namespace NUMINAMATH_CALUDE_only_three_digit_factorion_l2721_272107

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Sum of factorials of digits -/
def sumFactorialDigits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.map factorial).sum

/-- Predicate for a number being a factorion -/
def isFactorion (n : ℕ) : Prop :=
  n = sumFactorialDigits n

/-- Theorem: 145 is the only 3-digit factorion -/
theorem only_three_digit_factorion :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 → isFactorion n ↔ n = 145 :=
by sorry

end NUMINAMATH_CALUDE_only_three_digit_factorion_l2721_272107


namespace NUMINAMATH_CALUDE_f_neg_l2721_272194

-- Define an odd function f on the real numbers
def f : ℝ → ℝ := sorry

-- Define the property of f being odd
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define f for positive x
axiom f_pos : ∀ x : ℝ, x > 0 → f x = x^2 + 2*x - 3

-- Theorem to prove
theorem f_neg : ∀ x : ℝ, x < 0 → f x = -x^2 + 2*x + 3 := by sorry

end NUMINAMATH_CALUDE_f_neg_l2721_272194


namespace NUMINAMATH_CALUDE_derivative_at_one_equals_one_l2721_272141

theorem derivative_at_one_equals_one 
  (f : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (h : ∀ x, f x = x^3 - (deriv f 1) * x^2 + 1) : 
  deriv f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_equals_one_l2721_272141


namespace NUMINAMATH_CALUDE_theresa_video_games_l2721_272144

/-- The number of video games each person has -/
structure VideoGames where
  theresa : ℕ
  julia : ℕ
  tory : ℕ
  alex : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (vg : VideoGames) : Prop :=
  vg.theresa = 3 * vg.julia + 5 ∧
  vg.julia = vg.tory / 3 ∧
  vg.tory = 2 * vg.alex ∧
  vg.tory = 6

/-- The theorem to prove -/
theorem theresa_video_games (vg : VideoGames) (h : satisfies_conditions vg) : vg.theresa = 11 := by
  sorry

#check theresa_video_games

end NUMINAMATH_CALUDE_theresa_video_games_l2721_272144


namespace NUMINAMATH_CALUDE_range_of_Z_l2721_272102

theorem range_of_Z (a b : ℝ) (h : a^2 + 3*a*b + 9*b^2 = 4) :
  ∃ (z : ℝ), z = a^2 + 9*b^2 ∧ 8/3 ≤ z ∧ z ≤ 8 ∧
  (∀ (w : ℝ), w = a^2 + 9*b^2 → 8/3 ≤ w ∧ w ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_Z_l2721_272102


namespace NUMINAMATH_CALUDE_wolverine_workout_theorem_l2721_272151

/-- Calculates Wolverine's workout hours given Rayman's workout hours -/
def wolverine_workout_hours (rayman_hours : ℕ) : ℕ :=
  let junior_hours := 2 * rayman_hours
  let combined_hours := rayman_hours + junior_hours
  2 * combined_hours

theorem wolverine_workout_theorem (rayman_hours : ℕ) 
  (h : rayman_hours = 10) : wolverine_workout_hours rayman_hours = 60 := by
  sorry

#eval wolverine_workout_hours 10

end NUMINAMATH_CALUDE_wolverine_workout_theorem_l2721_272151


namespace NUMINAMATH_CALUDE_pentagon_reconstruction_l2721_272118

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A pentagon in 2D space -/
structure Pentagon where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D
  E : Point2D

/-- The reflected points of a pentagon -/
structure ReflectedPoints where
  A1 : Point2D
  B1 : Point2D
  C1 : Point2D
  D1 : Point2D
  E1 : Point2D

/-- Function to reflect a point with respect to another point -/
def reflect (p : Point2D) (center : Point2D) : Point2D :=
  { x := 2 * center.x - p.x
    y := 2 * center.y - p.y }

/-- Theorem stating that a pentagon can be reconstructed from its reflected points -/
theorem pentagon_reconstruction (reflectedPoints : ReflectedPoints) :
  ∃! (original : Pentagon),
    reflectedPoints.A1 = reflect original.A original.B ∧
    reflectedPoints.B1 = reflect original.B original.C ∧
    reflectedPoints.C1 = reflect original.C original.D ∧
    reflectedPoints.D1 = reflect original.D original.E ∧
    reflectedPoints.E1 = reflect original.E original.A :=
  sorry


end NUMINAMATH_CALUDE_pentagon_reconstruction_l2721_272118


namespace NUMINAMATH_CALUDE_good_number_iff_divisible_by_8_l2721_272123

def is_good_number (n : ℕ) : Prop :=
  ∃ k : ℤ, n = (2*k - 3) + (2*k - 1) + (2*k + 1) + (2*k + 3)

theorem good_number_iff_divisible_by_8 (n : ℕ) :
  is_good_number n ↔ n % 8 = 0 := by sorry

end NUMINAMATH_CALUDE_good_number_iff_divisible_by_8_l2721_272123


namespace NUMINAMATH_CALUDE_multiply_54_46_l2721_272110

theorem multiply_54_46 : 54 * 46 = 2484 := by
  sorry

end NUMINAMATH_CALUDE_multiply_54_46_l2721_272110


namespace NUMINAMATH_CALUDE_first_quadrant_iff_sin_cos_sum_gt_one_l2721_272155

theorem first_quadrant_iff_sin_cos_sum_gt_one (α : Real) :
  (0 < α ∧ α < Real.pi / 2) ↔ (Real.sin α + Real.cos α > 1) := by
  sorry

end NUMINAMATH_CALUDE_first_quadrant_iff_sin_cos_sum_gt_one_l2721_272155


namespace NUMINAMATH_CALUDE_lawrence_county_summer_break_l2721_272136

/-- The number of kids who stayed home during summer break in Lawrence county -/
theorem lawrence_county_summer_break (total_kids : ℕ) (camp_kids : ℕ) (h1 : total_kids = 1538832) (h2 : camp_kids = 893835) :
  total_kids - camp_kids = 644997 := by
  sorry

#check lawrence_county_summer_break

end NUMINAMATH_CALUDE_lawrence_county_summer_break_l2721_272136


namespace NUMINAMATH_CALUDE_largest_common_term_l2721_272167

def first_sequence (n : ℕ) : ℕ := 3 + 8 * n

def second_sequence (m : ℕ) : ℕ := 5 + 9 * m

theorem largest_common_term :
  ∃ (n m : ℕ),
    first_sequence n = second_sequence m ∧
    first_sequence n = 131 ∧
    first_sequence n ≤ 150 ∧
    ∀ (k l : ℕ), first_sequence k = second_sequence l → first_sequence k ≤ 150 → first_sequence k ≤ 131 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l2721_272167


namespace NUMINAMATH_CALUDE_triangle_area_5_5_6_l2721_272197

/-- The area of a triangle with sides 5, 5, and 6 units is 12 square units. -/
theorem triangle_area_5_5_6 : ∃ (A : ℝ), A = 12 ∧ A = Real.sqrt (8 * (8 - 5) * (8 - 5) * (8 - 6)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_5_5_6_l2721_272197


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_l2721_272152

theorem reciprocal_of_negative_one :
  ∃ x : ℝ, x * (-1) = 1 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_l2721_272152


namespace NUMINAMATH_CALUDE_pen_cost_l2721_272156

theorem pen_cost (pen ink : ℝ) 
  (total_cost : pen + ink = 2.50)
  (price_difference : pen = ink + 2) : 
  pen = 2.25 := by sorry

end NUMINAMATH_CALUDE_pen_cost_l2721_272156


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2721_272164

theorem quadratic_root_difference (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + 5*x₁ + k = 0 ∧ 
   x₂^2 + 5*x₂ + k = 0 ∧ 
   |x₁ - x₂| = 3) → 
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2721_272164


namespace NUMINAMATH_CALUDE_recurring_decimal_product_l2721_272173

theorem recurring_decimal_product : 
  ∃ (s : ℚ), (s = 456 / 999) ∧ (7 * s = 355 / 111) := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_product_l2721_272173
