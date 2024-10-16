import Mathlib

namespace NUMINAMATH_CALUDE_exists_same_color_configuration_l2558_255868

/-- A color type with two possible values -/
inductive Color
| Red
| Blue

/-- A grid of cells with colors -/
def Grid := Fin 5 → Fin 41 → Color

/-- A configuration of three rows and three columns -/
structure Configuration where
  rows : Fin 3 → Fin 5
  cols : Fin 3 → Fin 41

/-- Check if a configuration has all intersections of the same color -/
def Configuration.allSameColor (grid : Grid) (config : Configuration) : Prop :=
  ∃ c : Color, ∀ i j : Fin 3, grid (config.rows i) (config.cols j) = c

/-- Main theorem: There exists a configuration with all intersections of the same color -/
theorem exists_same_color_configuration (grid : Grid) :
  ∃ config : Configuration, config.allSameColor grid := by
  sorry


end NUMINAMATH_CALUDE_exists_same_color_configuration_l2558_255868


namespace NUMINAMATH_CALUDE_bicycle_wheel_radius_l2558_255858

theorem bicycle_wheel_radius (diameter : ℝ) (h : diameter = 26) : 
  diameter / 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_wheel_radius_l2558_255858


namespace NUMINAMATH_CALUDE_most_likely_red_balls_l2558_255885

theorem most_likely_red_balls
  (total_balls : ℕ)
  (red_probability : ℚ)
  (h_total : total_balls = 20)
  (h_prob : red_probability = 1/5) :
  (red_probability * total_balls : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_most_likely_red_balls_l2558_255885


namespace NUMINAMATH_CALUDE_oatmeal_cookies_count_l2558_255846

def cookies_per_bag : ℕ := 9
def chocolate_chip_cookies : ℕ := 13
def number_of_baggies : ℕ := 6

def total_cookies : ℕ := cookies_per_bag * number_of_baggies

theorem oatmeal_cookies_count :
  total_cookies - chocolate_chip_cookies = 41 :=
by sorry

end NUMINAMATH_CALUDE_oatmeal_cookies_count_l2558_255846


namespace NUMINAMATH_CALUDE_candy_cost_l2558_255808

theorem candy_cost (total_cents : ℕ) (num_gumdrops : ℕ) (h1 : total_cents = 224) (h2 : num_gumdrops = 28) :
  total_cents / num_gumdrops = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l2558_255808


namespace NUMINAMATH_CALUDE_zero_det_necessary_not_sufficient_for_parallel_l2558_255864

/-- Represents a line in the Cartesian plane of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if two lines are parallel -/
def are_parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- The determinant of the coefficients of two lines -/
def coeff_det (l₁ l₂ : Line) : ℝ :=
  l₁.a * l₂.b - l₂.a * l₁.b

/-- Theorem stating that zero determinant is necessary but not sufficient for parallel lines -/
theorem zero_det_necessary_not_sufficient_for_parallel (l₁ l₂ : Line) :
  (are_parallel l₁ l₂ → coeff_det l₁ l₂ = 0) ∧
  ¬(coeff_det l₁ l₂ = 0 → are_parallel l₁ l₂) :=
sorry

end NUMINAMATH_CALUDE_zero_det_necessary_not_sufficient_for_parallel_l2558_255864


namespace NUMINAMATH_CALUDE_m_minus_n_values_l2558_255828

theorem m_minus_n_values (m n : ℤ) 
  (h1 : |m| = 3)
  (h2 : |n| = 5)
  (h3 : m + n > 0) :
  m - n = -2 ∨ m - n = -8 := by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_values_l2558_255828


namespace NUMINAMATH_CALUDE_fraction_sum_evaluation_l2558_255850

theorem fraction_sum_evaluation (p q r : ℝ) 
  (h : p / (30 - p) + q / (75 - q) + r / (45 - r) = 9) :
  6 / (30 - p) + 15 / (75 - q) + 9 / (45 - r) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_evaluation_l2558_255850


namespace NUMINAMATH_CALUDE_puppy_food_bags_l2558_255865

/-- Calculates the number of bags of special dog food needed for a puppy's first year -/
def bags_needed : ℕ :=
  let days_in_year : ℕ := 365
  let ounces_per_pound : ℕ := 16
  let bag_weight : ℕ := 5
  let first_period : ℕ := 60
  let first_period_daily_food : ℕ := 2
  let second_period_daily_food : ℕ := 4
  let first_period_total : ℕ := first_period * first_period_daily_food
  let second_period : ℕ := days_in_year - first_period
  let second_period_total : ℕ := second_period * second_period_daily_food
  let total_ounces : ℕ := first_period_total + second_period_total
  let total_pounds : ℕ := (total_ounces + ounces_per_pound - 1) / ounces_per_pound
  (total_pounds + bag_weight - 1) / bag_weight

theorem puppy_food_bags : bags_needed = 17 := by
  sorry

end NUMINAMATH_CALUDE_puppy_food_bags_l2558_255865


namespace NUMINAMATH_CALUDE_product_properties_l2558_255890

theorem product_properties (x y z : ℕ) : 
  x = 15 ∧ y = 5 ∧ z = 8 →
  (x * y * z = 600) ∧
  ((x - 10) * y * z = 200) ∧
  ((x + 5) * y * z = 1200) := by
sorry

end NUMINAMATH_CALUDE_product_properties_l2558_255890


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2558_255836

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.cos (9 * x) - Real.cos (5 * x) - Real.sqrt 2 * Real.cos (4 * x) + Real.sin (9 * x) + Real.sin (5 * x) = 0) →
  (∃ k : ℤ, x = π / 8 + π * k / 2 ∨ x = π / 20 + 2 * π * k / 5 ∨ x = π / 12 + 2 * π * k / 9) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2558_255836


namespace NUMINAMATH_CALUDE_inequality_and_function_minimum_l2558_255845

-- Define the set A
def A (a : ℕ+) : Set ℝ := {x : ℝ | |x - 2| < a}

-- State the theorem
theorem inequality_and_function_minimum (a : ℕ+) 
  (h1 : (3/2 : ℝ) ∈ A a) 
  (h2 : (1/2 : ℝ) ∉ A a) :
  (a = 1) ∧ 
  (∀ x : ℝ, |x + a| + |x - 2| ≥ 3) ∧ 
  (∃ x : ℝ, |x + a| + |x - 2| = 3) := by
sorry

end NUMINAMATH_CALUDE_inequality_and_function_minimum_l2558_255845


namespace NUMINAMATH_CALUDE_candy_mix_equations_correct_l2558_255873

/-- Represents the candy mixing problem -/
structure CandyMix where
  x : ℝ  -- quantity of 36 yuan/kg candy
  y : ℝ  -- quantity of 20 yuan/kg candy
  total_weight : ℝ  -- total weight of mixed candy
  mixed_price : ℝ  -- price of mixed candy per kg
  high_price : ℝ  -- price of more expensive candy per kg
  low_price : ℝ  -- price of less expensive candy per kg

/-- The system of equations correctly describes the candy mixing problem -/
theorem candy_mix_equations_correct (mix : CandyMix) 
  (h1 : mix.total_weight = 100)
  (h2 : mix.mixed_price = 28)
  (h3 : mix.high_price = 36)
  (h4 : mix.low_price = 20) :
  (mix.x + mix.y = mix.total_weight) ∧ 
  (mix.high_price * mix.x + mix.low_price * mix.y = mix.mixed_price * mix.total_weight) :=
sorry

end NUMINAMATH_CALUDE_candy_mix_equations_correct_l2558_255873


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l2558_255884

def shelves_used (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (initial_stock - books_sold) / books_per_shelf

theorem coloring_book_shelves :
  shelves_used 87 33 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l2558_255884


namespace NUMINAMATH_CALUDE_cone_slant_height_l2558_255888

/-- Given a cone with base radius 3 cm and curved surface area 141.3716694115407 cm²,
    prove that its slant height is 15 cm. -/
theorem cone_slant_height (r : ℝ) (csa : ℝ) (h1 : r = 3) (h2 : csa = 141.3716694115407) :
  csa / (Real.pi * r) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l2558_255888


namespace NUMINAMATH_CALUDE_gcd_360_128_l2558_255877

theorem gcd_360_128 : Nat.gcd 360 128 = 8 := by sorry

end NUMINAMATH_CALUDE_gcd_360_128_l2558_255877


namespace NUMINAMATH_CALUDE_rectangular_field_diagonal_l2558_255817

/-- Given a rectangular field with width 4 m and area 12 m², 
    prove that its diagonal is 5 m. -/
theorem rectangular_field_diagonal : 
  ∀ (w l d : ℝ), 
    w = 4 → 
    w * l = 12 → 
    d ^ 2 = w ^ 2 + l ^ 2 → 
    d = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_diagonal_l2558_255817


namespace NUMINAMATH_CALUDE_usual_time_to_catch_bus_l2558_255842

/-- Proves that given a person walking at 3/5 of their usual speed and missing the bus by 5 minutes, their usual time to catch the bus is 7.5 minutes. -/
theorem usual_time_to_catch_bus : ∀ (usual_speed : ℝ) (usual_time : ℝ),
  usual_time > 0 →
  usual_speed > 0 →
  (3/5 * usual_speed) * (usual_time + 5) = usual_speed * usual_time →
  usual_time = 7.5 := by
sorry

end NUMINAMATH_CALUDE_usual_time_to_catch_bus_l2558_255842


namespace NUMINAMATH_CALUDE_positive_reals_inequality_l2558_255830

theorem positive_reals_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y^3 + y ≤ x - x^3) : y < x ∧ x < 1 ∧ x^2 + y^2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequality_l2558_255830


namespace NUMINAMATH_CALUDE_y_equation_solution_l2558_255840

theorem y_equation_solution (y : ℝ) (c d : ℕ+) 
  (h1 : y^2 + 4*y + 4/y + 1/y^2 = 30)
  (h2 : y = c + Real.sqrt d) :
  c + d = 5 := by
  sorry

end NUMINAMATH_CALUDE_y_equation_solution_l2558_255840


namespace NUMINAMATH_CALUDE_solution_set_l2558_255894

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- Define the condition that f' is the derivative of f
def is_derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, deriv f x = f' x

-- Define the condition that 2f'(x) > f(x) for all x
def condition (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, 2 * f' x > f x

-- Define the inequality we want to solve
def inequality (f : ℝ → ℝ) (x : ℝ) : Prop :=
  Real.exp ((x - 1) / 2) * f x < f (2 * x - 1)

-- State the theorem
theorem solution_set (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  is_derivative f f' → condition f f' →
  (∀ x, inequality f x ↔ x > 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_l2558_255894


namespace NUMINAMATH_CALUDE_function_characterization_l2558_255878

theorem function_characterization (f : ℚ → ℚ) 
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_characterization_l2558_255878


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l2558_255829

/-- The orthocenter of a triangle is the point where all three altitudes of the triangle intersect. -/
def Orthocenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Given three points A, B, and C in the plane, this theorem states that the orthocenter
    of the triangle formed by these points has specific coordinates. -/
theorem orthocenter_of_triangle (A B C : ℝ × ℝ) 
  (hA : A = (5, -1)) (hB : B = (4, -8)) (hC : C = (-4, -4)) :
  Orthocenter A B C = (3, -5) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l2558_255829


namespace NUMINAMATH_CALUDE_exactly_one_solves_l2558_255819

theorem exactly_one_solves (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2)
  (h_B : p_B = 1/3)
  (h_C : p_C = 1/4)
  (h_independent : True) -- Representing the independence condition
  : p_A * (1 - p_B) * (1 - p_C) + 
    (1 - p_A) * p_B * (1 - p_C) + 
    (1 - p_A) * (1 - p_B) * p_C = 11/24 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_solves_l2558_255819


namespace NUMINAMATH_CALUDE_black_duck_count_l2558_255860

/-- Represents the number of fish per duck of each color --/
structure FishPerDuck where
  white : ℕ
  black : ℕ
  multicolor : ℕ

/-- Represents the number of ducks of each color --/
structure DuckCounts where
  white : ℕ
  black : ℕ
  multicolor : ℕ

/-- The theorem stating the number of black ducks --/
theorem black_duck_count 
  (fish_per_duck : FishPerDuck)
  (duck_counts : DuckCounts)
  (total_fish : ℕ)
  (h1 : fish_per_duck.white = 5)
  (h2 : fish_per_duck.black = 10)
  (h3 : fish_per_duck.multicolor = 12)
  (h4 : duck_counts.white = 3)
  (h5 : duck_counts.multicolor = 6)
  (h6 : total_fish = 157)
  (h7 : total_fish = 
    fish_per_duck.white * duck_counts.white + 
    fish_per_duck.black * duck_counts.black + 
    fish_per_duck.multicolor * duck_counts.multicolor) :
  duck_counts.black = 7 := by
  sorry

end NUMINAMATH_CALUDE_black_duck_count_l2558_255860


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l2558_255818

theorem intersection_point_k_value :
  ∀ (k : ℝ),
  (∃ (y : ℝ), -3 * (-6) + 2 * y = k ∧ 0.75 * (-6) + y = 16) →
  k = 59 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l2558_255818


namespace NUMINAMATH_CALUDE_encircling_stripe_probability_theorem_l2558_255889

/-- Represents a cube with 6 faces -/
structure Cube :=
  (faces : Fin 6 → Bool)

/-- The probability of a stripe on a single face -/
def stripe_prob : ℚ := 2/3

/-- The probability of a dot on a single face -/
def dot_prob : ℚ := 1/3

/-- The number of valid stripe configurations that encircle the cube -/
def valid_configurations : ℕ := 12

/-- The probability of a continuous stripe encircling the cube -/
def encircling_stripe_probability : ℚ := 768/59049

/-- Theorem stating the probability of a continuous stripe encircling the cube -/
theorem encircling_stripe_probability_theorem :
  encircling_stripe_probability = 
    (stripe_prob ^ 6) * valid_configurations :=
by sorry

end NUMINAMATH_CALUDE_encircling_stripe_probability_theorem_l2558_255889


namespace NUMINAMATH_CALUDE_strawberry_theft_l2558_255861

/-- Calculates the number of stolen strawberries given the daily harvest rate, 
    number of days, strawberries given away, and final count. -/
def stolen_strawberries (daily_harvest : ℕ) (days : ℕ) (given_away : ℕ) (final_count : ℕ) : ℕ :=
  daily_harvest * days - given_away - final_count

/-- Proves that the number of stolen strawberries is 30 given the specific conditions. -/
theorem strawberry_theft : 
  stolen_strawberries 5 30 20 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_theft_l2558_255861


namespace NUMINAMATH_CALUDE_quadratic_equation_one_solution_l2558_255866

theorem quadratic_equation_one_solution (k : ℝ) : 
  (∃! x : ℝ, (k + 2) * x^2 + 2 * k * x + 1 = 0) ↔ (k = -2 ∨ k = -1 ∨ k = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_solution_l2558_255866


namespace NUMINAMATH_CALUDE_power_mod_eleven_l2558_255893

theorem power_mod_eleven : 6^305 % 11 = 10 := by sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l2558_255893


namespace NUMINAMATH_CALUDE_all_lines_pass_through_common_point_l2558_255886

/-- A line in 2D space represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Checks if three numbers form a geometric progression -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

theorem all_lines_pass_through_common_point :
  ∀ l : Line, isGeometricProgression l.a l.b l.c →
  l.contains (-1) 1 := by sorry

end NUMINAMATH_CALUDE_all_lines_pass_through_common_point_l2558_255886


namespace NUMINAMATH_CALUDE_range_of_m_l2558_255824

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the necessary condition
def necessary_condition (m : ℝ) (x : ℝ) : Prop :=
  x < m - 1 ∨ x > m + 1

theorem range_of_m :
  ∀ m : ℝ,
    (∀ x : ℝ, f x > 0 → necessary_condition m x) ∧
    (∃ x : ℝ, necessary_condition m x ∧ f x ≤ 0) →
    0 ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2558_255824


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2558_255848

/-- The complex number z defined as (2-i)^2 -/
def z : ℂ := (2 - Complex.I) ^ 2

/-- Theorem stating that z lies in the fourth quadrant of the complex plane -/
theorem z_in_fourth_quadrant : 
  z.re > 0 ∧ z.im < 0 := by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2558_255848


namespace NUMINAMATH_CALUDE_segment_length_implies_product_l2558_255857

/-- Given that the length of the segment between the points (3a, 2a-5) and (5, 0) is 3√10 units,
    prove that the product of all possible values of a is -40/13. -/
theorem segment_length_implies_product (a : ℝ) : 
  (((3*a - 5)^2 + (2*a - 5)^2) = 90) → 
  (∃ b : ℝ, (a = b ∨ a = -8/13) ∧ a * b = -40/13) :=
by sorry

end NUMINAMATH_CALUDE_segment_length_implies_product_l2558_255857


namespace NUMINAMATH_CALUDE_ninth_grade_class_problem_l2558_255897

theorem ninth_grade_class_problem (total : ℕ) (math : ℕ) (foreign : ℕ) (science_only : ℕ) (math_and_foreign : ℕ) :
  total = 120 →
  math = 85 →
  foreign = 75 →
  science_only = 20 →
  math_and_foreign = 40 →
  ∃ (math_only : ℕ), math_only = 45 ∧ math_only = math - math_and_foreign :=
by sorry

end NUMINAMATH_CALUDE_ninth_grade_class_problem_l2558_255897


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l2558_255831

def boat_speed : ℝ := 18
def stream_speed : ℝ := 6

def upstream_speed : ℝ := boat_speed - stream_speed
def downstream_speed : ℝ := boat_speed + stream_speed

theorem upstream_downstream_time_ratio :
  upstream_speed / downstream_speed = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l2558_255831


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2558_255826

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An increasing sequence -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_properties (a : ℕ → ℝ) (h : geometric_sequence a) :
  (a 1 < a 2 ∧ a 2 < a 3 → increasing_sequence a) ∧
  (increasing_sequence a → a 1 < a 2 ∧ a 2 < a 3) ∧
  (a 1 ≥ a 2 ∧ a 2 ≥ a 3 → ¬increasing_sequence a) ∧
  (¬increasing_sequence a → a 1 ≥ a 2 ∧ a 2 ≥ a 3) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2558_255826


namespace NUMINAMATH_CALUDE_proportional_relationship_l2558_255839

-- Define the proportionality constant
def k : ℝ := 2

-- Define the functional relationship
def f (x : ℝ) : ℝ := k * x + 3

-- State the theorem
theorem proportional_relationship (x y : ℝ) :
  (∀ x, y - 3 = k * x) →  -- (y-3) is directly proportional to x
  (f 2 = 7) →             -- when x=2, y=7
  (∀ x, f x = 2 * x + 3) ∧ -- functional relationship
  (f 4 = 11) ∧            -- when x=4, y=11
  (f⁻¹ 4 = 1/2)           -- when y=4, x=1/2
  := by sorry

end NUMINAMATH_CALUDE_proportional_relationship_l2558_255839


namespace NUMINAMATH_CALUDE_factorial_less_than_power_l2558_255883

theorem factorial_less_than_power : 
  Nat.factorial 999 < 500^999 := by
  sorry

end NUMINAMATH_CALUDE_factorial_less_than_power_l2558_255883


namespace NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l2558_255898

theorem cylinder_height_in_hemisphere (r c h : ℝ) : 
  r > 0 → c > 0 → h > 0 →
  r = 7 → c = 3 →
  h^2 + c^2 = r^2 →
  h = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l2558_255898


namespace NUMINAMATH_CALUDE_part_a_part_b_part_c_l2558_255855

-- Define what it means for a number to be TOP
def is_TOP (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  (n / 10000) * (n % 10) = ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10)

-- Part a
theorem part_a : is_TOP 23498 := by sorry

-- Part b
theorem part_b : ∃ (s : Finset ℕ), 
  (∀ n ∈ s, is_TOP n ∧ n / 10000 = 1 ∧ n % 10 = 2) ∧ 
  (∀ n, is_TOP n ∧ n / 10000 = 1 ∧ n % 10 = 2 → n ∈ s) ∧
  Finset.card s = 6 := by sorry

-- Part c
theorem part_c : ∃ (s : Finset ℕ),
  (∀ n ∈ s, is_TOP n ∧ n / 10000 = 9) ∧
  (∀ n, is_TOP n ∧ n / 10000 = 9 → n ∈ s) ∧
  Finset.card s = 112 := by sorry

end NUMINAMATH_CALUDE_part_a_part_b_part_c_l2558_255855


namespace NUMINAMATH_CALUDE_fraction_addition_l2558_255870

theorem fraction_addition : (3 : ℚ) / 8 + (9 : ℚ) / 12 = (9 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2558_255870


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l2558_255862

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 3 / 10) :
  a / c = 25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l2558_255862


namespace NUMINAMATH_CALUDE_number_in_scientific_notation_l2558_255856

/-- Definition of scientific notation -/
def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * (10 : ℝ) ^ b ∧ 1 ≤ a ∧ a < 10

/-- The number to be expressed in scientific notation -/
def number : ℝ := 123000

/-- Theorem stating that 123000 can be expressed as 1.23 × 10^5 in scientific notation -/
theorem number_in_scientific_notation :
  scientific_notation number 1.23 5 :=
sorry

end NUMINAMATH_CALUDE_number_in_scientific_notation_l2558_255856


namespace NUMINAMATH_CALUDE_second_duck_bread_pieces_l2558_255841

theorem second_duck_bread_pieces : 
  ∀ (total_bread pieces_left first_duck_fraction last_duck_pieces : ℕ),
  total_bread = 100 →
  pieces_left = 30 →
  first_duck_fraction = 2 →  -- Represents 1/2
  last_duck_pieces = 7 →
  ∃ (second_duck_pieces : ℕ),
    second_duck_pieces = total_bread - pieces_left - (total_bread / first_duck_fraction) - last_duck_pieces ∧
    second_duck_pieces = 13 := by
  sorry

end NUMINAMATH_CALUDE_second_duck_bread_pieces_l2558_255841


namespace NUMINAMATH_CALUDE_wife_selection_probability_l2558_255881

theorem wife_selection_probability 
  (p_husband : ℝ) 
  (p_only_one : ℝ) 
  (h1 : p_husband = 1/7)
  (h2 : p_only_one = 0.28571428571428575) : 
  ∃ p_wife : ℝ, p_wife = 1/5 ∧ 
  p_only_one = p_husband * (1 - p_wife) + p_wife * (1 - p_husband) :=
sorry

end NUMINAMATH_CALUDE_wife_selection_probability_l2558_255881


namespace NUMINAMATH_CALUDE_jump_distance_difference_l2558_255854

theorem jump_distance_difference (grasshopper_jump frog_jump : ℕ) 
  (h1 : grasshopper_jump = 13)
  (h2 : frog_jump = 11) :
  grasshopper_jump - frog_jump = 2 := by
  sorry

end NUMINAMATH_CALUDE_jump_distance_difference_l2558_255854


namespace NUMINAMATH_CALUDE_coefficient_of_minus_five_ab_l2558_255810

/-- The coefficient of a monomial is the numerical factor multiplying the variables. -/
def coefficient (m : ℤ) (x : String) : ℤ :=
  m

/-- A monomial is represented as an integer multiplied by a string of variables. -/
def Monomial := ℤ × String

theorem coefficient_of_minus_five_ab :
  let m : Monomial := (-5, "ab")
  coefficient m.1 m.2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_minus_five_ab_l2558_255810


namespace NUMINAMATH_CALUDE_solve_for_s_l2558_255800

theorem solve_for_s (s t : ℚ) 
  (eq1 : 8 * s + 7 * t = 160) 
  (eq2 : s = t - 3) : 
  s = 139 / 15 := by
sorry

end NUMINAMATH_CALUDE_solve_for_s_l2558_255800


namespace NUMINAMATH_CALUDE_min_magnitude_vector_sum_l2558_255809

/-- The minimum magnitude of the vector sum of two specific unit vectors -/
theorem min_magnitude_vector_sum :
  let a : ℝ × ℝ := (Real.cos (25 * π / 180), Real.sin (25 * π / 180))
  let b : ℝ × ℝ := (Real.sin (20 * π / 180), Real.cos (20 * π / 180))
  ∃ (min_val : ℝ), min_val = Real.sqrt 2 / 2 ∧
    ∀ (t : ℝ), Real.sqrt ((a.1 + t * b.1)^2 + (a.2 + t * b.2)^2) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_magnitude_vector_sum_l2558_255809


namespace NUMINAMATH_CALUDE_perfect_square_digits_l2558_255882

theorem perfect_square_digits (a b x y : ℕ) : 
  (∃ n : ℕ, a = n^2) →  -- a is a perfect square
  (∃ m : ℕ, b = m^2) →  -- b is a perfect square
  a % 10 = 1 →         -- unit digit of a is 1
  (a / 10) % 10 = x →  -- tens digit of a is x
  b % 10 = 6 →         -- unit digit of b is 6
  (b / 10) % 10 = y →  -- tens digit of b is y
  Even x ∧ Odd y :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_digits_l2558_255882


namespace NUMINAMATH_CALUDE_book_selection_combinations_l2558_255813

theorem book_selection_combinations :
  let mystery_count : ℕ := 5
  let fantasy_count : ℕ := 4
  let biography_count : ℕ := 6
  mystery_count * fantasy_count * biography_count = 120 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_combinations_l2558_255813


namespace NUMINAMATH_CALUDE_large_power_of_two_appears_early_l2558_255811

/-- Represents the state of cards on the table at any given time -/
structure CardState where
  totalCards : Nat
  oddCards : Nat
  maxPowerOfTwo : Nat

/-- The initial state of cards -/
def initialState : CardState :=
  { totalCards := 100, oddCards := 43, maxPowerOfTwo := 0 }

/-- Function to calculate the next state after one minute -/
def nextState (state : CardState) : CardState :=
  { totalCards := state.totalCards + 1,
    oddCards := if state.oddCards = 43 then 44 else 44,
    maxPowerOfTwo := state.maxPowerOfTwo + 1 }

/-- Function to calculate the state after n minutes -/
def stateAfterMinutes (n : Nat) : CardState :=
  match n with
  | 0 => initialState
  | n + 1 => nextState (stateAfterMinutes n)

theorem large_power_of_two_appears_early (n : Nat) :
  (stateAfterMinutes n).maxPowerOfTwo ≥ 10000 →
  (stateAfterMinutes 1440).maxPowerOfTwo ≥ 10000 :=
by
  sorry

#check large_power_of_two_appears_early

end NUMINAMATH_CALUDE_large_power_of_two_appears_early_l2558_255811


namespace NUMINAMATH_CALUDE_four_math_six_english_arrangements_l2558_255895

/-- The number of ways to arrange books and a trophy on a shelf -/
def shelfArrangements (mathBooks : ℕ) (englishBooks : ℕ) : ℕ :=
  2 * 2 * (Nat.factorial mathBooks) * (Nat.factorial englishBooks)

/-- Theorem stating the number of arrangements for 4 math books and 6 English books -/
theorem four_math_six_english_arrangements :
  shelfArrangements 4 6 = 69120 := by
  sorry

#eval shelfArrangements 4 6

end NUMINAMATH_CALUDE_four_math_six_english_arrangements_l2558_255895


namespace NUMINAMATH_CALUDE_expression_evaluation_l2558_255875

theorem expression_evaluation (a b : ℤ) (ha : a = -1) (hb : b = 4) :
  (a + b)^2 - 2*a*(a - b) + (a + 2*b)*(a - 2*b) = -64 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2558_255875


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2558_255835

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (L m : Line) (α : Plane) 
  (h1 : parallel m L) 
  (h2 : perpendicular m α) : 
  perpendicular L α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2558_255835


namespace NUMINAMATH_CALUDE_arrangement_exists_l2558_255871

/-- A type representing a 10x10 table of real numbers -/
def Table := Fin 10 → Fin 10 → ℝ

/-- A predicate to check if two cells are adjacent in the table -/
def adjacent (i j k l : Fin 10) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨ 
  (i = k ∧ l.val + 1 = j.val) ∨ 
  (j = l ∧ i.val + 1 = k.val) ∨ 
  (j = l ∧ k.val + 1 = i.val)

/-- The main theorem statement -/
theorem arrangement_exists (S : Finset ℝ) (h : S.card = 100) :
  ∃ (f : Table), 
    (∀ x ∈ S, ∃ i j, f i j = x) ∧ 
    (∀ i j k l, adjacent i j k l → |f i j - f k l| ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_exists_l2558_255871


namespace NUMINAMATH_CALUDE_equation_solution_l2558_255896

theorem equation_solution (x : ℝ) : 3*x - 5 = 10*x + 9 → 4*(x + 7) = 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2558_255896


namespace NUMINAMATH_CALUDE_ranch_problem_l2558_255869

theorem ranch_problem :
  ∃! (s c : ℕ), s > 0 ∧ c > 0 ∧ 25 * s + 26 * c = 1000 ∧ c > s := by
  sorry

end NUMINAMATH_CALUDE_ranch_problem_l2558_255869


namespace NUMINAMATH_CALUDE_vector_operation_l2558_255874

theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (-3, 4)) :
  2 • a - b = (7, -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l2558_255874


namespace NUMINAMATH_CALUDE_solve_bowtie_equation_l2558_255833

-- Define the operation ⊛
noncomputable def bowtie (a b : ℝ) : ℝ := a + 3 * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem solve_bowtie_equation (g : ℝ) : bowtie 5 g = 14 → g = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_bowtie_equation_l2558_255833


namespace NUMINAMATH_CALUDE_vector_coordinates_proof_l2558_255832

theorem vector_coordinates_proof (a : ℝ × ℝ) (b : ℝ × ℝ) :
  let x := a.1
  let y := a.2
  b = (1, 2) →
  Real.sqrt (x^2 + y^2) = 3 →
  x * b.1 + y * b.2 = 0 →
  (x = -6 * Real.sqrt 5 / 5 ∧ y = 3 * Real.sqrt 5 / 5) ∨
  (x = 6 * Real.sqrt 5 / 5 ∧ y = -3 * Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_vector_coordinates_proof_l2558_255832


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l2558_255803

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l2558_255803


namespace NUMINAMATH_CALUDE_prime_square_in_A_implies_prime_in_A_l2558_255851

def A : Set ℕ := {n : ℕ | ∃ (a b : ℤ), b ≠ 0 ∧ n = a^2 + 2*b^2}

theorem prime_square_in_A_implies_prime_in_A (p : ℕ) (hp : Nat.Prime p) (hp2 : p^2 ∈ A) : p ∈ A := by
  sorry

end NUMINAMATH_CALUDE_prime_square_in_A_implies_prime_in_A_l2558_255851


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l2558_255843

theorem min_value_of_fraction (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  (∀ x y : ℝ, x > y ∧ x * y = 1 → (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2) ∧
  ∃ x y : ℝ, x > y ∧ x * y = 1 ∧ (x^2 + y^2) / (x - y) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l2558_255843


namespace NUMINAMATH_CALUDE_max_displayed_games_l2558_255806

/-- Represents the number of games that can be displayed for each genre -/
structure DisplayedGames where
  action : ℕ
  adventure : ℕ
  simulation : ℕ

/-- Represents the shelf capacity for each genre -/
structure ShelfCapacity where
  action : ℕ
  adventure : ℕ
  simulation : ℕ

/-- Represents the total number of games for each genre -/
structure TotalGames where
  action : ℕ
  adventure : ℕ
  simulation : ℕ

def store_promotion : ℕ := 10

def total_games : TotalGames :=
  { action := 73, adventure := 51, simulation := 39 }

def shelf_capacity : ShelfCapacity :=
  { action := 60, adventure := 45, simulation := 35 }

def displayed_games (t : TotalGames) (s : ShelfCapacity) : DisplayedGames :=
  { action := min (t.action - store_promotion) s.action + store_promotion,
    adventure := min (t.adventure - store_promotion) s.adventure + store_promotion,
    simulation := min (t.simulation - store_promotion) s.simulation + store_promotion }

def total_displayed (d : DisplayedGames) : ℕ :=
  d.action + d.adventure + d.simulation

theorem max_displayed_games :
  total_displayed (displayed_games total_games shelf_capacity) = 160 :=
by sorry

end NUMINAMATH_CALUDE_max_displayed_games_l2558_255806


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l2558_255812

theorem smallest_constant_inequality (x y : ℝ) :
  (∀ D : ℝ, (∀ x y : ℝ, x^4 + y^4 + 1 ≥ D * (x^2 + y^2)) → D ≤ Real.sqrt 2) ∧
  (∀ x y : ℝ, x^4 + y^4 + 1 ≥ Real.sqrt 2 * (x^2 + y^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l2558_255812


namespace NUMINAMATH_CALUDE_clothing_store_problem_l2558_255816

/-- The clothing store problem -/
theorem clothing_store_problem 
  (cost : ℝ) 
  (initial_price : ℝ) 
  (initial_volume : ℝ) 
  (price_increase : ℝ) 
  (volume_decrease : ℝ)
  (h1 : cost = 50)
  (h2 : initial_price = 60)
  (h3 : initial_volume = 800)
  (h4 : price_increase = 5)
  (h5 : volume_decrease = 100) :
  let sales_volume (x : ℝ) := initial_volume - (volume_decrease / price_increase) * (x - initial_price)
  let profit (x : ℝ) := (x - cost) * sales_volume x
  ∃ (max_price : ℝ) (max_profit : ℝ),
    -- 1. Sales volume at 70 yuan
    sales_volume 70 = 600 ∧
    -- 2. Profit at 70 yuan
    profit 70 = 12000 ∧
    -- 3. Profit function
    (∀ x, profit x = -20 * x^2 + 3000 * x - 100000) ∧
    -- 4. Maximum profit
    (∀ x, profit x ≤ max_profit) ∧ max_price = 75 ∧ max_profit = 12500 ∧
    -- 5. Selling prices for 12000 yuan profit
    profit 70 = 12000 ∧ profit 80 = 12000 ∧
    (∀ x, profit x = 12000 → (x = 70 ∨ x = 80)) := by
  sorry

end NUMINAMATH_CALUDE_clothing_store_problem_l2558_255816


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2558_255823

-- Define the equation
def equation (x : ℝ) : Prop := (x - 2) * (x + 5) = 0

-- Define sufficient condition
def sufficient (p q : Prop) : Prop := p → q

-- Define necessary condition
def necessary (p q : Prop) : Prop := q → p

-- Theorem statement
theorem x_eq_2_sufficient_not_necessary :
  (sufficient (x = 2) (equation x)) ∧ ¬(necessary (x = 2) (equation x)) :=
sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2558_255823


namespace NUMINAMATH_CALUDE_roberto_outfits_l2558_255876

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets : ℕ) : ℕ :=
  trousers * shirts * jackets

/-- Theorem stating the number of outfits Roberto can create -/
theorem roberto_outfits :
  let trousers : ℕ := 5
  let shirts : ℕ := 8
  let jackets : ℕ := 4
  number_of_outfits trousers shirts jackets = 160 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l2558_255876


namespace NUMINAMATH_CALUDE_triangle_similarity_properties_l2558_255891

/-- Triangle properties for medians and altitudes similarity --/
theorem triangle_similarity_properties (a b c : ℝ) (h_order : a ≤ b ∧ b ≤ c) :
  (∀ (ma mb mc : ℝ), 
    (ma = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2)) →
    (mb = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2)) →
    (mc = (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)) →
    (a / mc = b / mb ∧ b / mb = c / ma) →
    (2 * b^2 = a^2 + c^2)) ∧
  (∀ (ha hb hc : ℝ),
    (ha * a = hb * b ∧ hb * b = hc * c) →
    (ha / hb = b / a ∧ ha / hc = c / a ∧ hb / hc = c / b)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_similarity_properties_l2558_255891


namespace NUMINAMATH_CALUDE_sara_initial_peaches_l2558_255807

/-- The number of peaches Sara picked at the orchard -/
def peaches_picked : ℕ := 37

/-- The total number of peaches Sara has now -/
def total_peaches_now : ℕ := 61

/-- The initial number of peaches Sara had -/
def initial_peaches : ℕ := total_peaches_now - peaches_picked

theorem sara_initial_peaches : initial_peaches = 24 := by
  sorry

end NUMINAMATH_CALUDE_sara_initial_peaches_l2558_255807


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l2558_255880

/-- The number of books in the 'Crazy Silly School' series -/
def total_books : ℕ := 13

/-- The number of books already read -/
def books_read : ℕ := 9

/-- The number of books left to read -/
def books_left : ℕ := 4

/-- Theorem stating that the total number of books is equal to the sum of books read and books left -/
theorem crazy_silly_school_books : 
  total_books = books_read + books_left := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l2558_255880


namespace NUMINAMATH_CALUDE_nate_age_when_ember_is_14_l2558_255802

-- Define the initial ages
def nate_initial_age : ℕ := 14
def ember_initial_age : ℕ := nate_initial_age / 2

-- Define the target age for Ember
def ember_target_age : ℕ := 14

-- Calculate the age difference
def age_difference : ℕ := ember_target_age - ember_initial_age

-- Theorem statement
theorem nate_age_when_ember_is_14 :
  nate_initial_age + age_difference = 21 :=
sorry

end NUMINAMATH_CALUDE_nate_age_when_ember_is_14_l2558_255802


namespace NUMINAMATH_CALUDE_fourth_angle_is_85_l2558_255887

/-- A quadrilateral with three known angles -/
structure Quadrilateral where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  sum_360 : angle1 + angle2 + angle3 + angle4 = 360

/-- The theorem stating that the fourth angle is 85° -/
theorem fourth_angle_is_85 (q : Quadrilateral) 
  (h1 : q.angle1 = 75) 
  (h2 : q.angle2 = 80) 
  (h3 : q.angle3 = 120) : 
  q.angle4 = 85 := by
  sorry


end NUMINAMATH_CALUDE_fourth_angle_is_85_l2558_255887


namespace NUMINAMATH_CALUDE_angle_D_measure_l2558_255820

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180

-- Define the quadrilateral formed by drawing a line
structure Quadrilateral (t : Triangle) where
  D : ℝ
  line_sum_180 : D + (180 - t.A - t.B) = 180

-- Theorem statement
theorem angle_D_measure (t : Triangle) (q : Quadrilateral t) 
  (h1 : t.A = 85) (h2 : t.B = 34) : q.D = 119 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l2558_255820


namespace NUMINAMATH_CALUDE_apple_distribution_l2558_255825

theorem apple_distribution (total_apples : ℕ) (num_babies : ℕ) (min_apples : ℕ) (max_apples : ℕ) :
  total_apples = 30 →
  num_babies = 7 →
  min_apples = 3 →
  max_apples = 6 →
  ∃ (removed : ℕ), 
    (total_apples - removed) % num_babies = 0 ∧
    (total_apples - removed) / num_babies ≥ min_apples ∧
    (total_apples - removed) / num_babies ≤ max_apples ∧
    removed = 2 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l2558_255825


namespace NUMINAMATH_CALUDE_max_reflections_theorem_l2558_255847

/-- The angle between two intersecting lines in degrees -/
def angle_between_lines : ℝ := 10

/-- The maximum number of reflections before hitting perpendicularly -/
def max_reflections : ℕ := 18

/-- Theorem stating the maximum number of reflections -/
theorem max_reflections_theorem : 
  ∀ (n : ℕ), n * angle_between_lines ≤ 180 → n ≤ max_reflections :=
sorry

end NUMINAMATH_CALUDE_max_reflections_theorem_l2558_255847


namespace NUMINAMATH_CALUDE_no_real_arithmetic_progression_l2558_255837

theorem no_real_arithmetic_progression : ¬ ∃ (a b : ℝ), 
  (b - a = a - 15) ∧ (a * b - b = b - a) := by
  sorry

end NUMINAMATH_CALUDE_no_real_arithmetic_progression_l2558_255837


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_open_interval_l2558_255814

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.exp x}

-- Theorem statement
theorem A_intersect_B_equals_open_interval : 
  A ∩ B = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_open_interval_l2558_255814


namespace NUMINAMATH_CALUDE_bush_distance_theorem_l2558_255815

/-- The distance between equally spaced bushes along a road -/
def bush_distance (n : ℕ) (d : ℝ) : ℝ :=
  d * (n - 1)

/-- Theorem: Given 10 equally spaced bushes where the distance between
    the first and fifth bush is 100 feet, the distance between the first
    and last bush is 225 feet. -/
theorem bush_distance_theorem :
  bush_distance 5 100 = 100 →
  bush_distance 10 (100 / 4) = 225 := by
  sorry

end NUMINAMATH_CALUDE_bush_distance_theorem_l2558_255815


namespace NUMINAMATH_CALUDE_largest_constant_for_good_array_l2558_255804

def isGoodArray (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a < b ∧
  Nat.lcm a b + Nat.lcm (a + 2) (b + 2) = 2 * Nat.lcm (a + 1) (b + 1)

theorem largest_constant_for_good_array :
  (∃ c : ℚ, c > 0 ∧
    (∀ a b : ℕ, isGoodArray a b → b > c * a^3) ∧
    (∀ ε > 0, ∃ a b : ℕ, isGoodArray a b ∧ b ≤ (c + ε) * a^3)) ∧
  (let c := (1/2 : ℚ); 
   (∀ a b : ℕ, isGoodArray a b → b > c * a^3) ∧
   (∀ ε > 0, ∃ a b : ℕ, isGoodArray a b ∧ b ≤ (c + ε) * a^3)) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_for_good_array_l2558_255804


namespace NUMINAMATH_CALUDE_expression_simplification_l2558_255834

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2 - x) + 4*(x + 2) - 5*(3 - 2*x) = 19*x - 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2558_255834


namespace NUMINAMATH_CALUDE_equation_system_ratio_l2558_255867

theorem equation_system_ratio (x y z : ℝ) 
  (eq1 : 2*x - 3*y - z = 0)
  (eq2 : x + 3*y - 14*z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 3*x*y) / (y^2 + z^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_equation_system_ratio_l2558_255867


namespace NUMINAMATH_CALUDE_julia_car_rental_cost_l2558_255872

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, 
    number of days, and miles driven. -/
def carRentalCost (dailyRate : ℚ) (mileageRate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  dailyRate * days + mileageRate * miles

/-- Theorem stating that the total cost for Julia's car rental is $215 -/
theorem julia_car_rental_cost :
  carRentalCost 30 0.25 3 500 = 215 := by
  sorry

end NUMINAMATH_CALUDE_julia_car_rental_cost_l2558_255872


namespace NUMINAMATH_CALUDE_even_decreasing_implies_inequality_l2558_255827

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem even_decreasing_implies_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) (h_decr : decreasing_on_nonneg f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_implies_inequality_l2558_255827


namespace NUMINAMATH_CALUDE_even_result_more_likely_l2558_255849

/-- Represents a calculator operation -/
inductive Operation
  | Add : Nat → Operation
  | Subtract : Nat → Operation
  | Multiply : Nat → Operation

/-- Represents a sequence of calculator operations -/
def OperationSequence := List Operation

/-- Applies a single operation to a number -/
def applyOperation (n : Int) (op : Operation) : Int :=
  match op with
  | Operation.Add m => n + m
  | Operation.Subtract m => n - m
  | Operation.Multiply m => n * m

/-- Applies a sequence of operations to an initial number -/
def applySequence (initial : Int) (seq : OperationSequence) : Int :=
  seq.foldl applyOperation initial

/-- Probability of getting an even result from a random operation sequence -/
noncomputable def probEvenResult (seqLength : Nat) : Real :=
  sorry

theorem even_result_more_likely (seqLength : Nat) :
  probEvenResult seqLength > 1 / 2 := by sorry

end NUMINAMATH_CALUDE_even_result_more_likely_l2558_255849


namespace NUMINAMATH_CALUDE_f_monotonicity_and_max_value_l2558_255863

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * x^2 - 6 * x + 2

-- State the theorem
theorem f_monotonicity_and_max_value :
  -- Part 1: Monotonicity
  (∀ x y : ℝ, x < y ∧ y < 3/4 → f x > f y) ∧
  (∀ x y : ℝ, 3/4 < x ∧ x < y → f x < f y) ∧
  -- Part 2: Maximum value on [2, 4]
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → f x ≤ 42) ∧
  (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ f x = 42) :=
by sorry


end NUMINAMATH_CALUDE_f_monotonicity_and_max_value_l2558_255863


namespace NUMINAMATH_CALUDE_expression_simplification_l2558_255844

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (3 / (a - 1) + (a - 3) / (a^2 - 1)) / (a / (a + 1)) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2558_255844


namespace NUMINAMATH_CALUDE_function_range_l2558_255879

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem function_range :
  {y | ∃ x ∈ domain, f x = y} = {y | -1 ≤ y ∧ y < 3} := by sorry

end NUMINAMATH_CALUDE_function_range_l2558_255879


namespace NUMINAMATH_CALUDE_sum_of_product_sequence_l2558_255853

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ 2 * (a 3) = a 2

def arithmetic_sequence (b : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  b 1 = 1 ∧ S 3 = b 2 + 4

theorem sum_of_product_sequence
  (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ) (T : ℕ → ℚ) :
  geometric_sequence a →
  arithmetic_sequence b S →
  (∀ n : ℕ, T n = (a n) * (b n)) →
  ∀ n : ℕ, T n = 2 - (n + 2) * (1/2)^n :=
by sorry

end NUMINAMATH_CALUDE_sum_of_product_sequence_l2558_255853


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l2558_255838

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

theorem cubic_function_extrema (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0) →
  a ∈ Set.Iic (-3) ∪ Set.Ioi 6 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l2558_255838


namespace NUMINAMATH_CALUDE_cows_not_black_l2558_255801

theorem cows_not_black (total : ℕ) (black : ℕ) : total = 18 → black = (total / 2 + 5) → total - black = 4 := by
  sorry

end NUMINAMATH_CALUDE_cows_not_black_l2558_255801


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l2558_255805

structure Survey where
  population_size : ℕ
  sample_size : ℕ
  has_distinct_subgroups : Bool

def is_large_population (s : Survey) : Bool :=
  s.population_size ≥ 1000

def is_small_population (s : Survey) : Bool :=
  s.population_size < 100

def stratified_sampling_appropriate (s : Survey) : Bool :=
  is_large_population s ∧ s.has_distinct_subgroups

def simple_random_sampling_appropriate (s : Survey) : Bool :=
  is_small_population s ∧ ¬s.has_distinct_subgroups

theorem appropriate_sampling_methods 
  (survey_A survey_B : Survey)
  (h_A : survey_A.population_size = 20000 ∧ survey_A.sample_size = 200 ∧ survey_A.has_distinct_subgroups = true)
  (h_B : survey_B.population_size = 15 ∧ survey_B.sample_size = 3 ∧ survey_B.has_distinct_subgroups = false) :
  stratified_sampling_appropriate survey_A ∧ simple_random_sampling_appropriate survey_B :=
sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l2558_255805


namespace NUMINAMATH_CALUDE_prime_square_mod_twelve_l2558_255821

theorem prime_square_mod_twelve (p : Nat) (h_prime : Nat.Prime p) (h_gt_three : p > 3) :
  p^2 % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_twelve_l2558_255821


namespace NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_l2558_255899

theorem five_fourths_of_twelve_fifths (x : ℚ) : x = 5/4 * (12/5) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_l2558_255899


namespace NUMINAMATH_CALUDE_product_mod_25_l2558_255892

theorem product_mod_25 : 68 * 97 * 113 ≡ 23 [ZMOD 25] := by sorry

end NUMINAMATH_CALUDE_product_mod_25_l2558_255892


namespace NUMINAMATH_CALUDE_point_on_line_l2558_255852

theorem point_on_line (m : ℝ) : (5 : ℝ) = 2 * m + 1 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2558_255852


namespace NUMINAMATH_CALUDE_initial_average_calculation_l2558_255859

theorem initial_average_calculation (n : ℕ) (correct_avg : ℚ) (error : ℚ) : 
  n = 10 → 
  correct_avg = 6 → 
  error = 10 → 
  (n * correct_avg - error) / n = 5 := by
sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l2558_255859


namespace NUMINAMATH_CALUDE_probability_of_s_in_statistics_l2558_255822

def word : String := "statistics"

def count_letter (w : String) (c : Char) : Nat :=
  w.toList.filter (· = c) |>.length

theorem probability_of_s_in_statistics :
  (count_letter word 's' : ℚ) / word.length = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_s_in_statistics_l2558_255822
