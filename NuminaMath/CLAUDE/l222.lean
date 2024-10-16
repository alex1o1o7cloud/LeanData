import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_properties_l222_22259

def z : ℂ := (1 - Complex.I)^2 + 1 + 3 * Complex.I

theorem complex_number_properties :
  (z = 3 + 3 * Complex.I) ∧
  (Complex.abs z = 3 * Real.sqrt 2) ∧
  (∃ (a b : ℝ), z^2 + a * z + b = 1 - Complex.I ∧ a = -6 ∧ b = 10) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l222_22259


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l222_22238

/-- Represents a repeating decimal where the digit repeats infinitely after the decimal point. -/
def repeating_decimal (d : ℕ) := (d : ℚ) / 9

/-- The sum of the repeating decimals 0.333... and 0.222... is equal to 5/9. -/
theorem sum_of_repeating_decimals : 
  repeating_decimal 3 + repeating_decimal 2 = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l222_22238


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l222_22298

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l222_22298


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l222_22284

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis. -/
def symmetricXAxis (p q : Point) : Prop :=
  q.x = p.x ∧ q.y = -p.y

/-- 
Given point P(-1, 2) and point Q symmetric to P with respect to the x-axis,
prove that the coordinates of Q are (-1, -2).
-/
theorem symmetric_point_coordinates :
  let P : Point := ⟨-1, 2⟩
  let Q : Point := ⟨-1, -2⟩
  symmetricXAxis P Q → Q = ⟨-1, -2⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l222_22284


namespace NUMINAMATH_CALUDE_pedestrians_collinear_at_most_twice_l222_22233

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a pedestrian's motion in 2D space -/
structure Pedestrian where
  initial_pos : Point2D
  velocity : Point2D

/-- Three pedestrians walking in straight lines -/
def three_pedestrians (p1 p2 p3 : Pedestrian) : Prop :=
  -- Pedestrians have constant velocities
  ∀ t : ℝ, ∃ (pos1 pos2 pos3 : Point2D),
    pos1 = Point2D.mk (p1.initial_pos.x + p1.velocity.x * t) (p1.initial_pos.y + p1.velocity.y * t) ∧
    pos2 = Point2D.mk (p2.initial_pos.x + p2.velocity.x * t) (p2.initial_pos.y + p2.velocity.y * t) ∧
    pos3 = Point2D.mk (p3.initial_pos.x + p3.velocity.x * t) (p3.initial_pos.y + p3.velocity.y * t)

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- The main theorem -/
theorem pedestrians_collinear_at_most_twice
  (p1 p2 p3 : Pedestrian)
  (h_not_initially_collinear : ¬are_collinear p1.initial_pos p2.initial_pos p3.initial_pos)
  (h_walking : three_pedestrians p1 p2 p3) :
  ∃ (t1 t2 : ℝ), ∀ t : ℝ,
    are_collinear
      (Point2D.mk (p1.initial_pos.x + p1.velocity.x * t) (p1.initial_pos.y + p1.velocity.y * t))
      (Point2D.mk (p2.initial_pos.x + p2.velocity.x * t) (p2.initial_pos.y + p2.velocity.y * t))
      (Point2D.mk (p3.initial_pos.x + p3.velocity.x * t) (p3.initial_pos.y + p3.velocity.y * t))
    → t = t1 ∨ t = t2 :=
  sorry

end NUMINAMATH_CALUDE_pedestrians_collinear_at_most_twice_l222_22233


namespace NUMINAMATH_CALUDE_min_product_abc_l222_22270

theorem min_product_abc (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b + c = 3)
  (h5 : a ≤ 3 * b ∧ a ≤ 3 * c)
  (h6 : b ≤ 3 * a ∧ b ≤ 3 * c)
  (h7 : c ≤ 3 * a ∧ c ≤ 3 * b) :
  81 / 125 ≤ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_min_product_abc_l222_22270


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l222_22247

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (h4 : a + b = 7 * (a - b) + 14) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l222_22247


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l222_22291

/-- The perimeter of a semicircle with radius 12 units is approximately 61.7 units. -/
theorem semicircle_perimeter_approx : 
  let r : ℝ := 12
  let perimeter := 2 * r + π * r
  ∃ ε > 0, abs (perimeter - 61.7) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l222_22291


namespace NUMINAMATH_CALUDE_rectangular_plot_poles_l222_22251

/-- The number of poles needed to enclose a rectangular plot -/
def num_poles (length width pole_distance : ℕ) : ℕ :=
  2 * (length + width) / pole_distance + 4

theorem rectangular_plot_poles :
  num_poles 90 50 4 = 74 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_poles_l222_22251


namespace NUMINAMATH_CALUDE_alberts_age_to_marys_age_ratio_l222_22225

-- Define the ages as natural numbers
def Betty : ℕ := 7
def Albert : ℕ := 4 * Betty
def Mary : ℕ := Albert - 14

-- Define the ratio of Albert's age to Mary's age
def age_ratio : ℚ := Albert / Mary

-- Theorem statement
theorem alberts_age_to_marys_age_ratio :
  age_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_alberts_age_to_marys_age_ratio_l222_22225


namespace NUMINAMATH_CALUDE_fifth_selected_number_is_12_l222_22288

-- Define the type for student numbers
def StudentNumber := Fin 50

-- Define the random number table as a list of natural numbers
def randomNumberTable : List ℕ :=
  [0627, 4313, 2432, 5327, 0941, 2512, 6317, 6323, 2616, 8045, 6011,
   1410, 9577, 7424, 6762, 4281, 1457, 2042, 5332, 3732, 2707, 3607,
   5124, 5179, 3014, 2310, 2118, 2191, 3726, 3890, 0140, 0523, 2617]

-- Define a function to check if a number is valid (between 01 and 50)
def isValidNumber (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 50

-- Define a function to select valid numbers from the table
def selectValidNumbers (table : List ℕ) : List StudentNumber :=
  sorry

-- State the theorem
theorem fifth_selected_number_is_12 :
  (selectValidNumbers randomNumberTable).nthLe 4 sorry = ⟨12, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_fifth_selected_number_is_12_l222_22288


namespace NUMINAMATH_CALUDE_cyclist_speed_l222_22215

/-- Proves that given a hiker walking at 4 km/h and a cyclist who stops 5 minutes after passing the hiker,
    if it takes the hiker 17.5 minutes to catch up to the cyclist, then the cyclist's speed is 14 km/h. -/
theorem cyclist_speed (hiker_speed : ℝ) (cyclist_ride_time : ℝ) (hiker_catch_up_time : ℝ) :
  hiker_speed = 4 →
  cyclist_ride_time = 5 / 60 →
  hiker_catch_up_time = 17.5 / 60 →
  ∃ (cyclist_speed : ℝ),
    cyclist_speed * cyclist_ride_time = hiker_speed * (cyclist_ride_time + hiker_catch_up_time) ∧
    cyclist_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_l222_22215


namespace NUMINAMATH_CALUDE_intersection_M_N_l222_22286

-- Define the sets M and N
def M : Set ℝ := {x | (x + 2) * (x - 2) ≤ 0}
def N : Set ℝ := {x | x - 1 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l222_22286


namespace NUMINAMATH_CALUDE_price_increase_condition_l222_22230

/-- Represents the fruit purchase and sale scenario -/
structure FruitSale where
  quantity : ℝ  -- Initial quantity in kg
  price : ℝ     -- Purchase price per kg
  loss : ℝ      -- Fraction of loss during transportation
  profit : ℝ    -- Desired minimum profit fraction
  increase : ℝ  -- Fraction of price increase

/-- Theorem stating the condition for the required price increase -/
theorem price_increase_condition (sale : FruitSale) 
  (h1 : sale.quantity = 200)
  (h2 : sale.price = 5)
  (h3 : sale.loss = 0.05)
  (h4 : sale.profit = 0.2) :
  (1 - sale.loss) * (1 + sale.increase) ≥ (1 + sale.profit) :=
sorry

end NUMINAMATH_CALUDE_price_increase_condition_l222_22230


namespace NUMINAMATH_CALUDE_unique_positive_solution_l222_22283

-- Define the polynomial function
def g (x : ℝ) : ℝ := x^8 + 5*x^7 + 10*x^6 + 729*x^5 - 379*x^4

-- Theorem statement
theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ g x = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l222_22283


namespace NUMINAMATH_CALUDE_literary_readers_l222_22202

theorem literary_readers (total : ℕ) (science_fiction : ℕ) (both : ℕ) 
  (h1 : total = 400) 
  (h2 : science_fiction = 250) 
  (h3 : both = 80) 
  (h4 : science_fiction ≥ both) 
  (h5 : total ≥ science_fiction) : 
  ∃ literary : ℕ, literary = total - science_fiction + both ∧ literary = 230 := by
sorry

end NUMINAMATH_CALUDE_literary_readers_l222_22202


namespace NUMINAMATH_CALUDE_cube_sum_plus_triple_product_l222_22241

theorem cube_sum_plus_triple_product (x y : ℝ) (h : x + y = 1) :
  x^3 + y^3 + 3*x*y = 1 := by sorry

end NUMINAMATH_CALUDE_cube_sum_plus_triple_product_l222_22241


namespace NUMINAMATH_CALUDE_event_children_count_l222_22219

/-- Calculates the number of children at an event after adding more children --/
theorem event_children_count (total_guests men_count added_children : ℕ) : 
  total_guests = 80 →
  men_count = 40 →
  added_children = 10 →
  let women_count := men_count / 2
  let initial_children := total_guests - (men_count + women_count)
  initial_children + added_children = 30 := by
  sorry

#check event_children_count

end NUMINAMATH_CALUDE_event_children_count_l222_22219


namespace NUMINAMATH_CALUDE_calculate_required_hours_per_week_l222_22278

/-- Proves that given an initial work plan and a period of unavailable work time,
    the required hours per week to meet the financial goal can be calculated. -/
theorem calculate_required_hours_per_week 
  (initial_hours_per_week : ℝ)
  (initial_weeks : ℝ)
  (financial_goal : ℝ)
  (unavailable_weeks : ℝ)
  (h1 : initial_hours_per_week = 25)
  (h2 : initial_weeks = 15)
  (h3 : financial_goal = 4500)
  (h4 : unavailable_weeks = 3)
  : (initial_hours_per_week * initial_weeks) / (initial_weeks - unavailable_weeks) = 31.25 := by
  sorry

end NUMINAMATH_CALUDE_calculate_required_hours_per_week_l222_22278


namespace NUMINAMATH_CALUDE_cheryl_egg_difference_l222_22292

/-- The number of eggs found by Kevin -/
def kevin_eggs : ℕ := 5

/-- The number of eggs found by Bonnie -/
def bonnie_eggs : ℕ := 13

/-- The number of eggs found by George -/
def george_eggs : ℕ := 9

/-- The number of eggs found by Cheryl -/
def cheryl_eggs : ℕ := 56

/-- Theorem stating that Cheryl found 29 more eggs than the other three children combined -/
theorem cheryl_egg_difference : 
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_egg_difference_l222_22292


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_set_l222_22277

theorem absolute_value_equation_solution_set :
  {x : ℝ | |x / (x - 1)| = x / (x - 1)} = {x : ℝ | x ≤ 0 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_set_l222_22277


namespace NUMINAMATH_CALUDE_age_difference_l222_22257

/-- Proves that given Sachin's age is 14 years and the ratio of Sachin's age to Rahul's age is 6:9,
    the difference between Rahul's age and Sachin's age is 7 years. -/
theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 14 →
  sachin_age * 9 = rahul_age * 6 →
  rahul_age - sachin_age = 7 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l222_22257


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l222_22222

theorem units_digit_of_expression (k : ℕ) : k = 2025^2 + 3^2025 → (k^2 + 3^k) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l222_22222


namespace NUMINAMATH_CALUDE_sum_of_fractions_simplification_l222_22279

theorem sum_of_fractions_simplification (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h_sum : a + b + c + d = 0) :
  (1 / (b^2 + c^2 + d^2 - a^2)) + 
  (1 / (a^2 + c^2 + d^2 - b^2)) + 
  (1 / (a^2 + b^2 + d^2 - c^2)) + 
  (1 / (a^2 + b^2 + c^2 - d^2)) = 4 / d^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_simplification_l222_22279


namespace NUMINAMATH_CALUDE_not_ripe_apples_l222_22287

theorem not_ripe_apples (total : ℕ) (good : ℕ) (h1 : total = 14) (h2 : good = 8) :
  total - good = 6 := by
  sorry

end NUMINAMATH_CALUDE_not_ripe_apples_l222_22287


namespace NUMINAMATH_CALUDE_G_representation_and_difference_l222_22266

def G : ℚ := 817 / 999

theorem G_representation_and_difference : 
  (G = 817 / 999) ∧ (999 - 817 = 182) := by sorry

end NUMINAMATH_CALUDE_G_representation_and_difference_l222_22266


namespace NUMINAMATH_CALUDE_green_balls_count_l222_22275

theorem green_balls_count (total : ℕ) (red blue green : ℕ) : 
  red + blue + green = total →
  red = total / 3 →
  blue = (2 * total) / 7 →
  green = 2 * blue - 8 →
  green = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l222_22275


namespace NUMINAMATH_CALUDE_rational_equation_solution_l222_22262

theorem rational_equation_solution (x : ℝ) : 
  (x^2 - 7*x + 10) / (x^2 - 9*x + 8) = (x^2 - 3*x - 18) / (x^2 - 4*x - 21) ↔ x = 11 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l222_22262


namespace NUMINAMATH_CALUDE_add_10000_seconds_to_5_45_00_l222_22263

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- Converts a time to total seconds -/
def toSeconds (t : Time) : Nat :=
  sorry

theorem add_10000_seconds_to_5_45_00 :
  let start_time : Time := ⟨5, 45, 0⟩
  let end_time : Time := ⟨8, 31, 40⟩
  addSeconds start_time 10000 = end_time :=
sorry

end NUMINAMATH_CALUDE_add_10000_seconds_to_5_45_00_l222_22263


namespace NUMINAMATH_CALUDE_monochromatic_triangle_in_K17_l222_22249

/-- A coloring of the edges of a complete graph -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A triangle in a graph -/
def Triangle (n : ℕ) := { t : Fin n × Fin n × Fin n // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2 }

/-- A monochromatic triangle in a coloring -/
def MonochromaticTriangle (n : ℕ) (c : Coloring n) (t : Triangle n) : Prop :=
  c t.val.1 t.val.2.1 = c t.val.1 t.val.2.2 ∧ c t.val.1 t.val.2.2 = c t.val.2.1 t.val.2.2

/-- The main theorem: any 3-coloring of K₁₇ contains a monochromatic triangle -/
theorem monochromatic_triangle_in_K17 :
  ∀ c : Coloring 17, ∃ t : Triangle 17, MonochromaticTriangle 17 c t := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_in_K17_l222_22249


namespace NUMINAMATH_CALUDE_bowling_ball_surface_area_l222_22274

theorem bowling_ball_surface_area :
  let diameter : ℝ := 9
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius^2
  surface_area = 81 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_surface_area_l222_22274


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l222_22208

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 4| + |x - a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 2 x > 10} = {x : ℝ | x > 8 ∨ x < -2} := by sorry

-- Part II
theorem range_of_a_part_ii :
  (∀ x : ℝ, f a x ≥ 1) → (a ≥ 5 ∨ a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l222_22208


namespace NUMINAMATH_CALUDE_problem_solution_l222_22232

def f (x a : ℝ) := |2*x - a| + |2*x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x (-1) ≤ 2 ↔ x ∈ Set.Icc (-1/2) (1/2)) ∧
  (Set.Icc (1/2) 1 ⊆ {x : ℝ | f x a ≤ |2*x + 1|} → a ∈ Set.Icc 0 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l222_22232


namespace NUMINAMATH_CALUDE_quadratic_with_rational_roots_has_even_coefficient_l222_22285

theorem quadratic_with_rational_roots_has_even_coefficient
  (a b c : ℕ+) -- a, b, c are positive integers
  (h_rational_roots : ∃ (p q r s : ℤ), (p * r ≠ 0 ∧ q * s ≠ 0) ∧
    (a * (p * s)^2 + b * (p * s) * (q * r) + c * (q * r)^2 = 0)) :
  Even a ∨ Even b ∨ Even c :=
sorry

end NUMINAMATH_CALUDE_quadratic_with_rational_roots_has_even_coefficient_l222_22285


namespace NUMINAMATH_CALUDE_beth_crayon_count_l222_22209

/-- The number of crayon packs Beth has -/
def num_packs : ℕ := 4

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 10

/-- The number of extra crayons Beth has -/
def extra_crayons : ℕ := 6

/-- The total number of crayons Beth has -/
def total_crayons : ℕ := num_packs * crayons_per_pack + extra_crayons

theorem beth_crayon_count : total_crayons = 46 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayon_count_l222_22209


namespace NUMINAMATH_CALUDE_derivative_properties_neg_l222_22218

open Real

-- Define the properties of functions f and g
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

def positive_derivative_pos (f : ℝ → ℝ) : Prop := ∀ x > 0, deriv f x > 0

def negative_derivative_pos (g : ℝ → ℝ) : Prop := ∀ x > 0, deriv g x < 0

-- State the theorem
theorem derivative_properties_neg
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (hf_even : even_function f)
  (hg_odd : odd_function g)
  (hf_pos : positive_derivative_pos f)
  (hg_pos : negative_derivative_pos g) :
  ∀ x < 0, deriv f x < 0 ∧ deriv g x < 0 :=
sorry

end NUMINAMATH_CALUDE_derivative_properties_neg_l222_22218


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l222_22253

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 9) (h2 : Nat.lcm a b = 200) :
  a * b = 1800 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l222_22253


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l222_22226

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (- p.2, - p.1)

/-- The original center of the circle -/
def original_center : ℝ × ℝ := (3, -4)

/-- The expected center after reflection -/
def expected_reflected_center : ℝ × ℝ := (4, -3)

theorem reflection_of_circle_center :
  reflect_about_y_eq_neg_x original_center = expected_reflected_center :=
by sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l222_22226


namespace NUMINAMATH_CALUDE_simplify_square_roots_l222_22204

theorem simplify_square_roots :
  Real.sqrt 8 - Real.sqrt 32 + Real.sqrt 72 - Real.sqrt 50 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l222_22204


namespace NUMINAMATH_CALUDE_closed_path_vector_sum_l222_22216

/-- The sum of vectors forming a closed path in a plane is equal to the zero vector. -/
theorem closed_path_vector_sum (A B C D E F : ℝ × ℝ) : 
  (B.1 - A.1, B.2 - A.2) + (C.1 - B.1, C.2 - B.2) + (D.1 - C.1, D.2 - C.2) + 
  (E.1 - D.1, E.2 - D.2) + (F.1 - E.1, F.2 - E.2) + (A.1 - F.1, A.2 - F.2) = (0, 0) := by
sorry

end NUMINAMATH_CALUDE_closed_path_vector_sum_l222_22216


namespace NUMINAMATH_CALUDE_arithmetic_progression_equality_l222_22294

theorem arithmetic_progression_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((a + b) / 2 = (Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2) → a = b := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_equality_l222_22294


namespace NUMINAMATH_CALUDE_conference_attendance_theorem_l222_22206

/-- The percentage of attendees who paid their conference fee in full but did not register at least two weeks in advance -/
def late_payment_percentage : ℝ := 10

/-- The percentage of conference attendees who registered at least two weeks in advance -/
def early_registration_percentage : ℝ := 86.67

/-- The percentage of attendees who registered at least two weeks in advance and paid their conference fee in full -/
def early_registration_full_payment_percentage : ℝ := 96.3

theorem conference_attendance_theorem :
  (100 - late_payment_percentage) / 100 * early_registration_full_payment_percentage = early_registration_percentage :=
by sorry

end NUMINAMATH_CALUDE_conference_attendance_theorem_l222_22206


namespace NUMINAMATH_CALUDE_rikki_poetry_pricing_l222_22242

-- Define the constants
def words_per_interval : ℕ := 25
def minutes_per_interval : ℕ := 5
def total_minutes : ℕ := 120
def expected_earnings : ℚ := 6

-- Define the function to calculate the price per word
def price_per_word : ℚ :=
  let intervals : ℕ := total_minutes / minutes_per_interval
  let total_words : ℕ := words_per_interval * intervals
  expected_earnings / total_words

-- Theorem statement
theorem rikki_poetry_pricing :
  price_per_word = 1/100 := by sorry

end NUMINAMATH_CALUDE_rikki_poetry_pricing_l222_22242


namespace NUMINAMATH_CALUDE_cubic_factorization_l222_22289

theorem cubic_factorization :
  ∀ x : ℝ, 343 * x^3 + 125 = (7 * x + 5) * (49 * x^2 - 35 * x + 25) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l222_22289


namespace NUMINAMATH_CALUDE_minimum_horses_and_ponies_l222_22200

theorem minimum_horses_and_ponies (ponies horses : ℕ) : 
  (3 * ponies % 10 = 0) →  -- 3/10 of ponies have horseshoes
  (5 * (3 * ponies / 10) % 8 = 0) →  -- 5/8 of ponies with horseshoes are from Iceland
  (horses = ponies + 3) →  -- 3 more horses than ponies
  (∀ p h, p < ponies ∨ h < horses → 
    3 * p % 10 ≠ 0 ∨ 
    5 * (3 * p / 10) % 8 ≠ 0 ∨ 
    h ≠ p + 3) →  -- minimality condition
  ponies + horses = 163 := by
sorry

end NUMINAMATH_CALUDE_minimum_horses_and_ponies_l222_22200


namespace NUMINAMATH_CALUDE_jony_turnaround_block_l222_22271

/-- Represents the walking scenario of Jony along Sunrise Boulevard -/
structure WalkingScenario where
  start_block : ℕ
  end_block : ℕ
  block_length : ℕ
  walking_speed : ℕ
  walking_time : ℕ

/-- Calculates the block where Jony turns around -/
def turnaround_block (scenario : WalkingScenario) : ℕ :=
  let total_distance := scenario.walking_speed * scenario.walking_time
  let start_to_end_distance := (scenario.end_block - scenario.start_block) * scenario.block_length
  let extra_distance := total_distance - start_to_end_distance
  let extra_blocks := extra_distance / scenario.block_length
  scenario.end_block + extra_blocks

/-- Theorem stating that Jony turns around at block 110 -/
theorem jony_turnaround_block :
  let scenario : WalkingScenario := {
    start_block := 10,
    end_block := 70,
    block_length := 40,
    walking_speed := 100,
    walking_time := 40
  }
  turnaround_block scenario = 110 := by
  sorry

end NUMINAMATH_CALUDE_jony_turnaround_block_l222_22271


namespace NUMINAMATH_CALUDE_fraction_product_l222_22260

theorem fraction_product : (2 : ℚ) / 3 * 5 / 7 * 11 / 13 * 17 / 19 = 1870 / 5187 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l222_22260


namespace NUMINAMATH_CALUDE_reciprocal_relationship_l222_22290

theorem reciprocal_relationship (a b : ℝ) : (a + b)^2 - (a - b)^2 = 4 → a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_relationship_l222_22290


namespace NUMINAMATH_CALUDE_eight_flavors_twentyeight_sundaes_l222_22245

/-- The number of unique two scoop sundaes with distinct flavors given n flavors of ice cream -/
def uniqueSundaes (n : ℕ) : ℕ := Nat.choose n 2

/-- Theorem stating that with 8 flavors, there are 28 unique two scoop sundaes -/
theorem eight_flavors_twentyeight_sundaes : uniqueSundaes 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_eight_flavors_twentyeight_sundaes_l222_22245


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l222_22217

-- Define a quadratic polynomial P(x) = ax^2 + bx + c
def P (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic polynomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- State the theorem
theorem quadratic_discriminant (a b c : ℝ) (h₁ : a ≠ 0) :
  (∃! x, P a b c x = x - 2) →
  (∃! x, P a b c x = 1 - x/2) →
  discriminant a b c = -1/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l222_22217


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l222_22267

theorem point_in_third_quadrant (a b : ℝ) (h : a < b ∧ b < 0) :
  (a - b < 0) ∧ (b < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l222_22267


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l222_22235

theorem magnitude_of_complex_fraction (z : ℂ) : 
  z = (3 + Complex.I) / (1 + Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l222_22235


namespace NUMINAMATH_CALUDE_boys_age_problem_l222_22227

theorem boys_age_problem (age1 age2 age3 : ℕ) : 
  age1 + age2 + age3 = 29 →
  age1 = age2 →
  age3 = 11 →
  age1 = 9 ∧ age2 = 9 := by
sorry

end NUMINAMATH_CALUDE_boys_age_problem_l222_22227


namespace NUMINAMATH_CALUDE_equation_solutions_l222_22243

theorem equation_solutions : 
  ∀ x : ℝ, (x^2 + x)^2 + (x^2 + x) - 6 = 0 ↔ x = -2 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l222_22243


namespace NUMINAMATH_CALUDE_triangle_area_l222_22256

theorem triangle_area (a b c : ℝ) (h1 : c^2 = (a - b)^2 + 6) (h2 : c = π/3) :
  (1/2) * a * b * Real.sin (π/3) = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l222_22256


namespace NUMINAMATH_CALUDE_only_solutions_for_equation_l222_22207

theorem only_solutions_for_equation (x y : ℕ) : 
  33^x + 31 = 2^y ↔ (x = 0 ∧ y = 5) ∨ (x = 1 ∧ y = 6) := by
  sorry

end NUMINAMATH_CALUDE_only_solutions_for_equation_l222_22207


namespace NUMINAMATH_CALUDE_illumination_theorem_l222_22272

/-- Calculates the total number of nights a house can be illuminated given an initial number of candles. -/
def totalNights (initialCandles : ℕ) : ℕ :=
  let rec aux (candles stubs nights : ℕ) : ℕ :=
    if candles = 0 then
      nights + (stubs / 4)
    else
      aux (candles - 1) (stubs + 1) (nights + 1)
  aux initialCandles 0 0

/-- Theorem stating that 43 initial candles result in 57 nights of illumination. -/
theorem illumination_theorem :
  totalNights 43 = 57 := by
  sorry

end NUMINAMATH_CALUDE_illumination_theorem_l222_22272


namespace NUMINAMATH_CALUDE_A_inter_B_eq_two_three_l222_22265

def A : Set ℕ := {x | (x - 2) * (x - 4) ≤ 0}

def B : Set ℕ := {x | x ≤ 3}

theorem A_inter_B_eq_two_three : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_A_inter_B_eq_two_three_l222_22265


namespace NUMINAMATH_CALUDE_phone_number_pricing_l222_22261

theorem phone_number_pricing (X Y : ℤ) : 
  (0 < X ∧ X < 250) →
  (0 < Y ∧ Y < 250) →
  125 * X - 64 * Y = 5 →
  ((X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205)) := by
sorry

end NUMINAMATH_CALUDE_phone_number_pricing_l222_22261


namespace NUMINAMATH_CALUDE_equality_condition_for_sum_squares_equation_l222_22254

theorem equality_condition_for_sum_squares_equation (a b c : ℝ) :
  (a^2 + b^2 + c^2 = a*b + b*c + a*c) ↔ (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_equality_condition_for_sum_squares_equation_l222_22254


namespace NUMINAMATH_CALUDE_unique_prime_with_remainder_l222_22211

theorem unique_prime_with_remainder : ∃! n : ℕ, 
  40 < n ∧ n < 50 ∧ 
  Nat.Prime n ∧ 
  n % 9 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_remainder_l222_22211


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l222_22281

theorem sufficient_but_not_necessary (a b : ℝ) :
  (((2 : ℝ) ^ a > (2 : ℝ) ^ b ∧ (2 : ℝ) ^ b > 1) → (a ^ (1/3) > b ^ (1/3))) ∧
  ¬(∀ a b : ℝ, a ^ (1/3) > b ^ (1/3) → ((2 : ℝ) ^ a > (2 : ℝ) ^ b ∧ (2 : ℝ) ^ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l222_22281


namespace NUMINAMATH_CALUDE_composite_condition_l222_22248

def is_composite (m : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ m = a * b

theorem composite_condition (n : ℕ) (hn : 0 < n) : 
  is_composite (3^(2*n+1) - 2^(2*n+1) - 6*n) ↔ n > 1 :=
sorry

end NUMINAMATH_CALUDE_composite_condition_l222_22248


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l222_22236

/-- Represents a parabola of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 3 ∧ p.h = 1 ∧ p.k = 1 →
  let shifted := shift (shift p 2 0) 0 2
  shifted.a = 3 ∧ shifted.h = 3 ∧ shifted.k = 3 := by
  sorry

#check parabola_shift_theorem

end NUMINAMATH_CALUDE_parabola_shift_theorem_l222_22236


namespace NUMINAMATH_CALUDE_parallelogram_area_60_16_l222_22250

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 60 cm and height 16 cm is 960 square centimeters -/
theorem parallelogram_area_60_16 : 
  parallelogram_area 60 16 = 960 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_60_16_l222_22250


namespace NUMINAMATH_CALUDE_fair_game_conditions_l222_22221

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  total : Nat
  white : Nat
  green : Nat
  black : Nat

/-- Checks if the game is fair given the bag contents -/
def isFairGame (bag : BagContents) : Prop :=
  bag.green = bag.black

/-- Theorem stating the conditions for a fair game -/
theorem fair_game_conditions (x : Nat) :
  let bag : BagContents := {
    total := 15,
    white := x,
    green := 2 * x,
    black := 15 - x - 2 * x
  }
  isFairGame bag ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fair_game_conditions_l222_22221


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l222_22220

/-- Given a triangle with sides 5, 12, and 13, and a similar triangle with perimeter 150,
    the longest side of the similar triangle is 65. -/
theorem similar_triangle_longest_side : ∀ (a b c : ℝ) (x : ℝ),
  a = 5 → b = 12 → c = 13 →
  a * x + b * x + c * x = 150 →
  max (a * x) (max (b * x) (c * x)) = 65 := by
sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_l222_22220


namespace NUMINAMATH_CALUDE_zahs_to_bahs_conversion_l222_22299

/-- Conversion rates between different currencies -/
structure CurrencyRates where
  bah_to_rah : ℚ
  rah_to_yah : ℚ
  yah_to_zah : ℚ

/-- Given conversion rates, calculate the number of bahs equivalent to a given number of zahs -/
def zahs_to_bahs (rates : CurrencyRates) (zahs : ℚ) : ℚ :=
  zahs / rates.yah_to_zah / rates.rah_to_yah / rates.bah_to_rah

/-- Theorem stating the equivalence between 1500 zahs and 400/3 bahs -/
theorem zahs_to_bahs_conversion (rates : CurrencyRates) 
  (h1 : rates.bah_to_rah = 3)
  (h2 : rates.rah_to_yah = 3/2)
  (h3 : rates.yah_to_zah = 5/2) : 
  zahs_to_bahs rates 1500 = 400/3 := by
  sorry

#eval zahs_to_bahs ⟨3, 3/2, 5/2⟩ 1500

end NUMINAMATH_CALUDE_zahs_to_bahs_conversion_l222_22299


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l222_22255

theorem concentric_circles_ratio (R r k : ℝ) (h1 : R > r) (h2 : k > 0) :
  (π * R^2 - π * r^2) = k * (π * r^2) → R / r = Real.sqrt (k + 1) :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l222_22255


namespace NUMINAMATH_CALUDE_rectangular_paper_to_hexagon_l222_22276

/-- A rectangular sheet of paper with sides a and b can be folded into a regular hexagon
    if and only if the aspect ratio b/a is between 1/2 and 2. -/
theorem rectangular_paper_to_hexagon (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x : ℝ), x > 0 ∧ x < a ∧ x < b ∧ (a - x)^2 + (b - x)^2 = x^2) ↔
  (1/2 < b/a ∧ b/a < 2) :=
sorry

end NUMINAMATH_CALUDE_rectangular_paper_to_hexagon_l222_22276


namespace NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l222_22258

theorem max_value_x_cubed_over_y_fourth (x y : ℝ) 
  (h1 : 3 ≤ x * y^2 ∧ x * y^2 ≤ 8) 
  (h2 : 4 ≤ x^2 / y ∧ x^2 / y ≤ 9) :
  ∃ (M : ℝ), M = 27 ∧ x^3 / y^4 ≤ M ∧ 
  ∃ (x₀ y₀ : ℝ), 3 ≤ x₀ * y₀^2 ∧ x₀ * y₀^2 ≤ 8 ∧ 
                 4 ≤ x₀^2 / y₀ ∧ x₀^2 / y₀ ≤ 9 ∧ 
                 x₀^3 / y₀^4 = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l222_22258


namespace NUMINAMATH_CALUDE_sequence_growth_l222_22228

theorem sequence_growth (a : ℕ → ℕ) (h1 : a 1 > a 0) 
  (h2 : ∀ n : ℕ, n ≥ 2 → a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_growth_l222_22228


namespace NUMINAMATH_CALUDE_cuts_for_331_pieces_l222_22231

/-- The number of cuts needed to transform initial sheets into a given number of pieces -/
def number_of_cuts (initial_sheets : ℕ) (final_pieces : ℕ) : ℕ :=
  (final_pieces - initial_sheets) / 6

/-- Theorem stating that 54 cuts are needed to transform 7 sheets into 331 pieces -/
theorem cuts_for_331_pieces : number_of_cuts 7 331 = 54 := by
  sorry

end NUMINAMATH_CALUDE_cuts_for_331_pieces_l222_22231


namespace NUMINAMATH_CALUDE_derivative_of_f_l222_22229

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

theorem derivative_of_f (x : ℝ) :
  deriv f x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l222_22229


namespace NUMINAMATH_CALUDE_line_through_point_l222_22224

theorem line_through_point (b : ℚ) : 
  (b * 3 + (b - 2) * (-5) = b - 1) → b = 11/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l222_22224


namespace NUMINAMATH_CALUDE_longest_diagonal_twice_side_l222_22293

/-- Regular octagon with side length a, shortest diagonal b, and longest diagonal c -/
structure RegularOctagon where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a

/-- Theorem: In a regular octagon, the longest diagonal is twice the side length -/
theorem longest_diagonal_twice_side (octagon : RegularOctagon) : octagon.c = 2 * octagon.a := by
  sorry

#check longest_diagonal_twice_side

end NUMINAMATH_CALUDE_longest_diagonal_twice_side_l222_22293


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_160_l222_22252

/-- The sum of the digits in the binary representation of 160 is 2. -/
theorem sum_of_binary_digits_160 : 
  (Nat.digits 2 160).sum = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_160_l222_22252


namespace NUMINAMATH_CALUDE_georgia_muffin_batches_l222_22240

/-- Calculates the number of muffin batches needed for a given number of months,
    students, and muffins per batch. -/
def muffin_batches (months : ℕ) (students : ℕ) (muffins_per_batch : ℕ) : ℕ :=
  months * (students / muffins_per_batch)

theorem georgia_muffin_batches :
  muffin_batches 9 24 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_georgia_muffin_batches_l222_22240


namespace NUMINAMATH_CALUDE_outfit_combinations_l222_22246

theorem outfit_combinations (shirts : Nat) (ties : Nat) (hats : Nat) :
  shirts = 8 → ties = 6 → hats = 4 → shirts * ties * hats = 192 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l222_22246


namespace NUMINAMATH_CALUDE_percentage_problem_l222_22280

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.1 * x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l222_22280


namespace NUMINAMATH_CALUDE_functional_equation_polynomial_l222_22213

/-- A polynomial that satisfies the functional equation P(X^2 + 1) = P(X)^2 + 1 and P(0) = 0 is equal to the identity function. -/
theorem functional_equation_polynomial (P : Polynomial ℝ) 
  (h1 : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1)
  (h2 : P.eval 0 = 0) : 
  P = Polynomial.X :=
sorry

end NUMINAMATH_CALUDE_functional_equation_polynomial_l222_22213


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l222_22239

/-- The area of a right triangle with legs of 45 inches and 48 inches is 1080 square inches. -/
theorem right_triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun leg1 leg2 area =>
    leg1 = 45 ∧ leg2 = 48 → area = (1 / 2) * leg1 * leg2 → area = 1080

/-- Proof of the theorem -/
theorem right_triangle_area_proof : right_triangle_area 45 48 1080 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l222_22239


namespace NUMINAMATH_CALUDE_hotel_room_charge_comparison_l222_22268

/-- Given the room charges for three hotels P, R, and G, prove that R's charge is 170% greater than G's. -/
theorem hotel_room_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.7 * R) 
  (h2 : P = G - 0.1 * G) : 
  (R - G) / G = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charge_comparison_l222_22268


namespace NUMINAMATH_CALUDE_strawberry_theorem_l222_22203

def strawberry_problem (brother_baskets : ℕ) (strawberries_per_basket : ℕ) 
  (kimberly_multiplier : ℕ) (parents_difference : ℕ) (family_members : ℕ) : Prop :=
  let brother_strawberries := brother_baskets * strawberries_per_basket
  let kimberly_strawberries := kimberly_multiplier * brother_strawberries
  let parents_strawberries := kimberly_strawberries - parents_difference
  let total_strawberries := brother_strawberries + kimberly_strawberries + parents_strawberries
  (total_strawberries / family_members = 168)

theorem strawberry_theorem : 
  strawberry_problem 3 15 8 93 4 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_theorem_l222_22203


namespace NUMINAMATH_CALUDE_stock_price_increase_l222_22273

theorem stock_price_increase (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_year1 := initial_price * 1.20
  let price_after_year2 := price_after_year1 * 0.75
  let price_after_year3 := initial_price * 1.035
  let increase_percentage := (price_after_year3 / price_after_year2 - 1) * 100
  increase_percentage = 15 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l222_22273


namespace NUMINAMATH_CALUDE_triangle_stack_impossibility_l222_22297

theorem triangle_stack_impossibility : ¬ ∃ (n : ℕ), 6 * n = 165 := by
  sorry

end NUMINAMATH_CALUDE_triangle_stack_impossibility_l222_22297


namespace NUMINAMATH_CALUDE_bryan_books_and_magazines_l222_22295

theorem bryan_books_and_magazines (books_per_shelf : ℕ) (magazines_per_shelf : ℕ) (num_shelves : ℕ) :
  books_per_shelf = 23 →
  magazines_per_shelf = 61 →
  num_shelves = 29 →
  books_per_shelf * num_shelves + magazines_per_shelf * num_shelves = 2436 :=
by
  sorry

end NUMINAMATH_CALUDE_bryan_books_and_magazines_l222_22295


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l222_22201

-- Problem 1
theorem problem_1 : 1/2 + (-2/3) + 4/5 + (-1/2) + (-1/3) = -1/5 := by sorry

-- Problem 2
theorem problem_2 : 2 * (-3)^3 - 4 * (-3) + 15 = -27 := by sorry

-- Problem 3
theorem problem_3 : (1/8 - 1/3 + 1 + 1/6) * (-48) = -46 := by sorry

-- Problem 4
theorem problem_4 : -2^4 - 32 / ((-2)^3 + 4) = -8 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l222_22201


namespace NUMINAMATH_CALUDE_adjustment_schemes_no_adjacent_boys_arrangements_specific_position_arrangements_l222_22296

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 4
def total_people : ℕ := num_boys + num_girls

-- Statement A
theorem adjustment_schemes :
  (Nat.choose total_people 3) * 2 = 70 := by sorry

-- Statement B
theorem no_adjacent_boys_arrangements :
  (Nat.factorial num_girls) * (Nat.factorial (num_girls + 1) / Nat.factorial (num_girls + 1 - num_boys)) = 1440 := by sorry

-- Statement D
theorem specific_position_arrangements :
  Nat.factorial total_people - 2 * Nat.factorial (total_people - 1) + Nat.factorial (total_people - 2) = 3720 := by sorry

end NUMINAMATH_CALUDE_adjustment_schemes_no_adjacent_boys_arrangements_specific_position_arrangements_l222_22296


namespace NUMINAMATH_CALUDE_angle_preservation_under_inversion_l222_22237

-- Define the types for geometric objects and inversion
def GeometricObject : Type := sorry
def Circle : GeometricObject := sorry
def Line : GeometricObject := sorry
def Inversion : Type := sorry

-- Define the angle between two geometric objects
def angle (a b : GeometricObject) : ℝ := sorry

-- Define the image of a geometric object under inversion
def image (i : Inversion) (g : GeometricObject) : GeometricObject := sorry

-- Statement of the theorem
theorem angle_preservation_under_inversion 
  (a b : GeometricObject) (i : Inversion) : 
  angle a b = angle (image i a) (image i b) := by sorry

end NUMINAMATH_CALUDE_angle_preservation_under_inversion_l222_22237


namespace NUMINAMATH_CALUDE_elvin_internet_charge_l222_22223

/-- Represents Elvin's monthly telephone bill structure -/
structure TelephoneBill where
  fixedCharge : ℝ
  callCharge : ℝ

/-- Calculates the total bill for a given month -/
def totalBill (bill : TelephoneBill) : ℝ :=
  bill.fixedCharge + bill.callCharge

theorem elvin_internet_charge : 
  ∀ (jan : TelephoneBill) (feb : TelephoneBill),
    totalBill jan = 50 →
    totalBill feb = 76 →
    feb.callCharge = 2 * jan.callCharge →
    jan.fixedCharge = feb.fixedCharge →
    jan.fixedCharge = 24 := by
  sorry


end NUMINAMATH_CALUDE_elvin_internet_charge_l222_22223


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l222_22282

/-- The eccentricity of a hyperbola with equation x²/4 - y² = 1 is √5/2 -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = Real.sqrt 5 / 2 ∧ 
  ∀ (x y : ℝ), x^2 / 4 - y^2 = 1 → e = Real.sqrt (1 + 1^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l222_22282


namespace NUMINAMATH_CALUDE_equation_solution_l222_22205

theorem equation_solution : 
  ∃! x : ℝ, Real.sqrt x + 3 * Real.sqrt (x^2 + 8*x) + Real.sqrt (x + 8) = 42 - 3*x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l222_22205


namespace NUMINAMATH_CALUDE_constant_sequence_l222_22212

def is_prime (n : ℤ) : Prop := Nat.Prime n.natAbs

theorem constant_sequence
  (a : ℕ → ℤ)  -- Sequence of integers
  (d : ℤ)      -- Integer d
  (h1 : ∀ n, is_prime (a n))  -- |a_n| is prime for all n
  (h2 : ∀ n, a (n + 2) = a (n + 1) + a n + d)  -- Recurrence relation
  : ∀ n, a n = a 0  -- Conclusion: sequence is constant
  := by sorry

end NUMINAMATH_CALUDE_constant_sequence_l222_22212


namespace NUMINAMATH_CALUDE_expression_equals_zero_l222_22234

theorem expression_equals_zero (a : ℚ) (h : a = 4/3) : 
  (6*a^2 - 15*a + 5) * (3*a - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l222_22234


namespace NUMINAMATH_CALUDE_smallest_divisor_of_1025_l222_22214

theorem smallest_divisor_of_1025 : 
  ∀ n : ℕ, n > 1 → n ∣ 1025 → n ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_1025_l222_22214


namespace NUMINAMATH_CALUDE_quadratic_root_sum_property_l222_22269

theorem quadratic_root_sum_property (a b c : ℝ) (x₁ x₂ : ℝ) (p q r : ℝ) 
  (h1 : a ≠ 0)
  (h2 : a * x₁^2 + b * x₁ + c = 0)
  (h3 : a * x₂^2 + b * x₂ + c = 0)
  (h4 : p = x₁ + x₂)
  (h5 : q = x₁^2 + x₂^2)
  (h6 : r = x₁^3 + x₂^3) :
  a * r + b * q + c * p = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_property_l222_22269


namespace NUMINAMATH_CALUDE_unique_digit_equation_sum_l222_22244

theorem unique_digit_equation_sum : ∃! (A B C D E : ℕ), 
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10) ∧ 
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E) ∧
  ((10 * A + B) * (10 * C + B) = 100 * C + 10 * D + E) ∧
  (D = C + 1 ∧ E = D + 1) ∧
  (A + B + C + D + E = 11) := by
sorry

end NUMINAMATH_CALUDE_unique_digit_equation_sum_l222_22244


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l222_22264

theorem geometric_sequence_second_term
  (a : ℕ → ℕ)  -- Sequence of natural numbers
  (h1 : a 1 = 1)  -- First term is 1
  (h2 : a 3 = 9)  -- Third term is 9
  (h_ratio : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 3 * a n)  -- Common ratio is 3
  : a 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l222_22264


namespace NUMINAMATH_CALUDE_point_not_in_transformed_plane_l222_22210

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def applySimiliarity (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point satisfies a plane equation -/
def satisfiesPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Main theorem: Point A does not belong to the image of plane a under the similarity transformation -/
theorem point_not_in_transformed_plane :
  let A : Point3D := { x := 5, y := 0, z := -6 }
  let a : Plane := { a := 6, b := -1, c := -1, d := 7 }
  let k : ℝ := 2/7
  let a_transformed := applySimiliarity a k
  ¬ satisfiesPlane A a_transformed :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_transformed_plane_l222_22210
