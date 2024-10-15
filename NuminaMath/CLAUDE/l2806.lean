import Mathlib

namespace NUMINAMATH_CALUDE_total_amount_paid_l2806_280674

def apple_quantity : ℕ := 8
def apple_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 45

theorem total_amount_paid : 
  apple_quantity * apple_rate + mango_quantity * mango_rate = 965 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l2806_280674


namespace NUMINAMATH_CALUDE_sum_ratio_l2806_280683

def sean_sum : ℕ → ℕ
| 0 => 0
| (n + 1) => sean_sum n + if (n + 1) * 3 ≤ 600 then (n + 1) * 3 else 0

def julie_sum : ℕ → ℕ
| 0 => 0
| (n + 1) => julie_sum n + if n + 1 ≤ 300 then n + 1 else 0

theorem sum_ratio :
  (sean_sum 200 : ℚ) / (julie_sum 300 : ℚ) = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_ratio_l2806_280683


namespace NUMINAMATH_CALUDE_max_figures_9x9_grid_l2806_280620

/-- Represents a square grid -/
structure Grid (n : ℕ) :=
  (size : ℕ)
  (h_size : size = n)

/-- Represents a square figure -/
structure Figure (m : ℕ) :=
  (size : ℕ)
  (h_size : size = m)

/-- The maximum number of non-overlapping figures that can fit in a grid -/
def max_figures (g : Grid n) (f : Figure m) : ℕ :=
  (g.size / f.size) ^ 2

/-- Theorem: The maximum number of non-overlapping 2x2 squares in a 9x9 grid is 16 -/
theorem max_figures_9x9_grid :
  ∀ (g : Grid 9) (f : Figure 2),
    max_figures g f = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_figures_9x9_grid_l2806_280620


namespace NUMINAMATH_CALUDE_no_solution_equation_l2806_280625

theorem no_solution_equation :
  ¬ ∃ (x : ℝ), 6 + 3.5 * x = 2.5 * x - 30 + x := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2806_280625


namespace NUMINAMATH_CALUDE_people_on_stairs_l2806_280660

/-- The number of ways to arrange people on steps. -/
def arrange_people (num_people : ℕ) (num_steps : ℕ) (max_per_step : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements for the given problem. -/
theorem people_on_stairs :
  arrange_people 4 7 3 = 2394 := by
  sorry

end NUMINAMATH_CALUDE_people_on_stairs_l2806_280660


namespace NUMINAMATH_CALUDE_equation_solution_l2806_280665

theorem equation_solution : ∃! x : ℝ, (x^2 + 2*x + 3) / (x^2 - 1) = x + 3 :=
by
  -- The unique solution is x = 1
  use 1
  constructor
  -- Prove that x = 1 satisfies the equation
  · sorry
  -- Prove that any other solution must equal 1
  · sorry

end NUMINAMATH_CALUDE_equation_solution_l2806_280665


namespace NUMINAMATH_CALUDE_candy_solution_l2806_280690

/-- Represents the candy distribution problem -/
def candy_problem (billy_initial caleb_initial andy_initial father_bought billy_received caleb_received : ℕ) : Prop :=
  let andy_received := father_bought - billy_received - caleb_received
  let billy_final := billy_initial + billy_received
  let caleb_final := caleb_initial + caleb_received
  let andy_final := andy_initial + andy_received
  andy_final - caleb_final = 4

/-- Theorem stating the solution to the candy distribution problem -/
theorem candy_solution :
  candy_problem 6 11 9 36 8 11 := by
  sorry

#check candy_solution

end NUMINAMATH_CALUDE_candy_solution_l2806_280690


namespace NUMINAMATH_CALUDE_line_point_k_value_l2806_280614

/-- A line contains the points (3, 5), (-3, k), and (-9, -2). The value of k is 3/2. -/
theorem line_point_k_value (k : ℚ) : 
  (∃ (m b : ℚ), 5 = m * 3 + b ∧ k = m * (-3) + b ∧ -2 = m * (-9) + b) → k = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_line_point_k_value_l2806_280614


namespace NUMINAMATH_CALUDE_special_collection_books_l2806_280681

/-- The number of books in the special collection at the beginning of the month -/
def initial_books : ℕ := 75

/-- The percentage of loaned books that are returned -/
def return_rate : ℚ := 65 / 100

/-- The number of books in the special collection at the end of the month -/
def final_books : ℕ := 54

/-- The number of books loaned out during the month -/
def loaned_books : ℚ := 60.00000000000001

theorem special_collection_books :
  initial_books = final_books + (loaned_books - loaned_books * return_rate).ceil :=
sorry

end NUMINAMATH_CALUDE_special_collection_books_l2806_280681


namespace NUMINAMATH_CALUDE_arrange_digits_eq_16_l2806_280676

/-- The number of ways to arrange the digits of 47,770 into a 5-digit number not beginning with 0 -/
def arrange_digits : ℕ :=
  let digits : List ℕ := [4, 7, 7, 7, 0]
  let total_digits : ℕ := 5
  let non_zero_digits : ℕ := 4
  let repeated_digit : ℕ := 7
  let repeated_count : ℕ := 3

  /- Number of ways to place 0 in the last 4 positions -/
  let zero_placements : ℕ := total_digits - 1

  /- Number of ways to arrange the remaining digits -/
  let remaining_arrangements : ℕ := Nat.factorial non_zero_digits / Nat.factorial repeated_count

  zero_placements * remaining_arrangements

theorem arrange_digits_eq_16 : arrange_digits = 16 := by
  sorry

end NUMINAMATH_CALUDE_arrange_digits_eq_16_l2806_280676


namespace NUMINAMATH_CALUDE_problem_statement_l2806_280622

theorem problem_statement (a b : ℕ) (m : ℝ) 
  (h1 : a > 1) 
  (h2 : b > 1) 
  (h3 : a * (b + Real.sin m) = b + Real.cos m) : 
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2806_280622


namespace NUMINAMATH_CALUDE_largest_x_floor_ratio_l2806_280635

theorem largest_x_floor_ratio : 
  ∀ x : ℝ, (↑(⌊x⌋) / x = 8 / 9) → x ≤ 63 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_x_floor_ratio_l2806_280635


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2806_280668

/-- A point in the 2D Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry about the y-axis -/
def symmetricAboutYAxis (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = p.y

/-- The theorem stating that if A(2,5) is symmetric with B about the y-axis, then B(-2,5) -/
theorem symmetric_point_coordinates :
  let A : Point := ⟨2, 5⟩
  let B : Point := ⟨-2, 5⟩
  symmetricAboutYAxis A B → B = ⟨-2, 5⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2806_280668


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2806_280645

theorem complex_equation_solution (a : ℝ) : 
  (2 + a * Complex.I) / (1 + Complex.I) = (3 : ℂ) + Complex.I → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2806_280645


namespace NUMINAMATH_CALUDE_milk_composition_equation_l2806_280687

/-- Represents the nutritional composition of a bottle of milk -/
structure MilkComposition where
  protein : ℝ
  fat : ℝ
  carbohydrate : ℝ

/-- The total content of carbohydrates, protein, and fat in grams -/
def total_content : ℝ := 30

/-- Theorem stating the correct equation for the milk composition -/
theorem milk_composition_equation (m : MilkComposition) 
  (h1 : m.carbohydrate = 1.5 * m.protein)
  (h2 : m.carbohydrate + m.protein + m.fat = total_content) :
  (5/2) * m.protein + m.fat = total_content := by
  sorry

end NUMINAMATH_CALUDE_milk_composition_equation_l2806_280687


namespace NUMINAMATH_CALUDE_xy_value_l2806_280664

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^2 + y^2 = 2) (h2 : x^4 + y^4 = 14/8) : x * y = 3 * Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2806_280664


namespace NUMINAMATH_CALUDE_circular_view_not_rectangular_prism_l2806_280646

/-- A geometric body in three-dimensional space. -/
class GeometricBody :=
(has_circular_view : Bool)

/-- A Rectangular Prism is a type of GeometricBody. -/
def RectangularPrism : GeometricBody :=
{ has_circular_view := false }

/-- Theorem: If a geometric body has a circular view from some direction, it cannot be a Rectangular Prism. -/
theorem circular_view_not_rectangular_prism (body : GeometricBody) :
  body.has_circular_view → body ≠ RectangularPrism :=
sorry

end NUMINAMATH_CALUDE_circular_view_not_rectangular_prism_l2806_280646


namespace NUMINAMATH_CALUDE_num_outfits_is_480_l2806_280640

/-- Number of shirts available --/
def num_shirts : ℕ := 8

/-- Number of ties available --/
def num_ties : ℕ := 5

/-- Number of pants available --/
def num_pants : ℕ := 3

/-- Number of belts available --/
def num_belts : ℕ := 4

/-- Number of belts that can be worn with a tie --/
def num_belts_with_tie : ℕ := 2

/-- Calculate the number of different outfits --/
def num_outfits : ℕ :=
  let outfits_without_tie := num_shirts * num_pants * (num_belts + 1)
  let outfits_with_tie := num_shirts * num_pants * num_ties * (num_belts_with_tie + 1)
  outfits_without_tie + outfits_with_tie

/-- Theorem stating that the number of different outfits is 480 --/
theorem num_outfits_is_480 : num_outfits = 480 := by
  sorry

end NUMINAMATH_CALUDE_num_outfits_is_480_l2806_280640


namespace NUMINAMATH_CALUDE_equation_holds_l2806_280607

theorem equation_holds (a b c : ℝ) (h : a^2 + c^2 = 2*b^2) : 
  (a+b)*(a+c) + (c+a)*(c+b) = 2*(b+a)*(b+c) := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l2806_280607


namespace NUMINAMATH_CALUDE_pony_daily_food_cost_l2806_280608

def annual_expenses : ℕ := 15890
def monthly_pasture_rent : ℕ := 500
def weekly_lessons : ℕ := 2
def lesson_cost : ℕ := 60
def months_per_year : ℕ := 12
def weeks_per_year : ℕ := 52
def days_per_year : ℕ := 365

theorem pony_daily_food_cost :
  (annual_expenses - (monthly_pasture_rent * months_per_year + weekly_lessons * lesson_cost * weeks_per_year)) / days_per_year = 10 :=
by sorry

end NUMINAMATH_CALUDE_pony_daily_food_cost_l2806_280608


namespace NUMINAMATH_CALUDE_program_output_l2806_280637

theorem program_output : 
  let initial_value := 2
  let after_multiplication := initial_value * 2
  let final_value := after_multiplication + 6
  final_value = 10 := by sorry

end NUMINAMATH_CALUDE_program_output_l2806_280637


namespace NUMINAMATH_CALUDE_inequality_proof_l2806_280652

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a / d < b / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2806_280652


namespace NUMINAMATH_CALUDE_max_value_expression_l2806_280689

theorem max_value_expression :
  (∃ x : ℝ, |x - 1| - |x + 4| - 5 = 0) ∧
  (∀ x : ℝ, |x - 1| - |x + 4| - 5 ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_max_value_expression_l2806_280689


namespace NUMINAMATH_CALUDE_calculate_expression_l2806_280658

theorem calculate_expression : (3.242 * 12) / 100 = 0.38904 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2806_280658


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l2806_280667

theorem average_of_three_numbers (y : ℝ) : (15 + 24 + y) / 3 = 20 → y = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l2806_280667


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l2806_280605

/-- Given a line in vector form, prove its equivalence to slope-intercept form -/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 1) + (-1 : ℝ) * (y + 1) = 0 ↔ y = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l2806_280605


namespace NUMINAMATH_CALUDE_vector_length_on_number_line_l2806_280634

theorem vector_length_on_number_line : 
  ∀ (A B : ℝ), A = -1 → B = 2 → abs (B - A) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_vector_length_on_number_line_l2806_280634


namespace NUMINAMATH_CALUDE_unattainable_y_value_l2806_280691

theorem unattainable_y_value (x : ℝ) (h : x ≠ -4/3) :
  ¬∃ x, (2 - x) / (3 * x + 4) = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_y_value_l2806_280691


namespace NUMINAMATH_CALUDE_equation_solution_l2806_280696

theorem equation_solution (y : ℝ) : 
  (y^2 - 11*y + 24)/(y - 1) + (4*y^2 + 20*y - 25)/(4*y - 5) = 5 → y = 3 ∨ y = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2806_280696


namespace NUMINAMATH_CALUDE_problem_solution_l2806_280633

theorem problem_solution (x y z : ℤ) 
  (h1 : x + y = 74)
  (h2 : (x + y) + y + z = 164)
  (h3 : z - y = 16) :
  x = 37 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2806_280633


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2806_280679

theorem regular_polygon_sides (exterior_angle : ℝ) : 
  exterior_angle = 45 → (360 : ℝ) / exterior_angle = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2806_280679


namespace NUMINAMATH_CALUDE_f_one_upper_bound_l2806_280632

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - m * x + 5

-- State the theorem
theorem f_one_upper_bound (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ -2 → f m x₁ > f m x₂) →
  f m 1 ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_f_one_upper_bound_l2806_280632


namespace NUMINAMATH_CALUDE_quadrilateral_circumscribed_circle_l2806_280627

/-- The quadrilateral formed by four lines has a circumscribed circle -/
theorem quadrilateral_circumscribed_circle 
  (l₁ : Real → Real → Prop) 
  (l₂ : Real → Real → Real → Prop)
  (l₃ : Real → Real → Prop)
  (l₄ : Real → Real → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ x + 3*y - 15 = 0)
  (h₂ : ∀ x y k, l₂ x y k ↔ k*x - y - 6 = 0)
  (h₃ : ∀ x y, l₃ x y ↔ x + 5*y = 0)
  (h₄ : ∀ x y, l₄ x y ↔ y = 0) :
  ∃ (k : Real) (circle : Real → Real → Prop),
    k = -8/15 ∧
    (∀ x y, circle x y ↔ x^2 + y^2 - 15*x - 159*y = 0) ∧
    (∀ x y, (l₁ x y ∨ l₂ x y k ∨ l₃ x y ∨ l₄ x y) → circle x y) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_circumscribed_circle_l2806_280627


namespace NUMINAMATH_CALUDE_quadratic_expression_equals_64_l2806_280692

theorem quadratic_expression_equals_64 (x : ℝ) : 
  (2*x + 3)^2 + 2*(2*x + 3)*(5 - 2*x) + (5 - 2*x)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equals_64_l2806_280692


namespace NUMINAMATH_CALUDE_t_less_than_p_l2806_280653

theorem t_less_than_p (j p t : ℝ) (h1 : j = 0.75 * p) (h2 : j = 0.8 * t) (h3 : t = 6.25) :
  (p - t) / p = 0.8 := by sorry

end NUMINAMATH_CALUDE_t_less_than_p_l2806_280653


namespace NUMINAMATH_CALUDE_original_number_proof_l2806_280618

theorem original_number_proof : ∃! N : ℤ, 
  (N - 8) % 5 = 4 ∧ 
  (N - 8) % 7 = 4 ∧ 
  (N - 8) % 9 = 4 ∧ 
  N = 326 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2806_280618


namespace NUMINAMATH_CALUDE_train_stations_distance_l2806_280698

/-- The distance between two stations given train meeting points -/
theorem train_stations_distance
  (first_meet_offset : ℝ)  -- Distance from midpoint to first meeting point
  (second_meet_distance : ℝ)  -- Distance from eastern station to second meeting point
  (h1 : first_meet_offset = 10)  -- First meeting 10 km west of midpoint
  (h2 : second_meet_distance = 40)  -- Second meeting 40 km from eastern station
  : ℝ :=
by
  -- The distance between the stations
  let distance : ℝ := 140
  -- Proof goes here
  sorry

#check train_stations_distance

end NUMINAMATH_CALUDE_train_stations_distance_l2806_280698


namespace NUMINAMATH_CALUDE_sin_squared_minus_three_sin_plus_two_range_l2806_280639

theorem sin_squared_minus_three_sin_plus_two_range :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi →
  ∃ y : ℝ, y = Real.sin x ^ 2 - 3 * Real.sin x + 2 ∧ 0 ≤ y ∧ y ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_sin_squared_minus_three_sin_plus_two_range_l2806_280639


namespace NUMINAMATH_CALUDE_x_value_l2806_280624

-- Define the problem statement
theorem x_value (x : ℝ) : x = 70 * (1 + 0.12) → x = 78.4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2806_280624


namespace NUMINAMATH_CALUDE_square_sum_equals_25_l2806_280649

theorem square_sum_equals_25 (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) :
  x^2 + y^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_25_l2806_280649


namespace NUMINAMATH_CALUDE_correct_number_of_plants_l2806_280671

/-- The number of large salads Anna needs -/
def salads_needed : ℕ := 12

/-- The fraction of lettuce that will survive (not lost to insects and rabbits) -/
def survival_rate : ℚ := 1/2

/-- The number of large salads each lettuce plant provides -/
def salads_per_plant : ℕ := 3

/-- The number of lettuce plants Anna should grow -/
def plants_to_grow : ℕ := 8

/-- Theorem stating that the number of plants Anna should grow is correct -/
theorem correct_number_of_plants : 
  plants_to_grow * salads_per_plant * survival_rate ≥ salads_needed := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_plants_l2806_280671


namespace NUMINAMATH_CALUDE_point_location_l2806_280610

theorem point_location (x y : ℝ) (h : (x + y)^2 = x^2 + y^2 - 2) :
  (x * y = -1) ∧ (x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_point_location_l2806_280610


namespace NUMINAMATH_CALUDE_zeros_of_f_shifted_l2806_280675

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem zeros_of_f_shifted (x : ℝ) : 
  f (x - 1) = 0 ↔ x = 0 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_f_shifted_l2806_280675


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l2806_280629

theorem sum_of_squares_and_products (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 48)
  (h5 : x*y + y*z + z*x = 26) : 
  x + y + z = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l2806_280629


namespace NUMINAMATH_CALUDE_octagon_area_l2806_280662

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
theorem octagon_area (r : ℝ) (h : r = 3) : 
  let s := 2 * r * Real.sqrt ((1 - 1 / Real.sqrt 2) / 2)
  let area_triangle := 1 / 2 * s^2 * (1 / Real.sqrt 2)
  8 * area_triangle = 48 * (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_l2806_280662


namespace NUMINAMATH_CALUDE_triangle_shape_l2806_280601

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_shape (t : Triangle) (h : t.a * Real.cos t.A = t.b * Real.cos t.B) :
  (t.A = t.B) ∨ (t.A + t.B = Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l2806_280601


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2806_280642

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1 / x + 4 / y) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2806_280642


namespace NUMINAMATH_CALUDE_fourth_root_of_256000000_l2806_280663

theorem fourth_root_of_256000000 : (256000000 : ℝ) ^ (1/4 : ℝ) = 40 * (10 : ℝ).sqrt := by sorry

end NUMINAMATH_CALUDE_fourth_root_of_256000000_l2806_280663


namespace NUMINAMATH_CALUDE_james_weight_vest_savings_l2806_280628

/-- The amount James saves by assembling his own weight vest -/
theorem james_weight_vest_savings : 
  let weight_vest_cost : ℝ := 250
  let weight_plates_pounds : ℝ := 200
  let weight_plates_cost_per_pound : ℝ := 1.2
  let ready_made_vest_cost : ℝ := 700
  let ready_made_vest_discount : ℝ := 100
  
  let james_vest_cost := weight_vest_cost + weight_plates_pounds * weight_plates_cost_per_pound
  let discounted_ready_made_vest_cost := ready_made_vest_cost - ready_made_vest_discount
  
  discounted_ready_made_vest_cost - james_vest_cost = 110 := by
  sorry

end NUMINAMATH_CALUDE_james_weight_vest_savings_l2806_280628


namespace NUMINAMATH_CALUDE_marias_score_is_correct_score_difference_average_score_correct_l2806_280695

/-- Maria's score in a game, given that it was 50 points more than John's and their average was 112 -/
def marias_score : ℕ := 137

/-- John's score in the game -/
def johns_score : ℕ := marias_score - 50

/-- The average score of Maria and John -/
def average_score : ℕ := 112

theorem marias_score_is_correct : marias_score = 137 := by
  sorry

theorem score_difference : marias_score = johns_score + 50 := by
  sorry

theorem average_score_correct : (marias_score + johns_score) / 2 = average_score := by
  sorry

end NUMINAMATH_CALUDE_marias_score_is_correct_score_difference_average_score_correct_l2806_280695


namespace NUMINAMATH_CALUDE_polynomial_divisibility_and_divisor_l2806_280673

theorem polynomial_divisibility_and_divisor (m : ℤ) : 
  (∀ x : ℝ, (5 * x^2 - 9 * x + m) % (x - 2) = 0) →
  (m = -2 ∧ 2 % |m| = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_and_divisor_l2806_280673


namespace NUMINAMATH_CALUDE_barbed_wire_height_l2806_280630

/-- Calculates the height of a barbed wire fence around a square field. -/
theorem barbed_wire_height 
  (field_area : ℝ) 
  (wire_cost_per_meter : ℝ) 
  (gate_width : ℝ) 
  (num_gates : ℕ) 
  (total_cost : ℝ) 
  (h : field_area = 3136) 
  (h1 : wire_cost_per_meter = 3.5) 
  (h2 : gate_width = 1) 
  (h3 : num_gates = 2) 
  (h4 : total_cost = 2331) : 
  Real.sqrt field_area * 4 - (gate_width * num_gates) * wire_cost_per_meter * 
    (total_cost / (Real.sqrt field_area * 4 - gate_width * num_gates) / wire_cost_per_meter) = 2331 :=
sorry

end NUMINAMATH_CALUDE_barbed_wire_height_l2806_280630


namespace NUMINAMATH_CALUDE_probability_three_red_balls_l2806_280619

/-- The probability of picking 3 red balls from a bag containing 4 red, 5 blue, and 3 green balls -/
theorem probability_three_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) :
  total_balls = red_balls + blue_balls + green_balls →
  red_balls = 4 →
  blue_balls = 5 →
  green_balls = 3 →
  (red_balls : ℚ) / total_balls * (red_balls - 1) / (total_balls - 1) * (red_balls - 2) / (total_balls - 2) = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_red_balls_l2806_280619


namespace NUMINAMATH_CALUDE_sum_of_xy_is_30_l2806_280602

-- Define the matrix
def matrix (x y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, 5, 10],
    ![4, x, y],
    ![4, y, x]]

-- State the theorem
theorem sum_of_xy_is_30 (x y : ℝ) (h1 : x ≠ y) (h2 : Matrix.det (matrix x y) = 0) :
  x + y = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_is_30_l2806_280602


namespace NUMINAMATH_CALUDE_circus_ticket_ratio_l2806_280678

theorem circus_ticket_ratio : 
  ∀ (num_kids num_adults : ℕ) 
    (total_cost kid_ticket_cost : ℚ),
  num_kids = 6 →
  num_adults = 2 →
  total_cost = 50 →
  kid_ticket_cost = 5 →
  (kid_ticket_cost / ((total_cost - num_kids * kid_ticket_cost) / num_adults)) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_circus_ticket_ratio_l2806_280678


namespace NUMINAMATH_CALUDE_housing_units_with_vcr_l2806_280677

theorem housing_units_with_vcr (H : ℝ) (H_pos : H > 0) : 
  let cable_tv := (1 / 5 : ℝ) * H
  let vcr := F * H
  let both := (1 / 4 : ℝ) * cable_tv
  let neither := (3 / 4 : ℝ) * H
  ∃ F : ℝ, F = (1 / 10 : ℝ) ∧ cable_tv + vcr - both = H - neither :=
by sorry

end NUMINAMATH_CALUDE_housing_units_with_vcr_l2806_280677


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l2806_280655

theorem sum_of_three_consecutive_cubes_divisible_by_nine (a : ℤ) :
  ∃ k : ℤ, a^3 + (a+1)^3 + (a+2)^3 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l2806_280655


namespace NUMINAMATH_CALUDE_two_propositions_have_true_converse_l2806_280617

-- Define the propositions
def proposition1 := "Vertical angles are equal"
def proposition2 := "Supplementary angles of the same side are complementary, and two lines are parallel"
def proposition3 := "Corresponding angles of congruent triangles are equal"
def proposition4 := "If the squares of two real numbers are equal, then the two real numbers are equal"

-- Define a function to check if a proposition has a true converse
def hasValidConverse (p : String) : Bool :=
  match p with
  | "Vertical angles are equal" => false
  | "Supplementary angles of the same side are complementary, and two lines are parallel" => true
  | "Corresponding angles of congruent triangles are equal" => false
  | "If the squares of two real numbers are equal, then the two real numbers are equal" => true
  | _ => false

-- Theorem statement
theorem two_propositions_have_true_converse :
  (hasValidConverse proposition1).toNat +
  (hasValidConverse proposition2).toNat +
  (hasValidConverse proposition3).toNat +
  (hasValidConverse proposition4).toNat = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_propositions_have_true_converse_l2806_280617


namespace NUMINAMATH_CALUDE_negation_equivalence_l2806_280604

universe u

-- Define the universe of discourse
variable {Person : Type u}

-- Define predicates
variable (Teacher : Person → Prop)
variable (ExcellentInMath : Person → Prop)
variable (PoorInMath : Person → Prop)

-- Define the theorem
theorem negation_equivalence :
  (∃ x, Teacher x ∧ PoorInMath x) ↔ ¬(∀ x, Teacher x → ExcellentInMath x) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2806_280604


namespace NUMINAMATH_CALUDE_positive_X_value_l2806_280621

-- Define the # relation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- State the theorem
theorem positive_X_value (X : ℝ) (h1 : hash X 7 = 250) (h2 : X > 0) : X = Real.sqrt 201 := by
  sorry

end NUMINAMATH_CALUDE_positive_X_value_l2806_280621


namespace NUMINAMATH_CALUDE_alternating_seating_card_sum_l2806_280656

theorem alternating_seating_card_sum (n : ℕ) (h : n ≥ 3) :
  ∃ (m : ℕ),
    (∀ (i : ℕ) (hi : i ≤ n), ∃ (b : ℕ), b ≤ n ∧ b = i) ∧  -- Boys' cards
    (∀ (j : ℕ) (hj : n < j ∧ j ≤ 2*n), ∃ (g : ℕ), n < g ∧ g ≤ 2*n ∧ g = j) ∧  -- Girls' cards
    (∀ (k : ℕ) (hk : k ≤ n),
      ∃ (b g₁ g₂ : ℕ),
        b ≤ n ∧ n < g₁ ∧ g₁ ≤ 2*n ∧ n < g₂ ∧ g₂ ≤ 2*n ∧
        b + g₁ + g₂ = m) ↔
  Odd n :=
by sorry

end NUMINAMATH_CALUDE_alternating_seating_card_sum_l2806_280656


namespace NUMINAMATH_CALUDE_system_solution_l2806_280648

theorem system_solution : ∃ (x y z : ℝ), 
  (x + y = 1) ∧ (y + z = 2) ∧ (z + x = 3) ∧ (x = 1) ∧ (y = 0) ∧ (z = 2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2806_280648


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l2806_280600

theorem modulo_eleven_residue : (178 + 4 * 28 + 8 * 62 + 3 * 21) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l2806_280600


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2806_280612

theorem inequality_equivalence (y : ℝ) : 
  3/20 + |y - 1/5| < 1/4 ↔ y ∈ Set.Ioo (1/10 : ℝ) (3/10 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2806_280612


namespace NUMINAMATH_CALUDE_distance_to_point_l2806_280644

theorem distance_to_point : ∀ (x y : ℝ), x = 7 ∧ y = -24 →
  Real.sqrt (x^2 + y^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_l2806_280644


namespace NUMINAMATH_CALUDE_forces_form_hyperboloid_rulings_l2806_280641

-- Define a 3D vector
def Vector3D := ℝ × ℝ × ℝ

-- Define a line in 3D space
structure Line3D where
  point : Vector3D
  direction : Vector3D

-- Define a force as a line with a magnitude
structure Force where
  line : Line3D
  magnitude : ℝ

-- Define the concept of equilibrium
def is_equilibrium (forces : List Force) : Prop := sorry

-- Define the concept of non-coplanarity
def are_non_coplanar (lines : List Line3D) : Prop := sorry

-- Define the concept of a hyperboloid
def is_hyperboloid_ruling (lines : List Line3D) : Prop := sorry

-- The main theorem
theorem forces_form_hyperboloid_rulings 
  (forces : List Force) 
  (h_count : forces.length = 4)
  (h_equilibrium : is_equilibrium forces)
  (h_non_coplanar : are_non_coplanar (forces.map Force.line)) :
  is_hyperboloid_ruling (forces.map Force.line) := by sorry

end NUMINAMATH_CALUDE_forces_form_hyperboloid_rulings_l2806_280641


namespace NUMINAMATH_CALUDE_onion_piece_per_student_l2806_280631

/-- Represents the pizza distribution problem --/
structure PizzaDistribution where
  students : ℕ
  pizzas : ℕ
  slices_per_pizza : ℕ
  cheese_per_student : ℕ
  leftover_cheese : ℕ
  leftover_onion : ℕ

/-- Calculates the number of onion pieces per student --/
def onion_per_student (pd : PizzaDistribution) : ℕ :=
  let total_slices := pd.pizzas * pd.slices_per_pizza
  let total_cheese := pd.students * pd.cheese_per_student
  let used_slices := total_slices - pd.leftover_cheese - pd.leftover_onion
  let onion_slices := used_slices - total_cheese
  onion_slices / pd.students

/-- Theorem stating that each student gets 1 piece of onion pizza --/
theorem onion_piece_per_student (pd : PizzaDistribution) 
  (h1 : pd.students = 32)
  (h2 : pd.pizzas = 6)
  (h3 : pd.slices_per_pizza = 18)
  (h4 : pd.cheese_per_student = 2)
  (h5 : pd.leftover_cheese = 8)
  (h6 : pd.leftover_onion = 4) :
  onion_per_student pd = 1 := by
  sorry

end NUMINAMATH_CALUDE_onion_piece_per_student_l2806_280631


namespace NUMINAMATH_CALUDE_equation_solutions_l2806_280666

theorem equation_solutions :
  (∀ x : ℚ, x + 1/4 = 7/4 → x = 3/2) ∧
  (∀ x : ℚ, 2/3 + x = 3/4 → x = 1/12) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2806_280666


namespace NUMINAMATH_CALUDE_league_teams_count_l2806_280659

theorem league_teams_count (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_league_teams_count_l2806_280659


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2806_280686

-- Define the sets
def U : Set ℕ := Set.univ
def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 2, 3, 6, 8}

-- State the theorem
theorem intersection_with_complement : A ∩ (U \ B) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2806_280686


namespace NUMINAMATH_CALUDE_pyramid_surface_area_change_l2806_280670

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  a : ℝ  -- length
  b : ℝ  -- width
  c : ℝ  -- height

/-- Represents a quadrilateral pyramid -/
structure QuadPyramid where
  base : Point3D  -- center of the base
  apex : Point3D

/-- Calculates the surface area of a quadrilateral pyramid -/
def surfaceArea (p : Parallelepiped) (q : QuadPyramid) : ℝ := sorry

/-- Position of the apex on L₂ -/
inductive ApexPosition
  | Midpoint
  | Between
  | Vertex

/-- Theorem about the surface area of the pyramid -/
theorem pyramid_surface_area_change
  (p : Parallelepiped)
  (q : QuadPyramid)
  (h₁ : q.base.z = 0)  -- base is on the xy-plane
  (h₂ : q.apex.z = p.c)  -- apex is on the top face
  :
  (∀ (pos₁ pos₂ : ApexPosition),
    pos₁ = ApexPosition.Midpoint ∧ pos₂ = ApexPosition.Between →
      surfaceArea p q < surfaceArea p { q with apex := sorry }) ∧
  (∀ (pos₁ pos₂ : ApexPosition),
    pos₁ = ApexPosition.Between ∧ pos₂ = ApexPosition.Vertex →
      surfaceArea p q < surfaceArea p { q with apex := sorry }) ∧
  (∀ (pos : ApexPosition),
    pos = ApexPosition.Vertex →
      ∀ (q' : QuadPyramid), surfaceArea p q ≤ surfaceArea p q') :=
sorry

end NUMINAMATH_CALUDE_pyramid_surface_area_change_l2806_280670


namespace NUMINAMATH_CALUDE_min_triangle_area_l2806_280609

/-- The minimum area of a triangle with vertices A(0,0), B(30,10), and C(p,q) where p and q are integers -/
theorem min_triangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 10)
  ∃ (min_area : ℝ), min_area = 5/2 ∧ 
    ∀ (p q : ℤ), 
      let C : ℝ × ℝ := (p, q)
      let area := (1/2) * |(-p : ℝ) + 3*q|
      area ≥ min_area :=
by sorry

end NUMINAMATH_CALUDE_min_triangle_area_l2806_280609


namespace NUMINAMATH_CALUDE_function_equality_l2806_280626

theorem function_equality (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2) : 
  ∀ x, f x = (x + 1)^2 := by
sorry

end NUMINAMATH_CALUDE_function_equality_l2806_280626


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2806_280654

theorem isosceles_triangle_perimeter (base height : ℝ) (h1 : base = 10) (h2 : height = 6) :
  let side := Real.sqrt (height ^ 2 + (base / 2) ^ 2)
  2 * side + base = 2 * Real.sqrt 61 + 10 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2806_280654


namespace NUMINAMATH_CALUDE_loss_of_30_notation_l2806_280643

def profit_notation (amount : ℤ) : ℤ := amount

def loss_notation (amount : ℤ) : ℤ := -amount

theorem loss_of_30_notation :
  profit_notation 20 = 20 →
  loss_notation 30 = -30 :=
by
  sorry

end NUMINAMATH_CALUDE_loss_of_30_notation_l2806_280643


namespace NUMINAMATH_CALUDE_complex_power_eight_l2806_280616

theorem complex_power_eight : (Complex.I + 1 : ℂ) ^ 8 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_eight_l2806_280616


namespace NUMINAMATH_CALUDE_ball_max_height_l2806_280657

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 45

/-- Theorem stating the maximum height reached by the ball -/
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 69.5 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l2806_280657


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2806_280684

theorem negative_fraction_comparison : -4/3 < -5/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2806_280684


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l2806_280672

theorem digit_sum_puzzle (a b : ℕ) : 
  a ∈ (Set.Icc 1 9) → 
  b ∈ (Set.Icc 1 9) → 
  82 * 10 * a + 7 + 6 * b = 190 → 
  a + 2 * b = 7 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l2806_280672


namespace NUMINAMATH_CALUDE_cannot_determine_charles_loss_l2806_280682

def willie_initial : ℕ := 48
def charles_initial : ℕ := 14
def willie_future : ℕ := 13

theorem cannot_determine_charles_loss :
  ∀ (charles_loss : ℕ),
  ∃ (willie_loss : ℕ),
  willie_initial - willie_loss = willie_future ∧
  charles_initial ≥ charles_loss ∧
  ∃ (charles_loss' : ℕ),
  charles_loss' ≠ charles_loss ∧
  charles_initial ≥ charles_loss' ∧
  willie_initial - willie_loss = willie_future :=
by sorry

end NUMINAMATH_CALUDE_cannot_determine_charles_loss_l2806_280682


namespace NUMINAMATH_CALUDE_cupcakes_sold_katie_sold_20_l2806_280669

/-- Represents the cupcake sale problem -/
def cupcake_sale (initial : ℕ) (additional : ℕ) (final : ℕ) : ℕ :=
  initial + additional - final

/-- Theorem: The number of cupcakes sold is equal to the total made minus the final number -/
theorem cupcakes_sold (initial additional final : ℕ) :
  cupcake_sale initial additional final = (initial + additional) - final :=
by
  sorry

/-- Corollary: In Katie's specific case, she sold 20 cupcakes -/
theorem katie_sold_20 :
  cupcake_sale 26 20 26 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcakes_sold_katie_sold_20_l2806_280669


namespace NUMINAMATH_CALUDE_equation_one_integral_root_l2806_280693

/-- The equation has exactly one integral root -/
theorem equation_one_integral_root :
  ∃! (x : ℤ), x - 8 / (x - 3) = 2 - 8 / (x - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_one_integral_root_l2806_280693


namespace NUMINAMATH_CALUDE_colony_leadership_arrangements_l2806_280688

def colony_size : ℕ := 12
def num_deputies : ℕ := 2
def subordinates_per_deputy : ℕ := 3

def leadership_arrangements : ℕ :=
  colony_size *
  (colony_size - 1) *
  (colony_size - 2) *
  (Nat.choose (colony_size - num_deputies - 1) subordinates_per_deputy) *
  (Nat.choose (colony_size - num_deputies - 1 - subordinates_per_deputy) subordinates_per_deputy)

theorem colony_leadership_arrangements :
  leadership_arrangements = 2209600 :=
by sorry

end NUMINAMATH_CALUDE_colony_leadership_arrangements_l2806_280688


namespace NUMINAMATH_CALUDE_line_slope_l2806_280680

/-- The slope of the line defined by the equation x/4 + y/3 = 1 is -3/4 -/
theorem line_slope (x y : ℝ) : 
  (x / 4 + y / 3 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l2806_280680


namespace NUMINAMATH_CALUDE_probability_two_red_one_green_l2806_280694

def total_marbles : ℕ := 4 + 5 + 3 + 2

def red_marbles : ℕ := 4
def green_marbles : ℕ := 5

def marbles_drawn : ℕ := 3

theorem probability_two_red_one_green :
  (Nat.choose red_marbles 2 * Nat.choose green_marbles 1) / Nat.choose total_marbles marbles_drawn = 15 / 182 :=
sorry

end NUMINAMATH_CALUDE_probability_two_red_one_green_l2806_280694


namespace NUMINAMATH_CALUDE_tournament_rounds_l2806_280697

/-- Represents a table tennis tournament with the given rules --/
structure TableTennisTournament where
  players : ℕ
  champion_losses : ℕ

/-- Calculates the number of rounds in the tournament --/
def rounds (t : TableTennisTournament) : ℕ :=
  2 * (t.players - 1) + t.champion_losses

/-- Theorem stating that a tournament with 15 players and a champion who lost once has 29 rounds --/
theorem tournament_rounds :
  ∀ t : TableTennisTournament,
    t.players = 15 →
    t.champion_losses = 1 →
    rounds t = 29 :=
by
  sorry

#check tournament_rounds

end NUMINAMATH_CALUDE_tournament_rounds_l2806_280697


namespace NUMINAMATH_CALUDE_point_above_line_t_range_l2806_280613

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define what it means for a point to be above the line
def above_line (x y : ℝ) : Prop := x - 2*y + 4 < 0

-- Theorem statement
theorem point_above_line_t_range :
  ∀ t : ℝ, above_line (-2) t → t > 1 :=
by sorry

end NUMINAMATH_CALUDE_point_above_line_t_range_l2806_280613


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2806_280685

theorem complex_number_in_second_quadrant :
  let z : ℂ := (2 * Complex.I) / (2 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2806_280685


namespace NUMINAMATH_CALUDE_george_movie_cost_l2806_280638

/-- The total cost of George's visit to the movie theater -/
def total_cost (ticket_price : ℝ) (nachos_price : ℝ) : ℝ :=
  ticket_price + nachos_price

/-- Theorem: George's total cost for the movie theater visit is $24 -/
theorem george_movie_cost :
  ∀ (ticket_price : ℝ) (nachos_price : ℝ),
    ticket_price = 16 →
    nachos_price = ticket_price / 2 →
    total_cost ticket_price nachos_price = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_george_movie_cost_l2806_280638


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l2806_280647

/-- Given a parabola with equation x = 4y², prove that its latus rectum has the equation x = -1/16 -/
theorem latus_rectum_of_parabola (y : ℝ) :
  let x := 4 * y^2
  (∃ p : ℝ, p = 1/8 ∧ x = -p) → x = -1/16 := by
  sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l2806_280647


namespace NUMINAMATH_CALUDE_johns_order_cost_l2806_280651

/-- The total cost of John's food order for a massive restaurant. -/
def total_cost (beef_amount : ℕ) (beef_price : ℕ) (chicken_amount_multiplier : ℕ) (chicken_price : ℕ) : ℕ :=
  beef_amount * beef_price + (beef_amount * chicken_amount_multiplier) * chicken_price

/-- Proof that John's total food order cost is $14000. -/
theorem johns_order_cost :
  total_cost 1000 8 2 3 = 14000 :=
by sorry

end NUMINAMATH_CALUDE_johns_order_cost_l2806_280651


namespace NUMINAMATH_CALUDE_range_of_a_l2806_280661

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 4*x + a ≥ -2 * x^2 + 1) ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2806_280661


namespace NUMINAMATH_CALUDE_triangle_area_l2806_280606

/-- The area of a triangle with vertices at (2,1,0), (3,3,2), and (5,8,1) is √170/2 -/
theorem triangle_area : ℝ := by
  -- Define the vertices of the triangle
  let a : Fin 3 → ℝ := ![2, 1, 0]
  let b : Fin 3 → ℝ := ![3, 3, 2]
  let c : Fin 3 → ℝ := ![5, 8, 1]

  -- Calculate the area using the cross product method
  let area := (1/2 : ℝ) * Real.sqrt ((b 0 - a 0) * (c 1 - a 1) - (b 1 - a 1) * (c 0 - a 0))^2 +
                                    ((b 1 - a 1) * (c 2 - a 2) - (b 2 - a 2) * (c 1 - a 1))^2 +
                                    ((b 2 - a 2) * (c 0 - a 0) - (b 0 - a 0) * (c 2 - a 2))^2

  -- Prove that the calculated area equals √170/2
  have : area = Real.sqrt 170 / 2 := by sorry

  exact area

end NUMINAMATH_CALUDE_triangle_area_l2806_280606


namespace NUMINAMATH_CALUDE_contained_circle_radius_l2806_280636

/-- An isosceles trapezoid with specific dimensions -/
structure IsoscelesTrapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  isIsosceles : BC = DA
  dimensionsGiven : AB = 6 ∧ BC = 5 ∧ CD = 4

/-- Circles centered at the vertices of the trapezoid -/
structure VertexCircles where
  radiusAB : ℝ
  radiusCD : ℝ
  radiusGiven : radiusAB = 3 ∧ radiusCD = 2

/-- A circle contained within and tangent to all vertex circles -/
structure ContainedCircle where
  radius : ℝ
  isTangent : True  -- Placeholder for tangency condition

/-- The main theorem -/
theorem contained_circle_radius 
  (t : IsoscelesTrapezoid) 
  (v : VertexCircles) 
  (c : ContainedCircle) : 
  c.radius = (-60 + 48 * Real.sqrt 3) / 23 :=
sorry

end NUMINAMATH_CALUDE_contained_circle_radius_l2806_280636


namespace NUMINAMATH_CALUDE_expression_evaluation_l2806_280699

theorem expression_evaluation (a : ℝ) (h : a = 2023) : 
  ((a + 1) / (a - 1) + 1) / (2 * a / (a^2 - 1)) = 2024 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2806_280699


namespace NUMINAMATH_CALUDE_cost_price_is_4_l2806_280603

/-- The cost price of a pen in yuan. -/
def cost_price : ℝ := 4

/-- The retail price of a pen in the first scenario. -/
def retail_price1 : ℝ := 7

/-- The retail price of a pen in the second scenario. -/
def retail_price2 : ℝ := 8

/-- The number of pens sold in the first scenario. -/
def num_pens1 : ℕ := 20

/-- The number of pens sold in the second scenario. -/
def num_pens2 : ℕ := 15

theorem cost_price_is_4 : 
  num_pens1 * (retail_price1 - cost_price) = num_pens2 * (retail_price2 - cost_price) → 
  cost_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_4_l2806_280603


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l2806_280611

/-- The number of cups of flour Mary has already put in -/
def flour_already_added : ℕ := 2

/-- The number of cups of flour Mary needs to add -/
def flour_to_add : ℕ := 5

/-- The total number of cups of flour in the recipe -/
def total_flour : ℕ := flour_already_added + flour_to_add

theorem recipe_flour_amount : total_flour = 7 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l2806_280611


namespace NUMINAMATH_CALUDE_abs_neg_three_equals_three_l2806_280623

theorem abs_neg_three_equals_three : abs (-3 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_equals_three_l2806_280623


namespace NUMINAMATH_CALUDE_remaining_files_indeterminate_l2806_280650

/-- Represents the state of Dave's phone -/
structure PhoneState where
  apps : ℕ
  files : ℕ

/-- Represents the change in Dave's phone state -/
structure PhoneStateChange where
  initialState : PhoneState
  finalState : PhoneState
  appsDeleted : ℕ

/-- Predicate to check if a PhoneStateChange is valid according to the problem conditions -/
def isValidPhoneStateChange (change : PhoneStateChange) : Prop :=
  change.initialState.apps = 16 ∧
  change.initialState.files = 77 ∧
  change.finalState.apps = 5 ∧
  change.appsDeleted = 11 ∧
  change.initialState.apps - change.appsDeleted = change.finalState.apps ∧
  change.finalState.files ≤ change.initialState.files

/-- Theorem stating that the number of remaining files cannot be uniquely determined -/
theorem remaining_files_indeterminate (change : PhoneStateChange) 
  (h : isValidPhoneStateChange change) :
  ∃ (x y : ℕ), x ≠ y ∧ 
    isValidPhoneStateChange { change with finalState := { change.finalState with files := x } } ∧
    isValidPhoneStateChange { change with finalState := { change.finalState with files := y } } :=
  sorry

end NUMINAMATH_CALUDE_remaining_files_indeterminate_l2806_280650


namespace NUMINAMATH_CALUDE_problem_statement_l2806_280615

open Set Real

def M (a : ℝ) : Set ℝ := {x | (x + a) * (x - 1) ≤ 0}
def N : Set ℝ := {x | 4 * x^2 - 4 * x - 3 < 0}

theorem problem_statement (a : ℝ) (h : a > 0) :
  (M a ∪ N = Icc (-2) (3/2) → a = 2) ∧
  (N ∪ (univ \ M a) = univ → 0 < a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l2806_280615
