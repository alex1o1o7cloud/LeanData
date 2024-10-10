import Mathlib

namespace two_digit_numbers_with_specific_remainders_l953_95314

theorem two_digit_numbers_with_specific_remainders :
  let S := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 4 = 3 ∧ n % 3 = 2}
  S = {11, 23, 35, 47, 59, 71, 83, 95} := by
  sorry

end two_digit_numbers_with_specific_remainders_l953_95314


namespace victoria_rice_packets_l953_95393

def rice_packets (initial_balance : ℕ) (rice_cost : ℕ) (wheat_flour_packets : ℕ) (wheat_flour_cost : ℕ) (soda_cost : ℕ) (remaining_balance : ℕ) : ℕ :=
  (initial_balance - (wheat_flour_packets * wheat_flour_cost + soda_cost + remaining_balance)) / rice_cost

theorem victoria_rice_packets :
  rice_packets 500 20 3 25 150 235 = 2 := by
sorry

end victoria_rice_packets_l953_95393


namespace median_sum_bounds_l953_95341

/-- The sum of the medians of a triangle is less than its perimeter and greater than its semiperimeter -/
theorem median_sum_bounds (a b c ma mb mc : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hma : ma = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2))
  (hmb : mb = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2))
  (hmc : mc = (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)) :
  (a + b + c) / 2 < ma + mb + mc ∧ ma + mb + mc < a + b + c := by
  sorry

end median_sum_bounds_l953_95341


namespace soccer_league_games_l953_95338

/-- The number of games played in a soccer league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a league of 12 teams, where each team plays 4 games with every other team,
    the total number of games played is 264. -/
theorem soccer_league_games :
  total_games 12 4 = 264 := by
  sorry

end soccer_league_games_l953_95338


namespace negation_of_universal_absolute_value_l953_95357

theorem negation_of_universal_absolute_value :
  (¬ ∀ x : ℝ, x = |x|) ↔ (∃ x : ℝ, x ≠ |x|) := by
  sorry

end negation_of_universal_absolute_value_l953_95357


namespace compound_interest_calculation_l953_95318

/-- Given a principal amount where the simple interest for 2 years at 5% per annum is $50,
    prove that the compound interest for the same principal, rate, and time is $51.25. -/
theorem compound_interest_calculation (P : ℝ) : 
  P * 0.05 * 2 = 50 → P * (1 + 0.05)^2 - P = 51.25 := by sorry

end compound_interest_calculation_l953_95318


namespace project_completion_time_l953_95374

/-- The time (in days) it takes for person A to complete the project alone -/
def time_A : ℝ := 20

/-- The time (in days) it takes for person B to complete the project alone -/
def time_B : ℝ := 30

/-- The number of days before project completion that A quits -/
def quit_time : ℝ := 15

/-- The total time to complete the project when A and B work together, with A quitting early -/
def total_time : ℝ := 36

theorem project_completion_time :
  (1 / time_A + 1 / time_B) * (total_time - quit_time) + (1 / time_B) * quit_time = 1 :=
sorry

end project_completion_time_l953_95374


namespace fraction_equals_zero_l953_95352

theorem fraction_equals_zero (x : ℝ) (h : x = 1) : (2 * x - 2) / (x - 2) = 0 := by
  sorry

end fraction_equals_zero_l953_95352


namespace existence_of_integer_representation_l953_95356

theorem existence_of_integer_representation (n : ℤ) :
  ∃ (a b : ℤ), n = ⌊(a : ℝ) * Real.sqrt 2⌋ + ⌊(b : ℝ) * Real.sqrt 3⌋ := by
  sorry

end existence_of_integer_representation_l953_95356


namespace luis_task_completion_l953_95394

-- Define the start time and end time of the third task
def start_time : Nat := 9 * 60  -- 9:00 AM in minutes
def end_third_task : Nat := 12 * 60 + 30  -- 12:30 PM in minutes

-- Define the number of tasks
def num_tasks : Nat := 4

-- Define the theorem
theorem luis_task_completion :
  ∀ (task_duration : Nat),
  (end_third_task - start_time = 3 * task_duration) →
  (start_time + num_tasks * task_duration = 13 * 60 + 40) :=
by
  sorry


end luis_task_completion_l953_95394


namespace negative_a_cubed_div_squared_l953_95375

theorem negative_a_cubed_div_squared (a : ℝ) : (-a)^3 / (-a)^2 = -a := by sorry

end negative_a_cubed_div_squared_l953_95375


namespace vlad_sister_height_difference_l953_95346

/-- Converts feet and inches to total inches -/
def height_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Calculates the height difference in inches between two people -/
def height_difference (height1_feet height1_inches height2_feet height2_inches : ℕ) : ℕ :=
  (height_to_inches height1_feet height1_inches) - (height_to_inches height2_feet height2_inches)

theorem vlad_sister_height_difference :
  height_difference 6 3 2 10 = 41 := by
  sorry

end vlad_sister_height_difference_l953_95346


namespace congruent_side_length_l953_95350

-- Define the triangle
structure IsoscelesTriangle where
  base : ℝ
  area : ℝ
  side : ℝ

-- Define our specific triangle
def ourTriangle : IsoscelesTriangle where
  base := 24
  area := 60
  side := 13

-- Theorem statement
theorem congruent_side_length (t : IsoscelesTriangle) 
  (h1 : t.base = 24) 
  (h2 : t.area = 60) : 
  t.side = 13 := by
  sorry

#check congruent_side_length

end congruent_side_length_l953_95350


namespace line_intersection_x_axis_l953_95359

/-- A line with slope 3/4 passing through (-12, -39) intersects the x-axis at x = 40 -/
theorem line_intersection_x_axis :
  ∀ (f : ℝ → ℝ),
  (∀ x y, f y - f x = (3/4) * (y - x)) →  -- Slope condition
  f (-12) = -39 →                         -- Point condition
  ∃ x, f x = 0 ∧ x = 40 :=                -- Intersection with x-axis
by
  sorry


end line_intersection_x_axis_l953_95359


namespace isosceles_triangle_from_side_equation_l953_95370

/-- Given a triangle with sides a, b, and c satisfying a² + bc = b² + ac, prove it's isosceles --/
theorem isosceles_triangle_from_side_equation 
  (a b c : ℝ) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_equation : a^2 + b*c = b^2 + a*c) : 
  a = b :=
sorry

end isosceles_triangle_from_side_equation_l953_95370


namespace absolute_value_sum_zero_l953_95389

theorem absolute_value_sum_zero (a b : ℝ) :
  |a - 3| + |b + 6| = 0 → (a + b - 2 = -5 ∧ a - b - 2 = 7) := by
  sorry

end absolute_value_sum_zero_l953_95389


namespace consecutive_even_numbers_sum_l953_95330

theorem consecutive_even_numbers_sum (x : ℤ) : 
  (Even x) → (x + 2)^2 - x^2 = 84 → x + (x + 2) = 42 := by
  sorry

end consecutive_even_numbers_sum_l953_95330


namespace xiaojuan_savings_l953_95301

/-- Xiaojuan's original savings in yuan -/
def original_savings : ℝ := 12.4

/-- Amount Xiaojuan's mother gave her in yuan -/
def mother_gift : ℝ := 5

/-- Amount spent on dictionary in addition to half of mother's gift -/
def extra_dictionary_cost : ℝ := 0.4

/-- Amount left after all purchases -/
def remaining_amount : ℝ := 5.2

theorem xiaojuan_savings :
  original_savings / 2 + (mother_gift / 2 + extra_dictionary_cost) + remaining_amount = mother_gift + original_savings := by
  sorry

#check xiaojuan_savings

end xiaojuan_savings_l953_95301


namespace multiple_of_all_positive_integers_l953_95378

theorem multiple_of_all_positive_integers (n : ℤ) : 
  (∀ m : ℕ+, ∃ k : ℤ, n = k * m) ↔ n = 0 := by
  sorry

end multiple_of_all_positive_integers_l953_95378


namespace tire_price_proof_l953_95340

/-- The regular price of a single tire -/
def regular_price : ℝ := 104.17

/-- The discounted price of three tires -/
def discounted_price (p : ℝ) : ℝ := 3 * (0.8 * p)

/-- The price of the fourth tire -/
def fourth_tire_price : ℝ := 5

/-- The total price paid for four tires -/
def total_price : ℝ := 255

/-- Theorem stating that the regular price of a tire is approximately 104.17 dollars 
    given the discount and total price conditions -/
theorem tire_price_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  discounted_price regular_price + fourth_tire_price = total_price - ε :=
sorry

end tire_price_proof_l953_95340


namespace rectangular_field_shortcut_l953_95382

theorem rectangular_field_shortcut (x y : ℝ) (hxy : 0 < x ∧ x < y) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y → x / y = 5/12 := by
  sorry

end rectangular_field_shortcut_l953_95382


namespace celine_change_l953_95395

/-- The price of a laptop in dollars -/
def laptop_price : ℕ := 600

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 400

/-- The number of laptops Celine buys -/
def laptops_bought : ℕ := 2

/-- The number of smartphones Celine buys -/
def smartphones_bought : ℕ := 4

/-- The amount of money Celine has in dollars -/
def money_available : ℕ := 3000

/-- The change Celine receives after her purchase -/
theorem celine_change : 
  money_available - (laptop_price * laptops_bought + smartphone_price * smartphones_bought) = 200 := by
  sorry

end celine_change_l953_95395


namespace paul_bought_101_books_l953_95360

/-- Calculates the number of books bought given initial and final book counts -/
def books_bought (initial_count final_count : ℕ) : ℕ :=
  final_count - initial_count

/-- Proves that Paul bought 101 books -/
theorem paul_bought_101_books (initial_count final_count : ℕ) 
  (h1 : initial_count = 50)
  (h2 : final_count = 151) :
  books_bought initial_count final_count = 101 := by
  sorry

end paul_bought_101_books_l953_95360


namespace range_of_a_l953_95304

-- Define the set M
def M (a : ℝ) : Set ℝ := {x : ℝ | (a * x - 5) / (x^2 - a) < 0}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (3 ∈ M a) ∧ (5 ∉ M a) ↔ (1 ≤ a ∧ a < 5/3) ∨ (9 < a ∧ a ≤ 25) :=
by sorry

end range_of_a_l953_95304


namespace complex_number_in_second_quadrant_l953_95398

theorem complex_number_in_second_quadrant :
  let z : ℂ := (1 + Complex.I) / (1 - Complex.I)^2
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end complex_number_in_second_quadrant_l953_95398


namespace complement_A_intersect_B_l953_95344

def A : Set ℝ := {x | x^2 + x - 6 < 0}
def B : Set ℝ := {x | x > 1}

theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Ici (2 : ℝ) := by sorry

end complement_A_intersect_B_l953_95344


namespace parabola_line_intersection_l953_95302

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line
def line (x y a : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - a)

-- Define the condition for F being outside the circle with diameter CD
def F_outside_circle (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - 1) * (x₂ - 1) + y₁ * y₂ > 0

theorem parabola_line_intersection (a : ℝ) :
  a < 0 →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line x₁ y₁ a ∧ line x₂ y₂ a ∧
    F_outside_circle x₁ y₁ x₂ y₂) ↔
  -3 < a ∧ a < -2 * Real.sqrt 5 + 3 :=
sorry

end parabola_line_intersection_l953_95302


namespace ellipse_intersection_range_l953_95329

/-- Ellipse C with center at origin, right focus at (√3, 0), and eccentricity √3/2 -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Line l: y = kx + √2 -/
def line_l (k x y : ℝ) : Prop :=
  y = k * x + Real.sqrt 2

/-- Points A and B are distinct intersections of ellipse C and line l -/
def distinct_intersections (k x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
  line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

/-- Dot product of OA and OB is greater than 2 -/
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ > 2

/-- The range of k satisfies the given conditions -/
theorem ellipse_intersection_range :
  ∀ k : ℝ,
    (∀ x₁ y₁ x₂ y₂ : ℝ,
      distinct_intersections k x₁ y₁ x₂ y₂ →
      dot_product_condition x₁ y₁ x₂ y₂) →
    (k ∈ Set.Ioo (-Real.sqrt 3 / 3) (-1/2) ∪ Set.Ioo (1/2) (Real.sqrt 3 / 3)) :=
by sorry

end ellipse_intersection_range_l953_95329


namespace rubys_math_homework_l953_95355

/-- Ruby's math homework problem -/
theorem rubys_math_homework :
  ∀ (ruby_math ruby_reading nina_math nina_reading : ℕ),
  ruby_reading = 2 →
  nina_math = 4 * ruby_math →
  nina_reading = 8 * ruby_reading →
  nina_math + nina_reading = 48 →
  ruby_math = 6 := by
sorry

end rubys_math_homework_l953_95355


namespace cubic_geometric_roots_l953_95327

theorem cubic_geometric_roots (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 + a*x^2 + b*x + c = 0 ∧
    y^3 + a*y^2 + b*y + c = 0 ∧
    z^3 + a*z^2 + b*z + c = 0 ∧
    ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 ∧ y = x*q ∧ z = x*q^2) ↔
  (b^3 = a^3*c ∧
   c ≠ 0 ∧
   ∃ m : ℝ, m^3 = -c ∧ a < m ∧ m < -a/3) :=
by sorry

end cubic_geometric_roots_l953_95327


namespace geometric_sequence_from_formula_l953_95326

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_from_formula (c q : ℝ) (hcq : c * q ≠ 0) :
  is_geometric_sequence (fun n => c * q ^ n) :=
sorry

end geometric_sequence_from_formula_l953_95326


namespace integer_divisibility_in_range_l953_95322

theorem integer_divisibility_in_range (n : ℕ+) : 
  ∃ (a b c : ℤ), 
    (n : ℤ)^2 < a ∧ a < b ∧ b < c ∧ c < (n : ℤ)^2 + n + 3 * Real.sqrt n ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (b * c) % a = 0 := by
  sorry

end integer_divisibility_in_range_l953_95322


namespace triangle_type_indeterminate_l953_95367

theorem triangle_type_indeterminate (A B C : ℝ) 
  (triangle_sum : A + B + C = π) 
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C) 
  (inequality : Real.sin A * Real.sin C > Real.cos A * Real.cos C) : 
  ¬(∀ α : ℝ, (0 < α ∧ α < π) → 
    ((A < π/2 ∧ B < π/2 ∧ C < π/2) ∨ 
     (A = π/2 ∨ B = π/2 ∨ C = π/2) ∨ 
     (A > π/2 ∨ B > π/2 ∨ C > π/2))) :=
by sorry

end triangle_type_indeterminate_l953_95367


namespace product_of_solutions_l953_95315

theorem product_of_solutions (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (abs (5 * x₁) + 4 = abs (40 - 5)) ∧ 
   (abs (5 * x₂) + 4 = abs (40 - 5)) ∧
   x₁ * x₂ = -961 / 25) :=
by sorry

end product_of_solutions_l953_95315


namespace equation_solution_l953_95376

theorem equation_solution (x : ℝ) : (x + 6) / (x - 3) = 4 → x = 6 := by
  sorry

end equation_solution_l953_95376


namespace f_geq_one_for_a_eq_two_g_min_value_g_min_value_exists_l953_95358

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |x + 2/a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x + f a (-x)

-- Theorem for part (1)
theorem f_geq_one_for_a_eq_two (x : ℝ) : f 2 x ≥ 1 := by sorry

-- Theorem for part (2)
theorem g_min_value (a : ℝ) : ∀ x : ℝ, g a x ≥ 4 * Real.sqrt 2 := by sorry

-- Theorem for existence of x that achieves the minimum value
theorem g_min_value_exists (a : ℝ) : ∃ x : ℝ, g a x = 4 * Real.sqrt 2 := by sorry

end

end f_geq_one_for_a_eq_two_g_min_value_g_min_value_exists_l953_95358


namespace percentage_of_boys_from_school_A_l953_95335

theorem percentage_of_boys_from_school_A (total_boys : ℕ) (boys_A_not_science : ℕ) 
  (h1 : total_boys = 450)
  (h2 : boys_A_not_science = 63)
  (h3 : (30 : ℚ) / 100 = 1 - (boys_A_not_science : ℚ) / ((20 : ℚ) / 100 * total_boys)) :
  (20 : ℚ) / 100 = (boys_A_not_science : ℚ) / (0.7 * total_boys) :=
by
  sorry

end percentage_of_boys_from_school_A_l953_95335


namespace gray_area_is_65_l953_95390

/-- Given two overlapping rectangles, calculates the area of the gray part -/
def gray_area (width1 length1 width2 length2 black_area : ℕ) : ℕ :=
  width2 * length2 - (width1 * length1 - black_area)

/-- Theorem stating that the area of the gray part is 65 -/
theorem gray_area_is_65 :
  gray_area 8 10 12 9 37 = 65 := by
  sorry

end gray_area_is_65_l953_95390


namespace average_speed_first_part_is_35_l953_95343

-- Define the total trip duration in hours
def total_trip_duration : ℝ := 24

-- Define the average speed for the entire trip in miles per hour
def average_speed_entire_trip : ℝ := 50

-- Define the duration of the first part of the trip in hours
def first_part_duration : ℝ := 4

-- Define the speed for the remaining part of the trip in miles per hour
def remaining_part_speed : ℝ := 53

-- Define the average speed for the first part of the trip
def average_speed_first_part : ℝ := 35

-- Theorem statement
theorem average_speed_first_part_is_35 :
  total_trip_duration * average_speed_entire_trip =
  first_part_duration * average_speed_first_part +
  (total_trip_duration - first_part_duration) * remaining_part_speed :=
by sorry

end average_speed_first_part_is_35_l953_95343


namespace sqrt_undefined_for_positive_integer_l953_95308

theorem sqrt_undefined_for_positive_integer (x : ℕ+) :
  (¬ ∃ (y : ℝ), y ^ 2 = (x : ℝ) - 3) ↔ (x = 1 ∨ x = 2) := by
  sorry

end sqrt_undefined_for_positive_integer_l953_95308


namespace circus_ticket_sales_l953_95396

theorem circus_ticket_sales (total_tickets : ℕ) (adult_price children_price : ℚ) 
  (total_receipts : ℚ) (h1 : total_tickets = 522) 
  (h2 : adult_price = 15) (h3 : children_price = 8) (h4 : total_receipts = 5086) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * children_price = total_receipts ∧
    adult_tickets = 130 := by
  sorry

end circus_ticket_sales_l953_95396


namespace triangle_solution_l953_95381

theorem triangle_solution (c t r : ℝ) (hc : c = 30) (ht : t = 336) (hr : r = 8) :
  ∃ (a b : ℝ),
    a + b + c = 2 * (t / r) ∧
    t = r * (a + b + c) / 2 ∧
    t^2 = (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c) ∧
    a = 26 ∧
    b = 28 := by
  sorry

end triangle_solution_l953_95381


namespace max_multiplication_table_sum_l953_95380

theorem max_multiplication_table_sum : 
  ∀ (a b c d e f : ℕ), 
    a ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    b ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    c ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    d ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    e ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    f ∈ ({3, 5, 7, 11, 17, 19} : Set ℕ) → 
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → 
    b ≠ c → b ≠ d → b ≠ e → b ≠ f → 
    c ≠ d → c ≠ e → c ≠ f → 
    d ≠ e → d ≠ f → 
    e ≠ f → 
    (a * d + a * e + a * f + b * d + b * e + b * f + c * d + c * e + c * f) ≤ 961 :=
by sorry

end max_multiplication_table_sum_l953_95380


namespace zero_in_interval_l953_95325

def f (x : ℝ) := 2*x + 3*x

theorem zero_in_interval :
  ∃ x ∈ Set.Ioo (-1 : ℝ) 0, f x = 0 :=
by sorry

end zero_in_interval_l953_95325


namespace equipment_production_calculation_l953_95373

/-- Given a total production and a sample with known quantities from two equipment types,
    calculate the total production of the second equipment type. -/
theorem equipment_production_calculation
  (total_production : ℕ) -- Total number of pieces produced
  (sample_size : ℕ) -- Size of the sample
  (sample_a : ℕ) -- Number of pieces from equipment A in the sample
  (h1 : total_production = 4800)
  (h2 : sample_size = 80)
  (h3 : sample_a = 50)
  : ∃ (total_b : ℕ), total_b = 1800 ∧ total_b + (total_production - total_b) = total_production :=
by
  sorry

#check equipment_production_calculation

end equipment_production_calculation_l953_95373


namespace quadratic_rewrite_l953_95349

theorem quadratic_rewrite (b : ℝ) (h1 : b < 0) :
  (∃ m : ℝ, ∀ x : ℝ, x^2 + b*x + 1/4 = (x + m)^2 + 1/6) →
  b = -1 / Real.sqrt 3 := by
sorry

end quadratic_rewrite_l953_95349


namespace abc_theorem_l953_95397

theorem abc_theorem (a b c : ℕ+) (x y z w : ℝ) 
  (h_order : a ≤ b ∧ b ≤ c)
  (h_eq : (a : ℝ) ^ x = (b : ℝ) ^ y ∧ (b : ℝ) ^ y = (c : ℝ) ^ z ∧ (c : ℝ) ^ z = 70 ^ w)
  (h_sum : 1 / x + 1 / y + 1 / z = 1 / w) :
  c = 7 := by
  sorry

end abc_theorem_l953_95397


namespace number_increase_l953_95351

theorem number_increase (n : ℕ) (m : ℕ) (increase : ℕ) : n = 18 → m = 12 → increase = m * n - n := by
  sorry

end number_increase_l953_95351


namespace specific_cube_surface_area_l953_95365

/-- Represents the heights of cuts in the cube -/
structure CutHeights where
  h1 : ℝ
  h2 : ℝ
  h3 : ℝ

/-- Calculates the total surface area of a stacked solid formed from a cube -/
def totalSurfaceArea (cubeSideLength : ℝ) (cuts : CutHeights) : ℝ :=
  sorry

/-- Theorem stating the total surface area of the specific cut and stacked cube -/
theorem specific_cube_surface_area :
  let cubeSideLength : ℝ := 2
  let cuts : CutHeights := { h1 := 1/4, h2 := 1/4 + 1/5, h3 := 1/4 + 1/5 + 1/8 }
  totalSurfaceArea cubeSideLength cuts = 12 := by
  sorry

end specific_cube_surface_area_l953_95365


namespace geometric_progression_sum_l953_95363

theorem geometric_progression_sum (p q : ℝ) : 
  p ≠ q →                  -- Two distinct geometric progressions
  p + q = 3 →              -- Sum of common ratios is 3
  1 * p^5 + 1 * q^5 = 573 →  -- Sum of sixth terms is 573 (1 is the first term)
  1 * p^4 + 1 * q^4 = 161    -- Sum of fifth terms is 161
  := by sorry

end geometric_progression_sum_l953_95363


namespace perpendicular_lines_min_value_l953_95313

theorem perpendicular_lines_min_value (b : ℝ) (a : ℝ) (h1 : b > 1) :
  ((b^2 + 1) * (-1 / a) * (b - 1) = -1) →
  (∀ a' : ℝ, ((b^2 + 1) * (-1 / a') * (b - 1) = -1) → a ≤ a') →
  a = 2 * Real.sqrt 2 + 2 := by
  sorry

end perpendicular_lines_min_value_l953_95313


namespace fishing_line_sections_l953_95353

/-- The number of reels of fishing line John buys -/
def num_reels : ℕ := 3

/-- The length of fishing line in each reel (in meters) -/
def reel_length : ℕ := 100

/-- The length of each section John cuts the fishing line into (in meters) -/
def section_length : ℕ := 10

/-- The total number of sections John gets from cutting all the fishing line -/
def total_sections : ℕ := (num_reels * reel_length) / section_length

theorem fishing_line_sections :
  total_sections = 30 := by sorry

end fishing_line_sections_l953_95353


namespace carnival_spending_theorem_l953_95303

def carnival_spending (total_budget food_cost : ℕ) : ℕ :=
  let ride_cost := 2 * food_cost
  let total_spent := food_cost + ride_cost
  total_budget - total_spent

theorem carnival_spending_theorem :
  carnival_spending 100 20 = 40 :=
by sorry

end carnival_spending_theorem_l953_95303


namespace min_value_abc_l953_95383

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 27 ∧ a₀ + 3 * b₀ + 9 * c₀ = 27 :=
sorry

end min_value_abc_l953_95383


namespace point_is_centroid_l953_95311

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a triangle ABC in a real inner product space, if P is any point in the space and G satisfies PG = 1/3(PA + PB + PC), then G is the centroid of triangle ABC -/
theorem point_is_centroid (A B C P G : V) :
  (G - P) = (1 / 3 : ℝ) • ((A - P) + (B - P) + (C - P)) →
  G = (1 / 3 : ℝ) • (A + B + C) :=
sorry

end point_is_centroid_l953_95311


namespace same_color_probability_l953_95336

/-- The probability of drawing two balls of the same color from a bag containing
    8 green balls, 5 red balls, and 3 blue balls, with replacement. -/
theorem same_color_probability (green red blue : ℕ) (total : ℕ) :
  green = 8 → red = 5 → blue = 3 → total = green + red + blue →
  (green^2 + red^2 + blue^2 : ℚ) / total^2 = 49 / 128 := by
  sorry

end same_color_probability_l953_95336


namespace bell_pepper_pieces_l953_95387

/-- The number of bell peppers Tamia has -/
def num_peppers : ℕ := 5

/-- The number of large slices each pepper is cut into -/
def slices_per_pepper : ℕ := 20

/-- The number of smaller pieces each selected large slice is cut into -/
def pieces_per_slice : ℕ := 3

/-- The total number of bell pepper pieces Tamia will have -/
def total_pieces : ℕ := 
  let total_slices := num_peppers * slices_per_pepper
  let slices_to_cut := total_slices / 2
  let smaller_pieces := slices_to_cut * pieces_per_slice
  let remaining_slices := total_slices - slices_to_cut
  smaller_pieces + remaining_slices

theorem bell_pepper_pieces : total_pieces = 200 := by
  sorry

end bell_pepper_pieces_l953_95387


namespace post_office_problem_l953_95392

/-- Proves that given the conditions from the post office problem, each month has 30 days. -/
theorem post_office_problem (letters_per_day : ℕ) (packages_per_day : ℕ) 
  (total_mail : ℕ) (num_months : ℕ) :
  letters_per_day = 60 →
  packages_per_day = 20 →
  total_mail = 14400 →
  num_months = 6 →
  (total_mail / (letters_per_day + packages_per_day)) / num_months = 30 := by
  sorry

end post_office_problem_l953_95392


namespace todd_ate_cupcakes_l953_95348

def initial_cupcakes : ℕ := 38
def packages : ℕ := 3
def cupcakes_per_package : ℕ := 8

theorem todd_ate_cupcakes : 
  initial_cupcakes - packages * cupcakes_per_package = 14 := by
  sorry

end todd_ate_cupcakes_l953_95348


namespace x_squared_minus_y_equals_three_l953_95384

theorem x_squared_minus_y_equals_three (x y : ℝ) :
  |x + 1| + (2 * x - y)^2 = 0 → x^2 - y = 3 := by
  sorry

end x_squared_minus_y_equals_three_l953_95384


namespace fish_remaining_l953_95364

theorem fish_remaining (initial : ℝ) (given_away : ℝ) (remaining : ℝ) : 
  initial = 47.0 → given_away = 22.0 → remaining = initial - given_away → remaining = 25.0 := by
  sorry

end fish_remaining_l953_95364


namespace simplify_tan_cot_expression_l953_95310

theorem simplify_tan_cot_expression :
  let tan_60 := Real.sqrt 3
  let cot_60 := 1 / Real.sqrt 3
  (tan_60^3 + cot_60^3) / (tan_60 + cot_60) = 7/3 := by
  sorry

end simplify_tan_cot_expression_l953_95310


namespace razorback_tshirt_profit_l953_95321

/-- The Razorback T-shirt Shop problem -/
theorem razorback_tshirt_profit :
  let total_shirts : ℕ := 245
  let total_revenue : ℚ := 2205
  let profit_per_shirt : ℚ := total_revenue / total_shirts
  profit_per_shirt = 9 := by sorry

end razorback_tshirt_profit_l953_95321


namespace b_oxen_count_main_theorem_l953_95312

/-- Represents the number of oxen and months for each person --/
structure Grazing :=
  (oxen : ℕ)
  (months : ℕ)

/-- Calculates the total grazing cost --/
def total_cost (a b c : Grazing) (cost_per_ox_month : ℚ) : ℚ :=
  (a.oxen * a.months + b.oxen * b.months + c.oxen * c.months : ℚ) * cost_per_ox_month

/-- Theorem: Given the conditions, b put 12 oxen for grazing --/
theorem b_oxen_count (total_rent : ℚ) (c_rent : ℚ) : ℕ :=
  let a : Grazing := ⟨10, 7⟩
  let b : Grazing := ⟨12, 5⟩  -- We claim b put 12 oxen
  let c : Grazing := ⟨15, 3⟩
  let cost_per_ox_month : ℚ := c_rent / (c.oxen * c.months)
  have h1 : total_cost a b c cost_per_ox_month = total_rent := by sorry
  have h2 : c.oxen * c.months * cost_per_ox_month = c_rent := by sorry
  b.oxen

/-- The main theorem stating that given the conditions, b put 12 oxen for grazing --/
theorem main_theorem : b_oxen_count 140 36 = 12 := by sorry

end b_oxen_count_main_theorem_l953_95312


namespace age_difference_proof_l953_95331

theorem age_difference_proof (ramesh_age mahesh_age : ℝ) : 
  ramesh_age / mahesh_age = 2 / 5 →
  (ramesh_age + 10) / (mahesh_age + 10) = 10 / 15 →
  mahesh_age - ramesh_age = 7.5 := by
sorry

end age_difference_proof_l953_95331


namespace spades_in_deck_l953_95368

/-- 
Given a deck of 52 cards containing some spades, prove that if the probability 
of not drawing a spade on the first draw is 0.75, then there are 13 spades in the deck.
-/
theorem spades_in_deck (total_cards : ℕ) (prob_not_spade : ℚ) (num_spades : ℕ) : 
  total_cards = 52 →
  prob_not_spade = 3/4 →
  (total_cards - num_spades : ℚ) / total_cards = prob_not_spade →
  num_spades = 13 := by
  sorry

end spades_in_deck_l953_95368


namespace count_eight_digit_cyclic_fixed_points_l953_95305

def is_eight_digit (n : ℕ) : Prop := 10^7 ≤ n ∧ n < 10^8

def last_digit (n : ℕ) : ℕ := n % 10

def cyclic_permutation (n : ℕ) : ℕ :=
  let d := (Nat.log 10 n) + 1
  (n % 10) * 10^(d-1) + n / 10

def iterative_permutation (n : ℕ) (k : ℕ) : ℕ :=
  Nat.iterate cyclic_permutation k n

theorem count_eight_digit_cyclic_fixed_points :
  (∃ (S : Finset ℕ), (∀ a ∈ S, is_eight_digit a ∧ last_digit a ≠ 0 ∧
    iterative_permutation a 4 = a) ∧ S.card = 9^4) := by sorry

end count_eight_digit_cyclic_fixed_points_l953_95305


namespace tank_water_fraction_l953_95371

theorem tank_water_fraction (tank_capacity : ℚ) (initial_fraction : ℚ) (added_water : ℚ) : 
  tank_capacity = 56 →
  initial_fraction = 3/4 →
  added_water = 7 →
  (initial_fraction * tank_capacity + added_water) / tank_capacity = 7/8 := by
sorry

end tank_water_fraction_l953_95371


namespace inscribed_cube_sphere_surface_area_l953_95309

theorem inscribed_cube_sphere_surface_area (cube_surface_area : ℝ) (sphere_surface_area : ℝ) :
  cube_surface_area = 6 →
  ∃ (cube_edge : ℝ) (sphere_radius : ℝ),
    cube_edge > 0 ∧
    sphere_radius > 0 ∧
    cube_surface_area = 6 * cube_edge^2 ∧
    sphere_radius = (cube_edge * Real.sqrt 3) / 2 ∧
    sphere_surface_area = 4 * Real.pi * sphere_radius^2 ∧
    sphere_surface_area = 3 * Real.pi :=
by sorry

end inscribed_cube_sphere_surface_area_l953_95309


namespace square_areas_product_equality_l953_95337

theorem square_areas_product_equality (α : Real) : 
  (Real.cos α)^4 * (Real.sin α)^4 = ((Real.cos α)^2 * (Real.sin α)^2)^2 := by
  sorry

end square_areas_product_equality_l953_95337


namespace tan_thirteen_pi_thirds_l953_95320

theorem tan_thirteen_pi_thirds : Real.tan (13 * Real.pi / 3) = Real.sqrt 3 := by
  sorry

end tan_thirteen_pi_thirds_l953_95320


namespace competition_participants_count_l953_95347

/-- Represents the math competition scenario -/
structure Competition where
  fullScore : ℕ
  initialGoldThreshold : ℕ
  initialSilverLowerThreshold : ℕ
  initialSilverUpperThreshold : ℕ
  changedGoldThreshold : ℕ
  changedSilverLowerThreshold : ℕ
  changedSilverUpperThreshold : ℕ
  initialGoldCount : ℕ
  initialSilverCount : ℕ
  nonMedalCount : ℕ
  changedGoldCount : ℕ
  changedSilverCount : ℕ
  changedGoldAverage : ℕ
  changedSilverAverage : ℕ

/-- The theorem to be proved -/
theorem competition_participants_count (c : Competition) 
  (h1 : c.fullScore = 120)
  (h2 : c.initialGoldThreshold = 100)
  (h3 : c.initialSilverLowerThreshold = 80)
  (h4 : c.initialSilverUpperThreshold = 99)
  (h5 : c.changedGoldThreshold = 90)
  (h6 : c.changedSilverLowerThreshold = 70)
  (h7 : c.changedSilverUpperThreshold = 89)
  (h8 : c.initialSilverCount = c.initialGoldCount + 8)
  (h9 : c.nonMedalCount = c.initialGoldCount + c.initialSilverCount + 9)
  (h10 : c.changedGoldCount = c.initialGoldCount + 5)
  (h11 : c.changedSilverCount = c.initialSilverCount + 5)
  (h12 : c.changedGoldCount * c.changedGoldAverage = c.changedSilverCount * c.changedSilverAverage)
  (h13 : c.changedGoldAverage = 95)
  (h14 : c.changedSilverAverage = 75) :
  c.initialGoldCount + c.initialSilverCount + c.nonMedalCount = 125 :=
sorry


end competition_participants_count_l953_95347


namespace regular_pyramid_volume_l953_95328

theorem regular_pyramid_volume (b : ℝ) (h : b = 2) :
  ∀ V : ℝ, V ≤ (16 * Real.pi) / (9 * Real.sqrt 3) → V < 3.25 := by sorry

end regular_pyramid_volume_l953_95328


namespace petting_zoo_theorem_l953_95332

theorem petting_zoo_theorem (total_animals : ℕ) (carrot_eaters : ℕ) (hay_eaters : ℕ) (both_eaters : ℕ) :
  total_animals = 75 →
  carrot_eaters = 26 →
  hay_eaters = 56 →
  both_eaters = 14 →
  total_animals - (carrot_eaters + hay_eaters - both_eaters) = 7 :=
by sorry

end petting_zoo_theorem_l953_95332


namespace four_adjacent_squares_l953_95334

/-- A square in a plane -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ
  is_square : IsSquare vertices

/-- Two vertices are adjacent if they are consecutive in the cyclic order of the square -/
def adjacent (s : Square) (i j : Fin 4) : Prop :=
  (j = i + 1) ∨ (i = 3 ∧ j = 0)

/-- A square shares two adjacent vertices with another square -/
def shares_adjacent_vertices (s1 s2 : Square) : Prop :=
  ∃ (i j : Fin 4), adjacent s1 i j ∧ s1.vertices i = s2.vertices 0 ∧ s1.vertices j = s2.vertices 1

/-- The main theorem: there are exactly 4 squares sharing adjacent vertices with a given square -/
theorem four_adjacent_squares (s : Square) :
  ∃! (squares : Finset Square), squares.card = 4 ∧
    ∀ s' ∈ squares, shares_adjacent_vertices s s' :=
  sorry

end four_adjacent_squares_l953_95334


namespace factor_divisibility_l953_95339

theorem factor_divisibility : ∃ (n m : ℕ), (4 ∣ 24) ∧ (9 ∣ 180) := by
  sorry

end factor_divisibility_l953_95339


namespace chicken_coop_max_area_l953_95323

/-- The maximum area of a rectangular chicken coop with one side against a wall --/
theorem chicken_coop_max_area :
  let wall_length : ℝ := 15
  let fence_length : ℝ := 24
  let area (x : ℝ) : ℝ := x * (fence_length - x) / 2
  let max_area : ℝ := 72
  ∀ x, 0 < x ∧ x ≤ wall_length → area x ≤ max_area :=
by sorry

end chicken_coop_max_area_l953_95323


namespace custom_mult_factorial_difference_l953_95379

-- Define the custom multiplication operation
def custom_mult (a b : ℕ) : ℕ := a * b + a + b

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to calculate the chained custom multiplication
def chained_custom_mult (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => custom_mult (chained_custom_mult n) (n + 1)

theorem custom_mult_factorial_difference :
  factorial 10 - chained_custom_mult 9 = 1 := by
  sorry


end custom_mult_factorial_difference_l953_95379


namespace least_integer_with_digit_removal_property_l953_95307

theorem least_integer_with_digit_removal_property : ∃ (n : ℕ), 
  n > 0 ∧ 
  (n % 10 = 5 ∧ n / 10 = 9) ∧
  n = 19 * (n % 10) ∧
  (∀ m : ℕ, m > 0 → m < n → 
    (m % 10 ≠ 19 * (m / 10) ∨ m / 10 = 0)) := by
  sorry

end least_integer_with_digit_removal_property_l953_95307


namespace polynomial_decomposition_l953_95317

theorem polynomial_decomposition (x y : ℝ) :
  x^7 + x^6*y + x^5*y^2 + x^4*y^3 + x^3*y^4 + x^2*y^5 + x*y^6 + y^7 = (x + y)*(x^2 + y^2)*(x^4 + y^4) := by
  sorry

end polynomial_decomposition_l953_95317


namespace bookmarks_end_of_march_l953_95345

/-- Represents the number of pages bookmarked on each day of the week -/
def weekly_bookmarks : Fin 7 → ℕ
| 0 => 25  -- Monday
| 1 => 30  -- Tuesday
| 2 => 35  -- Wednesday
| 3 => 40  -- Thursday
| 4 => 45  -- Friday
| 5 => 50  -- Saturday
| _ => 55  -- Sunday

/-- The current number of bookmarked pages -/
def current_bookmarks : ℕ := 400

/-- The number of days in March -/
def march_days : ℕ := 31

/-- March starts on a Monday (represented by 0) -/
def march_start : Fin 7 := 0

/-- Calculates the total number of bookmarked pages at the end of March -/
def total_bookmarks_end_of_march : ℕ :=
  current_bookmarks +
  (march_days / 7 * (Finset.sum Finset.univ weekly_bookmarks)) +
  (Finset.sum (Finset.range (march_days % 7)) (λ i => weekly_bookmarks ((i + march_start) % 7)))

/-- Theorem stating that the total number of bookmarked pages at the end of March is 1610 -/
theorem bookmarks_end_of_march :
  total_bookmarks_end_of_march = 1610 := by sorry

end bookmarks_end_of_march_l953_95345


namespace exists_respectful_quadratic_with_zero_at_neg_one_l953_95388

/-- A respectful quadratic polynomial. -/
structure RespectfulQuadratic where
  a : ℝ
  b : ℝ

/-- The polynomial function for a respectful quadratic. -/
def q (p : RespectfulQuadratic) (x : ℝ) : ℝ :=
  x^2 + p.a * x + p.b

/-- The condition that q(q(x)) = 0 has exactly four real roots. -/
def hasFourRoots (p : RespectfulQuadratic) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄ ∧
    ∀ (x : ℝ), q p (q p x) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄

/-- The main theorem stating the existence of a respectful quadratic polynomial
    satisfying the required conditions. -/
theorem exists_respectful_quadratic_with_zero_at_neg_one :
  ∃ (p : RespectfulQuadratic), hasFourRoots p ∧ q p (-1) = 0 := by
  sorry

end exists_respectful_quadratic_with_zero_at_neg_one_l953_95388


namespace sin_shift_stretch_l953_95300

/-- Given a function f(x) = sin(2x), prove that shifting it right by π/12 and
    stretching x-coordinates by a factor of 2 results in g(x) = sin(x - π/6) -/
theorem sin_shift_stretch (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x)
  let shift : ℝ → ℝ := λ x => x - π / 12
  let stretch : ℝ → ℝ := λ x => x / 2
  let g : ℝ → ℝ := λ x => Real.sin (x - π / 6)
  (f ∘ shift ∘ stretch) x = g x :=
by sorry

end sin_shift_stretch_l953_95300


namespace base5_division_theorem_l953_95354

/-- Converts a base-5 number to decimal --/
def toDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base-5 --/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Represents a number in base-5 --/
structure Base5Number where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 5

/-- Division operation for Base5Number --/
def base5Div (a b : Base5Number) : Base5Number :=
  { digits := toBase5 ((toDecimal a.digits) / (toDecimal b.digits))
    valid := sorry }

theorem base5_division_theorem :
  let a : Base5Number := ⟨[1, 0, 3, 2], sorry⟩  -- 2301 in base 5
  let b : Base5Number := ⟨[2, 2], sorry⟩        -- 22 in base 5
  let result : Base5Number := ⟨[2, 0, 1], sorry⟩  -- 102 in base 5
  base5Div a b = result := by sorry

end base5_division_theorem_l953_95354


namespace trigonometric_identity_l953_95386

theorem trigonometric_identity (α : ℝ) :
  (Real.cos (4 * α - 3 * Real.pi) ^ 2 - 4 * Real.cos (2 * α - Real.pi) ^ 2 + 3) /
  (Real.cos (4 * α + 3 * Real.pi) ^ 2 + 4 * Real.cos (2 * α + Real.pi) ^ 2 - 1) =
  Real.tan (2 * α) ^ 4 := by
  sorry

end trigonometric_identity_l953_95386


namespace symmetric_point_xoz_l953_95377

/-- Represents a point in 3D Cartesian coordinates -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOz plane in 3D Cartesian coordinates -/
def xOzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Symmetry with respect to the xOz plane -/
def symmetricPointXOZ (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

/-- Theorem: The point symmetric to (-1, 2, 1) with respect to xOz plane is (-1, -2, 1) -/
theorem symmetric_point_xoz :
  let original := Point3D.mk (-1) 2 1
  symmetricPointXOZ original = Point3D.mk (-1) (-2) 1 := by
  sorry

end symmetric_point_xoz_l953_95377


namespace unpartnered_students_count_l953_95366

/-- Represents the number of students in a class -/
structure ClassCount where
  males : ℕ
  females : ℕ

/-- The number of students unable to partner with the opposite gender -/
def unpartnered_students (classes : List ClassCount) : ℕ :=
  let total_males := classes.map (·.males) |>.sum
  let total_females := classes.map (·.females) |>.sum
  Int.natAbs (total_males - total_females)

/-- The main theorem stating the number of unpartnered students -/
theorem unpartnered_students_count : 
  let classes : List ClassCount := [
    ⟨18, 12⟩,  -- First 6th grade class
    ⟨16, 20⟩,  -- Second 6th grade class
    ⟨13, 19⟩,  -- Third 6th grade class
    ⟨23, 21⟩   -- 7th grade class
  ]
  unpartnered_students classes = 2 := by
  sorry

end unpartnered_students_count_l953_95366


namespace find_b_l953_95306

theorem find_b : ∃ b : ℝ,
  let p : ℝ → ℝ := λ x ↦ 2 * x - 3
  let q : ℝ → ℝ := λ x ↦ 5 * x - b
  p (q 3) = 13 → b = 7 := by
  sorry

end find_b_l953_95306


namespace necessary_but_not_sufficient_l953_95391

theorem necessary_but_not_sufficient : 
  (∀ x y : ℝ, x > 3 ∧ y ≥ 3 → x^2 + y^2 ≥ 9) ∧ 
  (∃ x y : ℝ, x^2 + y^2 ≥ 9 ∧ ¬(x > 3 ∧ y ≥ 3)) := by
  sorry

end necessary_but_not_sufficient_l953_95391


namespace ratio_of_sum_to_difference_l953_95385

theorem ratio_of_sum_to_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
sorry

end ratio_of_sum_to_difference_l953_95385


namespace chip_credit_card_balance_l953_95333

/-- Calculates the balance on a credit card after two months, given an initial balance,
    monthly interest rate, and an additional charge in the second month. -/
def balance_after_two_months (initial_balance : ℝ) (interest_rate : ℝ) (additional_charge : ℝ) : ℝ :=
  let balance_after_first_month := initial_balance * (1 + interest_rate)
  let balance_before_second_interest := balance_after_first_month + additional_charge
  balance_before_second_interest * (1 + interest_rate)

/-- Theorem stating that given the specific conditions of Chip's credit card,
    the balance after two months is $48.00. -/
theorem chip_credit_card_balance :
  balance_after_two_months 50 0.2 20 = 48 :=
by sorry

end chip_credit_card_balance_l953_95333


namespace line_parametric_equation_l953_95316

/-- Parametric equation of a line passing through (1, 5) with slope angle π/3 -/
theorem line_parametric_equation :
  let M : ℝ × ℝ := (1, 5)
  let slope_angle : ℝ := π / 3
  let parametric_equation (t : ℝ) : ℝ × ℝ :=
    (M.1 + t * Real.cos slope_angle, M.2 + t * Real.sin slope_angle)
  ∀ t : ℝ, parametric_equation t = (1 + (1/2) * t, 5 + (Real.sqrt 3 / 2) * t) :=
by sorry

end line_parametric_equation_l953_95316


namespace sum_of_coordinates_is_16_l953_95399

/-- Given two points A and B in a 2D plane, where:
  - A is at the origin (0, 0)
  - B is on the line y = 6
  - The slope of segment AB is 3/5
  Prove that the sum of the x- and y-coordinates of B is 16. -/
theorem sum_of_coordinates_is_16 (B : ℝ × ℝ) : 
  B.2 = 6 ∧ 
  (B.2 - 0) / (B.1 - 0) = 3 / 5 → 
  B.1 + B.2 = 16 := by
  sorry

end sum_of_coordinates_is_16_l953_95399


namespace red_light_probability_l953_95372

theorem red_light_probability (p_first : ℝ) (p_both : ℝ) :
  p_first = 1/2 →
  p_both = 1/5 →
  p_both / p_first = 2/5 :=
by sorry

end red_light_probability_l953_95372


namespace infinite_sum_equals_floor_l953_95369

noncomputable def infiniteSum (x : ℝ) : ℕ → ℝ
  | 0 => ⌊(x + 1) / 2⌋
  | n + 1 => infiniteSum x n + ⌊(x + 2^(n+1)) / 2^(n+2)⌋

theorem infinite_sum_equals_floor (x : ℝ) :
  (∀ y : ℝ, ⌊2 * y⌋ = ⌊y⌋ + ⌊y + 1/2⌋) →
  (∃ N : ℕ, ∀ n ≥ N, ⌊(x + 2^n) / 2^(n+1)⌋ = 0) →
  (∃ M : ℕ, ∀ m ≥ M, infiniteSum x m = ⌊x⌋) :=
by sorry

end infinite_sum_equals_floor_l953_95369


namespace total_fallen_blocks_l953_95362

/-- Represents the heights of three stacks of blocks -/
structure BlockStacks where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of fallen blocks -/
def fallen_blocks (stacks : BlockStacks) (standing_second standing_third : ℕ) : ℕ :=
  stacks.first + (stacks.second - standing_second) + (stacks.third - standing_third)

theorem total_fallen_blocks : 
  let stacks : BlockStacks := { 
    first := 7, 
    second := 7 + 5, 
    third := 7 + 5 + 7 
  }
  fallen_blocks stacks 2 3 = 33 := by
  sorry

#eval fallen_blocks { first := 7, second := 7 + 5, third := 7 + 5 + 7 } 2 3

end total_fallen_blocks_l953_95362


namespace no_two_digit_primes_with_digit_sum_nine_l953_95342

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem no_two_digit_primes_with_digit_sum_nine :
  ∀ n : ℕ, is_two_digit n → digit_sum n = 9 → ¬ Nat.Prime n :=
sorry

end no_two_digit_primes_with_digit_sum_nine_l953_95342


namespace cyclic_quadrilateral_similarity_theorem_l953_95361

-- Define cyclic quadrilateral
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define similarity of quadrilaterals
def are_similar_quadrilaterals (A B C D A' B' C' D' : Point) : Prop := sorry

-- Define area of a triangle
def area_triangle (A B C : Point) : ℝ := sorry

-- Define distance between two points
def distance (A B : Point) : ℝ := sorry

theorem cyclic_quadrilateral_similarity_theorem 
  (A B C D A' B' C' D' : Point) 
  (h1 : is_cyclic_quadrilateral A B C D) 
  (h2 : is_cyclic_quadrilateral A' B' C' D')
  (h3 : are_similar_quadrilaterals A B C D A' B' C' D') :
  (distance A A')^2 * area_triangle B C D + (distance C C')^2 * area_triangle A B D = 
  (distance B B')^2 * area_triangle A C D + (distance D D')^2 * area_triangle A B C := by
sorry

end cyclic_quadrilateral_similarity_theorem_l953_95361


namespace same_heads_probability_l953_95324

def num_pennies_keiko : ℕ := 2
def num_pennies_ephraim : ℕ := 3

def total_outcomes : ℕ := 2^num_pennies_keiko * 2^num_pennies_ephraim

def favorable_outcomes : ℕ := 6

theorem same_heads_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 16 := by sorry

end same_heads_probability_l953_95324


namespace min_value_of_f_l953_95319

/-- The function f(x) = x^2 + 16x + 20 -/
def f (x : ℝ) : ℝ := x^2 + 16*x + 20

/-- The minimum value of f(x) is -44 -/
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ (m = -44) := by
  sorry

end min_value_of_f_l953_95319
