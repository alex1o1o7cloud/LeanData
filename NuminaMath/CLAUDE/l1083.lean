import Mathlib

namespace NUMINAMATH_CALUDE_coefficient_sum_equals_eight_l1083_108360

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x - 1) ^ 4

-- Define the coefficients a₀, a₁, a₂, a₃, a₄
variables (a₀ a₁ a₂ a₃ a₄ : ℝ)

-- State the theorem
theorem coefficient_sum_equals_eight :
  (∀ x, f x = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ = 8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_equals_eight_l1083_108360


namespace NUMINAMATH_CALUDE_document_download_income_increase_sales_target_increase_basketball_success_rate_l1083_108355

-- Define percentages as real numbers between 0 and 1
def Percentage := { x : ℝ // 0 ≤ x ∧ x ≤ 1 }

-- 1. Document download percentages
theorem document_download (a b : Percentage) :
  (a.val + b.val = 1) → ((a.val = 0.58 ∧ b.val = 0.42) ∨ (a.val = 0.42 ∧ b.val = 0.58)) :=
sorry

-- 2. Xiao Ming's income increase
theorem income_increase (last_year current_year : ℝ) (h : current_year = 1.24 * last_year) :
  current_year > last_year :=
sorry

-- 3. Shopping mall sales target
theorem sales_target_increase (august_target september_target : ℝ) 
  (h : september_target = 1.5 * august_target) :
  september_target > 0.5 * august_target :=
sorry

-- 4. Luo Luo's basketball shot success rate
theorem basketball_success_rate (attempts successes : ℕ) :
  attempts = 5 ∧ successes = 5 → (successes : ℝ) / attempts = 1 :=
sorry

end NUMINAMATH_CALUDE_document_download_income_increase_sales_target_increase_basketball_success_rate_l1083_108355


namespace NUMINAMATH_CALUDE_kelly_cheese_packages_l1083_108394

/-- The number of packages of string cheese needed for school lunches --/
def string_cheese_packages (days_per_week : ℕ) (oldest_daily : ℕ) (youngest_daily : ℕ) 
  (cheeses_per_package : ℕ) (num_weeks : ℕ) : ℕ :=
  let total_cheeses := (oldest_daily + youngest_daily) * days_per_week * num_weeks
  (total_cheeses + cheeses_per_package - 1) / cheeses_per_package

/-- Theorem: Kelly needs 2 packages of string cheese for 4 weeks of school lunches --/
theorem kelly_cheese_packages : 
  string_cheese_packages 5 2 1 30 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_kelly_cheese_packages_l1083_108394


namespace NUMINAMATH_CALUDE_bicycle_problem_l1083_108390

/-- Prove that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) 
  (h1 : distance = 12)
  (h2 : speed_ratio = 1.2)
  (h3 : time_difference = 1/6) : 
  let speed_B := 
    distance * (speed_ratio - 1) / (distance * speed_ratio * time_difference - distance * time_difference)
  speed_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_problem_l1083_108390


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l1083_108300

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l1083_108300


namespace NUMINAMATH_CALUDE_tangent_lines_intersection_l1083_108386

/-- Given a circle and four tangent points, proves that the diagonals of the
    trapezoid formed by the tangent lines intersect on the y-axis and that
    the line connecting two specific tangent points passes through this
    intersection point. -/
theorem tangent_lines_intersection
  (ξ η : ℝ)
  (h_ξ_pos : 0 < ξ)
  (h_ξ_lt_1 : ξ < 1)
  (h_circle_eq : ξ^2 + η^2 = 1) :
  ∃ y : ℝ,
    (∀ x : ℝ, x ≠ 0 →
      (y = -((2 * ξ) / (1 + η + ξ)) * x + (1 - η - ξ) / (1 + η + ξ) ↔
       y = ((2 * ξ) / (1 - η + ξ)) * x + (1 + η - ξ) / (1 - η + ξ))) ∧
    y = η / (ξ + 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_intersection_l1083_108386


namespace NUMINAMATH_CALUDE_minimum_distance_between_curves_l1083_108304

noncomputable def min_distance : ℝ := Real.sqrt 2 / 2 * (1 - Real.log 2)

theorem minimum_distance_between_curves :
  ∃ (a b : ℝ),
    (1/2 : ℝ) * Real.exp a = (1/2 : ℝ) * Real.exp a ∧
    b = b ∧
    ∀ (x y : ℝ),
      (1/2 : ℝ) * Real.exp x = (1/2 : ℝ) * Real.exp x →
      y = y →
      Real.sqrt ((x - y)^2 + ((1/2 : ℝ) * Real.exp x - y)^2) ≥ min_distance :=
by sorry

end NUMINAMATH_CALUDE_minimum_distance_between_curves_l1083_108304


namespace NUMINAMATH_CALUDE_work_fraction_after_twenty_days_l1083_108353

/-- Proves that the fraction of work completed after 20 days is 15/64 -/
theorem work_fraction_after_twenty_days 
  (W : ℝ) -- Total work to be done
  (initial_workers : ℕ := 10) -- Initial number of workers
  (initial_duration : ℕ := 100) -- Initial planned duration in days
  (work_time : ℕ := 20) -- Time worked before firing workers
  (fired_workers : ℕ := 2) -- Number of workers fired
  (remaining_time : ℕ := 75) -- Time to complete the remaining work
  (F : ℝ) -- Fraction of work completed after 20 days
  (h1 : initial_workers * (W / initial_duration) = work_time * (F * W / work_time)) -- Work rate equality for first 20 days
  (h2 : (initial_workers - fired_workers) * ((1 - F) * W / remaining_time) = initial_workers * (W / initial_duration)) -- Work rate equality for remaining work
  : F = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_work_fraction_after_twenty_days_l1083_108353


namespace NUMINAMATH_CALUDE_super_soup_revenue_theorem_l1083_108333

def super_soup_revenue (
  initial_stores : ℕ)
  (initial_avg_revenue : ℝ)
  (new_stores_2019 : ℕ)
  (new_revenue_2019 : ℝ)
  (closed_stores_2019 : ℕ)
  (closed_revenue_2019 : ℝ)
  (closed_expense_2019 : ℝ)
  (new_stores_2020 : ℕ)
  (new_revenue_2020 : ℝ)
  (closed_stores_2020 : ℕ)
  (closed_revenue_2020 : ℝ)
  (closed_expense_2020 : ℝ)
  (avg_expense : ℝ) : ℝ :=
  let initial_revenue := initial_stores * initial_avg_revenue
  let revenue_2019 := initial_revenue + new_stores_2019 * new_revenue_2019 - closed_stores_2019 * closed_revenue_2019
  let net_revenue_2019 := revenue_2019 + closed_stores_2019 * (closed_revenue_2019 - closed_expense_2019)
  let stores_2019 := initial_stores + new_stores_2019 - closed_stores_2019
  let revenue_2020 := net_revenue_2019 + new_stores_2020 * new_revenue_2020 - closed_stores_2020 * closed_revenue_2020
  let net_revenue_2020 := revenue_2020 + closed_stores_2020 * (closed_expense_2020 - closed_revenue_2020)
  let final_stores := stores_2019 + new_stores_2020 - closed_stores_2020
  net_revenue_2020 - final_stores * avg_expense

theorem super_soup_revenue_theorem :
  super_soup_revenue 23 500000 5 450000 2 300000 350000 10 600000 6 350000 380000 400000 = 5130000 := by
  sorry

end NUMINAMATH_CALUDE_super_soup_revenue_theorem_l1083_108333


namespace NUMINAMATH_CALUDE_percentage_soccer_players_is_12_5_l1083_108320

/-- The percentage of students who play sports that also play soccer -/
def percentage_soccer_players (total_students : ℕ) (sports_percentage : ℚ) (soccer_players : ℕ) : ℚ :=
  (soccer_players : ℚ) / (sports_percentage * total_students) * 100

/-- Theorem: The percentage of students who play sports that also play soccer is 12.5% -/
theorem percentage_soccer_players_is_12_5 :
  percentage_soccer_players 400 (52 / 100) 26 = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_soccer_players_is_12_5_l1083_108320


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1083_108341

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio of 3:2,
    prove that its diagonal length is √673.92 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 72) →
  (length / width = 3 / 2) →
  Real.sqrt (length^2 + width^2) = Real.sqrt 673.92 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1083_108341


namespace NUMINAMATH_CALUDE_max_r_value_l1083_108379

theorem max_r_value (r : ℕ) (m n : ℕ → ℤ) 
  (h1 : r ≥ 2)
  (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ r → |m i * n j - m j * n i| = 1) :
  r ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_r_value_l1083_108379


namespace NUMINAMATH_CALUDE_tricycle_count_l1083_108330

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) 
  (h1 : total_children = 12) 
  (h2 : total_wheels = 32) : ∃ (bicycles tricycles : ℕ), 
  bicycles + tricycles = total_children ∧ 
  2 * bicycles + 3 * tricycles = total_wheels ∧ 
  tricycles = 8 := by
sorry

end NUMINAMATH_CALUDE_tricycle_count_l1083_108330


namespace NUMINAMATH_CALUDE_abc_inequality_l1083_108371

theorem abc_inequality (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 1000)
  (h_sum : b * c * (1 - a) + a * (b + c) = 110) (h_a_lt_1 : a < 1) :
  10 < c ∧ c < 100 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1083_108371


namespace NUMINAMATH_CALUDE_midpoint_locus_l1083_108389

/-- Given a circle C with center (3,6) and radius 2√5, and a fixed point Q(-3,-6),
    the locus of the midpoint M of any point P on C and Q is described by the equation x^2 + y^2 = 5. -/
theorem midpoint_locus (P : ℝ × ℝ) (M : ℝ × ℝ) :
  (P.1 - 3)^2 + (P.2 - 6)^2 = 20 →
  M.1 = (P.1 + (-3)) / 2 →
  M.2 = (P.2 + (-6)) / 2 →
  M.1^2 + M.2^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_locus_l1083_108389


namespace NUMINAMATH_CALUDE_point_on_line_l1083_108399

theorem point_on_line (k : ℝ) : 
  (1 + 3 * k * (-1/3) = -4 * 4) → k = 17 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l1083_108399


namespace NUMINAMATH_CALUDE_grapes_needed_theorem_l1083_108362

/-- The amount of grapes needed in a year after a 20% increase in production -/
def grapes_needed_after_increase (initial_usage : ℝ) : ℝ :=
  2 * (initial_usage * 1.2)

/-- Theorem stating that given an initial grape usage of 90 kg per 6 months 
    and a 20% increase in production, the total amount of grapes needed in a year is 216 kg -/
theorem grapes_needed_theorem :
  grapes_needed_after_increase 90 = 216 := by
  sorry

#eval grapes_needed_after_increase 90

end NUMINAMATH_CALUDE_grapes_needed_theorem_l1083_108362


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1083_108398

def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def B : Set ℝ := {-2, -1, 0, 1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1083_108398


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1083_108382

theorem polynomial_simplification (x : ℝ) :
  (2 * x^2 + 5 * x - 3) - (2 * x^2 + 9 * x - 6) = -4 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1083_108382


namespace NUMINAMATH_CALUDE_four_times_three_plus_two_l1083_108329

theorem four_times_three_plus_two : (4 * 3) + 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_four_times_three_plus_two_l1083_108329


namespace NUMINAMATH_CALUDE_class_project_funding_l1083_108350

/-- Calculates the total amount gathered by a class for a project --/
def total_amount_gathered (total_students : ℕ) (full_payment : ℕ) (half_paying_students : ℕ) : ℕ :=
  let full_paying_students := total_students - half_paying_students
  let full_amount := full_paying_students * full_payment
  let half_amount := half_paying_students * (full_payment / 2)
  full_amount + half_amount

/-- Proves that the class gathered $1150 for their project --/
theorem class_project_funding :
  total_amount_gathered 25 50 4 = 1150 := by
  sorry

end NUMINAMATH_CALUDE_class_project_funding_l1083_108350


namespace NUMINAMATH_CALUDE_count_divisible_by_3_5_7_60_l1083_108307

def count_divisible (n : ℕ) (d : ℕ) : ℕ := n / d

def count_divisible_by_3_5_7 (upper_bound : ℕ) : ℕ :=
  let div3 := count_divisible upper_bound 3
  let div5 := count_divisible upper_bound 5
  let div7 := count_divisible upper_bound 7
  let div3_5 := count_divisible upper_bound 15
  let div3_7 := count_divisible upper_bound 21
  let div5_7 := count_divisible upper_bound 35
  div3 + div5 + div7 - (div3_5 + div3_7 + div5_7)

theorem count_divisible_by_3_5_7_60 : count_divisible_by_3_5_7 60 = 33 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_3_5_7_60_l1083_108307


namespace NUMINAMATH_CALUDE_fencing_problem_l1083_108387

theorem fencing_problem (area : ℝ) (uncovered_side : ℝ) :
  area = 600 ∧ uncovered_side = 10 →
  ∃ width : ℝ, 
    area = uncovered_side * width ∧
    uncovered_side + 2 * width = 130 :=
by sorry

end NUMINAMATH_CALUDE_fencing_problem_l1083_108387


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_l1083_108324

theorem rectangle_length_fraction (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ) :
  square_area = 2025 →
  rectangle_area = 180 →
  rectangle_breadth = 10 →
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_l1083_108324


namespace NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_l1083_108373

-- Define a type for lines in space
variable (Line : Type)

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Axiom: If two lines are parallel to the same line, they are parallel to each other
axiom parallel_transitivity :
  ∀ (l1 l2 l3 : Line), parallel l1 l3 → parallel l2 l3 → parallel l1 l2

-- Theorem: Two lines parallel to the same line are parallel to each other
theorem lines_parallel_to_same_line_are_parallel
  (l1 l2 l3 : Line) (h1 : parallel l1 l3) (h2 : parallel l2 l3) :
  parallel l1 l2 :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_l1083_108373


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l1083_108372

/-- The volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume 
  (l : ℝ) (α : ℝ) (h_l : l > 0) (h_α : 0 < α ∧ α < π / 2) :
  let volume := (l^3 * Real.sqrt 3 * Real.sin (2 * α) * Real.cos α) / 8
  ∃ (V : ℝ), V = volume ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l1083_108372


namespace NUMINAMATH_CALUDE_percent_greater_average_l1083_108321

theorem percent_greater_average (M N : ℝ) (h : M > N) :
  (M - N) / ((M + N) / 2) * 100 = 200 * (M - N) / (M + N) := by
  sorry

end NUMINAMATH_CALUDE_percent_greater_average_l1083_108321


namespace NUMINAMATH_CALUDE_quadratic_equation_at_negative_two_l1083_108367

theorem quadratic_equation_at_negative_two :
  let x : ℤ := -2
  x^2 + 6*x - 10 = -18 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_at_negative_two_l1083_108367


namespace NUMINAMATH_CALUDE_first_rectangle_height_l1083_108392

/-- Proves that the height of the first rectangle is 5 inches -/
theorem first_rectangle_height : 
  ∀ (h : ℝ), -- height of the first rectangle
  (4 * h = 3 * 6 + 2) → -- area of first = area of second + 2
  h = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_rectangle_height_l1083_108392


namespace NUMINAMATH_CALUDE_chicken_surprise_servings_l1083_108385

/-- Calculates the number of servings for Chicken Surprise recipe -/
theorem chicken_surprise_servings 
  (chicken_pounds : ℝ) 
  (stuffing_ounces : ℝ) 
  (serving_size_ounces : ℝ) : 
  chicken_pounds = 4.5 ∧ 
  stuffing_ounces = 24 ∧ 
  serving_size_ounces = 8 → 
  (chicken_pounds * 16 + stuffing_ounces) / serving_size_ounces = 12 := by
sorry


end NUMINAMATH_CALUDE_chicken_surprise_servings_l1083_108385


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l1083_108361

theorem simultaneous_equations_solution :
  ∃! (x y : ℚ), 3 * x - 4 * y = 11 ∧ 9 * x + 6 * y = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l1083_108361


namespace NUMINAMATH_CALUDE_no_valid_assignment_l1083_108303

/-- Represents a mapping of characters to digits -/
def DigitAssignment := Char → Nat

/-- Checks if a DigitAssignment is valid for the given cryptarithmic problem -/
def is_valid_assignment (assignment : DigitAssignment) : Prop :=
  let s := assignment 'S'
  let t := assignment 'T'
  let i := assignment 'I'
  let k := assignment 'K'
  let m := assignment 'M'
  let a := assignment 'A'
  (s ≠ 0) ∧ 
  (m ≠ 0) ∧
  (s ≠ t) ∧ (s ≠ i) ∧ (s ≠ k) ∧ (s ≠ m) ∧ (s ≠ a) ∧
  (t ≠ i) ∧ (t ≠ k) ∧ (t ≠ m) ∧ (t ≠ a) ∧
  (i ≠ k) ∧ (i ≠ m) ∧ (i ≠ a) ∧
  (k ≠ m) ∧ (k ≠ a) ∧
  (m ≠ a) ∧
  (s < 10) ∧ (t < 10) ∧ (i < 10) ∧ (k < 10) ∧ (m < 10) ∧ (a < 10) ∧
  (10000 * s + 1000 * t + 100 * i + 10 * k + s +
   10000 * s + 1000 * t + 100 * i + 10 * k + s =
   100000 * m + 10000 * a + 1000 * s + 100 * t + 10 * i + k + s)

theorem no_valid_assignment : ¬∃ (assignment : DigitAssignment), is_valid_assignment assignment :=
sorry

end NUMINAMATH_CALUDE_no_valid_assignment_l1083_108303


namespace NUMINAMATH_CALUDE_sqrt_225_equals_15_l1083_108342

theorem sqrt_225_equals_15 : Real.sqrt 225 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_225_equals_15_l1083_108342


namespace NUMINAMATH_CALUDE_problem_statement_l1083_108317

theorem problem_statement (x₁ x₂ x₃ x₄ n : ℝ) 
  (h1 : x₁ ≠ x₂)
  (h2 : (x₁ + x₃) * (x₁ + x₄) = n - 10)
  (h3 : (x₂ + x₃) * (x₂ + x₄) = n - 10)
  (h4 : x₁ + x₂ + x₃ + x₄ = 0) :
  let p := (x₁ + x₃) * (x₂ + x₃) + (x₁ + x₄) * (x₂ + x₄)
  p = 2 * n - 20 := by
  sorry


end NUMINAMATH_CALUDE_problem_statement_l1083_108317


namespace NUMINAMATH_CALUDE_mark_remaining_hours_l1083_108368

def sick_days : ℕ := 10
def vacation_days : ℕ := 10
def hours_per_day : ℕ := 8
def used_fraction : ℚ := 1/2

theorem mark_remaining_hours : 
  (sick_days + vacation_days) * (1 - used_fraction) * hours_per_day = 80 := by
  sorry

end NUMINAMATH_CALUDE_mark_remaining_hours_l1083_108368


namespace NUMINAMATH_CALUDE_tangent_circles_count_l1083_108397

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Determines if a circle is tangent to two other circles -/
def is_tangent_to_both (c : Circle) (c1 c2 : Circle) : Prop :=
  are_externally_tangent c c1 ∧ are_externally_tangent c c2

/-- The main theorem to be proven -/
theorem tangent_circles_count 
  (O1 O2 : Circle) 
  (h_tangent : are_externally_tangent O1 O2) 
  (h_radius1 : O1.radius = 2) 
  (h_radius2 : O2.radius = 4) : 
  ∃! (s : Finset Circle), 
    Finset.card s = 5 ∧ 
    ∀ c ∈ s, c.radius = 6 ∧ is_tangent_to_both c O1 O2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l1083_108397


namespace NUMINAMATH_CALUDE_remaining_money_after_gifts_l1083_108391

def initial_budget : ℚ := 999
def shoes_cost : ℚ := 165
def yoga_mat_cost : ℚ := 85
def sports_watch_cost : ℚ := 215
def hand_weights_cost : ℚ := 60

theorem remaining_money_after_gifts :
  initial_budget - (shoes_cost + yoga_mat_cost + sports_watch_cost + hand_weights_cost) = 474 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_after_gifts_l1083_108391


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1083_108316

theorem regular_polygon_exterior_angle (n : ℕ) (exterior_angle : ℝ) :
  n ≥ 3 → exterior_angle = 36 → n = 10 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1083_108316


namespace NUMINAMATH_CALUDE_opposite_of_negative_nine_l1083_108318

theorem opposite_of_negative_nine :
  ∃ x : ℤ, x + (-9) = 0 ∧ x = 9 :=
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_nine_l1083_108318


namespace NUMINAMATH_CALUDE_unique_exaggeration_combination_l1083_108328

/-- Represents the number of people who exaggerated the wolf's tail length --/
structure TailExaggeration where
  simple : Nat
  creative : Nat

/-- Calculates the final tail length given the number of simple and creative people --/
def finalTailLength (e : TailExaggeration) : Nat :=
  (2 ^ e.simple) * (3 ^ e.creative)

/-- Theorem stating that there is a unique combination of simple and creative people
    that results in a tail length of 864 meters --/
theorem unique_exaggeration_combination :
  ∃! e : TailExaggeration, finalTailLength e = 864 :=
sorry

end NUMINAMATH_CALUDE_unique_exaggeration_combination_l1083_108328


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_min_value_2a_plus_b_equality_l1083_108336

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 1) + 2 / (b - 2) = 1 / 2) : 
  2 * a + b ≥ 16 := by
  sorry

theorem min_value_2a_plus_b_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 1) + 2 / (b - 2) = 1 / 2) : 
  (2 * a + b = 16) ↔ (a = 3 ∧ b = 10) := by
  sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_min_value_2a_plus_b_equality_l1083_108336


namespace NUMINAMATH_CALUDE_unique_divisible_digit_l1083_108310

def is_divisible (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def seven_digit_number (A : ℕ) : ℕ := 3538080 + A

theorem unique_divisible_digit :
  ∃! A : ℕ,
    A < 10 ∧
    is_divisible (seven_digit_number A) 2 ∧
    is_divisible (seven_digit_number A) 3 ∧
    is_divisible (seven_digit_number A) 4 ∧
    is_divisible (seven_digit_number A) 5 ∧
    is_divisible (seven_digit_number A) 6 ∧
    is_divisible (seven_digit_number A) 8 ∧
    is_divisible (seven_digit_number A) 9 ∧
    A = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_digit_l1083_108310


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l1083_108305

theorem greatest_prime_factor_of_341 : ∃ (p : ℕ), p.Prime ∧ p ∣ 341 ∧ ∀ (q : ℕ), q.Prime → q ∣ 341 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l1083_108305


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_modulo_l1083_108335

theorem arithmetic_sequence_sum_modulo (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 5 →
  aₙ = 145 →
  d = 5 →
  n = (aₙ - a₁) / d + 1 →
  (n * (a₁ + aₙ) / 2) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_modulo_l1083_108335


namespace NUMINAMATH_CALUDE_prob_three_correct_is_five_twelfths_l1083_108384

-- Define the probability of A and B guessing correctly
def prob_A_correct : ℚ := 3/4
def prob_B_correct : ℚ := 2/3

-- Define the function to calculate the probability of exactly three correct guesses
def prob_three_correct : ℚ :=
  let p_A := prob_A_correct
  let p_B := prob_B_correct
  let q_A := 1 - p_A
  let q_B := 1 - p_B
  
  -- Calculate the probability of each scenario
  let scenario1 := p_A * p_A * p_A * p_B * q_B * q_B * q_B
  let scenario2 := p_A * p_A * p_A * p_B * q_B * p_B * q_B
  let scenario3 := p_A * p_A * p_A * p_B * p_B * q_B * q_B
  let scenario4 := p_A * p_A * p_A * q_B * p_B * p_B * q_B
  
  -- Sum up all scenarios
  scenario1 + scenario2 + scenario3 + scenario4

-- Theorem statement
theorem prob_three_correct_is_five_twelfths :
  prob_three_correct = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_correct_is_five_twelfths_l1083_108384


namespace NUMINAMATH_CALUDE_least_possible_difference_l1083_108309

theorem least_possible_difference (x y z : ℤ) : 
  x < y ∧ y < z ∧ 
  y - x > 5 ∧ 
  Even x ∧ Odd y ∧ Odd z →
  ∀ w, w = z - x → w ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l1083_108309


namespace NUMINAMATH_CALUDE_propositions_P_and_Q_l1083_108332

theorem propositions_P_and_Q : 
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 → 1/a + 1/b > 3) ∧
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_propositions_P_and_Q_l1083_108332


namespace NUMINAMATH_CALUDE_matrix_multiplication_l1083_108363

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 6]

theorem matrix_multiplication :
  A * B = !![17, -3; 16, -24] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_l1083_108363


namespace NUMINAMATH_CALUDE_last_remaining_number_l1083_108364

/-- Represents the process of skipping and marking numbers -/
def josephus_process (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that for n = 50, the last remaining number is 49 -/
theorem last_remaining_number : josephus_process 50 = 49 := by
  sorry

end NUMINAMATH_CALUDE_last_remaining_number_l1083_108364


namespace NUMINAMATH_CALUDE_set_operations_l1083_108325

def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

theorem set_operations :
  (Set.compl (A ∩ B) = {x | x < 3 ∨ x ≥ 6}) ∧
  ((Set.compl B ∪ A) = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1083_108325


namespace NUMINAMATH_CALUDE_number_multiplied_by_7000_l1083_108381

theorem number_multiplied_by_7000 : ∃ x : ℝ, x * 7000 = (28000 : ℝ) * (100 : ℝ)^1 ∧ x = 400 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_7000_l1083_108381


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l1083_108396

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (given_line : Line) 
  (point : ℝ × ℝ) : 
  ∃ (result_line : Line), 
    result_line.contains point.1 point.2 ∧ 
    result_line.parallel given_line ∧
    result_line.a = 1 ∧ 
    result_line.b = 2 ∧ 
    result_line.c = -3 :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l1083_108396


namespace NUMINAMATH_CALUDE_lcm_problem_l1083_108344

theorem lcm_problem (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 28) :
  ∃ (a' c' : ℕ), a' ∣ a ∧ c' ∣ c ∧ Nat.lcm a' c' = 35 ∧ 
  ∀ (x y : ℕ), x ∣ a → y ∣ c → Nat.lcm x y ≥ 35 :=
sorry

end NUMINAMATH_CALUDE_lcm_problem_l1083_108344


namespace NUMINAMATH_CALUDE_correct_answer_l1083_108352

theorem correct_answer (x : ℤ) (h : x + 5 = 35) : x - 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l1083_108352


namespace NUMINAMATH_CALUDE_daria_concert_friends_l1083_108393

def ticket_cost : ℕ := 90
def current_money : ℕ := 189
def additional_money_needed : ℕ := 171

def total_cost : ℕ := current_money + additional_money_needed

def total_tickets : ℕ := total_cost / ticket_cost

def number_of_friends : ℕ := total_tickets - 1

theorem daria_concert_friends : number_of_friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_daria_concert_friends_l1083_108393


namespace NUMINAMATH_CALUDE_wire_pieces_lengths_l1083_108331

/-- Represents the lengths of four pieces of wire --/
structure WirePieces where
  piece1 : ℝ
  piece2 : ℝ
  piece3 : ℝ
  piece4 : ℝ

/-- The total length of the wire is 72 feet --/
def totalLength : ℝ := 72

/-- Theorem stating the correct lengths of the wire pieces --/
theorem wire_pieces_lengths : ∃ (w : WirePieces),
  w.piece1 = 14.75 ∧
  w.piece2 = 11.75 ∧
  w.piece3 = 21.5 ∧
  w.piece4 = 24 ∧
  w.piece1 = w.piece2 + 3 ∧
  w.piece3 = 2 * w.piece2 - 2 ∧
  w.piece4 = (w.piece1 + w.piece2 + w.piece3) / 2 ∧
  w.piece1 + w.piece2 + w.piece3 + w.piece4 = totalLength := by
  sorry

end NUMINAMATH_CALUDE_wire_pieces_lengths_l1083_108331


namespace NUMINAMATH_CALUDE_difference_percentages_l1083_108345

theorem difference_percentages : (800 * 75 / 100) - (1200 * 7 / 8) = 450 := by
  sorry

end NUMINAMATH_CALUDE_difference_percentages_l1083_108345


namespace NUMINAMATH_CALUDE_roots_product_l1083_108334

theorem roots_product (a b c d : ℝ) (h1 : 36 * a^3 - 66 * a^2 + 31 * a - 4 = 0)
  (h2 : 36 * b^3 - 66 * b^2 + 31 * b - 4 = 0)
  (h3 : 36 * c^3 - 66 * c^2 + 31 * c - 4 = 0)
  (h4 : b - a = c - b) -- arithmetic progression
  (h5 : a < b ∧ b < c) -- ordering of roots
  : a * c = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_l1083_108334


namespace NUMINAMATH_CALUDE_inequality_solution_l1083_108380

def solution_set (m : ℝ) : Set ℝ :=
  if m < -4 then {x | -1 < x ∧ x < 1 / (m + 3)}
  else if m = -4 then ∅
  else if m > -4 ∧ m < -3 then {x | 1 / (m + 3) < x ∧ x < -1}
  else if m = -3 then {x | x > -1}
  else {x | x < -1 ∨ x > 1 / (m + 3)}

theorem inequality_solution (m : ℝ) :
  {x : ℝ | ((m + 3) * x - 1) * (x + 1) > 0} = solution_set m := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1083_108380


namespace NUMINAMATH_CALUDE_integer_sum_of_fourth_powers_l1083_108326

theorem integer_sum_of_fourth_powers (a b c : ℤ) (h : a = b + c) :
  a^4 + b^4 + c^4 = 2 * (a^2 - b*c)^2 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_of_fourth_powers_l1083_108326


namespace NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l1083_108347

theorem largest_whole_number_nine_times_less_than_150 :
  ∃ (x : ℤ), x = 16 ∧ (∀ y : ℤ, 9 * y < 150 → y ≤ x) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l1083_108347


namespace NUMINAMATH_CALUDE_set_equality_proof_l1083_108359

theorem set_equality_proof (M N : Set ℕ) : M = {3, 2} → N = {2, 3} → M = N := by
  sorry

end NUMINAMATH_CALUDE_set_equality_proof_l1083_108359


namespace NUMINAMATH_CALUDE_yolanda_three_point_average_l1083_108314

theorem yolanda_three_point_average (total_points season_games free_throws_per_game two_point_baskets_per_game : ℕ)
  (h1 : total_points = 345)
  (h2 : season_games = 15)
  (h3 : free_throws_per_game = 4)
  (h4 : two_point_baskets_per_game = 5) :
  (total_points - (free_throws_per_game * 1 + two_point_baskets_per_game * 2) * season_games) / (3 * season_games) = 3 := by
  sorry

end NUMINAMATH_CALUDE_yolanda_three_point_average_l1083_108314


namespace NUMINAMATH_CALUDE_robot_fifth_minute_distance_l1083_108369

def robot_distance (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 2
  | k + 1 => 2 * robot_distance k

theorem robot_fifth_minute_distance :
  robot_distance 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_robot_fifth_minute_distance_l1083_108369


namespace NUMINAMATH_CALUDE_min_value_constrained_min_value_achieved_l1083_108366

theorem min_value_constrained (x y : ℝ) (h : 2 * x + 8 * y = 3) :
  x^2 + 4 * y^2 - 2 * x ≥ -19/20 := by
  sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, 2 * x + 8 * y = 3 ∧ x^2 + 4 * y^2 - 2 * x < -19/20 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_constrained_min_value_achieved_l1083_108366


namespace NUMINAMATH_CALUDE_f_properties_l1083_108370

noncomputable def f (x : ℝ) := Real.exp x - x + (1/2) * x^2

theorem f_properties :
  (∃ (x₀ : ℝ), f x₀ = 1 ∧ ∀ (x : ℝ), f x ≥ f x₀) ∧  -- Minimum value is 1
  (∀ (M : ℝ), ∃ (x : ℝ), f x > M) ∧                -- No maximum value
  (∀ (a b : ℝ), (∀ (x : ℝ), (1/2) * x^2 - f x ≤ a * x + b) →
    (1 - a) * b ≥ -Real.exp 1 / 2) ∧               -- Minimum value of (1-a)b
  (∃ (a b : ℝ), (∀ (x : ℝ), (1/2) * x^2 - f x ≤ a * x + b) ∧
    (1 - a) * b = -Real.exp 1 / 2) :=               -- Minimum is attained
by sorry

end NUMINAMATH_CALUDE_f_properties_l1083_108370


namespace NUMINAMATH_CALUDE_quadratic_roots_bound_l1083_108339

theorem quadratic_roots_bound (a b c : ℝ) (x₁ x₂ : ℝ) (ha : a > 0) :
  let P : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (P x₁ = 0 ∧ P x₂ = 0) →
  (abs x₁ ≤ 1 ∧ abs x₂ ≤ 1) ↔ (a + b + c ≥ 0 ∧ a - b + c ≥ 0 ∧ a - c ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_bound_l1083_108339


namespace NUMINAMATH_CALUDE_nine_points_chords_l1083_108349

/-- The number of different chords that can be drawn by connecting two points
    out of n points on the circumference of a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: There are 36 different chords that can be drawn by connecting two
    points out of nine points on the circumference of a circle -/
theorem nine_points_chords : num_chords 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nine_points_chords_l1083_108349


namespace NUMINAMATH_CALUDE_square_sum_divisibility_problem_l1083_108388

theorem square_sum_divisibility_problem :
  ∃ a b : ℕ, a^2 + b^2 = 2018 ∧ 7 ∣ (a + b) ∧
  ((a = 43 ∧ b = 13) ∨ (a = 13 ∧ b = 43)) ∧
  (∀ x y : ℕ, x^2 + y^2 = 2018 ∧ 7 ∣ (x + y) → (x = 43 ∧ y = 13) ∨ (x = 13 ∧ y = 43)) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_divisibility_problem_l1083_108388


namespace NUMINAMATH_CALUDE_sum_of_fractions_integer_l1083_108376

theorem sum_of_fractions_integer (n : ℕ+) :
  (1/2 + 1/3 + 1/5 + 1/n.val : ℚ).isInt → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_integer_l1083_108376


namespace NUMINAMATH_CALUDE_line_slope_m_l1083_108356

theorem line_slope_m (m : ℝ) : 
  m > 0 → 
  ((m - 4) / (2 - m) = 2 * m) →
  m = (3 + Real.sqrt 41) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_line_slope_m_l1083_108356


namespace NUMINAMATH_CALUDE_extreme_values_imply_a_range_extreme_values_imply_a_in_range_l1083_108319

/-- A function f with two extreme values in R -/
structure TwoExtremeFunction (f : ℝ → ℝ) : Prop where
  has_two_extremes : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (∀ (x : ℝ), f x ≤ f x₁) ∧ 
    (∀ (x : ℝ), f x ≤ f x₂)

/-- The main theorem -/
theorem extreme_values_imply_a_range 
  (a : ℝ) 
  (ha : a ≠ 0) 
  (f : ℝ → ℝ)
  (hf : f = λ x => (1 + a * x^2) * Real.exp x)
  (h_two_extremes : TwoExtremeFunction f) :
  a < 0 ∨ a > 1 := by
  sorry

/-- The range of a as a set -/
def a_range : Set ℝ := {a | a < 0 ∨ a > 1}

/-- An equivalent formulation of the main theorem using sets -/
theorem extreme_values_imply_a_in_range 
  (a : ℝ) 
  (ha : a ≠ 0) 
  (f : ℝ → ℝ)
  (hf : f = λ x => (1 + a * x^2) * Real.exp x)
  (h_two_extremes : TwoExtremeFunction f) :
  a ∈ a_range := by
  sorry

end NUMINAMATH_CALUDE_extreme_values_imply_a_range_extreme_values_imply_a_in_range_l1083_108319


namespace NUMINAMATH_CALUDE_motorcyclist_speed_l1083_108315

theorem motorcyclist_speed 
  (hiker_speed : ℝ)
  (time_to_stop : ℝ)
  (time_to_catch_up : ℝ)
  (h1 : hiker_speed = 6)
  (h2 : time_to_stop = 0.2)
  (h3 : time_to_catch_up = 0.8) :
  ∃ (motorcyclist_speed : ℝ),
    motorcyclist_speed * time_to_stop = 
    hiker_speed * (time_to_stop + time_to_catch_up) ∧
    motorcyclist_speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_motorcyclist_speed_l1083_108315


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l1083_108312

/-- The y-coordinate of the point on the y-axis equidistant from A(-3, 1) and B(2, 5) is 19/8 -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℝ, ((-3 - 0)^2 + (1 - y)^2 = (2 - 0)^2 + (5 - y)^2) ∧ y = 19/8 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l1083_108312


namespace NUMINAMATH_CALUDE_sphere_radius_from_great_circle_area_l1083_108348

theorem sphere_radius_from_great_circle_area (A : ℝ) (R : ℝ) :
  A = 4 * Real.pi → A = Real.pi * R^2 → R = 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_great_circle_area_l1083_108348


namespace NUMINAMATH_CALUDE_line_m_equation_l1083_108302

-- Define the xy-plane
structure XYPlane where
  x : ℝ
  y : ℝ

-- Define a line in the xy-plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given lines and points
def line_l : Line := { a := 3, b := -1, c := 0 }
def point_P : XYPlane := { x := -3, y := 2 }
def point_P'' : XYPlane := { x := 2, y := -1 }

-- Define the reflection operation
def reflect (p : XYPlane) (l : Line) : XYPlane :=
  sorry

-- State the theorem
theorem line_m_equation :
  ∃ (line_m : Line),
    (line_m.a ≠ line_l.a ∨ line_m.b ≠ line_l.b) ∧
    (line_m.a * 0 + line_m.b * 0 + line_m.c = 0) ∧
    (∃ (point_P' : XYPlane),
      reflect point_P line_l = point_P' ∧
      reflect point_P' line_m = point_P'') ∧
    line_m.a = 1 ∧ line_m.b = 3 ∧ line_m.c = 0 :=
  sorry

end NUMINAMATH_CALUDE_line_m_equation_l1083_108302


namespace NUMINAMATH_CALUDE_minimum_value_implies_c_l1083_108375

def f (c : ℝ) (x : ℝ) : ℝ := x^4 - 8*x^2 + c

theorem minimum_value_implies_c (c : ℝ) :
  (∃ x₀ ∈ Set.Icc (-1) 3, f c x₀ = -14 ∧ ∀ x ∈ Set.Icc (-1) 3, f c x ≥ -14) →
  c = 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_implies_c_l1083_108375


namespace NUMINAMATH_CALUDE_petyas_friends_count_l1083_108377

/-- The number of friends Petya has -/
def num_friends : ℕ := 19

/-- The number of stickers Petya has -/
def total_stickers : ℕ := num_friends * 5 + 8

theorem petyas_friends_count :
  (num_friends * 5 + 8 = total_stickers) ∧
  (num_friends * 6 = total_stickers + 11) :=
by sorry

end NUMINAMATH_CALUDE_petyas_friends_count_l1083_108377


namespace NUMINAMATH_CALUDE_g_g_is_odd_l1083_108357

def f (x : ℝ) := x^3

def g (x : ℝ) := f (f x)

theorem g_g_is_odd (h1 : ∀ x, f (-x) = -f x) : 
  ∀ x, g (g (-x)) = -(g (g x)) := by sorry

end NUMINAMATH_CALUDE_g_g_is_odd_l1083_108357


namespace NUMINAMATH_CALUDE_prime_sum_ways_8_l1083_108338

/-- A function that returns the number of unique ways to sum prime numbers to form a given natural number,
    where the prime numbers in the sum are in non-decreasing order. -/
def prime_sum_ways (n : ℕ) : ℕ := sorry

/-- A function that checks if a list of natural numbers is a valid prime sum for a given number,
    where the numbers in the list are prime and in non-decreasing order. -/
def is_valid_prime_sum (n : ℕ) (sum : List ℕ) : Prop := sorry

theorem prime_sum_ways_8 : prime_sum_ways 8 = 2 := by sorry

end NUMINAMATH_CALUDE_prime_sum_ways_8_l1083_108338


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1083_108378

/-- The line 4x - 3y = 0 intersects the circle x^2 + y^2 = 36 -/
theorem line_intersects_circle :
  ∃ (x y : ℝ), 4 * x - 3 * y = 0 ∧ x^2 + y^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1083_108378


namespace NUMINAMATH_CALUDE_high_school_nine_games_l1083_108323

/-- The number of teams in the league -/
def num_teams : ℕ := 9

/-- The number of games each team plays against non-league opponents -/
def non_league_games : ℕ := 6

/-- The total number of games played in a season -/
def total_games : ℕ := 126

/-- Theorem stating the total number of games in a season -/
theorem high_school_nine_games :
  (num_teams * (num_teams - 1)) + (num_teams * non_league_games) = total_games :=
sorry

end NUMINAMATH_CALUDE_high_school_nine_games_l1083_108323


namespace NUMINAMATH_CALUDE_complex_number_conditions_complex_number_on_line_l1083_108306

def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)

theorem complex_number_conditions (a : ℝ) :
  (z a).re < 0 ∧ (z a).im > 0 ↔ -2 < a ∧ a < 1 :=
sorry

theorem complex_number_on_line (a : ℝ) :
  (z a).re = (z a).im ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_complex_number_conditions_complex_number_on_line_l1083_108306


namespace NUMINAMATH_CALUDE_range_of_m_l1083_108358

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, Real.exp (|2*x + 1|) + m ≥ 0) ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1083_108358


namespace NUMINAMATH_CALUDE_annual_concert_ticket_sales_l1083_108313

theorem annual_concert_ticket_sales 
  (total_tickets : ℕ) 
  (student_price non_student_price : ℚ) 
  (total_revenue : ℚ) 
  (h1 : total_tickets = 150)
  (h2 : student_price = 5)
  (h3 : non_student_price = 8)
  (h4 : total_revenue = 930) :
  ∃ (student_tickets : ℕ), 
    student_tickets = 90 ∧ 
    ∃ (non_student_tickets : ℕ), 
      student_tickets + non_student_tickets = total_tickets ∧
      student_price * student_tickets + non_student_price * non_student_tickets = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_annual_concert_ticket_sales_l1083_108313


namespace NUMINAMATH_CALUDE_range_of_a_l1083_108340

-- Define the conditions
def condition_p (a : ℝ) : Prop := ∃ m : ℝ, m ∈ Set.Icc (-1) 1 ∧ a^2 - 5*a + 5 ≥ m + 2

def condition_q (a : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + a*x₁ + 2 = 0 ∧ x₂^2 + a*x₂ + 2 = 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (condition_p a ∨ condition_q a) ∧ ¬(condition_p a ∧ condition_q a) →
  a ≤ 1 ∨ (2 * Real.sqrt 2 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1083_108340


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1083_108365

theorem trigonometric_identity : 
  3.4173 * Real.sin (2 * Real.pi / 17) + Real.sin (4 * Real.pi / 17) - 
  Real.sin (6 * Real.pi / 17) - 0.5 * Real.sin (8 * Real.pi / 17) = 
  8 * Real.sin (2 * Real.pi / 17) ^ 3 * Real.cos (Real.pi / 17) ^ 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1083_108365


namespace NUMINAMATH_CALUDE_loan_amount_correct_l1083_108337

/-- The amount of money (in Rs.) that A lent to B -/
def loan_amount : ℝ := 3500

/-- B's net interest rate per annum (as a decimal) -/
def net_interest_rate : ℝ := 0.01

/-- B's gain in 3 years (in Rs.) -/
def gain_in_three_years : ℝ := 105

/-- Proves that the loan amount is correct given the conditions -/
theorem loan_amount_correct : 
  loan_amount * net_interest_rate * 3 = gain_in_three_years :=
by sorry

end NUMINAMATH_CALUDE_loan_amount_correct_l1083_108337


namespace NUMINAMATH_CALUDE_function_range_l1083_108343

theorem function_range (a : ℝ) (h_a : a ≠ 0) :
  (∀ x : ℝ, a * Real.sin x - 1/2 * Real.cos (2*x) + a - 3/a + 1/2 ≤ 0) →
  (0 < a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l1083_108343


namespace NUMINAMATH_CALUDE_x_plus_y_equals_two_l1083_108374

theorem x_plus_y_equals_two (x y : ℝ) (h : |x - 6| + (y + 4)^2 = 0) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_two_l1083_108374


namespace NUMINAMATH_CALUDE_equation_solution_l1083_108301

theorem equation_solution (c d : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + 15 = 27 ∧ (x = c ∨ x = d)) →
  c ≥ d →
  3*c - d = 6 + 4*Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1083_108301


namespace NUMINAMATH_CALUDE_bottle_caps_added_l1083_108346

theorem bottle_caps_added (initial_caps : ℕ) (final_caps : ℕ) (added_caps : ℕ) : 
  initial_caps = 7 → final_caps = 14 → added_caps = final_caps - initial_caps → added_caps = 7 :=
by sorry

end NUMINAMATH_CALUDE_bottle_caps_added_l1083_108346


namespace NUMINAMATH_CALUDE_line_through_circle_center_l1083_108383

/-- The value of 'a' when the line 3x + y + a = 0 passes through the center of the circle x^2 + y^2 + 2x - 4y = 0 -/
theorem line_through_circle_center (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 ∧ 3*x + y + a = 0 ∧ 
   ∀ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' = 0 → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l1083_108383


namespace NUMINAMATH_CALUDE_basketball_free_throws_l1083_108354

theorem basketball_free_throws 
  (two_point_shots : ℕ) 
  (three_point_shots : ℕ) 
  (free_throws : ℕ) : 
  (3 * three_point_shots = 2 * two_point_shots) → 
  (free_throws = two_point_shots + 1) → 
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 84) → 
  free_throws = 16 := by
sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l1083_108354


namespace NUMINAMATH_CALUDE_drop_1m_l1083_108395

def water_level_change (change : ℝ) : ℝ := change

axiom rise_positive (x : ℝ) : x > 0 → water_level_change x > 0
axiom rise_4m : water_level_change 4 = 4

theorem drop_1m : water_level_change (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_drop_1m_l1083_108395


namespace NUMINAMATH_CALUDE_binomial_16_13_l1083_108322

theorem binomial_16_13 : Nat.choose 16 13 = 560 := by
  sorry

end NUMINAMATH_CALUDE_binomial_16_13_l1083_108322


namespace NUMINAMATH_CALUDE_gaussian_function_properties_l1083_108311

-- Define the Gaussian function (floor function)
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem statement
theorem gaussian_function_properties :
  -- 1. The range of floor is ℤ
  (∀ n : ℤ, ∃ x : ℝ, floor x = n) ∧
  -- 2. floor is not an odd function
  (∃ x : ℝ, floor (-x) ≠ -floor x) ∧
  -- 3. x - floor x is periodic with period 1
  (∀ x : ℝ, x - floor x = (x + 1) - floor (x + 1)) ∧
  -- 4. floor is not monotonically increasing on ℝ
  (∃ x y : ℝ, x < y ∧ floor x > floor y) :=
by sorry

end NUMINAMATH_CALUDE_gaussian_function_properties_l1083_108311


namespace NUMINAMATH_CALUDE_cone_base_diameter_l1083_108351

theorem cone_base_diameter (r : ℝ) (h1 : r > 0) : 
  (π * r^2 + π * r * (2 * r) = 3 * π) → 2 * r = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l1083_108351


namespace NUMINAMATH_CALUDE_equation_solution_l1083_108308

theorem equation_solution (x : ℝ) :
  x ≠ -1 → x ≠ 1 → (x / (x + 1) = 2 / (x^2 - 1)) → x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1083_108308


namespace NUMINAMATH_CALUDE_line_equation_60_degrees_l1083_108327

theorem line_equation_60_degrees (x y : ℝ) :
  let angle : ℝ := 60 * π / 180  -- Convert 60° to radians
  let slope : ℝ := Real.tan angle
  let y_intercept : ℝ := -1
  (slope * x - y - y_intercept = 0) ↔ (Real.sqrt 3 * x - y - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_60_degrees_l1083_108327
