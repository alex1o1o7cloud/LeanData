import Mathlib

namespace NUMINAMATH_CALUDE_sin_90_degrees_l3968_396889

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l3968_396889


namespace NUMINAMATH_CALUDE_linear_equation_solve_l3968_396836

theorem linear_equation_solve (x y : ℝ) : 
  x + 2 * y = 6 → y = (-x + 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solve_l3968_396836


namespace NUMINAMATH_CALUDE_job_completion_time_l3968_396871

/-- Given that:
    - A can do a job in 45 days
    - A and B working together can finish 4 times the amount of work in 72 days
    Prove that B can do the job alone in 30 days -/
theorem job_completion_time (a_time : ℝ) (combined_time : ℝ) (combined_work : ℝ) (b_time : ℝ) :
  a_time = 45 →
  combined_time = 72 →
  combined_work = 4 →
  (1 / a_time + 1 / b_time) * combined_time = combined_work →
  b_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3968_396871


namespace NUMINAMATH_CALUDE_least_multiplier_for_72_l3968_396866

theorem least_multiplier_for_72 (n : ℕ) : n = 62087668 ↔ 
  n > 0 ∧
  (∀ m : ℕ, m > 0 → m < n →
    (¬(112 ∣ (72 * m)) ∨
     ¬(199 ∣ (72 * m)) ∨
     ¬∃ k : ℕ, 72 * m = k * k)) ∧
  (112 ∣ (72 * n)) ∧
  (199 ∣ (72 * n)) ∧
  ∃ k : ℕ, 72 * n = k * k :=
sorry

end NUMINAMATH_CALUDE_least_multiplier_for_72_l3968_396866


namespace NUMINAMATH_CALUDE_linear_function_theorem_l3968_396828

/-- A linear function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem linear_function_theorem (x : ℝ) :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) →  -- f is linear
  (∀ x : ℝ, f x = 3 * (f⁻¹ x) + 5) →        -- f(x) = 3f^(-1)(x) + 5
  f 0 = 3 →                                 -- f(0) = 3
  f 3 = 3 * Real.sqrt 3 + 3 :=               -- f(3) = 3√3 + 3
by sorry

end NUMINAMATH_CALUDE_linear_function_theorem_l3968_396828


namespace NUMINAMATH_CALUDE_restaurant_tables_difference_l3968_396877

theorem restaurant_tables_difference (total_tables : ℕ) (total_capacity : ℕ) 
  (new_table_capacity : ℕ) (original_table_capacity : ℕ) :
  total_tables = 40 →
  total_capacity = 212 →
  new_table_capacity = 6 →
  original_table_capacity = 4 →
  ∃ (new_tables original_tables : ℕ),
    new_tables + original_tables = total_tables ∧
    new_table_capacity * new_tables + original_table_capacity * original_tables = total_capacity ∧
    new_tables - original_tables = 12 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_tables_difference_l3968_396877


namespace NUMINAMATH_CALUDE_maple_trees_equation_l3968_396881

/-- The number of maple trees initially in the park -/
def initial_maple_trees : ℕ := 2

/-- The number of maple trees planted -/
def planted_maple_trees : ℕ := 9

/-- The final number of maple trees after planting -/
def final_maple_trees : ℕ := 11

/-- Theorem stating that the initial number of maple trees plus the planted ones equals the final number -/
theorem maple_trees_equation : 
  initial_maple_trees + planted_maple_trees = final_maple_trees := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_equation_l3968_396881


namespace NUMINAMATH_CALUDE_coordinates_wrt_y_axis_l3968_396803

/-- Given a point A(x,y) in a 2D coordinate system, this theorem states that 
    the coordinates of A with respect to the y-axis are (-x,y) -/
theorem coordinates_wrt_y_axis (x y : ℝ) : 
  let A : ℝ × ℝ := (x, y)
  let A_wrt_y : ℝ × ℝ := (-x, y)
  (∀ p : ℝ × ℝ, p.1 = 0 → (A.1 - p.1)^2 + (A.2 - p.2)^2 = (A_wrt_y.1 - p.1)^2 + (A_wrt_y.2 - p.2)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_y_axis_l3968_396803


namespace NUMINAMATH_CALUDE_tooth_extraction_cost_l3968_396875

def cleaning_cost : ℕ := 70
def filling_cost : ℕ := 120
def total_fillings : ℕ := 2
def total_bill_factor : ℕ := 5

theorem tooth_extraction_cost :
  let total_bill := filling_cost * total_bill_factor
  let cleaning_and_fillings_cost := cleaning_cost + (filling_cost * total_fillings)
  total_bill - cleaning_and_fillings_cost = 290 :=
by sorry

end NUMINAMATH_CALUDE_tooth_extraction_cost_l3968_396875


namespace NUMINAMATH_CALUDE_painting_areas_l3968_396849

/-- Represents the areas painted in square decimeters -/
structure PaintedAreas where
  blue : ℝ
  green : ℝ
  yellow : ℝ

/-- The total amount of each paint color available in square decimeters -/
def total_paint : ℝ := 38

/-- Theorem stating the correct areas given the painting conditions -/
theorem painting_areas : ∃ (areas : PaintedAreas),
  -- All paint is used
  areas.blue + areas.yellow + areas.green = 2 * total_paint ∧
  -- Green paint mixture ratio
  areas.green = (2 * areas.yellow + areas.blue) / 3 ∧
  -- Grass area is 6 more than sky area
  areas.green = areas.blue + 6 ∧
  -- Correct areas
  areas.blue = 27 ∧
  areas.green = 33 ∧
  areas.yellow = 16 := by
  sorry

end NUMINAMATH_CALUDE_painting_areas_l3968_396849


namespace NUMINAMATH_CALUDE_parallel_tangents_slope_l3968_396839

noncomputable def y₁ (x : ℝ) : ℝ := 2 * Real.sin x

noncomputable def y₂ (x : ℝ) : ℝ := 2 * Real.sqrt x * (x / 3 + 1)

noncomputable def y₁_deriv (x : ℝ) : ℝ := 2 * Real.cos x

noncomputable def y₂_deriv (x : ℝ) : ℝ := Real.sqrt x + 1 / Real.sqrt x

def x_range (x : ℝ) : Prop := 0 ≤ x ∧ x < 2 * Real.pi

theorem parallel_tangents_slope (x₁ x₂ : ℝ) 
  (h₁ : x_range x₁) 
  (h₂ : x₂ > 0) 
  (h_parallel : y₁_deriv x₁ = y₂_deriv x₂) : 
  (y₂ x₂ - y₁ x₁) / (x₂ - x₁) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_slope_l3968_396839


namespace NUMINAMATH_CALUDE_four_divides_sum_of_squares_l3968_396801

theorem four_divides_sum_of_squares (a b c : ℕ+) :
  4 ∣ (a^2 + b^2 + c^2) ↔ (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) := by
  sorry

end NUMINAMATH_CALUDE_four_divides_sum_of_squares_l3968_396801


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l3968_396852

theorem wire_ratio_proof (total_length longer_length shorter_length : ℕ) 
  (h1 : total_length = 80)
  (h2 : shorter_length = 30)
  (h3 : longer_length = total_length - shorter_length) :
  Nat.gcd shorter_length longer_length * 3 = shorter_length ∧
  Nat.gcd shorter_length longer_length * 5 = longer_length :=
by sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l3968_396852


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l3968_396885

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ n

theorem fifth_term_of_sequence (x : ℝ) :
  let a₁ : ℝ := 4
  let a₂ : ℝ := 12 * x^2
  let a₃ : ℝ := 36 * x^4
  let a₄ : ℝ := 108 * x^6
  let r : ℝ := 3 * x^2
  geometric_sequence a₁ r 4 = 324 * x^8 :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l3968_396885


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3968_396854

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope m passing through point (x₀, y₀) -/
structure Line where
  m : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- Theorem about an ellipse with specific properties and its intersection with a line -/
theorem ellipse_intersection_theorem (C : Ellipse) (l : Line) :
  C.a^2 = 12 ∧ C.b = 2 ∧ (C.a^2 - C.b^2 = 8) ∧ 
  l.m = 1 ∧ l.x₀ = -2 ∧ l.y₀ = 1 →
  (∃ A B : ℝ × ℝ,
    (A.1^2 / 12 + A.2^2 / 4 = 1) ∧
    (B.1^2 / 12 + B.2^2 / 4 = 1) ∧
    (A.2 = A.1 + 3) ∧
    (B.2 = B.1 + 3) ∧
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 42 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3968_396854


namespace NUMINAMATH_CALUDE_minimum_m_value_l3968_396833

noncomputable def f (x : ℝ) : ℝ := Real.log x + (2*x + 1) / x

theorem minimum_m_value (m : ℤ) :
  (∃ x : ℝ, x > 1 ∧ f x < (m * (x - 1) + 2) / x) →
  m ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_minimum_m_value_l3968_396833


namespace NUMINAMATH_CALUDE_actions_probability_is_one_four_hundredth_l3968_396822

/-- The probability of selecting specific letters from given words -/
def select_probability (total : ℕ) (choose : ℕ) (specific : ℕ) : ℚ :=
  (specific : ℚ) / (Nat.choose total choose : ℚ)

/-- The probability of selecting all letters from ACTIONS -/
def actions_probability : ℚ :=
  (select_probability 5 3 1) * (select_probability 5 2 1) * (select_probability 4 1 1)

/-- Theorem stating the probability of selecting all letters from ACTIONS -/
theorem actions_probability_is_one_four_hundredth :
  actions_probability = 1 / 400 := by sorry

end NUMINAMATH_CALUDE_actions_probability_is_one_four_hundredth_l3968_396822


namespace NUMINAMATH_CALUDE_symmetry_implies_periodicity_l3968_396887

/-- A function is symmetric about a line x = a -/
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

/-- A function is symmetric about a point (m, n) -/
def SymmetricAboutPoint (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  ∀ x, 2 * n - f x = f (2 * m - x)

/-- A function is periodic with period p -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem symmetry_implies_periodicity
  (f : ℝ → ℝ) (a m n : ℝ) (ha : a ≠ 0) (hm : m ≠ a)
  (h_line : SymmetricAboutLine f a)
  (h_point : SymmetricAboutPoint f m n) :
  IsPeriodic f (4 * (m - a)) :=
sorry

end NUMINAMATH_CALUDE_symmetry_implies_periodicity_l3968_396887


namespace NUMINAMATH_CALUDE_initial_population_theorem_l3968_396800

def village_population (P : ℕ) : Prop :=
  ⌊(P : ℝ) * 0.95 * 0.80⌋ = 3553

theorem initial_population_theorem :
  ∃ P : ℕ, village_population P ∧ P ≥ 4678 ∧ P < 4679 :=
sorry

end NUMINAMATH_CALUDE_initial_population_theorem_l3968_396800


namespace NUMINAMATH_CALUDE_certain_number_proof_l3968_396892

theorem certain_number_proof (x : ℚ) (n : ℚ) : 
  x = 6 → 9 - (4/x) = n + (8/x) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3968_396892


namespace NUMINAMATH_CALUDE_provision_duration_l3968_396860

theorem provision_duration 
  (initial_soldiers : ℕ) 
  (initial_consumption : ℚ) 
  (new_soldiers : ℕ) 
  (new_consumption : ℚ) 
  (new_duration : ℕ) 
  (h1 : initial_soldiers = 1200)
  (h2 : initial_consumption = 3)
  (h3 : new_soldiers = 1728)
  (h4 : new_consumption = 5/2)
  (h5 : new_duration = 25) : 
  ∃ (initial_duration : ℕ), 
    initial_duration = 30 ∧ 
    (initial_soldiers : ℚ) * initial_consumption * initial_duration = 
    (new_soldiers : ℚ) * new_consumption * new_duration :=
by sorry

end NUMINAMATH_CALUDE_provision_duration_l3968_396860


namespace NUMINAMATH_CALUDE_one_eighth_percent_of_800_l3968_396899

theorem one_eighth_percent_of_800 : (1 / 8 * (1 / 100) * 800 : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_percent_of_800_l3968_396899


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3968_396802

/-- Given a geometric sequence {a_n} where a_5 = 4, prove that a_3a_7 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_a5 : a 5 = 4) : a 3 * a 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3968_396802


namespace NUMINAMATH_CALUDE_brick_length_calculation_l3968_396867

/-- Calculates the length of a brick given wall and brick specifications --/
theorem brick_length_calculation (wall_length wall_width wall_height : ℝ)
  (mortar_percentage : ℝ) (brick_count : ℕ) (brick_width brick_height : ℝ) :
  wall_length = 10 ∧ wall_width = 4 ∧ wall_height = 5 ∧
  mortar_percentage = 0.1 ∧ brick_count = 6000 ∧
  brick_width = 15 ∧ brick_height = 8 →
  ∃ (brick_length : ℝ),
    brick_length = 250 ∧
    (wall_length * wall_width * wall_height * (1 - mortar_percentage) * 1000000) =
    (brick_length * brick_width * brick_height * brick_count) :=
by sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l3968_396867


namespace NUMINAMATH_CALUDE_exponent_calculation_l3968_396851

theorem exponent_calculation (a n m k : ℝ) 
  (h1 : a^n = 2) 
  (h2 : a^m = 3) 
  (h3 : a^k = 4) : 
  a^(2*n + m - 2*k) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_exponent_calculation_l3968_396851


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3968_396844

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is increasing on a set S if f(x) ≤ f(y) for all x, y in S with x ≤ y -/
def IncreasingOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (h_odd : OddFunction f)
  (h_incr : IncreasingOn f (Set.Iic 0))
  (h_f2 : f 2 = 4) :
  {x : ℝ | 4 + f (x^2 - x) > 0} = Set.univ :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3968_396844


namespace NUMINAMATH_CALUDE_managers_salary_l3968_396840

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 150 →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase) / (num_employees + 1) - avg_salary = salary_increase →
  (num_employees + 1) * ((num_employees * avg_salary + (num_employees + 1) * salary_increase) / (num_employees + 1)) - num_employees * avg_salary = 4650 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_l3968_396840


namespace NUMINAMATH_CALUDE_class_size_with_error_l3968_396863

/-- Represents a class with a marking error -/
structure ClassWithError where
  n : ℕ  -- number of pupils
  S : ℕ  -- correct sum of marks
  wrong_mark : ℕ  -- wrongly entered mark
  correct_mark : ℕ  -- correct mark

/-- The conditions of the problem -/
def problem_conditions (c : ClassWithError) : Prop :=
  c.wrong_mark = 79 ∧
  c.correct_mark = 45 ∧
  (c.S + (c.wrong_mark - c.correct_mark)) / c.n = 3/2 * (c.S / c.n)

/-- The theorem stating the solution -/
theorem class_size_with_error (c : ClassWithError) :
  problem_conditions c → c.n = 68 :=
by sorry

end NUMINAMATH_CALUDE_class_size_with_error_l3968_396863


namespace NUMINAMATH_CALUDE_unused_sector_angle_l3968_396804

/-- Given a circular piece of paper with radius 20 cm, from which a sector is removed
    to form a cone with radius 15 cm and volume 900π cubic cm,
    prove that the measure of the angle of the unused sector is 90°. -/
theorem unused_sector_angle (r_paper : ℝ) (r_cone : ℝ) (v_cone : ℝ) :
  r_paper = 20 →
  r_cone = 15 →
  v_cone = 900 * Real.pi →
  ∃ (h : ℝ) (s : ℝ),
    v_cone = (1/3) * Real.pi * r_cone^2 * h ∧
    s^2 = r_cone^2 + h^2 ∧
    s ≤ r_paper ∧
    (2 * Real.pi * r_cone) / (2 * Real.pi * r_paper) * 360 = 270 :=
by sorry

end NUMINAMATH_CALUDE_unused_sector_angle_l3968_396804


namespace NUMINAMATH_CALUDE_negation_of_implication_l3968_396861

theorem negation_of_implication (a b c : ℝ) :
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3968_396861


namespace NUMINAMATH_CALUDE_log_function_passes_through_point_l3968_396821

-- Define the logarithm function for any base a > 0 and a ≠ 1
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = log_a(x-1) + 2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x - 1) + 2

-- Theorem statement
theorem log_function_passes_through_point (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  f a 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_function_passes_through_point_l3968_396821


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_center_l3968_396838

/-- Given two circles C₁ and C₂, prove that the trajectory of the center of a moving circle
    that is tangent to both C₁ and C₂ forms an ellipse with a specific equation. -/
theorem trajectory_of_moving_circle_center
  (C₁ : ∀ x y : ℝ, (x - 4)^2 + y^2 = 169)
  (C₂ : ∀ x y : ℝ, (x + 4)^2 + y^2 = 9)
  (moving_circle_inside_C₁ : True)
  (moving_circle_tangent_C₁_inside : True)
  (moving_circle_tangent_C₂_outside : True) :
  ∃ M : ℝ × ℝ, (M.1^2 / 64 + M.2^2 / 48 = 1) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_center_l3968_396838


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l3968_396809

/-- Given a wire cut into two pieces of lengths x and y, where x forms a square and y forms a regular octagon with equal perimeters, prove that x/y = 1 -/
theorem wire_cut_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_square : 4 * (x / 4) = x) 
  (h_octagon : 8 * (y / 8) = y)
  (h_equal_perimeter : 4 * (x / 4) = 8 * (y / 8)) : 
  x / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l3968_396809


namespace NUMINAMATH_CALUDE_solution_set_implies_m_value_l3968_396823

theorem solution_set_implies_m_value (m : ℝ) 
  (h : ∀ x : ℝ, x - m > 5 ↔ x > 2) : 
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_value_l3968_396823


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3968_396898

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 2) : 
  ((2 * sin α - 3 * cos α) / (4 * sin α - 9 * cos α) = -1) ∧ 
  (4 * sin α ^ 2 - 3 * sin α * cos α - 5 * cos α ^ 2 = 1) := by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3968_396898


namespace NUMINAMATH_CALUDE_crackers_per_friend_l3968_396896

theorem crackers_per_friend (initial_crackers : ℕ) (friends : ℕ) (remaining_crackers : ℕ) :
  initial_crackers = 23 →
  friends = 2 →
  remaining_crackers = 11 →
  (initial_crackers - remaining_crackers) / friends = 6 :=
by sorry

end NUMINAMATH_CALUDE_crackers_per_friend_l3968_396896


namespace NUMINAMATH_CALUDE_unfair_coin_expected_value_l3968_396810

/-- The expected value of an unfair coin flip -/
theorem unfair_coin_expected_value :
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let gain_heads : ℚ := 5
  let loss_tails : ℚ := 9
  p_heads * gain_heads + p_tails * (-loss_tails) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_unfair_coin_expected_value_l3968_396810


namespace NUMINAMATH_CALUDE_log_one_over_twenty_five_base_five_l3968_396812

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_over_twenty_five_base_five : log 5 (1/25) = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_one_over_twenty_five_base_five_l3968_396812


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3968_396897

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 6 < 0} = Set.Ioo (-3 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3968_396897


namespace NUMINAMATH_CALUDE_ratio_of_arithmetic_sums_l3968_396876

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem ratio_of_arithmetic_sums : 
  let n₁ := (60 - 4) / 4 + 1
  let n₂ := (72 - 6) / 6 + 1
  (arithmetic_sum 4 4 n₁) / (arithmetic_sum 6 6 n₂) = 40 / 39 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_arithmetic_sums_l3968_396876


namespace NUMINAMATH_CALUDE_min_value_theorem_l3968_396848

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + a*b + a*c + b*c = 6 + 2 * Real.sqrt 5) :
  3*a + b + 2*c ≥ 2 * Real.sqrt 10 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3968_396848


namespace NUMINAMATH_CALUDE_precious_stones_count_l3968_396826

theorem precious_stones_count (N : ℕ) (W : ℝ) : 
  (N > 0) →
  (W > 0) →
  (0.35 * W = 3 * (W / N)) →
  (5/13 * (0.65 * W) = 3 * ((0.65 * W) / (N - 3))) →
  N = 10 := by
sorry

end NUMINAMATH_CALUDE_precious_stones_count_l3968_396826


namespace NUMINAMATH_CALUDE_larger_number_from_sum_and_difference_l3968_396891

theorem larger_number_from_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 40)
  (diff_eq : x - y = 6) :
  max x y = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_from_sum_and_difference_l3968_396891


namespace NUMINAMATH_CALUDE_problem_solution_l3968_396845

theorem problem_solution (x y : ℝ) 
  (h1 : 2 * x + y = 7) 
  (h2 : (x + y) / 3 = 1.6666666666666667) : 
  x + 2 * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3968_396845


namespace NUMINAMATH_CALUDE_windfall_percentage_increase_l3968_396811

theorem windfall_percentage_increase 
  (initial_balance : ℝ)
  (weekly_investment : ℝ)
  (weeks_in_year : ℕ)
  (final_balance : ℝ)
  (h1 : initial_balance = 250000)
  (h2 : weekly_investment = 2000)
  (h3 : weeks_in_year = 52)
  (h4 : final_balance = 885000) :
  let balance_before_windfall := initial_balance + weekly_investment * weeks_in_year
  let windfall := final_balance - balance_before_windfall
  (windfall / balance_before_windfall) * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_windfall_percentage_increase_l3968_396811


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_squared_l3968_396864

theorem square_plus_reciprocal_squared (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x^2) = 7 → x^4 + (1/x^4) = 47 := by
sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_squared_l3968_396864


namespace NUMINAMATH_CALUDE_manuscript_cost_theorem_l3968_396873

/-- The total cost to copy and bind multiple manuscripts. -/
def total_cost (num_copies : ℕ) (pages_per_copy : ℕ) (copy_cost_per_page : ℚ) (binding_cost_per_copy : ℚ) : ℚ :=
  num_copies * (pages_per_copy * copy_cost_per_page + binding_cost_per_copy)

/-- Theorem stating the total cost for the given manuscript copying and binding scenario. -/
theorem manuscript_cost_theorem :
  total_cost 10 400 (5 / 100) 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_theorem_l3968_396873


namespace NUMINAMATH_CALUDE_B_coords_when_A_on_y_axis_a_value_when_AB_parallel_x_axis_l3968_396820

-- Define points A and B in the Cartesian coordinate system
def A (a : ℝ) : ℝ × ℝ := (a + 1, -3)
def B (a : ℝ) : ℝ × ℝ := (3, 2 * a + 1)

-- Theorem 1: When A lies on the y-axis, B has coordinates (3, -1)
theorem B_coords_when_A_on_y_axis (a : ℝ) :
  A a = (0, -3) → B a = (3, -1) := by sorry

-- Theorem 2: When AB is parallel to x-axis, a = -2
theorem a_value_when_AB_parallel_x_axis (a : ℝ) :
  (A a).2 = (B a).2 → a = -2 := by sorry

end NUMINAMATH_CALUDE_B_coords_when_A_on_y_axis_a_value_when_AB_parallel_x_axis_l3968_396820


namespace NUMINAMATH_CALUDE_unripe_orange_harvest_l3968_396815

/-- The number of sacks of unripe oranges harvested per day -/
def daily_unripe_harvest : ℕ := 65

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- The total number of sacks of unripe oranges harvested over the harvest period -/
def total_unripe_harvest : ℕ := daily_unripe_harvest * harvest_days

theorem unripe_orange_harvest : total_unripe_harvest = 390 := by
  sorry

end NUMINAMATH_CALUDE_unripe_orange_harvest_l3968_396815


namespace NUMINAMATH_CALUDE_f_properties_l3968_396870

noncomputable def f (x : ℝ) := x^2 / Real.log x

theorem f_properties :
  let e := Real.exp 1
  ∀ x ∈ Set.Icc (Real.exp (1/4)) e,
    (∀ y ∈ Set.Icc (Real.exp (1/4)) e, f y ≤ f e) ∧
    (f (Real.sqrt e) ≤ f x) ∧
    (∃ t ∈ Set.Icc (2/(e^2)) (1/e), 
      (∃ x₁ ∈ Set.Icc (1/e) 1, t * f x₁ = x₁) ∧
      (∃ x₂ ∈ Set.Ioc 1 (e^2), t * f x₂ = x₂) ∧
      (∀ s < 2/(e^2), ¬∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (1/e) 1 ∪ Set.Ioc 1 (e^2) ∧
                                  x₂ ∈ Set.Icc (1/e) 1 ∪ Set.Ioc 1 (e^2) ∧
                                  s * f x₁ = x₁ ∧ s * f x₂ = x₂) ∧
      (∀ s ≥ 1/e, ¬∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (1/e) 1 ∪ Set.Ioc 1 (e^2) ∧
                                  x₂ ∈ Set.Icc (1/e) 1 ∪ Set.Ioc 1 (e^2) ∧
                                  s * f x₁ = x₁ ∧ s * f x₂ = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3968_396870


namespace NUMINAMATH_CALUDE_first_expression_value_l3968_396841

theorem first_expression_value (a : ℝ) (E : ℝ) : 
  a = 30 → 
  (E + (3 * a - 8)) / 2 = 79 → 
  E = 76 := by
sorry

end NUMINAMATH_CALUDE_first_expression_value_l3968_396841


namespace NUMINAMATH_CALUDE_expand_cube_difference_l3968_396893

theorem expand_cube_difference (x y : ℝ) : (x + y) * (x^2 - x*y + y^2) = x^3 + y^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_cube_difference_l3968_396893


namespace NUMINAMATH_CALUDE_set_equality_l3968_396869

theorem set_equality (A : Set ℕ) : 
  ({1, 3} : Set ℕ) ⊆ A ∧ ({1, 3} : Set ℕ) ∪ A = {1, 3, 5} → A = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3968_396869


namespace NUMINAMATH_CALUDE_maximize_quadrilateral_area_l3968_396832

/-- Given a rectangle ABCD with length 2 and width 1, and points E on AB and F on AD
    such that AE = 2AF, the area of quadrilateral CDFE is maximized when AF = 3/4,
    and the maximum area is 7/8 square units. -/
theorem maximize_quadrilateral_area (A B C D E F : ℝ × ℝ) :
  let rectangle_length : ℝ := 2
  let rectangle_width : ℝ := 1
  let ABCD_is_rectangle := 
    (A.1 = B.1 - rectangle_length) ∧ 
    (A.2 = D.2) ∧ 
    (B.2 = C.2) ∧ 
    (C.1 = D.1 + rectangle_length) ∧ 
    (A.2 = B.2 + rectangle_width)
  let E_on_AB := E.2 = A.2
  let F_on_AD := F.1 = A.1
  let AE_equals_2AF := E.1 - A.1 = 2 * (F.2 - A.2)
  let area_CDFE (x : ℝ) := 2 * x^2 - 3 * x + 2
  ABCD_is_rectangle → E_on_AB → F_on_AD → AE_equals_2AF →
    (∃ (x : ℝ), x = 3/4 ∧ 
      (∀ (y : ℝ), 0 ≤ y ∧ y ≤ 1 → area_CDFE y ≤ area_CDFE x) ∧
      area_CDFE x = 7/8) := by
  sorry

end NUMINAMATH_CALUDE_maximize_quadrilateral_area_l3968_396832


namespace NUMINAMATH_CALUDE_bees_12_feet_apart_l3968_396843

/-- Represents the position of a bee in 3D space -/
structure Position where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the movement cycle of a bee -/
structure MovementCycle where
  steps : List Position

/-- Calculates the position of a bee after a given number of steps -/
def beePosition (start : Position) (cycle : MovementCycle) (steps : ℕ) : Position :=
  sorry

/-- Calculates the distance between two positions -/
def distance (p1 p2 : Position) : ℝ :=
  sorry

/-- Determines the direction of movement for a bee given its current and next position -/
def movementDirection (current next : Position) : String :=
  sorry

/-- The theorem to be proved -/
theorem bees_12_feet_apart :
  ∀ (steps : ℕ),
  let start := Position.mk 0 0 0
  let cycleA := MovementCycle.mk [Position.mk 2 0 0, Position.mk 0 2 0]
  let cycleB := MovementCycle.mk [Position.mk 0 (-2) 1, Position.mk (-1) 0 0]
  let posA := beePosition start cycleA steps
  let posB := beePosition start cycleB steps
  let nextA := beePosition start cycleA (steps + 1)
  let nextB := beePosition start cycleB (steps + 1)
  distance posA posB = 12 →
  (∀ (s : ℕ), s < steps → distance (beePosition start cycleA s) (beePosition start cycleB s) < 12) →
  movementDirection posA nextA = "east" ∧ movementDirection posB nextB = "upwards" :=
sorry

end NUMINAMATH_CALUDE_bees_12_feet_apart_l3968_396843


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3968_396895

/-- Represents the size of each stratum in the population -/
structure StratumSize where
  under30 : ℕ
  between30and40 : ℕ
  over40 : ℕ

/-- Represents the sample size for each stratum -/
structure StratumSample where
  under30 : ℕ
  between30and40 : ℕ
  over40 : ℕ

/-- Calculates the stratified sample size for a given population and total sample size -/
def stratifiedSample (populationSize : ℕ) (sampleSize : ℕ) (strata : StratumSize) : StratumSample :=
  { under30 := sampleSize * strata.under30 / populationSize,
    between30and40 := sampleSize * strata.between30and40 / populationSize,
    over40 := sampleSize * strata.over40 / populationSize }

theorem stratified_sampling_theorem (populationSize : ℕ) (sampleSize : ℕ) (strata : StratumSize) :
  populationSize = 100 →
  sampleSize = 20 →
  strata.under30 = 20 →
  strata.between30and40 = 60 →
  strata.over40 = 20 →
  let sample := stratifiedSample populationSize sampleSize strata
  sample.under30 = 4 ∧ sample.between30and40 = 12 ∧ sample.over40 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3968_396895


namespace NUMINAMATH_CALUDE_hexagon_diagonals_from_vertex_l3968_396883

/-- The number of diagonals that can be drawn from one vertex of a hexagon -/
def diagonals_from_vertex_hexagon : ℕ := 3

/-- Theorem stating that the number of diagonals from one vertex of a hexagon is 3 -/
theorem hexagon_diagonals_from_vertex :
  diagonals_from_vertex_hexagon = 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_from_vertex_l3968_396883


namespace NUMINAMATH_CALUDE_prime_congruent_three_mod_four_divides_x_l3968_396807

theorem prime_congruent_three_mod_four_divides_x (p : ℕ) (x₀ y₀ : ℕ) :
  Prime p →
  p % 4 = 3 →
  x₀ > 0 →
  y₀ > 0 →
  (p + 2) * x₀^2 - (p + 1) * y₀^2 + p * x₀ + (p + 2) * y₀ = 1 →
  p ∣ x₀ := by
  sorry

end NUMINAMATH_CALUDE_prime_congruent_three_mod_four_divides_x_l3968_396807


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l3968_396880

/-- Given consecutive integers x, y, and z where x > y > z, 
    2x + 3y + 3z = 5y + 11, and z = 3, prove that x = 5 -/
theorem consecutive_integers_problem (x y z : ℤ) 
  (consecutive : (x = y + 1) ∧ (y = z + 1))
  (order : x > y ∧ y > z)
  (equation : 2*x + 3*y + 3*z = 5*y + 11)
  (z_value : z = 3) :
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l3968_396880


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l3968_396813

theorem smallest_prime_dividing_sum : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^11 + 5^13) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (3^11 + 5^13) → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l3968_396813


namespace NUMINAMATH_CALUDE_power_product_equality_l3968_396831

theorem power_product_equality (a b : ℝ) : (-a * b)^3 * (-3 * b)^2 = -9 * a^3 * b^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l3968_396831


namespace NUMINAMATH_CALUDE_rows_sum_equal_l3968_396882

def first_row : List ℕ := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 200]
def second_row : List ℕ := [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]

theorem rows_sum_equal : 
  (first_row.sum = second_row.sum + 155) := by sorry

end NUMINAMATH_CALUDE_rows_sum_equal_l3968_396882


namespace NUMINAMATH_CALUDE_parallel_segments_k_value_l3968_396847

/-- Given four points A(-3, 0), B(0, -3), X(0, 9), and Y(18, k) on a Cartesian plane,
    if segment AB is parallel to segment XY, then k = -9. -/
theorem parallel_segments_k_value (k : ℝ) : 
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (0, -3)
  let X : ℝ × ℝ := (0, 9)
  let Y : ℝ × ℝ := (18, k)
  (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1) →
  k = -9 := by
sorry

end NUMINAMATH_CALUDE_parallel_segments_k_value_l3968_396847


namespace NUMINAMATH_CALUDE_focus_of_specific_parabola_l3968_396862

/-- A parabola is defined by its quadratic equation coefficients -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (x, y) -/
def focus (p : Parabola) : ℝ × ℝ :=
  sorry

/-- Theorem: The focus of the parabola y = 9x^2 + 6x - 2 is at (-1/3, -107/36) -/
theorem focus_of_specific_parabola :
  let p : Parabola := { a := 9, b := 6, c := -2 }
  focus p = (-1/3, -107/36) := by
  sorry

end NUMINAMATH_CALUDE_focus_of_specific_parabola_l3968_396862


namespace NUMINAMATH_CALUDE_new_person_weight_l3968_396825

/-- Given a group of 8 people where one person weighing 55 kg is replaced,
    if the average weight increases by 2.5 kg, then the new person weighs 75 kg. -/
theorem new_person_weight (initial_total : ℝ) (new_weight : ℝ) : 
  (initial_total - 55 + new_weight) / 8 = initial_total / 8 + 2.5 →
  new_weight = 75 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l3968_396825


namespace NUMINAMATH_CALUDE_one_zero_in_interval_l3968_396814

def f (x : ℝ) := -x^2 + 8*x - 14

theorem one_zero_in_interval :
  ∃! x, x ∈ Set.Icc 2 5 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_one_zero_in_interval_l3968_396814


namespace NUMINAMATH_CALUDE_parabola_shift_l3968_396818

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (h v : ℝ) : Parabola where
  f := fun x => p.f (x + h) + v

theorem parabola_shift :
  let p : Parabola := ⟨fun x => x^2⟩
  let shifted := shift p 2 (-5)
  ∀ x, shifted.f x = (x + 2)^2 - 5 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_l3968_396818


namespace NUMINAMATH_CALUDE_prob_and_expectation_l3968_396846

variable (K N M : ℕ) (p : ℝ)

-- Probability that exactly M out of K items are known by at least one of N agents
def prob_exact_M_known : ℝ := 
  (Nat.choose K M : ℝ) * (1 - (1 - p)^N)^M * (1 - p)^(N * (K - M))

-- Expected number of items known by at least one agent
def expected_items_known : ℝ := K * (1 - (1 - p)^N)

-- Theorem statement
theorem prob_and_expectation (h_p : 0 ≤ p ∧ p ≤ 1) (h_K : K > 0) (h_N : N > 0) (h_M : M ≤ K) :
  (prob_exact_M_known K N M p = (Nat.choose K M : ℝ) * (1 - (1 - p)^N)^M * (1 - p)^(N * (K - M))) ∧
  (expected_items_known K N p = K * (1 - (1 - p)^N)) := by sorry

end NUMINAMATH_CALUDE_prob_and_expectation_l3968_396846


namespace NUMINAMATH_CALUDE_divisibility_by_12321_l3968_396835

theorem divisibility_by_12321 (a : ℤ) : 
  (∃ k : ℕ, 12321 ∣ (a^k + 1)) ↔ 
  (∃ n : ℤ, a ≡ 11 [ZMOD 111] ∨ 
            a ≡ 41 [ZMOD 111] ∨ 
            a ≡ 62 [ZMOD 111] ∨ 
            a ≡ 65 [ZMOD 111] ∨ 
            a ≡ 77 [ZMOD 111] ∨ 
            a ≡ 95 [ZMOD 111] ∨ 
            a ≡ 101 [ZMOD 111] ∨ 
            a ≡ 104 [ZMOD 111] ∨ 
            a ≡ 110 [ZMOD 111]) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_12321_l3968_396835


namespace NUMINAMATH_CALUDE_jordan_seven_miles_time_l3968_396859

/-- Jordan's running time for a given distance -/
def jordanTime (distance : ℝ) : ℝ := sorry

/-- Steve's running time for a given distance -/
def steveTime (distance : ℝ) : ℝ := sorry

/-- Theorem stating Jordan's time for 7 miles given the conditions -/
theorem jordan_seven_miles_time :
  (jordanTime 3 = 2 / 3 * steveTime 5) →
  (steveTime 5 = 40) →
  (∀ d₁ d₂ : ℝ, jordanTime d₁ / d₁ = jordanTime d₂ / d₂) →
  jordanTime 7 = 185 / 3 := by
  sorry

end NUMINAMATH_CALUDE_jordan_seven_miles_time_l3968_396859


namespace NUMINAMATH_CALUDE_solution_approximation_l3968_396872

def equation (x : ℝ) : Prop :=
  (0.66^3 - x^3) = 0.5599999999999999 * ((0.66^2) + 0.066 + x^2)

theorem solution_approximation : ∃ x : ℝ, equation x ∧ abs (x - 0.1) < 1e-6 := by
  sorry

end NUMINAMATH_CALUDE_solution_approximation_l3968_396872


namespace NUMINAMATH_CALUDE_lcm_of_385_and_180_l3968_396806

theorem lcm_of_385_and_180 :
  let a := 385
  let b := 180
  let hcf := 30
  Nat.lcm a b = 2310 :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_385_and_180_l3968_396806


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_97_l3968_396837

theorem factor_t_squared_minus_97 (t : ℝ) : t^2 - 97 = (t - Real.sqrt 97) * (t + Real.sqrt 97) := by sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_97_l3968_396837


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l3968_396817

theorem angle_in_second_quadrant (α : Real) (x : Real) :
  -- α is in the second quadrant
  π / 2 < α ∧ α < π →
  -- P(x,6) is on the terminal side of α
  x < 0 →
  -- sin α = 3/5
  Real.sin α = 3 / 5 →
  -- x = -8
  x = -8 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l3968_396817


namespace NUMINAMATH_CALUDE_four_points_reciprocal_sum_l3968_396855

theorem four_points_reciprocal_sum (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 
    1 / |x - a| + 1 / |x - b| + 1 / |x - c| + 1 / |x - d| ≤ 40 := by
  sorry

end NUMINAMATH_CALUDE_four_points_reciprocal_sum_l3968_396855


namespace NUMINAMATH_CALUDE_arithmetic_operations_l3968_396874

theorem arithmetic_operations : 
  (-3 : ℤ) + 2 = -1 ∧ (-3 : ℤ) * 2 = -6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l3968_396874


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3968_396834

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x - 3

-- State the theorem
theorem quadratic_function_properties :
  ∀ m : ℝ,
  (m > 0) →
  (∀ x : ℝ, f m x < 0 ↔ -1 < x ∧ x < 3) →
  (m = 1) ∧
  (∀ x : ℝ, 2 * x^2 - 4 * x + 3 > 2 * x - 1 ↔ x < 1 ∨ x > 2) ∧
  (∃ a : ℝ, 0 < a ∧ a < 1 ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f m (a^x) - 4 * a^(x+1) ≥ -4) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f m (a^x) - 4 * a^(x+1) = -4) ∧
    a = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3968_396834


namespace NUMINAMATH_CALUDE_min_value_theorem_l3968_396858

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 6 ∧ ∀ (x : ℝ), x = 3/(a-1) + 2/(b-1) → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3968_396858


namespace NUMINAMATH_CALUDE_no_perfect_square_in_range_l3968_396829

theorem no_perfect_square_in_range : 
  ∀ m : ℤ, 4 ≤ m ∧ m ≤ 12 → ¬∃ k : ℤ, 2 * m^2 + 3 * m + 2 = k^2 := by
  sorry

#check no_perfect_square_in_range

end NUMINAMATH_CALUDE_no_perfect_square_in_range_l3968_396829


namespace NUMINAMATH_CALUDE_set_operations_l3968_396884

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set Nat := {4, 5, 6, 7, 8, 9}
def B : Set Nat := {1, 2, 3, 4, 5, 6}

theorem set_operations :
  (A ∪ B = U) ∧
  (A ∩ B = {4, 5, 6}) ∧
  (U \ (A ∩ B) = {1, 2, 3, 7, 8, 9}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3968_396884


namespace NUMINAMATH_CALUDE_product_coefficient_sum_l3968_396888

theorem product_coefficient_sum (a b c d : ℝ) : 
  (∀ x, (4 * x^2 - 6 * x + 5) * (9 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  12 * a + 6 * b + 3 * c + d = -27 := by
sorry

end NUMINAMATH_CALUDE_product_coefficient_sum_l3968_396888


namespace NUMINAMATH_CALUDE_rope_average_length_l3968_396819

/-- Given 6 ropes where one third have an average length of 70 cm and the rest have an average length of 85 cm, prove that the overall average length is 80 cm. -/
theorem rope_average_length : 
  let total_ropes : ℕ := 6
  let third_ropes : ℕ := total_ropes / 3
  let remaining_ropes : ℕ := total_ropes - third_ropes
  let third_avg_length : ℝ := 70
  let remaining_avg_length : ℝ := 85
  let total_length : ℝ := (third_ropes : ℝ) * third_avg_length + (remaining_ropes : ℝ) * remaining_avg_length
  let overall_avg_length : ℝ := total_length / (total_ropes : ℝ)
  overall_avg_length = 80 := by
sorry

end NUMINAMATH_CALUDE_rope_average_length_l3968_396819


namespace NUMINAMATH_CALUDE_factorization_sum_l3968_396878

theorem factorization_sum (x y : ℝ) : 
  ∃ (a b c d e f g h j k : ℤ), 
    125 * x^9 - 216 * y^9 = (a*x + b*y) * (c*x^3 + d*x*y^2 + e*y^3) * (f*x + g*y) * (h*x^3 + j*x*y^2 + k*y^3) ∧
    a + b + c + d + e + f + g + h + j + k = 16 :=
by sorry

end NUMINAMATH_CALUDE_factorization_sum_l3968_396878


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3968_396827

/-- Given a parabola x^2 = 2py (p > 0) with a point (x, l) on the parabola
    such that the distance from this point to the focus is 3,
    prove that the distance from the focus to the directrix is 4. -/
theorem parabola_focus_directrix_distance
  (p : ℝ) (x l : ℝ) (h_p : p > 0) (h_parabola : x^2 = 2*p*l)
  (h_focus_distance : (x^2 + (l - p/2)^2) = 3^2) :
  p = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3968_396827


namespace NUMINAMATH_CALUDE_cube_root_and_square_root_l3968_396894

theorem cube_root_and_square_root (x y : ℝ) 
  (h1 : (x - 1) ^ (1/3 : ℝ) = 2) 
  (h2 : (y + 2) ^ (1/2 : ℝ) = 3) : 
  x - 2*y = -5 := by sorry

end NUMINAMATH_CALUDE_cube_root_and_square_root_l3968_396894


namespace NUMINAMATH_CALUDE_running_time_ratio_l3968_396879

theorem running_time_ratio (danny_time steve_time : ℝ) 
  (h1 : danny_time = 25)
  (h2 : steve_time / 2 + 12.5 = danny_time) : 
  danny_time / steve_time = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_running_time_ratio_l3968_396879


namespace NUMINAMATH_CALUDE_student_average_score_l3968_396808

theorem student_average_score (math physics chem : ℕ) : 
  math + physics = 32 →
  chem = physics + 20 →
  (math + chem) / 2 = 26 := by
sorry

end NUMINAMATH_CALUDE_student_average_score_l3968_396808


namespace NUMINAMATH_CALUDE_equal_numbers_exist_l3968_396850

/-- A quadratic polynomial function -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Theorem: Given a quadratic polynomial and real numbers l, t, v satisfying certain conditions,
    there exist at least two equal numbers among l, t, and v. -/
theorem equal_numbers_exist (a b c l t v : ℝ) (ha : a ≠ 0)
    (h1 : QuadraticPolynomial a b c l = t + v)
    (h2 : QuadraticPolynomial a b c t = l + v)
    (h3 : QuadraticPolynomial a b c v = l + t) :
    (l = t ∨ l = v ∨ t = v) := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_exist_l3968_396850


namespace NUMINAMATH_CALUDE_triangle_decomposition_l3968_396842

theorem triangle_decomposition (a b c : ℝ) 
  (h1 : b + c > a) (h2 : a + c > b) (h3 : a + b > c) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    a = y + z ∧ b = x + z ∧ c = x + y :=
by sorry

end NUMINAMATH_CALUDE_triangle_decomposition_l3968_396842


namespace NUMINAMATH_CALUDE_theater_pricing_l3968_396830

/-- The price of orchestra seats in dollars -/
def orchestra_price : ℝ := 12

/-- The total number of tickets sold -/
def total_tickets : ℕ := 380

/-- The total revenue in dollars -/
def total_revenue : ℝ := 3320

/-- The difference between balcony and orchestra tickets sold -/
def ticket_difference : ℕ := 240

/-- The price of balcony seats in dollars -/
def balcony_price : ℝ := 8

theorem theater_pricing :
  ∃ (orchestra_tickets : ℕ) (balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = total_tickets ∧
    balcony_tickets = orchestra_tickets + ticket_difference ∧
    orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_theater_pricing_l3968_396830


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3968_396853

theorem quadratic_one_solution (m : ℚ) : 
  (∃! x : ℚ, 3 * x^2 - 7 * x + m = 0) → m = 49/12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3968_396853


namespace NUMINAMATH_CALUDE_food_festival_total_cost_l3968_396816

def food_festival_cost (hot_dog_price1 hot_dog_price2 hot_dog_price3 : ℚ)
                       (ice_cream_price1 ice_cream_price2 : ℚ)
                       (lemonade_price1 lemonade_price2 lemonade_price3 : ℚ) : ℚ :=
  3 * hot_dog_price1 + 3 * hot_dog_price2 + 2 * hot_dog_price3 +
  2 * ice_cream_price1 + 3 * ice_cream_price2 +
  lemonade_price1 + lemonade_price2 + lemonade_price3

theorem food_festival_total_cost :
  food_festival_cost 0.60 0.75 0.90 1.50 2.00 2.50 3.00 3.50 = 23.85 := by
  sorry

end NUMINAMATH_CALUDE_food_festival_total_cost_l3968_396816


namespace NUMINAMATH_CALUDE_james_marbles_l3968_396890

theorem james_marbles (total_marbles : ℕ) (num_bags : ℕ) (marbles_per_bag : ℕ) :
  total_marbles = 28 →
  num_bags = 4 →
  marbles_per_bag * num_bags = total_marbles →
  total_marbles - marbles_per_bag = 21 :=
by sorry

end NUMINAMATH_CALUDE_james_marbles_l3968_396890


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3968_396868

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℝ) : Prop := ∃ m : ℝ, m * m = n

-- Define a function to check if a number is in its simplest quadratic radical form
def isSimplestQuadraticRadical (n : ℝ) : Prop :=
  n > 0 ∧ ¬(isPerfectSquare n) ∧ ∀ m : ℝ, m > 1 → ¬(isPerfectSquare (n / (m * m)))

-- Theorem statement
theorem simplest_quadratic_radical :
  isSimplestQuadraticRadical 6 ∧
  ¬(isSimplestQuadraticRadical 4) ∧
  ¬(isSimplestQuadraticRadical 0.5) ∧
  ¬(isSimplestQuadraticRadical 12) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3968_396868


namespace NUMINAMATH_CALUDE_smallest_area_of_2020th_square_l3968_396805

theorem smallest_area_of_2020th_square (n : ℕ) : 
  n > 0 →
  n^2 = 2019 + (n^2 - 2019) →
  (∀ i : Fin 2019, 1 = 1) →
  n^2 - 2019 ≠ 1 →
  n^2 - 2019 ≥ 6 ∧ 
  ∀ m : ℕ, m > 0 → m^2 = 2019 + (m^2 - 2019) → (∀ i : Fin 2019, 1 = 1) → m^2 - 2019 ≠ 1 → m^2 - 2019 ≥ n^2 - 2019 :=
by sorry

#check smallest_area_of_2020th_square

end NUMINAMATH_CALUDE_smallest_area_of_2020th_square_l3968_396805


namespace NUMINAMATH_CALUDE_common_value_proof_l3968_396856

theorem common_value_proof (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : 40 * a * b = 1800) :
  4 * a = 60 ∧ 5 * b = 60 := by
sorry

end NUMINAMATH_CALUDE_common_value_proof_l3968_396856


namespace NUMINAMATH_CALUDE_system_solution_l3968_396824

theorem system_solution (x y : ℝ) : 
  0 < x + y → 
  x + y ≠ 1 → 
  2*x - y ≠ 0 → 
  (x + y) * (2 ^ (y - 2*x)) = 6.25 → 
  (x + y) * (1 / (2*x - y)) = 5 → 
  x = 9 ∧ y = 16 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3968_396824


namespace NUMINAMATH_CALUDE_area_between_specific_lines_l3968_396865

/-- Line passing through two points -/
structure Line where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- Calculate the area between two lines within a given x-range -/
noncomputable def areaBetweenLines (l1 l2 : Line) (x_start x_end : ℝ) : ℝ :=
  sorry

/-- The problem statement -/
theorem area_between_specific_lines :
  let line1 : Line := { x1 := 0, y1 := 5, x2 := 10, y2 := 2 }
  let line2 : Line := { x1 := 2, y1 := 6, x2 := 6, y2 := 0 }
  areaBetweenLines line1 line2 2 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_between_specific_lines_l3968_396865


namespace NUMINAMATH_CALUDE_shirt_price_is_correct_l3968_396886

/-- The price of a shirt and sweater with given conditions -/
def shirt_price (total_cost sweater_price : ℝ) : ℝ :=
  let shirt_price := sweater_price - 7.43
  let discounted_sweater_price := sweater_price * 0.9
  shirt_price

theorem shirt_price_is_correct (total_cost sweater_price : ℝ) :
  total_cost = 80.34 ∧ 
  shirt_price total_cost sweater_price + sweater_price * 0.9 = total_cost →
  shirt_price total_cost sweater_price = 38.76 :=
by
  sorry

#eval shirt_price 80.34 46.19

end NUMINAMATH_CALUDE_shirt_price_is_correct_l3968_396886


namespace NUMINAMATH_CALUDE_rocky_day1_miles_l3968_396857

def rocky_training (day1 : ℝ) : Prop :=
  let day2 := 2 * day1
  let day3 := 3 * day2
  day1 + day2 + day3 = 36

theorem rocky_day1_miles : ∃ (x : ℝ), rocky_training x ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_rocky_day1_miles_l3968_396857
