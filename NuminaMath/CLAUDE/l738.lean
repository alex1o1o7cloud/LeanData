import Mathlib

namespace NUMINAMATH_CALUDE_equation_graph_is_axes_l738_73857

/-- The set of points (x, y) satisfying (x-y)^2 = x^2 + y^2 is equivalent to the union of the x-axis and y-axis -/
theorem equation_graph_is_axes (x y : ℝ) : 
  (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_graph_is_axes_l738_73857


namespace NUMINAMATH_CALUDE_bob_gardening_project_cost_l738_73824

/-- Calculates the total cost of a gardening project -/
def gardening_project_cost (num_rose_bushes : ℕ) (cost_per_rose_bush : ℕ) 
  (gardener_hourly_rate : ℕ) (hours_per_day : ℕ) (num_days : ℕ)
  (soil_volume : ℕ) (soil_cost_per_unit : ℕ) : ℕ :=
  num_rose_bushes * cost_per_rose_bush +
  gardener_hourly_rate * hours_per_day * num_days +
  soil_volume * soil_cost_per_unit

/-- The total cost of Bob's gardening project is $4100 -/
theorem bob_gardening_project_cost :
  gardening_project_cost 20 150 30 5 4 100 5 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_bob_gardening_project_cost_l738_73824


namespace NUMINAMATH_CALUDE_coffee_beans_cost_l738_73818

/-- Proves the amount spent on coffee beans given initial amount, cost of tumbler, and remaining amount -/
theorem coffee_beans_cost (initial_amount : ℕ) (tumbler_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 50 →
  tumbler_cost = 30 →
  remaining_amount = 10 →
  initial_amount - tumbler_cost - remaining_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_coffee_beans_cost_l738_73818


namespace NUMINAMATH_CALUDE_min_value_inequality_l738_73872

theorem min_value_inequality (a b c d : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 + (5/d - 1)^2 ≥ 5 * (5^(1/5) - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l738_73872


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l738_73814

theorem factorization_difference_of_squares (a : ℝ) : 4 * a^2 - 1 = (2*a + 1) * (2*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l738_73814


namespace NUMINAMATH_CALUDE_unique_positive_solution_l738_73848

theorem unique_positive_solution (n : ℕ) (hn : n > 1) :
  ∀ x : ℝ, x > 0 → (x^n - n*x + n - 1 = 0) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l738_73848


namespace NUMINAMATH_CALUDE_product_of_935421_and_625_l738_73846

theorem product_of_935421_and_625 : 935421 * 625 = 584638125 := by
  sorry

end NUMINAMATH_CALUDE_product_of_935421_and_625_l738_73846


namespace NUMINAMATH_CALUDE_third_batch_size_l738_73843

/-- Proves that given the conditions of the problem, the number of students in the third batch is 60 -/
theorem third_batch_size :
  let batch1_size : ℕ := 40
  let batch2_size : ℕ := 50
  let batch1_avg : ℚ := 45
  let batch2_avg : ℚ := 55
  let batch3_avg : ℚ := 65
  let total_avg : ℚ := 56333333333333336 / 1000000000000000
  let batch3_size : ℕ := 60
  (batch1_size * batch1_avg + batch2_size * batch2_avg + batch3_size * batch3_avg) / 
    (batch1_size + batch2_size + batch3_size) = total_avg :=
by sorry


end NUMINAMATH_CALUDE_third_batch_size_l738_73843


namespace NUMINAMATH_CALUDE_termite_ridden_homes_l738_73819

theorem termite_ridden_homes (total_homes : ℝ) (termite_ridden : ℝ) 
  (h1 : termite_ridden > 0) 
  (h2 : termite_ridden / total_homes ≤ 1) 
  (h3 : (3/4) * (termite_ridden / total_homes) = 1/4) : 
  termite_ridden / total_homes = 1/3 := by
sorry

end NUMINAMATH_CALUDE_termite_ridden_homes_l738_73819


namespace NUMINAMATH_CALUDE_series_sum_equals_closed_form_l738_73832

/-- The sum of the series Σ(n=1 to ∞) (-1)^(n+1)/(3n-2) -/
noncomputable def seriesSum : ℝ := ∑' n, ((-1 : ℝ)^(n+1)) / (3*n - 2)

/-- The closed form of the series sum -/
noncomputable def closedForm : ℝ := (1/3) * (Real.log 2 + 2 * Real.pi / Real.sqrt 3)

/-- Theorem stating that the series sum equals the closed form -/
theorem series_sum_equals_closed_form : seriesSum = closedForm := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_closed_form_l738_73832


namespace NUMINAMATH_CALUDE_ratio_problem_l738_73861

theorem ratio_problem (p q r s : ℚ) 
  (h1 : p / q = 8)
  (h2 : r / q = 5)
  (h3 : r / s = 3 / 4) :
  s^2 / p^2 = 25 / 36 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l738_73861


namespace NUMINAMATH_CALUDE_power_sum_theorem_l738_73816

theorem power_sum_theorem (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 ∧ 
  ∃ a b c d : ℝ, (a + b = c + d ∧ a^3 + b^3 = c^3 + d^3 ∧ a^4 + b^4 ≠ c^4 + d^4) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_theorem_l738_73816


namespace NUMINAMATH_CALUDE_number_of_teachers_l738_73840

/-- Represents the total number of people (teachers and students) in the school -/
def total : ℕ := 2400

/-- Represents the size of the stratified sample -/
def sample_size : ℕ := 160

/-- Represents the number of students in the sample -/
def students_in_sample : ℕ := 150

/-- Calculates the number of teachers in the school based on the given information -/
def teachers : ℕ := total - (total * students_in_sample / sample_size)

/-- Theorem stating that the number of teachers in the school is 150 -/
theorem number_of_teachers : teachers = 150 := by sorry

end NUMINAMATH_CALUDE_number_of_teachers_l738_73840


namespace NUMINAMATH_CALUDE_dogs_not_doing_anything_l738_73858

def total_dogs : ℕ := 500

def running_dogs : ℕ := (18 * total_dogs) / 100
def playing_dogs : ℕ := (3 * total_dogs) / 20
def barking_dogs : ℕ := (7 * total_dogs) / 100
def digging_dogs : ℕ := total_dogs / 10
def agility_dogs : ℕ := 12
def sleeping_dogs : ℕ := (2 * total_dogs) / 25
def eating_dogs : ℕ := total_dogs / 5

def dogs_doing_something : ℕ := 
  running_dogs + playing_dogs + barking_dogs + digging_dogs + 
  agility_dogs + sleeping_dogs + eating_dogs

theorem dogs_not_doing_anything : 
  total_dogs - dogs_doing_something = 98 := by sorry

end NUMINAMATH_CALUDE_dogs_not_doing_anything_l738_73858


namespace NUMINAMATH_CALUDE_sphere_cylinder_surface_area_difference_l738_73850

/-- The difference between the surface area of a sphere and the lateral surface area of its inscribed cylinder is zero. -/
theorem sphere_cylinder_surface_area_difference (R : ℝ) (R_pos : R > 0) : 
  4 * Real.pi * R^2 - (2 * Real.pi * R * (2 * R)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_surface_area_difference_l738_73850


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l738_73813

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 7)
  (x_ge_2 : x ≥ 2)
  (y_ge_2 : y ≥ 2)
  (z_ge_2 : z ≥ 2) :
  ∃ (max : ℝ), max = Real.sqrt 69 ∧ 
    ∀ a b c : ℝ, a + b + c = 7 → a ≥ 2 → b ≥ 2 → c ≥ 2 →
      Real.sqrt (2 * a + 3) + Real.sqrt (2 * b + 3) + Real.sqrt (2 * c + 3) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l738_73813


namespace NUMINAMATH_CALUDE_shirt_cost_l738_73849

theorem shirt_cost (initial_amount : ℕ) (num_shirts : ℕ) (pants_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 109 →
  num_shirts = 2 →
  pants_cost = 13 →
  remaining_amount = 74 →
  (initial_amount - remaining_amount - pants_cost) / num_shirts = 11 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l738_73849


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l738_73852

theorem simplify_nested_roots (a : ℝ) (ha : a > 0) :
  (((a^16)^(1/12))^(1/4))^6 * (((a^16)^(1/4))^(1/12))^3 = a^3 := by sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l738_73852


namespace NUMINAMATH_CALUDE_negative_cube_squared_l738_73887

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l738_73887


namespace NUMINAMATH_CALUDE_right_triangle_sum_squares_l738_73864

theorem right_triangle_sum_squares (A B C : ℝ × ℝ) : 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 →  -- Right triangle condition
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = ((A.1 - C.1)^2 + (A.2 - C.2)^2) →  -- BC is hypotenuse
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4 →  -- BC = 2
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.1 - C.1)^2 + (A.2 - C.2)^2 + (C.1 - B.1)^2 + (C.2 - B.2)^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sum_squares_l738_73864


namespace NUMINAMATH_CALUDE_bicycle_weight_is_12_l738_73856

-- Define the weight of a bicycle and a car
def bicycle_weight : ℝ := sorry
def car_weight : ℝ := sorry

-- State the theorem
theorem bicycle_weight_is_12 :
  (10 * bicycle_weight = 4 * car_weight) →
  (3 * car_weight = 90) →
  bicycle_weight = 12 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_weight_is_12_l738_73856


namespace NUMINAMATH_CALUDE_floor_area_approx_l738_73830

/-- The length of the floor in feet -/
def floor_length_ft : ℝ := 15

/-- The width of the floor in feet -/
def floor_width_ft : ℝ := 10

/-- The conversion factor from feet to meters -/
def ft_to_m : ℝ := 0.3048

/-- The area of the floor in square meters -/
def floor_area_m2 : ℝ := floor_length_ft * ft_to_m * floor_width_ft * ft_to_m

theorem floor_area_approx :
  ∃ ε > 0, abs (floor_area_m2 - 13.93) < ε :=
sorry

end NUMINAMATH_CALUDE_floor_area_approx_l738_73830


namespace NUMINAMATH_CALUDE_tangent_line_problem_l738_73870

theorem tangent_line_problem (a : ℝ) :
  (∃ l : Set (ℝ × ℝ),
    -- l is a line
    (∃ m k : ℝ, l = {(x, y) | y = m*x + k}) ∧
    -- l passes through (1,0)
    (1, 0) ∈ l ∧
    -- l is tangent to y = x^3
    (∃ x₀ y₀ : ℝ, (x₀, y₀) ∈ l ∧ y₀ = x₀^3 ∧ m = 3*x₀^2) ∧
    -- l is tangent to y = ax^2 + (15/4)x - 9
    (∃ x₁ y₁ : ℝ, (x₁, y₁) ∈ l ∧ y₁ = a*x₁^2 + (15/4)*x₁ - 9 ∧ m = 2*a*x₁ + 15/4)) →
  a = -25/64 ∨ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l738_73870


namespace NUMINAMATH_CALUDE_cycle_gain_percentage_l738_73807

def cycleA_cp : ℚ := 1000
def cycleB_cp : ℚ := 3000
def cycleC_cp : ℚ := 6000

def cycleB_discount_rate : ℚ := 10 / 100
def cycleC_tax_rate : ℚ := 5 / 100

def cycleA_sp : ℚ := 2000
def cycleB_sp : ℚ := 4500
def cycleC_sp : ℚ := 8000

def cycleA_sales_tax_rate : ℚ := 5 / 100
def cycleB_selling_discount_rate : ℚ := 8 / 100

def total_cp : ℚ := cycleA_cp + cycleB_cp * (1 - cycleB_discount_rate) + cycleC_cp * (1 + cycleC_tax_rate)
def total_sp : ℚ := cycleA_sp * (1 + cycleA_sales_tax_rate) + cycleB_sp * (1 - cycleB_selling_discount_rate) + cycleC_sp

def overall_gain : ℚ := total_sp - total_cp
def gain_percentage : ℚ := (overall_gain / total_cp) * 100

theorem cycle_gain_percentage :
  gain_percentage = 42.4 := by sorry

end NUMINAMATH_CALUDE_cycle_gain_percentage_l738_73807


namespace NUMINAMATH_CALUDE_cone_slant_height_l738_73875

/-- Given a cone with base radius 1 and lateral surface that unfolds into a semicircle,
    prove that its slant height is 2. -/
theorem cone_slant_height (r : ℝ) (s : ℝ) (h1 : r = 1) (h2 : π * s = 2 * π * r) : s = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l738_73875


namespace NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l738_73877

-- Part 1
theorem simplify_expression (x y : ℝ) : 3 * x^2 + 2 * x * y - 4 * y^2 - 3 * x * y + 4 * y^2 - 3 * x^2 = -x * y := by
  sorry

-- Part 2
theorem simplify_and_evaluate (a b : ℝ) (h1 : a = 2) (h2 : b = -3) : 
  a + (5 * a - 3 * b) - 2 * (a - 2 * b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l738_73877


namespace NUMINAMATH_CALUDE_range_of_a_l738_73865

/-- A function f : ℝ → ℝ is even -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f : ℝ → ℝ is increasing on [0, +∞) -/
def IsIncreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

/-- The condition that f(ax+1) ≤ f(x-2) holds for all x in [1/2, 1] -/
def Condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, 1/2 ≤ x → x ≤ 1 → f (a*x + 1) ≤ f (x - 2)

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
    (h1 : IsEven f)
    (h2 : IsIncreasingOnNonnegative f)
    (h3 : Condition f a) :
  -2 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l738_73865


namespace NUMINAMATH_CALUDE_area_preserved_l738_73844

-- Define the transformation
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - 1, p.2 + 2)

-- Define a quadrilateral as a set of four points in ℝ²
def Quadrilateral := Fin 4 → ℝ × ℝ

-- Define the area of a quadrilateral
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Define F and F'
def F : Quadrilateral := sorry
def F' : Quadrilateral := fun i => f (F i)

-- Theorem statement
theorem area_preserved (h : area F = 6) : area F' = area F := by sorry

end NUMINAMATH_CALUDE_area_preserved_l738_73844


namespace NUMINAMATH_CALUDE_min_sum_squares_roots_l738_73851

theorem min_sum_squares_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 4 * m * x₁ + 2 * m^2 + 3 * m - 2 = 0) →
  (2 * x₂^2 - 4 * m * x₂ + 2 * m^2 + 3 * m - 2 = 0) →
  (∀ m' : ℝ, ∃ x₁' x₂' : ℝ, 2 * x₁'^2 - 4 * m' * x₁' + 2 * m'^2 + 3 * m' - 2 = 0 ∧
                             2 * x₂'^2 - 4 * m' * x₂' + 2 * m'^2 + 3 * m' - 2 = 0) →
  x₁^2 + x₂^2 ≥ 8/9 ∧ (x₁^2 + x₂^2 = 8/9 ↔ m = 2/3) := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_roots_l738_73851


namespace NUMINAMATH_CALUDE_star_composition_l738_73881

-- Define the * operation
def star (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem star_composition (A B : Set α) : star A (star A B) = A ∩ B := by
  sorry

end NUMINAMATH_CALUDE_star_composition_l738_73881


namespace NUMINAMATH_CALUDE_fifth_score_calculation_l738_73845

theorem fifth_score_calculation (score1 score2 score3 score4 score5 : ℝ) :
  score1 = 85 ∧ score2 = 90 ∧ score3 = 87 ∧ score4 = 92 →
  (score1 + score2 + score3 + score4 + score5) / 5 = 89 →
  score5 = 91 := by
sorry

end NUMINAMATH_CALUDE_fifth_score_calculation_l738_73845


namespace NUMINAMATH_CALUDE_absolute_value_of_squared_negative_l738_73800

theorem absolute_value_of_squared_negative : |(-2)^2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_squared_negative_l738_73800


namespace NUMINAMATH_CALUDE_cubic_root_sum_reciprocal_squares_l738_73892

theorem cubic_root_sum_reciprocal_squares : 
  ∀ (α β γ : ℝ), 
    (α^3 - 15*α^2 + 26*α - 8 = 0) → 
    (β^3 - 15*β^2 + 26*β - 8 = 0) → 
    (γ^3 - 15*γ^2 + 26*γ - 8 = 0) → 
    α ≠ β → β ≠ γ → γ ≠ α →
    1/α^2 + 1/β^2 + 1/γ^2 = 916/64 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_reciprocal_squares_l738_73892


namespace NUMINAMATH_CALUDE_intersection_M_N_l738_73821

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}
def N : Set ℝ := {y | ∃ x ∈ (Set.Ioo 0 2), y = Real.log (2*x - x^2)}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l738_73821


namespace NUMINAMATH_CALUDE_dog_walking_distance_l738_73822

theorem dog_walking_distance (total_weekly_miles : ℝ) (dog1_daily_miles : ℝ) (days_per_week : ℕ) :
  total_weekly_miles = 70 ∧ 
  dog1_daily_miles = 2 ∧ 
  days_per_week = 7 →
  (total_weekly_miles - dog1_daily_miles * days_per_week) / days_per_week = 8 :=
by sorry

end NUMINAMATH_CALUDE_dog_walking_distance_l738_73822


namespace NUMINAMATH_CALUDE_square_difference_formula_l738_73811

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : x^2 - y^2 = 16 / 225 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l738_73811


namespace NUMINAMATH_CALUDE_number_calculation_l738_73847

theorem number_calculation (x : ℝ) : (0.1 * 0.3 * 0.5 * x = 90) → x = 6000 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l738_73847


namespace NUMINAMATH_CALUDE_b_in_terms_of_a_l738_73866

theorem b_in_terms_of_a (k : ℝ) (a b : ℝ) 
  (ha : a = 3 + 3^k) 
  (hb : b = 3 + 3^(-k)) : 
  b = (3*a - 8) / (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_b_in_terms_of_a_l738_73866


namespace NUMINAMATH_CALUDE_gray_area_calculation_l738_73859

theorem gray_area_calculation (r_small : ℝ) (r_large : ℝ) : 
  r_small = 2 →
  r_large = 3 * r_small →
  (π * r_large^2 - π * r_small^2) = 32 * π := by
  sorry

end NUMINAMATH_CALUDE_gray_area_calculation_l738_73859


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l738_73888

theorem binomial_square_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 16 * x + 16 = (r * x + s)^2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l738_73888


namespace NUMINAMATH_CALUDE_basketball_shooting_probability_unique_shot_probability_l738_73882

/-- The probability of passing a basketball shooting test -/
def pass_probability : ℝ := 0.784

/-- The probability of making a single shot -/
def shot_probability : ℝ := 0.4

/-- The number of shooting opportunities -/
def max_attempts : ℕ := 3

/-- Theorem stating that the given shot probability results in the specified pass probability -/
theorem basketball_shooting_probability :
  shot_probability + (1 - shot_probability) * shot_probability + 
  (1 - shot_probability)^2 * shot_probability = pass_probability := by
  sorry

/-- Theorem stating that the shot probability is the unique solution -/
theorem unique_shot_probability :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 →
  (p + (1 - p) * p + (1 - p)^2 * p = pass_probability) →
  p = shot_probability := by
  sorry

end NUMINAMATH_CALUDE_basketball_shooting_probability_unique_shot_probability_l738_73882


namespace NUMINAMATH_CALUDE_travel_ways_count_l738_73805

/-- The number of highways from A to B -/
def num_highways : ℕ := 3

/-- The number of railways from A to B -/
def num_railways : ℕ := 2

/-- The total number of ways to travel from A to B -/
def total_ways : ℕ := num_highways + num_railways

theorem travel_ways_count : total_ways = 5 := by
  sorry

end NUMINAMATH_CALUDE_travel_ways_count_l738_73805


namespace NUMINAMATH_CALUDE_periodic_odd_quadratic_function_properties_l738_73867

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f (-x) = -f x

def is_quadratic_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ A B C : ℝ, ∀ x, a ≤ x ∧ x ≤ b → f x = A * x^2 + B * x + C

theorem periodic_odd_quadratic_function_properties
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 5)
  (h_odd : is_odd_on_interval f (-1) 1)
  (h_quadratic : is_quadratic_on_interval f 1 4)
  (h_min : f 2 = -5 ∧ ∀ x, f x ≥ -5) :
  (f 1 + f 4 = 0) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 4 → f x = 2 * (x - 2)^2 - 5) :=
by sorry

end NUMINAMATH_CALUDE_periodic_odd_quadratic_function_properties_l738_73867


namespace NUMINAMATH_CALUDE_vector_operation_result_l738_73855

theorem vector_operation_result :
  let a : ℝ × ℝ × ℝ := (-3, 5, 2)
  let b : ℝ × ℝ × ℝ := (1, -1, 3)
  let c : ℝ × ℝ × ℝ := (2, 0, -4)
  a - 4 • b + c = (-5, 9, -14) :=
by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l738_73855


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l738_73838

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 2 + a 6 = 3) →
  (a 6 + a 10 = 12) →
  (a 8 + a 12 = 24) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l738_73838


namespace NUMINAMATH_CALUDE_total_consumption_theorem_l738_73802

/-- Represents the amount of liquid consumed by each person --/
structure Consumption where
  elijah : Float
  emilio : Float
  isabella : Float
  xavier_soda : Float
  xavier_fruit_punch : Float

/-- Converts pints to cups --/
def pints_to_cups (pints : Float) : Float := pints * 2

/-- Converts liters to cups --/
def liters_to_cups (liters : Float) : Float := liters * 4.22675

/-- Converts gallons to cups --/
def gallons_to_cups (gallons : Float) : Float := gallons * 16

/-- Calculates the total cups consumed based on the given consumption --/
def total_cups (c : Consumption) : Float :=
  c.elijah + c.emilio + c.isabella + c.xavier_soda + c.xavier_fruit_punch

/-- Theorem stating that the total cups consumed is equal to 80.68025 --/
theorem total_consumption_theorem (c : Consumption)
  (h1 : c.elijah = pints_to_cups 8.5)
  (h2 : c.emilio = pints_to_cups 9.5)
  (h3 : c.isabella = liters_to_cups 3)
  (h4 : c.xavier_soda = gallons_to_cups 2 * 0.6)
  (h5 : c.xavier_fruit_punch = gallons_to_cups 2 * 0.4) :
  total_cups c = 80.68025 := by
  sorry


end NUMINAMATH_CALUDE_total_consumption_theorem_l738_73802


namespace NUMINAMATH_CALUDE_volleyball_prob_l738_73880

-- Define the set of sports
inductive Sport
| Soccer
| Basketball
| Volleyball

-- Define the probability space
def sportProbabilitySpace : Type := Sport

-- Define the probability measure
axiom prob : sportProbabilitySpace → ℝ

-- Axioms for probability measure
axiom prob_nonneg : ∀ s : sportProbabilitySpace, 0 ≤ prob s
axiom prob_sum_one : (prob Sport.Soccer) + (prob Sport.Basketball) + (prob Sport.Volleyball) = 1

-- Axiom for equal probability of each sport
axiom equal_prob : prob Sport.Soccer = prob Sport.Basketball ∧ 
                   prob Sport.Basketball = prob Sport.Volleyball

-- Theorem: The probability of choosing volleyball is 1/3
theorem volleyball_prob : prob Sport.Volleyball = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_prob_l738_73880


namespace NUMINAMATH_CALUDE_catch_time_correct_l738_73894

/-- Represents the pursuit scenario between a smuggler and coast guard -/
structure Pursuit where
  initial_distance : ℝ
  initial_smuggler_speed : ℝ
  initial_coast_guard_speed : ℝ
  speed_change_time : ℝ
  new_speed_ratio : ℝ

/-- Calculates the time when the coast guard catches the smuggler -/
def catch_time (p : Pursuit) : ℝ :=
  sorry

/-- Theorem stating that the coast guard catches the smuggler after 6 hours and 36 minutes -/
theorem catch_time_correct (p : Pursuit) : 
  p.initial_distance = 15 ∧ 
  p.initial_smuggler_speed = 13 ∧ 
  p.initial_coast_guard_speed = 15 ∧
  p.speed_change_time = 3 ∧
  p.new_speed_ratio = 18/15 →
  catch_time p = 6 + 36/60 := by
  sorry

end NUMINAMATH_CALUDE_catch_time_correct_l738_73894


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l738_73835

/-- An arithmetic sequence with a_4 = 3 and a_12 = 19 has a common difference of 2 -/
theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- Arithmetic sequence condition
  a 4 = 3 →
  a 12 = 19 →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l738_73835


namespace NUMINAMATH_CALUDE_inverse_of_A_l738_73834

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem inverse_of_A :
  A⁻¹ = !![(-1 : ℝ), -3; -2, -5] := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l738_73834


namespace NUMINAMATH_CALUDE_total_distance_calculation_l738_73893

/-- Calculates the total distance driven by Darius, Julia, and Thomas in miles and kilometers -/
def total_distance (darius_miles : ℝ) (julia_miles : ℝ) (thomas_miles : ℝ) (detour_miles : ℝ) (km_per_mile : ℝ) : ℝ × ℝ :=
  let darius_total := darius_miles * 2 + detour_miles
  let julia_total := julia_miles * 2 + detour_miles
  let thomas_total := thomas_miles * 2
  let total_miles := darius_total + julia_total + thomas_total
  let total_km := total_miles * km_per_mile
  (total_miles, total_km)

theorem total_distance_calculation :
  total_distance 679 998 1205 120 1.60934 = (6004, 9665.73616) := by
  sorry

end NUMINAMATH_CALUDE_total_distance_calculation_l738_73893


namespace NUMINAMATH_CALUDE_tan_symmetry_cos_squared_plus_sin_min_value_l738_73869

-- Define the tangent function
noncomputable def tan (x : ℝ) := Real.tan x

-- Define the cosine function
noncomputable def cos (x : ℝ) := Real.cos x

-- Define the sine function
noncomputable def sin (x : ℝ) := Real.sin x

-- Proposition ①
theorem tan_symmetry (k : ℤ) :
  ∀ x : ℝ, tan (k * π / 2 + x) = -tan (k * π / 2 - x) :=
sorry

-- Proposition ④
theorem cos_squared_plus_sin_min_value :
  ∃ x : ℝ, ∀ y : ℝ, cos y ^ 2 + sin y ≥ cos x ^ 2 + sin x ∧ cos x ^ 2 + sin x = -1 :=
sorry

end NUMINAMATH_CALUDE_tan_symmetry_cos_squared_plus_sin_min_value_l738_73869


namespace NUMINAMATH_CALUDE_consecutive_color_draw_probability_l738_73876

def num_green_chips : ℕ := 4
def num_blue_chips : ℕ := 3
def num_red_chips : ℕ := 5
def total_chips : ℕ := num_green_chips + num_blue_chips + num_red_chips

theorem consecutive_color_draw_probability :
  (Nat.factorial 3 * Nat.factorial num_green_chips * Nat.factorial num_blue_chips * Nat.factorial num_red_chips) / 
  Nat.factorial total_chips = 1 / 4620 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_color_draw_probability_l738_73876


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l738_73833

theorem arctan_equation_solution (x : ℝ) :
  2 * Real.arctan (1/3) + Real.arctan (1/10) + Real.arctan (1/x) = π/4 → x = 37/3 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l738_73833


namespace NUMINAMATH_CALUDE_geometric_seq_arithmetic_property_l738_73873

/-- Given a geometric sequence with common ratio q, prove that if the sums S_m, S_n, and S_l
    form an arithmetic sequence, then for any natural number k, the terms a_{m+k}, a_{n+k},
    and a_{l+k} also form an arithmetic sequence. -/
theorem geometric_seq_arithmetic_property
  (a : ℝ) (q : ℝ) (m n l k : ℕ) :
  let a_seq : ℕ → ℝ := λ i => a * q ^ (i - 1)
  let S : ℕ → ℝ := λ i => if q = 1 then a * i else a * (1 - q^i) / (1 - q)
  (2 * S n = S m + S l) →
  2 * a_seq (n + k) = a_seq (m + k) + a_seq (l + k) :=
by sorry

end NUMINAMATH_CALUDE_geometric_seq_arithmetic_property_l738_73873


namespace NUMINAMATH_CALUDE_geometric_and_arithmetic_sequences_l738_73890

/-- A geometric sequence with a₁ = 2 and a₄ = 16 -/
def geometric_sequence (n : ℕ) : ℝ :=
  2 * (2 : ℝ) ^ (n - 1)

/-- An arithmetic sequence with b₃ = a₃ and b₅ = a₅ -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  -16 + 12 * (n - 1)

theorem geometric_and_arithmetic_sequences :
  (∀ n : ℕ, geometric_sequence n = 2^n) ∧
  (arithmetic_sequence 3 = geometric_sequence 3 ∧
   arithmetic_sequence 5 = geometric_sequence 5) ∧
  (∀ n : ℕ, arithmetic_sequence n = 12*n - 28) := by
  sorry

end NUMINAMATH_CALUDE_geometric_and_arithmetic_sequences_l738_73890


namespace NUMINAMATH_CALUDE_birthday_gift_cost_l738_73889

def boss_contribution : ℕ := 15
def todd_contribution : ℕ := 2 * boss_contribution
def remaining_employees : ℕ := 5
def employee_contribution : ℕ := 11

theorem birthday_gift_cost :
  boss_contribution + todd_contribution + (remaining_employees * employee_contribution) = 100 := by
  sorry

end NUMINAMATH_CALUDE_birthday_gift_cost_l738_73889


namespace NUMINAMATH_CALUDE_coin_stacking_arrangements_l738_73862

/-- Represents the number of ways to arrange n indistinguishable objects in k positions -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Represents the number of valid orientation arrangements -/
def validOrientations : ℕ := 9

/-- Represents the total number of gold coins -/
def goldCoins : ℕ := 5

/-- Represents the total number of silver coins -/
def silverCoins : ℕ := 3

/-- Represents the total number of coins -/
def totalCoins : ℕ := goldCoins + silverCoins

/-- The number of distinguishable arrangements for stacking coins -/
def distinguishableArrangements : ℕ := 
  binomial totalCoins goldCoins * validOrientations

theorem coin_stacking_arrangements :
  distinguishableArrangements = 504 := by sorry

end NUMINAMATH_CALUDE_coin_stacking_arrangements_l738_73862


namespace NUMINAMATH_CALUDE_maria_car_rental_cost_l738_73815

/-- Calculates the total cost of a car rental given the daily rate, per-mile rate, number of days, and miles driven. -/
def carRentalCost (dailyRate perMileRate : ℚ) (days miles : ℕ) : ℚ :=
  dailyRate * days + perMileRate * miles

/-- Theorem stating that Maria's car rental cost is $275 given the specified conditions. -/
theorem maria_car_rental_cost :
  carRentalCost 30 0.25 5 500 = 275 := by
  sorry

end NUMINAMATH_CALUDE_maria_car_rental_cost_l738_73815


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l738_73808

theorem logarithm_equation_solution (x : ℝ) (h1 : Real.log x + Real.log (x - 3) = 1) (h2 : x > 0) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equation_solution_l738_73808


namespace NUMINAMATH_CALUDE_special_square_area_l738_73804

/-- A square with two vertices on a parabola and one side on a line -/
structure SpecialSquare where
  /-- The parabola on which two vertices of the square lie -/
  parabola : ℝ → ℝ
  /-- The line on which one side of the square lies -/
  line : ℝ → ℝ
  /-- Condition that the parabola is y = x^2 -/
  parabola_eq : parabola = fun x ↦ x^2
  /-- Condition that the line is y = 2x - 17 -/
  line_eq : line = fun x ↦ 2*x - 17

/-- The area of the special square is either 80 or 1280 -/
theorem special_square_area (s : SpecialSquare) :
  ∃ (area : ℝ), (area = 80 ∨ area = 1280) ∧ 
  (∃ (side : ℝ), side^2 = area ∧
   ∃ (x₁ y₁ x₂ y₂ : ℝ),
     y₁ = s.parabola x₁ ∧
     y₂ = s.parabola x₂ ∧
     (∃ (x₃ y₃ : ℝ), y₃ = s.line x₃ ∧
      side = ((x₃ - x₁)^2 + (y₃ - y₁)^2)^(1/2))) :=
by sorry

end NUMINAMATH_CALUDE_special_square_area_l738_73804


namespace NUMINAMATH_CALUDE_third_side_length_l738_73825

/-- A triangle with sides a, b, and c is valid if it satisfies the triangle inequality theorem --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given two sides of a triangle with lengths 1 and 3, the third side must be 3 --/
theorem third_side_length :
  ∀ x : ℝ, is_valid_triangle 1 3 x → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_l738_73825


namespace NUMINAMATH_CALUDE_thirteen_people_in_line_l738_73868

/-- The number of people in line at an amusement park ride -/
def people_in_line (people_in_front : ℕ) (position_from_back : ℕ) : ℕ :=
  people_in_front + 1 + (position_from_back - 1)

/-- Theorem stating that there are 13 people in line given the conditions -/
theorem thirteen_people_in_line :
  people_in_line 7 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_people_in_line_l738_73868


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficients_l738_73854

theorem quadratic_rational_root_even_coefficients 
  (a b c : ℤ) (x : ℚ) : 
  (a * x^2 + b * x + c = 0) → (Even a ∧ Even b ∧ Even c) :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficients_l738_73854


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_product_l738_73809

theorem gcd_lcm_sum_product (a b : ℕ) (ha : a = 8) (hb : b = 12) :
  (Nat.gcd a b + Nat.lcm a b) * Nat.gcd a b = 112 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_product_l738_73809


namespace NUMINAMATH_CALUDE_min_common_roots_quadratic_trinomials_l738_73885

theorem min_common_roots_quadratic_trinomials 
  (n : ℕ) 
  (f : Fin n → ℝ → ℝ) 
  (h1 : n = 1004)
  (h2 : ∀ i : Fin n, ∃ a b c : ℝ, ∀ x, f i x = x^2 + a*x + b)
  (h3 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ 2007 → ∃ i : Fin n, ∃ x : ℝ, f i x = 0 ∧ x = k)
  : (∀ i j : Fin n, i ≠ j → ∀ x : ℝ, f i x ≠ f j x) :=
sorry

end NUMINAMATH_CALUDE_min_common_roots_quadratic_trinomials_l738_73885


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l738_73823

theorem complex_fraction_equality : Complex.I / (1 + Complex.I) = (1 / 2 : ℂ) + (1 / 2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l738_73823


namespace NUMINAMATH_CALUDE_specific_pyramid_perimeter_l738_73853

/-- A square pyramid with specific dimensions and properties -/
structure SquarePyramid where
  base_edge : ℝ
  lateral_edge : ℝ
  front_view_isosceles : Prop
  side_view_isosceles : Prop
  views_congruent : Prop

/-- The perimeter of the front view of a square pyramid -/
def front_view_perimeter (p : SquarePyramid) : ℝ := sorry

/-- Theorem stating the perimeter of the front view for a specific square pyramid -/
theorem specific_pyramid_perimeter :
  ∀ (p : SquarePyramid),
    p.base_edge = 2 ∧
    p.lateral_edge = Real.sqrt 3 ∧
    p.front_view_isosceles ∧
    p.side_view_isosceles ∧
    p.views_congruent →
    front_view_perimeter p = 2 + 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_perimeter_l738_73853


namespace NUMINAMATH_CALUDE_arrangements_count_l738_73895

/-- Represents the number of liberal arts classes -/
def liberal_arts_classes : ℕ := 2

/-- Represents the number of science classes -/
def science_classes : ℕ := 3

/-- Represents the total number of classes -/
def total_classes : ℕ := liberal_arts_classes + science_classes

/-- Function to calculate the number of arrangements -/
def arrangements : ℕ :=
  (science_classes.choose liberal_arts_classes) *
  (liberal_arts_classes.factorial) *
  (science_classes - liberal_arts_classes) *
  (liberal_arts_classes.factorial)

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_count : arrangements = 24 := by sorry

end NUMINAMATH_CALUDE_arrangements_count_l738_73895


namespace NUMINAMATH_CALUDE_cow_ratio_l738_73820

theorem cow_ratio (total : ℕ) (females males : ℕ) : 
  total = 300 →
  females + males = total →
  females = 2 * (females / 2) →
  males = 2 * (males / 2) →
  females / 2 = males / 2 + 50 →
  females = 2 * males :=
by
  sorry

end NUMINAMATH_CALUDE_cow_ratio_l738_73820


namespace NUMINAMATH_CALUDE_problem_1_l738_73899

theorem problem_1 (x : ℝ) : x^4 * x^3 * x - (x^4)^2 + (-2*x)^3 = -8*x^3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l738_73899


namespace NUMINAMATH_CALUDE_successive_integers_product_l738_73810

theorem successive_integers_product (n : ℤ) : n * (n + 1) = 9506 → n = 97 := by
  sorry

end NUMINAMATH_CALUDE_successive_integers_product_l738_73810


namespace NUMINAMATH_CALUDE_mary_shopping_total_l738_73837

def store1_total : ℚ := 13.04 + 12.27
def store2_total : ℚ := 44.15 + 25.50
def store3_total : ℚ := 2 * 9.99 * (1 - 0.1)
def store4_total : ℚ := 30.93 + 7.42
def store5_total : ℚ := 20.75 * (1 + 0.05)

def total_spent : ℚ := store1_total + store2_total + store3_total + store4_total + store5_total

theorem mary_shopping_total :
  total_spent = 173.08 := by sorry

end NUMINAMATH_CALUDE_mary_shopping_total_l738_73837


namespace NUMINAMATH_CALUDE_cistern_filling_time_l738_73874

-- Define the time to fill 1/11 of the cistern
def time_for_one_eleventh : ℝ := 3

-- Define the function to calculate the time to fill the entire cistern
def time_for_full_cistern : ℝ := time_for_one_eleventh * 11

-- Theorem statement
theorem cistern_filling_time : time_for_full_cistern = 33 := by
  sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l738_73874


namespace NUMINAMATH_CALUDE_max_d_value_l738_73863

def is_valid_number (d e : ℕ) : Prop :=
  d < 10 ∧ e < 10 ∧ e % 2 = 0

def number_value (d e : ℕ) : ℕ :=
  505220 + d * 1000 + e

theorem max_d_value :
  ∃ (d : ℕ), d = 8 ∧
  ∀ (d' e : ℕ), is_valid_number d' e →
  number_value d' e % 22 = 0 →
  d' ≤ d :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l738_73863


namespace NUMINAMATH_CALUDE_infinitely_many_benelux_couples_l738_73817

/-- Definition of a Benelux couple -/
def is_benelux_couple (m n : ℕ) : Prop :=
  1 < m ∧ m < n ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ m ↔ p ∣ n)) ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ (m + 1) ↔ p ∣ (n + 1)))

/-- Theorem: There are infinitely many Benelux couples -/
theorem infinitely_many_benelux_couples :
  ∀ N : ℕ, ∃ m n : ℕ, N < m ∧ is_benelux_couple m n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_benelux_couples_l738_73817


namespace NUMINAMATH_CALUDE_logarithm_inequality_l738_73826

theorem logarithm_inequality (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : 0 < c) 
  (h4 : c < 1) : 
  a * Real.log c / Real.log b < b * Real.log c / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l738_73826


namespace NUMINAMATH_CALUDE_rotate_point_A_l738_73886

/-- Rotates a point 90 degrees clockwise about the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

theorem rotate_point_A :
  let A : ℝ × ℝ := (-4, 1)
  rotate90Clockwise A = (1, 4) := by sorry

end NUMINAMATH_CALUDE_rotate_point_A_l738_73886


namespace NUMINAMATH_CALUDE_largest_number_game_l738_73839

theorem largest_number_game (a b c d : ℤ) : 
  (let game := λ (x y z w : ℤ) => (x + y + z) / 3 + w
   ({game a b c d, game a b d c, game a c d b, game b c d a} : Set ℤ) = {17, 21, 23, 29}) →
  (max a (max b (max c d)) = 21) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_game_l738_73839


namespace NUMINAMATH_CALUDE_emily_beads_count_l738_73860

/-- The number of necklaces Emily made -/
def necklaces : ℕ := 26

/-- The number of beads required for each necklace -/
def beads_per_necklace : ℕ := 2

/-- The total number of beads Emily had -/
def total_beads : ℕ := necklaces * beads_per_necklace

theorem emily_beads_count : total_beads = 52 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l738_73860


namespace NUMINAMATH_CALUDE_product_of_exponents_l738_73829

theorem product_of_exponents (p r s : ℕ) : 
  4^p + 4^2 = 280 → 
  3^r + 29 = 56 → 
  7^s + 6^3 + 7^2 = 728 → 
  p * r * s = 27 := by
  sorry

end NUMINAMATH_CALUDE_product_of_exponents_l738_73829


namespace NUMINAMATH_CALUDE_total_peaches_l738_73891

theorem total_peaches (red yellow green : ℕ) 
  (h1 : red = 7) 
  (h2 : yellow = 15) 
  (h3 : green = 8) : 
  red + yellow + green = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l738_73891


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l738_73897

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l738_73897


namespace NUMINAMATH_CALUDE_sequence_elements_l738_73831

theorem sequence_elements : ∃ (n₁ n₂ : ℕ+), 
  (n₁.val^2 + n₁.val = 12) ∧ 
  (n₂.val^2 + n₂.val = 30) ∧ 
  (∀ n : ℕ+, n.val^2 + n.val ≠ 18) ∧ 
  (∀ n : ℕ+, n.val^2 + n.val ≠ 25) := by
  sorry

end NUMINAMATH_CALUDE_sequence_elements_l738_73831


namespace NUMINAMATH_CALUDE_distance_difference_after_three_hours_l738_73884

/-- Represents a cyclist with a constant cycling rate -/
structure Cyclist where
  name : String
  rate : ℝ  -- cycling rate in miles per hour

/-- Calculates the distance traveled by a cyclist in a given time -/
def distanceTraveled (cyclist : Cyclist) (time : ℝ) : ℝ :=
  cyclist.rate * time

/-- Proves that the difference in distance traveled between Carlos and Diana after 3 hours is 15 miles -/
theorem distance_difference_after_three_hours 
  (carlos : Cyclist)
  (diana : Cyclist)
  (h1 : carlos.rate = 20)
  (h2 : diana.rate = 15)
  : distanceTraveled carlos 3 - distanceTraveled diana 3 = 15 := by
  sorry

#check distance_difference_after_three_hours

end NUMINAMATH_CALUDE_distance_difference_after_three_hours_l738_73884


namespace NUMINAMATH_CALUDE_g_range_l738_73801

/-- The function f(x) = 2x^2 + 3x - 2 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 2

/-- The function g(x) = f(f(x)) -/
def g (x : ℝ) : ℝ := f (f x)

/-- The domain of g -/
def g_domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

theorem g_range :
  ∀ y ∈ g '' g_domain, -2 ≤ y ∧ y ≤ 424 :=
sorry

end NUMINAMATH_CALUDE_g_range_l738_73801


namespace NUMINAMATH_CALUDE_salt_solution_problem_l738_73836

/-- Given a solution with initial volume and salt percentage, prove that
    adding water to reach a specific salt percentage results in the correct
    initial salt percentage. -/
theorem salt_solution_problem (initial_volume : ℝ) (water_added : ℝ) 
    (final_salt_percentage : ℝ) (initial_salt_percentage : ℝ) : 
    initial_volume = 64 →
    water_added = 16 →
    final_salt_percentage = 0.08 →
    initial_salt_percentage * initial_volume = 
      final_salt_percentage * (initial_volume + water_added) →
    initial_salt_percentage = 0.1 := by
  sorry

#check salt_solution_problem

end NUMINAMATH_CALUDE_salt_solution_problem_l738_73836


namespace NUMINAMATH_CALUDE_michaels_art_show_earnings_l738_73878

/-- Calculates Michael's earnings from an art show -/
def michaels_earnings (
  extra_large_price : ℝ)
  (large_price : ℝ)
  (medium_price : ℝ)
  (small_price : ℝ)
  (extra_large_sold : ℕ)
  (large_sold : ℕ)
  (medium_sold : ℕ)
  (small_sold : ℕ)
  (large_discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (material_cost : ℝ)
  (commission_rate : ℝ) : ℝ :=
  let extra_large_revenue := extra_large_price * extra_large_sold
  let large_revenue := large_price * large_sold * (1 - large_discount_rate)
  let medium_revenue := medium_price * medium_sold
  let small_revenue := small_price * small_sold
  let total_revenue := extra_large_revenue + large_revenue + medium_revenue + small_revenue
  let sales_tax := total_revenue * sales_tax_rate
  let total_collected := total_revenue + sales_tax
  let commission := total_revenue * commission_rate
  let total_deductions := material_cost + commission
  total_collected - total_deductions

/-- Theorem stating Michael's earnings from the art show -/
theorem michaels_art_show_earnings :
  michaels_earnings 150 100 80 60 3 5 8 10 0.1 0.05 300 0.1 = 1733 := by
  sorry

end NUMINAMATH_CALUDE_michaels_art_show_earnings_l738_73878


namespace NUMINAMATH_CALUDE_sum_multiple_of_three_l738_73898

theorem sum_multiple_of_three (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 3 * k) 
  (hb : ∃ m : ℤ, b = 9 * m) : 
  ∃ n : ℤ, a + b = 3 * n := by
sorry

end NUMINAMATH_CALUDE_sum_multiple_of_three_l738_73898


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_1_range_of_a_for_inequality_l738_73803

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 2|

-- Part 1
theorem solution_set_when_a_is_1 :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -3 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 1 → f a x ≤ |x + 4|) ↔ a ∈ Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_1_range_of_a_for_inequality_l738_73803


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l738_73883

theorem sqrt_equation_solution (y : ℝ) : Real.sqrt (4 - 5*y) = 8 → y = -12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l738_73883


namespace NUMINAMATH_CALUDE_impossibility_of_identical_remainders_l738_73871

theorem impossibility_of_identical_remainders :
  ¬ ∃ (a : Fin 100 → ℕ) (r : ℕ),
    r ≠ 0 ∧
    ∀ i : Fin 100, a i % a (i.succ) = r :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_identical_remainders_l738_73871


namespace NUMINAMATH_CALUDE_prob_at_least_one_is_correct_l738_73828

/-- The probability that person A tells the truth -/
def prob_A : ℝ := 0.8

/-- The probability that person B tells the truth -/
def prob_B : ℝ := 0.6

/-- The probability that person C tells the truth -/
def prob_C : ℝ := 0.75

/-- The probability that at least one person tells the truth -/
def prob_at_least_one : ℝ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

theorem prob_at_least_one_is_correct : prob_at_least_one = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_is_correct_l738_73828


namespace NUMINAMATH_CALUDE_addie_stamp_ratio_l738_73812

theorem addie_stamp_ratio (parker_initial stamps : ℕ) (parker_final : ℕ) (addie_total : ℕ) : 
  parker_initial = 18 → 
  parker_final = 36 → 
  addie_total = 72 → 
  (parker_final - parker_initial) * 4 = addie_total := by
sorry

end NUMINAMATH_CALUDE_addie_stamp_ratio_l738_73812


namespace NUMINAMATH_CALUDE_mean_equality_problem_l738_73842

theorem mean_equality_problem (x y : ℚ) : 
  (((7 : ℚ) + 11 + 19 + 23) / 4 = (14 + x + y) / 3) →
  x = 2 * y →
  x = 62 / 3 ∧ y = 31 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l738_73842


namespace NUMINAMATH_CALUDE_watch_payment_in_dimes_l738_73806

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of cents in a dime -/
def cents_per_dime : ℕ := 10

/-- The cost of the watch in dollars -/
def watch_cost : ℕ := 9

/-- Calculates the number of dimes needed to pay for an item given its cost in dollars -/
def dimes_needed (cost : ℕ) : ℕ :=
  (cost * cents_per_dollar) / cents_per_dime

theorem watch_payment_in_dimes :
  dimes_needed watch_cost = 90 := by
  sorry

end NUMINAMATH_CALUDE_watch_payment_in_dimes_l738_73806


namespace NUMINAMATH_CALUDE_total_flowers_in_two_weeks_l738_73896

/-- Represents the flowers Miriam takes care of in a day -/
structure DailyFlowers where
  roses : ℕ
  tulips : ℕ
  daisies : ℕ
  lilies : ℕ
  sunflowers : ℕ

/-- Calculates the total number of flowers for a day -/
def totalFlowers (df : DailyFlowers) : ℕ :=
  df.roses + df.tulips + df.daisies + df.lilies + df.sunflowers

/-- Represents Miriam's work schedule for a week -/
structure WeekSchedule where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  saturday : ℕ

/-- Miriam's work hours in the first week -/
def firstWeekSchedule : WeekSchedule :=
  { monday := 4, tuesday := 5, wednesday := 3, thursday := 6, saturday := 5 }

/-- Flowers taken care of in the first week -/
def firstWeekFlowers : DailyFlowers :=
  { roses := 40, tulips := 50, daisies := 36, lilies := 48, sunflowers := 55 }

/-- Calculates the improved number of flowers with 20% increase -/
def improvePerformance (n : ℕ) : ℕ :=
  n + (n / 5)

/-- Theorem stating that the total number of flowers Miriam takes care of in two weeks is 504 -/
theorem total_flowers_in_two_weeks :
  let secondWeekFlowers : DailyFlowers :=
    { roses := improvePerformance firstWeekFlowers.roses,
      tulips := improvePerformance firstWeekFlowers.tulips,
      daisies := improvePerformance firstWeekFlowers.daisies,
      lilies := improvePerformance firstWeekFlowers.lilies,
      sunflowers := improvePerformance firstWeekFlowers.sunflowers }
  totalFlowers firstWeekFlowers + totalFlowers secondWeekFlowers = 504 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_in_two_weeks_l738_73896


namespace NUMINAMATH_CALUDE_expression_value_at_three_l738_73879

theorem expression_value_at_three : 
  let x : ℝ := 3
  (x^8 + 24*x^4 + 144) / (x^4 + 12) = 93 := by
sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l738_73879


namespace NUMINAMATH_CALUDE_min_value_expression_l738_73841

theorem min_value_expression (a : ℚ) : 
  |2*a + 1| + 1 ≥ 1 ∧ ∃ a : ℚ, |2*a + 1| + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l738_73841


namespace NUMINAMATH_CALUDE_cyclist_problem_solution_l738_73827

/-- Represents the cyclist's journey to the bus stop -/
structure CyclistJourney where
  usual_speed : ℝ
  usual_time : ℝ
  reduced_speed_ratio : ℝ
  miss_time : ℝ
  bus_cover_ratio : ℝ

/-- Theorem stating the solution to the cyclist problem -/
theorem cyclist_problem_solution (journey : CyclistJourney) 
  (h1 : journey.reduced_speed_ratio = 4/5)
  (h2 : journey.miss_time = 5)
  (h3 : journey.bus_cover_ratio = 1/3)
  (h4 : journey.usual_time * journey.reduced_speed_ratio = journey.usual_time + journey.miss_time)
  (h5 : journey.usual_time > 0) :
  journey.usual_time = 20 ∧ 
  (journey.usual_time * journey.bus_cover_ratio = journey.usual_time * (1 - journey.bus_cover_ratio)) := by
  sorry

#check cyclist_problem_solution

end NUMINAMATH_CALUDE_cyclist_problem_solution_l738_73827
