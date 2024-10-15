import Mathlib

namespace NUMINAMATH_CALUDE_circle_center_correct_l3269_326911

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 6*y - 12 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (-2, 3)

/-- Theorem: The center of the circle given by the equation x^2 + 4x + y^2 - 6y - 12 = 0 is (-2, 3) -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x + 2)^2 + (y - 3)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l3269_326911


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3269_326942

theorem negative_fraction_comparison : -3/2 < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3269_326942


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3269_326916

/-- The focal length of a hyperbola with equation x²/9 - y²/4 = 1 is 2√13 -/
theorem hyperbola_focal_length : 
  ∀ (x y : ℝ), x^2/9 - y^2/4 = 1 → 
  ∃ (f : ℝ), f = 2 * Real.sqrt 13 ∧ f = 2 * Real.sqrt ((9 : ℝ) + (4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3269_326916


namespace NUMINAMATH_CALUDE_cos_negative_245_deg_l3269_326972

theorem cos_negative_245_deg (a : ℝ) (h : Real.cos (25 * π / 180) = a) :
  Real.cos (-245 * π / 180) = -Real.sqrt (1 - a^2) := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_245_deg_l3269_326972


namespace NUMINAMATH_CALUDE_stock_market_investment_l3269_326975

theorem stock_market_investment (initial_investment : ℝ) (h_positive : initial_investment > 0) :
  let first_year := initial_investment * 1.75
  let second_year := initial_investment * 1.225
  (first_year - second_year) / first_year = 0.3 := by
sorry

end NUMINAMATH_CALUDE_stock_market_investment_l3269_326975


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l3269_326927

/-- Given two real numbers x and y, where x ≠ 0, y ≠ 0, and x ≠ ±y, 
    prove that the given complex expression simplifies to (x-y)^(1/3) / (x+y) -/
theorem complex_expression_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y ∧ x ≠ -y) :
  let numerator := (x^9 - x^6*y^3)^(1/3) - y^2 * ((8*x^6/y^3 - 8*x^3))^(1/3) + 
                   x*y^3 * (y^3 - y^6/x^3)^(1/2)
  let denominator := x^(8/3)*(x^2 - 2*y^2) + (x^2*y^12)^(1/3)
  numerator / denominator = (x-y)^(1/3) / (x+y) := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l3269_326927


namespace NUMINAMATH_CALUDE_multiplication_equation_solution_l3269_326906

theorem multiplication_equation_solution : ∃ x : ℕ, 80641 * x = 806006795 ∧ x = 9995 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equation_solution_l3269_326906


namespace NUMINAMATH_CALUDE_divisor_problem_l3269_326933

theorem divisor_problem : ∃ (d : ℕ), d > 0 ∧ (1019 + 6) % d = 0 ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3269_326933


namespace NUMINAMATH_CALUDE_chord_intersection_lengths_l3269_326915

-- Define the circle and its properties
def circle_radius : ℝ := 6

-- Define the chord EJ and its properties
def chord_length : ℝ := 10

-- Define the point M where EJ intersects GH
def point_M (x : ℝ) : Prop := 0 < x ∧ x < 2 * circle_radius

-- Define the lengths of GM and MH
def length_GM (x : ℝ) : ℝ := x
def length_MH (x : ℝ) : ℝ := 2 * circle_radius - x

-- Theorem statement
theorem chord_intersection_lengths :
  ∃ x : ℝ, point_M x ∧ 
    length_GM x = 6 + Real.sqrt 11 ∧
    length_MH x = 6 - Real.sqrt 11 :=
sorry

end NUMINAMATH_CALUDE_chord_intersection_lengths_l3269_326915


namespace NUMINAMATH_CALUDE_point_above_plane_l3269_326973

theorem point_above_plane (a : ℝ) : 
  (∀ x y : ℝ, x = 1 ∧ y = 2 → x + y + a > 0) ↔ a > -3 :=
sorry

end NUMINAMATH_CALUDE_point_above_plane_l3269_326973


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l3269_326934

theorem rectangular_box_volume : ∃ (x : ℕ), 
  x > 0 ∧ 20 * x^3 = 160 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l3269_326934


namespace NUMINAMATH_CALUDE_square_area_l3269_326964

-- Define the coordinates of the vertex and diagonal intersection
def vertex : ℝ × ℝ := (-6, -4)
def diagonal_intersection : ℝ × ℝ := (3, 2)

-- Define the theorem
theorem square_area (v : ℝ × ℝ) (d : ℝ × ℝ) (h1 : v = vertex) (h2 : d = diagonal_intersection) :
  let diagonal_length := Real.sqrt ((d.1 - v.1)^2 + (d.2 - v.2)^2)
  (diagonal_length^2) / 2 = 58.5 := by sorry

end NUMINAMATH_CALUDE_square_area_l3269_326964


namespace NUMINAMATH_CALUDE_stating_broken_flagpole_height_l3269_326937

/-- Represents a broken flagpole scenario -/
structure BrokenFlagpole where
  initial_height : ℝ
  distance_from_base : ℝ
  break_height : ℝ

/-- 
Theorem stating that for a flagpole of height 8 meters, if it breaks at a point x meters 
above the ground and the upper part touches the ground 3 meters away from the base, 
then x = √73 / 2.
-/
theorem broken_flagpole_height (f : BrokenFlagpole) 
    (h1 : f.initial_height = 8)
    (h2 : f.distance_from_base = 3) :
  f.break_height = Real.sqrt 73 / 2 := by
  sorry


end NUMINAMATH_CALUDE_stating_broken_flagpole_height_l3269_326937


namespace NUMINAMATH_CALUDE_students_playing_neither_l3269_326931

/-- Theorem: In a class of 39 students, where 26 play football, 20 play tennis, and 17 play both,
    the number of students who play neither football nor tennis is 10. -/
theorem students_playing_neither (N F T B : ℕ) 
  (h_total : N = 39)
  (h_football : F = 26)
  (h_tennis : T = 20)
  (h_both : B = 17) :
  N - (F + T - B) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_l3269_326931


namespace NUMINAMATH_CALUDE_julie_landscaping_rate_l3269_326923

/-- Julie's landscaping business problem -/
theorem julie_landscaping_rate :
  ∀ (mowing_rate : ℝ),
  let weeding_rate : ℝ := 8
  let mowing_hours : ℝ := 25
  let weeding_hours : ℝ := 3
  let total_earnings : ℝ := 248
  (2 * (mowing_rate * mowing_hours + weeding_rate * weeding_hours) = total_earnings) →
  mowing_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_julie_landscaping_rate_l3269_326923


namespace NUMINAMATH_CALUDE_workers_wage_before_promotion_l3269_326947

theorem workers_wage_before_promotion (wage_increase_percentage : ℝ) (new_wage : ℝ) : 
  wage_increase_percentage = 0.60 →
  new_wage = 45 →
  (1 + wage_increase_percentage) * (new_wage / (1 + wage_increase_percentage)) = 28.125 := by
sorry

end NUMINAMATH_CALUDE_workers_wage_before_promotion_l3269_326947


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l3269_326912

/-- The function f(x) = x^2 + (1-a)x - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x - a

/-- Theorem stating that if the solution set of f(f(x)) < 0 is empty,
    then -3 ≤ a ≤ 2√2 - 3 -/
theorem empty_solution_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) → -3 ≤ a ∧ a ≤ 2 * Real.sqrt 2 - 3 :=
by sorry


end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l3269_326912


namespace NUMINAMATH_CALUDE_pedoes_inequality_pedoes_inequality_equality_condition_l3269_326925

/-- Pedoe's inequality for triangles -/
theorem pedoes_inequality (a b c a₁ b₁ c₁ Δ Δ₁ : ℝ) 
  (h_abc : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_a₁b₁c₁ : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁)
  (h_Δ : 0 < Δ)
  (h_Δ₁ : 0 < Δ₁)
  (h_abc_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_a₁b₁c₁_triangle : a₁ + b₁ > c₁ ∧ b₁ + c₁ > a₁ ∧ c₁ + a₁ > b₁)
  (h_Δ_def : Δ = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4)
  (h_Δ₁_def : Δ₁ = Real.sqrt ((a₁ + b₁ + c₁) * (b₁ + c₁ - a₁) * (c₁ + a₁ - b₁) * (a₁ + b₁ - c₁)) / 4) :
  a^2 * (b₁^2 + c₁^2 - a₁^2) + b^2 * (c₁^2 + a₁^2 - b₁^2) + c^2 * (a₁^2 + b₁^2 - c₁^2) ≥ 16 * Δ * Δ₁ :=
by sorry

/-- Condition for equality in Pedoe's inequality -/
theorem pedoes_inequality_equality_condition (a b c a₁ b₁ c₁ Δ Δ₁ : ℝ) 
  (h_abc : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_a₁b₁c₁ : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁)
  (h_Δ : 0 < Δ)
  (h_Δ₁ : 0 < Δ₁)
  (h_abc_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_a₁b₁c₁_triangle : a₁ + b₁ > c₁ ∧ b₁ + c₁ > a₁ ∧ c₁ + a₁ > b₁)
  (h_Δ_def : Δ = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4)
  (h_Δ₁_def : Δ₁ = Real.sqrt ((a₁ + b₁ + c₁) * (b₁ + c₁ - a₁) * (c₁ + a₁ - b₁) * (a₁ + b₁ - c₁)) / 4) :
  (a^2 * (b₁^2 + c₁^2 - a₁^2) + b^2 * (c₁^2 + a₁^2 - b₁^2) + c^2 * (a₁^2 + b₁^2 - c₁^2) = 16 * Δ * Δ₁) ↔
  (∃ (k : ℝ), k > 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁) :=
by sorry

end NUMINAMATH_CALUDE_pedoes_inequality_pedoes_inequality_equality_condition_l3269_326925


namespace NUMINAMATH_CALUDE_japanese_students_fraction_l3269_326962

theorem japanese_students_fraction (J : ℚ) (h : J > 0) : 
  let S := 3 * J
  let seniors_studying := (1/3) * S
  let juniors_studying := (3/4) * J
  let total_students := S + J
  (seniors_studying + juniors_studying) / total_students = 7/16 := by
sorry

end NUMINAMATH_CALUDE_japanese_students_fraction_l3269_326962


namespace NUMINAMATH_CALUDE_non_monotonic_interval_l3269_326914

-- Define the function
def f (x : ℝ) : ℝ := |2*x - 1|

-- Define the property of being non-monotonic in an interval
def is_non_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧ 
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- State the theorem
theorem non_monotonic_interval (k : ℝ) :
  is_non_monotonic f (k - 1) (k + 1) ↔ -1 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_non_monotonic_interval_l3269_326914


namespace NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l3269_326985

-- Define the properties of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

-- State the theorem
theorem even_periodic_function_monotonicity (f : ℝ → ℝ) 
  (h_even : is_even f) (h_period : has_period f 2) :
  increasing_on f 0 1 ↔ decreasing_on f 3 4 :=
sorry

end NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l3269_326985


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3269_326960

/-- Two lines are parallel if and only if their slopes are equal and not equal to 1/2 -/
def are_parallel (m : ℝ) : Prop :=
  m / 1 = 1 / m ∧ m / 1 ≠ 1 / 2

/-- The condition m = -1 is sufficient for the lines to be parallel -/
theorem sufficient_condition (m : ℝ) :
  m = -1 → are_parallel m :=
sorry

/-- The condition m = -1 is not necessary for the lines to be parallel -/
theorem not_necessary_condition :
  ∃ m : ℝ, m ≠ -1 ∧ are_parallel m :=
sorry

/-- m = -1 is a sufficient but not necessary condition for the lines to be parallel -/
theorem sufficient_but_not_necessary :
  (∀ m : ℝ, m = -1 → are_parallel m) ∧
  (∃ m : ℝ, m ≠ -1 ∧ are_parallel m) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3269_326960


namespace NUMINAMATH_CALUDE_andy_profit_per_cake_l3269_326998

/-- Calculates the profit per cake for Andy's cake business -/
def profit_per_cake (ingredient_cost_two_cakes : ℚ) (packaging_cost_per_cake : ℚ) (selling_price : ℚ) : ℚ :=
  selling_price - (ingredient_cost_two_cakes / 2 + packaging_cost_per_cake)

/-- Proves that Andy's profit per cake is $8 -/
theorem andy_profit_per_cake :
  profit_per_cake 12 1 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_andy_profit_per_cake_l3269_326998


namespace NUMINAMATH_CALUDE_paths_A_to_D_l3269_326938

/-- Represents a point in the graph --/
inductive Point : Type
| A : Point
| B : Point
| C : Point
| D : Point

/-- Represents a direct path between two points --/
inductive DirectPath : Point → Point → Type
| AB : DirectPath Point.A Point.B
| BC : DirectPath Point.B Point.C
| CD : DirectPath Point.C Point.D
| AC : DirectPath Point.A Point.C
| BD : DirectPath Point.B Point.D

/-- Counts the number of paths between two points --/
def countPaths (start finish : Point) : ℕ :=
  sorry

/-- The main theorem stating that there are 12 paths from A to D --/
theorem paths_A_to_D :
  countPaths Point.A Point.D = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_paths_A_to_D_l3269_326938


namespace NUMINAMATH_CALUDE_pizza_slices_per_pizza_l3269_326913

theorem pizza_slices_per_pizza 
  (num_people : ℕ) 
  (slices_per_person : ℕ) 
  (num_pizzas : ℕ) 
  (h1 : num_people = 10) 
  (h2 : slices_per_person = 2) 
  (h3 : num_pizzas = 5) : 
  (num_people * slices_per_person) / num_pizzas = 4 :=
by sorry

end NUMINAMATH_CALUDE_pizza_slices_per_pizza_l3269_326913


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l3269_326988

/-- Converts a list of digits in base 3 to a base 10 integer -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number -/
def base3Digits : List Nat := [1, 2, 0, 2, 1]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Digits = 142 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l3269_326988


namespace NUMINAMATH_CALUDE_class_survey_is_comprehensive_l3269_326999

/-- Represents a survey population -/
structure SurveyPopulation where
  size : ℕ
  is_finite : Bool

/-- Defines what makes a survey comprehensive -/
def is_comprehensive_survey (pop : SurveyPopulation) : Prop :=
  pop.is_finite ∧ pop.size > 0

/-- Represents a class of students -/
def class_of_students : SurveyPopulation :=
  { size := 30,  -- Assuming an average class size
    is_finite := true }

/-- Theorem stating that a survey of a class is suitable for a comprehensive survey -/
theorem class_survey_is_comprehensive :
  is_comprehensive_survey class_of_students := by
  sorry


end NUMINAMATH_CALUDE_class_survey_is_comprehensive_l3269_326999


namespace NUMINAMATH_CALUDE_simplify_expression_l3269_326983

theorem simplify_expression (x : ℝ) : (2*x + 25) + (150*x^2 + 2*x + 25) = 150*x^2 + 4*x + 50 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3269_326983


namespace NUMINAMATH_CALUDE_absolute_value_equals_cosine_roots_l3269_326987

theorem absolute_value_equals_cosine_roots :
  ∃! (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℝ, |x| = Real.cos x ↔ (x = a ∨ x = b ∨ x = c)) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equals_cosine_roots_l3269_326987


namespace NUMINAMATH_CALUDE_max_cables_sixty_cables_achievable_l3269_326965

/-- Represents the network of computers in the organization -/
structure ComputerNetwork where
  total_employees : ℕ
  brand_a_computers : ℕ
  brand_b_computers : ℕ
  cables : ℕ

/-- Predicate to check if the network satisfies the given conditions -/
def valid_network (n : ComputerNetwork) : Prop :=
  n.total_employees = 50 ∧
  n.brand_a_computers = 30 ∧
  n.brand_b_computers = 20 ∧
  n.cables ≤ n.brand_a_computers * n.brand_b_computers ∧
  n.cables ≥ 2 * n.brand_a_computers

/-- Predicate to check if all employees can communicate -/
def all_can_communicate (n : ComputerNetwork) : Prop :=
  n.cables ≥ n.total_employees - 1

/-- Theorem stating the maximum number of cables -/
theorem max_cables (n : ComputerNetwork) :
  valid_network n → all_can_communicate n → n.cables ≤ 60 :=
by
  sorry

/-- Theorem stating that 60 cables is achievable -/
theorem sixty_cables_achievable :
  ∃ n : ComputerNetwork, valid_network n ∧ all_can_communicate n ∧ n.cables = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_max_cables_sixty_cables_achievable_l3269_326965


namespace NUMINAMATH_CALUDE_cost_price_percentage_l3269_326921

theorem cost_price_percentage (selling_price cost_price : ℝ) :
  selling_price > 0 →
  cost_price > 0 →
  (selling_price - cost_price) / cost_price = 11.11111111111111 / 100 →
  cost_price / selling_price = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l3269_326921


namespace NUMINAMATH_CALUDE_square_plot_area_l3269_326992

/-- Given a square plot with a fence, prove that the area is 289 square feet
    when the price per foot is 54 and the total cost is 3672. -/
theorem square_plot_area (side_length : ℝ) : 
  side_length > 0 →
  (4 * side_length * 54 = 3672) →
  side_length^2 = 289 := by
  sorry


end NUMINAMATH_CALUDE_square_plot_area_l3269_326992


namespace NUMINAMATH_CALUDE_company_average_salary_associates_avg_salary_l3269_326957

theorem company_average_salary 
  (num_managers : ℕ) 
  (num_associates : ℕ) 
  (avg_salary_managers : ℚ) 
  (avg_salary_company : ℚ) : ℚ :=
  let total_employees := num_managers + num_associates
  let total_salary := avg_salary_company * total_employees
  let managers_salary := avg_salary_managers * num_managers
  let associates_salary := total_salary - managers_salary
  associates_salary / num_associates

theorem associates_avg_salary 
  (h1 : company_average_salary 15 75 90000 40000 = 30000) : 
  company_average_salary 15 75 90000 40000 = 30000 := by
  sorry

end NUMINAMATH_CALUDE_company_average_salary_associates_avg_salary_l3269_326957


namespace NUMINAMATH_CALUDE_willy_finishes_series_in_30_days_l3269_326908

/-- Calculates the number of days needed to finish a TV series -/
def days_to_finish_series (seasons : ℕ) (episodes_per_season : ℕ) (episodes_per_day : ℕ) : ℕ :=
  (seasons * episodes_per_season) / episodes_per_day

/-- Proves that it takes 30 days to finish the given TV series -/
theorem willy_finishes_series_in_30_days :
  days_to_finish_series 3 20 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_willy_finishes_series_in_30_days_l3269_326908


namespace NUMINAMATH_CALUDE_speed_limit_calculation_l3269_326936

/-- Proves that given a distance of 150 miles traveled in 2 hours,
    and driving 15 mph above the speed limit, the speed limit is 60 mph. -/
theorem speed_limit_calculation (distance : ℝ) (time : ℝ) (speed_above_limit : ℝ) 
    (h1 : distance = 150)
    (h2 : time = 2)
    (h3 : speed_above_limit = 15) :
    distance / time - speed_above_limit = 60 := by
  sorry

end NUMINAMATH_CALUDE_speed_limit_calculation_l3269_326936


namespace NUMINAMATH_CALUDE_kath_friends_count_l3269_326932

/-- The number of friends Kath took to the movie --/
def num_friends : ℕ :=
  -- Define this value
  sorry

/-- The number of Kath's siblings --/
def num_siblings : ℕ := 2

/-- The regular admission cost in dollars --/
def regular_cost : ℕ := 8

/-- The discount amount in dollars --/
def discount : ℕ := 3

/-- The total amount Kath paid in dollars --/
def total_paid : ℕ := 30

/-- The actual cost per person after discount --/
def discounted_cost : ℕ := regular_cost - discount

/-- The total number of people in Kath's group --/
def total_people : ℕ := total_paid / discounted_cost

theorem kath_friends_count :
  num_friends = total_people - (num_siblings + 1) ∧
  num_friends = 3 :=
sorry

end NUMINAMATH_CALUDE_kath_friends_count_l3269_326932


namespace NUMINAMATH_CALUDE_count_solutions_x_plus_y_plus_z_15_l3269_326941

/-- The number of solutions to x + y + z = 15 where x, y, and z are positive integers -/
def num_solutions : ℕ := 91

/-- Theorem stating that the number of solutions to x + y + z = 15 where x, y, and z are positive integers is 91 -/
theorem count_solutions_x_plus_y_plus_z_15 :
  (Finset.filter (fun t : ℕ × ℕ × ℕ => t.1 + t.2.1 + t.2.2 = 15 ∧ t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0)
    (Finset.product (Finset.range 16) (Finset.product (Finset.range 16) (Finset.range 16)))).card = num_solutions := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_x_plus_y_plus_z_15_l3269_326941


namespace NUMINAMATH_CALUDE_find_x_value_l3269_326935

def A (x : ℕ) : Set ℕ := {1, 4, x}
def B (x : ℕ) : Set ℕ := {1, x^2}

theorem find_x_value (x : ℕ) (h : A x ∪ B x = A x) : x = 0 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l3269_326935


namespace NUMINAMATH_CALUDE_breadth_is_ten_l3269_326984

/-- A rectangular plot with specific properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ
  area_eq : area = 20 * breadth
  length_eq : length = breadth + 10

/-- The breadth of a rectangular plot with the given properties is 10 meters -/
theorem breadth_is_ten (plot : RectangularPlot) : plot.breadth = 10 := by
  sorry

end NUMINAMATH_CALUDE_breadth_is_ten_l3269_326984


namespace NUMINAMATH_CALUDE_power_of_seven_inverse_l3269_326991

theorem power_of_seven_inverse (x y : ℕ) : 
  (2^x : ℕ) = Nat.gcd 180 (2^Nat.succ x) →
  (3^y : ℕ) = Nat.gcd 180 (3^Nat.succ y) →
  (1/7 : ℚ)^(y - x) = 1 :=
by sorry

end NUMINAMATH_CALUDE_power_of_seven_inverse_l3269_326991


namespace NUMINAMATH_CALUDE_moon_earth_distance_in_scientific_notation_l3269_326905

/-- The average distance from the moon to the earth in meters -/
def moon_earth_distance : ℝ := 384000000

/-- The scientific notation representation of the moon-earth distance -/
def moon_earth_distance_scientific : ℝ := 3.84 * (10 ^ 8)

theorem moon_earth_distance_in_scientific_notation :
  moon_earth_distance = moon_earth_distance_scientific := by
  sorry

end NUMINAMATH_CALUDE_moon_earth_distance_in_scientific_notation_l3269_326905


namespace NUMINAMATH_CALUDE_porter_earnings_l3269_326971

/-- Porter's daily rate in dollars -/
def daily_rate : ℚ := 8

/-- Number of regular working days per week -/
def regular_days : ℕ := 5

/-- Number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Overtime pay rate as a multiplier of regular rate -/
def overtime_rate : ℚ := 3/2

/-- Tax deduction rate -/
def tax_rate : ℚ := 1/10

/-- Insurance and benefits deduction rate -/
def insurance_rate : ℚ := 1/20

/-- Calculate Porter's monthly earnings after deductions and overtime -/
def monthly_earnings : ℚ :=
  let regular_weekly := daily_rate * regular_days
  let overtime_daily := daily_rate * overtime_rate
  let total_weekly := regular_weekly + overtime_daily
  let monthly_before_deductions := total_weekly * weeks_per_month
  let deductions := monthly_before_deductions * (tax_rate + insurance_rate)
  monthly_before_deductions - deductions

theorem porter_earnings :
  monthly_earnings = 1768/10 := by sorry

end NUMINAMATH_CALUDE_porter_earnings_l3269_326971


namespace NUMINAMATH_CALUDE_m_range_l3269_326966

theorem m_range : ∀ m : ℝ, m = 3 * Real.sqrt 2 - 1 → 3 < m ∧ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l3269_326966


namespace NUMINAMATH_CALUDE_grandma_salad_ratio_l3269_326950

/-- Proves that the ratio of cherry tomatoes to mushrooms is 2:1 given the conditions of Grandma's salad --/
theorem grandma_salad_ratio : ∀ (cherry_tomatoes pickles bacon_bits : ℕ),
  pickles = 4 * cherry_tomatoes →
  bacon_bits = 4 * pickles →
  bacon_bits / 3 = 32 →
  cherry_tomatoes / 3 = 2 :=
by
  sorry

#check grandma_salad_ratio

end NUMINAMATH_CALUDE_grandma_salad_ratio_l3269_326950


namespace NUMINAMATH_CALUDE_connie_marble_count_l3269_326974

def marble_problem (connie_marbles juan_marbles : ℕ) : Prop :=
  juan_marbles = connie_marbles + 175 ∧ juan_marbles = 498

theorem connie_marble_count :
  ∀ connie_marbles juan_marbles,
    marble_problem connie_marbles juan_marbles →
    connie_marbles = 323 :=
by
  sorry

end NUMINAMATH_CALUDE_connie_marble_count_l3269_326974


namespace NUMINAMATH_CALUDE_ordered_pairs_theorem_l3269_326955

def S : Set (ℕ × ℕ) := {(8, 4), (9, 3), (2, 1)}

def satisfies_conditions (pair : ℕ × ℕ) : Prop :=
  let (x, y) := pair
  x > y ∧ (x - y = 2 * x / y ∨ x - y = 2 * y / x)

theorem ordered_pairs_theorem :
  ∀ (pair : ℕ × ℕ), pair ∈ S ↔ satisfies_conditions pair ∧ pair.1 > 0 ∧ pair.2 > 0 :=
sorry

end NUMINAMATH_CALUDE_ordered_pairs_theorem_l3269_326955


namespace NUMINAMATH_CALUDE_sample_size_is_thirty_l3269_326970

/-- Represents the ratio of young, middle-aged, and elderly employees -/
structure EmployeeRatio :=
  (young : ℕ)
  (middle : ℕ)
  (elderly : ℕ)

/-- Calculates the total sample size given the ratio and number of young employees in the sample -/
def calculateSampleSize (ratio : EmployeeRatio) (youngInSample : ℕ) : ℕ :=
  let totalRatio := ratio.young + ratio.middle + ratio.elderly
  (youngInSample * totalRatio) / ratio.young

/-- Theorem stating that for the given ratio and number of young employees, the sample size is 30 -/
theorem sample_size_is_thirty :
  let ratio : EmployeeRatio := { young := 7, middle := 5, elderly := 3 }
  let youngInSample : ℕ := 14
  calculateSampleSize ratio youngInSample = 30 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_thirty_l3269_326970


namespace NUMINAMATH_CALUDE_pelican_fish_count_l3269_326949

theorem pelican_fish_count (P : ℕ) : 
  (P + 7 = P + 7) →  -- Kingfisher caught 7 more fish than the pelican
  (3 * (P + (P + 7)) = P + 86) →  -- Fisherman caught 3 times the total and 86 more than the pelican
  P = 13 := by
sorry

end NUMINAMATH_CALUDE_pelican_fish_count_l3269_326949


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3269_326990

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a ≥ 6 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    (a₀^2 + b₀^2) / c₀ + (a₀^2 + c₀^2) / b₀ + (b₀^2 + c₀^2) / a₀ = 6 :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_expression_l3269_326990


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3269_326982

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3269_326982


namespace NUMINAMATH_CALUDE_rhombus_area_l3269_326940

/-- The area of a rhombus with vertices at (0, 3.5), (10, 0), (0, -3.5), and (-10, 0) is 70 square units. -/
theorem rhombus_area : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (10, 0), (0, -3.5), (-10, 0)]
  let diag1 : ℝ := |3.5 - (-3.5)|
  let diag2 : ℝ := |10 - (-10)|
  (diag1 * diag2) / 2 = 70 := by
  sorry

#check rhombus_area

end NUMINAMATH_CALUDE_rhombus_area_l3269_326940


namespace NUMINAMATH_CALUDE_divisibility_problem_l3269_326918

theorem divisibility_problem (a : ℕ) :
  (∃! n : Fin 4, ¬ (
    (n = 0 → a % 2 = 0) ∧
    (n = 1 → a % 4 = 0) ∧
    (n = 2 → a % 12 = 0) ∧
    (n = 3 → a % 24 = 0)
  )) →
  ¬(a % 24 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3269_326918


namespace NUMINAMATH_CALUDE_city_mpg_is_24_l3269_326909

/-- Represents the fuel efficiency of a car in different driving conditions. -/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℝ
  city_miles_per_tankful : ℝ
  highway_city_mpg_difference : ℝ

/-- Calculates the city miles per gallon given the car's fuel efficiency data. -/
def city_mpg (car : CarFuelEfficiency) : ℝ :=
  -- The actual calculation is not provided here
  sorry

/-- Theorem stating that for the given car fuel efficiency data, 
    the city miles per gallon is 24. -/
theorem city_mpg_is_24 (car : CarFuelEfficiency) 
  (h1 : car.highway_miles_per_tankful = 462)
  (h2 : car.city_miles_per_tankful = 336)
  (h3 : car.highway_city_mpg_difference = 9) :
  city_mpg car = 24 := by
  sorry

end NUMINAMATH_CALUDE_city_mpg_is_24_l3269_326909


namespace NUMINAMATH_CALUDE_pencil_length_l3269_326961

theorem pencil_length : ∀ (L : ℝ),
  (1/8 : ℝ) * L +  -- Black part
  (1/2 : ℝ) * ((7/8 : ℝ) * L) +  -- White part
  (7/2 : ℝ) = L  -- Blue part
  → L = 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_length_l3269_326961


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l3269_326901

theorem quadratic_root_implies_k (k : ℝ) : 
  (3 * ((-15 - Real.sqrt 229) / 4)^2 + 15 * ((-15 - Real.sqrt 229) / 4) + k = 0) → 
  k = -1/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l3269_326901


namespace NUMINAMATH_CALUDE_gcd_1443_999_l3269_326976

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1443_999_l3269_326976


namespace NUMINAMATH_CALUDE_two_satisfying_functions_l3269_326929

/-- A function satisfying the given functional equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * f y + c

/-- The set of all functions satisfying the functional equation -/
def SatisfyingFunctions : Set (ℝ → ℝ) :=
  {f | SatisfyingFunction f}

/-- The constant zero function -/
def ZeroFunction : ℝ → ℝ := λ _ => 0

/-- The square function -/
def SquareFunction : ℝ → ℝ := λ x => x^2

theorem two_satisfying_functions :
  SatisfyingFunctions = {ZeroFunction, SquareFunction} := by sorry

end NUMINAMATH_CALUDE_two_satisfying_functions_l3269_326929


namespace NUMINAMATH_CALUDE_present_age_of_b_l3269_326953

theorem present_age_of_b (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- In 10 years, A will be twice as old as B was 10 years ago
  (a = b + 8) →              -- A is currently 8 years older than B
  b = 38                     -- B's present age is 38
  := by sorry

end NUMINAMATH_CALUDE_present_age_of_b_l3269_326953


namespace NUMINAMATH_CALUDE_quiche_cost_is_15_l3269_326920

/-- Represents the cost of a single quiche -/
def quiche_cost : ℝ := sorry

/-- Represents the number of quiches ordered -/
def num_quiches : ℕ := 2

/-- Represents the cost of a single croissant -/
def croissant_cost : ℝ := 3

/-- Represents the number of croissants ordered -/
def num_croissants : ℕ := 6

/-- Represents the cost of a single biscuit -/
def biscuit_cost : ℝ := 2

/-- Represents the number of biscuits ordered -/
def num_biscuits : ℕ := 6

/-- Represents the discount rate -/
def discount_rate : ℝ := 0.1

/-- Represents the discounted total cost -/
def discounted_total : ℝ := 54

/-- Theorem stating that the cost of each quiche is $15 -/
theorem quiche_cost_is_15 : quiche_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_quiche_cost_is_15_l3269_326920


namespace NUMINAMATH_CALUDE_divisibility_999_from_50_l3269_326900

/-- A function that extracts 50 consecutive digits from a 999-digit number starting at a given index -/
def extract_50_digits (n : ℕ) (start_index : ℕ) : ℕ := sorry

/-- Predicate to check if a number is a valid 999-digit number -/
def is_999_digit_number (n : ℕ) : Prop := sorry

theorem divisibility_999_from_50 (n : ℕ) (h1 : is_999_digit_number n)
  (h2 : ∀ i, i ≤ 950 → extract_50_digits n i % 2^50 = 0) :
  n % 2^999 = 0 := by sorry

end NUMINAMATH_CALUDE_divisibility_999_from_50_l3269_326900


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l3269_326948

theorem negation_of_existence (p : ℝ → Prop) : (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + x + 1 = 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l3269_326948


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3269_326986

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (2*x + 1)/(x - 2)) ↔ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3269_326986


namespace NUMINAMATH_CALUDE_complex_power_100_l3269_326981

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_100 : ((1 + i) / (1 - i)) ^ 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_100_l3269_326981


namespace NUMINAMATH_CALUDE_survey_respondents_l3269_326930

theorem survey_respondents (preferred_x : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) : 
  preferred_x = 360 → ratio_x = 9 → ratio_y = 1 → 
  ∃ (total : ℕ), total = preferred_x + (preferred_x * ratio_y) / ratio_x ∧ total = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l3269_326930


namespace NUMINAMATH_CALUDE_a_55_mod_45_l3269_326994

/-- Definition of a_n as a function that concatenates integers from 1 to n -/
def a (n : ℕ) : ℕ := sorry

/-- The remainder when a_55 is divided by 45 is 10 -/
theorem a_55_mod_45 : a 55 % 45 = 10 := by sorry

end NUMINAMATH_CALUDE_a_55_mod_45_l3269_326994


namespace NUMINAMATH_CALUDE_range_of_f_range_of_g_l3269_326969

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4*a*x + 2*a + 6

-- Define the function g
def g (a : ℝ) : ℝ := 2 - a * |a + 3|

-- Theorem 1: The range of f is [0,+∞) iff a = -1 or a = 3/2
theorem range_of_f (a : ℝ) : 
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f a x = y) ∧ (∀ x : ℝ, f a x ≥ 0) ↔ 
  a = -1 ∨ a = 3/2 :=
sorry

-- Theorem 2: When f(x) ≥ 0 for all x, the range of g(a) is [-19/4, 4]
theorem range_of_g : 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≥ 0) → 
    (∀ y : ℝ, -19/4 ≤ y ∧ y ≤ 4 → ∃ a : ℝ, g a = y) ∧ 
    (∀ a : ℝ, -19/4 ≤ g a ∧ g a ≤ 4)) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_g_l3269_326969


namespace NUMINAMATH_CALUDE_simplify_expression_l3269_326919

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3269_326919


namespace NUMINAMATH_CALUDE_problem_statement_l3269_326963

theorem problem_statement : (1 / ((-8^2)^4)) * (-8)^9 = -8 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3269_326963


namespace NUMINAMATH_CALUDE_emily_lost_lives_l3269_326904

theorem emily_lost_lives (initial_lives : ℕ) (lives_gained : ℕ) (final_lives : ℕ) : 
  initial_lives = 42 → lives_gained = 24 → final_lives = 41 → 
  ∃ (lives_lost : ℕ), initial_lives - lives_lost + lives_gained = final_lives ∧ lives_lost = 25 := by
sorry

end NUMINAMATH_CALUDE_emily_lost_lives_l3269_326904


namespace NUMINAMATH_CALUDE_pet_center_final_count_l3269_326944

/-- 
Given:
- initial_dogs: The initial number of dogs in the pet center
- initial_cats: The initial number of cats in the pet center
- adopted_dogs: The number of dogs adopted
- new_cats: The number of new cats collected

Prove that the final number of pets in the pet center is 57.
-/
theorem pet_center_final_count 
  (initial_dogs : ℕ) 
  (initial_cats : ℕ) 
  (adopted_dogs : ℕ) 
  (new_cats : ℕ) 
  (h1 : initial_dogs = 36)
  (h2 : initial_cats = 29)
  (h3 : adopted_dogs = 20)
  (h4 : new_cats = 12) :
  initial_dogs - adopted_dogs + initial_cats + new_cats = 57 :=
by
  sorry


end NUMINAMATH_CALUDE_pet_center_final_count_l3269_326944


namespace NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l3269_326989

/-- Given a point C with coordinates (x, -5) and its reflection D over the y-axis,
    the sum of all coordinate values of C and D is -10. -/
theorem sum_of_coordinates_after_reflection (x : ℝ) :
  let C : ℝ × ℝ := (x, -5)
  let D : ℝ × ℝ := (-x, -5)  -- reflection of C over y-axis
  (C.1 + C.2 + D.1 + D.2) = -10 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l3269_326989


namespace NUMINAMATH_CALUDE_happy_water_consumption_l3269_326978

/-- Given Happy's current water consumption and recommended increase percentage,
    calculate the new recommended number of cups per week. -/
theorem happy_water_consumption (current : ℝ) (increase_percent : ℝ) :
  current = 25 → increase_percent = 75 →
  current + (increase_percent / 100) * current = 43.75 := by
  sorry

end NUMINAMATH_CALUDE_happy_water_consumption_l3269_326978


namespace NUMINAMATH_CALUDE_zeroes_at_end_of_600_times_50_l3269_326946

theorem zeroes_at_end_of_600_times_50 : ∃ n : ℕ, 600 * 50 = n * 10000 ∧ n % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zeroes_at_end_of_600_times_50_l3269_326946


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l3269_326977

-- Define a polynomial with real coefficients
def RealPolynomial := ℝ → ℝ

-- Define the functional equation
def SatisfiesFunctionalEquation (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, 1 + P x = (1 / 2) * (P (x - 1) + P (x + 1))

-- Define the quadratic form
def IsQuadraticForm (P : RealPolynomial) : Prop :=
  ∃ b c : ℝ, ∀ x : ℝ, P x = x^2 + b * x + c

-- Theorem statement
theorem polynomial_functional_equation :
  ∀ P : RealPolynomial, SatisfiesFunctionalEquation P → IsQuadraticForm P :=
by
  sorry


end NUMINAMATH_CALUDE_polynomial_functional_equation_l3269_326977


namespace NUMINAMATH_CALUDE_max_sum_after_swap_l3269_326943

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Swaps the first and last digits of a ThreeDigitNumber -/
def ThreeDigitNumber.swap (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.ones
  tens := n.tens
  ones := n.hundreds
  is_valid := by sorry

/-- The main theorem to prove -/
theorem max_sum_after_swap
  (a b c : ThreeDigitNumber)
  (h : a.toNat + b.toNat + c.toNat = 2019) :
  (a.swap.toNat + b.swap.toNat + c.swap.toNat) ≤ 2118 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_after_swap_l3269_326943


namespace NUMINAMATH_CALUDE_bacteria_after_three_hours_l3269_326922

/-- Represents the number of bacteria after a given time -/
def bacteria_count (initial_count : ℕ) (split_interval : ℕ) (total_time : ℕ) : ℕ :=
  initial_count * 2 ^ (total_time / split_interval)

/-- Theorem stating that the number of bacteria after 3 hours is 64 -/
theorem bacteria_after_three_hours :
  bacteria_count 1 30 180 = 64 := by
  sorry

#check bacteria_after_three_hours

end NUMINAMATH_CALUDE_bacteria_after_three_hours_l3269_326922


namespace NUMINAMATH_CALUDE_rectangle_ratio_l3269_326952

/-- Given a rectangle divided into three congruent smaller rectangles,
    where each smaller rectangle is similar to the large rectangle,
    the ratio of the longer side to the shorter side is √3 : 1 for all rectangles. -/
theorem rectangle_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_similar : x / y = (3 * y) / x) : x / y = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l3269_326952


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3269_326956

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4}
  A ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3269_326956


namespace NUMINAMATH_CALUDE_intersection_M_N_l3269_326995

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {-1, 1, 2}

theorem intersection_M_N : M ∩ N = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3269_326995


namespace NUMINAMATH_CALUDE_cylinder_cross_section_angle_l3269_326917

/-- Given a cylinder cut by a plane where the cross-section has an eccentricity of 2√2/3,
    the acute dihedral angle between this cross-section and the cylinder's base is arccos(1/3). -/
theorem cylinder_cross_section_angle (e : ℝ) (θ : ℝ) : 
  e = 2 * Real.sqrt 2 / 3 →
  θ = Real.arccos (1/3) →
  θ = Real.arccos (Real.sqrt (1 - e^2)) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_angle_l3269_326917


namespace NUMINAMATH_CALUDE_count_negative_rationals_l3269_326954

theorem count_negative_rationals : 
  let S : Finset ℚ := {-5, -(-3), 3.14, |-2/7|, -(2^3), 0}
  (S.filter (λ x => x < 0)).card = 2 := by sorry

end NUMINAMATH_CALUDE_count_negative_rationals_l3269_326954


namespace NUMINAMATH_CALUDE_distinct_odd_numbers_count_l3269_326993

-- Define the given number as a list of digits
def given_number : List Nat := [3, 4, 3, 9, 6]

-- Function to check if a number is odd
def is_odd (n : Nat) : Bool :=
  n % 2 = 1

-- Function to count distinct permutations
def count_distinct_permutations (digits : List Nat) : Nat :=
  sorry

-- Function to count distinct odd permutations
def count_distinct_odd_permutations (digits : List Nat) : Nat :=
  sorry

-- Theorem statement
theorem distinct_odd_numbers_count :
  count_distinct_odd_permutations given_number = 36 := by
  sorry

end NUMINAMATH_CALUDE_distinct_odd_numbers_count_l3269_326993


namespace NUMINAMATH_CALUDE_juniper_bones_l3269_326967

theorem juniper_bones (b : ℕ) : 2 * b - 2 = (b + b) - 2 := by sorry

end NUMINAMATH_CALUDE_juniper_bones_l3269_326967


namespace NUMINAMATH_CALUDE_circle_ratio_l3269_326926

theorem circle_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) 
  (h_area : π * r₂^2 - π * r₁^2 = 4 * π * r₁^2) : 
  r₁ / r₂ = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l3269_326926


namespace NUMINAMATH_CALUDE_P_intersect_M_l3269_326907

def P : Set ℤ := {x : ℤ | 0 ≤ x ∧ x < 3}
def M : Set ℤ := {x : ℤ | x^2 ≤ 9}

theorem P_intersect_M : P ∩ M = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_P_intersect_M_l3269_326907


namespace NUMINAMATH_CALUDE_f_r_correct_l3269_326968

/-- The number of ways to select k elements from a permutation of n elements,
    such that any two selected elements are separated by at least r elements
    in the original permutation. -/
def f_r (n k r : ℕ) : ℕ :=
  Nat.choose (n - k * r + r) k

/-- Theorem stating that f_r(n, k, r) correctly counts the number of ways to select
    k elements from a permutation of n elements with the given separation condition. -/
theorem f_r_correct (n k r : ℕ) :
  f_r n k r = Nat.choose (n - k * r + r) k :=
by sorry

end NUMINAMATH_CALUDE_f_r_correct_l3269_326968


namespace NUMINAMATH_CALUDE_set_operation_result_arithmetic_expression_result_l3269_326997

-- Define the sets A, B, and C
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -2 < x ∧ x < 2}
def C : Set ℝ := {x | -3 < x ∧ x < 5}

-- Theorem 1: Set operation result
theorem set_operation_result : (A ∪ B) ∩ C = {x : ℝ | -2 < x ∧ x < 5} := by sorry

-- Theorem 2: Arithmetic expression result
theorem arithmetic_expression_result :
  (2 + 1/4)^(1/2) - (-9.6)^0 - (3 + 3/8)^(-2/3) + (1.5)^(-2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_set_operation_result_arithmetic_expression_result_l3269_326997


namespace NUMINAMATH_CALUDE_max_elevation_l3269_326910

/-- The elevation function of a particle thrown vertically upwards -/
def s (t : ℝ) : ℝ := 240 * t - 24 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ (t : ℝ), ∀ (t' : ℝ), s t ≥ s t' ∧ s t = 600 := by
  sorry

end NUMINAMATH_CALUDE_max_elevation_l3269_326910


namespace NUMINAMATH_CALUDE_monogram_combinations_l3269_326924

theorem monogram_combinations : ∀ n k : ℕ, 
  n = 14 ∧ k = 2 → (n.choose k) = 91 :=
by
  sorry

#check monogram_combinations

end NUMINAMATH_CALUDE_monogram_combinations_l3269_326924


namespace NUMINAMATH_CALUDE_problem_l3269_326928

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + a^(-x)

theorem problem (a : ℝ) (h1 : a > 1) (h2 : f a 1 = 3) :
  (f a 2 = 7) ∧
  (∀ x₁ x₂, 0 ≤ x₂ ∧ x₂ < x₁ → f a x₁ > f a x₂) ∧
  (∀ m x, 0 ≤ x ∧ x ≤ 1 → 
    f a (2*x) - m * f a x ≥ min (2 - 2*m) (min (-m^2/4 - 2) (7 - 3*m))) := by
  sorry

end NUMINAMATH_CALUDE_problem_l3269_326928


namespace NUMINAMATH_CALUDE_tv_selection_theorem_l3269_326980

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of Type A TVs -/
def typeA : ℕ := 4

/-- The number of Type B TVs -/
def typeB : ℕ := 5

/-- The total number of TVs to be selected -/
def selectTotal : ℕ := 3

/-- The number of ways to select TVs satisfying the given conditions -/
def selectWays : ℕ :=
  typeA * binomial typeB (selectTotal - 1) +
  binomial typeA (selectTotal - 1) * typeB

theorem tv_selection_theorem : selectWays = 70 := by sorry

end NUMINAMATH_CALUDE_tv_selection_theorem_l3269_326980


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l3269_326945

/-- Represents the dimensions and properties of a metallic sheet and the box formed from it. -/
structure MetallicSheet where
  length : ℝ
  width : ℝ
  cutSquareSize : ℝ
  boxVolume : ℝ

/-- Calculates the volume of the box formed from the metallic sheet. -/
def boxVolumeCalc (sheet : MetallicSheet) : ℝ :=
  (sheet.length - 2 * sheet.cutSquareSize) * 
  (sheet.width - 2 * sheet.cutSquareSize) * 
  sheet.cutSquareSize

/-- Theorem stating the width of the metallic sheet given the conditions. -/
theorem metallic_sheet_width 
  (sheet : MetallicSheet)
  (h1 : sheet.length = 48)
  (h2 : sheet.cutSquareSize = 8)
  (h3 : sheet.boxVolume = 5120)
  (h4 : boxVolumeCalc sheet = sheet.boxVolume) :
  sheet.width = 36 :=
sorry

end NUMINAMATH_CALUDE_metallic_sheet_width_l3269_326945


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3269_326959

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℚ)
  (h_geo : geometric_sequence a)
  (h_1 : a 0 = 32)
  (h_2 : a 1 = -48)
  (h_3 : a 2 = 72)
  (h_4 : a 3 = -108)
  (h_5 : a 4 = 162) :
  ∃ r : ℚ, r = -3/2 ∧ ∀ n : ℕ, a (n + 1) = r * a n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3269_326959


namespace NUMINAMATH_CALUDE_cubic_equation_root_l3269_326951

theorem cubic_equation_root (a b : ℚ) : 
  (3 + Real.sqrt 5) ^ 3 + a * (3 + Real.sqrt 5) ^ 2 + b * (3 + Real.sqrt 5) + 20 = 0 → 
  b = -26 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l3269_326951


namespace NUMINAMATH_CALUDE_product_xyz_is_zero_l3269_326939

theorem product_xyz_is_zero 
  (x y z : ℝ) 
  (h1 : x + 2/y = 2) 
  (h2 : y + 2/z = 2) : 
  x * y * z = 0 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_is_zero_l3269_326939


namespace NUMINAMATH_CALUDE_pill_cost_calculation_l3269_326902

/-- The cost of one pill in dollars -/
def pill_cost : ℝ := 1.50

/-- The number of pills John takes per day -/
def pills_per_day : ℕ := 2

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The percentage of the cost that John pays (insurance covers the rest) -/
def john_payment_percentage : ℝ := 0.60

/-- The amount John pays for pills in a month in dollars -/
def john_monthly_payment : ℝ := 54

theorem pill_cost_calculation :
  pill_cost = john_monthly_payment / (pills_per_day * days_in_month * john_payment_percentage) :=
sorry

end NUMINAMATH_CALUDE_pill_cost_calculation_l3269_326902


namespace NUMINAMATH_CALUDE_water_bill_calculation_l3269_326958

def weekly_income : ℝ := 500
def tax_rate : ℝ := 0.10
def tithe_rate : ℝ := 0.10
def remaining_amount : ℝ := 345

theorem water_bill_calculation :
  let after_tax := weekly_income * (1 - tax_rate)
  let after_tithe := after_tax - (weekly_income * tithe_rate)
  let water_bill := after_tithe - remaining_amount
  water_bill = 55 := by sorry

end NUMINAMATH_CALUDE_water_bill_calculation_l3269_326958


namespace NUMINAMATH_CALUDE_line_parallel_plane_perpendicular_line_l3269_326979

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_plane_perpendicular_line 
  (l m : Line) (α : Plane) :
  l ≠ m →
  parallel l α →
  perpendicular m α →
  perpendicularLines l m :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_perpendicular_line_l3269_326979


namespace NUMINAMATH_CALUDE_error_percentage_division_vs_multiplication_error_percentage_proof_l3269_326903

theorem error_percentage_division_vs_multiplication : ℝ → Prop :=
  fun x => x ≠ 0 →
    let correct_result := 5 * x
    let incorrect_result := x / 10
    let error := correct_result - incorrect_result
    let percentage_error := (error / correct_result) * 100
    percentage_error = 98

-- The proof is omitted
theorem error_percentage_proof : ∀ x : ℝ, error_percentage_division_vs_multiplication x :=
sorry

end NUMINAMATH_CALUDE_error_percentage_division_vs_multiplication_error_percentage_proof_l3269_326903


namespace NUMINAMATH_CALUDE_natalie_shopping_money_left_l3269_326996

def initial_amount : ℕ := 26
def jumper_cost : ℕ := 9
def tshirt_cost : ℕ := 4
def heels_cost : ℕ := 5

theorem natalie_shopping_money_left :
  initial_amount - (jumper_cost + tshirt_cost + heels_cost) = 8 := by
  sorry

end NUMINAMATH_CALUDE_natalie_shopping_money_left_l3269_326996
