import Mathlib

namespace NUMINAMATH_CALUDE_closed_set_properties_l167_16792

-- Definition of a closed set
def is_closed_set (M : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Set M = {-2, -1, 0, 1, 2}
def M : Set ℤ := {-2, -1, 0, 1, 2}

-- Set of positive integers
def positive_integers : Set ℤ := {n : ℤ | n > 0}

-- Set of multiples of 3
def multiples_of_three : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem closed_set_properties :
  ¬(is_closed_set M) ∧
  ¬(is_closed_set positive_integers) ∧
  (is_closed_set multiples_of_three) ∧
  (∃ A₁ A₂ : Set ℤ, is_closed_set A₁ ∧ is_closed_set A₂ ∧ ¬(is_closed_set (A₁ ∪ A₂))) :=
by sorry

end NUMINAMATH_CALUDE_closed_set_properties_l167_16792


namespace NUMINAMATH_CALUDE_cistern_fill_time_with_leak_l167_16790

/-- Proves that a cistern with given fill and empty rates takes 2 additional hours to fill due to a leak -/
theorem cistern_fill_time_with_leak 
  (normal_fill_time : ℝ) 
  (empty_time : ℝ) 
  (h1 : normal_fill_time = 4) 
  (h2 : empty_time = 12) : 
  let fill_rate := 1 / normal_fill_time
  let leak_rate := 1 / empty_time
  let effective_rate := fill_rate - leak_rate
  let time_with_leak := 1 / effective_rate
  time_with_leak - normal_fill_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_with_leak_l167_16790


namespace NUMINAMATH_CALUDE_base12_addition_l167_16723

/-- Converts a base 12 number to base 10 --/
def toBase10 (x : ℕ) (y : ℕ) (z : ℕ) : ℕ := x * 144 + y * 12 + z

/-- Converts a base 10 number to base 12 --/
def toBase12 (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := n / 144
  let b := (n % 144) / 12
  let c := n % 12
  (a, b, c)

theorem base12_addition : 
  let x := toBase10 11 4 8  -- B48 in base 12
  let y := toBase10 5 7 10  -- 57A in base 12
  toBase12 (x + y) = (5, 11, 6) := by sorry

end NUMINAMATH_CALUDE_base12_addition_l167_16723


namespace NUMINAMATH_CALUDE_missing_score_is_86_l167_16729

def recorded_scores : List ℝ := [81, 73, 83, 73]
def mean : ℝ := 79.2
def total_games : ℕ := 5

theorem missing_score_is_86 :
  let total_sum := mean * total_games
  let recorded_sum := recorded_scores.sum
  total_sum - recorded_sum = 86 := by
  sorry

end NUMINAMATH_CALUDE_missing_score_is_86_l167_16729


namespace NUMINAMATH_CALUDE_smallest_class_number_is_four_l167_16785

/-- Represents a systematic sampling of classes. -/
structure SystematicSampling where
  total_classes : ℕ
  selected_classes : ℕ
  sum_of_selected : ℕ

/-- Calculates the smallest class number in a systematic sampling. -/
def smallest_class_number (s : SystematicSampling) : ℕ :=
  (s.sum_of_selected - s.selected_classes * (s.selected_classes - 1) * (s.total_classes / s.selected_classes) / 2) / s.selected_classes

/-- Theorem stating the smallest class number for the given conditions. -/
theorem smallest_class_number_is_four (s : SystematicSampling) 
  (h1 : s.total_classes = 24)
  (h2 : s.selected_classes = 4)
  (h3 : s.sum_of_selected = 52) :
  smallest_class_number s = 4 := by
  sorry

#eval smallest_class_number ⟨24, 4, 52⟩

end NUMINAMATH_CALUDE_smallest_class_number_is_four_l167_16785


namespace NUMINAMATH_CALUDE_bobbys_paycheck_l167_16704

/-- Calculates Bobby's final paycheck amount --/
def calculate_paycheck (salary : ℝ) (performance_rate : ℝ) (federal_tax_rate : ℝ) 
  (state_tax_rate : ℝ) (local_tax_rate : ℝ) (health_insurance : ℝ) (life_insurance : ℝ) 
  (parking_fee : ℝ) (retirement_rate : ℝ) : ℝ :=
  let bonus := salary * performance_rate
  let total_income := salary + bonus
  let federal_tax := total_income * federal_tax_rate
  let state_tax := total_income * state_tax_rate
  let local_tax := total_income * local_tax_rate
  let total_taxes := federal_tax + state_tax + local_tax
  let other_deductions := health_insurance + life_insurance + parking_fee
  let retirement_contribution := salary * retirement_rate
  total_income - total_taxes - other_deductions - retirement_contribution

/-- Theorem stating that Bobby's final paycheck amount is $176.98 --/
theorem bobbys_paycheck : 
  calculate_paycheck 450 0.12 (1/3) 0.08 0.05 50 20 10 0.03 = 176.98 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_paycheck_l167_16704


namespace NUMINAMATH_CALUDE_floor_division_equality_l167_16735

/-- Floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem: For any positive real number a and any integer n,
    the floor of (floor of a) divided by n is equal to the floor of a divided by n -/
theorem floor_division_equality (a : ℝ) (n : ℤ) (h1 : 0 < a) (h2 : n ≠ 0) :
  floor ((floor a : ℝ) / n) = floor (a / n) := by
  sorry

end NUMINAMATH_CALUDE_floor_division_equality_l167_16735


namespace NUMINAMATH_CALUDE_cylinder_volume_on_sphere_l167_16793

/-- Given a cylinder with height 1 and bases on a sphere of diameter 2, its volume is 3π/4 -/
theorem cylinder_volume_on_sphere (h : ℝ) (d : ℝ) (V : ℝ) : 
  h = 1 → d = 2 → V = (3 * Real.pi) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_on_sphere_l167_16793


namespace NUMINAMATH_CALUDE_chip_cost_calculation_l167_16742

def days_per_week : ℕ := 5
def weeks : ℕ := 4
def total_spent : ℚ := 10

def total_days : ℕ := days_per_week * weeks

theorem chip_cost_calculation :
  total_spent / total_days = 1/2 := by sorry

end NUMINAMATH_CALUDE_chip_cost_calculation_l167_16742


namespace NUMINAMATH_CALUDE_cake_bread_weight_difference_l167_16766

/-- Given that 4 cakes weigh 800 g and 3 cakes plus 5 pieces of bread weigh 1100 g,
    prove that a cake is 100 g heavier than a piece of bread. -/
theorem cake_bread_weight_difference :
  ∀ (cake_weight bread_weight : ℕ),
    4 * cake_weight = 800 →
    3 * cake_weight + 5 * bread_weight = 1100 →
    cake_weight - bread_weight = 100 := by
  sorry

end NUMINAMATH_CALUDE_cake_bread_weight_difference_l167_16766


namespace NUMINAMATH_CALUDE_message_spread_time_l167_16727

theorem message_spread_time (n : ℕ) : ∃ (m : ℕ), m ≥ 5 ∧ 2^(m+1) - 2 > 55 ∧ ∀ (k : ℕ), k < m → 2^(k+1) - 2 ≤ 55 := by
  sorry

end NUMINAMATH_CALUDE_message_spread_time_l167_16727


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l167_16797

-- Define the complex number z
variable (z : ℂ)

-- Define the condition
axiom z_condition : z * (1 + 2*Complex.I) = Complex.abs (4 - 3*Complex.I)

-- State the theorem
theorem imaginary_part_of_z :
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l167_16797


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l167_16755

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | x ≤ -3 ∨ x ≥ 4}

-- Theorem statement
theorem quadratic_inequality_theorem (a b c : ℝ) 
  (h : ∀ x, f a b c x ≥ 0 ↔ x ∈ solution_set a b c) : 
  (a > 0) ∧ 
  (∀ x, f c (-b) a x < 0 ↔ x < -1/4 ∨ x > 1/3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l167_16755


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l167_16775

/-- A parabola passing through the points (-1, -6) and (1, 0) -/
def Parabola (x y : ℝ) : Prop :=
  ∃ (m n : ℝ), y = x^2 + m*x + n ∧ -6 = 1 - m + n ∧ 0 = 1 + m + n

/-- The intersection point of the parabola with the y-axis -/
def YAxisIntersection (x y : ℝ) : Prop :=
  Parabola x y ∧ x = 0

theorem parabola_y_axis_intersection :
  ∀ x y, YAxisIntersection x y → x = 0 ∧ y = -4 := by sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l167_16775


namespace NUMINAMATH_CALUDE_number_of_outfits_l167_16796

/-- The number of shirts of each color -/
def shirts_per_color : ℕ := 4

/-- The number of pants -/
def pants : ℕ := 7

/-- The number of hats of each color -/
def hats_per_color : ℕ := 6

/-- The number of colors -/
def colors : ℕ := 3

/-- Theorem: The number of outfits with different colored shirts and hats -/
theorem number_of_outfits : 
  shirts_per_color * pants * hats_per_color * (colors - 1) = 1008 := by
  sorry

end NUMINAMATH_CALUDE_number_of_outfits_l167_16796


namespace NUMINAMATH_CALUDE_income_spent_on_food_l167_16739

/-- Proves the percentage of income spent on food given other expenses -/
theorem income_spent_on_food (F : ℝ) : 
  F ≥ 0 ∧ F ≤ 100 →
  (100 - F - 25 - 0.8 * (75 - 0.75 * F) = 8) →
  F = 46.67 := by
  sorry

end NUMINAMATH_CALUDE_income_spent_on_food_l167_16739


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l167_16758

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem tenth_term_of_sequence (a₁ a₂ : ℤ) (h₁ : a₁ = 2) (h₂ : a₂ = 1) :
  arithmetic_sequence a₁ (a₂ - a₁) 10 = -7 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l167_16758


namespace NUMINAMATH_CALUDE_cone_volume_from_sector_l167_16757

/-- Given a cone whose lateral surface develops into a sector with central angle 4π/3 and area 6π,
    the volume of the cone is (4√5/3)π. -/
theorem cone_volume_from_sector (θ r l h V : ℝ) : 
  θ = (4 / 3) * Real.pi →  -- Central angle of the sector
  (1 / 2) * l^2 * θ = 6 * Real.pi →  -- Area of the sector
  2 * Real.pi * r = θ * l →  -- Circumference of base equals arc length of sector
  h^2 + r^2 = l^2 →  -- Pythagorean theorem for cone dimensions
  V = (1 / 3) * Real.pi * r^2 * h →  -- Volume formula for cone
  V = (4 * Real.sqrt 5 / 3) * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_sector_l167_16757


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l167_16736

/-- A cube with volume 8x cubic units and surface area 4x square units has x = 5400 --/
theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 4*x) → x = 5400 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l167_16736


namespace NUMINAMATH_CALUDE_f_properties_l167_16730

noncomputable def f (x : ℝ) : ℝ := x / (1 - abs x)

noncomputable def g (x : ℝ) : ℝ := f x + x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ y, ∃ x, f x = y) ∧   -- range of f is ℝ
  (∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ g a = 0 ∧ g b = 0 ∧ g c = 0) ∧  -- g has exactly three zeros
  (∀ x₁ x₂, x₁ ≠ x₂ → ¬(f x₁ ≠ f x₂))  -- negation of the incorrect statement
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l167_16730


namespace NUMINAMATH_CALUDE_sphere_radius_is_sqrt_six_over_four_l167_16706

/-- A sphere circumscribing a right circular cone -/
structure CircumscribedCone where
  /-- The radius of the circumscribing sphere -/
  sphere_radius : ℝ
  /-- The diameter of the base of the cone -/
  base_diameter : ℝ
  /-- Assertion that the base diameter is 1 -/
  base_diameter_is_one : base_diameter = 1
  /-- Assertion that the apex of the cone is on the sphere -/
  apex_on_sphere : True
  /-- Assertion about the perpendicularity condition -/
  perpendicular_condition : True

/-- Theorem stating that the radius of the circumscribing sphere is √6/4 -/
theorem sphere_radius_is_sqrt_six_over_four (cone : CircumscribedCone) :
  cone.sphere_radius = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_is_sqrt_six_over_four_l167_16706


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_cosine_sum_l167_16741

theorem seventh_root_of_unity_cosine_sum (z : ℂ) (α : ℝ) (h1 : z ^ 7 = 1) (h2 : z ≠ 1) (h3 : z = Complex.exp (Complex.I * α)) :
  Real.cos α + Real.cos (2 * α) + Real.cos (4 * α) = -1/2 := by sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_cosine_sum_l167_16741


namespace NUMINAMATH_CALUDE_inequality_solution_set_l167_16715

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2*(a - 2) * x < 4) ↔ a ∈ Set.Ioc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l167_16715


namespace NUMINAMATH_CALUDE_optimal_workers_theorem_l167_16737

/-- The number of workers that should process part P to minimize processing time -/
def optimal_workers_for_P (total_P : ℕ) (total_Q : ℕ) (total_workers : ℕ) 
  (P_rate : ℚ) (Q_rate : ℚ) : ℕ :=
  137

/-- The theorem stating that 137 workers should process part P for optimal time -/
theorem optimal_workers_theorem (total_P : ℕ) (total_Q : ℕ) (total_workers : ℕ) 
  (P_rate : ℚ) (Q_rate : ℚ) :
  total_P = 6000 →
  total_Q = 2000 →
  total_workers = 214 →
  5 * P_rate = 3 * Q_rate →
  optimal_workers_for_P total_P total_Q total_workers P_rate Q_rate = 137 :=
by
  sorry

#check optimal_workers_theorem

end NUMINAMATH_CALUDE_optimal_workers_theorem_l167_16737


namespace NUMINAMATH_CALUDE_problem_statement_l167_16708

theorem problem_statement (a b : ℝ) : (a - 1)^2 + |b + 2| = 0 → (a + b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l167_16708


namespace NUMINAMATH_CALUDE_problem_1_l167_16749

theorem problem_1 : -3⁻¹ * Real.sqrt 27 + |1 - Real.sqrt 3| + (-1)^2023 = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l167_16749


namespace NUMINAMATH_CALUDE_calculate_expression_l167_16733

theorem calculate_expression : 
  (8/27)^(2/3) + Real.log 3 / Real.log 12 + 2 * Real.log 2 / Real.log 12 = 13/9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l167_16733


namespace NUMINAMATH_CALUDE_smallest_blue_chips_l167_16781

theorem smallest_blue_chips (total : ℕ) (h_total : total = 49) :
  ∃ (blue red prime : ℕ),
    blue + red = total ∧
    red = blue + prime ∧
    Nat.Prime prime ∧
    ∀ (b r p : ℕ), b + r = total → r = b + p → Nat.Prime p → blue ≤ b :=
by sorry

end NUMINAMATH_CALUDE_smallest_blue_chips_l167_16781


namespace NUMINAMATH_CALUDE_tax_rate_is_ten_percent_l167_16776

/-- Calculates the tax rate given the total amount spent, sales tax, and cost of tax-free items -/
def calculate_tax_rate (total_amount : ℚ) (sales_tax : ℚ) (tax_free_cost : ℚ) : ℚ :=
  let taxable_cost := total_amount - tax_free_cost - sales_tax
  (sales_tax / taxable_cost) * 100

/-- Theorem stating that the tax rate is 10% given the problem conditions -/
theorem tax_rate_is_ten_percent 
  (total_amount : ℚ) 
  (sales_tax : ℚ) 
  (tax_free_cost : ℚ)
  (h1 : total_amount = 25)
  (h2 : sales_tax = 3/10)
  (h3 : tax_free_cost = 217/10) :
  calculate_tax_rate total_amount sales_tax tax_free_cost = 10 := by
  sorry

#eval calculate_tax_rate 25 (3/10) (217/10)

end NUMINAMATH_CALUDE_tax_rate_is_ten_percent_l167_16776


namespace NUMINAMATH_CALUDE_cubic_difference_equals_2011_l167_16713

theorem cubic_difference_equals_2011 (x y : ℕ+) (h : x.val^2 - y.val^2 = 53) :
  x.val^3 - y.val^3 - 2 * (x.val + y.val) + 10 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_equals_2011_l167_16713


namespace NUMINAMATH_CALUDE_green_beans_weight_l167_16731

theorem green_beans_weight (green_beans rice sugar : ℝ) 
  (h1 : rice = green_beans - 30)
  (h2 : sugar = green_beans - 10)
  (h3 : (2/3) * rice + green_beans + (4/5) * sugar = 120) :
  green_beans = 60 := by
  sorry

end NUMINAMATH_CALUDE_green_beans_weight_l167_16731


namespace NUMINAMATH_CALUDE_f_min_value_l167_16783

/-- The function f(x) = |3-x| + |x-7| -/
def f (x : ℝ) := |3 - x| + |x - 7|

/-- The minimum value of f(x) is 4 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ 4 ∧ ∃ y : ℝ, f y = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l167_16783


namespace NUMINAMATH_CALUDE_regular_nonagon_side_equals_diagonal_difference_l167_16711

/-- A regular nonagon -/
structure RegularNonagon where
  -- Define the necessary properties of a regular nonagon
  side_length : ℝ
  longest_diagonal : ℝ
  shortest_diagonal : ℝ
  side_length_pos : 0 < side_length
  longest_diagonal_pos : 0 < longest_diagonal
  shortest_diagonal_pos : 0 < shortest_diagonal
  longest_ge_shortest : shortest_diagonal ≤ longest_diagonal

/-- 
The side length of a regular nonagon is equal to the difference 
between its longest diagonal and shortest diagonal 
-/
theorem regular_nonagon_side_equals_diagonal_difference 
  (n : RegularNonagon) : 
  n.side_length = n.longest_diagonal - n.shortest_diagonal :=
sorry

end NUMINAMATH_CALUDE_regular_nonagon_side_equals_diagonal_difference_l167_16711


namespace NUMINAMATH_CALUDE_john_remaining_money_l167_16752

def john_savings : ℕ := 5555
def ticket_cost : ℕ := 1200
def visa_cost : ℕ := 200

def base_8_to_10 (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem john_remaining_money :
  base_8_to_10 john_savings - ticket_cost - visa_cost = 1525 := by sorry

end NUMINAMATH_CALUDE_john_remaining_money_l167_16752


namespace NUMINAMATH_CALUDE_lowest_price_for_electronic_component_l167_16746

/-- Calculates the lowest price per component to break even -/
def lowest_break_even_price (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (units_sold : ℕ) : ℚ :=
  (production_cost + shipping_cost + (fixed_costs / units_sold))

theorem lowest_price_for_electronic_component :
  let production_cost : ℚ := 80
  let shipping_cost : ℚ := 3
  let fixed_costs : ℚ := 16500
  let units_sold : ℕ := 150
  lowest_break_even_price production_cost shipping_cost fixed_costs units_sold = 193 := by
sorry

#eval lowest_break_even_price 80 3 16500 150

end NUMINAMATH_CALUDE_lowest_price_for_electronic_component_l167_16746


namespace NUMINAMATH_CALUDE_no_matrix_satisfies_condition_l167_16703

theorem no_matrix_satisfies_condition : 
  ¬∃ (N : Matrix (Fin 2) (Fin 2) ℝ), 
    ∀ (x y z w : ℝ), 
      N * !![x, y; z, w] = !![2*x, 3*y; 4*z, 5*w] := by
sorry

end NUMINAMATH_CALUDE_no_matrix_satisfies_condition_l167_16703


namespace NUMINAMATH_CALUDE_cylinder_height_l167_16784

/-- The height of a right cylinder with radius 3 feet and surface area 36π square feet is 3 feet. -/
theorem cylinder_height (r h : ℝ) : 
  r = 3 → 2 * Real.pi * r^2 + 2 * Real.pi * r * h = 36 * Real.pi → h = 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_l167_16784


namespace NUMINAMATH_CALUDE_rice_sales_profit_l167_16740

-- Define the linear function
def sales_function (a b x : ℝ) : ℝ := a * x + b

-- Define the profit function
def profit_function (x y : ℝ) : ℝ := (x - 4) * y

-- Define the theorem
theorem rice_sales_profit 
  (a b : ℝ) 
  (h1 : ∀ x, 4 ≤ x → x ≤ 7 → sales_function a b x ≥ 0)
  (h2 : sales_function a b 5 = 950)
  (h3 : sales_function a b 6 = 900) :
  (a = -50 ∧ b = 1200) ∧
  (profit_function 6 (sales_function a b 6) = 1800) ∧
  (∀ x, 4 ≤ x → x ≤ 7 → profit_function x (sales_function a b x) ≤ 2550) ∧
  (profit_function 7 (sales_function a b 7) = 2550) := by
  sorry

end NUMINAMATH_CALUDE_rice_sales_profit_l167_16740


namespace NUMINAMATH_CALUDE_inequality_system_solution_l167_16786

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x - a > 1 ∧ 2*x - 3 > a) ↔ x > a + 1) → a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l167_16786


namespace NUMINAMATH_CALUDE_remainder_sum_l167_16767

theorem remainder_sum (a b : ℤ) 
  (ha : a % 60 = 53) 
  (hb : b % 50 = 24) : 
  (a + b) % 20 = 17 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l167_16767


namespace NUMINAMATH_CALUDE_trigonometric_relationship_l167_16718

def relationship (x y z : ℝ) : Prop :=
  z^4 - 2*z^2*(x^2 + y^2 - 2*x^2*y^2) + (x^2 - y^2)^2 = 0

theorem trigonometric_relationship 
  (x y : ℝ) (hx : x ∈ Set.Icc (-1 : ℝ) 1) (hy : y ∈ Set.Icc (-1 : ℝ) 1) :
  ∃ (z₁ z₂ z₃ z₄ : ℝ), 
    (∀ z, relationship x y z ↔ z = z₁ ∨ z = z₂ ∨ z = z₃ ∨ z = z₄) ∧
    (x = y ∨ x = -y) → (x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) → 
      (∃ (w₁ w₂ w₃ : ℝ), ∀ z, relationship x y z ↔ z = w₁ ∨ z = w₂ ∨ z = w₃) ∧
    (x = 0 ∨ x = 1 ∨ x = -1 ∨ y = 0 ∨ y = 1 ∨ y = -1) → 
      (∃ (v₁ v₂ : ℝ), ∀ z, relationship x y z ↔ z = v₁ ∨ z = v₂) ∧
    ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ 
     (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) → 
      (∃ (u : ℝ), ∀ z, relationship x y z ↔ z = u) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_relationship_l167_16718


namespace NUMINAMATH_CALUDE_greatest_number_of_fruit_baskets_l167_16744

theorem greatest_number_of_fruit_baskets : Nat.gcd (Nat.gcd 18 27) 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_of_fruit_baskets_l167_16744


namespace NUMINAMATH_CALUDE_tangent_and_inequality_imply_m_range_l167_16770

open Real

noncomputable def f (x : ℝ) : ℝ := x / (Real.exp x)

theorem tangent_and_inequality_imply_m_range :
  (∀ x ∈ Set.Ioo (1/2) (3/2), f x < 1 / (m + 6*x - 3*x^2)) →
  m ∈ Set.Icc (-9/4) (ℯ - 3) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_inequality_imply_m_range_l167_16770


namespace NUMINAMATH_CALUDE_original_number_is_ten_l167_16754

theorem original_number_is_ten : ∃ x : ℝ, 3 * (2 * x + 8) = 84 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_ten_l167_16754


namespace NUMINAMATH_CALUDE_music_students_percentage_l167_16787

/-- Given a total number of students and the number of students taking dance and art,
    prove that the percentage of students taking music is 20%. -/
theorem music_students_percentage
  (total : ℕ)
  (dance : ℕ)
  (art : ℕ)
  (h1 : total = 400)
  (h2 : dance = 120)
  (h3 : art = 200) :
  (total - dance - art) / total * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_music_students_percentage_l167_16787


namespace NUMINAMATH_CALUDE_elena_garden_petals_l167_16762

/-- The number of lilies in Elena's garden -/
def num_lilies : ℕ := 8

/-- The number of tulips in Elena's garden -/
def num_tulips : ℕ := 5

/-- The number of petals on each lily -/
def petals_per_lily : ℕ := 6

/-- The number of petals on each tulip -/
def petals_per_tulip : ℕ := 3

/-- The total number of flower petals in Elena's garden -/
def total_petals : ℕ := num_lilies * petals_per_lily + num_tulips * petals_per_tulip

theorem elena_garden_petals : total_petals = 63 := by
  sorry

end NUMINAMATH_CALUDE_elena_garden_petals_l167_16762


namespace NUMINAMATH_CALUDE_volume_ratio_is_three_to_five_l167_16751

/-- A square-based right pyramid where side faces form a 60° angle with the base -/
structure SquarePyramid where
  base_side : ℝ
  side_face_angle : ℝ
  side_face_angle_eq : side_face_angle = π / 3

/-- The ratio of volumes created by the bisector plane -/
def volume_ratio (p : SquarePyramid) : ℝ × ℝ := sorry

/-- Theorem: The ratio of volumes is 3:5 -/
theorem volume_ratio_is_three_to_five (p : SquarePyramid) : 
  volume_ratio p = (3, 5) := by sorry

end NUMINAMATH_CALUDE_volume_ratio_is_three_to_five_l167_16751


namespace NUMINAMATH_CALUDE_tan_105_degrees_l167_16769

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l167_16769


namespace NUMINAMATH_CALUDE_negation_proposition_l167_16724

theorem negation_proposition :
  (∀ x : ℝ, x < 0 → x^2 ≤ 0) ↔ ¬(∃ x₀ : ℝ, x₀ < 0 ∧ x₀^2 > 0) :=
sorry

end NUMINAMATH_CALUDE_negation_proposition_l167_16724


namespace NUMINAMATH_CALUDE_f_positive_solution_a_range_l167_16710

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 3|

-- Theorem for the solution of f(x) > 0
theorem f_positive_solution :
  ∀ x : ℝ, f x > 0 ↔ x < -4 ∨ x > 2/3 := by sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∀ x : ℝ, a - 3*|x - 3| < f x) ↔ a < 7 := by sorry

end NUMINAMATH_CALUDE_f_positive_solution_a_range_l167_16710


namespace NUMINAMATH_CALUDE_max_non_managers_l167_16782

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 9 →
  (managers : ℚ) / non_managers > 7 / 32 →
  non_managers ≤ 41 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l167_16782


namespace NUMINAMATH_CALUDE_age_ratio_proof_l167_16714

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 42 →  -- The total of the ages of a, b, and c is 42
  b = 16 →  -- b is 16 years old
  b = 2 * c  -- The ratio of b's age to c's age is 2:1
  := by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l167_16714


namespace NUMINAMATH_CALUDE_program_output_correct_l167_16738

def program_transformation (a₀ b₀ c₀ : ℕ) : ℕ × ℕ × ℕ :=
  let a₁ := b₀
  let b₁ := c₀
  let c₁ := a₁
  (a₁, b₁, c₁)

theorem program_output_correct :
  program_transformation 2 3 4 = (3, 4, 3) := by sorry

end NUMINAMATH_CALUDE_program_output_correct_l167_16738


namespace NUMINAMATH_CALUDE_equation_solution_l167_16719

theorem equation_solution : 
  ∃ x : ℚ, (2*x + 1)/4 - 1 = x - (10*x + 1)/12 ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l167_16719


namespace NUMINAMATH_CALUDE_angle_A_magnitude_max_area_l167_16748

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.cos t.C = (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A

-- Theorem for part I
theorem angle_A_magnitude (t : Triangle) (h : given_condition t) : t.A = π / 6 :=
sorry

-- Theorem for part II
theorem max_area (t : Triangle) (h1 : given_condition t) (h2 : t.a = 2) :
  ∃ (area : ℝ), area ≤ 2 + Real.sqrt 3 ∧
  ∀ (other_area : ℝ), (∃ (t' : Triangle), t'.a = 2 ∧ given_condition t' ∧ 
    other_area = (1 / 2) * t'.b * t'.c * Real.sin t'.A) → other_area ≤ area :=
sorry

end NUMINAMATH_CALUDE_angle_A_magnitude_max_area_l167_16748


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l167_16759

theorem polynomial_factor_implies_coefficients 
  (p q : ℚ) 
  (h : ∃ (a b : ℚ), px^4 + qx^3 + 45*x^2 - 25*x + 10 = (5*x^2 - 3*x + 2)*(a*x^2 + b*x + 5)) :
  p = 25/2 ∧ q = -65/2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l167_16759


namespace NUMINAMATH_CALUDE_mice_pairing_impossible_l167_16701

/-- Represents the number of mice in the family -/
def total_mice : ℕ := 24

/-- Represents the number of mice that go to the warehouse each night -/
def mice_per_night : ℕ := 4

/-- Represents the number of new pairings a mouse makes each night -/
def new_pairings_per_night : ℕ := mice_per_night - 1

/-- Represents the number of pairings each mouse needs to make -/
def required_pairings : ℕ := total_mice - 1

/-- Theorem stating that it's impossible for each mouse to pair with every other mouse exactly once -/
theorem mice_pairing_impossible : 
  ¬(required_pairings % new_pairings_per_night = 0) := by sorry

end NUMINAMATH_CALUDE_mice_pairing_impossible_l167_16701


namespace NUMINAMATH_CALUDE_average_string_length_l167_16750

theorem average_string_length : 
  let string1 : ℝ := 2.5
  let string2 : ℝ := 3.5
  let string3 : ℝ := 4.5
  let total_length : ℝ := string1 + string2 + string3
  let num_strings : ℕ := 3
  (total_length / num_strings) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_average_string_length_l167_16750


namespace NUMINAMATH_CALUDE_josh_marbles_count_l167_16771

def final_marbles (initial found traded broken : ℕ) : ℕ :=
  initial + found - traded - broken

theorem josh_marbles_count : final_marbles 357 146 32 10 = 461 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_count_l167_16771


namespace NUMINAMATH_CALUDE_train_length_l167_16779

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 15 → speed * time * (5 / 18) = 375 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l167_16779


namespace NUMINAMATH_CALUDE_quadratic_sum_l167_16728

/-- The quadratic function under consideration -/
def f (x : ℝ) : ℝ := 4 * x^2 - 48 * x - 128

/-- The same quadratic function in completed square form -/
def g (x : ℝ) (a b c : ℝ) : ℝ := a * (x + b)^2 + c

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, f x = g x a b c) → a + b + c = -274 := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l167_16728


namespace NUMINAMATH_CALUDE_root_property_l167_16778

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 4*x - a

-- State the theorem
theorem root_property (a : ℝ) (x₁ x₂ x₃ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < 2)
  (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) (h₅ : f a x₃ = 0)
  (h₆ : x₁ < x₂) (h₇ : x₂ < x₃) :
  x₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l167_16778


namespace NUMINAMATH_CALUDE_circle_constant_l167_16709

/-- A circle in the xy-plane defined by the equation x^2 - 8x + y^2 + 10y + c = 0 with radius 5 -/
def Circle (c : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + c = 0 ↔ (x - 4)^2 + (y + 5)^2 = 25

theorem circle_constant : ∃! c : ℝ, Circle c :=
  sorry

end NUMINAMATH_CALUDE_circle_constant_l167_16709


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l167_16700

theorem sum_with_radical_conjugate :
  (12 - Real.sqrt 2023) + (12 + Real.sqrt 2023) = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l167_16700


namespace NUMINAMATH_CALUDE_fair_division_of_walls_l167_16763

/-- The number of people in Amanda's family -/
def family_size : ℕ := 5

/-- The number of rooms with 4 walls -/
def rooms_with_4_walls : ℕ := 5

/-- The number of rooms with 5 walls -/
def rooms_with_5_walls : ℕ := 4

/-- The total number of walls in the house -/
def total_walls : ℕ := rooms_with_4_walls * 4 + rooms_with_5_walls * 5

/-- The number of walls each person should paint for fair division -/
def walls_per_person : ℕ := total_walls / family_size

theorem fair_division_of_walls :
  walls_per_person = 8 := by sorry

end NUMINAMATH_CALUDE_fair_division_of_walls_l167_16763


namespace NUMINAMATH_CALUDE_kerosene_mixture_problem_l167_16716

/-- A mixture problem involving two liquids with different kerosene concentrations -/
theorem kerosene_mixture_problem :
  let first_liquid_kerosene_percent : ℝ := 25
  let second_liquid_kerosene_percent : ℝ := 30
  let second_liquid_parts : ℝ := 4
  let mixture_kerosene_percent : ℝ := 27
  let first_liquid_parts : ℝ := 6

  first_liquid_kerosene_percent / 100 * first_liquid_parts +
  second_liquid_kerosene_percent / 100 * second_liquid_parts =
  mixture_kerosene_percent / 100 * (first_liquid_parts + second_liquid_parts) :=
by sorry

end NUMINAMATH_CALUDE_kerosene_mixture_problem_l167_16716


namespace NUMINAMATH_CALUDE_deductive_reasoning_syllogism_form_l167_16717

/-- Represents the characteristics of deductive reasoning -/
structure DeductiveReasoning where
  generalToSpecific : Bool
  alwaysCorrect : Bool
  syllogismForm : Bool
  dependsOnPremisesAndForm : Bool

/-- Theorem stating that the general pattern of deductive reasoning is the syllogism form -/
theorem deductive_reasoning_syllogism_form (dr : DeductiveReasoning) :
  dr.generalToSpecific ∧
  ¬dr.alwaysCorrect ∧
  dr.dependsOnPremisesAndForm →
  dr.syllogismForm :=
by sorry

end NUMINAMATH_CALUDE_deductive_reasoning_syllogism_form_l167_16717


namespace NUMINAMATH_CALUDE_perpendicular_lines_l167_16756

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, 2 * y + x + 3 = 0 ∧ 3 * y + a * x + 2 = 0 → 
    ((-1/2) * (-a/3) = -1)) → 
  a = -6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l167_16756


namespace NUMINAMATH_CALUDE_function_transformation_l167_16799

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 0) : 
  f 1 + 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_function_transformation_l167_16799


namespace NUMINAMATH_CALUDE_equation_solution_l167_16712

theorem equation_solution : ∃ x : ℝ, 7 * (4 * x + 3) - 9 = -3 * (2 - 9 * x) + 5 * x ∧ x = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l167_16712


namespace NUMINAMATH_CALUDE_trig_identity_proof_l167_16734

theorem trig_identity_proof (a : ℝ) : 
  Real.cos (a + π/6) * Real.sin (a - π/3) + Real.sin (a + π/6) * Real.cos (a - π/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l167_16734


namespace NUMINAMATH_CALUDE_eggs_solution_l167_16753

def eggs_problem (dozen_count : ℕ) (price_per_egg : ℚ) (tax_rate : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_eggs := dozen_count * 12
  let original_cost := total_eggs * price_per_egg
  let discounted_cost := original_cost * (1 - discount_rate)
  let tax_amount := discounted_cost * tax_rate
  discounted_cost + tax_amount

theorem eggs_solution :
  eggs_problem 3 (1/2) (5/100) (10/100) = 1701/100 := by
  sorry

end NUMINAMATH_CALUDE_eggs_solution_l167_16753


namespace NUMINAMATH_CALUDE_h_odd_f_increasing_inequality_solution_l167_16777

noncomputable section

variable (f : ℝ → ℝ)
variable (h : ℝ → ℝ)

axiom f_property : ∀ x y : ℝ, f x + f y = f (x + y) + 1
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 1
axiom h_def : ∀ x : ℝ, h x = f x - 1

theorem h_odd : ∀ x : ℝ, h (-x) = -h x := by sorry

theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ > x₂ → f x₁ > f x₂ := by sorry

theorem inequality_solution (t : ℝ) :
  (t = 1 → ¬∃ x, f (x^2) - f (3*t*x) + f (2*t^2 + 2*t - x) < 1) ∧
  (t > 1 → ∀ x, f (x^2) - f (3*t*x) + f (2*t^2 + 2*t - x) < 1 ↔ t+1 < x ∧ x < 2*t) ∧
  (t < 1 → ∀ x, f (x^2) - f (3*t*x) + f (2*t^2 + 2*t - x) < 1 ↔ 2*t < x ∧ x < t+1) := by sorry

end

end NUMINAMATH_CALUDE_h_odd_f_increasing_inequality_solution_l167_16777


namespace NUMINAMATH_CALUDE_gary_shortage_l167_16761

def gary_initial_amount : ℝ := 73
def snake_cost : ℝ := 55
def snake_food_cost : ℝ := 12
def habitat_original_cost : ℝ := 35
def habitat_discount_rate : ℝ := 0.15

def total_spent : ℝ := snake_cost + snake_food_cost + 
  (habitat_original_cost * (1 - habitat_discount_rate))

theorem gary_shortage : 
  total_spent - gary_initial_amount = 23.75 := by sorry

end NUMINAMATH_CALUDE_gary_shortage_l167_16761


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_container_l167_16774

def container_capacity : ℕ := 1125
def bottle_type1_capacity : ℕ := 45
def bottle_type2_capacity : ℕ := 75

theorem min_bottles_to_fill_container :
  ∃ (n1 n2 : ℕ),
    n1 * bottle_type1_capacity + n2 * bottle_type2_capacity = container_capacity ∧
    ∀ (m1 m2 : ℕ), 
      m1 * bottle_type1_capacity + m2 * bottle_type2_capacity = container_capacity →
      n1 + n2 ≤ m1 + m2 ∧
    n1 + n2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_bottles_to_fill_container_l167_16774


namespace NUMINAMATH_CALUDE_estimate_fish_population_l167_16726

/-- Estimates the number of fish in a lake using the mark-recapture method. -/
theorem estimate_fish_population (initial_marked : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  initial_marked = 100 →
  second_catch = 200 →
  marked_in_second = 25 →
  (initial_marked * second_catch) / marked_in_second = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l167_16726


namespace NUMINAMATH_CALUDE_rachel_lost_lives_l167_16732

/- Define the initial number of lives -/
def initial_lives : ℕ := 10

/- Define the number of lives gained -/
def lives_gained : ℕ := 26

/- Define the final number of lives -/
def final_lives : ℕ := 32

/- Theorem: Rachel lost 4 lives in the hard part -/
theorem rachel_lost_lives :
  ∃ (lives_lost : ℕ), initial_lives - lives_lost + lives_gained = final_lives ∧ lives_lost = 4 := by
  sorry

end NUMINAMATH_CALUDE_rachel_lost_lives_l167_16732


namespace NUMINAMATH_CALUDE_function_decreasing_iff_a_in_range_l167_16722

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 - 2*a)^x else Real.log x / Real.log a + 1/3

theorem function_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) ↔ 0 < a ∧ a ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_function_decreasing_iff_a_in_range_l167_16722


namespace NUMINAMATH_CALUDE_odd_function_sum_l167_16773

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_sum (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f 4 = 5) :
  f 4 + f (-4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l167_16773


namespace NUMINAMATH_CALUDE_sum_123_consecutive_even_from_2_l167_16772

/-- Sum of consecutive even numbers -/
def sum_consecutive_even (start : ℕ) (count : ℕ) : ℕ :=
  count * (2 * start + (count - 1) * 2) / 2

/-- Theorem: The sum of 123 consecutive even numbers starting from 2 is 15252 -/
theorem sum_123_consecutive_even_from_2 :
  sum_consecutive_even 2 123 = 15252 := by
  sorry

end NUMINAMATH_CALUDE_sum_123_consecutive_even_from_2_l167_16772


namespace NUMINAMATH_CALUDE_martha_clothes_count_l167_16747

/-- Calculates the total number of clothes Martha takes home given the number of jackets and t-shirts bought -/
def total_clothes (jackets_bought : ℕ) (tshirts_bought : ℕ) : ℕ :=
  let free_jackets := jackets_bought / 2
  let free_tshirts := tshirts_bought / 3
  jackets_bought + free_jackets + tshirts_bought + free_tshirts

/-- Proves that Martha takes home 18 clothes when buying 4 jackets and 9 t-shirts -/
theorem martha_clothes_count : total_clothes 4 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_martha_clothes_count_l167_16747


namespace NUMINAMATH_CALUDE_motorboat_speed_l167_16745

/-- Prove that the maximum speed of a motorboat in still water is 40 km/h given the specified conditions -/
theorem motorboat_speed (flood_rate : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  flood_rate = 10 →
  downstream_distance = 2 →
  upstream_distance = 1.2 →
  (downstream_distance / (v + flood_rate) = upstream_distance / (v - flood_rate)) →
  v = 40 :=
by
  sorry

#check motorboat_speed

end NUMINAMATH_CALUDE_motorboat_speed_l167_16745


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_55_l167_16705

/-- The sum of n consecutive positive integers starting from a -/
def consecutiveSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- Predicate to check if a sequence of n consecutive integers starting from a sums to 55 -/
def isValidSequence (a n : ℕ) : Prop :=
  a > 0 ∧ consecutiveSum a n = 55

theorem max_consecutive_integers_sum_55 :
  (∃ a : ℕ, isValidSequence a 10) ∧
  (∀ n : ℕ, n > 10 → ¬∃ a : ℕ, isValidSequence a n) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_55_l167_16705


namespace NUMINAMATH_CALUDE_sales_solution_l167_16788

def sales_problem (s1 s2 s3 s5 s6 average : ℕ) : Prop :=
  let total := average * 6
  let known_sum := s1 + s2 + s3 + s5 + s6
  total - known_sum = 6122

theorem sales_solution :
  sales_problem 5266 5744 5864 6588 4916 5750 := by
  sorry

end NUMINAMATH_CALUDE_sales_solution_l167_16788


namespace NUMINAMATH_CALUDE_intersection_of_sets_l167_16765

theorem intersection_of_sets (A B : Set ℝ) (a : ℝ) : 
  A = {-1, 0, 1} → 
  B = {a + 1, 2 * a} → 
  A ∩ B = {0} → 
  a = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l167_16765


namespace NUMINAMATH_CALUDE_factorial_division_l167_16760

theorem factorial_division :
  (10 : ℕ).factorial = 3628800 →
  (10 : ℕ).factorial / (4 : ℕ).factorial = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l167_16760


namespace NUMINAMATH_CALUDE_pool_filling_solution_l167_16768

/-- Represents the pool filling problem -/
def PoolFilling (totalVolume fillRate initialTime leakRate : ℝ) : Prop :=
  let initialVolume := fillRate * initialTime
  let remainingVolume := totalVolume - initialVolume
  let netFillRate := fillRate - leakRate
  let additionalTime := remainingVolume / netFillRate
  initialTime + additionalTime = 220

/-- Theorem stating the solution to the pool filling problem -/
theorem pool_filling_solution :
  PoolFilling 4000 20 20 2 := by sorry

end NUMINAMATH_CALUDE_pool_filling_solution_l167_16768


namespace NUMINAMATH_CALUDE_constant_distance_between_bikers_l167_16791

def distance_between_bikers (t : ℝ) (initial_distance : ℝ) (speed_a : ℝ) (speed_b : ℝ) : ℝ :=
  initial_distance + speed_b * t - speed_a * t

theorem constant_distance_between_bikers
  (speed_a : ℝ)
  (speed_b : ℝ)
  (initial_distance : ℝ)
  (h1 : speed_a = 350 / 7)
  (h2 : speed_b = 500 / 10)
  (h3 : initial_distance = 75)
  (t : ℝ) :
  distance_between_bikers t initial_distance speed_a speed_b = initial_distance :=
by sorry

end NUMINAMATH_CALUDE_constant_distance_between_bikers_l167_16791


namespace NUMINAMATH_CALUDE_angle_D_measure_l167_16702

-- Define a scalene triangle DEF
structure ScaleneTriangle where
  D : ℝ
  E : ℝ
  F : ℝ
  scalene : D ≠ E ∧ E ≠ F ∧ D ≠ F
  sum_180 : D + E + F = 180

-- Theorem statement
theorem angle_D_measure (t : ScaleneTriangle) 
  (h1 : t.D = 2 * t.E) 
  (h2 : t.F = t.E - 20) : 
  t.D = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l167_16702


namespace NUMINAMATH_CALUDE_solve_for_s_l167_16794

theorem solve_for_s (s t : ℝ) 
  (eq1 : 8 * s + 4 * t = 160) 
  (eq2 : t = 2 * s - 3) : 
  s = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_s_l167_16794


namespace NUMINAMATH_CALUDE_tan_alpha_value_l167_16798

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = -5) :
  Real.tan α = -23/16 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l167_16798


namespace NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l167_16707

theorem tens_digit_of_2023_pow_2024_minus_2025 :
  (2023^2024 - 2025) % 100 / 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l167_16707


namespace NUMINAMATH_CALUDE_games_mike_can_buy_l167_16795

/-- The maximum number of games that can be bought given initial money, spent money, and game cost. -/
def max_games_buyable (initial_money spent_money game_cost : ℕ) : ℕ :=
  (initial_money - spent_money) / game_cost

/-- Theorem stating that given the specific values in the problem, the maximum number of games that can be bought is 4. -/
theorem games_mike_can_buy :
  max_games_buyable 42 10 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_games_mike_can_buy_l167_16795


namespace NUMINAMATH_CALUDE_exists_solution_l167_16720

theorem exists_solution : ∃ x : ℝ, x + 2.75 + 0.158 = 2.911 := by sorry

end NUMINAMATH_CALUDE_exists_solution_l167_16720


namespace NUMINAMATH_CALUDE_movie_watching_times_l167_16721

/-- Represents the duration of the movie in minutes -/
def movie_duration : ℕ := 120

/-- Represents the time difference in minutes between when Camila and Maverick started watching -/
def camila_maverick_diff : ℕ := 30

/-- Represents the time difference in minutes between when Maverick and Daniella started watching -/
def maverick_daniella_diff : ℕ := 45

/-- Represents the number of minutes Daniella has left to watch -/
def daniella_remaining : ℕ := 30

/-- Theorem stating that Camila and Maverick have finished watching when Daniella has 30 minutes left -/
theorem movie_watching_times :
  let camila_watched := movie_duration + maverick_daniella_diff + camila_maverick_diff
  let maverick_watched := movie_duration + maverick_daniella_diff
  let daniella_watched := movie_duration - daniella_remaining
  camila_watched ≥ movie_duration ∧ maverick_watched ≥ movie_duration ∧ daniella_watched < movie_duration :=
by sorry

end NUMINAMATH_CALUDE_movie_watching_times_l167_16721


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l167_16789

theorem intersection_point_of_lines (x y : ℚ) :
  (8 * x - 5 * y = 40) ∧ (6 * x + 2 * y = 14) ↔ x = 75 / 23 ∧ y = -64 / 23 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l167_16789


namespace NUMINAMATH_CALUDE_work_completion_proof_l167_16764

/-- The original number of men working on a task -/
def original_men : ℕ := 20

/-- The number of days it takes the original group to complete the work -/
def original_days : ℕ := 10

/-- The number of men removed from the original group -/
def removed_men : ℕ := 10

/-- The number of additional days it takes to complete the work with fewer men -/
def additional_days : ℕ := 10

theorem work_completion_proof :
  (original_men * original_days = (original_men - removed_men) * (original_days + additional_days)) →
  original_men = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_proof_l167_16764


namespace NUMINAMATH_CALUDE_point_coordinates_l167_16780

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between a point and the x-axis -/
def distanceToXAxis (p : Point) : ℝ := |p.y|

/-- The distance between a point and the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- Predicate to check if a point is in the fourth quadrant -/
def inFourthQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y < 0

theorem point_coordinates (p : Point) 
  (h1 : inFourthQuadrant p) 
  (h2 : distanceToXAxis p = 2) 
  (h3 : distanceToYAxis p = 5) : 
  p = Point.mk 5 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l167_16780


namespace NUMINAMATH_CALUDE_paul_chickens_sold_to_neighbor_l167_16725

/-- The number of chickens Paul sold to his neighbor -/
def chickens_sold_to_neighbor (initial_chickens : ℕ) (sold_to_customer : ℕ) (left_for_market : ℕ) : ℕ :=
  initial_chickens - sold_to_customer - left_for_market

theorem paul_chickens_sold_to_neighbor :
  chickens_sold_to_neighbor 80 25 43 = 12 := by
  sorry

end NUMINAMATH_CALUDE_paul_chickens_sold_to_neighbor_l167_16725


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l167_16743

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + x)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 63 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l167_16743
