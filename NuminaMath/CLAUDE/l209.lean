import Mathlib

namespace NUMINAMATH_CALUDE_lamplighter_monkey_distance_l209_20923

/-- Calculates the total distance traveled by a Lamplighter monkey under specific conditions. -/
theorem lamplighter_monkey_distance (initial_swing_speed initial_run_speed : ℝ)
  (wind_resistance_factor branch_weight_factor : ℝ)
  (run_time swing_time : ℝ) :
  initial_swing_speed = 10 →
  initial_run_speed = 15 →
  wind_resistance_factor = 0.9 →
  branch_weight_factor = 1.05 →
  run_time = 5 →
  swing_time = 10 →
  let adjusted_swing_speed := initial_swing_speed * wind_resistance_factor
  let adjusted_run_speed := initial_run_speed * branch_weight_factor
  let total_distance := adjusted_run_speed * run_time + adjusted_swing_speed * swing_time
  total_distance = 168.75 := by
sorry


end NUMINAMATH_CALUDE_lamplighter_monkey_distance_l209_20923


namespace NUMINAMATH_CALUDE_obtuse_minus_right_is_acute_straight_minus_acute_is_obtuse_l209_20990

/-- Definition of an obtuse angle -/
def is_obtuse_angle (α : ℝ) : Prop := 90 < α ∧ α < 180

/-- Definition of a right angle -/
def is_right_angle (α : ℝ) : Prop := α = 90

/-- Definition of an acute angle -/
def is_acute_angle (α : ℝ) : Prop := 0 < α ∧ α < 90

/-- Definition of a straight angle -/
def is_straight_angle (α : ℝ) : Prop := α = 180

/-- Theorem: When an obtuse angle is cut by a right angle, the remaining angle is acute -/
theorem obtuse_minus_right_is_acute (α β : ℝ) 
  (h1 : is_obtuse_angle α) (h2 : is_right_angle β) : 
  is_acute_angle (α - β) := by sorry

/-- Theorem: When a straight angle is cut by an acute angle, the remaining angle is obtuse -/
theorem straight_minus_acute_is_obtuse (α β : ℝ) 
  (h1 : is_straight_angle α) (h2 : is_acute_angle β) : 
  is_obtuse_angle (α - β) := by sorry

end NUMINAMATH_CALUDE_obtuse_minus_right_is_acute_straight_minus_acute_is_obtuse_l209_20990


namespace NUMINAMATH_CALUDE_constant_value_l209_20928

/-- The function f(x) = x + 4 -/
def f (x : ℝ) : ℝ := x + 4

/-- The theorem stating that if the equation has a solution x = 0.4, then c = 1 -/
theorem constant_value (c : ℝ) :
  (∃ x : ℝ, x = 0.4 ∧ (3 * f (x - 2)) / f 0 + 4 = f (2 * x + c)) → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l209_20928


namespace NUMINAMATH_CALUDE_book_pages_calculation_l209_20902

theorem book_pages_calculation (pages_per_day : ℕ) (days_read : ℕ) (fraction_read : ℚ) : 
  pages_per_day = 12 →
  days_read = 15 →
  fraction_read = 3/4 →
  (pages_per_day * days_read : ℚ) / fraction_read = 240 := by
sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l209_20902


namespace NUMINAMATH_CALUDE_tree_initial_height_l209_20960

/-- Given a tree with constant yearly growth for 6 years, prove its initial height. -/
theorem tree_initial_height (growth_rate : ℝ) (h1 : growth_rate = 0.4) : ∃ (initial_height : ℝ),
  initial_height + 6 * growth_rate = (initial_height + 4 * growth_rate) * (1 + 1/7) ∧
  initial_height = 4 := by
  sorry

end NUMINAMATH_CALUDE_tree_initial_height_l209_20960


namespace NUMINAMATH_CALUDE_square_quotient_theorem_l209_20926

theorem square_quotient_theorem (a b : ℕ+) (h : (a * b + 1) ∣ (a^2 + b^2)) :
  ∃ k : ℕ, (a^2 + b^2) / (a * b + 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_quotient_theorem_l209_20926


namespace NUMINAMATH_CALUDE_problem_1_l209_20961

theorem problem_1 (a : ℝ) : 
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) ↔ a ∈ Set.Iic (-1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_problem_1_l209_20961


namespace NUMINAMATH_CALUDE_limit_equals_third_derivative_at_one_l209_20936

-- Define a real-valued function f that is differentiable on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- State the theorem
theorem limit_equals_third_derivative_at_one :
  (∀ ε > 0, ∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ) \ {1},
    |((f (1 + (x - 1)) - f 1) / (3 * (x - 1))) - (1/3 * deriv f 1)| < ε) :=
sorry

end NUMINAMATH_CALUDE_limit_equals_third_derivative_at_one_l209_20936


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l209_20924

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 13*x + 4 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = 13 ∧ r₁ * r₂ = 4 ∧ r₁^2 + r₂^2 = 161 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l209_20924


namespace NUMINAMATH_CALUDE_infinitely_many_satisfying_functions_l209_20905

/-- A function that satisfies the given conditions -/
def satisfying_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = x^2) ∧ (Set.range f = Set.Icc 1 4)

/-- There exist infinitely many functions satisfying the given conditions -/
theorem infinitely_many_satisfying_functions :
  ∃ (S : Set (ℝ → ℝ)), Set.Infinite S ∧ ∀ f ∈ S, satisfying_function f :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_satisfying_functions_l209_20905


namespace NUMINAMATH_CALUDE_carla_lemonade_consumption_l209_20945

/-- The number of glasses of lemonade Carla can drink in a given time period. -/
def glasses_of_lemonade (time_minutes : ℕ) (rate_minutes : ℕ) : ℕ :=
  time_minutes / rate_minutes

/-- Proves that Carla can drink 11 glasses of lemonade in 3 hours and 40 minutes. -/
theorem carla_lemonade_consumption : 
  glasses_of_lemonade 220 20 = 11 := by
  sorry

#eval glasses_of_lemonade 220 20

end NUMINAMATH_CALUDE_carla_lemonade_consumption_l209_20945


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l209_20978

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ

/-- A diameter of a circle -/
structure Diameter where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Given a circle with center (1, 2) and one endpoint of a diameter at (4, 6),
    the other endpoint of the diameter is at (-2, -2) -/
theorem circle_diameter_endpoint (P : Circle) (d : Diameter) :
  P.center = (1, 2) →
  d.endpoint1 = (4, 6) →
  d.endpoint2 = (-2, -2) := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l209_20978


namespace NUMINAMATH_CALUDE_ticket_sales_total_l209_20929

/-- Calculates the total sales from ticket sales given the number of tickets sold and prices. -/
theorem ticket_sales_total (total_tickets : ℕ) (child_tickets : ℕ) (adult_price : ℕ) (child_price : ℕ)
  (h1 : total_tickets = 42)
  (h2 : child_tickets = 16)
  (h3 : adult_price = 5)
  (h4 : child_price = 3) :
  (total_tickets - child_tickets) * adult_price + child_tickets * child_price = 178 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_total_l209_20929


namespace NUMINAMATH_CALUDE_car_resale_gain_l209_20997

/-- Calculates the percentage gain when reselling a car -/
theorem car_resale_gain (original_price selling_price_2 : ℝ) (loss_percent : ℝ) : 
  original_price = 50561.80 →
  loss_percent = 11 →
  selling_price_2 = 54000 →
  (selling_price_2 - (original_price * (1 - loss_percent / 100))) / (original_price * (1 - loss_percent / 100)) * 100 = 20 :=
by sorry

end NUMINAMATH_CALUDE_car_resale_gain_l209_20997


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l209_20984

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (((x^2 + y^2 + z^2) * (4*x^2 + y^2 + z^2)).sqrt) / (x*y*z) ≥ (3/2 : ℝ) :=
sorry

theorem lower_bound_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (((x^2 + y^2 + z^2) * (4*x^2 + y^2 + z^2)).sqrt) / (x*y*z) = (3/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l209_20984


namespace NUMINAMATH_CALUDE_max_value_of_a_l209_20912

theorem max_value_of_a (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_six : a^2 + b^2 + c^2 = 6) : 
  ∀ x : ℝ, x ≤ 2 ∧ (∃ a₀ b₀ c₀ : ℝ, a₀ + b₀ + c₀ = 0 ∧ a₀^2 + b₀^2 + c₀^2 = 6 ∧ a₀ = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l209_20912


namespace NUMINAMATH_CALUDE_expected_value_of_game_l209_20994

def roll_value (n : ℕ) : ℝ :=
  if n % 2 = 0 then 3 * n else 0

def fair_8_sided_die : Finset ℕ := Finset.range 8

theorem expected_value_of_game : 
  (fair_8_sided_die.sum (λ i => (roll_value (i + 1)) / 8)) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_game_l209_20994


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_in_interval_l209_20921

theorem inequality_holds_iff_p_in_interval (p : ℝ) :
  (∀ x : ℝ, -9 < (3 * x^2 + p * x - 6) / (x^2 - x + 1) ∧ 
             (3 * x^2 + p * x - 6) / (x^2 - x + 1) < 6) ↔ 
  -3 < p ∧ p < 6 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_in_interval_l209_20921


namespace NUMINAMATH_CALUDE_largest_package_size_l209_20991

theorem largest_package_size (alex_markers becca_markers charlie_markers : ℕ) 
  (h_alex : alex_markers = 36)
  (h_becca : becca_markers = 45)
  (h_charlie : charlie_markers = 60) :
  Nat.gcd alex_markers (Nat.gcd becca_markers charlie_markers) = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l209_20991


namespace NUMINAMATH_CALUDE_total_spent_is_23_88_l209_20982

def green_grape_price : ℝ := 2.79
def red_grape_price : ℝ := 3.25
def regular_cherry_price : ℝ := 4.90
def organic_cherry_price : ℝ := 5.75

def green_grape_weight : ℝ := 2.5
def red_grape_weight : ℝ := 1.8
def regular_cherry_weight : ℝ := 1.2
def organic_cherry_weight : ℝ := 0.9

def total_spent : ℝ :=
  green_grape_price * green_grape_weight +
  red_grape_price * red_grape_weight +
  regular_cherry_price * regular_cherry_weight +
  organic_cherry_price * organic_cherry_weight

theorem total_spent_is_23_88 : total_spent = 23.88 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_23_88_l209_20982


namespace NUMINAMATH_CALUDE_cost_of_three_pencils_four_pens_l209_20943

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a pen -/
def pen_cost : ℝ := sorry

/-- The cost of 8 pencils and 3 pens is $5.60 -/
axiom first_equation : 8 * pencil_cost + 3 * pen_cost = 5.60

/-- The cost of 2 pencils and 5 pens is $4.25 -/
axiom second_equation : 2 * pencil_cost + 5 * pen_cost = 4.25

/-- The cost of 3 pencils and 4 pens is approximately $9.68 -/
theorem cost_of_three_pencils_four_pens :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |3 * pencil_cost + 4 * pen_cost - 9.68| < ε :=
sorry

end NUMINAMATH_CALUDE_cost_of_three_pencils_four_pens_l209_20943


namespace NUMINAMATH_CALUDE_carrot_broccoli_ratio_is_two_to_one_l209_20967

/-- Represents the sales data for a farmers' market --/
structure MarketSales where
  total : ℕ
  broccoli : ℕ
  cauliflower : ℕ
  spinach_offset : ℕ

/-- Calculates the ratio of carrot sales to broccoli sales --/
def carrot_broccoli_ratio (sales : MarketSales) : ℚ :=
  let carrot_sales := sales.total - sales.broccoli - sales.cauliflower - 
    (sales.spinach_offset + (sales.total - sales.broccoli - sales.cauliflower - sales.spinach_offset) / 2)
  carrot_sales / sales.broccoli

/-- Theorem stating that the ratio of carrot sales to broccoli sales is 2:1 --/
theorem carrot_broccoli_ratio_is_two_to_one (sales : MarketSales) 
  (h1 : sales.total = 380)
  (h2 : sales.broccoli = 57)
  (h3 : sales.cauliflower = 136)
  (h4 : sales.spinach_offset = 16) :
  carrot_broccoli_ratio sales = 2 := by
  sorry

end NUMINAMATH_CALUDE_carrot_broccoli_ratio_is_two_to_one_l209_20967


namespace NUMINAMATH_CALUDE_square_sum_inequality_l209_20904

theorem square_sum_inequality (x y : ℝ) : x^2 + y^2 + 1 ≥ x + y + x*y := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l209_20904


namespace NUMINAMATH_CALUDE_square_land_side_length_l209_20995

theorem square_land_side_length 
  (area : ℝ) 
  (h_area : area = Real.sqrt 1024) 
  (h_square : ∀ s, s * s = area → s = Real.sqrt area) : 
  ∃ side : ℝ, side * side = area ∧ side = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l209_20995


namespace NUMINAMATH_CALUDE_twenty_seven_in_base_two_l209_20979

theorem twenty_seven_in_base_two : 
  27 = 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 := by
  sorry

end NUMINAMATH_CALUDE_twenty_seven_in_base_two_l209_20979


namespace NUMINAMATH_CALUDE_total_marbles_l209_20937

theorem total_marbles (blue red orange : ℕ) : 
  blue = red + orange → -- Half of the marbles are blue
  red = 6 →             -- There are 6 red marbles
  orange = 6 →          -- There are 6 orange marbles
  blue + red + orange = 24 := by
sorry

end NUMINAMATH_CALUDE_total_marbles_l209_20937


namespace NUMINAMATH_CALUDE_complex_sum_equals_polar_form_l209_20941

theorem complex_sum_equals_polar_form : 
  5 * Complex.exp (Complex.I * (3 * Real.pi / 7)) + 
  15 * Complex.exp (Complex.I * (23 * Real.pi / 14)) = 
  20 * Real.sqrt ((3 + Real.cos (13 * Real.pi / 14)) / 4) * 
  Complex.exp (Complex.I * (29 * Real.pi / 28)) := by sorry

end NUMINAMATH_CALUDE_complex_sum_equals_polar_form_l209_20941


namespace NUMINAMATH_CALUDE_min_value_expression_l209_20931

/-- Given positive real numbers m and n, vectors a and b, where a is parallel to b,
    prove that the minimum value of 1/m + 2/n is 3 + 2√2 -/
theorem min_value_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (a b : Fin 2 → ℝ) 
  (ha : a = ![m, 1]) 
  (hb : b = ![1-n, 1]) 
  (h_parallel : ∃ (k : ℝ), a = k • b) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 2/y ≥ 1/m + 2/n) → 
  1/m + 2/n = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l209_20931


namespace NUMINAMATH_CALUDE_monkey_reach_top_l209_20934

/-- The time it takes for a monkey to climb a greased pole -/
def monkey_climb_time (pole_height : ℕ) (ascend : ℕ) (slip : ℕ) : ℕ :=
  let effective_progress := ascend - slip
  let full_cycles := (pole_height - ascend) / effective_progress
  2 * full_cycles + 1

/-- Theorem stating that the monkey will reach the top of the pole in 17 minutes -/
theorem monkey_reach_top : monkey_climb_time 10 2 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_monkey_reach_top_l209_20934


namespace NUMINAMATH_CALUDE_inequality_solution_difference_l209_20957

theorem inequality_solution_difference : ∃ (M m : ℝ),
  (∀ x, 4 * x * (x - 5) ≤ 375 → x ≤ M ∧ m ≤ x) ∧
  (4 * M * (M - 5) ≤ 375) ∧
  (4 * m * (m - 5) ≤ 375) ∧
  (M - m = 20) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_difference_l209_20957


namespace NUMINAMATH_CALUDE_factorial_ratio_l209_20955

theorem factorial_ratio : Nat.factorial 30 / Nat.factorial 28 = 870 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l209_20955


namespace NUMINAMATH_CALUDE_matrix_identity_sum_l209_20925

open Matrix

variable {n : Type*} [DecidableEq n] [Fintype n]

theorem matrix_identity_sum (B : Matrix n n ℝ) :
  Invertible B →
  (B - 3 • 1) * (B - 5 • 1) = 0 →
  B + 15 • B⁻¹ = 8 • 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_identity_sum_l209_20925


namespace NUMINAMATH_CALUDE_a_3_equals_negative_10_l209_20987

def a (n : ℕ) : ℤ := (-1)^n * (n^2 + 1)

theorem a_3_equals_negative_10 : a 3 = -10 := by
  sorry

end NUMINAMATH_CALUDE_a_3_equals_negative_10_l209_20987


namespace NUMINAMATH_CALUDE_problem_solution_l209_20922

theorem problem_solution (a b : ℚ) (h1 : 7 * a + 3 * b = 0) (h2 : a = 2 * b - 3) :
  5 * b - 4 * a = 141 / 17 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l209_20922


namespace NUMINAMATH_CALUDE_power_of_two_triples_l209_20977

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def satisfies_condition (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_power_of_two (a * b - c) ∧
  is_power_of_two (b * c - a) ∧
  is_power_of_two (c * a - b)

theorem power_of_two_triples :
  ∀ a b c : ℕ, satisfies_condition a b c ↔
    ((a, b, c) = (2, 2, 2) ∨
     (a, b, c) = (2, 2, 3) ∨
     (a, b, c) = (3, 5, 7) ∨
     (a, b, c) = (2, 6, 11)) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_triples_l209_20977


namespace NUMINAMATH_CALUDE_value_of_expression_l209_20971

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l209_20971


namespace NUMINAMATH_CALUDE_no_common_points_l209_20969

theorem no_common_points : ¬∃ (x y : ℝ), 
  (x^2 + 4*y^2 = 4) ∧ (4*x^2 + y^2 = 4) ∧ (x^2 + y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_common_points_l209_20969


namespace NUMINAMATH_CALUDE_coin_toss_probability_l209_20952

def toss_outcomes (n : ℕ) : ℕ := 2^n

def favorable_outcomes : ℕ := 5

theorem coin_toss_probability :
  let mina_tosses : ℕ := 2
  let liam_tosses : ℕ := 3
  let total_outcomes : ℕ := toss_outcomes mina_tosses * toss_outcomes liam_tosses
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 32 := by sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l209_20952


namespace NUMINAMATH_CALUDE_factorization_equality_l209_20920

theorem factorization_equality (x y : ℝ) : 6 * x^3 * y^2 + 3 * x^2 * y^2 = 3 * x^2 * y^2 * (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l209_20920


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l209_20938

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  a / (Real.sin A) = c / (Real.sin C) ∧
  a = 3 ∧ b = Real.sqrt 3 ∧ A = π / 3 →
  B = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l209_20938


namespace NUMINAMATH_CALUDE_man_upstream_speed_l209_20935

/-- Given a man's speed in still water and downstream, calculate his upstream speed -/
theorem man_upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 20)
  (h2 : speed_downstream = 25) :
  speed_still - (speed_downstream - speed_still) = 15 := by
  sorry


end NUMINAMATH_CALUDE_man_upstream_speed_l209_20935


namespace NUMINAMATH_CALUDE_f_extrema_l209_20948

def f (x : ℝ) : ℝ := -x^3 + 3*x - 1

theorem f_extrema :
  (∃ x : ℝ, f x = 1 ∧ ∀ y : ℝ, f y ≤ f x) ∧
  (∃ x : ℝ, f x = -3 ∧ ∀ y : ℝ, f y ≥ f x) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l209_20948


namespace NUMINAMATH_CALUDE_curve_properties_l209_20983

-- Define the curve
def curve (x y : ℝ) : Prop := x^3 + x*y + y^3 = 3

-- Define symmetry with respect to y = -x
def symmetric_about_neg_x (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ f (-y) (-x)

-- Define a point being on the curve
def point_on_curve (x y : ℝ) : Prop := curve x y

-- Define the concept of a curve approaching a line
def approaches_line (f : ℝ → ℝ → Prop) (m b : ℝ) : Prop :=
  ∀ ε > 0, ∃ M > 0, ∀ x y, f x y → (|x| > M ∨ |y| > M) → |y - (m*x + b)| < ε

theorem curve_properties :
  symmetric_about_neg_x curve ∧
  point_on_curve (Real.rpow 3 (1/3 : ℝ)) 0 ∧
  point_on_curve 1 1 ∧
  point_on_curve 0 (Real.rpow 3 (1/3 : ℝ)) ∧
  approaches_line curve (-1) 0 :=
sorry

end NUMINAMATH_CALUDE_curve_properties_l209_20983


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l209_20911

theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ x y : ℝ, x < y → x < y + 1) ∧ 
  (∃ x y : ℝ, x < y + 1 ∧ ¬(x < y)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l209_20911


namespace NUMINAMATH_CALUDE_berry_ratio_l209_20915

theorem berry_ratio (total berries stacy steve skylar : ℕ) : 
  total = 1100 →
  stacy = 800 →
  stacy = 4 * steve →
  total = stacy + steve + skylar →
  steve = 2 * skylar := by
sorry

end NUMINAMATH_CALUDE_berry_ratio_l209_20915


namespace NUMINAMATH_CALUDE_right_triangle_existence_l209_20944

theorem right_triangle_existence (c h : ℝ) (hc : c > 0) (hh : h > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 ∧ (a * b) / c = h ↔ h ≤ c / 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l209_20944


namespace NUMINAMATH_CALUDE_quadratic_range_on_unit_interval_l209_20986

/-- The range of a quadratic function on [0, 1] -/
theorem quadratic_range_on_unit_interval (a b c : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∀ x, x ∈ Set.Icc 0 1 → f x ∈ Set.Icc (min (f 0) (min (f 1) (f (-b/(2*a))))) (max (f 0) (f 1))) ∧
  ((-2*a ≤ b ∧ b ≤ 0 ∧ a + b + c ≥ c) →
    Set.Icc (-b^2/(4*a) + c) (a + b + c) = Set.Icc (min (f 0) (min (f 1) (f (-b/(2*a))))) (max (f 0) (f 1))) ∧
  ((b < -2*a ∨ b > 0) →
    Set.Icc c (a + b + c) = Set.Icc (min (f 0) (min (f 1) (f (-b/(2*a))))) (max (f 0) (f 1))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_range_on_unit_interval_l209_20986


namespace NUMINAMATH_CALUDE_point_on_hyperbola_l209_20940

/-- A point (x, y) lies on the hyperbola y = -4/x if and only if xy = -4 -/
def lies_on_hyperbola (x y : ℝ) : Prop := x * y = -4

/-- The point (-2, 2) lies on the hyperbola y = -4/x -/
theorem point_on_hyperbola : lies_on_hyperbola (-2) 2 := by sorry

end NUMINAMATH_CALUDE_point_on_hyperbola_l209_20940


namespace NUMINAMATH_CALUDE_square_hexagon_side_ratio_l209_20998

theorem square_hexagon_side_ratio :
  ∀ (s_s s_h : ℝ),
  s_s > 0 → s_h > 0 →
  s_s^2 = (3 * s_h^2 * Real.sqrt 3) / 2 →
  s_s / s_h = Real.sqrt ((3 * Real.sqrt 3) / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_square_hexagon_side_ratio_l209_20998


namespace NUMINAMATH_CALUDE_simplify_expression_l209_20903

theorem simplify_expression (x y : ℝ) : (5 - 4*x) - (2 + 7*x - y) = 3 - 11*x + y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l209_20903


namespace NUMINAMATH_CALUDE_unique_a_l209_20913

theorem unique_a : ∃! a : ℝ, (∃ m : ℤ, a + 2/3 = m) ∧ (∃ n : ℤ, 1/a - 3/4 = n) := by
  sorry

end NUMINAMATH_CALUDE_unique_a_l209_20913


namespace NUMINAMATH_CALUDE_polar_equations_and_intersection_ratio_l209_20953

-- Define the line l in Cartesian coordinates
def line_l (x y : ℝ) : Prop := x = 4

-- Define the curve C in Cartesian coordinates
def curve_C (x y φ : ℝ) : Prop := x = 1 + Real.sqrt 2 * Real.cos φ ∧ y = 1 + Real.sqrt 2 * Real.sin φ

-- Define the transformation from Cartesian to polar coordinates
def to_polar (x y ρ θ : ℝ) : Prop := x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- State the theorem
theorem polar_equations_and_intersection_ratio :
  ∀ (x y ρ θ φ α : ℝ),
  (line_l x y → ρ * Real.cos θ = 4) ∧
  (curve_C x y φ → ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) ∧
  (0 < α ∧ α < Real.pi / 4 →
    ∃ (ρ_A ρ_B : ℝ),
      ρ_A = 2 * (Real.cos α + Real.sin α) ∧
      ρ_B = 4 / Real.cos α ∧
      1 / 2 < ρ_A / ρ_B ∧ ρ_A / ρ_B ≤ (Real.sqrt 2 + 1) / 4) := by
  sorry

end NUMINAMATH_CALUDE_polar_equations_and_intersection_ratio_l209_20953


namespace NUMINAMATH_CALUDE_garage_spokes_count_l209_20976

/-- The number of bicycles in the garage -/
def num_bicycles : ℕ := 4

/-- The number of wheels per bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of spokes per wheel -/
def spokes_per_wheel : ℕ := 10

/-- The total number of spokes in the garage -/
def total_spokes : ℕ := num_bicycles * wheels_per_bicycle * spokes_per_wheel

theorem garage_spokes_count : total_spokes = 80 := by
  sorry

end NUMINAMATH_CALUDE_garage_spokes_count_l209_20976


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l209_20908

theorem log_sum_equals_two :
  2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l209_20908


namespace NUMINAMATH_CALUDE_group_four_frequency_and_relative_frequency_l209_20933

/-- Given a sample with capacity 50 and frequencies for groups 1, 2, 3, and 5,
    prove the frequency and relative frequency of group 4 -/
theorem group_four_frequency_and_relative_frequency 
  (total_capacity : ℕ) 
  (freq_1 freq_2 freq_3 freq_5 : ℕ) 
  (h1 : total_capacity = 50)
  (h2 : freq_1 = 8)
  (h3 : freq_2 = 11)
  (h4 : freq_3 = 10)
  (h5 : freq_5 = 9) :
  ∃ (freq_4 : ℕ) (rel_freq_4 : ℚ),
    freq_4 = total_capacity - (freq_1 + freq_2 + freq_3 + freq_5) ∧
    rel_freq_4 = freq_4 / total_capacity ∧
    freq_4 = 12 ∧
    rel_freq_4 = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_group_four_frequency_and_relative_frequency_l209_20933


namespace NUMINAMATH_CALUDE_smallest_twice_cube_thrice_square_l209_20950

theorem smallest_twice_cube_thrice_square :
  (∃ k : ℕ, k > 0 ∧
    (∃ n : ℕ, k = 2 * n^3) ∧
    (∃ m : ℕ, k = 3 * m^2) ∧
    (∀ j : ℕ, j > 0 →
      (∃ p : ℕ, j = 2 * p^3) →
      (∃ q : ℕ, j = 3 * q^2) →
      j ≥ k)) →
  (∃ k : ℕ, k = 432 ∧
    (∃ n : ℕ, k = 2 * n^3) ∧
    (∃ m : ℕ, k = 3 * m^2) ∧
    (∀ j : ℕ, j > 0 →
      (∃ p : ℕ, j = 2 * p^3) →
      (∃ q : ℕ, j = 3 * q^2) →
      j ≥ k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_twice_cube_thrice_square_l209_20950


namespace NUMINAMATH_CALUDE_congruence_problem_l209_20974

theorem congruence_problem (x : ℤ) 
  (h1 : (4 + x) % 16 = 9)
  (h2 : (6 + x) % 36 = 16)
  (h3 : (8 + x) % 64 = 36) :
  x % 48 = 37 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l209_20974


namespace NUMINAMATH_CALUDE_min_value_theorem_l209_20946

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^4 + y^4 + z^4 = 1) :
  (x^3 / (1 - x^8)) + (y^3 / (1 - y^8)) + (z^3 / (1 - z^8)) ≥ 9 * (3^(1/4)) / 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l209_20946


namespace NUMINAMATH_CALUDE_special_angle_calculation_l209_20975

theorem special_angle_calculation :
  let tan30 := Real.sqrt 3 / 3
  let cos60 := 1 / 2
  let sin45 := Real.sqrt 2 / 2
  Real.sqrt 3 * tan30 + 2 * cos60 - Real.sqrt 2 * sin45 = 1 := by sorry

end NUMINAMATH_CALUDE_special_angle_calculation_l209_20975


namespace NUMINAMATH_CALUDE_line_equation_proof_l209_20989

/-- Given a line defined by the equation (3, -4) · ((x, y) - (-2, 8)) = 0,
    prove that its slope-intercept form y = mx + b has m = 3/4 and b = 9.5 -/
theorem line_equation_proof :
  let line_eq := fun (x y : ℝ) => (3 * (x + 2) + (-4) * (y - 8) = 0)
  ∃ (m b : ℝ), m = 3/4 ∧ b = 9.5 ∧ ∀ x y, line_eq x y ↔ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l209_20989


namespace NUMINAMATH_CALUDE_leo_current_weight_l209_20909

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 98

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 170 - leo_weight

/-- Theorem stating that Leo's current weight is 98 pounds -/
theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = 170) →
  leo_weight = 98 := by
sorry

end NUMINAMATH_CALUDE_leo_current_weight_l209_20909


namespace NUMINAMATH_CALUDE_remainder_of_108_times_112_div_11_l209_20927

theorem remainder_of_108_times_112_div_11 : (108 * 112) % 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_108_times_112_div_11_l209_20927


namespace NUMINAMATH_CALUDE_circle_radius_is_three_l209_20968

/-- Given a circle where the product of three inches and its circumference
    equals twice its area, prove that its radius is 3 inches. -/
theorem circle_radius_is_three (r : ℝ) (h : 3 * (2 * π * r) = 2 * (π * r^2)) : r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_three_l209_20968


namespace NUMINAMATH_CALUDE_max_notebooks_with_10_dollars_l209_20954

/-- Represents the number of notebooks in a pack -/
inductive PackSize
  | single
  | pack4
  | pack7

/-- Returns the cost of a pack given its size -/
def packCost (size : PackSize) : ℕ :=
  match size with
  | PackSize.single => 1
  | PackSize.pack4 => 3
  | PackSize.pack7 => 5

/-- Returns the number of notebooks in a pack given its size -/
def packNotebooks (size : PackSize) : ℕ :=
  match size with
  | PackSize.single => 1
  | PackSize.pack4 => 4
  | PackSize.pack7 => 7

/-- Represents a purchase of notebook packs -/
structure Purchase where
  single : ℕ
  pack4 : ℕ
  pack7 : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.single * packCost PackSize.single +
  p.pack4 * packCost PackSize.pack4 +
  p.pack7 * packCost PackSize.pack7

/-- Calculates the total number of notebooks in a purchase -/
def totalNotebooks (p : Purchase) : ℕ :=
  p.single * packNotebooks PackSize.single +
  p.pack4 * packNotebooks PackSize.pack4 +
  p.pack7 * packNotebooks PackSize.pack7

/-- The maximum number of notebooks that can be purchased with $10 is 14 -/
theorem max_notebooks_with_10_dollars :
  (∀ p : Purchase, totalCost p ≤ 10 → totalNotebooks p ≤ 14) ∧
  (∃ p : Purchase, totalCost p ≤ 10 ∧ totalNotebooks p = 14) :=
sorry

end NUMINAMATH_CALUDE_max_notebooks_with_10_dollars_l209_20954


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l209_20939

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (5 : ℚ) / 8 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (5 : ℚ) / 8 → q ≤ q') →
  q - p = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l209_20939


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l209_20916

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![-2, 4]

-- State the theorem
theorem vector_difference_magnitude : ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l209_20916


namespace NUMINAMATH_CALUDE_quadratic_points_ordering_l209_20965

theorem quadratic_points_ordering (m : ℝ) (y₁ y₂ y₃ : ℝ) :
  ((-1)^2 + 2*(-1) + m = y₁) →
  (3^2 + 2*3 + m = y₂) →
  ((1/2)^2 + 2*(1/2) + m = y₃) →
  (y₂ > y₃ ∧ y₃ > y₁) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_points_ordering_l209_20965


namespace NUMINAMATH_CALUDE_sum_of_coordinates_for_symmetric_points_l209_20956

-- Define the points P and Q
def P (x : ℝ) : ℝ × ℝ := (x, -3)
def Q (y : ℝ) : ℝ × ℝ := (4, y)

-- Define the property of being symmetric with respect to the origin
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

-- Theorem statement
theorem sum_of_coordinates_for_symmetric_points (x y : ℝ) :
  symmetric_about_origin (P x) (Q y) → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_for_symmetric_points_l209_20956


namespace NUMINAMATH_CALUDE_series_sum_l209_20972

/-- The sum of the series ∑_{n=0}^{∞} (-1)^n / (3n + 1) is equal to (1/3) * (ln(2) + π/√3) -/
theorem series_sum : 
  ∑' (n : ℕ), ((-1)^n : ℝ) / (3*n + 1) = (1/3) * (Real.log 2 + Real.pi / Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l209_20972


namespace NUMINAMATH_CALUDE_sector_arc_length_l209_20949

theorem sector_arc_length (central_angle : Real) (radius : Real) 
  (h1 : central_angle = 1/5)
  (h2 : radius = 5) : 
  central_angle * radius = 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l209_20949


namespace NUMINAMATH_CALUDE_solve_linear_equation_l209_20958

theorem solve_linear_equation (x y : ℝ) :
  2 * x + y = 5 → x = (5 - y) / 2 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l209_20958


namespace NUMINAMATH_CALUDE_derivative_zero_neither_necessary_nor_sufficient_l209_20930

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define what it means for a function to have an extremum at a point
def has_extremum (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (a - ε) (a + ε), f x ≤ f a ∨ f x ≥ f a

-- Define the statement to be proven
theorem derivative_zero_neither_necessary_nor_sufficient :
  ¬(∀ f : ℝ → ℝ, ∀ a : ℝ, (has_extremum f a ↔ HasDerivAt f 0 a)) :=
sorry

end NUMINAMATH_CALUDE_derivative_zero_neither_necessary_nor_sufficient_l209_20930


namespace NUMINAMATH_CALUDE_baby_guppies_count_is_36_l209_20947

/-- The number of baby guppies Amber saw several days after buying 7 guppies,
    given that she later saw 9 more baby guppies and now has 52 guppies in total. -/
def baby_guppies_count : ℕ := by sorry

/-- The initial number of guppies Amber bought. -/
def initial_guppies : ℕ := 7

/-- The number of additional baby guppies Amber saw two days after the first group. -/
def additional_baby_guppies : ℕ := 9

/-- The total number of guppies Amber has now. -/
def total_guppies : ℕ := 52

theorem baby_guppies_count_is_36 :
  baby_guppies_count = 36 ∧
  initial_guppies + baby_guppies_count + additional_baby_guppies = total_guppies := by
  sorry

end NUMINAMATH_CALUDE_baby_guppies_count_is_36_l209_20947


namespace NUMINAMATH_CALUDE_singing_competition_winner_l209_20959

/-- Represents the contestants in the singing competition -/
inductive Contestant : Type
  | one | two | three | four | five | six

/-- Represents the students making guesses -/
inductive Student : Type
  | A | B | C | D

def guess (s : Student) (c : Contestant) : Prop :=
  match s with
  | Student.A => c = Contestant.four ∨ c = Contestant.five
  | Student.B => c ≠ Contestant.three
  | Student.C => c = Contestant.one ∨ c = Contestant.two ∨ c = Contestant.six
  | Student.D => c ≠ Contestant.four ∧ c ≠ Contestant.five ∧ c ≠ Contestant.six

theorem singing_competition_winner :
  ∃! (winner : Contestant),
    (∃! (correct_guesser : Student), guess correct_guesser winner) ∧
    (∀ (c : Contestant), c ≠ winner → ¬ guess Student.A c ∧ ¬ guess Student.B c ∧ ¬ guess Student.C c ∧ ¬ guess Student.D c) ∧
    winner = Contestant.three :=
by sorry

end NUMINAMATH_CALUDE_singing_competition_winner_l209_20959


namespace NUMINAMATH_CALUDE_monday_visitors_l209_20985

/-- Represents the number of visitors to a library in a week -/
structure LibraryVisitors where
  monday : ℕ
  tuesday : ℕ
  remainingDays : ℕ

/-- Theorem: Given the conditions of the library visitors problem, prove that there were 50 visitors on Monday -/
theorem monday_visitors (v : LibraryVisitors) : v.monday = 50 :=
  by
  have h1 : v.tuesday = 2 * v.monday := by sorry
  have h2 : v.remainingDays = 5 * 20 := by sorry
  have h3 : v.monday + v.tuesday + v.remainingDays = 250 := by sorry
  sorry


end NUMINAMATH_CALUDE_monday_visitors_l209_20985


namespace NUMINAMATH_CALUDE_bisection_method_for_f_l209_20980

/-- The function f(x) = x^5 + 8x^3 - 1 -/
def f (x : ℝ) : ℝ := x^5 + 8*x^3 - 1

/-- Theorem stating the properties of the bisection method for f(x) -/
theorem bisection_method_for_f :
  f 0 < 0 →
  f 0.5 > 0 →
  ∃ (a b : ℝ), a = 0 ∧ b = 0.5 ∧
    (∃ (x : ℝ), a < x ∧ x < b ∧ f x = 0) ∧
    ((a + b) / 2 = 0.25) :=
sorry

end NUMINAMATH_CALUDE_bisection_method_for_f_l209_20980


namespace NUMINAMATH_CALUDE_first_number_in_ratio_l209_20942

/-- Given two positive integers with a ratio of 3:4 and an LCM of 84, prove that the first number is 21 -/
theorem first_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 3 / 4 → 
  Nat.lcm a b = 84 → 
  a = 21 := by
  sorry

end NUMINAMATH_CALUDE_first_number_in_ratio_l209_20942


namespace NUMINAMATH_CALUDE_subset_implies_a_squared_equals_two_l209_20901

/-- Given sets A and B, where A = {0, 2, 3} and B = {2, a² + 1}, and B is a subset of A,
    prove that a² = 2, where a is a real number. -/
theorem subset_implies_a_squared_equals_two (a : ℝ) : 
  let A : Set ℝ := {0, 2, 3}
  let B : Set ℝ := {2, a^2 + 1}
  B ⊆ A → a^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_squared_equals_two_l209_20901


namespace NUMINAMATH_CALUDE_total_pokemon_cards_l209_20973

def melanie_cards : ℝ := 7.5
def benny_cards : ℝ := 9
def sandy_cards : ℝ := 5.2
def jessica_cards : ℝ := 12.8

theorem total_pokemon_cards :
  (melanie_cards * 12 + benny_cards * 12 + sandy_cards * 12 + jessica_cards * 12) = 414 := by
  sorry

end NUMINAMATH_CALUDE_total_pokemon_cards_l209_20973


namespace NUMINAMATH_CALUDE_calculate_expression_l209_20970

-- Define the variables and their relationships
def x : ℝ := 70 * (1 + 0.11)
def y : ℝ := x * (1 + 0.15)
def z : ℝ := y * (1 - 0.20)

-- State the theorem
theorem calculate_expression : 3 * z - 2 * x + y = 148.407 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l209_20970


namespace NUMINAMATH_CALUDE_fox_initial_coins_l209_20918

def bridge_crossings (initial_coins : ℕ) : ℕ := 
  ((initial_coins * 2 - 40) * 2 - 40) * 2 - 40

theorem fox_initial_coins : 
  ∃ (x : ℕ), bridge_crossings x = 0 ∧ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_fox_initial_coins_l209_20918


namespace NUMINAMATH_CALUDE_fermat_like_equation_implies_power_l209_20919

theorem fermat_like_equation_implies_power (n p x y k : ℕ) : 
  Odd n → 
  n > 1 → 
  Nat.Prime p → 
  Odd p → 
  x^n + y^n = p^k → 
  ∃ t : ℕ, n = p^t := by
sorry

end NUMINAMATH_CALUDE_fermat_like_equation_implies_power_l209_20919


namespace NUMINAMATH_CALUDE_zoo_treats_problem_l209_20932

/-- The percentage of pieces of bread Jane brings compared to treats -/
def jane_bread_percentage (jane_treats : ℕ) (jane_bread : ℕ) (wanda_treats : ℕ) (wanda_bread : ℕ) : ℚ :=
  (jane_bread : ℚ) / (jane_treats : ℚ) * 100

/-- The problem statement -/
theorem zoo_treats_problem (jane_treats : ℕ) (jane_bread : ℕ) (wanda_treats : ℕ) (wanda_bread : ℕ) :
  wanda_treats = jane_treats / 2 →
  wanda_bread = 3 * wanda_treats →
  wanda_bread = 90 →
  jane_treats + jane_bread + wanda_treats + wanda_bread = 225 →
  jane_bread_percentage jane_treats jane_bread wanda_treats wanda_bread = 75 := by
  sorry

end NUMINAMATH_CALUDE_zoo_treats_problem_l209_20932


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l209_20988

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (-2, 1)
  let b : ℝ × ℝ := (m, 3)
  are_parallel a b → m = -6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l209_20988


namespace NUMINAMATH_CALUDE_circle_translation_l209_20981

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the translation
def translation : ℝ × ℝ := (-5, -3)

-- Define the translated circle
def translated_circle (x y : ℝ) : Prop := (x+5)^2 + (y+3)^2 = 16

-- Theorem statement
theorem circle_translation :
  ∀ (x y : ℝ), original_circle (x + 5) (y + 3) ↔ translated_circle x y :=
by sorry

end NUMINAMATH_CALUDE_circle_translation_l209_20981


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l209_20964

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l209_20964


namespace NUMINAMATH_CALUDE_soda_price_calculation_l209_20951

theorem soda_price_calculation (remy_morning : ℕ) (nick_diff : ℕ) (evening_sales : ℚ) (evening_increase : ℚ) :
  remy_morning = 55 →
  nick_diff = 6 →
  evening_sales = 55 →
  evening_increase = 3 →
  ∃ (price : ℚ), price = 1/2 ∧ 
    (remy_morning + (remy_morning - nick_diff)) * price + evening_increase = evening_sales :=
by
  sorry

end NUMINAMATH_CALUDE_soda_price_calculation_l209_20951


namespace NUMINAMATH_CALUDE_C_equiv_C_param_l209_20914

/-- A semicircular curve C in the polar coordinate system -/
def C : Set (ℝ × ℝ) := {(p, θ) | p = 2 * Real.cos θ ∧ 0 ≤ θ ∧ θ ≤ Real.pi / 2}

/-- The parametric representation of curve C -/
def C_param : Set (ℝ × ℝ) := {(x, y) | ∃ α, 0 ≤ α ∧ α ≤ Real.pi ∧ x = 1 + Real.cos α ∧ y = Real.sin α}

/-- Theorem stating that the parametric representation is equivalent to the polar representation -/
theorem C_equiv_C_param : C = C_param := by sorry

end NUMINAMATH_CALUDE_C_equiv_C_param_l209_20914


namespace NUMINAMATH_CALUDE_probability_a2_selected_l209_20900

-- Define the sets of students
def english_students : Finset (Fin 2) := Finset.univ
def japanese_students : Finset (Fin 3) := Finset.univ

-- Define the total number of possible outcomes
def total_outcomes : ℕ := (english_students.card * japanese_students.card)

-- Define the number of outcomes where A₂ is selected
def a2_outcomes : ℕ := japanese_students.card

-- Theorem statement
theorem probability_a2_selected :
  (a2_outcomes : ℚ) / total_outcomes = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_probability_a2_selected_l209_20900


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l209_20910

theorem polynomial_division_remainder (x : ℝ) : 
  x^1004 % ((x^2 + 1) * (x - 1)) = x^2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l209_20910


namespace NUMINAMATH_CALUDE_smallest_y_value_l209_20996

theorem smallest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = 1) : 
  ∀ (z : ℤ), z ≥ -10 ∨ ¬∃ (w : ℤ), w * z + 3 * w + 2 * z = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_y_value_l209_20996


namespace NUMINAMATH_CALUDE_percentage_commutativity_l209_20917

theorem percentage_commutativity (x : ℝ) (h : (30 / 100) * (40 / 100) * x = 60) :
  (40 / 100) * (30 / 100) * x = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutativity_l209_20917


namespace NUMINAMATH_CALUDE_distance_between_points_l209_20962

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 9)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l209_20962


namespace NUMINAMATH_CALUDE_max_speed_theorem_l209_20993

/-- Represents a pair of observed values (speed, defective products) -/
structure Observation where
  speed : ℝ
  defects : ℝ

/-- The regression line equation -/
def regression_line (slope : ℝ) (intercept : ℝ) (x : ℝ) : ℝ :=
  slope * x + intercept

/-- Theorem: Maximum speed given observations and max defects -/
theorem max_speed_theorem (observations : List Observation) 
  (max_defects : ℝ) (slope : ℝ) (intercept : ℝ) :
  observations = [
    ⟨8, 5⟩, ⟨12, 8⟩, ⟨14, 9⟩, ⟨16, 11⟩
  ] →
  max_defects = 10 →
  slope = 51 / 70 →
  intercept = -6 / 7 →
  (∀ x, regression_line slope intercept x ≤ max_defects → x ≤ 14) ∧
  regression_line slope intercept 14 ≤ max_defects :=
by sorry

end NUMINAMATH_CALUDE_max_speed_theorem_l209_20993


namespace NUMINAMATH_CALUDE_rice_cake_slices_l209_20966

theorem rice_cake_slices (num_cakes : ℕ) (cake_length : ℝ) (overlap : ℝ) (num_slices : ℕ) :
  num_cakes = 5 →
  cake_length = 2.7 →
  overlap = 0.3 →
  num_slices = 6 →
  (num_cakes * cake_length - (num_cakes - 1) * overlap) / num_slices = 2.05 := by
  sorry

end NUMINAMATH_CALUDE_rice_cake_slices_l209_20966


namespace NUMINAMATH_CALUDE_fraction_equality_l209_20963

theorem fraction_equality : (1 / 5 - 1 / 6) / (1 / 4 - 1 / 5) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l209_20963


namespace NUMINAMATH_CALUDE_total_cookies_l209_20992

/-- The number of cookies each kid has -/
structure Cookies where
  chris : ℕ
  kenny : ℕ
  glenn : ℕ
  dan : ℕ
  anne : ℕ

/-- The conditions of the cookie distribution -/
def cookie_conditions (c : Cookies) : Prop :=
  c.chris * 3 = c.kenny ∧
  c.glenn = c.chris * 4 ∧
  c.glenn = 24 ∧
  c.dan = 2 * (c.chris + c.kenny) ∧
  c.anne * 2 = c.kenny

/-- The theorem stating the total number of cookies -/
theorem total_cookies (c : Cookies) (h : cookie_conditions c) : 
  c.chris + c.kenny + c.glenn + c.dan + c.anne = 105 := by
  sorry

#check total_cookies

end NUMINAMATH_CALUDE_total_cookies_l209_20992


namespace NUMINAMATH_CALUDE_installment_payment_installment_payment_proof_l209_20999

theorem installment_payment (cash_price : ℕ) (down_payment : ℕ) (first_four : ℕ) 
  (last_four : ℕ) (installment_difference : ℕ) : ℕ :=
  let total_installment := cash_price + installment_difference
  let first_four_total := 4 * first_four
  let last_four_total := 4 * last_four
  let middle_four_total := total_installment - down_payment - first_four_total - last_four_total
  let middle_four_monthly := middle_four_total / 4
  middle_four_monthly

#check @installment_payment

theorem installment_payment_proof 
  (h1 : installment_payment 450 100 40 30 70 = 35) : True := by
  sorry

end NUMINAMATH_CALUDE_installment_payment_installment_payment_proof_l209_20999


namespace NUMINAMATH_CALUDE_solve_for_y_l209_20907

theorem solve_for_y (x y : ℝ) (h1 : x^2 = y + 7) (h2 : x = 6) : y = 29 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l209_20907


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l209_20906

theorem trigonometric_inequality (x : ℝ) :
  9.286 * (Real.sin x)^3 * Real.sin (π/2 - 3*x) + (Real.cos x)^3 * Real.cos (π/2 - 3*x) > 3*Real.sqrt 3/8 →
  ∃ n : ℤ, π/12 + n*π/2 < x ∧ x < π/6 + n*π/2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l209_20906
