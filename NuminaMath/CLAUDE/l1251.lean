import Mathlib

namespace amount_paid_l1251_125134

def lemonade_cups : ℕ := 2
def lemonade_price : ℚ := 2
def sandwich_count : ℕ := 2
def sandwich_price : ℚ := 2.5
def change_received : ℚ := 11

def total_cost : ℚ := lemonade_cups * lemonade_price + sandwich_count * sandwich_price

theorem amount_paid (paid : ℚ) : paid = 20 ↔ paid = total_cost + change_received := by
  sorry

end amount_paid_l1251_125134


namespace carly_nail_trimming_l1251_125180

/-- Calculates the total number of nails trimmed by a pet groomer --/
def total_nails_trimmed (total_dogs : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) : ℕ :=
  let four_legged_dogs := total_dogs - three_legged_dogs
  let nails_per_four_legged_dog := 4 * nails_per_paw
  let nails_per_three_legged_dog := 3 * nails_per_paw
  four_legged_dogs * nails_per_four_legged_dog + three_legged_dogs * nails_per_three_legged_dog

theorem carly_nail_trimming :
  total_nails_trimmed 11 3 4 = 164 := by
  sorry

end carly_nail_trimming_l1251_125180


namespace triangle_area_from_square_areas_l1251_125188

theorem triangle_area_from_square_areas (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) :
  (1/2) * a * b = 24 :=
by sorry

end triangle_area_from_square_areas_l1251_125188


namespace total_cds_l1251_125168

def dawn_cds : ℕ := 10
def kristine_cds : ℕ := dawn_cds + 7

theorem total_cds : dawn_cds + kristine_cds = 27 := by
  sorry

end total_cds_l1251_125168


namespace journey_distance_l1251_125141

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance / (2 * speed1) + distance / (2 * speed2) = total_time ∧ 
    distance = 224 := by
  sorry

end journey_distance_l1251_125141


namespace cos_squared_minus_sin_squared_15_deg_l1251_125153

theorem cos_squared_minus_sin_squared_15_deg : 
  Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end cos_squared_minus_sin_squared_15_deg_l1251_125153


namespace min_P_over_Q_l1251_125147

theorem min_P_over_Q (x P Q : ℝ) (hx : x > 0) (hP : P > 0) (hQ : Q > 0)
  (hP_def : x^2 + 1/x^2 = P) (hQ_def : x^3 - 1/x^3 = Q) :
  ∀ y : ℝ, y > 0 → y^2 + 1/y^2 = P → y^3 - 1/y^3 = Q → P / Q ≥ 1 / Real.sqrt 3 :=
by sorry

end min_P_over_Q_l1251_125147


namespace cistern_depth_is_correct_l1251_125191

/-- Represents a rectangular cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  total_wet_area : ℝ

/-- Calculates the total wet surface area of a cistern --/
def wet_surface_area (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem stating that for a cistern with given dimensions and wet surface area, the depth is 1.25 m --/
theorem cistern_depth_is_correct (c : Cistern) 
    (h1 : c.length = 6)
    (h2 : c.width = 4)
    (h3 : c.total_wet_area = 49)
    (h4 : wet_surface_area c = c.total_wet_area) :
    c.depth = 1.25 := by
  sorry

end cistern_depth_is_correct_l1251_125191


namespace fish_tank_ratio_l1251_125164

/-- Given 3 fish tanks with a total of 100 fish, where one tank has 20 fish
    and the other two have an equal number of fish, prove that the ratio of fish
    in each of the other two tanks to the first tank is 2:1 -/
theorem fish_tank_ratio :
  ∀ (fish_in_other_tanks : ℕ),
  3 * 20 + 2 * fish_in_other_tanks = 100 →
  fish_in_other_tanks = 2 * 20 :=
by sorry

end fish_tank_ratio_l1251_125164


namespace students_per_group_l1251_125127

/-- Given a total of 64 students, with 36 not picked, and divided into 4 groups,
    prove that there are 7 students in each group. -/
theorem students_per_group :
  ∀ (total : ℕ) (not_picked : ℕ) (groups : ℕ),
    total = 64 →
    not_picked = 36 →
    groups = 4 →
    (total - not_picked) / groups = 7 :=
by
  sorry

end students_per_group_l1251_125127


namespace negation_of_proposition_l1251_125136

theorem negation_of_proposition (p : (x : ℝ) → x > 1 → x^3 + 1 > 8*x) :
  (¬ ∀ (x : ℝ), x > 1 → x^3 + 1 > 8*x) ↔ 
  (∃ (x : ℝ), x > 1 ∧ x^3 + 1 ≤ 8*x) :=
by sorry

end negation_of_proposition_l1251_125136


namespace arithmetic_mean_reciprocals_first_five_primes_l1251_125102

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem arithmetic_mean_reciprocals_first_five_primes :
  let reciprocals := first_five_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length) = 2927 / 11550 := by
  sorry

end arithmetic_mean_reciprocals_first_five_primes_l1251_125102


namespace equation_solutions_l1251_125186

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 25 = 0 ↔ x = 5 ∨ x = -5) ∧
  (∀ x : ℝ, (2*x - 1)^3 = -8 ↔ x = -1/2) ∧
  (∀ x : ℝ, 4*(x + 1)^2 = 8 ↔ x = -1 - Real.sqrt 2 ∨ x = -1 + Real.sqrt 2) :=
by sorry

end equation_solutions_l1251_125186


namespace rent_increase_problem_l1251_125146

theorem rent_increase_problem (initial_average : ℝ) (new_average : ℝ) (num_friends : ℕ) 
  (increase_percentage : ℝ) (h1 : initial_average = 800) (h2 : new_average = 850) 
  (h3 : num_friends = 4) (h4 : increase_percentage = 0.16) : 
  ∃ (original_rent : ℝ), 
    (num_friends * new_average - num_friends * initial_average) / increase_percentage = original_rent ∧ 
    original_rent = 1250 := by
sorry

end rent_increase_problem_l1251_125146


namespace time_after_317h_58m_30s_l1251_125173

def hours_to_12hour_clock (h : ℕ) : ℕ :=
  h % 12

def add_time (start_hour start_minute start_second : ℕ) 
             (add_hours add_minutes add_seconds : ℕ) : ℕ × ℕ × ℕ :=
  let total_seconds := start_second + add_seconds
  let total_minutes := start_minute + add_minutes + total_seconds / 60
  let total_hours := start_hour + add_hours + total_minutes / 60
  (hours_to_12hour_clock total_hours, total_minutes % 60, total_seconds % 60)

theorem time_after_317h_58m_30s : 
  let (A, B, C) := add_time 3 0 0 317 58 30
  A + B + C = 96 := by sorry

end time_after_317h_58m_30s_l1251_125173


namespace max_d_is_one_l1251_125174

def a (n : ℕ) : ℕ := 105 + n^2 + 3*n

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_is_one :
  ∀ n : ℕ, n ≥ 1 → d n ≤ 1 ∧ ∃ m : ℕ, m ≥ 1 ∧ d m = 1 :=
by sorry

end max_d_is_one_l1251_125174


namespace sum_of_x_satisfying_condition_l1251_125105

def X : Finset ℕ := {0, 1, 2}

def g : ℕ → ℕ
| 0 => 0
| 1 => 2
| 2 => 1
| _ => 0

def f : ℕ → ℕ
| 0 => 2
| 1 => 1
| 2 => 0
| _ => 0

theorem sum_of_x_satisfying_condition : 
  (X.filter (fun x => f (g x) > g (f x))).sum id = 2 := by
  sorry

end sum_of_x_satisfying_condition_l1251_125105


namespace odd_expressions_l1251_125148

theorem odd_expressions (m n p : ℕ) 
  (hm : m % 2 = 1) 
  (hn : n % 2 = 1) 
  (hp : p % 2 = 0) 
  (hm_pos : 0 < m) 
  (hn_pos : 0 < n) 
  (hp_pos : 0 < p) : 
  ((2 * m * n + 5)^2) % 2 = 1 ∧ (5 * m * n + p) % 2 = 1 := by
  sorry

end odd_expressions_l1251_125148


namespace geometric_sequence_ratio_l1251_125139

/-- Given a geometric sequence {a_n} with sum of first n terms S_n,
    if a_1 + a_3 = 5/4 and a_2 + a_4 = 5/2, then S_6 / S_3 = 9 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 + a 3 = 5/4)
  (h2 : a 2 + a 4 = 5/2)
  (h_geom : ∀ n : ℕ, a (n+1) / a n = a 2 / a 1)
  (h_sum : ∀ n : ℕ, S n = a 1 * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)) :
  S 6 / S 3 = 9 := by
sorry

end geometric_sequence_ratio_l1251_125139


namespace ellipse_chord_slope_l1251_125128

/-- Given an ellipse defined by 4x^2 + 9y^2 = 144 and a point P(3, 2) inside it,
    the slope of the line containing the chord with P as its midpoint is -2/3 -/
theorem ellipse_chord_slope (x y : ℝ) :
  4 * x^2 + 9 * y^2 = 144 →  -- Ellipse equation
  ∃ (x1 y1 x2 y2 : ℝ),       -- Endpoints of the chord
    4 * x1^2 + 9 * y1^2 = 144 ∧   -- First endpoint on ellipse
    4 * x2^2 + 9 * y2^2 = 144 ∧   -- Second endpoint on ellipse
    (x1 + x2) / 2 = 3 ∧           -- P is midpoint (x-coordinate)
    (y1 + y2) / 2 = 2 →           -- P is midpoint (y-coordinate)
    (y2 - y1) / (x2 - x1) = -2/3  -- Slope of the chord
:= by sorry

end ellipse_chord_slope_l1251_125128


namespace number_2018_in_equation_31_l1251_125194

def first_term (n : ℕ) : ℕ := 2 * n^2

theorem number_2018_in_equation_31 :
  ∃ k : ℕ, k ≥ first_term 31 ∧ k ≤ first_term 32 ∧ k = 2018 :=
by sorry

end number_2018_in_equation_31_l1251_125194


namespace chinese_chess_sets_l1251_125170

theorem chinese_chess_sets (go_cost : ℕ) (chinese_chess_cost : ℕ) (total_sets : ℕ) (total_cost : ℕ) :
  go_cost = 24 →
  chinese_chess_cost = 18 →
  total_sets = 14 →
  total_cost = 300 →
  ∃ (go_sets chinese_chess_sets : ℕ),
    go_sets + chinese_chess_sets = total_sets ∧
    go_cost * go_sets + chinese_chess_cost * chinese_chess_sets = total_cost ∧
    chinese_chess_sets = 6 := by
  sorry

end chinese_chess_sets_l1251_125170


namespace value_of_x_l1251_125117

theorem value_of_x : ∃ X : ℚ, (3/4 : ℚ) * (1/8 : ℚ) * X = (1/2 : ℚ) * (1/4 : ℚ) * 120 ∧ X = 160 := by
  sorry

end value_of_x_l1251_125117


namespace sibling_ages_sum_l1251_125198

theorem sibling_ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 180 → a + b + c = 26 := by
  sorry

end sibling_ages_sum_l1251_125198


namespace quadratic_roots_l1251_125114

theorem quadratic_roots (x y : ℝ) : 
  x + y = 8 → 
  |x - y| = 10 → 
  x^2 - 8*x - 9 = 0 ∧ y^2 - 8*y - 9 = 0 :=
by sorry

end quadratic_roots_l1251_125114


namespace cassidy_grades_below_B_l1251_125183

/-- The number of grades below B that Cassidy received -/
def grades_below_B : ℕ := sorry

/-- The base grounding period in days -/
def base_grounding : ℕ := 14

/-- The additional grounding days for each grade below B -/
def extra_days_per_grade : ℕ := 3

/-- The total grounding period in days -/
def total_grounding : ℕ := 26

theorem cassidy_grades_below_B :
  grades_below_B * extra_days_per_grade + base_grounding = total_grounding ∧
  grades_below_B = 4 := by sorry

end cassidy_grades_below_B_l1251_125183


namespace range_of_a_l1251_125132

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + a * x + 1 < 0) → a ∈ Set.Iio 0 ∪ Set.Ioi 4 :=
sorry

end range_of_a_l1251_125132


namespace polynomial_factorization_l1251_125187

theorem polynomial_factorization (a b c : ℝ) :
  2*a*(b - c)^3 + 3*b*(c - a)^3 + 2*c*(a - b)^3 = (a - b)*(b - c)*(c - a)*(5*b - c) := by
  sorry

end polynomial_factorization_l1251_125187


namespace complex_square_root_minus_100_plus_44i_l1251_125108

theorem complex_square_root_minus_100_plus_44i :
  {z : ℂ | z^2 = -100 + 44*I} = {2 + 11*I, -2 - 11*I} := by sorry

end complex_square_root_minus_100_plus_44i_l1251_125108


namespace largest_integer_less_than_100_with_remainder_5_mod_8_l1251_125123

theorem largest_integer_less_than_100_with_remainder_5_mod_8 :
  ∀ n : ℕ, n < 100 → n % 8 = 5 → n ≤ 93 :=
by
  sorry

end largest_integer_less_than_100_with_remainder_5_mod_8_l1251_125123


namespace star_example_l1251_125166

-- Define the * operation
def star (a b c d : ℚ) : ℚ := a * c * (d / (b + 1))

-- Theorem statement
theorem star_example : star 5 11 9 4 = 15 := by
  sorry

end star_example_l1251_125166


namespace battery_price_is_56_l1251_125144

/-- The price of a battery given the total cost of four tires and one battery, and the cost of each tire. -/
def battery_price (total_cost : ℕ) (tire_price : ℕ) : ℕ :=
  total_cost - 4 * tire_price

/-- Theorem stating that the battery price is $56 given the conditions. -/
theorem battery_price_is_56 :
  battery_price 224 42 = 56 := by
  sorry

end battery_price_is_56_l1251_125144


namespace r_earnings_l1251_125129

def daily_earnings (p q r : ℝ) : Prop :=
  9 * (p + q + r) = 1980 ∧
  5 * (p + r) = 600 ∧
  7 * (q + r) = 910

theorem r_earnings (p q r : ℝ) (h : daily_earnings p q r) : r = 30 := by
  sorry

end r_earnings_l1251_125129


namespace hyperbola_parameter_l1251_125109

/-- Given a parabola y^2 = 16x and a hyperbola (x^2/a^2) - (y^2/b^2) = 1 where:
    1. The right focus of the hyperbola coincides with the focus of the parabola (4, 0)
    2. The left directrix of the hyperbola is x = -3
    Then a^2 = 12 -/
theorem hyperbola_parameter (a b : ℝ) : 
  (∃ (x y : ℝ), y^2 = 16*x) → -- Parabola exists
  (∃ (x y : ℝ), (x^2/a^2) - (y^2/b^2) = 1) → -- Hyperbola exists
  (4 : ℝ) = a^2/(2*a) → -- Right focus of hyperbola is (4, 0)
  (-3 : ℝ) = -a^2/(2*a) → -- Left directrix of hyperbola is x = -3
  a^2 = 12 := by sorry

end hyperbola_parameter_l1251_125109


namespace sqrt_four_equals_two_l1251_125179

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end sqrt_four_equals_two_l1251_125179


namespace solution_set_implies_a_b_values_solution_on_interval_implies_a_range_three_integer_solutions_implies_a_range_l1251_125158

-- Define the function f
def f (x a b : ℝ) : ℝ := x^2 + (3-a)*x + 2 + 2*a + b

-- Theorem 1
theorem solution_set_implies_a_b_values (a b : ℝ) :
  (∀ x, f x a b > 0 ↔ x < -4 ∨ x > 2) →
  a = 1 ∧ b = -12 := by sorry

-- Theorem 2
theorem solution_on_interval_implies_a_range (a b : ℝ) :
  (∃ x ∈ Set.Icc 1 3, f x a b ≤ b) →
  a ≤ -6 ∨ a ≥ 20 := by sorry

-- Theorem 3
theorem three_integer_solutions_implies_a_range (a b : ℝ) :
  (∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x ∈ s, f x a b < 12 + b) →
  (3 ≤ a ∧ a < 4) ∨ (10 < a ∧ a ≤ 11) := by sorry

end solution_set_implies_a_b_values_solution_on_interval_implies_a_range_three_integer_solutions_implies_a_range_l1251_125158


namespace cyclist_speed_ratio_l1251_125172

theorem cyclist_speed_ratio :
  ∀ (v₁ v₂ : ℝ), v₁ > v₂ → v₁ > 0 → v₂ > 0 →
  (v₁ + v₂ = 25) →
  (v₁ - v₂ = 10 / 3) →
  (v₁ / v₂ = 17 / 13) := by
sorry

end cyclist_speed_ratio_l1251_125172


namespace perspective_drawing_preserves_parallel_equal_l1251_125126

/-- A plane figure -/
structure PlaneFigure where
  -- Add necessary fields

/-- A perspective drawing of a plane figure -/
structure PerspectiveDrawing where
  -- Add necessary fields

/-- A line segment in a plane figure or perspective drawing -/
structure LineSegment where
  -- Add necessary fields

/-- Predicate to check if two line segments are parallel -/
def are_parallel (s1 s2 : LineSegment) : Prop := sorry

/-- Predicate to check if two line segments are equal in length -/
def are_equal (s1 s2 : LineSegment) : Prop := sorry

/-- Function to get the corresponding line segments in a perspective drawing -/
def perspective_line_segments (pf : PlaneFigure) (pd : PerspectiveDrawing) (s1 s2 : LineSegment) : 
  (LineSegment × LineSegment) := sorry

theorem perspective_drawing_preserves_parallel_equal 
  (pf : PlaneFigure) (pd : PerspectiveDrawing) (s1 s2 : LineSegment) :
  are_parallel s1 s2 → are_equal s1 s2 → 
  let (p1, p2) := perspective_line_segments pf pd s1 s2
  are_parallel p1 p2 ∧ are_equal p1 p2 :=
by sorry

end perspective_drawing_preserves_parallel_equal_l1251_125126


namespace work_completion_time_l1251_125119

/-- The number of days it takes A to complete the work alone -/
def a_days : ℕ := 30

/-- The total payment for the work in Rupees -/
def total_payment : ℕ := 1000

/-- B's share of the payment in Rupees -/
def b_share : ℕ := 600

/-- The number of days it takes B to complete the work alone -/
def b_days : ℕ := 20

theorem work_completion_time :
  a_days = 30 ∧ 
  total_payment = 1000 ∧ 
  b_share = 600 →
  b_days = 20 :=
by sorry

end work_completion_time_l1251_125119


namespace inequality_solution_set_l1251_125156

theorem inequality_solution_set (x : ℝ) : 
  (1 / (x^2 - 4) + 4 / (2*x^2 + 7*x + 6) ≤ 1 / (2*x + 3) + 4 / (2*x^3 + 3*x^2 - 8*x - 12)) ↔ 
  (x ∈ Set.Ioo (-2 : ℝ) (-3/2) ∪ Set.Ico 1 2 ∪ Set.Ici 5) :=
by sorry

end inequality_solution_set_l1251_125156


namespace school_teachers_count_l1251_125165

/-- The number of departments in the school -/
def num_departments : ℕ := 7

/-- The number of teachers in each department -/
def teachers_per_department : ℕ := 20

/-- The total number of teachers in the school -/
def total_teachers : ℕ := num_departments * teachers_per_department

theorem school_teachers_count : total_teachers = 140 := by
  sorry

end school_teachers_count_l1251_125165


namespace coffee_shop_sales_l1251_125193

theorem coffee_shop_sales (teas lattes extra : ℕ) : 
  teas = 6 → lattes = 32 → 4 * teas + extra = lattes → extra = 8 := by sorry

end coffee_shop_sales_l1251_125193


namespace lcm_of_20_45_28_l1251_125113

theorem lcm_of_20_45_28 : Nat.lcm (Nat.lcm 20 45) 28 = 1260 := by
  sorry

end lcm_of_20_45_28_l1251_125113


namespace slope_angle_of_sqrt_three_line_l1251_125101

theorem slope_angle_of_sqrt_three_line :
  let line : ℝ → ℝ := λ x ↦ Real.sqrt 3 * x
  let slope : ℝ := Real.sqrt 3
  let angle : ℝ := 60 * Real.pi / 180
  (∀ x, line x = slope * x) ∧
  slope = Real.tan angle :=
by sorry

end slope_angle_of_sqrt_three_line_l1251_125101


namespace complex_square_sum_zero_l1251_125131

theorem complex_square_sum_zero (i : ℂ) (h : i^2 = -1) : 
  (1 + i)^2 + (1 - i)^2 = 0 := by
  sorry

end complex_square_sum_zero_l1251_125131


namespace complex_ratio_theorem_l1251_125143

theorem complex_ratio_theorem (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 1)
  (h₂ : Complex.abs z₂ = (5/2 : ℝ))
  (h₃ : Complex.abs (3 * z₁ - 2 * z₂) = 7) :
  z₁ / z₂ = -1/5 + Complex.I * Real.sqrt 3 / 5 ∨
  z₁ / z₂ = -1/5 - Complex.I * Real.sqrt 3 / 5 := by
sorry

end complex_ratio_theorem_l1251_125143


namespace max_y_value_l1251_125152

theorem max_y_value (x y : ℝ) (h : x^2 + y^2 = 20*x + 54*y) :
  y ≤ 27 + Real.sqrt 829 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 20*x₀ + 54*y₀ ∧ y₀ = 27 + Real.sqrt 829 := by
  sorry

end max_y_value_l1251_125152


namespace right_triangle_angle_bisector_square_area_l1251_125189

/-- Given a right triangle where the bisector of the right angle cuts the hypotenuse
    into segments of lengths a and b, the area of the square whose side is this bisector
    is equal to 2a²b² / (a² + b²). -/
theorem right_triangle_angle_bisector_square_area
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let bisector_length := Real.sqrt (2 * a^2 * b^2 / (a^2 + b^2))
  (bisector_length)^2 = 2 * a^2 * b^2 / (a^2 + b^2) := by
  sorry

end right_triangle_angle_bisector_square_area_l1251_125189


namespace x_value_proof_l1251_125175

theorem x_value_proof (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x - y^2 = 3) (h4 : x^2 + y^4 = 13) : 
  x = (3 + Real.sqrt 17) / 2 := by
  sorry

end x_value_proof_l1251_125175


namespace largest_number_value_l1251_125182

theorem largest_number_value (a b c : ℕ) : 
  a < b ∧ b < c ∧
  a + b + c = 80 ∧
  c = b + 9 ∧
  b = a + 4 ∧
  a * b = 525 →
  c = 34 := by
sorry

end largest_number_value_l1251_125182


namespace market_fruit_count_l1251_125137

/-- Calculates the total number of apples and oranges in a market -/
def total_fruits (num_apples : ℕ) (apple_orange_diff : ℕ) : ℕ :=
  num_apples + (num_apples - apple_orange_diff)

/-- Theorem: Given a market with 164 apples and 27 more apples than oranges,
    the total number of apples and oranges is 301 -/
theorem market_fruit_count : total_fruits 164 27 = 301 := by
  sorry

end market_fruit_count_l1251_125137


namespace letter_puzzle_solutions_l1251_125199

/-- A function that checks if a number is a single digit (1 to 9) -/
def isSingleDigit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- A function that checks if two numbers are distinct -/
def areDistinct (a b : ℕ) : Prop := a ≠ b

/-- A function that checks if a number is a two-digit number -/
def isTwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that constructs a two-digit number from two single digits -/
def twoDigitConstruct (b a : ℕ) : ℕ := 10 * b + a

/-- The main theorem stating the only solutions to A^B = BA -/
theorem letter_puzzle_solutions :
  ∀ A B : ℕ,
  isSingleDigit A →
  isSingleDigit B →
  areDistinct A B →
  isTwoDigitNumber (twoDigitConstruct B A) →
  twoDigitConstruct B A ≠ B * A →
  A^B = twoDigitConstruct B A →
  ((A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3)) :=
by sorry

end letter_puzzle_solutions_l1251_125199


namespace train_speed_clicks_l1251_125130

theorem train_speed_clicks (x : ℝ) : x > 0 →
  let t := (2400 : ℝ) / 5280
  t ≠ 0.25 ∧ t ≠ 1 ∧ t ≠ 2 ∧ t ≠ 3 := by
  sorry

end train_speed_clicks_l1251_125130


namespace marbles_per_bag_is_ten_l1251_125110

/-- The number of marbles in each bag of blue marbles --/
def marbles_per_bag : ℕ := sorry

/-- The initial number of green marbles --/
def initial_green : ℕ := 26

/-- The number of bags of blue marbles bought --/
def blue_bags : ℕ := 6

/-- The number of green marbles given away --/
def green_gift : ℕ := 6

/-- The number of blue marbles given away --/
def blue_gift : ℕ := 8

/-- The total number of marbles Janelle has after giving away the gift --/
def final_total : ℕ := 72

theorem marbles_per_bag_is_ten :
  (initial_green - green_gift) + (blue_bags * marbles_per_bag - blue_gift) = final_total →
  marbles_per_bag = 10 := by
  sorry

end marbles_per_bag_is_ten_l1251_125110


namespace refuel_cost_is_950_l1251_125185

/-- Calculates the total cost to refuel a fleet of planes --/
def total_refuel_cost (small_plane_count : ℕ) (large_plane_count : ℕ) (special_plane_count : ℕ)
  (small_tank_size : ℝ) (large_tank_size_factor : ℝ) (special_tank_size : ℝ)
  (regular_fuel_cost : ℝ) (special_fuel_cost : ℝ)
  (regular_service_fee : ℝ) (special_service_fee : ℝ) : ℝ :=
  let large_tank_size := small_tank_size * (1 + large_tank_size_factor)
  let regular_fuel_volume := small_plane_count * small_tank_size + large_plane_count * large_tank_size
  let regular_fuel_cost := regular_fuel_volume * regular_fuel_cost
  let special_fuel_cost := special_plane_count * special_tank_size * special_fuel_cost
  let regular_service_cost := (small_plane_count + large_plane_count) * regular_service_fee
  let special_service_cost := special_plane_count * special_service_fee
  regular_fuel_cost + special_fuel_cost + regular_service_cost + special_service_cost

/-- The total cost to refuel all five planes is $950 --/
theorem refuel_cost_is_950 :
  total_refuel_cost 2 2 1 60 0.5 200 0.5 1 100 200 = 950 := by
  sorry

end refuel_cost_is_950_l1251_125185


namespace joan_initial_oranges_l1251_125125

/-- Proves that Joan initially picked 37 oranges given the conditions -/
theorem joan_initial_oranges (initial : ℕ) (sold : ℕ) (remaining : ℕ)
  (h1 : sold = 10)
  (h2 : remaining = 27)
  (h3 : initial = remaining + sold) :
  initial = 37 := by
  sorry

end joan_initial_oranges_l1251_125125


namespace line_in_plane_equivalence_l1251_125138

-- Define a type for geometric objects
inductive GeometricObject
| Line : GeometricObject
| Plane : GeometricObject

-- Define a predicate for "is in"
def isIn (a b : GeometricObject) : Prop := sorry

-- Define the subset relation
def subset (a b : GeometricObject) : Prop := sorry

-- Theorem statement
theorem line_in_plane_equivalence (l : GeometricObject) (α : GeometricObject) :
  (l = GeometricObject.Line ∧ α = GeometricObject.Plane ∧ isIn l α) ↔ subset l α :=
sorry

end line_in_plane_equivalence_l1251_125138


namespace cube_volume_ratio_l1251_125112

theorem cube_volume_ratio (e : ℝ) (h : e > 0) :
  let small_cube_volume := e^3
  let large_cube_volume := (4*e)^3
  large_cube_volume / small_cube_volume = 64 := by
  sorry

end cube_volume_ratio_l1251_125112


namespace fixed_point_on_line_l1251_125120

/-- For any real number m, the line mx-y+1-3m=0 passes through the point (3, 1) -/
theorem fixed_point_on_line (m : ℝ) : m * 3 - 1 + 1 - 3 * m = 0 := by
  sorry

end fixed_point_on_line_l1251_125120


namespace f_composition_negative_three_l1251_125142

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 / (5 - x) else Real.log x / Real.log 4

theorem f_composition_negative_three : f (f (-3)) = -3/2 := by
  sorry

end f_composition_negative_three_l1251_125142


namespace distance_between_red_lights_l1251_125100

/-- Represents the color of a light -/
inductive LightColor
| Red
| Green

/-- Calculates the position of the nth red light in the sequence -/
def redLightPosition (n : ℕ) : ℕ :=
  (n - 1) / 3 * 7 + (n - 1) % 3 + 1

/-- The distance between lights in inches -/
def light_spacing : ℕ := 8

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The main theorem stating the distance between the 4th and 19th red lights -/
theorem distance_between_red_lights :
  (redLightPosition 19 - redLightPosition 4) * light_spacing / inches_per_foot = 
    (22671 : ℚ) / 1000 := by sorry

end distance_between_red_lights_l1251_125100


namespace product_repeating_decimal_three_and_eight_l1251_125115

/-- The product of 0.3̄ and 8 is equal to 8/3 -/
theorem product_repeating_decimal_three_and_eight :
  (∃ x : ℚ, x = 1/3 ∧ (∃ d : ℕ → ℕ, ∀ n, d n < 10 ∧ x = ∑' k, (d k : ℚ) / 10^(k+1)) ∧ x * 8 = 8/3) :=
by sorry

end product_repeating_decimal_three_and_eight_l1251_125115


namespace base_conversion_3275_to_octal_l1251_125197

theorem base_conversion_3275_to_octal :
  (6 * 8^3 + 3 * 8^2 + 2 * 8^1 + 3 * 8^0 : ℕ) = 3275 := by
  sorry

end base_conversion_3275_to_octal_l1251_125197


namespace monotonically_increasing_iff_a_geq_one_third_l1251_125162

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

theorem monotonically_increasing_iff_a_geq_one_third :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≥ 1/3 := by
  sorry

end monotonically_increasing_iff_a_geq_one_third_l1251_125162


namespace fraction_sum_equals_decimal_l1251_125159

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 10 + (3 : ℚ) / 100 + (3 : ℚ) / 1000 = (333 : ℚ) / 1000 := by
  sorry

end fraction_sum_equals_decimal_l1251_125159


namespace max_a_value_l1251_125149

noncomputable def f (x : ℝ) : ℝ := 1 - Real.sqrt (x + 1)

noncomputable def g (a x : ℝ) : ℝ := Real.log (a * x^2 - 3 * x + 1)

theorem max_a_value :
  (∀ x₁ : ℝ, x₁ ≥ 0 → ∃ x₂ : ℝ, f x₁ = g a x₂) →
  ∀ a' : ℝ, (∀ x₁ : ℝ, x₁ ≥ 0 → ∃ x₂ : ℝ, f x₁ = g a' x₂) →
  a' ≤ 9/4 :=
by sorry

end max_a_value_l1251_125149


namespace monotone_decreasing_implies_k_nonpositive_l1251_125161

/-- A function f(x) = kx² + (3k-2)x - 5 is monotonically decreasing on [1, +∞) -/
def is_monotone_decreasing (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y → f y < f x

/-- The main theorem stating that if f(x) = kx² + (3k-2)x - 5 is monotonically
    decreasing on [1, +∞), then k ∈ (-∞, 0] -/
theorem monotone_decreasing_implies_k_nonpositive (k : ℝ) :
  is_monotone_decreasing (fun x => k*x^2 + (3*k-2)*x - 5) k →
  k ∈ Set.Iic 0 :=
by sorry

end monotone_decreasing_implies_k_nonpositive_l1251_125161


namespace product_equality_l1251_125184

theorem product_equality (x y : ℝ) 
  (h : (x + Real.sqrt (1 + y^2)) * (y + Real.sqrt (1 + x^2)) = 1) :
  (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = 1 := by
  sorry

end product_equality_l1251_125184


namespace paula_meal_combinations_l1251_125145

/-- The number of meat options available --/
def meat_options : ℕ := 3

/-- The number of vegetable options available --/
def vegetable_options : ℕ := 5

/-- The number of dessert options available --/
def dessert_options : ℕ := 5

/-- The number of vegetables Paula must choose --/
def vegetables_to_choose : ℕ := 3

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The total number of meal combinations Paula can construct --/
def total_meals : ℕ :=
  meat_options * choose vegetable_options vegetables_to_choose * dessert_options

theorem paula_meal_combinations :
  total_meals = 150 :=
sorry

end paula_meal_combinations_l1251_125145


namespace chord_equation_l1251_125151

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by y² = 6x -/
def Parabola := {p : Point | p.y^2 = 6 * p.x}

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a point bisects a chord of the parabola -/
def bisectsChord (p : Point) (l : Line) : Prop :=
  p ∈ Parabola ∧ 
  ∃ a b : Point, a ≠ b ∧ 
    a ∈ Parabola ∧ b ∈ Parabola ∧
    a.onLine l ∧ b.onLine l ∧
    p.x = (a.x + b.x) / 2 ∧ p.y = (a.y + b.y) / 2

/-- The main theorem to be proved -/
theorem chord_equation : 
  let p := Point.mk 4 1
  let l := Line.mk 3 (-1) (-11)
  p ∈ Parabola ∧ bisectsChord p l := by sorry

end chord_equation_l1251_125151


namespace ratio_equivalence_l1251_125169

theorem ratio_equivalence (x : ℝ) : 
  (20 / 10 = 25 / x) → x = 12.5 := by
  sorry

end ratio_equivalence_l1251_125169


namespace point_coordinate_product_l1251_125103

theorem point_coordinate_product : 
  ∀ y₁ y₂ : ℝ,
  (((4 - (-2))^2 + (y₁ - 5)^2 = 13^2) ∧
   ((4 - (-2))^2 + (y₂ - 5)^2 = 13^2) ∧
   (∀ y : ℝ, ((4 - (-2))^2 + (y - 5)^2 = 13^2) → (y = y₁ ∨ y = y₂))) →
  y₁ * y₂ = -108 := by
sorry

end point_coordinate_product_l1251_125103


namespace farm_legs_l1251_125104

/-- The total number of animal legs on a farm with ducks, dogs, and spiders -/
def total_legs (num_ducks : ℕ) (num_dogs : ℕ) (num_spiders : ℕ) (num_three_legged_dogs : ℕ) : ℕ :=
  2 * num_ducks + 4 * (num_dogs - num_three_legged_dogs) + 3 * num_three_legged_dogs + 8 * num_spiders

/-- Theorem stating that the total number of animal legs on the farm is 55 -/
theorem farm_legs : total_legs 6 5 3 1 = 55 := by
  sorry

end farm_legs_l1251_125104


namespace cafeteria_pies_l1251_125171

theorem cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) :
  initial_apples = 250 →
  handed_out = 33 →
  apples_per_pie = 7 →
  (initial_apples - handed_out) / apples_per_pie = 31 :=
by
  sorry

end cafeteria_pies_l1251_125171


namespace sequence_with_special_sums_l1251_125190

theorem sequence_with_special_sums : ∃ (seq : Fin 20 → ℝ), 
  (∀ i : Fin 18, seq i + seq (i + 1) + seq (i + 2) > 0) ∧ 
  (Finset.sum Finset.univ seq < 0) := by
  sorry

end sequence_with_special_sums_l1251_125190


namespace gym_attendance_l1251_125155

theorem gym_attendance (initial_lifters : ℕ) : 
  initial_lifters + 5 - 2 = 19 → initial_lifters = 16 := by
  sorry

end gym_attendance_l1251_125155


namespace factor_expression_l1251_125154

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end factor_expression_l1251_125154


namespace equation_solution_l1251_125181

theorem equation_solution : ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
  sorry

end equation_solution_l1251_125181


namespace three_digit_number_proof_l1251_125121

/-- Given a three-digit number satisfying specific conditions, prove it equals 824 -/
theorem three_digit_number_proof (x y z : ℕ) : 
  z^2 = x * y →
  y = (x + z) / 6 →
  100 * x + 10 * y + z - 396 = 100 * z + 10 * y + x →
  100 * x + 10 * y + z = 824 := by
  sorry

end three_digit_number_proof_l1251_125121


namespace rotten_bananas_percentage_l1251_125111

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percentage = 15 / 100)
  (h4 : good_fruits_percentage = 878 / 1000)
  : (total_bananas - (total_oranges + total_bananas - 
     (good_fruits_percentage * (total_oranges + total_bananas)).floor - 
     (rotten_oranges_percentage * total_oranges).floor)) / total_bananas = 8 / 100 := by
  sorry

end rotten_bananas_percentage_l1251_125111


namespace pet_ownership_percentage_l1251_125118

/-- Represents the school with students and their pet ownership. -/
structure School where
  total_students : ℕ
  cat_owners : ℕ
  dog_owners : ℕ
  rabbit_owners : ℕ
  h_no_multiple_pets : cat_owners + dog_owners + rabbit_owners ≤ total_students

/-- Calculates the percentage of students owning at least one pet. -/
def percentage_pet_owners (s : School) : ℚ :=
  (s.cat_owners + s.dog_owners + s.rabbit_owners : ℚ) / s.total_students * 100

/-- Theorem stating that in the given school, 48% of students own at least one pet. -/
theorem pet_ownership_percentage (s : School) 
    (h_total : s.total_students = 500)
    (h_cats : s.cat_owners = 80)
    (h_dogs : s.dog_owners = 120)
    (h_rabbits : s.rabbit_owners = 40) : 
    percentage_pet_owners s = 48 := by sorry

end pet_ownership_percentage_l1251_125118


namespace arithmetic_sequence_transformation_l1251_125140

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given arithmetic sequence and its transformation -/
theorem arithmetic_sequence_transformation
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ)
  (h1 : ArithmeticSequence a)
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : ∀ n : ℕ, b n = 3 * a n + 4) :
  ArithmeticSequence b :=
sorry

end arithmetic_sequence_transformation_l1251_125140


namespace gcd_of_sequence_l1251_125122

theorem gcd_of_sequence (n : ℕ) : 
  ∃ d : ℕ, d > 0 ∧ 
  (∀ m : ℕ, d ∣ (7^(m+2) + 8^(2*m+1))) ∧
  (∀ k : ℕ, k > 0 → (∀ m : ℕ, k ∣ (7^(m+2) + 8^(2*m+1))) → k ≤ d) ∧
  d = 57 := by
sorry

end gcd_of_sequence_l1251_125122


namespace decimal_point_problem_l1251_125160

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
  sorry

end decimal_point_problem_l1251_125160


namespace quadratic_root_c_value_l1251_125163

theorem quadratic_root_c_value (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 13 * x + c = 0 ↔ x = (-13 + Real.sqrt 19) / 4 ∨ x = (-13 - Real.sqrt 19) / 4) →
  c = 18.75 := by
sorry

end quadratic_root_c_value_l1251_125163


namespace inverse_variation_sqrt_l1251_125133

/-- Given that y varies inversely as √x and y = 3 when x = 4, prove that y = √2 when x = 18 -/
theorem inverse_variation_sqrt (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, y x * Real.sqrt x = k) →  -- y varies inversely as √x
  y 4 = 3 →                      -- y = 3 when x = 4
  y 18 = Real.sqrt 2 :=          -- y = √2 when x = 18
by
  sorry

end inverse_variation_sqrt_l1251_125133


namespace store_distribution_problem_l1251_125167

/-- Represents the number of ways to distribute stores among cities -/
def distributionCount (totalStores : ℕ) (totalCities : ℕ) (maxStoresPerCity : ℕ) : ℕ :=
  sorry

/-- The specific problem conditions -/
theorem store_distribution_problem :
  distributionCount 4 5 2 = 45 := by sorry

end store_distribution_problem_l1251_125167


namespace nathan_daily_hours_l1251_125192

/-- Proves that Nathan played 3 hours per day given the conditions of the problem -/
theorem nathan_daily_hours : ∃ x : ℕ, 
  (14 * x + 5 * 7 = 77) ∧ x = 3 := by
  sorry

end nathan_daily_hours_l1251_125192


namespace batsman_average_l1251_125196

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℝ) :
  total_innings = 12 →
  last_innings_score = 92 →
  average_increase = 2 →
  (((total_innings - 1 : ℝ) * ((total_innings * (70 : ℝ) - last_innings_score : ℝ) / total_innings) + last_innings_score) / total_innings) - 
  ((total_innings * (70 : ℝ) - last_innings_score : ℝ) / total_innings) = average_increase →
  (((total_innings - 1 : ℝ) * ((total_innings * (70 : ℝ) - last_innings_score : ℝ) / total_innings) + last_innings_score) / total_innings) = 70 :=
by sorry

end batsman_average_l1251_125196


namespace ellipse_and_line_properties_l1251_125124

/-- Ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Line l with given properties -/
structure Line where
  m : ℝ

/-- Theorem stating the properties of the ellipse and line -/
theorem ellipse_and_line_properties
  (C : Ellipse)
  (e : ℝ)
  (min_dist : ℝ)
  (l : Line)
  (AB : ℝ)
  (h1 : e = Real.sqrt 3 / 3)
  (h2 : min_dist = Real.sqrt 3 - 1)
  (h3 : AB = 8 * Real.sqrt 3 / 5) :
  (∃ x y, x^2 / 3 + y^2 / 2 = 1) ∧
  (l.m = 1 ∨ l.m = -1) :=
sorry

end ellipse_and_line_properties_l1251_125124


namespace perpendicular_parallel_transitive_l1251_125116

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields/axioms for a line in 3D space
  -- This is a simplified representation

/-- Represents perpendicularity between two lines -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  -- Define what it means for two lines to be perpendicular
  sorry

/-- Represents parallelism between two lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  -- Define what it means for two lines to be parallel
  sorry

/-- Theorem: If line a is parallel to line b, and line l is perpendicular to a,
    then l is also perpendicular to b -/
theorem perpendicular_parallel_transitive (a b l : Line3D) :
  parallel a b → perpendicular l a → perpendicular l b := by
  sorry

end perpendicular_parallel_transitive_l1251_125116


namespace round_robin_tournament_matches_l1251_125150

theorem round_robin_tournament_matches (n : ℕ) (h : n = 6) : 
  (n * (n - 1)) / 2 = Nat.choose n 2 := by
  sorry

end round_robin_tournament_matches_l1251_125150


namespace janet_earnings_theorem_l1251_125195

/-- Calculates Janet's earnings per hour based on the number of posts checked and payment rates. -/
def janet_earnings_per_hour (text_posts image_posts video_posts : ℕ) 
  (text_rate image_rate video_rate : ℚ) : ℚ :=
  text_posts * text_rate + image_posts * image_rate + video_posts * video_rate

/-- Proves that Janet's earnings per hour equal $69.50 given the specified conditions. -/
theorem janet_earnings_theorem : 
  janet_earnings_per_hour 150 80 20 0.25 0.30 0.40 = 69.50 := by
  sorry

end janet_earnings_theorem_l1251_125195


namespace sum_of_fractions_l1251_125157

theorem sum_of_fractions : 
  (2 / 10 : ℚ) + (3 / 10 : ℚ) + (5 / 10 : ℚ) + (6 / 10 : ℚ) + (7 / 10 : ℚ) + 
  (9 / 10 : ℚ) + (14 / 10 : ℚ) + (15 / 10 : ℚ) + (20 / 10 : ℚ) + (41 / 10 : ℚ) = 
  (122 / 10 : ℚ) := by
  sorry

end sum_of_fractions_l1251_125157


namespace computer_games_count_l1251_125178

def polo_shirt_price : ℕ := 26
def necklace_price : ℕ := 83
def computer_game_price : ℕ := 90
def polo_shirt_count : ℕ := 3
def necklace_count : ℕ := 2
def rebate : ℕ := 12
def total_cost_after_rebate : ℕ := 322

theorem computer_games_count :
  ∃ (n : ℕ), 
    n * computer_game_price + 
    polo_shirt_count * polo_shirt_price + 
    necklace_count * necklace_price - 
    rebate = total_cost_after_rebate ∧ 
    n = 1 := by sorry

end computer_games_count_l1251_125178


namespace polynomial_division_theorem_l1251_125107

theorem polynomial_division_theorem (x : ℝ) :
  x^6 + 5*x^4 + 3 = (x - 2) * (x^5 + 2*x^4 + 9*x^3 + 18*x^2 + 36*x + 72) + 147 := by
  sorry

end polynomial_division_theorem_l1251_125107


namespace function_form_proof_l1251_125135

theorem function_form_proof (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x + Real.cos x ^ 2| ≤ 3/4)
  (h2 : ∀ x, |f x - Real.sin x ^ 2| ≤ 1/4) :
  ∀ x, f x = 3/4 - Real.cos x ^ 2 := by
  sorry

end function_form_proof_l1251_125135


namespace find_a_l1251_125176

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {1, 2, a^2 + 2*a - 3}

-- Define set A
def A (a : ℝ) : Set ℝ := {|a - 2|, 2}

-- Define the complement of A with respect to U
def complementA (a : ℝ) : Set ℝ := (U a) \ (A a)

-- Theorem statement
theorem find_a : ∃ a : ℝ, (U a = {1, 2, a^2 + 2*a - 3}) ∧ 
                          (A a = {|a - 2|, 2}) ∧ 
                          (complementA a = {0}) ∧ 
                          (a = 1) := by
  sorry

end find_a_l1251_125176


namespace inequality_solution_min_value_theorem_equality_condition_l1251_125106

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | 0 < x ∧ x < 2/3}

-- Theorem for the solution set of the inequality
theorem inequality_solution : 
  {x : ℝ | f x + |x + 1| < 2} = solution_set :=
sorry

-- Theorem for the minimum value of (4/m) + (1/n)
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∃ (a : ℝ), (∀ x : ℝ, g x ≥ a) ∧ m + n = a) →
  (4/m + 1/n ≥ 9/2) :=
sorry

-- Theorem for the equality condition
theorem equality_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∃ (a : ℝ), (∀ x : ℝ, g x ≥ a) ∧ m + n = a) →
  (4/m + 1/n = 9/2 ↔ m = 4/3 ∧ n = 2/3) :=
sorry

end inequality_solution_min_value_theorem_equality_condition_l1251_125106


namespace composition_value_l1251_125177

/-- Given two functions f and g, and a composition condition, prove that d equals 18 -/
theorem composition_value (c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = 5*x + c)
  (hg : ∀ x, g x = c*x + 3)
  (hcomp : ∀ x, f (g x) = 15*x + d) :
  d = 18 := by
  sorry

end composition_value_l1251_125177
