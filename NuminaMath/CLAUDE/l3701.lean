import Mathlib

namespace NUMINAMATH_CALUDE_collinear_points_k_value_l3701_370196

/-- Three points are collinear if they lie on the same straight line -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

/-- The theorem stating that if (-2, -4), (5, k), and (15, 1) are collinear, then k = -33/17 -/
theorem collinear_points_k_value :
  ∀ k : ℝ, collinear (-2, -4) (5, k) (15, 1) → k = -33/17 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l3701_370196


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l3701_370115

theorem solve_system_of_equations (u v : ℚ) 
  (eq1 : 5 * u - 6 * v = 28)
  (eq2 : 3 * u + 5 * v = -13) :
  2 * u + 3 * v = -7767 / 645 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l3701_370115


namespace NUMINAMATH_CALUDE_curve_area_range_l3701_370169

theorem curve_area_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*m*x + 2 = 0) →
  (∀ (x y : ℝ), x^2 + y^2 + 2*m*x + 2 = 0 → π * ((x + m)^2 + y^2) ≥ 4 * π) →
  m ≤ -Real.sqrt 6 ∨ m ≥ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_curve_area_range_l3701_370169


namespace NUMINAMATH_CALUDE_lindas_savings_l3701_370184

theorem lindas_savings (savings : ℝ) : 
  (3 / 5 : ℝ) * savings + 400 = savings → savings = 1000 := by
sorry

end NUMINAMATH_CALUDE_lindas_savings_l3701_370184


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3701_370125

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  (x₁^2 + x₁ - 3 = 0) → 
  (x₂^2 + x₂ - 3 = 0) → 
  x₁^3 - 4*x₂^2 + 19 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3701_370125


namespace NUMINAMATH_CALUDE_expected_value_red_balls_l3701_370151

/-- The expected value of drawing red balls in a specific scenario -/
theorem expected_value_red_balls :
  let total_balls : ℕ := 6
  let red_balls : ℕ := 4
  let white_balls : ℕ := 2
  let num_draws : ℕ := 6
  let p : ℚ := red_balls / total_balls
  let E_ξ : ℚ := num_draws * p
  E_ξ = 4 := by sorry

end NUMINAMATH_CALUDE_expected_value_red_balls_l3701_370151


namespace NUMINAMATH_CALUDE_golf_cost_l3701_370120

/-- If 5 rounds of golf cost $400, then one round of golf costs $80 -/
theorem golf_cost (total_cost : ℝ) (num_rounds : ℕ) (cost_per_round : ℝ) 
  (h1 : total_cost = 400)
  (h2 : num_rounds = 5)
  (h3 : total_cost = num_rounds * cost_per_round) : 
  cost_per_round = 80 := by
  sorry

end NUMINAMATH_CALUDE_golf_cost_l3701_370120


namespace NUMINAMATH_CALUDE_special_function_at_three_l3701_370198

/-- An increasing function satisfying a specific functional equation -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ 
  (∀ x, f (f x - 2^x) = 3)

/-- The value of the special function at 3 is 9 -/
theorem special_function_at_three 
  (f : ℝ → ℝ) (hf : SpecialFunction f) : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_three_l3701_370198


namespace NUMINAMATH_CALUDE_smallest_odd_four_digit_number_l3701_370195

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def swap_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 * c + 100 * b + 10 * a + d

theorem smallest_odd_four_digit_number (n : ℕ) : 
  is_four_digit n ∧ 
  n % 2 = 1 ∧
  swap_digits n - n = 5940 ∧
  n % 9 = 8 ∧
  (∀ m : ℕ, is_four_digit m ∧ m % 2 = 1 ∧ swap_digits m - m = 5940 ∧ m % 9 = 8 → n ≤ m) →
  n = 1979 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_four_digit_number_l3701_370195


namespace NUMINAMATH_CALUDE_cos_max_value_l3701_370163

open Real

theorem cos_max_value (x : ℝ) :
  let f := fun x => 3 - 2 * cos (x + π / 4)
  (∀ x, f x ≤ 5) ∧
  (∃ k : ℤ, f (2 * k * π + 3 * π / 4) = 5) :=
sorry

end NUMINAMATH_CALUDE_cos_max_value_l3701_370163


namespace NUMINAMATH_CALUDE_expression_evaluation_l3701_370141

theorem expression_evaluation : 
  let x : ℕ := 3
  x + x^2 * (x^(x^2)) = 177150 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3701_370141


namespace NUMINAMATH_CALUDE_square_side_length_l3701_370182

theorem square_side_length (s : ℝ) (h : s > 0) :
  s^2 = 6 * (4 * s) → s = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3701_370182


namespace NUMINAMATH_CALUDE_apple_tree_production_l3701_370160

theorem apple_tree_production (first_season : ℕ) : 
  (first_season : ℝ) + 0.8 * first_season + 1.6 * first_season = 680 →
  first_season = 200 := by
sorry

end NUMINAMATH_CALUDE_apple_tree_production_l3701_370160


namespace NUMINAMATH_CALUDE_planes_parallel_from_perpendicular_lines_l3701_370116

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_from_perpendicular_lines 
  (α β : Plane) (m n : Line) :
  perpendicular m α → 
  perpendicular n β → 
  line_parallel m n → 
  parallel α β :=
by sorry

end NUMINAMATH_CALUDE_planes_parallel_from_perpendicular_lines_l3701_370116


namespace NUMINAMATH_CALUDE_bianca_birthday_money_l3701_370137

def birthday_money (friends : Fin 8 → ℝ) (tax_rate : ℝ) : ℝ :=
  let total := (friends 0) + (friends 1) + (friends 2) + (friends 3) +
                (friends 4) + (friends 5) + (friends 6) + (friends 7)
  let tax := tax_rate * total
  total - tax

theorem bianca_birthday_money :
  let friends := fun i => match i with
    | 0 => 10
    | 1 => 15
    | 2 => 20
    | 3 => 12
    | 4 => 18
    | 5 => 22
    | 6 => 16
    | 7 => 12
  birthday_money friends 0.1 = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_bianca_birthday_money_l3701_370137


namespace NUMINAMATH_CALUDE_sum_of_squares_l3701_370152

theorem sum_of_squares (x y z : ℝ) 
  (sum_eq : x + y + z = 13)
  (product_eq : x * y * z = 72)
  (sum_reciprocals_eq : 1/x + 1/y + 1/z = 3/4) :
  x^2 + y^2 + z^2 = 61 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3701_370152


namespace NUMINAMATH_CALUDE_derivative_f_l3701_370194

noncomputable def f (x : ℝ) : ℝ :=
  (4^x * (Real.log 4 * Real.sin (4*x) - 4 * Real.cos (4*x))) / (16 + Real.log 4^2)

theorem derivative_f (x : ℝ) :
  deriv f x = 4^x * Real.sin (4*x) :=
sorry

end NUMINAMATH_CALUDE_derivative_f_l3701_370194


namespace NUMINAMATH_CALUDE_pen_pencil_ratio_l3701_370159

theorem pen_pencil_ratio : 
  ∀ (num_pencils num_pens : ℕ),
    num_pencils = 42 →
    num_pencils = num_pens + 7 →
    (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_ratio_l3701_370159


namespace NUMINAMATH_CALUDE_min_cos_for_valid_sqrt_l3701_370193

theorem min_cos_for_valid_sqrt (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (2 * Real.cos x - 1)) ↔ Real.cos x ≥ (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_min_cos_for_valid_sqrt_l3701_370193


namespace NUMINAMATH_CALUDE_physics_marks_proof_l3701_370155

def english_marks : ℕ := 96
def math_marks : ℕ := 95
def chemistry_marks : ℕ := 87
def biology_marks : ℕ := 92
def average_marks : ℚ := 90.4
def total_subjects : ℕ := 5

theorem physics_marks_proof :
  ∃ (physics_marks : ℕ),
    (english_marks + math_marks + chemistry_marks + biology_marks + physics_marks : ℚ) / total_subjects = average_marks ∧
    physics_marks = 82 := by
  sorry

end NUMINAMATH_CALUDE_physics_marks_proof_l3701_370155


namespace NUMINAMATH_CALUDE_tomatoes_left_l3701_370148

theorem tomatoes_left (initial : ℕ) (picked_day1 : ℕ) (picked_day2 : ℕ) 
  (h1 : initial = 171) 
  (h2 : picked_day1 = 134) 
  (h3 : picked_day2 = 30) : 
  initial - picked_day1 - picked_day2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_l3701_370148


namespace NUMINAMATH_CALUDE_or_equivalence_l3701_370190

-- Define the propositions
variable (p : Prop)  -- Athlete A's trial jump exceeded 2 meters
variable (q : Prop)  -- Athlete B's trial jump exceeded 2 meters

-- Define the statement "At least one of Athlete A or B exceeded 2 meters in their trial jump"
def atLeastOneExceeded (p q : Prop) : Prop :=
  p ∨ q

-- Theorem stating the equivalence
theorem or_equivalence :
  (p ∨ q) ↔ atLeastOneExceeded p q :=
sorry

end NUMINAMATH_CALUDE_or_equivalence_l3701_370190


namespace NUMINAMATH_CALUDE_polynomial_value_l3701_370140

theorem polynomial_value (p q : ℝ) : 
  (2*p - q + 3)^2 + 6*(2*p - q + 3) + 6 = (p + 4*q)^2 + 6*(p + 4*q) + 6 →
  p - 5*q + 3 ≠ 0 →
  (5*(p + q + 1))^2 + 6*(5*(p + q + 1)) + 6 = 46 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3701_370140


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_five_l3701_370106

/-- A geometric sequence with common ratio not equal to 1 -/
structure GeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  q_ne_one : q ≠ 1
  geom_prop : ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
def sum_n (g : GeometricSequence) (n : ℕ) : ℚ :=
  (g.a 1) * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_sum_five (g : GeometricSequence) 
  (h1 : g.a 1 * g.a 2 * g.a 3 * g.a 4 * g.a 5 = 1 / 1024)
  (h2 : 2 * g.a 4 = g.a 2 + g.a 3) : 
  sum_n g 5 = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_five_l3701_370106


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l3701_370166

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term of the sequence is given by a * r^(n-1) -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The 8th term of a geometric sequence with first term 5 and common ratio 2/3
    is equal to 640/2187 -/
theorem eighth_term_of_sequence :
  geometric_sequence 5 (2/3) 8 = 640/2187 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l3701_370166


namespace NUMINAMATH_CALUDE_smallest_non_odd_units_digit_zero_not_in_odd_units_digits_smallest_non_odd_units_digit_is_zero_l3701_370132

def OddUnitsDigits : Set ℕ := {1, 3, 5, 7, 9}
def StandardDigits : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem smallest_non_odd_units_digit : 
  ∀ d ∈ StandardDigits, d ∉ OddUnitsDigits → d ≥ 0 :=
by sorry

theorem zero_not_in_odd_units_digits : 0 ∉ OddUnitsDigits :=
by sorry

theorem smallest_non_odd_units_digit_is_zero : 
  ∀ d ∈ StandardDigits, d ∉ OddUnitsDigits → d ≥ 0 ∧ 0 ∉ OddUnitsDigits ∧ 0 ∈ StandardDigits :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_odd_units_digit_zero_not_in_odd_units_digits_smallest_non_odd_units_digit_is_zero_l3701_370132


namespace NUMINAMATH_CALUDE_sum_of_solutions_l3701_370111

theorem sum_of_solutions (S : ℝ) : 
  ∃ (N₁ N₂ : ℝ), N₁ ≠ 0 ∧ N₂ ≠ 0 ∧ 
  (6 * N₁ + 2 / N₁ = S) ∧ 
  (6 * N₂ + 2 / N₂ = S) ∧ 
  (N₁ + N₂ = S / 6) := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l3701_370111


namespace NUMINAMATH_CALUDE_annika_three_times_hans_age_l3701_370165

/-- The number of years in the future when Annika will be three times as old as Hans -/
def future_years : ℕ := 4

/-- Hans's current age -/
def hans_current_age : ℕ := 8

/-- Annika's current age -/
def annika_current_age : ℕ := 32

theorem annika_three_times_hans_age :
  annika_current_age + future_years = 3 * (hans_current_age + future_years) :=
by sorry

end NUMINAMATH_CALUDE_annika_three_times_hans_age_l3701_370165


namespace NUMINAMATH_CALUDE_expression_equivalence_l3701_370129

/-- Prove that the given expression is equivalent to 4xy(x^2 + y^2)/(x^4 + y^4) -/
theorem expression_equivalence (x y : ℝ) :
  let P := x^2 + y^2
  let Q := x*y
  ((P + Q) / (P - Q)) - ((P - Q) / (P + Q)) = (4*x*y*(x^2 + y^2)) / (x^4 + y^4) := by
sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3701_370129


namespace NUMINAMATH_CALUDE_quadratic_max_value_l3701_370188

theorem quadratic_max_value :
  let f : ℝ → ℝ := fun x ↦ -3 * x^2 + 6 * x + 4
  ∃ m : ℝ, (∀ x : ℝ, f x ≤ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l3701_370188


namespace NUMINAMATH_CALUDE_spanning_rectangles_odd_l3701_370189

/-- Represents a 2 × 1 rectangle used to cover the cube surface -/
structure Rectangle :=
  (spans_two_faces : Bool)

/-- Represents the surface of a 9 × 9 × 9 cube -/
structure CubeSurface :=
  (side_length : Nat)
  (covering : List Rectangle)

/-- Axiom: The cube is 9 × 9 × 9 -/
axiom cube_size : ∀ (c : CubeSurface), c.side_length = 9

/-- Axiom: The surface is completely covered without gaps or overlaps -/
axiom complete_coverage : ∀ (c : CubeSurface), c.covering.length * 2 = 6 * c.side_length^2

/-- Main theorem: The number of rectangles spanning two faces is odd -/
theorem spanning_rectangles_odd (c : CubeSurface) : 
  Odd (c.covering.filter Rectangle.spans_two_faces).length :=
sorry

end NUMINAMATH_CALUDE_spanning_rectangles_odd_l3701_370189


namespace NUMINAMATH_CALUDE_xiaogangSavings_l3701_370103

/-- Represents the correct inequality for Xiaogang's savings plan -/
theorem xiaogangSavings (x : ℕ) (initialSavings : ℕ) (monthlySavings : ℕ) (targetAmount : ℕ) : 
  initialSavings = 50 → monthlySavings = 30 → targetAmount = 280 →
  (monthlySavings * x + initialSavings ≥ targetAmount ↔ 
   x ≥ (targetAmount - initialSavings) / monthlySavings) :=
by sorry

end NUMINAMATH_CALUDE_xiaogangSavings_l3701_370103


namespace NUMINAMATH_CALUDE_dave_spent_29_dollars_l3701_370109

/-- Represents the cost of rides for a day at the fair -/
structure DayAtFair where
  rides : List ℕ

/-- Calculates the total cost of rides for a day -/
def totalCost (day : DayAtFair) : ℕ :=
  day.rides.sum

/-- Represents Dave's two days at the fair -/
def davesFairDays : List DayAtFair := [
  { rides := [4, 5, 3, 2] },  -- First day
  { rides := [5, 6, 4] }     -- Second day
]

theorem dave_spent_29_dollars : 
  (davesFairDays.map totalCost).sum = 29 := by
  sorry

end NUMINAMATH_CALUDE_dave_spent_29_dollars_l3701_370109


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l3701_370130

theorem perfect_square_trinomial_m_values (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 2*(m-3)*x + 16 = (a*x + b)^2) → 
  (m = 7 ∨ m = -1) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l3701_370130


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_l3701_370178

theorem greatest_integer_with_gcf_five : ∃ n : ℕ, n < 200 ∧ Nat.gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 → Nat.gcd m 30 = 5 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_l3701_370178


namespace NUMINAMATH_CALUDE_intersection_condition_implies_a_geq_5_l3701_370138

open Set Real

theorem intersection_condition_implies_a_geq_5 (a : ℝ) :
  let A := {x : ℝ | x ≤ a}
  let B := {x : ℝ | x^2 - 5*x < 0}
  A ∩ B = B → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_implies_a_geq_5_l3701_370138


namespace NUMINAMATH_CALUDE_equal_squares_from_sum_product_l3701_370186

theorem equal_squares_from_sum_product (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h : a * (b + c + d) = b * (a + c + d) ∧ 
       b * (a + c + d) = c * (a + b + d) ∧ 
       c * (a + b + d) = d * (a + b + c)) : 
  a^2 = b^2 ∧ b^2 = c^2 ∧ c^2 = d^2 := by
sorry

end NUMINAMATH_CALUDE_equal_squares_from_sum_product_l3701_370186


namespace NUMINAMATH_CALUDE_rectangles_on_clock_face_l3701_370180

/-- The number of equally spaced points on a circle -/
def n : ℕ := 12

/-- A function that calculates the number of rectangles that can be formed
    by selecting 4 vertices from n equally spaced points on a circle -/
def count_rectangles (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of rectangles formed is 15 when n = 12 -/
theorem rectangles_on_clock_face : count_rectangles n = 15 := by sorry

end NUMINAMATH_CALUDE_rectangles_on_clock_face_l3701_370180


namespace NUMINAMATH_CALUDE_monotonically_decreasing_interval_l3701_370147

-- Define the function f(x) = x³ - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Theorem statement
theorem monotonically_decreasing_interval :
  ∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_interval_l3701_370147


namespace NUMINAMATH_CALUDE_sine_cosine_power_sum_l3701_370135

theorem sine_cosine_power_sum (x : ℝ) (h : Real.sin x + Real.cos x = -1) :
  ∀ n : ℕ, (Real.sin x)^n + (Real.cos x)^n = (-1)^n := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_power_sum_l3701_370135


namespace NUMINAMATH_CALUDE_teal_color_survey_l3701_370149

theorem teal_color_survey (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 →
  more_green = 90 →
  both = 40 →
  neither = 25 →
  ∃ (more_blue : ℕ), more_blue = 75 ∧ 
    more_blue + (more_green - both) + neither = total :=
by sorry

end NUMINAMATH_CALUDE_teal_color_survey_l3701_370149


namespace NUMINAMATH_CALUDE_ghost_entrance_exit_ways_l3701_370122

/-- The number of windows in the haunted mansion -/
def num_windows : ℕ := 8

/-- The number of ways Georgie can enter and exit the mansion -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem: The number of ways Georgie can enter and exit the mansion is 56 -/
theorem ghost_entrance_exit_ways : num_ways = 56 := by
  sorry

end NUMINAMATH_CALUDE_ghost_entrance_exit_ways_l3701_370122


namespace NUMINAMATH_CALUDE_power_quotient_23_l3701_370164

theorem power_quotient_23 : (23 ^ 11) / (23 ^ 8) = 12167 := by sorry

end NUMINAMATH_CALUDE_power_quotient_23_l3701_370164


namespace NUMINAMATH_CALUDE_largest_lcm_with_18_l3701_370185

theorem largest_lcm_with_18 :
  (List.map (lcm 18) [3, 6, 9, 12, 15, 18]).maximum? = some 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_18_l3701_370185


namespace NUMINAMATH_CALUDE_function_properties_l3701_370181

/-- Given a function f(x) = a*sin(x) + b*cos(x) where ab ≠ 0, a and b are real numbers,
    and for all x in ℝ, f(x) ≥ f(5π/6), then:
    1. f(π/3) = 0
    2. The line passing through (a, b) intersects the graph of f(x) -/
theorem function_properties (a b : ℝ) (h1 : a * b ≠ 0) :
  let f := fun (x : ℝ) ↦ a * Real.sin x + b * Real.cos x
  (∀ x : ℝ, f x ≥ f (5 * Real.pi / 6)) →
  (f (Real.pi / 3) = 0) ∧
  (∃ x : ℝ, f x = a * x + b) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3701_370181


namespace NUMINAMATH_CALUDE_necessary_condition_for_inequality_l3701_370118

theorem necessary_condition_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 2 3 → x^2 - a ≤ 0) → 
  (a ≥ 8 ∧ ∃ b : ℝ, b ≥ 8 ∧ ∃ y : ℝ, y ∈ Set.Icc 2 3 ∧ y^2 - b > 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_condition_for_inequality_l3701_370118


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l3701_370170

def digits : List Nat := [0, 1, 3, 5]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n ≤ 999 ∧ (∀ d, d ∈ digits → (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d))

def largest_number : Nat :=
  531

def smallest_number : Nat :=
  103

theorem sum_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n, is_valid_number n → n ≤ largest_number) ∧
  (∀ n, is_valid_number n → n ≥ smallest_number) ∧
  largest_number + smallest_number = 634 :=
sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l3701_370170


namespace NUMINAMATH_CALUDE_weight_difference_l3701_370105

/-- Antonio's weight in kilograms -/
def antonio_weight : ℕ := 50

/-- Total weight of Antonio and his sister in kilograms -/
def total_weight : ℕ := 88

/-- Antonio's sister's weight in kilograms -/
def sister_weight : ℕ := total_weight - antonio_weight

theorem weight_difference :
  antonio_weight > sister_weight ∧
  antonio_weight - sister_weight = 12 := by
  sorry

#check weight_difference

end NUMINAMATH_CALUDE_weight_difference_l3701_370105


namespace NUMINAMATH_CALUDE_part_one_part_two_l3701_370126

/-- The quadratic equation -/
def quadratic (k x : ℝ) : ℝ := k * x^2 + 4 * x + 1

/-- Part 1: Prove that if x = -1 is a solution, then k = 3 -/
theorem part_one (k : ℝ) :
  quadratic k (-1) = 0 → k = 3 := by sorry

/-- Part 2: Prove that if the equation has two real roots and k ≠ 0, then k ≤ 4 and k ≠ 0 -/
theorem part_two (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0) →
  k ≠ 0 →
  k ≤ 4 ∧ k ≠ 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3701_370126


namespace NUMINAMATH_CALUDE_blue_has_most_marbles_blue_greater_than_others_l3701_370117

/-- Represents the colors of marbles -/
inductive Color
| Red
| Blue
| Yellow

/-- Represents the marble counting problem -/
structure MarbleCounting where
  total : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- The conditions of the marble counting problem -/
def marbleProblem : MarbleCounting where
  total := 24
  red := 24 / 4
  blue := 24 / 4 + 6
  yellow := 24 - (24 / 4 + (24 / 4 + 6))

/-- Function to determine which color has the most marbles -/
def mostMarbles (mc : MarbleCounting) : Color :=
  if mc.blue > mc.red ∧ mc.blue > mc.yellow then Color.Blue
  else if mc.red > mc.blue ∧ mc.red > mc.yellow then Color.Red
  else Color.Yellow

/-- Theorem stating that blue has the most marbles in the given problem -/
theorem blue_has_most_marbles :
  mostMarbles marbleProblem = Color.Blue :=
by
  sorry

/-- Theorem proving that the number of blue marbles is greater than both red and yellow -/
theorem blue_greater_than_others (mc : MarbleCounting) :
  mc.blue > mc.red ∧ mc.blue > mc.yellow →
  mostMarbles mc = Color.Blue :=
by
  sorry

end NUMINAMATH_CALUDE_blue_has_most_marbles_blue_greater_than_others_l3701_370117


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3701_370104

theorem sqrt_product_equality : Real.sqrt 54 * Real.sqrt 50 * Real.sqrt 6 = 90 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3701_370104


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3701_370100

/-- Given that the constant term in the expansion of (x + a/√x)^6 is 15, 
    prove that the positive value of a is 1. -/
theorem constant_term_expansion (a : ℝ) (h : a > 0) : 
  (∃ (x : ℝ), (x + a / Real.sqrt x)^6 = 15 + x * (1 + 1/x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3701_370100


namespace NUMINAMATH_CALUDE_unique_solution_system_l3701_370175

/-- The system of equations has exactly one real solution -/
theorem unique_solution_system :
  ∃! (x y z : ℝ), x + y = 2 ∧ x * y - z^2 = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3701_370175


namespace NUMINAMATH_CALUDE_parabola_equation_theorem_l3701_370187

/-- A parabola with vertex at the origin, x-axis as the axis of symmetry, 
    and passing through the point (4, -2) -/
structure Parabola where
  -- The parabola passes through (4, -2)
  passes_through : (4 : ℝ)^2 + (-2 : ℝ)^2 ≠ 0

/-- The equation of the parabola is either y^2 = x or x^2 = -8y -/
def parabola_equation (p : Parabola) : Prop :=
  (∀ x y : ℝ, y^2 = x) ∨ (∀ x y : ℝ, x^2 = -8*y)

/-- Theorem stating that the parabola satisfies one of the two equations -/
theorem parabola_equation_theorem (p : Parabola) : parabola_equation p := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_theorem_l3701_370187


namespace NUMINAMATH_CALUDE_fourth_power_sum_l3701_370127

theorem fourth_power_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (sum_squares_eq : a^2 + b^2 + c^2 = 5)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 7) :
  a^4 + b^4 + c^4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l3701_370127


namespace NUMINAMATH_CALUDE_parallel_implies_a_values_l_passes_through_point_l3701_370173

-- Define the lines l and n
def l (a x y : ℝ) : Prop := (a + 2) * x + a * y - 2 = 0
def n (a x y : ℝ) : Prop := (a - 2) * x + 3 * y - 6 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ a, (∃ k ≠ 0, ∀ x y, f a x y ↔ g a (k * x) (k * y))

-- Theorem 1: If l is parallel to n, then a = 6 or a = -1
theorem parallel_implies_a_values :
  parallel l n → ∀ a, (a = 6 ∨ a = -1) :=
sorry

-- Theorem 2: Line l always passes through the point (1, -1)
theorem l_passes_through_point :
  ∀ a, l a 1 (-1) :=
sorry

end NUMINAMATH_CALUDE_parallel_implies_a_values_l_passes_through_point_l3701_370173


namespace NUMINAMATH_CALUDE_line_symmetry_l3701_370150

/-- Given two lines l₁ and l, prove that l₂ is symmetric to l₁ with respect to l -/
theorem line_symmetry (x y : ℝ) : 
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 2 * x + y - 4 = 0
  let l : ℝ → ℝ → Prop := λ x y ↦ 3 * x + 4 * y - 1 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ 2 * x + y - 6 = 0
  (∀ x y, l₁ x y ↔ l₂ x y) ∧ 
  (∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₂ x₂ y₂ → 
    ∃ x₀ y₀, l x₀ y₀ ∧ 
    (x₀ - x₁)^2 + (y₀ - y₁)^2 = (x₀ - x₂)^2 + (y₀ - y₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_line_symmetry_l3701_370150


namespace NUMINAMATH_CALUDE_equilateral_cone_central_angle_l3701_370113

/-- Represents a cone with an equilateral triangle as its axial section -/
structure EquilateralCone where
  /-- The radius of the base of the cone -/
  r : ℝ
  /-- The slant height of the cone, which is twice the radius for an equilateral axial section -/
  slant_height : ℝ
  /-- Condition that the slant height is twice the radius -/
  slant_height_eq : slant_height = 2 * r

/-- The central angle of the side surface development of an equilateral cone is π radians (180°) -/
theorem equilateral_cone_central_angle (cone : EquilateralCone) :
  Real.pi = (2 * Real.pi * cone.r) / cone.slant_height :=
by sorry

end NUMINAMATH_CALUDE_equilateral_cone_central_angle_l3701_370113


namespace NUMINAMATH_CALUDE_one_correct_statement_l3701_370156

theorem one_correct_statement (a b : ℤ) : 
  (∃! n : Nat, n < 3 ∧ n > 0 ∧
    ((n = 1 → (Even (a + 5*b) → Even (a - 7*b))) ∧
     (n = 2 → ((a + b) % 3 = 0 → a % 3 = 0 ∧ b % 3 = 0)) ∧
     (n = 3 → (Prime (a + b) → ¬ Prime (a - b))))) := by
  sorry

end NUMINAMATH_CALUDE_one_correct_statement_l3701_370156


namespace NUMINAMATH_CALUDE_chandler_total_rolls_l3701_370123

/-- The total number of rolls Chandler needs to sell for the school fundraiser -/
def total_rolls_to_sell : ℕ :=
  let grandmother_rolls := 3
  let uncle_rolls := 4
  let neighbor_rolls := 3
  let additional_rolls := 2
  grandmother_rolls + uncle_rolls + neighbor_rolls + additional_rolls

/-- Theorem stating that Chandler needs to sell 12 rolls in total -/
theorem chandler_total_rolls : total_rolls_to_sell = 12 := by
  sorry

end NUMINAMATH_CALUDE_chandler_total_rolls_l3701_370123


namespace NUMINAMATH_CALUDE_sum_of_parts_l3701_370114

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 60) (h2 : y = 45) (h3 : x ≥ 0) (h4 : y ≥ 0) :
  10 * x + 22 * y = 1140 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_l3701_370114


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l3701_370121

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_small_number :
  toScientificNotation 0.000000032 = ScientificNotation.mk 3.2 (-8) sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l3701_370121


namespace NUMINAMATH_CALUDE_cube_of_product_l3701_370139

theorem cube_of_product (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_l3701_370139


namespace NUMINAMATH_CALUDE_average_income_P_and_Q_l3701_370158

theorem average_income_P_and_Q (P Q R : ℕ) : 
  (Q + R) / 2 = 6250 →
  (P + R) / 2 = 5200 →
  P = 4000 →
  (P + Q) / 2 = 5050 := by
  sorry

end NUMINAMATH_CALUDE_average_income_P_and_Q_l3701_370158


namespace NUMINAMATH_CALUDE_julie_age_l3701_370101

theorem julie_age (julie aaron : ℕ) 
  (h1 : julie = 4 * aaron) 
  (h2 : julie + 10 = 2 * (aaron + 10)) : 
  julie = 20 := by
sorry

end NUMINAMATH_CALUDE_julie_age_l3701_370101


namespace NUMINAMATH_CALUDE_gcf_90_108_l3701_370176

theorem gcf_90_108 : Nat.gcd 90 108 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_90_108_l3701_370176


namespace NUMINAMATH_CALUDE_opinion_change_difference_is_twenty_percent_l3701_370162

/-- Represents the percentage of students who like science -/
structure ScienceOpinion where
  initial_like : ℚ
  final_like : ℚ

/-- Calculate the difference between maximum and minimum percentage of students who changed their opinion -/
def opinion_change_difference (opinion : ScienceOpinion) : ℚ :=
  let initial_dislike := 1 - opinion.initial_like
  let final_dislike := 1 - opinion.final_like
  let min_change := |opinion.final_like - opinion.initial_like|
  let max_change := min opinion.initial_like final_dislike + min initial_dislike opinion.final_like
  max_change - min_change

/-- Theorem statement for the specific problem -/
theorem opinion_change_difference_is_twenty_percent :
  ∃ (opinion : ScienceOpinion),
    opinion.initial_like = 2/5 ∧
    opinion.final_like = 4/5 ∧
    opinion_change_difference opinion = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_opinion_change_difference_is_twenty_percent_l3701_370162


namespace NUMINAMATH_CALUDE_curve_translation_l3701_370153

/-- The original curve equation -/
def original_curve (x y : ℝ) : Prop :=
  x^2 - y^2 - 2*x - 2*y - 1 = 0

/-- The transformed curve equation -/
def transformed_curve (x' y' : ℝ) : Prop :=
  x'^2 - y'^2 = 1

/-- The translation vector -/
def translation : ℝ × ℝ := (1, -1)

/-- Theorem stating that the given translation transforms the original curve to the transformed curve -/
theorem curve_translation :
  ∀ (x y : ℝ), original_curve x y ↔ transformed_curve (x - translation.1) (y - translation.2) :=
by sorry

end NUMINAMATH_CALUDE_curve_translation_l3701_370153


namespace NUMINAMATH_CALUDE_some_number_value_l3701_370145

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 25 * 35 * 63) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3701_370145


namespace NUMINAMATH_CALUDE_canvas_bag_break_even_trips_eq_300_l3701_370168

/-- The number of shopping trips required for a canvas bag to become the lower-carbon solution compared to plastic bags. -/
def canvas_bag_break_even_trips (canvas_bag_co2_pounds : ℕ) (plastic_bag_co2_ounces : ℕ) (bags_per_trip : ℕ) (ounces_per_pound : ℕ) : ℕ :=
  (canvas_bag_co2_pounds * ounces_per_pound) / (plastic_bag_co2_ounces * bags_per_trip)

/-- Theorem stating that 300 shopping trips are required for the canvas bag to become the lower-carbon solution. -/
theorem canvas_bag_break_even_trips_eq_300 :
  canvas_bag_break_even_trips 600 4 8 16 = 300 := by
  sorry

end NUMINAMATH_CALUDE_canvas_bag_break_even_trips_eq_300_l3701_370168


namespace NUMINAMATH_CALUDE_lisa_spoon_count_l3701_370174

/-- The number of spoons Lisa has after replacing her old cutlery -/
def total_spoons (num_children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) 
  (large_spoons : ℕ) (teaspoons : ℕ) : ℕ :=
  num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

/-- Proof that Lisa has 39 spoons in total -/
theorem lisa_spoon_count : 
  total_spoons 4 3 2 10 15 = 39 := by
  sorry

end NUMINAMATH_CALUDE_lisa_spoon_count_l3701_370174


namespace NUMINAMATH_CALUDE_profit_share_ratio_l3701_370144

/-- The ratio of profit shares for two investors is proportional to their investments -/
theorem profit_share_ratio (p_investment q_investment : ℕ) 
  (hp : p_investment = 52000)
  (hq : q_investment = 65000) :
  ∃ (k : ℕ), k ≠ 0 ∧ 
    p_investment * 5 = q_investment * 4 * k ∧ 
    q_investment * 4 = p_investment * 4 * k :=
sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l3701_370144


namespace NUMINAMATH_CALUDE_fraction_sum_proof_l3701_370183

theorem fraction_sum_proof : 
  let a : ℚ := 12 / 15
  let b : ℚ := 7 / 9
  let c : ℚ := 1 + 1 / 6
  let sum : ℚ := a + b + c
  sum = 247 / 90 ∧ (∀ n d : ℕ, n ≠ 0 → d ≠ 0 → (n : ℚ) / d = sum → n ≥ 247 ∧ d ≥ 90) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_proof_l3701_370183


namespace NUMINAMATH_CALUDE_rectangle_sections_3x5_l3701_370179

/-- The number of rectangular sections (including squares) in a grid --/
def rectangleCount (width height : ℕ) : ℕ :=
  let squareCount := (width * (width + 1) * height * (height + 1)) / 4
  let rectangleCount := (width * (width + 1) * height * (height + 1)) / 4 - (width * height)
  squareCount + rectangleCount

/-- Theorem stating that the number of rectangular sections in a 3x5 grid is 72 --/
theorem rectangle_sections_3x5 :
  rectangleCount 3 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_sections_3x5_l3701_370179


namespace NUMINAMATH_CALUDE_chris_birthday_savings_l3701_370199

/-- Chris's birthday savings problem -/
theorem chris_birthday_savings 
  (grandmother : ℕ) 
  (aunt_uncle : ℕ) 
  (parents : ℕ) 
  (total_now : ℕ) 
  (h1 : grandmother = 25)
  (h2 : aunt_uncle = 20)
  (h3 : parents = 75)
  (h4 : total_now = 279) :
  total_now - (grandmother + aunt_uncle + parents) = 159 := by
sorry

end NUMINAMATH_CALUDE_chris_birthday_savings_l3701_370199


namespace NUMINAMATH_CALUDE_distribution_within_one_std_dev_l3701_370119

-- Define a symmetric distribution type
structure SymmetricDistribution where
  -- The cumulative distribution function (CDF)
  cdf : ℝ → ℝ
  -- The mean of the distribution
  mean : ℝ
  -- The standard deviation of the distribution
  std_dev : ℝ
  -- Symmetry property
  symmetry : ∀ x, cdf (mean - x) + cdf (mean + x) = 1
  -- Property that 84% of the distribution is less than mean + std_dev
  eighty_four_percent : cdf (mean + std_dev) = 0.84

-- Theorem statement
theorem distribution_within_one_std_dev 
  (d : SymmetricDistribution) : 
  d.cdf (d.mean + d.std_dev) - d.cdf (d.mean - d.std_dev) = 0.68 := by
  sorry

end NUMINAMATH_CALUDE_distribution_within_one_std_dev_l3701_370119


namespace NUMINAMATH_CALUDE_tan_x_is_zero_l3701_370171

theorem tan_x_is_zero (x : Real) 
  (h1 : 0 ≤ x ∧ x ≤ π) 
  (h2 : 3 * Real.sin (x / 2) = Real.sqrt (1 + Real.sin x) - Real.sqrt (1 - Real.sin x)) : 
  Real.tan x = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_is_zero_l3701_370171


namespace NUMINAMATH_CALUDE_one_nonnegative_solution_l3701_370134

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -9*x :=
by sorry

end NUMINAMATH_CALUDE_one_nonnegative_solution_l3701_370134


namespace NUMINAMATH_CALUDE_grocery_bagging_l3701_370157

/-- The number of ways to distribute n distinct objects into k indistinguishable containers,
    allowing empty containers. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- There are 5 different items and 3 identical bags. -/
theorem grocery_bagging : distribute 5 3 = 41 := by
  sorry

end NUMINAMATH_CALUDE_grocery_bagging_l3701_370157


namespace NUMINAMATH_CALUDE_triangle_base_length_l3701_370167

/-- Proves that a triangle with height 8 cm and area 24 cm² has a base length of 6 cm -/
theorem triangle_base_length (height : ℝ) (area : ℝ) (base : ℝ) : 
  height = 8 → area = 24 → area = (base * height) / 2 → base = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3701_370167


namespace NUMINAMATH_CALUDE_mary_needs_four_cups_l3701_370192

/-- The number of cups of flour Mary needs to add to her cake -/
def additional_flour (total_required : ℕ) (already_added : ℕ) : ℕ :=
  total_required - already_added

/-- Proof that Mary needs to add 4 more cups of flour -/
theorem mary_needs_four_cups : additional_flour 10 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mary_needs_four_cups_l3701_370192


namespace NUMINAMATH_CALUDE_difference_of_squares_l3701_370102

theorem difference_of_squares (x : ℝ) : (2 + 3*x) * (2 - 3*x) = 4 - 9*x^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3701_370102


namespace NUMINAMATH_CALUDE_complex_determinant_solution_l3701_370124

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem complex_determinant_solution :
  ∀ z : ℂ, det 1 (-1) z (z * i) = 2 → z = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_determinant_solution_l3701_370124


namespace NUMINAMATH_CALUDE_work_increase_percentage_l3701_370177

/-- Proves that when 1/7 of the members are absent in an office, 
    the percentage increase in work for each remaining person is 100/6. -/
theorem work_increase_percentage (p : ℝ) (p_pos : p > 0) : 
  let absent_fraction : ℝ := 1/7
  let remaining_fraction : ℝ := 1 - absent_fraction
  let work_increase_ratio : ℝ := 1 / remaining_fraction
  let percentage_increase : ℝ := (work_increase_ratio - 1) * 100
  percentage_increase = 100/6 := by
sorry

#eval (100 : ℚ) / 6  -- To show the approximate decimal value

end NUMINAMATH_CALUDE_work_increase_percentage_l3701_370177


namespace NUMINAMATH_CALUDE_rotten_apples_percentage_l3701_370131

theorem rotten_apples_percentage (total_apples : ℕ) (smelling_ratio : ℚ) (non_smelling_rotten : ℕ) :
  total_apples = 200 →
  smelling_ratio = 7/10 →
  non_smelling_rotten = 24 →
  (non_smelling_rotten : ℚ) / ((1 - smelling_ratio) * total_apples) = 2/5 :=
by
  sorry

end NUMINAMATH_CALUDE_rotten_apples_percentage_l3701_370131


namespace NUMINAMATH_CALUDE_pen_pricing_gain_percentage_l3701_370146

theorem pen_pricing_gain_percentage 
  (cost_price selling_price : ℝ) 
  (h : 20 * cost_price = 12 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 200 / 3 :=
by sorry

end NUMINAMATH_CALUDE_pen_pricing_gain_percentage_l3701_370146


namespace NUMINAMATH_CALUDE_nine_bounces_before_pocket_l3701_370154

/-- Represents a rectangular pool table -/
structure PoolTable where
  width : ℝ
  height : ℝ

/-- Represents a ball's position and direction -/
structure Ball where
  x : ℝ
  y : ℝ
  dx : ℝ
  dy : ℝ

/-- Counts the number of wall bounces before the ball enters a corner pocket -/
def countBounces (table : PoolTable) (ball : Ball) : ℕ :=
  sorry

/-- Theorem stating that a ball on a 12x10 table bounces 9 times before entering a pocket -/
theorem nine_bounces_before_pocket (table : PoolTable) (ball : Ball) :
  table.width = 12 ∧ table.height = 10 ∧ 
  ball.x = 0 ∧ ball.y = 0 ∧ ball.dx = 1 ∧ ball.dy = 1 →
  countBounces table ball = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_bounces_before_pocket_l3701_370154


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l3701_370133

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let nonzero_digits := digits.filter (· ≠ 0)
  (nonzero_digits.reverse.take 2).foldl (fun acc d => acc * 10 + d) 0

theorem last_two_nonzero_digits_80_factorial :
  last_two_nonzero_digits (factorial 80) = 12 := by sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l3701_370133


namespace NUMINAMATH_CALUDE_a_minus_b_plus_c_value_l3701_370107

theorem a_minus_b_plus_c_value (a b c : ℝ) :
  (abs a = 1) → (abs b = 2) → (abs c = 3) → (a > b) → (b > c) →
  ((a - b + c = 0) ∨ (a - b + c = -2)) := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_plus_c_value_l3701_370107


namespace NUMINAMATH_CALUDE_three_teams_of_four_from_twelve_l3701_370108

-- Define the number of participants
def n : ℕ := 12

-- Define the number of teams
def k : ℕ := 3

-- Define the number of players per team
def m : ℕ := 4

-- Theorem statement
theorem three_teams_of_four_from_twelve (h1 : n = 12) (h2 : k = 3) (h3 : m = 4) (h4 : n = k * m) : 
  (Nat.choose n m * Nat.choose (n - m) m * Nat.choose (n - 2*m) m) / (Nat.factorial k) = 5775 := by
  sorry

end NUMINAMATH_CALUDE_three_teams_of_four_from_twelve_l3701_370108


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l3701_370142

theorem rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 15) 
  (h2 : b * c = 18) 
  (h3 : c * a = 20) 
  (h4 : max a (max b c) = 2 * min a (min b c)) : 
  a * b * c = 30 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l3701_370142


namespace NUMINAMATH_CALUDE_president_secretary_selection_l3701_370110

theorem president_secretary_selection (n : ℕ) (h : n = 6) :
  (n * (n - 1) : ℕ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_president_secretary_selection_l3701_370110


namespace NUMINAMATH_CALUDE_gamma_donuts_l3701_370197

/-- Proves that Gamma received 8 donuts given the conditions of the problem -/
theorem gamma_donuts : 
  ∀ (gamma_donuts : ℕ),
  (40 : ℕ) = 8 + 3 * gamma_donuts + gamma_donuts →
  gamma_donuts = 8 := by
sorry

end NUMINAMATH_CALUDE_gamma_donuts_l3701_370197


namespace NUMINAMATH_CALUDE_probability_problem_l3701_370136

-- Define the sample space and events
def Ω : Type := Unit
def A₁ : Set Ω := sorry
def A₂ : Set Ω := sorry
def A₃ : Set Ω := sorry
def B : Set Ω := sorry

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Theorem statement
theorem probability_problem :
  -- 1. A₁, A₂, and A₃ are pairwise mutually exclusive
  (A₁ ∩ A₂ = ∅ ∧ A₁ ∩ A₃ = ∅ ∧ A₂ ∩ A₃ = ∅) ∧
  -- 2. P(B|A₁) = 1/3
  P B / P A₁ = 1/3 ∧
  -- 3. P(B) = 19/48
  P B = 19/48 ∧
  -- 4. A₂ and B are not independent events
  P (A₂ ∩ B) ≠ P A₂ * P B :=
by sorry

end NUMINAMATH_CALUDE_probability_problem_l3701_370136


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3701_370143

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3701_370143


namespace NUMINAMATH_CALUDE_distance_between_harper_and_jack_l3701_370128

/-- The distance between two runners at the end of a race --/
def distance_between_runners (race_length : ℕ) (jack_position : ℕ) : ℕ :=
  race_length - jack_position

/-- Theorem: The distance between Harper and Jack when Harper finished the race is 848 meters --/
theorem distance_between_harper_and_jack :
  let race_length_meters : ℕ := 1000  -- 1 km = 1000 meters
  let jack_position : ℕ := 152
  distance_between_runners race_length_meters jack_position = 848 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_harper_and_jack_l3701_370128


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l3701_370161

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (2/5 : ℂ) + (4/9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (2/5 : ℂ) - (4/9 : ℂ) * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l3701_370161


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3701_370172

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 6).sum (λ k => (Nat.choose 5 k) * (2^(5-k)) * x^(5-k)) = 
  40 * x^2 + (Finset.range 6).sum (λ k => if k ≠ 3 then (Nat.choose 5 k) * (2^(5-k)) * x^(5-k) else 0) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3701_370172


namespace NUMINAMATH_CALUDE_division_problem_l3701_370112

theorem division_problem (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 172)
  (h2 : quotient = 10)
  (h3 : remainder = 2)
  (h4 : dividend = quotient * divisor + remainder) :
  divisor = 17 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3701_370112


namespace NUMINAMATH_CALUDE_fraction_simplification_l3701_370191

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 3 + Real.sqrt 48) = (5 * Real.sqrt 3) / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3701_370191
