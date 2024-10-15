import Mathlib

namespace NUMINAMATH_CALUDE_ratio_equivalence_l4112_411254

theorem ratio_equivalence (a b : ℚ) (h : 5 * a = 6 * b) :
  (a / b = 6 / 5) ∧ (b / a = 5 / 6) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalence_l4112_411254


namespace NUMINAMATH_CALUDE_find_principal_l4112_411204

/-- Given a sum of money P (principal) and an interest rate R,
    calculate the amount after T years with simple interest. -/
def simpleInterest (P R T : ℚ) : ℚ :=
  P + (P * R * T) / 100

theorem find_principal (R : ℚ) :
  ∃ P : ℚ,
    simpleInterest P R 1 = 1717 ∧
    simpleInterest P R 2 = 1734 ∧
    P = 1700 := by
  sorry

end NUMINAMATH_CALUDE_find_principal_l4112_411204


namespace NUMINAMATH_CALUDE_unique_number_outside_range_l4112_411260

theorem unique_number_outside_range 
  (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = (a * x + b) / (c * x + d))
  (hf19 : f 19 = 19)
  (hf97 : f 97 = 97)
  (hfinv : ∀ x, x ≠ -d/c → f (f x) = x) :
  ∃! y, ∀ x, f x ≠ y ∧ y = 58 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_outside_range_l4112_411260


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4112_411265

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7 * Complex.I
  let z₂ : ℂ := 4 - 7 * Complex.I
  (z₁ / z₂) + (z₂ / z₁) = -66 / 65 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4112_411265


namespace NUMINAMATH_CALUDE_number_ordering_l4112_411287

theorem number_ordering : 
  (-1.1 : ℝ) < -0.75 ∧ 
  -0.75 < -2/3 ∧ 
  -2/3 < 1/200 ∧ 
  1/200 = (0.005 : ℝ) ∧ 
  0.005 < 4/6 ∧ 
  4/6 < 5/7 ∧ 
  5/7 < 11/15 ∧ 
  11/15 < 1 := by sorry

end NUMINAMATH_CALUDE_number_ordering_l4112_411287


namespace NUMINAMATH_CALUDE_avery_chicken_count_l4112_411209

theorem avery_chicken_count :
  ∀ (eggs_per_chicken : ℕ) (eggs_per_carton : ℕ) (filled_cartons : ℕ),
    eggs_per_chicken = 6 →
    eggs_per_carton = 12 →
    filled_cartons = 10 →
    filled_cartons * eggs_per_carton / eggs_per_chicken = 20 :=
by sorry

end NUMINAMATH_CALUDE_avery_chicken_count_l4112_411209


namespace NUMINAMATH_CALUDE_part_one_part_two_l4112_411255

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | (x - a) * (x + 1) > 0}
def Q : Set ℝ := {x | |x - 1| ≤ 1}

-- Part 1
theorem part_one : (Set.univ \ P 1) ∪ Q = {x | -1 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : P a ∩ Q = ∅) : a > 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4112_411255


namespace NUMINAMATH_CALUDE_max_cards_no_sum_l4112_411299

/-- Given a positive integer k, prove that from 2k+1 cards numbered 1 to 2k+1,
    the maximum number of cards that can be selected such that no selected number
    is the sum of two other selected numbers is k+1. -/
theorem max_cards_no_sum (k : ℕ) : ∃ (S : Finset ℕ),
  S.card = k + 1 ∧
  S.toSet ⊆ Finset.range (2*k + 2) ∧
  (∀ x ∈ S, ∀ y ∈ S, ∀ z ∈ S, x + y ≠ z) ∧
  (∀ T : Finset ℕ, T.toSet ⊆ Finset.range (2*k + 2) →
    (∀ x ∈ T, ∀ y ∈ T, ∀ z ∈ T, x + y ≠ z) →
    T.card ≤ k + 1) :=
sorry

end NUMINAMATH_CALUDE_max_cards_no_sum_l4112_411299


namespace NUMINAMATH_CALUDE_benny_savings_theorem_l4112_411219

/-- The amount of money Benny adds to his piggy bank in January -/
def january_savings : ℕ := 19

/-- The amount of money Benny adds to his piggy bank in February -/
def february_savings : ℕ := january_savings

/-- The amount of money Benny adds to his piggy bank in March -/
def march_savings : ℕ := 8

/-- The total amount of money in Benny's piggy bank by the end of March -/
def total_savings : ℕ := january_savings + february_savings + march_savings

/-- Theorem stating that the total amount in Benny's piggy bank by the end of March is $46 -/
theorem benny_savings_theorem : total_savings = 46 := by
  sorry

end NUMINAMATH_CALUDE_benny_savings_theorem_l4112_411219


namespace NUMINAMATH_CALUDE_chocolate_bars_unsold_l4112_411251

theorem chocolate_bars_unsold (total_bars : ℕ) (price_per_bar : ℕ) (revenue : ℕ) : 
  total_bars = 11 → price_per_bar = 4 → revenue = 16 → total_bars - (revenue / price_per_bar) = 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_unsold_l4112_411251


namespace NUMINAMATH_CALUDE_prop_p_and_q_false_iff_a_gt_1_l4112_411258

-- Define the propositions p and q
def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → a^x > a^y

def q (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, Real.log (a*x^2 - x + a) = y

-- State the theorem
theorem prop_p_and_q_false_iff_a_gt_1 :
  ∀ a : ℝ, (¬(p a ∧ q a)) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_prop_p_and_q_false_iff_a_gt_1_l4112_411258


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l4112_411225

/-- Calculates the time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (lamp_post_time : ℝ) 
  (h1 : train_length = 75) 
  (h2 : bridge_length = 150) 
  (h3 : lamp_post_time = 2.5) : 
  (train_length + bridge_length) / (train_length / lamp_post_time) = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_train_bridge_crossing_time_l4112_411225


namespace NUMINAMATH_CALUDE_regression_lines_intersect_l4112_411205

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the y-value for a given x-value on a regression line -/
def RegressionLine.evaluate (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

/-- Represents a dataset used for linear regression -/
structure Dataset where
  size : ℕ
  avg_x : ℝ
  avg_y : ℝ

/-- Theorem: Two regression lines from datasets with the same average x and y intersect at (avg_x, avg_y) -/
theorem regression_lines_intersect (data1 data2 : Dataset) (line1 line2 : RegressionLine)
    (h1 : data1.avg_x = data2.avg_x)
    (h2 : data1.avg_y = data2.avg_y)
    (h3 : line1.evaluate data1.avg_x = data1.avg_y)
    (h4 : line2.evaluate data2.avg_x = data2.avg_y) :
    line1.evaluate data1.avg_x = line2.evaluate data2.avg_x := by
  sorry


end NUMINAMATH_CALUDE_regression_lines_intersect_l4112_411205


namespace NUMINAMATH_CALUDE_trees_survived_difference_l4112_411221

theorem trees_survived_difference (initial_trees died_trees : ℕ) 
  (h1 : initial_trees = 11)
  (h2 : died_trees = 2) :
  initial_trees - died_trees - died_trees = 7 := by
  sorry

end NUMINAMATH_CALUDE_trees_survived_difference_l4112_411221


namespace NUMINAMATH_CALUDE_inequality_proof_l4112_411226

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4112_411226


namespace NUMINAMATH_CALUDE_exists_x0_where_P_less_than_Q_l4112_411256

/-- Given two polynomials P and Q, and an interval (r,s) satisfying certain conditions,
    there exists a real x₀ such that P(x₀) < Q(x₀) -/
theorem exists_x0_where_P_less_than_Q
  (a b c d p q r s : ℝ)
  (P : ℝ → ℝ)
  (Q : ℝ → ℝ)
  (h_P : ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + d)
  (h_Q : ∀ x, Q x = x^2 + p*x + q)
  (h_interval : s - r > 2)
  (h_negative : ∀ x, r < x ∧ x < s → P x < 0 ∧ Q x < 0)
  (h_positive_right : ∀ x, x > s → P x > 0 ∧ Q x > 0)
  (h_positive_left : ∀ x, x < r → P x > 0 ∧ Q x > 0) :
  ∃ x₀, P x₀ < Q x₀ :=
by sorry

end NUMINAMATH_CALUDE_exists_x0_where_P_less_than_Q_l4112_411256


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l4112_411208

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, -6)
  let b : ℝ × ℝ := (-1, m)
  parallel a b → m = 3 := by
  sorry

#check parallel_vectors_m_value

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l4112_411208


namespace NUMINAMATH_CALUDE_matchstick_subtraction_theorem_l4112_411228

/-- Represents a collection of matchsticks -/
structure MatchstickSet :=
  (count : ℕ)

/-- Represents a Roman numeral -/
inductive RomanNumeral
  | I
  | V
  | X
  | L
  | C
  | D
  | M

/-- Function to determine if a given number of matchsticks can form a Roman numeral -/
def can_form_roman_numeral (m : MatchstickSet) (r : RomanNumeral) : Prop :=
  match r with
  | RomanNumeral.I => m.count ≥ 1
  | RomanNumeral.V => m.count ≥ 2
  | _ => false  -- For simplicity, we only consider I and V in this problem

/-- The main theorem to prove -/
theorem matchstick_subtraction_theorem :
  ∀ (initial : MatchstickSet),
    initial.count = 10 →
    ∃ (removed : MatchstickSet) (remaining : MatchstickSet),
      removed.count = 7 ∧
      remaining.count = initial.count - removed.count ∧
      can_form_roman_numeral remaining RomanNumeral.I ∧
      can_form_roman_numeral remaining RomanNumeral.V :=
sorry

end NUMINAMATH_CALUDE_matchstick_subtraction_theorem_l4112_411228


namespace NUMINAMATH_CALUDE_set_intersection_complement_l4112_411270

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem set_intersection_complement :
  A ∩ (Set.univ \ B) = {x | 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_set_intersection_complement_l4112_411270


namespace NUMINAMATH_CALUDE_marble_problem_l4112_411288

theorem marble_problem (g j : ℕ) 
  (hg : g % 8 = 5) 
  (hj : j % 8 = 6) : 
  (g + 5 + j) % 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l4112_411288


namespace NUMINAMATH_CALUDE_problem_1_l4112_411292

theorem problem_1 : 12 - (-10) + 7 = 29 := by sorry

end NUMINAMATH_CALUDE_problem_1_l4112_411292


namespace NUMINAMATH_CALUDE_log_simplification_l4112_411284

theorem log_simplification (a b c d x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hx : 0 < x) (hy : 0 < y) :
  Real.log (a^2 / b) + Real.log (b / c) + Real.log (c / d^2) - Real.log ((a^2 * y) / (d^2 * x)) = Real.log (x / y) := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l4112_411284


namespace NUMINAMATH_CALUDE_spring_outing_speeds_l4112_411247

theorem spring_outing_speeds (distance : ℝ) (bus_head_start : ℝ) (car_earlier_arrival : ℝ) :
  distance = 90 →
  bus_head_start = 0.5 →
  car_earlier_arrival = 0.25 →
  ∃ (bus_speed car_speed : ℝ),
    car_speed = 1.5 * bus_speed ∧
    distance / bus_speed - distance / car_speed = bus_head_start + car_earlier_arrival ∧
    bus_speed = 40 ∧
    car_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_spring_outing_speeds_l4112_411247


namespace NUMINAMATH_CALUDE_path_area_is_78_l4112_411269

/-- Represents the dimensions of a garden with flower beds and paths. -/
structure GardenDimensions where
  rows : Nat
  columns : Nat
  bedLength : Nat
  bedWidth : Nat
  pathWidth : Nat

/-- Calculates the total area of paths in a garden given its dimensions. -/
def pathArea (g : GardenDimensions) : Nat :=
  let totalWidth := g.pathWidth + g.columns * g.bedLength + (g.columns - 1) * g.pathWidth + g.pathWidth
  let totalHeight := g.pathWidth + g.rows * g.bedWidth + (g.rows - 1) * g.pathWidth + g.pathWidth
  let totalArea := totalWidth * totalHeight
  let bedArea := g.rows * g.columns * g.bedLength * g.bedWidth
  totalArea - bedArea

/-- Theorem stating that the path area for the given garden dimensions is 78 square feet. -/
theorem path_area_is_78 (g : GardenDimensions) 
    (h1 : g.rows = 3) 
    (h2 : g.columns = 2) 
    (h3 : g.bedLength = 6) 
    (h4 : g.bedWidth = 2) 
    (h5 : g.pathWidth = 1) : 
  pathArea g = 78 := by
  sorry

end NUMINAMATH_CALUDE_path_area_is_78_l4112_411269


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l4112_411252

theorem quadratic_integer_roots (a : ℤ) :
  (∃ x y : ℤ, (a + 1) * x^2 - (a^2 + 1) * x + 2 * a^3 - 6 = 0 ∧
               (a + 1) * y^2 - (a^2 + 1) * y + 2 * a^3 - 6 = 0 ∧
               x ≠ y) ↔
  (a = 0 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l4112_411252


namespace NUMINAMATH_CALUDE_unique_solution_l4112_411286

theorem unique_solution : ∃! (x y : ℝ), 
  (2 * x + 3 * y = (7 - 2 * x) + (7 - 3 * y)) ∧ 
  (x - 2 * y = (x - 2) + (2 * y - 2)) ∧
  x = 2 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l4112_411286


namespace NUMINAMATH_CALUDE_number_of_baskets_is_one_l4112_411245

-- Define the total number of peaches
def total_peaches : ℕ := 10

-- Define the number of red peaches per basket
def red_peaches_per_basket : ℕ := 4

-- Define the number of green peaches per basket
def green_peaches_per_basket : ℕ := 6

-- Theorem to prove
theorem number_of_baskets_is_one :
  let peaches_per_basket := red_peaches_per_basket + green_peaches_per_basket
  total_peaches / peaches_per_basket = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_of_baskets_is_one_l4112_411245


namespace NUMINAMATH_CALUDE_line_intersection_range_l4112_411214

theorem line_intersection_range (m : ℝ) : 
  (∀ x y : ℝ, y = (m + 1) * x + m - 1 → (x = 0 → y ≤ 0)) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_range_l4112_411214


namespace NUMINAMATH_CALUDE_parallelogram_side_length_comparison_l4112_411210

/-- Represents a parallelogram in 2D space -/
structure Parallelogram :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Checks if parallelogram inner is inside parallelogram outer -/
def is_inside (inner outer : Parallelogram) : Prop := sorry

/-- Checks if the vertices of inner are on the edges of outer -/
def vertices_on_edges (inner outer : Parallelogram) : Prop := sorry

/-- Checks if the sides of para1 are parallel to the sides of para2 -/
def sides_parallel (para1 para2 : Parallelogram) : Prop := sorry

/-- Computes the length of a side of a parallelogram -/
def side_length (p : Parallelogram) (side : Fin 4) : ℝ := sorry

theorem parallelogram_side_length_comparison 
  (P1 P2 P3 : Parallelogram) 
  (h1 : is_inside P3 P2)
  (h2 : is_inside P2 P1)
  (h3 : vertices_on_edges P3 P2)
  (h4 : vertices_on_edges P2 P1)
  (h5 : sides_parallel P3 P1) :
  ∃ (side : Fin 4), side_length P3 side ≥ (side_length P1 side) / 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_comparison_l4112_411210


namespace NUMINAMATH_CALUDE_ivanov_exaggerating_l4112_411294

-- Define the probabilities of machine breakdowns
def p1 : ℝ := 0.4
def p2 : ℝ := 0.3
def p3 : ℝ := 0

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the expected number of breakdowns per day
def expected_breakdowns_per_day : ℝ := p1 + p2 + p3

-- Define the expected number of breakdowns per week
def expected_breakdowns_per_week : ℝ := expected_breakdowns_per_day * days_in_week

-- Theorem statement
theorem ivanov_exaggerating : expected_breakdowns_per_week < 12 := by
  sorry

end NUMINAMATH_CALUDE_ivanov_exaggerating_l4112_411294


namespace NUMINAMATH_CALUDE_vegetable_project_profit_l4112_411237

-- Define the constants
def initial_investment : ℝ := 600000
def first_year_expense : ℝ := 80000
def annual_expense_increase : ℝ := 20000
def annual_income : ℝ := 260000

-- Define the net profit function
def f (n : ℝ) : ℝ := -n^2 + 19*n - 60

-- Define the theorem to prove
theorem vegetable_project_profit (n : ℝ) :
  f n = n * annual_income - 
    (n * first_year_expense + (n * (n - 1) / 2) * annual_expense_increase) - 
    initial_investment / 10000 ∧
  (∀ m : ℝ, m < 5 → f m ≤ 0) ∧
  (∀ m : ℝ, m ≥ 5 → f m > 0) :=
sorry

end NUMINAMATH_CALUDE_vegetable_project_profit_l4112_411237


namespace NUMINAMATH_CALUDE_dance_event_relation_l4112_411266

/-- Represents a dance event with boys and girls -/
structure DanceEvent where
  b : ℕ  -- Total number of boys
  g : ℕ  -- Total number of girls

/-- The number of girls the nth boy dances with -/
def girlsForBoy (n : ℕ) : ℕ := 7 + 2 * (n - 1)

/-- Axiom: The last boy dances with all girls -/
axiom last_boy_dances_all (event : DanceEvent) : girlsForBoy event.b = event.g

/-- Theorem: The relationship between boys and girls in the dance event -/
theorem dance_event_relation (event : DanceEvent) : event.b = (event.g - 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_dance_event_relation_l4112_411266


namespace NUMINAMATH_CALUDE_binomial_coefficient_7_4_l4112_411274

theorem binomial_coefficient_7_4 : Nat.choose 7 4 = 35 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_7_4_l4112_411274


namespace NUMINAMATH_CALUDE_boat_against_stream_distance_l4112_411227

/-- The distance a boat travels against the stream in one hour -/
def distance_against_stream (downstream_distance : ℝ) (still_water_speed : ℝ) : ℝ :=
  still_water_speed - (downstream_distance - still_water_speed)

/-- Theorem: Given a boat that travels 13 km downstream in one hour with a still water speed of 9 km/hr,
    the distance it travels against the stream in one hour is 5 km. -/
theorem boat_against_stream_distance :
  distance_against_stream 13 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_boat_against_stream_distance_l4112_411227


namespace NUMINAMATH_CALUDE_equation_equality_l4112_411203

theorem equation_equality (x y z : ℝ) (h : x / y = 3 / z) : 9 * y^2 = x^2 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l4112_411203


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l4112_411234

theorem sandy_correct_sums :
  ∀ (c i : ℕ),
    c + i = 50 →
    3 * c - 2 * i - (50 - c) = 100 →
    c = 25 := by
  sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l4112_411234


namespace NUMINAMATH_CALUDE_average_of_x_and_y_l4112_411276

theorem average_of_x_and_y (x y : ℝ) : 
  (4 + 6 + 8 + x + y) / 5 = 20 → (x + y) / 2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_average_of_x_and_y_l4112_411276


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l4112_411290

theorem triangle_perimeter_bound (A B C : ℝ) (R : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  ConvexOn ℝ (Set.Ioo 0 π) Real.sin →
  2 * R * (Real.sin A + Real.sin B + Real.sin C) ≤ 3 * Real.sqrt 3 * R :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l4112_411290


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4112_411240

theorem quadratic_inequality (a b c : ℝ) (h : (a + b + c) * c ≤ 0) : b^2 ≥ 4*a*c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4112_411240


namespace NUMINAMATH_CALUDE_product_inequality_l4112_411281

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l4112_411281


namespace NUMINAMATH_CALUDE_marble_ratio_proof_l4112_411259

def marble_problem (initial_marbles : ℕ) (lost_marbles : ℕ) (final_marbles : ℕ) : Prop :=
  let marbles_after_loss := initial_marbles - lost_marbles
  let marbles_given_away := 2 * lost_marbles
  let marbles_before_dog_ate := marbles_after_loss - marbles_given_away
  let marbles_eaten_by_dog := marbles_before_dog_ate - final_marbles
  (2 * marbles_eaten_by_dog = lost_marbles) ∧ (marbles_eaten_by_dog > 0) ∧ (lost_marbles > 0)

theorem marble_ratio_proof : marble_problem 24 4 10 := by sorry

end NUMINAMATH_CALUDE_marble_ratio_proof_l4112_411259


namespace NUMINAMATH_CALUDE_amaya_total_marks_l4112_411275

/-- Represents the marks scored in different subjects -/
structure Marks where
  music : ℕ
  social_studies : ℕ
  arts : ℕ
  maths : ℕ

/-- Calculates the total marks scored in all subjects -/
def total_marks (m : Marks) : ℕ :=
  m.music + m.social_studies + m.arts + m.maths

/-- Theorem stating the total marks Amaya scored -/
theorem amaya_total_marks (m : Marks) 
  (h1 : m.maths + 20 = m.arts)
  (h2 : m.social_studies = m.music + 10)
  (h3 : m.maths = m.arts - m.arts / 10)
  (h4 : m.music = 70) :
  total_marks m = 530 := by
  sorry

#eval total_marks ⟨70, 80, 200, 180⟩

end NUMINAMATH_CALUDE_amaya_total_marks_l4112_411275


namespace NUMINAMATH_CALUDE_area_of_region_l4112_411232

-- Define the region
def region (x y : ℝ) : Prop :=
  Real.sqrt (Real.arcsin y) ≤ Real.sqrt (Real.arccos x) ∧ 
  -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1

-- State the theorem
theorem area_of_region : 
  MeasureTheory.volume {p : ℝ × ℝ | region p.1 p.2} = 1 + π / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l4112_411232


namespace NUMINAMATH_CALUDE_lcm_problem_l4112_411201

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l4112_411201


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4112_411295

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32) :
  a 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4112_411295


namespace NUMINAMATH_CALUDE_subset_range_equivalence_l4112_411291

theorem subset_range_equivalence (a : ℝ) :
  ({x : ℝ | x^2 + 2*(1-a)*x + (3-a) ≤ 0} ⊆ Set.Icc 0 3) ↔ (-1 ≤ a ∧ a ≤ 18/7) := by
  sorry

end NUMINAMATH_CALUDE_subset_range_equivalence_l4112_411291


namespace NUMINAMATH_CALUDE_polynomial_nonnegative_min_value_of_f_l4112_411218

-- Part a
theorem polynomial_nonnegative (x : ℝ) (h : x ≥ 1) :
  x^3 - 5*x^2 + 8*x - 4 ≥ 0 := by sorry

-- Part b
def f (a b : ℝ) := a*b*(a+b-10) + 8*(a+b)

theorem min_value_of_f :
  ∃ (min : ℝ), min = 8 ∧ 
  ∀ (a b : ℝ), a ≥ 1 → b ≥ 1 → f a b ≥ min := by sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_min_value_of_f_l4112_411218


namespace NUMINAMATH_CALUDE_questionnaires_from_unit_D_l4112_411239

/-- Represents the number of questionnaires drawn from each unit -/
structure SampleDistribution where
  unitA : ℕ
  unitB : ℕ
  unitC : ℕ
  unitD : ℕ

/-- The sample distribution forms an arithmetic sequence -/
def is_arithmetic_sequence (s : SampleDistribution) : Prop :=
  s.unitB - s.unitA = s.unitC - s.unitB ∧ s.unitC - s.unitB = s.unitD - s.unitC

/-- The total sample size is 150 -/
def total_sample_size (s : SampleDistribution) : ℕ :=
  s.unitA + s.unitB + s.unitC + s.unitD

theorem questionnaires_from_unit_D 
  (s : SampleDistribution)
  (h1 : is_arithmetic_sequence s)
  (h2 : total_sample_size s = 150)
  (h3 : s.unitB = 30) :
  s.unitD = 60 := by
  sorry

end NUMINAMATH_CALUDE_questionnaires_from_unit_D_l4112_411239


namespace NUMINAMATH_CALUDE_problem_solution_l4112_411248

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

theorem problem_solution (a : ℝ) (h : a > 0) :
  -- Part 1
  (∀ x, f 1 x ≥ 3 * x + 2 ↔ x ≥ 3 ∨ x ≤ -1) ∧
  -- Part 2
  ((∀ x, f a x ≤ 0 ↔ x ≤ -1) → a = 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l4112_411248


namespace NUMINAMATH_CALUDE_n_squared_not_divides_n_factorial_l4112_411217

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem n_squared_not_divides_n_factorial (n : ℕ) :
  (¬ divides (n^2) (n!)) ↔ (Nat.Prime n ∨ n = 4) := by sorry

end NUMINAMATH_CALUDE_n_squared_not_divides_n_factorial_l4112_411217


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l4112_411271

/-- Arithmetic sequence with first term 1 and common difference 2 -/
def arithmetic_seq (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- Geometric sequence with first term 1 and common ratio 2 -/
def geometric_seq (n : ℕ) : ℕ := 2^(n - 1)

/-- The sum of specific terms in the arithmetic sequence -/
theorem sum_of_specific_terms :
  arithmetic_seq (geometric_seq 2) + 
  arithmetic_seq (geometric_seq 3) + 
  arithmetic_seq (geometric_seq 4) = 25 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l4112_411271


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4112_411216

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a n > 0) →  -- All terms are positive
  a 1 = 3 →  -- First term is 3
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 1 + a 2 + a 3 = 21 →  -- Sum of first three terms is 21
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4112_411216


namespace NUMINAMATH_CALUDE_subtract_sqrt_25_equals_negative_2_l4112_411229

theorem subtract_sqrt_25_equals_negative_2 : 3 - Real.sqrt 25 = -2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_sqrt_25_equals_negative_2_l4112_411229


namespace NUMINAMATH_CALUDE_parallel_condition_l4112_411249

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ m₁ ≠ 0 ∧ m₂ ≠ 0

/-- The first line: ax - y + 3 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x - y + 3 = 0

/-- The second line: 2x - (a + 1)y + 4 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  2 * x - (a + 1) * y + 4 = 0

/-- a = -2 is a sufficient but not necessary condition for the lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  (a = -2 → are_parallel a (-1) 3 2 (-(a + 1)) 4) ∧
  ¬(are_parallel a (-1) 3 2 (-(a + 1)) 4 → a = -2) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l4112_411249


namespace NUMINAMATH_CALUDE_expression_value_l4112_411231

theorem expression_value : (85 + 32 / 113) * 113 = 9635 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4112_411231


namespace NUMINAMATH_CALUDE_quadrilateral_is_parallelogram_l4112_411241

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

-- Define an acute angle
structure AcuteAngle where
  vertex : Point2D
  side1 : Line2D
  side2 : Line2D

-- Function to calculate the distance from a point to a line
def distancePointToLine (p : Point2D) (l : Line2D) : ℝ :=
  sorry

-- Function to check if a quadrilateral is convex
def isConvex (q : Quadrilateral) : Prop :=
  sorry

-- Function to check if a quadrilateral is within an angle
def isWithinAngle (q : Quadrilateral) (angle : AcuteAngle) : Prop :=
  sorry

-- Function to check if a quadrilateral is a parallelogram
def isParallelogram (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem quadrilateral_is_parallelogram
  (q : Quadrilateral)
  (angle : AcuteAngle)
  (h_convex : isConvex q)
  (h_within : isWithinAngle q angle)
  (h_distance1 : distancePointToLine q.A angle.side1 + distancePointToLine q.C angle.side1 =
                 distancePointToLine q.B angle.side1 + distancePointToLine q.D angle.side1)
  (h_distance2 : distancePointToLine q.A angle.side2 + distancePointToLine q.C angle.side2 =
                 distancePointToLine q.B angle.side2 + distancePointToLine q.D angle.side2) :
  isParallelogram q :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_is_parallelogram_l4112_411241


namespace NUMINAMATH_CALUDE_min_value_sum_l4112_411296

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Define a as the point where f(x) reaches its minimum value
def a : ℝ := sorry

-- Define b as the minimum value of f(x)
def b : ℝ := f a

-- Theorem statement
theorem min_value_sum :
  ∀ x : ℝ, f x ≥ b ∧ a + b = -3 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l4112_411296


namespace NUMINAMATH_CALUDE_inequality_proof_l4112_411253

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4112_411253


namespace NUMINAMATH_CALUDE_some_number_calculation_l4112_411289

theorem some_number_calculation (X : ℝ) : 
  2 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1600.0000000000002 → X = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_calculation_l4112_411289


namespace NUMINAMATH_CALUDE_range_of_expression_l4112_411243

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 - 6*x = 0) :
  ∃ (min max : ℝ), min = Real.sqrt 5 ∧ max = Real.sqrt 53 ∧
  min ≤ Real.sqrt (2*x^2 + y^2 - 4*x + 5) ∧
  Real.sqrt (2*x^2 + y^2 - 4*x + 5) ≤ max :=
sorry

end NUMINAMATH_CALUDE_range_of_expression_l4112_411243


namespace NUMINAMATH_CALUDE_stanley_distance_difference_l4112_411244

-- Define the constants
def running_distance : ℝ := 4.8
def walking_distance_meters : ℝ := 950

-- Define the conversion factor
def meters_per_kilometer : ℝ := 1000

-- Define the theorem
theorem stanley_distance_difference :
  running_distance - (walking_distance_meters / meters_per_kilometer) = 3.85 := by
  sorry

end NUMINAMATH_CALUDE_stanley_distance_difference_l4112_411244


namespace NUMINAMATH_CALUDE_translated_line_equation_l4112_411246

/-- 
Theorem: The equation of a line with slope 2 passing through the point (2, 5) is y = 2x + 1.
-/
theorem translated_line_equation (x y : ℝ) : 
  (∃ b : ℝ, y = 2 * x + b) ∧ (2 = 2 ∧ 5 = 2 * 2 + y - 2 * x) → y = 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_translated_line_equation_l4112_411246


namespace NUMINAMATH_CALUDE_equation_solution_l4112_411242

theorem equation_solution (x : ℝ) : 
  x ≠ 1 → -x^2 = (2*x + 4)/(x - 1) → x = -2 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4112_411242


namespace NUMINAMATH_CALUDE_parabola_translation_l4112_411238

/-- Given a parabola y = 3x² in the original coordinate system,
    if the x-axis is translated 2 units up and the y-axis is translated 2 units to the right,
    then the equation of the parabola in the new coordinate system is y = 3(x + 2)² - 2 -/
theorem parabola_translation (x y : ℝ) :
  (y = 3 * x^2) →
  (∀ x' y', x' = x - 2 ∧ y' = y - 2) →
  (y = 3 * (x + 2)^2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l4112_411238


namespace NUMINAMATH_CALUDE_three_digit_squares_ending_1001_l4112_411293

theorem three_digit_squares_ending_1001 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → (n^2 % 10000 = 1001 ↔ n = 501 ∨ n = 749) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_squares_ending_1001_l4112_411293


namespace NUMINAMATH_CALUDE_product_equality_l4112_411236

theorem product_equality (a b c d e f : ℝ) 
  (sum_zero : a + b + c + d + e + f = 0)
  (sum_cubes_zero : a^3 + b^3 + c^3 + d^3 + e^3 + f^3 = 0) :
  (a+c)*(a+d)*(a+e)*(a+f) = (b+c)*(b+d)*(b+e)*(b+f) := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l4112_411236


namespace NUMINAMATH_CALUDE_fundraising_problem_l4112_411278

/-- Represents a student with their fundraising goal -/
structure Student where
  name : String
  goal : ℕ

/-- Represents a day's fundraising activity -/
structure FundraisingDay where
  income : ℕ
  expense : ℕ

/-- The fundraising problem -/
theorem fundraising_problem 
  (students : List Student)
  (collective_goal : ℕ)
  (fundraising_days : List FundraisingDay)
  (h1 : students.length = 8)
  (h2 : collective_goal = 3500)
  (h3 : fundraising_days.length = 5)
  (h4 : students.map Student.goal = [350, 450, 500, 550, 600, 650, 450, 550])
  (h5 : fundraising_days.map FundraisingDay.income = [800, 950, 500, 700, 550])
  (h6 : fundraising_days.map FundraisingDay.expense = [100, 150, 50, 75, 100]) :
  (students.map Student.goal = [350, 450, 500, 550, 600, 650, 450, 550]) ∧
  ((fundraising_days.map (λ d => d.income - d.expense)).sum + 3975 = collective_goal + (students.map Student.goal).sum) :=
sorry

end NUMINAMATH_CALUDE_fundraising_problem_l4112_411278


namespace NUMINAMATH_CALUDE_one_twelfth_day_in_minutes_l4112_411267

/-- Proves that 1/12 of a day is equal to 120 minutes -/
theorem one_twelfth_day_in_minutes : 
  (∀ (hours_per_day minutes_per_hour : ℕ), 
    hours_per_day = 24 → 
    minutes_per_hour = 60 → 
    (1 / 12 : ℚ) * (hours_per_day * minutes_per_hour) = 120) := by
  sorry

#check one_twelfth_day_in_minutes

end NUMINAMATH_CALUDE_one_twelfth_day_in_minutes_l4112_411267


namespace NUMINAMATH_CALUDE_sector_central_angle_measures_l4112_411282

/-- A circular sector with given perimeter and area -/
structure Sector where
  perimeter : ℝ
  area : ℝ

/-- The possible radian measures of the central angle for a sector with given perimeter and area -/
def centralAngleMeasures (s : Sector) : Set ℝ :=
  {α : ℝ | ∃ r : ℝ, r > 0 ∧ 2 * r + α * r = s.perimeter ∧ 1 / 2 * α * r^2 = s.area}

/-- Theorem: For a sector with perimeter 6 and area 2, the central angle measure is either 1 or 4 radians -/
theorem sector_central_angle_measures :
  centralAngleMeasures ⟨6, 2⟩ = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_measures_l4112_411282


namespace NUMINAMATH_CALUDE_congruence_solution_a_l4112_411233

theorem congruence_solution_a (x : Int) : 
  (8 * x) % 13 = 3 % 13 ↔ x % 13 = 2 % 13 := by sorry

end NUMINAMATH_CALUDE_congruence_solution_a_l4112_411233


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4112_411206

theorem complex_fraction_simplification (a : ℝ) (h : a ≠ 2) :
  let x := Real.rpow (Real.sqrt 5 - Real.sqrt 3) (1/3) * Real.rpow (8 + 2 * Real.sqrt 15) (1/6) - Real.rpow a (1/3)
  let y := Real.rpow (Real.sqrt 20 + Real.sqrt 12) (1/3) * Real.rpow (8 - 2 * Real.sqrt 15) (1/6) - 2 * Real.rpow (2*a) (1/3) + Real.rpow (a^2) (1/3)
  x / y = 1 / (Real.rpow 2 (1/3) - Real.rpow a (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4112_411206


namespace NUMINAMATH_CALUDE_unknown_van_capacity_l4112_411277

/-- Represents the fleet of vans with their capacities -/
structure Fleet :=
  (total_vans : Nat)
  (standard_capacity : Nat)
  (large_vans : Nat)
  (small_vans : Nat)
  (unknown_van : Nat)
  (total_capacity : Nat)

/-- Theorem stating the capacity of the unknown van -/
theorem unknown_van_capacity (f : Fleet)
  (h1 : f.total_vans = 6)
  (h2 : f.standard_capacity = 8000)
  (h3 : f.large_vans = 3)
  (h4 : f.small_vans = 2)
  (h5 : f.unknown_van = 1)
  (h6 : f.total_capacity = 57600)
  (h7 : f.large_vans * (f.standard_capacity + f.standard_capacity / 2) +
        f.small_vans * f.standard_capacity +
        (f.total_capacity - (f.large_vans * (f.standard_capacity + f.standard_capacity / 2) +
                             f.small_vans * f.standard_capacity)) =
        f.total_capacity) :
  (f.total_capacity - (f.large_vans * (f.standard_capacity + f.standard_capacity / 2) +
                       f.small_vans * f.standard_capacity)) =
  (f.standard_capacity * 7 / 10) :=
by sorry

end NUMINAMATH_CALUDE_unknown_van_capacity_l4112_411277


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l4112_411273

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l4112_411273


namespace NUMINAMATH_CALUDE_floor_sum_example_l4112_411272

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l4112_411272


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_3_mod_8_l4112_411207

theorem largest_integer_less_than_100_remainder_3_mod_8 :
  ∃ n : ℕ, n < 100 ∧ n % 8 = 3 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 3 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_3_mod_8_l4112_411207


namespace NUMINAMATH_CALUDE_unique_solution_l4112_411222

def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - p.2 = 1 ∧ p.1 + 4 * p.2 = 5}

theorem unique_solution : solution_set = {(1, 1)} := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l4112_411222


namespace NUMINAMATH_CALUDE_total_cleaning_time_is_136_l4112_411297

/-- The time in minutes Richard takes to clean his room once -/
def richard_time : ℕ := 22

/-- The time in minutes Cory takes to clean her room once -/
def cory_time : ℕ := richard_time + 3

/-- The time in minutes Blake takes to clean his room once -/
def blake_time : ℕ := cory_time - 4

/-- The number of times they clean their rooms per week -/
def cleanings_per_week : ℕ := 2

/-- The total time spent cleaning rooms by all three people in a week -/
def total_cleaning_time : ℕ := (richard_time + cory_time + blake_time) * cleanings_per_week

theorem total_cleaning_time_is_136 : total_cleaning_time = 136 := by
  sorry

end NUMINAMATH_CALUDE_total_cleaning_time_is_136_l4112_411297


namespace NUMINAMATH_CALUDE_gumball_probability_l4112_411215

/-- Represents a jar of gumballs -/
structure GumballJar where
  blue : ℕ
  pink : ℕ

/-- The probability of drawing a blue gumball -/
def prob_blue (jar : GumballJar) : ℚ :=
  jar.blue / (jar.blue + jar.pink)

/-- The probability of drawing a pink gumball -/
def prob_pink (jar : GumballJar) : ℚ :=
  jar.pink / (jar.blue + jar.pink)

theorem gumball_probability (jar : GumballJar) :
  (prob_blue jar) ^ 2 = 36 / 49 →
  prob_pink jar = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l4112_411215


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l4112_411263

theorem complex_number_magnitude (a : ℝ) :
  let i : ℂ := Complex.I
  let z : ℂ := (a - i)^2
  (∃ b : ℝ, z = b * i) → Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l4112_411263


namespace NUMINAMATH_CALUDE_cats_at_rescue_center_l4112_411262

/-- The number of puppies Hartley has -/
def num_puppies : ℕ := 4

/-- The weight of each puppy in kilograms -/
def puppy_weight : ℚ := 7.5

/-- The weight of each cat in kilograms -/
def cat_weight : ℚ := 2.5

/-- The difference in total weight between cats and puppies in kilograms -/
def weight_difference : ℚ := 5

/-- The number of cats at the rescue center -/
def num_cats : ℕ := 14

theorem cats_at_rescue_center : 
  (↑num_cats : ℚ) * cat_weight = 
    ↑num_puppies * puppy_weight + weight_difference := by
  sorry

end NUMINAMATH_CALUDE_cats_at_rescue_center_l4112_411262


namespace NUMINAMATH_CALUDE_unique_function_is_lcm_l4112_411213

def satisfies_conditions (f : ℕ → ℕ → ℕ) : Prop :=
  (∀ m n, f m n = f n m) ∧
  (∀ n, f n n = n) ∧
  (∀ m n, n > m → (n - m) * f m n = n * f m (n - m))

theorem unique_function_is_lcm :
  ∀ f : ℕ → ℕ → ℕ, satisfies_conditions f → ∀ m n, f m n = Nat.lcm m n := by
  sorry

end NUMINAMATH_CALUDE_unique_function_is_lcm_l4112_411213


namespace NUMINAMATH_CALUDE_uphill_distance_is_six_l4112_411298

/-- Represents the travel problem with given conditions -/
structure TravelProblem where
  total_time : ℚ
  total_distance : ℕ
  speed_uphill : ℕ
  speed_flat : ℕ
  speed_downhill : ℕ

/-- Checks if a solution satisfies the problem conditions -/
def is_valid_solution (problem : TravelProblem) (uphill_distance : ℕ) (flat_distance : ℕ) : Prop :=
  let downhill_distance := problem.total_distance - uphill_distance - flat_distance
  uphill_distance + flat_distance ≤ problem.total_distance ∧
  (uphill_distance : ℚ) / problem.speed_uphill +
  (flat_distance : ℚ) / problem.speed_flat +
  (downhill_distance : ℚ) / problem.speed_downhill = problem.total_time

/-- The main theorem stating that 6 km is the correct uphill distance -/
theorem uphill_distance_is_six (problem : TravelProblem) 
  (h1 : problem.total_time = 67 / 30)
  (h2 : problem.total_distance = 10)
  (h3 : problem.speed_uphill = 4)
  (h4 : problem.speed_flat = 5)
  (h5 : problem.speed_downhill = 6) :
  ∃ (flat_distance : ℕ), is_valid_solution problem 6 flat_distance ∧
  ∀ (other_uphill : ℕ) (other_flat : ℕ),
    other_uphill ≠ 6 → ¬ is_valid_solution problem other_uphill other_flat :=
by sorry


end NUMINAMATH_CALUDE_uphill_distance_is_six_l4112_411298


namespace NUMINAMATH_CALUDE_two_sided_icing_count_l4112_411212

/-- Represents a cubic cake with icing on specific faces -/
structure CubeCake where
  size : Nat
  has_top_icing : Bool
  has_bottom_icing : Bool
  has_side_icing : Bool
  has_middle_layer_icing : Bool

/-- Counts the number of 1×1×1 sub-cubes with icing on exactly two sides -/
def count_two_sided_icing (cake : CubeCake) : Nat :=
  sorry

/-- The main theorem stating that a 5×5×5 cake with specific icing has 24 sub-cubes with icing on two sides -/
theorem two_sided_icing_count :
  let cake : CubeCake := {
    size := 5,
    has_top_icing := true,
    has_bottom_icing := false,
    has_side_icing := true,
    has_middle_layer_icing := true
  }
  count_two_sided_icing cake = 24 := by sorry

end NUMINAMATH_CALUDE_two_sided_icing_count_l4112_411212


namespace NUMINAMATH_CALUDE_base_value_l4112_411220

theorem base_value (b x y : ℕ) (h1 : b^x * 4^y = 59049) (h2 : x - y = 10) (h3 : x = 10) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_value_l4112_411220


namespace NUMINAMATH_CALUDE_sum_of_integers_l4112_411261

theorem sum_of_integers (x y : ℕ+) (h1 : x.val^2 + y.val^2 = 181) (h2 : x.val * y.val = 90) :
  x.val + y.val = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l4112_411261


namespace NUMINAMATH_CALUDE_student_calculation_error_l4112_411285

theorem student_calculation_error (x y : ℝ) : 
  (5/4 : ℝ) * x = (4/5 : ℝ) * x + 36 ∧ 
  (7/3 : ℝ) * y = (3/7 : ℝ) * y + 28 → 
  x = 80 ∧ y = 14.7 := by
sorry

end NUMINAMATH_CALUDE_student_calculation_error_l4112_411285


namespace NUMINAMATH_CALUDE_increasing_function_t_bound_l4112_411257

def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 3

theorem increasing_function_t_bound (t : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → f t x₁ < f t x₂) →
  t ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_t_bound_l4112_411257


namespace NUMINAMATH_CALUDE_initial_crayons_count_l4112_411202

/-- The initial number of crayons in the drawer -/
def initial_crayons : ℕ := sorry

/-- The number of pencils in the drawer -/
def pencils : ℕ := 26

/-- The number of crayons added to the drawer -/
def added_crayons : ℕ := 12

/-- The total number of crayons after adding -/
def total_crayons : ℕ := 53

theorem initial_crayons_count : initial_crayons = 41 := by
  sorry

end NUMINAMATH_CALUDE_initial_crayons_count_l4112_411202


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4112_411223

/-- Given that x and y are positive real numbers, x² and y vary inversely,
    and y = 25 when x = 3, prove that x = √3/4 when y = 1200. -/
theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_inverse : ∃ k : ℝ, ∀ x y, x * x * y = k)
  (h_initial : 3 * 3 * 25 = 9 * 25) :
  y = 1200 → x = Real.sqrt 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4112_411223


namespace NUMINAMATH_CALUDE_two_digit_numbers_problem_l4112_411230

theorem two_digit_numbers_problem :
  ∃ (x y : ℕ), 
    x > y ∧ 
    x ≥ 10 ∧ x < 100 ∧ 
    y ≥ 10 ∧ y < 100 ∧ 
    1000 * x + y = 2 * (1000 * y + 10 * x) + 590 ∧
    2 * x + 3 * y = 72 ∧
    x = 21 ∧ 
    y = 10 ∧
    ∀ (a b : ℕ), 
      (a > b ∧ 
       a ≥ 10 ∧ a < 100 ∧ 
       b ≥ 10 ∧ b < 100 ∧ 
       1000 * a + b = 2 * (1000 * b + 10 * a) + 590 ∧
       2 * a + 3 * b = 72) → 
      (a = 21 ∧ b = 10) :=
by sorry


end NUMINAMATH_CALUDE_two_digit_numbers_problem_l4112_411230


namespace NUMINAMATH_CALUDE_same_price_at_12_sheets_unique_equal_price_at_12_sheets_l4112_411250

/-- Represents the pricing structure of a photo company -/
structure PhotoCompany where
  perSheetCost : ℚ
  sittingFee : ℚ

/-- Calculates the total cost for a given number of sheets -/
def totalCost (company : PhotoCompany) (sheets : ℚ) : ℚ :=
  company.perSheetCost * sheets + company.sittingFee

/-- John's Photo World pricing -/
def johnsPhotoWorld : PhotoCompany :=
  { perSheetCost := 2.75, sittingFee := 125 }

/-- Sam's Picture Emporium pricing -/
def samsPictureEmporium : PhotoCompany :=
  { perSheetCost := 1.50, sittingFee := 140 }

/-- Theorem stating that the companies charge the same for 12 sheets -/
theorem same_price_at_12_sheets :
  totalCost johnsPhotoWorld 12 = totalCost samsPictureEmporium 12 := by
  sorry

/-- Theorem stating that 12 is the unique number of sheets where prices are equal -/
theorem unique_equal_price_at_12_sheets :
  ∀ x : ℚ, totalCost johnsPhotoWorld x = totalCost samsPictureEmporium x ↔ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_same_price_at_12_sheets_unique_equal_price_at_12_sheets_l4112_411250


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_five_sixths_l4112_411268

theorem sqrt_difference_equals_five_sixths : 
  Real.sqrt (9 / 4) - Real.sqrt (4 / 9) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_five_sixths_l4112_411268


namespace NUMINAMATH_CALUDE_excellent_round_probability_l4112_411280

/-- Represents the result of a single dart throw -/
inductive DartThrow
| Miss  : DartThrow  -- Didn't land in 8th ring or higher
| Hit   : DartThrow  -- Landed in 8th ring or higher

/-- Represents a round of 3 dart throws -/
def Round := (DartThrow × DartThrow × DartThrow)

/-- Determines if a round is excellent (at least 2 hits) -/
def is_excellent (r : Round) : Bool :=
  match r with
  | (DartThrow.Hit, DartThrow.Hit, _) => true
  | (DartThrow.Hit, _, DartThrow.Hit) => true
  | (_, DartThrow.Hit, DartThrow.Hit) => true
  | _ => false

/-- The total number of rounds in the experiment -/
def total_rounds : Nat := 20

/-- The number of excellent rounds observed -/
def excellent_rounds : Nat := 12

/-- Theorem: The probability of an excellent round is 0.6 -/
theorem excellent_round_probability :
  (excellent_rounds : ℚ) / total_rounds = 0.6 := by sorry

end NUMINAMATH_CALUDE_excellent_round_probability_l4112_411280


namespace NUMINAMATH_CALUDE_tan_4530_degrees_l4112_411200

theorem tan_4530_degrees : Real.tan (4530 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_4530_degrees_l4112_411200


namespace NUMINAMATH_CALUDE_hyperbola_y_foci_coeff_signs_l4112_411235

/-- A curve represented by the equation ax^2 + by^2 = 1 -/
structure Curve where
  a : ℝ
  b : ℝ

/-- Predicate to check if a curve is a hyperbola with foci on the y-axis -/
def is_hyperbola_y_foci (c : Curve) : Prop :=
  ∃ (p q : ℝ), p > 0 ∧ q > 0 ∧ ∀ (x y : ℝ), c.a * x^2 + c.b * y^2 = 1 ↔ x^2/p - y^2/q = 1

/-- Theorem stating that if a curve is a hyperbola with foci on the y-axis,
    then its 'a' coefficient is negative and 'b' coefficient is positive -/
theorem hyperbola_y_foci_coeff_signs (c : Curve) :
  is_hyperbola_y_foci c → c.a < 0 ∧ c.b > 0 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_y_foci_coeff_signs_l4112_411235


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l4112_411224

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 45 and 320 -/
def product : ℕ := 45 * 320

/-- Theorem: The number of trailing zeros in the product 45 × 320 is 2 -/
theorem product_trailing_zeros : trailingZeros product = 2 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l4112_411224


namespace NUMINAMATH_CALUDE_fraction_equality_l4112_411279

theorem fraction_equality (a b : ℝ) (h : (4*a + 3*b) / (4*a - 3*b) = 4) : a / b = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4112_411279


namespace NUMINAMATH_CALUDE_circular_film_radius_l4112_411283

/-- The radius of a circular film formed by pouring a liquid from a rectangular box into water -/
theorem circular_film_radius (box_length box_width box_height film_thickness : ℝ) 
  (h1 : box_length = 8)
  (h2 : box_width = 4)
  (h3 : box_height = 15)
  (h4 : film_thickness = 0.2)
  : ∃ (r : ℝ), r = Real.sqrt (2400 / Real.pi) ∧ 
    π * r^2 * film_thickness = box_length * box_width * box_height :=
by sorry

end NUMINAMATH_CALUDE_circular_film_radius_l4112_411283


namespace NUMINAMATH_CALUDE_expression_equality_l4112_411211

theorem expression_equality :
  ((-3)^2 ≠ -3^2) ∧
  ((-3)^2 = 3^2) ∧
  ((-2)^3 = -2^3) ∧
  (|-2|^3 = |-2^3|) :=
by sorry

end NUMINAMATH_CALUDE_expression_equality_l4112_411211


namespace NUMINAMATH_CALUDE_part_one_part_two_l4112_411264

-- Part I
theorem part_one (a : ℝ) (h : ∀ x, x ∈ Set.Icc (-6 : ℝ) 2 ↔ |a * x - 1| ≤ 2) : 
  a = -1/2 := by sorry

-- Part II
theorem part_two (m : ℝ) (h : ∃ x : ℝ, |4 * x + 1| - |2 * x - 3| ≤ 7 - 3 * m) : 
  m ∈ Set.Iic (7/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4112_411264
