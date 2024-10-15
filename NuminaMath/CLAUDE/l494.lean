import Mathlib

namespace NUMINAMATH_CALUDE_line_up_ways_l494_49482

def number_of_people : ℕ := 5

theorem line_up_ways (n : ℕ) (h : n = number_of_people) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 :=
sorry

end NUMINAMATH_CALUDE_line_up_ways_l494_49482


namespace NUMINAMATH_CALUDE_rectangle_area_l494_49421

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 206) :
  L * B = 2520 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l494_49421


namespace NUMINAMATH_CALUDE_probability_black_white_balls_l494_49459

/-- The probability of picking one black ball and one white ball from a jar -/
theorem probability_black_white_balls (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ)
  (h1 : total_balls = black_balls + white_balls + green_balls)
  (h2 : black_balls = 3)
  (h3 : white_balls = 3)
  (h4 : green_balls = 1) :
  (black_balls * white_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2) = 3 / 7 := by
sorry

end NUMINAMATH_CALUDE_probability_black_white_balls_l494_49459


namespace NUMINAMATH_CALUDE_min_cut_off_length_is_82_l494_49406

/-- Represents the rope cutting problem with given constraints -/
def RopeCuttingProblem (total_length : ℕ) (piece_lengths : List ℕ) (max_pieces : ℕ) : Prop :=
  total_length = 89 ∧
  piece_lengths = [7, 3, 1] ∧
  max_pieces = 25

/-- The minimum length of rope that must be cut off -/
def MinCutOffLength (total_length : ℕ) (piece_lengths : List ℕ) (max_pieces : ℕ) : ℕ := 82

/-- Theorem stating the minimum cut-off length for the rope cutting problem -/
theorem min_cut_off_length_is_82
  (total_length : ℕ) (piece_lengths : List ℕ) (max_pieces : ℕ)
  (h : RopeCuttingProblem total_length piece_lengths max_pieces) :
  MinCutOffLength total_length piece_lengths max_pieces = 82 := by
  sorry

end NUMINAMATH_CALUDE_min_cut_off_length_is_82_l494_49406


namespace NUMINAMATH_CALUDE_seconds_in_misfortune_day_l494_49413

/-- The number of minutes in a day on the island of Misfortune -/
def minutes_per_day : ℕ := 77

/-- The number of seconds in a minute on the island of Misfortune -/
def seconds_per_minute : ℕ := 91

/-- Theorem: The number of seconds in a day on the island of Misfortune is 1001 -/
theorem seconds_in_misfortune_day : 
  minutes_per_day * seconds_per_minute = 1001 := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_misfortune_day_l494_49413


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l494_49415

theorem cubic_roots_sum (a b c : ℝ) (h1 : a < b) (h2 : b < c)
  (ha : a^3 - 3*a + 1 = 0) (hb : b^3 - 3*b + 1 = 0) (hc : c^3 - 3*c + 1 = 0) :
  1 / (a^2 + b) + 1 / (b^2 + c) + 1 / (c^2 + a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l494_49415


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_even_integers_l494_49447

def sum_of_first_n_even_integers (n : ℕ) : ℕ := 2 * n * (n + 1)

def sum_of_five_consecutive_even_integers (largest : ℕ) : ℕ :=
  (largest - 8) + (largest - 6) + (largest - 4) + (largest - 2) + largest

theorem largest_of_five_consecutive_even_integers :
  ∃ (largest : ℕ), 
    sum_of_first_n_even_integers 15 = sum_of_five_consecutive_even_integers largest ∧
    largest = 52 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_even_integers_l494_49447


namespace NUMINAMATH_CALUDE_sweater_markup_l494_49468

theorem sweater_markup (wholesale : ℝ) (retail : ℝ) (h1 : retail > 0) (h2 : wholesale > 0) :
  (0.4 * retail = 1.35 * wholesale) →
  (retail - wholesale) / wholesale * 100 = 237.5 := by
sorry

end NUMINAMATH_CALUDE_sweater_markup_l494_49468


namespace NUMINAMATH_CALUDE_expand_and_simplify_l494_49437

theorem expand_and_simplify (y : ℝ) (h : y ≠ 0) :
  (3 / 4) * (4 / y - 7 * y^3) = 3 / y - 21 * y^3 / 4 := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l494_49437


namespace NUMINAMATH_CALUDE_polygon_sides_l494_49405

theorem polygon_sides (sum_interior_angles : ℝ) (h : sum_interior_angles = 1680) :
  ∃ (n : ℕ), n = 12 ∧ (n - 2) * 180 > sum_interior_angles ∧ (n - 2) * 180 ≤ sum_interior_angles + 180 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l494_49405


namespace NUMINAMATH_CALUDE_sets_A_B_characterization_l494_49486

theorem sets_A_B_characterization (A B : Set ℤ) :
  (A ∪ B = Set.univ) ∧
  (∀ x, x ∈ A → x - 1 ∈ B) ∧
  (∀ x y, x ∈ B ∧ y ∈ B → x + y ∈ A) →
  ((A = {x | ∃ k, x = 2 * k} ∧ B = {x | ∃ k, x = 2 * k + 1}) ∨
   (A = Set.univ ∧ B = Set.univ)) :=
by sorry

end NUMINAMATH_CALUDE_sets_A_B_characterization_l494_49486


namespace NUMINAMATH_CALUDE_arc_problem_l494_49458

theorem arc_problem (X Y Z : ℝ × ℝ) (d : ℝ) : 
  X.1 = 0 ∧ X.2 = 0 ∧  -- Assume X is at origin
  Y.1 = 15 ∧ Y.2 = 0 ∧  -- Assume Y is on x-axis
  Z.1^2 + Z.2^2 = (3 + d)^2 ∧  -- XZ = 3 + d
  (Z.1 - 15)^2 + Z.2^2 = (12 + d)^2 →  -- YZ = 12 + d
  d = 5 := by
sorry

end NUMINAMATH_CALUDE_arc_problem_l494_49458


namespace NUMINAMATH_CALUDE_solution_bijection_l494_49488

def equation_x (x : Fin 10 → ℕ+) : Prop :=
  (x 0) + 2^3 * (x 1) + 3^3 * (x 2) + 4^3 * (x 3) + 5^3 * (x 4) + 
  6^3 * (x 5) + 7^3 * (x 6) + 8^3 * (x 7) + 9^3 * (x 8) + 10^3 * (x 9) = 3025

def equation_y (y : Fin 10 → ℕ) : Prop :=
  (y 0) + 2^3 * (y 1) + 3^3 * (y 2) + 4^3 * (y 3) + 5^3 * (y 4) + 
  6^3 * (y 5) + 7^3 * (y 6) + 8^3 * (y 7) + 9^3 * (y 8) + 10^3 * (y 9) = 0

theorem solution_bijection :
  ∃ (f : {x : Fin 10 → ℕ+ // equation_x x} → {y : Fin 10 → ℕ // equation_y y}),
    Function.Bijective f ∧
    f ⟨λ _ => 1, sorry⟩ = ⟨λ _ => 0, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_solution_bijection_l494_49488


namespace NUMINAMATH_CALUDE_expression_value_l494_49426

theorem expression_value (x y z : ℚ) 
  (eq1 : 2 * x - y = 4)
  (eq2 : 3 * x + z = 7)
  (eq3 : y = 2 * z) :
  6 * x - 3 * y + 3 * z = 51 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l494_49426


namespace NUMINAMATH_CALUDE_dusty_change_l494_49439

def single_layer_cost : ℝ := 4
def single_layer_tax_rate : ℝ := 0.05
def double_layer_cost : ℝ := 7
def double_layer_tax_rate : ℝ := 0.10
def fruit_tart_cost : ℝ := 5
def fruit_tart_tax_rate : ℝ := 0.08

def single_layer_quantity : ℕ := 7
def double_layer_quantity : ℕ := 5
def fruit_tart_quantity : ℕ := 3

def payment_amount : ℝ := 200

theorem dusty_change :
  let single_layer_total := single_layer_quantity * (single_layer_cost * (1 + single_layer_tax_rate))
  let double_layer_total := double_layer_quantity * (double_layer_cost * (1 + double_layer_tax_rate))
  let fruit_tart_total := fruit_tart_quantity * (fruit_tart_cost * (1 + fruit_tart_tax_rate))
  let total_cost := single_layer_total + double_layer_total + fruit_tart_total
  payment_amount - total_cost = 115.90 := by
  sorry

end NUMINAMATH_CALUDE_dusty_change_l494_49439


namespace NUMINAMATH_CALUDE_A_intersect_B_empty_l494_49425

/-- The set A defined by the equation (y-3)/(x-2) = a+1 -/
def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = a + 1}

/-- The set B defined by the equation (a^2-1)x + (a-1)y = 15 -/
def B (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (a^2 - 1) * p.1 + (a - 1) * p.2 = 15}

/-- The theorem stating that A ∩ B is empty if and only if a is in the set {-1, -4, 1, 5/2} -/
theorem A_intersect_B_empty (a : ℝ) :
  A a ∩ B a = ∅ ↔ a ∈ ({-1, -4, 1, (5:ℝ)/2} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_empty_l494_49425


namespace NUMINAMATH_CALUDE_cupcakes_remaining_l494_49414

/-- The number of cupcake packages Maggi had -/
def packages : ℝ := 3.5

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 7

/-- The number of cupcakes Maggi ate -/
def eaten_cupcakes : ℝ := 5.75

/-- The number of cupcakes left after Maggi ate some -/
def cupcakes_left : ℝ := packages * cupcakes_per_package - eaten_cupcakes

theorem cupcakes_remaining : cupcakes_left = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_remaining_l494_49414


namespace NUMINAMATH_CALUDE_parabola_coeff_sum_l494_49440

/-- A parabola with equation y = px^2 + qx + r, vertex (-3, 7), and passing through (-6, 4) -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ
  vertex_x : ℝ := -3
  vertex_y : ℝ := 7
  point_x : ℝ := -6
  point_y : ℝ := 4
  eq_at_vertex : 7 = p * (-3)^2 + q * (-3) + r
  eq_at_point : 4 = p * (-6)^2 + q * (-6) + r

/-- The sum of coefficients p, q, and r for the parabola is 7/3 -/
theorem parabola_coeff_sum (par : Parabola) : par.p + par.q + par.r = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coeff_sum_l494_49440


namespace NUMINAMATH_CALUDE_collin_savings_l494_49489

/-- Represents the number of cans Collin collected from various sources --/
structure CanCollection where
  home : ℕ
  grandparents : ℕ
  neighbor : ℕ
  dad : ℕ

/-- Calculates the total number of cans collected --/
def total_cans (c : CanCollection) : ℕ :=
  c.home + c.grandparents + c.neighbor + c.dad

/-- Represents the recycling scenario for Collin --/
structure RecyclingScenario where
  collection : CanCollection
  price_per_can : ℚ
  savings_ratio : ℚ

/-- Calculates the amount Collin will put into savings --/
def savings_amount (s : RecyclingScenario) : ℚ :=
  s.savings_ratio * s.price_per_can * (total_cans s.collection)

/-- Theorem stating that Collin will put $43.00 into savings --/
theorem collin_savings (s : RecyclingScenario) 
  (h1 : s.collection.home = 12)
  (h2 : s.collection.grandparents = 3 * s.collection.home)
  (h3 : s.collection.neighbor = 46)
  (h4 : s.collection.dad = 250)
  (h5 : s.price_per_can = 1/4)
  (h6 : s.savings_ratio = 1/2) :
  savings_amount s = 43 := by
  sorry

end NUMINAMATH_CALUDE_collin_savings_l494_49489


namespace NUMINAMATH_CALUDE_triangle_proof_l494_49473

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_proof (ABC : Triangle) 
  (h1 : ABC.B.sin = 1/3)
  (h2 : ABC.a^2 - ABC.b^2 + ABC.c^2 = 2 ∨ ABC.a * ABC.c * ABC.B.cos = -1)
  (h3 : ABC.A.sin * ABC.C.sin = Real.sqrt 2 / 3) :
  (ABC.a * ABC.c = 3 * Real.sqrt 2 / 4) ∧ 
  (ABC.b = 1/2) := by
sorry

end NUMINAMATH_CALUDE_triangle_proof_l494_49473


namespace NUMINAMATH_CALUDE_min_extractions_to_reverse_l494_49404

/-- Represents a stack of cards -/
def CardStack := List Nat

/-- Represents an extraction operation on a card stack -/
def Extraction := CardStack → CardStack

/-- Checks if a card stack is in reverse order -/
def is_reversed (stack : CardStack) : Prop :=
  stack = List.reverse (List.range stack.length)

/-- Theorem: Minimum number of extractions to reverse a card stack -/
theorem min_extractions_to_reverse (n : Nat) :
  ∃ (k : Nat) (extractions : List Extraction),
    k = n / 2 + 1 ∧
    extractions.length = k ∧
    is_reversed (extractions.foldl (λ acc f => f acc) (List.range n)) :=
  sorry

end NUMINAMATH_CALUDE_min_extractions_to_reverse_l494_49404


namespace NUMINAMATH_CALUDE_remainder_mod_seven_l494_49455

theorem remainder_mod_seven : (9^5 + 8^4 + 7^9) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_seven_l494_49455


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l494_49496

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (1, 4) and (7, 10) is 11. -/
theorem midpoint_coordinate_sum : 
  let x1 : ℝ := 1
  let y1 : ℝ := 4
  let x2 : ℝ := 7
  let y2 : ℝ := 10
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 11 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l494_49496


namespace NUMINAMATH_CALUDE_remainder_problem_l494_49428

theorem remainder_problem (N : ℤ) : N % 357 = 36 → N % 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l494_49428


namespace NUMINAMATH_CALUDE_unique_solution_x_squared_minus_x_minus_one_l494_49497

theorem unique_solution_x_squared_minus_x_minus_one (x : ℝ) :
  x^2 - x - 1 = (x + 1)^0 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_x_squared_minus_x_minus_one_l494_49497


namespace NUMINAMATH_CALUDE_fifteen_is_counterexample_l494_49464

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_counterexample (n : ℕ) : Prop :=
  ¬(is_prime n) ∧ ¬(is_prime (n - 3))

theorem fifteen_is_counterexample :
  is_counterexample 15 :=
sorry

end NUMINAMATH_CALUDE_fifteen_is_counterexample_l494_49464


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l494_49423

theorem smallest_solution_of_equation :
  ∃ (x : ℝ), x = 39/8 ∧
  x ≠ 0 ∧ x ≠ 3 ∧
  (3*x)/(x-3) + (3*x^2-27)/x = 14 ∧
  ∀ (y : ℝ), y ≠ 0 → y ≠ 3 → (3*y)/(y-3) + (3*y^2-27)/y = 14 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l494_49423


namespace NUMINAMATH_CALUDE_work_completion_time_l494_49419

/-- Given a work that can be completed by X in 40 days, prove that if X works for 8 days
    and Y finishes the remaining work in 32 days, then Y would take 40 days to complete
    the entire work alone. -/
theorem work_completion_time (x_total_days y_completion_days : ℕ) 
    (x_worked_days : ℕ) (h1 : x_total_days = 40) (h2 : x_worked_days = 8) 
    (h3 : y_completion_days = 32) : 
    (x_total_days : ℚ) * y_completion_days / (x_total_days - x_worked_days) = 40 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l494_49419


namespace NUMINAMATH_CALUDE_inequality_addition_l494_49457

theorem inequality_addition (x y : ℝ) (h : x < y) : x + 5 < y + 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l494_49457


namespace NUMINAMATH_CALUDE_sequence_closed_form_l494_49461

def recurrence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = Real.sqrt ((a n + 2 - Real.sqrt (2 - a n)) / 2)

theorem sequence_closed_form (a : ℕ → ℝ) :
  recurrence a ∧ a 0 = Real.sqrt 2 / 2 →
  ∀ n, a n = Real.sqrt 2 * Real.cos (π / 4 + π / (12 * 2^n)) :=
sorry

end NUMINAMATH_CALUDE_sequence_closed_form_l494_49461


namespace NUMINAMATH_CALUDE_dividing_line_theorem_l494_49480

/-- A configuration of six unit squares in two rows of three in the coordinate plane -/
structure SquareGrid :=
  (width : ℕ := 3)
  (height : ℕ := 2)

/-- A line extending from (2,0) to (k,k) -/
structure DividingLine :=
  (k : ℝ)

/-- The area above and below the line formed by the DividingLine -/
def areas (grid : SquareGrid) (line : DividingLine) : ℝ × ℝ :=
  sorry

/-- Theorem stating that k = 4 divides the grid such that the area above is twice the area below -/
theorem dividing_line_theorem (grid : SquareGrid) :
  ∃ (line : DividingLine), 
    let (area_below, area_above) := areas grid line
    line.k = 4 ∧ area_above = 2 * area_below := by sorry

end NUMINAMATH_CALUDE_dividing_line_theorem_l494_49480


namespace NUMINAMATH_CALUDE_remaining_distance_l494_49427

theorem remaining_distance (total_distance driven_distance : ℕ) 
  (h1 : total_distance = 1200)
  (h2 : driven_distance = 384) :
  total_distance - driven_distance = 816 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l494_49427


namespace NUMINAMATH_CALUDE_visitor_and_revenue_properties_l494_49438

/-- Represents the daily change in visitors (in 10,000 people) --/
def visitor_changes : List ℝ := [1.6, 0.8, 0.4, -0.4, -0.8, 0.2, -1.2]

/-- The ticket price per person in yuan --/
def ticket_price : ℝ := 15

/-- Theorem stating the properties of visitor numbers and revenue --/
theorem visitor_and_revenue_properties (a : ℝ) : 
  let visitors_day3 := a + visitor_changes[0] + visitor_changes[1]
  let max_visitors := (List.map (λ i => a + (List.take i visitor_changes).sum) (List.range 7)).maximum?
  let total_visitors := a * 7 + visitor_changes.sum
  (visitors_day3 = a + 2.4) ∧ 
  (max_visitors = some (a + 2.8)) ∧
  (a = 2 → total_visitors * ticket_price * 10000 = 4.08 * 10^6) := by sorry

end NUMINAMATH_CALUDE_visitor_and_revenue_properties_l494_49438


namespace NUMINAMATH_CALUDE_max_abs_diff_on_interval_l494_49475

open Real

-- Define the functions
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x^3

-- Define the absolute difference function
def abs_diff (x : ℝ) : ℝ := |f x - g x|

-- State the theorem
theorem max_abs_diff_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ 
  ∀ (x : ℝ), x ∈ Set.Icc 0 1 → abs_diff x ≤ abs_diff c ∧
  abs_diff c = 4/27 :=
sorry

end NUMINAMATH_CALUDE_max_abs_diff_on_interval_l494_49475


namespace NUMINAMATH_CALUDE_triangle_angle_c_l494_49411

theorem triangle_angle_c (A B C : ℝ) (h1 : A + B = 110) : C = 70 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l494_49411


namespace NUMINAMATH_CALUDE_selling_price_equation_l494_49436

/-- Represents the selling price of pants in a store -/
def selling_price (X : ℝ) : ℝ :=
  let initial_price := X
  let discount_rate := 0.1
  let bulk_discount := 5
  let markup_rate := 0.25
  let discounted_price := initial_price * (1 - discount_rate)
  let final_purchase_cost := discounted_price - bulk_discount
  let marked_up_price := final_purchase_cost * (1 + markup_rate)
  marked_up_price

/-- Theorem stating the relationship between initial purchase price and selling price -/
theorem selling_price_equation (X : ℝ) :
  selling_price X = 1.125 * X - 6.25 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_equation_l494_49436


namespace NUMINAMATH_CALUDE_ten_point_circle_triangles_l494_49481

/-- A circle with 10 points and chords connecting every pair of points. -/
structure CircleWithChords where
  num_points : ℕ
  no_triple_intersections : Bool

/-- The number of triangles formed by chord intersections inside the circle. -/
def num_triangles (c : CircleWithChords) : ℕ := sorry

/-- Theorem stating that the number of triangles is 210 for a circle with 10 points. -/
theorem ten_point_circle_triangles (c : CircleWithChords) : 
  c.num_points = 10 → c.no_triple_intersections → num_triangles c = 210 := by
  sorry

end NUMINAMATH_CALUDE_ten_point_circle_triangles_l494_49481


namespace NUMINAMATH_CALUDE_arithmetic_sequence_range_l494_49403

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_range (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 ≤ 7)
  (h_a6 : a 6 ≥ 9) :
  (a 10 > 11 ∧ ∀ M : ℝ, ∃ N : ℝ, a 10 > N ∧ N > M) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_range_l494_49403


namespace NUMINAMATH_CALUDE_dads_nickels_l494_49409

/-- The number of nickels Tim had initially -/
def initial_nickels : ℕ := 9

/-- The number of nickels Tim has now -/
def current_nickels : ℕ := 12

/-- The number of nickels Tim's dad gave him -/
def nickels_from_dad : ℕ := current_nickels - initial_nickels

theorem dads_nickels : nickels_from_dad = 3 := by
  sorry

end NUMINAMATH_CALUDE_dads_nickels_l494_49409


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l494_49493

/-- Given a point A with coordinates (3,2), prove that the point symmetric 
    to A' with respect to the y-axis has coordinates (1,2), where A' is obtained 
    by translating A 4 units left along the x-axis. -/
theorem symmetric_point_coordinates : 
  let A : ℝ × ℝ := (3, 2)
  let A' : ℝ × ℝ := (A.1 - 4, A.2)
  let symmetric_point : ℝ × ℝ := (-A'.1, A'.2)
  symmetric_point = (1, 2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l494_49493


namespace NUMINAMATH_CALUDE_factory_composition_diagram_l494_49474

/-- Represents different types of diagrams --/
inductive Diagram
  | ProgramFlowchart
  | ProcessFlow
  | KnowledgeStructure
  | OrganizationalStructure

/-- Represents the purpose of a diagram --/
inductive DiagramPurpose
  | RepresentComposition
  | RepresentProcedures
  | RepresentKnowledge

/-- Associates a diagram type with its primary purpose --/
def diagramPurpose (d : Diagram) : DiagramPurpose :=
  match d with
  | Diagram.ProgramFlowchart => DiagramPurpose.RepresentProcedures
  | Diagram.ProcessFlow => DiagramPurpose.RepresentProcedures
  | Diagram.KnowledgeStructure => DiagramPurpose.RepresentKnowledge
  | Diagram.OrganizationalStructure => DiagramPurpose.RepresentComposition

/-- The theorem stating that the Organizational Structure Diagram 
    is used to represent the composition of a factory --/
theorem factory_composition_diagram :
  diagramPurpose Diagram.OrganizationalStructure = DiagramPurpose.RepresentComposition :=
by sorry


end NUMINAMATH_CALUDE_factory_composition_diagram_l494_49474


namespace NUMINAMATH_CALUDE_sunTzu_nests_count_l494_49476

/-- Geometric sequence with first term a and common ratio r -/
def geometricSeq (a : ℕ) (r : ℕ) : ℕ → ℕ := fun n => a * r ^ (n - 1)

/-- The number of nests in Sun Tzu's Arithmetic problem -/
def sunTzuNests : ℕ := geometricSeq 9 9 4

theorem sunTzu_nests_count : sunTzuNests = 6561 := by
  sorry

end NUMINAMATH_CALUDE_sunTzu_nests_count_l494_49476


namespace NUMINAMATH_CALUDE_percentage_problem_l494_49469

theorem percentage_problem (P : ℝ) :
  (0.15 * 0.30 * (P / 100) * 4800 = 108) → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l494_49469


namespace NUMINAMATH_CALUDE_cubic_function_properties_l494_49487

/-- A cubic function with a maximum at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2

/-- The derivative of f -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem cubic_function_properties (a b : ℝ) :
  f a b 1 = 3 ∧ f_deriv a b 1 = 0 →
  a = -6 ∧ b = 9 ∧ ∀ x, f a b x ≥ 0 ∧ (∃ x₀, f a b x₀ = 0) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l494_49487


namespace NUMINAMATH_CALUDE_A_power_93_l494_49456

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

theorem A_power_93 : A ^ 93 = A := by sorry

end NUMINAMATH_CALUDE_A_power_93_l494_49456


namespace NUMINAMATH_CALUDE_partnership_investment_l494_49429

/-- Represents a partnership with three partners -/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  duration : ℝ
  a_share : ℝ
  b_share : ℝ

/-- Theorem stating the conditions and the result to be proven -/
theorem partnership_investment (p : Partnership)
  (ha : p.a_investment = 11000)
  (hc : p.c_investment = 23000)
  (hd : p.duration = 8)
  (hsa : p.a_share = 2431)
  (hsb : p.b_share = 3315) :
  p.b_investment = 15000 := by
  sorry


end NUMINAMATH_CALUDE_partnership_investment_l494_49429


namespace NUMINAMATH_CALUDE_cosine_amplitude_l494_49407

/-- Given a cosine function y = a * cos(bx) where a > 0 and b > 0,
    if the maximum value is 3 and the minimum value is -3, then a = 3 -/
theorem cosine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, a * Real.cos (b * x) ≤ 3) 
  (hmin : ∀ x, a * Real.cos (b * x) ≥ -3)
  (hreach_max : ∃ x, a * Real.cos (b * x) = 3)
  (hreach_min : ∃ x, a * Real.cos (b * x) = -3) : 
  a = 3 := by sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l494_49407


namespace NUMINAMATH_CALUDE_triangle_properties_l494_49490

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  b = 2 * Real.sqrt 3 →
  (Real.cos B) / (Real.cos C) = b / (2 * a - c) →
  1 / (Real.tan A) + 1 / (Real.tan B) = (Real.sin C) / (Real.sqrt 3 * Real.sin A * Real.cos B) →
  4 * Real.sqrt 3 * S + 3 * (b^2 - a^2) = 3 * c^2 →
  S = Real.sqrt 3 / 3 ∧
  (0 < A ∧ A < Real.pi / 2 ∧ 0 < B ∧ B < Real.pi / 2 ∧ 0 < C ∧ C < Real.pi / 2 →
    (Real.sqrt 3 + 1) / 2 < (b + c) / a ∧ (b + c) / a < Real.sqrt 3 + 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l494_49490


namespace NUMINAMATH_CALUDE_exponential_function_values_l494_49435

-- Define the exponential function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_function_values 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : f a 3 = 8) : 
  f a 4 = 16 ∧ f a (-4) = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_values_l494_49435


namespace NUMINAMATH_CALUDE_not_coplanar_implies_no_intersection_l494_49412

-- Define a point in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define a line in 3D space as two points
def Line3D := Point3D × Point3D

-- Define a function to check if four points are coplanar
def are_coplanar (E F G H : Point3D) : Prop := sorry

-- Define a function to check if two lines intersect
def lines_intersect (l1 l2 : Line3D) : Prop := sorry

theorem not_coplanar_implies_no_intersection 
  (E F G H : Point3D) : 
  ¬(are_coplanar E F G H) → ¬(lines_intersect (E, F) (G, H)) := by
  sorry

end NUMINAMATH_CALUDE_not_coplanar_implies_no_intersection_l494_49412


namespace NUMINAMATH_CALUDE_phones_to_repair_per_person_l494_49434

theorem phones_to_repair_per_person
  (initial_phones : ℕ)
  (repaired_phones : ℕ)
  (new_phones : ℕ)
  (h1 : initial_phones = 15)
  (h2 : repaired_phones = 3)
  (h3 : new_phones = 6)
  (h4 : repaired_phones ≤ initial_phones) :
  (initial_phones - repaired_phones + new_phones) / 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_phones_to_repair_per_person_l494_49434


namespace NUMINAMATH_CALUDE_engineering_exam_pass_percentage_l494_49446

theorem engineering_exam_pass_percentage
  (total_male : ℕ)
  (total_female : ℕ)
  (male_eng_percent : ℚ)
  (female_eng_percent : ℚ)
  (male_pass_percent : ℚ)
  (female_pass_percent : ℚ)
  (h1 : total_male = 120)
  (h2 : total_female = 100)
  (h3 : male_eng_percent = 25 / 100)
  (h4 : female_eng_percent = 20 / 100)
  (h5 : male_pass_percent = 20 / 100)
  (h6 : female_pass_percent = 25 / 100)
  : (↑(Nat.floor ((male_eng_percent * male_pass_percent * total_male + female_eng_percent * female_pass_percent * total_female) / (male_eng_percent * total_male + female_eng_percent * total_female) * 100)) : ℚ) = 22 := by
  sorry

end NUMINAMATH_CALUDE_engineering_exam_pass_percentage_l494_49446


namespace NUMINAMATH_CALUDE_line_translation_coincidence_l494_49424

/-- 
Given a line y = kx + 2 in the Cartesian plane,
prove that if the line is translated upward by 3 units
and then rightward by 2 units, and the resulting line
coincides with the original line, then k = 3/2.
-/
theorem line_translation_coincidence (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 2 ↔ y = k * (x - 2) + 5) → k = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_line_translation_coincidence_l494_49424


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l494_49460

theorem complex_magnitude_one (n : ℕ) (a : ℝ) (z : ℂ)
  (h_n : n ≥ 2)
  (h_a : 0 < a ∧ a < (n + 1 : ℝ) / (n - 1 : ℝ))
  (h_z : z^(n+1) - a * z^n + a * z - 1 = 0) :
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l494_49460


namespace NUMINAMATH_CALUDE_initial_peanuts_count_l494_49491

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Mary adds to the box -/
def peanuts_added : ℕ := 2

/-- The total number of peanuts after Mary adds some -/
def total_peanuts : ℕ := 6

/-- Theorem stating that the initial number of peanuts is 4 -/
theorem initial_peanuts_count : initial_peanuts = 4 :=
  by sorry

end NUMINAMATH_CALUDE_initial_peanuts_count_l494_49491


namespace NUMINAMATH_CALUDE_max_product_of_three_l494_49443

def S : Finset Int := {-9, -7, -3, 1, 4, 6}

theorem max_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S)
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  a * b * c ≤ 378 ∧ ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 378 :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_three_l494_49443


namespace NUMINAMATH_CALUDE_count_zeros_100_to_50_l494_49444

/-- The number of zeros following the numeral one in the expanded form of 100^50 -/
def zeros_after_one_in_100_to_50 : ℕ := 100

/-- Theorem stating that the number of zeros following the numeral one
    in the expanded form of 100^50 is equal to 100 -/
theorem count_zeros_100_to_50 :
  zeros_after_one_in_100_to_50 = 100 := by sorry

end NUMINAMATH_CALUDE_count_zeros_100_to_50_l494_49444


namespace NUMINAMATH_CALUDE_final_values_l494_49433

def sequence_operations (a b c : ℕ) : ℕ × ℕ × ℕ :=
  let a' := b
  let b' := c
  let c' := a'
  (a', b', c')

theorem final_values :
  sequence_operations 10 20 30 = (20, 30, 20) := by sorry

end NUMINAMATH_CALUDE_final_values_l494_49433


namespace NUMINAMATH_CALUDE_divide_by_approximate_700_l494_49483

-- Define the approximation tolerance
def tolerance : ℝ := 0.001

-- Define the condition from the problem
def condition (x : ℝ) : Prop :=
  abs (49 / x - 700) < tolerance

-- State the theorem
theorem divide_by_approximate_700 :
  ∃ x : ℝ, condition x ∧ abs (x - 0.07) < tolerance :=
sorry

end NUMINAMATH_CALUDE_divide_by_approximate_700_l494_49483


namespace NUMINAMATH_CALUDE_largest_digit_for_two_digit_quotient_l494_49452

theorem largest_digit_for_two_digit_quotient :
  ∀ n : ℕ, n ≤ 4 ∧ (n * 100 + 5) / 5 < 100 ∧
  ∀ m : ℕ, m > n → (m * 100 + 5) / 5 ≥ 100 →
  4 = n :=
sorry

end NUMINAMATH_CALUDE_largest_digit_for_two_digit_quotient_l494_49452


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_18_l494_49416

theorem factorization_of_2m_squared_minus_18 (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_18_l494_49416


namespace NUMINAMATH_CALUDE_a_equals_four_l494_49449

theorem a_equals_four (a : ℝ) (h : a * 2 * (2^3) = 2^6) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_four_l494_49449


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l494_49442

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4) / (1 - r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l494_49442


namespace NUMINAMATH_CALUDE_decimal_0_03_is_3_percent_l494_49422

/-- Converts a decimal fraction to a percentage -/
def decimal_to_percentage (d : ℝ) : ℝ := d * 100

/-- The decimal fraction we're working with -/
def given_decimal : ℝ := 0.03

/-- Theorem: The percentage equivalent of 0.03 is 3% -/
theorem decimal_0_03_is_3_percent :
  decimal_to_percentage given_decimal = 3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_0_03_is_3_percent_l494_49422


namespace NUMINAMATH_CALUDE_sum_of_powers_implies_sum_power_l494_49472

theorem sum_of_powers_implies_sum_power (a b : ℝ) : 
  a^2009 + b^2009 = 0 → (a + b)^2009 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_implies_sum_power_l494_49472


namespace NUMINAMATH_CALUDE_square_area_on_xz_l494_49400

/-- A right-angled triangle with squares on each side -/
structure RightTriangleWithSquares where
  /-- Length of side XZ -/
  x : ℝ
  /-- The sum of areas of squares on all sides is 500 -/
  area_sum : x^2 / 2 + x^2 + 5 * x^2 / 4 = 500

/-- The area of the square on side XZ is 2000/11 -/
theorem square_area_on_xz (t : RightTriangleWithSquares) : t.x^2 = 2000 / 11 := by
  sorry

#check square_area_on_xz

end NUMINAMATH_CALUDE_square_area_on_xz_l494_49400


namespace NUMINAMATH_CALUDE_a_range_l494_49430

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3/4| ≤ 1/4
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- Define the range of x for p
def p_range (x : ℝ) : Prop := 1/2 ≤ x ∧ x ≤ 1

-- Define the range of x for q
def q_range (x a : ℝ) : Prop := a ≤ x ∧ x ≤ a + 1

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p_range x → q_range x a) ∧
  ¬(∀ x, q_range x a → p_range x)

-- State the theorem
theorem a_range :
  ∀ a : ℝ, sufficient_not_necessary a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_a_range_l494_49430


namespace NUMINAMATH_CALUDE_binary_to_base5_conversion_l494_49462

-- Define the binary number
def binary_num : ℕ := 0b101101

-- Define the base-5 number
def base5_num : ℕ := 140

-- Theorem statement
theorem binary_to_base5_conversion :
  (binary_num : ℕ).digits 5 = base5_num.digits 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_base5_conversion_l494_49462


namespace NUMINAMATH_CALUDE_y_value_proof_l494_49448

theorem y_value_proof (y : ℝ) (h : 8 / y^3 = y / 32) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l494_49448


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l494_49450

theorem algebraic_expression_simplification (a : ℝ) (h : a = Real.sqrt 2) :
  a / (a^2 - 2*a + 1) / (1 + 1 / (a - 1)) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l494_49450


namespace NUMINAMATH_CALUDE_alberts_to_bettys_age_ratio_l494_49418

/-- Proves that the ratio of Albert's age to Betty's age is 4:1 given the specified conditions -/
theorem alberts_to_bettys_age_ratio :
  ∀ (albert_age mary_age betty_age : ℕ),
    albert_age = 2 * mary_age →
    mary_age = albert_age - 14 →
    betty_age = 7 →
    (albert_age : ℚ) / betty_age = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_alberts_to_bettys_age_ratio_l494_49418


namespace NUMINAMATH_CALUDE_sum_of_squares_over_factorial_l494_49494

theorem sum_of_squares_over_factorial : (1^2 + 2^2 + 3^2 + 4^2) / (1 * 2 * 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_over_factorial_l494_49494


namespace NUMINAMATH_CALUDE_class_size_l494_49453

theorem class_size (n : ℕ) (h1 : n < 50) (h2 : n % 8 = 5) (h3 : n % 6 = 3) : n = 21 ∨ n = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l494_49453


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l494_49463

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 9^n - 1 → 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 7 ∧ q ≠ 7 ∧ 
    x = 2^(Nat.log 2 x) * 7 * p * q) →
  x = 728 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l494_49463


namespace NUMINAMATH_CALUDE_cream_cake_problem_l494_49401

def creamPerCake : ℕ := 75
def totalCream : ℕ := 500
def totalCakes : ℕ := 50
def cakesPerBox : ℕ := 6

theorem cream_cake_problem :
  (totalCream / creamPerCake : ℕ) = 6 ∧
  (totalCakes + cakesPerBox - 1) / cakesPerBox = 9 := by
  sorry

end NUMINAMATH_CALUDE_cream_cake_problem_l494_49401


namespace NUMINAMATH_CALUDE_proportion_problem_l494_49408

theorem proportion_problem (y : ℝ) : (0.75 / 2 = 3 / y) → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l494_49408


namespace NUMINAMATH_CALUDE_oliver_stickers_l494_49471

theorem oliver_stickers (initial_stickers : ℕ) (remaining_stickers : ℕ) 
  (h1 : initial_stickers = 135)
  (h2 : remaining_stickers = 54)
  (h3 : ∃ x : ℚ, 0 ≤ x ∧ x < 1 ∧ 
    remaining_stickers = initial_stickers - (x * initial_stickers).floor - 
    ((2/5 : ℚ) * (initial_stickers - (x * initial_stickers).floor)).floor) :
  ∃ x : ℚ, x = 1/3 ∧ 
    remaining_stickers = initial_stickers - (x * initial_stickers).floor - 
    ((2/5 : ℚ) * (initial_stickers - (x * initial_stickers).floor)).floor :=
sorry

end NUMINAMATH_CALUDE_oliver_stickers_l494_49471


namespace NUMINAMATH_CALUDE_square_area_is_400_l494_49484

/-- A square is cut into five rectangles of equal area, with one rectangle having a width of 5. -/
structure CutSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The width of the rectangle with known width -/
  known_width : ℝ
  /-- The area of each rectangle -/
  rectangle_area : ℝ
  /-- The known width is 5 -/
  known_width_is_5 : known_width = 5
  /-- The square is divided into 5 rectangles of equal area -/
  five_equal_rectangles : side * side = 5 * rectangle_area

/-- The area of the square is 400 -/
theorem square_area_is_400 (s : CutSquare) : s.side * s.side = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_400_l494_49484


namespace NUMINAMATH_CALUDE_triangle_property_l494_49492

open Real

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.c / (t.b - t.a) = (sin t.A + sin t.B) / (sin t.A + sin t.C))
  (h2 : t.b = 2 * sqrt 2)
  (h3 : t.a + t.c = 3) :
  t.B = 2 * π / 3 ∧ 
  (1/2) * t.a * t.c * sin t.B = sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_property_l494_49492


namespace NUMINAMATH_CALUDE_probability_two_sunny_days_l494_49498

/-- The probability of exactly k sunny days in n days, given the probability of a sunny day --/
def probability_k_sunny_days (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The number of days in the holiday weekend --/
def num_days : ℕ := 5

/-- The probability of a sunny day --/
def prob_sunny : ℝ := 0.3

/-- The number of desired sunny days --/
def desired_sunny_days : ℕ := 2

theorem probability_two_sunny_days :
  probability_k_sunny_days num_days desired_sunny_days prob_sunny = 0.3087 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_sunny_days_l494_49498


namespace NUMINAMATH_CALUDE_range_of_a_l494_49431

-- Define the conditions
def p (x : ℝ) : Prop := |x - 2| < 3
def q (x a : ℝ) : Prop := 0 < x ∧ x < a

-- State the theorem
theorem range_of_a :
  (∀ x a : ℝ, q x a → p x) ∧ 
  (∃ x a : ℝ, p x ∧ ¬(q x a)) →
  ∀ a : ℝ, (∃ x : ℝ, q x a) ↔ (0 < a ∧ a ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l494_49431


namespace NUMINAMATH_CALUDE_louise_pictures_l494_49441

def total_pictures (vertical horizontal haphazard : ℕ) : ℕ :=
  vertical + horizontal + haphazard

theorem louise_pictures : 
  ∀ (vertical horizontal haphazard : ℕ),
    vertical = 10 →
    horizontal = vertical / 2 →
    haphazard = 5 →
    total_pictures vertical horizontal haphazard = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_louise_pictures_l494_49441


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l494_49479

/-- 
If the cost price of 50 articles is equal to the selling price of 15 articles, 
then the gain percent is 233.33%.
-/
theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 15 * S) : 
  (S - C) / C * 100 = 233.33 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l494_49479


namespace NUMINAMATH_CALUDE_line_translation_proof_l494_49451

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given distance -/
def translateLine (l : Line) (distance : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + distance }

theorem line_translation_proof :
  let originalLine : Line := { slope := 2, intercept := -3 }
  let translatedLine := translateLine originalLine 6
  translatedLine = { slope := 2, intercept := 3 } := by
sorry

end NUMINAMATH_CALUDE_line_translation_proof_l494_49451


namespace NUMINAMATH_CALUDE_barefoot_kids_l494_49432

theorem barefoot_kids (total : ℕ) (socks : ℕ) (shoes : ℕ) (both : ℕ) 
  (h1 : total = 35)
  (h2 : socks = 18)
  (h3 : shoes = 15)
  (h4 : both = 8) :
  total - (socks + shoes - both) = 10 := by
sorry

end NUMINAMATH_CALUDE_barefoot_kids_l494_49432


namespace NUMINAMATH_CALUDE_two_digit_reverse_diff_64_l494_49445

/-- Given a two-digit number, return the number formed by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A two-digit number is between 10 and 99, inclusive -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_reverse_diff_64 (N : ℕ) :
  is_two_digit N →
  N - reverse_digits N = 64 →
  N = 90 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_reverse_diff_64_l494_49445


namespace NUMINAMATH_CALUDE_supermarket_spending_l494_49466

theorem supermarket_spending (F : ℚ) : 
  (∃ (M : ℚ), 
    M = 150 ∧ 
    F * M + (1/3) * M + (1/10) * M + 10 = M) →
  F = 1/2 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l494_49466


namespace NUMINAMATH_CALUDE_f_max_min_l494_49470

-- Define the function
def f (x : ℝ) : ℝ := |x^2 - x| + |x + 1|

-- State the theorem
theorem f_max_min :
  ∃ (max min : ℝ),
    (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x ≤ max) ∧
    (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f x = max) ∧
    (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → min ≤ f x) ∧
    (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f x = min) ∧
    max = 7 ∧ min = 1 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_l494_49470


namespace NUMINAMATH_CALUDE_outer_circle_radius_l494_49465

theorem outer_circle_radius (r : ℝ) : 
  r > 0 ∧ 
  (π * (1.2 * r)^2 - π * 3^2) = (π * r^2 - π * 6^2) * 2.109375 → 
  r = 10 := by
  sorry

end NUMINAMATH_CALUDE_outer_circle_radius_l494_49465


namespace NUMINAMATH_CALUDE_greene_nursery_white_roses_l494_49410

/-- The number of white roses at Greene Nursery -/
def white_roses : ℕ := 6284 - (1491 + 3025)

/-- Theorem stating the number of white roses at Greene Nursery -/
theorem greene_nursery_white_roses :
  white_roses = 1768 :=
by sorry

end NUMINAMATH_CALUDE_greene_nursery_white_roses_l494_49410


namespace NUMINAMATH_CALUDE_circular_class_properties_l494_49467

/-- Represents a circular seating arrangement of students -/
structure CircularClass where
  totalStudents : ℕ
  boyOppositePositions : (ℕ × ℕ)
  everyOtherIsBoy : Bool

/-- Calculates the number of boys in the class -/
def numberOfBoys (c : CircularClass) : ℕ :=
  c.totalStudents / 2

/-- Theorem stating the properties of the circular class -/
theorem circular_class_properties (c : CircularClass) 
  (h1 : c.boyOppositePositions = (10, 40))
  (h2 : c.everyOtherIsBoy = true) :
  c.totalStudents = 60 ∧ numberOfBoys c = 30 := by
  sorry

#check circular_class_properties

end NUMINAMATH_CALUDE_circular_class_properties_l494_49467


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l494_49478

theorem diophantine_equation_solutions
  (a b c : ℤ) 
  (d : ℕ) 
  (h_d : d = Int.gcd a b) 
  (h_div : c % d = 0) 
  (x₀ y₀ : ℤ) 
  (h_particular : a * x₀ + b * y₀ = c) :
  ∀ (x y : ℤ), 
    (a * x + b * y = c) ↔ 
    (∃ (k : ℤ), x = x₀ + k * (b / d) ∧ y = y₀ - k * (a / d)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l494_49478


namespace NUMINAMATH_CALUDE_modified_lucas_units_digit_l494_49499

/-- Modified Lucas sequence -/
def L' : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | n + 2 => 2 * L' (n + 1) + L' n

/-- Function to get the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem stating that the units digit of L'_{L'_{20}} is d -/
theorem modified_lucas_units_digit :
  ∃ d : ℕ, d < 10 ∧ unitsDigit (L' (L' 20)) = d :=
sorry

end NUMINAMATH_CALUDE_modified_lucas_units_digit_l494_49499


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l494_49454

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = 0 ∧
  (a + Real.sqrt b) * (a - Real.sqrt b) = 25 →
  a + b = -25 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l494_49454


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l494_49402

theorem polygon_angle_sum (n : ℕ) : 
  (n ≥ 3) →
  (180 * (n - 2) = 3 * 360) →
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l494_49402


namespace NUMINAMATH_CALUDE_five_fold_f_of_one_l494_49417

def f (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem five_fold_f_of_one : f (f (f (f (f 1)))) = 4687 := by
  sorry

end NUMINAMATH_CALUDE_five_fold_f_of_one_l494_49417


namespace NUMINAMATH_CALUDE_quadratic_inequality_result_l494_49477

theorem quadratic_inequality_result (y : ℝ) (h : y^2 - 7*y + 12 < 0) :
  44 < y^2 + 7*y + 14 ∧ y^2 + 7*y + 14 < 58 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_result_l494_49477


namespace NUMINAMATH_CALUDE_runner_speed_increase_l494_49485

/-- Represents a runner's speed and time improvement factors -/
structure Runner where
  initialSpeed : ℝ
  speedIncrease1 : ℝ
  speedIncrease2 : ℝ
  timeFactor1 : ℝ

/-- Theorem: If increasing speed by speedIncrease1 makes the runner timeFactor1 times faster,
    then increasing speed by speedIncrease2 will make them speedRatio times faster -/
theorem runner_speed_increase (runner : Runner)
  (h1 : runner.speedIncrease1 = 2)
  (h2 : runner.timeFactor1 = 2.5)
  (h3 : runner.speedIncrease2 = 4)
  (h4 : runner.initialSpeed > 0)
  : (runner.initialSpeed + runner.speedIncrease2) / runner.initialSpeed = 4 := by
  sorry

end NUMINAMATH_CALUDE_runner_speed_increase_l494_49485


namespace NUMINAMATH_CALUDE_tangent_and_decreasing_interval_l494_49495

-- Define the function f
def f (m n : ℝ) (x : ℝ) : ℝ := m * x^3 + n * x^2

-- Define the derivative of f
def f_derivative (m n : ℝ) (x : ℝ) : ℝ := 3 * m * x^2 + 2 * n * x

-- Theorem statement
theorem tangent_and_decreasing_interval 
  (m n : ℝ) 
  (h1 : f m n (-1) = 2)
  (h2 : f_derivative m n (-1) = -3)
  (h3 : ∀ t : ℝ, ∀ x ∈ Set.Icc t (t + 1), 
        f_derivative m n x ≤ 0 → 
        -2 ≤ t ∧ t ≤ -1) :
  ∀ t : ℝ, (∀ x ∈ Set.Icc t (t + 1), f_derivative m n x ≤ 0) → 
    t ∈ Set.Icc (-2) (-1) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_decreasing_interval_l494_49495


namespace NUMINAMATH_CALUDE_geometric_series_sum_l494_49420

theorem geometric_series_sum : 
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 5
  let series_sum : ℚ := (a * (1 - r^n)) / (1 - r)
  series_sum = 341/1024 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l494_49420
