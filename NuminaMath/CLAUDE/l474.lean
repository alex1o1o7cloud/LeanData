import Mathlib

namespace sphere_surface_area_l474_47406

theorem sphere_surface_area (volume : ℝ) (h : volume = 72 * Real.pi) :
  let r := (3 * volume / (4 * Real.pi)) ^ (1/3)
  4 * Real.pi * r^2 = 36 * 2^(2/3) * Real.pi := by
  sorry

end sphere_surface_area_l474_47406


namespace average_first_21_multiples_of_5_l474_47456

theorem average_first_21_multiples_of_5 : 
  let n : ℕ := 21
  let multiples : List ℕ := List.range n |>.map (fun i => (i + 1) * 5)
  (multiples.sum : ℚ) / n = 55 := by
  sorry

end average_first_21_multiples_of_5_l474_47456


namespace multiple_of_p_l474_47453

theorem multiple_of_p (p q : ℚ) (k : ℚ) : 
  p / q = 3 / 5 → kp + q = 11 → k = 2 := by sorry

end multiple_of_p_l474_47453


namespace complex_fraction_equality_l474_47479

theorem complex_fraction_equality : Complex.I ^ 2 = -1 → (2 : ℂ) / (1 - Complex.I) = 1 + Complex.I := by
  sorry

end complex_fraction_equality_l474_47479


namespace clock_hand_positions_l474_47487

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hours : ℕ)
  (minutes : ℕ)

/-- The number of times the hour and minute hands coincide in 24 hours -/
def coincidences : ℕ := 22

/-- The number of times the hour and minute hands form a straight angle in 24 hours -/
def straight_angles : ℕ := 24

/-- The number of times the hour and minute hands form a right angle in 24 hours -/
def right_angles : ℕ := 48

/-- The number of full rotations the minute hand makes in 24 hours -/
def minute_rotations : ℕ := 24

/-- The number of full rotations the hour hand makes in 24 hours -/
def hour_rotations : ℕ := 2

theorem clock_hand_positions (c : Clock) :
  coincidences = 22 ∧
  straight_angles = 24 ∧
  right_angles = 48 :=
sorry

end clock_hand_positions_l474_47487


namespace counterexample_exists_l474_47426

theorem counterexample_exists : ∃ n : ℕ, Nat.Prime n ∧ ¬(Nat.Prime (n - 2)) :=
sorry

end counterexample_exists_l474_47426


namespace sqrt_equation_solution_l474_47466

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt (4 * x + 6) / Real.sqrt (8 * x + 2) = 2 / Real.sqrt 3) → x = 1 / 2 := by
  sorry

end sqrt_equation_solution_l474_47466


namespace perpendicular_line_properties_l474_47412

/-- Given a line l₁ and a point A, this theorem proves properties of the perpendicular line l₂ passing through A -/
theorem perpendicular_line_properties (x y : ℝ) :
  let l₁ : ℝ → ℝ → Prop := λ x y => 3 * x + 4 * y - 1 = 0
  let A : ℝ × ℝ := (3, 0)
  let l₂ : ℝ → ℝ → Prop := λ x y => 4 * x - 3 * y - 12 = 0
  -- l₂ passes through A
  (l₂ A.1 A.2) →
  -- l₂ is perpendicular to l₁
  (∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₁ x₂ y₂ → l₂ x₁ y₁ → l₂ x₂ y₂ → 
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (3) + (y₂ - y₁) * (4)) * ((x₂ - x₁) * (4) + (y₂ - y₁) * (-3)) = 0) →
  -- The equation of l₂ is correct
  (∀ x y, l₂ x y ↔ 4 * x - 3 * y - 12 = 0) ∧
  -- The area of the triangle is 6
  (let x_intercept := 3
   let y_intercept := 4
   (1 / 2 : ℝ) * x_intercept * y_intercept = 6) := by
  sorry


end perpendicular_line_properties_l474_47412


namespace optimal_triangle_height_l474_47493

/-- Given two parallel lines with distance b between them, and a segment of length a on one of the lines,
    the sum of areas of two triangles formed by connecting a point on the line segment to a point on the other parallel line
    is minimized when the height of one triangle is b√2/2. -/
theorem optimal_triangle_height (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let height := b * Real.sqrt 2 / 2
  let area (h : ℝ) := a * h / 2 + a * (b - h) ^ 2 / (2 * h)
  ∀ h, 0 < h ∧ h < b → area height ≤ area h := by
sorry

end optimal_triangle_height_l474_47493


namespace special_function_at_1001_l474_47489

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧
  (∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 3))

/-- The main theorem stating that f(1001) = 3 for any function satisfying the conditions -/
theorem special_function_at_1001 (f : ℝ → ℝ) (h : special_function f) : f 1001 = 3 := by
  sorry

end special_function_at_1001_l474_47489


namespace max_gcd_value_l474_47439

theorem max_gcd_value (m : ℕ+) : 
  (Nat.gcd (15 * m.val + 4) (14 * m.val + 3) ≤ 11) ∧ 
  (∃ m : ℕ+, Nat.gcd (15 * m.val + 4) (14 * m.val + 3) = 11) := by
  sorry

end max_gcd_value_l474_47439


namespace cool_drink_solution_l474_47445

/-- Proves that 12 liters of water were added to achieve the given conditions -/
theorem cool_drink_solution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (jasmine_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 80 →
  initial_concentration = 0.1 →
  jasmine_added = 8 →
  final_concentration = 0.16 →
  ∃ (water_added : ℝ),
    water_added = 12 ∧
    (initial_volume * initial_concentration + jasmine_added) / 
    (initial_volume + jasmine_added + water_added) = final_concentration :=
by
  sorry


end cool_drink_solution_l474_47445


namespace three_prime_divisors_of_nine_power_minus_one_l474_47435

theorem three_prime_divisors_of_nine_power_minus_one (n : ℕ) (x : ℕ) 
  (h1 : x = 9^n - 1)
  (h2 : (Nat.factors x).toFinset.card = 3)
  (h3 : 7 ∈ Nat.factors x) :
  x = 728 := by sorry

end three_prime_divisors_of_nine_power_minus_one_l474_47435


namespace complement_of_A_is_zero_l474_47459

def A : Set ℤ := {x | |x| ≥ 1}

theorem complement_of_A_is_zero : 
  (Set.univ : Set ℤ) \ A = {0} := by sorry

end complement_of_A_is_zero_l474_47459


namespace integer_roots_of_polynomial_l474_47427

theorem integer_roots_of_polynomial (a : ℤ) : 
  a = -4 →
  (∀ x : ℤ, x^4 - 16*x^3 + (81-2*a)*x^2 + (16*a-142)*x + a^2 - 21*a + 68 = 0 ↔ 
    x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 7) := by
  sorry

#check integer_roots_of_polynomial

end integer_roots_of_polynomial_l474_47427


namespace probability_one_success_out_of_three_l474_47407

/-- The probability of passing a single computer test -/
def p : ℚ := 1 / 3

/-- The number of tests taken -/
def n : ℕ := 3

/-- The number of successful attempts -/
def k : ℕ := 1

/-- Binomial coefficient function -/
def binomial_coeff (n k : ℕ) : ℚ := (n.choose k : ℚ)

/-- The probability of passing exactly k tests out of n attempts -/
def probability_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coeff n k * p^k * (1 - p)^(n - k)

theorem probability_one_success_out_of_three :
  probability_k_successes n k p = 4 / 9 := by sorry

end probability_one_success_out_of_three_l474_47407


namespace five_points_in_unit_triangle_close_pair_l474_47437

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  is_positive : side_length > 0

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is inside the triangle
def is_inside_triangle (t : EquilateralTriangle) (p : Point) : Prop :=
  sorry -- Actual implementation would go here

-- Define a function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ :=
  sorry -- Actual implementation would go here

-- Theorem statement
theorem five_points_in_unit_triangle_close_pair 
  (t : EquilateralTriangle) 
  (h_side : t.side_length = 1) 
  (points : Fin 5 → Point) 
  (h_inside : ∀ i, is_inside_triangle t (points i)) :
  ∃ i j, i ≠ j ∧ distance (points i) (points j) < 0.5 :=
sorry

end five_points_in_unit_triangle_close_pair_l474_47437


namespace pancake_fundraiser_l474_47434

/-- The civic league's pancake breakfast fundraiser problem -/
theorem pancake_fundraiser
  (pancake_price : ℝ)
  (bacon_price : ℝ)
  (pancake_stacks : ℕ)
  (bacon_slices : ℕ)
  (h1 : pancake_price = 4)
  (h2 : bacon_price = 2)
  (h3 : pancake_stacks = 60)
  (h4 : bacon_slices = 90) :
  pancake_price * (pancake_stacks : ℝ) + bacon_price * (bacon_slices : ℝ) = 420 := by
  sorry

end pancake_fundraiser_l474_47434


namespace solve_inequality_system_simplify_expression_l474_47451

-- Problem 1
theorem solve_inequality_system (x : ℝ) :
  (10 - 3 * x < -5 ∧ x / 3 ≥ 4 - (x - 2) / 2) ↔ x ≥ 6 := by sorry

-- Problem 2
theorem simplify_expression (a : ℝ) (h : a ≠ 0 ∧ a ≠ 1 ∧ a ≠ -1) :
  2 / (a + 1) - (a - 2) / (a^2 - 1) / (a * (a - 2) / (a^2 - 2*a + 1)) = 1 / a := by sorry

end solve_inequality_system_simplify_expression_l474_47451


namespace min_points_10th_game_l474_47415

def points_6_to_9 : List ℕ := [18, 15, 16, 19]

def total_points_6_to_9 : ℕ := points_6_to_9.sum

def average_greater_after_9_than_5 (first_5_total : ℕ) : Prop :=
  (first_5_total + total_points_6_to_9) / 9 > first_5_total / 5

def first_5_not_exceed_85 (first_5_total : ℕ) : Prop :=
  first_5_total ≤ 85

theorem min_points_10th_game (first_5_total : ℕ) 
  (h1 : average_greater_after_9_than_5 first_5_total)
  (h2 : first_5_not_exceed_85 first_5_total) :
  ∃ (points_10th : ℕ), 
    (first_5_total + total_points_6_to_9 + points_10th) / 10 > 17 ∧
    ∀ (x : ℕ), x < points_10th → 
      (first_5_total + total_points_6_to_9 + x) / 10 ≤ 17 :=
by sorry

end min_points_10th_game_l474_47415


namespace unique_product_sum_relation_l474_47424

theorem unique_product_sum_relation (a b c : ℕ+) :
  (a * b * c = 8 * (a + b + c)) ∧ 
  ((c = 2 * a + b) ∨ (b = 2 * a + c) ∨ (a = 2 * b + c)) →
  a * b * c = 136 :=
sorry

end unique_product_sum_relation_l474_47424


namespace bus_driver_max_regular_hours_l474_47430

/-- Proves that the maximum number of regular hours is 40 given the conditions --/
theorem bus_driver_max_regular_hours : 
  let regular_rate : ℚ := 16
  let overtime_rate : ℚ := regular_rate * (1 + 3/4)
  let total_compensation : ℚ := 1340
  let total_hours : ℕ := 65
  let max_regular_hours : ℕ := 40
  regular_rate * max_regular_hours + 
  overtime_rate * (total_hours - max_regular_hours) = total_compensation := by
sorry


end bus_driver_max_regular_hours_l474_47430


namespace polygon_25_sides_l474_47498

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 2

/-- Number of diagonals in a polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Number of triangles formed by choosing any three vertices of a polygon with n sides -/
def numTriangles (n : ℕ) : ℕ := n.choose 3

theorem polygon_25_sides (P : ConvexPolygon 25) : 
  numDiagonals 25 = 275 ∧ numTriangles 25 = 2300 := by
  sorry


end polygon_25_sides_l474_47498


namespace quadratic_inequality_always_positive_l474_47404

theorem quadratic_inequality_always_positive (m : ℝ) :
  (∀ x : ℝ, x^2 - (m - 4)*x - m + 7 > 0) ↔ m > -2 ∧ m < 6 :=
by sorry

end quadratic_inequality_always_positive_l474_47404


namespace remaining_flight_time_l474_47422

/-- Calculates the remaining time on a flight given the total flight duration and activity durations. -/
theorem remaining_flight_time (total_duration activity1 activity2 activity3 : ℕ) :
  total_duration = 360 ∧ 
  activity1 = 90 ∧ 
  activity2 = 40 ∧ 
  activity3 = 120 →
  total_duration - (activity1 + activity2 + activity3) = 110 := by
  sorry

#check remaining_flight_time

end remaining_flight_time_l474_47422


namespace neg_p_true_when_k_3_k_range_when_p_or_q_false_l474_47488

-- Define propositions p and q
def p (k : ℝ) : Prop := ∃ x : ℝ, k * x^2 + 1 ≤ 0
def q (k : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * k * x + 1 > 0

-- Theorem 1: When k = 3, ¬p is true
theorem neg_p_true_when_k_3 : ∀ x : ℝ, 3 * x^2 + 1 > 0 := by sorry

-- Theorem 2: The set of k for which both p and q are false
theorem k_range_when_p_or_q_false : 
  {k : ℝ | ¬(p k) ∧ ¬(q k)} = {k : ℝ | k ≤ -1 ∨ k ≥ 1} := by sorry

end neg_p_true_when_k_3_k_range_when_p_or_q_false_l474_47488


namespace probability_one_boy_one_girl_l474_47484

/-- The probability of selecting one boy and one girl from a group of 3 boys and 2 girls, when choosing 2 students out of 5 -/
theorem probability_one_boy_one_girl :
  let total_students : ℕ := 5
  let num_boys : ℕ := 3
  let num_girls : ℕ := 2
  let students_to_select : ℕ := 2
  let total_combinations := Nat.choose total_students students_to_select
  let favorable_outcomes := Nat.choose num_boys 1 * Nat.choose num_girls 1
  (favorable_outcomes : ℚ) / total_combinations = 3 / 5 :=
by sorry

end probability_one_boy_one_girl_l474_47484


namespace at_least_one_genuine_product_l474_47408

theorem at_least_one_genuine_product (total : Nat) (genuine : Nat) (defective : Nat) (selected : Nat) :
  total = genuine + defective →
  total = 12 →
  genuine = 10 →
  defective = 2 →
  selected = 3 →
  ∀ (selection : Finset (Fin total)),
    selection.card = selected →
    ∃ (i : Fin total), i ∈ selection ∧ i.val < genuine :=
by sorry

end at_least_one_genuine_product_l474_47408


namespace nephews_count_l474_47420

/-- The number of nephews Alden had 10 years ago -/
def alden_nephews_10_years_ago : ℕ := 50

/-- The number of nephews Alden has now -/
def alden_nephews_now : ℕ := 2 * alden_nephews_10_years_ago

/-- The number of additional nephews Vihaan has compared to Alden -/
def vihaan_additional_nephews : ℕ := 60

/-- The number of nephews Vihaan has -/
def vihaan_nephews : ℕ := alden_nephews_now + vihaan_additional_nephews

/-- The total number of nephews Alden and Vihaan have together -/
def total_nephews : ℕ := alden_nephews_now + vihaan_nephews

theorem nephews_count : total_nephews = 260 := by
  sorry

end nephews_count_l474_47420


namespace max_quad_area_l474_47452

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the lines
def Line1 (m x y : ℝ) : Prop := m*x - y + 1 = 0
def Line2 (m x y : ℝ) : Prop := x + m*y - m = 0

-- Define the quadrilateral area
def QuadArea (A B C D : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_quad_area :
  ∀ (m : ℝ) (A B C D : ℝ × ℝ),
    Circle A.1 A.2 ∧ Circle B.1 B.2 ∧ Circle C.1 C.2 ∧ Circle D.1 D.2 →
    Line1 m A.1 A.2 ∧ Line1 m C.1 C.2 →
    Line2 m B.1 B.2 ∧ Line2 m D.1 D.2 →
    QuadArea A B C D ≤ 7 :=
sorry

end max_quad_area_l474_47452


namespace coat_cost_price_l474_47400

theorem coat_cost_price (markup_percentage : ℝ) (final_price : ℝ) (cost_price : ℝ) : 
  markup_percentage = 0.25 →
  final_price = 275 →
  final_price = cost_price * (1 + markup_percentage) →
  cost_price = 220 := by
  sorry

end coat_cost_price_l474_47400


namespace residue_mod_29_l474_47463

theorem residue_mod_29 : ∃ k : ℤ, -1237 = 29 * k + 10 := by sorry

end residue_mod_29_l474_47463


namespace max_value_theorem_l474_47433

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  2*x + y ≤ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 2/y₀ = 1 ∧ 2*x₀ + y₀ = 8 :=
sorry

end max_value_theorem_l474_47433


namespace jade_tower_solution_l474_47464

/-- The number of Lego pieces in Jade's tower problem -/
def jade_tower_problem (width_per_level : ℕ) (num_levels : ℕ) (pieces_left : ℕ) : Prop :=
  width_per_level * num_levels + pieces_left = 100

/-- Theorem stating the solution to Jade's Lego tower problem -/
theorem jade_tower_solution : jade_tower_problem 7 11 23 := by
  sorry

end jade_tower_solution_l474_47464


namespace fraction_sum_inequality_l474_47494

theorem fraction_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end fraction_sum_inequality_l474_47494


namespace circle_line_intersection_chord_length_l474_47401

/-- Given a circle and a line, proves that the radius of the circle is 11 
    when the chord formed by their intersection has length 6 -/
theorem circle_line_intersection_chord_length (a : ℝ) : 
  (∃ x y : ℝ, (x + 2)^2 + (y - 2)^2 = a ∧ x + y + 2 = 0) →  -- Circle intersects line
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + 2)^2 + (y₁ - 2)^2 = a ∧ 
    (x₂ + 2)^2 + (y₂ - 2)^2 = a ∧ 
    x₁ + y₁ + 2 = 0 ∧ 
    x₂ + y₂ + 2 = 0 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) →  -- Chord length is 6
  a = 11 := by
sorry

end circle_line_intersection_chord_length_l474_47401


namespace quadratic_intersection_l474_47480

/-- Represents a quadratic function of the form y = x^2 + px + q -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  h : -2 * p + q = 2023

/-- The x-coordinate of the intersection point -/
def intersection_x : ℝ := -2

/-- The y-coordinate of the intersection point -/
def intersection_y : ℝ := 2027

/-- Theorem stating that all quadratic functions satisfying the condition intersect at a single point -/
theorem quadratic_intersection (f : QuadraticFunction) : 
  (intersection_x^2 + f.p * intersection_x + f.q) = intersection_y := by
  sorry

end quadratic_intersection_l474_47480


namespace xy_sum_over_five_l474_47475

theorem xy_sum_over_five (x y : ℝ) (h1 : x * y > 0) (h2 : 1 / x + 1 / y = 15) (h3 : 1 / (x * y) = 5) :
  (x + y) / 5 = 3 / 5 := by
  sorry

end xy_sum_over_five_l474_47475


namespace equal_numbers_theorem_l474_47447

theorem equal_numbers_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : a + b^2 + c^2 = c + a^2 + b^2) :
  a = b ∨ a = c ∨ b = c :=
sorry

end equal_numbers_theorem_l474_47447


namespace painted_cells_theorem_l474_47436

/-- Represents a rectangular grid with painted cells -/
structure PaintedGrid where
  rows : ℕ
  cols : ℕ
  painted_cells : ℕ

/-- Calculates the number of painted cells in a grid with the given painting pattern -/
def calculate_painted_cells (k l : ℕ) : ℕ :=
  (2 * k + 1) * (2 * l + 1) - k * l

/-- Theorem stating the possible numbers of painted cells given the conditions -/
theorem painted_cells_theorem :
  ∀ (grid : PaintedGrid),
  (∃ (k l : ℕ), 
    grid.rows = 2 * k + 1 ∧ 
    grid.cols = 2 * l + 1 ∧ 
    k * l = 74) →
  grid.painted_cells = 373 ∨ grid.painted_cells = 301 :=
by sorry

end painted_cells_theorem_l474_47436


namespace chocolate_profit_example_l474_47470

/-- Calculates the profit from selling chocolates given the following conditions:
  * Number of chocolate bars
  * Cost price per bar
  * Total selling price
  * Packaging cost per bar
-/
def chocolate_profit (num_bars : ℕ) (cost_price : ℚ) (total_selling_price : ℚ) (packaging_cost : ℚ) : ℚ :=
  total_selling_price - (num_bars * (cost_price + packaging_cost))

theorem chocolate_profit_example :
  chocolate_profit 5 5 90 2 = 55 := by
  sorry

end chocolate_profit_example_l474_47470


namespace factorization_equality_l474_47460

theorem factorization_equality (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) := by
  sorry

end factorization_equality_l474_47460


namespace total_birds_l474_47414

/-- The number of birds on an oak tree --/
def bird_count (bluebirds cardinals goldfinches sparrows robins swallows : ℕ) : Prop :=
  -- There are twice as many cardinals as bluebirds
  cardinals = 2 * bluebirds ∧
  -- The number of goldfinches is equal to the product of bluebirds and swallows
  goldfinches = bluebirds * swallows ∧
  -- The number of sparrows is half the sum of cardinals and goldfinches
  2 * sparrows = cardinals + goldfinches ∧
  -- The number of robins is 2 less than the quotient of bluebirds divided by swallows
  robins + 2 = bluebirds / swallows ∧
  -- There are 12 swallows
  swallows = 12 ∧
  -- The number of swallows is half as many as the number of bluebirds
  2 * swallows = bluebirds

theorem total_birds (bluebirds cardinals goldfinches sparrows robins swallows : ℕ) :
  bird_count bluebirds cardinals goldfinches sparrows robins swallows →
  bluebirds + cardinals + goldfinches + sparrows + robins + swallows = 540 :=
by sorry

end total_birds_l474_47414


namespace blocks_per_friend_l474_47474

theorem blocks_per_friend (total_blocks : ℕ) (num_friends : ℕ) (blocks_per_friend : ℕ) : 
  total_blocks = 28 → num_friends = 4 → blocks_per_friend = total_blocks / num_friends → blocks_per_friend = 7 := by
  sorry

end blocks_per_friend_l474_47474


namespace rectangle_area_l474_47465

theorem rectangle_area (short_side : ℝ) (perimeter : ℝ) 
  (h1 : short_side = 11) 
  (h2 : perimeter = 52) : 
  short_side * (perimeter / 2 - short_side) = 165 := by
  sorry

end rectangle_area_l474_47465


namespace mistaken_division_multiplication_l474_47461

/-- Given a number x and another number n, where x is mistakenly divided by n instead of being multiplied,
    and the percentage error in the result is 99%, prove that n = 10. -/
theorem mistaken_division_multiplication (x : ℝ) (n : ℝ) (h : x ≠ 0) :
  (x / n) / (x * n) = 1 / 100 → n = 10 := by
  sorry

end mistaken_division_multiplication_l474_47461


namespace xy_max_and_x2_y2_min_l474_47486

theorem xy_max_and_x2_y2_min (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 1) :
  (∃ (x0 y0 : ℝ), x0 > 0 ∧ y0 > 0 ∧ x0 + 2*y0 = 1 ∧ x0*y0 = 1/8 ∧ ∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 2*y' = 1 → x'*y' ≤ 1/8) ∧
  (x^2 + y^2 ≥ 1/5 ∧ ∃ (x1 y1 : ℝ), x1 > 0 ∧ y1 > 0 ∧ x1 + 2*y1 = 1 ∧ x1^2 + y1^2 = 1/5) :=
by sorry

end xy_max_and_x2_y2_min_l474_47486


namespace each_student_receives_six_apples_l474_47416

/-- The number of apples Anita has -/
def total_apples : ℕ := 360

/-- The number of students in Anita's class -/
def num_students : ℕ := 60

/-- The number of apples each student should receive -/
def apples_per_student : ℕ := total_apples / num_students

/-- Theorem stating that each student should receive 6 apples -/
theorem each_student_receives_six_apples : apples_per_student = 6 := by
  sorry

end each_student_receives_six_apples_l474_47416


namespace max_min_values_l474_47490

-- Define the conditions
def positive_xy (x y : ℝ) : Prop := x > 0 ∧ y > 0
def constraint (x y : ℝ) : Prop := 3 * x + 2 * y = 10

-- Define the theorem
theorem max_min_values (x y : ℝ) 
  (h1 : positive_xy x y) (h2 : constraint x y) : 
  (∃ (m : ℝ), m = Real.sqrt (3 * x) + Real.sqrt (2 * y) ∧ 
    m ≤ 2 * Real.sqrt 5 ∧ 
    ∀ (x' y' : ℝ), positive_xy x' y' → constraint x' y' → 
      Real.sqrt (3 * x') + Real.sqrt (2 * y') ≤ m) ∧
  (∃ (n : ℝ), n = 3 / x + 2 / y ∧ 
    n ≥ 5 / 2 ∧ 
    ∀ (x' y' : ℝ), positive_xy x' y' → constraint x' y' → 
      3 / x' + 2 / y' ≥ n) :=
by sorry

end max_min_values_l474_47490


namespace sequence_on_line_is_arithmetic_l474_47409

/-- Given a sequence {a_n} where (n, a_n) lies on the line y = 2x,
    prove that it is an arithmetic sequence with common difference 2 -/
theorem sequence_on_line_is_arithmetic (a : ℕ → ℝ) :
  (∀ n : ℕ, a n = 2 * n) →
  (∀ n : ℕ, a (n + 1) - a n = 2) :=
by sorry

end sequence_on_line_is_arithmetic_l474_47409


namespace quadratic_equation_root_l474_47472

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 5 * x + m^2 - 2 * m = 0) ∧ 
  (m * 0^2 + 5 * 0 + m^2 - 2 * m = 0) → 
  m = 2 := by
  sorry

end quadratic_equation_root_l474_47472


namespace speed_decrease_percentage_l474_47417

theorem speed_decrease_percentage (distance : ℝ) (fast_speed slow_speed : ℝ) 
  (h_distance_positive : distance > 0)
  (h_fast_speed_positive : fast_speed > 0)
  (h_slow_speed_positive : slow_speed > 0)
  (h_fast_time : distance / fast_speed = 40)
  (h_slow_time : distance / slow_speed = 50) :
  (fast_speed - slow_speed) / fast_speed = 1/5 := by
sorry

end speed_decrease_percentage_l474_47417


namespace smallest_n_divisibility_l474_47432

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  36 ∣ n^2 ∧ 1024 ∣ n^3 ∧ 
  ∀ (m : ℕ), m > 0 → 36 ∣ m^2 → 1024 ∣ m^3 → n ≤ m :=
by
  -- Proof goes here
  sorry

end smallest_n_divisibility_l474_47432


namespace bug_probability_l474_47421

/-- The probability of the bug being at its starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - Q n)

/-- The problem statement -/
theorem bug_probability : Q 8 = 547 / 2187 := by
  sorry

end bug_probability_l474_47421


namespace S_infinite_l474_47442

/-- The expression 2^(n^3+1) - 3^(n^2+1) + 5^(n+1) for positive integer n -/
def f (n : ℕ+) : ℤ := 2^(n.val^3+1) - 3^(n.val^2+1) + 5^(n.val+1)

/-- The set of prime numbers that divide f(n) for some positive integer n -/
def S : Set ℕ := {p : ℕ | Nat.Prime p ∧ ∃ (n : ℕ+), ∃ (k : ℤ), f n = k * p}

/-- Theorem stating that the set S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end S_infinite_l474_47442


namespace not_in_A_iff_less_than_neg_three_l474_47403

-- Define the set A
def A : Set ℝ := {x | x + 3 ≥ 0}

-- State the theorem
theorem not_in_A_iff_less_than_neg_three (a : ℝ) : a ∉ A ↔ a < -3 := by sorry

end not_in_A_iff_less_than_neg_three_l474_47403


namespace distinct_integers_with_swapped_digits_l474_47405

def has_2n_digits (x : ℕ) (n : ℕ) : Prop :=
  10^(2*n - 1) ≤ x ∧ x < 10^(2*n)

def first_n_digits (x : ℕ) (n : ℕ) : ℕ :=
  x / 10^n

def last_n_digits (x : ℕ) (n : ℕ) : ℕ :=
  x % 10^n

theorem distinct_integers_with_swapped_digits (n : ℕ) (a b : ℕ) :
  n > 0 →
  a ≠ b →
  a > 0 ∧ b > 0 →
  has_2n_digits a n →
  has_2n_digits b n →
  a ∣ b →
  first_n_digits a n = last_n_digits b n →
  last_n_digits a n = first_n_digits b n →
  ((a = 2442 ∧ b = 4224) ∨ (a = 3993 ∧ b = 9339)) :=
by sorry

end distinct_integers_with_swapped_digits_l474_47405


namespace max_value_fraction_l474_47402

theorem max_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ x' y', -3 ≤ x' ∧ x' ≤ -1 → 1 ≤ y' ∧ y' ≤ 3 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = -2 :=
sorry

end max_value_fraction_l474_47402


namespace running_preference_related_to_gender_certainty_running_preference_related_to_gender_l474_47471

/-- Represents the data from the survey about running preferences among university students. -/
structure RunningPreferenceSurvey where
  total_students : ℕ
  boys : ℕ
  boys_not_liking : ℕ
  girls_liking : ℕ

/-- Calculates the chi-square value for the given survey data. -/
def calculate_chi_square (survey : RunningPreferenceSurvey) : ℚ :=
  let girls := survey.total_students - survey.boys
  let boys_liking := survey.boys - survey.boys_not_liking
  let girls_not_liking := girls - survey.girls_liking
  let n := survey.total_students
  let a := boys_liking
  let b := survey.boys_not_liking
  let c := survey.girls_liking
  let d := girls_not_liking
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating that the chi-square value for the given survey data is greater than 6.635,
    indicating a 99% certainty that liking running is related to gender. -/
theorem running_preference_related_to_gender (survey : RunningPreferenceSurvey)
  (h1 : survey.total_students = 200)
  (h2 : survey.boys = 120)
  (h3 : survey.boys_not_liking = 50)
  (h4 : survey.girls_liking = 30) :
  calculate_chi_square survey > 6635 / 1000 :=
sorry

/-- Corollary stating that there is a 99% certainty that liking running is related to gender. -/
theorem certainty_running_preference_related_to_gender (survey : RunningPreferenceSurvey)
  (h1 : survey.total_students = 200)
  (h2 : survey.boys = 120)
  (h3 : survey.boys_not_liking = 50)
  (h4 : survey.girls_liking = 30) :
  ∃ (certainty : ℚ), certainty = 99 / 100 ∧
  calculate_chi_square survey > 6635 / 1000 :=
sorry

end running_preference_related_to_gender_certainty_running_preference_related_to_gender_l474_47471


namespace vowel_soup_combinations_l474_47468

/-- The number of vowels available -/
def num_vowels : ℕ := 5

/-- The length of the words to be formed -/
def word_length : ℕ := 6

/-- The number of times each vowel appears in the bowl -/
def vowel_count : ℕ := 7

/-- The total number of six-letter words that can be formed -/
def total_combinations : ℕ := num_vowels ^ word_length

theorem vowel_soup_combinations :
  total_combinations = 15625 :=
sorry

end vowel_soup_combinations_l474_47468


namespace first_part_second_part_l474_47457

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + 2*a*x - 3

-- Theorem for the first part of the problem
theorem first_part (a : ℝ) : f a (a + 1) - f a a = 9 → a = 2 := by
  sorry

-- Theorem for the second part of the problem
theorem second_part (a : ℝ) : 
  (∀ x, f a x ≥ -4) ∧ (∃ x, f a x = -4) ↔ (a = 1 ∨ a = -1) := by
  sorry

end first_part_second_part_l474_47457


namespace circumcircle_radius_isosceles_triangle_l474_47425

/-- Given a triangle with two sides of length a and a third side of length b,
    the radius of its circumcircle is a²/√(4a² - b²) -/
theorem circumcircle_radius_isosceles_triangle (a b : ℝ) (ha : a > 0) (hb : b > 0) (hc : b < 2*a) :
  ∃ R : ℝ, R = a^2 / Real.sqrt (4*a^2 - b^2) ∧ 
  R > 0 ∧ 
  R * Real.sqrt (4*a^2 - b^2) = a^2 :=
sorry

end circumcircle_radius_isosceles_triangle_l474_47425


namespace total_time_is_twelve_years_l474_47478

def years_to_get_in_shape : ℕ := 2
def years_to_learn_climbing (y : ℕ) : ℕ := 2 * y
def number_of_mountains : ℕ := 7
def months_per_mountain : ℕ := 5
def months_to_learn_diving : ℕ := 13
def years_of_diving : ℕ := 2

def total_time : ℕ :=
  years_to_get_in_shape +
  years_to_learn_climbing years_to_get_in_shape +
  (number_of_mountains * months_per_mountain + months_to_learn_diving) / 12 +
  years_of_diving

theorem total_time_is_twelve_years :
  total_time = 12 := by sorry

end total_time_is_twelve_years_l474_47478


namespace mode_is_97_l474_47449

/-- Represents a score in the stem-and-leaf plot -/
structure Score where
  stem : Nat
  leaf : Nat

/-- The list of all scores from the stem-and-leaf plot -/
def scores : List Score := [
  ⟨6, 5⟩, ⟨6, 5⟩,
  ⟨7, 1⟩, ⟨7, 3⟩, ⟨7, 3⟩, ⟨7, 6⟩,
  ⟨8, 0⟩, ⟨8, 0⟩, ⟨8, 4⟩, ⟨8, 4⟩, ⟨8, 8⟩, ⟨8, 8⟩, ⟨8, 8⟩,
  ⟨9, 2⟩, ⟨9, 2⟩, ⟨9, 5⟩, ⟨9, 7⟩, ⟨9, 7⟩, ⟨9, 7⟩, ⟨9, 7⟩,
  ⟨10, 1⟩, ⟨10, 1⟩, ⟨10, 1⟩, ⟨10, 4⟩, ⟨10, 6⟩,
  ⟨11, 0⟩, ⟨11, 0⟩, ⟨11, 0⟩
]

/-- Convert a Score to its numerical value -/
def scoreValue (s : Score) : Nat :=
  s.stem * 10 + s.leaf

/-- Count the occurrences of a value in the list of scores -/
def countOccurrences (value : Nat) : Nat :=
  (scores.filter (fun s => scoreValue s = value)).length

/-- The mode is the most frequent score -/
def isMode (value : Nat) : Prop :=
  ∀ (other : Nat), countOccurrences value ≥ countOccurrences other

/-- Theorem: The mode of the scores is 97 -/
theorem mode_is_97 : isMode 97 := by
  sorry

end mode_is_97_l474_47449


namespace probability_select_A_l474_47483

/-- The probability of selecting a specific person when choosing 2 from 5 -/
def probability_select_person (total : ℕ) (choose : ℕ) : ℚ :=
  (total - 1).choose (choose - 1) / total.choose choose

/-- The group size -/
def group_size : ℕ := 5

/-- The number of people to choose -/
def choose_size : ℕ := 2

theorem probability_select_A :
  probability_select_person group_size choose_size = 2/5 := by
  sorry

end probability_select_A_l474_47483


namespace defective_product_selection_l474_47418

/-- The set of possible numbers of defective products when selecting from a pool --/
def PossibleDefectives (total : ℕ) (defective : ℕ) (selected : ℕ) : Set ℕ :=
  {n : ℕ | n ≤ min defective selected ∧ n ≤ selected ∧ defective - n ≤ total - selected}

/-- Theorem stating the possible values for the number of defective products selected --/
theorem defective_product_selection (total : ℕ) (defective : ℕ) (selected : ℕ)
  (h_total : total = 8)
  (h_defective : defective = 2)
  (h_selected : selected = 3) :
  PossibleDefectives total defective selected = {0, 1, 2} :=
by sorry

end defective_product_selection_l474_47418


namespace interest_equality_implies_second_sum_l474_47485

/-- Proves that given a total sum of 2678, if the interest on the first part for 8 years at 3% per annum
    is equal to the interest on the second part for 3 years at 5% per annum, then the second part is 1648. -/
theorem interest_equality_implies_second_sum (total : ℝ) (first : ℝ) (second : ℝ) :
  total = 2678 →
  first + second = total →
  (first * (3/100) * 8) = (second * (5/100) * 3) →
  second = 1648 := by
sorry

end interest_equality_implies_second_sum_l474_47485


namespace nacl_formed_l474_47482

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String

-- Define the moles of substances
structure Moles where
  nh4cl : ℝ
  naoh : ℝ
  nacl : ℝ

-- Define the reaction and initial moles
def reaction : Reaction :=
  { reactant1 := "NH4Cl"
  , reactant2 := "NaOH"
  , product1 := "NaCl"
  , product2 := "NH3"
  , product3 := "H2O" }

def initial_moles : Moles :=
  { nh4cl := 2
  , naoh := 2
  , nacl := 0 }

-- Theorem statement
theorem nacl_formed (r : Reaction) (m : Moles) :
  r = reaction ∧ m = initial_moles →
  m.nacl + 2 = 2 :=
by sorry

end nacl_formed_l474_47482


namespace tim_weekly_earnings_l474_47477

/-- Tim's daily tasks -/
def daily_tasks : ℕ := 100

/-- Payment per task in dollars -/
def payment_per_task : ℚ := 6/5

/-- Days worked per week -/
def days_per_week : ℕ := 6

/-- Tim's weekly earnings in dollars -/
def weekly_earnings : ℚ := daily_tasks * payment_per_task * days_per_week

theorem tim_weekly_earnings :
  weekly_earnings = 720 := by sorry

end tim_weekly_earnings_l474_47477


namespace root_in_interval_l474_47419

-- Define the function
def f (x : ℝ) := x^3 - 2*x - 5

-- Theorem statement
theorem root_in_interval :
  (∃ x ∈ Set.Icc 2 3, f x = 0) →  -- root exists in [2,3]
  f 2.5 > 0 →                    -- f(2.5) > 0
  (∃ x ∈ Set.Ioo 2 2.5, f x = 0) -- root exists in (2,2.5)
  := by sorry

end root_in_interval_l474_47419


namespace min_value_of_f_l474_47462

def f (x : ℕ+) : ℚ := (x.val^2 + 33) / x.val

theorem min_value_of_f :
  (∀ x : ℕ+, f x ≥ 23/2) ∧ (∃ x : ℕ+, f x = 23/2) := by sorry

end min_value_of_f_l474_47462


namespace sum_interior_angles_heptagon_l474_47440

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A heptagon is a polygon with 7 sides -/
def heptagon_sides : ℕ := 7

/-- Theorem: The sum of the interior angles of a heptagon is 900 degrees -/
theorem sum_interior_angles_heptagon :
  sum_interior_angles heptagon_sides = 900 := by
  sorry

end sum_interior_angles_heptagon_l474_47440


namespace trig_equation_solution_l474_47411

theorem trig_equation_solution (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.cos (3*x))^2 = 1 ↔ 
  (∃ k : ℤ, x = k * Real.pi + Real.pi / 2 ∨ 
            x = k * Real.pi / 2 + Real.pi / 4 ∨ 
            x = k * Real.pi / 3 + Real.pi / 6) :=
by sorry

end trig_equation_solution_l474_47411


namespace polynomial_divisibility_l474_47423

theorem polynomial_divisibility (a b c d e : ℤ) :
  (∀ x : ℤ, (7 : ℤ) ∣ (a * x^4 + b * x^3 + c * x^2 + d * x + e)) →
  ((7 : ℤ) ∣ a) ∧ ((7 : ℤ) ∣ b) ∧ ((7 : ℤ) ∣ c) ∧ ((7 : ℤ) ∣ d) ∧ ((7 : ℤ) ∣ e) :=
by sorry

end polynomial_divisibility_l474_47423


namespace hyperbola_asymptotes_l474_47496

/-- Given a hyperbola with equation x²/m - y²/n = 1 where mn ≠ 0, 
    eccentricity 2, and one focus at (1, 0), 
    prove that its asymptotes are √3x ± y = 0 -/
theorem hyperbola_asymptotes 
  (m n : ℝ) 
  (h1 : m * n ≠ 0) 
  (h2 : ∀ x y : ℝ, x^2 / m - y^2 / n = 1) 
  (h3 : (Real.sqrt (m + n)) / (Real.sqrt m) = 2) 
  (h4 : ∃ x y : ℝ, x^2 / m - y^2 / n = 1 ∧ x = 1 ∧ y = 0) : 
  ∃ k : ℝ, k = Real.sqrt 3 ∧ 
    (∀ x y : ℝ, (k * x = y ∨ k * x = -y) ↔ x^2 / m - y^2 / n = 0) :=
sorry

end hyperbola_asymptotes_l474_47496


namespace relationship_significance_l474_47431

/-- The critical value for a 2x2 contingency table at 0.05 significance level -/
def critical_value : ℝ := 3.841

/-- The observed K^2 value from a 2x2 contingency table -/
def observed_value : ℝ := 4.013

/-- The maximum probability of making a mistake -/
def max_error_probability : ℝ := 0.05

/-- Theorem stating the relationship between the observed value, critical value, and maximum error probability -/
theorem relationship_significance (h : observed_value > critical_value) :
  max_error_probability = 0.05 := by
  sorry

end relationship_significance_l474_47431


namespace problem_solution_l474_47455

theorem problem_solution (x : ℚ) : 
  2 + 1 / (1 + 1 / (2 + 2 / (3 + x))) = 144 / 53 → x = 3 / 4 := by
  sorry

end problem_solution_l474_47455


namespace tape_length_calculation_l474_47443

/-- Calculate the total length of overlapping tape sheets -/
def totalTapeLength (sheetLength : ℝ) (overlap : ℝ) (numSheets : ℕ) : ℝ :=
  sheetLength + (numSheets - 1 : ℝ) * (sheetLength - overlap)

/-- Theorem: The total length of 64 sheets of tape, each 25 cm long, 
    with a 3 cm overlap between consecutive sheets, is 1411 cm -/
theorem tape_length_calculation :
  totalTapeLength 25 3 64 = 1411 := by
  sorry

end tape_length_calculation_l474_47443


namespace number_satisfying_condition_l474_47467

theorem number_satisfying_condition : ∃ x : ℝ, (0.1 * x = 0.2 * 650 + 190) ∧ (x = 3200) := by
  sorry

end number_satisfying_condition_l474_47467


namespace inequality_and_equality_condition_l474_47491

theorem inequality_and_equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a^2 + 1 / b^2 + 8 * a * b ≥ 8 ∧
  (1 / a^2 + 1 / b^2 + 8 * a * b = 8 ↔ a = 1/2 ∧ b = 1/2) := by
  sorry

end inequality_and_equality_condition_l474_47491


namespace string_cheese_calculation_l474_47428

/-- The number of string cheeses in each package for Kelly's kids' lunches. -/
def string_cheeses_per_package : ℕ := by sorry

theorem string_cheese_calculation (days_per_week : ℕ) (oldest_daily : ℕ) (youngest_daily : ℕ) 
  (weeks : ℕ) (packages : ℕ) (h1 : days_per_week = 5) (h2 : oldest_daily = 2) 
  (h3 : youngest_daily = 1) (h4 : weeks = 4) (h5 : packages = 2) : 
  string_cheeses_per_package = 30 := by sorry

end string_cheese_calculation_l474_47428


namespace union_of_A_and_B_l474_47469

def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℕ := {3, 4, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4, 5, 7} := by sorry

end union_of_A_and_B_l474_47469


namespace number_of_values_in_calculation_l474_47492

theorem number_of_values_in_calculation 
  (initial_average : ℝ)
  (correct_average : ℝ)
  (incorrect_value : ℝ)
  (correct_value : ℝ)
  (h1 : initial_average = 46)
  (h2 : correct_average = 51)
  (h3 : incorrect_value = 25)
  (h4 : correct_value = 75) :
  ∃ (n : ℕ), n > 0 ∧ 
    n * initial_average + (correct_value - incorrect_value) = n * correct_average ∧
    n = 10 := by
sorry

end number_of_values_in_calculation_l474_47492


namespace age_difference_l474_47476

-- Define variables for ages
variable (a b c d : ℕ)

-- Define the conditions
def condition1 : Prop := a + b = b + c + 15
def condition2 : Prop := a + d = c + d + 12
def condition3 : Prop := a = d + 3

-- Theorem statement
theorem age_difference (h1 : condition1 a b c) (h2 : condition2 a c d) (h3 : condition3 a d) :
  a - c = 12 := by sorry

end age_difference_l474_47476


namespace extreme_value_at_negative_three_l474_47481

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 3*x - 9

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a + 3

theorem extreme_value_at_negative_three (a : ℝ) :
  (∃ (x : ℝ), f' a x = 0) ∧ f' a (-3) = 0 → a = 5 := by sorry

end extreme_value_at_negative_three_l474_47481


namespace more_boys_than_girls_l474_47497

theorem more_boys_than_girls (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 100 →
  boys + girls = total →
  3 * girls = 2 * boys →
  boys - girls = 20 :=
by
  sorry

end more_boys_than_girls_l474_47497


namespace system_equation_ratio_l474_47495

theorem system_equation_ratio (x y c d : ℝ) (h1 : 4 * x - 3 * y = c) (h2 : 2 * y - 8 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 2 := by
  sorry

end system_equation_ratio_l474_47495


namespace arithmetic_mean_value_l474_47450

/-- A normal distribution with given properties -/
structure NormalDistribution where
  σ : ℝ  -- standard deviation
  x : ℝ  -- value 2 standard deviations below the mean
  h : x = μ - 2 * σ  -- relation between x, μ, and σ

/-- The arithmetic mean of a normal distribution satisfying given conditions -/
def arithmetic_mean (d : NormalDistribution) : ℝ := 
  d.x + 2 * d.σ

/-- Theorem stating the arithmetic mean of the specific normal distribution -/
theorem arithmetic_mean_value (d : NormalDistribution) 
  (h_σ : d.σ = 1.5) (h_x : d.x = 11) : arithmetic_mean d = 14 := by
  sorry

end arithmetic_mean_value_l474_47450


namespace median_is_106_l474_47448

-- Define the list
def list_size (n : ℕ) : ℕ := if n ≤ 150 then n else 0

-- Define the sum of the list sizes
def total_elements : ℕ := (Finset.range 151).sum list_size

-- Define the median position
def median_position : ℕ := (total_elements + 1) / 2

-- Theorem statement
theorem median_is_106 : 
  ∃ (cumsum : ℕ → ℕ), 
    (∀ n, cumsum n = (Finset.range (n + 1)).sum list_size) ∧
    (cumsum 105 < median_position) ∧
    (median_position ≤ cumsum 106) :=
sorry

end median_is_106_l474_47448


namespace inequality_proof_l474_47429

theorem inequality_proof (α : ℝ) (m : ℕ) (a b : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2) (h3 : m ≥ 1) (h4 : 0 < a) (h5 : 0 < b) : 
  a / (Real.cos α)^m + b / (Real.sin α)^m ≥ (a^(2/(m+2)) + b^(2/(m+2)))^((m+2)/2) := by
  sorry

end inequality_proof_l474_47429


namespace matthew_crackers_l474_47473

theorem matthew_crackers (total_crackers : ℕ) (crackers_per_friend : ℕ) (num_friends : ℕ) :
  total_crackers = 8 →
  crackers_per_friend = 2 →
  total_crackers = num_friends * crackers_per_friend →
  num_friends = 4 := by
sorry

end matthew_crackers_l474_47473


namespace arithmetic_sequence_problem_l474_47410

theorem arithmetic_sequence_problem (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 1 = 2 →                                            -- given: a_1 = 2
  a 3 + a 5 = 8 →                                      -- given: a_3 + a_5 = 8
  a 7 = 6 :=                                           -- to prove: a_7 = 6
by sorry

end arithmetic_sequence_problem_l474_47410


namespace initial_money_theorem_l474_47413

def meat_cost : ℕ := 17
def chicken_cost : ℕ := 22
def veggies_cost : ℕ := 43
def eggs_cost : ℕ := 5
def dog_food_cost : ℕ := 45
def cat_food_cost : ℕ := 18
def money_left : ℕ := 35

def total_spent : ℕ := meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost + cat_food_cost

theorem initial_money_theorem : 
  meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost + cat_food_cost + money_left = 185 := by
  sorry

end initial_money_theorem_l474_47413


namespace quadratic_root_problem_l474_47441

theorem quadratic_root_problem (m : ℝ) : 
  (1 : ℝ) ^ 2 + m * 1 - 4 = 0 → 
  ∃ (x : ℝ), x ≠ 1 ∧ x ^ 2 + m * x - 4 = 0 ∧ x = -4 := by
sorry

end quadratic_root_problem_l474_47441


namespace ellipse_line_intersection_l474_47454

def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def line (k m x y : ℝ) : Prop := y = k * x + m

def perpendicular_bisector (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - (y₁ + y₂) / 2) / (x - (x₁ + x₂) / 2) = -(x₂ - x₁) / (y₂ - y₁)

theorem ellipse_line_intersection (k m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line k m x₁ y₁ ∧ line k m x₂ y₂ ∧
    perpendicular_bisector x₁ y₁ x₂ y₂ 0 (-1/2)) →
  2 * k^2 + 1 = 2 * m := by
sorry

end ellipse_line_intersection_l474_47454


namespace ball_returns_in_five_throws_l474_47499

/-- The number of elements in the circular arrangement -/
def n : ℕ := 13

/-- The number of elements skipped in each throw -/
def skip : ℕ := 4

/-- The number of throws needed to return to the starting element -/
def throws : ℕ := 5

/-- Function to calculate the next position after a throw -/
def nextPosition (current : ℕ) : ℕ :=
  (current + skip + 1) % n

/-- Theorem stating that it takes 5 throws to return to the starting position -/
theorem ball_returns_in_five_throws :
  (throws.iterate nextPosition 0) % n = 0 := by sorry

end ball_returns_in_five_throws_l474_47499


namespace canada_trip_problem_l474_47438

/-- Represents the exchange rate from US dollars to Canadian dollars -/
def exchange_rate : ℚ := 15 / 9

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem canada_trip_problem (d : ℕ) :
  (exchange_rate * d - 120 = d) → 
  d = 180 ∧ sum_of_digits d = 9 := by
  sorry

end canada_trip_problem_l474_47438


namespace m_zero_sufficient_not_necessary_l474_47458

-- Define an arithmetic sequence
def is_arithmetic_seq (b : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n, b (n + 1) - b n = m

-- Define an equal difference of squares sequence
def is_equal_diff_squares_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1)^2 - a n^2 = d

theorem m_zero_sufficient_not_necessary :
  (∀ b : ℕ → ℝ, is_arithmetic_seq b 0 → is_equal_diff_squares_seq b) ∧
  (∃ b : ℕ → ℝ, ∃ m : ℝ, m ≠ 0 ∧ is_arithmetic_seq b m ∧ is_equal_diff_squares_seq b) :=
by sorry

end m_zero_sufficient_not_necessary_l474_47458


namespace sum_squares_plus_product_lower_bound_l474_47446

theorem sum_squares_plus_product_lower_bound 
  (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a + b + c = 3) : 
  a^2 + b^2 + c^2 + a*b*c ≥ 4 := by
  sorry

end sum_squares_plus_product_lower_bound_l474_47446


namespace min_boxes_for_treat_bags_l474_47444

/-- Represents the number of items in each box -/
structure BoxSizes where
  chocolate : Nat
  mint : Nat
  caramel : Nat

/-- Represents the number of boxes of each item -/
structure Boxes where
  chocolate : Nat
  mint : Nat
  caramel : Nat

/-- Calculates the total number of boxes -/
def totalBoxes (b : Boxes) : Nat :=
  b.chocolate + b.mint + b.caramel

/-- Checks if the given number of boxes results in complete treat bags with no leftovers -/
def isValidDistribution (sizes : BoxSizes) (boxes : Boxes) : Prop :=
  sizes.chocolate * boxes.chocolate = sizes.mint * boxes.mint ∧
  sizes.chocolate * boxes.chocolate = sizes.caramel * boxes.caramel

/-- The main theorem stating the minimum number of boxes needed -/
theorem min_boxes_for_treat_bags : ∃ (boxes : Boxes),
  let sizes : BoxSizes := ⟨50, 40, 25⟩
  isValidDistribution sizes boxes ∧ 
  totalBoxes boxes = 17 ∧
  (∀ (other : Boxes), isValidDistribution sizes other → totalBoxes other ≥ totalBoxes boxes) := by
  sorry

end min_boxes_for_treat_bags_l474_47444
