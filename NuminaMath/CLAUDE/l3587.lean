import Mathlib

namespace quadratic_sum_l3587_358748

theorem quadratic_sum (x y : ℝ) 
  (h1 : 4 * x + 2 * y = 12) 
  (h2 : 2 * x + 4 * y = 20) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 544 := by
  sorry

end quadratic_sum_l3587_358748


namespace solution_of_equation_l3587_358762

theorem solution_of_equation (z : ℂ) : 
  (z^6 - 6*z^4 + 9*z^2 = 0) ↔ (z = -Real.sqrt 3 ∨ z = 0 ∨ z = Real.sqrt 3) := by
  sorry

end solution_of_equation_l3587_358762


namespace arithmetic_geometric_mean_difference_bound_l3587_358701

theorem arithmetic_geometric_mean_difference_bound (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a < b) : 
  (a + b) / 2 - Real.sqrt (a * b) < (b - a)^2 / (8 * a) := by
  sorry

end arithmetic_geometric_mean_difference_bound_l3587_358701


namespace sandra_savings_l3587_358711

-- Define the number of notepads
def num_notepads : ℕ := 8

-- Define the original price per notepad
def original_price : ℚ := 375 / 100

-- Define the discount rate
def discount_rate : ℚ := 25 / 100

-- Define the savings calculation
def savings : ℚ :=
  num_notepads * original_price - num_notepads * (original_price * (1 - discount_rate))

-- Theorem to prove
theorem sandra_savings : savings = 15 / 2 := by
  sorry

end sandra_savings_l3587_358711


namespace min_value_3x_plus_4y_l3587_358744

theorem min_value_3x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 5 * x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 5 :=
by sorry

end min_value_3x_plus_4y_l3587_358744


namespace discriminant_of_quadratic_equation_l3587_358779

/-- The discriminant of a quadratic equation ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The quadratic equation 6x² + (6 + 1/6)x + 1/6 -/
def quadratic_equation (x : ℚ) : ℚ := 6*x^2 + (6 + 1/6)*x + 1/6

theorem discriminant_of_quadratic_equation : 
  discriminant 6 (6 + 1/6) (1/6) = 1225/36 := by
  sorry

end discriminant_of_quadratic_equation_l3587_358779


namespace digit_sum_theorem_l3587_358783

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem digit_sum_theorem (a b c d : ℕ) (square : ℕ) :
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit square →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * 100 + 60 + b - (400 + c * 10 + d) = 2 →
  a + b + c + d = 10 ∨ a + b + c + d = 18 ∨ a + b + c + d = 19 :=
by sorry

end digit_sum_theorem_l3587_358783


namespace all_graphs_different_l3587_358765

-- Define the equations
def eq1 (x y : ℝ) : Prop := y = 2 * x - 1
def eq2 (x y : ℝ) : Prop := y = (4 * x^2 - 1) / (2 * x + 1)
def eq3 (x y : ℝ) : Prop := (2 * x + 1) * y = 4 * x^2 - 1

-- Define the graph of an equation as the set of points (x, y) that satisfy it
def graph (eq : ℝ → ℝ → Prop) : Set (ℝ × ℝ) := {p : ℝ × ℝ | eq p.1 p.2}

-- Theorem stating that all graphs are different
theorem all_graphs_different :
  graph eq1 ≠ graph eq2 ∧ graph eq1 ≠ graph eq3 ∧ graph eq2 ≠ graph eq3 :=
sorry

end all_graphs_different_l3587_358765


namespace not_always_equal_distribution_l3587_358760

/-- Represents the state of the pies on plates -/
structure PieState where
  numPlates : Nat
  totalPies : Nat
  blackPies : Nat
  whitePies : Nat

/-- Represents a move in the game -/
inductive Move
  | transfer : Nat → Move

/-- Checks if a pie state is valid -/
def isValidState (state : PieState) : Prop :=
  state.numPlates = 20 ∧
  state.totalPies = 40 ∧
  state.blackPies + state.whitePies = state.totalPies

/-- Checks if a pie state has equal distribution -/
def hasEqualDistribution (state : PieState) : Prop :=
  state.blackPies = state.whitePies

/-- Applies a move to a pie state -/
def applyMove (state : PieState) (move : Move) : PieState :=
  match move with
  | Move.transfer n => 
      { state with 
        blackPies := state.blackPies + n,
        whitePies := state.whitePies - n
      }

/-- Theorem: It's not always possible to achieve equal distribution -/
theorem not_always_equal_distribution :
  ∃ (initialState : PieState),
    isValidState initialState ∧
    ∀ (moves : List Move),
      ¬hasEqualDistribution (moves.foldl applyMove initialState) :=
sorry

end not_always_equal_distribution_l3587_358760


namespace airplane_seats_l3587_358785

/-- Represents the total number of seats on an airplane -/
def total_seats : ℕ := 216

/-- Represents the number of seats in First Class -/
def first_class_seats : ℕ := 36

/-- Theorem stating that the total number of seats on the airplane is 216 -/
theorem airplane_seats :
  (first_class_seats : ℚ) + (1 : ℚ) / 3 * total_seats + (1 : ℚ) / 2 * total_seats = total_seats :=
by sorry

end airplane_seats_l3587_358785


namespace compare_cubic_and_quadratic_diff_l3587_358714

theorem compare_cubic_and_quadratic_diff (a b : ℝ) :
  (a ≥ b → a^3 - b^3 ≥ a*b^2 - a^2*b) ∧
  (a < b → a^3 - b^3 ≤ a*b^2 - a^2*b) :=
by sorry

end compare_cubic_and_quadratic_diff_l3587_358714


namespace ABCD_equals_one_l3587_358773

theorem ABCD_equals_one :
  let A := Real.sqrt 3003 + Real.sqrt 3004
  let B := -Real.sqrt 3003 - Real.sqrt 3004
  let C := Real.sqrt 3003 - Real.sqrt 3004
  let D := Real.sqrt 3004 - Real.sqrt 3003
  A * B * C * D = 1 := by
  sorry

end ABCD_equals_one_l3587_358773


namespace dandelion_picking_average_l3587_358770

/-- Represents the number of dandelions picked by Billy and George -/
structure DandelionPicks where
  billy_initial : ℕ
  george_initial : ℕ
  billy_additional : ℕ
  george_additional : ℕ

/-- Calculates the average number of dandelions picked -/
def average_picks (d : DandelionPicks) : ℚ :=
  (d.billy_initial + d.george_initial + d.billy_additional + d.george_additional : ℚ) / 2

/-- Theorem stating the average number of dandelions picked by Billy and George -/
theorem dandelion_picking_average :
  ∃ d : DandelionPicks,
    d.billy_initial = 36 ∧
    d.george_initial = (2 * d.billy_initial) / 5 ∧
    d.billy_additional = (5 * d.billy_initial) / 3 ∧
    d.george_additional = (7 * d.george_initial) / 2 ∧
    average_picks d = 79.5 :=
by
  sorry


end dandelion_picking_average_l3587_358770


namespace max_tickets_purchasable_l3587_358729

def ticket_price : ℚ := 15.75
def processing_fee : ℚ := 1.25
def budget : ℚ := 150.00

theorem max_tickets_purchasable :
  ∀ n : ℕ, n * (ticket_price + processing_fee) ≤ budget ↔ n ≤ 8 :=
by sorry

end max_tickets_purchasable_l3587_358729


namespace n_div_16_equals_4_pow_8086_l3587_358707

theorem n_div_16_equals_4_pow_8086 (n : ℕ) : n = 16^4044 → n / 16 = 4^8086 := by
  sorry

end n_div_16_equals_4_pow_8086_l3587_358707


namespace rectangular_hall_dimension_difference_l3587_358733

/-- Proves that for a rectangular hall with width equal to half its length
    and an area of 800 square meters, the difference between its length
    and width is 20 meters. -/
theorem rectangular_hall_dimension_difference
  (length width : ℝ)
  (h1 : width = length / 2)
  (h2 : length * width = 800) :
  length - width = 20 := by
  sorry

end rectangular_hall_dimension_difference_l3587_358733


namespace g_zero_l3587_358799

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the relationship between h, f, and g
axiom h_def : h = f * g

-- Define the constant terms of f and h
axiom f_const : f.coeff 0 = -6
axiom h_const : h.coeff 0 = 12

-- Theorem to prove
theorem g_zero : g.eval 0 = -2 := by sorry

end g_zero_l3587_358799


namespace triangle_side_lengths_l3587_358745

theorem triangle_side_lengths : 
  let S := {(a, b, c) : ℕ × ℕ × ℕ | 
    a ≤ b ∧ b ≤ c ∧ 
    b^2 = a * c ∧ 
    (a = 100 ∨ c = 100)}
  S = {(49,70,100), (64,80,100), (81,90,100), (100,100,100), 
       (100,110,121), (100,120,144), (100,130,169), (100,140,196), 
       (100,150,225), (100,160,256)} := by
  sorry

end triangle_side_lengths_l3587_358745


namespace equal_area_triangles_l3587_358797

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem equal_area_triangles :
  triangleArea 13 13 10 = triangleArea 13 13 24 := by
  sorry

end equal_area_triangles_l3587_358797


namespace max_sum_abs_on_circle_l3587_358771

theorem max_sum_abs_on_circle : 
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ 
  (∀ x y : ℝ, x^2 + y^2 = 4 → |x| + |y| ≤ M) ∧
  (∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = M) := by
  sorry

end max_sum_abs_on_circle_l3587_358771


namespace cos_50_minus_tan_40_equals_sqrt_3_l3587_358708

theorem cos_50_minus_tan_40_equals_sqrt_3 :
  4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end cos_50_minus_tan_40_equals_sqrt_3_l3587_358708


namespace no_integer_roots_l3587_358763

theorem no_integer_roots (a b c : ℤ) (ha : a ≠ 0)
  (h0 : Odd (a * 0^2 + b * 0 + c))
  (h1 : Odd (a * 1^2 + b * 1 + c)) :
  ∀ t : ℤ, a * t^2 + b * t + c ≠ 0 := by
sorry

end no_integer_roots_l3587_358763


namespace regular_star_polygon_n_value_l3587_358706

/-- An n-pointed regular star polygon -/
structure RegularStarPolygon where
  n : ℕ
  edgeCount : ℕ
  edgeCount_eq : edgeCount = 2 * n
  angleA : ℝ
  angleB : ℝ
  angle_difference : angleB - angleA = 15

/-- The theorem stating that for a regular star polygon with the given properties, n must be 24 -/
theorem regular_star_polygon_n_value (star : RegularStarPolygon) : star.n = 24 := by
  sorry

end regular_star_polygon_n_value_l3587_358706


namespace binary_repr_24_l3587_358775

/-- The binary representation of a natural number -/
def binary_repr (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Theorem: The binary representation of 24 is [false, false, false, true, true] -/
theorem binary_repr_24 : binary_repr 24 = [false, false, false, true, true] := by
  sorry

end binary_repr_24_l3587_358775


namespace binary_search_sixteen_people_l3587_358705

/-- The number of tests required to identify one infected person in a group of size n using binary search. -/
def numTestsBinarySearch (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else 1 + numTestsBinarySearch (n / 2)

/-- Theorem: For a group of 16 people with one infected person, 4 tests are required using binary search. -/
theorem binary_search_sixteen_people :
  numTestsBinarySearch 16 = 4 := by
  sorry

end binary_search_sixteen_people_l3587_358705


namespace arithmetic_relations_l3587_358767

theorem arithmetic_relations : 
  (10 * 100 = 1000) ∧ 
  (10 * 1000 = 10000) ∧ 
  (10000 / 100 = 100) ∧ 
  (1000 / 10 = 100) := by
  sorry

end arithmetic_relations_l3587_358767


namespace expression_evaluation_l3587_358792

theorem expression_evaluation : (2^(1^(0^2)))^3 + (3^(1^2))^0 + 4^(0^1) = 10 := by
  sorry

end expression_evaluation_l3587_358792


namespace intersection_of_lines_l3587_358787

/-- Two lines intersect if and only if their slopes are not equal -/
def lines_intersect (m₁ m₂ : ℝ) : Prop := m₁ ≠ m₂

/-- The slope of a line in the form y = mx + b is m -/
def slope_of_line (m : ℝ) : ℝ := m

theorem intersection_of_lines :
  let line1_slope : ℝ := -1  -- slope of x + y - 1 = 0
  let line2_slope : ℝ := 1   -- slope of y = x - 1
  lines_intersect (slope_of_line line1_slope) (slope_of_line line2_slope) :=
by
  sorry

end intersection_of_lines_l3587_358787


namespace at_least_one_negative_l3587_358790

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  ¬(0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) := by
sorry

end at_least_one_negative_l3587_358790


namespace measure_one_kg_possible_l3587_358727

/-- Represents a balance scale with two pans -/
structure BalanceScale :=
  (left_pan : ℝ)
  (right_pan : ℝ)

/-- Represents the state of the weighing process -/
structure WeighingState :=
  (scale : BalanceScale)
  (remaining_grain : ℝ)
  (weighings_left : ℕ)

/-- Performs a single weighing operation -/
def perform_weighing (state : WeighingState) : WeighingState :=
  sorry

/-- Checks if the current state has isolated 1 kg of grain -/
def has_isolated_one_kg (state : WeighingState) : Prop :=
  sorry

/-- Theorem stating that it's possible to measure 1 kg of grain under the given conditions -/
theorem measure_one_kg_possible :
  ∃ (initial_state : WeighingState),
    initial_state.scale.left_pan = 0 ∧
    initial_state.scale.right_pan = 0 ∧
    initial_state.remaining_grain = 19 ∧
    initial_state.weighings_left = 3 ∧
    ∃ (final_state : WeighingState),
      final_state = (perform_weighing ∘ perform_weighing ∘ perform_weighing) initial_state ∧
      has_isolated_one_kg final_state :=
by
  sorry

end measure_one_kg_possible_l3587_358727


namespace otimes_composition_l3587_358786

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x^3 - y

-- Theorem statement
theorem otimes_composition (h : ℝ) : otimes h (otimes h h) = h := by
  sorry

end otimes_composition_l3587_358786


namespace expression_value_l3587_358722

theorem expression_value : 
  let x : ℝ := 2
  2 * x^2 + 3 * x^2 = 20 := by sorry

end expression_value_l3587_358722


namespace solid_volume_l3587_358702

/-- A solid with specific face dimensions -/
structure Solid where
  square_side : ℝ
  rect_length : ℝ
  rect_width : ℝ
  trapezoid_leg : ℝ

/-- The volume of the solid -/
noncomputable def volume (s : Solid) : ℝ := sorry

/-- Theorem stating that the volume of the specified solid is 552 dm³ -/
theorem solid_volume (s : Solid) 
  (h1 : s.square_side = 1) 
  (h2 : s.rect_length = 0.4)
  (h3 : s.rect_width = 0.2)
  (h4 : s.trapezoid_leg = 1.3) :
  volume s = 0.552 := by sorry

end solid_volume_l3587_358702


namespace equation_solutions_l3587_358740

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 + 2*x₁ = 0 ∧ x₂^2 + 2*x₂ = 0) ∧ x₁ = 0 ∧ x₂ = -2) ∧
  (∃ y₁ y₂ : ℝ, (3*y₁^2 + 2*y₁ - 1 = 0 ∧ 3*y₂^2 + 2*y₂ - 1 = 0) ∧ y₁ = 1/3 ∧ y₂ = -1) :=
by sorry

end equation_solutions_l3587_358740


namespace a_less_than_one_l3587_358713

/-- The sequence a_n defined recursively -/
def a (k : ℕ) : ℕ → ℚ
  | 0 => 1 / k
  | n + 1 => a k n + (1 / (n + 1)^2) * (a k n)^2

/-- The theorem stating the condition for a_n < 1 for all n -/
theorem a_less_than_one (k : ℕ) : (∀ n : ℕ, a k n < 1) ↔ k ≥ 3 := by sorry

end a_less_than_one_l3587_358713


namespace dave_tickets_remaining_l3587_358795

theorem dave_tickets_remaining (initial_tickets used_tickets : ℕ) :
  initial_tickets = 127 →
  used_tickets = 84 →
  initial_tickets - used_tickets = 43 :=
by sorry

end dave_tickets_remaining_l3587_358795


namespace scatter_plot_correct_placement_l3587_358789

/-- Represents a variable in a scatter plot -/
inductive Variable
| Forecast
| Explanatory

/-- Represents an axis in a scatter plot -/
inductive Axis
| X
| Y

/-- Determines the correct axis placement for a given variable -/
def correct_axis_placement (v : Variable) : Axis :=
  match v with
  | Variable.Forecast => Axis.Y
  | Variable.Explanatory => Axis.X

/-- Theorem stating the correct placement of variables in a scatter plot -/
theorem scatter_plot_correct_placement :
  (correct_axis_placement Variable.Forecast = Axis.Y) ∧
  (correct_axis_placement Variable.Explanatory = Axis.X) := by
  sorry

end scatter_plot_correct_placement_l3587_358789


namespace angles_do_not_determine_triangle_uniquely_l3587_358752

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  sum_angles : α + β + γ = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define a function to check if two triangles have the same angles
def SameAngles (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem statement
theorem angles_do_not_determine_triangle_uniquely :
  ∃ (t1 t2 : Triangle), SameAngles t1 t2 ∧ t1 ≠ t2 := by sorry

end angles_do_not_determine_triangle_uniquely_l3587_358752


namespace least_five_digit_divisible_by_12_15_18_l3587_358737

theorem least_five_digit_divisible_by_12_15_18 :
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ 
    12 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n →
    10080 ≤ n :=
by sorry

end least_five_digit_divisible_by_12_15_18_l3587_358737


namespace birthday_candles_distribution_l3587_358755

/-- The number of people sharing the candles -/
def num_people : ℕ := 4

/-- The number of candles Ambika has -/
def ambika_candles : ℕ := 4

/-- The ratio of Aniyah's candles to Ambika's candles -/
def aniyah_ratio : ℕ := 6

/-- The total number of candles -/
def total_candles : ℕ := aniyah_ratio * ambika_candles + ambika_candles

/-- The number of candles each person gets when shared equally -/
def candles_per_person : ℕ := total_candles / num_people

theorem birthday_candles_distribution :
  candles_per_person = 7 :=
sorry

end birthday_candles_distribution_l3587_358755


namespace garbage_collection_total_l3587_358734

/-- The total amount of garbage collected by two groups, where one group collected 387 pounds and the other collected 39 pounds less. -/
theorem garbage_collection_total : 
  let lizzie_group := 387
  let other_group := lizzie_group - 39
  lizzie_group + other_group = 735 := by
  sorry

end garbage_collection_total_l3587_358734


namespace complex_equation_solution_l3587_358761

theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : Complex.abs z = 2) 
  (h2 : (z - a)^2 = a) : 
  a = 2 := by
sorry

end complex_equation_solution_l3587_358761


namespace marker_cost_l3587_358756

/-- The cost of notebooks and markers -/
theorem marker_cost (n m : ℝ) 
  (eq1 : 3 * n + 4 * m = 5.70)
  (eq2 : 5 * n + 2 * m = 4.90) : 
  m = 0.9857 := by
sorry

end marker_cost_l3587_358756


namespace square_plus_reciprocal_square_l3587_358777

theorem square_plus_reciprocal_square (x : ℝ) (h : x ≠ 0) :
  x^4 + (1/x^4) = 47 → x^2 + (1/x^2) = 7 := by
  sorry

end square_plus_reciprocal_square_l3587_358777


namespace xyz_value_l3587_358742

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + x * y * z = 15) :
  x * y * z = 9 / 2 := by
sorry

end xyz_value_l3587_358742


namespace total_cost_of_clothing_l3587_358709

/-- The total cost of a shirt, pants, and shoes given specific pricing conditions -/
theorem total_cost_of_clothing (pants_price : ℝ) : 
  pants_price = 120 →
  let shirt_price := (3/4) * pants_price
  let shoes_price := pants_price + 10
  shirt_price + pants_price + shoes_price = 340 := by
sorry

end total_cost_of_clothing_l3587_358709


namespace arithmetic_sequence_difference_l3587_358719

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_difference
  (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 7 - a 3 = 20) :
  a 2008 - a 2000 = 40 :=
by sorry

end arithmetic_sequence_difference_l3587_358719


namespace local_max_implies_local_min_l3587_358768

-- Define the function f
variable (f : ℝ → ℝ)

-- Define x₀ as a real number not equal to 0
variable (x₀ : ℝ)
variable (h₁ : x₀ ≠ 0)

-- Define that x₀ is a local maximum point of f
def isLocalMax (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x ≤ f x₀

variable (h₂ : isLocalMax f x₀)

-- Define what it means for a point to be a local minimum
def isLocalMin (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x₀ ≤ f x

-- Theorem statement
theorem local_max_implies_local_min :
  isLocalMin (fun x => -f (-x)) (-x₀) := by sorry

end local_max_implies_local_min_l3587_358768


namespace runner_speed_increase_l3587_358726

theorem runner_speed_increase (v : ℝ) (h : v > 0) : 
  (v + 2) / v = 2.5 → (v + 4) / v = 4 := by
  sorry

end runner_speed_increase_l3587_358726


namespace equation_solution_l3587_358743

theorem equation_solution (n : ℝ) : 
  (1 / (n + 1) + 2 / (n + 1) + n / (n + 1) + 1 / (n + 2) = 4) ↔ 
  (n = (-3 + Real.sqrt 6) / 3 ∨ n = (-3 - Real.sqrt 6) / 3) := by
  sorry

end equation_solution_l3587_358743


namespace quadratic_solution_sum_l3587_358750

theorem quadratic_solution_sum (x p q : ℝ) : 
  (5 * x^2 + 7 = 4 * x - 12) →
  (∃ (i : ℂ), x = p + q * i ∨ x = p - q * i) →
  p + q^2 = 101 / 25 := by
  sorry

end quadratic_solution_sum_l3587_358750


namespace daughter_weight_l3587_358784

/-- Represents the weights of family members in kilograms -/
structure FamilyWeights where
  mother : ℝ
  daughter : ℝ
  grandchild : ℝ

/-- Conditions for the family weights problem -/
def FamilyWeightsProblem (w : FamilyWeights) : Prop :=
  w.mother + w.daughter + w.grandchild = 150 ∧
  w.daughter + w.grandchild = 60 ∧
  w.grandchild = (1 / 5) * w.mother

/-- The weight of the daughter is 42 kg given the conditions -/
theorem daughter_weight (w : FamilyWeights) 
  (h : FamilyWeightsProblem w) : w.daughter = 42 := by
  sorry

end daughter_weight_l3587_358784


namespace kamal_math_marks_l3587_358739

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  english : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  mathematics : ℕ
  average : ℕ
  total_subjects : ℕ

/-- Theorem stating that given Kamal's marks and average, his Mathematics marks must be 60 -/
theorem kamal_math_marks (kamal : StudentMarks) 
  (h1 : kamal.english = 76)
  (h2 : kamal.physics = 72)
  (h3 : kamal.chemistry = 65)
  (h4 : kamal.biology = 82)
  (h5 : kamal.average = 71)
  (h6 : kamal.total_subjects = 5) :
  kamal.mathematics = 60 := by
  sorry

end kamal_math_marks_l3587_358739


namespace equation_solutions_l3587_358700

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.log (x^2 - 5*x + 10) = 2

-- State the theorem
theorem equation_solutions :
  ∃ (x₁ x₂ : ℝ), equation x₁ ∧ equation x₂ ∧ 
  (abs (x₁ - 4.4) < 0.01) ∧ (abs (x₂ - 0.6) < 0.01) :=
sorry

end equation_solutions_l3587_358700


namespace power_equation_solution_l3587_358736

theorem power_equation_solution (n : ℕ) : 
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^22 → n = 21 := by
sorry

end power_equation_solution_l3587_358736


namespace largest_n_inequality_l3587_358758

theorem largest_n_inequality : 
  ∀ n : ℕ, (1/4 : ℚ) + (n/8 : ℚ) < (3/2 : ℚ) ↔ n ≤ 9 := by sorry

end largest_n_inequality_l3587_358758


namespace arithmetic_geometric_sequence_l3587_358780

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric progression -/
def geometric_prog (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_seq a →
  geometric_prog (a 1) (a 2) (a 5) →
  a 5 = 9 := by
sorry

end arithmetic_geometric_sequence_l3587_358780


namespace money_lent_to_B_l3587_358732

/-- Proves that the amount lent to B is 4000, given the problem conditions --/
theorem money_lent_to_B (total : ℕ) (rate_A rate_B : ℚ) (years : ℕ) (interest_diff : ℕ) :
  total = 10000 →
  rate_A = 15 / 100 →
  rate_B = 18 / 100 →
  years = 2 →
  interest_diff = 360 →
  ∃ (amount_A amount_B : ℕ),
    amount_A + amount_B = total ∧
    amount_A * rate_A * years = (amount_B * rate_B * years + interest_diff) ∧
    amount_B = 4000 :=
by sorry

end money_lent_to_B_l3587_358732


namespace polynomial_factorization_l3587_358774

theorem polynomial_factorization (x : ℝ) :
  (x^4 - 4*x^2 + 1) * (x^4 + 3*x^2 + 1) + 10*x^4 = 
  (x + 1)^2 * (x - 1)^2 * (x^2 + x + 1) * (x^2 - x + 1) := by
  sorry

end polynomial_factorization_l3587_358774


namespace initial_bunnies_l3587_358717

theorem initial_bunnies (initial : ℕ) : 
  (3 / 5 : ℚ) * initial + 2 * ((3 / 5 : ℚ) * initial) = 54 → initial = 30 := by
  sorry

end initial_bunnies_l3587_358717


namespace quadratic_minimum_l3587_358764

/-- Given a quadratic function f(x) = x^2 - 2x + m with a minimum value of 1 
    on the interval [3, +∞), prove that m = -2. -/
theorem quadratic_minimum (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x : ℝ, x ≥ 3 → f x = x^2 - 2*x + m) →
  (∀ x : ℝ, x ≥ 3 → f x ≥ 1) →
  (∃ x : ℝ, x ≥ 3 ∧ f x = 1) →
  m = -2 := by
  sorry

end quadratic_minimum_l3587_358764


namespace expressions_not_equivalent_l3587_358724

theorem expressions_not_equivalent :
  ∃ x : ℝ, (x^2 + 1 ≠ 0 ∧ x^2 + 2*x + 1 ≠ 0) →
    (x^2 + x + 1) / (x^2 + 1) ≠ ((x + 1)^2) / (x^2 + 2*x + 1) :=
by sorry

end expressions_not_equivalent_l3587_358724


namespace largest_stamps_per_page_l3587_358725

theorem largest_stamps_per_page (stamps_book1 stamps_book2 : ℕ) 
  (h1 : stamps_book1 = 960) 
  (h2 : stamps_book2 = 1200) 
  (h3 : stamps_book1 > 0) 
  (h4 : stamps_book2 > 0) : 
  ∃ (stamps_per_page : ℕ), 
    stamps_per_page > 0 ∧ 
    stamps_book1 % stamps_per_page = 0 ∧ 
    stamps_book2 % stamps_per_page = 0 ∧ 
    stamps_book1 / stamps_per_page ≥ 2 ∧ 
    stamps_book2 / stamps_per_page ≥ 2 ∧ 
    ∀ (n : ℕ), n > stamps_per_page → 
      (stamps_book1 % n ≠ 0 ∨ 
       stamps_book2 % n ≠ 0 ∨ 
       stamps_book1 / n < 2 ∨ 
       stamps_book2 / n < 2) :=
by
  sorry

end largest_stamps_per_page_l3587_358725


namespace enemy_plane_hit_probability_l3587_358794

/-- The probability that the enemy plane is hit given A's and B's hit probabilities -/
theorem enemy_plane_hit_probability (p_A p_B : ℝ) (h_A : p_A = 0.6) (h_B : p_B = 0.4) :
  1 - (1 - p_A) * (1 - p_B) = 0.76 := by
  sorry

end enemy_plane_hit_probability_l3587_358794


namespace area_of_region_l3587_358710

/-- The region defined by the inequality |4x-14| + |3y-9| ≤ 6 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |4 * p.1 - 14| + |3 * p.2 - 9| ≤ 6}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The area of the region defined by |4x-14| + |3y-9| ≤ 6 is 6 -/
theorem area_of_region : area Region = 6 := by sorry

end area_of_region_l3587_358710


namespace geraint_on_time_speed_l3587_358738

/-- The distance Geraint cycles to work in kilometers. -/
def distance : ℝ := sorry

/-- The time in hours that Geraint's journey should take to arrive on time. -/
def on_time : ℝ := sorry

/-- The speed in km/h at which Geraint arrives on time. -/
def on_time_speed : ℝ := sorry

/-- Theorem stating that Geraint's on-time speed is 20 km/h. -/
theorem geraint_on_time_speed : 
  (distance / 15 = on_time + 1/6) →  -- At 15 km/h, he's 10 minutes (1/6 hour) late
  (distance / 30 = on_time - 1/6) →  -- At 30 km/h, he's 10 minutes (1/6 hour) early
  on_time_speed = 20 := by
  sorry

end geraint_on_time_speed_l3587_358738


namespace three_digit_number_rearrangement_l3587_358731

def digit_sum (n : ℕ) : ℕ := n / 100 + (n / 10) % 10 + n % 10

def rearrangement_sum (abc : ℕ) : ℕ :=
  let a := abc / 100
  let b := (abc / 10) % 10
  let c := abc % 10
  (a * 100 + c * 10 + b) +
  (b * 100 + c * 10 + a) +
  (b * 100 + a * 10 + c) +
  (c * 100 + a * 10 + b) +
  (c * 100 + b * 10 + a)

theorem three_digit_number_rearrangement (abc : ℕ) :
  abc ≥ 100 ∧ abc < 1000 ∧ rearrangement_sum abc = 2670 → abc = 528 := by
  sorry

end three_digit_number_rearrangement_l3587_358731


namespace function_equality_implies_a_value_l3587_358712

open Real

theorem function_equality_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, x₀ + exp (x₀ - a) - (log (x₀ + 2) - 4 * exp (a - x₀)) = 3) →
  a = -log 2 - 1 := by
sorry

end function_equality_implies_a_value_l3587_358712


namespace quadratic_rational_root_parity_l3587_358746

theorem quadratic_rational_root_parity (a b c : ℤ) (x : ℚ) : 
  (a * x^2 + b * x + c = 0) → ¬(Odd a ∧ Odd b ∧ Odd c) :=
by
  sorry

end quadratic_rational_root_parity_l3587_358746


namespace statements_b_and_c_are_correct_l3587_358757

theorem statements_b_and_c_are_correct (a b c d : ℝ) :
  (ab > 0 ∧ b*c - a*d > 0 → c/a - d/b > 0) ∧
  (a > b ∧ c > d → a - d > b - c) := by
sorry

end statements_b_and_c_are_correct_l3587_358757


namespace four_vertices_unique_distances_five_vertices_not_unique_distances_l3587_358776

/-- Represents a regular 13-sided polygon -/
structure RegularPolygon13 where
  vertices : Fin 13 → ℝ × ℝ

/-- Calculates the distance between two vertices in a regular 13-sided polygon -/
def distance (p : RegularPolygon13) (v1 v2 : Fin 13) : ℝ := sorry

/-- Checks if all pairwise distances in a set of vertices are unique -/
def all_distances_unique (p : RegularPolygon13) (vs : Finset (Fin 13)) : Prop := sorry

theorem four_vertices_unique_distances (p : RegularPolygon13) :
  ∃ (vs : Finset (Fin 13)), vs.card = 4 ∧ all_distances_unique p vs := sorry

theorem five_vertices_not_unique_distances (p : RegularPolygon13) :
  ¬∃ (vs : Finset (Fin 13)), vs.card = 5 ∧ all_distances_unique p vs := sorry

end four_vertices_unique_distances_five_vertices_not_unique_distances_l3587_358776


namespace problem_statement_l3587_358772

theorem problem_statement : 
  (3 * (0.6 * 40) - (4/5 * 25) / 2) * (Real.sqrt 16 - 3) = 62 := by
  sorry

end problem_statement_l3587_358772


namespace T_10_mod_5_l3587_358730

def T : ℕ → ℕ
  | 0 => 0
  | 1 => 2
  | (n+2) =>
    let c₁ := T n
    let c₂ := T (n+1)
    c₁ + c₂

theorem T_10_mod_5 : T 10 % 5 = 4 := by
  sorry

end T_10_mod_5_l3587_358730


namespace inequalities_not_equivalent_l3587_358788

-- Define the two inequalities
def inequality1 (x : ℝ) : Prop := x + 3 - 1 / (x - 1) > -x + 2 - 1 / (x - 1)
def inequality2 (x : ℝ) : Prop := x + 3 > -x + 2

-- Theorem stating that the inequalities are not equivalent
theorem inequalities_not_equivalent : ¬(∀ x : ℝ, inequality1 x ↔ inequality2 x) := by
  sorry

end inequalities_not_equivalent_l3587_358788


namespace complex_fraction_simplification_l3587_358749

theorem complex_fraction_simplification :
  let z : ℂ := (1 + I) / (3 - 4*I)
  z = -(1/25) + (7/25)*I :=
by sorry

end complex_fraction_simplification_l3587_358749


namespace parabola_latus_rectum_l3587_358791

/-- 
Given a parabola with equation y^2 = 2px and its latus rectum with equation x = -2,
prove that p = 4.
-/
theorem parabola_latus_rectum (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Equation of the parabola
  (∀ y : ℝ, y^2 = 2*p*(-2)) → -- Equation of the latus rectum
  p = 4 := by
sorry

end parabola_latus_rectum_l3587_358791


namespace problem_solution_l3587_358781

theorem problem_solution (x : ℝ) 
  (h : x - Real.sqrt (x^2 - 4) + 1 / (x + Real.sqrt (x^2 - 4)) = 10) :
  x^2 - Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 225/16 := by
  sorry

end problem_solution_l3587_358781


namespace thirtieth_roots_with_real_fifth_power_l3587_358778

theorem thirtieth_roots_with_real_fifth_power (ω : ℂ) (h : ω^3 = 1 ∧ ω ≠ 1) :
  ∃! (s : Finset ℂ), 
    (∀ z ∈ s, z^30 = 1) ∧ 
    (∀ z ∈ s, ∃ r : ℝ, z^5 = r) ∧
    s.card = 10 :=
sorry

end thirtieth_roots_with_real_fifth_power_l3587_358778


namespace car_mpg_difference_l3587_358728

/-- Proves that the difference between highway and city miles per gallon is 9 --/
theorem car_mpg_difference (highway_miles : ℕ) (city_miles : ℕ) (city_mpg : ℕ) :
  highway_miles = 462 →
  city_miles = 336 →
  city_mpg = 24 →
  (highway_miles / (city_miles / city_mpg)) - city_mpg = 9 := by
  sorry

#check car_mpg_difference

end car_mpg_difference_l3587_358728


namespace rectangular_field_posts_l3587_358704

/-- Calculates the number of posts needed for a rectangular fence -/
def num_posts (length width post_spacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let num_sections := perimeter / post_spacing
  num_sections

theorem rectangular_field_posts :
  num_posts 6 8 2 = 14 :=
by sorry

end rectangular_field_posts_l3587_358704


namespace coeff_x_neg_one_proof_l3587_358754

/-- The coefficient of x^(-1) in the expansion of (√x - 2/x)^7 -/
def coeff_x_neg_one : ℤ := -280

/-- The binomial coefficient (7 choose 3) -/
def binom_7_3 : ℕ := Nat.choose 7 3

theorem coeff_x_neg_one_proof :
  coeff_x_neg_one = binom_7_3 * (-8) :=
sorry

end coeff_x_neg_one_proof_l3587_358754


namespace square_garden_side_length_l3587_358796

/-- Given a square garden with a perimeter of 112 meters, prove that the length of each side is 28 meters. -/
theorem square_garden_side_length :
  ∀ (side_length : ℝ),
  (4 * side_length = 112) →
  side_length = 28 :=
by sorry

end square_garden_side_length_l3587_358796


namespace percentage_difference_l3587_358782

theorem percentage_difference (x y : ℝ) (P : ℝ) : 
  x = y * 0.9 →                 -- x is 10% less than y
  y = 125 * (1 + P / 100) →     -- y is P% more than 125
  x = 123.75 →                  -- x is equal to 123.75
  P = 10 :=                     -- P is equal to 10
by
  sorry

end percentage_difference_l3587_358782


namespace range_of_t_l3587_358741

/-- Set A definition -/
def A : Set ℝ := {x : ℝ | (x + 8) / (x - 5) ≤ 0}

/-- Set B definition -/
def B (t : ℝ) : Set ℝ := {x : ℝ | t + 1 ≤ x ∧ x ≤ 2*t - 1}

/-- Theorem stating the range of t -/
theorem range_of_t (t : ℝ) : 
  (∃ x, x ∈ B t) → -- B is non-empty
  (A ∩ B t = ∅) → -- A and B have no intersection
  t ≥ 4 := by sorry

end range_of_t_l3587_358741


namespace q_polynomial_form_l3587_358718

theorem q_polynomial_form (q : ℝ → ℝ) 
  (h : ∀ x, q x + (2 * x^5 + 5 * x^4 + 4 * x^3 + 12 * x) = 3 * x^4 + 14 * x^3 + 32 * x^2 + 17 * x + 3) :
  ∀ x, q x = -2 * x^5 - 2 * x^4 + 10 * x^3 + 32 * x^2 + 5 * x + 3 := by
  sorry

end q_polynomial_form_l3587_358718


namespace possible_values_of_c_l3587_358716

theorem possible_values_of_c (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (h : a^3 - b^3 = a^2 - b^2) :
  {c : ℤ | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ a^3 - b^3 = a^2 - b^2 ∧ c = ⌊9 * a * b⌋} = {1, 2, 3} :=
by sorry

end possible_values_of_c_l3587_358716


namespace max_value_4tau_minus_n_l3587_358715

/-- τ(n) is the number of positive divisors of n -/
def τ (n : ℕ+) : ℕ := (Nat.divisors n.val).card

/-- The maximum value of 4τ(n) - n over all positive integers n is 12 -/
theorem max_value_4tau_minus_n :
  (∀ n : ℕ+, (4 * τ n : ℤ) - n.val ≤ 12) ∧
  (∃ n : ℕ+, (4 * τ n : ℤ) - n.val = 12) :=
sorry

end max_value_4tau_minus_n_l3587_358715


namespace terms_before_negative_23_l3587_358703

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem terms_before_negative_23 :
  let a₁ := 101
  let d := -4
  ∃ n : ℕ, 
    (arithmetic_sequence a₁ d n = -23) ∧ 
    (∀ k : ℕ, k < n → arithmetic_sequence a₁ d k > -23) ∧
    n - 1 = 31 :=
by sorry

end terms_before_negative_23_l3587_358703


namespace quadratic_inequality_solution_set_l3587_358751

theorem quadratic_inequality_solution_set (x : ℝ) :
  {x : ℝ | x^2 - 3*x + 2 ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end quadratic_inequality_solution_set_l3587_358751


namespace derivative_zero_necessary_not_sufficient_l3587_358735

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Define what it means for a function to be differentiable
def IsDifferentiable (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f

-- Define what it means for a point to be an extremum
def IsExtremum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- Theorem stating that f'(x) = 0 is necessary but not sufficient for x to be an extremum
theorem derivative_zero_necessary_not_sufficient (h : IsDifferentiable f) :
  (IsExtremum f x → deriv f x = 0) ∧
  ¬(deriv f x = 0 → IsExtremum f x) :=
sorry

end derivative_zero_necessary_not_sufficient_l3587_358735


namespace expression_evaluation_l3587_358720

theorem expression_evaluation :
  (2^1000 + 5^1001)^2 - (2^1000 - 5^1001)^2 = 20 * 10^1000 := by
  sorry

end expression_evaluation_l3587_358720


namespace cubic_equation_solution_l3587_358753

theorem cubic_equation_solution (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27 * x^3) 
  (h3 : a - b = x) : 
  a = (x * (1 + Real.sqrt (115 / 3))) / 2 ∨ 
  a = (x * (1 - Real.sqrt (115 / 3))) / 2 := by
sorry

end cubic_equation_solution_l3587_358753


namespace difference_of_sixes_in_7669_l3587_358798

/-- Given a natural number n, returns the digit at the i-th place (0-indexed from right) -/
def digit_at_place (n : ℕ) (i : ℕ) : ℕ := 
  (n / (10 ^ i)) % 10

/-- Given a natural number n, returns the place value of the digit at the i-th place -/
def place_value (n : ℕ) (i : ℕ) : ℕ := 
  digit_at_place n i * (10 ^ i)

theorem difference_of_sixes_in_7669 : 
  place_value 7669 2 - place_value 7669 1 = 540 := by sorry

end difference_of_sixes_in_7669_l3587_358798


namespace rectangle_dimension_change_l3587_358747

theorem rectangle_dimension_change (L W x : ℝ) (h_positive : L > 0 ∧ W > 0) :
  let new_area := L * (1 + x / 100) * W * (1 - x / 100)
  let original_area := L * W
  new_area = original_area * (1 + 4 / 100) →
  x = 20 := by
sorry

end rectangle_dimension_change_l3587_358747


namespace interest_for_one_rupee_l3587_358759

/-- Given that for 5000 rs, the interest is 200 paise, prove that the interest for 1 rs is 0.04 paise. -/
theorem interest_for_one_rupee (interest_5000 : ℝ) (h : interest_5000 = 200) :
  interest_5000 / 5000 = 0.04 := by
sorry

end interest_for_one_rupee_l3587_358759


namespace simplify_expression_l3587_358721

theorem simplify_expression (x y z : ℝ) : (x - (y - z)) - ((x - y) - z) = 2 * z := by
  sorry

end simplify_expression_l3587_358721


namespace exists_islands_with_inverse_area_relation_l3587_358766

/-- Represents a rectangular island with length and width in kilometers. -/
structure Island where
  length : ℝ
  width : ℝ

/-- Calculates the area of an island in square kilometers. -/
def islandArea (i : Island) : ℝ :=
  i.length * i.width

/-- Calculates the coastal water area of an island in square kilometers. 
    Coastal water is defined as the area within 50 km of the shore. -/
def coastalWaterArea (i : Island) : ℝ :=
  (i.length + 100) * (i.width + 100) - islandArea i

/-- Theorem stating that there exist two islands where the first has smaller area
    but larger coastal water area compared to the second. -/
theorem exists_islands_with_inverse_area_relation : 
  ∃ (i1 i2 : Island), 
    islandArea i1 < islandArea i2 ∧ 
    coastalWaterArea i1 > coastalWaterArea i2 :=
sorry

end exists_islands_with_inverse_area_relation_l3587_358766


namespace equation_solutions_l3587_358723

theorem equation_solutions : 
  ∀ n m : ℕ, m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ 
  ((n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) ∨ (n = 6 ∧ m = 54) ∨ (n = 6 ∧ m = 27)) := by
sorry

end equation_solutions_l3587_358723


namespace snatch_percentage_increase_l3587_358793

/-- Calculates the percentage increase in Snatch weight given initial weights and new total -/
theorem snatch_percentage_increase
  (initial_clean_jerk : ℝ)
  (initial_snatch : ℝ)
  (new_total : ℝ)
  (h1 : initial_clean_jerk = 80)
  (h2 : initial_snatch = 50)
  (h3 : new_total = 250)
  (h4 : 2 * initial_clean_jerk + new_snatch = new_total)
  : (new_snatch - initial_snatch) / initial_snatch * 100 = 80 :=
by
  sorry

#check snatch_percentage_increase

end snatch_percentage_increase_l3587_358793


namespace isosceles_triangle_vertex_angle_l3587_358769

-- Define an isosceles triangle
structure IsoscelesTriangle :=
  (vertex_angle : ℝ)
  (base_angle : ℝ)
  (is_isosceles : base_angle = base_angle)

-- Define the exterior angle
def exterior_angle (t : IsoscelesTriangle) : ℝ := 100

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (t : IsoscelesTriangle) 
  (h_exterior : exterior_angle t = 100) :
  t.vertex_angle = 20 ∨ t.vertex_angle = 80 :=
sorry

end isosceles_triangle_vertex_angle_l3587_358769
