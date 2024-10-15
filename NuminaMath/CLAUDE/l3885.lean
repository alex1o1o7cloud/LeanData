import Mathlib

namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3885_388528

/-- The number of ways to place n distinguishable balls into k indistinguishable boxes -/
def placeBalls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 31 ways to place 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : placeBalls 5 3 = 31 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3885_388528


namespace NUMINAMATH_CALUDE_sum_of_twenty_numbers_l3885_388545

theorem sum_of_twenty_numbers : 
  let numbers : List Nat := [87, 91, 94, 88, 93, 91, 89, 87, 92, 86, 90, 92, 88, 90, 91, 86, 89, 92, 95, 88]
  numbers.sum = 1799 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_twenty_numbers_l3885_388545


namespace NUMINAMATH_CALUDE_unique_divisible_by_29_l3885_388592

/-- Converts a base 7 number of the form 34x1 to decimal --/
def base7ToDecimal (x : ℕ) : ℕ := 3 * 7^3 + 4 * 7^2 + x * 7 + 1

/-- Checks if a number is divisible by 29 --/
def isDivisibleBy29 (n : ℕ) : Prop := n % 29 = 0

theorem unique_divisible_by_29 :
  ∃! x : ℕ, x < 7 ∧ isDivisibleBy29 (base7ToDecimal x) :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_by_29_l3885_388592


namespace NUMINAMATH_CALUDE_board_longest_piece_length_l3885_388522

/-- Given a board of length 240 cm cut into four pieces, prove that the longest piece is 120 cm -/
theorem board_longest_piece_length :
  ∀ (L M T F : ℝ),
    L + M + T + F = 240 →
    L = M + T + F →
    M = L / 2 - 10 →
    T ^ 2 = L - M →
    L = 120 := by
  sorry

end NUMINAMATH_CALUDE_board_longest_piece_length_l3885_388522


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l3885_388560

theorem quadratic_root_existence (a b c : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : a * x₁^2 + b * x₁ + c = 0)
  (h₂ : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃, (1/2 * a * x₃^2 + b * x₃ + c = 0) ∧ 
    ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l3885_388560


namespace NUMINAMATH_CALUDE_triangle_angle_properties_l3885_388599

theorem triangle_angle_properties (α : Real) (h1 : 0 < α) (h2 : α < π) 
  (h3 : Real.sin α + Real.cos α = 1/5) : 
  (Real.tan α = -4/3) ∧ 
  ((Real.sin (3*π/2 + α) * Real.sin (π/2 - α) * Real.tan (π - α)^3) / 
   (Real.cos (π/2 + α) * Real.cos (3*π/2 - α)) = -4/3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_properties_l3885_388599


namespace NUMINAMATH_CALUDE_line_intercepts_l3885_388594

/-- A line in the 2D plane defined by the equation y = x + 3 -/
structure Line where
  slope : ℝ := 1
  y_intercept : ℝ := 3

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ × ℝ :=
  (-l.y_intercept, 0)

/-- The y-intercept of a line -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.y_intercept)

theorem line_intercepts (l : Line) :
  x_intercept l = (-3, 0) ∧ y_intercept l = (0, 3) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_l3885_388594


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3885_388527

/-- The parabola -/
def parabola (x : ℝ) : ℝ := x^2

/-- The line parallel to the tangent line -/
def parallel_line (x y : ℝ) : Prop := 2*x - y + 4 = 0

/-- The proposed tangent line -/
def tangent_line (x y : ℝ) : Prop := 2*x - y - 1 = 0

/-- Theorem: The tangent line to the parabola y = x^2 that is parallel to 2x - y + 4 = 0 
    has the equation 2x - y - 1 = 0 -/
theorem tangent_line_equation : 
  ∃ (x₀ y₀ : ℝ), 
    y₀ = parabola x₀ ∧ 
    tangent_line x₀ y₀ ∧
    ∀ (x y : ℝ), parallel_line x y ↔ ∃ (k : ℝ), y = k*x + (k*2 - 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3885_388527


namespace NUMINAMATH_CALUDE_sugar_and_salt_pricing_l3885_388571

/-- Given the price of sugar and salt, prove the cost of a specific quantity -/
theorem sugar_and_salt_pricing
  (price_2kg_sugar_5kg_salt : ℝ)
  (price_1kg_sugar : ℝ)
  (h1 : price_2kg_sugar_5kg_salt = 5.50)
  (h2 : price_1kg_sugar = 1.50) :
  3 * price_1kg_sugar + (price_2kg_sugar_5kg_salt - 2 * price_1kg_sugar) / 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_sugar_and_salt_pricing_l3885_388571


namespace NUMINAMATH_CALUDE_calculate_expression_l3885_388552

theorem calculate_expression : 5 * 7 + 9 * 4 - 36 / 3 = 59 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3885_388552


namespace NUMINAMATH_CALUDE_f_2_equals_216_l3885_388508

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + 3*x^3 + 4*x^2 + 5*x + 6

theorem f_2_equals_216 : f 2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_f_2_equals_216_l3885_388508


namespace NUMINAMATH_CALUDE_cube_values_l3885_388564

def f (n : ℕ) : ℤ := n^3 - 18*n^2 + 115*n - 391

def is_cube (x : ℤ) : Prop := ∃ y : ℤ, y^3 = x

theorem cube_values :
  {n : ℕ | is_cube (f n)} = {7, 11, 12, 25} := by sorry

end NUMINAMATH_CALUDE_cube_values_l3885_388564


namespace NUMINAMATH_CALUDE_friends_video_count_l3885_388548

/-- The number of videos watched by three friends. -/
def total_videos (kelsey ekon uma : ℕ) : ℕ := kelsey + ekon + uma

/-- Theorem stating the total number of videos watched by the three friends. -/
theorem friends_video_count :
  ∀ (kelsey ekon uma : ℕ),
  kelsey = 160 →
  kelsey = ekon + 43 →
  uma = ekon + 17 →
  total_videos kelsey ekon uma = 411 :=
by
  sorry

end NUMINAMATH_CALUDE_friends_video_count_l3885_388548


namespace NUMINAMATH_CALUDE_order_of_variables_l3885_388500

theorem order_of_variables (a b c d : ℝ) 
  (h1 : a > b) 
  (h2 : d > c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  b < c ∧ c < a ∧ a < d :=
sorry

end NUMINAMATH_CALUDE_order_of_variables_l3885_388500


namespace NUMINAMATH_CALUDE_dish_heating_rate_l3885_388581

/-- Given the initial and final temperatures of a dish and the time taken to heat it,
    calculate the heating rate in degrees per minute. -/
theorem dish_heating_rate 
  (initial_temp : ℝ) 
  (final_temp : ℝ) 
  (heating_time : ℝ) 
  (h1 : initial_temp = 20) 
  (h2 : final_temp = 100) 
  (h3 : heating_time = 16) : 
  (final_temp - initial_temp) / heating_time = 5 := by
  sorry

#check dish_heating_rate

end NUMINAMATH_CALUDE_dish_heating_rate_l3885_388581


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3885_388559

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3885_388559


namespace NUMINAMATH_CALUDE_coeff_x3_is_42_l3885_388530

/-- First polynomial: x^5 - 4x^4 + 7x^3 - 5x^2 + 3x - 2 -/
def p1 (x : ℝ) : ℝ := x^5 - 4*x^4 + 7*x^3 - 5*x^2 + 3*x - 2

/-- Second polynomial: 3x^2 - 5x + 6 -/
def p2 (x : ℝ) : ℝ := 3*x^2 - 5*x + 6

/-- The product of the two polynomials -/
def product (x : ℝ) : ℝ := p1 x * p2 x

/-- Theorem: The coefficient of x^3 in the product of p1 and p2 is 42 -/
theorem coeff_x3_is_42 : ∃ (a b c d e f : ℝ), product x = a*x^5 + b*x^4 + 42*x^3 + d*x^2 + e*x + f :=
sorry

end NUMINAMATH_CALUDE_coeff_x3_is_42_l3885_388530


namespace NUMINAMATH_CALUDE_min_gumballs_for_given_machine_l3885_388595

/-- Represents the number of gumballs of each color -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)

/-- The minimum number of gumballs needed to guarantee at least 4 of the same color -/
def minGumballs (machine : GumballMachine) : ℕ := sorry

/-- Theorem stating the minimum number of gumballs needed for the given machine -/
theorem min_gumballs_for_given_machine :
  let machine : GumballMachine := ⟨8, 10, 6⟩
  minGumballs machine = 10 := by sorry

end NUMINAMATH_CALUDE_min_gumballs_for_given_machine_l3885_388595


namespace NUMINAMATH_CALUDE_distinct_and_no_real_solutions_l3885_388585

theorem distinct_and_no_real_solutions : 
  ∀ b c : ℕ+, 
    (∃ x y : ℝ, x ≠ y ∧ x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0) ∧ 
    (∀ z : ℝ, z^2 + c*z + b ≠ 0) → 
    ((b = 3 ∧ c = 1) ∨ (b = 3 ∧ c = 2)) := by
  sorry

end NUMINAMATH_CALUDE_distinct_and_no_real_solutions_l3885_388585


namespace NUMINAMATH_CALUDE_quadratic_vertex_on_x_axis_l3885_388563

-- Define the quadratic function
def f (x m : ℝ) : ℝ := x^2 - x + m

-- Define the condition for the vertex being on the x-axis
def vertex_on_x_axis (m : ℝ) : Prop :=
  let x₀ := 1/2  -- x-coordinate of the vertex
  f x₀ m = 0

-- Theorem statement
theorem quadratic_vertex_on_x_axis (m : ℝ) :
  vertex_on_x_axis m → m = 1/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_on_x_axis_l3885_388563


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3885_388526

theorem trigonometric_identities :
  (∃ x : ℝ, x = 75 * π / 180 ∧ (Real.cos x)^2 = (2 - Real.sqrt 3) / 4) ∧
  (∃ y z : ℝ, y = π / 180 ∧ z = 44 * π / 180 ∧
    Real.tan y + Real.tan z + Real.tan y * Real.tan z = 1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3885_388526


namespace NUMINAMATH_CALUDE_inequality_range_l3885_388536

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 4 * x + a > 1 - 2 * x^2) ↔ a > 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3885_388536


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3885_388544

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (2 : ℝ) / (3 * Real.sqrt 7 + 2 * Real.sqrt 13) =
    (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = 6 ∧
    B = 7 ∧
    C = -4 ∧
    D = 13 ∧
    E = 11 ∧
    Int.gcd A E = 1 ∧
    Int.gcd C E = 1 ∧
    ¬∃ (k : ℤ), k > 1 ∧ k ^ 2 ∣ B ∧
    ¬∃ (k : ℤ), k > 1 ∧ k ^ 2 ∣ D :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3885_388544


namespace NUMINAMATH_CALUDE_min_area_hyperbola_triangle_l3885_388512

/-- A point on the hyperbola xy = 1 -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : x * y = 1

/-- An isosceles right triangle on the hyperbola xy = 1 -/
structure HyperbolaTriangle where
  A : HyperbolaPoint
  B : HyperbolaPoint
  C : HyperbolaPoint
  is_right_angle : (B.x - A.x) * (C.x - A.x) + (B.y - A.y) * (C.y - A.y) = 0
  is_isosceles : (B.x - A.x)^2 + (B.y - A.y)^2 = (C.x - A.x)^2 + (C.y - A.y)^2

/-- The area of a triangle given by three points -/
def triangleArea (A B C : HyperbolaPoint) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

/-- The theorem stating the minimum area of an isosceles right triangle on the hyperbola xy = 1 -/
theorem min_area_hyperbola_triangle :
  ∀ T : HyperbolaTriangle, triangleArea T.A T.B T.C ≥ 3 * Real.sqrt 3 := by
  sorry

#check min_area_hyperbola_triangle

end NUMINAMATH_CALUDE_min_area_hyperbola_triangle_l3885_388512


namespace NUMINAMATH_CALUDE_perfect_non_spiral_shells_l3885_388551

theorem perfect_non_spiral_shells (total_perfect : ℕ) (total_broken : ℕ) 
  (h1 : total_perfect = 17)
  (h2 : total_broken = 52)
  (h3 : total_broken / 2 = total_broken - total_broken / 2)  -- Half of broken shells are spiral
  (h4 : total_broken / 2 = (total_perfect - (total_perfect - (total_broken / 2 - 21))) + 21) :
  total_perfect - (total_perfect - (total_broken / 2 - 21)) = 12 := by
  sorry

#check perfect_non_spiral_shells

end NUMINAMATH_CALUDE_perfect_non_spiral_shells_l3885_388551


namespace NUMINAMATH_CALUDE_soldiers_food_calculation_l3885_388503

/-- Given the following conditions:
    1. Soldiers on the second side are given 2 pounds less food than the first side.
    2. The first side has 4000 soldiers.
    3. The second side has 500 soldiers fewer than the first side.
    4. The total amount of food both sides are eating altogether every day is 68000 pounds.

    Prove that the amount of food each soldier on the first side needs every day is 10 pounds. -/
theorem soldiers_food_calculation (food_first : ℝ) : 
  (4000 : ℝ) * food_first + (4000 - 500) * (food_first - 2) = 68000 → food_first = 10 := by
  sorry

end NUMINAMATH_CALUDE_soldiers_food_calculation_l3885_388503


namespace NUMINAMATH_CALUDE_wilted_flowers_calculation_l3885_388534

/-- 
Given:
- initial_flowers: The initial number of flowers picked
- flowers_per_bouquet: The number of flowers in each bouquet
- bouquets_made: The number of bouquets that could be made after some flowers wilted

Prove:
The number of wilted flowers is equal to the initial number of flowers minus
the product of the number of bouquets made and the number of flowers per bouquet.
-/
theorem wilted_flowers_calculation (initial_flowers flowers_per_bouquet bouquets_made : ℕ) :
  initial_flowers - (bouquets_made * flowers_per_bouquet) = 
  initial_flowers - bouquets_made * flowers_per_bouquet :=
by sorry

end NUMINAMATH_CALUDE_wilted_flowers_calculation_l3885_388534


namespace NUMINAMATH_CALUDE_marble_selection_ways_l3885_388524

def total_marbles : ℕ := 15
def specific_colors : ℕ := 5
def marbles_to_choose : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem marble_selection_ways :
  (specific_colors * choose (total_marbles - specific_colors - 1) (marbles_to_choose - 1)) = 630 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l3885_388524


namespace NUMINAMATH_CALUDE_complement_of_union_l3885_388598

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

-- Theorem statement
theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3885_388598


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_4410_l3885_388541

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_of_4410 :
  largest_perfect_square_factor 4410 = 441 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_4410_l3885_388541


namespace NUMINAMATH_CALUDE_cyclic_sum_extrema_l3885_388584

def cyclic_sum (a : List ℕ) : ℕ :=
  (List.zip a (a.rotate 1)).map (fun (x, y) => x * y) |>.sum

def is_permutation (a : List ℕ) (n : ℕ) : Prop :=
  a.length = n ∧ a.toFinset = Finset.range n

def max_permutation (n : ℕ) : List ℕ :=
  (List.range ((n + 1) / 2)).map (fun i => 2 * i + 1) ++
  (List.range (n / 2)).reverse.map (fun i => 2 * (i + 1))

def min_permutation (n : ℕ) : List ℕ :=
  if n % 2 = 0 then
    (List.range (n / 2)).reverse.map (fun i => n - 2 * i) ++
    (List.range (n / 2)).map (fun i => 2 * i + 1)
  else
    (List.range ((n + 1) / 2)).reverse.map (fun i => n - 2 * i) ++
    (List.range (n / 2)).map (fun i => 2 * i + 2)

theorem cyclic_sum_extrema (n : ℕ) (a : List ℕ) (h : is_permutation a n) :
  cyclic_sum a ≤ cyclic_sum (max_permutation n) ∧
  cyclic_sum (min_permutation n) ≤ cyclic_sum a := by sorry

end NUMINAMATH_CALUDE_cyclic_sum_extrema_l3885_388584


namespace NUMINAMATH_CALUDE_coin_grid_intersection_probability_l3885_388555

/-- Probability of a coin intersecting grid lines -/
theorem coin_grid_intersection_probability
  (grid_edge_length : ℝ)
  (coin_diameter : ℝ)
  (h_grid : grid_edge_length = 6)
  (h_coin : coin_diameter = 2) :
  (1 : ℝ) - (grid_edge_length - coin_diameter)^2 / grid_edge_length^2 = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_coin_grid_intersection_probability_l3885_388555


namespace NUMINAMATH_CALUDE_committee_meeting_attendance_l3885_388586

theorem committee_meeting_attendance :
  ∀ (associate_profs assistant_profs : ℕ),
    2 * associate_profs + assistant_profs = 11 →
    associate_profs + 2 * assistant_profs = 16 →
    associate_profs + assistant_profs = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_committee_meeting_attendance_l3885_388586


namespace NUMINAMATH_CALUDE_equation_solution_l3885_388505

theorem equation_solution : ∃! x : ℝ, (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3885_388505


namespace NUMINAMATH_CALUDE_factor_expression_l3885_388513

theorem factor_expression (y : ℝ) : 5 * y * (y - 2) + 9 * (y - 2) = (y - 2) * (5 * y + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3885_388513


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3885_388531

theorem pure_imaginary_condition (a : ℝ) : 
  let i : ℂ := Complex.I
  let z : ℂ := (a + i) / (1 + i)
  (z.re = 0) → a = -1 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3885_388531


namespace NUMINAMATH_CALUDE_total_seats_calculation_l3885_388573

/-- The number of students per bus -/
def students_per_bus : ℝ := 14.0

/-- The number of buses -/
def number_of_buses : ℝ := 2.0

/-- The total number of seats taken up by students -/
def total_seats : ℝ := students_per_bus * number_of_buses

theorem total_seats_calculation : total_seats = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_seats_calculation_l3885_388573


namespace NUMINAMATH_CALUDE_calculate_food_price_l3885_388589

/-- Given a total bill that includes tax and tip, calculate the original food price -/
theorem calculate_food_price (total : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (food_price : ℝ) : 
  total = 211.20 ∧ 
  tax_rate = 0.10 ∧ 
  tip_rate = 0.20 ∧ 
  total = food_price * (1 + tax_rate) * (1 + tip_rate) → 
  food_price = 160 := by
  sorry

end NUMINAMATH_CALUDE_calculate_food_price_l3885_388589


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3885_388565

theorem partial_fraction_decomposition (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ 3 →
    1 / (x^3 - 3*x^2 - 13*x + 15) = A / (x - 1) + B / (x - 3) + C / ((x - 3)^2)) →
  A = 1/4 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3885_388565


namespace NUMINAMATH_CALUDE_distance_between_points_l3885_388538

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, -3)
  let p2 : ℝ × ℝ := (5, 6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3885_388538


namespace NUMINAMATH_CALUDE_log_2_base_10_bounds_l3885_388549

-- Define the given conditions
axiom pow_10_4 : (10 : ℝ) ^ 4 = 10000
axiom pow_10_5 : (10 : ℝ) ^ 5 = 100000
axiom pow_2_12 : (2 : ℝ) ^ 12 = 4096
axiom pow_2_15 : (2 : ℝ) ^ 15 = 32768

-- State the theorem
theorem log_2_base_10_bounds :
  0.30 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_log_2_base_10_bounds_l3885_388549


namespace NUMINAMATH_CALUDE_natural_number_power_equality_l3885_388540

theorem natural_number_power_equality (p q : ℕ) (h : p^p + q^q = p^q + q^p) : p = q := by
  sorry

end NUMINAMATH_CALUDE_natural_number_power_equality_l3885_388540


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3885_388517

/-- The eccentricity of an ellipse with equation x²/16 + y²/12 = 1 is 1/2 -/
theorem ellipse_eccentricity : ∃ e : ℝ,
  (∀ x y : ℝ, x^2/16 + y^2/12 = 1 → 
    ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧
      x^2/a^2 + y^2/b^2 = 1 ∧
      c^2 = a^2 - b^2 ∧
      e = c/a) ∧
  e = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3885_388517


namespace NUMINAMATH_CALUDE_max_hangers_buyable_l3885_388596

def total_budget : ℝ := 60
def tissue_cost : ℝ := 34.8
def hanger_cost : ℝ := 1.6

theorem max_hangers_buyable : 
  ⌊(total_budget - tissue_cost) / hanger_cost⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_max_hangers_buyable_l3885_388596


namespace NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l3885_388562

/-- Given a parallelogram with area 242 sq m and base 11 m, prove its altitude to base ratio is 2 -/
theorem parallelogram_altitude_base_ratio : 
  ∀ (area base altitude : ℝ), 
  area = 242 ∧ base = 11 ∧ area = base * altitude → 
  altitude / base = 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l3885_388562


namespace NUMINAMATH_CALUDE_salad_cost_l3885_388533

/-- The cost of the salad given breakfast and lunch costs -/
theorem salad_cost (muffin_cost coffee_cost soup_cost lemonade_cost : ℝ)
  (h1 : muffin_cost = 2)
  (h2 : coffee_cost = 4)
  (h3 : soup_cost = 3)
  (h4 : lemonade_cost = 0.75)
  (h5 : muffin_cost + coffee_cost + 3 = soup_cost + lemonade_cost + (muffin_cost + coffee_cost)) :
  soup_cost + lemonade_cost + 3 - (soup_cost + lemonade_cost) = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_salad_cost_l3885_388533


namespace NUMINAMATH_CALUDE_cuboid_surface_area_formula_l3885_388576

/-- The surface area of a cuboid with edges of length a, b, and c. -/
def cuboidSurfaceArea (a b c : ℝ) : ℝ := 2 * a * b + 2 * b * c + 2 * a * c

/-- Theorem: The surface area of a cuboid with edges of length a, b, and c
    is equal to 2ab + 2bc + 2ac. -/
theorem cuboid_surface_area_formula (a b c : ℝ) :
  cuboidSurfaceArea a b c = 2 * a * b + 2 * b * c + 2 * a * c := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_formula_l3885_388576


namespace NUMINAMATH_CALUDE_bicycle_trip_average_speed_l3885_388518

/-- Proves that for a bicycle trip with two parts:
    1. 10 km at 12 km/hr
    2. 12 km at 10 km/hr
    The average speed for the entire trip is 660/61 km/hr. -/
theorem bicycle_trip_average_speed :
  let distance1 : ℝ := 10
  let speed1 : ℝ := 12
  let distance2 : ℝ := 12
  let speed2 : ℝ := 10
  let total_distance : ℝ := distance1 + distance2
  let total_time : ℝ := distance1 / speed1 + distance2 / speed2
  let average_speed : ℝ := total_distance / total_time
  average_speed = 660 / 61 := by
sorry

end NUMINAMATH_CALUDE_bicycle_trip_average_speed_l3885_388518


namespace NUMINAMATH_CALUDE_power_seven_135_mod_12_l3885_388580

theorem power_seven_135_mod_12 : 7^135 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_135_mod_12_l3885_388580


namespace NUMINAMATH_CALUDE_child_b_share_l3885_388557

def total_money : ℕ := 12000
def num_children : ℕ := 5
def ratio : List ℕ := [2, 3, 4, 5, 6]

theorem child_b_share :
  let total_parts := ratio.sum
  let part_value := total_money / total_parts
  let child_b_parts := ratio[1]
  child_b_parts * part_value = 1800 := by sorry

end NUMINAMATH_CALUDE_child_b_share_l3885_388557


namespace NUMINAMATH_CALUDE_sum_of_squares_cubic_roots_l3885_388542

theorem sum_of_squares_cubic_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 5 * p - 7 = 0) →
  (3 * q^3 - 2 * q^2 + 5 * q - 7 = 0) →
  (3 * r^3 - 2 * r^2 + 5 * r - 7 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_cubic_roots_l3885_388542


namespace NUMINAMATH_CALUDE_grassy_plot_length_l3885_388532

/-- Represents the dimensions and cost of a rectangular grassy plot with a gravel path. -/
structure GrassyPlot where
  width : ℝ  -- Width of the grassy plot in meters
  pathWidth : ℝ  -- Width of the gravel path in meters
  gravelCost : ℝ  -- Cost of gravelling in rupees
  gravelRate : ℝ  -- Cost of gravelling per square meter in rupees

/-- Calculates the length of the grassy plot given its specifications. -/
def calculateLength (plot : GrassyPlot) : ℝ :=
  -- Implementation not provided as per instructions
  sorry

/-- Theorem stating that given the specified conditions, the length of the grassy plot is 100 meters. -/
theorem grassy_plot_length 
  (plot : GrassyPlot) 
  (h1 : plot.width = 65) 
  (h2 : plot.pathWidth = 2.5) 
  (h3 : plot.gravelCost = 425) 
  (h4 : plot.gravelRate = 0.5) : 
  calculateLength plot = 100 := by
  sorry

end NUMINAMATH_CALUDE_grassy_plot_length_l3885_388532


namespace NUMINAMATH_CALUDE_damaged_chair_percentage_is_40_l3885_388597

/-- Represents the number of office chairs initially -/
def initial_chairs : ℕ := 80

/-- Represents the number of legs each chair has -/
def legs_per_chair : ℕ := 5

/-- Represents the number of round tables -/
def tables : ℕ := 20

/-- Represents the number of legs each table has -/
def legs_per_table : ℕ := 3

/-- Represents the total number of legs remaining after damage -/
def remaining_legs : ℕ := 300

/-- Calculates the percentage of chairs damaged and disposed of -/
def damaged_chair_percentage : ℚ :=
  let total_initial_legs := initial_chairs * legs_per_chair + tables * legs_per_table
  let disposed_legs := total_initial_legs - remaining_legs
  let disposed_chairs := disposed_legs / legs_per_chair
  (disposed_chairs : ℚ) / initial_chairs * 100

/-- Theorem stating that the percentage of chairs damaged and disposed of is 40% -/
theorem damaged_chair_percentage_is_40 :
  damaged_chair_percentage = 40 := by sorry

end NUMINAMATH_CALUDE_damaged_chair_percentage_is_40_l3885_388597


namespace NUMINAMATH_CALUDE_investment_ratio_from_profit_ratio_and_time_l3885_388535

/-- Given two partners p and q, proves that if the ratio of their profits is 7:10,
    p invests for 7 months, and q invests for 14 months,
    then the ratio of their investments is 7:5. -/
theorem investment_ratio_from_profit_ratio_and_time (p q : ℝ) :
  (p * 7) / (q * 14) = 7 / 10 → p / q = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_from_profit_ratio_and_time_l3885_388535


namespace NUMINAMATH_CALUDE_peanuts_in_box_l3885_388587

/-- Given a box with an initial number of peanuts and an additional number of peanuts added,
    calculate the total number of peanuts in the box. -/
def total_peanuts (initial : Nat) (added : Nat) : Nat :=
  initial + added

/-- Theorem stating that if there are initially 4 peanuts in a box and 8 more are added,
    the total number of peanuts in the box is 12. -/
theorem peanuts_in_box : total_peanuts 4 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l3885_388587


namespace NUMINAMATH_CALUDE_cicely_hundredth_birthday_l3885_388553

def cicely_birthday_problem (birth_year : ℕ) (twenty_first_year : ℕ) (hundredth_year : ℕ) : Prop :=
  (twenty_first_year - birth_year = 21) ∧ 
  (twenty_first_year = 1939) ∧
  (hundredth_year - birth_year = 100)

theorem cicely_hundredth_birthday : 
  ∃ (birth_year : ℕ), cicely_birthday_problem birth_year 1939 2018 := by
  sorry

end NUMINAMATH_CALUDE_cicely_hundredth_birthday_l3885_388553


namespace NUMINAMATH_CALUDE_solution_to_equation_l3885_388572

theorem solution_to_equation : ∃ (x y : ℝ), 2 * x - y = 5 ∧ x = 3 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3885_388572


namespace NUMINAMATH_CALUDE_towel_rate_proof_l3885_388556

/-- Proves that given the specified towel purchases and average price, the unknown rate must be 300. -/
theorem towel_rate_proof (price1 price2 avg_price : ℕ) (count1 count2 count_unknown : ℕ) :
  price1 = 100 →
  price2 = 150 →
  avg_price = 165 →
  count1 = 3 →
  count2 = 5 →
  count_unknown = 2 →
  ∃ (unknown_rate : ℕ),
    (count1 * price1 + count2 * price2 + count_unknown * unknown_rate) / (count1 + count2 + count_unknown) = avg_price ∧
    unknown_rate = 300 :=
by sorry

end NUMINAMATH_CALUDE_towel_rate_proof_l3885_388556


namespace NUMINAMATH_CALUDE_gas_station_candy_boxes_l3885_388529

theorem gas_station_candy_boxes : 
  let chocolate : Real := 3.5
  let sugar : Real := 5.25
  let gum : Real := 2.75
  let licorice : Real := 4.5
  let sour : Real := 7.125
  chocolate + sugar + gum + licorice + sour = 23.125 := by
  sorry

end NUMINAMATH_CALUDE_gas_station_candy_boxes_l3885_388529


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3885_388506

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500 ∧ 
   ∀ m : ℕ, m > n → m * (m + 1) ≥ 500) → 
  n + (n + 1) = 43 := by
sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3885_388506


namespace NUMINAMATH_CALUDE_max_chord_length_l3885_388547

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define a point on the parabola
structure Point := (x : ℝ) (y : ℝ)

-- Define a chord on the parabola
structure Chord := (A : Point) (B : Point)

-- Define the condition for the midpoint of the chord
def midpointCondition (c : Chord) : Prop := (c.A.y + c.B.y) / 2 = 4

-- Define the length of the chord
def chordLength (c : Chord) : ℝ := abs (c.A.x - c.B.x)

-- Theorem statement
theorem max_chord_length :
  ∀ c : Chord,
  parabola c.A.x c.A.y →
  parabola c.B.x c.B.y →
  midpointCondition c →
  ∃ maxLength : ℝ, maxLength = 12 ∧ ∀ otherChord : Chord,
    parabola otherChord.A.x otherChord.A.y →
    parabola otherChord.B.x otherChord.B.y →
    midpointCondition otherChord →
    chordLength otherChord ≤ maxLength :=
sorry

end NUMINAMATH_CALUDE_max_chord_length_l3885_388547


namespace NUMINAMATH_CALUDE_ice_cream_shop_sales_l3885_388516

/-- Given a ratio of sugar cones to waffle cones and the number of waffle cones sold,
    calculate the number of sugar cones sold. -/
def sugar_cones_sold (sugar_ratio : ℕ) (waffle_ratio : ℕ) (waffle_cones : ℕ) : ℕ :=
  (sugar_ratio * waffle_cones) / waffle_ratio

/-- Theorem stating that given the specific ratio and number of waffle cones,
    the number of sugar cones sold is 45. -/
theorem ice_cream_shop_sales : sugar_cones_sold 5 4 36 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_shop_sales_l3885_388516


namespace NUMINAMATH_CALUDE_douglas_fir_price_l3885_388588

theorem douglas_fir_price (total_trees : ℕ) (douglas_fir_count : ℕ) (ponderosa_price : ℕ) (total_paid : ℕ) :
  total_trees = 850 →
  douglas_fir_count = 350 →
  ponderosa_price = 225 →
  total_paid = 217500 →
  ∃ (douglas_price : ℕ),
    douglas_price * douglas_fir_count + ponderosa_price * (total_trees - douglas_fir_count) = total_paid ∧
    douglas_price = 300 :=
by sorry

end NUMINAMATH_CALUDE_douglas_fir_price_l3885_388588


namespace NUMINAMATH_CALUDE_like_terms_exponents_l3885_388525

theorem like_terms_exponents (m n : ℤ) : 
  (∃ (x y : ℝ), 2 * x^(2*m) * y^6 = -3 * x^8 * y^(2*n)) → m = 4 ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l3885_388525


namespace NUMINAMATH_CALUDE_susan_third_turn_move_l3885_388523

/-- A board game with the following properties:
  * The game board has 48 spaces from start to finish
  * A player moves 8 spaces forward on the first turn
  * On the second turn, the player moves 2 spaces forward but then 5 spaces backward
  * After the third turn, the player needs to move 37 more spaces to win
-/
structure BoardGame where
  total_spaces : Nat
  first_turn_move : Nat
  second_turn_forward : Nat
  second_turn_backward : Nat
  spaces_left_after_third_turn : Nat

/-- The specific game Susan is playing -/
def susans_game : BoardGame :=
  { total_spaces := 48
  , first_turn_move := 8
  , second_turn_forward := 2
  , second_turn_backward := 5
  , spaces_left_after_third_turn := 37 }

/-- Calculate the number of spaces moved on the third turn -/
def third_turn_move (game : BoardGame) : Nat :=
  game.total_spaces -
  (game.first_turn_move + game.second_turn_forward - game.second_turn_backward) -
  game.spaces_left_after_third_turn

/-- Theorem: Susan moved 6 spaces on the third turn -/
theorem susan_third_turn_move :
  third_turn_move susans_game = 6 := by
  sorry

end NUMINAMATH_CALUDE_susan_third_turn_move_l3885_388523


namespace NUMINAMATH_CALUDE_goose_eggs_count_l3885_388520

theorem goose_eggs_count (
  total_eggs : ℕ
) (
  hatched_ratio : Real
) (
  first_month_survival_ratio : Real
) (
  first_year_death_ratio : Real
) (
  first_year_survivors : ℕ
) (
  h1 : hatched_ratio = 1 / 4
) (
  h2 : first_month_survival_ratio = 4 / 5
) (
  h3 : first_year_death_ratio = 2 / 5
) (
  h4 : first_year_survivors = 120
) (
  h5 : (hatched_ratio * first_month_survival_ratio * (1 - first_year_death_ratio) * total_eggs : Real) = first_year_survivors
) : total_eggs = 800 := by
  sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l3885_388520


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3885_388579

theorem lcm_of_ratio_and_hcf (a b c : ℕ+) : 
  (∃ (k : ℕ+), a = 2 * k ∧ b = 3 * k ∧ c = 5 * k) → 
  Nat.gcd a (Nat.gcd b c) = 6 →
  Nat.lcm a (Nat.lcm b c) = 180 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3885_388579


namespace NUMINAMATH_CALUDE_common_difference_is_two_l3885_388583

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the 4th and 6th terms is 10 -/
  sum_4_6 : a₁ + 3*d + (a₁ + 5*d) = 10
  /-- The sum of the first 5 terms is 5 -/
  sum_5 : 5*a₁ + 10*d = 5

/-- The common difference of the arithmetic sequence is 2 -/
theorem common_difference_is_two (seq : ArithmeticSequence) : seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l3885_388583


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l3885_388577

/-- A quadratic equation is of the form ax² + bx + c = 0, where a ≠ 0 --/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² + 2x = 0 is a quadratic equation --/
theorem equation_is_quadratic :
  is_quadratic_equation (λ x => x^2 + 2*x) :=
sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l3885_388577


namespace NUMINAMATH_CALUDE_plane_perpendicular_condition_l3885_388515

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Plane → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_condition 
  (a b : Line) (α β γ : Plane) :
  perp α γ → para γ β → perp α β :=
by sorry

end NUMINAMATH_CALUDE_plane_perpendicular_condition_l3885_388515


namespace NUMINAMATH_CALUDE_bus_cost_a_to_b_l3885_388546

/-- The cost to travel by bus between two points -/
def busCost (distance : ℝ) (costPerKm : ℝ) : ℝ :=
  distance * costPerKm

/-- Theorem: The cost to travel by bus from A to B is $900 -/
theorem bus_cost_a_to_b : 
  let distanceAB : ℝ := 4500
  let busCostPerKm : ℝ := 0.20
  busCost distanceAB busCostPerKm = 900 := by sorry

end NUMINAMATH_CALUDE_bus_cost_a_to_b_l3885_388546


namespace NUMINAMATH_CALUDE_symmetric_function_equality_l3885_388509

-- Define a function that is symmetric with respect to x = 1
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 - x) = f x

-- Define the theorem
theorem symmetric_function_equality (f : ℝ → ℝ) (h : SymmetricFunction f) :
  ∃! a : ℝ, f (a - 1) = f 5 ∧ a = 6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_equality_l3885_388509


namespace NUMINAMATH_CALUDE_present_ages_l3885_388566

/-- Represents the ages of Rahul, Deepak, and Karan -/
structure Ages where
  rahul : ℕ
  deepak : ℕ
  karan : ℕ

/-- The present age ratio between Rahul, Deepak, and Karan -/
def age_ratio (ages : Ages) : Prop :=
  4 * ages.deepak = 3 * ages.rahul ∧ 5 * ages.deepak = 3 * ages.karan

/-- In 8 years, the sum of Rahul's and Deepak's ages will equal Karan's age -/
def future_age_sum (ages : Ages) : Prop :=
  ages.rahul + ages.deepak + 16 = ages.karan

/-- Rahul's age after 6 years will be 26 years -/
def rahul_future_age (ages : Ages) : Prop :=
  ages.rahul + 6 = 26

/-- The theorem to be proved -/
theorem present_ages (ages : Ages) 
  (h1 : age_ratio ages) 
  (h2 : future_age_sum ages) 
  (h3 : rahul_future_age ages) : 
  ages.deepak = 15 ∧ ages.karan = 51 := by
  sorry

end NUMINAMATH_CALUDE_present_ages_l3885_388566


namespace NUMINAMATH_CALUDE_fifth_odd_multiple_of_five_under_hundred_fifth_odd_multiple_of_five_under_hundred_proof_l3885_388519

theorem fifth_odd_multiple_of_five_under_hundred : ℕ → Prop :=
  fun n =>
    (∃ k, n = 5 * (2 * k + 1)) ∧  -- n is odd and a multiple of 5
    n < 100 ∧  -- n is less than 100
    (∃ m, m = 5 ∧  -- m is the count of numbers satisfying the conditions
      ∀ i, i < n →
        (∃ j, i = 5 * (2 * j + 1)) ∧ i < 100 →
        i ≤ m * 9) →  -- there are exactly 4 numbers before n satisfying the conditions
    n = 45  -- the fifth such number is 45

-- The proof of this theorem is omitted
theorem fifth_odd_multiple_of_five_under_hundred_proof : fifth_odd_multiple_of_five_under_hundred 45 := by
  sorry

end NUMINAMATH_CALUDE_fifth_odd_multiple_of_five_under_hundred_fifth_odd_multiple_of_five_under_hundred_proof_l3885_388519


namespace NUMINAMATH_CALUDE_bus_ride_is_75_minutes_l3885_388502

/-- Calculates the bus ride duration given the total trip time, train ride duration, and walking time. -/
def bus_ride_duration (total_trip_time : ℕ) (train_ride_duration : ℕ) (walking_time : ℕ) : ℕ :=
  let total_minutes := total_trip_time * 60
  let train_minutes := train_ride_duration * 60
  let waiting_time := walking_time * 2
  total_minutes - train_minutes - waiting_time - walking_time

/-- Proves that given the specified conditions, the bus ride duration is 75 minutes. -/
theorem bus_ride_is_75_minutes :
  bus_ride_duration 8 6 15 = 75 := by
  sorry

#eval bus_ride_duration 8 6 15

end NUMINAMATH_CALUDE_bus_ride_is_75_minutes_l3885_388502


namespace NUMINAMATH_CALUDE_money_left_calculation_l3885_388570

def initial_amount : ℝ := 200

def notebook_cost : ℝ := 4
def book_cost : ℝ := 12
def pen_cost : ℝ := 2
def sticker_pack_cost : ℝ := 6
def shoes_cost : ℝ := 40
def tshirt_cost : ℝ := 18

def notebooks_bought : ℕ := 7
def books_bought : ℕ := 2
def pens_bought : ℕ := 5
def sticker_packs_bought : ℕ := 3

def sales_tax_rate : ℝ := 0.05

def lunch_cost : ℝ := 15
def tip_amount : ℝ := 3
def transportation_cost : ℝ := 8
def charity_amount : ℝ := 10

def total_mall_purchase_cost : ℝ :=
  notebook_cost * notebooks_bought +
  book_cost * books_bought +
  pen_cost * pens_bought +
  sticker_pack_cost * sticker_packs_bought +
  shoes_cost +
  tshirt_cost

def total_mall_cost_with_tax : ℝ :=
  total_mall_purchase_cost * (1 + sales_tax_rate)

def total_expenses : ℝ :=
  total_mall_cost_with_tax +
  lunch_cost +
  tip_amount +
  transportation_cost +
  charity_amount

theorem money_left_calculation :
  initial_amount - total_expenses = 19.10 := by
  sorry

end NUMINAMATH_CALUDE_money_left_calculation_l3885_388570


namespace NUMINAMATH_CALUDE_al_investment_l3885_388504

/-- Represents the investment scenario with four participants -/
structure Investment where
  al : ℝ
  betty : ℝ
  clare : ℝ
  dave : ℝ

/-- Defines the conditions of the investment problem -/
def valid_investment (i : Investment) : Prop :=
  i.al > 0 ∧ i.betty > 0 ∧ i.clare > 0 ∧ i.dave > 0 ∧  -- Each begins with a positive amount
  i.al ≠ i.betty ∧ i.al ≠ i.clare ∧ i.al ≠ i.dave ∧    -- Each begins with a different amount
  i.betty ≠ i.clare ∧ i.betty ≠ i.dave ∧ i.clare ≠ i.dave ∧
  i.al + i.betty + i.clare + i.dave = 2000 ∧           -- Total initial investment
  (i.al - 150) + (3 * i.betty) + (3 * i.clare) + (i.dave - 50) = 2500  -- Total after one year

/-- Theorem stating that under the given conditions, Al's original portion was 450 -/
theorem al_investment (i : Investment) (h : valid_investment i) : i.al = 450 := by
  sorry


end NUMINAMATH_CALUDE_al_investment_l3885_388504


namespace NUMINAMATH_CALUDE_valid_lineup_count_l3885_388575

/- Define the total number of players -/
def total_players : ℕ := 18

/- Define the number of quadruplets -/
def quadruplets : ℕ := 4

/- Define the number of starters to select -/
def starters : ℕ := 8

/- Define the function to calculate combinations -/
def combination (n k : ℕ) : ℕ := (Nat.choose n k)

/- Theorem statement -/
theorem valid_lineup_count :
  combination total_players starters - combination (total_players - quadruplets) (starters - quadruplets) =
  42757 := by sorry

end NUMINAMATH_CALUDE_valid_lineup_count_l3885_388575


namespace NUMINAMATH_CALUDE_company_workforce_l3885_388569

theorem company_workforce (initial_employees : ℕ) : 
  (initial_employees * 6 / 10 : ℚ) = (initial_employees + 20) * 11 / 20 →
  initial_employees + 20 = 240 := by
  sorry

end NUMINAMATH_CALUDE_company_workforce_l3885_388569


namespace NUMINAMATH_CALUDE_jumping_contest_l3885_388521

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) : 
  grasshopper_jump = 25 →
  frog_jump = grasshopper_jump + 32 →
  mouse_jump = frog_jump - 26 →
  mouse_jump = 31 := by
  sorry

end NUMINAMATH_CALUDE_jumping_contest_l3885_388521


namespace NUMINAMATH_CALUDE_break_even_point_manuals_l3885_388514

/-- The break-even point for manual production -/
theorem break_even_point_manuals :
  let average_cost (Q : ℝ) := 100 + 100000 / Q
  let planned_price := 300
  ∃ Q : ℝ, Q > 0 ∧ average_cost Q = planned_price ∧ Q = 500 :=
by sorry

end NUMINAMATH_CALUDE_break_even_point_manuals_l3885_388514


namespace NUMINAMATH_CALUDE_tims_weekly_water_consumption_l3885_388578

/-- Calculates Tim's weekly water consumption in ounces -/
theorem tims_weekly_water_consumption :
  let quart_to_oz : ℚ → ℚ := (· * 32)
  let daily_bottle_oz := 2 * quart_to_oz 1.5
  let daily_total_oz := daily_bottle_oz + 20
  let weekly_oz := 7 * daily_total_oz
  weekly_oz = 812 := by sorry

end NUMINAMATH_CALUDE_tims_weekly_water_consumption_l3885_388578


namespace NUMINAMATH_CALUDE_centroid_circle_area_l3885_388574

/-- Given a circle with diameter 'd', the area of the circle traced by the centroid of a triangle
    formed by the diameter and a point on the circumference is (25/900) times the area of the original circle. -/
theorem centroid_circle_area (d : ℝ) (h : d > 0) :
  ∃ (A_centroid A_circle : ℝ),
    A_circle = π * (d/2)^2 ∧
    A_centroid = π * (d/6)^2 ∧
    A_centroid = (25/900) * A_circle :=
by sorry

end NUMINAMATH_CALUDE_centroid_circle_area_l3885_388574


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3885_388582

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive side lengths
  a^2 + b^2 = c^2 ∧        -- Pythagorean theorem (right triangle)
  a + b + c = 40 ∧         -- perimeter condition
  (1/2) * a * b = 24 ∧     -- area condition
  c = 18.8 := by            -- hypotenuse length
  sorry


end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3885_388582


namespace NUMINAMATH_CALUDE_lower_limit_proof_l3885_388561

def is_prime (n : ℕ) : Prop := sorry

def count_primes_between (a b : ℝ) : ℕ := sorry

theorem lower_limit_proof : 
  ∀ x : ℕ, x ≤ 19 ↔ count_primes_between x (87/5) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_lower_limit_proof_l3885_388561


namespace NUMINAMATH_CALUDE_total_acorns_formula_l3885_388591

/-- The total number of acorns for Shawna, Sheila, Danny, and Ella -/
def total_acorns (x y : ℝ) : ℝ :=
  let shawna := x
  let sheila := 5.3 * x
  let danny := sheila + y
  let ella := 2 * (danny - shawna)
  shawna + sheila + danny + ella

/-- Theorem stating the total number of acorns -/
theorem total_acorns_formula (x y : ℝ) :
  total_acorns x y = 20.2 * x + 3 * y := by
  sorry

end NUMINAMATH_CALUDE_total_acorns_formula_l3885_388591


namespace NUMINAMATH_CALUDE_negation_of_implication_l3885_388554

theorem negation_of_implication (m : ℝ) : 
  (¬(m > 0 → ∃ x : ℝ, x^2 + x - m = 0)) ↔ 
  (m ≤ 0 → ¬∃ x : ℝ, x^2 + x - m = 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3885_388554


namespace NUMINAMATH_CALUDE_number_puzzle_l3885_388558

theorem number_puzzle : ∃ x : ℝ, x = 280 ∧ (x / 5 + 4 = x / 4 - 10) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3885_388558


namespace NUMINAMATH_CALUDE_quadratic_other_x_intercept_l3885_388537

/-- Given a quadratic function f(x) = ax² + bx + c with vertex (5,9) and
    one x-intercept at (0,0), prove that the x-coordinate of the other x-intercept is 10. -/
theorem quadratic_other_x_intercept
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : f 5 = 9 ∧ (∀ y, f 5 ≤ f y))  -- Vertex at (5,9)
  (h3 : f 0 = 0)  -- x-intercept at (0,0)
  : ∃ x, x ≠ 0 ∧ f x = 0 ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_other_x_intercept_l3885_388537


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l3885_388539

theorem divisibility_by_seven (n a b d : ℤ) 
  (h1 : 0 ≤ b ∧ b ≤ 9)
  (h2 : 0 ≤ a)
  (h3 : n = 10 * a + b)
  (h4 : d = a - 2 * b) :
  7 ∣ n ↔ 7 ∣ d := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l3885_388539


namespace NUMINAMATH_CALUDE_gala_dinner_seating_l3885_388511

/-- The number of couples to be seated -/
def num_couples : ℕ := 6

/-- The total number of people to be seated -/
def total_people : ℕ := 2 * num_couples

/-- The number of ways to arrange the husbands -/
def husband_arrangements : ℕ := (total_people - 1).factorial

/-- The number of equivalent arrangements due to rotation and reflection -/
def equivalent_arrangements : ℕ := 2 * total_people

/-- The number of unique seating arrangements -/
def unique_arrangements : ℕ := husband_arrangements / equivalent_arrangements

theorem gala_dinner_seating :
  unique_arrangements = 5760 :=
sorry

end NUMINAMATH_CALUDE_gala_dinner_seating_l3885_388511


namespace NUMINAMATH_CALUDE_bookcase_organization_l3885_388567

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves containing mystery books -/
def mystery_shelves : ℕ := 3

/-- The number of shelves containing picture books -/
def picture_shelves : ℕ := 5

/-- The total number of books in the bookcase -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem bookcase_organization :
  total_books = 72 := by sorry

end NUMINAMATH_CALUDE_bookcase_organization_l3885_388567


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3885_388590

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), A = 31/9 ∧ B = 5/9 ∧
  ∀ (x : ℝ), x ≠ 7 ∧ x ≠ -2 →
  (4*x^2 + 7*x + 3) / (x^2 - 5*x - 14) = A / (x - 7) + B / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3885_388590


namespace NUMINAMATH_CALUDE_smallest_b_value_l3885_388510

theorem smallest_b_value (a b c : ℚ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : 2 * b = a + c)  -- arithmetic sequence condition
  (h4 : c^2 = a * b)    -- geometric sequence condition
  : b ≥ (1/2 : ℚ) ∧ ∃ (a' b' c' : ℚ), 
    a' < b' ∧ b' < c' ∧ 
    2 * b' = a' + c' ∧ 
    c'^2 = a' * b' ∧ 
    b' = (1/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3885_388510


namespace NUMINAMATH_CALUDE_solve_tangerines_l3885_388593

def tangerines_problem (initial_eaten : ℕ) (later_eaten : ℕ) : Prop :=
  ∃ (total : ℕ), 
    (initial_eaten + later_eaten = total) ∧ 
    (total - initial_eaten - later_eaten = 0)

theorem solve_tangerines : tangerines_problem 10 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_tangerines_l3885_388593


namespace NUMINAMATH_CALUDE_value_of_y_l3885_388543

theorem value_of_y : ∃ y : ℚ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l3885_388543


namespace NUMINAMATH_CALUDE_alvin_marbles_lost_l3885_388568

/-- Proves that Alvin lost 18 marbles in the first game -/
theorem alvin_marbles_lost (initial_marbles : ℕ) (won_marbles : ℕ) (final_marbles : ℕ) 
  (h1 : initial_marbles = 57)
  (h2 : won_marbles = 25)
  (h3 : final_marbles = 64) :
  initial_marbles - (final_marbles - won_marbles) = 18 := by
  sorry

#check alvin_marbles_lost

end NUMINAMATH_CALUDE_alvin_marbles_lost_l3885_388568


namespace NUMINAMATH_CALUDE_range_of_m_l3885_388501

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 1) * x + 2

-- Define the condition that the solution set is ℝ
def solution_set_is_real (m : ℝ) : Prop :=
  ∀ x, f m x > 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  solution_set_is_real m ↔ (1 ≤ m ∧ m < 9) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3885_388501


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_count_l3885_388507

/-- The number of digits in the numbers we're considering -/
def num_digits : ℕ := 6

/-- The total number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 9 * 10^5

/-- The number of 6-digit numbers with no zeros -/
def six_digit_numbers_no_zero : ℕ := 9^6

/-- The number of 6-digit numbers with at least one zero -/
def six_digit_numbers_with_zero : ℕ := total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_zero_count :
  six_digit_numbers_with_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_count_l3885_388507


namespace NUMINAMATH_CALUDE_proposition_analysis_l3885_388550

theorem proposition_analysis (P Q : Prop) 
  (h_P : P ↔ (2 + 2 = 5))
  (h_Q : Q ↔ (3 > 2)) : 
  (¬(P ∧ Q)) ∧ (¬P) := by
  sorry

end NUMINAMATH_CALUDE_proposition_analysis_l3885_388550
