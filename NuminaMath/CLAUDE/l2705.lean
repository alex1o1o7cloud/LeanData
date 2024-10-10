import Mathlib

namespace not_p_sufficient_not_necessary_for_not_q_l2705_270529

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| ≤ 4
def q (x : ℝ) : Prop := x^2 < 5*x - 6

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l2705_270529


namespace marbles_sum_theorem_l2705_270546

/-- The number of yellow marbles Mary and Joan have in total -/
def total_marbles (mary_marbles joan_marbles : ℕ) : ℕ :=
  mary_marbles + joan_marbles

/-- Theorem stating that Mary and Joan have 12 yellow marbles in total -/
theorem marbles_sum_theorem :
  total_marbles 9 3 = 12 := by sorry

end marbles_sum_theorem_l2705_270546


namespace tan_periodic_angle_l2705_270538

theorem tan_periodic_angle (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (1720 * π / 180) → n = -80 :=
by sorry

end tan_periodic_angle_l2705_270538


namespace modulus_of_complex_number_l2705_270585

theorem modulus_of_complex_number : 
  let i : ℂ := Complex.I
  let z : ℂ := 2 * i + 2 / (1 + i)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_complex_number_l2705_270585


namespace ben_winning_strategy_l2705_270571

/-- Represents the state of a card (0 for letter, 1 for number) -/
inductive CardState
| Letter : CardState
| Number : CardState

/-- Represents a configuration of cards -/
def Configuration := Vector CardState 2019

/-- Represents a move in the game -/
structure Move where
  position : Fin 2019

/-- Applies a move to a configuration -/
def applyMove (config : Configuration) (move : Move) : Configuration :=
  sorry

/-- Checks if all cards are showing numbers -/
def allNumbers (config : Configuration) : Prop :=
  sorry

theorem ben_winning_strategy :
  ∀ (initial_config : Configuration),
  ∃ (moves : List Move),
    moves.length ≤ 2019 ∧
    allNumbers (moves.foldl applyMove initial_config) :=
  sorry

end ben_winning_strategy_l2705_270571


namespace min_value_sum_squares_l2705_270558

theorem min_value_sum_squares (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 8) 
  (h2 : e * f * g * h = 16) : 
  ∀ x y z w : ℝ, x * y * z * w = 8 → 
  ∀ p q r s : ℝ, p * q * r * s = 16 →
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 
  (x * p)^2 + (y * q)^2 + (z * r)^2 + (w * s)^2 ∧
  (∃ x y z w p q r s : ℝ, x * y * z * w = 8 ∧ p * q * r * s = 16 ∧
  (x * p)^2 + (y * q)^2 + (z * r)^2 + (w * s)^2 = 32) :=
sorry

end min_value_sum_squares_l2705_270558


namespace unique_quadratic_solution_l2705_270560

/-- Given a quadratic equation ax^2 + 15x + c = 0 with exactly one solution,
    where a + c = 36 and a < c, prove that a = (36 - √1071) / 2 and c = (36 + √1071) / 2 -/
theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 15 * x + c = 0) → 
  (a + c = 36) → 
  (a < c) → 
  (a = (36 - Real.sqrt 1071) / 2 ∧ c = (36 + Real.sqrt 1071) / 2) := by
sorry

end unique_quadratic_solution_l2705_270560


namespace toms_average_strokes_l2705_270592

/-- Represents the number of rounds Tom plays -/
def rounds : ℕ := 9

/-- Represents the par value per hole -/
def par_per_hole : ℕ := 3

/-- Represents the number of strokes Tom was over par -/
def strokes_over_par : ℕ := 9

/-- Calculates Tom's average number of strokes per hole -/
def average_strokes_per_hole : ℚ :=
  (rounds * par_per_hole + strokes_over_par) / rounds

/-- Theorem stating that Tom's average number of strokes per hole is 4 -/
theorem toms_average_strokes :
  average_strokes_per_hole = 4 := by sorry

end toms_average_strokes_l2705_270592


namespace sine_function_period_l2705_270527

/-- Given a function f(x) = √3 * sin(ωx + φ) where ω > 0, 
    if the distance between adjacent symmetry axes of the graph is 2π, 
    then ω = 1/2 -/
theorem sine_function_period (ω φ : ℝ) (h_ω_pos : ω > 0) :
  (∀ x : ℝ, ∃ k : ℤ, (x + 2 * π) = x + 2 * k * π / ω) →
  ω = 1 / 2 := by
sorry

end sine_function_period_l2705_270527


namespace fraction_multiplication_equality_l2705_270562

theorem fraction_multiplication_equality : 
  (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5040.000000000001 = 756.0000000000001 := by
  sorry

end fraction_multiplication_equality_l2705_270562


namespace abc_value_l2705_270582

theorem abc_value (a b c : ℂ) 
  (h1 : a * b + 5 * b = -20)
  (h2 : b * c + 5 * c = -20)
  (h3 : c * a + 5 * a = -20) : 
  a * b * c = -100 := by
sorry

end abc_value_l2705_270582


namespace february_first_is_monday_l2705_270590

/-- Represents the days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the month of February in a specific year -/
structure February where
  /-- The day of the week that February 1 falls on -/
  first_day : Weekday
  /-- The number of days in February -/
  num_days : Nat
  /-- The number of Mondays in February -/
  num_mondays : Nat
  /-- The number of Thursdays in February -/
  num_thursdays : Nat

/-- Theorem stating that if February has exactly four Mondays and four Thursdays,
    then February 1 must fall on a Monday -/
theorem february_first_is_monday (feb : February) 
  (h1 : feb.num_mondays = 4)
  (h2 : feb.num_thursdays = 4) : 
  feb.first_day = Weekday.Monday := by
  sorry


end february_first_is_monday_l2705_270590


namespace piggy_bank_dimes_l2705_270573

/-- Proves that given $5.55 in dimes and quarters, with three more dimes than quarters, the number of dimes is 18 -/
theorem piggy_bank_dimes (total : ℚ) (dimes quarters : ℕ) : 
  total = (5 : ℚ) + (55 : ℚ) / 100 →
  dimes = quarters + 3 →
  (10 : ℚ) * dimes + (25 : ℚ) * quarters = total * 100 →
  dimes = 18 := by
sorry

end piggy_bank_dimes_l2705_270573


namespace find_number_l2705_270520

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 69 ∧ x = 7 := by sorry

end find_number_l2705_270520


namespace quadrilateral_area_in_possible_areas_all_possible_areas_achievable_l2705_270551

/-- Represents a point on the side of a square --/
inductive SidePoint
| A1 | A2 | A3  -- Points on side AB
| B1 | B2 | B3  -- Points on side BC
| C1 | C2 | C3  -- Points on side CD
| D1 | D2 | D3  -- Points on side DA

/-- Represents a quadrilateral formed by choosing points from each side of a square --/
structure Quadrilateral :=
  (p1 : SidePoint)
  (p2 : SidePoint)
  (p3 : SidePoint)
  (p4 : SidePoint)

/-- Calculates the area of a quadrilateral formed by choosing points from each side of a square --/
def area (q : Quadrilateral) : ℝ :=
  sorry  -- The actual calculation would go here

/-- The set of possible areas for quadrilaterals formed in the given square --/
def possible_areas : Set ℝ := {6, 7, 7.5, 8, 8.5, 9, 10}

/-- Theorem stating that the area of any quadrilateral formed in the given square
    must be one of the values in the possible_areas set --/
theorem quadrilateral_area_in_possible_areas (q : Quadrilateral) :
  area q ∈ possible_areas :=
sorry

/-- Theorem stating that every value in the possible_areas set
    is achievable by some quadrilateral in the given square --/
theorem all_possible_areas_achievable :
  ∀ a ∈ possible_areas, ∃ q : Quadrilateral, area q = a :=
sorry

end quadrilateral_area_in_possible_areas_all_possible_areas_achievable_l2705_270551


namespace math_contest_correct_answers_l2705_270503

theorem math_contest_correct_answers 
  (total_problems : ℕ)
  (correct_points : ℤ)
  (incorrect_points : ℤ)
  (total_score : ℤ)
  (min_guesses : ℕ)
  (h1 : total_problems = 15)
  (h2 : correct_points = 6)
  (h3 : incorrect_points = -3)
  (h4 : total_score = 45)
  (h5 : min_guesses ≥ 4)
  : ∃ (correct_answers : ℕ), 
    correct_answers * correct_points + (total_problems - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 10 := by
  sorry

end math_contest_correct_answers_l2705_270503


namespace total_cost_for_20_products_l2705_270501

/-- The total cost function for producing products -/
def total_cost (fixed_cost marginal_cost : ℝ) (n : ℕ) : ℝ :=
  fixed_cost + marginal_cost * n

/-- Theorem: The total cost for producing 20 products is $16000 -/
theorem total_cost_for_20_products
  (fixed_cost : ℝ)
  (marginal_cost : ℝ)
  (h1 : fixed_cost = 12000)
  (h2 : marginal_cost = 200) :
  total_cost fixed_cost marginal_cost 20 = 16000 := by
  sorry

#eval total_cost 12000 200 20

end total_cost_for_20_products_l2705_270501


namespace quadratic_inequality_l2705_270500

theorem quadratic_inequality (x : ℝ) : x^2 - 4*x - 21 < 0 ↔ -3 < x ∧ x < 7 := by
  sorry

end quadratic_inequality_l2705_270500


namespace one_third_of_six_y_plus_three_l2705_270598

theorem one_third_of_six_y_plus_three (y : ℝ) : (1 / 3) * (6 * y + 3) = 2 * y + 1 := by
  sorry

end one_third_of_six_y_plus_three_l2705_270598


namespace altitude_triangle_min_side_range_l2705_270564

/-- A triangle with side lengths a, b, c, perimeter 1, and altitudes that form a new triangle -/
structure AltitudeTriangle where
  a : Real
  b : Real
  c : Real
  perimeter_one : a + b + c = 1
  altitudes_form_triangle : 1/a + 1/b > 1/c ∧ 1/b + 1/c > 1/a ∧ 1/c + 1/a > 1/b
  a_smallest : a ≤ b ∧ a ≤ c

theorem altitude_triangle_min_side_range (t : AltitudeTriangle) :
  (3 - Real.sqrt 5) / 4 < t.a ∧ t.a ≤ 1/3 := by
  sorry

end altitude_triangle_min_side_range_l2705_270564


namespace train_speed_l2705_270524

/-- Calculate the speed of a train given its length and time to pass an observer -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 180) (h2 : time = 9) :
  length / time = 20 := by
  sorry

#check train_speed

end train_speed_l2705_270524


namespace cockroach_search_l2705_270595

/-- The cockroach's search problem -/
theorem cockroach_search (D : ℝ) (h : D > 0) :
  ∃ (path : ℕ → ℝ × ℝ),
    (∀ n, dist (path n) (path (n+1)) ≤ 1) ∧
    (∀ n, dist (path (n+1)) (D, 0) < dist (path n) (D, 0) ∨
          dist (path (n+1)) (D, 0) = dist (path n) (D, 0)) ∧
    (∃ n, path n = (D, 0)) ∧
    (∃ n, path n = (D, 0) ∧ n ≤ ⌊(3/2 * D + 7)⌋) :=
sorry


end cockroach_search_l2705_270595


namespace cartesian_polar_equivalence_l2705_270578

-- Define the set of points in Cartesian coordinates
def cartesian_set : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

-- Define the set of points in polar coordinates
def polar_set : Set (ℝ × ℝ) := {p | Real.sqrt (p.1^2 + p.2^2) = 3}

-- Theorem stating the equivalence of the two sets
theorem cartesian_polar_equivalence : cartesian_set = polar_set := by sorry

end cartesian_polar_equivalence_l2705_270578


namespace one_slice_remains_l2705_270535

/-- Calculates the number of slices of bread remaining after eating and making toast. -/
def remaining_slices (initial_slices : ℕ) (eaten_twice : ℕ) (slices_per_toast : ℕ) (toast_made : ℕ) : ℕ :=
  initial_slices - (2 * eaten_twice) - (slices_per_toast * toast_made)

/-- Theorem stating that given the specific conditions, 1 slice of bread remains. -/
theorem one_slice_remains : remaining_slices 27 3 2 10 = 1 := by
  sorry

#eval remaining_slices 27 3 2 10

end one_slice_remains_l2705_270535


namespace max_min_equation_characterization_l2705_270533

theorem max_min_equation_characterization (x y : ℝ) : 
  max x (x^2) + min y (y^2) = 1 ↔ 
    (y = 1 - x^2 ∧ y ≤ 0) ∨
    (x^2 + y^2 = 1 ∧ ((x ≤ -1 ∨ x > 0) ∧ 0 < y ∧ y < 1)) ∨
    (y^2 = 1 - x ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1) :=
by sorry

end max_min_equation_characterization_l2705_270533


namespace sqrt_6000_approx_l2705_270504

/-- Approximate value of the square root of 6 -/
def sqrt_6_approx : ℝ := 2.45

/-- Approximate value of the square root of 60 -/
def sqrt_60_approx : ℝ := 7.75

/-- Theorem stating that the square root of 6000 is approximately 77.5 -/
theorem sqrt_6000_approx : ∃ (ε : ℝ), ε > 0 ∧ |Real.sqrt 6000 - 77.5| < ε :=
sorry

end sqrt_6000_approx_l2705_270504


namespace intercepted_triangle_area_l2705_270513

/-- The region defined by the inequality |x - 1| + |y - 2| ≤ 2 -/
def diamond_region (x y : ℝ) : Prop :=
  abs (x - 1) + abs (y - 2) ≤ 2

/-- The line y = 3x + 1 -/
def intercepting_line (x y : ℝ) : Prop :=
  y = 3 * x + 1

/-- The triangle intercepted by the line from the diamond region -/
def intercepted_triangle (x y : ℝ) : Prop :=
  diamond_region x y ∧ intercepting_line x y

/-- The area of the intercepted triangle -/
noncomputable def triangle_area : ℝ := 2

theorem intercepted_triangle_area :
  triangle_area = 2 :=
sorry

end intercepted_triangle_area_l2705_270513


namespace resort_flat_fee_is_40_l2705_270523

/-- Represents the pricing scheme of a resort -/
structure ResortPricing where
  flatFee : ℕ  -- Flat fee for the first night
  additionalNightFee : ℕ  -- Fee for each additional night

/-- Calculates the total cost for a stay -/
def totalCost (pricing : ResortPricing) (nights : ℕ) : ℕ :=
  pricing.flatFee + (nights - 1) * pricing.additionalNightFee

/-- Theorem stating the flat fee given the conditions -/
theorem resort_flat_fee_is_40 :
  ∀ (pricing : ResortPricing),
    totalCost pricing 5 = 320 →
    totalCost pricing 8 = 530 →
    pricing.flatFee = 40 := by
  sorry


end resort_flat_fee_is_40_l2705_270523


namespace function_value_range_l2705_270584

theorem function_value_range (x : ℝ) :
  x ∈ Set.Icc 1 4 →
  2 ≤ x^2 - 4*x + 6 ∧ x^2 - 4*x + 6 ≤ 6 :=
by sorry

end function_value_range_l2705_270584


namespace inverse_function_point_l2705_270548

-- Define a function f and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the property of f_inv being the inverse of f
def is_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- State the theorem
theorem inverse_function_point 
  (h1 : is_inverse f f_inv) 
  (h2 : f 3 = -1) : 
  f_inv (-3) = 3 := by
sorry

end inverse_function_point_l2705_270548


namespace plant_growth_thirty_percent_plant_growth_day_l2705_270570

/-- Represents the growth of a plant over time. -/
structure PlantGrowth where
  initialLength : ℝ
  dailyGrowth : ℝ

/-- Calculates the length of the plant on a given day. -/
def plantLengthOnDay (p : PlantGrowth) (day : ℕ) : ℝ :=
  p.initialLength + p.dailyGrowth * (day - 1 : ℝ)

/-- Theorem: The plant grows 30% between the 4th day and the 10th day. -/
theorem plant_growth_thirty_percent (p : PlantGrowth) 
  (h1 : p.initialLength = 11)
  (h2 : p.dailyGrowth = 0.6875) :
  plantLengthOnDay p 10 = plantLengthOnDay p 4 * 1.3 := by
  sorry

/-- Corollary: The 10th day is the first day when the plant's length is at least 30% greater than on the 4th day. -/
theorem plant_growth_day (p : PlantGrowth)
  (h1 : p.initialLength = 11)
  (h2 : p.dailyGrowth = 0.6875) :
  (∀ d : ℕ, d < 10 → plantLengthOnDay p d < plantLengthOnDay p 4 * 1.3) ∧
  plantLengthOnDay p 10 ≥ plantLengthOnDay p 4 * 1.3 := by
  sorry

end plant_growth_thirty_percent_plant_growth_day_l2705_270570


namespace isosceles_triangle_l2705_270563

theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ) : 
  b * Real.cos C = c * Real.cos B → B = C := by sorry

end isosceles_triangle_l2705_270563


namespace regular_polygon_sides_l2705_270566

/-- A regular polygon is a polygon with all sides of equal length. -/
structure RegularPolygon where
  side_length : ℝ
  num_sides : ℕ
  perimeter : ℝ
  perimeter_eq : perimeter = side_length * num_sides

/-- Theorem: A regular polygon with side length 16 cm and perimeter 80 cm has 5 sides. -/
theorem regular_polygon_sides (p : RegularPolygon) 
  (h1 : p.side_length = 16) 
  (h2 : p.perimeter = 80) : 
  p.num_sides = 5 := by
  sorry

end regular_polygon_sides_l2705_270566


namespace cuboid_height_from_cube_l2705_270576

/-- The length of wire needed to make a cube with given edge length -/
def cube_wire_length (edge : ℝ) : ℝ := 12 * edge

/-- The length of wire needed to make a cuboid with given dimensions -/
def cuboid_wire_length (length width height : ℝ) : ℝ :=
  4 * (length + width + height)

theorem cuboid_height_from_cube (cube_edge length width : ℝ) 
  (h_cube_edge : cube_edge = 10)
  (h_length : length = 8)
  (h_width : width = 5) :
  ∃ (height : ℝ), 
    cube_wire_length cube_edge = cuboid_wire_length length width height ∧ 
    height = 17 := by
  sorry

end cuboid_height_from_cube_l2705_270576


namespace det_2x2_matrix_l2705_270569

theorem det_2x2_matrix (x : ℝ) : 
  Matrix.det !![5, x; 4, 3] = 15 - 4 * x := by sorry

end det_2x2_matrix_l2705_270569


namespace composition_equality_l2705_270542

-- Define the functions f and h
def f (m n x : ℝ) : ℝ := m * x + n
def h (p q r x : ℝ) : ℝ := p * x^2 + q * x + r

-- State the theorem
theorem composition_equality (m n p q r : ℝ) :
  (∀ x, f m n (h p q r x) = h p q r (f m n x)) ↔ (m = p ∧ n = 0) := by
  sorry

end composition_equality_l2705_270542


namespace couch_price_after_changes_l2705_270543

theorem couch_price_after_changes (initial_price : ℝ) 
  (h_initial : initial_price = 62500) : 
  let increase_factor := 1.2
  let decrease_factor := 0.8
  let final_factor := (increase_factor ^ 3) * (decrease_factor ^ 3)
  initial_price * final_factor = 55296 := by sorry

end couch_price_after_changes_l2705_270543


namespace extreme_value_cubic_l2705_270559

/-- Given a cubic function f(x) = ax³ + 3x² + 3x + 3,
    if f has an extreme value at x = 1, then a = -3 -/
theorem extreme_value_cubic (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + 3 * x^2 + 3 * x + 3
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = -3 := by
  sorry

end extreme_value_cubic_l2705_270559


namespace range_of_c_l2705_270531

/-- A condition is sufficient but not necessary -/
def SufficientButNotNecessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

theorem range_of_c (a c : ℝ) :
  SufficientButNotNecessary (a ≥ 1/8) (∀ x > 0, 2*x + a/x ≥ c) →
  c ≤ 1 := by
  sorry

end range_of_c_l2705_270531


namespace min_box_value_l2705_270534

theorem min_box_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a*x + b) * (b*x + 2*a) = 36*x^2 + box*x + 72) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  (∃ a' b' box', 
    (∀ x, (a'*x + b') * (b'*x + 2*a') = 36*x^2 + box'*x + 72) ∧
    a' ≠ b' ∧ b' ≠ box' ∧ a' ≠ box' ∧
    box' < box) →
  box ≥ 332 :=
by sorry

end min_box_value_l2705_270534


namespace coin_value_difference_l2705_270537

theorem coin_value_difference (p n d : ℕ) : 
  p + n + d = 3030 →
  p ≥ 1 →
  n ≥ 1 →
  d ≥ 1 →
  (∀ p' n' d' : ℕ, p' + n' + d' = 3030 ∧ p' ≥ 1 ∧ n' ≥ 1 ∧ d' ≥ 1 →
    p' + 5 * n' + 10 * d' ≤ 30286 ∧
    p' + 5 * n' + 10 * d' ≥ 3043) →
  30286 - 3043 = 27243 :=
by sorry

end coin_value_difference_l2705_270537


namespace is_arithmetic_sequence_l2705_270528

def S (n : ℕ) : ℝ := 2 * n + 1

theorem is_arithmetic_sequence :
  ∀ n : ℕ, S (n + 1) - S n = S 1 - S 0 :=
by
  sorry

end is_arithmetic_sequence_l2705_270528


namespace directional_vector_for_line_l2705_270512

/-- A directional vector for a line ax + by + c = 0 is a vector (u, v) such that
    for any point (x, y) on the line, (x + u, y + v) is also on the line. -/
def IsDirectionalVector (a b c : ℝ) (u v : ℝ) : Prop :=
  ∀ x y : ℝ, a * x + b * y + c = 0 → a * (x + u) + b * (y + v) + c = 0

/-- The line 2x + 3y - 1 = 0 -/
def Line (x y : ℝ) : Prop := 2 * x + 3 * y - 1 = 0

/-- Theorem: (1, -2/3) is a directional vector for the line 2x + 3y - 1 = 0 -/
theorem directional_vector_for_line :
  IsDirectionalVector 2 3 (-1) 1 (-2/3) :=
sorry

end directional_vector_for_line_l2705_270512


namespace jordan_fourth_period_blocks_l2705_270583

/-- Represents the number of shots blocked by a hockey goalie in each period of a game --/
structure GoalieBlocks where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculates the total number of shots blocked in a game --/
def totalBlocks (blocks : GoalieBlocks) : ℕ :=
  blocks.first + blocks.second + blocks.third + blocks.fourth

/-- Theorem: Given the conditions of Jordan's game, he blocked 4 shots in the fourth period --/
theorem jordan_fourth_period_blocks :
  ∀ (blocks : GoalieBlocks),
    blocks.first = 4 →
    blocks.second = 2 * blocks.first →
    blocks.third = blocks.second - 3 →
    totalBlocks blocks = 21 →
    blocks.fourth = 4 := by
  sorry


end jordan_fourth_period_blocks_l2705_270583


namespace golden_section_steel_l2705_270597

theorem golden_section_steel (m : ℝ) : 
  (1000 + (m - 1000) * 0.618 = 1618 ∨ m - (m - 1000) * 0.618 = 1618) → 
  (m = 2000 ∨ m = 2618) := by
sorry

end golden_section_steel_l2705_270597


namespace three_number_sum_l2705_270550

theorem three_number_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →
  b = 7 →
  (a + b + c) / 3 = a + 15 →
  (a + b + c) / 3 = c - 20 →
  a + b + c = 36 := by
sorry

end three_number_sum_l2705_270550


namespace sum_of_reciprocals_l2705_270574

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) :
  1 / x + 1 / y = 5 := by sorry

end sum_of_reciprocals_l2705_270574


namespace arithmetic_sequence_problem_l2705_270565

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The problem statement -/
theorem arithmetic_sequence_problem 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 6 = seq.S 3) 
  (h2 : seq.a 6 = 12) : 
  seq.a 8 = 16 := by
  sorry

end arithmetic_sequence_problem_l2705_270565


namespace linear_system_existence_l2705_270575

theorem linear_system_existence :
  ∃ m : ℝ, ∀ x y : ℝ, (m - 1) * x - y = 1 ∧ m ≠ 1 := by
  sorry

end linear_system_existence_l2705_270575


namespace sunzi_car_problem_l2705_270572

theorem sunzi_car_problem (x : ℕ) : 
  (x / 4 : ℚ) + 1 = (x - 9 : ℚ) / 3 ↔ 
  (∃ (cars : ℕ), 
    (x / 4 + 1 = cars) ∧ 
    ((x - 9) / 3 = cars - 1)) :=
by sorry

end sunzi_car_problem_l2705_270572


namespace minimum_packaging_cost_l2705_270587

/-- Calculates the minimum cost for packaging a collection given box dimensions, cost per box, and total volume to be packaged -/
theorem minimum_packaging_cost 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (cost_per_box : ℝ) 
  (total_volume : ℝ) 
  (h_box_length : box_length = 20)
  (h_box_width : box_width = 20)
  (h_box_height : box_height = 15)
  (h_cost_per_box : cost_per_box = 0.70)
  (h_total_volume : total_volume = 3060000) :
  ⌈total_volume / (box_length * box_width * box_height)⌉ * cost_per_box = 357 :=
by sorry

end minimum_packaging_cost_l2705_270587


namespace number_of_girls_in_school_l2705_270517

/-- The number of girls in a school, given certain sampling conditions -/
theorem number_of_girls_in_school :
  ∀ (total_students sample_size : ℕ) (girls_in_school : ℕ),
  total_students = 1600 →
  sample_size = 200 →
  girls_in_school ≤ total_students →
  (girls_in_school : ℚ) / (total_students - girls_in_school : ℚ) = 95 / 105 →
  girls_in_school = 760 := by
sorry

end number_of_girls_in_school_l2705_270517


namespace shuttlecock_mass_probability_l2705_270591

variable (ξ : ℝ)

-- Define the probabilities given in the problem
def P_less_than_4_8 : ℝ := 0.3
def P_not_less_than_4_85 : ℝ := 0.32

-- Define the probability we want to prove
def P_between_4_8_and_4_85 : ℝ := 1 - P_less_than_4_8 - P_not_less_than_4_85

-- Theorem statement
theorem shuttlecock_mass_probability :
  P_between_4_8_and_4_85 = 0.38 := by sorry

end shuttlecock_mass_probability_l2705_270591


namespace pythagorean_triple_for_eleven_l2705_270521

theorem pythagorean_triple_for_eleven : ∃ (b c : ℕ), 11^2 + b^2 = c^2 ∧ c = 61 := by
  sorry

end pythagorean_triple_for_eleven_l2705_270521


namespace representable_integers_l2705_270549

theorem representable_integers (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2004) : 
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ k = (m * n + 1) / (m + n) := by
  sorry

end representable_integers_l2705_270549


namespace parabola_equation_l2705_270567

/-- Given a parabola y^2 = mx (m > 0) whose directrix is at a distance of 3 from the line x = 1,
    prove that the equation of the parabola is y^2 = 8x. -/
theorem parabola_equation (m : ℝ) (h_m_pos : m > 0) : 
  (∃ (k : ℝ), k = -m/4 ∧ |k - 1| = 3) → 
  (∀ (x y : ℝ), y^2 = m*x ↔ y^2 = 8*x) :=
sorry

end parabola_equation_l2705_270567


namespace z_less_than_y_percentage_l2705_270514

/-- Given w, x, y, z are real numbers satisfying certain conditions,
    prove that z is 46% less than y. -/
theorem z_less_than_y_percentage (w x y z : ℝ) 
  (hw : w = 0.6 * x)
  (hx : x = 0.6 * y)
  (hz : z = 1.5 * w) :
  z = 0.54 * y := by
  sorry

end z_less_than_y_percentage_l2705_270514


namespace carrie_tshirt_purchase_l2705_270526

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℝ := 9.15

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 22

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℝ := tshirt_cost * num_tshirts

theorem carrie_tshirt_purchase : total_cost = 201.30 := by
  sorry

end carrie_tshirt_purchase_l2705_270526


namespace arithmetic_sequence_problem_l2705_270580

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying 4a_3 + a_11 - 3a_5 = 10, prove that 1/5 * a_4 = 1 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h : 4 * seq.a 3 + seq.a 11 - 3 * seq.a 5 = 10) : 
  1/5 * seq.a 4 = 1 := by
  sorry

end arithmetic_sequence_problem_l2705_270580


namespace opposite_of_negative_two_l2705_270532

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  -- The proof goes here
  sorry

end opposite_of_negative_two_l2705_270532


namespace touchdown_points_l2705_270589

theorem touchdown_points (total_points : ℕ) (num_touchdowns : ℕ) (points_per_touchdown : ℕ) :
  total_points = 21 →
  num_touchdowns = 3 →
  total_points = num_touchdowns * points_per_touchdown →
  points_per_touchdown = 7 := by
sorry

end touchdown_points_l2705_270589


namespace cloud9_diving_company_revenue_l2705_270547

/-- Cloud 9 Diving Company's financial calculation -/
theorem cloud9_diving_company_revenue 
  (individual_bookings : ℕ) 
  (group_bookings : ℕ) 
  (cancellations : ℕ) 
  (h1 : individual_bookings = 12000)
  (h2 : group_bookings = 16000)
  (h3 : cancellations = 1600) :
  individual_bookings + group_bookings - cancellations = 26400 :=
by sorry

end cloud9_diving_company_revenue_l2705_270547


namespace cube_sum_greater_than_mixed_products_l2705_270599

theorem cube_sum_greater_than_mixed_products {a b : ℝ} (ha : a > 0) (hb : b > 0) (hnq : a ≠ b) :
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end cube_sum_greater_than_mixed_products_l2705_270599


namespace max_wrong_questions_l2705_270588

theorem max_wrong_questions (total_questions : ℕ) (success_threshold : ℚ) : 
  total_questions = 50 → success_threshold = 85 / 100 → 
  ∃ max_wrong : ℕ, max_wrong = 7 ∧ 
  (↑(total_questions - max_wrong) / ↑total_questions ≥ success_threshold) ∧
  ∀ wrong : ℕ, wrong > max_wrong → 
  (↑(total_questions - wrong) / ↑total_questions < success_threshold) :=
by
  sorry

end max_wrong_questions_l2705_270588


namespace quadratic_inequality_l2705_270510

theorem quadratic_inequality (a b c A B C : ℝ) 
  (ha : a ≠ 0) (hA : A ≠ 0)
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end quadratic_inequality_l2705_270510


namespace box_length_l2705_270518

/-- The length of a rectangular box given its filling rate, width, depth, and filling time. -/
theorem box_length (fill_rate : ℝ) (width depth : ℝ) (fill_time : ℝ) :
  fill_rate = 3 →
  width = 4 →
  depth = 3 →
  fill_time = 20 →
  fill_rate * fill_time / (width * depth) = 5 := by
sorry

end box_length_l2705_270518


namespace substitution_remainder_l2705_270522

/-- Represents the number of ways to make substitutions in a basketball game -/
def substitution_ways (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the remainder when the number of substitution ways is divided by 1000 -/
theorem substitution_remainder :
  substitution_ways 15 5 4 % 1000 = 301 := by
  sorry

end substitution_remainder_l2705_270522


namespace toms_deck_cost_l2705_270594

/-- Calculate the cost of a deck of cards -/
def deck_cost (rare_count : ℕ) (uncommon_count : ℕ) (common_count : ℕ) 
              (rare_price : ℚ) (uncommon_price : ℚ) (common_price : ℚ) : ℚ :=
  rare_count * rare_price + uncommon_count * uncommon_price + common_count * common_price

/-- The total cost of Tom's deck is $32 -/
theorem toms_deck_cost : 
  deck_cost 19 11 30 1 (1/2) (1/4) = 32 := by
  sorry

end toms_deck_cost_l2705_270594


namespace arithmetic_sequence_sum_l2705_270507

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ + a₂ = -1 and a₃ = 4,
    prove that a₄ + a₅ = 17. -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 1 + a 2 = -1)
    (h_third : a 3 = 4) : 
  a 4 + a 5 = 17 := by
  sorry

end arithmetic_sequence_sum_l2705_270507


namespace tournament_configuration_impossible_l2705_270579

structure Tournament where
  num_teams : Nat
  games_played : Fin num_teams → Nat

def is_valid_configuration (t : Tournament) : Prop :=
  t.num_teams = 12 ∧
  (∃ i : Fin t.num_teams, t.games_played i = 11) ∧
  (∃ i j k : Fin t.num_teams, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    t.games_played i = 9 ∧ t.games_played j = 9 ∧ t.games_played k = 9) ∧
  (∃ i j : Fin t.num_teams, i ≠ j ∧ 
    t.games_played i = 6 ∧ t.games_played j = 6) ∧
  (∃ i j k l : Fin t.num_teams, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
    t.games_played i = 4 ∧ t.games_played j = 4 ∧ t.games_played k = 4 ∧ t.games_played l = 4) ∧
  (∃ i j : Fin t.num_teams, i ≠ j ∧ 
    t.games_played i = 1 ∧ t.games_played j = 1)

theorem tournament_configuration_impossible :
  ¬∃ t : Tournament, is_valid_configuration t := by
  sorry

end tournament_configuration_impossible_l2705_270579


namespace cosine_sum_and_square_l2705_270541

theorem cosine_sum_and_square (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 6 + α) + (Real.cos (4 * π / 3 + α))^2 = (2 - Real.sqrt 3) / 3 := by
  sorry

end cosine_sum_and_square_l2705_270541


namespace square_difference_l2705_270539

theorem square_difference (a b : ℝ) 
  (h1 : a^2 - a*b = 10) 
  (h2 : a*b - b^2 = -15) : 
  a^2 - b^2 = -5 := by
sorry

end square_difference_l2705_270539


namespace min_singing_in_shower_l2705_270561

/-- Represents the youth summer village population -/
structure Village where
  total : ℕ
  notWorking : ℕ
  withFamilies : ℕ
  workingNoFamilySinging : ℕ

/-- The minimum number of people who like to sing in the shower -/
def minSingingInShower (v : Village) : ℕ := v.workingNoFamilySinging

theorem min_singing_in_shower (v : Village) 
  (h1 : v.total = 100)
  (h2 : v.notWorking = 50)
  (h3 : v.withFamilies = 25)
  (h4 : v.workingNoFamilySinging = 50)
  (h5 : v.workingNoFamilySinging ≤ v.total - v.notWorking)
  (h6 : v.workingNoFamilySinging ≤ v.total - v.withFamilies) :
  minSingingInShower v = 50 := by
  sorry

#check min_singing_in_shower

end min_singing_in_shower_l2705_270561


namespace last_three_digits_of_7_to_83_l2705_270506

theorem last_three_digits_of_7_to_83 :
  7^83 ≡ 886 [ZMOD 1000] := by sorry

end last_three_digits_of_7_to_83_l2705_270506


namespace peter_bird_count_l2705_270508

/-- The fraction of birds that are ducks -/
def duck_fraction : ℚ := 1/3

/-- The cost of chicken feed per bird in dollars -/
def chicken_feed_cost : ℚ := 2

/-- The total cost to feed all chickens in dollars -/
def total_chicken_feed_cost : ℚ := 20

/-- The total number of birds Peter has -/
def total_birds : ℕ := 15

theorem peter_bird_count :
  (1 - duck_fraction) * total_birds = total_chicken_feed_cost / chicken_feed_cost :=
by sorry

end peter_bird_count_l2705_270508


namespace light_bulb_probability_l2705_270502

theorem light_bulb_probability (qualification_rate : ℝ) 
  (h1 : qualification_rate = 0.99) : 
  ℝ :=
by
  -- The probability of selecting a qualified light bulb
  -- is equal to the qualification rate
  sorry

#check light_bulb_probability

end light_bulb_probability_l2705_270502


namespace chloe_trivia_score_l2705_270596

/-- Chloe's trivia game score calculation -/
theorem chloe_trivia_score (first_round : ℕ) (last_round_loss : ℕ) (total_points : ℕ) 
  (h1 : first_round = 40)
  (h2 : last_round_loss = 4)
  (h3 : total_points = 86) :
  ∃ second_round : ℕ, second_round = 50 ∧ 
    first_round + second_round - last_round_loss = total_points :=
by sorry

end chloe_trivia_score_l2705_270596


namespace union_of_sets_l2705_270568

theorem union_of_sets : 
  let M : Set ℤ := {-1, 0, 1}
  let N : Set ℤ := {0, 1, 2}
  M ∪ N = {-1, 0, 1, 2} := by sorry

end union_of_sets_l2705_270568


namespace intersection_of_A_and_B_l2705_270593

def A : Set ℝ := {x | x > -2}
def B : Set ℝ := {x | 1 - x > 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-2) 1 := by sorry

end intersection_of_A_and_B_l2705_270593


namespace inequality_solution_l2705_270552

/-- The solution set of the inequality |ax-2|+|ax-a| ≥ 2 when a = 1 -/
def solution_set_a1 : Set ℝ := {x | x ≥ 2.5 ∨ x ≤ 0.5}

/-- The inequality |ax-2|+|ax-a| ≥ 2 -/
def inequality (a x : ℝ) : Prop := |a*x - 2| + |a*x - a| ≥ 2

theorem inequality_solution :
  (∀ x, inequality 1 x ↔ x ∈ solution_set_a1) ∧
  (∀ a, a > 0 → (∀ x, inequality a x) ↔ a ≥ 4) :=
sorry

end inequality_solution_l2705_270552


namespace triangle_area_inequality_l2705_270519

theorem triangle_area_inequality (a b c S_triangle : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hS : S_triangle > 0)
  (h_triangle : S_triangle = Real.sqrt ((a + b + c) * (a + b - c) * (b + c - a) * (c + a - b) / 16)) :
  1 - (8 * ((a - b)^2 + (b - c)^2 + (c - a)^2)) / (a + b + c)^2 
  ≤ (432 * S_triangle^2) / (a + b + c)^4 
  ∧ (432 * S_triangle^2) / (a + b + c)^4 
  ≤ 1 - (2 * ((a - b)^2 + (b - c)^2 + (c - a)^2)) / (a + b + c)^2 := by
  sorry

end triangle_area_inequality_l2705_270519


namespace f_max_value_l2705_270557

/-- The quadratic function f(x) = 10x - 2x^2 -/
def f (x : ℝ) : ℝ := 10 * x - 2 * x^2

/-- The maximum value of f(x) is 12.5 -/
theorem f_max_value : ∃ (M : ℝ), M = 12.5 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end f_max_value_l2705_270557


namespace f_continuous_at_2_delta_epsilon_relation_l2705_270553

def f (x : ℝ) : ℝ := -3 * x^2 - 5

theorem f_continuous_at_2 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by sorry

theorem delta_epsilon_relation :
  ∀ ε > 0, ∃ δ > 0, δ = ε / 3 ∧
    ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by sorry

end f_continuous_at_2_delta_epsilon_relation_l2705_270553


namespace intersection_point_is_unique_l2705_270525

/-- The line equation in R³ --/
def line (x y z : ℝ) : Prop :=
  (x + 2) / 1 = (y - 2) / 0 ∧ (x + 2) / 1 = (z + 3) / 0

/-- The plane equation in R³ --/
def plane (x y z : ℝ) : Prop :=
  2 * x - 3 * y - 5 * z - 7 = 0

/-- The intersection point of the line and the plane --/
def intersection_point : ℝ × ℝ × ℝ := (-1, 2, -3)

theorem intersection_point_is_unique :
  ∃! p : ℝ × ℝ × ℝ, 
    line p.1 p.2.1 p.2.2 ∧ 
    plane p.1 p.2.1 p.2.2 ∧ 
    p = intersection_point :=
by
  sorry

end intersection_point_is_unique_l2705_270525


namespace even_function_four_roots_sum_zero_l2705_270544

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f has exactly four real roots if there exist exactly four distinct real numbers that make f(x) = 0 -/
def HasFourRealRoots (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧
    ∀ x, f x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d

/-- The theorem stating that for an even function with exactly four real roots, the sum of its roots is zero -/
theorem even_function_four_roots_sum_zero (f : ℝ → ℝ) 
    (h_even : IsEven f) (h_four_roots : HasFourRealRoots f) :
    ∃ (a b c d : ℝ), HasFourRealRoots f ∧ a + b + c + d = 0 := by
  sorry

end even_function_four_roots_sum_zero_l2705_270544


namespace cloth_cost_price_l2705_270530

/-- Given a trader selling cloth, calculates the cost price per meter. -/
def cost_price_per_meter (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_meters) / total_meters

/-- Proves that the cost price of one metre of cloth is 128 rupees. -/
theorem cloth_cost_price :
  cost_price_per_meter 60 8400 12 = 128 := by
  sorry

#eval cost_price_per_meter 60 8400 12

end cloth_cost_price_l2705_270530


namespace score_mode_l2705_270586

/-- Represents a score in the stem-and-leaf plot -/
structure Score where
  stem : Nat
  leaf : Nat

/-- The list of all scores from the stem-and-leaf plot -/
def scores : List Score := [
  {stem := 5, leaf := 5}, {stem := 5, leaf := 5},
  {stem := 6, leaf := 4}, {stem := 6, leaf := 8},
  {stem := 7, leaf := 2}, {stem := 7, leaf := 6}, {stem := 7, leaf := 6}, {stem := 7, leaf := 9},
  {stem := 8, leaf := 1}, {stem := 8, leaf := 3}, {stem := 8, leaf := 3}, {stem := 8, leaf := 3}, {stem := 8, leaf := 9}, {stem := 8, leaf := 9},
  {stem := 9, leaf := 0}, {stem := 9, leaf := 5}, {stem := 9, leaf := 5}, {stem := 9, leaf := 5}, {stem := 9, leaf := 7}, {stem := 9, leaf := 8},
  {stem := 10, leaf := 2}, {stem := 10, leaf := 2}, {stem := 10, leaf := 2}, {stem := 10, leaf := 3}, {stem := 10, leaf := 3}, {stem := 10, leaf := 3}, {stem := 10, leaf := 4},
  {stem := 11, leaf := 0}, {stem := 11, leaf := 0}, {stem := 11, leaf := 1}
]

/-- Converts a Score to its numerical value -/
def scoreValue (s : Score) : Nat := s.stem * 10 + s.leaf

/-- Defines the mode of a list of scores -/
def mode (l : List Score) : Set Nat := sorry

/-- Theorem stating that the mode of the given scores is {83, 95, 102, 103} -/
theorem score_mode : mode scores = {83, 95, 102, 103} := by sorry

end score_mode_l2705_270586


namespace kiley_crayons_l2705_270540

theorem kiley_crayons (initial_crayons : ℕ) (remaining_crayons : ℕ) 
  (h1 : initial_crayons = 48)
  (h2 : remaining_crayons = 18)
  (f : ℚ) -- fraction of crayons Kiley took
  (h3 : 0 ≤ f ∧ f < 1)
  (h4 : remaining_crayons = (initial_crayons : ℚ) * (1 - f) / 2) :
  f = 1/4 := by
  sorry

end kiley_crayons_l2705_270540


namespace afternoon_campers_l2705_270509

theorem afternoon_campers (morning_campers : ℕ) (afternoon_difference : ℕ) : 
  morning_campers = 52 → 
  afternoon_difference = 9 → 
  morning_campers + afternoon_difference = 61 :=
by
  sorry

end afternoon_campers_l2705_270509


namespace bothMiss_mutually_exclusive_with_hitAtLeastOnce_bothMiss_complement_of_hitAtLeastOnce_l2705_270581

-- Define the sample space for two shots
inductive ShotOutcome
| Hit
| Miss

-- Define the type for a two-shot experiment
def TwoShots := (ShotOutcome × ShotOutcome)

-- Define the event "hitting the target at least once"
def hitAtLeastOnce (outcome : TwoShots) : Prop :=
  outcome.1 = ShotOutcome.Hit ∨ outcome.2 = ShotOutcome.Hit

-- Define the event "both shots miss"
def bothMiss (outcome : TwoShots) : Prop :=
  outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Miss

-- Theorem stating that "both shots miss" is mutually exclusive with "hitting at least once"
theorem bothMiss_mutually_exclusive_with_hitAtLeastOnce :
  ∀ (outcome : TwoShots), ¬(hitAtLeastOnce outcome ∧ bothMiss outcome) :=
sorry

-- Theorem stating that "both shots miss" is the complement of "hitting at least once"
theorem bothMiss_complement_of_hitAtLeastOnce :
  ∀ (outcome : TwoShots), hitAtLeastOnce outcome ↔ ¬(bothMiss outcome) :=
sorry

end bothMiss_mutually_exclusive_with_hitAtLeastOnce_bothMiss_complement_of_hitAtLeastOnce_l2705_270581


namespace min_sum_and_min_product_l2705_270577

/-- An arithmetic sequence with sum S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of first n terms
  is_arithmetic : ∀ n : ℕ, S (n + 2) - S (n + 1) = S (n + 1) - S n

/-- The specific arithmetic sequence satisfying given conditions -/
def special_sequence (a : ArithmeticSequence) : Prop :=
  a.S 10 = 0 ∧ a.S 15 = 25

theorem min_sum_and_min_product (a : ArithmeticSequence) 
  (h : special_sequence a) :
  (∀ n : ℕ, n > 0 → a.S n ≥ a.S 5) ∧ 
  (∀ n : ℕ, n > 0 → n * (a.S n) ≥ -49) := by
  sorry

end min_sum_and_min_product_l2705_270577


namespace union_of_A_and_B_l2705_270556

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | x*(x-2)*(x-5) < 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | 1 < x ∧ x < 5} := by sorry

end union_of_A_and_B_l2705_270556


namespace sheela_net_monthly_income_l2705_270554

/-- Calculates the total net monthly income for Sheela given her various income sources and tax rates. -/
theorem sheela_net_monthly_income 
  (savings_deposit : ℝ)
  (savings_deposit_percentage : ℝ)
  (freelance_income : ℝ)
  (annual_interest : ℝ)
  (freelance_tax_rate : ℝ)
  (interest_tax_rate : ℝ)
  (h1 : savings_deposit = 5000)
  (h2 : savings_deposit_percentage = 0.20)
  (h3 : freelance_income = 3000)
  (h4 : annual_interest = 2400)
  (h5 : freelance_tax_rate = 0.10)
  (h6 : interest_tax_rate = 0.05) :
  ∃ (total_net_monthly_income : ℝ), 
    total_net_monthly_income = 27890 :=
by sorry

end sheela_net_monthly_income_l2705_270554


namespace student_contribution_l2705_270545

/-- Proves that if 30 students contribute equally every Friday for 2 months (8 Fridays)
    and collect a total of $480, then each student's contribution per Friday is $2. -/
theorem student_contribution
  (num_students : ℕ)
  (num_fridays : ℕ)
  (total_amount : ℕ)
  (h1 : num_students = 30)
  (h2 : num_fridays = 8)
  (h3 : total_amount = 480) :
  total_amount / (num_students * num_fridays) = 2 :=
by sorry

end student_contribution_l2705_270545


namespace dvd_price_percentage_l2705_270511

theorem dvd_price_percentage (srp : ℝ) (h1 : srp > 0) : 
  let marked_price := 0.6 * srp
  let bob_price := 0.4 * marked_price
  bob_price / srp = 0.24 := by
sorry

end dvd_price_percentage_l2705_270511


namespace age_difference_l2705_270516

theorem age_difference (a b c : ℕ) : 
  b = 2 * c → 
  a + b + c = 22 → 
  b = 8 → 
  a - b = 2 :=
by sorry

end age_difference_l2705_270516


namespace sqrt_equation_solution_l2705_270515

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (10 + 3 * z) = 12 :=
by
  sorry

end sqrt_equation_solution_l2705_270515


namespace triangle_angle_inequality_l2705_270505

theorem triangle_angle_inequality (y : ℝ) (p q : ℝ) : 
  y > 0 →
  y + 10 > 0 →
  y + 5 > 0 →
  4*y > 0 →
  y + 10 + y + 5 > 4*y →
  y + 10 + 4*y > y + 5 →
  y + 5 + 4*y > y + 10 →
  4*y > y + 10 →
  4*y > y + 5 →
  p < y →
  y < q →
  q - p ≥ 5/3 :=
by sorry

end triangle_angle_inequality_l2705_270505


namespace units_digit_of_quotient_l2705_270536

theorem units_digit_of_quotient : ∃ n : ℕ, (7^1993 + 5^1993) / 6 = 10 * n + 2 := by
  sorry

end units_digit_of_quotient_l2705_270536


namespace inequality_proof_l2705_270555

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum : a + b + c = 1) :
  (1/a + 1/(b*c)) * (1/b + 1/(c*a)) * (1/c + 1/(a*b)) ≥ 1728 := by
  sorry

end inequality_proof_l2705_270555
