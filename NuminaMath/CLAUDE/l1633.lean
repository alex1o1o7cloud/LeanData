import Mathlib

namespace snowfall_difference_l1633_163362

/-- Snowfall difference calculation -/
theorem snowfall_difference (bald_mountain : ℝ) (billy_mountain : ℝ) (mount_pilot : ℝ) :
  bald_mountain = 1.5 →
  billy_mountain = 3.5 →
  mount_pilot = 1.26 →
  (billy_mountain + mount_pilot - bald_mountain) * 100 = 326 := by
  sorry

end snowfall_difference_l1633_163362


namespace inequality_proof_l1633_163334

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b > 0) :
  a / b^2 + b / a^2 > 1 / a + 1 / b := by
sorry

end inequality_proof_l1633_163334


namespace angle_inequality_equivalence_l1633_163332

theorem angle_inequality_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ < 0) ↔
  (Real.pi / 2 < θ ∧ θ < 3 * Real.pi / 2) := by sorry

end angle_inequality_equivalence_l1633_163332


namespace max_points_tournament_l1633_163339

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (games_per_pair : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculate the total number of games in the tournament -/
def total_games (t : Tournament) : ℕ :=
  t.num_teams.choose 2 * t.games_per_pair

/-- Calculate the maximum points achievable by the top three teams -/
def max_points_top_three (t : Tournament) : ℕ :=
  let games_against_lower := (t.num_teams - 3) * t.games_per_pair
  let points_from_lower := games_against_lower * t.points_for_win
  let games_among_top := 2 * t.games_per_pair
  let points_among_top := games_among_top * t.points_for_win / 2
  points_from_lower + points_among_top

/-- The main theorem stating the maximum points for top three teams -/
theorem max_points_tournament :
  ∀ t : Tournament,
    t.num_teams = 8 →
    t.games_per_pair = 2 →
    t.points_for_win = 3 →
    t.points_for_draw = 1 →
    t.points_for_loss = 0 →
    max_points_top_three t = 36 := by
  sorry

end max_points_tournament_l1633_163339


namespace absolute_value_nonnegative_l1633_163360

theorem absolute_value_nonnegative (x : ℝ) : ¬(|x| < 0) := by
  sorry

end absolute_value_nonnegative_l1633_163360


namespace ice_cream_combinations_l1633_163345

/-- The number of ways to distribute n indistinguishable items among k distinguishable categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 5 indistinguishable items among 3 distinguishable categories is 21 -/
theorem ice_cream_combinations : distribute 5 3 = 21 := by sorry

end ice_cream_combinations_l1633_163345


namespace unique_three_digit_perfect_square_product_l1633_163304

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that converts a three-digit number to its cyclic permutations -/
def cyclic_permutations (n : ℕ) : Fin 3 → ℕ
| 0 => n
| 1 => (n % 100) * 10 + n / 100
| 2 => (n % 10) * 100 + n / 10

/-- The main theorem stating that 243 is the only three-digit number satisfying the given conditions -/
theorem unique_three_digit_perfect_square_product :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  (n / 100 ≤ n / 10 % 10 ∧ n / 100 ≤ n % 10) ∧
  is_perfect_square (cyclic_permutations n 0 * cyclic_permutations n 1 * cyclic_permutations n 2) ∧
  n = 243 :=
sorry

end unique_three_digit_perfect_square_product_l1633_163304


namespace paper_stack_height_l1633_163341

/-- Given a stack of paper where 800 sheets are 4 cm thick, 
    prove that a 6 cm high stack would contain 1200 sheets. -/
theorem paper_stack_height (sheets : ℕ) (height : ℝ) : 
  (800 : ℝ) / 4 = sheets / height → sheets = 1200 ∧ height = 6 :=
by sorry

end paper_stack_height_l1633_163341


namespace expression_simplification_l1633_163318

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 - 2) / x) * ((y^2 - 2) / y) - ((x^2 + 2) / y) * ((y^2 + 2) / x) = -4 * (x / y + y / x) := by
  sorry

end expression_simplification_l1633_163318


namespace red_tiles_181_implies_total_2116_l1633_163314

/-- Represents a square floor tiled with congruent square tiles -/
structure SquareFloor :=
  (side : ℕ)

/-- Calculates the number of red tiles on a square floor -/
def red_tiles (floor : SquareFloor) : ℕ :=
  4 * floor.side - 2

/-- Calculates the total number of tiles on a square floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side * floor.side

/-- Theorem stating that a square floor with 181 red tiles has 2116 total tiles -/
theorem red_tiles_181_implies_total_2116 :
  ∀ (floor : SquareFloor), red_tiles floor = 181 → total_tiles floor = 2116 :=
by
  sorry

end red_tiles_181_implies_total_2116_l1633_163314


namespace correct_num_ways_to_choose_l1633_163322

/-- The number of humanities courses -/
def num_humanities : ℕ := 4

/-- The number of natural science courses -/
def num_sciences : ℕ := 3

/-- The total number of courses to be chosen -/
def courses_to_choose : ℕ := 3

/-- The number of conflicting course pairs (A₁ and B₁) -/
def num_conflicts : ℕ := 1

/-- The function that calculates the number of ways to choose courses -/
def num_ways_to_choose : ℕ := sorry

theorem correct_num_ways_to_choose :
  num_ways_to_choose = 25 := by sorry

end correct_num_ways_to_choose_l1633_163322


namespace frequency_distribution_theorem_l1633_163398

-- Define the frequency of the first group
def f1 : ℕ := 6

-- Define the frequencies of the second and third groups based on the ratio
def f2 : ℕ := 2 * f1
def f3 : ℕ := 3 * f1

-- Define the sum of frequencies for the first three groups
def sum_first_three : ℕ := f1 + f2 + f3

-- Define the total number of students
def total_students : ℕ := 48

-- Theorem statement
theorem frequency_distribution_theorem :
  sum_first_three < total_students ∧ 
  total_students - sum_first_three > 0 ∧
  total_students - sum_first_three < f3 :=
by sorry

end frequency_distribution_theorem_l1633_163398


namespace tangent_point_divides_equally_l1633_163375

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicQuadrilateralWithInscribedCircle where
  -- Sides of the quadrilateral
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Ensure all sides are positive
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  -- The quadrilateral is cyclic (inscribed in a circle)
  is_cyclic : True
  -- The quadrilateral has an inscribed circle
  has_inscribed_circle : True

/-- Theorem: In a cyclic quadrilateral with an inscribed circle, 
    if the consecutive sides have lengths 80, 120, 100, and 140, 
    then the point of tangency of the inscribed circle on the side 
    of length 100 divides it into two equal segments. -/
theorem tangent_point_divides_equally 
  (Q : CyclicQuadrilateralWithInscribedCircle) 
  (h1 : Q.a = 80) 
  (h2 : Q.b = 120) 
  (h3 : Q.c = 100) 
  (h4 : Q.d = 140) : 
  ∃ (x y : ℝ), x + y = 100 ∧ x = y :=
sorry

end tangent_point_divides_equally_l1633_163375


namespace roots_of_polynomial_l1633_163348

/-- The polynomial function we're considering -/
def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 5*x - 10

/-- The roots of the polynomial -/
def roots : Set ℝ := {-2, Real.sqrt 5, -Real.sqrt 5}

/-- Theorem stating that the given set contains all roots of the polynomial -/
theorem roots_of_polynomial :
  ∀ x : ℝ, f x = 0 ↔ x ∈ roots := by sorry

end roots_of_polynomial_l1633_163348


namespace ellipse_equation_l1633_163347

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  c : ℝ  -- Half of focal distance
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_c_less_a : c < a
  h_pythagoras : a^2 = b^2 + c^2

/-- The standard form equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation (e : Ellipse)
  (h_sum : e.a + e.b = 5)  -- Half of the sum of axes lengths
  (h_focal : e.c = 2 * Real.sqrt 5) :
  standard_equation e = fun x y ↦ x^2 / 36 + y^2 / 16 = 1 := by
  sorry

#check ellipse_equation

end ellipse_equation_l1633_163347


namespace profit_percentage_calculation_l1633_163380

theorem profit_percentage_calculation (selling_price : ℝ) (cost_price : ℝ) 
  (h : cost_price = 0.95 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (1 / 0.95 - 1) * 100 := by
  sorry

end profit_percentage_calculation_l1633_163380


namespace money_distribution_l1633_163365

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (ac_sum : A + C = 200)
  (c_amount : C = 30) :
  B + C = 330 := by
sorry

end money_distribution_l1633_163365


namespace total_rabbits_caught_l1633_163350

/-- Represents the number of rabbits caught on a given day -/
def rabbits_caught (day : ℕ) : ℕ :=
  203 - 3 * day

/-- Represents the number of squirrels caught on a given day -/
def squirrels_caught (day : ℕ) : ℕ :=
  16 + 2 * day

/-- The day when more squirrels are caught than rabbits -/
def crossover_day : ℕ :=
  38

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a : ℤ) (d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem total_rabbits_caught : 
  arithmetic_sum crossover_day 200 (-3) = 5491 := by
  sorry

end total_rabbits_caught_l1633_163350


namespace diagonals_sum_bounds_l1633_163373

/-- A convex pentagon in a 2D plane -/
structure ConvexPentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  convex : Bool

/-- Calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the perimeter of the pentagon -/
def perimeter (p : ConvexPentagon) : ℝ :=
  distance p.A p.B + distance p.B p.C + distance p.C p.D + distance p.D p.E + distance p.E p.A

/-- Calculate the sum of diagonals of the pentagon -/
def sumDiagonals (p : ConvexPentagon) : ℝ :=
  distance p.A p.C + distance p.B p.D + distance p.C p.E + distance p.D p.A + distance p.B p.E

/-- Theorem: The sum of diagonals is greater than the perimeter but less than twice the perimeter -/
theorem diagonals_sum_bounds (p : ConvexPentagon) (h : p.convex = true) :
  perimeter p < sumDiagonals p ∧ sumDiagonals p < 2 * perimeter p := by sorry

end diagonals_sum_bounds_l1633_163373


namespace max_value_on_curve_l1633_163396

-- Define the curve
def on_curve (x y b : ℝ) : Prop := x^2/4 + y^2/b^2 = 1

-- Define the function to maximize
def f (x y : ℝ) : ℝ := x^2 + 2*y

-- State the theorem
theorem max_value_on_curve (b : ℝ) (h : b > 0) :
  (∃ (x y : ℝ), on_curve x y b ∧ 
    ∀ (x' y' : ℝ), on_curve x' y' b → f x y ≥ f x' y') →
  ((0 < b ∧ b ≤ 4 → ∃ (x y : ℝ), on_curve x y b ∧ f x y = b^2/4 + 4) ∧
   (b > 4 → ∃ (x y : ℝ), on_curve x y b ∧ f x y = 2*b)) :=
by sorry

end max_value_on_curve_l1633_163396


namespace fraction_product_simplification_l1633_163306

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end fraction_product_simplification_l1633_163306


namespace proposition_p_or_q_is_true_l1633_163359

open Real

theorem proposition_p_or_q_is_true :
  (∀ x > 0, exp x > 1 + x) ∨
  (∀ f : ℝ → ℝ, (∀ x, f x + 2 = -(f (-x) + 2)) → 
   ∀ x, f (x - 0) + 0 = f (-(x - 0)) + 4) := by sorry

end proposition_p_or_q_is_true_l1633_163359


namespace sum_of_parts_of_complex_number_l1633_163330

theorem sum_of_parts_of_complex_number : ∃ (z : ℂ), 
  z = (Complex.I * 2 - 3) * (Complex.I - 2) / Complex.I ∧ 
  z.re + z.im = -11 := by
  sorry

end sum_of_parts_of_complex_number_l1633_163330


namespace beidou_timing_accuracy_l1633_163302

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem beidou_timing_accuracy : 
  toScientificNotation 0.0000000099 = ScientificNotation.mk 9.9 (-9) sorry := by
  sorry

end beidou_timing_accuracy_l1633_163302


namespace correct_notation_of_expression_l1633_163392

/-- Predicate to check if an expression is correctly written in standard algebraic notation -/
def is_correct_notation : Set ℝ → Prop :=
  sorry

/-- The specific expression we're checking -/
def expression : Set ℝ := {x | ∃ y, y = |4| / 3 ∧ x = y}

/-- Theorem stating that the given expression is correctly notated -/
theorem correct_notation_of_expression : is_correct_notation expression :=
  sorry

end correct_notation_of_expression_l1633_163392


namespace first_nonzero_digit_after_decimal_1_97_l1633_163382

theorem first_nonzero_digit_after_decimal_1_97 : ∃ (n : ℕ) (d : ℕ), 
  0 < d ∧ d < 10 ∧ 
  (∃ (k : ℕ), 10^n ≤ k * 97 ∧ k * 97 < 10^(n+1) ∧ 
  (10^(n+1) * 1 - k * 97) / 97 = d) ∧
  d = 3 := by
sorry

end first_nonzero_digit_after_decimal_1_97_l1633_163382


namespace saree_price_calculation_l1633_163320

theorem saree_price_calculation (P : ℝ) : 
  (P * (1 - 0.15) * (1 - 0.05) = 323) → P = 400 := by
  sorry

end saree_price_calculation_l1633_163320


namespace three_points_distance_is_four_l1633_163343

/-- The quadratic function f(x) = x^2 - 2x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- A point (x, y) is on the graph of f if y = f(x) -/
def on_graph (x y : ℝ) : Prop := y = f x

/-- The distance of a point (x, y) from the x-axis is the absolute value of y -/
def distance_from_x_axis (y : ℝ) : ℝ := |y|

/-- There exist exactly three points on the graph of f with distance m from the x-axis -/
def three_points_with_distance (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    on_graph x₁ y₁ ∧ on_graph x₂ y₂ ∧ on_graph x₃ y₃ ∧
    distance_from_x_axis y₁ = m ∧
    distance_from_x_axis y₂ = m ∧
    distance_from_x_axis y₃ = m ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    ∀ x y : ℝ, on_graph x y → distance_from_x_axis y = m → (x = x₁ ∨ x = x₂ ∨ x = x₃)

theorem three_points_distance_is_four :
  ∀ m : ℝ, three_points_with_distance m → m = 4 :=
by sorry

end three_points_distance_is_four_l1633_163343


namespace perpendicular_line_through_point_l1633_163329

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The problem statement -/
theorem perpendicular_line_through_point :
  ∃ (l : Line),
    perpendicular l (Line.mk 1 (-2) (-1)) ∧
    point_on_line 1 1 l ∧
    l = Line.mk 2 1 (-3) := by
  sorry

end perpendicular_line_through_point_l1633_163329


namespace function_six_monotonic_intervals_l1633_163307

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * |x^3| - (a/2) * x^2 + (3-a) * |x| + b

theorem function_six_monotonic_intervals (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (-x)) →
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (∀ x : ℝ, x > 0 → (x^2 - a*x + (3-a) = 0 ↔ (x = x₁ ∨ x = x₂)))) →
  2 < a ∧ a < 3 :=
sorry

end function_six_monotonic_intervals_l1633_163307


namespace equation_solution_range_l1633_163352

-- Define the equation
def equation (x k : ℝ) : Prop :=
  (x^2 + k*x + 3) / (x - 1) = 3*x + k

-- Define the condition for exactly one positive real solution
def has_one_positive_solution (k : ℝ) : Prop :=
  ∃! x : ℝ, x > 0 ∧ equation x k

-- Theorem statement
theorem equation_solution_range :
  ∀ k : ℝ, has_one_positive_solution k ↔ (k = -33/8 ∨ k = -4 ∨ k ≥ -3) :=
by sorry

end equation_solution_range_l1633_163352


namespace twenty_percent_of_twenty_l1633_163353

theorem twenty_percent_of_twenty : (20 : ℝ) / 100 * 20 = 4 := by
  sorry

end twenty_percent_of_twenty_l1633_163353


namespace reader_group_total_l1633_163344

/-- Represents the number of readers in a group reading different types of books. -/
structure ReaderGroup where
  sci_fi : ℕ     -- Number of readers who read science fiction
  literary : ℕ   -- Number of readers who read literary works
  both : ℕ       -- Number of readers who read both

/-- Calculates the total number of readers in the group. -/
def total_readers (g : ReaderGroup) : ℕ :=
  g.sci_fi + g.literary - g.both

/-- Theorem stating that for the given reader numbers, the total is 650. -/
theorem reader_group_total :
  ∃ (g : ReaderGroup), g.sci_fi = 250 ∧ g.literary = 550 ∧ g.both = 150 ∧ total_readers g = 650 :=
by
  sorry

#check reader_group_total

end reader_group_total_l1633_163344


namespace expression_evaluation_l1633_163363

theorem expression_evaluation : 11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end expression_evaluation_l1633_163363


namespace largest_three_digit_sum_15_l1633_163370

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_sum_15 :
  ∀ n : ℕ, is_three_digit n → digit_sum n = 15 → n ≤ 960 :=
by sorry

end largest_three_digit_sum_15_l1633_163370


namespace double_reflection_result_l1633_163378

def reflect_over_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_over_y_axis (reflect_over_y_equals_x p)

theorem double_reflection_result :
  double_reflection (7, -3) = (3, 7) := by sorry

end double_reflection_result_l1633_163378


namespace fifi_closet_hangers_l1633_163374

theorem fifi_closet_hangers :
  let pink : ℕ := 7
  let green : ℕ := 4
  let blue : ℕ := green - 1
  let yellow : ℕ := blue - 1
  pink + green + blue + yellow = 16 :=
by sorry

end fifi_closet_hangers_l1633_163374


namespace certain_number_divided_by_ten_l1633_163323

theorem certain_number_divided_by_ten (x : ℝ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end certain_number_divided_by_ten_l1633_163323


namespace max_profit_at_70_l1633_163399

-- Define the linear function for weekly sales quantity
def sales_quantity (x : ℝ) : ℝ := -2 * x + 200

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 40) * (sales_quantity x)

-- Theorem stating the maximum profit and the price at which it occurs
theorem max_profit_at_70 :
  ∃ (max_profit : ℝ), max_profit = 1800 ∧
  ∀ (x : ℝ), profit x ≤ max_profit ∧
  profit 70 = max_profit :=
sorry

#check max_profit_at_70

end max_profit_at_70_l1633_163399


namespace gasoline_price_increase_l1633_163335

theorem gasoline_price_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (price_increase : ℝ) 
  (budget_increase : ℝ) 
  (quantity_decrease : ℝ) 
  (h1 : budget_increase = 0.15)
  (h2 : quantity_decrease = 0.08000000000000007)
  (h3 : original_price > 0)
  (h4 : original_quantity > 0) :
  original_price * original_quantity * (1 + budget_increase) = 
  (original_price * (1 + price_increase)) * (original_quantity * (1 - quantity_decrease)) →
  price_increase = 0.25 := by
sorry

end gasoline_price_increase_l1633_163335


namespace reflected_ray_is_correct_l1633_163368

/-- The line of reflection --/
def reflection_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = -1}

/-- Point P --/
def P : ℝ × ℝ := (1, 1)

/-- Point Q --/
def Q : ℝ × ℝ := (2, 3)

/-- The reflected ray --/
def reflected_ray : Set (ℝ × ℝ) := {p : ℝ × ℝ | 5 * p.1 - 4 * p.2 + 2 = 0}

/-- Theorem stating that the reflected ray is correct --/
theorem reflected_ray_is_correct : 
  ∃ (M : ℝ × ℝ), 
    (M ∈ reflected_ray) ∧ 
    (Q ∈ reflected_ray) ∧
    (∀ (X : ℝ × ℝ), X ∈ reflection_line → (X.1 - P.1) * (X.1 - M.1) + (X.2 - P.2) * (X.2 - M.2) = 0) :=
sorry

end reflected_ray_is_correct_l1633_163368


namespace rhombus_perimeter_l1633_163387

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 40 := by
  sorry

#check rhombus_perimeter

end rhombus_perimeter_l1633_163387


namespace circle_center_condition_l1633_163336

-- Define the equation of the circle
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*y + 3 - a = 0

-- Define the condition for the center to be in the second quadrant
def center_in_second_quadrant (a : ℝ) : Prop :=
  a < 0 ∧ 1 > 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a < -2

-- Theorem statement
theorem circle_center_condition (a : ℝ) :
  (∃ x y : ℝ, circle_equation x y a) ∧
  center_in_second_quadrant a
  ↔ a_range a :=
sorry

end circle_center_condition_l1633_163336


namespace weight_difference_l1633_163369

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9

theorem weight_difference : mildred_weight - carol_weight = 50 := by
  sorry

end weight_difference_l1633_163369


namespace hyperbola_asymptotes_l1633_163371

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal length and real axis relationship
def focal_length_relation (a : ℝ) : Prop := 2 * (2 * a) = 4 * a

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y, hyperbola x y a b) →
  focal_length_relation a →
  (∀ x y, asymptote_equation x y) :=
sorry

end hyperbola_asymptotes_l1633_163371


namespace function_shift_and_value_l1633_163372

theorem function_shift_and_value (φ : Real) 
  (h1 : 0 < φ ∧ φ < π / 2) 
  (f : ℝ → ℝ) 
  (h2 : ∀ x, f x = 2 * Real.sin (x + φ)) 
  (g : ℝ → ℝ) 
  (h3 : ∀ x, g x = f (x + π / 3)) 
  (h4 : ∀ x, g x = g (-x)) : 
  f (π / 6) = Real.sqrt 3 := by
sorry

end function_shift_and_value_l1633_163372


namespace triangle_third_side_l1633_163391

theorem triangle_third_side (a b area c : ℝ) : 
  a = 2 * Real.sqrt 2 →
  b = 3 →
  area = 3 →
  area = (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) →
  (c = Real.sqrt 5 ∨ c = Real.sqrt 29) := by
sorry

end triangle_third_side_l1633_163391


namespace gcd_lcm_sum_l1633_163321

theorem gcd_lcm_sum : Nat.gcd 24 54 + Nat.lcm 48 18 = 150 := by sorry

end gcd_lcm_sum_l1633_163321


namespace total_individual_packs_l1633_163349

def cookies_packs : ℕ := 3
def cookies_per_pack : ℕ := 4
def noodles_packs : ℕ := 4
def noodles_per_pack : ℕ := 8
def juice_packs : ℕ := 5
def juice_per_pack : ℕ := 6
def snacks_packs : ℕ := 2
def snacks_per_pack : ℕ := 10

theorem total_individual_packs :
  cookies_packs * cookies_per_pack +
  noodles_packs * noodles_per_pack +
  juice_packs * juice_per_pack +
  snacks_packs * snacks_per_pack = 94 := by
  sorry

end total_individual_packs_l1633_163349


namespace student_marks_l1633_163325

/-- Calculate the total marks secured in an exam given the following conditions:
  * total_questions: The total number of questions in the exam
  * correct_answers: The number of questions answered correctly
  * marks_per_correct: The number of marks awarded for each correct answer
  * marks_lost_per_wrong: The number of marks lost for each wrong answer
-/
def calculate_total_marks (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : ℤ :=
  (correct_answers * marks_per_correct : ℤ) - 
  ((total_questions - correct_answers) * marks_lost_per_wrong)

/-- Theorem stating that under the given conditions, the student secures 160 marks -/
theorem student_marks : 
  calculate_total_marks 60 44 4 1 = 160 := by
  sorry

end student_marks_l1633_163325


namespace quadratic_equal_roots_l1633_163394

theorem quadratic_equal_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - 4*x + m - 1 = 0 ∧
    ∀ y : ℝ, y^2 - 4*y + m - 1 = 0 → y = x) →
  m = 5 ∧ ∃ x : ℝ, x^2 - 4*x + m - 1 = 0 ∧ x = 2 :=
by sorry

end quadratic_equal_roots_l1633_163394


namespace square_sum_value_l1633_163310

theorem square_sum_value (a b : ℝ) : 
  (a^2 + b^2 + 2) * (a^2 + b^2 - 2) = 45 → a^2 + b^2 = 7 := by
  sorry

end square_sum_value_l1633_163310


namespace arithmetic_sequence_ratio_l1633_163357

/-- Given two arithmetic sequences, prove the ratio of their 4th terms -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) : 
  (∀ n, S n / T n = (7 * n + 2) / (n + 3)) →  -- Given condition
  (∀ n, S n = (a 1 + a n) * n / 2) →  -- Definition of S_n for arithmetic sequence
  (∀ n, T n = (b 1 + b n) * n / 2) →  -- Definition of T_n for arithmetic sequence
  a 4 / b 4 = 51 / 10 := by
sorry

end arithmetic_sequence_ratio_l1633_163357


namespace circle_ratio_l1633_163346

theorem circle_ratio (R r c d : ℝ) (h1 : R > r) (h2 : r > 0) (h3 : c > 0) (h4 : d > 0) :
  π * R^2 = (c / d) * (π * R^2 - π * r^2 + 2 * (2 * r^2)) →
  R / r = Real.sqrt (c * (4 - π) / (d * π - c * π)) :=
by sorry

end circle_ratio_l1633_163346


namespace sqrt_225_equals_15_l1633_163355

theorem sqrt_225_equals_15 : Real.sqrt 225 = 15 := by
  sorry

end sqrt_225_equals_15_l1633_163355


namespace tan_30_plus_3sin_30_l1633_163327

theorem tan_30_plus_3sin_30 :
  Real.tan (30 * Real.pi / 180) + 3 * Real.sin (30 * Real.pi / 180) = (2 * Real.sqrt 3 + 9) / 6 := by
  sorry

end tan_30_plus_3sin_30_l1633_163327


namespace younger_brother_height_l1633_163316

theorem younger_brother_height (h1 h2 : ℝ) (h1_positive : 0 < h1) (h2_positive : 0 < h2) 
  (height_difference : h2 - h1 = 12) (height_sum : h1 + h2 = 308) (h1_smaller : h1 < h2) : h1 = 148 :=
by sorry

end younger_brother_height_l1633_163316


namespace floor_sum_example_l1633_163308

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by sorry

end floor_sum_example_l1633_163308


namespace bread_distribution_l1633_163361

theorem bread_distribution (a d : ℚ) (h1 : d > 0) 
  (h2 : (a - 2*d) + (a - d) + a + (a + d) + (a + 2*d) = 100)
  (h3 : (1/7) * (a + (a + d) + (a + 2*d)) = (a - 2*d) + (a - d)) :
  a - 2*d = 5/3 := by
sorry

end bread_distribution_l1633_163361


namespace inverse_i_minus_inverse_l1633_163315

/-- Given a complex number i where i^2 = -1, prove that (i - i⁻¹)⁻¹ = -i/2 -/
theorem inverse_i_minus_inverse (i : ℂ) (h : i^2 = -1) : (i - i⁻¹)⁻¹ = -i/2 := by
  sorry

end inverse_i_minus_inverse_l1633_163315


namespace chord_length_l1633_163305

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end chord_length_l1633_163305


namespace cyclic_quadrilateral_angle_problem_l1633_163364

-- Define the cyclic quadrilateral ABCD
def CyclicQuadrilateral (A B C D : Point) : Prop := sorry

-- Define the angle measure
def AngleMeasure (P Q R : Point) : ℝ := sorry

-- Define a point inside a triangle
def PointInsideTriangle (X A B C : Point) : Prop := sorry

-- Define angle bisector
def AngleBisector (A X B C : Point) : Prop := sorry

theorem cyclic_quadrilateral_angle_problem 
  (A B C D X : Point) 
  (h1 : CyclicQuadrilateral A B C D) 
  (h2 : AngleMeasure A D B = 48)
  (h3 : AngleMeasure B D C = 56)
  (h4 : PointInsideTriangle X A B C)
  (h5 : AngleMeasure B C X = 24)
  (h6 : AngleBisector A X B C) :
  AngleMeasure C B X = 38 := by
sorry

end cyclic_quadrilateral_angle_problem_l1633_163364


namespace horner_rule_operations_l1633_163384

/-- Horner's Rule evaluation steps for a polynomial -/
def hornerSteps (coeffs : List ℤ) (x : ℤ) : List ℤ :=
  match coeffs with
  | [] => []
  | a :: as => List.scanl (fun acc b => acc * x + b) a as

/-- Number of multiplications in Horner's Rule -/
def numMultiplications (coeffs : List ℤ) : ℕ :=
  match coeffs with
  | [] => 0
  | [_] => 0
  | _ :: _ => coeffs.length - 1

/-- Number of additions in Horner's Rule -/
def numAdditions (coeffs : List ℤ) : ℕ :=
  match coeffs with
  | [] => 0
  | [_] => 0
  | _ :: _ => coeffs.length - 1

/-- The polynomial f(x) = 5x^6 + 4x^5 + x^4 + 3x^3 - 81x^2 + 9x - 1 -/
def f : List ℤ := [5, 4, 1, 3, -81, 9, -1]

theorem horner_rule_operations :
  numMultiplications f = 6 ∧ numAdditions f = 6 :=
sorry

end horner_rule_operations_l1633_163384


namespace absolute_value_minus_sqrt_l1633_163324

theorem absolute_value_minus_sqrt (a : ℝ) (h : a < -1) : |1 + a| - Real.sqrt (a^2) = -1 := by
  sorry

end absolute_value_minus_sqrt_l1633_163324


namespace exist_three_numbers_equal_sum_l1633_163354

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: Existence of three different natural numbers with equal sum of number and its digits -/
theorem exist_three_numbers_equal_sum :
  ∃ (m n p : ℕ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ m + S m = n + S n ∧ n + S n = p + S p :=
sorry

end exist_three_numbers_equal_sum_l1633_163354


namespace set_operations_and_subset_l1633_163376

def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 2}

theorem set_operations_and_subset :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  (∀ a : ℝ, C a ⊆ (A ∪ B) → 2 ≤ a ∧ a ≤ 8) :=
by sorry

end set_operations_and_subset_l1633_163376


namespace no_distributive_laws_hold_l1633_163379

-- Define the * operation
def star (a b : ℝ) : ℝ := a + b + a * b

-- Theorem statement
theorem no_distributive_laws_hold :
  ∃ x y z : ℝ,
    (star x (y + z) ≠ star (star x y) (star x z)) ∧
    (x + star y z ≠ star (x + y) (x + z)) ∧
    (star x (star y z) ≠ star (star x y) (star x z)) :=
by
  sorry

end no_distributive_laws_hold_l1633_163379


namespace gcd_max_value_l1633_163356

theorem gcd_max_value (m : ℕ+) : 
  Nat.gcd (14 * m.val + 4) (9 * m.val + 2) ≤ 8 ∧ 
  ∃ n : ℕ+, Nat.gcd (14 * n.val + 4) (9 * n.val + 2) = 8 := by
  sorry

end gcd_max_value_l1633_163356


namespace conditional_probability_not_first_class_l1633_163395

def total_products : ℕ := 8
def first_class_products : ℕ := 6
def selected_products : ℕ := 2

theorem conditional_probability_not_first_class 
  (h1 : total_products = 8)
  (h2 : first_class_products = 6)
  (h3 : selected_products = 2)
  (h4 : first_class_products < total_products)
  (h5 : selected_products ≤ total_products) :
  (Nat.choose first_class_products 1 * Nat.choose (total_products - first_class_products) 1) / 
  (Nat.choose total_products selected_products - Nat.choose first_class_products selected_products) = 12 / 13 :=
sorry

end conditional_probability_not_first_class_l1633_163395


namespace graphs_intersection_l1633_163317

/-- 
Given non-zero real numbers k and b, this theorem states that the graphs of 
y = kx + b and y = kb/x can only intersect in the first and third quadrants when kb > 0.
-/
theorem graphs_intersection (k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0) :
  (∀ x y : ℝ, y = k * x + b ∧ y = k * b / x → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) ↔ k * b > 0 :=
by sorry

end graphs_intersection_l1633_163317


namespace egg_packing_problem_l1633_163328

/-- The number of baskets containing eggs -/
def num_baskets : ℕ := 21

/-- The number of eggs in each basket -/
def eggs_per_basket : ℕ := 48

/-- The number of eggs each box can hold -/
def eggs_per_box : ℕ := 28

/-- The number of boxes needed to pack all the eggs -/
def boxes_needed : ℕ := (num_baskets * eggs_per_basket) / eggs_per_box

theorem egg_packing_problem : boxes_needed = 36 := by
  sorry

end egg_packing_problem_l1633_163328


namespace even_digit_sum_pairs_count_l1633_163377

/-- Given a natural number, returns true if its digit sum is even -/
def has_even_digit_sum (n : ℕ) : Bool :=
  sorry

/-- Returns the count of natural numbers less than 10^6 where both
    the number and its successor have even digit sums -/
def count_even_digit_sum_pairs : ℕ :=
  sorry

/-- The main theorem stating that the count of natural numbers less than 10^6
    where both the number and its successor have even digit sums is 45454 -/
theorem even_digit_sum_pairs_count :
  count_even_digit_sum_pairs = 45454 := by sorry

end even_digit_sum_pairs_count_l1633_163377


namespace least_integer_square_48_more_than_double_l1633_163319

theorem least_integer_square_48_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 48 ∧ ∀ y : ℤ, y^2 = 2*y + 48 → x ≤ y :=
by sorry

end least_integer_square_48_more_than_double_l1633_163319


namespace tank_filling_ratio_l1633_163300

/-- Proves that the ratio of initial water to total capacity is 1/2 given specific tank conditions -/
theorem tank_filling_ratio 
  (capacity : ℝ) 
  (inflow_rate : ℝ) 
  (outflow_rate1 : ℝ) 
  (outflow_rate2 : ℝ) 
  (fill_time : ℝ) 
  (h1 : capacity = 10) 
  (h2 : inflow_rate = 0.5) 
  (h3 : outflow_rate1 = 0.25) 
  (h4 : outflow_rate2 = 1/6) 
  (h5 : fill_time = 60) : 
  (capacity - (inflow_rate - outflow_rate1 - outflow_rate2) * fill_time) / capacity = 1/2 := by
  sorry

end tank_filling_ratio_l1633_163300


namespace outside_bookshop_discount_l1633_163301

/-- The price of a math textbook in the school bookshop -/
def school_price : ℝ := 45

/-- The amount Peter saves by buying 3 math textbooks from outside bookshops -/
def savings : ℝ := 27

/-- The number of textbooks Peter buys -/
def num_textbooks : ℕ := 3

/-- The percentage discount offered by outside bookshops -/
def discount_percentage : ℝ := 20

theorem outside_bookshop_discount :
  let outside_price := school_price - (savings / num_textbooks)
  discount_percentage = (school_price - outside_price) / school_price * 100 :=
by sorry

end outside_bookshop_discount_l1633_163301


namespace sequence_general_term_l1633_163381

theorem sequence_general_term (a : ℕ → ℝ) :
  (∀ n > 0, a (n - 1)^2 = a n^2 + 4) →
  a 1 = 1 →
  (∀ n > 0, a n > 0) →
  ∀ n > 0, a n = Real.sqrt (4 * n - 3) := by
  sorry

end sequence_general_term_l1633_163381


namespace small_bottles_sold_percentage_l1633_163383

theorem small_bottles_sold_percentage 
  (initial_small : ℕ) 
  (initial_big : ℕ) 
  (big_sold_percent : ℚ) 
  (total_remaining : ℕ) :
  initial_small = 6000 →
  initial_big = 10000 →
  big_sold_percent = 15/100 →
  total_remaining = 13780 →
  ∃ (small_sold_percent : ℚ),
    small_sold_percent = 12/100 ∧
    (initial_small * (1 - small_sold_percent)).floor + 
    (initial_big * (1 - big_sold_percent)).floor = total_remaining :=
by sorry

end small_bottles_sold_percentage_l1633_163383


namespace exists_functions_satisfying_equations_l1633_163337

/-- A function defined on non-zero real numbers -/
def NonZeroRealFunction := {f : ℝ → ℝ // ∀ x ≠ 0, f x ≠ 0}

/-- The property that f and g satisfy the given equations -/
def SatisfiesEquations (f g : NonZeroRealFunction) : Prop :=
  ∀ x ≠ 0, f.val x + g.val (1/x) = x ∧ g.val x + f.val (1/x) = 1/x

theorem exists_functions_satisfying_equations :
  ∃ f g : NonZeroRealFunction, SatisfiesEquations f g ∧ f.val 1 = 1/2 ∧ g.val 1 = 1/2 := by
  sorry

end exists_functions_satisfying_equations_l1633_163337


namespace average_weight_increase_l1633_163340

theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 8 * initial_average
  let new_total := initial_total - 50 + 70
  let new_average := new_total / 8
  new_average - initial_average = 2.5 := by sorry

end average_weight_increase_l1633_163340


namespace parallel_vectors_imply_lambda_l1633_163338

/-- Given two 2D vectors a and b, if a + 3b is parallel to 2a - b, then the second component of b is -8/3 -/
theorem parallel_vectors_imply_lambda (a b : ℝ × ℝ) (h : a = (-3, 2) ∧ b.1 = 4) :
  (∃ (k : ℝ), k ≠ 0 ∧ k • (a + 3 • b) = 2 • a - b) → b.2 = -8/3 := by
  sorry

end parallel_vectors_imply_lambda_l1633_163338


namespace graph_is_two_intersecting_lines_l1633_163390

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop := x^2 - y^2 = 0

/-- Definition of two intersecting lines -/
def two_intersecting_lines (f : ℝ → ℝ → Prop) : Prop :=
  ∃ g h : ℝ → ℝ, (∀ x y, f x y ↔ (y = g x ∨ y = h x)) ∧
  (∃ x₀, g x₀ ≠ h x₀)

/-- Theorem stating that the graph of x^2 - y^2 = 0 represents two intersecting lines -/
theorem graph_is_two_intersecting_lines :
  two_intersecting_lines graph_equation := by sorry

end graph_is_two_intersecting_lines_l1633_163390


namespace new_cost_percentage_l1633_163311

/-- The cost function -/
def cost (t c a x : ℝ) (n : ℕ) : ℝ := t * c * (a * x) ^ n

/-- Theorem stating the relationship between the original and new cost -/
theorem new_cost_percentage (t c a x : ℝ) (n : ℕ) :
  let O := cost t c a x n
  let E := cost t (2*c) (2*a) x (n+2)
  E = 2^(n+1) * x^2 * O :=
by sorry

end new_cost_percentage_l1633_163311


namespace hyperbola_and_chord_equation_l1633_163389

-- Define the hyperbola C
def hyperbola_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 18 + y^2 / 14 = 1

-- Define the common focal point condition
def common_focal_point (C : (ℝ → ℝ → Prop)) (E : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), C x y ∧ E x y

-- Define point A on hyperbola C
def point_A_on_C (C : (ℝ → ℝ → Prop)) : Prop :=
  C 3 (Real.sqrt 7)

-- Define point P as midpoint of chord AB
def point_P_midpoint (C : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ 1 = (x₁ + x₂) / 2 ∧ 2 = (y₁ + y₂) / 2

-- Main theorem
theorem hyperbola_and_chord_equation :
  ∀ (a b : ℝ),
    (hyperbola_C 3 (Real.sqrt 7) a b) →
    (common_focal_point (hyperbola_C · · a b) ellipse) →
    (point_A_on_C (hyperbola_C · · a b)) →
    (point_P_midpoint (hyperbola_C · · a b)) →
    (∀ (x y : ℝ), hyperbola_C x y a b ↔ x^2 / 2 - y^2 / 2 = 1) ∧
    (∃ (m c : ℝ), ∀ (x y : ℝ), (hyperbola_C x y a b ∧ y = m * x + c) → x - 2 * y + 3 = 0) :=
sorry

end hyperbola_and_chord_equation_l1633_163389


namespace f_satisfies_properties_l1633_163386

-- Define the function f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_satisfies_properties :
  -- Property 1: f(x₁x₂) = f(x₁)f(x₂)
  (∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂) ∧
  -- Property 2: f'(x) > 0 for x ∈ (0, +∞)
  (∀ x : ℝ, x > 0 → HasDerivAt f (2 * x) x) ∧
  (∀ x : ℝ, x > 0 → 2 * x > 0) ∧
  -- Property 3: f'(x) is an odd function
  (∀ x : ℝ, HasDerivAt f (2 * (-x)) (-x) ∧ HasDerivAt f (2 * x) x ∧ 2 * (-x) = -(2 * x)) := by
  sorry

end f_satisfies_properties_l1633_163386


namespace remaining_water_bottles_l1633_163358

/-- Calculates the number of remaining water bottles after a soccer match --/
theorem remaining_water_bottles (initial_bottles : ℕ) 
  (first_break_players : ℕ) (first_break_bottles_per_player : ℕ)
  (second_break_players : ℕ) (second_break_extra_bottles : ℕ)
  (third_break_players : ℕ) : 
  initial_bottles = 5 * 12 →
  first_break_players = 11 →
  first_break_bottles_per_player = 2 →
  second_break_players = 14 →
  second_break_extra_bottles = 4 →
  third_break_players = 12 →
  initial_bottles - 
  (first_break_players * first_break_bottles_per_player +
   second_break_players + second_break_extra_bottles +
   third_break_players) = 8 := by
sorry

end remaining_water_bottles_l1633_163358


namespace trigonometric_identities_trigonometric_value_l1633_163388

theorem trigonometric_identities (α : Real) :
  (Real.tan (3 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-α - Real.pi) * Real.sin (-Real.pi + α) * Real.cos (α + 5 * Real.pi / 2)) = -1 / Real.sin α :=
by sorry

theorem trigonometric_value (α : Real) (h : Real.tan α = 1/4) :
  1 / (2 * Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α) = 17/20 :=
by sorry

end trigonometric_identities_trigonometric_value_l1633_163388


namespace square_area_ratio_l1633_163385

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 16 * b) : a^2 = 16 * b^2 := by
  sorry

end square_area_ratio_l1633_163385


namespace paul_dog_food_needed_l1633_163313

/-- The amount of dog food needed per day for a given weight in pounds -/
def dogFoodNeeded (weight : ℕ) : ℕ := weight / 10

/-- The total weight of Paul's dogs in pounds -/
def totalDogWeight : ℕ := 20 + 40 + 10 + 30 + 50

/-- Theorem: Paul needs 15 pounds of dog food per day for his five dogs -/
theorem paul_dog_food_needed : dogFoodNeeded totalDogWeight = 15 := by
  sorry

end paul_dog_food_needed_l1633_163313


namespace distribution_theorem_l1633_163333

/-- Represents the number of communities --/
def n : ℕ := 5

/-- Represents the number of fitness equipment --/
def k : ℕ := 7

/-- Represents the number of communities that must receive at least 2 items --/
def m : ℕ := 2

/-- Represents the minimum number of items each of the m communities must receive --/
def min_items : ℕ := 2

/-- The number of ways to distribute k identical items among n recipients,
    where m specific recipients must receive at least min_items each --/
def distribution_schemes (n k m min_items : ℕ) : ℕ := sorry

theorem distribution_theorem : distribution_schemes n k m min_items = 35 := by sorry

end distribution_theorem_l1633_163333


namespace dye_mixture_volume_l1633_163351

theorem dye_mixture_volume : 
  let water_volume : ℚ := 20 * (3/5)
  let vinegar_volume : ℚ := 18 * (5/6)
  water_volume + vinegar_volume = 27
  := by sorry

end dye_mixture_volume_l1633_163351


namespace triangle_incenter_inequality_l1633_163312

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the intersection points of angle bisectors with opposite sides
def angleBisectorIntersection (t : Triangle) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_incenter_inequality (t : Triangle) :
  let I := incenter t
  let (A', B', C') := angleBisectorIntersection t
  let ratio := (distance I t.A * distance I t.B * distance I t.C) /
               (distance A' t.A * distance B' t.B * distance C' t.C)
  (1 / 4 : ℝ) < ratio ∧ ratio ≤ (8 / 27 : ℝ) := by sorry

end triangle_incenter_inequality_l1633_163312


namespace choir_members_count_l1633_163303

theorem choir_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 300 ∧ 
  n % 6 = 1 ∧ 
  n % 8 = 3 ∧ 
  n % 9 = 2 := by
  sorry

end choir_members_count_l1633_163303


namespace graph_symmetry_l1633_163331

-- Define a general real-valued function
variable (f : ℝ → ℝ)

-- Define the symmetry property about the y-axis
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Theorem statement
theorem graph_symmetry (f : ℝ → ℝ) : 
  symmetric_about_y_axis f ↔ 
  ∀ x y : ℝ, (x, y) ∈ (Set.range (λ x => (x, f x))) ↔ 
              (-x, y) ∈ (Set.range (λ x => (x, f x))) :=
sorry

end graph_symmetry_l1633_163331


namespace choir_average_age_l1633_163309

theorem choir_average_age
  (num_females : ℕ)
  (num_males : ℕ)
  (avg_age_females : ℚ)
  (avg_age_males : ℚ)
  (total_people : ℕ)
  (h1 : num_females = 12)
  (h2 : num_males = 15)
  (h3 : avg_age_females = 28)
  (h4 : avg_age_males = 35)
  (h5 : total_people = num_females + num_males) :
  (num_females * avg_age_females + num_males * avg_age_males) / total_people = 31.89 := by
  sorry

end choir_average_age_l1633_163309


namespace hypotenuse_of_right_isosceles_triangle_l1633_163342

-- Define the triangle
def right_isosceles_triangle (leg : ℝ) (hypotenuse : ℝ) : Prop :=
  leg > 0 ∧ hypotenuse > 0 ∧ hypotenuse^2 = 2 * leg^2

-- Theorem statement
theorem hypotenuse_of_right_isosceles_triangle :
  ∀ (leg : ℝ) (hypotenuse : ℝ),
  right_isosceles_triangle leg hypotenuse →
  leg = 8 →
  hypotenuse = 8 * Real.sqrt 2 :=
by sorry

end hypotenuse_of_right_isosceles_triangle_l1633_163342


namespace yard_length_26_trees_l1633_163367

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (distance_between : ℕ) : ℕ :=
  (num_trees - 1) * distance_between

/-- Theorem: The length of a yard with 26 trees planted at equal distances,
    with one tree at each end and 16 meters between consecutive trees, is 400 meters. -/
theorem yard_length_26_trees : yard_length 26 16 = 400 := by
  sorry

end yard_length_26_trees_l1633_163367


namespace union_complement_equality_l1633_163397

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_equality : M ∪ (U \ N) = {0, 2, 4, 6, 8} := by
  sorry

end union_complement_equality_l1633_163397


namespace sample_size_calculation_l1633_163366

/-- Given three districts with population ratios 2:3:5 and 100 people sampled from the largest district,
    the total sample size is 200. -/
theorem sample_size_calculation (ratio_a ratio_b ratio_c : ℕ) 
  (largest_district_sample : ℕ) :
  ratio_a = 2 → ratio_b = 3 → ratio_c = 5 → 
  largest_district_sample = 100 →
  (ratio_a + ratio_b + ratio_c : ℚ) * largest_district_sample / ratio_c = 200 :=
by sorry

end sample_size_calculation_l1633_163366


namespace bird_tree_stone_ratio_l1633_163393

theorem bird_tree_stone_ratio :
  let num_stones : ℕ := 40
  let num_trees : ℕ := 3 * num_stones
  let num_birds : ℕ := 400
  let combined_trees_stones : ℕ := num_trees + num_stones
  (num_birds : ℚ) / combined_trees_stones = 2 / 1 := by
  sorry

end bird_tree_stone_ratio_l1633_163393


namespace order_of_logarithms_and_root_l1633_163326

theorem order_of_logarithms_and_root (a b c : ℝ) : 
  a = 2 * Real.log 0.99 → 
  b = Real.log 0.98 → 
  c = Real.sqrt 0.96 - 1 → 
  c < b ∧ b < a := by
sorry

end order_of_logarithms_and_root_l1633_163326
