import Mathlib

namespace negation_of_proposition_l2401_240129

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ a : ℝ, a ≥ 0 → a^4 + a^2 ≥ 0)) ↔ (∃ a : ℝ, a ≥ 0 ∧ a^4 + a^2 < 0) :=
by sorry

end negation_of_proposition_l2401_240129


namespace min_value_function_l2401_240187

theorem min_value_function (x : ℝ) (h : x > 2) : 
  (x^2 - 4*x + 8) / (x - 2) ≥ 4 ∧ ∃ y > 2, (y^2 - 4*y + 8) / (y - 2) = 4 :=
sorry

end min_value_function_l2401_240187


namespace domain_of_f_l2401_240172

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x - 3) ^ (1/3) + (5 - 2 * x) ^ (1/3)

-- State the theorem
theorem domain_of_f :
  ∀ x : ℝ, ∃ y : ℝ, f x = y :=
by
  sorry

end domain_of_f_l2401_240172


namespace quadratic_inequality_solution_l2401_240162

theorem quadratic_inequality_solution (x : ℝ) :
  -3 * x^2 + 5 * x + 4 < 0 ↔ x < 3/4 ∨ x > 1 := by
  sorry

end quadratic_inequality_solution_l2401_240162


namespace A_intersect_B_empty_l2401_240161

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem A_intersect_B_empty : A ∩ B = ∅ := by sorry

end A_intersect_B_empty_l2401_240161


namespace not_sufficient_nor_necessary_l2401_240137

/-- Two lines ax + 2y = 3 and x + (a-1)y = 1 are parallel -/
def are_parallel (a : ℝ) : Prop :=
  a = 3 ∨ a = -1

/-- a = 2 is a sufficient condition for parallelism -/
def is_sufficient : Prop :=
  ∀ a : ℝ, a = 2 → are_parallel a

/-- a = 2 is a necessary condition for parallelism -/
def is_necessary : Prop :=
  ∀ a : ℝ, are_parallel a → a = 2

theorem not_sufficient_nor_necessary : ¬is_sufficient ∧ ¬is_necessary :=
sorry

end not_sufficient_nor_necessary_l2401_240137


namespace min_value_theorem_l2401_240165

-- Define the equation
def equation (x y : ℝ) : Prop := y^2 - 2*x + 4 = 0

-- Define the expression to minimize
def expression (x y : ℝ) : ℝ := x^2 + y^2 + 2*x

-- Theorem statement
theorem min_value_theorem :
  ∃ (min : ℝ), min = -8 ∧
  (∀ (x y : ℝ), equation x y → expression x y ≥ min) ∧
  (∃ (x y : ℝ), equation x y ∧ expression x y = min) :=
sorry

end min_value_theorem_l2401_240165


namespace seven_balls_two_boxes_l2401_240176

def distribute_balls (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n / 2 + 1)) (λ k => Nat.choose n k)

theorem seven_balls_two_boxes : distribute_balls 7 = 64 := by
  sorry

end seven_balls_two_boxes_l2401_240176


namespace emily_flight_remaining_time_l2401_240125

/-- Given a flight duration and a series of activities, calculate the remaining time -/
def remaining_flight_time (flight_duration : ℕ) (tv_episodes : ℕ) (tv_episode_duration : ℕ) 
  (sleep_duration : ℕ) (movies : ℕ) (movie_duration : ℕ) : ℕ :=
  flight_duration - (tv_episodes * tv_episode_duration + sleep_duration + movies * movie_duration)

/-- Theorem: Given Emily's flight and activities, prove that 45 minutes remain -/
theorem emily_flight_remaining_time : 
  remaining_flight_time 600 3 25 270 2 105 = 45 := by
  sorry

end emily_flight_remaining_time_l2401_240125


namespace surface_generates_solid_l2401_240141

/-- A right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ

/-- A cone formed by rotating a right-angled triangle -/
structure Cone where
  base_radius : ℝ
  height : ℝ

/-- Rotation of a right-angled triangle around one of its right-angle sides -/
def rotate_triangle (t : RightTriangle) : Cone :=
  { base_radius := t.base, height := t.height }

/-- Theorem: Rotating a right-angled triangle generates a solid (cone) -/
theorem surface_generates_solid (t : RightTriangle) :
  ∃ (c : Cone), c = rotate_triangle t :=
sorry

end surface_generates_solid_l2401_240141


namespace hockey_league_games_l2401_240149

theorem hockey_league_games (n : ℕ) (total_games : ℕ) 
  (hn : n = 17) (htotal : total_games = 1360) : 
  (total_games * 2) / (n * (n - 1)) = 5 := by
  sorry

end hockey_league_games_l2401_240149


namespace quadratic_roots_problem_l2401_240128

theorem quadratic_roots_problem (a b : ℝ) : 
  (∀ t : ℝ, t^2 - 12*t + 20 = 0 ↔ t = a ∨ t = b) →
  a > b →
  a - b = 8 →
  b = 2 := by sorry

end quadratic_roots_problem_l2401_240128


namespace trig_expression_equality_l2401_240171

theorem trig_expression_equality : 
  (Real.sin (20 * π / 180) * Real.sqrt (1 + Real.cos (40 * π / 180))) / 
  (Real.cos (50 * π / 180)) = Real.sqrt 2 / 2 := by
  sorry

end trig_expression_equality_l2401_240171


namespace hexagon_area_right_triangle_l2401_240182

/-- Given a right-angled triangle with hypotenuse c and sum of legs d,
    the area of the hexagon formed by the outer vertices of squares
    drawn on the sides of the triangle is c^2 + d^2. -/
theorem hexagon_area_right_triangle (c d : ℝ) (h : c > 0) (h' : d > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = d ∧ a^2 + b^2 = c^2 ∧
  (c^2 + a^2 + b^2 + 2*a*b : ℝ) = c^2 + d^2 := by
  sorry

end hexagon_area_right_triangle_l2401_240182


namespace negation_of_existence_negation_of_inequality_l2401_240138

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_inequality :
  (¬ ∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) ↔ (∀ x : ℝ, |x - 2| + |x - 4| > 3) :=
by sorry

end negation_of_existence_negation_of_inequality_l2401_240138


namespace ratio_equivalences_l2401_240127

theorem ratio_equivalences (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a * d = b * c) : 
  (a / b = c / d) ∧ (b / a = d / c) ∧ (a / c = b / d) := by
  sorry

end ratio_equivalences_l2401_240127


namespace correct_operation_l2401_240131

theorem correct_operation (a b : ℝ) : 5 * a * b - 3 * a * b = 2 * a * b := by
  sorry

end correct_operation_l2401_240131


namespace triangle_angle_c_ninety_degrees_l2401_240179

/-- Given a triangle ABC with sides a, b, c and angles A, B, C respectively,
    if b^2 + c^2 - bc = a^2 and a/b = √3, then angle C = 90°. -/
theorem triangle_angle_c_ninety_degrees 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_triangle : A + B + C = Real.pi)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : b^2 + c^2 - b*c = a^2)
  (h_ratio : a/b = Real.sqrt 3) : 
  C = Real.pi/2 := by
sorry


end triangle_angle_c_ninety_degrees_l2401_240179


namespace gcd_of_175_100_75_base_conversion_l2401_240105

-- Part 1: GCD of 175, 100, and 75
theorem gcd_of_175_100_75 : Nat.gcd 175 (Nat.gcd 100 75) = 25 := by sorry

-- Part 2: Base conversion
def base_6_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

def decimal_to_base_8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem base_conversion :
  (base_6_to_decimal [5, 1, 0, 1] = 227) ∧
  (decimal_to_base_8 227 = [3, 4, 3]) := by sorry

end gcd_of_175_100_75_base_conversion_l2401_240105


namespace number_property_l2401_240111

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : ℕ),
    0 < k ∧ 
    1 ≤ a ∧ a ≤ 9 ∧
    n = 12 * a ∧
    n % 10 ≠ 0

theorem number_property :
  ∀ (N : ℕ),
    (N % 10 ≠ 0) →
    (∃ (N' : ℕ), 
      (∃ (k : ℕ) (m : ℕ) (n : ℕ),
        N = m + 10^k * (N' / 10^k % 10) + 10^(k+1) * n ∧
        N' = m + 10^(k+1) * n ∧
        m < 10^k) ∧
      N = 6 * N') →
    is_valid_number N :=
sorry

end number_property_l2401_240111


namespace parabola_intersection_theorem_l2401_240180

/-- A parabola with parameter p > 0 and two points A and B on it, intersected by a line through its focus F -/
structure ParabolaWithIntersection where
  p : ℝ
  hp : p > 0
  A : ℝ × ℝ
  B : ℝ × ℝ
  hA : A.2^2 = 2 * p * A.1
  hB : B.2^2 = 2 * p * B.1
  hAF : A.1 = 3 - p/2
  hBF : B.1 = 2 - p/2

/-- The theorem stating that under the given conditions, p = 12/5 -/
theorem parabola_intersection_theorem (pwi : ParabolaWithIntersection) : pwi.p = 12/5 := by
  sorry

end parabola_intersection_theorem_l2401_240180


namespace circle_equation_l2401_240147

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The equation of a line ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem circle_equation (C : Circle) (L : Line) (P1 P2 : Point) :
  L.a = 2 ∧ L.b = 1 ∧ L.c = -1 →  -- Line equation: 2x + y - 1 = 0
  C.h * L.a + C.k * L.b + L.c = 0 →  -- Center is on the line
  (0 - C.h)^2 + (0 - C.k)^2 = C.r^2 →  -- Circle passes through origin
  (P1.x - C.h)^2 + (P1.y - C.k)^2 = C.r^2 →  -- Circle passes through P1
  P1.x = -1 ∧ P1.y = -5 →  -- P1 coordinates
  C.h = 2 ∧ C.k = -3 ∧ C.r^2 = 13 →  -- Circle equation coefficients
  ∀ x y : ℝ, (x - C.h)^2 + (y - C.k)^2 = C.r^2 ↔ (x - 2)^2 + (y + 3)^2 = 13 :=
by sorry

end circle_equation_l2401_240147


namespace tv_sales_effect_l2401_240158

theorem tv_sales_effect (P Q : ℝ) (h_pos_P : P > 0) (h_pos_Q : Q > 0) :
  let new_price := 0.8 * P
  let new_quantity := 1.8 * Q
  let original_value := P * Q
  let new_value := new_price * new_quantity
  (new_value - original_value) / original_value = 0.44 := by
sorry

end tv_sales_effect_l2401_240158


namespace find_Z_l2401_240133

theorem find_Z : ∃ Z : ℝ, (100 + 20 / Z) * Z = 9020 ∧ Z = 90 := by
  sorry

end find_Z_l2401_240133


namespace solution_set_of_inequality_l2401_240175

theorem solution_set_of_inequality (x : ℝ) : 
  (2 * x) / (3 * x - 1) > 1 ↔ 1/3 < x ∧ x < 1 := by
  sorry

end solution_set_of_inequality_l2401_240175


namespace hundredth_digit_of_17_over_99_l2401_240146

/-- The 100th digit after the decimal point in the decimal representation of 17/99 is 7 -/
theorem hundredth_digit_of_17_over_99 : ∃ (d : ℕ), d = 7 ∧ 
  (∃ (a b : ℕ) (s : List ℕ), 
    (17 : ℚ) / 99 = (a : ℚ) + (b : ℚ) / 10 + (s.foldr (λ x acc => acc / 10 + (x : ℚ) / 10) 0) ∧
    s.length = 99 ∧
    d = s.reverse.head!) :=
sorry

end hundredth_digit_of_17_over_99_l2401_240146


namespace complex_equation_solution_l2401_240177

theorem complex_equation_solution (z : ℂ) (h : Complex.I * z = 2 + 3 * Complex.I) : z = 3 - 2 * Complex.I := by
  sorry

end complex_equation_solution_l2401_240177


namespace repeating_decimal_fraction_l2401_240120

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a repeating decimal of the form 0.ẋyȧ -/
def RepeatingDecimal (x y : Digit) : ℚ :=
  (x.val * 100 + y.val * 10 + 3) / 999

/-- The theorem to be proved -/
theorem repeating_decimal_fraction (x y : Digit) (a : ℤ) 
  (h1 : x ≠ y)
  (h2 : RepeatingDecimal x y = a / 27) :
  a = 37 := by
  sorry

end repeating_decimal_fraction_l2401_240120


namespace hex_grid_half_path_l2401_240194

/-- Represents a point in the hexagonal grid -/
structure HexPoint where
  x : ℤ
  y : ℤ

/-- Represents a direction in the hexagonal grid -/
inductive HexDirection
  | Horizontal
  | LeftDiagonal
  | RightDiagonal

/-- Calculates the distance between two points in the hexagonal grid -/
def hexDistance (a b : HexPoint) : ℕ :=
  sorry

/-- Represents a path in the hexagonal grid -/
def HexPath := List HexDirection

/-- Checks if a path is valid (follows the grid lines) -/
def isValidPath (path : HexPath) (start finish : HexPoint) : Prop :=
  sorry

/-- Calculates the length of a path -/
def pathLength (path : HexPath) : ℕ :=
  sorry

/-- Checks if a path is the shortest between two points -/
def isShortestPath (path : HexPath) (start finish : HexPoint) : Prop :=
  isValidPath path start finish ∧
  pathLength path = hexDistance start finish

/-- Counts the number of steps in a single direction -/
def countDirectionSteps (path : HexPath) (direction : HexDirection) : ℕ :=
  sorry

theorem hex_grid_half_path (a b : HexPoint) (path : HexPath) :
  isShortestPath path a b →
  hexDistance a b = 100 →
  ∃ (direction : HexDirection), countDirectionSteps path direction = 50 :=
sorry

end hex_grid_half_path_l2401_240194


namespace geese_count_l2401_240153

def geese_problem (n : ℕ) : Prop :=
  -- The number of geese is an integer (implied by ℕ)
  -- After each lake, the number of remaining geese is an integer
  (∀ k : ℕ, k ≤ 7 → ∃ m : ℕ, n * 2^(7 - k) - (2^(7 - k) - 1) = m) ∧
  -- The process continues for exactly 7 lakes
  -- At each lake, half of the remaining geese plus half a goose land (implied by the formula)
  -- After 7 lakes, no geese remain
  n * 2^0 - (2^0 - 1) = 0

theorem geese_count : ∃ n : ℕ, geese_problem n ∧ n = 127 := by
  sorry

end geese_count_l2401_240153


namespace total_weight_of_sand_l2401_240101

/-- The total weight of sand in two jugs with different capacities and sand densities -/
theorem total_weight_of_sand (jug1_capacity jug2_capacity : ℝ)
  (fill_percentage : ℝ)
  (density1 density2 : ℝ) :
  jug1_capacity = 2 →
  jug2_capacity = 3 →
  fill_percentage = 0.7 →
  density1 = 4 →
  density2 = 5 →
  jug1_capacity * fill_percentage * density1 +
  jug2_capacity * fill_percentage * density2 = 16.1 := by
  sorry

#check total_weight_of_sand

end total_weight_of_sand_l2401_240101


namespace similar_right_triangles_shortest_side_l2401_240197

theorem similar_right_triangles_shortest_side 
  (leg1 : ℝ) (hyp1 : ℝ) (hyp2 : ℝ) 
  (h_right : leg1 ^ 2 + (hyp1 ^ 2 - leg1 ^ 2) = hyp1 ^ 2) 
  (h_leg1 : leg1 = 15) 
  (h_hyp1 : hyp1 = 17) 
  (h_hyp2 : hyp2 = 51) : 
  (leg1 * hyp2 / hyp1) = 24 := by sorry

end similar_right_triangles_shortest_side_l2401_240197


namespace point_on_line_l2401_240151

theorem point_on_line (m n p : ℝ) : 
  (m = n / 5 - 2 / 5) ∧ (m + p = (n + 15) / 5 - 2 / 5) → p = 3 := by
  sorry

end point_on_line_l2401_240151


namespace polynomial_remainder_theorem_l2401_240115

-- Define the polynomial g(x)
def g (c d x : ℝ) : ℝ := c * x^3 - 7 * x^2 + d * x - 4

-- State the theorem
theorem polynomial_remainder_theorem (c d : ℝ) :
  (g c d 2 = -4) ∧ (g c d (-1) = -22) → c = 19/3 ∧ d = -8/3 := by
  sorry

end polynomial_remainder_theorem_l2401_240115


namespace domain_of_f_sqrt_x_minus_2_l2401_240181

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-1) 0

-- State the theorem
theorem domain_of_f_sqrt_x_minus_2 :
  (∀ x ∈ domain_f_x_plus_1, f (x + 1) ∈ Set.Icc 0 1) →
  {x : ℝ | f (Real.sqrt x - 2) ∈ Set.Icc 0 1} = Set.Icc 4 9 :=
by sorry

end domain_of_f_sqrt_x_minus_2_l2401_240181


namespace students_wanting_fruit_l2401_240184

theorem students_wanting_fruit (red_apples green_apples extra_fruit : ℕ) 
  (h1 : red_apples = 42)
  (h2 : green_apples = 7)
  (h3 : extra_fruit = 40) :
  red_apples + green_apples - extra_fruit = 40 := by
  sorry

end students_wanting_fruit_l2401_240184


namespace cubic_expression_equals_1000_l2401_240164

theorem cubic_expression_equals_1000 (α : ℝ) (h : α = 6) : 
  α^3 + 3*α^2*4 + 3*α*16 + 64 = 1000 := by
  sorry

end cubic_expression_equals_1000_l2401_240164


namespace ellipse_major_axis_length_l2401_240163

/-- The length of the major axis of an ellipse with given foci and y-axis tangency -/
theorem ellipse_major_axis_length : 
  let f₁ : ℝ × ℝ := (1, -3 + 2 * Real.sqrt 3)
  let f₂ : ℝ × ℝ := (1, -3 - 2 * Real.sqrt 3)
  ∀ (e : Set (ℝ × ℝ)), 
    (∃ (p : ℝ × ℝ), p ∈ e ∧ p.1 = 0) →  -- Tangent to y-axis
    (∀ (q : ℝ × ℝ), q ∈ e → ∃ (a b : ℝ), a * (q.1 - f₁.1)^2 + b * (q.2 - f₁.2)^2 = 1 ∧ 
                                         a * (q.1 - f₂.1)^2 + b * (q.2 - f₂.2)^2 = 1) →
    (∃ (major_axis : ℝ), major_axis = 4 * Real.sqrt 3) :=
by sorry

end ellipse_major_axis_length_l2401_240163


namespace minimal_fraction_sum_l2401_240121

theorem minimal_fraction_sum (a b : ℕ+) (h : (45:ℚ)/110 < (a:ℚ)/(b:ℚ) ∧ (a:ℚ)/(b:ℚ) < (50:ℚ)/110) :
  (∃ (c d : ℕ+), (45:ℚ)/110 < (c:ℚ)/(d:ℚ) ∧ (c:ℚ)/(d:ℚ) < (50:ℚ)/110 ∧ c+d ≤ a+b) →
  (3:ℚ)/7 = (a:ℚ)/(b:ℚ) :=
sorry

end minimal_fraction_sum_l2401_240121


namespace scooter_profit_percentage_l2401_240108

theorem scooter_profit_percentage 
  (C : ℝ)  -- Cost of the scooter
  (h1 : C * 0.1 = 500)  -- 10% of cost spent on repairs
  (h2 : C + 1100 = C * 1.22)  -- Sold for a profit of $1100, which is 22% more than cost
  : (1100 / C) * 100 = 22 :=
by sorry

end scooter_profit_percentage_l2401_240108


namespace A_power_101_l2401_240126

def A : Matrix (Fin 3) (Fin 3) ℕ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_101 : A ^ 101 = A ^ 2 := by
  sorry

end A_power_101_l2401_240126


namespace temperature_equation_initial_temperature_temperature_increase_l2401_240112

/-- Represents the temperature in °C at a given time t in minutes -/
def temperature (t : ℝ) : ℝ := 7 * t + 30

theorem temperature_equation (t : ℝ) (h : t < 10) :
  temperature t = 7 * t + 30 :=
by sorry

theorem initial_temperature :
  temperature 0 = 30 :=
by sorry

theorem temperature_increase (t₁ t₂ : ℝ) (h₁ : t₁ < 10) (h₂ : t₂ < 10) (h₃ : t₁ < t₂) :
  temperature t₂ - temperature t₁ = 7 * (t₂ - t₁) :=
by sorry

end temperature_equation_initial_temperature_temperature_increase_l2401_240112


namespace min_distance_is_14000_l2401_240109

/-- Represents the problem of transporting and planting poles along a road -/
structure PolePlantingProblem where
  numPoles : ℕ
  startDistance : ℕ
  poleSpacing : ℕ
  maxPolesPerTrip : ℕ

/-- Calculates the minimum total distance traveled for a given pole planting problem -/
def minTotalDistance (p : PolePlantingProblem) : ℕ :=
  sorry

/-- The specific pole planting problem instance -/
def specificProblem : PolePlantingProblem :=
  { numPoles := 20
  , startDistance := 500
  , poleSpacing := 50
  , maxPolesPerTrip := 3 }

/-- Theorem stating that the minimum total distance for the specific problem is 14000 meters -/
theorem min_distance_is_14000 :
  minTotalDistance specificProblem = 14000 :=
sorry

end min_distance_is_14000_l2401_240109


namespace unique_three_digit_odd_l2401_240130

/-- A function that returns true if a number is a three-digit odd number -/
def isThreeDigitOdd (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 2 = 1

/-- A function that returns the sum of squares of digits of a number -/
def sumOfSquaresOfDigits (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a * a + b * b + c * c

/-- The main theorem stating that 803 is the only three-digit odd number
    satisfying the given condition -/
theorem unique_three_digit_odd : ∀ n : ℕ, 
  isThreeDigitOdd n ∧ (n / 11 : ℚ) = (sumOfSquaresOfDigits n : ℚ) → n = 803 :=
by
  sorry

#check unique_three_digit_odd

end unique_three_digit_odd_l2401_240130


namespace hyperbola_standard_form_l2401_240167

/-- A hyperbola with given asymptote and point -/
structure Hyperbola where
  -- Asymptote equation: 3x + 4y = 0
  asymptote_slope : ℝ
  asymptote_slope_eq : asymptote_slope = -3/4
  -- Point on the hyperbola
  point : ℝ × ℝ
  point_eq : point = (4, 6)

/-- The standard form of a hyperbola -/
def standard_form (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem stating the standard form of the hyperbola -/
theorem hyperbola_standard_form (h : Hyperbola) :
  ∃ (a b : ℝ), a^2 = 48 ∧ b^2 = 27 ∧ 
  ∀ (x y : ℝ), standard_form a b x y ↔ 
    (∃ (t : ℝ), x = 3*t ∧ y = -4*t) ∨ (x, y) = h.point :=
sorry

end hyperbola_standard_form_l2401_240167


namespace perpendicular_lines_b_value_l2401_240166

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The first line equation: y = 3x + 7 -/
def line1 (x y : ℝ) : Prop := y = 3 * x + 7

/-- The second line equation: 4y + bx = 12 -/
def line2 (x y b : ℝ) : Prop := 4 * y + b * x = 12

/-- The theorem stating that if the two given lines are perpendicular, then b = 4/3 -/
theorem perpendicular_lines_b_value (b : ℝ) :
  (∀ x y, line1 x y → line2 x y b → perpendicular 3 (-b/4)) →
  b = 4/3 := by
  sorry

end perpendicular_lines_b_value_l2401_240166


namespace angle_sum_theorem_l2401_240116

open Real

theorem angle_sum_theorem (x : ℝ) :
  (0 ≤ x ∧ x ≤ 2 * π) →
  (sin x ^ 5 - cos x ^ 5 = 1 / cos x - 1 / sin x) →
  ∃ (y : ℝ), (0 ≤ y ∧ y ≤ 2 * π) ∧
             (sin y ^ 5 - cos y ^ 5 = 1 / cos y - 1 / sin y) ∧
             x + y = 3 * π / 2 := by
  sorry

end angle_sum_theorem_l2401_240116


namespace divisibility_in_sequence_l2401_240148

theorem divisibility_in_sequence (n : ℕ) (a : Fin (n + 1) → ℤ) :
  ∃ (i j : Fin (n + 1)), i ≠ j ∧ (n : ℤ) ∣ (a i - a j) := by
  sorry

end divisibility_in_sequence_l2401_240148


namespace cinema_rows_l2401_240156

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 5^2 + tens * 5^1 + ones * 5^0

/-- Calculates the number of rows needed in a cinema -/
def calculateRows (seats : Nat) (peoplePerRow : Nat) : Nat :=
  (seats + peoplePerRow - 1) / peoplePerRow

theorem cinema_rows :
  let seatsBase5 : Nat := 312
  let peoplePerRow : Nat := 3
  let seatsBase10 : Nat := base5ToBase10 seatsBase5
  calculateRows seatsBase10 peoplePerRow = 28 := by
  sorry

end cinema_rows_l2401_240156


namespace height_difference_is_9_l2401_240135

/-- The height of the Empire State Building in meters -/
def empire_state_height : ℝ := 443

/-- The height of the Petronas Towers in meters -/
def petronas_height : ℝ := 452

/-- The height difference between the Petronas Towers and the Empire State Building -/
def height_difference : ℝ := petronas_height - empire_state_height

theorem height_difference_is_9 : height_difference = 9 := by
  sorry

end height_difference_is_9_l2401_240135


namespace johns_socks_theorem_l2401_240159

/-- The number of pairs of matched socks John initially had -/
def initial_pairs : ℕ := 9

/-- The number of individual socks John loses -/
def lost_socks : ℕ := 5

/-- The greatest number of pairs of matched socks John can have left after losing socks -/
def remaining_pairs : ℕ := 7

theorem johns_socks_theorem :
  (2 * initial_pairs - lost_socks ≥ 2 * remaining_pairs) ∧
  (2 * (initial_pairs - 1) - lost_socks < 2 * remaining_pairs) := by
  sorry

end johns_socks_theorem_l2401_240159


namespace slope_theorem_l2401_240103

/-- Given two points R(-3,9) and S(3,y) on a coordinate plane, 
    if the slope of the line through R and S is -2, then y = -3. -/
theorem slope_theorem (y : ℝ) : 
  let R : ℝ × ℝ := (-3, 9)
  let S : ℝ × ℝ := (3, y)
  let slope := (S.2 - R.2) / (S.1 - R.1)
  slope = -2 → y = -3 := by
sorry

end slope_theorem_l2401_240103


namespace roll_sum_less_than_12_prob_value_l2401_240168

def roll_sum_less_than_12_prob : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := total_outcomes - 15
  favorable_outcomes / total_outcomes

theorem roll_sum_less_than_12_prob_value : 
  roll_sum_less_than_12_prob = 49 / 64 := by sorry

end roll_sum_less_than_12_prob_value_l2401_240168


namespace inscribed_quadrilateral_property_l2401_240102

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a line -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Checks if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a point lies on a circle -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is inscribed in a circle -/
def Quadrilateral.inscribed (q : Quadrilateral) (c : Circle) : Prop :=
  q.A.onCircle c ∧ q.B.onCircle c ∧ q.C.onCircle c ∧ q.D.onCircle c

/-- Represents the tangent line at a point on a circle -/
def tangentLine (p : Point) (c : Circle) : Line :=
  sorry

/-- Checks if two lines intersect at a point -/
def Line.intersectAt (l1 l2 : Line) (p : Point) : Prop :=
  p.onLine l1 ∧ p.onLine l2

/-- Calculates the distance between two points -/
def Point.dist (p1 p2 : Point) : ℝ :=
  sorry

theorem inscribed_quadrilateral_property (c : Circle) (q : Quadrilateral) (K : Point) :
  q.inscribed c →
  Line.intersectAt (tangentLine q.B c) (tangentLine q.D c) K →
  K.onLine (Line.mk 0 1 0) →  -- Assuming y-axis is the line AC
  q.A.dist q.B * q.C.dist q.D = q.B.dist q.C * q.A.dist q.D ∧
  ∀ (P Q R : Point) (l : Line),
    l.intersectAt (Line.mk 1 0 0) P →  -- Assuming x-axis is the line BA
    l.intersectAt (Line.mk 1 1 0) Q →  -- Assuming y=x is the line BD
    l.intersectAt (Line.mk 0 1 0) R →  -- Assuming y-axis is the line BC
    P.dist Q = Q.dist R :=
by sorry

end inscribed_quadrilateral_property_l2401_240102


namespace binary_to_hex_conversion_l2401_240178

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its hexadecimal representation -/
def decimal_to_hex (n : Nat) : String :=
  let rec aux (m : Nat) (acc : String) : String :=
    if m = 0 then
      if acc.isEmpty then "0" else acc
    else
      let digit := m % 16
      let hex_digit := if digit < 10 then 
        Char.toString (Char.ofNat (digit + 48))
      else
        Char.toString (Char.ofNat (digit + 55))
      aux (m / 16) (hex_digit ++ acc)
  aux n ""

/-- The binary number 1011101₂ -/
def binary_number : List Bool := [true, false, true, true, true, false, true]

theorem binary_to_hex_conversion :
  (binary_to_decimal binary_number = 93) ∧
  (decimal_to_hex 93 = "5D") := by
  sorry

end binary_to_hex_conversion_l2401_240178


namespace average_speed_calculation_l2401_240140

def initial_reading : ℕ := 2332
def final_reading : ℕ := 2552
def time_day1 : ℕ := 6
def time_day2 : ℕ := 4

theorem average_speed_calculation :
  let total_distance : ℕ := final_reading - initial_reading
  let total_time : ℕ := time_day1 + time_day2
  (total_distance : ℚ) / total_time = 22 := by sorry

end average_speed_calculation_l2401_240140


namespace tims_weekly_earnings_l2401_240199

def visitors_per_day : ℕ := 100
def days_normal : ℕ := 6
def earnings_per_visit : ℚ := 1 / 100

def total_visitors : ℕ := visitors_per_day * days_normal + 2 * (visitors_per_day * days_normal)

def total_earnings : ℚ := (total_visitors : ℚ) * earnings_per_visit

theorem tims_weekly_earnings : total_earnings = 18 := by
  sorry

end tims_weekly_earnings_l2401_240199


namespace coin_probability_l2401_240114

theorem coin_probability (p : ℝ) : 
  (p ≥ 0 ∧ p ≤ 1) →  -- p is a probability
  (p * (1 - p)^4 = 1/32) →  -- probability of HTTT = 0.03125
  p = 1/2 := by
sorry

end coin_probability_l2401_240114


namespace complex_fraction_simplification_l2401_240150

theorem complex_fraction_simplification :
  (3 + 5*Complex.I) / (-2 + 3*Complex.I) = 9/13 - 19/13 * Complex.I :=
by sorry

end complex_fraction_simplification_l2401_240150


namespace express_train_speed_l2401_240152

/-- 
Given two trains traveling towards each other from towns 390 km apart,
where the freight train travels 30 km/h slower than the express train,
and they pass each other after 3 hours,
prove that the speed of the express train is 80 km/h.
-/
theorem express_train_speed (distance : ℝ) (time : ℝ) (speed_difference : ℝ) 
  (h1 : distance = 390)
  (h2 : time = 3)
  (h3 : speed_difference = 30) : 
  ∃ (express_speed : ℝ), 
    express_speed * time + (express_speed - speed_difference) * time = distance ∧ 
    express_speed = 80 := by
  sorry

end express_train_speed_l2401_240152


namespace additional_apples_needed_l2401_240117

def apples_needed (current_apples : ℕ) (people : ℕ) (apples_per_person : ℕ) : ℕ :=
  if people * apples_per_person ≤ current_apples then
    0
  else
    people * apples_per_person - current_apples

theorem additional_apples_needed :
  apples_needed 68 14 5 = 2 := by
  sorry

end additional_apples_needed_l2401_240117


namespace radical_conjugate_sum_product_l2401_240195

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = -6 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 4 → 
  a + b = 2 := by
sorry

end radical_conjugate_sum_product_l2401_240195


namespace fraction_equation_solution_l2401_240154

theorem fraction_equation_solution (x : ℚ) :
  (x + 4) / (x - 3) = (x - 2) / (x + 2) → x = -2 / 11 := by
  sorry

end fraction_equation_solution_l2401_240154


namespace integer_values_less_than_sqrt2_l2401_240169

theorem integer_values_less_than_sqrt2 (x : ℤ) : 
  (|x| : ℝ) < Real.sqrt 2 → x = -1 ∨ x = 0 ∨ x = 1 := by
  sorry

end integer_values_less_than_sqrt2_l2401_240169


namespace cubic_equation_solutions_l2401_240155

theorem cubic_equation_solutions (t : ℝ) :
  let f := fun x : ℝ => x^3 - 2*t*x^2 + t^3
  (f t = 0) ∧
  (f ((t*(1+Real.sqrt 5))/2) = 0) ∧
  (f ((t*(1-Real.sqrt 5))/2) = 0) :=
by sorry

end cubic_equation_solutions_l2401_240155


namespace subset_implies_zero_intersection_of_A_and_B_l2401_240139

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {x | 2 < x ∧ x < 6}

-- Theorem 1: If {x | ax = 1} is a subset of any set, then a = 0
theorem subset_implies_zero (a : ℝ) (h : ∀ S : Set ℝ, {x | a * x = 1} ⊆ S) : a = 0 := by
  sorry

-- Theorem 2: A ∩ B = {x | 2 < x ∧ x < 4}
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 4} := by
  sorry

end subset_implies_zero_intersection_of_A_and_B_l2401_240139


namespace sum_of_coefficients_l2401_240106

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₂ + a₄ = 41 := by
  sorry

end sum_of_coefficients_l2401_240106


namespace soda_bottle_duration_l2401_240104

/-- Calculates the number of days a bottle of soda will last -/
def soda_duration (bottle_volume : ℚ) (daily_consumption : ℚ) : ℚ :=
  (bottle_volume * 1000) / daily_consumption

theorem soda_bottle_duration :
  let bottle_volume : ℚ := 2
  let daily_consumption : ℚ := 500
  soda_duration bottle_volume daily_consumption = 4 := by
  sorry

end soda_bottle_duration_l2401_240104


namespace parallel_lines_coefficient_l2401_240132

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 : ℝ} : 
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) → m1 = m2

/-- The slope of a line ax + by + c = 0 is -a/b when b ≠ 0 -/
axiom slope_of_line {a b c : ℝ} (h : b ≠ 0) :
  ∃ m : ℝ, m = -a/b ∧ ∀ x y : ℝ, a*x + b*y + c = 0 ↔ y = m*x + (-c/b)

theorem parallel_lines_coefficient (a : ℝ) :
  (∀ x y : ℝ, a*x + 2*y + 2 = 0 ↔ 3*x - y - 2 = 0) → a = -6 :=
by sorry

end parallel_lines_coefficient_l2401_240132


namespace systematic_sampling_interval_72_8_l2401_240119

/-- Calculates the interval for systematic sampling. -/
def systematicSamplingInterval (totalPopulation sampleSize : ℕ) : ℕ :=
  totalPopulation / sampleSize

/-- Proves that for a population of 72 and sample size of 8, the systematic sampling interval is 9. -/
theorem systematic_sampling_interval_72_8 :
  systematicSamplingInterval 72 8 = 9 := by
  sorry

end systematic_sampling_interval_72_8_l2401_240119


namespace same_result_as_five_minus_seven_l2401_240193

theorem same_result_as_five_minus_seven : 5 - 7 = -2 := by
  sorry

end same_result_as_five_minus_seven_l2401_240193


namespace numerical_trick_l2401_240189

theorem numerical_trick (x : ℝ) : ((6 * x - 21) / 3) - 2 * x = -7 := by
  sorry

end numerical_trick_l2401_240189


namespace factorization_x4_plus_324_l2401_240170

theorem factorization_x4_plus_324 (x : ℝ) : 
  x^4 + 324 = (x^2 - 18*x + 162) * (x^2 + 18*x + 162) := by sorry

end factorization_x4_plus_324_l2401_240170


namespace murtha_pebble_collection_l2401_240142

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Murtha's pebble collection over 20 days -/
theorem murtha_pebble_collection : sum_of_first_n 20 = 210 := by
  sorry

end murtha_pebble_collection_l2401_240142


namespace white_squares_20th_row_l2401_240124

/-- The total number of squares in the nth row of the modified "stair-step" figure -/
def totalSquares (n : ℕ) : ℕ := 3 * n

/-- The number of white squares in the nth row of the modified "stair-step" figure -/
def whiteSquares (n : ℕ) : ℕ := (totalSquares n) / 2

theorem white_squares_20th_row :
  whiteSquares 20 = 30 := by sorry

end white_squares_20th_row_l2401_240124


namespace triangle_circle_relation_l2401_240113

/-- For a triangle with circumcircle radius R, excircle radius p, and distance d between their centers, d^2 = R^2 + 2Rp. -/
theorem triangle_circle_relation (R p d : ℝ) : d^2 = R^2 + 2*R*p :=
sorry

end triangle_circle_relation_l2401_240113


namespace check_amount_l2401_240122

theorem check_amount (total_parts : ℕ) (expensive_parts : ℕ) (cheap_price : ℕ) (expensive_price : ℕ) : 
  total_parts = 59 → 
  expensive_parts = 40 → 
  cheap_price = 20 → 
  expensive_price = 50 → 
  (total_parts - expensive_parts) * cheap_price + expensive_parts * expensive_price = 2380 := by
sorry

end check_amount_l2401_240122


namespace students_liking_computing_l2401_240186

theorem students_liking_computing (total : ℕ) (both : ℕ) 
  (h1 : total = 33)
  (h2 : both = 3)
  (h3 : ∀ (pe_only computing_only : ℕ), 
    pe_only + computing_only + both = total → 
    pe_only = computing_only / 2) :
  ∃ (pe_only computing_only : ℕ),
    pe_only + computing_only + both = total ∧
    pe_only = computing_only / 2 ∧
    computing_only + both = 23 := by
sorry

end students_liking_computing_l2401_240186


namespace point_on_line_l2401_240110

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line (p1 p2 p3 : Point) 
  (h1 : p1 = ⟨6, 12⟩) 
  (h2 : p2 = ⟨0, -6⟩) 
  (h3 : p3 = ⟨3, 3⟩) : 
  collinear p1 p2 p3 := by
  sorry


end point_on_line_l2401_240110


namespace circle_intersection_theorem_l2401_240123

/-- Given a circle and a line intersecting at two points, prove the value of m and the equation of the circle with the intersection points as diameter. -/
theorem circle_intersection_theorem (x y : ℝ) (m : ℝ) : 
  let circle := x^2 + y^2 - 2*x - 4*y + m
  let line := x + 2*y - 4
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (circle = 0 ∧ line = 0 → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) ∧
    (x₁ * x₂ + y₁ * y₂ = 0) →
    (m = 8/5 ∧ 
     ∀ (x y : ℝ), x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔ 
       ((x - x₁)*(x - x₂) + (y - y₁)*(y - y₂) = 0)) :=
by sorry

end circle_intersection_theorem_l2401_240123


namespace divisibility_check_l2401_240144

theorem divisibility_check (n : ℕ) : 
  n = 1493826 → 
  n % 3 = 0 ∧ 
  ¬(n % 9 = 0) := by
  sorry

end divisibility_check_l2401_240144


namespace parabola_equation_l2401_240192

-- Define a parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the condition that the parabola passes through a point
def passes_through (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

-- Define the condition that the vertex is at the origin
def vertex_at_origin (p : Parabola) : Prop :=
  p.equation 0 0

-- Define the condition that the axis of symmetry is a coordinate axis
def axis_is_coordinate (p : Parabola) : Prop :=
  (∀ x y : ℝ, p.equation x y ↔ p.equation x (-y)) ∨
  (∀ x y : ℝ, p.equation x y ↔ p.equation (-x) y)

-- Theorem statement
theorem parabola_equation 
  (p : Parabola) 
  (h1 : vertex_at_origin p) 
  (h2 : axis_is_coordinate p) 
  (h3 : passes_through p (-2) (-4)) :
  (∀ x y : ℝ, p.equation x y ↔ y^2 = -8*x) ∨
  (∀ x y : ℝ, p.equation x y ↔ x^2 = -y) :=
sorry

end parabola_equation_l2401_240192


namespace simplify_and_rationalize_l2401_240174

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) *
  (Real.sqrt 7 / Real.sqrt 8) * (Real.sqrt 9 / Real.sqrt 10) =
  3 * Real.sqrt 1050 / 320 := by
  sorry

end simplify_and_rationalize_l2401_240174


namespace range_of_a_l2401_240157

theorem range_of_a (x a : ℝ) : 
  (∀ x, (|x + 1| > 2 → x > a) ∧ 
  (|x + 1| ≤ 2 → x ≤ a) ∧ 
  (∃ x, x ≤ a ∧ |x + 1| > 2)) → 
  a ≥ 1 :=
by sorry

end range_of_a_l2401_240157


namespace initial_customers_l2401_240196

theorem initial_customers (stayed : ℕ) (left : ℕ) : stayed = 3 → left = stayed + 5 → stayed + left = 11 := by
  sorry

end initial_customers_l2401_240196


namespace sin_C_value_sin_law_variation_area_inequality_l2401_240183

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.S = (t.a + t.b)^2 - t.c^2 ∧ t.a + t.b = 4

-- Theorem statements
theorem sin_C_value (t : Triangle) (h : triangle_conditions t) : 
  Real.sin t.C = 8 / 17 := by sorry

theorem sin_law_variation (t : Triangle) : 
  (t.a^2 - t.b^2) / t.c^2 = Real.sin (t.A - t.B) / Real.sin t.C := by sorry

theorem area_inequality (t : Triangle) : 
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * Real.sqrt 3 * t.S := by sorry

end sin_C_value_sin_law_variation_area_inequality_l2401_240183


namespace intersection_of_A_and_B_l2401_240118

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end intersection_of_A_and_B_l2401_240118


namespace greater_number_problem_l2401_240173

theorem greater_number_problem (x y : ℝ) (h1 : x ≥ y) (h2 : x > 0) (h3 : y > 0) 
  (h4 : x * y = 2048) (h5 : x + y - (x - y) = 64) : x = 64 := by
  sorry

end greater_number_problem_l2401_240173


namespace abs_sum_inequality_l2401_240143

theorem abs_sum_inequality (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + 6| > a) → a < 5 := by sorry

end abs_sum_inequality_l2401_240143


namespace difference_is_198_l2401_240160

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  hundreds_lt_10 : hundreds < 10
  tens_lt_10 : tens < 10
  units_lt_10 : units < 10
  hundreds_gt_0 : hundreds > 0

/-- Condition: hundreds digit is 2 more than units digit -/
def hundreds_2_more_than_units (n : ThreeDigitNumber) : Prop :=
  n.hundreds = n.units + 2

/-- The value of the three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed number -/
def reversed (n : ThreeDigitNumber) : Nat :=
  100 * n.units + 10 * n.tens + n.hundreds

/-- The main theorem -/
theorem difference_is_198 (n : ThreeDigitNumber) 
  (h : hundreds_2_more_than_units n) : 
  value n - reversed n = 198 := by
  sorry


end difference_is_198_l2401_240160


namespace rectangular_room_shorter_side_l2401_240136

/-- Given a rectangular room with perimeter 50 feet and area 126 square feet,
    prove that the length of the shorter side is 9 feet. -/
theorem rectangular_room_shorter_side
  (perimeter : ℝ)
  (area : ℝ)
  (h_perimeter : perimeter = 50)
  (h_area : area = 126) :
  ∃ (length width : ℝ),
    length > 0 ∧
    width > 0 ∧
    length ≥ width ∧
    2 * (length + width) = perimeter ∧
    length * width = area ∧
    width = 9 := by
  sorry

end rectangular_room_shorter_side_l2401_240136


namespace basket_weight_proof_l2401_240188

def basket_problem (num_pears : ℕ) (pear_weight : ℝ) (total_weight : ℝ) : Prop :=
  let pears_weight := num_pears * pear_weight
  let basket_weight := total_weight - pears_weight
  basket_weight = 0.46

theorem basket_weight_proof :
  basket_problem 30 0.36 11.26 := by
  sorry

end basket_weight_proof_l2401_240188


namespace not_divisible_by_power_of_five_l2401_240134

theorem not_divisible_by_power_of_five (n : ℕ+) (k : ℕ+) 
  (h : k < 5^n.val - 5^(n.val - 1)) : 
  ¬ (5^n.val ∣ 2^k.val - 1) :=
sorry

end not_divisible_by_power_of_five_l2401_240134


namespace solution_set_l2401_240185

/-- A function representing the quadratic expression inside the absolute value -/
def f (a b x : ℝ) : ℝ := x^2 + 2*a*x + 3*a + b

/-- The condition for the inequality to have exactly one solution -/
def has_unique_solution (a b : ℝ) : Prop :=
  ∃! x, |f a b x| ≤ 4

/-- The theorem stating the solution set -/
theorem solution_set :
  ∀ a : ℝ, has_unique_solution a (a^2 - 3*a + 4) :=
sorry

end solution_set_l2401_240185


namespace operation_not_equal_33_l2401_240107

/-- Given single digit positive integers a and b, where x = 1/5 a and z = 1/5 b,
    prove that (10a + b) - (10x + z) ≠ 33 -/
theorem operation_not_equal_33 (a b : ℕ) (x z : ℕ) 
  (ha : 0 < a ∧ a < 10) (hb : 0 < b ∧ b < 10)
  (hx : x = a / 5) (hz : z = b / 5)
  (hx_pos : 0 < x) (hz_pos : 0 < z) : 
  (10 * a + b) - (10 * x + z) ≠ 33 := by
  sorry

end operation_not_equal_33_l2401_240107


namespace problem_part1_problem_part2_l2401_240190

-- Part 1
theorem problem_part1 (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = |a * x - 1|) 
  (h2 : Set.Icc (-3) 1 = {x | f x ≤ 2}) : a = -1 := by sorry

-- Part 2
theorem problem_part2 (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) 
  (m : ℝ) (h2 : ∃ x, f (2 * x + 1) - f (x - 1) ≤ 3 - 2 * m) : m ≤ 5/2 := by sorry

end problem_part1_problem_part2_l2401_240190


namespace ratio_to_percent_l2401_240100

theorem ratio_to_percent (a b : ℕ) (h : a = 15 ∧ b = 25) : 
  (a : ℝ) / b * 100 = 60 := by
  sorry

end ratio_to_percent_l2401_240100


namespace simplify_expression_l2401_240198

theorem simplify_expression (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 4*x + 4) = |x - 2| + |x + 2| :=
by sorry

end simplify_expression_l2401_240198


namespace digit_seven_place_value_l2401_240191

theorem digit_seven_place_value (n : ℕ) (p : ℕ) : 
  7 * 10^p - 7 = 693 → p = 2 := by sorry

end digit_seven_place_value_l2401_240191


namespace polynomial_equality_l2401_240145

theorem polynomial_equality : 11^5 - 5 * 11^4 + 10 * 11^3 - 10 * 11^2 + 5 * 11 - 1 = 100000 := by
  sorry

end polynomial_equality_l2401_240145
