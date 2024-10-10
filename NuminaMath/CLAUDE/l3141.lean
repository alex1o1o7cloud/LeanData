import Mathlib

namespace white_to_brown_dog_weight_ratio_l3141_314154

def brown_dog_weight : ℝ := 4
def black_dog_weight : ℝ := brown_dog_weight + 1
def grey_dog_weight : ℝ := black_dog_weight - 2
def average_weight : ℝ := 5
def num_dogs : ℕ := 4

def white_dog_weight : ℝ := average_weight * num_dogs - (brown_dog_weight + black_dog_weight + grey_dog_weight)

theorem white_to_brown_dog_weight_ratio :
  white_dog_weight / brown_dog_weight = 2 := by
  sorry

end white_to_brown_dog_weight_ratio_l3141_314154


namespace gcd_lcm_problem_l3141_314108

theorem gcd_lcm_problem (x y : ℕ+) (u v : ℕ) : 
  (u = Nat.gcd x y ∧ v = Nat.lcm x y) → 
  (x * y * u * v = 3600 ∧ u + v = 32) → 
  ((x = 6 ∧ y = 10) ∨ (x = 10 ∧ y = 6)) ∧ u = 2 ∧ v = 30 := by
  sorry

end gcd_lcm_problem_l3141_314108


namespace linda_spent_25_dollars_l3141_314113

/-- The amount Linda spent on her purchases -/
def linda_total_spent (coloring_book_price : ℚ) (coloring_book_quantity : ℕ)
  (peanut_pack_price : ℚ) (peanut_pack_quantity : ℕ) (stuffed_animal_price : ℚ) : ℚ :=
  coloring_book_price * coloring_book_quantity +
  peanut_pack_price * peanut_pack_quantity +
  stuffed_animal_price

theorem linda_spent_25_dollars :
  linda_total_spent 4 2 (3/2) 4 11 = 25 := by
  sorry

end linda_spent_25_dollars_l3141_314113


namespace bug_positions_l3141_314183

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Set of positions reachable by the bug in at most n steps -/
def reachablePositions (n : ℕ) : Set ℚ :=
  {x | ∃ (k : ℕ), k ≤ n ∧ ∃ (steps : List (ℚ → ℚ)),
    steps.length = k ∧
    (∀ step ∈ steps, step = (· + 2) ∨ step = (· / 2)) ∧
    x = (steps.foldl (λ acc f => f acc) 1)}

/-- The main theorem -/
theorem bug_positions (n : ℕ) :
  (reachablePositions n).ncard = fib (n + 4) - (n + 4) :=
sorry

end bug_positions_l3141_314183


namespace cubic_equation_solution_l3141_314146

theorem cubic_equation_solution :
  ∃ x : ℝ, 2 * x^3 + 24 * x = 3 - 12 * x^2 ∧ x = Real.rpow (19/2) (1/3) - 2 :=
by
  sorry

end cubic_equation_solution_l3141_314146


namespace cookie_brownie_difference_l3141_314161

def cookies_remaining (initial_cookies : ℕ) (daily_cookie_consumption : ℕ) (days : ℕ) : ℕ :=
  initial_cookies - daily_cookie_consumption * days

def brownies_remaining (initial_brownies : ℕ) (daily_brownie_consumption : ℕ) (days : ℕ) : ℕ :=
  initial_brownies - daily_brownie_consumption * days

theorem cookie_brownie_difference :
  let initial_cookies : ℕ := 60
  let initial_brownies : ℕ := 10
  let daily_cookie_consumption : ℕ := 3
  let daily_brownie_consumption : ℕ := 1
  let days : ℕ := 7
  cookies_remaining initial_cookies daily_cookie_consumption days -
  brownies_remaining initial_brownies daily_brownie_consumption days = 36 := by
  sorry

end cookie_brownie_difference_l3141_314161


namespace bowtie_equation_solution_l3141_314140

-- Define the operation ⊗
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem bowtie_equation_solution :
  ∃ y : ℝ, bowtie 5 y = 12 → y = 42 :=
by sorry

end bowtie_equation_solution_l3141_314140


namespace min_value_f_max_value_y_l3141_314185

/-- The minimum value of f(x) = 4/x + x for x > 0 is 4 -/
theorem min_value_f (x : ℝ) (hx : x > 0) :
  (4 / x + x) ≥ 4 ∧ ∃ x₀ > 0, 4 / x₀ + x₀ = 4 := by sorry

/-- The maximum value of y = x(1 - 3x) for 0 < x < 1/3 is 1/12 -/
theorem max_value_y (x : ℝ) (hx1 : x > 0) (hx2 : x < 1/3) :
  x * (1 - 3 * x) ≤ 1/12 ∧ ∃ x₀ ∈ (Set.Ioo 0 (1/3)), x₀ * (1 - 3 * x₀) = 1/12 := by sorry

end min_value_f_max_value_y_l3141_314185


namespace triangle_theorem_l3141_314153

theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  c - a = 1 →
  b = Real.sqrt 7 →
  B = π / 3 ∧
  (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 :=
by sorry

end triangle_theorem_l3141_314153


namespace salt_solution_problem_l3141_314197

theorem salt_solution_problem (initial_volume : ℝ) (added_water : ℝ) (final_salt_percentage : ℝ) :
  initial_volume = 80 →
  added_water = 20 →
  final_salt_percentage = 8 →
  let final_volume := initial_volume + added_water
  let initial_salt_amount := (initial_volume * final_salt_percentage) / final_volume
  let initial_salt_percentage := (initial_salt_amount / initial_volume) * 100
  initial_salt_percentage = 10 := by sorry

end salt_solution_problem_l3141_314197


namespace gcd_factorial_eight_and_factorial_six_squared_l3141_314186

theorem gcd_factorial_eight_and_factorial_six_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by sorry

end gcd_factorial_eight_and_factorial_six_squared_l3141_314186


namespace min_value_of_expression_l3141_314188

theorem min_value_of_expression (x : ℝ) (h : x > 3) :
  x + 4 / (x - 3) ≥ 7 ∧ (x + 4 / (x - 3) = 7 ↔ x = 5) := by
  sorry

end min_value_of_expression_l3141_314188


namespace projectile_max_height_l3141_314105

/-- The height function of the projectile --/
def h (t : ℝ) : ℝ := -16 * t^2 + 64 * t + 36

/-- The maximum height reached by the projectile --/
theorem projectile_max_height :
  ∃ (max : ℝ), max = 100 ∧ ∀ (t : ℝ), h t ≤ max :=
by sorry

end projectile_max_height_l3141_314105


namespace pedro_excess_squares_l3141_314191

-- Define the initial number of squares and multipliers for each player
def jesus_initial : ℕ := 60
def jesus_multiplier : ℕ := 2
def linden_initial : ℕ := 75
def linden_multiplier : ℕ := 3
def pedro_initial : ℕ := 200
def pedro_multiplier : ℕ := 4

-- Calculate the final number of squares for each player
def jesus_final : ℕ := jesus_initial * jesus_multiplier
def linden_final : ℕ := linden_initial * linden_multiplier
def pedro_final : ℕ := pedro_initial * pedro_multiplier

-- Define the theorem to be proved
theorem pedro_excess_squares : 
  pedro_final - (jesus_final + linden_final) = 455 := by
  sorry

end pedro_excess_squares_l3141_314191


namespace train_stop_time_l3141_314178

/-- Proves that a train with given speeds stops for 10 minutes per hour -/
theorem train_stop_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 48 → 
  speed_with_stops = 40 → 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 10 := by
  sorry

#check train_stop_time

end train_stop_time_l3141_314178


namespace complex_modulus_problem_l3141_314158

theorem complex_modulus_problem (z : ℂ) : z = (Complex.I - 2) / (1 + Complex.I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_modulus_problem_l3141_314158


namespace quadratic_coefficient_l3141_314181

/-- Given points (2, y₁) and (-2, y₂) on the graph of y = ax² + bx + d, where y₁ - y₂ = -8, prove that b = -2 -/
theorem quadratic_coefficient (a d y₁ y₂ : ℝ) : 
  y₁ = 4 * a + 2 * b + d →
  y₂ = 4 * a - 2 * b + d →
  y₁ - y₂ = -8 →
  b = -2 :=
by
  sorry

end quadratic_coefficient_l3141_314181


namespace small_kite_area_l3141_314123

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a kite given its vertices -/
def kiteArea (a b c d : Point) : ℝ :=
  let base := c.x - a.x
  let height := b.y - a.y
  base * height

/-- The grid spacing in inches -/
def gridSpacing : ℝ := 2

theorem small_kite_area :
  let a := Point.mk 0 6
  let b := Point.mk 3 10
  let c := Point.mk 6 6
  let d := Point.mk 3 0
  kiteArea a b c d * gridSpacing^2 = 72 := by
  sorry

end small_kite_area_l3141_314123


namespace solution_set_equivalence_l3141_314164

/-- An odd function -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- An even function -/
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

/-- F(x) is increasing on (-∞, 0) -/
def increasing_on_neg (F : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → F x₁ < F x₂

theorem solution_set_equivalence (f g : ℝ → ℝ) 
  (hf : odd_function f) (hg : even_function g)
  (hF : increasing_on_neg (λ x => f x * g x))
  (hg2 : g 2 = 0) :
  {x : ℝ | f x * g x < 0} = {x : ℝ | x < -2 ∨ (0 < x ∧ x < 2)} :=
by sorry

end solution_set_equivalence_l3141_314164


namespace smallest_x_solution_l3141_314130

theorem smallest_x_solution (w x y z : ℝ) 
  (non_neg : w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0)
  (eq1 : y = x - 2003)
  (eq2 : z = 2*y - 2003)
  (eq3 : w = 3*z - 2003) :
  x ≥ 10015/3 ∧ 
  (x = 10015/3 → y = 4006/3 ∧ z = 2003/3 ∧ w = 0) :=
by sorry

end smallest_x_solution_l3141_314130


namespace hyperbola_equation_l3141_314101

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the right focus
def right_focus (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the asymptote
def asymptote (a b : ℝ) (x y : ℝ) : Prop :=
  y = b / a * x ∨ y = -b / a * x

-- Define an equilateral triangle
def equilateral_triangle (A B C : ℝ × ℝ) (side_length : ℝ) : Prop :=
  dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

theorem hyperbola_equation (a b : ℝ) (F A : ℝ × ℝ) :
  a > 0 → b > 0 →
  hyperbola a b F.1 F.2 →
  F = right_focus a →
  asymptote a b A.1 A.2 →
  equilateral_triangle (0, 0) F A 2 →
  ∃ (x y : ℝ), x^2 - y^2 / 3 = 1 := by sorry


end hyperbola_equation_l3141_314101


namespace vector_perpendicular_condition_l3141_314129

theorem vector_perpendicular_condition (k : ℝ) : 
  let a : ℝ × ℝ := (-1, k)
  let b : ℝ × ℝ := (3, 1)
  (a.1 + b.1, a.2 + b.2) • a = 0 → k = -2 ∨ k = 1 := by
sorry

end vector_perpendicular_condition_l3141_314129


namespace absolute_value_equation_solution_l3141_314192

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |5 * x - 7| + 2 = 2 ∧ x = 7/5 := by
  sorry

end absolute_value_equation_solution_l3141_314192


namespace debate_team_boys_l3141_314127

theorem debate_team_boys (girls : ℕ) (groups : ℕ) (group_size : ℕ) (boys : ℕ) : 
  girls = 45 →
  groups = 8 →
  group_size = 7 →
  groups * group_size = girls + boys →
  boys = 11 := by
sorry

end debate_team_boys_l3141_314127


namespace veranda_area_l3141_314195

/-- Calculates the area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) : 
  room_length = 18 ∧ room_width = 12 ∧ veranda_width = 2 →
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 136 := by
  sorry

end veranda_area_l3141_314195


namespace line_through_points_l3141_314133

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- Given distinct vectors p and q, if m*p + (5/6)*q lies on the line through p and q, then m = 1/6 -/
theorem line_through_points (p q : V) (m : ℝ) 
  (h_distinct : p ≠ q)
  (h_on_line : ∃ t : ℝ, m • p + (5/6) • q = p + t • (q - p)) :
  m = 1/6 := by
  sorry

end line_through_points_l3141_314133


namespace A_equals_B_l3141_314156

def A : Set ℤ := {x | ∃ n : ℤ, x = 2*n - 1}
def B : Set ℤ := {x | ∃ n : ℤ, x = 2*n + 1}

theorem A_equals_B : A = B := by sorry

end A_equals_B_l3141_314156


namespace birds_can_gather_l3141_314198

/-- Represents a configuration of birds on trees -/
structure BirdConfiguration (n : ℕ) where
  positions : Fin n → Fin n

/-- The sum of bird positions in a configuration -/
def sum_positions (n : ℕ) (config : BirdConfiguration n) : ℕ :=
  (Finset.univ.sum fun i => (config.positions i).val) + n

/-- A bird movement that preserves the sum of positions -/
def valid_movement (n : ℕ) (config1 config2 : BirdConfiguration n) : Prop :=
  sum_positions n config1 = sum_positions n config2

/-- All birds are on the same tree -/
def all_gathered (n : ℕ) (config : BirdConfiguration n) : Prop :=
  ∃ k : Fin n, ∀ i : Fin n, config.positions i = k

/-- Initial configuration with one bird on each tree -/
def initial_config (n : ℕ) : BirdConfiguration n :=
  ⟨id⟩

/-- Theorem: Birds can gather on one tree iff n is odd and greater than 1 -/
theorem birds_can_gather (n : ℕ) :
  (∃ (config : BirdConfiguration n), valid_movement n (initial_config n) config ∧ all_gathered n config) ↔
  n % 2 = 1 ∧ n > 1 :=
sorry

end birds_can_gather_l3141_314198


namespace parabola_intersection_slope_l3141_314125

/-- Parabola C: y² = 4x -/
def C (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola C -/
def focus : ℝ × ℝ := (1, 0)

/-- Point M -/
def M : ℝ × ℝ := (0, 2)

/-- Line with slope k passing through the focus -/
def line (k x : ℝ) : ℝ := k*(x - focus.1)

/-- Intersection points of the line and the parabola -/
def intersectionPoints (k : ℝ) : Set (ℝ × ℝ) :=
  {p | C p.1 p.2 ∧ p.2 = line k p.1}

/-- Vector from M to a point P -/
def vector_MP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - M.1, P.2 - M.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem parabola_intersection_slope (k : ℝ) :
  (∃ A B, A ∈ intersectionPoints k ∧ B ∈ intersectionPoints k ∧ A ≠ B ∧
    dot_product (vector_MP A) (vector_MP B) = 0) →
  k = 8 := by sorry

end parabola_intersection_slope_l3141_314125


namespace perpendicular_line_equation_l3141_314138

/-- Given a line L1 with equation x - 2y - 2 = 0 and a point P(1, 0),
    the line L2 passing through P and perpendicular to L1 has the equation 2x + y - 2 = 0 -/
theorem perpendicular_line_equation (L1 : (ℝ × ℝ) → Prop) (P : ℝ × ℝ) :
  L1 = λ (x, y) => x - 2*y - 2 = 0 →
  P = (1, 0) →
  ∃ (L2 : (ℝ × ℝ) → Prop),
    (∀ (x y : ℝ), L2 (x, y) ↔ 2*x + y - 2 = 0) ∧
    L2 P ∧
    (∀ (v w : ℝ × ℝ), L1 v ∧ L1 w → L2 v ∧ L2 w →
      (v.1 - w.1) * (v.1 - w.1) + (v.2 - w.2) * (v.2 - w.2) = 0) :=
by sorry

end perpendicular_line_equation_l3141_314138


namespace child_ticket_cost_l3141_314166

/-- Proves that the cost of each child's ticket is $7 -/
theorem child_ticket_cost (num_adults num_children : ℕ) (concession_cost total_cost adult_ticket_cost : ℚ) :
  num_adults = 5 →
  num_children = 2 →
  concession_cost = 12 →
  total_cost = 76 →
  adult_ticket_cost = 10 →
  (total_cost - concession_cost - num_adults * adult_ticket_cost) / num_children = 7 :=
by sorry

end child_ticket_cost_l3141_314166


namespace correct_system_of_equations_l3141_314103

-- Define the number of people and price of goods
variable (x y : ℤ)

-- Define the conditions
def condition1 (x y : ℤ) : Prop := 8 * x - 3 = y
def condition2 (x y : ℤ) : Prop := 7 * x + 4 = y

-- Theorem statement
theorem correct_system_of_equations :
  (∀ x y : ℤ, condition1 x y ∧ condition2 x y →
    (8 * x - 3 = y ∧ 7 * x + 4 = y)) :=
by sorry

end correct_system_of_equations_l3141_314103


namespace gcd_8321_6489_l3141_314169

theorem gcd_8321_6489 : Nat.gcd 8321 6489 = 1 := by
  sorry

end gcd_8321_6489_l3141_314169


namespace second_train_speed_l3141_314120

/-- Calculates the speed of the second train given the parameters of two trains crossing each other. -/
theorem second_train_speed
  (length1 : ℝ)
  (speed1 : ℝ)
  (length2 : ℝ)
  (time_to_cross : ℝ)
  (h1 : length1 = 270)
  (h2 : speed1 = 120)
  (h3 : length2 = 230.04)
  (h4 : time_to_cross = 9)
  : ∃ (speed2 : ℝ), speed2 = 80 := by
  sorry

end second_train_speed_l3141_314120


namespace sum_of_two_numbers_l3141_314170

theorem sum_of_two_numbers (x y : ℕ) : y = x + 4 → y = 30 → x + y = 56 := by
  sorry

end sum_of_two_numbers_l3141_314170


namespace inequality_proof_l3141_314115

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a + b + 1)) + (1 / (b + c + 1)) + (1 / (a + c + 1)) ≤ 1 := by
  sorry

end inequality_proof_l3141_314115


namespace system_solution_l3141_314193

theorem system_solution :
  ∃! (X Y Z : ℝ),
    0.15 * 40 = 0.25 * X + 2 ∧
    0.30 * 60 = 0.20 * Y + 3 ∧
    0.10 * Z = X - Y ∧
    X = 16 ∧ Y = 75 ∧ Z = -590 := by
  sorry

end system_solution_l3141_314193


namespace hyperbola_eccentricity_l3141_314126

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  ∃ (m : ℝ), 
    (a^2 / c)^2 + m^2 = c^2 ∧ 
    2 * c * m = 4 * a * b → 
    (a^2 + b^2) / a^2 = 3 :=
by sorry

end hyperbola_eccentricity_l3141_314126


namespace smallest_bookmark_count_l3141_314117

theorem smallest_bookmark_count (b : ℕ) : 
  (b > 0) →
  (b % 5 = 4) →
  (b % 6 = 3) →
  (b % 8 = 7) →
  (∀ x : ℕ, x > 0 ∧ x % 5 = 4 ∧ x % 6 = 3 ∧ x % 8 = 7 → x ≥ b) →
  b = 39 := by
sorry

end smallest_bookmark_count_l3141_314117


namespace cubic_equation_solution_l3141_314111

theorem cubic_equation_solution (m : ℝ) (h : m^2 + m - 1 = 0) :
  m^3 + 2*m^2 + 2005 = 2006 := by
  sorry

end cubic_equation_solution_l3141_314111


namespace point_coordinates_l3141_314190

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The third quadrant of the 2D plane -/
def ThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The distance of a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point in the third quadrant with distance 3 to the x-axis
    and distance 5 to the y-axis has coordinates (-5, -3) -/
theorem point_coordinates (p : Point) 
  (h1 : ThirdQuadrant p) 
  (h2 : DistanceToXAxis p = 3) 
  (h3 : DistanceToYAxis p = 5) : 
  p = Point.mk (-5) (-3) := by
  sorry

end point_coordinates_l3141_314190


namespace hyperbola_equation_l3141_314132

theorem hyperbola_equation (P Q : ℝ × ℝ) : 
  P = (-3, 2 * Real.sqrt 7) → 
  Q = (-6 * Real.sqrt 2, 7) → 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), (y^2 / a^2) - (x^2 / b^2) = 1 ↔ 
      ((x, y) = P ∨ (x, y) = Q)) ∧
    a^2 = 25 ∧ b^2 = 75 :=
by sorry

end hyperbola_equation_l3141_314132


namespace adoption_time_proof_l3141_314196

/-- The number of days required to adopt all puppies -/
def adoption_days (initial_puppies : ℕ) (new_puppies : ℕ) (adopted_per_day : ℕ) : ℕ :=
  (initial_puppies + new_puppies) / adopted_per_day

/-- Theorem stating that it takes 9 days to adopt all puppies under given conditions -/
theorem adoption_time_proof :
  adoption_days 2 34 4 = 9 :=
by sorry

end adoption_time_proof_l3141_314196


namespace quadratic_real_roots_range_l3141_314175

theorem quadratic_real_roots_range (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 - 2 * k * x + (k - 3) = 0) →
  k ≥ 3/4 ∧ k ≠ 1 :=
by sorry

end quadratic_real_roots_range_l3141_314175


namespace accuracy_of_3_145e8_l3141_314141

/-- Represents the level of accuracy for a number -/
inductive Accuracy
  | HundredThousand
  | Million
  | TenMillion
  | HundredMillion

/-- Determines the accuracy of a number in scientific notation -/
def accuracy_of_scientific_notation (mantissa : Float) (exponent : Int) : Accuracy :=
  match exponent with
  | 8 => Accuracy.HundredThousand
  | 9 => Accuracy.Million
  | 10 => Accuracy.TenMillion
  | 11 => Accuracy.HundredMillion
  | _ => Accuracy.HundredThousand  -- Default case

theorem accuracy_of_3_145e8 :
  accuracy_of_scientific_notation 3.145 8 = Accuracy.HundredThousand :=
by sorry

end accuracy_of_3_145e8_l3141_314141


namespace second_month_sale_l3141_314112

def sale_month1 : ℕ := 7435
def sale_month3 : ℕ := 7855
def sale_month4 : ℕ := 8230
def sale_month5 : ℕ := 7560
def sale_month6 : ℕ := 6000
def average_sale : ℕ := 7500
def num_months : ℕ := 6

theorem second_month_sale :
  sale_month1 + sale_month3 + sale_month4 + sale_month5 + sale_month6 + 7920 = average_sale * num_months :=
by sorry

end second_month_sale_l3141_314112


namespace tree_height_proof_l3141_314109

/-- Given a tree that is currently 180 inches tall and 50% taller than its original height,
    prove that its original height was 10 feet. -/
theorem tree_height_proof :
  let current_height_inches : ℝ := 180
  let growth_factor : ℝ := 1.5
  let inches_per_foot : ℝ := 12
  current_height_inches / growth_factor / inches_per_foot = 10
  := by sorry

end tree_height_proof_l3141_314109


namespace zero_of_f_l3141_314151

def f (x : ℝ) := x + 1

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = -1 := by sorry

end zero_of_f_l3141_314151


namespace angle_on_line_l3141_314131

theorem angle_on_line (α : Real) : 
  (0 ≤ α) ∧ (α < π) ∧ 
  (∃ (x y : Real), x + 2 * y = 0 ∧ 
    x = Real.cos α ∧ y = Real.sin α) →
  Real.sin (π / 2 - 2 * α) = 3 / 5 := by
sorry

end angle_on_line_l3141_314131


namespace circular_garden_area_l3141_314135

/-- The area of a circular garden with radius 8 units, where the length of the fence
    (circumference) is 1/4 of the area of the garden. -/
theorem circular_garden_area : 
  let r : ℝ := 8
  let circumference := 2 * Real.pi * r
  let area := Real.pi * r^2
  circumference = (1/4) * area →
  area = 64 * Real.pi :=
by
  sorry

end circular_garden_area_l3141_314135


namespace complex_magnitude_product_l3141_314162

theorem complex_magnitude_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end complex_magnitude_product_l3141_314162


namespace cosine_equation_solution_l3141_314106

theorem cosine_equation_solution (A ω φ b : ℝ) (h_A : A > 0) :
  (∀ x, 2 * (Real.cos (x + Real.sin (2 * x)))^2 = A * Real.sin (ω * x + φ) + b) →
  A = Real.sqrt 2 := by
  sorry

end cosine_equation_solution_l3141_314106


namespace line_bisects_circle_coefficient_product_range_l3141_314134

/-- Given a line that always bisects the circumference of a circle, prove the range of the product of its coefficients. -/
theorem line_bisects_circle_coefficient_product_range
  (a b : ℝ)
  (h_bisect : ∀ (x y : ℝ), 4 * a * x - 3 * b * y + 48 = 0 →
    (x ^ 2 + y ^ 2 + 6 * x - 8 * y + 1 = 0 →
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        x₁ ^ 2 + y₁ ^ 2 + 6 * x₁ - 8 * y₁ + 1 = 0 ∧
        x₂ ^ 2 + y₂ ^ 2 + 6 * x₂ - 8 * y₂ + 1 = 0 ∧
        4 * a * x₁ - 3 * b * y₁ + 48 = 0 ∧
        4 * a * x₂ - 3 * b * y₂ + 48 = 0 ∧
        (x₁ - x₂) ^ 2 + (y₁ - y₂) ^ 2 = 4 * ((x - 3) ^ 2 + (y - 4) ^ 2))) :
  a * b ≤ 4 ∧ ∀ (k : ℝ), k < 4 → ∃ (a' b' : ℝ), a' * b' = k ∧
    ∀ (x y : ℝ), 4 * a' * x - 3 * b' * y + 48 = 0 →
      (x ^ 2 + y ^ 2 + 6 * x - 8 * y + 1 = 0 →
        ∃ (x₁ y₁ x₂ y₂ : ℝ),
          x₁ ^ 2 + y₁ ^ 2 + 6 * x₁ - 8 * y₁ + 1 = 0 ∧
          x₂ ^ 2 + y₂ ^ 2 + 6 * x₂ - 8 * y₂ + 1 = 0 ∧
          4 * a' * x₁ - 3 * b' * y₁ + 48 = 0 ∧
          4 * a' * x₂ - 3 * b' * y₂ + 48 = 0 ∧
          (x₁ - x₂) ^ 2 + (y₁ - y₂) ^ 2 = 4 * ((x - 3) ^ 2 + (y - 4) ^ 2)) :=
by sorry

end line_bisects_circle_coefficient_product_range_l3141_314134


namespace sufficient_not_necessary_l3141_314124

theorem sufficient_not_necessary (a b : ℝ → ℝ) : 
  (∀ x, |a x + b x| + |a x - b x| ≤ 1 → (a x)^2 + (b x)^2 ≤ 1) ∧ 
  (∃ x, (a x)^2 + (b x)^2 ≤ 1 ∧ |a x + b x| + |a x - b x| > 1) := by
  sorry

end sufficient_not_necessary_l3141_314124


namespace table_runner_coverage_l3141_314182

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) 
  (coverage_percentage : ℝ) (two_layer_area : ℝ) : 
  total_runner_area = 208 →
  table_area = 175 →
  coverage_percentage = 0.8 →
  two_layer_area = 24 →
  ∃ (three_layer_area : ℝ),
    three_layer_area = 22 ∧
    total_runner_area = (coverage_percentage * table_area - two_layer_area - three_layer_area) +
                        2 * two_layer_area +
                        3 * three_layer_area :=
by sorry

end table_runner_coverage_l3141_314182


namespace coordinate_sum_of_point_B_l3141_314137

/-- Given two points A and B in a 2D plane, where:
  - A is at (0, 0)
  - B is on the line y = 5
  - The slope of segment AB is 3/4
  Prove that the sum of the x- and y-coordinates of point B is 35/3 -/
theorem coordinate_sum_of_point_B (B : ℝ × ℝ) : 
  B.2 = 5 ∧ 
  (B.2 - 0) / (B.1 - 0) = 3 / 4 → 
  B.1 + B.2 = 35 / 3 := by
  sorry

end coordinate_sum_of_point_B_l3141_314137


namespace inequality_proof_l3141_314145

theorem inequality_proof (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (2*a - b)^2 / (a - b)^2 + (2*b - c)^2 / (b - c)^2 + (2*c - a)^2 / (c - a)^2 ≥ 5 := by
  sorry

end inequality_proof_l3141_314145


namespace second_account_interest_rate_l3141_314187

/-- Proves that the interest rate of the second account is 4% given the problem conditions --/
theorem second_account_interest_rate :
  ∀ (first_amount second_amount first_rate second_rate total_interest : ℝ),
    first_amount = 1000 →
    second_amount = first_amount + 800 →
    first_rate = 0.02 →
    total_interest = 92 →
    total_interest = first_rate * first_amount + second_rate * second_amount →
    second_rate = 0.04 := by
  sorry

end second_account_interest_rate_l3141_314187


namespace fifth_largest_divisor_l3141_314128

def n : ℕ := 2025000000

-- Define a function to get the kth largest divisor
def kth_largest_divisor (k : ℕ) (n : ℕ) : ℕ :=
  sorry

theorem fifth_largest_divisor :
  kth_largest_divisor 5 n = 126562500 :=
sorry

end fifth_largest_divisor_l3141_314128


namespace square_greater_than_abs_l3141_314199

theorem square_greater_than_abs (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end square_greater_than_abs_l3141_314199


namespace tinas_pens_l3141_314171

theorem tinas_pens (pink green blue : ℕ) : 
  pink = 12 ∧ 
  green = pink - 9 ∧ 
  blue = green + 3 → 
  pink + green + blue = 21 := by
  sorry

end tinas_pens_l3141_314171


namespace rectangle_area_diagonal_l3141_314180

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end rectangle_area_diagonal_l3141_314180


namespace convex_hull_perimeter_bounds_l3141_314118

/-- A regular polygon inscribed in a unit circle -/
structure RegularPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The convex hull formed by the vertices of two regular polygons inscribed in a unit circle -/
structure ConvexHull where
  p1 : RegularPolygon
  p2 : RegularPolygon

/-- The perimeter of the convex hull -/
noncomputable def perimeter (ch : ConvexHull) : ℝ :=
  sorry

theorem convex_hull_perimeter_bounds (ch : ConvexHull) 
  (h1 : ch.p1.n = 6) 
  (h2 : ch.p2.n = 7) : 
  6.1610929 ≤ perimeter ch ∧ perimeter ch ≤ 6.1647971 :=
sorry

end convex_hull_perimeter_bounds_l3141_314118


namespace intersection_A_complement_B_a_range_when_intersection_empty_l3141_314174

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem for part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 2) = {x | 1 < x ∧ x ≤ 2} := by sorry

-- Theorem for part 2
theorem a_range_when_intersection_empty :
  ∀ a : ℝ, A ∩ B a = ∅ → a ≥ 3 := by sorry

end intersection_A_complement_B_a_range_when_intersection_empty_l3141_314174


namespace units_digit_of_3_power_2004_l3141_314184

theorem units_digit_of_3_power_2004 : 3^2004 % 10 = 1 := by
  sorry

end units_digit_of_3_power_2004_l3141_314184


namespace price_difference_per_can_l3141_314152

/-- Proves that the difference in price per can between the local grocery store and the bulk warehouse is 25 cents -/
theorem price_difference_per_can (bulk_price : ℚ) (bulk_cans : ℕ) (grocery_price : ℚ) (grocery_cans : ℕ) 
  (h1 : bulk_price = 12) 
  (h2 : bulk_cans = 48) 
  (h3 : grocery_price = 6) 
  (h4 : grocery_cans = 12) : 
  (grocery_price / grocery_cans - bulk_price / bulk_cans) * 100 = 25 := by
  sorry

end price_difference_per_can_l3141_314152


namespace antonia_emails_l3141_314150

theorem antonia_emails :
  ∀ (total : ℕ),
  (1 : ℚ) / 4 * total = total - (3 : ℚ) / 4 * total →
  (2 : ℚ) / 5 * ((3 : ℚ) / 4 * total) = ((3 : ℚ) / 4 * total) - 180 →
  total = 400 :=
by
  sorry

end antonia_emails_l3141_314150


namespace morgans_mean_score_l3141_314160

def scores : List ℝ := [78, 82, 90, 95, 98, 102, 105]
def alex_count : ℕ := 4
def morgan_count : ℕ := 3
def alex_mean : ℝ := 91.5

theorem morgans_mean_score (h1 : scores.length = alex_count + morgan_count)
                            (h2 : alex_count * alex_mean = (scores.take alex_count).sum) :
  (scores.drop alex_count).sum / morgan_count = 94.67 := by
  sorry

end morgans_mean_score_l3141_314160


namespace pages_left_to_read_l3141_314110

theorem pages_left_to_read (total_pages : ℕ) (saturday_morning : ℕ) (saturday_night : ℕ) : 
  total_pages = 360 →
  saturday_morning = 40 →
  saturday_night = 10 →
  total_pages - (saturday_morning + saturday_night + 2 * (saturday_morning + saturday_night)) = 210 := by
  sorry

end pages_left_to_read_l3141_314110


namespace curve_transformation_l3141_314147

/-- Given a 2x2 matrix A and its inverse, prove that if A transforms a curve F to y = 2x, then F is y = -3x -/
theorem curve_transformation (a b : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; 7, 3]
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![b, -2; -7, a]
  A * A_inv = 1 →
  (∀ x y : ℝ, (A.mulVec ![x, y] = ![x', y'] ∧ y' = 2*x') → y = -3*x) :=
by sorry

end curve_transformation_l3141_314147


namespace ratio_PC_PB_is_zero_l3141_314189

/-- A square with side length 6, where N is the midpoint of AB and P is the intersection of BD and CN -/
structure SquareABCD where
  -- Define the square
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Conditions
  is_square : A = (0, 0) ∧ B = (6, 0) ∧ C = (6, 6) ∧ D = (0, 6)
  -- Define N as midpoint of AB
  N : ℝ × ℝ
  N_is_midpoint : N = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  -- Define P as intersection of BD and CN
  P : ℝ × ℝ
  P_on_BD : (P.2 - D.2) = ((B.2 - D.2) / (B.1 - D.1)) * (P.1 - D.1)
  P_on_CN : (P.2 - N.2) = ((C.2 - N.2) / (C.1 - N.1)) * (P.1 - N.1)

/-- The ratio of PC to PB is 0 -/
theorem ratio_PC_PB_is_zero (square : SquareABCD) : 
  let PC := Real.sqrt ((square.P.1 - square.C.1)^2 + (square.P.2 - square.C.2)^2)
  let PB := Real.sqrt ((square.P.1 - square.B.1)^2 + (square.P.2 - square.B.2)^2)
  PC / PB = 0 := by
  sorry

end ratio_PC_PB_is_zero_l3141_314189


namespace inverse_proportion_problem_l3141_314167

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) 
  (h1 : InverselyProportional x y)
  (h2 : x + y = 28)
  (h3 : x - y = 8) :
  (∃ z : ℝ, z = 7 → y = 180 / 7) := by
  sorry

end inverse_proportion_problem_l3141_314167


namespace power_product_equality_l3141_314142

theorem power_product_equality (x : ℝ) (h : x > 0) : 
  x^x * x^x = x^(2*x) ∧ x^x * x^x = (x^2)^x :=
by sorry

end power_product_equality_l3141_314142


namespace min_C_over_D_l3141_314163

theorem min_C_over_D (x C D : ℝ) (hx : x ≠ 0) 
  (hC : x^3 + 1/x^3 = C) (hD : x - 1/x = D) :
  ∀ y : ℝ, y ≠ 0 → y^3 + 1/y^3 / (y - 1/y) ≥ 3 :=
sorry

end min_C_over_D_l3141_314163


namespace peanut_cluster_probability_l3141_314121

def total_chocolates : ℕ := 50
def caramels : ℕ := 3
def nougats : ℕ := 2 * caramels
def truffles : ℕ := caramels + 6
def peanut_clusters : ℕ := total_chocolates - (caramels + nougats + truffles)

theorem peanut_cluster_probability : 
  (peanut_clusters : ℚ) / total_chocolates = 32 / 50 := by sorry

end peanut_cluster_probability_l3141_314121


namespace sock_pair_combinations_l3141_314104

theorem sock_pair_combinations (white brown blue : ℕ) 
  (h_white : white = 5) 
  (h_brown : brown = 5) 
  (h_blue : blue = 2) 
  (h_total : white + brown + blue = 12) : 
  (white.choose 2) + (brown.choose 2) + (blue.choose 2) = 21 := by
  sorry

end sock_pair_combinations_l3141_314104


namespace ninas_allowance_l3141_314139

theorem ninas_allowance (game_cost : ℝ) (tax_rate : ℝ) (savings_rate : ℝ) (weeks : ℕ) :
  game_cost = 50 →
  tax_rate = 0.1 →
  savings_rate = 0.5 →
  weeks = 11 →
  ∃ (allowance : ℝ),
    allowance * savings_rate * weeks = game_cost * (1 + tax_rate) ∧
    allowance = 10 := by
  sorry

end ninas_allowance_l3141_314139


namespace middle_digit_is_two_l3141_314148

theorem middle_digit_is_two (ABCDE : ℕ) : 
  ABCDE ≥ 10000 ∧ ABCDE < 100000 →
  4 * (10 * ABCDE + 4) = 400000 + ABCDE →
  (ABCDE / 100) % 10 = 2 := by
sorry

end middle_digit_is_two_l3141_314148


namespace sum_powers_l3141_314194

theorem sum_powers (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  ¬(∀ (a b c d : ℝ), (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) := by
sorry

end sum_powers_l3141_314194


namespace function_equality_implies_sum_greater_than_four_l3141_314168

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 2 / x

theorem function_equality_implies_sum_greater_than_four
  (x₁ x₂ : ℝ)
  (h₁ : x₁ > 0)
  (h₂ : x₂ > 0)
  (h₃ : x₁ ≠ x₂)
  (h₄ : f x₁ = f x₂) :
  x₁ + x₂ > 4 :=
by sorry

end function_equality_implies_sum_greater_than_four_l3141_314168


namespace tower_difference_l3141_314116

/-- The number of blocks Randy used to build different structures -/
structure BlockCounts where
  total : ℕ
  house : ℕ
  tower : ℕ
  bridge : ℕ

/-- The theorem stating the difference in blocks used for the tower versus the house and bridge combined -/
theorem tower_difference (b : BlockCounts) (h1 : b.total = 250) (h2 : b.house = 65) (h3 : b.tower = 120) (h4 : b.bridge = 45) :
  b.tower - (b.house + b.bridge) = 10 := by
  sorry


end tower_difference_l3141_314116


namespace divisible_by_2000_arrangement_l3141_314155

theorem divisible_by_2000_arrangement (nums : List ℕ) (h : nums.length = 23) :
  ∃ (arrangement : List ℕ → ℕ), arrangement nums % 2000 = 0 := by
  sorry

end divisible_by_2000_arrangement_l3141_314155


namespace parallel_sufficient_not_necessary_l3141_314107

-- Define the basic types
variable (Line : Type) (Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (at_least_parallel_to_one : Line → Plane → Prop)

-- Define the given conditions
variable (a : Line) (β : Plane)

-- State the theorem
theorem parallel_sufficient_not_necessary :
  (∀ (l : Line), parallel a l → in_plane l β → at_least_parallel_to_one a β) ∧
  (∃ (a : Line) (β : Plane), at_least_parallel_to_one a β ∧ ¬(∃ (l : Line), in_plane l β ∧ parallel a l)) :=
sorry

end parallel_sufficient_not_necessary_l3141_314107


namespace simple_interest_period_l3141_314100

theorem simple_interest_period (P : ℝ) : 
  (P * 4 * 5 / 100 = 1680) → 
  (P * 5 * 4 / 100 = 1680) → 
  ∃ T : ℝ, T = 5 ∧ P * 4 * T / 100 = 1680 := by
sorry

end simple_interest_period_l3141_314100


namespace binomial_coefficient_19_13_l3141_314149

theorem binomial_coefficient_19_13 :
  (Nat.choose 18 11 = 31824) →
  (Nat.choose 18 10 = 18564) →
  (Nat.choose 20 13 = 77520) →
  Nat.choose 19 13 = 27132 := by
sorry

end binomial_coefficient_19_13_l3141_314149


namespace new_person_age_l3141_314179

/-- Given a group of people with an initial average age and size, 
    calculate the age of a new person that changes the average to a new value. -/
theorem new_person_age (n : ℕ) (initial_avg new_avg : ℚ) : 
  n = 17 → 
  initial_avg = 14 → 
  new_avg = 15 → 
  (n : ℚ) * initial_avg + (new_avg * ((n : ℚ) + 1) - (n : ℚ) * initial_avg) = 32 := by
  sorry

#check new_person_age

end new_person_age_l3141_314179


namespace optimal_price_and_profit_l3141_314176

-- Define the purchase price
def purchase_price : ℝ := 50

-- Define the linear relationship between price and sales volume
def sales_volume (x : ℝ) : ℝ := -20 * x + 2600

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - purchase_price) * sales_volume x

-- Define the constraint that selling price is not lower than purchase price
def price_constraint (x : ℝ) : Prop := x ≥ purchase_price

-- Define the constraint that profit per shirt should not exceed 30% of purchase price
def profit_constraint (x : ℝ) : Prop := x - purchase_price ≤ 0.3 * purchase_price

-- Theorem statement
theorem optimal_price_and_profit :
  ∃ (x : ℝ), 
    price_constraint x ∧ 
    profit_constraint x ∧ 
    (∀ y : ℝ, price_constraint y → profit_constraint y → profit x ≥ profit y) ∧
    x = 65 ∧
    profit x = 19500 := by
  sorry

end optimal_price_and_profit_l3141_314176


namespace students_not_coming_to_class_l3141_314165

theorem students_not_coming_to_class (pieces_per_student : ℕ) 
  (total_pieces_last_monday : ℕ) (total_pieces_this_monday : ℕ) :
  pieces_per_student = 4 →
  total_pieces_last_monday = 40 →
  total_pieces_this_monday = 28 →
  total_pieces_last_monday / pieces_per_student - 
  total_pieces_this_monday / pieces_per_student = 3 :=
by
  sorry

end students_not_coming_to_class_l3141_314165


namespace initial_strawberry_weight_l3141_314114

/-- The weight of strawberries initially collected by Marco and his dad -/
def initial_weight : ℕ := sorry

/-- The additional weight of strawberries found -/
def additional_weight : ℕ := 30

/-- Marco's strawberry weight after finding more -/
def marco_weight : ℕ := 36

/-- Marco's dad's strawberry weight after finding more -/
def dad_weight : ℕ := 16

/-- Theorem stating that the initial weight of strawberries is 22 pounds -/
theorem initial_strawberry_weight : initial_weight = 22 := by
  sorry

end initial_strawberry_weight_l3141_314114


namespace water_bottles_profit_l3141_314172

def water_bottles_problem (total_bottles : ℕ) (standard_rate_bottles : ℕ) (standard_rate_price : ℚ)
  (discount_threshold : ℕ) (discount_rate : ℚ) (selling_rate_bottles : ℕ) (selling_rate_price : ℚ) : Prop :=
  let standard_price_per_bottle : ℚ := standard_rate_price / standard_rate_bottles
  let total_cost_without_discount : ℚ := total_bottles * standard_price_per_bottle
  let total_cost_with_discount : ℚ := total_cost_without_discount * (1 - discount_rate)
  let selling_price_per_bottle : ℚ := selling_rate_price / selling_rate_bottles
  let total_revenue : ℚ := total_bottles * selling_price_per_bottle
  let profit : ℚ := total_revenue - total_cost_with_discount
  (total_bottles > discount_threshold) ∧ (profit = 325)

theorem water_bottles_profit :
  water_bottles_problem 1500 6 3 1200 (1/10) 3 2 :=
sorry

end water_bottles_profit_l3141_314172


namespace grandmother_age_l3141_314136

theorem grandmother_age (M : ℕ) (x : ℕ) :
  (2 * M : ℝ) = M →  -- Number of grandfathers is twice that of grandmothers
  (77 : ℝ) < (M * x + 2 * M * (x - 5)) / (3 * M) →  -- Average age of all pensioners > 77
  (M * x + 2 * M * (x - 5)) / (3 * M) < 78 →  -- Average age of all pensioners < 78
  x = 81 :=
by sorry

end grandmother_age_l3141_314136


namespace relationship_abc_l3141_314144

theorem relationship_abc (a b c : ℝ) :
  (∃ u v : ℝ, u - v = a ∧ u^2 - v^2 = b ∧ u^3 - v^3 = c) →
  3 * b^2 + a^4 = 4 * a * c := by
  sorry

end relationship_abc_l3141_314144


namespace square_twelve_y_minus_five_l3141_314177

theorem square_twelve_y_minus_five (y : ℝ) (h : 6 * y^2 + 7 = 4 * y + 13) : 
  (12 * y - 5)^2 = 161 := by
  sorry

end square_twelve_y_minus_five_l3141_314177


namespace transformed_variance_l3141_314157

variable {n : ℕ}
variable (x : Fin n → ℝ)

def variance (x : Fin n → ℝ) : ℝ := sorry

theorem transformed_variance
  (h : variance x = 3) :
  variance (fun i => 2 * x i + 4) = 12 := by sorry

end transformed_variance_l3141_314157


namespace distribute_5_3_l3141_314173

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    with each box containing at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects into 3 distinct boxes,
    with each box containing at least one object, is 150. -/
theorem distribute_5_3 : distribute 5 3 = 150 := by sorry

end distribute_5_3_l3141_314173


namespace equation_solution_l3141_314102

theorem equation_solution : ∃ x : ℝ, x * 400 = 173 * 2400 + 125 * 480 / 60 ∧ x = 1039.3 := by
  sorry

end equation_solution_l3141_314102


namespace interval_equivalence_l3141_314159

theorem interval_equivalence (a : ℝ) : -1 < a ∧ a < 1 ↔ |a| < 1 := by
  sorry

end interval_equivalence_l3141_314159


namespace apartment_count_l3141_314119

theorem apartment_count
  (num_entrances : ℕ)
  (initial_number : ℕ)
  (new_number : ℕ)
  (h1 : num_entrances = 5)
  (h2 : initial_number = 636)
  (h3 : new_number = 242)
  (h4 : initial_number > new_number) :
  (initial_number - new_number) / 2 * num_entrances = 985 :=
by sorry

end apartment_count_l3141_314119


namespace abs_inequality_solution_set_l3141_314143

theorem abs_inequality_solution_set (x : ℝ) :
  |x + 1| - |x - 2| > 1 ↔ x > 1 := by sorry

end abs_inequality_solution_set_l3141_314143


namespace probability_of_valid_selection_l3141_314122

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def valid_selection (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 12 ∧
  1 ≤ b ∧ b ≤ 12 ∧
  1 ≤ c ∧ c ≤ 12 ∧
  1 ≤ d ∧ d ≤ 12 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  is_multiple a b ∧ is_multiple b c ∧ is_multiple c d

def count_valid_selections : ℕ := sorry

theorem probability_of_valid_selection :
  (count_valid_selections : ℚ) / (12 * 11 * 10 * 9) = 13 / 845 := by sorry

end probability_of_valid_selection_l3141_314122
