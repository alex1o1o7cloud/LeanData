import Mathlib

namespace ali_baba_walk_possible_l1601_160172

/-- Represents a cell in the cave -/
structure Cell where
  row : Nat
  col : Nat
  isBlack : Bool

/-- Represents the state of the cave -/
structure CaveState where
  m : Nat
  n : Nat
  coins : Cell → Nat

/-- Represents a move in the cave -/
inductive Move
  | up
  | down
  | left
  | right

/-- Predicate to check if a move is valid -/
def isValidMove (state : CaveState) (pos : Cell) (move : Move) : Prop :=
  match move with
  | Move.up => pos.row > 0
  | Move.down => pos.row < state.m - 1
  | Move.left => pos.col > 0
  | Move.right => pos.col < state.n - 1

/-- Function to apply a move and update the cave state -/
def applyMove (state : CaveState) (pos : Cell) (move : Move) : CaveState :=
  sorry

/-- Predicate to check if the final state is correct -/
def isCorrectFinalState (state : CaveState) : Prop :=
  ∀ cell, (cell.isBlack → state.coins cell = 1) ∧ (¬cell.isBlack → state.coins cell = 0)

/-- Theorem stating that Ali Baba's walk is possible -/
theorem ali_baba_walk_possible (m n : Nat) :
  ∃ (initialState : CaveState) (moves : List Move),
    initialState.m = m ∧
    initialState.n = n ∧
    (∀ cell, initialState.coins cell = 0) ∧
    isCorrectFinalState (moves.foldl (λ s m => applyMove s (sorry) m) initialState) :=
  sorry

end ali_baba_walk_possible_l1601_160172


namespace factorial_difference_l1601_160115

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end factorial_difference_l1601_160115


namespace black_number_equals_sum_of_white_numbers_l1601_160118

theorem black_number_equals_sum_of_white_numbers :
  ∃ (a b c d : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  (Real.sqrt (c + d * Real.sqrt 7) = Real.sqrt (a + b * Real.sqrt 2) + Real.sqrt (a - b * Real.sqrt 2)) := by
  sorry

end black_number_equals_sum_of_white_numbers_l1601_160118


namespace trigonometric_equation_l1601_160195

theorem trigonometric_equation (α β : Real) 
  (h : (Real.cos α)^3 / Real.cos β + (Real.sin α)^3 / Real.sin β = 2) :
  (Real.sin β)^3 / Real.sin α + (Real.cos β)^3 / Real.cos α = 1/2 := by
  sorry

end trigonometric_equation_l1601_160195


namespace fraction_equality_l1601_160124

theorem fraction_equality : (1012^2 - 1003^2) / (1019^2 - 996^2) = 9 / 23 := by
  sorry

end fraction_equality_l1601_160124


namespace inequality_solution_length_l1601_160144

theorem inequality_solution_length (a b : ℝ) : 
  (∀ x, (a + 1 ≤ 3 * x + 6 ∧ 3 * x + 6 ≤ b - 2) ↔ ((a - 5) / 3 ≤ x ∧ x ≤ (b - 8) / 3)) →
  ((b - 8) / 3 - (a - 5) / 3 = 18) →
  b - a = 57 := by
sorry

end inequality_solution_length_l1601_160144


namespace negation_of_existence_negation_of_square_lt_one_l1601_160139

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem negation_of_square_lt_one :
  (¬ ∃ x : ℝ, x^2 < 1) ↔ (∀ x : ℝ, x^2 ≥ 1) := by sorry

end negation_of_existence_negation_of_square_lt_one_l1601_160139


namespace emma_walk_distance_l1601_160197

theorem emma_walk_distance
  (total_time : ℝ)
  (bike_speed : ℝ)
  (walk_speed : ℝ)
  (bike_fraction : ℝ)
  (walk_fraction : ℝ)
  (h_total_time : total_time = 1)
  (h_bike_speed : bike_speed = 20)
  (h_walk_speed : walk_speed = 6)
  (h_bike_fraction : bike_fraction = 1/3)
  (h_walk_fraction : walk_fraction = 2/3)
  (h_fractions : bike_fraction + walk_fraction = 1) :
  let total_distance := (bike_speed * bike_fraction + walk_speed * walk_fraction) * total_time
  let walk_distance := total_distance * walk_fraction
  ∃ (ε : ℝ), abs (walk_distance - 5.2) < ε ∧ ε < 0.1 :=
by sorry

end emma_walk_distance_l1601_160197


namespace sandy_nickels_borrowed_l1601_160104

/-- Given the initial number of nickels and the remaining number of nickels,
    calculate the number of nickels borrowed. -/
def nickels_borrowed (initial : Nat) (remaining : Nat) : Nat :=
  initial - remaining

theorem sandy_nickels_borrowed :
  let initial_nickels : Nat := 31
  let remaining_nickels : Nat := 11
  nickels_borrowed initial_nickels remaining_nickels = 20 := by
  sorry

end sandy_nickels_borrowed_l1601_160104


namespace right_triangle_equality_l1601_160130

/-- For a right triangle with sides a and b, and hypotenuse c, 
    the equation √(a^2 + b^2) = a + b is true if and only if 
    the angle θ between sides a and b is 90°. -/
theorem right_triangle_equality (a b c : ℝ) (θ : Real) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : c^2 = a^2 + b^2) -- Pythagorean theorem
  (h5 : θ = Real.arccos (b / c)) -- Definition of θ
  : Real.sqrt (a^2 + b^2) = a + b ↔ θ = π / 2 := by
  sorry

end right_triangle_equality_l1601_160130


namespace increasing_quadratic_function_parameter_range_l1601_160183

theorem increasing_quadratic_function_parameter_range 
  (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = x^2 - 2*a*x + 2) 
  (h2 : ∀ x y, x ≥ 3 → y ≥ 3 → x < y → f x < f y) : 
  a ∈ Set.Iic 3 := by
sorry

end increasing_quadratic_function_parameter_range_l1601_160183


namespace inequality_proof_l1601_160190

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  |b / a - b / c| + |c / a - c / b| + |b * c + 1| > 1 := by
  sorry

end inequality_proof_l1601_160190


namespace jim_shopping_cost_l1601_160157

theorem jim_shopping_cost (lamp_cost : ℕ) (bulb_cost : ℕ) (num_lamps : ℕ) (num_bulbs : ℕ) 
  (h1 : lamp_cost = 7)
  (h2 : bulb_cost = lamp_cost - 4)
  (h3 : num_lamps = 2)
  (h4 : num_bulbs = 6) :
  num_lamps * lamp_cost + num_bulbs * bulb_cost = 32 := by
sorry

end jim_shopping_cost_l1601_160157


namespace smallest_n_for_geometric_sum_l1601_160148

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The statement to prove -/
theorem smallest_n_for_geometric_sum : 
  ∀ n : ℕ, n > 0 → 
    (geometric_sum (1/3) (1/3) n = 80/243 ↔ n ≥ 5) ∧ 
    (geometric_sum (1/3) (1/3) 5 = 80/243) :=
sorry

end smallest_n_for_geometric_sum_l1601_160148


namespace quadratic_roots_reciprocal_sum_l1601_160152

theorem quadratic_roots_reciprocal_sum (x₁ x₂ : ℝ) :
  x₁^2 - 4*x₁ - 2 = 0 →
  x₂^2 - 4*x₂ - 2 = 0 →
  x₁ ≠ x₂ →
  (1/x₁) + (1/x₂) = -2 := by
sorry

end quadratic_roots_reciprocal_sum_l1601_160152


namespace polar_to_rectangular_equivalence_l1601_160167

/-- Proves that the given polar equation is equivalent to the given rectangular equation. -/
theorem polar_to_rectangular_equivalence :
  ∀ (r φ x y : ℝ),
  (r = 2 / (4 - Real.sin φ)) ↔ 
  (x^2 / (2/Real.sqrt 15)^2 + (y - 2/15)^2 / (8/15)^2 = 1 ∧
   x = r * Real.cos φ ∧
   y = r * Real.sin φ) :=
by sorry

end polar_to_rectangular_equivalence_l1601_160167


namespace correct_ab_sample_size_l1601_160170

/-- Represents the number of students to be drawn with blood type AB in a stratified sampling -/
def stratified_sample_ab (total_students : ℕ) (ab_students : ℕ) (sample_size : ℕ) : ℕ :=
  (ab_students * sample_size) / total_students

/-- Theorem stating the correct number of AB blood type students in the sample -/
theorem correct_ab_sample_size :
  stratified_sample_ab 500 50 60 = 6 := by sorry

end correct_ab_sample_size_l1601_160170


namespace rowing_speed_in_still_water_l1601_160155

/-- The speed of a man rowing a boat in still water, given his downstream performance and current speed -/
theorem rowing_speed_in_still_water 
  (distance : Real) 
  (time : Real) 
  (current_speed : Real) : 
  (distance / 1000) / (time / 3600) - current_speed = 22 :=
by
  -- Assuming:
  -- distance = 80 (meters)
  -- time = 11.519078473722104 (seconds)
  -- current_speed = 3 (km/h)
  sorry

#check rowing_speed_in_still_water


end rowing_speed_in_still_water_l1601_160155


namespace function_decreasing_interval_l1601_160122

/-- The function f(x) = xe^x + 1 is decreasing on the interval (-∞, -1) -/
theorem function_decreasing_interval (x : ℝ) : 
  x < -1 → (fun x => x * Real.exp x + 1) '' Set.Ioi x ⊆ Set.Iio ((x * Real.exp x + 1) : ℝ) := by
  sorry

end function_decreasing_interval_l1601_160122


namespace mean_temperature_l1601_160125

def temperatures : List ℝ := [78, 76, 80, 83, 85]

theorem mean_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 80.4 := by
  sorry

end mean_temperature_l1601_160125


namespace cubic_factorization_l1601_160108

theorem cubic_factorization (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by sorry

end cubic_factorization_l1601_160108


namespace drinks_left_calculation_l1601_160154

-- Define the initial amounts of drinks
def initial_coke : ℝ := 35.5
def initial_cider : ℝ := 27.2

-- Define the amount of coke drunk
def coke_drunk : ℝ := 1.75

-- Theorem statement
theorem drinks_left_calculation :
  initial_coke + initial_cider - coke_drunk = 60.95 := by
  sorry

end drinks_left_calculation_l1601_160154


namespace escalator_steps_l1601_160132

/-- The number of steps counted by the slower person -/
def walker_count : ℕ := 50

/-- The number of steps counted by the faster person -/
def trotman_count : ℕ := 75

/-- The speed ratio between the faster and slower person -/
def speed_ratio : ℕ := 3

/-- The number of visible steps on the stopped escalator -/
def visible_steps : ℕ := 100

/-- Theorem stating that the number of visible steps on the stopped escalator is 100 -/
theorem escalator_steps :
  ∀ (v : ℚ), v > 0 →
  walker_count + walker_count / v = trotman_count + trotman_count / (speed_ratio * v) →
  visible_steps = walker_count + walker_count / v :=
by sorry

end escalator_steps_l1601_160132


namespace simplified_fraction_ratio_l1601_160129

theorem simplified_fraction_ratio (k : ℝ) : 
  let original := (6 * k + 12) / 6
  let simplified := k + 2
  ∃ (a b : ℤ), (simplified = a * k + b) ∧ (a / b = 1 / 2) :=
by sorry

end simplified_fraction_ratio_l1601_160129


namespace students_at_1544_l1601_160176

/-- Calculates the number of students in the computer lab at a given time -/
def studentsInLab (initialTime startTime endTime : Nat) (initialStudents : Nat) 
  (enterInterval enterCount : Nat) (leaveInterval leaveCount : Nat) : Nat :=
  let totalMinutes := endTime - initialTime
  let enterTimes := (totalMinutes - (startTime - initialTime)) / enterInterval
  let leaveTimes := (totalMinutes - (startTime - initialTime)) / leaveInterval
  initialStudents + enterTimes * enterCount - leaveTimes * leaveCount

theorem students_at_1544 :
  studentsInLab 1500 1503 1544 20 3 4 10 8 = 44 := by
  sorry

end students_at_1544_l1601_160176


namespace sum_of_three_cubes_not_2002_l1601_160177

theorem sum_of_three_cubes_not_2002 : ¬∃ (a b c : ℕ+), a.val^3 + b.val^3 + c.val^3 = 2002 := by
  sorry

end sum_of_three_cubes_not_2002_l1601_160177


namespace range_of_a_l1601_160179

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| ≤ 1 → x^2 - 5*x + 4 ≤ 0) →
  a ∈ Set.Icc 2 3 :=
by sorry

end range_of_a_l1601_160179


namespace yard_area_l1601_160128

/-- The area of a rectangular yard with two cut-out areas -/
theorem yard_area (yard_length yard_width square_side rectangle_length rectangle_width : ℕ) 
  (h1 : yard_length = 20)
  (h2 : yard_width = 18)
  (h3 : square_side = 3)
  (h4 : rectangle_length = 4)
  (h5 : rectangle_width = 2) :
  yard_length * yard_width - (square_side * square_side + rectangle_length * rectangle_width) = 343 := by
  sorry

#check yard_area

end yard_area_l1601_160128


namespace perpendicular_slope_l1601_160138

/-- Given a line with equation 4x - 5y = 20, the slope of the perpendicular line is -5/4 -/
theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 5 * y = 20) → (slope_of_perpendicular_line = -5/4) :=
by sorry

end perpendicular_slope_l1601_160138


namespace geometry_relations_l1601_160163

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (l : Line) (m : Line) (α β : Plane)
  (h1 : subset l α)
  (h2 : subset m β) :
  (perpendicular l β → plane_perpendicular α β) ∧
  (parallel α β → line_parallel l β) :=
sorry

end geometry_relations_l1601_160163


namespace midpoint_distance_theorem_l1601_160192

theorem midpoint_distance_theorem (t : ℝ) : 
  let A : ℝ × ℝ := (t - 3, 0)
  let B : ℝ × ℝ := (-1, t + 2)
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ((midpoint.1 - A.1)^2 + (midpoint.2 - A.2)^2 = t^2 + 1) →
  (t = Real.sqrt 2 ∨ t = -Real.sqrt 2) := by
sorry

end midpoint_distance_theorem_l1601_160192


namespace range_of_x_l1601_160111

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem range_of_x (a : ℝ) (h1 : a > 1) (h2 : ∀ x, f x a ≥ 3) (h3 : ∃ x, f x a = 3) :
  ∀ x, f x a ≤ 5 ↔ 3 ≤ x ∧ x ≤ 8 := by
  sorry

end range_of_x_l1601_160111


namespace fib_last_digit_periodic_l1601_160133

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Period of Fibonacci sequence modulo 10 -/
def fibPeriod : ℕ := 60

/-- Theorem: The last digit of Fibonacci numbers repeats with period 60 -/
theorem fib_last_digit_periodic (n : ℕ) : fib n % 10 = fib (n + fibPeriod) % 10 := by
  sorry

end fib_last_digit_periodic_l1601_160133


namespace tangent_circles_ratio_l1601_160171

/-- Two circles are tangent if the distance between their centers is equal to the sum of their radii -/
def are_tangent (center1 center2 : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (2 * radius)^2

/-- Definition of circle C₁ -/
def circle_C1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}

/-- Definition of circle C₂ -/
def circle_C2 (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - b)^2 + (p.2 - c)^2 = a^2}

theorem tangent_circles_ratio (a b c : ℝ) (ha : a > 0) :
  are_tangent (0, 0) (b, c) a →
  (b^2 + c^2) / a^2 = 4 := by
  sorry

end tangent_circles_ratio_l1601_160171


namespace inscribed_sphere_radius_tetrahedron_l1601_160193

/-- Given a tetrahedron with volume V, face areas S₁, S₂, S₃, S₄, and an inscribed sphere of radius R,
    prove that R = 3V / (S₁ + S₂ + S₃ + S₄) -/
theorem inscribed_sphere_radius_tetrahedron (V : ℝ) (S₁ S₂ S₃ S₄ : ℝ) (R : ℝ) 
    (h_volume : V > 0)
    (h_areas : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0)
    (h_inscribed : R > 0) :
  R = 3 * V / (S₁ + S₂ + S₃ + S₄) :=
sorry

end inscribed_sphere_radius_tetrahedron_l1601_160193


namespace gcd_840_1764_gcd_561_255_l1601_160147

-- Part 1: GCD of 840 and 1764
theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by sorry

-- Part 2: GCD of 561 and 255
theorem gcd_561_255 : Nat.gcd 561 255 = 51 := by sorry

end gcd_840_1764_gcd_561_255_l1601_160147


namespace train_speed_calculation_l1601_160198

/-- Proves that a train with given length, crossing a bridge of given length in a given time, has a specific speed in km/h -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 200 →
  crossing_time = 31.99744020478362 →
  (((train_length + bridge_length) / crossing_time) * 3.6) = 36 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l1601_160198


namespace no_squares_in_range_l1601_160136

theorem no_squares_in_range : ¬ ∃ (x y a b : ℕ),
  988 ≤ x ∧ x < y ∧ y ≤ 1991 ∧
  x * y + x = a^2 ∧
  x * y + y = b^2 :=
sorry

end no_squares_in_range_l1601_160136


namespace det_E_l1601_160159

/-- A 2×2 matrix representing a dilation centered at the origin with scale factor 9 -/
def E : Matrix (Fin 2) (Fin 2) ℝ := !![9, 0; 0, 9]

/-- The determinant of E is 81 -/
theorem det_E : Matrix.det E = 81 := by sorry

end det_E_l1601_160159


namespace inequality_theorem_l1601_160185

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hxy₁ : x₁ * y₁ - z₁^2 > 0) (hxy₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ∧
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔
   x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂) :=
by sorry

end inequality_theorem_l1601_160185


namespace chocolate_distribution_l1601_160174

theorem chocolate_distribution (num_students : ℕ) (num_choices : ℕ) 
  (h1 : num_students = 211) (h2 : num_choices = 35) : 
  ∃ (group_size : ℕ), group_size ≥ 7 ∧ 
  (∀ (group : ℕ), group ≤ group_size) ∧ 
  (num_students ≤ group_size * num_choices) :=
sorry

end chocolate_distribution_l1601_160174


namespace roots_reciprocal_sum_l1601_160121

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  (5 * x₁^2 - 3 * x₁ - 2 = 0) → 
  (5 * x₂^2 - 3 * x₂ - 2 = 0) → 
  (x₁ ≠ x₂) →
  (1 / x₁ + 1 / x₂ = -3/2) := by
  sorry

end roots_reciprocal_sum_l1601_160121


namespace cos_300_degrees_l1601_160119

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end cos_300_degrees_l1601_160119


namespace select_players_correct_l1601_160188

/-- The number of ways to select k players from m teams, each with n players,
    such that no two selected players are from the same team -/
def select_players (m n k : ℕ) : ℕ :=
  Nat.choose m k * n^k

/-- Theorem stating that select_players gives the correct number of ways
    to form the committee under the given conditions -/
theorem select_players_correct (m n k : ℕ) (h : k ≤ m) :
  select_players m n k = Nat.choose m k * n^k :=
by sorry

end select_players_correct_l1601_160188


namespace black_pens_count_l1601_160116

theorem black_pens_count (total_pens : ℕ) (red_pens black_pens : ℕ) : 
  (3 : ℚ) / 10 * total_pens = red_pens →
  (1 : ℚ) / 5 * total_pens = black_pens →
  red_pens = 12 →
  black_pens = 8 := by
  sorry

end black_pens_count_l1601_160116


namespace seashell_count_l1601_160103

theorem seashell_count (sally_shells tom_shells jessica_shells : ℕ) 
  (h1 : sally_shells = 9)
  (h2 : tom_shells = 7)
  (h3 : jessica_shells = 5) :
  sally_shells + tom_shells + jessica_shells = 21 := by
  sorry

end seashell_count_l1601_160103


namespace abs_ratio_eq_sqrt_seven_thirds_l1601_160184

theorem abs_ratio_eq_sqrt_seven_thirds 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h : a^2 + b^2 = 5*a*b) : 
  |((a+b)/(a-b))| = Real.sqrt (7/3) := by
sorry

end abs_ratio_eq_sqrt_seven_thirds_l1601_160184


namespace costume_ball_same_gender_dance_l1601_160164

/-- Represents a person at the costume ball -/
structure Person :=
  (partners : Nat)

/-- Represents the costume ball -/
structure CostumeBall :=
  (people : Finset Person)
  (total_people : Nat)
  (total_dances : Nat)

/-- The costume ball satisfies the given conditions -/
def valid_costume_ball (ball : CostumeBall) : Prop :=
  ball.total_people = 20 ∧
  (ball.people.filter (λ p => p.partners = 3)).card = 11 ∧
  (ball.people.filter (λ p => p.partners = 5)).card = 1 ∧
  (ball.people.filter (λ p => p.partners = 6)).card = 8 ∧
  ball.total_dances = (11 * 3 + 1 * 5 + 8 * 6) / 2

theorem costume_ball_same_gender_dance (ball : CostumeBall) 
  (h : valid_costume_ball ball) : 
  ¬ (∀ (dance : Nat), dance < ball.total_dances → 
    ∃ (p1 p2 : Person), p1 ∈ ball.people ∧ p2 ∈ ball.people ∧ p1 ≠ p2) :=
by sorry

end costume_ball_same_gender_dance_l1601_160164


namespace prize_money_calculation_l1601_160186

theorem prize_money_calculation (total : ℚ) (rica_share : ℚ) (rica_spent : ℚ) (rica_left : ℚ) : 
  rica_share = 3 / 8 * total →
  rica_spent = 1 / 5 * rica_share →
  rica_left = rica_share - rica_spent →
  rica_left = 300 →
  total = 1000 := by
sorry

end prize_money_calculation_l1601_160186


namespace coefficient_implies_a_value_l1601_160109

theorem coefficient_implies_a_value (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = (1 + a * x)^5 * (1 - 2*x)^4) ∧
   (∃ c : ℝ → ℝ, (∀ x, f x = c 0 + c 1 * x + c 2 * x^2 + c 3 * x^3 + c 4 * x^4 + c 5 * x^5 + c 6 * x^6 + c 7 * x^7 + c 8 * x^8 + c 9 * x^9) ∧
    c 2 = -16)) →
  a = 2 := by
sorry

end coefficient_implies_a_value_l1601_160109


namespace cuboid_diagonal_l1601_160180

theorem cuboid_diagonal (x y z : ℝ) 
  (h1 : y * z = Real.sqrt 2)
  (h2 : z * x = Real.sqrt 3)
  (h3 : x * y = Real.sqrt 6) :
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 6 := by
sorry

end cuboid_diagonal_l1601_160180


namespace not_all_pairs_perfect_square_l1601_160142

theorem not_all_pairs_perfect_square (d : ℕ) (h1 : d > 0) (h2 : d ≠ 2) (h3 : d ≠ 5) (h4 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k^2 :=
by sorry

end not_all_pairs_perfect_square_l1601_160142


namespace even_digits_in_base_7_of_789_l1601_160134

def base_7_representation (n : ℕ) : List ℕ :=
  sorry

def count_even_digits (digits : List ℕ) : ℕ :=
  sorry

theorem even_digits_in_base_7_of_789 :
  count_even_digits (base_7_representation 789) = 3 := by
  sorry

end even_digits_in_base_7_of_789_l1601_160134


namespace workers_contribution_problem_l1601_160143

/-- The number of workers who raised money by equal contribution -/
def number_of_workers : ℕ := 1200

/-- The original total contribution in paise (100 paise = 1 rupee) -/
def original_total : ℕ := 30000000  -- 3 lacs = 300,000 rupees = 30,000,000 paise

/-- The new total contribution if each worker contributed 50 rupees extra, in paise -/
def new_total : ℕ := 36000000  -- 3.60 lacs = 360,000 rupees = 36,000,000 paise

/-- The extra contribution per worker in paise -/
def extra_contribution : ℕ := 5000  -- 50 rupees = 5,000 paise

theorem workers_contribution_problem :
  (original_total / number_of_workers : ℚ) * number_of_workers = original_total ∧
  ((original_total / number_of_workers : ℚ) + extra_contribution) * number_of_workers = new_total :=
sorry

end workers_contribution_problem_l1601_160143


namespace base_conversion_equality_l1601_160117

def base_five_to_decimal (n : ℕ) : ℕ := 
  (n / 100) * 25 + ((n / 10) % 10) * 5 + (n % 10)

def base_b_to_decimal (n : ℕ) (b : ℕ) : ℕ := 
  (n / 100) * b^2 + ((n / 10) % 10) * b + (n % 10)

theorem base_conversion_equality :
  ∃ (b : ℕ), b > 0 ∧ base_five_to_decimal 132 = base_b_to_decimal 210 b ∧ b = 4 := by
  sorry

end base_conversion_equality_l1601_160117


namespace prime_ap_difference_greater_than_30000_l1601_160196

/-- An arithmetic progression of prime numbers -/
structure PrimeArithmeticProgression where
  terms : Fin 15 → ℕ
  is_prime : ∀ i, Nat.Prime (terms i)
  is_increasing : ∀ i j, i < j → terms i < terms j
  is_arithmetic : ∀ i j k, terms j - terms i = terms k - terms j ↔ j - i = k - j

/-- The common difference of an arithmetic progression -/
def common_difference (ap : PrimeArithmeticProgression) : ℕ :=
  ap.terms 1 - ap.terms 0

/-- Theorem: The common difference of an arithmetic progression of 15 primes is greater than 30000 -/
theorem prime_ap_difference_greater_than_30000 (ap : PrimeArithmeticProgression) :
  common_difference ap > 30000 := by
  sorry

end prime_ap_difference_greater_than_30000_l1601_160196


namespace smallest_positive_e_l1601_160101

/-- Represents a polynomial of degree 4 with integer coefficients -/
structure IntPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- Checks if a given number is a root of the polynomial -/
def isRoot (p : IntPolynomial) (x : ℚ) : Prop :=
  p.a * x^4 + p.b * x^3 + p.c * x^2 + p.d * x + p.e = 0

/-- The main theorem stating the smallest possible value of e -/
theorem smallest_positive_e (p : IntPolynomial) : 
  p.e > 0 → 
  isRoot p (-2) → 
  isRoot p 5 → 
  isRoot p 9 → 
  isRoot p (-1/3) → 
  p.e ≥ 90 ∧ ∃ q : IntPolynomial, q.e = 90 ∧ 
    isRoot q (-2) ∧ isRoot q 5 ∧ isRoot q 9 ∧ isRoot q (-1/3) :=
sorry

end smallest_positive_e_l1601_160101


namespace abc_magnitude_order_l1601_160120

/-- Given the definitions of a, b, and c, prove that b > c > a -/
theorem abc_magnitude_order :
  let a := (1/2) * Real.cos (16 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (16 * π / 180)
  let b := (2 * Real.tan (14 * π / 180)) / (1 + Real.tan (14 * π / 180) ^ 2)
  let c := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  b > c ∧ c > a := by sorry

end abc_magnitude_order_l1601_160120


namespace participation_plans_count_l1601_160131

/-- The number of students to choose from, excluding the pre-selected student -/
def n : ℕ := 3

/-- The number of students to be selected, excluding the pre-selected student -/
def k : ℕ := 2

/-- The total number of students participating (including the pre-selected student) -/
def total_participants : ℕ := k + 1

/-- The number of subjects -/
def subjects : ℕ := 3

theorem participation_plans_count : 
  (n.choose k) * (Nat.factorial total_participants) = 18 := by
  sorry

end participation_plans_count_l1601_160131


namespace units_digit_7_pow_2023_l1601_160151

def units_digit (n : ℕ) : ℕ := n % 10

def power_7_units_digit_pattern : List ℕ := [7, 9, 3, 1]

theorem units_digit_7_pow_2023 :
  units_digit (7^2023) = 3 :=
by
  sorry

end units_digit_7_pow_2023_l1601_160151


namespace james_pizza_slices_l1601_160106

theorem james_pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) (james_fraction : ℚ) : 
  num_pizzas = 2 → 
  slices_per_pizza = 6 → 
  james_fraction = 2/3 →
  (↑num_pizzas * ↑slices_per_pizza : ℚ) * james_fraction = 8 := by
  sorry

end james_pizza_slices_l1601_160106


namespace no_divisor_of_form_24k_plus_20_l1601_160166

theorem no_divisor_of_form_24k_plus_20 (n : ℕ) : ¬ ∃ (k : ℕ), (24 * k + 20) ∣ (3^n + 1) := by
  sorry

end no_divisor_of_form_24k_plus_20_l1601_160166


namespace min_investment_optimal_quantities_l1601_160191

/-- Represents the cost and quantity of stationery types A and B -/
structure Stationery where
  cost_A : ℕ
  cost_B : ℕ
  quantity_A : ℕ
  quantity_B : ℕ

/-- Defines the conditions of the stationery purchase problem -/
def stationery_problem (s : Stationery) : Prop :=
  s.cost_A * 2 + s.cost_B = 35 ∧
  s.cost_A + s.cost_B * 3 = 30 ∧
  s.quantity_A + s.quantity_B = 120 ∧
  975 ≤ s.cost_A * s.quantity_A + s.cost_B * s.quantity_B ∧
  s.cost_A * s.quantity_A + s.cost_B * s.quantity_B ≤ 1000

/-- Theorem stating the minimum investment for the stationery purchase -/
theorem min_investment (s : Stationery) :
  stationery_problem s →
  s.cost_A * s.quantity_A + s.cost_B * s.quantity_B ≥ 980 :=
by sorry

/-- Theorem stating the optimal purchase quantities -/
theorem optimal_quantities (s : Stationery) :
  stationery_problem s →
  s.cost_A * s.quantity_A + s.cost_B * s.quantity_B = 980 →
  s.quantity_A = 38 ∧ s.quantity_B = 82 :=
by sorry

end min_investment_optimal_quantities_l1601_160191


namespace trig_fraction_equality_l1601_160165

theorem trig_fraction_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ x : ℝ, (Real.sin x)^4 / a + (Real.cos x)^4 / b = 1 / (a + b)) :
  ∀ x : ℝ, (Real.sin x)^8 / a^3 + (Real.cos x)^8 / b^3 = 1 / (a + b)^3 := by
  sorry

end trig_fraction_equality_l1601_160165


namespace total_students_is_thirteen_l1601_160187

/-- The number of students in a presentation order, where Eunjeong's position is known. -/
def total_students (students_before_eunjeong : ℕ) (eunjeong_position_from_last : ℕ) : ℕ :=
  students_before_eunjeong + 1 + (eunjeong_position_from_last - 1)

/-- Theorem stating that the total number of students is 13 given the problem conditions. -/
theorem total_students_is_thirteen :
  total_students 7 6 = 13 := by sorry

end total_students_is_thirteen_l1601_160187


namespace mets_fan_count_l1601_160182

/-- Represents the number of fans for each team -/
structure FanCount where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The total number of fans in the town -/
def total_fans : ℕ := 330

/-- The fan count satisfies the given ratios and total -/
def is_valid_fan_count (fc : FanCount) : Prop :=
  3 * fc.mets = 2 * fc.yankees ∧
  4 * fc.red_sox = 5 * fc.mets ∧
  fc.yankees + fc.mets + fc.red_sox = total_fans

theorem mets_fan_count (fc : FanCount) (h : is_valid_fan_count fc) : fc.mets = 88 := by
  sorry


end mets_fan_count_l1601_160182


namespace bag_to_items_ratio_l1601_160178

/-- The cost of a shirt in dollars -/
def shirt_cost : ℚ := 7

/-- The cost of a pair of shoes in dollars -/
def shoes_cost : ℚ := shirt_cost + 3

/-- The total cost of 2 shirts and a pair of shoes in dollars -/
def total_cost_without_bag : ℚ := 2 * shirt_cost + shoes_cost

/-- The total cost of all items (including the bag) in dollars -/
def total_cost : ℚ := 36

/-- The cost of the bag in dollars -/
def bag_cost : ℚ := total_cost - total_cost_without_bag

/-- Theorem stating that the ratio of the bag cost to the total cost without bag is 1:2 -/
theorem bag_to_items_ratio :
  bag_cost / total_cost_without_bag = 1 / 2 := by sorry

end bag_to_items_ratio_l1601_160178


namespace ellipse_tangent_collinearity_and_min_area_l1601_160175

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the right focus F
def F : ℝ × ℝ := (1, 0)

-- Define the point P on the line x = 4
def P : ℝ → ℝ × ℝ := λ t => (4, t)

-- Define the tangent points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the area of triangle PAB
def area_PAB (t : ℝ) : ℝ := sorry

theorem ellipse_tangent_collinearity_and_min_area :
  -- Part 1: A, F, and B are collinear
  ∃ k : ℝ, (1 - k) * A.1 + k * B.1 = F.1 ∧ (1 - k) * A.2 + k * B.2 = F.2 ∧
  -- Part 2: The minimum area of triangle PAB is 9/2
  ∃ t : ℝ, area_PAB t = 9/2 ∧ ∀ s : ℝ, area_PAB s ≥ area_PAB t := by
sorry

end ellipse_tangent_collinearity_and_min_area_l1601_160175


namespace recurrence_sequence_theorem_l1601_160199

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) (k : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ ∀ n, (a (n + 1))^2 = (a n) * (a (n + 2)) + k

/-- Three terms form an arithmetic sequence -/
def IsArithmeticSequence (x y z : ℝ) : Prop := y - x = z - y

theorem recurrence_sequence_theorem (a : ℕ → ℝ) (k : ℝ) 
  (h : RecurrenceSequence a k) :
  (k = (a 2 - a 1)^2 → IsArithmeticSequence (a 1) (a 2) (a 3)) ∧ 
  (k = 0 → IsArithmeticSequence (a 2) (a 4) (a 5) → 
    (a 2) / (a 1) = 1 ∨ (a 2) / (a 1) = (1 + Real.sqrt 5) / 2) := by
  sorry


end recurrence_sequence_theorem_l1601_160199


namespace tangent_line_x_ln_x_l1601_160146

/-- The equation of the tangent line to y = x ln x at (1, 0) is x - y - 1 = 0 -/
theorem tangent_line_x_ln_x (x y : ℝ) : 
  (∀ t, t > 0 → y = t * Real.log t) →  -- Definition of the curve
  (x = 1 ∧ y = 0) →                    -- Point of tangency
  (x - y - 1 = 0) :=                   -- Equation of the tangent line
by sorry

end tangent_line_x_ln_x_l1601_160146


namespace function_properties_l1601_160156

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) :
  (∀ x, f (1 + x) = f (x - 1)) →
  (∀ x, f (1 - x) = -f (x - 1)) →
  (is_periodic f 2 ∧ is_odd f) :=
by sorry

end function_properties_l1601_160156


namespace symmetric_points_existence_l1601_160123

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a * Real.exp (-x) else Real.log (x / a)

theorem symmetric_points_existence (a : ℝ) (h : a > 0) :
  (∃ x₀ : ℝ, x₀ > 1 ∧ f a (-x₀) = f a x₀) ↔ 0 < a ∧ a < Real.exp (-1) :=
sorry

end symmetric_points_existence_l1601_160123


namespace symmetric_line_wrt_y_axis_symmetric_line_example_l1601_160135

/-- Given a line with equation y = mx + b, the line symmetric to it
    with respect to the y-axis has equation y = -mx + b -/
theorem symmetric_line_wrt_y_axis (m b : ℝ) :
  let original_line := fun (x : ℝ) => m * x + b
  let symmetric_line := fun (x : ℝ) => -m * x + b
  ∀ x y : ℝ, symmetric_line x = y ↔ original_line (-x) = y := by sorry

/-- The equation of the line symmetric to y = 2x + 1 with respect to the y-axis is y = -2x + 1 -/
theorem symmetric_line_example :
  let original_line := fun (x : ℝ) => 2 * x + 1
  let symmetric_line := fun (x : ℝ) => -2 * x + 1
  ∀ x y : ℝ, symmetric_line x = y ↔ original_line (-x) = y := by sorry

end symmetric_line_wrt_y_axis_symmetric_line_example_l1601_160135


namespace value_of_a_l1601_160102

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 9 * x^2 + 6 * x - 7

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 18 * x + 6

-- Theorem statement
theorem value_of_a (a : ℝ) : f' a (-1) = 4 → a = 16/3 := by
  sorry

end value_of_a_l1601_160102


namespace intersection_of_sets_l1601_160112

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 4, 6}
  A ∩ B = {2} := by
sorry

end intersection_of_sets_l1601_160112


namespace perpendicular_planes_parallel_l1601_160126

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line3D where
  point : Point3D
  direction : Point3D

structure Plane3D where
  point : Point3D
  normal : Point3D

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  l.direction.x * p.normal.x + l.direction.y * p.normal.y + l.direction.z * p.normal.z = 0

-- Define parallelism between two planes
def parallel (p1 p2 : Plane3D) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    p1.normal.x = k * p2.normal.x ∧
    p1.normal.y = k * p2.normal.y ∧
    p1.normal.z = k * p2.normal.z

-- State the theorem
theorem perpendicular_planes_parallel (l : Line3D) (p1 p2 : Plane3D) :
  perpendicular l p1 → perpendicular l p2 → parallel p1 p2 :=
sorry

end perpendicular_planes_parallel_l1601_160126


namespace grocery_bag_capacity_l1601_160127

theorem grocery_bag_capacity (bag_capacity : ℕ) (green_beans : ℕ) (milk : ℕ) (carrot_multiplier : ℕ) :
  bag_capacity = 20 →
  green_beans = 4 →
  milk = 6 →
  carrot_multiplier = 2 →
  bag_capacity - (green_beans + milk + carrot_multiplier * green_beans) = 2 := by
  sorry

end grocery_bag_capacity_l1601_160127


namespace complex_sum_equality_l1601_160181

theorem complex_sum_equality : 
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -5
  let R : ℂ := 3*I
  let T : ℂ := 1 + 5*I
  B - Q + R + T = -1 + 10*I := by
  sorry

end complex_sum_equality_l1601_160181


namespace right_triangle_perimeter_l1601_160113

theorem right_triangle_perimeter (area : ℝ) (leg : ℝ) (h1 : area = 36) (h2 : leg = 12) :
  ∃ (perimeter : ℝ), perimeter = 18 + 6 * Real.sqrt 5 := by
  sorry

end right_triangle_perimeter_l1601_160113


namespace largest_prime_divisor_of_20112021_base5_l1601_160168

/-- Converts a base 5 number represented as a string to a natural number -/
def base5ToNat (s : String) : ℕ := sorry

/-- Checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- Finds the largest prime divisor of a natural number -/
def largestPrimeDivisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_20112021_base5 :
  largestPrimeDivisor (base5ToNat "20112021") = 419 := by sorry

end largest_prime_divisor_of_20112021_base5_l1601_160168


namespace five_dogs_not_eating_any_l1601_160160

/-- The number of dogs that do not eat any of the three foods (watermelon, salmon, chicken) -/
def dogs_not_eating_any (total dogs_watermelon dogs_salmon dogs_chicken dogs_watermelon_salmon dogs_chicken_salmon_not_watermelon : ℕ) : ℕ :=
  total - (dogs_watermelon + dogs_salmon + dogs_chicken - dogs_watermelon_salmon - dogs_chicken_salmon_not_watermelon)

/-- Theorem stating that 5 dogs do not eat any of the three foods -/
theorem five_dogs_not_eating_any :
  dogs_not_eating_any 75 15 54 20 12 7 = 5 := by
  sorry

end five_dogs_not_eating_any_l1601_160160


namespace total_jelly_beans_l1601_160153

/-- The number of vanilla jelly beans -/
def vanilla_jb : ℕ := 120

/-- The number of grape jelly beans -/
def grape_jb : ℕ := 5 * vanilla_jb + 50

/-- The total number of jelly beans -/
def total_jb : ℕ := vanilla_jb + grape_jb

theorem total_jelly_beans : total_jb = 770 := by
  sorry

end total_jelly_beans_l1601_160153


namespace marble_probability_l1601_160173

theorem marble_probability (total : ℕ) (p_white p_red_or_blue : ℚ) :
  total = 90 →
  p_white = 1/3 →
  p_red_or_blue = 7/15 →
  ∃ (white red blue green : ℕ),
    white + red + blue + green = total ∧
    p_white = white / total ∧
    p_red_or_blue = (red + blue) / total ∧
    green / total = 1/5 :=
by sorry

end marble_probability_l1601_160173


namespace xyz_value_l1601_160114

theorem xyz_value (a b c x y z : ℂ) 
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (sum_prod : x * y + x * z + y * z = 7)
  (sum : x + y + z = 4) :
  x * y * z = 6 := by
sorry

end xyz_value_l1601_160114


namespace james_hats_per_yard_l1601_160149

/-- The number of yards of velvet needed to make one cloak -/
def yards_per_cloak : ℕ := 3

/-- The total number of yards of velvet needed for 6 cloaks and 12 hats -/
def total_yards : ℕ := 21

/-- The number of cloaks made -/
def num_cloaks : ℕ := 6

/-- The number of hats made -/
def num_hats : ℕ := 12

/-- The number of hats James can make out of one yard of velvet -/
def hats_per_yard : ℕ := 4

theorem james_hats_per_yard :
  (total_yards - num_cloaks * yards_per_cloak) * hats_per_yard = num_hats := by
  sorry

end james_hats_per_yard_l1601_160149


namespace range_of_x2_plus_y2_l1601_160162

theorem range_of_x2_plus_y2 (x y : ℝ) 
  (h1 : x - 2*y + 1 ≥ 0) 
  (h2 : y ≥ x) 
  (h3 : x ≥ 0) : 
  0 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 2 := by
  sorry

#check range_of_x2_plus_y2

end range_of_x2_plus_y2_l1601_160162


namespace root_square_minus_three_x_minus_one_l1601_160194

theorem root_square_minus_three_x_minus_one (m : ℝ) : 
  m^2 - 3*m - 1 = 0 → 2*m^2 - 6*m = 2 := by
  sorry

end root_square_minus_three_x_minus_one_l1601_160194


namespace tangent_line_perpendicular_l1601_160145

/-- Given a quadratic function f(x) = ax² + 2, prove that if its tangent line
    at x = 1 is perpendicular to the line 2x - y + 1 = 0, then a = -1/4. -/
theorem tangent_line_perpendicular (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 + 2
  let f' : ℝ → ℝ := λ x ↦ 2 * a * x
  let tangent_slope : ℝ := f' 1
  let perpendicular_line_slope : ℝ := 2
  tangent_slope * perpendicular_line_slope = -1 → a = -1/4 := by
  sorry

end tangent_line_perpendicular_l1601_160145


namespace ornamental_bangles_pairs_l1601_160141

/-- The number of bangles in a dozen -/
def bangles_per_dozen : ℕ := 12

/-- The number of dozens in a box -/
def dozens_per_box : ℕ := 2

/-- The number of boxes needed -/
def num_boxes : ℕ := 20

/-- The number of bangles in a pair -/
def bangles_per_pair : ℕ := 2

theorem ornamental_bangles_pairs :
  (num_boxes * dozens_per_box * bangles_per_dozen) / bangles_per_pair = 240 := by
  sorry

end ornamental_bangles_pairs_l1601_160141


namespace prime_square_mod_twelve_l1601_160137

theorem prime_square_mod_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) :
  p^2 % 12 = 1 := by
  sorry

end prime_square_mod_twelve_l1601_160137


namespace factor_expression_l1601_160105

theorem factor_expression (a m : ℝ) : a * m^2 - a = a * (m - 1) * (m + 1) := by
  sorry

end factor_expression_l1601_160105


namespace product_of_numbers_l1601_160110

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 194) : x * y = -25 := by
  sorry

end product_of_numbers_l1601_160110


namespace jerrys_action_figures_l1601_160140

/-- The problem of Jerry's action figures -/
theorem jerrys_action_figures 
  (total : ℕ) -- Total number of action figures after adding
  (added : ℕ) -- Number of added action figures
  (h1 : total = 10) -- Given: The total number of action figures after adding is 10
  (h2 : added = 6) -- Given: The number of added action figures is 6
  : total - added = 4 := by
  sorry

end jerrys_action_figures_l1601_160140


namespace coefficient_a3_equals_80_l1601_160169

theorem coefficient_a3_equals_80 :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) (x : ℝ),
  (2 * x^2 + 1)^5 = a₀ + a₁ * x^2 + a₂ * x^4 + a₃ * x^6 + a₄ * x^8 + a₅ * x^10 →
  a₃ = 80 := by
sorry

end coefficient_a3_equals_80_l1601_160169


namespace actual_average_height_l1601_160100

/-- Proves that the actual average height of students is 174.62 cm given the initial conditions --/
theorem actual_average_height (n : ℕ) (initial_avg : ℝ) 
  (h1_recorded h1_actual h2_recorded h2_actual h3_recorded h3_actual : ℝ) :
  n = 50 ∧ 
  initial_avg = 175 ∧
  h1_recorded = 151 ∧ h1_actual = 136 ∧
  h2_recorded = 162 ∧ h2_actual = 174 ∧
  h3_recorded = 185 ∧ h3_actual = 169 →
  (n : ℝ) * initial_avg - (h1_recorded - h1_actual + h2_recorded - h2_actual + h3_recorded - h3_actual) = n * 174.62 :=
by sorry

end actual_average_height_l1601_160100


namespace point_D_coordinates_l1601_160107

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-1, 5)

theorem point_D_coordinates :
  let AD : ℝ × ℝ := (3 * (B.1 - A.1), 3 * (B.2 - A.2))
  let D : ℝ × ℝ := (A.1 + AD.1, A.2 + AD.2)
  D = (-7, 9) := by sorry

end point_D_coordinates_l1601_160107


namespace alexs_score_l1601_160189

theorem alexs_score (total_students : ℕ) (average_without_alex : ℚ) (average_with_alex : ℚ) :
  total_students = 20 →
  average_without_alex = 75 →
  average_with_alex = 76 →
  (total_students - 1) * average_without_alex + 95 = total_students * average_with_alex :=
by sorry

end alexs_score_l1601_160189


namespace schedule_count_eq_42_l1601_160158

/-- The number of employees -/
def n : ℕ := 6

/-- The number of days -/
def d : ℕ := 3

/-- The number of employees working each day -/
def k : ℕ := 2

/-- Calculates the number of ways to schedule employees with given restrictions -/
def schedule_count : ℕ :=
  Nat.choose n (2 * k) * Nat.choose (n - 2 * k) k - 
  2 * (Nat.choose (n - 1) k * Nat.choose (n - 1 - k) k) +
  Nat.choose (n - 2) k * Nat.choose (n - 2 - k) k

theorem schedule_count_eq_42 : schedule_count = 42 := by
  sorry

end schedule_count_eq_42_l1601_160158


namespace position_of_2005_2004_l1601_160161

/-- The sum of numerator and denominator for the fraction 2005/2004 -/
def target_sum : ℕ := 2005 + 2004

/-- The position of a fraction in the sequence -/
def position (n d : ℕ) : ℕ :=
  let s := n + d
  (s - 1) * (s - 2) / 2 + (s - n)

/-- The theorem stating the position of 2005/2004 in the sequence -/
theorem position_of_2005_2004 : position 2005 2004 = 8028032 := by
  sorry


end position_of_2005_2004_l1601_160161


namespace capital_city_free_after_year_l1601_160150

/-- Represents the state of a city (under spell or not) -/
inductive SpellState
| Free
| UnderSpell

/-- Represents the kingdom with 12 cities -/
structure Kingdom where
  cities : Fin 12 → SpellState

/-- Represents the magician's action on a city -/
def magicianAction (s : SpellState) : SpellState :=
  match s with
  | SpellState.Free => SpellState.UnderSpell
  | SpellState.UnderSpell => SpellState.Free

/-- Applies the magician's transformation to the kingdom -/
def monthlyTransformation (k : Kingdom) (startCity : Fin 12) : Kingdom :=
  { cities := λ i => 
      if i.val < startCity.val 
      then k.cities i 
      else magicianAction (k.cities i) }

/-- The state of the kingdom after 12 months -/
def afterTwelveMonths (k : Kingdom) : Kingdom :=
  (List.range 12).foldl (λ acc i => monthlyTransformation acc i) k

/-- The theorem to be proved -/
theorem capital_city_free_after_year (k : Kingdom) (capitalCity : Fin 12) :
  k.cities capitalCity = SpellState.Free →
  (afterTwelveMonths k).cities capitalCity = SpellState.Free :=
sorry

end capital_city_free_after_year_l1601_160150
