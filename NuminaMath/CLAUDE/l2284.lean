import Mathlib

namespace software_contract_probability_l2284_228415

/-- Given probabilities for a computer company's contract scenarios, 
    prove the probability of not getting the software contract. -/
theorem software_contract_probability 
  (p_hardware : ℝ) 
  (p_at_least_one : ℝ) 
  (p_both : ℝ) 
  (h1 : p_hardware = 4/5)
  (h2 : p_at_least_one = 9/10)
  (h3 : p_both = 3/10) :
  1 - (p_at_least_one - p_hardware + p_both) = 3/5 :=
by sorry

end software_contract_probability_l2284_228415


namespace age_problem_l2284_228442

theorem age_problem :
  ∃ (x y z w v : ℕ),
    x + y + z = 74 ∧
    x = 7 * w ∧
    y = 2 * w + 2 * v ∧
    z = 2 * w + 3 * v ∧
    x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ v > 0 ∧
    x = 28 ∧ y = 20 ∧ z = 26 :=
by sorry

end age_problem_l2284_228442


namespace no_gcd_solution_l2284_228499

theorem no_gcd_solution : ¬∃ (a b c : ℕ), 
  (Nat.gcd a b = Nat.factorial 30 + 111) ∧ 
  (Nat.gcd b c = Nat.factorial 40 + 234) ∧ 
  (Nat.gcd c a = Nat.factorial 50 + 666) := by
sorry

end no_gcd_solution_l2284_228499


namespace triangle_area_l2284_228473

/-- Given a triangle with sides AC, BC, and BD, prove that its area is 14 -/
theorem triangle_area (AC BC BD : ℝ) (h1 : AC = 4) (h2 : BC = 3) (h3 : BD = 10) :
  (1 / 2 : ℝ) * (BD - BC) * AC = 14 := by
  sorry

end triangle_area_l2284_228473


namespace perpendicular_vectors_l2284_228434

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

/-- Given two vectors a and b in ℝ², where a = (1,3) and b = (x,-1),
    if a is perpendicular to b, then x = 3 -/
theorem perpendicular_vectors (x : ℝ) : 
  perpendicular (1, 3) (x, -1) → x = 3 := by
  sorry

end perpendicular_vectors_l2284_228434


namespace xyz_value_l2284_228409

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 27)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : 
  x * y * z = 6 := by
sorry

end xyz_value_l2284_228409


namespace x_value_when_y_is_5_l2284_228407

-- Define the constant ratio
def k : ℚ := (5 * 3 - 6) / (2 * 2 + 10)

-- Define the relationship between x and y
def relation (x y : ℚ) : Prop := (5 * x - 6) / (2 * y + 10) = k

-- State the theorem
theorem x_value_when_y_is_5 :
  ∀ x : ℚ, relation x 2 → relation 3 2 → relation x 5 → x = 53 / 14 :=
sorry

end x_value_when_y_is_5_l2284_228407


namespace intersection_point_l2284_228452

def f (x : ℝ) : ℝ := 4 * x - 2

theorem intersection_point :
  ∃ (x : ℝ), f x = 0 ∧ x = 1/2 := by
  sorry

end intersection_point_l2284_228452


namespace smallest_other_integer_l2284_228425

theorem smallest_other_integer (m n x : ℕ) : 
  m = 30 → 
  x > 0 → 
  Nat.gcd m n = x + 3 → 
  Nat.lcm m n = x * (x + 3) → 
  n ≥ 70 ∧ ∃ (n' : ℕ), n' = 70 ∧ 
    Nat.gcd m n' = x + 3 ∧ 
    Nat.lcm m n' = x * (x + 3) := by
  sorry

end smallest_other_integer_l2284_228425


namespace factorization_xy_squared_minus_x_l2284_228460

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end factorization_xy_squared_minus_x_l2284_228460


namespace average_age_proof_l2284_228441

def average_age_after_leaving (initial_people : ℕ) (initial_average : ℚ) 
  (leaving_age1 : ℕ) (leaving_age2 : ℕ) : ℚ :=
  let total_age := initial_people * initial_average
  let remaining_age := total_age - (leaving_age1 + leaving_age2)
  let remaining_people := initial_people - 2
  remaining_age / remaining_people

theorem average_age_proof :
  average_age_after_leaving 7 28 22 25 = 29.8 := by
  sorry

end average_age_proof_l2284_228441


namespace polynomial_factorization_l2284_228445

theorem polynomial_factorization (m n : ℝ) : 
  (∀ x, x^2 + m*x + 6 = (x - 2)*(x + n)) → m = -5 := by
  sorry

end polynomial_factorization_l2284_228445


namespace perimeter_of_square_with_semicircular_arcs_l2284_228493

/-- The perimeter of a region bounded by four semicircular arcs constructed on the sides of a square with side length 1/π is equal to 2. -/
theorem perimeter_of_square_with_semicircular_arcs (π : ℝ) (h : π > 0) : 
  let side_length : ℝ := 1 / π
  let semicircle_length : ℝ := π * side_length / 2
  let num_semicircles : ℕ := 4
  num_semicircles * semicircle_length = 2 := by
  sorry

end perimeter_of_square_with_semicircular_arcs_l2284_228493


namespace perpendicular_vectors_l2284_228435

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

theorem perpendicular_vectors :
  let v1 := (2 * a.1 + b.1, 2 * a.2 + b.2)
  let v2 := (a.1 - 2 * b.1, a.2 - 2 * b.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0 := by sorry

end perpendicular_vectors_l2284_228435


namespace pocket_balls_theorem_l2284_228485

/-- Represents the number of balls in each pocket -/
def pocket_balls : List Nat := [2, 4, 5]

/-- The total number of ways to take a ball from any pocket -/
def total_ways_one_ball : Nat := pocket_balls.sum

/-- The total number of ways to take one ball from each pocket -/
def total_ways_three_balls : Nat := pocket_balls.prod

theorem pocket_balls_theorem :
  total_ways_one_ball = 11 ∧ total_ways_three_balls = 40 := by
  sorry

end pocket_balls_theorem_l2284_228485


namespace runner_picture_probability_l2284_228490

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lapTime : ℕ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the state of the race at a given time -/
def RaceState :=
  ℕ  -- time in seconds

/-- Represents the camera setup -/
structure Camera where
  coverageFraction : ℚ
  centerPosition : ℚ  -- fraction of track from start line

/-- Calculate the position of a runner at a given time -/
def runnerPosition (r : Runner) (t : ℕ) : ℚ :=
  sorry

/-- Check if a runner is in the camera's view -/
def isInPicture (r : Runner) (t : ℕ) (c : Camera) : Bool :=
  sorry

/-- The main theorem to prove -/
theorem runner_picture_probability :
  let alice : Runner := ⟨"Alice", 120, true⟩
  let ben : Runner := ⟨"Ben", 100, false⟩
  let camera : Camera := ⟨1/3, 0⟩
  let raceTime : ℕ := 900
  let totalOverlapTime : ℚ := 40/3
  (totalOverlapTime / 60 : ℚ) = 1333/6000 := by
  sorry

end runner_picture_probability_l2284_228490


namespace circle_center_and_radius_l2284_228446

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def Circle.center (c : Circle) : ℝ × ℝ := sorry

/-- The radius of a circle -/
def Circle.radius (c : Circle) : ℝ := sorry

/-- The given circle equation -/
def givenCircle : Circle :=
  { equation := fun x y => x^2 + y^2 - 2*x - 3 = 0 }

theorem circle_center_and_radius :
  Circle.center givenCircle = (1, 0) ∧ Circle.radius givenCircle = 2 := by sorry

end circle_center_and_radius_l2284_228446


namespace odd_floor_time_building_floor_time_l2284_228472

theorem odd_floor_time (total_floors : ℕ) (even_floor_time : ℕ) (total_time : ℕ) : ℕ :=
  let odd_floors := (total_floors + 1) / 2
  let even_floors := total_floors / 2
  let even_total_time := even_floors * even_floor_time
  let odd_total_time := total_time - even_total_time
  odd_total_time / odd_floors

/-- 
Given a building with 10 floors, where:
- It takes 15 seconds to reach each even-numbered floor
- It takes 120 seconds (2 minutes) to reach the 10th floor
Prove that it takes 9 seconds to reach each odd-numbered floor
-/
theorem building_floor_time : odd_floor_time 10 15 120 = 9 := by
  sorry

end odd_floor_time_building_floor_time_l2284_228472


namespace f_has_local_minimum_in_interval_l2284_228479

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log x

theorem f_has_local_minimum_in_interval :
  ∃ x₀ : ℝ, 1/2 < x₀ ∧ x₀ < 1 ∧ IsLocalMin f x₀ := by sorry

end f_has_local_minimum_in_interval_l2284_228479


namespace pushup_difference_l2284_228447

-- Define the number of push-ups for Zachary and the total
def zachary_pushups : ℕ := 44
def total_pushups : ℕ := 146

-- Define David's push-ups
def david_pushups : ℕ := total_pushups - zachary_pushups

-- State the theorem
theorem pushup_difference :
  david_pushups > zachary_pushups ∧
  david_pushups - zachary_pushups = 58 := by
  sorry

end pushup_difference_l2284_228447


namespace triangle_third_vertex_l2284_228426

/-- Given an obtuse triangle with vertices at (8, 6), (0, 0), and (x, 0),
    if the area of the triangle is 48 square units, then x = 16 or x = -16 -/
theorem triangle_third_vertex (x : ℝ) : 
  let v1 : ℝ × ℝ := (8, 6)
  let v2 : ℝ × ℝ := (0, 0)
  let v3 : ℝ × ℝ := (x, 0)
  let triangle_area := (1/2 : ℝ) * |v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2)|
  (triangle_area = 48) → (x = 16 ∨ x = -16) :=
by sorry

end triangle_third_vertex_l2284_228426


namespace circles_and_line_properties_l2284_228418

-- Define Circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

-- Define Circle D
def circle_D (x y : ℝ) : Prop := (x - 5)^2 + (y - 4)^2 = 4

-- Define the tangent line l
def line_l (x y : ℝ) : Prop := x = 5 ∨ 7*x - 24*y + 61 = 0

-- Theorem statement
theorem circles_and_line_properties :
  -- Part 1: Circles C and D are externally tangent
  (∃ (x y : ℝ), circle_C x y ∧ circle_D x y) ∧
  -- The distance between centers is equal to the sum of radii
  ((2 - 5)^2 + (0 - 4)^2 : ℝ) = (3 + 2)^2 ∧
  -- Part 2: Line l is tangent to Circle C and passes through (5,4)
  (∀ (x y : ℝ), line_l x y → 
    -- Line passes through (5,4)
    (x = 5 ∧ y = 4 ∨ 7*5 - 24*4 + 61 = 0) ∧
    -- Line is tangent to Circle C (distance from center to line is equal to radius)
    ((2*7 + 0*(-24) - 61)^2 / (7^2 + (-24)^2) : ℝ) = 3^2) :=
sorry

end circles_and_line_properties_l2284_228418


namespace min_value_theorem_l2284_228414

-- Define the function f
def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem min_value_theorem (a b c d : ℝ) (h1 : a < (2/3) * b) 
  (h2 : ∀ x y : ℝ, x < y → f a b c d x < f a b c d y) :
  ∃ m : ℝ, m = 1 ∧ ∀ k : ℝ, k = c / (2*b - 3*a) → m ≤ k :=
sorry

end min_value_theorem_l2284_228414


namespace expand_and_simplify_l2284_228440

theorem expand_and_simplify (x : ℝ) : 4 * (x + 3) * (2 * x + 7) = 8 * x^2 + 52 * x + 84 := by
  sorry

end expand_and_simplify_l2284_228440


namespace necessary_not_sufficient_l2284_228401

open Set

def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

theorem necessary_not_sufficient :
  (∀ x, x ∈ M ∩ P → (x ∈ M ∨ x ∈ P)) ∧
  (∃ x, (x ∈ M ∨ x ∈ P) ∧ x ∉ M ∩ P) :=
by sorry

end necessary_not_sufficient_l2284_228401


namespace total_renovation_time_is_79_5_l2284_228400

/-- Represents the renovation time for a house with specific room conditions. -/
def house_renovation_time (bedroom_time : ℝ) (bedroom_count : ℕ) (garden_time : ℝ) : ℝ :=
  let kitchen_time := 1.5 * bedroom_time
  let terrace_time := garden_time - 2
  let basement_time := 0.75 * kitchen_time
  let non_living_time := bedroom_time * bedroom_count + kitchen_time + garden_time + terrace_time + basement_time
  non_living_time + 2 * non_living_time

/-- Theorem stating that the total renovation time for the given house is 79.5 hours. -/
theorem total_renovation_time_is_79_5 :
  house_renovation_time 4 3 3 = 79.5 := by
  sorry

#eval house_renovation_time 4 3 3

end total_renovation_time_is_79_5_l2284_228400


namespace ping_pong_rackets_sold_l2284_228443

theorem ping_pong_rackets_sold (total_sales : ℝ) (avg_price : ℝ) (h1 : total_sales = 735) (h2 : avg_price = 9.8) :
  total_sales / avg_price = 75 := by
  sorry

end ping_pong_rackets_sold_l2284_228443


namespace find_N_l2284_228489

theorem find_N : ∃ N : ℕ, (10 + 11 + 12 + 13) / 4 = (1000 + 1001 + 1002 + 1003) / N ∧ N = 348 := by
  sorry

end find_N_l2284_228489


namespace inequality_solution_l2284_228438

theorem inequality_solution (x : ℝ) (h : x ≠ 4) : (x^2 + 4) / ((x - 4)^2) ≥ 0 := by
  sorry

end inequality_solution_l2284_228438


namespace imaginary_part_of_i_squared_times_one_plus_i_l2284_228458

theorem imaginary_part_of_i_squared_times_one_plus_i :
  Complex.im (Complex.I^2 * (1 + Complex.I)) = -1 := by
  sorry

end imaginary_part_of_i_squared_times_one_plus_i_l2284_228458


namespace sequence_problem_l2284_228421

def geometric_sequence (a : ℕ → ℝ) := ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1

def arithmetic_sequence (b : ℕ → ℝ) := ∀ n : ℕ, b (n + 1) - b n = b 2 - b 1

theorem sequence_problem (a b : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : arithmetic_sequence b) 
  (h3 : a 1 * a 6 * a 11 = -3 * Real.sqrt 3) 
  (h4 : b 1 + b 6 + b 11 = 7 * Real.pi) : 
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 := by
  sorry

end sequence_problem_l2284_228421


namespace matrix_transformation_l2284_228436

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 6; 2, 7]
def matrix_B (x : ℤ) : Matrix (Fin 2) (Fin 2) ℤ := !![6, 2; 1, x]

theorem matrix_transformation (x : ℤ) : 
  Matrix.det matrix_A = Matrix.det (matrix_B x) → x = 20 := by
  sorry

end matrix_transformation_l2284_228436


namespace bernardo_silvia_game_sum_of_digits_l2284_228461

theorem bernardo_silvia_game (N : ℕ) : N = 38 ↔ 
  (27 * N + 900 < 2000) ∧ 
  (27 * N + 900 ≥ 1925) ∧ 
  (∀ k : ℕ, k < N → (27 * k + 900 < 1925 ∨ 27 * k + 900 ≥ 2000)) :=
sorry

theorem sum_of_digits (N : ℕ) : N = 38 → (N % 10 + N / 10) = 11 :=
sorry

end bernardo_silvia_game_sum_of_digits_l2284_228461


namespace solution_set_F_max_value_F_inequality_holds_l2284_228450

-- Define the function F(x) = |x + 2| - 3|x|
def F (x : ℝ) : ℝ := |x + 2| - 3 * |x|

-- Theorem 1: The solution set of F(x) ≥ 0 is {x | -1/2 ≤ x ≤ 1}
theorem solution_set_F : 
  {x : ℝ | F x ≥ 0} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: The maximum value of F(x) is 2
theorem max_value_F : 
  ∃ (x : ℝ), F x = 2 ∧ ∀ (y : ℝ), F y ≤ 2 := by sorry

-- Corollary: The inequality F(x) ≥ a holds for all a ∈ (-∞, 2]
theorem inequality_holds :
  ∀ (a : ℝ), a ≤ 2 → ∃ (x : ℝ), F x ≥ a := by sorry

end solution_set_F_max_value_F_inequality_holds_l2284_228450


namespace min_c_value_l2284_228492

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b) (hbc : b < c)
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 2003 ∧ y = |x - a| + |x - b| + |x - c|) :
  ∀ c' : ℕ, (0 < c' ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < c' ∧
    ∃! (x y : ℝ), 2 * x + y = 2003 ∧ y = |x - a'| + |x - b'| + |x - c'|) → c' ≥ 1002 :=
by sorry

end min_c_value_l2284_228492


namespace damage_cost_is_1450_l2284_228487

/-- Calculates the total cost of damages caused by Jack --/
def total_damage_cost (num_tires : ℕ) (cost_per_tire : ℕ) (window_cost : ℕ) : ℕ :=
  num_tires * cost_per_tire + window_cost

/-- Proves that the total cost of damages is $1450 --/
theorem damage_cost_is_1450 :
  total_damage_cost 3 250 700 = 1450 :=
by sorry

end damage_cost_is_1450_l2284_228487


namespace quadrant_line_conditions_l2284_228449

/-- A line passing through the first, third, and fourth quadrants -/
structure QuadrantLine where
  k : ℝ
  b : ℝ
  first_quadrant : ∃ x y, x > 0 ∧ y > 0 ∧ y = k * x + b
  third_quadrant : ∃ x y, x < 0 ∧ y < 0 ∧ y = k * x + b
  fourth_quadrant : ∃ x y, x > 0 ∧ y < 0 ∧ y = k * x + b

/-- Theorem stating the conditions on k and b for a line passing through the first, third, and fourth quadrants -/
theorem quadrant_line_conditions (l : QuadrantLine) : l.k > 0 ∧ l.b < 0 := by
  sorry

end quadrant_line_conditions_l2284_228449


namespace problem_solution_l2284_228481

theorem problem_solution (a b m n : ℝ) : 
  (a = -(-(3 : ℝ))) → 
  (b = (-((1 : ℝ)/(2 : ℝ)))⁻¹) → 
  (|m - a| + |n + b| = 0) → 
  (m = 3 ∧ n = -2) := by sorry

end problem_solution_l2284_228481


namespace feeding_to_total_ratio_l2284_228466

/-- Represents the time Larry spends on his dog in minutes -/
structure DogTime where
  walking_playing : ℕ  -- Time spent walking and playing (in minutes)
  total : ℕ           -- Total time spent on the dog (in minutes)

/-- The ratio of feeding time to total time is 1:6 -/
theorem feeding_to_total_ratio (t : DogTime) 
  (h1 : t.walking_playing = 30 * 2)
  (h2 : t.total = 72) : 
  (t.total - t.walking_playing) * 6 = t.total :=
by sorry

end feeding_to_total_ratio_l2284_228466


namespace radical_product_simplification_l2284_228439

theorem radical_product_simplification (p : ℝ) :
  Real.sqrt (15 * p^3) * Real.sqrt (20 * p^2) * Real.sqrt (30 * p^5) = 30 * p^5 * Real.sqrt 10 := by
  sorry

end radical_product_simplification_l2284_228439


namespace additional_driving_hours_l2284_228455

/-- The number of hours Carl drives per day before promotion -/
def hours_per_day : ℝ := 2

/-- The number of days in a week -/
def days_per_week : ℝ := 7

/-- The total number of hours Carl drives in two weeks after promotion -/
def total_hours_two_weeks : ℝ := 40

/-- The number of weeks in the given period -/
def num_weeks : ℝ := 2

theorem additional_driving_hours :
  let hours_before := hours_per_day * days_per_week
  let hours_after := total_hours_two_weeks / num_weeks
  hours_after - hours_before = 6 := by sorry

end additional_driving_hours_l2284_228455


namespace velocity_at_2s_l2284_228423

-- Define the displacement function
def S (t : ℝ) : ℝ := 10 * t - t^2

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 10 - 2 * t

-- Theorem statement
theorem velocity_at_2s :
  v 2 = 6 := by sorry

end velocity_at_2s_l2284_228423


namespace bill_selling_price_l2284_228424

theorem bill_selling_price (purchase_price : ℝ) : 
  (purchase_price * 1.1 : ℝ) = 550 ∧ 
  (purchase_price * 0.9 * 1.3 : ℝ) - (purchase_price * 1.1 : ℝ) = 35 :=
by sorry

end bill_selling_price_l2284_228424


namespace cos_4theta_value_l2284_228454

/-- If e^(iθ) = (3 - i√2) / 4, then cos 4θ = 121/256 -/
theorem cos_4theta_value (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (3 - Complex.I * Real.sqrt 2) / 4) : 
  Real.cos (4 * θ) = 121 / 256 := by
  sorry

end cos_4theta_value_l2284_228454


namespace eight_coin_stack_exists_fourteen_mm_stack_has_eight_coins_l2284_228404

/-- Represents the types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the thickness of a given coin in millimeters --/
def coinThickness (c : Coin) : ℚ :=
  match c with
  | Coin.Penny => 155/100
  | Coin.Nickel => 195/100
  | Coin.Dime => 135/100
  | Coin.Quarter => 175/100

/-- Represents a stack of coins --/
def CoinStack := List Coin

/-- Calculates the height of a coin stack in millimeters --/
def stackHeight (stack : CoinStack) : ℚ :=
  stack.foldl (fun acc c => acc + coinThickness c) 0

/-- Theorem: There exists a stack of 8 coins with a height of exactly 14 mm --/
theorem eight_coin_stack_exists : ∃ (stack : CoinStack), stackHeight stack = 14 ∧ stack.length = 8 := by
  sorry

/-- Theorem: Any stack of coins with a height of exactly 14 mm must contain 8 coins --/
theorem fourteen_mm_stack_has_eight_coins (stack : CoinStack) :
  stackHeight stack = 14 → stack.length = 8 := by
  sorry

end eight_coin_stack_exists_fourteen_mm_stack_has_eight_coins_l2284_228404


namespace cathy_worked_180_hours_l2284_228470

/-- Calculates the total hours worked by Cathy over 2 months, given the following conditions:
  * Normal work schedule is 20 hours per week
  * There are 4 weeks in a month
  * The job lasts for 2 months
  * Cathy covers an additional week of shifts (20 hours) due to Chris's illness
-/
def cathys_total_hours (hours_per_week : ℕ) (weeks_per_month : ℕ) (months : ℕ) (extra_week_hours : ℕ) : ℕ :=
  hours_per_week * weeks_per_month * months + extra_week_hours

/-- Proves that Cathy worked 180 hours during the 2 months -/
theorem cathy_worked_180_hours :
  cathys_total_hours 20 4 2 20 = 180 := by
  sorry

end cathy_worked_180_hours_l2284_228470


namespace binary_10111_is_23_l2284_228475

/-- Converts a binary number represented as a list of bits (0 or 1) to its decimal equivalent -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.reverse.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number we want to convert -/
def binary_number : List Nat := [1, 0, 1, 1, 1]

/-- Theorem stating that the decimal representation of (10111)₂ is 23 -/
theorem binary_10111_is_23 : binary_to_decimal binary_number = 23 := by
  sorry

end binary_10111_is_23_l2284_228475


namespace polynomial_value_at_negative_one_l2284_228410

theorem polynomial_value_at_negative_one (r : ℝ) : 
  (fun x : ℝ => 3 * x^4 - 2 * x^3 + x^2 + 4 * x + r) (-1) = 0 → r = -2 := by
  sorry

end polynomial_value_at_negative_one_l2284_228410


namespace every_multiple_of_2_is_even_is_universal_l2284_228476

-- Define what it means for a number to be even
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define what it means for a number to be a multiple of 2
def MultipleOf2 (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define what a universal proposition is
def UniversalProposition (P : ℤ → Prop) : Prop :=
  ∀ x : ℤ, P x

-- Statement to prove
theorem every_multiple_of_2_is_even_is_universal :
  UniversalProposition (λ n : ℤ => MultipleOf2 n → IsEven n) :=
sorry

end every_multiple_of_2_is_even_is_universal_l2284_228476


namespace function_characterization_l2284_228419

theorem function_characterization
  (f : ℤ → ℤ)
  (h : ∀ m n : ℤ, f m + f n = max (f (m + n)) (f (m - n))) :
  ∃ k : ℕ, ∀ x : ℤ, f x = k * |x| :=
sorry

end function_characterization_l2284_228419


namespace correct_calculation_l2284_228428

theorem correct_calculation (x : ℤ) : 954 - x = 468 → 954 + x = 1440 := by
  sorry

end correct_calculation_l2284_228428


namespace modulo_eleven_residue_l2284_228416

theorem modulo_eleven_residue : (308 + 6 * 44 + 8 * 165 + 3 * 18) % 11 = 10 := by
  sorry

end modulo_eleven_residue_l2284_228416


namespace trig_inequality_l2284_228459

open Real

theorem trig_inequality (a b c d : ℝ) : 
  a = sin (sin (2009 * π / 180)) →
  b = sin (cos (2009 * π / 180)) →
  c = cos (sin (2009 * π / 180)) →
  d = cos (cos (2009 * π / 180)) →
  b < a ∧ a < d ∧ d < c := by sorry

end trig_inequality_l2284_228459


namespace certain_number_proof_l2284_228491

theorem certain_number_proof (x : ℕ) : x > 72 ∧ x ∣ (72 * 14) ∧ (∀ y : ℕ, y > 72 ∧ y ∣ (72 * 14) → x ≤ y) → x = 84 := by
  sorry

end certain_number_proof_l2284_228491


namespace jen_addition_problem_l2284_228427

/-- Rounds a natural number to the nearest hundred. -/
def roundToNearestHundred (n : ℕ) : ℕ :=
  (n + 50) / 100 * 100

/-- The problem statement -/
theorem jen_addition_problem :
  roundToNearestHundred (178 + 269) = 400 := by
  sorry

end jen_addition_problem_l2284_228427


namespace special_polynomial_sum_l2284_228462

/-- A monic polynomial of degree 4 satisfying specific conditions -/
def SpecialPolynomial (p : ℝ → ℝ) : Prop :=
  (∀ x, ∃ a b c d : ℝ, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  p 1 = 20 ∧ p 2 = 40 ∧ p 3 = 60

/-- The sum of p(0) and p(4) for a special polynomial p is 92 -/
theorem special_polynomial_sum (p : ℝ → ℝ) (h : SpecialPolynomial p) : 
  p 0 + p 4 = 92 := by
  sorry

end special_polynomial_sum_l2284_228462


namespace no_integer_roots_l2284_228486

theorem no_integer_roots : ∀ (x : ℤ), x^3 - 3*x^2 - 10*x + 20 ≠ 0 := by
  sorry

end no_integer_roots_l2284_228486


namespace complement_determines_set_l2284_228402

def U : Set Nat := {0, 1, 2, 3}

theorem complement_determines_set (A : Set Nat) 
  (h1 : U = {0, 1, 2, 3})
  (h2 : (U \ A) = {2}) : 
  A = {0, 1, 3} := by
sorry

end complement_determines_set_l2284_228402


namespace michaels_additional_money_michael_needs_additional_money_l2284_228444

/-- Calculates the additional money Michael needs to buy all items for Mother's Day. -/
theorem michaels_additional_money (michael_money : ℝ) 
  (cake_price discount_cake : ℝ) (bouquet_price tax_bouquet : ℝ) 
  (balloons_price : ℝ) (perfume_price_gbp discount_perfume gbp_to_usd : ℝ) 
  (album_price_eur tax_album eur_to_usd : ℝ) : ℝ :=
  let cake_cost := cake_price * (1 - discount_cake)
  let bouquet_cost := bouquet_price * (1 + tax_bouquet)
  let balloons_cost := balloons_price
  let perfume_cost := perfume_price_gbp * (1 - discount_perfume) * gbp_to_usd
  let album_cost := album_price_eur * (1 + tax_album) * eur_to_usd
  let total_cost := cake_cost + bouquet_cost + balloons_cost + perfume_cost + album_cost
  total_cost - michael_money

/-- Proves that Michael needs an additional $78.90 to buy all items. -/
theorem michael_needs_additional_money :
  michaels_additional_money 50 20 0.1 36 0.05 5 30 0.15 1.4 25 0.08 1.2 = 78.9 := by
  sorry

end michaels_additional_money_michael_needs_additional_money_l2284_228444


namespace mary_pizza_order_l2284_228408

def large_pizza_slices : ℕ := 8
def slices_eaten : ℕ := 7
def slices_remaining : ℕ := 9

theorem mary_pizza_order : 
  ∃ (pizzas_ordered : ℕ), 
    pizzas_ordered * large_pizza_slices = slices_eaten + slices_remaining ∧ 
    pizzas_ordered = 2 := by
  sorry

end mary_pizza_order_l2284_228408


namespace sufficient_not_necessary_l2284_228478

theorem sufficient_not_necessary (m : ℝ) (h : m > 0) :
  (∀ a b : ℝ, a > b ∧ b > 0 → (b + m) / (a + m) > b / a) ∧
  (∃ a b : ℝ, (b + m) / (a + m) > b / a ∧ ¬(a > b ∧ b > 0)) :=
sorry

end sufficient_not_necessary_l2284_228478


namespace school_trip_buses_l2284_228494

/-- The number of buses needed for a school trip -/
def buses_needed (students : ℕ) (seats_per_bus : ℕ) : ℕ :=
  (students + seats_per_bus - 1) / seats_per_bus

/-- Proof that 5 buses are needed for 45 students with 9 seats per bus -/
theorem school_trip_buses :
  buses_needed 45 9 = 5 := by
  sorry

end school_trip_buses_l2284_228494


namespace complex_location_l2284_228420

theorem complex_location (z : ℂ) (h : (z - 3) * (2 - Complex.I) = 5) : 
  0 < z.re ∧ 0 < z.im := by
  sorry

end complex_location_l2284_228420


namespace profit_formula_l2284_228405

-- Define variables
variable (C S P p n : ℝ)

-- Define the conditions
def condition1 : Prop := P = p * ((C + S) / 2)
def condition2 : Prop := P = S / n - C

-- Theorem statement
theorem profit_formula 
  (h1 : condition1 C S P p)
  (h2 : condition2 C S P n)
  : P = (S * (2 * n * p + 2 * p - n)) / (n * (2 * p + n)) :=
by sorry

end profit_formula_l2284_228405


namespace train_length_problem_l2284_228453

theorem train_length_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) : 
  faster_speed = 46 →
  slower_speed = 36 →
  passing_time = 36 →
  ∃ (train_length : ℝ), 
    train_length = 50 ∧ 
    2 * train_length = (faster_speed - slower_speed) * (5/18) * passing_time :=
by sorry

end train_length_problem_l2284_228453


namespace product_is_even_l2284_228412

theorem product_is_even (a b c : ℤ) : 
  ∃ k : ℤ, (7 * a + b - 2 * c + 1) * (3 * a - 5 * b + 4 * c + 10) = 2 * k := by
sorry

end product_is_even_l2284_228412


namespace arithmetic_sequence_sum_l2284_228495

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Theorem: In an arithmetic sequence with positive terms, 
    if a₂ = 1 - a₁ and a₄ = 9 - a₃, then a₄ + a₅ = 27 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
  (h1 : seq.a 2 = 1 - seq.a 1)
  (h2 : seq.a 4 = 9 - seq.a 3) :
  seq.a 4 + seq.a 5 = 27 := by
  sorry

end arithmetic_sequence_sum_l2284_228495


namespace intersection_line_slope_l2284_228497

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 2*y + 4 = 0

-- Define the intersection points
def intersection (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧
  circle2 C.1 C.2 ∧ circle2 D.1 D.2 ∧
  C ≠ D

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) (h : intersection C D) :
  (D.2 - C.2) / (D.1 - C.1) = 1 := by
  sorry

end intersection_line_slope_l2284_228497


namespace abs_sum_equality_l2284_228465

theorem abs_sum_equality (a b c : ℤ) (h : |a - b| + |c - a| = 1) :
  |a - c| + |c - b| + |b - a| = 2 := by
sorry

end abs_sum_equality_l2284_228465


namespace job_completion_time_l2284_228456

theorem job_completion_time (total_work : ℝ) (time_together time_person2 : ℝ) 
  (h1 : time_together > 0)
  (h2 : time_person2 > 0)
  (h3 : total_work > 0)
  (h4 : total_work / time_together = total_work / time_person2 + total_work / (24 : ℝ)) :
  total_work / (total_work / time_together - total_work / time_person2) = 24 := by
sorry

end job_completion_time_l2284_228456


namespace construction_cost_l2284_228467

/-- The cost of hiring builders to construct houses -/
theorem construction_cost
  (builders_per_floor : ℕ)
  (days_per_floor : ℕ)
  (pay_per_day : ℕ)
  (num_builders : ℕ)
  (num_houses : ℕ)
  (floors_per_house : ℕ)
  (h1 : builders_per_floor = 3)
  (h2 : days_per_floor = 30)
  (h3 : pay_per_day = 100)
  (h4 : num_builders = 6)
  (h5 : num_houses = 5)
  (h6 : floors_per_house = 6) :
  (num_houses * floors_per_house * days_per_floor * pay_per_day * num_builders) / builders_per_floor = 270000 :=
by sorry

end construction_cost_l2284_228467


namespace root_sum_reciprocal_l2284_228496

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a + 2 = 0) → 
  (b^3 - 2*b + 2 = 0) → 
  (c^3 - 2*c + 2 = 0) → 
  (1/(a+1) + 1/(b+1) + 1/(c+1) = -1) := by
sorry

end root_sum_reciprocal_l2284_228496


namespace miller_rabin_correct_for_primes_l2284_228480

/-- Miller-Rabin primality test function -/
def miller_rabin (n : ℕ) : Bool := sorry

/-- Definition of primality -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem miller_rabin_correct_for_primes (n : ℕ) (h : is_prime n) : 
  miller_rabin n = true := by sorry

end miller_rabin_correct_for_primes_l2284_228480


namespace alchemerion_age_proof_l2284_228431

/-- Alchemerion's age in years -/
def alchemerion_age : ℕ := 277

/-- Alchemerion's son's age in years -/
def son_age : ℕ := alchemerion_age / 3

/-- Alchemerion's father's age in years -/
def father_age : ℕ := 2 * alchemerion_age + 40

/-- The sum of Alchemerion's, his son's, and his father's ages -/
def total_age : ℕ := alchemerion_age + son_age + father_age

theorem alchemerion_age_proof :
  alchemerion_age = 3 * son_age ∧
  father_age = 2 * alchemerion_age + 40 ∧
  total_age = 1240 →
  alchemerion_age = 277 := by
  sorry

end alchemerion_age_proof_l2284_228431


namespace ratio_of_divisor_sums_l2284_228464

def N : ℕ := 34 * 34 * 63 * 270

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry
def sum_of_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_of_odd_divisors N) * 14 = sum_of_even_divisors N := by sorry

end ratio_of_divisor_sums_l2284_228464


namespace square_difference_representation_l2284_228469

theorem square_difference_representation (n : ℕ) :
  (∃ (a b : ℤ), n + a^2 = b^2) ↔ n % 4 ≠ 2 := by sorry

end square_difference_representation_l2284_228469


namespace quadratic_inequality_roots_l2284_228413

/-- Given a quadratic function f(x) = -2x^2 + cx - 8, 
    where f(x) < 0 only when x ∈ (-∞, 2) ∪ (6, ∞),
    prove that c = 16 -/
theorem quadratic_inequality_roots (c : ℝ) : 
  (∀ x : ℝ, -2 * x^2 + c * x - 8 < 0 ↔ x < 2 ∨ x > 6) → 
  c = 16 := by
  sorry

end quadratic_inequality_roots_l2284_228413


namespace intersection_of_A_and_B_l2284_228429

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}
def B : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, 0, 2} := by sorry

end intersection_of_A_and_B_l2284_228429


namespace equation_solution_l2284_228457

theorem equation_solution :
  ∀ x : ℚ, (40 : ℚ) / 60 = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end equation_solution_l2284_228457


namespace sequence_p_bounded_l2284_228471

def isPrime (n : ℕ) : Prop := sorry

def largestPrimeDivisor (n : ℕ) : ℕ := sorry

def sequenceP : ℕ → ℕ
  | 0 => 2  -- Assuming the sequence starts with 2
  | 1 => 3  -- Assuming the second prime is 3
  | (n + 2) => largestPrimeDivisor (sequenceP (n + 1) + sequenceP n + 2008)

theorem sequence_p_bounded :
  ∃ (M : ℕ), ∀ (n : ℕ), sequenceP n ≤ M :=
sorry

end sequence_p_bounded_l2284_228471


namespace existence_of_abcd_l2284_228484

theorem existence_of_abcd (n : ℕ) (h : n > 1) : ∃ (a b c d : ℕ),
  a = 3*n - 1 ∧
  b = n + 1 ∧
  c = 3*n + 1 ∧
  d = n - 1 ∧
  a + b = 4*n ∧
  c + d = 4*n ∧
  a * b - c * d = 4*n :=
by
  sorry

end existence_of_abcd_l2284_228484


namespace ratio_and_linear_equation_l2284_228451

theorem ratio_and_linear_equation (c d : ℝ) : 
  c / d = 4 → c = 20 - 6 * d → d = 2 := by sorry

end ratio_and_linear_equation_l2284_228451


namespace breakfast_customers_count_l2284_228468

/-- The number of customers during breakfast on Friday -/
def breakfast_customers : ℕ := 73

/-- The number of customers during lunch on Friday -/
def lunch_customers : ℕ := 127

/-- The number of customers during dinner on Friday -/
def dinner_customers : ℕ := 87

/-- The predicted number of customers for Saturday -/
def saturday_prediction : ℕ := 574

/-- Theorem stating that the number of customers during breakfast on Friday is 73 -/
theorem breakfast_customers_count : 
  breakfast_customers = 
    saturday_prediction / 2 - (lunch_customers + dinner_customers) :=
by
  sorry

#check breakfast_customers_count

end breakfast_customers_count_l2284_228468


namespace max_imaginary_part_of_roots_l2284_228411

open Complex

theorem max_imaginary_part_of_roots (z : ℂ) :
  z^6 - z^5 + z^4 - z^3 + z^2 - z + 1 = 0 →
  ∃ (φ : ℝ), -π/2 ≤ φ ∧ φ ≤ π/2 ∧
  (∀ (w : ℂ), w^6 - w^5 + w^4 - w^3 + w^2 - w + 1 = 0 →
    w.im ≤ Real.sin φ) ∧
  φ = (900 * π) / (7 * 180) :=
sorry

end max_imaginary_part_of_roots_l2284_228411


namespace acid_mixture_proof_l2284_228422

theorem acid_mixture_proof :
  let volume1 : ℝ := 4
  let concentration1 : ℝ := 0.60
  let volume2 : ℝ := 16
  let concentration2 : ℝ := 0.75
  let total_volume : ℝ := 20
  let final_concentration : ℝ := 0.72
  (volume1 * concentration1 + volume2 * concentration2) / total_volume = final_concentration ∧
  volume1 + volume2 = total_volume := by
sorry

end acid_mixture_proof_l2284_228422


namespace largest_equal_division_l2284_228432

theorem largest_equal_division (tim_sweets peter_sweets : ℕ) 
  (h1 : tim_sweets = 36) (h2 : peter_sweets = 44) : 
  Nat.gcd tim_sweets peter_sweets = 4 := by
  sorry

end largest_equal_division_l2284_228432


namespace initial_money_calculation_l2284_228430

theorem initial_money_calculation (initial_money : ℚ) : 
  (2 / 5 : ℚ) * initial_money = 200 → initial_money = 500 := by
  sorry

#check initial_money_calculation

end initial_money_calculation_l2284_228430


namespace more_stable_lower_variance_l2284_228417

/-- Represents an athlete's assessment scores -/
structure AthleteScores where
  variance : ℝ
  assessmentCount : ℕ

/-- Defines the stability of an athlete's scores based on variance -/
def moreStable (a b : AthleteScores) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two athletes with the same average score but different variances,
    the athlete with lower variance has more stable scores -/
theorem more_stable_lower_variance 
  (athleteA athleteB : AthleteScores)
  (hCount : athleteA.assessmentCount = athleteB.assessmentCount)
  (hCountPos : athleteA.assessmentCount > 0)
  (hVarA : athleteA.variance = 1.43)
  (hVarB : athleteB.variance = 0.82) :
  moreStable athleteB athleteA := by
  sorry

#check more_stable_lower_variance

end more_stable_lower_variance_l2284_228417


namespace ellipse_C_equation_l2284_228482

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 5 - y^2 / 4 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci and vertices of ellipse C
def ellipse_C_foci (x y : ℝ) : Prop := (x = Real.sqrt 5 ∧ y = 0) ∨ (x = -Real.sqrt 5 ∧ y = 0)
def ellipse_C_vertices (x y : ℝ) : Prop := (x = 3 ∧ y = 0) ∨ (x = -3 ∧ y = 0)

-- Theorem statement
theorem ellipse_C_equation :
  (∀ x y : ℝ, hyperbola x y → 
    ((x = Real.sqrt 5 ∧ y = 0) ∨ (x = -Real.sqrt 5 ∧ y = 0) → ellipse_C_vertices x y) ∧
    ((x = 3 ∧ y = 0) ∨ (x = -3 ∧ y = 0) → ellipse_C_foci x y)) →
  (∀ x y : ℝ, ellipse_C_foci x y ∨ ellipse_C_vertices x y → ellipse_C x y) :=
sorry

end ellipse_C_equation_l2284_228482


namespace orange_juice_bottles_l2284_228448

/-- Represents the number of bottles of each juice type -/
structure JuiceBottles where
  orange : ℕ
  apple : ℕ
  grape : ℕ

/-- Represents the cost in cents of each juice type -/
structure JuiceCosts where
  orange : ℕ
  apple : ℕ
  grape : ℕ

/-- The main theorem to prove -/
theorem orange_juice_bottles (b : JuiceBottles) (c : JuiceCosts) : 
  c.orange = 70 ∧ 
  c.apple = 60 ∧ 
  c.grape = 80 ∧ 
  b.orange + b.apple + b.grape = 100 ∧ 
  c.orange * b.orange + c.apple * b.apple + c.grape * b.grape = 7250 ∧
  b.apple = b.grape ∧
  b.orange = 2 * b.apple →
  b.orange = 50 := by
sorry

end orange_juice_bottles_l2284_228448


namespace tensor_A_equals_result_l2284_228437

def A : Set ℕ := {0, 2, 3}

def tensor_operation (S : Set ℕ) : Set ℕ :=
  {x | ∃ a b, a ∈ S ∧ b ∈ S ∧ x = a + b}

theorem tensor_A_equals_result : tensor_operation A = {0, 2, 3, 4, 5, 6} := by
  sorry

end tensor_A_equals_result_l2284_228437


namespace box_interior_surface_area_l2284_228433

theorem box_interior_surface_area :
  let original_length : ℕ := 25
  let original_width : ℕ := 35
  let corner_size : ℕ := 7
  let original_area := original_length * original_width
  let corner_area := corner_size * corner_size
  let total_corner_area := 4 * corner_area
  let remaining_area := original_area - total_corner_area
  remaining_area = 679 := by sorry

end box_interior_surface_area_l2284_228433


namespace product_of_fractions_l2284_228406

theorem product_of_fractions :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end product_of_fractions_l2284_228406


namespace quadratic_roots_integer_l2284_228463

theorem quadratic_roots_integer (p : ℤ) :
  (∃ x y : ℤ, x ≠ y ∧ x^2 + p*x + p + 4 = 0 ∧ y^2 + p*y + p + 4 = 0) →
  p = 8 ∨ p = -4 := by
sorry

end quadratic_roots_integer_l2284_228463


namespace school_trip_equation_correct_l2284_228474

/-- Represents the scenario of a school trip to Shaoshan -/
structure SchoolTrip where
  distance : ℝ
  delay : ℝ
  speedRatio : ℝ

/-- The equation representing the travel times for bus and car -/
def travelTimeEquation (trip : SchoolTrip) (x : ℝ) : Prop :=
  trip.distance / x = trip.distance / (trip.speedRatio * x) + trip.delay

/-- Theorem stating that the given equation correctly represents the scenario -/
theorem school_trip_equation_correct (x : ℝ) : 
  let trip : SchoolTrip := { 
    distance := 50,
    delay := 1/6,
    speedRatio := 1.2
  }
  travelTimeEquation trip x :=
by sorry

end school_trip_equation_correct_l2284_228474


namespace expression_evaluation_l2284_228498

theorem expression_evaluation (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (h : x = 1 / z) :
  (x^3 - 1/x^3) * (z^3 + 1/z^3) = x^6 - 1/x^6 := by
  sorry

end expression_evaluation_l2284_228498


namespace number_calculation_l2284_228403

theorem number_calculation (n x : ℝ) (h1 : x = 0.8999999999999999) (h2 : n / x = 0.01) :
  n = 0.008999999999999999 := by
sorry

end number_calculation_l2284_228403


namespace at_least_n_minus_two_have_real_root_l2284_228483

/-- A linear function of the form ax + b where a ≠ 0 -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0

/-- The product of all LinearFunctions except the i-th one -/
def productExcept (funcs : List LinearFunction) (i : Nat) : LinearFunction → LinearFunction :=
  sorry

/-- The polynomial formed by the sum of the product of n-1 functions and the remaining function -/
def formPolynomial (funcs : List LinearFunction) (i : Nat) : LinearFunction :=
  sorry

/-- A function has a real root if there exists a real number x such that f(x) = 0 -/
def hasRealRoot (f : LinearFunction) : Prop :=
  ∃ x : ℝ, f.a * x + f.b = 0

/-- The main theorem -/
theorem at_least_n_minus_two_have_real_root (funcs : List LinearFunction) :
  funcs.length ≥ 3 →
  ∃ (roots : List LinearFunction),
    roots.length ≥ funcs.length - 2 ∧
    ∀ f ∈ roots, ∃ i, f = formPolynomial funcs i ∧ hasRealRoot f :=
  sorry

end at_least_n_minus_two_have_real_root_l2284_228483


namespace rectangle_area_l2284_228488

/-- A rectangle with three congruent circles inside -/
structure RectangleWithCircles where
  -- The length of the rectangle
  length : ℝ
  -- The width of the rectangle
  width : ℝ
  -- The diameter of each circle
  circle_diameter : ℝ
  -- The circles are congruent
  circles_congruent : True
  -- Each circle is tangent to two sides of the rectangle
  circles_tangent : True
  -- The circle centered at F is tangent to sides JK and LM
  circle_f_tangent : True
  -- The diameter of circle F is 5
  circle_f_diameter : circle_diameter = 5

/-- The area of the rectangle JKLM is 50 -/
theorem rectangle_area (r : RectangleWithCircles) : r.length * r.width = 50 := by
  sorry

end rectangle_area_l2284_228488


namespace truck_profit_analysis_l2284_228477

def initial_cost : ℕ := 490000
def first_year_expense : ℕ := 60000
def annual_expense_increase : ℕ := 20000
def annual_income : ℕ := 250000

def profit_function (n : ℕ) : ℤ := -n^2 + 20*n - 49

def option1_sell_price : ℕ := 40000
def option2_sell_price : ℕ := 130000

theorem truck_profit_analysis :
  -- 1. Profit function
  (∀ n : ℕ, profit_function n = annual_income * n - (first_year_expense * n + (n * (n - 1) / 2) * annual_expense_increase) - initial_cost) ∧
  -- 2. Profit exceeds 150,000 in 5th year
  (profit_function 5 > 150 ∧ ∀ k < 5, profit_function k ≤ 150) ∧
  -- 3. Maximum profit at n = 10
  (∀ n : ℕ, profit_function n ≤ profit_function 10) ∧
  -- 4. Maximum average annual profit at n = 7
  (∀ n : ℕ, n ≠ 0 → profit_function n / n ≤ profit_function 7 / 7) ∧
  -- 5. Both options yield 550,000 total profit
  (profit_function 10 + option1_sell_price = 550000 ∧
   profit_function 7 + option2_sell_price = 550000) ∧
  -- 6. Option 2 is more time-efficient
  (7 < 10) :=
by sorry

end truck_profit_analysis_l2284_228477
