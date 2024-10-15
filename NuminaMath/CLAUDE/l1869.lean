import Mathlib

namespace NUMINAMATH_CALUDE_searchlight_revolutions_per_minute_l1869_186961

/-- 
Given a searchlight that completes one revolution in a time period where 
half of that period is 10 seconds of darkness, prove that the number of 
revolutions per minute is 3.
-/
theorem searchlight_revolutions_per_minute : 
  ∀ (r : ℝ), 
  (r > 0) →  -- r is positive (revolutions per minute)
  (60 / r / 2 = 10) →  -- half the period of one revolution is 10 seconds
  r = 3 := by sorry

end NUMINAMATH_CALUDE_searchlight_revolutions_per_minute_l1869_186961


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1869_186942

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 4

/-- The total number of people that can ride the wheel at the same time -/
def total_people : ℕ := 20

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := total_people / num_seats

theorem ferris_wheel_capacity : people_per_seat = 5 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1869_186942


namespace NUMINAMATH_CALUDE_handshake_arrangements_l1869_186953

/-- The number of ways to arrange 10 people into two rings of 5, where each person in a ring is connected to 3 others -/
def M : ℕ := sorry

/-- The number of ways to select 5 people from 10 -/
def choose_five_from_ten : ℕ := sorry

/-- The number of arrangements within a ring of 5 -/
def ring_arrangements : ℕ := sorry

theorem handshake_arrangements :
  M = choose_five_from_ten * ring_arrangements * ring_arrangements ∧
  M % 1000 = 288 := by sorry

end NUMINAMATH_CALUDE_handshake_arrangements_l1869_186953


namespace NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_proof_l1869_186974

theorem bridge_length_calculation (train_length : Real) (train_speed_kmh : Real) (time_to_pass : Real) : Real :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * time_to_pass
  let bridge_length := total_distance - train_length
  bridge_length

theorem bridge_length_proof :
  bridge_length_calculation 250 35 41.142857142857146 = 150 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_proof_l1869_186974


namespace NUMINAMATH_CALUDE_road_travel_cost_l1869_186980

/-- The cost of traveling two intersecting roads on a rectangular lawn -/
theorem road_travel_cost (lawn_length lawn_width road_width travel_cost_per_sqm : ℕ) : 
  lawn_length = 90 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  travel_cost_per_sqm = 3 →
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * travel_cost_per_sqm = 4200 :=
by sorry

end NUMINAMATH_CALUDE_road_travel_cost_l1869_186980


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1869_186998

theorem expand_and_simplify (x : ℝ) : (x + 3) * (x - 4) = x^2 - x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1869_186998


namespace NUMINAMATH_CALUDE_child_tickets_sold_l1869_186996

theorem child_tickets_sold (adult_price child_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : total_revenue = 5100) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 400 := by
  sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l1869_186996


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l1869_186987

def solution_set : Set (ℤ × ℤ) := {(-1, -1), (0, -1), (-1, 0), (0, 0), (5, 2), (-6, 2)}

theorem integer_solutions_of_equation :
  {(x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l1869_186987


namespace NUMINAMATH_CALUDE_function_decomposition_into_symmetric_parts_l1869_186908

/-- A function is symmetric about the y-axis if f(x) = f(-x) for all x ∈ ℝ -/
def SymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

/-- A function is symmetric about the vertical line x = a if f(x) = f(2a - x) for all x ∈ ℝ -/
def SymmetricAboutVerticalLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f x = f (2 * a - x)

/-- Main theorem: Any function on ℝ can be represented as the sum of two symmetric functions -/
theorem function_decomposition_into_symmetric_parts (f : ℝ → ℝ) :
  ∃ (f₁ f₂ : ℝ → ℝ) (a : ℝ),
    (∀ x : ℝ, f x = f₁ x + f₂ x) ∧
    SymmetricAboutYAxis f₁ ∧
    a > 0 ∧
    SymmetricAboutVerticalLine f₂ a :=
  sorry

end NUMINAMATH_CALUDE_function_decomposition_into_symmetric_parts_l1869_186908


namespace NUMINAMATH_CALUDE_valid_points_characterization_l1869_186943

def is_in_second_quadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0

def satisfies_inequality (x y : ℤ) : Prop := y ≤ x + 4

def is_valid_point (x y : ℤ) : Prop :=
  is_in_second_quadrant x y ∧ satisfies_inequality x y

def valid_points : Set (ℤ × ℤ) :=
  {(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-3, 1)}

theorem valid_points_characterization :
  ∀ x y : ℤ, is_valid_point x y ↔ (x, y) ∈ valid_points := by sorry

end NUMINAMATH_CALUDE_valid_points_characterization_l1869_186943


namespace NUMINAMATH_CALUDE_subtract_point_six_from_forty_five_point_nine_l1869_186935

theorem subtract_point_six_from_forty_five_point_nine : 45.9 - 0.6 = 45.3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_point_six_from_forty_five_point_nine_l1869_186935


namespace NUMINAMATH_CALUDE_integer_less_than_sqrt_23_l1869_186924

theorem integer_less_than_sqrt_23 : ∃ n : ℤ, (n : ℝ) < Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_integer_less_than_sqrt_23_l1869_186924


namespace NUMINAMATH_CALUDE_triangle_angles_from_bisector_ratio_l1869_186934

theorem triangle_angles_from_bisector_ratio :
  ∀ (α β γ : ℝ),
  (α > 0) → (β > 0) → (γ > 0) →
  (α + β + γ = 180) →
  (∃ (k : ℝ), k > 0 ∧
    (α/2 + β/2 = 37*k) ∧
    (β/2 + γ/2 = 41*k) ∧
    (γ/2 + α/2 = 42*k)) →
  (α = 72 ∧ β = 66 ∧ γ = 42) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_from_bisector_ratio_l1869_186934


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l1869_186925

theorem polynomial_identity_sum_of_squares : 
  ∀ (p q r s t u : ℤ), 
  (∀ y, 729 * y^3 + 64 = (p * y^2 + q * y + r) * (s * y^2 + t * y + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 543106 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l1869_186925


namespace NUMINAMATH_CALUDE_circus_performance_time_l1869_186981

/-- Represents the time each entertainer stands on their back legs -/
structure CircusTime where
  pulsar : ℝ
  polly : ℝ
  petra : ℝ
  penny : ℝ
  parker : ℝ

/-- Calculates the total time all entertainers stand on their back legs -/
def totalTime (ct : CircusTime) : ℝ :=
  ct.pulsar + ct.polly + ct.petra + ct.penny + ct.parker

/-- Theorem stating the conditions and the result to be proved -/
theorem circus_performance_time :
  ∀ (ct : CircusTime),
    ct.pulsar = 10 →
    ct.polly = 3 * ct.pulsar →
    ct.petra = ct.polly / 6 →
    ct.penny = 2 * (ct.pulsar + ct.polly + ct.petra) →
    ct.parker = (ct.pulsar + ct.polly + ct.petra + ct.penny) / 4 →
    totalTime ct = 168.75 := by
  sorry


end NUMINAMATH_CALUDE_circus_performance_time_l1869_186981


namespace NUMINAMATH_CALUDE_fractional_expression_transformation_l1869_186967

theorem fractional_expression_transformation (x : ℝ) :
  let A : ℝ → ℝ := λ x => x^2 - 2*x
  x / (x + 2) = A x / (x^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_fractional_expression_transformation_l1869_186967


namespace NUMINAMATH_CALUDE_bisection_method_root_location_l1869_186918

def f (x : ℝ) := x^3 - 6*x^2 + 4

theorem bisection_method_root_location :
  (∃ r ∈ Set.Ioo 0 1, f r = 0) →
  (f 0 > 0) →
  (f 1 < 0) →
  (f (1/2) > 0) →
  ∃ r ∈ Set.Ioo (1/2) 1, f r = 0 := by sorry

end NUMINAMATH_CALUDE_bisection_method_root_location_l1869_186918


namespace NUMINAMATH_CALUDE_reflect_P_across_x_axis_l1869_186900

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-2, 3)

theorem reflect_P_across_x_axis : 
  reflect_x P = (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_across_x_axis_l1869_186900


namespace NUMINAMATH_CALUDE_pants_cost_rita_pants_cost_l1869_186913

/-- Calculates the cost of each pair of pants given Rita's shopping information -/
theorem pants_cost (initial_money : ℕ) (remaining_money : ℕ) (num_dresses : ℕ) (dress_cost : ℕ) 
  (num_pants : ℕ) (num_jackets : ℕ) (jacket_cost : ℕ) (transportation_cost : ℕ) : ℕ :=
  let total_spent := initial_money - remaining_money
  let dress_total := num_dresses * dress_cost
  let jacket_total := num_jackets * jacket_cost
  let pants_total := total_spent - dress_total - jacket_total - transportation_cost
  pants_total / num_pants

/-- Proves that each pair of pants costs $12 given Rita's shopping information -/
theorem rita_pants_cost : pants_cost 400 139 5 20 3 4 30 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_rita_pants_cost_l1869_186913


namespace NUMINAMATH_CALUDE_vector_b_determination_l1869_186923

def vector_a : ℝ × ℝ := (4, 3)

theorem vector_b_determination (b : ℝ × ℝ) 
  (h1 : (b.1 * vector_a.1 + b.2 * vector_a.2) / Real.sqrt (vector_a.1^2 + vector_a.2^2) = 4)
  (h2 : b.1 = 2) :
  b = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_b_determination_l1869_186923


namespace NUMINAMATH_CALUDE_piggy_bank_savings_l1869_186907

theorem piggy_bank_savings (x y : ℕ) : 
  x + y = 290 →  -- Total number of coins
  2 * (y / 4) = x / 3 →  -- Relationship between coin values
  2 * y + x = 406  -- Total amount saved
  := by sorry

end NUMINAMATH_CALUDE_piggy_bank_savings_l1869_186907


namespace NUMINAMATH_CALUDE_vaishali_saree_stripes_l1869_186949

theorem vaishali_saree_stripes :
  ∀ (brown gold blue : ℕ),
    gold = 3 * brown →
    blue = 5 * gold →
    blue = 60 →
    brown = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_vaishali_saree_stripes_l1869_186949


namespace NUMINAMATH_CALUDE_diaz_future_age_l1869_186929

/-- Proves Diaz's age 20 years from now given the conditions in the problem -/
theorem diaz_future_age (sierra_age : ℕ) (diaz_age : ℕ) : 
  sierra_age = 30 →
  10 * diaz_age - 40 = 10 * sierra_age + 20 →
  diaz_age + 20 = 56 := by
  sorry

end NUMINAMATH_CALUDE_diaz_future_age_l1869_186929


namespace NUMINAMATH_CALUDE_liar_knight_difference_district_A_l1869_186933

/-- Represents the number of residents in the city -/
def total_residents : ℕ := 50

/-- Represents the number of questions asked -/
def num_questions : ℕ := 4

/-- Represents the number of affirmative answers given by a knight -/
def knight_affirmative : ℕ := 1

/-- Represents the number of affirmative answers given by a liar -/
def liar_affirmative : ℕ := 3

/-- Represents the total number of affirmative answers given -/
def total_affirmative : ℕ := 290

/-- Theorem stating the difference between liars and knights in District A -/
theorem liar_knight_difference_district_A :
  ∃ (knights_A liars_A : ℕ),
    knights_A + liars_A ≤ total_residents ∧
    knights_A * knight_affirmative * num_questions +
    liars_A * liar_affirmative * num_questions ≤ total_affirmative ∧
    liars_A = knights_A + 3 := by
  sorry

end NUMINAMATH_CALUDE_liar_knight_difference_district_A_l1869_186933


namespace NUMINAMATH_CALUDE_solution_set_m_zero_solution_set_real_l1869_186956

-- Define the inequality
def inequality (m : ℝ) (x : ℝ) : Prop :=
  (m - 1) * x^2 + (m - 1) * x + 2 > 0

-- Part 1: Solution set when m = 0
theorem solution_set_m_zero :
  {x : ℝ | inequality 0 x} = Set.Ioo (-2) 1 := by sorry

-- Part 2: Range of m for solution set = ℝ
theorem solution_set_real :
  ∀ m : ℝ, (∀ x : ℝ, inequality m x) ↔ 1 ≤ m ∧ m < 9 := by sorry

end NUMINAMATH_CALUDE_solution_set_m_zero_solution_set_real_l1869_186956


namespace NUMINAMATH_CALUDE_height_difference_is_half_l1869_186997

/-- A circle tangent to the parabola y = x^2 + 1 at two points -/
structure TangentCircle where
  /-- x-coordinate of one tangent point -/
  a : ℝ
  /-- y-coordinate of the circle's center -/
  b : ℝ
  /-- Radius of the circle -/
  r : ℝ
  /-- The circle is tangent to the parabola at (a, a^2 + 1) and (-a, a^2 + 1) -/
  tangent_condition : (a^2 + ((a^2 + 1) - b)^2 = r^2) ∧ 
                      ((-a)^2 + (((-a)^2 + 1) - b)^2 = r^2)
  /-- The circle's center is on the y-axis -/
  center_on_y_axis : b > 0

/-- The difference in height between the circle's center and tangent points -/
def height_difference (c : TangentCircle) : ℝ :=
  c.b - (c.a^2 + 1)

/-- Theorem: The height difference is always 1/2 -/
theorem height_difference_is_half (c : TangentCircle) : 
  height_difference c = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_is_half_l1869_186997


namespace NUMINAMATH_CALUDE_min_unsuccessful_placements_l1869_186938

/-- A board is represented as a function from (Fin 8 × Fin 8) to Int -/
def Board := Fin 8 → Fin 8 → Int

/-- A tetromino is represented as a list of four pairs of coordinates -/
def Tetromino := List (Fin 8 × Fin 8)

/-- A valid board has only 1 and -1 as values -/
def validBoard (b : Board) : Prop :=
  ∀ i j, b i j = 1 ∨ b i j = -1

/-- A valid tetromino has four distinct cells within the board -/
def validTetromino (t : Tetromino) : Prop :=
  t.length = 4 ∧ t.Nodup

/-- The sum of a tetromino's cells on a board -/
def tetrominoSum (b : Board) (t : Tetromino) : Int :=
  t.foldl (fun sum (i, j) => sum + b i j) 0

/-- An unsuccessful placement has a non-zero sum -/
def unsuccessfulPlacement (b : Board) (t : Tetromino) : Prop :=
  tetrominoSum b t ≠ 0

/-- The main theorem -/
theorem min_unsuccessful_placements (b : Board) (h : validBoard b) :
  ∃ (unsuccessfulPlacements : List Tetromino),
    unsuccessfulPlacements.length ≥ 36 ∧
    ∀ t ∈ unsuccessfulPlacements, validTetromino t ∧ unsuccessfulPlacement b t :=
  sorry

end NUMINAMATH_CALUDE_min_unsuccessful_placements_l1869_186938


namespace NUMINAMATH_CALUDE_solve_water_problem_l1869_186920

def water_problem (initial_water evaporated_water rain_duration rain_rate final_water : ℝ) : Prop :=
  let water_after_evaporation := initial_water - evaporated_water
  let rainwater_added := (rain_duration / 10) * rain_rate
  let water_after_rain := water_after_evaporation + rainwater_added
  let water_drained := water_after_rain - final_water
  water_drained = 3500

theorem solve_water_problem :
  water_problem 6000 2000 30 350 1550 := by
  sorry

end NUMINAMATH_CALUDE_solve_water_problem_l1869_186920


namespace NUMINAMATH_CALUDE_custom_operation_solution_l1869_186993

-- Define the custom operation *
def star (a b : ℝ) : ℝ := 4 * a - 2 * b

-- State the theorem
theorem custom_operation_solution :
  ∃ x : ℝ, star 3 (star 4 x) = 10 ∧ x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_solution_l1869_186993


namespace NUMINAMATH_CALUDE_circle_center_transformation_l1869_186948

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (-2, 6)
  let reflected := reflect_x initial_center
  let final_position := translate_right reflected 5
  final_position = (3, -6) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l1869_186948


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_l1869_186901

/-- The maximum squared radius of a sphere fitting within two congruent right circular cones -/
theorem max_sphere_radius_squared (base_radius height intersection_distance : ℝ) 
  (hr : base_radius = 4)
  (hh : height = 10)
  (hi : intersection_distance = 4) : 
  ∃ (r : ℝ), r^2 = (528 - 32 * Real.sqrt 116) / 29 ∧ 
  ∀ (s : ℝ), s^2 ≤ (528 - 32 * Real.sqrt 116) / 29 := by
  sorry

#check max_sphere_radius_squared

end NUMINAMATH_CALUDE_max_sphere_radius_squared_l1869_186901


namespace NUMINAMATH_CALUDE_part_a_part_b_part_c_l1869_186911

/-- Rachel's jump length in cm -/
def rachel_jump : ℕ := 168

/-- Joel's jump length in cm -/
def joel_jump : ℕ := 120

/-- Mark's jump length in cm -/
def mark_jump : ℕ := 72

/-- Theorem for part (a) -/
theorem part_a (n : ℕ) : 
  n > 0 → 5 * rachel_jump = n * joel_jump → n = 7 := by sorry

/-- Theorem for part (b) -/
theorem part_b (r t : ℕ) : 
  r > 0 → t > 0 → 11 ≤ t → t ≤ 19 → r * joel_jump = t * mark_jump → r = 9 ∧ t = 15 := by sorry

/-- Theorem for part (c) -/
theorem part_c (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 → 
  a * rachel_jump = b * joel_jump → 
  b * joel_jump = c * mark_jump → 
  (∀ c' : ℕ, c' > 0 → c' * mark_jump = a * rachel_jump → c ≤ c') → 
  c = 35 := by sorry

end NUMINAMATH_CALUDE_part_a_part_b_part_c_l1869_186911


namespace NUMINAMATH_CALUDE_impossibleTransformation_l1869_186952

/-- Represents the three possible colors of the sides of the 99-gon -/
inductive Color
| Red
| Blue
| Yellow

/-- Represents the coloring of the 99-gon -/
def Coloring := Fin 99 → Color

/-- The initial coloring of the 99-gon -/
def initialColoring : Coloring :=
  fun i => match i.val % 3 with
    | 0 => Color.Red
    | 1 => Color.Blue
    | _ => Color.Yellow

/-- The target coloring of the 99-gon -/
def targetColoring : Coloring :=
  fun i => match i.val % 3 with
    | 0 => Color.Blue
    | 1 => Color.Red
    | _ => if i.val == 98 then Color.Blue else Color.Yellow

/-- Checks if a coloring is valid (no adjacent sides have the same color) -/
def isValidColoring (c : Coloring) : Prop :=
  ∀ i : Fin 98, c i ≠ c (i.succ)

/-- Represents a single color change operation -/
def colorChange (c : Coloring) (i : Fin 99) (newColor : Color) : Coloring :=
  fun j => if j = i then newColor else c j

/-- Theorem stating the impossibility of transforming the initial coloring to the target coloring -/
theorem impossibleTransformation :
  ¬∃ (steps : List (Fin 99 × Color)),
    (steps.foldl (fun acc (i, col) => colorChange acc i col) initialColoring = targetColoring) ∧
    (∀ step ∈ steps, isValidColoring (colorChange (steps.foldl (fun acc (i, col) => colorChange acc i col) initialColoring) step.fst step.snd)) :=
sorry


end NUMINAMATH_CALUDE_impossibleTransformation_l1869_186952


namespace NUMINAMATH_CALUDE_factor_implies_b_value_l1869_186954

theorem factor_implies_b_value (a b : ℤ) : 
  (∃ (c : ℤ), (X^2 - 2*X - 1) * (a*X - c) = a*X^3 + b*X^2 + 2) → b = -6 :=
by sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l1869_186954


namespace NUMINAMATH_CALUDE_least_n_with_zero_in_factorization_l1869_186970

/-- A function that checks if a positive integer contains the digit 0 -/
def containsZero (n : ℕ+) : Prop :=
  ∃ (k : ℕ), n.val = 10 * k ∨ n.val % 10 = 0

/-- A function that checks if all factorizations of 10^n contain a zero -/
def allFactorizationsContainZero (n : ℕ) : Prop :=
  ∀ (a b : ℕ+), a * b = 10^n → (containsZero a ∨ containsZero b)

/-- The main theorem stating that 8 is the least positive integer satisfying the condition -/
theorem least_n_with_zero_in_factorization :
  (allFactorizationsContainZero 8) ∧
  (∀ m : ℕ, m < 8 → ¬(allFactorizationsContainZero m)) :=
sorry

end NUMINAMATH_CALUDE_least_n_with_zero_in_factorization_l1869_186970


namespace NUMINAMATH_CALUDE_expr_is_monomial_l1869_186915

-- Define what a monomial is
def is_monomial (expr : ℚ → ℚ) : Prop :=
  ∃ (a : ℚ) (n : ℕ), ∀ x, expr x = a * x^n

-- Define the expression y/2023
def expr (y : ℚ) : ℚ := y / 2023

-- Theorem statement
theorem expr_is_monomial : is_monomial expr :=
sorry

end NUMINAMATH_CALUDE_expr_is_monomial_l1869_186915


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1869_186994

theorem p_necessary_not_sufficient_for_q :
  (∃ a b c : ℝ, a > b ∧ ¬(a * c^2 > b * c^2)) ∧
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1869_186994


namespace NUMINAMATH_CALUDE_min_red_chips_l1869_186965

/-- Represents the number of chips of each color -/
structure ChipCount where
  red : Nat
  blue : Nat

/-- Checks if a number is prime -/
def isPrime (n : Nat) : Prop := sorry

theorem min_red_chips :
  ∀ (chips : ChipCount),
  chips.red + chips.blue = 70 →
  isPrime (chips.red + 2 * chips.blue) →
  chips.red ≥ 69 :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_l1869_186965


namespace NUMINAMATH_CALUDE_darrel_nickels_l1869_186955

def quarters : ℕ := 76
def dimes : ℕ := 85
def pennies : ℕ := 150
def fee_percentage : ℚ := 10 / 100
def amount_after_fee : ℚ := 27

def quarter_value : ℚ := 25 / 100
def dime_value : ℚ := 10 / 100
def nickel_value : ℚ := 5 / 100
def penny_value : ℚ := 1 / 100

theorem darrel_nickels :
  let total_before_fee := amount_after_fee / (1 - fee_percentage)
  let known_coins_value := quarters * quarter_value + dimes * dime_value + pennies * penny_value
  let nickel_value_sum := total_before_fee - known_coins_value
  (nickel_value_sum / nickel_value : ℚ) = 20 := by sorry

end NUMINAMATH_CALUDE_darrel_nickels_l1869_186955


namespace NUMINAMATH_CALUDE_sum_of_factors_72_l1869_186946

/-- Sum of positive factors of a natural number n -/
def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

/-- The theorem stating that the sum of positive factors of 72 is 195 -/
theorem sum_of_factors_72 : sum_of_factors 72 = 195 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_72_l1869_186946


namespace NUMINAMATH_CALUDE_series_sum_equals_eleven_twentieths_l1869_186903

theorem series_sum_equals_eleven_twentieths : 
  (1 / 3 : ℚ) + (1 / 5 : ℚ) + (1 / 7 : ℚ) + (1 / 9 : ℚ) = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_eleven_twentieths_l1869_186903


namespace NUMINAMATH_CALUDE_max_t_value_min_y_value_equality_condition_l1869_186990

-- Define the inequality function
def f (x t : ℝ) : ℝ := |3*x + 2| + |3*x - 1| - t

-- Part 1: Maximum value of t
theorem max_t_value :
  (∀ x : ℝ, f x 3 ≥ 0) ∧ 
  (∀ t : ℝ, t > 3 → ∃ x : ℝ, f x t < 0) :=
sorry

-- Part 2: Minimum value of y
theorem min_y_value :
  ∀ m n : ℝ, m > 0 → n > 0 → 4*m + 5*n = 3 →
  1 / (m + 2*n) + 4 / (3*m + 3*n) ≥ 3 :=
sorry

-- Equality condition
theorem equality_condition :
  ∀ m n : ℝ, m > 0 → n > 0 → 4*m + 5*n = 3 →
  (1 / (m + 2*n) + 4 / (3*m + 3*n) = 3 ↔ m = 1/3 ∧ n = 1/3) :=
sorry

end NUMINAMATH_CALUDE_max_t_value_min_y_value_equality_condition_l1869_186990


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1869_186982

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) :
  (∀ n, a (n + 1) - a n = 2) →  -- arithmetic sequence with common difference 2
  (a 4)^2 = a 2 * a 5 →  -- a_2, a_4, a_5 form a geometric sequence
  a 2 = -8 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1869_186982


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1869_186991

/-- The area of a quadrilateral with non-perpendicular diagonals -/
theorem quadrilateral_area (a b c d φ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hφ : 0 < φ ∧ φ < π / 2) :
  let S := Real.tan φ * |a^2 + c^2 - b^2 - d^2| / 4
  ∃ (d₁ d₂ : ℝ), d₁ > 0 ∧ d₂ > 0 ∧ S = d₁ * d₂ * Real.sin φ / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1869_186991


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1869_186979

/-- Given that x and y are inversely proportional, prove that y = -49 when x = -8,
    given the conditions that x + y = 42 and x = 2y for some values of x and y. -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) 
    (h1 : x * y = k)  -- x and y are inversely proportional
    (h2 : ∃ (a b : ℝ), a + b = 42 ∧ a = 2 * b ∧ a * b = k) : 
  (-8 : ℝ) * y = k → y = -49 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1869_186979


namespace NUMINAMATH_CALUDE_quadratic_sum_has_root_l1869_186957

/-- A quadratic polynomial with a positive leading coefficient -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > 0

/-- The value of a quadratic polynomial at a given point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Two polynomials have a common root -/
def has_common_root (p q : QuadraticPolynomial) : Prop :=
  ∃ x, p.eval x = 0 ∧ q.eval x = 0

theorem quadratic_sum_has_root (p₁ p₂ p₃ : QuadraticPolynomial)
  (h₁₂ : has_common_root p₁ p₂)
  (h₂₃ : has_common_root p₂ p₃)
  (h₃₁ : has_common_root p₃ p₁) :
  ∃ x, (p₁.eval x + p₂.eval x + p₃.eval x = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_sum_has_root_l1869_186957


namespace NUMINAMATH_CALUDE_tourism_revenue_scientific_notation_l1869_186976

/-- Represents the tourism revenue in yuan -/
def tourism_revenue : ℝ := 12.41e9

/-- Represents the scientific notation of the tourism revenue -/
def scientific_notation : ℝ := 1.241e9

/-- Theorem stating that the tourism revenue is equal to its scientific notation representation -/
theorem tourism_revenue_scientific_notation : tourism_revenue = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_tourism_revenue_scientific_notation_l1869_186976


namespace NUMINAMATH_CALUDE_arithmetic_puzzle_2016_l1869_186947

/-- Represents a basic arithmetic operation --/
inductive Operation
  | Add
  | Subtract
  | Multiply
  | Divide

/-- Represents an arithmetic expression --/
inductive Expr
  | Num (n : ℕ)
  | Op (op : Operation) (e1 e2 : Expr)
  | Paren (e : Expr)

/-- Evaluates an arithmetic expression --/
def eval : Expr → ℚ
  | Expr.Num n => n
  | Expr.Op Operation.Add e1 e2 => eval e1 + eval e2
  | Expr.Op Operation.Subtract e1 e2 => eval e1 - eval e2
  | Expr.Op Operation.Multiply e1 e2 => eval e1 * eval e2
  | Expr.Op Operation.Divide e1 e2 => eval e1 / eval e2
  | Expr.Paren e => eval e

/-- Checks if an expression uses digits 1 through 9 in sequence --/
def usesDigitsInSequence : Expr → Bool
  | _ => sorry  -- Implementation omitted for brevity

theorem arithmetic_puzzle_2016 :
  ∃ (e : Expr), usesDigitsInSequence e ∧ eval e = 2016 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_puzzle_2016_l1869_186947


namespace NUMINAMATH_CALUDE_convex_polygon_with_equal_diagonals_l1869_186964

/-- A convex polygon with n sides and all diagonals equal -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 4
  all_diagonals_equal : Bool

/-- Theorem: If a convex n-gon (n ≥ 4) has all diagonals equal, then n is either 4 or 5 -/
theorem convex_polygon_with_equal_diagonals 
  {n : ℕ} (F : ConvexPolygon n) (h : F.all_diagonals_equal = true) : 
  n = 4 ∨ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_with_equal_diagonals_l1869_186964


namespace NUMINAMATH_CALUDE_age_problem_l1869_186905

theorem age_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 28 →
  (a + c) / 2 = 29 →
  b = 26 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1869_186905


namespace NUMINAMATH_CALUDE_cube_sum_implies_sum_l1869_186985

theorem cube_sum_implies_sum (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_implies_sum_l1869_186985


namespace NUMINAMATH_CALUDE_mike_shortfall_l1869_186921

def max_marks : ℕ := 800
def pass_percentage : ℚ := 30 / 100
def mike_score : ℕ := 212

theorem mike_shortfall :
  (↑max_marks * pass_percentage).floor - mike_score = 28 :=
sorry

end NUMINAMATH_CALUDE_mike_shortfall_l1869_186921


namespace NUMINAMATH_CALUDE_intersection_condition_union_condition_l1869_186914

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + a^2 - 12 = 0}

-- Part 1
theorem intersection_condition (a : ℝ) : A ∩ B a = A → a = -2 := by sorry

-- Part 2
theorem union_condition (a : ℝ) : A ∪ B a = A → a ≥ 4 ∨ a < -4 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_union_condition_l1869_186914


namespace NUMINAMATH_CALUDE_sausage_cost_per_pound_l1869_186944

theorem sausage_cost_per_pound : 
  let packages : ℕ := 3
  let pounds_per_package : ℕ := 2
  let total_cost : ℕ := 24
  let total_pounds := packages * pounds_per_package
  let cost_per_pound := total_cost / total_pounds
  cost_per_pound = 4 := by sorry

end NUMINAMATH_CALUDE_sausage_cost_per_pound_l1869_186944


namespace NUMINAMATH_CALUDE_mike_owes_jennifer_l1869_186910

theorem mike_owes_jennifer (rate : ℚ) (rooms : ℚ) (amount_owed : ℚ) : 
  rate = 13 / 3 → rooms = 8 / 5 → amount_owed = rate * rooms → amount_owed = 104 / 15 := by
  sorry

end NUMINAMATH_CALUDE_mike_owes_jennifer_l1869_186910


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l1869_186919

theorem largest_n_divisibility : 
  ∀ n : ℕ, n > 1098 → ¬(n + 11 ∣ n^3 + 101) ∧ (1098 + 11 ∣ 1098^3 + 101) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l1869_186919


namespace NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l1869_186936

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 4 * y + 6 = 0
def l₂ (a x y : ℝ) : Prop := ((3/4) * a + 1) * x + a * y - 3/2 = 0

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y ↔ l₂ a x y) ↔ a = 4 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → x * x + y * y = 0) ↔ (a = 0 ∨ a = -20/3) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l1869_186936


namespace NUMINAMATH_CALUDE_no_counterexamples_l1869_186927

def sum_of_digits (n : ℕ) : ℕ := sorry

def has_no_zero_digit (n : ℕ) : Prop := sorry

theorem no_counterexamples :
  ¬ ∃ N : ℕ, 
    (sum_of_digits N = 5) ∧ 
    (has_no_zero_digit N) ∧ 
    (Nat.Prime N) ∧ 
    (N % 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_counterexamples_l1869_186927


namespace NUMINAMATH_CALUDE_smallest_angle_representation_l1869_186986

theorem smallest_angle_representation (k : ℤ) (α : ℝ) : 
  (19 * π / 5 = 2 * k * π + α) → 
  (∀ β : ℝ, ∃ m : ℤ, 19 * π / 5 = 2 * m * π + β → |α| ≤ |β|) → 
  α = -π / 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_representation_l1869_186986


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l1869_186937

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l1869_186937


namespace NUMINAMATH_CALUDE_multiples_properties_l1869_186960

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 5 * k) 
  (hb : ∃ m : ℤ, b = 4 * m) : 
  (∃ n : ℤ, b = 2 * n) ∧ (∃ p : ℤ, a - b = 5 * p) := by
  sorry

end NUMINAMATH_CALUDE_multiples_properties_l1869_186960


namespace NUMINAMATH_CALUDE_probability_closer_to_center_l1869_186972

theorem probability_closer_to_center (r : ℝ) (h : r > 0) :
  let outer_circle_area := π * r^2
  let inner_circle_area := π * r
  let probability := inner_circle_area / outer_circle_area
  probability = 1/4 := by
sorry

end NUMINAMATH_CALUDE_probability_closer_to_center_l1869_186972


namespace NUMINAMATH_CALUDE_largest_number_l1869_186950

theorem largest_number (a b c d e : ℚ) 
  (sum1 : a + b + c + d = 210)
  (sum2 : a + b + c + e = 230)
  (sum3 : a + b + d + e = 250)
  (sum4 : a + c + d + e = 270)
  (sum5 : b + c + d + e = 290) :
  max a (max b (max c (max d e))) = 102.5 := by
sorry

end NUMINAMATH_CALUDE_largest_number_l1869_186950


namespace NUMINAMATH_CALUDE_twelve_digit_divisibility_l1869_186909

theorem twelve_digit_divisibility (n : ℕ) (h : 100000 ≤ n ∧ n < 1000000) :
  ∃ k : ℕ, 1000001 * n + n = 1000001 * k := by
  sorry

end NUMINAMATH_CALUDE_twelve_digit_divisibility_l1869_186909


namespace NUMINAMATH_CALUDE_stratified_random_most_appropriate_l1869_186912

/-- Represents a laboratory with a certain number of mice -/
structure Laboratory where
  mice : ℕ

/-- Represents a sampling method -/
inductive SamplingMethod
  | EqualFromEach
  | FullyRandom
  | ArbitraryStratified
  | StratifiedRandom

/-- The problem setup -/
def biochemistryLabs : List Laboratory := [
  { mice := 18 },
  { mice := 24 },
  { mice := 54 },
  { mice := 48 }
]

/-- The total number of mice to be selected -/
def selectionSize : ℕ := 24

/-- Function to determine the most appropriate sampling method -/
def mostAppropriateSamplingMethod (labs : List Laboratory) (selectionSize : ℕ) : SamplingMethod :=
  SamplingMethod.StratifiedRandom

/-- Theorem stating that StratifiedRandom is the most appropriate method -/
theorem stratified_random_most_appropriate :
  mostAppropriateSamplingMethod biochemistryLabs selectionSize = SamplingMethod.StratifiedRandom := by
  sorry


end NUMINAMATH_CALUDE_stratified_random_most_appropriate_l1869_186912


namespace NUMINAMATH_CALUDE_units_digit_of_2_to_2010_l1869_186959

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the cycle of units digits for powers of 2
def powerOfTwoCycle : List ℕ := [2, 4, 8, 6]

-- Theorem statement
theorem units_digit_of_2_to_2010 :
  unitsDigit (2^2010) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_to_2010_l1869_186959


namespace NUMINAMATH_CALUDE_largest_ball_on_specific_torus_l1869_186932

/-- The radius of the largest spherical ball that can be placed atop a torus -/
def largest_ball_radius (torus_center : ℝ × ℝ × ℝ) (torus_radius : ℝ) : ℝ :=
  let (x, y, z) := torus_center
  2

/-- Theorem: The radius of the largest spherical ball on a specific torus is 2 -/
theorem largest_ball_on_specific_torus :
  largest_ball_radius (4, 0, 2) 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_ball_on_specific_torus_l1869_186932


namespace NUMINAMATH_CALUDE_town_population_proof_l1869_186969

/-- The annual decrease rate of the town's population -/
def annual_decrease_rate : ℝ := 0.2

/-- The population after 2 years -/
def population_after_2_years : ℝ := 19200

/-- The initial population of the town -/
def initial_population : ℝ := 30000

theorem town_population_proof :
  let remaining_rate := 1 - annual_decrease_rate
  (remaining_rate ^ 2) * initial_population = population_after_2_years :=
by sorry

end NUMINAMATH_CALUDE_town_population_proof_l1869_186969


namespace NUMINAMATH_CALUDE_clock_hands_right_angle_period_l1869_186939

/-- The number of times clock hands are at right angles in 12 hours -/
def right_angles_per_12_hours : ℕ := 22

/-- The number of times clock hands are at right angles in the given period -/
def given_right_angles : ℕ := 88

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

theorem clock_hands_right_angle_period :
  (given_right_angles / right_angles_per_12_hours) * 12 = hours_per_day :=
sorry

end NUMINAMATH_CALUDE_clock_hands_right_angle_period_l1869_186939


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1869_186962

theorem solution_satisfies_system :
  let eq1 (x y : ℝ) := y + Real.sqrt (y - 3 * x) + 3 * x = 12
  let eq2 (x y : ℝ) := y^2 + y - 3 * x - 9 * x^2 = 144
  (eq1 (-24) 72 ∧ eq2 (-24) 72) ∧
  (eq1 (-4/3) 12 ∧ eq2 (-4/3) 12) := by
  sorry

#check solution_satisfies_system

end NUMINAMATH_CALUDE_solution_satisfies_system_l1869_186962


namespace NUMINAMATH_CALUDE_survey_analysis_l1869_186926

structure SurveyData where
  total : Nat
  aged_50_below_not_return : Nat
  aged_50_above_return : Nat
  aged_50_above_total : Nat

def chi_square (a b c d : Nat) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem survey_analysis (data : SurveyData) 
  (h1 : data.total = 100)
  (h2 : data.aged_50_below_not_return = 55)
  (h3 : data.aged_50_above_return = 15)
  (h4 : data.aged_50_above_total = 40) :
  let a := data.total - data.aged_50_above_total - data.aged_50_below_not_return
  let b := data.aged_50_below_not_return
  let c := data.aged_50_above_return
  let d := data.aged_50_above_total - data.aged_50_above_return
  (c : ℚ) / data.aged_50_above_total = 3 / 8 ∧ 
  chi_square a b c d > 10828 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_survey_analysis_l1869_186926


namespace NUMINAMATH_CALUDE_percent_relation_l1869_186945

theorem percent_relation (x y z : ℝ) (p : ℝ) 
  (h1 : y = 0.75 * x) 
  (h2 : z = 2 * x) 
  (h3 : p / 100 * z = 1.2 * y) : 
  p = 45 := by sorry

end NUMINAMATH_CALUDE_percent_relation_l1869_186945


namespace NUMINAMATH_CALUDE_contrapositive_diagonals_parallelogram_l1869_186977

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define what it means for diagonals to bisect each other
def diagonals_bisect (q : Quadrilateral) : Prop :=
  let mid1 := (q.vertices 0 + q.vertices 2) / 2
  let mid2 := (q.vertices 1 + q.vertices 3) / 2
  mid1 = mid2

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.vertices 0 - q.vertices 1) = (q.vertices 3 - q.vertices 2) ∧
  (q.vertices 0 - q.vertices 3) = (q.vertices 1 - q.vertices 2)

-- The theorem to prove
theorem contrapositive_diagonals_parallelogram :
  ∀ q : Quadrilateral, ¬(is_parallelogram q) → ¬(diagonals_bisect q) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_diagonals_parallelogram_l1869_186977


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l1869_186902

/-- Represents a right triangle with side lengths 6, 8, and 10 -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (is_right : a^2 + b^2 = c^2)
  (side_lengths : a = 6 ∧ b = 8 ∧ c = 10)

/-- Represents three mutually externally tangent circles -/
structure TangentCircles :=
  (r₁ : ℝ) (r₂ : ℝ) (r₃ : ℝ)
  (tangent_condition : r₁ + r₂ = 6 ∧ r₁ + r₃ = 8 ∧ r₂ + r₃ = 10)

/-- The sum of the areas of three mutually externally tangent circles
    centered at the vertices of a 6-8-10 right triangle is 56π -/
theorem sum_of_circle_areas (t : Triangle) (c : TangentCircles) :
  π * (c.r₁^2 + c.r₂^2 + c.r₃^2) = 56 * π :=
sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l1869_186902


namespace NUMINAMATH_CALUDE_traffic_class_multiple_l1869_186966

theorem traffic_class_multiple (drunk_drivers : ℕ) (total_students : ℕ) (M : ℕ) : 
  drunk_drivers = 6 →
  total_students = 45 →
  total_students = drunk_drivers + (M * drunk_drivers - 3) →
  M = 7 := by
sorry

end NUMINAMATH_CALUDE_traffic_class_multiple_l1869_186966


namespace NUMINAMATH_CALUDE_paths_from_A_to_D_l1869_186978

/-- The number of paths between two adjacent points -/
def paths_between_adjacent : ℕ := 2

/-- The number of direct paths from A to D -/
def direct_paths : ℕ := 1

/-- The total number of paths from A to D -/
def total_paths : ℕ := paths_between_adjacent^3 + direct_paths

/-- Theorem stating that the total number of paths from A to D is 9 -/
theorem paths_from_A_to_D : total_paths = 9 := by sorry

end NUMINAMATH_CALUDE_paths_from_A_to_D_l1869_186978


namespace NUMINAMATH_CALUDE_lunch_cost_proof_l1869_186963

/-- Proves that under given conditions, one person's lunch cost is $45 --/
theorem lunch_cost_proof (cost_A cost_R cost_J : ℚ) : 
  cost_A = (2/3) * cost_R →
  cost_R = cost_J →
  cost_A + cost_R + cost_J = 120 →
  cost_J = 45 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_proof_l1869_186963


namespace NUMINAMATH_CALUDE_three_from_eight_committee_l1869_186941

/-- The number of ways to select k items from n items without replacement and where order doesn't matter. -/
def combinations (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

/-- Theorem: There are 56 ways to select 3 people from a group of 8 people where order doesn't matter. -/
theorem three_from_eight_committee : combinations 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_from_eight_committee_l1869_186941


namespace NUMINAMATH_CALUDE_expected_heads_3000_tosses_l1869_186904

/-- A coin toss experiment with a fair coin -/
structure CoinTossExperiment where
  numTosses : ℕ
  probHeads : ℝ
  probHeads_eq : probHeads = 0.5

/-- The expected frequency of heads in a coin toss experiment -/
def expectedHeads (e : CoinTossExperiment) : ℝ :=
  e.numTosses * e.probHeads

/-- Theorem: The expected frequency of heads for 3000 tosses of a fair coin is 1500 -/
theorem expected_heads_3000_tosses (e : CoinTossExperiment) 
    (h : e.numTosses = 3000) : expectedHeads e = 1500 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_3000_tosses_l1869_186904


namespace NUMINAMATH_CALUDE_two_correct_probability_l1869_186951

/-- The number of packages and houses -/
def n : ℕ := 5

/-- The probability of exactly 2 out of n packages being delivered correctly -/
def prob_two_correct (n : ℕ) : ℚ :=
  if n ≥ 2 then
    (n.choose 2 : ℚ) / n.factorial
  else 0

theorem two_correct_probability :
  prob_two_correct n = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_two_correct_probability_l1869_186951


namespace NUMINAMATH_CALUDE_average_odd_one_digit_l1869_186992

def is_odd_one_digit (n : ℕ) : Prop := n % 2 = 1 ∧ n ≥ 1 ∧ n ≤ 9

def odd_one_digit_numbers : List ℕ := [1, 3, 5, 7, 9]

theorem average_odd_one_digit : 
  (List.sum odd_one_digit_numbers) / (List.length odd_one_digit_numbers) = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_odd_one_digit_l1869_186992


namespace NUMINAMATH_CALUDE_interest_rate_equation_l1869_186984

/-- Proves that the interest rate R satisfies the equation for the given conditions -/
theorem interest_rate_equation (P : ℝ) (n : ℝ) (R : ℝ) : 
  P = 10000 → n = 2 → P * ((1 + R/100)^n - (1 + n*R/100)) = 36 → R = 6 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_equation_l1869_186984


namespace NUMINAMATH_CALUDE_special_polynomial_at_seven_l1869_186958

/-- A monic polynomial of degree 7 satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∀ x, ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆, p x = x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) ∧
  p 0 = 0 ∧ p 1 = 1 ∧ p 2 = 2 ∧ p 3 = 3 ∧ p 4 = 4 ∧ p 5 = 5 ∧ p 6 = 6

/-- The theorem stating that any polynomial satisfying the special conditions will have p(7) = 5047 -/
theorem special_polynomial_at_seven (p : ℝ → ℝ) (h : special_polynomial p) : p 7 = 5047 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_at_seven_l1869_186958


namespace NUMINAMATH_CALUDE_seashell_difference_l1869_186989

theorem seashell_difference (craig_shells : ℕ) (craig_ratio : ℕ) (brian_ratio : ℕ) : 
  craig_shells = 54 → 
  craig_ratio = 9 → 
  brian_ratio = 7 → 
  craig_shells - (craig_shells / craig_ratio * brian_ratio) = 12 := by
sorry

end NUMINAMATH_CALUDE_seashell_difference_l1869_186989


namespace NUMINAMATH_CALUDE_square_root_problem_l1869_186973

theorem square_root_problem (x : ℝ) :
  (Real.sqrt 1.21) / (Real.sqrt x) + (Real.sqrt 1.44) / (Real.sqrt 0.49) = 3.0892857142857144 →
  x = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1869_186973


namespace NUMINAMATH_CALUDE_max_value_on_circle_l1869_186975

theorem max_value_on_circle (x y : ℝ) :
  x^2 + y^2 = 25 →
  ∃ (t_max : ℝ), t_max = 6 * Real.sqrt 10 ∧
  ∀ t, t = Real.sqrt (18 * y - 6 * x + 50) + Real.sqrt (8 * y + 6 * x + 50) →
  t ≤ t_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l1869_186975


namespace NUMINAMATH_CALUDE_robin_extra_gum_l1869_186995

/-- The number of extra pieces of gum Robin has -/
def extra_gum (packages : ℕ) (pieces_per_package : ℕ) (total_pieces : ℕ) : ℕ :=
  total_pieces - (packages * pieces_per_package)

/-- Theorem: Robin has 8 extra pieces of gum -/
theorem robin_extra_gum :
  extra_gum 43 23 997 = 8 := by
  sorry

end NUMINAMATH_CALUDE_robin_extra_gum_l1869_186995


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1869_186968

theorem condition_necessary_not_sufficient (x y : ℝ) :
  (x + y > 3 → (x > 1 ∨ y > 2)) ∧
  ¬((x > 1 ∨ y > 2) → x + y > 3) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1869_186968


namespace NUMINAMATH_CALUDE_zero_point_of_odd_function_l1869_186917

/-- A function f is odd if f(-x) = -f(x) for all x. -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem zero_point_of_odd_function (f : ℝ → ℝ) (x₀ : ℝ) :
  IsOdd f →
  f x₀ + Real.exp x₀ = 0 →
  Real.exp (-x₀) * f (-x₀) - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_point_of_odd_function_l1869_186917


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1869_186988

theorem solution_satisfies_system :
  let x : ℝ := -8/3
  let y : ℝ := -4/5
  (Real.sqrt (1 - 3*x) - 1 = Real.sqrt (5*y - 3*x)) ∧
  (Real.sqrt (5 - 5*y) + Real.sqrt (5*y - 3*x) = 5) := by
sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l1869_186988


namespace NUMINAMATH_CALUDE_floor_fraction_theorem_l1869_186916

theorem floor_fraction_theorem (d : ℝ) : 
  (∃ x : ℤ, x = ⌊d⌋ ∧ 3 * (x : ℝ)^2 + 10 * (x : ℝ) - 40 = 0) ∧ 
  (∃ y : ℝ, y = d - ⌊d⌋ ∧ 4 * y^2 - 20 * y + 19 = 0) →
  d = -9/2 := by
sorry

end NUMINAMATH_CALUDE_floor_fraction_theorem_l1869_186916


namespace NUMINAMATH_CALUDE_cube_sum_of_three_numbers_l1869_186930

theorem cube_sum_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_prod_eq : x * y + x * z + y * z = 1)
  (prod_eq : x * y * z = 1) :
  x^3 + y^3 + z^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_three_numbers_l1869_186930


namespace NUMINAMATH_CALUDE_total_edges_theorem_l1869_186971

/-- A graph with the properties described in the problem -/
structure WonderGraph where
  n : ℕ  -- number of cities
  a : ℕ  -- number of roads
  connected : Bool  -- graph is connected
  at_most_one_edge : Bool  -- at most one edge between any two vertices
  indirect_path : Bool  -- indirect path exists between directly connected vertices

/-- The number of subgraphs with even degree vertices -/
def num_even_subgraphs (G : WonderGraph) : ℕ := sorry

/-- The total number of edges in all subgraphs with even degree vertices -/
def total_edges_in_even_subgraphs (G : WonderGraph) : ℕ := sorry

/-- Main theorem: The total number of edges in all subgraphs with even degree vertices is ar/2 -/
theorem total_edges_theorem (G : WonderGraph) :
  total_edges_in_even_subgraphs G = G.a * (num_even_subgraphs G) / 2 :=
sorry

end NUMINAMATH_CALUDE_total_edges_theorem_l1869_186971


namespace NUMINAMATH_CALUDE_banana_sharing_l1869_186983

theorem banana_sharing (jefferson_bananas : ℕ) (walter_bananas : ℕ) : 
  jefferson_bananas = 56 →
  walter_bananas = jefferson_bananas - (jefferson_bananas / 4) →
  (jefferson_bananas + walter_bananas) / 2 = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_sharing_l1869_186983


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1869_186922

theorem quadratic_factorization (a b : ℤ) :
  (∀ x : ℝ, 25 * x^2 - 155 * x - 150 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -66 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1869_186922


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_problem_l1869_186931

theorem consecutive_odd_integers_problem (n : ℕ) : 
  n ≥ 3 ∧ n ≤ 9 ∧ n % 2 = 1 →
  (n - 2) + n + (n + 2) = ((n - 2) * n * (n + 2)) / 9 →
  n = 5 := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_problem_l1869_186931


namespace NUMINAMATH_CALUDE_peter_money_carried_l1869_186906

/-- The amount of money Peter carried to the market -/
def money_carried : ℝ := sorry

/-- The price of potatoes per kilo -/
def potato_price : ℝ := 2

/-- The quantity of potatoes bought in kilos -/
def potato_quantity : ℝ := 6

/-- The price of tomatoes per kilo -/
def tomato_price : ℝ := 3

/-- The quantity of tomatoes bought in kilos -/
def tomato_quantity : ℝ := 9

/-- The price of cucumbers per kilo -/
def cucumber_price : ℝ := 4

/-- The quantity of cucumbers bought in kilos -/
def cucumber_quantity : ℝ := 5

/-- The price of bananas per kilo -/
def banana_price : ℝ := 5

/-- The quantity of bananas bought in kilos -/
def banana_quantity : ℝ := 3

/-- The amount of money Peter has remaining after buying all items -/
def money_remaining : ℝ := 426

theorem peter_money_carried :
  money_carried = 
    potato_price * potato_quantity +
    tomato_price * tomato_quantity +
    cucumber_price * cucumber_quantity +
    banana_price * banana_quantity +
    money_remaining :=
by sorry

end NUMINAMATH_CALUDE_peter_money_carried_l1869_186906


namespace NUMINAMATH_CALUDE_option_b_more_favorable_example_option_b_more_favorable_l1869_186928

/-- Represents the financial data for a business --/
structure FinancialData where
  planned_revenue : ℕ
  advances_received : ℕ
  monthly_expenses : ℕ

/-- Calculates the tax payable under option (a) --/
def tax_option_a (data : FinancialData) : ℕ :=
  let total_income := data.planned_revenue + data.advances_received
  let tax := total_income * 6 / 100
  let insurance_contributions := data.monthly_expenses * 12
  let deduction := min (tax / 2) insurance_contributions
  tax - deduction

/-- Calculates the tax payable under option (b) --/
def tax_option_b (data : FinancialData) : ℕ :=
  let total_income := data.planned_revenue + data.advances_received
  let annual_expenses := data.monthly_expenses * 12
  let tax_base := max 0 (total_income - annual_expenses)
  let tax := max (total_income / 100) (tax_base * 15 / 100)
  tax

/-- Theorem stating that option (b) results in lower tax --/
theorem option_b_more_favorable (data : FinancialData) :
  tax_option_b data < tax_option_a data :=
by sorry

/-- Example financial data --/
def example_data : FinancialData :=
  { planned_revenue := 120000000
  , advances_received := 30000000
  , monthly_expenses := 11790000 }

/-- Proof that option (b) is more favorable for the example data --/
theorem example_option_b_more_favorable :
  tax_option_b example_data < tax_option_a example_data :=
by sorry

end NUMINAMATH_CALUDE_option_b_more_favorable_example_option_b_more_favorable_l1869_186928


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_three_l1869_186940

theorem sqrt_expression_equals_sqrt_three :
  Real.sqrt 48 - 6 * Real.sqrt (1/3) - Real.sqrt 18 / Real.sqrt 6 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_three_l1869_186940


namespace NUMINAMATH_CALUDE_combined_molecular_weight_mixture_l1869_186999

-- Define atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define molecular weights
def molecular_weight_CaO : ℝ := atomic_weight_Ca + atomic_weight_O
def molecular_weight_CO2 : ℝ := atomic_weight_C + 2 * atomic_weight_O
def molecular_weight_HNO3 : ℝ := atomic_weight_H + atomic_weight_N + 3 * atomic_weight_O

-- Define the mixture composition
def moles_CaO : ℝ := 5
def moles_CO2 : ℝ := 3
def moles_HNO3 : ℝ := 2

-- Theorem statement
theorem combined_molecular_weight_mixture :
  moles_CaO * molecular_weight_CaO +
  moles_CO2 * molecular_weight_CO2 +
  moles_HNO3 * molecular_weight_HNO3 = 538.45 := by
  sorry

end NUMINAMATH_CALUDE_combined_molecular_weight_mixture_l1869_186999
