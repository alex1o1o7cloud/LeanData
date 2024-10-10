import Mathlib

namespace sin_two_alpha_value_l662_66286

theorem sin_two_alpha_value (α : ℝ) 
  (h : (Real.cos (π - 2*α)) / (Real.sin (α - π/4)) = -Real.sqrt 2 / 2) : 
  Real.sin (2*α) = -3/4 := by
sorry

end sin_two_alpha_value_l662_66286


namespace scientific_notation_equality_coefficient_range_l662_66258

-- Define the number we want to express in scientific notation
def number : ℕ := 18480000

-- Define the components of the scientific notation
def coefficient : ℝ := 1.848
def exponent : ℕ := 7

-- Theorem to prove
theorem scientific_notation_equality :
  (coefficient * (10 : ℝ) ^ exponent : ℝ) = number := by
  sorry

-- Verify that the coefficient is between 1 and 10
theorem coefficient_range :
  1 < coefficient ∧ coefficient < 10 := by
  sorry

end scientific_notation_equality_coefficient_range_l662_66258


namespace students_behind_yoongi_count_l662_66251

/-- The number of students in the line. -/
def total_students : ℕ := 20

/-- Jungkook's position in the line. -/
def jungkook_position : ℕ := 3

/-- The number of students between Jungkook and Yoongi. -/
def students_between : ℕ := 5

/-- Yoongi's position in the line. -/
def yoongi_position : ℕ := jungkook_position + students_between + 1

/-- The number of students behind Yoongi. -/
def students_behind_yoongi : ℕ := total_students - yoongi_position

theorem students_behind_yoongi_count : students_behind_yoongi = 11 := by sorry

end students_behind_yoongi_count_l662_66251


namespace dan_baseball_cards_l662_66273

theorem dan_baseball_cards (initial_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 97) 
  (h2 : remaining_cards = 82) : 
  initial_cards - remaining_cards = 15 := by
  sorry

end dan_baseball_cards_l662_66273


namespace children_share_distribution_l662_66224

theorem children_share_distribution (total : ℝ) (share_ac : ℝ) 
  (h1 : total = 15800)
  (h2 : share_ac = 7022.222222222222) :
  total - share_ac = 8777.777777777778 := by
  sorry

end children_share_distribution_l662_66224


namespace boat_speed_in_still_water_l662_66223

/-- Given a boat that travels 8 km along a stream and 2 km against the stream in one hour,
    prove that its speed in still water is 5 km/hr. -/
theorem boat_speed_in_still_water :
  ∀ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = 8 →
    boat_speed - stream_speed = 2 →
    boat_speed = 5 := by
  sorry

end boat_speed_in_still_water_l662_66223


namespace trigonometric_problem_l662_66200

open Real

theorem trigonometric_problem (α β : Real)
  (h1 : α ∈ Set.Ioo 0 (π/2))
  (h2 : β ∈ Set.Ioo (π/2) π)
  (h3 : cos β = -1/3)
  (h4 : sin (α + β) = (4 - Real.sqrt 2) / 6) :
  tan (2 * β) = (4 * Real.sqrt 2) / 7 ∧ α = π / 4 := by
  sorry

end trigonometric_problem_l662_66200


namespace vector_scalar_product_l662_66228

/-- Given two vectors in R², prove that their scalar product equals 14 -/
theorem vector_scalar_product (a b : ℝ × ℝ) : 
  a = (2, 3) → b = (-1, 2) → (a + 2 • b) • b = 14 := by
  sorry

end vector_scalar_product_l662_66228


namespace corner_cut_pentagon_area_l662_66270

/-- A pentagon formed by cutting a triangular corner from a rectangular piece of paper -/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {13, 19, 20, 25, 31}

/-- The area of a CornerCutPentagon -/
def area (p : CornerCutPentagon) : ℕ :=
  745

theorem corner_cut_pentagon_area (p : CornerCutPentagon) : area p = 745 := by
  sorry

end corner_cut_pentagon_area_l662_66270


namespace sin_equals_cos_690_l662_66271

theorem sin_equals_cos_690 (n : ℤ) (h1 : -180 ≤ n ∧ n ≤ 180) 
  (h2 : Real.sin (n * π / 180) = Real.cos (690 * π / 180)) : n = 60 := by
  sorry

end sin_equals_cos_690_l662_66271


namespace range_of_a_given_decreasing_function_l662_66245

-- Define a decreasing function on the real line
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- State the theorem
theorem range_of_a_given_decreasing_function (f : ℝ → ℝ) (h : DecreasingFunction f) :
  ∀ a : ℝ, a ∈ Set.univ :=
by
  sorry

end range_of_a_given_decreasing_function_l662_66245


namespace least_whole_number_for_ratio_l662_66203

theorem least_whole_number_for_ratio (x : ℕ) : x = 3 ↔ 
  (x > 0 ∧ 
   ∀ y : ℕ, y > 0 → y < x → (6 - y : ℚ) / (7 - y) ≥ 16 / 21) ∧
  (6 - x : ℚ) / (7 - x) < 16 / 21 :=
by sorry

end least_whole_number_for_ratio_l662_66203


namespace prob_not_square_l662_66216

def total_figures : ℕ := 10
def num_triangles : ℕ := 5
def num_squares : ℕ := 3
def num_circles : ℕ := 2

theorem prob_not_square :
  (num_triangles + num_circles : ℚ) / total_figures = 7 / 10 :=
sorry

end prob_not_square_l662_66216


namespace second_year_interest_rate_problem_solution_l662_66260

/-- Given an initial investment, interest rates, and final value, calculate the second year's interest rate -/
theorem second_year_interest_rate 
  (initial_investment : ℝ) 
  (first_year_rate : ℝ) 
  (final_value : ℝ) : ℝ :=
  let first_year_value := initial_investment * (1 + first_year_rate)
  let second_year_rate := (final_value / first_year_value) - 1
  second_year_rate * 100

/-- Prove that the second year's interest rate is 4% given the problem conditions -/
theorem problem_solution :
  second_year_interest_rate 15000 0.05 16380 = 4 := by
  sorry

end second_year_interest_rate_problem_solution_l662_66260


namespace geometric_sequence_sum_l662_66278

theorem geometric_sequence_sum (u v : ℝ) : 
  (∃ (a r : ℝ), a ≠ 0 ∧ r ≠ 0 ∧
    u = a * r^3 ∧
    v = a * r^4 ∧
    4 = a * r^5 ∧
    1 = a * r^6) →
  u + v = 80 := by
sorry

end geometric_sequence_sum_l662_66278


namespace hyperbola_m_equation_l662_66244

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- The equation of a hyperbola in the form y²/a - x²/b = c -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / h.a - x^2 / h.b = h.c

/-- Two hyperbolas have common asymptotes if they have the same a/b ratio -/
def common_asymptotes (h1 h2 : Hyperbola) : Prop :=
  h1.a / h1.b = h2.a / h2.b

theorem hyperbola_m_equation 
  (n : Hyperbola)
  (hn_eq : hyperbola_equation n = fun x y ↦ y^2 / 4 - x^2 / 2 = 1)
  (m : Hyperbola)
  (hm_asymp : common_asymptotes m n)
  (hm_point : hyperbola_equation m (-2) 4) :
  hyperbola_equation m = fun x y ↦ y^2 / 8 - x^2 / 4 = 1 := by
  sorry

end hyperbola_m_equation_l662_66244


namespace smallest_right_triangle_area_l662_66267

theorem smallest_right_triangle_area :
  let side1 : ℝ := 6
  let side2 : ℝ := 8
  let area1 : ℝ := (1/2) * side1 * side2
  let area2 : ℝ := (1/2) * side1 * Real.sqrt (side2^2 - side1^2)
  min area1 area2 = 6 * Real.sqrt 7 := by
  sorry

end smallest_right_triangle_area_l662_66267


namespace binary_subtraction_l662_66276

def binary_to_decimal (b : ℕ) : ℕ := 
  if b = 0 then 0
  else if b % 10 = 1 then 1 + 2 * (binary_to_decimal (b / 10))
  else 2 * (binary_to_decimal (b / 10))

def binary_1111111111 : ℕ := 1111111111
def binary_11111 : ℕ := 11111

theorem binary_subtraction :
  binary_to_decimal binary_1111111111 - binary_to_decimal binary_11111 = 992 := by
  sorry

end binary_subtraction_l662_66276


namespace sum_one_implies_not_both_greater_than_one_l662_66204

theorem sum_one_implies_not_both_greater_than_one (a b : ℝ) :
  a + b = 1 → ¬(a > 1 ∧ b > 1) := by
sorry

end sum_one_implies_not_both_greater_than_one_l662_66204


namespace player_A_can_destroy_six_cups_six_cups_is_maximum_l662_66280

/-- Represents the state of the game with cups and pebbles -/
structure GameState where
  cups : ℕ
  pebbles : List ℕ

/-- Represents a move in the game -/
inductive Move
  | redistribute : List ℕ → Move
  | destroy_empty : Move
  | switch : ℕ → ℕ → Move

/-- Player A's strategy function -/
def player_A_strategy (state : GameState) : List ℕ :=
  sorry

/-- Player B's action function -/
def player_B_action (state : GameState) (move : Move) : GameState :=
  sorry

/-- Simulates the game for a given number of moves -/
def play_game (initial_state : GameState) (num_moves : ℕ) : GameState :=
  sorry

/-- Theorem stating that player A can guarantee at least 6 cups are destroyed -/
theorem player_A_can_destroy_six_cups :
  ∃ (strategy : GameState → List ℕ),
    ∀ (num_moves : ℕ),
      let final_state := play_game {cups := 10, pebbles := List.replicate 10 10} num_moves
      final_state.cups ≤ 4 :=
sorry

/-- Theorem stating that 6 is the maximum number of cups that can be guaranteed to be destroyed -/
theorem six_cups_is_maximum :
  ∀ (strategy : GameState → List ℕ),
    ∃ (num_moves : ℕ),
      let final_state := play_game {cups := 10, pebbles := List.replicate 10 10} num_moves
      final_state.cups > 4 :=
sorry

end player_A_can_destroy_six_cups_six_cups_is_maximum_l662_66280


namespace simplify_power_l662_66227

theorem simplify_power (y : ℝ) : (3 * y^4)^4 = 81 * y^16 := by sorry

end simplify_power_l662_66227


namespace sqrt_expression_defined_l662_66246

theorem sqrt_expression_defined (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x^2 - 2 * (a - 1) * x + 3 * a - 3 ≥ 0) ↔ a ≥ 1 := by
sorry

end sqrt_expression_defined_l662_66246


namespace expression_simplification_l662_66240

theorem expression_simplification (x : ℝ) (h : x = -2) :
  (1 - 2 / (x + 1)) / ((x^2 - x) / (x^2 - 1)) = 3 / 2 := by
  sorry

end expression_simplification_l662_66240


namespace inverse_negation_correct_l662_66233

/-- Represents a triangle ABC -/
structure Triangle where
  isIsosceles : Bool
  hasEqualAngles : Bool

/-- The original proposition -/
def originalProposition (t : Triangle) : Prop :=
  ¬t.isIsosceles → ¬t.hasEqualAngles

/-- The inverse negation of the original proposition -/
def inverseNegation (t : Triangle) : Prop :=
  t.hasEqualAngles → t.isIsosceles

/-- Theorem stating that the inverse negation is correct -/
theorem inverse_negation_correct :
  ∀ t : Triangle, inverseNegation t ↔ ¬(¬originalProposition t) :=
sorry

end inverse_negation_correct_l662_66233


namespace equation_has_two_solutions_l662_66221

-- Define the equation
def equation (x : ℝ) : Prop := Real.sqrt (9 - x) = x * Real.sqrt (9 - x)

-- Theorem statement
theorem equation_has_two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ equation a ∧ equation b ∧ 
  ∀ (x : ℝ), equation x → (x = a ∨ x = b) :=
sorry

end equation_has_two_solutions_l662_66221


namespace relay_race_selection_methods_l662_66291

/-- The number of students good at sprinting -/
def total_students : ℕ := 6

/-- The number of students to be selected for the relay race -/
def selected_students : ℕ := 4

/-- The number of possible positions for A and B (they must be consecutive with A before B) -/
def positions_for_AB : ℕ := 3

/-- The number of remaining students to be selected -/
def remaining_students : ℕ := total_students - 2

/-- The number of positions to be filled by the remaining students -/
def positions_to_fill : ℕ := selected_students - 2

theorem relay_race_selection_methods :
  (positions_for_AB * (remaining_students.factorial / (remaining_students - positions_to_fill).factorial)) = 36 := by
  sorry

end relay_race_selection_methods_l662_66291


namespace carol_peanuts_count_l662_66229

/-- The number of peanuts Carol initially collects -/
def initial_peanuts : ℕ := 2

/-- The number of peanuts Carol's father gives her -/
def given_peanuts : ℕ := 5

/-- The total number of peanuts Carol has -/
def total_peanuts : ℕ := initial_peanuts + given_peanuts

theorem carol_peanuts_count : total_peanuts = 7 := by
  sorry

end carol_peanuts_count_l662_66229


namespace circle_tangent_perpendicular_l662_66218

-- Define a structure for a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate the angle between three points
def angle (p1 p2 p3 : Point) : ℝ := sorry

-- Define a predicate to check if three points are collinear
def collinear (p1 p2 p3 : Point) : Prop := sorry

theorem circle_tangent_perpendicular (A B C : Point) :
  ¬collinear A B C →
  ∃ (α β γ : ℝ),
    (β + γ + angle B A C = π / 2 ∨ β + γ + angle B A C = -π / 2) ∧
    (γ + α + angle A B C = π / 2 ∨ γ + α + angle A B C = -π / 2) ∧
    (α + β + angle A C B = π / 2 ∨ α + β + angle A C B = -π / 2) :=
sorry

end circle_tangent_perpendicular_l662_66218


namespace custom_operation_result_l662_66261

-- Define the custom operation *
def star (a b : ℕ) : ℕ := a + 2 * b

-- State the theorem
theorem custom_operation_result : star (star 2 4) 6 = 22 := by
  sorry

end custom_operation_result_l662_66261


namespace inequality_system_solution_l662_66274

theorem inequality_system_solution (k : ℝ) : 
  (∀ x : ℝ, (2 * x + 9 > 6 * x + 1 ∧ x - k < 1) ↔ x < 2) →
  k ≥ 1 :=
by sorry

end inequality_system_solution_l662_66274


namespace problem_1_problem_2_problem_3_problem_4_l662_66297

-- 1. 0.175÷0.25÷4 = 0.175
theorem problem_1 : (0.175 / 0.25) / 4 = 0.175 := by sorry

-- 2. 1.4×99+1.4 = 140
theorem problem_2 : 1.4 * 99 + 1.4 = 140 := by sorry

-- 3. 3.6÷4-1.2×6 = -6.3
theorem problem_3 : 3.6 / 4 - 1.2 * 6 = -6.3 := by sorry

-- 4. (3.2+0.16)÷0.8 = 4.2
theorem problem_4 : (3.2 + 0.16) / 0.8 = 4.2 := by sorry

end problem_1_problem_2_problem_3_problem_4_l662_66297


namespace inequality_system_solution_l662_66282

theorem inequality_system_solution (x : ℝ) : 
  ((x + 3) / 2 ≤ x + 2 ∧ 2 * (x + 4) > 4 * x + 2) ↔ (-1 ≤ x ∧ x < 3) :=
by sorry

end inequality_system_solution_l662_66282


namespace juan_distance_l662_66230

/-- Given a speed and time, calculate the distance traveled. -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Juan's distance traveled is 80 miles. -/
theorem juan_distance : distance 10 8 = 80 := by
  sorry

end juan_distance_l662_66230


namespace ab_length_not_unique_l662_66208

/-- Given two line segments AC and BC with lengths 1 and 3 respectively,
    the length of AB cannot be uniquely determined. -/
theorem ab_length_not_unique (AC BC : ℝ) (hAC : AC = 1) (hBC : BC = 3) :
  ¬ ∃! AB : ℝ, (0 < AB ∧ AB < AC + BC) ∨ (AB = AC + BC ∨ AB = |BC - AC|) :=
sorry

end ab_length_not_unique_l662_66208


namespace solve_system_l662_66294

-- Define the variables x and y
variable (x y : ℤ)

-- State the theorem
theorem solve_system : 
  (3:ℝ)^x = 27^(y+1) → (16:ℝ)^y = 2^(x-8) → 2*x + y = -29 := by
  sorry

end solve_system_l662_66294


namespace binomial_expansion_example_l662_66293

theorem binomial_expansion_example : (7 + 2)^3 = 729 := by
  sorry

end binomial_expansion_example_l662_66293


namespace parallel_line_through_point_l662_66298

/-- A line in 2D space represented by the equation f(x,y) = 0 -/
structure Line2D where
  f : ℝ → ℝ → ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The equation f(x,y) - f(x₁,y₁) - f(x₂,y₂) = 0 represents a line parallel to 
    the original line and passing through P₂ -/
theorem parallel_line_through_point (l : Line2D) (P₁ P₂ : Point2D) 
  (h₁ : l.f P₁.x P₁.y = 0)  -- P₁ is on the line l
  (h₂ : l.f P₂.x P₂.y ≠ 0)  -- P₂ is not on the line l
  : ∃ (m : Line2D), 
    (∀ x y, m.f x y = l.f x y - l.f P₁.x P₁.y - l.f P₂.x P₂.y) ∧ 
    (m.f P₂.x P₂.y = 0) ∧
    (∃ k : ℝ, ∀ x y, m.f x y = k * l.f x y) := by
  sorry

end parallel_line_through_point_l662_66298


namespace charity_race_fundraising_l662_66234

/-- Proves that the amount raised by each of the ten students is $20 -/
theorem charity_race_fundraising
  (total_students : ℕ)
  (special_students : ℕ)
  (regular_amount : ℕ)
  (total_raised : ℕ)
  (h1 : total_students = 30)
  (h2 : special_students = 10)
  (h3 : regular_amount = 30)
  (h4 : total_raised = 800)
  (h5 : total_raised = special_students * X + (total_students - special_students) * regular_amount)
  : X = 20 := by
  sorry

end charity_race_fundraising_l662_66234


namespace solution_product_l662_66217

theorem solution_product (p q : ℝ) : 
  (p - 3) * (3 * p + 18) = p^2 - 15 * p + 54 →
  (q - 3) * (3 * q + 18) = q^2 - 15 * q + 54 →
  p ≠ q →
  (p + 2) * (q + 2) = -80 := by
sorry

end solution_product_l662_66217


namespace arithmetic_equality_l662_66202

theorem arithmetic_equality : 5 * 7 + 9 * 4 - 36 / 3 = 59 := by
  sorry

end arithmetic_equality_l662_66202


namespace ball_bounce_distance_l662_66287

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (numBounces : ℕ) : ℝ :=
  let descentDistances := List.range (numBounces + 1) |>.map (fun n => initialHeight * bounceRatio^n)
  let ascentDistances := List.range numBounces |>.map (fun n => initialHeight * bounceRatio^(n+1))
  (descentDistances.sum + ascentDistances.sum)

/-- The problem statement -/
theorem ball_bounce_distance :
  let initialHeight : ℝ := 20
  let bounceRatio : ℝ := 2/3
  let numBounces : ℕ := 4
  abs (totalDistance initialHeight bounceRatio numBounces - 80) < 1 := by
  sorry

end ball_bounce_distance_l662_66287


namespace diagonal_length_of_quadrilateral_l662_66220

/-- The length of a diagonal in a quadrilateral with given area and offsets -/
theorem diagonal_length_of_quadrilateral (area : ℝ) (offset1 offset2 : ℝ) :
  area = 210 →
  offset1 = 9 →
  offset2 = 6 →
  (∃ d : ℝ, area = 0.5 * d * (offset1 + offset2) ∧ d = 28) :=
by sorry

end diagonal_length_of_quadrilateral_l662_66220


namespace machine_production_l662_66205

/-- The number of shirts produced by a machine in a given time -/
def shirts_produced (shirts_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  shirts_per_minute * minutes

/-- Theorem: A machine that produces 6 shirts per minute, operating for 12 minutes, will produce 72 shirts -/
theorem machine_production :
  shirts_produced 6 12 = 72 := by
  sorry

end machine_production_l662_66205


namespace divisor_and_expression_l662_66269

theorem divisor_and_expression (k : ℕ) : 
  (30^k : ℕ) ∣ 929260 → 3^k - k^3 = 2 := by
  sorry

end divisor_and_expression_l662_66269


namespace painters_work_days_l662_66289

/-- Represents the time taken to complete a job given a number of painters -/
def time_to_complete (num_painters : ℕ) (work_days : ℚ) : ℚ := num_painters * work_days

/-- Proves that if 6 painters can finish a job in 2 work-days, 
    then 4 painters will take 3 work-days to finish the same job -/
theorem painters_work_days (initial_painters : ℕ) (initial_days : ℚ) 
  (new_painters : ℕ) : 
  initial_painters = 6 → initial_days = 2 → new_painters = 4 →
  time_to_complete new_painters (3 : ℚ) = time_to_complete initial_painters initial_days :=
by
  sorry

end painters_work_days_l662_66289


namespace linear_system_solution_l662_66279

theorem linear_system_solution (x y : ℚ) 
  (eq1 : 3 * x - y = 9) 
  (eq2 : 2 * y - x = 1) : 
  5 * x + 4 * y = 39 := by
sorry

end linear_system_solution_l662_66279


namespace snow_clearing_volume_l662_66299

/-- The volume of snow on a rectangular pathway -/
def snow_volume (length width depth : ℚ) : ℚ :=
  length * width * depth

/-- Proof that the volume of snow on the given pathway is 67.5 cubic feet -/
theorem snow_clearing_volume :
  let length : ℚ := 30
  let width : ℚ := 3
  let depth : ℚ := 3/4
  snow_volume length width depth = 67.5 := by
sorry

end snow_clearing_volume_l662_66299


namespace adam_has_more_apples_l662_66296

/-- The number of apples Adam has -/
def adam_apples : ℕ := 10

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 2

/-- The number of apples Michael has -/
def michael_apples : ℕ := 5

/-- Theorem: Adam has 3 more apples than the combined total of Jackie's and Michael's apples -/
theorem adam_has_more_apples : adam_apples - (jackie_apples + michael_apples) = 3 := by
  sorry


end adam_has_more_apples_l662_66296


namespace inequality_proof_l662_66281

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : b > 1) : 
  (a^2 / (b - 1)) + (b^2 / (a - 1)) ≥ 8 := by
  sorry

end inequality_proof_l662_66281


namespace no_positive_integers_satisfying_conditions_l662_66275

theorem no_positive_integers_satisfying_conditions : ¬∃ (a b c d : ℕ+) (p : ℕ),
  (a.val * b.val = c.val * d.val) ∧ 
  (a.val + b.val + c.val + d.val = p) ∧
  Nat.Prime p :=
sorry

end no_positive_integers_satisfying_conditions_l662_66275


namespace arithmetic_sequence_cosine_l662_66239

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 8 + a 15 = Real.pi →
  Real.cos (a 4 + a 12) = -1/2 := by
sorry

end arithmetic_sequence_cosine_l662_66239


namespace polynomial_value_at_two_l662_66265

def f (x : ℝ) : ℝ := 4*x^5 + 2*x^4 + 3*x^3 - 2*x^2 - 2500*x + 434

theorem polynomial_value_at_two :
  f 2 = -3390 :=
by sorry

end polynomial_value_at_two_l662_66265


namespace linear_equation_solutions_l662_66219

theorem linear_equation_solutions (x y : ℝ) : 
  (x = 1 ∧ y = 2 → 2*x + y = 4) ∧
  (x = 2 ∧ y = 0 → 2*x + y = 4) ∧
  (x = 0.5 ∧ y = 3 → 2*x + y = 4) ∧
  (x = -2 ∧ y = 4 → 2*x + y ≠ 4) := by
  sorry

#check linear_equation_solutions

end linear_equation_solutions_l662_66219


namespace square_of_sum_of_squares_is_sum_of_squares_l662_66288

def is_sum_of_two_squares (x : ℕ) : Prop :=
  ∃ (a b : ℕ), x = a^2 + b^2 ∧ a > 0 ∧ b > 0

theorem square_of_sum_of_squares_is_sum_of_squares (n : ℕ) :
  (is_sum_of_two_squares (n - 1) ∧ 
   is_sum_of_two_squares n ∧ 
   is_sum_of_two_squares (n + 1)) →
  (is_sum_of_two_squares (n^2 - 1) ∧ 
   is_sum_of_two_squares (n^2) ∧ 
   is_sum_of_two_squares (n^2 + 1)) :=
by sorry

end square_of_sum_of_squares_is_sum_of_squares_l662_66288


namespace square_area_error_l662_66249

theorem square_area_error (s : ℝ) (s' : ℝ) (h : s' = 1.04 * s) :
  (s' ^ 2 - s ^ 2) / s ^ 2 = 0.0816 := by
  sorry

end square_area_error_l662_66249


namespace box_cube_volume_l662_66237

/-- Given a box with dimensions 10 cm x 18 cm x 4 cm, filled completely with 60 identical cubes,
    the volume of each cube is 8 cubic centimeters. -/
theorem box_cube_volume (length width height : ℕ) (num_cubes : ℕ) (cube_volume : ℕ) :
  length = 10 ∧ width = 18 ∧ height = 4 ∧ num_cubes = 60 →
  length * width * height = num_cubes * cube_volume →
  cube_volume = 8 := by
  sorry

#check box_cube_volume

end box_cube_volume_l662_66237


namespace ellipse_area_condition_l662_66285

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    if the area of the right triangle formed by a point on the ellipse,
    the center, and the right focus is √3, then a² = 2√3 + 4 and b² = 2√3 -/
theorem ellipse_area_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let c := Real.sqrt (a^2 - b^2)
  let triangle_area (x y : ℝ) := (1/2) * (c/2) * y
  ∃ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ∧ triangle_area x y = Real.sqrt 3 →
  a^2 = 2 * Real.sqrt 3 + 4 ∧ b^2 = 2 * Real.sqrt 3 := by
  sorry

end ellipse_area_condition_l662_66285


namespace smallest_n_proof_l662_66209

/-- The capacity of adults on a single bench section -/
def adult_capacity : ℕ := 8

/-- The capacity of children on a single bench section -/
def child_capacity : ℕ := 12

/-- Predicate to check if a number of bench sections can seat an equal number of adults and children -/
def can_seat_equally (n : ℕ) : Prop :=
  ∃ (x : ℕ), x > 0 ∧ adult_capacity * n = x ∧ child_capacity * n = x

/-- The smallest positive integer number of bench sections that can seat an equal number of adults and children -/
def smallest_n : ℕ := 3

theorem smallest_n_proof :
  (can_seat_equally smallest_n) ∧
  (∀ m : ℕ, m > 0 ∧ m < smallest_n → ¬(can_seat_equally m)) :=
sorry

end smallest_n_proof_l662_66209


namespace expression_equals_36_l662_66235

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end expression_equals_36_l662_66235


namespace complement_intersection_theorem_l662_66277

def U : Set Nat := {0,1,2,3,4,5,6,7,8,9}
def A : Set Nat := {0,1,3,5,8}
def B : Set Nat := {2,4,5,6,8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7,9} := by sorry

end complement_intersection_theorem_l662_66277


namespace perfect_square_product_l662_66248

theorem perfect_square_product (a b c d : ℤ) (h : a + b + c + d = 0) :
  ∃ k : ℤ, (a * b - c * d) * (b * c - a * d) * (c * a - b * d) = k ^ 2 := by
  sorry

end perfect_square_product_l662_66248


namespace triangle_abc_area_l662_66268

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 3 under the given conditions. -/
theorem triangle_abc_area (a b c : ℝ) (A B C : ℝ) : 
  a = Real.sqrt 5 →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  (1/2) * a * c * Real.sin B = 3 := by
  sorry

end triangle_abc_area_l662_66268


namespace circle_area_ratio_l662_66272

/-- Given two circles X and Y, if an arc of 60° on circle X has the same length as an arc of 40° on circle Y, 
    then the ratio of the area of circle X to the area of circle Y is 9/4. -/
theorem circle_area_ratio (X Y : ℝ → ℝ → Prop) (R_X R_Y : ℝ) :
  (∃ L : ℝ, L = (60 / 360) * (2 * Real.pi * R_X) ∧ L = (40 / 360) * (2 * Real.pi * R_Y)) →
  (R_X > 0 ∧ R_Y > 0) →
  (X = λ x y => (x - 0)^2 + (y - 0)^2 = R_X^2) →
  (Y = λ x y => (x - 0)^2 + (y - 0)^2 = R_Y^2) →
  (Real.pi * R_X^2) / (Real.pi * R_Y^2) = 9/4 := by
  sorry

end circle_area_ratio_l662_66272


namespace mike_dogs_count_l662_66207

/-- Represents the number of dogs Mike has -/
def number_of_dogs : ℕ := 2

/-- Weight of a cup of dog food in pounds -/
def cup_weight : ℚ := 1/4

/-- Number of cups each dog eats per feeding -/
def cups_per_feeding : ℕ := 6

/-- Number of feedings per day -/
def feedings_per_day : ℕ := 2

/-- Number of bags of dog food Mike buys per month -/
def bags_per_month : ℕ := 9

/-- Weight of each bag of dog food in pounds -/
def bag_weight : ℕ := 20

/-- Number of days in a month -/
def days_per_month : ℕ := 30

theorem mike_dogs_count :
  number_of_dogs = 
    (bags_per_month * bag_weight) / 
    (cups_per_feeding * feedings_per_day * cup_weight * days_per_month) := by
  sorry

end mike_dogs_count_l662_66207


namespace extremum_condition_l662_66225

/-- A function f: ℝ → ℝ has an extremum at x₀ if f(x₀) is either a maximum or minimum value of f. -/
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ ∀ x, f x ≥ f x₀

/-- Theorem: For a differentiable function f: ℝ → ℝ, f'(x₀) = 0 is a necessary but not sufficient
    condition for f(x₀) to be an extremum of f(x). -/
theorem extremum_condition (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x₀ : ℝ, HasExtremumAt f x₀ → deriv f x₀ = 0) ∧
  ¬(∀ x₀ : ℝ, deriv f x₀ = 0 → HasExtremumAt f x₀) :=
sorry

end extremum_condition_l662_66225


namespace square_ceiling_lights_l662_66243

/-- The number of lights on each side of the square ceiling -/
def lights_per_side : ℕ := 20

/-- The minimum number of lights needed for the entire square ceiling -/
def min_lights_needed : ℕ := 4 * lights_per_side - 4

theorem square_ceiling_lights : min_lights_needed = 76 := by
  sorry

end square_ceiling_lights_l662_66243


namespace morning_campers_count_l662_66211

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 39

/-- The additional number of campers who went rowing in the morning compared to the afternoon -/
def additional_morning_campers : ℕ := 5

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := afternoon_campers + additional_morning_campers

theorem morning_campers_count : morning_campers = 44 := by
  sorry

end morning_campers_count_l662_66211


namespace painting_supplies_theorem_l662_66266

/-- Represents the cost and quantity of painting supplies -/
structure PaintingSupplies where
  brush_cost : ℝ
  board_cost : ℝ
  total_items : ℕ
  max_cost : ℝ

/-- Theorem stating the properties of the painting supplies purchase -/
theorem painting_supplies_theorem (ps : PaintingSupplies) 
  (h1 : 340 / ps.brush_cost = 300 / ps.board_cost)
  (h2 : ps.brush_cost = ps.board_cost + 2)
  (h3 : ps.total_items = 30)
  (h4 : ∀ a : ℕ, a ≤ ps.total_items → 
    ps.brush_cost * (ps.total_items - a) + ps.board_cost * a ≤ ps.max_cost) :
  ps.brush_cost = 17 ∧ ps.board_cost = 15 ∧ 
  (∃ min_boards : ℕ, min_boards = 18 ∧ 
    ∀ a : ℕ, a < min_boards → 
      ps.brush_cost * (ps.total_items - a) + ps.board_cost * a > ps.max_cost) := by
  sorry

#check painting_supplies_theorem

end painting_supplies_theorem_l662_66266


namespace smallest_n_for_factorization_l662_66253

theorem smallest_n_for_factorization : 
  let can_be_factored (n : ℤ) := ∃ (A B : ℤ), 
    (A * B = 60) ∧ 
    (6 * B + A = n) ∧ 
    (∀ x, 6 * x^2 + n * x + 60 = (6 * x + A) * (x + B))
  ∀ n : ℤ, can_be_factored n → n ≥ 66
  ∧ can_be_factored 66 :=
by sorry

end smallest_n_for_factorization_l662_66253


namespace range_of_m_l662_66262

-- Define propositions P and Q as functions of m
def P (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the theorem
theorem range_of_m : 
  (∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m)) → 
  (∀ m : ℝ, (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) ↔ (P m ∨ Q m) ∧ ¬(P m ∧ Q m)) :=
sorry

end range_of_m_l662_66262


namespace symmetry_of_curves_l662_66214

-- Define the original curve
def original_curve (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

-- Define the point of symmetry
def point_of_symmetry : ℝ × ℝ := (3, 5)

-- Define the symmetric curve
def symmetric_curve (x y : ℝ) : Prop := (x - 6)^2 + 4*(y - 10)^2 = 4

-- Theorem statement
theorem symmetry_of_curves :
  ∀ (x y : ℝ), original_curve x y ↔ symmetric_curve (2*point_of_symmetry.1 - x) (2*point_of_symmetry.2 - y) :=
by sorry

end symmetry_of_curves_l662_66214


namespace systematic_sampling_first_stage_l662_66232

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a stage in the sampling process -/
inductive SamplingStage
  | First
  | Later

/-- Defines the relationship between sampling methods and stages -/
def sampling_relationship (method : SamplingMethod) (stage : SamplingStage) : Prop :=
  match method, stage with
  | SamplingMethod.Systematic, SamplingStage.First => true
  | _, _ => false

/-- Theorem stating that systematic sampling generally uses simple random sampling in the first stage -/
theorem systematic_sampling_first_stage :
  sampling_relationship SamplingMethod.Systematic SamplingStage.First = true :=
by
  sorry

#check systematic_sampling_first_stage

end systematic_sampling_first_stage_l662_66232


namespace base_conversion_problem_l662_66263

theorem base_conversion_problem (n C D : ℕ) : 
  n > 0 ∧ 
  C < 8 ∧ 
  D < 6 ∧ 
  n = 8 * C + D ∧ 
  n = 6 * D + C → 
  n = 43 := by sorry

end base_conversion_problem_l662_66263


namespace pyramid_volume_l662_66241

/-- 
Given a pyramid with a rhombic base:
- d₁ and d₂ are the diagonals of the rhombus base
- d₁ > d₂
- The height of the pyramid passes through the vertex of the acute angle of the rhombus
- Q is the area of the diagonal section conducted through the shorter diagonal

This theorem states that the volume of such a pyramid is (d₁ / 12) * √(16Q² - d₁²d₂²)
-/
theorem pyramid_volume (d₁ d₂ Q : ℝ) (h₁ : d₁ > d₂) (h₂ : d₁ > 0) (h₃ : d₂ > 0) (h₄ : Q > 0) :
  let volume := d₁ / 12 * Real.sqrt (16 * Q^2 - d₁^2 * d₂^2)
  volume > 0 ∧ volume^3 = (d₁^3 / 1728) * (16 * Q^2 - d₁^2 * d₂^2) := by
  sorry

end pyramid_volume_l662_66241


namespace newberg_airport_passengers_l662_66236

/-- The number of passengers who landed on time in Newberg last year -/
def on_time_passengers : ℕ := 14507

/-- The number of passengers who landed late in Newberg last year -/
def late_passengers : ℕ := 213

/-- The number of passengers who had connecting flights in Newberg last year -/
def connecting_passengers : ℕ := 320

/-- The total number of passengers who landed in Newberg last year -/
def total_passengers : ℕ := on_time_passengers + late_passengers + connecting_passengers

theorem newberg_airport_passengers :
  total_passengers = 15040 :=
sorry

end newberg_airport_passengers_l662_66236


namespace division_multiplication_equality_l662_66201

theorem division_multiplication_equality : (-150) / (-50) * (1/3 : ℚ) = 1 := by sorry

end division_multiplication_equality_l662_66201


namespace time_after_2051_hours_l662_66231

/-- Calculates the time on a 12-hour clock after a given number of hours have passed -/
def timeAfter (startTime : Nat) (hoursPassed : Nat) : Nat :=
  (startTime + hoursPassed) % 12

/-- Proves that 2051 hours after 9 o'clock, it will be 8 o'clock on a 12-hour clock -/
theorem time_after_2051_hours :
  timeAfter 9 2051 = 8 := by
  sorry

#eval timeAfter 9 2051  -- This should output 8

end time_after_2051_hours_l662_66231


namespace solve_equation_l662_66222

theorem solve_equation (m n : ℝ) : 
  |m - 2| + n^2 - 8*n + 16 = 0 → m = 2 ∧ n = 4 := by
  sorry

end solve_equation_l662_66222


namespace min_value_tangent_l662_66259

/-- Given a function f(x) = 2cos(x) - 3sin(x) that reaches its minimum value when x = θ,
    prove that tan(θ) = -3/2 --/
theorem min_value_tangent (θ : ℝ) (h : ∀ x, 2 * Real.cos x - 3 * Real.sin x ≥ 2 * Real.cos θ - 3 * Real.sin θ) :
  Real.tan θ = -3/2 := by
  sorry

end min_value_tangent_l662_66259


namespace four_students_three_communities_l662_66256

/-- The number of ways to assign students to communities -/
def assignStudents (num_students : ℕ) (num_communities : ℕ) : ℕ :=
  num_communities ^ num_students

/-- Theorem stating that assigning 4 students to 3 communities results in 3^4 arrangements -/
theorem four_students_three_communities :
  assignStudents 4 3 = 3^4 := by
  sorry

end four_students_three_communities_l662_66256


namespace natasha_dimes_problem_l662_66292

theorem natasha_dimes_problem :
  ∃! n : ℕ, 100 < n ∧ n < 200 ∧
    n % 3 = 2 ∧
    n % 4 = 2 ∧
    n % 5 = 2 ∧
    n % 7 = 2 ∧
    n = 182 := by
  sorry

end natasha_dimes_problem_l662_66292


namespace sword_length_proof_l662_66247

/-- The length of Christopher's sword in inches -/
def christopher_sword_length : ℕ := 15

/-- The length of Jameson's sword in inches -/
def jameson_sword_length : ℕ := 2 * christopher_sword_length + 3

/-- The length of June's sword in inches -/
def june_sword_length : ℕ := jameson_sword_length + 5

theorem sword_length_proof :
  (jameson_sword_length = 2 * christopher_sword_length + 3) ∧
  (june_sword_length = jameson_sword_length + 5) ∧
  (june_sword_length = christopher_sword_length + 23) →
  christopher_sword_length = 15 := by
sorry

#eval christopher_sword_length

end sword_length_proof_l662_66247


namespace equiangular_polygons_unique_angle_l662_66255

theorem equiangular_polygons_unique_angle : ∃! x : ℝ,
  0 < x ∧ x < 180 ∧
  ∃ n₁ : ℕ, n₁ ≥ 3 ∧ x = 180 - 360 / n₁ ∧
  ∃ n₃ : ℕ, n₃ ≥ 3 ∧ 3/2 * x = 180 - 360 / n₃ ∧
  n₁ ≠ n₃ := by sorry

end equiangular_polygons_unique_angle_l662_66255


namespace max_self_intersection_points_seven_segments_l662_66238

/-- A closed polyline is a sequence of connected line segments that form a closed loop. -/
def ClosedPolyline (n : ℕ) := Fin n → ℝ × ℝ

/-- The number of self-intersection points in a closed polyline. -/
def selfIntersectionPoints (p : ClosedPolyline 7) : ℕ := sorry

/-- The maximum number of self-intersection points in any closed polyline with 7 segments. -/
def maxSelfIntersectionPoints : ℕ := sorry

/-- Theorem: The maximum number of self-intersection points in a closed polyline with 7 segments is 14. -/
theorem max_self_intersection_points_seven_segments :
  maxSelfIntersectionPoints = 14 := by sorry

end max_self_intersection_points_seven_segments_l662_66238


namespace minimum_value_of_function_minimum_value_achieved_l662_66210

theorem minimum_value_of_function (x : ℝ) (h : x > 1) :
  2 * x + 2 / (x - 1) ≥ 6 :=
sorry

theorem minimum_value_achieved (x : ℝ) (h : x > 1) :
  2 * x + 2 / (x - 1) = 6 ↔ x = 2 :=
sorry

end minimum_value_of_function_minimum_value_achieved_l662_66210


namespace sqrt_x_plus_one_real_range_l662_66213

theorem sqrt_x_plus_one_real_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
  sorry

end sqrt_x_plus_one_real_range_l662_66213


namespace toy_price_reduction_l662_66206

theorem toy_price_reduction :
  ∃! x : ℕ, 1 ≤ x ∧ x ≤ 12 ∧
  (∃ y : ℕ, 1 ≤ y ∧ y ≤ 100 ∧ (13 - x) * y = 781) ∧
  (∀ z : ℕ, z > x → ¬∃ y : ℕ, 1 ≤ y ∧ y ≤ 100 ∧ (13 - z) * y = 781) :=
by sorry

end toy_price_reduction_l662_66206


namespace transverse_axis_length_l662_66284

-- Define the hyperbola M
def hyperbola_M (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the parabola N
def parabola_N (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of parabola N
def focus_N : ℝ × ℝ := (2, 0)

-- Define the condition that the right focus of M is the focus of N
def right_focus_condition (a b : ℝ) : Prop := (a, 0) = focus_N

-- Define the intersection points P and Q
def intersection_points (P Q : ℝ × ℝ) (a b : ℝ) : Prop :=
  hyperbola_M a b P.1 P.2 ∧ parabola_N P.1 P.2 ∧
  hyperbola_M a b Q.1 Q.2 ∧ parabola_N Q.1 Q.2

-- Define the condition that PF = FQ
def PF_equals_FQ (P Q : ℝ × ℝ) : Prop :=
  (P.1 - focus_N.1)^2 + (P.2 - focus_N.2)^2 =
  (Q.1 - focus_N.1)^2 + (Q.2 - focus_N.2)^2

-- Main theorem
theorem transverse_axis_length (a b : ℝ) (P Q : ℝ × ℝ) :
  hyperbola_M a b a 0 →
  right_focus_condition a b →
  intersection_points P Q a b →
  PF_equals_FQ P Q →
  2 * a = 4 * Real.sqrt 2 - 4 :=
sorry

end transverse_axis_length_l662_66284


namespace cabbage_sales_proof_l662_66250

def price_per_kg : ℝ := 2
def earnings_wednesday : ℝ := 30
def earnings_friday : ℝ := 24
def earnings_today : ℝ := 42

theorem cabbage_sales_proof :
  (earnings_wednesday + earnings_friday + earnings_today) / price_per_kg = 48 := by
  sorry

end cabbage_sales_proof_l662_66250


namespace plane_distance_l662_66212

/-- Given a plane flying east at 300 km/h and west at 400 km/h for a total of 7 hours,
    the distance traveled from the airport is 1200 km. -/
theorem plane_distance (speed_east speed_west total_time : ℝ) 
    (h1 : speed_east = 300)
    (h2 : speed_west = 400)
    (h3 : total_time = 7) : 
  (total_time * speed_east * speed_west) / (speed_east + speed_west) = 1200 := by
  sorry

end plane_distance_l662_66212


namespace paperclip_capacity_l662_66295

/-- Given that a box of volume 18 cm³ can hold 60 paperclips, and the storage density
    decreases by 10% in larger boxes, prove that a box of volume 72 cm³ can hold 216 paperclips. -/
theorem paperclip_capacity (small_volume small_capacity large_volume : ℝ) 
    (h1 : small_volume = 18)
    (h2 : small_capacity = 60)
    (h3 : large_volume = 72)
    (h4 : large_volume > small_volume) :
    let density_ratio := large_volume / small_volume
    let unadjusted_capacity := small_capacity * density_ratio
    let adjusted_capacity := unadjusted_capacity * 0.9
    adjusted_capacity = 216 := by
  sorry


end paperclip_capacity_l662_66295


namespace a_range_l662_66264

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_decreasing : ∀ x y, x < y → f x > f y
axiom f_domain : ∀ x, f x ≠ 0 → -7 < x ∧ x < 7
axiom f_condition : ∀ a, f (1 - a) + f (2*a - 5) < 0

-- Theorem statement
theorem a_range : 
  ∃ a₁ a₂, a₁ = 4 ∧ a₂ = 6 ∧ 
  (∀ a, (f (1 - a) + f (2*a - 5) < 0) → a₁ < a ∧ a < a₂) :=
sorry

end a_range_l662_66264


namespace range_of_a_l662_66215

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| ≥ 1
def q (x a : ℝ) : Prop := x ≤ a

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (p q : Prop) : Prop :=
  (q → p) ∧ ¬(p → q)

-- Theorem statement
theorem range_of_a (x a : ℝ) :
  (∀ x, necessary_not_sufficient (p x) (q x a)) →
  a ≤ -2 :=
by sorry

end range_of_a_l662_66215


namespace count_square_functions_l662_66257

-- Define the type for our function
def SquareFunction := Set ℤ → Set ℤ

-- Define what it means for a function to be in our family
def is_in_family (f : SquareFunction) : Prop :=
  ∃ (domain : Set ℤ),
    (∀ x ∈ domain, f domain = {y | ∃ x ∈ domain, y = x^2}) ∧
    (f domain = {1, 4})

-- State the theorem
theorem count_square_functions : 
  ∃! (n : ℕ), ∃ (functions : Finset SquareFunction),
    functions.card = n ∧
    (∀ f ∈ functions, is_in_family f) ∧
    (∀ f, is_in_family f → f ∈ functions) ∧
    n = 8 := by sorry

end count_square_functions_l662_66257


namespace dinosaur_book_cost_l662_66242

def dictionary_cost : ℕ := 11
def cookbook_cost : ℕ := 7
def total_cost : ℕ := 37

theorem dinosaur_book_cost :
  ∃ (dinosaur_cost : ℕ), 
    dictionary_cost + dinosaur_cost + cookbook_cost = total_cost ∧
    dinosaur_cost = 19 :=
by sorry

end dinosaur_book_cost_l662_66242


namespace sum_in_base_6_l662_66226

/-- Converts a number from base 6 to base 10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a number from base 10 to base 6 --/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

theorem sum_in_base_6 :
  let a := toBase10 [4, 3, 2, 1]  -- 1234₆
  let b := toBase10 [4, 3, 2]     -- 234₆
  let c := toBase10 [4, 3]        -- 34₆
  toBase6 (a + b + c) = [0, 5, 5, 2] -- 2550₆
  := by sorry

end sum_in_base_6_l662_66226


namespace flower_bed_area_l662_66252

theorem flower_bed_area (total_posts : ℕ) (post_spacing : ℝ) 
  (h1 : total_posts = 24)
  (h2 : post_spacing = 5)
  (h3 : ∃ (short_side long_side : ℕ), 
    short_side + 1 + long_side + 1 = total_posts ∧ 
    long_side + 1 = 3 * (short_side + 1)) :
  (short_side * post_spacing) * (long_side * post_spacing) = 600 := by
  sorry

end flower_bed_area_l662_66252


namespace divisibility_property_l662_66254

theorem divisibility_property (a b c d e n : ℤ) 
  (h_odd : Odd n)
  (h_sum_div : n ∣ (a + b + c + d + e))
  (h_sum_squares_div : n ∣ (a^2 + b^2 + c^2 + d^2 + e^2)) :
  n ∣ (a^5 + b^5 + c^5 + d^5 + e^5 - 5*a*b*c*d*e) := by
  sorry

end divisibility_property_l662_66254


namespace total_weight_of_mixtures_l662_66290

/-- Represents a mixture of vegetable ghee -/
structure Mixture where
  ratio_a : ℚ
  ratio_b : ℚ
  total_volume : ℚ

/-- Calculates the weight of a mixture in kg -/
def mixture_weight (m : Mixture) (weight_a weight_b : ℚ) : ℚ :=
  let total_ratio := m.ratio_a + m.ratio_b
  let volume_a := (m.ratio_a / total_ratio) * m.total_volume
  let volume_b := (m.ratio_b / total_ratio) * m.total_volume
  (volume_a * weight_a + volume_b * weight_b) / 1000

def mixture1 : Mixture := ⟨3, 2, 6⟩
def mixture2 : Mixture := ⟨5, 3, 4⟩
def mixture3 : Mixture := ⟨9, 4, 6.5⟩

def weight_a : ℚ := 900
def weight_b : ℚ := 750

theorem total_weight_of_mixtures :
  mixture_weight mixture1 weight_a weight_b +
  mixture_weight mixture2 weight_a weight_b +
  mixture_weight mixture3 weight_a weight_b = 13.965 := by
  sorry

end total_weight_of_mixtures_l662_66290


namespace common_roots_product_l662_66283

-- Define the two polynomial equations
def poly1 (K : ℝ) (x : ℝ) : ℝ := x^3 + K*x + 20
def poly2 (L : ℝ) (x : ℝ) : ℝ := x^3 + L*x^2 + 100

-- Define the theorem
theorem common_roots_product (K L : ℝ) :
  (∃ (u v : ℝ), u ≠ v ∧ 
    poly1 K u = 0 ∧ poly1 K v = 0 ∧
    poly2 L u = 0 ∧ poly2 L v = 0) →
  (∃ (p : ℝ), p = 10 * Real.rpow 2 (1/3) ∧
    ∃ (u v : ℝ), u ≠ v ∧ 
      poly1 K u = 0 ∧ poly1 K v = 0 ∧
      poly2 L u = 0 ∧ poly2 L v = 0 ∧
      u * v = p) :=
by sorry

end common_roots_product_l662_66283
