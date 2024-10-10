import Mathlib

namespace function_composition_property_l2213_221337

theorem function_composition_property (n : ℕ) :
  (∃ (f g : Fin n → Fin n), ∀ i : Fin n, 
    (f (g i) = i ∧ g (f i) ≠ i) ∨ (g (f i) = i ∧ f (g i) ≠ i)) ↔ 
  Even n :=
by sorry

end function_composition_property_l2213_221337


namespace shirt_price_reduction_l2213_221325

theorem shirt_price_reduction (original_price : ℝ) (h : original_price > 0) :
  let first_sale_price := 0.8 * original_price
  let final_price := 0.8 * first_sale_price
  final_price / original_price = 0.64 := by
sorry

end shirt_price_reduction_l2213_221325


namespace largest_digit_sum_l2213_221375

theorem largest_digit_sum (a b c : ℕ) (y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, c are digits
  (100 * a + 10 * b + c = 800 / y) →  -- 0.abc = 1/y
  (0 < y ∧ y ≤ 10) →  -- 0 < y ≤ 10
  (∃ (a' b' c' : ℕ), a' < 10 ∧ b' < 10 ∧ c' < 10 ∧ 
    100 * a' + 10 * b' + c' = 800 / y ∧ 
    a' + b' + c' = 8 ∧
    ∀ (x y z : ℕ), x < 10 → y < 10 → z < 10 → 
      100 * x + 10 * y + z = 800 / y → x + y + z ≤ 8) :=
by sorry

end largest_digit_sum_l2213_221375


namespace nh3_moles_produced_l2213_221373

structure Reaction where
  reactants : List (String × ℚ)
  products : List (String × ℚ)

def initial_moles : List (String × ℚ) := [
  ("NH4Cl", 3),
  ("KOH", 3),
  ("Na2CO3", 1),
  ("H3PO4", 1)
]

def reaction1 : Reaction := {
  reactants := [("NH4Cl", 2), ("Na2CO3", 1)],
  products := [("NH3", 2), ("CO2", 1), ("NaCl", 2), ("H2O", 1)]
}

def reaction2 : Reaction := {
  reactants := [("KOH", 2), ("H3PO4", 1)],
  products := [("K2HPO4", 1), ("H2O", 2)]
}

def limiting_reactant (reaction : Reaction) (available : List (String × ℚ)) : String :=
  sorry

def moles_produced (reaction : Reaction) (product : String) (limiting : String) : ℚ :=
  sorry

theorem nh3_moles_produced : 
  moles_produced reaction1 "NH3" (limiting_reactant reaction1 initial_moles) = 2 :=
sorry

end nh3_moles_produced_l2213_221373


namespace lines_perp_to_plane_are_parallel_line_not_perp_to_intersection_not_perp_to_other_plane_l2213_221321

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_to_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)
variable (intersection_line : Plane → Plane → Line)

-- Theorem 1: Two lines perpendicular to the same plane are parallel
theorem lines_perp_to_plane_are_parallel
  (p : Plane) (l1 l2 : Line)
  (h1 : perpendicular_to_plane l1 p)
  (h2 : perpendicular_to_plane l2 p) :
  parallel l1 l2 :=
sorry

-- Theorem 2: In perpendicular planes, a line not perpendicular to the intersection
-- is not perpendicular to the other plane
theorem line_not_perp_to_intersection_not_perp_to_other_plane
  (p1 p2 : Plane) (l : Line)
  (h1 : planes_perpendicular p1 p2)
  (h2 : in_plane l p1)
  (h3 : ¬ perpendicular l (intersection_line p1 p2)) :
  ¬ perpendicular_to_plane l p2 :=
sorry

end lines_perp_to_plane_are_parallel_line_not_perp_to_intersection_not_perp_to_other_plane_l2213_221321


namespace triangle_side_ratio_bound_one_half_is_greatest_bound_l2213_221333

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_ratio_bound (t : Triangle) :
  (t.a ^ 2 + t.b ^ 2) / t.c ^ 2 > 1 / 2 :=
sorry

theorem one_half_is_greatest_bound :
  ∀ ε > 0, ∃ t : Triangle, (t.a ^ 2 + t.b ^ 2) / t.c ^ 2 < 1 / 2 + ε :=
sorry

end triangle_side_ratio_bound_one_half_is_greatest_bound_l2213_221333


namespace emily_sleep_duration_l2213_221394

/-- Calculates the time Emily slept during her flight -/
def time_emily_slept (flight_duration : ℕ) (num_episodes : ℕ) (episode_duration : ℕ) 
  (num_movies : ℕ) (movie_duration : ℕ) (remaining_time : ℕ) : ℚ :=
  let total_flight_minutes := flight_duration * 60
  let total_tv_minutes := num_episodes * episode_duration
  let total_movie_minutes := num_movies * movie_duration
  let sleep_minutes := total_flight_minutes - total_tv_minutes - total_movie_minutes - remaining_time
  (sleep_minutes : ℚ) / 60

/-- Theorem stating that Emily slept for 4.5 hours -/
theorem emily_sleep_duration :
  time_emily_slept 10 3 25 2 105 45 = 4.5 := by sorry

end emily_sleep_duration_l2213_221394


namespace absolute_value_equation_roots_l2213_221353

theorem absolute_value_equation_roots : ∃ (x y : ℝ), 
  (x^2 - 3*|x| - 10 = 0) ∧ 
  (y^2 - 3*|y| - 10 = 0) ∧ 
  (x + y = 0) ∧ 
  (x * y = -25) := by
  sorry

end absolute_value_equation_roots_l2213_221353


namespace perpendicular_edges_count_l2213_221371

/-- A cube is a three-dimensional shape with 6 square faces -/
structure Cube where
  -- Add necessary fields here

/-- An edge of a cube -/
structure Edge (c : Cube) where
  -- Add necessary fields here

/-- Predicate to check if two edges are perpendicular -/
def perpendicular (c : Cube) (e1 e2 : Edge c) : Prop :=
  sorry

theorem perpendicular_edges_count (c : Cube) (e : Edge c) :
  (∃ (s : Finset (Edge c)), s.card = 8 ∧ ∀ e' ∈ s, perpendicular c e e') ∧
  ¬∃ (s : Finset (Edge c)), s.card > 8 ∧ ∀ e' ∈ s, perpendicular c e e' :=
sorry

end perpendicular_edges_count_l2213_221371


namespace q_div_p_equals_168_l2213_221360

/-- The number of slips in the hat -/
def total_slips : ℕ := 60

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 15

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The number of slips with each number -/
def slips_per_number : ℕ := 4

/-- The probability that all drawn slips bear the same number -/
def p : ℚ := (distinct_numbers : ℚ) / Nat.choose total_slips drawn_slips

/-- The probability that three slips bear one number and two bear a different number -/
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2 : ℚ) / Nat.choose total_slips drawn_slips

/-- The main theorem stating the ratio of q to p -/
theorem q_div_p_equals_168 : q / p = 168 := by sorry

end q_div_p_equals_168_l2213_221360


namespace triangle_max_side_length_l2213_221304

theorem triangle_max_side_length (D E F : Real) (side1 side2 : Real) :
  -- Triangle DEF exists
  0 < D ∧ 0 < E ∧ 0 < F ∧
  D + E + F = Real.pi ∧
  -- Given condition
  Real.cos (2 * D) + Real.cos (2 * E) + Real.cos (2 * F) = 1 ∧
  -- Two sides have lengths 8 and 15
  side1 = 8 ∧ side2 = 15 →
  -- The maximum length of the third side is 17
  ∃ side3 : Real, side3 ≤ 17 ∧
    ∀ x : Real, (∃ D' E' F' : Real,
      0 < D' ∧ 0 < E' ∧ 0 < F' ∧
      D' + E' + F' = Real.pi ∧
      Real.cos (2 * D') + Real.cos (2 * E') + Real.cos (2 * F') = 1 ∧
      x = ((side1^2 + side2^2 - 2 * side1 * side2 * Real.cos F')^(1/2))) →
    x ≤ 17 :=
by sorry

end triangle_max_side_length_l2213_221304


namespace square_of_a_l2213_221351

theorem square_of_a (a b c d : ℕ+) 
  (h1 : a < b) (h2 : b ≤ c) (h3 : c < d)
  (h4 : a * d = b * c)
  (h5 : Real.sqrt d - Real.sqrt a ≤ 1) :
  ∃ (n : ℕ), a = n^2 := by
  sorry

end square_of_a_l2213_221351


namespace first_tree_growth_rate_l2213_221348

/-- The daily growth rate of the first tree -/
def first_tree_growth : ℝ := 1

/-- The daily growth rate of the second tree -/
def second_tree_growth : ℝ := 2 * first_tree_growth

/-- The daily growth rate of the third tree -/
def third_tree_growth : ℝ := 2

/-- The daily growth rate of the fourth tree -/
def fourth_tree_growth : ℝ := 3

/-- The number of days the trees grew -/
def days : ℕ := 4

/-- The total growth of all trees -/
def total_growth : ℝ := 32

theorem first_tree_growth_rate :
  first_tree_growth * days +
  second_tree_growth * days +
  third_tree_growth * days +
  fourth_tree_growth * days = total_growth :=
by sorry

end first_tree_growth_rate_l2213_221348


namespace smallest_multiple_of_42_and_56_not_18_l2213_221390

theorem smallest_multiple_of_42_and_56_not_18 : 
  ∃ (n : ℕ), n > 0 ∧ 42 ∣ n ∧ 56 ∣ n ∧ ¬(18 ∣ n) ∧
  ∀ (m : ℕ), m > 0 → 42 ∣ m → 56 ∣ m → ¬(18 ∣ m) → n ≤ m :=
by sorry

end smallest_multiple_of_42_and_56_not_18_l2213_221390


namespace gcd_102_238_l2213_221368

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l2213_221368


namespace probability_red_ball_is_four_fifths_l2213_221328

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The experiment setup -/
def experiment : List Container :=
  [{ red := 5, green := 5 },  -- Container A
   { red := 7, green := 3 },  -- Container B
   { red := 7, green := 3 }]  -- Container C

/-- The probability of selecting a red ball in the described experiment -/
def probability_red_ball : ℚ :=
  (experiment.map (fun c => c.red / (c.red + c.green))).sum / experiment.length

theorem probability_red_ball_is_four_fifths :
  probability_red_ball = 4/5 := by
  sorry

end probability_red_ball_is_four_fifths_l2213_221328


namespace no_solution_condition_l2213_221334

theorem no_solution_condition (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) ≠ 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end no_solution_condition_l2213_221334


namespace salary_proof_l2213_221365

/-- Represents the man's salary in dollars -/
def salary : ℝ := 190000

/-- Theorem stating that given the spending conditions, the salary is $190000 -/
theorem salary_proof :
  let food_expense := (1 / 5 : ℝ) * salary
  let rent_expense := (1 / 10 : ℝ) * salary
  let clothes_expense := (3 / 5 : ℝ) * salary
  let remaining := salary - (food_expense + rent_expense + clothes_expense)
  remaining = 19000 := by sorry

end salary_proof_l2213_221365


namespace reciprocal_of_point_three_l2213_221315

theorem reciprocal_of_point_three (h : (0.3 : ℚ) = 3/10) : 
  (0.3 : ℚ)⁻¹ = 10/3 := by
  sorry

end reciprocal_of_point_three_l2213_221315


namespace gcd_factorial_problem_l2213_221329

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 5040 := by
  sorry

end gcd_factorial_problem_l2213_221329


namespace unknown_number_proof_l2213_221389

theorem unknown_number_proof (X : ℕ) : 
  1000 + X + 1000 + 30 + 1000 + 40 + 1000 + 10 = 4100 → X = 20 := by
  sorry

end unknown_number_proof_l2213_221389


namespace raft_cannot_turn_l2213_221347

/-- A raft is a shape with a measurable area -/
class Raft :=
  (area : ℝ)

/-- A canal is a path with a width and ability to turn -/
class Canal :=
  (width : ℝ)
  (turn_angle : ℝ)

/-- Determines if a raft can turn in a given canal -/
def can_turn (r : Raft) (c : Canal) : Prop :=
  sorry

/-- Theorem: A raft with area ≥ 2√2 cannot turn in a canal of width 1 with a 90° turn -/
theorem raft_cannot_turn (r : Raft) (c : Canal) :
  r.area ≥ 2 * Real.sqrt 2 →
  c.width = 1 →
  c.turn_angle = Real.pi / 2 →
  ¬(can_turn r c) :=
sorry

end raft_cannot_turn_l2213_221347


namespace unique_satisfying_polynomial_l2213_221399

/-- A polynomial satisfying the given conditions -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  (P 0 = 0) ∧ (∀ x, P (x^2 + 1) = P x^2 + 1)

/-- Theorem stating that the identity function is the only polynomial satisfying the conditions -/
theorem unique_satisfying_polynomial :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P → (∀ x, P x = x) :=
by sorry

end unique_satisfying_polynomial_l2213_221399


namespace total_pupils_l2213_221358

def number_of_girls : ℕ := 542
def number_of_boys : ℕ := 387

theorem total_pupils : number_of_girls + number_of_boys = 929 := by
  sorry

end total_pupils_l2213_221358


namespace box_surface_area_l2213_221346

/-- The surface area of a rectangular parallelepiped with dimensions a, b, c -/
def surfaceArea (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

theorem box_surface_area :
  ∀ a b c : ℕ,
    0 < a ∧ a < 10 →
    0 < b ∧ b < 10 →
    0 < c ∧ c < 10 →
    a * b * c = 280 →
    surfaceArea a b c = 262 := by
  sorry

end box_surface_area_l2213_221346


namespace smaller_factor_of_4536_l2213_221302

theorem smaller_factor_of_4536 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4536 → 
  min a b = 63 := by
sorry

end smaller_factor_of_4536_l2213_221302


namespace speed_ratio_is_one_third_l2213_221311

/-- The problem setup for two moving objects A and B -/
structure MovementProblem where
  vA : ℝ  -- Speed of A
  vB : ℝ  -- Speed of B
  initialDistance : ℝ  -- Initial distance of B from O

/-- The conditions of the problem -/
def satisfiesConditions (p : MovementProblem) : Prop :=
  p.initialDistance = 300 ∧
  p.vA = |p.initialDistance - p.vB| ∧
  7 * p.vA = |p.initialDistance - 7 * p.vB|

/-- The theorem to be proved -/
theorem speed_ratio_is_one_third (p : MovementProblem) 
  (h : satisfiesConditions p) : p.vA / p.vB = 1 / 3 := by
  sorry


end speed_ratio_is_one_third_l2213_221311


namespace reciprocal_of_lcm_l2213_221306

def a : ℕ := 24
def b : ℕ := 195

theorem reciprocal_of_lcm (a b : ℕ) : (1 : ℚ) / (Nat.lcm a b) = 1 / 1560 := by
  sorry

end reciprocal_of_lcm_l2213_221306


namespace region_area_l2213_221322

/-- The region in the plane defined by |x + 2y| + |x - 2y| ≤ 6 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 + 2*p.2| + |p.1 - 2*p.2| ≤ 6}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem region_area : area Region = 9 := by
  sorry

end region_area_l2213_221322


namespace job_completion_time_l2213_221349

theorem job_completion_time (a_time b_time : ℝ) (combined_time : ℝ) (combined_work : ℝ) : 
  a_time = 15 →
  combined_time = 8 →
  combined_work = 0.9333333333333333 →
  combined_work = combined_time * (1 / a_time + 1 / b_time) →
  b_time = 20 := by
  sorry

end job_completion_time_l2213_221349


namespace partner_count_l2213_221350

theorem partner_count (P A : ℕ) (h1 : P / A = 2 / 63) (h2 : P / (A + 50) = 1 / 34) : P = 20 := by
  sorry

end partner_count_l2213_221350


namespace bill_difference_l2213_221379

theorem bill_difference (mike_tip joe_tip : ℝ) (mike_percent joe_percent : ℝ) 
  (h1 : mike_tip = 5)
  (h2 : joe_tip = 10)
  (h3 : mike_percent = 20)
  (h4 : joe_percent = 25)
  (h5 : mike_tip = mike_percent / 100 * mike_bill)
  (h6 : joe_tip = joe_percent / 100 * joe_bill) :
  |mike_bill - joe_bill| = 15 :=
sorry

end bill_difference_l2213_221379


namespace train_speed_time_reduction_l2213_221395

theorem train_speed_time_reduction :
  ∀ (v S : ℝ),
  v > 0 → S > 0 →
  let original_time := S / v
  let new_speed := 1.25 * v
  let new_time := S / new_speed
  (original_time - new_time) / original_time = 0.2 := by
  sorry

end train_speed_time_reduction_l2213_221395


namespace rs_length_l2213_221369

/-- Triangle ABC with altitude CH, points R and S on CH -/
structure TriangleWithAltitude where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point H on the altitude CH -/
  H : ℝ × ℝ
  /-- Point R on CH, tangent point of inscribed circle in ACH -/
  R : ℝ × ℝ
  /-- Point S on CH, tangent point of inscribed circle in BCH -/
  S : ℝ × ℝ
  /-- CH is an altitude of triangle ABC -/
  altitude : (C.1 - H.1) * (B.1 - A.1) + (C.2 - H.2) * (B.2 - A.2) = 0
  /-- AB = 13 -/
  ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13
  /-- AC = 12 -/
  ac_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 12
  /-- BC = 5 -/
  bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 5
  /-- R is on CH -/
  r_on_ch : ∃ t : ℝ, R = (C.1 + t * (H.1 - C.1), C.2 + t * (H.2 - C.2))
  /-- S is on CH -/
  s_on_ch : ∃ t : ℝ, S = (C.1 + t * (H.1 - C.1), C.2 + t * (H.2 - C.2))

/-- The main theorem to prove -/
theorem rs_length (t : TriangleWithAltitude) : 
  Real.sqrt ((t.R.1 - t.S.1)^2 + (t.R.2 - t.S.2)^2) = 24 / 13 := by
  sorry

end rs_length_l2213_221369


namespace sum_of_all_expressions_l2213_221386

/-- Represents an expression formed by replacing * with + or - in 1 * 2 * 3 * 4 * 5 * 6 -/
def Expression := List (Bool × ℕ)

/-- Generates all possible expressions -/
def generateExpressions : List Expression :=
  sorry

/-- Evaluates a single expression -/
def evaluateExpression (expr : Expression) : ℤ :=
  sorry

/-- Sums the results of all expressions -/
def sumAllExpressions : ℤ :=
  (generateExpressions.map evaluateExpression).sum

theorem sum_of_all_expressions :
  sumAllExpressions = 32 := by
  sorry

end sum_of_all_expressions_l2213_221386


namespace primitive_roots_existence_l2213_221378

theorem primitive_roots_existence (p : Nat) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ x : Nat, IsPrimitiveRoot x p ∧ IsPrimitiveRoot (4 * x) p :=
sorry

end primitive_roots_existence_l2213_221378


namespace student_ticket_cost_l2213_221362

/-- Proves that the cost of each student ticket is $6 given the conditions of the problem -/
theorem student_ticket_cost (adult_ticket_cost : ℕ) (num_students : ℕ) (num_adults : ℕ) (total_revenue : ℕ) :
  adult_ticket_cost = 8 →
  num_students = 20 →
  num_adults = 12 →
  total_revenue = 216 →
  ∃ (student_ticket_cost : ℕ), 
    student_ticket_cost * num_students + adult_ticket_cost * num_adults = total_revenue ∧
    student_ticket_cost = 6 :=
by
  sorry

end student_ticket_cost_l2213_221362


namespace triangle_inequality_for_powers_l2213_221338

theorem triangle_inequality_for_powers (a b c : ℝ) :
  (∀ n : ℕ, a^n + b^n > c^n ∧ a^n + c^n > b^n ∧ b^n + c^n > a^n) ↔ 
  ((a = b ∧ a > c) ∨ (a = b ∧ b = c)) :=
sorry

end triangle_inequality_for_powers_l2213_221338


namespace arithmetic_sequence_terms_l2213_221392

theorem arithmetic_sequence_terms (a₁ : ℝ) (d : ℝ) (aₙ : ℝ) (n : ℕ) :
  a₁ = 2.5 →
  d = 4 →
  aₙ = 46.5 →
  aₙ = a₁ + (n - 1) * d →
  n = 12 := by
sorry

end arithmetic_sequence_terms_l2213_221392


namespace smallest_b_for_quadratic_inequality_l2213_221319

theorem smallest_b_for_quadratic_inequality :
  ∀ b : ℝ, b^2 - 16*b + 55 ≥ 0 → b ≥ 5 :=
by sorry

end smallest_b_for_quadratic_inequality_l2213_221319


namespace bert_stamp_cost_l2213_221307

/-- The total cost of stamps Bert purchased -/
def total_cost (type_a_count type_b_count type_c_count : ℕ) 
               (type_a_price type_b_price type_c_price : ℕ) : ℕ :=
  type_a_count * type_a_price + 
  type_b_count * type_b_price + 
  type_c_count * type_c_price

/-- Theorem stating the total cost of Bert's stamp purchase -/
theorem bert_stamp_cost : 
  total_cost 150 90 60 2 3 5 = 870 := by
  sorry

end bert_stamp_cost_l2213_221307


namespace complex_number_equality_l2213_221396

theorem complex_number_equality : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by sorry

end complex_number_equality_l2213_221396


namespace subsets_with_sum_2008_l2213_221387

def set_63 : Finset ℕ := Finset.range 64 \ {0}

theorem subsets_with_sum_2008 : 
  (Finset.filter (fun S => S.sum id = 2008) (Finset.powerset set_63)).card = 6 := by
  sorry

end subsets_with_sum_2008_l2213_221387


namespace intersection_implies_m_range_l2213_221314

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x^2 - 4*m*x + 2*m + 6 = 0}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem intersection_implies_m_range (m : ℝ) : (A m ∩ B).Nonempty → m ≤ -1 := by
  sorry

end intersection_implies_m_range_l2213_221314


namespace natural_number_puzzle_l2213_221339

def first_digit (n : ℕ) : ℕ := n.div (10 ^ (n.log 10))

def last_digit (n : ℕ) : ℕ := n % 10

def swap_first_last (n : ℕ) : ℕ :=
  let d := n.log 10
  last_digit n * 10^d + (n - first_digit n * 10^d - last_digit n) + first_digit n

theorem natural_number_puzzle (x : ℕ) :
  first_digit x = 2 →
  last_digit x = 5 →
  swap_first_last x = 2 * x + 2 →
  x ≤ 10000 →
  x = 25 ∨ x = 295 ∨ x = 2995 := by
  sorry

end natural_number_puzzle_l2213_221339


namespace inverse_proportion_example_l2213_221303

/-- Represents an inverse proportional relationship between two variables -/
def InverseProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function f(x) = 3/x is inversely proportional -/
theorem inverse_proportion_example : InverseProportion (fun x => 3 / x) := by
  sorry

end inverse_proportion_example_l2213_221303


namespace fraction_of_week_worked_l2213_221336

/-- Proves that given a usual work week of 40 hours, an hourly rate of $15, and a weekly salary of $480, the fraction of the usual week worked is 4/5. -/
theorem fraction_of_week_worked 
  (usual_hours : ℕ) 
  (hourly_rate : ℚ) 
  (weekly_salary : ℚ) 
  (h1 : usual_hours = 40)
  (h2 : hourly_rate = 15)
  (h3 : weekly_salary = 480) :
  (weekly_salary / hourly_rate) / usual_hours = 4/5 :=
by sorry

end fraction_of_week_worked_l2213_221336


namespace fraction_denominator_expression_l2213_221300

theorem fraction_denominator_expression 
  (x y a b : ℝ) 
  (h1 : x / y = 3) 
  (h2 : (2 * a - x) / (3 * b - y) = 3) 
  (h3 : a / b = 4.5) : 
  ∃ (E : ℝ), (2 * a - x) / E = 3 ∧ E = 3 * b - y := by
  sorry

end fraction_denominator_expression_l2213_221300


namespace polar_to_rectangular_conversion_l2213_221383

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4 * Real.sqrt 2
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 4 ∧ y = 4 := by sorry

end polar_to_rectangular_conversion_l2213_221383


namespace range_of_f_l2213_221343

def f (x : ℝ) : ℝ := x^2 - 3*x

def domain : Set ℝ := {1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-2, 0} := by sorry

end range_of_f_l2213_221343


namespace problem_statement_l2213_221374

theorem problem_statement :
  (∀ x : ℝ, (x + 8) * (x + 11) < (x + 9) * (x + 10)) ∧
  (Real.sqrt 5 - 2 > Real.sqrt 6 - Real.sqrt 5) := by
  sorry

end problem_statement_l2213_221374


namespace initial_men_count_l2213_221381

theorem initial_men_count (initial_days : ℝ) (additional_men : ℕ) (final_days : ℝ) :
  initial_days = 18 →
  additional_men = 450 →
  final_days = 13.090909090909092 →
  ∃ (initial_men : ℕ), 
    initial_men * initial_days = (initial_men + additional_men) * final_days ∧
    initial_men = 1200 := by
  sorry

end initial_men_count_l2213_221381


namespace base7_135_equals_base10_75_l2213_221305

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- Theorem stating that 135 in base 7 is equal to 75 in base 10 --/
theorem base7_135_equals_base10_75 : base7ToBase10 1 3 5 = 75 := by
  sorry

end base7_135_equals_base10_75_l2213_221305


namespace distance_to_origin_l2213_221359

/-- The distance from point P(3,4) to the origin in the Cartesian coordinate system is 5. -/
theorem distance_to_origin : Real.sqrt (3^2 + 4^2) = 5 := by
  sorry

end distance_to_origin_l2213_221359


namespace cubic_zeros_sum_less_than_two_l2213_221309

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

noncomputable def F (a b c x : ℝ) : ℝ := f a b c x - x * Real.exp (-x)

theorem cubic_zeros_sum_less_than_two (a b c : ℝ) (ha : a ≠ 0) 
    (h1 : 6 * a + b = 0) (h2 : f a b c 1 = 4 * a) :
    ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    0 ≤ x₁ ∧ x₃ ≤ 3 ∧
    F a b c x₁ = 0 ∧ F a b c x₂ = 0 ∧ F a b c x₃ = 0 ∧
    x₁ + x₂ + x₃ < 2 := by
  sorry

end cubic_zeros_sum_less_than_two_l2213_221309


namespace class_fraction_proof_l2213_221385

theorem class_fraction_proof (G : ℚ) (B : ℚ) (T : ℚ) (h1 : B / G = 3 / 2) (h2 : T = B + G) :
  (G / 2) / T = 1 / 5 := by
  sorry

end class_fraction_proof_l2213_221385


namespace condition_relationship_l2213_221320

theorem condition_relationship (x₁ x₂ : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 4 ∧ x₂ > 4 → x₁ + x₂ > 8 ∧ x₁ * x₂ > 16) ∧
  (∃ x₁ x₂ : ℝ, x₁ + x₂ > 8 ∧ x₁ * x₂ > 16 ∧ ¬(x₁ > 4 ∧ x₂ > 4)) :=
by sorry

end condition_relationship_l2213_221320


namespace min_solutions_in_interval_l2213_221342

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem min_solutions_in_interval 
  (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_period : has_period f 3) 
  (h_root : f 2 = 0) : 
  ∃ (a b c d : ℝ), 0 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < 6 ∧ 
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 :=
sorry

end min_solutions_in_interval_l2213_221342


namespace pentagon_area_increase_l2213_221341

/-- The increase in area when expanding a convex pentagon's boundary --/
theorem pentagon_area_increase (P s : ℝ) (h : P > 0) (h' : s > 0) :
  let increase := s * P + π * s^2
  increase = s * P + π * s^2 := by
  sorry

end pentagon_area_increase_l2213_221341


namespace decimal_sum_and_product_l2213_221382

theorem decimal_sum_and_product :
  let sum := 0.5 + 0.03 + 0.007
  sum = 0.537 ∧ 3 * sum = 1.611 :=
by sorry

end decimal_sum_and_product_l2213_221382


namespace g_uniqueness_l2213_221354

/-- The functional equation for g -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y

theorem g_uniqueness (g : ℝ → ℝ) (h1 : g 1 = 1) (h2 : FunctionalEquation g) :
    ∀ x : ℝ, g x = 4^x - 3^x := by
  sorry

end g_uniqueness_l2213_221354


namespace equation_solution_l2213_221340

theorem equation_solution :
  ∃! x : ℚ, x ≠ 0 ∧ x ≠ 2 ∧ (2 * x) / (x - 2) - 2 = 1 / (x * (x - 2)) ∧ x = 1 / 4 := by
  sorry

end equation_solution_l2213_221340


namespace sector_area_l2213_221361

/-- Given an arc length of 4 cm corresponding to a central angle of 2 radians,
    the area of the sector enclosed by this central angle is 4 cm². -/
theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (h1 : arc_length = 4) (h2 : central_angle = 2) :
  let radius := arc_length / central_angle
  let sector_area := (1 / 2) * radius^2 * central_angle
  sector_area = 4 := by
sorry

end sector_area_l2213_221361


namespace smallest_k_with_remainder_one_k_400_satisfies_conditions_smallest_k_is_400_l2213_221393

theorem smallest_k_with_remainder_one (k : ℕ) : k > 1 ∧ 
  k % 19 = 1 ∧ k % 7 = 1 ∧ k % 3 = 1 → k ≥ 400 := by
  sorry

theorem k_400_satisfies_conditions : 
  400 > 1 ∧ 400 % 19 = 1 ∧ 400 % 7 = 1 ∧ 400 % 3 = 1 := by
  sorry

theorem smallest_k_is_400 : 
  ∃! k : ℕ, k > 1 ∧ k % 19 = 1 ∧ k % 7 = 1 ∧ k % 3 = 1 ∧ 
  ∀ m : ℕ, (m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1) → k ≤ m := by
  sorry

end smallest_k_with_remainder_one_k_400_satisfies_conditions_smallest_k_is_400_l2213_221393


namespace stratified_sampling_city_B_l2213_221327

theorem stratified_sampling_city_B (total_points : ℕ) (city_B_points : ℕ) (sample_size : ℕ) :
  total_points = 450 →
  city_B_points = 150 →
  sample_size = 90 →
  (city_B_points : ℚ) / (total_points : ℚ) * (sample_size : ℚ) = 30 := by
  sorry

end stratified_sampling_city_B_l2213_221327


namespace all_hyperprimes_l2213_221301

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def isSegmentPrime (n : ℕ) : Prop :=
  ∀ start len : ℕ, len > 0 → start + len ≤ (Nat.digits 10 n).length →
    isPrime (Nat.digits 10 n |> List.take len |> List.drop start |> List.foldl (· * 10 + ·) 0)

def isHyperprime (n : ℕ) : Prop := n > 0 ∧ isSegmentPrime n

theorem all_hyperprimes :
  {n : ℕ | isHyperprime n} = {2, 3, 5, 7, 23, 37, 53, 73, 373} := by sorry

end all_hyperprimes_l2213_221301


namespace length_PQ_is_sqrt_82_l2213_221380

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 8*y - 5 = 0

-- Define the line y = 2x
def line_center (x y : ℝ) : Prop :=
  y = 2*x

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  y = x - 1

-- Define the point M
def point_M : ℝ × ℝ := (3, 2)

-- Theorem statement
theorem length_PQ_is_sqrt_82 :
  ∀ (P Q : ℝ × ℝ),
  circle_C (-2) 1 →  -- Point A on circle C
  circle_C 5 0 →     -- Point B on circle C
  (∃ (cx cy : ℝ), circle_C cx cy ∧ line_center cx cy) →  -- Center of C on y = 2x
  line_m P.1 P.2 →   -- P is on line m
  line_m Q.1 Q.2 →   -- Q is on line m
  circle_C P.1 P.2 → -- P is on circle C
  circle_C Q.1 Q.2 → -- Q is on circle C
  line_m point_M.1 point_M.2 →  -- M is on line m
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 82 :=
by
  sorry

end length_PQ_is_sqrt_82_l2213_221380


namespace cubic_root_from_quadratic_l2213_221344

theorem cubic_root_from_quadratic : ∀ r : ℝ, 
  (r^2 = r + 2) → (r^3 = 3*r + 2) ∧ (3 * 2 = 6) := by
  sorry

end cubic_root_from_quadratic_l2213_221344


namespace combined_travel_time_l2213_221324

/-- Given a car that takes 4.5 hours to reach station B, and a train that takes 2 hours longer
    than the car to cover the same distance, the combined time for both to reach station B
    is 11 hours. -/
theorem combined_travel_time (car_time train_time : ℝ) : 
  car_time = 4.5 →
  train_time = car_time + 2 →
  car_time + train_time = 11 := by
sorry

end combined_travel_time_l2213_221324


namespace train_length_is_200_l2213_221313

/-- The length of a train that crosses a 200-meter bridge in 10 seconds
    and passes a lamp post on the bridge in 5 seconds. -/
def train_length : ℝ := 200

/-- The length of the bridge in meters. -/
def bridge_length : ℝ := 200

/-- The time taken to cross the bridge in seconds. -/
def bridge_crossing_time : ℝ := 10

/-- The time taken to pass the lamp post in seconds. -/
def lamppost_passing_time : ℝ := 5

/-- Theorem stating that the train length is 200 meters given the conditions. -/
theorem train_length_is_200 :
  train_length = 200 :=
by sorry

end train_length_is_200_l2213_221313


namespace factory_weekly_production_l2213_221363

/-- Represents a toy production line with a daily production rate -/
structure ProductionLine where
  dailyRate : ℕ

/-- Represents a factory with multiple production lines -/
structure Factory where
  lines : List ProductionLine
  daysPerWeek : ℕ

/-- Calculates the total weekly production of a factory -/
def weeklyProduction (factory : Factory) : ℕ :=
  (factory.lines.map (λ line => line.dailyRate * factory.daysPerWeek)).sum

/-- The theorem stating the total weekly production of the given factory -/
theorem factory_weekly_production :
  let lineA : ProductionLine := ⟨1500⟩
  let lineB : ProductionLine := ⟨1800⟩
  let lineC : ProductionLine := ⟨2200⟩
  let factory : Factory := ⟨[lineA, lineB, lineC], 5⟩
  weeklyProduction factory = 27500 := by
  sorry


end factory_weekly_production_l2213_221363


namespace pencil_cost_l2213_221357

/-- Given Mrs. Hilt's initial amount and the amount left after buying a pencil,
    prove that the cost of the pencil is the difference between these two amounts. -/
theorem pencil_cost (initial_amount amount_left : ℕ) 
    (h1 : initial_amount = 15)
    (h2 : amount_left = 4) :
    initial_amount - amount_left = 11 := by
  sorry

end pencil_cost_l2213_221357


namespace contrapositive_equality_l2213_221384

theorem contrapositive_equality (a b : ℝ) : 
  (¬(|a| = |b|) → ¬(a = -b)) ↔ (a = -b → |a| = |b|) := by sorry

end contrapositive_equality_l2213_221384


namespace willow_count_l2213_221367

theorem willow_count (total : ℕ) (diff : ℕ) : 
  total = 83 →
  diff = 11 →
  ∃ (willows oaks : ℕ),
    willows + oaks = total ∧
    oaks = willows + diff ∧
    willows = 36 := by
  sorry

end willow_count_l2213_221367


namespace inequality_solution_l2213_221372

theorem inequality_solution (x : ℝ) : 
  x^3 - 3*x^2 - 4*x - 12 ≤ 0 ∧ 2*x + 6 > 0 → x ∈ Set.Icc (-2) 3 := by
  sorry

end inequality_solution_l2213_221372


namespace max_value_sqrt_sum_l2213_221356

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 6) :
  Real.sqrt (x + 3) + Real.sqrt (y + 3) + Real.sqrt (z + 3) ≤ 3 * Real.sqrt 5 ∧
  ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 6 ∧
    Real.sqrt (a + 3) + Real.sqrt (b + 3) + Real.sqrt (c + 3) = 3 * Real.sqrt 5 :=
by sorry

end max_value_sqrt_sum_l2213_221356


namespace negation_of_all_is_some_not_l2213_221364

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (M : U → Prop)  -- M x means "x is a member of the math club"
variable (E : U → Prop)  -- E x means "x enjoys puzzles"

-- State the theorem
theorem negation_of_all_is_some_not :
  (¬ ∀ x, M x → E x) ↔ (∃ x, M x ∧ ¬ E x) := by sorry

end negation_of_all_is_some_not_l2213_221364


namespace root_condition_implies_m_range_l2213_221398

theorem root_condition_implies_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 - 2*m*x + 4 = 0 ∧ y^2 - 2*m*y + 4 = 0 ∧ x > 1 ∧ y < 1) →
  m > 5/2 :=
by sorry

end root_condition_implies_m_range_l2213_221398


namespace polynomial_ascending_powers_x_l2213_221317

-- Define the polynomial
def p (x y : ℝ) : ℝ := x^3 - 5*x*y^2 - 7*y^3 + 8*x^2*y

-- Define a function to extract the degree of x in a term
def degree_x (term : ℝ → ℝ → ℝ) : ℕ :=
  sorry  -- Implementation details omitted

-- Define the ascending order of terms with respect to x
def ascending_order_x (term1 term2 : ℝ → ℝ → ℝ) : Prop :=
  degree_x term1 ≤ degree_x term2

-- State the theorem
theorem polynomial_ascending_powers_x :
  ∃ (term1 term2 term3 term4 : ℝ → ℝ → ℝ),
    (∀ x y, p x y = term1 x y + term2 x y + term3 x y + term4 x y) ∧
    (ascending_order_x term1 term2) ∧
    (ascending_order_x term2 term3) ∧
    (ascending_order_x term3 term4) ∧
    (∀ x y, term1 x y = -7*y^3) ∧
    (∀ x y, term2 x y = -5*x*y^2) ∧
    (∀ x y, term3 x y = 8*x^2*y) ∧
    (∀ x y, term4 x y = x^3) :=
  sorry

end polynomial_ascending_powers_x_l2213_221317


namespace quadratic_roots_modulus_l2213_221316

theorem quadratic_roots_modulus (a : ℝ) : 
  (∀ x : ℂ, (a * x^2 + x + 1 = 0) → Complex.abs x < 1) → a > 1 := by
  sorry

end quadratic_roots_modulus_l2213_221316


namespace fraction_square_equals_twentyfive_l2213_221391

theorem fraction_square_equals_twentyfive : (123456^2 : ℚ) / (24691^2 : ℚ) = 25 := by sorry

end fraction_square_equals_twentyfive_l2213_221391


namespace intersection_condition_l2213_221388

theorem intersection_condition (m : ℤ) : 
  let A : Set ℤ := {0, m}
  let B : Set ℤ := {n : ℤ | n^2 - 3*n < 0}
  (A ∩ B).Nonempty → m = 1 ∨ m = 2 := by
sorry

end intersection_condition_l2213_221388


namespace max_distance_theorem_l2213_221332

def vector_a : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (x, y)
def vector_b : ℝ × ℝ := (1, 2)

theorem max_distance_theorem (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (max_val : ℝ), max_val = Real.sqrt 5 + 1 ∧
    ∀ (a : ℝ × ℝ), a = vector_a (x, y) →
      ‖a - vector_b‖ ≤ max_val ∧
      ∃ (a' : ℝ × ℝ), a' = vector_a (x', y') ∧ x'^2 + y'^2 = 1 ∧ ‖a' - vector_b‖ = max_val :=
by
  sorry

end max_distance_theorem_l2213_221332


namespace ab_power_is_negative_eight_l2213_221330

theorem ab_power_is_negative_eight (a b : ℝ) (h : |a + 2| + (b - 3)^2 = 0) : a^b = -8 := by
  sorry

end ab_power_is_negative_eight_l2213_221330


namespace sheila_cinnamon_balls_l2213_221323

/-- The number of days Sheila can place cinnamon balls -/
def days : ℕ := 10

/-- The total number of cinnamon balls Sheila bought -/
def total_balls : ℕ := 50

/-- The number of family members Sheila placed a cinnamon ball for every day -/
def family_members : ℕ := total_balls / days

theorem sheila_cinnamon_balls : family_members = 5 := by
  sorry

end sheila_cinnamon_balls_l2213_221323


namespace red_rows_in_specific_grid_l2213_221310

/-- Represents the grid coloring problem -/
structure GridColoring where
  total_rows : ℕ
  squares_per_row : ℕ
  blue_rows : ℕ
  green_squares : ℕ
  red_squares_per_row : ℕ

/-- Calculates the number of red rows in the grid -/
def red_rows (g : GridColoring) : ℕ :=
  let total_squares := g.total_rows * g.squares_per_row
  let blue_squares := g.blue_rows * g.squares_per_row
  let red_squares := total_squares - blue_squares - g.green_squares
  red_squares / g.red_squares_per_row

/-- Theorem stating the number of red rows in the specific problem -/
theorem red_rows_in_specific_grid :
  let g : GridColoring := {
    total_rows := 10,
    squares_per_row := 15,
    blue_rows := 4,
    green_squares := 66,
    red_squares_per_row := 6
  }
  red_rows g = 4 := by sorry

end red_rows_in_specific_grid_l2213_221310


namespace unpainted_squares_count_l2213_221335

/-- Calculates the number of unpainted squares in a grid strip with a repeating pattern -/
def unpainted_squares (width : ℕ) (length : ℕ) (pattern_width : ℕ) 
  (unpainted_per_pattern : ℕ) (unpainted_remainder : ℕ) : ℕ :=
  let complete_patterns := length / pattern_width
  let remainder_columns := length % pattern_width
  complete_patterns * unpainted_per_pattern + unpainted_remainder

/-- The number of unpainted squares in a 5x250 grid with the given pattern is 812 -/
theorem unpainted_squares_count :
  unpainted_squares 5 250 4 13 6 = 812 := by
  sorry

end unpainted_squares_count_l2213_221335


namespace adams_initial_money_l2213_221397

/-- Adam's initial money problem -/
theorem adams_initial_money :
  ∀ (x : ℤ), (x - 2 + 5 = 8) → x = 5 := by
  sorry

end adams_initial_money_l2213_221397


namespace unique_solution_quadratic_inequality_l2213_221345

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, |x^2 + 3*a*x + 4*a| ≤ 3) ↔ (a = 8 + 2*Real.sqrt 13 ∨ a = 8 - 2*Real.sqrt 13) :=
sorry

end unique_solution_quadratic_inequality_l2213_221345


namespace logical_conditions_l2213_221376

-- Define a proposition type to represent logical statements
variable (A B : Prop)

-- Define sufficient condition
def is_sufficient_condition (A B : Prop) : Prop :=
  A → B

-- Define necessary condition
def is_necessary_condition (A B : Prop) : Prop :=
  B → A

-- Define necessary and sufficient condition
def is_necessary_and_sufficient_condition (A B : Prop) : Prop :=
  (A → B) ∧ (B → A)

-- Theorem statement
theorem logical_conditions :
  (is_sufficient_condition A B ↔ (A → B)) ∧
  (is_necessary_condition A B ↔ (B → A)) ∧
  (is_necessary_and_sufficient_condition A B ↔ ((A → B) ∧ (B → A))) :=
by sorry

end logical_conditions_l2213_221376


namespace prob_sum_15_three_dice_l2213_221326

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The sum we're looking for -/
def targetSum : ℕ := 15

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces^numDice

/-- The number of favorable outcomes (sum of 15) -/
def favorableOutcomes : ℕ := 7

/-- Theorem: The probability of rolling a sum of 15 with three standard 6-faced dice is 7/72 -/
theorem prob_sum_15_three_dice : 
  (favorableOutcomes : ℚ) / totalOutcomes = 7 / 72 := by sorry

end prob_sum_15_three_dice_l2213_221326


namespace bobby_shoe_cost_l2213_221366

/-- Calculates the total cost of Bobby's handmade shoes -/
def calculate_total_cost (mold_cost : ℝ) (material_cost : ℝ) (material_discount : ℝ) 
  (hourly_rate : ℝ) (rate_increase : ℝ) (work_hours : ℝ) (work_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_material := material_cost * (1 - material_discount)
  let new_hourly_rate := hourly_rate + rate_increase
  let work_cost := work_hours * new_hourly_rate * work_discount
  let subtotal := mold_cost + discounted_material + work_cost
  let total := subtotal * (1 + tax_rate)
  total

/-- Theorem stating that Bobby's total cost is $1005.40 -/
theorem bobby_shoe_cost : 
  calculate_total_cost 250 150 0.2 75 10 8 0.8 0.1 = 1005.40 := by
  sorry

end bobby_shoe_cost_l2213_221366


namespace quadratic_roots_implications_l2213_221331

theorem quadratic_roots_implications (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    (7 * x₁^2 - (a + 13) * x₁ + a^2 - a - 2 = 0) ∧
    (7 * x₂^2 - (a + 13) * x₂ + a^2 - a - 2 = 0) ∧
    (0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < 2)) →
  (a ∈ Set.Ioo (-2 : ℝ) (-1) ∪ Set.Ioo 3 4) ∧
  (∀ a' ∈ Set.Ioo 3 4, a'^3 > a'^2 - a' + 1) ∧
  (∀ a' ∈ Set.Ioo (-2 : ℝ) (-1), a'^3 < a'^2 - a' + 1) :=
by sorry

end quadratic_roots_implications_l2213_221331


namespace student_selection_count_l2213_221352

theorem student_selection_count (n m k : ℕ) (hn : n = 60) (hm : m = 2) (hk : k = 5) :
  (Nat.choose n k - Nat.choose (n - m) k : ℕ) =
  (Nat.choose m 1 * Nat.choose (n - 1) (k - 1) -
   Nat.choose m 2 * Nat.choose (n - m) (k - 2) : ℕ) :=
by sorry

end student_selection_count_l2213_221352


namespace statement_2_statement_3_l2213_221355

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Statement 2
theorem statement_2 (m n : Line) (α : Plane) :
  parallel m α → perpendicular n α → perpendicular_lines n m :=
sorry

-- Statement 3
theorem statement_3 (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → perpendicular_planes α β :=
sorry

end statement_2_statement_3_l2213_221355


namespace mark_payment_l2213_221312

def bread_cost : ℚ := 21/5
def cheese_cost : ℚ := 41/20
def nickel_value : ℚ := 1/20
def dime_value : ℚ := 1/10
def quarter_value : ℚ := 1/4
def num_nickels : ℕ := 8

theorem mark_payment (total_cost change payment : ℚ) :
  total_cost = bread_cost + cheese_cost →
  change = num_nickels * nickel_value + dime_value + quarter_value →
  payment = total_cost + change →
  payment = 7 := by sorry

end mark_payment_l2213_221312


namespace consecutive_squares_equivalence_l2213_221308

theorem consecutive_squares_equivalence (n : ℤ) : 
  (∃ a : ℤ, n = a^2 + (a + 1)^2) ↔ (∃ b : ℤ, 2*n - 1 = b^2) :=
by sorry

end consecutive_squares_equivalence_l2213_221308


namespace johns_hats_cost_l2213_221377

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of weeks John can wear a different hat each day -/
def weeks_of_different_hats : ℕ := 2

/-- The cost of each hat in dollars -/
def cost_per_hat : ℕ := 50

/-- The total cost of John's hats -/
def total_cost : ℕ := weeks_of_different_hats * days_in_week * cost_per_hat

theorem johns_hats_cost : total_cost = 700 := by
  sorry

end johns_hats_cost_l2213_221377


namespace triple_a_award_distribution_l2213_221318

theorem triple_a_award_distribution (n : Nat) (k : Nat) (h1 : n = 10) (h2 : k = 7) :
  (Nat.choose (n - k + k - 1) (n - k)) = 84 := by
  sorry

end triple_a_award_distribution_l2213_221318


namespace least_positive_period_is_30_l2213_221370

/-- A function satisfying the given condition -/
def PeriodicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least positive period of a function -/
def IsLeastPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ IsPeriod f p ∧ ∀ q : ℝ, 0 < q ∧ q < p → ¬IsPeriod f q

theorem least_positive_period_is_30 :
  ∀ f : ℝ → ℝ, PeriodicFunction f → IsLeastPositivePeriod f 30 :=
sorry

end least_positive_period_is_30_l2213_221370
