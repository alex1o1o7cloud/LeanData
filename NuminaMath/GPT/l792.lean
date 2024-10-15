import Mathlib

namespace NUMINAMATH_GPT_geometric_series_sum_l792_79215

theorem geometric_series_sum :
  let a := (1 / 2 : ℝ)
  let r := (1 / 2 : ℝ)
  let n := 6
  (a * (1 - r^n) / (1 - r)) = (63 / 64 : ℝ) := 
by 
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l792_79215


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l792_79271

theorem sufficient_not_necessary_condition (a : ℝ) : (a = 2 → (a^2 - a) * 1 + 1 = 0) ∧ (¬ ((a^2 - a) * 1 + 1 = 0 → a = 2)) :=
by sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l792_79271


namespace NUMINAMATH_GPT_find_b_l792_79246

theorem find_b (a b c : ℝ) (h1 : a + b + c = 120) (h2 : a + 5 = b - 5) (h3 : b - 5 = c^2) : b = 61.25 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_b_l792_79246


namespace NUMINAMATH_GPT_domain_of_function_l792_79297

/-- The domain of the function \( y = \lg (12 + x - x^2) \) is the interval \(-3 < x < 4\). -/
theorem domain_of_function :
  {x : ℝ | 12 + x - x^2 > 0} = {x : ℝ | -3 < x ∧ x < 4} :=
sorry

end NUMINAMATH_GPT_domain_of_function_l792_79297


namespace NUMINAMATH_GPT_circle_diameter_eq_l792_79280

-- Definitions
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0
def point_A (x y : ℝ) : Prop := x = 0 ∧ y = 3
def point_B (x y : ℝ) : Prop := x = -4 ∧ y = 0
def midpoint_AB (x y : ℝ) : Prop := x = -2 ∧ y = 3 / 2 -- Midpoint of A(0,3) and B(-4,0)
def diameter_AB : ℝ := 5

-- The equation of the circle with diameter AB
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 3 * y = 0

-- The proof statement
theorem circle_diameter_eq :
  (∃ A B : ℝ × ℝ, point_A A.1 A.2 ∧ point_B B.1 B.2 ∧ 
                   midpoint_AB ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧ diameter_AB = 5) →
  (∀ x y : ℝ, circle_eq x y) :=
sorry

end NUMINAMATH_GPT_circle_diameter_eq_l792_79280


namespace NUMINAMATH_GPT_find_some_number_l792_79208

theorem find_some_number (some_number : ℝ) :
  (0.0077 * some_number) / (0.04 * 0.1 * 0.007) = 990.0000000000001 → 
  some_number = 3.6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_some_number_l792_79208


namespace NUMINAMATH_GPT_smaller_number_is_25_l792_79290

theorem smaller_number_is_25 (x y : ℕ) (h1 : x + y = 62) (h2 : y = x + 12) : x = 25 :=
by sorry

end NUMINAMATH_GPT_smaller_number_is_25_l792_79290


namespace NUMINAMATH_GPT_cost_of_article_l792_79262

theorem cost_of_article (C G1 G2 : ℝ) (h1 : G1 = 380 - C) (h2 : G2 = 450 - C) (h3 : G2 = 1.10 * G1) : 
  C = 320 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_article_l792_79262


namespace NUMINAMATH_GPT_find_value_of_squares_l792_79254

-- Defining the conditions
variable (a b c : ℝ)
variable (h1 : a^2 + 3 * b = 10)
variable (h2 : b^2 + 5 * c = 0)
variable (h3 : c^2 + 7 * a = -21)

-- Stating the theorem to prove the desired result
theorem find_value_of_squares : a^2 + b^2 + c^2 = 83 / 4 :=
   sorry

end NUMINAMATH_GPT_find_value_of_squares_l792_79254


namespace NUMINAMATH_GPT_yellow_marbles_count_l792_79222

theorem yellow_marbles_count 
  (total_marbles red_marbles blue_marbles : ℕ) 
  (h_total : total_marbles = 85) 
  (h_red : red_marbles = 14) 
  (h_blue : blue_marbles = 3 * red_marbles) :
  (total_marbles - (red_marbles + blue_marbles)) = 29 :=
by
  sorry

end NUMINAMATH_GPT_yellow_marbles_count_l792_79222


namespace NUMINAMATH_GPT_quadratic_eq_real_roots_l792_79248

theorem quadratic_eq_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 4 * x + 2 = 0) →
  (∃ y : ℝ, a * y^2 - 4 * y + 2 = 0) →
  a ≤ 2 ∧ a ≠ 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_eq_real_roots_l792_79248


namespace NUMINAMATH_GPT_people_remaining_on_bus_l792_79235

theorem people_remaining_on_bus
  (students_left : ℕ) (students_right : ℕ) (students_back : ℕ)
  (students_aisle : ℕ) (teachers : ℕ) (bus_driver : ℕ) 
  (students_off1 : ℕ) (teachers_off1 : ℕ)
  (students_off2 : ℕ) (teachers_off2 : ℕ)
  (students_off3 : ℕ) :
  students_left = 42 ∧ students_right = 38 ∧ students_back = 5 ∧
  students_aisle = 15 ∧ teachers = 2 ∧ bus_driver = 1 ∧
  students_off1 = 14 ∧ teachers_off1 = 1 ∧
  students_off2 = 18 ∧ teachers_off2 = 1 ∧
  students_off3 = 5 →
  (students_left + students_right + students_back + students_aisle + teachers + bus_driver) -
  (students_off1 + teachers_off1 + students_off2 + teachers_off2 + students_off3) = 64 :=
by {
  sorry
}

end NUMINAMATH_GPT_people_remaining_on_bus_l792_79235


namespace NUMINAMATH_GPT_probability_odd_sum_is_one_half_probability_2x_plus_y_less_than_10_is_seven_eighteenths_l792_79253

def num_faces : ℕ := 6
def possible_outcomes : ℕ := num_faces * num_faces

def count_odd_sum_outcomes : ℕ := 18 -- From solution steps
def probability_odd_sum : ℚ := count_odd_sum_outcomes / possible_outcomes

def count_2x_plus_y_less_than_10 : ℕ := 14 -- From solution steps
def probability_2x_plus_y_less_than_10 : ℚ := count_2x_plus_y_less_than_10 / possible_outcomes

theorem probability_odd_sum_is_one_half :
  probability_odd_sum = 1 / 2 :=
sorry

theorem probability_2x_plus_y_less_than_10_is_seven_eighteenths :
  probability_2x_plus_y_less_than_10 = 7 / 18 :=
sorry

end NUMINAMATH_GPT_probability_odd_sum_is_one_half_probability_2x_plus_y_less_than_10_is_seven_eighteenths_l792_79253


namespace NUMINAMATH_GPT_find_n_between_50_and_150_l792_79274

theorem find_n_between_50_and_150 :
  ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 150 ∧
  n % 7 = 0 ∧ 
  n % 9 = 3 ∧ 
  n % 6 = 3 ∧ 
  n % 4 = 1 ∧
  n = 105 :=
by
  sorry

end NUMINAMATH_GPT_find_n_between_50_and_150_l792_79274


namespace NUMINAMATH_GPT_prob_geometry_given_algebra_l792_79282

variable (algebra geometry : ℕ) (total : ℕ)

/-- Proof of the probability of selecting a geometry question on the second draw,
    given that an algebra question is selected on the first draw. -/
theorem prob_geometry_given_algebra : 
  algebra = 3 ∧ geometry = 2 ∧ total = 5 →
  (algebra / (total : ℚ)) * (geometry / (total - 1 : ℚ)) = 1 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_prob_geometry_given_algebra_l792_79282


namespace NUMINAMATH_GPT_quadratic_function_proof_l792_79264

theorem quadratic_function_proof (a c : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x^2 - 4 * x + c)
  (h_sol_set : ∀ x, f x < 0 → (-1 < x ∧ x < 5)) :
  (a = 1 ∧ c = -5) ∧ (∀ x, 0 ≤ x ∧ x ≤ 3 → -9 ≤ f x ∧ f x ≤ -5) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_proof_l792_79264


namespace NUMINAMATH_GPT_alarm_prob_l792_79224

theorem alarm_prob (pA pB : ℝ) (hA : pA = 0.80) (hB : pB = 0.90) : 
  (1 - (1 - pA) * (1 - pB)) = 0.98 :=
by 
  sorry

end NUMINAMATH_GPT_alarm_prob_l792_79224


namespace NUMINAMATH_GPT_problem1421_part1_problem1421_part2_problem1421_part3_problem1421_part4_l792_79230

theorem problem1421_part1 (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ)
  (h_total : total_balls = 20) (h_red : red_balls = 5) (h_yellow : yellow_balls = 15) :
  (red_balls < yellow_balls) := by 
  sorry  -- Solution Proof for Part 1

theorem problem1421_part2 (total_balls : ℕ) (red_balls : ℕ) (h_total : total_balls = 20) 
  (h_red : red_balls = 5) :
  (red_balls / total_balls = 1 / 4) := by 
  sorry  -- Solution Proof for Part 2

theorem problem1421_part3 (red_balls total_balls m : ℕ) (h_red : red_balls = 5) 
  (h_total : total_balls = 20) :
  ((red_balls + m) / (total_balls + m) = 3 / 4) → (m = 40) := by 
  sorry  -- Solution Proof for Part 3

theorem problem1421_part4 (total_balls red_balls additional_balls x : ℕ) 
  (h_total : total_balls = 20) (h_red : red_balls = 5) (h_additional : additional_balls = 18):
  (total_balls + additional_balls = 38) → ((red_balls + x) / 38 = 1 / 2) → 
  (x = 14) ∧ ((additional_balls - x) = 4) := by 
  sorry  -- Solution Proof for Part 4

end NUMINAMATH_GPT_problem1421_part1_problem1421_part2_problem1421_part3_problem1421_part4_l792_79230


namespace NUMINAMATH_GPT_inequality_proof_l792_79250

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_geq : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l792_79250


namespace NUMINAMATH_GPT_part1_part2a_part2b_l792_79223

-- Definitions and conditions
def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (-3, 2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def scalar_mul (k : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (k * u.1, k * u.2)
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def collinear (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- Proof statements

-- Part 1: Verify the dot product computation
theorem part1 : dot_product (vector_add vector_a vector_b) (vector_sub vector_a vector_b) = -8 := by
  sorry

-- Part 2a: Verify the value of k for parallel vectors
theorem part2a : collinear (vector_add (scalar_mul (1/3) vector_a) vector_b) (vector_sub vector_a (scalar_mul 3 vector_b)) := by
  sorry

-- Part 2b: Verify antiparallel direction
theorem part2b : collinear (vector_add (scalar_mul (1/3) vector_a) vector_b) (scalar_mul (-1) (vector_sub vector_a (scalar_mul 3 vector_b))) := by
  sorry

end NUMINAMATH_GPT_part1_part2a_part2b_l792_79223


namespace NUMINAMATH_GPT_tan_alpha_is_neg_5_over_12_l792_79296

variables (α : ℝ) (h1 : Real.sin α = 5/13) (h2 : π/2 < α ∧ α < π)

theorem tan_alpha_is_neg_5_over_12 : Real.tan α = -5/12 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_is_neg_5_over_12_l792_79296


namespace NUMINAMATH_GPT_distinct_intersection_points_l792_79205

theorem distinct_intersection_points :
  let S1 := { p : ℝ × ℝ | (p.1 + p.2 - 7) * (2 * p.1 - 3 * p.2 + 9) = 0 }
  let S2 := { p : ℝ × ℝ | (p.1 - p.2 - 2) * (4 * p.1 + 3 * p.2 - 18) = 0 }
  ∃! (p1 p2 p3 : ℝ × ℝ), p1 ∈ S1 ∧ p1 ∈ S2 ∧ p2 ∈ S1 ∧ p2 ∈ S2 ∧ p3 ∈ S1 ∧ p3 ∈ S2 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 :=
sorry

end NUMINAMATH_GPT_distinct_intersection_points_l792_79205


namespace NUMINAMATH_GPT_ordering_of_powers_l792_79245

theorem ordering_of_powers :
  (3:ℕ)^15 < 10^9 ∧ 10^9 < (5:ℕ)^13 :=
by
  sorry

end NUMINAMATH_GPT_ordering_of_powers_l792_79245


namespace NUMINAMATH_GPT_exists_pos_integers_l792_79216

theorem exists_pos_integers (r : ℚ) (hr : r > 0) : 
  ∃ a b c d : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ r = (a^3 + b^3) / (c^3 + d^3) :=
by sorry

end NUMINAMATH_GPT_exists_pos_integers_l792_79216


namespace NUMINAMATH_GPT_find_room_length_l792_79232

theorem find_room_length (w : ℝ) (A : ℝ) (h_w : w = 8) (h_A : A = 96) : (A / w = 12) :=
by
  rw [h_w, h_A]
  norm_num

end NUMINAMATH_GPT_find_room_length_l792_79232


namespace NUMINAMATH_GPT_cans_of_beans_is_two_l792_79201

-- Define the problem parameters
variable (C B T : ℕ)

-- Conditions based on the problem statement
axiom chili_can : C = 1
axiom tomato_to_bean_ratio : T = 3 * B / 2
axiom quadruple_batch_cans : 4 * (C + B + T) = 24

-- Prove the number of cans of beans is 2
theorem cans_of_beans_is_two : B = 2 :=
by
  -- Include conditions
  have h1 : C = 1 := by sorry
  have h2 : T = 3 * B / 2 := by sorry
  have h3 : 4 * (C + B + T) = 24 := by sorry
  -- Derive the answer (Proof omitted)
  sorry

end NUMINAMATH_GPT_cans_of_beans_is_two_l792_79201


namespace NUMINAMATH_GPT_quadratic_range_m_l792_79291

theorem quadratic_range_m (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m * x + 1 = 0 ∧ y^2 + m * y + 1 = 0) ↔ (m < -2 ∨ m > 2) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_range_m_l792_79291


namespace NUMINAMATH_GPT_difference_of_interests_l792_79286

def investment_in_funds (X Y : ℝ) (total_investment : ℝ) : ℝ := X + Y
def interest_earned (investment_rate : ℝ) (amount : ℝ) : ℝ := investment_rate * amount

variable (X : ℝ) (Y : ℝ)
variable (total_investment : ℝ) (rate_X : ℝ) (rate_Y : ℝ)
variable (investment_X : ℝ) 

axiom h1 : total_investment = 100000
axiom h2 : rate_X = 0.23
axiom h3 : rate_Y = 0.17
axiom h4 : investment_X = 42000
axiom h5 : investment_in_funds X Y total_investment = total_investment - investment_X

-- We need to show the difference in interest is 200
theorem difference_of_interests : 
  let interest_X := interest_earned rate_X investment_X
  let investment_Y := total_investment - investment_X
  let interest_Y := interest_earned rate_Y investment_Y
  interest_Y - interest_X = 200 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_interests_l792_79286


namespace NUMINAMATH_GPT_find_circle_center_l792_79200

theorem find_circle_center :
  ∀ x y : ℝ,
  (x^2 + 4*x + y^2 - 6*y = 20) →
  (x + 2, y - 3) = (-2, 3) := by
  sorry

end NUMINAMATH_GPT_find_circle_center_l792_79200


namespace NUMINAMATH_GPT_students_spring_outing_l792_79277

theorem students_spring_outing (n : ℕ) (h1 : n = 5) : 2^n = 32 :=
  by {
    sorry
  }

end NUMINAMATH_GPT_students_spring_outing_l792_79277


namespace NUMINAMATH_GPT_binary_subtraction_result_l792_79239

theorem binary_subtraction_result :
  let x := 0b1101101 -- binary notation for 109
  let y := 0b11101   -- binary notation for 29
  let z := 0b101010  -- binary notation for 42
  let product := x * y
  let result := product - z
  result = 0b10000010001 := -- binary notation for 3119
by
  sorry

end NUMINAMATH_GPT_binary_subtraction_result_l792_79239


namespace NUMINAMATH_GPT_A_holds_15_l792_79206

def cards : List (ℕ × ℕ) := [(1, 3), (1, 5), (3, 5)]

variables (A_card B_card C_card : ℕ × ℕ)

-- Conditions from the problem
def C_not_35 : Prop := C_card ≠ (3, 5)
def A_says_not_3 (A_card B_card : ℕ × ℕ) : Prop := ¬(A_card.1 = 3 ∧ B_card.1 = 3 ∨ A_card.2 = 3 ∧ B_card.2 = 3)
def B_says_not_1 (B_card C_card : ℕ × ℕ) : Prop := ¬(B_card.1 = 1 ∧ C_card.1 = 1 ∨ B_card.2 = 1 ∧ C_card.2 = 1)

-- Question to prove
theorem A_holds_15 : 
  ∃ (A_card B_card C_card : ℕ × ℕ),
    A_card ∈ cards ∧ B_card ∈ cards ∧ C_card ∈ cards ∧
    A_card ≠ B_card ∧ B_card ≠ C_card ∧ A_card ≠ C_card ∧
    C_not_35 C_card ∧
    A_says_not_3 A_card B_card ∧
    B_says_not_1 B_card C_card ->
    A_card = (1, 5) :=
sorry

end NUMINAMATH_GPT_A_holds_15_l792_79206


namespace NUMINAMATH_GPT_tickets_to_be_sold_l792_79273

theorem tickets_to_be_sold : 
  let total_tickets := 200
  let jude_tickets := 16
  let andrea_tickets := 4 * jude_tickets
  let sandra_tickets := 2 * jude_tickets + 8
  total_tickets - (jude_tickets + andrea_tickets + sandra_tickets) = 80 := by
  sorry

end NUMINAMATH_GPT_tickets_to_be_sold_l792_79273


namespace NUMINAMATH_GPT_not_square_or_cube_l792_79233

theorem not_square_or_cube (n : ℕ) (h : n > 1) : 
  ¬ (∃ a : ℕ, 2^n - 1 = a^2) ∧ ¬ (∃ a : ℕ, 2^n - 1 = a^3) :=
by
  sorry

end NUMINAMATH_GPT_not_square_or_cube_l792_79233


namespace NUMINAMATH_GPT_shortest_distance_from_curve_to_line_l792_79214

noncomputable def curve (x : ℝ) : ℝ := Real.log (2 * x - 1)

def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

theorem shortest_distance_from_curve_to_line : 
  ∃ (x y : ℝ), y = curve x ∧ line x y ∧ 
  (∀ (x₀ y₀ : ℝ), y₀ = curve x₀ → ∃ (x₀ y₀ : ℝ), 
    y₀ = curve x₀ ∧ d = Real.sqrt 5) :=
sorry

end NUMINAMATH_GPT_shortest_distance_from_curve_to_line_l792_79214


namespace NUMINAMATH_GPT_range_of_a_l792_79247

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a - 1) * x < a - 1 ↔ x > 1) : a < 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_l792_79247


namespace NUMINAMATH_GPT_find_a_in_terms_of_x_l792_79257

variable (a b x : ℝ)

theorem find_a_in_terms_of_x (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) : a = 3 * x :=
sorry

end NUMINAMATH_GPT_find_a_in_terms_of_x_l792_79257


namespace NUMINAMATH_GPT_min_value_polynomial_expression_at_k_eq_1_is_0_l792_79225

-- Definition of the polynomial expression
def polynomial_expression (k x y : ℝ) : ℝ :=
  3 * x^2 - 4 * k * x * y + (2 * k^2 + 1) * y^2 - 6 * x - 2 * y + 4

-- Proof statement
theorem min_value_polynomial_expression_at_k_eq_1_is_0 :
  (∀ x y : ℝ, polynomial_expression 1 x y ≥ 0) ∧ (∃ x y : ℝ, polynomial_expression 1 x y = 0) :=
by
  -- Expected proof here. For now, we indicate sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_min_value_polynomial_expression_at_k_eq_1_is_0_l792_79225


namespace NUMINAMATH_GPT_excircle_problem_l792_79202

-- Define the data structure for a triangle with incenter and excircle properties
structure TriangleWithIncenterAndExcircle (α : Type) [LinearOrderedField α] :=
  (A B C I X : α)
  (is_incenter : Boolean)  -- condition for point I being the incenter
  (is_excircle_center_opposite_A : Boolean)  -- condition for point X being the excircle center opposite A
  (I_A_I : I ≠ A)
  (X_A_X : X ≠ A)

-- Define the problem statement
theorem excircle_problem
  (α : Type) [LinearOrderedField α]
  (T : TriangleWithIncenterAndExcircle α)
  (h_incenter : T.is_incenter)
  (h_excircle_center : T.is_excircle_center_opposite_A)
  (h_not_eq_I : T.I ≠ T.A)
  (h_not_eq_X : T.X ≠ T.A)
  : 
    (T.I * T.X = T.A * T.B) ∧ 
    (T.I * (T.B * T.C) = T.X * (T.B * T.C)) :=
by
  sorry

end NUMINAMATH_GPT_excircle_problem_l792_79202


namespace NUMINAMATH_GPT_min_value_64_l792_79211

noncomputable def min_value_expr (a b c d e f g h : ℝ) : ℝ :=
  (a * e) ^ 2 + (b * f) ^ 2 + (c * g) ^ 2 + (d * h) ^ 2

theorem min_value_64 
  (a b c d e f g h : ℝ) 
  (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16)
  (h3 : a + b + c + d = e * f * g) :
  min_value_expr a b c d e f g h = 64 := 
sorry

end NUMINAMATH_GPT_min_value_64_l792_79211


namespace NUMINAMATH_GPT_number_of_triangles_fitting_in_square_l792_79217

-- Define the conditions for the right triangle and the square
def right_triangle_height := 2
def right_triangle_width := 2
def square_side := 2

-- Define the areas
def area_triangle := (1 / 2) * right_triangle_height * right_triangle_width
def area_square := square_side * square_side

-- Define the proof statement to show the number of right triangles fitting in the square is 2
theorem number_of_triangles_fitting_in_square : (area_square / area_triangle) = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_triangles_fitting_in_square_l792_79217


namespace NUMINAMATH_GPT_root_expr_value_eq_175_div_11_l792_79270

noncomputable def root_expr_value (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + bc + ca = 25) (h3 : abc = 10) : ℝ :=
  (a / (1 / a + b * c)) + (b / (1 / b + c * a)) + (c / (1 / c + a * b))

theorem root_expr_value_eq_175_div_11 (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : ab + bc + ca = 25) 
  (h3 : abc = 10) : 
  root_expr_value a b c h1 h2 h3 = 175 / 11 := 
sorry

end NUMINAMATH_GPT_root_expr_value_eq_175_div_11_l792_79270


namespace NUMINAMATH_GPT_fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25_l792_79293

def positive_integers_up_to (n : ℕ) : List ℕ :=
  List.range' 1 n

def divisible_by_lcm (lcm : ℕ) (lst : List ℕ) : List ℕ :=
  lst.filter (λ x => x % lcm = 0)

noncomputable def fraction_divisible_by_both (n a b : ℕ) : ℚ :=
  let lcm_ab := Nat.lcm a b
  let elems := positive_integers_up_to n
  let divisible_elems := divisible_by_lcm lcm_ab elems
  divisible_elems.length / n

theorem fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25 :
  fraction_divisible_by_both 100 3 4 = (2 : ℚ) / 25 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25_l792_79293


namespace NUMINAMATH_GPT_rate_of_change_area_at_t4_l792_79236

variable (t : ℝ)

def a (t : ℝ) : ℝ := 2 * t + 1

def b (t : ℝ) : ℝ := 3 * t + 2

def S (t : ℝ) : ℝ := a t * b t

theorem rate_of_change_area_at_t4 :
  (deriv S 4) = 55 := by
  sorry

end NUMINAMATH_GPT_rate_of_change_area_at_t4_l792_79236


namespace NUMINAMATH_GPT_sunset_duration_l792_79260

theorem sunset_duration (changes : ℕ) (interval : ℕ) (total_changes : ℕ) (h1 : total_changes = 12) (h2 : interval = 10) : ∃ hours : ℕ, hours = 2 :=
by
  sorry

end NUMINAMATH_GPT_sunset_duration_l792_79260


namespace NUMINAMATH_GPT_sum_last_two_digits_pow_mod_eq_zero_l792_79295

/-
Given condition: 
Sum of the last two digits of \( 9^{25} + 11^{25} \)
-/
theorem sum_last_two_digits_pow_mod_eq_zero : 
  let a := 9
  let b := 11
  let n := 25 
  (a ^ n + b ^ n) % 100 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_last_two_digits_pow_mod_eq_zero_l792_79295


namespace NUMINAMATH_GPT_sum_of_n_with_unformable_postage_120_equals_43_l792_79268

theorem sum_of_n_with_unformable_postage_120_equals_43 :
  ∃ n1 n2 : ℕ, n1 = 21 ∧ n2 = 22 ∧ 
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, k = 7 * a + n1 * b + (n1 + 1) * c) ∧ 
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, k = 7 * a + n2 * b + (n2 + 1) * c) ∧ 
  (120 = 7 * a + n1 * b + (n1 + 1) * c → a = 0 ∧ b = 0 ∧ c = 0) ∧
  (120 = 7 * a + n2 * b + (n2 + 1) * c → a = 0 ∧ b = 0 ∧ c = 0) ∧
  (n1 + n2 = 43) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_n_with_unformable_postage_120_equals_43_l792_79268


namespace NUMINAMATH_GPT_shoveling_problem_l792_79294

variable (S : ℝ) -- Wayne's son's shoveling rate (driveways per hour)
variable (W : ℝ) -- Wayne's shoveling rate (driveways per hour)
variable (T : ℝ) -- Time it takes for Wayne's son to shovel the driveway alone (hours)

theorem shoveling_problem 
  (h1 : W = 6 * S)
  (h2 : (S + W) * 3 = 1) : T = 21 := 
by
  sorry

end NUMINAMATH_GPT_shoveling_problem_l792_79294


namespace NUMINAMATH_GPT_trucks_needed_l792_79269

-- Definitions of the conditions
def total_apples : ℕ := 80
def apples_transported : ℕ := 56
def truck_capacity : ℕ := 4

-- Definition to calculate the remaining apples
def remaining_apples : ℕ := total_apples - apples_transported

-- The theorem statement
theorem trucks_needed : remaining_apples / truck_capacity = 6 := by
  sorry

end NUMINAMATH_GPT_trucks_needed_l792_79269


namespace NUMINAMATH_GPT_number_of_cartons_of_pencils_l792_79272

theorem number_of_cartons_of_pencils (P E : ℕ) 
  (h1 : P + E = 100) 
  (h2 : 6 * P + 3 * E = 360) : 
  P = 20 := 
by
  sorry

end NUMINAMATH_GPT_number_of_cartons_of_pencils_l792_79272


namespace NUMINAMATH_GPT_solve_for_y_l792_79220

theorem solve_for_y (y : ℝ) (h : 3 * y ^ (1 / 4) - 3 * y ^ (1 / 2) / y ^ (1 / 4) = 13 - 2 * y ^ (1 / 4)) :
  y = (13 / 2) ^ 4 :=
by sorry

end NUMINAMATH_GPT_solve_for_y_l792_79220


namespace NUMINAMATH_GPT_problem1_solutions_problem2_solutions_l792_79251

-- Problem 1: Solve x² - 7x + 6 = 0

theorem problem1_solutions (x : ℝ) : 
  x^2 - 7 * x + 6 = 0 ↔ (x = 1 ∨ x = 6) := by
  sorry

-- Problem 2: Solve (2x + 3)² = (x - 3)² 

theorem problem2_solutions (x : ℝ) : 
  (2 * x + 3)^2 = (x - 3)^2 ↔ (x = 0 ∨ x = -6) := by
  sorry

end NUMINAMATH_GPT_problem1_solutions_problem2_solutions_l792_79251


namespace NUMINAMATH_GPT_line_parallel_to_plane_l792_79279

-- Defining conditions
def vector_a : ℝ × ℝ × ℝ := (1, -1, 3)
def vector_n : ℝ × ℝ × ℝ := (0, 3, 1)

-- Lean theorem statement
theorem line_parallel_to_plane : 
  let ⟨a1, a2, a3⟩ := vector_a;
  let ⟨n1, n2, n3⟩ := vector_n;
  a1 * n1 + a2 * n2 + a3 * n3 = 0 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_line_parallel_to_plane_l792_79279


namespace NUMINAMATH_GPT_xy_sum_l792_79234

theorem xy_sum (x y : ℝ) (h1 : 2 / x + 3 / y = 4) (h2 : 2 / x - 3 / y = -2) : x + y = 3 := by
  sorry

end NUMINAMATH_GPT_xy_sum_l792_79234


namespace NUMINAMATH_GPT_icosahedron_edge_probability_l792_79228

theorem icosahedron_edge_probability :
  let vertices := 12
  let total_pairs := vertices * (vertices - 1) / 2
  let edges := 30
  let probability := edges.toFloat / total_pairs.toFloat
  probability = 5 / 11 :=
by
  sorry

end NUMINAMATH_GPT_icosahedron_edge_probability_l792_79228


namespace NUMINAMATH_GPT_binary_to_octal_101110_l792_79284

theorem binary_to_octal_101110 : 
  ∀ (binary_to_octal : ℕ → ℕ), 
  binary_to_octal 0b101110 = 0o56 :=
by
  sorry

end NUMINAMATH_GPT_binary_to_octal_101110_l792_79284


namespace NUMINAMATH_GPT_variance_scaled_data_l792_79276

noncomputable def variance (data : List ℝ) : ℝ :=
  let n := data.length
  let mean := data.sum / n
  (data.map (λ x => (x - mean) ^ 2)).sum / n

theorem variance_scaled_data (data : List ℝ) (h_len : data.length > 0) (h_var : variance data = 4) :
  variance (data.map (λ x => 2 * x)) = 16 :=
by
  sorry

end NUMINAMATH_GPT_variance_scaled_data_l792_79276


namespace NUMINAMATH_GPT_function_bounded_in_interval_l792_79292

variables {f : ℝ → ℝ}

theorem function_bounded_in_interval (h : ∀ x y : ℝ, x > y → f x ^ 2 ≤ f y) : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_function_bounded_in_interval_l792_79292


namespace NUMINAMATH_GPT_points_connected_l792_79226

theorem points_connected (m l : ℕ) (h1 : l < m) (h2 : Even (l * m)) :
  ∃ points : Finset (ℕ × ℕ), ∀ p ∈ points, (∃ q, q ∈ points ∧ (p ≠ q → p.snd = q.snd → p.fst = q.fst)) :=
sorry

end NUMINAMATH_GPT_points_connected_l792_79226


namespace NUMINAMATH_GPT_neon_signs_blink_together_l792_79209

-- Define the time intervals for the blinks
def blink_interval1 : ℕ := 7
def blink_interval2 : ℕ := 11
def blink_interval3 : ℕ := 13

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- State the theorem
theorem neon_signs_blink_together : Nat.lcm (Nat.lcm blink_interval1 blink_interval2) blink_interval3 = 1001 := by
  sorry

end NUMINAMATH_GPT_neon_signs_blink_together_l792_79209


namespace NUMINAMATH_GPT_avg_scores_relation_l792_79213

variables (class_avg top8_avg other32_avg : ℝ)

theorem avg_scores_relation (h1 : 40 = 40) 
  (h2 : top8_avg = class_avg + 3) :
  other32_avg = top8_avg - 3.75 :=
sorry

end NUMINAMATH_GPT_avg_scores_relation_l792_79213


namespace NUMINAMATH_GPT_vertices_of_square_l792_79265

-- Define lattice points as points with integer coordinates
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Define the distance between two lattice points
def distance (P Q : LatticePoint) : ℤ :=
  (P.x - Q.x) * (P.x - Q.x) + (P.y - Q.y) * (P.y - Q.y)

-- Define the area of a triangle formed by three lattice points using the determinant method
def area (P Q R : LatticePoint) : ℤ :=
  (Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x)

-- Prove that three distinct lattice points form the vertices of a square given the condition
theorem vertices_of_square (P Q R : LatticePoint) (h₀ : P ≠ Q) (h₁ : Q ≠ R) (h₂ : P ≠ R)
    (h₃ : (distance P Q + distance Q R) < 8 * (area P Q R) + 1) :
    ∃ S : LatticePoint, S ≠ P ∧ S ≠ Q ∧ S ≠ R ∧
    (distance P Q = distance Q R ∧ distance Q R = distance R S ∧ distance R S = distance S P) := 
by sorry

end NUMINAMATH_GPT_vertices_of_square_l792_79265


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l792_79218

theorem solve_equation1 (x : ℝ) (h : 4 * x^2 - 81 = 0) : x = 9/2 ∨ x = -9/2 := 
sorry

theorem solve_equation2 (x : ℝ) (h : 8 * (x + 1)^3 = 27) : x = 1/2 := 
sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l792_79218


namespace NUMINAMATH_GPT_N_vector_3_eq_result_vector_l792_79298

noncomputable def matrix_N : Matrix (Fin 2) (Fin 2) ℝ :=
-- The matrix N is defined such that:
-- N * (vector 3 -2) = (vector 4 1)
-- N * (vector -2 3) = (vector 1 2)
sorry

def vector_1 : Fin 2 → ℝ := fun | ⟨0,_⟩ => 3 | ⟨1,_⟩ => -2
def vector_2 : Fin 2 → ℝ := fun | ⟨0,_⟩ => -2 | ⟨1,_⟩ => 3
def vector_3 : Fin 2 → ℝ := fun | ⟨0,_⟩ => 7 | ⟨1,_⟩ => 0
def result_vector : Fin 2 → ℝ := fun | ⟨0,_⟩ => 14 | ⟨1,_⟩ => 7

theorem N_vector_3_eq_result_vector :
  matrix_N.mulVec vector_3 = result_vector := by
  -- Given conditions:
  -- matrix_N.mulVec vector_1 = vector_4
  -- and matrix_N.mulVec vector_2 = vector_5
  sorry

end NUMINAMATH_GPT_N_vector_3_eq_result_vector_l792_79298


namespace NUMINAMATH_GPT_f_expression_when_x_gt_1_l792_79267

variable (f : ℝ → ℝ)

-- conditions
def f_even : Prop := ∀ x, f (x + 1) = f (-x + 1)
def f_defn_when_x_lt_1 : Prop := ∀ x, x < 1 → f x = x ^ 2 + 1

-- theorem to prove
theorem f_expression_when_x_gt_1 (h_even : f_even f) (h_defn : f_defn_when_x_lt_1 f) : 
  ∀ x, x > 1 → f x = x ^ 2 - 4 * x + 5 := 
by
  sorry

end NUMINAMATH_GPT_f_expression_when_x_gt_1_l792_79267


namespace NUMINAMATH_GPT_windmere_zoo_two_legged_birds_l792_79242

theorem windmere_zoo_two_legged_birds (b m u : ℕ) (head_count : b + m + u = 300) (leg_count : 2 * b + 4 * m + 3 * u = 710) : b = 230 :=
sorry

end NUMINAMATH_GPT_windmere_zoo_two_legged_birds_l792_79242


namespace NUMINAMATH_GPT_solution_set_of_inequality_l792_79287

theorem solution_set_of_inequality (x : ℝ) (h : (2 * x - 1) / x < 0) : 0 < x ∧ x < 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l792_79287


namespace NUMINAMATH_GPT_decreasing_interval_implies_a_ge_two_l792_79238

-- The function f is given
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 3

-- Defining the condition for f(x) being decreasing in the interval (-8, 2)
def is_decreasing_in_interval (a : ℝ) : Prop :=
  ∀ x y : ℝ, (-8 < x ∧ x < y ∧ y < 2) → f x a > f y a

-- The proof statement
theorem decreasing_interval_implies_a_ge_two (a : ℝ) (h : is_decreasing_in_interval a) : a ≥ 2 :=
sorry

end NUMINAMATH_GPT_decreasing_interval_implies_a_ge_two_l792_79238


namespace NUMINAMATH_GPT_pancakes_needed_l792_79299

theorem pancakes_needed (initial_pancakes : ℕ) (num_people : ℕ) (pancakes_left : ℕ) :
  initial_pancakes = 12 → num_people = 8 → pancakes_left = initial_pancakes - num_people →
  (num_people - pancakes_left) = 4 :=
by
  intros initial_pancakes_eq num_people_eq pancakes_left_eq
  sorry

end NUMINAMATH_GPT_pancakes_needed_l792_79299


namespace NUMINAMATH_GPT_abs_iff_neg_one_lt_x_lt_one_l792_79204

theorem abs_iff_neg_one_lt_x_lt_one (x : ℝ) : |x| < 1 ↔ -1 < x ∧ x < 1 :=
by
  sorry

end NUMINAMATH_GPT_abs_iff_neg_one_lt_x_lt_one_l792_79204


namespace NUMINAMATH_GPT_prime_q_exists_l792_79207

theorem prime_q_exists (p : ℕ) (pp : Nat.Prime p) : 
  ∃ q, Nat.Prime q ∧ (∀ n, n > 0 → ¬ q ∣ n ^ p - p) := 
sorry

end NUMINAMATH_GPT_prime_q_exists_l792_79207


namespace NUMINAMATH_GPT_ratio_income_to_expenditure_l792_79219

theorem ratio_income_to_expenditure (I E S : ℕ) 
  (h1 : I = 10000) 
  (h2 : S = 3000) 
  (h3 : S = I - E) : I / Nat.gcd I E = 10 ∧ E / Nat.gcd I E = 7 := by 
  sorry

end NUMINAMATH_GPT_ratio_income_to_expenditure_l792_79219


namespace NUMINAMATH_GPT_min_value_18_solve_inequality_l792_79231

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (1/a^3) + (1/b^3) + (1/c^3) + 27 * a * b * c

theorem min_value_18 (a b c : ℝ) (h : a > 0) (h' : b > 0) (h'' : c > 0) :
  min_value a b c ≥ 18 :=
by sorry

theorem solve_inequality (x : ℝ) :
  abs (x + 1) - 2 * x < 18 ↔ x > -(19/3) :=
by sorry

end NUMINAMATH_GPT_min_value_18_solve_inequality_l792_79231


namespace NUMINAMATH_GPT_prime_consecutive_fraction_equivalence_l792_79283

theorem prime_consecutive_fraction_equivalence (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hq_p_consec : p + 1 ≤ q ∧ Nat.Prime (p + 1) -> p + 1 = q) (hpq : p < q) (frac_eq : p / q = 4 / 5) :
  25 / 7 + (2 * q - p) / (2 * q + p) = 4 := sorry

end NUMINAMATH_GPT_prime_consecutive_fraction_equivalence_l792_79283


namespace NUMINAMATH_GPT_wizard_viable_combinations_l792_79285

def wizard_combination_problem : Prop :=
  let total_combinations := 4 * 6
  let incompatible_combinations := 3
  let viable_combinations := total_combinations - incompatible_combinations
  viable_combinations = 21

theorem wizard_viable_combinations : wizard_combination_problem :=
by
  sorry

end NUMINAMATH_GPT_wizard_viable_combinations_l792_79285


namespace NUMINAMATH_GPT_solution_inequality_1_solution_inequality_2_l792_79258

theorem solution_inequality_1 (x : ℝ) : -x^2 + 4*x + 5 < 0 ↔ (x < -1 ∨ x > 5) :=
by sorry

theorem solution_inequality_2 (x : ℝ) : 2*x^2 - 5*x + 2 ≤ 0 ↔ (1/2 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_GPT_solution_inequality_1_solution_inequality_2_l792_79258


namespace NUMINAMATH_GPT_fraction_simplification_l792_79278

theorem fraction_simplification :
  (3 / 7 + 4 / 5) / (5 / 12 + 2 / 3) = 516 / 455 := by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l792_79278


namespace NUMINAMATH_GPT_initial_volume_of_mixture_l792_79252

-- Define the initial condition volumes for p and q
def initial_volumes (x : ℕ) : ℕ × ℕ := (3 * x, 2 * x)

-- Define the final condition volumes for p and q after adding 2 liters of q
def final_volumes (x : ℕ) : ℕ × ℕ := (3 * x, 2 * x + 2)

-- Define the initial total volume of the mixture
def initial_volume (x : ℕ) : ℕ := 5 * x

-- The theorem stating the solution
theorem initial_volume_of_mixture (x : ℕ) (h : 3 * x / (2 * x + 2) = 5 / 4) : 5 * x = 25 := 
by sorry

end NUMINAMATH_GPT_initial_volume_of_mixture_l792_79252


namespace NUMINAMATH_GPT_upper_bound_of_third_inequality_l792_79241

variable (x : ℤ)

theorem upper_bound_of_third_inequality : (3 < x ∧ x < 10) →
                                          (5 < x ∧ x < 18) →
                                          (∃ n, n > x ∧ x > -2) →
                                          (0 < x ∧ x < 8) →
                                          (x + 1 < 9) →
                                          x < 8 :=
by { sorry }

end NUMINAMATH_GPT_upper_bound_of_third_inequality_l792_79241


namespace NUMINAMATH_GPT_product_of_two_numbers_l792_79288

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 + y^2 = 170) : x * y = -67 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l792_79288


namespace NUMINAMATH_GPT_triangle_angles_l792_79259

theorem triangle_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 45) : B = 90 ∧ C = 45 :=
sorry

end NUMINAMATH_GPT_triangle_angles_l792_79259


namespace NUMINAMATH_GPT_fraction_of_book_finished_l792_79212

variables (x y : ℝ)

theorem fraction_of_book_finished (h1 : x = y + 90) (h2 : x + y = 270) : x / 270 = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_fraction_of_book_finished_l792_79212


namespace NUMINAMATH_GPT_infinitely_many_positive_integers_l792_79281

theorem infinitely_many_positive_integers (k : ℕ) (m := 13 * k + 1) (h : m ≠ 8191) :
  8191 = 2 ^ 13 - 1 → ∃ (m : ℕ), ∀ k : ℕ, (13 * k + 1) ≠ 8191 ∧ ∃ (t : ℕ), (2 ^ (13 * k) - 1) = 8191 * m * t := by
  intros
  sorry

end NUMINAMATH_GPT_infinitely_many_positive_integers_l792_79281


namespace NUMINAMATH_GPT_ratio_a6_b6_l792_79237

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Define the nth term of sequence a
noncomputable def b_n (n : ℕ) : ℝ := sorry -- Define the nth term of sequence b
noncomputable def S_n (n : ℕ) : ℝ := sorry -- Define the sum of the first n terms of sequence a
noncomputable def T_n (n : ℕ) : ℝ := sorry -- Define the sum of the first n terms of sequence b

axiom condition (n : ℕ) : S_n n / T_n n = (2 * n) / (3 * n + 1)

theorem ratio_a6_b6 : a_n 6 / b_n 6 = 11 / 17 :=
by
  sorry

end NUMINAMATH_GPT_ratio_a6_b6_l792_79237


namespace NUMINAMATH_GPT_find_a_tangent_to_curve_l792_79275

theorem find_a_tangent_to_curve (a : ℝ) :
  (∃ (x₀ : ℝ), y = x - 1 ∧ y = e^(x + a) ∧ (e^(x₀ + a) = 1)) → a = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_tangent_to_curve_l792_79275


namespace NUMINAMATH_GPT_students_passed_in_dixon_lecture_l792_79227

theorem students_passed_in_dixon_lecture :
  let ratio_collins := 18 / 30
  let students_dixon := 45
  ∃ y, ratio_collins = y / students_dixon ∧ y = 27 :=
by
  sorry

end NUMINAMATH_GPT_students_passed_in_dixon_lecture_l792_79227


namespace NUMINAMATH_GPT_container_dimensions_l792_79249

theorem container_dimensions (a b c : ℝ) 
  (h1 : a * b * 16 = 2400)
  (h2 : a * c * 10 = 2400)
  (h3 : b * c * 9.6 = 2400) :
  a = 12 ∧ b = 12.5 ∧ c = 20 :=
by
  sorry

end NUMINAMATH_GPT_container_dimensions_l792_79249


namespace NUMINAMATH_GPT_digits_of_result_l792_79229

theorem digits_of_result 
  (u1 u2 t1 t2 h1 h2 : ℕ) 
  (hu_condition : u1 = u2 + 6)
  (units_column : u1 - u2 = 5)
  (tens_column : t1 - t2 = 9)
  (no_borrowing : u2 < u1) 
  : (h1, u1 - u2) = (4, 5) := 
sorry

end NUMINAMATH_GPT_digits_of_result_l792_79229


namespace NUMINAMATH_GPT_Sally_next_birthday_age_l792_79256

variables (a m s d : ℝ)

def Adam_older_than_Mary := a = 1.3 * m
def Mary_younger_than_Sally := m = 0.75 * s
def Sally_younger_than_Danielle := s = 0.8 * d
def Sum_ages := a + m + s + d = 60

theorem Sally_next_birthday_age (a m s d : ℝ) 
  (H1 : Adam_older_than_Mary a m)
  (H2 : Mary_younger_than_Sally m s)
  (H3 : Sally_younger_than_Danielle s d)
  (H4 : Sum_ages a m s d) : 
  s + 1 = 16 := 
by sorry

end NUMINAMATH_GPT_Sally_next_birthday_age_l792_79256


namespace NUMINAMATH_GPT_shaina_keeps_chocolate_l792_79255

theorem shaina_keeps_chocolate :
  let total_chocolate := (60 : ℚ) / 7
  let number_of_piles := 5
  let weight_per_pile := total_chocolate / number_of_piles
  let given_weight_back := (1 / 2) * weight_per_pile
  let kept_weight := weight_per_pile - given_weight_back
  kept_weight = 6 / 7 :=
by
  sorry

end NUMINAMATH_GPT_shaina_keeps_chocolate_l792_79255


namespace NUMINAMATH_GPT_ratio_of_distances_l792_79221

/-- 
  Given two points A and B moving along intersecting lines with constant,
  but different velocities v_A and v_B respectively, prove that there exists a 
  point P such that at any moment in time, the ratio of distances AP to BP equals 
  the ratio of their velocities.
-/
theorem ratio_of_distances (A B : ℝ → ℝ × ℝ) (v_A v_B : ℝ)
  (intersecting_lines : ∃ t, A t = B t)
  (diff_velocities : v_A ≠ v_B) :
  ∃ P : ℝ × ℝ, ∀ t, (dist P (A t) / dist P (B t)) = v_A / v_B := 
sorry

end NUMINAMATH_GPT_ratio_of_distances_l792_79221


namespace NUMINAMATH_GPT_arabella_total_learning_time_l792_79289

-- Define the conditions
def arabella_first_step_time := 30 -- in minutes
def arabella_second_step_time := arabella_first_step_time / 2 -- half the time of the first step
def arabella_third_step_time := arabella_first_step_time + arabella_second_step_time -- sum of the first and second steps

-- Define the total time spent
def arabella_total_time := arabella_first_step_time + arabella_second_step_time + arabella_third_step_time

-- The theorem to prove
theorem arabella_total_learning_time : arabella_total_time = 90 := 
  sorry

end NUMINAMATH_GPT_arabella_total_learning_time_l792_79289


namespace NUMINAMATH_GPT_geometric_sequence_sum_x_l792_79263

variable {α : Type*} [Field α]

theorem geometric_sequence_sum_x (a : ℕ → α) (S : ℕ → α) (x : α) 
  (h₁ : ∀ n, S n = x * (3:α)^n + 1)
  (h₂ : ∀ n, a n = S n - S (n - 1)) :
  ∃ x, x = -1 :=
by
  let a1 := S 1
  let a2 := S 2 - S 1
  let a3 := S 3 - S 2
  have ha1 : a1 = 3 * x + 1 := sorry
  have ha2 : a2 = 6 * x := sorry
  have ha3 : a3 = 18 * x := sorry
  have h_geom : (6 * x)^2 = (3 * x + 1) * 18 * x := sorry
  have h_solve : 18 * x * (x + 1) = 0 := sorry
  have h_x_neg1 : x = 0 ∨ x = -1 := sorry
  exact ⟨-1, sorry⟩

end NUMINAMATH_GPT_geometric_sequence_sum_x_l792_79263


namespace NUMINAMATH_GPT_fill_tank_time_l792_79244

theorem fill_tank_time (hA : ∀ t : Real, t > 0 → (t / 10) = 1) 
                       (hB : ∀ t : Real, t > 0 → (t / 20) = 1) 
                       (hC : ∀ t : Real, t > 0 → (t / 30) = 1) : 
                       (60 / 7 : Real) = 60 / 7 :=
by
    sorry

end NUMINAMATH_GPT_fill_tank_time_l792_79244


namespace NUMINAMATH_GPT_find_angle_C_find_area_triangle_l792_79210

open Real

-- Let the angles and sides of the triangle be defined as follows
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
axiom condition1 : (a^2 + b^2 - c^2) * (tan C) = sqrt 2 * a * b
axiom condition2 : c = 2
axiom condition3 : b = 2 * sqrt 2

-- Proof statements
theorem find_angle_C :
  C = pi / 4 ∨ C = 3 * pi / 4 :=
sorry

theorem find_area_triangle :
  C = pi / 4 → a = 2 → (1 / 2) * a * b * sin C = 2 :=
sorry

end NUMINAMATH_GPT_find_angle_C_find_area_triangle_l792_79210


namespace NUMINAMATH_GPT_fraction_problem_l792_79240

-- Definitions given in the conditions
variables {p q r s : ℚ}
variables (h₁ : p / q = 8)
variables (h₂ : r / q = 5)
variables (h₃ : r / s = 3 / 4)

-- Statement to prove
theorem fraction_problem : s / p = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_fraction_problem_l792_79240


namespace NUMINAMATH_GPT_find_constant_k_l792_79266

theorem find_constant_k 
  (k : ℝ)
  (h : ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) : 
  k = -16 :=
sorry

end NUMINAMATH_GPT_find_constant_k_l792_79266


namespace NUMINAMATH_GPT_green_balls_count_l792_79261

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def yellow_balls : ℕ := 2
def red_balls : ℕ := 15
def purple_balls : ℕ := 3
def probability_neither_red_nor_purple : ℝ := 0.7

theorem green_balls_count (G : ℕ) :
  (white_balls + G + yellow_balls) / total_balls = probability_neither_red_nor_purple →
  G = 18 := 
by
  sorry

end NUMINAMATH_GPT_green_balls_count_l792_79261


namespace NUMINAMATH_GPT_vershoks_per_arshin_l792_79243

theorem vershoks_per_arshin (plank_length_arshins : ℝ) (plank_width_vershoks : ℝ) 
    (room_side_length_arshins : ℝ) (total_planks : ℕ) (n : ℝ)
    (h1 : plank_length_arshins = 6) (h2 : plank_width_vershoks = 6)
    (h3 : room_side_length_arshins = 12) (h4 : total_planks = 64) 
    (h5 : (total_planks : ℝ) * (plank_length_arshins * (plank_width_vershoks / n)) = room_side_length_arshins^2) :
    n = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_vershoks_per_arshin_l792_79243


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l792_79203

noncomputable def volume_of_cylinder (height : ℝ) (circumference : ℝ) : ℝ := 
  let r := circumference / (2 * Real.pi)
  Real.pi * r^2 * height

theorem cylinder_volume_ratio :
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  max V1 V2 / min V1 V2 = 5 / 3 :=
by
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  have hV1 : V1 = 90 / Real.pi := sorry
  have hV2 : V2 = 150 / Real.pi := sorry
  sorry

end NUMINAMATH_GPT_cylinder_volume_ratio_l792_79203
