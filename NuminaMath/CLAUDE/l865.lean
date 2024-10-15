import Mathlib

namespace NUMINAMATH_CALUDE_remaining_distance_l865_86539

def total_journey : ℕ := 1200
def distance_driven : ℕ := 642

theorem remaining_distance : total_journey - distance_driven = 558 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l865_86539


namespace NUMINAMATH_CALUDE_probability_system_l865_86529

/-- Given a probability system with parameters p and q, prove that the probabilities x, y, and z satisfy specific relations. -/
theorem probability_system (p q x y z : ℝ) : 
  z = p * y + q * x → 
  x = p + q * x^2 → 
  y = q + p * y^2 → 
  x ≠ y → 
  p + q = 1 → 
  0 ≤ p ∧ p ≤ 1 → 
  0 ≤ q ∧ q ≤ 1 → 
  0 ≤ x ∧ x ≤ 1 → 
  0 ≤ y ∧ y ≤ 1 → 
  0 ≤ z ∧ z ≤ 1 → 
  x = 1 ∧ y = q / p ∧ z = 2 * q :=
by sorry

end NUMINAMATH_CALUDE_probability_system_l865_86529


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l865_86580

theorem gcd_of_three_numbers :
  Nat.gcd 105 (Nat.gcd 1001 2436) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l865_86580


namespace NUMINAMATH_CALUDE_log_two_plus_log_five_equals_one_l865_86516

theorem log_two_plus_log_five_equals_one : Real.log 2 + Real.log 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_two_plus_log_five_equals_one_l865_86516


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l865_86566

-- Problem 1
theorem problem_one : 
  Real.rpow 0.064 (-1/3) - Real.rpow (-1/8) 0 + Real.rpow 16 (3/4) + Real.rpow 0.25 (1/2) = 10 := by
  sorry

-- Problem 2
theorem problem_two :
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l865_86566


namespace NUMINAMATH_CALUDE_sum_less_than_one_l865_86523

theorem sum_less_than_one (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) : 
  x * (1 - y) + y * (1 - z) + z * (1 - x) < 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_less_than_one_l865_86523


namespace NUMINAMATH_CALUDE_two_m_minus_b_is_zero_l865_86500

/-- A line passing through two points (1, 3) and (-1, 1) -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The line passes through the points (1, 3) and (-1, 1) -/
def line_through_points (l : Line) : Prop :=
  3 = l.m * 1 + l.b ∧ 1 = l.m * (-1) + l.b

/-- Theorem stating that 2m - b = 0 for the line passing through (1, 3) and (-1, 1) -/
theorem two_m_minus_b_is_zero (l : Line) (h : line_through_points l) : 
  2 * l.m - l.b = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_m_minus_b_is_zero_l865_86500


namespace NUMINAMATH_CALUDE_polar_midpoint_specific_l865_86521

/-- The midpoint of a line segment in polar coordinates -/
def polar_midpoint (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: The midpoint of the line segment with endpoints (10, π/4) and (10, 3π/4) in polar coordinates -/
theorem polar_midpoint_specific :
  let (r, θ) := polar_midpoint 10 (π/4) 10 (3*π/4)
  r = 5 * Real.sqrt 2 ∧ θ = π/2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π :=
sorry

end NUMINAMATH_CALUDE_polar_midpoint_specific_l865_86521


namespace NUMINAMATH_CALUDE_line_x_intercept_l865_86502

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def xIntercept (l : Line) : ℝ := sorry

/-- The theorem stating that the line through (6, 2) and (2, 6) has x-intercept at x = 8 -/
theorem line_x_intercept :
  let l : Line := { x₁ := 6, y₁ := 2, x₂ := 2, y₂ := 6 }
  xIntercept l = 8 := by sorry

end NUMINAMATH_CALUDE_line_x_intercept_l865_86502


namespace NUMINAMATH_CALUDE_no_solution_implies_a_equals_six_l865_86549

theorem no_solution_implies_a_equals_six (a : ℝ) : 
  (∀ x y : ℝ, (x + 2*y = 4 ∧ 3*x + a*y = 6) → False) → a = 6 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_equals_six_l865_86549


namespace NUMINAMATH_CALUDE_multiply_negatives_l865_86552

theorem multiply_negatives : (-4 : ℚ) * (-(-(1/2))) = -2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_negatives_l865_86552


namespace NUMINAMATH_CALUDE_existence_of_non_square_product_l865_86514

theorem existence_of_non_square_product (d : ℕ) 
  (h_d_pos : d > 0) 
  (h_d_neq_2 : d ≠ 2) 
  (h_d_neq_5 : d ≠ 5) 
  (h_d_neq_13 : d ≠ 13) : 
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
               b ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
               a ≠ b ∧ 
               ¬∃ (k : ℕ), a * b - 1 = k * k :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_square_product_l865_86514


namespace NUMINAMATH_CALUDE_fruit_salad_count_l865_86501

/-- Given a fruit salad with red grapes, green grapes, and raspberries, 
    this theorem proves the total number of fruits in the salad. -/
theorem fruit_salad_count (red_grapes green_grapes raspberries : ℕ) : 
  red_grapes = 67 →
  red_grapes = 3 * green_grapes + 7 →
  raspberries = green_grapes - 5 →
  red_grapes + green_grapes + raspberries = 102 := by
  sorry

#check fruit_salad_count

end NUMINAMATH_CALUDE_fruit_salad_count_l865_86501


namespace NUMINAMATH_CALUDE_olivia_spent_89_dollars_l865_86589

/-- Calculates the amount spent at a supermarket given initial amount, amount collected, and amount left --/
def amount_spent (initial : ℕ) (collected : ℕ) (left : ℕ) : ℕ :=
  initial + collected - left

theorem olivia_spent_89_dollars : amount_spent 100 148 159 = 89 := by
  sorry

end NUMINAMATH_CALUDE_olivia_spent_89_dollars_l865_86589


namespace NUMINAMATH_CALUDE_cat_arrangements_eq_six_l865_86591

/-- The number of distinct arrangements of the letters in the word "CAT" -/
def cat_arrangements : ℕ :=
  Nat.factorial 3

theorem cat_arrangements_eq_six :
  cat_arrangements = 6 := by
  sorry

end NUMINAMATH_CALUDE_cat_arrangements_eq_six_l865_86591


namespace NUMINAMATH_CALUDE_remaining_wire_length_l865_86544

-- Define the length of the iron wire
def wire_length (a b : ℝ) : ℝ := 5 * a + 4 * b

-- Define the perimeter of the rectangle
def rectangle_perimeter (a b : ℝ) : ℝ := 2 * (a + b)

-- Theorem statement
theorem remaining_wire_length (a b : ℝ) :
  wire_length a b - rectangle_perimeter a b = 3 * a + 2 * b := by
  sorry

end NUMINAMATH_CALUDE_remaining_wire_length_l865_86544


namespace NUMINAMATH_CALUDE_parallel_line_equation_l865_86526

/-- Given a line with slope 2/3 and y-intercept 5, 
    prove that a parallel line 5 units away has the equation 
    y = (2/3)x + (5 ± (5√13)/3) -/
theorem parallel_line_equation (x y : ℝ) : 
  let given_line := λ x : ℝ => (2/3) * x + 5
  let distance := 5
  let parallel_line := λ x : ℝ => (2/3) * x + c
  let c_diff := |c - 5|
  (∀ x, |parallel_line x - given_line x| = distance) →
  (c = 5 + (5 * Real.sqrt 13) / 3 ∨ c = 5 - (5 * Real.sqrt 13) / 3) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l865_86526


namespace NUMINAMATH_CALUDE_right_quadrilateral_area_l865_86534

/-- A quadrilateral with right angles at B and D, diagonal AC = 3, and two sides with distinct integer lengths. -/
structure RightQuadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  right_angle_B : AB * BC = 0
  right_angle_D : CD * DA = 0
  diagonal_AC : AB^2 + BC^2 = 9
  distinct_integer_sides : ∃ (x y : ℕ), x ≠ y ∧ ((AB = x ∧ CD = y) ∨ (AB = x ∧ DA = y) ∨ (BC = x ∧ CD = y) ∨ (BC = x ∧ DA = y))

/-- The area of a RightQuadrilateral is √2 + √5. -/
theorem right_quadrilateral_area (q : RightQuadrilateral) : Real.sqrt 2 + Real.sqrt 5 = q.AB * q.BC / 2 + q.CD * q.DA / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_quadrilateral_area_l865_86534


namespace NUMINAMATH_CALUDE_store_transaction_result_l865_86543

theorem store_transaction_result : 
  let selling_price : ℝ := 960
  let profit_margin : ℝ := 0.2
  let cost_profit_item : ℝ := selling_price / (1 + profit_margin)
  let cost_loss_item : ℝ := selling_price / (1 - profit_margin)
  let total_cost : ℝ := cost_profit_item + cost_loss_item
  let total_revenue : ℝ := 2 * selling_price
  total_cost - total_revenue = 80
  := by sorry

end NUMINAMATH_CALUDE_store_transaction_result_l865_86543


namespace NUMINAMATH_CALUDE_geometric_progression_product_l865_86528

/-- For a geometric progression with n terms, first term a, and common ratio r,
    where P is the product of the n terms and T is the sum of the squares of the terms,
    the following equation holds. -/
theorem geometric_progression_product (n : ℕ) (a r : ℝ) (P T : ℝ) 
    (h1 : P = a^n * r^(n * (n - 1) / 2))
    (h2 : T = a^2 * (1 - r^(2*n)) / (1 - r^2)) 
    (h3 : r ≠ 1) : 
  P = T^(n/2) * ((1 - r^2) / (1 - r^(2*n)))^(n/2) * r^(n*(n-1)/2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_product_l865_86528


namespace NUMINAMATH_CALUDE_solution_set_theorem_a_range_theorem_l865_86531

/-- The function f(x) defined as |x| + 2|x-a| where a > 0 -/
def f (a : ℝ) (x : ℝ) : ℝ := |x| + 2 * |x - a|

/-- The solution set of f(x) ≤ 4 when a = 1 -/
def solution_set : Set ℝ := {x : ℝ | x ∈ Set.Icc (-2/3) 2}

/-- The range of a for which f(x) ≥ 4 always holds -/
def a_range : Set ℝ := {a : ℝ | a ∈ Set.Ici 4}

/-- Theorem stating the solution set of f(x) ≤ 4 when a = 1 -/
theorem solution_set_theorem :
  ∀ x : ℝ, f 1 x ≤ 4 ↔ x ∈ solution_set := by sorry

/-- Theorem stating the range of a for which f(x) ≥ 4 always holds -/
theorem a_range_theorem :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 4) ↔ a ∈ a_range := by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_a_range_theorem_l865_86531


namespace NUMINAMATH_CALUDE_cos_equality_317_degrees_l865_86511

theorem cos_equality_317_degrees (n : ℕ) (h1 : n ≤ 180) (h2 : Real.cos (n * π / 180) = Real.cos (317 * π / 180)) : n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_317_degrees_l865_86511


namespace NUMINAMATH_CALUDE_three_connected_iff_sequence_from_K4_l865_86525

/-- A simple graph. -/
structure Graph (V : Type*) where
  adj : V → V → Prop

/-- A graph is 3-connected if removing any two vertices does not disconnect the graph. -/
def ThreeConnected (G : Graph V) : Prop := sorry

/-- The complete graph on 4 vertices. -/
def K4 (V : Type*) : Graph V := sorry

/-- Remove an edge from a graph. -/
def removeEdge (G : Graph V) (e : V × V) : Graph V := sorry

/-- Theorem 3.2.3 (Tutte, 1966): A graph is 3-connected if and only if it can be constructed
    from K4 by adding edges one at a time. -/
theorem three_connected_iff_sequence_from_K4 {V : Type*} (G : Graph V) :
  ThreeConnected G ↔ 
  ∃ (n : ℕ) (sequence : ℕ → Graph V),
    (sequence 0 = K4 V) ∧
    (sequence n = G) ∧
    (∀ i < n, ∃ e, sequence i = removeEdge (sequence (i + 1)) e) :=
  sorry

end NUMINAMATH_CALUDE_three_connected_iff_sequence_from_K4_l865_86525


namespace NUMINAMATH_CALUDE_help_desk_services_percentage_l865_86570

theorem help_desk_services_percentage (total_hours software_hours help_user_hours : ℝ) 
  (h1 : total_hours = 68.33333333333333)
  (h2 : software_hours = 24)
  (h3 : help_user_hours = 17) :
  (total_hours - software_hours - help_user_hours) / total_hours * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_help_desk_services_percentage_l865_86570


namespace NUMINAMATH_CALUDE_card_difference_l865_86592

theorem card_difference (janet brenda mara : ℕ) : 
  janet > brenda →
  mara = 2 * janet →
  janet + brenda + mara = 211 →
  mara = 150 - 40 →
  janet - brenda = 9 := by
sorry

end NUMINAMATH_CALUDE_card_difference_l865_86592


namespace NUMINAMATH_CALUDE_f_of_three_equals_e_squared_l865_86556

theorem f_of_three_equals_e_squared 
  (f : ℝ → ℝ) 
  (h : ∀ x > 0, f (Real.log x + 1) = x) : 
  f 3 = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_three_equals_e_squared_l865_86556


namespace NUMINAMATH_CALUDE_problem_statement_l865_86555

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x + 1| + |x + a|

-- Define the theorem
theorem problem_statement 
  (a b : ℝ)
  (m n : ℝ)
  (h1 : ∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1)
  (h2 : m > 0)
  (h3 : n > 0)
  (h4 : 1/(2*m) + 2/n + 2*a = 0) :
  a = -1 ∧ 4*m^2 + n^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l865_86555


namespace NUMINAMATH_CALUDE_price_difference_is_500_l865_86520

/-- The price difference between enhanced and basic computers -/
def price_difference (total_basic_printer : ℝ) (price_basic : ℝ) : ℝ :=
  let price_printer := total_basic_printer - price_basic
  let price_enhanced := 8 * price_printer - price_printer
  price_enhanced - price_basic

/-- Theorem stating the price difference between enhanced and basic computers -/
theorem price_difference_is_500 :
  price_difference 2500 2125 = 500 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_is_500_l865_86520


namespace NUMINAMATH_CALUDE_certain_number_is_eleven_l865_86572

theorem certain_number_is_eleven (n : ℕ) : 
  (0 < n) → (n < 11) → (n = 1) → (∃ k : ℕ, 18888 - n = 11 * k) → 
  ∀ m : ℕ, (∃ j : ℕ, 18888 - n = m * j) → m = 11 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_is_eleven_l865_86572


namespace NUMINAMATH_CALUDE_tan_half_product_l865_86593

theorem tan_half_product (a b : Real) : 
  3 * (Real.cos a + Real.sin b) + 7 * (Real.cos a * Real.cos b + 1) = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = 3 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -3 := by
sorry

end NUMINAMATH_CALUDE_tan_half_product_l865_86593


namespace NUMINAMATH_CALUDE_rainy_days_probability_l865_86594

/-- The probability of rain on any given day -/
def p : ℚ := 1/5

/-- The number of days considered -/
def n : ℕ := 10

/-- The number of rainy days we're interested in -/
def k : ℕ := 3

/-- The probability of exactly k rainy days out of n days -/
def prob_k_rainy_days (p : ℚ) (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem rainy_days_probability : 
  prob_k_rainy_days p n k = 1966080/9765625 := by sorry

end NUMINAMATH_CALUDE_rainy_days_probability_l865_86594


namespace NUMINAMATH_CALUDE_total_spent_is_684_l865_86563

/-- Calculates the total amount spent by Christy and Tanya on face moisturizers and body lotions with discounts applied. -/
def total_spent (face_moisturizer_price : ℝ) (body_lotion_price : ℝ)
                (tanya_face : ℕ) (tanya_body : ℕ)
                (christy_face : ℕ) (christy_body : ℕ)
                (face_discount : ℝ) (body_discount : ℝ) : ℝ :=
  let tanya_total := (1 - face_discount) * (face_moisturizer_price * tanya_face) +
                     (1 - body_discount) * (body_lotion_price * tanya_body)
  let christy_total := (1 - face_discount) * (face_moisturizer_price * christy_face) +
                       (1 - body_discount) * (body_lotion_price * christy_body)
  tanya_total + christy_total

/-- Theorem stating that the total amount spent by Christy and Tanya is $684 under the given conditions. -/
theorem total_spent_is_684 :
  total_spent 50 60 2 4 3 5 0.1 0.15 = 684 ∧
  total_spent 50 60 2 4 3 5 0.1 0.15 = 2 * total_spent 50 60 2 4 2 4 0.1 0.15 :=
by sorry


end NUMINAMATH_CALUDE_total_spent_is_684_l865_86563


namespace NUMINAMATH_CALUDE_probability_at_least_one_boy_l865_86565

/-- The probability of selecting at least one boy from a group of 5 girls and 2 boys,
    given that girl A is already selected and a total of 3 people are to be selected. -/
theorem probability_at_least_one_boy (total_girls : ℕ) (total_boys : ℕ) 
  (h_girls : total_girls = 5) (h_boys : total_boys = 2) (selection_size : ℕ) 
  (h_selection : selection_size = 3) :
  (Nat.choose (total_boys + total_girls - 1) (selection_size - 1) - 
   Nat.choose (total_girls - 1) (selection_size - 1)) / 
  Nat.choose (total_boys + total_girls - 1) (selection_size - 1) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_boy_l865_86565


namespace NUMINAMATH_CALUDE_train_length_proof_l865_86562

/-- The length of two trains that pass each other under specific conditions -/
theorem train_length_proof (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 36) : 
  let L := (v_fast - v_slow) * t / (2 * 3600)
  L * 1000 = 50 := by sorry

end NUMINAMATH_CALUDE_train_length_proof_l865_86562


namespace NUMINAMATH_CALUDE_executive_committee_formation_l865_86548

theorem executive_committee_formation (total_members : ℕ) (committee_size : ℕ) (president : ℕ) :
  total_members = 30 →
  committee_size = 5 →
  president = 1 →
  Nat.choose (total_members - president) (committee_size - president) = 25839 :=
by sorry

end NUMINAMATH_CALUDE_executive_committee_formation_l865_86548


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l865_86597

theorem not_sufficient_not_necessary (a b : ℝ) :
  ¬(∀ a b : ℝ, 0 < a * b ∧ a * b < 1 → b < 1 / a) ∧
  ¬(∀ a b : ℝ, b < 1 / a → 0 < a * b ∧ a * b < 1) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l865_86597


namespace NUMINAMATH_CALUDE_subtract_seven_percent_l865_86598

theorem subtract_seven_percent (a : ℝ) : a - 0.07 * a = 0.93 * a := by
  sorry

end NUMINAMATH_CALUDE_subtract_seven_percent_l865_86598


namespace NUMINAMATH_CALUDE_third_game_difference_l865_86569

/-- The number of people who watched the second game -/
def second_game_viewers : ℕ := 80

/-- The number of people who watched the first game -/
def first_game_viewers : ℕ := second_game_viewers - 20

/-- The total number of people who watched the games last week -/
def last_week_total : ℕ := 200

/-- The total number of people who watched the games this week -/
def this_week_total : ℕ := last_week_total + 35

/-- The number of people who watched the third game -/
def third_game_viewers : ℕ := this_week_total - (first_game_viewers + second_game_viewers)

theorem third_game_difference : 
  third_game_viewers - second_game_viewers = 15 := by sorry

end NUMINAMATH_CALUDE_third_game_difference_l865_86569


namespace NUMINAMATH_CALUDE_multiplication_of_powers_l865_86519

theorem multiplication_of_powers (b : ℝ) : 3 * b^3 * (2 * b^2) = 6 * b^5 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_powers_l865_86519


namespace NUMINAMATH_CALUDE_correct_transformation_l865_86512

theorem correct_transformation (a x y : ℝ) : 
  ax = ay → 3 - ax = 3 - ay := by
sorry

end NUMINAMATH_CALUDE_correct_transformation_l865_86512


namespace NUMINAMATH_CALUDE_local_call_cost_is_five_cents_l865_86583

/-- Represents the cost structure and duration of Freddy's phone calls -/
structure CallData where
  local_duration : ℕ
  international_duration : ℕ
  international_cost_per_minute : ℕ
  total_cost_cents : ℕ

/-- Calculates the cost of a local call per minute -/
def local_call_cost_per_minute (data : CallData) : ℚ :=
  (data.total_cost_cents - data.international_duration * data.international_cost_per_minute) / data.local_duration

/-- Theorem stating that the local call cost per minute is 5 cents -/
theorem local_call_cost_is_five_cents (data : CallData) 
    (h1 : data.local_duration = 45)
    (h2 : data.international_duration = 31)
    (h3 : data.international_cost_per_minute = 25)
    (h4 : data.total_cost_cents = 1000) :
    local_call_cost_per_minute data = 5 := by
  sorry

#eval local_call_cost_per_minute {
  local_duration := 45,
  international_duration := 31,
  international_cost_per_minute := 25,
  total_cost_cents := 1000
}

end NUMINAMATH_CALUDE_local_call_cost_is_five_cents_l865_86583


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l865_86532

theorem inequality_and_equality_condition (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_not_all_zero : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) : 
  (2*x^2 - x + y + z)/(x + y^2 + z^2) + 
  (2*y^2 + x - y + z)/(x^2 + y + z^2) + 
  (2*z^2 + x + y - z)/(x^2 + y^2 + z) ≥ 3 ∧
  ((2*x^2 - x + y + z)/(x + y^2 + z^2) + 
   (2*y^2 + x - y + z)/(x^2 + y + z^2) + 
   (2*z^2 + x + y - z)/(x^2 + y^2 + z) = 3 ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l865_86532


namespace NUMINAMATH_CALUDE_opposite_numbers_quotient_l865_86547

theorem opposite_numbers_quotient (a b : ℝ) :
  a ≠ b → a = -b → a / b = -1 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_quotient_l865_86547


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l865_86579

theorem algebraic_expression_value (x : ℝ) :
  x^2 + x + 5 = 8 → 2*x^2 + 2*x - 4 = 2 := by
  sorry

#check algebraic_expression_value

end NUMINAMATH_CALUDE_algebraic_expression_value_l865_86579


namespace NUMINAMATH_CALUDE_youngsville_population_l865_86518

theorem youngsville_population (P : ℝ) : 
  (P * 1.25 * 0.6 = 513) → P = 684 := by
  sorry

end NUMINAMATH_CALUDE_youngsville_population_l865_86518


namespace NUMINAMATH_CALUDE_true_conjunction_with_negation_l865_86524

theorem true_conjunction_with_negation (p q : Prop) (hp : p) (hq : ¬q) : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_true_conjunction_with_negation_l865_86524


namespace NUMINAMATH_CALUDE_ratio_x_to_2y_l865_86584

theorem ratio_x_to_2y (x y : ℝ) (h : (7 * x + 8 * y) / (x - 2 * y) = 29) : 
  x / (2 * y) = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_to_2y_l865_86584


namespace NUMINAMATH_CALUDE_circle_no_intersection_with_axes_l865_86504

theorem circle_no_intersection_with_axes (k : ℝ) :
  (k > 0) →
  (∀ x y : ℝ, x^2 + y^2 - 2*k*x + 2*y + 2 = 0 → (x ≠ 0 ∧ y ≠ 0)) →
  k > 1 ∧ k < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_no_intersection_with_axes_l865_86504


namespace NUMINAMATH_CALUDE_product_of_solutions_l865_86561

theorem product_of_solutions (x₁ x₂ : ℝ) : 
  (|20 / x₁ + 1| = 4) → 
  (|20 / x₂ + 1| = 4) → 
  (x₁ ≠ x₂) →
  (x₁ * x₂ = -80 / 3) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l865_86561


namespace NUMINAMATH_CALUDE_workers_completion_time_l865_86596

/-- Given two workers who can each complete a task in 32 days, 
    prove they can complete the task together in 16 days -/
theorem workers_completion_time (work_rate_A work_rate_B : ℝ) : 
  work_rate_A = 1 / 32 →
  work_rate_B = 1 / 32 →
  1 / (work_rate_A + work_rate_B) = 16 := by
sorry

end NUMINAMATH_CALUDE_workers_completion_time_l865_86596


namespace NUMINAMATH_CALUDE_domain_of_y_l865_86568

-- Define the function f with domain [0,4]
def f : Set ℝ := Set.Icc 0 4

-- Define the function y
def y (f : Set ℝ) (x : ℝ) : Prop :=
  (x + 3 ∈ f) ∧ (x^2 ∈ f)

-- Theorem statement
theorem domain_of_y (f : Set ℝ) :
  f = Set.Icc 0 4 →
  {x : ℝ | y f x} = Set.Icc (-2) 1 := by
sorry

end NUMINAMATH_CALUDE_domain_of_y_l865_86568


namespace NUMINAMATH_CALUDE_problem_proof_l865_86541

theorem problem_proof : |Real.sqrt 3 - 2| - Real.sqrt ((-3)^2) + 2 * Real.sqrt 9 = 5 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l865_86541


namespace NUMINAMATH_CALUDE_min_distance_for_ten_trees_l865_86535

/-- Calculates the minimum distance to water trees in a row -/
def min_watering_distance (num_trees : ℕ) (tree_distance : ℕ) : ℕ :=
  let well_to_tree := tree_distance
  let tree_to_well := tree_distance
  let full_trips := (num_trees - 1) / 2
  let full_trip_distance := full_trips * (well_to_tree + tree_to_well)
  let remaining_trees := (num_trees - 1) % 2
  let last_trip_distance := remaining_trees * (well_to_tree + tree_to_well)
  full_trip_distance + last_trip_distance + (num_trees - 1) * tree_distance

theorem min_distance_for_ten_trees :
  min_watering_distance 10 10 = 410 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_for_ten_trees_l865_86535


namespace NUMINAMATH_CALUDE_twenty_percent_of_number_is_fifty_l865_86558

theorem twenty_percent_of_number_is_fifty (x : ℝ) : (20 / 100) * x = 50 → x = 250 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_of_number_is_fifty_l865_86558


namespace NUMINAMATH_CALUDE_ellipse_properties_l865_86513

/-- Properties of an ellipse with equation (x^2 / 25) + (y^2 / 9) = 1 -/
theorem ellipse_properties :
  let a : ℝ := 5
  let b : ℝ := 3
  let c : ℝ := 4
  let ellipse := fun (x y : ℝ) ↦ x^2 / 25 + y^2 / 9 = 1
  -- Eccentricity
  (c / a = 0.8) ∧
  -- Foci
  (ellipse (-c) 0 ∧ ellipse c 0) ∧
  -- Vertices
  (ellipse (-a) 0 ∧ ellipse a 0) := by
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l865_86513


namespace NUMINAMATH_CALUDE_total_cookies_and_brownies_l865_86546

theorem total_cookies_and_brownies :
  let cookie_bags : ℕ := 272
  let cookies_per_bag : ℕ := 45
  let brownie_bags : ℕ := 158
  let brownies_per_bag : ℕ := 32
  cookie_bags * cookies_per_bag + brownie_bags * brownies_per_bag = 17296 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_and_brownies_l865_86546


namespace NUMINAMATH_CALUDE_partition_M_theorem_l865_86575

/-- The set M containing elements from 1 to 12 -/
def M : Finset ℕ := Finset.range 12

/-- Predicate to check if a set is a valid partition of M -/
def is_valid_partition (A B C : Finset ℕ) : Prop :=
  A ∪ B ∪ C = M ∧ A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
  A.card = 4 ∧ B.card = 4 ∧ C.card = 4

/-- Predicate to check if C satisfies the ordering condition -/
def C_ordered (C : Finset ℕ) : Prop :=
  ∃ c₁ c₂ c₃ c₄, C = {c₁, c₂, c₃, c₄} ∧ c₁ < c₂ ∧ c₂ < c₃ ∧ c₃ < c₄

/-- Predicate to check if A, B, and C satisfy the sum condition -/
def sum_condition (A B C : Finset ℕ) : Prop :=
  ∃ a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ c₁ c₂ c₃ c₄,
    A = {a₁, a₂, a₃, a₄} ∧ B = {b₁, b₂, b₃, b₄} ∧ C = {c₁, c₂, c₃, c₄} ∧
    a₁ + b₁ = c₁ ∧ a₂ + b₂ = c₂ ∧ a₃ + b₃ = c₃ ∧ a₄ + b₄ = c₄

theorem partition_M_theorem :
  ∀ A B C : Finset ℕ,
    is_valid_partition A B C →
    C_ordered C →
    sum_condition A B C →
    C = {8, 9, 10, 12} ∨ C = {7, 9, 11, 12} ∨ C = {6, 10, 11, 12} :=
sorry

end NUMINAMATH_CALUDE_partition_M_theorem_l865_86575


namespace NUMINAMATH_CALUDE_additive_inverses_solution_l865_86571

theorem additive_inverses_solution (x : ℝ) : (6 * x - 12) + (4 + 2 * x) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverses_solution_l865_86571


namespace NUMINAMATH_CALUDE_destiny_snack_bags_l865_86506

theorem destiny_snack_bags (chocolate_bars : Nat) (cookies : Nat) 
  (h1 : chocolate_bars = 18) (h2 : cookies = 12) :
  Nat.gcd chocolate_bars cookies = 6 := by
  sorry

end NUMINAMATH_CALUDE_destiny_snack_bags_l865_86506


namespace NUMINAMATH_CALUDE_intersection_sum_l865_86599

theorem intersection_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + 7 ∧ y = 4 * x + b → x = 8 ∧ y = 11) → 
  b + m = -20.5 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l865_86599


namespace NUMINAMATH_CALUDE_geometric_sequence_incorrect_statement_l865_86553

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_incorrect_statement
  (a : ℕ → ℝ) (q : ℝ) (h1 : geometric_sequence a q) (h2 : q ≠ 1) :
  ¬(a 2 > a 1 → ∀ n : ℕ, a (n + 1) > a n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_incorrect_statement_l865_86553


namespace NUMINAMATH_CALUDE_fraction_value_l865_86590

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l865_86590


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l865_86582

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let sideLengthOfCube := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / sideLengthOfCube) * (box.width / sideLengthOfCube) * (box.depth / sideLengthOfCube)

/-- The theorem stating that for a box with given dimensions, 
    the smallest number of identical cubes that can fill it completely is 84 -/
theorem smallest_number_of_cubes_for_given_box : 
  smallestNumberOfCubes ⟨49, 42, 14⟩ = 84 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l865_86582


namespace NUMINAMATH_CALUDE_law_of_sines_symmetry_l865_86522

/-- The Law of Sines for a triangle ABC with sides a, b, c and angles A, B, C -/
def law_of_sines (a b c : ℝ) (A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

/-- A property representing symmetry in mathematical expressions -/
def has_symmetry (P : Prop) : Prop :=
  -- This is a placeholder definition. In a real implementation, 
  -- this would need to be defined based on specific criteria for symmetry.
  true

/-- Theorem stating that the Law of Sines exhibits mathematical symmetry -/
theorem law_of_sines_symmetry (a b c : ℝ) (A B C : ℝ) :
  has_symmetry (law_of_sines a b c A B C) :=
sorry

end NUMINAMATH_CALUDE_law_of_sines_symmetry_l865_86522


namespace NUMINAMATH_CALUDE_tangent_circles_sum_l865_86573

-- Define the circles w1 and w2
def w1 (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 20*y - 75 = 0
def w2 (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 20*y + 115 = 0

-- Define the condition for a circle to be externally tangent to w1
def externally_tangent_w1 (cx cy r : ℝ) : Prop :=
  (cx + 4)^2 + (cy - 10)^2 = (r + 11)^2

-- Define the condition for a circle to be internally tangent to w2
def internally_tangent_w2 (cx cy r : ℝ) : Prop :=
  (cx - 6)^2 + (cy - 10)^2 = (7 - r)^2

-- Define the theorem
theorem tangent_circles_sum (p q : ℕ) (h_coprime : Nat.Coprime p q) :
  (∃ (m : ℝ), m > 0 ∧ m^2 = p / q ∧
    (∃ (cx cy r : ℝ), cy = m * cx ∧
      externally_tangent_w1 cx cy r ∧
      internally_tangent_w2 cx cy r) ∧
    (∀ (a : ℝ), a > 0 → a < m →
      ¬∃ (cx cy r : ℝ), cy = a * cx ∧
        externally_tangent_w1 cx cy r ∧
        internally_tangent_w2 cx cy r)) →
  p + q = 181 := by sorry

end NUMINAMATH_CALUDE_tangent_circles_sum_l865_86573


namespace NUMINAMATH_CALUDE_water_speed_calculation_l865_86530

theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  still_water_speed = 12 →
  distance = 8 →
  time = 4 →
  ∃ water_speed : ℝ, water_speed = still_water_speed - (distance / time) :=
by
  sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l865_86530


namespace NUMINAMATH_CALUDE_distance_between_towns_l865_86559

/-- The distance between two towns given train speeds and meeting time -/
theorem distance_between_towns (express_speed : ℝ) (speed_difference : ℝ) (meeting_time : ℝ) : 
  express_speed = 80 →
  speed_difference = 30 →
  meeting_time = 3 →
  (express_speed + (express_speed - speed_difference)) * meeting_time = 390 := by
sorry

end NUMINAMATH_CALUDE_distance_between_towns_l865_86559


namespace NUMINAMATH_CALUDE_triangle_area_is_24_l865_86545

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (0, 0)
def vertex2 : ℝ × ℝ := (0, 6)
def vertex3 : ℝ × ℝ := (8, 10)

-- Define the triangle area calculation function
def triangleArea (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  let x1 := v1.1
  let y1 := v1.2
  let x2 := v2.1
  let y2 := v2.2
  let x3 := v3.1
  let y3 := v3.2
  0.5 * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- State the theorem
theorem triangle_area_is_24 :
  triangleArea vertex1 vertex2 vertex3 = 24 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_is_24_l865_86545


namespace NUMINAMATH_CALUDE_min_a2_plus_b2_l865_86540

-- Define the quadratic function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (2*b + 1) * x - a - 2

-- State the theorem
theorem min_a2_plus_b2 (a b : ℝ) (ha : a ≠ 0) 
  (hroot : ∃ x ∈ Set.Icc 3 4, f a b x = 0) : 
  ∃ min_val : ℝ, (∀ a' b' : ℝ, a'^2 + b'^2 ≥ min_val) ∧ 
  (∃ a₀ b₀ : ℝ, a₀^2 + b₀^2 = min_val) ∧ min_val = 1/100 :=
sorry

end NUMINAMATH_CALUDE_min_a2_plus_b2_l865_86540


namespace NUMINAMATH_CALUDE_solve_linear_equation_l865_86576

theorem solve_linear_equation : ∃ x : ℝ, 4 * x - 5 = 3 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l865_86576


namespace NUMINAMATH_CALUDE_total_orange_balloons_l865_86574

-- Define the initial number of orange balloons
def initial_orange_balloons : ℝ := 9.0

-- Define the number of orange balloons found
def found_orange_balloons : ℝ := 2.0

-- Theorem to prove
theorem total_orange_balloons :
  initial_orange_balloons + found_orange_balloons = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_total_orange_balloons_l865_86574


namespace NUMINAMATH_CALUDE_sample_size_is_30_l865_86551

/-- Represents the company's employee data and sampling information -/
structure CompanyData where
  total_employees : ℕ
  young_employees : ℕ
  sample_young : ℕ
  h_young_le_total : young_employees ≤ total_employees

/-- Calculates the sample size based on stratified sampling -/
def calculate_sample_size (data : CompanyData) : ℕ :=
  (data.sample_young * data.total_employees) / data.young_employees

/-- Proves that the sample size is 30 given the specific company data -/
theorem sample_size_is_30 (data : CompanyData) 
  (h_total : data.total_employees = 900)
  (h_young : data.young_employees = 450)
  (h_sample_young : data.sample_young = 15) :
  calculate_sample_size data = 30 := by
  sorry

#eval calculate_sample_size ⟨900, 450, 15, by norm_num⟩

end NUMINAMATH_CALUDE_sample_size_is_30_l865_86551


namespace NUMINAMATH_CALUDE_red_bead_count_l865_86554

/-- Represents a necklace with blue and red beads. -/
structure Necklace where
  blue_count : Nat
  red_count : Nat
  is_valid : Bool

/-- Checks if a necklace satisfies the given conditions. -/
def is_valid_necklace (n : Necklace) : Prop :=
  n.blue_count = 30 ∧
  n.is_valid = true ∧
  n.red_count > 0 ∧
  n.red_count % 2 = 0 ∧
  n.red_count = 2 * n.blue_count

theorem red_bead_count (n : Necklace) :
  is_valid_necklace n → n.red_count = 60 := by
  sorry

#check red_bead_count

end NUMINAMATH_CALUDE_red_bead_count_l865_86554


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l865_86509

theorem ball_distribution_theorem (num_colors num_students total_balls min_balls_per_color min_balls_per_box : ℕ) 
  (h1 : num_colors = 20)
  (h2 : num_students = 20)
  (h3 : total_balls = 800)
  (h4 : min_balls_per_color ≥ 10)
  (h5 : min_balls_per_box ≥ 10) :
  ∃ (balls_per_student : ℕ), 
    balls_per_student * num_students = total_balls ∧ 
    ∃ (num_boxes : ℕ), 
      num_boxes % num_students = 0 ∧ 
      num_boxes * min_balls_per_box ≤ total_balls ∧
      (num_boxes / num_students) * min_balls_per_box = balls_per_student :=
by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l865_86509


namespace NUMINAMATH_CALUDE_lagrange_interpolation_identities_l865_86577

theorem lagrange_interpolation_identities 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hca : c ≠ a) : 
  (1 / ((a - b) * (a - c)) + 1 / ((b - c) * (b - a)) + 1 / ((c - a) * (c - b)) = 0) ∧
  (a / ((a - b) * (a - c)) + b / ((b - c) * (b - a)) + c / ((c - a) * (c - b)) = 0) ∧
  (a^2 / ((a - b) * (a - c)) + b^2 / ((b - c) * (b - a)) + c^2 / ((c - a) * (c - b)) = 1) :=
by sorry

end NUMINAMATH_CALUDE_lagrange_interpolation_identities_l865_86577


namespace NUMINAMATH_CALUDE_product_of_x_values_l865_86508

theorem product_of_x_values (x : ℝ) : 
  (|10 / x - 4| = 3) → 
  (∃ y : ℝ, (|10 / y - 4| = 3) ∧ x * y = 100 / 7) :=
by sorry

end NUMINAMATH_CALUDE_product_of_x_values_l865_86508


namespace NUMINAMATH_CALUDE_intersection_A_B_l865_86533

def A : Set ℝ := {x | ∃ k : ℤ, 2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi}

def B : Set ℝ := {x | -5 ≤ x ∧ x < 4}

theorem intersection_A_B : A ∩ B = {x | -Real.pi < x ∧ x < 0 ∨ Real.pi < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l865_86533


namespace NUMINAMATH_CALUDE_intersection_points_count_l865_86517

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Predicate to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Predicate to check if a line intersects a circle -/
def intersectsCircle (l : Line) (c : Circle) : Prop :=
  sorry

/-- Predicate to check if two lines intersect -/
def intersectLines (l1 l2 : Line) : Prop :=
  sorry

/-- Function to count intersection points between a line and a circle -/
def countIntersections (l : Line) (c : Circle) : ℕ :=
  sorry

/-- Main theorem -/
theorem intersection_points_count 
  (c : Circle) (l1 l2 : Line) 
  (h1 : isTangent l1 c)
  (h2 : intersectsCircle l2 c)
  (h3 : ¬ isTangent l2 c)
  (h4 : intersectLines l1 l2) :
  countIntersections l1 c + countIntersections l2 c = 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_count_l865_86517


namespace NUMINAMATH_CALUDE_lemonade_sales_difference_l865_86505

/-- 
Given:
- x: number of glasses of plain lemonade sold
- y: number of glasses of strawberry lemonade sold
- p: price of each glass of plain lemonade
- s: price of each glass of strawberry lemonade
- The total amount from plain lemonade is 1.5 times the total amount from strawberry lemonade

Prove that the difference between the total amount made from plain lemonade and 
strawberry lemonade is equal to 0.5 * (y * s)
-/
theorem lemonade_sales_difference 
  (x y p s : ℝ) 
  (h : x * p = 1.5 * (y * s)) : 
  x * p - y * s = 0.5 * (y * s) := by
  sorry


end NUMINAMATH_CALUDE_lemonade_sales_difference_l865_86505


namespace NUMINAMATH_CALUDE_min_value_of_sum_squares_l865_86581

theorem min_value_of_sum_squares (a b c m : ℝ) 
  (sum_eq_one : a + b + c = 1) 
  (m_def : m = a^2 + b^2 + c^2) : 
  m ≥ 1/3 ∧ ∃ (a b c : ℝ), a + b + c = 1 ∧ a^2 + b^2 + c^2 = 1/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_squares_l865_86581


namespace NUMINAMATH_CALUDE_commission_change_point_l865_86503

/-- The sales amount where the commission rate changes -/
def X : ℝ := 10000

/-- The total sales amount -/
def total_sales : ℝ := 32500

/-- The amount remitted to the parent company -/
def remitted_amount : ℝ := 31100

/-- The commission rate for sales up to X -/
def commission_rate_low : ℝ := 0.05

/-- The commission rate for sales exceeding X -/
def commission_rate_high : ℝ := 0.04

theorem commission_change_point :
  X = 10000 ∧
  total_sales = 32500 ∧
  remitted_amount = 31100 ∧
  commission_rate_low = 0.05 ∧
  commission_rate_high = 0.04 ∧
  remitted_amount = total_sales - (commission_rate_low * X + commission_rate_high * (total_sales - X)) :=
by sorry

end NUMINAMATH_CALUDE_commission_change_point_l865_86503


namespace NUMINAMATH_CALUDE_potato_fetch_time_l865_86550

-- Define the constants
def football_fields : ℕ := 6
def yards_per_field : ℕ := 200
def feet_per_yard : ℕ := 3
def dog_speed : ℕ := 400  -- feet per minute

-- Define the theorem
theorem potato_fetch_time :
  let total_distance : ℕ := football_fields * yards_per_field * feet_per_yard
  let fetch_time : ℕ := total_distance / dog_speed
  fetch_time = 9 := by sorry

end NUMINAMATH_CALUDE_potato_fetch_time_l865_86550


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l865_86585

theorem merchant_pricing_strategy 
  (list_price : ℝ) 
  (cost_price : ℝ) 
  (marked_price : ℝ) 
  (selling_price : ℝ) 
  (h1 : cost_price = 0.7 * list_price)  -- 30% discount on purchase
  (h2 : selling_price = 0.8 * marked_price)  -- 20% discount on sale
  (h3 : cost_price = 0.7 * selling_price)  -- 30% profit on selling price
  : marked_price = 1.25 * list_price :=
by sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l865_86585


namespace NUMINAMATH_CALUDE_average_velocity_first_30_seconds_l865_86586

-- Define the velocity function
def v (t : ℝ) : ℝ := t^2 - 3*t + 8

-- Define the time interval
def t_start : ℝ := 0
def t_end : ℝ := 30

-- Theorem statement
theorem average_velocity_first_30_seconds :
  (∫ t in t_start..t_end, v t) / (t_end - t_start) = 263 := by
  sorry

end NUMINAMATH_CALUDE_average_velocity_first_30_seconds_l865_86586


namespace NUMINAMATH_CALUDE_no_positive_and_negative_rational_l865_86538

theorem no_positive_and_negative_rational : ¬∃ (q : ℚ), q > 0 ∧ q < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_and_negative_rational_l865_86538


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l865_86578

theorem imaginary_part_of_z : 
  let z : ℂ := (1 - Complex.I) / Complex.I
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l865_86578


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l865_86564

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → a^2016 + b^2017 = 1 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l865_86564


namespace NUMINAMATH_CALUDE_vector_relationships_l865_86588

/-- Given two vectors OA and OB in R², this theorem states the value of m in OB
    when OA is perpendicular to OB and when OA is parallel to OB. -/
theorem vector_relationships (OA OB : ℝ × ℝ) (m : ℝ) : 
  OA = (-1, 2) → OB = (3, m) →
  ((OA.1 * OB.1 + OA.2 * OB.2 = 0 → m = 3/2) ∧
   (∃ k : ℝ, OB = (k * OA.1, k * OA.2) → m = -6)) := by
  sorry

end NUMINAMATH_CALUDE_vector_relationships_l865_86588


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l865_86527

theorem inequality_and_equality_conditions (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 2) :
  3*x + 8*x*y + 16*x*y*z ≤ 12 ∧ 
  (3*x + 8*x*y + 16*x*y*z = 12 ↔ x = 1 ∧ y = 3/4 ∧ z = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l865_86527


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l865_86537

def kilowatt_hours : ℝ := 448000

theorem scientific_notation_equivalence : 
  kilowatt_hours = 4.48 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l865_86537


namespace NUMINAMATH_CALUDE_exactly_two_statements_true_l865_86595

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry about y-axis
def symmetricAboutYAxis (a b : Point2D) : Prop :=
  a.x = -b.x ∧ a.y = b.y

-- Define symmetry about x-axis
def symmetricAboutXAxis (a b : Point2D) : Prop :=
  a.x = b.x ∧ a.y = -b.y

-- Define the four statements
def statement1 (a b : Point2D) : Prop :=
  symmetricAboutYAxis a b → a.y = b.y

def statement2 (a b : Point2D) : Prop :=
  a.y = b.y → symmetricAboutYAxis a b

def statement3 (a b : Point2D) : Prop :=
  a.x = b.x → symmetricAboutXAxis a b

def statement4 (a b : Point2D) : Prop :=
  symmetricAboutXAxis a b → a.x = b.x

-- Theorem stating that exactly two of the statements are true
theorem exactly_two_statements_true :
  ∃ (a b : Point2D),
    (statement1 a b ∧ ¬statement2 a b ∧ ¬statement3 a b ∧ statement4 a b) :=
  sorry

end NUMINAMATH_CALUDE_exactly_two_statements_true_l865_86595


namespace NUMINAMATH_CALUDE_area_between_circles_l865_86515

theorem area_between_circles (r : ℝ) (R : ℝ) : 
  r = 3 →  -- radius of smaller circle
  R = 3 * r →  -- radius of larger circle is three times the smaller
  π * R^2 - π * r^2 = 72 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_l865_86515


namespace NUMINAMATH_CALUDE_ellipse_focal_chord_area_l865_86567

/-- Given an ellipse with equation x²/4 + y²/m = 1 (m > 0), where the focal chord F₁F₂ is the diameter
    of a circle intersecting the ellipse at point P in the first quadrant, if the area of triangle PF₁F₂
    is 1, then m = 1. -/
theorem ellipse_focal_chord_area (m : ℝ) (x y : ℝ) (F₁ F₂ P : ℝ × ℝ) : 
  m > 0 → 
  x^2 / 4 + y^2 / m = 1 →
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 16 →  -- F₁F₂ is diameter of circle with radius 2
  P.1^2 / 4 + P.2^2 / m = 1 →  -- P is on the ellipse
  P.1 ≥ 0 ∧ P.2 ≥ 0 →  -- P is in the first quadrant
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = 16 →  -- P is on the circle
  abs ((P.1 - F₁.1) * (P.2 - F₂.2) - (P.2 - F₁.2) * (P.1 - F₂.1)) / 2 = 1 →  -- Area of triangle PF₁F₂ is 1
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focal_chord_area_l865_86567


namespace NUMINAMATH_CALUDE_company_females_count_l865_86507

theorem company_females_count (total_employees : ℕ) 
  (advanced_degrees : ℕ) (males_college_only : ℕ) (females_advanced : ℕ) 
  (h1 : total_employees = 148)
  (h2 : advanced_degrees = 78)
  (h3 : males_college_only = 31)
  (h4 : females_advanced = 53) :
  total_employees - advanced_degrees - males_college_only + females_advanced = 92 :=
by sorry

end NUMINAMATH_CALUDE_company_females_count_l865_86507


namespace NUMINAMATH_CALUDE_least_number_with_remainder_least_number_is_205517_least_number_unique_l865_86560

theorem least_number_with_remainder (n : ℕ) : 
  (n % 45 = 2 ∧ n % 59 = 2 ∧ n % 77 = 2) → n ≥ 205517 :=
by
  sorry

theorem least_number_is_205517 : 
  205517 % 45 = 2 ∧ 205517 % 59 = 2 ∧ 205517 % 77 = 2 :=
by
  sorry

theorem least_number_unique : 
  ∃! n : ℕ, (n % 45 = 2 ∧ n % 59 = 2 ∧ n % 77 = 2) ∧ 
  ∀ m : ℕ, (m % 45 = 2 ∧ m % 59 = 2 ∧ m % 77 = 2) → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_least_number_is_205517_least_number_unique_l865_86560


namespace NUMINAMATH_CALUDE_x_value_l865_86510

theorem x_value (x y : ℚ) (h1 : x / y = 7 / 3) (h2 : y = 21) : x = 49 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l865_86510


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_l865_86587

/-- Given a point P(4, -3) on the terminal side of angle α, prove that cos(α) = 4/5 -/
theorem cos_alpha_for_point (α : Real) : 
  (∃ (P : Real × Real), P = (4, -3) ∧ P.1 = 4 * Real.cos α ∧ P.2 = 4 * Real.sin α) →
  Real.cos α = 4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_l865_86587


namespace NUMINAMATH_CALUDE_smartphone_price_decrease_l865_86557

/-- The average percentage decrease in price for a smartphone that underwent two price reductions -/
theorem smartphone_price_decrease (original_price final_price : ℝ) 
  (h1 : original_price = 2000)
  (h2 : final_price = 1280) : 
  (original_price - final_price) / original_price / 2 * 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_decrease_l865_86557


namespace NUMINAMATH_CALUDE_initial_peanuts_l865_86536

theorem initial_peanuts (added : ℕ) (final : ℕ) (h1 : added = 6) (h2 : final = 10) :
  final - added = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_peanuts_l865_86536


namespace NUMINAMATH_CALUDE_negation_of_proposition_quadratic_inequality_negation_l865_86542

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬∀ x, p x) ↔ ∃ x, ¬(p x) := by sorry

theorem quadratic_inequality_negation :
  (¬∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_quadratic_inequality_negation_l865_86542
