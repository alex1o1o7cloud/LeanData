import Mathlib

namespace NUMINAMATH_CALUDE_vasya_arrives_first_l1041_104109

/-- Represents the relative step length and step count of two people walking to school -/
structure WalkingData where
  vasya_step_length : ℝ
  petya_step_length : ℝ
  vasya_step_count : ℝ
  petya_step_count : ℝ

/-- The conditions of the problem -/
def walking_conditions (data : WalkingData) : Prop :=
  data.vasya_step_length > 0 ∧
  data.petya_step_length = 0.75 * data.vasya_step_length ∧
  data.petya_step_count = 1.25 * data.vasya_step_count

/-- Theorem stating that Vasya travels further in the same time -/
theorem vasya_arrives_first (data : WalkingData) 
  (h : walking_conditions data) : 
  data.vasya_step_length * data.vasya_step_count > 
  data.petya_step_length * data.petya_step_count := by
  sorry

#check vasya_arrives_first

end NUMINAMATH_CALUDE_vasya_arrives_first_l1041_104109


namespace NUMINAMATH_CALUDE_element_in_set_l1041_104179

def M : Set (ℕ × ℕ) := {(1, 2)}

theorem element_in_set : (1, 2) ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l1041_104179


namespace NUMINAMATH_CALUDE_inequality_proof_l1041_104111

theorem inequality_proof (x : ℝ) : 3 * (2 * x - 1) - 2 * (x + 1) ≤ 1 → x ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1041_104111


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1041_104149

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def has_equal_intercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line2D),
    (point_on_line ⟨1, 2⟩ l1 ∧ has_equal_intercepts l1) ∧
    (point_on_line ⟨1, 2⟩ l2 ∧ has_equal_intercepts l2) ∧
    ((l1.a = 1 ∧ l1.b = 1 ∧ l1.c = -3) ∨ (l2.a = 2 ∧ l2.b = -1 ∧ l2.c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1041_104149


namespace NUMINAMATH_CALUDE_shirts_before_buying_l1041_104112

/-- Given that Sarah bought new shirts and now owns a total number of shirts,
    prove that the number of shirts she had before buying new ones is correct. -/
theorem shirts_before_buying (new_shirts total_shirts : ℕ) 
  (h1 : new_shirts = 8)
  (h2 : total_shirts = 17) :
  total_shirts - new_shirts = 9 := by
  sorry

end NUMINAMATH_CALUDE_shirts_before_buying_l1041_104112


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l1041_104155

theorem smallest_k_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 13 = 1 ∧
  k % 8 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 8 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l1041_104155


namespace NUMINAMATH_CALUDE_x_value_proof_l1041_104119

theorem x_value_proof (x y : ℝ) 
  (eq1 : 3 * x - 2 * y = 7)
  (eq2 : x^2 + 3 * y = 17) : 
  x = 3.5 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l1041_104119


namespace NUMINAMATH_CALUDE_sum_greater_than_product_iff_one_l1041_104142

theorem sum_greater_than_product_iff_one (m n : ℕ+) :
  m + n > m * n ↔ m = 1 ∨ n = 1 := by sorry

end NUMINAMATH_CALUDE_sum_greater_than_product_iff_one_l1041_104142


namespace NUMINAMATH_CALUDE_share_purchase_price_l1041_104115

/-- Given a company paying dividend and an investor's return on investment,
    calculate the purchase price of shares. -/
theorem share_purchase_price
  (dividend_rate : ℝ)
  (face_value : ℝ)
  (roi : ℝ)
  (h1 : dividend_rate = 15.5)
  (h2 : face_value = 50)
  (h3 : roi = 25) :
  (dividend_rate / 100 * face_value) / (roi / 100) = 31 :=
sorry

end NUMINAMATH_CALUDE_share_purchase_price_l1041_104115


namespace NUMINAMATH_CALUDE_paper_folding_volumes_l1041_104187

/-- Given a square paper with side length 1, prove the volume of a cone and max volume of a rectangular prism --/
theorem paper_folding_volumes (ε : ℝ) (hε : ε = 0.0001) :
  ∃ (V_cone V_prism : ℝ),
    (abs (V_cone - (π / 6)) < ε) ∧
    (abs (V_prism - (1 / (3 * Real.sqrt 3))) < ε) ∧
    (∀ (a b c : ℝ), 2 * (a * b + b * c + c * a) = 1 → a * b * c ≤ V_prism) := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_volumes_l1041_104187


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l1041_104133

/-- A regular decagon -/
structure RegularDecagon where
  -- Add necessary properties here

/-- The number of diagonals in a regular decagon -/
def num_diagonals (d : RegularDecagon) : ℕ := 35

/-- The number of ways to choose 3 diagonals from a regular decagon -/
def num_diagonal_trios (d : RegularDecagon) : ℕ := 6545

/-- The number of ways to choose 5 points from a regular decagon such that no three points are consecutive -/
def num_valid_point_sets (d : RegularDecagon) : ℕ := 252

/-- The probability that three randomly chosen diagonals of a regular decagon intersect inside the decagon -/
def intersection_probability (d : RegularDecagon) : ℚ :=
  num_valid_point_sets d / num_diagonal_trios d

theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  intersection_probability d = 252 / 6545 := by
  sorry


end NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l1041_104133


namespace NUMINAMATH_CALUDE_target_probabilities_l1041_104126

/-- The probability of hitting the target for both A and B -/
def p : ℝ := 0.6

/-- The probability that both A and B hit the target -/
def prob_both_hit : ℝ := p * p

/-- The probability that exactly one of A and B hits the target -/
def prob_one_hit : ℝ := 2 * p * (1 - p)

/-- The probability that at least one of A and B hits the target -/
def prob_at_least_one_hit : ℝ := 1 - (1 - p) * (1 - p)

theorem target_probabilities :
  prob_both_hit = 0.36 ∧
  prob_one_hit = 0.48 ∧
  prob_at_least_one_hit = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_target_probabilities_l1041_104126


namespace NUMINAMATH_CALUDE_girls_in_class_l1041_104154

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 35) (h2 : ratio_girls = 3) (h3 : ratio_boys = 4) : 
  (ratio_girls * total) / (ratio_girls + ratio_boys) = 15 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l1041_104154


namespace NUMINAMATH_CALUDE_monica_students_l1041_104196

/-- The number of students Monica sees each day -/
def total_students : ℕ :=
  let first_class := 20
  let second_third_classes := 25 + 25
  let fourth_class := first_class / 2
  let fifth_sixth_classes := 28 + 28
  first_class + second_third_classes + fourth_class + fifth_sixth_classes

/-- Monica sees 136 students each day -/
theorem monica_students : total_students = 136 := by
  sorry

end NUMINAMATH_CALUDE_monica_students_l1041_104196


namespace NUMINAMATH_CALUDE_max_value_of_s_l1041_104104

theorem max_value_of_s (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let s := min x (min (y + 1/x) (1/y))
  s ≤ Real.sqrt 2 ∧ 
  (s = Real.sqrt 2 ↔ x = Real.sqrt 2 ∧ y = 1 / Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_s_l1041_104104


namespace NUMINAMATH_CALUDE_transportation_theorem_l1041_104107

/-- Represents the transportation problem between cities A, B, C, and D. -/
structure TransportationProblem where
  supplies_C : ℝ
  supplies_D : ℝ
  cost_C_to_A : ℝ
  cost_C_to_B : ℝ
  cost_D_to_A : ℝ
  cost_D_to_B : ℝ
  x : ℝ  -- Amount transported from D to B

/-- The total transportation cost as a function of x -/
def total_cost (p : TransportationProblem) : ℝ :=
  p.cost_C_to_A * (200 - (p.supplies_D - p.x)) + 
  p.cost_C_to_B * (300 - p.x) + 
  p.cost_D_to_A * (p.supplies_D - p.x) + 
  p.cost_D_to_B * p.x

theorem transportation_theorem (p : TransportationProblem) 
  (h1 : p.supplies_C = 240)
  (h2 : p.supplies_D = 260)
  (h3 : p.cost_C_to_A = 20)
  (h4 : p.cost_C_to_B = 25)
  (h5 : p.cost_D_to_A = 15)
  (h6 : p.cost_D_to_B = 30)
  (h7 : 60 ≤ p.x ∧ p.x ≤ 260) : 
  (∃ (w : ℝ), w = total_cost p ∧ w = 10 * p.x + 10200) ∧
  (∀ (m : ℝ), (∀ (x : ℝ), 60 ≤ x → x ≤ 260 → 
    (10 - m) * x + 10200 ≥ 10320) ↔ (0 < m ∧ m ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_transportation_theorem_l1041_104107


namespace NUMINAMATH_CALUDE_basic_computer_printer_price_l1041_104173

/-- The total price of a basic computer and printer, given specific conditions -/
theorem basic_computer_printer_price : ∃ (printer_price : ℝ),
  let basic_computer_price : ℝ := 2000
  let enhanced_computer_price : ℝ := basic_computer_price + 500
  let total_price : ℝ := basic_computer_price + printer_price
  printer_price = (1 / 6) * (enhanced_computer_price + printer_price) →
  total_price = 2500 := by
sorry

end NUMINAMATH_CALUDE_basic_computer_printer_price_l1041_104173


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1041_104113

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1041_104113


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l1041_104181

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem plywood_cut_perimeter_difference :
  let original := Rectangle.mk 9 6
  let pieces : List Rectangle := [
    Rectangle.mk 9 2,  -- Configuration 1
    Rectangle.mk 6 3   -- Configuration 2
  ]
  let perimeters := pieces.map perimeter
  let max_perimeter := perimeters.maximum?
  let min_perimeter := perimeters.minimum?
  ∀ (max min : ℝ), max_perimeter = some max → min_perimeter = some min →
    max - min = 6 :=
by sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l1041_104181


namespace NUMINAMATH_CALUDE_savings_comparison_l1041_104152

theorem savings_comparison (last_year_salary : ℝ) (last_year_savings_rate : ℝ) 
  (salary_increase_rate : ℝ) (this_year_savings_rate : ℝ) 
  (h1 : last_year_savings_rate = 0.06)
  (h2 : salary_increase_rate = 0.20)
  (h3 : this_year_savings_rate = 0.05) :
  (this_year_savings_rate * (1 + salary_increase_rate) * last_year_salary) / 
  (last_year_savings_rate * last_year_salary) = 1 := by
sorry

end NUMINAMATH_CALUDE_savings_comparison_l1041_104152


namespace NUMINAMATH_CALUDE_remainder_sum_l1041_104148

theorem remainder_sum (a b : ℤ) (h1 : a % 60 = 49) (h2 : b % 40 = 29) : (a + b) % 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1041_104148


namespace NUMINAMATH_CALUDE_find_x_l1041_104198

theorem find_x (y z : ℝ) (h1 : (20 + 40 + 60 + x) / 4 = (10 + 70 + y + z) / 4 + 9)
                         (h2 : y + z = 110) : x = 106 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1041_104198


namespace NUMINAMATH_CALUDE_trig_inequality_l1041_104125

theorem trig_inequality : ∀ (a b c : ℝ),
  a = Real.sin (2 * Real.pi / 7) →
  b = Real.tan (5 * Real.pi / 7) →
  c = Real.cos (5 * Real.pi / 7) →
  b < c ∧ c < a :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_l1041_104125


namespace NUMINAMATH_CALUDE_olivia_spent_89_dollars_l1041_104170

/-- Calculates the amount spent at a supermarket given initial amount, amount collected, and amount left --/
def amount_spent (initial : ℕ) (collected : ℕ) (left : ℕ) : ℕ :=
  initial + collected - left

theorem olivia_spent_89_dollars : amount_spent 100 148 159 = 89 := by
  sorry

end NUMINAMATH_CALUDE_olivia_spent_89_dollars_l1041_104170


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1041_104138

/-- Acme T-Shirt Company's pricing function -/
def acme_cost (x : ℕ) : ℚ := 75 + 12 * x

/-- Gamma T-shirt Company's pricing function -/
def gamma_cost (x : ℕ) : ℚ := 16 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_for_acme : ℕ := 19

theorem acme_cheaper_at_min_shirts :
  acme_cost min_shirts_for_acme < gamma_cost min_shirts_for_acme ∧
  ∀ n : ℕ, n < min_shirts_for_acme → acme_cost n ≥ gamma_cost n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1041_104138


namespace NUMINAMATH_CALUDE_two_face_painted_count_l1041_104146

/-- Represents a 3x3x3 cube made up of smaller cubes --/
structure Cube3x3x3 where
  /-- The total number of smaller cubes --/
  total_cubes : Nat
  /-- All outer faces of the large cube are painted --/
  outer_faces_painted : Bool

/-- Counts the number of smaller cubes painted on exactly two faces --/
def count_two_face_painted (c : Cube3x3x3) : Nat :=
  12

/-- Theorem stating that in a 3x3x3 painted cube, 12 smaller cubes are painted on exactly two faces --/
theorem two_face_painted_count (c : Cube3x3x3) 
    (h1 : c.total_cubes = 27) 
    (h2 : c.outer_faces_painted = true) : 
  count_two_face_painted c = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_face_painted_count_l1041_104146


namespace NUMINAMATH_CALUDE_henry_tournament_points_l1041_104108

/-- A structure representing a tic-tac-toe tournament result -/
structure TournamentResult where
  win_points : ℕ
  loss_points : ℕ
  draw_points : ℕ
  wins : ℕ
  losses : ℕ
  draws : ℕ

/-- Calculate the total points for a given tournament result -/
def calculate_points (result : TournamentResult) : ℕ :=
  result.win_points * result.wins +
  result.loss_points * result.losses +
  result.draw_points * result.draws

/-- Theorem: Henry's tournament result yields 44 points -/
theorem henry_tournament_points :
  let henry_result : TournamentResult := {
    win_points := 5,
    loss_points := 2,
    draw_points := 3,
    wins := 2,
    losses := 2,
    draws := 10
  }
  calculate_points henry_result = 44 := by sorry

end NUMINAMATH_CALUDE_henry_tournament_points_l1041_104108


namespace NUMINAMATH_CALUDE_salary_problem_l1041_104118

theorem salary_problem (total_salary : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ) 
  (h1 : total_salary = 14000)
  (h2 : a_spend_rate = 0.8)
  (h3 : b_spend_rate = 0.85)
  (h4 : (1 - a_spend_rate) * (total_salary - b_salary) = (1 - b_spend_rate) * b_salary) :
  b_salary = 8000 :=
by sorry

end NUMINAMATH_CALUDE_salary_problem_l1041_104118


namespace NUMINAMATH_CALUDE_barbaras_candies_l1041_104184

/-- Barbara's candy counting problem -/
theorem barbaras_candies (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 9 → bought = 18 → total = initial + bought → total = 27 := by
  sorry

end NUMINAMATH_CALUDE_barbaras_candies_l1041_104184


namespace NUMINAMATH_CALUDE_pentagon_area_theorem_l1041_104110

theorem pentagon_area_theorem (u v : ℤ) 
  (h1 : 0 < v) (h2 : v < u) 
  (h3 : (2 * u * v : ℤ) + (8 * u * v : ℤ) = 902) : 
  2 * u + v = 29 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_theorem_l1041_104110


namespace NUMINAMATH_CALUDE_fraction_value_l1041_104171

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1041_104171


namespace NUMINAMATH_CALUDE_x_squared_eq_zero_is_quadratic_l1041_104147

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: x^2 = 0 is a quadratic equation in one variable -/
theorem x_squared_eq_zero_is_quadratic : is_quadratic_equation f :=
sorry

end NUMINAMATH_CALUDE_x_squared_eq_zero_is_quadratic_l1041_104147


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1041_104199

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ r : ℝ, r > 0 ∧ a = 30 * r ∧ 7/4 = a * r) : a = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1041_104199


namespace NUMINAMATH_CALUDE_buoy_radius_l1041_104145

/-- The radius of a buoy given the dimensions of the hole it leaves -/
theorem buoy_radius (hole_width : ℝ) (hole_depth : ℝ) : 
  hole_width = 30 → hole_depth = 10 → ∃ r : ℝ, r = 16.25 ∧ 
  ∃ x : ℝ, x^2 + (hole_width/2)^2 = (x + hole_depth)^2 ∧ r = x + hole_depth :=
by sorry

end NUMINAMATH_CALUDE_buoy_radius_l1041_104145


namespace NUMINAMATH_CALUDE_walking_distance_multiple_l1041_104124

/-- Prove that the multiple M is 4 given the walking distances of Rajesh and Hiro -/
theorem walking_distance_multiple (total_distance hiro_distance rajesh_distance : ℝ) 
  (h1 : total_distance = 25)
  (h2 : rajesh_distance = 18)
  (h3 : total_distance = hiro_distance + rajesh_distance)
  (h4 : ∃ M : ℝ, rajesh_distance = M * hiro_distance - 10) :
  ∃ M : ℝ, M = 4 ∧ rajesh_distance = M * hiro_distance - 10 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_multiple_l1041_104124


namespace NUMINAMATH_CALUDE_sin_750_degrees_l1041_104117

theorem sin_750_degrees : Real.sin (750 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_750_degrees_l1041_104117


namespace NUMINAMATH_CALUDE_min_nSn_value_l1041_104128

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))

/-- The main theorem -/
theorem min_nSn_value (seq : ArithmeticSequence) 
    (h1 : seq.S 10 = 0) 
    (h2 : seq.S 15 = 25) : 
  ∃ n : ℕ, ∀ m : ℕ, n * seq.S n ≤ m * seq.S m ∧ n * seq.S n = -49 := by
  sorry

end NUMINAMATH_CALUDE_min_nSn_value_l1041_104128


namespace NUMINAMATH_CALUDE_distance_traveled_l1041_104121

/-- Proves that given a constant speed and time, the distance traveled is equal to speed multiplied by time -/
theorem distance_traveled (speed : ℝ) (time : ℝ) : 
  speed = 6 → time = 16 → speed * time = 96 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l1041_104121


namespace NUMINAMATH_CALUDE_arrange_in_order_l1041_104165

def Ψ : ℤ := -(1006 : ℤ)

def Ω : ℤ := -(1007 : ℤ)

def Θ : ℤ := -(1008 : ℤ)

theorem arrange_in_order : Θ < Ω ∧ Ω < Ψ := by
  sorry

end NUMINAMATH_CALUDE_arrange_in_order_l1041_104165


namespace NUMINAMATH_CALUDE_red_card_events_l1041_104139

-- Define the set of cards
inductive Card : Type
| Black : Card
| Red : Card
| White : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the event "A gets the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "B gets the red card"
def B_gets_red (d : Distribution) : Prop := d Person.B = Card.Red

-- Define mutually exclusive events
def mutually_exclusive (P Q : Distribution → Prop) : Prop :=
  ∀ d : Distribution, ¬(P d ∧ Q d)

-- Define opposite events
def opposite_events (P Q : Distribution → Prop) : Prop :=
  ∀ d : Distribution, P d ↔ ¬Q d

-- Theorem statement
theorem red_card_events :
  (mutually_exclusive A_gets_red B_gets_red) ∧
  ¬(opposite_events A_gets_red B_gets_red) := by
  sorry

end NUMINAMATH_CALUDE_red_card_events_l1041_104139


namespace NUMINAMATH_CALUDE_f_properties_l1041_104172

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2 + 1

theorem f_properties :
  let f := f
  ∃ (period : ℝ),
    (f (5 * Real.pi / 4) = Real.sqrt 3) ∧
    (period > 0 ∧ ∀ (x : ℝ), f (x + period) = f x) ∧
    (∀ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) → period ≤ p) ∧
    (f (-Real.pi / 5) < f (7 * Real.pi / 8)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1041_104172


namespace NUMINAMATH_CALUDE_banana_distribution_l1041_104143

theorem banana_distribution (total_bananas : Nat) (num_groups : Nat) (bananas_per_group : Nat) :
  total_bananas = 407 →
  num_groups = 11 →
  bananas_per_group = total_bananas / num_groups →
  bananas_per_group = 37 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_l1041_104143


namespace NUMINAMATH_CALUDE_largest_triangle_area_l1041_104150

/-- The largest area of a triangle ABC, where A = (2,1), B = (5,3), and C = (p,q) 
    lie on the parabola y = -x^2 + 7x - 10, with 2 ≤ p ≤ 5 -/
theorem largest_triangle_area : 
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (5, 3)
  let C : ℝ → ℝ × ℝ := λ p => (p, -p^2 + 7*p - 10)
  let triangle_area : ℝ → ℝ := λ p => 
    (1/2) * abs (A.1 * B.2 + B.1 * (C p).2 + (C p).1 * A.2 - 
                 A.2 * B.1 - B.2 * (C p).1 - (C p).2 * A.1)
  ∃ (max_area : ℝ), max_area = 13/8 ∧ 
    ∀ p : ℝ, 2 ≤ p ∧ p ≤ 5 → triangle_area p ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_largest_triangle_area_l1041_104150


namespace NUMINAMATH_CALUDE_element_four_in_B_l1041_104144

def U : Set ℕ := {x | x ≤ 7}

theorem element_four_in_B (A B : Set ℕ) 
  (h1 : U = A ∪ B) 
  (h2 : A ∩ (Bᶜ) = {2, 3, 5, 7}) : 
  4 ∈ B := by
  sorry

end NUMINAMATH_CALUDE_element_four_in_B_l1041_104144


namespace NUMINAMATH_CALUDE_tan_Y_in_right_triangle_l1041_104130

theorem tan_Y_in_right_triangle (Y : Real) (opposite hypotenuse : ℝ) 
  (h1 : opposite = 8)
  (h2 : hypotenuse = 17)
  (h3 : 0 < opposite)
  (h4 : opposite < hypotenuse) :
  Real.tan Y = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_Y_in_right_triangle_l1041_104130


namespace NUMINAMATH_CALUDE_rebus_solution_l1041_104105

theorem rebus_solution :
  ∃! (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    A + B * 10 + B = C * 10 + A ∧
    C = 1 ∧ B = 9 ∧ A = 6 := by
  sorry

end NUMINAMATH_CALUDE_rebus_solution_l1041_104105


namespace NUMINAMATH_CALUDE_discount_savings_l1041_104120

/-- Given a store with an 8% discount and a customer who pays $184 for an item, 
    prove that the amount saved is $16. -/
theorem discount_savings (discount_rate : ℝ) (paid_amount : ℝ) (saved_amount : ℝ) : 
  discount_rate = 0.08 →
  paid_amount = 184 →
  saved_amount = 16 →
  paid_amount / (1 - discount_rate) * discount_rate = saved_amount := by
sorry

end NUMINAMATH_CALUDE_discount_savings_l1041_104120


namespace NUMINAMATH_CALUDE_some_number_value_l1041_104186

theorem some_number_value (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 25 * some_number * 63) :
  some_number = 35 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1041_104186


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l1041_104192

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (|5 * x₁| + 2 = 47 ∧ |5 * x₂| + 2 = 47) ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -81) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l1041_104192


namespace NUMINAMATH_CALUDE_area_ACE_is_60_l1041_104177

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the intersection point O of diagonals AC and BD
def O (q : Quadrilateral) : Point := sorry

-- Define the height DE of triangle DBC
def DE (q : Quadrilateral) : Real := 15

-- Define the length of DC
def DC (q : Quadrilateral) : Real := 17

-- Define the areas of triangles
def area_ABO (q : Quadrilateral) : Real := sorry
def area_DCO (q : Quadrilateral) : Real := sorry
def area_ACE (q : Quadrilateral) : Real := sorry

-- State the theorem
theorem area_ACE_is_60 (q : Quadrilateral) :
  area_ABO q = area_DCO q → area_ACE q = 60 := by
  sorry

end NUMINAMATH_CALUDE_area_ACE_is_60_l1041_104177


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1041_104122

/-- The equation of a line perpendicular to 2x-y+1=0 and passing through (0,1) -/
theorem perpendicular_line_equation :
  let l₁ : ℝ → ℝ → Prop := λ x y => 2*x - y + 1 = 0
  let p : ℝ × ℝ := (0, 1)
  let l₂ : ℝ → ℝ → Prop := λ x y => x + 2*y - 2 = 0
  (∀ x y, l₂ x y ↔ (y - p.2 = -(1/(2:ℝ)) * (x - p.1))) ∧
  (∀ x y, l₁ x y → ∀ x' y', l₂ x' y' → (y - y') * (x - x') = -(1:ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1041_104122


namespace NUMINAMATH_CALUDE_xiaoning_pe_score_l1041_104135

/-- Calculates the comprehensive score for physical education -/
def calculate_comprehensive_score (midterm_score : ℝ) (final_score : ℝ) : ℝ :=
  0.3 * midterm_score + 0.7 * final_score

/-- Proves that Xiaoning's physical education comprehensive score is 87 points -/
theorem xiaoning_pe_score :
  let max_score : ℝ := 100
  let midterm_weight : ℝ := 0.3
  let final_weight : ℝ := 0.7
  let xiaoning_midterm : ℝ := 80
  let xiaoning_final : ℝ := 90
  calculate_comprehensive_score xiaoning_midterm xiaoning_final = 87 := by
sorry


end NUMINAMATH_CALUDE_xiaoning_pe_score_l1041_104135


namespace NUMINAMATH_CALUDE_circle_equation_correct_l1041_104178

def circle_equation (x y : ℝ) : Prop :=
  (x - 4)^2 + (y + 6)^2 = 16

def is_on_circle (x y : ℝ) : Prop :=
  ((x - 4)^2 + (y + 6)^2) = 16

theorem circle_equation_correct :
  ∀ x y : ℝ, is_on_circle x y ↔ 
    ((x - 4)^2 + (y + 6)^2 = 16 ∧ 
     (x - 4)^2 + (y - (-6))^2 = 4^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l1041_104178


namespace NUMINAMATH_CALUDE_movie_theater_ticket_sales_l1041_104132

/-- Theorem: Movie Theater Ticket Sales
Given the prices and quantities of different types of movie tickets,
prove that the number of evening tickets sold is 300. -/
theorem movie_theater_ticket_sales
  (matinee_price : ℕ) (evening_price : ℕ) (threeD_price : ℕ)
  (matinee_quantity : ℕ) (threeD_quantity : ℕ)
  (total_revenue : ℕ) :
  matinee_price = 5 →
  evening_price = 12 →
  threeD_price = 20 →
  matinee_quantity = 200 →
  threeD_quantity = 100 →
  total_revenue = 6600 →
  ∃ evening_quantity : ℕ,
    evening_quantity = 300 ∧
    total_revenue = matinee_price * matinee_quantity +
                    evening_price * evening_quantity +
                    threeD_price * threeD_quantity :=
by sorry

end NUMINAMATH_CALUDE_movie_theater_ticket_sales_l1041_104132


namespace NUMINAMATH_CALUDE_equation_solution_l1041_104141

theorem equation_solution : ∃ x : ℚ, (1 / 4 + 8 / x = 13 / x + 1 / 8) ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1041_104141


namespace NUMINAMATH_CALUDE_unique_solution_l1041_104190

/-- For every positive integer n, there exists a positive integer c_n
    such that a^n + b^n = c_n^(n+1) -/
def satisfies_condition (a b : ℕ+) : Prop :=
  ∀ n : ℕ+, ∃ c_n : ℕ+, (a : ℕ)^(n : ℕ) + (b : ℕ)^(n : ℕ) = (c_n : ℕ)^((n : ℕ) + 1)

/-- The only pair of positive integers (a,b) satisfying the condition is (2,2) -/
theorem unique_solution :
  ∀ a b : ℕ+, satisfies_condition a b ↔ a = 2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1041_104190


namespace NUMINAMATH_CALUDE_factor_congruence_l1041_104182

theorem factor_congruence (n : ℕ+) (k : ℕ) :
  k ∣ (2 * n.val)^(2^n.val) + 1 → k ≡ 1 [MOD 2^(n.val + 1)] := by
  sorry

end NUMINAMATH_CALUDE_factor_congruence_l1041_104182


namespace NUMINAMATH_CALUDE_production_rates_and_minimum_machines_l1041_104114

/-- Represents the production rate of machine A in kg per hour -/
def machine_a_rate : ℝ := 60

/-- Represents the production rate of machine B in kg per hour -/
def machine_b_rate : ℝ := 50

/-- The difference in production rate between machine A and B -/
def rate_difference : ℝ := 10

/-- The total number of machines used -/
def total_machines : ℕ := 18

/-- The minimum required production in kg per hour -/
def min_production : ℝ := 1000

theorem production_rates_and_minimum_machines :
  (machine_a_rate = machine_b_rate + rate_difference) ∧
  (600 / machine_a_rate = 500 / machine_b_rate) ∧
  (∃ (m : ℕ), m ≤ total_machines ∧ 
    machine_a_rate * m + machine_b_rate * (total_machines - m) ≥ min_production ∧
    ∀ (n : ℕ), n < m → 
      machine_a_rate * n + machine_b_rate * (total_machines - n) < min_production) :=
by sorry

end NUMINAMATH_CALUDE_production_rates_and_minimum_machines_l1041_104114


namespace NUMINAMATH_CALUDE_direction_vector_of_line_l1041_104157

/-- Given a line 2x - 3y + 1 = 0, prove that (3, 2) is a direction vector --/
theorem direction_vector_of_line (x y : ℝ) : 
  (2 * x - 3 * y + 1 = 0) → (∃ (t : ℝ), x = 3 * t ∧ y = 2 * t) := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_of_line_l1041_104157


namespace NUMINAMATH_CALUDE_total_flowers_and_sticks_l1041_104188

theorem total_flowers_and_sticks (num_pots : ℕ) (flowers_per_pot : ℕ) (sticks_per_pot : ℕ) 
  (h1 : num_pots = 466) 
  (h2 : flowers_per_pot = 53) 
  (h3 : sticks_per_pot = 181) : 
  num_pots * flowers_per_pot + num_pots * sticks_per_pot = 109044 :=
by sorry

end NUMINAMATH_CALUDE_total_flowers_and_sticks_l1041_104188


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1041_104191

theorem solution_set_inequality (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1041_104191


namespace NUMINAMATH_CALUDE_coefficient_of_x_in_triple_expansion_l1041_104140

theorem coefficient_of_x_in_triple_expansion (x : ℝ) : 
  let expansion := (1 + x)^3 + (1 + x)^3 + (1 + x)^3
  ∃ a b c d : ℝ, expansion = a + 9*x + b*x^2 + c*x^3 + d*x^4 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_in_triple_expansion_l1041_104140


namespace NUMINAMATH_CALUDE_remainder_problem_l1041_104168

theorem remainder_problem : (7^6 + 8^7 + 9^8) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1041_104168


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1041_104151

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1041_104151


namespace NUMINAMATH_CALUDE_beach_towel_usage_per_person_per_day_l1041_104185

theorem beach_towel_usage_per_person_per_day :
  let num_families : ℕ := 3
  let people_per_family : ℕ := 4
  let total_days : ℕ := 7
  let towels_per_load : ℕ := 14
  let total_loads : ℕ := 6
  let total_people : ℕ := num_families * people_per_family
  let total_towels : ℕ := towels_per_load * total_loads
  let towels_per_day : ℕ := total_towels / total_days
  towels_per_day / total_people = 1 :=
by sorry

end NUMINAMATH_CALUDE_beach_towel_usage_per_person_per_day_l1041_104185


namespace NUMINAMATH_CALUDE_smallest_number_l1041_104164

def numbers : Set ℤ := {0, -2, -1, 3}

theorem smallest_number (n : ℤ) (hn : n ∈ numbers) : -2 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1041_104164


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_fifteen_l1041_104136

theorem factorial_ratio_equals_fifteen : (Nat.factorial 10 * Nat.factorial 7 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 8) = 15 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_fifteen_l1041_104136


namespace NUMINAMATH_CALUDE_extremum_implies_a_and_monotonicity_l1041_104103

/-- The function f(x) = ax^3 - x^2 with a ∈ ℝ -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * x

theorem extremum_implies_a_and_monotonicity (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1/3 ∧ |x - 1/3| < ε → |f a x| ≤ |f a (1/3)|) →
  (a = 6 ∧
   (∀ (x y : ℝ), (x < y ∧ y < 0) → f a x < f a y) ∧
   (∀ (x y : ℝ), (0 < x ∧ x < y ∧ y < 1/3) → f a x > f a y) ∧
   (∀ (x y : ℝ), (1/3 < x ∧ x < y) → f a x < f a y)) :=
by sorry

end NUMINAMATH_CALUDE_extremum_implies_a_and_monotonicity_l1041_104103


namespace NUMINAMATH_CALUDE_passing_mark_is_160_l1041_104163

/-- Represents an exam with a total number of marks and a passing mark. -/
structure Exam where
  total : ℕ
  passing : ℕ

/-- The condition that a candidate scoring 40% fails by 40 marks -/
def condition1 (e : Exam) : Prop :=
  (40 * e.total) / 100 = e.passing - 40

/-- The condition that a candidate scoring 60% passes by 20 marks -/
def condition2 (e : Exam) : Prop :=
  (60 * e.total) / 100 = e.passing + 20

/-- Theorem stating that given the conditions, the passing mark is 160 -/
theorem passing_mark_is_160 (e : Exam) 
  (h1 : condition1 e) (h2 : condition2 e) : e.passing = 160 := by
  sorry


end NUMINAMATH_CALUDE_passing_mark_is_160_l1041_104163


namespace NUMINAMATH_CALUDE_inequality_condition_l1041_104129

theorem inequality_condition (x : ℝ) : (x - Real.pi) * (x - Real.exp 1) ≤ 0 ↔ Real.exp 1 < x ∧ x < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l1041_104129


namespace NUMINAMATH_CALUDE_product_zero_implications_l1041_104189

theorem product_zero_implications (a b c : ℝ) : 
  (((a * b * c = 0) → (a = 0 ∨ b = 0 ∨ c = 0)) ∧
   ((a = 0 ∨ b = 0 ∨ c = 0) → (a * b * c = 0)) ∧
   ((a * b * c ≠ 0) → (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)) ∧
   ((a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → (a * b * c ≠ 0))) :=
by sorry

end NUMINAMATH_CALUDE_product_zero_implications_l1041_104189


namespace NUMINAMATH_CALUDE_range_of_a_l1041_104180

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}
def B : Set ℝ := {x | x^2 - 6*x + 5 < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : (A a ∩ B = ∅) → (a ≥ 6 ∨ a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1041_104180


namespace NUMINAMATH_CALUDE_certain_number_is_eleven_l1041_104167

theorem certain_number_is_eleven (n : ℕ) : 
  (0 < n) → (n < 11) → (n = 1) → (∃ k : ℕ, 18888 - n = 11 * k) → 
  ∀ m : ℕ, (∃ j : ℕ, 18888 - n = m * j) → m = 11 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_is_eleven_l1041_104167


namespace NUMINAMATH_CALUDE_current_speed_l1041_104176

/-- Calculates the speed of the current given boat travel information -/
theorem current_speed (boat_speed : ℝ) (distance : ℝ) (time_against : ℝ) (time_with : ℝ) :
  boat_speed = 15.6 →
  distance = 96 →
  time_against = 8 →
  time_with = 5 →
  ∃ (current_speed : ℝ),
    distance = time_against * (boat_speed - current_speed) ∧
    distance = time_with * (boat_speed + current_speed) ∧
    current_speed = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l1041_104176


namespace NUMINAMATH_CALUDE_square_perimeter_l1041_104100

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 200 → 
  side^2 = area → 
  perimeter = 4 * side → 
  perimeter = 40 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1041_104100


namespace NUMINAMATH_CALUDE_tim_morning_run_hours_l1041_104161

/-- Tim's running schedule -/
structure RunningSchedule where
  runs_per_week : ℕ
  total_hours_per_week : ℕ
  morning_equals_evening : Bool

/-- Calculate the number of hours Tim runs in the morning each day -/
def morning_run_hours (schedule : RunningSchedule) : ℚ :=
  if schedule.morning_equals_evening then
    (schedule.total_hours_per_week : ℚ) / (2 * schedule.runs_per_week)
  else
    0

/-- Theorem: Tim runs 1 hour in the morning each day -/
theorem tim_morning_run_hours :
  let tims_schedule : RunningSchedule := {
    runs_per_week := 5,
    total_hours_per_week := 10,
    morning_equals_evening := true
  }
  morning_run_hours tims_schedule = 1 := by sorry

end NUMINAMATH_CALUDE_tim_morning_run_hours_l1041_104161


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l1041_104175

theorem hcf_from_lcm_and_product (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 600) 
  (h_product : a * b = 18000) : 
  Nat.gcd a b = 30 := by
  sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l1041_104175


namespace NUMINAMATH_CALUDE_sports_conference_games_l1041_104169

/-- The number of games in a sports conference season --/
def conference_games (total_teams : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let divisions := total_teams / teams_per_division
  let intra_division_total := divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_division_total := (total_teams * (total_teams - teams_per_division) / 2) * inter_division_games
  intra_division_total + inter_division_total

theorem sports_conference_games :
  conference_games 16 8 3 2 = 296 := by
  sorry

end NUMINAMATH_CALUDE_sports_conference_games_l1041_104169


namespace NUMINAMATH_CALUDE_pablo_stack_difference_l1041_104158

/-- The height of Pablo's toy block stacks -/
def PabloStacks : ℕ → ℕ
| 0 => 5  -- First stack
| 1 => PabloStacks 0 + 2  -- Second stack
| 2 => PabloStacks 1 - 5  -- Third stack
| 3 => 21 - (PabloStacks 0 + PabloStacks 1 + PabloStacks 2)  -- Last stack
| _ => 0  -- Any other index

theorem pablo_stack_difference : PabloStacks 3 - PabloStacks 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_pablo_stack_difference_l1041_104158


namespace NUMINAMATH_CALUDE_additive_inverses_solution_l1041_104166

theorem additive_inverses_solution (x : ℝ) : (6 * x - 12) + (4 + 2 * x) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverses_solution_l1041_104166


namespace NUMINAMATH_CALUDE_sector_max_area_l1041_104137

/-- Given a sector of a circle with perimeter c (c > 0), 
    prove that the maximum area is c^2/16 and occurs when the arc length is c/2 -/
theorem sector_max_area (c : ℝ) (hc : c > 0) :
  let area (L : ℝ) := (c - L) * L / 4
  ∃ (L : ℝ), L = c / 2 ∧ 
    (∀ x, area x ≤ area L) ∧
    area L = c^2 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l1041_104137


namespace NUMINAMATH_CALUDE_marble_count_l1041_104159

theorem marble_count (red_marbles : ℕ) (green_marbles : ℕ) (yellow_marbles : ℕ) (total_marbles : ℕ) :
  red_marbles = 20 →
  green_marbles = 3 * red_marbles →
  yellow_marbles = (20 * green_marbles) / 100 →
  total_marbles = green_marbles + 3 * green_marbles →
  total_marbles - red_marbles - green_marbles - yellow_marbles = 148 :=
by sorry

end NUMINAMATH_CALUDE_marble_count_l1041_104159


namespace NUMINAMATH_CALUDE_bobs_age_l1041_104183

/-- Represents the ages of four siblings -/
structure SiblingAges where
  susan : ℕ
  arthur : ℕ
  tom : ℕ
  bob : ℕ

/-- Theorem stating Bob's age given the problem conditions -/
theorem bobs_age (ages : SiblingAges) : 
  ages.susan = 15 ∧ 
  ages.arthur = ages.susan + 2 ∧ 
  ages.tom = ages.bob - 3 ∧ 
  ages.susan + ages.arthur + ages.tom + ages.bob = 51 →
  ages.bob = 11 := by
sorry

end NUMINAMATH_CALUDE_bobs_age_l1041_104183


namespace NUMINAMATH_CALUDE_james_beat_record_by_296_l1041_104106

/-- Calculates the total points scored by James in a football season -/
def james_total_points (
  touchdowns_per_game : ℕ)
  (touchdown_points : ℕ)
  (games_in_season : ℕ)
  (two_point_conversions : ℕ)
  (field_goals : ℕ)
  (field_goal_points : ℕ)
  (extra_point_attempts : ℕ)
  (bonus_touchdown_sets : ℕ)
  (bonus_touchdowns_per_set : ℕ)
  (bonus_multiplier : ℕ) : ℕ :=
  let regular_touchdown_points := touchdowns_per_game * games_in_season * touchdown_points
  let bonus_touchdown_points := bonus_touchdown_sets * bonus_touchdowns_per_set * touchdown_points * bonus_multiplier
  let two_point_conversion_points := two_point_conversions * 2
  let field_goal_points := field_goals * field_goal_points
  let extra_point_points := extra_point_attempts
  regular_touchdown_points + bonus_touchdown_points + two_point_conversion_points + field_goal_points + extra_point_points

/-- Theorem stating that James beat the old record by 296 points -/
theorem james_beat_record_by_296 :
  james_total_points 4 6 15 6 8 3 20 5 3 2 - 300 = 296 := by
  sorry

end NUMINAMATH_CALUDE_james_beat_record_by_296_l1041_104106


namespace NUMINAMATH_CALUDE_baker_sales_difference_l1041_104194

/-- Represents the baker's sales data --/
structure BakerSales where
  usual_pastries : ℕ
  usual_bread : ℕ
  today_pastries : ℕ
  today_bread : ℕ
  pastry_price : ℕ
  bread_price : ℕ

/-- Calculates the difference between today's sales and average daily sales --/
def sales_difference (s : BakerSales) : ℕ :=
  let usual_total := s.usual_pastries * s.pastry_price + s.usual_bread * s.bread_price
  let today_total := s.today_pastries * s.pastry_price + s.today_bread * s.bread_price
  today_total - usual_total

/-- Theorem stating the difference in sales --/
theorem baker_sales_difference :
  ∃ (s : BakerSales),
    s.usual_pastries = 20 ∧
    s.usual_bread = 10 ∧
    s.today_pastries = 14 ∧
    s.today_bread = 25 ∧
    s.pastry_price = 2 ∧
    s.bread_price = 4 ∧
    sales_difference s = 48 := by
  sorry

end NUMINAMATH_CALUDE_baker_sales_difference_l1041_104194


namespace NUMINAMATH_CALUDE_fib_odd_index_not_divisible_by_4k_plus_3_prime_l1041_104123

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define prime numbers of the form 4k + 3
def isPrime4kPlus3 (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ k : ℕ, p = 4 * k + 3

-- Theorem statement
theorem fib_odd_index_not_divisible_by_4k_plus_3_prime (n : ℕ) (p : ℕ) 
  (h_prime : isPrime4kPlus3 p) : ¬(p ∣ fib (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_fib_odd_index_not_divisible_by_4k_plus_3_prime_l1041_104123


namespace NUMINAMATH_CALUDE_equation_solutions_l1041_104197

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (3 * x₁^2 - 4 * x₁ = 2 * x₁ ∧ x₁ = 0) ∧
                (3 * x₂^2 - 4 * x₂ = 2 * x₂ ∧ x₂ = 2)) ∧
  (∃ y₁ y₂ : ℝ, (y₁ * (y₁ + 8) = 16 ∧ y₁ = -4 + 4 * Real.sqrt 2) ∧
                (y₂ * (y₂ + 8) = 16 ∧ y₂ = -4 - 4 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1041_104197


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_6_5_8_9_l1041_104156

theorem largest_three_digit_divisible_by_6_5_8_9 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 6 ∣ n ∧ 5 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n → n ≤ 720 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_6_5_8_9_l1041_104156


namespace NUMINAMATH_CALUDE_solve_equation_l1041_104131

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1041_104131


namespace NUMINAMATH_CALUDE_kim_total_points_l1041_104195

/-- Represents the points awarded for each round in the contest -/
structure RoundPoints where
  easy : Nat
  average : Nat
  hard : Nat

/-- Represents the number of correct answers in each round -/
structure CorrectAnswers where
  easy : Nat
  average : Nat
  hard : Nat

/-- Calculates the total points given the round points and correct answers -/
def calculateTotalPoints (points : RoundPoints) (answers : CorrectAnswers) : Nat :=
  points.easy * answers.easy + points.average * answers.average + points.hard * answers.hard

/-- Theorem: Given the contest conditions, Kim's total points are 38 -/
theorem kim_total_points :
  let points : RoundPoints := ⟨2, 3, 5⟩
  let answers : CorrectAnswers := ⟨6, 2, 4⟩
  calculateTotalPoints points answers = 38 := by
  sorry


end NUMINAMATH_CALUDE_kim_total_points_l1041_104195


namespace NUMINAMATH_CALUDE_georgia_yellow_buttons_l1041_104162

/-- The number of yellow buttons Georgia has -/
def yellow_buttons : ℕ := sorry

/-- The number of black buttons Georgia has -/
def black_buttons : ℕ := 2

/-- The number of green buttons Georgia has -/
def green_buttons : ℕ := 3

/-- The number of buttons Georgia gives to Mary -/
def buttons_given : ℕ := 4

/-- The number of buttons Georgia has left after giving buttons to Mary -/
def buttons_left : ℕ := 5

/-- Theorem stating that Georgia has 4 yellow buttons -/
theorem georgia_yellow_buttons : yellow_buttons = 4 := by
  sorry

end NUMINAMATH_CALUDE_georgia_yellow_buttons_l1041_104162


namespace NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l1041_104174

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l1041_104174


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1041_104101

theorem rectangle_max_area (l w : ℝ) : 
  l + w = 30 →  -- Perimeter condition (half of 60)
  l - w = 10 →  -- Difference between length and width
  l * w ≤ 200   -- Maximum area
  := by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1041_104101


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l1041_104134

theorem simplify_nested_expression (x : ℝ) :
  2 * (1 - (2 * (1 - (1 + (2 * (1 - x)))))) = 8 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l1041_104134


namespace NUMINAMATH_CALUDE_inequality_proof_l1041_104153

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  Real.sqrt (a^3 + a) + Real.sqrt (b^3 + b) + Real.sqrt (c^3 + c) ≥ 2 * Real.sqrt (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1041_104153


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l1041_104160

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (θ : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (h3 : θ = Real.pi / 3) -- 60° in radians
  (h4 : c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ)) -- Law of Cosines
  : c = Real.sqrt 109 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l1041_104160


namespace NUMINAMATH_CALUDE_painted_cubes_l1041_104102

theorem painted_cubes (n : ℕ) (h : n = 4) :
  let total_cubes := n^3
  let unpainted_cubes := (n - 2)^3
  let painted_cubes := total_cubes - unpainted_cubes
  painted_cubes = 42 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_l1041_104102


namespace NUMINAMATH_CALUDE_cos_two_pi_thirds_minus_alpha_l1041_104127

theorem cos_two_pi_thirds_minus_alpha (α : ℝ) (h : Real.sin (α - π/6) = 3/5) :
  Real.cos (2*π/3 - α) = 3/5 := by sorry

end NUMINAMATH_CALUDE_cos_two_pi_thirds_minus_alpha_l1041_104127


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1041_104116

theorem solve_linear_equation (x : ℝ) (h : 3*x - 4*x + 7*x = 120) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1041_104116


namespace NUMINAMATH_CALUDE_pizza_slices_left_l1041_104193

def large_pizza_slices : ℕ := 12
def small_pizza_slices : ℕ := 8
def num_large_pizzas : ℕ := 2
def num_small_pizzas : ℕ := 1

def dean_eaten : ℕ := large_pizza_slices / 2
def frank_eaten : ℕ := 3
def sammy_eaten : ℕ := large_pizza_slices / 3
def nancy_cheese_eaten : ℕ := 2
def nancy_pepperoni_eaten : ℕ := 1
def olivia_eaten : ℕ := 2

def total_slices : ℕ := num_large_pizzas * large_pizza_slices + num_small_pizzas * small_pizza_slices

def total_eaten : ℕ := dean_eaten + frank_eaten + sammy_eaten + nancy_cheese_eaten + nancy_pepperoni_eaten + olivia_eaten

theorem pizza_slices_left : total_slices - total_eaten = 14 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l1041_104193
