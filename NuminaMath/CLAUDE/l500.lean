import Mathlib

namespace min_abs_z_given_constraint_l500_50001

open Complex

theorem min_abs_z_given_constraint (z : ℂ) (h : abs (z - 2*I) = 1) : 
  abs z ≥ 1 ∧ ∃ w : ℂ, abs (w - 2*I) = 1 ∧ abs w = 1 := by
  sorry

end min_abs_z_given_constraint_l500_50001


namespace complex_square_root_of_negative_four_l500_50083

theorem complex_square_root_of_negative_four (z : ℂ) 
  (h1 : z^2 = -4)
  (h2 : z.im > 0) : 
  z = Complex.I * 2 := by
  sorry

end complex_square_root_of_negative_four_l500_50083


namespace vegetable_planting_methods_l500_50040

def num_vegetables : ℕ := 4
def num_plots : ℕ := 3
def num_to_select : ℕ := 3

theorem vegetable_planting_methods :
  (num_vegetables - 1).choose (num_to_select - 1) * num_to_select.factorial = 18 := by
  sorry

end vegetable_planting_methods_l500_50040


namespace inverse_proportion_problem_l500_50074

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 35) (h3 : y = 3 * x) :
  y = -21 → x = -10.9375 := by
sorry

end inverse_proportion_problem_l500_50074


namespace ratio_of_fractions_l500_50052

theorem ratio_of_fractions (A B : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) 
  (h : (2 / 3) * A = (3 / 7) * B) : 
  A / B = 9 / 14 := by
sorry

end ratio_of_fractions_l500_50052


namespace unique_equal_expression_l500_50089

theorem unique_equal_expression (x : ℝ) (h : x > 0) :
  (x^(x+1) + x^(x+1) = 2*x^(x+1)) ∧
  (x^(x+1) + x^(x+1) ≠ x^(2*x+2)) ∧
  (x^(x+1) + x^(x+1) ≠ (2*x)^(x+1)) ∧
  (x^(x+1) + x^(x+1) ≠ (2*x)^(2*x+2)) :=
by sorry

end unique_equal_expression_l500_50089


namespace inequality_system_solution_l500_50014

theorem inequality_system_solution (x : ℝ) :
  (5 * x + 1 > 3 * (x - 1)) ∧ ((1 / 2) * x < 3) → -2 < x ∧ x < 6 := by
  sorry

end inequality_system_solution_l500_50014


namespace volume_increase_l500_50021

/-- Represents a rectangular solid -/
structure RectangularSolid where
  baseArea : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular solid -/
def volume (solid : RectangularSolid) : ℝ :=
  solid.baseArea * solid.height

/-- Theorem: Increase in volume of a rectangular solid -/
theorem volume_increase (solid : RectangularSolid) 
  (h1 : solid.baseArea = 12)
  (h2 : 5 > 0) :
  volume { baseArea := solid.baseArea, height := solid.height + 5 } - volume solid = 60 := by
  sorry

end volume_increase_l500_50021


namespace total_notebooks_purchased_l500_50054

def john_purchases : List Nat := [2, 4, 6, 8, 10]
def wife_purchases : List Nat := [3, 7, 5, 9, 11]

theorem total_notebooks_purchased : 
  (List.sum john_purchases) + (List.sum wife_purchases) = 65 := by
  sorry

end total_notebooks_purchased_l500_50054


namespace age_difference_l500_50041

theorem age_difference (A B : ℕ) : B = 34 → A + 10 = 2 * (B - 10) → A - B = 4 := by
  sorry

end age_difference_l500_50041


namespace strawberry_plants_l500_50066

theorem strawberry_plants (P : ℕ) : 
  24 * P - 4 = 500 → P = 21 := by
  sorry

end strawberry_plants_l500_50066


namespace shelves_used_l500_50065

def initial_stock : ℕ := 17
def new_shipment : ℕ := 10
def bears_per_shelf : ℕ := 9

theorem shelves_used (initial_stock new_shipment bears_per_shelf : ℕ) :
  initial_stock = 17 →
  new_shipment = 10 →
  bears_per_shelf = 9 →
  (initial_stock + new_shipment) / bears_per_shelf = 3 := by
  sorry

end shelves_used_l500_50065


namespace problem_solution_l500_50097

theorem problem_solution : 
  ∀ x y : ℤ, x > y ∧ y > 0 ∧ x + y + x * y = 152 → x = 16 := by
sorry

end problem_solution_l500_50097


namespace unique_intersecting_line_l500_50064

/-- A parabola defined by y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = 8 * p.1}

/-- The point M with coordinates (2, 4) -/
def M : ℝ × ℝ := (2, 4)

/-- A line that passes through point M and intersects the parabola at exactly one point -/
def UniqueLine (l : Set (ℝ × ℝ)) : Prop :=
  M ∈ l ∧ (∃! p, p ∈ l ∩ Parabola)

/-- The theorem stating that there is exactly one unique line passing through M
    that intersects the parabola at exactly one point -/
theorem unique_intersecting_line :
  ∃! l : Set (ℝ × ℝ), UniqueLine l :=
sorry

end unique_intersecting_line_l500_50064


namespace sequence_squared_l500_50092

theorem sequence_squared (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) > a n) 
  (h3 : ∀ n, (a (n + 1))^2 + (a n)^2 + 1 = 2 * ((a (n + 1)) * (a n) + a (n + 1) + a n)) :
  ∀ n, a n = n^2 := by sorry

end sequence_squared_l500_50092


namespace buratino_arrival_time_l500_50011

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds hours and minutes to a given time -/
def addTime (t : Time) (h : Nat) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + h * 60 + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem buratino_arrival_time :
  let departureTime : Time := { hours := 13, minutes := 40 }
  let normalJourneyTime : Real := 7.5
  let fasterJourneyTime : Real := normalJourneyTime * 4 / 5
  let timeDifference : Real := normalJourneyTime - fasterJourneyTime
  timeDifference = 1.5 →
  addTime departureTime 7 30 = { hours := 21, minutes := 10 } :=
by sorry

end buratino_arrival_time_l500_50011


namespace game_probability_game_probability_value_l500_50004

theorem game_probability : ℝ :=
  let total_outcomes : ℕ := 16 * 16
  let matching_outcomes : ℕ := 16
  let non_matching_outcomes : ℕ := total_outcomes - matching_outcomes
  (non_matching_outcomes : ℝ) / total_outcomes

theorem game_probability_value : game_probability = 15 / 16 := by sorry

end game_probability_game_probability_value_l500_50004


namespace least_k_for_convergence_l500_50008

def u : ℕ → ℚ
  | 0 => 1/4
  | n + 1 => 2 * u n - 2 * (u n)^2

def L : ℚ := 1/2

theorem least_k_for_convergence :
  (∀ k : ℕ, k < 10 → |u k - L| > 1/2^1000) ∧
  |u 10 - L| ≤ 1/2^1000 := by sorry

end least_k_for_convergence_l500_50008


namespace smallest_integer_satisfying_inequality_l500_50002

theorem smallest_integer_satisfying_inequality :
  ∃ x : ℤ, (∀ y : ℤ, 8 - 7 * y ≥ 4 * y - 3 → x ≤ y) ∧ (8 - 7 * x ≥ 4 * x - 3) :=
by
  sorry

end smallest_integer_satisfying_inequality_l500_50002


namespace quadratic_inequality_solution_l500_50009

theorem quadratic_inequality_solution (a c : ℝ) (h : ∀ x, (a * x^2 + 5 * x + c > 0) ↔ (1/3 < x ∧ x < 1/2)) :
  (a = -6 ∧ c = -1) ∧
  (∀ b : ℝ, 
    (∀ x, (a * x^2 + (a * c + b) * x + b * c ≥ 0) ↔ 
      ((b > 6 ∧ 1 ≤ x ∧ x ≤ b/6) ∨
       (b = 6 ∧ x = 1) ∨
       (b < 6 ∧ b/6 ≤ x ∧ x ≤ 1)))) :=
by sorry

end quadratic_inequality_solution_l500_50009


namespace negation_of_all_students_punctual_l500_50080

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a student and being punctual
variable (student : U → Prop)
variable (punctual : U → Prop)

-- State the theorem
theorem negation_of_all_students_punctual :
  (¬ ∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) :=
sorry

end negation_of_all_students_punctual_l500_50080


namespace first_five_valid_numbers_l500_50076

def random_table : List (List Nat) := [
  [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76],
  [63, 01, 63, 78, 59, 16, 95, 56, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79],
  [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38, 15, 51, 00, 13, 42, 99, 66, 02, 79, 54]
]

def start_row : Nat := 8
def start_col : Nat := 7
def max_bag_number : Nat := 799

def is_valid_number (n : Nat) : Bool :=
  n <= max_bag_number

def find_valid_numbers (table : List (List Nat)) (row : Nat) (col : Nat) (count : Nat) : List Nat :=
  sorry

theorem first_five_valid_numbers :
  find_valid_numbers random_table start_row start_col 5 = [785, 667, 199, 507, 175] :=
sorry

end first_five_valid_numbers_l500_50076


namespace smallest_right_triangle_area_l500_50056

theorem smallest_right_triangle_area :
  ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 4 ∨ b = 4 ∨ c = 4) →
  (a = 6 ∨ b = 6 ∨ c = 6) →
  a^2 + b^2 = c^2 →
  (1/2 * a * b) ≥ 4 * Real.sqrt 5 :=
by sorry

end smallest_right_triangle_area_l500_50056


namespace solution_equation1_solution_equation2_l500_50015

-- Define the equations
def equation1 (x : ℝ) : Prop := (2*x - 5)/6 - (3*x + 1)/2 = 1
def equation2 (x : ℝ) : Prop := 3*x - 7*(x - 1) = 3 - 2*(x + 3)

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = -2 := by sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = 5 := by sorry

end solution_equation1_solution_equation2_l500_50015


namespace weighted_graph_vertex_labeling_l500_50059

-- Define a graph type
structure Graph (V : Type) where
  edges : V → V → Prop

-- Define a weight function type
def WeightFunction (V : Type) := V → V → ℝ

-- Define the property of distinct positive weights
def DistinctPositiveWeights (V : Type) (f : WeightFunction V) :=
  ∀ u v w : V, u ≠ v → v ≠ w → u ≠ w → f u v > 0 ∧ f u v ≠ f v w ∧ f u v ≠ f u w

-- Define the degenerate triangle property
def DegenerateTriangle (V : Type) (f : WeightFunction V) :=
  ∀ a b c : V, 
    (f a b = f a c + f b c) ∨ 
    (f a c = f a b + f b c) ∨ 
    (f b c = f a b + f a c)

-- Define the vertex labeling function type
def VertexLabeling (V : Type) := V → ℝ

-- State the theorem
theorem weighted_graph_vertex_labeling 
  (V : Type) 
  (G : Graph V) 
  (f : WeightFunction V) 
  (h1 : DistinctPositiveWeights V f) 
  (h2 : DegenerateTriangle V f) :
  ∃ w : VertexLabeling V, ∀ u v : V, f u v = |w u - w v| :=
sorry

end weighted_graph_vertex_labeling_l500_50059


namespace rod_length_proof_l500_50045

/-- The length of a rod in meters, given the number of pieces and the length of each piece in centimeters. -/
def rod_length_meters (num_pieces : ℕ) (piece_length_cm : ℕ) : ℚ :=
  (num_pieces * piece_length_cm : ℚ) / 100

/-- Theorem stating that a rod from which 45 pieces of 85 cm can be cut is 38.25 meters long. -/
theorem rod_length_proof : rod_length_meters 45 85 = 38.25 := by
  sorry

end rod_length_proof_l500_50045


namespace rectangle_dimension_increase_l500_50075

/-- Given a rectangle with original length L and breadth B, prove that if the breadth is
    increased by 25% and the total area is increased by 37.5%, then the length must be
    increased by 10% -/
theorem rectangle_dimension_increase (L B : ℝ) (L_pos : L > 0) (B_pos : B > 0) :
  let new_B := 1.25 * B
  let new_area := 1.375 * (L * B)
  ∃ x : ℝ, x = 0.1 ∧ new_area = (L * (1 + x)) * new_B := by
  sorry

end rectangle_dimension_increase_l500_50075


namespace consecutive_math_majors_probability_l500_50038

/-- The number of people sitting around the table -/
def total_people : ℕ := 11

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 3

/-- The number of chemistry majors -/
def chemistry_majors : ℕ := 3

/-- The probability of all math majors sitting consecutively -/
def prob_consecutive_math_majors : ℚ := 1 / 4320

theorem consecutive_math_majors_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let favorable_arrangements := (total_people - math_majors + 1) * Nat.factorial math_majors
  Rat.cast favorable_arrangements / Rat.cast total_arrangements = prob_consecutive_math_majors := by
  sorry

end consecutive_math_majors_probability_l500_50038


namespace equation_roots_l500_50085

theorem equation_roots :
  let f (x : ℝ) := x^2 - 2*x - 2/x + 1/x^2 - 13
  ∃ (a b c d : ℝ),
    (a = (5 + Real.sqrt 21) / 2) ∧
    (b = (5 - Real.sqrt 21) / 2) ∧
    (c = (-3 + Real.sqrt 5) / 2) ∧
    (d = (-3 - Real.sqrt 5) / 2) ∧
    (∀ x : ℝ, x ≠ 0 → (f x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d))) :=
by sorry

end equation_roots_l500_50085


namespace sids_remaining_fraction_l500_50088

/-- Proves that the fraction of Sid's original money left after purchases is 1/2 -/
theorem sids_remaining_fraction (initial : ℝ) (accessories : ℝ) (snacks : ℝ) (extra : ℝ) 
  (h1 : initial = 48)
  (h2 : accessories = 12)
  (h3 : snacks = 8)
  (h4 : extra = 4)
  (h5 : initial - (accessories + snacks) = initial * (1/2) + extra) :
  (initial - (accessories + snacks)) / initial = 1/2 := by
  sorry

end sids_remaining_fraction_l500_50088


namespace power_inequality_l500_50098

theorem power_inequality (x y a b : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : a > b) (h4 : b > 1) :
  a^x > b^y := by sorry

end power_inequality_l500_50098


namespace opposite_of_sqrt3_plus_a_l500_50067

theorem opposite_of_sqrt3_plus_a (a b : ℝ) (h : |a - 3*b| + Real.sqrt (b + 1) = 0) :
  -(Real.sqrt 3 + a) = 3 - Real.sqrt 3 := by
sorry

end opposite_of_sqrt3_plus_a_l500_50067


namespace unique_four_digit_int_l500_50073

/-- Represents a four-digit positive integer -/
structure FourDigitInt where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_pos : a > 0
  a_lt_10 : a < 10
  b_lt_10 : b < 10
  c_lt_10 : c < 10
  d_lt_10 : d < 10

def to_int (n : FourDigitInt) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

theorem unique_four_digit_int :
  ∃! (n : FourDigitInt),
    n.a + n.b + n.c + n.d = 16 ∧
    n.b + n.c = 10 ∧
    n.a - n.d = 2 ∧
    (to_int n) % 9 = 0 ∧
    to_int n = 4622 :=
by sorry

end unique_four_digit_int_l500_50073


namespace complex_sum_of_powers_l500_50026

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_sum_of_powers (h : i^2 = -1) : (1 + i)^30 + (1 - i)^30 = 0 := by
  sorry

end complex_sum_of_powers_l500_50026


namespace network_connections_l500_50084

/-- 
Given a network of switches where:
- There are 30 switches
- Each switch is directly connected to exactly 4 other switches
This theorem states that the total number of connections in the network is 60.
-/
theorem network_connections (n : ℕ) (c : ℕ) : 
  n = 30 → c = 4 → (n * c) / 2 = 60 := by
  sorry

end network_connections_l500_50084


namespace milan_phone_rate_l500_50078

/-- Calculates the rate per minute for a phone service given the total bill, monthly fee, and minutes used. -/
def rate_per_minute (total_bill : ℚ) (monthly_fee : ℚ) (minutes : ℕ) : ℚ :=
  (total_bill - monthly_fee) / minutes

/-- Proves that given the specific conditions, the rate per minute is $0.12 -/
theorem milan_phone_rate :
  let total_bill : ℚ := 23.36
  let monthly_fee : ℚ := 2
  let minutes : ℕ := 178
  rate_per_minute total_bill monthly_fee minutes = 0.12 := by
sorry


end milan_phone_rate_l500_50078


namespace adrianna_gum_purchase_l500_50006

/-- The number of gum pieces Adrianna bought in the second store visit -/
def second_store_purchase (initial_gum : ℕ) (first_store_purchase : ℕ) (total_friends : ℕ) : ℕ :=
  total_friends - (initial_gum + first_store_purchase)

/-- Theorem: Given Adrianna's initial 10 pieces of gum, 3 pieces bought from the first store,
    and 15 friends who received gum, the number of gum pieces bought in the second store visit is 2. -/
theorem adrianna_gum_purchase :
  second_store_purchase 10 3 15 = 2 := by
  sorry

end adrianna_gum_purchase_l500_50006


namespace right_triangle_area_l500_50096

theorem right_triangle_area (a b : ℝ) (ha : a = 45) (hb : b = 48) :
  (1 / 2 : ℝ) * a * b = 1080 := by
  sorry

end right_triangle_area_l500_50096


namespace polynomial_problem_l500_50082

-- Define the polynomials
def B (x : ℝ) : ℝ := 4 * x^2 - 5 * x - 7

theorem polynomial_problem (A : ℝ → ℝ) 
  (h : ∀ x, A x - 2 * (B x) = -2 * x^2 + 10 * x + 14) :
  (∀ x, A x = 6 * x^2) ∧ 
  (∀ x, A x + 2 * (B x) = 14 * x^2 - 10 * x - 14) ∧
  (A (-1) + 2 * (B (-1)) = 10) := by
  sorry

end polynomial_problem_l500_50082


namespace jerome_toy_cars_l500_50053

theorem jerome_toy_cars (original : ℕ) (total : ℕ) (last_month : ℕ) :
  original = 25 →
  total = 40 →
  total = original + last_month + 2 * last_month →
  last_month = 5 := by
  sorry

end jerome_toy_cars_l500_50053


namespace inequality_proof_l500_50013

-- Define the set A
def A : Set ℝ := {x | x > 1}

-- State the theorem
theorem inequality_proof (m n : ℝ) (hm : m ∈ A) (hn : n ∈ A) (h_sum : m + n = 4) :
  n^2 / (m - 1) + m^2 / (n - 1) ≥ 8 := by
  sorry

end inequality_proof_l500_50013


namespace blue_red_ratio_13_l500_50087

/-- Represents the ratio of blue to red face areas in a cube cutting problem -/
def blue_to_red_ratio (n : ℕ) : ℚ :=
  (6 * n^3 - 6 * n^2) / (6 * n^2)

/-- Theorem stating that for a cube of side length 13, the ratio of blue to red face areas is 12 -/
theorem blue_red_ratio_13 : blue_to_red_ratio 13 = 12 := by
  sorry

#eval blue_to_red_ratio 13

end blue_red_ratio_13_l500_50087


namespace sprint_team_total_distance_l500_50099

theorem sprint_team_total_distance (team_size : ℕ) (distance_per_person : ℝ) :
  team_size = 250 →
  distance_per_person = 7.5 →
  team_size * distance_per_person = 1875 := by
sorry

end sprint_team_total_distance_l500_50099


namespace investment_rate_calculation_l500_50018

/-- Prove that if Rs. 1600 is divided into two parts, where one part (P1) is Rs. 1100
    invested at 6% and the other part (P2) is the remainder, and the total annual
    interest from both parts is Rs. 85, then P2 must be invested at 3.8%. -/
theorem investment_rate_calculation (total : ℝ) (p1 : ℝ) (p2 : ℝ) (r1 : ℝ) (total_interest : ℝ) :
  total = 1600 →
  p1 = 1100 →
  p2 = total - p1 →
  r1 = 6 →
  total_interest = 85 →
  p1 * r1 / 100 + p2 * (total_interest - p1 * r1 / 100) / p2 = total_interest →
  (total_interest - p1 * r1 / 100) / p2 * 100 = 3.8 := by
  sorry

end investment_rate_calculation_l500_50018


namespace number_problem_l500_50070

theorem number_problem (x : ℝ) : 2 * x - 12 = 20 → x = 16 := by
  sorry

end number_problem_l500_50070


namespace ellipse_properties_l500_50086

noncomputable def ellipseC (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def pointA : ℝ × ℝ := (0, 1)

def arithmeticSequence (BF1 F1F2 BF2 : ℝ) : Prop :=
  2 * F1F2 = Real.sqrt 3 * (BF1 + BF2)

def lineL (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 + 2)}

def outsideCircle (A P Q : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (Q.1 - A.1) + (P.2 - A.2) * (Q.2 - A.2) > 0

theorem ellipse_properties (a b : ℝ) (h : a > b ∧ b > 0) :
  pointA ∈ ellipseC a b h →
  (∀ B ∈ ellipseC a b h, ∃ F1 F2 : ℝ × ℝ,
    arithmeticSequence (dist B F1) (dist F1 F2) (dist B F2)) →
  (ellipseC a b h = ellipseC 2 1 ⟨by norm_num, by norm_num⟩) ∧
  (∀ k : ℝ, (∀ P Q : ℝ × ℝ, P ∈ ellipseC 2 1 ⟨by norm_num, by norm_num⟩ →
                            Q ∈ ellipseC 2 1 ⟨by norm_num, by norm_num⟩ →
                            P ∈ lineL k → Q ∈ lineL k → P ≠ Q →
                            outsideCircle pointA P Q) ↔
             (k < -3/10 ∨ k > 1/2)) :=
by sorry

end ellipse_properties_l500_50086


namespace four_students_arrangement_l500_50055

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange four students in a line
    with three students standing together. -/
def arrangements_with_restriction : ℕ :=
  permutations 2 * permutations 3

theorem four_students_arrangement :
  permutations 4 - arrangements_with_restriction = 12 := by
  sorry

end four_students_arrangement_l500_50055


namespace line_slope_angle_l500_50071

theorem line_slope_angle (x y : ℝ) :
  x - Real.sqrt 3 * y + 1 = 0 →
  Real.arctan (Real.sqrt 3 / 3) = 30 * Real.pi / 180 :=
by sorry

end line_slope_angle_l500_50071


namespace poetic_line_contrast_l500_50003

/-- Represents a poetic line with two parts -/
structure PoeticLine :=
  (part1 : String)
  (part2 : String)

/-- Determines if a given part of a poetic line represents stillness -/
def isStillness (part : String) : Prop :=
  sorry

/-- Determines if a given part of a poetic line represents motion -/
def isMotion (part : String) : Prop :=
  sorry

/-- Determines if a poetic line contrasts stillness and motion -/
def contrastsStillnessAndMotion (line : PoeticLine) : Prop :=
  (isStillness line.part1 ∧ isMotion line.part2) ∨ (isMotion line.part1 ∧ isStillness line.part2)

/-- The four poetic lines given in the problem -/
def lineA : PoeticLine :=
  { part1 := "The bridge echoes with the distant barking of dogs"
  , part2 := "and the courtyard is empty with people asleep" }

def lineB : PoeticLine :=
  { part1 := "The stove fire illuminates the heaven and earth"
  , part2 := "and the red stars are mixed with the purple smoke" }

def lineC : PoeticLine :=
  { part1 := "The cold trees begin to have bird activities"
  , part2 := "and the frosty bridge has no human passage yet" }

def lineD : PoeticLine :=
  { part1 := "The crane cries over the quiet Chu mountain"
  , part2 := "and the frost is white on the autumn river in the morning" }

theorem poetic_line_contrast :
  contrastsStillnessAndMotion lineA ∧
  contrastsStillnessAndMotion lineB ∧
  contrastsStillnessAndMotion lineC ∧
  ¬contrastsStillnessAndMotion lineD :=
sorry

end poetic_line_contrast_l500_50003


namespace max_abs_z_plus_i_l500_50095

theorem max_abs_z_plus_i :
  ∀ (x y : ℝ), 
    x^2/4 + y^2 = 1 →
    ∀ (z : ℂ), 
      z = x + y * Complex.I →
      ∀ (w : ℂ), 
        Complex.abs w = Complex.abs (z + Complex.I) →
        Complex.abs w ≤ 4 * Real.sqrt 3 / 3 :=
by sorry

end max_abs_z_plus_i_l500_50095


namespace ship_cargo_after_loading_l500_50007

/-- The total cargo on a ship after loading additional cargo in the Bahamas -/
theorem ship_cargo_after_loading (initial_cargo additional_cargo : ℕ) :
  initial_cargo = 5973 →
  additional_cargo = 8723 →
  initial_cargo + additional_cargo = 14696 := by
  sorry

end ship_cargo_after_loading_l500_50007


namespace students_per_section_after_changes_l500_50057

theorem students_per_section_after_changes 
  (initial_students_per_section : ℕ)
  (new_sections : ℕ)
  (total_sections_after : ℕ)
  (new_students : ℕ)
  (h1 : initial_students_per_section = 24)
  (h2 : new_sections = 3)
  (h3 : total_sections_after = 16)
  (h4 : new_students = 24) :
  (initial_students_per_section * (total_sections_after - new_sections) + new_students) / total_sections_after = 21 :=
by sorry

end students_per_section_after_changes_l500_50057


namespace arithmetic_geometric_sequence_l500_50030

/-- Given an arithmetic sequence with common difference d ≠ 0,
    if a₁, a₃, a₇ form a geometric sequence, then a₁/d = 2 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  (∀ n, a (n + 1) = a n + d) →
  (∃ r, a 3 = a 1 * r ∧ a 7 = a 3 * r) →
  a 1 / d = 2 :=
sorry

end arithmetic_geometric_sequence_l500_50030


namespace inequality_proof_l500_50010

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + n^n / x^n ≥ n + 1 → n^n = n^n :=
by
  sorry

#check inequality_proof

end inequality_proof_l500_50010


namespace centerville_library_budget_percentage_l500_50024

/-- Proves that the percentage of Centerville's annual budget spent on the public library is 15% -/
theorem centerville_library_budget_percentage
  (library_expense : ℕ)
  (park_percentage : ℚ)
  (remaining_budget : ℕ)
  (h1 : library_expense = 3000)
  (h2 : park_percentage = 24 / 100)
  (h3 : remaining_budget = 12200)
  : ∃ (total_budget : ℕ), 
    (library_expense : ℚ) / total_budget = 15 / 100 :=
sorry

end centerville_library_budget_percentage_l500_50024


namespace horizontal_distance_calculation_l500_50039

/-- Given a vertical climb and a ratio of vertical to horizontal movement,
    calculate the horizontal distance traveled. -/
theorem horizontal_distance_calculation
  (vertical_climb : ℝ)
  (vertical_ratio : ℝ)
  (horizontal_ratio : ℝ)
  (h_positive : vertical_climb > 0)
  (h_ratio_positive : vertical_ratio > 0 ∧ horizontal_ratio > 0)
  (h_climb : vertical_climb = 1350)
  (h_ratio : vertical_ratio / horizontal_ratio = 1 / 2) :
  vertical_climb * horizontal_ratio / vertical_ratio = 2700 := by
  sorry

end horizontal_distance_calculation_l500_50039


namespace employee_device_distribution_l500_50091

theorem employee_device_distribution (E : ℝ) (E_pos : E > 0) : 
  let cell_phone := (2/3 : ℝ) * E
  let pager := (2/5 : ℝ) * E
  let both := (0.4 : ℝ) * E
  let neither := E - (cell_phone + pager - both)
  neither = (1/3 : ℝ) * E := by
sorry

end employee_device_distribution_l500_50091


namespace complex_equation_sum_l500_50035

theorem complex_equation_sum (a b : ℝ) :
  (a - 2 * Complex.I) * Complex.I = b + Complex.I →
  a + b = 3 := by
sorry

end complex_equation_sum_l500_50035


namespace composite_function_ratio_l500_50094

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem
theorem composite_function_ratio :
  (f (g (f 2))) / (g (f (g 2))) = 41 / 7 := by
  sorry

end composite_function_ratio_l500_50094


namespace max_b_value_l500_50062

theorem max_b_value (a b c : ℕ) : 
  a * b * c = 240 →
  1 < c →
  c < b →
  b < a →
  b ≤ 10 :=
sorry

end max_b_value_l500_50062


namespace x_value_proof_l500_50079

theorem x_value_proof (x : ℚ) (h : 3/5 - 1/4 = 4/x) : x = 80/7 := by
  sorry

end x_value_proof_l500_50079


namespace symmetric_line_equation_l500_50017

/-- Given two lines l₁ and l₂ in the xy-plane, where l₁ has the equation 2x - y + 1 = 0
    and l₂ is symmetric to l₁ with respect to the line y = -x,
    prove that the equation of l₂ is x - 2y + 1 = 0 -/
theorem symmetric_line_equation :
  ∀ (l₁ l₂ : Set (ℝ × ℝ)),
  (∀ x y, (x, y) ∈ l₁ ↔ 2 * x - y + 1 = 0) →
  (∀ x y, (x, y) ∈ l₂ ↔ ∃ x' y', (x', y') ∈ l₁ ∧ x + x' = y + y') →
  (∀ x y, (x, y) ∈ l₂ ↔ x - 2 * y + 1 = 0) :=
by sorry


end symmetric_line_equation_l500_50017


namespace second_question_probability_l500_50049

theorem second_question_probability 
  (p_first : ℝ) 
  (p_neither : ℝ) 
  (p_both : ℝ) 
  (h1 : p_first = 0.65)
  (h2 : p_neither = 0.20)
  (h3 : p_both = 0.40)
  : ∃ p_second : ℝ, p_second = 0.75 ∧ 
    p_first + p_second - p_both + p_neither = 1 :=
sorry

end second_question_probability_l500_50049


namespace percentage_70_79_is_800_27_l500_50028

/-- Represents the frequency distribution of test scores -/
structure ScoreDistribution where
  score_90_100 : Nat
  score_80_89 : Nat
  score_70_79 : Nat
  score_60_69 : Nat
  score_below_60 : Nat

/-- Calculates the percentage of students in the 70%-79% range -/
def percentage_70_79 (dist : ScoreDistribution) : Rat :=
  let total := dist.score_90_100 + dist.score_80_89 + dist.score_70_79 + dist.score_60_69 + dist.score_below_60
  (dist.score_70_79 : Rat) / total * 100

/-- The given frequency distribution -/
def history_class_distribution : ScoreDistribution :=
  { score_90_100 := 5
    score_80_89 := 7
    score_70_79 := 8
    score_60_69 := 4
    score_below_60 := 3 }

theorem percentage_70_79_is_800_27 :
  percentage_70_79 history_class_distribution = 800 / 27 := by
  sorry

end percentage_70_79_is_800_27_l500_50028


namespace complex_number_problem_l500_50090

theorem complex_number_problem (z : ℂ) : 
  (∃ (b : ℝ), z = b * I) → 
  (∃ (c : ℝ), (z + 2)^2 + 8 * I = c * I) → 
  z = 2 * I := by
sorry

end complex_number_problem_l500_50090


namespace running_distance_proof_l500_50081

/-- Calculates the total distance run over a number of days, given a constant daily distance. -/
def totalDistance (dailyDistance : ℕ) (days : ℕ) : ℕ :=
  dailyDistance * days

/-- Proves that running 1700 meters for 6 consecutive days results in a total distance of 10200 meters. -/
theorem running_distance_proof :
  let dailyDistance : ℕ := 1700
  let days : ℕ := 6
  totalDistance dailyDistance days = 10200 := by
sorry

end running_distance_proof_l500_50081


namespace birds_and_storks_on_fence_l500_50037

theorem birds_and_storks_on_fence : 
  let initial_birds : ℕ := 2
  let additional_birds : ℕ := 5
  let storks : ℕ := 4
  let total_birds : ℕ := initial_birds + additional_birds
  (total_birds - storks) = 3 := by
  sorry

end birds_and_storks_on_fence_l500_50037


namespace exists_non_complementary_acute_angles_l500_50012

-- Define what an acute angle is
def is_acute_angle (angle : ℝ) : Prop := 0 < angle ∧ angle < 90

-- Define what complementary angles are
def are_complementary (angle1 angle2 : ℝ) : Prop := angle1 + angle2 = 90

-- Theorem statement
theorem exists_non_complementary_acute_angles :
  ∃ (angle1 angle2 : ℝ), is_acute_angle angle1 ∧ is_acute_angle angle2 ∧ ¬(are_complementary angle1 angle2) := by
  sorry

end exists_non_complementary_acute_angles_l500_50012


namespace faster_train_speed_l500_50061

/-- The speed of the faster train given two trains moving in opposite directions --/
theorem faster_train_speed
  (slow_speed : ℝ)
  (length_slow : ℝ)
  (length_fast : ℝ)
  (crossing_time : ℝ)
  (h_slow_speed : slow_speed = 60)
  (h_length_slow : length_slow = 1.10)
  (h_length_fast : length_fast = 0.9)
  (h_crossing_time : crossing_time = 47.99999999999999 / 3600) :
  ∃ (fast_speed : ℝ), fast_speed = 90 ∧
    (fast_speed + slow_speed) * crossing_time = length_slow + length_fast :=
by sorry

end faster_train_speed_l500_50061


namespace salesperson_allocation_l500_50005

/-- Represents the problem of determining the number of salespersons to send to a branch office --/
theorem salesperson_allocation
  (total_salespersons : ℕ)
  (initial_avg_income : ℝ)
  (hq_income_increase : ℝ)
  (branch_income_factor : ℝ)
  (h_total : total_salespersons = 100)
  (h_hq_increase : hq_income_increase = 0.2)
  (h_branch_factor : branch_income_factor = 3.5)
  (x : ℕ) :
  (((total_salespersons - x) * (1 + hq_income_increase) * initial_avg_income ≥ 
    total_salespersons * initial_avg_income) ∧
   (x * branch_income_factor * initial_avg_income ≥ 
    0.5 * total_salespersons * initial_avg_income)) →
  (x = 15 ∨ x = 16) :=
by sorry

end salesperson_allocation_l500_50005


namespace wall_photo_dimensions_l500_50046

/-- Given a rectangular paper with width 12 inches surrounded by a wall photo 2 inches wide,
    if the area of the wall photo is 96 square inches,
    then the length of the rectangular paper is 2 inches. -/
theorem wall_photo_dimensions (paper_length : ℝ) : 
  (paper_length + 4) * 16 = 96 → paper_length = 2 := by
  sorry

end wall_photo_dimensions_l500_50046


namespace hypergeom_problem_l500_50044

/-- Hypergeometric distribution parameters -/
structure HyperGeomParams where
  N : ℕ  -- Population size
  M : ℕ  -- Number of successes in the population
  n : ℕ  -- Number of draws
  h1 : M ≤ N
  h2 : n ≤ N

/-- Probability of k successes in n draws -/
def prob_k_successes (p : HyperGeomParams) (k : ℕ) : ℚ :=
  (Nat.choose p.M k * Nat.choose (p.N - p.M) (p.n - k)) / Nat.choose p.N p.n

/-- Expected value of hypergeometric distribution -/
def expected_value (p : HyperGeomParams) : ℚ :=
  (p.n * p.M : ℚ) / p.N

/-- Theorem for the specific problem -/
theorem hypergeom_problem (p : HyperGeomParams) 
    (h3 : p.N = 10) (h4 : p.M = 5) (h5 : p.n = 4) : 
    prob_k_successes p 3 = 5 / 21 ∧ expected_value p = 2 := by
  sorry


end hypergeom_problem_l500_50044


namespace pension_fund_strategy_optimizes_portfolio_l500_50043

/-- Represents different types of assets --/
inductive AssetType
  | DebtAsset
  | EquityAsset

/-- Represents an investment portfolio --/
structure Portfolio where
  debtAssets : ℝ
  equityAssets : ℝ

/-- Represents the investment strategy --/
structure InvestmentStrategy where
  portfolio : Portfolio
  maxEquityProportion : ℝ

/-- Defines the concept of a balanced portfolio --/
def isBalanced (s : InvestmentStrategy) : Prop :=
  s.portfolio.equityAssets / (s.portfolio.debtAssets + s.portfolio.equityAssets) ≤ s.maxEquityProportion

/-- Defines the concept of an optimized portfolio --/
def isOptimized (s : InvestmentStrategy) : Prop :=
  isBalanced s ∧ s.portfolio.equityAssets > 0 ∧ s.portfolio.debtAssets > 0

/-- Main theorem: The investment strategy optimizes the portfolio and balances returns and risks --/
theorem pension_fund_strategy_optimizes_portfolio (s : InvestmentStrategy) 
  (h1 : s.portfolio.debtAssets > 0)
  (h2 : s.portfolio.equityAssets > 0)
  (h3 : s.maxEquityProportion = 0.3)
  (h4 : isBalanced s) :
  isOptimized s :=
sorry


end pension_fund_strategy_optimizes_portfolio_l500_50043


namespace last_gift_probability_theorem_l500_50058

/-- Represents a circular arrangement of houses -/
structure CircularArrangement where
  numHouses : ℕ
  startHouse : ℕ

/-- Probability of moving to either neighbor -/
def moveProbability : ℚ := 1/2

/-- The probability that a specific house is the last to receive a gift -/
def lastGiftProbability (ca : CircularArrangement) : ℚ :=
  1 / (ca.numHouses - 1 : ℚ)

theorem last_gift_probability_theorem (ca : CircularArrangement) 
  (h1 : ca.numHouses = 2014) 
  (h2 : ca.startHouse < ca.numHouses) 
  (h3 : moveProbability = 1/2) :
  lastGiftProbability ca = 1/2013 := by
  sorry

end last_gift_probability_theorem_l500_50058


namespace coin_value_equality_l500_50069

theorem coin_value_equality (n : ℕ) : 
  25 * 25 + 20 * 10 = 15 * 25 + 10 * 10 + n * 50 → n = 7 :=
by sorry

end coin_value_equality_l500_50069


namespace triangle_area_with_given_base_and_height_l500_50093

/-- The area of a triangle with base 8 cm and height 10 cm is 40 square centimeters. -/
theorem triangle_area_with_given_base_and_height :
  let base : ℝ := 8
  let height : ℝ := 10
  let area : ℝ := (1 / 2) * base * height
  area = 40 := by sorry

end triangle_area_with_given_base_and_height_l500_50093


namespace f_problem_l500_50063

noncomputable section

variable (f : ℝ → ℝ)

axiom f_increasing : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom f_domain : ∀ x, 0 < x → ∃ y, f x = y
axiom f_property : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y
axiom f_6 : f 6 = 1

theorem f_problem :
  (f 1 = 0) ∧
  (∀ x, 0 < x → (f (x + 3) - f (1 / x) < 2 ↔ 0 < x ∧ x < (-3 + 3 * Real.sqrt 17) / 2)) :=
by sorry

end

end f_problem_l500_50063


namespace intersection_point_equality_l500_50072

/-- Given a system of linear equations and its solution, prove that the intersection
    point of two related lines is the same as the solution. -/
theorem intersection_point_equality (x y : ℝ) :
  x - y = -5 →
  x + 2*y = -2 →
  x = -4 →
  y = 1 →
  ∃! (x' y' : ℝ), y' = x' + 5 ∧ y' = -1/2 * x' - 1 ∧ x' = -4 ∧ y' = 1 :=
by sorry


end intersection_point_equality_l500_50072


namespace star_calculation_l500_50034

-- Define the new operation
def star (m n : Int) : Int := m - n + 1

-- Theorem statement
theorem star_calculation : star (star 2 3) 2 = -1 := by
  sorry

end star_calculation_l500_50034


namespace function_inequality_l500_50036

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x ∈ (Set.Ioo 0 (π/2)), f' x = deriv f x) →
  (∀ x ∈ (Set.Ioo 0 (π/2)), f x - f' x * (tan x) < 0) →
  (f 1 / sin 1 > 2 * f (π/6)) :=
sorry

end function_inequality_l500_50036


namespace annual_growth_rate_annual_growth_rate_proof_l500_50047

/-- Given a monthly growth rate, calculate the annual growth rate -/
theorem annual_growth_rate (P : ℝ) : ℝ := 
  (1 + P)^11 - 1

/-- The annual growth rate is equal to (1+P)^11 - 1, where P is the monthly growth rate -/
theorem annual_growth_rate_proof (P : ℝ) : 
  annual_growth_rate P = (1 + P)^11 - 1 := by
  sorry

end annual_growth_rate_annual_growth_rate_proof_l500_50047


namespace no_partition_of_integers_l500_50048

theorem no_partition_of_integers : ¬ ∃ (A B C : Set ℤ), 
  (A ∪ B ∪ C = Set.univ) ∧ 
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
  (∀ n : ℤ, (n ∈ A ∧ (n - 50) ∈ B ∧ (n + 2011) ∈ C) ∨
            (n ∈ A ∧ (n - 50) ∈ C ∧ (n + 2011) ∈ B) ∨
            (n ∈ B ∧ (n - 50) ∈ A ∧ (n + 2011) ∈ C) ∨
            (n ∈ B ∧ (n - 50) ∈ C ∧ (n + 2011) ∈ A) ∨
            (n ∈ C ∧ (n - 50) ∈ A ∧ (n + 2011) ∈ B) ∨
            (n ∈ C ∧ (n - 50) ∈ B ∧ (n + 2011) ∈ A)) :=
by
  sorry


end no_partition_of_integers_l500_50048


namespace quartic_equation_solutions_l500_50032

theorem quartic_equation_solutions :
  ∀ x : ℝ, x^4 - x^2 - 2 = 0 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 := by
  sorry

end quartic_equation_solutions_l500_50032


namespace aubrey_distance_to_school_l500_50029

/-- The distance from Aubrey's home to his school -/
def distance_to_school (journey_time : ℝ) (average_speed : ℝ) : ℝ :=
  journey_time * average_speed

/-- Theorem stating the distance from Aubrey's home to his school -/
theorem aubrey_distance_to_school :
  distance_to_school 4 22 = 88 := by
  sorry

end aubrey_distance_to_school_l500_50029


namespace paint_cost_per_quart_paint_cost_example_l500_50020

/-- The cost of paint per quart for a cube with given dimensions and coverage -/
theorem paint_cost_per_quart (cube_side : ℝ) (coverage_per_quart : ℝ) (total_cost : ℝ) : ℝ :=
  let surface_area := 6 * cube_side^2
  let quarts_needed := surface_area / coverage_per_quart
  total_cost / quarts_needed

/-- The cost of paint per quart is $3.20 for the given conditions -/
theorem paint_cost_example : paint_cost_per_quart 10 120 16 = 3.20 := by
  sorry

end paint_cost_per_quart_paint_cost_example_l500_50020


namespace trigonometric_identities_l500_50068

theorem trigonometric_identities :
  -- Part 1
  (Real.sin (76 * π / 180) * Real.cos (74 * π / 180) + Real.sin (14 * π / 180) * Real.cos (16 * π / 180) = 1/2) ∧
  -- Part 2
  ((1 - Real.tan (59 * π / 180)) * (1 - Real.tan (76 * π / 180)) = 2) ∧
  -- Part 3
  ((Real.sin (7 * π / 180) + Real.cos (15 * π / 180) * Real.sin (8 * π / 180)) / 
   (Real.cos (7 * π / 180) - Real.sin (15 * π / 180) * Real.sin (8 * π / 180)) = 2 - Real.sqrt 3) := by
  sorry

end trigonometric_identities_l500_50068


namespace second_term_of_geometric_series_l500_50051

theorem second_term_of_geometric_series 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = 1 / 4) 
  (h2 : S = 10) 
  (h3 : S = a / (1 - r)) 
  (h4 : second_term = a * r) : second_term = 1.875 :=
by
  sorry

end second_term_of_geometric_series_l500_50051


namespace min_value_quadratic_l500_50016

theorem min_value_quadratic :
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) :=
by sorry

end min_value_quadratic_l500_50016


namespace circle_line_intersection_l500_50027

/-- Given a circle (x+a)^2 + y^2 = 4 and a line x - y - 4 = 0 intersecting the circle
    to form a chord of length 2√2, prove that a = -2 or a = -6 -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, (x + a)^2 + y^2 = 4 ∧ x - y - 4 = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + a)^2 + y₁^2 = 4 ∧ x₁ - y₁ - 4 = 0 ∧
    (x₂ + a)^2 + y₂^2 = 4 ∧ x₂ - y₂ - 4 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →
  a = -2 ∨ a = -6 :=
by sorry

end circle_line_intersection_l500_50027


namespace minimum_angle_after_8_steps_l500_50060

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Vector type -/
structure Vector2D where
  x : ℕ
  y : ℕ

/-- Function to perform one step of vector replacement -/
def replaceVector (v1 v2 : Vector2D) : (Vector2D × Vector2D) :=
  if v1.x * v1.x + v1.y * v1.y ≤ v2.x * v2.x + v2.y * v2.y then
    ({ x := v1.x + v2.x, y := v1.y + v2.y }, v2)
  else
    (v1, { x := v1.x + v2.x, y := v1.y + v2.y })

/-- Function to perform n steps of vector replacement -/
def performSteps (n : ℕ) (v1 v2 : Vector2D) : (Vector2D × Vector2D) :=
  match n with
  | 0 => (v1, v2)
  | n + 1 => 
    let (newV1, newV2) := replaceVector v1 v2
    performSteps n newV1 newV2

/-- Cotangent of the angle between two vectors -/
def cotangentAngle (v1 v2 : Vector2D) : ℚ :=
  let dotProduct := v1.x * v2.x + v1.y * v2.y
  let crossProduct := v1.x * v2.y - v1.y * v2.x
  dotProduct / crossProduct

/-- Main theorem -/
theorem minimum_angle_after_8_steps : 
  let initialV1 : Vector2D := { x := 1, y := 0 }
  let initialV2 : Vector2D := { x := 0, y := 1 }
  let (finalV1, finalV2) := performSteps 8 initialV1 initialV2
  cotangentAngle finalV1 finalV2 = 987 := by sorry

end minimum_angle_after_8_steps_l500_50060


namespace kayak_production_sum_l500_50050

/-- Calculates the sum of a geometric sequence -/
def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

/-- The number of kayaks built in the first month -/
def initial_kayaks : ℕ := 8

/-- The ratio of kayaks built between consecutive months -/
def kayak_ratio : ℕ := 3

/-- The number of months of kayak production -/
def num_months : ℕ := 6

theorem kayak_production_sum :
  geometric_sum initial_kayaks kayak_ratio num_months = 2912 := by
  sorry

end kayak_production_sum_l500_50050


namespace regular_tile_area_theorem_l500_50019

/-- Represents the properties of a tiled wall -/
structure TiledWall where
  total_area : ℝ
  regular_tile_length : ℝ
  regular_tile_width : ℝ
  jumbo_tile_length : ℝ
  jumbo_tile_width : ℝ
  jumbo_tile_ratio : ℝ
  regular_tile_count_ratio : ℝ

/-- The area covered by regular tiles in a tiled wall -/
def regular_tile_area (wall : TiledWall) : ℝ :=
  wall.total_area * wall.regular_tile_count_ratio

/-- Theorem stating the area covered by regular tiles in a specific wall configuration -/
theorem regular_tile_area_theorem (wall : TiledWall) 
  (h1 : wall.total_area = 220)
  (h2 : wall.jumbo_tile_ratio = 1/3)
  (h3 : wall.regular_tile_count_ratio = 2/3)
  (h4 : wall.jumbo_tile_length = 3 * wall.regular_tile_length)
  (h5 : wall.jumbo_tile_width = wall.regular_tile_width)
  : regular_tile_area wall = 146.67 := by
  sorry

#check regular_tile_area_theorem

end regular_tile_area_theorem_l500_50019


namespace hyperbola_through_points_l500_50042

/-- The standard form of a hyperbola passing through two given points -/
theorem hyperbola_through_points :
  let P₁ : ℝ × ℝ := (3, -4 * Real.sqrt 2)
  let P₂ : ℝ × ℝ := (9/4, 5)
  let hyperbola (x y : ℝ) := 49 * x^2 - 7 * y^2 = 113
  (hyperbola P₁.1 P₁.2) ∧ (hyperbola P₂.1 P₂.2) := by sorry

end hyperbola_through_points_l500_50042


namespace min_turns_10x10_grid_l500_50031

/-- Represents a city grid -/
structure CityGrid where
  parallel_streets : ℕ
  intersecting_streets : ℕ

/-- Represents a bus route in the city -/
structure BusRoute where
  turns : ℕ
  closed : Bool
  covers_all_intersections : Bool

/-- The minimum number of turns for a valid bus route -/
def min_turns (city : CityGrid) : ℕ := 2 * (city.parallel_streets + city.intersecting_streets)

/-- Theorem stating the minimum number of turns for a 10x10 grid city -/
theorem min_turns_10x10_grid :
  let city : CityGrid := ⟨10, 10⟩
  let route : BusRoute := ⟨min_turns city, true, true⟩
  route.turns = 20 ∧
  ∀ (other_route : BusRoute),
    (other_route.closed ∧ other_route.covers_all_intersections) →
    other_route.turns ≥ route.turns :=
by sorry

end min_turns_10x10_grid_l500_50031


namespace teacher_age_l500_50000

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 23 →
  student_avg_age = 22 →
  new_avg_age = student_avg_age + 1 →
  (num_students : ℝ) * student_avg_age + (new_avg_age * (num_students + 1) - student_avg_age * num_students) = 46 * (num_students + 1) :=
by
  sorry

end teacher_age_l500_50000


namespace intersection_theorem_l500_50022

def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

def N : Set ℝ := {x | ∃ y, y = Real.log (1 - x^2)}

theorem intersection_theorem : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_theorem_l500_50022


namespace fg_length_l500_50077

/-- Represents a parallelogram ABCD and a right triangle DFG with specific properties -/
structure GeometricFigures where
  AB : ℝ
  AD : ℝ
  DG : ℝ
  area_equality : AB * AD = 1/2 * DG * AB

/-- The length of FG in the given geometric configuration is 8 -/
theorem fg_length (figures : GeometricFigures) 
  (h1 : figures.AB = 8)
  (h2 : figures.AD = 3)
  (h3 : figures.DG = 6) :
  figures.AB = 8 := by sorry

end fg_length_l500_50077


namespace words_lost_proof_l500_50033

/-- The number of letters in the language --/
def num_letters : ℕ := 69

/-- The index of the forbidden letter --/
def forbidden_letter_index : ℕ := 7

/-- The number of words lost due to prohibition --/
def words_lost : ℕ := 139

/-- Theorem stating the number of words lost due to prohibition --/
theorem words_lost_proof :
  (num_letters : ℕ) = 69 →
  (forbidden_letter_index : ℕ) = 7 →
  (words_lost : ℕ) = 139 :=
by
  sorry

#check words_lost_proof

end words_lost_proof_l500_50033


namespace marble_difference_is_negative_21_l500_50023

/-- The number of marbles Jonny has minus the number of marbles Marissa has -/
def marbleDifference : ℤ :=
  let mara_marbles := 12 * 2
  let markus_marbles := 2 * 13
  let jonny_marbles := 18
  let marissa_marbles := 3 * 5 + 3 * 8
  jonny_marbles - marissa_marbles

/-- Theorem stating the difference in marbles between Jonny and Marissa -/
theorem marble_difference_is_negative_21 : marbleDifference = -21 := by
  sorry

end marble_difference_is_negative_21_l500_50023


namespace unique_solution_system_l500_50025

theorem unique_solution_system (x y : ℝ) :
  x^2 + y^2 = 2 ∧ 
  (x^2 / (2 - y)) + (y^2 / (2 - x)) = 2 →
  x = 1 ∧ y = 1 :=
by sorry

end unique_solution_system_l500_50025
