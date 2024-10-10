import Mathlib

namespace liquid_rise_ratio_l3099_309900

/-- Represents a right circular cone filled with liquid -/
structure LiquidCone where
  radius : ℝ
  height : ℝ
  volume : ℝ

/-- Represents the scenario with two cones and a marble -/
structure TwoConesScenario where
  narrow_cone : LiquidCone
  wide_cone : LiquidCone
  marble_radius : ℝ

/-- The rise of liquid level in a cone after dropping the marble -/
def liquid_rise (cone : LiquidCone) (marble_volume : ℝ) : ℝ :=
  sorry

theorem liquid_rise_ratio (scenario : TwoConesScenario) :
  scenario.narrow_cone.radius = 4 ∧
  scenario.wide_cone.radius = 8 ∧
  scenario.narrow_cone.volume = scenario.wide_cone.volume ∧
  scenario.marble_radius = 1.5 →
  let marble_volume := (4/3) * Real.pi * scenario.marble_radius^3
  (liquid_rise scenario.narrow_cone marble_volume) /
  (liquid_rise scenario.wide_cone marble_volume) = 4 := by
  sorry

end liquid_rise_ratio_l3099_309900


namespace cost_of_six_books_cost_of_six_books_proof_l3099_309966

/-- Given that two identical books cost $36, prove that six of these books cost $108. -/
theorem cost_of_six_books : ℝ → Prop :=
  fun (cost_of_two_books : ℝ) =>
    cost_of_two_books = 36 →
    6 * (cost_of_two_books / 2) = 108

-- The proof goes here
theorem cost_of_six_books_proof : cost_of_six_books 36 := by
  sorry

end cost_of_six_books_cost_of_six_books_proof_l3099_309966


namespace inequality_of_four_terms_l3099_309950

theorem inequality_of_four_terms (x y z w : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + x)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) > 2 :=
sorry

end inequality_of_four_terms_l3099_309950


namespace representatives_formula_l3099_309921

/-- Represents the number of representatives for a given number of students -/
def num_representatives (x : ℕ) : ℕ :=
  if x % 10 > 6 then
    x / 10 + 1
  else
    x / 10

/-- The greatest integer function -/
def floor (r : ℚ) : ℤ :=
  ⌊r⌋

theorem representatives_formula (x : ℕ) :
  (num_representatives x : ℤ) = floor ((x + 3 : ℚ) / 10) :=
sorry

end representatives_formula_l3099_309921


namespace perpendicular_vectors_m_value_l3099_309941

def vector_a : ℝ × ℝ := (1, -2)
def vector_b (m : ℝ) : ℝ × ℝ := (6, m)

theorem perpendicular_vectors_m_value :
  (∀ m : ℝ, vector_a.1 * (vector_b m).1 + vector_a.2 * (vector_b m).2 = 0) →
  (∃ m : ℝ, vector_b m = (6, 3)) :=
by sorry

end perpendicular_vectors_m_value_l3099_309941


namespace unique_solution_absolute_value_equation_l3099_309907

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 25| + |x - 21| = |2*x - 42| :=
by
  -- The proof goes here
  sorry

end unique_solution_absolute_value_equation_l3099_309907


namespace parkway_elementary_soccer_l3099_309968

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (boys_soccer_percentage : ℚ) :
  total_students = 420 →
  boys = 312 →
  soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  (total_students - boys - (soccer_players - (boys_soccer_percentage * soccer_players).floor)) = 73 :=
by
  sorry

end parkway_elementary_soccer_l3099_309968


namespace unique_solution_system_l3099_309981

theorem unique_solution_system (a b c d : ℝ) : 
  (a * b + c + d = 3) ∧
  (b * c + d + a = 5) ∧
  (c * d + a + b = 2) ∧
  (d * a + b + c = 6) →
  (a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3) :=
by sorry

end unique_solution_system_l3099_309981


namespace solution_set_implies_m_value_l3099_309946

theorem solution_set_implies_m_value (m : ℝ) :
  (∀ x : ℝ, x^2 - (m + 2) * x > 0 ↔ x < 0 ∨ x > 2) →
  m = 0 := by
  sorry

end solution_set_implies_m_value_l3099_309946


namespace solution_to_equation_l3099_309956

theorem solution_to_equation (x : ℝ) : 
  (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ 
  (x = 2 ∨ x = -2) := by
sorry

end solution_to_equation_l3099_309956


namespace lcm_1260_980_l3099_309910

theorem lcm_1260_980 : Nat.lcm 1260 980 = 8820 := by
  sorry

end lcm_1260_980_l3099_309910


namespace cos_difference_l3099_309943

theorem cos_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1) 
  (h2 : Real.cos A + Real.cos B = 3/2) : 
  Real.cos (A - B) = 5/8 := by
sorry

end cos_difference_l3099_309943


namespace min_value_expression_lower_bound_achievable_l3099_309992

theorem min_value_expression (x : ℝ) : 
  (x^2 + 11) / Real.sqrt (x^2 + 5) ≥ 2 * Real.sqrt 6 := by
  sorry

theorem lower_bound_achievable : 
  ∃ x : ℝ, (x^2 + 11) / Real.sqrt (x^2 + 5) = 2 * Real.sqrt 6 := by
  sorry

end min_value_expression_lower_bound_achievable_l3099_309992


namespace trucks_needed_l3099_309962

def total_apples : ℕ := 42
def transported_apples : ℕ := 22
def truck_capacity : ℕ := 4

theorem trucks_needed : 
  (total_apples - transported_apples) / truck_capacity = 5 := by
  sorry

end trucks_needed_l3099_309962


namespace matrix_equation_proof_l3099_309995

theorem matrix_equation_proof : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![4, 4; 2, 4]
  N^4 - 3 • N^3 + 3 • N^2 - N = !![16, 24; 8, 12] := by
  sorry

end matrix_equation_proof_l3099_309995


namespace bird_migration_l3099_309938

theorem bird_migration (total : ℕ) (to_asia : ℕ) (difference : ℕ) : ℕ :=
  let to_africa := to_asia + difference
  to_africa

#check bird_migration 8 31 11 = 42

end bird_migration_l3099_309938


namespace subtract_negative_self_l3099_309954

theorem subtract_negative_self (a : ℤ) : -a - (-a) = 0 := by sorry

end subtract_negative_self_l3099_309954


namespace greatest_k_for_root_difference_l3099_309998

theorem greatest_k_for_root_difference (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + k*x₁ + 8 = 0 ∧ 
    x₂^2 + k*x₂ + 8 = 0 ∧ 
    |x₁ - x₂| = 2*Real.sqrt 15) →
  k ≤ Real.sqrt 92 :=
by sorry

end greatest_k_for_root_difference_l3099_309998


namespace combined_earnings_proof_l3099_309926

/-- Given Dwayne's annual earnings and the difference between Brady's and Dwayne's earnings,
    calculate their combined annual earnings. -/
def combinedEarnings (dwayneEarnings : ℕ) (earningsDifference : ℕ) : ℕ :=
  dwayneEarnings + (dwayneEarnings + earningsDifference)

/-- Theorem stating that given the specific values from the problem,
    the combined earnings of Brady and Dwayne are $3450. -/
theorem combined_earnings_proof :
  combinedEarnings 1500 450 = 3450 := by
  sorry

end combined_earnings_proof_l3099_309926


namespace expected_difference_coffee_tea_days_l3099_309984

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieOutcome
| One
| Two
| Three
| Four
| Five
| Six
| Seven
| Eight

/-- Represents the drink choice based on the die roll -/
inductive DrinkChoice
| Coffee
| Tea

/-- Function to determine the drink choice based on the die outcome -/
def choosedrink (outcome : DieOutcome) : DrinkChoice :=
  match outcome with
  | DieOutcome.Two | DieOutcome.Three | DieOutcome.Five | DieOutcome.Seven => DrinkChoice.Coffee
  | _ => DrinkChoice.Tea

/-- Number of days in a leap year -/
def leapYearDays : Nat := 366

/-- Probability of rolling a number that results in drinking coffee -/
def probCoffee : ℚ := 4 / 7

/-- Probability of rolling a number that results in drinking tea -/
def probTea : ℚ := 3 / 7

/-- Expected number of days drinking coffee in a leap year -/
def expectedCoffeeDays : ℚ := probCoffee * leapYearDays

/-- Expected number of days drinking tea in a leap year -/
def expectedTeaDays : ℚ := probTea * leapYearDays

/-- Theorem stating the expected difference between coffee and tea days -/
theorem expected_difference_coffee_tea_days :
  ⌊expectedCoffeeDays - expectedTeaDays⌋ = 52 := by sorry

end expected_difference_coffee_tea_days_l3099_309984


namespace consecutive_sum_iff_not_power_of_two_l3099_309976

theorem consecutive_sum_iff_not_power_of_two (n : ℕ) (h : n ≥ 3) :
  (∃ (a k : ℕ), k > 0 ∧ n = (k * (2 * a + k - 1)) / 2) ↔ ¬∃ (m : ℕ), n = 2^m :=
sorry

end consecutive_sum_iff_not_power_of_two_l3099_309976


namespace new_time_ratio_l3099_309980

-- Define the distances and speed ratio
def first_trip_distance : ℝ := 100
def second_trip_distance : ℝ := 500
def speed_ratio : ℝ := 4

-- Theorem statement
theorem new_time_ratio (v : ℝ) (hv : v > 0) :
  let t1 := first_trip_distance / v
  let t2 := second_trip_distance / (speed_ratio * v)
  t2 / t1 = 1.25 := by
sorry

end new_time_ratio_l3099_309980


namespace arithmetic_sequence_sum_l3099_309957

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 1) →  -- arithmetic sequence with common difference 1
  (a 2 + a 4 + a 6 = 9) →       -- given condition
  (a 5 + a 7 + a 9 = 18) :=     -- conclusion to prove
by
  sorry

end arithmetic_sequence_sum_l3099_309957


namespace find_k_value_l3099_309948

theorem find_k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) →
  k = 6 := by
sorry

end find_k_value_l3099_309948


namespace min_coach_handshakes_zero_l3099_309994

/-- The total number of handshakes in the gymnastics competition -/
def total_handshakes : ℕ := 325

/-- The number of gymnasts -/
def num_gymnasts : ℕ := 26

/-- The number of handshakes between gymnasts -/
def gymnast_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of handshakes involving coaches -/
def coach_handshakes (total : ℕ) (n : ℕ) : ℕ := total - gymnast_handshakes n

theorem min_coach_handshakes_zero :
  coach_handshakes total_handshakes num_gymnasts = 0 :=
by sorry

end min_coach_handshakes_zero_l3099_309994


namespace remainder_twice_sum_first_150_l3099_309929

theorem remainder_twice_sum_first_150 : 
  (2 * (List.range 150).sum) % 10000 = 2650 := by
  sorry

end remainder_twice_sum_first_150_l3099_309929


namespace smallest_product_l3099_309974

def digits : List Nat := [4, 5, 6, 7]

def valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat, valid_arrangement a b c d →
    product a b c d ≥ 2622 :=
by sorry

end smallest_product_l3099_309974


namespace special_number_is_perfect_square_l3099_309963

theorem special_number_is_perfect_square (n : ℕ) :
  ∃ k : ℕ, 4 * 10^(2*n+2) - 4 * 10^(n+1) + 1 = k^2 := by
  sorry

end special_number_is_perfect_square_l3099_309963


namespace line_passes_through_P_and_parallel_to_tangent_l3099_309906

-- Define the curve
def f (x : ℝ) : ℝ := 3*x^2 - 4*x + 2

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the point M
def M : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at M
def m : ℝ := (6 * M.1 - 4)

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 2*x - y + 4 = 0

theorem line_passes_through_P_and_parallel_to_tangent :
  line_equation P.1 P.2 ∧
  ∀ (x y : ℝ), line_equation x y → (y - P.2) = m * (x - P.1) :=
sorry

end line_passes_through_P_and_parallel_to_tangent_l3099_309906


namespace first_level_spots_l3099_309952

/-- Represents the number of open parking spots on each level of a 4-story parking area -/
structure ParkingArea where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The parking area satisfies the given conditions -/
def validParkingArea (p : ParkingArea) : Prop :=
  p.second = p.first + 7 ∧
  p.third = p.second + 6 ∧
  p.fourth = 14 ∧
  p.first + p.second + p.third + p.fourth = 46

theorem first_level_spots (p : ParkingArea) (h : validParkingArea p) : p.first = 4 := by
  sorry

end first_level_spots_l3099_309952


namespace pencil_and_pen_choices_l3099_309983

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (set1_size : ℕ) (set2_size : ℕ) : ℕ :=
  set1_size * set2_size

/-- Theorem: Choosing one item from a set of 3 and one from a set of 5 results in 15 possibilities -/
theorem pencil_and_pen_choices :
  choose_one_from_each 3 5 = 15 := by
  sorry

end pencil_and_pen_choices_l3099_309983


namespace substitution_theorem_l3099_309917

def num_players : ℕ := 15
def starting_players : ℕ := 5
def bench_players : ℕ := 10
def max_substitutions : ℕ := 4

def substitution_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => 5 * (11 - k) * substitution_ways k

def total_substitution_ways : ℕ :=
  List.sum (List.map substitution_ways (List.range (max_substitutions + 1)))

theorem substitution_theorem :
  total_substitution_ways = 5073556 ∧
  total_substitution_ways % 100 = 56 := by
  sorry

end substitution_theorem_l3099_309917


namespace prob_at_least_one_target_l3099_309944

/-- The number of cards in the modified deck -/
def deck_size : ℕ := 54

/-- The number of cards that are diamonds, aces, or jokers -/
def target_cards : ℕ := 18

/-- The probability of drawing a card that is not a diamond, ace, or joker -/
def prob_not_target : ℚ := (deck_size - target_cards) / deck_size

/-- The probability of drawing two cards with replacement, where at least one is a diamond, ace, or joker -/
theorem prob_at_least_one_target : 
  1 - prob_not_target ^ 2 = 5 / 9 := by sorry

end prob_at_least_one_target_l3099_309944


namespace root_sum_reciprocals_l3099_309989

theorem root_sum_reciprocals (p q r s t : ℂ) : 
  p^5 + 10*p^4 + 20*p^3 + 15*p^2 + 8*p + 5 = 0 →
  q^5 + 10*q^4 + 20*q^3 + 15*q^2 + 8*q + 5 = 0 →
  r^5 + 10*r^4 + 20*r^3 + 15*r^2 + 8*r + 5 = 0 →
  s^5 + 10*s^4 + 20*s^3 + 15*s^2 + 8*s + 5 = 0 →
  t^5 + 10*t^4 + 20*t^3 + 15*t^2 + 8*t + 5 = 0 →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 4 := by
sorry

end root_sum_reciprocals_l3099_309989


namespace unique_plane_through_three_points_perpendicular_line_implies_parallel_planes_parallel_to_plane_not_implies_parallel_lines_perpendicular_to_plane_implies_parallel_lines_l3099_309913

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (collinear : Point → Point → Point → Prop)
variable (on_plane : Point → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Proposition A
theorem unique_plane_through_three_points 
  (p q r : Point) (h : ¬ collinear p q r) :
  ∃! π : Plane, on_plane p π ∧ on_plane q π ∧ on_plane r π :=
sorry

-- Proposition B
theorem perpendicular_line_implies_parallel_planes 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular m β) :
  parallel_planes α β :=
sorry

-- Proposition C (negation)
theorem parallel_to_plane_not_implies_parallel_lines 
  (m n : Line) (α : Plane) :
  parallel_line_plane m α ∧ parallel_line_plane n α → 
  ¬ (parallel_lines m n → True) :=
sorry

-- Proposition D
theorem perpendicular_to_plane_implies_parallel_lines 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular n α) :
  parallel_lines m n :=
sorry

end unique_plane_through_three_points_perpendicular_line_implies_parallel_planes_parallel_to_plane_not_implies_parallel_lines_perpendicular_to_plane_implies_parallel_lines_l3099_309913


namespace midpoint_trajectory_l3099_309930

/-- The equation of the trajectory of the midpoint of a line segment -/
theorem midpoint_trajectory (x₁ y₁ x y : ℝ) : 
  y₁ = 2 * x₁^2 + 1 →  -- P is on the curve y = 2x^2 + 1
  x = (x₁ + 0) / 2 →   -- x-coordinate of midpoint
  y = (y₁ + (-1)) / 2  -- y-coordinate of midpoint
  → y = 4 * x^2 := by
sorry

end midpoint_trajectory_l3099_309930


namespace solve_equation_l3099_309947

theorem solve_equation : ∃ y : ℝ, (2 * y) / 3 = 30 ∧ y = 45 := by
  sorry

end solve_equation_l3099_309947


namespace division_problem_l3099_309934

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 271 →
  quotient = 9 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 30 := by
sorry

end division_problem_l3099_309934


namespace zeros_product_greater_than_e_squared_l3099_309993

/-- Given a function f(x) = (ln x) / x - a with two zeros m and n, prove that mn > e² -/
theorem zeros_product_greater_than_e_squared (a : ℝ) (m n : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (Real.log x) / x - a = 0 ∧ (Real.log y) / y - a = 0) →
  (Real.log m) / m - a = 0 →
  (Real.log n) / n - a = 0 →
  m * n > Real.exp 2 := by
  sorry

end zeros_product_greater_than_e_squared_l3099_309993


namespace repetend_of_five_seventeenths_l3099_309955

theorem repetend_of_five_seventeenths :
  ∃ (n : ℕ), (5 : ℚ) / 17 = (n : ℚ) / 999999999999 ∧ 
  n = 294117647058 :=
sorry

end repetend_of_five_seventeenths_l3099_309955


namespace smallest_prime_with_digit_sum_23_l3099_309960

def digit_sum (n : ℕ) : ℕ := sorry

def is_prime (n : ℕ) : Prop := sorry

theorem smallest_prime_with_digit_sum_23 :
  (∀ p : ℕ, is_prime p ∧ digit_sum p = 23 → p ≥ 599) ∧
  is_prime 599 ∧
  digit_sum 599 = 23 := by sorry

end smallest_prime_with_digit_sum_23_l3099_309960


namespace additional_workers_for_earlier_completion_l3099_309922

/-- Calculates the number of additional workers needed to complete a task earlier -/
def additional_workers (original_days : ℕ) (actual_days : ℕ) (original_workers : ℕ) : ℕ :=
  ⌊(original_workers * original_days / actual_days - original_workers : ℚ)⌋.toNat

/-- Proves that 6 additional workers are needed to complete the task 3 days earlier -/
theorem additional_workers_for_earlier_completion :
  additional_workers 10 7 15 = 6 := by
  sorry

#eval additional_workers 10 7 15

end additional_workers_for_earlier_completion_l3099_309922


namespace set_intersection_problem_l3099_309925

theorem set_intersection_problem (m : ℝ) : 
  let A : Set ℝ := {-1, 3, m}
  let B : Set ℝ := {3, 4}
  B ∩ A = B → m = 4 := by
sorry

end set_intersection_problem_l3099_309925


namespace list_length_contradiction_l3099_309942

theorem list_length_contradiction (list_I list_II : List ℕ) : 
  (list_I = [3, 4, 8, 19]) →
  (list_II.length = list_I.length + 1) →
  (list_II.length - list_I.length = 6) →
  False :=
by sorry

end list_length_contradiction_l3099_309942


namespace apples_distribution_l3099_309987

/-- Given 48 apples distributed evenly among 7 children, prove that 1 child receives fewer than 7 apples -/
theorem apples_distribution (total_apples : Nat) (num_children : Nat) 
  (h1 : total_apples = 48) 
  (h2 : num_children = 7) : 
  (num_children - (total_apples % num_children)) = 1 := by
  sorry

end apples_distribution_l3099_309987


namespace remainder_problem_l3099_309972

theorem remainder_problem (divisor : ℕ) (a b : ℕ) (rem_a rem_sum : ℕ) 
  (h_divisor : divisor = 13)
  (h_a : a = 242)
  (h_b : b = 698)
  (h_rem_a : a % divisor = rem_a)
  (h_rem_a_val : rem_a = 8)
  (h_rem_sum : (a + b) % divisor = rem_sum)
  (h_rem_sum_val : rem_sum = 4) :
  b % divisor = 9 := by
  sorry


end remainder_problem_l3099_309972


namespace reflection_problem_l3099_309997

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x - y + 7 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define point N
def N : ℝ × ℝ := (1, 0)

-- Define the intersection point M
def M : ℝ × ℝ := (-2, 1)

-- Define the symmetric point P
def P : ℝ × ℝ := (-2, -1)

-- Define line l₃
def l₃ (x y : ℝ) : Prop := y = (1/3) * x - (1/3)

-- Define the parallel lines at distance √10 from l₃
def parallel_line₁ (x y : ℝ) : Prop := y = (1/3) * x + 3
def parallel_line₂ (x y : ℝ) : Prop := y = (1/3) * x - (11/3)

theorem reflection_problem :
  (∀ x y, l₁ x y ∧ l₂ x y → (x, y) = M) ∧
  P = (-2, -1) ∧
  (∀ x y, l₃ x y ↔ y = (1/3) * x - (1/3)) ∧
  (∀ x y, (parallel_line₁ x y ∨ parallel_line₂ x y) ↔
    ∃ d, d = Real.sqrt 10 ∧ 
    (y - ((1/3) * x - (1/3)))^2 / (1 + (1/3)^2) = d^2) :=
by sorry

end reflection_problem_l3099_309997


namespace power_seven_mod_eight_l3099_309911

theorem power_seven_mod_eight : 7^135 % 8 = 7 := by
  sorry

end power_seven_mod_eight_l3099_309911


namespace pipe_b_rate_is_50_l3099_309953

/-- Represents the water tank system with three pipes -/
structure WaterTankSystem where
  tank_capacity : ℕ
  pipe_a_rate : ℕ
  pipe_b_rate : ℕ
  pipe_c_rate : ℕ
  cycle_time : ℕ
  total_time : ℕ

/-- Calculates the volume filled in one cycle -/
def volume_per_cycle (system : WaterTankSystem) : ℤ :=
  system.pipe_a_rate * 1 + system.pipe_b_rate * 2 - system.pipe_c_rate * 2

/-- Theorem stating that the rate of Pipe B must be 50 L/min -/
theorem pipe_b_rate_is_50 (system : WaterTankSystem) 
  (h1 : system.tank_capacity = 2000)
  (h2 : system.pipe_a_rate = 200)
  (h3 : system.pipe_c_rate = 25)
  (h4 : system.cycle_time = 5)
  (h5 : system.total_time = 40)
  (h6 : (system.total_time / system.cycle_time : ℤ) * volume_per_cycle system = system.tank_capacity) :
  system.pipe_b_rate = 50 := by
  sorry

end pipe_b_rate_is_50_l3099_309953


namespace seventh_eleventh_150th_decimal_l3099_309979

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The period length of a rational number's decimal representation -/
def periodLength (q : ℚ) : ℕ := sorry

theorem seventh_eleventh_150th_decimal :
  (7 : ℚ) / 11 ∈ {q : ℚ | periodLength q = 2 ∧ decimalRepresentation q 150 = 3} := by
  sorry

end seventh_eleventh_150th_decimal_l3099_309979


namespace bert_sandwiches_l3099_309940

def sandwiches_problem (initial_sandwiches : ℕ) : ℕ :=
  let day1_remaining := initial_sandwiches / 2
  let day2_remaining := day1_remaining - (2 * day1_remaining / 3)
  let day3_eaten := (2 * day1_remaining / 3) - 2
  day2_remaining - min day2_remaining day3_eaten

theorem bert_sandwiches :
  sandwiches_problem 36 = 0 := by
  sorry

end bert_sandwiches_l3099_309940


namespace oliver_shelf_capacity_l3099_309978

/-- The number of books Oliver can fit on a shelf -/
def books_per_shelf (total_books librarian_books shelves : ℕ) : ℕ :=
  (total_books - librarian_books) / shelves

/-- Theorem: Oliver can fit 4 books on a shelf -/
theorem oliver_shelf_capacity :
  books_per_shelf 46 10 9 = 4 := by
  sorry

end oliver_shelf_capacity_l3099_309978


namespace eating_contest_l3099_309909

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight : ℕ) 
  (noah_burgers jacob_pies mason_hotdogs : ℕ) :
  hot_dog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  noah_burgers = 8 →
  mason_hotdogs * hot_dog_weight = 30 →
  mason_hotdogs = 15 := by
  sorry

end eating_contest_l3099_309909


namespace coefficient_a2_equals_56_l3099_309920

/-- Given a polynomial equality, prove that the coefficient a₂ equals 56 -/
theorem coefficient_a2_equals_56 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) : 
  (∀ x : ℝ, 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 
    = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7) 
  → a₂ = 56 := by
  sorry

end coefficient_a2_equals_56_l3099_309920


namespace h1n1_spread_properties_l3099_309958

/-- Represents the spread of H1N1 flu in a community -/
def H1N1Spread (x : ℝ) : Prop :=
  (1 + x)^2 = 36 ∧ x > 0

theorem h1n1_spread_properties (x : ℝ) (hx : H1N1Spread x) :
  x = 5 ∧ (1 + x)^3 > 200 := by
  sorry

#check h1n1_spread_properties

end h1n1_spread_properties_l3099_309958


namespace abs_neg_three_not_pm_three_l3099_309967

theorem abs_neg_three_not_pm_three : ¬(|(-3 : ℤ)| = 3 ∧ |(-3 : ℤ)| = -3) := by
  sorry

end abs_neg_three_not_pm_three_l3099_309967


namespace coordinate_plane_conditions_l3099_309928

-- Define a point on the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions and their corresponding geometric interpretations
theorem coordinate_plane_conditions (p : Point) :
  (p.x = 3 → p ∈ {q : Point | q.x = 3}) ∧
  (p.x < 3 → p ∈ {q : Point | q.x < 3}) ∧
  (p.x > 3 → p ∈ {q : Point | q.x > 3}) ∧
  (p.y = 2 → p ∈ {q : Point | q.y = 2}) ∧
  (p.y > 2 → p ∈ {q : Point | q.y > 2}) := by
  sorry

end coordinate_plane_conditions_l3099_309928


namespace harvest_season_duration_l3099_309936

/-- Calculates the number of weeks in a harvest season based on weekly earnings, rent, and total savings. -/
def harvest_season_weeks (weekly_earnings : ℕ) (weekly_rent : ℕ) (total_savings : ℕ) : ℕ :=
  total_savings / (weekly_earnings - weekly_rent)

/-- Proves that the number of weeks in the harvest season is 1181 given the specified conditions. -/
theorem harvest_season_duration :
  harvest_season_weeks 491 216 324775 = 1181 := by
  sorry

end harvest_season_duration_l3099_309936


namespace second_train_crossing_time_l3099_309949

/-- Represents the time in seconds for two bullet trains to cross each other -/
def crossing_time : ℝ := 16.666666666666668

/-- Represents the length of each bullet train in meters -/
def train_length : ℝ := 120

/-- Represents the time in seconds for the first bullet train to cross a telegraph post -/
def first_train_time : ℝ := 10

theorem second_train_crossing_time :
  let first_train_speed := train_length / first_train_time
  let second_train_time := train_length / ((2 * train_length / crossing_time) - first_train_speed)
  second_train_time = 50 := by sorry

end second_train_crossing_time_l3099_309949


namespace least_possible_third_side_length_l3099_309933

/-- Given a right triangle with two sides of 8 units and 15 units, 
    the least possible length of the third side is √161 units. -/
theorem least_possible_third_side_length : ∀ a b c : ℝ,
  a = 8 →
  b = 15 →
  (a = c ∧ b * b = c * c - a * a) ∨ 
  (b = c ∧ a * a = c * c - b * b) ∨
  (c * c = a * a + b * b) →
  c ≥ Real.sqrt 161 :=
by
  sorry

end least_possible_third_side_length_l3099_309933


namespace bert_profit_is_correct_l3099_309965

/-- Represents a product in Bert's shop -/
structure Product where
  price : ℝ
  tax_rate : ℝ

/-- Represents a customer's purchase -/
structure Purchase where
  products : List Product
  discount_rate : ℝ

/-- Calculates Bert's profit given the purchases -/
def calculate_profit (purchases : List Purchase) : ℝ :=
  sorry

/-- The actual purchases made by customers -/
def actual_purchases : List Purchase :=
  [
    { products := [
        { price := 90, tax_rate := 0.1 },
        { price := 50, tax_rate := 0.05 }
      ], 
      discount_rate := 0.1
    },
    { products := [
        { price := 30, tax_rate := 0.12 },
        { price := 20, tax_rate := 0.03 }
      ], 
      discount_rate := 0.15
    },
    { products := [
        { price := 15, tax_rate := 0.09 }
      ], 
      discount_rate := 0
    }
  ]

/-- Bert's profit per item -/
def profit_per_item : ℝ := 10

theorem bert_profit_is_correct : 
  calculate_profit actual_purchases = 50.05 :=
sorry

end bert_profit_is_correct_l3099_309965


namespace complex_magnitude_squared_l3099_309902

theorem complex_magnitude_squared (z₁ z₂ : ℂ) :
  let z₁ : ℂ := 3 * Real.sqrt 2 - 5*I
  let z₂ : ℂ := 2 * Real.sqrt 5 + 4*I
  ‖z₁ * z₂‖^2 = 1548 := by
sorry

end complex_magnitude_squared_l3099_309902


namespace shorter_diagonal_of_parallelepiped_l3099_309912

/-- Represents a parallelepiped with a rhombus base -/
structure Parallelepiped where
  base_side : ℝ
  lateral_edge : ℝ
  lateral_angle : ℝ
  section_area : ℝ

/-- Theorem: The shorter diagonal of the base rhombus in the given parallelepiped is 60 -/
theorem shorter_diagonal_of_parallelepiped (p : Parallelepiped) 
  (h1 : p.base_side = 60)
  (h2 : p.lateral_edge = 80)
  (h3 : p.lateral_angle = Real.pi / 3)  -- 60 degrees in radians
  (h4 : p.section_area = 7200) :
  ∃ (shorter_diagonal : ℝ), shorter_diagonal = 60 :=
by sorry

end shorter_diagonal_of_parallelepiped_l3099_309912


namespace smallest_cut_length_l3099_309971

theorem smallest_cut_length (z : ℕ) : z ≥ 9 →
  (∃ x y : ℕ, x = z / 2 ∧ y = z - 2) →
  (13 - z / 2 + 22 - z ≤ 25 - z) →
  (13 - z / 2 + 25 - z ≤ 22 - z) →
  (22 - z + 25 - z ≤ 13 - z / 2) →
  ∀ w : ℕ, w ≥ 9 → w < z →
    ¬((13 - w / 2 + 22 - w ≤ 25 - w) ∧
      (13 - w / 2 + 25 - w ≤ 22 - w) ∧
      (22 - w + 25 - w ≤ 13 - w / 2)) :=
by sorry

end smallest_cut_length_l3099_309971


namespace bottom_right_not_divisible_by_2011_l3099_309969

/-- Represents a cell on the board -/
structure Cell where
  x : Nat
  y : Nat

/-- Represents the board configuration -/
structure Board where
  size : Nat
  markedCells : List Cell

/-- Counts the number of paths from (0,0) to (x,y) that don't pass through marked cells -/
def countPaths (board : Board) (x y : Nat) : Nat :=
  sorry

theorem bottom_right_not_divisible_by_2011 (board : Board) :
  board.size = 2012 →
  (∀ c ∈ board.markedCells, c.x = c.y ∧ c.x ≠ 0 ∧ c.x ≠ 2011) →
  ¬ (countPaths board 2011 2011 % 2011 = 0) :=
by sorry

end bottom_right_not_divisible_by_2011_l3099_309969


namespace chess_match_duration_l3099_309916

-- Define the given conditions
def polly_time_per_move : ℕ := 28
def peter_time_per_move : ℕ := 40
def total_moves : ℕ := 30

-- Define the theorem
theorem chess_match_duration :
  (total_moves / 2 * polly_time_per_move + total_moves / 2 * peter_time_per_move) / 60 = 17 := by
  sorry

end chess_match_duration_l3099_309916


namespace iron_chain_links_count_l3099_309932

/-- Represents a piece of an iron chain -/
structure ChainPiece where
  length : ℝ
  links : ℕ

/-- Calculates the length of a chain piece given the number of links and internal diameter -/
def chainLength (links : ℕ) (internalDiameter : ℝ) : ℝ :=
  (links : ℝ) * internalDiameter + 1

theorem iron_chain_links_count :
  let shortPiece : ChainPiece := ⟨22, 9⟩
  let longPiece : ChainPiece := ⟨36, 15⟩
  let internalDiameter : ℝ := 7/3

  (longPiece.links = shortPiece.links + 6) ∧
  (chainLength shortPiece.links internalDiameter = shortPiece.length) ∧
  (chainLength longPiece.links internalDiameter = longPiece.length) :=
by
  sorry

end iron_chain_links_count_l3099_309932


namespace a_can_ensure_segments_l3099_309977

/-- Represents a point on the circle -/
structure Point where
  has_piece : Bool

/-- Represents a segment between two points -/
structure Segment where
  point1 : Point
  point2 : Point

/-- Represents the state of the game -/
structure GameState where
  n : Nat
  points : List Point
  segments : List Segment

/-- Player A's strategy -/
def player_a_strategy (state : GameState) : GameState :=
  sorry

/-- Player B's strategy -/
def player_b_strategy (state : GameState) : GameState :=
  sorry

/-- Counts the number of segments connecting a point with a piece and a point without a piece -/
def count_valid_segments (state : GameState) : Nat :=
  sorry

/-- Main theorem -/
theorem a_can_ensure_segments (n : Nat) (h : n ≥ 2) :
  ∃ (initial_state : GameState),
    initial_state.n = n ∧
    initial_state.points.length = 3 * n ∧
    (∀ (b_strategy : GameState → GameState),
      let final_state := (player_a_strategy ∘ b_strategy)^[n] initial_state
      count_valid_segments final_state ≥ (n - 1) / 6) :=
  sorry

end a_can_ensure_segments_l3099_309977


namespace range_of_a_l3099_309919

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (-2 < x - 1 ∧ x - 1 < 3 ∧ x - a > 0) ↔ (-1 < x ∧ x < 4)) →
  a ≤ -1 :=
by sorry

end range_of_a_l3099_309919


namespace square_side_length_l3099_309937

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 1 / 9 → side * side = area → side = 1 / 3 := by
  sorry

end square_side_length_l3099_309937


namespace study_tour_problem_l3099_309901

/-- Represents a bus type with seat capacity and rental fee -/
structure BusType where
  seats : ℕ
  fee : ℕ

/-- Calculates the number of buses needed for a given number of participants and bus type -/
def busesNeeded (participants : ℕ) (busType : BusType) : ℕ :=
  (participants + busType.seats - 1) / busType.seats

/-- Calculates the total rental cost for a given number of participants and bus type -/
def rentalCost (participants : ℕ) (busType : BusType) : ℕ :=
  (busesNeeded participants busType) * busType.fee

theorem study_tour_problem (x y : ℕ) (typeA typeB : BusType)
    (h1 : 45 * y + 15 = x)
    (h2 : 60 * (y - 3) = x)
    (h3 : typeA.seats = 45)
    (h4 : typeA.fee = 200)
    (h5 : typeB.seats = 60)
    (h6 : typeB.fee = 300) :
    x = 600 ∧ y = 13 ∧ rentalCost x typeA < rentalCost x typeB := by
  sorry

end study_tour_problem_l3099_309901


namespace expo_assignment_count_l3099_309931

/-- Represents the four pavilions at the Shanghai World Expo -/
inductive Pavilion
  | China
  | UK
  | Australia
  | Russia

/-- The total number of volunteers -/
def total_volunteers : Nat := 5

/-- The number of pavilions -/
def num_pavilions : Nat := 4

/-- A function that represents a valid assignment of volunteers to pavilions -/
def is_valid_assignment (assignment : Pavilion → Nat) : Prop :=
  (∀ p : Pavilion, assignment p > 0) ∧
  (assignment Pavilion.China + assignment Pavilion.UK + 
   assignment Pavilion.Australia + assignment Pavilion.Russia = total_volunteers)

/-- The number of ways for A and B to be assigned to pavilions -/
def num_ways_AB : Nat := num_pavilions * num_pavilions

/-- The theorem to be proved -/
theorem expo_assignment_count :
  (∃ (ways : Nat), ways = num_ways_AB ∧
    ∃ (remaining_assignments : Nat),
      ways * remaining_assignments = 72 ∧
      ∀ (assignment : Pavilion → Nat),
        is_valid_assignment assignment →
        remaining_assignments > 0) := by
  sorry

end expo_assignment_count_l3099_309931


namespace quadratic_equation_solution_l3099_309986

theorem quadratic_equation_solution :
  ∃ (m n p : ℕ) (x₁ x₂ : ℚ),
    -- The equation is satisfied by both solutions
    x₁ * (5 * x₁ - 11) = -2 ∧
    x₂ * (5 * x₂ - 11) = -2 ∧
    -- Solutions are in the required form
    x₁ = (m + Real.sqrt n) / p ∧
    x₂ = (m - Real.sqrt n) / p ∧
    -- m, n, and p have a greatest common divisor of 1
    Nat.gcd m (Nat.gcd n p) = 1 ∧
    -- Sum of m, n, and p is 102
    m + n + p = 102 := by
  sorry

end quadratic_equation_solution_l3099_309986


namespace reciprocal_of_negative_two_l3099_309988

theorem reciprocal_of_negative_two :
  ∃ x : ℚ, x * (-2) = 1 ∧ x = -1/2 := by sorry

end reciprocal_of_negative_two_l3099_309988


namespace base_19_representation_of_1987_l3099_309996

theorem base_19_representation_of_1987 :
  ∃! (x y z b : ℕ), 
    x * b^2 + y * b + z = 1987 ∧
    x + y + z = 25 ∧
    x < b ∧ y < b ∧ z < b ∧
    x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 := by
  sorry

end base_19_representation_of_1987_l3099_309996


namespace simplify_expression_l3099_309904

theorem simplify_expression : -(-3) - 4 + (-5) = 3 - 4 - 5 := by sorry

end simplify_expression_l3099_309904


namespace maria_white_towels_l3099_309945

/-- The number of white towels Maria bought -/
def white_towels (green_towels given_away remaining : ℕ) : ℕ :=
  (remaining + given_away) - green_towels

/-- Proof that Maria bought 21 white towels -/
theorem maria_white_towels : 
  white_towels 35 34 22 = 21 := by
  sorry

end maria_white_towels_l3099_309945


namespace product_when_c_is_one_l3099_309970

theorem product_when_c_is_one (a b c : ℕ+) (h1 : a * b * c = a * b ^ 3) (h2 : c = 1) :
  a * b * c = a := by
  sorry

end product_when_c_is_one_l3099_309970


namespace area_is_two_l3099_309991

open Real MeasureTheory

noncomputable def area_bounded_by_curves : ℝ :=
  ∫ x in (1/Real.exp 1)..Real.exp 1, (1/x)

theorem area_is_two : area_bounded_by_curves = 2 := by
  sorry

end area_is_two_l3099_309991


namespace hyperbola_vertices_distance_l3099_309915

/-- The distance between the vertices of a hyperbola given by the equation
    16x^2 + 64x - 4y^2 + 8y + 36 = 0 is 1. -/
theorem hyperbola_vertices_distance :
  let f : ℝ → ℝ → ℝ := fun x y => 16 * x^2 + 64 * x - 4 * y^2 + 8 * y + 36
  ∃ x₁ x₂ y₁ y₂ : ℝ,
    (∀ x y, f x y = 0 ↔ 4 * (x + 2)^2 - (y - 1)^2 = 1) ∧
    (x₁, y₁) ∈ {p : ℝ × ℝ | f p.1 p.2 = 0} ∧
    (x₂, y₂) ∈ {p : ℝ × ℝ | f p.1 p.2 = 0} ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 1 ∧
    ∀ x y, f x y = 0 → (x - x₁)^2 + (y - y₁)^2 ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2 :=
by sorry

end hyperbola_vertices_distance_l3099_309915


namespace last_two_nonzero_digits_of_100_factorial_l3099_309923

-- Define 100!
def factorial_100 : ℕ := Nat.factorial 100

-- Define the function to get the last two nonzero digits
def last_two_nonzero_digits (n : ℕ) : ℕ :=
  n % 100

-- Theorem statement
theorem last_two_nonzero_digits_of_100_factorial :
  last_two_nonzero_digits (factorial_100 / (10^24)) = 76 := by
  sorry

#eval last_two_nonzero_digits (factorial_100 / (10^24))

end last_two_nonzero_digits_of_100_factorial_l3099_309923


namespace updated_mean_after_corrections_l3099_309990

/-- Calculates the updated mean of a set of observations after correcting errors -/
theorem updated_mean_after_corrections (n : ℕ) (initial_mean : ℚ) 
  (n1 n2 n3 : ℕ) (error1 error2 error3 : ℚ) : 
  n = 50 → 
  initial_mean = 200 → 
  n1 = 20 → 
  n2 = 15 → 
  n3 = 15 → 
  error1 = -6 → 
  error2 = -5 → 
  error3 = 3 → 
  (initial_mean * n + n1 * error1 + n2 * error2 + n3 * error3) / n = 197 := by
  sorry

#eval (200 * 50 + 20 * (-6) + 15 * (-5) + 15 * 3) / 50

end updated_mean_after_corrections_l3099_309990


namespace lowry_earnings_l3099_309999

/-- Calculates the total earnings from bonsai sales with discounts applied --/
def bonsai_earnings (small_price medium_price big_price : ℚ)
                    (small_discount medium_discount big_discount : ℚ)
                    (small_count medium_count big_count : ℕ)
                    (small_discount_threshold medium_discount_threshold big_discount_threshold : ℕ) : ℚ :=
  let small_total := small_price * small_count
  let medium_total := medium_price * medium_count
  let big_total := big_price * big_count
  let small_discounted := if small_count ≥ small_discount_threshold then small_total * (1 - small_discount) else small_total
  let medium_discounted := if medium_count ≥ medium_discount_threshold then medium_total * (1 - medium_discount) else medium_total
  let big_discounted := if big_count > big_discount_threshold then big_total * (1 - big_discount) else big_total
  small_discounted + medium_discounted + big_discounted

theorem lowry_earnings :
  bonsai_earnings 30 45 60 0.1 0.15 0.05 8 5 7 4 3 5 = 806.25 := by
  sorry

end lowry_earnings_l3099_309999


namespace coefficient_of_x_l3099_309939

theorem coefficient_of_x (x y : ℝ) (some : ℝ) 
  (h1 : x / (2 * y) = 3 / 2)
  (h2 : (some * x + 5 * y) / (x - 2 * y) = 26) :
  some = 7 := by
  sorry

end coefficient_of_x_l3099_309939


namespace cylinder_volume_ratio_l3099_309903

theorem cylinder_volume_ratio : 
  let cylinder1_height : ℝ := 10
  let cylinder1_circumference : ℝ := 6
  let cylinder2_height : ℝ := 6
  let cylinder2_circumference : ℝ := 10
  let volume1 := π * (cylinder1_circumference / (2 * π))^2 * cylinder1_height
  let volume2 := π * (cylinder2_circumference / (2 * π))^2 * cylinder2_height
  volume2 / volume1 = 5 / 3 := by
sorry

end cylinder_volume_ratio_l3099_309903


namespace sqrt_sum_greater_than_one_l3099_309985

theorem sqrt_sum_greater_than_one (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt a + Real.sqrt b > 1 := by
  sorry

end sqrt_sum_greater_than_one_l3099_309985


namespace farmer_apples_l3099_309924

theorem farmer_apples (apples_given_away apples_left : ℕ) 
  (h1 : apples_given_away = 88) 
  (h2 : apples_left = 39) : 
  apples_given_away + apples_left = 127 := by
  sorry

end farmer_apples_l3099_309924


namespace repair_cost_calculation_l3099_309918

-- Define the parameters
def purchase_price : ℝ := 4700
def selling_price : ℝ := 5800
def gain_percent : ℝ := 1.7543859649122806

-- Define the theorem
theorem repair_cost_calculation (repair_cost : ℝ) :
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = gain_percent →
  repair_cost = 1000 := by
  sorry

end repair_cost_calculation_l3099_309918


namespace rectangle_area_l3099_309914

/-- The area of a rectangle formed by three identical smaller rectangles -/
theorem rectangle_area (shorter_side : ℝ) (h : shorter_side = 7) : 
  let longer_side := 3 * shorter_side
  let large_rectangle_length := 3 * shorter_side
  let large_rectangle_width := longer_side
  large_rectangle_length * large_rectangle_width = 441 :=
by sorry

end rectangle_area_l3099_309914


namespace nested_sqrt_value_l3099_309905

theorem nested_sqrt_value (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 := by
  sorry

end nested_sqrt_value_l3099_309905


namespace sqrt_of_nine_l3099_309959

theorem sqrt_of_nine : Real.sqrt 9 = 3 := by
  sorry

end sqrt_of_nine_l3099_309959


namespace store_inventory_difference_l3099_309935

theorem store_inventory_difference (regular_soda diet_soda apples : ℕ) 
  (h1 : regular_soda = 72) 
  (h2 : diet_soda = 32) 
  (h3 : apples = 78) : 
  (regular_soda + diet_soda) - apples = 26 := by
  sorry

end store_inventory_difference_l3099_309935


namespace reseat_ten_women_l3099_309908

def reseat_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | k + 3 => reseat_ways (k + 2) + reseat_ways (k + 1)

theorem reseat_ten_women :
  reseat_ways 10 = 89 :=
by sorry

end reseat_ten_women_l3099_309908


namespace ellipse_parabola_intersection_range_l3099_309951

/-- The range of 'a' for which the ellipse x^2 + 4(y - a)^2 = 4 and the parabola x^2 = 2y intersect -/
theorem ellipse_parabola_intersection_range :
  ∀ (a : ℝ),
  (∃ (x y : ℝ), x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) →
  -1 ≤ a ∧ a ≤ 17/8 :=
by sorry

end ellipse_parabola_intersection_range_l3099_309951


namespace pizza_slices_left_is_three_l3099_309973

/-- Calculates the number of pizza slices left after John and Sam eat -/
def pizza_slices_left (total : ℕ) (john_ate : ℕ) (sam_ate_multiplier : ℕ) : ℕ :=
  total - (john_ate + sam_ate_multiplier * john_ate)

/-- Theorem: The number of pizza slices left is 3 -/
theorem pizza_slices_left_is_three :
  pizza_slices_left 12 3 2 = 3 := by
  sorry

#eval pizza_slices_left 12 3 2

end pizza_slices_left_is_three_l3099_309973


namespace parabola_hyperbola_focus_l3099_309975

/-- The value of p for which the focus of the parabola y² = 2px coincides with 
    the right focus of the hyperbola x²/4 - y²/5 = 1 -/
theorem parabola_hyperbola_focus (p : ℝ) : 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2/4 - y^2/5 = 1 ∧ 
   x = (Real.sqrt (4 + 5 : ℝ)) ∧ y = 0) → 
  p = 6 := by sorry

end parabola_hyperbola_focus_l3099_309975


namespace tangent_line_at_one_inequality_holds_l3099_309961

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*a * Real.log x + (a-2)*x

-- Part 1: Tangent line equation when a = 1
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 4*x + 2*y - 3 = 0 :=
sorry

-- Part 2: Inequality holds for a ≤ -1/2
theorem inequality_holds (a : ℝ) (h : a ≤ -1/2) :
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
    (f a x₂ - f a x₁) / (x₂ - x₁) > a :=
sorry

end tangent_line_at_one_inequality_holds_l3099_309961


namespace translation_correctness_given_translations_correct_l3099_309927

/-- Represents a word in either Russian or Kurdish -/
structure Word :=
  (value : String)

/-- Represents a sentence in either Russian or Kurdish -/
structure Sentence :=
  (words : List Word)

/-- Defines the rules for Kurdish sentence structure -/
def kurdishSentenceStructure (s : Sentence) : Prop :=
  -- The predicate is at the end of the sentence
  -- The subject starts the sentence, followed by the object
  -- Noun-adjective constructions follow the "S (adjective determinant) O (determined word)" structure
  -- The determined word takes the suffix "e"
  sorry

/-- Translates a Russian sentence to Kurdish -/
def translateToKurdish (russianSentence : Sentence) : Sentence :=
  sorry

/-- Verifies that the translated sentence follows Kurdish sentence structure -/
theorem translation_correctness (russianSentence : Sentence) :
  kurdishSentenceStructure (translateToKurdish russianSentence) :=
  sorry

/-- Specific sentences from the problem -/
def sentence1 : Sentence := sorry -- "The lazy lion eats meat"
def sentence2 : Sentence := sorry -- "The healthy poor man carries the burden"
def sentence3 : Sentence := sorry -- "The bull of the poor man does not understand the poor man"

/-- Verifies the correctness of the given translations -/
theorem given_translations_correct :
  kurdishSentenceStructure (translateToKurdish sentence1) ∧
  kurdishSentenceStructure (translateToKurdish sentence2) ∧
  kurdishSentenceStructure (translateToKurdish sentence3) :=
  sorry

end translation_correctness_given_translations_correct_l3099_309927


namespace cubic_inequality_l3099_309982

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 + 44*x - 16 > 0 ↔ 
  (x > 4 ∧ x < 4 + 2*Real.sqrt 3) ∨ x > 4 + 2*Real.sqrt 3 :=
by sorry

end cubic_inequality_l3099_309982


namespace point_M_coordinates_l3099_309964

def f (x : ℝ) : ℝ := 2 * x^2 + 1

theorem point_M_coordinates (x₀ y₀ : ℝ) :
  (∃ M : ℝ × ℝ, M.1 = x₀ ∧ M.2 = y₀ ∧ 
   (deriv f) x₀ = -8 ∧ f x₀ = y₀) →
  x₀ = -2 ∧ y₀ = 9 := by
sorry

end point_M_coordinates_l3099_309964
