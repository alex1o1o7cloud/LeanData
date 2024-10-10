import Mathlib

namespace absolute_value_equation_solutions_l3780_378086

theorem absolute_value_equation_solutions (x : ℝ) :
  |5 * x - 4| = 29 ↔ x = -5 ∨ x = 33/5 :=
by sorry

end absolute_value_equation_solutions_l3780_378086


namespace unique_base_representation_l3780_378026

theorem unique_base_representation : ∃! n : ℕ+, 
  ∃ A B : ℕ, 
    (0 ≤ A ∧ A < 7) ∧ 
    (0 ≤ B ∧ B < 7) ∧
    (0 ≤ A ∧ A < 5) ∧ 
    (0 ≤ B ∧ B < 5) ∧
    (n : ℕ) = 7 * A + B ∧
    (n : ℕ) = 5 * B + A ∧
    (n : ℕ) = 17 := by
  sorry

end unique_base_representation_l3780_378026


namespace largest_invalid_sum_l3780_378015

def is_valid_sum (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b % 6 ≠ 0 ∧ n = 42 * a + b

theorem largest_invalid_sum : 
  (∀ m : ℕ, m > 252 → is_valid_sum m) ∧ ¬ is_valid_sum 252 :=
sorry

end largest_invalid_sum_l3780_378015


namespace distribute_negative_two_over_parentheses_l3780_378088

theorem distribute_negative_two_over_parentheses (x : ℝ) : -2 * (x - 3) = -2 * x + 6 := by
  sorry

end distribute_negative_two_over_parentheses_l3780_378088


namespace initial_birds_count_l3780_378008

/-- The number of birds initially sitting in a tree -/
def initial_birds : ℕ := sorry

/-- The number of birds that flew up to join the initial birds -/
def additional_birds : ℕ := 81

/-- The total number of birds after additional birds joined -/
def total_birds : ℕ := 312

/-- Theorem stating that the number of birds initially sitting in the tree is 231 -/
theorem initial_birds_count : initial_birds = 231 := by
  sorry

end initial_birds_count_l3780_378008


namespace playground_boys_count_l3780_378052

theorem playground_boys_count (total_children girls : ℕ) 
  (h1 : total_children = 117) 
  (h2 : girls = 77) : 
  total_children - girls = 40 := by
sorry

end playground_boys_count_l3780_378052


namespace distinct_values_mod_p_l3780_378078

theorem distinct_values_mod_p (p : ℕ) (a b : Fin p) (hp : Nat.Prime p) (hab : a ≠ b) :
  let f : Fin p → ℕ := λ n => (Finset.range (p - 1)).sum (λ i => (i + 1) * n^(i + 1))
  ¬ (f a ≡ f b [MOD p]) := by
  sorry

end distinct_values_mod_p_l3780_378078


namespace abs_neg_five_equals_five_l3780_378081

theorem abs_neg_five_equals_five : |(-5 : ℤ)| = 5 := by sorry

end abs_neg_five_equals_five_l3780_378081


namespace geometric_sequence_sum_l3780_378003

/-- Given a geometric sequence {aₙ}, prove that if a₁ + a₂ + a₃ = 2 and a₃ + a₄ + a₅ = 8, 
    then a₄ + a₅ + a₆ = ±16 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∃ (q : ℝ), ∀ n, a (n + 1) = q * a n) 
  (h_sum1 : a 1 + a 2 + a 3 = 2) (h_sum2 : a 3 + a 4 + a 5 = 8) : 
  a 4 + a 5 + a 6 = 16 ∨ a 4 + a 5 + a 6 = -16 := by
  sorry


end geometric_sequence_sum_l3780_378003


namespace wizard_elixir_combinations_l3780_378029

/-- The number of valid combinations for a wizard's elixir --/
def validCombinations (herbs : ℕ) (gems : ℕ) (incompatible : ℕ) : ℕ :=
  herbs * gems - incompatible

/-- Theorem: Given 4 herbs, 6 gems, and 3 invalid combinations, 
    the number of valid combinations is 21 --/
theorem wizard_elixir_combinations : 
  validCombinations 4 6 3 = 21 := by
  sorry

end wizard_elixir_combinations_l3780_378029


namespace six_digit_number_rotation_l3780_378073

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def rotate_last_to_first (n : ℕ) : ℕ :=
  let d := n % 10
  let r := n / 10
  d * 100000 + r

theorem six_digit_number_rotation (n : ℕ) :
  is_six_digit n ∧ rotate_last_to_first n = n / 3 → n = 428571 ∨ n = 857142 := by
  sorry

end six_digit_number_rotation_l3780_378073


namespace alex_phone_bill_l3780_378054

/-- Calculates the total cost of a cell phone plan based on usage --/
def calculate_phone_bill (base_cost : ℚ) (text_cost : ℚ) (extra_minute_cost : ℚ) 
  (extra_data_cost : ℚ) (texts_sent : ℕ) (hours_used : ℚ) (data_used : ℚ) : ℚ :=
  let text_charge := text_cost * texts_sent
  let extra_minutes := max (hours_used - 35) 0 * 60
  let minute_charge := extra_minute_cost * extra_minutes
  let extra_data := max (data_used - 2) 0 * 1000
  let data_charge := extra_data_cost * extra_data
  base_cost + text_charge + minute_charge + data_charge

/-- The total cost of Alex's cell phone plan in February is $126.30 --/
theorem alex_phone_bill : 
  calculate_phone_bill 30 0.07 0.12 0.15 150 36.5 2.5 = 126.30 := by
  sorry

end alex_phone_bill_l3780_378054


namespace sqrt_three_divided_by_sum_l3780_378090

theorem sqrt_three_divided_by_sum : 
  Real.sqrt 3 / (Real.sqrt (1/3) + Real.sqrt (3/16)) = 12/7 := by
  sorry

end sqrt_three_divided_by_sum_l3780_378090


namespace regular_polygon_properties_l3780_378092

theorem regular_polygon_properties :
  ∀ n : ℕ,
  (n ≥ 3) →
  (n - 2) * 180 = 3 * 360 + 180 →
  n = 9 ∧ ((n - 2) * 180 / n : ℚ) = 140 :=
by sorry

end regular_polygon_properties_l3780_378092


namespace solve_system_for_b_l3780_378071

theorem solve_system_for_b :
  ∀ (x y b : ℝ),
  (4 * x + 2 * y = b) →
  (3 * x + 4 * y = 3 * b) →
  (x = 3) →
  (b = -15) := by
sorry

end solve_system_for_b_l3780_378071


namespace unique_ambiguous_sum_l3780_378079

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 36

def sum_triple (a b c : ℕ) : ℕ := a + b + c

theorem unique_ambiguous_sum :
  ∃ (s : ℕ), 
    (∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℕ), 
      is_valid_triple a₁ b₁ c₁ ∧ 
      is_valid_triple a₂ b₂ c₂ ∧ 
      sum_triple a₁ b₁ c₁ = s ∧ 
      sum_triple a₂ b₂ c₂ = s ∧ 
      (a₁, b₁, c₁) ≠ (a₂, b₂, c₂)) ∧
    (∀ (t : ℕ), 
      t ≠ s → 
      ∀ (x y z u v w : ℕ), 
        is_valid_triple x y z → 
        is_valid_triple u v w → 
        sum_triple x y z = t → 
        sum_triple u v w = t → 
        (x, y, z) = (u, v, w)) →
  s = 13 :=
sorry

end unique_ambiguous_sum_l3780_378079


namespace symmetric_points_sum_l3780_378080

/-- Given two points P and P' that are symmetric with respect to the origin,
    prove that 2a+b = -3 --/
theorem symmetric_points_sum (a b : ℝ) : 
  (2*a + 1 = -1 ∧ 4 = -(3*b - 1)) → 2*a + b = -3 := by
  sorry

end symmetric_points_sum_l3780_378080


namespace percentage_needed_to_pass_l3780_378018

def total_marks : ℕ := 2075
def pradeep_score : ℕ := 390
def failed_by : ℕ := 25

def passing_mark : ℕ := pradeep_score + failed_by

def percentage_to_pass : ℚ := (passing_mark : ℚ) / (total_marks : ℚ) * 100

theorem percentage_needed_to_pass :
  ∃ (ε : ℚ), abs (percentage_to_pass - 20) < ε ∧ ε > 0 :=
sorry

end percentage_needed_to_pass_l3780_378018


namespace integer_solutions_of_equation_l3780_378011

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x^2 = 1 + 4*y^3*(y + 2) ↔ 
    (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = -2) :=
by sorry

end integer_solutions_of_equation_l3780_378011


namespace inequality_relation_l3780_378061

theorem inequality_relation (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end inequality_relation_l3780_378061


namespace twelve_not_feasible_fourteen_feasible_l3780_378022

/-- Represents the conditions for forming a convex equiangular hexagon from equilateral triangular tiles. -/
def IsValidHexagonConfiguration (n ℓ a b c : ℕ) : Prop :=
  n = ℓ^2 - a^2 - b^2 - c^2 ∧ 
  ℓ > a + b ∧ 
  ℓ > a + c ∧ 
  ℓ > b + c

/-- States that 12 is not a feasible number of tiles for forming a convex equiangular hexagon. -/
theorem twelve_not_feasible : ¬ ∃ (ℓ a b c : ℕ), IsValidHexagonConfiguration 12 ℓ a b c :=
sorry

/-- States that 14 is a feasible number of tiles for forming a convex equiangular hexagon. -/
theorem fourteen_feasible : ∃ (ℓ a b c : ℕ), IsValidHexagonConfiguration 14 ℓ a b c :=
sorry

end twelve_not_feasible_fourteen_feasible_l3780_378022


namespace function_minimum_implies_inequality_l3780_378006

open Real

/-- Given a function f(x) = ax^2 + bx - 2ln(x) where a > 0 and b is real,
    if f(x) ≥ f(2) for all x > 0, then ln(a) < -b - 1 -/
theorem function_minimum_implies_inequality (a b : ℝ) (ha : a > 0) :
  (∀ x > 0, a * x^2 + b * x - 2 * log x ≥ a * 2^2 + b * 2 - 2 * log 2) →
  log a < -b - 1 :=
by sorry

end function_minimum_implies_inequality_l3780_378006


namespace fraction_simplest_form_l3780_378036

theorem fraction_simplest_form (a b c : ℝ) :
  (a^2 - b^2 + c^2 + 2*b*c) / (a^2 - c^2 + b^2 + 2*a*b) =
  (a^2 - b^2 + c^2 + 2*b*c) / (a^2 - c^2 + b^2 + 2*a*b) := by sorry

end fraction_simplest_form_l3780_378036


namespace fraction_split_l3780_378050

theorem fraction_split (n d a b : ℕ) (h1 : d = a * b) (h2 : Nat.gcd a b = 1) (h3 : n = 58) (h4 : d = 77) (h5 : a = 11) (h6 : b = 7) :
  ∃ (x y : ℤ), (n : ℚ) / d = (x : ℚ) / b + (y : ℚ) / a :=
sorry

end fraction_split_l3780_378050


namespace x_24_equals_one_l3780_378001

theorem x_24_equals_one (x : ℂ) (h : x + 1/x = -Real.sqrt 3) : x^24 = 1 := by
  sorry

end x_24_equals_one_l3780_378001


namespace wire_cutting_l3780_378025

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 140 ∧ 
  ratio = 2 / 5 ∧ 
  shorter_piece + (1 + ratio) * shorter_piece = total_length →
  shorter_piece = 40 := by
  sorry

end wire_cutting_l3780_378025


namespace weight_problem_l3780_378046

/-- Given three weights a, b, and c, prove that their average weights satisfy certain conditions -/
theorem weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 43 →
  b = 31 →
  (a + b) / 2 = 40 :=
by
  sorry


end weight_problem_l3780_378046


namespace least_exponent_sum_for_500_l3780_378031

/-- Given a natural number n, returns the set of exponents of 2 in its binary representation -/
def binaryExponents (n : ℕ) : Finset ℕ :=
  sorry

/-- The sum of exponents of 2 in the binary representation of n -/
def sumOfExponents (n : ℕ) : ℕ :=
  (binaryExponents n).sum id

/-- Checks if a set of exponents represents a valid sum of powers of 2 for a given number -/
def isValidRepresentation (n : ℕ) (exponents : Finset ℕ) : Prop :=
  (exponents.sum (fun i => 2^i) = n) ∧ (exponents.card ≥ 3)

theorem least_exponent_sum_for_500 :
  ∀ (exponents : Finset ℕ),
    isValidRepresentation 500 exponents →
    sumOfExponents 500 ≤ (exponents.sum id) :=
by sorry

end least_exponent_sum_for_500_l3780_378031


namespace stream_speed_l3780_378023

theorem stream_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 11)
  (h2 : upstream_speed = 8) :
  let stream_speed := (downstream_speed - upstream_speed) / 2
  stream_speed = 1.5 := by sorry

end stream_speed_l3780_378023


namespace sqrt_sum_power_inequality_l3780_378016

theorem sqrt_sum_power_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (Real.sqrt x + Real.sqrt y)^8 ≥ 64 * x * y * (x + y)^2 := by
  sorry

end sqrt_sum_power_inequality_l3780_378016


namespace tampa_bay_bucs_players_l3780_378030

/-- The initial number of football players in the Tampa Bay Bucs team. -/
def initial_football_players : ℕ := 13

/-- The initial number of cheerleaders in the Tampa Bay Bucs team. -/
def initial_cheerleaders : ℕ := 16

/-- The number of football players who quit. -/
def quitting_football_players : ℕ := 10

/-- The number of cheerleaders who quit. -/
def quitting_cheerleaders : ℕ := 4

/-- The total number of people left after some quit. -/
def remaining_total : ℕ := 15

theorem tampa_bay_bucs_players :
  initial_football_players = 13 ∧
  (initial_football_players - quitting_football_players) +
  (initial_cheerleaders - quitting_cheerleaders) = remaining_total :=
sorry

end tampa_bay_bucs_players_l3780_378030


namespace amara_clothing_donation_l3780_378047

theorem amara_clothing_donation :
  ∀ (initial remaining thrown_away : ℕ) (first_donation : ℕ),
    initial = 100 →
    remaining = 65 →
    thrown_away = 15 →
    initial - remaining = first_donation + 3 * first_donation + thrown_away →
    first_donation = 5 := by
  sorry

end amara_clothing_donation_l3780_378047


namespace brick_length_calculation_brick_length_is_20cm_l3780_378076

/-- Given a courtyard and brick specifications, calculate the length of each brick. -/
theorem brick_length_calculation (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_width : ℝ) (total_bricks : ℕ) : ℝ :=
  let courtyard_area := courtyard_length * courtyard_width * 10000 -- Convert to cm²
  let brick_area := courtyard_area / total_bricks
  brick_area / brick_width

/-- Prove that for the given specifications, the brick length is 20 cm. -/
theorem brick_length_is_20cm : 
  brick_length_calculation 30 16 10 24000 = 20 := by
  sorry

end brick_length_calculation_brick_length_is_20cm_l3780_378076


namespace equation_standard_form_and_coefficients_l3780_378075

theorem equation_standard_form_and_coefficients :
  ∀ x : ℝ, x * (x + 1) = 2 * x - 1 ↔ x^2 - x + 1 = 0 ∧
  1 = 1 ∧ -1 = -1 ∧ 1 = 1 := by sorry

end equation_standard_form_and_coefficients_l3780_378075


namespace max_value_expression_l3780_378013

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a * b + b * c + 2 * c * a ≤ 9 / 2 := by
sorry

end max_value_expression_l3780_378013


namespace complex_magnitude_l3780_378096

theorem complex_magnitude (z : ℂ) : z = 5 / (2 + Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l3780_378096


namespace car_travel_distance_l3780_378091

/-- Represents the distance traveled by a car given its speed and time -/
def distance_traveled (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

theorem car_travel_distance :
  let initial_distance : ℚ := 3
  let initial_time : ℚ := 4
  let total_time : ℚ := 120
  distance_traveled (initial_distance / initial_time) total_time = 90 := by
  sorry

end car_travel_distance_l3780_378091


namespace min_value_on_line_l3780_378041

theorem min_value_on_line (x y : ℝ) (h : x + 2 * y = 3) :
  ∃ (m : ℝ), m = 4 * Real.sqrt 2 ∧ ∀ (a b : ℝ), a + 2 * b = 3 → 2^a + 4^b ≥ m := by
  sorry

end min_value_on_line_l3780_378041


namespace simple_interest_rate_proof_l3780_378004

/-- The rate at which a sum becomes 4 times of itself in 15 years at simple interest -/
def simple_interest_rate : ℝ := 20

/-- The time period in years -/
def time_period : ℝ := 15

/-- The factor by which the sum increases -/
def growth_factor : ℝ := 4

theorem simple_interest_rate_proof : 
  (1 + simple_interest_rate * time_period / 100) = growth_factor := by
  sorry

#check simple_interest_rate_proof

end simple_interest_rate_proof_l3780_378004


namespace max_handshakers_l3780_378033

/-- Given a room with N people, where N > 4, and at least two people have not shaken
    hands with everyone else, the maximum number of people who could have shaken
    hands with everyone else is N-2. -/
theorem max_handshakers (N : ℕ) (h1 : N > 4) (h2 : ∃ (a b : ℕ), a ≠ b ∧ a < N ∧ b < N ∧ 
  (∃ (c : ℕ), c < N ∧ c ≠ a ∧ c ≠ b)) : 
  ∃ (M : ℕ), M = N - 2 ∧ 
  (∀ (k : ℕ), k ≤ N → (∃ (S : Finset ℕ), S.card = k ∧ 
    (∀ (i j : ℕ), i ∈ S → j ∈ S → i ≠ j → (∃ (H : Prop), H)) → k ≤ M)) :=
sorry

end max_handshakers_l3780_378033


namespace salary_increase_l3780_378035

/-- Prove that adding a manager's salary increases the average salary by 100 --/
theorem salary_increase (num_employees : ℕ) (avg_salary : ℚ) (manager_salary : ℚ) :
  num_employees = 20 →
  avg_salary = 1700 →
  manager_salary = 3800 →
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 100 := by
  sorry

end salary_increase_l3780_378035


namespace root_product_equation_l3780_378087

theorem root_product_equation (x₁ x₂ x₃ : ℝ) : 
  (Real.sqrt 2023 * x₁^3 - 4047 * x₁^2 + 3 = 0) →
  (Real.sqrt 2023 * x₂^3 - 4047 * x₂^2 + 3 = 0) →
  (Real.sqrt 2023 * x₃^3 - 4047 * x₃^2 + 3 = 0) →
  x₁ < x₂ → x₂ < x₃ →
  x₂ * (x₁ + x₃) = 4046 := by
sorry

end root_product_equation_l3780_378087


namespace cubic_inequality_for_negative_numbers_l3780_378066

theorem cubic_inequality_for_negative_numbers (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(a^3 > b^3) := by
  sorry

end cubic_inequality_for_negative_numbers_l3780_378066


namespace ellipse_hyperbola_circles_lines_theorem_l3780_378019

-- Ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 9 - x^2 / 16 = 1

-- Circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*y - 1 = 0

-- Lines
def line1 (a x y : ℝ) : Prop := a^2 * x - y + 6 = 0
def line2 (a x y : ℝ) : Prop := 4 * x - (a - 3) * y + 9 = 0

theorem ellipse_hyperbola_circles_lines_theorem :
  (∃ (F₁ F₂ P : ℝ × ℝ), ellipse P.1 P.2 ∧ |P.1 - F₁.1| + |P.2 - F₁.2| = 3 ∧ |P.1 - F₂.1| + |P.2 - F₂.2| ≠ 1) ∧
  (∀ (x y : ℝ), hyperbola x y → (|y| - |3/4 * x| = 12/5)) ∧
  (∃ (t₁ t₂ : ℝ × ℝ), t₁ ≠ t₂ ∧ (∀ (x y : ℝ), circle1 x y → (x - t₁.1)^2 + (y - t₁.2)^2 = 0) ∧
                                 (∀ (x y : ℝ), circle2 x y → (x - t₁.1)^2 + (y - t₁.2)^2 = 0) ∧
                                 (∀ (x y : ℝ), circle1 x y → (x - t₂.1)^2 + (y - t₂.2)^2 = 0) ∧
                                 (∀ (x y : ℝ), circle2 x y → (x - t₂.1)^2 + (y - t₂.2)^2 = 0)) ∧
  (∃ (a : ℝ), a ≠ -1 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ), line1 a x₁ y₁ ∧ line2 a x₂ y₂ → (x₁ - x₂) * (y₁ - y₂) ≠ -1) :=
sorry

end ellipse_hyperbola_circles_lines_theorem_l3780_378019


namespace inequality_solutions_l3780_378077

theorem inequality_solutions :
  (∀ x : ℝ, 5 * x + 3 < 11 + x ↔ x < 2) ∧
  (∀ x : ℝ, 2 * x + 1 < 3 * x + 3 ∧ (x + 1) / 2 ≤ (1 - x) / 6 + 1 ↔ -2 < x ∧ x ≤ 1) := by
  sorry

end inequality_solutions_l3780_378077


namespace bruce_purchase_cost_l3780_378094

/-- The total cost of Bruce's purchase of grapes and mangoes -/
def total_cost (grape_quantity : ℕ) (grape_price : ℕ) (mango_quantity : ℕ) (mango_price : ℕ) : ℕ :=
  grape_quantity * grape_price + mango_quantity * mango_price

/-- Theorem stating the total cost of Bruce's purchase -/
theorem bruce_purchase_cost :
  total_cost 8 70 11 55 = 1165 := by
  sorry

#eval total_cost 8 70 11 55

end bruce_purchase_cost_l3780_378094


namespace extreme_value_condition_l3780_378065

/-- The function f(x) defined as x^3 + ax^2 + 3x - 9 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

/-- The derivative of f(x) with respect to x -/
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem extreme_value_condition (a : ℝ) : 
  (∀ x : ℝ, f a x = f_prime a x) → 
  f_prime a (-3) = 0 → 
  a = 5 := by sorry

end extreme_value_condition_l3780_378065


namespace equation_solution_l3780_378028

theorem equation_solution : ∃! x : ℚ, (3 / 5) * (1 / 9) * x = 6 := by
  sorry

end equation_solution_l3780_378028


namespace cost_price_of_article_l3780_378002

/-- 
Given an article where the profit obtained by selling it for Rs. 66 
is equal to the loss obtained by selling it for Rs. 52, 
prove that the cost price of the article is Rs. 59.
-/
theorem cost_price_of_article (cost_price : ℤ) : cost_price = 59 :=
  sorry

end cost_price_of_article_l3780_378002


namespace cone_base_radius_l3780_378099

/-- Given a sector with radius 15 cm and central angle 120 degrees used to form a cone without seam loss, 
    the radius of the base of the cone is 5 cm. -/
theorem cone_base_radius (sector_radius : ℝ) (central_angle : ℝ) (base_radius : ℝ) : 
  sector_radius = 15 → 
  central_angle = 120 → 
  base_radius = (central_angle / 360) * sector_radius → 
  base_radius = 5 := by
sorry

end cone_base_radius_l3780_378099


namespace range_of_n_l3780_378012

def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (n m : ℝ) : Set ℝ := {x | n - m < x ∧ x < n + m}

theorem range_of_n (h : ∀ n : ℝ, (∃ x, x ∈ A ∩ B n 1) → ∃ x, x ∈ A ∩ B n 1) :
  ∀ n : ℝ, n ∈ Set.Ioo (-2 : ℝ) 2 :=
sorry

end range_of_n_l3780_378012


namespace only_5008300_has_no_zeros_l3780_378009

/-- Represents a natural number and how it's pronounced in English --/
structure NumberPronunciation where
  value : Nat
  pronunciation : String

/-- Counts the number of times "zero" appears in a string --/
def countZeros (s : String) : Nat :=
  s.split (· = ' ') |>.filter (· = "zero") |>.length

/-- The main theorem stating that only 5008300 has no zeros when pronounced --/
theorem only_5008300_has_no_zeros (numbers : List NumberPronunciation) 
    (h1 : NumberPronunciation.mk 5008300 "five million eight thousand three hundred" ∈ numbers)
    (h2 : NumberPronunciation.mk 500800 "five hundred thousand eight hundred" ∈ numbers)
    (h3 : NumberPronunciation.mk 5080000 "five million eighty thousand" ∈ numbers) :
    ∃! n : NumberPronunciation, n ∈ numbers ∧ countZeros n.pronunciation = 0 :=
  sorry

end only_5008300_has_no_zeros_l3780_378009


namespace irreducible_fraction_l3780_378098

theorem irreducible_fraction : (201920192019 : ℚ) / 191719171917 = 673 / 639 := by sorry

end irreducible_fraction_l3780_378098


namespace coin_sum_problem_l3780_378083

/-- Calculates the total sum of money in rupees given the number of 20 paise and 25 paise coins. -/
def total_sum_in_rupees (coins_20_paise : ℕ) (coins_25_paise : ℕ) : ℚ :=
  (coins_20_paise * 20 + coins_25_paise * 25) / 100

/-- Proves that given 336 total coins, with 260 coins of 20 paise and the rest being 25 paise coins, 
    the total sum of money is 71 rupees. -/
theorem coin_sum_problem (total_coins : ℕ) (coins_20_paise : ℕ) 
  (h1 : total_coins = 336)
  (h2 : coins_20_paise = 260)
  (h3 : total_coins = coins_20_paise + (total_coins - coins_20_paise)) :
  total_sum_in_rupees coins_20_paise (total_coins - coins_20_paise) = 71 := by
  sorry

#eval total_sum_in_rupees 260 76

end coin_sum_problem_l3780_378083


namespace vector_problems_l3780_378069

def a : ℝ × ℝ := (-3, 2)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (3, -1)

theorem vector_problems :
  (∃ t : ℝ, (∀ s : ℝ, ‖a + s • b‖ ≥ ‖a + t • b‖) ∧ ‖a + t • b‖ = 7 / Real.sqrt 5 ∧ t = 4/5) ∧
  (∃ t : ℝ, ∃ k : ℝ, a - t • b = k • c ∧ t = 3/5) :=
by sorry

end vector_problems_l3780_378069


namespace satisfaction_ratings_properties_l3780_378058

def satisfaction_ratings : List ℝ := [5, 7, 8, 9, 7, 5, 10, 8, 4, 7]

def mode (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem satisfaction_ratings_properties :
  mode satisfaction_ratings = 7 ∧
  range satisfaction_ratings = 6 ∧
  variance satisfaction_ratings = 3.2 :=
by sorry

end satisfaction_ratings_properties_l3780_378058


namespace trig_identity_proof_l3780_378053

theorem trig_identity_proof : 
  (Real.sqrt 3 / Real.sin (20 * π / 180)) - (1 / Real.sin (70 * π / 180)) = 4 := by
sorry

end trig_identity_proof_l3780_378053


namespace prime_cube_minus_one_not_divisible_by_40_l3780_378043

theorem prime_cube_minus_one_not_divisible_by_40 (p : ℕ) (hp : Prime p) (hp_ge_7 : p ≥ 7) :
  ¬(40 ∣ p^3 - 1) :=
sorry

end prime_cube_minus_one_not_divisible_by_40_l3780_378043


namespace billy_bumper_car_rides_l3780_378095

/-- Calculates the number of bumper car rides given the number of ferris wheel rides,
    the cost per ride, and the total number of tickets used. -/
def bumper_car_rides (ferris_wheel_rides : ℕ) (cost_per_ride : ℕ) (total_tickets : ℕ) : ℕ :=
  (total_tickets - ferris_wheel_rides * cost_per_ride) / cost_per_ride

theorem billy_bumper_car_rides :
  bumper_car_rides 7 5 50 = 3 := by
  sorry

end billy_bumper_car_rides_l3780_378095


namespace highest_power_of_two_dividing_13_4_minus_11_4_l3780_378005

theorem highest_power_of_two_dividing_13_4_minus_11_4 :
  ∃ (n : ℕ), 2^n = (Nat.gcd (13^4 - 11^4) (2^32 : ℕ)) ∧
  ∀ (m : ℕ), 2^m ∣ (13^4 - 11^4) → m ≤ n :=
by sorry

end highest_power_of_two_dividing_13_4_minus_11_4_l3780_378005


namespace orange_profit_l3780_378063

theorem orange_profit : 
  let buy_rate : ℚ := 10 / 11  -- Cost in r per orange when buying
  let sell_rate : ℚ := 11 / 10  -- Revenue in r per orange when selling
  let num_oranges : ℕ := 110
  let cost : ℚ := buy_rate * num_oranges
  let revenue : ℚ := sell_rate * num_oranges
  let profit : ℚ := revenue - cost
  profit = 21 := by sorry

end orange_profit_l3780_378063


namespace first_group_size_l3780_378051

/-- The number of men in the first group -/
def M : ℕ := sorry

/-- The number of acres that can be reaped by M men in 15 days -/
def acres_first_group : ℕ := 120

/-- The number of days it takes M men to reap 120 acres -/
def days_first_group : ℕ := 15

/-- The number of men in the second group -/
def men_second_group : ℕ := 20

/-- The number of acres that can be reaped by 20 men in 30 days -/
def acres_second_group : ℕ := 480

/-- The number of days it takes 20 men to reap 480 acres -/
def days_second_group : ℕ := 30

theorem first_group_size :
  M = 10 :=
sorry

end first_group_size_l3780_378051


namespace area_triangle_ABG_l3780_378059

/-- Given a rectangle ABCD and a square AEFG, where AB = 6, AD = 4, and the area of triangle ADE is 2,
    prove that the area of triangle ABG is 3. -/
theorem area_triangle_ABG (A B C D E F G : ℝ × ℝ) : 
  (∀ X Y, X ≠ Y → (X = A ∧ Y = B) ∨ (X = B ∧ Y = C) ∨ (X = C ∧ Y = D) ∨ (X = D ∧ Y = A) → 
    (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (Y.1 - X.1)^2 + (Y.2 - X.2)^2) →  -- ABCD is a rectangle
  (∀ X Y, X ≠ Y → (X = A ∧ Y = E) ∨ (X = E ∧ Y = F) ∨ (X = F ∧ Y = G) ∨ (X = G ∧ Y = A) → 
    (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (E.1 - A.1)^2 + (E.2 - A.2)^2) →  -- AEFG is a square
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36 →  -- AB = 6
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = 16 →  -- AD = 4
  abs ((E.1 - A.1) * (D.2 - A.2) - (E.2 - A.2) * (D.1 - A.1)) / 2 = 2 →  -- Area of triangle ADE = 2
  abs ((G.1 - A.1) * (B.2 - A.2) - (G.2 - A.2) * (B.1 - A.1)) / 2 = 3  -- Area of triangle ABG = 3
  := by sorry

end area_triangle_ABG_l3780_378059


namespace toy_production_difference_l3780_378089

/-- The difference in daily toy production between actual and planned rates --/
theorem toy_production_difference (total_toys : ℕ) (planned_days : ℕ) (actual_days : ℕ) 
  (h1 : total_toys = 10080)
  (h2 : planned_days = 14)
  (h3 : actual_days = planned_days - 2) :
  (total_toys / actual_days) - (total_toys / planned_days) = 120 := by
  sorry

end toy_production_difference_l3780_378089


namespace sams_remaining_dimes_l3780_378084

theorem sams_remaining_dimes 
  (initial_dimes : ℕ) 
  (borrowed_dimes : ℕ) 
  (h1 : initial_dimes = 8) 
  (h2 : borrowed_dimes = 4) :
  initial_dimes - borrowed_dimes = 4 :=
by sorry

end sams_remaining_dimes_l3780_378084


namespace natural_numbers_less_than_10_l3780_378039

theorem natural_numbers_less_than_10 : 
  {n : ℕ | n < 10} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

end natural_numbers_less_than_10_l3780_378039


namespace trajectory_and_fixed_point_l3780_378085

-- Define the plane
variable (P : ℝ × ℝ)

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the foot of the perpendicular Q
def Q (P : ℝ × ℝ) : ℝ × ℝ := (-1, P.2)

-- Define the dot product of 2D vectors
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem trajectory_and_fixed_point (P : ℝ × ℝ) : 
  (dot (P.1 + 1, P.2) (2, -P.2) = dot (P.1 - 1, P.2) (-2, P.2)) →
  (∃ (C : Set (ℝ × ℝ)) (E : ℝ × ℝ), 
    C = {p : ℝ × ℝ | p.2^2 = 4 * p.1} ∧
    E = (1, 0) ∧
    (∀ (k m : ℝ), 
      let M := (m^2, 2*m)
      let N := (-1, -1/m + m)
      (M ∈ C ∧ N.1 = -1) →
      (∃ (r : ℝ), (M.1 - E.1)^2 + (M.2 - E.2)^2 = r^2 ∧
                  (N.1 - E.1)^2 + (N.2 - E.2)^2 = r^2))) :=
sorry

end trajectory_and_fixed_point_l3780_378085


namespace tiles_needed_is_108_l3780_378048

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- The dimensions of a tile in inches -/
def tile : Dimensions := ⟨4, 6⟩

/-- The dimensions of the floor in feet -/
def floor : Dimensions := ⟨3, 6⟩

/-- The number of tiles needed to cover the floor -/
def tiles_needed : ℕ :=
  (area ⟨feet_to_inches floor.length, feet_to_inches floor.width⟩) / (area tile)

theorem tiles_needed_is_108 : tiles_needed = 108 := by
  sorry

#eval tiles_needed

end tiles_needed_is_108_l3780_378048


namespace complex_equation_sum_l3780_378021

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a + i) / i = 1 + b * i → a + b = 0 := by sorry

end complex_equation_sum_l3780_378021


namespace complex_calculation_l3780_378055

theorem complex_calculation : ((7 - 3 * Complex.I) - 3 * (2 - 5 * Complex.I)) * (1 + 2 * Complex.I) = -23 + 14 * Complex.I := by
  sorry

end complex_calculation_l3780_378055


namespace area_of_triangle_pqs_l3780_378024

/-- Represents a trapezoid PQRS -/
structure Trapezoid where
  pq : ℝ
  rs : ℝ
  area : ℝ

/-- Theorem: Given a trapezoid PQRS with an area of 20, where RS is three times the length of PQ,
    the area of triangle PQS is 5. -/
theorem area_of_triangle_pqs (t : Trapezoid) 
    (h1 : t.area = 20)
    (h2 : t.rs = 3 * t.pq) : 
    t.area / 4 = 5 := by
  sorry

end area_of_triangle_pqs_l3780_378024


namespace nicoles_age_l3780_378074

theorem nicoles_age (nicole_age sally_age : ℕ) : 
  nicole_age = 3 * sally_age →
  nicole_age + sally_age + 8 = 40 →
  nicole_age = 24 := by
sorry

end nicoles_age_l3780_378074


namespace train_journey_time_l3780_378067

/-- Proves that if a train moving at 6/7 of its usual speed arrives 10 minutes late, then its usual journey time is 7 hours -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0) 
  (h3 : (6 / 7 * usual_speed) * (usual_time + 10 / 60) = usual_speed * usual_time) : 
  usual_time = 7 := by
  sorry

#check train_journey_time

end train_journey_time_l3780_378067


namespace pool_depth_calculation_l3780_378020

/-- Calculates the depth of a rectangular pool given its dimensions and draining specifications. -/
theorem pool_depth_calculation (width : ℝ) (length : ℝ) (drain_rate : ℝ) (drain_time : ℝ) (capacity_percentage : ℝ) :
  width = 50 →
  length = 150 →
  drain_rate = 60 →
  drain_time = 1000 →
  capacity_percentage = 0.8 →
  (width * length * (drain_rate * drain_time / capacity_percentage)) / (width * length) = 10 :=
by
  sorry

#check pool_depth_calculation

end pool_depth_calculation_l3780_378020


namespace intersection_complement_equals_two_l3780_378056

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {1, 3}

theorem intersection_complement_equals_two :
  A ∩ (U \ B) = {2} := by sorry

end intersection_complement_equals_two_l3780_378056


namespace donovan_candles_count_l3780_378064

/-- The number of candles Donovan brought in -/
def donovans_candles : ℕ := 20

/-- The number of candles in Kalani's bedroom -/
def bedroom_candles : ℕ := 20

/-- The number of candles in the living room -/
def living_room_candles : ℕ := bedroom_candles / 2

/-- The total number of candles in the house -/
def total_candles : ℕ := 50

theorem donovan_candles_count :
  donovans_candles = total_candles - bedroom_candles - living_room_candles :=
by sorry

end donovan_candles_count_l3780_378064


namespace evaluate_expression_l3780_378040

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/3) (hz : z = -3) :
  x^2 * y^3 * z^2 = 1/48 := by
  sorry

end evaluate_expression_l3780_378040


namespace green_marble_probability_l3780_378014

/-- Represents a bag of marbles with specific colors and quantities -/
structure MarbleBag where
  color1 : String
  count1 : ℕ
  color2 : String
  count2 : ℕ

/-- Calculates the probability of drawing a specific color from a bag -/
def drawProbability (bag : MarbleBag) (color : String) : ℚ :=
  if color == bag.color1 then
    bag.count1 / (bag.count1 + bag.count2)
  else if color == bag.color2 then
    bag.count2 / (bag.count1 + bag.count2)
  else
    0

/-- The main theorem stating the probability of drawing a green marble -/
theorem green_marble_probability
  (bagX : MarbleBag)
  (bagY : MarbleBag)
  (bagZ : MarbleBag)
  (hX : bagX = ⟨"white", 5, "black", 5⟩)
  (hY : bagY = ⟨"green", 4, "red", 6⟩)
  (hZ : bagZ = ⟨"green", 3, "purple", 7⟩) :
  let probWhiteX := drawProbability bagX "white"
  let probGreenY := drawProbability bagY "green"
  let probBlackX := drawProbability bagX "black"
  let probGreenZ := drawProbability bagZ "green"
  probWhiteX * probGreenY + probBlackX * probGreenZ = 7 / 20 := by
  sorry


end green_marble_probability_l3780_378014


namespace tv_show_watch_time_l3780_378032

theorem tv_show_watch_time : 
  let regular_seasons : ℕ := 9
  let episodes_per_regular_season : ℕ := 22
  let episodes_in_last_season : ℕ := 26
  let episode_duration : ℚ := 1/2
  
  let total_episodes : ℕ := 
    regular_seasons * episodes_per_regular_season + episodes_in_last_season
  
  let total_watch_time : ℚ := total_episodes * episode_duration
  
  total_watch_time = 112 := by
  sorry

end tv_show_watch_time_l3780_378032


namespace distinct_power_representations_l3780_378049

theorem distinct_power_representations : ∃ (N : ℕ) 
  (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℕ), 
  (∃ (x y : ℕ), a₁ = x^2 ∧ a₂ = y^2) ∧
  (∃ (x y : ℕ), b₁ = x^3 ∧ b₂ = y^3) ∧
  (∃ (x y : ℕ), c₁ = x^5 ∧ c₂ = y^5) ∧
  (∃ (x y : ℕ), d₁ = x^7 ∧ d₂ = y^7) ∧
  N = a₁ - a₂ ∧
  N = b₁ - b₂ ∧
  N = c₁ - c₂ ∧
  N = d₁ - d₂ ∧
  a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ c₁ ≠ d₁ :=
by
  sorry

end distinct_power_representations_l3780_378049


namespace investment_time_is_two_years_l3780_378097

/-- Calculates the time period of investment given the principal, interest rates, and interest difference. -/
def calculate_investment_time (principal : ℚ) (rate_high : ℚ) (rate_low : ℚ) (interest_diff : ℚ) : ℚ :=
  interest_diff / (principal * (rate_high - rate_low))

theorem investment_time_is_two_years 
  (principal : ℚ) 
  (rate_high : ℚ) 
  (rate_low : ℚ) 
  (interest_diff : ℚ) :
  principal = 2500 ∧ 
  rate_high = 18 / 100 ∧ 
  rate_low = 12 / 100 ∧ 
  interest_diff = 300 → 
  calculate_investment_time principal rate_high rate_low interest_diff = 2 :=
by
  sorry

#eval calculate_investment_time 2500 (18/100) (12/100) 300

end investment_time_is_two_years_l3780_378097


namespace geometric_sequence_term_count_l3780_378034

/-- Given a geometric sequence {a_n} with a_1 = 1, q = 1/2, and a_n = 1/64, prove that the number of terms n is 7. -/
theorem geometric_sequence_term_count (a : ℕ → ℚ) :
  a 1 = 1 →
  (∀ k : ℕ, a (k + 1) = a k * (1/2)) →
  (∃ n : ℕ, a n = 1/64) →
  ∃ n : ℕ, n = 7 ∧ a n = 1/64 := by
  sorry

end geometric_sequence_term_count_l3780_378034


namespace brians_pencils_l3780_378068

/-- Given Brian's initial pencil count, the number he gives away, and the number he buys,
    prove that his final pencil count is equal to the initial count minus the number given away
    plus the number bought. -/
theorem brians_pencils (initial : ℕ) (given_away : ℕ) (bought : ℕ) :
  initial - given_away + bought = initial - given_away + bought :=
by sorry

end brians_pencils_l3780_378068


namespace percentage_difference_l3780_378010

theorem percentage_difference (x y z : ℝ) (hx : x = 5 * y) (hz : z = 1.2 * y) :
  (z - y) / x = 0.04 := by
  sorry

end percentage_difference_l3780_378010


namespace panthers_games_count_l3780_378082

theorem panthers_games_count : ∀ (initial_games : ℕ) (initial_wins : ℕ),
  initial_wins = (60 * initial_games) / 100 →
  (initial_wins + 4) * 2 = initial_games + 8 →
  initial_games + 8 = 48 :=
by
  sorry

end panthers_games_count_l3780_378082


namespace solution_set_f_gt_g_min_m_for_inequality_l3780_378038

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| - |x + 1|
def g (x : ℝ) : ℝ := -x

-- Theorem for the solution of f(x) > g(x)
theorem solution_set_f_gt_g :
  {x : ℝ | f x > g x} = {x : ℝ | -3 < x ∧ x < 1 ∨ x > 3} := by sorry

-- Theorem for the minimum value of m
theorem min_m_for_inequality (x : ℝ) :
  ∃ m : ℝ, (∀ x : ℝ, f x - 2*x ≤ 2*(g x) + m) ∧
  (∀ m' : ℝ, (∀ x : ℝ, f x - 2*x ≤ 2*(g x) + m') → m ≤ m') ∧
  m = 3 := by sorry

end solution_set_f_gt_g_min_m_for_inequality_l3780_378038


namespace work_time_ratio_l3780_378072

/-- Given two workers A and B who can complete a job together in 6 days,
    and B can complete the job alone in 36 days,
    prove that the ratio of the time A takes to complete the job alone
    to the time B takes is 1:5. -/
theorem work_time_ratio
  (time_together : ℝ)
  (time_B : ℝ)
  (h_together : time_together = 6)
  (h_B : time_B = 36)
  (time_A : ℝ)
  (h_combined_rate : 1 / time_A + 1 / time_B = 1 / time_together) :
  time_A / time_B = 1 / 5 := by
  sorry

end work_time_ratio_l3780_378072


namespace number_and_remainder_l3780_378000

theorem number_and_remainder : ∃ x : ℤ, 2 * x - 3 = 7 ∧ x % 2 = 1 := by
  sorry

end number_and_remainder_l3780_378000


namespace total_crayons_l3780_378093

/-- Given that each child has 6 crayons and there are 12 children, prove that the total number of crayons is 72. -/
theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) (h1 : crayons_per_child = 6) (h2 : num_children = 12) :
  crayons_per_child * num_children = 72 := by
  sorry

#check total_crayons

end total_crayons_l3780_378093


namespace tenth_root_unity_sum_l3780_378060

theorem tenth_root_unity_sum (z : ℂ) : 
  z = Complex.exp (3 * Real.pi * Complex.I / 5) →
  z / (1 + z^2) + z^3 / (1 + z^6) + z^5 / (1 + z^10) = (z + z^3 - 1/2) / 3 := by
  sorry

end tenth_root_unity_sum_l3780_378060


namespace divisibility_product_l3780_378007

theorem divisibility_product (a b c d : ℤ) : a ∣ b → c ∣ d → (a * c) ∣ (b * d) := by
  sorry

end divisibility_product_l3780_378007


namespace expression_evaluation_l3780_378045

theorem expression_evaluation : 4 * (8 - 2)^2 - 6 = 138 := by
  sorry

end expression_evaluation_l3780_378045


namespace min_value_and_inequality_l3780_378070

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 3| + |x - 1|

-- Theorem statement
theorem min_value_and_inequality :
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 4 → 1/a + 4/b ≥ 9/4) := by
  sorry

end min_value_and_inequality_l3780_378070


namespace possible_polynomials_g_l3780_378044

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the property that g must satisfy
def satisfies_condition (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 + 12 * x + 4

-- Theorem statement
theorem possible_polynomials_g :
  ∀ g : ℝ → ℝ, satisfies_condition g ↔ (∀ x, g x = 3 * x + 2 ∨ g x = -3 * x - 2) :=
by sorry

end possible_polynomials_g_l3780_378044


namespace continuous_stripe_probability_l3780_378062

-- Define the cube structure
structure Cube where
  faces : Fin 6 → Fin 4

-- Define the probability of a continuous stripe
def probability_continuous_stripe (c : Cube) : ℚ :=
  3 * (1 / 4) ^ 12

-- Theorem statement
theorem continuous_stripe_probability :
  ∀ c : Cube, probability_continuous_stripe c = 3 / 16777216 := by
  sorry

end continuous_stripe_probability_l3780_378062


namespace line_increase_l3780_378042

/-- Given an initial number of lines and an increased number of lines with a specific percentage increase, 
    prove that the increase in the number of lines is 110. -/
theorem line_increase (L : ℝ) : 
  let L' : ℝ := 240
  let percent_increase : ℝ := 84.61538461538461
  (L' - L) / L * 100 = percent_increase →
  L' - L = 110 := by
sorry

end line_increase_l3780_378042


namespace books_sold_l3780_378057

theorem books_sold (initial_books : Real) (bought_books : Real) (current_books : Real)
  (h1 : initial_books = 4.5)
  (h2 : bought_books = 175.3)
  (h3 : current_books = 62.8) :
  initial_books + bought_books - current_books = 117 := by
  sorry

end books_sold_l3780_378057


namespace sugar_amount_l3780_378027

/-- The number of cups of flour Mary still needs to add -/
def flour_needed : ℕ := 21

/-- The difference between the total cups of flour and sugar in the recipe -/
def flour_sugar_difference : ℕ := 8

/-- The number of cups of sugar the recipe calls for -/
def sugar_in_recipe : ℕ := flour_needed - flour_sugar_difference

theorem sugar_amount : sugar_in_recipe = 13 := by
  sorry

end sugar_amount_l3780_378027


namespace car_speed_second_hour_l3780_378037

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 70) 
  (h2 : average_speed = 80) : 
  (2 * average_speed - speed_first_hour) = 90 := by
  sorry

end car_speed_second_hour_l3780_378037


namespace similar_triangles_solution_l3780_378017

/-- Two similar right triangles, one with legs 12 and 9, the other with legs x and 6 -/
def similar_triangles (x : ℝ) : Prop :=
  (12 : ℝ) / x = 9 / 6

theorem similar_triangles_solution :
  ∃ x : ℝ, similar_triangles x ∧ x = 8 := by
sorry

end similar_triangles_solution_l3780_378017
