import Mathlib

namespace tax_rate_calculation_l341_34175

theorem tax_rate_calculation (tax_rate_percent : ℝ) (base_amount : ℝ) :
  tax_rate_percent = 82 ∧ base_amount = 100 →
  tax_rate_percent / 100 * base_amount = 82 := by
sorry

end tax_rate_calculation_l341_34175


namespace birth_year_age_problem_l341_34187

theorem birth_year_age_problem :
  ∀ Y : ℕ,
  1900 ≤ Y → Y ≤ 1988 →
  (1988 - Y = (Y % 100 / 10) * (Y % 10)) →
  (Y = 1964 ∧ 1988 - Y = 24) := by
sorry

end birth_year_age_problem_l341_34187


namespace unique_solution_l341_34194

-- Define the possible colors
inductive Color
| Red
| Blue

-- Define a structure for a child's outfit
structure Outfit :=
  (tshirt : Color)
  (shorts : Color)

-- Define the children
structure Children :=
  (Alyna : Outfit)
  (Bohdan : Outfit)
  (Vika : Outfit)
  (Grysha : Outfit)

-- Define the conditions
def satisfies_conditions (c : Children) : Prop :=
  (c.Alyna.tshirt = Color.Red) ∧
  (c.Bohdan.tshirt = Color.Red) ∧
  (c.Alyna.shorts ≠ c.Bohdan.shorts) ∧
  (c.Vika.tshirt ≠ c.Grysha.tshirt) ∧
  (c.Vika.shorts = Color.Blue) ∧
  (c.Grysha.shorts = Color.Blue) ∧
  (c.Alyna.tshirt ≠ c.Vika.tshirt) ∧
  (c.Alyna.shorts ≠ c.Vika.shorts)

-- Define the correct answer
def correct_answer : Children :=
  { Alyna := { tshirt := Color.Red, shorts := Color.Red },
    Bohdan := { tshirt := Color.Red, shorts := Color.Blue },
    Vika := { tshirt := Color.Blue, shorts := Color.Blue },
    Grysha := { tshirt := Color.Red, shorts := Color.Blue } }

-- The theorem to prove
theorem unique_solution :
  ∀ c : Children, satisfies_conditions c → c = correct_answer :=
sorry

end unique_solution_l341_34194


namespace percent_equality_l341_34141

theorem percent_equality (x : ℝ) : 
  (75 / 100) * 600 = (50 / 100) * x → x = 900 := by sorry

end percent_equality_l341_34141


namespace equation_holds_l341_34100

theorem equation_holds (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
  sorry

end equation_holds_l341_34100


namespace square_side_factor_l341_34117

theorem square_side_factor : ∃ f : ℝ, 
  (∀ s : ℝ, s > 0 → s^2 = 20 * (f*s)^2) ∧ f = Real.sqrt 5 / 10 := by
  sorry

end square_side_factor_l341_34117


namespace linear_function_k_value_l341_34157

/-- Given a linear function y = kx + 3 passing through the point (2, 5), prove that k = 1 -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3) → 
  (5 : ℝ) = k * 2 + 3 → 
  k = 1 := by sorry

end linear_function_k_value_l341_34157


namespace picture_placement_l341_34178

/-- Given a wall of width 27 feet and a centered picture of width 5 feet,
    the distance from the end of the wall to the nearest edge of the picture is 11 feet. -/
theorem picture_placement (wall_width picture_width : ℝ) (h1 : wall_width = 27) (h2 : picture_width = 5) :
  (wall_width - picture_width) / 2 = 11 := by
  sorry

end picture_placement_l341_34178


namespace system_solution_is_e_l341_34126

theorem system_solution_is_e (x y z : ℝ) : 
  x = Real.exp (Real.log y) ∧ 
  y = Real.exp (Real.log z) ∧ 
  z = Real.exp (Real.log x) → 
  x = y ∧ y = z ∧ x = Real.exp 1 := by
  sorry

end system_solution_is_e_l341_34126


namespace prime_with_integer_roots_l341_34128

theorem prime_with_integer_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 300*p = 0 ∧ y^2 + p*y - 300*p = 0) → 
  1 < p ∧ p ≤ 11 := by
sorry

end prime_with_integer_roots_l341_34128


namespace probability_divisible_by_9_l341_34179

def number_set : Set ℕ := {n | 8 ≤ n ∧ n ≤ 28}

def is_divisible_by_9 (a b c : ℕ) : Prop :=
  (a + b + c) % 9 = 0

def favorable_outcomes : ℕ := 150

def total_outcomes : ℕ := 1330

theorem probability_divisible_by_9 :
  (favorable_outcomes : ℚ) / total_outcomes = 15 / 133 := by
  sorry

end probability_divisible_by_9_l341_34179


namespace trig_expression_equals_four_l341_34105

theorem trig_expression_equals_four :
  1 / Real.cos (80 * π / 180) - Real.sqrt 3 / Real.cos (10 * π / 180) = 4 := by
  sorry

end trig_expression_equals_four_l341_34105


namespace product_nonpositive_implies_factor_nonpositive_l341_34156

theorem product_nonpositive_implies_factor_nonpositive (a b : ℝ) : 
  a * b ≤ 0 → a ≤ 0 ∨ b ≤ 0 := by sorry

end product_nonpositive_implies_factor_nonpositive_l341_34156


namespace train_speed_l341_34109

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 360) (h2 : time = 30) :
  length / time = 12 := by
  sorry

end train_speed_l341_34109


namespace f_even_and_increasing_l341_34136

-- Define the function f(x) = 2|x|
def f (x : ℝ) : ℝ := 2 * abs x

-- State the theorem
theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧ -- f is even
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) -- f is increasing on (0, +∞)
  := by sorry

end f_even_and_increasing_l341_34136


namespace max_slope_on_circle_l341_34169

theorem max_slope_on_circle (x y : ℝ) :
  x^2 + y^2 - 2*x - 2 = 0 →
  (∀ a b : ℝ, a^2 + b^2 - 2*a - 2 = 0 → (y + 1) / (x + 1) ≤ (b + 1) / (a + 1)) →
  (y + 1) / (x + 1) = 2 + Real.sqrt 6 :=
by sorry

end max_slope_on_circle_l341_34169


namespace sum_product_inequality_l341_34147

theorem sum_product_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := by
sorry

end sum_product_inequality_l341_34147


namespace product_of_numbers_l341_34120

theorem product_of_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 := by
  sorry

end product_of_numbers_l341_34120


namespace klinker_age_relation_l341_34181

/-- Represents the ages of Mr. Klinker, Julie, and Tim -/
structure Ages where
  klinker : ℕ
  julie : ℕ
  tim : ℕ

/-- The current ages -/
def currentAges : Ages := { klinker := 48, julie := 12, tim := 8 }

/-- The number of years to pass -/
def yearsLater : ℕ := 12

/-- Calculates the ages after a given number of years -/
def agesAfter (initial : Ages) (years : ℕ) : Ages :=
  { klinker := initial.klinker + years
  , julie := initial.julie + years
  , tim := initial.tim + years }

/-- Theorem stating that after 12 years, Mr. Klinker will be twice as old as Julie and thrice as old as Tim -/
theorem klinker_age_relation :
  let futureAges := agesAfter currentAges yearsLater
  futureAges.klinker = 2 * futureAges.julie ∧ futureAges.klinker = 3 * futureAges.tim :=
by sorry

end klinker_age_relation_l341_34181


namespace cricket_average_l341_34189

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (current_average : ℕ) : 
  innings = 12 → 
  next_runs = 178 → 
  increase = 10 → 
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase → 
  current_average = 48 := by
sorry

end cricket_average_l341_34189


namespace cube_surface_area_l341_34197

/-- The surface area of a cube with edge length 2 is 24 -/
theorem cube_surface_area : 
  let edge_length : ℝ := 2
  let surface_area_formula (x : ℝ) := 6 * x * x
  surface_area_formula edge_length = 24 := by
sorry

end cube_surface_area_l341_34197


namespace gdp_equality_l341_34176

/-- Represents the GDP value in trillion yuan -/
def gdp_trillion : ℝ := 33.5

/-- Represents the GDP value in scientific notation -/
def gdp_scientific : ℝ := 3.35 * (10 ^ 13)

/-- Theorem stating that the GDP value in trillion yuan is equal to its scientific notation -/
theorem gdp_equality : gdp_trillion * (10 ^ 12) = gdp_scientific := by sorry

end gdp_equality_l341_34176


namespace rationalize_denominator_l341_34150

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end rationalize_denominator_l341_34150


namespace right_triangle_angle_l341_34114

theorem right_triangle_angle (α β : ℝ) : 
  α + β + 90 = 180 → β = 70 → α = 20 := by
  sorry

end right_triangle_angle_l341_34114


namespace medicine_supply_duration_l341_34138

-- Define the given conditions
def pills_per_supply : ℕ := 90
def pill_fraction : ℚ := 3/4
def days_between_doses : ℕ := 3
def days_per_month : ℕ := 30

-- Define the theorem
theorem medicine_supply_duration :
  (pills_per_supply * days_between_doses / pill_fraction) / days_per_month = 12 := by
  sorry

end medicine_supply_duration_l341_34138


namespace count_triples_satisfying_equation_l341_34101

theorem count_triples_satisfying_equation : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun t : ℕ × ℕ × ℕ => 
      let (x, y, z) := t
      (x^y)^z = 64 ∧ x > 0 ∧ y > 0 ∧ z > 0)
    (Finset.product (Finset.range 64) (Finset.product (Finset.range 64) (Finset.range 64)))).card
  ∧ n = 9 := by
sorry

end count_triples_satisfying_equation_l341_34101


namespace ln_inequality_l341_34123

theorem ln_inequality (x : ℝ) (h : x > 0) : Real.log x ≤ x - 1 := by
  sorry

end ln_inequality_l341_34123


namespace semicircle_radius_is_24_over_5_l341_34159

/-- A right triangle with a semicircle inscribed -/
structure RightTriangleWithSemicircle where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side QR -/
  qr : ℝ
  /-- Radius of the inscribed semicircle -/
  r : ℝ
  /-- PQ is positive -/
  pq_pos : 0 < pq
  /-- QR is positive -/
  qr_pos : 0 < qr
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : pq^2 = qr^2 + (pq - qr)^2
  /-- The radius satisfies the relation with sides -/
  radius_relation : r = (pq * qr) / (pq + qr + (pq - qr))

/-- The main theorem: For a right triangle with PQ = 15 and QR = 8, 
    the radius of the inscribed semicircle is 24/5 -/
theorem semicircle_radius_is_24_over_5 :
  ∃ (t : RightTriangleWithSemicircle), t.pq = 15 ∧ t.qr = 8 ∧ t.r = 24/5 := by
  sorry

end semicircle_radius_is_24_over_5_l341_34159


namespace quadratic_inequality_solution_set_l341_34164

/-- Given that the solution set of ax² + bx + c > 0 is {x | -4 < x < 7},
    prove that the solution set of cx² - bx + a > 0 is {x | x < -1/7 or x > 1/4} -/
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, (a * x^2 + b * x + c > 0) ↔ (-4 < x ∧ x < 7)) :
  ∀ x : ℝ, (c * x^2 - b * x + a > 0) ↔ (x < -1/7 ∨ x > 1/4) :=
sorry

end quadratic_inequality_solution_set_l341_34164


namespace sin_2alpha_value_l341_34195

theorem sin_2alpha_value (α : Real) (h : Real.sin α + Real.cos α = 1/3) :
  Real.sin (2 * α) = -8/9 := by
  sorry

end sin_2alpha_value_l341_34195


namespace range_of_a_l341_34163

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + 2*x + a ≥ 0) → a ≥ -8 := by
  sorry

end range_of_a_l341_34163


namespace f_satisfies_conditions_l341_34121

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x

-- State the theorem
theorem f_satisfies_conditions :
  -- Condition 1: The domain is ℝ (implicitly satisfied by the definition)
  -- Condition 2: For any a, b ∈ ℝ where a + b = 0, f(a) + f(b) = 0
  (∀ a b : ℝ, a + b = 0 → f a + f b = 0) ∧
  -- Condition 3: For any x ∈ ℝ, if m < 0, then f(x) > f(x + m)
  (∀ x m : ℝ, m < 0 → f x > f (x + m)) :=
by
  sorry

end f_satisfies_conditions_l341_34121


namespace arithmetic_sequence_problem_l341_34155

theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 = 4 →
  a 4 = 2 →
  a 8 = -2 := by
sorry

end arithmetic_sequence_problem_l341_34155


namespace square_circle_puzzle_l341_34142

theorem square_circle_puzzle (x y : ℚ) 
  (eq1 : 5 * x + 2 * y = 39)
  (eq2 : 3 * x + 3 * y = 27) :
  x = 7 ∧ y = 2 := by
  sorry

end square_circle_puzzle_l341_34142


namespace operation_twice_equals_twenty_l341_34190

theorem operation_twice_equals_twenty (v : ℝ) : 
  (v - v / 3) - ((v - v / 3) / 3) = 20 → v = 45 := by
sorry

end operation_twice_equals_twenty_l341_34190


namespace working_hours_growth_equation_l341_34144

theorem working_hours_growth_equation 
  (initial_hours : ℝ) 
  (final_hours : ℝ) 
  (growth_rate : ℝ) 
  (h1 : initial_hours = 40) 
  (h2 : final_hours = 48.4) :
  initial_hours * (1 + growth_rate)^2 = final_hours := by
sorry

end working_hours_growth_equation_l341_34144


namespace service_fee_calculation_l341_34165

/-- Calculate the service fee for ticket purchase --/
theorem service_fee_calculation (num_tickets : ℕ) (ticket_price total_paid : ℚ) :
  num_tickets = 3 →
  ticket_price = 44 →
  total_paid = 150 →
  total_paid - (num_tickets : ℚ) * ticket_price = 18 := by
  sorry

end service_fee_calculation_l341_34165


namespace cos_double_angle_special_l341_34107

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and a point (3,4) on its terminal side, prove that cos 2α = -7/25 -/
theorem cos_double_angle_special (α : Real) 
  (h1 : ∃ (x y : Real), x = 3 ∧ y = 4 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
                        y = Real.sin α * Real.sqrt (x^2 + y^2)) : 
  Real.cos (2 * α) = -7/25 := by
  sorry

end cos_double_angle_special_l341_34107


namespace badminton_probabilities_l341_34158

/-- Represents the state of the badminton game -/
inductive GameState
  | Playing (a b c : ℕ) -- number of consecutive losses for each player
  | Winner (player : Fin 3)

/-- Represents a single game outcome -/
inductive GameOutcome
  | Win
  | Lose

/-- The rules of the badminton game -/
def next_state (s : GameState) (outcome : GameOutcome) : GameState :=
  sorry

/-- The probability of a player winning a single game -/
def win_probability : ℚ := 1/2

/-- The probability of A winning four consecutive games -/
def prob_a_wins_four : ℚ := sorry

/-- The probability of needing a fifth game -/
def prob_fifth_game : ℚ := sorry

/-- The probability of C being the ultimate winner -/
def prob_c_wins : ℚ := sorry

theorem badminton_probabilities :
  prob_a_wins_four = 1/16 ∧
  prob_fifth_game = 3/4 ∧
  prob_c_wins = 7/16 :=
sorry

end badminton_probabilities_l341_34158


namespace frank_reading_time_l341_34177

theorem frank_reading_time (pages_per_day : ℕ) (total_pages : ℕ) (h1 : pages_per_day = 8) (h2 : total_pages = 576) :
  total_pages / pages_per_day = 72 := by
  sorry

end frank_reading_time_l341_34177


namespace rectangle_length_percentage_l341_34196

theorem rectangle_length_percentage (area : ℝ) (breadth : ℝ) (length : ℝ) : 
  area = 460 →
  breadth = 20 →
  area = length * breadth →
  (length - breadth) / breadth * 100 = 15 :=
by
  sorry

end rectangle_length_percentage_l341_34196


namespace complex_power_sum_l341_34152

theorem complex_power_sum (z : ℂ) (h : z + z⁻¹ = -Real.sqrt 2) : z^12 + z⁻¹^12 = -2 := by
  sorry

end complex_power_sum_l341_34152


namespace f_value_at_ln_one_third_l341_34183

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x) / (2^x + 1) + a * x

theorem f_value_at_ln_one_third (a : ℝ) :
  (f a (Real.log 3) = 3) → (f a (Real.log (1/3)) = -2) := by
  sorry

end f_value_at_ln_one_third_l341_34183


namespace nested_subtraction_simplification_l341_34161

theorem nested_subtraction_simplification (y : ℝ) : 1 - (2 - (3 - (4 - (5 - y)))) = 3 - y := by
  sorry

end nested_subtraction_simplification_l341_34161


namespace decimal_to_fraction_l341_34160

theorem decimal_to_fraction : 
  (3.68 : ℚ) = 92 / 25 := by sorry

end decimal_to_fraction_l341_34160


namespace preimage_of_3_1_l341_34182

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2*p.2, 2*p.1 - p.2)

-- Theorem statement
theorem preimage_of_3_1 : f (1, 1) = (3, 1) := by
  sorry

end preimage_of_3_1_l341_34182


namespace sector_central_angle_l341_34125

-- Define the sector
structure Sector where
  circumference : ℝ
  area : ℝ

-- Define the given sector
def given_sector : Sector := { circumference := 6, area := 2 }

-- Define the possible central angles
def possible_angles : Set ℝ := {1, 4}

-- Theorem statement
theorem sector_central_angle (s : Sector) (h1 : s = given_sector) :
  ∃ θ ∈ possible_angles, 
    ∃ r l : ℝ, 
      r > 0 ∧ 
      l > 0 ∧ 
      2 * r + l = s.circumference ∧ 
      1 / 2 * r * l = s.area ∧ 
      θ = l / r :=
sorry

end sector_central_angle_l341_34125


namespace solution_set_equals_given_values_l341_34115

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The set of solutions to the equation n = 2S(n)³ + 8 -/
def SolutionSet : Set ℕ := {n : ℕ | n > 0 ∧ n = 2 * (S n)^3 + 8}

/-- The theorem stating that the solution set contains exactly 10, 2008, and 13726 -/
theorem solution_set_equals_given_values : 
  SolutionSet = {10, 2008, 13726} := by sorry

end solution_set_equals_given_values_l341_34115


namespace ratio_of_segments_l341_34110

/-- Given points P, Q, R, and S on a line in that order, with PQ = 3, QR = 6, and PS = 20,
    the ratio of PR to QS is 9/17. -/
theorem ratio_of_segments (P Q R S : ℝ) : 
  P < Q ∧ Q < R ∧ R < S →  -- Points are in order on the line
  Q - P = 3 →              -- PQ = 3
  R - Q = 6 →              -- QR = 6
  S - P = 20 →             -- PS = 20
  (R - P) / (S - Q) = 9 / 17 := by
sorry

end ratio_of_segments_l341_34110


namespace hexagon_to_square_area_equality_l341_34132

/-- Proves that a square with side length s = √(3√3/2) * a has the same area as a regular hexagon with side length a -/
theorem hexagon_to_square_area_equality (a : ℝ) (h : a > 0) :
  let s := Real.sqrt (3 * Real.sqrt 3 / 2) * a
  s^2 = (3 * Real.sqrt 3 / 2) * a^2 := by
  sorry

#check hexagon_to_square_area_equality

end hexagon_to_square_area_equality_l341_34132


namespace common_root_values_l341_34116

theorem common_root_values (a b c d k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hk1 : a * k^4 + b * k^3 + c * k^2 + d * k + a = 0)
  (hk2 : b * k^4 + c * k^3 + d * k^2 + a * k + b = 0) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end common_root_values_l341_34116


namespace probability_of_even_sum_l341_34168

def number_of_balls : ℕ := 12

def is_even_sum (a b : ℕ) : Prop := Even (a + b)

theorem probability_of_even_sum :
  let total_outcomes := number_of_balls * (number_of_balls - 1)
  let favorable_outcomes := (number_of_balls / 2) * ((number_of_balls / 2) - 1) * 2
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 11 := by
  sorry

end probability_of_even_sum_l341_34168


namespace incenter_bisects_orthocenter_circumcenter_angle_l341_34124

-- Define the types for points and triangles
variable (Point : Type)
variable (Triangle : Type)

-- Define the properties of the triangle
variable (is_acute : Triangle → Prop)
variable (orthocenter : Triangle → Point)
variable (circumcenter : Triangle → Point)
variable (incenter : Triangle → Point)

-- Define the angle bisector property
variable (bisects_angle : Point → Point → Point → Point → Prop)

theorem incenter_bisects_orthocenter_circumcenter_angle 
  (ABC : Triangle) (H O I : Point) :
  is_acute ABC →
  H = orthocenter ABC →
  O = circumcenter ABC →
  I = incenter ABC →
  bisects_angle I A H O :=
sorry

end incenter_bisects_orthocenter_circumcenter_angle_l341_34124


namespace division_remainder_problem_l341_34199

theorem division_remainder_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) 
  (h1 : dividend = 127)
  (h2 : divisor = 14)
  (h3 : quotient = 9)
  (h4 : dividend = divisor * quotient + (dividend % divisor)) :
  dividend % divisor = 1 := by
  sorry

end division_remainder_problem_l341_34199


namespace base_seven_digits_of_1234_l341_34111

theorem base_seven_digits_of_1234 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 1234 ∧ 1234 < 7^n ∧ n = 4 := by
  sorry

end base_seven_digits_of_1234_l341_34111


namespace yellow_balls_count_l341_34104

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 100 ∧
  white = 50 ∧
  green = 30 ∧
  red = 7 ∧
  purple = 3 ∧
  prob = 9/10 ∧
  prob = (white + green + (total - white - green - red - purple)) / total →
  total - white - green - red - purple = 10 := by
  sorry

end yellow_balls_count_l341_34104


namespace monotonic_increasing_interval_l341_34174

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + a|

-- State the theorem
theorem monotonic_increasing_interval (a : ℝ) :
  (∀ x ≥ 3, Monotone (fun x => f a x)) ↔ a = -6 :=
sorry

end monotonic_increasing_interval_l341_34174


namespace square_of_nine_ones_l341_34184

theorem square_of_nine_ones : (111111111 : ℕ)^2 = 12345678987654321 := by
  sorry

end square_of_nine_ones_l341_34184


namespace tracy_art_fair_sales_l341_34153

/-- The number of paintings sold at Tracy's art fair booth --/
def paintings_sold (group1_count group1_paintings group2_count group2_paintings group3_count group3_paintings : ℕ) : ℕ :=
  group1_count * group1_paintings + group2_count * group2_paintings + group3_count * group3_paintings

/-- Theorem stating the total number of paintings sold at Tracy's art fair booth --/
theorem tracy_art_fair_sales : paintings_sold 4 2 12 1 4 4 = 36 := by
  sorry

end tracy_art_fair_sales_l341_34153


namespace circle_area_with_diameter_10_l341_34108

theorem circle_area_with_diameter_10 :
  ∀ (d : ℝ) (A : ℝ), 
    d = 10 →
    A = π * (d / 2)^2 →
    A = 25 * π := by
  sorry

end circle_area_with_diameter_10_l341_34108


namespace students_with_dogs_l341_34145

theorem students_with_dogs (total_students : ℕ) (girls_percentage : ℚ) (boys_percentage : ℚ)
  (girls_with_dogs_percentage : ℚ) (boys_with_dogs_percentage : ℚ)
  (h1 : total_students = 100)
  (h2 : girls_percentage = 1/2)
  (h3 : boys_percentage = 1/2)
  (h4 : girls_with_dogs_percentage = 1/5)
  (h5 : boys_with_dogs_percentage = 1/10) :
  (total_students : ℚ) * girls_percentage * girls_with_dogs_percentage +
  (total_students : ℚ) * boys_percentage * boys_with_dogs_percentage = 15 :=
by sorry

end students_with_dogs_l341_34145


namespace puppies_sold_l341_34118

/-- Given a pet store scenario, prove the number of puppies sold -/
theorem puppies_sold (initial_puppies cages_used puppies_per_cage : ℕ) :
  initial_puppies - (cages_used * puppies_per_cage) =
  initial_puppies - cages_used * puppies_per_cage :=
by sorry

end puppies_sold_l341_34118


namespace triangle_area_set_S_is_two_horizontal_lines_l341_34198

/-- The set of points A(x, y) for which the area of triangle ABC is 2,
    where B(1, 0) and C(-1, 0) are fixed points -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; abs y = 2}

/-- The area of triangle ABC given point A(x, y) and fixed points B(1, 0) and C(-1, 0) -/
def triangleArea (x y : ℝ) : ℝ := abs y

theorem triangle_area_set :
  ∀ (x y : ℝ), (x, y) ∈ S ↔ triangleArea x y = 2 :=
by sorry

theorem S_is_two_horizontal_lines :
  S = {p : ℝ × ℝ | let (x, y) := p; y = 2 ∨ y = -2} :=
by sorry

end triangle_area_set_S_is_two_horizontal_lines_l341_34198


namespace john_popcorn_profit_l341_34146

/-- Calculates the profit from selling popcorn bags -/
def popcorn_profit (cost_price selling_price number_of_bags : ℕ) : ℕ :=
  (selling_price - cost_price) * number_of_bags

/-- Theorem: John's profit from selling 30 bags of popcorn is $120 -/
theorem john_popcorn_profit :
  popcorn_profit 4 8 30 = 120 := by
  sorry

end john_popcorn_profit_l341_34146


namespace ab_equals_six_l341_34140

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l341_34140


namespace compare_exponential_expressions_l341_34166

theorem compare_exponential_expressions :
  let a := 4^(1/4)
  let b := 5^(1/5)
  let c := 16^(1/16)
  let d := 25^(1/25)
  (a > b ∧ a > c ∧ a > d) ∧
  (b > c ∧ b > d) :=
by sorry

end compare_exponential_expressions_l341_34166


namespace football_team_right_handed_players_l341_34173

theorem football_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (multiple_position : ℕ)
  (left_to_right_ratio : ℚ)
  (h1 : total_players = 120)
  (h2 : throwers = 60)
  (h3 : multiple_position = 20)
  (h4 : left_to_right_ratio = 2 / 3)
  (h5 : throwers + multiple_position ≤ total_players) :
  throwers + multiple_position + ((total_players - (throwers + multiple_position)) / (1 + left_to_right_ratio⁻¹)) = 104 :=
by sorry

end football_team_right_handed_players_l341_34173


namespace production_average_l341_34119

theorem production_average (n : ℕ) : 
  (n * 50 + 110) / (n + 1) = 55 → n = 11 := by
  sorry

end production_average_l341_34119


namespace decreasing_function_range_l341_34171

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then (a - 5) * x - 2
  else x^2 - 2 * (a + 1) * x + 3 * a

-- Define the condition for the function to be decreasing
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

-- Theorem statement
theorem decreasing_function_range (a : ℝ) :
  is_decreasing (f a) ↔ a ∈ Set.Icc 1 4 :=
sorry

end decreasing_function_range_l341_34171


namespace intersection_line_of_circles_l341_34172

theorem intersection_line_of_circles (x y : ℝ) : 
  (x^2 + y^2 - 4*x = 0 ∧ x^2 + y^2 + 4*y = 0) → y = -x := by
  sorry

end intersection_line_of_circles_l341_34172


namespace gcd_n_cube_minus_27_and_n_plus_3_l341_34103

theorem gcd_n_cube_minus_27_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^3 - 27) (n + 3) = if (n + 3) % 9 = 0 then 9 else 1 := by sorry

end gcd_n_cube_minus_27_and_n_plus_3_l341_34103


namespace molly_current_age_l341_34131

/-- Represents the ages of Sandy, Molly, and Danny -/
structure Ages where
  sandy : ℕ
  molly : ℕ
  danny : ℕ

/-- The ratio of ages is 4:3:5 -/
def age_ratio (a : Ages) : Prop :=
  ∃ (x : ℕ), a.sandy = 4 * x ∧ a.molly = 3 * x ∧ a.danny = 5 * x

/-- Sandy's age after 6 years is 30 -/
def sandy_future_age (a : Ages) : Prop :=
  a.sandy + 6 = 30

/-- Theorem stating that under the given conditions, Molly's current age is 18 -/
theorem molly_current_age (a : Ages) :
  age_ratio a → sandy_future_age a → a.molly = 18 := by
  sorry


end molly_current_age_l341_34131


namespace sum_of_powers_of_three_and_negative_three_l341_34122

theorem sum_of_powers_of_three_and_negative_three : 
  (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
sorry

end sum_of_powers_of_three_and_negative_three_l341_34122


namespace prob_product_divisible_l341_34106

/-- Represents a standard 6-sided die --/
def StandardDie : Type := Fin 6

/-- The probability of rolling a number on a standard die --/
def prob_roll (n : Nat) : ℚ := if n ≥ 1 ∧ n ≤ 6 then 1 / 6 else 0

/-- The probability that a single die roll is not divisible by 2, 3, and 5 --/
def prob_not_divisible : ℚ := 5 / 18

/-- The number of dice rolled --/
def num_dice : Nat := 6

/-- The probability that the product of 6 dice rolls is divisible by 2, 3, or 5 --/
theorem prob_product_divisible :
  1 - prob_not_divisible ^ num_dice = 33996599 / 34012224 := by sorry

end prob_product_divisible_l341_34106


namespace larger_number_proof_l341_34188

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 := by
  sorry

end larger_number_proof_l341_34188


namespace ab_one_sufficient_not_necessary_l341_34167

theorem ab_one_sufficient_not_necessary (a b : ℝ) : 
  (a * b = 1 → a^2 + b^2 ≥ 2) ∧ 
  ∃ a b : ℝ, a^2 + b^2 ≥ 2 ∧ a * b ≠ 1 :=
sorry

end ab_one_sufficient_not_necessary_l341_34167


namespace prob_at_least_one_2_or_4_is_5_9_l341_34133

/-- The probability of rolling a 2 or 4 on a single fair 6-sided die -/
def prob_2_or_4 : ℚ := 1/3

/-- The probability of not rolling a 2 or 4 on a single fair 6-sided die -/
def prob_not_2_or_4 : ℚ := 2/3

/-- The probability of at least one die showing 2 or 4 when rolling two fair 6-sided dice -/
def prob_at_least_one_2_or_4 : ℚ := 1 - (prob_not_2_or_4 * prob_not_2_or_4)

theorem prob_at_least_one_2_or_4_is_5_9 : 
  prob_at_least_one_2_or_4 = 5/9 := by sorry

end prob_at_least_one_2_or_4_is_5_9_l341_34133


namespace miguel_book_pages_l341_34139

/-- The number of pages Miguel read in his book over two weeks --/
def total_pages : ℕ :=
  let first_four_days := 4 * 48
  let next_five_days := 5 * 35
  let subsequent_four_days := 4 * 28
  let last_day := 19
  first_four_days + next_five_days + subsequent_four_days + last_day

/-- Theorem stating that the total number of pages in Miguel's book is 498 --/
theorem miguel_book_pages : total_pages = 498 := by
  sorry

end miguel_book_pages_l341_34139


namespace circuit_disconnection_possibilities_l341_34162

theorem circuit_disconnection_possibilities :
  let n : ℕ := 7  -- number of resistors
  let total_possibilities : ℕ := 2^n - 1  -- total number of ways at least one resistor can be disconnected
  total_possibilities = 63 := by
  sorry

end circuit_disconnection_possibilities_l341_34162


namespace inequality_chain_l341_34192

theorem inequality_chain (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 * a * b) / (a + b) < Real.sqrt (a * b) ∧
  Real.sqrt (a * b) < (a + b) / 2 ∧
  (a + b) / 2 < Real.sqrt ((a^2 + b^2) / 2) :=
by sorry

end inequality_chain_l341_34192


namespace quiz_goal_achievement_l341_34135

theorem quiz_goal_achievement (total_quizzes : ℕ) (goal_percentage : ℚ)
  (completed_quizzes : ℕ) (as_scored : ℕ) (remaining_quizzes : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : goal_percentage = 85 / 100)
  (h3 : completed_quizzes = 40)
  (h4 : as_scored = 32)
  (h5 : remaining_quizzes = 20)
  (h6 : completed_quizzes + remaining_quizzes = total_quizzes) :
  (total_quizzes * goal_percentage).floor - as_scored = remaining_quizzes - 1 :=
sorry

end quiz_goal_achievement_l341_34135


namespace root_ratio_sum_squared_l341_34113

theorem root_ratio_sum_squared (k₁ k₂ : ℝ) (a b : ℝ) : 
  (∀ x, k₁ * (x^2 - x) + x + 3 = 0 → (x = a ∨ x = b)) →
  (∀ x, k₂ * (x^2 - x) + x + 3 = 0 → (x = a ∨ x = b)) →
  a / b + b / a = 2 →
  k₁^2 + k₂^2 = 194 := by
sorry

end root_ratio_sum_squared_l341_34113


namespace right_triangle_arithmetic_progression_81_l341_34112

theorem right_triangle_arithmetic_progression_81 :
  ∃ (a d : ℕ), 
    (a > 0) ∧ (d > 0) ∧
    (a - d)^2 + a^2 = (a + d)^2 ∧
    (81 = a - d ∨ 81 = a ∨ 81 = a + d) :=
by sorry

end right_triangle_arithmetic_progression_81_l341_34112


namespace comic_book_problem_l341_34185

theorem comic_book_problem (initial_books : ℕ) : 
  (initial_books / 3 + 15 = 45) → initial_books = 90 := by
  sorry

end comic_book_problem_l341_34185


namespace m_value_range_l341_34180

/-- The equation x^2 + 2√2x + m = 0 has two distinct real roots -/
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + 2 * Real.sqrt 2 * x + m = 0 ∧ y^2 + 2 * Real.sqrt 2 * y + m = 0

/-- The solution set of the inequality 4x^2 + 4(m-2)x + 1 > 0 is ℝ -/
def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 > 0

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := m ≤ 1 ∨ (2 ≤ m ∧ m < 3)

theorem m_value_range :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
by sorry

end m_value_range_l341_34180


namespace inequality_system_solution_l341_34129

theorem inequality_system_solution (x : ℝ) :
  (4 * x - 6 < 2 * x ∧ (3 * x - 1) / 2 ≥ 2 * x - 1) ↔ x ≤ 1 :=
by sorry

end inequality_system_solution_l341_34129


namespace correct_expansion_l341_34102

theorem correct_expansion (x : ℝ) : (-3*x + 2) * (-3*x - 2) = 9*x^2 - 4 := by
  sorry

end correct_expansion_l341_34102


namespace system_of_equations_solution_l341_34137

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x - y = -3) ∧ (4 * x - 5 * y = -21) :=
by
  -- Proof goes here
  sorry

end system_of_equations_solution_l341_34137


namespace tax_rate_percentage_l341_34170

/-- A tax rate of $65 per $100.00 is equivalent to 65% -/
theorem tax_rate_percentage : 
  let tax_amount : ℚ := 65
  let base_amount : ℚ := 100
  (tax_amount / base_amount) * 100 = 65 := by sorry

end tax_rate_percentage_l341_34170


namespace max_b_no_lattice_points_l341_34143

theorem max_b_no_lattice_points :
  let max_b : ℚ := 67 / 199
  ∀ m : ℚ, 1/3 < m → m < max_b →
    ∀ x : ℕ, 0 < x → x ≤ 200 →
      ∀ y : ℤ, y ≠ ⌊m * x + 3⌋ ∧
    ∀ b : ℚ, b > max_b →
      ∃ m : ℚ, 1/3 < m ∧ m < b ∧
        ∃ x : ℕ, 0 < x ∧ x ≤ 200 ∧
          ∃ y : ℤ, y = ⌊m * x + 3⌋ := by
  sorry

end max_b_no_lattice_points_l341_34143


namespace min_value_expression_min_value_achievable_l341_34130

theorem min_value_expression (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2023 + 3 * Real.cos (2 * x) ≥ 2017 :=
by sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2023 + 3 * Real.cos (2 * x) = 2017 :=
by sorry

end min_value_expression_min_value_achievable_l341_34130


namespace quadratic_sum_l341_34127

/-- Given a quadratic function f(x) = -3x^2 + 27x + 135, 
    prove that when written in the form a(x+b)^2 + c,
    the sum of a, b, and c is 197.75 -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = -3*x^2 + 27*x + 135) →
  (∀ x, f x = a*(x+b)^2 + c) →
  a + b + c = 197.75 := by
  sorry

end quadratic_sum_l341_34127


namespace ibrahim_savings_is_55_l341_34148

/-- The amount of money Ibrahim has in savings -/
def ibrahimSavings (mp3Cost cdCost fatherContribution amountLacking : ℕ) : ℕ :=
  (mp3Cost + cdCost) - fatherContribution - amountLacking

/-- Theorem stating that Ibrahim's savings are 55 euros -/
theorem ibrahim_savings_is_55 :
  ibrahimSavings 120 19 20 64 = 55 := by
  sorry

end ibrahim_savings_is_55_l341_34148


namespace set_equality_l341_34151

def A : Set ℤ := {-2, -1, 0, 1, 2}

theorem set_equality : {y : ℤ | ∃ x ∈ A, y = |x + 1|} = {0, 1, 2, 3} := by
  sorry

end set_equality_l341_34151


namespace age_ratio_l341_34154

theorem age_ratio (a b : ℕ) : 
  (a - 4 = b + 4) → 
  ((a + 4) = 3 * (b - 4)) → 
  (a : ℚ) / b = 5 / 3 := by
sorry

end age_ratio_l341_34154


namespace problem_solution_l341_34193

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 6) : 
  (x^5 + 3*y^3) / 9 = 99 := by
  sorry

end problem_solution_l341_34193


namespace triangle_abc_properties_l341_34149

theorem triangle_abc_properties (A B C : Real) (a b c : Real) (S : Real) :
  c = Real.sqrt 3 →
  b = 1 →
  C = 2 * π / 3 →  -- 120° in radians
  B = π / 6 ∧      -- 30° in radians
  S = Real.sqrt 3 / 4 :=
by sorry

end triangle_abc_properties_l341_34149


namespace divisibility_when_prime_exists_counterexample_for_composite_l341_34186

-- Define the divisibility relation
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

-- Statement for the case when n is prime
theorem divisibility_when_prime (m n : ℕ) (h1 : n > 1) (h2 : ∀ k, 1 < k → k < n → ¬ divides k n) 
  (h3 : divides (m + n) (m * n)) : divides n m := by sorry

-- Statement for the case when n is a product of two distinct primes
theorem exists_counterexample_for_composite : 
  ∃ m n p q : ℕ, p ≠ q ∧ p.Prime ∧ q.Prime ∧ n = p * q ∧ 
  divides (m + n) (m * n) ∧ ¬ divides n m := by sorry

end divisibility_when_prime_exists_counterexample_for_composite_l341_34186


namespace greatest_sum_consecutive_integers_l341_34191

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → m ≤ n) → n + (n + 1) = 43 := by
  sorry

end greatest_sum_consecutive_integers_l341_34191


namespace surveyor_distance_theorem_l341_34134

/-- The distance traveled by the surveyor when he heard the blast -/
def surveyorDistance : ℝ := 122

/-- The time it takes for the fuse to burn (in seconds) -/
def fuseTime : ℝ := 20

/-- The speed of the surveyor (in yards per second) -/
def surveyorSpeed : ℝ := 6

/-- The speed of sound (in feet per second) -/
def soundSpeed : ℝ := 960

/-- Conversion factor from yards to feet -/
def yardsToFeet : ℝ := 3

theorem surveyor_distance_theorem :
  let t := (soundSpeed * fuseTime) / (soundSpeed - surveyorSpeed * yardsToFeet)
  surveyorDistance = surveyorSpeed * t := by sorry

end surveyor_distance_theorem_l341_34134
