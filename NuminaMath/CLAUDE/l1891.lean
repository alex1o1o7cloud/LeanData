import Mathlib

namespace negation_and_converse_of_divisibility_proposition_l1891_189198

def last_digit (n : ℤ) : ℤ := n % 10

theorem negation_and_converse_of_divisibility_proposition :
  (¬ (∀ n : ℤ, (last_digit n = 0 ∨ last_digit n = 5) → n % 5 = 0) ↔ 
   (∃ n : ℤ, (last_digit n = 0 ∨ last_digit n = 5) ∧ n % 5 ≠ 0)) ∧
  ((∀ n : ℤ, n % 5 = 0 → (last_digit n = 0 ∨ last_digit n = 5)) ↔
   (∀ n : ℤ, (last_digit n ≠ 0 ∧ last_digit n ≠ 5) → n % 5 ≠ 0)) :=
by sorry

end negation_and_converse_of_divisibility_proposition_l1891_189198


namespace square_side_length_l1891_189137

theorem square_side_length (side : ℕ) : side ^ 2 < 20 → side = 4 := by
  sorry

end square_side_length_l1891_189137


namespace town_distance_proof_l1891_189185

/-- Given a map distance and a scale, calculates the actual distance between two towns. -/
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Theorem stating that for a map distance of 7.5 inches and a scale of 1 inch = 8 miles,
    the actual distance between two towns is 60 miles. -/
theorem town_distance_proof :
  let map_distance : ℝ := 7.5
  let scale : ℝ := 8
  actual_distance map_distance scale = 60 := by
  sorry

end town_distance_proof_l1891_189185


namespace janes_total_hours_l1891_189158

/-- Jane's exercise routine -/
structure ExerciseRoutine where
  hours_per_day : ℕ
  days_per_week : ℕ
  weeks : ℕ

/-- Calculate total exercise hours -/
def total_hours (routine : ExerciseRoutine) : ℕ :=
  routine.hours_per_day * routine.days_per_week * routine.weeks

/-- Jane's specific routine -/
def janes_routine : ExerciseRoutine :=
  { hours_per_day := 1
    days_per_week := 5
    weeks := 8 }

/-- Theorem: Jane's total exercise hours equal 40 -/
theorem janes_total_hours : total_hours janes_routine = 40 := by
  sorry

end janes_total_hours_l1891_189158


namespace round_trip_time_l1891_189109

/-- Calculates the total time for a round trip by boat given the boat's speed in standing water,
    the stream's speed, and the distance to the destination. -/
theorem round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 14) 
  (h2 : stream_speed = 1.2) 
  (h3 : distance = 4864) : 
  (distance / (boat_speed + stream_speed)) + (distance / (boat_speed - stream_speed)) = 700 := by
  sorry

#check round_trip_time

end round_trip_time_l1891_189109


namespace hyperbola_triangle_perimeter_l1891_189136

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/16 - y^2/9 = 1

-- Define the points A, B, F₁ (left focus), and F₂ (right focus)
variable (A B F₁ F₂ : ℝ × ℝ)

-- Define the conditions
def chord_passes_through_left_focus : Prop :=
  ∃ t : ℝ, A = (1 - t) • F₁ + t • B ∧ 0 ≤ t ∧ t ≤ 1

def chord_length_is_6 : Prop :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6

-- Define the theorem
theorem hyperbola_triangle_perimeter
  (h1 : hyperbola F₁.1 F₁.2)
  (h2 : hyperbola F₂.1 F₂.2)
  (h3 : chord_passes_through_left_focus A B F₁)
  (h4 : chord_length_is_6 A B) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
  Real.sqrt ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) +
  Real.sqrt ((B.1 - F₂.1)^2 + (B.2 - F₂.2)^2) = 28 :=
by sorry

end hyperbola_triangle_perimeter_l1891_189136


namespace specific_pairings_probability_l1891_189182

/-- The probability of two specific pairings occurring simultaneously in a class of 32 students -/
theorem specific_pairings_probability (n : ℕ) (h : n = 32) : 
  (1 : ℚ) / (n - 1) * (1 : ℚ) / (n - 3) = 1 / 899 :=
sorry

end specific_pairings_probability_l1891_189182


namespace extended_ohara_triple_49_64_l1891_189112

/-- Definition of an Extended O'Hara triple -/
def is_extended_ohara_triple (a b x : ℕ) : Prop :=
  2 * Real.sqrt a + Real.sqrt b = x

/-- Theorem: If (49, 64, x) is an Extended O'Hara triple, then x = 22 -/
theorem extended_ohara_triple_49_64 (x : ℕ) :
  is_extended_ohara_triple 49 64 x → x = 22 := by
  sorry

end extended_ohara_triple_49_64_l1891_189112


namespace lcm_factor_proof_l1891_189165

theorem lcm_factor_proof (A B : ℕ+) (X : ℕ+) : 
  Nat.gcd A B = 23 →
  Nat.lcm A B = 23 * 13 * X →
  A = 322 →
  X = 14 := by
sorry

end lcm_factor_proof_l1891_189165


namespace profit_maximizing_price_optimal_selling_price_is_14_l1891_189193

/-- Profit function given price increase -/
def profit (x : ℝ) : ℝ :=
  (100 - 10 * x) * ((10 + x) - 8)

/-- The price increase that maximizes profit -/
def optimal_price_increase : ℝ := 4

theorem profit_maximizing_price :
  optimal_price_increase = 4 ∧
  ∀ x : ℝ, profit x ≤ profit optimal_price_increase :=
sorry

/-- The optimal selling price -/
def optimal_selling_price : ℝ :=
  10 + optimal_price_increase

theorem optimal_selling_price_is_14 :
  optimal_selling_price = 14 :=
sorry

end profit_maximizing_price_optimal_selling_price_is_14_l1891_189193


namespace money_redistribution_theorem_l1891_189164

/-- Represents the money redistribution problem among three friends. -/
def MoneyRedistribution (a j t : ℚ) : Prop :=
  -- Initial conditions
  (t = 24) ∧
  -- First redistribution (Amy's turn)
  let a₁ := a - 2*j - t
  let j₁ := 3*j
  let t₁ := 2*t
  -- Second redistribution (Jan's turn)
  let a₂ := 2*a₁
  let j₂ := j₁ - (a₁ + t₁)
  let t₂ := 3*t₁
  -- Final redistribution (Toy's turn)
  let a₃ := 3*a₂
  let j₃ := 3*j₂
  let t₃ := t₂ - (a₃ - a₂ + j₃ - j₂)
  -- Final condition
  (t₃ = 24) →
  -- Conclusion
  (a + j + t = 72)

/-- The total amount of money among the three friends is 72 dollars. -/
theorem money_redistribution_theorem (a j t : ℚ) :
  MoneyRedistribution a j t → (a + j + t = 72) :=
by
  sorry


end money_redistribution_theorem_l1891_189164


namespace log_max_min_sum_l1891_189115

theorem log_max_min_sum (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (let f := fun x => Real.log x / Real.log a
   max (f a) (f (2 * a)) + min (f a) (f (2 * a)) = 3) →
  a = 2 := by sorry

end log_max_min_sum_l1891_189115


namespace power_boat_travel_time_l1891_189179

/-- Represents the scenario of a power boat and raft on a river --/
structure RiverScenario where
  r : ℝ  -- Speed of the river current (and raft)
  p : ℝ  -- Speed of the power boat relative to the river
  t : ℝ  -- Time taken by power boat from A to B

/-- The conditions of the problem --/
def scenario_conditions (s : RiverScenario) : Prop :=
  s.r > 0 ∧ s.p > 0 ∧ s.t > 0 ∧
  (s.p + s.r) * s.t + (s.p - s.r) * (9 - s.t) = 9 * s.r

/-- The theorem to be proved --/
theorem power_boat_travel_time (s : RiverScenario) :
  scenario_conditions s → s.t = 4.5 := by
  sorry


end power_boat_travel_time_l1891_189179


namespace sin_inequality_l1891_189160

theorem sin_inequality : 
  Real.sin (11 * π / 180) < Real.sin (168 * π / 180) ∧ 
  Real.sin (168 * π / 180) < Real.cos (10 * π / 180) := by
  sorry

end sin_inequality_l1891_189160


namespace geometric_progression_terms_l1891_189190

-- Define the geometric progression
def geometric_progression (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

-- Define the sum of a geometric progression
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

theorem geometric_progression_terms (a : ℚ) :
  (geometric_progression a (1/3) 4 = 1/54) →
  (geometric_sum a (1/3) 5 = 121/162) →
  ∃ n : ℕ, geometric_sum a (1/3) n = 121/162 ∧ n = 5 :=
by
  sorry

#check geometric_progression_terms

end geometric_progression_terms_l1891_189190


namespace ground_lines_perpendicular_l1891_189177

-- Define a type for lines
def Line : Type := ℝ × ℝ → Prop

-- Define a relation for parallel lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Define a relation for perpendicular lines
def Perpendicular (l1 l2 : Line) : Prop := sorry

-- Define a set of lines on the ground
def GroundLines : Set Line := sorry

-- Define the ruler's orientation
def RulerOrientation : Line := sorry

-- Theorem statement
theorem ground_lines_perpendicular 
  (always_parallel : ∀ (r : Line), ∃ (g : Line), g ∈ GroundLines ∧ Parallel r g) :
  ∀ (l1 l2 : Line), l1 ∈ GroundLines → l2 ∈ GroundLines → l1 ≠ l2 → Perpendicular l1 l2 :=
sorry

end ground_lines_perpendicular_l1891_189177


namespace bananas_permutations_l1891_189186

/-- The number of distinct permutations of a word with repeated letters -/
def distinct_permutations (total : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial total / (List.prod (List.map Nat.factorial repetitions))

/-- Theorem: The number of distinct permutations of BANANAS is 420 -/
theorem bananas_permutations :
  distinct_permutations 7 [3, 2] = 420 := by
  sorry

#eval distinct_permutations 7 [3, 2]

end bananas_permutations_l1891_189186


namespace min_value_reciprocal_sum_l1891_189161

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 4) :
  (2/x + 3/y) ≥ 25/4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 4 ∧ 2/x₀ + 3/y₀ = 25/4 :=
sorry

end min_value_reciprocal_sum_l1891_189161


namespace bush_distance_theorem_l1891_189134

/-- The distance between equally spaced bushes along a road -/
def bush_distance (n : ℕ) (d : ℝ) : ℝ :=
  d * (n - 1)

/-- Theorem: Given 10 equally spaced bushes where the distance between
    the first and fifth bush is 100 feet, the distance between the first
    and last bush is 225 feet. -/
theorem bush_distance_theorem :
  bush_distance 5 100 = 100 →
  bush_distance 10 (100 / 4) = 225 := by
  sorry

end bush_distance_theorem_l1891_189134


namespace minus_six_otimes_minus_two_l1891_189126

-- Define the new operation ⊗
def otimes (a b : ℚ) : ℚ := a^2 + b

-- Theorem statement
theorem minus_six_otimes_minus_two : otimes (-6) (-2) = 34 := by sorry

end minus_six_otimes_minus_two_l1891_189126


namespace fraction_value_l1891_189167

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) := by
  sorry

end fraction_value_l1891_189167


namespace sum_of_multiples_l1891_189175

def smallest_two_digit_multiple_of_5 : ℕ := 10

def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 := by
  sorry

end sum_of_multiples_l1891_189175


namespace num_valid_teams_eq_930_l1891_189101

/-- Represents a debater in the team -/
inductive Debater
| Boy : Fin 4 → Debater
| Girl : Fin 4 → Debater

/-- Represents a debate team -/
def DebateTeam := Fin 4 → Debater

/-- Check if Boy A is in the team -/
def has_boy_A (team : DebateTeam) : Prop :=
  ∃ i, team i = Debater.Boy 0

/-- Check if Girl B is in the team -/
def has_girl_B (team : DebateTeam) : Prop :=
  ∃ i, team i = Debater.Girl 1

/-- Check if Boy A is not the first debater -/
def boy_A_not_first (team : DebateTeam) : Prop :=
  team 0 ≠ Debater.Boy 0

/-- Check if Girl B is not the fourth debater -/
def girl_B_not_fourth (team : DebateTeam) : Prop :=
  team 3 ≠ Debater.Girl 1

/-- Check if the team satisfies all constraints -/
def valid_team (team : DebateTeam) : Prop :=
  boy_A_not_first team ∧
  girl_B_not_fourth team ∧
  (has_boy_A team → has_girl_B team)

/-- The number of valid debate teams -/
def num_valid_teams : ℕ := sorry

theorem num_valid_teams_eq_930 : num_valid_teams = 930 := by sorry

end num_valid_teams_eq_930_l1891_189101


namespace inequality_proof_l1891_189141

/-- An odd function f with the given property -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- The derivative of f -/
noncomputable def f' : ℝ → ℝ := sorry

/-- The property x * f'(x) - f(x) < 0 for x ≠ 0 -/
axiom f_property (x : ℝ) (h : x ≠ 0) : x * f' x - f x < 0

/-- The main theorem -/
theorem inequality_proof :
  f (-3) / (-3) < f (Real.exp 1) / (Real.exp 1) ∧
  f (Real.exp 1) / (Real.exp 1) < f (Real.log 2) / (Real.log 2) := by
  sorry

end inequality_proof_l1891_189141


namespace alyssa_cookie_count_l1891_189188

/-- The number of cookies Aiyanna has -/
def aiyanna_cookies : ℕ := 140

/-- The difference in cookies between Aiyanna and Alyssa -/
def cookie_difference : ℕ := 11

/-- The number of cookies Alyssa has -/
def alyssa_cookies : ℕ := aiyanna_cookies - cookie_difference

theorem alyssa_cookie_count : alyssa_cookies = 129 := by
  sorry

end alyssa_cookie_count_l1891_189188


namespace discount_percentage_calculation_l1891_189178

theorem discount_percentage_calculation (cost_price marked_price : ℝ) (profit_percentage : ℝ) :
  cost_price = 95 →
  marked_price = 125 →
  profit_percentage = 25 →
  ∃ (discount_percentage : ℝ),
    discount_percentage = 5 ∧
    marked_price * (1 - discount_percentage / 100) = cost_price * (1 + profit_percentage / 100) :=
by sorry

end discount_percentage_calculation_l1891_189178


namespace parabola_line_intersection_l1891_189120

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point P
def point_P : ℝ × ℝ := (1, 2)

-- Define a line with slope 1
def line_with_slope_1 (b : ℝ) (x y : ℝ) : Prop := y = x + b

-- Define the intersection points A and B
def intersection_points (b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola_C x₁ y₁ ∧ parabola_C x₂ y₂ ∧
    line_with_slope_1 b x₁ y₁ ∧ line_with_slope_1 b x₂ y₂ ∧
    x₁ ≠ x₂

-- Define the condition for circle AB passing through P
def circle_condition (b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola_C x₁ y₁ ∧ parabola_C x₂ y₂ ∧
    line_with_slope_1 b x₁ y₁ ∧ line_with_slope_1 b x₂ y₂ ∧
    (x₁ - point_P.1) * (x₂ - point_P.1) + (y₁ - point_P.2) * (y₂ - point_P.2) = 0

-- Theorem statement
theorem parabola_line_intersection :
  ∃ (b : ℝ), intersection_points b ∧ circle_condition b ∧ b = -7 :=
sorry

end parabola_line_intersection_l1891_189120


namespace smallest_m_is_13_l1891_189157

/-- The set of complex numbers with real part between 1/2 and √2/2 -/
def S : Set ℂ :=
  {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

/-- Property that for all n ≥ m, there exists a z in S such that z^n = 1 -/
def property (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, z ∈ S ∧ z^n = 1

/-- Theorem stating that 13 is the smallest positive integer satisfying the property -/
theorem smallest_m_is_13 : 
  (property 13 ∧ ∀ m : ℕ, 0 < m → m < 13 → ¬property m) := by
  sorry

end smallest_m_is_13_l1891_189157


namespace bus_fare_problem_l1891_189110

/-- Represents the denominations of coins available -/
inductive Coin : Type
  | Ten : Coin
  | Fifteen : Coin
  | Twenty : Coin

/-- The value of a coin in kopecks -/
def coinValue : Coin → ℕ
  | Coin.Ten => 10
  | Coin.Fifteen => 15
  | Coin.Twenty => 20

/-- A list of coins -/
def CoinList : Type := List Coin

/-- The total value of a list of coins in kopecks -/
def totalValue (coins : CoinList) : ℕ :=
  coins.foldl (fun acc c => acc + coinValue c) 0

/-- A function that checks if it's possible to distribute coins to passengers -/
def canDistribute (coins : CoinList) (passengers : ℕ) (farePerPassenger : ℕ) : Prop :=
  ∃ (distribution : List CoinList),
    distribution.length = passengers ∧
    (∀ c ∈ distribution, totalValue c = farePerPassenger) ∧
    distribution.join = coins

theorem bus_fare_problem :
  (¬ ∃ (coins : CoinList), coins.length = 24 ∧ canDistribute coins 20 5) ∧
  (∃ (coins : CoinList), coins.length = 25 ∧ canDistribute coins 20 5) := by
  sorry

end bus_fare_problem_l1891_189110


namespace sum_of_binary_digits_345_l1891_189171

/-- The sum of the digits in the binary representation of a natural number -/
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- The theorem stating that the sum of the digits in the binary representation of 345 is 5 -/
theorem sum_of_binary_digits_345 : sum_of_binary_digits 345 = 5 := by
  sorry

end sum_of_binary_digits_345_l1891_189171


namespace math_score_proof_l1891_189122

/-- Represents the scores for each subject --/
structure Scores where
  ethics : ℕ
  korean : ℕ
  science : ℕ
  social : ℕ
  math : ℕ

/-- Calculates the average score --/
def average (s : Scores) : ℚ :=
  (s.ethics + s.korean + s.science + s.social + s.math) / 5

theorem math_score_proof (s : Scores) 
  (h1 : s.ethics = 82)
  (h2 : s.korean = 90)
  (h3 : s.science = 88)
  (h4 : s.social = 84)
  (h5 : average s = 88) :
  s.math = 96 := by
  sorry

#eval average { ethics := 82, korean := 90, science := 88, social := 84, math := 96 }

end math_score_proof_l1891_189122


namespace triangle_side_ratio_l1891_189199

theorem triangle_side_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  let A : ℝ := 2 * Real.pi / 3
  a^2 = 2*b*c + 3*c^2 →
  c / b = 1 / 2 := by sorry

end triangle_side_ratio_l1891_189199


namespace f_3_bounds_l1891_189130

/-- Given a quadratic function f(x) = ax^2 - c with specific constraints on f(1) and f(2),
    prove that f(3) is bounded between -1 and 20. -/
theorem f_3_bounds (a c : ℝ) (h1 : -4 ≤ a - c ∧ a - c ≤ -1) (h2 : -1 ≤ 4*a - c ∧ 4*a - c ≤ 5) :
  -1 ≤ 9*a - c ∧ 9*a - c ≤ 20 := by
  sorry

end f_3_bounds_l1891_189130


namespace max_sum_of_digits_is_24_max_sum_of_digits_is_achievable_l1891_189106

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits possible in a 24-hour format display -/
def maxSumOfDigits : Nat := 24

theorem max_sum_of_digits_is_24 :
  ∀ t : Time24, timeSumOfDigits t ≤ maxSumOfDigits :=
by sorry

theorem max_sum_of_digits_is_achievable :
  ∃ t : Time24, timeSumOfDigits t = maxSumOfDigits :=
by sorry

end max_sum_of_digits_is_24_max_sum_of_digits_is_achievable_l1891_189106


namespace joshua_bottle_caps_l1891_189152

theorem joshua_bottle_caps (initial : ℕ) (final : ℕ) (bought : ℕ) : 
  initial = 40 → final = 47 → final = initial + bought → bought = 7 := by
  sorry

end joshua_bottle_caps_l1891_189152


namespace arithmetic_proof_l1891_189104

theorem arithmetic_proof : 4 * 5 - 3 + 2^3 - 3 * 2 = 19 := by
  sorry

end arithmetic_proof_l1891_189104


namespace courtyard_diagonal_length_l1891_189183

/-- Represents the length of the diagonal of a rectangular courtyard -/
def diagonal_length (side_ratio : ℚ) (paving_cost : ℚ) (cost_per_sqm : ℚ) : ℚ :=
  let longer_side := 4 * (paving_cost / cost_per_sqm / (12 * side_ratio)).sqrt
  let shorter_side := 3 * (paving_cost / cost_per_sqm / (12 * side_ratio)).sqrt
  (longer_side^2 + shorter_side^2).sqrt

/-- Theorem: The diagonal length of the courtyard is 50 meters -/
theorem courtyard_diagonal_length :
  diagonal_length (4/3) 600 0.5 = 50 := by
  sorry

#eval diagonal_length (4/3) 600 0.5

end courtyard_diagonal_length_l1891_189183


namespace number_system_generalization_l1891_189192

-- Define the number systems
inductive NumberSystem
| Natural
| Integer
| Rational
| Real
| Complex

-- Define the basic operations
inductive Operation
| Addition
| Subtraction
| Multiplication
| Division
| SquareRoot

-- Define a function to check if an operation is executable in a given number system
def is_executable (op : Operation) (ns : NumberSystem) : Prop :=
  match op, ns with
  | Operation.Subtraction, NumberSystem.Natural => false
  | Operation.Division, NumberSystem.Integer => false
  | Operation.SquareRoot, NumberSystem.Rational => false
  | Operation.SquareRoot, NumberSystem.Real => false
  | _, _ => true

-- Define the theorem
theorem number_system_generalization (op : Operation) :
  ∃ ns : NumberSystem, is_executable op ns :=
sorry

end number_system_generalization_l1891_189192


namespace candy_bar_profit_l1891_189184

def candy_bars_bought : ℕ := 1500
def buying_price : ℚ := 3 / 8
def selling_price : ℚ := 2 / 3
def booth_setup_cost : ℚ := 50

def total_cost : ℚ := candy_bars_bought * buying_price
def total_revenue : ℚ := candy_bars_bought * selling_price
def net_profit : ℚ := total_revenue - total_cost - booth_setup_cost

theorem candy_bar_profit : net_profit = 387.5 := by sorry

end candy_bar_profit_l1891_189184


namespace line_direction_vector_l1891_189166

/-- Given a line passing through points (-3, 4) and (4, -1) with direction vector (a, a/2), prove a = -10 -/
theorem line_direction_vector (a : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ k * (4 - (-3)) = a ∧ k * (-1 - 4) = a/2) → 
  a = -10 := by
sorry

end line_direction_vector_l1891_189166


namespace lee_cookies_proportion_l1891_189105

/-- Given that Lee can make 24 cookies with 3 cups of flour, 
    this theorem proves he can make 40 cookies with 5 cups of flour, 
    assuming a proportional relationship between flour and cookies. -/
theorem lee_cookies_proportion (flour_cups : ℚ) (cookies : ℕ) 
  (h1 : flour_cups > 0)
  (h2 : cookies > 0)
  (h3 : flour_cups / 3 = cookies / 24) :
  5 * cookies / flour_cups = 40 := by
  sorry

end lee_cookies_proportion_l1891_189105


namespace abc_range_l1891_189117

theorem abc_range (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_sum : a + b + c = 1) (h_sum_squares : a^2 + b^2 + c^2 = 3) :
  -1 < a * b * c ∧ a * b * c < 5/27 := by
  sorry

end abc_range_l1891_189117


namespace square_sum_product_equality_l1891_189146

theorem square_sum_product_equality : 
  (2^2 + 92 * 3^2) * (4^2 + 92 * 5^2) = 1388^2 + 92 * 2^2 := by
  sorry

end square_sum_product_equality_l1891_189146


namespace action_figures_ratio_l1891_189138

theorem action_figures_ratio (initial : ℕ) (sold : ℕ) (remaining : ℕ) : 
  initial = 24 →
  remaining = initial - sold →
  12 = remaining - remaining / 3 →
  (sold : ℚ) / initial = 1 / 4 := by
  sorry

end action_figures_ratio_l1891_189138


namespace paula_bumper_car_rides_l1891_189111

/-- Calculates the number of bumper car rides Paula can take given the total tickets,
    go-kart ticket cost, and bumper car ticket cost. -/
def bumper_car_rides (total_tickets go_kart_cost bumper_car_cost : ℕ) : ℕ :=
  (total_tickets - go_kart_cost) / bumper_car_cost

/-- Proves that Paula can ride the bumper cars 4 times given the conditions. -/
theorem paula_bumper_car_rides :
  let total_tickets : ℕ := 24
  let go_kart_cost : ℕ := 4
  let bumper_car_cost : ℕ := 5
  bumper_car_rides total_tickets go_kart_cost bumper_car_cost = 4 := by
  sorry


end paula_bumper_car_rides_l1891_189111


namespace person_age_puzzle_l1891_189168

theorem person_age_puzzle : ∃ (A : ℕ), 4 * (A + 3) - 4 * (A - 3) = A ∧ A = 24 := by
  sorry

end person_age_puzzle_l1891_189168


namespace josh_marbles_l1891_189151

/-- The number of marbles Josh lost -/
def marbles_lost : ℕ := sorry

/-- The number of marbles Josh initially had -/
def initial_marbles : ℕ := 7

/-- The number of new marbles Josh found -/
def new_marbles : ℕ := 10

/-- The difference between marbles found and marbles lost -/
def difference : ℕ := 2

theorem josh_marbles : marbles_lost = 8 :=
by
  sorry

end josh_marbles_l1891_189151


namespace cos_sum_seventh_roots_unity_l1891_189102

theorem cos_sum_seventh_roots_unity : 
  Real.cos (2 * π / 7) + Real.cos (4 * π / 7) + Real.cos (6 * π / 7) = -1/2 := by
  sorry

end cos_sum_seventh_roots_unity_l1891_189102


namespace strawberry_theft_l1891_189133

/-- Calculates the number of stolen strawberries given the daily harvest rate, 
    number of days, strawberries given away, and final count. -/
def stolen_strawberries (daily_harvest : ℕ) (days : ℕ) (given_away : ℕ) (final_count : ℕ) : ℕ :=
  daily_harvest * days - given_away - final_count

/-- Proves that the number of stolen strawberries is 30 given the specific conditions. -/
theorem strawberry_theft : 
  stolen_strawberries 5 30 20 100 = 30 := by
  sorry

end strawberry_theft_l1891_189133


namespace concentric_circles_theorem_l1891_189189

/-- Given two concentric circles where the area between them is equal to twice the area of the smaller circle -/
theorem concentric_circles_theorem (a b : ℝ) (h : a > 0) (h' : b > 0) (h_concentric : a < b)
  (h_area : π * b^2 - π * a^2 = 2 * π * a^2) :
  (a / b = 1 / Real.sqrt 3) ∧ (π * a^2 / (π * b^2) = 1 / 3) := by
  sorry

end concentric_circles_theorem_l1891_189189


namespace rocky_total_miles_l1891_189191

/-- Rocky's training schedule for the first three days -/
def rocky_training : Fin 3 → ℕ
| 0 => 4  -- Day 1: 4 miles
| 1 => 4 * 2  -- Day 2: Double day 1
| 2 => 4 * 2 * 3  -- Day 3: Triple day 2

/-- The total miles Rocky ran in the first three days of training -/
theorem rocky_total_miles :
  (Finset.sum Finset.univ rocky_training) = 36 := by
  sorry

end rocky_total_miles_l1891_189191


namespace quadratic_equation_solution_l1891_189169

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = (1 : ℝ) / 3 ∧ x₂ = (3 : ℝ) / 2 ∧ 
  (∀ x : ℝ, -6 * x^2 + 11 * x - 3 = 0 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end quadratic_equation_solution_l1891_189169


namespace simplify_expression_l1891_189124

theorem simplify_expression (x y : ℚ) (hx : x = 5) (hy : y = 2) :
  (10 * x^2 * y^3) / (15 * x * y^2) = 20 / 3 := by
  sorry

end simplify_expression_l1891_189124


namespace eight_bead_bracelet_arrangements_l1891_189148

/-- The number of unique arrangements of n distinct beads on a bracelet, 
    considering only rotational symmetry -/
def bracelet_arrangements (n : ℕ) : ℕ := Nat.factorial n / n

/-- Theorem: The number of unique arrangements of 8 distinct beads on a bracelet, 
    considering only rotational symmetry, is 5040 -/
theorem eight_bead_bracelet_arrangements : 
  bracelet_arrangements 8 = 5040 := by
  sorry

end eight_bead_bracelet_arrangements_l1891_189148


namespace person_B_processes_8_components_per_hour_l1891_189121

/-- The number of components processed per hour by person B -/
def components_per_hour_B : ℕ := sorry

/-- The number of components processed per hour by person A -/
def components_per_hour_A : ℕ := components_per_hour_B + 2

/-- The time it takes for person A to process 25 components -/
def time_A : ℚ := 25 / components_per_hour_A

/-- The time it takes for person B to process 20 components -/
def time_B : ℚ := 20 / components_per_hour_B

/-- Theorem stating that person B processes 8 components per hour -/
theorem person_B_processes_8_components_per_hour :
  components_per_hour_B = 8 ∧ time_A = time_B := by sorry

end person_B_processes_8_components_per_hour_l1891_189121


namespace geometric_sequence_a10_l1891_189129

def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a10 (a : ℕ → ℤ) (q : ℤ) :
  is_geometric_sequence a →
  (∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q) →
  a 4 * a 7 = -512 →
  a 3 + a 8 = 124 →
  a 10 = 512 := by
  sorry

end geometric_sequence_a10_l1891_189129


namespace y_coordinate_of_C_l1891_189162

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Checks if a quadrilateral has a vertical line of symmetry -/
def hasVerticalSymmetry (q : Quadrilateral) : Prop := sorry

theorem y_coordinate_of_C (q : Quadrilateral) :
  q.A = ⟨0, 0⟩ →
  q.B = ⟨0, 1⟩ →
  q.D = ⟨3, 1⟩ →
  q.C.x = q.B.x →
  hasVerticalSymmetry q →
  area q = 18 →
  q.C.y = 11 := by sorry

end y_coordinate_of_C_l1891_189162


namespace min_distance_parabola_point_l1891_189135

/-- The minimum value of |y| + |PQ| for a point P(x, y) on the parabola x² = -4y and Q(-2√2, 0) -/
theorem min_distance_parabola_point : 
  let Q : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
  ∃ (min : ℝ), min = 2 ∧ 
    ∀ (P : ℝ × ℝ), (P.1 ^ 2 = -4 * P.2) → 
      abs P.2 + Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) ≥ min :=
by sorry

end min_distance_parabola_point_l1891_189135


namespace partner_q_investment_time_l1891_189150

/-- Represents the investment and profit information for a business partnership --/
structure Partnership where
  investmentRatio : Fin 3 → ℚ
  profitRatio : Fin 3 → ℚ
  investmentTime : Fin 3 → ℚ

/-- Theorem stating the investment time for partner Q given the conditions --/
theorem partner_q_investment_time (p : Partnership) :
  p.investmentRatio 0 = 3 ∧
  p.investmentRatio 1 = 4 ∧
  p.investmentRatio 2 = 5 ∧
  p.profitRatio 0 = 9 ∧
  p.profitRatio 1 = 16 ∧
  p.profitRatio 2 = 25 ∧
  p.investmentTime 0 = 4 ∧
  p.investmentTime 2 = 10 →
  p.investmentTime 1 = 8 :=
by sorry

end partner_q_investment_time_l1891_189150


namespace money_value_difference_l1891_189123

/-- Proves that the percentage difference between Etienne's and Diana's money is -12.5% --/
theorem money_value_difference (exchange_rate : ℝ) (diana_dollars : ℝ) (etienne_euros : ℝ) :
  exchange_rate = 1.5 →
  diana_dollars = 600 →
  etienne_euros = 350 →
  ((diana_dollars - etienne_euros * exchange_rate) / diana_dollars) * 100 = 12.5 := by
  sorry

end money_value_difference_l1891_189123


namespace division_of_power_sixteen_l1891_189155

theorem division_of_power_sixteen (m : ℕ) : m = 16^2024 → m/8 = 8 * 16^2020 := by
  sorry

end division_of_power_sixteen_l1891_189155


namespace time_to_fill_leaking_pool_l1891_189194

/-- Time to fill a leaking pool -/
theorem time_to_fill_leaking_pool 
  (pool_capacity : ℝ) 
  (filling_rate : ℝ) 
  (leaking_rate : ℝ) 
  (h1 : pool_capacity = 60) 
  (h2 : filling_rate = 1.6) 
  (h3 : leaking_rate = 0.1) : 
  pool_capacity / (filling_rate - leaking_rate) = 40 := by
sorry

end time_to_fill_leaking_pool_l1891_189194


namespace llama_to_goat_ratio_l1891_189159

def goat_cost : ℕ := 400
def num_goats : ℕ := 3
def total_spent : ℕ := 4800

def llama_cost : ℕ := goat_cost + goat_cost / 2

def num_llamas : ℕ := (total_spent - num_goats * goat_cost) / llama_cost

theorem llama_to_goat_ratio :
  num_llamas * 1 = num_goats * 2 :=
by sorry

end llama_to_goat_ratio_l1891_189159


namespace polynomial_identity_l1891_189154

theorem polynomial_identity (P : ℝ → ℝ) 
  (h1 : P 0 = 0) 
  (h2 : ∀ x : ℝ, P (x^2 + 1) = (P x)^2 + 1) : 
  ∀ x : ℝ, P x = x := by
  sorry

end polynomial_identity_l1891_189154


namespace great_grandchildren_count_l1891_189140

theorem great_grandchildren_count (age : ℕ) (grandchildren : ℕ) (n : ℕ) 
  (h1 : age = 91)
  (h2 : grandchildren = 11)
  (h3 : grandchildren * n * age = n * 1000 + n) :
  n = 1 := by
  sorry

end great_grandchildren_count_l1891_189140


namespace lecture_average_minutes_heard_l1891_189196

/-- Calculates the average number of minutes heard in a lecture --/
theorem lecture_average_minutes_heard 
  (total_duration : ℝ) 
  (total_attendees : ℕ) 
  (full_lecture_percent : ℝ) 
  (missed_lecture_percent : ℝ) 
  (half_lecture_percent : ℝ) 
  (h1 : total_duration = 90)
  (h2 : total_attendees = 200)
  (h3 : full_lecture_percent = 0.3)
  (h4 : missed_lecture_percent = 0.2)
  (h5 : half_lecture_percent = 0.4 * (1 - full_lecture_percent - missed_lecture_percent))
  (h6 : full_lecture_percent + missed_lecture_percent + half_lecture_percent + 
        (1 - full_lecture_percent - missed_lecture_percent - half_lecture_percent) = 1) :
  (full_lecture_percent * total_duration * total_attendees + 
   0 * missed_lecture_percent * total_attendees + 
   (total_duration / 2) * half_lecture_percent * total_attendees + 
   (3 * total_duration / 4) * (1 - full_lecture_percent - missed_lecture_percent - half_lecture_percent) * total_attendees) / 
   total_attendees = 56.25 := by
sorry

end lecture_average_minutes_heard_l1891_189196


namespace diamond_equation_solution_l1891_189116

/-- Diamond operation -/
def diamond (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

/-- Theorem stating the unique solution to A ◊ 7 = 76 -/
theorem diamond_equation_solution :
  ∃! A : ℝ, diamond A 7 = 76 ∧ A = 12 := by
sorry

end diamond_equation_solution_l1891_189116


namespace cosine_equality_l1891_189172

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → 
  Real.cos (n * π / 180) = Real.cos (820 * π / 180) → 
  n = 100 := by
sorry

end cosine_equality_l1891_189172


namespace solution_set_equivalence_l1891_189174

/-- The solution set of the system of equations {x - 2y = 1, x^3 - 6xy - 8y^3 = 1} 
    is equivalent to the line y = (x-1)/2 -/
theorem solution_set_equivalence (x y : ℝ) : 
  (x - 2*y = 1 ∧ x^3 - 6*x*y - 8*y^3 = 1) ↔ y = (x - 1) / 2 := by
  sorry

#check solution_set_equivalence

end solution_set_equivalence_l1891_189174


namespace problem_statement_l1891_189149

theorem problem_statement : (-0.125)^2003 * (-8)^2004 = -8 := by
  sorry

end problem_statement_l1891_189149


namespace inequality_proof_l1891_189114

theorem inequality_proof (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
  a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y) := by
  sorry

end inequality_proof_l1891_189114


namespace completing_square_quadratic_l1891_189156

theorem completing_square_quadratic (x : ℝ) :
  (x^2 - 4*x - 1 = 0) ↔ ((x - 2)^2 = 5) :=
by sorry

end completing_square_quadratic_l1891_189156


namespace hall_mat_expenditure_l1891_189100

/-- Calculates the total expenditure for covering a rectangular floor with mat. -/
def total_expenditure (length width cost_per_sqm : ℝ) : ℝ :=
  length * width * cost_per_sqm

/-- Proves that the total expenditure for covering a 20m × 15m floor with mat at Rs. 50 per square meter is Rs. 15,000. -/
theorem hall_mat_expenditure :
  total_expenditure 20 15 50 = 15000 := by
  sorry

end hall_mat_expenditure_l1891_189100


namespace unit_digit_of_product_l1891_189108

-- Define the numbers
def a : ℕ := 7858
def b : ℕ := 1086
def c : ℕ := 4582
def d : ℕ := 9783

-- Define the product
def product : ℕ := a * b * c * d

-- Theorem statement
theorem unit_digit_of_product : product % 10 = 8 := by
  sorry

end unit_digit_of_product_l1891_189108


namespace triangle_angle_bisector_theorem_l1891_189125

/-- In a triangle ABC with ∠C = 120°, given sides a and b, and angle bisector lc,
    the equation 1/a + 1/b = 1/lc holds. -/
theorem triangle_angle_bisector_theorem (a b lc : ℝ) (ha : a > 0) (hb : b > 0) (hlc : lc > 0) :
  let angle_C : ℝ := 120 * Real.pi / 180
  1 / a + 1 / b = 1 / lc :=
by sorry

end triangle_angle_bisector_theorem_l1891_189125


namespace sufficient_not_necessary_l1891_189142

theorem sufficient_not_necessary (a b : ℝ) :
  (((a - b) * a^2 < 0 → a < b) ∧
   ∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0) :=
by sorry

end sufficient_not_necessary_l1891_189142


namespace equation_roots_count_l1891_189176

-- Define the function f
def f (x : ℝ) : ℝ := |x| - 1

-- Define the iterative composition of f
def f_n (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => id
  | n + 1 => f ∘ (f_n n)

-- Theorem statement
theorem equation_roots_count :
  ∃! (roots : Finset ℝ), (∀ x ∈ roots, f_n 10 x = -1/2) ∧ (Finset.card roots = 20) := by
  sorry

end equation_roots_count_l1891_189176


namespace wage_decrease_hours_increase_l1891_189145

theorem wage_decrease_hours_increase 
  (original_wage : ℝ) 
  (original_hours : ℝ) 
  (wage_decrease_percent : ℝ) 
  (new_hours : ℝ) 
  (h1 : wage_decrease_percent = 20) 
  (h2 : original_wage > 0) 
  (h3 : original_hours > 0) 
  (h4 : new_hours > 0) 
  (h5 : original_wage * original_hours = (original_wage * (1 - wage_decrease_percent / 100)) * new_hours) :
  (new_hours - original_hours) / original_hours * 100 = 25 := by
sorry

end wage_decrease_hours_increase_l1891_189145


namespace probability_of_all_successes_l1891_189139

-- Define the number of trials
def n : ℕ := 7

-- Define the probability of success in each trial
def p : ℚ := 2/7

-- Define the number of successes we're interested in
def k : ℕ := 7

-- State the theorem
theorem probability_of_all_successes :
  (n.choose k) * p^k * (1 - p)^(n - k) = 128/823543 := by
  sorry

end probability_of_all_successes_l1891_189139


namespace bumper_car_line_joiners_l1891_189173

/-- The number of new people who joined a line for bumper cars at a fair -/
theorem bumper_car_line_joiners (initial : ℕ) (left : ℕ) (final : ℕ) : 
  initial = 12 → left = 10 → final = 17 → final - (initial - left) = 15 := by
  sorry

end bumper_car_line_joiners_l1891_189173


namespace hadley_walk_distance_l1891_189118

/-- The total distance Hadley walked in his boots -/
def total_distance (grocery_store_distance pet_store_distance home_distance : ℕ) : ℕ :=
  grocery_store_distance + pet_store_distance + home_distance

/-- Theorem stating the total distance Hadley walked -/
theorem hadley_walk_distance :
  ∃ (grocery_store_distance pet_store_distance home_distance : ℕ),
    grocery_store_distance = 2 ∧
    pet_store_distance = 2 - 1 ∧
    home_distance = 4 - 1 ∧
    total_distance grocery_store_distance pet_store_distance home_distance = 6 := by
  sorry

end hadley_walk_distance_l1891_189118


namespace second_game_points_l1891_189180

/-- The number of points scored in each of the four games -/
structure GamePoints where
  game1 : ℕ
  game2 : ℕ
  game3 : ℕ
  game4 : ℕ

/-- The conditions of the basketball game scenario -/
def basketball_scenario (p : GamePoints) : Prop :=
  p.game1 = 10 ∧
  p.game3 = 6 ∧
  p.game4 = (p.game1 + p.game2 + p.game3) / 3 ∧
  p.game1 + p.game2 + p.game3 + p.game4 = 40

theorem second_game_points :
  ∃ p : GamePoints, basketball_scenario p ∧ p.game2 = 14 := by
  sorry

end second_game_points_l1891_189180


namespace base7_to_base9_conversion_l1891_189127

/-- Converts a number from base 7 to base 10 --/
def base7To10 (n : Nat) : Nat :=
  (n % 10) + 7 * ((n / 10) % 10) + 49 * (n / 100)

/-- Converts a number from base 10 to base 9 --/
def base10To9 (n : Nat) : Nat :=
  if n < 9 then n
  else (n % 9) + 10 * (base10To9 (n / 9))

theorem base7_to_base9_conversion :
  base10To9 (base7To10 536) = 332 :=
sorry

end base7_to_base9_conversion_l1891_189127


namespace cos_two_alpha_value_l1891_189181

theorem cos_two_alpha_value (α : Real) (h : Real.tan (α + π/4) = 1/3) : 
  Real.cos (2*α) = 3/5 := by sorry

end cos_two_alpha_value_l1891_189181


namespace derivative_at_pi_third_l1891_189131

theorem derivative_at_pi_third (f : ℝ → ℝ) (h : ∀ x, f x = x + Real.sin x) :
  deriv f (π / 3) = 3 / 2 := by
  sorry

end derivative_at_pi_third_l1891_189131


namespace ring_weights_sum_to_total_l1891_189143

/-- The weight of the orange ring in ounces -/
def orange_weight : ℚ := 0.08333333333333333

/-- The weight of the purple ring in ounces -/
def purple_weight : ℚ := 0.3333333333333333

/-- The weight of the white ring in ounces -/
def white_weight : ℚ := 0.4166666666666667

/-- The total weight of all rings in ounces -/
def total_weight : ℚ := 0.8333333333333333

/-- Theorem stating that the sum of individual ring weights equals the total weight -/
theorem ring_weights_sum_to_total : 
  orange_weight + purple_weight + white_weight = total_weight := by
  sorry

end ring_weights_sum_to_total_l1891_189143


namespace min_distinct_values_l1891_189119

/-- Given a list of 3000 positive integers with a unique mode occurring exactly 12 times,
    the minimum number of distinct values in the list is 273. -/
theorem min_distinct_values (L : List ℕ+) : 
  L.length = 3000 →
  ∃! m : ℕ+, (L.count m = 12 ∧ ∀ n : ℕ+, L.count n ≤ L.count m) →
  L.toFinset.card ≥ 273 :=
by sorry

end min_distinct_values_l1891_189119


namespace sum_of_first_40_digits_eq_72_l1891_189113

/-- The sum of the first 40 digits after the decimal point in the decimal representation of 1/2222 -/
def sum_of_first_40_digits : ℕ :=
  -- Define the sum here
  72

/-- Theorem stating that the sum of the first 40 digits after the decimal point
    in the decimal representation of 1/2222 is equal to 72 -/
theorem sum_of_first_40_digits_eq_72 :
  sum_of_first_40_digits = 72 := by
  -- Proof goes here
  sorry

end sum_of_first_40_digits_eq_72_l1891_189113


namespace angle_value_l1891_189170

theorem angle_value (θ : Real) (h : Real.tan θ = 2) : Real.sin (2 * θ + Real.pi / 2) = -3/5 := by
  sorry

end angle_value_l1891_189170


namespace function_composition_l1891_189107

theorem function_composition (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x^2 - 2*x + 5) :
  ∀ x : ℝ, f (x - 1) = x^2 - 4*x + 8 := by
sorry

end function_composition_l1891_189107


namespace three_digit_number_problem_l1891_189187

/-- 
Given a 3-digit number represented by its digits x, y, and z,
if four times the number equals 1464 and the sum of its digits is 15,
then the number is 366.
-/
theorem three_digit_number_problem (x y z : ℕ) : 
  x < 10 → y < 10 → z < 10 →
  4 * (100 * x + 10 * y + z) = 1464 →
  x + y + z = 15 →
  100 * x + 10 * y + z = 366 := by
  sorry

#check three_digit_number_problem

end three_digit_number_problem_l1891_189187


namespace dual_expression_problem_l1891_189153

theorem dual_expression_problem (x : ℝ) :
  (Real.sqrt (20 - x) + Real.sqrt (4 - x) = 8) →
  (Real.sqrt (20 - x) - Real.sqrt (4 - x) = 2) ∧ (x = -5) := by
  sorry


end dual_expression_problem_l1891_189153


namespace negative_two_a_cubed_l1891_189132

theorem negative_two_a_cubed (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end negative_two_a_cubed_l1891_189132


namespace modulus_of_z_l1891_189103

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z as (1+i)^2
def z : ℂ := (1 + i)^2

-- Theorem stating that the modulus of z is 2
theorem modulus_of_z : Complex.abs z = 2 := by
  sorry

end modulus_of_z_l1891_189103


namespace p_and_not_q_l1891_189195

-- Define proposition p
def p : Prop := ∀ a : ℝ, a > 1 → a^2 > a

-- Define proposition q
def q : Prop := ∀ a : ℝ, a > 0 → a > 1/a

-- Theorem to prove
theorem p_and_not_q : p ∧ ¬q := by
  sorry

end p_and_not_q_l1891_189195


namespace square_greater_than_linear_for_less_than_negative_one_l1891_189197

theorem square_greater_than_linear_for_less_than_negative_one (x : ℝ) :
  x < -1 → x^2 > 1 + x := by
  sorry

end square_greater_than_linear_for_less_than_negative_one_l1891_189197


namespace measure_45_seconds_l1891_189147

/-- Represents a fuse that can be lit from either end -/
structure Fuse :=
  (burn_time : ℝ)
  (is_uniform : Bool)

/-- Represents the state of burning a fuse -/
inductive BurnState
  | Unlit
  | LitOneEnd
  | LitBothEnds

/-- Represents the result of burning fuses -/
structure BurnResult :=
  (time : ℝ)
  (fuse1 : BurnState)
  (fuse2 : BurnState)

/-- Function to simulate burning fuses -/
def burn_fuses (f1 f2 : Fuse) : BurnResult :=
  sorry

theorem measure_45_seconds (f1 f2 : Fuse) 
  (h1 : f1.burn_time = 60)
  (h2 : f2.burn_time = 60) :
  ∃ (result : BurnResult), result.time = 45 :=
sorry

end measure_45_seconds_l1891_189147


namespace identify_fake_coin_in_two_weighings_l1891_189144

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a coin -/
inductive Coin
  | A : Coin
  | B : Coin
  | C : Coin
  | D : Coin

/-- Represents a weighing operation -/
def weigh (left right : List Coin) : WeighResult :=
  sorry

/-- Represents the process of identifying the fake coin -/
def identifyFakeCoin : Coin :=
  sorry

/-- Theorem stating that it's possible to identify the fake coin in two weighings -/
theorem identify_fake_coin_in_two_weighings :
  ∃ (fakeCoin : Coin),
    (∀ (c : Coin), c ≠ fakeCoin → (weigh [c] [fakeCoin] = WeighResult.Equal ↔ c = fakeCoin)) →
    identifyFakeCoin = fakeCoin :=
  sorry

end identify_fake_coin_in_two_weighings_l1891_189144


namespace estimate_fish_population_l1891_189163

/-- Estimates the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population
  (initial_marked : ℕ)
  (second_catch : ℕ)
  (marked_in_second : ℕ)
  (initial_marked_pos : 0 < initial_marked)
  (second_catch_pos : 0 < second_catch)
  (marked_in_second_pos : 0 < marked_in_second)
  (marked_in_second_le_second_catch : marked_in_second ≤ second_catch)
  (marked_in_second_le_initial_marked : marked_in_second ≤ initial_marked) :
  (initial_marked * second_catch) / marked_in_second = 1500 :=
sorry

#eval (40 * 300) / 8

end estimate_fish_population_l1891_189163


namespace expected_weekly_rainfall_is_20_point_5_l1891_189128

/-- Represents the possible daily rainfall amounts in inches -/
inductive DailyRainfall
  | NoRain
  | LightRain
  | HeavyRain

/-- The probability of each rainfall outcome -/
def rainProbability : DailyRainfall → ℝ
  | DailyRainfall.NoRain => 0.3
  | DailyRainfall.LightRain => 0.3
  | DailyRainfall.HeavyRain => 0.4

/-- The amount of rainfall for each outcome in inches -/
def rainAmount : DailyRainfall → ℝ
  | DailyRainfall.NoRain => 0
  | DailyRainfall.LightRain => 3
  | DailyRainfall.HeavyRain => 8

/-- The number of days in the week -/
def daysInWeek : ℕ := 5

/-- The expected total rainfall for the week -/
def expectedWeeklyRainfall : ℝ :=
  daysInWeek * (rainProbability DailyRainfall.NoRain * rainAmount DailyRainfall.NoRain +
                rainProbability DailyRainfall.LightRain * rainAmount DailyRainfall.LightRain +
                rainProbability DailyRainfall.HeavyRain * rainAmount DailyRainfall.HeavyRain)

/-- Theorem: The expected total rainfall for the week is 20.5 inches -/
theorem expected_weekly_rainfall_is_20_point_5 :
  expectedWeeklyRainfall = 20.5 := by sorry

end expected_weekly_rainfall_is_20_point_5_l1891_189128
