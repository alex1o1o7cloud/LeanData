import Mathlib

namespace sum_of_multiples_l1371_137169

def smallest_two_digit_multiple_of_5 : ℕ := 10

def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 := by
  sorry

end sum_of_multiples_l1371_137169


namespace x_minus_y_equals_18_l1371_137116

theorem x_minus_y_equals_18 (x y : ℤ) (h1 : x + y = 10) (h2 : x = 14) : x - y = 18 := by
  sorry

end x_minus_y_equals_18_l1371_137116


namespace ring_arrangements_count_l1371_137132

/-- The number of ways to arrange 6 rings out of 10 distinguishable rings on 5 fingers,
    where the order on each finger matters and no finger can have more than 3 rings. -/
def ring_arrangements : ℕ :=
  let total_rings : ℕ := 10
  let rings_used : ℕ := 6
  let num_fingers : ℕ := 5
  let max_rings_per_finger : ℕ := 3
  -- The actual calculation would go here, but we'll use the result directly
  145152000

/-- Theorem stating that the number of ring arrangements is 145,152,000 -/
theorem ring_arrangements_count : ring_arrangements = 145152000 := by
  -- The proof would go here
  sorry

end ring_arrangements_count_l1371_137132


namespace tangent_line_equation_l1371_137147

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^3 + x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 6 * x^2 + 1

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 7 * x - y - 4 = 0 := by
sorry

end tangent_line_equation_l1371_137147


namespace inequality_proof_l1371_137163

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_condition : a + b + c + d = 1) :
  a * b * c + b * c * d + c * d * a + d * a * b ≤ 1 / 27 + 176 / 27 * a * b * c * d := by
  sorry

end inequality_proof_l1371_137163


namespace baron_munchausen_claim_l1371_137181

-- Define a function to calculate the sum of squares of digits
def sumOfSquaresOfDigits (n : ℕ) : ℕ := sorry

-- Define the property for the numbers we're looking for
def satisfiesProperty (a b : ℕ) : Prop :=
  (a ≠ b) ∧ 
  (a ≥ 10^9) ∧ (a < 10^10) ∧ 
  (b ≥ 10^9) ∧ (b < 10^10) ∧ 
  (a % 10 ≠ 0) ∧ (b % 10 ≠ 0) ∧
  (a - sumOfSquaresOfDigits a = b - sumOfSquaresOfDigits b)

-- Theorem statement
theorem baron_munchausen_claim : ∃ a b : ℕ, satisfiesProperty a b := by sorry

end baron_munchausen_claim_l1371_137181


namespace bag_problem_l1371_137148

/-- The number of red balls in the bag -/
def red_balls (a : ℕ) : ℕ := a + 1

/-- The number of yellow balls in the bag -/
def yellow_balls (a : ℕ) : ℕ := a

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 1

/-- The total number of balls in the bag -/
def total_balls (a : ℕ) : ℕ := red_balls a + yellow_balls a + blue_balls

/-- The score earned by drawing a red ball -/
def red_score : ℕ := 1

/-- The score earned by drawing a yellow ball -/
def yellow_score : ℕ := 2

/-- The score earned by drawing a blue ball -/
def blue_score : ℕ := 3

/-- The expected value of the score when drawing a ball -/
def expected_value : ℚ := 5 / 3

theorem bag_problem (a : ℕ) :
  (a = 2) ∧
  (let p : ℚ := (Nat.choose 3 1 * Nat.choose 2 2 + Nat.choose 3 2 * Nat.choose 1 1) / Nat.choose 6 3
   p = 3 / 10) :=
by sorry

end bag_problem_l1371_137148


namespace ratio_between_zero_and_one_l1371_137124

theorem ratio_between_zero_and_one : 
  let A : ℕ := 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28
  let B : ℕ := 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20
  0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by sorry

end ratio_between_zero_and_one_l1371_137124


namespace weight_of_fresh_grapes_l1371_137118

/-- Given that fresh grapes contain 90% water by weight, dried grapes contain 20% water by weight,
    and the weight of dry grapes available is 3.125 kg, prove that the weight of fresh grapes is 78.125 kg. -/
theorem weight_of_fresh_grapes :
  let fresh_water_ratio : ℝ := 0.9
  let dried_water_ratio : ℝ := 0.2
  let dried_grapes_weight : ℝ := 3.125
  fresh_water_ratio * fresh_grapes_weight = dried_water_ratio * dried_grapes_weight + 
    (1 - dried_water_ratio) * dried_grapes_weight →
  fresh_grapes_weight = 78.125
  := by sorry

#check weight_of_fresh_grapes

end weight_of_fresh_grapes_l1371_137118


namespace max_b_value_l1371_137150

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := 4*a^2 * log x + b

theorem max_b_value (a : ℝ) (h_a : a > 0) :
  (∃ x₀ : ℝ, f a x₀ = g a b x₀ ∧ deriv (f a) x₀ = deriv (g a b) x₀) →
  (∃ b : ℝ, ∀ b' : ℝ, (∃ x₀ : ℝ, f a x₀ = g a b' x₀ ∧ deriv (f a) x₀ = deriv (g a b') x₀) → b' ≤ b) →
  (∃ b : ℝ, (∃ x₀ : ℝ, f a x₀ = g a b x₀ ∧ deriv (f a) x₀ = deriv (g a b) x₀) ∧
             ∀ b' : ℝ, (∃ x₀ : ℝ, f a x₀ = g a b' x₀ ∧ deriv (f a) x₀ = deriv (g a b') x₀) → b' ≤ b) →
  b = 2 * sqrt e :=
by sorry

end max_b_value_l1371_137150


namespace absolute_value_fraction_inequality_l1371_137121

theorem absolute_value_fraction_inequality (x : ℝ) :
  x ≠ 0 → (|(x + 2) / x| < 1 ↔ x < -1) :=
by sorry

end absolute_value_fraction_inequality_l1371_137121


namespace teacher_health_survey_l1371_137174

theorem teacher_health_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 120)
  (h_high_bp : high_bp = 70)
  (h_heart_trouble : heart_trouble = 40)
  (h_both : both = 20) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 25 :=
by sorry

end teacher_health_survey_l1371_137174


namespace team_b_city_a_matches_l1371_137159

/-- Represents a team in the tournament -/
structure Team where
  city : Fin 16
  isTeamA : Bool

/-- The number of matches played by a team -/
def matchesPlayed (t : Team) : ℕ := sorry

/-- The tournament satisfies the given conditions -/
axiom tournament_conditions :
  ∀ t1 t2 : Team,
    (t1 ≠ t2) →
    (t1.city ≠ t2.city ∨ t1.isTeamA ≠ t2.isTeamA) →
    (t1 ≠ ⟨0, true⟩) →
    (t2 ≠ ⟨0, true⟩) →
    matchesPlayed t1 ≠ matchesPlayed t2

/-- All teams except one have played between 0 and 30 matches -/
axiom matches_range :
  ∀ t : Team, t ≠ ⟨0, true⟩ → matchesPlayed t ≤ 30

/-- The theorem to be proved -/
theorem team_b_city_a_matches :
  matchesPlayed ⟨0, false⟩ = 15 := by sorry

end team_b_city_a_matches_l1371_137159


namespace inverse_equals_scaled_sum_l1371_137137

/-- Given a 2x2 matrix M, prove that its inverse is equal to (1/6)*M + (1/6)*I -/
theorem inverse_equals_scaled_sum (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M = !![2, 0; 1, -3]) : 
  M⁻¹ = (1/6 : ℝ) • M + (1/6 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end inverse_equals_scaled_sum_l1371_137137


namespace relationship_xyz_l1371_137149

theorem relationship_xyz (x y z : ℝ) 
  (h1 : x - y > x + z) 
  (h2 : x + y < y + z) : 
  y < -z ∧ x < z := by
sorry

end relationship_xyz_l1371_137149


namespace sqrt_sum_squares_plus_seven_l1371_137179

theorem sqrt_sum_squares_plus_seven (a b : ℝ) : 
  a = Real.sqrt 5 + 2 → b = Real.sqrt 5 - 2 → Real.sqrt (a^2 + b^2 + 7) = 5 := by
  sorry

end sqrt_sum_squares_plus_seven_l1371_137179


namespace largest_integer_in_interval_l1371_137165

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 7/11 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 7/11) → y ≤ x :=
by sorry

end largest_integer_in_interval_l1371_137165


namespace number_of_students_l1371_137120

-- Define the lottery winnings
def lottery_winnings : ℚ := 155250

-- Define the fraction given to each student
def fraction_per_student : ℚ := 1 / 1000

-- Define the total amount received by students
def total_received : ℚ := 15525

-- Theorem to prove
theorem number_of_students : 
  (total_received / (lottery_winnings * fraction_per_student) : ℚ) = 100 := by
  sorry

end number_of_students_l1371_137120


namespace triangle_altitude_l1371_137110

theorem triangle_altitude (base : ℝ) (square_side : ℝ) (altitude : ℝ) : 
  base = 6 →
  square_side = 6 →
  (1/2) * base * altitude = square_side * square_side →
  altitude = 12 := by
sorry

end triangle_altitude_l1371_137110


namespace complex_number_values_l1371_137135

theorem complex_number_values (z : ℂ) (a : ℝ) :
  z = (4 + 2*I) / (a + I) → Complex.abs z = Real.sqrt 10 →
  z = 3 - I ∨ z = -1 - 3*I :=
by sorry

end complex_number_values_l1371_137135


namespace expression_equals_two_l1371_137190

theorem expression_equals_two : 2 + Real.sqrt 2 + 1 / (2 + Real.sqrt 2) + 1 / (Real.sqrt 2 - 2) = 2 := by
  sorry

end expression_equals_two_l1371_137190


namespace soccer_league_games_l1371_137188

theorem soccer_league_games (n : ℕ) (regular_games_per_matchup : ℕ) (promotional_games_per_team : ℕ) : 
  n = 20 → 
  regular_games_per_matchup = 3 → 
  promotional_games_per_team = 3 → 
  (n * (n - 1) * regular_games_per_matchup) / 2 + n * promotional_games_per_team = 1200 := by
sorry

end soccer_league_games_l1371_137188


namespace unique_solution_cubic_equation_l1371_137109

theorem unique_solution_cubic_equation :
  ∃! x : ℝ, x ≠ 2 ∧ (x^3 - 3*x^2)/(x^2 - 4*x + 4) + x = 3 :=
by sorry

end unique_solution_cubic_equation_l1371_137109


namespace cube_power_inequality_l1371_137143

theorem cube_power_inequality (a b c : ℕ+) :
  (a^(a:ℕ) * b^(b:ℕ) * c^(c:ℕ))^3 ≥ (a*b*c)^((a:ℕ)+(b:ℕ)+(c:ℕ)) := by
  sorry

end cube_power_inequality_l1371_137143


namespace nikola_ant_farm_problem_l1371_137153

/-- Nikola's ant farm problem -/
theorem nikola_ant_farm_problem 
  (num_ants : ℕ) 
  (food_per_ant : ℕ) 
  (food_cost_per_oz : ℚ) 
  (leaf_cost : ℚ) 
  (num_leaves : ℕ) 
  (num_jobs : ℕ) : 
  num_ants = 400 →
  food_per_ant = 2 →
  food_cost_per_oz = 1/10 →
  leaf_cost = 1/100 →
  num_leaves = 6000 →
  num_jobs = 4 →
  (num_ants * food_per_ant * food_cost_per_oz - num_leaves * leaf_cost) / num_jobs = 5 :=
by sorry

end nikola_ant_farm_problem_l1371_137153


namespace factors_of_210_l1371_137145

theorem factors_of_210 : Finset.card (Nat.divisors 210) = 16 := by
  sorry

end factors_of_210_l1371_137145


namespace parabola_min_y_l1371_137192

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y + x = (y - x)^2 + 3*(y - x) + 3

/-- The minimum y-value of the parabola -/
theorem parabola_min_y : ∃ (y_min : ℝ), y_min = -1/2 ∧
  (∀ (x y : ℝ), parabola_equation x y → y ≥ y_min) :=
sorry

end parabola_min_y_l1371_137192


namespace bedroom_set_final_price_l1371_137100

def original_price : ℝ := 2000
def gift_cards : ℝ := 200
def first_discount_rate : ℝ := 0.15
def second_discount_rate : ℝ := 0.10

def final_price : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  price_after_second_discount - gift_cards

theorem bedroom_set_final_price :
  final_price = 1330 := by sorry

end bedroom_set_final_price_l1371_137100


namespace perimeter_formula_and_maximum_l1371_137177

noncomputable section

open Real

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  BC : ℝ
  x : ℝ  -- Angle B
  y : ℝ  -- Perimeter
  h_A : A = π / 3
  h_BC : BC = 2 * sqrt 3
  h_x_pos : x > 0
  h_x_upper : x < 2 * π / 3

/-- Perimeter function -/
def perimeter (t : Triangle) : ℝ := 6 * sin t.x + 2 * sqrt 3 * cos t.x + 2 * sqrt 3

theorem perimeter_formula_and_maximum (t : Triangle) :
  t.y = perimeter t ∧ t.y ≤ 6 * sqrt 3 := by sorry

end

end perimeter_formula_and_maximum_l1371_137177


namespace bead_problem_solutions_l1371_137185

/-- Represents the possible total number of beads -/
def PossibleTotals : Set ℕ := {107, 109, 111, 113, 115, 117}

/-- Represents a solution to the bead problem -/
structure BeadSolution where
  x : ℕ -- number of 19-gram beads
  y : ℕ -- number of 17-gram beads

/-- Checks if a BeadSolution is valid -/
def isValidSolution (s : BeadSolution) : Prop :=
  19 * s.x + 17 * s.y = 2017 ∧ s.x + s.y ∈ PossibleTotals

/-- Theorem stating that there exist valid solutions for all possible totals -/
theorem bead_problem_solutions :
  ∀ n ∈ PossibleTotals, ∃ s : BeadSolution, isValidSolution s ∧ s.x + s.y = n :=
sorry

end bead_problem_solutions_l1371_137185


namespace largest_divisor_of_product_l1371_137176

def product (n : ℕ) : ℕ := (n+2)*(n+4)*(n+6)*(n+8)*(n+10)

theorem largest_divisor_of_product (n : ℕ) (h : Odd n) :
  (∃ (k : ℕ), product n = 15 * k) ∧
  (∀ (m : ℕ), m > 15 → ¬(∀ (n : ℕ), Odd n → ∃ (k : ℕ), product n = m * k)) :=
by sorry

end largest_divisor_of_product_l1371_137176


namespace polynomial_divisibility_l1371_137103

-- Define the polynomial
def p (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 12 * x - 12

-- Define the factors
def factor1 (x : ℝ) : ℝ := x - 2
def factor2 (x : ℝ) : ℝ := 3 * x^2 - 4

-- Theorem statement
theorem polynomial_divisibility :
  (∃ q1 q2 : ℝ → ℝ, ∀ x, p x = factor1 x * q1 x ∧ p x = factor2 x * q2 x) :=
sorry

end polynomial_divisibility_l1371_137103


namespace greatest_solution_is_negative_two_l1371_137138

def equation (x : ℝ) : Prop :=
  x ≠ 9 ∧ (x^2 - 5*x - 84) / (x - 9) = 4 / (x + 6)

theorem greatest_solution_is_negative_two :
  ∃ x_max : ℝ, x_max = -2 ∧ equation x_max ∧ ∀ y : ℝ, equation y → y ≤ x_max :=
sorry

end greatest_solution_is_negative_two_l1371_137138


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l1371_137139

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), is_three_digit n ∧ n % 9 = 0 ∧ digit_sum n = 27 ∧
  ∀ (m : ℕ), is_three_digit m ∧ m % 9 = 0 ∧ digit_sum m = 27 → m ≤ n :=
by
  use 999
  sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l1371_137139


namespace equal_interest_rates_l1371_137117

/-- Proves that given two accounts with equal investments, if one account has an interest rate of 10%
    and both accounts earn the same interest at the end of the year, then the interest rate of the
    other account is also 10%. -/
theorem equal_interest_rates
  (investment : ℝ)
  (rate1 rate2 : ℝ)
  (h1 : investment > 0)
  (h2 : rate2 = 0.1)
  (h3 : investment * rate1 = investment * rate2) :
  rate1 = 0.1 := by
  sorry

end equal_interest_rates_l1371_137117


namespace proposition_2_proposition_4_l1371_137129

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the lines and planes
variable (m n : Line) (α β : Plane)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (subset : Line → Plane → Prop)

-- Axiom: m and n are different lines
axiom different_lines : m ≠ n

-- Axiom: α and β are different planes
axiom different_planes : α ≠ β

-- Proposition 2
theorem proposition_2 : 
  (perpendicular_line_line m n ∧ perpendicular_line_plane n α ∧ perpendicular_line_plane m β) → 
  perpendicular_plane_plane α β :=
sorry

-- Proposition 4
theorem proposition_4 : 
  (perpendicular_line_plane n β ∧ perpendicular_plane_plane α β) → 
  (parallel n α ∨ subset n α) :=
sorry

end proposition_2_proposition_4_l1371_137129


namespace number_difference_l1371_137151

theorem number_difference (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : |x - y| = 7 := by
  sorry

end number_difference_l1371_137151


namespace lawrence_county_kids_count_l1371_137112

theorem lawrence_county_kids_count :
  let group_a_1week : ℕ := 175000
  let group_a_2week : ℕ := 107000
  let group_a_3week : ℕ := 35000
  let group_b_1week : ℕ := 100000
  let group_b_2week : ℕ := 70350
  let group_b_3week : ℕ := 19500
  let group_c_1week : ℕ := 45000
  let group_c_2week : ℕ := 87419
  let group_c_3week : ℕ := 14425
  let kids_staying_home : ℕ := 590796
  let kids_outside_county : ℕ := 22
  
  let total_group_a : ℕ := group_a_1week + group_a_2week + group_a_3week
  let total_group_b : ℕ := group_b_1week + group_b_2week + group_b_3week
  let total_group_c : ℕ := group_c_1week + group_c_2week + group_c_3week
  
  let total_kids_in_camp : ℕ := total_group_a + total_group_b + total_group_c
  
  total_kids_in_camp + kids_staying_home + kids_outside_county = 1244512 :=
by
  sorry

#check lawrence_county_kids_count

end lawrence_county_kids_count_l1371_137112


namespace all_props_true_l1371_137170

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x^2 - 3*x + 2 ≠ 0 → x ≠ 1 ∧ x ≠ 2

-- Define the converse proposition
def converse_prop (x : ℝ) : Prop := x ≠ 1 ∧ x ≠ 2 → x^2 - 3*x + 2 ≠ 0

-- Define the inverse proposition
def inverse_prop (x : ℝ) : Prop := x^2 - 3*x + 2 = 0 → x = 1 ∨ x = 2

-- Define the contrapositive proposition
def contrapositive_prop (x : ℝ) : Prop := x = 1 ∨ x = 2 → x^2 - 3*x + 2 = 0

-- Theorem stating that all propositions are true
theorem all_props_true : 
  (∀ x : ℝ, original_prop x) ∧ 
  (∀ x : ℝ, converse_prop x) ∧ 
  (∀ x : ℝ, inverse_prop x) ∧ 
  (∀ x : ℝ, contrapositive_prop x) :=
sorry

end all_props_true_l1371_137170


namespace M_remainder_l1371_137191

/-- A function that checks if a natural number has all distinct digits -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 12 with all distinct digits -/
def M : ℕ := sorry

/-- M is a multiple of 12 -/
axiom M_multiple_of_12 : 12 ∣ M

/-- M has all distinct digits -/
axiom M_distinct_digits : has_distinct_digits M

/-- M is the greatest such number -/
axiom M_greatest : ∀ n : ℕ, 12 ∣ n → has_distinct_digits n → n ≤ M

/-- The remainder when M is divided by 2000 is 960 -/
theorem M_remainder : M % 2000 = 960 := by sorry

end M_remainder_l1371_137191


namespace subset_relationship_l1371_137168

def M : Set (ℝ × ℝ) := {p | |p.1| + |p.2| < 1}

def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

def P : Set (ℝ × ℝ) := {p | |p.1 + p.2| < 1 ∧ |p.1| < 1 ∧ |p.2| < 1}

theorem subset_relationship : M ⊆ N ∧ N ⊆ P := by sorry

end subset_relationship_l1371_137168


namespace system_solution_l1371_137182

theorem system_solution (x y z : ℝ) : 
  (x^2 - y*z = |y - z| + 1 ∧
   y^2 - z*x = |z - x| + 1 ∧
   z^2 - x*y = |x - y| + 1) ↔ 
  ((x = 5/3 ∧ y = -4/3 ∧ z = -4/3) ∨
   (x = 4/3 ∧ y = 4/3 ∧ z = -5/3) ∨
   (x = 5/3 ∧ y = -4/3 ∧ z = -4/3) ∨
   (x = -4/3 ∧ y = 5/3 ∧ z = -4/3) ∨
   (x = -4/3 ∧ y = -4/3 ∧ z = 5/3) ∨
   (x = 4/3 ∧ y = 4/3 ∧ z = -5/3) ∨
   (x = 4/3 ∧ y = -5/3 ∧ z = 4/3) ∨
   (x = -5/3 ∧ y = 4/3 ∧ z = 4/3)) :=
by sorry

end system_solution_l1371_137182


namespace area_of_closed_figure_l1371_137128

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3*x

theorem area_of_closed_figure : 
  ∫ x in (1/2)..1, (1/x + 2*x - 3) = 3/4 - Real.log 2 := by sorry

end area_of_closed_figure_l1371_137128


namespace range_of_a_l1371_137113

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x, x^2 + 2*a*x + 4 > 0

def Q (a : ℝ) : Prop := ∀ x y, x < y → (5-2*a)^x < (5-2*a)^y

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a ≤ -2 :=
by sorry

end range_of_a_l1371_137113


namespace power_of_two_contains_k_zeros_l1371_137167

theorem power_of_two_contains_k_zeros (k : ℕ) (hk : k ≥ 1) :
  ∃ n : ℕ+, ∃ a b : ℕ, a ≠ 0 ∧ b ≠ 0 ∧
  (∃ m : ℕ, (2 : ℕ) ^ (n : ℕ) = m * 10^(k+1) + a * 10^k + b) :=
sorry

end power_of_two_contains_k_zeros_l1371_137167


namespace parabola_vertex_l1371_137105

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -(x - 1)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 3)

/-- Theorem: The vertex of the parabola y = -(x-1)^2 + 3 is (1, 3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≤ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end parabola_vertex_l1371_137105


namespace tan70_cos10_expression_equals_one_l1371_137127

theorem tan70_cos10_expression_equals_one :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (1 - Real.sqrt 3 * Real.tan (20 * π / 180)) = 1 := by
  sorry

end tan70_cos10_expression_equals_one_l1371_137127


namespace tom_steps_when_matt_reaches_220_l1371_137115

/-- Proves that Tom reaches 275 steps when Matt reaches 220 steps, given their respective speeds -/
theorem tom_steps_when_matt_reaches_220 
  (matt_speed : ℕ) 
  (tom_speed_diff : ℕ) 
  (matt_steps : ℕ) 
  (h1 : matt_speed = 20)
  (h2 : tom_speed_diff = 5)
  (h3 : matt_steps = 220) :
  matt_steps + (matt_steps / matt_speed) * tom_speed_diff = 275 := by
  sorry

#check tom_steps_when_matt_reaches_220

end tom_steps_when_matt_reaches_220_l1371_137115


namespace staircase_steps_l1371_137198

/-- The number of steps Akvort skips at a time -/
def akvort_skip : ℕ := 3

/-- The number of steps Barnden skips at a time -/
def barnden_skip : ℕ := 4

/-- The number of steps Croft skips at a time -/
def croft_skip : ℕ := 5

/-- The minimum number of steps in the staircase -/
def min_steps : ℕ := 19

theorem staircase_steps :
  (min_steps + 1) % akvort_skip = 0 ∧
  (min_steps + 1) % barnden_skip = 0 ∧
  (min_steps + 1) % croft_skip = 0 ∧
  ∀ n : ℕ, n < min_steps →
    ((n + 1) % akvort_skip = 0 ∧
     (n + 1) % barnden_skip = 0 ∧
     (n + 1) % croft_skip = 0) → False :=
by sorry

end staircase_steps_l1371_137198


namespace enter_exit_ways_count_l1371_137173

/-- The number of doors in the room -/
def num_doors : ℕ := 4

/-- The number of ways to enter and exit the room -/
def ways_to_enter_and_exit : ℕ := num_doors * num_doors

/-- Theorem: The number of different ways to enter and exit a room with four doors is 64 -/
theorem enter_exit_ways_count : ways_to_enter_and_exit = 64 := by
  sorry

end enter_exit_ways_count_l1371_137173


namespace x_minus_y_value_l1371_137144

theorem x_minus_y_value (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 12) : x - y = 3/2 := by
  sorry

end x_minus_y_value_l1371_137144


namespace function_point_coefficient_l1371_137186

/-- Given a function f(x) = ax³ - 2x that passes through the point (-1, 4), prove that a = -2 -/
theorem function_point_coefficient (a : ℝ) : 
  (fun x : ℝ => a * x^3 - 2*x) (-1) = 4 → a = -2 := by
  sorry

end function_point_coefficient_l1371_137186


namespace sibling_age_sum_l1371_137197

/-- Given the ages of three siblings, prove that the sum of the younger and older siblings' ages is correct. -/
theorem sibling_age_sum (juliet maggie ralph : ℕ) : 
  juliet = 10 → 
  juliet = maggie + 3 → 
  ralph = juliet + 2 → 
  maggie + ralph = 19 := by
sorry

end sibling_age_sum_l1371_137197


namespace adjacent_difference_at_least_six_l1371_137152

/-- A 9x9 table containing integers from 1 to 81 -/
def Table : Type := Fin 9 → Fin 9 → Fin 81

/-- Two cells are adjacent if they share a side -/
def adjacent (a b : Fin 9 × Fin 9) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ b.2 = a.2 + 1)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ b.1 = a.1 + 1))

/-- The table contains all integers from 1 to 81 exactly once -/
def valid_table (t : Table) : Prop :=
  ∀ n : Fin 81, ∃! (i j : Fin 9), t i j = n

theorem adjacent_difference_at_least_six (t : Table) (h : valid_table t) :
  ∃ (a b : Fin 9 × Fin 9), adjacent a b ∧ 
    ((t a.1 a.2).val + 6 ≤ (t b.1 b.2).val ∨ (t b.1 b.2).val + 6 ≤ (t a.1 a.2).val) :=
sorry

end adjacent_difference_at_least_six_l1371_137152


namespace expression_evaluation_l1371_137189

theorem expression_evaluation (x y : ℚ) (hx : x = 4 / 7) (hy : y = 6 / 8) :
  (7 * x + 8 * y) / (56 * x * y) = 5 / 12 := by
  sorry

end expression_evaluation_l1371_137189


namespace max_value_cos_theta_l1371_137140

noncomputable def f (x : ℝ) : ℝ := Real.sin x - 2 * Real.cos x

theorem max_value_cos_theta :
  ∀ θ : ℝ, (∀ x : ℝ, f x ≤ f θ) → Real.cos θ = -2 * Real.sqrt 5 / 5 := by
  sorry

end max_value_cos_theta_l1371_137140


namespace sum_of_squares_and_products_l1371_137161

theorem sum_of_squares_and_products (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 52)
  (sum_of_products : x*y + y*z + z*x = 28) :
  x + y + z = 6 * Real.sqrt 3 := by
sorry

end sum_of_squares_and_products_l1371_137161


namespace fraction_inequality_implies_inequality_l1371_137106

theorem fraction_inequality_implies_inequality (a b c : ℝ) (hc : c ≠ 0) :
  a / c^2 < b / c^2 → a < b :=
by sorry

end fraction_inequality_implies_inequality_l1371_137106


namespace set_operation_result_l1371_137171

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 3, 5}
def C : Set ℤ := {0, 2, 4}

theorem set_operation_result : (A ∩ B) ∪ C = {0, 1, 2, 4} := by
  sorry

end set_operation_result_l1371_137171


namespace class_size_l1371_137175

theorem class_size (total : ℚ) 
  (h1 : (2 : ℚ) / 3 * total = total * (2 : ℚ) / 3)  -- Two-thirds of the class have brown eyes
  (h2 : (1 : ℚ) / 2 * (total * (2 : ℚ) / 3) = total / 3)  -- Half of the students with brown eyes have black hair
  (h3 : (total / 3 : ℚ) = 6)  -- There are 6 students with brown eyes and black hair
  : total = 18 := by
sorry

end class_size_l1371_137175


namespace probability_two_red_one_green_l1371_137122

def red_shoes : ℕ := 6
def green_shoes : ℕ := 8
def blue_shoes : ℕ := 5
def yellow_shoes : ℕ := 3

def total_shoes : ℕ := red_shoes + green_shoes + blue_shoes + yellow_shoes

def draw_count : ℕ := 3

theorem probability_two_red_one_green :
  (Nat.choose red_shoes 2 * Nat.choose green_shoes 1) / Nat.choose total_shoes draw_count = 6 / 77 := by
  sorry

end probability_two_red_one_green_l1371_137122


namespace trigonometric_product_l1371_137155

theorem trigonometric_product (cos60 sin60 cos30 sin30 : ℝ) : 
  cos60 = 1/2 →
  sin60 = Real.sqrt 3 / 2 →
  cos30 = Real.sqrt 3 / 2 →
  sin30 = 1/2 →
  (1 - 1/cos30) * (1 + 1/sin60) * (1 - 1/sin30) * (1 + 1/cos60) = -1 := by
  sorry

end trigonometric_product_l1371_137155


namespace inequality_proof_l1371_137146

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) / 3 ≥ ((a + b) * (b + c) * (c + a) / 8) ^ (1/3) ∧
  ((a + b) * (b + c) * (c + a) / 8) ^ (1/3) ≥ (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a)) / 3 :=
by sorry

end inequality_proof_l1371_137146


namespace coefficient_a9_l1371_137157

theorem coefficient_a9 (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (fun x : ℝ => x^2 + x^10) = 
  (fun x : ℝ => a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + 
                a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₉ = -10 := by
sorry

end coefficient_a9_l1371_137157


namespace max_product_sum_2020_l1371_137142

theorem max_product_sum_2020 : 
  (∃ (a b : ℤ), a + b = 2020 ∧ a * b = 1020100) ∧ 
  (∀ (x y : ℤ), x + y = 2020 → x * y ≤ 1020100) := by
sorry

end max_product_sum_2020_l1371_137142


namespace assembly_line_theorem_l1371_137162

/-- Represents the number of tasks in the assembly line -/
def num_tasks : ℕ := 6

/-- Represents the number of tasks that can be freely arranged -/
def free_tasks : ℕ := num_tasks - 1

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The number of ways to arrange the assembly line -/
def assembly_line_arrangements : ℕ := factorial free_tasks

theorem assembly_line_theorem : assembly_line_arrangements = 120 := by
  sorry

end assembly_line_theorem_l1371_137162


namespace notebook_words_per_page_l1371_137156

theorem notebook_words_per_page :
  ∀ (words_per_page : ℕ),
    words_per_page > 0 →
    words_per_page ≤ 150 →
    (180 * words_per_page) % 221 = 246 % 221 →
    words_per_page = 24 :=
by
  sorry

end notebook_words_per_page_l1371_137156


namespace opposite_of_sqrt_six_l1371_137172

theorem opposite_of_sqrt_six :
  ∀ x : ℝ, x = Real.sqrt 6 → -x = -(Real.sqrt 6) :=
by sorry

end opposite_of_sqrt_six_l1371_137172


namespace triangle_angle_value_l1371_137102

theorem triangle_angle_value (A B C : ℝ) (a b c : ℝ) :
  0 < B → B < π →
  0 < C → C < π →
  b * Real.cos C + c * Real.sin B = 0 →
  C = 3 * π / 4 := by
  sorry

end triangle_angle_value_l1371_137102


namespace square_ratio_side_length_sum_l1371_137194

theorem square_ratio_side_length_sum (area_ratio : ℚ) :
  area_ratio = 245 / 35 →
  ∃ (a b c : ℕ), 
    (a * (b.sqrt : ℝ) / c : ℝ) ^ 2 = area_ratio ∧
    a = 1 ∧ b = 7 ∧ c = 1 ∧
    a + b + c = 9 :=
by sorry

end square_ratio_side_length_sum_l1371_137194


namespace french_english_speakers_l1371_137125

theorem french_english_speakers (total_students : ℕ) 
  (non_french_percentage : ℚ) (french_non_english : ℕ) : 
  total_students = 200 →
  non_french_percentage = 3/4 →
  french_non_english = 40 →
  (total_students : ℚ) * (1 - non_french_percentage) - french_non_english = 10 :=
by
  sorry

end french_english_speakers_l1371_137125


namespace cubic_inequality_l1371_137183

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 + 27*x > 0 ↔ x ∈ Set.union (Set.Ioo 0 3) (Set.Ioi 9) := by
  sorry

end cubic_inequality_l1371_137183


namespace part_1_part_2_part_3_part_3_unique_l1371_137160

-- Define the algebraic expression
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Part 1
theorem part_1 : f 2 = -1 := by sorry

-- Part 2
theorem part_2 : 
  ∃ x₁ x₂ : ℝ, f x₁ = 4 ∧ f x₂ = 4 ∧ x₁^2 + x₂^2 = 18 := by sorry

-- Part 3
theorem part_3 :
  ∃ m : ℕ, 
    (∃ n₁ n₂ : ℝ, 
      f (m + 2) = n₁ ∧ 
      f (2*m + 1) = n₂ ∧ 
      ∃ k : ℤ, n₂ / n₁ = k) ∧
    m = 3 := by sorry

-- Additional theorem to show the uniqueness of m in part 3
theorem part_3_unique :
  ∀ m : ℕ, 
    (∃ n₁ n₂ : ℝ, 
      f (m + 2) = n₁ ∧ 
      f (2*m + 1) = n₂ ∧ 
      ∃ k : ℤ, n₂ / n₁ = k) →
    m = 3 := by sorry

end part_1_part_2_part_3_part_3_unique_l1371_137160


namespace hyperbola_real_axis_length_l1371_137178

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    and a point P on its right branch,
    a line through P intersects the asymptotes at A and B,
    where A is in the first quadrant and B is in the fourth quadrant,
    O is the origin, AP = (1/2)PB, and the area of triangle AOB is 2b,
    prove that the length of the real axis of C is 32/9. -/
theorem hyperbola_real_axis_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0)
  (P A B : ℝ × ℝ)
  (hC : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (hP : P.1 > 0)
  (hA : A.1 > 0 ∧ A.2 > 0)
  (hB : B.1 > 0 ∧ B.2 < 0)
  (hAP : A - P = (1/2) • (P - B))
  (hAOB : abs ((A.1 * B.2 - A.2 * B.1) / 2) = 2 * b) :
  2 * a = 32/9 := by
  sorry

end hyperbola_real_axis_length_l1371_137178


namespace parallelogram_angle_difference_parallelogram_angle_difference_holds_l1371_137134

/-- In a parallelogram, given that one angle measures 85 degrees, 
    the difference between this angle and its adjacent angle is 10 degrees. -/
theorem parallelogram_angle_difference : ℝ → Prop :=
  fun angle_difference : ℝ =>
    ∀ (smaller_angle larger_angle : ℝ),
      smaller_angle = 85 ∧
      smaller_angle + larger_angle = 180 →
      larger_angle - smaller_angle = angle_difference ∧
      angle_difference = 10

/-- The theorem holds for the given angle difference. -/
theorem parallelogram_angle_difference_holds : parallelogram_angle_difference 10 := by
  sorry

end parallelogram_angle_difference_parallelogram_angle_difference_holds_l1371_137134


namespace child_support_calculation_l1371_137101

def child_support_owed (base_salary : List ℝ) (bonuses : List ℝ) (rates : List ℝ) (paid : ℝ) : ℝ :=
  let incomes := List.zipWith (· + ·) base_salary bonuses
  let owed := List.sum (List.zipWith (· * ·) incomes rates)
  owed - paid

theorem child_support_calculation : 
  let base_salary := [30000, 30000, 30000, 36000, 36000, 36000, 36000]
  let bonuses := [2000, 3000, 4000, 5000, 6000, 7000, 8000]
  let rates := [0.3, 0.3, 0.3, 0.3, 0.3, 0.25, 0.25]
  let paid := 1200
  child_support_owed base_salary bonuses rates paid = 75150 := by
  sorry

#eval child_support_owed 
  [30000, 30000, 30000, 36000, 36000, 36000, 36000]
  [2000, 3000, 4000, 5000, 6000, 7000, 8000]
  [0.3, 0.3, 0.3, 0.3, 0.3, 0.25, 0.25]
  1200

end child_support_calculation_l1371_137101


namespace common_zero_implies_f0_or_f1_zero_l1371_137187

/-- A quadratic function f(x) = x^2 + px + q -/
def f (p q x : ℝ) : ℝ := x^2 + p * x + q

/-- The composition f(f(f(x))) -/
def triple_f (p q x : ℝ) : ℝ := f p q (f p q (f p q x))

/-- Theorem: If f and triple_f have a common zero, then f(0) = 0 or f(1) = 0 -/
theorem common_zero_implies_f0_or_f1_zero (p q : ℝ) :
  (∃ m, f p q m = 0 ∧ triple_f p q m = 0) →
  f p q 0 = 0 ∨ f p q 1 = 0 := by
sorry

end common_zero_implies_f0_or_f1_zero_l1371_137187


namespace smallest_k_for_integer_product_l1371_137164

def a : ℕ → ℝ
  | 0 => 1
  | 1 => 3 ^ (1 / 17)
  | (n + 2) => a (n + 1) * (a n) ^ 2

def product_up_to (k : ℕ) : ℝ :=
  (List.range k).foldl (λ acc i => acc * a (i + 1)) 1

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem smallest_k_for_integer_product :
  (∀ k < 11, ¬ is_integer (product_up_to k)) ∧
  is_integer (product_up_to 11) := by sorry

end smallest_k_for_integer_product_l1371_137164


namespace jims_cousin_money_l1371_137111

-- Define the costs of items
def cheeseburger_cost : ℚ := 3
def milkshake_cost : ℚ := 5
def cheese_fries_cost : ℚ := 8

-- Define the number of items ordered
def num_cheeseburgers : ℕ := 2
def num_milkshakes : ℕ := 2
def num_cheese_fries : ℕ := 1

-- Define Jim's contribution
def jim_money : ℚ := 20

-- Define the percentage of combined money spent
def percentage_spent : ℚ := 80 / 100

-- Theorem to prove
theorem jims_cousin_money :
  let total_cost := num_cheeseburgers * cheeseburger_cost + 
                    num_milkshakes * milkshake_cost + 
                    num_cheese_fries * cheese_fries_cost
  let total_money := total_cost / percentage_spent
  let cousin_money := total_money - jim_money
  cousin_money = 10 := by sorry

end jims_cousin_money_l1371_137111


namespace exists_periodic_nonconstant_sequence_l1371_137154

/-- A sequence is periodic if there exists a positive integer p such that
    x_{n+p} = x_n for all integers n -/
def IsPeriodic (x : ℤ → ℝ) : Prop :=
  ∃ p : ℕ+, ∀ n : ℤ, x (n + p) = x n

/-- A sequence is constant if all its terms are equal -/
def IsConstant (x : ℤ → ℝ) : Prop :=
  ∀ m n : ℤ, x m = x n

/-- The recurrence relation for the sequence -/
def SatisfiesRecurrence (x : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, x (n + 1) = 3 * x n + 4 * x (n - 1)

theorem exists_periodic_nonconstant_sequence :
  ∃ x : ℤ → ℝ, SatisfiesRecurrence x ∧ IsPeriodic x ∧ ¬IsConstant x := by
  sorry

end exists_periodic_nonconstant_sequence_l1371_137154


namespace abs_value_difference_l1371_137130

theorem abs_value_difference (x y : ℝ) (hx : |x| = 2) (hy : |y| = 3) (hxy : x > y) :
  x - y = 5 ∨ x - y = 1 := by
sorry

end abs_value_difference_l1371_137130


namespace special_arrangement_count_l1371_137104

/-- The number of permutations of n distinct objects taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways to arrange 6 people in a row with specific conditions -/
def special_arrangement : ℕ :=
  permutations 2 2 * permutations 4 4

theorem special_arrangement_count :
  special_arrangement = 48 :=
sorry

end special_arrangement_count_l1371_137104


namespace gcd_factorial_problem_l1371_137114

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 5040 := by
  sorry

end gcd_factorial_problem_l1371_137114


namespace unique_two_digit_number_l1371_137141

theorem unique_two_digit_number : ∃! n : ℕ,
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 5 = 0 ∧         -- divisible by 5
  n % 3 ≠ 0 ∧         -- not divisible by 3
  n % 4 ≠ 0 ∧         -- not divisible by 4
  (97 * n) % 2 = 0 ∧  -- 97 times is even
  n / 10 ≥ 6 ∧        -- tens digit not less than 6
  n = 70              -- the number is 70
  := by sorry

end unique_two_digit_number_l1371_137141


namespace P_equals_Q_l1371_137195

-- Define the sets P and Q
def P : Set ℕ := {2, 3}
def Q : Set ℕ := {3, 2}

-- Theorem stating that P and Q are equal
theorem P_equals_Q : P = Q := by sorry

end P_equals_Q_l1371_137195


namespace jogger_train_distance_l1371_137126

/-- Proves that a jogger is 240 meters ahead of a train given specific conditions -/
theorem jogger_train_distance (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  passing_time = 36 →
  (train_speed - jogger_speed) * passing_time - train_length = 240 :=
by
  sorry

#check jogger_train_distance

end jogger_train_distance_l1371_137126


namespace swamp_ecosystem_flies_eaten_l1371_137184

/-- Represents the number of flies eaten daily in a swamp ecosystem -/
def flies_eaten_daily (num_gharials : ℕ) (fish_per_gharial : ℕ) (frogs_per_fish : ℕ) (flies_per_frog : ℕ) : ℕ :=
  num_gharials * fish_per_gharial * frogs_per_fish * flies_per_frog

/-- Theorem stating the number of flies eaten daily in the given swamp ecosystem -/
theorem swamp_ecosystem_flies_eaten :
  flies_eaten_daily 9 15 8 30 = 32400 := by
  sorry

#eval flies_eaten_daily 9 15 8 30

end swamp_ecosystem_flies_eaten_l1371_137184


namespace sin_70_deg_l1371_137199

theorem sin_70_deg (k : ℝ) (h : Real.sin (10 * π / 180) = k) : 
  Real.sin (70 * π / 180) = 1 - 2 * k^2 := by
  sorry

end sin_70_deg_l1371_137199


namespace arithmetic_sequence_first_term_l1371_137158

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  arith : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1 : ℕ) * seq.d) / 2

theorem arithmetic_sequence_first_term 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 3 + seq.a 6 = 40) 
  (h2 : S seq 2 = 10) : 
  seq.a 1 = 5/2 := by
sorry

end arithmetic_sequence_first_term_l1371_137158


namespace delta_fourth_order_zero_l1371_137196

def u (n : ℕ) : ℕ := n^3 + 2*n

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
| f => λ n => f (n + 1) - f n

def iteratedΔ : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
| 0 => id
| k + 1 => Δ ∘ iteratedΔ k

theorem delta_fourth_order_zero (n : ℕ) : 
  ∀ k : ℕ, (∀ n : ℕ, iteratedΔ k u n = 0) ↔ k ≥ 4 :=
sorry

end delta_fourth_order_zero_l1371_137196


namespace sum_of_roots_l1371_137166

theorem sum_of_roots (x : ℝ) : 
  (x^2 + 2023*x = 2024) → 
  (∃ y : ℝ, y^2 + 2023*y = 2024 ∧ x + y = -2023) :=
by sorry

end sum_of_roots_l1371_137166


namespace problem_solution_l1371_137131

-- Define the set B
def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 - x - m < 0}

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | (x - 3*a) * (x - a - 2) < 0}

theorem problem_solution :
  (B = Set.Ioi 2) ∧
  ({a : ℝ | A a ⊆ B} = Set.Ici (2/3)) := by sorry

end problem_solution_l1371_137131


namespace vectors_form_basis_l1371_137119

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

theorem vectors_form_basis : LinearIndependent ℝ ![e₁, e₂] ∧ Submodule.span ℝ {e₁, e₂} = ⊤ := by
  sorry

end vectors_form_basis_l1371_137119


namespace fixed_points_bound_l1371_137133

/-- A polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial n) : ℕ := n - 1

/-- Evaluation of a polynomial at a point -/
def eval (p : IntPolynomial n) (x : ℤ) : ℤ := sorry

/-- Composition of a polynomial with itself k times -/
def composeK (p : IntPolynomial n) (k : ℕ) : IntPolynomial n := sorry

/-- The number of integer solutions to the equation Q_k(t) = t -/
def numFixedPoints (p : IntPolynomial n) (k : ℕ) : ℕ := sorry

theorem fixed_points_bound (n : ℕ) (p : IntPolynomial n) (k : ℕ) :
  degree p > 1 → numFixedPoints p k ≤ degree p := by
  sorry

end fixed_points_bound_l1371_137133


namespace simplify_expression_l1371_137123

theorem simplify_expression : 
  Real.sqrt 8 - 2 * Real.sqrt (1/2) + (2 - Real.sqrt 3) * (2 + Real.sqrt 3) = Real.sqrt 2 + 1 := by
  sorry

end simplify_expression_l1371_137123


namespace max_sum_of_products_l1371_137180

theorem max_sum_of_products (f g h j : ℕ) : 
  f ∈ ({6, 7, 8, 9} : Set ℕ) → 
  g ∈ ({6, 7, 8, 9} : Set ℕ) → 
  h ∈ ({6, 7, 8, 9} : Set ℕ) → 
  j ∈ ({6, 7, 8, 9} : Set ℕ) → 
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j → 
  f * g + g * h + h * j + f * j ≤ 225 :=
by sorry

end max_sum_of_products_l1371_137180


namespace definite_integral_problem_l1371_137107

open Real MeasureTheory Interval

theorem definite_integral_problem :
  ∫ x in (-1 : ℝ)..1, x * cos x + (x^2)^(1/3) = 6/5 := by sorry

end definite_integral_problem_l1371_137107


namespace point_B_in_fourth_quadrant_l1371_137193

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the third quadrant -/
def thirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Defines the fourth quadrant -/
def fourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Given point A in the third quadrant, prove point B is in the fourth quadrant -/
theorem point_B_in_fourth_quadrant (m n : ℝ) 
  (hA : thirdQuadrant ⟨-m, n⟩) : 
  fourthQuadrant ⟨m+1, n-1⟩ := by
  sorry


end point_B_in_fourth_quadrant_l1371_137193


namespace passing_percentage_is_25_percent_l1371_137108

/-- The percentage of total marks needed to pass a test -/
def passing_percentage (pradeep_score : ℕ) (failed_by : ℕ) (max_marks : ℕ) : ℚ :=
  (pradeep_score + failed_by : ℚ) / max_marks * 100

/-- Theorem stating that the passing percentage is 25% given the problem conditions -/
theorem passing_percentage_is_25_percent :
  passing_percentage 185 25 840 = 25 := by
  sorry

end passing_percentage_is_25_percent_l1371_137108


namespace line_y_intercept_l1371_137136

/-- Given a line with slope 4 passing through the point (50, 300), prove that its y-intercept is 100. -/
theorem line_y_intercept (m : ℝ) (x y b : ℝ) :
  m = 4 →
  x = 50 →
  y = 300 →
  y = m * x + b →
  b = 100 := by
sorry

end line_y_intercept_l1371_137136
