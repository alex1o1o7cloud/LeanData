import Mathlib

namespace birdhouse_wood_pieces_l3872_387246

/-- The number of pieces of wood used for each birdhouse -/
def wood_pieces : ℕ := sorry

/-- The cost of wood per piece in dollars -/
def cost_per_piece : ℚ := 3/2

/-- The profit made on each birdhouse in dollars -/
def profit_per_birdhouse : ℚ := 11/2

/-- The total price for two birdhouses in dollars -/
def price_for_two : ℚ := 32

theorem birdhouse_wood_pieces :
  (2 * ((wood_pieces : ℚ) * cost_per_piece + profit_per_birdhouse) = price_for_two) →
  wood_pieces = 7 := by sorry

end birdhouse_wood_pieces_l3872_387246


namespace thirtieth_triangular_number_l3872_387259

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end thirtieth_triangular_number_l3872_387259


namespace smallest_non_factor_product_is_eight_l3872_387261

/-- Given two distinct positive integers that are factors of 60, 
    this function returns the smallest product of these integers 
    that is not a factor of 60. -/
def smallest_non_factor_product : ℕ → ℕ → ℕ :=
  fun x y =>
    if x ≠ y ∧ x > 0 ∧ y > 0 ∧ 60 % x = 0 ∧ 60 % y = 0 ∧ 60 % (x * y) ≠ 0 then
      x * y
    else
      0

theorem smallest_non_factor_product_is_eight :
  ∀ x y : ℕ, x ≠ y → x > 0 → y > 0 → 60 % x = 0 → 60 % y = 0 → 60 % (x * y) ≠ 0 →
  smallest_non_factor_product x y ≥ 8 ∧
  ∃ a b : ℕ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ 60 % a = 0 ∧ 60 % b = 0 ∧ 60 % (a * b) ≠ 0 ∧ a * b = 8 :=
by sorry

#check smallest_non_factor_product_is_eight

end smallest_non_factor_product_is_eight_l3872_387261


namespace tangent_slope_at_zero_l3872_387295

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

theorem tangent_slope_at_zero : 
  (deriv f) 0 = 2 :=
by sorry

end tangent_slope_at_zero_l3872_387295


namespace odd_divisibility_l3872_387221

theorem odd_divisibility (n : ℕ) (h : Odd n) : n ∣ (2^(n.factorial) - 1) := by
  sorry

end odd_divisibility_l3872_387221


namespace hydrolysis_weight_change_l3872_387266

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Sodium in g/mol -/
def Na_weight : ℝ := 22.99

/-- Molecular weight of acetylsalicylic acid (C9H8O4) in g/mol -/
def acetylsalicylic_acid_weight : ℝ := 9 * C_weight + 8 * H_weight + 4 * O_weight

/-- Molecular weight of sodium hydroxide (NaOH) in g/mol -/
def sodium_hydroxide_weight : ℝ := Na_weight + O_weight + H_weight

/-- Molecular weight of salicylic acid (C7H6O3) in g/mol -/
def salicylic_acid_weight : ℝ := 7 * C_weight + 6 * H_weight + 3 * O_weight

/-- Molecular weight of sodium acetate (CH3COONa) in g/mol -/
def sodium_acetate_weight : ℝ := 2 * C_weight + 3 * H_weight + 2 * O_weight + Na_weight

/-- Theorem stating that the overall molecular weight change during the hydrolysis reaction is 0 g/mol -/
theorem hydrolysis_weight_change :
  acetylsalicylic_acid_weight + sodium_hydroxide_weight = salicylic_acid_weight + sodium_acetate_weight :=
by sorry

end hydrolysis_weight_change_l3872_387266


namespace average_income_Q_R_l3872_387282

/-- The average monthly income of P and Q is Rs. 5050, 
    the average monthly income of P and R is Rs. 5200, 
    and the monthly income of P is Rs. 4000. 
    Prove that the average monthly income of Q and R is Rs. 6250. -/
theorem average_income_Q_R (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 → 
  (P + R) / 2 = 5200 → 
  P = 4000 → 
  (Q + R) / 2 = 6250 := by
sorry

end average_income_Q_R_l3872_387282


namespace divisibility_condition_l3872_387276

theorem divisibility_condition (n : ℕ+) : 
  (5^(n.val - 1) + 3^(n.val - 1)) ∣ (5^n.val + 3^n.val) ↔ n = 1 := by
  sorry

end divisibility_condition_l3872_387276


namespace lunch_choices_l3872_387227

theorem lunch_choices (chicken_types : ℕ) (drink_types : ℕ) 
  (h1 : chicken_types = 3) (h2 : drink_types = 2) : 
  chicken_types * drink_types = 6 := by
sorry

end lunch_choices_l3872_387227


namespace calculate_fraction_l3872_387291

theorem calculate_fraction : (2015^2) / (2014^2 + 2016^2 - 2) = 1/2 := by
  sorry

end calculate_fraction_l3872_387291


namespace august_tips_fraction_l3872_387242

theorem august_tips_fraction (total_months : ℕ) (other_months : ℕ) (august_multiplier : ℕ) :
  total_months = other_months + 1 →
  august_multiplier = 8 →
  (august_multiplier : ℚ) / (august_multiplier + other_months : ℚ) = 4 / 7 := by
  sorry

end august_tips_fraction_l3872_387242


namespace mentor_fraction_is_one_seventh_l3872_387292

/-- Represents the mentorship program in a school --/
structure MentorshipProgram where
  seventh_graders : ℕ
  tenth_graders : ℕ
  mentored_seventh : ℕ
  mentoring_tenth : ℕ

/-- Conditions of the mentorship program --/
def valid_program (p : MentorshipProgram) : Prop :=
  p.mentoring_tenth = p.mentored_seventh ∧
  4 * p.mentoring_tenth = p.tenth_graders ∧
  3 * p.mentored_seventh = p.seventh_graders

/-- The fraction of students with a mentor --/
def mentor_fraction (p : MentorshipProgram) : ℚ :=
  p.mentored_seventh / (p.seventh_graders + p.tenth_graders)

/-- Theorem stating that the fraction of students with a mentor is 1/7 --/
theorem mentor_fraction_is_one_seventh (p : MentorshipProgram) 
  (h : valid_program p) : mentor_fraction p = 1 / 7 := by
  sorry

end mentor_fraction_is_one_seventh_l3872_387292


namespace bills_age_l3872_387200

/-- Proves Bill's age given the conditions of the problem -/
theorem bills_age :
  ∀ (bill_age caroline_age : ℕ),
    bill_age = 2 * caroline_age - 1 →
    bill_age + caroline_age = 26 →
    bill_age = 17 := by
  sorry

end bills_age_l3872_387200


namespace smallest_candy_count_l3872_387263

theorem smallest_candy_count : ∃ n : ℕ, 
  (n ≥ 100) ∧ (n < 1000) ∧ 
  ((n + 7) % 6 = 0) ∧ 
  ((n - 5) % 9 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < n ∧ m < 1000 → 
    ((m + 7) % 6 ≠ 0) ∨ ((m - 5) % 9 ≠ 0)) ∧
  n = 113 := by
sorry

end smallest_candy_count_l3872_387263


namespace exists_k_greater_than_two_l3872_387253

/-- Given a linear function y = (k-2)x + 3 that is increasing,
    prove that there exists a value of k greater than 2. -/
theorem exists_k_greater_than_two (k : ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ < x₂ → (k - 2) * x₁ + 3 < (k - 2) * x₂ + 3) : 
  ∃ k' : ℝ, k' > 2 := by
sorry

end exists_k_greater_than_two_l3872_387253


namespace arithmetic_geometric_harmonic_mean_sum_of_squares_l3872_387278

theorem arithmetic_geometric_harmonic_mean_sum_of_squares
  (x y z : ℝ)
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
  sorry

end arithmetic_geometric_harmonic_mean_sum_of_squares_l3872_387278


namespace min_saltwater_animals_is_1136_l3872_387298

/-- The minimum number of saltwater animals Tyler has -/
def min_saltwater_animals : ℕ :=
  let freshwater_aquariums : ℕ := 52
  let full_freshwater_aquariums : ℕ := 38
  let animals_per_full_freshwater : ℕ := 64
  let total_freshwater_animals : ℕ := 6310
  let saltwater_aquariums : ℕ := 28
  let full_saltwater_aquariums : ℕ := 18
  let animals_per_full_saltwater : ℕ := 52
  let min_animals_per_saltwater : ℕ := 20
  
  let full_saltwater_animals : ℕ := full_saltwater_aquariums * animals_per_full_saltwater
  let min_remaining_saltwater_animals : ℕ := (saltwater_aquariums - full_saltwater_aquariums) * min_animals_per_saltwater
  
  full_saltwater_animals + min_remaining_saltwater_animals

theorem min_saltwater_animals_is_1136 : min_saltwater_animals = 1136 := by
  sorry

end min_saltwater_animals_is_1136_l3872_387298


namespace cube_sum_digits_equals_square_l3872_387226

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem cube_sum_digits_equals_square (n : ℕ) :
  n > 0 ∧ n < 1000 ∧ (sum_of_digits n)^3 = n^2 ↔ n = 1 ∨ n = 27 := by
  sorry

end cube_sum_digits_equals_square_l3872_387226


namespace correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_C_l3872_387214

theorem correct_calculation : 2 * Real.sqrt 5 * Real.sqrt 5 = 10 :=
by sorry

theorem incorrect_calculation_A : ¬(Real.sqrt 2 + Real.sqrt 5 = Real.sqrt 7) :=
by sorry

theorem incorrect_calculation_B : ¬(2 * Real.sqrt 3 - Real.sqrt 3 = 2) :=
by sorry

theorem incorrect_calculation_C : ¬(Real.sqrt (3^2 - 2^2) = 1) :=
by sorry

end correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_C_l3872_387214


namespace tan_equation_solutions_l3872_387250

open Real

noncomputable def S (x : ℝ) := tan x + x

theorem tan_equation_solutions :
  let a := arctan 500
  ∃ (sols : Finset ℝ), (∀ x ∈ sols, 0 ≤ x ∧ x ≤ a ∧ tan x = tan (S x)) ∧ Finset.card sols = 160 :=
sorry

end tan_equation_solutions_l3872_387250


namespace password_equation_l3872_387201

theorem password_equation : ∃ (A B C P Q R : ℕ),
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ P < 10 ∧ Q < 10 ∧ R < 10) ∧
  (A ≠ B ∧ A ≠ C ∧ A ≠ P ∧ A ≠ Q ∧ A ≠ R ∧
   B ≠ C ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ R ∧
   C ≠ P ∧ C ≠ Q ∧ C ≠ R ∧
   P ≠ Q ∧ P ≠ R ∧
   Q ≠ R) ∧
  3 * (100000 * A + 10000 * B + 1000 * C + 100 * P + 10 * Q + R) =
  4 * (100000 * P + 10000 * Q + 1000 * R + 100 * A + 10 * B + C) :=
by sorry

end password_equation_l3872_387201


namespace f_properties_l3872_387238

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x ≥ 0 then Real.log (x^2 - 2*x + 2)
  else Real.log (x^2 + 2*x + 2)

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, f x = f (-x)) ∧  -- f is even
  (∀ x : ℝ, x < 0 → f x = Real.log (x^2 + 2*x + 2)) ∧  -- expression for x < 0
  (StrictMonoOn f (Set.Ioo (-1) 0) ∧ StrictMonoOn f (Set.Ioi 1)) := by
  sorry


end f_properties_l3872_387238


namespace business_investment_proof_l3872_387225

/-- Praveen's initial investment in the business -/
def praveenInvestment : ℕ := 35280

/-- Hari's investment in the business -/
def hariInvestment : ℕ := 10080

/-- Praveen's investment duration in months -/
def praveenDuration : ℕ := 12

/-- Hari's investment duration in months -/
def hariDuration : ℕ := 7

/-- Praveen's share in the profit ratio -/
def praveenShare : ℕ := 2

/-- Hari's share in the profit ratio -/
def hariShare : ℕ := 3

theorem business_investment_proof :
  praveenInvestment * praveenDuration * hariShare = 
  hariInvestment * hariDuration * praveenShare := by
  sorry

end business_investment_proof_l3872_387225


namespace product_of_sums_powers_l3872_387254

theorem product_of_sums_powers : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * (3^6 + 1^6) = 2394400 := by
  sorry

end product_of_sums_powers_l3872_387254


namespace nested_multiplication_l3872_387205

theorem nested_multiplication : 3 * (3 * (3 * (3 * (3 * (3 * 2) * 2) * 2) * 2) * 2) * 2 = 1458 := by
  sorry

end nested_multiplication_l3872_387205


namespace expand_and_simplify_l3872_387287

theorem expand_and_simplify (x y : ℝ) : (2*x + 3*y)^2 - (2*x - 3*y)^2 = 24*x*y := by
  sorry

end expand_and_simplify_l3872_387287


namespace refrigerator_cash_savings_l3872_387284

/-- Calculates the savings when buying a refrigerator with cash instead of installments. -/
theorem refrigerator_cash_savings 
  (cash_price : ℕ) 
  (deposit : ℕ) 
  (num_installments : ℕ) 
  (installment_amount : ℕ) 
  (h1 : cash_price = 8000)
  (h2 : deposit = 3000)
  (h3 : num_installments = 30)
  (h4 : installment_amount = 300) :
  deposit + num_installments * installment_amount - cash_price = 4000 :=
by sorry

end refrigerator_cash_savings_l3872_387284


namespace quadratic_factoring_l3872_387237

/-- 
Given a quadratic function y = x^2 - 2x + 3, 
prove that it is equivalent to y = (x - 1)^2 + 2 
when factored into the form y = (x - h)^2 + k
-/
theorem quadratic_factoring (x : ℝ) : 
  x^2 - 2*x + 3 = (x - 1)^2 + 2 := by
  sorry

end quadratic_factoring_l3872_387237


namespace salt_from_seawater_l3872_387297

/-- Calculates the amount of salt obtained from seawater after evaporation -/
def salt_after_evaporation (volume : ℝ) (salt_concentration : ℝ) : ℝ :=
  volume * 1000 * salt_concentration

/-- Theorem: 2 liters of seawater with 20% salt concentration yields 400 ml of salt after evaporation -/
theorem salt_from_seawater :
  salt_after_evaporation 2 0.2 = 400 := by
  sorry

#eval salt_after_evaporation 2 0.2

end salt_from_seawater_l3872_387297


namespace h_x_equality_l3872_387275

theorem h_x_equality (x : ℝ) (h : ℝ → ℝ) : 
  (2 * x^5 + 4 * x^3 + h x = 7 * x^3 - 5 * x^2 + 9 * x + 3) → 
  (h x = -2 * x^5 + 3 * x^3 - 5 * x^2 + 9 * x + 3) :=
by sorry

end h_x_equality_l3872_387275


namespace bus_problem_solution_l3872_387202

/-- Represents the problem of distributing passengers among buses --/
structure BusProblem where
  m : ℕ  -- Initial number of buses
  n : ℕ  -- Number of passengers per bus after redistribution
  initialPassengers : ℕ  -- Initial number of passengers per bus
  maxCapacity : ℕ  -- Maximum capacity of each bus

/-- The conditions of the bus problem --/
def validBusProblem (bp : BusProblem) : Prop :=
  bp.m ≥ 2 ∧
  bp.initialPassengers = 22 ∧
  bp.maxCapacity = 32 ∧
  bp.n ≤ bp.maxCapacity ∧
  bp.initialPassengers * bp.m + 1 = bp.n * (bp.m - 1)

/-- The theorem stating the solution to the bus problem --/
theorem bus_problem_solution (bp : BusProblem) (h : validBusProblem bp) :
  bp.m = 24 ∧ bp.n * (bp.m - 1) = 529 := by
  sorry

#check bus_problem_solution

end bus_problem_solution_l3872_387202


namespace sequence_equality_l3872_387223

def x : ℕ → ℚ
  | 0 => 1
  | n + 1 => x n / (2 + x n)

def y : ℕ → ℚ
  | 0 => 1
  | n + 1 => y n ^ 2 / (1 + 2 * y n)

theorem sequence_equality (n : ℕ) : y n = x (2^n - 1) := by
  sorry

end sequence_equality_l3872_387223


namespace white_surface_fraction_is_two_thirds_l3872_387280

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  white_cube_count : ℕ
  black_cube_count : ℕ

/-- Calculates the fraction of white surface area for a given LargeCube -/
def white_surface_fraction (c : LargeCube) : ℚ :=
  -- The actual calculation is not implemented here
  0

/-- The specific cube described in the problem -/
def problem_cube : LargeCube :=
  { edge_length := 4
  , small_cube_count := 64
  , white_cube_count := 30
  , black_cube_count := 34 }

theorem white_surface_fraction_is_two_thirds :
  white_surface_fraction problem_cube = 2/3 := by
  sorry

end white_surface_fraction_is_two_thirds_l3872_387280


namespace coffee_cream_ratio_l3872_387296

/-- Represents the amount of coffee and cream in a cup -/
structure Coffee :=
  (coffee : ℚ)
  (cream : ℚ)

/-- Calculates the ratio of cream in two coffees -/
def creamRatio (c1 c2 : Coffee) : ℚ :=
  c1.cream / c2.cream

theorem coffee_cream_ratio :
  let max_initial := Coffee.mk 14 0
  let maxine_initial := Coffee.mk 16 0
  let max_after_drinking := Coffee.mk (max_initial.coffee - 4) 0
  let max_final := Coffee.mk max_after_drinking.coffee 3
  let maxine_with_cream := Coffee.mk maxine_initial.coffee 3
  let maxine_final := Coffee.mk (maxine_with_cream.coffee * 14 / 19) (maxine_with_cream.cream * 14 / 19)
  creamRatio max_final maxine_final = 19 / 14 := by
  sorry

end coffee_cream_ratio_l3872_387296


namespace a_minus_b_equals_two_l3872_387229

theorem a_minus_b_equals_two (a b : ℝ) 
  (eq1 : 4 * a + 3 * b = 8) 
  (eq2 : 3 * a + 4 * b = 6) : 
  a - b = 2 := by
sorry

end a_minus_b_equals_two_l3872_387229


namespace wicket_keeper_age_difference_l3872_387281

theorem wicket_keeper_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℕ) 
  (h1 : team_size = 11)
  (h2 : captain_age = 24)
  (h3 : team_avg_age = 21)
  (h4 : ∃ (remaining_avg_age : ℕ), remaining_avg_age = team_avg_age - 1 ∧ 
    (team_size - 2) * remaining_avg_age + captain_age + (captain_age + x) = team_size * team_avg_age) :
  x = 3 :=
sorry

end wicket_keeper_age_difference_l3872_387281


namespace quadratic_minimum_l3872_387243

theorem quadratic_minimum (x : ℝ) : 
  ∃ (min_x : ℝ), ∀ y : ℝ, 2 * x^2 + 6 * x - 5 ≥ 2 * min_x^2 + 6 * min_x - 5 ∧ min_x = -3/2 :=
by sorry

end quadratic_minimum_l3872_387243


namespace opposite_number_any_real_l3872_387207

theorem opposite_number_any_real (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x + y = 0) → 
  (∃ b : ℝ, a + b = 0 ∧ b = -a) → 
  True :=
by sorry

end opposite_number_any_real_l3872_387207


namespace math_marks_calculation_l3872_387241

theorem math_marks_calculation (english physics chemistry biology : ℕ)
  (average : ℕ) (total_subjects : ℕ) (h1 : english = 73)
  (h2 : physics = 92) (h3 : chemistry = 64) (h4 : biology = 82)
  (h5 : average = 76) (h6 : total_subjects = 5) :
  average * total_subjects - (english + physics + chemistry + biology) = 69 :=
by sorry

end math_marks_calculation_l3872_387241


namespace escalator_length_l3872_387213

/-- The length of an escalator given two people walking in opposite directions -/
theorem escalator_length 
  (time_A : ℝ) 
  (time_B : ℝ) 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (h1 : time_A = 100) 
  (h2 : time_B = 300) 
  (h3 : speed_A = 3) 
  (h4 : speed_B = 2) : 
  (speed_A - speed_B) / (1 / time_A - 1 / time_B) = 150 := by
  sorry

end escalator_length_l3872_387213


namespace rotated_line_equation_l3872_387231

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a line 90 degrees counterclockwise around a given point -/
def rotateLine90 (l : Line) (p : Point) : Line :=
  sorry

/-- The original line l₀ -/
def l₀ : Line :=
  { slope := 1, yIntercept := 1 }

/-- The point P around which the line is rotated -/
def P : Point :=
  { x := 3, y := 1 }

/-- The resulting line l after rotation -/
def l : Line :=
  rotateLine90 l₀ P

theorem rotated_line_equation :
  l.slope * P.x + l.yIntercept = P.y ∧ l.slope = -1 →
  ∀ x y, y + x - 4 = 0 ↔ y = l.slope * x + l.yIntercept :=
sorry

end rotated_line_equation_l3872_387231


namespace chalk_pieces_count_l3872_387262

/-- Given a box capacity and number of full boxes, calculates the total number of chalk pieces -/
def total_chalk_pieces (box_capacity : ℕ) (full_boxes : ℕ) : ℕ :=
  box_capacity * full_boxes

/-- Proves that the total number of chalk pieces is 3492 -/
theorem chalk_pieces_count :
  let box_capacity := 18
  let full_boxes := 194
  total_chalk_pieces box_capacity full_boxes = 3492 := by
  sorry

end chalk_pieces_count_l3872_387262


namespace katies_old_friends_games_l3872_387290

theorem katies_old_friends_games 
  (total_friends_games : ℕ) 
  (new_friends_games : ℕ) 
  (h1 : total_friends_games = 141) 
  (h2 : new_friends_games = 88) : 
  total_friends_games - new_friends_games = 53 := by
sorry

end katies_old_friends_games_l3872_387290


namespace dogwood_trees_after_five_years_l3872_387215

/-- Calculates the total number of dogwood trees in the park after a given number of years -/
def total_dogwood_trees (initial_trees : ℕ) (trees_today : ℕ) (trees_tomorrow : ℕ) 
  (growth_rate_today : ℕ) (growth_rate_tomorrow : ℕ) (years : ℕ) : ℕ :=
  initial_trees + 
  (trees_today + growth_rate_today * years) + 
  (trees_tomorrow + growth_rate_tomorrow * years)

/-- Theorem stating that the total number of dogwood trees after 5 years is 130 -/
theorem dogwood_trees_after_five_years : 
  total_dogwood_trees 39 41 20 2 4 5 = 130 := by
  sorry

#eval total_dogwood_trees 39 41 20 2 4 5

end dogwood_trees_after_five_years_l3872_387215


namespace sara_red_balloons_l3872_387286

def total_red_balloons : ℕ := 55
def sandy_red_balloons : ℕ := 24

theorem sara_red_balloons : ∃ (sara_balloons : ℕ), 
  sara_balloons + sandy_red_balloons = total_red_balloons ∧ sara_balloons = 31 := by
  sorry

end sara_red_balloons_l3872_387286


namespace barneys_inventory_l3872_387234

/-- The number of items left in Barney's grocery store -/
def items_left (restocked : ℕ) (sold : ℕ) (in_storeroom : ℕ) : ℕ :=
  (restocked - sold) + in_storeroom

/-- Theorem stating the total number of items left in Barney's grocery store -/
theorem barneys_inventory : items_left 4458 1561 575 = 3472 := by
  sorry

end barneys_inventory_l3872_387234


namespace divisibility_of_p_and_q_l3872_387271

def ones (n : ℕ) : ℕ := (10^n - 1) / 9

def p (n : ℕ) : ℕ := (ones n) * (10^(3*n) + 9*10^(2*n) + 8*10^n + 7)

def q (n : ℕ) : ℕ := (ones (n+1)) * (10^(3*(n+1)) + 9*10^(2*(n+1)) + 8*10^(n+1) + 7)

theorem divisibility_of_p_and_q (n : ℕ) (h : (1987 : ℕ) ∣ ones n) :
  (1987 : ℕ) ∣ p n ∧ (1987 : ℕ) ∣ q n := by
  sorry

end divisibility_of_p_and_q_l3872_387271


namespace first_number_remainder_l3872_387249

/-- A permutation of numbers from 1 to 2023 -/
def Arrangement := Fin 2023 → Fin 2023

/-- Property that any three numbers with one in between have different remainders when divided by 3 -/
def ValidArrangement (arr : Arrangement) : Prop :=
  ∀ i : Fin 2020, (arr i % 3) ≠ (arr (i + 2) % 3) ∧ (arr i % 3) ≠ (arr (i + 4) % 3) ∧ (arr (i + 2) % 3) ≠ (arr (i + 4) % 3)

/-- Theorem stating that the first number in a valid arrangement must have remainder 1 when divided by 3 -/
theorem first_number_remainder (arr : Arrangement) (h : ValidArrangement arr) : arr 0 % 3 = 1 := by
  sorry

end first_number_remainder_l3872_387249


namespace lemon_juice_test_point_l3872_387233

theorem lemon_juice_test_point (lower_bound upper_bound : ℝ) 
  (h_lower : lower_bound = 500)
  (h_upper : upper_bound = 1500)
  (golden_ratio : ℝ) 
  (h_golden : golden_ratio = 0.618) : 
  let x₁ := lower_bound + golden_ratio * (upper_bound - lower_bound)
  let x₂ := upper_bound + lower_bound - x₁
  x₂ = 882 := by
sorry

end lemon_juice_test_point_l3872_387233


namespace inscribed_prism_lateral_area_l3872_387283

theorem inscribed_prism_lateral_area (sphere_surface_area : ℝ) (prism_height : ℝ) :
  sphere_surface_area = 24 * Real.pi →
  prism_height = 4 →
  ∃ (prism_lateral_area : ℝ),
    prism_lateral_area = 32 ∧
    prism_lateral_area = 4 * prism_height * (Real.sqrt ((4 * sphere_surface_area / Real.pi) / 4 - prism_height^2 / 2)) :=
by sorry

end inscribed_prism_lateral_area_l3872_387283


namespace sarahs_bowling_score_l3872_387257

theorem sarahs_bowling_score (sarah_score greg_score : ℕ) : 
  sarah_score = greg_score + 40 →
  (sarah_score + greg_score) / 2 = 102 →
  sarah_score = 122 := by
sorry

end sarahs_bowling_score_l3872_387257


namespace equation_solution_l3872_387272

theorem equation_solution : ∃ x : ℚ, (2 * x + 1) / 5 - x / 10 = 2 ∧ x = 6 := by
  sorry

end equation_solution_l3872_387272


namespace seating_arrangements_l3872_387267

/-- The number of ways to arrange n people in a row. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row with k specific people consecutive. -/
def consecutiveArrangements (n k : ℕ) : ℕ := 
  (Nat.factorial (n - k + 1)) * (Nat.factorial k)

/-- The number of people to be seated. -/
def totalPeople : ℕ := 10

/-- The number of specific individuals who refuse to sit consecutively. -/
def specificIndividuals : ℕ := 4

/-- The number of ways to arrange people with restrictions. -/
def arrangementsWithRestrictions : ℕ := 
  totalArrangements totalPeople - consecutiveArrangements totalPeople specificIndividuals

theorem seating_arrangements :
  arrangementsWithRestrictions = 3507840 := by sorry

end seating_arrangements_l3872_387267


namespace smoothie_ingredients_sum_l3872_387220

/-- The amount of strawberries used in cups -/
def strawberries : ℝ := 0.2

/-- The amount of yogurt used in cups -/
def yogurt : ℝ := 0.1

/-- The amount of orange juice used in cups -/
def orange_juice : ℝ := 0.2

/-- The total amount of ingredients used for the smoothies -/
def total_ingredients : ℝ := strawberries + yogurt + orange_juice

theorem smoothie_ingredients_sum :
  total_ingredients = 0.5 := by sorry

end smoothie_ingredients_sum_l3872_387220


namespace no_two_cubes_between_squares_l3872_387216

theorem no_two_cubes_between_squares : ¬∃ (n a b : ℤ), n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n+1)^2 := by
  sorry

end no_two_cubes_between_squares_l3872_387216


namespace multiply_72519_9999_l3872_387270

theorem multiply_72519_9999 : 72519 * 9999 = 724817481 := by
  sorry

end multiply_72519_9999_l3872_387270


namespace functional_equation_solution_l3872_387230

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (f = fun x ↦ x - 1) ∨ (f = fun x ↦ -x - 1) := by
  sorry

end functional_equation_solution_l3872_387230


namespace sum_base8_equals_1207_l3872_387217

/-- Converts a base-8 number represented as a list of digits to its decimal equivalent -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Converts a decimal number to its base-8 representation as a list of digits -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimalToBase8 (n / 8)

/-- The sum of 527₈, 165₈, and 273₈ in base 8 is equal to 1207₈ -/
theorem sum_base8_equals_1207 :
  let a := base8ToDecimal [7, 2, 5]
  let b := base8ToDecimal [5, 6, 1]
  let c := base8ToDecimal [3, 7, 2]
  decimalToBase8 (a + b + c) = [7, 0, 2, 1] := by
  sorry

end sum_base8_equals_1207_l3872_387217


namespace min_value_fraction_l3872_387244

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem min_value_fraction (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 7 = a 6 + 2 * a 5) →
  (∃ m n : ℕ, a m * a n = 16 * (a 1)^2) →
  (∃ m n : ℕ, ∀ k l : ℕ, 1 / k + 9 / l ≥ 1 / m + 9 / n) →
  (∃ m n : ℕ, 1 / m + 9 / n = 11 / 4) :=
by sorry

end min_value_fraction_l3872_387244


namespace volume_Q4_l3872_387219

/-- Represents the volume of the i-th polyhedron in the sequence --/
def Q (i : ℕ) : ℝ :=
  sorry

/-- The volume difference between consecutive polyhedra --/
def ΔQ (i : ℕ) : ℝ :=
  sorry

theorem volume_Q4 :
  Q 0 = 8 →
  (∀ i : ℕ, ΔQ (i + 1) = (1 / 2) * ΔQ i) →
  ΔQ 1 = 4 →
  Q 4 = 15.5 :=
by
  sorry

end volume_Q4_l3872_387219


namespace mark_households_visited_mark_collection_proof_l3872_387252

theorem mark_households_visited (days : ℕ) (total_collected : ℕ) (donation : ℕ) : ℕ :=
  let households_per_day := 20
  have days_collecting := 5
  have half_households_donate := households_per_day / 2
  have donation_amount := 2 * 20
  have total_collected_calculated := days_collecting * half_households_donate * donation_amount
  households_per_day

theorem mark_collection_proof 
  (days : ℕ) 
  (total_collected : ℕ) 
  (donation : ℕ) 
  (h1 : days = 5) 
  (h2 : donation = 2 * 20) 
  (h3 : total_collected = 2000) :
  mark_households_visited days total_collected donation = 20 := by
  sorry

end mark_households_visited_mark_collection_proof_l3872_387252


namespace sample_size_is_number_of_individuals_l3872_387264

/-- Definition of a sample in statistics -/
structure Sample (α : Type) where
  elements : List α

/-- Definition of sample size -/
def sampleSize {α : Type} (s : Sample α) : ℕ :=
  s.elements.length

/-- Theorem: The sample size is the number of individuals in the sample -/
theorem sample_size_is_number_of_individuals {α : Type} (s : Sample α) :
  sampleSize s = s.elements.length := by
  sorry

end sample_size_is_number_of_individuals_l3872_387264


namespace computer_price_proof_l3872_387228

theorem computer_price_proof (P : ℝ) : 
  1.20 * P = 351 → 2 * P = 585 → P = 292.50 := by sorry

end computer_price_proof_l3872_387228


namespace twenty_people_handshakes_l3872_387256

/-- The number of handshakes when n people shake hands with each other exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 20 people, the total number of handshakes is 190. -/
theorem twenty_people_handshakes :
  handshakes 20 = 190 := by
  sorry

end twenty_people_handshakes_l3872_387256


namespace division_remainder_l3872_387210

theorem division_remainder (j : ℕ) (h1 : j > 0) (h2 : 132 % (j^2) = 12) : 250 % j = 0 := by
  sorry

end division_remainder_l3872_387210


namespace triangle_construction_l3872_387248

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)
  (hγ : γ = 2 * π / 3)  -- 120° in radians
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angles : α + β + γ = π)

-- Define the new triangles
structure NewTriangle :=
  (x y z : ℝ)
  (θ φ ψ : ℝ)
  (h_triangle : x + y > z ∧ y + z > x ∧ z + x > y)
  (h_angles : θ + φ + ψ = π)

-- Statement of the theorem
theorem triangle_construction (abc : Triangle) :
  ∃ (t1 t2 : NewTriangle),
    -- First new triangle
    (t1.x = abc.a ∧ t1.y = abc.c ∧ t1.z = abc.a + abc.b) ∧
    (t1.θ = π / 3 ∧ t1.φ = abc.α ∧ t1.ψ = π / 3 + abc.β) ∧
    -- Second new triangle
    (t2.x = abc.b ∧ t2.y = abc.c ∧ t2.z = abc.a + abc.b) ∧
    (t2.θ = π / 3 ∧ t2.φ = abc.β ∧ t2.ψ = π / 3 + abc.α) :=
by sorry

end triangle_construction_l3872_387248


namespace complement_A_intersect_B_l3872_387251

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = Set.Icc 1 2 := by sorry

end complement_A_intersect_B_l3872_387251


namespace two_intersecting_lines_determine_plane_l3872_387268

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop := sorry

/-- A line lies on a plane -/
def lineOnPlane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Two lines determine a unique plane -/
theorem two_intersecting_lines_determine_plane (l1 l2 : Line3D) :
  intersect l1 l2 → ∃! p : Plane3D, lineOnPlane l1 p ∧ lineOnPlane l2 p :=
sorry

end two_intersecting_lines_determine_plane_l3872_387268


namespace function_value_at_sqrt_two_l3872_387240

/-- Given a function f : ℝ → ℝ satisfying the equation 2 * f x + f (x^2 - 1) = 1 for all real x,
    prove that f(√2) = 1/3 -/
theorem function_value_at_sqrt_two (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, 2 * f x + f (x^2 - 1) = 1) : 
    f (Real.sqrt 2) = 1/3 := by
  sorry

end function_value_at_sqrt_two_l3872_387240


namespace triangle_inequality_sum_l3872_387222

theorem triangle_inequality_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  3 ≤ (a / (b + c - a)).sqrt + (b / (a + c - b)).sqrt + (c / (a + b - c)).sqrt :=
sorry

end triangle_inequality_sum_l3872_387222


namespace seating_theorem_l3872_387203

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  total_people : ℕ
  rows_with_seven : ℕ
  rows_with_six : ℕ

/-- Checks if a seating arrangement is valid --/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.total_people = 53 ∧
  s.rows_with_seven * 7 + s.rows_with_six * 6 = s.total_people

/-- The theorem to be proved --/
theorem seating_theorem :
  ∃ (s : SeatingArrangement), is_valid_arrangement s ∧ s.rows_with_seven = 5 :=
sorry

end seating_theorem_l3872_387203


namespace brick_width_calculation_l3872_387288

theorem brick_width_calculation (wall_length wall_width wall_height : ℝ)
                                (brick_length brick_height : ℝ)
                                (num_bricks : ℕ) :
  wall_length = 800 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_length = 80 →
  brick_height = 6 →
  num_bricks = 2000 →
  ∃ brick_width : ℝ,
    num_bricks * (brick_length * brick_width * brick_height) = wall_length * wall_width * wall_height ∧
    brick_width = 5.625 := by
  sorry

end brick_width_calculation_l3872_387288


namespace geometric_sequence_arithmetic_mean_l3872_387204

/-- Given a geometric sequence {a_n} with common ratio q = -2 and a_3 * a_7 = 4 * a_4,
    prove that the arithmetic mean of a_8 and a_11 is -56. -/
theorem geometric_sequence_arithmetic_mean 
  (a : ℕ → ℝ) -- The geometric sequence
  (h1 : ∀ n, a (n + 1) = a n * (-2)) -- Common ratio q = -2
  (h2 : a 3 * a 7 = 4 * a 4) -- Given condition
  : (a 8 + a 11) / 2 = -56 := by
sorry

end geometric_sequence_arithmetic_mean_l3872_387204


namespace prob_rain_sunday_and_monday_l3872_387224

-- Define the probabilities
def prob_rain_saturday : ℝ := 0.8
def prob_rain_sunday : ℝ := 0.3
def prob_rain_monday_if_sunday : ℝ := 0.5
def prob_rain_monday_if_not_sunday : ℝ := 0.1

-- Define the independence of Saturday and Sunday
axiom saturday_sunday_independent : True

-- Theorem to prove
theorem prob_rain_sunday_and_monday : 
  prob_rain_sunday * prob_rain_monday_if_sunday = 0.15 := by
  sorry

end prob_rain_sunday_and_monday_l3872_387224


namespace f_equiv_g_l3872_387235

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t + 1

-- Theorem stating that f and g are equivalent for all real numbers
theorem f_equiv_g : ∀ x : ℝ, f x = g x := by sorry

end f_equiv_g_l3872_387235


namespace average_weight_increase_l3872_387232

/-- Proves that replacing a person weighing 65 kg with a person weighing 105 kg
    in a group of 8 people increases the average weight by 5 kg. -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 8 * initial_average
  let new_total := initial_total - 65 + 105
  let new_average := new_total / 8
  new_average - initial_average = 5 := by
  sorry

end average_weight_increase_l3872_387232


namespace inequality_equivalence_l3872_387285

theorem inequality_equivalence (x y : ℝ) : (y - x)^2 < x^2 ↔ y > 0 ∧ y < 2*x := by
  sorry

end inequality_equivalence_l3872_387285


namespace triangle_angle_B_l3872_387211

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) : 
  a = Real.sqrt 3 →
  b = Real.sqrt 6 →
  A = π / 6 →
  (B = π / 4 ∨ B = 3 * π / 4) := by
  sorry

end triangle_angle_B_l3872_387211


namespace binary_101101_conversion_l3872_387258

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, x) => acc + if x then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem binary_101101_conversion :
  let binary := [true, false, true, true, false, true]
  (binary_to_decimal binary = 45) ∧
  (decimal_to_base7 (binary_to_decimal binary) = [6, 3]) := by
  sorry

end binary_101101_conversion_l3872_387258


namespace sqrt_equality_implies_specific_integers_l3872_387273

theorem sqrt_equality_implies_specific_integers :
  ∀ a b : ℕ+,
  a < b →
  Real.sqrt (1 + Real.sqrt (33 + 16 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b →
  a = 1 ∧ b = 17 := by
sorry

end sqrt_equality_implies_specific_integers_l3872_387273


namespace fraction_product_theorem_l3872_387239

theorem fraction_product_theorem (fractions : Finset (ℕ × ℕ)) : 
  (fractions.card = 48) →
  (∀ (n : ℕ), n ∈ fractions.image Prod.fst → 2 ≤ n ∧ n ≤ 49) →
  (∀ (d : ℕ), d ∈ fractions.image Prod.snd → 2 ≤ d ∧ d ≤ 49) →
  (∀ (k : ℕ), 2 ≤ k ∧ k ≤ 49 → (fractions.filter (λ f => f.fst = k)).card = 1) →
  (∀ (k : ℕ), 2 ≤ k ∧ k ≤ 49 → (fractions.filter (λ f => f.snd = k)).card = 1) →
  (∃ (f : ℕ × ℕ), f ∈ fractions ∧ f.fst % f.snd = 0) ∨
  (∃ (subset : Finset (ℕ × ℕ)), subset ⊆ fractions ∧ subset.card ≤ 25 ∧ 
    (subset.prod (λ f => f.fst) % subset.prod (λ f => f.snd) = 0)) :=
by sorry

end fraction_product_theorem_l3872_387239


namespace min_value_quadratic_l3872_387218

theorem min_value_quadratic (x y : ℝ) :
  x^2 + y^2 - 10*x + 6*y + 25 ≥ -9 ∧
  ∃ (a b : ℝ), a^2 + b^2 - 10*a + 6*b + 25 = -9 :=
by sorry

end min_value_quadratic_l3872_387218


namespace machine_job_time_l3872_387293

theorem machine_job_time (y : ℝ) : 
  (1 / (y + 8) + 1 / (y + 3) + 1 / (1.5 * y) = 1 / y) →
  y = (-25 + Real.sqrt 421) / 6 :=
by sorry

end machine_job_time_l3872_387293


namespace find_number_l3872_387265

theorem find_number : ∃ x : ℝ, 1.35 + 0.321 + x = 1.794 ∧ x = 0.123 := by
  sorry

end find_number_l3872_387265


namespace impossible_to_change_all_signs_l3872_387269

/-- Represents a point in the decagon configuration -/
structure Point where
  value : Int
  mk_point : value = 1 ∨ value = -1

/-- Represents the decagon configuration -/
structure DecagonConfig where
  points : Finset Point
  mk_config : points.card = 220

/-- Represents an operation on the decagon -/
inductive Operation
  | side : Operation
  | diagonal : Operation

/-- Applies an operation to the decagon configuration -/
def apply_operation (config : DecagonConfig) (op : Operation) : DecagonConfig :=
  sorry

/-- Checks if all points in the configuration are -1 -/
def all_negative (config : DecagonConfig) : Prop :=
  ∀ p ∈ config.points, p.value = -1

/-- Main theorem: It's impossible to change all signs to their opposites -/
theorem impossible_to_change_all_signs (initial_config : DecagonConfig) :
  ¬∃ (ops : List Operation), all_negative (ops.foldl apply_operation initial_config) :=
sorry

end impossible_to_change_all_signs_l3872_387269


namespace quadratic_shift_theorem_l3872_387247

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a vertical shift to a quadratic function -/
def verticalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, b := f.b, c := f.c + shift }

/-- Applies a horizontal shift to a quadratic function -/
def horizontalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, b := 2 * f.a * shift + f.b, c := f.a * shift^2 + f.b * shift + f.c }

/-- The main theorem stating that shifting y = -2x^2 down 3 units and left 1 unit 
    results in y = -2(x + 1)^2 - 3 -/
theorem quadratic_shift_theorem :
  let f : QuadraticFunction := { a := -2, b := 0, c := 0 }
  let shifted := horizontalShift (verticalShift f (-3)) (-1)
  shifted.a = -2 ∧ shifted.b = 4 ∧ shifted.c = -5 := by
  sorry

end quadratic_shift_theorem_l3872_387247


namespace camp_III_selected_count_l3872_387299

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  totalStudents : Nat
  sampleSize : Nat
  startNumber : Nat
  campIIIStart : Nat
  campIIIEnd : Nat

/-- Calculates the number of students selected from Camp III in a systematic sample -/
def countCampIIISelected (s : SystematicSample) : Nat :=
  let interval := s.totalStudents / s.sampleSize
  let firstCampIII := s.startNumber + interval * ((s.campIIIStart - s.startNumber + interval - 1) / interval)
  let lastSelected := s.startNumber + interval * (s.sampleSize - 1)
  if firstCampIII > s.campIIIEnd then 0
  else ((min lastSelected s.campIIIEnd) - firstCampIII) / interval + 1

theorem camp_III_selected_count (s : SystematicSample) 
  (h1 : s.totalStudents = 600) 
  (h2 : s.sampleSize = 50) 
  (h3 : s.startNumber = 3) 
  (h4 : s.campIIIStart = 496) 
  (h5 : s.campIIIEnd = 600) : 
  countCampIIISelected s = 8 := by
  sorry

end camp_III_selected_count_l3872_387299


namespace total_players_l3872_387236

theorem total_players (kabaddi : ℕ) (kho_kho_only : ℕ) (both : ℕ) 
  (h1 : kabaddi = 10) 
  (h2 : kho_kho_only = 40) 
  (h3 : both = 5) : 
  kabaddi + kho_kho_only - both = 50 := by
  sorry

#check total_players

end total_players_l3872_387236


namespace revenue_division_l3872_387245

theorem revenue_division (total_revenue : ℝ) (ratio_sum : ℕ) (salary_ratio rent_ratio marketing_ratio : ℕ) :
  total_revenue = 10000 →
  ratio_sum = 3 + 5 + 2 + 7 →
  salary_ratio = 3 →
  rent_ratio = 2 →
  marketing_ratio = 7 →
  (salary_ratio + rent_ratio + marketing_ratio) * (total_revenue / ratio_sum) = 7058.88 := by
  sorry

end revenue_division_l3872_387245


namespace feline_sanctuary_count_l3872_387212

theorem feline_sanctuary_count :
  let lions : ℕ := 12
  let tigers : ℕ := 14
  let cougars : ℕ := (lions + tigers) / 2
  lions + tigers + cougars = 39 :=
by sorry

end feline_sanctuary_count_l3872_387212


namespace class_size_l3872_387209

theorem class_size (N M S : ℕ) 
  (h1 : N - M = 10)
  (h2 : N - S = 15)
  (h3 : N - (M + S - 7) = 2)
  (h4 : M + S = N + 7) : N = 34 := by
  sorry

end class_size_l3872_387209


namespace problem_solution_l3872_387279

theorem problem_solution (x y : ℝ) (h1 : x / y = 15 / 5) (h2 : y = 25) : x = 75 := by
  sorry

end problem_solution_l3872_387279


namespace sign_sum_theorem_l3872_387289

theorem sign_sum_theorem (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hsum : a + b + c = 0) :
  (a / |a| + b / |b| + c / |c| - (a * b * c) / |a * b * c|) = 2 ∨
  (a / |a| + b / |b| + c / |c| - (a * b * c) / |a * b * c|) = -2 :=
by sorry

end sign_sum_theorem_l3872_387289


namespace weed_ratio_l3872_387260

/-- Represents the number of weeds pulled on each day -/
structure WeedCount where
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℚ
  friday : ℚ

/-- The problem of Sarah's weed pulling -/
def weed_problem (w : WeedCount) : Prop :=
  w.tuesday = 25 ∧
  w.wednesday = 3 * w.tuesday ∧
  w.thursday = (1 : ℚ) / 5 * w.wednesday ∧
  w.friday = w.thursday - 10 ∧
  w.tuesday + w.wednesday + w.thursday + w.friday = 120

/-- The theorem stating the ratio of weeds pulled on Thursday to Wednesday -/
theorem weed_ratio (w : WeedCount) (h : weed_problem w) : 
  w.thursday / w.wednesday = (1 : ℚ) / 5 := by
  sorry


end weed_ratio_l3872_387260


namespace sector_area_l3872_387274

theorem sector_area (centralAngle : Real) (radius : Real) : 
  centralAngle = π / 6 → radius = 2 → (1 / 2) * centralAngle * radius^2 = π / 3 := by
  sorry

end sector_area_l3872_387274


namespace allocation_schemes_count_l3872_387255

/-- The number of ways to choose k items from n items without replacement and where order matters. -/
def A (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def C (n k : ℕ) : ℕ := sorry

/-- The total number of allocation schemes for assigning 3 people to 7 communities with at most 2 people per community. -/
def totalAllocationSchemes : ℕ := A 7 3 + C 3 2 * C 1 1 * A 7 2

theorem allocation_schemes_count :
  totalAllocationSchemes = 336 := by sorry

end allocation_schemes_count_l3872_387255


namespace only_translation_preserves_pattern_l3872_387294

-- Define the pattern
structure Pattern where
  square_side : ℝ
  triangle_hypotenuse : ℝ
  line_segments_length : ℝ
  total_length : ℝ
  triangle_faces_away : Bool

-- Define the line and the repeating pattern
def infinite_line_with_pattern : Pattern :=
  { square_side := 1
  , triangle_hypotenuse := 1
  , line_segments_length := 2
  , total_length := 4
  , triangle_faces_away := true
  }

-- Define the rigid motion transformations
inductive RigidMotion
  | Rotation (center : ℝ × ℝ) (angle : ℝ)
  | Translation (distance : ℝ)
  | ReflectionAcross
  | ReflectionPerpendicular (point : ℝ)

-- Theorem statement
theorem only_translation_preserves_pattern :
  ∀ (motion : RigidMotion),
    (∃ (k : ℤ), motion = RigidMotion.Translation (↑k * infinite_line_with_pattern.total_length)) ↔
    (motion ≠ RigidMotion.ReflectionAcross ∧
     (∀ (center : ℝ × ℝ) (angle : ℝ), motion ≠ RigidMotion.Rotation center angle) ∧
     (∀ (point : ℝ), motion ≠ RigidMotion.ReflectionPerpendicular point) ∧
     (∃ (distance : ℝ), motion = RigidMotion.Translation distance ∧
        distance = ↑k * infinite_line_with_pattern.total_length)) :=
by sorry

end only_translation_preserves_pattern_l3872_387294


namespace C_formula_l3872_387208

/-- 
C(n, p) represents the number of decompositions of n into sums of powers of p, 
where each power p^k appears at most p^2 - 1 times
-/
def C (n p : ℕ) : ℕ := sorry

/-- Theorem stating the formula for C(n, p) -/
theorem C_formula (n p : ℕ) (hp : p > 1) : C n p = n / p + 1 := by sorry

end C_formula_l3872_387208


namespace range_of_a_l3872_387206

theorem range_of_a (x a : ℝ) : 
  (∀ x, (-3 ≤ x ∧ x ≤ 3) ↔ x < a) →
  a ∈ Set.Ioi 3 :=
sorry

end range_of_a_l3872_387206


namespace probability_red_card_equal_suits_l3872_387277

structure Deck :=
  (total_cards : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = (red_suits + black_suits) * cards_per_suit)
  (h_equal_suits : red_suits = black_suits)

def probability_red_card (d : Deck) : ℚ :=
  (d.red_suits * d.cards_per_suit : ℚ) / d.total_cards

theorem probability_red_card_equal_suits (d : Deck) :
  probability_red_card d = 1 :=
sorry

end probability_red_card_equal_suits_l3872_387277
