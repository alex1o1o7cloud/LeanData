import Mathlib

namespace least_positive_integer_divisible_by_four_primes_l3361_336173

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (p₁ p₂ p₃ p₄ : Nat), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    210 % p₁ = 0 ∧ 210 % p₂ = 0 ∧ 210 % p₃ = 0 ∧ 210 % p₄ = 0 ∧
    ∀ n : Nat, n > 0 ∧ n < 210 → 
      ¬∃ (q₁ q₂ q₃ q₄ : Nat), 
        Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧
        q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
        n % q₁ = 0 ∧ n % q₂ = 0 ∧ n % q₃ = 0 ∧ n % q₄ = 0 :=
by sorry

end least_positive_integer_divisible_by_four_primes_l3361_336173


namespace min_triple_sum_bound_l3361_336172

def circle_arrangement (n : ℕ) := Fin n → ℕ

theorem min_triple_sum_bound (arr : circle_arrangement 10) :
  ∀ i : Fin 10, arr i ∈ Finset.range 11 →
  (∀ i j : Fin 10, i ≠ j → arr i ≠ arr j) →
  ∃ i : Fin 10, arr i + arr ((i + 1) % 10) + arr ((i + 2) % 10) ≤ 15 :=
sorry

end min_triple_sum_bound_l3361_336172


namespace three_hour_charge_l3361_336137

/-- Represents the therapy pricing structure and calculates total charges --/
structure TherapyPricing where
  first_hour : ℝ
  subsequent_hour : ℝ
  service_fee_rate : ℝ
  first_hour_premium : ℝ
  eight_hour_total : ℝ

/-- Calculates the total charge for a given number of hours --/
def total_charge (p : TherapyPricing) (hours : ℕ) : ℝ :=
  let base_charge := p.first_hour + p.subsequent_hour * (hours - 1)
  base_charge * (1 + p.service_fee_rate)

/-- Theorem stating the total charge for 3 hours of therapy --/
theorem three_hour_charge (p : TherapyPricing) : 
  p.first_hour = p.subsequent_hour + p.first_hour_premium →
  p.service_fee_rate = 0.1 →
  p.first_hour_premium = 50 →
  total_charge p 8 = p.eight_hour_total →
  p.eight_hour_total = 900 →
  total_charge p 3 = 371.87 := by
  sorry

end three_hour_charge_l3361_336137


namespace f_properties_l3361_336121

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x * x^2

theorem f_properties :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 →
  (∀ x ∈ Set.Ioo 0 (Real.exp (-1/2)), f x ≥ f (Real.exp (-1/2))) ∧
  (∀ x ∈ Set.Ioi (Real.exp (-1/2)), f x ≥ f (Real.exp (-1/2))) ∧
  f (Real.exp (-1/2)) = -1 / Real.exp 1 :=
by sorry

end f_properties_l3361_336121


namespace science_club_enrollment_l3361_336119

theorem science_club_enrollment (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_math : math = 95)
  (h_physics : physics = 70)
  (h_both : both = 25) :
  total - (math + physics - both) = 10 := by
  sorry

end science_club_enrollment_l3361_336119


namespace arithmetic_sequence_common_difference_l3361_336192

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h_sum1 : a 3 + a 6 = 11)
  (h_sum2 : a 5 + a 8 = 39) :
  d = 7 := by sorry

end arithmetic_sequence_common_difference_l3361_336192


namespace expected_value_coin_flip_l3361_336143

/-- The expected value of winnings for a coin flip game -/
theorem expected_value_coin_flip :
  let p_heads : ℚ := 2 / 5
  let p_tails : ℚ := 3 / 5
  let win_heads : ℚ := 5
  let loss_tails : ℚ := 3
  p_heads * win_heads - p_tails * loss_tails = 1 / 5 := by
  sorry

end expected_value_coin_flip_l3361_336143


namespace gina_payment_is_90_l3361_336175

/-- Calculates the total payment for Gina's order given her painting rates and order details. -/
def total_payment (rose_rate : ℕ) (lily_rate : ℕ) (rose_order : ℕ) (lily_order : ℕ) (hourly_rate : ℕ) : ℕ :=
  let rose_time := rose_order / rose_rate
  let lily_time := lily_order / lily_rate
  let total_time := rose_time + lily_time
  total_time * hourly_rate

/-- Proves that Gina's total payment for the given order is $90. -/
theorem gina_payment_is_90 : total_payment 6 7 6 14 30 = 90 := by
  sorry

#eval total_payment 6 7 6 14 30

end gina_payment_is_90_l3361_336175


namespace smoking_lung_cancer_relationship_l3361_336139

-- Define the confidence level in the smoking-lung cancer relationship
def confidence_level : ℝ := 0.99

-- Define the probability of making a mistake in the conclusion
def error_probability : ℝ := 0.01

-- Define a sample size
def sample_size : ℕ := 100

-- Define a predicate for having lung cancer
def has_lung_cancer : (ℕ → Prop) := sorry

-- Define a predicate for being a smoker
def is_smoker : (ℕ → Prop) := sorry

-- Theorem stating that high confidence in the smoking-lung cancer relationship
-- does not preclude the possibility of a sample with no lung cancer cases
theorem smoking_lung_cancer_relationship 
  (h1 : confidence_level > 0.99) 
  (h2 : error_probability ≤ 0.01) :
  ∃ (sample : Finset ℕ), 
    (∀ i ∈ sample, is_smoker i) ∧ 
    (Finset.card sample = sample_size) ∧
    (∀ i ∈ sample, ¬has_lung_cancer i) := by
  sorry

end smoking_lung_cancer_relationship_l3361_336139


namespace union_equality_implies_m_values_l3361_336169

def A : Set ℝ := {-1, 2}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem union_equality_implies_m_values (m : ℝ) :
  A ∪ B m = A → m = 0 ∨ m = 1 ∨ m = -1/2 := by
  sorry

end union_equality_implies_m_values_l3361_336169


namespace normal_hours_calculation_l3361_336103

/-- Represents a worker's pay structure and a specific workday --/
structure WorkDay where
  regularRate : ℝ  -- Regular hourly rate
  overtimeMultiplier : ℝ  -- Overtime rate multiplier
  totalHours : ℝ  -- Total hours worked on a specific day
  totalEarnings : ℝ  -- Total earnings for the specific day
  normalHours : ℝ  -- Normal working hours per day

/-- Theorem stating that given the specific conditions, the normal working hours are 7.5 --/
theorem normal_hours_calculation (w : WorkDay) 
  (h1 : w.regularRate = 3.5)
  (h2 : w.overtimeMultiplier = 1.5)
  (h3 : w.totalHours = 10.5)
  (h4 : w.totalEarnings = 42)
  : w.normalHours = 7.5 := by
  sorry

end normal_hours_calculation_l3361_336103


namespace stratified_sample_male_count_l3361_336199

/-- Calculates the number of male athletes in a stratified sample -/
def maleAthletesInSample (totalMale : ℕ) (totalFemale : ℕ) (sampleSize : ℕ) : ℕ :=
  (totalMale * sampleSize) / (totalMale + totalFemale)

/-- Theorem: In a stratified sample of 14 athletes drawn from a population of 32 male and 24 female athletes, the number of male athletes in the sample is 8 -/
theorem stratified_sample_male_count :
  maleAthletesInSample 32 24 14 = 8 := by
  sorry

end stratified_sample_male_count_l3361_336199


namespace publishing_break_even_point_l3361_336122

/-- Represents the break-even point calculation for a publishing company --/
theorem publishing_break_even_point 
  (fixed_cost : ℝ) 
  (variable_cost_per_book : ℝ) 
  (selling_price_per_book : ℝ) 
  (h1 : fixed_cost = 56430)
  (h2 : variable_cost_per_book = 8.25)
  (h3 : selling_price_per_book = 21.75) :
  ∃ (x : ℝ), 
    x = 4180 ∧ 
    fixed_cost + x * variable_cost_per_book = x * selling_price_per_book :=
by sorry

end publishing_break_even_point_l3361_336122


namespace decimal_expansion_contains_all_digits_l3361_336165

theorem decimal_expansion_contains_all_digits (p : ℕ) (hp : p.Prime) (hp_large : p > 10^9) 
  (hq : (4*p + 1).Prime) : 
  ∀ d : Fin 10, ∃ n : ℕ, (10^n - 1) % (4*p + 1) = d.val * ((10^n - 1) / (4*p + 1)) :=
sorry

end decimal_expansion_contains_all_digits_l3361_336165


namespace max_blocks_in_box_l3361_336194

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular solid given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- Represents the box dimensions -/
def box : Dimensions := ⟨3, 4, 2⟩

/-- Represents the block dimensions -/
def block : Dimensions := ⟨2, 1, 2⟩

/-- Theorem stating that the maximum number of blocks that can fit in the box is 6 -/
theorem max_blocks_in_box :
  ∃ (n : ℕ), n = 6 ∧ 
  n * volume block ≤ volume box ∧
  ∀ m : ℕ, m * volume block ≤ volume box → m ≤ n :=
by sorry

end max_blocks_in_box_l3361_336194


namespace prime_roots_range_l3361_336190

theorem prime_roots_range (p : ℕ) (h_prime : Nat.Prime p) 
  (h_roots : ∃ x y : ℤ, x^2 + p*x - 444*p = 0 ∧ y^2 + p*y - 444*p = 0) : 
  31 < p ∧ p ≤ 41 :=
sorry

end prime_roots_range_l3361_336190


namespace max_toys_theorem_l3361_336167

def max_toys_purchasable (initial_amount : ℚ) (game_cost : ℚ) (tax_rate : ℚ) (toy_cost : ℚ) : ℕ :=
  let total_game_cost := game_cost * (1 + tax_rate)
  let remaining_money := initial_amount - total_game_cost
  (remaining_money / toy_cost).floor.toNat

theorem max_toys_theorem :
  max_toys_purchasable 57 27 (8/100) 6 = 4 := by
  sorry

end max_toys_theorem_l3361_336167


namespace ace_of_hearts_probability_l3361_336189

def standard_deck : ℕ := 52
def jokers : ℕ := 2
def total_cards : ℕ := standard_deck + jokers
def ace_of_hearts : ℕ := 1

theorem ace_of_hearts_probability :
  (ace_of_hearts : ℚ) / total_cards = 1 / 54 :=
sorry

end ace_of_hearts_probability_l3361_336189


namespace smallest_number_l3361_336138

theorem smallest_number (a b c d e : ℚ) : 
  a = 3.4 ∧ b = 7/2 ∧ c = 1.7 ∧ d = 27/10 ∧ e = 2.9 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d ∧ c ≤ e := by
sorry

end smallest_number_l3361_336138


namespace loans_equal_at_start_l3361_336136

/-- Represents the loan details for a person -/
structure Loan where
  principal : ℝ
  dailyInterestRate : ℝ

/-- Calculates the balance of a loan after t days -/
def loanBalance (loan : Loan) (t : ℝ) : ℝ :=
  loan.principal * (1 + loan.dailyInterestRate * t)

theorem loans_equal_at_start (claudia bob diana : Loan)
  (h_claudia : claudia = { principal := 200, dailyInterestRate := 0.04 })
  (h_bob : bob = { principal := 300, dailyInterestRate := 0.03 })
  (h_diana : diana = { principal := 500, dailyInterestRate := 0.02 }) :
  ∃ t : ℝ, t = 0 ∧ loanBalance claudia t + loanBalance bob t = loanBalance diana t :=
sorry

end loans_equal_at_start_l3361_336136


namespace number_solution_l3361_336158

theorem number_solution : ∃ x : ℝ, (50 + 5 * 12 / (180 / x) = 51) ∧ x = 3 := by
  sorry

end number_solution_l3361_336158


namespace factorization_condition_l3361_336159

-- Define the polynomial
def polynomial (m : ℤ) (x y : ℤ) : ℤ := x^2 + 5*x*y + 2*x + m*y - 2*m

-- Define what it means for a polynomial to have two linear factors with integer coefficients
def has_two_linear_factors (m : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), 
    ∀ (x y : ℤ), polynomial m x y = (a*x + b*y + c) * (d*x + e*y + f)

-- State the theorem
theorem factorization_condition (m : ℤ) : 
  has_two_linear_factors m ↔ (m = 0 ∨ m = 10) := by sorry

end factorization_condition_l3361_336159


namespace scientific_notation_42000_l3361_336185

theorem scientific_notation_42000 :
  (42000 : ℝ) = 4.2 * (10 : ℝ)^4 := by sorry

end scientific_notation_42000_l3361_336185


namespace samantha_score_l3361_336128

/-- Calculates the score for a revised AMC 8 contest --/
def calculate_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℤ :=
  2 * correct - incorrect

/-- Proves that Samantha's score is 25 given the problem conditions --/
theorem samantha_score :
  let correct : ℕ := 15
  let incorrect : ℕ := 5
  let unanswered : ℕ := 5
  let total_questions : ℕ := correct + incorrect + unanswered
  total_questions = 25 →
  calculate_score correct incorrect unanswered = 25 := by
  sorry

end samantha_score_l3361_336128


namespace quadratic_equations_solutions_l3361_336198

theorem quadratic_equations_solutions :
  (∀ x : ℝ, 2 * x^2 - 8 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x : ℝ, x^2 + 10 * x + 9 = 0 ↔ x = -9 ∨ x = -1) ∧
  (∀ x : ℝ, 5 * x^2 - 4 * x - 1 = 0 ↔ x = -1/5 ∨ x = 1) ∧
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1) :=
by sorry

end quadratic_equations_solutions_l3361_336198


namespace number_difference_l3361_336117

theorem number_difference (x y : ℝ) 
  (sum_eq : x + y = 15) 
  (diff_eq : x - y = 10) 
  (square_diff_eq : x^2 - y^2 = 150) : 
  x - y = 10 := by
  sorry

end number_difference_l3361_336117


namespace small_circle_area_l3361_336127

theorem small_circle_area (large_circle_area : ℝ) (num_small_circles : ℕ) :
  large_circle_area = 120 →
  num_small_circles = 6 →
  ∃ small_circle_area : ℝ,
    small_circle_area = large_circle_area / (3 * num_small_circles) ∧
    small_circle_area = 40 := by
  sorry

end small_circle_area_l3361_336127


namespace quadratic_inequality_solution_l3361_336168

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 42*x + 400 ≤ 16 ↔ 16 ≤ x ∧ x ≤ 24 := by sorry

end quadratic_inequality_solution_l3361_336168


namespace intersection_formula_l3361_336129

/-- Given complex numbers a and b on a circle centered at the origin,
    u is the intersection of tangents at a and b -/
def intersection_of_tangents (a b : ℂ) : ℂ := sorry

/-- a and b lie on a circle centered at the origin -/
def on_circle (a b : ℂ) : Prop := sorry

theorem intersection_formula {a b : ℂ} (h : on_circle a b) :
  intersection_of_tangents a b = 2 * a * b / (a + b) := by sorry

end intersection_formula_l3361_336129


namespace approximating_functions_theorem1_approximating_functions_theorem2_l3361_336116

-- Define the concept of "approximating functions"
def approximating_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → -1 ≤ f x - g x ∧ f x - g x ≤ 1

-- Define the functions
def f1 (x : ℝ) : ℝ := x - 5
def f2 (x : ℝ) : ℝ := x^2 - 4*x
def g1 (x : ℝ) : ℝ := x^2 - 1
def g2 (x : ℝ) : ℝ := 2*x^2 - x

-- State the theorems to be proved
theorem approximating_functions_theorem1 :
  approximating_functions f1 f2 3 4 := by sorry

theorem approximating_functions_theorem2 :
  approximating_functions g1 g2 0 1 := by sorry

end approximating_functions_theorem1_approximating_functions_theorem2_l3361_336116


namespace fold_point_area_l3361_336174

/-- Definition of a triangle ABC --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Definition of a fold point --/
def FoldPoint (t : Triangle) (P : ℝ × ℝ) : Prop :=
  sorry  -- Definition of fold point

/-- Set of all fold points of a triangle --/
def FoldPointSet (t : Triangle) : Set (ℝ × ℝ) :=
  {P | FoldPoint t P}

/-- Area of a set in ℝ² --/
noncomputable def Area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry  -- Definition of area

theorem fold_point_area (t : Triangle) : 
  t.A.1 = 0 ∧ t.A.2 = 0 ∧
  t.B.1 = 36 ∧ t.B.2 = 0 ∧
  t.C.1 = 0 ∧ t.C.2 = 72 →
  Area (FoldPointSet t) = 270 * Real.pi - 324 * Real.sqrt 3 :=
sorry

end fold_point_area_l3361_336174


namespace probability_of_red_ball_l3361_336125

theorem probability_of_red_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = 5)
  (h2 : red_balls = 3)
  (h3 : white_balls = 2)
  (h4 : total_balls = red_balls + white_balls) :
  (red_balls : ℚ) / total_balls = 3 / 5 := by
  sorry

end probability_of_red_ball_l3361_336125


namespace number_added_at_end_l3361_336196

theorem number_added_at_end (x : ℝ) : (26.3 * 12 * 20) / 3 + x = 2229 → x = 125 := by
  sorry

end number_added_at_end_l3361_336196


namespace solve_equation_l3361_336106

theorem solve_equation : 
  ∃ y : ℚ, (y^2 - 9*y + 8)/(y - 1) + (3*y^2 + 16*y - 12)/(3*y - 2) = -3 ∧ y = -1/2 := by
  sorry

end solve_equation_l3361_336106


namespace final_position_16_meters_l3361_336151

/-- Represents a back-and-forth race between two runners -/
structure Race where
  distance : ℝ  -- Total distance of the race (one way)
  meetPoint : ℝ  -- Distance from B to meeting point
  catchPoint : ℝ  -- Distance from finish when B catches A

/-- Calculates the final position of runner A when B finishes the race -/
def finalPositionA (race : Race) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the final position of A in the given race scenario -/
theorem final_position_16_meters (race : Race) 
  (h1 : race.meetPoint = 24)
  (h2 : race.catchPoint = 48) :
  finalPositionA race = 16 := by
  sorry

end final_position_16_meters_l3361_336151


namespace arithmetic_sequence_sum_l3361_336132

/-- An arithmetic sequence with first term 2 and the sum of the second and third terms equal to 13 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ 
  a 2 + a 3 = 13 ∧ 
  ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 4 + a 5 + a 6 = 42 := by
  sorry

end arithmetic_sequence_sum_l3361_336132


namespace prob_two_tails_proof_l3361_336155

/-- The probability of getting exactly 2 tails when tossing 3 fair coins -/
def prob_two_tails : ℚ := 3 / 8

/-- A fair coin has a probability of 1/2 for each outcome -/
def fair_coin (outcome : Bool) : ℚ := 1 / 2

/-- The number of possible outcomes when tossing 3 coins -/
def total_outcomes : ℕ := 2^3

/-- The number of outcomes with exactly 2 tails when tossing 3 coins -/
def favorable_outcomes : ℕ := 3

theorem prob_two_tails_proof :
  prob_two_tails = favorable_outcomes / total_outcomes :=
sorry

end prob_two_tails_proof_l3361_336155


namespace james_pays_37_50_l3361_336153

/-- Calculates the amount James pays for singing lessons -/
def james_payment (total_lessons : ℕ) (free_lessons : ℕ) (full_price_lessons : ℕ) 
  (lesson_cost : ℚ) (uncle_payment_fraction : ℚ) : ℚ :=
  let paid_lessons := total_lessons - free_lessons
  let discounted_lessons := paid_lessons - full_price_lessons
  let half_price_lessons := (discounted_lessons + 1) / 2
  let total_paid_lessons := full_price_lessons + half_price_lessons
  let total_cost := total_paid_lessons * lesson_cost
  (1 - uncle_payment_fraction) * total_cost

/-- Theorem stating that James pays $37.50 for his singing lessons -/
theorem james_pays_37_50 : 
  james_payment 20 1 10 5 (1/2) = 37.5 := by
  sorry

end james_pays_37_50_l3361_336153


namespace turtle_problem_l3361_336115

theorem turtle_problem (initial_turtles : ℕ) : initial_turtles = 9 →
  let additional_turtles := 3 * initial_turtles - 2
  let total_turtles := initial_turtles + additional_turtles
  let remaining_turtles := total_turtles / 2
  remaining_turtles = 17 := by
sorry

end turtle_problem_l3361_336115


namespace imaginary_part_of_z_l3361_336148

theorem imaginary_part_of_z (z : ℂ) (h : z * (3 - 4*I) = 5) : z.im = 5/7 := by
  sorry

end imaginary_part_of_z_l3361_336148


namespace smallest_four_digit_multiple_of_8_with_digit_sum_20_l3361_336162

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if the number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_multiple_of_8_with_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → n % 8 = 0 → digit_sum n = 20 → n ≥ 1071 :=
sorry

end smallest_four_digit_multiple_of_8_with_digit_sum_20_l3361_336162


namespace x_range_theorem_l3361_336170

theorem x_range_theorem (x : ℝ) : 
  (∀ a ∈ Set.Ioo 0 1, (a - 3) * x^2 < (4 * a - 2) * x) ↔ 
  (x ≤ -1 ∨ x ≥ 2/3) := by
sorry

end x_range_theorem_l3361_336170


namespace eggs_per_box_l3361_336146

theorem eggs_per_box (total_eggs : ℝ) (num_boxes : ℝ) 
  (h1 : total_eggs = 3.0) 
  (h2 : num_boxes = 2.0) 
  (h3 : num_boxes ≠ 0) : 
  total_eggs / num_boxes = 1.5 := by
  sorry

end eggs_per_box_l3361_336146


namespace folded_rope_length_l3361_336188

/-- Represents the length of a rope folded three times -/
structure FoldedRope where
  total_length : ℝ
  distance_1_3 : ℝ

/-- The properties of a rope folded three times as described in the problem -/
def is_valid_folded_rope (rope : FoldedRope) : Prop :=
  rope.distance_1_3 = rope.total_length / 4

/-- The main theorem stating the relationship between the distance between points (1) and (3)
    and the total length of the rope -/
theorem folded_rope_length (rope : FoldedRope) 
  (h : is_valid_folded_rope rope) 
  (h_distance : rope.distance_1_3 = 30) : 
  rope.total_length = 120 := by
  sorry

#check folded_rope_length

end folded_rope_length_l3361_336188


namespace hexagon_area_lower_bound_l3361_336181

/-- Triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- Area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Area of the hexagon formed by extending the sides of the triangle -/
def hexagon_area (t : Triangle) : ℝ := sorry

/-- The area of the hexagon is at least 13 times the area of the triangle -/
theorem hexagon_area_lower_bound (t : Triangle) :
  hexagon_area t ≥ 13 * area t := by
  sorry

end hexagon_area_lower_bound_l3361_336181


namespace fraction_simplification_l3361_336179

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end fraction_simplification_l3361_336179


namespace smaller_root_of_quadratic_l3361_336142

theorem smaller_root_of_quadratic (x : ℝ) : 
  (x + 1) * (x - 1) = 0 → x = -1 ∨ x = 1 → -1 ≤ 1 → -1 = min x (-x) := by
sorry

end smaller_root_of_quadratic_l3361_336142


namespace police_speed_l3361_336186

/-- Proves that the speed of a police officer chasing a thief is 40 km/hr given specific conditions --/
theorem police_speed (thief_speed : ℝ) (police_station_distance : ℝ) (police_delay : ℝ) (catch_time : ℝ) :
  thief_speed = 20 →
  police_station_distance = 60 →
  police_delay = 1 →
  catch_time = 4 →
  (police_station_distance + thief_speed * (police_delay + catch_time)) / catch_time = 40 :=
by
  sorry


end police_speed_l3361_336186


namespace minimal_points_double_star_l3361_336102

/-- Represents a regular n-pointed double star polygon -/
structure DoubleStarPolygon where
  n : ℕ
  angleA : ℝ
  angleB : ℝ

/-- Conditions for a valid double star polygon -/
def isValidDoubleStarPolygon (d : DoubleStarPolygon) : Prop :=
  d.n > 0 ∧
  d.angleA > 0 ∧
  d.angleB > 0 ∧
  d.angleA = d.angleB + 15 ∧
  d.n * 15 = 360

theorem minimal_points_double_star :
  ∀ d : DoubleStarPolygon, isValidDoubleStarPolygon d → d.n ≥ 24 :=
by sorry

end minimal_points_double_star_l3361_336102


namespace z_squared_in_first_quadrant_l3361_336108

theorem z_squared_in_first_quadrant (z : ℂ) (h : (z - I) / (1 + I) = 2 - 2*I) :
  (z^2).re > 0 ∧ (z^2).im > 0 :=
by sorry

end z_squared_in_first_quadrant_l3361_336108


namespace quadratic_inequality_range_l3361_336197

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

-- State the theorem
theorem quadratic_inequality_range :
  ∀ a : ℝ, (¬ ∃ x : ℝ, f a x ≤ 0) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

end quadratic_inequality_range_l3361_336197


namespace correct_guess_and_multiply_l3361_336156

def coin_head_prob : ℚ := 2/3
def aaron_head_guess_prob : ℚ := 2/3

def correct_guess_prob : ℚ := 
  coin_head_prob * aaron_head_guess_prob + (1 - coin_head_prob) * (1 - aaron_head_guess_prob)

theorem correct_guess_and_multiply :
  correct_guess_prob = 5/9 ∧ 9000 * correct_guess_prob = 5000 := by sorry

end correct_guess_and_multiply_l3361_336156


namespace rounding_shift_l3361_336182

/-- Rounding function that rounds to the nearest integer -/
noncomputable def f (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 1/2 then ⌊x⌋ else ⌈x⌉

/-- Theorem stating that adding an integer to the input of f
    is equivalent to adding the same integer to the output of f -/
theorem rounding_shift (x : ℝ) (m : ℤ) : f (x + m) = f x + m := by
  sorry

end rounding_shift_l3361_336182


namespace systematic_sampling_problem_l3361_336110

/-- Systematic sampling problem -/
theorem systematic_sampling_problem 
  (total_students : ℕ) 
  (num_groups : ℕ) 
  (group_size : ℕ) 
  (sample_size : ℕ) 
  (sixteenth_group_num : ℕ) :
  total_students = 160 →
  num_groups = 20 →
  group_size = 8 →
  sample_size = 20 →
  sixteenth_group_num = 126 →
  ∃ (first_group_num : ℕ), 
    first_group_num + (15 * group_size) = sixteenth_group_num ∧
    first_group_num = 6 :=
by sorry

end systematic_sampling_problem_l3361_336110


namespace twenty_five_percent_less_than_80_l3361_336100

theorem twenty_five_percent_less_than_80 (x : ℚ) : x + (1/4) * x = 60 → x = 48 := by
  sorry

end twenty_five_percent_less_than_80_l3361_336100


namespace monotonic_at_most_one_zero_l3361_336166

/-- A function f: ℝ → ℝ is monotonic if for all x₁ < x₂, either f(x₁) ≤ f(x₂) or f(x₁) ≥ f(x₂) -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → (f x₁ ≤ f x₂ ∨ f x₁ ≥ f x₂)

/-- A real number x is a zero of f if f(x) = 0 -/
def IsZero (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

/-- The number of zeros of f is at most one -/
def AtMostOneZero (f : ℝ → ℝ) : Prop :=
  ∀ x y, IsZero f x → IsZero f y → x = y

theorem monotonic_at_most_one_zero (f : ℝ → ℝ) (h : Monotonic f) : AtMostOneZero f := by
  sorry

end monotonic_at_most_one_zero_l3361_336166


namespace triangle_area_l3361_336152

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  b + c = 5 →
  Real.tan B + Real.tan C + Real.sqrt 3 = Real.sqrt 3 * Real.tan B * Real.tan C →
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 4 :=
by sorry

end triangle_area_l3361_336152


namespace functional_equation_solution_l3361_336113

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y = f (x - y)) →
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) :=
by sorry

end functional_equation_solution_l3361_336113


namespace system_properties_l3361_336140

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  (x + 3 * y = 4 - a) ∧ (x - y = 3 * a)

-- Theorem statement
theorem system_properties :
  ∀ (x y a : ℝ), system x y a →
    ((x + y = 0) → (a = -2)) ∧
    (x + 2 * y = 3) ∧
    (y = -x / 2 + 3 / 2) :=
by sorry

end system_properties_l3361_336140


namespace unique_b_value_l3361_336191

theorem unique_b_value (b h a : ℕ) (hb_pos : 0 < b) (hh_pos : 0 < h) (hb_lt_h : b < h)
  (heq : b^2 + h^2 = b*(a + h) + a*h) : b = 2 := by
  sorry

end unique_b_value_l3361_336191


namespace min_value_a_l3361_336161

theorem min_value_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, |x - a| + |1 - x| ≥ 1) : 
  (∀ b : ℝ, b > 0 ∧ (∀ x : ℝ, |x - b| + |1 - x| ≥ 1) → b ≥ 2) ∧ 
  (∃ c : ℝ, c > 0 ∧ (∀ x : ℝ, |x - c| + |1 - x| ≥ 1) ∧ c = 2) :=
sorry

end min_value_a_l3361_336161


namespace students_in_multiple_activities_l3361_336154

theorem students_in_multiple_activities 
  (total_students : ℕ) 
  (debate_only : ℕ) 
  (singing_only : ℕ) 
  (dance_only : ℕ) 
  (no_activity : ℕ) 
  (h1 : total_students = 55)
  (h2 : debate_only = 10)
  (h3 : singing_only = 18)
  (h4 : dance_only = 8)
  (h5 : no_activity = 5) :
  total_students - (debate_only + singing_only + dance_only + no_activity) = 14 := by
  sorry

end students_in_multiple_activities_l3361_336154


namespace spa_nail_polish_l3361_336101

/-- The number of girls who went to the spa -/
def num_girls : ℕ := 8

/-- The number of fingers on each girl's hands -/
def fingers_per_girl : ℕ := 10

/-- The number of toes on each girl's feet -/
def toes_per_girl : ℕ := 10

/-- The total number of digits polished at the spa -/
def total_digits_polished : ℕ := num_girls * (fingers_per_girl + toes_per_girl)

theorem spa_nail_polish :
  total_digits_polished = 160 := by sorry

end spa_nail_polish_l3361_336101


namespace director_selection_probability_l3361_336147

def total_actors : ℕ := 5
def golden_rooster_winners : ℕ := 2
def hundred_flowers_winners : ℕ := 3

def probability_select_2_golden_1_hundred : ℚ := 3 / 10

theorem director_selection_probability :
  (golden_rooster_winners.choose 2 * hundred_flowers_winners) / 
  (total_actors.choose 3) = probability_select_2_golden_1_hundred := by
  sorry

end director_selection_probability_l3361_336147


namespace sum_of_repeating_decimals_l3361_336107

def repeating_decimal_123 : ℚ := 123 / 999
def repeating_decimal_0123 : ℚ := 123 / 9999
def repeating_decimal_000123 : ℚ := 123 / 999999

theorem sum_of_repeating_decimals :
  repeating_decimal_123 + repeating_decimal_0123 + repeating_decimal_000123 =
  (123 * 1000900) / (999 * 9999 * 100001) := by
  sorry

end sum_of_repeating_decimals_l3361_336107


namespace smallest_prime_12_less_than_square_l3361_336193

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- Main theorem
theorem smallest_prime_12_less_than_square : 
  ∀ n : ℕ, n > 0 → is_prime n → (∃ m : ℕ, is_perfect_square m ∧ n = m - 12) → n ≥ 13 :=
sorry

end smallest_prime_12_less_than_square_l3361_336193


namespace sum_of_powers_of_i_is_zero_l3361_336171

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_is_zero :
  i^23456 + i^23457 + i^23458 + i^23459 = 0 :=
by sorry

end sum_of_powers_of_i_is_zero_l3361_336171


namespace floor_neg_sqrt_64_over_9_l3361_336104

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64 / 9)⌋ = -3 := by sorry

end floor_neg_sqrt_64_over_9_l3361_336104


namespace girls_count_in_school_l3361_336144

theorem girls_count_in_school (total_students : ℕ) (boys_avg_age girls_avg_age school_avg_age : ℚ) : 
  total_students = 652 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  school_avg_age = 11.75 →
  ∃ (girls_count : ℕ), 
    girls_count = 162 ∧ 
    (total_students - girls_count) * boys_avg_age + girls_count * girls_avg_age = total_students * school_avg_age :=
by sorry

end girls_count_in_school_l3361_336144


namespace garden_area_difference_l3361_336157

/-- Represents a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def garden_area (g : Garden) : ℝ := g.length * g.width

/-- Represents a shed in the garden -/
structure Shed where
  length : ℝ
  width : ℝ

/-- Calculates the area of a shed -/
def shed_area (s : Shed) : ℝ := s.length * s.width

theorem garden_area_difference : 
  let karl_garden : Garden := { length := 30, width := 50 }
  let makenna_garden : Garden := { length := 35, width := 55 }
  let makenna_shed : Shed := { length := 5, width := 10 }
  (garden_area makenna_garden - shed_area makenna_shed) - garden_area karl_garden = 375 := by
  sorry

end garden_area_difference_l3361_336157


namespace min_zeros_odd_periodic_function_l3361_336133

/-- A function f: ℝ → ℝ is odd -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ has period p -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem min_zeros_odd_periodic_function
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_period : HasPeriod f 3)
  (h_f2 : f 2 = 0) :
  ∃ (S : Finset ℝ), S.card ≥ 7 ∧ (∀ x ∈ S, 0 < x ∧ x < 6 ∧ f x = 0) :=
sorry

end min_zeros_odd_periodic_function_l3361_336133


namespace bus_time_calculation_l3361_336184

def minutes_in_day : ℕ := 24 * 60

def time_to_minutes (hours minutes : ℕ) : ℕ :=
  hours * 60 + minutes

theorem bus_time_calculation (leave_home arrive_bus arrive_home class_duration num_classes other_activities : ℕ) :
  leave_home = time_to_minutes 7 0 →
  arrive_bus = time_to_minutes 7 45 →
  arrive_home = time_to_minutes 17 15 →
  class_duration = 55 →
  num_classes = 8 →
  other_activities = time_to_minutes 1 45 →
  arrive_home - leave_home - (class_duration * num_classes + other_activities) = 25 := by
  sorry

#check bus_time_calculation

end bus_time_calculation_l3361_336184


namespace no_spiky_two_digit_integers_l3361_336134

/-- A two-digit positive integer is spiky if it equals the sum of its tens digit and 
    the cube of its units digit subtracted by twice the tens digit. -/
def IsSpiky (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ 
  ∃ a b : ℕ, n = 10 * a + b ∧ 
             n = a + b^3 - 2*a

/-- There are no spiky two-digit positive integers. -/
theorem no_spiky_two_digit_integers : ¬∃ n : ℕ, IsSpiky n := by
  sorry

#check no_spiky_two_digit_integers

end no_spiky_two_digit_integers_l3361_336134


namespace consecutive_primes_expression_l3361_336149

theorem consecutive_primes_expression (p q : ℕ) : 
  Prime p → Prime q → p < q → p.succ = q → (p : ℚ) / q = 4 / 5 → 
  25 / 7 + ((2 * q - p) : ℚ) / (2 * q + p) = 4 := by
sorry

end consecutive_primes_expression_l3361_336149


namespace potato_cost_theorem_l3361_336112

-- Define the given conditions
def people_count : ℕ := 40
def potatoes_per_person : ℚ := 3/2
def bag_weight : ℕ := 20
def bag_cost : ℕ := 5

-- Define the theorem
theorem potato_cost_theorem : 
  (people_count : ℚ) * potatoes_per_person / bag_weight * bag_cost = 15 := by
  sorry

end potato_cost_theorem_l3361_336112


namespace angle_D_measure_l3361_336130

theorem angle_D_measure (A B C D : ℝ) : 
  -- ABCD is a convex quadrilateral
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  0 < D ∧ D < π ∧
  A + B + C + D = 2 * π ∧
  -- ∠C = 57°
  C = 57 * π / 180 ∧
  -- sin ∠A + sin ∠B = √2
  Real.sin A + Real.sin B = Real.sqrt 2 ∧
  -- cos ∠A + cos ∠B = 2 - √2
  Real.cos A + Real.cos B = 2 - Real.sqrt 2
  -- Then ∠D = 168°
  → D = 168 * π / 180 := by
sorry

end angle_D_measure_l3361_336130


namespace modular_congruence_13_pow_6_mod_11_l3361_336164

theorem modular_congruence_13_pow_6_mod_11 : 
  ∃ m : ℕ, 13^6 ≡ m [ZMOD 11] ∧ m < 11 → m = 9 := by
  sorry

end modular_congruence_13_pow_6_mod_11_l3361_336164


namespace sum_of_solutions_is_zero_l3361_336163

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (18 * x₁) / 27 = 7 / x₁ ∧ 
  (18 * x₂) / 27 = 7 / x₂ ∧ 
  x₁ + x₂ = 0 := by
sorry

end sum_of_solutions_is_zero_l3361_336163


namespace line_segments_and_midpoints_l3361_336160

/-- The number of line segments that can be formed with n points on a line -/
def num_segments (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of unique midpoints of line segments formed by n points on a line -/
def num_midpoints (n : ℕ) : ℕ := 2 * n - 3

/-- The number of points on the line -/
def num_points : ℕ := 10

theorem line_segments_and_midpoints :
  num_segments num_points = 45 ∧ num_midpoints num_points = 17 := by
  sorry

end line_segments_and_midpoints_l3361_336160


namespace boat_speed_in_still_water_l3361_336109

/-- 
Given a boat that covers the same distance downstream and upstream,
with known travel times and stream speed, this theorem proves
the speed of the boat in still water.
-/
theorem boat_speed_in_still_water 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (stream_speed : ℝ) 
  (h1 : downstream_time = 1)
  (h2 : upstream_time = 1.5)
  (h3 : stream_speed = 3) : 
  ∃ (boat_speed : ℝ), boat_speed = 15 ∧ 
    downstream_time * (boat_speed + stream_speed) = 
    upstream_time * (boat_speed - stream_speed) :=
by sorry

end boat_speed_in_still_water_l3361_336109


namespace blocks_remaining_problem_l3361_336131

/-- Given a person with an initial number of blocks and a number of blocks used,
    calculate the remaining number of blocks. -/
def remaining_blocks (initial : ℕ) (used : ℕ) : ℕ :=
  initial - used

/-- Theorem stating that given 78 initial blocks and 19 used blocks,
    the remaining number of blocks is 59. -/
theorem blocks_remaining_problem :
  remaining_blocks 78 19 = 59 := by
  sorry

end blocks_remaining_problem_l3361_336131


namespace school_year_work_hours_l3361_336180

/-- Amy's work schedule and earnings -/
structure WorkSchedule where
  summer_weeks : ℕ
  summer_hours_per_week : ℕ
  summer_earnings : ℕ
  school_year_weeks : ℕ
  school_year_target_earnings : ℕ

/-- Calculate the required hours per week during school year -/
def required_school_year_hours_per_week (schedule : WorkSchedule) : ℚ :=
  let hourly_wage : ℚ := schedule.summer_earnings / (schedule.summer_weeks * schedule.summer_hours_per_week)
  let total_hours_needed : ℚ := schedule.school_year_target_earnings / hourly_wage
  total_hours_needed / schedule.school_year_weeks

/-- Theorem stating the required hours per week during school year -/
theorem school_year_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.summer_weeks = 8)
  (h2 : schedule.summer_hours_per_week = 40)
  (h3 : schedule.summer_earnings = 3200)
  (h4 : schedule.school_year_weeks = 32)
  (h5 : schedule.school_year_target_earnings = 4000) :
  required_school_year_hours_per_week schedule = 12.5 := by
  sorry


end school_year_work_hours_l3361_336180


namespace consecutive_integers_product_plus_one_is_perfect_square_l3361_336178

theorem consecutive_integers_product_plus_one_is_perfect_square (n : ℤ) :
  ∃ m : ℤ, (n - 1) * n * (n + 1) * (n + 2) + 1 = m ^ 2 := by
  sorry

end consecutive_integers_product_plus_one_is_perfect_square_l3361_336178


namespace bob_pennies_l3361_336141

theorem bob_pennies (a b : ℕ) : 
  (b + 2 = 4 * (a - 2)) →
  (b - 2 = 3 * (a + 2)) →
  b = 62 :=
by sorry

end bob_pennies_l3361_336141


namespace sqrt_23_parts_x_minus_y_value_l3361_336187

-- Part 1: Integer and decimal parts of √23
theorem sqrt_23_parts :
  ∃ (n : ℕ) (d : ℝ), n = 4 ∧ d = Real.sqrt 23 - 4 ∧
  Real.sqrt 23 = n + d ∧ 0 ≤ d ∧ d < 1 := by sorry

-- Part 2: x-y given 9+√3=x+y
theorem x_minus_y_value (x : ℤ) (y : ℝ) 
  (h1 : 9 + Real.sqrt 3 = x + y)
  (h2 : 0 < y) (h3 : y < 1) :
  x - y = 11 - Real.sqrt 3 := by sorry

end sqrt_23_parts_x_minus_y_value_l3361_336187


namespace megans_files_l3361_336123

theorem megans_files (deleted_files : ℕ) (files_per_folder : ℕ) (num_folders : ℕ) :
  deleted_files = 21 →
  files_per_folder = 8 →
  num_folders = 9 →
  deleted_files + (files_per_folder * num_folders) = 93 :=
by sorry

end megans_files_l3361_336123


namespace infinite_primes_quadratic_equation_l3361_336135

theorem infinite_primes_quadratic_equation :
  ∀ (S : Finset Nat), ∃ (p : Nat) (x y : Int),
    Prime p ∧ p ∉ S ∧ x^2 + x + 1 = p * y := by
  sorry

end infinite_primes_quadratic_equation_l3361_336135


namespace integral_equality_l3361_336145

theorem integral_equality : ∫ (x : ℝ) in Set.Icc π (2*π), (1 - Real.cos x) / (x - Real.sin x)^2 = 1 / (2*π) := by
  sorry

end integral_equality_l3361_336145


namespace min_triangle_count_l3361_336105

structure Graph (n : ℕ) :=
  (m : ℕ)
  (edges : Finset (Fin n × Fin n))
  (edge_count : edges.card = m)
  (edge_distinct : ∀ (e : Fin n × Fin n), e ∈ edges → e.1 ≠ e.2)

def triangle_count (n : ℕ) (G : Graph n) : ℕ := sorry

theorem min_triangle_count (n : ℕ) (G : Graph n) :
  triangle_count n G ≥ (4 * G.m : ℚ) / (3 * n) * (G.m - n^2 / 4) :=
sorry

end min_triangle_count_l3361_336105


namespace max_product_2015_l3361_336120

/-- Given the digits 2, 0, 1, and 5, the maximum product obtained by rearranging
    these digits into two numbers and multiplying them is 1050. -/
theorem max_product_2015 : ∃ (a b : ℕ),
  (a ≤ 99 ∧ b ≤ 99) ∧
  (∀ (d : ℕ), d ∈ [a.div 10, a % 10, b.div 10, b % 10] → d ∈ [2, 0, 1, 5]) ∧
  (a * b = 1050) ∧
  (∀ (c d : ℕ), c ≤ 99 → d ≤ 99 →
    (∀ (e : ℕ), e ∈ [c.div 10, c % 10, d.div 10, d % 10] → e ∈ [2, 0, 1, 5]) →
    c * d ≤ 1050) :=
by sorry

end max_product_2015_l3361_336120


namespace mrs_franklin_valentines_l3361_336124

theorem mrs_franklin_valentines (given_away : ℕ) (left : ℕ) 
  (h1 : given_away = 42) (h2 : left = 16) : 
  given_away + left = 58 := by
  sorry

end mrs_franklin_valentines_l3361_336124


namespace complex_product_real_l3361_336114

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 3 - 2*I
  let z₂ : ℂ := 1 + a*I
  (z₁ * z₂).im = 0 → a = 2/3 := by sorry

end complex_product_real_l3361_336114


namespace complex_sum_equality_l3361_336150

theorem complex_sum_equality : 
  8 * Complex.exp (2 * π * I / 13) + 8 * Complex.exp (15 * π * I / 26) = 
  8 * Real.sqrt 3 * Complex.exp (19 * π * I / 52) := by sorry

end complex_sum_equality_l3361_336150


namespace square_sum_geq_product_l3361_336176

theorem square_sum_geq_product (x y z : ℝ) : x + y + z ≥ x * y * z → x^2 + y^2 + z^2 ≥ x * y * z := by
  sorry

end square_sum_geq_product_l3361_336176


namespace determinant_zero_l3361_336126

-- Define the cubic equation
def cubic_equation (x s t : ℝ) : Prop := x^3 + s*x^2 + t*x = 0

-- Define the determinant of the 3x3 matrix
def matrix_determinant (x y z : ℝ) : ℝ :=
  x * (z * y - x * x) - y * (y * y - x * z) + z * (y * z - z * x)

-- Theorem statement
theorem determinant_zero (x y z s t : ℝ) 
  (hx : cubic_equation x s t) 
  (hy : cubic_equation y s t) 
  (hz : cubic_equation z s t) : 
  matrix_determinant x y z = 0 := by sorry

end determinant_zero_l3361_336126


namespace qin_jiushao_evaluation_l3361_336183

-- Define the polynomial coefficients
def a₀ : ℝ := 12
def a₁ : ℝ := 35
def a₂ : ℝ := -8
def a₃ : ℝ := 79
def a₄ : ℝ := 6
def a₅ : ℝ := 5
def a₆ : ℝ := 3

-- Define the evaluation point
def x : ℝ := -4

-- Define Qin Jiushao's algorithm
def qin_jiushao (a : Fin 7 → ℝ) (x : ℝ) : ℝ :=
  (((((a 6 * x + a 5) * x + a 4) * x + a 3) * x + a 2) * x + a 1) * x + a 0

-- Theorem statement
theorem qin_jiushao_evaluation :
  qin_jiushao (fun i => [a₀, a₁, a₂, a₃, a₄, a₅, a₆].get i) x = 220 := by
  sorry

end qin_jiushao_evaluation_l3361_336183


namespace triangle_medians_inequalities_l3361_336118

-- Define a structure for a triangle with medians and circumradius
structure Triangle where
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  R : ℝ
  h_positive : m_a > 0 ∧ m_b > 0 ∧ m_c > 0 ∧ R > 0

-- Theorem statement
theorem triangle_medians_inequalities (t : Triangle) : 
  t.m_a^2 + t.m_b^2 + t.m_c^2 ≤ (27 * t.R^2) / 4 ∧ 
  t.m_a + t.m_b + t.m_c ≤ (9 * t.R) / 2 := by
  sorry


end triangle_medians_inequalities_l3361_336118


namespace xy_sum_is_two_l3361_336177

theorem xy_sum_is_two (x y : ℝ) 
  (hx : (x - 1)^3 + 1997*(x - 1) = -1) 
  (hy : (y - 1)^3 + 1997*(y - 1) = 1) : 
  x + y = 2 := by sorry

end xy_sum_is_two_l3361_336177


namespace line_segment_parameter_sum_squares_l3361_336195

/-- Given a line segment connecting (1, -3) and (4, 9), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that a^2 + b^2 + c^2 + d^2 = 163 -/
theorem line_segment_parameter_sum_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = 1 ∧ d = -3) →
  (a + b = 4 ∧ c + d = 9) →
  a^2 + b^2 + c^2 + d^2 = 163 := by
sorry

end line_segment_parameter_sum_squares_l3361_336195


namespace cookie_eating_contest_l3361_336111

theorem cookie_eating_contest (first_friend second_friend : ℚ) 
  (h1 : first_friend = 5/6)
  (h2 : second_friend = 2/3) :
  first_friend - second_friend = 1/6 := by
  sorry

end cookie_eating_contest_l3361_336111
