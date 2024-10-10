import Mathlib

namespace right_triangle_side_length_l293_29338

theorem right_triangle_side_length 
  (Q R S : ℝ × ℝ) 
  (right_angle_Q : (R.1 - Q.1) * (S.1 - Q.1) + (R.2 - Q.2) * (S.2 - Q.2) = 0) 
  (cos_R : (R.1 - Q.1) / Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 4/9)
  (RS_length : Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 9) :
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = Real.sqrt 65 := by
  sorry

end right_triangle_side_length_l293_29338


namespace extremum_of_f_l293_29337

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_of_f (a b : ℝ) :
  (f' a b 2 = 0) →  -- Extremum at x=2
  (f' a b 1 = -3) →  -- Tangent line at x=1 parallel to y=-3x-2
  (∃ x, f a b x = -4 ∧ ∀ y, f a b y ≥ f a b x) :=
by
  sorry

end extremum_of_f_l293_29337


namespace problem_statement_l293_29395

theorem problem_statement (a b : ℝ) : 
  (a + b + 1 = -2) → (a + b - 1) * (1 - a - b) = -16 := by sorry

end problem_statement_l293_29395


namespace race_distance_l293_29306

/-- Prove that the total distance of a race is 240 meters given the specified conditions -/
theorem race_distance (D : ℝ) 
  (h1 : D / 60 * 100 = D + 160) : D = 240 := by
  sorry

#check race_distance

end race_distance_l293_29306


namespace squirrel_acorns_l293_29348

theorem squirrel_acorns (num_squirrels : ℕ) (acorns_collected : ℕ) (acorns_needed_per_squirrel : ℕ) :
  num_squirrels = 5 →
  acorns_collected = 575 →
  acorns_needed_per_squirrel = 130 →
  (num_squirrels * acorns_needed_per_squirrel - acorns_collected) / num_squirrels = 15 :=
by sorry

end squirrel_acorns_l293_29348


namespace arithmetic_progression_with_prime_factor_constraint_l293_29370

theorem arithmetic_progression_with_prime_factor_constraint :
  ∀ (a b c : ℕ), 
    0 < a → a < b → b < c →
    b - a = c - b →
    (∀ p : ℕ, Prime p → p > 3 → (p ∣ a ∨ p ∣ b ∨ p ∣ c) → False) →
    ∃ (k m n : ℕ), 
      (a = k ∧ b = 2*k ∧ c = 3*k) ∨
      (a = 2*k ∧ b = 3*k ∧ c = 4*k) ∨
      (a = 2*k ∧ b = 9*k ∧ c = 16*k) ∧
      k = 2^m * 3^n :=
by sorry

end arithmetic_progression_with_prime_factor_constraint_l293_29370


namespace james_barbell_cost_l293_29330

/-- The final cost of James' new barbell purchase -/
def final_barbell_cost (old_barbell_cost : ℝ) (price_increase_rate : ℝ) 
  (sales_tax_rate : ℝ) (trade_in_value : ℝ) : ℝ :=
  let new_barbell_cost := old_barbell_cost * (1 + price_increase_rate)
  let total_cost_with_tax := new_barbell_cost * (1 + sales_tax_rate)
  total_cost_with_tax - trade_in_value

/-- Theorem stating the final cost of James' new barbell -/
theorem james_barbell_cost : 
  final_barbell_cost 250 0.30 0.10 100 = 257.50 := by
  sorry

end james_barbell_cost_l293_29330


namespace camel_cost_l293_29371

/-- The cost relationship between animals and the cost of a camel --/
theorem camel_cost (camel horse ox elephant : ℝ) 
  (h1 : 10 * camel = 24 * horse)
  (h2 : 16 * horse = 4 * ox)
  (h3 : 6 * ox = 4 * elephant)
  (h4 : 10 * elephant = 120000) :
  camel = 4800 := by
  sorry

end camel_cost_l293_29371


namespace gcd_78_182_l293_29331

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := by
  sorry

end gcd_78_182_l293_29331


namespace mean_of_remaining_numbers_l293_29308

def numbers : List ℕ := [1871, 1997, 2020, 2028, 2113, 2125, 2140, 2222, 2300]

theorem mean_of_remaining_numbers :
  (∃ (subset : List ℕ), subset.length = 7 ∧ subset.sum / 7 = 2100 ∧ subset.toFinset ⊆ numbers.toFinset) →
  (numbers.sum - (2100 * 7)) / 2 = 1158 := by
sorry

end mean_of_remaining_numbers_l293_29308


namespace product_of_roots_l293_29365

theorem product_of_roots (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2017 ∧ y₁^3 - 3*x₁^2*y₁ = 2016)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2017 ∧ y₂^3 - 3*x₂^2*y₂ = 2016)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2017 ∧ y₃^3 - 3*x₃^2*y₃ = 2016) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1008 := by
  sorry

end product_of_roots_l293_29365


namespace price_decrease_l293_29344

/-- Given an article with an original price of 700 rupees and a price decrease of 24%,
    the new price after the decrease is 532 rupees. -/
theorem price_decrease (original_price : ℝ) (decrease_percentage : ℝ) (new_price : ℝ) :
  original_price = 700 →
  decrease_percentage = 24 →
  new_price = original_price * (1 - decrease_percentage / 100) →
  new_price = 532 := by
sorry

end price_decrease_l293_29344


namespace quadratic_inequality_no_solution_l293_29363

theorem quadratic_inequality_no_solution (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - m * x + (m - 1) ≤ 0) ↔ m ≤ -2 * Real.sqrt 3 / 3 :=
sorry

end quadratic_inequality_no_solution_l293_29363


namespace men_per_table_l293_29310

/-- Given a restaurant scenario, prove the number of men at each table. -/
theorem men_per_table (num_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) : 
  num_tables = 6 → women_per_table = 3 → total_customers = 48 → 
  (total_customers - num_tables * women_per_table) / num_tables = 5 := by
  sorry

end men_per_table_l293_29310


namespace f_properties_l293_29368

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^2 + 12 * x - 15

-- Theorem statement
theorem f_properties :
  -- 1. Zeros of f(x)
  (∃ x : ℝ, f x = 0 ↔ x = -5 ∨ x = 1) ∧
  -- 2. Minimum and maximum values on [-3, 3]
  (∀ x : ℝ, x ∈ Set.Icc (-3) 3 → f x ≥ -27) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ f x = -27) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-3) 3 → f x ≤ 48) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ f x = 48) ∧
  -- 3. f(x) is increasing on [-2, +∞)
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ici (-2) → x₂ ∈ Set.Ici (-2) → x₁ < x₂ → f x₁ < f x₂) :=
by sorry


end f_properties_l293_29368


namespace remainder_problem_l293_29383

theorem remainder_problem : 123456789012 % 200 = 12 := by
  sorry

end remainder_problem_l293_29383


namespace root_product_equals_twenty_l293_29343

theorem root_product_equals_twenty :
  (32 : ℝ) ^ (1/5) * (16 : ℝ) ^ (1/4) * (25 : ℝ) ^ (1/2) = 20 := by
  sorry

end root_product_equals_twenty_l293_29343


namespace total_cases_giving_one_card_l293_29346

def blue_cards : ℕ := 3
def yellow_cards : ℕ := 5

theorem total_cases_giving_one_card : blue_cards + yellow_cards = 8 := by
  sorry

end total_cases_giving_one_card_l293_29346


namespace eighth_term_of_specific_sequence_l293_29362

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  firstTerm : ℚ
  lastTerm : ℚ
  numTerms : ℕ

/-- Calculates the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  let commonDiff := (seq.lastTerm - seq.firstTerm) / (seq.numTerms - 1)
  seq.firstTerm + (n - 1) * commonDiff

/-- Theorem: The 8th term of the specified arithmetic sequence is 731/29 -/
theorem eighth_term_of_specific_sequence :
  let seq : ArithmeticSequence := ⟨3, 95, 30⟩
  nthTerm seq 8 = 731 / 29 := by
  sorry

end eighth_term_of_specific_sequence_l293_29362


namespace limit_fraction_powers_three_five_l293_29305

/-- The limit of (3^n + 5^n) / (3^(n-1) + 5^(n-1)) as n approaches infinity is 5 -/
theorem limit_fraction_powers_three_five :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |((3 : ℝ)^n + 5^n) / ((3 : ℝ)^(n-1) + 5^(n-1)) - 5| < ε :=
sorry

end limit_fraction_powers_three_five_l293_29305


namespace angle_around_point_l293_29381

theorem angle_around_point (a b : ℝ) (h1 : a + b + 200 = 360) (h2 : a = b) : a = 80 := by
  sorry

end angle_around_point_l293_29381


namespace intersection_of_A_and_B_l293_29317

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l293_29317


namespace dog_length_calculation_l293_29382

/-- Represents the length of a dog's body parts in inches -/
structure DogMeasurements where
  tail_length : ℝ
  body_length : ℝ
  head_length : ℝ

/-- Calculates the overall length of a dog given its measurements -/
def overall_length (d : DogMeasurements) : ℝ :=
  d.body_length + d.head_length

/-- Theorem stating the overall length of a dog with specific proportions -/
theorem dog_length_calculation (d : DogMeasurements) 
  (h1 : d.tail_length = d.body_length / 2)
  (h2 : d.head_length = d.body_length / 6)
  (h3 : d.tail_length = 9) :
  overall_length d = 21 := by
  sorry

#check dog_length_calculation

end dog_length_calculation_l293_29382


namespace polynomial_characterization_l293_29318

-- Define S(k) as the sum of digits of k in decimal representation
def S (k : ℕ) : ℕ := sorry

-- Define the property that P(x) must satisfy
def satisfies_property (P : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2016 → (S (P n) = P (S n) ∧ P n > 0)

-- Define the set of valid polynomials
def valid_polynomial (P : ℕ → ℕ) : Prop :=
  (∃ c : ℕ, c ≥ 1 ∧ c ≤ 9 ∧ (∀ x : ℕ, P x = c)) ∨
  (∀ x : ℕ, P x = x)

-- Theorem statement
theorem polynomial_characterization :
  ∀ P : ℕ → ℕ, satisfies_property P → valid_polynomial P :=
sorry

end polynomial_characterization_l293_29318


namespace cylinder_cross_section_area_l293_29320

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents the cross-section area of a sliced cylinder -/
def crossSectionArea (c : Cylinder) (arcAngle : ℝ) : ℝ :=
  sorry

theorem cylinder_cross_section_area :
  let c : Cylinder := { radius := 8, height := 5 }
  let arcAngle : ℝ := 90 * π / 180  -- 90 degrees in radians
  crossSectionArea c arcAngle = 16 * π * Real.sqrt 2 + 32 := by
  sorry

end cylinder_cross_section_area_l293_29320


namespace syllogism_validity_l293_29354

theorem syllogism_validity (a b c : Prop) : 
  ((b → c) ∧ (a → b)) → (a → c) := by sorry

end syllogism_validity_l293_29354


namespace simple_interest_problem_l293_29302

/-- Proves that given a sum P at simple interest for 10 years, 
    if increasing the interest rate by 3% results in $300 more interest, 
    then P = $1000. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 3) * 10 / 100 - P * R * 10 / 100 = 300) → P = 1000 := by
  sorry

end simple_interest_problem_l293_29302


namespace consecutive_integers_product_336_sum_21_l293_29329

theorem consecutive_integers_product_336_sum_21 :
  ∃ (x : ℤ), (x * (x + 1) * (x + 2) = 336) ∧ (x + (x + 1) + (x + 2) = 21) := by
  sorry

end consecutive_integers_product_336_sum_21_l293_29329


namespace pencils_given_to_dorothy_l293_29316

theorem pencils_given_to_dorothy (initial_pencils : ℕ) (remaining_pencils : ℕ) 
  (h1 : initial_pencils = 142) 
  (h2 : remaining_pencils = 111) : 
  initial_pencils - remaining_pencils = 31 := by
  sorry

end pencils_given_to_dorothy_l293_29316


namespace sum_of_fractions_l293_29347

theorem sum_of_fractions : (3 : ℚ) / 7 + 9 / 12 = 33 / 28 := by
  sorry

end sum_of_fractions_l293_29347


namespace total_trip_time_l293_29378

def driving_time : ℝ := 5

theorem total_trip_time :
  let traffic_time := 2 * driving_time
  driving_time + traffic_time = 15 := by sorry

end total_trip_time_l293_29378


namespace sin_150_degrees_l293_29358

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l293_29358


namespace sock_pairs_count_l293_29399

/-- Given a number of sock pairs, calculate the number of ways to select two socks
    from different pairs. -/
def nonMatchingSelections (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The problem statement -/
theorem sock_pairs_count : ∃ (n : ℕ), n > 0 ∧ nonMatchingSelections n = 112 :=
  sorry

end sock_pairs_count_l293_29399


namespace solve_inequality_m_neg_four_solve_inequality_x_greater_than_one_l293_29387

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| - |2*x + m|

-- Theorem for part (1)
theorem solve_inequality_m_neg_four :
  ∀ x : ℝ, f x (-4) < 0 ↔ x < 5/3 ∨ x > 3 := by sorry

-- Theorem for part (2)
theorem solve_inequality_x_greater_than_one :
  ∀ m : ℝ, (∀ x : ℝ, x > 1 → f x m < 0) ↔ m ≥ -2 := by sorry

end solve_inequality_m_neg_four_solve_inequality_x_greater_than_one_l293_29387


namespace reciprocal_of_negative_one_point_five_l293_29300

theorem reciprocal_of_negative_one_point_five :
  let x : ℚ := -3/2  -- -1.5 as a rational number
  let y : ℚ := -2/3  -- The proposed reciprocal
  (∀ z : ℚ, z ≠ 0 → ∃ w : ℚ, z * w = 1) →  -- Definition of reciprocal
  x * y = 1 ∧ y * x = 1 :=  -- Proving y is the reciprocal of x
by sorry

end reciprocal_of_negative_one_point_five_l293_29300


namespace marshmallow_challenge_l293_29339

/-- The marshmallow challenge problem -/
theorem marshmallow_challenge 
  (haley michael brandon : ℕ) 
  (haley_marshmallows : haley = 8)
  (brandon_half_michael : brandon = michael / 2)
  (total_marshmallows : haley + michael + brandon = 44) :
  michael / haley = 3 := by
  sorry

end marshmallow_challenge_l293_29339


namespace percentage_no_conditions_is_13_33_l293_29396

/-- Represents the survey results of teachers' health conditions -/
structure TeacherSurvey where
  total : ℕ
  highBP : ℕ
  heartTrouble : ℕ
  diabetes : ℕ
  highBP_heartTrouble : ℕ
  heartTrouble_diabetes : ℕ
  highBP_diabetes : ℕ
  all_three : ℕ

/-- Calculates the percentage of teachers with no health conditions -/
def percentageWithNoConditions (survey : TeacherSurvey) : ℚ :=
  let withConditions := 
    survey.highBP + survey.heartTrouble + survey.diabetes -
    survey.highBP_heartTrouble - survey.heartTrouble_diabetes - survey.highBP_diabetes +
    survey.all_three
  let withoutConditions := survey.total - withConditions
  (withoutConditions : ℚ) / survey.total * 100

/-- The main theorem stating that the percentage of teachers with no health conditions is 13.33% -/
theorem percentage_no_conditions_is_13_33 (survey : TeacherSurvey) 
  (h1 : survey.total = 150)
  (h2 : survey.highBP = 80)
  (h3 : survey.heartTrouble = 60)
  (h4 : survey.diabetes = 30)
  (h5 : survey.highBP_heartTrouble = 20)
  (h6 : survey.heartTrouble_diabetes = 10)
  (h7 : survey.highBP_diabetes = 15)
  (h8 : survey.all_three = 5) :
  percentageWithNoConditions survey = 40/3 := by
  sorry

end percentage_no_conditions_is_13_33_l293_29396


namespace right_triangle_has_one_right_angle_l293_29322

-- Define a right triangle
structure RightTriangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  has_right_angle : ∃ i, angles i = 90

-- Theorem: A right triangle has exactly one right angle
theorem right_triangle_has_one_right_angle (t : RightTriangle) : 
  (∃! i, t.angles i = 90) :=
by sorry

end right_triangle_has_one_right_angle_l293_29322


namespace total_onion_weight_is_10_9_l293_29334

/-- The total weight of onions grown by Sara, Sally, Fred, and Jack -/
def total_onion_weight : ℝ :=
  let sara_onions := 4
  let sara_weight := 0.5
  let sally_onions := 5
  let sally_weight := 0.4
  let fred_onions := 9
  let fred_weight := 0.3
  let jack_onions := 7
  let jack_weight := 0.6
  sara_onions * sara_weight +
  sally_onions * sally_weight +
  fred_onions * fred_weight +
  jack_onions * jack_weight

/-- Proof that the total weight of onions is 10.9 pounds -/
theorem total_onion_weight_is_10_9 :
  total_onion_weight = 10.9 := by sorry

end total_onion_weight_is_10_9_l293_29334


namespace coin_pile_impossibility_l293_29386

/-- Represents a pile of coins -/
structure CoinPile :=
  (coins : ℕ)

/-- Represents the state of all coin piles -/
structure CoinState :=
  (piles : List CoinPile)

/-- Allowed operations on coin piles -/
inductive CoinOperation
  | Combine : CoinPile → CoinPile → CoinOperation
  | Split : CoinPile → CoinOperation

/-- Applies a coin operation to a coin state -/
def applyOperation (state : CoinState) (op : CoinOperation) : CoinState :=
  sorry

/-- Checks if a coin state matches the target configuration -/
def isTargetState (state : CoinState) : Prop :=
  ∃ (p1 p2 p3 : CoinPile),
    state.piles = [p1, p2, p3] ∧
    p1.coins = 52 ∧ p2.coins = 48 ∧ p3.coins = 5

/-- The main theorem stating the impossibility of reaching the target state -/
theorem coin_pile_impossibility :
  ∀ (initial : CoinState) (ops : List CoinOperation),
    initial.piles = [CoinPile.mk 51, CoinPile.mk 49, CoinPile.mk 5] →
    ¬(isTargetState (ops.foldl applyOperation initial)) :=
  sorry

end coin_pile_impossibility_l293_29386


namespace range_of_a_l293_29303

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 5 * x + 6 = 0}

-- Theorem statement
theorem range_of_a (a : ℝ) : Set.Nonempty (A a) → a ∈ Set.Iic (25/24) := by
  sorry

end range_of_a_l293_29303


namespace roshesmina_pennies_theorem_l293_29311

/-- Represents a piggy bank with a given number of compartments and pennies per compartment -/
structure PiggyBank where
  compartments : Nat
  penniesPerCompartment : Nat

/-- Calculates the total number of pennies in a piggy bank -/
def totalPennies (pb : PiggyBank) : Nat :=
  pb.compartments * pb.penniesPerCompartment

/-- Represents Roshesmina's piggy bank -/
def roshesminaBank : PiggyBank :=
  { compartments := 12,
    penniesPerCompartment := 2 }

/-- Adds a specified number of pennies to each compartment of a piggy bank -/
def addPennies (pb : PiggyBank) (amount : Nat) : PiggyBank :=
  { compartments := pb.compartments,
    penniesPerCompartment := pb.penniesPerCompartment + amount }

/-- Theorem stating that after adding 6 pennies to each compartment of Roshesmina's piggy bank, 
    the total number of pennies is 96 -/
theorem roshesmina_pennies_theorem :
  totalPennies (addPennies roshesminaBank 6) = 96 := by
  sorry

end roshesmina_pennies_theorem_l293_29311


namespace min_value_theorem_l293_29301

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (4 / x + 1 / (y + 1) ≥ 9 / 4) ∧
  (4 / x + 1 / (y + 1) = 9 / 4 ↔ x = 2 ∧ y = 1) :=
by sorry

end min_value_theorem_l293_29301


namespace oil_leak_before_fixing_l293_29328

theorem oil_leak_before_fixing (total_leak : ℕ) (leak_during_work : ℕ) 
  (h1 : total_leak = 11687)
  (h2 : leak_during_work = 5165) :
  total_leak - leak_during_work = 6522 := by
sorry

end oil_leak_before_fixing_l293_29328


namespace jesse_room_area_l293_29314

/-- The area of a rectangular room -/
def room_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of Jesse's room is 96 square feet -/
theorem jesse_room_area : room_area 12 8 = 96 := by
  sorry

end jesse_room_area_l293_29314


namespace cost_of_type_B_theorem_l293_29375

/-- The cost of purchasing type B books given the total number of books,
    the number of type A books purchased, and the unit price of type B books. -/
def cost_of_type_B (total_books : ℕ) (type_A_books : ℕ) (unit_price_B : ℕ) : ℕ :=
  unit_price_B * (total_books - type_A_books)

/-- Theorem stating that the cost of purchasing type B books
    is equal to 8(100-x) given the specified conditions. -/
theorem cost_of_type_B_theorem (x : ℕ) (h : x ≤ 100) :
  cost_of_type_B 100 x 8 = 8 * (100 - x) :=
by sorry

end cost_of_type_B_theorem_l293_29375


namespace solution_set_absolute_value_inequality_l293_29360

theorem solution_set_absolute_value_inequality :
  {x : ℝ | |2 - x| ≥ 1} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by
  sorry

end solution_set_absolute_value_inequality_l293_29360


namespace total_time_remaining_wallpaper_l293_29336

/-- Represents the time in hours to remove wallpaper from one wall -/
def time_per_wall : ℕ := 2

/-- Represents the number of walls in the dining room -/
def dining_room_walls : ℕ := 4

/-- Represents the number of walls in the living room -/
def living_room_walls : ℕ := 4

/-- Represents the number of walls already completed in the dining room -/
def completed_walls : ℕ := 1

/-- Theorem stating the total time to remove wallpaper from the remaining walls -/
theorem total_time_remaining_wallpaper :
  time_per_wall * (dining_room_walls + living_room_walls - completed_walls) = 14 := by
  sorry

end total_time_remaining_wallpaper_l293_29336


namespace x_value_is_36_l293_29313

theorem x_value_is_36 (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 110) : x = 36 := by
  sorry

end x_value_is_36_l293_29313


namespace billys_bicycles_l293_29312

/-- The number of spokes per wheel -/
def spokes_per_wheel : ℕ := 10

/-- The total number of spokes in the garage -/
def total_spokes : ℕ := 80

/-- The number of wheels per bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of bicycles owned by Billy's family -/
def number_of_bicycles : ℕ := total_spokes / (spokes_per_wheel * wheels_per_bicycle)

theorem billys_bicycles : number_of_bicycles = 4 := by
  sorry

end billys_bicycles_l293_29312


namespace line_y_axis_intersection_l293_29353

def line (x y : ℝ) : Prop := y = 2 * x + 1

def y_axis (x : ℝ) : Prop := x = 0

def intersection_point : Set (ℝ × ℝ) := {(0, 1)}

theorem line_y_axis_intersection :
  {p : ℝ × ℝ | line p.1 p.2 ∧ y_axis p.1} = intersection_point := by
sorry

end line_y_axis_intersection_l293_29353


namespace intersection_of_P_and_Q_l293_29392

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x ≤ 1}
def Q : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_P_and_Q :
  P ∩ Q = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by sorry

end intersection_of_P_and_Q_l293_29392


namespace sum_of_100th_row_general_row_sum_formula_l293_29379

/-- Represents the sum of numbers in the nth row of the triangular array -/
def rowSum (n : ℕ) : ℕ :=
  2^n - 3 * (n - 1)

/-- The triangular array is defined with 0, 1, 2, 3, ... along the sides,
    and interior numbers are obtained by adding the two adjacent numbers
    in the previous row and adding 1 to each sum. -/
axiom array_definition : True

theorem sum_of_100th_row :
  rowSum 100 = 2^100 - 297 :=
by sorry

theorem general_row_sum_formula (n : ℕ) (h : n > 0) :
  rowSum n = 2^n - 3 * (n - 1) :=
by sorry

end sum_of_100th_row_general_row_sum_formula_l293_29379


namespace largest_prime_factor_of_3913_l293_29357

theorem largest_prime_factor_of_3913 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 3913 ∧ ∀ (q : ℕ), q.Prime → q ∣ 3913 → q ≤ p ∧ p = 23 := by
  sorry

end largest_prime_factor_of_3913_l293_29357


namespace nancy_carrots_l293_29388

/-- Calculates the total number of carrots Nancy has -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) : ℕ :=
  initial - thrown_out + new_picked

/-- Proves that Nancy's total carrots is 31 given the problem conditions -/
theorem nancy_carrots : total_carrots 12 2 21 = 31 := by
  sorry

end nancy_carrots_l293_29388


namespace ruby_apples_remaining_l293_29356

theorem ruby_apples_remaining (initial : ℕ) (taken : ℕ) (remaining : ℕ) : 
  initial = 6357912 → taken = 2581435 → remaining = 3776477 → initial - taken = remaining := by
  sorry

end ruby_apples_remaining_l293_29356


namespace find_number_l293_29326

theorem find_number : ∃! x : ℕ, 220080 = (x + 445) * (2 * (x - 445)) + 80 := by
  sorry

end find_number_l293_29326


namespace C_share_of_profit_l293_29385

def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 2000
def total_profit : ℕ := 252000

theorem C_share_of_profit :
  (investment_C : ℚ) / (investment_A + investment_B + investment_C) * total_profit = 36000 :=
by sorry

end C_share_of_profit_l293_29385


namespace pants_price_problem_l293_29377

theorem pants_price_problem (total_cost belt_price pants_price : ℝ) : 
  total_cost = 70.93 →
  pants_price = belt_price - 2.93 →
  total_cost = belt_price + pants_price →
  pants_price = 34.00 := by
  sorry

end pants_price_problem_l293_29377


namespace quadratic_inequality_l293_29319

theorem quadratic_inequality (z : ℝ) : z^2 - 40*z + 400 ≤ 36 ↔ 14 ≤ z ∧ z ≤ 26 := by
  sorry

end quadratic_inequality_l293_29319


namespace cubic_derivative_root_existence_l293_29323

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The roots of a cubic polynomial -/
structure CubicRoots where
  a : ℝ
  b : ℝ
  c : ℝ
  h_order : a ≤ b ∧ b ≤ c

/-- Theorem: The derivative of a cubic polynomial has a root in the specified interval -/
theorem cubic_derivative_root_existence (f : CubicPolynomial) (roots : CubicRoots) :
  ∃ x : ℝ, x ∈ Set.Icc ((roots.b + roots.c) / 2) ((roots.b + 2 * roots.c) / 3) ∧
    (3 * x^2 + 2 * f.p * x + f.q) = 0 :=
sorry

end cubic_derivative_root_existence_l293_29323


namespace prob_two_red_crayons_l293_29390

/-- The probability of selecting 2 red crayons from a jar containing 6 crayons (3 red, 2 blue, 1 green) -/
theorem prob_two_red_crayons (total : ℕ) (red : ℕ) (blue : ℕ) (green : ℕ) :
  total = 6 →
  red = 3 →
  blue = 2 →
  green = 1 →
  (Nat.choose red 2 : ℚ) / (Nat.choose total 2) = 1 / 5 := by
  sorry

end prob_two_red_crayons_l293_29390


namespace max_equal_ending_digits_of_squares_max_equal_ending_digits_is_tight_l293_29333

/-- The maximum number of equal non-zero digits that can appear at the end of a perfect square in base 10 -/
def max_equal_ending_digits : ℕ := 3

/-- A function that returns the number of equal non-zero digits at the end of a number in base 10 -/
def count_equal_ending_digits (n : ℕ) : ℕ := sorry

theorem max_equal_ending_digits_of_squares :
  ∀ n : ℕ, count_equal_ending_digits (n^2) ≤ max_equal_ending_digits :=
by sorry

theorem max_equal_ending_digits_is_tight :
  ∃ n : ℕ, count_equal_ending_digits (n^2) = max_equal_ending_digits :=
by sorry

end max_equal_ending_digits_of_squares_max_equal_ending_digits_is_tight_l293_29333


namespace intersection_M_N_l293_29345

def M : Set ℝ := {x | ∃ t : ℝ, x = Real.exp (-t * Real.log 2)}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x ≤ 1} := by sorry

end intersection_M_N_l293_29345


namespace car_speed_l293_29393

/-- The speed of a car in km/h given the tire's rotation rate and circumference -/
theorem car_speed (revolutions_per_minute : ℝ) (tire_circumference : ℝ) : 
  revolutions_per_minute = 400 → 
  tire_circumference = 1 → 
  (revolutions_per_minute * tire_circumference * 60) / 1000 = 24 := by
sorry

end car_speed_l293_29393


namespace caravan_camel_count_l293_29366

/-- Represents the number of camels in the caravan -/
def num_camels : ℕ := 6

/-- Represents the number of hens in the caravan -/
def num_hens : ℕ := 60

/-- Represents the number of goats in the caravan -/
def num_goats : ℕ := 35

/-- Represents the number of keepers in the caravan -/
def num_keepers : ℕ := 10

/-- Represents the difference between the total number of feet and heads -/
def feet_head_difference : ℕ := 193

theorem caravan_camel_count : 
  (2 * num_hens + 4 * num_goats + 4 * num_camels + 2 * num_keepers) = 
  (num_hens + num_goats + num_camels + num_keepers + feet_head_difference) := by
  sorry

end caravan_camel_count_l293_29366


namespace social_event_handshakes_l293_29342

/-- Represents the social event setup -/
structure SocialEvent where
  total_people : Nat
  group_a_size : Nat
  group_b_size : Nat
  group_b_connections : Nat

/-- Calculate the number of handshakes in the social event -/
def calculate_handshakes (event : SocialEvent) : Nat :=
  let group_b_internal := event.group_b_size.choose 2
  let group_ab_handshakes := event.group_b_size * event.group_b_connections
  group_b_internal + group_ab_handshakes

/-- Theorem stating the number of handshakes in the given social event -/
theorem social_event_handshakes :
  ∃ (event : SocialEvent),
    event.total_people = 40 ∧
    event.group_a_size = 25 ∧
    event.group_b_size = 15 ∧
    event.group_b_connections = 5 ∧
    calculate_handshakes event = 180 := by
  sorry

end social_event_handshakes_l293_29342


namespace product_of_first_1001_primes_factors_product_of_first_1001_primes_not_factor_l293_29374

def first_n_primes (n : ℕ) : List ℕ :=
  sorry

def product_of_list (l : List ℕ) : ℕ :=
  sorry

def is_factor (a b : ℕ) : Prop :=
  ∃ k : ℕ, b = a * k

theorem product_of_first_1001_primes_factors (n : ℕ) :
  let P := product_of_list (first_n_primes 1001)
  (n = 2002 ∨ n = 3003 ∨ n = 5005 ∨ n = 6006) →
  is_factor n P :=
sorry

theorem product_of_first_1001_primes_not_factor :
  let P := product_of_list (first_n_primes 1001)
  ¬ is_factor 7007 P :=
sorry

end product_of_first_1001_primes_factors_product_of_first_1001_primes_not_factor_l293_29374


namespace no_sin_4x_function_of_sin_x_l293_29307

open Real

theorem no_sin_4x_function_of_sin_x : ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, sin (4 * x) = f (sin x) := by
  sorry

end no_sin_4x_function_of_sin_x_l293_29307


namespace sean_final_houses_l293_29349

/-- Calculates the final number of houses Sean has after a series of transactions in Monopoly. -/
def final_houses (initial : ℕ) (traded_for_money : ℕ) (bought : ℕ) (traded_for_marvin : ℕ) (sold_for_atlantic : ℕ) (traded_for_hotels : ℕ) : ℕ :=
  initial - traded_for_money + bought - traded_for_marvin - sold_for_atlantic - traded_for_hotels

/-- Theorem stating that Sean ends up with 20 houses after the given transactions. -/
theorem sean_final_houses : 
  final_houses 45 15 18 5 7 16 = 20 := by
  sorry

end sean_final_houses_l293_29349


namespace gcd_problem_l293_29355

theorem gcd_problem (b : ℤ) (h : 1039 ∣ b) : Int.gcd (b^2 + 7*b + 18) (b + 6) = 6 := by
  sorry

end gcd_problem_l293_29355


namespace water_gun_game_theorem_l293_29359

/-- Represents a student with a position -/
structure Student where
  position : ℝ × ℝ

/-- The environment of the water gun game -/
structure WaterGunGame where
  n : ℕ
  students : Fin (2*n+1) → Student
  distinct_distances : ∀ i j k l, i ≠ j → k ≠ l → 
    (students i).position ≠ (students j).position → 
    (students k).position ≠ (students l).position →
    dist (students i).position (students j).position ≠ 
    dist (students k).position (students l).position

/-- A student squirts another student -/
def squirts (game : WaterGunGame) (i j : Fin (2*game.n+1)) : Prop :=
  ∀ k, k ≠ j → 
    dist (game.students i).position (game.students j).position < 
    dist (game.students i).position (game.students k).position

theorem water_gun_game_theorem (game : WaterGunGame) : 
  (∃ i j, i ≠ j ∧ squirts game i j ∧ squirts game j i) ∧ 
  (∃ i, ∀ j, ¬squirts game j i) :=
sorry

end water_gun_game_theorem_l293_29359


namespace parallelogram_area_in_regular_hexagon_l293_29391

/-- The area of the parallelogram formed by connecting every second vertex of a regular hexagon --/
theorem parallelogram_area_in_regular_hexagon (side_length : ℝ) (h : side_length = 12) :
  let large_triangle_area := Real.sqrt 3 / 4 * (2 * side_length) ^ 2
  let small_triangle_area := Real.sqrt 3 / 4 * side_length ^ 2
  large_triangle_area - 3 * small_triangle_area = 36 * Real.sqrt 3 := by
  sorry

end parallelogram_area_in_regular_hexagon_l293_29391


namespace power_inequality_l293_29367

theorem power_inequality (x y : ℝ) 
  (h1 : x^5 > y^4) 
  (h2 : y^5 > x^4) : 
  x^3 > y^2 := by
sorry

end power_inequality_l293_29367


namespace probability_of_two_queens_or_at_least_one_king_l293_29389

-- Define a standard deck
def standard_deck : ℕ := 52

-- Define the number of queens in a deck
def queens_in_deck : ℕ := 4

-- Define the number of kings in a deck
def kings_in_deck : ℕ := 4

-- Define the probability of the event
def prob_two_queens_or_at_least_one_king : ℚ := 2 / 13

-- State the theorem
theorem probability_of_two_queens_or_at_least_one_king :
  let p := (queens_in_deck * (queens_in_deck - 1) / 2 +
            kings_in_deck * (standard_deck - kings_in_deck) +
            kings_in_deck * (kings_in_deck - 1) / 2) /
           (standard_deck * (standard_deck - 1) / 2)
  p = prob_two_queens_or_at_least_one_king :=
by sorry

end probability_of_two_queens_or_at_least_one_king_l293_29389


namespace smallest_number_between_10_and_11_l293_29351

theorem smallest_number_between_10_and_11 (x y z : ℝ) 
  (sum_eq : x + y + z = 150)
  (y_eq : y = 3 * x + 10)
  (z_eq : z = x^2 - 5) :
  ∃ w, w = min x (min y z) ∧ 10 < w ∧ w < 11 :=
sorry

end smallest_number_between_10_and_11_l293_29351


namespace no_alpha_sequence_exists_l293_29398

theorem no_alpha_sequence_exists :
  ¬ ∃ (α : ℝ) (a : ℕ → ℝ),
    (0 < α ∧ α < 1) ∧
    (∀ n, 0 < a n) ∧
    (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) :=
by sorry

end no_alpha_sequence_exists_l293_29398


namespace probability_black_or_white_ball_l293_29332

theorem probability_black_or_white_ball 
  (p_red : ℝ) 
  (p_white : ℝ) 
  (h1 : p_red = 0.45) 
  (h2 : p_white = 0.25) 
  (h3 : 0 ≤ p_red ∧ p_red ≤ 1) 
  (h4 : 0 ≤ p_white ∧ p_white ≤ 1) : 
  p_red + p_white + (1 - p_red - p_white) = 1 ∧ 1 - p_red = 0.55 := by
sorry

end probability_black_or_white_ball_l293_29332


namespace min_value_sum_l293_29327

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∃ (m : ℝ), m = 10 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x * y = 1 → 
    (x + 1/x) + (y + 1/y) ≥ m := by
  sorry

end min_value_sum_l293_29327


namespace chessboard_repaint_theorem_l293_29373

/-- Represents a chessboard of size n × n -/
structure Chessboard (n : ℕ) where
  size : n ≥ 3

/-- Represents an L-shaped tetromino and its rotations -/
inductive Tetromino
  | L
  | RotatedL90
  | RotatedL180
  | RotatedL270

/-- Represents a move that repaints a tetromino on the chessboard -/
def Move (n : ℕ) := Fin n → Fin n → Tetromino

/-- Predicate to check if a series of moves can repaint the entire chessboard -/
def CanRepaintEntireBoard (n : ℕ) (moves : List (Move n)) : Prop :=
  sorry

/-- Main theorem: The chessboard can be entirely repainted if and only if n is even and n ≥ 4 -/
theorem chessboard_repaint_theorem (n : ℕ) (b : Chessboard n) :
  (∃ (moves : List (Move n)), CanRepaintEntireBoard n moves) ↔ (Even n ∧ n ≥ 4) :=
sorry

end chessboard_repaint_theorem_l293_29373


namespace tangent_line_equation_l293_29361

/-- A line passing through (b, 0) and tangent to a circle of radius r centered at (0, 0),
    forming a triangle in the first quadrant with area S, has the equation rx - bry - rb = 0 --/
theorem tangent_line_equation (b r S : ℝ) (hb : b > 0) (hr : r > 0) (hS : S > 0) :
  ∃ (x y : ℝ → ℝ), 
    (∀ t, x t = b ∧ y t = 0) →  -- Line passes through (b, 0)
    (∃ t, (x t)^2 + (y t)^2 = r^2) →  -- Line touches the circle
    (∃ h, S = (1/2) * b * h) →  -- Triangle area
    (∀ t, r * (x t) - b * r * (y t) - r * b = 0) :=  -- Equation of the line
  sorry

end tangent_line_equation_l293_29361


namespace custom_op_two_five_l293_29321

/-- Custom binary operation on real numbers -/
def custom_op (a b : ℝ) : ℝ := 4 * a + 3 * b

/-- Theorem stating that 2 ⊗ 5 = 23 under the custom operation -/
theorem custom_op_two_five : custom_op 2 5 = 23 := by
  sorry

end custom_op_two_five_l293_29321


namespace correct_prediction_probability_l293_29394

def n_monday : ℕ := 5
def n_tuesday : ℕ := 6
def n_total : ℕ := n_monday + n_tuesday
def n_correct : ℕ := 7
def n_correct_monday : ℕ := 3

theorem correct_prediction_probability :
  let p : ℝ := 1 / 2
  (Nat.choose n_monday n_correct_monday * p ^ n_monday * (1 - p) ^ (n_monday - n_correct_monday)) *
  (Nat.choose n_tuesday (n_correct - n_correct_monday) * p ^ (n_correct - n_correct_monday) * (1 - p) ^ (n_tuesday - (n_correct - n_correct_monday))) /
  (Nat.choose n_total n_correct * p ^ n_correct * (1 - p) ^ (n_total - n_correct)) = 5 / 11 := by
  sorry

end correct_prediction_probability_l293_29394


namespace g_evaluation_l293_29369

def g (x : ℝ) : ℝ := 3 * x^3 + 5 * x^2 - 6 * x + 4

theorem g_evaluation : 3 * g 2 - 2 * g (-1) = 84 := by
  sorry

end g_evaluation_l293_29369


namespace money_sum_existence_l293_29335

theorem money_sum_existence : ∃ (k n : ℕ), 
  1 ≤ k ∧ k ≤ 9 ∧ n ≥ 1 ∧
  (k * (100 * n + 10 + 1) = 10666612) ∧
  (k * (n + 2) = (1 + 0 + 6 + 6 + 6 + 6 + 1 + 2)) :=
sorry

end money_sum_existence_l293_29335


namespace min_vertical_distance_l293_29341

/-- The vertical distance between |x| and -x^2-4x-3 -/
def verticalDistance (x : ℝ) : ℝ := |x| - (-x^2 - 4*x - 3)

/-- The minimum vertical distance between |x| and -x^2-4x-3 is 3/4 -/
theorem min_vertical_distance :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), verticalDistance x₀ ≤ verticalDistance x ∧ verticalDistance x₀ = 3/4 := by
  sorry


end min_vertical_distance_l293_29341


namespace grocery_bill_calculation_l293_29380

/-- Calculates the new total bill for a grocery delivery order with item substitutions -/
theorem grocery_bill_calculation
  (original_order : ℝ)
  (tomatoes_old tomatoes_new : ℝ)
  (lettuce_old lettuce_new : ℝ)
  (celery_old celery_new : ℝ)
  (delivery_and_tip : ℝ)
  (h1 : original_order = 25)
  (h2 : tomatoes_old = 0.99)
  (h3 : tomatoes_new = 2.20)
  (h4 : lettuce_old = 1.00)
  (h5 : lettuce_new = 1.75)
  (h6 : celery_old = 1.96)
  (h7 : celery_new = 2.00)
  (h8 : delivery_and_tip = 8.00) :
  original_order + (tomatoes_new - tomatoes_old) + (lettuce_new - lettuce_old) +
  (celery_new - celery_old) + delivery_and_tip = 35 :=
by sorry

end grocery_bill_calculation_l293_29380


namespace point_B_coordinates_l293_29376

/-- Given points A and C, and the relation between vectors AB and BC, 
    prove that the coordinates of point B are (-2, 5/3) -/
theorem point_B_coordinates 
  (A B C : ℝ × ℝ) 
  (hA : A = (2, 3)) 
  (hC : C = (0, 1)) 
  (h_vec : B - A = -2 • (C - B)) : 
  B = (-2, 5/3) := by
  sorry

end point_B_coordinates_l293_29376


namespace cara_age_l293_29372

/-- Given the ages of three generations in a family, prove Cara's age. -/
theorem cara_age (cara_age mom_age grandma_age : ℕ) 
  (h1 : cara_age + 20 = mom_age)
  (h2 : mom_age + 15 = grandma_age)
  (h3 : grandma_age = 75) :
  cara_age = 40 := by
  sorry

end cara_age_l293_29372


namespace anderson_pet_food_weight_l293_29315

/-- Calculates the total weight of pet food in ounces -/
def total_pet_food_ounces (cat_food_bags : ℕ) (cat_food_weight : ℕ) 
                          (dog_food_bags : ℕ) (dog_food_extra_weight : ℕ) 
                          (ounces_per_pound : ℕ) : ℕ :=
  let total_cat_food := cat_food_bags * cat_food_weight
  let dog_food_weight := cat_food_weight + dog_food_extra_weight
  let total_dog_food := dog_food_bags * dog_food_weight
  let total_weight := total_cat_food + total_dog_food
  total_weight * ounces_per_pound

/-- Theorem: The total weight of pet food Mrs. Anderson bought is 256 ounces -/
theorem anderson_pet_food_weight : 
  total_pet_food_ounces 2 3 2 2 16 = 256 := by
  sorry

end anderson_pet_food_weight_l293_29315


namespace integer_equation_solution_l293_29325

theorem integer_equation_solution (x y z : ℤ) :
  x^2 * (y - z) + y^2 * (z - x) + z^2 * (x - y) = 2 ↔ 
  ∃ k : ℤ, x = k + 1 ∧ y = k ∧ z = k - 1 :=
by sorry

end integer_equation_solution_l293_29325


namespace simplify_logarithmic_expression_l293_29350

theorem simplify_logarithmic_expression (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  Real.log (Real.cos x * Real.tan x + 1 - 2 * Real.sin (x / 2) ^ 2) +
  Real.log (Real.sqrt 2 * Real.cos (x - Real.pi / 4)) -
  Real.log (1 + Real.sin (2 * x)) = 0 := by
  sorry

end simplify_logarithmic_expression_l293_29350


namespace minus_510_in_third_quadrant_l293_29352

-- Define a function to normalize an angle to the range [0, 360)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

-- Define a function to determine the quadrant of an angle
def getQuadrant (angle : Int) : Nat :=
  let normalizedAngle := normalizeAngle angle
  if 0 < normalizedAngle && normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle && normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle && normalizedAngle < 270 then 3
  else 4

-- Theorem statement
theorem minus_510_in_third_quadrant :
  getQuadrant (-510) = 3 :=
sorry

end minus_510_in_third_quadrant_l293_29352


namespace coin_value_is_70_rupees_l293_29364

/-- Calculates the total value in rupees given the number of coins and their values -/
def total_value_in_rupees (total_coins : ℕ) (coins_20_paise : ℕ) : ℚ :=
  let coins_25_paise := total_coins - coins_20_paise
  let value_20_paise := coins_20_paise * 20
  let value_25_paise := coins_25_paise * 25
  let total_paise := value_20_paise + value_25_paise
  total_paise / 100

/-- Proves that the total value of the given coins is 70 rupees -/
theorem coin_value_is_70_rupees :
  total_value_in_rupees 324 220 = 70 := by
  sorry

end coin_value_is_70_rupees_l293_29364


namespace comic_books_count_l293_29304

theorem comic_books_count (total : ℕ) 
  (h1 : (30 : ℚ) / 100 * total = (total - (70 : ℚ) / 100 * total))
  (h2 : (70 : ℚ) / 100 * total ≥ 120)
  (h3 : ∀ n : ℕ, n < total → (70 : ℚ) / 100 * n < 120) : 
  total = 172 := by
sorry

end comic_books_count_l293_29304


namespace existence_of_m_n_l293_29397

theorem existence_of_m_n (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m + n ≤ (p + 1) / 2 ∧ p ∣ 2^n * 3^m - 1 := by
  sorry

end existence_of_m_n_l293_29397


namespace divisor_proof_l293_29340

theorem divisor_proof (dividend quotient remainder divisor : ℤ) : 
  dividend = 474232 →
  quotient = 594 →
  remainder = -968 →
  dividend = divisor * quotient + remainder →
  divisor = 800 := by
sorry

end divisor_proof_l293_29340


namespace prob_ace_of_spades_l293_29384

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)
  (rank_count : (cards.image Prod.fst).card = 13)
  (suit_count : (cards.image Prod.snd).card = 4)

/-- The probability of drawing a specific card from a shuffled deck -/
def prob_draw_specific_card (d : Deck) : ℚ :=
  1 / 52

/-- Theorem: The probability of drawing the Ace of Spades from a shuffled standard deck is 1/52 -/
theorem prob_ace_of_spades (d : Deck) :
  prob_draw_specific_card d = 1 / 52 := by
  sorry

end prob_ace_of_spades_l293_29384


namespace cricket_team_average_age_l293_29309

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (remaining_age_diff : ℝ),
    team_size = 15 →
    captain_age = 32 →
    wicket_keeper_age_diff = 5 →
    remaining_age_diff = 2 →
    ∃ (team_avg_age : ℝ),
      team_avg_age * team_size =
        captain_age + (captain_age + wicket_keeper_age_diff) +
        (team_size - 2) * (team_avg_age - remaining_age_diff) ∧
      team_avg_age = 21.5 := by
sorry


end cricket_team_average_age_l293_29309


namespace opposite_numbers_example_l293_29324

theorem opposite_numbers_example : -(-(5 : ℤ)) = -(-|5|) → -(-(5 : ℤ)) + (-|5|) = 0 := by
  sorry

end opposite_numbers_example_l293_29324
