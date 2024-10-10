import Mathlib

namespace restaurant_bill_solution_l3672_367273

/-- Represents the restaurant bill problem -/
def restaurant_bill_problem (adults : ℕ) (meal_cost : ℕ) (total_bill : ℕ) : Prop :=
  ∃ children : ℕ, 
    adults * meal_cost + children * meal_cost = total_bill

/-- Theorem stating the solution to the restaurant bill problem -/
theorem restaurant_bill_solution :
  restaurant_bill_problem 2 8 56 → ∃ children : ℕ, children = 5 :=
by
  sorry

#check restaurant_bill_solution

end restaurant_bill_solution_l3672_367273


namespace quadratic_real_roots_l3672_367232

/-- The quadratic equation x^2 - 6x + m = 0 has real roots if and only if m ≤ 9 -/
theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + m = 0) ↔ m ≤ 9 := by
  sorry

end quadratic_real_roots_l3672_367232


namespace b_fourth_congruence_l3672_367247

theorem b_fourth_congruence (n : ℕ+) (b : ℤ) (h : b^2 ≡ 1 [ZMOD n]) :
  b^4 ≡ 1 [ZMOD n] := by
  sorry

end b_fourth_congruence_l3672_367247


namespace lineup_probability_l3672_367270

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

theorem lineup_probability :
  let valid_arrangements := Nat.choose 14 9 + 6 * Nat.choose 13 8
  let total_arrangements := Nat.choose total_children num_boys
  (valid_arrangements : ℚ) / total_arrangements =
    probability_no_more_than_five_girls_between_first_and_last_boys :=
by
  sorry

def probability_no_more_than_five_girls_between_first_and_last_boys : ℚ :=
  (Nat.choose 14 9 + 6 * Nat.choose 13 8 : ℚ) / Nat.choose total_children num_boys

end lineup_probability_l3672_367270


namespace ball_probability_l3672_367201

theorem ball_probability (m : ℕ) : 
  (3 : ℚ) / (3 + 4 + m) = 1 / 3 → m = 2 := by
  sorry

end ball_probability_l3672_367201


namespace S_not_equal_T_l3672_367236

-- Define the set S
def S : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}

-- Define the set T
def T : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}

-- Theorem statement
theorem S_not_equal_T : S ≠ T := by
  sorry

end S_not_equal_T_l3672_367236


namespace tom_dance_years_l3672_367233

/-- The number of years Tom danced -/
def years_danced (
  dances_per_week : ℕ
) (
  hours_per_dance : ℕ
) (
  weeks_per_year : ℕ
) (
  total_hours_danced : ℕ
) : ℕ :=
  total_hours_danced / (dances_per_week * hours_per_dance * weeks_per_year)

theorem tom_dance_years :
  years_danced 4 2 52 4160 = 10 := by
  sorry

end tom_dance_years_l3672_367233


namespace geometric_sequence_property_l3672_367244

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence property
  q > 0 →  -- Common ratio is positive
  (3 * a 1 + 2 * a 2 = a 3) →  -- Arithmetic sequence property
  q = 3 := by
sorry

end geometric_sequence_property_l3672_367244


namespace check_error_l3672_367293

theorem check_error (x y : ℤ) 
  (h1 : 10 ≤ x ∧ x ≤ 99) 
  (h2 : 10 ≤ y ∧ y ≤ 99) 
  (h3 : 100 * y + x - (100 * x + y) = 2046) 
  (h4 : x = (3 * y) / 2) : 
  x = 66 := by sorry

end check_error_l3672_367293


namespace sophomore_count_l3672_367215

theorem sophomore_count (n : ℕ) : 
  n > 1000 → -- Ensure n is large enough to accommodate all students
  (60 : ℚ) / n = (27 : ℚ) / 450 →
  n = 450 + 250 + 300 :=
by
  sorry

end sophomore_count_l3672_367215


namespace jordyn_total_cost_l3672_367290

/-- The total amount Jordyn would pay for the fruits with discounts, sales tax, and service charge -/
def total_cost (cherry_price olives_price grapes_price : ℚ)
               (cherry_quantity olives_quantity grapes_quantity : ℕ)
               (cherry_discount olives_discount grapes_discount : ℚ)
               (sales_tax service_charge : ℚ) : ℚ :=
  let cherry_total := cherry_price * cherry_quantity
  let olives_total := olives_price * olives_quantity
  let grapes_total := grapes_price * grapes_quantity
  let cherry_discounted := cherry_total * (1 - cherry_discount)
  let olives_discounted := olives_total * (1 - olives_discount)
  let grapes_discounted := grapes_total * (1 - grapes_discount)
  let subtotal := cherry_discounted + olives_discounted + grapes_discounted
  let with_tax := subtotal * (1 + sales_tax)
  with_tax * (1 + service_charge)

/-- The theorem stating the total cost Jordyn would pay -/
theorem jordyn_total_cost :
  total_cost 5 7 11 50 75 25 (12/100) (8/100) (15/100) (5/100) (2/100) = 1002.32 := by
  sorry

end jordyn_total_cost_l3672_367290


namespace smallest_multiple_l3672_367258

theorem smallest_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 5 = 0) ∧ 
  (n % 8 = 0) ∧ 
  (n % 2 = 0) ∧ 
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (m % 5 = 0) ∧ (m % 8 = 0) ∧ (m % 2 = 0) → n ≤ m) ∧
  n = 120 := by
sorry

end smallest_multiple_l3672_367258


namespace segment_length_l3672_367221

/-- Given a line segment CD with points R and S on it, prove that CD has length 273.6 -/
theorem segment_length (C D R S : ℝ) : 
  (R > (C + D) / 2) →  -- R is on the same side of the midpoint as S
  (S > (C + D) / 2) →  -- S is on the same side of the midpoint as R
  (R - C) / (D - R) = 3 / 5 →  -- R divides CD in ratio 3:5
  (S - C) / (D - S) = 4 / 7 →  -- S divides CD in ratio 4:7
  S - R = 3 →  -- RS = 3
  D - C = 273.6 :=  -- CD = 273.6
by sorry

end segment_length_l3672_367221


namespace graph_of_S_l3672_367220

theorem graph_of_S (a b t : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (sum_eq : a + b = 2) (prod_eq : a * b = t - 1) (ht : 1 < t ∧ t < 2) :
  (a - b)^2 = 8 - 4*t := by
  sorry

end graph_of_S_l3672_367220


namespace faster_walking_speed_l3672_367292

theorem faster_walking_speed 
  (actual_speed : ℝ) 
  (actual_distance : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_speed = 8) 
  (h2 : actual_distance = 40) 
  (h3 : additional_distance = 20) : 
  ∃ (faster_speed : ℝ), 
    faster_speed = (actual_distance + additional_distance) / (actual_distance / actual_speed) ∧ 
    faster_speed = 12 := by
  sorry


end faster_walking_speed_l3672_367292


namespace two_letter_language_max_words_l3672_367295

/-- A language with two letters and specific word formation rules -/
structure TwoLetterLanguage where
  alphabet : Finset Char
  max_word_length : ℕ
  is_valid_word : List Char → Prop
  no_concatenation : ∀ (w1 w2 : List Char), is_valid_word w1 → is_valid_word w2 → ¬is_valid_word (w1 ++ w2)

/-- The maximum number of words in the specific two-letter language -/
def max_word_count (L : TwoLetterLanguage) : ℕ := 16056

/-- Theorem stating the maximum number of words in the specific two-letter language -/
theorem two_letter_language_max_words (L : TwoLetterLanguage) 
  (h1 : L.alphabet.card = 2)
  (h2 : L.max_word_length = 13)
  : max_word_count L = 16056 := by
  sorry

end two_letter_language_max_words_l3672_367295


namespace b_work_time_l3672_367250

-- Define the work completion time for A and the combined team
def a_time : ℝ := 6
def combined_time : ℝ := 3

-- Define the total payment and C's payment
def total_payment : ℝ := 5000
def c_payment : ℝ := 625.0000000000002

-- Define B's work completion time (to be proved)
def b_time : ℝ := 8

-- Theorem statement
theorem b_work_time : 
  (1 / a_time + 1 / b_time + c_payment / total_payment / combined_time = 1 / combined_time) → 
  b_time = 8 :=
by sorry

end b_work_time_l3672_367250


namespace solution_mixing_l3672_367253

theorem solution_mixing (x y : Real) :
  x + y = 40 →
  0.30 * x + 0.80 * y = 0.45 * 40 →
  y = 12 →
  x = 28 →
  0.30 * 28 + 0.80 * 12 = 0.45 * 40 :=
by sorry

end solution_mixing_l3672_367253


namespace watermelon_puree_volume_watermelon_puree_volume_proof_l3672_367288

/-- Given the conditions of Carla's smoothie recipe, prove that she uses 500 ml of watermelon puree. -/
theorem watermelon_puree_volume : ℝ → Prop :=
  fun watermelon_puree : ℝ =>
    let total_volume : ℝ := 4 * 150
    let cream_volume : ℝ := 100
    (total_volume = watermelon_puree + cream_volume) → (watermelon_puree = 500)

/-- Proof of the watermelon puree volume theorem -/
theorem watermelon_puree_volume_proof : watermelon_puree_volume 500 := by
  sorry

#check watermelon_puree_volume
#check watermelon_puree_volume_proof

end watermelon_puree_volume_watermelon_puree_volume_proof_l3672_367288


namespace inverse_modulo_31_l3672_367200

theorem inverse_modulo_31 (h : (17⁻¹ : ZMod 31) = 13) : (21⁻¹ : ZMod 31) = 6 := by
  sorry

end inverse_modulo_31_l3672_367200


namespace batsman_total_score_l3672_367269

/-- Represents the score of a batsman in cricket --/
structure BatsmanScore where
  total : ℝ
  boundaries : ℕ
  sixes : ℕ
  runningPercentage : ℝ

/-- The total score of a batsman is 120 runs given the specified conditions --/
theorem batsman_total_score 
  (score : BatsmanScore) 
  (h1 : score.boundaries = 5) 
  (h2 : score.sixes = 5) 
  (h3 : score.runningPercentage = 58.333333333333336) :
  score.total = 120 := by
  sorry

end batsman_total_score_l3672_367269


namespace irrationality_classification_l3672_367241

-- Define rational numbers
def isRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- Define irrational numbers
def isIrrational (x : ℝ) : Prop := ¬ (isRational x)

theorem irrationality_classification :
  isRational (-2) ∧ 
  isRational (1/2) ∧ 
  isIrrational (Real.sqrt 3) ∧ 
  isRational 2 :=
sorry

end irrationality_classification_l3672_367241


namespace secretary_typing_orders_l3672_367257

/-- The number of letters to be typed -/
def total_letters : ℕ := 12

/-- The number of the letter that has been typed -/
def typed_letter : ℕ := 10

/-- Calculates the number of possible typing orders for the remaining letters -/
def possible_orders : ℕ :=
  Finset.sum (Finset.range 10) (fun k =>
    Nat.choose 9 k * (k + 1) * (k + 2))

/-- Theorem stating the number of possible typing orders -/
theorem secretary_typing_orders :
  possible_orders = 5166 := by
  sorry

end secretary_typing_orders_l3672_367257


namespace share_ratio_l3672_367277

/-- 
Given:
- The total amount of money is $400
- A's share is $160
- A gets a certain fraction (x) as much as B and C together
- B gets 6/9 as much as A and C together

Prove that the ratio of A's share to the combined share of B and C is 2:3
-/
theorem share_ratio (total : ℕ) (a b c : ℕ) (x : ℚ) :
  total = 400 →
  a = 160 →
  a = x * (b + c) →
  b = (6/9 : ℚ) * (a + c) →
  a + b + c = total →
  (a : ℚ) / ((b + c) : ℚ) = 2/3 := by
  sorry

end share_ratio_l3672_367277


namespace fourth_week_distance_l3672_367263

def running_schedule (week1_distance : ℝ) : ℕ → ℝ
  | 1 => week1_distance * 7
  | 2 => (2 * week1_distance + 3) * 7
  | 3 => (2 * week1_distance + 3) * 9
  | 4 => (2 * week1_distance + 3) * 9 * 0.9 * 0.5 * 5
  | _ => 0

theorem fourth_week_distance :
  running_schedule 2 4 = 20.25 := by
  sorry

end fourth_week_distance_l3672_367263


namespace three_teachers_three_students_arrangements_l3672_367271

/-- The number of arrangements for teachers and students in a row --/
def arrangements (num_teachers num_students : ℕ) : ℕ :=
  (num_teachers + 1).factorial * num_students.factorial

/-- Theorem: The number of arrangements for 3 teachers and 3 students,
    where no two students are adjacent, is 144 --/
theorem three_teachers_three_students_arrangements :
  arrangements 3 3 = 144 :=
sorry

end three_teachers_three_students_arrangements_l3672_367271


namespace leahs_coins_value_l3672_367299

/-- Represents the value of a coin in cents -/
def coinValue (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | _ => 0

/-- Calculates the total value of coins given their quantities -/
def totalValue (pennies nickels dimes : ℕ) : ℕ :=
  pennies * coinValue "penny" + nickels * coinValue "nickel" + dimes * coinValue "dime"

theorem leahs_coins_value :
  ∀ (pennies nickels dimes : ℕ),
    pennies + nickels + dimes = 17 →
    nickels + 2 = pennies →
    totalValue pennies nickels dimes = 68 :=
by sorry

end leahs_coins_value_l3672_367299


namespace circle_equation_with_tangent_line_l3672_367214

/-- The equation of a circle with center (1, -1) tangent to the line x + y - √6 = 0 --/
theorem circle_equation_with_tangent_line :
  ∃ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 3 ∧
  (x + y - Real.sqrt 6 = 0 → 
    ∃ (x₀ y₀ : ℝ), (x₀ - 1)^2 + (y₀ + 1)^2 = 3 ∧ x₀ + y₀ - Real.sqrt 6 = 0) :=
by sorry

end circle_equation_with_tangent_line_l3672_367214


namespace square_root_of_product_plus_one_l3672_367283

theorem square_root_of_product_plus_one (a : ℕ) (h : a = 25) : 
  Real.sqrt (a * (a + 1) * (a + 2) * (a + 3) + 1) = a^2 + 3*a + 1 := by
  sorry

end square_root_of_product_plus_one_l3672_367283


namespace opposite_of_2023_l3672_367264

theorem opposite_of_2023 : 
  ∀ x : ℤ, (x + 2023 = 0) ↔ (x = -2023) :=
by
  sorry

end opposite_of_2023_l3672_367264


namespace bank_max_profit_rate_l3672_367286

/-- The bank's profit function --/
def profit (x : ℝ) : ℝ := 480 * x^2 - 10000 * x^3

/-- The derivative of the profit function --/
def profit_derivative (x : ℝ) : ℝ := 960 * x - 30000 * x^2

theorem bank_max_profit_rate :
  ∃ x : ℝ, x ∈ Set.Ioo 0 0.048 ∧
    (∀ y ∈ Set.Ioo 0 0.048, profit y ≤ profit x) ∧
    x = 0.032 := by
  sorry

end bank_max_profit_rate_l3672_367286


namespace arithmetic_sequence_property_l3672_367248

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 3)
  (h_sum : a 3 + a 4 = 9) :
  a 1 * a 6 = 14 :=
sorry

end arithmetic_sequence_property_l3672_367248


namespace euler_family_mean_age_l3672_367281

def euler_family_ages : List ℕ := [9, 9, 9, 9, 18, 21, 21]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem euler_family_mean_age :
  mean euler_family_ages = 13.71 := by
  sorry

end euler_family_mean_age_l3672_367281


namespace inequality_solution_l3672_367280

theorem inequality_solution (x : ℝ) : 
  x ≠ -2 ∧ x ≠ 2 →
  ((2 * x + 1) / (x + 2) - (x - 3) / (3 * x - 6) ≤ 0 ↔ 
   (x > -2 ∧ x < 0) ∨ (x > 2 ∧ x ≤ 14/5)) :=
by sorry

end inequality_solution_l3672_367280


namespace probability_greater_than_two_l3672_367256

def standard_die := Finset.range 6

def favorable_outcomes : Finset Nat :=
  standard_die.filter (λ x => x > 2)

theorem probability_greater_than_two :
  (favorable_outcomes.card : ℚ) / standard_die.card = 2 / 3 := by
  sorry

end probability_greater_than_two_l3672_367256


namespace sequence_value_l3672_367223

/-- Given a sequence {aₙ} satisfying aₙ₊₁ = 1 / (1 - aₙ) for all n ≥ 1,
    and a₂ = 2, prove that a₁ = 1/2 -/
theorem sequence_value (a : ℕ → ℚ)
  (h₁ : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 1 / (1 - a n))
  (h₂ : a 2 = 2) :
  a 1 = 1/2 := by
sorry

end sequence_value_l3672_367223


namespace new_regression_equation_l3672_367212

-- Define the initial regression line
def initial_regression (x : ℝ) : ℝ := 2 * x - 0.4

-- Define the sample size and mean x
def sample_size : ℕ := 10
def mean_x : ℝ := 2

-- Define the removed points
def removed_point1 : ℝ × ℝ := (-3, 1)
def removed_point2 : ℝ × ℝ := (3, -1)

-- Define the new slope
def new_slope : ℝ := 3

-- Theorem statement
theorem new_regression_equation :
  let new_mean_x := (mean_x * sample_size - (removed_point1.1 + removed_point2.1)) / (sample_size - 2)
  let new_mean_y := (initial_regression mean_x * sample_size - (removed_point1.2 + removed_point2.2)) / (sample_size - 2)
  let new_intercept := new_mean_y - new_slope * new_mean_x
  ∀ x, new_slope * x + new_intercept = 3 * x - 3 :=
by sorry

end new_regression_equation_l3672_367212


namespace star_calculation_l3672_367298

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 - 2*y

-- State the theorem
theorem star_calculation :
  let a := star 5 14
  let b := star 4 6
  star (2^a) (4^b) = -512.421875 := by sorry

end star_calculation_l3672_367298


namespace roots_opposite_signs_l3672_367239

/-- Given an equation (x^2 - dx)/(cx - k) = (m-2)/(m+2) where c, d, and k are constants,
    prove that when m = 2(c - d)/(c + d), the equation has roots which are numerically
    equal but of opposite signs. -/
theorem roots_opposite_signs (c d k : ℝ) :
  let m := 2 * (c - d) / (c + d)
  let f := fun x => (x^2 - d*x) / (c*x - k) - (m - 2) / (m + 2)
  ∃ (r : ℝ), f r = 0 ∧ f (-r) = 0 := by
sorry

end roots_opposite_signs_l3672_367239


namespace shekar_science_score_l3672_367209

/-- Represents a student's scores across 5 subjects -/
structure StudentScores where
  mathematics : ℕ
  science : ℕ
  social_studies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average score -/
def average (s : StudentScores) : ℚ :=
  (s.mathematics + s.science + s.social_studies + s.english + s.biology) / 5

theorem shekar_science_score :
  ∀ (s : StudentScores),
    s.mathematics = 76 →
    s.social_studies = 82 →
    s.english = 67 →
    s.biology = 55 →
    average s = 69 →
    s.science = 65 := by
  sorry

#check shekar_science_score

end shekar_science_score_l3672_367209


namespace sum_of_primes_floor_condition_l3672_367208

theorem sum_of_primes_floor_condition : 
  (∃ p₁ p₂ : ℕ, 
    p₁.Prime ∧ p₂.Prime ∧ p₁ ≠ p₂ ∧
    (∃ n₁ : ℕ+, 5 * p₁ = ⌊(n₁.val ^ 2 : ℚ) / 5⌋) ∧
    (∃ n₂ : ℕ+, 5 * p₂ = ⌊(n₂.val ^ 2 : ℚ) / 5⌋) ∧
    (∀ p : ℕ, p.Prime → 
      (∃ n : ℕ+, 5 * p = ⌊(n.val ^ 2 : ℚ) / 5⌋) → 
      p = p₁ ∨ p = p₂) ∧
    p₁ + p₂ = 52) :=
  sorry

end sum_of_primes_floor_condition_l3672_367208


namespace brooke_math_problems_l3672_367279

/-- The number of math problems Brooke has -/
def num_math_problems : ℕ := sorry

/-- The number of social studies problems Brooke has -/
def num_social_studies_problems : ℕ := 6

/-- The number of science problems Brooke has -/
def num_science_problems : ℕ := 10

/-- The time (in minutes) it takes to solve one math problem -/
def time_per_math_problem : ℚ := 2

/-- The time (in minutes) it takes to solve one social studies problem -/
def time_per_social_studies_problem : ℚ := 1/2

/-- The time (in minutes) it takes to solve one science problem -/
def time_per_science_problem : ℚ := 3/2

/-- The total time (in minutes) it takes Brooke to complete all homework -/
def total_homework_time : ℚ := 48

theorem brooke_math_problems : 
  num_math_problems = 15 ∧
  (num_math_problems : ℚ) * time_per_math_problem + 
  (num_social_studies_problems : ℚ) * time_per_social_studies_problem +
  (num_science_problems : ℚ) * time_per_science_problem = total_homework_time :=
sorry

end brooke_math_problems_l3672_367279


namespace ratio_equality_counterexample_l3672_367226

theorem ratio_equality_counterexample (a b c d : ℝ) 
  (h1 : a ≠ 0) (h2 : c ≠ 0) (h3 : a / b = c / d) : 
  ¬ ((a + d) / (b + c) = a / b) := by
  sorry

end ratio_equality_counterexample_l3672_367226


namespace opposite_of_seven_l3672_367228

/-- The opposite of a real number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℝ) : ℝ := -a

/-- The opposite of 7 is -7. -/
theorem opposite_of_seven : opposite 7 = -7 := by
  sorry

end opposite_of_seven_l3672_367228


namespace rent_increase_proof_l3672_367207

/-- Given a group of 4 friends with an initial average rent and a new average rent after
    one friend's rent is increased, proves that the original rent of the friend whose rent
    was increased is equal to a specific value. -/
theorem rent_increase_proof (initial_avg : ℝ) (new_avg : ℝ) (increase_rate : ℝ) :
  initial_avg = 800 →
  new_avg = 850 →
  increase_rate = 0.25 →
  (4 : ℝ) * new_avg - (4 : ℝ) * initial_avg = increase_rate * ((4 : ℝ) * new_avg - (4 : ℝ) * initial_avg) / increase_rate :=
by sorry

#check rent_increase_proof

end rent_increase_proof_l3672_367207


namespace expression_equality_l3672_367217

theorem expression_equality : (2023^2 - 2015^2) / (2030^2 - 2008^2) = 4/11 := by
  sorry

end expression_equality_l3672_367217


namespace fouad_ahmed_age_multiple_l3672_367289

theorem fouad_ahmed_age_multiple : ∃ x : ℕ, (26 + x) % 11 = 0 ∧ (26 + x) / 11 = 3 := by
  sorry

end fouad_ahmed_age_multiple_l3672_367289


namespace parabola_properties_l3672_367230

/-- Represents a parabola of the form y = ax^2 -/
structure Parabola where
  a : ℝ
  h : a > 0

/-- Compares the steepness of two parabolas at a given x -/
def steeper_at (p1 p2 : Parabola) (x : ℝ) : Prop :=
  p1.a * x^2 > p2.a * x^2

/-- A parabola p1 is considered steeper than p2 if it's steeper for all non-zero x -/
def steeper (p1 p2 : Parabola) : Prop :=
  ∀ x ≠ 0, steeper_at p1 p2 x

/-- A parabola p approaches the x-axis as its 'a' approaches 0 -/
def approaches_x_axis (p : Parabola → Prop) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ q : Parabola, q.a < δ → p q → ∀ x, |q.a * x^2| < ε

theorem parabola_properties :
  ∀ p : Parabola,
    (0 < p.a ∧ p.a < 1 → steeper {a := 1, h := by norm_num} p) ∧
    (p.a > 1 → steeper p {a := 1, h := by norm_num}) ∧
    (approaches_x_axis (λ q ↦ q.a < p.a)) :=
by sorry

end parabola_properties_l3672_367230


namespace parallel_plane_line_l3672_367272

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelPL : Plane → Line → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularPL : Plane → Line → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the theorem
theorem parallel_plane_line 
  (l m n : Line) 
  (α β : Plane) 
  (h_distinct_lines : l ≠ m ∧ l ≠ n ∧ m ≠ n) 
  (h_distinct_planes : α ≠ β) 
  (h_parallel_planes : parallelPP α β) 
  (h_line_in_plane : subset l α) : 
  parallelPL β l :=
sorry

end parallel_plane_line_l3672_367272


namespace book_shelf_problem_l3672_367297

theorem book_shelf_problem (paperbacks hardbacks : ℕ) 
  (h1 : paperbacks = 2)
  (h2 : hardbacks = 6)
  (h3 : Nat.choose paperbacks 1 * Nat.choose hardbacks 2 + 
        Nat.choose paperbacks 2 * Nat.choose hardbacks 1 = 36) :
  paperbacks + hardbacks = 8 := by
sorry

end book_shelf_problem_l3672_367297


namespace min_value_of_f_l3672_367268

def f (x : ℝ) : ℝ := 5 * x^2 - 30 * x + 2000

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 1955 :=
by sorry

end min_value_of_f_l3672_367268


namespace quartic_polynomial_satisfies_conditions_l3672_367224

theorem quartic_polynomial_satisfies_conditions :
  let p : ℝ → ℝ := λ x => -x^4 + 2*x^2 - 5*x + 1
  (p 1 = -3) ∧ (p 2 = -5) ∧ (p 3 = -11) ∧ (p 4 = -27) ∧ (p 5 = -59) := by
  sorry

end quartic_polynomial_satisfies_conditions_l3672_367224


namespace largest_triple_product_digit_sum_l3672_367202

def is_single_digit_prime (p : Nat) : Prop :=
  p ≥ 2 ∧ p < 10 ∧ Nat.Prime p

def is_valid_triple (d e : Nat) : Prop :=
  is_single_digit_prime d ∧ 
  is_single_digit_prime e ∧ 
  Nat.Prime (d + 10 * e)

def product_of_triple (d e : Nat) : Nat :=
  d * e * (d + 10 * e)

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_triple_product_digit_sum :
  ∃ (d e : Nat),
    is_valid_triple d e ∧
    (∀ (d' e' : Nat), is_valid_triple d' e' → product_of_triple d' e' ≤ product_of_triple d e) ∧
    sum_of_digits (product_of_triple d e) = 21 :=
by sorry

end largest_triple_product_digit_sum_l3672_367202


namespace profit_sales_ratio_change_l3672_367255

/-- Calculates the percent change between two ratios -/
def percent_change (old_ratio new_ratio : ℚ) : ℚ :=
  ((new_ratio - old_ratio) / old_ratio) * 100

theorem profit_sales_ratio_change :
  let first_quarter_profit : ℚ := 5
  let first_quarter_sales : ℚ := 15
  let third_quarter_profit : ℚ := 14
  let third_quarter_sales : ℚ := 35
  let first_quarter_ratio := first_quarter_profit / first_quarter_sales
  let third_quarter_ratio := third_quarter_profit / third_quarter_sales
  percent_change first_quarter_ratio third_quarter_ratio = 20 := by
sorry

#eval percent_change (5/15) (14/35)

end profit_sales_ratio_change_l3672_367255


namespace simplify_expression_l3672_367260

theorem simplify_expression (a : ℝ) (h : 2 < a ∧ a < 3) :
  (2 - a)^(1/3) + (3 - a)^(1/4) = 5 - 2*a :=
by sorry

end simplify_expression_l3672_367260


namespace golden_ratio_expression_l3672_367216

theorem golden_ratio_expression (S : ℝ) (h : S^2 + S - 1 = 0) :
  S^(S^(S^2 - S⁻¹) - S⁻¹) - S⁻¹ = 0 := by sorry

end golden_ratio_expression_l3672_367216


namespace sine_cosine_inequality_l3672_367254

theorem sine_cosine_inequality (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c :=
sorry

end sine_cosine_inequality_l3672_367254


namespace subcommittee_formation_count_l3672_367229

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 8
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (choose total_republicans subcommittee_republicans) * (choose total_democrats subcommittee_democrats) = 11760 := by
  sorry

end subcommittee_formation_count_l3672_367229


namespace product_equality_l3672_367205

theorem product_equality : 72519 * 31415.927 = 2277666538.233 := by
  sorry

end product_equality_l3672_367205


namespace perpendicular_line_equation_l3672_367276

/-- Given a line L1 with equation x + 2y - 1 = 0 and a point A(1,2),
    prove that the line L2 passing through A and perpendicular to L1
    has the equation 2x - y = 0 -/
theorem perpendicular_line_equation (L1 : Set (ℝ × ℝ)) (A : ℝ × ℝ) :
  (L1 = {(x, y) | x + 2*y - 1 = 0}) →
  (A = (1, 2)) →
  (∃ L2 : Set (ℝ × ℝ), L2 = {(x, y) | 2*x - y = 0} ∧ 
    A ∈ L2 ∧ 
    (∀ (v w : ℝ × ℝ), v ∈ L1 → w ∈ L1 → v ≠ w → 
      (∀ (p q : ℝ × ℝ), p ∈ L2 → q ∈ L2 → p ≠ q → 
        ((v.1 - w.1) * (p.1 - q.1) + (v.2 - w.2) * (p.2 - q.2) = 0)))) :=
by sorry

end perpendicular_line_equation_l3672_367276


namespace max_n_A_theorem_l3672_367245

/-- A set of four distinct positive integers -/
structure FourSet where
  a₁ : ℕ+
  a₂ : ℕ+
  a₃ : ℕ+
  a₄ : ℕ+
  distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄

/-- Sum of elements in a FourSet -/
def S_A (A : FourSet) : ℕ+ :=
  A.a₁ + A.a₂ + A.a₃ + A.a₄

/-- Number of pairs (i, j) with 1 ≤ i < j ≤ 4 such that (aᵢ + a_j) divides S_A -/
def n_A (A : FourSet) : ℕ :=
  let pairs := [(A.a₁, A.a₂), (A.a₁, A.a₃), (A.a₁, A.a₄), (A.a₂, A.a₃), (A.a₂, A.a₄), (A.a₃, A.a₄)]
  (pairs.filter (fun (x, y) => (S_A A).val % (x + y).val = 0)).length

/-- Theorem stating the maximum value of n_A and the form of A when this maximum is achieved -/
theorem max_n_A_theorem (A : FourSet) :
  n_A A ≤ 4 ∧
  (n_A A = 4 →
    (∃ c : ℕ+, A.a₁ = c ∧ A.a₂ = 5 * c ∧ A.a₃ = 7 * c ∧ A.a₄ = 11 * c) ∨
    (∃ c : ℕ+, A.a₁ = c ∧ A.a₂ = 11 * c ∧ A.a₃ = 19 * c ∧ A.a₄ = 29 * c)) := by
  sorry

end max_n_A_theorem_l3672_367245


namespace tara_had_fifteen_l3672_367203

/-- The amount of money Megan has -/
def megan_money : ℕ := sorry

/-- The amount of money Tara has -/
def tara_money : ℕ := megan_money + 4

/-- The cost of the scooter -/
def scooter_cost : ℕ := 26

/-- Theorem stating that Tara had $15 -/
theorem tara_had_fifteen :
  (megan_money + tara_money = scooter_cost) →
  tara_money = 15 := by
  sorry

end tara_had_fifteen_l3672_367203


namespace reciprocal_sum_property_l3672_367213

theorem reciprocal_sum_property (x y : ℝ) (h : x > 0) (h' : y > 0) (h'' : 1 / x + 1 / y = 1) :
  (x - 1) * (y - 1) = 1 :=
by sorry

end reciprocal_sum_property_l3672_367213


namespace star_divided_by_square_equals_sixteen_l3672_367231

-- Define symbols as natural numbers
variable (triangle circle square star : ℕ)

-- Define the conditions from the problem
axiom condition1 : triangle + triangle = star
axiom condition2 : circle = square + square
axiom condition3 : triangle = circle + circle + circle + circle

-- The theorem to prove
theorem star_divided_by_square_equals_sixteen : star / square = 16 := by
  sorry

end star_divided_by_square_equals_sixteen_l3672_367231


namespace max_a_value_l3672_367284

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) → 
  a ≤ 1 ∧ ∀ b : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - b ≥ 0) → b ≤ a :=
by sorry

end max_a_value_l3672_367284


namespace largest_multiple_of_15_under_500_l3672_367225

theorem largest_multiple_of_15_under_500 : ∃ (n : ℕ), n * 15 = 495 ∧ 
  495 < 500 ∧ 
  ∀ (m : ℕ), m * 15 < 500 → m * 15 ≤ 495 := by
  sorry

end largest_multiple_of_15_under_500_l3672_367225


namespace top_number_after_folds_l3672_367252

/-- Represents a 4x4 grid of numbers -/
def Grid := Fin 4 → Fin 4 → Fin 16

/-- The initial configuration of the grid -/
def initial_grid : Grid :=
  fun i j => ⟨i.val * 4 + j.val + 1, by sorry⟩

/-- Fold the right half over the left half -/
def fold_right_left (g : Grid) : Grid :=
  fun i j => g i (Fin.cast (by sorry) (3 - j))

/-- Fold the top half over the bottom half -/
def fold_top_bottom (g : Grid) : Grid :=
  fun i j => g (Fin.cast (by sorry) (3 - i)) j

/-- Fold the bottom half over the top half -/
def fold_bottom_top (g : Grid) : Grid :=
  fun i j => g (Fin.cast (by sorry) (3 - i)) j

/-- Fold the left half over the right half -/
def fold_left_right (g : Grid) : Grid :=
  fun i j => g i (Fin.cast (by sorry) (3 - j))

/-- Apply all folding operations in sequence -/
def apply_all_folds (g : Grid) : Grid :=
  fold_left_right ∘ fold_bottom_top ∘ fold_top_bottom ∘ fold_right_left $ g

theorem top_number_after_folds :
  (apply_all_folds initial_grid 0 0).val = 1 := by sorry

end top_number_after_folds_l3672_367252


namespace f_strictly_increasing_l3672_367294

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem f_strictly_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < exp 1 → f x₁ < f x₂ := by
  sorry

end f_strictly_increasing_l3672_367294


namespace association_members_after_four_years_l3672_367251

/-- Represents the number of people in the association after k years -/
def association_members (k : ℕ) : ℕ :=
  match k with
  | 0 => 20
  | n + 1 => 4 * association_members n - 18

/-- The number of people in the association after 4 years is 3590 -/
theorem association_members_after_four_years :
  association_members 4 = 3590 := by
  sorry

end association_members_after_four_years_l3672_367251


namespace sum_of_three_consecutive_odd_numbers_l3672_367210

theorem sum_of_three_consecutive_odd_numbers (n : ℕ) (h : n = 21) :
  n + (n + 2) + (n + 4) = 69 := by
  sorry

end sum_of_three_consecutive_odd_numbers_l3672_367210


namespace parabola_coef_sum_l3672_367285

/-- A parabola with equation x = ay^2 + by + c passing through points (6, -3) and (4, -1) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  point1 : 6 = a * (-3)^2 + b * (-3) + c
  point2 : 4 = a * (-1)^2 + b * (-1) + c

/-- The sum of coefficients a, b, and c of the parabola equals -2 -/
theorem parabola_coef_sum (p : Parabola) : p.a + p.b + p.c = -2 := by
  sorry

end parabola_coef_sum_l3672_367285


namespace arithmetic_equation_l3672_367242

theorem arithmetic_equation : 8 + 15 / 3 - 4 * 2 + 2^3 = 13 := by
  sorry

end arithmetic_equation_l3672_367242


namespace set_operations_l3672_367261

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x < 4}

theorem set_operations :
  (A ∪ B = {x | -1 < x ∧ x < 4}) ∧
  (A ∩ B = {x | 1 ≤ x ∧ x < 3}) := by
  sorry

end set_operations_l3672_367261


namespace cubic_equation_implications_l3672_367249

theorem cubic_equation_implications (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_not_equal : ¬(x = y ∧ y = z))
  (h_equation : x^3 + y^3 + z^3 - 3*x*y*z - 3*(x^2 + y^2 + z^2 - x*y - y*z - z*x) = 0) :
  (x + y + z = 3) ∧ 
  (x^2*(1+y) + y^2*(1+z) + z^2*(1+x) > 6) := by
  sorry

end cubic_equation_implications_l3672_367249


namespace quadratic_roots_relation_l3672_367266

theorem quadratic_roots_relation (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ ∃ y, 2*y^2 - 7*y + 6 = 0 ∧ x = y - 3) → 
  c = 3/2 := by
sorry

end quadratic_roots_relation_l3672_367266


namespace max_additional_tiles_l3672_367262

/-- Represents a rectangular board --/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a tile on the board --/
structure Tile :=
  (width : ℕ)
  (height : ℕ)

/-- The number of cells a tile covers --/
def Tile.area (t : Tile) : ℕ := t.width * t.height

/-- The total number of cells on a board --/
def Board.total_cells (b : Board) : ℕ := b.rows * b.cols

/-- The number of cells covered by a list of tiles --/
def covered_cells (tiles : List Tile) : ℕ :=
  tiles.foldl (λ acc t => acc + t.area) 0

theorem max_additional_tiles (board : Board) (initial_tiles : List Tile) :
  board.rows = 10 ∧ 
  board.cols = 9 ∧ 
  initial_tiles.length = 7 ∧ 
  ∀ t ∈ initial_tiles, t.width = 2 ∧ t.height = 1 →
  ∃ (max_additional : ℕ), 
    max_additional = 38 ∧
    covered_cells initial_tiles + 2 * max_additional = board.total_cells :=
by sorry

end max_additional_tiles_l3672_367262


namespace parallel_lines_at_distance_1_perpendicular_lines_at_distance_sqrt2_l3672_367274

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ → Prop

-- Define the distance between a point and a line
def distance_point_line (p : Point) (l : Line) : ℝ := sorry

-- Define when two lines are parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Define when two lines are perpendicular
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define the given lines and point
def line1 : Line := λ x y _ ↦ 3 * x + 4 * y - 2 = 0
def line2 : Line := λ x y _ ↦ x + 3 * y - 5 = 0
def P : Point := (-1, 0)

-- Theorem for the first part
theorem parallel_lines_at_distance_1 :
  ∃ (l1 l2 : Line),
    (∀ x y z, l1 x y z ↔ 3 * x + 4 * y + 3 = z) ∧
    (∀ x y z, l2 x y z ↔ 3 * x + 4 * y - 7 = z) ∧
    parallel l1 line1 ∧
    parallel l2 line1 ∧
    (∀ p, distance_point_line p l1 = 1) ∧
    (∀ p, distance_point_line p l2 = 1) := sorry

-- Theorem for the second part
theorem perpendicular_lines_at_distance_sqrt2 :
  ∃ (l1 l2 : Line),
    (∀ x y z, l1 x y z ↔ 3 * x - y + 9 = z) ∧
    (∀ x y z, l2 x y z ↔ 3 * x - y - 3 = z) ∧
    perpendicular l1 line2 ∧
    perpendicular l2 line2 ∧
    distance_point_line P l1 = Real.sqrt 2 ∧
    distance_point_line P l2 = Real.sqrt 2 := sorry

end parallel_lines_at_distance_1_perpendicular_lines_at_distance_sqrt2_l3672_367274


namespace shaded_area_is_54_l3672_367259

/-- The area of a right triangle with base 12 cm and height 9 cm is 54 cm². -/
theorem shaded_area_is_54 :
  let base : ℝ := 12
  let height : ℝ := 9
  (1 / 2 : ℝ) * base * height = 54 := by sorry

end shaded_area_is_54_l3672_367259


namespace horner_method_equals_direct_evaluation_l3672_367204

/-- Horner's method for evaluating a polynomial --/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (λ acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 + 2x^4 - 3x^2 + 7x - 2 --/
def f (x : ℝ) : ℝ := x^5 + 2*x^4 - 3*x^2 + 7*x - 2

/-- Coefficients of the polynomial in reverse order --/
def coeffs : List ℝ := [-2, 7, 0, -3, 2, 1]

theorem horner_method_equals_direct_evaluation :
  horner coeffs 2 = f 2 := by sorry

end horner_method_equals_direct_evaluation_l3672_367204


namespace rationalize_denominator_l3672_367240

theorem rationalize_denominator (x : ℝ) : 
  x > 0 → (7 / Real.sqrt (98 : ℝ)) = Real.sqrt 2 / 2 := by
  sorry

end rationalize_denominator_l3672_367240


namespace sales_tax_difference_l3672_367206

theorem sales_tax_difference (price : ℝ) (high_rate low_rate : ℝ) 
  (h1 : price = 30)
  (h2 : high_rate = 0.075)
  (h3 : low_rate = 0.07) : 
  price * high_rate - price * low_rate = 0.15 := by
  sorry

#check sales_tax_difference

end sales_tax_difference_l3672_367206


namespace rectangle_side_difference_l3672_367278

theorem rectangle_side_difference (A d x y : ℝ) (h1 : A > 0) (h2 : d > 0) (h3 : x > y) (h4 : x * y = A) (h5 : x^2 + y^2 = d^2) : x - y = 2 * Real.sqrt A := by
  sorry

end rectangle_side_difference_l3672_367278


namespace joans_video_game_cost_l3672_367227

/-- Calculates the total cost of video games with discount and tax --/
def totalCost (basketballPrice racingPrice actionPrice : ℝ) 
               (discount tax : ℝ) : ℝ :=
  let discountedTotal := (basketballPrice + racingPrice + actionPrice) * (1 - discount)
  discountedTotal * (1 + tax)

/-- Theorem stating the total cost of Joan's video game purchase --/
theorem joans_video_game_cost :
  let basketballPrice := 5.2
  let racingPrice := 4.23
  let actionPrice := 7.12
  let discount := 0.1
  let tax := 0.06
  ∃ (cost : ℝ), abs (totalCost basketballPrice racingPrice actionPrice discount tax - cost) < 0.005 ∧ cost = 15.79 :=
by
  sorry


end joans_video_game_cost_l3672_367227


namespace m_range_l3672_367275

theorem m_range (m : ℝ) : 
  (¬(∃ x₀ : ℝ, m * x₀^2 + 1 < 1) ∧ ∀ x : ℝ, x^2 + m*x + 1 ≥ 0) ↔ 
  -2 ≤ m ∧ m ≤ 2 := by
sorry

end m_range_l3672_367275


namespace democrats_in_house_l3672_367235

theorem democrats_in_house (total : ℕ) (difference : ℕ) (democrats : ℕ) : 
  total = 434 → 
  difference = 30 → 
  total = democrats + (democrats + difference) → 
  democrats = 202 := by
sorry

end democrats_in_house_l3672_367235


namespace boys_in_class_l3672_367267

theorem boys_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 56) (h2 : ratio_girls = 4) (h3 : ratio_boys = 3) : 
  (total * ratio_boys) / (ratio_girls + ratio_boys) = 24 := by
  sorry

end boys_in_class_l3672_367267


namespace thirty_percent_less_than_ninety_l3672_367243

theorem thirty_percent_less_than_ninety (x : ℝ) : x = 50 ↔ (1 + 1/4) * x = 90 * (1 - 3/10) := by
  sorry

end thirty_percent_less_than_ninety_l3672_367243


namespace derivative_at_zero_l3672_367234

/-- Given a function f where f(x) = x^2 + 2f'(1), prove that f'(0) = 0 -/
theorem derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2 * (deriv f 1)) :
  deriv f 0 = 0 := by
  sorry

end derivative_at_zero_l3672_367234


namespace binomial_coefficient_sum_l3672_367246

theorem binomial_coefficient_sum (a : ℝ) : 
  (∃ n : ℕ, ∃ x : ℝ, (2 : ℝ)^n = 256 ∧ (∀ k : ℕ, k ≤ n → ∃ c : ℝ, c * (a/x + 3)^(n-k) * x^k = c * (a + 3*x)^(n-k))) → 
  (a = -1 ∨ a = -5) := by
sorry

end binomial_coefficient_sum_l3672_367246


namespace samantha_sleep_hours_l3672_367219

/-- Represents the number of hours Samantha sleeps per night -/
def samantha_sleep : ℝ := 8

/-- Represents the number of hours Samantha's baby sister sleeps per night -/
def baby_sister_sleep : ℝ := 2.5 * samantha_sleep

/-- Represents the number of hours Samantha's father sleeps per night -/
def father_sleep : ℝ := 0.5 * baby_sister_sleep

theorem samantha_sleep_hours :
  samantha_sleep = 8 ∧
  baby_sister_sleep = 2.5 * samantha_sleep ∧
  father_sleep = 0.5 * baby_sister_sleep ∧
  7 * father_sleep = 70 := by
  sorry

#check samantha_sleep_hours

end samantha_sleep_hours_l3672_367219


namespace find_y_when_x_is_12_l3672_367296

-- Define the inverse proportionality constant
def k : ℝ := 675

-- Define the relationship between x and y
def inverse_proportional (x y : ℝ) : Prop := x * y = k

-- State the theorem
theorem find_y_when_x_is_12 (x y : ℝ) 
  (h1 : inverse_proportional x y) 
  (h2 : x + y = 60) 
  (h3 : x = 3 * y) :
  x = 12 → y = 56.25 := by
  sorry

end find_y_when_x_is_12_l3672_367296


namespace midpoint_sum_invariant_l3672_367222

/-- A polygon in the Cartesian plane -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- Create a new polygon from the midpoints of the sides of a given polygon -/
def midpointPolygon (p : Polygon) : Polygon := sorry

/-- Sum of y-coordinates of a polygon's vertices -/
def sumYCoordinates (p : Polygon) : ℝ := sorry

theorem midpoint_sum_invariant (n : ℕ) (Q1 : Polygon) :
  n ≥ 3 →
  Q1.vertices.length = n →
  let Q2 := midpointPolygon Q1
  let Q3 := midpointPolygon Q2
  sumYCoordinates Q3 = sumYCoordinates Q1 := by sorry

end midpoint_sum_invariant_l3672_367222


namespace lemons_count_l3672_367211

/-- Represents the contents of Tania's fruit baskets -/
structure FruitBaskets where
  total_fruits : ℕ
  mangoes : ℕ
  pears : ℕ
  pawpaws : ℕ
  oranges_basket3 : ℕ
  kiwis_basket4 : ℕ
  oranges_basket4 : ℕ

/-- The number of lemons in Tania's baskets -/
def count_lemons (baskets : FruitBaskets) : ℕ :=
  (baskets.total_fruits - (baskets.mangoes + baskets.pears + baskets.pawpaws + 
   baskets.oranges_basket3 + baskets.kiwis_basket4 + baskets.oranges_basket4)) / 3

/-- Theorem stating that the number of lemons in Tania's baskets is 8 -/
theorem lemons_count (baskets : FruitBaskets) 
  (h1 : baskets.total_fruits = 83)
  (h2 : baskets.mangoes = 18)
  (h3 : baskets.pears = 14)
  (h4 : baskets.pawpaws = 10)
  (h5 : baskets.oranges_basket3 = 5)
  (h6 : baskets.kiwis_basket4 = 8)
  (h7 : baskets.oranges_basket4 = 4) :
  count_lemons baskets = 8 := by
  sorry

end lemons_count_l3672_367211


namespace least_n_for_determinant_l3672_367238

theorem least_n_for_determinant (n : ℕ) : n ≥ 1 → (∀ k < n, 2^(k-1) < 2015) → 2^(n-1) ≥ 2015 → n = 12 := by
  sorry

end least_n_for_determinant_l3672_367238


namespace radical_calculation_l3672_367282

theorem radical_calculation : 
  Real.sqrt (1 / 4) * Real.sqrt 16 - (Real.sqrt (1 / 9))⁻¹ - Real.sqrt 0 + Real.sqrt 45 / Real.sqrt 5 = 2 := by
  sorry

end radical_calculation_l3672_367282


namespace left_handed_fiction_readers_count_l3672_367291

/-- Represents a book club with members and their preferences. -/
structure BookClub where
  total_members : ℕ
  fiction_readers : ℕ
  left_handed : ℕ
  right_handed_non_fiction : ℕ

/-- Calculates the number of left-handed fiction readers in the book club. -/
def left_handed_fiction_readers (club : BookClub) : ℕ :=
  club.total_members - (club.left_handed + club.fiction_readers - club.right_handed_non_fiction)

/-- Theorem stating that in a specific book club configuration, 
    the number of left-handed fiction readers is 5. -/
theorem left_handed_fiction_readers_count :
  let club : BookClub := {
    total_members := 25,
    fiction_readers := 15,
    left_handed := 12,
    right_handed_non_fiction := 3
  }
  left_handed_fiction_readers club = 5 := by
  sorry

end left_handed_fiction_readers_count_l3672_367291


namespace mans_upstream_speed_l3672_367265

/-- Given a man's rowing speeds, calculate his upstream speed -/
theorem mans_upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 45)
  (h2 : speed_downstream = 60) :
  speed_still - (speed_downstream - speed_still) = 30 := by
  sorry

#check mans_upstream_speed

end mans_upstream_speed_l3672_367265


namespace equation_solution_l3672_367218

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end equation_solution_l3672_367218


namespace solution_set_when_m_eq_3_m_range_when_f_geq_8_l3672_367287

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

-- Theorem for part I
theorem solution_set_when_m_eq_3 :
  {x : ℝ | f x 3 ≥ 6} = {x : ℝ | x ≤ -2 ∨ x ≥ 4} := by sorry

-- Theorem for part II
theorem m_range_when_f_geq_8 :
  (∀ x : ℝ, f x m ≥ 8) ↔ m ≤ -9 ∨ m ≥ 7 := by sorry

end solution_set_when_m_eq_3_m_range_when_f_geq_8_l3672_367287


namespace quadratic_inequality_solution_l3672_367237

/-- Given a quadratic inequality ax^2 - bx + 1 < 0 with solution set {x | x < -1/2 or x > 2}, 
    prove that a - b = 1/2 -/
theorem quadratic_inequality_solution (a b : ℝ) 
  (h : ∀ x, x < -1/2 ∨ x > 2 ↔ a * x^2 - b * x + 1 < 0) : 
  a - b = 1/2 := by sorry

end quadratic_inequality_solution_l3672_367237
