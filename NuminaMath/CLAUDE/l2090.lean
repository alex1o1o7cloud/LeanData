import Mathlib

namespace range_of_c_l2090_209020

-- Define propositions p and q
def p (c : ℝ) : Prop := 2 < 3 * c
def q (c : ℝ) : Prop := ∀ x : ℝ, 2 * x^2 + 4 * c * x + 1 > 0

-- Theorem statement
theorem range_of_c (c : ℝ) 
  (h : (p c ∨ q c) ∨ (p c ∧ q c)) : 
  2/3 < c ∧ c < Real.sqrt 2 / 2 := by
  sorry

end range_of_c_l2090_209020


namespace floor_plus_self_equation_l2090_209047

theorem floor_plus_self_equation (r : ℝ) : ⌊r⌋ + r = 15.4 ↔ r = 7.4 := by
  sorry

end floor_plus_self_equation_l2090_209047


namespace arithmetic_sequence_sum_l2090_209027

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n, S n = (n : ℝ) / 2 * (a 1 + a n)) →  -- sum formula
  (a 4 + a 8 = 4) →  -- given condition
  (S 11 + a 6 = 24) :=
by sorry

end arithmetic_sequence_sum_l2090_209027


namespace intersection_A_B_union_A_B_range_of_a_l2090_209040

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 * x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2 * x + a > 0}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -1} := by sorry

theorem range_of_a (a : ℝ) (h : C a ∪ B = C a) : a > -4 := by sorry

end intersection_A_B_union_A_B_range_of_a_l2090_209040


namespace square_difference_formula_inapplicable_l2090_209091

theorem square_difference_formula_inapplicable :
  ∀ (a b : ℝ), ¬∃ (x y : ℝ), (-a + b) * (-b + a) = x^2 - y^2 := by
  sorry

#check square_difference_formula_inapplicable

end square_difference_formula_inapplicable_l2090_209091


namespace tan_sum_simplification_l2090_209019

theorem tan_sum_simplification :
  (1 + Real.tan (10 * π / 180)) * (1 + Real.tan (35 * π / 180)) = 2 :=
by
  -- Proof goes here
  sorry

end tan_sum_simplification_l2090_209019


namespace correct_mean_calculation_l2090_209009

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) 
  (incorrect_value1 incorrect_value2 incorrect_value3 : ℚ)
  (correct_value1 correct_value2 correct_value3 : ℚ) :
  n = 50 ∧ 
  initial_mean = 350 ∧
  incorrect_value1 = 150 ∧ correct_value1 = 180 ∧
  incorrect_value2 = 200 ∧ correct_value2 = 235 ∧
  incorrect_value3 = 270 ∧ correct_value3 = 290 →
  (n : ℚ) * initial_mean + (correct_value1 - incorrect_value1) + 
  (correct_value2 - incorrect_value2) + (correct_value3 - incorrect_value3) = n * 351.7 := by
  sorry

end correct_mean_calculation_l2090_209009


namespace negation_of_universal_proposition_l2090_209097

theorem negation_of_universal_proposition (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (¬ ∀ x : ℝ, a^x > 0) ↔ (∃ x₀ : ℝ, a^x₀ ≤ 0) := by sorry

end negation_of_universal_proposition_l2090_209097


namespace circle_symmetry_l2090_209088

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5

-- Define the symmetrical circle
def symmetrical_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

-- Define symmetry with respect to the origin
def symmetrical_wrt_origin (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ g (-x) (-y)

-- Theorem statement
theorem circle_symmetry :
  symmetrical_wrt_origin original_circle symmetrical_circle :=
sorry

end circle_symmetry_l2090_209088


namespace dice_configuration_dots_l2090_209055

/-- Represents a die face with a number of dots -/
structure DieFace where
  dots : Nat
  valid : dots ≥ 1 ∧ dots ≤ 6

/-- Represents a die with six faces -/
structure Die where
  faces : Fin 6 → DieFace
  sum_opposite : ∀ i : Fin 3, (faces i).dots + (faces (i + 3)).dots = 7

/-- Represents the configuration of 4 dice glued together -/
structure DiceConfiguration where
  dice : Fin 4 → Die
  face_c : DieFace
  face_c_is_six : face_c.dots = 6

/-- The theorem to be proved -/
theorem dice_configuration_dots (config : DiceConfiguration) :
  ∃ (face_a face_b face_d : DieFace),
    face_a.dots = 3 ∧
    face_b.dots = 5 ∧
    config.face_c.dots = 6 ∧
    face_d.dots = 5 := by
  sorry

end dice_configuration_dots_l2090_209055


namespace expression_simplification_and_evaluation_l2090_209006

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - 1 / (x + 1)) / (x / (x^2 + 2*x + 1)) = Real.sqrt 2 := by
  sorry

end expression_simplification_and_evaluation_l2090_209006


namespace expected_black_pairs_in_circular_deal_l2090_209061

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of black cards in a standard deck -/
def BlackCards : ℕ := 26

/-- Expected number of pairs of adjacent black cards in a circular deal -/
def ExpectedBlackPairs : ℚ := 650 / 51

theorem expected_black_pairs_in_circular_deal :
  let total_cards := StandardDeck
  let black_cards := BlackCards
  let prob_next_black : ℚ := (black_cards - 1) / (total_cards - 1)
  black_cards * prob_next_black = ExpectedBlackPairs :=
sorry

end expected_black_pairs_in_circular_deal_l2090_209061


namespace cubic_equation_solutions_l2090_209060

theorem cubic_equation_solutions :
  let f : ℂ → ℂ := λ x => (x^3 - 4*x^2*(Real.sqrt 3) + 12*x - 8*(Real.sqrt 3)) + (2*x - 2*(Real.sqrt 3))
  ∃ (z₁ z₂ z₃ : ℂ),
    f z₁ = 0 ∧ f z₂ = 0 ∧ f z₃ = 0 ∧
    z₁ = 2 * Real.sqrt 3 ∧
    z₂ = 2 * Real.sqrt 3 + Complex.I * Real.sqrt 2 ∧
    z₃ = 2 * Real.sqrt 3 - Complex.I * Real.sqrt 2 :=
by
  sorry

end cubic_equation_solutions_l2090_209060


namespace dot_product_NO_NM_l2090_209049

-- Define the function f(x) = x^2 + 3
def f (x : ℝ) : ℝ := x^2 + 3

-- Define the theorem
theorem dot_product_NO_NM :
  ∀ x : ℝ,
  0 < x → x < 2 →
  let M : ℝ × ℝ := (x, f x)
  let N : ℝ × ℝ := (0, 1)
  let O : ℝ × ℝ := (0, 0)
  (M.1 - O.1)^2 + (M.2 - O.2)^2 = 27 →
  let NO : ℝ × ℝ := (N.1 - O.1, N.2 - O.2)
  let NM : ℝ × ℝ := (M.1 - N.1, M.2 - N.2)
  NO.1 * NM.1 + NO.2 * NM.2 = -4 :=
by sorry

end dot_product_NO_NM_l2090_209049


namespace sanchez_grade_calculation_l2090_209077

theorem sanchez_grade_calculation (total_students : ℕ) (below_b_percentage : ℚ) 
  (h1 : total_students = 60) 
  (h2 : below_b_percentage = 40 / 100) : 
  ↑total_students * (1 - below_b_percentage) = 36 := by
  sorry

end sanchez_grade_calculation_l2090_209077


namespace circle_radius_problem_l2090_209056

theorem circle_radius_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  (π * x^2 = π * y^2) →   -- Circles have the same area
  (2 * π * x = 12 * π) →  -- Circumference of circle x is 12π
  (∃ v, y = 2 * v) →      -- Radius of circle y is twice some value v
  (∃ v, y = 2 * v ∧ v = 3) :=
by sorry

end circle_radius_problem_l2090_209056


namespace function_equation_solution_l2090_209079

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y) * (x + f y) = x^2 * f y + y^2 * f x) →
  (∀ x : ℝ, f x = 0 ∨ f x = x) :=
by sorry

end function_equation_solution_l2090_209079


namespace interest_difference_implies_principal_l2090_209021

/-- Proves that if the difference between compound interest and simple interest
    on a sum at 10% per annum for 2 years is Rs. 65, then the sum is Rs. 6500. -/
theorem interest_difference_implies_principal (P : ℝ) : 
  P * (1 + 0.1)^2 - P - (P * 0.1 * 2) = 65 → P = 6500 := by
  sorry

end interest_difference_implies_principal_l2090_209021


namespace bernoulli_inequality_l2090_209083

theorem bernoulli_inequality (n : ℕ+) (x : ℝ) (h : x > -1) :
  (1 + x)^(n : ℝ) ≥ 1 + n * x :=
by sorry

end bernoulli_inequality_l2090_209083


namespace max_value_S_l2090_209039

theorem max_value_S (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c = 5) :
  ∃ (max : ℝ), max = 18 ∧ ∀ (a' b' c' : ℝ), a' ≥ 0 → b' ≥ 0 → c' ≥ 0 → a' + b' + c' = 5 →
    2*a' + 2*a'*b' + a'*b'*c' ≤ max :=
by sorry

end max_value_S_l2090_209039


namespace power_of_power_equals_729_l2090_209067

theorem power_of_power_equals_729 : (3^3)^2 = 729 := by
  sorry

end power_of_power_equals_729_l2090_209067


namespace root_negative_implies_inequality_l2090_209059

theorem root_negative_implies_inequality (a : ℝ) : 
  (∃ x : ℝ, x - 2*a + 4 = 0 ∧ x < 0) → (a - 3) * (a - 4) > 0 := by
  sorry

end root_negative_implies_inequality_l2090_209059


namespace john_squat_wrap_vs_sleeves_l2090_209073

/-- Given a raw squat weight, calculates the difference between
    the additional weight from wraps versus sleeves -/
def wrapVsSleevesDifference (rawSquat : ℝ) : ℝ :=
  0.25 * rawSquat - 30

theorem john_squat_wrap_vs_sleeves :
  wrapVsSleevesDifference 600 = 120 := by
  sorry

end john_squat_wrap_vs_sleeves_l2090_209073


namespace triangle_inequality_l2090_209065

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) :
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7/3 := by
  sorry

end triangle_inequality_l2090_209065


namespace toms_final_amount_l2090_209010

/-- Calculates the final amount of money Tom has after washing cars -/
def final_amount (initial_amount : ℝ) (supply_percentage : ℝ) (total_earnings : ℝ) (earnings_percentage : ℝ) : ℝ :=
  let amount_after_supplies := initial_amount * (1 - supply_percentage)
  let earnings := total_earnings * earnings_percentage
  amount_after_supplies + earnings

/-- Theorem stating that Tom's final amount is 114.5 dollars -/
theorem toms_final_amount :
  final_amount 74 0.15 86 0.6 = 114.5 := by
  sorry

end toms_final_amount_l2090_209010


namespace smallest_divisor_of_4500_l2090_209098

theorem smallest_divisor_of_4500 : 
  ∀ n : ℕ, n > 0 ∧ n ∣ (4499 + 1) → n ≥ 2 :=
by sorry

end smallest_divisor_of_4500_l2090_209098


namespace hidden_message_last_word_l2090_209033

/-- Represents a color in the embroidery --/
inductive Color
| X | Dot | Ampersand | Colon | Star | GreaterThan | LessThan | S | Equals | Zh

/-- Represents a cell in the embroidery grid --/
structure Cell :=
  (number : ℕ)
  (color : Color)

/-- Represents the embroidery system --/
structure EmbroiderySystem :=
  (p : ℕ)
  (grid : List Cell)
  (letterMapping : Fin 33 → Fin 100)
  (colorMapping : Fin 10 → Color)

/-- Represents a decoded message --/
def DecodedMessage := List Char

/-- Function to decode the embroidery --/
def decodeEmbroidery (system : EmbroiderySystem) : DecodedMessage :=
  sorry

/-- The last word of the decoded message --/
def lastWord (message : DecodedMessage) : String :=
  sorry

/-- Theorem stating that the last word of the decoded message is "магистратура" --/
theorem hidden_message_last_word (system : EmbroiderySystem) :
  lastWord (decodeEmbroidery system) = "магистратура" :=
  sorry

end hidden_message_last_word_l2090_209033


namespace bcm_percentage_is_twenty_percent_l2090_209011

/-- The percentage of Black Copper Marans (BCM) in a flock of chickens -/
def bcm_percentage (total_chickens : ℕ) (bcm_hen_percentage : ℚ) (bcm_hens : ℕ) : ℚ :=
  (bcm_hens : ℚ) / (bcm_hen_percentage * total_chickens)

/-- Theorem stating that the percentage of BCM in a flock of 100 chickens is 20%,
    given that 80% of BCM are hens and there are 16 BCM hens -/
theorem bcm_percentage_is_twenty_percent :
  bcm_percentage 100 (4/5) 16 = 1/5 := by
  sorry

#eval bcm_percentage 100 (4/5) 16

end bcm_percentage_is_twenty_percent_l2090_209011


namespace expression_value_l2090_209029

theorem expression_value : 
  let x : ℕ := 3
  x + x * (x ^ (x + 1)) = 246 :=
by sorry

end expression_value_l2090_209029


namespace gcd_of_390_455_546_l2090_209030

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end gcd_of_390_455_546_l2090_209030


namespace function_inequality_l2090_209089

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h' : ∀ x, (x - 2) * deriv f x ≤ 0) : 
  f (-3) + f 3 ≤ 2 * f 2 := by
  sorry

end function_inequality_l2090_209089


namespace xy_squared_equals_one_l2090_209053

theorem xy_squared_equals_one 
  (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 6) : 
  x^2 * y^2 = 1 := by
sorry

end xy_squared_equals_one_l2090_209053


namespace solve_for_y_l2090_209096

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = 5) : y = 17 := by
  sorry

end solve_for_y_l2090_209096


namespace equation_solutions_l2090_209054

/-- The integer part of a real number -/
noncomputable def intPart (x : ℝ) : ℤ :=
  Int.floor x

/-- The fractional part of a real number -/
noncomputable def fracPart (x : ℝ) : ℝ :=
  x - intPart x

/-- The solutions to the equation [x] · {x} = 1991x -/
theorem equation_solutions :
  ∀ x : ℝ, intPart x * fracPart x = 1991 * x ↔ x = 0 ∨ x = -1 / 1992 := by
  sorry

end equation_solutions_l2090_209054


namespace smallest_non_odd_ending_digit_l2090_209086

def is_odd_ending_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def is_digit (n : ℕ) : Prop :=
  n ≤ 9

theorem smallest_non_odd_ending_digit :
  ∀ d : ℕ, is_digit d → 
    (¬is_odd_ending_digit d → d ≥ 0) ∧
    (∀ d' : ℕ, is_digit d' → ¬is_odd_ending_digit d' → d ≤ d') :=
by sorry

end smallest_non_odd_ending_digit_l2090_209086


namespace solution_comparison_l2090_209043

theorem solution_comparison (p q r s : ℝ) (hp : p ≠ 0) (hr : r ≠ 0) :
  (-q / p > -s / r) ↔ (s * r > q * p) :=
by sorry

end solution_comparison_l2090_209043


namespace monomial_exponent_equality_l2090_209048

/-- Two monomials are of the same type if they have the same exponents for each variable. -/
def same_type_monomial (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ i, m1 i = m2 i

/-- The exponents of a monomial of the form x^a * y^b. -/
def monomial_exponents (a b : ℕ) : ℕ → ℕ
| 0 => a  -- exponent of x
| 1 => b  -- exponent of y
| _ => 0  -- all other variables have exponent 0

theorem monomial_exponent_equality (m : ℕ) :
  same_type_monomial (monomial_exponents (2 * m) 3) (monomial_exponents 6 3) →
  m = 3 := by
  sorry

end monomial_exponent_equality_l2090_209048


namespace negation_equivalence_l2090_209026

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 2*x - 3 ≤ 0) ↔ (∀ x : ℝ, x > 0 → x^2 - 2*x - 3 > 0) :=
by sorry

end negation_equivalence_l2090_209026


namespace f_value_at_2_l2090_209045

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 4

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : f a b (-2) = 2 → f a b 2 = -10 := by
  sorry

end f_value_at_2_l2090_209045


namespace total_subjects_is_six_l2090_209051

/-- 
Given a student's marks:
- The average mark in n subjects is 74
- The average mark in 5 subjects is 74
- The mark in the last subject is 74
Prove that the total number of subjects is 6
-/
theorem total_subjects_is_six (n : ℕ) (average_n : ℝ) (average_5 : ℝ) (last_subject : ℝ) :
  average_n = 74 →
  average_5 = 74 →
  last_subject = 74 →
  n * average_n = 5 * average_5 + last_subject →
  n = 6 := by
  sorry

end total_subjects_is_six_l2090_209051


namespace pie_slices_served_yesterday_l2090_209057

def slices_served_yesterday (lunch_today dinner_today total_today : ℕ) : ℕ :=
  total_today - (lunch_today + dinner_today)

theorem pie_slices_served_yesterday : 
  slices_served_yesterday 7 5 12 = 0 :=
by
  sorry

end pie_slices_served_yesterday_l2090_209057


namespace scientific_notation_of_small_number_l2090_209068

theorem scientific_notation_of_small_number : 
  ∃ (a : ℝ) (n : ℤ), 0.0000001 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 := by
  sorry

end scientific_notation_of_small_number_l2090_209068


namespace chessboard_3x1_rectangles_impossible_l2090_209023

theorem chessboard_3x1_rectangles_impossible : ¬ ∃ n : ℕ, 3 * n = 64 := by sorry

end chessboard_3x1_rectangles_impossible_l2090_209023


namespace intersection_condition_l2090_209003

open Set Real

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem intersection_condition (m : ℝ) :
  A ∩ B m = B m ↔ ((-2 * sqrt 2 < m ∧ m < 2 * sqrt 2) ∨ m = 3) :=
sorry

end intersection_condition_l2090_209003


namespace largest_of_five_consecutive_integers_l2090_209035

theorem largest_of_five_consecutive_integers (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧  -- all positive
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧  -- consecutive
  a * b * c * d * e = 15120 →  -- product is 15120
  e = 10 :=  -- largest is 10
by sorry

end largest_of_five_consecutive_integers_l2090_209035


namespace min_type_a_robots_l2090_209018

/-- Represents the material handling capacity of robot A in kg per hour -/
def robot_a_capacity : ℝ := 150

/-- Represents the material handling capacity of robot B in kg per hour -/
def robot_b_capacity : ℝ := 120

/-- Represents the total number of robots to be purchased -/
def total_robots : ℕ := 20

/-- Represents the minimum material handling requirement in kg per hour -/
def min_handling_requirement : ℝ := 2800

/-- Calculates the total material handling capacity given the number of type A robots -/
def total_capacity (num_a : ℕ) : ℝ :=
  (num_a : ℝ) * robot_a_capacity + ((total_robots - num_a) : ℝ) * robot_b_capacity

/-- States that the minimum number of type A robots required is 14 -/
theorem min_type_a_robots : 
  ∀ n : ℕ, n < 14 → total_capacity n < min_handling_requirement ∧
  total_capacity 14 ≥ min_handling_requirement := by sorry

end min_type_a_robots_l2090_209018


namespace square_of_binomial_constant_l2090_209074

theorem square_of_binomial_constant (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 9*x^2 + 27*x + a = (3*x + b)^2) → a = 81/4 := by
  sorry

end square_of_binomial_constant_l2090_209074


namespace present_age_of_b_l2090_209004

theorem present_age_of_b (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) → 
  (a = b + 11) → 
  b = 41 := by
sorry

end present_age_of_b_l2090_209004


namespace money_redistribution_total_l2090_209017

/-- Represents the money redistribution problem with three friends -/
def MoneyRedistribution (a j t : ℚ) : Prop :=
  -- Initial conditions
  t = 36 ∧
  -- After Amy's redistribution
  ∃ a1 j1 t1,
    t1 = 2 * t ∧
    j1 = 2 * j ∧
    a1 = a - (t + j) ∧
    -- After Jan's redistribution
    ∃ a2 j2 t2,
      t2 = 2 * t1 ∧
      a2 = 2 * a1 ∧
      j2 = 2 * j - (a1 + 72) ∧
      -- After Toy's redistribution
      ∃ a3 j3 t3,
        a3 = 2 * a2 ∧
        j3 = 2 * j2 ∧
        t3 = t2 - (a2 + j2) ∧
        t3 = 36 ∧
        a3 + j3 + t3 = 252

/-- The theorem stating that the total amount of money is 252 -/
theorem money_redistribution_total (a j t : ℚ) :
  MoneyRedistribution a j t → a + j + t = 252 := by
  sorry

end money_redistribution_total_l2090_209017


namespace sqrt_sum_equals_six_sqrt_five_l2090_209076

theorem sqrt_sum_equals_six_sqrt_five :
  Real.sqrt ((5 - 3 * Real.sqrt 5) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 5) ^ 2) = 6 * Real.sqrt 5 := by
  sorry

end sqrt_sum_equals_six_sqrt_five_l2090_209076


namespace star_drawing_probability_l2090_209058

def total_stars : ℕ := 12
def red_stars : ℕ := 3
def gold_stars : ℕ := 4
def silver_stars : ℕ := 5
def stars_drawn : ℕ := 6

theorem star_drawing_probability : 
  (red_stars / total_stars) * 
  (Nat.choose gold_stars 3 * Nat.choose silver_stars 2) / 
  (Nat.choose (total_stars - 1) (stars_drawn - 1)) = 5 / 231 := by
  sorry

end star_drawing_probability_l2090_209058


namespace first_us_space_shuttle_is_columbia_l2090_209028

/-- Represents a space shuttle -/
structure SpaceShuttle where
  name : String
  country : String
  year : Nat
  manned_flight_completed : Bool

/-- The world's first space shuttle developed by the United States in 1981 -/
def first_us_space_shuttle : SpaceShuttle :=
  { name := "Columbia"
  , country := "United States"
  , year := 1981
  , manned_flight_completed := true }

/-- Theorem stating that the first US space shuttle's name is Columbia -/
theorem first_us_space_shuttle_is_columbia :
  first_us_space_shuttle.name = "Columbia" :=
by sorry

end first_us_space_shuttle_is_columbia_l2090_209028


namespace min_value_fraction_sum_l2090_209094

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (9 / a + 1 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 9 / a₀ + 1 / b₀ = 16 := by
  sorry

end min_value_fraction_sum_l2090_209094


namespace equation_satisfied_l2090_209044

theorem equation_satisfied (a b c : ℤ) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 3 ↔ (a = c + 1 ∧ b - 1 = a) :=
by sorry

end equation_satisfied_l2090_209044


namespace problem_statement_l2090_209038

theorem problem_statement (x y : ℝ) : 
  let a := x^3 * y
  let b := x^2 * y^2
  let c := x * y^3
  (a * c + b^2 - 2 * x^4 * y^4 = 0) ∧ 
  (a * y^2 + c * x^2 = 2 * x * y * b) ∧ 
  ¬(∀ x y : ℝ, a * b * c + b^3 > 0) :=
by sorry

end problem_statement_l2090_209038


namespace young_inequality_l2090_209072

theorem young_inequality (p q a b : ℝ) (hp : 0 < p) (hq : 0 < q) (hpq : 1/p + 1/q = 1) (ha : 0 < a) (hb : 0 < b) :
  a * b ≤ a^p / p + b^q / q := by
  sorry

end young_inequality_l2090_209072


namespace largest_prime_factor_of_sum_of_divisors_360_l2090_209037

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_360 :
  ∃ (N : ℕ), sum_of_divisors 360 = N ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ N → p ≤ 13) ∧
  13 ∣ N ∧ Nat.Prime 13 :=
sorry

end largest_prime_factor_of_sum_of_divisors_360_l2090_209037


namespace total_time_first_to_seventh_l2090_209046

/-- Represents the travel times between stations in hours -/
def travel_times : List Real := [3, 2, 1.5, 4, 1, 2.5]

/-- Represents the break times at stations in minutes -/
def break_times : List Real := [45, 30, 15]

/-- Converts hours to minutes -/
def hours_to_minutes (hours : Real) : Real := hours * 60

/-- Calculates the total travel time in minutes -/
def total_travel_time : Real := (travel_times.map hours_to_minutes).sum

/-- Calculates the total break time in minutes -/
def total_break_time : Real := break_times.sum

/-- Theorem stating the total time from first to seventh station -/
theorem total_time_first_to_seventh : 
  total_travel_time + total_break_time = 930 := by sorry

end total_time_first_to_seventh_l2090_209046


namespace correct_senior_sample_l2090_209069

/-- Represents the number of students to be selected from each grade level in a stratified sample -/
structure StratifiedSample where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- Calculates the correct stratified sample given the school's demographics -/
def calculateStratifiedSample (totalStudents : ℕ) (freshmen : ℕ) (sophomoreProbability : ℚ) (sampleSize : ℕ) : StratifiedSample :=
  sorry

theorem correct_senior_sample :
  let totalStudents : ℕ := 2000
  let freshmen : ℕ := 760
  let sophomoreProbability : ℚ := 37/100
  let sampleSize : ℕ := 20
  let sample := calculateStratifiedSample totalStudents freshmen sophomoreProbability sampleSize
  sample.seniors = 5 := by sorry

end correct_senior_sample_l2090_209069


namespace clover_total_distance_l2090_209082

/-- Clover's daily morning walk distance in miles -/
def morning_walk : ℝ := 1.5

/-- Clover's daily evening walk distance in miles -/
def evening_walk : ℝ := 1.5

/-- Number of days Clover walks -/
def days : ℕ := 30

/-- Theorem stating the total distance Clover walks in 30 days -/
theorem clover_total_distance : 
  (morning_walk + evening_walk) * days = 90 := by
  sorry

end clover_total_distance_l2090_209082


namespace fruit_cost_l2090_209084

/-- The cost of fruit combinations -/
theorem fruit_cost (x y z : ℚ) : 
  (2 * x + y + 4 * z = 6) →
  (4 * x + 2 * y + 2 * z = 4) →
  (4 * x + 2 * y + 5 * z = 8) :=
by sorry

end fruit_cost_l2090_209084


namespace probability_theorem_l2090_209015

def red_marbles : ℕ := 15
def blue_marbles : ℕ := 9
def green_marbles : ℕ := 6
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles

def probability_two_blue_one_red_one_green : ℚ :=
  (Nat.choose blue_marbles 2 * Nat.choose red_marbles 1 * Nat.choose green_marbles 1) /
  Nat.choose total_marbles 4

theorem probability_theorem :
  probability_two_blue_one_red_one_green = 5 / 812 := by
  sorry

end probability_theorem_l2090_209015


namespace probability_of_no_defective_pens_l2090_209034

theorem probability_of_no_defective_pens (total_pens : Nat) (defective_pens : Nat) (pens_bought : Nat) :
  total_pens = 12 →
  defective_pens = 3 →
  pens_bought = 2 →
  (1 - defective_pens / total_pens) * (1 - (defective_pens) / (total_pens - 1)) = 6/11 := by
  sorry

end probability_of_no_defective_pens_l2090_209034


namespace meals_left_theorem_l2090_209008

/-- Calculates the number of meals left to be distributed given the initial number of meals,
    additional meals provided, and meals already distributed. -/
def meals_left_to_distribute (initial_meals : ℕ) (additional_meals : ℕ) (distributed_meals : ℕ) : ℕ :=
  initial_meals + additional_meals - distributed_meals

/-- Theorem stating that the number of meals left to distribute is correct
    given the problem conditions. -/
theorem meals_left_theorem (initial_meals additional_meals distributed_meals : ℕ)
    (h1 : initial_meals = 113)
    (h2 : additional_meals = 50)
    (h3 : distributed_meals = 85) :
    meals_left_to_distribute initial_meals additional_meals distributed_meals = 78 := by
  sorry

end meals_left_theorem_l2090_209008


namespace collinear_vectors_m_equals_six_l2090_209085

/-- Two vectors are collinear if the determinant of their components is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given plane vectors a and b, if they are collinear, then m = 6 -/
theorem collinear_vectors_m_equals_six :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-3, m)
  collinear a b → m = 6 := by
sorry

end collinear_vectors_m_equals_six_l2090_209085


namespace valid_x_values_l2090_209081

def is_valid_x (x : ℕ) : Prop :=
  13 ≤ x ∧ x ≤ 20 ∧
  (132 + x) % 3 = 0 ∧
  ∃ (s : ℕ), 3 * s = 132 + 3 * x

theorem valid_x_values :
  ∀ x : ℕ, is_valid_x x ↔ (x = 15 ∨ x = 18) :=
sorry

end valid_x_values_l2090_209081


namespace min_value_of_sequence_l2090_209062

theorem min_value_of_sequence (a : ℕ → ℝ) :
  a 1 = 2 →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2 * n) →
  (∀ n : ℕ, n ≥ 1 → a n / n ≥ 2) ∧ (∃ n : ℕ, n ≥ 1 ∧ a n / n = 2) :=
by sorry

end min_value_of_sequence_l2090_209062


namespace max_elves_theorem_l2090_209071

/-- Represents the type of inhabitant -/
inductive InhabitantType
| Elf
| Dwarf

/-- Represents whether an inhabitant wears a cap -/
inductive CapStatus
| WithCap
| WithoutCap

/-- Represents the statement an inhabitant can make -/
inductive Statement
| RightIsElf
| RightHasCap

/-- Represents an inhabitant in the line -/
structure Inhabitant :=
  (type : InhabitantType)
  (capStatus : CapStatus)
  (statement : Statement)

/-- Determines if an inhabitant tells the truth based on their type and cap status -/
def tellsTruth (i : Inhabitant) : Prop :=
  match i.type, i.capStatus with
  | InhabitantType.Elf, CapStatus.WithoutCap => True
  | InhabitantType.Elf, CapStatus.WithCap => False
  | InhabitantType.Dwarf, CapStatus.WithoutCap => False
  | InhabitantType.Dwarf, CapStatus.WithCap => True

/-- Represents the line of inhabitants -/
def Line := Vector Inhabitant 60

/-- Checks if the line configuration is valid according to the problem rules -/
def isValidLine (line : Line) : Prop := sorry

/-- Counts the number of elves without caps in the line -/
def countElvesWithoutCaps (line : Line) : Nat := sorry

/-- Counts the number of elves with caps in the line -/
def countElvesWithCaps (line : Line) : Nat := sorry

/-- Main theorem: Maximum number of elves without caps is 59 and with caps is 30 -/
theorem max_elves_theorem (line : Line) (h : isValidLine line) : 
  countElvesWithoutCaps line ≤ 59 ∧ countElvesWithCaps line ≤ 30 := by sorry

end max_elves_theorem_l2090_209071


namespace orchestra_members_count_l2090_209092

theorem orchestra_members_count : ∃! x : ℕ, 
  150 < x ∧ x < 250 ∧ 
  x % 4 = 2 ∧ 
  x % 5 = 3 ∧ 
  x % 8 = 4 ∧ 
  x % 9 = 5 ∧ 
  x = 58 :=
by sorry

end orchestra_members_count_l2090_209092


namespace constant_derivative_implies_linear_l2090_209064

/-- A function whose derivative is zero everywhere has a straight line graph -/
theorem constant_derivative_implies_linear (f : ℝ → ℝ) :
  (∀ x, deriv f x = 0) → ∃ a b : ℝ, ∀ x, f x = a * x + b :=
sorry

end constant_derivative_implies_linear_l2090_209064


namespace alexis_initial_budget_l2090_209041

/-- Alexis's shopping expenses and remaining budget --/
structure ShoppingBudget where
  shirt : ℕ
  pants : ℕ
  coat : ℕ
  socks : ℕ
  belt : ℕ
  shoes : ℕ
  remaining : ℕ

/-- Calculate the initial budget given the shopping expenses and remaining amount --/
def initialBudget (s : ShoppingBudget) : ℕ :=
  s.shirt + s.pants + s.coat + s.socks + s.belt + s.shoes + s.remaining

/-- Alexis's actual shopping expenses and remaining budget --/
def alexisShopping : ShoppingBudget :=
  { shirt := 30
  , pants := 46
  , coat := 38
  , socks := 11
  , belt := 18
  , shoes := 41
  , remaining := 16 }

/-- Theorem stating that Alexis's initial budget was $200 --/
theorem alexis_initial_budget :
  initialBudget alexisShopping = 200 := by
  sorry

end alexis_initial_budget_l2090_209041


namespace sqrt_equation_solution_l2090_209090

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt ((2 / x) + 2) = 3 / 2 → x = 8 := by
sorry

end sqrt_equation_solution_l2090_209090


namespace min_triangulation_l2090_209095

/-- A regular polygon with n sides, where n ≥ 5 -/
structure RegularPolygon where
  n : ℕ
  n_ge_5 : n ≥ 5

/-- A triangulation of a regular polygon -/
structure Triangulation (p : RegularPolygon) where
  num_triangles : ℕ
  is_valid : Bool  -- Represents the validity of the triangulation

/-- The number of acute triangles in a valid triangulation is at least n -/
def min_acute_triangles (p : RegularPolygon) (t : Triangulation p) : Prop :=
  t.is_valid → t.num_triangles ≥ p.n

/-- The number of obtuse triangles in a valid triangulation is at least n -/
def min_obtuse_triangles (p : RegularPolygon) (t : Triangulation p) : Prop :=
  t.is_valid → t.num_triangles ≥ p.n

/-- The main theorem: both acute and obtuse triangulations have a minimum of n triangles -/
theorem min_triangulation (p : RegularPolygon) :
  (∀ t : Triangulation p, min_acute_triangles p t) ∧
  (∀ t : Triangulation p, min_obtuse_triangles p t) :=
sorry

end min_triangulation_l2090_209095


namespace prob_non_yellow_specific_l2090_209025

/-- The probability of selecting a non-yellow jelly bean -/
def prob_non_yellow (red green yellow blue : ℕ) : ℚ :=
  (red + green + blue) / (red + green + yellow + blue)

/-- Theorem: The probability of selecting a non-yellow jelly bean from a bag
    containing 4 red, 7 green, 9 yellow, and 10 blue jelly beans is 7/10 -/
theorem prob_non_yellow_specific : prob_non_yellow 4 7 9 10 = 7/10 := by
  sorry

end prob_non_yellow_specific_l2090_209025


namespace solution_set_part_i_range_of_m_part_ii_l2090_209000

-- Define the function f
def f (x m : ℝ) : ℝ := |2*x| + |2*x + 3| + m

-- Part I
theorem solution_set_part_i : 
  {x : ℝ | f x (-2) ≤ 3} = {x : ℝ | -2 ≤ x ∧ x ≤ 1/2} := by sorry

-- Part II
theorem range_of_m_part_ii :
  ∀ m : ℝ, (∀ x < 0, f x m ≥ x + 2/x) → m ≥ -3 - 2*Real.sqrt 2 := by sorry

end solution_set_part_i_range_of_m_part_ii_l2090_209000


namespace max_chocolates_bob_l2090_209031

/-- Given that Bob and Carol share 36 chocolates, and Carol eats a positive multiple
    of Bob's chocolates, prove that the maximum number of chocolates Bob could have eaten is 18. -/
theorem max_chocolates_bob (total : ℕ) (bob carol : ℕ) (k : ℕ) : 
  total = 36 →
  bob + carol = total →
  carol = k * bob →
  k > 0 →
  bob ≤ 18 := by
  sorry

end max_chocolates_bob_l2090_209031


namespace product_equality_l2090_209016

theorem product_equality (a b : ℝ) (h1 : 4 * a = 30) (h2 : 5 * b = 30) : 40 * a * b = 1800 := by
  sorry

end product_equality_l2090_209016


namespace f_min_at_neg_one_l2090_209032

/-- The quadratic function we want to minimize -/
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 2

/-- The theorem stating that f is minimized at x = -1 -/
theorem f_min_at_neg_one :
  ∀ x : ℝ, f (-1) ≤ f x :=
sorry

end f_min_at_neg_one_l2090_209032


namespace quadratic_sum_l2090_209078

-- Define the quadratic function
def f (x : ℝ) : ℝ := 4 * x^2 - 28 * x - 48

-- Define the completed square form
def g (x a b c : ℝ) : ℝ := a * (x + b)^2 + c

-- Theorem statement
theorem quadratic_sum (a b c : ℝ) :
  (∀ x, f x = g x a b c) → a + b + c = -96.5 := by
  sorry

end quadratic_sum_l2090_209078


namespace curve_C_properties_l2090_209063

noncomputable section

/-- Curve C in parametric form -/
def curve_C (φ : ℝ) : ℝ × ℝ := (3 * Real.cos φ, 3 + 3 * Real.sin φ)

/-- Polar equation of a curve -/
structure PolarEquation where
  f : ℝ → ℝ

/-- Line with slope angle and passing through a point -/
structure Line where
  slope_angle : ℝ
  point : ℝ × ℝ

/-- Intersection points of a line and a curve -/
structure Intersection where
  M : ℝ × ℝ
  N : ℝ × ℝ

/-- Main theorem statement -/
theorem curve_C_properties :
  ∃ (polar_eq : PolarEquation) (l : Line) (int : Intersection),
    (∀ θ : ℝ, polar_eq.f θ = 6 * Real.sin θ) ∧
    l.slope_angle = 135 * π / 180 ∧
    l.point = (1, 2) ∧
    (let (xM, yM) := int.M
     let (xN, yN) := int.N
     1 / Real.sqrt ((xM - 1)^2 + (yM - 2)^2) +
     1 / Real.sqrt ((xN - 1)^2 + (yN - 2)^2) = 6 / 7) := by
  sorry

end

end curve_C_properties_l2090_209063


namespace range_of_expression_l2090_209007

theorem range_of_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y ≤ 1) :
  2/3 ≤ 4*x^2 + 4*y^2 + (1 - x - y)^2 ∧ 4*x^2 + 4*y^2 + (1 - x - y)^2 ≤ 4 := by
  sorry

end range_of_expression_l2090_209007


namespace arithmetic_geometric_mean_inequality_two_variables_l2090_209080

theorem arithmetic_geometric_mean_inequality_two_variables
  (a b : ℝ) : (a^2 + b^2) / 2 ≥ a * b ∧ 
  ((a^2 + b^2) / 2 = a * b ↔ a = b) := by
  sorry

end arithmetic_geometric_mean_inequality_two_variables_l2090_209080


namespace geometric_sequence_condition_l2090_209005

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_condition (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_pos : a 1 > 0) :
  (∀ h : a 3 < a 6, a 1 < a 3) ∧
  ¬(∀ h : a 1 < a 3, a 3 < a 6) :=
sorry

end geometric_sequence_condition_l2090_209005


namespace intersection_of_M_and_N_l2090_209013

def M : Set ℕ := {1, 2, 3, 6, 7}
def N : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end intersection_of_M_and_N_l2090_209013


namespace angle_east_southwest_is_135_l2090_209093

/-- Represents a circle with 8 equally spaced rays --/
structure EightRayCircle where
  /-- The measure of the angle between adjacent rays in degrees --/
  angle_between_rays : ℝ
  /-- The angle between adjacent rays is 45° --/
  angle_is_45 : angle_between_rays = 45

/-- The measure of the smaller angle between East and Southwest rays in degrees --/
def angle_east_southwest (circle : EightRayCircle) : ℝ :=
  3 * circle.angle_between_rays

theorem angle_east_southwest_is_135 (circle : EightRayCircle) :
  angle_east_southwest circle = 135 := by
  sorry

end angle_east_southwest_is_135_l2090_209093


namespace no_solution_exists_l2090_209050

theorem no_solution_exists : ¬∃ (x y : ℤ), (x + y = 2021 ∧ (10*x + y = 2221 ∨ x + 10*y = 2221)) := by
  sorry

end no_solution_exists_l2090_209050


namespace p_plus_q_equals_26_l2090_209022

theorem p_plus_q_equals_26 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-2 * x^2 + 8 * x + 34) / (x - 3)) →
  P + Q = 26 := by
sorry

end p_plus_q_equals_26_l2090_209022


namespace sin_double_alpha_l2090_209042

theorem sin_double_alpha (α : Real) : 
  Real.sin (45 * π / 180 + α) = Real.sqrt 5 / 5 → Real.sin (2 * α) = -3 / 5 := by
sorry

end sin_double_alpha_l2090_209042


namespace quadratic_sum_l2090_209099

-- Define the quadratic function
def f (x : ℝ) : ℝ := -4 * x^2 + 20 * x - 88

-- Define the general form a(x+b)^2 + c
def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x + b)^2 + c

-- Theorem statement
theorem quadratic_sum (a b c : ℝ) :
  (∀ x, f x = g a b c x) → a + b + c = -70.5 := by
  sorry

end quadratic_sum_l2090_209099


namespace triangle_vector_division_l2090_209024

/-- Given a triangle ABC with point M on side BC such that BM:MC = 2:5,
    and vectors AB = a and AC = b, prove that AM = (2/7)a + (5/7)b. -/
theorem triangle_vector_division (A B C M : EuclideanSpace ℝ (Fin 3))
  (a b : EuclideanSpace ℝ (Fin 3)) (h : B ≠ C) :
  (B - M) = (5 / 7 : ℝ) • (C - B) →
  (A - B) = a →
  (A - C) = -b →
  (A - M) = (2 / 7 : ℝ) • a + (5 / 7 : ℝ) • b := by
  sorry

end triangle_vector_division_l2090_209024


namespace julys_husband_age_l2090_209087

/-- Given information about Hannah and July's ages, and July's husband's age relative to July,
    prove that July's husband is 25 years old. -/
theorem julys_husband_age :
  ∀ (hannah_initial_age : ℕ) 
    (july_initial_age : ℕ) 
    (years_passed : ℕ) 
    (age_difference_husband : ℕ),
  hannah_initial_age = 6 →
  hannah_initial_age = 2 * july_initial_age →
  years_passed = 20 →
  age_difference_husband = 2 →
  july_initial_age + years_passed + age_difference_husband = 25 :=
by
  sorry

end julys_husband_age_l2090_209087


namespace original_number_proof_l2090_209014

theorem original_number_proof (x : ℝ) : x * 1.2 = 1800 → x = 1500 := by
  sorry

end original_number_proof_l2090_209014


namespace min_y_value_l2090_209052

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 14*x + 48*y) : y ≥ -1 := by
  sorry

end min_y_value_l2090_209052


namespace expression_factorization_l2090_209036

theorem expression_factorization (b : ℝ) :
  (8 * b^3 + 104 * b^2 - 9) - (-9 * b^3 + b^2 - 9) = b^2 * (17 * b + 103) := by
  sorry

end expression_factorization_l2090_209036


namespace extended_volume_calculation_l2090_209001

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points that are either inside or within one unit
    of a rectangular parallelepiped with the given dimensions -/
def extended_volume (d : Dimensions) : ℝ :=
  sorry

/-- The dimensions of the given rectangular parallelepiped -/
def given_dimensions : Dimensions :=
  { length := 2, width := 3, height := 7 }

theorem extended_volume_calculation :
  extended_volume given_dimensions = (372 + 112 * Real.pi) / 3 := by
  sorry

end extended_volume_calculation_l2090_209001


namespace divisible_by_six_l2090_209075

theorem divisible_by_six (n : ℕ) : ∃ k : ℤ, (n - 1) * n * (n^3 + 1) = 6 * k := by
  sorry

end divisible_by_six_l2090_209075


namespace number_equality_l2090_209012

theorem number_equality (x : ℝ) (h1 : x > 0) (h2 : (2/3) * x = (49/216) * (1/x)) : x = 7/12 := by
  sorry

end number_equality_l2090_209012


namespace function_point_relation_l2090_209066

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is indeed the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Given condition: f^(-1)(-2) = 0
axiom condition : f_inv (-2) = 0

-- Theorem to prove
theorem function_point_relation :
  f (-5 + 5) = -2 :=
sorry

end function_point_relation_l2090_209066


namespace grandma_olga_grandchildren_l2090_209070

/-- Represents the number of grandchildren Grandma Olga has -/
def total_grandchildren (num_daughters num_sons : ℕ) 
                        (sons_per_daughter daughters_per_son : ℕ) : ℕ :=
  num_daughters * sons_per_daughter + num_sons * daughters_per_son

/-- Theorem stating that Grandma Olga has 33 grandchildren -/
theorem grandma_olga_grandchildren : 
  total_grandchildren 3 3 6 5 = 33 := by
  sorry

end grandma_olga_grandchildren_l2090_209070


namespace right_triangle_sides_from_medians_l2090_209002

/-- Given a right-angled triangle with medians ka and kb, prove the lengths of its sides. -/
theorem right_triangle_sides_from_medians (ka kb : ℝ) 
  (h_ka : ka = 30) (h_kb : kb = 40) : ∃ (a b c : ℝ),
  -- Definition of medians
  ka^2 = (1/4) * (2*b^2 + 2*c^2 - a^2) ∧ 
  kb^2 = (1/4) * (2*a^2 + 2*c^2 - b^2) ∧
  -- Pythagorean theorem
  a^2 + b^2 = c^2 ∧
  -- Side lengths
  a = 20 * Real.sqrt (11/3) ∧
  b = 40 / Real.sqrt 3 ∧
  c = 20 * Real.sqrt 5 := by
sorry

end right_triangle_sides_from_medians_l2090_209002
