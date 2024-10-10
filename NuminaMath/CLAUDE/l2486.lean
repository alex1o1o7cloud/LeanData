import Mathlib

namespace line_integral_equals_five_halves_l2486_248687

/-- Line segment from (0,0) to (4,3) -/
def L : Set (ℝ × ℝ) := {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (4*t, 3*t)}

/-- The function to be integrated -/
def f (p : ℝ × ℝ) : ℝ := p.1 - p.2

theorem line_integral_equals_five_halves :
  ∫ p in L, f p = 5/2 := by sorry

end line_integral_equals_five_halves_l2486_248687


namespace special_day_price_l2486_248655

theorem special_day_price (original_price : ℝ) (first_discount_percent : ℝ) (second_discount_percent : ℝ) : 
  original_price = 240 →
  first_discount_percent = 40 →
  second_discount_percent = 25 →
  let first_discounted_price := original_price * (1 - first_discount_percent / 100)
  let special_day_price := first_discounted_price * (1 - second_discount_percent / 100)
  special_day_price = 108 := by
sorry

end special_day_price_l2486_248655


namespace remainder_of_198_digits_mod_9_l2486_248652

/-- Represents the sequence of digits formed by concatenating consecutive natural numbers -/
def consecutiveDigitSequence (n : ℕ) : List ℕ :=
  sorry

/-- Computes the sum of digits in the sequence up to the nth digit -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the sum of the first 198 digits in the sequence,
    when divided by 9, has a remainder of 6 -/
theorem remainder_of_198_digits_mod_9 :
  sumOfDigits 198 % 9 = 6 := by
  sorry

end remainder_of_198_digits_mod_9_l2486_248652


namespace unique_n_with_prime_divisor_property_l2486_248605

theorem unique_n_with_prime_divisor_property : 
  ∃! (n : ℕ), n > 0 ∧ 
  (∃ (p : ℕ), Prime p ∧
    (∀ (q : ℕ), Prime q → q ∣ (n^2 + 3) → q ≤ p) ∧
    (∀ (q : ℕ), Prime q → q ∣ (n^4 + 6) → p ≤ q) ∧
    p ∣ (n^2 + 3) ∧ p ∣ (n^4 + 6)) ∧
  n = 3 := by
sorry

end unique_n_with_prime_divisor_property_l2486_248605


namespace product_of_numbers_l2486_248693

theorem product_of_numbers (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ = 2 * Real.sqrt 1703)
  (h2 : |x₁ - x₂| = 90) : 
  x₁ * x₂ = -322 := by
sorry

end product_of_numbers_l2486_248693


namespace customers_left_l2486_248625

theorem customers_left (initial_customers : ℕ) (remaining_tables : ℕ) (people_per_table : ℕ) : 
  initial_customers = 21 → 
  remaining_tables = 3 → 
  people_per_table = 3 → 
  initial_customers - (remaining_tables * people_per_table) = 12 := by
sorry

end customers_left_l2486_248625


namespace zero_is_root_of_polynomial_l2486_248627

theorem zero_is_root_of_polynomial : ∃ (x : ℝ), 12 * x^4 + 38 * x^3 - 51 * x^2 + 40 * x = 0 := by
  sorry

end zero_is_root_of_polynomial_l2486_248627


namespace vector_operation_l2486_248615

/-- Given two 2D vectors a and b, prove that 3a - b equals (4, 2) -/
theorem vector_operation (a b : Fin 2 → ℝ) 
  (ha : a = ![1, 1]) 
  (hb : b = ![-1, 1]) : 
  (3 • a) - b = ![4, 2] := by sorry

end vector_operation_l2486_248615


namespace problem_solution_l2486_248680

/-- The problem setup and proof statements -/
theorem problem_solution :
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (2, -2)
  let C : ℝ × ℝ := (4, 1)
  let D : ℝ × ℝ := (5, -4)
  let a : ℝ × ℝ := (1, -5)
  let b : ℝ × ℝ := (2, 3)
  let k : ℝ := -1/3
  -- Part 1
  (B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2) ∧
  -- Part 2
  ∃ (t : ℝ), t ≠ 0 ∧ (k * a.1 - b.1, k * a.2 - b.2) = (t * (a.1 + 3 * b.1), t * (a.2 + 3 * b.2)) :=
by sorry

end problem_solution_l2486_248680


namespace additive_multiplicative_inverse_sum_l2486_248684

theorem additive_multiplicative_inverse_sum (a b : ℝ) : 
  (a + a = 0) → (b * b = 1) → (a + b = 1 ∨ a + b = -1) := by sorry

end additive_multiplicative_inverse_sum_l2486_248684


namespace prob_two_consecutive_wins_l2486_248621

/-- The probability of player A winning exactly two consecutive games in a three-game series -/
theorem prob_two_consecutive_wins (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/4) (h2 : p2 = 1/3) (h3 : p3 = 1/3) : 
  p1 * p2 * (1 - p3) + (1 - p1) * p2 * p3 = 5/36 := by
  sorry

end prob_two_consecutive_wins_l2486_248621


namespace expression_evaluation_l2486_248620

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 2
  (1/2 * x - 2 * (x - 1/3 * y^2) + (-3/2 * x + 1/3 * y^2)) = 10 := by
  sorry

end expression_evaluation_l2486_248620


namespace dog_to_rabbit_age_ratio_l2486_248653

/- Define the ages of the animals -/
def cat_age : ℕ := 8
def dog_age : ℕ := 12

/- Define the rabbit's age as half of the cat's age -/
def rabbit_age : ℕ := cat_age / 2

/- Define the ratio of the dog's age to the rabbit's age -/
def age_ratio : ℚ := dog_age / rabbit_age

/- Theorem statement -/
theorem dog_to_rabbit_age_ratio :
  age_ratio = 3 :=
sorry

end dog_to_rabbit_age_ratio_l2486_248653


namespace distinct_role_selection_l2486_248616

theorem distinct_role_selection (n : ℕ) (k : ℕ) : 
  n ≥ k → (n * (n - 1) * (n - 2) = (n.factorial) / ((n - k).factorial)) → 
  (8 * 7 * 6 = 336) :=
by sorry

end distinct_role_selection_l2486_248616


namespace square_root_decimal_shift_l2486_248602

theorem square_root_decimal_shift (x : ℝ) (hx : x > 0) :
  ∃ y : ℝ, y > 0 ∧ y^2 = x ∧ (100 * x).sqrt = 10 * y :=
by sorry

end square_root_decimal_shift_l2486_248602


namespace triangle_isosceles_or_right_angled_l2486_248614

/-- A triangle with sides a, b, and c is either isosceles or right-angled if (a - b) * (a² + b² - c²) = 0 --/
theorem triangle_isosceles_or_right_angled (a b c : ℝ) (h : (a - b) * (a^2 + b^2 - c^2) = 0) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
sorry

end triangle_isosceles_or_right_angled_l2486_248614


namespace correct_international_letters_l2486_248686

/-- The number of international letters in a mailing scenario. -/
def num_international_letters : ℕ :=
  let total_letters : ℕ := 4
  let standard_postage : ℚ := 108 / 100  -- $1.08
  let international_charge : ℚ := 14 / 100  -- $0.14
  let total_cost : ℚ := 460 / 100  -- $4.60
  2

/-- Proof that the number of international letters is correct. -/
theorem correct_international_letters : 
  let total_letters : ℕ := 4
  let standard_postage : ℚ := 108 / 100  -- $1.08
  let international_charge : ℚ := 14 / 100  -- $0.14
  let total_cost : ℚ := 460 / 100  -- $4.60
  num_international_letters = 2 ∧
  (num_international_letters : ℚ) * (standard_postage + international_charge) + 
  (total_letters - num_international_letters : ℚ) * standard_postage = total_cost := by
  sorry

end correct_international_letters_l2486_248686


namespace total_marbles_eq_4_9r_l2486_248631

/-- The total number of marbles in a bag given the number of red marbles -/
def total_marbles (r : ℝ) : ℝ :=
  let blue := 1.3 * r
  let green := 2 * blue
  r + blue + green

/-- Theorem stating that the total number of marbles is 4.9 times the number of red marbles -/
theorem total_marbles_eq_4_9r (r : ℝ) : total_marbles r = 4.9 * r := by
  sorry

end total_marbles_eq_4_9r_l2486_248631


namespace smallest_bases_sum_is_correct_l2486_248623

/-- Represents a number in a given base -/
def representationInBase (n : ℕ) (base : ℕ) : ℕ := 
  (n / base) * base + (n % base)

/-- The smallest possible sum of bases c and d where 83 in base c equals 38 in base d -/
def smallestBasesSum : ℕ := 27

theorem smallest_bases_sum_is_correct :
  ∀ c d : ℕ, c ≥ 2 → d ≥ 2 →
  representationInBase 83 c = representationInBase 38 d →
  c + d ≥ smallestBasesSum :=
sorry

end smallest_bases_sum_is_correct_l2486_248623


namespace percent_boys_in_class_l2486_248650

theorem percent_boys_in_class (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 49)
  (h2 : boys_ratio = 3)
  (h3 : girls_ratio = 4) :
  (boys_ratio * total_students : ℚ) / ((boys_ratio + girls_ratio) * total_students) * 100 = 42.86 := by
  sorry

end percent_boys_in_class_l2486_248650


namespace vertex_landing_probability_l2486_248694

/-- Square vertices -/
def square_vertices : List (Int × Int) := [(2, 2), (-2, 2), (-2, -2), (2, -2)]

/-- All boundary points of the square -/
def boundary_points : List (Int × Int) := [
  (2, 2), (-2, 2), (-2, -2), (2, -2),  -- vertices
  (1, 2), (0, 2), (-1, 2),             -- top edge
  (1, -2), (0, -2), (-1, -2),          -- bottom edge
  (2, 1), (2, 0), (2, -1),             -- right edge
  (-2, 1), (-2, 0), (-2, -1)           -- left edge
]

/-- Neighboring points function -/
def neighbors (x y : Int) : List (Int × Int) := [
  (x, y+1), (x+1, y+1), (x+1, y),
  (x+1, y-1), (x, y-1), (x-1, y-1),
  (x-1, y), (x-1, y+1)
]

/-- Theorem: Probability of landing on a vertex is 1/4 -/
theorem vertex_landing_probability :
  let start := (0, 0)
  let p_vertex := (square_vertices.length : ℚ) / (boundary_points.length : ℚ)
  p_vertex = 1/4 := by sorry

end vertex_landing_probability_l2486_248694


namespace fraction_equals_93_l2486_248662

theorem fraction_equals_93 : (3025 - 2880)^2 / 225 = 93 := by
  sorry

end fraction_equals_93_l2486_248662


namespace largest_integer_in_interval_l2486_248677

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), (2 : ℚ) / 7 < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < (3 : ℚ) / 4 ∧ 
  ∀ (y : ℤ), ((2 : ℚ) / 7 < (y : ℚ) / 6 ∧ (y : ℚ) / 6 < (3 : ℚ) / 4) → y ≤ x :=
by
  -- The proof goes here
  sorry

end largest_integer_in_interval_l2486_248677


namespace inverse_variation_proof_l2486_248608

/-- Given that x² varies inversely with y⁴, prove that x² = 4 when y = 4, given x = 8 when y = 2 -/
theorem inverse_variation_proof (x y : ℝ) (h1 : ∃ k : ℝ, ∀ x y, x^2 * y^4 = k) 
  (h2 : ∃ x₀ y₀ : ℝ, x₀ = 8 ∧ y₀ = 2 ∧ x₀^2 * y₀^4 = k) : 
  ∃ x₁ : ℝ, x₁^2 = 4 ∧ x₁^2 * 4^4 = k :=
sorry

end inverse_variation_proof_l2486_248608


namespace smallest_two_base_representation_l2486_248695

/-- Represents a number in a given base with two identical digits --/
def twoDigitNumber (digit : Nat) (base : Nat) : Nat :=
  digit * base + digit

/-- Checks if a number is valid in a given base --/
def isValidInBase (n : Nat) (base : Nat) : Prop :=
  n < base

theorem smallest_two_base_representation : 
  ∀ n : Nat, n < 24 → 
  ¬(∃ (a b : Nat), 
    isValidInBase a 5 ∧ 
    isValidInBase b 7 ∧ 
    n = twoDigitNumber a 5 ∧ 
    n = twoDigitNumber b 7) ∧
  (∃ (a b : Nat),
    isValidInBase a 5 ∧
    isValidInBase b 7 ∧
    24 = twoDigitNumber a 5 ∧
    24 = twoDigitNumber b 7) :=
by sorry

#check smallest_two_base_representation

end smallest_two_base_representation_l2486_248695


namespace perpendicular_line_through_point_l2486_248665

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 4 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  (perpendicular_line point_A.1 point_A.2) ∧
  (∀ x y : ℝ, perpendicular_line x y → given_line x y →
    (y - point_A.2) = 2 * (x - point_A.1)) :=
sorry

end perpendicular_line_through_point_l2486_248665


namespace circle_product_theorem_l2486_248647

/-- A circular permutation of five elements -/
def CircularPerm (α : Type) := Fin 5 → α

/-- The condition for the first part of the problem -/
def FirstCondition (a b c d e : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧
  a + b + c + d + e = 1 ∧
  ∀ π : CircularPerm ℝ, π 0 = a ∧ π 1 = b ∧ π 2 = c ∧ π 3 = d ∧ π 4 = e →
    ∃ i : Fin 5, π i * π ((i + 1) % 5) ≥ 1/9

/-- The condition for the second part of the problem -/
def SecondCondition (a b c d e : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧
  a + b + c + d + e = 1

/-- The theorem statement combining both parts of the problem -/
theorem circle_product_theorem :
  (∃ a b c d e : ℝ, FirstCondition a b c d e) ∧
  (∀ a b c d e : ℝ, SecondCondition a b c d e →
    ∃ π : CircularPerm ℝ, π 0 = a ∧ π 1 = b ∧ π 2 = c ∧ π 3 = d ∧ π 4 = e ∧
      ∀ i : Fin 5, π i * π ((i + 1) % 5) ≤ 1/9) :=
by sorry

end circle_product_theorem_l2486_248647


namespace complex_modulus_example_l2486_248628

theorem complex_modulus_example : Complex.abs (-5 - (8/3)*Complex.I) = 17/3 := by
  sorry

end complex_modulus_example_l2486_248628


namespace absolute_value_inequality_l2486_248617

theorem absolute_value_inequality (m : ℝ) : 
  (∀ x : ℝ, x + |x - 1| > m) → m < 1 := by
  sorry

end absolute_value_inequality_l2486_248617


namespace teresa_age_at_birth_l2486_248674

/-- Calculates Teresa's age when Michiko was born given current ages and Morio's age at Michiko's birth -/
def teresaAgeAtBirth (teresaCurrentAge marioCurrentAge marioAgeAtBirth : ℕ) : ℕ :=
  marioAgeAtBirth - (marioCurrentAge - teresaCurrentAge)

theorem teresa_age_at_birth :
  teresaAgeAtBirth 59 71 38 = 26 := by
  sorry

end teresa_age_at_birth_l2486_248674


namespace no_unique_p_for_expected_value_l2486_248679

theorem no_unique_p_for_expected_value :
  ¬ ∃! p₀ : ℝ, 0 < p₀ ∧ p₀ < 1 ∧ 6 * p₀^2 - 5 * p₀^3 = 1.5 := by
  sorry

end no_unique_p_for_expected_value_l2486_248679


namespace symmetric_increasing_function_property_l2486_248689

/-- A function that is increasing on (-∞, 2) and its graph shifted by 2 is symmetric about x=0 -/
def symmetric_increasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y ∧ y < 2 → f x < f y) ∧
  (∀ x, f (x + 2) = f (2 - x))

/-- If f is a symmetric increasing function, then f(0) < f(3) -/
theorem symmetric_increasing_function_property (f : ℝ → ℝ) 
  (h : symmetric_increasing_function f) : f 0 < f 3 := by
  sorry

end symmetric_increasing_function_property_l2486_248689


namespace min_value_expression_l2486_248658

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a*b*c) ≥ 216 := by
  sorry

end min_value_expression_l2486_248658


namespace mary_jenny_red_marbles_equal_l2486_248629

/-- Represents the number of marbles collected by each person -/
structure MarbleCollection where
  red : ℕ
  blue : ℕ

/-- Given information about marble collections -/
def problem_setup (mary anie jenny : MarbleCollection) : Prop :=
  mary.blue = anie.blue / 2 ∧
  anie.red = mary.red + 20 ∧
  anie.blue = 2 * jenny.blue ∧
  jenny.red = 30 ∧
  jenny.blue = 25

/-- Theorem stating that Mary and Jenny collected the same number of red marbles -/
theorem mary_jenny_red_marbles_equal 
  (mary anie jenny : MarbleCollection) 
  (h : problem_setup mary anie jenny) : 
  mary.red = jenny.red := by
  sorry

end mary_jenny_red_marbles_equal_l2486_248629


namespace students_passed_at_least_one_subject_l2486_248676

theorem students_passed_at_least_one_subject 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 32) 
  (h2 : failed_english = 56) 
  (h3 : failed_both = 12) : 
  100 - (failed_hindi + failed_english - failed_both) = 24 := by
  sorry

end students_passed_at_least_one_subject_l2486_248676


namespace min_value_theorem_l2486_248661

theorem min_value_theorem (m : ℝ) (a b : ℝ) :
  0 < m → m < 1 →
  ({x : ℝ | x^2 - 2*x + 1 - m^2 < 0} = {x : ℝ | a < x ∧ x < b}) →
  (∀ x : ℝ, x^2 - 2*x + 1 - m^2 < 0 ↔ a < x ∧ x < b) →
  (∀ x : ℝ, 1/(8*a + 2*b) - 1/(3*a - 3*b) ≥ 2/5) ∧
  (∃ x : ℝ, 1/(8*a + 2*b) - 1/(3*a - 3*b) = 2/5) :=
by sorry

end min_value_theorem_l2486_248661


namespace revenue_decrease_l2486_248622

def previous_revenue : ℝ := 69.0
def decrease_percentage : ℝ := 30.434782608695656

theorem revenue_decrease (previous_revenue : ℝ) (decrease_percentage : ℝ) :
  previous_revenue * (1 - decrease_percentage / 100) = 48.0 := by
  sorry

end revenue_decrease_l2486_248622


namespace fifteenth_student_age_l2486_248630

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_size = 4)
  (h4 : avg_age_group1 = 14)
  (h5 : group2_size = 9)
  (h6 : avg_age_group2 = 16)
  : ℝ := by
  sorry

#check fifteenth_student_age

end fifteenth_student_age_l2486_248630


namespace choose_one_from_each_set_l2486_248663

theorem choose_one_from_each_set : 
  ∀ (novels textbooks : ℕ), 
  novels = 5 → 
  textbooks = 6 → 
  novels * textbooks = 30 := by
sorry

end choose_one_from_each_set_l2486_248663


namespace min_value_expression_l2486_248657

theorem min_value_expression (a : ℝ) (h : a > 0) :
  (a - 1) * (4 * a - 1) / a ≥ -1 ∧
  ∃ a₀ > 0, (a₀ - 1) * (4 * a₀ - 1) / a₀ = -1 :=
sorry

end min_value_expression_l2486_248657


namespace geometric_sequence_product_l2486_248678

/-- Given a geometric sequence {aₙ}, prove that if a₃ · a₄ = 5, then a₁ · a₂ · a₅ · a₆ = 5 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) 
  (h_prod : a 3 * a 4 = 5) : a 1 * a 2 * a 5 * a 6 = 5 :=
by sorry

end geometric_sequence_product_l2486_248678


namespace M_remainder_1000_l2486_248612

/-- The greatest integer multiple of 9 with no two digits being the same -/
def M : ℕ :=
  sorry

/-- M has no repeated digits -/
axiom M_distinct_digits : ∀ d₁ d₂, d₁ ≠ d₂ → (M / 10^d₁ % 10) ≠ (M / 10^d₂ % 10)

/-- M is divisible by 9 -/
axiom M_div_by_9 : M % 9 = 0

/-- M is the greatest such number -/
axiom M_greatest : ∀ n : ℕ, n % 9 = 0 → (∀ d₁ d₂, d₁ ≠ d₂ → (n / 10^d₁ % 10) ≠ (n / 10^d₂ % 10)) → n ≤ M

theorem M_remainder_1000 : M % 1000 = 810 := by
  sorry

end M_remainder_1000_l2486_248612


namespace bob_remaining_corn_l2486_248642

/-- Represents the amount of corn in bushels and ears -/
structure CornAmount where
  bushels : ℚ
  ears : ℕ

/-- Calculates the remaining corn after giving some away -/
def remaining_corn (initial : CornAmount) (given_away : List CornAmount) : ℕ :=
  sorry

/-- Theorem stating that Bob has 357 ears of corn left -/
theorem bob_remaining_corn :
  let initial := CornAmount.mk 50 0
  let given_away := [
    CornAmount.mk 8 0,    -- Terry
    CornAmount.mk 3 0,    -- Jerry
    CornAmount.mk 12 0,   -- Linda
    CornAmount.mk 0 21    -- Stacy
  ]
  let ears_per_bushel := 14
  remaining_corn initial given_away = 357 := by
  sorry

end bob_remaining_corn_l2486_248642


namespace salary_increase_percentage_l2486_248699

theorem salary_increase_percentage (S : ℝ) (x : ℝ) 
  (h1 : S + 0.15 * S = 575) 
  (h2 : S + x * S = 600) : 
  x = 0.2 := by
sorry

end salary_increase_percentage_l2486_248699


namespace partner_b_profit_share_l2486_248601

/-- Calculates the share of profit for partner B given the investment ratios and total profit -/
theorem partner_b_profit_share 
  (invest_a invest_b invest_c : ℚ) 
  (total_profit : ℚ)
  (h1 : invest_a = 3 * invest_b)
  (h2 : invest_b = (2/3) * invest_c)
  (h3 : total_profit = 5500) :
  (invest_b / (invest_a + invest_b + invest_c)) * total_profit = 1000 := by
  sorry

end partner_b_profit_share_l2486_248601


namespace quadratic_roots_max_value_l2486_248600

theorem quadratic_roots_max_value (a b u v : ℝ) : 
  (∀ x, x^2 - a*x + b = 0 ↔ x = u ∨ x = v) →
  (u + v = u^2 + v^2) →
  (u + v = u^4 + v^4) →
  (u + v = u^18 + v^18) →
  (∃ (M : ℝ), ∀ (a' b' u' v' : ℝ), 
    (∀ x, x^2 - a'*x + b' = 0 ↔ x = u' ∨ x = v') →
    (u' + v' = u'^2 + v'^2) →
    (u' + v' = u'^4 + v'^4) →
    (u' + v' = u'^18 + v'^18) →
    1/u'^20 + 1/v'^20 ≤ M) →
  1/u^20 + 1/v^20 = 2 :=
sorry

end quadratic_roots_max_value_l2486_248600


namespace penguin_colony_ratio_l2486_248668

theorem penguin_colony_ratio :
  ∀ (initial_penguins end_first_year_penguins current_penguins : ℕ),
  end_first_year_penguins = 3 * initial_penguins →
  current_penguins = 3 * end_first_year_penguins + 129 →
  current_penguins = 1077 →
  end_first_year_penguins / initial_penguins = 3 :=
by
  sorry

end penguin_colony_ratio_l2486_248668


namespace coin_distribution_l2486_248659

def sum_of_integers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem coin_distribution (n : ℕ) (h : n = 20) :
  ∃ k : ℕ, sum_of_integers n = 3 * k ∧ ¬∃ m : ℕ, sum_of_integers n + 100 = 3 * m := by
  sorry

end coin_distribution_l2486_248659


namespace least_intersection_size_l2486_248690

theorem least_intersection_size (total students_with_glasses students_with_pets : ℕ) 
  (h_total : total = 35)
  (h_glasses : students_with_glasses = 18)
  (h_pets : students_with_pets = 25) :
  (students_with_glasses + students_with_pets - total : ℤ) = 8 := by
  sorry

end least_intersection_size_l2486_248690


namespace homework_percentage_l2486_248633

theorem homework_percentage (total_angle : ℝ) (less_than_one_hour_angle : ℝ) :
  total_angle = 360 →
  less_than_one_hour_angle = 90 →
  (1 - less_than_one_hour_angle / total_angle) * 100 = 75 := by
  sorry

end homework_percentage_l2486_248633


namespace railway_length_scientific_notation_l2486_248696

theorem railway_length_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 95500 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 9.55 ∧ n = 4 := by
  sorry

end railway_length_scientific_notation_l2486_248696


namespace cosine_inequality_solution_l2486_248644

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 (Real.pi / 2)) → 
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), Real.cos (x + y) ≥ Real.cos x - Real.cos y) → 
  y = 0 :=
sorry

end cosine_inequality_solution_l2486_248644


namespace train_speed_l2486_248667

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : bridge_length = 300)
  (h3 : crossing_time = 24) :
  (train_length + bridge_length) / crossing_time = 400 / 24 := by
sorry

end train_speed_l2486_248667


namespace faster_walking_speed_l2486_248651

/-- Given a person who walked 50 km at 10 km/hr, if they had walked at a faster speed
    that would allow them to cover an additional 20 km in the same time,
    prove that the faster speed would be 14 km/hr. -/
theorem faster_walking_speed (actual_distance : ℝ) (actual_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 50)
  (h2 : actual_speed = 10)
  (h3 : additional_distance = 20) :
  let time := actual_distance / actual_speed
  let total_distance := actual_distance + additional_distance
  let faster_speed := total_distance / time
  faster_speed = 14 := by
  sorry

end faster_walking_speed_l2486_248651


namespace inequality_proof_l2486_248641

theorem inequality_proof (a b : ℝ) : (a^4 + a^2*b^2 + b^4) / 3 ≥ (a^3*b + b^3*a) / 2 := by
  sorry

end inequality_proof_l2486_248641


namespace egypt_traditional_growth_l2486_248618

-- Define the set of countries
inductive Country
| UnitedStates
| Japan
| France
| Egypt

-- Define the development status of a country
inductive DevelopmentStatus
| Developed
| Developing

-- Define the population growth pattern
inductive PopulationGrowthPattern
| Modern
| Traditional

-- Function to determine the development status of a country
def developmentStatus (c : Country) : DevelopmentStatus :=
  match c with
  | Country.Egypt => DevelopmentStatus.Developing
  | _ => DevelopmentStatus.Developed

-- Function to determine the population growth pattern based on development status
def growthPattern (s : DevelopmentStatus) : PopulationGrowthPattern :=
  match s with
  | DevelopmentStatus.Developed => PopulationGrowthPattern.Modern
  | DevelopmentStatus.Developing => PopulationGrowthPattern.Traditional

-- Theorem: Egypt is the only country with a traditional population growth pattern
theorem egypt_traditional_growth : 
  ∀ c : Country, 
    growthPattern (developmentStatus c) = PopulationGrowthPattern.Traditional ↔ 
    c = Country.Egypt :=
  sorry


end egypt_traditional_growth_l2486_248618


namespace distance_between_bars_l2486_248672

/-- The distance between two bars given the walking times and speeds of two people --/
theorem distance_between_bars 
  (pierrot_extra_distance : ℝ) 
  (pierrot_time_after : ℝ) 
  (jeannot_time_after : ℝ) 
  (pierrot_speed_halved : ℝ → ℝ) 
  (jeannot_speed_halved : ℝ → ℝ) :
  ∃ (d : ℝ),
    pierrot_extra_distance = 200 ∧
    pierrot_time_after = 8 ∧
    jeannot_time_after = 18 ∧
    (∀ x, pierrot_speed_halved x = x / 2) ∧
    (∀ x, jeannot_speed_halved x = x / 2) ∧
    d > 0 ∧
    (d - pierrot_extra_distance) / (pierrot_speed_halved (d - pierrot_extra_distance) / pierrot_time_after) = 
      d / (jeannot_speed_halved d / jeannot_time_after) ∧
    2 * d - pierrot_extra_distance = 1000 :=
by sorry

end distance_between_bars_l2486_248672


namespace xinyu_taxi_fare_10km_l2486_248660

/-- Calculates the taxi fare in Xinyu city -/
def taxi_fare (distance : ℝ) : ℝ :=
  let base_fare := 5
  let mid_rate := 1.6
  let long_rate := 2.4
  let mid_distance := 6
  let long_distance := 2
  base_fare + mid_rate * mid_distance + long_rate * long_distance

/-- The total taxi fare for a 10 km journey in Xinyu city is 19.4 yuan -/
theorem xinyu_taxi_fare_10km : taxi_fare 10 = 19.4 := by
  sorry

end xinyu_taxi_fare_10km_l2486_248660


namespace intersection_of_M_and_N_l2486_248669

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by sorry

end intersection_of_M_and_N_l2486_248669


namespace train_crossing_bridge_time_l2486_248626

/-- Proves the time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 165) 
  (h2 : train_speed_kmph = 72) 
  (h3 : bridge_length = 660) : 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 41.25 := by
  sorry

end train_crossing_bridge_time_l2486_248626


namespace sum_congruence_mod_9_l2486_248692

theorem sum_congruence_mod_9 : 
  (1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999) % 9 = 6 := by
  sorry

end sum_congruence_mod_9_l2486_248692


namespace candy_problem_l2486_248613

theorem candy_problem (x : ℝ) : 
  let day1_remainder := x / 2 - 3
  let day2_remainder := day1_remainder * 3/4 - 5
  let day3_remainder := day2_remainder * 4/5
  day3_remainder = 9 → x = 136 := by sorry

end candy_problem_l2486_248613


namespace pure_imaginary_quotient_l2486_248673

/-- Given a real number a and i as the imaginary unit, if (a-i)/(1+i) is a pure imaginary number, then a = 1 -/
theorem pure_imaginary_quotient (a : ℝ) : 
  (∃ (b : ℝ), (a - Complex.I) / (1 + Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end pure_imaginary_quotient_l2486_248673


namespace solution_set_equivalence_l2486_248670

open Set

/-- The solution set of the inequality -x^2 + ax + b ≥ 0 -/
def SolutionSet (a b : ℝ) : Set ℝ := {x | -x^2 + a*x + b ≥ 0}

/-- The theorem stating the equivalence of the solution sets -/
theorem solution_set_equivalence (a b : ℝ) :
  SolutionSet a b = Icc (-2) 3 →
  {x : ℝ | x^2 - 5*a*x + b > 0} = {x : ℝ | x < 2 ∨ x > 3} := by
  sorry

end solution_set_equivalence_l2486_248670


namespace slope_range_l2486_248607

-- Define the points
def A₁ : ℝ × ℝ := (-2, 0)
def A₂ : ℝ × ℝ := (2, 0)
def B₁ (x : ℝ) : ℝ × ℝ := (x, 2)
def B₂ (x : ℝ) : ℝ × ℝ := (x, -2)
def P (x y : ℝ) : ℝ × ℝ := (x, y)
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the dot product
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Define the equation of the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the condition for the line passing through B and intersecting the ellipse
def intersects_ellipse (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    on_ellipse x₁ y₁ ∧
    on_ellipse x₂ y₂ ∧
    y₁ = k * x₁ + 2 ∧
    y₂ = k * x₂ + 2 ∧
    x₁ ≠ x₂

-- Define the condition for the ratio of triangle areas
def area_ratio_condition (x₁ x₂ : ℝ) : Prop :=
  1/2 < |x₁| / |x₂| ∧ |x₁| / |x₂| < 1

-- Main theorem
theorem slope_range :
  ∀ k : ℝ,
    intersects_ellipse k ∧
    (∃ x₁ x₂ : ℝ, area_ratio_condition x₁ x₂) ↔
    (k > Real.sqrt 2 / 2 ∧ k < 3 * Real.sqrt 14 / 14) ∨
    (k < -Real.sqrt 2 / 2 ∧ k > -3 * Real.sqrt 14 / 14) :=
sorry

end slope_range_l2486_248607


namespace baseball_glove_price_l2486_248654

theorem baseball_glove_price :
  let cards_price : ℝ := 25
  let bat_price : ℝ := 10
  let cleats_price : ℝ := 10
  let total_sales : ℝ := 79
  let discount_rate : ℝ := 0.2
  let other_items_total : ℝ := cards_price + bat_price + 2 * cleats_price
  let glove_discounted_price : ℝ := total_sales - other_items_total
  let glove_original_price : ℝ := glove_discounted_price / (1 - discount_rate)
  glove_original_price = 42.5 := by
sorry

end baseball_glove_price_l2486_248654


namespace wade_average_points_l2486_248649

/-- Represents a basketball team with Wade and his teammates -/
structure BasketballTeam where
  wade_avg : ℝ
  teammates_avg : ℝ
  total_points : ℝ
  num_games : ℝ

/-- Theorem stating Wade's average points per game -/
theorem wade_average_points (team : BasketballTeam)
  (h1 : team.teammates_avg = 40)
  (h2 : team.total_points = 300)
  (h3 : team.num_games = 5) :
  team.wade_avg = 20 := by
  sorry

#check wade_average_points

end wade_average_points_l2486_248649


namespace complex_number_quadrant_l2486_248698

theorem complex_number_quadrant (a : ℝ) : 
  (((2*a + 2*Complex.I) / (1 + Complex.I)).im ≠ 0 ∧ 
   ((2*a + 2*Complex.I) / (1 + Complex.I)).re = 0) → 
  (2*a < 0 ∧ 2 > 0) := by
  sorry

end complex_number_quadrant_l2486_248698


namespace gym_membership_cost_theorem_l2486_248632

/-- Calculates the total cost of a gym membership for a given number of years -/
def gymMembershipCost (monthlyFee : ℕ) (downPayment : ℕ) (years : ℕ) : ℕ :=
  monthlyFee * 12 * years + downPayment

/-- Theorem: The total cost for a 3-year gym membership with a $12 monthly fee and $50 down payment is $482 -/
theorem gym_membership_cost_theorem :
  gymMembershipCost 12 50 3 = 482 := by
  sorry

end gym_membership_cost_theorem_l2486_248632


namespace distance_on_line_l2486_248685

/-- The distance between two points (5, b) and (10, d) on the line y = 2x + 3 is 5√5. -/
theorem distance_on_line : ∀ b d : ℝ,
  b = 2 * 5 + 3 →
  d = 2 * 10 + 3 →
  Real.sqrt ((10 - 5)^2 + (d - b)^2) = 5 * Real.sqrt 5 := by
  sorry

end distance_on_line_l2486_248685


namespace mongolian_olympiad_inequality_l2486_248611

theorem mongolian_olympiad_inequality 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^4 + b^4 + c^4 + a^2/(b+c)^2 + b^2/(c+a)^2 + c^2/(a+b)^2 ≥ a*b + b*c + c*a :=
by sorry

end mongolian_olympiad_inequality_l2486_248611


namespace angle_BOK_formula_l2486_248675

/-- Represents a trihedral angle with vertex O and edges OA, OB, and OC -/
structure TrihedralAngle where
  α : ℝ  -- Angle BOC
  β : ℝ  -- Angle COA
  γ : ℝ  -- Angle AOB

/-- Represents a sphere inscribed in a trihedral angle -/
structure InscribedSphere (t : TrihedralAngle) where
  K : Point₃  -- Point where the sphere touches face BOC

/-- The angle BOK in a trihedral angle with an inscribed sphere -/
noncomputable def angleBOK (t : TrihedralAngle) (s : InscribedSphere t) : ℝ :=
  sorry

/-- Theorem stating that the angle BOK is equal to (α + γ - β) / 2 -/
theorem angle_BOK_formula (t : TrihedralAngle) (s : InscribedSphere t) :
  angleBOK t s = (t.α + t.γ - t.β) / 2 := by
  sorry

end angle_BOK_formula_l2486_248675


namespace condition_a_equals_one_sufficient_not_necessary_l2486_248619

-- Define the quadratic equation
def has_real_roots (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a = 2*x

-- Theorem statement
theorem condition_a_equals_one_sufficient_not_necessary :
  (has_real_roots 1) ∧ (∃ a : ℝ, a ≠ 1 ∧ has_real_roots a) :=
sorry

end condition_a_equals_one_sufficient_not_necessary_l2486_248619


namespace max_triangles_three_families_ten_lines_l2486_248656

/-- Represents a family of parallel lines -/
structure ParallelLineFamily :=
  (num_lines : ℕ)

/-- Represents the configuration of three families of parallel lines -/
structure ThreeParallelLineFamilies :=
  (family1 : ParallelLineFamily)
  (family2 : ParallelLineFamily)
  (family3 : ParallelLineFamily)

/-- Calculates the maximum number of triangles formed by three families of parallel lines -/
def max_triangles (config : ThreeParallelLineFamilies) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of triangles formed by three families of 10 parallel lines is 150 -/
theorem max_triangles_three_families_ten_lines :
  ∀ (config : ThreeParallelLineFamilies),
    config.family1.num_lines = 10 →
    config.family2.num_lines = 10 →
    config.family3.num_lines = 10 →
    max_triangles config = 150 :=
  sorry

end max_triangles_three_families_ten_lines_l2486_248656


namespace solve_equation_l2486_248609

theorem solve_equation (b : ℚ) (h : b + b / 4 = 5 / 2) : b = 2 := by
  sorry

end solve_equation_l2486_248609


namespace complex_equality_l2486_248681

theorem complex_equality (z : ℂ) : z = -1.5 - (1/6)*I →
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
  Complex.abs (z - 2) = Complex.abs (z - 3*I) := by
  sorry

end complex_equality_l2486_248681


namespace f_five_zeros_a_range_l2486_248683

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then
    Real.log x ^ 2 - floor (Real.log x) - 2
  else if x ≤ 0 then
    Real.exp (-x) - a * x - 1
  else
    0  -- This case is not specified in the original problem, so we set it to 0

-- State the theorem
theorem f_five_zeros_a_range (a : ℝ) :
  (∃ (s : Finset ℝ), s.card = 5 ∧ ∀ x ∈ s, f a x = 0) →
  a ∈ Set.Iic (-1 : ℝ) :=
sorry

end f_five_zeros_a_range_l2486_248683


namespace oil_leak_during_repairs_l2486_248634

theorem oil_leak_during_repairs 
  (total_leaked : ℕ) 
  (leaked_before_repairs : ℕ) 
  (h1 : total_leaked = 6206)
  (h2 : leaked_before_repairs = 2475) :
  total_leaked - leaked_before_repairs = 3731 := by
sorry

end oil_leak_during_repairs_l2486_248634


namespace female_worker_wage_l2486_248697

/-- Represents the daily wage of workers in rupees -/
structure DailyWage where
  male : ℕ
  female : ℕ
  child : ℕ

/-- Represents the number of workers in each category -/
structure WorkerCount where
  male : ℕ
  female : ℕ
  child : ℕ

def totalWorkers (w : WorkerCount) : ℕ :=
  w.male + w.female + w.child

def averageWage (w : WorkerCount) (d : DailyWage) : ℚ :=
  (w.male * d.male + w.female * d.female + w.child * d.child) / totalWorkers w

theorem female_worker_wage (w : WorkerCount) (d : DailyWage) :
  w.male = 20 →
  w.female = 15 →
  w.child = 5 →
  d.male = 35 →
  d.child = 8 →
  averageWage w d = 26 →
  d.female = 20 :=
by
  sorry


end female_worker_wage_l2486_248697


namespace tangent_line_circle_product_l2486_248643

/-- Given a line ax + by - 3 = 0 tangent to the circle x^2 + y^2 + 4x - 1 = 0 at point P(-1, 2),
    the product ab equals 2. -/
theorem tangent_line_circle_product (a b : ℝ) : 
  (∀ x y, a * x + b * y - 3 = 0 → x^2 + y^2 + 4*x - 1 = 0 → (x + 1)^2 + (y - 2)^2 ≠ 0) →
  a * (-1) + b * 2 - 3 = 0 →
  (-1)^2 + 2^2 + 4*(-1) - 1 = 0 →
  a * b = 2 := by
sorry


end tangent_line_circle_product_l2486_248643


namespace f_monotonic_increase_interval_l2486_248610

open Real

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 6) + cos (2 * x - π / 3)

theorem f_monotonic_increase_interval :
  ∀ k : ℤ, StrictMonoOn f (Set.Ioo (k * π - π / 3) (k * π + π / 6)) :=
by sorry

end f_monotonic_increase_interval_l2486_248610


namespace probability_sum_12_probability_sum_12_is_19_216_l2486_248648

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ 3

/-- The number of ways to roll a sum of 12 with three dice -/
def waysToRoll12 : ℕ := 19

/-- The probability of rolling a sum of 12 with three standard six-faced dice -/
theorem probability_sum_12 : ℚ :=
  waysToRoll12 / totalOutcomes

/-- Proof that the probability of rolling a sum of 12 with three standard six-faced dice is 19/216 -/
theorem probability_sum_12_is_19_216 : probability_sum_12 = 19 / 216 := by
  sorry

end probability_sum_12_probability_sum_12_is_19_216_l2486_248648


namespace emily_widget_difference_l2486_248640

-- Define the variables
variable (t : ℝ)
variable (w : ℝ)

-- Define the conditions
def monday_production := w * t
def tuesday_production := (w + 6) * (t - 3)

-- Define the relationship between w and t
axiom w_eq_2t : w = 2 * t

-- State the theorem
theorem emily_widget_difference :
  monday_production - tuesday_production = 18 := by
  sorry

end emily_widget_difference_l2486_248640


namespace monotonic_increasing_range_l2486_248636

theorem monotonic_increasing_range (a : Real) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x > 0, Monotone (fun x => a^x + (1+a)^x)) →
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) := by
  sorry

end monotonic_increasing_range_l2486_248636


namespace problem_statement_l2486_248624

theorem problem_statement (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 2)  -- absolute value of m is 2
  : (a + b) / m - m^2 + 2 * c * d = -2 := by
  sorry

end problem_statement_l2486_248624


namespace chord_length_implies_a_values_l2486_248635

theorem chord_length_implies_a_values (a : ℝ) : 
  (∃ (x y : ℝ), (x - a)^2 + y^2 = 4 ∧ x - y = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - a)^2 + y₁^2 = 4 ∧ x₁ - y₁ = 1 ∧
    (x₂ - a)^2 + y₂^2 = 4 ∧ x₂ - y₂ = 1 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →
  a = -1 ∨ a = 3 :=
by sorry

end chord_length_implies_a_values_l2486_248635


namespace final_painting_width_l2486_248645

theorem final_painting_width (total_paintings : ℕ) (total_area : ℕ) 
  (small_paintings : ℕ) (small_painting_side : ℕ) 
  (large_painting_length : ℕ) (large_painting_width : ℕ)
  (final_painting_height : ℕ) :
  total_paintings = 5 →
  total_area = 200 →
  small_paintings = 3 →
  small_painting_side = 5 →
  large_painting_length = 10 →
  large_painting_width = 8 →
  final_painting_height = 5 →
  (total_area - 
    (small_paintings * small_painting_side * small_painting_side + 
     large_painting_length * large_painting_width)) / final_painting_height = 9 := by
  sorry

#check final_painting_width

end final_painting_width_l2486_248645


namespace impossible_coin_probabilities_l2486_248671

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end impossible_coin_probabilities_l2486_248671


namespace largest_prime_divisor_of_factorial_sum_l2486_248638

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_prime_divisor_of_factorial_sum :
  ∃ (p : ℕ), is_prime p ∧ 
    p ∣ (factorial 10 + factorial 11) ∧ 
    ∀ (q : ℕ), is_prime q → q ∣ (factorial 10 + factorial 11) → q ≤ p :=
by sorry

end largest_prime_divisor_of_factorial_sum_l2486_248638


namespace total_working_days_l2486_248637

/-- Commute options for a person over a period of working days. -/
structure CommuteData where
  /-- Number of days driving car in the morning and riding bicycle in the afternoon -/
  car_morning_bike_afternoon : ℕ
  /-- Number of days riding bicycle in the morning and driving car in the afternoon -/
  bike_morning_car_afternoon : ℕ
  /-- Number of days using only bicycle both morning and afternoon -/
  bike_only : ℕ

/-- Theorem stating the total number of working days based on given commute data. -/
theorem total_working_days (data : CommuteData) : 
  data.car_morning_bike_afternoon + data.bike_morning_car_afternoon + data.bike_only = 23 :=
  by
  have morning_car : data.car_morning_bike_afternoon + data.bike_only = 12 := by sorry
  have afternoon_bike : data.bike_morning_car_afternoon + data.bike_only = 20 := by sorry
  have total_car : data.car_morning_bike_afternoon + data.bike_morning_car_afternoon = 14 := by sorry
  sorry

#check total_working_days

end total_working_days_l2486_248637


namespace projectile_max_height_l2486_248606

/-- The height function of the projectile -/
def f (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- Theorem stating that the maximum height of the projectile is 161 meters -/
theorem projectile_max_height :
  ∃ (t : ℝ), ∀ (s : ℝ), f s ≤ f t ∧ f t = 161 := by
  sorry

end projectile_max_height_l2486_248606


namespace identical_cuts_different_shapes_l2486_248682

/-- Represents a polygon --/
structure Polygon where
  area : ℝ

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Represents a triangle --/
structure Triangle where
  base : ℝ
  height : ℝ

/-- The theorem stating that it's possible to cut identical pieces from two identical polygons
    such that one remaining shape is a square and the other is a triangle --/
theorem identical_cuts_different_shapes (original : Polygon) :
  ∃ (cut_piece : ℝ) (square : Square) (triangle : Triangle),
    original.area = square.side ^ 2 + cut_piece ∧
    original.area = (1 / 2) * triangle.base * triangle.height + cut_piece ∧
    square.side ^ 2 = (1 / 2) * triangle.base * triangle.height :=
sorry

end identical_cuts_different_shapes_l2486_248682


namespace equation_solutions_range_l2486_248639

-- Define the equation
def equation (x a : ℝ) : Prop := |2^x - a| = 1

-- Define the condition of having two unequal real solutions
def has_two_unequal_solutions (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ equation x a ∧ equation y a

-- State the theorem
theorem equation_solutions_range :
  ∀ a : ℝ, has_two_unequal_solutions a ↔ a > 1 :=
by sorry

end equation_solutions_range_l2486_248639


namespace ascending_order_abc_l2486_248666

theorem ascending_order_abc (a b c : ℝ) : 
  a = (2 * Real.tan (70 * π / 180)) / (1 + Real.tan (70 * π / 180)^2) →
  b = Real.sqrt ((1 + Real.cos (109 * π / 180)) / 2) →
  c = (Real.sqrt 3 / 2) * Real.cos (81 * π / 180) + (1 / 2) * Real.sin (99 * π / 180) →
  b < c ∧ c < a := by
  sorry

end ascending_order_abc_l2486_248666


namespace cattle_milk_production_l2486_248688

/-- Given a herd of dairy cows, calculates the daily milk production per cow -/
def daily_milk_per_cow (num_cows : ℕ) (weekly_milk : ℕ) : ℚ :=
  (weekly_milk : ℚ) / 7 / num_cows

theorem cattle_milk_production :
  daily_milk_per_cow 52 364000 = 1000 := by
  sorry

end cattle_milk_production_l2486_248688


namespace b_minus_c_subscription_l2486_248664

/-- Represents the business subscription problem -/
structure BusinessSubscription where
  total_subscription : ℕ
  total_profit : ℕ
  c_profit : ℕ
  a_more_than_b : ℕ

/-- Theorem stating the difference between B's and C's subscriptions -/
theorem b_minus_c_subscription (bs : BusinessSubscription)
  (h1 : bs.total_subscription = 50000)
  (h2 : bs.total_profit = 35000)
  (h3 : bs.c_profit = 8400)
  (h4 : bs.a_more_than_b = 4000) :
  ∃ (b_sub c_sub : ℕ), b_sub - c_sub = 10000 ∧
    ∃ (a_sub : ℕ), a_sub + b_sub + c_sub = bs.total_subscription ∧
    a_sub = b_sub + bs.a_more_than_b ∧
    bs.c_profit * bs.total_subscription = c_sub * bs.total_profit :=
by sorry

end b_minus_c_subscription_l2486_248664


namespace star_example_l2486_248646

-- Define the star operation
def star (m n p q : ℚ) : ℚ := m * p * (n / q)

-- Theorem statement
theorem star_example : star (5/9) (6/4) = 135/2 := by sorry

end star_example_l2486_248646


namespace merchant_revenue_l2486_248691

/-- Calculates the total revenue for a set of vegetables --/
def total_revenue (quantities : List ℝ) (prices : List ℝ) (sold_percentages : List ℝ) : ℝ :=
  List.sum (List.zipWith3 (fun q p s => q * p * s) quantities prices sold_percentages)

/-- The total revenue generated by the merchant is $134.1 --/
theorem merchant_revenue : 
  let quantities : List ℝ := [20, 18, 12, 25, 10]
  let prices : List ℝ := [2, 3, 4, 1, 5]
  let sold_percentages : List ℝ := [0.6, 0.4, 0.75, 0.5, 0.8]
  total_revenue quantities prices sold_percentages = 134.1 := by
  sorry

end merchant_revenue_l2486_248691


namespace parallelogram_vector_subtraction_l2486_248604

-- Define a parallelogram ABCD
structure Parallelogram (V : Type*) [AddCommGroup V] :=
  (A B C D : V)
  (parallelogram_condition : A - B = D - C)

-- Define the theorem
theorem parallelogram_vector_subtraction 
  {V : Type*} [AddCommGroup V] (ABCD : Parallelogram V) :
  ABCD.A - ABCD.C - (ABCD.B - ABCD.C) = ABCD.D - ABCD.C :=
by sorry

end parallelogram_vector_subtraction_l2486_248604


namespace carpet_coverage_theorem_l2486_248603

/-- Represents the problem of covering a corridor with carpets --/
structure CarpetProblem where
  totalCarpetLength : ℕ
  numCarpets : ℕ
  corridorLength : ℕ

/-- Calculates the maximum number of uncovered sections in a carpet problem --/
def maxUncoveredSections (problem : CarpetProblem) : ℕ :=
  sorry

/-- Theorem stating that for the given problem, the maximum number of uncovered sections is 11 --/
theorem carpet_coverage_theorem (problem : CarpetProblem) 
  (h1 : problem.totalCarpetLength = 1000)
  (h2 : problem.numCarpets = 20)
  (h3 : problem.corridorLength = 100) :
  maxUncoveredSections problem = 11 :=
sorry

end carpet_coverage_theorem_l2486_248603
