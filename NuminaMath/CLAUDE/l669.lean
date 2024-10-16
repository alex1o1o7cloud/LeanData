import Mathlib

namespace NUMINAMATH_CALUDE_calculation_proof_equation_solution_l669_66907

-- Problem 1
theorem calculation_proof : -1^2024 + |(-3)| - (Real.pi + 1)^0 = 1 := by sorry

-- Problem 2
theorem equation_solution : 
  ∃ x : ℝ, (2 / (x + 2) = 4 / (x^2 - 4)) ∧ (x = 4) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_equation_solution_l669_66907


namespace NUMINAMATH_CALUDE_compound_interest_principal_l669_66958

/-- Proves that given a final amount of 8400 after 1 year of compound interest
    at 5% per annum (compounded annually), the initial principal amount is 8000. -/
theorem compound_interest_principal (final_amount : ℝ) (interest_rate : ℝ) (time : ℝ) :
  final_amount = 8400 ∧
  interest_rate = 0.05 ∧
  time = 1 →
  ∃ initial_principal : ℝ,
    initial_principal = 8000 ∧
    final_amount = initial_principal * (1 + interest_rate) ^ time :=
by
  sorry

#check compound_interest_principal

end NUMINAMATH_CALUDE_compound_interest_principal_l669_66958


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l669_66905

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 3 - 1
  1 - (x / (x + 1)) / (x / (x^2 - 1)) = 3 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l669_66905


namespace NUMINAMATH_CALUDE_expansion_of_binomial_l669_66987

theorem expansion_of_binomial (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (2*x - 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₁ + a₂ + a₃ + a₄ = -80 ∧ (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 625) := by
  sorry

end NUMINAMATH_CALUDE_expansion_of_binomial_l669_66987


namespace NUMINAMATH_CALUDE_rationalize_and_sum_l669_66908

theorem rationalize_and_sum (a b c d e : ℤ) : 
  (∃ (A B C D E : ℤ),
    (3 : ℝ) / (4 * Real.sqrt 7 + 5 * Real.sqrt 2) = 
      (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = a ∧ B = b ∧ C = c ∧ D = d ∧ E = e) →
  a + b + c + d + e = 68 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_sum_l669_66908


namespace NUMINAMATH_CALUDE_sets_and_range_l669_66993

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a - 2 < x ∧ x < a}
def B : Set ℝ := {x | 3 / (x - 1) ≥ 1}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ≤ 1 ∨ x > 4}

-- State the theorem
theorem sets_and_range (a : ℝ) : 
  A a ⊆ complement_B → (a ≤ 1 ∨ a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_sets_and_range_l669_66993


namespace NUMINAMATH_CALUDE_lemonade_mixture_problem_l669_66982

theorem lemonade_mixture_problem (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 100 →  -- Percentage of lemonade in first solution
  (0.6799999999999997 * (100 - x) + 0.32 * 55 = 72) →  -- Mixture equation
  x = 20 := by sorry

end NUMINAMATH_CALUDE_lemonade_mixture_problem_l669_66982


namespace NUMINAMATH_CALUDE_x_range_when_f_lg_x_gt_f_1_l669_66934

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → f y < f x

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem x_range_when_f_lg_x_gt_f_1 (heven : is_even f) (hdec : is_decreasing_on_nonneg f) :
  (∀ x, f (lg x) > f 1) → (∀ x, x > (1/10) ∧ x < 10) :=
sorry

end NUMINAMATH_CALUDE_x_range_when_f_lg_x_gt_f_1_l669_66934


namespace NUMINAMATH_CALUDE_part_one_part_two_l669_66966

-- Define the inequality function
def inequality (a x : ℝ) : Prop := (a * x - 1) * (x + 1) > 0

-- Part 1
theorem part_one : 
  (∀ x : ℝ, inequality (-2) x ↔ -1 < x ∧ x < -1/2) :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  (∀ x : ℝ, inequality a x ↔ 
    (a < -1 ∧ -1 < x ∧ x < 1/a) ∨
    (a = -1 ∧ False) ∨
    (-1 < a ∧ a < 0 ∧ 1/a < x ∧ x < -1) ∨
    (a = 0 ∧ x < -1) ∨
    (a > 0 ∧ (x < -1 ∨ x > 1/a))) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l669_66966


namespace NUMINAMATH_CALUDE_average_cat_weight_in_pounds_l669_66953

def cat_weights : List Real := [3.5, 7.2, 4.8, 6, 5.5, 9, 4, 7.5]
def kg_to_pounds : Real := 2.20462

theorem average_cat_weight_in_pounds :
  let total_weight_kg := cat_weights.sum
  let average_weight_kg := total_weight_kg / cat_weights.length
  let average_weight_pounds := average_weight_kg * kg_to_pounds
  average_weight_pounds = 13.0925 := by sorry

end NUMINAMATH_CALUDE_average_cat_weight_in_pounds_l669_66953


namespace NUMINAMATH_CALUDE_cube_root_negative_three_l669_66901

theorem cube_root_negative_three (x : ℝ) : x^(1/3) = (-3)^(1/3) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_negative_three_l669_66901


namespace NUMINAMATH_CALUDE_distribute_five_books_four_students_l669_66983

/-- The number of ways to distribute n different books to k students,
    with each student getting at least one book -/
def distribute (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n

/-- Theorem: There are 240 ways to distribute 5 different books to 4 students,
    with each student getting at least one book -/
theorem distribute_five_books_four_students :
  distribute 5 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_books_four_students_l669_66983


namespace NUMINAMATH_CALUDE_good_numbers_characterization_l669_66914

/-- A natural number is good if every natural divisor of n, when increased by 1, is a divisor of n+1 -/
def IsGood (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → (d + 1) ∣ (n + 1)

/-- Characterization of good numbers -/
theorem good_numbers_characterization (n : ℕ) :
  IsGood n ↔ n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_good_numbers_characterization_l669_66914


namespace NUMINAMATH_CALUDE_smallest_shift_l669_66996

-- Define a periodic function g with period 30
def g (x : ℝ) : ℝ := sorry

-- State the periodicity of g
axiom g_periodic (x : ℝ) : g (x + 30) = g x

-- Define the property we want to prove
def property (b : ℝ) : Prop :=
  ∀ x, g ((x - b) / 3) = g (x / 3)

-- State the theorem
theorem smallest_shift :
  (∃ b > 0, property b) ∧ 
  (∀ b > 0, property b → b ≥ 90) ∧
  property 90 := by sorry

end NUMINAMATH_CALUDE_smallest_shift_l669_66996


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l669_66945

theorem complex_on_imaginary_axis (a b : ℝ) : 
  (Complex.I * Complex.I = -1) →
  (∃ (y : ℝ), (a + Complex.I) / (b - 3 * Complex.I) = Complex.I * y) →
  a * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l669_66945


namespace NUMINAMATH_CALUDE_mode_is_97_l669_66976

/-- Represents a score in the stem-and-leaf plot -/
structure Score where
  stem : Nat
  leaf : Nat

/-- The list of all scores from the stem-and-leaf plot -/
def scores : List Score := [
  ⟨6, 5⟩, ⟨6, 5⟩,
  ⟨7, 1⟩, ⟨7, 3⟩, ⟨7, 3⟩, ⟨7, 6⟩,
  ⟨8, 0⟩, ⟨8, 0⟩, ⟨8, 4⟩, ⟨8, 4⟩, ⟨8, 8⟩, ⟨8, 8⟩, ⟨8, 8⟩,
  ⟨9, 2⟩, ⟨9, 2⟩, ⟨9, 5⟩, ⟨9, 7⟩, ⟨9, 7⟩, ⟨9, 7⟩, ⟨9, 7⟩,
  ⟨10, 1⟩, ⟨10, 1⟩, ⟨10, 1⟩, ⟨10, 4⟩, ⟨10, 6⟩,
  ⟨11, 0⟩, ⟨11, 0⟩, ⟨11, 0⟩
]

/-- Convert a Score to its numerical value -/
def scoreValue (s : Score) : Nat :=
  s.stem * 10 + s.leaf

/-- Count the occurrences of a value in the list of scores -/
def countOccurrences (value : Nat) : Nat :=
  (scores.filter (fun s => scoreValue s = value)).length

/-- The mode is the most frequent score -/
def isMode (value : Nat) : Prop :=
  ∀ (other : Nat), countOccurrences value ≥ countOccurrences other

/-- Theorem: The mode of the scores is 97 -/
theorem mode_is_97 : isMode 97 := by
  sorry

end NUMINAMATH_CALUDE_mode_is_97_l669_66976


namespace NUMINAMATH_CALUDE_percentage_relation_l669_66970

theorem percentage_relation (a b : ℝ) (h : a = 2 * b) : 4 * b = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l669_66970


namespace NUMINAMATH_CALUDE_number_1985_in_column_2_l669_66960

/-- The column number (1-5) in which a given odd positive integer appears when arranged in 5 columns -/
def column_number (n : ℕ) : ℕ :=
  let row := (n - 1) / 5 + 1
  let pos_in_row := (n - 1) % 5 + 1
  if row % 2 = 1 then
    pos_in_row
  else
    6 - pos_in_row

theorem number_1985_in_column_2 : column_number 1985 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_1985_in_column_2_l669_66960


namespace NUMINAMATH_CALUDE_absent_children_count_l669_66912

/-- Given a school with a total of 700 children, where each child was supposed to get 2 bananas,
    but due to absences, each present child got 4 bananas instead,
    prove that the number of absent children is 350. -/
theorem absent_children_count (total_children : ℕ) (bananas_per_child_original : ℕ) 
    (bananas_per_child_actual : ℕ) (absent_children : ℕ) : 
    total_children = 700 → 
    bananas_per_child_original = 2 →
    bananas_per_child_actual = 4 →
    absent_children = total_children - (total_children * bananas_per_child_original) / bananas_per_child_actual →
    absent_children = 350 := by
  sorry

end NUMINAMATH_CALUDE_absent_children_count_l669_66912


namespace NUMINAMATH_CALUDE_two_diamonds_balance_three_dots_l669_66929

-- Define the symbols
variable (triangle diamond dot : ℕ)

-- Define the balance relationships
axiom balance1 : 3 * triangle + diamond = 9 * dot
axiom balance2 : triangle = diamond + dot

-- Theorem to prove
theorem two_diamonds_balance_three_dots : 2 * diamond = 3 * dot := by
  sorry

end NUMINAMATH_CALUDE_two_diamonds_balance_three_dots_l669_66929


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l669_66938

/-- Theorem: The area of a square with a diagonal of 12 centimeters is 72 square centimeters. -/
theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) :
  diagonal = 12 →
  area = diagonal^2 / 2 →
  area = 72 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l669_66938


namespace NUMINAMATH_CALUDE_circle_problem_l669_66943

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a line is tangent to a circle -/
def Circle.tangentTo (c : Circle) (m a : ℝ) : Prop :=
  let d := |m * c.center.1 - c.center.2 + a| / Real.sqrt (m^2 + 1)
  d = c.radius

/-- The equation of the circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_problem :
  ∃ c : Circle,
    c.contains (2, -1) ∧
    c.tangentTo 1 1 ∧
    c.center.2 = -2 * c.center.1 ∧
    (∀ x y, c.equation x y ↔ ((x - 1)^2 + (y + 2)^2 = 2 ∨ (x - 9)^2 + (y + 18)^2 = 338)) :=
by sorry

end NUMINAMATH_CALUDE_circle_problem_l669_66943


namespace NUMINAMATH_CALUDE_total_cost_calculation_l669_66950

-- Define the prices and quantities
def shirt_price : ℝ := 15
def shirt_quantity : ℕ := 4
def pants_price : ℝ := 40
def pants_quantity : ℕ := 2
def suit_price : ℝ := 150
def suit_quantity : ℕ := 1
def sweater_price : ℝ := 30
def sweater_quantity : ℕ := 2
def tie_price : ℝ := 20
def tie_quantity : ℕ := 3
def shoes_price : ℝ := 80
def shoes_quantity : ℕ := 1

-- Define the discounts
def shirt_discount : ℝ := 0.2
def pants_discount : ℝ := 0.3
def tie_discount : ℝ := 0.5
def shoes_discount : ℝ := 0.25
def coupon_discount : ℝ := 0.1

-- Define reward points
def reward_points : ℕ := 500
def reward_point_value : ℝ := 0.05

-- Define sales tax
def sales_tax_rate : ℝ := 0.05

-- Define the theorem
theorem total_cost_calculation :
  let shirt_total := shirt_price * shirt_quantity * (1 - shirt_discount)
  let pants_total := pants_price * pants_quantity * (1 - pants_discount)
  let suit_total := suit_price * suit_quantity
  let sweater_total := sweater_price * sweater_quantity
  let tie_total := tie_price * tie_quantity - tie_price * tie_discount
  let shoes_total := shoes_price * shoes_quantity * (1 - shoes_discount)
  let subtotal := shirt_total + pants_total + suit_total + sweater_total + tie_total + shoes_total
  let after_coupon := subtotal * (1 - coupon_discount)
  let after_rewards := after_coupon - (reward_points * reward_point_value)
  let final_total := after_rewards * (1 + sales_tax_rate)
  final_total = 374.43 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l669_66950


namespace NUMINAMATH_CALUDE_prime_triples_divisibility_l669_66928

theorem prime_triples_divisibility (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧
  p ∣ q^r + 1 ∧ q ∣ r^p + 1 ∧ r ∣ p^q + 1 →
  (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
by sorry

end NUMINAMATH_CALUDE_prime_triples_divisibility_l669_66928


namespace NUMINAMATH_CALUDE_calculation_one_l669_66999

theorem calculation_one : (-2) + (-7) + 9 - (-12) = 12 := by sorry

end NUMINAMATH_CALUDE_calculation_one_l669_66999


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_range_l669_66985

/-- Given two predicates p and q on real numbers, where p is a sufficient but not necessary condition for q, 
    this theorem states that the lower bound of k in p is 2, and k can be any real number greater than 2. -/
theorem sufficient_not_necessary_condition_range (k : ℝ) : 
  (∀ x, x ≥ k → (2 - x) / (x + 1) < 0) ∧ 
  (∃ x, (2 - x) / (x + 1) < 0 ∧ x < k) → 
  k > 2 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_range_l669_66985


namespace NUMINAMATH_CALUDE_solution_difference_l669_66956

def is_solution (x : ℝ) : Prop :=
  (4 * x - 12) / (x^2 + 2*x - 15) = x + 2

theorem solution_difference (p q : ℝ) 
  (hp : is_solution p) 
  (hq : is_solution q) 
  (hdistinct : p ≠ q) 
  (horder : p > q) : 
  p - q = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l669_66956


namespace NUMINAMATH_CALUDE_prescription_duration_l669_66959

/-- The number of days a prescription lasts -/
def prescription_days : ℕ := 30

/-- The daily dose in pills -/
def daily_dose : ℕ := 2

/-- The number of pills remaining after 4/5 of the days -/
def remaining_pills : ℕ := 12

/-- Theorem stating that the prescription lasts for 30 days -/
theorem prescription_duration :
  prescription_days = 30 ∧
  remaining_pills = (1/5 : ℚ) * (prescription_days * daily_dose) :=
by sorry

end NUMINAMATH_CALUDE_prescription_duration_l669_66959


namespace NUMINAMATH_CALUDE_trigonometric_ratio_proof_trigonometric_expression_simplification_l669_66910

theorem trigonometric_ratio_proof (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := by
  sorry

theorem trigonometric_expression_simplification (α : Real) :
  (Real.sin (π/2 + α) * Real.cos (5*π/2 - α) * Real.tan (-π + α)) /
  (Real.tan (7*π - α) * Real.sin (π + α)) = Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_ratio_proof_trigonometric_expression_simplification_l669_66910


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l669_66975

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^5 + 2 = (x^2 - 2*x + 3) * q + (-4*x^2 - 3*x + 2) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l669_66975


namespace NUMINAMATH_CALUDE_cosine_amplitude_l669_66900

theorem cosine_amplitude (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (∀ x, a * Real.cos (b * x - c) ≤ 3) ∧
  (∃ x, a * Real.cos (b * x - c) = 3) ∧
  (∀ x, a * Real.cos (b * x - c) = a * Real.cos (b * (x + 2 * Real.pi) - c)) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l669_66900


namespace NUMINAMATH_CALUDE_complex_equation_solution_l669_66977

theorem complex_equation_solution (a b : ℝ) (z : ℂ) 
  (h1 : z = Complex.mk a b) 
  (h2 : Complex.I / z = Complex.mk 2 (-1)) : 
  a - b = -3/5 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l669_66977


namespace NUMINAMATH_CALUDE_picture_frame_length_l669_66936

/-- Given a rectangular picture frame with height 12 inches and perimeter 44 inches, 
    prove that its length is 10 inches. -/
theorem picture_frame_length (height : ℝ) (perimeter : ℝ) (length : ℝ) : 
  height = 12 → perimeter = 44 → perimeter = 2 * (length + height) → length = 10 := by
  sorry

end NUMINAMATH_CALUDE_picture_frame_length_l669_66936


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l669_66951

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Theorem statement
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l669_66951


namespace NUMINAMATH_CALUDE_mistaken_division_multiplication_l669_66986

/-- Given a number x and another number n, where x is mistakenly divided by n instead of being multiplied,
    and the percentage error in the result is 99%, prove that n = 10. -/
theorem mistaken_division_multiplication (x : ℝ) (n : ℝ) (h : x ≠ 0) :
  (x / n) / (x * n) = 1 / 100 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_division_multiplication_l669_66986


namespace NUMINAMATH_CALUDE_product_change_theorem_l669_66930

theorem product_change_theorem (k : ℝ) (x y z : ℝ) (h1 : x * y * z = k) :
  ∃ (p q : ℝ),
    1.805 * (1 - p / 100) * (1 + q / 100) = 1 ∧
    Real.log p - Real.cos q = 0 ∧
    x * 1.805 * y * (1 - p / 100) * z * (1 + q / 100) = k := by
  sorry

end NUMINAMATH_CALUDE_product_change_theorem_l669_66930


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l669_66940

/-- Calculates the gain percent from a purchase, repair, and sale of an item. -/
theorem gain_percent_calculation 
  (purchase_price : ℝ) 
  (repair_costs : ℝ) 
  (selling_price : ℝ) 
  (h1 : purchase_price = 800)
  (h2 : repair_costs = 200)
  (h3 : selling_price = 1400) : 
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 40 := by
  sorry

#check gain_percent_calculation

end NUMINAMATH_CALUDE_gain_percent_calculation_l669_66940


namespace NUMINAMATH_CALUDE_polynomial_calculation_l669_66952

/-- A polynomial of degree 4 with specific properties -/
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

/-- Theorem stating the result of the calculation -/
theorem polynomial_calculation (a b c d : ℝ) 
  (h1 : P a b c d 1 = 1993)
  (h2 : P a b c d 2 = 3986)
  (h3 : P a b c d 3 = 5979) :
  (1/4) * (P a b c d 11 + P a b c d (-7)) = 4693 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_calculation_l669_66952


namespace NUMINAMATH_CALUDE_tangent_line_at_2_monotonicity_intervals_l669_66906

/-- The function f(x) = 3x - x^3 -/
def f (x : ℝ) : ℝ := 3 * x - x^3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 - 3 * x^2

theorem tangent_line_at_2 :
  ∃ (m b : ℝ), m = -9 ∧ b = 16 ∧
  ∀ x, f x = m * (x - 2) + f 2 + b - f 2 :=
sorry

theorem monotonicity_intervals :
  (∀ x y, x < y ∧ y < -1 → f y < f x) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_monotonicity_intervals_l669_66906


namespace NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l669_66946

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_from_perpendicular_lines
  (m n : Line) (α β γ : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h_parallel : parallel m n)
  (h_perp_n : perpendicular n α)
  (h_perp_m : perpendicular m β) :
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l669_66946


namespace NUMINAMATH_CALUDE_aunt_gift_amount_l669_66932

theorem aunt_gift_amount (jade_initial : ℕ) (julia_initial : ℕ) (total_after : ℕ) 
    (h1 : jade_initial = 38)
    (h2 : julia_initial = jade_initial / 2)
    (h3 : total_after = 97)
    (h4 : ∃ (gift : ℕ), total_after = jade_initial + julia_initial + 2 * gift) :
  ∃ (gift : ℕ), gift = 20 ∧ total_after = jade_initial + julia_initial + 2 * gift :=
by sorry

end NUMINAMATH_CALUDE_aunt_gift_amount_l669_66932


namespace NUMINAMATH_CALUDE_probability_theorem_l669_66992

/-- Represents the number of students in each language class and their combinations --/
structure LanguageEnrollment where
  total : ℕ
  french : ℕ
  spanish : ℕ
  german : ℕ
  french_spanish : ℕ
  french_german : ℕ
  spanish_german : ℕ
  all_three : ℕ

/-- Calculates the probability of selecting at least one student from each language class --/
def probability_all_languages (e : LanguageEnrollment) : ℚ :=
  let total_combinations := (e.total.choose 3)
  let favorable_outcomes := 
    (e.french - e.french_spanish - e.french_german + e.all_three) * 
    (e.spanish - e.french_spanish - e.spanish_german + e.all_three) * 
    (e.german - e.french_german - e.spanish_german + e.all_three) +
    e.french_spanish * (e.german - e.french_german - e.spanish_german + e.all_three) +
    e.french_german * (e.spanish - e.french_spanish - e.spanish_german + e.all_three) +
    e.spanish_german * (e.french - e.french_spanish - e.french_german + e.all_three)
  favorable_outcomes / total_combinations

/-- The main theorem to prove --/
theorem probability_theorem (e : LanguageEnrollment) 
  (h1 : e.total = 40)
  (h2 : e.french = 26)
  (h3 : e.spanish = 29)
  (h4 : e.german = 12)
  (h5 : e.french_spanish = 9)
  (h6 : e.french_german = 9)
  (h7 : e.spanish_german = 9)
  (h8 : e.all_three = 2) :
  probability_all_languages e = 76 / 4940 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l669_66992


namespace NUMINAMATH_CALUDE_units_digit_of_fib_F_15_l669_66942

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- State that the units digit of Fibonacci numbers repeats every 60 terms
axiom fib_units_period (n : ℕ) : fib n % 10 = fib (n % 60) % 10

-- Define F_15
def F_15 : ℕ := fib 15

-- Theorem to prove
theorem units_digit_of_fib_F_15 : fib (fib 15) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fib_F_15_l669_66942


namespace NUMINAMATH_CALUDE_total_holiday_savings_l669_66902

/-- The total money saved for holiday spending by Victory and Sam -/
theorem total_holiday_savings (sam_savings : ℕ) (victory_savings : ℕ) : 
  sam_savings = 1200 → 
  victory_savings = sam_savings - 200 →
  sam_savings + victory_savings = 2200 := by
sorry

end NUMINAMATH_CALUDE_total_holiday_savings_l669_66902


namespace NUMINAMATH_CALUDE_inequality_proof_l669_66980

theorem inequality_proof (α : ℝ) (m : ℕ) (a b : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2) (h3 : m ≥ 1) (h4 : 0 < a) (h5 : 0 < b) : 
  a / (Real.cos α)^m + b / (Real.sin α)^m ≥ (a^(2/(m+2)) + b^(2/(m+2)))^((m+2)/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l669_66980


namespace NUMINAMATH_CALUDE_max_oranges_for_teacher_l669_66995

theorem max_oranges_for_teacher (n : ℕ) : 
  let k := 8
  let remainder := n % k
  remainder ≤ 7 ∧ ∃ m : ℕ, n = m * k + 7 :=
by sorry

end NUMINAMATH_CALUDE_max_oranges_for_teacher_l669_66995


namespace NUMINAMATH_CALUDE_ratio_sum_l669_66927

theorem ratio_sum (a b c d : ℝ) : 
  a / b = 2 / 3 ∧ 
  b / c = 3 / 4 ∧ 
  c / d = 4 / 5 ∧ 
  d = 672 → 
  a + b + c + d = 1881.6 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_l669_66927


namespace NUMINAMATH_CALUDE_ring_toss_losers_l669_66937

theorem ring_toss_losers (winners : ℕ) (ratio : ℚ) (h1 : winners = 28) (h2 : ratio = 4/1) : 
  (winners : ℚ) / ratio = 7 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_losers_l669_66937


namespace NUMINAMATH_CALUDE_lemniscate_orthogonal_trajectories_l669_66989

-- Define the lemniscate family
def lemniscate (a : ℝ) (ρ φ : ℝ) : Prop :=
  ρ^2 = a * Real.cos (2 * φ)

-- Define the orthogonal trajectory
def orthogonal_trajectory (C : ℝ) (ρ φ : ℝ) : Prop :=
  ρ^2 = C * Real.sin (2 * φ)

-- Theorem statement
theorem lemniscate_orthogonal_trajectories (a C : ℝ) (ρ φ : ℝ) :
  lemniscate a ρ φ → orthogonal_trajectory C ρ φ :=
by
  sorry

end NUMINAMATH_CALUDE_lemniscate_orthogonal_trajectories_l669_66989


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l669_66948

theorem division_multiplication_problem : 5 / (-1/5) * 5 = -125 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l669_66948


namespace NUMINAMATH_CALUDE_three_digit_number_operation_l669_66915

theorem three_digit_number_operation (a b c : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧ a = c + 3 →
  (4 * (100 * a + 10 * b + c) - (100 * c + 10 * b + a)) % 10 = 7 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_operation_l669_66915


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l669_66903

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The property that the range of a function is [0, +∞) -/
def HasNonnegativeRange (f : ℝ → ℝ) : Prop :=
  ∀ y, (∃ x, f x = y) → y ≥ 0

/-- The solution set of f(x) < c is an open interval of length 8 -/
def HasSolutionSetOfLength8 (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∃ m, ∀ x, f x < c ↔ m < x ∧ x < m + 8

theorem quadratic_function_theorem (a b : ℝ) :
  HasNonnegativeRange (QuadraticFunction a b) →
  HasSolutionSetOfLength8 (QuadraticFunction a b) 16 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l669_66903


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l669_66944

theorem cubic_sum_minus_product (a b c : ℝ) 
  (h1 : a + b + c = 10) 
  (h2 : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 100 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l669_66944


namespace NUMINAMATH_CALUDE_minor_axis_length_l669_66955

-- Define the ellipse C₁
def ellipse_C1 (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the hyperbola C₂
def hyperbola_C2 (x y : ℝ) : Prop :=
  x^2 - y^2 / 4 = 1

-- Define the condition that C₁ and C₂ share a common focus
def shared_focus (a b : ℝ) : Prop :=
  a^2 - b^2 = 5

-- Define the condition that C₁ trisects AB
def trisects_AB (a b : ℝ) : Prop :=
  4 * (b^2 + 5) = 9 * (4 * b^4 + 20 * b^2) / (b^2 + 4)

-- Main theorem
theorem minor_axis_length (a b : ℝ) :
  ellipse_C1 0 0 a b →
  hyperbola_C2 0 0 →
  shared_focus a b →
  trisects_AB a b →
  2 * b = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_minor_axis_length_l669_66955


namespace NUMINAMATH_CALUDE_arrangement_count_is_540_l669_66990

/-- The number of ways to arrange teachers and students into groups and locations -/
def arrangement_count : ℕ :=
  (Nat.choose 6 2) * (Nat.choose 4 2) * (Nat.choose 2 2) * (Nat.factorial 3)

/-- Theorem stating that the number of arrangements is 540 -/
theorem arrangement_count_is_540 : arrangement_count = 540 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_540_l669_66990


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l669_66974

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 2 → a^2 > 2*a) ∧ 
  (∃ a, a^2 > 2*a ∧ ¬(a > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l669_66974


namespace NUMINAMATH_CALUDE_number_line_position_l669_66939

theorem number_line_position : 
  ∀ (total_distance : ℝ) (num_steps : ℕ) (step_number : ℕ),
    total_distance = 32 →
    num_steps = 8 →
    step_number = 5 →
    (step_number : ℝ) * (total_distance / num_steps) = 20 :=
by sorry

end NUMINAMATH_CALUDE_number_line_position_l669_66939


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l669_66911

theorem max_value_of_expression (x : ℝ) :
  x^6 / (x^12 + 3*x^9 - 6*x^6 + 12*x^3 + 27) ≤ 1 / (6*Real.sqrt 3 + 6) :=
sorry

theorem max_value_achievable :
  ∃ x : ℝ, x^6 / (x^12 + 3*x^9 - 6*x^6 + 12*x^3 + 27) = 1 / (6*Real.sqrt 3 + 6) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l669_66911


namespace NUMINAMATH_CALUDE_no_solution_exists_l669_66965

theorem no_solution_exists : ¬∃ x : ℝ, x > 0 ∧ x * Real.sqrt (9 - x) + Real.sqrt (9 * x - x^3) ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l669_66965


namespace NUMINAMATH_CALUDE_remainder_divisibility_l669_66954

theorem remainder_divisibility (n : ℕ) (h : n % 10 = 7) : n % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l669_66954


namespace NUMINAMATH_CALUDE_curve_sum_invariant_under_translation_l669_66924

-- Define a type for points in a plane
variable (P : Type) [AddCommGroup P]

-- Define a type for convex curves in the plane
variable (Curve : Type) [AddCommGroup Curve]

-- Define a parallel translation operation
variable (T : P → P)

-- Define an operation to apply translation to a curve
variable (applyTranslation : (P → P) → Curve → Curve)

-- Define a sum operation for curves
variable (curveSum : Curve → Curve → Curve)

-- Define a congruence relation for curves
variable (congruent : Curve → Curve → Prop)

-- Statement of the theorem
theorem curve_sum_invariant_under_translation 
  (K₁ K₂ : Curve) :
  congruent (curveSum K₁ K₂) (curveSum (applyTranslation T K₁) (applyTranslation T K₂)) :=
sorry

end NUMINAMATH_CALUDE_curve_sum_invariant_under_translation_l669_66924


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l669_66920

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : x * y = 1) : 
  (5 * x + 3) - (2 * x * y - 5 * y) = 16 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l669_66920


namespace NUMINAMATH_CALUDE_class_size_difference_l669_66994

theorem class_size_difference (total_students : ℕ) (total_professors : ℕ) (class_sizes : List ℕ) :
  total_students = 200 →
  total_professors = 4 →
  class_sizes = [100, 50, 30, 20] →
  (class_sizes.sum = total_students) →
  let t := (class_sizes.sum : ℚ) / total_professors
  let s := (class_sizes.map (λ size => size * size)).sum / total_students
  t - s = -19 := by
  sorry

end NUMINAMATH_CALUDE_class_size_difference_l669_66994


namespace NUMINAMATH_CALUDE_star_calculation_l669_66919

/-- The star operation on rational numbers -/
def star (a b : ℚ) : ℚ := 2 * a - b + 1

/-- Theorem stating that 1 ☆ [3 ☆ (-2)] = -6 -/
theorem star_calculation : star 1 (star 3 (-2)) = -6 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l669_66919


namespace NUMINAMATH_CALUDE_complex_equal_parts_l669_66935

/-- Given a complex number z = (2+ai)/(1+2i) where a is real,
    if the real part of z equals its imaginary part, then a = -6 -/
theorem complex_equal_parts (a : ℝ) :
  let z : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)
  Complex.re z = Complex.im z → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_complex_equal_parts_l669_66935


namespace NUMINAMATH_CALUDE_book_sale_price_l669_66991

theorem book_sale_price (total_books : ℕ) (sold_books : ℕ) (unsold_books : ℕ) (total_amount : ℚ) : 
  sold_books = (2 : ℕ) * total_books / 3 →
  unsold_books = 30 →
  sold_books + unsold_books = total_books →
  total_amount = 255 →
  total_amount / sold_books = 17/4 := by
  sorry

#eval (17 : ℚ) / 4  -- This should evaluate to 4.25

end NUMINAMATH_CALUDE_book_sale_price_l669_66991


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l669_66972

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  a_5_eq_10 : a 5 = 10
  S_15_eq_240 : S 15 = 240

/-- The b sequence derived from the arithmetic sequence -/
def b (seq : ArithmeticSequence) (n : ℕ) : ℝ := seq.a (3^n)

/-- The sum of the first n terms of the b sequence -/
def T (seq : ArithmeticSequence) (n : ℕ) : ℝ := sorry

/-- Main theorem encapsulating the problem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 2*n) ∧
  (∀ n, seq.S n = n*(n+1)) ∧
  (∀ n, T seq n = 3^(n+1) - 3) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l669_66972


namespace NUMINAMATH_CALUDE_ellipse_equation_l669_66957

-- Define the ellipse parameters
variable (a b : ℝ)

-- Define the points A, B, and C
variable (A B C : ℝ × ℝ)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem ellipse_equation 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a ≠ b)
  (h4 : ∀ x y, a * x^2 + b * y^2 = 1 ↔ (x, y) = A ∨ (x, y) = B)
  (h5 : A.1 + A.2 = 1 ∧ B.1 + B.2 = 1)
  (h6 : C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h7 : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8)
  (h8 : (C.2 - O.2) / (C.1 - O.1) = Real.sqrt 2 / 2) :
  a = 1/3 ∧ b = Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l669_66957


namespace NUMINAMATH_CALUDE_work_completion_l669_66978

/-- The number of men in the first group that can complete a work in 18 days, 
    working 7 hours a day, given that 12 men can complete the same work in 12 days,
    also working 7 hours a day. -/
def number_of_men : ℕ := 8

theorem work_completion :
  ∀ (hours_per_day : ℕ) (days_first_group : ℕ) (days_second_group : ℕ),
    hours_per_day > 0 →
    days_first_group > 0 →
    days_second_group > 0 →
    number_of_men * hours_per_day * days_first_group = 12 * hours_per_day * days_second_group →
    hours_per_day = 7 →
    days_first_group = 18 →
    days_second_group = 12 →
    number_of_men = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l669_66978


namespace NUMINAMATH_CALUDE_blue_balloons_most_l669_66909

/-- Represents the color of a balloon -/
inductive BalloonColor
  | Red
  | Blue
  | Yellow

/-- Counts the number of balloons of a given color -/
def count_balloons (color : BalloonColor) : ℕ :=
  match color with
  | BalloonColor.Red => 6
  | BalloonColor.Blue => 12
  | BalloonColor.Yellow => 6

theorem blue_balloons_most : 
  (∀ c : BalloonColor, c ≠ BalloonColor.Blue → count_balloons BalloonColor.Blue > count_balloons c) ∧ 
  count_balloons BalloonColor.Red + count_balloons BalloonColor.Blue + count_balloons BalloonColor.Yellow = 24 ∧
  count_balloons BalloonColor.Blue = count_balloons BalloonColor.Red + 6 ∧
  count_balloons BalloonColor.Red = 24 / 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_balloons_most_l669_66909


namespace NUMINAMATH_CALUDE_unique_minimum_condition_l669_66916

/-- The function f(x) = ax³ + e^x has a unique minimum value if and only if a is in the range [-e²/12, 0) --/
theorem unique_minimum_condition (a : ℝ) :
  (∃ x₀ : ℝ, ∀ x : ℝ, a * x^3 + Real.exp x ≥ a * x₀^3 + Real.exp x₀ ∧
    (a * x^3 + Real.exp x = a * x₀^3 + Real.exp x₀ → x = x₀)) ↔
  a ∈ Set.Icc (-(Real.exp 2 / 12)) 0 ∧ a ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_minimum_condition_l669_66916


namespace NUMINAMATH_CALUDE_peanut_butter_servings_l669_66962

/-- Represents the amount of peanut butter in tablespoons -/
def peanut_butter : ℚ := 29 + 5 / 7

/-- Represents the size of one serving in tablespoons -/
def serving_size : ℚ := 2

/-- Represents the number of servings in the jar -/
def num_servings : ℚ := peanut_butter / serving_size

/-- Theorem stating that the number of servings in the jar is 14 3/7 -/
theorem peanut_butter_servings : num_servings = 14 + 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_servings_l669_66962


namespace NUMINAMATH_CALUDE_exponent_calculation_l669_66973

theorem exponent_calculation : (8^3 / 8^2) * 3^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l669_66973


namespace NUMINAMATH_CALUDE_sequence_difference_l669_66981

theorem sequence_difference (x : ℕ → ℕ)
  (h1 : x 1 = 1)
  (h2 : ∀ n, x n < x (n + 1))
  (h3 : ∀ n, x (n + 1) ≤ 2 * n) :
  ∀ k : ℕ, k > 0 → ∃ i j, k = x i - x j :=
by sorry

end NUMINAMATH_CALUDE_sequence_difference_l669_66981


namespace NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_for_ln_negative_l669_66947

theorem x_negative_necessary_not_sufficient_for_ln_negative :
  (∀ x, Real.log (x + 1) < 0 → x < 0) ∧
  (∃ x, x < 0 ∧ Real.log (x + 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_for_ln_negative_l669_66947


namespace NUMINAMATH_CALUDE_intersection_and_range_l669_66949

def A : Set ℝ := {x | x^2 + 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + 2*a^2 - 2 = 0}

theorem intersection_and_range :
  (A ∩ B 1 = {-4}) ∧
  (∀ a : ℝ, A ∩ B a = B a ↔ a < -1 ∨ a > 3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_range_l669_66949


namespace NUMINAMATH_CALUDE_notebook_cost_l669_66931

theorem notebook_cost (initial_money : ℕ) (notebooks_bought : ℕ) (books_bought : ℕ) 
  (book_cost : ℕ) (money_left : ℕ) :
  initial_money = 56 →
  notebooks_bought = 7 →
  books_bought = 2 →
  book_cost = 7 →
  money_left = 14 →
  ∃ (notebook_cost : ℕ), 
    notebook_cost * notebooks_bought + book_cost * books_bought = initial_money - money_left ∧
    notebook_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l669_66931


namespace NUMINAMATH_CALUDE_only_negative_four_has_no_sqrt_l669_66933

-- Define the set of numbers we're considering
def numbers : Set ℝ := {-4, 0, 0.5, 2}

-- Define what it means for a real number to have a square root
def has_sqrt (x : ℝ) : Prop := ∃ y : ℝ, y^2 = x

-- State the theorem
theorem only_negative_four_has_no_sqrt :
  ∀ x ∈ numbers, ¬(has_sqrt x) ↔ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_only_negative_four_has_no_sqrt_l669_66933


namespace NUMINAMATH_CALUDE_rectangle_area_l669_66918

/-- The area of a rectangle with perimeter 60 and length-to-width ratio 3:2 is 216 -/
theorem rectangle_area (l w : ℝ) : 
  (2 * l + 2 * w = 60) →  -- Perimeter condition
  (l = (3/2) * w) →       -- Length-to-width ratio condition
  (l * w = 216) :=        -- Area calculation
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l669_66918


namespace NUMINAMATH_CALUDE_probability_same_color_is_two_fifths_l669_66904

/-- Represents the number of white balls in the box -/
def num_white_balls : ℕ := 3

/-- Represents the number of black balls in the box -/
def num_black_balls : ℕ := 2

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := num_white_balls + num_black_balls

/-- Calculates the number of ways to choose 2 balls from the total number of balls -/
def total_ways_to_choose : ℕ := total_balls.choose 2

/-- Calculates the number of ways to choose 2 white balls -/
def ways_to_choose_white : ℕ := num_white_balls.choose 2

/-- Calculates the number of ways to choose 2 black balls -/
def ways_to_choose_black : ℕ := num_black_balls.choose 2

/-- Calculates the total number of ways to choose 2 balls of the same color -/
def same_color_ways : ℕ := ways_to_choose_white + ways_to_choose_black

/-- The probability of drawing two balls of the same color -/
def probability_same_color : ℚ := same_color_ways / total_ways_to_choose

theorem probability_same_color_is_two_fifths :
  probability_same_color = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_same_color_is_two_fifths_l669_66904


namespace NUMINAMATH_CALUDE_money_distribution_l669_66967

theorem money_distribution (ram gopal krishan : ℚ) : 
  (ram / gopal = 7 / 17) →
  (gopal / krishan = 7 / 17) →
  (ram = 490) →
  (krishan = 2890) :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l669_66967


namespace NUMINAMATH_CALUDE_value_of_M_l669_66964

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.55 * 4500) ∧ (M = 9900) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l669_66964


namespace NUMINAMATH_CALUDE_prob_sum_15_three_dice_l669_66968

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The sum we're looking for -/
def targetSum : ℕ := 15

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces^numDice

/-- The number of favorable outcomes (sum of 15) -/
def favorableOutcomes : ℕ := 7

/-- Theorem: The probability of rolling a sum of 15 with three standard 6-faced dice is 7/72 -/
theorem prob_sum_15_three_dice : 
  (favorableOutcomes : ℚ) / totalOutcomes = 7 / 72 := by sorry

end NUMINAMATH_CALUDE_prob_sum_15_three_dice_l669_66968


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_350_by_175_percent_l669_66922

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial + (percentage / 100) * initial = initial * (1 + percentage / 100) := by sorry

theorem increase_350_by_175_percent :
  350 + (175 / 100) * 350 = 962.5 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_350_by_175_percent_l669_66922


namespace NUMINAMATH_CALUDE_stratified_sampling_city_B_l669_66969

theorem stratified_sampling_city_B (total_points : ℕ) (city_B_points : ℕ) (sample_size : ℕ) :
  total_points = 450 →
  city_B_points = 150 →
  sample_size = 90 →
  (city_B_points : ℚ) / (total_points : ℚ) * (sample_size : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_city_B_l669_66969


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_7799_l669_66941

theorem largest_prime_factor_of_7799 : ∃ p : Nat, p.Prime ∧ p ∣ 7799 ∧ ∀ q : Nat, q.Prime → q ∣ 7799 → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_7799_l669_66941


namespace NUMINAMATH_CALUDE_product_97_103_l669_66923

theorem product_97_103 : 97 * 103 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_97_103_l669_66923


namespace NUMINAMATH_CALUDE_classroom_benches_l669_66925

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- The number of students that can be seated in the classroom -/
def studentsInBase7 : Nat := 321

/-- The number of students that sit on one bench -/
def studentsPerBench : Nat := 3

/-- The number of benches in the classroom -/
def numberOfBenches : Nat := base7ToBase10 studentsInBase7 / studentsPerBench

theorem classroom_benches :
  numberOfBenches = 54 := by
  sorry

end NUMINAMATH_CALUDE_classroom_benches_l669_66925


namespace NUMINAMATH_CALUDE_prob_white_given_popped_l669_66971

/-- Represents the color of a kernel -/
inductive KernelColor
  | White
  | Yellow
  | Blue

/-- The probability of selecting a kernel of a given color -/
def selectProb (c : KernelColor) : ℚ :=
  match c with
  | KernelColor.White => 2/5
  | KernelColor.Yellow => 1/5
  | KernelColor.Blue => 2/5

/-- The probability of a kernel popping given its color -/
def popProb (c : KernelColor) : ℚ :=
  match c with
  | KernelColor.White => 1/4
  | KernelColor.Yellow => 3/4
  | KernelColor.Blue => 1/2

/-- The probability that a randomly selected kernel that popped was white -/
theorem prob_white_given_popped :
  (selectProb KernelColor.White * popProb KernelColor.White) /
  (selectProb KernelColor.White * popProb KernelColor.White +
   selectProb KernelColor.Yellow * popProb KernelColor.Yellow +
   selectProb KernelColor.Blue * popProb KernelColor.Blue) = 2/9 := by
  sorry


end NUMINAMATH_CALUDE_prob_white_given_popped_l669_66971


namespace NUMINAMATH_CALUDE_cylindrical_tank_capacity_l669_66984

theorem cylindrical_tank_capacity (x : ℝ) 
  (h1 : 0.24 * x = 72) : x = 300 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_tank_capacity_l669_66984


namespace NUMINAMATH_CALUDE_paul_initial_pens_l669_66963

/-- Represents the number of items Paul has before and after the garage sale. -/
structure PaulItems where
  initial_books : ℕ
  initial_pens : ℕ
  remaining_books : ℕ
  remaining_pens : ℕ
  sold_pens : ℕ

/-- Theorem stating that Paul's initial number of pens is 42. -/
theorem paul_initial_pens (items : PaulItems)
    (h1 : items.initial_books = 143)
    (h2 : items.remaining_books = 113)
    (h3 : items.remaining_pens = 19)
    (h4 : items.sold_pens = 23) :
    items.initial_pens = 42 := by
  sorry

#check paul_initial_pens

end NUMINAMATH_CALUDE_paul_initial_pens_l669_66963


namespace NUMINAMATH_CALUDE_smallest_six_digit_divisible_l669_66921

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

theorem smallest_six_digit_divisible : 
  ∀ n : ℕ, 
    100000 ≤ n → 
    n < 1000000 → 
    (is_divisible_by n 25 ∧ 
     is_divisible_by n 35 ∧ 
     is_divisible_by n 45 ∧ 
     is_divisible_by n 15) → 
    n ≥ 100800 :=
sorry

end NUMINAMATH_CALUDE_smallest_six_digit_divisible_l669_66921


namespace NUMINAMATH_CALUDE_G_of_two_eq_six_l669_66997

noncomputable def G (x : ℝ) : ℝ :=
  1.2 * Real.sqrt (abs (x + 1.5)) + (7 / Real.pi) * Real.arctan (1.1 * Real.sqrt (abs (x + 1.5)))

theorem G_of_two_eq_six : G 2 = 6 := by sorry

end NUMINAMATH_CALUDE_G_of_two_eq_six_l669_66997


namespace NUMINAMATH_CALUDE_rob_quarters_l669_66998

def quarters : ℕ → ℚ
  | n => (n : ℚ) * (1 / 4)

def dimes : ℕ → ℚ
  | n => (n : ℚ) * (1 / 10)

def nickels : ℕ → ℚ
  | n => (n : ℚ) * (1 / 20)

def pennies : ℕ → ℚ
  | n => (n : ℚ) * (1 / 100)

theorem rob_quarters (x : ℕ) :
  quarters x + dimes 3 + nickels 5 + pennies 12 = 242 / 100 →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_rob_quarters_l669_66998


namespace NUMINAMATH_CALUDE_first_equation_is_root_multiplying_root_multiplying_with_root_two_l669_66979

/-- A quadratic equation ax^2 + bx + c = 0 is root-multiplying if it has two real roots and one root is twice the other -/
def is_root_multiplying (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 2 * x

/-- The first part of the theorem -/
theorem first_equation_is_root_multiplying :
  is_root_multiplying 1 (-3) 2 :=
sorry

/-- The second part of the theorem -/
theorem root_multiplying_with_root_two (a b : ℝ) :
  is_root_multiplying a b (-6) ∧ (∃ x : ℝ, a * x^2 + b * x - 6 = 0 ∧ x = 2) →
  (a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9) :=
sorry

end NUMINAMATH_CALUDE_first_equation_is_root_multiplying_root_multiplying_with_root_two_l669_66979


namespace NUMINAMATH_CALUDE_field_path_area_and_cost_l669_66913

/-- Represents the dimensions of a rectangular field with a path around it -/
structure FieldWithPath where
  field_length : ℝ
  field_width : ℝ
  path_width : ℝ

/-- Calculates the area of the path around a rectangular field -/
def path_area (f : FieldWithPath) : ℝ :=
  (f.field_length + 2 * f.path_width) * (f.field_width + 2 * f.path_width) - f.field_length * f.field_width

/-- Calculates the cost of constructing the path given a cost per square meter -/
def path_cost (f : FieldWithPath) (cost_per_sqm : ℝ) : ℝ :=
  path_area f * cost_per_sqm

/-- Theorem stating the area of the path and its construction cost for the given field -/
theorem field_path_area_and_cost :
  let f : FieldWithPath := { field_length := 75, field_width := 40, path_width := 2.5 }
  path_area f = 600 ∧ path_cost f 2 = 1200 := by sorry

end NUMINAMATH_CALUDE_field_path_area_and_cost_l669_66913


namespace NUMINAMATH_CALUDE_inequality_proof_l669_66988

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l669_66988


namespace NUMINAMATH_CALUDE_laura_savings_l669_66917

def original_price : ℝ := 3.00
def discount_rate : ℝ := 0.30
def num_notebooks : ℕ := 7

theorem laura_savings : 
  (num_notebooks : ℝ) * original_price * discount_rate = 6.30 := by
  sorry

end NUMINAMATH_CALUDE_laura_savings_l669_66917


namespace NUMINAMATH_CALUDE_abs_neg_three_fourths_l669_66961

theorem abs_neg_three_fourths : |(-3 : ℚ) / 4| = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_fourths_l669_66961


namespace NUMINAMATH_CALUDE_M_superset_N_l669_66926

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | ∃ a ∈ M, x = a^2}

theorem M_superset_N : M ⊇ N := by
  sorry

end NUMINAMATH_CALUDE_M_superset_N_l669_66926
