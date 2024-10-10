import Mathlib

namespace first_year_after_2010_with_sum_of_digits_10_l4015_401518

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isFirstYearAfter2010WithSumOfDigits10 (year : Nat) : Prop :=
  year > 2010 ∧ 
  sumOfDigits year = 10 ∧ 
  ∀ y, 2010 < y ∧ y < year → sumOfDigits y ≠ 10

theorem first_year_after_2010_with_sum_of_digits_10 :
  isFirstYearAfter2010WithSumOfDigits10 2035 := by sorry

end first_year_after_2010_with_sum_of_digits_10_l4015_401518


namespace ratio_equality_l4015_401525

theorem ratio_equality (a b : ℝ) (h : 7 * a = 8 * b) : (a / 8) / (b / 7) = 1 := by
  sorry

end ratio_equality_l4015_401525


namespace solve_exponential_equation_l4015_401570

theorem solve_exponential_equation :
  ∃ w : ℝ, (2 : ℝ)^(2*w) = 8^(w-4) → w = 12 := by
  sorry

end solve_exponential_equation_l4015_401570


namespace greatest_integer_problem_l4015_401564

theorem greatest_integer_problem : 
  ∃ (m : ℕ), 
    (m < 150) ∧ 
    (∃ (a : ℕ), m = 9 * a - 2) ∧ 
    (∃ (b : ℕ), m = 5 * b + 4) ∧ 
    (∀ (n : ℕ), 
      (n < 150) → 
      (∃ (c : ℕ), n = 9 * c - 2) → 
      (∃ (d : ℕ), n = 5 * d + 4) → 
      n ≤ m) ∧
    m = 124 :=
by sorry

end greatest_integer_problem_l4015_401564


namespace divisible_by_fifteen_l4015_401503

theorem divisible_by_fifteen (x : ℤ) : 
  (∃ k : ℤ, x^2 + 2*x + 6 = 15 * k) ↔ 
  (∃ t : ℤ, x = 15*t - 6 ∨ x = 15*t + 4) := by
sorry

end divisible_by_fifteen_l4015_401503


namespace regular_hexagon_interior_angle_l4015_401573

/-- The measure of each interior angle in a regular hexagon is 120 degrees. -/
theorem regular_hexagon_interior_angle : ℝ := by
  -- Define a regular hexagon
  let regular_hexagon : Nat := 6

  -- Define the formula for the sum of interior angles of a polygon
  let sum_of_interior_angles (n : Nat) : ℝ := (n - 2) * 180

  -- Calculate the sum of interior angles for a hexagon
  let total_angle_sum : ℝ := sum_of_interior_angles regular_hexagon

  -- Calculate the measure of each interior angle
  let interior_angle : ℝ := total_angle_sum / regular_hexagon

  -- Prove that the interior angle is 120 degrees
  sorry

end regular_hexagon_interior_angle_l4015_401573


namespace sum_of_two_smallest_prime_factors_of_540_l4015_401511

theorem sum_of_two_smallest_prime_factors_of_540 :
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧
  p ∣ 540 ∧ q ∣ 540 ∧
  (∀ (r : ℕ), Nat.Prime r → r ∣ 540 → r ≥ p) ∧
  (∀ (r : ℕ), Nat.Prime r → r ∣ 540 → r ≠ p → r ≥ q) ∧
  p + q = 5 := by
sorry

end sum_of_two_smallest_prime_factors_of_540_l4015_401511


namespace volleyball_count_l4015_401545

theorem volleyball_count (total : ℕ) (soccer : ℕ) (basketball : ℕ) (tennis : ℕ) (baseball : ℕ) (hockey : ℕ) (volleyball : ℕ) :
  total = 180 →
  soccer = 20 →
  basketball = soccer + 5 →
  tennis = 2 * soccer →
  baseball = soccer + 10 →
  hockey = tennis / 2 →
  volleyball = total - (soccer + basketball + tennis + baseball + hockey) →
  volleyball = 45 := by
sorry

end volleyball_count_l4015_401545


namespace no_matrix_sine_exists_l4015_401578

open Matrix

/-- Definition of matrix sine function -/
noncomputable def matrixSine (A : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ∑' n, ((-1)^n / (2*n+1).factorial : ℝ) • (A^(2*n+1))

/-- The statement to be proved -/
theorem no_matrix_sine_exists : 
  ¬ ∃ A : Matrix (Fin 2) (Fin 2) ℝ, matrixSine A = ![![1, 1996], ![0, 1]] :=
sorry

end no_matrix_sine_exists_l4015_401578


namespace edge_sum_is_96_l4015_401593

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  -- Three dimensions in geometric progression
  a : ℝ
  r : ℝ
  -- Volume is 512 cm³
  volume_eq : a * (a * r) * (a * r * r) = 512
  -- Surface area is 384 cm²
  surface_area_eq : 2 * (a * (a * r) + a * (a * r * r) + (a * r) * (a * r * r)) = 384

/-- The sum of all edge lengths of the rectangular solid is 96 cm -/
theorem edge_sum_is_96 (solid : RectangularSolid) :
  4 * (solid.a + solid.a * solid.r + solid.a * solid.r * solid.r) = 96 := by
  sorry

end edge_sum_is_96_l4015_401593


namespace charlie_spent_56250_l4015_401542

/-- The amount Charlie spent on acorns -/
def charlie_spent (alice_acorns bob_acorns charlie_acorns : ℕ) 
  (bob_total : ℚ) (alice_multiplier : ℕ) : ℚ :=
  let bob_price := bob_total / bob_acorns
  let alice_price := alice_multiplier * bob_price
  let average_price := (bob_price + alice_price) / 2
  charlie_acorns * average_price

/-- Theorem stating that Charlie spent $56,250 on acorns -/
theorem charlie_spent_56250 :
  charlie_spent 3600 2400 4500 6000 9 = 56250 := by
  sorry

end charlie_spent_56250_l4015_401542


namespace red_beads_in_necklace_l4015_401555

/-- Represents the number of red beads in each group -/
def redBeadsInGroup (n : ℕ) : ℕ := 2 * n

/-- Represents the total number of red beads up to the nth group -/
def totalRedBeads (n : ℕ) : ℕ := n * (n + 1)

/-- Represents the total number of beads (red and white) up to the nth group -/
def totalBeads (n : ℕ) : ℕ := n + totalRedBeads n

theorem red_beads_in_necklace :
  ∃ n : ℕ, totalBeads n ≤ 99 ∧ totalBeads (n + 1) > 99 ∧ totalRedBeads n = 90 := by
  sorry

end red_beads_in_necklace_l4015_401555


namespace no_integer_satisfies_conditions_l4015_401585

theorem no_integer_satisfies_conditions : 
  ¬∃ (n : ℤ), (30 - 6 * n > 18) ∧ (2 * n + 5 = 11) := by
  sorry

end no_integer_satisfies_conditions_l4015_401585


namespace bill_sunday_saturday_difference_l4015_401515

theorem bill_sunday_saturday_difference (bill_sat bill_sun julia_sun : ℕ) : 
  bill_sun > bill_sat →
  julia_sun = 2 * bill_sun →
  bill_sat + bill_sun + julia_sun = 32 →
  bill_sun = 9 →
  bill_sun - bill_sat = 4 :=
by
  sorry

end bill_sunday_saturday_difference_l4015_401515


namespace product_inequality_l4015_401589

theorem product_inequality (a b c x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hx₃ : 0 < x₃) (hx₄ : 0 < x₄) (hx₅ : 0 < x₅)
  (sum_abc : a + b + c = 1)
  (prod_x : x₁ * x₂ * x₃ * x₄ * x₅ = 1) :
  (a * x₁^2 + b * x₁ + c) * 
  (a * x₂^2 + b * x₂ + c) * 
  (a * x₃^2 + b * x₃ + c) * 
  (a * x₄^2 + b * x₄ + c) * 
  (a * x₅^2 + b * x₅ + c) ≥ 1 := by
sorry

end product_inequality_l4015_401589


namespace difference_largest_smallest_l4015_401502

/-- Represents a three-digit positive integer with no repeated digits -/
structure ThreeDigitNoRepeat where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : 
    1 ≤ hundreds ∧ hundreds ≤ 9 ∧
    0 ≤ tens ∧ tens ≤ 9 ∧
    0 ≤ ones ∧ ones ≤ 9 ∧
    hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

/-- Converts a ThreeDigitNoRepeat to its integer value -/
def ThreeDigitNoRepeat.toNat (n : ThreeDigitNoRepeat) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The largest three-digit positive integer with no repeated digits -/
def largest : ThreeDigitNoRepeat := {
  hundreds := 9
  tens := 8
  ones := 7
  is_valid := by sorry
}

/-- The smallest three-digit positive integer with no repeated digits -/
def smallest : ThreeDigitNoRepeat := {
  hundreds := 1
  tens := 0
  ones := 2
  is_valid := by sorry
}

/-- The main theorem -/
theorem difference_largest_smallest : 
  largest.toNat - smallest.toNat = 885 := by sorry

end difference_largest_smallest_l4015_401502


namespace mollys_current_age_l4015_401565

/-- Given the ratio of Sandy's age to Molly's age and Sandy's age after 6 years, 
    calculate Molly's current age. -/
theorem mollys_current_age 
  (sandy_age : ℕ) 
  (molly_age : ℕ) 
  (h1 : sandy_age / molly_age = 4 / 3)  -- Ratio of ages
  (h2 : sandy_age + 6 = 30)             -- Sandy's age after 6 years
  : molly_age = 18 :=
by sorry

end mollys_current_age_l4015_401565


namespace rectangle_area_l4015_401577

/-- Given a rectangle with length thrice its breadth and diagonal 26 meters,
    prove that its area is 202.8 square meters. -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let d := 26
  d^2 = l^2 + b^2 → b * l = 202.8 := by
sorry

end rectangle_area_l4015_401577


namespace intersection_reciprocal_sum_l4015_401547

/-- Given a line intersecting y = x^2 at (x₁, x₁²) and (x₂, x₂²), and the x-axis at (x₃, 0), 
    prove that 1/x₁ + 1/x₂ = 1/x₃ -/
theorem intersection_reciprocal_sum (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0) (h₃ : x₃ ≠ 0)
  (line_eq : ∃ (k m : ℝ), ∀ x y, y = k * x + m ↔ (x = x₁ ∧ y = x₁^2) ∨ (x = x₂ ∧ y = x₂^2) ∨ (x = x₃ ∧ y = 0)) :
  1 / x₁ + 1 / x₂ = 1 / x₃ := by
  sorry

end intersection_reciprocal_sum_l4015_401547


namespace arctan_sum_special_l4015_401598

theorem arctan_sum_special : Real.arctan (3/4) + Real.arctan (4/3) = π/2 := by
  sorry

end arctan_sum_special_l4015_401598


namespace statue_cost_l4015_401521

theorem statue_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) :
  selling_price = 620 →
  profit_percentage = 25 →
  selling_price = original_cost * (1 + profit_percentage / 100) →
  original_cost = 496 := by
sorry

end statue_cost_l4015_401521


namespace B_equals_zero_one_two_l4015_401557

def B : Set ℤ := {x | -3 < 2*x - 1 ∧ 2*x - 1 < 5}

theorem B_equals_zero_one_two : B = {0, 1, 2} := by sorry

end B_equals_zero_one_two_l4015_401557


namespace sum_of_repeating_decimals_l4015_401590

def repeating_decimal_4 : ℚ := 4/9
def repeating_decimal_7 : ℚ := 7/9
def repeating_decimal_3 : ℚ := 1/3

theorem sum_of_repeating_decimals :
  repeating_decimal_4 + repeating_decimal_7 - repeating_decimal_3 = 8/9 :=
by sorry

end sum_of_repeating_decimals_l4015_401590


namespace five_solutions_l4015_401528

/-- The system of equations has exactly 5 real solutions -/
theorem five_solutions (x y z w : ℝ) : 
  (x = z + w + z*w*x ∧
   y = w + x + w*x*y ∧
   z = x + y + x*y*z ∧
   w = y + z + y*z*w) →
  ∃! (sol : Finset (ℝ × ℝ × ℝ × ℝ)), 
    sol.card = 5 ∧ 
    ∀ (a b c d : ℝ), (a, b, c, d) ∈ sol ↔ 
      (a = c + d + c*d*a ∧
       b = d + a + d*a*b ∧
       c = a + b + a*b*c ∧
       d = b + c + b*c*d) :=
by sorry

end five_solutions_l4015_401528


namespace solution_equivalence_l4015_401535

def solution_set : Set ℝ := {x | 1 < x ∧ x ≤ 3}

def inequality_system (x : ℝ) : Prop := 1 - x < 0 ∧ x - 3 ≤ 0

theorem solution_equivalence : 
  ∀ x : ℝ, x ∈ solution_set ↔ inequality_system x := by
  sorry

end solution_equivalence_l4015_401535


namespace problem_statement_l4015_401501

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a ^ b = 343) (h5 : b ^ c = 10) (h6 : a ^ c = 7) : b ^ b = 1000 := by
  sorry

end problem_statement_l4015_401501


namespace fraction_simplification_l4015_401563

theorem fraction_simplification :
  (270 : ℚ) / 24 * 7 / 210 * 6 / 4 = 9 / 2 := by
  sorry

end fraction_simplification_l4015_401563


namespace solve_for_B_l4015_401579

theorem solve_for_B : ∀ (A B : ℕ), 
  (A ≥ 1 ∧ A ≤ 9) →  -- Ensure A is a single digit
  (B ≥ 0 ∧ B ≤ 9) →  -- Ensure B is a single digit
  632 - (100 * A + 10 * B + 1) = 41 → 
  B = 9 := by
sorry

end solve_for_B_l4015_401579


namespace min_groups_for_30_students_max_12_l4015_401576

/-- Given a total number of students and a maximum group size, 
    calculate the minimum number of equal-sized groups. -/
def min_groups (total_students : ℕ) (max_group_size : ℕ) : ℕ :=
  let divisors := (Finset.range total_students).filter (λ d => total_students % d = 0)
  let valid_divisors := divisors.filter (λ d => d ≤ max_group_size)
  total_students / valid_divisors.max' (by sorry)

/-- The theorem stating that for 30 students and a maximum group size of 12, 
    the minimum number of equal-sized groups is 3. -/
theorem min_groups_for_30_students_max_12 :
  min_groups 30 12 = 3 := by sorry

end min_groups_for_30_students_max_12_l4015_401576


namespace problem_statement_l4015_401541

theorem problem_statement (a b : ℝ) (hab : a * b > 0) (hab2 : a^2 * b = 4) :
  (∃ (m : ℝ), ∀ (a b : ℝ), a * b > 0 → a^2 * b = 4 → a + b ≥ m ∧ 
    ∀ (m' : ℝ), (∀ (a b : ℝ), a * b > 0 → a^2 * b = 4 → a + b ≥ m') → m' ≤ m) ∧
  (∀ (x : ℝ), 2 * |x - 1| + |x| ≤ a + b ↔ -1/3 ≤ x ∧ x ≤ 5/3) :=
sorry

end problem_statement_l4015_401541


namespace park_rose_bushes_l4015_401540

/-- Calculate the final number of rose bushes in the park -/
def final_rose_bushes (initial : ℕ) (planned : ℕ) (rate : ℕ) (removed : ℕ) : ℕ :=
  initial + planned * rate - removed

/-- Theorem stating the final number of rose bushes in the park -/
theorem park_rose_bushes : final_rose_bushes 2 4 3 5 = 9 := by
  sorry

end park_rose_bushes_l4015_401540


namespace older_friend_age_l4015_401533

theorem older_friend_age (younger_age older_age : ℕ) : 
  older_age - younger_age = 2 →
  younger_age + older_age = 74 →
  older_age = 38 :=
by
  sorry

end older_friend_age_l4015_401533


namespace pulley_centers_distance_l4015_401549

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (hr₁ : r₁ = 10)
  (hr₂ : r₂ = 6)
  (hcd : contact_distance = 30) :
  let center_distance := Real.sqrt ((contact_distance ^ 2) + ((r₁ - r₂) ^ 2))
  center_distance = 2 * Real.sqrt 229 := by sorry

end pulley_centers_distance_l4015_401549


namespace truncated_cone_volume_l4015_401562

/-- The volume of a truncated cone with specific diagonal properties -/
theorem truncated_cone_volume 
  (l : ℝ) 
  (α : ℝ) 
  (h_positive : l > 0)
  (h_angle : 0 < α ∧ α < π)
  (h_diagonal_ratio : ∃ (k : ℝ), k > 0 ∧ 2 * k = l ∧ k = l / 3)
  : ∃ (V : ℝ), V = (7 / 54) * π * l^3 * Real.sin α * Real.sin (α / 2) :=
sorry

end truncated_cone_volume_l4015_401562


namespace system_solution_l4015_401584

theorem system_solution : 
  ∃ (x₁ x₂ y₁ y₂ : ℝ), 
    (2 * x₁^2 - 5 * x₁ + 3 = 0) ∧ 
    (y₁ = 3 * x₁ + 1) ∧
    (2 * x₂^2 - 5 * x₂ + 3 = 0) ∧ 
    (y₂ = 3 * x₂ + 1) ∧
    (x₁ = 1.5 ∧ y₁ = 5.5) ∧ 
    (x₂ = 1 ∧ y₂ = 4) ∧
    (∀ (x y : ℝ), (2 * x^2 - 5 * x + 3 = 0) ∧ (y = 3 * x + 1) → 
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end system_solution_l4015_401584


namespace williams_books_l4015_401583

theorem williams_books (w : ℕ) : 
  (3 * w + 8 + 4 = w + 2 * 8) → w = 2 := by
  sorry

end williams_books_l4015_401583


namespace f_3_equals_3_l4015_401500

def f (x : ℝ) : ℝ := 2 * (x - 1) - 1

theorem f_3_equals_3 : f 3 = 3 := by
  sorry

end f_3_equals_3_l4015_401500


namespace circle_radius_tripled_area_l4015_401572

theorem circle_radius_tripled_area (r : ℝ) : r > 0 →
  (π * (r + 3)^2 = 3 * π * r^2) → r = (3 * (1 + Real.sqrt 3)) / 2 := by
  sorry

end circle_radius_tripled_area_l4015_401572


namespace horners_rule_polynomial_l4015_401523

theorem horners_rule_polynomial (x : ℝ) : 
  x^3 + 2*x^2 + x - 1 = ((x + 2)*x + 1)*x - 1 := by
  sorry

end horners_rule_polynomial_l4015_401523


namespace boys_usual_time_to_school_l4015_401561

theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) : 
  (usual_time > 0) →
  (3 / 2 * usual_rate * (usual_time - 4) = usual_rate * usual_time) →
  usual_time = 12 := by
  sorry

end boys_usual_time_to_school_l4015_401561


namespace no_digit_satisfies_property_l4015_401586

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Checks if the decimal representation of a natural number ends with at least k repetitions of a digit -/
def EndsWithRepeatedDigit (num : ℕ) (d : Digit) (k : ℕ) : Prop :=
  ∃ m : ℕ, num % (10^k) = d.val * ((10^k - 1) / 9)

/-- The main theorem stating that no digit satisfies the given property -/
theorem no_digit_satisfies_property : 
  ¬ ∃ z : Digit, ∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ EndsWithRepeatedDigit (n^9) z k :=
by sorry


end no_digit_satisfies_property_l4015_401586


namespace smallest_divisible_by_999_l4015_401594

theorem smallest_divisible_by_999 :
  ∃ (a : ℕ), (∀ (n : ℕ), Odd n → (999 ∣ 2^(5*n) + a*5^n)) ∧ 
  (∀ (b : ℕ), b < a → ∃ (m : ℕ), Odd m ∧ ¬(999 ∣ 2^(5*m) + b*5^m)) ∧
  a = 539 := by
sorry

end smallest_divisible_by_999_l4015_401594


namespace eighth_fib_is_21_l4015_401588

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Theorem: The 8th term of the Fibonacci sequence is 21
theorem eighth_fib_is_21 : fib 7 = 21 := by
  sorry

end eighth_fib_is_21_l4015_401588


namespace circle_center_l4015_401559

/-- The center of a circle defined by the equation 4x^2 - 8x + 4y^2 - 16y + 20 = 0 is (1, 2) -/
theorem circle_center (x y : ℝ) : 
  (4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0) → 
  (∃ (h : ℝ), h = 0 ∧ (x - 1)^2 + (y - 2)^2 = h) := by
sorry

end circle_center_l4015_401559


namespace remainder_of_2357916_div_8_l4015_401587

theorem remainder_of_2357916_div_8 : 2357916 % 8 = 4 := by
  sorry

end remainder_of_2357916_div_8_l4015_401587


namespace nicholas_crackers_l4015_401504

theorem nicholas_crackers (marcus_crackers : ℕ) (mona_crackers : ℕ) (nicholas_crackers : ℕ) : 
  marcus_crackers = 27 →
  marcus_crackers = 3 * mona_crackers →
  nicholas_crackers = mona_crackers + 6 →
  nicholas_crackers = 15 :=
by
  sorry

end nicholas_crackers_l4015_401504


namespace greatest_integer_with_gcd_eight_gcd_of_136_and_16_is_136_less_than_150_main_result_l4015_401553

theorem greatest_integer_with_gcd_eight (n : ℕ) : n < 150 ∧ n.gcd 16 = 8 → n ≤ 136 :=
by sorry

theorem gcd_of_136_and_16 : Nat.gcd 136 16 = 8 :=
by sorry

theorem is_136_less_than_150 : 136 < 150 :=
by sorry

theorem main_result : ∃ (n : ℕ), n < 150 ∧ n.gcd 16 = 8 ∧ 
  ∀ (m : ℕ), m < 150 ∧ m.gcd 16 = 8 → m ≤ n :=
by sorry

end greatest_integer_with_gcd_eight_gcd_of_136_and_16_is_136_less_than_150_main_result_l4015_401553


namespace journey_time_proof_l4015_401544

/-- Proves that given a round trip where the outbound journey is at 60 km/h, 
    the return journey is at 90 km/h, and the total time is 2 hours, 
    the time taken for the outbound journey is 72 minutes. -/
theorem journey_time_proof (distance : ℝ) 
    (h1 : distance / 60 + distance / 90 = 2) : 
    distance / 60 * 60 = 72 := by
  sorry

#check journey_time_proof

end journey_time_proof_l4015_401544


namespace triangle_4_5_6_l4015_401574

/-- A triangle can be formed from three line segments if the sum of any two sides is greater than the third side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that line segments of lengths 4, 5, and 6 can form a triangle. -/
theorem triangle_4_5_6 : can_form_triangle 4 5 6 := by
  sorry

end triangle_4_5_6_l4015_401574


namespace longest_tape_measure_l4015_401524

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 100) 
  (hb : b = 225) 
  (hc : c = 780) : 
  Nat.gcd a (Nat.gcd b c) = 5 := by
  sorry

end longest_tape_measure_l4015_401524


namespace cubic_equation_roots_progression_l4015_401575

/-- Represents a cubic equation x³ + ax² + bx + c = 0 -/
structure CubicEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- The roots of a cubic equation -/
structure CubicRoots (α : Type*) [Field α] where
  x₁ : α
  x₂ : α
  x₃ : α

/-- Checks if the roots form an arithmetic progression -/
def is_arithmetic_progression {α : Type*} [Field α] (roots : CubicRoots α) : Prop :=
  roots.x₁ - roots.x₂ = roots.x₂ - roots.x₃

/-- Checks if the roots form a geometric progression -/
def is_geometric_progression {α : Type*} [Field α] (roots : CubicRoots α) : Prop :=
  roots.x₂ / roots.x₁ = roots.x₃ / roots.x₂

/-- Checks if the roots form a harmonic sequence -/
def is_harmonic_sequence {α : Type*} [Field α] (roots : CubicRoots α) : Prop :=
  (roots.x₁ - roots.x₂) / (roots.x₂ - roots.x₃) = roots.x₁ / roots.x₃

theorem cubic_equation_roots_progression {α : Type*} [Field α] (eq : CubicEquation α) (roots : CubicRoots α) :
  (is_arithmetic_progression roots ↔ (2 * eq.a^3 + 27 * eq.c) / (9 * eq.a) = eq.b) ∧
  (is_geometric_progression roots ↔ eq.b = eq.a * (eq.c^(1/3))) ∧
  (is_harmonic_sequence roots ↔ eq.a = (2 * eq.b^3 + 27 * eq.c) / (9 * eq.b^2)) :=
sorry

end cubic_equation_roots_progression_l4015_401575


namespace quadratic_minimum_value_l4015_401538

theorem quadratic_minimum_value : 
  ∀ x : ℝ, 3 * x^2 - 18 * x + 12 ≥ -15 ∧ 
  ∃ x : ℝ, 3 * x^2 - 18 * x + 12 = -15 := by
  sorry

end quadratic_minimum_value_l4015_401538


namespace sams_book_count_l4015_401566

theorem sams_book_count :
  let used_adventure_books : ℝ := 13.0
  let used_mystery_books : ℝ := 17.0
  let new_crime_books : ℝ := 15.0
  let total_books := used_adventure_books + used_mystery_books + new_crime_books
  total_books = 45.0 := by sorry

end sams_book_count_l4015_401566


namespace union_of_sets_l4015_401520

def M : Set Int := {-1, 3, -5}
def N (a : Int) : Set Int := {a + 2, a^2 - 6}

theorem union_of_sets :
  ∃ a : Int, (M ∩ N a = {3}) → (M ∪ N a = {-5, -1, 3, 5}) := by
  sorry

end union_of_sets_l4015_401520


namespace subset_implies_m_equals_two_l4015_401534

def A (m : ℝ) : Set ℝ := {-2, 3, 4*m - 4}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_equals_two (m : ℝ) :
  B m ⊆ A m → m = 2 := by
  sorry

end subset_implies_m_equals_two_l4015_401534


namespace triangle_side_length_l4015_401508

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π / 3 → a = Real.sqrt 3 → b = 1 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  c ^ 2 = a ^ 2 + b ^ 2 →
  c = 2 := by sorry

end triangle_side_length_l4015_401508


namespace c_share_is_56_l4015_401567

/-- Represents the share of money for each person -/
structure Share where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions of the money distribution problem -/
def moneyDistribution (s : Share) : Prop :=
  s.b = 0.65 * s.a ∧ 
  s.c = 0.40 * s.a ∧ 
  s.a + s.b + s.c = 287

/-- Theorem stating that under the given conditions, C's share is 56 -/
theorem c_share_is_56 :
  ∃ s : Share, moneyDistribution s ∧ s.c = 56 := by
  sorry


end c_share_is_56_l4015_401567


namespace range_of_f_l4015_401569

-- Define the function
def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.range f, -8 ≤ y ∧ y ≤ 8 ∧
  ∀ z, -8 ≤ z ∧ z ≤ 8 → ∃ x, f x = z :=
by sorry

end range_of_f_l4015_401569


namespace triangle_properties_l4015_401536

-- Define the triangle ABC
def triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Condition 1: 2c = a + 2b*cos(A)
  2 * c = a + 2 * b * Real.cos A ∧
  -- Condition 2: Area of triangle ABC is √3
  (1 / 2) * a * c * Real.sin B = Real.sqrt 3 ∧
  -- Condition 3: b = √13
  b = Real.sqrt 13

-- Theorem statement
theorem triangle_properties (a b c A B C : ℝ) 
  (h : triangle a b c A B C) : 
  B = Real.pi / 3 ∧ 
  a + b + c = 5 + Real.sqrt 13 :=
by sorry

end triangle_properties_l4015_401536


namespace common_roots_solution_l4015_401532

/-- Two cubic polynomials with common roots -/
def poly1 (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 14*x + 8

def poly2 (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + 17*x + 10

/-- The polynomials have two distinct common roots -/
def has_two_common_roots (a b : ℝ) : Prop :=
  ∃ r s : ℝ, r ≠ s ∧ poly1 a r = 0 ∧ poly1 a s = 0 ∧ poly2 b r = 0 ∧ poly2 b s = 0

/-- The main theorem -/
theorem common_roots_solution :
  has_two_common_roots 7 8 :=
sorry

end common_roots_solution_l4015_401532


namespace greatest_a_for_equation_l4015_401509

theorem greatest_a_for_equation :
  ∃ (a : ℝ), 
    (∀ (x : ℝ), (5 * Real.sqrt ((2 * x)^2 + 1) - 4 * x^2 - 1) / (Real.sqrt (1 + 4 * x^2) + 3) = 3 → x ≤ a) ∧
    (5 * Real.sqrt ((2 * a)^2 + 1) - 4 * a^2 - 1) / (Real.sqrt (1 + 4 * a^2) + 3) = 3 ∧
    a = Real.sqrt ((5 + Real.sqrt 10) / 2) :=
by sorry

end greatest_a_for_equation_l4015_401509


namespace cereal_eating_time_l4015_401516

/-- The time it takes for two people to eat a certain amount of cereal together -/
def eating_time (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- Mr. Fat's eating rate in pounds per minute -/
def mr_fat_rate : ℚ := 1 / 20

/-- Mr. Thin's eating rate in pounds per minute -/
def mr_thin_rate : ℚ := 1 / 30

/-- The amount of cereal to be eaten in pounds -/
def cereal_amount : ℚ := 3

theorem cereal_eating_time :
  eating_time mr_fat_rate mr_thin_rate cereal_amount = 36 := by
  sorry

end cereal_eating_time_l4015_401516


namespace fraction_to_decimal_l4015_401560

theorem fraction_to_decimal (h : 625 = 5^4) : 17 / 625 = 0.0272 := by
  sorry

end fraction_to_decimal_l4015_401560


namespace monroe_collection_legs_l4015_401591

def spider_count : ℕ := 8
def ant_count : ℕ := 12
def spider_legs : ℕ := 8
def ant_legs : ℕ := 6

def total_legs : ℕ := spider_count * spider_legs + ant_count * ant_legs

theorem monroe_collection_legs : total_legs = 136 := by
  sorry

end monroe_collection_legs_l4015_401591


namespace xiaoming_red_pens_l4015_401517

/-- The number of red pens bought by Xiaoming -/
def red_pens : ℕ := 36

/-- The total number of pens bought -/
def total_pens : ℕ := 66

/-- The original price of a red pen in yuan -/
def red_pen_price : ℚ := 5

/-- The original price of a black pen in yuan -/
def black_pen_price : ℚ := 9

/-- The discount rate for red pens -/
def red_discount : ℚ := 85 / 100

/-- The discount rate for black pens -/
def black_discount : ℚ := 80 / 100

/-- The discount rate on the total price -/
def total_discount : ℚ := 18 / 100

theorem xiaoming_red_pens :
  red_pens = 36 ∧
  red_pens ≤ total_pens ∧
  (red_pen_price * red_pens + black_pen_price * (total_pens - red_pens)) * (1 - total_discount) =
  red_pen_price * red_discount * red_pens + black_pen_price * black_discount * (total_pens - red_pens) :=
by sorry

end xiaoming_red_pens_l4015_401517


namespace stella_spent_40_l4015_401554

/-- Represents the inventory and financial data of Stella's antique shop --/
structure AntiqueShop where
  dolls : ℕ
  clocks : ℕ
  glasses : ℕ
  doll_price : ℕ
  clock_price : ℕ
  glass_price : ℕ
  profit : ℕ

/-- Calculates the total revenue from selling all items --/
def total_revenue (shop : AntiqueShop) : ℕ :=
  shop.dolls * shop.doll_price + shop.clocks * shop.clock_price + shop.glasses * shop.glass_price

/-- Theorem stating that Stella spent $40 to buy everything --/
theorem stella_spent_40 (shop : AntiqueShop) 
    (h1 : shop.dolls = 3)
    (h2 : shop.clocks = 2)
    (h3 : shop.glasses = 5)
    (h4 : shop.doll_price = 5)
    (h5 : shop.clock_price = 15)
    (h6 : shop.glass_price = 4)
    (h7 : shop.profit = 25) : 
  total_revenue shop - shop.profit = 40 := by
  sorry

end stella_spent_40_l4015_401554


namespace cube_root_of_three_times_two_to_seven_l4015_401505

theorem cube_root_of_three_times_two_to_seven (x : ℝ) :
  x = Real.rpow 2 7 + Real.rpow 2 7 + Real.rpow 2 7 →
  Real.rpow x (1/3) = 4 * Real.rpow 6 (1/3) :=
by sorry

end cube_root_of_three_times_two_to_seven_l4015_401505


namespace fraction_problem_l4015_401599

theorem fraction_problem (N : ℝ) (x : ℝ) : 
  (0.40 * N = 420) → 
  (x * (1/3) * (2/5) * N = 35) → 
  x = 1/4 := by sorry

end fraction_problem_l4015_401599


namespace cannot_obtain_1998_pow_7_initial_condition_final_not_divisible_main_result_l4015_401551

def board_operation (n : ℕ) : ℕ :=
  let last_digit := n % 10
  (n / 10) + 5 * last_digit

theorem cannot_obtain_1998_pow_7 (n : ℕ) (h : 7 ∣ n) :
  ∀ k : ℕ, 7 ∣ (board_operation^[k] n) ∧ (board_operation^[k] n) ≠ 1998^7 :=
by sorry

theorem initial_condition : 7 ∣ 7^1998 :=
by sorry

theorem final_not_divisible : ¬(7 ∣ 1998^7) :=
by sorry

theorem main_result : ∀ k : ℕ, (board_operation^[k] 7^1998) ≠ 1998^7 :=
by sorry

end cannot_obtain_1998_pow_7_initial_condition_final_not_divisible_main_result_l4015_401551


namespace quadratic_inequality_solution_l4015_401530

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + 2

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | 2/a ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem quadratic_inequality_solution (a : ℝ) (h : a < 0) :
  {x : ℝ | f a x ≥ 0} = solution_set a :=
by sorry

end quadratic_inequality_solution_l4015_401530


namespace packages_per_truck_l4015_401558

theorem packages_per_truck (total_packages : ℕ) (num_trucks : ℕ) 
  (h1 : total_packages = 490) (h2 : num_trucks = 7) :
  total_packages / num_trucks = 70 := by
  sorry

end packages_per_truck_l4015_401558


namespace lecture_slides_theorem_our_lecture_slides_l4015_401548

/-- Represents a lecture with slides -/
structure Lecture where
  duration : ℕ  -- Duration of the lecture in minutes
  initial_slides : ℕ  -- Number of slides changed in the initial period
  initial_period : ℕ  -- Initial period in minutes
  total_slides : ℕ  -- Total number of slides used

/-- Calculates the total number of slides used in a lecture -/
def calculate_total_slides (l : Lecture) : ℕ :=
  (l.duration * l.initial_slides) / l.initial_period

/-- Theorem stating that for the given lecture conditions, the total slides used is 100 -/
theorem lecture_slides_theorem (l : Lecture) 
  (h1 : l.duration = 50)
  (h2 : l.initial_slides = 4)
  (h3 : l.initial_period = 2) :
  calculate_total_slides l = 100 := by
  sorry

/-- The specific lecture instance -/
def our_lecture : Lecture := {
  duration := 50,
  initial_slides := 4,
  initial_period := 2,
  total_slides := 100
}

/-- Proof that our specific lecture uses 100 slides -/
theorem our_lecture_slides : 
  calculate_total_slides our_lecture = 100 := by
  sorry

end lecture_slides_theorem_our_lecture_slides_l4015_401548


namespace isosceles_trapezoid_slope_sum_l4015_401543

-- Define the trapezoid ABCD
structure IsoscelesTrapezoid where
  A : ℤ × ℤ
  B : ℤ × ℤ
  C : ℤ × ℤ
  D : ℤ × ℤ

-- Define the conditions
def validTrapezoid (t : IsoscelesTrapezoid) : Prop :=
  t.A = (15, 15) ∧
  t.D = (16, 20) ∧
  t.B.1 ≠ t.A.1 ∧ t.B.2 ≠ t.A.2 ∧  -- No horizontal or vertical sides
  t.C.1 ≠ t.D.1 ∧ t.C.2 ≠ t.D.2 ∧
  (t.B.2 - t.A.2) * (t.D.1 - t.C.1) = (t.B.1 - t.A.1) * (t.D.2 - t.C.2) ∧  -- AB || CD
  (t.C.2 - t.B.2) * (t.D.1 - t.A.1) ≠ (t.C.1 - t.B.1) * (t.D.2 - t.A.2) ∧  -- BC not || AD
  (t.D.2 - t.A.2) * (t.C.1 - t.B.1) ≠ (t.D.1 - t.A.1) * (t.C.2 - t.B.2)    -- CD not || AB

-- Define the slope of AB
def slopeAB (t : IsoscelesTrapezoid) : ℚ :=
  (t.B.2 - t.A.2) / (t.B.1 - t.A.1)

-- Define the theorem
theorem isosceles_trapezoid_slope_sum (t : IsoscelesTrapezoid) 
  (h : validTrapezoid t) : 
  ∃ (slopes : List ℚ), (∀ s ∈ slopes, ∃ t' : IsoscelesTrapezoid, validTrapezoid t' ∧ slopeAB t' = s) ∧
                       slopes.sum = 5 :=
sorry

end isosceles_trapezoid_slope_sum_l4015_401543


namespace symmetric_cubic_at_1_l4015_401597

/-- A cubic function f(x) = x³ + ax² + bx + 2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 2

/-- The function f is symmetric about the point (2,0) -/
def is_symmetric_about_2_0 (a b : ℝ) : Prop :=
  ∀ x : ℝ, f a b (x + 2) = f a b (2 - x)

theorem symmetric_cubic_at_1 (a b : ℝ) : 
  is_symmetric_about_2_0 a b → f a b 1 = 4 := by
sorry

end symmetric_cubic_at_1_l4015_401597


namespace baker_cakes_theorem_l4015_401529

/-- The number of cakes Baker made initially -/
def total_cakes : ℕ := 48

/-- The number of cakes Baker sold -/
def sold_cakes : ℕ := 44

/-- The number of cakes Baker has left -/
def remaining_cakes : ℕ := 4

/-- Theorem stating that the total number of cakes is equal to the sum of sold and remaining cakes -/
theorem baker_cakes_theorem : total_cakes = sold_cakes + remaining_cakes := by
  sorry

end baker_cakes_theorem_l4015_401529


namespace fuel_mixture_proof_l4015_401531

def tank_capacity : ℝ := 200
def ethanol_percentage_A : ℝ := 0.12
def ethanol_percentage_B : ℝ := 0.16
def total_ethanol : ℝ := 30

theorem fuel_mixture_proof (x : ℝ) 
  (hx : x ≥ 0 ∧ x ≤ 100) 
  (h_ethanol : ethanol_percentage_A * x + ethanol_percentage_B * (tank_capacity - x) = total_ethanol) :
  x = 50 := by
sorry

end fuel_mixture_proof_l4015_401531


namespace store_pants_price_l4015_401571

theorem store_pants_price (selling_price : ℝ) (price_difference : ℝ) (store_price : ℝ) : 
  selling_price = 34 →
  price_difference = 8 →
  store_price = selling_price - price_difference →
  store_price = 26 := by
sorry

end store_pants_price_l4015_401571


namespace second_month_sale_l4015_401539

def sale_first : ℕ := 6435
def sale_third : ℕ := 6855
def sale_fourth : ℕ := 7230
def sale_fifth : ℕ := 6562
def sale_sixth : ℕ := 7391
def average_sale : ℕ := 6900
def num_months : ℕ := 6

theorem second_month_sale :
  sale_first + sale_third + sale_fourth + sale_fifth + sale_sixth +
  (average_sale * num_months - (sale_first + sale_third + sale_fourth + sale_fifth + sale_sixth)) = 
  average_sale * num_months :=
by sorry

end second_month_sale_l4015_401539


namespace base_7_divisibility_l4015_401552

def is_base_7_digit (x : ℕ) : Prop := x ≤ 6

def base_7_to_decimal (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

theorem base_7_divisibility (x : ℕ) : 
  is_base_7_digit x → (base_7_to_decimal x) % 29 = 0 → x = 6 := by
  sorry

end base_7_divisibility_l4015_401552


namespace bruno_coconut_capacity_l4015_401513

theorem bruno_coconut_capacity (total_coconuts : ℕ) (barbie_capacity : ℕ) (total_trips : ℕ) 
  (h1 : total_coconuts = 144)
  (h2 : barbie_capacity = 4)
  (h3 : total_trips = 12) :
  (total_coconuts - barbie_capacity * total_trips) / total_trips = 8 := by
  sorry

end bruno_coconut_capacity_l4015_401513


namespace final_tree_count_l4015_401581

/-- The number of dogwood trees in the park after planting -/
def total_trees (initial : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  initial + planted_today + planted_tomorrow

/-- Theorem stating the final number of dogwood trees in the park -/
theorem final_tree_count :
  total_trees 7 5 4 = 16 := by
  sorry

end final_tree_count_l4015_401581


namespace necessary_not_sufficient_condition_l4015_401510

theorem necessary_not_sufficient_condition (x : ℝ) : 
  (∀ y : ℝ, y > 2 → y > 1) ∧ (∃ z : ℝ, z > 1 ∧ z ≤ 2) := by
  sorry

end necessary_not_sufficient_condition_l4015_401510


namespace expression_evaluation_l4015_401568

theorem expression_evaluation :
  let x : ℚ := -1/2
  3 * x^2 - (5*x - 3*(2*x - 1) + 7*x^2) = -9/2 := by
sorry

end expression_evaluation_l4015_401568


namespace same_type_monomials_result_l4015_401550

/-- 
Given two monomials of the same type: -x^3 * y^a and 6x^b * y,
prove that (a - b)^3 = -8
-/
theorem same_type_monomials_result (a b : ℤ) : 
  (∀ x y : ℝ, -x^3 * y^a = 6 * x^b * y) → (a - b)^3 = -8 := by
  sorry

end same_type_monomials_result_l4015_401550


namespace library_repacking_l4015_401527

theorem library_repacking (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) :
  initial_boxes = 1584 →
  books_per_initial_box = 45 →
  books_per_new_box = 47 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 28 := by
  sorry

end library_repacking_l4015_401527


namespace max_a_value_l4015_401556

theorem max_a_value (a b : ℕ) (h : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120) :
  a ≤ 20 ∧ ∃ b₀ : ℕ, 5 * Nat.lcm 20 b₀ + 2 * Nat.gcd 20 b₀ = 120 := by
  sorry

end max_a_value_l4015_401556


namespace distinct_primes_count_l4015_401507

theorem distinct_primes_count (n : ℕ) : n = 95 * 97 * 99 * 101 * 103 → 
  (Finset.card (Nat.factors n).toFinset) = 7 := by
sorry

end distinct_primes_count_l4015_401507


namespace seven_story_pagoda_top_lights_verify_total_lights_l4015_401546

/-- Represents a pagoda with a given number of stories and a total number of lights -/
structure Pagoda where
  stories : ℕ
  total_lights : ℕ
  lights_ratio : ℕ

/-- Calculates the number of lights at the top of the pagoda -/
def top_lights (p : Pagoda) : ℕ :=
  p.total_lights / (2^p.stories - 1)

/-- Theorem stating that a 7-story pagoda with 381 total lights and a doubling ratio has 3 lights at the top -/
theorem seven_story_pagoda_top_lights :
  let p := Pagoda.mk 7 381 2
  top_lights p = 3 := by
  sorry

/-- Verifies that the sum of lights across all stories equals the total lights -/
theorem verify_total_lights (p : Pagoda) :
  (top_lights p) * (2^p.stories - 1) = p.total_lights := by
  sorry

end seven_story_pagoda_top_lights_verify_total_lights_l4015_401546


namespace smallest_n_for_Q_less_than_threshold_l4015_401514

def Q (n : ℕ) : ℚ :=
  (3^(n-1) : ℚ) / ((3*n - 2 : ℕ).factorial * n.factorial)

theorem smallest_n_for_Q_less_than_threshold : 
  ∀ k : ℕ, k > 0 → Q k < 1/1500 ↔ k ≥ 10 :=
sorry

end smallest_n_for_Q_less_than_threshold_l4015_401514


namespace quadratic_function_properties_l4015_401506

def f (x : ℝ) : ℝ := x^2 + 22*x + 105

theorem quadratic_function_properties :
  (∀ x, f x = x^2 + 22*x + 105) ∧
  (∃ a b : ℤ, ∀ x, f x = x^2 + a*x + b) ∧
  (∃ r₁ r₂ r₃ r₄ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄ ∧
    f (f r₁) = 0 ∧ f (f r₂) = 0 ∧ f (f r₃) = 0 ∧ f (f r₄) = 0) ∧
  (∃ d : ℝ, d ≠ 0 ∧ ∃ r₁ r₂ r₃ r₄ : ℝ,
    f (f r₁) = 0 ∧ f (f r₂) = 0 ∧ f (f r₃) = 0 ∧ f (f r₄) = 0 ∧
    r₂ = r₁ + d ∧ r₃ = r₂ + d ∧ r₄ = r₃ + d) ∧
  (∀ g : ℝ → ℝ, (∃ a b : ℤ, ∀ x, g x = x^2 + a*x + b) →
    (∃ r₁ r₂ r₃ r₄ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄ ∧
      g (g r₁) = 0 ∧ g (g r₂) = 0 ∧ g (g r₃) = 0 ∧ g (g r₄) = 0) →
    (∃ d : ℝ, d ≠ 0 ∧ ∃ r₁ r₂ r₃ r₄ : ℝ,
      g (g r₁) = 0 ∧ g (g r₂) = 0 ∧ g (g r₃) = 0 ∧ g (g r₄) = 0 ∧
      r₂ = r₁ + d ∧ r₃ = r₂ + d ∧ r₄ = r₃ + d) →
    (1 + a + b ≥ 1 + 22 + 105)) :=
by sorry

end quadratic_function_properties_l4015_401506


namespace percent_democrats_voters_l4015_401537

theorem percent_democrats_voters (d r : ℝ) : 
  d + r = 100 →
  0.75 * d + 0.2 * r = 53 →
  d = 60 :=
by sorry

end percent_democrats_voters_l4015_401537


namespace equation_solution_l4015_401580

theorem equation_solution :
  ∀ N : ℚ, (5 + 6 + 7) / 3 = (2020 + 2021 + 2022) / N → N = 1010.5 := by
  sorry

end equation_solution_l4015_401580


namespace greatest_prime_divisor_plus_floor_sqrt_equality_l4015_401596

def greatest_prime_divisor (n : ℕ) : ℕ := sorry

def floor_sqrt (n : ℕ) : ℕ := sorry

theorem greatest_prime_divisor_plus_floor_sqrt_equality (n : ℕ) :
  n ≥ 2 →
  (greatest_prime_divisor n + floor_sqrt n = greatest_prime_divisor (n + 1) + floor_sqrt (n + 1)) ↔
  n = 3 := by sorry

end greatest_prime_divisor_plus_floor_sqrt_equality_l4015_401596


namespace triangle_perimeter_with_inscribed_circles_triangle_perimeter_l4015_401512

/-- Represents an equilateral triangle with inscribed circles -/
structure TriangleWithCircles where
  /-- The side length of the equilateral triangle -/
  side_length : ℝ
  /-- The radius of each inscribed circle -/
  circle_radius : ℝ
  /-- Assumption that the circles touch two sides of the triangle and each other -/
  circles_touch_sides_and_each_other : True

/-- Theorem stating the perimeter of the triangle given the inscribed circles -/
theorem triangle_perimeter_with_inscribed_circles
  (t : TriangleWithCircles)
  (h : t.circle_radius = 2) :
  t.side_length = 2 * Real.sqrt 3 + 4 :=
sorry

/-- Corollary calculating the perimeter of the triangle -/
theorem triangle_perimeter
  (t : TriangleWithCircles)
  (h : t.circle_radius = 2) :
  3 * t.side_length = 6 * Real.sqrt 3 + 12 :=
sorry

end triangle_perimeter_with_inscribed_circles_triangle_perimeter_l4015_401512


namespace odd_sum_not_divisible_by_three_l4015_401522

theorem odd_sum_not_divisible_by_three (x y z : ℕ) 
  (h_odd_x : Odd x) (h_odd_y : Odd y) (h_odd_z : Odd z)
  (h_positive_x : x > 0) (h_positive_y : y > 0) (h_positive_z : z > 0)
  (h_gcd : Nat.gcd x (Nat.gcd y z) = 1)
  (h_divisible : (x^2 + y^2 + z^2) % (x + y + z) = 0) :
  ¬(((x + y + z) - 2) % 3 = 0) := by
  sorry

end odd_sum_not_divisible_by_three_l4015_401522


namespace tan_two_fifths_pi_plus_theta_l4015_401595

theorem tan_two_fifths_pi_plus_theta (θ : ℝ) 
  (h : Real.sin ((12 / 5) * π + θ) + 2 * Real.sin ((11 / 10) * π - θ) = 0) : 
  Real.tan ((2 / 5) * π + θ) = 2 := by
  sorry

end tan_two_fifths_pi_plus_theta_l4015_401595


namespace absolute_difference_simplification_l4015_401526

theorem absolute_difference_simplification (a b : ℝ) 
  (ha : a < 0) (hab : a * b < 0) : 
  |a - b - 3| - |4 + b - a| = -1 := by
  sorry

end absolute_difference_simplification_l4015_401526


namespace product_remainder_l4015_401519

theorem product_remainder (a b c : ℕ) (ha : a % 10 = 3) (hb : b % 10 = 2) (hc : c % 10 = 4) :
  (a * b * c) % 10 = 4 := by
  sorry

end product_remainder_l4015_401519


namespace valid_arrangements_l4015_401592

/-- The number of ways to arrange 4 boys and 4 girls in a row with specific conditions -/
def arrangement_count : ℕ := 504

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 4

/-- The condition that adjacent individuals must be of opposite genders -/
def opposite_gender_adjacent : Prop := sorry

/-- The condition that a specific boy must stand next to a specific girl -/
def specific_pair_adjacent : Prop := sorry

/-- Theorem stating that the number of valid arrangements is 504 -/
theorem valid_arrangements :
  (num_boys = 4) →
  (num_girls = 4) →
  opposite_gender_adjacent →
  specific_pair_adjacent →
  arrangement_count = 504 := by
  sorry

end valid_arrangements_l4015_401592


namespace fraction_transformation_l4015_401582

theorem fraction_transformation (x : ℝ) (h : x ≠ 3) : -1 / (3 - x) = 1 / (x - 3) := by
  sorry

end fraction_transformation_l4015_401582
