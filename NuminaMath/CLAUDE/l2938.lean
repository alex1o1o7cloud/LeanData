import Mathlib

namespace sum_of_digits_in_special_number_l2938_293828

theorem sum_of_digits_in_special_number (A B C D E : ℕ) : 
  (A < 10) → (B < 10) → (C < 10) → (D < 10) → (E < 10) →
  A ≠ B → A ≠ C → A ≠ D → A ≠ E → 
  B ≠ C → B ≠ D → B ≠ E → 
  C ≠ D → C ≠ E → 
  D ≠ E →
  (100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E) % 9 = 0 →
  A + B + C + D + E = 9 := by
sorry

end sum_of_digits_in_special_number_l2938_293828


namespace p_or_q_is_true_l2938_293895

theorem p_or_q_is_true : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → Real.exp x ≥ 1) ∨ 
  (∃ x : ℝ, x^2 + x + 1 < 0) :=
by sorry

end p_or_q_is_true_l2938_293895


namespace actual_average_height_l2938_293848

/-- The number of boys in the class -/
def num_boys : ℕ := 50

/-- The initially calculated average height in cm -/
def initial_avg : ℝ := 183

/-- The incorrectly recorded height of the first boy in cm -/
def incorrect_height1 : ℝ := 166

/-- The actual height of the first boy in cm -/
def actual_height1 : ℝ := 106

/-- The incorrectly recorded height of the second boy in cm -/
def incorrect_height2 : ℝ := 175

/-- The actual height of the second boy in cm -/
def actual_height2 : ℝ := 190

/-- Conversion factor from cm to feet -/
def cm_to_feet : ℝ := 30.48

/-- Theorem stating that the actual average height of the boys is approximately 5.98 feet -/
theorem actual_average_height :
  let total_height := num_boys * initial_avg
  let corrected_total := total_height - (incorrect_height1 - actual_height1) + (actual_height2 - incorrect_height2)
  let actual_avg_cm := corrected_total / num_boys
  let actual_avg_feet := actual_avg_cm / cm_to_feet
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |actual_avg_feet - 5.98| < ε :=
by sorry

end actual_average_height_l2938_293848


namespace max_product_sum_300_l2938_293823

theorem max_product_sum_300 :
  (∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500) ∧
  (∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500) := by
  sorry

end max_product_sum_300_l2938_293823


namespace quadratic_function_theorem_l2938_293827

/-- A quadratic function f(x) = (x + a)(bx + 2a) where a, b ∈ ℝ, 
    which is an even function with range (-∞, 4] -/
def f (x a b : ℝ) : ℝ := (x + a) * (b * x + 2 * a)

/-- f is an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The range of f is (-∞, 4] -/
def has_range_neg_inf_to_4 (f : ℝ → ℝ) : Prop := 
  (∀ y, y ≤ 4 → ∃ x, f x = y) ∧ (∀ x, f x ≤ 4)

theorem quadratic_function_theorem (a b : ℝ) : 
  is_even_function (f · a b) → has_range_neg_inf_to_4 (f · a b) → 
  ∀ x, f x a b = -2 * x^2 + 4 := by sorry

end quadratic_function_theorem_l2938_293827


namespace no_solution_condition_l2938_293809

theorem no_solution_condition (b : ℝ) : 
  (∀ a x : ℝ, a > 1 → a^(2 - 2*x^2) + (b + 4)*a^(1 - x^2) + 3*b + 4 ≠ 0) ↔ 
  (0 < b ∧ b < 4) :=
sorry

end no_solution_condition_l2938_293809


namespace smallest_n_for_non_integer_expression_l2938_293844

theorem smallest_n_for_non_integer_expression : ∃ n : ℕ, n > 0 ∧ n = 11 ∧
  ∃ k : ℕ, k < n ∧
    (∀ a m : ℕ, a % n = k ∧ m > 0 →
      ¬(∃ z : ℤ, (a^m + 3^m : ℤ) = z * (a^2 - 3*a + 1))) ∧
    (∀ n' : ℕ, 0 < n' ∧ n' < n →
      ∀ k' : ℕ, k' < n' →
        ∃ a m : ℕ, a % n' = k' ∧ m > 0 ∧
          ∃ z : ℤ, (a^m + 3^m : ℤ) = z * (a^2 - 3*a + 1)) :=
by
  sorry


end smallest_n_for_non_integer_expression_l2938_293844


namespace bugs_eating_flowers_l2938_293858

/-- Given 7 bugs, each eating 4 flowers, prove that the total number of flowers eaten is 28. -/
theorem bugs_eating_flowers :
  let number_of_bugs : ℕ := 7
  let flowers_per_bug : ℕ := 4
  number_of_bugs * flowers_per_bug = 28 := by
  sorry

end bugs_eating_flowers_l2938_293858


namespace bobby_candy_problem_l2938_293899

theorem bobby_candy_problem (x : ℕ) : 
  (x - 5 - 9 = 7) → x = 21 := by
  sorry

end bobby_candy_problem_l2938_293899


namespace lemonade_glasses_l2938_293889

/-- Calculates the total number of glasses of lemonade that can be served -/
def total_glasses (glasses_per_pitcher : ℕ) (num_pitchers : ℕ) : ℕ :=
  glasses_per_pitcher * num_pitchers

/-- Theorem: Given 5 glasses per pitcher and 6 pitchers, the total glasses served is 30 -/
theorem lemonade_glasses : total_glasses 5 6 = 30 := by
  sorry

end lemonade_glasses_l2938_293889


namespace parabola_intersection_difference_l2938_293822

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6
def g (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

-- Define the set of intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Theorem statement
theorem parabola_intersection_difference :
  ∃ (a c : ℝ), a ∈ intersection_points ∧ c ∈ intersection_points ∧ c ≥ a ∧ c - a = 2/5 :=
sorry

end parabola_intersection_difference_l2938_293822


namespace arithmetic_sequence_problem_geometric_sequence_problem_l2938_293853

-- Arithmetic Sequence
theorem arithmetic_sequence_problem (d n a_n : ℤ) (h1 : d = 2) (h2 : n = 15) (h3 : a_n = -10) :
  let a_1 := a_n - (n - 1) * d
  let S_n := n * (a_1 + a_n) / 2
  a_1 = -38 ∧ S_n = -360 := by sorry

-- Geometric Sequence
theorem geometric_sequence_problem (a_2 a_3 a_4 : ℚ) (h1 : a_2 + a_3 = 6) (h2 : a_3 + a_4 = 12) :
  let q := a_3 / a_2
  let a_1 := a_2 / q
  let S_10 := a_1 * (1 - q^10) / (1 - q)
  q = 2 ∧ S_10 = 1023 := by sorry

end arithmetic_sequence_problem_geometric_sequence_problem_l2938_293853


namespace subset_implies_m_values_l2938_293805

def P : Set ℝ := {x | x^2 = 1}
def Q (m : ℝ) : Set ℝ := {x | m * x = 1}

theorem subset_implies_m_values (m : ℝ) : Q m ⊆ P → m = 0 ∨ m = 1 ∨ m = -1 := by
  sorry

end subset_implies_m_values_l2938_293805


namespace josie_cabinet_unlock_time_l2938_293881

/-- Represents the shopping trip details -/
structure ShoppingTrip where
  total_time : ℕ
  shopping_time : ℕ
  cart_wait_time : ℕ
  restocking_wait_time : ℕ
  checkout_wait_time : ℕ

/-- Calculates the time spent waiting for the cabinet to be unlocked -/
def cabinet_unlock_time (trip : ShoppingTrip) : ℕ :=
  trip.total_time - trip.shopping_time - trip.cart_wait_time - trip.restocking_wait_time - trip.checkout_wait_time

/-- Theorem stating that Josie waited 13 minutes for the cabinet to be unlocked -/
theorem josie_cabinet_unlock_time :
  let trip := ShoppingTrip.mk 90 42 3 14 18
  cabinet_unlock_time trip = 13 := by sorry

end josie_cabinet_unlock_time_l2938_293881


namespace gcd_14m_21n_l2938_293846

theorem gcd_14m_21n (m n : ℕ+) (h : Nat.gcd m n = 18) : Nat.gcd (14 * m) (21 * n) = 126 := by
  sorry

end gcd_14m_21n_l2938_293846


namespace product_of_digits_3545_l2938_293847

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def units_digit (n : ℕ) : ℕ :=
  n % 10

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem product_of_digits_3545 :
  let n := 3545
  ¬ is_divisible_by_3 n ∧ units_digit n * tens_digit n = 20 :=
by sorry

end product_of_digits_3545_l2938_293847


namespace cos_squared_difference_eq_sqrt_three_half_l2938_293815

theorem cos_squared_difference_eq_sqrt_three_half :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end cos_squared_difference_eq_sqrt_three_half_l2938_293815


namespace solve_quadratic_equation_l2938_293802

theorem solve_quadratic_equation (x : ℝ) :
  2 * (x - 3)^2 - 98 = 0 → x = 10 ∨ x = -4 := by
  sorry

end solve_quadratic_equation_l2938_293802


namespace ceiling_neg_sqrt_eight_l2938_293893

theorem ceiling_neg_sqrt_eight : ⌈-Real.sqrt 8⌉ = -2 := by
  sorry

end ceiling_neg_sqrt_eight_l2938_293893


namespace equation_solution_l2938_293872

-- Define the function f
def f (x : ℝ) : ℝ := x + 4

-- State the theorem
theorem equation_solution :
  ∃ (x : ℝ), (3 * f (x - 2)) / f 0 + 4 = f (2 * x + 1) ∧ x = 2 / 5 := by
  sorry

end equation_solution_l2938_293872


namespace octal_536_to_base7_l2938_293864

def octal_to_decimal (n : ℕ) : ℕ :=
  5 * 8^2 + 3 * 8^1 + 6 * 8^0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  [1, 0, 1, 0]

theorem octal_536_to_base7 :
  decimal_to_base7 (octal_to_decimal 536) = [1, 0, 1, 0] := by
  sorry

end octal_536_to_base7_l2938_293864


namespace arithmetic_sequence_property_l2938_293877

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₁ + a₉ = 10, prove that a₅ = 5 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 9 = 10) : 
  a 5 = 5 := by
sorry

end arithmetic_sequence_property_l2938_293877


namespace hyperbola_minimum_value_l2938_293836

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

-- Define the slope of the asymptote
def AsymptopeSlope (a b : ℝ) : Prop :=
  b / a = Real.sqrt 3

-- Define the eccentricity
def Eccentricity (a b : ℝ) (e : ℝ) : Prop :=
  e = Real.sqrt (1 + b^2 / a^2)

-- State the theorem
theorem hyperbola_minimum_value (a b e : ℝ) :
  Hyperbola a b →
  AsymptopeSlope a b →
  Eccentricity a b e →
  a > 0 →
  b > 0 →
  (∀ a' b' e' : ℝ, Hyperbola a' b' → AsymptopeSlope a' b' → Eccentricity a' b' e' →
    a' > 0 → b' > 0 → (a'^2 + e') / b' ≥ (a^2 + e) / b) →
  (a^2 + e) / b = 2 * Real.sqrt 6 / 3 :=
sorry

end hyperbola_minimum_value_l2938_293836


namespace intersection_line_equation_l2938_293833

/-- Given two lines that intersect at (2, 3), prove that the line passing through
    the points defined by their coefficients has the equation 2x + 3y + 1 = 0 -/
theorem intersection_line_equation 
  (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : a₁ * 2 + b₁ * 3 + 1 = 0)
  (h₂ : a₂ * 2 + b₂ * 3 + 1 = 0) :
  ∃ (k : ℝ), k ≠ 0 ∧ 2 * a₁ + 3 * b₁ + k = 0 ∧ 2 * a₂ + 3 * b₂ + k = 0 :=
sorry

end intersection_line_equation_l2938_293833


namespace greatest_base12_divisible_by_7_l2938_293867

/-- Converts a base 12 number to decimal --/
def base12ToDecimal (a b c : Nat) : Nat :=
  a * 12^2 + b * 12 + c

/-- Checks if a number is divisible by 7 --/
def isDivisibleBy7 (n : Nat) : Prop :=
  n % 7 = 0

/-- Theorem: BB6₁₂ is the greatest 3-digit base 12 positive integer divisible by 7 --/
theorem greatest_base12_divisible_by_7 :
  let bb6 := base12ToDecimal 11 11 6
  isDivisibleBy7 bb6 ∧
  ∀ n, n > bb6 → n ≤ base12ToDecimal 11 11 11 →
    ¬(isDivisibleBy7 n) :=
by sorry

end greatest_base12_divisible_by_7_l2938_293867


namespace finish_time_l2938_293856

/-- Time (in days) for A and B to finish a can of coffee together -/
def coffee_together : ℝ := 10

/-- Time (in days) for A to finish a can of coffee alone -/
def coffee_A : ℝ := 12

/-- Time (in days) for A and B to finish a pound of tea together -/
def tea_together : ℝ := 12

/-- Time (in days) for B to finish a pound of tea alone -/
def tea_B : ℝ := 20

/-- A won't drink coffee if there's tea, and B won't drink tea if there's coffee -/
axiom preference : True

/-- The time it takes for A and B to finish a pound of tea and a can of coffee -/
def total_time : ℝ := 35

theorem finish_time : total_time = 35 := by sorry

end finish_time_l2938_293856


namespace absent_fraction_l2938_293840

theorem absent_fraction (total : ℕ) (present : ℕ) 
  (h1 : total = 28) 
  (h2 : present = 20) : 
  (total - present : ℚ) / total = 2 / 7 := by
  sorry

end absent_fraction_l2938_293840


namespace sum_of_a_and_b_is_twelve_l2938_293852

/-- Given a function y of x, prove that a + b = 12 -/
theorem sum_of_a_and_b_is_twelve 
  (y : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, y x = a + b / (x + 1))
  (h2 : y (-2) = 2)
  (h3 : y (-6) = 6) :
  a + b = 12 := by
  sorry

end sum_of_a_and_b_is_twelve_l2938_293852


namespace vector_magnitude_direction_comparison_l2938_293816

theorem vector_magnitude_direction_comparison
  (a b : ℝ × ℝ)
  (h1 : a ≠ (0, 0))
  (h2 : b ≠ (0, 0))
  (h3 : ∃ (k : ℝ), k > 0 ∧ a = k • b)
  (h4 : ‖a‖ > ‖b‖) :
  ¬ (∀ (x y : ℝ × ℝ), (∃ (k : ℝ), k > 0 ∧ x = k • y) → x > y) :=
by sorry

end vector_magnitude_direction_comparison_l2938_293816


namespace triangle_third_side_length_l2938_293806

theorem triangle_third_side_length (a b c : ℕ) : 
  a = 2 → b = 14 → c % 2 = 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (b - a < c ∧ c - a < b ∧ c - b < a) →
  c = 14 := by sorry

end triangle_third_side_length_l2938_293806


namespace rectangular_field_perimeter_l2938_293854

theorem rectangular_field_perimeter
  (area : ℝ) (width : ℝ) (h_area : area = 300) (h_width : width = 15) :
  2 * (area / width + width) = 70 :=
by sorry

end rectangular_field_perimeter_l2938_293854


namespace curve_intersection_perpendicular_l2938_293871

/-- The curve C: x^2 + y^2 - 2x - 4y + m = 0 -/
def curve_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

/-- The line: x + 2y - 3 = 0 -/
def line (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem curve_intersection_perpendicular (m : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    curve_C x1 y1 m ∧ curve_C x2 y2 m ∧
    line x1 y1 ∧ line x2 y2 ∧
    perpendicular x1 y1 x2 y2) →
  m = 12/5 := by sorry

end curve_intersection_perpendicular_l2938_293871


namespace sum_and_product_membership_l2938_293880

def P : Set ℤ := {x | ∃ k, x = 2 * k - 1}
def Q : Set ℤ := {y | ∃ n, y = 2 * n}

theorem sum_and_product_membership (x y : ℤ) (hx : x ∈ P) (hy : y ∈ Q) :
  (x + y) ∈ P ∧ (x * y) ∈ Q := by
  sorry

end sum_and_product_membership_l2938_293880


namespace count_with_3_or_7_l2938_293841

/-- The set of digits that are neither 3 nor 7 -/
def other_digits : Finset Nat := {0, 1, 2, 4, 5, 6, 8, 9}

/-- The set of non-zero digits that are neither 3 nor 7 -/
def non_zero_other_digits : Finset Nat := {1, 2, 4, 5, 6, 8, 9}

/-- The count of four-digit numbers without 3 or 7 -/
def count_without_3_or_7 : Nat :=
  (Finset.card non_zero_other_digits) * (Finset.card other_digits)^3

/-- The total count of four-digit numbers -/
def total_four_digit_numbers : Nat := 9000

theorem count_with_3_or_7 :
  total_four_digit_numbers - count_without_3_or_7 = 5416 := by
  sorry

end count_with_3_or_7_l2938_293841


namespace direct_proportion_constant_zero_l2938_293862

/-- A function f : ℝ → ℝ is a direct proportion function if there exists a non-zero constant k such that f(x) = k * x for all x -/
def IsDirectProportionFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- Given that y = x + b is a direct proportion function, b must be zero -/
theorem direct_proportion_constant_zero (b : ℝ) :
  IsDirectProportionFunction (fun x ↦ x + b) → b = 0 := by
  sorry


end direct_proportion_constant_zero_l2938_293862


namespace first_discount_percentage_l2938_293842

/-- Proves that given a shirt with a list price of 150, a final price of 105 after two successive discounts, and a second discount of 12.5%, the first discount percentage is 20%. -/
theorem first_discount_percentage 
  (list_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) 
  (h1 : list_price = 150)
  (h2 : final_price = 105)
  (h3 : second_discount = 12.5) :
  ∃ (first_discount : ℝ),
    first_discount = 20 ∧ 
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) := by
  sorry

end first_discount_percentage_l2938_293842


namespace cos_inequality_range_l2938_293876

theorem cos_inequality_range (θ : Real) : 
  θ ∈ Set.Icc (-Real.pi) Real.pi →
  (3 * Real.sqrt 2 * Real.cos (θ + Real.pi / 4) < 4 * (Real.sin θ ^ 3 - Real.cos θ ^ 3)) ↔
  θ ∈ Set.Ioc (-Real.pi) (-3 * Real.pi / 4) ∪ Set.Ioo (Real.pi / 4) Real.pi :=
by sorry

end cos_inequality_range_l2938_293876


namespace point_d_is_multiple_of_fifteen_l2938_293831

/-- Represents a point on the number line -/
structure Point where
  value : ℤ

/-- Represents the number line with four special points -/
structure NumberLine where
  w : Point
  x : Point
  y : Point
  z : Point
  consecutive : w.value < x.value ∧ x.value < y.value ∧ y.value < z.value
  multiples_of_three : (w.value % 3 = 0 ∧ y.value % 3 = 0) ∨ (x.value % 3 = 0 ∧ z.value % 3 = 0)
  multiples_of_five : (w.value % 5 = 0 ∧ z.value % 5 = 0) ∨ (x.value % 5 = 0 ∧ y.value % 5 = 0)

/-- The point D, which is 5 units away from one of the multiples of 5 -/
def point_d (nl : NumberLine) : Point :=
  if nl.w.value % 5 = 0 then { value := nl.w.value + 5 }
  else if nl.x.value % 5 = 0 then { value := nl.x.value + 5 }
  else if nl.y.value % 5 = 0 then { value := nl.y.value + 5 }
  else { value := nl.z.value + 5 }

theorem point_d_is_multiple_of_fifteen (nl : NumberLine) :
  (point_d nl).value % 15 = 0 := by
  sorry

end point_d_is_multiple_of_fifteen_l2938_293831


namespace colby_mango_harvest_l2938_293857

theorem colby_mango_harvest (x : ℝ) :
  (x ≥ 0) →                             -- Non-negative harvest
  (x - 20 ≥ 0) →                        -- Enough to sell 20 kg to market
  (8 * ((x - 20) / 2) = 160) →          -- 160 mangoes left after sales
  (x = 60) :=
by
  sorry

end colby_mango_harvest_l2938_293857


namespace alicia_remaining_art_l2938_293811

/-- Represents the types of art in Alicia's collection -/
inductive ArtType
  | Medieval
  | Renaissance
  | Modern

/-- Calculates the remaining art pieces after donation -/
def remaining_art (initial : Nat) (donate_percent : Nat) : Nat :=
  initial - (initial * donate_percent / 100)

/-- Theorem stating the remaining art pieces after Alicia's donations -/
theorem alicia_remaining_art :
  (remaining_art 70 65 = 25) ∧
  (remaining_art 120 30 = 84) ∧
  (remaining_art 150 45 = 83) := by
  sorry

#check alicia_remaining_art

end alicia_remaining_art_l2938_293811


namespace unique_sequence_exists_l2938_293879

def sequence_property (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n : ℕ, n ≥ 1 → a n * a (n + 2) = (a (n + 1))^3 + 1

theorem unique_sequence_exists : ∃! a : ℕ → ℕ, sequence_property a := by
  sorry

end unique_sequence_exists_l2938_293879


namespace rectangle_dimension_change_l2938_293875

theorem rectangle_dimension_change (original_length original_width : ℝ) 
  (h_positive_length : original_length > 0)
  (h_positive_width : original_width > 0) :
  let new_length := 1.4 * original_length
  let new_width := original_width * (1 - 0.2857)
  new_length * new_width = original_length * original_width := by
  sorry

end rectangle_dimension_change_l2938_293875


namespace bob_garden_area_l2938_293863

/-- Calculates the area of a garden given property dimensions and garden proportions. -/
def garden_area (property_width property_length : ℝ) (garden_width_ratio garden_length_ratio : ℝ) : ℝ :=
  (property_width * garden_width_ratio) * (property_length * garden_length_ratio)

/-- Theorem stating that Bob's garden area is 28125 square feet. -/
theorem bob_garden_area :
  garden_area 1000 2250 (1/8) (1/10) = 28125 := by
  sorry

end bob_garden_area_l2938_293863


namespace man_son_age_ratio_l2938_293886

/-- Represents the age ratio between a man and his son after two years -/
def age_ratio (son_age : ℕ) (age_difference : ℕ) : ℚ :=
  let man_age := son_age + age_difference
  (man_age + 2) / (son_age + 2)

/-- Theorem stating the age ratio between a man and his son after two years -/
theorem man_son_age_ratio :
  age_ratio 22 24 = 2 := by
  sorry

end man_son_age_ratio_l2938_293886


namespace function_symmetry_l2938_293870

/-- Given a function f and a real number a, proves that if f(x) = |x|(e^(ax) - e^(-ax)) + 2 
    and f(10) = 1, then f(-10) = 3 -/
theorem function_symmetry (f : ℝ → ℝ) (a : ℝ) 
    (h1 : ∀ x, f x = |x| * (Real.exp (a * x) - Real.exp (-a * x)) + 2)
    (h2 : f 10 = 1) : 
  f (-10) = 3 := by
  sorry

end function_symmetry_l2938_293870


namespace waiter_customers_theorem_l2938_293868

/-- The number of initial customers that satisfies the given condition -/
def initial_customers : ℕ := 33

/-- The condition given in the problem -/
theorem waiter_customers_theorem : 
  (initial_customers - 31 + 26 = 28) := by sorry

end waiter_customers_theorem_l2938_293868


namespace function_existence_condition_l2938_293843

theorem function_existence_condition (k a : ℕ) :
  (∃ f : ℕ → ℕ, ∀ n, (f^[k] n = n + a)) ↔ (a ≥ 0 ∧ k ∣ a) :=
by sorry

end function_existence_condition_l2938_293843


namespace students_walking_home_l2938_293890

theorem students_walking_home (bus auto bike scooter : ℚ)
  (h_bus : bus = 2/5)
  (h_auto : auto = 1/5)
  (h_bike : bike = 1/10)
  (h_scooter : scooter = 1/10)
  : 1 - (bus + auto + bike + scooter) = 1/5 := by
  sorry

end students_walking_home_l2938_293890


namespace inverse_proportion_value_l2938_293810

/-- For the inverse proportion function y = -8/x, when x = -2, y = 4 -/
theorem inverse_proportion_value : 
  let f : ℝ → ℝ := λ x => -8 / x
  f (-2) = 4 := by sorry

end inverse_proportion_value_l2938_293810


namespace unique_solution_factorial_equation_l2938_293860

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem unique_solution_factorial_equation :
  ∃! (k n : ℕ), factorial n + 3 * n + 8 = k^2 ∧ k = 4 ∧ n = 2 := by
  sorry

end unique_solution_factorial_equation_l2938_293860


namespace cotton_amount_l2938_293892

/-- Given:
  * Kevin plants corn and cotton
  * He harvests 30 pounds of corn and x pounds of cotton
  * Corn sells for $5 per pound
  * Cotton sells for $10 per pound
  * Total revenue from selling all corn and cotton is $640
Prove that x = 49 -/
theorem cotton_amount (x : ℝ) : 
  (30 * 5 + x * 10 = 640) → x = 49 := by
  sorry

end cotton_amount_l2938_293892


namespace isosceles_trapezoid_rotation_l2938_293834

/-- An isosceles trapezoid with specific dimensions -/
structure IsoscelesTrapezoid where
  smallBase : ℝ
  largeBase : ℝ
  acuteAngle : ℝ
  heightPerp : ℝ

/-- A solid of revolution formed by rotating an isosceles trapezoid around its smaller base -/
structure SolidOfRevolution where
  trapezoid : IsoscelesTrapezoid
  surfaceArea : ℝ
  volume : ℝ

/-- The theorem stating the surface area and volume of the solid of revolution -/
theorem isosceles_trapezoid_rotation (t : IsoscelesTrapezoid) 
  (h1 : t.smallBase = 2)
  (h2 : t.largeBase = 3)
  (h3 : t.acuteAngle = π / 3)
  (h4 : t.heightPerp = 3) :
  ∃ (s : SolidOfRevolution), 
    s.trapezoid = t ∧ 
    s.surfaceArea = 4 * π * Real.sqrt 3 ∧ 
    s.volume = 2 * π := by
  sorry

end isosceles_trapezoid_rotation_l2938_293834


namespace min_value_theorem_l2938_293888

def f (a x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

theorem min_value_theorem (a : ℝ) : 
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f a x ≤ f a y) ∧ 
  (∀ x ∈ Set.Icc 0 1, f a x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc 0 1, f a x = 2) ↔ 
  a = 0 ∨ a = 3 + Real.sqrt 5 :=
sorry

end min_value_theorem_l2938_293888


namespace line_through_point_l2938_293866

/-- Proves that the value of k is -10 for a line passing through (-1/3, -2) --/
theorem line_through_point (k : ℝ) : 
  (2 - 3 * k * (-1/3) = 4 * (-2)) → k = -10 := by
  sorry

end line_through_point_l2938_293866


namespace no_quadratic_term_implies_m_value_l2938_293851

theorem no_quadratic_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, (x + m) * (x^2 + 2*x - 1) = a*x^3 + b*x + c) → m = -2 := by
  sorry

end no_quadratic_term_implies_m_value_l2938_293851


namespace bridget_middle_score_l2938_293832

/-- Represents the test scores of the four students -/
structure Scores where
  hannah : ℝ
  ella : ℝ
  cassie : ℝ
  bridget : ℝ

/-- Defines the conditions given in the problem -/
def SatisfiesConditions (s : Scores) : Prop :=
  (s.cassie > s.hannah) ∧ (s.cassie > s.ella) ∧
  (s.bridget ≥ s.hannah) ∧ (s.bridget ≥ s.ella)

/-- Defines what it means for a student to have the middle score -/
def HasMiddleScore (name : String) (s : Scores) : Prop :=
  match name with
  | "Bridget" => (s.bridget > min s.hannah s.ella) ∧ (s.bridget < max s.cassie s.ella)
  | _ => False

/-- The main theorem stating that if the conditions are satisfied, Bridget must have the middle score -/
theorem bridget_middle_score (s : Scores) :
  SatisfiesConditions s → HasMiddleScore "Bridget" s := by
  sorry


end bridget_middle_score_l2938_293832


namespace max_k_inequality_l2938_293897

theorem max_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_condition : a^2 + b^2 + c^2 = 2*(a*b + b*c + c*a)) :
  ∃ (k : ℝ), k > 0 ∧ k = 2 ∧
  ∀ (k' : ℝ), k' > 0 →
    (1 / (k'*a*b + c^2) + 1 / (k'*b*c + a^2) + 1 / (k'*c*a + b^2) ≥ (k' + 3) / (a^2 + b^2 + c^2)) →
    k' ≤ k :=
by sorry

end max_k_inequality_l2938_293897


namespace sum_of_positive_reals_l2938_293873

theorem sum_of_positive_reals (p q r s : ℝ) : 
  0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s →
  p^2 + q^2 = 2500 →
  r^2 + s^2 = 2500 →
  p * r = 1200 →
  q * s = 1200 →
  p + q + r + s = 140 := by
sorry

end sum_of_positive_reals_l2938_293873


namespace sports_club_overlap_l2938_293807

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h1 : total = 28)
  (h2 : badminton = 17)
  (h3 : tennis = 19)
  (h4 : neither = 2)
  : badminton + tennis - (total - neither) = 10 := by
  sorry

end sports_club_overlap_l2938_293807


namespace number_division_problem_l2938_293826

theorem number_division_problem : ∃ (n : ℕ), 
  n = 220025 ∧ 
  (n / (555 + 445) = 2 * (555 - 445)) ∧ 
  (n % (555 + 445) = 25) := by
  sorry

end number_division_problem_l2938_293826


namespace power_relation_l2938_293819

theorem power_relation (a b : ℕ) : 2^a = 8^(b+1) → 3^a / 27^b = 27 := by
  sorry

end power_relation_l2938_293819


namespace crew_diff_1000_tons_crew_estimate_min_tonnage_crew_estimate_max_tonnage_l2938_293861

-- Define the regression equation
def crew_estimate (tonnage : ℝ) : ℝ := 9.5 + 0.0062 * tonnage

-- Define the tonnage range
def min_tonnage : ℝ := 192
def max_tonnage : ℝ := 3246

-- Theorem 1: Difference in crew members for 1000 tons difference
theorem crew_diff_1000_tons : 
  ∀ (x : ℝ), crew_estimate (x + 1000) - crew_estimate x = 6 := by sorry

-- Theorem 2: Estimated crew for minimum tonnage
theorem crew_estimate_min_tonnage : 
  ⌊crew_estimate min_tonnage⌋ = 11 := by sorry

-- Theorem 3: Estimated crew for maximum tonnage
theorem crew_estimate_max_tonnage : 
  ⌊crew_estimate max_tonnage⌋ = 30 := by sorry

end crew_diff_1000_tons_crew_estimate_min_tonnage_crew_estimate_max_tonnage_l2938_293861


namespace triangle_coverage_theorem_l2938_293817

-- Define the original equilateral triangle
structure EquilateralTriangle where
  area : ℝ
  isEquilateral : Bool

-- Define a point inside the triangle
structure Point where
  x : ℝ
  y : ℝ
  insideTriangle : Bool

-- Define the smaller equilateral triangles
structure SmallerTriangle where
  area : ℝ
  sidesParallel : Bool
  containsPoint : Point → Bool

-- Main theorem
theorem triangle_coverage_theorem 
  (original : EquilateralTriangle)
  (points : Finset Point)
  (h1 : original.area = 1)
  (h2 : original.isEquilateral = true)
  (h3 : points.card = 5)
  (h4 : ∀ p ∈ points, p.insideTriangle = true) :
  ∃ (t1 t2 t3 : SmallerTriangle),
    (t1.sidesParallel = true ∧ t2.sidesParallel = true ∧ t3.sidesParallel = true) ∧
    (t1.area + t2.area + t3.area ≤ 0.64) ∧
    (∀ p ∈ points, t1.containsPoint p = true ∨ t2.containsPoint p = true ∨ t3.containsPoint p = true) :=
by sorry

end triangle_coverage_theorem_l2938_293817


namespace concentric_circles_ratio_l2938_293874

theorem concentric_circles_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : π * b^2 - π * a^2 = 4 * (π * a^2)) : a / b = Real.sqrt 5 / 5 := by
  sorry

end concentric_circles_ratio_l2938_293874


namespace fraction_problem_l2938_293887

theorem fraction_problem (a b : ℚ) : 
  b / (a - 2) = 3 / 4 →
  b / (a + 9) = 5 / 7 →
  b / a = 165 / 222 :=
by sorry

end fraction_problem_l2938_293887


namespace janet_change_l2938_293855

def muffin_price : ℚ := 75 / 100
def num_muffins : ℕ := 12
def amount_paid : ℚ := 20

theorem janet_change :
  amount_paid - (num_muffins : ℚ) * muffin_price = 11 := by
  sorry

end janet_change_l2938_293855


namespace min_value_expression_equality_condition_l2938_293878

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (16 / x) + (108 / y) + x * y ≥ 36 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ x y, (16 / x) + (108 / y) + x * y = 36 :=
by sorry

end min_value_expression_equality_condition_l2938_293878


namespace min_perimeter_isosceles_triangles_l2938_293838

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℚ :=
  (t.base : ℚ) * (((t.leg : ℚ)^2 - ((t.base : ℚ) / 2)^2).sqrt) / 2

/-- Theorem: Minimum perimeter of two noncongruent isosceles triangles with same area and base ratio 9:8 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    9 * t1.base = 8 * t2.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      area s1 = area s2 →
      9 * s1.base = 8 * s2.base →
      perimeter t1 ≤ perimeter s1 :=
by sorry

#eval perimeter { leg := 90, base := 144 } -- Expected output: 324

end min_perimeter_isosceles_triangles_l2938_293838


namespace factor_x_squared_minus_196_l2938_293830

theorem factor_x_squared_minus_196 (x : ℝ) : x^2 - 196 = (x - 14) * (x + 14) := by
  sorry

end factor_x_squared_minus_196_l2938_293830


namespace starters_count_l2938_293884

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of twins
def num_twins : ℕ := 2

-- Define the number of triplets
def num_triplets : ℕ := 3

-- Define the number of starters to choose
def num_starters : ℕ := 7

-- Define the function to calculate the number of ways to choose starters
def choose_starters (total : ℕ) (twins : ℕ) (triplets : ℕ) (starters : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

-- Theorem statement
theorem starters_count :
  choose_starters total_players num_twins num_triplets num_starters = 5148 := by
  sorry

end starters_count_l2938_293884


namespace expected_balls_theorem_l2938_293812

/-- Represents a system of balls arranged in a circle -/
structure BallSystem :=
  (n : ℕ)  -- number of balls

/-- Represents a swap operation on the ball system -/
structure SwapOperation :=
  (isAdjacent : Bool)  -- whether the swap is between adjacent balls only

/-- Calculates the probability of a ball remaining in its original position after a swap -/
def probabilityAfterSwap (sys : BallSystem) (op : SwapOperation) : ℚ :=
  if op.isAdjacent then
    (sys.n - 2 : ℚ) / sys.n * 2 / 3 + 2 / sys.n
  else
    (sys.n - 2 : ℚ) / sys.n

/-- Calculates the expected number of balls in their original positions after two swaps -/
def expectedBallsInOriginalPosition (sys : BallSystem) (op1 op2 : SwapOperation) : ℚ :=
  sys.n * probabilityAfterSwap sys op1 * probabilityAfterSwap sys op2

theorem expected_balls_theorem (sys : BallSystem) (op1 op2 : SwapOperation) :
  sys.n = 8 ∧ ¬op1.isAdjacent ∧ op2.isAdjacent →
  expectedBallsInOriginalPosition sys op1 op2 = 2 := by
  sorry

#eval expectedBallsInOriginalPosition ⟨8⟩ ⟨false⟩ ⟨true⟩

end expected_balls_theorem_l2938_293812


namespace power_of_negative_one_difference_l2938_293891

theorem power_of_negative_one_difference : (-1)^2004 - (-1)^2003 = 2 := by
  sorry

end power_of_negative_one_difference_l2938_293891


namespace max_min_difference_is_16_l2938_293845

def f (x : ℝ) := |x - 1| + |x - 2| + |x - 3|

theorem max_min_difference_is_16 :
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc (-4 : ℝ) 4 ∧ x_min ∈ Set.Icc (-4 : ℝ) 4 ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, f x ≤ f x_max) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, f x_min ≤ f x) ∧
  f x_max - f x_min = 16 :=
sorry

end max_min_difference_is_16_l2938_293845


namespace fish_count_ratio_l2938_293824

/-- The ratio of fish counted on day 2 to fish counted on day 1 -/
theorem fish_count_ratio : 
  ∀ (fish_day1 fish_day2 sharks_total : ℕ) 
    (shark_percentage : ℚ),
  fish_day1 = 15 →
  sharks_total = 15 →
  shark_percentage = 1/4 →
  (↑fish_day1 * shark_percentage).floor + 
    (↑fish_day2 * shark_percentage).floor = sharks_total →
  fish_day2 / fish_day1 = 16/5 := by
sorry

end fish_count_ratio_l2938_293824


namespace bottle_cap_distribution_l2938_293882

theorem bottle_cap_distribution (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) : 
  total_caps = 12 → num_groups = 6 → caps_per_group = total_caps / num_groups → caps_per_group = 2 := by
  sorry

#check bottle_cap_distribution

end bottle_cap_distribution_l2938_293882


namespace equation_is_circle_l2938_293869

/-- A conic section type -/
inductive ConicSection
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines if an equation represents a circle -/
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ h k r, ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The given equation -/
def equation (x y : ℝ) : Prop :=
  (x - 3)^2 = -(3*y + 1)^2 + 45

theorem equation_is_circle :
  is_circle equation :=
sorry

end equation_is_circle_l2938_293869


namespace complex_sum_simplification_l2938_293883

theorem complex_sum_simplification :
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  z₁^12 + z₂^12 = 2 := by
sorry

end complex_sum_simplification_l2938_293883


namespace fencing_cost_calculation_l2938_293821

-- Define the plot dimensions
def length : ℝ := 60
def breadth : ℝ := 40

-- Define the cost per meter of fencing
def cost_per_meter : ℝ := 26.50

-- Calculate the perimeter
def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost
def total_cost : ℝ := perimeter * cost_per_meter

-- Theorem to prove
theorem fencing_cost_calculation :
  total_cost = 5300 :=
by sorry

end fencing_cost_calculation_l2938_293821


namespace data_grouping_l2938_293835

theorem data_grouping (data : Set ℤ) (max_val min_val class_interval : ℤ) :
  max_val = 42 →
  min_val = 8 →
  class_interval = 5 →
  ∀ x ∈ data, min_val ≤ x ∧ x ≤ max_val →
  (max_val - min_val) / class_interval + 1 = 7 :=
by sorry

end data_grouping_l2938_293835


namespace expression_simplification_l2938_293865

theorem expression_simplification (a b : ℝ) (h : a * b ≠ 0) :
  (3 * a^3 * b - 12 * a^2 * b^2 - 6 * a * b^3) / (-3 * a * b) - 4 * a * b = -a^2 + 2 * b^2 := by
  sorry

end expression_simplification_l2938_293865


namespace discount_is_eleven_l2938_293898

/-- The discount on the first two books when ordering four books online --/
def discount_on_first_two_books : ℝ :=
  let free_shipping_threshold : ℝ := 50
  let book1_price : ℝ := 13
  let book2_price : ℝ := 15
  let book3_price : ℝ := 10
  let book4_price : ℝ := 10
  let additional_spend_needed : ℝ := 9
  let total_without_discount : ℝ := book1_price + book2_price + book3_price + book4_price
  let total_with_discount : ℝ := free_shipping_threshold + additional_spend_needed
  total_with_discount - total_without_discount

/-- Theorem stating that the discount on the first two books is $11.00 --/
theorem discount_is_eleven : discount_on_first_two_books = 11 := by
  sorry

end discount_is_eleven_l2938_293898


namespace three_Z_five_equals_fourteen_l2938_293837

-- Define the operation Z
def Z (a b : ℤ) : ℤ := b + 12 * a - a^3

-- Theorem statement
theorem three_Z_five_equals_fourteen : Z 3 5 = 14 := by
  sorry

end three_Z_five_equals_fourteen_l2938_293837


namespace profit_percentage_previous_year_l2938_293859

theorem profit_percentage_previous_year 
  (revenue_previous : ℝ) 
  (profit_previous : ℝ) 
  (revenue_decline : ℝ) 
  (profit_percentage_current : ℝ) 
  (profit_ratio : ℝ) 
  (h1 : revenue_decline = 0.3)
  (h2 : profit_percentage_current = 0.1)
  (h3 : profit_ratio = 0.6999999999999999)
  (h4 : profit_previous * profit_ratio = 
        (1 - revenue_decline) * revenue_previous * profit_percentage_current) :
  profit_previous / revenue_previous = 0.1 := by
sorry

end profit_percentage_previous_year_l2938_293859


namespace orthocenters_collinear_l2938_293813

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a function to check if three points are collinear
def collinear (p q r : Point) : Prop :=
  ∃ (t : ℝ), q.1 - p.1 = t * (r.1 - p.1) ∧ q.2 - p.2 = t * (r.2 - p.2)

-- Define a function to get the intersection point of two lines
noncomputable def intersect (l1 l2 : Line) : Point :=
  sorry

-- Define a function to get the orthocenter of a triangle
noncomputable def orthocenter (a b c : Point) : Point :=
  sorry

-- Main theorem
theorem orthocenters_collinear
  (l1 l2 l3 l4 : Line)
  (p1 p2 p3 p4 p5 p6 : Point)
  (h1 : p1 = intersect l1 l2)
  (h2 : p2 = intersect l1 l3)
  (h3 : p3 = intersect l1 l4)
  (h4 : p4 = intersect l2 l3)
  (h5 : p5 = intersect l2 l4)
  (h6 : p6 = intersect l3 l4)
  : collinear
      (orthocenter p1 p2 p4)
      (orthocenter p1 p3 p5)
      (orthocenter p2 p3 p6) :=
by
  sorry

end orthocenters_collinear_l2938_293813


namespace carls_garden_area_l2938_293803

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  total_posts : ℕ
  post_distance : ℕ
  longer_side_posts : ℕ
  shorter_side_posts : ℕ

/-- Calculates the area of the garden -/
def garden_area (g : Garden) : ℕ :=
  (g.shorter_side_posts - 1) * (g.longer_side_posts - 1) * g.post_distance * g.post_distance

/-- Theorem stating the area of Carl's garden -/
theorem carls_garden_area :
  ∀ g : Garden,
    g.total_posts = 36 ∧
    g.post_distance = 6 ∧
    g.longer_side_posts = 3 * g.shorter_side_posts ∧
    g.total_posts = 2 * (g.longer_side_posts + g.shorter_side_posts - 2) →
    garden_area g = 2016 := by
  sorry

end carls_garden_area_l2938_293803


namespace vector_BC_l2938_293839

/-- Given two vectors AB and AC in 2D space, prove that the vector BC is their difference -/
theorem vector_BC (AB AC : Fin 2 → ℝ) (h1 : AB = ![2, -1]) (h2 : AC = ![-4, 1]) :
  AC - AB = ![-6, 2] := by
  sorry

end vector_BC_l2938_293839


namespace hcl_moles_equal_one_l2938_293850

-- Define the chemical reaction
structure Reaction where
  naoh : ℕ  -- moles of Sodium hydroxide
  hcl : ℕ   -- moles of Hydrochloric acid
  h2o : ℕ   -- moles of Water produced

-- Define the balanced reaction
def balanced_reaction (r : Reaction) : Prop :=
  r.naoh = r.hcl ∧ r.naoh = r.h2o

-- Theorem statement
theorem hcl_moles_equal_one (r : Reaction) 
  (h1 : r.naoh = 1)  -- 1 mole of Sodium hydroxide is used
  (h2 : r.h2o = 1)   -- The reaction produces 1 mole of Water
  (h3 : balanced_reaction r) : -- The reaction is balanced
  r.hcl = 1 := by sorry

end hcl_moles_equal_one_l2938_293850


namespace line_intersects_circle_l2938_293894

theorem line_intersects_circle (a : ℝ) : ∃ (x y : ℝ), 
  y = a * x - a + 1 ∧ x^2 + y^2 = 8 := by
  sorry

#check line_intersects_circle

end line_intersects_circle_l2938_293894


namespace deans_height_l2938_293814

theorem deans_height (depth water_depth : ℝ) (h1 : water_depth = 10 * depth) (h2 : water_depth = depth + 81) : depth = 9 := by
  sorry

end deans_height_l2938_293814


namespace kims_earrings_l2938_293801

/-- Proves that Kim brings 5 pairs of earrings on the third day to have enough gumballs for 42 days -/
theorem kims_earrings (gumballs_per_pair : ℕ) (day1_pairs : ℕ) (daily_consumption : ℕ) (total_days : ℕ) :
  gumballs_per_pair = 9 →
  day1_pairs = 3 →
  daily_consumption = 3 →
  total_days = 42 →
  let day2_pairs := 2 * day1_pairs
  let day3_pairs := day2_pairs - 1
  let total_gumballs := gumballs_per_pair * (day1_pairs + day2_pairs + day3_pairs)
  total_gumballs = daily_consumption * total_days →
  day3_pairs = 5 := by
sorry


end kims_earrings_l2938_293801


namespace triangle_perimeter_l2938_293829

/-- The perimeter of a right triangle formed by specific lines --/
theorem triangle_perimeter : 
  ∀ (l₁ l₂ l₃ : Set (ℝ × ℝ)),
  (∃ (m : ℝ), l₁ = {(x, y) | y = m * x}) →  -- l₁ passes through origin
  (l₂ = {(x, y) | x = 2}) →                 -- l₂ is x = 2
  (l₃ = {(x, y) | y = 2 - (Real.sqrt 5 / 5) * x}) →  -- l₃ is y = 2 - (√5/5)x
  (∃ (p₁ p₂ p₃ : ℝ × ℝ), p₁ ∈ l₁ ∩ l₂ ∧ p₂ ∈ l₁ ∩ l₃ ∧ p₃ ∈ l₂ ∩ l₃) →  -- intersection points exist
  (∃ (v₁ v₂ : ℝ × ℝ), v₁ ∈ l₁ ∧ v₂ ∈ l₃ ∧ (v₁.1 - v₂.1) * (v₁.2 - v₂.2) = 0) →  -- right angle condition
  let perimeter := 2 + (12 * Real.sqrt 5 - 10) / 5 + 2 * Real.sqrt 6
  ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
    p₁ ∈ l₁ ∩ l₂ ∧ p₂ ∈ l₁ ∩ l₃ ∧ p₃ ∈ l₂ ∩ l₃ ∧
    Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) + 
    Real.sqrt ((p₂.1 - p₃.1)^2 + (p₂.2 - p₃.2)^2) + 
    Real.sqrt ((p₃.1 - p₁.1)^2 + (p₃.2 - p₁.2)^2) = perimeter :=
by
  sorry

end triangle_perimeter_l2938_293829


namespace monthly_bill_increase_l2938_293825

theorem monthly_bill_increase (original_bill : ℝ) (increase_percentage : ℝ) : 
  original_bill = 60 →
  increase_percentage = 0.30 →
  original_bill + (increase_percentage * original_bill) = 78 := by
  sorry

end monthly_bill_increase_l2938_293825


namespace polynomial_symmetry_l2938_293818

/-- Given a polynomial function f(x) = ax^7 - bx^5 + cx^3 + 2, 
    prove that f(5) + f(-5) = 4 -/
theorem polynomial_symmetry (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^7 - b * x^5 + c * x^3 + 2
  f 5 + f (-5) = 4 := by
  sorry

end polynomial_symmetry_l2938_293818


namespace arithmetic_mean_of_range_l2938_293800

def integerRange : List Int := List.range 10 |> List.map (λ x => x - 4)

theorem arithmetic_mean_of_range : 
  (integerRange.sum : ℚ) / integerRange.length = 1/2 := by
  sorry

end arithmetic_mean_of_range_l2938_293800


namespace composite_figure_area_l2938_293804

/-- The area of a composite figure with specific properties -/
theorem composite_figure_area : 
  let equilateral_triangle_area := Real.sqrt 3 / 4
  let rectangle_area := 1
  let right_triangle_area := 1 / 2
  2 * equilateral_triangle_area + rectangle_area + right_triangle_area = Real.sqrt 3 / 2 + 3 / 2 := by
  sorry

end composite_figure_area_l2938_293804


namespace first_negative_term_l2938_293885

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem first_negative_term (a₁ d : ℝ) (h₁ : a₁ = 51) (h₂ : d = -4) :
  ∀ k < 14, arithmetic_sequence a₁ d k ≥ 0 ∧
  arithmetic_sequence a₁ d 14 < 0 := by
sorry

end first_negative_term_l2938_293885


namespace greatest_possible_award_l2938_293808

theorem greatest_possible_award (total_prize : ℕ) (num_winners : ℕ) (min_award : ℕ) 
  (h1 : total_prize = 400)
  (h2 : num_winners = 20)
  (h3 : min_award = 20)
  (h4 : 2 * (total_prize / 5) = 3 * (num_winners / 5) * min_award) : 
  ∃ (max_award : ℕ), max_award = 100 ∧ 
  (∀ (award : ℕ), award ≤ max_award ∧ 
    (∃ (distribution : List ℕ), 
      distribution.length = num_winners ∧
      (∀ x ∈ distribution, min_award ≤ x) ∧
      distribution.sum = total_prize ∧
      award ∈ distribution)) := by
  sorry

end greatest_possible_award_l2938_293808


namespace arthur_walked_five_and_half_miles_l2938_293849

/-- The distance Arthur walked in miles -/
def arthurs_distance (east west north : ℕ) (block_length : ℚ) : ℚ :=
  (east + west + north : ℚ) * block_length

/-- Proof that Arthur walked 5.5 miles -/
theorem arthur_walked_five_and_half_miles :
  arthurs_distance 8 4 10 (1/4) = 5.5 := by
  sorry

end arthur_walked_five_and_half_miles_l2938_293849


namespace max_perpendicular_faces_theorem_l2938_293820

/-- The maximum number of faces perpendicular to the base in an n-sided pyramid -/
def max_perpendicular_faces (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1) / 2

/-- Theorem stating the maximum number of faces perpendicular to the base in an n-sided pyramid -/
theorem max_perpendicular_faces_theorem (n : ℕ) (h : n > 2) :
  max_perpendicular_faces n = if n % 2 = 0 then n / 2 else (n + 1) / 2 :=
by sorry

end max_perpendicular_faces_theorem_l2938_293820


namespace sin_239_equals_neg_cos_31_l2938_293896

theorem sin_239_equals_neg_cos_31 (a : ℝ) (h : Real.cos (31 * π / 180) = a) :
  Real.sin (239 * π / 180) = -a := by
  sorry

end sin_239_equals_neg_cos_31_l2938_293896
