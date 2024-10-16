import Mathlib

namespace NUMINAMATH_CALUDE_completing_square_quadratic_l1082_108201

theorem completing_square_quadratic (x : ℝ) :
  (x^2 - 4*x - 1 = 0) ↔ ((x - 2)^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l1082_108201


namespace NUMINAMATH_CALUDE_gcd_problem_l1082_108233

theorem gcd_problem : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 85 ∧ Nat.gcd n 30 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1082_108233


namespace NUMINAMATH_CALUDE_function_properties_l1082_108268

open Function

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivatives of f and g
variable (f' g' : ℝ → ℝ)

-- State the conditions
variable (h1 : ∀ x, HasDerivAt f (f' x) x)
variable (h2 : ∀ x, HasDerivAt g (g' x) x)
variable (h3 : ∀ x, f x = g ((x + 1) / 2) + x)
variable (h4 : Even f)
variable (h5 : Odd (fun x ↦ g' (x + 1)))

-- State the theorem
theorem function_properties :
  f' 1 = 1 ∧ g' (3/2) = 2 ∧ g' 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1082_108268


namespace NUMINAMATH_CALUDE_prob_odd_sum_is_two_thirds_l1082_108293

/-- A card is represented by a natural number between 1 and 4 -/
def Card := {n : ℕ // 1 ≤ n ∧ n ≤ 4}

/-- The set of all possible cards -/
def allCards : Finset Card := sorry

/-- A function to check if the sum of two cards is odd -/
def isOddSum (c1 c2 : Card) : Bool := sorry

/-- The set of all pairs of cards -/
def allPairs : Finset (Card × Card) := sorry

/-- The set of all pairs of cards with odd sum -/
def oddSumPairs : Finset (Card × Card) := sorry

/-- The probability of drawing two cards with odd sum -/
def probOddSum : ℚ := (Finset.card oddSumPairs : ℚ) / (Finset.card allPairs : ℚ)

theorem prob_odd_sum_is_two_thirds : probOddSum = 2/3 := by sorry

end NUMINAMATH_CALUDE_prob_odd_sum_is_two_thirds_l1082_108293


namespace NUMINAMATH_CALUDE_probability_at_most_five_digits_value_l1082_108220

/-- The probability of having at most five different digits in a randomly chosen
    sequence of seven digits, where each digit can be any of the digits 0 through 9. -/
def probability_at_most_five_digits : ℚ :=
  1 - (↑(Nat.choose 10 6 * 6 * (7 * 6 * 5 * 4 * 3 * 2 * 1 / 2)) / 10^7 +
       ↑(Nat.choose 10 7 * (7 * 6 * 5 * 4 * 3 * 2 * 1)) / 10^7)

/-- Theorem stating that the probability of having at most five different digits
    in the described sequence is equal to 0.622. -/
theorem probability_at_most_five_digits_value :
  probability_at_most_five_digits = 311 / 500 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_most_five_digits_value_l1082_108220


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l1082_108264

/-- An isosceles right triangle with perimeter 10 + 10√2 has a hypotenuse of length 10 -/
theorem isosceles_right_triangle_hypotenuse (a c : ℝ) : 
  a > 0 → 
  c > 0 → 
  a^2 + a^2 = c^2 → 
  2*a + c = 10 + 10*Real.sqrt 2 → 
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l1082_108264


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1082_108238

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3, 4, 5, 6}

-- Define set A
def A : Finset Nat := {1, 2}

-- Define set B
def B : Finset Nat := {0, 2, 5}

-- Theorem statement
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {0, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1082_108238


namespace NUMINAMATH_CALUDE_initial_drawer_pencils_count_l1082_108283

/-- The number of pencils initially in the drawer -/
def initial_drawer_pencils : ℕ := sorry

/-- The number of pencils initially on the desk -/
def initial_desk_pencils : ℕ := 19

/-- The number of pencils added to the desk -/
def added_desk_pencils : ℕ := 16

/-- The total number of pencils after the addition -/
def total_pencils : ℕ := 78

theorem initial_drawer_pencils_count : 
  initial_drawer_pencils = 43 :=
by sorry

end NUMINAMATH_CALUDE_initial_drawer_pencils_count_l1082_108283


namespace NUMINAMATH_CALUDE_quartic_equation_roots_l1082_108214

theorem quartic_equation_roots (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
    x₁^4 + p*x₁^3 + 3*x₁^2 + p*x₁ + 4 = 0 ∧
    x₂^4 + p*x₂^3 + 3*x₂^2 + p*x₂ + 4 = 0) ↔ 
  (p ≤ -2 ∨ p ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_quartic_equation_roots_l1082_108214


namespace NUMINAMATH_CALUDE_distinct_arrangements_l1082_108216

theorem distinct_arrangements (n : ℕ) (h : n = 8) : Nat.factorial n = 40320 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_l1082_108216


namespace NUMINAMATH_CALUDE_cos_15_cos_30_minus_sin_15_sin_150_l1082_108213

theorem cos_15_cos_30_minus_sin_15_sin_150 : 
  Real.cos (15 * π / 180) * Real.cos (30 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (150 * π / 180) = 
  Real.sqrt 2 / 2 :=
by
  -- Assuming sin 150° = sin 30°
  have h1 : Real.sin (150 * π / 180) = Real.sin (30 * π / 180) := by sorry
  sorry

end NUMINAMATH_CALUDE_cos_15_cos_30_minus_sin_15_sin_150_l1082_108213


namespace NUMINAMATH_CALUDE_double_points_imply_m_less_than_one_l1082_108204

/-- A double point is a point where the ordinate is twice its abscissa -/
def DoublePoint (x y : ℝ) : Prop := y = 2 * x

/-- The quadratic function -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x - m

theorem double_points_imply_m_less_than_one (m : ℝ) 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : DoublePoint x₁ y₁)
  (h₂ : DoublePoint x₂ y₂)
  (h₃ : f m x₁ = y₁)
  (h₄ : f m x₂ = y₂)
  (h₅ : x₁ < 1)
  (h₆ : 1 < x₂) :
  m < 1 := by sorry

end NUMINAMATH_CALUDE_double_points_imply_m_less_than_one_l1082_108204


namespace NUMINAMATH_CALUDE_largest_product_digit_sum_l1082_108236

def is_two_digit_prime_less_than_30 (p : ℕ) : Prop :=
  p ≥ 10 ∧ p < 30 ∧ Nat.Prime p

def largest_product (d e : ℕ) : ℕ :=
  d * e * (100 * d + e)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem largest_product_digit_sum :
  ∃ (d e : ℕ),
    is_two_digit_prime_less_than_30 d ∧
    is_two_digit_prime_less_than_30 e ∧
    d ≠ e ∧
    Nat.Prime (100 * d + e) ∧
    (∀ (d' e' : ℕ),
      is_two_digit_prime_less_than_30 d' ∧
      is_two_digit_prime_less_than_30 e' ∧
      d' ≠ e' ∧
      Nat.Prime (100 * d' + e') →
      largest_product d' e' ≤ largest_product d e) ∧
    sum_of_digits (largest_product d e) = 19 :=
  sorry

end NUMINAMATH_CALUDE_largest_product_digit_sum_l1082_108236


namespace NUMINAMATH_CALUDE_angle_after_rotation_l1082_108225

def rotation_result (initial_angle rotation : ℕ) : ℕ :=
  (rotation - initial_angle) % 360

theorem angle_after_rotation (initial_angle : ℕ) (h1 : initial_angle = 70) (rotation : ℕ) (h2 : rotation = 960) :
  rotation_result initial_angle rotation = 170 := by
  sorry

end NUMINAMATH_CALUDE_angle_after_rotation_l1082_108225


namespace NUMINAMATH_CALUDE_modified_lucas_105_mod_9_l1082_108200

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => modifiedLucas n + modifiedLucas (n + 1)

theorem modified_lucas_105_mod_9 : modifiedLucas 104 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_modified_lucas_105_mod_9_l1082_108200


namespace NUMINAMATH_CALUDE_isosceles_triangle_arctan_sum_l1082_108202

/-- In an isosceles triangle ABC where AB = AC, 
    arctan(c/(a+b)) + arctan(a/(b+c)) = π/4 -/
theorem isosceles_triangle_arctan_sum (a b c : ℝ) (α : ℝ) :
  b = c →  -- AB = AC implies b = c
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  α > 0 ∧ α < π →  -- Valid angle measure
  Real.arctan (c / (a + b)) + Real.arctan (a / (b + c)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_arctan_sum_l1082_108202


namespace NUMINAMATH_CALUDE_no_integer_root_for_any_a_l1082_108254

theorem no_integer_root_for_any_a : ∀ (a : ℤ), ¬∃ (x : ℤ), x^2 - 2023*x + 2022*a + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_root_for_any_a_l1082_108254


namespace NUMINAMATH_CALUDE_shaded_area_in_circle_l1082_108258

theorem shaded_area_in_circle (r : ℝ) (h : r = 6) : 
  let angle : ℝ := π / 3  -- 60° in radians
  let triangle_area : ℝ := (1/2) * r * r * Real.sin angle
  let sector_area : ℝ := (angle / (2 * π)) * π * r^2
  2 * triangle_area + 2 * sector_area = 36 * Real.sqrt 3 + 12 * π := by
sorry

end NUMINAMATH_CALUDE_shaded_area_in_circle_l1082_108258


namespace NUMINAMATH_CALUDE_inequality_proof_l1082_108278

theorem inequality_proof (a b : ℝ) (h : a + b ≠ 0) :
  (a + b) / (a^2 - a*b + b^2) ≤ 4 / |a + b| ∧
  ((a + b) / (a^2 - a*b + b^2) = 4 / |a + b| ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1082_108278


namespace NUMINAMATH_CALUDE_alex_phone_bill_l1082_108290

/-- Calculates the total cost of a cell phone plan based on usage --/
def calculate_phone_bill (base_cost : ℚ) (text_cost : ℚ) (extra_minute_cost : ℚ) 
  (extra_data_cost : ℚ) (texts_sent : ℕ) (hours_used : ℚ) (data_used : ℚ) : ℚ :=
  let text_charge := text_cost * texts_sent
  let extra_minutes := max (hours_used - 35) 0 * 60
  let minute_charge := extra_minute_cost * extra_minutes
  let extra_data := max (data_used - 2) 0 * 1000
  let data_charge := extra_data_cost * extra_data
  base_cost + text_charge + minute_charge + data_charge

/-- The total cost of Alex's cell phone plan in February is $126.30 --/
theorem alex_phone_bill : 
  calculate_phone_bill 30 0.07 0.12 0.15 150 36.5 2.5 = 126.30 := by
  sorry

end NUMINAMATH_CALUDE_alex_phone_bill_l1082_108290


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1082_108279

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + (2*m - 3)*x + (m^2 - 3) = 0) ↔ m ≤ 7/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1082_108279


namespace NUMINAMATH_CALUDE_log_identity_l1082_108205

theorem log_identity : Real.log 2 ^ 3 + 3 * Real.log 2 * Real.log 5 + Real.log 5 ^ 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l1082_108205


namespace NUMINAMATH_CALUDE_meal_price_calculation_l1082_108251

/-- Calculates the total price of a meal including tip -/
theorem meal_price_calculation (appetizer_cost entree_cost dessert_cost : ℚ)
  (num_entrees : ℕ) (tip_percentage : ℚ) :
  appetizer_cost = 9 ∧ 
  entree_cost = 20 ∧ 
  num_entrees = 2 ∧
  dessert_cost = 11 ∧
  tip_percentage = 30 / 100 →
  appetizer_cost + num_entrees * entree_cost + dessert_cost + 
  (appetizer_cost + num_entrees * entree_cost + dessert_cost) * tip_percentage = 78 := by
  sorry

end NUMINAMATH_CALUDE_meal_price_calculation_l1082_108251


namespace NUMINAMATH_CALUDE_expression_evaluation_l1082_108249

theorem expression_evaluation (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  x * y + (3 * x * y - 4 * x^2) - 2 * (x * y - 2 * x^2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1082_108249


namespace NUMINAMATH_CALUDE_sum_of_squares_l1082_108284

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 2*b = 8)
  (eq2 : b^2 + 4*c + 1 = -6)
  (eq3 : c^2 + 6*a = -15) :
  a^2 + b^2 + c^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1082_108284


namespace NUMINAMATH_CALUDE_cartesian_to_polar_l1082_108265

theorem cartesian_to_polar (x y : ℝ) : 
  x = 2 ∧ y = -2 * Real.sqrt 3 →
  ∃ (ρ θ : ℝ), 
    ρ = 4 ∧ 
    θ = -2 * Real.pi / 3 ∧
    x = ρ * Real.cos θ ∧
    y = ρ * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_l1082_108265


namespace NUMINAMATH_CALUDE_solution_set_l1082_108237

theorem solution_set (a : ℝ) (h : (4 : ℝ)^a = 2^(a + 2)) :
  {x : ℝ | a^(2*x + 1) > a^(x - 1)} = {x : ℝ | x > -2} := by sorry

end NUMINAMATH_CALUDE_solution_set_l1082_108237


namespace NUMINAMATH_CALUDE_average_discount_rate_proof_l1082_108241

theorem average_discount_rate_proof (bag_marked bag_sold shoes_marked shoes_sold jacket_marked jacket_sold : ℝ) 
  (h1 : bag_marked = 80)
  (h2 : bag_sold = 68)
  (h3 : shoes_marked = 120)
  (h4 : shoes_sold = 96)
  (h5 : jacket_marked = 150)
  (h6 : jacket_sold = 135) :
  let bag_discount := (bag_marked - bag_sold) / bag_marked
  let shoes_discount := (shoes_marked - shoes_sold) / shoes_marked
  let jacket_discount := (jacket_marked - jacket_sold) / jacket_marked
  (bag_discount + shoes_discount + jacket_discount) / 3 = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_average_discount_rate_proof_l1082_108241


namespace NUMINAMATH_CALUDE_intersection_above_axis_implies_no_roots_l1082_108210

/-- 
Given that the graphs of y = ax², y = bx, and y = c intersect at a point above the x-axis,
prove that the equation ax² + bx + c = 0 has no real roots.
-/
theorem intersection_above_axis_implies_no_roots 
  (a b c : ℝ) 
  (ha : a > 0)
  (hc : c > 0)
  (h_intersect : ∃ (m : ℝ), a * m^2 = b * m ∧ b * m = c) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_above_axis_implies_no_roots_l1082_108210


namespace NUMINAMATH_CALUDE_missing_number_solution_l1082_108227

theorem missing_number_solution : ∃ x : ℤ, 10010 - 12 * x * 2 = 9938 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_solution_l1082_108227


namespace NUMINAMATH_CALUDE_complex_calculation_l1082_108291

theorem complex_calculation : ((7 - 3 * Complex.I) - 3 * (2 - 5 * Complex.I)) * (1 + 2 * Complex.I) = -23 + 14 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l1082_108291


namespace NUMINAMATH_CALUDE_medium_pizza_slices_l1082_108292

theorem medium_pizza_slices :
  -- Define the number of slices for small and large pizzas
  let small_slices : ℕ := 6
  let large_slices : ℕ := 12
  -- Define the total number of pizzas and the number of each size
  let total_pizzas : ℕ := 15
  let small_pizzas : ℕ := 4
  let medium_pizzas : ℕ := 5
  -- Define the total number of slices
  let total_slices : ℕ := 136
  -- Calculate the number of large pizzas
  let large_pizzas : ℕ := total_pizzas - small_pizzas - medium_pizzas
  -- Define the number of slices in a medium pizza as a variable
  ∀ medium_slices : ℕ,
  -- If the total slices equation holds
  (small_pizzas * small_slices + medium_pizzas * medium_slices + large_pizzas * large_slices = total_slices) →
  -- Then the number of slices in a medium pizza must be 8
  medium_slices = 8 := by
sorry

end NUMINAMATH_CALUDE_medium_pizza_slices_l1082_108292


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1082_108271

theorem inequality_system_solution (x : ℝ) : 
  3 * x > x + 6 ∧ (1/2) * x < -x + 5 → 3 < x ∧ x < 10/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1082_108271


namespace NUMINAMATH_CALUDE_symmetric_through_swaps_l1082_108239

/-- A binary digit (0 or 1) -/
inductive BinaryDigit : Type
| zero : BinaryDigit
| one : BinaryDigit

/-- A sequence of binary digits -/
def BinarySequence := List BinaryDigit

/-- Swap operation that exchanges two elements in a list at given indices -/
def swap (seq : BinarySequence) (i j : Nat) : BinarySequence :=
  sorry

/-- Check if a sequence is symmetric -/
def isSymmetric (seq : BinarySequence) : Prop :=
  sorry

/-- The main theorem stating that any binary sequence of length 1999 can be made symmetric through swaps -/
theorem symmetric_through_swaps (seq : BinarySequence) (h : seq.length = 1999) :
  ∃ (swapSequence : List (Nat × Nat)), 
    isSymmetric (swapSequence.foldl (λ s (i, j) => swap s i j) seq) :=
  sorry

end NUMINAMATH_CALUDE_symmetric_through_swaps_l1082_108239


namespace NUMINAMATH_CALUDE_value_of_a_l1082_108295

theorem value_of_a (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 / b = 2) (h2 : b^2 / c = 4) (h3 : c^2 / a = 6) :
  a = (384 : ℝ)^(1/7) := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l1082_108295


namespace NUMINAMATH_CALUDE_curly_bracket_calculation_l1082_108270

-- Define the ceiling function for rational numbers
def ceiling (a : ℚ) : ℤ := Int.ceil a

-- Define the curly bracket notation
def curly_bracket (a : ℚ) : ℤ := ceiling a

-- Theorem statement
theorem curly_bracket_calculation :
  (curly_bracket (-6 + 5/6) : ℚ) - 
  (curly_bracket 5 : ℚ) * (curly_bracket (-1 - 3/4) : ℚ) / (curly_bracket (59/10) : ℚ) = -5 := by
  sorry


end NUMINAMATH_CALUDE_curly_bracket_calculation_l1082_108270


namespace NUMINAMATH_CALUDE_completing_square_result_l1082_108221

theorem completing_square_result (x : ℝ) : 
  x^2 - 4*x - 1 = 0 → (x - 2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l1082_108221


namespace NUMINAMATH_CALUDE_no_real_solution_cubic_equation_l1082_108243

theorem no_real_solution_cubic_equation :
  ∀ x : ℝ, x > 0 → 4 * x^(1/3) - 3 * (x / x^(2/3)) ≠ 10 + 2 * x^(1/3) + x^(2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_cubic_equation_l1082_108243


namespace NUMINAMATH_CALUDE_smallest_y_for_square_l1082_108217

theorem smallest_y_for_square (y : ℕ+) (M : ℤ) : 
  (∀ k : ℕ+, k < y → ¬∃ N : ℤ, (2310 : ℤ) * k = N^2) →
  (∃ N : ℤ, (2310 : ℤ) * y = N^2) →
  y = 2310 := by sorry

end NUMINAMATH_CALUDE_smallest_y_for_square_l1082_108217


namespace NUMINAMATH_CALUDE_hyperbola_equation_hyperbola_eccentricity_l1082_108235

/-- Represents a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_c_eq : c = 2
  h_asymptote : b = a

/-- The equation of the hyperbola is (x^2 / 2) - (y^2 / 2) = 1 -/
theorem hyperbola_equation (h : Hyperbola) : 
  h.a^2 = 2 ∧ h.b^2 = 2 :=
sorry

/-- The eccentricity of the hyperbola is √2 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  Real.sqrt (h.c^2 / h.a^2) = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_hyperbola_eccentricity_l1082_108235


namespace NUMINAMATH_CALUDE_max_xy_value_l1082_108208

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 112 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l1082_108208


namespace NUMINAMATH_CALUDE_triangle_area_product_l1082_108266

theorem triangle_area_product (p q : ℝ) : 
  p > 0 → q > 0 → (1/2 * (24/p) * (24/q) = 48) → p * q = 12 := by sorry

end NUMINAMATH_CALUDE_triangle_area_product_l1082_108266


namespace NUMINAMATH_CALUDE_sequence_term_value_l1082_108256

-- Define the sequence sum
def S (n : ℕ) : ℤ := n^2 - 6*n

-- Define the m-th term of the sequence
def a (m : ℕ) : ℤ := 2*m - 7

-- Theorem statement
theorem sequence_term_value (m : ℕ) :
  (∀ n : ℕ, S n = n^2 - 6*n) →
  (∀ k : ℕ, k ≥ 2 → a k = S k - S (k-1)) →
  (5 < a m ∧ a m < 8) →
  m = 7 := by sorry

end NUMINAMATH_CALUDE_sequence_term_value_l1082_108256


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1082_108247

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_2 : a 2 = 5)
  (h_4 : a 4 = 20) :
  a 6 = 80 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1082_108247


namespace NUMINAMATH_CALUDE_two_cars_in_garage_l1082_108299

/-- Represents the number of wheels on various vehicles --/
structure VehicleWheels where
  lawnmower : Nat
  bicycle : Nat
  tricycle : Nat
  unicycle : Nat
  car : Nat

/-- Calculates the total number of wheels for non-car vehicles --/
def nonCarWheels (v : VehicleWheels) (numBicycles : Nat) : Nat :=
  v.lawnmower + numBicycles * v.bicycle + v.tricycle + v.unicycle

/-- Theorem stating that given the conditions in the problem, there are 2 cars in the garage --/
theorem two_cars_in_garage (totalWheels : Nat) (v : VehicleWheels) (numBicycles : Nat) :
  totalWheels = 22 →
  v.lawnmower = 4 →
  v.bicycle = 2 →
  v.tricycle = 3 →
  v.unicycle = 1 →
  v.car = 4 →
  numBicycles = 3 →
  (totalWheels - nonCarWheels v numBicycles) / v.car = 2 :=
by sorry

end NUMINAMATH_CALUDE_two_cars_in_garage_l1082_108299


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l1082_108257

theorem chord_length_concentric_circles (R r : ℝ) (h : R > r) :
  R^2 - r^2 = 20 →
  ∃ c : ℝ, c^2 / 4 + r^2 = R^2 ∧ c = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l1082_108257


namespace NUMINAMATH_CALUDE_rectangle_dimension_increase_l1082_108206

/-- Given a rectangle with original length L and breadth B, prove that if the breadth is
    increased by 25% and the total area is increased by 37.5%, then the length must be
    increased by 10% -/
theorem rectangle_dimension_increase (L B : ℝ) (L_pos : L > 0) (B_pos : B > 0) :
  let new_B := 1.25 * B
  let new_area := 1.375 * (L * B)
  ∃ x : ℝ, x = 0.1 ∧ new_area = (L * (1 + x)) * new_B := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_increase_l1082_108206


namespace NUMINAMATH_CALUDE_opposite_of_reciprocal_l1082_108240

theorem opposite_of_reciprocal : -(1 / (-1/3)) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_reciprocal_l1082_108240


namespace NUMINAMATH_CALUDE_money_division_l1082_108261

/-- The problem of dividing money among A, B, and C -/
theorem money_division (a b c : ℚ) : 
  a + b + c = 360 →  -- Total amount is $360
  a = (1/3) * (b + c) →  -- A gets 1/3 of B and C combined
  a = b + 10 →  -- A gets $10 more than B
  ∃ x : ℚ, b = x * (a + c) ∧ x = 2/7  -- B gets a fraction of A and C combined
  := by sorry

end NUMINAMATH_CALUDE_money_division_l1082_108261


namespace NUMINAMATH_CALUDE_jerome_toy_cars_l1082_108215

theorem jerome_toy_cars (original : ℕ) (total : ℕ) (last_month : ℕ) :
  original = 25 →
  total = 40 →
  total = original + last_month + 2 * last_month →
  last_month = 5 := by
  sorry

end NUMINAMATH_CALUDE_jerome_toy_cars_l1082_108215


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1082_108297

/-- The point P(1+m^2, -1) lies in the fourth quadrant for any real number m. -/
theorem point_in_fourth_quadrant (m : ℝ) : 
  let x : ℝ := 1 + m^2
  let y : ℝ := -1
  x > 0 ∧ y < 0 := by
sorry


end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1082_108297


namespace NUMINAMATH_CALUDE_blood_expiration_time_l1082_108280

-- Define the number of seconds in a day
def seconds_per_day : ℕ := 24 * 60 * 60

-- Define the expiration time in seconds (8!)
def expiration_time : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

-- Define the donation time (noon)
def donation_hour : ℕ := 12

-- Theorem statement
theorem blood_expiration_time :
  (expiration_time / seconds_per_day = 0) ∧
  (expiration_time % seconds_per_day / 3600 + donation_hour = 23) :=
sorry

end NUMINAMATH_CALUDE_blood_expiration_time_l1082_108280


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l1082_108242

theorem cube_sum_inequality (x y z : ℝ) : 
  x^3 + y^3 + z^3 + 3*x*y*z ≥ x^2*(y+z) + y^2*(z+x) + z^2*(x+y) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l1082_108242


namespace NUMINAMATH_CALUDE_simplify_expression_l1082_108267

theorem simplify_expression (a b : ℝ) : (a + b)^2 - a*(a + 2*b) = b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1082_108267


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1082_108289

theorem trig_identity_proof : 
  (Real.sqrt 3 / Real.sin (20 * π / 180)) - (1 / Real.sin (70 * π / 180)) = 4 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1082_108289


namespace NUMINAMATH_CALUDE_max_value_of_f_l1082_108203

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (8 - x))

theorem max_value_of_f :
  ∃ (M : ℝ) (x₀ : ℝ),
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 8 → f x ≤ M) ∧
    (0 ≤ x₀ ∧ x₀ ≤ 8) ∧
    (f x₀ = M) ∧
    (x₀ = 200 / 27) ∧
    (M = 12 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1082_108203


namespace NUMINAMATH_CALUDE_pencil_case_notebook_prices_l1082_108228

theorem pencil_case_notebook_prices :
  ∀ (notebook_price pencil_case_price : ℚ),
    pencil_case_price = notebook_price + 3 →
    (200 : ℚ) / notebook_price = (350 : ℚ) / pencil_case_price →
    notebook_price = 4 ∧ pencil_case_price = 7 := by
  sorry

end NUMINAMATH_CALUDE_pencil_case_notebook_prices_l1082_108228


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l1082_108252

theorem polynomial_functional_equation :
  ∃ (q : ℝ → ℝ), (∀ x, q x = -2 * x + 4) ∧
                 (∀ x, q (q x) = x * q x + 2 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l1082_108252


namespace NUMINAMATH_CALUDE_total_children_l1082_108285

/-- The number of happy children -/
def happy_children : ℕ := 30

/-- The number of sad children -/
def sad_children : ℕ := 10

/-- The number of children who are neither happy nor sad -/
def neutral_children : ℕ := 20

/-- The number of boys -/
def boys : ℕ := 17

/-- The number of girls -/
def girls : ℕ := 43

/-- The number of happy boys -/
def happy_boys : ℕ := 6

/-- The number of sad girls -/
def sad_girls : ℕ := 4

/-- The number of boys who are neither happy nor sad -/
def neutral_boys : ℕ := 5

/-- Theorem stating that the total number of children is 60 -/
theorem total_children : boys + girls = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_children_l1082_108285


namespace NUMINAMATH_CALUDE_no_reciprocal_roots_l1082_108244

theorem no_reciprocal_roots (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬∃ (n : ℕ), a * (1 / n : ℚ)^2 + b * (1 / n : ℚ) + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_reciprocal_roots_l1082_108244


namespace NUMINAMATH_CALUDE_student_marks_average_l1082_108288

theorem student_marks_average (P C M B : ℝ) 
  (h1 : P + C + M + B = P + B + 180)
  (h2 : P = 1.20 * B) :
  (C + M) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_average_l1082_108288


namespace NUMINAMATH_CALUDE_remainder_plus_3255_l1082_108223

theorem remainder_plus_3255 (n : ℤ) (h : n % 5 = 2) : (n + 3255) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_plus_3255_l1082_108223


namespace NUMINAMATH_CALUDE_cookie_price_l1082_108296

/-- Proves that the price of each cookie is $0.50 given the conditions of the basketball team's sales and purchases. -/
theorem cookie_price (cupcake_count : ℕ) (cupcake_price : ℚ) (cookie_count : ℕ) 
  (basketball_count : ℕ) (basketball_price : ℚ) (drink_count : ℕ) (drink_price : ℚ) :
  cupcake_count = 50 →
  cupcake_price = 2 →
  cookie_count = 40 →
  basketball_count = 2 →
  basketball_price = 40 →
  drink_count = 20 →
  drink_price = 2 →
  ∃ (cookie_price : ℚ),
    cupcake_count * cupcake_price + cookie_count * cookie_price = 
    basketball_count * basketball_price + drink_count * drink_price ∧
    cookie_price = 1/2 := by
  sorry

#check cookie_price

end NUMINAMATH_CALUDE_cookie_price_l1082_108296


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_2n_gt_sqrt_n_l1082_108273

theorem negation_of_proposition (p : ℕ → Prop) : 
  (¬∀ n, p n) ↔ (∃ n, ¬p n) :=
by sorry

theorem negation_of_2n_gt_sqrt_n :
  (¬∀ n : ℕ, 2^n > Real.sqrt n) ↔ (∃ n : ℕ, 2^n ≤ Real.sqrt n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_2n_gt_sqrt_n_l1082_108273


namespace NUMINAMATH_CALUDE_one_multiple_choice_one_true_false_prob_at_least_one_multiple_choice_prob_l1082_108219

-- Define the total number of questions and their types
def total_questions : ℕ := 5
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 20

-- Theorem for the first probability
theorem one_multiple_choice_one_true_false_prob :
  (multiple_choice_questions * true_false_questions * 2) / total_outcomes = 3 / 5 := by
  sorry

-- Theorem for the second probability
theorem at_least_one_multiple_choice_prob :
  1 - (true_false_questions * (true_false_questions - 1) / 2) / total_outcomes = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_multiple_choice_one_true_false_prob_at_least_one_multiple_choice_prob_l1082_108219


namespace NUMINAMATH_CALUDE_inequality_solution_l1082_108276

theorem inequality_solution (a b c d : ℝ) : 
  (∀ x : ℝ, ((x - a) * (x - b) * (x - c)) / (x - d) ≤ 0 ↔ 
    (x < -4 ∨ (1 ≤ x ∧ x ≤ 5) ∨ (24 ≤ x ∧ x ≤ 26))) →
  a < b →
  b < c →
  a + 3*b + 3*c + 4*d = 72 := by
sorry


end NUMINAMATH_CALUDE_inequality_solution_l1082_108276


namespace NUMINAMATH_CALUDE_divisibility_relation_l1082_108298

theorem divisibility_relation :
  (∀ n : ℤ, n % 6 = 0 → n % 2 = 0) ∧
  (∃ n : ℤ, n % 2 = 0 ∧ n % 6 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_relation_l1082_108298


namespace NUMINAMATH_CALUDE_equation_solution_l1082_108246

theorem equation_solution : 
  ∃ x : ℝ, (((1 - x) / (x - 4)) + (1 / (4 - x)) = 1) ∧ (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1082_108246


namespace NUMINAMATH_CALUDE_parabola_equation_l1082_108212

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := x + 2 * y + 3 = 0

-- Define the two possible standard equations for the parabola
def vertical_parabola (x y : ℝ) : Prop := x^2 = -6 * y
def horizontal_parabola (x y : ℝ) : Prop := y^2 = -12 * x

-- Theorem statement
theorem parabola_equation (C : Set (ℝ × ℝ)) :
  (∃ (x y : ℝ), (x, y) ∈ C ∧ focus_line x y) →
  (∀ (x y : ℝ), (x, y) ∈ C → vertical_parabola x y ∨ horizontal_parabola x y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1082_108212


namespace NUMINAMATH_CALUDE_min_value_theorem_l1082_108224

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ x : ℝ, 1 ≤ x → x ≤ 4 → a * x + b - 3 ≤ 0) : 
  1 / a - b ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1082_108224


namespace NUMINAMATH_CALUDE_x_value_l1082_108281

theorem x_value (x y : ℝ) (h : x / (x - 1) = (y^2 + 3*y - 5) / (y^2 + 3*y - 7)) :
  x = (y^2 + 3*y - 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1082_108281


namespace NUMINAMATH_CALUDE_ellipse_equation_and_intersection_range_l1082_108286

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the line x - y + 2√2 = 0
def Line := {p : ℝ × ℝ | p.1 - p.2 + 2 * Real.sqrt 2 = 0}

theorem ellipse_equation_and_intersection_range :
  ∃ (a b c : ℝ),
    -- Conditions
    (0, -1) ∈ Ellipse a b ∧  -- One vertex at (0, -1)
    (c, 0) ∈ Ellipse a b ∧   -- Right focus on x-axis
    (∀ (x y : ℝ), (x, y) ∈ Line → ((x - c)^2 + y^2).sqrt = 3) ∧  -- Distance from right focus to line is 3
    -- Conclusions
    (Ellipse a b = Ellipse (Real.sqrt 3) 1) ∧  -- Equation of ellipse
    (∀ m : ℝ, (∃ (p q : ℝ × ℝ), p ≠ q ∧ p ∈ Ellipse (Real.sqrt 3) 1 ∧ q ∈ Ellipse (Real.sqrt 3) 1 ∧
                p.2 = p.1 + m ∧ q.2 = q.1 + m) ↔ -2 < m ∧ m < 2)  -- Intersection range
    := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_intersection_range_l1082_108286


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1082_108211

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, x^2 + x - c < 0 ↔ -2 < x ∧ x < 1) → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1082_108211


namespace NUMINAMATH_CALUDE_john_trip_spending_difference_l1082_108229

/-- Calculates the difference between the amount spent on a trip and the remaining amount --/
def tripSpendingDifference (initial : ℕ) (remaining : ℕ) : ℕ :=
  (initial - remaining) - remaining

theorem john_trip_spending_difference :
  let initial := 1600
  let remaining := 500
  tripSpendingDifference initial remaining = 600 ∧
  remaining < initial - remaining := by
  sorry

end NUMINAMATH_CALUDE_john_trip_spending_difference_l1082_108229


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1082_108230

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 4]

-- Define the parallelism condition
def are_parallel (u v : Fin 2 → ℝ) : Prop :=
  u 0 * v 1 = u 1 * v 0

-- Theorem statement
theorem parallel_vectors_m_value :
  ∀ m : ℝ, are_parallel a (λ i ↦ 2 * a i + b m i) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1082_108230


namespace NUMINAMATH_CALUDE_rectangle_area_l1082_108231

/-- A rectangle with three congruent circles inside -/
structure RectangleWithCircles where
  -- The length of the rectangle
  length : ℝ
  -- The width of the rectangle
  width : ℝ
  -- The diameter of each circle
  circle_diameter : ℝ
  -- The circles are congruent
  circles_congruent : True
  -- Each circle is tangent to two sides of the rectangle
  circles_tangent : True
  -- The circle centered at F is tangent to sides JK and LM
  circle_f_tangent : True
  -- The diameter of circle F is 5
  circle_f_diameter : circle_diameter = 5

/-- The area of the rectangle JKLM is 50 -/
theorem rectangle_area (r : RectangleWithCircles) : r.length * r.width = 50 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1082_108231


namespace NUMINAMATH_CALUDE_largest_package_size_l1082_108282

def markers_elliot : ℕ := 60
def markers_tara : ℕ := 36
def markers_sam : ℕ := 90

theorem largest_package_size : ∃ (n : ℕ), n > 0 ∧ 
  markers_elliot % n = 0 ∧ 
  markers_tara % n = 0 ∧ 
  markers_sam % n = 0 ∧
  ∀ (m : ℕ), m > n → 
    (markers_elliot % m = 0 ∧ 
     markers_tara % m = 0 ∧ 
     markers_sam % m = 0) → False :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l1082_108282


namespace NUMINAMATH_CALUDE_student_number_problem_l1082_108260

theorem student_number_problem (x : ℝ) : 3 * x - 220 = 110 → x = 110 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1082_108260


namespace NUMINAMATH_CALUDE_cistern_filling_time_l1082_108222

theorem cistern_filling_time (x : ℝ) : 
  x > 0 ∧                            -- x is positive (time can't be negative or zero)
  (1 / x + 1 / 12 - 1 / 20 = 1 / 7.5) -- combined rate equation
  → x = 10 := by
sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l1082_108222


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_l1082_108253

theorem sum_of_x_solutions (y : ℝ) (h1 : y = 9) (h2 : ∃ x : ℝ, x^2 + y^2 = 225) : 
  ∃ x₁ x₂ : ℝ, x₁^2 + y^2 = 225 ∧ x₂^2 + y^2 = 225 ∧ x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_l1082_108253


namespace NUMINAMATH_CALUDE_existence_of_more_good_numbers_l1082_108218

/-- A function that determines if a natural number is "good" or "bad" --/
def isGoodNumber (x : ℕ) : Bool := sorry

/-- The count of n-digit numbers that are "good" --/
def countGoodNumbers (n : ℕ) : ℕ := sorry

/-- The count of n-digit numbers that are "bad" --/
def countBadNumbers (n : ℕ) : ℕ := sorry

/-- The total count of n-digit numbers --/
def totalNumbers (n : ℕ) : ℕ := 9 * (10 ^ (n - 1))

theorem existence_of_more_good_numbers :
  ∃ n : ℕ, n ≥ 4 ∧ countGoodNumbers n > countBadNumbers n :=
sorry

end NUMINAMATH_CALUDE_existence_of_more_good_numbers_l1082_108218


namespace NUMINAMATH_CALUDE_distance_C_D_l1082_108255

/-- An ellipse with equation 4(x-2)^2 + 16y^2 = 64 -/
structure Ellipse where
  -- The equation is implicitly defined by the structure

/-- Point C is an endpoint of the minor axis -/
def C (e : Ellipse) : ℝ × ℝ := sorry

/-- Point D is an endpoint of the major axis -/
def D (e : Ellipse) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The distance between C and D is 2√5 -/
theorem distance_C_D (e : Ellipse) : distance (C e) (D e) = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_distance_C_D_l1082_108255


namespace NUMINAMATH_CALUDE_regular_icosahedron_has_12_vertices_l1082_108277

/-- A regular icosahedron is a polyhedron with equilateral triangles as faces -/
structure RegularIcosahedron where
  /-- The number of faces in the icosahedron -/
  faces : ℕ
  /-- The number of vertices in the icosahedron -/
  vertices : ℕ
  /-- The number of edges in the icosahedron -/
  edges : ℕ
  /-- All faces are equilateral triangles -/
  all_faces_equilateral : True
  /-- Euler's formula for polyhedra: V - E + F = 2 -/
  euler_formula : vertices - edges + faces = 2
  /-- Each face is a triangle, so 3F = 2E -/
  face_edge_relation : 3 * faces = 2 * edges
  /-- Each vertex has degree 5 in an icosahedron -/
  vertex_degree_five : 5 * vertices = 2 * edges

/-- Theorem: A regular icosahedron has 12 vertices -/
theorem regular_icosahedron_has_12_vertices (i : RegularIcosahedron) : i.vertices = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_icosahedron_has_12_vertices_l1082_108277


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1082_108250

/-- Given a quadratic function y = x^2 - 1840x + 2009 with roots m and n,
    prove that (m^2 - 1841m + 2009)(n^2 - 1841n + 2009) = 2009 -/
theorem quadratic_roots_property (m n : ℝ) : 
  m^2 - 1840*m + 2009 = 0 →
  n^2 - 1840*n + 2009 = 0 →
  (m^2 - 1841*m + 2009) * (n^2 - 1841*n + 2009) = 2009 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1082_108250


namespace NUMINAMATH_CALUDE_batch_size_proof_l1082_108234

theorem batch_size_proof :
  ∃! n : ℕ, 500 ≤ n ∧ n ≤ 600 ∧ n % 20 = 13 ∧ n % 27 = 20 ∧ n = 533 := by
  sorry

end NUMINAMATH_CALUDE_batch_size_proof_l1082_108234


namespace NUMINAMATH_CALUDE_transfer_increases_averages_l1082_108232

/-- Represents a group of students with their average grade and count -/
structure StudentGroup where
  avg_grade : ℝ
  count : ℕ

/-- Checks if transferring students increases average grades in both groups -/
def increases_averages (group_a group_b : StudentGroup) (grade1 grade2 : ℝ) : Prop :=
  let new_a := StudentGroup.mk
    ((group_a.avg_grade * group_a.count - grade1 - grade2) / (group_a.count - 2))
    (group_a.count - 2)
  let new_b := StudentGroup.mk
    ((group_b.avg_grade * group_b.count + grade1 + grade2) / (group_b.count + 2))
    (group_b.count + 2)
  new_a.avg_grade > group_a.avg_grade ∧ new_b.avg_grade > group_b.avg_grade

theorem transfer_increases_averages :
  let group_a := StudentGroup.mk 44.2 10
  let group_b := StudentGroup.mk 38.8 10
  let grade1 := 41
  let grade2 := 44
  increases_averages group_a group_b grade1 grade2 := by
  sorry

end NUMINAMATH_CALUDE_transfer_increases_averages_l1082_108232


namespace NUMINAMATH_CALUDE_weight_of_new_person_l1082_108275

theorem weight_of_new_person (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) :
  n = 5 →
  avg_increase = 1.5 →
  replaced_weight = 65 →
  ∃ (new_weight : ℝ), new_weight = 72.5 ∧
    n * avg_increase = new_weight - replaced_weight :=
by sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l1082_108275


namespace NUMINAMATH_CALUDE_impossible_equal_tokens_l1082_108226

/-- Represents the state of tokens --/
structure TokenState where
  green : ℕ
  red : ℕ

/-- Represents a token exchange operation --/
inductive Exchange
  | GreenToRed
  | RedToGreen

/-- Applies an exchange to a token state --/
def applyExchange (state : TokenState) (ex : Exchange) : TokenState :=
  match ex with
  | Exchange.GreenToRed => 
      if state.green ≥ 1 then ⟨state.green - 1, state.red + 5⟩ else state
  | Exchange.RedToGreen => 
      if state.red ≥ 1 then ⟨state.green + 5, state.red - 1⟩ else state

/-- A sequence of exchanges --/
def ExchangeSequence := List Exchange

/-- Applies a sequence of exchanges to a token state --/
def applyExchangeSequence (state : TokenState) (seq : ExchangeSequence) : TokenState :=
  seq.foldl applyExchange state

/-- The theorem to be proved --/
theorem impossible_equal_tokens : 
  ∀ (seq : ExchangeSequence), 
  let finalState := applyExchangeSequence ⟨1, 0⟩ seq
  finalState.green ≠ finalState.red :=
sorry

end NUMINAMATH_CALUDE_impossible_equal_tokens_l1082_108226


namespace NUMINAMATH_CALUDE_group_dynamics_index_difference_l1082_108207

theorem group_dynamics_index_difference :
  let n : ℕ := 35
  let k1 : ℕ := 15
  let k2 : ℕ := 5
  let k3 : ℕ := 8
  let l1 : ℕ := 6
  let l2 : ℕ := 10
  let index_females : ℚ := ((n - k1 + k2) / n : ℚ) * (1 + k3/10)
  let index_males : ℚ := ((n - (n - k1) + l1) / n : ℚ) * (1 + l2/10)
  index_females - index_males = 3/35 := by
  sorry

end NUMINAMATH_CALUDE_group_dynamics_index_difference_l1082_108207


namespace NUMINAMATH_CALUDE_round_table_seats_l1082_108262

/-- Represents a round table with equally spaced seats -/
structure RoundTable where
  total_seats : ℕ
  seat_numbers : Fin total_seats → ℕ

/-- Two seats are opposite if they are half the total number of seats apart -/
def opposite (t : RoundTable) (s1 s2 : Fin t.total_seats) : Prop :=
  (t.seat_numbers s2 - t.seat_numbers s1) % t.total_seats = t.total_seats / 2

theorem round_table_seats (t : RoundTable) (s1 s2 : Fin t.total_seats) :
  t.seat_numbers s1 = 10 ∧ t.seat_numbers s2 = 29 ∧ opposite t s1 s2 → t.total_seats = 38 :=
by
  sorry


end NUMINAMATH_CALUDE_round_table_seats_l1082_108262


namespace NUMINAMATH_CALUDE_coordinate_sum_of_A_l1082_108294

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the coordinate plane -/
structure Line where
  m : ℝ
  b : ℝ

/-- The theorem statement -/
theorem coordinate_sum_of_A (A B C : Point) (l₁ l₂ l₃ : Line) (a b : ℝ) :
  B.y = 0 →  -- B is on Ox axis
  C.x = 0 →  -- C is on Oy axis
  (l₁.m = a ∧ l₁.b = 4) ∨ (l₁.m = 2 ∧ l₁.b = b) ∨ (l₁.m = a/2 ∧ l₁.b = 8) →  -- l₁ is one of the given lines
  (l₂.m = a ∧ l₂.b = 4) ∨ (l₂.m = 2 ∧ l₂.b = b) ∨ (l₂.m = a/2 ∧ l₂.b = 8) →  -- l₂ is one of the given lines
  (l₃.m = a ∧ l₃.b = 4) ∨ (l₃.m = 2 ∧ l₃.b = b) ∨ (l₃.m = a/2 ∧ l₃.b = 8) →  -- l₃ is one of the given lines
  l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃ →  -- All lines are different
  (A.y = l₁.m * A.x + l₁.b) ∧ (B.y = l₁.m * B.x + l₁.b) →  -- A and B are on l₁
  (B.y = l₂.m * B.x + l₂.b) ∧ (C.y = l₂.m * C.x + l₂.b) →  -- B and C are on l₂
  (A.y = l₃.m * A.x + l₃.b) ∧ (C.y = l₃.m * C.x + l₃.b) →  -- A and C are on l₃
  A.x + A.y = 13 ∨ A.x + A.y = 20 := by
sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_A_l1082_108294


namespace NUMINAMATH_CALUDE_power_of_product_l1082_108263

/-- For all real numbers m and n, (2m²n)³ = 8m⁶n³ -/
theorem power_of_product (m n : ℝ) : (2 * m^2 * n)^3 = 8 * m^6 * n^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1082_108263


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1082_108259

-- Define the variables
variable (P : ℝ) -- Principal amount
variable (R : ℝ) -- Original interest rate in percentage

-- Define the theorem
theorem simple_interest_problem :
  (P * (R + 3) * 2) / 100 - (P * R * 2) / 100 = 300 →
  P = 5000 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1082_108259


namespace NUMINAMATH_CALUDE_parallelogram_construction_l1082_108248

-- Define a structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a structure for a circle
structure Circle where
  center : Point2D
  radius : ℝ

-- Define the problem statement
theorem parallelogram_construction (A B C : Point2D) (r : ℝ) 
  (h1 : ∃ (circle : Circle), circle.center = A ∧ circle.radius = r ∧ 
    (B.x - A.x)^2 + (B.y - A.y)^2 ≤ r^2 ∧ 
    (C.x - A.x)^2 + (C.y - A.y)^2 ≤ r^2) :
  ∃ (D : Point2D), 
    (A.x + C.x = B.x + D.x) ∧ 
    (A.y + C.y = B.y + D.y) ∧
    (A.x - B.x = D.x - C.x) ∧ 
    (A.y - B.y = D.y - C.y) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_construction_l1082_108248


namespace NUMINAMATH_CALUDE_max_distance_MP_l1082_108245

-- Define the equilateral triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 8 ∧ dist B C = 8 ∧ dist C A = 8

-- Define the point O satisfying the given condition
def PointO (A B C O : ℝ × ℝ) : Prop :=
  (O.1 - A.1, O.2 - A.2) = 2 • (O.1 - B.1, O.2 - B.2) + 3 • (O.1 - C.1, O.2 - C.2)

-- Define a point M on the sides of triangle ABC
def PointM (A B C M : ℝ × ℝ) : Prop :=
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)) ∨
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)) ∨
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * C.1 + (1 - t) * A.1, t * C.2 + (1 - t) * A.2))

-- Define a point P such that |OP| = √19
def PointP (O P : ℝ × ℝ) : Prop :=
  dist O P = Real.sqrt 19

theorem max_distance_MP (A B C O M P : ℝ × ℝ) :
  Triangle A B C →
  PointO A B C O →
  PointM A B C M →
  PointP O P →
  (∀ M' P', PointM A B C M' → PointP O P' → dist M P ≤ dist M' P') →
  dist M P = 3 * Real.sqrt 19 :=
sorry

end NUMINAMATH_CALUDE_max_distance_MP_l1082_108245


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1082_108287

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1082_108287


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_for_q_l1082_108269

theorem condition_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (|x| = x) → (x^2 + x ≥ 0)) ∧
  (∃ x : ℝ, (x^2 + x ≥ 0) ∧ (|x| ≠ x)) :=
by sorry

end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_for_q_l1082_108269


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1082_108272

theorem coin_flip_probability : ∃ p : ℝ, 
  p > 0 ∧ p < 1 ∧ 
  p^2 + (1-p)^2 = 4*p*(1-p) ∧
  ∀ q : ℝ, (q > 0 ∧ q < 1 ∧ q^2 + (1-q)^2 = 4*q*(1-q)) → q ≤ p ∧
  p = (3 + Real.sqrt 3) / 6 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1082_108272


namespace NUMINAMATH_CALUDE_tree_spacing_l1082_108209

theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 800 →
  num_trees = 26 →
  num_trees ≥ 2 →
  (yard_length / (num_trees - 1 : ℝ)) = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l1082_108209


namespace NUMINAMATH_CALUDE_continued_fraction_value_l1082_108274

def continued_fraction (a b c d : ℚ) : ℚ :=
  -1 / (a - 1 / (b - 1 / (c - 1 / d)))

theorem continued_fraction_value : continued_fraction 2 2 2 2 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l1082_108274
