import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_roots_l528_52884

theorem sum_of_roots (r s : ℝ) : 
  (r ≠ s) → 
  (2 * (r^2 + 1/r^2) - 3 * (r + 1/r) = 1) → 
  (2 * (s^2 + 1/s^2) - 3 * (s + 1/s) = 1) → 
  (r + s = -5/2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l528_52884


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l528_52811

/-- The minimum number of additional coins needed for distribution. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let total_coins_needed := (num_friends * (num_friends + 1)) / 2
  if total_coins_needed > initial_coins then
    total_coins_needed - initial_coins
  else
    0

/-- Theorem stating the minimum number of additional coins needed for Alex's distribution. -/
theorem alex_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 80) :
  min_additional_coins num_friends initial_coins = 40 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l528_52811


namespace NUMINAMATH_CALUDE_factorial_problem_l528_52824

-- Define the factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_problem : (factorial 13 - factorial 12) / factorial 11 = 144 := by
  sorry

end NUMINAMATH_CALUDE_factorial_problem_l528_52824


namespace NUMINAMATH_CALUDE_cubic_expression_value_l528_52889

theorem cubic_expression_value (x : ℝ) (h : x^2 + x - 3 = 0) :
  x^3 + 2*x^2 - 2*x + 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l528_52889


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l528_52891

theorem negation_of_existence_proposition :
  (¬ ∃ n : ℕ, n^2 ≥ 2^n) ↔ (∀ n : ℕ, n^2 < 2^n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l528_52891


namespace NUMINAMATH_CALUDE_three_values_of_sum_l528_52850

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- Define the property that both domain and range are [a, b]
def domain_range_equal (a b : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → a ≤ f x ∧ f x ≤ b) ∧
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y)

-- Theorem stating that there are exactly 3 different values of a+b
theorem three_values_of_sum :
  ∃! (s : Finset ℝ), s.card = 3 ∧ 
  (∀ x, x ∈ s ↔ ∃ a b, domain_range_equal a b ∧ a + b = x) :=
sorry

end NUMINAMATH_CALUDE_three_values_of_sum_l528_52850


namespace NUMINAMATH_CALUDE_smallest_three_digit_middle_ring_l528_52858

/-- Checks if a number is composite -/
def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- Checks if a number can be expressed as a product of numbers from 1 to 26 -/
def is_expressible (n : ℕ) : Prop := ∃ (factors : List ℕ), (factors.all (λ x => 1 ≤ x ∧ x ≤ 26)) ∧ (factors.prod = n)

/-- The smallest three-digit middle ring number -/
def smallest_middle_ring : ℕ := 106

theorem smallest_three_digit_middle_ring :
  is_composite smallest_middle_ring ∧
  ¬(is_expressible smallest_middle_ring) ∧
  ∀ n < smallest_middle_ring, n ≥ 100 → is_composite n → is_expressible n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_middle_ring_l528_52858


namespace NUMINAMATH_CALUDE_exactly_two_true_l528_52825

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the four propositions
def Prop1 (f : ℝ → ℝ) : Prop := IsOdd f → f 0 = 0
def Prop2 (f : ℝ → ℝ) : Prop := f 0 = 0 → IsOdd f
def Prop3 (f : ℝ → ℝ) : Prop := ¬(IsOdd f) → f 0 ≠ 0
def Prop4 (f : ℝ → ℝ) : Prop := f 0 ≠ 0 → ¬(IsOdd f)

-- The main theorem
theorem exactly_two_true (f : ℝ → ℝ) : 
  IsOdd f → (Prop1 f ∧ Prop4 f ∧ ¬Prop2 f ∧ ¬Prop3 f) := by sorry

end NUMINAMATH_CALUDE_exactly_two_true_l528_52825


namespace NUMINAMATH_CALUDE_f_is_quadratic_l528_52868

/-- Definition of a quadratic equation in standard form -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l528_52868


namespace NUMINAMATH_CALUDE_square_side_equals_circle_circumference_divided_by_four_l528_52834

theorem square_side_equals_circle_circumference_divided_by_four (π : ℝ) (h : π = Real.pi) :
  let r : ℝ := 3
  let c : ℝ := 2 * π * r
  let y : ℝ := c / 4
  y = 3 * π / 2 := by
sorry

end NUMINAMATH_CALUDE_square_side_equals_circle_circumference_divided_by_four_l528_52834


namespace NUMINAMATH_CALUDE_circle_radius_l528_52805

/-- Given a circle with equation x^2 + y^2 - 4x - 2y - 5 = 0, its radius is √10 -/
theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 4*x - 2*y - 5 = 0) → 
  ∃ (center_x center_y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l528_52805


namespace NUMINAMATH_CALUDE_percentage_relation_l528_52846

theorem percentage_relation (x y z : ℝ) 
  (h1 : x = 1.3 * y) 
  (h2 : y = 0.5 * z) : 
  x = 0.65 * z := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l528_52846


namespace NUMINAMATH_CALUDE_circle_bisecting_two_circles_l528_52866

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 8)^2 = 1
def C2 (x y : ℝ) : Prop := (x - 6)^2 + (y + 6)^2 = 9

-- Define a circle C with center (a, 0) and radius r
def C (x y a r : ℝ) : Prop := (x - a)^2 + y^2 = r^2

-- Define the property of C bisecting C1 and C2
def bisects (a r : ℝ) : Prop :=
  ∀ x y : ℝ, C1 x y → (C x y a r ∨ C x y a r)
  ∧ ∀ x y : ℝ, C2 x y → (C x y a r ∨ C x y a r)

-- Theorem statement
theorem circle_bisecting_two_circles :
  ∀ a r : ℝ, bisects a r → C x y 0 9 :=
sorry

end NUMINAMATH_CALUDE_circle_bisecting_two_circles_l528_52866


namespace NUMINAMATH_CALUDE_shop_owner_profit_l528_52883

/-- Calculates the percentage profit of a shop owner who cheats while buying and selling -/
theorem shop_owner_profit (buy_cheat : ℝ) (sell_cheat : ℝ) : 
  buy_cheat = 0.12 → sell_cheat = 0.3 → 
  (((1 + buy_cheat) / (1 - sell_cheat) - 1) * 100 : ℝ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_shop_owner_profit_l528_52883


namespace NUMINAMATH_CALUDE_fraction_sum_squared_l528_52863

theorem fraction_sum_squared : 
  (2/10 + 3/100 + 5/1000 + 7/10000)^2 = 0.05555649 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_squared_l528_52863


namespace NUMINAMATH_CALUDE_number_problem_l528_52886

theorem number_problem (x : ℝ) : 0.5 * x = 0.8 * 150 + 80 → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l528_52886


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l528_52888

theorem power_fraction_simplification :
  (2^2023 + 2^2019) / (2^2023 - 2^2019) = 17 / 15 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l528_52888


namespace NUMINAMATH_CALUDE_first_business_donation_is_half_dollar_l528_52890

/-- Represents the fundraising scenario for Didi's soup kitchen --/
structure FundraisingScenario where
  num_cakes : ℕ
  slices_per_cake : ℕ
  price_per_slice : ℚ
  second_business_donation : ℚ
  total_raised : ℚ

/-- Calculates the donation per slice from the first business owner --/
def first_business_donation_per_slice (scenario : FundraisingScenario) : ℚ :=
  let total_slices := scenario.num_cakes * scenario.slices_per_cake
  let sales_revenue := total_slices * scenario.price_per_slice
  let total_business_donations := scenario.total_raised - sales_revenue
  let second_business_total := total_slices * scenario.second_business_donation
  let first_business_total := total_business_donations - second_business_total
  first_business_total / total_slices

/-- Theorem stating that the first business owner's donation per slice is $0.50 --/
theorem first_business_donation_is_half_dollar (scenario : FundraisingScenario) 
  (h1 : scenario.num_cakes = 10)
  (h2 : scenario.slices_per_cake = 8)
  (h3 : scenario.price_per_slice = 1)
  (h4 : scenario.second_business_donation = 1/4)
  (h5 : scenario.total_raised = 140) :
  first_business_donation_per_slice scenario = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_first_business_donation_is_half_dollar_l528_52890


namespace NUMINAMATH_CALUDE_expression_simplification_l528_52830

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2-x) + 4*(3+x) - 5*(2+3*x) = -6*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l528_52830


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l528_52819

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 1/12) :
  (∀ a b : ℕ+, a ≠ b → (a : ℚ)⁻¹ + (b : ℚ)⁻¹ = 1/12 → (x + y : ℕ) ≤ (a + b : ℕ)) ∧ (x + y : ℕ) = 49 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l528_52819


namespace NUMINAMATH_CALUDE_expression_factorization_l528_52851

theorem expression_factorization (x y z : ℝ) :
  29.52 * x^2 * y - y^2 * z + z^2 * x - x^2 * z + y^2 * x + z^2 * y - 2 * x * y * z =
  (y - z) * (x + y) * (x - z) := by sorry

end NUMINAMATH_CALUDE_expression_factorization_l528_52851


namespace NUMINAMATH_CALUDE_seashells_count_l528_52862

theorem seashells_count (joan_shells jessica_shells : ℕ) 
  (h1 : joan_shells = 6) 
  (h2 : jessica_shells = 8) : 
  joan_shells + jessica_shells = 14 := by
  sorry

end NUMINAMATH_CALUDE_seashells_count_l528_52862


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l528_52853

theorem product_of_sums_equals_difference_of_powers : 
  (5 + 2) * (5^3 + 2^3) * (5^9 + 2^9) * (5^27 + 2^27) * (5^81 + 2^81) = 5^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l528_52853


namespace NUMINAMATH_CALUDE_pistachio_problem_l528_52845

theorem pistachio_problem (total : ℕ) (shell_percent : ℚ) (open_percent : ℚ) 
  (h1 : total = 80)
  (h2 : shell_percent = 95 / 100)
  (h3 : open_percent = 75 / 100) :
  ⌊(shell_percent * total : ℚ) * open_percent⌋ = 57 := by
sorry

#eval ⌊(95 / 100 : ℚ) * 80 * (75 / 100 : ℚ)⌋

end NUMINAMATH_CALUDE_pistachio_problem_l528_52845


namespace NUMINAMATH_CALUDE_guppies_count_l528_52843

/-- The number of guppies Rick bought -/
def guppies : ℕ := sorry

/-- The number of clowns Tim bought -/
def clowns : ℕ := sorry

/-- The number of tetras bought -/
def tetras : ℕ := sorry

/-- The total number of animals bought -/
def total_animals : ℕ := 330

theorem guppies_count :
  (tetras = 4 * clowns) →
  (clowns = 2 * guppies) →
  (guppies + clowns + tetras = total_animals) →
  guppies = 30 := by sorry

end NUMINAMATH_CALUDE_guppies_count_l528_52843


namespace NUMINAMATH_CALUDE_f_upper_bound_l528_52840

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  f 1 = 1 ∧
  ∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂

theorem f_upper_bound (f : ℝ → ℝ) (h : f_properties f) :
  ∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x :=
sorry

end NUMINAMATH_CALUDE_f_upper_bound_l528_52840


namespace NUMINAMATH_CALUDE_maria_towel_problem_l528_52807

/-- Represents the number of towels Maria has -/
structure TowelCount where
  green : ℕ
  white : ℕ
  blue : ℕ

/-- Calculates the total number of towels -/
def TowelCount.total (t : TowelCount) : ℕ :=
  t.green + t.white + t.blue

/-- Represents the number of towels given away each day -/
structure DailyGiveaway where
  green : ℕ
  white : ℕ
  blue : ℕ

/-- Calculates the remaining towels after giving away for a number of days -/
def remainingTowels (initial : TowelCount) (daily : DailyGiveaway) (days : ℕ) : TowelCount :=
  { green := initial.green - daily.green * days,
    white := initial.white - daily.white * days,
    blue := initial.blue - daily.blue * days }

theorem maria_towel_problem :
  let initial := TowelCount.mk 35 21 15
  let daily := DailyGiveaway.mk 3 1 1
  let days := 7
  let remaining := remainingTowels initial daily days
  remaining.total = 36 := by sorry

end NUMINAMATH_CALUDE_maria_towel_problem_l528_52807


namespace NUMINAMATH_CALUDE_triangle_circles_QR_length_l528_52823

-- Define the right triangle DEF
def Triangle (DE EF DF : ℝ) := DE = 5 ∧ EF = 12 ∧ DF = 13

-- Define the circle centered at Q
def CircleQ (Q E D : ℝ × ℝ) := 
  (Q.1 - E.1)^2 + (Q.2 - E.2)^2 = (Q.1 - D.1)^2 + (Q.2 - D.2)^2

-- Define the circle centered at R
def CircleR (R D F : ℝ × ℝ) := 
  (R.1 - D.1)^2 + (R.2 - D.2)^2 = (R.1 - F.1)^2 + (R.2 - F.2)^2

-- Define the tangency conditions
def TangentQ (Q E : ℝ × ℝ) := True  -- Placeholder for tangency condition
def TangentR (R D : ℝ × ℝ) := True  -- Placeholder for tangency condition

-- State the theorem
theorem triangle_circles_QR_length 
  (D E F Q R : ℝ × ℝ) 
  (h_triangle : Triangle (dist D E) (dist E F) (dist D F))
  (h_circleQ : CircleQ Q E D)
  (h_circleR : CircleR R D F)
  (h_tangentQ : TangentQ Q E)
  (h_tangentR : TangentR R D) :
  dist Q R = 5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_circles_QR_length_l528_52823


namespace NUMINAMATH_CALUDE_integer_parabola_coeff_sum_l528_52828

/-- A parabola with integer coefficients passing through specific points -/
structure IntegerParabola where
  a : ℤ
  b : ℤ
  c : ℤ
  passes_through_origin : 1 = a * 0^2 + b * 0 + c
  passes_through_two_nine : 9 = a * 2^2 + b * 2 + c
  vertex_at_one_four : 4 = a * 1^2 + b * 1 + c

/-- The sum of coefficients of the integer parabola is 4 -/
theorem integer_parabola_coeff_sum (p : IntegerParabola) : p.a + p.b + p.c = 4 := by
  sorry


end NUMINAMATH_CALUDE_integer_parabola_coeff_sum_l528_52828


namespace NUMINAMATH_CALUDE_difference_divisible_by_nine_l528_52801

theorem difference_divisible_by_nine (a b : ℤ) : 
  ∃ k : ℤ, (3 * a + 2)^2 - (3 * b + 2)^2 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_difference_divisible_by_nine_l528_52801


namespace NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l528_52804

theorem min_values_ab_and_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2 * a + b) :
  (∀ x y, x > 0 → y > 0 → x * y = 2 * x + y → a * b ≤ x * y) ∧
  (∀ x y, x > 0 → y > 0 → x * y = 2 * x + y → a + 2 * b ≤ x + 2 * y) ∧
  a * b = 8 ∧ a + 2 * b = 9 := by
sorry

end NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l528_52804


namespace NUMINAMATH_CALUDE_max_value_polynomial_l528_52810

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (max : ℝ), ∀ (a b : ℝ), a + b = 5 → 
    a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 ≤ max) ∧
  (x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ 6084/17) ∧
  (∃ (x₀ y₀ : ℝ), x₀ + y₀ = 5 ∧ 
    x₀^4*y₀ + x₀^3*y₀ + x₀^2*y₀ + x₀*y₀ + x₀*y₀^2 + x₀*y₀^3 + x₀*y₀^4 = 6084/17) :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l528_52810


namespace NUMINAMATH_CALUDE_eight_digit_integers_count_l528_52864

/-- The number of choices for the first digit -/
def first_digit_choices : ℕ := 9

/-- The number of choices for each of the remaining seven digits -/
def remaining_digit_choices : ℕ := 5

/-- The number of remaining digits -/
def remaining_digits : ℕ := 7

/-- The total number of different 8-digit positive integers under the given conditions -/
def total_combinations : ℕ := first_digit_choices * remaining_digit_choices ^ remaining_digits

theorem eight_digit_integers_count : total_combinations = 703125 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_integers_count_l528_52864


namespace NUMINAMATH_CALUDE_hyperbola_distance_inequality_l528_52865

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the left focus
def left_focus : ℝ × ℝ := sorry

-- Define a point on the right branch of the hyperbola
def right_branch_point (P : ℝ × ℝ) : Prop := 
  hyperbola P.1 P.2 ∧ P.1 > 0

-- State the theorem
theorem hyperbola_distance_inequality 
  (P₁ P₂ : ℝ × ℝ) 
  (h₁ : right_branch_point P₁) 
  (h₂ : right_branch_point P₂) : 
  dist left_focus P₁ + dist left_focus P₂ - dist P₁ P₂ ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_distance_inequality_l528_52865


namespace NUMINAMATH_CALUDE_katies_friends_games_l528_52867

/-- Given that Katie has 57 new games and 39 old games, and she has 62 more games than her friends,
    prove that her friends have 34 new games. -/
theorem katies_friends_games (katie_new : ℕ) (katie_old : ℕ) (difference : ℕ) 
    (h1 : katie_new = 57)
    (h2 : katie_old = 39)
    (h3 : difference = 62) :
  katie_new + katie_old - difference = 34 := by
  sorry

end NUMINAMATH_CALUDE_katies_friends_games_l528_52867


namespace NUMINAMATH_CALUDE_smallest_m_divisibility_l528_52812

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_m_divisibility : 
  ∃ (M : ℕ), 
    (M > 0) ∧ 
    (is_divisible M (7^2) ∨ is_divisible (M+1) (7^2) ∨ is_divisible (M+2) (7^2)) ∧
    (is_divisible M (5^2) ∨ is_divisible (M+1) (5^2) ∨ is_divisible (M+2) (5^2)) ∧
    (is_divisible M (3^2) ∨ is_divisible (M+1) (3^2) ∨ is_divisible (M+2) (3^2)) ∧
    (∀ (N : ℕ), N < M →
      ¬((is_divisible N (7^2) ∨ is_divisible (N+1) (7^2) ∨ is_divisible (N+2) (7^2)) ∧
        (is_divisible N (5^2) ∨ is_divisible (N+1) (5^2) ∨ is_divisible (N+2) (5^2)) ∧
        (is_divisible N (3^2) ∨ is_divisible (N+1) (3^2) ∨ is_divisible (N+2) (3^2)))) ∧
    M = 98 :=
by sorry


end NUMINAMATH_CALUDE_smallest_m_divisibility_l528_52812


namespace NUMINAMATH_CALUDE_probability_abs_diff_gt_half_l528_52870

/-- A coin flip result -/
inductive CoinFlip
| Heads
| Tails

/-- The result of the number selection process -/
inductive NumberSelection
| Uniform : ℝ → NumberSelection
| Zero
| One

/-- The process of selecting a number based on coin flips -/
def selectNumber (flip1 : CoinFlip) (flip2 : CoinFlip) (u : ℝ) : NumberSelection :=
  match flip1 with
  | CoinFlip.Heads => match flip2 with
    | CoinFlip.Heads => NumberSelection.Zero
    | CoinFlip.Tails => NumberSelection.One
  | CoinFlip.Tails => NumberSelection.Uniform u

/-- The probability measure for the problem -/
noncomputable def P : Set (NumberSelection × NumberSelection) → ℝ := sorry

/-- The event that |x-y| > 1/2 -/
def event : Set (NumberSelection × NumberSelection) :=
  {pair | let (x, y) := pair
          match x, y with
          | NumberSelection.Uniform x', NumberSelection.Uniform y' => |x' - y'| > 1/2
          | NumberSelection.Zero, NumberSelection.Uniform y' => y' < 1/2
          | NumberSelection.One, NumberSelection.Uniform y' => y' < 1/2
          | NumberSelection.Uniform x', NumberSelection.Zero => x' > 1/2
          | NumberSelection.Uniform x', NumberSelection.One => x' < 1/2
          | NumberSelection.Zero, NumberSelection.One => true
          | NumberSelection.One, NumberSelection.Zero => true
          | _, _ => false}

theorem probability_abs_diff_gt_half :
  P event = 7/16 := by sorry

end NUMINAMATH_CALUDE_probability_abs_diff_gt_half_l528_52870


namespace NUMINAMATH_CALUDE_fifteen_blue_points_l528_52895

/-- Represents the configuration of points on a line -/
structure LineConfiguration where
  red_points : Fin 2 → ℕ
  blue_left : Fin 2 → ℕ
  blue_right : Fin 2 → ℕ

/-- The number of segments containing a red point with blue endpoints -/
def segments_count (config : LineConfiguration) (i : Fin 2) : ℕ :=
  config.blue_left i * config.blue_right i

/-- The total number of blue points -/
def total_blue_points (config : LineConfiguration) : ℕ :=
  config.blue_left 0 + config.blue_right 0

/-- Theorem stating that there are exactly 15 blue points -/
theorem fifteen_blue_points (config : LineConfiguration) 
  (h1 : segments_count config 0 = 56)
  (h2 : segments_count config 1 = 50)
  (h3 : config.blue_left 0 + config.blue_right 0 = config.blue_left 1 + config.blue_right 1) :
  total_blue_points config = 15 := by
  sorry


end NUMINAMATH_CALUDE_fifteen_blue_points_l528_52895


namespace NUMINAMATH_CALUDE_average_speed_calculation_l528_52842

def initial_reading : ℕ := 3223
def final_reading : ℕ := 3443
def total_time : ℕ := 12

def distance : ℕ := final_reading - initial_reading

def average_speed : ℚ := distance / total_time

theorem average_speed_calculation : 
  (average_speed : ℚ) = 55/3 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l528_52842


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l528_52808

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + a| - x - 2

-- Theorem for part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 0} = {x : ℝ | x < 0 ∨ x > 2} := by sorry

-- Theorem for part II
theorem range_of_a (a : ℝ) (h : a > -1) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-a) 1 ∧ f a x₀ ≤ 0) →
  a ∈ Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l528_52808


namespace NUMINAMATH_CALUDE_proposition_truth_l528_52837

theorem proposition_truth (x y : ℝ) : x + y ≥ 5 → x ≥ 3 ∨ y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l528_52837


namespace NUMINAMATH_CALUDE_surface_area_of_S_l528_52852

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents the solid S' formed by removing a tunnel from the cube -/
structure Solid where
  cube : Cube
  tunnelStart : Point3D
  tunnelEnd : Point3D

/-- Calculate the surface area of the solid S' -/
def surfaceAreaS' (s : Solid) : ℝ :=
  sorry

theorem surface_area_of_S' (c : Cube) (e i j k : Point3D) :
  c.sideLength = 12 ∧
  e.x = 12 ∧ e.y = 12 ∧ e.z = 12 ∧
  i.x = 9 ∧ i.y = 12 ∧ i.z = 12 ∧
  j.x = 12 ∧ j.y = 9 ∧ j.z = 12 ∧
  k.x = 12 ∧ k.y = 12 ∧ k.z = 9 →
  surfaceAreaS' { cube := c, tunnelStart := i, tunnelEnd := k } = 840 + 45 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_S_l528_52852


namespace NUMINAMATH_CALUDE_banana_consumption_l528_52878

theorem banana_consumption (n : ℕ) (a : ℝ) (h1 : n = 7) (h2 : a > 0) : 
  (a * (2^(n-1))) = 128 ∧ 
  (a * (2^n - 1)) / (2 - 1) = 254 → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_banana_consumption_l528_52878


namespace NUMINAMATH_CALUDE_concyclic_AQTP_l528_52826

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the properties and relations
variable (on_circle : Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Point → Prop)
variable (collinear : Point → Point → Point → Prop)
variable (tangent_intersection : Circle → Point → Point → Point → Prop)
variable (concyclic : Point → Point → Point → Point → Prop)

-- State the theorem
theorem concyclic_AQTP 
  (Γ₁ Γ₂ : Circle) 
  (A B P Q T : Point) :
  intersect Γ₁ Γ₂ A B →
  on_circle P Γ₁ →
  on_circle Q Γ₂ →
  collinear P B Q →
  tangent_intersection Γ₂ P Q T →
  concyclic A Q T P :=
sorry

end NUMINAMATH_CALUDE_concyclic_AQTP_l528_52826


namespace NUMINAMATH_CALUDE_factor_expression_l528_52803

theorem factor_expression (x : ℝ) : 12 * x^2 + 8 * x = 4 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l528_52803


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l528_52847

theorem min_value_sqrt_sum (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 1)^2 + (x + 2)^2) ≥ Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l528_52847


namespace NUMINAMATH_CALUDE_third_month_sale_is_10389_l528_52841

/-- Calculates the sale in the third month given the sales for other months and the average -/
def third_month_sale (sale1 sale2 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale4 + sale5 + sale6)

/-- Proves that the sale in the third month is 10389 given the conditions -/
theorem third_month_sale_is_10389 :
  third_month_sale 4000 6524 7230 6000 12557 7000 = 10389 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_is_10389_l528_52841


namespace NUMINAMATH_CALUDE_at_least_one_fraction_less_than_two_l528_52869

theorem at_least_one_fraction_less_than_two (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_fraction_less_than_two_l528_52869


namespace NUMINAMATH_CALUDE_skirt_width_is_four_l528_52816

-- Define the width of the rectangle for each skirt
def width : ℝ := sorry

-- Define the length of the rectangle for each skirt
def length : ℝ := 12

-- Define the number of skirts
def num_skirts : ℕ := 3

-- Define the area of material for the bodice
def bodice_area : ℝ := 2 + 2 * 5

-- Define the cost per square foot of material
def cost_per_sqft : ℝ := 3

-- Define the total cost of material
def total_cost : ℝ := 468

-- Theorem statement
theorem skirt_width_is_four :
  width = 4 ∧
  length * width * num_skirts + bodice_area = total_cost / cost_per_sqft :=
sorry

end NUMINAMATH_CALUDE_skirt_width_is_four_l528_52816


namespace NUMINAMATH_CALUDE_ryegrass_percentage_in_y_l528_52896

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
structure FinalMixture where
  x_percentage : ℝ
  y_percentage : ℝ
  ryegrass_percentage : ℝ

/-- Theorem stating the percentage of ryegrass in seed mixture Y -/
theorem ryegrass_percentage_in_y
  (x : SeedMixture)
  (y : SeedMixture)
  (final : FinalMixture)
  (hx_ryegrass : x.ryegrass = 0.4)
  (hx_bluegrass : x.bluegrass = 0.6)
  (hy_fescue : y.fescue = 0.75)
  (hfinal_x : final.x_percentage = 0.13333333333333332)
  (hfinal_y : final.y_percentage = 1 - final.x_percentage)
  (hfinal_ryegrass : final.ryegrass_percentage = 0.27)
  : y.ryegrass = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ryegrass_percentage_in_y_l528_52896


namespace NUMINAMATH_CALUDE_inequality_relations_l528_52879

theorem inequality_relations (a b : ℝ) (h : a > b) : (a - 3 > b - 3) ∧ (-4 * a < -4 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relations_l528_52879


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l528_52809

def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l528_52809


namespace NUMINAMATH_CALUDE_average_pqr_l528_52880

theorem average_pqr (p q r : ℝ) (h : (5 / 4) * (p + q + r) = 15) :
  (p + q + r) / 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_average_pqr_l528_52880


namespace NUMINAMATH_CALUDE_revenue_maximized_at_20_l528_52899

-- Define the revenue function
def R (p : ℝ) : ℝ := p * (160 - 4 * p)

-- State the theorem
theorem revenue_maximized_at_20 :
  ∃ (p_max : ℝ), p_max ≤ 40 ∧ 
  ∀ (p : ℝ), p ≤ 40 → R p ≤ R p_max ∧
  p_max = 20 := by
  sorry

end NUMINAMATH_CALUDE_revenue_maximized_at_20_l528_52899


namespace NUMINAMATH_CALUDE_rhombus_diagonals_l528_52818

/-- A rhombus with side length 1 and one angle of 120° has diagonals of length 1 and √3. -/
theorem rhombus_diagonals (s : ℝ) (α : ℝ) (d₁ d₂ : ℝ) 
  (h_side : s = 1)
  (h_angle : α = 120 * π / 180) :
  d₁ = 1 ∧ d₂ = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_diagonals_l528_52818


namespace NUMINAMATH_CALUDE_star_diameter_scientific_notation_l528_52814

/-- Represents the diameter of the star in meters -/
def star_diameter : ℝ := 16600000000

/-- Represents the coefficient in scientific notation -/
def coefficient : ℝ := 1.66

/-- Represents the exponent in scientific notation -/
def exponent : ℕ := 10

/-- Theorem stating that the star's diameter is correctly expressed in scientific notation -/
theorem star_diameter_scientific_notation : 
  star_diameter = coefficient * (10 : ℝ) ^ exponent := by
  sorry

end NUMINAMATH_CALUDE_star_diameter_scientific_notation_l528_52814


namespace NUMINAMATH_CALUDE_median_equal_mean_l528_52855

def set_elements (n : ℝ) : List ℝ := [n, n+4, n+7, n+10, n+14]

theorem median_equal_mean (n : ℝ) (h : n + 7 = 14) : 
  (List.sum (set_elements n)) / (List.length (set_elements n)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_median_equal_mean_l528_52855


namespace NUMINAMATH_CALUDE_fraction_multiplication_equality_l528_52882

theorem fraction_multiplication_equality : 
  (5 / 8 : ℚ)^2 * (3 / 4 : ℚ)^2 * (2 / 3 : ℚ) = 75 / 512 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_equality_l528_52882


namespace NUMINAMATH_CALUDE_cars_return_to_start_l528_52871

/-- Represents the state of cars on a circular track -/
def TrackState (n : ℕ) := Fin n → Fin n

/-- The permutation of car positions after one hour -/
def hourlyPermutation (n : ℕ) : TrackState n → TrackState n := sorry

/-- Theorem: There exists a time when all cars return to their original positions -/
theorem cars_return_to_start (n : ℕ) : 
  ∃ d : ℕ+, ∀ initial : TrackState n, (hourlyPermutation n)^[d] initial = initial := by
  sorry


end NUMINAMATH_CALUDE_cars_return_to_start_l528_52871


namespace NUMINAMATH_CALUDE_ratio_equality_l528_52815

theorem ratio_equality (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) : 
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l528_52815


namespace NUMINAMATH_CALUDE_factor_expression_l528_52897

theorem factor_expression (b : ℝ) : 294 * b^3 + 63 * b^2 - 21 * b = 21 * b * (14 * b^2 + 3 * b - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l528_52897


namespace NUMINAMATH_CALUDE_exist_ten_special_integers_l528_52849

theorem exist_ten_special_integers : 
  ∃ (a : Fin 10 → ℕ+), 
    (∀ i j, i ≠ j → ¬(a i ∣ a j)) ∧ 
    (∀ i j, (a i)^2 ∣ a j) := by
  sorry

end NUMINAMATH_CALUDE_exist_ten_special_integers_l528_52849


namespace NUMINAMATH_CALUDE_largest_base_sum_not_sixteen_l528_52860

/-- Represents a number in a given base --/
structure BaseNumber (base : ℕ) where
  digits : List ℕ
  valid : ∀ d ∈ digits, d < base

/-- Computes the sum of digits of a BaseNumber --/
def sumOfDigits {base : ℕ} (n : BaseNumber base) : ℕ :=
  n.digits.sum

/-- Represents 11^4 in different bases --/
def elevenFourth (base : ℕ) : BaseNumber base :=
  if base ≥ 7 then
    ⟨[1, 4, 6, 4, 1], sorry⟩
  else if base = 6 then
    ⟨[1, 5, 0, 4, 1], sorry⟩
  else
    ⟨[], sorry⟩  -- Undefined for bases less than 6

/-- The theorem to be proved --/
theorem largest_base_sum_not_sixteen :
  (∃ b : ℕ, b > 0 ∧ sumOfDigits (elevenFourth b) ≠ 16) ∧
  (∀ b : ℕ, b > 6 → sumOfDigits (elevenFourth b) = 16) :=
sorry

end NUMINAMATH_CALUDE_largest_base_sum_not_sixteen_l528_52860


namespace NUMINAMATH_CALUDE_system_solutions_l528_52832

theorem system_solutions : 
  (∃ (x y : ℝ), x * (y - 1) + y * (x + 1) = 6 ∧ (x - 1) * (y + 1) = 1) ∧
  (∀ (x y : ℝ), x * (y - 1) + y * (x + 1) = 6 ∧ (x - 1) * (y + 1) = 1 → 
    (x = 4/3 ∧ y = 2) ∨ (x = -2 ∧ y = -4/3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l528_52832


namespace NUMINAMATH_CALUDE_printer_X_time_l528_52831

/-- The time it takes for printer X to complete the job alone -/
def T_x : ℝ := 16

/-- The time it takes for printer Y to complete the job alone -/
def T_y : ℝ := 10

/-- The time it takes for printer Z to complete the job alone -/
def T_z : ℝ := 20

/-- The ratio of X's time to Y and Z's combined time -/
def ratio : ℝ := 2.4

theorem printer_X_time :
  T_x = 16 ∧ 
  T_y = 10 ∧ 
  T_z = 20 ∧ 
  ratio = 2.4 →
  T_x = ratio * (1 / (1 / T_y + 1 / T_z)) :=
by sorry

end NUMINAMATH_CALUDE_printer_X_time_l528_52831


namespace NUMINAMATH_CALUDE_g_difference_at_3_and_neg_3_l528_52892

def g (x : ℝ) : ℝ := x^6 + 5*x^2 + 3*x

theorem g_difference_at_3_and_neg_3 : g 3 - g (-3) = 18 := by
  sorry

end NUMINAMATH_CALUDE_g_difference_at_3_and_neg_3_l528_52892


namespace NUMINAMATH_CALUDE_gravel_weight_l528_52839

/-- Proves that the weight of gravel in a cement mixture is 10 pounds given the specified conditions. -/
theorem gravel_weight (total_weight : ℝ) (sand_fraction : ℝ) (water_fraction : ℝ) :
  total_weight = 23.999999999999996 →
  sand_fraction = 1 / 3 →
  water_fraction = 1 / 4 →
  total_weight - (sand_fraction * total_weight + water_fraction * total_weight) = 10 := by
  sorry

end NUMINAMATH_CALUDE_gravel_weight_l528_52839


namespace NUMINAMATH_CALUDE_reimbursement_calculation_l528_52854

/-- Calculates the total reimbursement for a sales rep based on daily mileage -/
def total_reimbursement (rate : ℚ) (miles : List ℚ) : ℚ :=
  (miles.map (· * rate)).sum

/-- Proves that the total reimbursement for the given mileage and rate is $36.00 -/
theorem reimbursement_calculation : 
  let rate : ℚ := 36 / 100
  let daily_miles : List ℚ := [18, 26, 20, 20, 16]
  total_reimbursement rate daily_miles = 36 := by
  sorry

#eval total_reimbursement (36 / 100) [18, 26, 20, 20, 16]

end NUMINAMATH_CALUDE_reimbursement_calculation_l528_52854


namespace NUMINAMATH_CALUDE_egg_difference_solution_l528_52893

/-- Represents the problem of calculating the difference between eggs in perfect condition
    in undropped trays and cracked eggs in dropped trays. -/
def egg_difference_problem (total_eggs : ℕ) (num_trays : ℕ) (dropped_trays : ℕ)
  (first_tray_capacity : ℕ) (second_tray_capacity : ℕ) (third_tray_capacity : ℕ)
  (first_tray_cracked : ℕ) (second_tray_cracked : ℕ) (third_tray_cracked : ℕ) : Prop :=
  let total_dropped_capacity := first_tray_capacity + second_tray_capacity + third_tray_capacity
  let undropped_eggs := total_eggs - total_dropped_capacity
  let total_cracked := first_tray_cracked + second_tray_cracked + third_tray_cracked
  undropped_eggs - total_cracked = 8

/-- The main theorem stating the solution to the egg problem. -/
theorem egg_difference_solution :
  egg_difference_problem 60 5 3 15 12 10 7 5 3 := by
  sorry

end NUMINAMATH_CALUDE_egg_difference_solution_l528_52893


namespace NUMINAMATH_CALUDE_inequality_proof_l528_52835

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 / b + b^2 / c + c^2 / a ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l528_52835


namespace NUMINAMATH_CALUDE_triangle_side_length_l528_52856

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively.
    Prove that if c = 10, A = 45°, and C = 30°, then b = 5(√6 + √2). -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  c = 10 → A = π/4 → C = π/6 → b = 5 * (Real.sqrt 6 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l528_52856


namespace NUMINAMATH_CALUDE_veridux_female_managers_l528_52874

/-- Calculates the number of female managers given the total number of employees,
    female employees, total managers, and male associates. -/
def female_managers (total_employees : ℕ) (female_employees : ℕ) (total_managers : ℕ) (male_associates : ℕ) : ℕ :=
  total_managers - (total_employees - female_employees - male_associates)

/-- Theorem stating that given the conditions from the problem, 
    the number of female managers is 40. -/
theorem veridux_female_managers :
  female_managers 250 90 40 160 = 40 := by
  sorry

#eval female_managers 250 90 40 160

end NUMINAMATH_CALUDE_veridux_female_managers_l528_52874


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l528_52800

/-- The constant k in the inverse variation relationship -/
def k : ℝ := 192

/-- The relationship between z and x -/
def relation (z x : ℝ) : Prop := 3 * z = k / (x^3)

theorem inverse_variation_problem (z₁ z₂ x₁ x₂ : ℝ) 
  (h₁ : relation z₁ x₁)
  (h₂ : z₁ = 8)
  (h₃ : x₁ = 2)
  (h₄ : x₂ = 4) :
  z₂ = 1 ∧ relation z₂ x₂ := by
  sorry


end NUMINAMATH_CALUDE_inverse_variation_problem_l528_52800


namespace NUMINAMATH_CALUDE_circle_radius_l528_52820

/-- Given a circle with diameter 26 centimeters, prove that its radius is 13 centimeters. -/
theorem circle_radius (diameter : ℝ) (h : diameter = 26) : diameter / 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l528_52820


namespace NUMINAMATH_CALUDE_expression_evaluation_l528_52872

theorem expression_evaluation :
  ∃ (m : ℕ+), (3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2 = m * 10^1003 ∧ m = 56 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l528_52872


namespace NUMINAMATH_CALUDE_sum_of_powers_l528_52877

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^10 + b^10 = 123 -/
theorem sum_of_powers (a b : ℝ) 
  (h1 : a = Real.sqrt 6)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^4 + b^4 = 7)
  (h4 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l528_52877


namespace NUMINAMATH_CALUDE_triangle_equation_implies_right_triangle_l528_52857

/-- A triangle with side lengths satisfying a certain equation is a right triangle -/
theorem triangle_equation_implies_right_triangle 
  (a b c : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq : a^3 + (Real.sqrt 2 / 4) * b^3 + (Real.sqrt 3 / 9) * c^3 - (Real.sqrt 6 / 2) * a * b * c = 0) :
  a^2 + b^2 = c^2 := by
  sorry

#check triangle_equation_implies_right_triangle

end NUMINAMATH_CALUDE_triangle_equation_implies_right_triangle_l528_52857


namespace NUMINAMATH_CALUDE_tea_party_wait_time_l528_52836

/-- Mad Hatter's clock speed relative to real time -/
def mad_hatter_clock_speed : ℚ := 5/4

/-- March Hare's clock speed relative to real time -/
def march_hare_clock_speed : ℚ := 5/6

/-- The agreed meeting time on their clocks (in hours after noon) -/
def meeting_time : ℚ := 5

/-- Calculate the real time when someone arrives based on their clock speed -/
def real_arrival_time (clock_speed : ℚ) : ℚ :=
  meeting_time / clock_speed

theorem tea_party_wait_time :
  real_arrival_time march_hare_clock_speed - real_arrival_time mad_hatter_clock_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_tea_party_wait_time_l528_52836


namespace NUMINAMATH_CALUDE_compound_oxygen_count_l528_52833

/-- Represents the number of atoms of an element in a compound -/
@[ext] structure AtomCount where
  al : ℕ
  o : ℕ
  h : ℕ

/-- Calculates the molecular weight of a compound given its atom counts -/
def molecularWeight (atoms : AtomCount) : ℕ :=
  27 * atoms.al + 16 * atoms.o + atoms.h

/-- Theorem stating that a compound with 1 Al, 3 H, and molecular weight 78 has 3 O atoms -/
theorem compound_oxygen_count :
  ∃ (atoms : AtomCount),
    atoms.al = 1 ∧
    atoms.h = 3 ∧
    molecularWeight atoms = 78 ∧
    atoms.o = 3 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_count_l528_52833


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l528_52875

/-- Given an agricultural experiment with two plots of seeds, calculate the percentage of total seeds that germinated. -/
theorem seed_germination_percentage 
  (seeds_plot1 : ℕ) 
  (seeds_plot2 : ℕ) 
  (germination_rate_plot1 : ℚ) 
  (germination_rate_plot2 : ℚ) 
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 25 / 100)
  (h4 : germination_rate_plot2 = 40 / 100) :
  (((seeds_plot1 : ℚ) * germination_rate_plot1 + (seeds_plot2 : ℚ) * germination_rate_plot2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) = 31 / 100 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l528_52875


namespace NUMINAMATH_CALUDE_chocolate_bars_distribution_l528_52859

theorem chocolate_bars_distribution (large_box_total : ℕ) (small_boxes : ℕ) (bars_per_small_box : ℕ) :
  large_box_total = 375 →
  small_boxes = 15 →
  large_box_total = small_boxes * bars_per_small_box →
  bars_per_small_box = 25 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_distribution_l528_52859


namespace NUMINAMATH_CALUDE_jacket_price_before_tax_l528_52898

def initial_amount : ℚ := 13.99
def shirt_price : ℚ := 12.14
def discount_rate : ℚ := 0.05
def additional_money : ℚ := 7.43
def tax_rate : ℚ := 0.10

def discounted_shirt_price : ℚ := shirt_price * (1 - discount_rate)
def money_left : ℚ := initial_amount + additional_money - discounted_shirt_price

theorem jacket_price_before_tax :
  ∃ (x : ℚ), x * (1 + tax_rate) = money_left ∧ x = 8.99 := by sorry

end NUMINAMATH_CALUDE_jacket_price_before_tax_l528_52898


namespace NUMINAMATH_CALUDE_equal_coin_count_theorem_l528_52885

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | HalfDollar
  | OneDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .HalfDollar => 50
  | .OneDollar => 100

/-- The total value of coins in cents --/
def totalValue : ℕ := 332

/-- The number of different coin types --/
def numCoinTypes : ℕ := 5

theorem equal_coin_count_theorem :
  ∃ (n : ℕ), 
    n > 0 ∧ 
    n * (coinValue CoinType.Penny + coinValue CoinType.Nickel + 
         coinValue CoinType.Dime + coinValue CoinType.HalfDollar + 
         coinValue CoinType.OneDollar) = totalValue ∧
    n * numCoinTypes = 10 := by
  sorry

end NUMINAMATH_CALUDE_equal_coin_count_theorem_l528_52885


namespace NUMINAMATH_CALUDE_fixed_points_sum_zero_l528_52894

open Real

/-- The sum of fixed points of natural logarithm and exponential functions is zero -/
theorem fixed_points_sum_zero :
  ∃ t₁ t₂ : ℝ, 
    (exp t₁ = -t₁) ∧ 
    (log t₂ = -t₂) ∧ 
    (t₁ + t₂ = 0) := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_sum_zero_l528_52894


namespace NUMINAMATH_CALUDE_hexagon_area_in_circle_l528_52876

/-- The area of a regular hexagon inscribed in a circle with radius 2 units is 6√3 square units. -/
theorem hexagon_area_in_circle (r : ℝ) (h : r = 2) : 
  let hexagon_area := 6 * (r^2 * Real.sqrt 3 / 4)
  hexagon_area = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_in_circle_l528_52876


namespace NUMINAMATH_CALUDE_bird_flight_theorem_l528_52829

/-- The height of the church tower in feet -/
def church_height : ℝ := 150

/-- The height of the catholic tower in feet -/
def catholic_height : ℝ := 200

/-- The distance between the two towers in feet -/
def tower_distance : ℝ := 350

/-- The distance of the grain from the church tower in feet -/
def grain_distance : ℝ := 200

theorem bird_flight_theorem :
  ∀ (x : ℝ),
  (x^2 + church_height^2 = (tower_distance - x)^2 + catholic_height^2) →
  x = grain_distance :=
by sorry

end NUMINAMATH_CALUDE_bird_flight_theorem_l528_52829


namespace NUMINAMATH_CALUDE_minimal_polynomial_reciprocal_l528_52844

theorem minimal_polynomial_reciprocal (x : ℂ) 
  (h1 : x^9 = 1) 
  (h2 : x^3 ≠ 1) : 
  x^5 - x^4 + x^3 = 1 / (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_minimal_polynomial_reciprocal_l528_52844


namespace NUMINAMATH_CALUDE_vehicle_ownership_l528_52821

theorem vehicle_ownership (total_adults : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ) (bicycle_owners : ℕ)
  (car_and_motorcycle : ℕ) (motorcycle_and_bicycle : ℕ) (car_and_bicycle : ℕ)
  (h1 : total_adults = 500)
  (h2 : car_owners = 400)
  (h3 : motorcycle_owners = 200)
  (h4 : bicycle_owners = 150)
  (h5 : car_and_motorcycle = 100)
  (h6 : motorcycle_and_bicycle = 50)
  (h7 : car_and_bicycle = 30)
  (h8 : total_adults ≤ car_owners + motorcycle_owners + bicycle_owners - car_and_motorcycle - motorcycle_and_bicycle - car_and_bicycle) :
  car_owners - car_and_motorcycle - car_and_bicycle = 270 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_ownership_l528_52821


namespace NUMINAMATH_CALUDE_roots_of_equation_l528_52838

theorem roots_of_equation (x : ℝ) : (x + 1)^2 = 0 ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l528_52838


namespace NUMINAMATH_CALUDE_expression_evaluation_l528_52881

theorem expression_evaluation (a : ℝ) (h : a^2 + a = 6) :
  (a^2 - 2*a) / (a^2 - 1) / (a - 1 - (2*a - 1) / (a + 1)) = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l528_52881


namespace NUMINAMATH_CALUDE_modulus_of_neg_one_plus_i_l528_52813

theorem modulus_of_neg_one_plus_i :
  Complex.abs (-1 + Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_neg_one_plus_i_l528_52813


namespace NUMINAMATH_CALUDE_divisor_product_1024_implies_16_l528_52806

/-- Given a positive integer n, returns the product of all its positive integer divisors. -/
def divisorProduct (n : ℕ+) : ℕ+ :=
  sorry

/-- Theorem: If the product of the positive integer divisors of n is 1024, then n = 16. -/
theorem divisor_product_1024_implies_16 (n : ℕ+) :
  divisorProduct n = 1024 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_divisor_product_1024_implies_16_l528_52806


namespace NUMINAMATH_CALUDE_initial_roses_count_l528_52822

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of roses added to the vase -/
def added_roses : ℕ := 10

/-- The total number of roses after adding -/
def total_roses : ℕ := 16

/-- Theorem stating that the initial number of roses is 6 -/
theorem initial_roses_count : initial_roses = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_roses_count_l528_52822


namespace NUMINAMATH_CALUDE_race_probability_l528_52802

theorem race_probability (pA pB pC pD pE : ℚ) 
  (hA : pA = 1/4) 
  (hB : pB = 1/8) 
  (hC : pC = 1/12) 
  (hD : pD = 1/20) 
  (hE : pE = 1/30) : 
  pA + pB + pC + pD + pE = 65/120 := by
sorry

end NUMINAMATH_CALUDE_race_probability_l528_52802


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l528_52861

theorem square_area_equal_perimeter_triangle (a b c : Real) (h1 : a = 7.5) (h2 : b = 5.3) (h3 : c = 11.2) :
  let triangle_perimeter := a + b + c
  let square_side := triangle_perimeter / 4
  square_side ^ 2 = 36 := by sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l528_52861


namespace NUMINAMATH_CALUDE_right_triangle_perfect_square_l528_52873

theorem right_triangle_perfect_square (a b c : ℕ) : 
  Prime a →
  a^2 + b^2 = c^2 →
  ∃ (n : ℕ), 2 * (a + b + 1) = n^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perfect_square_l528_52873


namespace NUMINAMATH_CALUDE_product_digit_exclusion_l528_52887

theorem product_digit_exclusion : ∃ d : ℕ, d < 10 ∧ 
  (32 % 10 ≠ d) ∧ ((1024 / 32) % 10 ≠ d) := by
  sorry

end NUMINAMATH_CALUDE_product_digit_exclusion_l528_52887


namespace NUMINAMATH_CALUDE_ninety_six_times_one_hundred_four_l528_52817

theorem ninety_six_times_one_hundred_four : 96 * 104 = 9984 := by
  sorry

end NUMINAMATH_CALUDE_ninety_six_times_one_hundred_four_l528_52817


namespace NUMINAMATH_CALUDE_parallelogram_area_from_side_and_diagonals_l528_52848

/-- The area of a parallelogram given one side and two diagonals -/
theorem parallelogram_area_from_side_and_diagonals
  (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ)
  (h_side : side = 51)
  (h_diag1 : diagonal1 = 40)
  (h_diag2 : diagonal2 = 74) :
  let s := (side + diagonal1 / 2 + diagonal2 / 2) / 2
  4 * Real.sqrt (s * (s - side) * (s - diagonal1 / 2) * (s - diagonal2 / 2)) = 1224 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_from_side_and_diagonals_l528_52848


namespace NUMINAMATH_CALUDE_independence_test_most_appropriate_l528_52827

/-- Represents the survey data --/
structure SurveyData where
  male_total : Nat
  male_opposing : Nat
  female_total : Nat
  female_opposing : Nat
  deriving Repr

/-- Represents different statistical methods --/
inductive StatMethod
  | Mean
  | Regression
  | IndependenceTest
  | Probability
  deriving Repr

/-- Determines the most appropriate method for analyzing the relationship
    between gender and judgment in the survey --/
def most_appropriate_method (data : SurveyData) : StatMethod :=
  StatMethod.IndependenceTest

/-- Theorem stating that the independence test is the most appropriate method
    for the given survey data --/
theorem independence_test_most_appropriate (data : SurveyData) 
    (h1 : data.male_total = 2548)
    (h2 : data.male_opposing = 1560)
    (h3 : data.female_total = 2452)
    (h4 : data.female_opposing = 1200) :
    most_appropriate_method data = StatMethod.IndependenceTest := by
  sorry


end NUMINAMATH_CALUDE_independence_test_most_appropriate_l528_52827
