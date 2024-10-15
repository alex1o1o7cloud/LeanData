import Mathlib

namespace NUMINAMATH_CALUDE_binomial_seven_two_l964_96430

theorem binomial_seven_two : Nat.choose 7 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_two_l964_96430


namespace NUMINAMATH_CALUDE_fraction_difference_simplest_form_l964_96467

theorem fraction_difference_simplest_form :
  let a := 5
  let b := 19
  let c := 2
  let d := 23
  let numerator := a * d - c * b
  let denominator := b * d
  (numerator : ℚ) / denominator = 77 / 437 ∧
  ∀ (x y : ℤ), x ≠ 0 → (77 : ℚ) / 437 = (x : ℚ) / y → (x = 77 ∧ y = 437 ∨ x = -77 ∧ y = -437) :=
by sorry

end NUMINAMATH_CALUDE_fraction_difference_simplest_form_l964_96467


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l964_96451

theorem adult_ticket_cost (child_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (adult_count : ℕ)
  (h1 : child_cost = 6)
  (h2 : total_tickets = 225)
  (h3 : total_revenue = 1875)
  (h4 : adult_count = 175) :
  (total_revenue - child_cost * (total_tickets - adult_count)) / adult_count = 9 := by
  sorry

#eval (1875 - 6 * (225 - 175)) / 175  -- Should output 9

end NUMINAMATH_CALUDE_adult_ticket_cost_l964_96451


namespace NUMINAMATH_CALUDE_range_of_m_l964_96414

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 14 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (A ∪ B m = A) ↔ m ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l964_96414


namespace NUMINAMATH_CALUDE_g_difference_l964_96447

/-- The function g(x) = 3x^3 + 4x^2 - 3x + 2 -/
def g (x : ℝ) : ℝ := 3 * x^3 + 4 * x^2 - 3 * x + 2

/-- Theorem stating that g(x + h) - g(x) = h(9x^2 + 8x + 9xh + 4h + 3h^2 - 3) for all x and h -/
theorem g_difference (x h : ℝ) : 
  g (x + h) - g x = h * (9 * x^2 + 8 * x + 9 * x * h + 4 * h + 3 * h^2 - 3) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l964_96447


namespace NUMINAMATH_CALUDE_both_arithmetic_and_geometric_is_geometric_with_ratio_one_l964_96415

/-- A sequence that is both arithmetic and geometric -/
def BothArithmeticAndGeometric (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r)

/-- Theorem: A sequence that is both arithmetic and geometric is a geometric sequence with common ratio 1 -/
theorem both_arithmetic_and_geometric_is_geometric_with_ratio_one 
  (a : ℕ → ℝ) (h : BothArithmeticAndGeometric a) : 
  ∃ r : ℝ, r = 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r :=
by sorry

end NUMINAMATH_CALUDE_both_arithmetic_and_geometric_is_geometric_with_ratio_one_l964_96415


namespace NUMINAMATH_CALUDE_remainder_problem_l964_96460

theorem remainder_problem (n : ℤ) (k : ℤ) (h : n = 25 * k - 2) :
  (n^2 + 3*n + 5) % 25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l964_96460


namespace NUMINAMATH_CALUDE_value_of_x_l964_96421

theorem value_of_x (z y x : ℝ) (hz : z = 90) (hy : y = 1/3 * z) (hx : x = 1/2 * y) :
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l964_96421


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l964_96433

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, -1/2 < x ∧ x < 1/3 ↔ a*x^2 + 2*x + 20 > 0) → a = -12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l964_96433


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l964_96419

theorem area_between_concentric_circles :
  let r₁ : ℝ := 12  -- radius of larger circle
  let r₂ : ℝ := 7   -- radius of smaller circle
  let A₁ := π * r₁^2  -- area of larger circle
  let A₂ := π * r₂^2  -- area of smaller circle
  A₁ - A₂ = 95 * π := by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l964_96419


namespace NUMINAMATH_CALUDE_prob_zhong_guo_meng_correct_l964_96440

/-- The number of cards labeled "中" -/
def num_zhong : ℕ := 2

/-- The number of cards labeled "国" -/
def num_guo : ℕ := 2

/-- The number of cards labeled "梦" -/
def num_meng : ℕ := 1

/-- The total number of cards -/
def total_cards : ℕ := num_zhong + num_guo + num_meng

/-- The number of cards drawn -/
def cards_drawn : ℕ := 3

/-- The probability of drawing cards that form "中国梦" -/
def prob_zhong_guo_meng : ℚ := 2 / 5

theorem prob_zhong_guo_meng_correct :
  (num_zhong * num_guo * num_meng : ℚ) / (total_cards.choose cards_drawn) = prob_zhong_guo_meng := by
  sorry

end NUMINAMATH_CALUDE_prob_zhong_guo_meng_correct_l964_96440


namespace NUMINAMATH_CALUDE_range_of_a_l964_96485

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l964_96485


namespace NUMINAMATH_CALUDE_dairy_factory_profit_comparison_l964_96443

/-- Represents the profit calculation for a dairy factory --/
theorem dairy_factory_profit_comparison :
  let total_milk : ℝ := 20
  let fresh_milk_profit : ℝ := 500
  let yogurt_profit : ℝ := 1000
  let milk_powder_profit : ℝ := 1800
  let yogurt_capacity : ℝ := 6
  let milk_powder_capacity : ℝ := 2
  let days : ℝ := 4

  let plan_one_profit : ℝ := 
    (milk_powder_capacity * days * milk_powder_profit) + 
    ((total_milk - milk_powder_capacity * days) * fresh_milk_profit)

  let plan_two_milk_powder_days : ℝ := 
    (total_milk - yogurt_capacity * days) / (yogurt_capacity - milk_powder_capacity)
  
  let plan_two_yogurt_days : ℝ := days - plan_two_milk_powder_days

  let plan_two_profit : ℝ := 
    (plan_two_milk_powder_days * milk_powder_capacity * milk_powder_profit) + 
    (plan_two_yogurt_days * yogurt_capacity * yogurt_profit)

  plan_two_profit > plan_one_profit := by sorry

end NUMINAMATH_CALUDE_dairy_factory_profit_comparison_l964_96443


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l964_96458

theorem partial_fraction_sum (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l964_96458


namespace NUMINAMATH_CALUDE_lcm_of_40_and_14_l964_96480

theorem lcm_of_40_and_14 :
  let n : ℕ := 40
  let m : ℕ := 14
  let gcf : ℕ := 10
  Nat.gcd n m = gcf →
  Nat.lcm n m = 56 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_40_and_14_l964_96480


namespace NUMINAMATH_CALUDE_nell_card_difference_l964_96402

/-- Given Nell's card collection information, prove the difference between
    her final Ace cards and baseball cards. -/
theorem nell_card_difference
  (initial_baseball : ℕ)
  (initial_ace : ℕ)
  (final_baseball : ℕ)
  (final_ace : ℕ)
  (h1 : initial_baseball = 239)
  (h2 : initial_ace = 38)
  (h3 : final_baseball = 111)
  (h4 : final_ace = 376) :
  final_ace - final_baseball = 265 := by
  sorry

end NUMINAMATH_CALUDE_nell_card_difference_l964_96402


namespace NUMINAMATH_CALUDE_milk_selling_price_l964_96455

/-- Calculates the selling price of a milk-water mixture given the initial milk price, water percentage, and desired gain percentage. -/
def calculate_selling_price (milk_price : ℚ) (water_percentage : ℚ) (gain_percentage : ℚ) : ℚ :=
  let total_volume : ℚ := 1 + water_percentage
  let cost_price : ℚ := milk_price
  let selling_price : ℚ := cost_price * (1 + gain_percentage)
  selling_price / total_volume

/-- Proves that the selling price of the milk-water mixture is 15 rs per liter under the given conditions. -/
theorem milk_selling_price :
  calculate_selling_price 12 (20/100) (50/100) = 15 := by
  sorry

end NUMINAMATH_CALUDE_milk_selling_price_l964_96455


namespace NUMINAMATH_CALUDE_problem_sculpture_area_l964_96490

/-- Represents a pyramid-like sculpture made of unit cubes -/
structure PyramidSculpture where
  total_cubes : ℕ
  num_layers : ℕ
  layer_sizes : List ℕ
  (total_cubes_sum : total_cubes = layer_sizes.sum)
  (layer_count : num_layers = layer_sizes.length)

/-- Calculates the exposed surface area of a pyramid sculpture -/
def exposed_surface_area (p : PyramidSculpture) : ℕ :=
  sorry

/-- The specific pyramid sculpture described in the problem -/
def problem_sculpture : PyramidSculpture :=
  { total_cubes := 19
  , num_layers := 4
  , layer_sizes := [1, 3, 5, 10]
  , total_cubes_sum := by sorry
  , layer_count := by sorry
  }

/-- Theorem stating that the exposed surface area of the problem sculpture is 43 square meters -/
theorem problem_sculpture_area : exposed_surface_area problem_sculpture = 43 := by
  sorry

end NUMINAMATH_CALUDE_problem_sculpture_area_l964_96490


namespace NUMINAMATH_CALUDE_sum_of_seventh_eighth_ninth_l964_96418

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  sum_first_three : a 1 + a 2 + a 3 = 30
  sum_next_three : a 4 + a 5 + a 6 = 120

/-- The sum of the 7th, 8th, and 9th terms equals 480 -/
theorem sum_of_seventh_eighth_ninth (seq : GeometricSequence) : 
  seq.a 7 + seq.a 8 + seq.a 9 = 480 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seventh_eighth_ninth_l964_96418


namespace NUMINAMATH_CALUDE_bicycle_weight_proof_l964_96474

/-- The weight of one bicycle in pounds -/
def bicycle_weight : ℝ := 20

/-- The weight of one scooter in pounds -/
def scooter_weight : ℝ := 40

theorem bicycle_weight_proof :
  (10 * bicycle_weight = 5 * scooter_weight) ∧
  (5 * scooter_weight = 200) →
  bicycle_weight = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_bicycle_weight_proof_l964_96474


namespace NUMINAMATH_CALUDE_judy_spending_l964_96427

def carrot_price : ℕ := 1
def milk_price : ℕ := 3
def pineapple_price : ℕ := 4
def flour_price : ℕ := 5
def ice_cream_price : ℕ := 7

def carrot_quantity : ℕ := 5
def milk_quantity : ℕ := 4
def pineapple_quantity : ℕ := 2
def flour_quantity : ℕ := 2

def coupon_discount : ℕ := 10
def coupon_threshold : ℕ := 40

def shopping_total : ℕ :=
  carrot_price * carrot_quantity +
  milk_price * milk_quantity +
  pineapple_price * (pineapple_quantity / 2) +
  flour_price * flour_quantity +
  ice_cream_price

theorem judy_spending :
  shopping_total = 38 :=
by sorry

end NUMINAMATH_CALUDE_judy_spending_l964_96427


namespace NUMINAMATH_CALUDE_bakers_cakes_l964_96416

/-- The number of cakes Baker made is equal to the sum of cakes sold and cakes left. -/
theorem bakers_cakes (total sold left : ℕ) (h1 : sold = 145) (h2 : left = 72) (h3 : total = sold + left) :
  total = 217 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cakes_l964_96416


namespace NUMINAMATH_CALUDE_root_implies_a_in_interval_l964_96478

/-- Given that for all real m, the function f(x) = m(x^2 - 1) + x - a always has a root,
    prove that a is in the interval [-1, 1] -/
theorem root_implies_a_in_interval :
  (∀ m : ℝ, ∃ x : ℝ, m * (x^2 - 1) + x - a = 0) →
  a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_in_interval_l964_96478


namespace NUMINAMATH_CALUDE_predicted_height_at_10_l964_96468

/-- Represents the regression model for height prediction -/
def height_model (age : ℝ) : ℝ := 7.19 * age + 73.93

/-- Theorem stating that the predicted height at age 10 is approximately 145.83 cm -/
theorem predicted_height_at_10 :
  ∃ ε > 0, |height_model 10 - 145.83| < ε :=
sorry

end NUMINAMATH_CALUDE_predicted_height_at_10_l964_96468


namespace NUMINAMATH_CALUDE_pats_password_length_l964_96428

/-- Represents the structure of Pat's computer password -/
structure PasswordStructure where
  lowercase_count : ℕ
  uppercase_and_numbers_count : ℕ
  symbol_count : ℕ

/-- Calculates the total number of characters in Pat's password -/
def total_characters (p : PasswordStructure) : ℕ :=
  p.lowercase_count + p.uppercase_and_numbers_count + p.symbol_count

/-- Theorem stating the total number of characters in Pat's password -/
theorem pats_password_length :
  ∃ (p : PasswordStructure),
    p.lowercase_count = 8 ∧
    p.uppercase_and_numbers_count = p.lowercase_count / 2 ∧
    p.symbol_count = 2 ∧
    total_characters p = 14 := by
  sorry

end NUMINAMATH_CALUDE_pats_password_length_l964_96428


namespace NUMINAMATH_CALUDE_unique_n_for_prime_sequence_l964_96400

theorem unique_n_for_prime_sequence : ∃! (n : ℕ), 
  n > 0 ∧ 
  Nat.Prime (n + 1) ∧ 
  Nat.Prime (n + 3) ∧ 
  Nat.Prime (n + 7) ∧ 
  Nat.Prime (n + 9) ∧ 
  Nat.Prime (n + 13) ∧ 
  Nat.Prime (n + 15) :=
by sorry

end NUMINAMATH_CALUDE_unique_n_for_prime_sequence_l964_96400


namespace NUMINAMATH_CALUDE_f_max_min_in_interval_l964_96411

-- Define the function f
def f : ℝ → ℝ := sorry

-- f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- f satisfies f(1 + x) = f(1 - x) for all x
axiom f_symmetry : ∀ x, f (1 + x) = f (1 - x)

-- f is monotonically increasing in [-1, 1]
axiom f_monotone : ∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y

-- Theorem statement
theorem f_max_min_in_interval :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≤ f 1) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f 3 ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_in_interval_l964_96411


namespace NUMINAMATH_CALUDE_total_coins_l964_96404

/-- Given 5 piles of quarters, 5 piles of dimes, and 3 coins in each pile, 
    the total number of coins is 30. -/
theorem total_coins (piles_quarters piles_dimes coins_per_pile : ℕ) 
  (h1 : piles_quarters = 5)
  (h2 : piles_dimes = 5)
  (h3 : coins_per_pile = 3) :
  piles_quarters * coins_per_pile + piles_dimes * coins_per_pile = 30 :=
by sorry

end NUMINAMATH_CALUDE_total_coins_l964_96404


namespace NUMINAMATH_CALUDE_max_product_constrained_sum_l964_96409

theorem max_product_constrained_sum (x y : ℕ+) (h : 7 * x + 5 * y = 140) :
  x * y ≤ 140 ∧ ∃ (a b : ℕ+), 7 * a + 5 * b = 140 ∧ a * b = 140 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_sum_l964_96409


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l964_96438

def sum_of_two_digit_pairs (n : ℕ) : ℕ :=
  (n % 100) + ((n / 100) % 100) + ((n / 10000) % 100)

def alternating_sum_of_three_digit_groups (n : ℕ) : ℤ :=
  (n % 1000 : ℤ) - ((n / 1000) % 1000 : ℤ)

theorem smallest_addition_for_divisibility (n : ℕ) (k : ℕ) :
  (∀ m < k, ¬(456 ∣ (987654 + m))) ∧
  (456 ∣ (987654 + k)) ∧
  (19 ∣ sum_of_two_digit_pairs (987654 + k)) ∧
  (8 ∣ alternating_sum_of_three_digit_groups (987654 + k)) →
  k = 22 := by
  sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l964_96438


namespace NUMINAMATH_CALUDE_abs_negative_two_l964_96481

theorem abs_negative_two : abs (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_two_l964_96481


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_plus_sides_l964_96445

/-- The number of sides in a regular dodecagon -/
def dodecagon_sides : ℕ := 12

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The sum of the number of diagonals and sides in a regular dodecagon is 66 -/
theorem dodecagon_diagonals_plus_sides :
  num_diagonals dodecagon_sides + dodecagon_sides = 66 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_plus_sides_l964_96445


namespace NUMINAMATH_CALUDE_shape_is_cone_l964_96482

/-- The shape described by ρ = c sin φ in spherical coordinates is a cone -/
theorem shape_is_cone (c : ℝ) (h : c > 0) :
  ∃ (cone : Set (ℝ × ℝ × ℝ)),
    ∀ (ρ θ φ : ℝ),
      (ρ, θ, φ) ∈ cone ↔ ρ = c * Real.sin φ ∧ ρ ≥ 0 ∧ θ ∈ Set.Icc 0 (2 * Real.pi) ∧ φ ∈ Set.Icc 0 Real.pi :=
by sorry

end NUMINAMATH_CALUDE_shape_is_cone_l964_96482


namespace NUMINAMATH_CALUDE_expand_expression_l964_96493

theorem expand_expression (x : ℝ) : (15 * x + 17 + 3) * (3 * x) = 45 * x^2 + 60 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l964_96493


namespace NUMINAMATH_CALUDE_expression_evaluation_l964_96429

theorem expression_evaluation : -4 / (4 / 9) * (9 / 4) = -81 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l964_96429


namespace NUMINAMATH_CALUDE_cookie_distribution_l964_96439

theorem cookie_distribution (people : ℕ) (cookies_per_person : ℕ) 
  (h1 : people = 6) (h2 : cookies_per_person = 4) : 
  people * cookies_per_person = 24 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l964_96439


namespace NUMINAMATH_CALUDE_nonnegative_solutions_system_l964_96403

theorem nonnegative_solutions_system (x y z : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
  Real.sqrt (x + y) + Real.sqrt z = 7 →
  Real.sqrt (x + z) + Real.sqrt y = 7 →
  Real.sqrt (y + z) + Real.sqrt x = 5 →
  ((x = 1 ∧ y = 4 ∧ z = 4) ∨ (x = 1 ∧ y = 9 ∧ z = 9)) :=
by sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_system_l964_96403


namespace NUMINAMATH_CALUDE_sin_product_equality_l964_96424

theorem sin_product_equality : 
  Real.sin (25 * π / 180) * Real.sin (35 * π / 180) * Real.sin (60 * π / 180) * Real.sin (85 * π / 180) =
  Real.sin (20 * π / 180) * Real.sin (40 * π / 180) * Real.sin (75 * π / 180) * Real.sin (80 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_sin_product_equality_l964_96424


namespace NUMINAMATH_CALUDE_gcf_lcm_360_270_l964_96417

theorem gcf_lcm_360_270 :
  (Nat.gcd 360 270 = 90) ∧ (Nat.lcm 360 270 = 1080) := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_360_270_l964_96417


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l964_96466

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal length
def focal_length (c : ℝ) : Prop := c = 2

-- Define the eccentricity
def eccentricity (e a c : ℝ) : Prop := e = c / a

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : focal_length 2) 
  (hp : hyperbola a b 2 3) : 
  ∃ e, eccentricity e a 2 ∧ e = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l964_96466


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l964_96410

/-- Given a line in vector form, prove its slope-intercept form --/
theorem line_vector_to_slope_intercept :
  let vector_form : ℝ × ℝ → Prop := λ p => (3 : ℝ) * (p.1 - 2) + (-4 : ℝ) * (p.2 + 3) = 0
  ∃ m b : ℝ, m = 3/4 ∧ b = -9/2 ∧ ∀ x y : ℝ, vector_form (x, y) ↔ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l964_96410


namespace NUMINAMATH_CALUDE_Q_subset_complement_P_l964_96441

-- Define the sets P and Q
def P : Set ℝ := {x | x > 4}
def Q : Set ℝ := {x | -2 < x ∧ x < 2}

-- State the theorem
theorem Q_subset_complement_P : Q ⊆ (Set.univ \ P) := by sorry

end NUMINAMATH_CALUDE_Q_subset_complement_P_l964_96441


namespace NUMINAMATH_CALUDE_yara_ahead_of_theon_l964_96406

/-- Proves that Yara will be 3 hours ahead of Theon given their ship speeds and destination distance -/
theorem yara_ahead_of_theon (theon_speed yara_speed distance : ℝ) 
  (h1 : theon_speed = 15)
  (h2 : yara_speed = 30)
  (h3 : distance = 90) :
  yara_speed / distance - theon_speed / distance = 3 := by
  sorry

end NUMINAMATH_CALUDE_yara_ahead_of_theon_l964_96406


namespace NUMINAMATH_CALUDE_unique_x_intercept_l964_96425

/-- The parabola equation: x = -3y^2 + 2y + 4 -/
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 4

/-- X-intercept occurs when y = 0 -/
def x_intercept : ℝ := parabola 0

theorem unique_x_intercept : 
  ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_x_intercept_l964_96425


namespace NUMINAMATH_CALUDE_inequality_preservation_l964_96462

theorem inequality_preservation (m n : ℝ) (h : m > n) : m / 5 > n / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l964_96462


namespace NUMINAMATH_CALUDE_four_machines_completion_time_l964_96420

/-- A machine with a given work rate in jobs per hour -/
structure Machine where
  work_rate : ℚ

/-- The time taken for multiple machines to complete one job when working together -/
def time_to_complete (machines : List Machine) : ℚ :=
  1 / (machines.map (λ m => m.work_rate) |>.sum)

theorem four_machines_completion_time :
  let machine_a : Machine := ⟨1/4⟩
  let machine_b : Machine := ⟨1/2⟩
  let machine_c : Machine := ⟨1/6⟩
  let machine_d : Machine := ⟨1/3⟩
  let machines := [machine_a, machine_b, machine_c, machine_d]
  time_to_complete machines = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_four_machines_completion_time_l964_96420


namespace NUMINAMATH_CALUDE_dried_mushroom_weight_l964_96426

/-- 
Given:
- Fresh mushrooms contain 90% water by weight
- Dried mushrooms contain 12% water by weight
- We start with 22 kg of fresh mushrooms

Prove that the weight of dried mushrooms obtained is 2.5 kg
-/
theorem dried_mushroom_weight (fresh_water_content : ℝ) (dried_water_content : ℝ) 
  (fresh_weight : ℝ) (dried_weight : ℝ) :
  fresh_water_content = 0.90 →
  dried_water_content = 0.12 →
  fresh_weight = 22 →
  dried_weight = 2.5 →
  dried_weight = (1 - fresh_water_content) * fresh_weight / (1 - dried_water_content) :=
by sorry

end NUMINAMATH_CALUDE_dried_mushroom_weight_l964_96426


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l964_96448

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_23 :
  ∀ p : ℕ, is_prime p → digit_sum p = 23 → p ≥ 599 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l964_96448


namespace NUMINAMATH_CALUDE_pirate_treasure_l964_96461

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_l964_96461


namespace NUMINAMATH_CALUDE_positive_plus_negative_implies_negative_l964_96496

theorem positive_plus_negative_implies_negative (a b : ℝ) :
  a > 0 → a + b < 0 → b < 0 := by sorry

end NUMINAMATH_CALUDE_positive_plus_negative_implies_negative_l964_96496


namespace NUMINAMATH_CALUDE_expression_simplification_l964_96434

theorem expression_simplification (a b x y : ℝ) 
  (h1 : a * b * (x^2 - y^2) + x * y * (a^2 - b^2) ≠ 0)
  (h2 : x ≠ -a * y / b)
  (h3 : x ≠ b * y / a) :
  ((a * b * (x^2 + y^2) + x * y * (a^2 + b^2)) * ((a * x + b * y)^2 - 4 * a * b * x * y)) /
  (a * b * (x^2 - y^2) + x * y * (a^2 - b^2)) = a^2 * x^2 - b^2 * y^2 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l964_96434


namespace NUMINAMATH_CALUDE_uncovered_volume_is_229_l964_96483

def shoebox_volume : ℝ := 4 * 6 * 12

def object1_volume : ℝ := 5 * 3 * 1
def object2_volume : ℝ := 2 * 2 * 3
def object3_volume : ℝ := 4 * 2 * 4

def total_object_volume : ℝ := object1_volume + object2_volume + object3_volume

theorem uncovered_volume_is_229 : 
  shoebox_volume - total_object_volume = 229 := by sorry

end NUMINAMATH_CALUDE_uncovered_volume_is_229_l964_96483


namespace NUMINAMATH_CALUDE_bus_stop_optimal_location_l964_96453

/-- Represents the distance between two buildings in meters -/
def building_distance : ℝ := 250

/-- Represents the number of students in the first building -/
def students_building1 : ℕ := 100

/-- Represents the number of students in the second building -/
def students_building2 : ℕ := 150

/-- Calculates the total walking distance for all students given the bus stop location -/
def total_walking_distance (bus_stop_location : ℝ) : ℝ :=
  students_building2 * bus_stop_location + students_building1 * (building_distance - bus_stop_location)

/-- Theorem stating that the total walking distance is minimized when the bus stop is at the second building -/
theorem bus_stop_optimal_location :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ building_distance →
    total_walking_distance 0 ≤ total_walking_distance x :=
by sorry

end NUMINAMATH_CALUDE_bus_stop_optimal_location_l964_96453


namespace NUMINAMATH_CALUDE_oil_depth_in_specific_tank_l964_96412

/-- Represents a horizontal cylindrical tank --/
structure HorizontalCylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the depth of oil in a horizontal cylindrical tank --/
def oilDepth (tank : HorizontalCylindricalTank) (surfaceArea : ℝ) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem: The depth of oil in the specified tank with given surface area --/
theorem oil_depth_in_specific_tank :
  let tank : HorizontalCylindricalTank := ⟨12, 8⟩
  let surfaceArea : ℝ := 48
  oilDepth tank surfaceArea = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_oil_depth_in_specific_tank_l964_96412


namespace NUMINAMATH_CALUDE_parabola_directrix_l964_96472

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop :=
  y = (2 * x^2 - 8 * x + 6) / 16

/-- The directrix equation -/
def directrix (y : ℝ) : Prop :=
  y = -3/2

/-- Theorem stating that the given directrix is correct for the parabola -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ y_d : ℝ, directrix y_d ∧ 
  (∀ p q : ℝ × ℝ, parabola p.1 p.2 → 
    (p.1 - x)^2 + (p.2 - y)^2 = (p.2 - y_d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l964_96472


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l964_96491

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (m : ℕ) 
  (h1 : m > 0)
  (h2 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q))
  (h3 : ∀ n, a (n + 1) = q * a n)
  (h4 : S (2 * m) / S m = 9)
  (h5 : a (2 * m) / a m = (5 * m + 1) / (m - 1)) :
  q = 2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l964_96491


namespace NUMINAMATH_CALUDE_midpoint_distance_to_origin_l964_96464

theorem midpoint_distance_to_origin : 
  let p1 : ℝ × ℝ := (-6, 8)
  let p2 : ℝ × ℝ := (6, -8)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1^2 + midpoint.2^2).sqrt = 0 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_to_origin_l964_96464


namespace NUMINAMATH_CALUDE_simplest_form_product_l964_96435

theorem simplest_form_product (a b : ℕ) (h : a = 45 ∧ b = 75) : 
  let g := Nat.gcd a b
  (a / g) * (b / g) = 15 := by
sorry

end NUMINAMATH_CALUDE_simplest_form_product_l964_96435


namespace NUMINAMATH_CALUDE_ace_ten_king_of_hearts_probability_l964_96407

/-- The probability of drawing an Ace, then a 10, then the King of Hearts from a standard deck of 52 cards without replacement -/
theorem ace_ten_king_of_hearts_probability :
  let total_cards : ℕ := 52
  let aces : ℕ := 4
  let tens : ℕ := 4
  let king_of_hearts : ℕ := 1
  (aces / total_cards) * (tens / (total_cards - 1)) * (king_of_hearts / (total_cards - 2)) = 4 / 33150 := by
sorry

end NUMINAMATH_CALUDE_ace_ten_king_of_hearts_probability_l964_96407


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_is_135_l964_96408

/-- The measure of each interior angle of a regular octagon in degrees. -/
def regular_octagon_interior_angle : ℝ := 135

/-- Theorem stating that the measure of each interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle_is_135 :
  regular_octagon_interior_angle = 135 := by sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_is_135_l964_96408


namespace NUMINAMATH_CALUDE_rod_and_rope_problem_l964_96465

theorem rod_and_rope_problem (x y : ℝ) : 
  (x = y + 5 ∧ x / 2 = y - 5) ↔ 
  (x - y = 5 ∧ y - x / 2 = 5) := by sorry

end NUMINAMATH_CALUDE_rod_and_rope_problem_l964_96465


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l964_96450

-- Define sets A and B
def A : Set ℝ := {x | 2 * x + 1 < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l964_96450


namespace NUMINAMATH_CALUDE_keystone_arch_angle_l964_96479

/-- Represents a keystone arch configuration -/
structure KeystoneArch where
  num_trapezoids : ℕ
  trapezoids_are_congruent : Bool
  trapezoids_are_isosceles : Bool
  sides_meet_at_center : Bool

/-- Calculates the larger interior angle of a trapezoid in the keystone arch -/
def larger_interior_angle (arch : KeystoneArch) : ℝ :=
  sorry

/-- Theorem stating that the larger interior angle of each trapezoid in a 12-piece keystone arch is 97.5° -/
theorem keystone_arch_angle (arch : KeystoneArch) :
  arch.num_trapezoids = 12 ∧ 
  arch.trapezoids_are_congruent ∧ 
  arch.trapezoids_are_isosceles ∧ 
  arch.sides_meet_at_center →
  larger_interior_angle arch = 97.5 :=
sorry

end NUMINAMATH_CALUDE_keystone_arch_angle_l964_96479


namespace NUMINAMATH_CALUDE_laptop_price_calculation_l964_96498

/-- Calculate the total selling price of a laptop given the original price, discount rate, coupon value, and tax rate -/
def totalSellingPrice (originalPrice discountRate couponValue taxRate : ℝ) : ℝ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  let priceAfterCoupon := discountedPrice - couponValue
  let finalPrice := priceAfterCoupon * (1 + taxRate)
  finalPrice

/-- Theorem stating that the total selling price of the laptop is 908.5 dollars -/
theorem laptop_price_calculation :
  totalSellingPrice 1200 0.30 50 0.15 = 908.5 := by
  sorry


end NUMINAMATH_CALUDE_laptop_price_calculation_l964_96498


namespace NUMINAMATH_CALUDE_bernardo_wins_l964_96459

def game_winner (M : ℕ) : Prop :=
  M ≤ 999 ∧
  3 * M < 1000 ∧
  3 * M + 100 < 1000 ∧
  3 * (3 * M + 100) < 1000 ∧
  3 * (3 * M + 100) + 100 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem bernardo_wins :
  ∃ M : ℕ, game_winner M ∧
    (∀ N : ℕ, N < M → ¬game_winner N) ∧
    M = 67 ∧
    sum_of_digits M = 13 := by
  sorry

end NUMINAMATH_CALUDE_bernardo_wins_l964_96459


namespace NUMINAMATH_CALUDE_odd_even_function_sum_l964_96470

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem odd_even_function_sum (f g : ℝ → ℝ) (h : ℝ → ℝ) 
  (hf : IsOdd f) (hg : IsEven g) 
  (sum_eq : ∀ x ≠ 1, f x + g x = 1 / (x - 1)) :
  ∀ x ≠ 1, f x = x / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_even_function_sum_l964_96470


namespace NUMINAMATH_CALUDE_specific_prism_volume_l964_96422

/-- A rectangular prism with given edge length sum and proportions -/
structure RectangularPrism where
  edgeSum : ℝ
  width : ℝ
  height : ℝ
  length : ℝ
  edgeSum_eq : edgeSum = 4 * (width + height + length)
  height_prop : height = 2 * width
  length_prop : length = 4 * width

/-- The volume of a rectangular prism -/
def volume (p : RectangularPrism) : ℝ := p.width * p.height * p.length

/-- Theorem: The volume of the specific rectangular prism is 85184/343 -/
theorem specific_prism_volume :
  ∃ (p : RectangularPrism), p.edgeSum = 88 ∧ volume p = 85184 / 343 := by
  sorry

end NUMINAMATH_CALUDE_specific_prism_volume_l964_96422


namespace NUMINAMATH_CALUDE_correlation_coefficient_properties_l964_96492

/-- The correlation coefficient between two variables -/
def correlation_coefficient (X Y : Type*) [NormedAddCommGroup X] [NormedAddCommGroup Y] : ℝ := sorry

/-- The strength of correlation between two variables -/
def correlation_strength (X Y : Type*) [NormedAddCommGroup X] [NormedAddCommGroup Y] : ℝ := sorry

theorem correlation_coefficient_properties (X Y : Type*) [NormedAddCommGroup X] [NormedAddCommGroup Y] :
  let r := correlation_coefficient X Y
  ∃ (strength : ℝ → ℝ),
    (∀ x, |x| ≤ 1 → strength x ≥ 0) ∧
    (∀ x y, |x| ≤ 1 → |y| ≤ 1 → |x| < |y| → strength x < strength y) ∧
    (∀ x, |x| ≤ 1 → strength x = correlation_strength X Y) ∧
    |r| ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_properties_l964_96492


namespace NUMINAMATH_CALUDE_meeting_distance_calculation_l964_96499

/-- Represents the problem of calculating the distance to a meeting location --/
theorem meeting_distance_calculation (initial_speed : ℝ) (speed_increase : ℝ) 
  (late_time : ℝ) (early_time : ℝ) :
  initial_speed = 40 →
  speed_increase = 20 →
  late_time = 1.5 →
  early_time = 1 →
  ∃ (distance : ℝ) (total_time : ℝ),
    distance = initial_speed * (total_time + late_time) ∧
    distance = initial_speed + (initial_speed + speed_increase) * (total_time - early_time - 1) ∧
    distance = 420 := by
  sorry


end NUMINAMATH_CALUDE_meeting_distance_calculation_l964_96499


namespace NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l964_96446

theorem at_least_one_leq_neg_two (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1/b ≤ -2) ∨ (b + 1/c ≤ -2) ∨ (c + 1/a ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l964_96446


namespace NUMINAMATH_CALUDE_phi_value_l964_96431

theorem phi_value : ∃! (Φ : ℕ), Φ < 10 ∧ 504 / Φ = 40 + 3 * Φ :=
  sorry

end NUMINAMATH_CALUDE_phi_value_l964_96431


namespace NUMINAMATH_CALUDE_odd_function_value_l964_96463

def f (a : ℝ) (x : ℝ) : ℝ := a * x + a + 3

theorem odd_function_value (a : ℝ) :
  (∀ x : ℝ, f a x = -(f a (-x))) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l964_96463


namespace NUMINAMATH_CALUDE_equation_solution_l964_96401

def solution_set : Set ℝ := {-Real.sqrt 10, -Real.pi, -1, 1, Real.pi, Real.sqrt 10}

def domain (x : ℝ) : Prop :=
  (-Real.sqrt 10 ≤ x ∧ x ≤ -1) ∨ (1 ≤ x ∧ x ≤ Real.sqrt 10)

theorem equation_solution :
  ∀ x : ℝ, domain x →
    ((Real.sin (2 * x) - Real.pi * Real.sin x) * Real.sqrt (11 * x^2 - x^4 - 10) = 0 ↔ x ∈ solution_set) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l964_96401


namespace NUMINAMATH_CALUDE_lisa_candies_on_specific_days_l964_96486

/-- The number of candies Lisa eats on Mondays and Wednesdays -/
def candies_on_specific_days (total_candies : ℕ) (weeks : ℕ) (days_per_week : ℕ) 
  (specific_days : ℕ) : ℕ :=
  let candies_on_other_days := (days_per_week - specific_days) * weeks
  let remaining_candies := total_candies - candies_on_other_days
  remaining_candies / (specific_days * weeks)

/-- Theorem stating that Lisa eats 2 candies on Mondays and Wednesdays -/
theorem lisa_candies_on_specific_days : 
  candies_on_specific_days 36 4 7 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lisa_candies_on_specific_days_l964_96486


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l964_96487

/-- 
Given a point P with coordinates (2, -5), 
prove that its symmetric point P' with respect to the origin has coordinates (-2, 5).
-/
theorem symmetric_point_wrt_origin : 
  let P : ℝ × ℝ := (2, -5)
  let P' : ℝ × ℝ := (-P.1, -P.2)
  P' = (-2, 5) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l964_96487


namespace NUMINAMATH_CALUDE_complex_number_existence_l964_96489

theorem complex_number_existence : ∃! (z₁ z₂ : ℂ),
  (z₁ + 10 / z₁).im = 0 ∧
  (z₂ + 10 / z₂).im = 0 ∧
  (z₁ + 4).re = -(z₁ + 4).im ∧
  (z₂ + 4).re = -(z₂ + 4).im ∧
  z₁ = -1 - 3*I ∧
  z₂ = -3 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_existence_l964_96489


namespace NUMINAMATH_CALUDE_leap_day_2024_is_sunday_l964_96436

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Calculates the day of the week for a given number of days after a Sunday -/
def dayAfterSunday (days : Nat) : DayOfWeek :=
  match days % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

/-- The number of days between February 29, 2000, and February 29, 2024 -/
def daysBetween2000And2024 : Nat := 8766

theorem leap_day_2024_is_sunday :
  dayAfterSunday daysBetween2000And2024 = DayOfWeek.Sunday := by
  sorry

#check leap_day_2024_is_sunday

end NUMINAMATH_CALUDE_leap_day_2024_is_sunday_l964_96436


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l964_96449

theorem min_sum_of_squares (x y : ℝ) (h : (x + 4) * (y - 4) = 0) :
  ∃ (m : ℝ), m = 16 ∧ ∀ (a b : ℝ), (a + 4) * (b - 4) = 0 → a^2 + b^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l964_96449


namespace NUMINAMATH_CALUDE_train_crossing_time_l964_96452

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (h1 : train_length = 130) (h2 : train_speed_kmh = 144) : 
  train_length / (train_speed_kmh * 1000 / 3600) = 3.25 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l964_96452


namespace NUMINAMATH_CALUDE_odd_sum_product_equivalence_l964_96488

theorem odd_sum_product_equivalence (p q : ℕ) 
  (hp : p < 16 ∧ p % 2 = 1) 
  (hq : q < 16 ∧ q % 2 = 1) : 
  p * q + p + q = (p + 1) * (q + 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_product_equivalence_l964_96488


namespace NUMINAMATH_CALUDE_largest_possible_value_l964_96437

theorem largest_possible_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x * y + y * z + z * x = 1) :
  let M := 3 / (Real.sqrt 3 + 1)
  (x / (1 + y * z / x)) + (y / (1 + z * x / y)) + (z / (1 + x * y / z)) ≥ M ∧ 
  ∀ N > M, ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b + b * c + c * a = 1 ∧
    (a / (1 + b * c / a)) + (b / (1 + c * a / b)) + (c / (1 + a * b / c)) < N :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_value_l964_96437


namespace NUMINAMATH_CALUDE_unique_solution_modular_system_l964_96477

theorem unique_solution_modular_system :
  ∃! x : ℕ, x < 12 ∧ (5 * x + 3) % 15 = 7 % 15 ∧ x % 4 = 2 % 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_modular_system_l964_96477


namespace NUMINAMATH_CALUDE_chad_savings_theorem_l964_96423

def calculate_savings (mowing_yards : ℝ) (birthday_holidays : ℝ) (video_games : ℝ) (odd_jobs : ℝ) : ℝ :=
  let total_earnings := mowing_yards + birthday_holidays + video_games + odd_jobs
  let tax_rate := 0.1
  let taxes := tax_rate * total_earnings
  let after_tax := total_earnings - taxes
  let mowing_savings := 0.5 * mowing_yards
  let birthday_savings := 0.3 * birthday_holidays
  let video_games_savings := 0.4 * video_games
  let odd_jobs_savings := 0.2 * odd_jobs
  mowing_savings + birthday_savings + video_games_savings + odd_jobs_savings

theorem chad_savings_theorem :
  calculate_savings 600 250 150 150 = 465 := by
  sorry

end NUMINAMATH_CALUDE_chad_savings_theorem_l964_96423


namespace NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_5_is_correct_l964_96471

/-- A function that checks if a positive integer has only odd digits -/
def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

/-- The largest positive integer less than 10000 with only odd digits that is a multiple of 5 -/
def largest_odd_digit_multiple_of_5 : ℕ := 9995

theorem largest_odd_digit_multiple_of_5_is_correct :
  (largest_odd_digit_multiple_of_5 < 10000) ∧
  (has_only_odd_digits largest_odd_digit_multiple_of_5) ∧
  (largest_odd_digit_multiple_of_5 % 5 = 0) ∧
  (∀ n : ℕ, n < 10000 → has_only_odd_digits n → n % 5 = 0 → n ≤ largest_odd_digit_multiple_of_5) :=
by sorry

#eval largest_odd_digit_multiple_of_5

end NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_5_is_correct_l964_96471


namespace NUMINAMATH_CALUDE_olaf_game_ratio_l964_96497

theorem olaf_game_ratio : 
  ∀ (father_points son_points : ℕ),
  father_points = 7 →
  ∃ (x : ℕ), son_points = x * father_points →
  father_points + son_points = 28 →
  son_points / father_points = 3 := by
sorry

end NUMINAMATH_CALUDE_olaf_game_ratio_l964_96497


namespace NUMINAMATH_CALUDE_possible_values_of_a_l964_96494

def P : Set ℝ := {x | x^2 ≠ 1}

def M (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem possible_values_of_a : 
  ∀ a : ℝ, M a ⊆ P ↔ a ∈ ({1, -1, 0} : Set ℝ) := by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l964_96494


namespace NUMINAMATH_CALUDE_inequality_proof_l964_96442

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l964_96442


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l964_96469

def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

theorem z_in_fourth_quadrant :
  let z : ℂ := 3 / (1 + 2 * Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l964_96469


namespace NUMINAMATH_CALUDE_x_eighth_power_is_one_l964_96476

theorem x_eighth_power_is_one (x : ℂ) (h : x + 1/x = Real.sqrt 2) : x^8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_eighth_power_is_one_l964_96476


namespace NUMINAMATH_CALUDE_switches_in_A_after_process_l964_96444

/-- Represents a switch with its label and position -/
structure Switch where
  label : Nat
  position : Fin 5

/-- The set of all switches -/
def switches : Finset Switch := sorry

/-- The process of advancing switches for 1000 steps -/
def advance_switches : Finset Switch → Finset Switch := sorry

/-- Counts switches in position A -/
def count_switches_in_A : Finset Switch → Nat := sorry

/-- Main theorem: After 1000 steps, 725 switches are in position A -/
theorem switches_in_A_after_process : 
  count_switches_in_A (advance_switches switches) = 725 := by sorry

end NUMINAMATH_CALUDE_switches_in_A_after_process_l964_96444


namespace NUMINAMATH_CALUDE_quadratic_function_property_l964_96495

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
def QuadraticFunction (a b c : ℝ) (h : a ≠ 0) : ℝ → ℝ :=
  fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_property
  (a b c : ℝ) (h : a ≠ 0)
  (x₁ x₂ : ℝ) (hx : x₁ ≠ x₂)
  (hf : QuadraticFunction a b c h x₁ = QuadraticFunction a b c h x₂) :
  QuadraticFunction a b c h (x₁ + x₂) = c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l964_96495


namespace NUMINAMATH_CALUDE_beatles_collection_theorem_l964_96456

/-- The number of albums in either Andrew's or John's collection, but not both -/
def unique_albums (shared : ℕ) (andrew_total : ℕ) (john_unique : ℕ) : ℕ :=
  (andrew_total - shared) + john_unique

theorem beatles_collection_theorem :
  unique_albums 9 17 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_beatles_collection_theorem_l964_96456


namespace NUMINAMATH_CALUDE_prob_ace_king_same_suit_standard_deck_l964_96484

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (aces_per_suit : Nat)
  (kings_per_suit : Nat)

/-- Probability of drawing an Ace then a King of the same suit -/
def prob_ace_then_king_same_suit (d : Deck) : ℚ :=
  (d.aces_per_suit : ℚ) / d.total_cards * (d.kings_per_suit : ℚ) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing an Ace then a King of the same suit in a standard deck -/
theorem prob_ace_king_same_suit_standard_deck :
  let standard_deck : Deck :=
    { total_cards := 52
    , suits := 4
    , cards_per_suit := 13
    , aces_per_suit := 1
    , kings_per_suit := 1
    }
  prob_ace_then_king_same_suit standard_deck = 1 / 663 := by
  sorry


end NUMINAMATH_CALUDE_prob_ace_king_same_suit_standard_deck_l964_96484


namespace NUMINAMATH_CALUDE_largest_square_side_length_l964_96405

-- Define the lengths of sticks for each side
def side1 : List ℕ := [4, 4, 2, 3]
def side2 : List ℕ := [4, 4, 3, 1, 1]
def side3 : List ℕ := [4, 3, 3, 2, 1]
def side4 : List ℕ := [3, 3, 3, 2, 2]

-- Theorem statement
theorem largest_square_side_length :
  List.sum side1 = List.sum side2 ∧
  List.sum side2 = List.sum side3 ∧
  List.sum side3 = List.sum side4 ∧
  List.sum side4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_side_length_l964_96405


namespace NUMINAMATH_CALUDE_domain_exclusion_sum_l964_96432

theorem domain_exclusion_sum (C D : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 8 * x + 6 = 0 ↔ (x = C ∨ x = D)) →
  C + D = 4 := by
  sorry

end NUMINAMATH_CALUDE_domain_exclusion_sum_l964_96432


namespace NUMINAMATH_CALUDE_sum_always_positive_l964_96413

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f (x + 4)) ∧
  (∀ x ≥ 2, Monotone (fun y ↦ f y))

/-- Theorem statement -/
theorem sum_always_positive
  (f : ℝ → ℝ)
  (hf : special_function f)
  (x₁ x₂ : ℝ)
  (h_sum : x₁ + x₂ > 4)
  (h_prod : (x₁ - 2) * (x₂ - 2) < 0) :
  f x₁ + f x₂ > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_always_positive_l964_96413


namespace NUMINAMATH_CALUDE_intercept_sum_l964_96457

theorem intercept_sum : ∃ (x₀ y₀ : ℕ), 
  x₀ < 25 ∧ y₀ < 25 ∧
  (4 * x₀) % 25 = 2 % 25 ∧
  (5 * y₀ + 2) % 25 = 0 ∧
  x₀ + y₀ = 28 := by
sorry

end NUMINAMATH_CALUDE_intercept_sum_l964_96457


namespace NUMINAMATH_CALUDE_other_solution_of_quadratic_l964_96475

theorem other_solution_of_quadratic (x : ℚ) : 
  (48 * (3/4)^2 + 29 = 35 * (3/4) + 12) → 
  (48 * (1/3)^2 + 29 = 35 * (1/3) + 12) := by
  sorry

end NUMINAMATH_CALUDE_other_solution_of_quadratic_l964_96475


namespace NUMINAMATH_CALUDE_reaction_properties_l964_96473

-- Define the reaction components
structure Reaction where
  k2cr2o7 : ℕ
  hcl : ℕ
  kcl : ℕ
  crcl3 : ℕ
  cl2 : ℕ
  h2o : ℕ

-- Define oxidation states
def oxidation_state_cr_initial : Int := 6
def oxidation_state_cr_final : Int := 3
def oxidation_state_cl_initial : Int := -1
def oxidation_state_cl_final : Int := 0

-- Define the balanced equation
def balanced_reaction : Reaction := {
  k2cr2o7 := 2,
  hcl := 14,
  kcl := 2,
  crcl3 := 2,
  cl2 := 3,
  h2o := 7
}

-- Define the number of electrons transferred
def electrons_transferred : ℕ := 6

-- Define the oxidizing agent
def oxidizing_agent : String := "K2Cr2O7"

-- Define the element being oxidized
def element_oxidized : String := "Cl in HCl"

-- Define the oxidation product
def oxidation_product : String := "Cl2"

-- Define the mass ratio of oxidized to unoxidized HCl
def mass_ratio_oxidized_unoxidized : Rat := 3 / 4

-- Define the number of electrons transferred for 0.1 mol of Cl2
def electrons_transferred_for_0_1_mol_cl2 : ℕ := 120400000000000000000000

theorem reaction_properties :
  -- (1) Verify the oxidizing agent, element oxidized, and oxidation product
  (oxidizing_agent = "K2Cr2O7") ∧
  (element_oxidized = "Cl in HCl") ∧
  (oxidation_product = "Cl2") ∧
  -- (2) Verify the mass ratio of oxidized to unoxidized HCl
  (mass_ratio_oxidized_unoxidized = 3 / 4) ∧
  -- (3) Verify the number of electrons transferred for 0.1 mol of Cl2
  (electrons_transferred_for_0_1_mol_cl2 = 120400000000000000000000) := by
  sorry

#check reaction_properties

end NUMINAMATH_CALUDE_reaction_properties_l964_96473


namespace NUMINAMATH_CALUDE_alice_winning_strategy_l964_96454

theorem alice_winning_strategy (x : ℕ) (h : x ≤ 2020) :
  ∃ k : ℤ, (2021 - x)^2 - x^2 = 2021 * k := by
  sorry

end NUMINAMATH_CALUDE_alice_winning_strategy_l964_96454
