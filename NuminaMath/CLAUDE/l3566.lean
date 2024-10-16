import Mathlib

namespace NUMINAMATH_CALUDE_sandy_lemonade_sales_l3566_356638

theorem sandy_lemonade_sales (sunday_half_dollars : ℕ) (total_amount : ℚ) (half_dollar_value : ℚ) :
  sunday_half_dollars = 6 →
  total_amount = 11.5 →
  half_dollar_value = 0.5 →
  (total_amount - sunday_half_dollars * half_dollar_value) / half_dollar_value = 17 := by
sorry

end NUMINAMATH_CALUDE_sandy_lemonade_sales_l3566_356638


namespace NUMINAMATH_CALUDE_angle_A_measure_l3566_356696

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions to ensure it's a valid triangle
  true

-- Define the measure of an angle
def angle_measure (A B C : ℝ × ℝ) : ℝ :=
  sorry

-- Define the length of a side
def side_length (A B : ℝ × ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem angle_A_measure 
  (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_acute : angle_measure A B C < π / 2 ∧ 
             angle_measure B C A < π / 2 ∧ 
             angle_measure C A B < π / 2)
  (h_BC : side_length B C = 3)
  (h_AB : side_length A B = Real.sqrt 6)
  (h_angle_C : angle_measure B C A = π / 4) :
  angle_measure C A B = π / 3 :=
sorry

end NUMINAMATH_CALUDE_angle_A_measure_l3566_356696


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l3566_356694

theorem unique_modular_congruence : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l3566_356694


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_to_one_l3566_356682

theorem quadratic_roots_sum_to_one (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x + y = 1 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ b = -a :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_to_one_l3566_356682


namespace NUMINAMATH_CALUDE_inscribed_squares_problem_l3566_356687

theorem inscribed_squares_problem (a b : ℝ) : 
  let small_area : ℝ := 16
  let large_area : ℝ := 18
  let rotation_angle : ℝ := 30 * π / 180
  let small_side : ℝ := Real.sqrt small_area
  let large_side : ℝ := Real.sqrt large_area
  a + b = large_side ∧ 
  Real.sqrt (a^2 + b^2) = 2 * small_side * Real.cos rotation_angle →
  a * b = -15 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_problem_l3566_356687


namespace NUMINAMATH_CALUDE_perpendicular_iff_x_eq_neg_three_l3566_356629

/-- Two 2D vectors are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_iff_x_eq_neg_three :
  ∀ x : ℝ, perpendicular (x, -3) (2, -2) ↔ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_iff_x_eq_neg_three_l3566_356629


namespace NUMINAMATH_CALUDE_ball_returns_in_three_throws_l3566_356651

/-- The number of boys in the circle -/
def n : ℕ := 15

/-- The number of positions skipped in each throw (including the thrower) -/
def skip : ℕ := 5

/-- The sequence of positions the ball reaches -/
def ball_sequence (start : ℕ) : ℕ → ℕ
  | 0 => start
  | i + 1 => (ball_sequence start i + skip) % n

/-- The theorem stating that it takes 3 throws for the ball to return to the start -/
theorem ball_returns_in_three_throws (start : ℕ) (h : start > 0 ∧ start ≤ n) : 
  ball_sequence start 3 = start :=
sorry

end NUMINAMATH_CALUDE_ball_returns_in_three_throws_l3566_356651


namespace NUMINAMATH_CALUDE_inequality_solution_l3566_356613

theorem inequality_solution (x : ℝ) : 
  (x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5) →
  (1 / (x - 1) - 5 / (x - 2) + 5 / (x - 3) - 1 / (x - 5) < 1 / 24) ↔ 
  (x < -2 ∨ (1 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ 5 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3566_356613


namespace NUMINAMATH_CALUDE_sine_graph_transformation_l3566_356601

/-- Given two sine functions with different periods, prove that the graph of one
    can be obtained by transforming the graph of the other. -/
theorem sine_graph_transformation (x : ℝ) :
  let f (x : ℝ) := Real.sin (x + π/4)
  let g (x : ℝ) := Real.sin (3*x + π/4)
  ∃ (h : ℝ → ℝ), (∀ x, g x = f (h x)) ∧ (∀ x, h x = x/3) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_transformation_l3566_356601


namespace NUMINAMATH_CALUDE_inequality_solution_l3566_356612

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / (x * (x + 2)) < 1 / 4) ↔ (x < -1 ∨ x > 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3566_356612


namespace NUMINAMATH_CALUDE_total_cost_to_fill_displays_l3566_356674

-- Define the jewelry types
inductive JewelryType
| Necklace
| Ring
| Bracelet

-- Define the structure for jewelry information
structure JewelryInfo where
  capacity : Nat
  current : Nat
  price : Nat
  discountRules : List (Nat × Nat)

-- Define the jewelry store inventory
def inventory : JewelryType → JewelryInfo
| JewelryType.Necklace => ⟨12, 5, 4, [(4, 10), (6, 15)]⟩
| JewelryType.Ring => ⟨30, 18, 10, [(10, 5), (20, 10)]⟩
| JewelryType.Bracelet => ⟨15, 8, 5, [(7, 8), (10, 12)]⟩

-- Function to calculate the discounted price
def calculateDiscountedPrice (info : JewelryInfo) (quantity : Nat) : Nat :=
  let totalPrice := quantity * info.price
  let applicableDiscount := info.discountRules.foldl
    (fun acc (threshold, discount) => if quantity ≥ threshold then max acc discount else acc)
    0
  totalPrice - totalPrice * applicableDiscount / 100

-- Theorem statement
theorem total_cost_to_fill_displays :
  (calculateDiscountedPrice (inventory JewelryType.Necklace) (12 - 5)) +
  (calculateDiscountedPrice (inventory JewelryType.Ring) (30 - 18)) +
  (calculateDiscountedPrice (inventory JewelryType.Bracelet) (15 - 8)) = 170 := by
  sorry


end NUMINAMATH_CALUDE_total_cost_to_fill_displays_l3566_356674


namespace NUMINAMATH_CALUDE_total_weight_is_56_7_l3566_356683

/-- The total weight of five plastic rings in grams -/
def total_weight_in_grams : ℝ :=
  let orange_weight := 0.08333333333333333
  let purple_weight := 0.3333333333333333
  let white_weight := 0.4166666666666667
  let blue_weight := 0.5416666666666666
  let red_weight := 0.625
  let conversion_factor := 28.35
  (orange_weight + purple_weight + white_weight + blue_weight + red_weight) * conversion_factor

/-- Theorem stating that the total weight of the five plastic rings is 56.7 grams -/
theorem total_weight_is_56_7 : total_weight_in_grams = 56.7 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_56_7_l3566_356683


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l3566_356670

theorem fraction_to_zero_power (a b : ℤ) (h : b ≠ 0) :
  (a / b : ℚ) ^ 0 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l3566_356670


namespace NUMINAMATH_CALUDE_frustum_slant_height_l3566_356605

theorem frustum_slant_height (r₁ r₂ : ℝ) (h : r₁ = 2 ∧ r₂ = 5) :
  let l := (π * (r₁^2 + r₂^2)) / (π * (r₁ + r₂))
  l = 29 / 7 := by
  sorry

end NUMINAMATH_CALUDE_frustum_slant_height_l3566_356605


namespace NUMINAMATH_CALUDE_g_prime_zero_f_symmetry_f_prime_symmetry_l3566_356635

-- Define the functions and their derivatives
variable (f g : ℝ → ℝ)
variable (f' g' : ℝ → ℝ)

-- Define the conditions
axiom func_relation : ∀ x, f (x + 3) = g (-x) + 4
axiom deriv_relation : ∀ x, f' x + g' (1 + x) = 0
axiom g_even : ∀ x, g (2*x + 1) = g (-2*x + 1)

-- Define the derivative relationship
axiom f_deriv : ∀ x, (deriv f) x = f' x
axiom g_deriv : ∀ x, (deriv g) x = g' x

-- State the theorems to be proved
theorem g_prime_zero : g' 1 = 0 := by sorry

theorem f_symmetry : ∀ x, f (x + 4) = f x := by sorry

theorem f_prime_symmetry : ∀ x, f' (x + 2) = f' x := by sorry

end NUMINAMATH_CALUDE_g_prime_zero_f_symmetry_f_prime_symmetry_l3566_356635


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3566_356647

theorem quadratic_roots_condition (c : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ + c = 0 ∧ 
  x₂^2 - 2*x₂ + c = 0 ∧ 
  7*x₂ - 4*x₁ = 47 →
  c = -15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3566_356647


namespace NUMINAMATH_CALUDE_negative_five_greater_than_negative_sqrt_26_l3566_356669

theorem negative_five_greater_than_negative_sqrt_26 :
  -5 > -Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_greater_than_negative_sqrt_26_l3566_356669


namespace NUMINAMATH_CALUDE_hash_2_5_3_equals_1_l3566_356632

-- Define the # operation
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Theorem statement
theorem hash_2_5_3_equals_1 : hash 2 5 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_5_3_equals_1_l3566_356632


namespace NUMINAMATH_CALUDE_sock_ratio_is_one_to_two_l3566_356622

/-- Represents the order of socks --/
structure SockOrder where
  green : ℕ
  red : ℕ
  green_price : ℝ
  red_price : ℝ

/-- The original order --/
def original_order : SockOrder := {
  green := 6,
  red := 0,  -- We don't know this value yet
  green_price := 0,  -- We don't know this value yet
  red_price := 0  -- We don't know this value yet
}

/-- The swapped order --/
def swapped_order (o : SockOrder) : SockOrder := {
  green := o.red,
  red := o.green,
  green_price := o.green_price,
  red_price := o.red_price
}

/-- The cost of an order --/
def cost (o : SockOrder) : ℝ :=
  o.green * o.green_price + o.red * o.red_price

/-- The theorem to prove --/
theorem sock_ratio_is_one_to_two :
  ∃ (o : SockOrder),
    o.green = 6 ∧
    o.green_price = 3 * o.red_price ∧
    cost (swapped_order o) = 1.4 * cost o ∧
    o.green / o.red = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_sock_ratio_is_one_to_two_l3566_356622


namespace NUMINAMATH_CALUDE_compare_exponentials_l3566_356648

theorem compare_exponentials (a b c : ℝ) : 
  a = (0.4 : ℝ) ^ (0.3 : ℝ) → 
  b = (0.3 : ℝ) ^ (0.4 : ℝ) → 
  c = (0.3 : ℝ) ^ (-(0.2 : ℝ)) → 
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_compare_exponentials_l3566_356648


namespace NUMINAMATH_CALUDE_standard_pairs_parity_l3566_356646

/-- Represents the color of a square on the chessboard -/
inductive Color
| Red
| Blue

/-- Represents a chessboard with m rows and n columns -/
structure Chessboard (m n : ℕ) where
  colors : Fin m → Fin n → Color
  m_ge_3 : m ≥ 3
  n_ge_3 : n ≥ 3

/-- Count of blue squares on the edges (excluding corners) of the chessboard -/
def count_edge_blue (board : Chessboard m n) : ℕ := sorry

/-- Count of standard pairs (adjacent squares with different colors) on the chessboard -/
def count_standard_pairs (board : Chessboard m n) : ℕ := sorry

/-- Main theorem: The number of standard pairs is odd iff the number of blue edge squares is odd -/
theorem standard_pairs_parity (m n : ℕ) (board : Chessboard m n) :
  Odd (count_standard_pairs board) ↔ Odd (count_edge_blue board) := by sorry

end NUMINAMATH_CALUDE_standard_pairs_parity_l3566_356646


namespace NUMINAMATH_CALUDE_inequality_proof_l3566_356610

theorem inequality_proof (a b : ℝ) : a^2 + a*b + b^2 ≥ 3*(a + b - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3566_356610


namespace NUMINAMATH_CALUDE_mini_bank_withdrawal_l3566_356675

theorem mini_bank_withdrawal (d c : ℕ) : 
  (0 < c) → (c < 100) →
  (100 * c + d - 350 = 2 * (100 * d + c)) →
  (d = 14 ∧ c = 32) := by
sorry

end NUMINAMATH_CALUDE_mini_bank_withdrawal_l3566_356675


namespace NUMINAMATH_CALUDE_equal_savings_l3566_356639

/-- Represents a person's financial data -/
structure Person where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- The problem setup -/
def financialProblem (p1 p2 : Person) : Prop :=
  -- Income ratio condition
  p1.income * 4 = p2.income * 5 ∧
  -- Expenditure ratio condition
  p1.expenditure * 2 = p2.expenditure * 3 ∧
  -- P1's income is 5000
  p1.income = 5000 ∧
  -- Savings is income minus expenditure
  p1.savings = p1.income - p1.expenditure ∧
  p2.savings = p2.income - p2.expenditure ∧
  -- Both persons save the same amount
  p1.savings = p2.savings

/-- The theorem to prove -/
theorem equal_savings (p1 p2 : Person) :
  financialProblem p1 p2 → p1.savings = 2000 ∧ p2.savings = 2000 := by
  sorry

end NUMINAMATH_CALUDE_equal_savings_l3566_356639


namespace NUMINAMATH_CALUDE_max_value_polynomial_l3566_356662

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  ∃ (max : ℝ), max = 72.25 ∧ 
  ∀ (a b : ℝ), a + b = 5 → 
    a^5*b + a^4*b + a^3*b + a*b + a*b^2 + a*b^3 + a*b^5 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l3566_356662


namespace NUMINAMATH_CALUDE_pizza_order_l3566_356634

theorem pizza_order (cost_per_box : ℚ) (tip_ratio : ℚ) (total_paid : ℚ) : 
  cost_per_box = 7 →
  tip_ratio = 1 / 7 →
  total_paid = 40 →
  ∃ (num_boxes : ℕ), 
    (↑num_boxes * cost_per_box) * (1 + tip_ratio) = total_paid ∧
    num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_l3566_356634


namespace NUMINAMATH_CALUDE_tan_double_angle_l3566_356623

theorem tan_double_angle (α : Real) (h : Real.sin α + 2 * Real.cos α = 0) : 
  Real.tan (2 * α) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_double_angle_l3566_356623


namespace NUMINAMATH_CALUDE_shadow_length_l3566_356697

theorem shadow_length (h₁ s₁ h₂ : ℝ) (h_h₁ : h₁ = 20) (h_s₁ : s₁ = 10) (h_h₂ : h₂ = 40) :
  ∃ s₂ : ℝ, s₂ = 20 ∧ h₁ / s₁ = h₂ / s₂ :=
by sorry

end NUMINAMATH_CALUDE_shadow_length_l3566_356697


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3566_356695

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x < 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3566_356695


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l3566_356625

theorem two_digit_number_puzzle :
  ∀ x y : ℕ,
  (10 ≤ 10 * x + y) ∧ (10 * x + y < 100) →  -- two-digit number condition
  (x + y) * 3 = 10 * x + y - 2 →             -- puzzle condition
  x = 2 :=                                   -- conclusion
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l3566_356625


namespace NUMINAMATH_CALUDE_quadrilateral_fourth_angle_l3566_356618

theorem quadrilateral_fourth_angle
  (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 = 100)
  (h2 : angle2 = 60)
  (h3 : angle3 = 90)
  (h_sum : angle1 + angle2 + angle3 + angle4 = 360) :
  angle4 = 110 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_fourth_angle_l3566_356618


namespace NUMINAMATH_CALUDE_binomial_properties_l3566_356690

/-- A random variable following a binomial distribution B(n,p) -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

variable (ξ : BinomialRV)

/-- The expectation of a binomial random variable -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- The probability of getting 0 successes in a binomial distribution -/
def prob_zero (ξ : BinomialRV) : ℝ := (1 - ξ.p) ^ ξ.n

theorem binomial_properties (ξ : BinomialRV) 
  (h2 : 3 * expectation ξ + 2 = 9.2)
  (h3 : 9 * variance ξ = 12.96) :
  ξ.n = 6 ∧ ξ.p = 0.4 ∧ prob_zero ξ = 0.6^6 := by sorry

end NUMINAMATH_CALUDE_binomial_properties_l3566_356690


namespace NUMINAMATH_CALUDE_largest_six_digit_with_factorial_product_l3566_356676

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).foldl (· * ·) 1

theorem largest_six_digit_with_factorial_product :
  ∃ (n : ℕ), 
    100000 ≤ n ∧ 
    n ≤ 999999 ∧ 
    digit_product n = factorial 8 ∧
    ∀ (m : ℕ), 100000 ≤ m ∧ m ≤ 999999 ∧ digit_product m = factorial 8 → m ≤ n :=
by
  use 987542
  sorry

end NUMINAMATH_CALUDE_largest_six_digit_with_factorial_product_l3566_356676


namespace NUMINAMATH_CALUDE_quadratic_trinomial_constant_l3566_356664

/-- Given that x^{|m|}+(m-2)x-10 is a quadratic trinomial where m is a constant, prove that m = -2 -/
theorem quadratic_trinomial_constant (m : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, x^(|m|) + (m-2)*x - 10 = a*x^2 + b*x + c ∧ a ≠ 0 ∧ b ≠ 0) → 
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_constant_l3566_356664


namespace NUMINAMATH_CALUDE_required_weekly_hours_l3566_356640

/-- Calculates the required weekly work hours to meet a financial goal given previous work data and future plans. -/
theorem required_weekly_hours 
  (summer_weeks : ℕ) 
  (summer_hours_per_week : ℕ) 
  (summer_total_earnings : ℚ) 
  (future_weeks : ℕ) 
  (future_earnings_goal : ℚ) : 
  summer_weeks > 0 ∧ 
  summer_hours_per_week > 0 ∧ 
  summer_total_earnings > 0 ∧ 
  future_weeks > 0 ∧ 
  future_earnings_goal > 0 →
  (future_earnings_goal / (summer_total_earnings / (summer_weeks * summer_hours_per_week))) / future_weeks = 45 / 16 := by
  sorry

#eval (4500 : ℚ) / ((3600 : ℚ) / (8 * 45)) / 40

end NUMINAMATH_CALUDE_required_weekly_hours_l3566_356640


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l3566_356659

/-- A circle is defined by the equation x^2 + y^2 + Dx + Ey + F = 0 -/
def Circle (D E F : ℝ) := fun (x y : ℝ) => x^2 + y^2 + D*x + E*y + F = 0

/-- The specific circle we're interested in -/
def SpecificCircle := Circle (-4) (-6) 0

theorem circle_passes_through_points :
  (SpecificCircle 0 0) ∧ 
  (SpecificCircle 4 0) ∧ 
  (SpecificCircle (-1) 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l3566_356659


namespace NUMINAMATH_CALUDE_remainder_777_444_mod_13_l3566_356692

theorem remainder_777_444_mod_13 : 777^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_777_444_mod_13_l3566_356692


namespace NUMINAMATH_CALUDE_pedros_test_scores_l3566_356602

theorem pedros_test_scores :
  let scores : List ℕ := [92, 91, 89, 85, 78]
  let first_three : List ℕ := [92, 85, 78]
  ∀ (s : List ℕ),
    s.length = 5 →
    s.take 3 = first_three →
    s.sum / s.length = 87 →
    (∀ x ∈ s, x < 100) →
    s.Nodup →
    s = scores :=
by sorry

end NUMINAMATH_CALUDE_pedros_test_scores_l3566_356602


namespace NUMINAMATH_CALUDE_slope_parallel_sufficient_not_necessary_l3566_356689

-- Define a structure for a line with a slope
structure Line where
  slope : ℝ

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem slope_parallel_sufficient_not_necessary :
  ∃ (l1 l2 : Line),
    (parallel l1 l2 → l1.slope = l2.slope) ∧
    ∃ (l3 l4 : Line), l3.slope = l4.slope ∧ ¬ parallel l3 l4 := by
  sorry

end NUMINAMATH_CALUDE_slope_parallel_sufficient_not_necessary_l3566_356689


namespace NUMINAMATH_CALUDE_decimal_53_to_binary_binary_to_decimal_53_l3566_356631

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_53_to_binary :
  to_binary 53 = [true, false, true, false, true, true] :=
by sorry

theorem binary_to_decimal_53 :
  from_binary [true, false, true, false, true, true] = 53 :=
by sorry

end NUMINAMATH_CALUDE_decimal_53_to_binary_binary_to_decimal_53_l3566_356631


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3566_356643

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
  (∀ x, 3 * x * (x - 1) = 2 * (x + 2) + 8 ↔ a * x^2 + b * x + c = 0) →
  a = 3 ∧ b = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3566_356643


namespace NUMINAMATH_CALUDE_decimal_division_equality_l3566_356654

theorem decimal_division_equality : (0.05 : ℝ) / 0.005 = 10 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_equality_l3566_356654


namespace NUMINAMATH_CALUDE_cod_fish_sold_l3566_356619

theorem cod_fish_sold (total : ℕ) (haddock_percent : ℚ) (halibut_percent : ℚ) 
  (h1 : total = 220)
  (h2 : haddock_percent = 40 / 100)
  (h3 : halibut_percent = 40 / 100) :
  (total : ℚ) * (1 - haddock_percent - halibut_percent) = 44 := by sorry

end NUMINAMATH_CALUDE_cod_fish_sold_l3566_356619


namespace NUMINAMATH_CALUDE_genevieve_error_count_l3566_356688

/-- The number of lines of code Genevieve has written -/
def total_lines : ℕ := 4300

/-- The number of lines per debug block -/
def lines_per_block : ℕ := 100

/-- The number of errors found in the first block -/
def initial_errors : ℕ := 3

/-- The increase in errors found per block -/
def error_increase : ℕ := 1

/-- The number of completed debug blocks -/
def num_blocks : ℕ := total_lines / lines_per_block

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a₁ : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The total number of errors fixed -/
def total_errors : ℕ := arithmetic_sum num_blocks initial_errors error_increase

theorem genevieve_error_count :
  total_errors = 1032 := by sorry

end NUMINAMATH_CALUDE_genevieve_error_count_l3566_356688


namespace NUMINAMATH_CALUDE_simplify_expression_l3566_356672

theorem simplify_expression (x y : ℝ) : 3*x + 4*x - 2*x + 5*y - y = 5*x + 4*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3566_356672


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l3566_356603

/-- A circle C in the xy-plane -/
structure Circle where
  m : ℝ
  equation : ℝ → ℝ → Prop :=
    fun x y => x^2 + y^2 + m*x - 4 = 0

/-- A line in the xy-plane -/
def symmetry_line : ℝ → ℝ → Prop :=
  fun x y => x - y + 4 = 0

/-- Two points are symmetric with respect to a line -/
def symmetric (p1 p2 : ℝ × ℝ) (L : ℝ → ℝ → Prop) : Prop := sorry

theorem circle_symmetry_line (C : Circle) :
  (∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ 
    C.equation p1.1 p1.2 ∧ 
    C.equation p2.1 p2.2 ∧ 
    symmetric p1 p2 symmetry_line) →
  C.m = 8 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l3566_356603


namespace NUMINAMATH_CALUDE_nathalie_cake_fraction_l3566_356607

theorem nathalie_cake_fraction (cake_weight : ℝ) (num_parts : ℕ) 
  (pierre_amount : ℝ) (nathalie_fraction : ℝ) : 
  cake_weight = 400 →
  num_parts = 8 →
  pierre_amount = 100 →
  pierre_amount = 2 * (nathalie_fraction * cake_weight) →
  nathalie_fraction = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_nathalie_cake_fraction_l3566_356607


namespace NUMINAMATH_CALUDE_triangle_area_l3566_356667

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if A = 2π/3, a = 7, and b = 3, then the area of the triangle S_ABC = 15√3/4 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  A = 2 * Real.pi / 3 →
  a = 7 →
  b = 3 →
  (∃ (S_ABC : ℝ), S_ABC = (15 * Real.sqrt 3) / 4 ∧ S_ABC = (1/2) * b * c * Real.sin A) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3566_356667


namespace NUMINAMATH_CALUDE_alyssa_cut_roses_l3566_356661

/-- Represents the number of roses Alyssa cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Proves that Alyssa cut 11 roses given the initial and final number of roses -/
theorem alyssa_cut_roses : roses_cut 3 14 = 11 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_cut_roses_l3566_356661


namespace NUMINAMATH_CALUDE_cow_increase_is_24_l3566_356678

/-- Represents the number of cows at different stages --/
structure CowCount where
  initial : Nat
  after_deaths : Nat
  after_sales : Nat
  current : Nat

/-- Calculates the increase in cow count given the initial conditions and final count --/
def calculate_increase (c : CowCount) (bought : Nat) (gifted : Nat) : Nat :=
  c.current - (c.after_sales + bought + gifted)

/-- Theorem stating that the increase in cows is 24 given the problem conditions --/
theorem cow_increase_is_24 :
  let c := CowCount.mk 39 (39 - 25) ((39 - 25) - 6) 83
  let bought := 43
  let gifted := 8
  calculate_increase c bought gifted = 24 := by
  sorry

end NUMINAMATH_CALUDE_cow_increase_is_24_l3566_356678


namespace NUMINAMATH_CALUDE_bisecting_line_min_value_l3566_356666

/-- A line that bisects the circumference of a circle -/
structure BisetingLine where
  a : ℝ
  b : ℝ
  h1 : a ≥ b
  h2 : b > 0
  h3 : ∀ (x y : ℝ), a * x + 2 * b * y - 2 = 0 → x^2 + y^2 - 4*x - 2*y - 8 = 0

/-- The minimum value of 1/a + 2/b for a bisecting line is 6 -/
theorem bisecting_line_min_value (l : BisetingLine) : 
  (∀ (a' b' : ℝ), a' ≥ b' ∧ b' > 0 → 1 / a' + 2 / b' ≥ 1 / l.a + 2 / l.b) ∧
  1 / l.a + 2 / l.b = 6 :=
sorry

end NUMINAMATH_CALUDE_bisecting_line_min_value_l3566_356666


namespace NUMINAMATH_CALUDE_infinite_fraction_equals_sqrt_15_l3566_356617

theorem infinite_fraction_equals_sqrt_15 :
  ∃ x : ℝ, x > 0 ∧ x = 3 + 5 / (2 + 5 / x) → x = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_infinite_fraction_equals_sqrt_15_l3566_356617


namespace NUMINAMATH_CALUDE_total_combinations_eq_40_l3566_356671

/-- Represents the number of helper options for each day of the week --/
def helperOptions : Fin 5 → ℕ
  | 0 => 1  -- Monday
  | 1 => 2  -- Tuesday
  | 2 => 4  -- Wednesday
  | 3 => 5  -- Thursday
  | 4 => 1  -- Friday

/-- The total number of different combinations of helpers for the week --/
def totalCombinations : ℕ := (List.range 5).map helperOptions |>.prod

/-- Theorem stating that the total number of combinations is 40 --/
theorem total_combinations_eq_40 : totalCombinations = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_eq_40_l3566_356671


namespace NUMINAMATH_CALUDE_characterization_of_function_l3566_356644

theorem characterization_of_function (f : ℤ → ℝ) 
  (h1 : ∀ m n : ℤ, m < n → f m < f n)
  (h2 : ∀ m n : ℤ, ∃ k : ℤ, f m - f n = f k) :
  ∃ a : ℝ, ∃ t : ℤ, a > 0 ∧ ∀ n : ℤ, f n = a * (n + t) := by
sorry

end NUMINAMATH_CALUDE_characterization_of_function_l3566_356644


namespace NUMINAMATH_CALUDE_problem_statement_l3566_356677

theorem problem_statement (x : ℝ) (h : x^2 + x = 1) :
  3*x^4 + 3*x^3 + 3*x + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3566_356677


namespace NUMINAMATH_CALUDE_circle_tangent_parallel_l3566_356653

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric relationships
variable (circumscribed : Circle → Point → Point → Point → Point → Prop)
variable (passes_through : Circle → Point → Point → Prop)
variable (intersects : Point → Point → Circle → Point → Prop)
variable (tangent_at : Circle → Point → Point → Point → Prop)
variable (parallel : Point → Point → Point → Point → Prop)

-- State the theorem
theorem circle_tangent_parallel 
  (ω₁ ω₂ : Circle) 
  (A B C D E F : Point) :
  circumscribed ω₁ A B C D →
  passes_through ω₂ A B →
  intersects D B ω₂ E →
  intersects C A ω₂ F →
  E ≠ B →
  F ≠ A →
  tangent_at ω₁ C A E →
  tangent_at ω₂ F A D :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_parallel_l3566_356653


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l3566_356680

theorem binomial_expansion_example : 100 + 2 * (10 * 3) + 9 = (10 + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l3566_356680


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3566_356663

/-- Given a geometric sequence {a_n} where a_4 = 4, prove that a_3 * a_5 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h4 : a 4 = 4) : a 3 * a 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3566_356663


namespace NUMINAMATH_CALUDE_wang_house_number_l3566_356650

def is_valid_triplet (a b c : ℕ) : Prop :=
  a * b * c = 40 ∧ a > 0 ∧ b > 0 ∧ c > 0

def house_number (a b c : ℕ) : ℕ := a + b + c

def is_ambiguous (n : ℕ) : Prop :=
  ∃ a₁ b₁ c₁ a₂ b₂ c₂, 
    is_valid_triplet a₁ b₁ c₁ ∧ 
    is_valid_triplet a₂ b₂ c₂ ∧ 
    house_number a₁ b₁ c₁ = n ∧ 
    house_number a₂ b₂ c₂ = n ∧ 
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂)

theorem wang_house_number : 
  ∃! n, is_ambiguous n ∧ ∀ m, is_ambiguous m → m = n :=
by
  sorry

end NUMINAMATH_CALUDE_wang_house_number_l3566_356650


namespace NUMINAMATH_CALUDE_S_2023_eq_half_l3566_356630

def S : ℕ → ℚ
  | 0 => 1 / 2
  | n + 1 => if n % 2 = 0 then 1 / S n else -S n - 1

theorem S_2023_eq_half : S 2022 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_S_2023_eq_half_l3566_356630


namespace NUMINAMATH_CALUDE_parabola_vertex_l3566_356641

/-- The parabola equation -/
def parabola_equation (x : ℝ) : ℝ := x^2 - 4*x + 7

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 3)

/-- Theorem: The vertex of the parabola y = x^2 - 4x + 7 is at the point (2, 3) -/
theorem parabola_vertex :
  let (h, k) := vertex
  (∀ x, parabola_equation x = (x - h)^2 + k) ∧
  (∀ x, parabola_equation x ≥ parabola_equation h) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3566_356641


namespace NUMINAMATH_CALUDE_raw_silk_calculation_l3566_356652

/-- The amount of raw silk that results in 12 pounds of dried silk -/
def original_raw_silk : ℚ := 96 / 7

/-- The weight loss during drying in pounds -/
def weight_loss : ℚ := 3 + 12 / 16

theorem raw_silk_calculation (initial_raw : ℚ) (dried : ℚ) 
  (h1 : initial_raw = 30)
  (h2 : dried = 12)
  (h3 : initial_raw - weight_loss = dried) :
  original_raw_silk * (initial_raw - weight_loss) = dried * initial_raw :=
sorry

end NUMINAMATH_CALUDE_raw_silk_calculation_l3566_356652


namespace NUMINAMATH_CALUDE_basketball_team_combinations_l3566_356693

/-- The number of players in the basketball team -/
def total_players : ℕ := 12

/-- The number of players in the starting lineup (excluding the captain) -/
def starting_lineup : ℕ := 5

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The number of ways to select 1 captain from 12 players and then 5 players 
    from the remaining 11 for the starting lineup is equal to 5544 -/
theorem basketball_team_combinations : 
  total_players * choose (total_players - 1) starting_lineup = 5544 := by
  sorry


end NUMINAMATH_CALUDE_basketball_team_combinations_l3566_356693


namespace NUMINAMATH_CALUDE_odd_function_iff_graph_symmetry_solution_exists_when_p_zero_more_than_two_solutions_l3566_356686

-- Define the function f(x)
def f (p q x : ℝ) : ℝ := x * abs x + p * x + q

-- Statement 1: f(x) is an odd function if and only if q = 0
theorem odd_function_iff (p q : ℝ) :
  (∀ x : ℝ, f p q (-x) = -(f p q x)) ↔ q = 0 := by sorry

-- Statement 2: The graph of f(x) is symmetric about the point (0, q)
theorem graph_symmetry (p q : ℝ) :
  ∀ x : ℝ, f p q (x) - q = -(f p q (-x) - q) := by sorry

-- Statement 3: When p = 0, the equation f(x) = 0 always has at least one solution
theorem solution_exists_when_p_zero (q : ℝ) :
  ∃ x : ℝ, f 0 q x = 0 := by sorry

-- Statement 4: There exists a combination of p and q such that f(x) = 0 has more than two solutions
theorem more_than_two_solutions :
  ∃ p q : ℝ, ∃ x₁ x₂ x₃ : ℝ, (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    (f p q x₁ = 0 ∧ f p q x₂ = 0 ∧ f p q x₃ = 0) := by sorry

end NUMINAMATH_CALUDE_odd_function_iff_graph_symmetry_solution_exists_when_p_zero_more_than_two_solutions_l3566_356686


namespace NUMINAMATH_CALUDE_product_remainder_l3566_356621

theorem product_remainder (x y : ℤ) 
  (hx : x % 792 = 62) 
  (hy : y % 528 = 82) : 
  (x * y) % 66 = 24 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l3566_356621


namespace NUMINAMATH_CALUDE_inequality_proof_l3566_356624

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3566_356624


namespace NUMINAMATH_CALUDE_sum_of_N_and_K_is_8_l3566_356679

/-- The complex conjugate of a complex number -/
noncomputable def conj (z : ℂ) : ℂ := sorry

/-- The transformation function g -/
noncomputable def g (z : ℂ) : ℂ := 2 * Complex.I * conj z

/-- The polynomial P -/
def P (z : ℂ) : ℂ := z^4 + 6*z^3 + 2*z^2 + 4*z + 1

/-- The roots of P -/
noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry
noncomputable def z4 : ℂ := sorry

/-- The polynomial R -/
noncomputable def R (z : ℂ) : ℂ := z^4 + M*z^3 + N*z^2 + L*z + K
  where
  M : ℂ := sorry
  N : ℂ := sorry
  L : ℂ := sorry
  K : ℂ := sorry

theorem sum_of_N_and_K_is_8 : N + K = 8 := by sorry

end NUMINAMATH_CALUDE_sum_of_N_and_K_is_8_l3566_356679


namespace NUMINAMATH_CALUDE_geometric_sequence_304th_term_l3566_356615

/-- Given a geometric sequence with first term 8 and second term -8, the 304th term is -8 -/
theorem geometric_sequence_304th_term :
  ∀ (a : ℕ → ℝ), 
    a 1 = 8 →  -- First term is 8
    a 2 = -8 →  -- Second term is -8
    (∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence property
    a 304 = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_304th_term_l3566_356615


namespace NUMINAMATH_CALUDE_remaining_pictures_to_color_l3566_356604

/-- The number of pictures in each coloring book -/
def pictures_per_book : ℕ := 44

/-- The number of coloring books -/
def num_books : ℕ := 2

/-- The number of pictures already colored -/
def colored_pictures : ℕ := 20

/-- Theorem: Given two coloring books with 44 pictures each, and 20 pictures already colored,
    the number of pictures left to color is 68. -/
theorem remaining_pictures_to_color :
  (num_books * pictures_per_book) - colored_pictures = 68 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pictures_to_color_l3566_356604


namespace NUMINAMATH_CALUDE_P_in_first_quadrant_l3566_356668

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The point P with coordinates (2,1) -/
def P : Point :=
  { x := 2, y := 1 }

/-- Theorem stating that P lies in the first quadrant -/
theorem P_in_first_quadrant : isInFirstQuadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_first_quadrant_l3566_356668


namespace NUMINAMATH_CALUDE_r_upper_bound_r_7_upper_bound_l3566_356626

/-- The maximum number of pieces that can be placed on an n × n chessboard
    without forming a rectangle with sides parallel to grid lines. -/
def r (n : ℕ) : ℕ := sorry

/-- Theorem: Upper bound for r(n) -/
theorem r_upper_bound (n : ℕ) : r n ≤ (n + n * Real.sqrt (4 * n - 3)) / 2 := by sorry

/-- Theorem: Upper bound for r(7) -/
theorem r_7_upper_bound : r 7 ≤ 21 := by sorry

end NUMINAMATH_CALUDE_r_upper_bound_r_7_upper_bound_l3566_356626


namespace NUMINAMATH_CALUDE_adam_room_capacity_l3566_356637

/-- The number of action figures that can fit on each shelf. -/
def figures_per_shelf : ℕ := 10

/-- The number of shelves in Adam's room. -/
def number_of_shelves : ℕ := 8

/-- The total number of action figures Adam's room could hold. -/
def total_figures : ℕ := figures_per_shelf * number_of_shelves

/-- Theorem stating that the total number of action figures Adam's room could hold is 80. -/
theorem adam_room_capacity : total_figures = 80 := by sorry

end NUMINAMATH_CALUDE_adam_room_capacity_l3566_356637


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l3566_356627

def arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_specific :
  let a₁ : ℤ := -3
  let d : ℤ := 6
  let n : ℕ := 8
  let aₙ : ℤ := a₁ + (n - 1) * d
  aₙ = 39 →
  arithmetic_sequence_sum a₁ d n = 144 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l3566_356627


namespace NUMINAMATH_CALUDE_rebecca_egg_marble_difference_l3566_356660

/-- Given that Rebecca has 20 eggs and 6 marbles, prove that she has 14 more eggs than marbles. -/
theorem rebecca_egg_marble_difference :
  let eggs : ℕ := 20
  let marbles : ℕ := 6
  eggs - marbles = 14 := by sorry

end NUMINAMATH_CALUDE_rebecca_egg_marble_difference_l3566_356660


namespace NUMINAMATH_CALUDE_area_between_parabola_and_line_l3566_356657

-- Define the parabola function
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the line function
def line (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem area_between_parabola_and_line :
  ∫ x in (0)..(1), (line x - parabola x) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_area_between_parabola_and_line_l3566_356657


namespace NUMINAMATH_CALUDE_unique_prime_sum_diff_l3566_356616

theorem unique_prime_sum_diff : ∃! p : ℕ, 
  Prime p ∧ 
  (∃ q r : ℕ, Prime q ∧ Prime r ∧ p = q + r) ∧ 
  (∃ s t : ℕ, Prime s ∧ Prime t ∧ p = s - t) :=
by
  use 5
  sorry

end NUMINAMATH_CALUDE_unique_prime_sum_diff_l3566_356616


namespace NUMINAMATH_CALUDE_relationship_one_l3566_356665

theorem relationship_one (a b : ℝ) : (a - b)^2 + (a * b + 1)^2 = (a^2 + 1) * (b^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_relationship_one_l3566_356665


namespace NUMINAMATH_CALUDE_parallelogram_smaller_angle_measure_l3566_356614

/-- 
A parallelogram with one angle exceeding the other by 50 degrees has a smaller angle of 65 degrees.
-/
theorem parallelogram_smaller_angle_measure : 
  ∀ (a b : ℝ), a > 0 → b > 0 → 
  (a = b + 50) → (a + b = 180) →
  b = 65 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_smaller_angle_measure_l3566_356614


namespace NUMINAMATH_CALUDE_evening_painting_l3566_356645

/-- A dodecahedron is a polyhedron with 12 faces -/
def dodecahedron_faces : ℕ := 12

/-- The number of faces Samuel painted in the morning -/
def painted_faces : ℕ := 5

/-- The number of faces Samuel needs to paint in the evening -/
def remaining_faces : ℕ := dodecahedron_faces - painted_faces

theorem evening_painting : remaining_faces = 7 := by
  sorry

end NUMINAMATH_CALUDE_evening_painting_l3566_356645


namespace NUMINAMATH_CALUDE_fraction_simplification_and_evaluation_l3566_356681

theorem fraction_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 0 →
  (2 / (x^2 - 4)) / (1 / (x^2 - 2*x)) = 2*x / (x + 2) ∧
  (2 * (-1)) / ((-1) + 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_and_evaluation_l3566_356681


namespace NUMINAMATH_CALUDE_prob_exactly_one_correct_l3566_356606

variable (p₁ p₂ : ℝ)

-- A and B independently solve the same problem
axiom prob_A : 0 ≤ p₁ ∧ p₁ ≤ 1
axiom prob_B : 0 ≤ p₂ ∧ p₂ ≤ 1

-- The probability that exactly one person solves the problem
def prob_exactly_one : ℝ := p₁ * (1 - p₂) + p₂ * (1 - p₁)

-- Theorem stating that the probability of exactly one person solving is correct
theorem prob_exactly_one_correct :
  prob_exactly_one p₁ p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end NUMINAMATH_CALUDE_prob_exactly_one_correct_l3566_356606


namespace NUMINAMATH_CALUDE_easier_decryption_with_more_unique_letters_l3566_356628

def unique_letters (s : String) : Finset Char :=
  s.toList.toFinset

theorem easier_decryption_with_more_unique_letters 
  (word1 : String) (word2 : String) 
  (h1 : word1 = "термометр") (h2 : word2 = "ремонт") :
  (unique_letters word2).card > (unique_letters word1).card :=
by sorry

end NUMINAMATH_CALUDE_easier_decryption_with_more_unique_letters_l3566_356628


namespace NUMINAMATH_CALUDE_coin_distribution_formula_l3566_356685

/-- An arithmetic sequence representing the distribution of coins among people. -/
def CoinDistribution (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem coin_distribution_formula 
  (a₁ d : ℚ) 
  (h1 : (CoinDistribution a₁ d 1) + (CoinDistribution a₁ d 2) = 
        (CoinDistribution a₁ d 3) + (CoinDistribution a₁ d 4) + (CoinDistribution a₁ d 5))
  (h2 : (CoinDistribution a₁ d 1) + (CoinDistribution a₁ d 2) + (CoinDistribution a₁ d 3) + 
        (CoinDistribution a₁ d 4) + (CoinDistribution a₁ d 5) = 5) :
  ∀ n : ℕ, n ≥ 1 → n ≤ 5 → CoinDistribution a₁ d n = -1/6 * n + 3/2 :=
sorry

end NUMINAMATH_CALUDE_coin_distribution_formula_l3566_356685


namespace NUMINAMATH_CALUDE_fraction_simplification_l3566_356656

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3566_356656


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3566_356611

theorem quadratic_inequality_condition (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3566_356611


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l3566_356609

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) :
  (2 * r) ^ 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l3566_356609


namespace NUMINAMATH_CALUDE_hexagon_side_length_l3566_356684

/-- A regular hexagon with perimeter 42 cm has sides of length 7 cm each. -/
theorem hexagon_side_length (perimeter : ℝ) (num_sides : ℕ) (h1 : perimeter = 42) (h2 : num_sides = 6) :
  perimeter / num_sides = 7 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l3566_356684


namespace NUMINAMATH_CALUDE_wheel_configuration_theorem_l3566_356600

/-- Represents a wheel with spokes -/
structure Wheel where
  spokes : ℕ

/-- Represents a configuration of wheels -/
structure WheelConfiguration where
  wheels : List Wheel
  total_spokes : ℕ
  max_visible_spokes : ℕ

/-- Checks if a configuration is valid based on the problem conditions -/
def is_valid_configuration (config : WheelConfiguration) : Prop :=
  config.total_spokes ≥ 7 ∧
  config.max_visible_spokes ≤ 3 ∧
  (∀ w ∈ config.wheels, w.spokes ≤ config.max_visible_spokes)

theorem wheel_configuration_theorem :
  ∃ (config_three : WheelConfiguration),
    config_three.wheels.length = 3 ∧
    is_valid_configuration config_three ∧
  ¬∃ (config_two : WheelConfiguration),
    config_two.wheels.length = 2 ∧
    is_valid_configuration config_two :=
sorry

end NUMINAMATH_CALUDE_wheel_configuration_theorem_l3566_356600


namespace NUMINAMATH_CALUDE_ball_drawing_exclusivity_l3566_356633

structure Ball :=
  (color : String)

def Bag := Multiset Ball

def draw (bag : Bag) (n : ℕ) := Multiset Ball

def atLeastOneWhite (draw : Multiset Ball) : Prop := sorry
def bothWhite (draw : Multiset Ball) : Prop := sorry
def atLeastOneRed (draw : Multiset Ball) : Prop := sorry
def exactlyOneWhite (draw : Multiset Ball) : Prop := sorry
def exactlyTwoWhite (draw : Multiset Ball) : Prop := sorry
def bothRed (draw : Multiset Ball) : Prop := sorry

def mutuallyExclusive (e1 e2 : Multiset Ball → Prop) : Prop := sorry

def initialBag : Bag := sorry

theorem ball_drawing_exclusivity :
  let result := draw initialBag 2
  (mutuallyExclusive (exactlyOneWhite) (exactlyTwoWhite)) ∧
  (mutuallyExclusive (atLeastOneWhite) (bothRed)) ∧
  ¬(mutuallyExclusive (atLeastOneWhite) (bothWhite)) ∧
  ¬(mutuallyExclusive (atLeastOneWhite) (atLeastOneRed)) := by sorry

end NUMINAMATH_CALUDE_ball_drawing_exclusivity_l3566_356633


namespace NUMINAMATH_CALUDE_max_f_value_l3566_356691

/-- The function f(n) is the greatest common divisor of all numbers 
    obtained by permuting the digits of n, including permutations 
    with leading zeroes. -/
def f (n : ℕ+) : ℕ := sorry

/-- Theorem: The maximum value of f(n) for positive integers n 
    where f(n) ≠ n is 81. -/
theorem max_f_value : 
  (∃ (n : ℕ+), f n = 81 ∧ f n ≠ n) ∧ 
  (∀ (n : ℕ+), f n ≠ n → f n ≤ 81) :=
sorry

end NUMINAMATH_CALUDE_max_f_value_l3566_356691


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_l3566_356620

theorem coefficient_x3y5_in_expansion (x y : ℝ) :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * x^k * y^(8-k)) =
  56 * x^3 * y^5 + (Finset.range 9).sum (fun k => if k ≠ 3 then Nat.choose 8 k * x^k * y^(8-k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_l3566_356620


namespace NUMINAMATH_CALUDE_janet_action_figures_l3566_356608

theorem janet_action_figures (initial : ℕ) (sold : ℕ) (final_total : ℕ) :
  initial = 10 →
  sold = 6 →
  final_total = 24 →
  let remaining := initial - sold
  let brother_gift := 2 * remaining
  let before_new := remaining + brother_gift
  final_total - before_new = 12 :=
by sorry

end NUMINAMATH_CALUDE_janet_action_figures_l3566_356608


namespace NUMINAMATH_CALUDE_distance_cos80_sin80_to_cos20_sin20_l3566_356636

/-- The distance between points (cos 80°, sin 80°) and (cos 20°, sin 20°) is 1. -/
theorem distance_cos80_sin80_to_cos20_sin20 : 
  let A : ℝ × ℝ := (Real.cos (80 * π / 180), Real.sin (80 * π / 180))
  let B : ℝ × ℝ := (Real.cos (20 * π / 180), Real.sin (20 * π / 180))
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1 := by
sorry

end NUMINAMATH_CALUDE_distance_cos80_sin80_to_cos20_sin20_l3566_356636


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3566_356649

theorem necessary_but_not_sufficient (x y : ℝ) :
  (¬ ((x > 3) ∨ (y > 2)) → ¬ (x + y > 5)) ∧
  ¬ ((x > 3) ∨ (y > 2) → (x + y > 5)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3566_356649


namespace NUMINAMATH_CALUDE_min_sum_of_quadratic_roots_l3566_356642

theorem min_sum_of_quadratic_roots (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + a*x + 2*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 2*b*x + a = 0) :
  a + b ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_quadratic_roots_l3566_356642


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l3566_356698

theorem quadratic_root_sum (m n : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 2*n = 0 ∧ x = 2) → m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l3566_356698


namespace NUMINAMATH_CALUDE_purple_cars_count_l3566_356658

theorem purple_cars_count (purple red green : ℕ) : 
  green = 4 * red →
  red = purple + 6 →
  purple + red + green = 312 →
  purple = 47 := by
sorry

end NUMINAMATH_CALUDE_purple_cars_count_l3566_356658


namespace NUMINAMATH_CALUDE_foggy_day_walk_l3566_356673

/-- Represents a person walking on a straight road -/
structure Walker where
  speed : ℝ
  position : ℝ

/-- The problem setup and solution -/
theorem foggy_day_walk (visibility : ℝ) (alex ben : Walker) (initial_time : ℝ) :
  visibility = 100 →
  alex.speed = 4 →
  ben.speed = 6 →
  initial_time = 60 →
  alex.position = alex.speed * initial_time →
  ben.position = ben.speed * initial_time →
  ∃ (meeting_time : ℝ),
    meeting_time = 50 ∧
    abs (alex.position - alex.speed * meeting_time - (ben.position - ben.speed * meeting_time)) = visibility ∧
    abs (alex.position - alex.speed * meeting_time) = 40 ∧
    abs (ben.position - ben.speed * meeting_time) = 60 :=
by sorry

end NUMINAMATH_CALUDE_foggy_day_walk_l3566_356673


namespace NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l3566_356699

/-- Given a point P(3, -4), prove that its symmetric point P' about the x-axis has coordinates (3, 4) -/
theorem symmetric_point_about_x_axis :
  let P : ℝ × ℝ := (3, -4)
  let P' : ℝ × ℝ := (P.1, -P.2)
  P' = (3, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l3566_356699


namespace NUMINAMATH_CALUDE_f_zero_f_expression_intersection_complement_l3566_356655

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Given conditions
axiom f_property : ∀ (x y : ℝ), f (x + y) - f y = x * (x + 2 * y + 1)
axiom f_one : f 1 = 0

-- Define sets A and B
def A : Set ℝ := {a | ∀ x ∈ Set.Ioo 0 (1/2), f x + 3 < 2 * x + a}
def B : Set ℝ := {a | ∀ x ∈ Set.Icc (-2) 2, Monotone (fun x ↦ f x - a * x)}

-- Theorem statements
theorem f_zero : f 0 = -2 := sorry

theorem f_expression : ∀ x : ℝ, f x = x^2 + x - 2 := sorry

theorem intersection_complement : A ∩ (Set.univ \ B) = Set.Icc 1 5 := sorry

end NUMINAMATH_CALUDE_f_zero_f_expression_intersection_complement_l3566_356655
