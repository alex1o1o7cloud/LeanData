import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3447_344797

/-- Given real numbers m and n where m < n, the quadratic inequality
    x^2 - (m + n)x + mn > 0 has the solution set (-∞, m) ∪ (n, +∞),
    and specifying a = 1 makes this representation unique. -/
theorem quadratic_inequality_solution_set (m n : ℝ) (h : m < n) :
  ∃ (a b c : ℝ), a = 1 ∧
    (∀ x, a * x^2 + b * x + c > 0 ↔ x < m ∨ x > n) ∧
    (b = -(m + n) ∧ c = m * n) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3447_344797


namespace NUMINAMATH_CALUDE_integer_equation_existence_l3447_344794

theorem integer_equation_existence :
  (¬ ∃ (m n : ℕ+), m * (m + 2) = n * (n + 1)) ∧
  (¬ ∃ (m n : ℕ+), m * (m + 3) = n * (n + 1)) ∧
  (∀ k : ℕ+, k ≥ 4 → ∃ (m n : ℕ+), m * (m + k) = n * (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_integer_equation_existence_l3447_344794


namespace NUMINAMATH_CALUDE_combination_sum_equality_l3447_344701

def combination (n m : ℕ) : ℚ :=
  if n ≥ m then
    (List.range m).foldl (λ acc i => acc * (n - i : ℚ) / (i + 1)) 1
  else 0

theorem combination_sum_equality : combination 9 4 + combination 9 5 = combination 10 5 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equality_l3447_344701


namespace NUMINAMATH_CALUDE_largest_prime_common_divisor_l3447_344745

theorem largest_prime_common_divisor :
  ∃ (n : ℕ), n.Prime ∧ n ∣ 360 ∧ n ∣ 231 ∧
  ∀ (m : ℕ), m.Prime → m ∣ 360 → m ∣ 231 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_common_divisor_l3447_344745


namespace NUMINAMATH_CALUDE_C2H6_C3H8_impossible_l3447_344755

-- Define the heat released by combustion of 1 mol of each hydrocarbon
def heat_CH4 : ℝ := 889.5
def heat_C2H6 : ℝ := 1558.35
def heat_C2H4 : ℝ := 1409.6
def heat_C2H2 : ℝ := 1298.35
def heat_C3H8 : ℝ := 2217.8

-- Define the total heat released by the mixture
def total_heat : ℝ := 3037.6

-- Define the number of moles in the mixture
def total_moles : ℝ := 2

-- Theorem to prove that C₂H₆ and C₃H₈ combination is impossible
theorem C2H6_C3H8_impossible : 
  ¬(∃ (x y : ℝ), x + y = total_moles ∧ 
                  x * heat_C2H6 + y * heat_C3H8 = total_heat ∧
                  x > 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_C2H6_C3H8_impossible_l3447_344755


namespace NUMINAMATH_CALUDE_domain_of_linear_function_domain_of_rational_function_domain_of_square_root_function_domain_of_reciprocal_square_root_function_domain_of_rational_function_with_linear_denominator_domain_of_arcsin_function_l3447_344732

-- Function 1: z = 4 - x - 2y
theorem domain_of_linear_function (x y : ℝ) :
  ∃ z : ℝ, z = 4 - x - 2*y :=
sorry

-- Function 2: p = 3 / (x^2 + y^2)
theorem domain_of_rational_function (x y : ℝ) :
  (x ≠ 0 ∨ y ≠ 0) → ∃ p : ℝ, p = 3 / (x^2 + y^2) :=
sorry

-- Function 3: z = √(1 - x^2 - y^2)
theorem domain_of_square_root_function (x y : ℝ) :
  x^2 + y^2 ≤ 1 → ∃ z : ℝ, z = Real.sqrt (1 - x^2 - y^2) :=
sorry

-- Function 4: q = 1 / √(xy)
theorem domain_of_reciprocal_square_root_function (x y : ℝ) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) → ∃ q : ℝ, q = 1 / Real.sqrt (x*y) :=
sorry

-- Function 5: u = (x^2 * y) / (2x + 1 - y)
theorem domain_of_rational_function_with_linear_denominator (x y : ℝ) :
  2*x + 1 - y ≠ 0 → ∃ u : ℝ, u = (x^2 * y) / (2*x + 1 - y) :=
sorry

-- Function 6: v = arcsin(x + y)
theorem domain_of_arcsin_function (x y : ℝ) :
  -1 ≤ x + y ∧ x + y ≤ 1 → ∃ v : ℝ, v = Real.arcsin (x + y) :=
sorry

end NUMINAMATH_CALUDE_domain_of_linear_function_domain_of_rational_function_domain_of_square_root_function_domain_of_reciprocal_square_root_function_domain_of_rational_function_with_linear_denominator_domain_of_arcsin_function_l3447_344732


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3447_344758

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3447_344758


namespace NUMINAMATH_CALUDE_cosine_difference_simplification_l3447_344792

theorem cosine_difference_simplification (α β γ : ℝ) :
  Real.cos (α - β) * Real.cos (β - γ) - Real.sin (α - β) * Real.sin (β - γ) = Real.cos (α - γ) := by
  sorry

end NUMINAMATH_CALUDE_cosine_difference_simplification_l3447_344792


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3447_344772

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 1.5) : 
  ∃ (y : ℝ), y = 4 * x * (3 - 2 * x) ∧ ∀ (z : ℝ), z = 4 * x * (3 - 2 * x) → z ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3447_344772


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3447_344734

/-- Given that f(x) = x³(a·2ˣ - 2⁻ˣ) is an even function, prove that a = 1 -/
theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3447_344734


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3447_344724

theorem quadratic_equation_solution (a b c x₁ x₂ y₁ y₂ : ℝ) 
  (hb : b ≠ 0)
  (h1 : x₁^2 + a*x₂^2 = b)
  (h2 : x₂*y₁ - x₁*y₂ = a)
  (h3 : x₁*y₁ + a*x₂*y₂ = c) :
  y₁^2 + a*y₂^2 = (a^3 + c^2) / b := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3447_344724


namespace NUMINAMATH_CALUDE_f_is_monotonic_and_odd_l3447_344776

-- Define the function f(x) = -x
def f (x : ℝ) : ℝ := -x

-- State the theorem
theorem f_is_monotonic_and_odd :
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry


end NUMINAMATH_CALUDE_f_is_monotonic_and_odd_l3447_344776


namespace NUMINAMATH_CALUDE_Diamond_evaluation_l3447_344709

-- Define the Diamond operation
def Diamond (a b : ℝ) : ℝ := a * b^2 - b + 1

-- Theorem statement
theorem Diamond_evaluation : Diamond (Diamond 2 3) 4 = 253 := by
  sorry

end NUMINAMATH_CALUDE_Diamond_evaluation_l3447_344709


namespace NUMINAMATH_CALUDE_grid_marking_theorem_l3447_344705

/-- A bipartite graph representation of a 50x50 grid -/
structure Grid :=
  (edges : Finset (Fin 50 × Fin 50))

/-- Marking of edges in the grid -/
def Marking (g : Grid) := Finset (Fin 50 × Fin 50)

/-- Check if a marking results in even degree for all vertices -/
def isValidMarking (g : Grid) (m : Marking g) : Prop :=
  ∀ i : Fin 50, Even ((g.edges.filter (λ e => e.1 = i ∨ e.2 = i)).card - 
                      (m.filter (λ e => e.1 = i ∨ e.2 = i)).card)

theorem grid_marking_theorem (g : Grid) : 
  ∃ m : Marking g, m.card ≤ 99 ∧ isValidMarking g m := by
  sorry


end NUMINAMATH_CALUDE_grid_marking_theorem_l3447_344705


namespace NUMINAMATH_CALUDE_gcd_of_75_and_100_l3447_344725

theorem gcd_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_75_and_100_l3447_344725


namespace NUMINAMATH_CALUDE_cattle_selling_price_l3447_344727

/-- Proves that the selling price per pound for cattle is $1.60 given the specified conditions --/
theorem cattle_selling_price
  (num_cattle : ℕ)
  (purchase_price : ℝ)
  (feed_cost_percentage : ℝ)
  (weight_per_cattle : ℝ)
  (profit : ℝ)
  (h1 : num_cattle = 100)
  (h2 : purchase_price = 40000)
  (h3 : feed_cost_percentage = 0.20)
  (h4 : weight_per_cattle = 1000)
  (h5 : profit = 112000)
  : ∃ (selling_price_per_pound : ℝ),
    selling_price_per_pound = 1.60 ∧
    selling_price_per_pound * (num_cattle * weight_per_cattle) =
      purchase_price + (feed_cost_percentage * purchase_price) + profit :=
by
  sorry

end NUMINAMATH_CALUDE_cattle_selling_price_l3447_344727


namespace NUMINAMATH_CALUDE_union_equals_one_two_three_l3447_344722

def M : Set ℤ := {1, 3}
def N (a : ℤ) : Set ℤ := {1 - a, 3}

theorem union_equals_one_two_three (a : ℤ) : 
  M ∪ N a = {1, 2, 3} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_one_two_three_l3447_344722


namespace NUMINAMATH_CALUDE_green_peppers_half_total_l3447_344703

/-- The weight of green peppers bought by Dale's Vegetarian Restaurant -/
def green_peppers_weight : ℝ := 2.8333333333333335

/-- The total weight of peppers bought by Dale's Vegetarian Restaurant -/
def total_peppers_weight : ℝ := 5.666666666666667

/-- Theorem stating that the weight of green peppers is half the total weight of peppers -/
theorem green_peppers_half_total :
  green_peppers_weight = total_peppers_weight / 2 :=
by sorry

end NUMINAMATH_CALUDE_green_peppers_half_total_l3447_344703


namespace NUMINAMATH_CALUDE_ice_cream_cone_ratio_l3447_344729

def sugar_cones : ℕ := 45
def waffle_cones : ℕ := 36

theorem ice_cream_cone_ratio : 
  ∃ (a b : ℕ), a = 5 ∧ b = 4 ∧ sugar_cones * b = waffle_cones * a :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_cone_ratio_l3447_344729


namespace NUMINAMATH_CALUDE_fruit_basket_cost_l3447_344761

/-- Represents the contents of a fruit basket --/
structure FruitBasket where
  bananas : ℕ
  apples : ℕ
  oranges : ℕ
  kiwis : ℕ
  strawberries : ℕ
  avocados : ℕ
  grapes : ℕ
  melons : ℕ

/-- Represents the prices of individual fruits --/
structure FruitPrices where
  banana : ℚ
  apple : ℚ
  orange : ℚ
  kiwi : ℚ
  strawberry_dozen : ℚ
  avocado : ℚ
  grapes_half_bunch : ℚ
  melon : ℚ

/-- Calculates the total cost of the fruit basket after all discounts --/
def calculateTotalCost (basket : FruitBasket) (prices : FruitPrices) : ℚ :=
  sorry

/-- Theorem stating that the total cost of the given fruit basket is $35.43 --/
theorem fruit_basket_cost :
  let basket : FruitBasket := {
    bananas := 4,
    apples := 3,
    oranges := 4,
    kiwis := 2,
    strawberries := 24,
    avocados := 2,
    grapes := 1,
    melons := 1
  }
  let prices : FruitPrices := {
    banana := 1,
    apple := 2,
    orange := 3/2,
    kiwi := 5/4,
    strawberry_dozen := 4,
    avocado := 3,
    grapes_half_bunch := 2,
    melon := 7/2
  }
  calculateTotalCost basket prices = 3543/100 :=
sorry

end NUMINAMATH_CALUDE_fruit_basket_cost_l3447_344761


namespace NUMINAMATH_CALUDE_existence_and_pigeonhole_l3447_344781

def is_pairwise_coprime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1

theorem existence_and_pigeonhole :
  (∃ (S : Finset ℕ), S.card = 1328 ∧ S.toSet ⊆ Finset.range 1993 ∧
    ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → Nat.gcd (Nat.gcd a b) c > 1) ∧
  (∀ (T : Finset ℕ), T.card = 1329 → T.toSet ⊆ Finset.range 1993 →
    ∃ (a b c : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ is_pairwise_coprime a b c) :=
sorry

end NUMINAMATH_CALUDE_existence_and_pigeonhole_l3447_344781


namespace NUMINAMATH_CALUDE_geometric_progression_x_value_l3447_344736

/-- Given a geometric progression with first three terms 2x - 2, 2x + 2, and 4x + 6, prove that x = -2 -/
theorem geometric_progression_x_value (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*x + 2) = r * (2*x - 2) ∧ (4*x + 6) = r * (2*x + 2)) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_x_value_l3447_344736


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_five_l3447_344733

-- Define the repeating decimal 0.456̄
def repeating_decimal : ℚ := 456 / 999

-- State the theorem
theorem product_of_repeating_decimal_and_five :
  repeating_decimal * 5 = 760 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_five_l3447_344733


namespace NUMINAMATH_CALUDE_pedal_triangle_area_l3447_344790

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point
def Point := ℝ × ℝ

-- Function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Function to check if a triangle is inscribed in a circle
def isInscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Function to calculate the area of the pedal triangle
def pedalTriangleArea (t : Triangle) (p : Point) : ℝ := sorry

-- The main theorem
theorem pedal_triangle_area 
  (c : Circle) (t : Triangle) (p : Point) 
  (h1 : isInscribed t c) 
  (h2 : distance p c.center = d) :
  pedalTriangleArea t p = (1/4) * |1 - (d^2 / c.radius^2)| * triangleArea t := 
by sorry

end NUMINAMATH_CALUDE_pedal_triangle_area_l3447_344790


namespace NUMINAMATH_CALUDE_min_k_theorem_l3447_344751

/-- The set S of powers of 1996 -/
def S : Set ℕ := {n : ℕ | ∃ m : ℕ, n = 1996^m}

/-- Definition of a valid sequence pair -/
def ValidSequencePair (k : ℕ) (a b : ℕ → ℕ) : Prop :=
  (∀ i ∈ Finset.range k, a i ∈ S ∧ b i ∈ S) ∧
  (∀ i ∈ Finset.range k, a i ≠ b i) ∧
  (∀ i ∈ Finset.range (k-1), a i ≤ a (i+1) ∧ b i ≤ b (i+1)) ∧
  (Finset.sum (Finset.range k) a = Finset.sum (Finset.range k) b)

/-- The theorem stating the minimum k -/
theorem min_k_theorem :
  (∃ k : ℕ, ∃ a b : ℕ → ℕ, ValidSequencePair k a b) ∧
  (∀ k < 1997, ¬∃ a b : ℕ → ℕ, ValidSequencePair k a b) ∧
  (∃ a b : ℕ → ℕ, ValidSequencePair 1997 a b) :=
sorry

end NUMINAMATH_CALUDE_min_k_theorem_l3447_344751


namespace NUMINAMATH_CALUDE_flea_landing_product_l3447_344730

/-- The number of circles in the arrangement -/
def num_circles : ℕ := 12

/-- The number of steps the red flea takes clockwise -/
def red_steps : ℕ := 1991

/-- The number of steps the black flea takes counterclockwise -/
def black_steps : ℕ := 1949

/-- The final position of a flea after taking a number of steps -/
def final_position (steps : ℕ) : ℕ :=
  steps % num_circles

/-- The position of the black flea, adjusted for counterclockwise movement -/
def black_position : ℕ :=
  num_circles - (final_position black_steps)

theorem flea_landing_product :
  final_position red_steps * black_position = 77 := by
  sorry

end NUMINAMATH_CALUDE_flea_landing_product_l3447_344730


namespace NUMINAMATH_CALUDE_triangle_area_l3447_344757

theorem triangle_area : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (6, 1)
  let C : ℝ × ℝ := (10, 6)
  let v : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)
  let w : ℝ × ℝ := (B.1 - C.1, B.2 - C.2)
  abs (v.1 * w.2 - v.2 * w.1) / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3447_344757


namespace NUMINAMATH_CALUDE_pages_per_chapter_l3447_344731

theorem pages_per_chapter 
  (total_pages : ℕ) 
  (num_chapters : ℕ) 
  (h1 : total_pages = 555) 
  (h2 : num_chapters = 5) 
  (h3 : total_pages % num_chapters = 0) : 
  total_pages / num_chapters = 111 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_chapter_l3447_344731


namespace NUMINAMATH_CALUDE_power_multiplication_calculate_power_l3447_344710

theorem power_multiplication (a : ℕ) (m n : ℕ) : a * (a ^ n) = a ^ (n + 1) := by sorry

theorem calculate_power : 5000 * (5000 ^ 3000) = 5000 ^ 3001 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_calculate_power_l3447_344710


namespace NUMINAMATH_CALUDE_matrix_self_inverse_l3447_344763

theorem matrix_self_inverse (a b : ℚ) :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -2; a, b]
  A * A = 1 → a = 7.5 ∧ b = -4 := by
sorry

end NUMINAMATH_CALUDE_matrix_self_inverse_l3447_344763


namespace NUMINAMATH_CALUDE_quadratic_roots_close_existence_l3447_344711

theorem quadratic_roots_close_existence :
  ∃ (a b c : ℕ), a ≤ 2019 ∧ b ≤ 2019 ∧ c ≤ 2019 ∧
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (a * x₁^2 + b * x₁ + c = 0) ∧
  (a * x₂^2 + b * x₂ + c = 0) ∧
  |x₁ - x₂| < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_close_existence_l3447_344711


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3447_344766

/-- A line passing through point (2, 1) with equal intercepts on the coordinate axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through point (2, 1) -/
  point_condition : m * 2 + b = 1
  /-- The line has equal intercepts on the coordinate axes -/
  equal_intercepts : b = 0 ∨ m = -1

/-- The equation of a line with equal intercepts passing through (2, 1) is either 2x - y = 0 or x + y - 3 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 1/2 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 3) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3447_344766


namespace NUMINAMATH_CALUDE_prob_at_least_four_same_l3447_344717

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define the probability of exactly five dice showing the same value
def prob_all_same : ℚ := 1 / (num_sides ^ (num_dice - 1))

-- Define the probability of exactly four dice showing the same value
def prob_four_same : ℚ := 
  (num_dice.choose 4) * (1 / (num_sides ^ 3)) * ((num_sides - 1) / num_sides)

-- State the theorem
theorem prob_at_least_four_same : 
  prob_all_same + prob_four_same = 13 / 648 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_four_same_l3447_344717


namespace NUMINAMATH_CALUDE_triangle_side_minimization_l3447_344750

theorem triangle_side_minimization (t C : ℝ) (ht : t > 0) (hC : 0 < C ∧ C < π) :
  let min_c := 2 * Real.sqrt (t * Real.tan (C / 2))
  ∀ a b c : ℝ, (a > 0 ∧ b > 0 ∧ c > 0) →
    (1/2 * a * b * Real.sin C = t) →
    (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
    c ≥ min_c ∧
    (c = min_c ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_minimization_l3447_344750


namespace NUMINAMATH_CALUDE_inequality_range_l3447_344743

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3447_344743


namespace NUMINAMATH_CALUDE_problem_solution_l3447_344752

theorem problem_solution (m n : ℕ) 
  (h1 : m + 8 < n + 3)
  (h2 : (m + (m + 3) + (m + 8) + (n + 3) + (n + 4) + 2*n) / 6 = n)
  (h3 : ((m + 8) + (n + 3)) / 2 = n) : 
  m + n = 53 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3447_344752


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l3447_344744

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2 * y + 3 * z)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l3447_344744


namespace NUMINAMATH_CALUDE_apps_files_difference_l3447_344795

/-- Represents the state of Dave's phone --/
structure PhoneState where
  apps : ℕ
  files : ℕ

/-- The initial state of Dave's phone --/
def initial_state : PhoneState := { apps := 15, files := 24 }

/-- The final state of Dave's phone --/
def final_state : PhoneState := { apps := 21, files := 4 }

/-- Theorem stating the difference between apps and files in the final state --/
theorem apps_files_difference :
  final_state.apps - final_state.files = 17 := by
  sorry

end NUMINAMATH_CALUDE_apps_files_difference_l3447_344795


namespace NUMINAMATH_CALUDE_solution_set_equivalent_to_inequality_l3447_344719

def solution_set : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

def inequality (x : ℝ) : Prop := -x^2 + 3*x - 2 ≥ 0

theorem solution_set_equivalent_to_inequality :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalent_to_inequality_l3447_344719


namespace NUMINAMATH_CALUDE_equal_ratios_sum_l3447_344720

theorem equal_ratios_sum (K L M : ℚ) : 
  (4 : ℚ) / 7 = K / 63 ∧ (4 : ℚ) / 7 = 84 / L ∧ (4 : ℚ) / 7 = M / 98 → 
  K + L + M = 239 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_sum_l3447_344720


namespace NUMINAMATH_CALUDE_total_frogs_caught_l3447_344785

def initial_frogs : ℕ := 5
def additional_frogs : ℕ := 2

theorem total_frogs_caught :
  initial_frogs + additional_frogs = 7 := by sorry

end NUMINAMATH_CALUDE_total_frogs_caught_l3447_344785


namespace NUMINAMATH_CALUDE_correct_assignment_count_l3447_344782

/-- The number of ways to assign doctors and nurses to schools -/
def assignment_methods (doctors nurses schools : ℕ) : ℕ :=
  if doctors = 2 ∧ nurses = 4 ∧ schools = 2 then 12 else 0

/-- Theorem stating that there are 12 different assignment methods -/
theorem correct_assignment_count :
  assignment_methods 2 4 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_correct_assignment_count_l3447_344782


namespace NUMINAMATH_CALUDE_inequality_proof_l3447_344715

theorem inequality_proof (x y z : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < π/2) : 
  π/2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3447_344715


namespace NUMINAMATH_CALUDE_closest_to_fraction_l3447_344775

def options : List ℝ := [0.3, 3, 30, 300, 3000]

theorem closest_to_fraction (x : ℝ) (h : x = 613 / 0.307) :
  ∃ y ∈ options, ∀ z ∈ options, |x - y| ≤ |x - z| :=
sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l3447_344775


namespace NUMINAMATH_CALUDE_journey_solution_l3447_344726

/-- Represents the problem of Xiaogang and Xiaoqiang's journey --/
structure JourneyProblem where
  total_distance : ℝ
  meeting_time : ℝ
  xiaogang_extra_distance : ℝ
  xiaogang_remaining_time : ℝ

/-- Represents the solution to the journey problem --/
structure JourneySolution where
  xiaogang_speed : ℝ
  xiaoqiang_speed : ℝ
  xiaoqiang_remaining_time : ℝ

/-- Theorem stating the correct solution for the given problem --/
theorem journey_solution (p : JourneyProblem) 
  (h1 : p.meeting_time = 2)
  (h2 : p.xiaogang_extra_distance = 24)
  (h3 : p.xiaogang_remaining_time = 0.5) :
  ∃ (s : JourneySolution),
    s.xiaogang_speed = 16 ∧
    s.xiaoqiang_speed = 4 ∧
    s.xiaoqiang_remaining_time = 8 ∧
    p.total_distance = s.xiaogang_speed * (p.meeting_time + p.xiaogang_remaining_time) ∧
    p.total_distance = (s.xiaogang_speed * p.meeting_time - p.xiaogang_extra_distance) + (s.xiaoqiang_speed * s.xiaoqiang_remaining_time) :=
by
  sorry


end NUMINAMATH_CALUDE_journey_solution_l3447_344726


namespace NUMINAMATH_CALUDE_shortest_wire_length_approx_l3447_344784

/-- Represents the configuration of two cylindrical poles wrapped by a wire. -/
structure PolePair where
  small_diameter : ℝ
  large_diameter : ℝ
  height_difference : ℝ

/-- Calculates the length of the shortest wire that can wrap around both poles. -/
def shortest_wire_length (poles : PolePair) : ℝ :=
  sorry

/-- The specific pole configuration from the problem. -/
def problem_poles : PolePair :=
  { small_diameter := 5
  , large_diameter := 20
  , height_difference := 4 }

/-- Theorem stating that the shortest wire length for the given pole configuration
    is approximately 43.089 inches. -/
theorem shortest_wire_length_approx :
  abs (shortest_wire_length problem_poles - 43.089) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_shortest_wire_length_approx_l3447_344784


namespace NUMINAMATH_CALUDE_color_change_probability_l3447_344798

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDurations where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the probability of observing a color change in a traffic light cycle -/
def probabilityOfColorChange (d : TrafficLightDurations) : ℚ :=
  let totalCycleDuration := d.green + d.yellow + d.red
  let changeWindowDuration := 3 * d.yellow
  changeWindowDuration / totalCycleDuration

/-- The main theorem stating the probability of observing a color change -/
theorem color_change_probability (d : TrafficLightDurations) 
  (h1 : d.green = 45)
  (h2 : d.yellow = 5)
  (h3 : d.red = 45) :
  probabilityOfColorChange d = 3 / 19 := by
  sorry

#eval probabilityOfColorChange { green := 45, yellow := 5, red := 45 }

end NUMINAMATH_CALUDE_color_change_probability_l3447_344798


namespace NUMINAMATH_CALUDE_solutions_count_3x_2y_802_l3447_344765

theorem solutions_count_3x_2y_802 : 
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 2 * p.2 = 802 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 803) (Finset.range 402))).card = 133 := by
  sorry

end NUMINAMATH_CALUDE_solutions_count_3x_2y_802_l3447_344765


namespace NUMINAMATH_CALUDE_diagonal_angle_in_rectangular_parallelepiped_l3447_344723

/-- Given a rectangular parallelepiped with two non-intersecting diagonals of adjacent faces
    inclined at angles α and β to the plane of the base, the angle γ between these diagonals
    is equal to arccos(sin α * sin β). -/
theorem diagonal_angle_in_rectangular_parallelepiped
  (α β : Real)
  (h_α : 0 < α ∧ α < π / 2)
  (h_β : 0 < β ∧ β < π / 2) :
  ∃ γ : Real, γ = Real.arccos (Real.sin α * Real.sin β) ∧
    0 ≤ γ ∧ γ ≤ π := by
  sorry

end NUMINAMATH_CALUDE_diagonal_angle_in_rectangular_parallelepiped_l3447_344723


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3447_344742

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4 + a 6 + a 8 + a 10 = 80) →
  (a 1 + a 13 = 40) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3447_344742


namespace NUMINAMATH_CALUDE_meeting_percentage_theorem_l3447_344749

def work_day_hours : ℝ := 10
def first_meeting_minutes : ℝ := 45
def second_meeting_multiplier : ℝ := 3

def total_meeting_time : ℝ := first_meeting_minutes + second_meeting_multiplier * first_meeting_minutes
def work_day_minutes : ℝ := work_day_hours * 60

theorem meeting_percentage_theorem :
  (total_meeting_time / work_day_minutes) * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_meeting_percentage_theorem_l3447_344749


namespace NUMINAMATH_CALUDE_line_through_points_l3447_344778

-- Define the line
def line (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem line_through_points :
  ∀ (a b : ℝ),
  (line a b 3 = 10) →
  (line a b 7 = 22) →
  a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l3447_344778


namespace NUMINAMATH_CALUDE_problem_solution_l3447_344721

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ a b, a * b ≤ 1) ∧
  (∀ a b, 1 / a + 1 / b ≥ 2) ∧
  (∀ m : ℝ, (∀ x : ℝ, |x + m| - |x + 1| ≤ 1 / a + 1 / b) ↔ m ∈ Set.Icc (-1) 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3447_344721


namespace NUMINAMATH_CALUDE_angle_measures_l3447_344700

/-- Given supplementary angles A and B, where A is 6 times B, and B forms a complementary angle C,
    prove the measures of angles A, B, and C. -/
theorem angle_measures (A B C : ℝ) : 
  A + B = 180 →  -- A and B are supplementary
  A = 6 * B →    -- A is 6 times B
  B + C = 90 →   -- B and C are complementary
  A = 180 * 6 / 7 ∧ B = 180 / 7 ∧ C = 90 - 180 / 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_measures_l3447_344700


namespace NUMINAMATH_CALUDE_circle_tangent_existence_l3447_344768

/-- A line in a 2D plane -/
structure Line2D where
  slope : ℝ
  intercept : ℝ

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle2D where
  center : Point2D
  radius : ℝ

/-- Check if a point is on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if a circle is tangent to a line at a point -/
def circleTangentToLineAtPoint (c : Circle2D) (l : Line2D) (p : Point2D) : Prop :=
  pointOnLine p l ∧
  (c.center.x - p.x) * l.slope + (c.center.y - p.y) = 0 ∧
  (c.center.x - p.x)^2 + (c.center.y - p.y)^2 = c.radius^2

theorem circle_tangent_existence
  (l : Line2D) (p : Point2D) (r : ℝ) 
  (h_positive : r > 0) 
  (h_on_line : pointOnLine p l) :
  ∃ (c1 c2 : Circle2D), 
    c1.radius = r ∧
    c2.radius = r ∧
    circleTangentToLineAtPoint c1 l p ∧
    circleTangentToLineAtPoint c2 l p ∧
    c1 ≠ c2 :=
  sorry

end NUMINAMATH_CALUDE_circle_tangent_existence_l3447_344768


namespace NUMINAMATH_CALUDE_no_integer_solution_for_equation_l3447_344714

theorem no_integer_solution_for_equation : ∀ x y : ℤ, x^2 - y^2 ≠ 210 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_equation_l3447_344714


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3447_344708

-- Problem 1
theorem problem_1 : 
  (Real.sqrt 7 - Real.sqrt 13) * (Real.sqrt 7 + Real.sqrt 13) + 
  (Real.sqrt 3 + 1)^2 - (Real.sqrt 6 * Real.sqrt 3) / Real.sqrt 2 + 
  |-(Real.sqrt 3)| = -3 + 3 * Real.sqrt 3 := by sorry

-- Problem 2
theorem problem_2 {a : ℝ} (ha : a < 0) : 
  Real.sqrt (4 - (a + 1/a)^2) - Real.sqrt (4 + (a - 1/a)^2) = -2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3447_344708


namespace NUMINAMATH_CALUDE_coin_toss_sequence_count_l3447_344783

/-- The number of different sequences of 17 coin tosses with specific subsequence counts -/
def coin_toss_sequences : ℕ := sorry

/-- The sequence contains exactly 3 HH subsequences -/
axiom hh_count : coin_toss_sequences = sorry

/-- The sequence contains exactly 4 HT subsequences -/
axiom ht_count : coin_toss_sequences = sorry

/-- The sequence contains exactly 3 TH subsequences -/
axiom th_count : coin_toss_sequences = sorry

/-- The sequence contains exactly 6 TT subsequences -/
axiom tt_count : coin_toss_sequences = sorry

/-- The total number of coin tosses is 17 -/
axiom total_tosses : coin_toss_sequences = sorry

theorem coin_toss_sequence_count : coin_toss_sequences = 4200 := by sorry

end NUMINAMATH_CALUDE_coin_toss_sequence_count_l3447_344783


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l3447_344774

theorem ceiling_floor_sum : ⌈(7:ℚ)/3⌉ + ⌊-(7:ℚ)/3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l3447_344774


namespace NUMINAMATH_CALUDE_meal_combinations_l3447_344718

/-- The number of dishes available in the restaurant -/
def num_dishes : ℕ := 15

/-- The number of ways one person can choose their meal -/
def individual_choices (n : ℕ) : ℕ := n + n * n

/-- The total number of meal combinations for two people -/
def total_combinations (n : ℕ) : ℕ := (individual_choices n) * (individual_choices n)

/-- Theorem stating the total number of meal combinations -/
theorem meal_combinations : total_combinations num_dishes = 57600 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_l3447_344718


namespace NUMINAMATH_CALUDE_straw_hat_value_is_four_l3447_344777

/-- Represents the sheep problem scenario -/
structure SheepProblem where
  x : ℕ  -- number of sheep
  y : ℕ  -- number of times 10 yuan was taken
  z : ℕ  -- last amount taken by younger brother
  h1 : x^2 = x * x  -- price of each sheep equals number of sheep
  h2 : x^2 = 20 * y + 10 + z  -- total money distribution
  h3 : y ≥ 1  -- at least one round of 10 yuan taken
  h4 : z < 10  -- younger brother's last amount less than 10

/-- The value of the straw hat that equalizes the brothers' shares -/
def strawHatValue (p : SheepProblem) : ℕ := 10 - p.z

/-- Theorem stating the value of the straw hat is 4 yuan -/
theorem straw_hat_value_is_four (p : SheepProblem) : strawHatValue p = 4 := by
  sorry

#check straw_hat_value_is_four

end NUMINAMATH_CALUDE_straw_hat_value_is_four_l3447_344777


namespace NUMINAMATH_CALUDE_tan_product_simplification_l3447_344788

theorem tan_product_simplification :
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_simplification_l3447_344788


namespace NUMINAMATH_CALUDE_max_digits_product_5digit_4digit_l3447_344748

theorem max_digits_product_5digit_4digit :
  ∀ (a b : ℕ), 
    10000 ≤ a ∧ a < 100000 →
    1000 ≤ b ∧ b < 10000 →
    a * b < 10000000000 :=
by sorry

end NUMINAMATH_CALUDE_max_digits_product_5digit_4digit_l3447_344748


namespace NUMINAMATH_CALUDE_ln_inequality_and_range_l3447_344712

open Real

theorem ln_inequality_and_range (x : ℝ) (hx : x > 0) :
  (∀ x > 0, Real.log x ≤ x - 1) ∧
  (∀ a : ℝ, (∀ x > 0, Real.log x ≤ a * x + (a - 1) / x - 1) ↔ a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ln_inequality_and_range_l3447_344712


namespace NUMINAMATH_CALUDE_carpeting_cost_specific_room_l3447_344796

/-- Calculates the cost of carpeting a room given its dimensions and carpet specifications. -/
def carpeting_cost (room_length room_breadth carpet_width_cm carpet_cost_paise : ℚ) : ℚ :=
  let room_area : ℚ := room_length * room_breadth
  let carpet_width_m : ℚ := carpet_width_cm / 100
  let carpet_length : ℚ := room_area / carpet_width_m
  let total_cost_paise : ℚ := carpet_length * carpet_cost_paise
  total_cost_paise / 100

/-- Theorem stating that the cost of carpeting a specific room is 36 rupees. -/
theorem carpeting_cost_specific_room :
  carpeting_cost 15 6 75 30 = 36 := by
  sorry

end NUMINAMATH_CALUDE_carpeting_cost_specific_room_l3447_344796


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l3447_344779

theorem polynomial_division_degree (f d q r : Polynomial ℝ) :
  (Polynomial.degree f = 15) →
  (Polynomial.degree q = 7) →
  (r = 5 * X^2 + 3 * X - 8) →
  (f = d * q + r) →
  Polynomial.degree d = 8 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l3447_344779


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l3447_344754

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (garden : GardenDimensions) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (garden.length : ℝ) * step_length * (garden.width : ℝ) * step_length * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden -/
theorem mr_green_potato_yield :
  let garden := GardenDimensions.mk 18 25
  let step_length := 2.5
  let yield_per_sqft := 0.75
  expected_potato_yield garden step_length yield_per_sqft = 2109.375 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_potato_yield_l3447_344754


namespace NUMINAMATH_CALUDE_younger_brother_bricks_l3447_344702

theorem younger_brother_bricks (total_bricks : ℕ) (final_difference : ℕ) : 
  total_bricks = 26 ∧ final_difference = 2 → 
  ∃ (initial_younger : ℕ), 
    initial_younger = 16 ∧
    (total_bricks - initial_younger) + (initial_younger / 2) - 
    ((total_bricks - initial_younger + (initial_younger / 2)) / 2) + 5 = 
    initial_younger - (initial_younger / 2) + 
    ((total_bricks - initial_younger + (initial_younger / 2)) / 2) - 5 + final_difference :=
by
  sorry

#check younger_brother_bricks

end NUMINAMATH_CALUDE_younger_brother_bricks_l3447_344702


namespace NUMINAMATH_CALUDE_log_equality_condition_l3447_344793

theorem log_equality_condition (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq2 : q ≠ 2) :
  Real.log p + Real.log q = Real.log (2 * p + 3 * q) ↔ p = (3 * q) / (q - 2) :=
sorry

end NUMINAMATH_CALUDE_log_equality_condition_l3447_344793


namespace NUMINAMATH_CALUDE_grapes_purchased_l3447_344762

theorem grapes_purchased (grape_price mango_price mango_weight total_paid : ℕ) 
  (h1 : grape_price = 80)
  (h2 : mango_price = 55)
  (h3 : mango_weight = 9)
  (h4 : total_paid = 1135)
  : ∃ (grape_weight : ℕ), grape_weight * grape_price + mango_weight * mango_price = total_paid ∧ grape_weight = 8 := by
  sorry

end NUMINAMATH_CALUDE_grapes_purchased_l3447_344762


namespace NUMINAMATH_CALUDE_square_sum_difference_l3447_344764

theorem square_sum_difference (n : ℕ) : n^2 + (n+1)^2 - (n+2)^2 = n*(n-2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_difference_l3447_344764


namespace NUMINAMATH_CALUDE_ratio_equality_l3447_344739

theorem ratio_equality : (2^3001 * 5^3003) / 10^3002 = 5/2 := by sorry

end NUMINAMATH_CALUDE_ratio_equality_l3447_344739


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3447_344767

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x < 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3447_344767


namespace NUMINAMATH_CALUDE_percent_of_y_l3447_344771

theorem percent_of_y (y : ℝ) (h : y > 0) : ((1 / y) / 20 + (3 / y) / 10) / y = 35 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l3447_344771


namespace NUMINAMATH_CALUDE_fermat_sum_of_two_squares_l3447_344707

theorem fermat_sum_of_two_squares (p : ℕ) (h_prime : Nat.Prime p) (h_mod : p % 4 = 1) :
  ∃ a b : ℤ, p = a^2 + b^2 := by sorry

end NUMINAMATH_CALUDE_fermat_sum_of_two_squares_l3447_344707


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3447_344760

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {2, 3, 4}

-- Define set B
def B : Set Nat := {4, 5}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = A ∩ (U \ B) := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3447_344760


namespace NUMINAMATH_CALUDE_combination_18_choose_4_l3447_344753

theorem combination_18_choose_4 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_combination_18_choose_4_l3447_344753


namespace NUMINAMATH_CALUDE_least_seven_digit_binary_l3447_344747

theorem least_seven_digit_binary : ∀ n : ℕ, n > 0 → (
  (64 ≤ n ∧ (Nat.log 2 n).succ = 7) ↔ 
  (∀ m : ℕ, m > 0 ∧ m < n → (Nat.log 2 m).succ < 7)
) := by sorry

end NUMINAMATH_CALUDE_least_seven_digit_binary_l3447_344747


namespace NUMINAMATH_CALUDE_factorial_10_mod_13_l3447_344716

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_10_mod_13 : factorial 10 % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_10_mod_13_l3447_344716


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_difference_l3447_344780

theorem circle_radius_from_area_circumference_difference 
  (x y : ℝ) (h : x - y = 72 * Real.pi) : ∃ r : ℝ, r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 12 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_difference_l3447_344780


namespace NUMINAMATH_CALUDE_redwood_percentage_increase_l3447_344746

theorem redwood_percentage_increase (num_pines : ℕ) (total_trees : ℕ) : 
  num_pines = 600 → total_trees = 1320 → 
  (total_trees - num_pines : ℚ) / num_pines * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_redwood_percentage_increase_l3447_344746


namespace NUMINAMATH_CALUDE_art_arrangement_probability_l3447_344791

/-- The probability of arranging art pieces with specific conditions -/
theorem art_arrangement_probability (total_pieces : ℕ) (dali_paintings : ℕ) : 
  total_pieces = 12 →
  dali_paintings = 4 →
  (7 : ℚ) / 1485 = (
    (total_pieces - dali_paintings)  -- non-Dali pieces for first position
    * (total_pieces - dali_paintings)  -- positions for Dali group after first piece
    * (Nat.factorial (total_pieces - dali_paintings + 1))  -- arrangements of remaining pieces
  ) / (Nat.factorial total_pieces) := by
  sorry

#check art_arrangement_probability

end NUMINAMATH_CALUDE_art_arrangement_probability_l3447_344791


namespace NUMINAMATH_CALUDE_jimmy_cards_theorem_l3447_344704

def jimmy_cards_problem (initial_cards : ℕ) (cards_to_bob : ℕ) : Prop :=
  let cards_after_bob := initial_cards - cards_to_bob
  let cards_to_mary := 2 * cards_to_bob
  let final_cards := cards_after_bob - cards_to_mary
  initial_cards = 18 ∧ cards_to_bob = 3 → final_cards = 9

theorem jimmy_cards_theorem : jimmy_cards_problem 18 3 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_cards_theorem_l3447_344704


namespace NUMINAMATH_CALUDE_system_solution_l3447_344769

-- Define the system of equations
def system (x₁ x₂ x₃ x₄ x₅ y : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₅

-- Define the solution
def solution (x₁ x₂ x₃ x₄ x₅ y : ℝ) : Prop :=
  (y = 2 ∧ x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅) ∨
  (y ≠ 2 ∧
    ((y^2 + y - 1 ≠ 0 ∧ x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∨
     (y^2 + y - 1 = 0 ∧ (y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) ∧
      x₃ = y * x₂ - x₁ ∧
      x₄ = -y * x₂ - y * x₁ ∧
      x₅ = y * x₁ - x₂)))

-- Theorem statement
theorem system_solution (x₁ x₂ x₃ x₄ x₅ y : ℝ) :
  system x₁ x₂ x₃ x₄ x₅ y ↔ solution x₁ x₂ x₃ x₄ x₅ y := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3447_344769


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3447_344756

theorem decimal_to_fraction :
  (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3447_344756


namespace NUMINAMATH_CALUDE_library_books_remaining_l3447_344789

theorem library_books_remaining (initial_books : ℕ) (given_away : ℕ) (donated : ℕ) : 
  initial_books = 125 → given_away = 42 → donated = 31 → 
  initial_books - given_away - donated = 52 := by
sorry

end NUMINAMATH_CALUDE_library_books_remaining_l3447_344789


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3447_344770

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 + 1 = (x^2 - 4*x + 7) * q + (8*x - 62) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3447_344770


namespace NUMINAMATH_CALUDE_right_triangle_area_floor_l3447_344713

theorem right_triangle_area_floor (perimeter : ℝ) (inscribed_circle_area : ℝ) : 
  perimeter = 2008 →
  inscribed_circle_area = 100 * Real.pi ^ 3 →
  ⌊(perimeter / 2) * (inscribed_circle_area / Real.pi) ^ (1/2)⌋ = 31541 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_floor_l3447_344713


namespace NUMINAMATH_CALUDE_polynomial_value_at_zero_exists_satisfying_polynomial_l3447_344735

/-- A polynomial satisfying the given condition -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (|y^2 - P x| ≤ 2 * |x|) ↔ (|x^2 - P y| ≤ 2 * |y|)

/-- The set of all possible values of P(0) for polynomials satisfying the condition -/
def PossibleValues : Set ℝ :=
  {x | x < 0 ∨ x = 1}

/-- The main theorem stating that for any polynomial satisfying the condition,
    P(0) must be in the set of possible values -/
theorem polynomial_value_at_zero (P : ℝ → ℝ) (h : SatisfyingPolynomial P) :
  P 0 ∈ PossibleValues := by
  sorry

/-- The converse theorem stating that for any value in the set of possible values,
    there exists a polynomial satisfying the condition with P(0) equal to that value -/
theorem exists_satisfying_polynomial (x : ℝ) (h : x ∈ PossibleValues) :
  ∃ P : ℝ → ℝ, SatisfyingPolynomial P ∧ P 0 = x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_zero_exists_satisfying_polynomial_l3447_344735


namespace NUMINAMATH_CALUDE_parabola_directrix_l3447_344787

/-- The equation of the directrix of a parabola with equation x² = 4y and focus at (0, 1) -/
theorem parabola_directrix : ∃ (l : ℝ → ℝ), 
  (∀ x y : ℝ, x^2 = 4*y → (∀ t : ℝ, (x - 0)^2 + (y - 1)^2 = (x - t)^2 + (y - l t)^2)) → 
  (∀ t : ℝ, l t = -1) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3447_344787


namespace NUMINAMATH_CALUDE_derivative_reciprocal_sum_sqrt_derivative_reciprocal_sum_sqrt_value_l3447_344786

theorem derivative_reciprocal_sum_sqrt (x : ℝ) (h : x ≠ 1) :
  (fun x => 2 / (1 - x)) = (fun x => 1 / (1 - Real.sqrt x) + 1 / (1 + Real.sqrt x)) :=
by sorry

theorem derivative_reciprocal_sum_sqrt_value (x : ℝ) (h : x ≠ 1) :
  deriv (fun x => 2 / (1 - x)) x = 2 / (1 - x)^2 :=
by sorry

end NUMINAMATH_CALUDE_derivative_reciprocal_sum_sqrt_derivative_reciprocal_sum_sqrt_value_l3447_344786


namespace NUMINAMATH_CALUDE_smallest_root_of_unity_for_equation_l3447_344740

theorem smallest_root_of_unity_for_equation : ∃ (n : ℕ),
  (n > 0) ∧ 
  (∀ z : ℂ, z^6 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^6 - z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_root_of_unity_for_equation_l3447_344740


namespace NUMINAMATH_CALUDE_monotonically_decreasing_condition_l3447_344737

def f (x : ℝ) := x^2 - 2*x + 3

theorem monotonically_decreasing_condition (m : ℝ) :
  (∀ x y, x < y ∧ y < m → f x > f y) ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_condition_l3447_344737


namespace NUMINAMATH_CALUDE_arithmetic_sum_equals_168_l3447_344773

/-- The sum of an arithmetic sequence with first term 3, common difference 2, and 12 terms -/
def arithmetic_sum : ℕ := 
  let a₁ : ℕ := 3  -- first term
  let d : ℕ := 2   -- common difference
  let n : ℕ := 12  -- number of terms
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem stating that the sum of the arithmetic sequence is 168 -/
theorem arithmetic_sum_equals_168 : arithmetic_sum = 168 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_equals_168_l3447_344773


namespace NUMINAMATH_CALUDE_expression_value_l3447_344759

theorem expression_value : (3 * 12 + 18) / (6 - 3) = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3447_344759


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l3447_344706

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f y < f x

theorem decreasing_function_inequality
  (f : ℝ → ℝ) (h : decreasing_function f) :
  ∀ x : ℝ, f (2 * x - 1) < f 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l3447_344706


namespace NUMINAMATH_CALUDE_seashells_count_l3447_344799

theorem seashells_count (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) 
  (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
  sorry

end NUMINAMATH_CALUDE_seashells_count_l3447_344799


namespace NUMINAMATH_CALUDE_min_probability_theorem_l3447_344728

def closest_integer (m : ℤ) (k : ℤ) : ℤ := 
  sorry

def P (k : ℤ) : ℚ :=
  sorry

theorem min_probability_theorem :
  ∀ k : ℤ, k % 2 = 1 → 1 ≤ k → k ≤ 99 →
    P k ≥ 34/67 ∧ 
    ∃ k₀ : ℤ, k₀ % 2 = 1 ∧ 1 ≤ k₀ ∧ k₀ ≤ 99 ∧ P k₀ = 34/67 :=
  sorry

end NUMINAMATH_CALUDE_min_probability_theorem_l3447_344728


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3447_344738

theorem inequality_system_solution_set :
  let S := {x : ℝ | x - 3 < 2 ∧ 3 * x + 1 ≥ 2 * x}
  S = {x : ℝ | -1 ≤ x ∧ x < 5} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3447_344738


namespace NUMINAMATH_CALUDE_distance_to_place_distance_calculation_l3447_344741

/-- Calculates the distance to a place given rowing speeds and time -/
theorem distance_to_place (still_water_speed : ℝ) (current_velocity : ℝ) (total_time : ℝ) : ℝ :=
  let downstream_speed := still_water_speed + current_velocity
  let upstream_speed := still_water_speed - current_velocity
  let downstream_time := (total_time * upstream_speed) / (downstream_speed + upstream_speed)
  let distance := downstream_time * downstream_speed
  distance

/-- The distance to the place is approximately 10.83 km -/
theorem distance_calculation : 
  abs (distance_to_place 8 2.5 3 - 10.83) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_place_distance_calculation_l3447_344741
