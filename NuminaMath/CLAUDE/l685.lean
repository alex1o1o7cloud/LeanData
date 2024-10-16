import Mathlib

namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_positive_l685_68558

/-- The function f(x) = a(x^3 - x) is monotonically decreasing on the interval (-√3/3, √3/3) -/
def is_monotone_decreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, -Real.sqrt 3 / 3 < x ∧ x < y ∧ y < Real.sqrt 3 / 3 → f x > f y

/-- The definition of the function f(x) = a(x^3 - x) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x^3 - x)

theorem monotone_decreasing_implies_a_positive (a : ℝ) :
  is_monotone_decreasing (f a) a → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_positive_l685_68558


namespace NUMINAMATH_CALUDE_min_product_sum_l685_68557

theorem min_product_sum (a b c d : ℕ) : 
  a ∈ ({1, 3, 5, 7} : Set ℕ) → 
  b ∈ ({1, 3, 5, 7} : Set ℕ) → 
  c ∈ ({1, 3, 5, 7} : Set ℕ) → 
  d ∈ ({1, 3, 5, 7} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  48 ≤ a * b + b * c + c * d + d * a :=
by sorry

end NUMINAMATH_CALUDE_min_product_sum_l685_68557


namespace NUMINAMATH_CALUDE_split_cube_345_l685_68512

/-- The first number in the splitting of n³ -/
def first_split (n : ℕ) : ℕ := n * (n - 1) + 1

/-- A number is a splitting number of n³ if it's in the range of the first split to n³ -/
def is_splitting_number (k n : ℕ) : Prop :=
  first_split n ≤ k ∧ k ≤ n^3

theorem split_cube_345 (m : ℕ) (h1 : m > 1) (h2 : is_splitting_number 345 m) : m = 19 := by
  sorry

end NUMINAMATH_CALUDE_split_cube_345_l685_68512


namespace NUMINAMATH_CALUDE_sequence_ratio_proof_l685_68510

theorem sequence_ratio_proof (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a n > 0) →
  (∀ n, (a (n + 1))^2 = (a n) * (a (n + 2))) →
  (S 3 = 13) →
  (a 1 = 1) →
  ((a 3 + a 4) / (a 1 + a 2) = 9) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_proof_l685_68510


namespace NUMINAMATH_CALUDE_complement_of_union_equals_zero_five_l685_68590

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 4}

theorem complement_of_union_equals_zero_five :
  (U \ (A ∪ B)) = {0, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_zero_five_l685_68590


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_pi_12_l685_68592

theorem sin_2alpha_plus_pi_12 (α : ℝ) 
  (h1 : -π/6 < α ∧ α < π/6) 
  (h2 : Real.cos (α + π/6) = 4/5) : 
  Real.sin (2*α + π/12) = 17 * Real.sqrt 2 / 50 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_pi_12_l685_68592


namespace NUMINAMATH_CALUDE_circle_condition_l685_68570

/-- A quadratic equation in two variables represents a circle if and only if
    D^2 + E^2 - 4F > 0, where the equation is in the form x^2 + y^2 + Dx + Ey + F = 0 -/
def is_circle (D E F : ℝ) : Prop := D^2 + E^2 - 4*F > 0

/-- The equation x^2 + y^2 + x + y + k = 0 represents a circle -/
def represents_circle (k : ℝ) : Prop := is_circle 1 1 k

/-- If x^2 + y^2 + x + y + k = 0 represents a circle, then k < 1/2 -/
theorem circle_condition (k : ℝ) : represents_circle k → k < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l685_68570


namespace NUMINAMATH_CALUDE_logical_equivalences_l685_68537

theorem logical_equivalences :
  (∀ A B C : Prop,
    ((A ∨ B) → C) ↔ ((A → C) ∧ (B → C))) ∧
  (∀ A B C : Prop,
    (A → (B ∧ C)) ↔ ((A → B) ∧ (A → C))) := by
  sorry

end NUMINAMATH_CALUDE_logical_equivalences_l685_68537


namespace NUMINAMATH_CALUDE_hcf_of_12_and_15_l685_68578

theorem hcf_of_12_and_15 : 
  ∀ (hcf lcm : ℕ), 
    lcm = 60 → 
    12 * 15 = lcm * hcf → 
    hcf = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_hcf_of_12_and_15_l685_68578


namespace NUMINAMATH_CALUDE_product_of_primes_l685_68530

def largest_odd_one_digit_prime : ℕ := 7

def largest_two_digit_prime : ℕ := 97

def second_largest_two_digit_prime : ℕ := 89

theorem product_of_primes : 
  largest_odd_one_digit_prime * largest_two_digit_prime * second_largest_two_digit_prime = 60431 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_l685_68530


namespace NUMINAMATH_CALUDE_highest_power_of_three_l685_68571

def A (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem highest_power_of_three (n : ℕ) :
  (∃ k : ℕ, A n = 3^n * k ∧ k % 3 ≠ 0) ∧ A n % 3^(n+1) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_three_l685_68571


namespace NUMINAMATH_CALUDE_pet_store_multiple_is_three_profit_matches_given_l685_68562

/-- The multiple of Brandon's selling price that the pet store uses -/
def pet_store_multiple (brandon_price : ℕ) (pet_store_profit : ℕ) : ℕ := 
  (pet_store_profit + brandon_price - 5) / brandon_price

theorem pet_store_multiple_is_three : 
  let brandon_price := 100
  let pet_store_profit := 205
  pet_store_multiple brandon_price pet_store_profit = 3 := by
sorry

/-- The pet store's selling price -/
def pet_store_price (brandon_price : ℕ) (multiple : ℕ) : ℕ :=
  multiple * brandon_price + 5

/-- The pet store's profit -/
def calculate_profit (brandon_price : ℕ) (pet_store_price : ℕ) : ℕ :=
  pet_store_price - brandon_price

theorem profit_matches_given :
  let brandon_price := 100
  let multiple := 3
  let store_price := pet_store_price brandon_price multiple
  calculate_profit brandon_price store_price = 205 := by
sorry

end NUMINAMATH_CALUDE_pet_store_multiple_is_three_profit_matches_given_l685_68562


namespace NUMINAMATH_CALUDE_remainder_2024_3047_mod_800_l685_68534

theorem remainder_2024_3047_mod_800 : (2024 * 3047) % 800 = 728 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2024_3047_mod_800_l685_68534


namespace NUMINAMATH_CALUDE_return_probability_is_one_sixth_l685_68596

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron :=
  (vertices : Finset (Fin 4))
  (edges : Finset (Fin 4 × Fin 4))
  (adjacent : Fin 4 → Finset (Fin 4))
  (adjacent_sym : ∀ v₁ v₂, v₂ ∈ adjacent v₁ ↔ v₁ ∈ adjacent v₂)
  (adjacent_card : ∀ v, (adjacent v).card = 3)

/-- The probability of returning to the starting vertex in two moves -/
def return_probability (t : RegularTetrahedron) : ℚ :=
  1 / 6

/-- Theorem: The probability of returning to the starting vertex in two moves is 1/6 -/
theorem return_probability_is_one_sixth (t : RegularTetrahedron) :
  return_probability t = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_return_probability_is_one_sixth_l685_68596


namespace NUMINAMATH_CALUDE_tim_golf_balls_l685_68515

/-- The number of items in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of golf balls Tim has -/
def golf_ball_dozens : ℕ := 13

/-- The total number of golf balls Tim has -/
def total_golf_balls : ℕ := golf_ball_dozens * dozen

theorem tim_golf_balls : total_golf_balls = 156 := by
  sorry

end NUMINAMATH_CALUDE_tim_golf_balls_l685_68515


namespace NUMINAMATH_CALUDE_gear_replacement_proof_l685_68561

/-- Represents a pair of gears --/
structure GearPair where
  gear1 : ℕ
  gear2 : ℕ

/-- The initial gear pair --/
def initial_gears : GearPair := ⟨7, 9⟩

/-- The modified gear pair --/
def modified_gears : GearPair := ⟨4, 12⟩

/-- Checks if a gear pair has the ratio 7:9 or 9:7 --/
def has_initial_ratio (gp : GearPair) : Prop :=
  (gp.gear1 * 9 = gp.gear2 * 7) ∨ (gp.gear1 * 7 = gp.gear2 * 9)

/-- Checks if a gear pair has the ratio 1:3 or 3:1 --/
def has_final_ratio (gp : GearPair) : Prop :=
  (gp.gear1 * 3 = gp.gear2 * 1) ∨ (gp.gear1 * 1 = gp.gear2 * 3)

/-- Checks if two gear pairs differ by exactly 3 teeth for each gear --/
def differs_by_three (gp1 gp2 : GearPair) : Prop :=
  (gp1.gear1 + 3 = gp2.gear1 ∧ gp1.gear2 - 3 = gp2.gear2) ∨
  (gp1.gear1 - 3 = gp2.gear1 ∧ gp1.gear2 + 3 = gp2.gear2)

theorem gear_replacement_proof :
  has_initial_ratio initial_gears ∧
  has_final_ratio modified_gears ∧
  differs_by_three initial_gears modified_gears ∧
  (∀ gp1 gp2 : GearPair, 
    has_initial_ratio gp1 ∧ 
    has_final_ratio gp2 ∧ 
    differs_by_three gp1 gp2 → 
    gp1 = initial_gears ∧ gp2 = modified_gears) := by
  sorry

#check gear_replacement_proof

end NUMINAMATH_CALUDE_gear_replacement_proof_l685_68561


namespace NUMINAMATH_CALUDE_sum_of_factors_36_l685_68500

/-- The sum of positive factors of 36 is 91. -/
theorem sum_of_factors_36 : (Finset.filter (· ∣ 36) (Finset.range 37)).sum id = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_36_l685_68500


namespace NUMINAMATH_CALUDE_infinitely_many_square_repetitions_l685_68531

/-- The number of digits in a natural number -/
def num_digits (a : ℕ) : ℕ := sorry

/-- The repetition of a natural number -/
def repetition (a : ℕ) : ℕ := a * (10^(num_digits a)) + a

/-- There exist infinitely many natural numbers whose repetition is a perfect square -/
theorem infinitely_many_square_repetitions :
  ∀ n : ℕ, ∃ a > n, ∃ k : ℕ, repetition a = k^2 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_square_repetitions_l685_68531


namespace NUMINAMATH_CALUDE_simplify_expression_l685_68599

theorem simplify_expression (x : ℝ) (hx : x ≥ 0) : (3 * x^(1/2))^6 = 729 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l685_68599


namespace NUMINAMATH_CALUDE_regression_not_exact_l685_68527

-- Define the linear regression model
def linear_regression (x : ℝ) : ℝ := 0.5 * x - 85

-- Define the specific x value we're interested in
def x_value : ℝ := 200

-- Theorem stating that y is not necessarily exactly 15 when x = 200
theorem regression_not_exact : 
  ∃ (ε : ℝ), ε ≠ 0 ∧ linear_regression x_value + ε = 15 := by
  sorry

end NUMINAMATH_CALUDE_regression_not_exact_l685_68527


namespace NUMINAMATH_CALUDE_sum_of_products_l685_68508

theorem sum_of_products (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x^2 + x*y + y^2 = 75)
  (h2 : y^2 + y*z + z^2 = 16)
  (h3 : z^2 + x*z + x^2 = 91) :
  x*y + y*z + x*z = 40 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l685_68508


namespace NUMINAMATH_CALUDE_parabola_x_intercept_difference_l685_68598

/-- Represents a quadratic function of the form f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-value for a given x-value in a quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a quadratic function -/
def QuadraticFunction.contains_point (f : QuadraticFunction) (p : Point) : Prop :=
  f.eval p.x = p.y

/-- Calculates the difference between the x-intercepts of a quadratic function -/
noncomputable def x_intercept_difference (f : QuadraticFunction) : ℝ :=
  sorry

theorem parabola_x_intercept_difference :
  ∀ (f : QuadraticFunction),
  (∃ (v : Point), v.x = 3 ∧ v.y = -9 ∧ f.contains_point v) →
  f.contains_point ⟨5, 7⟩ →
  x_intercept_difference f = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_x_intercept_difference_l685_68598


namespace NUMINAMATH_CALUDE_one_match_probability_l685_68518

/-- The number of balls and boxes -/
def n : ℕ := 4

/-- The total number of ways to distribute balls into boxes -/
def total_arrangements : ℕ := n.factorial

/-- The number of ways to distribute balls with exactly one color match -/
def matching_arrangements : ℕ := n * ((n - 1).factorial)

/-- The probability of exactly one ball matching its box color -/
def probability_one_match : ℚ := matching_arrangements / total_arrangements

theorem one_match_probability :
  probability_one_match = 1/3 :=
sorry

end NUMINAMATH_CALUDE_one_match_probability_l685_68518


namespace NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l685_68585

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the center and right focus
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (1, 0)

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- State the theorem
theorem min_dot_product_on_ellipse :
  ∃ (min : ℝ), min = 1/2 ∧
  ∀ (P : ℝ × ℝ), is_on_ellipse P.1 P.2 →
    dot_product (P.1 - O.1, P.2 - O.2) (P.1 - F.1, P.2 - F.2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l685_68585


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l685_68522

-- Define the polynomial g(x)
def g (p q r s : ℝ) (x : ℂ) : ℂ := x^4 + p*x^3 + q*x^2 + r*x + s

-- State the theorem
theorem polynomial_sum_of_coefficients 
  (p q r s : ℝ) -- Real coefficients
  (h1 : g p q r s (3*Complex.I) = 0) -- g(3i) = 0
  (h2 : g p q r s (1 + 2*Complex.I) = 0) -- g(1+2i) = 0
  : p + q + r + s = -41 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l685_68522


namespace NUMINAMATH_CALUDE_base_conversion_1729_l685_68594

theorem base_conversion_1729 :
  (5 * 7^3 + 0 * 7^2 + 2 * 7^1 + 0 * 7^0 : ℕ) = 1729 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1729_l685_68594


namespace NUMINAMATH_CALUDE_diagonals_15_sided_polygon_l685_68587

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex polygon with 15 sides is 90 -/
theorem diagonals_15_sided_polygon : num_diagonals 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_15_sided_polygon_l685_68587


namespace NUMINAMATH_CALUDE_set_A_properties_l685_68529

def A : Set ℝ := {x | x^2 - 4 = 0}

theorem set_A_properties :
  (2 ∈ A) ∧
  (-2 ∈ A) ∧
  (A = {-2, 2}) ∧
  (∅ ⊆ A) := by
sorry

end NUMINAMATH_CALUDE_set_A_properties_l685_68529


namespace NUMINAMATH_CALUDE_partnership_profit_l685_68552

/-- Given the investments of three partners and the profit share of one partner, 
    calculate the total profit of the partnership. -/
theorem partnership_profit 
  (investment_A investment_B investment_C : ℕ) 
  (profit_share_A : ℕ) 
  (h1 : investment_A = 2400)
  (h2 : investment_B = 7200)
  (h3 : investment_C = 9600)
  (h4 : profit_share_A = 1125) :
  (investment_A + investment_B + investment_C) * profit_share_A / investment_A = 9000 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l685_68552


namespace NUMINAMATH_CALUDE_f_min_value_f_min_at_9_2_l685_68575

/-- The function f(x, y) defined in the problem -/
def f (x y : ℝ) : ℝ := x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

/-- Theorem stating that f(x, y) has a minimum value of 3 -/
theorem f_min_value (x y : ℝ) : f x y ≥ 3 := by sorry

/-- Theorem stating that f(9, 2) achieves the minimum value -/
theorem f_min_at_9_2 : f 9 2 = 3 := by sorry

end NUMINAMATH_CALUDE_f_min_value_f_min_at_9_2_l685_68575


namespace NUMINAMATH_CALUDE_hexagon_triangles_count_l685_68565

/-- The number of unit equilateral triangles needed to form a regular hexagon of side length n -/
def triangles_in_hexagon (n : ℕ) : ℕ := 6 * (n * (n + 1) / 2)

/-- Given that 6 unit equilateral triangles can form a regular hexagon with side length 1,
    prove that 126 unit equilateral triangles are needed to form a regular hexagon with side length 6 -/
theorem hexagon_triangles_count :
  triangles_in_hexagon 1 = 6 →
  triangles_in_hexagon 6 = 126 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_triangles_count_l685_68565


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l685_68547

theorem inequality_and_equality_condition (a b c : ℝ) (α : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c * (a^α + b^α + c^α) ≥ 
    a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ∧
  (a * b * c * (a^α + b^α + c^α) = 
    a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ↔
   a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l685_68547


namespace NUMINAMATH_CALUDE_paint_cans_calculation_l685_68521

theorem paint_cans_calculation (initial_cans : ℚ) : 
  (initial_cans / 2 - (initial_cans / 6 + 5) = 5) → initial_cans = 30 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_calculation_l685_68521


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l685_68597

-- Define a function to normalize an angle to the range [0, 360)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

-- Theorem statement
theorem same_terminal_side_angle :
  normalizeAngle (-390) = 330 :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l685_68597


namespace NUMINAMATH_CALUDE_carrie_turnip_mixture_l685_68589

-- Define the ratio of potatoes to turnips
def potatoTurnipRatio : ℚ := 5 / 2

-- Define the total amount of potatoes
def totalPotatoes : ℚ := 20

-- Define the amount of turnips that can be added
def turnipsToAdd : ℚ := totalPotatoes / potatoTurnipRatio

-- Theorem statement
theorem carrie_turnip_mixture :
  turnipsToAdd = 8 := by sorry

end NUMINAMATH_CALUDE_carrie_turnip_mixture_l685_68589


namespace NUMINAMATH_CALUDE_product_of_primes_in_equation_l685_68581

theorem product_of_primes_in_equation (p q : ℕ) : 
  Prime p → Prime q → p * 1 + q = 99 → p * q = 194 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_in_equation_l685_68581


namespace NUMINAMATH_CALUDE_associate_professor_pencils_l685_68564

theorem associate_professor_pencils :
  ∀ (A B P : ℕ),
    A + B = 7 →
    A + 2 * B = 11 →
    P * A + B = 10 →
    P = 2 := by
  sorry

end NUMINAMATH_CALUDE_associate_professor_pencils_l685_68564


namespace NUMINAMATH_CALUDE_segment_length_l685_68511

theorem segment_length : 
  let endpoints := {x : ℝ | |x - (27 : ℝ)^(1/3)| = 5}
  ∃ a b : ℝ, a ∈ endpoints ∧ b ∈ endpoints ∧ |a - b| = 10 :=
by sorry

end NUMINAMATH_CALUDE_segment_length_l685_68511


namespace NUMINAMATH_CALUDE_expression_evaluation_l685_68549

theorem expression_evaluation :
  let x : ℤ := -1
  (x - 1)^2 - x * (x + 3) + 2 * (x + 2) * (x - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l685_68549


namespace NUMINAMATH_CALUDE_union_determines_x_l685_68544

theorem union_determines_x (A B : Set ℕ) (x : ℕ) :
  A = {1, 2, x} →
  B = {2, 4, 5} →
  A ∪ B = {1, 2, 3, 4, 5} →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_union_determines_x_l685_68544


namespace NUMINAMATH_CALUDE_square_area_difference_l685_68514

-- Define the sides of the squares
def a : ℕ := 12
def b : ℕ := 9
def c : ℕ := 7
def d : ℕ := 3

-- Define the theorem
theorem square_area_difference : a ^ 2 + c ^ 2 - b ^ 2 - d ^ 2 = 103 := by
  sorry

end NUMINAMATH_CALUDE_square_area_difference_l685_68514


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l685_68555

theorem added_number_after_doubling (initial_number : ℕ) (added_number : ℕ) : 
  initial_number = 9 →
  3 * (2 * initial_number + added_number) = 93 →
  added_number = 13 := by
sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l685_68555


namespace NUMINAMATH_CALUDE_no_snow_probability_l685_68504

theorem no_snow_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 1/2) (h2 : p2 = 2/3) (h3 : p3 = 3/4) (h4 : p4 = 4/5) : 
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1/120 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l685_68504


namespace NUMINAMATH_CALUDE_g_max_min_l685_68536

noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 8 + 8 * Real.cos x ^ 8

theorem g_max_min :
  (∀ x, g x ≤ 8) ∧ (∃ x, g x = 8) ∧ (∀ x, 8/27 ≤ g x) ∧ (∃ x, g x = 8/27) :=
sorry

end NUMINAMATH_CALUDE_g_max_min_l685_68536


namespace NUMINAMATH_CALUDE_total_insects_eaten_l685_68525

theorem total_insects_eaten (num_geckos : ℕ) (insects_per_gecko : ℕ) (num_lizards : ℕ) : 
  num_geckos = 5 → insects_per_gecko = 6 → num_lizards = 3 → 
  num_geckos * insects_per_gecko + num_lizards * (2 * insects_per_gecko) = 66 := by
  sorry

#check total_insects_eaten

end NUMINAMATH_CALUDE_total_insects_eaten_l685_68525


namespace NUMINAMATH_CALUDE_base_8_5214_equals_2700_l685_68526

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ (digits.length - 1 - i))) 0

theorem base_8_5214_equals_2700 :
  base_8_to_10 [5, 2, 1, 4] = 2700 := by
  sorry

end NUMINAMATH_CALUDE_base_8_5214_equals_2700_l685_68526


namespace NUMINAMATH_CALUDE_line_point_theorem_l685_68563

/-- The line equation y = -2/3x + 10 -/
def line_equation (x y : ℝ) : Prop := y = -2/3 * x + 10

/-- Point P is where the line crosses the x-axis -/
def point_P : ℝ × ℝ := (15, 0)

/-- Point Q is where the line crosses the y-axis -/
def point_Q : ℝ × ℝ := (0, 10)

/-- Point T is on the line segment PQ -/
def point_T_on_PQ (r s : ℝ) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
  r = t * point_P.1 + (1 - t) * point_Q.1 ∧
  s = t * point_P.2 + (1 - t) * point_Q.2

/-- The area of triangle POQ is four times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs (point_P.1 * point_Q.2 - point_Q.1 * point_P.2) / 2 = 
  4 * abs (r * point_P.2 - point_P.1 * s) / 2

/-- Main theorem -/
theorem line_point_theorem (r s : ℝ) :
  line_equation r s →
  point_T_on_PQ r s →
  area_condition r s →
  r + s = 13.75 := by sorry

end NUMINAMATH_CALUDE_line_point_theorem_l685_68563


namespace NUMINAMATH_CALUDE_school_population_l685_68567

theorem school_population (t : ℕ) : 
  let g := 4 * t          -- number of girls
  let b := 6 * g          -- number of boys
  let s := t / 2          -- number of staff members
  b + g + t + s = 59 * t / 2 := by
sorry

end NUMINAMATH_CALUDE_school_population_l685_68567


namespace NUMINAMATH_CALUDE_conference_hall_seating_l685_68550

theorem conference_hall_seating
  (chairs_per_row : ℕ)
  (initial_chairs : ℕ)
  (expected_participants : ℕ)
  (h1 : chairs_per_row = 15)
  (h2 : initial_chairs = 195)
  (h3 : expected_participants = 120)
  : ∃ (removed_chairs : ℕ),
    removed_chairs = 75 ∧
    (initial_chairs - removed_chairs) % chairs_per_row = 0 ∧
    initial_chairs - removed_chairs ≥ expected_participants ∧
    initial_chairs - removed_chairs < expected_participants + chairs_per_row :=
by
  sorry

end NUMINAMATH_CALUDE_conference_hall_seating_l685_68550


namespace NUMINAMATH_CALUDE_secretary_work_time_l685_68543

theorem secretary_work_time (t1 t2 t3 : ℕ) : 
  t1 + t2 + t3 = 110 ∧ 
  t3 = 55 ∧ 
  2 * t2 = 3 * t1 ∧ 
  5 * t1 = 3 * t3 :=
by sorry

end NUMINAMATH_CALUDE_secretary_work_time_l685_68543


namespace NUMINAMATH_CALUDE_five_digit_number_divisible_by_9_l685_68546

theorem five_digit_number_divisible_by_9 (a b c d e : ℕ) : 
  (10000 * a + 1000 * b + 100 * c + 10 * d + e) % 9 = 0 →
  (100 * a + 10 * c + e) - (100 * b + 10 * d + a) = 760 →
  10000 ≤ (10000 * a + 1000 * b + 100 * c + 10 * d + e) →
  (10000 * a + 1000 * b + 100 * c + 10 * d + e) < 100000 →
  10000 * a + 1000 * b + 100 * c + 10 * d + e = 81828 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_number_divisible_by_9_l685_68546


namespace NUMINAMATH_CALUDE_sqrt_square_abs_l685_68574

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_l685_68574


namespace NUMINAMATH_CALUDE_value_calculation_l685_68584

theorem value_calculation (initial_number : ℕ) (h : initial_number = 26) : 
  ((((initial_number + 20) * 2) / 2) - 2) * 2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l685_68584


namespace NUMINAMATH_CALUDE_min_sum_distances_l685_68588

-- Define a rectangle in 2D space
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_rectangle : sorry -- Condition ensuring ABCD forms a rectangle

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the center of a rectangle
def center (r : Rectangle) : ℝ × ℝ := sorry

-- Define the sum of distances from a point to the corners
def sum_distances (r : Rectangle) (p : ℝ × ℝ) : ℝ :=
  distance p r.A + distance p r.B + distance p r.C + distance p r.D

-- Theorem statement
theorem min_sum_distances (r : Rectangle) :
  ∀ p : ℝ × ℝ, sum_distances r (center r) ≤ sum_distances r p :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_l685_68588


namespace NUMINAMATH_CALUDE_julia_played_with_12_on_monday_l685_68586

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 19 - 7

/-- The total number of kids Julia played with -/
def total_kids : ℕ := 19

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 7

theorem julia_played_with_12_on_monday :
  monday_kids = 12 :=
sorry

end NUMINAMATH_CALUDE_julia_played_with_12_on_monday_l685_68586


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l685_68533

/-- Given an arithmetic sequence {a_n}, if a₂ + 4a₇ + a₁₂ = 96, then 2a₃ + a₁₅ = 48 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 + 4 * a 7 + a 12 = 96 →                       -- given condition
  2 * a 3 + a 15 = 48 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l685_68533


namespace NUMINAMATH_CALUDE_initial_amount_proof_l685_68553

/-- Proves that Rs 100 at 5% interest for 48 years produces the same interest as Rs 600 at 10% interest for 4 years -/
theorem initial_amount_proof (amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time1 : ℝ) (time2 : ℝ) : 
  amount = 100 ∧ rate1 = 0.05 ∧ rate2 = 0.10 ∧ time1 = 48 ∧ time2 = 4 →
  amount * rate1 * time1 = 600 * rate2 * time2 :=
by
  sorry

#check initial_amount_proof

end NUMINAMATH_CALUDE_initial_amount_proof_l685_68553


namespace NUMINAMATH_CALUDE_target_probability_l685_68579

/-- The probability of hitting a target once -/
def p : ℝ := 0.6

/-- The number of shots -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in n shots -/
def prob_at_least_two (p : ℝ) (n : ℕ) : ℝ :=
  3 * p^2 * (1 - p) + p^3

theorem target_probability :
  prob_at_least_two p n = 0.648 :=
sorry

end NUMINAMATH_CALUDE_target_probability_l685_68579


namespace NUMINAMATH_CALUDE_power_of_power_l685_68542

theorem power_of_power : (3^4)^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l685_68542


namespace NUMINAMATH_CALUDE_cubic_polynomial_value_at_5_l685_68523

/-- A cubic polynomial satisfying specific conditions -/
def cubic_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, p x = a*x^3 + b*x^2 + c*x + d) ∧
  (p 1 = 1) ∧
  (p 2 = 1/8) ∧
  (p 3 = 1/27) ∧
  (p 4 = 1/64)

/-- The main theorem -/
theorem cubic_polynomial_value_at_5 (p : ℝ → ℝ) (h : cubic_polynomial p) :
  p 5 = -76/375 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_value_at_5_l685_68523


namespace NUMINAMATH_CALUDE_unreachable_one_if_not_div_three_l685_68580

/-- The operation of adding 3 repeatedly until divisible by 5, then dividing by 5 -/
def operation (n : ℕ) : ℕ :=
  let m := n + 3 * (5 - n % 5) % 5
  m / 5

/-- Predicate to check if a number can reach 1 through repeated applications of the operation -/
def can_reach_one (n : ℕ) : Prop :=
  ∃ k : ℕ, (operation^[k] n) = 1

/-- Theorem stating that numbers not divisible by 3 cannot reach 1 through the given operations -/
theorem unreachable_one_if_not_div_three (n : ℕ) (h : ¬ 3 ∣ n) : ¬ can_reach_one n :=
sorry

end NUMINAMATH_CALUDE_unreachable_one_if_not_div_three_l685_68580


namespace NUMINAMATH_CALUDE_triangle_equilateral_l685_68507

/-- Given a triangle ABC with angle C = 60° and c² = ab, prove that ABC is equilateral -/
theorem triangle_equilateral (a b c : ℝ) (angleC : ℝ) :
  angleC = π / 3 →  -- 60° in radians
  c^2 = a * b →
  a > 0 → b > 0 → c > 0 →
  a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l685_68507


namespace NUMINAMATH_CALUDE_hexagon_tile_difference_l685_68532

/-- Given a hexagonal figure with initial red and yellow tiles, prove the difference
    between yellow and red tiles after adding a border of yellow tiles. -/
theorem hexagon_tile_difference (initial_red : ℕ) (initial_yellow : ℕ) 
    (sides : ℕ) (tiles_per_side : ℕ) :
  initial_red = 15 →
  initial_yellow = 9 →
  sides = 6 →
  tiles_per_side = 4 →
  let new_yellow := initial_yellow + sides * tiles_per_side
  new_yellow - initial_red = 18 := by
sorry

end NUMINAMATH_CALUDE_hexagon_tile_difference_l685_68532


namespace NUMINAMATH_CALUDE_campers_rowing_total_l685_68591

/-- The total number of campers who went rowing throughout the day -/
def total_campers (morning afternoon evening : ℕ) : ℕ :=
  morning + afternoon + evening

/-- Theorem stating that the total number of campers who went rowing is 764 -/
theorem campers_rowing_total :
  total_campers 235 387 142 = 764 := by
  sorry

end NUMINAMATH_CALUDE_campers_rowing_total_l685_68591


namespace NUMINAMATH_CALUDE_particular_propositions_count_l685_68576

-- Define a proposition type
inductive Proposition
| ExistsDivisorImpossible
| PrismIsPolyhedron
| AllEquationsHaveRealSolutions
| SomeTrianglesAreAcute

-- Define a function to check if a proposition is particular
def isParticular (p : Proposition) : Bool :=
  match p with
  | Proposition.ExistsDivisorImpossible => true
  | Proposition.PrismIsPolyhedron => false
  | Proposition.AllEquationsHaveRealSolutions => false
  | Proposition.SomeTrianglesAreAcute => true

-- Define the list of all propositions
def allPropositions : List Proposition :=
  [Proposition.ExistsDivisorImpossible, Proposition.PrismIsPolyhedron,
   Proposition.AllEquationsHaveRealSolutions, Proposition.SomeTrianglesAreAcute]

-- Theorem: The number of particular propositions is 2
theorem particular_propositions_count :
  (allPropositions.filter isParticular).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_particular_propositions_count_l685_68576


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l685_68566

theorem gcd_of_three_numbers : Nat.gcd 9242 (Nat.gcd 13863 34657) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l685_68566


namespace NUMINAMATH_CALUDE_ace_spade_probability_l685_68548

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size_eq : cards.card = 52)

/-- Represents the event of drawing an Ace first and a spade second -/
def ace_spade_event (deck : Deck) : Finset (Nat × Nat × Nat × Nat) :=
  sorry

/-- The probability of the ace_spade_event -/
def ace_spade_prob (deck : Deck) : ℚ :=
  (ace_spade_event deck).card / deck.cards.card / (deck.cards.card - 1)

theorem ace_spade_probability (deck : Deck) : 
  ace_spade_prob deck = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_ace_spade_probability_l685_68548


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l685_68539

theorem square_root_equation_solution : 
  ∃ x : ℝ, (56^2 + 56^2) / x^2 = 8 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l685_68539


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l685_68519

theorem simultaneous_equations_solution :
  ∀ x y : ℝ,
  (3 * x^2 + x * y - 2 * y^2 = -5 ∧ x^2 + 2 * x * y + y^2 = 1) ↔
  ((x = 3/5 ∧ y = -8/5) ∨ (x = -3/5 ∧ y = 8/5)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l685_68519


namespace NUMINAMATH_CALUDE_gcf_36_45_l685_68554

theorem gcf_36_45 : Nat.gcd 36 45 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcf_36_45_l685_68554


namespace NUMINAMATH_CALUDE_coin_weight_verification_l685_68560

theorem coin_weight_verification (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ 
  x + y = 3 ∧ 
  x + (x + y) + (x + 2*y) = 9 ∧ 
  y + (x + y) + (x + 2*y) = x + 9 → 
  x = 1 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_coin_weight_verification_l685_68560


namespace NUMINAMATH_CALUDE_root_values_l685_68577

theorem root_values (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : p * m^3 + q * m^2 + r * m + s = 0)
  (h2 : q * m^3 + r * m^2 + (s + m * m) * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_root_values_l685_68577


namespace NUMINAMATH_CALUDE_expression_value_at_sqrt3_over_2_l685_68513

theorem expression_value_at_sqrt3_over_2 :
  let x : ℝ := Real.sqrt 3 / 2
  (1 + x) / (1 + Real.sqrt (1 + x)) + (1 - x) / (1 - Real.sqrt (1 - x)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_sqrt3_over_2_l685_68513


namespace NUMINAMATH_CALUDE_inverse_proportion_translation_l685_68541

/-- Given a non-zero constant k and a function f(x) = k/(x+1) - 2,
    if f(-3) = 1, then k = -6 -/
theorem inverse_proportion_translation (k : ℝ) (hk : k ≠ 0) :
  let f : ℝ → ℝ := λ x => k / (x + 1) - 2
  f (-3) = 1 → k = -6 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_translation_l685_68541


namespace NUMINAMATH_CALUDE_compound_interest_rate_equation_l685_68501

/-- Proves that the given compound interest scenario results in the specified equation for the interest rate. -/
theorem compound_interest_rate_equation (P r : ℝ) 
  (h1 : P * (1 + r)^3 = 310) 
  (h2 : P * (1 + r)^8 = 410) : 
  (1 + r)^5 = 410/310 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_equation_l685_68501


namespace NUMINAMATH_CALUDE_regular_poly15_distance_sum_l685_68540

/-- Regular 15-sided polygon -/
structure RegularPoly15 where
  vertices : Fin 15 → ℝ × ℝ
  is_regular : ∀ i j : Fin 15, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

/-- Distance between two vertices -/
def dist_vertices (p : RegularPoly15) (i j : Fin 15) : ℝ :=
  dist (p.vertices i) (p.vertices j)

/-- Theorem statement -/
theorem regular_poly15_distance_sum (p : RegularPoly15) :
  1 / dist_vertices p 0 2 + 1 / dist_vertices p 0 4 + 1 / dist_vertices p 0 7 =
  1 / dist_vertices p 0 1 := by
  sorry

end NUMINAMATH_CALUDE_regular_poly15_distance_sum_l685_68540


namespace NUMINAMATH_CALUDE_total_tylenol_grams_l685_68593

-- Define the parameters
def tablet_count : ℕ := 2
def tablet_mg : ℕ := 500
def hours_between_doses : ℕ := 4
def total_hours : ℕ := 12
def mg_per_gram : ℕ := 1000

-- Theorem statement
theorem total_tylenol_grams : 
  (total_hours / hours_between_doses) * tablet_count * tablet_mg / mg_per_gram = 3 := by
  sorry

end NUMINAMATH_CALUDE_total_tylenol_grams_l685_68593


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l685_68517

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statement
theorem intersection_complement_equality :
  N ∩ (U \ M) = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l685_68517


namespace NUMINAMATH_CALUDE_linear_function_increasing_l685_68503

/-- Given a linear function y = 2x + 1 and two points on this function,
    if the x-coordinate of the first point is less than the x-coordinate of the second point,
    then the y-coordinate of the first point is less than the y-coordinate of the second point. -/
theorem linear_function_increasing (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = 2 * x₁ + 1 →
  y₂ = 2 * x₂ + 1 →
  x₁ < x₂ →
  y₁ < y₂ :=
by sorry

end NUMINAMATH_CALUDE_linear_function_increasing_l685_68503


namespace NUMINAMATH_CALUDE_min_homeowners_l685_68502

theorem min_homeowners (total : ℕ) (men women : ℕ) (men_ratio women_ratio : ℚ) : 
  total = 150 →
  men + women = total →
  men_ratio = 1/10 →
  women_ratio = 1/5 →
  men ≥ 0 →
  women ≥ 0 →
  ∃ (homeowners : ℕ), homeowners = 16 ∧ 
    ∀ (h : ℕ), h ≥ men_ratio * men + women_ratio * women → h ≥ homeowners :=
by sorry

end NUMINAMATH_CALUDE_min_homeowners_l685_68502


namespace NUMINAMATH_CALUDE_customers_who_tipped_l685_68538

/-- The number of customers who left a tip at 'The Greasy Spoon' restaurant -/
theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) :
  initial_customers = 29 →
  additional_customers = 20 →
  non_tipping_customers = 34 →
  initial_customers + additional_customers - non_tipping_customers = 15 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_tipped_l685_68538


namespace NUMINAMATH_CALUDE_hair_length_calculation_l685_68556

/-- Calculates the final hair length after a series of changes. -/
def finalHairLength (initialLength : ℝ) (firstCutFraction : ℝ) (growth : ℝ) (secondCut : ℝ) : ℝ :=
  (initialLength - firstCutFraction * initialLength + growth) - secondCut

/-- Theorem stating that given the specific hair length changes, the final length is 14 inches. -/
theorem hair_length_calculation :
  finalHairLength 24 0.5 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_hair_length_calculation_l685_68556


namespace NUMINAMATH_CALUDE_largest_c_for_max_function_l685_68509

open Real

/-- Given positive real numbers a and b, the largest real c such that 
    c ≤ max(ax + 1/(ax), bx + 1/(bx)) for all positive real x is 2. -/
theorem largest_c_for_max_function (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ c : ℝ, ∀ x : ℝ, x > 0 → c ≤ max (a * x + 1 / (a * x)) (b * x + 1 / (b * x))) ∧
  (∀ c : ℝ, (∀ x : ℝ, x > 0 → c ≤ max (a * x + 1 / (a * x)) (b * x + 1 / (b * x))) → c ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_max_function_l685_68509


namespace NUMINAMATH_CALUDE_configurations_count_l685_68528

/-- The number of squares in the set -/
def total_squares : ℕ := 8

/-- The number of squares to be placed -/
def squares_to_place : ℕ := 2

/-- The number of distinct sides on which squares can be placed -/
def distinct_sides : ℕ := 2

/-- The number of configurations that can be formed -/
def num_configurations : ℕ := total_squares * (total_squares - 1)

theorem configurations_count :
  num_configurations = 56 :=
sorry

end NUMINAMATH_CALUDE_configurations_count_l685_68528


namespace NUMINAMATH_CALUDE_intersection_when_a_is_quarter_b_necessary_condition_for_a_l685_68506

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 3*a + 1}

-- Theorem for part 1
theorem intersection_when_a_is_quarter :
  A ∩ B (1/4) = {x | 1 < x ∧ x < 7/4} := by sorry

-- Theorem for part 2
theorem b_necessary_condition_for_a (a : ℝ) :
  (∀ x, x ∈ A → x ∈ B a) ↔ 1/3 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_quarter_b_necessary_condition_for_a_l685_68506


namespace NUMINAMATH_CALUDE_lcm_of_36_and_132_l685_68505

theorem lcm_of_36_and_132 (hcf : ℕ) (lcm : ℕ) :
  hcf = 12 →
  lcm = 36 * 132 / hcf →
  lcm = 396 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_132_l685_68505


namespace NUMINAMATH_CALUDE_parallel_condition_l685_68573

def a : ℝ × ℝ := (1, -4)
def b (x : ℝ) : ℝ × ℝ := (-1, x)
def c (x : ℝ) : ℝ × ℝ := a + 3 • (b x)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v = k • w

theorem parallel_condition (x : ℝ) :
  parallel a (c x) ↔ x = 4 := by sorry

end NUMINAMATH_CALUDE_parallel_condition_l685_68573


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l685_68520

theorem compare_negative_fractions : -2/3 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l685_68520


namespace NUMINAMATH_CALUDE_major_axis_length_for_specific_cylinder_l685_68545

/-- The length of the major axis of an ellipse formed by intersecting a right circular cylinder --/
def majorAxisLength (cylinderRadius : ℝ) (majorMinorRatio : ℝ) : ℝ :=
  2 * cylinderRadius * majorMinorRatio

/-- Theorem: The major axis length for the given conditions --/
theorem major_axis_length_for_specific_cylinder :
  majorAxisLength 3 1.2 = 7.2 :=
by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_for_specific_cylinder_l685_68545


namespace NUMINAMATH_CALUDE_chemistry_class_size_l685_68535

theorem chemistry_class_size :
  -- Total number of students
  let total : ℕ := 120
  -- Students in both chemistry and biology
  let chem_bio : ℕ := 35
  -- Students in both biology and physics
  let bio_phys : ℕ := 15
  -- Students in both chemistry and physics
  let chem_phys : ℕ := 10
  -- Function to calculate total students in a class
  let class_size (only : ℕ) (with_other1 : ℕ) (with_other2 : ℕ) := only + with_other1 + with_other2
  -- Constraint: Chemistry class is four times as large as biology class
  ∀ (bio_only : ℕ) (chem_only : ℕ) (phys_only : ℕ),
    class_size chem_only chem_bio chem_phys = 4 * class_size bio_only chem_bio bio_phys →
    -- Constraint: No student takes all three classes
    class_size bio_only chem_bio bio_phys + class_size chem_only chem_bio chem_phys + class_size phys_only bio_phys chem_phys = total →
    -- Conclusion: Chemistry class size is 198
    class_size chem_only chem_bio chem_phys = 198 :=
by
  sorry

end NUMINAMATH_CALUDE_chemistry_class_size_l685_68535


namespace NUMINAMATH_CALUDE_system_solution_l685_68572

theorem system_solution : 
  let x : ℚ := -24/13
  let y : ℚ := 18/13
  let z : ℚ := -23/13
  (3*x + 2*y = z - 1) ∧ 
  (2*x - y = 4*z + 2) ∧ 
  (x + 4*y = 3*z + 9) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l685_68572


namespace NUMINAMATH_CALUDE_age_relation_l685_68524

/-- Given that p is currently 3 times as old as q and p was 30 years old 3 years ago,
    prove that p will be twice as old as q in 11 years. -/
theorem age_relation (p q : ℕ) (x : ℕ) : 
  p = 3 * q →  -- p is 3 times as old as q
  p = 30 + 3 →  -- p was 30 years old 3 years ago
  p + x = 2 * (q + x) →  -- in x years, p will be twice as old as q
  x = 11 := by
sorry

end NUMINAMATH_CALUDE_age_relation_l685_68524


namespace NUMINAMATH_CALUDE_distance_between_points_l685_68568

/-- The distance between points A and B given specific square dimensions -/
theorem distance_between_points (small_perimeter : ℝ) (large_area : ℝ) : 
  small_perimeter = 8 → large_area = 25 → ∃ (dist : ℝ), dist^2 = 58 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l685_68568


namespace NUMINAMATH_CALUDE_estimate_boys_in_grade_l685_68569

theorem estimate_boys_in_grade (total_students : ℕ) (sample_size : ℕ) (girls_in_sample : ℕ) 
  (h1 : total_students = 1200)
  (h2 : sample_size = 20)
  (h3 : girls_in_sample = 8) :
  total_students - (girls_in_sample * total_students / sample_size) = 720 := by
  sorry

end NUMINAMATH_CALUDE_estimate_boys_in_grade_l685_68569


namespace NUMINAMATH_CALUDE_max_polygons_bound_l685_68582

/-- The number of points marked on the circle. -/
def num_points : ℕ := 12

/-- The minimum allowed internal angle at the circle's center (in degrees). -/
def min_angle : ℝ := 30

/-- A function that calculates the maximum number of distinct convex polygons
    that can be formed under the given conditions. -/
def max_polygons (n : ℕ) (θ : ℝ) : ℕ :=
  2^n - (n.choose 0 + n.choose 1 + n.choose 2)

/-- Theorem stating that the maximum number of distinct convex polygons
    satisfying the conditions is less than or equal to 4017. -/
theorem max_polygons_bound :
  max_polygons num_points min_angle ≤ 4017 :=
sorry

end NUMINAMATH_CALUDE_max_polygons_bound_l685_68582


namespace NUMINAMATH_CALUDE_group_arrangement_count_l685_68583

theorem group_arrangement_count :
  let total_men : ℕ := 4
  let total_women : ℕ := 5
  let small_group_size : ℕ := 2
  let large_group_size : ℕ := 5
  let small_group_count : ℕ := 2
  let large_group_count : ℕ := 1
  let total_people : ℕ := total_men + total_women
  let total_groups : ℕ := small_group_count + large_group_count

  -- Condition: At least one man and one woman in each group
  ∀ (men_in_small_group men_in_large_group : ℕ),
    (men_in_small_group ≥ 1 ∧ men_in_small_group < small_group_size) →
    (men_in_large_group ≥ 1 ∧ men_in_large_group < large_group_size) →
    (men_in_small_group * small_group_count + men_in_large_group * large_group_count = total_men) →

  -- The number of ways to arrange the groups
  (Nat.choose total_men 2 * Nat.choose total_women 3 +
   Nat.choose total_men 3 * Nat.choose total_women 2) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_group_arrangement_count_l685_68583


namespace NUMINAMATH_CALUDE_ab_negative_necessary_not_sufficient_l685_68559

-- Define what it means for an equation to represent a hyperbola
def represents_hyperbola (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, a * x^2 + b * y^2 = c ∧ 
  ((a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) ∧ 
  c ≠ 0

-- State the theorem
theorem ab_negative_necessary_not_sufficient :
  (∀ a b c : ℝ, represents_hyperbola a b c → a * b < 0) ∧
  ¬(∀ a b c : ℝ, a * b < 0 → represents_hyperbola a b c) :=
by sorry

end NUMINAMATH_CALUDE_ab_negative_necessary_not_sufficient_l685_68559


namespace NUMINAMATH_CALUDE_tan_alpha_negative_three_l685_68516

theorem tan_alpha_negative_three (α : Real) (h : Real.tan α = -3) :
  (3 * Real.sin α - 3 * Real.cos α) / (6 * Real.cos α + Real.sin α) = -4 ∧
  1 / (Real.sin α * Real.cos α + 1 + Real.cos (2 * α)) = -10 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_negative_three_l685_68516


namespace NUMINAMATH_CALUDE_julia_play_difference_l685_68595

/-- The number of kids Julia played tag with on Monday -/
def monday_tag : ℕ := 28

/-- The number of kids Julia played hide & seek with on Monday -/
def monday_hide_seek : ℕ := 15

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_tag : ℕ := 33

/-- The number of kids Julia played hide & seek with on Tuesday -/
def tuesday_hide_seek : ℕ := 21

/-- The difference in the total number of kids Julia played with on Tuesday compared to Monday -/
theorem julia_play_difference : 
  (tuesday_tag + tuesday_hide_seek) - (monday_tag + monday_hide_seek) = 11 := by
  sorry

end NUMINAMATH_CALUDE_julia_play_difference_l685_68595


namespace NUMINAMATH_CALUDE_miles_reads_128_pages_l685_68551

/-- Calculates the total number of pages Miles reads in 1/6 of a day -/
def total_pages_read (hours_in_day : ℚ) (reading_fraction : ℚ) 
  (novel_pages_per_hour : ℚ) (graphic_novel_pages_per_hour : ℚ) (comic_pages_per_hour : ℚ) 
  (time_fraction_per_book_type : ℚ) : ℚ :=
  let total_reading_hours := hours_in_day * reading_fraction
  let hours_per_book_type := total_reading_hours * time_fraction_per_book_type
  let novel_pages := novel_pages_per_hour * hours_per_book_type
  let graphic_novel_pages := graphic_novel_pages_per_hour * hours_per_book_type
  let comic_pages := comic_pages_per_hour * hours_per_book_type
  novel_pages + graphic_novel_pages + comic_pages

/-- Theorem stating that Miles reads 128 pages in total -/
theorem miles_reads_128_pages :
  total_pages_read 24 (1/6) 21 30 45 (1/3) = 128 := by
  sorry

end NUMINAMATH_CALUDE_miles_reads_128_pages_l685_68551
