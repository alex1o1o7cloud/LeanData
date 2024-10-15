import Mathlib

namespace NUMINAMATH_CALUDE_hide_and_seek_time_l1527_152743

/-- Represents the square wall in the hide and seek game -/
structure Square :=
  (side_length : ℝ)

/-- Represents a player in the hide and seek game -/
structure Player :=
  (speed : ℝ)
  (corner_pause : ℝ)

/-- Calculates the time needed for a player to see the other player -/
def time_to_see (s : Square) (a b : Player) : ℝ :=
  sorry

/-- Theorem stating that the minimum time for A to see B is 8 minutes -/
theorem hide_and_seek_time (s : Square) (a b : Player) :
  s.side_length = 100 ∧
  a.speed = 50 ∧
  b.speed = 30 ∧
  a.corner_pause = 1 ∧
  b.corner_pause = 1 →
  time_to_see s a b = 8 :=
sorry

end NUMINAMATH_CALUDE_hide_and_seek_time_l1527_152743


namespace NUMINAMATH_CALUDE_basketball_tryouts_l1527_152729

theorem basketball_tryouts (girls : ℕ) (called_back : ℕ) (not_selected : ℕ) :
  girls = 17 → called_back = 10 → not_selected = 39 →
  ∃ (boys : ℕ), girls + boys = called_back + not_selected ∧ boys = 32 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l1527_152729


namespace NUMINAMATH_CALUDE_set_intersection_equality_l1527_152745

def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x ≤ 3}

theorem set_intersection_equality : A ∩ B = A := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l1527_152745


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l1527_152796

theorem x_range_for_quadratic_inequality :
  ∀ x : ℝ,
  (∀ m : ℝ, |m| ≤ 2 → m * x^2 - 2*x - m + 1 < 0) →
  (-1 + Real.sqrt 7) / 2 < x ∧ x < (1 + Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l1527_152796


namespace NUMINAMATH_CALUDE_daniels_candies_l1527_152770

theorem daniels_candies (x : ℕ) : 
  (x : ℚ) * 3/8 - 3/2 - 6 = 10 ↔ x = 93 := by
  sorry

end NUMINAMATH_CALUDE_daniels_candies_l1527_152770


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1527_152728

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x * y = -2) 
  (h2 : y - 2 * x = 5) : 
  8 * x^3 * y - 8 * x^2 * y^2 + 2 * x * y^3 = -100 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1527_152728


namespace NUMINAMATH_CALUDE_problem_statement_l1527_152785

theorem problem_statement :
  ∀ m n : ℤ,
  m = -(-6) →
  -n = -1 →
  m * n - 7 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1527_152785


namespace NUMINAMATH_CALUDE_revolver_game_probability_l1527_152788

/-- Represents a six-shot revolver with one bullet -/
structure Revolver :=
  (chambers : Fin 6)
  (bullet : Fin 6)

/-- Represents the state of the game -/
inductive GameState
  | A
  | B

/-- The probability of firing the bullet on a single shot -/
def fire_probability : ℚ := 1 / 6

/-- The probability of not firing the bullet on a single shot -/
def not_fire_probability : ℚ := 1 - fire_probability

/-- The probability that A fires the bullet -/
noncomputable def prob_A_fires : ℚ :=
  fire_probability / (1 - not_fire_probability * not_fire_probability)

theorem revolver_game_probability :
  prob_A_fires = 6 / 11 :=
sorry

end NUMINAMATH_CALUDE_revolver_game_probability_l1527_152788


namespace NUMINAMATH_CALUDE_no_such_function_exists_l1527_152787

open Set
open Function
open Real

theorem no_such_function_exists :
  ¬∃ f : {x : ℝ | x > 0} → {x : ℝ | x > 0},
    ∀ (x y : ℝ) (hx : x > 0) (hy : y > 0),
      f ⟨x + y, add_pos hx hy⟩ ≥ f ⟨x, hx⟩ + y * f (f ⟨x, hx⟩) :=
by
  sorry


end NUMINAMATH_CALUDE_no_such_function_exists_l1527_152787


namespace NUMINAMATH_CALUDE_pizza_order_theorem_l1527_152767

def pizza_order_cost (base_price : ℕ) (topping_price : ℕ) (tip : ℕ) : Prop :=
  let pepperoni_cost : ℕ := base_price + topping_price
  let sausage_cost : ℕ := base_price + topping_price
  let olive_mushroom_cost : ℕ := base_price + 2 * topping_price
  let total_before_tip : ℕ := pepperoni_cost + sausage_cost + olive_mushroom_cost
  let total_with_tip : ℕ := total_before_tip + tip
  total_with_tip = 39

theorem pizza_order_theorem :
  pizza_order_cost 10 1 5 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_order_theorem_l1527_152767


namespace NUMINAMATH_CALUDE_complex_collinearity_l1527_152755

/-- A complex number represented as a point in the plane -/
structure ComplexPoint where
  re : ℝ
  im : ℝ

/-- Check if three ComplexPoints are collinear -/
def areCollinear (p q r : ComplexPoint) : Prop :=
  ∃ k : ℝ, (r.re - p.re, r.im - p.im) = k • (q.re - p.re, q.im - p.im)

theorem complex_collinearity :
  ∃! b : ℝ, areCollinear
    (ComplexPoint.mk 3 (-5))
    (ComplexPoint.mk 1 (-1))
    (ComplexPoint.mk (-2) b) ∧
  b = 5 := by sorry

end NUMINAMATH_CALUDE_complex_collinearity_l1527_152755


namespace NUMINAMATH_CALUDE_divisibility_of_f_l1527_152723

def f (x : ℕ) : ℕ := x^3 + 17

theorem divisibility_of_f :
  ∀ n : ℕ, n ≥ 2 →
  ∃ x : ℕ, (3^n ∣ f x) ∧ ¬(3^(n+1) ∣ f x) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_f_l1527_152723


namespace NUMINAMATH_CALUDE_probability_of_sum_three_is_one_over_216_l1527_152786

def standard_die := Finset.range 6

def roll_sum (a b c : ℕ) : ℕ := a + b + c

def probability_of_sum_three : ℚ :=
  (Finset.filter (λ (abc : ℕ × ℕ × ℕ) => roll_sum abc.1 abc.2.1 abc.2.2 = 3) 
    (standard_die.product (standard_die.product standard_die))).card / 
  (standard_die.card ^ 3 : ℚ)

theorem probability_of_sum_three_is_one_over_216 :
  probability_of_sum_three = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_probability_of_sum_three_is_one_over_216_l1527_152786


namespace NUMINAMATH_CALUDE_no_valid_coloring_l1527_152706

-- Define a color type
inductive Color
| Blue
| Red
| Green

-- Define a coloring function type
def Coloring := Nat → Color

-- Define the property that all three colors are used
def AllColorsUsed (f : Coloring) : Prop :=
  ∃ (a b c : Nat), a > 1 ∧ b > 1 ∧ c > 1 ∧ 
    f a = Color.Blue ∧ f b = Color.Red ∧ f c = Color.Green

-- Define the property that the product of two differently colored numbers
-- has a different color from both multipliers
def ValidColoring (f : Coloring) : Prop :=
  ∀ (a b : Nat), a > 1 → b > 1 → f a ≠ f b →
    f (a * b) ≠ f a ∧ f (a * b) ≠ f b

-- State the theorem
theorem no_valid_coloring :
  ¬∃ (f : Coloring), AllColorsUsed f ∧ ValidColoring f :=
sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l1527_152706


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l1527_152764

/-- An arithmetic sequence with sum of first n terms Sn = n^2 + bn + c -/
structure ArithmeticSequence where
  b : ℝ
  c : ℝ
  sum : ℕ+ → ℝ
  sum_eq : ∀ n : ℕ+, sum n = n.val ^ 2 + b * n.val + c

/-- The second term of the arithmetic sequence -/
def ArithmeticSequence.a2 (seq : ArithmeticSequence) : ℝ :=
  seq.sum 2 - seq.sum 1

/-- The third term of the arithmetic sequence -/
def ArithmeticSequence.a3 (seq : ArithmeticSequence) : ℝ :=
  seq.sum 3 - seq.sum 2

theorem arithmetic_sequence_theorem (seq : ArithmeticSequence) 
  (h : seq.a2 + seq.a3 = 4) : 
  seq.c = 0 ∧ seq.b = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l1527_152764


namespace NUMINAMATH_CALUDE_smallest_number_game_l1527_152783

theorem smallest_number_game (alice_number : ℕ) (bob_number : ℕ) : 
  alice_number = 45 →
  (∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ bob_number) →
  5 ∣ bob_number →
  bob_number > 0 →
  (∀ n : ℕ, n > 0 → (∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ n) → 5 ∣ n → n ≥ bob_number) →
  bob_number = 15 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_game_l1527_152783


namespace NUMINAMATH_CALUDE_gcd_315_168_l1527_152799

theorem gcd_315_168 : Nat.gcd 315 168 = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_315_168_l1527_152799


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1527_152732

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 48 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 3 + a 10 + a 11 = 48

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : sum_condition a) : 
  a 6 + a 7 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1527_152732


namespace NUMINAMATH_CALUDE_irrational_sum_equivalence_l1527_152747

theorem irrational_sum_equivalence 
  (a b c d : ℝ) 
  (ha : Irrational a) 
  (hb : Irrational b) 
  (hc : Irrational c) 
  (hd : Irrational d) 
  (hab : a + b = 1) :
  (c + d = 1) ↔ 
  (∀ n : ℕ+, ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋ + ⌊n * d⌋) :=
by sorry

end NUMINAMATH_CALUDE_irrational_sum_equivalence_l1527_152747


namespace NUMINAMATH_CALUDE_point_q_in_third_quadrant_l1527_152754

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the third quadrant -/
def is_in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If A is in the second quadrant, then Q is in the third quadrant -/
theorem point_q_in_third_quadrant (A : Point) (h : is_in_second_quadrant A) :
  let Q : Point := ⟨A.x, -A.y⟩
  is_in_third_quadrant Q :=
by
  sorry

end NUMINAMATH_CALUDE_point_q_in_third_quadrant_l1527_152754


namespace NUMINAMATH_CALUDE_carlas_sunflowers_l1527_152744

/-- The number of sunflowers Carla has -/
def num_sunflowers : ℕ := sorry

/-- The number of dandelions Carla has -/
def num_dandelions : ℕ := 8

/-- The number of seeds per sunflower -/
def seeds_per_sunflower : ℕ := 9

/-- The number of seeds per dandelion -/
def seeds_per_dandelion : ℕ := 12

/-- The percentage of seeds that come from dandelions -/
def dandelion_seed_percentage : ℚ := 64 / 100

theorem carlas_sunflowers : 
  num_sunflowers = 6 ∧
  num_dandelions * seeds_per_dandelion = 
    (dandelion_seed_percentage : ℚ) * 
    (num_sunflowers * seeds_per_sunflower + num_dandelions * seeds_per_dandelion) :=
by sorry

end NUMINAMATH_CALUDE_carlas_sunflowers_l1527_152744


namespace NUMINAMATH_CALUDE_sqrt_difference_abs_plus_two_sqrt_two_l1527_152748

theorem sqrt_difference_abs_plus_two_sqrt_two :
  |Real.sqrt 2 - Real.sqrt 3| + 2 * Real.sqrt 2 = Real.sqrt 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_abs_plus_two_sqrt_two_l1527_152748


namespace NUMINAMATH_CALUDE_ackermann_3_1_l1527_152735

def B : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => B m 1
  | m + 1, n + 1 => B m (B (m + 1) n)

theorem ackermann_3_1 : B 3 1 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ackermann_3_1_l1527_152735


namespace NUMINAMATH_CALUDE_integral_of_f_with_min_neg_one_l1527_152737

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

-- State the theorem
theorem integral_of_f_with_min_neg_one (m : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f m x ≤ f m y) ∧ 
  (∃ (x : ℝ), f m x = -1) →
  ∫ x in (1 : ℝ)..(2 : ℝ), f m x = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_of_f_with_min_neg_one_l1527_152737


namespace NUMINAMATH_CALUDE_difference_of_squares_fraction_l1527_152700

theorem difference_of_squares_fraction :
  (113^2 - 104^2) / 9 = 217 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_fraction_l1527_152700


namespace NUMINAMATH_CALUDE_inequality_solution_l1527_152710

theorem inequality_solution (x : ℝ) : 
  (x^2 - 3*x + 3)^(4*x^3 + 5*x^2) ≤ (x^2 - 3*x + 3)^(2*x^3 + 18*x) ↔ 
  x ≤ -9/2 ∨ (0 ≤ x ∧ x ≤ 1) ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1527_152710


namespace NUMINAMATH_CALUDE_one_refilling_cost_l1527_152702

/-- Given that Greyson spent 40 dollars on fuel this week and refilled 4 times,
    prove that the cost of one refilling is 10 dollars. -/
theorem one_refilling_cost (total_spent : ℝ) (num_refills : ℕ) 
  (h1 : total_spent = 40)
  (h2 : num_refills = 4) :
  total_spent / num_refills = 10 := by
  sorry

end NUMINAMATH_CALUDE_one_refilling_cost_l1527_152702


namespace NUMINAMATH_CALUDE_raccoon_carrots_l1527_152793

theorem raccoon_carrots (raccoon_per_hole rabbit_per_hole : ℕ) 
  (hole_difference : ℕ) (total_carrots : ℕ) : 
  raccoon_per_hole = 5 →
  rabbit_per_hole = 8 →
  hole_difference = 3 →
  raccoon_per_hole * (hole_difference + total_carrots / rabbit_per_hole) = total_carrots →
  total_carrots = 40 :=
by
  sorry

#check raccoon_carrots

end NUMINAMATH_CALUDE_raccoon_carrots_l1527_152793


namespace NUMINAMATH_CALUDE_triangle_gp_length_l1527_152730

-- Define the triangle DEF
structure Triangle :=
  (DE DF EF : ℝ)

-- Define the centroid G and point P
structure TrianglePoints (t : Triangle) :=
  (G P : ℝ × ℝ)

-- Define the length of GP
def lengthGP (t : Triangle) (tp : TrianglePoints t) : ℝ :=
  sorry

-- Theorem statement
theorem triangle_gp_length (t : Triangle) (tp : TrianglePoints t) 
  (h1 : t.DE = 10) (h2 : t.DF = 15) (h3 : t.EF = 17) : 
  lengthGP t tp = 4 * Real.sqrt 154 / 17 := by
  sorry

end NUMINAMATH_CALUDE_triangle_gp_length_l1527_152730


namespace NUMINAMATH_CALUDE_carla_school_distance_l1527_152746

theorem carla_school_distance (grocery_distance : ℝ) (soccer_distance : ℝ) 
  (mpg : ℝ) (gas_price : ℝ) (gas_spent : ℝ) :
  grocery_distance = 8 →
  soccer_distance = 12 →
  mpg = 25 →
  gas_price = 2.5 →
  gas_spent = 5 →
  ∃ (school_distance : ℝ),
    grocery_distance + school_distance + soccer_distance + 2 * school_distance = 
      (gas_spent / gas_price) * mpg ∧
    school_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_carla_school_distance_l1527_152746


namespace NUMINAMATH_CALUDE_sin_product_18_54_72_36_l1527_152705

theorem sin_product_18_54_72_36 :
  Real.sin (18 * π / 180) * Real.sin (54 * π / 180) *
  Real.sin (72 * π / 180) * Real.sin (36 * π / 180) =
  (Real.sqrt 5 + 1) / 16 := by sorry

end NUMINAMATH_CALUDE_sin_product_18_54_72_36_l1527_152705


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l1527_152780

open Real

theorem range_of_a_for_inequality (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ a * x₀ - log x₀ < 0) → a < 1 / exp 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l1527_152780


namespace NUMINAMATH_CALUDE_triangle_area_maximized_l1527_152751

theorem triangle_area_maximized (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  a / (Real.sin A) = c / (Real.sin C) ∧
  Real.tan A = 2 * Real.tan B ∧
  b = Real.sqrt 2 →
  (∀ A' B' C' a' b' c' : ℝ,
    0 < A' ∧ 0 < B' ∧ 0 < C' ∧
    A' + B' + C' = π ∧
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧
    a' / (Real.sin A') = b' / (Real.sin B') ∧
    a' / (Real.sin A') = c' / (Real.sin C') ∧
    Real.tan A' = 2 * Real.tan B' ∧
    b' = Real.sqrt 2 →
    1/2 * a * b * Real.sin C ≥ 1/2 * a' * b' * Real.sin C') →
  a = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_maximized_l1527_152751


namespace NUMINAMATH_CALUDE_white_paint_calculation_l1527_152776

/-- Given the total amount of paint and the amounts of green and brown paint,
    calculate the amount of white paint needed. -/
theorem white_paint_calculation (total green brown : ℕ) (h1 : total = 69) 
    (h2 : green = 15) (h3 : brown = 34) : total - (green + brown) = 20 := by
  sorry

end NUMINAMATH_CALUDE_white_paint_calculation_l1527_152776


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l1527_152775

/-- Represents the first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 3 + (n - 1) * n / 2

/-- Represents the last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- Represents the sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

theorem sum_of_15th_set : S 15 = 1725 := by sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l1527_152775


namespace NUMINAMATH_CALUDE_female_half_marathon_count_half_marathon_probability_no_significant_relation_l1527_152733

/-- Represents the number of students in each category --/
structure StudentCounts where
  male_half_marathon : ℕ
  male_mini_run : ℕ
  female_half_marathon : ℕ
  female_mini_run : ℕ

/-- The given student counts --/
def given_counts : StudentCounts := {
  male_half_marathon := 20,
  male_mini_run := 10,
  female_half_marathon := 10,  -- This is 'a', which we'll prove
  female_mini_run := 10
}

/-- The ratio of male to female students --/
def male_female_ratio : ℚ := 3 / 2

/-- Theorem stating the correct number of female students in half marathon --/
theorem female_half_marathon_count :
  given_counts.female_half_marathon = 10 := by sorry

/-- Theorem stating the probability of choosing half marathon --/
theorem half_marathon_probability :
  (given_counts.male_half_marathon + given_counts.female_half_marathon : ℚ) /
  (given_counts.male_half_marathon + given_counts.male_mini_run +
   given_counts.female_half_marathon + given_counts.female_mini_run) = 3 / 5 := by sorry

/-- Chi-square statistic calculation --/
def chi_square (c : StudentCounts) : ℚ :=
  let n := c.male_half_marathon + c.male_mini_run + c.female_half_marathon + c.female_mini_run
  let ad := c.male_half_marathon * c.female_mini_run
  let bc := c.male_mini_run * c.female_half_marathon
  n * (ad - bc)^2 / ((c.male_half_marathon + c.male_mini_run) *
                     (c.female_half_marathon + c.female_mini_run) *
                     (c.male_half_marathon + c.female_half_marathon) *
                     (c.male_mini_run + c.female_mini_run))

/-- Theorem stating that the chi-square statistic is less than the critical value --/
theorem no_significant_relation :
  chi_square given_counts < 2706 / 1000 := by sorry

end NUMINAMATH_CALUDE_female_half_marathon_count_half_marathon_probability_no_significant_relation_l1527_152733


namespace NUMINAMATH_CALUDE_determinant_scaling_l1527_152741

theorem determinant_scaling (a b c d : ℝ) :
  Matrix.det !![a, b; c, d] = 5 →
  Matrix.det !![3*a, 3*b; 2*c, 2*d] = 30 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l1527_152741


namespace NUMINAMATH_CALUDE_cream_strawberry_prices_l1527_152752

/-- Represents the price of a box of cream strawberries in yuan -/
@[ext] structure StrawberryPrice where
  price : ℚ
  price_positive : price > 0

/-- The problem of finding cream strawberry prices -/
theorem cream_strawberry_prices 
  (price_A price_B : StrawberryPrice)
  (price_difference : price_A.price = price_B.price + 10)
  (quantity_equality : 800 / price_A.price = 600 / price_B.price) :
  price_A.price = 40 ∧ price_B.price = 30 := by
  sorry

end NUMINAMATH_CALUDE_cream_strawberry_prices_l1527_152752


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l1527_152794

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l1527_152794


namespace NUMINAMATH_CALUDE_salt_solution_volume_l1527_152769

/-- Proves that given a solution with an initial salt concentration of 10%,
    if adding 18 gallons of water reduces the salt concentration to 8%,
    then the initial volume of the solution must be 72 gallons. -/
theorem salt_solution_volume 
  (initial_concentration : ℝ) 
  (final_concentration : ℝ) 
  (water_added : ℝ) 
  (initial_volume : ℝ) :
  initial_concentration = 0.10 →
  final_concentration = 0.08 →
  water_added = 18 →
  initial_concentration * initial_volume = 
    final_concentration * (initial_volume + water_added) →
  initial_volume = 72 :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_volume_l1527_152769


namespace NUMINAMATH_CALUDE_negation_of_implication_l1527_152721

theorem negation_of_implication (a b : ℝ) :
  ¬(ab = 0 → a = 0 ∨ b = 0) ↔ (ab = 0 → a ≠ 0 ∧ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1527_152721


namespace NUMINAMATH_CALUDE_train_average_speed_l1527_152739

/-- Given a train's travel data, prove its average speed -/
theorem train_average_speed : 
  ∀ (d1 d2 t1 t2 : ℝ),
  d1 = 290 ∧ d2 = 400 ∧ t1 = 4.5 ∧ t2 = 5.5 →
  (d1 + d2) / (t1 + t2) = 69 := by
sorry

end NUMINAMATH_CALUDE_train_average_speed_l1527_152739


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_20_l1527_152725

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 20 → n ≤ 9920 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_20_l1527_152725


namespace NUMINAMATH_CALUDE_valid_license_plates_l1527_152790

/-- The number of valid English letters for the license plate. --/
def validLetters : Nat := 24

/-- The number of positions to choose from for placing the letters. --/
def positionsForLetters : Nat := 4

/-- The number of letter positions to fill. --/
def letterPositions : Nat := 2

/-- The number of digit positions to fill. --/
def digitPositions : Nat := 3

/-- The number of possible digits (0-9). --/
def possibleDigits : Nat := 10

/-- Calculates the number of ways to choose k items from n items. --/
def choose (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of valid license plate combinations. --/
def totalCombinations : Nat :=
  choose positionsForLetters letterPositions * 
  validLetters ^ letterPositions * 
  possibleDigits ^ digitPositions

/-- Theorem stating that the total number of valid license plate combinations is 3,456,000. --/
theorem valid_license_plates : totalCombinations = 3456000 := by
  sorry

end NUMINAMATH_CALUDE_valid_license_plates_l1527_152790


namespace NUMINAMATH_CALUDE_digit_1234_is_4_l1527_152777

def decimal_sequence : ℕ → ℕ
  | 0 => 0  -- represents the decimal point
  | n+1 => 
    let k := (n-1) / 3 + 100
    if k ≤ 500 then
      match (n-1) % 3 with
      | 0 => k / 100
      | 1 => (k / 10) % 10
      | _ => k % 10
    else 0

theorem digit_1234_is_4 : decimal_sequence 1234 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_1234_is_4_l1527_152777


namespace NUMINAMATH_CALUDE_sales_amount_is_194_l1527_152703

/-- Represents the sales data for a stationery store --/
structure SalesData where
  eraser_price : ℝ
  regular_price : ℝ
  short_price : ℝ
  eraser_sold : ℕ
  regular_sold : ℕ
  short_sold : ℕ

/-- Calculates the total sales amount --/
def total_sales (data : SalesData) : ℝ :=
  data.eraser_price * data.eraser_sold +
  data.regular_price * data.regular_sold +
  data.short_price * data.short_sold

/-- Theorem stating that the total sales amount is $194 --/
theorem sales_amount_is_194 (data : SalesData) 
  (h1 : data.eraser_price = 0.8)
  (h2 : data.regular_price = 0.5)
  (h3 : data.short_price = 0.4)
  (h4 : data.eraser_sold = 200)
  (h5 : data.regular_sold = 40)
  (h6 : data.short_sold = 35) :
  total_sales data = 194 := by
  sorry

end NUMINAMATH_CALUDE_sales_amount_is_194_l1527_152703


namespace NUMINAMATH_CALUDE_expression_evaluation_l1527_152711

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y^2) :
  (x - 1 / x^2) * (y + 2 / y) = 2 * x^(5/2) - 1 / x :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1527_152711


namespace NUMINAMATH_CALUDE_cost_of_apples_and_bananas_l1527_152714

/-- The cost of apples in dollars per pound -/
def apple_cost : ℚ := 3 / 3

/-- The cost of bananas in dollars per pound -/
def banana_cost : ℚ := 2 / 2

/-- The total cost of apples and bananas -/
def total_cost (apple_pounds banana_pounds : ℚ) : ℚ :=
  apple_pounds * apple_cost + banana_pounds * banana_cost

theorem cost_of_apples_and_bananas :
  total_cost 9 6 = 15 := by sorry

end NUMINAMATH_CALUDE_cost_of_apples_and_bananas_l1527_152714


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1527_152798

theorem arithmetic_equality : 1357 + 3571 + 5713 - 7135 = 3506 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1527_152798


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1527_152724

theorem right_triangle_segment_ratio :
  ∀ (a b c r s : ℝ),
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  a / b = 2 / 5 →    -- Given ratio of sides
  r + s = c →        -- Perpendicular divides hypotenuse
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  r / s = 4 / 25 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1527_152724


namespace NUMINAMATH_CALUDE_kola_age_is_16_l1527_152765

/-- Kola's current age -/
def kola_age : ℕ := sorry

/-- Ola's current age -/
def ola_age : ℕ := sorry

/-- Kola's age is twice Ola's age when Kola was Ola's current age -/
axiom condition1 : kola_age = 2 * (ola_age - (kola_age - ola_age))

/-- Sum of their ages when Ola reaches Kola's current age is 36 -/
axiom condition2 : kola_age + (kola_age + (kola_age - ola_age)) = 36

/-- Theorem stating Kola's current age is 16 -/
theorem kola_age_is_16 : kola_age = 16 := by sorry

end NUMINAMATH_CALUDE_kola_age_is_16_l1527_152765


namespace NUMINAMATH_CALUDE_third_side_of_similar_altitude_triangle_l1527_152750

/-- A triangle with sides a, b, and c, where the triangle is similar to the triangle formed by its altitudes. -/
structure SimilarAltitudeTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  similar_to_altitude : a * b * c = 2 * (a^2 + b^2 + c^2)

/-- Theorem: In a triangle similar to its altitude triangle with two sides 9 and 4, the third side is 6. -/
theorem third_side_of_similar_altitude_triangle :
  ∀ (t : SimilarAltitudeTriangle), t.a = 9 → t.b = 4 → t.c = 6 := by
  sorry

#check third_side_of_similar_altitude_triangle

end NUMINAMATH_CALUDE_third_side_of_similar_altitude_triangle_l1527_152750


namespace NUMINAMATH_CALUDE_complex_on_line_l1527_152778

/-- Given a complex number z = (2a-i)/i that corresponds to a point on the line x-y=0 in the complex plane, prove that a = 1/2 --/
theorem complex_on_line (a : ℝ) : 
  let z : ℂ := (2*a - Complex.I) / Complex.I
  (z.re - z.im = 0) → a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_complex_on_line_l1527_152778


namespace NUMINAMATH_CALUDE_rogers_pennies_l1527_152716

/-- The number of pennies Roger collected initially -/
def pennies_collected : ℕ := sorry

/-- The number of nickels Roger collected -/
def nickels : ℕ := 36

/-- The number of dimes Roger collected -/
def dimes : ℕ := 15

/-- The number of coins Roger donated -/
def coins_donated : ℕ := 66

/-- The number of coins Roger had left after donating -/
def coins_left : ℕ := 27

/-- The total number of coins Roger had initially -/
def total_coins : ℕ := coins_donated + coins_left

theorem rogers_pennies :
  pennies_collected = total_coins - (nickels + dimes) :=
by sorry

end NUMINAMATH_CALUDE_rogers_pennies_l1527_152716


namespace NUMINAMATH_CALUDE_complement_of_union_is_four_l1527_152797

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_is_four :
  (U \ (A ∪ B)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_is_four_l1527_152797


namespace NUMINAMATH_CALUDE_sum_of_three_polynomials_no_roots_l1527_152768

/-- Given three quadratic polynomials, if the sum of any two has no roots, 
    then the sum of all three has no roots. -/
theorem sum_of_three_polynomials_no_roots 
  (a b c d e f : ℝ) 
  (h1 : ∀ x, (2*x^2 + (a + c)*x + (b + d)) ≠ 0)
  (h2 : ∀ x, (2*x^2 + (c + e)*x + (d + f)) ≠ 0)
  (h3 : ∀ x, (2*x^2 + (e + a)*x + (f + b)) ≠ 0) :
  ∀ x, (3*x^2 + (a + c + e)*x + (b + d + f)) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_polynomials_no_roots_l1527_152768


namespace NUMINAMATH_CALUDE_ahsme_unanswered_questions_l1527_152731

/-- Represents the scoring system for AHSME -/
structure ScoringSystem where
  initial : ℕ
  correct : ℕ
  wrong : ℤ
  unanswered : ℕ

/-- Calculates the score based on the given scoring system and number of questions -/
def calculate_score (system : ScoringSystem) (correct wrong unanswered : ℕ) : ℤ :=
  system.initial + system.correct * correct + system.wrong * wrong + system.unanswered * unanswered

theorem ahsme_unanswered_questions 
  (new_system : ScoringSystem)
  (old_system : ScoringSystem)
  (total_questions : ℕ)
  (new_score : ℕ)
  (old_score : ℕ)
  (h_new_system : new_system = ⟨0, 5, 0, 2⟩)
  (h_old_system : old_system = ⟨30, 4, -1, 0⟩)
  (h_total_questions : total_questions = 30)
  (h_new_score : new_score = 93)
  (h_old_score : old_score = 84) :
  ∃ (correct wrong unanswered : ℕ),
    correct + wrong + unanswered = total_questions ∧
    calculate_score new_system correct wrong unanswered = new_score ∧
    calculate_score old_system correct wrong unanswered = old_score ∧
    unanswered = 9 :=
by sorry


end NUMINAMATH_CALUDE_ahsme_unanswered_questions_l1527_152731


namespace NUMINAMATH_CALUDE_max_ab_value_l1527_152779

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := x^2 - 2*x + 2
def g (a b x : ℝ) : ℝ := -x^2 + a*x + b

-- State the theorem
theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x₀ : ℝ, f x₀ = g a b x₀ ∧ 
    (2*x₀ - 2) * (-2*x₀ + a) = -1) →
  ab ≤ 25/16 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀*b₀ = 25/16 :=
by
  sorry

end NUMINAMATH_CALUDE_max_ab_value_l1527_152779


namespace NUMINAMATH_CALUDE_unique_n_value_l1527_152734

theorem unique_n_value : ∃! n : ℕ, 
  50 ≤ n ∧ n ≤ 120 ∧ 
  ∃ k : ℕ, n = 8 * k ∧
  n % 7 = 5 ∧
  n % 6 = 3 ∧
  n = 208 := by
sorry

end NUMINAMATH_CALUDE_unique_n_value_l1527_152734


namespace NUMINAMATH_CALUDE_exam_mean_score_l1527_152753

/-- Given an exam where a score of 42 is 5 standard deviations below the mean
    and a score of 67 is 2.5 standard deviations above the mean,
    prove that the mean score is 440/7.5 -/
theorem exam_mean_score (μ σ : ℝ) 
  (h1 : 42 = μ - 5 * σ)
  (h2 : 67 = μ + 2.5 * σ) : 
  μ = 440 / 7.5 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l1527_152753


namespace NUMINAMATH_CALUDE_min_value_theorem_l1527_152742

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_value_theorem (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 7 = a 6 + 2 * a 5 →
  (∃ m n : ℕ, a m * a n = 8 * (a 1)^2) →
  (∃ m n : ℕ, 1 / m + 4 / n = 11 / 6 ∧
    ∀ k l : ℕ, 1 / k + 4 / l ≥ 11 / 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1527_152742


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1527_152773

theorem sum_of_cubes (a b c : ℝ) 
  (sum_eq : a + b + c = 7)
  (sum_products_eq : a * b + a * c + b * c = 11)
  (product_eq : a * b * c = -18) :
  a^3 + b^3 + c^3 = 151 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1527_152773


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l1527_152722

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 + 2*x - 3 = 0
def equation2 (x : ℝ) : Prop := 2*x^2 + 4*x - 3 = 0

-- Theorem for the first equation
theorem solution_equation1 : 
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -3 ∧ equation1 x₁ ∧ equation1 x₂) ∧
  (∀ x : ℝ, equation1 x → x = 1 ∨ x = -3) :=
sorry

-- Theorem for the second equation
theorem solution_equation2 : 
  (∃ x₁ x₂ : ℝ, x₁ = (-2 + Real.sqrt 10) / 2 ∧ x₂ = (-2 - Real.sqrt 10) / 2 ∧ equation2 x₁ ∧ equation2 x₂) ∧
  (∀ x : ℝ, equation2 x → x = (-2 + Real.sqrt 10) / 2 ∨ x = (-2 - Real.sqrt 10) / 2) :=
sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l1527_152722


namespace NUMINAMATH_CALUDE_special_triangle_ratio_constant_l1527_152791

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = c^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = a^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = b^2

-- Define the property AC^2 + BC^2 = 2 AB^2
def SpecialTriangleProperty (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 + (C.1 - B.1)^2 + (C.2 - B.2)^2 = 
  2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define point M as the midpoint of AB
def Midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define the angle equality ∠ACD = ∠BCD
def EqualAngles (A B C D : ℝ × ℝ) : Prop :=
  let v1 := (A.1 - C.1, A.2 - C.2)
  let v2 := (D.1 - C.1, D.2 - C.2)
  let v3 := (B.1 - C.1, B.2 - C.2)
  (v1.1 * v2.1 + v1.2 * v2.2)^2 / ((v1.1^2 + v1.2^2) * (v2.1^2 + v2.2^2)) =
  (v3.1 * v2.1 + v3.2 * v2.2)^2 / ((v3.1^2 + v3.2^2) * (v2.1^2 + v2.2^2))

-- Define D as the incenter of triangle CEM
def Incenter (D C E M : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = r^2 ∧
  (D.1 - E.1)^2 + (D.2 - E.2)^2 = r^2 ∧
  (D.1 - M.1)^2 + (D.2 - M.2)^2 = r^2

-- Main theorem
theorem special_triangle_ratio_constant 
  (A B C M D E : ℝ × ℝ) :
  Triangle A B C →
  SpecialTriangleProperty A B C →
  Midpoint M A B →
  EqualAngles A B C D →
  Incenter D C E M →
  (E.1 - M.1)^2 + (E.2 - M.2)^2 = 
  (1/9) * ((M.1 - C.1)^2 + (M.2 - C.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_ratio_constant_l1527_152791


namespace NUMINAMATH_CALUDE_b_value_l1527_152766

-- Define the functions p and q
def p (x : ℝ) : ℝ := 2 * x - 7
def q (x b : ℝ) : ℝ := 3 * x - b

-- State the theorem
theorem b_value (b : ℝ) : p (q 3 b) = 3 → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l1527_152766


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1527_152715

-- Define the start point of the line segment
def start_point : ℝ × ℝ := (1, 3)

-- Define the end point of the line segment
def end_point (x : ℝ) : ℝ × ℝ := (x, -4)

-- Define the length of the line segment
def segment_length : ℝ := 15

-- Theorem statement
theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  (end_point x).1 - (start_point.1) = -4 * Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1527_152715


namespace NUMINAMATH_CALUDE_total_animals_is_130_l1527_152761

/-- The total number of animals seen throughout the day -/
def total_animals (initial_beavers initial_chipmunks : ℕ) : ℕ :=
  let morning_total := initial_beavers + initial_chipmunks
  let afternoon_beavers := 2 * initial_beavers
  let afternoon_chipmunks := initial_chipmunks - 10
  morning_total + afternoon_beavers + afternoon_chipmunks

/-- Theorem stating the total number of animals seen is 130 -/
theorem total_animals_is_130 :
  total_animals 20 40 = 130 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_is_130_l1527_152761


namespace NUMINAMATH_CALUDE_tonys_rope_length_l1527_152727

/-- Represents a rope with its length and knot loss -/
structure Rope where
  length : Float
  knotLoss : Float

/-- Calculates the total length of ropes after tying them together -/
def totalRopeLength (ropes : List Rope) : Float :=
  let totalOriginalLength := ropes.map (·.length) |>.sum
  let totalLossFromKnots := ropes.map (·.knotLoss) |>.sum
  totalOriginalLength - totalLossFromKnots

/-- Theorem stating the total length of Tony's ropes after tying -/
theorem tonys_rope_length :
  let ropes : List Rope := [
    { length := 8, knotLoss := 1.2 },
    { length := 20, knotLoss := 1.5 },
    { length := 2, knotLoss := 1 },
    { length := 2, knotLoss := 1 },
    { length := 2, knotLoss := 1 },
    { length := 7, knotLoss := 0.8 },
    { length := 5, knotLoss := 1.2 },
    { length := 5, knotLoss := 1.2 }
  ]
  totalRopeLength ropes = 42.1 := by
  sorry

end NUMINAMATH_CALUDE_tonys_rope_length_l1527_152727


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1527_152756

/-- For any real number m, the line mx + y - 1 + 2m = 0 passes through the point (-2, 1) -/
theorem fixed_point_on_line (m : ℝ) : m * (-2) + 1 - 1 + 2 * m = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1527_152756


namespace NUMINAMATH_CALUDE_larger_integer_value_l1527_152707

theorem larger_integer_value (a b : ℕ+) 
  (h1 : (a : ℚ) / (b : ℚ) = 7 / 3) 
  (h2 : (a : ℕ) * b = 189) : 
  a = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l1527_152707


namespace NUMINAMATH_CALUDE_aldens_nephews_l1527_152719

theorem aldens_nephews (alden_now alden_past vihaan : ℕ) : 
  alden_now = 2 * alden_past →
  vihaan = alden_now + 60 →
  alden_now + vihaan = 260 →
  alden_past = 50 := by
sorry

end NUMINAMATH_CALUDE_aldens_nephews_l1527_152719


namespace NUMINAMATH_CALUDE_total_weight_of_good_fruits_l1527_152795

/-- Calculates the total weight in kilograms of fruits in good condition --/
def totalWeightOfGoodFruits (
  oranges bananas apples avocados grapes pineapples : ℕ
) (
  rottenOrangesPercent rottenBananasPercent rottenApplesPercent
  rottenAvocadosPercent rottenGrapesPercent rottenPineapplesPercent : ℚ
) (
  orangeWeight bananaWeight appleWeight avocadoWeight grapeWeight pineappleWeight : ℚ
) : ℚ :=
  let goodOranges := oranges - (oranges * rottenOrangesPercent).floor
  let goodBananas := bananas - (bananas * rottenBananasPercent).floor
  let goodApples := apples - (apples * rottenApplesPercent).floor
  let goodAvocados := avocados - (avocados * rottenAvocadosPercent).floor
  let goodGrapes := grapes - (grapes * rottenGrapesPercent).floor
  let goodPineapples := pineapples - (pineapples * rottenPineapplesPercent).floor

  (goodOranges * orangeWeight + goodBananas * bananaWeight +
   goodApples * appleWeight + goodAvocados * avocadoWeight +
   goodGrapes * grapeWeight + goodPineapples * pineappleWeight) / 1000

/-- The total weight of fruits in good condition is 204.585kg --/
theorem total_weight_of_good_fruits :
  totalWeightOfGoodFruits
    600 400 300 200 100 50
    (15/100) (5/100) (8/100) (10/100) (3/100) (20/100)
    150 120 100 80 5 1000 = 204585/1000 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_good_fruits_l1527_152795


namespace NUMINAMATH_CALUDE_unique_seven_numbers_sum_100_l1527_152701

theorem unique_seven_numbers_sum_100 (a₄ : ℕ) : 
  ∃! (a₁ a₂ a₃ a₅ a₆ a₇ : ℕ), 
    a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ ∧ a₅ < a₆ ∧ a₆ < a₇ ∧
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 100 :=
by sorry

end NUMINAMATH_CALUDE_unique_seven_numbers_sum_100_l1527_152701


namespace NUMINAMATH_CALUDE_periodic_function_property_l1527_152792

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β), if f(4) = 3, then f(2017) = -3 -/
theorem periodic_function_property (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 4 = 3 → f 2017 = -3 := by
  sorry


end NUMINAMATH_CALUDE_periodic_function_property_l1527_152792


namespace NUMINAMATH_CALUDE_sqrt_three_square_form_l1527_152759

theorem sqrt_three_square_form (a b m n : ℕ+) :
  a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 →
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n :=
by sorry

end NUMINAMATH_CALUDE_sqrt_three_square_form_l1527_152759


namespace NUMINAMATH_CALUDE_child_tickets_sold_l1527_152738

theorem child_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_receipts ∧
    child_tickets = 90 := by
  sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l1527_152738


namespace NUMINAMATH_CALUDE_james_field_goal_value_l1527_152758

/-- Represents the score of a basketball game -/
structure BasketballScore where
  fieldGoals : ℕ
  fieldGoalValue : ℕ
  twoPointers : ℕ
  totalScore : ℕ

/-- Theorem stating that given the conditions of James' game, each field goal is worth 3 points -/
theorem james_field_goal_value (score : BasketballScore) 
  (h1 : score.fieldGoals = 13)
  (h2 : score.twoPointers = 20)
  (h3 : score.totalScore = 79)
  (h4 : score.totalScore = score.fieldGoals * score.fieldGoalValue + score.twoPointers * 2) :
  score.fieldGoalValue = 3 := by
  sorry


end NUMINAMATH_CALUDE_james_field_goal_value_l1527_152758


namespace NUMINAMATH_CALUDE_value_of_3x_plus_y_l1527_152708

theorem value_of_3x_plus_y (x y : ℝ) (h : (2*x + y)^3 + x^3 + 3*x + y = 0) : 3*x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_value_of_3x_plus_y_l1527_152708


namespace NUMINAMATH_CALUDE_expression_equals_sum_l1527_152760

theorem expression_equals_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_sum_l1527_152760


namespace NUMINAMATH_CALUDE_adjacent_vertices_probability_l1527_152771

/-- A decagon is a polygon with 10 vertices -/
def Decagon : Type := Unit

/-- The number of vertices in a decagon -/
def num_vertices : ℕ := 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def adjacent_vertices : ℕ := 2

/-- The probability of choosing two adjacent vertices when randomly selecting 2 distinct vertices from a decagon -/
theorem adjacent_vertices_probability (d : Decagon) : ℚ :=
  2 / 9

/-- Proof of the theorem -/
lemma adjacent_vertices_probability_proof (d : Decagon) : 
  adjacent_vertices_probability d = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_vertices_probability_l1527_152771


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_equality_l1527_152704

/-- The minimum value of 1/m + 3/n given the conditions -/
theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hmn : m * n > 0) (h_line : m * 2 + n * 2 = 1) : 
  (1 / m + 3 / n : ℝ) ≥ 5 + 2 * Real.sqrt 6 := by
  sorry

/-- The conditions for equality in the minimum value theorem -/
theorem min_value_equality (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hmn : m * n > 0) (h_line : m * 2 + n * 2 = 1) : 
  (1 / m + 3 / n : ℝ) = 5 + 2 * Real.sqrt 6 ↔ m = Real.sqrt 3 / 3 ∧ n = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_equality_l1527_152704


namespace NUMINAMATH_CALUDE_dance_troupe_size_l1527_152762

/-- Represents the number of performers in a dance troupe with various skills -/
structure DanceTroupe where
  singers : ℕ
  dancers : ℕ
  instrumentalists : ℕ
  singer_dancers : ℕ
  singer_instrumentalists : ℕ
  dancer_instrumentalists : ℕ
  all_skilled : ℕ

/-- The conditions of the dance troupe problem -/
def dance_troupe_conditions (dt : DanceTroupe) : Prop :=
  dt.singers = 2 ∧
  dt.dancers = 26 ∧
  dt.instrumentalists = 22 ∧
  dt.singer_dancers = 8 ∧
  dt.singer_instrumentalists = 10 ∧
  dt.dancer_instrumentalists = 11 ∧
  (dt.singers + dt.dancers + dt.instrumentalists
    - dt.singer_dancers - dt.singer_instrumentalists - dt.dancer_instrumentalists + dt.all_skilled
    - (dt.singer_dancers + dt.singer_instrumentalists + dt.dancer_instrumentalists - 2 * dt.all_skilled)) =
  (dt.singer_dancers + dt.singer_instrumentalists + dt.dancer_instrumentalists - 2 * dt.all_skilled)

/-- The total number of performers in the dance troupe -/
def total_performers (dt : DanceTroupe) : ℕ :=
  dt.singers + dt.dancers + dt.instrumentalists
  - dt.singer_dancers - dt.singer_instrumentalists - dt.dancer_instrumentalists
  + dt.all_skilled

/-- Theorem stating that the total number of performers is 46 -/
theorem dance_troupe_size (dt : DanceTroupe) :
  dance_troupe_conditions dt → total_performers dt = 46 := by
  sorry


end NUMINAMATH_CALUDE_dance_troupe_size_l1527_152762


namespace NUMINAMATH_CALUDE_student_average_grade_l1527_152789

theorem student_average_grade
  (courses_last_year : ℕ)
  (courses_year_before : ℕ)
  (avg_grade_year_before : ℚ)
  (avg_grade_two_years : ℚ)
  (h1 : courses_last_year = 6)
  (h2 : courses_year_before = 5)
  (h3 : avg_grade_year_before = 70)
  (h4 : avg_grade_two_years = 86)
  : ∃ x : ℚ, x = 596 / 6 ∧ 
    (courses_year_before * avg_grade_year_before + courses_last_year * x) / 
    (courses_year_before + courses_last_year) = avg_grade_two_years :=
by sorry

end NUMINAMATH_CALUDE_student_average_grade_l1527_152789


namespace NUMINAMATH_CALUDE_unique_prime_sevens_l1527_152757

def A (n : ℕ+) : ℕ := 1 + 7 * (10^n.val - 1) / 9

def B (n : ℕ+) : ℕ := 3 + 7 * (10^n.val - 1) / 9

theorem unique_prime_sevens : 
  ∃! (n : ℕ+), Nat.Prime (A n) ∧ Nat.Prime (B n) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sevens_l1527_152757


namespace NUMINAMATH_CALUDE_beka_miles_l1527_152726

/-- The number of miles Jackson flew -/
def jackson_miles : ℕ := 563

/-- The difference in miles between Beka's and Jackson's flights -/
def difference_miles : ℕ := 310

/-- Theorem: Beka flew 873 miles -/
theorem beka_miles : jackson_miles + difference_miles = 873 := by
  sorry

end NUMINAMATH_CALUDE_beka_miles_l1527_152726


namespace NUMINAMATH_CALUDE_pond_depth_l1527_152718

theorem pond_depth (d : ℝ) 
  (h1 : ¬(d ≥ 10))  -- Adam's statement is false
  (h2 : ¬(d ≤ 8))   -- Ben's statement is false
  (h3 : d ≠ 7)      -- Carla's statement is false
  : 8 < d ∧ d < 10 := by
  sorry

end NUMINAMATH_CALUDE_pond_depth_l1527_152718


namespace NUMINAMATH_CALUDE_multiply_fractions_l1527_152749

theorem multiply_fractions : 8 * (1 / 11) * 33 = 24 := by sorry

end NUMINAMATH_CALUDE_multiply_fractions_l1527_152749


namespace NUMINAMATH_CALUDE_diplomats_not_speaking_russian_count_l1527_152774

/-- Represents the diplomats at a summit conference -/
structure DiplomatGroup where
  total : Nat
  latin_speakers : Nat
  neither_latin_nor_russian : Nat
  both_latin_and_russian : Nat

/-- Calculates the number of diplomats who did not speak Russian -/
def diplomats_not_speaking_russian (d : DiplomatGroup) : Nat :=
  d.total - (d.total - d.neither_latin_nor_russian - d.latin_speakers + d.both_latin_and_russian)

/-- Theorem stating the number of diplomats who did not speak Russian -/
theorem diplomats_not_speaking_russian_count :
  ∃ (d : DiplomatGroup),
    d.total = 120 ∧
    d.latin_speakers = 20 ∧
    d.neither_latin_nor_russian = (20 * d.total) / 100 ∧
    d.both_latin_and_russian = (10 * d.total) / 100 ∧
    diplomats_not_speaking_russian d = 20 := by
  sorry

end NUMINAMATH_CALUDE_diplomats_not_speaking_russian_count_l1527_152774


namespace NUMINAMATH_CALUDE_loom_weaving_rate_l1527_152720

/-- The rate at which an industrial loom weaves cloth, given the total amount of cloth woven and the time taken. -/
theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) (h : total_cloth = 26 ∧ total_time = 203.125) :
  total_cloth / total_time = 0.128 := by
sorry

end NUMINAMATH_CALUDE_loom_weaving_rate_l1527_152720


namespace NUMINAMATH_CALUDE_trig_inequality_l1527_152713

theorem trig_inequality (a b : Real) (ha : 0 < a ∧ a < π/2) (hb : 0 < b ∧ b < π/2) :
  (Real.sin a)^3 / Real.sin b + (Real.cos a)^3 / Real.cos b ≥ 1 / Real.cos (a - b) := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l1527_152713


namespace NUMINAMATH_CALUDE_e_integral_greater_than_ln_integral_l1527_152736

theorem e_integral_greater_than_ln_integral : ∫ (x : ℝ) in (0)..(1), Real.exp x > ∫ (x : ℝ) in (1)..(Real.exp 1), 1 / x := by
  sorry

end NUMINAMATH_CALUDE_e_integral_greater_than_ln_integral_l1527_152736


namespace NUMINAMATH_CALUDE_area_of_triangle_hyperbola_triangle_area_l1527_152781

/-- A hyperbola with center at the origin, foci on the x-axis, and eccentricity √2 -/
structure Hyperbola where
  eccentricity : ℝ
  passes_through : ℝ × ℝ
  eccentricity_eq : eccentricity = Real.sqrt 2
  point_on_hyperbola : passes_through = (4, Real.sqrt 10)

/-- A point M on the hyperbola where MF₁ ⟂ MF₂ -/
structure PointM (h : Hyperbola) where
  point : ℝ × ℝ
  on_hyperbola : point ∈ {p : ℝ × ℝ | p.1^2 - p.2^2 = 6}
  perpendicular : ∃ (f₁ f₂ : ℝ × ℝ), f₁.2 = 0 ∧ f₂.2 = 0 ∧ 
    (point.1 - f₁.1) * (point.1 - f₂.1) + point.2 * point.2 = 0

/-- The theorem stating that the area of triangle F₁MF₂ is 6 -/
theorem area_of_triangle (h : Hyperbola) (m : PointM h) : ℝ :=
  6

/-- The main theorem to be proved -/
theorem hyperbola_triangle_area (h : Hyperbola) (m : PointM h) :
  area_of_triangle h m = 6 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_hyperbola_triangle_area_l1527_152781


namespace NUMINAMATH_CALUDE_gcd_triple_characterization_l1527_152740

theorem gcd_triple_characterization (a b c : ℕ+) :
  (Nat.gcd a 20 = b) →
  (Nat.gcd b 15 = c) →
  (Nat.gcd a c = 5) →
  (∃ t : ℕ+, (a = 20 * t ∧ b = 20 ∧ c = 5) ∨
             (a = 20 * t - 10 ∧ b = 10 ∧ c = 5) ∨
             (a = 10 * t - 5 ∧ b = 5 ∧ c = 5)) :=
by sorry

end NUMINAMATH_CALUDE_gcd_triple_characterization_l1527_152740


namespace NUMINAMATH_CALUDE_car_average_speed_l1527_152784

/-- The average speed of a car given its distance traveled in two hours -/
theorem car_average_speed (d1 d2 : ℝ) (h1 : d1 = 98) (h2 : d2 = 60) :
  (d1 + d2) / 2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l1527_152784


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l1527_152712

theorem infinitely_many_solutions
  (a b c k : ℤ)
  (D : ℤ)
  (hD : D = b^2 - 4*a*c)
  (hD_pos : D > 0)
  (hD_nonsquare : ∀ m : ℤ, D ≠ m^2)
  (hk : k ≠ 0)
  (h_solution : ∃ (x₀ y₀ : ℤ), a*x₀^2 + b*x₀*y₀ + c*y₀^2 = k) :
  ∃ (S : Set (ℤ × ℤ)), (Set.Infinite S) ∧ (∀ (x y : ℤ), (x, y) ∈ S → a*x^2 + b*x*y + c*y^2 = k) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l1527_152712


namespace NUMINAMATH_CALUDE_min_width_rectangle_width_satisfies_min_area_min_width_is_four_l1527_152709

/-- The minimum width of a rectangular area with specific constraints -/
theorem min_width_rectangle (w : ℝ) (h1 : w > 0) : 
  w * (w + 20) ≥ 120 → w ≥ 4 := by sorry

/-- The width that satisfies the minimum area requirement -/
theorem width_satisfies_min_area : 
  4 * (4 + 20) ≥ 120 := by sorry

/-- Proof that 4 is the minimum width satisfying the constraints -/
theorem min_width_is_four : 
  ∃ (w : ℝ), w > 0 ∧ w * (w + 20) ≥ 120 ∧ ∀ (x : ℝ), x > 0 ∧ x * (x + 20) ≥ 120 → x ≥ w :=
by
  use 4
  sorry

end NUMINAMATH_CALUDE_min_width_rectangle_width_satisfies_min_area_min_width_is_four_l1527_152709


namespace NUMINAMATH_CALUDE_cylinder_height_ratio_l1527_152763

/-- 
Given two cylinders where:
- The first cylinder has height h and is 7/8 full of water
- The second cylinder has a radius 25% larger than the first
- All water from the first cylinder fills 3/5 of the second cylinder
Prove that the height of the second cylinder is 14/15 of h
-/
theorem cylinder_height_ratio (h : ℝ) (h' : ℝ) : 
  (7/8 : ℝ) * π * r^2 * h = (3/5 : ℝ) * π * (1.25 * r)^2 * h' → 
  h' = (14/15 : ℝ) * h :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_ratio_l1527_152763


namespace NUMINAMATH_CALUDE_square_rectangle_area_ratio_l1527_152717

theorem square_rectangle_area_ratio :
  ∀ (square_perimeter : ℝ) (rect_length rect_width : ℝ),
    square_perimeter = 256 →
    rect_length = 32 →
    rect_width = 64 →
    (square_perimeter / 4)^2 / (rect_length * rect_width) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_ratio_l1527_152717


namespace NUMINAMATH_CALUDE_total_cost_of_sarees_l1527_152782

/-- Calculates the final price of a saree after applying discounts -/
def apply_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (fun p d => p * (1 - d)) price

/-- Converts a price from one currency to INR -/
def convert_to_inr (price : ℝ) (rate : ℝ) : ℝ :=
  price * rate

/-- Applies sales tax to a price -/
def apply_sales_tax (price : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 + tax_rate)

/-- Theorem: The total cost of purchasing three sarees is 39421.08 INR -/
theorem total_cost_of_sarees : 
  let saree1_price : ℝ := 200
  let saree1_discounts : List ℝ := [0.20, 0.15, 0.05]
  let saree1_rate : ℝ := 75

  let saree2_price : ℝ := 150
  let saree2_discounts : List ℝ := [0.10, 0.07]
  let saree2_rate : ℝ := 100

  let saree3_price : ℝ := 180
  let saree3_discounts : List ℝ := [0.12]
  let saree3_rate : ℝ := 90

  let sales_tax : ℝ := 0.08

  let saree1_final := apply_sales_tax (convert_to_inr (apply_discounts saree1_price saree1_discounts) saree1_rate) sales_tax
  let saree2_final := apply_sales_tax (convert_to_inr (apply_discounts saree2_price saree2_discounts) saree2_rate) sales_tax
  let saree3_final := apply_sales_tax (convert_to_inr (apply_discounts saree3_price saree3_discounts) saree3_rate) sales_tax

  saree1_final + saree2_final + saree3_final = 39421.08 :=
by sorry


end NUMINAMATH_CALUDE_total_cost_of_sarees_l1527_152782


namespace NUMINAMATH_CALUDE_triple_base_exponent_l1527_152772

theorem triple_base_exponent (a b x : ℝ) (h1 : b ≠ 0) : 
  (3 * a) ^ (3 * b) = a ^ b * x ^ b → x = 27 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triple_base_exponent_l1527_152772
