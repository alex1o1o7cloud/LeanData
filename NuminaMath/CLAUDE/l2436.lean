import Mathlib

namespace kids_difference_l2436_243606

def kids_monday : ℕ := 6
def kids_wednesday : ℕ := 4

theorem kids_difference : kids_monday - kids_wednesday = 2 := by
  sorry

end kids_difference_l2436_243606


namespace polynomial_value_at_three_l2436_243699

theorem polynomial_value_at_three : 
  let x : ℤ := 3
  (x^5 : ℤ) - 5*x + 7*(x^3) = 417 := by
  sorry

end polynomial_value_at_three_l2436_243699


namespace prime_divisibility_l2436_243644

theorem prime_divisibility (p q : Nat) 
  (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) (hp5 : p > 5) (hq5 : q > 5) :
  (p ∣ (5^q - 2^q) → q ∣ (p - 1)) ∧ ¬(p*q ∣ (5^p - 2^p)*(5^q - 2^q)) := by
  sorry

end prime_divisibility_l2436_243644


namespace ratio_a_to_c_l2436_243624

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 5)
  (hdb : d / b = 2 / 5) :
  a / c = 125 / 16 := by
sorry

end ratio_a_to_c_l2436_243624


namespace object_properties_l2436_243613

-- Define the possible colors
inductive Color
| Red
| Blue
| Green

-- Define the shape property
structure Object where
  color : Color
  isRound : Bool

-- Define the conditions
axiom condition1 (obj : Object) : obj.isRound → (obj.color = Color.Red ∨ obj.color = Color.Blue)
axiom condition2 (obj : Object) : ¬obj.isRound → (obj.color ≠ Color.Red ∧ obj.color ≠ Color.Green)
axiom condition3 (obj : Object) : (obj.color = Color.Blue ∨ obj.color = Color.Green) → obj.isRound

-- Theorem to prove
theorem object_properties (obj : Object) : 
  obj.isRound ∧ (obj.color = Color.Red ∨ obj.color = Color.Blue) :=
by sorry

end object_properties_l2436_243613


namespace fruit_seller_apples_l2436_243660

theorem fruit_seller_apples (initial_apples : ℕ) (remaining_apples : ℕ) : 
  remaining_apples = 420 → 
  (initial_apples : ℚ) * (70 / 100) = remaining_apples → 
  initial_apples = 600 := by
sorry

end fruit_seller_apples_l2436_243660


namespace inequality_multiplication_l2436_243654

theorem inequality_multiplication (m n : ℝ) (h : m > n) : 2 * m > 2 * n := by
  sorry

end inequality_multiplication_l2436_243654


namespace sum_of_squares_l2436_243656

theorem sum_of_squares (a b : ℝ) : (a^2 + b^2) * (a^2 + b^2 + 4) = 12 → a^2 + b^2 = 2 := by
  sorry

end sum_of_squares_l2436_243656


namespace other_items_sales_percentage_l2436_243615

theorem other_items_sales_percentage 
  (total_sales_percentage : ℝ)
  (notebooks_sales_percentage : ℝ)
  (markers_sales_percentage : ℝ)
  (h1 : total_sales_percentage = 100)
  (h2 : notebooks_sales_percentage = 42)
  (h3 : markers_sales_percentage = 21) :
  total_sales_percentage - (notebooks_sales_percentage + markers_sales_percentage) = 37 := by
  sorry

end other_items_sales_percentage_l2436_243615


namespace abc_inequality_l2436_243690

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 := by
  sorry

end abc_inequality_l2436_243690


namespace instantaneous_velocity_at_3_l2436_243651

/-- The position function of a particle -/
def position (t : ℝ) : ℝ := t^3 - 2*t

/-- The velocity function of a particle -/
def velocity (t : ℝ) : ℝ := 3*t^2 - 2

theorem instantaneous_velocity_at_3 : velocity 3 = 25 := by
  sorry

end instantaneous_velocity_at_3_l2436_243651


namespace peters_children_l2436_243625

theorem peters_children (initial_savings : ℕ) (addition : ℕ) (num_children : ℕ) : 
  initial_savings = 642986 →
  addition = 642987 →
  (initial_savings + addition) % num_children = 0 →
  num_children = 642987 := by
sorry

end peters_children_l2436_243625


namespace cricketer_matches_l2436_243650

theorem cricketer_matches (total_average : ℝ) (first_8_average : ℝ) (last_4_average : ℝ)
  (h1 : total_average = 48)
  (h2 : first_8_average = 40)
  (h3 : last_4_average = 64) :
  ∃ (n : ℕ), n * total_average = 8 * first_8_average + 4 * last_4_average ∧ n = 12 := by
  sorry

end cricketer_matches_l2436_243650


namespace least_n_divisibility_l2436_243678

theorem least_n_divisibility (a b : ℕ+) : 
  (∃ (n : ℕ+), n = 1296 ∧ 
    (∀ (a b : ℕ+), 36 ∣ (a + b) → n ∣ (a * b) → 36 ∣ a ∧ 36 ∣ b) ∧
    (∀ (m : ℕ+), m < n → 
      ∃ (x y : ℕ+), 36 ∣ (x + y) ∧ m ∣ (x * y) ∧ (¬(36 ∣ x) ∨ ¬(36 ∣ y)))) :=
by
  sorry

#check least_n_divisibility

end least_n_divisibility_l2436_243678


namespace perfect_square_iff_even_exponents_kth_power_iff_divisible_exponents_l2436_243608

/-- A natural number is a perfect square if and only if each prime in its prime factorization appears an even number of times. -/
theorem perfect_square_iff_even_exponents (n : ℕ) :
  (∃ m : ℕ, n = m ^ 2) ↔ (∀ p : ℕ, Prime p → ∃ k : ℕ, n.factorization p = 2 * k) :=
sorry

/-- A natural number is a k-th power if and only if each prime in its prime factorization appears a number of times divisible by k. -/
theorem kth_power_iff_divisible_exponents (n k : ℕ) (hk : k > 0) :
  (∃ m : ℕ, n = m ^ k) ↔ (∀ p : ℕ, Prime p → ∃ l : ℕ, n.factorization p = k * l) :=
sorry

end perfect_square_iff_even_exponents_kth_power_iff_divisible_exponents_l2436_243608


namespace at_least_two_heads_probability_l2436_243635

def coin_toss_probability : ℕ → ℕ → ℚ
  | n, k => (Nat.choose n k : ℚ) * (1/2)^k * (1/2)^(n-k)

theorem at_least_two_heads_probability :
  coin_toss_probability 4 2 + coin_toss_probability 4 3 + coin_toss_probability 4 4 = 11/16 := by
  sorry

end at_least_two_heads_probability_l2436_243635


namespace discount_calculation_l2436_243631

theorem discount_calculation (original_price : ℝ) (original_price_pos : original_price > 0) :
  let first_discount := 0.3
  let second_discount := 0.25
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price / original_price = 0.525 := by
sorry

end discount_calculation_l2436_243631


namespace volunteer_arrangements_l2436_243676

theorem volunteer_arrangements (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 7 ∧ k = 3 ∧ m = 4 →
  (n.choose k) * (m.choose k) = 140 :=
by sorry

end volunteer_arrangements_l2436_243676


namespace ratio_x_to_y_l2436_243619

theorem ratio_x_to_y (x y : ℝ) (h : (12*x - 5*y) / (15*x - 3*y) = 4/7) : x/y = 23/24 := by
  sorry

end ratio_x_to_y_l2436_243619


namespace zeros_of_h_l2436_243664

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def g (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 - 1

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 2 * f a x - g a x

theorem zeros_of_h (a : ℝ) (x₁ x₂ : ℝ) :
  (∃ (x₁ x₂ : ℝ), h a x₁ = 0 ∧ h a x₂ = 0 ∧ x₁ ≠ x₂) →
  -1 < a ∧ a < 0 ∧ x₁ + x₂ > 2 / (a + 1) := by
  sorry

end zeros_of_h_l2436_243664


namespace line_parallel_to_plane_l2436_243696

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the non-parallel relation for lines and planes
variable (not_parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m n : Line) (α : Plane) :
  parallel_line m n → 
  not_parallel_line_plane n α → 
  not_parallel_line_plane m α → 
  parallel_line_plane m α :=
sorry

end line_parallel_to_plane_l2436_243696


namespace exists_empty_subsquare_l2436_243642

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in a 2D plane -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- A function to check if a point is inside a square -/
def isPointInSquare (p : Point) (s : Square) : Prop :=
  s.bottomLeft.x ≤ p.x ∧ p.x < s.bottomLeft.x + s.sideLength ∧
  s.bottomLeft.y ≤ p.y ∧ p.y < s.bottomLeft.y + s.sideLength

/-- The main theorem -/
theorem exists_empty_subsquare 
  (bigSquare : Square) 
  (points : Finset Point) 
  (h1 : bigSquare.sideLength = 4) 
  (h2 : points.card = 15) : 
  ∃ (smallSquare : Square), 
    smallSquare.sideLength = 1 ∧ 
    (∀ (p : Point), p ∈ points → ¬ isPointInSquare p smallSquare) :=
sorry

end exists_empty_subsquare_l2436_243642


namespace investors_in_both_l2436_243665

theorem investors_in_both (total : ℕ) (equities : ℕ) (both : ℕ)
  (h_total : total = 100)
  (h_equities : equities = 80)
  (h_both : both = 40)
  (h_invest : ∀ i, i ∈ Finset.range total → 
    (i ∈ Finset.range equities ∨ i ∈ Finset.range (total - equities + both)))
  : both = 40 := by
  sorry

end investors_in_both_l2436_243665


namespace fraction_simplification_l2436_243621

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hab : a ≠ b) :
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) := by
  sorry

end fraction_simplification_l2436_243621


namespace one_plane_through_line_parallel_to_skew_line_l2436_243692

-- Define the concept of a line in 3D space
structure Line3D where
  -- You might define a line using a point and a direction vector
  -- But for simplicity, we'll just declare it as an opaque type
  dummy : Unit

-- Define the concept of a plane in 3D space
structure Plane3D where
  -- Similar to Line3D, we'll keep this as an opaque type for simplicity
  dummy : Unit

-- Define what it means for two lines to be skew
def are_skew (a b : Line3D) : Prop :=
  -- Two lines are skew if they are neither intersecting nor parallel
  sorry

-- Define what it means for a plane to contain a line
def plane_contains_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

-- Define what it means for a plane to be parallel to a line
def plane_parallel_to_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

-- The main theorem
theorem one_plane_through_line_parallel_to_skew_line 
  (a b : Line3D) (h : are_skew a b) : 
  ∃! p : Plane3D, plane_contains_line p a ∧ plane_parallel_to_line p b :=
sorry

end one_plane_through_line_parallel_to_skew_line_l2436_243692


namespace parabola_circle_tangency_l2436_243668

/-- Given a parabola y² = 8x and a circle x² + y² + 6x + m = 0, 
    if the directrix of the parabola is tangent to the circle, then m = 8 -/
theorem parabola_circle_tangency (m : ℝ) : 
  (∀ y : ℝ, (∃! x : ℝ, x = -2 ∧ x^2 + y^2 + 6*x + m = 0)) → m = 8 := by
  sorry

end parabola_circle_tangency_l2436_243668


namespace books_from_second_shop_is_35_l2436_243640

/-- The number of books Rahim bought from the second shop -/
def books_from_second_shop : ℕ := sorry

/-- The total amount spent on books -/
def total_spent : ℕ := 6500 + 2000

/-- The total number of books bought -/
def total_books : ℕ := 65 + books_from_second_shop

/-- The average price per book -/
def average_price : ℚ := 85

theorem books_from_second_shop_is_35 :
  books_from_second_shop = 35 ∧
  65 * 100 = 6500 ∧
  books_from_second_shop * average_price = 2000 ∧
  average_price * total_books = total_spent := by sorry

end books_from_second_shop_is_35_l2436_243640


namespace smallest_a_for_nonprime_cube_sum_l2436_243695

theorem smallest_a_for_nonprime_cube_sum :
  ∃ (a : ℕ), a > 0 ∧ (∀ (x : ℤ), ¬ Prime (x^3 + a^3)) ∧
  (∀ (b : ℕ), b > 0 ∧ b < a → ∃ (y : ℤ), Prime (y^3 + b^3)) :=
by
  -- The proof goes here
  sorry

end smallest_a_for_nonprime_cube_sum_l2436_243695


namespace coin_problem_l2436_243693

/-- Represents the number of coins of each type in the bag -/
def num_coins : ℕ := sorry

/-- Represents the total value of coins in rupees -/
def total_value : ℚ := 140

/-- Theorem stating that if the bag contains an equal number of one rupee, 50 paise, and 25 paise coins, 
    and the total value is 140 rupees, then the number of coins of each type is 80 -/
theorem coin_problem : 
  (num_coins : ℚ) + (num_coins : ℚ) * (1/2) + (num_coins : ℚ) * (1/4) = total_value → 
  num_coins = 80 := by sorry

end coin_problem_l2436_243693


namespace max_value_polynomial_l2436_243626

theorem max_value_polynomial (x y : ℝ) (h : x + y = 3) :
  ∃ M : ℝ, M = 400 / 11 ∧ 
  ∀ a b : ℝ, a + b = 3 → 
  a^4 * b + a^3 * b + a^2 * b + a * b + a * b^2 + a * b^3 + a * b^4 ≤ M :=
by sorry

end max_value_polynomial_l2436_243626


namespace pen_collection_theorem_l2436_243683

/-- Calculates the final number of pens after a series of operations --/
def final_pen_count (initial : ℕ) (mike_gives : ℕ) (cindy_multiplier : ℕ) (sharon_takes : ℕ) : ℕ :=
  ((initial + mike_gives) * cindy_multiplier) - sharon_takes

/-- Proves that the final number of pens is 31 given the specific conditions --/
theorem pen_collection_theorem :
  final_pen_count 5 20 2 19 = 31 := by
  sorry

end pen_collection_theorem_l2436_243683


namespace perpendicular_lines_k_value_l2436_243632

theorem perpendicular_lines_k_value (k : ℝ) :
  (((k - 3) * 2 * (k - 3) + (5 - k) * (-2) = 0) →
  (k = 1 ∨ k = 4)) ∧
  ((k = 1 ∨ k = 4) →
  ((k - 3) * 2 * (k - 3) + (5 - k) * (-2) = 0)) := by sorry

end perpendicular_lines_k_value_l2436_243632


namespace trajectory_is_parabola_min_dot_product_l2436_243679

-- Define the fixed point F
def F : ℝ × ℝ := (0, 1)

-- Define the line l₁
def l₁ (x : ℝ) : ℝ := -1

-- Define the trajectory of point C
def trajectory (x y : ℝ) : Prop := x^2 = 4*y

-- Define a line passing through F
def l₂ (k : ℝ) (x : ℝ) : ℝ := k*x + 1

-- Define the dot product of two 2D vectors
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  (v₁.1 * v₂.1) + (v₁.2 * v₂.2)

-- Theorem 1: The trajectory of point C is x² = 4y
theorem trajectory_is_parabola :
  ∀ (x y : ℝ), trajectory x y ↔ x^2 = 4*y :=
sorry

-- Theorem 2: The minimum value of RP · RQ is 16
theorem min_dot_product :
  ∃ (k : ℝ), 
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      trajectory x₁ y₁ →
      trajectory x₂ y₂ →
      y₁ = l₂ k x₁ →
      y₂ = l₂ k x₂ →
      let R : ℝ × ℝ := (-2/k, l₁ (-2/k));
      let P : ℝ × ℝ := (x₁, y₁);
      let Q : ℝ × ℝ := (x₂, y₂);
      dot_product (P.1 - R.1, P.2 - R.2) (Q.1 - R.1, Q.2 - R.2) ≥ 16 ∧
      (∃ (x₁' y₁' x₂' y₂' : ℝ),
        trajectory x₁' y₁' ∧
        trajectory x₂' y₂' ∧
        y₁' = l₂ k x₁' ∧
        y₂' = l₂ k x₂' ∧
        let R' : ℝ × ℝ := (-2/k, l₁ (-2/k));
        let P' : ℝ × ℝ := (x₁', y₁');
        let Q' : ℝ × ℝ := (x₂', y₂');
        dot_product (P'.1 - R'.1, P'.2 - R'.2) (Q'.1 - R'.1, Q'.2 - R'.2) = 16) :=
sorry

end trajectory_is_parabola_min_dot_product_l2436_243679


namespace arithmetic_expressions_l2436_243688

theorem arithmetic_expressions : 
  ((-8) - (-7) - |(-3)| = -4) ∧ 
  (-2^2 + 3 * (-1)^2019 - 9 / (-3) = 2) := by
  sorry

end arithmetic_expressions_l2436_243688


namespace largest_divisible_by_seven_l2436_243630

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    n = A * 10000 + B * 1000 + B * 100 + C * 10 + A

theorem largest_divisible_by_seven :
  ∀ n : ℕ, is_valid_number n → n % 7 = 0 → n ≤ 98879 :=
by sorry

end largest_divisible_by_seven_l2436_243630


namespace x_squared_eq_one_is_quadratic_l2436_243680

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x² = 1 is a quadratic equation -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f := by
  sorry


end x_squared_eq_one_is_quadratic_l2436_243680


namespace zongzi_pricing_and_max_purchase_l2436_243612

/-- Represents the zongzi types -/
inductive ZongziType
| A
| B

/-- Represents the price and quantity information for zongzi -/
structure ZongziInfo where
  type : ZongziType
  amount_spent : ℝ
  quantity : ℕ

/-- Theorem for zongzi pricing and maximum purchase -/
theorem zongzi_pricing_and_max_purchase 
  (info_A : ZongziInfo) 
  (info_B : ZongziInfo) 
  (total_zongzi : ℕ) 
  (max_total_amount : ℝ) :
  info_A.type = ZongziType.A →
  info_B.type = ZongziType.B →
  info_A.amount_spent = 1200 →
  info_B.amount_spent = 800 →
  info_B.quantity = info_A.quantity + 50 →
  info_A.amount_spent / info_A.quantity = 2 * (info_B.amount_spent / info_B.quantity) →
  total_zongzi = 200 →
  max_total_amount = 1150 →
  ∃ (unit_price_A unit_price_B : ℝ) (max_quantity_A : ℕ),
    unit_price_A = 8 ∧
    unit_price_B = 4 ∧
    max_quantity_A = 87 ∧
    unit_price_A * max_quantity_A + unit_price_B * (total_zongzi - max_quantity_A) ≤ max_total_amount ∧
    ∀ (quantity_A : ℕ), 
      quantity_A > max_quantity_A →
      unit_price_A * quantity_A + unit_price_B * (total_zongzi - quantity_A) > max_total_amount :=
by sorry

end zongzi_pricing_and_max_purchase_l2436_243612


namespace system_solution_l2436_243689

theorem system_solution : 
  ∃! (x y : ℚ), (4 * x - 3 * y = 2) ∧ (5 * x + 4 * y = 3) ∧ x = 17/31 ∧ y = 2/31 := by
  sorry

end system_solution_l2436_243689


namespace sum_less_than_addends_implies_negative_l2436_243603

theorem sum_less_than_addends_implies_negative (a b : ℚ) : 
  (a + b < a ∧ a + b < b) → (a < 0 ∧ b < 0) := by
  sorry

end sum_less_than_addends_implies_negative_l2436_243603


namespace inverse_square_problem_l2436_243658

/-- Represents the inverse square relationship between x and y -/
def inverse_square_relation (k : ℝ) (x y : ℝ) : Prop :=
  x = k / (y * y)

theorem inverse_square_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : y₁ = 6)
  (h₂ : x₁ = 0.1111111111111111)
  (h₃ : y₂ = 2)
  (h₄ : ∃ k, inverse_square_relation k x₁ y₁ ∧ inverse_square_relation k x₂ y₂) :
  x₂ = 1 := by
  sorry

end inverse_square_problem_l2436_243658


namespace banner_nail_distance_l2436_243610

theorem banner_nail_distance (banner_length : ℝ) (num_nails : ℕ) (end_distance : ℝ) :
  banner_length = 20 →
  num_nails = 7 →
  end_distance = 1 →
  (banner_length - 2 * end_distance) / (num_nails - 1 : ℝ) = 3 :=
by sorry

end banner_nail_distance_l2436_243610


namespace seating_theorem_l2436_243698

/-- The number of ways to arrange 3 people in a row of 6 seats with exactly two adjacent empty seats -/
def seating_arrangements (total_seats : Nat) (people : Nat) (adjacent_empty : Nat) : Nat :=
  24 * 3

theorem seating_theorem :
  seating_arrangements 6 3 2 = 72 := by
  sorry

end seating_theorem_l2436_243698


namespace middle_number_divisible_by_four_l2436_243697

theorem middle_number_divisible_by_four (x : ℕ) :
  (∃ y : ℕ, (x - 1)^3 + x^3 + (x + 1)^3 = y^3) →
  4 ∣ x :=
by sorry

end middle_number_divisible_by_four_l2436_243697


namespace triangle_problem_l2436_243684

open Real

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_c : c = 2 * b * cos B)
  (h_C : C = 2 * π / 3) :
  B = π / 6 ∧ 
  (∀ p, p = 4 + 2 * sqrt 3 → a + b + c = p → 
    ∃ m, m = sqrt 7 ∧ m^2 = (a^2 + b^2) / 4 + c^2 / 16) ∧
  (∀ S, S = 3 * sqrt 3 / 4 → (1/2) * a * b * sin C = S → 
    ∃ m, m = sqrt 21 / 2 ∧ m^2 = (a^2 + b^2) / 4 + c^2 / 16) :=
by sorry


end triangle_problem_l2436_243684


namespace rope_length_proof_l2436_243602

theorem rope_length_proof (shorter_piece longer_piece original_length : ℝ) : 
  shorter_piece = 20 →
  longer_piece = 2 * shorter_piece →
  original_length = shorter_piece + longer_piece →
  original_length = 60 := by
sorry

end rope_length_proof_l2436_243602


namespace company_fund_problem_l2436_243629

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  (initial_fund = 60 * n - 10) →    -- Fund was $10 short for $60 bonuses
  (50 * n + 120 = initial_fund) →   -- $50 bonuses given, $120 remained
  (initial_fund = 770) :=           -- Prove initial fund was $770
by
  sorry

end company_fund_problem_l2436_243629


namespace jake_watched_19_hours_on_friday_l2436_243607

/-- Represents the number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Represents the total length of the show in hours -/
def show_length : ℕ := 52

/-- Calculates the hours Jake watched on Monday -/
def monday_hours : ℕ := hours_per_day / 2

/-- Represents the hours Jake watched on Tuesday -/
def tuesday_hours : ℕ := 4

/-- Calculates the hours Jake watched on Wednesday -/
def wednesday_hours : ℕ := hours_per_day / 4

/-- Calculates the total hours Jake watched from Monday to Wednesday -/
def mon_to_wed_total : ℕ := monday_hours + tuesday_hours + wednesday_hours

/-- Calculates the hours Jake watched on Thursday -/
def thursday_hours : ℕ := mon_to_wed_total / 2

/-- Calculates the total hours Jake watched from Monday to Thursday -/
def mon_to_thu_total : ℕ := mon_to_wed_total + thursday_hours

/-- Represents the hours Jake watched on Friday -/
def friday_hours : ℕ := show_length - mon_to_thu_total

theorem jake_watched_19_hours_on_friday : friday_hours = 19 := by
  sorry

end jake_watched_19_hours_on_friday_l2436_243607


namespace unique_n_l2436_243666

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

def digit_product (n : ℕ) : ℕ := 
  (n / 100) * ((n / 10) % 10) * (n % 10)

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem unique_n : 
  ∃! n : ℕ, 
    is_three_digit n ∧ 
    is_perfect_square n ∧ 
    is_two_digit (digit_sum n) ∧
    (∀ m : ℕ, is_three_digit m ∧ is_perfect_square m ∧ digit_product m = digit_product n → m = n) ∧
    (∃ m : ℕ, is_three_digit m ∧ is_perfect_square m ∧ m ≠ n ∧ digit_sum m = digit_sum n) ∧
    (∀ m : ℕ, is_three_digit m ∧ is_perfect_square m ∧ digit_sum m = digit_sum n →
      (∀ k : ℕ, is_three_digit k ∧ is_perfect_square k ∧ digit_product k = digit_product m → k = m)) ∧
    n = 841 :=
by sorry

end unique_n_l2436_243666


namespace f_max_value_implies_a_eq_three_l2436_243657

/-- The function f(x) = -4x^3 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -4 * x^3 + a * x

/-- The maximum value of f(x) on [-1,1] is 1 -/
def max_value_is_one (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a x ≤ 1 ∧ ∃ y : ℝ, y ∈ Set.Icc (-1) 1 ∧ f a y = 1

theorem f_max_value_implies_a_eq_three :
  ∀ a : ℝ, max_value_is_one a → a = 3 := by sorry

end f_max_value_implies_a_eq_three_l2436_243657


namespace sin_even_function_phi_l2436_243669

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem sin_even_function_phi (φ : ℝ) 
  (h1 : is_even_function (fun x ↦ Real.sin (x + φ)))
  (h2 : 0 ≤ φ ∧ φ ≤ π) :
  φ = π / 2 := by
  sorry

end sin_even_function_phi_l2436_243669


namespace fourth_root_equivalence_l2436_243649

-- Define y as a positive real number
variable (y : ℝ) (hy : y > 0)

-- State the theorem
theorem fourth_root_equivalence : (y^2 * y^(1/2))^(1/4) = y^(5/8) := by sorry

end fourth_root_equivalence_l2436_243649


namespace tammy_climbing_l2436_243601

/-- Tammy's mountain climbing problem -/
theorem tammy_climbing (total_time total_distance : ℝ) 
  (speed_diff time_diff : ℝ) : 
  total_time = 14 →
  speed_diff = 0.5 →
  time_diff = 2 →
  total_distance = 52 →
  ∃ (speed1 time1 : ℝ),
    speed1 * time1 + (speed1 + speed_diff) * (time1 - time_diff) = total_distance ∧
    time1 + (time1 - time_diff) = total_time ∧
    speed1 + speed_diff = 4 :=
by sorry

end tammy_climbing_l2436_243601


namespace octal_subtraction_correct_l2436_243605

/-- Represents a number in base 8 -/
def OctalNum := Nat

/-- Addition in base 8 -/
def octal_add (a b : OctalNum) : OctalNum :=
  sorry

/-- Subtraction in base 8 -/
def octal_sub (a b : OctalNum) : OctalNum :=
  sorry

/-- Conversion from decimal to octal -/
def to_octal (n : Nat) : OctalNum :=
  sorry

theorem octal_subtraction_correct :
  let a : OctalNum := to_octal 537
  let b : OctalNum := to_octal 261
  let c : OctalNum := to_octal 256
  octal_sub a b = c ∧ octal_add b c = a := by
  sorry

end octal_subtraction_correct_l2436_243605


namespace book_cost_price_l2436_243675

/-- Proves that given a book sold for Rs 70 with a 40% profit rate, the cost price of the book is Rs 50. -/
theorem book_cost_price (selling_price : ℝ) (profit_rate : ℝ) 
  (h1 : selling_price = 70)
  (h2 : profit_rate = 0.4) :
  selling_price / (1 + profit_rate) = 50 := by
  sorry

end book_cost_price_l2436_243675


namespace johns_first_second_distance_l2436_243677

/-- Represents the race scenario with John and James --/
structure RaceScenario where
  john_total_time : ℝ
  john_total_distance : ℝ
  james_top_speed_diff : ℝ
  james_initial_distance : ℝ
  james_initial_time : ℝ
  james_total_time : ℝ
  james_total_distance : ℝ

/-- Theorem stating John's distance in the first second --/
theorem johns_first_second_distance 
  (race : RaceScenario)
  (h_john_time : race.john_total_time = 13)
  (h_john_dist : race.john_total_distance = 100)
  (h_james_speed_diff : race.james_top_speed_diff = 2)
  (h_james_initial_dist : race.james_initial_distance = 10)
  (h_james_initial_time : race.james_initial_time = 2)
  (h_james_time : race.james_total_time = 11)
  (h_james_dist : race.james_total_distance = 100) :
  ∃ d : ℝ, d = 4 ∧ 
    (race.john_total_distance - d) / (race.john_total_time - 1) = 
    (race.james_total_distance - race.james_initial_distance) / (race.james_total_time - race.james_initial_time) - race.james_top_speed_diff :=
by sorry

end johns_first_second_distance_l2436_243677


namespace remaining_flour_l2436_243663

def flour_needed (total_required : ℕ) (already_added : ℕ) : ℕ :=
  total_required - already_added

theorem remaining_flour :
  flour_needed 9 2 = 7 :=
by sorry

end remaining_flour_l2436_243663


namespace vector_dot_product_l2436_243641

theorem vector_dot_product (a b : ℝ × ℝ) :
  a + b = (2, -4) →
  3 • a - b = (-10, 16) →
  a • b = -29 := by
sorry

end vector_dot_product_l2436_243641


namespace well_digging_hours_l2436_243653

/-- The number of hours worked on the first day by two men digging a well -/
def first_day_hours : ℕ := 20

/-- The total payment for both men over three days of work -/
def total_payment : ℕ := 660

/-- The hourly rate paid to each man -/
def hourly_rate : ℕ := 10

/-- The number of hours worked by both men on the second day -/
def second_day_hours : ℕ := 16

/-- The number of hours worked by both men on the third day -/
def third_day_hours : ℕ := 30

theorem well_digging_hours : 
  hourly_rate * (first_day_hours + second_day_hours + third_day_hours) = total_payment :=
by sorry

end well_digging_hours_l2436_243653


namespace square_area_error_percentage_l2436_243647

/-- If the side of a square is measured with a 2% excess error, 
    then the percentage of error in the calculated area of the square is 4.04%. -/
theorem square_area_error_percentage (s : ℝ) (s' : ℝ) (A : ℝ) (A' : ℝ) :
  s' = s * (1 + 0.02) →
  A = s^2 →
  A' = s'^2 →
  (A' - A) / A * 100 = 4.04 := by
  sorry

end square_area_error_percentage_l2436_243647


namespace letterArrangements_eq_25_l2436_243674

/-- The number of ways to arrange 15 letters with specific constraints -/
def letterArrangements : ℕ :=
  let totalLetters := 15
  let numA := 4
  let numB := 6
  let numC := 5
  let firstSection := 5
  let middleSection := 5
  let lastSection := 5
  -- Define the constraints
  let noCInFirst := true
  let noAInMiddle := true
  let noBInLast := true
  -- Calculate the number of arrangements
  25

/-- Theorem stating that the number of valid arrangements is 25 -/
theorem letterArrangements_eq_25 : letterArrangements = 25 := by
  sorry

end letterArrangements_eq_25_l2436_243674


namespace partial_fraction_decomposition_l2436_243639

theorem partial_fraction_decomposition (M₁ M₂ : ℚ) :
  (∀ x : ℚ, x ≠ 2 → x ≠ 3 → (45 * x - 82) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) →
  M₁ * M₂ = -424 := by
sorry

end partial_fraction_decomposition_l2436_243639


namespace simplify_and_sum_fraction_l2436_243648

theorem simplify_and_sum_fraction : ∃ (a b : ℕ), 
  (75 : ℚ) / 100 = (a : ℚ) / b ∧ 
  (∀ (c d : ℕ), (75 : ℚ) / 100 = (c : ℚ) / d → a ≤ c ∧ b ≤ d) ∧
  a + b = 7 := by
  sorry

end simplify_and_sum_fraction_l2436_243648


namespace remainder_three_to_27_mod_13_l2436_243687

theorem remainder_three_to_27_mod_13 : 3^27 % 13 = 1 := by
  sorry

end remainder_three_to_27_mod_13_l2436_243687


namespace cookies_on_floor_l2436_243628

/-- Calculates the number of cookies thrown on the floor given the initial and additional cookies baked by Alice and Bob, and the final number of edible cookies. -/
theorem cookies_on_floor (alice_initial bob_initial alice_additional bob_additional final_edible : ℕ) :
  alice_initial = 74 →
  bob_initial = 7 →
  alice_additional = 5 →
  bob_additional = 36 →
  final_edible = 93 →
  (alice_initial + bob_initial + alice_additional + bob_additional) - final_edible = 29 := by
  sorry

#check cookies_on_floor

end cookies_on_floor_l2436_243628


namespace cube_volume_ratio_l2436_243673

/-- The ratio of the volume of a cube with edge length 10 inches to the volume of a cube with edge length 3 feet -/
theorem cube_volume_ratio : 
  let inch_to_foot : ℚ := 1 / 12
  let cube1_edge : ℚ := 10
  let cube2_edge : ℚ := 3 / inch_to_foot
  let cube1_volume : ℚ := cube1_edge ^ 3
  let cube2_volume : ℚ := cube2_edge ^ 3
  cube1_volume / cube2_volume = 125 / 5832 := by
sorry

end cube_volume_ratio_l2436_243673


namespace linear_system_sum_theorem_l2436_243682

theorem linear_system_sum_theorem (a b c x y z : ℝ) 
  (eq1 : 23*x + b*y + c*z = 0)
  (eq2 : a*x + 33*y + c*z = 0)
  (eq3 : a*x + b*y + 52*z = 0)
  (ha : a ≠ 23)
  (hx : x ≠ 0) :
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1 := by
sorry

end linear_system_sum_theorem_l2436_243682


namespace inequality_proof_l2436_243671

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end inequality_proof_l2436_243671


namespace cellCount_after_8_days_l2436_243645

/-- The number of cells in a colony after a given number of days, 
    with specific growth and toxin conditions. -/
def cellCount (initialCells : ℕ) (days : ℕ) : ℕ :=
  let growthPeriods := days / 2
  let afterGrowth := initialCells * 3^growthPeriods
  if days ≥ 6 then
    (afterGrowth / 2 + if afterGrowth % 2 = 0 then 0 else 1) * 3^((days - 6) / 2)
  else
    afterGrowth

theorem cellCount_after_8_days : 
  cellCount 5 8 = 201 := by sorry

end cellCount_after_8_days_l2436_243645


namespace v_2002_equals_2_l2436_243620

def g : ℕ → ℕ
  | 1 => 5
  | 2 => 3
  | 3 => 1
  | 4 => 2
  | 5 => 4
  | _ => 0  -- Default case for completeness

def v : ℕ → ℕ
  | 0 => 5
  | n + 1 => g (v n)

theorem v_2002_equals_2 : v 2002 = 2 := by
  sorry

end v_2002_equals_2_l2436_243620


namespace product_line_size_l2436_243637

/-- Represents the product line of Company C -/
structure ProductLine where
  n : ℕ                  -- number of products
  prices : Fin n → ℝ     -- prices of products
  avg_price : ℝ          -- average price
  min_price : ℝ          -- minimum price
  max_price : ℝ          -- maximum price
  low_price_count : ℕ    -- count of products below $1000

/-- The product line satisfies the given conditions -/
def satisfies_conditions (pl : ProductLine) : Prop :=
  pl.avg_price = 1200 ∧
  (∀ i, pl.prices i ≥ 400) ∧
  pl.low_price_count = 10 ∧
  (∀ i, pl.prices i < 1000 ∨ pl.prices i ≥ 1000) ∧
  pl.max_price = 11000 ∧
  (∃ i, pl.prices i = pl.max_price)

/-- The theorem to be proved -/
theorem product_line_size (pl : ProductLine) 
  (h : satisfies_conditions pl) : pl.n = 20 := by
  sorry


end product_line_size_l2436_243637


namespace correct_meiosis_sequence_l2436_243604

-- Define the stages of meiosis
inductive MeiosisStage
  | Replication
  | Synapsis
  | Separation
  | Division

-- Define a sequence type
def Sequence := List MeiosisStage

-- Define the four given sequences
def sequenceA : Sequence := [MeiosisStage.Replication, MeiosisStage.Synapsis, MeiosisStage.Separation, MeiosisStage.Division]
def sequenceB : Sequence := [MeiosisStage.Synapsis, MeiosisStage.Replication, MeiosisStage.Separation, MeiosisStage.Division]
def sequenceC : Sequence := [MeiosisStage.Synapsis, MeiosisStage.Replication, MeiosisStage.Division, MeiosisStage.Separation]
def sequenceD : Sequence := [MeiosisStage.Replication, MeiosisStage.Separation, MeiosisStage.Synapsis, MeiosisStage.Division]

-- Define a function to check if a sequence is correct
def isCorrectSequence (s : Sequence) : Prop :=
  s = sequenceA

-- Theorem stating that sequenceA is the correct sequence
theorem correct_meiosis_sequence :
  isCorrectSequence sequenceA ∧
  ¬isCorrectSequence sequenceB ∧
  ¬isCorrectSequence sequenceC ∧
  ¬isCorrectSequence sequenceD :=
sorry

end correct_meiosis_sequence_l2436_243604


namespace inequality_solution_implies_a_range_l2436_243686

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x, (1 + a) * x > 1 + a ↔ x < 1) → a < -1 := by
  sorry

end inequality_solution_implies_a_range_l2436_243686


namespace grade_distribution_l2436_243691

theorem grade_distribution (total_students : ℕ) 
  (fraction_A : ℚ) (fraction_C : ℚ) (number_D : ℕ) :
  total_students = 100 →
  fraction_A = 1/5 →
  fraction_C = 1/2 →
  number_D = 5 →
  (total_students : ℚ) - (fraction_A * total_students + fraction_C * total_students + number_D) = 1/4 * total_students :=
by sorry

end grade_distribution_l2436_243691


namespace quadratic_completion_of_square_l2436_243681

theorem quadratic_completion_of_square :
  ∀ x : ℝ, x^2 - 6*x + 4 = 0 ↔ (x - 3)^2 = 5 :=
by sorry

end quadratic_completion_of_square_l2436_243681


namespace fraction_equals_zero_l2436_243638

theorem fraction_equals_zero (x : ℝ) : x / (x^2 - 1) = 0 → x = 0 := by
  sorry

end fraction_equals_zero_l2436_243638


namespace screenwriter_speed_l2436_243636

/-- Calculates the average words per minute for a given script and writing duration -/
def average_words_per_minute (total_words : ℕ) (total_hours : ℕ) : ℚ :=
  (total_words : ℚ) / (total_hours * 60 : ℚ)

/-- Theorem stating that a 30,000-word script written in 100 hours has an average writing speed of 5 words per minute -/
theorem screenwriter_speed : average_words_per_minute 30000 100 = 5 := by
  sorry

#eval average_words_per_minute 30000 100

end screenwriter_speed_l2436_243636


namespace passes_through_fixed_point_not_in_fourth_quadrant_min_area_min_area_line_eq_l2436_243652

/-- Given a line l: kx - 3y + 2k + 3 = 0, where k ∈ ℝ -/
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - 3 * y + 2 * k + 3 = 0

/-- The point (-2, 1) -/
def fixed_point : ℝ × ℝ := (-2, 1)

/-- The line passes through the fixed point for all values of k -/
theorem passes_through_fixed_point (k : ℝ) :
  line_l k (fixed_point.1) (fixed_point.2) := by sorry

/-- The line does not pass through the fourth quadrant when k ∈ [0, +∞) -/
theorem not_in_fourth_quadrant (k : ℝ) (hk : k ≥ 0) :
  ∀ x y, line_l k x y → (x ≤ 0 ∧ y ≥ 0) ∨ (x ≥ 0 ∧ y ≥ 0) := by sorry

/-- The area of triangle AOB formed by the line's intersections with the x and y axes -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  (1/6) * (4 * k + 9 / k + 12)

/-- The minimum area of triangle AOB is 4, occurring when k = 3/2 -/
theorem min_area :
  ∃ k, k > 0 ∧ triangle_area k = 4 ∧ ∀ k', k' > 0 → triangle_area k' ≥ 4 := by sorry

/-- The line equation at the minimum area point -/
def min_area_line (x y : ℝ) : Prop := x - 2 * y + 4 = 0

/-- The line equation at the minimum area point is x - 2y + 4 = 0 -/
theorem min_area_line_eq :
  ∃ k, k > 0 ∧ triangle_area k = 4 ∧ ∀ x y, line_l k x y ↔ min_area_line x y := by sorry

end passes_through_fixed_point_not_in_fourth_quadrant_min_area_min_area_line_eq_l2436_243652


namespace leonie_cats_l2436_243617

theorem leonie_cats : ∃ n : ℚ, n = (4 / 5) * n + (4 / 5) → n = 4 := by
  sorry

end leonie_cats_l2436_243617


namespace katie_earnings_l2436_243623

/-- The number of bead necklaces Katie sold -/
def bead_necklaces : ℕ := 4

/-- The number of gem stone necklaces Katie sold -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 3

/-- The total money Katie earned from selling necklaces -/
def total_earned : ℕ := (bead_necklaces + gem_necklaces) * necklace_cost

theorem katie_earnings : total_earned = 21 := by
  sorry

end katie_earnings_l2436_243623


namespace sally_and_fred_onions_l2436_243611

/-- The number of onions Sally and Fred have after giving some to Sara -/
def remaining_onions (sally_onions fred_onions given_onions : ℕ) : ℕ :=
  sally_onions + fred_onions - given_onions

/-- Theorem stating that Sally and Fred have 10 onions after giving some to Sara -/
theorem sally_and_fred_onions :
  remaining_onions 5 9 4 = 10 := by
  sorry

end sally_and_fred_onions_l2436_243611


namespace max_sum_squares_l2436_243672

theorem max_sum_squares : ∃ (m n : ℕ), 
  1 ≤ m ∧ m ≤ 2005 ∧ 
  1 ≤ n ∧ n ≤ 2005 ∧ 
  (n^2 + 2*m*n - 2*m^2)^2 = 1 ∧ 
  m^2 + n^2 = 702036 ∧ 
  ∀ (m' n' : ℕ), 
    1 ≤ m' ∧ m' ≤ 2005 → 
    1 ≤ n' ∧ n' ≤ 2005 → 
    (n'^2 + 2*m'*n' - 2*m'^2)^2 = 1 → 
    m'^2 + n'^2 ≤ 702036 :=
by sorry

end max_sum_squares_l2436_243672


namespace toy_sales_profit_maximization_l2436_243614

def weekly_sales (x : ℤ) (k : ℚ) (b : ℚ) : ℚ := k * x + b

theorem toy_sales_profit_maximization 
  (k : ℚ) (b : ℚ) 
  (h1 : weekly_sales 120 k b = 80) 
  (h2 : weekly_sales 140 k b = 40) 
  (h3 : ∀ x : ℤ, 100 ≤ x ∧ x ≤ 160) :
  (k = -2 ∧ b = 320) ∧
  (∀ x : ℤ, (x - 100) * (weekly_sales x k b) ≤ 1800) ∧
  ((130 - 100) * (weekly_sales 130 k b) = 1800) :=
sorry

end toy_sales_profit_maximization_l2436_243614


namespace johns_piggy_bank_l2436_243646

theorem johns_piggy_bank (total_coins quarters dimes nickels : ℕ) : 
  total_coins = 63 →
  quarters = 22 →
  dimes = quarters + 3 →
  total_coins = quarters + dimes + nickels →
  quarters - nickels = 6 :=
by sorry

end johns_piggy_bank_l2436_243646


namespace three_consecutive_free_throws_l2436_243662

/-- The probability of scoring a single free throw -/
def free_throw_probability : ℝ := 0.7

/-- The number of consecutive free throws -/
def num_throws : ℕ := 3

/-- The probability of scoring in three consecutive free throws -/
def three_consecutive_probability : ℝ := free_throw_probability ^ num_throws

theorem three_consecutive_free_throws :
  three_consecutive_probability = 0.343 := by
  sorry

end three_consecutive_free_throws_l2436_243662


namespace parabola_equation_from_conditions_l2436_243667

/-- A parabola is defined by its focus-directrix distance and a point it passes through. -/
structure Parabola where
  focus_directrix_distance : ℝ
  point : ℝ × ℝ

/-- The equation of a parabola in the form y^2 = ax, where a is a real number. -/
def parabola_equation (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y => y^2 = a * x

theorem parabola_equation_from_conditions (p : Parabola) 
  (h1 : p.focus_directrix_distance = 2)
  (h2 : p.point = (1, 2)) :
  parabola_equation 4 = fun x y => y^2 = 4 * x :=
by sorry

end parabola_equation_from_conditions_l2436_243667


namespace quadratic_roots_relation_l2436_243643

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∀ x, x^2 + m*x + n = 0 ↔ (x/2)^2 + p*(x/2) + m = 0) →
  n / p = 8 := by
  sorry

end quadratic_roots_relation_l2436_243643


namespace total_cost_construction_materials_l2436_243694

def cement_bags : ℕ := 500
def cement_price_per_bag : ℕ := 10
def sand_lorries : ℕ := 20
def sand_tons_per_lorry : ℕ := 10
def sand_price_per_ton : ℕ := 40

theorem total_cost_construction_materials : 
  cement_bags * cement_price_per_bag + 
  sand_lorries * sand_tons_per_lorry * sand_price_per_ton = 13000 :=
by sorry

end total_cost_construction_materials_l2436_243694


namespace min_students_in_class_l2436_243622

theorem min_students_in_class (b g : ℕ) : 
  (2 * b / 3 : ℚ) = (3 * g / 4 : ℚ) →
  b + g ≥ 17 ∧ 
  ∃ (b' g' : ℕ), b' + g' = 17 ∧ (2 * b' / 3 : ℚ) = (3 * g' / 4 : ℚ) := by
  sorry

end min_students_in_class_l2436_243622


namespace unique_mod_residue_l2436_243633

theorem unique_mod_residue :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -4321 [ZMOD 10] := by
  sorry

end unique_mod_residue_l2436_243633


namespace range_of_m_l2436_243655

-- Define the solution set A
def A (m : ℝ) : Set ℝ := {x : ℝ | |x^2 - 4*x + m| ≤ x + 4}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (0 ∈ A m) ∧ (2 ∉ A m) ↔ m ∈ Set.Icc (-4 : ℝ) (-2 : ℝ) := by sorry

end range_of_m_l2436_243655


namespace candy_distribution_l2436_243627

theorem candy_distribution (total_candies : ℕ) 
  (lollipops_per_boy : ℕ) (candy_canes_per_girl : ℕ) : 
  total_candies = 90 →
  lollipops_per_boy = 3 →
  candy_canes_per_girl = 2 →
  ∃ (num_boys num_girls : ℕ),
    num_boys * lollipops_per_boy = total_candies / 3 ∧
    num_girls * candy_canes_per_girl = total_candies * 2 / 3 ∧
    num_boys + num_girls = 40 :=
by sorry

end candy_distribution_l2436_243627


namespace y1_greater_than_y2_l2436_243600

-- Define the parabola function
def f (x : ℝ) : ℝ := -(x + 1)^2 + 3

-- Define the theorem
theorem y1_greater_than_y2 :
  ∀ y₁ y₂ : ℝ, f 1 = y₁ → f 2 = y₂ → y₁ > y₂ := by
  sorry

end y1_greater_than_y2_l2436_243600


namespace expression_simplification_l2436_243670

theorem expression_simplification (x y : ℚ) (hx : x = 1/2) (hy : y = -2) :
  ((2*x + y)^2 - (2*x - y)*(x + y) - 2*(x - 2*y)*(x + 2*y)) / y = -37/2 := by
  sorry

end expression_simplification_l2436_243670


namespace bacteria_growth_proof_l2436_243659

/-- The number of 30-second intervals in 5 minutes -/
def intervals : ℕ := 10

/-- The growth factor of bacteria population in one interval -/
def growth_factor : ℕ := 4

/-- The final number of bacteria after 5 minutes -/
def final_population : ℕ := 4194304

/-- The initial number of bacteria -/
def initial_population : ℕ := 4

theorem bacteria_growth_proof :
  initial_population * growth_factor ^ intervals = final_population :=
by sorry

end bacteria_growth_proof_l2436_243659


namespace smallest_positive_solution_tan_cos_l2436_243609

theorem smallest_positive_solution_tan_cos (x : ℝ) : 
  (x > 0 ∧ x = Real.pi / 8 ∧ Real.tan (2 * x) + Real.tan (4 * x) = Real.cos (2 * x)) ∧
  (∀ y : ℝ, y > 0 ∧ y < x → Real.tan (2 * y) + Real.tan (4 * y) ≠ Real.cos (2 * y)) :=
by sorry

end smallest_positive_solution_tan_cos_l2436_243609


namespace min_value_sum_l2436_243616

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  4*x + 9*y ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 1/y₀ = 1 ∧ 4*x₀ + 9*y₀ = 25 :=
by sorry

end min_value_sum_l2436_243616


namespace quadratic_properties_l2436_243685

def f (x : ℝ) := -x^2 + 2*x + 1

theorem quadratic_properties :
  (∀ x y : ℝ, f x ≤ f y → x = y ∨ (x < y ∧ f ((x + y) / 2) > f x) ∨ (y < x ∧ f ((x + y) / 2) > f x)) ∧
  (∃ x : ℝ, ∀ y : ℝ, f y ≤ f x) ∧
  (∃! x : ℝ, ∀ y : ℝ, f y ≤ f x) ∧
  (∀ x : ℝ, f x ≤ f 1) ∧
  f 1 = 2 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → -2 ≤ f x ∧ f x ≤ 2) :=
sorry

end quadratic_properties_l2436_243685


namespace almond_salami_cheese_cost_l2436_243618

/-- The cost of Sean's Sunday purchases -/
def sean_sunday_cost (almond_croissant : ℝ) (salami_cheese_croissant : ℝ) : ℝ :=
  almond_croissant + salami_cheese_croissant + 3 + 4 + 2 * 2.5

/-- Theorem stating the combined cost of almond and salami & cheese croissants -/
theorem almond_salami_cheese_cost :
  ∃ (almond_croissant salami_cheese_croissant : ℝ),
    sean_sunday_cost almond_croissant salami_cheese_croissant = 21 ∧
    almond_croissant + salami_cheese_croissant = 9 :=
by
  sorry

end almond_salami_cheese_cost_l2436_243618


namespace news_report_probability_l2436_243661

/-- The duration of the "Midday News" program in minutes -/
def program_duration : ℕ := 30

/-- The duration of the news report in minutes -/
def news_report_duration : ℕ := 5

/-- The time Xiao Zhang starts watching, in minutes after the program start -/
def watch_start_time : ℕ := 20

/-- The probability of watching the entire news report -/
def watch_probability : ℚ := 1 / 6

theorem news_report_probability :
  let favorable_time := program_duration - watch_start_time - news_report_duration + 1
  watch_probability = favorable_time / program_duration :=
by sorry

end news_report_probability_l2436_243661


namespace added_number_after_doubling_l2436_243634

theorem added_number_after_doubling (original : ℕ) (added : ℕ) : 
  original = 6 →
  3 * (2 * original + added) = 63 →
  added = 9 := by
sorry

end added_number_after_doubling_l2436_243634
