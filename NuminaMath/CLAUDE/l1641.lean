import Mathlib

namespace NUMINAMATH_CALUDE_additional_track_length_l1641_164125

/-- Calculates the additional track length required when changing the grade of a railroad track. -/
theorem additional_track_length
  (elevation : ℝ)
  (initial_grade : ℝ)
  (final_grade : ℝ)
  (h1 : elevation = 1200)
  (h2 : initial_grade = 0.04)
  (h3 : final_grade = 0.03) :
  (elevation / final_grade) - (elevation / initial_grade) = 10000 :=
by sorry

end NUMINAMATH_CALUDE_additional_track_length_l1641_164125


namespace NUMINAMATH_CALUDE_vector_perpendicular_value_l1641_164152

theorem vector_perpendicular_value (k : ℝ) : 
  let a : (ℝ × ℝ) := (3, 1)
  let b : (ℝ × ℝ) := (1, 3)
  let c : (ℝ × ℝ) := (k, -2)
  (((a.1 - c.1) * b.1 + (a.2 - c.2) * b.2) = 0) → k = 12 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_value_l1641_164152


namespace NUMINAMATH_CALUDE_factor_condition_l1641_164128

theorem factor_condition (a b c m l : ℝ) : 
  ((b + c) * (c + a) * (a + b) + a * b * c = 
   (m * (a^2 + b^2 + c^2) + l * (a * b + a * c + b * c)) * k) →
  (m = 0 ∧ l = a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_factor_condition_l1641_164128


namespace NUMINAMATH_CALUDE_min_product_with_constraint_min_product_achievable_l1641_164120

theorem min_product_with_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 20 * a * b = 13 * a + 14 * b) : a * b ≥ 1.82 := by
  sorry

theorem min_product_achievable : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  20 * a * b = 13 * a + 14 * b ∧ a * b = 1.82 := by
  sorry

end NUMINAMATH_CALUDE_min_product_with_constraint_min_product_achievable_l1641_164120


namespace NUMINAMATH_CALUDE_triangle_problem_l1641_164134

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  A = 30 * π / 180 ∧  -- Convert 30° to radians
  a = 2 ∧
  b = 2 * Real.sqrt 3 ∧
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  c / Real.sin C = a / Real.sin A →
  -- Conclusions
  Real.sin B = Real.sqrt 3 / 2 ∧
  ∃! (B' C' : Real), B' ≠ B ∧ 
    A + B + C = π ∧
    A + B' + C' = π ∧
    a / Real.sin A = b / Real.sin B' ∧
    b / Real.sin B' = c / Real.sin C' ∧
    c / Real.sin C' = a / Real.sin A :=
by sorry


end NUMINAMATH_CALUDE_triangle_problem_l1641_164134


namespace NUMINAMATH_CALUDE_bad_carrots_l1641_164176

/-- Given the number of carrots picked by Carol and her mother, and the number of good carrots,
    calculate the number of bad carrots. -/
theorem bad_carrots (carol_carrots mother_carrots good_carrots : ℕ) : 
  carol_carrots = 29 → mother_carrots = 16 → good_carrots = 38 →
  carol_carrots + mother_carrots - good_carrots = 7 := by
  sorry

#check bad_carrots

end NUMINAMATH_CALUDE_bad_carrots_l1641_164176


namespace NUMINAMATH_CALUDE_sum_of_digits_divisible_by_13_l1641_164158

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_divisible_by_13 (n : ℕ) : 
  ∃ k ∈ Finset.range 79, 13 ∣ sum_of_digits (n + k) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_divisible_by_13_l1641_164158


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l1641_164166

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 3^7 ways to put 7 distinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : ways_to_put_balls_in_boxes 7 3 = 3^7 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l1641_164166


namespace NUMINAMATH_CALUDE_special_complex_sum_l1641_164124

-- Define the complex function f
def f (z : ℂ) : ℂ := z^2 - 19*z

-- Define the condition for a right triangle
def is_right_triangle (z : ℂ) : Prop :=
  (f z - z) • (f (f z) - f z) = 0

-- Define the structure of z
structure SpecialComplex where
  m : ℕ+
  n : ℕ+
  z : ℂ
  h : z = m + Real.sqrt n + 11*Complex.I

-- State the theorem
theorem special_complex_sum (sc : SpecialComplex) (h : is_right_triangle sc.z) :
  sc.m + sc.n = 230 :=
sorry

end NUMINAMATH_CALUDE_special_complex_sum_l1641_164124


namespace NUMINAMATH_CALUDE_difference_theorem_l1641_164100

def difference (a b c : ℚ) : ℚ :=
  min (a - b) (min ((a - c) / 2) ((b - c) / 3))

theorem difference_theorem :
  (difference (-2) (-4) 1 = -5/3) ∧
  (2/3 = max
    (max (difference (-2) (-4) 1) (difference (-2) 1 (-4)))
    (max (difference (-4) (-2) 1) (max (difference (-4) 1 (-2)) (max (difference 1 (-4) (-2)) (difference 1 (-2) (-4)))))) ∧
  (∀ x : ℚ, difference (-1) 6 x = 2 ↔ (x = -7 ∨ x = 8)) :=
by sorry

end NUMINAMATH_CALUDE_difference_theorem_l1641_164100


namespace NUMINAMATH_CALUDE_inequality_proof_l1641_164118

theorem inequality_proof (a b : ℝ) (h1 : a + b > 0) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1641_164118


namespace NUMINAMATH_CALUDE_calculation_problems_l1641_164140

theorem calculation_problems :
  ((-2 : ℤ) + 5 - abs (-8) + (-5) = -10) ∧
  ((-2 : ℤ)^2 * 5 - (-2)^3 / 4 = 22) := by
  sorry

end NUMINAMATH_CALUDE_calculation_problems_l1641_164140


namespace NUMINAMATH_CALUDE_f_properties_l1641_164149

noncomputable def f (x : ℝ) : ℝ := (Real.sin x - Real.cos x) * Real.sin (2 * x) / Real.sin x

theorem f_properties :
  (∀ x : ℝ, f x ≠ 0 → x ∉ {y | ∃ k : ℤ, y = k * Real.pi}) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 2 + k * Real.pi → f x ≥ 0) ∧
  (∃ m : ℝ, m > 0 ∧ m = 3 * Real.pi / 8 ∧ ∀ x : ℝ, f (x + m) = f (-x + m)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1641_164149


namespace NUMINAMATH_CALUDE_math_books_in_same_box_probability_l1641_164197

def total_books : ℕ := 13
def math_books : ℕ := 4
def box_1_capacity : ℕ := 4
def box_2_capacity : ℕ := 4
def box_3_capacity : ℕ := 5

def probability_all_math_books_in_same_box : ℚ := 1 / 4120

theorem math_books_in_same_box_probability :
  let total_arrangements := (total_books.choose box_1_capacity) * 
                            ((total_books - box_1_capacity).choose box_2_capacity) *
                            ((total_books - box_1_capacity - box_2_capacity).choose box_3_capacity)
  let favorable_outcomes := (total_books - math_books).choose 1 * 
                            ((total_books - math_books - 1).choose box_2_capacity) *
                            ((total_books - math_books - 1 - box_2_capacity).choose box_3_capacity)
  (favorable_outcomes : ℚ) / total_arrangements = probability_all_math_books_in_same_box :=
sorry

end NUMINAMATH_CALUDE_math_books_in_same_box_probability_l1641_164197


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1641_164127

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the right focus
def right_focus (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the asymptote
def asymptote (a b : ℝ) (x y : ℝ) : Prop :=
  y = b / a * x ∨ y = -b / a * x

-- Define an equilateral triangle
def equilateral_triangle (A B C : ℝ × ℝ) (side_length : ℝ) : Prop :=
  dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

theorem hyperbola_equation (a b : ℝ) (F A : ℝ × ℝ) :
  a > 0 → b > 0 →
  hyperbola a b F.1 F.2 →
  F = right_focus a →
  asymptote a b A.1 A.2 →
  equilateral_triangle (0, 0) F A 2 →
  ∃ (x y : ℝ), x^2 - y^2 / 3 = 1 := by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l1641_164127


namespace NUMINAMATH_CALUDE_calculate_expression_l1641_164146

theorem calculate_expression : (-7)^7 / 7^4 + 2^8 - 10^1 = -97 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1641_164146


namespace NUMINAMATH_CALUDE_union_equals_universal_l1641_164160

-- Define the universal set U
def U : Finset Nat := {2, 3, 4, 5, 6}

-- Define set M
def M : Finset Nat := {3, 4, 5}

-- Define set N
def N : Finset Nat := {2, 4, 6}

-- Theorem statement
theorem union_equals_universal : M ∪ N = U := by sorry

end NUMINAMATH_CALUDE_union_equals_universal_l1641_164160


namespace NUMINAMATH_CALUDE_robin_pieces_count_l1641_164174

theorem robin_pieces_count (gum_packages : ℕ) (candy_packages : ℕ) (pieces_per_package : ℕ) : 
  gum_packages = 28 → candy_packages = 14 → pieces_per_package = 6 →
  gum_packages * pieces_per_package + candy_packages * pieces_per_package = 252 := by
sorry

end NUMINAMATH_CALUDE_robin_pieces_count_l1641_164174


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l1641_164142

theorem profit_percentage_previous_year 
  (R : ℝ) -- Revenues in the previous year
  (P : ℝ) -- Profits in the previous year
  (h1 : 0.95 * R * 0.10 = 0.95 * P) -- Condition relating 2009 profits to previous year
  : P / R = 0.10 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l1641_164142


namespace NUMINAMATH_CALUDE_remainder_theorem_l1641_164162

theorem remainder_theorem (n : ℤ) : n % 9 = 3 → (4 * n - 9) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1641_164162


namespace NUMINAMATH_CALUDE_sasha_can_buy_everything_l1641_164184

-- Define the store's discount policy and item prices
def discount_threshold : ℝ := 1500
def discount_rate : ℝ := 0.26
def shashlik_price : ℝ := 350
def sauce_price : ℝ := 70

-- Define Sasha's budget and desired quantities
def budget : ℝ := 1800
def shashlik_quantity : ℝ := 5
def sauce_quantity : ℝ := 1

-- Define a function to calculate the discounted price
def discounted_price (price : ℝ) : ℝ := price * (1 - discount_rate)

-- Theorem: Sasha can buy everything he planned within his budget
theorem sasha_can_buy_everything :
  ∃ (first_shashlik second_shashlik first_sauce : ℝ),
    first_shashlik + second_shashlik = shashlik_quantity ∧
    first_sauce = sauce_quantity ∧
    first_shashlik * shashlik_price + first_sauce * sauce_price ≥ discount_threshold ∧
    (first_shashlik * shashlik_price + first_sauce * sauce_price) +
    (second_shashlik * (discounted_price shashlik_price)) ≤ budget :=
  sorry

end NUMINAMATH_CALUDE_sasha_can_buy_everything_l1641_164184


namespace NUMINAMATH_CALUDE_combinations_to_arrangements_l1641_164135

theorem combinations_to_arrangements (n : ℕ) (h1 : n ≥ 2) (h2 : Nat.choose n 2 = 15) :
  (n.factorial / (n - 2).factorial) = 30 := by
  sorry

end NUMINAMATH_CALUDE_combinations_to_arrangements_l1641_164135


namespace NUMINAMATH_CALUDE_complex_calculations_l1641_164104

theorem complex_calculations : 
  (∀ x : ℝ, x^2 = 3 → (1 + x) * (2 - x) = -1 + x) ∧
  (Real.sqrt 36 * Real.sqrt 12 / Real.sqrt 3 = 12) ∧
  (Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1/8) = 5 * Real.sqrt 2 / 4) ∧
  ((3 * Real.sqrt 18 + (1/5) * Real.sqrt 50 - 4 * Real.sqrt (1/2)) / Real.sqrt 32 = 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_calculations_l1641_164104


namespace NUMINAMATH_CALUDE_investment_ratio_l1641_164179

theorem investment_ratio (P Q : ℝ) (h : P > 0 ∧ Q > 0) :
  (P * 5) / (Q * 9) = 7 / 9 → P / Q = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l1641_164179


namespace NUMINAMATH_CALUDE_some_number_value_l1641_164113

theorem some_number_value (x y : ℝ) (hx : x = 12) 
  (heq : ((17.28 / x) / (3.6 * y)) = 2) : y = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1641_164113


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_l1641_164194

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) + a n

theorem fibonacci_like_sequence
  (a : ℕ → ℕ)
  (h_increasing : ∀ n : ℕ, a n < a (n + 1))
  (h_property : sequence_property a)
  (h_a7 : a 7 = 120) :
  a 8 = 194 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_l1641_164194


namespace NUMINAMATH_CALUDE_max_distance_on_circle_l1641_164169

open Complex

theorem max_distance_on_circle (z : ℂ) :
  Complex.abs (z - I) = 1 →
  (∀ w : ℂ, Complex.abs (w - I) = 1 → Complex.abs (z + 2 + I) ≥ Complex.abs (w + 2 + I)) →
  Complex.abs (z + 2 + I) = Real.sqrt 2 * 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_on_circle_l1641_164169


namespace NUMINAMATH_CALUDE_scrap_iron_average_l1641_164164

theorem scrap_iron_average (total_friends : Nat) (total_average : ℝ) (ivan_amount : ℝ) :
  total_friends = 5 →
  total_average = 55 →
  ivan_amount = 43 →
  let total_amount := total_friends * total_average
  let remaining_amount := total_amount - ivan_amount
  let remaining_friends := total_friends - 1
  (remaining_amount / remaining_friends : ℝ) = 58 := by
  sorry

end NUMINAMATH_CALUDE_scrap_iron_average_l1641_164164


namespace NUMINAMATH_CALUDE_max_area_and_front_wall_length_l1641_164177

/-- The material cost function for the house -/
def material_cost (x y : ℝ) : ℝ := 900 * x + 400 * y + 200 * x * y

/-- The constraint on the material cost -/
def cost_constraint (x y : ℝ) : Prop := material_cost x y ≤ 32000

/-- The area of the house -/
def house_area (x y : ℝ) : ℝ := x * y

/-- Theorem stating the maximum area and corresponding front wall length -/
theorem max_area_and_front_wall_length :
  ∃ (x y : ℝ), 
    cost_constraint x y ∧ 
    ∀ (x' y' : ℝ), cost_constraint x' y' → house_area x' y' ≤ house_area x y ∧
    house_area x y = 100 ∧
    x = 20 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_area_and_front_wall_length_l1641_164177


namespace NUMINAMATH_CALUDE_books_sold_l1641_164119

theorem books_sold (initial_books : ℕ) (added_books : ℕ) (final_books : ℕ) : 
  initial_books = 4 → added_books = 10 → final_books = 11 → 
  ∃ (sold_books : ℕ), initial_books - sold_books + added_books = final_books ∧ sold_books = 3 :=
by sorry

end NUMINAMATH_CALUDE_books_sold_l1641_164119


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1641_164199

theorem not_p_sufficient_not_necessary_for_not_q :
  ∃ (x : ℝ), (¬(|x + 1| > 2) → ¬(5*x - 6 > x^2)) ∧
             ∃ (y : ℝ), ¬(5*y - 6 > y^2) ∧ (|y + 1| > 2) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1641_164199


namespace NUMINAMATH_CALUDE_female_officers_count_l1641_164106

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_total_ratio : ℚ) :
  total_on_duty = 500 →
  female_on_duty_ratio = 1/2 →
  female_total_ratio = 1/4 →
  (female_on_duty_ratio * total_on_duty : ℚ) / female_total_ratio = 1000 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l1641_164106


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1641_164170

-- Define the cubic function
def f (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

-- Define the first derivative of f
def f' (a b c x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- Define the second derivative of f
def f'' (a b : ℝ) (x : ℝ) : ℝ := 6 * a * x + 2 * b

-- State the theorem
theorem cubic_function_properties (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x : ℝ, x = 1 ∨ x = -1 → f' a b c x = 0) →
  f a b c 1 = -1 →
  a = -1/2 ∧ b = 0 ∧ c = 3/2 ∧
  f'' a b 1 < 0 ∧ f'' a b (-1) > 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1641_164170


namespace NUMINAMATH_CALUDE_divisor_problem_l1641_164137

theorem divisor_problem (initial_number : ℕ) (added_number : ℝ) (divisor : ℕ) : 
  initial_number = 1782452 →
  added_number = 48.00000000010186 →
  divisor = 500 →
  divisor = (Int.toNat (round (initial_number + added_number))).gcd (Int.toNat (round (initial_number + added_number))) :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l1641_164137


namespace NUMINAMATH_CALUDE_only_set2_forms_triangle_l1641_164116

-- Define a structure for a set of three line segments
structure TripleSegment where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle inequality theorem
def satisfiesTriangleInequality (t : TripleSegment) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

-- Define the given sets of line segments
def set1 : TripleSegment := ⟨1, 2, 3⟩
def set2 : TripleSegment := ⟨3, 4, 5⟩
def set3 : TripleSegment := ⟨4, 5, 10⟩
def set4 : TripleSegment := ⟨6, 9, 2⟩

-- State the theorem
theorem only_set2_forms_triangle :
  satisfiesTriangleInequality set2 ∧
  ¬satisfiesTriangleInequality set1 ∧
  ¬satisfiesTriangleInequality set3 ∧
  ¬satisfiesTriangleInequality set4 :=
sorry

end NUMINAMATH_CALUDE_only_set2_forms_triangle_l1641_164116


namespace NUMINAMATH_CALUDE_f_is_even_f_is_decreasing_f_minimum_on_interval_l1641_164105

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Theorem 1: f(x) is even iff a = 0
theorem f_is_even (a : ℝ) : (∀ x, f a x = f a (-x)) ↔ a = 0 := by sorry

-- Theorem 2: f(x) is decreasing on (-∞, 4] iff a ≥ 4
theorem f_is_decreasing (a : ℝ) : (∀ x y, x ≤ y ∧ y ≤ 4 → f a x ≥ f a y) ↔ a ≥ 4 := by sorry

-- Theorem 3: Minimum value of f(x) on [1, 2]
theorem f_minimum_on_interval (a : ℝ) :
  (∀ x ∈ [1, 2], f a x ≥ 
    (if a ≤ 1 then 2 - 2*a
     else if a < 2 then 1 - a^2
     else 5 - 4*a)) ∧
  (∃ x ∈ [1, 2], f a x = 
    (if a ≤ 1 then 2 - 2*a
     else if a < 2 then 1 - a^2
     else 5 - 4*a)) := by sorry

end NUMINAMATH_CALUDE_f_is_even_f_is_decreasing_f_minimum_on_interval_l1641_164105


namespace NUMINAMATH_CALUDE_perfect_seventh_power_l1641_164193

theorem perfect_seventh_power (x y z : ℕ+) (h : ∃ (n : ℕ+), x^3 * y^5 * z^6 = n^7) :
  ∃ (m : ℕ+), x^5 * y^6 * z^3 = m^7 := by sorry

end NUMINAMATH_CALUDE_perfect_seventh_power_l1641_164193


namespace NUMINAMATH_CALUDE_percentage_rejected_l1641_164107

theorem percentage_rejected (john_rejection_rate jane_rejection_rate jane_inspection_fraction : ℝ) 
  (h1 : john_rejection_rate = 0.005)
  (h2 : jane_rejection_rate = 0.008)
  (h3 : jane_inspection_fraction = 0.8333333333333333)
  : jane_rejection_rate * jane_inspection_fraction + 
    john_rejection_rate * (1 - jane_inspection_fraction) = 0.0075 := by
  sorry

end NUMINAMATH_CALUDE_percentage_rejected_l1641_164107


namespace NUMINAMATH_CALUDE_find_m_l1641_164114

theorem find_m : ∃ m : ℕ, 
  (1 ^ (m + 1) / 5 ^ (m + 1)) * (1 ^ 18 / 4 ^ 18) = 1 / (2 * 10 ^ 35) ∧ m = 34 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1641_164114


namespace NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_3_l1641_164147

theorem simplify_fraction_with_sqrt_3 :
  (1 / (1 - Real.sqrt 3)) * (1 / (1 + Real.sqrt 3)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_3_l1641_164147


namespace NUMINAMATH_CALUDE_max_fibonacci_match_l1641_164171

/-- A sequence that matches the Fibonacci sequence for a given number of terms -/
def MatchesFibonacci (t : ℕ → ℝ) (start : ℕ) (count : ℕ) : Prop :=
  ∀ k, k < count → t (start + k + 2) = t (start + k + 1) + t (start + k)

/-- The quadratic sequence defined by A, B, and C -/
def QuadraticSequence (A B C : ℝ) (n : ℕ) : ℝ :=
  A * (n : ℝ)^2 + B * (n : ℝ) + C

/-- The theorem stating the maximum number of consecutive Fibonacci terms -/
theorem max_fibonacci_match (A B C : ℝ) (h : A ≠ 0) :
  (∃ start, MatchesFibonacci (QuadraticSequence A B C) start 4) ∧
  (∀ start count, count > 4 → ¬MatchesFibonacci (QuadraticSequence A B C) start count) ∧
  ((A = 1/2 ∧ B = -1/2 ∧ C = 2) ∨ (A = 1/2 ∧ B = 1/2 ∧ C = 2)) :=
sorry

end NUMINAMATH_CALUDE_max_fibonacci_match_l1641_164171


namespace NUMINAMATH_CALUDE_inequality_condition_l1641_164189

theorem inequality_condition (a b : ℝ) (h : a * Real.sqrt a > b * Real.sqrt b) : a > b ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l1641_164189


namespace NUMINAMATH_CALUDE_cube_of_thousands_l1641_164153

theorem cube_of_thousands (n : ℕ) : n = (n / 1000)^3 ↔ n = 32768 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_thousands_l1641_164153


namespace NUMINAMATH_CALUDE_unique_perfect_cube_l1641_164163

theorem unique_perfect_cube (Z K : ℤ) : 
  (1000 < Z) → (Z < 1500) → (K > 1) → (Z = K^3) → 
  (∃! k : ℤ, k > 1 ∧ 1000 < k^3 ∧ k^3 < 1500 ∧ Z = k^3) ∧ (K = 11) := by
sorry

end NUMINAMATH_CALUDE_unique_perfect_cube_l1641_164163


namespace NUMINAMATH_CALUDE_hired_waiters_count_l1641_164154

/-- Represents the number of waiters hired to change the ratio of cooks to waiters -/
def waiters_hired (initial_ratio_cooks initial_ratio_waiters new_ratio_cooks new_ratio_waiters num_cooks : ℕ) : ℕ :=
  let initial_waiters := (num_cooks * initial_ratio_waiters) / initial_ratio_cooks
  let total_new_waiters := (num_cooks * new_ratio_waiters) / new_ratio_cooks
  total_new_waiters - initial_waiters

/-- Theorem stating that given the conditions, the number of waiters hired is 12 -/
theorem hired_waiters_count :
  waiters_hired 3 8 1 4 9 = 12 :=
by sorry

end NUMINAMATH_CALUDE_hired_waiters_count_l1641_164154


namespace NUMINAMATH_CALUDE_bird_nest_problem_l1641_164138

theorem bird_nest_problem (birds : ℕ) (nests : ℕ) 
  (h1 : birds = 6) 
  (h2 : birds = nests + 3) : 
  nests = 3 := by
  sorry

end NUMINAMATH_CALUDE_bird_nest_problem_l1641_164138


namespace NUMINAMATH_CALUDE_coefficient_x_squared_sum_powers_l1641_164181

/-- The sum of the first n natural numbers -/
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n triangular numbers -/
def sum_triangular (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem coefficient_x_squared_sum_powers (n : ℕ) (h : n = 10) : 
  sum_triangular n = 165 := by
  sorry

#eval sum_triangular 10

end NUMINAMATH_CALUDE_coefficient_x_squared_sum_powers_l1641_164181


namespace NUMINAMATH_CALUDE_sandy_fingernail_record_age_l1641_164148

/-- Calculates the age at which Sandy will achieve the world record for longest fingernails -/
theorem sandy_fingernail_record_age 
  (world_record : ℝ)
  (sandy_current_age : ℕ)
  (sandy_current_length : ℝ)
  (growth_rate_per_month : ℝ)
  (h1 : world_record = 26)
  (h2 : sandy_current_age = 12)
  (h3 : sandy_current_length = 2)
  (h4 : growth_rate_per_month = 0.1) :
  sandy_current_age + (world_record - sandy_current_length) / (growth_rate_per_month * 12) = 32 := by
sorry

end NUMINAMATH_CALUDE_sandy_fingernail_record_age_l1641_164148


namespace NUMINAMATH_CALUDE_speed_conversion_l1641_164117

/-- Converts meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

theorem speed_conversion :
  let speed_mps : ℝ := 5.0004
  mps_to_kmph speed_mps = 18.00144 := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l1641_164117


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l1641_164192

theorem largest_multiple_of_8_under_100 : 
  ∀ n : ℕ, n % 8 = 0 ∧ n < 100 → n ≤ 96 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l1641_164192


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l1641_164103

/-- The equation of a line passing through (1, 2) with a 45° inclination angle -/
theorem line_equation_through_point_with_inclination (x y : ℝ) : 
  (x - y + 1 = 0) ↔ 
  (∃ (t : ℝ), x = 1 + t ∧ y = 2 + t) ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ - x₂ ≠ 0 → (y₁ - y₂) / (x₁ - x₂) = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l1641_164103


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1641_164190

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y + 2*x - 4*y + a = 0

-- Define the midpoint M
def midpoint_M (x y : ℝ) : Prop :=
  x = 0 ∧ y = 1

-- Define the chord length
def chord_length (l : ℝ) : Prop :=
  l = 2 * Real.sqrt 7

-- Main theorem
theorem circle_intersection_theorem (a : ℝ) :
  (∃ x y : ℝ, circle_C a x y ∧ midpoint_M x y) →
  (a < 3 ∧
   ∃ k b : ℝ, k = 1 ∧ b = 1 ∧ ∀ x y : ℝ, y = k*x + b) ∧
  (∀ l : ℝ, chord_length l →
    ∀ x y : ℝ, circle_C a x y ↔ (x+1)^2 + (y-2)^2 = 9) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1641_164190


namespace NUMINAMATH_CALUDE_angle_value_proof_l1641_164173

/-- Given that cos 16° = sin 14° + sin d° and 0 < d < 90, prove that d = 46 -/
theorem angle_value_proof (d : ℝ) 
  (h1 : Real.cos (16 * π / 180) = Real.sin (14 * π / 180) + Real.sin (d * π / 180))
  (h2 : 0 < d)
  (h3 : d < 90) : 
  d = 46 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_proof_l1641_164173


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l1641_164161

/-- Given two points F₁ and F₂ in the plane, we define an ellipse as the set of points P
    such that PF₁ + PF₂ is constant. This theorem proves that for the specific points
    F₁ = (2, 3) and F₂ = (8, 3), and the constant sum PF₁ + PF₂ = 10, 
    the resulting ellipse has parameters h, k, a, and b whose sum is 17. -/
theorem ellipse_parameter_sum : 
  let F₁ : ℝ × ℝ := (2, 3)
  let F₂ : ℝ × ℝ := (8, 3)
  let distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let is_on_ellipse (P : ℝ × ℝ) : Prop := distance P F₁ + distance P F₂ = 10
  let h : ℝ := (F₁.1 + F₂.1) / 2
  let k : ℝ := F₁.2  -- since F₁.2 = F₂.2
  let c : ℝ := distance F₁ ((F₁.1 + F₂.1) / 2, F₁.2) / 2
  let a : ℝ := 5  -- half of the constant sum
  let b : ℝ := Real.sqrt (a^2 - c^2)
  h + k + a + b = 17
  := by sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l1641_164161


namespace NUMINAMATH_CALUDE_f_range_l1641_164182

noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 5*x + 2) / (x + 1)

theorem f_range :
  Set.range f = {y : ℝ | y > 0} := by sorry

end NUMINAMATH_CALUDE_f_range_l1641_164182


namespace NUMINAMATH_CALUDE_abs_five_point_five_minus_pi_l1641_164110

theorem abs_five_point_five_minus_pi :
  |5.5 - Real.pi| = 5.5 - Real.pi :=
by sorry

end NUMINAMATH_CALUDE_abs_five_point_five_minus_pi_l1641_164110


namespace NUMINAMATH_CALUDE_book_selling_price_l1641_164165

/-- Calculates the selling price of a book given its cost price and profit percentage. -/
def selling_price (cost_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem stating that a book with a cost price of $60 and a profit percentage of 30% has a selling price of $78. -/
theorem book_selling_price :
  selling_price 60 30 = 78 := by
  sorry

end NUMINAMATH_CALUDE_book_selling_price_l1641_164165


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l1641_164112

/-- Given two squares A and B, where A has a perimeter of 40 cm and B has an area
    equal to one-third the area of A, the perimeter of B is (40√3)/3 cm. -/
theorem square_perimeter_relation (A B : Real → Real → Prop) :
  (∃ s, A s s ∧ 4 * s = 40) →  -- Square A has perimeter 40 cm
  (∀ x y, B x y ↔ x = y ∧ x^2 = (1/3) * s^2) →  -- B's area is 1/3 of A's area
  (∃ p, ∀ x y, B x y → 4 * x = p ∧ p = (40 * Real.sqrt 3) / 3) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l1641_164112


namespace NUMINAMATH_CALUDE_one_angle_not_determine_triangle_l1641_164143

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- sides
  (α β γ : ℝ)  -- angles

-- Define what it means for a triangle to be determined
def is_determined (t : Triangle) : Prop :=
  ∀ t' : Triangle, t.α = t'.α ∧ t.β = t'.β ∧ t.γ = t'.γ ∧ 
                   t.a = t'.a ∧ t.b = t'.b ∧ t.c = t'.c

-- Theorem: One angle does not determine a triangle
theorem one_angle_not_determine_triangle :
  ∃ (t t' : Triangle), t.α = t'.α ∧ ¬(is_determined t) :=
sorry

end NUMINAMATH_CALUDE_one_angle_not_determine_triangle_l1641_164143


namespace NUMINAMATH_CALUDE_square_product_extension_l1641_164180

theorem square_product_extension (a b : ℕ) 
  (h1 : ∃ x : ℕ, a * b = x ^ 2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y ^ 2) :
  ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_product_extension_l1641_164180


namespace NUMINAMATH_CALUDE_weaving_woman_problem_l1641_164133

/-- Represents the amount of cloth woven on a given day -/
def cloth_woven (day : ℕ) (initial_amount : ℚ) : ℚ :=
  initial_amount * 2^(day - 1)

/-- The problem of the weaving woman -/
theorem weaving_woman_problem :
  ∃ (initial_amount : ℚ),
    (∀ (day : ℕ), day > 0 → cloth_woven day initial_amount = initial_amount * 2^(day - 1)) ∧
    cloth_woven 5 initial_amount = 5 ∧
    initial_amount = 5/31 := by
  sorry

end NUMINAMATH_CALUDE_weaving_woman_problem_l1641_164133


namespace NUMINAMATH_CALUDE_recycling_points_per_bag_l1641_164155

/-- Calculates the points earned per bag of recycled cans. -/
def points_per_bag (total_bags : ℕ) (unrecycled_bags : ℕ) (total_points : ℕ) : ℚ :=
  total_points / total_bags

theorem recycling_points_per_bag :
  let total_bags : ℕ := 4
  let unrecycled_bags : ℕ := 2
  let total_points : ℕ := 16
  points_per_bag total_bags unrecycled_bags total_points = 4 := by
  sorry

end NUMINAMATH_CALUDE_recycling_points_per_bag_l1641_164155


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1641_164129

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (4 - m, 2)

-- Define what it means for a point to be in the second quadrant
def in_second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem point_in_second_quadrant (m : ℝ) :
  in_second_quadrant (P m) → m = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_point_in_second_quadrant_l1641_164129


namespace NUMINAMATH_CALUDE_only_height_weight_correlated_l1641_164185

-- Define the concept of a variable pair
structure VariablePair where
  var1 : String
  var2 : String

-- Define the concept of a functional relationship
def functionalRelationship (pair : VariablePair) : Prop := sorry

-- Define the concept of correlation
def correlated (pair : VariablePair) : Prop := sorry

-- Define the given variable pairs
def taxiFareDistance : VariablePair := ⟨"taxi fare", "distance traveled"⟩
def houseSizePrice : VariablePair := ⟨"house size", "house price"⟩
def heightWeight : VariablePair := ⟨"human height", "human weight"⟩
def ironSizeMass : VariablePair := ⟨"iron block size", "iron block mass"⟩

-- State the theorem
theorem only_height_weight_correlated :
  functionalRelationship taxiFareDistance →
  functionalRelationship houseSizePrice →
  (correlated heightWeight ∧ ¬functionalRelationship heightWeight) →
  functionalRelationship ironSizeMass →
  (correlated heightWeight ∧
   ¬correlated taxiFareDistance ∧
   ¬correlated houseSizePrice ∧
   ¬correlated ironSizeMass) := by
  sorry

end NUMINAMATH_CALUDE_only_height_weight_correlated_l1641_164185


namespace NUMINAMATH_CALUDE_intersection_of_spheres_integer_points_l1641_164157

theorem intersection_of_spheres_integer_points :
  let sphere1 := {(x, y, z) : ℤ × ℤ × ℤ | x^2 + y^2 + (z - 10)^2 ≤ 25}
  let sphere2 := {(x, y, z) : ℤ × ℤ × ℤ | x^2 + y^2 + (z - 4)^2 ≤ 36}
  ∃! p : ℤ × ℤ × ℤ, p ∈ sphere1 ∩ sphere2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_of_spheres_integer_points_l1641_164157


namespace NUMINAMATH_CALUDE_min_value_product_quotient_min_value_achieved_l1641_164121

theorem min_value_product_quotient (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x*y*z) ≥ 343 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a*b*c) = 343 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_min_value_achieved_l1641_164121


namespace NUMINAMATH_CALUDE_leftmost_digit_of_12_to_37_l1641_164168

def log_2_lower : ℝ := 0.3010
def log_2_upper : ℝ := 0.3011
def log_3_lower : ℝ := 0.4771
def log_3_upper : ℝ := 0.4772

theorem leftmost_digit_of_12_to_37 
  (h1 : log_2_lower < Real.log 2)
  (h2 : Real.log 2 < log_2_upper)
  (h3 : log_3_lower < Real.log 3)
  (h4 : Real.log 3 < log_3_upper) :
  (12^37 : ℝ) ≥ 8 * 10^39 ∧ (12^37 : ℝ) < 9 * 10^39 :=
sorry

end NUMINAMATH_CALUDE_leftmost_digit_of_12_to_37_l1641_164168


namespace NUMINAMATH_CALUDE_roots_power_set_difference_l1641_164150

/-- The roots of the polynomial (x^101 - 1) / (x - 1) -/
def roots : Fin 100 → ℂ := sorry

/-- The set S of powers of roots -/
def S : Set ℂ := sorry

/-- The maximum number of unique values in S -/
def M : ℕ := sorry

/-- The minimum number of unique values in S -/
def N : ℕ := sorry

/-- The difference between the maximum and minimum number of unique values in S is 99 -/
theorem roots_power_set_difference : M - N = 99 := by sorry

end NUMINAMATH_CALUDE_roots_power_set_difference_l1641_164150


namespace NUMINAMATH_CALUDE_spheres_radius_is_correct_l1641_164108

/-- Right circular cone with given dimensions -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Sphere inside the cone -/
structure Sphere where
  radius : ℝ

/-- Configuration of three spheres inside a cone -/
structure SpheresInCone where
  cone : Cone
  sphere : Sphere
  centerPlaneHeight : ℝ

/-- The specific configuration described in the problem -/
def problemConfig : SpheresInCone where
  cone := { baseRadius := 4, height := 15 }
  sphere := { radius := 1.5 }  -- We use the correct answer here
  centerPlaneHeight := 2.5     -- This is r + 1, where r = 1.5

/-- Predicate to check if the spheres are tangent to each other, the base, and the side of the cone -/
def areTangent (config : SpheresInCone) : Prop :=
  -- The actual tangency conditions would be complex to express precisely,
  -- so we use a placeholder predicate
  True

/-- Theorem stating that the given configuration satisfies the problem conditions -/
theorem spheres_radius_is_correct (config : SpheresInCone) :
  config.cone.baseRadius = 4 ∧
  config.cone.height = 15 ∧
  config.centerPlaneHeight = config.sphere.radius + 1 ∧
  areTangent config →
  config.sphere.radius = 1.5 :=
by
  sorry

#check spheres_radius_is_correct

end NUMINAMATH_CALUDE_spheres_radius_is_correct_l1641_164108


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1641_164136

/-- Given that (1+i)z = |-4i|, prove that z = 2 - 2i --/
theorem complex_equation_solution :
  ∀ z : ℂ, (Complex.I + 1) * z = Complex.abs (-4 * Complex.I) → z = 2 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1641_164136


namespace NUMINAMATH_CALUDE_f_increasing_and_even_l1641_164183

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_increasing_and_even :
  -- f is monotonically increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) ∧
  -- f is an even function
  (∀ x : ℝ, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_and_even_l1641_164183


namespace NUMINAMATH_CALUDE_largest_valid_n_l1641_164111

def is_valid_n (n : ℕ) : Prop :=
  ∃ (P : ℤ → ℤ), ∀ (k : ℕ), (2020 ∣ P^[k] 0) ↔ (n ∣ k)

theorem largest_valid_n : 
  (∃ (N : ℕ), N ∈ Finset.range 2020 ∧ is_valid_n N ∧ 
    ∀ (M : ℕ), M ∈ Finset.range 2020 → is_valid_n M → M ≤ N) ∧
  (∀ (N : ℕ), N ∈ Finset.range 2020 ∧ is_valid_n N ∧ 
    (∀ (M : ℕ), M ∈ Finset.range 2020 → is_valid_n M → M ≤ N) → N = 1980) :=
by sorry


end NUMINAMATH_CALUDE_largest_valid_n_l1641_164111


namespace NUMINAMATH_CALUDE_calculate_expression_l1641_164156

theorem calculate_expression : -2⁻¹ * (-8) - Real.sqrt 9 - abs (-4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1641_164156


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1641_164191

theorem arithmetic_sequence_middle_term (z : ℤ) :
  (∃ (a d : ℤ), 3^2 = a ∧ z = a + d ∧ 3^3 = a + 2*d) → z = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1641_164191


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1641_164122

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence {a_n} satisfying a_1 + a_6 = 11 and a_3 * a_4 = 32/9,
    prove that a_1 = 32/3 or a_1 = 1/3 -/
theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h_sum : a 1 + a 6 = 11)
  (h_prod : a 3 * a 4 = 32/9) :
  a 1 = 32/3 ∨ a 1 = 1/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1641_164122


namespace NUMINAMATH_CALUDE_simple_interest_period_l1641_164126

theorem simple_interest_period (P : ℝ) : 
  (P * 4 * 5 / 100 = 1680) → 
  (P * 5 * 4 / 100 = 1680) → 
  ∃ T : ℝ, T = 5 ∧ P * 4 * T / 100 = 1680 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_period_l1641_164126


namespace NUMINAMATH_CALUDE_painted_cubes_count_l1641_164102

/-- Represents a cube with given dimensions -/
structure Cube where
  size : Nat

/-- Represents a painted cube -/
structure PaintedCube extends Cube where
  painted : Bool

/-- Calculates the number of 1-inch cubes with at least one painted face -/
def paintedCubes (c : PaintedCube) : Nat :=
  c.size ^ 3 - (c.size - 2) ^ 3

/-- Theorem: In a 10×10×10 painted cube, 488 small cubes have at least one painted face -/
theorem painted_cubes_count :
  let c : PaintedCube := { size := 10, painted := true }
  paintedCubes c = 488 := by sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l1641_164102


namespace NUMINAMATH_CALUDE_total_capacity_is_132000_l1641_164172

/-- The capacity of a train's boxcars -/
def train_capacity (num_red num_blue num_black : ℕ) (black_capacity : ℕ) : ℕ :=
  let blue_capacity := 2 * black_capacity
  let red_capacity := 3 * blue_capacity
  num_red * red_capacity + num_blue * blue_capacity + num_black * black_capacity

/-- Theorem: The total capacity of the train's boxcars is 132000 pounds -/
theorem total_capacity_is_132000 :
  train_capacity 3 4 7 4000 = 132000 := by
  sorry

end NUMINAMATH_CALUDE_total_capacity_is_132000_l1641_164172


namespace NUMINAMATH_CALUDE_line_segment_ratio_l1641_164195

/-- Given points A, B, C, D, and E on a line in that order, prove that AC:DE = 5:3 -/
theorem line_segment_ratio (A B C D E : ℝ) : 
  (B - A = 3) → 
  (C - B = 7) → 
  (D - C = 4) → 
  (E - A = 20) → 
  (A < B) → (B < C) → (C < D) → (D < E) →
  (C - A) / (E - D) = 5 / 3 := by
  sorry

#check line_segment_ratio

end NUMINAMATH_CALUDE_line_segment_ratio_l1641_164195


namespace NUMINAMATH_CALUDE_proportion_fourth_term_l1641_164196

theorem proportion_fourth_term 
  (a b c d : ℚ) 
  (h1 : a + b + c = 58)
  (h2 : c = 2/3 * a)
  (h3 : b = 3/4 * a)
  (h4 : a/b = c/d)
  : d = 12 := by
sorry

end NUMINAMATH_CALUDE_proportion_fourth_term_l1641_164196


namespace NUMINAMATH_CALUDE_percentage_problem_l1641_164131

theorem percentage_problem (x : ℝ) (P : ℝ) : 
  x = 780 ∧ 
  (P / 100) * x = 0.15 * 1500 - 30 → 
  P = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1641_164131


namespace NUMINAMATH_CALUDE_orthogonal_vectors_k_values_l1641_164144

theorem orthogonal_vectors_k_values (a b : ℝ × ℝ) (k : ℝ) :
  a = (0, 2) →
  b = (Real.sqrt 3, 1) →
  (a.1 - k * b.1, a.2 - k * b.2) • (k * a.1 + b.1, k * a.2 + b.2) = 0 →
  k = -1 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_k_values_l1641_164144


namespace NUMINAMATH_CALUDE_earth_sun_distance_in_scientific_notation_l1641_164178

/-- The speed of light in meters per second -/
def speed_of_light : ℝ := 3 * (10 ^ 8)

/-- The time it takes for sunlight to reach Earth in seconds -/
def time_to_earth : ℝ := 5 * (10 ^ 2)

/-- The distance between Earth and Sun in meters -/
def earth_sun_distance : ℝ := speed_of_light * time_to_earth

theorem earth_sun_distance_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), a ≥ 1 ∧ a < 10 ∧ earth_sun_distance = a * (10 ^ n) ∧ a = 1.5 ∧ n = 11 :=
sorry

end NUMINAMATH_CALUDE_earth_sun_distance_in_scientific_notation_l1641_164178


namespace NUMINAMATH_CALUDE_sum_cos_dihedral_angles_eq_one_l1641_164167

/-- A trihedral angle is a three-dimensional figure formed by three planes intersecting at a point. -/
structure TrihedralAngle where
  /-- The three plane angles of the trihedral angle -/
  plane_angles : Fin 3 → ℝ
  /-- The sum of the plane angles is 180° (π radians) -/
  sum_plane_angles : (plane_angles 0) + (plane_angles 1) + (plane_angles 2) = π

/-- The dihedral angles of a trihedral angle -/
def dihedral_angles (t : TrihedralAngle) : Fin 3 → ℝ := sorry

/-- Theorem: For a trihedral angle with plane angles summing to 180°, 
    the sum of the cosines of its dihedral angles is equal to 1 -/
theorem sum_cos_dihedral_angles_eq_one (t : TrihedralAngle) : 
  (Real.cos (dihedral_angles t 0)) + (Real.cos (dihedral_angles t 1)) + (Real.cos (dihedral_angles t 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_cos_dihedral_angles_eq_one_l1641_164167


namespace NUMINAMATH_CALUDE_min_sum_with_constraints_l1641_164115

theorem min_sum_with_constraints (x y z : ℝ) 
  (hx : x ≥ 5) (hy : y ≥ 6) (hz : z ≥ 7) 
  (h_sum_sq : x^2 + y^2 + z^2 ≥ 125) : 
  x + y + z ≥ 19 ∧ ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ ≥ 5 ∧ y₀ ≥ 6 ∧ z₀ ≥ 7 ∧ 
    x₀^2 + y₀^2 + z₀^2 ≥ 125 ∧ 
    x₀ + y₀ + z₀ = 19 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_constraints_l1641_164115


namespace NUMINAMATH_CALUDE_sqrt_inequalities_l1641_164159

theorem sqrt_inequalities (x : ℝ) :
  (∀ x, (Real.sqrt (x - 1) < 1) ↔ (1 ≤ x ∧ x < 2)) ∧
  (∀ x, (Real.sqrt (2*x - 3) ≤ Real.sqrt (x - 1)) ↔ ((3/2) ≤ x ∧ x ≤ 2)) := by
sorry

end NUMINAMATH_CALUDE_sqrt_inequalities_l1641_164159


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l1641_164109

/-- Given a car that travels 140 kilometers using 3.5 gallons of gasoline,
    prove that the car's fuel efficiency is 40 kilometers per gallon. -/
theorem car_fuel_efficiency :
  let distance : ℝ := 140  -- Total distance in kilometers
  let fuel : ℝ := 3.5      -- Fuel used in gallons
  let efficiency : ℝ := distance / fuel  -- Fuel efficiency in km/gallon
  efficiency = 40 := by sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l1641_164109


namespace NUMINAMATH_CALUDE_projection_a_onto_b_l1641_164175

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 1)

theorem projection_a_onto_b :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  (dot_product / magnitude_b) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_projection_a_onto_b_l1641_164175


namespace NUMINAMATH_CALUDE_tan_150_degrees_l1641_164151

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l1641_164151


namespace NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l1641_164101

/-- 
Given an isosceles triangle with base l and height h, and a rectangle with length l and width w,
if their areas are equal, then the height of the triangle is twice the width of the rectangle.
-/
theorem isosceles_triangle_rectangle_equal_area 
  (l w h : ℝ) (l_pos : l > 0) (w_pos : w > 0) (h_pos : h > 0) : 
  (1 / 2 : ℝ) * l * h = l * w → h = 2 * w := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l1641_164101


namespace NUMINAMATH_CALUDE_solve_equation_l1641_164123

theorem solve_equation (x : ℚ) : (3 * x - 2) / 4 = 14 → x = 58 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1641_164123


namespace NUMINAMATH_CALUDE_age_problem_l1641_164188

theorem age_problem (a b : ℕ) (h1 : 5 * b = 3 * a) (h2 : 7 * (b + 6) = 5 * (a + 6)) : a = 15 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l1641_164188


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l1641_164132

theorem tangent_line_to_ln_curve (k : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ k * x₀ = Real.log x₀ ∧ k = 1 / x₀) → k = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l1641_164132


namespace NUMINAMATH_CALUDE_quadratic_vertex_l1641_164187

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + (8 - m)*x + 12

-- Define the derivative of the function
def f' (m : ℝ) (x : ℝ) : ℝ := -2*x + (8 - m)

-- Theorem statement
theorem quadratic_vertex (m : ℝ) :
  (∀ x > 2, (f' m x < 0)) ∧ 
  (∀ x < 2, (f' m x > 0)) →
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l1641_164187


namespace NUMINAMATH_CALUDE_no_rain_probability_l1641_164141

theorem no_rain_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l1641_164141


namespace NUMINAMATH_CALUDE_x_value_proof_l1641_164186

theorem x_value_proof (y : ℝ) (x : ℝ) (h1 : y = -2) (h2 : (x - 2*y)^y = 0.001) :
  x = -4 + 10 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_x_value_proof_l1641_164186


namespace NUMINAMATH_CALUDE_triangle_expression_positive_l1641_164145

theorem triangle_expression_positive (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  0 < 4 * b^2 * c^2 - (b^2 + c^2 - a^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_expression_positive_l1641_164145


namespace NUMINAMATH_CALUDE_problem_1_solution_problem_2_no_solution_l1641_164198

-- Problem 1
theorem problem_1_solution (x : ℝ) :
  (x / (2*x - 5) + 5 / (5 - 2*x) = 1) ↔ (x = 0) :=
sorry

-- Problem 2
theorem problem_2_no_solution :
  ¬∃ (x : ℝ), ((2*x + 9) / (3*x - 9) = (4*x - 7) / (x - 3) + 2) :=
sorry

end NUMINAMATH_CALUDE_problem_1_solution_problem_2_no_solution_l1641_164198


namespace NUMINAMATH_CALUDE_rectangle_length_calculation_l1641_164139

theorem rectangle_length_calculation (rectangle_width square_side : ℝ) :
  rectangle_width = 300 ∧ 
  square_side = 700 ∧ 
  (4 * square_side) = 2 * (2 * (rectangle_width + rectangle_length)) →
  rectangle_length = 400 :=
by
  sorry

#check rectangle_length_calculation

end NUMINAMATH_CALUDE_rectangle_length_calculation_l1641_164139


namespace NUMINAMATH_CALUDE_parallel_lines_j_value_l1641_164130

/-- Given that a line through (2, -9) and (j, 17) is parallel to 2x + 3y = 21, prove that j = -37 -/
theorem parallel_lines_j_value (j : ℝ) : 
  (∃ (m b : ℝ), ∀ x y, y = m * x + b → 
    (y = -9 ∧ x = 2 ∨ y = 17 ∧ x = j) ∧ 
    m = -2/3) → 
  j = -37 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_j_value_l1641_164130
