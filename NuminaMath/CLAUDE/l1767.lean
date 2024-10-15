import Mathlib

namespace NUMINAMATH_CALUDE_smallestPalindromeNumber_satisfies_conditions_l1767_176769

/-- A function to check if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  (n.digits base).reverse = n.digits base

/-- The smallest positive integer greater than 10 that is a palindrome in base 2 and 4, and is odd -/
def smallestPalindromeNumber : ℕ := 17

/-- Theorem stating that smallestPalindromeNumber satisfies all conditions -/
theorem smallestPalindromeNumber_satisfies_conditions :
  smallestPalindromeNumber > 10 ∧
  isPalindrome smallestPalindromeNumber 2 ∧
  isPalindrome smallestPalindromeNumber 4 ∧
  Odd smallestPalindromeNumber ∧
  ∀ n : ℕ, n > 10 → isPalindrome n 2 → isPalindrome n 4 → Odd n →
    n ≥ smallestPalindromeNumber :=
by sorry

#eval smallestPalindromeNumber

end NUMINAMATH_CALUDE_smallestPalindromeNumber_satisfies_conditions_l1767_176769


namespace NUMINAMATH_CALUDE_jenny_lasagna_profit_l1767_176774

/-- Calculates the profit for Jenny's lasagna business -/
def lasagna_profit (cost_per_pan : ℝ) (num_pans : ℕ) (price_per_pan : ℝ) : ℝ :=
  num_pans * price_per_pan - num_pans * cost_per_pan

/-- Proves that Jenny's profit is $300.00 given the specified conditions -/
theorem jenny_lasagna_profit :
  lasagna_profit 10 20 25 = 300 :=
by sorry

end NUMINAMATH_CALUDE_jenny_lasagna_profit_l1767_176774


namespace NUMINAMATH_CALUDE_wendy_sold_nine_pastries_l1767_176765

/-- The number of pastries Wendy sold at the bake sale -/
def pastries_sold (cupcakes cookies leftover : ℕ) : ℕ :=
  cupcakes + cookies - leftover

/-- Proof that Wendy sold 9 pastries at the bake sale -/
theorem wendy_sold_nine_pastries :
  pastries_sold 4 29 24 = 9 := by
  sorry

end NUMINAMATH_CALUDE_wendy_sold_nine_pastries_l1767_176765


namespace NUMINAMATH_CALUDE_min_rectangles_to_cover_square_l1767_176786

-- Define the dimensions of the rectangle
def rectangle_width : ℕ := 3
def rectangle_height : ℕ := 4

-- Define the area of the rectangle
def rectangle_area : ℕ := rectangle_width * rectangle_height

-- Define the function to calculate the number of rectangles needed
def rectangles_needed (square_side : ℕ) : ℕ :=
  (square_side * square_side) / rectangle_area

-- Theorem statement
theorem min_rectangles_to_cover_square :
  ∃ (n : ℕ), 
    n > 0 ∧
    rectangles_needed n = 12 ∧
    ∀ (m : ℕ), m > 0 → rectangles_needed m ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_rectangles_to_cover_square_l1767_176786


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1767_176782

-- Define the radius of the larger circle
def R : ℝ := 10

-- Define the radius of the smaller circles
def r : ℝ := 5

-- Theorem statement
theorem shaded_area_calculation :
  let larger_circle_area := π * R^2
  let smaller_circle_area := π * r^2
  let shaded_area := larger_circle_area - 2 * smaller_circle_area
  shaded_area = 50 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1767_176782


namespace NUMINAMATH_CALUDE_inequality_proof_l1767_176729

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1767_176729


namespace NUMINAMATH_CALUDE_difference_of_squares_division_problem_solution_l1767_176798

theorem difference_of_squares_division (a b : ℕ) (h : a > b) : 
  (a ^ 2 - b ^ 2) / (a - b) = a + b := by sorry

theorem problem_solution : (125 ^ 2 - 117 ^ 2) / 8 = 242 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_problem_solution_l1767_176798


namespace NUMINAMATH_CALUDE_lucky_iff_power_of_two_l1767_176789

/-- Represents the three colors of cubes -/
inductive Color
  | White
  | Blue
  | Red

/-- Represents an arrangement of N cubes in a circle -/
def Arrangement (N : ℕ) := Fin N → Color

/-- Determines if an arrangement is good (final cube color doesn't depend on starting position) -/
def is_good (N : ℕ) (arr : Arrangement N) : Prop := sorry

/-- Determines if N is lucky (all arrangements of N cubes are good) -/
def is_lucky (N : ℕ) : Prop :=
  ∀ arr : Arrangement N, is_good N arr

/-- Main theorem: N is lucky if and only if it's a power of 2 -/
theorem lucky_iff_power_of_two (N : ℕ) :
  is_lucky N ↔ ∃ k : ℕ, N = 2^k :=
sorry

end NUMINAMATH_CALUDE_lucky_iff_power_of_two_l1767_176789


namespace NUMINAMATH_CALUDE_remainder_problem_l1767_176713

theorem remainder_problem (n : ℤ) : 
  ∃ (r : ℕ), r < 25 ∧ 
  n % 25 = r ∧ 
  (n + 15) % 5 = r % 5 → 
  r = 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1767_176713


namespace NUMINAMATH_CALUDE_set_operations_l1767_176796

-- Define the sets A and B
def A : Set ℝ := {x | x > 4}
def B : Set ℝ := {x | -6 < x ∧ x < 6}

-- Define the set difference operation
def setDiff (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

theorem set_operations :
  (A ∩ B = {x | 4 < x ∧ x < 6}) ∧
  (Bᶜ = {x | x ≥ 6 ∨ x ≤ -6}) ∧
  (setDiff A B = {x | x ≥ 6}) ∧
  (setDiff A (setDiff A B) = {x | 4 < x ∧ x < 6}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1767_176796


namespace NUMINAMATH_CALUDE_f_properties_l1767_176738

def f (x : ℝ) : ℝ := x * (x + 1) * (x - 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x > 2, ∀ y > x, f y > f x) ∧
  (∃! a b c, a < b ∧ b < c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1767_176738


namespace NUMINAMATH_CALUDE_union_when_a_is_4_intersection_equals_B_l1767_176797

def A : Set ℝ := {x | x^2 - 5*x - 14 < 0}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3*a - 2}

theorem union_when_a_is_4 : A ∪ B 4 = {x | -2 < x ∧ x ≤ 10} := by sorry

theorem intersection_equals_B (a : ℝ) : A ∩ B a = B a ↔ a < 3 := by sorry

end NUMINAMATH_CALUDE_union_when_a_is_4_intersection_equals_B_l1767_176797


namespace NUMINAMATH_CALUDE_expression_evaluation_l1767_176749

theorem expression_evaluation : 
  (2023^3 - 3 * 2023^2 * 2024 + 5 * 2023 * 2024^2 - 2024^3 + 5) / (2023 * 2024) = 4048 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1767_176749


namespace NUMINAMATH_CALUDE_min_value_on_line_l1767_176750

/-- Given real numbers x and y satisfying the equation x + 2y + 3 = 0,
    the minimum value of √(x² + y² - 2y + 1) is √5. -/
theorem min_value_on_line (x y : ℝ) (h : x + 2*y + 3 = 0) :
  ∃ (m : ℝ), m = Real.sqrt 5 ∧ ∀ (x' y' : ℝ), x' + 2*y' + 3 = 0 →
    m ≤ Real.sqrt (x'^2 + y'^2 - 2*y' + 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_line_l1767_176750


namespace NUMINAMATH_CALUDE_angle_1120_in_first_quadrant_l1767_176745

/-- An angle is in the first quadrant if its equivalent angle in [0, 360) is between 0 and 90 degrees. -/
def in_first_quadrant (angle : ℝ) : Prop :=
  0 ≤ (angle % 360) ∧ (angle % 360) < 90

/-- Theorem stating that 1120 degrees is in the first quadrant -/
theorem angle_1120_in_first_quadrant : in_first_quadrant 1120 := by
  sorry

end NUMINAMATH_CALUDE_angle_1120_in_first_quadrant_l1767_176745


namespace NUMINAMATH_CALUDE_pauls_crayons_l1767_176792

/-- Given Paul's crayon situation, prove the difference between given and lost crayons. -/
theorem pauls_crayons (initial : ℕ) (given : ℕ) (lost : ℕ) 
  (h1 : initial = 589) 
  (h2 : given = 571) 
  (h3 : lost = 161) : 
  given - lost = 410 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_l1767_176792


namespace NUMINAMATH_CALUDE_a_neither_sufficient_nor_necessary_for_b_l1767_176784

/-- Proposition A: The complex number z satisfies |z-3|+|z+3| is a constant -/
def propositionA (z : ℂ) : Prop :=
  ∃ c : ℝ, ∀ z : ℂ, Complex.abs (z - 3) + Complex.abs (z + 3) = c

/-- Proposition B: The trajectory of the point corresponding to the complex number z in the complex plane is an ellipse -/
def propositionB (z : ℂ) : Prop :=
  ∃ a b : ℝ, ∃ f₁ f₂ : ℂ, ∀ z : ℂ, Complex.abs (z - f₁) + Complex.abs (z - f₂) = a + b

/-- A is neither sufficient nor necessary for B -/
theorem a_neither_sufficient_nor_necessary_for_b :
  (¬∀ z : ℂ, propositionA z → propositionB z) ∧
  (¬∀ z : ℂ, propositionB z → propositionA z) :=
sorry

end NUMINAMATH_CALUDE_a_neither_sufficient_nor_necessary_for_b_l1767_176784


namespace NUMINAMATH_CALUDE_five_letter_words_count_l1767_176759

def alphabet_size : ℕ := 26
def vowel_count : ℕ := 5

theorem five_letter_words_count : 
  (alphabet_size^3 * vowel_count : ℕ) = 87880 := by
sorry

end NUMINAMATH_CALUDE_five_letter_words_count_l1767_176759


namespace NUMINAMATH_CALUDE_log_10_14_in_terms_of_r_and_s_l1767_176732

theorem log_10_14_in_terms_of_r_and_s (r s : ℝ) 
  (h1 : Real.log 2 / Real.log 9 = r) 
  (h2 : Real.log 7 / Real.log 2 = s) : 
  Real.log 14 / Real.log 10 = (s + 1) / (3 + 1 / (2 * r)) := by
  sorry

end NUMINAMATH_CALUDE_log_10_14_in_terms_of_r_and_s_l1767_176732


namespace NUMINAMATH_CALUDE_savings_percentage_is_twenty_percent_l1767_176737

def monthly_salary : ℝ := 6250
def savings_after_increase : ℝ := 250
def expense_increase_rate : ℝ := 0.2

theorem savings_percentage_is_twenty_percent :
  ∃ P : ℝ, 
    savings_after_increase = monthly_salary - (1 + expense_increase_rate) * (monthly_salary - (P / 100) * monthly_salary) ∧
    P = 20 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_is_twenty_percent_l1767_176737


namespace NUMINAMATH_CALUDE_odd_function_zero_l1767_176715

/-- Definition of an odd function -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Theorem: For an odd function f defined at 0, f(0) = 0 -/
theorem odd_function_zero (f : ℝ → ℝ) (h : IsOdd f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_zero_l1767_176715


namespace NUMINAMATH_CALUDE_odd_number_product_difference_l1767_176779

theorem odd_number_product_difference (x : ℤ) : 
  Odd x → x * (x + 2) - x * (x - 2) = 44 → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_product_difference_l1767_176779


namespace NUMINAMATH_CALUDE_max_garden_area_l1767_176746

/-- Represents a rectangular garden with given constraints -/
structure Garden where
  width : ℝ
  length : ℝ
  fence_length : ℝ
  fence_constraint : length + 2 * width = fence_length
  size_constraint : length ≥ 2 * width

/-- The area of a garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- The maximum area of a garden given the constraints -/
theorem max_garden_area :
  ∃ (g : Garden), g.fence_length = 480 ∧ 
    (∀ (h : Garden), h.fence_length = 480 → g.area ≥ h.area) ∧
    g.area = 28800 := by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l1767_176746


namespace NUMINAMATH_CALUDE_baguette_cost_is_two_l1767_176712

/-- The cost of a single baguette given the initial amount, number of items bought,
    cost of water, and remaining amount after purchase. -/
def baguette_cost (initial_amount : ℚ) (num_baguettes : ℕ) (num_water : ℕ) 
                  (water_cost : ℚ) (remaining_amount : ℚ) : ℚ :=
  (initial_amount - remaining_amount - num_water * water_cost) / num_baguettes

/-- Theorem stating that the cost of each baguette is $2 given the problem conditions. -/
theorem baguette_cost_is_two :
  baguette_cost 50 2 2 1 44 = 2 := by
  sorry

end NUMINAMATH_CALUDE_baguette_cost_is_two_l1767_176712


namespace NUMINAMATH_CALUDE_percentage_error_division_vs_multiplication_l1767_176706

theorem percentage_error_division_vs_multiplication (x : ℝ) : 
  let correct_result := 10 * x
  let incorrect_result := x / 10
  let error := correct_result - incorrect_result
  let percentage_error := (error / correct_result) * 100
  percentage_error = 99 := by
sorry

end NUMINAMATH_CALUDE_percentage_error_division_vs_multiplication_l1767_176706


namespace NUMINAMATH_CALUDE_intersection_property_l1767_176709

/-- Given a function f(x) = |sin x| and a line y = kx (k > 0) that intersect at exactly three points,
    with the maximum x-coordinate of the intersections being α, prove that:
    cos α / (sin α + sin 3α) = (1 + α²) / (4α) -/
theorem intersection_property (k α : ℝ) (hk : k > 0) 
    (h_intersections : ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ α ∧
      (∀ x, k * x = |Real.sin x| ↔ x = x₁ ∨ x = x₂ ∨ x = x₃))
    (h_max : ∀ x, k * x = |Real.sin x| → x ≤ α) :
  Real.cos α / (Real.sin α + Real.sin (3 * α)) = (1 + α^2) / (4 * α) := by
  sorry

end NUMINAMATH_CALUDE_intersection_property_l1767_176709


namespace NUMINAMATH_CALUDE_same_solution_implies_k_value_l1767_176751

theorem same_solution_implies_k_value (x : ℝ) (k : ℝ) : 
  (2 * x - 1 = 3) ∧ (3 * x + k = 0) → k = -6 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_k_value_l1767_176751


namespace NUMINAMATH_CALUDE_fibonacci_eighth_term_l1767_176781

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_eighth_term : fibonacci 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_eighth_term_l1767_176781


namespace NUMINAMATH_CALUDE_fruit_arrangement_theorem_l1767_176741

def num_apples : ℕ := 4
def num_oranges : ℕ := 3
def num_bananas : ℕ := 2
def total_fruits : ℕ := num_apples + num_oranges + num_bananas

-- Function to calculate the number of ways to arrange fruits
-- with the constraint that not all apples are consecutive
def arrange_fruits (a o b : ℕ) : ℕ := sorry

theorem fruit_arrangement_theorem :
  arrange_fruits num_apples num_oranges num_bananas = 150 := by sorry

end NUMINAMATH_CALUDE_fruit_arrangement_theorem_l1767_176741


namespace NUMINAMATH_CALUDE_inequality_proof_l1767_176722

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (sum_eq : a + b = c + d) (prod_gt : a * b > c * d) : 
  (Real.sqrt a + Real.sqrt b > Real.sqrt c + Real.sqrt d) ∧ 
  (|a - b| < |c - d|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1767_176722


namespace NUMINAMATH_CALUDE_inequality_holds_iff_k_in_range_l1767_176724

theorem inequality_holds_iff_k_in_range :
  ∀ k : ℝ, (∀ x : ℝ, k * x^2 + k * x - 3/4 < 0) ↔ k ∈ Set.Ioc (-3) 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_k_in_range_l1767_176724


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1767_176793

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 7*x + 6
  (f 1 = 0) ∧ (f 6 = 0) ∧
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1767_176793


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1767_176787

/-- 
An arithmetic sequence is defined by its first term, common difference, and last term.
This theorem proves that for an arithmetic sequence with first term 15, 
common difference 4, and last term 159, the number of terms is 37.
-/
theorem arithmetic_sequence_terms (first_term : ℕ) (common_diff : ℕ) (last_term : ℕ) :
  first_term = 15 → common_diff = 4 → last_term = 159 →
  (last_term - first_term) / common_diff + 1 = 37 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1767_176787


namespace NUMINAMATH_CALUDE_series_sum_l1767_176716

theorem series_sum : 
  (3/4 : ℚ) + 5/8 + 9/16 + 17/32 + 33/64 + 65/128 - (7/2 : ℚ) = -1/128 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l1767_176716


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l1767_176719

theorem difference_of_squares_example : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l1767_176719


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1767_176714

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the modified quadratic function
def g (a b c : ℝ) (x : ℝ) := a * (x^2 + 1) + b * (x - 1) + c - 2 * a * x

theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) :
  (∀ x : ℝ, f a b c x > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x : ℝ, g a b c x > 0 ↔ 0 < x ∧ x < 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1767_176714


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1767_176762

-- Define the function f
def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- Define the derivative of f
def f' (b c : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*b*x + c

theorem cubic_function_properties (b c d : ℝ) :
  (∀ k, (k < 0 ∨ k > 4) → (∃! x, f b c d x = k)) ∧
  (∀ k, (0 < k ∧ k < 4) → (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f b c d x = k ∧ f b c d y = k ∧ f b c d z = k)) →
  (∃ x, f b c d x = 4 ∧ f' b c x = 0) ∧
  (∃ x, f b c d x = 0 ∧ f' b c x = 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1767_176762


namespace NUMINAMATH_CALUDE_lateral_area_of_specific_prism_l1767_176731

/-- A prism with a square base and a circumscribed sphere -/
structure SquareBasePrism where
  /-- Side length of the square base -/
  baseSide : ℝ
  /-- Height of the prism -/
  height : ℝ
  /-- Volume of the circumscribed sphere -/
  sphereVolume : ℝ

/-- Theorem: The lateral area of a square-based prism with circumscribed sphere volume 4π/3 and base side length 1 is 4√2 -/
theorem lateral_area_of_specific_prism (p : SquareBasePrism) 
  (h1 : p.baseSide = 1)
  (h2 : p.sphereVolume = 4 * Real.pi / 3) : 
  4 * p.baseSide * p.height = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_lateral_area_of_specific_prism_l1767_176731


namespace NUMINAMATH_CALUDE_correct_division_result_l1767_176772

theorem correct_division_result (incorrect_divisor correct_divisor incorrect_quotient : ℕ)
  (h1 : incorrect_divisor = 63)
  (h2 : correct_divisor = 36)
  (h3 : incorrect_quotient = 24) :
  (incorrect_divisor * incorrect_quotient) / correct_divisor = 42 := by
sorry

end NUMINAMATH_CALUDE_correct_division_result_l1767_176772


namespace NUMINAMATH_CALUDE_mother_daughter_age_relation_l1767_176711

theorem mother_daughter_age_relation : 
  ∀ (mother_current_age daughter_future_age : ℕ),
    mother_current_age = 41 →
    daughter_future_age = 26 →
    ∃ (years_ago : ℕ),
      years_ago = 5 ∧
      mother_current_age - years_ago = 2 * (daughter_future_age - 3 - years_ago) :=
by
  sorry

end NUMINAMATH_CALUDE_mother_daughter_age_relation_l1767_176711


namespace NUMINAMATH_CALUDE_weight_of_new_person_l1767_176785

/-- Theorem: Weight of new person in group weight change scenario -/
theorem weight_of_new_person
  (n : ℕ)  -- Number of persons in the group
  (w : ℝ)  -- Initial total weight of the group
  (r : ℝ)  -- Weight of the person being replaced
  (d : ℝ)  -- Increase in average weight after replacement
  (h1 : n = 10)  -- There are 10 persons
  (h2 : r = 65)  -- The replaced person weighs 65 kg
  (h3 : d = 3.7)  -- The average weight increases by 3.7 kg
  : ∃ x : ℝ, (w - r + x) / n = w / n + d ∧ x = 102 :=
sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l1767_176785


namespace NUMINAMATH_CALUDE_kyle_to_grant_ratio_l1767_176791

def parker_distance : ℝ := 16

def grant_distance : ℝ := parker_distance * 1.25

def kyle_distance : ℝ := parker_distance + 24

theorem kyle_to_grant_ratio : kyle_distance / grant_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_kyle_to_grant_ratio_l1767_176791


namespace NUMINAMATH_CALUDE_parabola_translation_equivalence_l1767_176766

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The original parabola y = 2x^2 -/
def original_parabola : Parabola := { a := 2, h := 0, k := 0 }

/-- The transformed parabola y = 2(x - 4)^2 - 1 -/
def transformed_parabola : Parabola := { a := 2, h := 4, k := -1 }

/-- Translation of a parabola -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

theorem parabola_translation_equivalence :
  translate original_parabola 4 (-1) = transformed_parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_equivalence_l1767_176766


namespace NUMINAMATH_CALUDE_match_probabilities_l1767_176758

/-- A best-of-5 match where the probability of winning each game is 3/5 -/
structure Match :=
  (p : ℝ)
  (h_p : p = 3/5)

/-- The probability of winning 3 consecutive games -/
def prob_win_3_0 (m : Match) : ℝ := m.p^3

/-- The probability of winning the match after losing the first game -/
def prob_win_after_loss (m : Match) : ℝ :=
  m.p^3 + 3 * m.p^3 * (1 - m.p)

/-- The expected number of games played when losing the first game -/
def expected_games_after_loss (m : Match) : ℝ :=
  3 * (1 - m.p)^2 + 4 * (2 * m.p * (1 - m.p)^2 + m.p^3) + 5 * (3 * m.p^2 * (1 - m.p)^2 + m.p^3 * (1 - m.p))

theorem match_probabilities (m : Match) :
  prob_win_3_0 m = 27/125 ∧
  prob_win_after_loss m = 297/625 ∧
  expected_games_after_loss m = 534/125 :=
by sorry

end NUMINAMATH_CALUDE_match_probabilities_l1767_176758


namespace NUMINAMATH_CALUDE_eighteen_power_mnp_l1767_176770

theorem eighteen_power_mnp (m n p : ℕ) (P Q R : ℕ) 
  (hP : P = 2^m) (hQ : Q = 3^n) (hR : R = 5^p) :
  18^(m*n*p) = P^(n*p) * Q^(2*m*p) := by
  sorry

end NUMINAMATH_CALUDE_eighteen_power_mnp_l1767_176770


namespace NUMINAMATH_CALUDE_susan_scores_arithmetic_mean_l1767_176748

def susan_scores : List ℝ := [87, 90, 95, 98, 100]

theorem susan_scores_arithmetic_mean :
  (susan_scores.sum / susan_scores.length : ℝ) = 94 := by
  sorry

end NUMINAMATH_CALUDE_susan_scores_arithmetic_mean_l1767_176748


namespace NUMINAMATH_CALUDE_apple_in_B_l1767_176757

-- Define the boxes
inductive Box
| A
| B
| C

-- Define the location of the apple
def apple_location : Box := Box.B

-- Define the notes on the boxes
def note_A : Prop := apple_location = Box.A
def note_B : Prop := apple_location ≠ Box.B
def note_C : Prop := apple_location ≠ Box.A

-- Define the condition that only one note is true
def only_one_true : Prop :=
  (note_A ∧ ¬note_B ∧ ¬note_C) ∨
  (¬note_A ∧ note_B ∧ ¬note_C) ∨
  (¬note_A ∧ ¬note_B ∧ note_C)

-- Theorem to prove
theorem apple_in_B :
  only_one_true → apple_location = Box.B :=
by sorry

end NUMINAMATH_CALUDE_apple_in_B_l1767_176757


namespace NUMINAMATH_CALUDE_probability_derek_julia_captains_l1767_176736

theorem probability_derek_julia_captains (total_players : Nat) (num_teams : Nat) (team_size : Nat) (captains_per_team : Nat) :
  total_players = 64 →
  num_teams = 8 →
  team_size = 8 →
  captains_per_team = 2 →
  num_teams * team_size = total_players →
  (probability_both_captains : ℚ) = 5 / 84 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_derek_julia_captains_l1767_176736


namespace NUMINAMATH_CALUDE_factorization_problems_l1767_176700

theorem factorization_problems :
  (∀ x : ℝ, x^3 - 9*x = x*(x+3)*(x-3)) ∧
  (∀ a b : ℝ, a^3*b - 2*a^2*b + a*b = a*b*(a-1)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l1767_176700


namespace NUMINAMATH_CALUDE_line_passes_through_quadrants_l1767_176780

-- Define the type for quadrants
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define a function to check if a point (x, y) is in a given quadrant
def in_quadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.First => x > 0 ∧ y > 0
  | Quadrant.Second => x < 0 ∧ y > 0
  | Quadrant.Third => x < 0 ∧ y < 0
  | Quadrant.Fourth => x > 0 ∧ y < 0

-- Define a function to check if a line passes through a quadrant
def line_passes_through (m b : ℝ) (q : Quadrant) : Prop :=
  ∃ (x y : ℝ), y = m * x + b ∧ in_quadrant x y q

-- State the theorem
theorem line_passes_through_quadrants
  (a b c p : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h1 : (a + b) / c = p)
  (h2 : (b + c) / a = p)
  (h3 : (c + a) / b = p) :
  line_passes_through p p Quadrant.Second ∧
  line_passes_through p p Quadrant.Third :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_quadrants_l1767_176780


namespace NUMINAMATH_CALUDE_length_AB_line_MN_fixed_point_min_distance_PM_l1767_176799

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 4 = 0
def line_l (x y : ℝ) : Prop := x - 2*y + 5 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_C A.1 A.2 ∧
  circle_O B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Define the tangent points M and N
def tangent_points (P M N : ℝ × ℝ) : Prop :=
  line_l P.1 P.2 ∧
  circle_O M.1 M.2 ∧ circle_O N.1 N.2 ∧
  (P.1 - M.1) * M.1 + (P.2 - M.2) * M.2 = 0 ∧
  (P.1 - N.1) * N.1 + (P.2 - N.2) * N.2 = 0

-- Theorem statements
theorem length_AB : ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 / 5 := 
sorry

theorem line_MN_fixed_point : ∀ P M N : ℝ × ℝ, tangent_points P M N →
  ∃ t : ℝ, M.1 + t * (N.1 - M.1) = -4/5 ∧ M.2 + t * (N.2 - M.2) = 8/5 :=
sorry

theorem min_distance_PM : ∀ P : ℝ × ℝ, line_l P.1 P.2 →
  (∃ M : ℝ × ℝ, circle_O M.1 M.2 ∧ 
    ∀ N : ℝ × ℝ, circle_O N.1 N.2 → 
      Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) ≤ 
      Real.sqrt ((P.1 - N.1)^2 + (P.2 - N.2)^2)) ∧
  (∃ M : ℝ × ℝ, circle_O M.1 M.2 ∧ 
    Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) = 1) :=
sorry

end NUMINAMATH_CALUDE_length_AB_line_MN_fixed_point_min_distance_PM_l1767_176799


namespace NUMINAMATH_CALUDE_cost_of_one_ring_l1767_176760

/-- The cost of a single ring given the total cost and number of rings. -/
def ring_cost (total_cost : ℕ) (num_rings : ℕ) : ℕ :=
  total_cost / num_rings

/-- Theorem stating that the cost of one ring is $24 given the problem conditions. -/
theorem cost_of_one_ring :
  let total_cost : ℕ := 48
  let num_rings : ℕ := 2
  ring_cost total_cost num_rings = 24 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_ring_l1767_176760


namespace NUMINAMATH_CALUDE_pascal_triangle_probability_l1767_176703

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) :=
  sorry

/-- Counts the number of elements equal to a given value in Pascal's Triangle -/
def countElementsEqual (triangle : List (List ℕ)) (value : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of elements in Pascal's Triangle -/
def totalElements (triangle : List (List ℕ)) : ℕ :=
  sorry

theorem pascal_triangle_probability (n : ℕ) :
  n = 20 →
  let triangle := PascalTriangle n
  let ones := countElementsEqual triangle 1
  let twos := countElementsEqual triangle 2
  let total := totalElements triangle
  (ones + twos : ℚ) / total = 57 / 210 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_probability_l1767_176703


namespace NUMINAMATH_CALUDE_harry_apples_l1767_176707

/-- The number of apples Harry ends up with after buying more -/
def final_apples (initial : ℕ) (bought : ℕ) : ℕ := initial + bought

/-- Theorem: Harry ends up with 84 apples -/
theorem harry_apples : final_apples 79 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_harry_apples_l1767_176707


namespace NUMINAMATH_CALUDE_perfect_square_solution_l1767_176733

theorem perfect_square_solution : 
  ∃! (n : ℤ), ∃ (m : ℤ), n^2 + 20*n + 11 = m^2 :=
by
  -- The unique solution is n = 35
  use 35
  sorry

end NUMINAMATH_CALUDE_perfect_square_solution_l1767_176733


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1767_176788

/-- Given a circle with equation x^2 + y^2 - 2x + 6y = 0, 
    prove that its center is at (1, -3) and its radius is √10 -/
theorem circle_center_and_radius :
  ∃ (x y : ℝ), x^2 + y^2 - 2*x + 6*y = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -3) ∧ radius = Real.sqrt 10 ∧
    ∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 ↔ 
                   p.1^2 + p.2^2 - 2*p.1 + 6*p.2 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1767_176788


namespace NUMINAMATH_CALUDE_exactly_two_approve_probability_l1767_176794

def approval_rate : ℝ := 0.8
def num_voters : ℕ := 4
def num_approving : ℕ := 2

def probability_exactly_two_approve : ℝ := 
  (Nat.choose num_voters num_approving) * (approval_rate ^ num_approving) * ((1 - approval_rate) ^ (num_voters - num_approving))

theorem exactly_two_approve_probability :
  probability_exactly_two_approve = 0.1536 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_approve_probability_l1767_176794


namespace NUMINAMATH_CALUDE_complement_of_28_39_l1767_176775

/-- Represents an angle in degrees and minutes -/
structure Angle where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the complement of an angle -/
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  { degrees := totalMinutes / 60,
    minutes := totalMinutes % 60,
    valid := by sorry }

theorem complement_of_28_39 :
  let a : Angle := { degrees := 28, minutes := 39, valid := by sorry }
  complement a = { degrees := 61, minutes := 21, valid := by sorry } := by
  sorry

end NUMINAMATH_CALUDE_complement_of_28_39_l1767_176775


namespace NUMINAMATH_CALUDE_last_three_average_l1767_176727

theorem last_three_average (list : List ℝ) (h1 : list.length = 7) 
  (h2 : list.sum / 7 = 60) (h3 : (list.take 4).sum / 4 = 55) : 
  (list.drop 4).sum / 3 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_last_three_average_l1767_176727


namespace NUMINAMATH_CALUDE_prime_iff_divides_factorial_plus_one_l1767_176739

theorem prime_iff_divides_factorial_plus_one (n : ℕ) (h : n ≥ 2) :
  Nat.Prime n ↔ n ∣ (Nat.factorial (n - 1) + 1) :=
sorry

end NUMINAMATH_CALUDE_prime_iff_divides_factorial_plus_one_l1767_176739


namespace NUMINAMATH_CALUDE_inequality_proof_l1767_176725

theorem inequality_proof (a b : ℝ) (n : ℤ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1767_176725


namespace NUMINAMATH_CALUDE_man_business_ownership_l1767_176705

/-- 
Given a business valued at 10000 rs, if a man sells 3/5 of his shares for 2000 rs,
then he originally owned 1/3 of the business.
-/
theorem man_business_ownership (man_share : ℚ) : 
  (3 / 5 : ℚ) * man_share * 10000 = 2000 → man_share = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_man_business_ownership_l1767_176705


namespace NUMINAMATH_CALUDE_inequality_range_l1767_176756

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (1 + a) * x^2 + a * x + a < x^2 + 1) → 
  a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1767_176756


namespace NUMINAMATH_CALUDE_min_distance_four_points_l1767_176773

/-- Given four points in a metric space with specified distances between consecutive points,
    the theorem states that the minimum possible distance between the first and last points is 3. -/
theorem min_distance_four_points (X : Type*) [MetricSpace X] (P Q R S : X) :
  dist P Q = 12 →
  dist Q R = 7 →
  dist R S = 2 →
  ∃ (configuration : X → X), dist (configuration P) (configuration S) = 3 ∧
  (∀ (P' Q' R' S' : X),
    dist P' Q' = 12 →
    dist Q' R' = 7 →
    dist R' S' = 2 →
    dist P' S' ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_four_points_l1767_176773


namespace NUMINAMATH_CALUDE_prob_at_least_four_matching_dice_l1767_176764

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define the probability of getting at least four matching dice
def prob_at_least_four_matching : ℚ := 13 / 648

-- Theorem statement
theorem prob_at_least_four_matching_dice (n : ℕ) (s : ℕ) 
  (h1 : n = num_dice) (h2 : s = num_sides) : 
  prob_at_least_four_matching = 13 / 648 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_four_matching_dice_l1767_176764


namespace NUMINAMATH_CALUDE_lily_bought_20_ducks_l1767_176744

/-- The number of ducks Lily bought -/
def lily_ducks : ℕ := sorry

/-- The number of geese Lily bought -/
def lily_geese : ℕ := 10

/-- The number of ducks Rayden bought -/
def rayden_ducks : ℕ := 3 * lily_ducks

/-- The number of geese Rayden bought -/
def rayden_geese : ℕ := 4 * lily_geese

/-- The total number of birds Lily has -/
def lily_total : ℕ := lily_ducks + lily_geese

/-- The total number of birds Rayden has -/
def rayden_total : ℕ := rayden_ducks + rayden_geese

theorem lily_bought_20_ducks :
  lily_ducks = 20 ∧
  rayden_total = lily_total + 70 := by
  sorry

end NUMINAMATH_CALUDE_lily_bought_20_ducks_l1767_176744


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l1767_176720

-- Equation 1
theorem solve_equation_one : 
  {x : ℝ | x^2 - 9 = 0} = {3, -3} := by sorry

-- Equation 2
theorem solve_equation_two :
  {x : ℝ | (x + 1)^3 = -8/27} = {-5/3} := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l1767_176720


namespace NUMINAMATH_CALUDE_meals_without_restrictions_l1767_176728

theorem meals_without_restrictions (total_clients : ℕ) (vegan kosher gluten_free halal dairy_free nut_free : ℕ)
  (vegan_kosher vegan_gluten kosher_gluten halal_dairy gluten_nut : ℕ)
  (vegan_halal_gluten kosher_dairy_nut : ℕ)
  (h1 : total_clients = 80)
  (h2 : vegan = 15)
  (h3 : kosher = 18)
  (h4 : gluten_free = 12)
  (h5 : halal = 10)
  (h6 : dairy_free = 8)
  (h7 : nut_free = 4)
  (h8 : vegan_kosher = 5)
  (h9 : vegan_gluten = 6)
  (h10 : kosher_gluten = 3)
  (h11 : halal_dairy = 4)
  (h12 : gluten_nut = 2)
  (h13 : vegan_halal_gluten = 2)
  (h14 : kosher_dairy_nut = 1) :
  total_clients - (vegan + kosher + gluten_free + halal + dairy_free + nut_free - 
    (vegan_kosher + vegan_gluten + kosher_gluten + halal_dairy + gluten_nut) + 
    (vegan_halal_gluten + kosher_dairy_nut)) = 30 := by
  sorry


end NUMINAMATH_CALUDE_meals_without_restrictions_l1767_176728


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_sum_l1767_176702

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := sorry

-- Define the conditions
axiom a7_eq_4 : a 7 = 4
axiom a19_eq_2a9 : a 19 = 2 * a 9

-- Define b_n
def b (n : ℕ) : ℚ := 1 / (2 * n * a n)

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := sorry

theorem arithmetic_sequence_and_sum :
  (∀ n : ℕ, a n = (n + 1) / 2) ∧
  (∀ n : ℕ, S n = n / (n + 1)) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_sum_l1767_176702


namespace NUMINAMATH_CALUDE_elle_weekly_practice_hours_l1767_176771

/-- The number of minutes Elle practices piano on a weekday -/
def weekday_practice : ℕ := 30

/-- The number of weekdays Elle practices piano -/
def weekday_count : ℕ := 5

/-- The factor by which Elle's Saturday practice is longer than a weekday practice -/
def saturday_factor : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating that Elle spends 4 hours practicing piano each week -/
theorem elle_weekly_practice_hours : 
  (weekday_practice * weekday_count + weekday_practice * saturday_factor) / minutes_per_hour = 4 := by
  sorry

end NUMINAMATH_CALUDE_elle_weekly_practice_hours_l1767_176771


namespace NUMINAMATH_CALUDE_total_investment_l1767_176717

/-- Proves that the total investment of Vishal, Trishul, and Raghu is 5780 Rs. -/
theorem total_investment (raghu_investment : ℝ) 
  (h1 : raghu_investment = 2000)
  (h2 : ∃ trishul_investment : ℝ, trishul_investment = raghu_investment * 0.9)
  (h3 : ∃ vishal_investment : ℝ, vishal_investment = raghu_investment * 0.9 * 1.1) :
  raghu_investment + raghu_investment * 0.9 + raghu_investment * 0.9 * 1.1 = 5780 :=
by sorry


end NUMINAMATH_CALUDE_total_investment_l1767_176717


namespace NUMINAMATH_CALUDE_y_intercept_two_distance_from_origin_one_l1767_176735

-- Define the general equation of line l
def line_equation (a : ℝ) (x y : ℝ) : Prop :=
  x + (a + 1) * y + 2 - a = 0

-- Theorem 1: y-intercept is 2
theorem y_intercept_two :
  ∃ a : ℝ, (∀ x y : ℝ, line_equation a x y ↔ x - 3 * y + 6 = 0) ∧
  (∃ y : ℝ, line_equation a 0 y ∧ y = 2) :=
sorry

-- Theorem 2: distance from origin is 1
theorem distance_from_origin_one :
  ∃ a : ℝ, (∀ x y : ℝ, line_equation a x y ↔ 3 * x + 4 * y + 5 = 0) ∧
  (|2 - a| / Real.sqrt (1 + (a + 1)^2) = 1) :=
sorry

end NUMINAMATH_CALUDE_y_intercept_two_distance_from_origin_one_l1767_176735


namespace NUMINAMATH_CALUDE_closure_of_A_range_of_a_l1767_176754

-- Define set A
def A : Set ℝ := {x | x < -1 ∨ x > -1/2}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}

-- Theorem for the closure of A
theorem closure_of_A : 
  closure A = {x : ℝ | -1 ≤ x ∧ x ≤ -1/2} := by sorry

-- Theorem for the range of a
theorem range_of_a : 
  (∃ a : ℝ, A ∪ B a = Set.univ) ↔ ∃ a : ℝ, -3/2 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_closure_of_A_range_of_a_l1767_176754


namespace NUMINAMATH_CALUDE_sum_of_integers_l1767_176704

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 250)
  (h2 : x * y = 120)
  (h3 : x^2 - y^2 = 130) :
  x + y = 10 * Real.sqrt 4.9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1767_176704


namespace NUMINAMATH_CALUDE_trebled_result_is_69_l1767_176755

theorem trebled_result_is_69 :
  let x : ℕ := 7
  let doubled_plus_nine := 2 * x + 9
  let trebled_result := 3 * doubled_plus_nine
  trebled_result = 69 := by
sorry

end NUMINAMATH_CALUDE_trebled_result_is_69_l1767_176755


namespace NUMINAMATH_CALUDE_sum_of_consecutive_integers_l1767_176761

theorem sum_of_consecutive_integers (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 30 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_integers_l1767_176761


namespace NUMINAMATH_CALUDE_equation_solution_l1767_176742

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (x - 15))) = 54 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1767_176742


namespace NUMINAMATH_CALUDE_total_spending_l1767_176747

/-- Represents the amount spent by Ben -/
def ben_spent : ℝ := sorry

/-- Represents the amount spent by David -/
def david_spent : ℝ := sorry

/-- Ben spends $1 for every $0.75 David spends -/
axiom spending_ratio : david_spent = 0.75 * ben_spent

/-- Ben spends $12.50 more than David -/
axiom spending_difference : ben_spent = david_spent + 12.50

/-- The total amount spent by Ben and David -/
def total_spent : ℝ := ben_spent + david_spent

/-- Theorem: The total amount spent by Ben and David is $87.50 -/
theorem total_spending : total_spent = 87.50 := by sorry

end NUMINAMATH_CALUDE_total_spending_l1767_176747


namespace NUMINAMATH_CALUDE_min_sum_squares_l1767_176776

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + z = 1) : 
  ∃ (m : ℝ), (∀ a b c : ℝ, a + 2*b + c = 1 → a^2 + b^2 + c^2 ≥ m) ∧ 
  (∃ p q r : ℝ, p + 2*q + r = 1 ∧ p^2 + q^2 + r^2 = m) ∧ 
  m = 1/6 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1767_176776


namespace NUMINAMATH_CALUDE_square_root_of_nine_l1767_176710

-- Define the square root function
def square_root (x : ℝ) : Set ℝ :=
  {y : ℝ | y * y = x}

-- Theorem statement
theorem square_root_of_nine :
  square_root 9 = {3, -3} := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l1767_176710


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1767_176753

theorem product_sum_theorem (p q r s t : ℤ) :
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t →
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120 →
  p + q + r + s + t = 35 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1767_176753


namespace NUMINAMATH_CALUDE_book_price_increase_l1767_176726

/-- Calculates the new price of a book after a percentage increase -/
theorem book_price_increase (original_price : ℝ) (increase_percentage : ℝ) :
  original_price = 300 ∧ increase_percentage = 30 →
  original_price * (1 + increase_percentage / 100) = 390 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l1767_176726


namespace NUMINAMATH_CALUDE_win_sector_area_l1767_176743

/-- Theorem: Area of WIN sector in a circular spinner game --/
theorem win_sector_area (r : ℝ) (p_win : ℝ) (p_bonus_lose : ℝ) (h1 : r = 8)
  (h2 : p_win = 1 / 4) (h3 : p_bonus_lose = 1 / 8) :
  p_win * π * r^2 = 16 * π := by sorry

end NUMINAMATH_CALUDE_win_sector_area_l1767_176743


namespace NUMINAMATH_CALUDE_third_angle_is_40_l1767_176777

/-- A geometric configuration with an isosceles triangle connected to a right-angled triangle -/
structure GeometricConfig where
  -- Angles of the isosceles triangle
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Angles of the right-angled triangle
  δ : ℝ
  ε : ℝ
  ζ : ℝ

/-- Properties of the geometric configuration -/
def is_valid_config (c : GeometricConfig) : Prop :=
  c.α = 65 ∧ c.β = 65 ∧  -- Two angles of isosceles triangle are 65°
  c.α + c.β + c.γ = 180 ∧  -- Sum of angles in isosceles triangle is 180°
  c.δ = 90 ∧  -- One angle of right-angled triangle is 90°
  c.γ = c.ε ∧  -- Vertically opposite angles are equal
  c.δ + c.ε + c.ζ = 180  -- Sum of angles in right-angled triangle is 180°

/-- Theorem stating that the third angle of the right-angled triangle is 40° -/
theorem third_angle_is_40 (c : GeometricConfig) (h : is_valid_config c) : c.ζ = 40 :=
sorry

end NUMINAMATH_CALUDE_third_angle_is_40_l1767_176777


namespace NUMINAMATH_CALUDE_large_circle_radius_large_circle_radius_value_l1767_176734

/-- The radius of a circle that internally touches two circles of radius 2 and both internally
    and externally touches a third circle of radius 2 (where all three smaller circles are
    externally tangent to each other) is equal to 4 + 2√3. -/
theorem large_circle_radius : ℝ → ℝ → Prop :=
  fun (small_radius large_radius : ℝ) =>
    small_radius = 2 ∧
    (∃ (centers : Fin 3 → ℝ × ℝ) (large_center : ℝ × ℝ),
      (∀ i j, i ≠ j → dist (centers i) (centers j) = 2 * small_radius) ∧
      (∀ i, dist (centers i) large_center ≤ large_radius + small_radius) ∧
      (∃ k, dist (centers k) large_center = large_radius - small_radius) ∧
      (∃ l, dist (centers l) large_center = large_radius + small_radius)) →
    large_radius = 4 + 2 * Real.sqrt 3

theorem large_circle_radius_value : large_circle_radius 2 (4 + 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_large_circle_radius_large_circle_radius_value_l1767_176734


namespace NUMINAMATH_CALUDE_football_field_fertilizer_l1767_176767

/-- Given a football field and fertilizer distribution, calculate the total fertilizer used. -/
theorem football_field_fertilizer 
  (field_area : ℝ) 
  (partial_area : ℝ) 
  (partial_fertilizer : ℝ) 
  (h1 : field_area = 8400)
  (h2 : partial_area = 3500)
  (h3 : partial_fertilizer = 500)
  (h4 : partial_area > 0)
  (h5 : field_area > 0) :
  (field_area * partial_fertilizer) / partial_area = 1200 :=
by sorry

end NUMINAMATH_CALUDE_football_field_fertilizer_l1767_176767


namespace NUMINAMATH_CALUDE_new_boy_weight_l1767_176763

theorem new_boy_weight (original_count : ℕ) (original_average : ℝ) (new_average : ℝ) : 
  original_count = 5 →
  original_average = 35 →
  new_average = 36 →
  (original_count + 1) * new_average - original_count * original_average = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_new_boy_weight_l1767_176763


namespace NUMINAMATH_CALUDE_classroom_gpa_l1767_176740

theorem classroom_gpa (n : ℝ) (h : n > 0) : 
  (1/3 * n * 45 + 2/3 * n * 60) / n = 55 := by
  sorry

end NUMINAMATH_CALUDE_classroom_gpa_l1767_176740


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l1767_176790

theorem rectangle_length_proof (l w : ℝ) : l = w + 3 ∧ l * w = 4 → l = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l1767_176790


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l1767_176795

theorem square_sum_given_diff_and_product (x y : ℝ) 
  (h1 : x - y = 20) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 418 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l1767_176795


namespace NUMINAMATH_CALUDE_mom_tshirt_count_l1767_176708

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom buys -/
def packages_bought : ℕ := 71

/-- The total number of t-shirts Mom will have -/
def total_shirts : ℕ := shirts_per_package * packages_bought

theorem mom_tshirt_count : total_shirts = 426 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_count_l1767_176708


namespace NUMINAMATH_CALUDE_triangular_sequence_start_fifteenth_triangular_number_l1767_176752

/-- Triangular number function -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sequence of triangular numbers starts with 1, 3, 6, 10, ... -/
theorem triangular_sequence_start :
  [triangular_number 1, triangular_number 2, triangular_number 3, triangular_number 4] = [1, 3, 6, 10] := by sorry

/-- The 15th triangular number is 120 -/
theorem fifteenth_triangular_number :
  triangular_number 15 = 120 := by sorry

end NUMINAMATH_CALUDE_triangular_sequence_start_fifteenth_triangular_number_l1767_176752


namespace NUMINAMATH_CALUDE_coin_coverage_theorem_l1767_176783

/-- Represents the arrangement of 7 identical coins on an infinite plane -/
structure CoinArrangement where
  radius : ℝ
  num_coins : Nat
  touches_six : Bool

/-- Calculates the percentage of the plane covered by the coins -/
def coverage_percentage (arrangement : CoinArrangement) : ℝ :=
  sorry

/-- Theorem stating that the coverage percentage is 50π/√3 % -/
theorem coin_coverage_theorem (arrangement : CoinArrangement) 
  (h1 : arrangement.num_coins = 7)
  (h2 : arrangement.touches_six = true) : 
  coverage_percentage arrangement = (50 * Real.pi) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_coin_coverage_theorem_l1767_176783


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1767_176730

/-- The number of ways to arrange books on a shelf --/
def arrange_books (num_math_books : ℕ) (num_history_books : ℕ) : ℕ :=
  let remaining_books := num_math_books + (num_history_books - 2)
  num_history_books * (num_history_books - 1) * Nat.factorial remaining_books

/-- Theorem stating the correct number of arrangements for the given problem --/
theorem book_arrangement_count :
  arrange_books 5 4 = 60480 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1767_176730


namespace NUMINAMATH_CALUDE_product_of_symmetric_complex_l1767_176768

/-- Two complex numbers are symmetric about the angle bisector of the first and third quadrants if their real and imaginary parts are interchanged. -/
def symmetric_about_bisector (z₁ z₂ : ℂ) : Prop :=
  z₁.re = z₂.im ∧ z₁.im = z₂.re

theorem product_of_symmetric_complex : ∀ z₁ z₂ : ℂ,
  symmetric_about_bisector z₁ z₂ → z₁ = 1 + 2*I → z₁ * z₂ = 5*I :=
by sorry

end NUMINAMATH_CALUDE_product_of_symmetric_complex_l1767_176768


namespace NUMINAMATH_CALUDE_parabola_point_distance_l1767_176721

/-- For a parabola y = ax², if a point P(x₀, 2) on the parabola is at a distance of 3 
    from the focus, then the distance from P to the y-axis is 2√2. -/
theorem parabola_point_distance (a : ℝ) (x₀ : ℝ) :
  (2 = a * x₀^2) →                          -- P is on the parabola
  ((x₀ - 0)^2 + (2 - 1/(4*a))^2 = 3^2) →    -- Distance from P to focus is 3
  |x₀| = 2 * Real.sqrt 2 :=                 -- Distance from P to y-axis is 2√2
by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l1767_176721


namespace NUMINAMATH_CALUDE_initial_to_doubled_ratio_l1767_176723

theorem initial_to_doubled_ratio (x : ℝ) : 3 * (2 * x + 5) = 105 → x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_to_doubled_ratio_l1767_176723


namespace NUMINAMATH_CALUDE_smallest_modulus_w_l1767_176778

theorem smallest_modulus_w (w : ℂ) (h : Complex.abs (w - 8) + Complex.abs (w - 3 * I) = 15) :
  ∃ (w_min : ℂ), Complex.abs w_min ≤ Complex.abs w ∧ Complex.abs w_min = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_modulus_w_l1767_176778


namespace NUMINAMATH_CALUDE_circle_radius_l1767_176701

theorem circle_radius (x y : ℝ) (h : x + y = 100 * Real.pi) : 
  (∃ r : ℝ, x = Real.pi * r^2 ∧ y = 2 * Real.pi * r) → 
  (∃ r : ℝ, x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 10) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l1767_176701


namespace NUMINAMATH_CALUDE_train_speed_l1767_176718

/-- The speed of a train given its length and time to cross a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 275) (h2 : time = 7) :
  ∃ (speed : ℝ), abs (speed - 141.43) < 0.01 ∧ speed = (length / time) * 3.6 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l1767_176718
