import Mathlib

namespace NUMINAMATH_CALUDE_boarding_students_change_l2471_247137

theorem boarding_students_change (initial : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) 
  (h1 : increase_rate = 0.2) 
  (h2 : decrease_rate = 0.2) : 
  initial * (1 + increase_rate) * (1 - decrease_rate) = initial * 0.96 :=
by sorry

end NUMINAMATH_CALUDE_boarding_students_change_l2471_247137


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2471_247196

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  ((a^2 + b^2 + c^2 + d^2)^2 ≥ (a+b)*(b+c)*(c+d)*(d+a)) ∧ 
  ((a^2 + b^2 + c^2 + d^2)^2 = (a+b)*(b+c)*(c+d)*(d+a) ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2471_247196


namespace NUMINAMATH_CALUDE_total_revenue_is_2176_l2471_247133

def kitten_price : ℕ := 80
def puppy_price : ℕ := 150
def rabbit_price : ℕ := 45
def guinea_pig_price : ℕ := 30

def kitten_count : ℕ := 10
def puppy_count : ℕ := 8
def rabbit_count : ℕ := 4
def guinea_pig_count : ℕ := 6

def discount_rate : ℚ := 1/10

def total_revenue : ℚ := 
  (kitten_count * kitten_price + 
   puppy_count * puppy_price + 
   rabbit_count * rabbit_price + 
   guinea_pig_count * guinea_pig_price : ℚ) - 
  (min kitten_count puppy_count * discount_rate * (kitten_price + puppy_price))

theorem total_revenue_is_2176 : total_revenue = 2176 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_2176_l2471_247133


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l2471_247194

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : ∀ x y : ℝ, x < y → f x > f y) 
  (h_inequality : f a ≥ f (-2)) : 
  a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l2471_247194


namespace NUMINAMATH_CALUDE_multiplication_subtraction_difference_l2471_247188

theorem multiplication_subtraction_difference : ∃ (x : ℤ), x = 22 ∧ 3 * x - (62 - x) = 26 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_difference_l2471_247188


namespace NUMINAMATH_CALUDE_derivative_of_f_l2471_247144

-- Define the function f
def f (x : ℝ) : ℝ := 2016 * x^2

-- State the theorem
theorem derivative_of_f (x : ℝ) :
  deriv f x = 4032 * x := by sorry

-- Note: The 'deriv' function in Lean represents the derivative.

end NUMINAMATH_CALUDE_derivative_of_f_l2471_247144


namespace NUMINAMATH_CALUDE_apple_price_36kg_l2471_247163

/-- The price of apples for a given weight --/
def apple_price (l q : ℚ) (weight : ℚ) : ℚ :=
  if weight ≤ 30 then l * weight
  else l * 30 + q * (weight - 30)

theorem apple_price_36kg (l q : ℚ) : 
  (apple_price l q 20 = 100) → 
  (apple_price l q 33 = 168) → 
  (apple_price l q 36 = 186) := by
  sorry

#check apple_price_36kg

end NUMINAMATH_CALUDE_apple_price_36kg_l2471_247163


namespace NUMINAMATH_CALUDE_min_value_a_l2471_247176

theorem min_value_a (a : ℝ) : (∀ x ∈ Set.Ioc (0 : ℝ) (1/2), x^2 + a*x + 1 ≥ 0) ↔ a ≥ -5/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l2471_247176


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l2471_247191

theorem sqrt_x_minus_3_real (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 3) ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l2471_247191


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l2471_247103

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- Theorem stating that f is monotonically increasing on ℝ
theorem f_monotone_increasing : Monotone f := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l2471_247103


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2471_247141

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 10
  geometric_sum a r n = 29524/59049 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2471_247141


namespace NUMINAMATH_CALUDE_distinct_combinations_l2471_247135

def num_shirts : ℕ := 8
def num_ties : ℕ := 7
def num_jackets : ℕ := 3

theorem distinct_combinations : num_shirts * num_ties * num_jackets = 168 := by
  sorry

end NUMINAMATH_CALUDE_distinct_combinations_l2471_247135


namespace NUMINAMATH_CALUDE_watch_sale_price_l2471_247155

/-- The final sale price of a watch after two consecutive discounts --/
theorem watch_sale_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  initial_price = 20 ∧ discount1 = 0.20 ∧ discount2 = 0.25 →
  initial_price * (1 - discount1) * (1 - discount2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_watch_sale_price_l2471_247155


namespace NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l2471_247199

def f (x : ℝ) : ℝ := x^4 + x^2 + 7*x

theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 42 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l2471_247199


namespace NUMINAMATH_CALUDE_no_quadratic_trinomials_with_integer_roots_l2471_247192

theorem no_quadratic_trinomials_with_integer_roots : 
  ¬ ∃ (a b c x₁ x₂ x₃ x₄ : ℤ), 
    (∀ x : ℤ, a * x^2 + b * x + c = a * (x - x₁) * (x - x₂)) ∧ 
    (∀ x : ℤ, (a + 1) * x^2 + (b + 1) * x + (c + 1) = (a + 1) * (x - x₃) * (x - x₄)) := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_trinomials_with_integer_roots_l2471_247192


namespace NUMINAMATH_CALUDE_minimum_point_implies_b_greater_than_one_l2471_247119

theorem minimum_point_implies_b_greater_than_one (a b : ℝ) (hb : b ≠ 0) :
  let f := fun x : ℝ ↦ (x - b) * (x^2 + a*x + b)
  (∀ x, f b ≤ f x) →
  b > 1 := by
sorry

end NUMINAMATH_CALUDE_minimum_point_implies_b_greater_than_one_l2471_247119


namespace NUMINAMATH_CALUDE_pascal_triangle_elements_l2471_247160

/-- The number of elements in a row of Pascal's Triangle -/
def elementsInRow (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def sumOfElements (n : ℕ) : ℕ := 
  (List.range n).map elementsInRow |>.sum

/-- The number of elements in the first 25 rows of Pascal's Triangle is 325 -/
theorem pascal_triangle_elements : sumOfElements 25 = 325 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_elements_l2471_247160


namespace NUMINAMATH_CALUDE_smallest_perimeter_l2471_247136

/-- A triangle with side lengths that are three consecutive integers starting from 3 -/
structure ConsecutiveIntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : b = a + 1 ∧ c = b + 1
  start_from_three : a = 3

/-- The perimeter of a triangle -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ :=
  t.a + t.b + t.c

/-- Theorem: The perimeter of a triangle with side lengths 3, 4, and 5 is 12 units -/
theorem smallest_perimeter (t : ConsecutiveIntegerTriangle) : perimeter t = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l2471_247136


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l2471_247161

/-- The number of games played in a single-elimination tournament. -/
def games_played (num_teams : ℕ) : ℕ :=
  num_teams - 1

/-- Theorem: In a single-elimination tournament with 32 teams, 31 games are played. -/
theorem single_elimination_tournament_games :
  games_played 32 = 31 := by
  sorry

#eval games_played 32  -- Should output 31

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l2471_247161


namespace NUMINAMATH_CALUDE_range_when_p_true_range_when_p_or_q_true_and_p_and_q_false_l2471_247109

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-2) (-1), x^2 - a ≥ 0

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x - (a-2) = 0

-- Theorem 1
theorem range_when_p_true (a : ℝ) : p a → a ≤ 1 := by sorry

-- Theorem 2
theorem range_when_p_or_q_true_and_p_and_q_false (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Ioo (-2) 1 ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_range_when_p_true_range_when_p_or_q_true_and_p_and_q_false_l2471_247109


namespace NUMINAMATH_CALUDE_multiples_of_four_l2471_247156

theorem multiples_of_four (n : ℕ) : n = 20 ↔ (
  (∃ (m : List ℕ), 
    m.length = 24 ∧ 
    (∀ x ∈ m, x % 4 = 0) ∧
    (∀ x ∈ m, n ≤ x ∧ x ≤ 112) ∧
    (∀ y, n ≤ y ∧ y ≤ 112 ∧ y % 4 = 0 → y ∈ m)
  )
) := by sorry

end NUMINAMATH_CALUDE_multiples_of_four_l2471_247156


namespace NUMINAMATH_CALUDE_unique_factorization_l2471_247107

theorem unique_factorization (E F G H : ℕ+) : 
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H →
  E * F = 120 →
  G * H = 120 →
  E - F = G + H - 2 →
  E = 30 := by
sorry

end NUMINAMATH_CALUDE_unique_factorization_l2471_247107


namespace NUMINAMATH_CALUDE_john_needs_168_nails_l2471_247165

/-- The number of nails needed for a house wall -/
def nails_needed (num_planks : ℕ) (nails_per_plank : ℕ) : ℕ :=
  num_planks * nails_per_plank

/-- Theorem: John needs 168 nails for the house wall -/
theorem john_needs_168_nails :
  nails_needed 42 4 = 168 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_168_nails_l2471_247165


namespace NUMINAMATH_CALUDE_mean_temperature_l2471_247118

def temperatures : List ℝ := [75, 78, 80, 76, 77]

theorem mean_temperature : (temperatures.sum / temperatures.length : ℝ) = 77.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l2471_247118


namespace NUMINAMATH_CALUDE_final_amount_theorem_l2471_247146

def initial_amount : ℚ := 1499.9999999999998

def remaining_after_clothes (initial : ℚ) : ℚ := initial - (1/3 * initial)

def remaining_after_food (after_clothes : ℚ) : ℚ := after_clothes - (1/5 * after_clothes)

def remaining_after_travel (after_food : ℚ) : ℚ := after_food - (1/4 * after_food)

theorem final_amount_theorem :
  remaining_after_travel (remaining_after_food (remaining_after_clothes initial_amount)) = 600 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_theorem_l2471_247146


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l2471_247142

theorem sqrt_x_minus_3_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l2471_247142


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l2471_247198

/-- Given a right triangle with legs of lengths 6 and 8, and semicircles constructed
    on all its sides as diameters lying outside the triangle, the radius of the circle
    tangent to these semicircles is 144/23. -/
theorem tangent_circle_radius (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_a : a = 6) (h_b : b = 8) : ∃ r : ℝ, r = 144 / 23 ∧ 
  r > 0 ∧
  (∃ x y z : ℝ, x^2 + y^2 = (r + a/2)^2 ∧
               y^2 + z^2 = (r + b/2)^2 ∧
               z^2 + x^2 = (r + c/2)^2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l2471_247198


namespace NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l2471_247159

/-- The number of distinct arrangements of n beads on a necklace,
    considering rotational and reflectional symmetry -/
def necklace_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of distinct arrangements
    of 8 beads on a necklace is 2520 -/
theorem eight_bead_necklace_arrangements :
  necklace_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l2471_247159


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2471_247162

theorem inequality_solution_set (x : ℝ) : 
  2 / (x + 2) + 5 / (x + 4) ≥ 3 / 2 ↔ x ∈ Set.Icc (-4 : ℝ) (2/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2471_247162


namespace NUMINAMATH_CALUDE_odd_function_symmetry_symmetric_about_one_period_four_l2471_247181

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Statement 1
theorem odd_function_symmetry (h : ∀ x, f x = -f (-x)) :
  ∀ x, f (x - 1) = -f (-x + 1) :=
sorry

-- Statement 2
theorem symmetric_about_one (h : ∀ x, f (x - 1) = f (x + 1)) :
  ∀ x, f (1 - x) = f (1 + x) :=
sorry

-- Statement 4
theorem period_four (h1 : ∀ x, f (x + 1) = f (1 - x)) 
                    (h2 : ∀ x, f (x + 3) = f (3 - x)) :
  ∀ x, f x = f (x + 4) :=
sorry

end NUMINAMATH_CALUDE_odd_function_symmetry_symmetric_about_one_period_four_l2471_247181


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2471_247128

/-- An isosceles triangle with two sides of length 12 and one side of length 17 has a perimeter of 41 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → Prop :=
  fun (equal_side : ℝ) (third_side : ℝ) =>
    equal_side = 12 ∧ third_side = 17 →
    2 * equal_side + third_side = 41

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 12 17 :=
by
  sorry

#check isosceles_triangle_perimeter_proof

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2471_247128


namespace NUMINAMATH_CALUDE_travelers_meet_on_day_three_l2471_247183

/-- Distance traveled by the first traveler on day n -/
def d1 (n : ℕ) : ℕ := 3 * n - 1

/-- Distance traveled by the second traveler on day n -/
def d2 (n : ℕ) : ℕ := 2 * n + 1

/-- Total distance traveled by the first traveler after n days -/
def D1 (n : ℕ) : ℕ := (3 * n^2 + n) / 2

/-- Total distance traveled by the second traveler after n days -/
def D2 (n : ℕ) : ℕ := n^2 + 2 * n

theorem travelers_meet_on_day_three :
  ∃ n : ℕ, n > 0 ∧ D1 n = D2 n ∧ ∀ m : ℕ, 0 < m ∧ m < n → D1 m < D2 m :=
sorry

end NUMINAMATH_CALUDE_travelers_meet_on_day_three_l2471_247183


namespace NUMINAMATH_CALUDE_solution_pairs_l2471_247126

theorem solution_pairs : ∀ x y : ℝ, 
  (x + y + 4 = (12*x + 11*y) / (x^2 + y^2) ∧ 
   y - x + 3 = (11*x - 12*y) / (x^2 + y^2)) ↔ 
  ((x = 2 ∧ y = 1) ∨ (x = -2.5 ∧ y = -4.5)) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l2471_247126


namespace NUMINAMATH_CALUDE_set_operation_result_l2471_247130

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 2, 4}

-- Define set B
def B : Set Nat := {2, 3, 5}

-- Theorem statement
theorem set_operation_result :
  (U \ A) ∪ B = {0, 2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l2471_247130


namespace NUMINAMATH_CALUDE_cost_difference_formula_option1_more_cost_effective_at_50_l2471_247184

/-- Represents the cost difference between Option 2 and Option 1 for a customer
    buying 20 water dispensers and x water dispenser barrels, where x > 20. -/
def cost_difference (x : ℝ) : ℝ :=
  (45 * x + 6300) - (50 * x + 6000)

/-- Theorem stating that the cost difference between Option 2 and Option 1
    is always 300 - 5x yuan, for x > 20. -/
theorem cost_difference_formula (x : ℝ) (h : x > 20) :
  cost_difference x = 300 - 5 * x := by
  sorry

/-- Corollary stating that Option 1 is more cost-effective when x = 50. -/
theorem option1_more_cost_effective_at_50 :
  cost_difference 50 > 0 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_formula_option1_more_cost_effective_at_50_l2471_247184


namespace NUMINAMATH_CALUDE_expand_product_l2471_247138

theorem expand_product (x : ℝ) : (3*x - 4) * (2*x + 7) = 6*x^2 + 13*x - 28 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2471_247138


namespace NUMINAMATH_CALUDE_largest_value_l2471_247122

theorem largest_value (x y z w : ℝ) (h : x + 3 = y - 4 ∧ x + 3 = z + 2 ∧ x + 3 = w - 1) :
  y = max x (max y (max z w)) :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l2471_247122


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2471_247100

/-- Proves that given a simple interest of 100, an interest rate of 5% per annum,
    and a time period of 4 years, the principal sum is 500. -/
theorem simple_interest_problem (interest : ℕ) (rate : ℕ) (time : ℕ) (principal : ℕ) : 
  interest = 100 → rate = 5 → time = 4 → 
  interest = principal * rate * time / 100 →
  principal = 500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2471_247100


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l2471_247129

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m.val n.val = 12) :
  ∃ (k : ℕ+), k.val = Nat.gcd (8 * m.val) (18 * n.val) ∧ 
  ∀ (l : ℕ+), l.val = Nat.gcd (8 * m.val) (18 * n.val) → k ≤ l ∧ k.val = 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l2471_247129


namespace NUMINAMATH_CALUDE_equidistant_complex_function_l2471_247177

theorem equidistant_complex_function (a b : ℝ) :
  (∀ z : ℂ, ‖(a + b * I) * z^2 - z^2‖ = ‖(a + b * I) * z^2‖) →
  ‖(a + b * I)‖ = 10 →
  b^2 = 99.75 := by sorry

end NUMINAMATH_CALUDE_equidistant_complex_function_l2471_247177


namespace NUMINAMATH_CALUDE_addition_equality_l2471_247175

theorem addition_equality : 12 + 36 = 48 := by
  sorry

end NUMINAMATH_CALUDE_addition_equality_l2471_247175


namespace NUMINAMATH_CALUDE_max_notebooks_inequality_l2471_247174

/-- Represents the budget in dollars -/
def budget : ℝ := 500

/-- Represents the regular price per notebook in dollars -/
def regularPrice : ℝ := 10

/-- Represents the discount rate as a decimal -/
def discountRate : ℝ := 0.2

/-- Represents the threshold number of notebooks for the discount to apply -/
def discountThreshold : ℕ := 15

/-- Theorem stating that the maximum number of notebooks that can be purchased
    is represented by the inequality 10 × 0.8x ≤ 500 -/
theorem max_notebooks_inequality :
  ∀ x : ℝ, x > discountThreshold →
    (x = budget / (regularPrice * (1 - discountRate))) ↔ 
    (regularPrice * (1 - discountRate) * x ≤ budget) :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_inequality_l2471_247174


namespace NUMINAMATH_CALUDE_taqeeshas_grade_l2471_247101

theorem taqeeshas_grade (total_students : Nat) (initial_students : Nat) (initial_average : Nat) (new_average : Nat) :
  total_students = 17 →
  initial_students = 16 →
  initial_average = 77 →
  new_average = 78 →
  (initial_students * initial_average + (total_students - initial_students) * 94) / total_students = new_average :=
by sorry

end NUMINAMATH_CALUDE_taqeeshas_grade_l2471_247101


namespace NUMINAMATH_CALUDE_two_thousandth_digit_sum_l2471_247112

def is_valid_sequence (seq : List Nat) : Prop :=
  seq.length = 2000 ∧
  seq.head? = some 3 ∧
  ∀ i, i < 1999 → (seq.get? i).isSome ∧ (seq.get? (i+1)).isSome →
    (17 ∣ (seq.get! i * 10 + seq.get! (i+1))) ∨ (23 ∣ (seq.get! i * 10 + seq.get! (i+1)))

theorem two_thousandth_digit_sum (seq : List Nat) (a b : Nat) :
  is_valid_sequence seq →
  (seq.get? 1999 = some a ∨ seq.get? 1999 = some b) →
  a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_thousandth_digit_sum_l2471_247112


namespace NUMINAMATH_CALUDE_solution_set_equality_l2471_247169

-- Define the set of real numbers x that satisfy the inequality
def solution_set : Set ℝ := {x : ℝ | (x + 3) / (4 - x) ≥ 0 ∧ x ≠ 4}

-- Theorem stating that the solution set is equal to the interval [-3, 4)
theorem solution_set_equality : solution_set = Set.Icc (-3) 4 \ {4} :=
sorry

end NUMINAMATH_CALUDE_solution_set_equality_l2471_247169


namespace NUMINAMATH_CALUDE_sqrt_inequality_and_floor_l2471_247115

theorem sqrt_inequality_and_floor (n : ℕ) : 
  (Real.sqrt (n + 1) + 2 * Real.sqrt n < Real.sqrt (9 * n + 3)) ∧
  ¬∃ n : ℕ, ⌊Real.sqrt (n + 1) + 2 * Real.sqrt n⌋ < ⌊Real.sqrt (9 * n + 3)⌋ := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_and_floor_l2471_247115


namespace NUMINAMATH_CALUDE_spelling_contest_questions_l2471_247187

theorem spelling_contest_questions (drew_correct drew_wrong carla_correct : ℕ) 
  (h1 : drew_correct = 20)
  (h2 : drew_wrong = 6)
  (h3 : carla_correct = 14)
  (h4 : carla_correct + 2 * drew_wrong = drew_correct + drew_wrong) :
  drew_correct + drew_wrong = 26 :=
by sorry

end NUMINAMATH_CALUDE_spelling_contest_questions_l2471_247187


namespace NUMINAMATH_CALUDE_remaining_candy_l2471_247147

def initial_candy : Real := 520.75
def given_away : Real := 234.56

theorem remaining_candy : 
  (initial_candy / 2) - given_away = 25.815 := by sorry

end NUMINAMATH_CALUDE_remaining_candy_l2471_247147


namespace NUMINAMATH_CALUDE_greening_task_equation_l2471_247158

/-- Represents the greening task parameters and equation -/
theorem greening_task_equation (x : ℝ) (h : x > 0) : 
  (600 : ℝ) / (x / (1 + 0.25)) - 600 / x = 30 ↔ 
  60 * (1 + 0.25) / x - 60 / x = 30 :=
by sorry


end NUMINAMATH_CALUDE_greening_task_equation_l2471_247158


namespace NUMINAMATH_CALUDE_median_triangle_inequalities_l2471_247179

-- Define a structure for a triangle with angles
structure Triangle where
  α : Real
  β : Real
  γ : Real

-- Define a structure for a triangle formed from medians
structure MedianTriangle where
  α_m : Real
  β_m : Real
  γ_m : Real

-- Main theorem
theorem median_triangle_inequalities (T : Triangle) (M : MedianTriangle)
  (h1 : T.α > T.β)
  (h2 : T.β > T.γ)
  : T.α > M.α_m ∧
    T.α > M.β_m ∧
    M.γ_m > T.β ∧
    T.β > M.α_m ∧
    M.β_m > T.γ ∧
    M.γ_m > T.γ := by
  sorry

end NUMINAMATH_CALUDE_median_triangle_inequalities_l2471_247179


namespace NUMINAMATH_CALUDE_prob_same_first_last_pancake_l2471_247125

/-- Represents the types of pancake fillings -/
inductive Filling
  | Meat
  | CottageCheese
  | Strawberry

/-- Represents a plate of pancakes -/
structure PlatePancakes where
  total : Nat
  meat : Nat
  cheese : Nat
  strawberry : Nat

/-- Calculates the probability of selecting the same filling for first and last pancake -/
def probSameFirstLast (plate : PlatePancakes) : Rat :=
  sorry

/-- Theorem stating the probability of selecting the same filling for first and last pancake -/
theorem prob_same_first_last_pancake (plate : PlatePancakes) :
  plate.total = 10 ∧ plate.meat = 2 ∧ plate.cheese = 3 ∧ plate.strawberry = 5 →
  probSameFirstLast plate = 14 / 45 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_first_last_pancake_l2471_247125


namespace NUMINAMATH_CALUDE_minimum_red_chips_l2471_247151

/-- Represents the number of chips of each color in the box -/
structure ChipCount where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Checks if the chip count satisfies all given conditions -/
def satisfiesConditions (c : ChipCount) : Prop :=
  c.blue ≥ (3 * c.white) / 4 ∧
  c.blue ≤ c.red / 4 ∧
  60 ≤ c.white + c.blue ∧
  c.white + c.blue ≤ 80

/-- The minimum number of red chips that satisfies all conditions -/
def minRedChips : ℕ := 108

theorem minimum_red_chips :
  ∀ c : ChipCount, satisfiesConditions c → c.red ≥ minRedChips :=
sorry


end NUMINAMATH_CALUDE_minimum_red_chips_l2471_247151


namespace NUMINAMATH_CALUDE_triangle_properties_l2471_247106

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a = 3)
  (h2 : abc.c = 2)
  (h3 : Real.sin abc.A = Real.cos (π/2 - abc.B)) : 
  Real.cos abc.C = 7/9 ∧ 
  (1/2 * abc.a * abc.b * Real.sin abc.C) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2471_247106


namespace NUMINAMATH_CALUDE_second_number_calculation_second_number_is_190_l2471_247186

theorem second_number_calculation : ℝ → Prop :=
  fun x =>
    let first_number : ℝ := 1280
    let twenty_percent_of_650 : ℝ := 0.2 * 650
    let twenty_five_percent_of_first : ℝ := 0.25 * first_number
    x = twenty_five_percent_of_first - twenty_percent_of_650 → x = 190

-- The proof is omitted
theorem second_number_is_190 : ∃ x : ℝ, second_number_calculation x :=
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_second_number_is_190_l2471_247186


namespace NUMINAMATH_CALUDE_quadratic_function_m_l2471_247189

/-- A quadratic function g(x) with integer coefficients -/
def g (d e f : ℤ) (x : ℤ) : ℤ := d * x^2 + e * x + f

/-- The theorem stating that under given conditions, m = -1 -/
theorem quadratic_function_m (d e f m : ℤ) : 
  g d e f 2 = 0 ∧ 
  60 < g d e f 6 ∧ g d e f 6 < 70 ∧
  80 < g d e f 9 ∧ g d e f 9 < 90 ∧
  10000 * m < g d e f 100 ∧ g d e f 100 < 10000 * (m + 1) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_m_l2471_247189


namespace NUMINAMATH_CALUDE_remainder_double_mod_seven_l2471_247123

theorem remainder_double_mod_seven (n : ℤ) (h : n % 7 = 2) : (2 * n) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_double_mod_seven_l2471_247123


namespace NUMINAMATH_CALUDE_alexandra_rearrangement_time_l2471_247195

/-- The number of letters in Alexandra's name -/
def name_length : ℕ := 8

/-- The number of rearrangements Alexandra can write per minute -/
def rearrangements_per_minute : ℕ := 16

/-- Calculate the time required to write all rearrangements in hours -/
def time_to_write_all_rearrangements : ℕ :=
  (Nat.factorial name_length / rearrangements_per_minute) / 60

theorem alexandra_rearrangement_time :
  time_to_write_all_rearrangements = 42 := by sorry

end NUMINAMATH_CALUDE_alexandra_rearrangement_time_l2471_247195


namespace NUMINAMATH_CALUDE_macaron_difference_l2471_247116

/-- The number of macarons made by each person and given to kids --/
structure MacaronProblem where
  mitch : ℕ
  joshua : ℕ
  miles : ℕ
  renz : ℕ
  kids : ℕ
  macarons_per_kid : ℕ

/-- The conditions of the macaron problem --/
def validMacaronProblem (p : MacaronProblem) : Prop :=
  p.mitch = 20 ∧
  p.joshua = p.miles / 2 ∧
  p.joshua > p.mitch ∧
  p.renz = (3 * p.miles) / 4 - 1 ∧
  p.kids = 68 ∧
  p.macarons_per_kid = 2 ∧
  p.mitch + p.joshua + p.miles + p.renz = p.kids * p.macarons_per_kid

/-- The theorem stating the difference between Joshua's and Mitch's macarons --/
theorem macaron_difference (p : MacaronProblem) (h : validMacaronProblem p) :
  p.joshua - p.mitch = 27 := by
  sorry

end NUMINAMATH_CALUDE_macaron_difference_l2471_247116


namespace NUMINAMATH_CALUDE_triangle_side_c_l2471_247124

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_side_c (t : Triangle) 
  (h1 : t.a = 5) 
  (h2 : t.b = 7) 
  (h3 : t.B = 60 * π / 180) : 
  t.c = 8 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_c_l2471_247124


namespace NUMINAMATH_CALUDE_binomial_prob_properties_l2471_247153

/-- A binomial distribution with parameters n and p -/
structure BinomialDist where
  n : ℕ+
  p : ℝ
  h_p_pos : 0 < p
  h_p_lt_one : p < 1

/-- The probability that X is odd in a binomial distribution -/
noncomputable def prob_odd (b : BinomialDist) : ℝ :=
  (1 - (1 - 2*b.p)^b.n.val) / 2

/-- The probability that X is even in a binomial distribution -/
noncomputable def prob_even (b : BinomialDist) : ℝ :=
  1 - prob_odd b

theorem binomial_prob_properties (b : BinomialDist) :
  (prob_odd b + prob_even b = 1) ∧
  (b.p = 1/2 → prob_odd b = prob_even b) ∧
  (0 < b.p ∧ b.p < 1/2 → ∀ m : ℕ+, m < b.n → prob_odd ⟨m, b.p, b.h_p_pos, b.h_p_lt_one⟩ < prob_odd b) :=
by sorry

end NUMINAMATH_CALUDE_binomial_prob_properties_l2471_247153


namespace NUMINAMATH_CALUDE_smallest_n_for_doughnuts_l2471_247172

theorem smallest_n_for_doughnuts : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → (13 * m - 1) % 9 = 0 → m ≥ n) ∧
  (13 * n - 1) % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_doughnuts_l2471_247172


namespace NUMINAMATH_CALUDE_max_value_of_f_inequality_with_sum_constraint_l2471_247193

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

-- Theorem for the maximum value of f
theorem max_value_of_f : ∃ (s : ℝ), s = 3 ∧ ∀ (x : ℝ), f x ≤ s := by sorry

-- Theorem for the inequality
theorem inequality_with_sum_constraint (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) : 
  a^2 + b^2 + c^2 ≥ 3 := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_inequality_with_sum_constraint_l2471_247193


namespace NUMINAMATH_CALUDE_brick_length_calculation_l2471_247149

theorem brick_length_calculation (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_width : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 15 →
  brick_width = 0.1 →
  total_bricks = 18750 →
  (courtyard_length * courtyard_width * 10000) / (total_bricks * brick_width) = 20 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l2471_247149


namespace NUMINAMATH_CALUDE_valid_arrangements_l2471_247111

/-- The number of letters to be arranged -/
def n : ℕ := 8

/-- The number of pairs of repeated letters -/
def k : ℕ := 3

/-- The total number of unrestricted arrangements -/
def total_arrangements : ℕ := n.factorial / (2^k)

/-- The number of arrangements with one pair of identical letters together -/
def arrangements_one_pair : ℕ := k * ((n-1).factorial / (2^(k-1)))

/-- The number of arrangements with two pairs of identical letters together -/
def arrangements_two_pairs : ℕ := (k.choose 2) * ((n-2).factorial / (2^(k-2)))

/-- The number of arrangements with three pairs of identical letters together -/
def arrangements_three_pairs : ℕ := (n-3).factorial

/-- The theorem stating the number of valid arrangements -/
theorem valid_arrangements :
  total_arrangements - arrangements_one_pair + arrangements_two_pairs - arrangements_three_pairs = 2220 :=
sorry

end NUMINAMATH_CALUDE_valid_arrangements_l2471_247111


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2471_247190

def U : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {2,3,5,6}
def B : Set Nat := {1,3,4,6,7}

theorem intersection_A_complement_B : A ∩ (U \ B) = {2,5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2471_247190


namespace NUMINAMATH_CALUDE_journey_distance_l2471_247157

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 15 →
  speed1 = 21 →
  speed2 = 24 →
  ∃ (distance : ℝ),
    distance / 2 / speed1 + distance / 2 / speed2 = total_time ∧
    distance = 336 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l2471_247157


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2471_247121

/-- 
Given a quadratic equation x^2 - mx + m - 1 = 0 with two equal real roots,
prove that m = 2 and the roots are x = 1
-/
theorem quadratic_equal_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - m*x + m - 1 = 0 ∧ 
   ∀ y : ℝ, y^2 - m*y + m - 1 = 0 → y = x) →
  m = 2 ∧ ∃ x : ℝ, x^2 - m*x + m - 1 = 0 ∧ x = 1 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_equal_roots_l2471_247121


namespace NUMINAMATH_CALUDE_union_when_a_is_neg_two_intersection_equals_B_iff_l2471_247180

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 6}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a - 1 ≤ x ∧ x ≤ a + 1}

-- Theorem 1: When a = -2, A ∪ B = {x | -5 ≤ x ≤ 6}
theorem union_when_a_is_neg_two :
  A ∪ B (-2) = {x : ℝ | -5 ≤ x ∧ x ≤ 6} := by sorry

-- Theorem 2: A ∩ B = B if and only if a ≥ -1
theorem intersection_equals_B_iff (a : ℝ) :
  A ∩ B a = B a ↔ a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_union_when_a_is_neg_two_intersection_equals_B_iff_l2471_247180


namespace NUMINAMATH_CALUDE_parabola_y_relationship_l2471_247113

/-- A parabola defined by y = 2(x-1)² + c passing through three points -/
structure Parabola where
  c : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  eq_y₁ : y₁ = 2 * (-2 - 1)^2 + c
  eq_y₂ : y₂ = 2 * (0 - 1)^2 + c
  eq_y₃ : y₃ = 2 * (5/3 - 1)^2 + c

/-- Theorem stating the relationship between y₁, y₂, and y₃ for the given parabola -/
theorem parabola_y_relationship (p : Parabola) : p.y₁ > p.y₂ ∧ p.y₂ > p.y₃ := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_relationship_l2471_247113


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2471_247173

/-- Represents a repeating decimal with a two-digit repeating part -/
def repeating_decimal (a b : ℕ) : ℚ :=
  (10 * a + b) / 99

theorem repeating_decimal_to_fraction :
  repeating_decimal 2 7 = 3 / 11 ∧
  3 + 11 = 14 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2471_247173


namespace NUMINAMATH_CALUDE_max_pieces_is_sixteen_l2471_247140

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 16

/-- The size of a small cake piece in inches -/
def small_piece_size : ℕ := 4

/-- The area of the large cake in square inches -/
def large_cake_area : ℕ := large_cake_size * large_cake_size

/-- The area of a small cake piece in square inches -/
def small_piece_area : ℕ := small_piece_size * small_piece_size

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := large_cake_area / small_piece_area

theorem max_pieces_is_sixteen : max_pieces = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_is_sixteen_l2471_247140


namespace NUMINAMATH_CALUDE_complex_multiplication_simplification_l2471_247171

theorem complex_multiplication_simplification :
  let z₁ : ℂ := 5 + 3 * Complex.I
  let z₂ : ℂ := -2 - 6 * Complex.I
  let z₃ : ℂ := 1 - 2 * Complex.I
  (z₁ - z₂) * z₃ = 25 - 5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_simplification_l2471_247171


namespace NUMINAMATH_CALUDE_correct_product_l2471_247104

theorem correct_product (a b c : ℚ) (h1 : a = 0.25) (h2 : b = 3.4) (h3 : c = 0.85) 
  (h4 : (25 : ℤ) * 34 = 850) : a * b = c := by
  sorry

end NUMINAMATH_CALUDE_correct_product_l2471_247104


namespace NUMINAMATH_CALUDE_distinct_polygons_count_l2471_247152

/-- The number of distinct convex polygons with 4 or more sides that can be drawn
    using some or all of 15 points marked on a circle as vertices -/
def num_polygons : ℕ := 32192

/-- The total number of points marked on the circle -/
def num_points : ℕ := 15

/-- A function that calculates the number of subsets of size k from a set of n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of subsets of 15 points -/
def total_subsets : ℕ := 2^num_points

theorem distinct_polygons_count :
  num_polygons = total_subsets - (choose num_points 0 + choose num_points 1 + 
                                  choose num_points 2 + choose num_points 3) :=
sorry

end NUMINAMATH_CALUDE_distinct_polygons_count_l2471_247152


namespace NUMINAMATH_CALUDE_consecutive_multiples_problem_l2471_247132

/-- Given a set of 50 consecutive multiples of a number, prove that the number is 2 -/
theorem consecutive_multiples_problem (n : ℕ) (s : Set ℕ) : 
  (∃ k : ℕ, s = {k * n | k ∈ Finset.range 50}) →  -- s is a set of 50 consecutive multiples of n
  (56 ∈ s) →  -- The smallest number in s is 56
  (154 ∈ s) →  -- The greatest number in s is 154
  (∀ x ∈ s, 56 ≤ x ∧ x ≤ 154) →  -- All elements in s are between 56 and 154
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_multiples_problem_l2471_247132


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l2471_247120

theorem consecutive_numbers_sum (a : ℤ) : 
  (a + (a + 1) + (a + 2) = 184) ∧
  (a + (a + 1) + (a + 3) = 201) ∧
  (a + (a + 2) + (a + 3) = 212) ∧
  ((a + 1) + (a + 2) + (a + 3) = 226) →
  (a + 3 = 70) := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l2471_247120


namespace NUMINAMATH_CALUDE_point_transformation_l2471_247167

def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def rotate_x_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -z, y)

def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotate_z_90
    |> reflect_xy
    |> reflect_yz
    |> rotate_x_90
    |> reflect_xy

theorem point_transformation :
  transform initial_point = (2, -2, 2) := by sorry

end NUMINAMATH_CALUDE_point_transformation_l2471_247167


namespace NUMINAMATH_CALUDE_no_common_solution_l2471_247185

theorem no_common_solution : ¬∃ x : ℝ, (5*x - 2) / (6*x - 6) = 3/4 ∧ x^2 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l2471_247185


namespace NUMINAMATH_CALUDE_total_animals_l2471_247105

theorem total_animals (a b c : ℕ) (ha : a = 6) (hb : b = 8) (hc : c = 4) :
  a + b + c = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_l2471_247105


namespace NUMINAMATH_CALUDE_problem_solution_l2471_247145

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := (1/3) * x^3 - x
def g (x : ℝ) := 33 * f x + 3 * x

-- Define the sequence bₙ
def b (n : ℕ) : ℝ := g n ^ (1 / g (n + 1))

-- Theorem statement
theorem problem_solution :
  -- f(x) reaches its maximum value 2/3 when x = -1
  (f (-1) = 2/3 ∧ ∀ x, f x ≤ 2/3) ∧
  -- The graph of y = f(x+1) is symmetrical about the point (-1, 0)
  (∀ x, f (x + 1) = -f (-x - 1)) →
  -- 1. f(x) = (1/3)x³ - x is implied by the above conditions
  (∀ x, f x = (1/3) * x^3 - x) ∧
  -- 2. When x > 0, [1 + 1/g(x)]^g(x) < e
  (∀ x > 0, (1 + 1 / g x) ^ (g x) < Real.exp 1) ∧
  -- 3. The sequence bₙ has only one equal pair: b₂ = b₈
  (∀ n m : ℕ, n ≠ m → b n = b m ↔ (n = 2 ∧ m = 8) ∨ (n = 8 ∧ m = 2)) :=
sorry

end

end NUMINAMATH_CALUDE_problem_solution_l2471_247145


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l2471_247114

/-- Given a quadratic equation 2Ax^2 + 3Bx + 4C = 0 with roots r and s,
    prove that the value of p in the equation x^2 + px + q = 0 with roots r^2 and s^2
    is equal to (16AC - 9B^2) / (4A^2) -/
theorem quadratic_root_transformation (A B C : ℝ) (r s : ℝ) :
  (2 * A * r ^ 2 + 3 * B * r + 4 * C = 0) →
  (2 * A * s ^ 2 + 3 * B * s + 4 * C = 0) →
  ∃ q : ℝ, r ^ 2 ^ 2 + ((16 * A * C - 9 * B ^ 2) / (4 * A ^ 2)) * r ^ 2 + q = 0 ∧
           s ^ 2 ^ 2 + ((16 * A * C - 9 * B ^ 2) / (4 * A ^ 2)) * s ^ 2 + q = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l2471_247114


namespace NUMINAMATH_CALUDE_lawn_mowing_time_l2471_247127

theorem lawn_mowing_time (mary_rate tom_rate : ℚ) (mary_time : ℚ) : 
  mary_rate = 1/3 →
  tom_rate = 1/6 →
  mary_time = 1 →
  (1 - mary_rate * mary_time) / tom_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_lawn_mowing_time_l2471_247127


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2471_247178

theorem inequality_solution_set :
  {x : ℝ | x * (x - 1) ≥ x} = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2471_247178


namespace NUMINAMATH_CALUDE_count_valid_triples_l2471_247154

def validTriple (x y z : ℕ+) : Prop :=
  Nat.lcm x.val y.val = 120 ∧ 
  Nat.lcm x.val z.val = 450 ∧ 
  Nat.lcm y.val z.val = 180

theorem count_valid_triples : 
  ∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), 
    (∀ t ∈ s, validTriple t.1 t.2.1 t.2.2) ∧ 
    s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_count_valid_triples_l2471_247154


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l2471_247110

-- Problem 1
theorem problem_one : -1^4 - 7 / (2 - (-3)^2) = 0 := by sorry

-- Problem 2
-- Define a custom type for degrees and minutes
structure DegreeMinute where
  degrees : Int
  minutes : Int

-- Define addition for DegreeMinute
def add_degree_minute (a b : DegreeMinute) : DegreeMinute :=
  let total_minutes := a.minutes + b.minutes
  let extra_degrees := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  ⟨a.degrees + b.degrees + extra_degrees, remaining_minutes⟩

-- Define subtraction for DegreeMinute
def sub_degree_minute (a b : DegreeMinute) : DegreeMinute :=
  let total_minutes_a := a.degrees * 60 + a.minutes
  let total_minutes_b := b.degrees * 60 + b.minutes
  let diff_minutes := total_minutes_a - total_minutes_b
  ⟨diff_minutes / 60, diff_minutes % 60⟩

-- Define multiplication of DegreeMinute by Int
def mul_degree_minute (a : DegreeMinute) (n : Int) : DegreeMinute :=
  let total_minutes := (a.degrees * 60 + a.minutes) * n
  ⟨total_minutes / 60, total_minutes % 60⟩

theorem problem_two :
  sub_degree_minute
    (add_degree_minute ⟨56, 17⟩ ⟨12, 45⟩)
    (mul_degree_minute ⟨16, 21⟩ 4) = ⟨3, 38⟩ := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l2471_247110


namespace NUMINAMATH_CALUDE_weight_replacement_l2471_247131

theorem weight_replacement (n : ℕ) (avg_increase : ℝ) (new_weight : ℝ) :
  n = 10 →
  avg_increase = 6.3 →
  new_weight = 128 →
  ∃ (old_weight : ℝ),
    old_weight = new_weight - n * avg_increase ∧
    old_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l2471_247131


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2471_247108

theorem regular_polygon_sides (n : ℕ) (h : n > 0) :
  (360 : ℝ) / n = 12 → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2471_247108


namespace NUMINAMATH_CALUDE_optimal_pasture_length_l2471_247168

/-- Represents a rectangular cow pasture -/
structure Pasture where
  width : ℝ  -- Width of the pasture (perpendicular to the barn)
  length : ℝ  -- Length of the pasture (parallel to the barn)

/-- Calculates the area of the pasture -/
def Pasture.area (p : Pasture) : ℝ := p.width * p.length

/-- Theorem: The optimal length of the pasture that maximizes the area -/
theorem optimal_pasture_length (total_fence : ℝ) (barn_length : ℝ) :
  total_fence = 240 →
  barn_length = 600 →
  ∃ (optimal : Pasture),
    optimal.length = 120 ∧
    optimal.width = (total_fence - optimal.length) / 2 ∧
    ∀ (p : Pasture),
      p.length + 2 * p.width = total_fence →
      p.area ≤ optimal.area := by
  sorry

end NUMINAMATH_CALUDE_optimal_pasture_length_l2471_247168


namespace NUMINAMATH_CALUDE_common_tangents_count_l2471_247166

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 1 = 0

-- Define the number of common tangents
def num_common_tangents (C₁ C₂ : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  num_common_tangents C₁ C₂ = 3 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l2471_247166


namespace NUMINAMATH_CALUDE_factorization_problems_l2471_247134

theorem factorization_problems :
  (∀ x y : ℝ, xy - 1 - x + y = (y - 1) * (x + 1)) ∧
  (∀ a b : ℝ, (a^2 + b^2)^2 - 4 * a^2 * b^2 = (a + b)^2 * (a - b)^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l2471_247134


namespace NUMINAMATH_CALUDE_new_average_after_doubling_l2471_247150

/-- Theorem: New average after doubling marks -/
theorem new_average_after_doubling (n : ℕ) (original_average : ℝ) :
  n > 0 →
  let total_marks := n * original_average
  let doubled_marks := 2 * total_marks
  let new_average := doubled_marks / n
  new_average = 2 * original_average := by
  sorry

/-- Given problem as an example -/
example : 
  let n : ℕ := 25
  let original_average : ℝ := 70
  let total_marks := n * original_average
  let doubled_marks := 2 * total_marks
  let new_average := doubled_marks / n
  new_average = 140 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_doubling_l2471_247150


namespace NUMINAMATH_CALUDE_hcf_of_156_324_672_l2471_247148

theorem hcf_of_156_324_672 : Nat.gcd 156 (Nat.gcd 324 672) = 12 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_156_324_672_l2471_247148


namespace NUMINAMATH_CALUDE_equality_of_fractions_l2471_247170

theorem equality_of_fractions (x y z k : ℝ) 
  (h : 9 / (x + y) = k / (y + z) ∧ k / (y + z) = 15 / (x - z)) : k = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l2471_247170


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l2471_247143

-- Define an isosceles triangle with one angle of 70°
structure IsoscelesTriangle :=
  (angle1 : Real)
  (angle2 : Real)
  (angle3 : Real)
  (isIsosceles : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3))
  (has70Degree : angle1 = 70 ∨ angle2 = 70 ∨ angle3 = 70)
  (sumIs180 : angle1 + angle2 + angle3 = 180)

-- Theorem statement
theorem isosceles_triangle_angles (t : IsoscelesTriangle) :
  (t.angle1 = 55 ∧ t.angle2 = 55 ∧ t.angle3 = 70) ∨
  (t.angle1 = 55 ∧ t.angle2 = 70 ∧ t.angle3 = 55) ∨
  (t.angle1 = 70 ∧ t.angle2 = 55 ∧ t.angle3 = 55) ∨
  (t.angle1 = 70 ∧ t.angle2 = 70 ∧ t.angle3 = 40) ∨
  (t.angle1 = 70 ∧ t.angle2 = 40 ∧ t.angle3 = 70) ∨
  (t.angle1 = 40 ∧ t.angle2 = 70 ∧ t.angle3 = 70) :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angles_l2471_247143


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2471_247164

/-- Given a geometric sequence {a_n} with common ratio q = 2 and S_n as the sum of the first n terms,
    prove that S_4 / a_2 = 15/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →  -- Definition of sum of geometric sequence
  q = 2 →  -- Given condition
  S 4 / a 2 = 15 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2471_247164


namespace NUMINAMATH_CALUDE_check_cashing_mistake_l2471_247102

theorem check_cashing_mistake (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →
  (10 ≤ y ∧ y ≤ 99) →
  (100 * y + x) - (100 * x + y) = 1820 →
  ∃ x y, y = x + 18 ∧ y = 2 * x :=
sorry

end NUMINAMATH_CALUDE_check_cashing_mistake_l2471_247102


namespace NUMINAMATH_CALUDE_will_summer_earnings_l2471_247182

/-- The amount of money Will spent on mower blades -/
def mower_blades_cost : ℕ := 41

/-- The number of games Will could buy with the remaining money -/
def number_of_games : ℕ := 7

/-- The cost of each game -/
def game_cost : ℕ := 9

/-- The total money Will made mowing lawns -/
def total_money : ℕ := mower_blades_cost + number_of_games * game_cost

theorem will_summer_earnings : total_money = 104 := by
  sorry

end NUMINAMATH_CALUDE_will_summer_earnings_l2471_247182


namespace NUMINAMATH_CALUDE_outlet_pipe_time_l2471_247117

theorem outlet_pipe_time (inlet1 inlet2 outlet : ℚ) 
  (h1 : inlet1 = 1 / 18)
  (h2 : inlet2 = 1 / 20)
  (h3 : inlet1 + inlet2 - outlet = 1 / 12) :
  outlet = 1 / 45 := by
  sorry

end NUMINAMATH_CALUDE_outlet_pipe_time_l2471_247117


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_118_l2471_247139

theorem alpha_plus_beta_equals_118 :
  ∀ α β : ℝ,
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96*x + 2209) / (x^2 + 63*x - 3969)) →
  α + β = 118 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_118_l2471_247139


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l2471_247197

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (Nat.factorial 15 + Nat.factorial 17) ∧ 
  ∀ q : ℕ, q.Prime → q ∣ (Nat.factorial 15 + Nat.factorial 17) → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l2471_247197
