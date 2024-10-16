import Mathlib

namespace NUMINAMATH_CALUDE_solution_to_system_l2011_201196

theorem solution_to_system (x y z : ℝ) :
  3 * (x^2 + y^2 + z^2) = 1 →
  x^2*y^2 + y^2*z^2 + z^2*x^2 = x*y*z*(x + y + z)^3 →
  ((x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ (x = -1/3 ∧ y = -1/3 ∧ z = -1/3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_to_system_l2011_201196


namespace NUMINAMATH_CALUDE_arcade_spending_amount_l2011_201120

def weekly_allowance : ℚ := 345/100

def arcade_spending (x : ℚ) : Prop :=
  let remaining_after_arcade := weekly_allowance - x
  let toy_store_spending := (1/3) * remaining_after_arcade
  let candy_store_spending := 92/100
  remaining_after_arcade - toy_store_spending = candy_store_spending

theorem arcade_spending_amount :
  ∃ (x : ℚ), arcade_spending x ∧ x = 207/100 := by sorry

end NUMINAMATH_CALUDE_arcade_spending_amount_l2011_201120


namespace NUMINAMATH_CALUDE_bonus_distribution_l2011_201127

theorem bonus_distribution (total_amount : ℕ) (total_notes : ℕ) 
  (h1 : total_amount = 160) 
  (h2 : total_notes = 25) : 
  ∃ (x y z : ℕ), 
    x + y + z = total_notes ∧ 
    2*x + 5*y + 10*z = total_amount ∧ 
    y = z ∧ 
    x = 5 ∧ y = 10 ∧ z = 10 := by
  sorry

end NUMINAMATH_CALUDE_bonus_distribution_l2011_201127


namespace NUMINAMATH_CALUDE_leap_year_53_sundays_l2011_201130

/-- The number of days in a leap year -/
def leap_year_days : ℕ := 366

/-- The number of complete weeks in a leap year -/
def complete_weeks : ℕ := leap_year_days / 7

/-- The number of extra days beyond complete weeks in a leap year -/
def extra_days : ℕ := leap_year_days % 7

/-- The number of possible combinations for the extra days -/
def extra_day_combinations : ℕ := 7

/-- The number of combinations that include a Sunday -/
def sunday_combinations : ℕ := 2

/-- The probability of a randomly chosen leap year having 53 Sundays -/
def prob_53_sundays : ℚ := sunday_combinations / extra_day_combinations

theorem leap_year_53_sundays : 
  prob_53_sundays = 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_leap_year_53_sundays_l2011_201130


namespace NUMINAMATH_CALUDE_abs_f_range_l2011_201178

/-- A function whose range is [-2, 3] -/
def f : ℝ → ℝ :=
  sorry

/-- The range of f is [-2, 3] -/
axiom f_range : Set.range f = Set.Icc (-2) 3

/-- Theorem: If the range of f(x) is [-2, 3], then the range of |f(x)| is [0, 3] -/
theorem abs_f_range :
  Set.range (fun x ↦ |f x|) = Set.Icc 0 3 :=
sorry

end NUMINAMATH_CALUDE_abs_f_range_l2011_201178


namespace NUMINAMATH_CALUDE_motion_rate_of_change_l2011_201101

-- Define the law of motion
def s (t : ℝ) : ℝ := 2 * t^2 + 1

-- Define the rate of change function
def rate_of_change (d : ℝ) : ℝ := 4 + 2 * d

-- Theorem statement
theorem motion_rate_of_change (d : ℝ) :
  let t₁ := 1
  let t₂ := 1 + d
  (s t₂ - s t₁) / (t₂ - t₁) = rate_of_change d :=
by sorry

end NUMINAMATH_CALUDE_motion_rate_of_change_l2011_201101


namespace NUMINAMATH_CALUDE_max_product_arithmetic_sequence_l2011_201157

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem max_product_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a6 : a 6 = 4) :
  (∃ x : ℝ, a 4 * a 7 ≤ x) ∧ a 4 * a 7 ≤ 18 ∧ (∃ d : ℝ, a 4 * a 7 = 18) :=
sorry

end NUMINAMATH_CALUDE_max_product_arithmetic_sequence_l2011_201157


namespace NUMINAMATH_CALUDE_largest_non_sum_of_100_composites_l2011_201174

/-- A number is composite if it's the product of two integers greater than 1 -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number can be expressed as the sum of 100 composite numbers -/
def IsSumOf100Composites (n : ℕ) : Prop :=
  ∃ (f : Fin 100 → ℕ), (∀ i, IsComposite (f i)) ∧ n = (Finset.univ.sum f)

/-- 403 is the largest integer that cannot be expressed as the sum of 100 composites -/
theorem largest_non_sum_of_100_composites :
  (¬ IsSumOf100Composites 403) ∧ (∀ n > 403, IsSumOf100Composites n) :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_100_composites_l2011_201174


namespace NUMINAMATH_CALUDE_angle_x_value_l2011_201129

theorem angle_x_value (equilateral_angle : ℝ) (isosceles_vertex : ℝ) (straight_line_sum : ℝ) :
  equilateral_angle = 60 →
  isosceles_vertex = 30 →
  straight_line_sum = 180 →
  ∃ x y : ℝ,
    y + y + isosceles_vertex = straight_line_sum ∧
    x + y + equilateral_angle = straight_line_sum ∧
    x = 45 :=
by sorry

end NUMINAMATH_CALUDE_angle_x_value_l2011_201129


namespace NUMINAMATH_CALUDE_tan_sixty_degrees_l2011_201170

theorem tan_sixty_degrees : Real.tan (60 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sixty_degrees_l2011_201170


namespace NUMINAMATH_CALUDE_total_lunch_is_fifteen_l2011_201141

/-- The total amount spent on lunch given the conditions -/
def total_lunch_amount (your_amount : ℕ) (friend_amount : ℕ) : ℕ :=
  your_amount + friend_amount

/-- Theorem: The total amount spent on lunch is $15 -/
theorem total_lunch_is_fifteen :
  ∃ (your_amount : ℕ),
    (your_amount + 1 = 8) →
    (total_lunch_amount your_amount 8 = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_total_lunch_is_fifteen_l2011_201141


namespace NUMINAMATH_CALUDE_sum_of_digits_l2011_201154

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem sum_of_digits (x y : ℕ) : 
  (x < 10) → 
  (y < 10) → 
  is_divisible_by (653 * 100 + x * 10 + y) 80 → 
  x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_l2011_201154


namespace NUMINAMATH_CALUDE_shirt_cost_calculation_l2011_201182

theorem shirt_cost_calculation :
  let discounted_shirts := 3
  let discounted_shirt_price := 15
  let first_discount := 0.1
  let second_discount := 0.05
  let taxed_shirts := 2
  let taxed_shirt_price := 20
  let first_tax := 0.05
  let second_tax := 0.03

  let discounted_price := discounted_shirt_price * (1 - first_discount) * (1 - second_discount)
  let taxed_price := taxed_shirt_price * (1 + first_tax) * (1 + second_tax)

  let total_cost := discounted_shirts * discounted_price + taxed_shirts * taxed_price

  total_cost = 81.735 := by sorry

end NUMINAMATH_CALUDE_shirt_cost_calculation_l2011_201182


namespace NUMINAMATH_CALUDE_expression_evaluation_l2011_201128

theorem expression_evaluation :
  let x : ℝ := 3
  let y : ℝ := Real.sqrt 3
  (x - 2*y)^2 - (x + 2*y)*(x - 2*y) + 4*x*y = 24 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2011_201128


namespace NUMINAMATH_CALUDE_remainder_256_div_13_l2011_201184

theorem remainder_256_div_13 : ∃ q r : ℤ, 256 = 13 * q + r ∧ 0 ≤ r ∧ r < 13 ∧ r = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_256_div_13_l2011_201184


namespace NUMINAMATH_CALUDE_x_gt_neg_two_necessary_not_sufficient_l2011_201111

theorem x_gt_neg_two_necessary_not_sufficient :
  (∃ x : ℝ, x > -2 ∧ (x + 2) * (x - 3) ≥ 0) ∧
  (∀ x : ℝ, (x + 2) * (x - 3) < 0 → x > -2) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_neg_two_necessary_not_sufficient_l2011_201111


namespace NUMINAMATH_CALUDE_selling_price_ratio_l2011_201194

theorem selling_price_ratio (CP : ℝ) (SP1 SP2 : ℝ) 
  (h1 : SP1 = CP + 0.5 * CP) 
  (h2 : SP2 = CP + 3 * CP) : 
  SP2 / SP1 = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l2011_201194


namespace NUMINAMATH_CALUDE_expression_equality_l2011_201181

theorem expression_equality : 
  (-(-2) + (1 + Real.pi) ^ 0 - |1 - Real.sqrt 2| + Real.sqrt 8 - Real.cos (45 * π / 180)) = 
  2 + 5 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2011_201181


namespace NUMINAMATH_CALUDE_square_difference_l2011_201102

theorem square_difference (a b : ℝ) 
  (h1 : 3 * a + 3 * b = 18) 
  (h2 : a - b = 4) : 
  a^2 - b^2 = 24 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l2011_201102


namespace NUMINAMATH_CALUDE_cookie_recipe_l2011_201149

-- Define the conversion rates
def quart_to_pint : ℚ := 2
def pint_to_cup : ℚ := 1/4

-- Define the recipe for 24 cookies
def milk_for_24 : ℚ := 4  -- in quarts
def sugar_for_24 : ℚ := 6  -- in cups

-- Define the number of cookies we want to bake
def cookies_to_bake : ℚ := 6

-- Define the scaling factor
def scaling_factor : ℚ := cookies_to_bake / 24

-- Theorem to prove
theorem cookie_recipe :
  (milk_for_24 * quart_to_pint * scaling_factor = 2) ∧
  (sugar_for_24 * scaling_factor = 1.5) := by
  sorry


end NUMINAMATH_CALUDE_cookie_recipe_l2011_201149


namespace NUMINAMATH_CALUDE_ship_journey_l2011_201138

theorem ship_journey (D : ℝ) (speed : ℝ) (h1 : D > 0) (h2 : speed = 30) :
  D / 2 - 200 = D / 3 →
  D = 1200 ∧ (D / 2) / speed = 20 :=
by sorry

end NUMINAMATH_CALUDE_ship_journey_l2011_201138


namespace NUMINAMATH_CALUDE_strawberry_rows_l2011_201168

/-- Given that each row of strawberry plants produces 268 kg of fruit
    and the total harvest is 1876 kg, prove that there are 7 rows of strawberry plants. -/
theorem strawberry_rows (yield_per_row : ℕ) (total_harvest : ℕ) 
  (h1 : yield_per_row = 268)
  (h2 : total_harvest = 1876) :
  total_harvest / yield_per_row = 7 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_rows_l2011_201168


namespace NUMINAMATH_CALUDE_prism_in_sphere_lateral_edge_l2011_201119

/-- A prism with a square base and lateral edges perpendicular to the base -/
structure Prism where
  base_side : ℝ
  lateral_edge : ℝ

/-- A sphere -/
structure Sphere where
  radius : ℝ

/-- Theorem: The length of the lateral edge of a prism inscribed in a sphere -/
theorem prism_in_sphere_lateral_edge 
  (p : Prism) 
  (s : Sphere) 
  (h1 : p.base_side = 1) 
  (h2 : s.radius = 1) 
  (h3 : s.radius = Real.sqrt (p.base_side^2 + p.base_side^2 + p.lateral_edge^2) / 2) : 
  p.lateral_edge = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_prism_in_sphere_lateral_edge_l2011_201119


namespace NUMINAMATH_CALUDE_correct_calculation_l2011_201151

theorem correct_calculation (a b : ℝ) : 6 * a^2 * b - b * a^2 = 5 * a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2011_201151


namespace NUMINAMATH_CALUDE_fraction_equality_l2011_201100

theorem fraction_equality : (1012^2 - 1003^2) / (1019^2 - 996^2) = 9 / 23 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2011_201100


namespace NUMINAMATH_CALUDE_estimate_percentage_negative_attitude_l2011_201172

theorem estimate_percentage_negative_attitude 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (negative_attitude_count : ℕ) 
  (h1 : total_population = 2500)
  (h2 : sample_size = 400)
  (h3 : negative_attitude_count = 360) :
  (negative_attitude_count : ℝ) / (sample_size : ℝ) * 100 = 90 := by
sorry

end NUMINAMATH_CALUDE_estimate_percentage_negative_attitude_l2011_201172


namespace NUMINAMATH_CALUDE_fox_kolobok_meeting_l2011_201148

theorem fox_kolobok_meeting (n : ℕ) (m : ℕ) (h1 : n = 14) (h2 : m = 92) :
  ∃ (i j : ℕ) (f : ℕ → ℕ), i ≠ j ∧ i < n ∧ j < n ∧ f i = f j ∧ (∀ k < n, f k ≤ m) :=
by
  sorry

end NUMINAMATH_CALUDE_fox_kolobok_meeting_l2011_201148


namespace NUMINAMATH_CALUDE_divisors_of_300_l2011_201155

/-- Given that 300 = 2 × 2 × 3 × 5 × 5, prove that 300 has 18 divisors -/
theorem divisors_of_300 : ∃ (d : Finset Nat), Finset.card d = 18 ∧ 
  (∀ x : Nat, x ∈ d ↔ (x ∣ 300)) := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_300_l2011_201155


namespace NUMINAMATH_CALUDE_line_parallel_plane_theorem_l2011_201103

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the contained relation for lines and planes
variable (containedInPlane : Line → Plane → Prop)

-- Theorem statement
theorem line_parallel_plane_theorem 
  (a b : Line) (α : Plane) :
  parallelLine a b → parallelLinePlane a α →
  containedInPlane b α ∨ parallelLinePlane b α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_theorem_l2011_201103


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l2011_201113

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l2011_201113


namespace NUMINAMATH_CALUDE_closest_to_500_div_025_l2011_201190

def options : List ℝ := [1000, 1500, 2000, 2500, 3000]

theorem closest_to_500_div_025 :
  ∃ (x : ℝ), x ∈ options ∧ 
  ∀ (y : ℝ), y ∈ options → |x - 500/0.25| ≤ |y - 500/0.25| ∧
  x = 2000 :=
by sorry

end NUMINAMATH_CALUDE_closest_to_500_div_025_l2011_201190


namespace NUMINAMATH_CALUDE_smallest_a_l2011_201185

-- Define the polynomial
def P (a b x : ℤ) : ℤ := x^3 - a*x^2 + b*x - 1806

-- Define the property of having three positive integer roots
def has_three_positive_integer_roots (a b : ℤ) : Prop :=
  ∃ (x y z : ℤ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  P a b x = 0 ∧ P a b y = 0 ∧ P a b z = 0

-- State the theorem
theorem smallest_a :
  ∃ (a : ℤ), has_three_positive_integer_roots a (a*56 - 1806) ∧
  (∀ (a' : ℤ), has_three_positive_integer_roots a' (a'*56 - 1806) → a ≤ a') :=
sorry

end NUMINAMATH_CALUDE_smallest_a_l2011_201185


namespace NUMINAMATH_CALUDE_coin_arrangements_l2011_201161

/-- Represents the number of gold coins -/
def num_gold_coins : Nat := 4

/-- Represents the number of silver coins -/
def num_silver_coins : Nat := 4

/-- Represents the total number of coins -/
def total_coins : Nat := num_gold_coins + num_silver_coins

/-- Calculates the number of ways to arrange gold and silver coins -/
def color_arrangements : Nat := Nat.choose total_coins num_gold_coins

/-- Calculates the number of valid orientations (face up or down) -/
def orientation_arrangements : Nat := total_coins + 1

/-- Theorem: The number of distinguishable arrangements of 8 coins (4 gold and 4 silver)
    stacked so that no two adjacent coins are face to face is 630 -/
theorem coin_arrangements :
  color_arrangements * orientation_arrangements = 630 := by
  sorry

end NUMINAMATH_CALUDE_coin_arrangements_l2011_201161


namespace NUMINAMATH_CALUDE_inequality_solution_abs_inequality_l2011_201118

def f (x : ℝ) := |x - 2|

theorem inequality_solution :
  ∀ x : ℝ, (f x + f (x + 1) ≥ 5) ↔ (x ≥ 4 ∨ x ≤ -1) :=
sorry

theorem abs_inequality :
  ∀ a b : ℝ, |a| > 1 → |a*b - 2| > |a| * |b/a - 2| → |b| > 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_abs_inequality_l2011_201118


namespace NUMINAMATH_CALUDE_smallest_separating_degree_l2011_201144

/-- A point on the coordinate plane with a color -/
structure ColoredPoint where
  x : ℝ
  y : ℝ
  color : Bool  -- True for red, False for blue

/-- A set of N points is permissible if their x-coordinates are distinct -/
def isPermissible (points : Finset ColoredPoint) : Prop :=
  ∀ p q : ColoredPoint, p ∈ points → q ∈ points → p ≠ q → p.x ≠ q.x

/-- A polynomial P separates a set of points if no red points are above
    and no blue points below its graph, or vice versa -/
def separates (P : ℝ → ℝ) (points : Finset ColoredPoint) : Prop :=
  (∀ p ∈ points, p.color = true → P p.x ≥ p.y) ∧
  (∀ p ∈ points, p.color = false → P p.x ≤ p.y) ∨
  (∀ p ∈ points, p.color = true → P p.x ≤ p.y) ∧
  (∀ p ∈ points, p.color = false → P p.x ≥ p.y)

/-- The main theorem: For any N ≥ 3, the smallest degree k of a polynomial
    that can separate any permissible set of N points is N-2 -/
theorem smallest_separating_degree (N : ℕ) (h : N ≥ 3) :
  ∃ k : ℕ, (∀ points : Finset ColoredPoint, points.card = N → isPermissible points →
    ∃ P : ℝ → ℝ, (∃ coeffs : Finset ℝ, coeffs.card ≤ k + 1 ∧
      P = fun x ↦ (coeffs.toList.enum.map (fun (i, a) ↦ a * x ^ i)).sum) ∧
    separates P points) ∧
  (∀ k' : ℕ, k' < k →
    ∃ points : Finset ColoredPoint, points.card = N ∧ isPermissible points ∧
    ∀ P : ℝ → ℝ, (∃ coeffs : Finset ℝ, coeffs.card ≤ k' + 1 ∧
      P = fun x ↦ (coeffs.toList.enum.map (fun (i, a) ↦ a * x ^ i)).sum) →
    ¬separates P points) ∧
  k = N - 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_separating_degree_l2011_201144


namespace NUMINAMATH_CALUDE_inequality_proof_l2011_201183

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^4 + b^4 + c^4 + d^4 - 4*a*b*c*d ≥ 4*(a - b)^2 * Real.sqrt (a*b*c*d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2011_201183


namespace NUMINAMATH_CALUDE_toothpick_pattern_l2011_201116

/-- 
Given a sequence where:
- The first term is 6
- Each successive term increases by 5 more than the previous increase
Prove that the 150th term is equal to 751
-/
theorem toothpick_pattern (n : ℕ) (a : ℕ → ℕ) : 
  a 1 = 6 ∧ 
  (∀ k, k ≥ 1 → a (k + 1) - a k = a k - a (k - 1) + 5) →
  a 150 = 751 :=
sorry

end NUMINAMATH_CALUDE_toothpick_pattern_l2011_201116


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_8_l2011_201156

theorem smallest_four_digit_mod_8 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 8 = 3 → n ≥ 1003 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_8_l2011_201156


namespace NUMINAMATH_CALUDE_matthews_friends_l2011_201145

theorem matthews_friends (total_crackers : ℕ) (crackers_per_friend : ℕ) 
  (h1 : total_crackers = 36) (h2 : crackers_per_friend = 6) : 
  total_crackers / crackers_per_friend = 6 := by
  sorry

end NUMINAMATH_CALUDE_matthews_friends_l2011_201145


namespace NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l2011_201169

-- Define the point P as the intersection of two lines
def P : ℝ × ℝ := (2, 1)

-- Define line l1
def l1 (x y : ℝ) : Prop := 4 * x - y + 1 = 0

-- Define the condition that a line passes through point P
def passes_through_P (a b c : ℝ) : Prop := a * P.1 + b * P.2 + c = 0

-- Theorem for case I
theorem parallel_line_equation :
  ∀ (a b c : ℝ), passes_through_P a b c → (∀ x y, a * x + b * y + c = 0 ↔ 4 * x - y - 7 = 0) → 
  ∃ k, a = 4 * k ∧ b = -k ∧ c = -7 * k := by sorry

-- Theorem for case II
theorem perpendicular_line_equation :
  ∀ (a b c : ℝ), passes_through_P a b c → (∀ x y, a * x + b * y + c = 0 ↔ x + 4 * y - 6 = 0) → 
  ∃ k, a = k ∧ b = 4 * k ∧ c = -6 * k := by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l2011_201169


namespace NUMINAMATH_CALUDE_bag_price_problem_l2011_201106

theorem bag_price_problem (P : ℝ) : 
  (P - P * 0.95 * 0.96 = 44) → P = 500 := by
  sorry

end NUMINAMATH_CALUDE_bag_price_problem_l2011_201106


namespace NUMINAMATH_CALUDE_primitive_root_modulo_power_of_prime_l2011_201147

theorem primitive_root_modulo_power_of_prime
  (p : Nat) (x α : Nat)
  (h_prime : Nat.Prime p)
  (h_alpha : α ≥ 2)
  (h_primitive_root : IsPrimitiveRoot x p)
  (h_not_congruent : ¬ (x^(p^(α-2)*(p-1)) ≡ 1 [MOD p^α])) :
  IsPrimitiveRoot x (p^α) :=
sorry

end NUMINAMATH_CALUDE_primitive_root_modulo_power_of_prime_l2011_201147


namespace NUMINAMATH_CALUDE_proposition_implication_l2011_201131

theorem proposition_implication (P : ℕ+ → Prop) 
  (h1 : ∀ k : ℕ+, P k → P (k + 1)) 
  (h2 : ¬ P 9) : 
  ¬ P 8 := by
  sorry

end NUMINAMATH_CALUDE_proposition_implication_l2011_201131


namespace NUMINAMATH_CALUDE_arithmetic_sequence_odd_numbers_l2011_201137

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_odd_numbers :
  ∀ n : ℕ, n > 0 → arithmetic_sequence 1 2 n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_odd_numbers_l2011_201137


namespace NUMINAMATH_CALUDE_library_book_loans_l2011_201143

theorem library_book_loans (initial_A initial_B initial_C final_A final_B final_C : ℕ)
  (return_rate_A return_rate_B return_rate_C : ℚ) :
  initial_A = 75 →
  initial_B = 100 →
  initial_C = 150 →
  final_A = 54 →
  final_B = 82 →
  final_C = 121 →
  return_rate_A = 65/100 →
  return_rate_B = 1/2 →
  return_rate_C = 7/10 →
  ∃ (loaned_A loaned_B loaned_C : ℕ),
    loaned_A + loaned_B + loaned_C = 420 ∧
    loaned_A ≤ loaned_B ∧
    loaned_B ≤ loaned_C ∧
    (↑loaned_A : ℚ) * return_rate_A = final_A ∧
    (↑loaned_B : ℚ) * return_rate_B = final_B ∧
    (↑loaned_C : ℚ) * return_rate_C = final_C :=
by sorry

end NUMINAMATH_CALUDE_library_book_loans_l2011_201143


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2011_201189

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a + b + c = 6 →
  A = π/3 ∧ a = 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l2011_201189


namespace NUMINAMATH_CALUDE_expression_value_l2011_201125

theorem expression_value (a b c k : ℤ) 
  (ha : a = 10) (hb : b = 15) (hc : c = 3) (hk : k = 2) :
  (a - (b - k * c)) - ((a - b) - k * c) = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2011_201125


namespace NUMINAMATH_CALUDE_complex_inequality_l2011_201188

theorem complex_inequality (x y a b : ℝ) 
  (h1 : x^2 + y^2 ≤ 1) 
  (h2 : a^2 + b^2 ≤ 2) : 
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l2011_201188


namespace NUMINAMATH_CALUDE_min_triangle_area_l2011_201186

/-- Given a line that passes through (1,2) and intersects positive semi-axes, 
    prove that the minimum area of the triangle formed is 4 -/
theorem min_triangle_area (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : 1/m + 2/n = 1) : 
  ∃ (A B : ℝ × ℝ), 
    A.1 > 0 ∧ A.2 = 0 ∧ 
    B.1 = 0 ∧ B.2 > 0 ∧
    (∀ (x y : ℝ), x/m + y/n = 1 → (x = A.1 ∧ y = 0) ∨ (x = 0 ∧ y = B.2)) ∧
    (∀ (C : ℝ × ℝ), C.1 > 0 ∧ C.2 > 0 ∧ C.1/m + C.2/n = 1 → 
      1/2 * A.1 * B.2 ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l2011_201186


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2011_201180

theorem perfect_square_trinomial (k : ℝ) : 
  (∀ x, ∃ a b : ℝ, x^2 - (k-1)*x + 25 = (a*x + b)^2) ↔ (k = 11 ∨ k = -9) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2011_201180


namespace NUMINAMATH_CALUDE_complex_equidistant_modulus_l2011_201117

theorem complex_equidistant_modulus (z : ℂ) : 
  Complex.abs z = Complex.abs (z - 1) ∧ 
  Complex.abs z = Complex.abs (z - Complex.I) → 
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equidistant_modulus_l2011_201117


namespace NUMINAMATH_CALUDE_function_period_l2011_201124

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def condition1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 + x) = f (2 - x)
def condition2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (5 + x) = f (5 - x)

-- Define the period
def is_period (T : ℝ) (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + T) = f x

-- Theorem statement
theorem function_period (f : ℝ → ℝ) 
  (h1 : condition1 f) (h2 : condition2 f) : 
  (∃ T : ℝ, T > 0 ∧ is_period T f ∧ ∀ S : ℝ, S > 0 ∧ is_period S f → T ≤ S) ∧
  (∀ T : ℝ, T > 0 ∧ is_period T f ∧ (∀ S : ℝ, S > 0 ∧ is_period S f → T ≤ S) → T = 6) :=
sorry

end NUMINAMATH_CALUDE_function_period_l2011_201124


namespace NUMINAMATH_CALUDE_apple_box_weight_proof_l2011_201158

/-- The number of apple boxes -/
def num_boxes : ℕ := 7

/-- The number of boxes whose initial weight equals the final weight of all boxes -/
def num_equal_boxes : ℕ := 3

/-- The amount of apples removed from each box (in kg) -/
def removed_weight : ℕ := 20

/-- The initial weight of apples in each box (in kg) -/
def initial_weight : ℕ := 35

theorem apple_box_weight_proof :
  initial_weight * num_boxes - removed_weight * num_boxes = initial_weight * num_equal_boxes :=
by sorry

end NUMINAMATH_CALUDE_apple_box_weight_proof_l2011_201158


namespace NUMINAMATH_CALUDE_function_inequality_l2011_201109

open Set

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_sym : ∀ x, f x = f (2 - x))
  (h_mono : ∀ x ∈ Iio 1, (x - 1) * deriv f x < 0) :
  f 3 < f 0 ∧ f 0 < f (1/2) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l2011_201109


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2011_201152

-- Define the sets M and N
def M : Set ℝ := {x | x + 1 ≥ 0}
def N : Set ℝ := {x | x - 2 < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2011_201152


namespace NUMINAMATH_CALUDE_sum_of_number_and_its_square_l2011_201177

theorem sum_of_number_and_its_square (x : ℝ) : x = 4 → x + x^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_its_square_l2011_201177


namespace NUMINAMATH_CALUDE_sean_has_45_whistles_l2011_201142

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := 13

/-- The number of additional whistles Sean has compared to Charles -/
def sean_additional_whistles : ℕ := 32

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := charles_whistles + sean_additional_whistles

theorem sean_has_45_whistles : sean_whistles = 45 := by
  sorry

end NUMINAMATH_CALUDE_sean_has_45_whistles_l2011_201142


namespace NUMINAMATH_CALUDE_next_signal_time_l2011_201192

def factory_interval : ℕ := 18
def train_interval : ℕ := 24
def lighthouse_interval : ℕ := 36
def start_time : ℕ := 480  -- 8:00 AM in minutes since midnight

def next_simultaneous_signal (f t l s : ℕ) : ℕ :=
  s + Nat.lcm (Nat.lcm f t) l

theorem next_signal_time :
  next_simultaneous_signal factory_interval train_interval lighthouse_interval start_time = 552 := by
  sorry

#eval next_simultaneous_signal factory_interval train_interval lighthouse_interval start_time

end NUMINAMATH_CALUDE_next_signal_time_l2011_201192


namespace NUMINAMATH_CALUDE_sum_of_first_four_terms_l2011_201135

/-- Geometric sequence with given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem sum_of_first_four_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 = 9 →
  a 5 = 243 →
  (a 1) + (a 2) + (a 3) + (a 4) = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_four_terms_l2011_201135


namespace NUMINAMATH_CALUDE_product_inequality_l2011_201139

theorem product_inequality (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 1/2) : 
  ((1 - x) / (1 + x)) * ((1 - y) / (1 + y)) * ((1 - z) / (1 + z)) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l2011_201139


namespace NUMINAMATH_CALUDE_correct_ab_sample_size_l2011_201126

/-- Represents the number of students to be drawn with blood type AB in a stratified sampling -/
def stratified_sample_ab (total_students : ℕ) (ab_students : ℕ) (sample_size : ℕ) : ℕ :=
  (ab_students * sample_size) / total_students

/-- Theorem stating the correct number of AB blood type students in the sample -/
theorem correct_ab_sample_size :
  stratified_sample_ab 500 50 60 = 6 := by sorry

end NUMINAMATH_CALUDE_correct_ab_sample_size_l2011_201126


namespace NUMINAMATH_CALUDE_parabola_symmetric_axis_given_parabola_symmetric_axis_l2011_201193

/-- The symmetric axis of a parabola y = (x - h)^2 + k is x = h -/
theorem parabola_symmetric_axis (h k : ℝ) :
  let f : ℝ → ℝ := λ x => (x - h)^2 + k
  ∃! a : ℝ, ∀ x y : ℝ, f x = f y → |x - a| = |y - a| :=
by sorry

/-- The symmetric axis of the parabola y = (x - 2)^2 + 1 is x = 2 -/
theorem given_parabola_symmetric_axis :
  let f : ℝ → ℝ := λ x => (x - 2)^2 + 1
  ∃! a : ℝ, ∀ x y : ℝ, f x = f y → |x - a| = |y - a| ∧ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_axis_given_parabola_symmetric_axis_l2011_201193


namespace NUMINAMATH_CALUDE_function_relationship_l2011_201132

def f (b c : ℝ) (x : ℝ) : ℝ := x^2 - b*x + c

theorem function_relationship (b c : ℝ) :
  (∀ x, f b c (1 + x) = f b c (1 - x)) →
  f b c 0 = 3 →
  ∀ x, f b c (3^x) ≥ f b c (2^x) := by
  sorry

end NUMINAMATH_CALUDE_function_relationship_l2011_201132


namespace NUMINAMATH_CALUDE_last_two_digits_squares_l2011_201150

theorem last_two_digits_squares (a b : ℕ) :
  (50 ∣ (a + b) ∨ 50 ∣ (a - b)) → a^2 ≡ b^2 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_squares_l2011_201150


namespace NUMINAMATH_CALUDE_residue_neg_1234_mod_31_l2011_201162

theorem residue_neg_1234_mod_31 : Int.mod (-1234) 31 = 6 := by
  sorry

end NUMINAMATH_CALUDE_residue_neg_1234_mod_31_l2011_201162


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2011_201140

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_negative_three :
  reciprocal (-3) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2011_201140


namespace NUMINAMATH_CALUDE_johns_grocery_spending_l2011_201107

theorem johns_grocery_spending (total_spent : ℚ) 
  (meat_fraction : ℚ) (bakery_fraction : ℚ) (candy_spent : ℚ) :
  total_spent = 24 →
  meat_fraction = 1/3 →
  bakery_fraction = 1/6 →
  candy_spent = 6 →
  total_spent - (meat_fraction * total_spent + bakery_fraction * total_spent) - candy_spent = 1/4 * total_spent :=
by sorry

end NUMINAMATH_CALUDE_johns_grocery_spending_l2011_201107


namespace NUMINAMATH_CALUDE_min_value_ab_l2011_201195

theorem min_value_ab (a b : ℝ) (h : 0 < a ∧ 0 < b) (eq : 1/a + 4/b = Real.sqrt (a*b)) : 
  4 ≤ a * b ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 1/a₀ + 4/b₀ = Real.sqrt (a₀*b₀) ∧ a₀ * b₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l2011_201195


namespace NUMINAMATH_CALUDE_inequalities_in_quadrants_I_and_II_exists_points_in_both_quadrants_l2011_201112

/-- Represents a point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Defines the region satisfying the given inequalities -/
def SatisfiesInequalities (p : Point) : Prop :=
  p.y > -2 * p.x + 3 ∧ p.y > 1/2 * p.x + 1

/-- Checks if a point is in Quadrant I -/
def InQuadrantI (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Checks if a point is in Quadrant II -/
def InQuadrantII (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating that all points satisfying the inequalities are in Quadrants I and II -/
theorem inequalities_in_quadrants_I_and_II :
  ∀ p : Point, SatisfiesInequalities p → (InQuadrantI p ∨ InQuadrantII p) :=
by
  sorry

/-- Theorem stating that there exist points in both Quadrants I and II that satisfy the inequalities -/
theorem exists_points_in_both_quadrants :
  (∃ p : Point, SatisfiesInequalities p ∧ InQuadrantI p) ∧
  (∃ p : Point, SatisfiesInequalities p ∧ InQuadrantII p) :=
by
  sorry

end NUMINAMATH_CALUDE_inequalities_in_quadrants_I_and_II_exists_points_in_both_quadrants_l2011_201112


namespace NUMINAMATH_CALUDE_existence_of_special_set_l2011_201134

theorem existence_of_special_set (n : ℕ) (h : n ≥ 2) :
  ∃ (S : Finset ℤ), Finset.card S = n ∧
    ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → (a - b)^2 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_set_l2011_201134


namespace NUMINAMATH_CALUDE_sashas_initial_questions_l2011_201104

/-- Proves that given Sasha's completion rate, work time, and remaining questions,
    the initial number of questions is 60. -/
theorem sashas_initial_questions
  (completion_rate : ℕ)
  (work_time : ℕ)
  (remaining_questions : ℕ)
  (h1 : completion_rate = 15)
  (h2 : work_time = 2)
  (h3 : remaining_questions = 30) :
  completion_rate * work_time + remaining_questions = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_sashas_initial_questions_l2011_201104


namespace NUMINAMATH_CALUDE_matrix_product_equals_A_l2011_201191

variable {R : Type*} [Field R]
variable (d e f x y z : R)

def A : Matrix (Fin 3) (Fin 3) R :=
  ![![0, d, -e],
    ![-d, 0, f],
    ![e, -f, 0]]

def B : Matrix (Fin 3) (Fin 3) R :=
  ![![x^2 + 1, x*y, x*z],
    ![x*y, y^2 + 1, y*z],
    ![x*z, y*z, z^2 + 1]]

theorem matrix_product_equals_A :
  A d e f * B x y z = A d e f := by sorry

end NUMINAMATH_CALUDE_matrix_product_equals_A_l2011_201191


namespace NUMINAMATH_CALUDE_rectangular_solid_width_l2011_201153

/-- The surface area of a rectangular solid given its length, width, and depth. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The width of a rectangular solid with length 9 meters, depth 5 meters, 
    and surface area 314 square meters is 8 meters. -/
theorem rectangular_solid_width : 
  ∃ (w : ℝ), w = 8 ∧ surface_area 9 w 5 = 314 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_width_l2011_201153


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2011_201176

/-- Given a boat traveling downstream with a current of 3 km/hr,
    prove that its speed in still water is 15 km/hr if it travels 3.6 km in 12 minutes. -/
theorem boat_speed_in_still_water : ∀ (b : ℝ),
  (b + 3) * (1 / 5) = 3.6 →
  b = 15 := by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2011_201176


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_theorem_l2011_201167

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-1, 0)
def right_focus : ℝ × ℝ := (1, 0)

-- Define the upper vertex
noncomputable def upper_vertex : ℝ × ℝ := (0, 1)

-- Define the slope condition
def slope_condition (M N : ℝ × ℝ) : Prop :=
  let Q := upper_vertex
  let k_QM := (M.2 - Q.2) / (M.1 - Q.1)
  let k_QN := (N.2 - Q.2) / (N.1 - Q.1)
  k_QM + k_QN = 1

-- Define the fixed point
def fixed_point : ℝ × ℝ := (-2, -1)

-- Theorem statement
theorem ellipse_fixed_point_theorem (M N : ℝ × ℝ) :
  ellipse M.1 M.2 →
  ellipse N.1 N.2 →
  M ≠ upper_vertex →
  N ≠ upper_vertex →
  slope_condition M N →
  ∃ (k t : ℝ), M.2 = k * M.1 + t ∧ N.2 = k * N.1 + t ∧ fixed_point.2 = k * fixed_point.1 + t :=
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_theorem_l2011_201167


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2011_201197

theorem inequality_solution_set 
  (h : ∀ x : ℝ, x^2 - 2*a*x + a > 0) :
  {t : ℝ | a^(t^2 + 2*t - 3) < 1} = {t : ℝ | t < -3 ∨ t > 1} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2011_201197


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2011_201166

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 36) (h3 : x > 0) (h4 : y > 0) :
  1 / x + 1 / y = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2011_201166


namespace NUMINAMATH_CALUDE_brush_cost_is_correct_l2011_201171

/-- The cost of a set of brushes for Maria's painting project -/
def brush_cost : ℝ := 20

/-- The cost of canvas for Maria's painting project -/
def canvas_cost : ℝ := 3 * brush_cost

/-- The cost of paint for Maria's painting project -/
def paint_cost : ℝ := 40

/-- The total cost of materials for Maria's painting project -/
def total_cost : ℝ := brush_cost + canvas_cost + paint_cost

/-- Theorem stating that the brush cost is correct given the problem conditions -/
theorem brush_cost_is_correct :
  brush_cost = 20 ∧
  canvas_cost = 3 * brush_cost ∧
  paint_cost = 40 ∧
  total_cost = 120 := by
  sorry

end NUMINAMATH_CALUDE_brush_cost_is_correct_l2011_201171


namespace NUMINAMATH_CALUDE_common_number_in_list_l2011_201199

theorem common_number_in_list (list : List ℝ) : 
  list.length = 9 →
  (list.take 5).sum / 5 = 7 →
  (list.drop 4).sum / 5 = 9 →
  list.sum / 9 = 73 / 9 →
  ∃ x ∈ list.take 5 ∩ list.drop 4, x = 7 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_list_l2011_201199


namespace NUMINAMATH_CALUDE_range_of_m_l2011_201187

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - 1| < 2 → -1 < x ∧ x < m + 1) ∧ 
   ∃ x, (-1 < x ∧ x < m + 1 ∧ ¬(|x - 1| < 2))) →
  m > 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2011_201187


namespace NUMINAMATH_CALUDE_hyperbola_range_l2011_201133

/-- The equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) + y^2 / (m + 1) = 1 ∧ (m + 2) * (m + 1) < 0

/-- The range of m for which the equation represents a hyperbola -/
theorem hyperbola_range :
  ∀ m : ℝ, is_hyperbola m ↔ -2 < m ∧ m < -1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_range_l2011_201133


namespace NUMINAMATH_CALUDE_modified_lucas_units_digit_l2011_201159

/-- Modified Lucas sequence -/
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 1
  | n + 2 => M (n + 1) + M n + 2

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

theorem modified_lucas_units_digit :
  unitsDigit (M (M 6)) = unitsDigit (M 11) :=
sorry

end NUMINAMATH_CALUDE_modified_lucas_units_digit_l2011_201159


namespace NUMINAMATH_CALUDE_initial_tank_capacity_initial_tank_capacity_solution_l2011_201108

theorem initial_tank_capacity 
  (initial_tanks : ℕ) 
  (additional_tanks : ℕ) 
  (fish_per_additional_tank : ℕ) 
  (total_fish : ℕ) : ℕ :=
  let fish_in_additional_tanks := additional_tanks * fish_per_additional_tank
  let remaining_fish := total_fish - fish_in_additional_tanks
  remaining_fish / initial_tanks

theorem initial_tank_capacity_solution 
  (h1 : initial_tanks = 3)
  (h2 : additional_tanks = 3)
  (h3 : fish_per_additional_tank = 10)
  (h4 : total_fish = 75) :
  initial_tank_capacity initial_tanks additional_tanks fish_per_additional_tank total_fish = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_tank_capacity_initial_tank_capacity_solution_l2011_201108


namespace NUMINAMATH_CALUDE_coefficient_of_x2y_div_3_l2011_201114

/-- Definition of a coefficient in a monomial -/
def coefficient (term : ℚ × (ℕ → ℕ)) : ℚ := term.1

/-- The monomial x^2 * y / 3 -/
def monomial : ℚ × (ℕ → ℕ) := (1/3, fun n => if n = 1 then 2 else if n = 2 then 1 else 0)

/-- Theorem: The coefficient of x^2 * y / 3 is 1/3 -/
theorem coefficient_of_x2y_div_3 : coefficient monomial = 1/3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x2y_div_3_l2011_201114


namespace NUMINAMATH_CALUDE_complex_multiplication_l2011_201163

theorem complex_multiplication (a : ℝ) (h1 : a > 0) (h2 : Complex.abs (1 + a * Complex.I) = Real.sqrt 5) :
  (1 + a * Complex.I) * (1 + Complex.I) = -1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2011_201163


namespace NUMINAMATH_CALUDE_wire_length_ratio_l2011_201179

/-- The length of each wire piece used by Bonnie, in inches -/
def bonnie_wire_length : ℕ := 8

/-- The number of wire pieces used by Bonnie -/
def bonnie_wire_count : ℕ := 12

/-- The length of each wire piece used by Clyde, in inches -/
def clyde_wire_length : ℕ := 2

/-- The side length of Clyde's unit cubes, in inches -/
def clyde_cube_side : ℕ := 1

/-- The number of wire pieces needed for one cube frame -/
def wire_pieces_per_cube : ℕ := 12

theorem wire_length_ratio :
  (bonnie_wire_length * bonnie_wire_count : ℚ) / 
  (clyde_wire_length * wire_pieces_per_cube * bonnie_wire_length ^ 3) = 1 / 128 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l2011_201179


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_not_negative_l2011_201160

theorem absolute_value_of_negative_not_negative (x : ℝ) (h : x < 0) : |x| ≠ x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_not_negative_l2011_201160


namespace NUMINAMATH_CALUDE_fruit_cup_cost_l2011_201164

-- Define the cost of a muffin
def muffin_cost : ℚ := 2

-- Define the number of items each person had
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def kiera_fruit_cups : ℕ := 1

-- Define the total cost of their breakfast
def total_cost : ℚ := 17

-- Theorem to prove
theorem fruit_cup_cost (fruit_cup_cost : ℚ) : 
  (francis_muffins * muffin_cost + francis_fruit_cups * fruit_cup_cost) +
  (kiera_muffins * muffin_cost + kiera_fruit_cups * fruit_cup_cost) = total_cost →
  fruit_cup_cost = 9/5 := by
sorry

end NUMINAMATH_CALUDE_fruit_cup_cost_l2011_201164


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2011_201122

theorem fractional_equation_solution :
  ∀ x : ℝ, x ≠ 3 → ((x - 2) / (x - 3) = 2 / (x - 3) ↔ x = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2011_201122


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l2011_201121

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 → -- angles are complementary
  a = 5 * b → -- ratio of angles is 5:1
  |a - b| = 60 := by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l2011_201121


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2011_201175

-- Problem 1
theorem problem_1 : (Real.sqrt 50 * Real.sqrt 32) / Real.sqrt 8 - 4 * Real.sqrt 2 = 6 * Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 : Real.sqrt 12 - 6 * Real.sqrt (1/3) + Real.sqrt 48 = 4 * Real.sqrt 3 := by sorry

-- Problem 3
theorem problem_3 : (Real.sqrt 5 + 3) * (3 - Real.sqrt 5) - (Real.sqrt 3 - 1)^2 = 2 * Real.sqrt 3 := by sorry

-- Problem 4
theorem problem_4 : (Real.sqrt 24 + Real.sqrt 50) / Real.sqrt 2 - 6 * Real.sqrt (1/3) = 5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2011_201175


namespace NUMINAMATH_CALUDE_option_d_is_deductive_reasoning_l2011_201136

/-- A predicate representing periodic functions --/
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

/-- A predicate representing trigonometric functions --/
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry

/-- Definition of deductive reasoning --/
def IsDeductiveReasoning (premise1 premise2 conclusion : Prop) : Prop :=
  (premise1 ∧ premise2) → conclusion

/-- The tangent function --/
noncomputable def tan : ℝ → ℝ := sorry

/-- Theorem stating that the reasoning in option D is deductive --/
theorem option_d_is_deductive_reasoning :
  IsDeductiveReasoning
    (∀ f, IsTrigonometric f → IsPeriodic f)
    (IsTrigonometric tan)
    (IsPeriodic tan) :=
sorry

end NUMINAMATH_CALUDE_option_d_is_deductive_reasoning_l2011_201136


namespace NUMINAMATH_CALUDE_f_has_zero_in_interval_l2011_201198

def f (x : ℝ) := 3 * x - x^2

theorem f_has_zero_in_interval :
  ∃ x : ℝ, x ∈ Set.Icc (-1) 0 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_zero_in_interval_l2011_201198


namespace NUMINAMATH_CALUDE_factorial_sum_equals_720_l2011_201110

def factorial : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) * factorial n

theorem factorial_sum_equals_720 : 
  5 * factorial 5 + 4 * factorial 4 + factorial 4 = 720 := by
sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_720_l2011_201110


namespace NUMINAMATH_CALUDE_triangle_third_side_existence_l2011_201105

theorem triangle_third_side_existence (a b c : ℝ) : 
  a = 3 → b = 5 → c = 4 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_existence_l2011_201105


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2011_201123

-- Define the line l and circle C
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y - k + 1 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Theorem statement
theorem line_circle_intersection :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
  (∃ (x y : ℝ), line_l k x y ∧ circle_C x y) →
  (∀ (x y : ℝ), line_l k x y → x = 1 ∧ y = 1) ∧
  (∃ (chord_length : ℝ), chord_length = Real.sqrt 8 ∧ 
    ∀ (x y : ℝ), line_l k x y ∧ circle_C x y → 
      (x - A.1)^2 + (y - A.2)^2 ≥ chord_length^2) ∧
  (∃ (max_chord : ℝ), max_chord = 4 ∧
    ∀ (x y : ℝ), line_l k x y ∧ circle_C x y → 
      (x - A.1)^2 + (y - A.2)^2 ≤ max_chord^2) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2011_201123


namespace NUMINAMATH_CALUDE_toby_camera_roll_photos_l2011_201115

/-- The number of photos on Toby's camera roll initially -/
def initial_photos : ℕ := 79

/-- The number of photos Toby deleted initially -/
def deleted_initially : ℕ := 7

/-- The number of photos Toby added of his cat -/
def added_photos : ℕ := 15

/-- The number of photos Toby deleted after editing -/
def deleted_after_editing : ℕ := 3

/-- The final number of photos on Toby's camera roll -/
def final_photos : ℕ := 84

theorem toby_camera_roll_photos :
  initial_photos - deleted_initially + added_photos - deleted_after_editing = final_photos :=
by sorry

end NUMINAMATH_CALUDE_toby_camera_roll_photos_l2011_201115


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2011_201146

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2011_201146


namespace NUMINAMATH_CALUDE_parabola_vertex_l2011_201165

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x + 1)^2 - 6

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-1, -6)

/-- Theorem: The vertex of the parabola y = -2(x+1)^2 - 6 is at the point (-1, -6) -/
theorem parabola_vertex :
  let (h, k) := vertex
  ∀ x y, parabola x y → (x - h)^2 ≤ (y - k) / (-2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2011_201165


namespace NUMINAMATH_CALUDE_phi_subset_singleton_zero_l2011_201173

-- Define Φ as a set
variable (Φ : Set ℕ)

-- Theorem stating that Φ is a subset of {0}
theorem phi_subset_singleton_zero : Φ ⊆ {0} := by
  sorry

end NUMINAMATH_CALUDE_phi_subset_singleton_zero_l2011_201173
