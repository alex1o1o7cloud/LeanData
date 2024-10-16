import Mathlib

namespace NUMINAMATH_CALUDE_angle_equivalence_same_quadrant_as_2016_l3981_398165

theorem angle_equivalence (θ : ℝ) : 
  θ ≡ (θ % 360) [PMOD 360] :=
sorry

theorem same_quadrant_as_2016 : 
  (2016 : ℝ) % 360 = 216 :=
sorry

end NUMINAMATH_CALUDE_angle_equivalence_same_quadrant_as_2016_l3981_398165


namespace NUMINAMATH_CALUDE_jason_pepper_spray_l3981_398161

def total_animals (raccoons : ℕ) (squirrel_multiplier : ℕ) : ℕ :=
  raccoons + raccoons * squirrel_multiplier

theorem jason_pepper_spray :
  total_animals 12 6 = 84 :=
by sorry

end NUMINAMATH_CALUDE_jason_pepper_spray_l3981_398161


namespace NUMINAMATH_CALUDE_quadratic_intersection_range_l3981_398192

/-- The range of a for which the intersection of A and B is non-empty -/
theorem quadratic_intersection_range (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^2 - 2 * x - 2 * a
  let A : Set ℝ := {x | f x > 0}
  let B : Set ℝ := {x | 1 < x ∧ x < 3}
  (A ∩ B).Nonempty → a < -2 ∨ a > 6/7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_range_l3981_398192


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_sum_l3981_398170

theorem geometric_sequence_minimum_sum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- Positive terms
  (a 7 = a 6 + 2 * a 5) →  -- Given condition
  (∃ m n, Real.sqrt (a m * a n) = 4 * a 1) →  -- Given condition
  (∃ min : ℝ, min = 2/3 ∧ ∀ m n, 1/m + 1/n ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_sum_l3981_398170


namespace NUMINAMATH_CALUDE_greatest_divisor_3815_4521_l3981_398182

def is_greatest_divisor (d n1 n2 r1 r2 : ℕ) : Prop :=
  d > 0 ∧
  n1 % d = r1 ∧
  n2 % d = r2 ∧
  ∀ k : ℕ, k > d → (n1 % k ≠ r1 ∨ n2 % k ≠ r2)

theorem greatest_divisor_3815_4521 :
  is_greatest_divisor 64 3815 4521 31 33 := by sorry

end NUMINAMATH_CALUDE_greatest_divisor_3815_4521_l3981_398182


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3981_398135

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 4*x - 1 = 0 ∧ a * y^2 - 4*y - 1 = 0) ↔ 
  (a > -4 ∧ a ≠ 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3981_398135


namespace NUMINAMATH_CALUDE_book_price_increase_l3981_398146

theorem book_price_increase (original_price : ℝ) (h : original_price > 0) :
  let price_after_first_increase := original_price * 1.15
  let final_price := price_after_first_increase * 1.15
  (final_price - original_price) / original_price = 0.3225 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l3981_398146


namespace NUMINAMATH_CALUDE_sum_of_nth_row_l3981_398105

/-- Represents the sum of numbers in the nth row of the triangular array -/
def row_sum (n : ℕ) : ℕ := 2^n

/-- The first row sum is 2 -/
axiom first_row : row_sum 1 = 2

/-- Each subsequent row sum is double the previous row sum -/
axiom double_previous (n : ℕ) : n ≥ 1 → row_sum (n + 1) = 2 * row_sum n

/-- The sum of numbers in the nth row of the triangular array is 2^n -/
theorem sum_of_nth_row (n : ℕ) : n ≥ 1 → row_sum n = 2^n := by sorry

end NUMINAMATH_CALUDE_sum_of_nth_row_l3981_398105


namespace NUMINAMATH_CALUDE_odd_power_difference_divisibility_l3981_398103

theorem odd_power_difference_divisibility (a b : ℕ) (ha : Odd a) (hb : Odd b) :
  ∃ k : ℕ, (2^2018 ∣ b^k - a^2) ∨ (2^2018 ∣ a^k - b^2) := by
  sorry

end NUMINAMATH_CALUDE_odd_power_difference_divisibility_l3981_398103


namespace NUMINAMATH_CALUDE_second_month_sale_l3981_398190

def average_sale : ℕ := 6800
def num_months : ℕ := 6
def sale_month1 : ℕ := 6435
def sale_month3 : ℕ := 7230
def sale_month4 : ℕ := 6562
def sale_month5 : ℕ := 6791
def sale_month6 : ℕ := 6791

theorem second_month_sale :
  ∃ (sale_month2 : ℕ),
    sale_month2 = 13991 ∧
    (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / num_months = average_sale :=
by sorry

end NUMINAMATH_CALUDE_second_month_sale_l3981_398190


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l3981_398178

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l3981_398178


namespace NUMINAMATH_CALUDE_find_a_l3981_398176

-- Define the complex numbers a, b, c
variable (a b c : ℂ)

-- Define the conditions
def condition1 : Prop := a + b + c = 5
def condition2 : Prop := a * b + b * c + c * a = 7
def condition3 : Prop := a * b * c = 6
def condition4 : Prop := a.im = 0  -- a is real

-- Theorem statement
theorem find_a (h1 : condition1 a b c) (h2 : condition2 a b c) 
                (h3 : condition3 a b c) (h4 : condition4 a) : 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_find_a_l3981_398176


namespace NUMINAMATH_CALUDE_line_with_equal_intercepts_through_intersection_l3981_398174

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The intersection point of two lines -/
def intersection (l1 l2 : Line) : ℝ × ℝ := sorry

/-- Check if a point lies on a line -/
def on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop := sorry

/-- Check if a line has equal intercepts on coordinate axes -/
def equal_intercepts (l : Line) : Prop := sorry

theorem line_with_equal_intercepts_through_intersection 
  (l1 l2 : Line) 
  (h1 : l1 = Line.mk 1 2 (-11)) 
  (h2 : l2 = Line.mk 2 1 (-10)) :
  ∃ (l : Line), 
    on_line (intersection l1 l2) l ∧ 
    equal_intercepts l ∧ 
    (l = Line.mk 4 (-3) 0 ∨ l = Line.mk 1 1 (-7)) := by
  sorry

end NUMINAMATH_CALUDE_line_with_equal_intercepts_through_intersection_l3981_398174


namespace NUMINAMATH_CALUDE_roots_inequality_l3981_398173

theorem roots_inequality (m : ℝ) (x₁ x₂ : ℝ) (hm : m < -2) 
  (hx : x₁ < x₂) (hf₁ : Real.log x₁ - x₁ = m) (hf₂ : Real.log x₂ - x₂ = m) :
  x₁ * x₂^2 < 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_inequality_l3981_398173


namespace NUMINAMATH_CALUDE_range_of_m_l3981_398139

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  ∀ m : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m) ↔ m < 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3981_398139


namespace NUMINAMATH_CALUDE_least_number_divisible_by_first_five_primes_l3981_398150

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

def is_divisible_by_all (n : Nat) (list : List Nat) : Prop :=
  ∀ m ∈ list, n % m = 0

theorem least_number_divisible_by_first_five_primes :
  ∃ n : Nat, n > 0 ∧ is_divisible_by_all n first_five_primes ∧
  ∀ k : Nat, k > 0 ∧ is_divisible_by_all k first_five_primes → n ≤ k :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_first_five_primes_l3981_398150


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3981_398121

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27 * x^3) 
  (h3 : a - b = 2 * x) : 
  a = 3.041 * x ∨ a = -1.041 * x := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3981_398121


namespace NUMINAMATH_CALUDE_unique_solution_diophantine_equation_l3981_398120

theorem unique_solution_diophantine_equation :
  ∀ x y : ℕ+, 2 * x^2 + 5 * y^2 = 11 * (x * y - 11) ↔ x = 14 ∧ y = 27 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_diophantine_equation_l3981_398120


namespace NUMINAMATH_CALUDE_l_shaped_area_l3981_398137

/-- The area of an L-shaped region formed by subtracting three squares from a larger square -/
theorem l_shaped_area (total_side : ℝ) (small_side1 small_side2 large_side : ℝ) :
  total_side = 7 ∧ 
  small_side1 = 2 ∧ 
  small_side2 = 2 ∧ 
  large_side = 5 →
  total_side^2 - (small_side1^2 + small_side2^2 + large_side^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_l3981_398137


namespace NUMINAMATH_CALUDE_power_sum_equality_l3981_398166

theorem power_sum_equality : 2^123 + 8^5 / 8^3 = 2^123 + 64 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3981_398166


namespace NUMINAMATH_CALUDE_unknown_denomination_is_five_l3981_398197

/-- Represents Marly's bill distribution and the resulting $100 bills -/
structure BillDistribution where
  twenty_bills : ℕ
  ten_bills : ℕ
  unknown_bills : ℕ
  hundred_bills : ℕ

/-- Calculates the total value of bills -/
def total_value (d : BillDistribution) (unknown_denomination : ℕ) : ℕ :=
  d.twenty_bills * 20 + d.ten_bills * 10 + d.unknown_bills * unknown_denomination

/-- Theorem stating that the unknown denomination is $5 -/
theorem unknown_denomination_is_five (d : BillDistribution) 
    (h1 : d.twenty_bills = 10)
    (h2 : d.ten_bills = 8)
    (h3 : d.unknown_bills = 4)
    (h4 : d.hundred_bills = 3)
    (h5 : total_value d 5 = d.hundred_bills * 100) :
  5 = (d.hundred_bills * 100 - (d.twenty_bills * 20 + d.ten_bills * 10)) / d.unknown_bills :=
by
  sorry

end NUMINAMATH_CALUDE_unknown_denomination_is_five_l3981_398197


namespace NUMINAMATH_CALUDE_smallest_cube_with_more_than_half_remaining_l3981_398133

theorem smallest_cube_with_more_than_half_remaining : 
  ∀ n : ℕ, n > 0 → ((n : ℚ) - 4)^3 > (n : ℚ)^3 / 2 ↔ n ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_with_more_than_half_remaining_l3981_398133


namespace NUMINAMATH_CALUDE_basketball_lineup_selections_l3981_398129

/-- The number of ways to select a starting lineup from a basketball team -/
def starting_lineup_selections (total_players : ℕ) (lineup_size : ℕ) (captain_count : ℕ) : ℕ :=
  total_players * (Nat.choose (total_players - 1) (lineup_size - 1))

/-- Theorem: The number of ways to select a starting lineup of 5 players from a team of 12,
    where one player is designated as the captain and the other four positions are interchangeable,
    is equal to 3960. -/
theorem basketball_lineup_selections :
  starting_lineup_selections 12 5 1 = 3960 := by
  sorry

#eval starting_lineup_selections 12 5 1

end NUMINAMATH_CALUDE_basketball_lineup_selections_l3981_398129


namespace NUMINAMATH_CALUDE_total_fish_bought_is_89_l3981_398107

/-- Represents the number of fish bought on each visit -/
structure FishPurchase where
  goldfish : Nat
  bluefish : Nat
  greenfish : Nat
  purplefish : Nat
  redfish : Nat

/-- Calculates the total number of fish in a purchase -/
def totalFish (purchase : FishPurchase) : Nat :=
  purchase.goldfish + purchase.bluefish + purchase.greenfish + purchase.purplefish + purchase.redfish

/-- Theorem: The total number of fish Roden bought is 89 -/
theorem total_fish_bought_is_89 
  (visit1 : FishPurchase := { goldfish := 15, bluefish := 7, greenfish := 0, purplefish := 0, redfish := 0 })
  (visit2 : FishPurchase := { goldfish := 10, bluefish := 12, greenfish := 5, purplefish := 0, redfish := 0 })
  (visit3 : FishPurchase := { goldfish := 3, bluefish := 7, greenfish := 9, purplefish := 0, redfish := 0 })
  (visit4 : FishPurchase := { goldfish := 4, bluefish := 8, greenfish := 6, purplefish := 2, redfish := 1 }) :
  totalFish visit1 + totalFish visit2 + totalFish visit3 + totalFish visit4 = 89 := by
  sorry


end NUMINAMATH_CALUDE_total_fish_bought_is_89_l3981_398107


namespace NUMINAMATH_CALUDE_victor_sticker_count_l3981_398134

/-- The number of stickers Victor has -/
def total_stickers (flower animal insect space : ℕ) : ℕ :=
  flower + animal + insect + space

theorem victor_sticker_count :
  ∀ (flower animal insect space : ℕ),
    flower = 12 →
    animal = 8 →
    insect = animal - 3 →
    space = flower + 7 →
    total_stickers flower animal insect space = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_victor_sticker_count_l3981_398134


namespace NUMINAMATH_CALUDE_decimal_point_shift_l3981_398172

theorem decimal_point_shift (x : ℝ) : (x / 10 = x - 0.72) → x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_shift_l3981_398172


namespace NUMINAMATH_CALUDE_tangent_circles_a_values_l3981_398193

/-- Two circles that intersect at exactly one point -/
structure TangentCircles where
  /-- The parameter 'a' in the equation of the second circle -/
  a : ℝ
  /-- The first circle: x^2 + y^2 = 4 -/
  circle1 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ x^2 + y^2 = 4
  /-- The second circle: (x-a)^2 + y^2 = 1 -/
  circle2 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - a)^2 + y^2 = 1
  /-- The circles intersect at exactly one point -/
  intersect_at_one_point : ∃! p : ℝ × ℝ, circle1 p.1 p.2 ∧ circle2 p.1 p.2

/-- The theorem stating that 'a' must be in the set {1, -1, 3, -3} -/
theorem tangent_circles_a_values (tc : TangentCircles) : tc.a = 1 ∨ tc.a = -1 ∨ tc.a = 3 ∨ tc.a = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_a_values_l3981_398193


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l3981_398128

def stamp_price : ℕ := 45
def available_money : ℕ := 4500  -- $45 in cents

theorem max_stamps_purchasable :
  ∀ n : ℕ, n * stamp_price ≤ available_money → n ≤ 100 ∧
  ∃ m : ℕ, m = 100 ∧ m * stamp_price ≤ available_money :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l3981_398128


namespace NUMINAMATH_CALUDE_boat_license_combinations_l3981_398145

def possible_letters : Nat := 3
def digits_per_license : Nat := 6
def possible_digits : Nat := 10

theorem boat_license_combinations :
  possible_letters * possible_digits ^ digits_per_license = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_combinations_l3981_398145


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l3981_398171

theorem smallest_cube_root_with_small_fraction (n : ℕ) (r : ℝ) : 
  (0 < n) →
  (0 < r) →
  (r < 1 / 500) →
  (∃ m : ℕ, (n + r)^3 = m) →
  (∀ k < n, ∀ s : ℝ, (0 < s) → (s < 1 / 500) → (∃ l : ℕ, (k + s)^3 = l) → False) →
  n = 17 := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l3981_398171


namespace NUMINAMATH_CALUDE_trapezoid_xy_length_l3981_398104

-- Define the trapezoid and its properties
structure Trapezoid where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  wx_parallel_zy : (X.1 - W.1) * (Y.2 - Z.2) = (X.2 - W.2) * (Y.1 - Z.1)
  wy_perp_zy : (W.1 - Y.1) * (Z.1 - Y.1) + (W.2 - Y.2) * (Z.2 - Y.2) = 0

-- Define the given conditions
def trapezoid_conditions (t : Trapezoid) : Prop :=
  let (_, y2) := t.Y
  let (_, z2) := t.Z
  let yz_length := Real.sqrt ((t.Y.1 - t.Z.1)^2 + (y2 - z2)^2)
  let tan_z := (t.W.2 - t.Z.2) / (t.W.1 - t.Z.1)
  let tan_x := (t.W.2 - t.X.2) / (t.X.1 - t.W.1)
  yz_length = 15 ∧ tan_z = 2 ∧ tan_x = 2.5

-- State the theorem
theorem trapezoid_xy_length (t : Trapezoid) (h : trapezoid_conditions t) :
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 6 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_xy_length_l3981_398104


namespace NUMINAMATH_CALUDE_xiaoming_class_ratio_l3981_398142

theorem xiaoming_class_ratio (n : ℕ) (h1 : 30 < n) (h2 : n < 40) : ¬ ∃ k : ℕ, n = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_class_ratio_l3981_398142


namespace NUMINAMATH_CALUDE_inequalities_proof_l3981_398151

theorem inequalities_proof :
  (∀ x : ℝ, 2 * x^2 + 5 * x + 3 > x^2 + 3 * x + 1) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → Real.sqrt a > Real.sqrt b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3981_398151


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3981_398179

theorem polynomial_factorization (x : ℝ) : 
  x^4 - 64 = (x - 2 * Real.sqrt 2) * (x + 2 * Real.sqrt 2) * (x^2 + 8) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3981_398179


namespace NUMINAMATH_CALUDE_constant_killing_time_l3981_398187

/-- The time it takes for lions to kill deers -/
def killing_time (n : ℕ) : ℝ :=
  14

/-- Given conditions -/
axiom condition_14 : killing_time 14 = 14
axiom condition_100 : killing_time 100 = 14

/-- Theorem: For any positive number of lions, it takes 14 minutes to kill the same number of deers -/
theorem constant_killing_time (n : ℕ) (h : n > 0) : killing_time n = 14 := by
  sorry

end NUMINAMATH_CALUDE_constant_killing_time_l3981_398187


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_diff_l3981_398117

-- Define the repeating decimals
def repeating_234 : ℚ := 234 / 999
def repeating_567 : ℚ := 567 / 999
def repeating_891 : ℚ := 891 / 999

-- State the theorem
theorem repeating_decimal_sum_diff : 
  repeating_234 + repeating_567 - repeating_891 = -10 / 111 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_diff_l3981_398117


namespace NUMINAMATH_CALUDE_preimage_of_two_negative_four_l3981_398116

/-- A function that maps (x, y) to (x-y, x+y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

/-- Theorem stating that the preimage of (2, -4) under f is (-1, -3) -/
theorem preimage_of_two_negative_four (p : ℝ × ℝ) :
  f p = (2, -4) ↔ p = (-1, -3) := by sorry

end NUMINAMATH_CALUDE_preimage_of_two_negative_four_l3981_398116


namespace NUMINAMATH_CALUDE_modular_congruence_l3981_398199

theorem modular_congruence (x : ℤ) : 
  (5 * x + 9) % 18 = 4 → (3 * x + 15) % 18 = 12 := by sorry

end NUMINAMATH_CALUDE_modular_congruence_l3981_398199


namespace NUMINAMATH_CALUDE_leet_puzzle_solution_l3981_398155

theorem leet_puzzle_solution :
  ∀ (L E T M : ℕ),
    L ≠ 0 →
    L < 10 ∧ E < 10 ∧ T < 10 ∧ M < 10 →
    1000 * L + 110 * E + T + 100 * L + 10 * M + T = 1000 * T + L →
    T = L + 1 →
    1000 * E + 100 * L + 10 * M + 0 = 1880 :=
by
  sorry

end NUMINAMATH_CALUDE_leet_puzzle_solution_l3981_398155


namespace NUMINAMATH_CALUDE_solve_equation_l3981_398180

theorem solve_equation (x : ℚ) : 
  (x - 30) / 2 = (5 - 3*x) / 6 + 2 → x = 167/6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3981_398180


namespace NUMINAMATH_CALUDE_lily_milk_problem_l3981_398185

theorem lily_milk_problem (initial_milk : ℚ) (given_milk : ℚ) (remaining_milk : ℚ) : 
  initial_milk = 5 ∧ given_milk = 18/4 ∧ remaining_milk = initial_milk - given_milk → remaining_milk = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_lily_milk_problem_l3981_398185


namespace NUMINAMATH_CALUDE_variance_scaled_sample_l3981_398102

variable (s : ℝ) (x : Fin 5 → ℝ)

def variance (x : Fin 5 → ℝ) : ℝ := sorry

def scaled_sample (x : Fin 5 → ℝ) : Fin 5 → ℝ := fun i => 2 * x i

theorem variance_scaled_sample (h : variance x = 3) : 
  variance (scaled_sample x) = 12 := by sorry

end NUMINAMATH_CALUDE_variance_scaled_sample_l3981_398102


namespace NUMINAMATH_CALUDE_intersecting_sphere_yz_radius_l3981_398156

/-- A sphere intersecting two planes -/
structure IntersectingSphere where
  /-- Center of the intersection circle with xy-plane -/
  xy_center : ℝ × ℝ × ℝ
  /-- Radius of the intersection circle with xy-plane -/
  xy_radius : ℝ
  /-- Center of the intersection circle with yz-plane -/
  yz_center : ℝ × ℝ × ℝ
  /-- Radius of the intersection circle with yz-plane -/
  yz_radius : ℝ

/-- The theorem stating the radius of the yz-plane intersection -/
theorem intersecting_sphere_yz_radius (sphere : IntersectingSphere) 
  (h1 : sphere.xy_center = (3, 5, 0))
  (h2 : sphere.xy_radius = 2)
  (h3 : sphere.yz_center = (0, 5, -8)) :
  sphere.yz_radius = Real.sqrt 59 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_sphere_yz_radius_l3981_398156


namespace NUMINAMATH_CALUDE_sin_75_plus_sin_15_l3981_398112

theorem sin_75_plus_sin_15 : Real.sin (75 * π / 180) + Real.sin (15 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_plus_sin_15_l3981_398112


namespace NUMINAMATH_CALUDE_choir_performance_theorem_l3981_398140

/-- Represents the number of singers joining in each verse of a choir performance --/
structure ChoirPerformance where
  total : ℕ
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- Calculates the number of singers joining in the fifth verse --/
def fifthVerseSingers (c : ChoirPerformance) : ℕ :=
  c.total - (c.first + c.second + c.third + c.fourth)

/-- Theorem stating the number of singers joining in the fifth verse --/
theorem choir_performance_theorem (c : ChoirPerformance) 
  (h_total : c.total = 60)
  (h_first : c.first = c.total / 2)
  (h_second : c.second = (c.total - c.first) / 3)
  (h_third : c.third = (c.total - c.first - c.second) / 4)
  (h_fourth : c.fourth = (c.total - c.first - c.second - c.third) / 5) :
  fifthVerseSingers c = 12 := by
  sorry

#eval fifthVerseSingers { total := 60, first := 30, second := 10, third := 5, fourth := 3, fifth := 12 }

end NUMINAMATH_CALUDE_choir_performance_theorem_l3981_398140


namespace NUMINAMATH_CALUDE_sum_and_difference_problem_l3981_398100

theorem sum_and_difference_problem (a b : ℤ) : 
  a + b = 56 → 
  a = b + 12 → 
  a = 22 → 
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_and_difference_problem_l3981_398100


namespace NUMINAMATH_CALUDE_sally_monday_seashells_l3981_398164

/-- The number of seashells Sally picked on Monday -/
def monday_seashells : ℕ := sorry

/-- The number of seashells Sally picked on Tuesday -/
def tuesday_seashells : ℕ := sorry

/-- The price of each seashell in dollars -/
def seashell_price : ℚ := 6/5

/-- The total amount Sally can make by selling all seashells in dollars -/
def total_amount : ℕ := 54

/-- Theorem stating the number of seashells Sally picked on Monday -/
theorem sally_monday_seashells : 
  monday_seashells = 30 ∧
  tuesday_seashells = monday_seashells / 2 ∧
  seashell_price * (monday_seashells + tuesday_seashells : ℚ) = total_amount := by
  sorry

end NUMINAMATH_CALUDE_sally_monday_seashells_l3981_398164


namespace NUMINAMATH_CALUDE_gcd_20020_11011_l3981_398126

theorem gcd_20020_11011 : Nat.gcd 20020 11011 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_20020_11011_l3981_398126


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l3981_398160

theorem cube_sum_inequality (n : ℕ) : 
  (∀ a b c : ℕ, (a + b + c)^3 ≤ n * (a^3 + b^3 + c^3)) ↔ n ≥ 9 := by sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l3981_398160


namespace NUMINAMATH_CALUDE_water_drinkers_l3981_398147

theorem water_drinkers (total : ℕ) (juice_percent : ℚ) (water_percent : ℚ) (juice_drinkers : ℕ) : ℕ :=
  let water_drinkers : ℕ := 60
  have h1 : juice_percent = 70 / 100 := by sorry
  have h2 : water_percent = 30 / 100 := by sorry
  have h3 : juice_percent + water_percent = 1 := by sorry
  have h4 : juice_drinkers = 140 := by sorry
  have h5 : ↑juice_drinkers / ↑total = juice_percent := by sorry
  have h6 : ↑water_drinkers / ↑total = water_percent := by sorry
  water_drinkers

#check water_drinkers

end NUMINAMATH_CALUDE_water_drinkers_l3981_398147


namespace NUMINAMATH_CALUDE_inequality_problem_l3981_398127

theorem inequality_problem (r p q : ℝ) 
  (hr : r < 0) 
  (hpq : p * q ≠ 0) 
  (hineq : p^2 * r > q^2 * r) : 
  ¬((-p > -q) ∧ (-p < q) ∧ (1 < -q/p) ∧ (1 > q/p)) :=
sorry

end NUMINAMATH_CALUDE_inequality_problem_l3981_398127


namespace NUMINAMATH_CALUDE_sector_area_given_arc_length_l3981_398159

/-- Given a circular sector where the arc length corresponding to a central angle of 2 radians is 4 cm, 
    the area of this sector is 4 cm². -/
theorem sector_area_given_arc_length (r : ℝ) (h : 2 * r = 4) : r * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_given_arc_length_l3981_398159


namespace NUMINAMATH_CALUDE_average_molar_mass_of_compound_l3981_398186

/-- Given a compound where 4 moles weigh 672 grams, prove that its average molar mass is 168 grams/mole -/
theorem average_molar_mass_of_compound (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 672)
  (h2 : num_moles = 4) :
  total_weight / num_moles = 168 := by
  sorry

end NUMINAMATH_CALUDE_average_molar_mass_of_compound_l3981_398186


namespace NUMINAMATH_CALUDE_min_moves_to_equalize_l3981_398125

/-- Represents a casket with coins -/
structure Casket :=
  (coins : ℕ)

/-- Represents the circular arrangement of caskets -/
def CasketCircle := Vector Casket 7

/-- A move transfers one coin between neighboring caskets -/
def Move := Fin 7 → Fin 7

/-- Checks if a move is valid (transfers to a neighboring casket) -/
def isValidMove (m : Move) : Prop :=
  ∀ i, m i = (i + 1) % 7 ∨ m i = (i + 6) % 7 ∨ m i = i

/-- Applies a move to a casket circle -/
def applyMove (circle : CasketCircle) (m : Move) : CasketCircle :=
  sorry

/-- Checks if all caskets have the same number of coins -/
def isEqualized (circle : CasketCircle) : Prop :=
  ∀ i j, (circle.get i).coins = (circle.get j).coins

/-- The initial arrangement of caskets -/
def initialCircle : CasketCircle :=
  Vector.ofFn (λ i => match i with
    | 0 => ⟨9⟩
    | 1 => ⟨17⟩
    | 2 => ⟨12⟩
    | 3 => ⟨5⟩
    | 4 => ⟨18⟩
    | 5 => ⟨10⟩
    | 6 => ⟨20⟩)

/-- The main theorem to be proved -/
theorem min_moves_to_equalize :
  ∃ (moves : List Move),
    moves.length = 22 ∧
    (∀ m ∈ moves, isValidMove m) ∧
    isEqualized (moves.foldl applyMove initialCircle) ∧
    (∀ (otherMoves : List Move),
      otherMoves.length < 22 →
      ¬isEqualized (otherMoves.foldl applyMove initialCircle)) :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_equalize_l3981_398125


namespace NUMINAMATH_CALUDE_max_eggs_per_basket_l3981_398191

def total_red_eggs : ℕ := 30
def total_blue_eggs : ℕ := 45
def min_eggs_per_basket : ℕ := 5

theorem max_eggs_per_basket :
  ∃ (n : ℕ), n ≥ min_eggs_per_basket ∧
             n ∣ total_red_eggs ∧
             n ∣ total_blue_eggs ∧
             ∀ (m : ℕ), m ≥ min_eggs_per_basket ∧
                        m ∣ total_red_eggs ∧
                        m ∣ total_blue_eggs →
                        m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_eggs_per_basket_l3981_398191


namespace NUMINAMATH_CALUDE_abcd_16_bits_l3981_398154

def base_16_to_decimal (a b c d : ℕ) : ℕ :=
  a * 16^3 + b * 16^2 + c * 16 + d

def bits_required (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem abcd_16_bits :
  bits_required (base_16_to_decimal 10 11 12 13) = 16 := by
  sorry

end NUMINAMATH_CALUDE_abcd_16_bits_l3981_398154


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l3981_398198

theorem square_difference_of_sum_and_diff (x y : ℝ) 
  (sum_eq : x + y = 5) 
  (diff_eq : x - y = 10) : 
  x^2 - y^2 = 50 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l3981_398198


namespace NUMINAMATH_CALUDE_billy_can_play_24_songs_l3981_398194

/-- The number of songs in Billy's music book -/
def total_songs : ℕ := 52

/-- The number of songs Billy still needs to learn -/
def songs_to_learn : ℕ := 28

/-- The number of songs Billy can play -/
def playable_songs : ℕ := total_songs - songs_to_learn

theorem billy_can_play_24_songs : playable_songs = 24 := by
  sorry

end NUMINAMATH_CALUDE_billy_can_play_24_songs_l3981_398194


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l3981_398136

/-- The area of an equilateral triangle with altitude √15 is 5√3 -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = Real.sqrt 15) :
  let base := 2 * Real.sqrt 5
  let area := (1 / 2) * base * h
  area = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l3981_398136


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3981_398106

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 → Complex.im ((2 : ℂ) + i) / i = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3981_398106


namespace NUMINAMATH_CALUDE_concave_integral_inequality_l3981_398189

open Set Function MeasureTheory

variable {m : ℕ}
variable (P : Set (EuclideanSpace ℝ (Fin m)))
variable (f : EuclideanSpace ℝ (Fin m) → ℝ)
variable (ξ : EuclideanSpace ℝ (Fin m))

theorem concave_integral_inequality
  (h_nonempty : Set.Nonempty P)
  (h_compact : IsCompact P)
  (h_convex : Convex ℝ P)
  (h_concave : ConcaveOn ℝ P f)
  (h_nonneg : ∀ x ∈ P, 0 ≤ f x) :
  ∫ x in P, ⟪ξ, x⟫_ℝ * f x ≤ 
    ((m + 1 : ℝ) / (m + 2 : ℝ) * ⨆ (x : EuclideanSpace ℝ (Fin m)) (h : x ∈ P), ⟪ξ, x⟫_ℝ + 
     (1 : ℝ) / (m + 2 : ℝ) * ⨅ (x : EuclideanSpace ℝ (Fin m)) (h : x ∈ P), ⟪ξ, x⟫_ℝ) * 
    ∫ x in P, f x :=
sorry

end NUMINAMATH_CALUDE_concave_integral_inequality_l3981_398189


namespace NUMINAMATH_CALUDE_smallest_square_enclosing_circle_l3981_398119

theorem smallest_square_enclosing_circle (r : ℝ) (h : r = 5) :
  (2 * r) ^ 2 = 100 := by sorry

end NUMINAMATH_CALUDE_smallest_square_enclosing_circle_l3981_398119


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l3981_398115

theorem min_value_squared_sum (a b c d : ℝ) (h1 : a * b = 2) (h2 : c * d = 18) :
  (a * c)^2 + (b * d)^2 ≥ 12 ∧ ∃ (a' b' c' d' : ℝ), a' * b' = 2 ∧ c' * d' = 18 ∧ (a' * c')^2 + (b' * d')^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l3981_398115


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l3981_398162

theorem polynomial_product_expansion :
  ∀ x : ℝ, (3*x^2 + 2*x + 1) * (2*x^2 + 3*x + 4) = 6*x^4 + 13*x^3 + 20*x^2 + 11*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l3981_398162


namespace NUMINAMATH_CALUDE_teacher_engineer_ratio_l3981_398163

theorem teacher_engineer_ratio 
  (t : ℕ) -- number of teachers
  (e : ℕ) -- number of engineers
  (h_total : t + e > 0) -- ensure total group size is positive
  (h_avg : (40 * t + 55 * e) / (t + e) = 45) -- overall average age is 45
  : t = 2 * e := by
sorry

end NUMINAMATH_CALUDE_teacher_engineer_ratio_l3981_398163


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3981_398188

theorem polynomial_remainder_theorem (f : ℝ → ℝ) (a b c l m n p q r : ℝ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (∃ Q₁ : ℝ → ℝ, ∀ x, f x = (x - a) * (x - b) * Q₁ x + (p * x + l)) →
  (∃ Q₂ : ℝ → ℝ, ∀ x, f x = (x - b) * (x - c) * Q₂ x + (q * x + m)) →
  (∃ Q₃ : ℝ → ℝ, ∀ x, f x = (x - c) * (x - a) * Q₃ x + (r * x + n)) →
  l * (1 / a - 1 / b) + m * (1 / b - 1 / c) + n * (1 / c - 1 / a) = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3981_398188


namespace NUMINAMATH_CALUDE_initial_clean_and_jerk_was_80kg_l3981_398148

/-- Represents John's weightlifting progress --/
structure Weightlifting where
  initial_snatch : ℝ
  initial_clean_and_jerk : ℝ
  new_combined_total : ℝ

/-- Calculates the new Snatch weight after an 80% increase --/
def new_snatch (w : Weightlifting) : ℝ :=
  w.initial_snatch * 1.8

/-- Calculates the new Clean & Jerk weight after doubling --/
def new_clean_and_jerk (w : Weightlifting) : ℝ :=
  w.initial_clean_and_jerk * 2

/-- Theorem stating that John's initial Clean & Jerk weight was 80 kg --/
theorem initial_clean_and_jerk_was_80kg (w : Weightlifting) 
  (h1 : w.initial_snatch = 50)
  (h2 : new_snatch w + new_clean_and_jerk w = w.new_combined_total)
  (h3 : w.new_combined_total = 250) : 
  w.initial_clean_and_jerk = 80 := by
  sorry


end NUMINAMATH_CALUDE_initial_clean_and_jerk_was_80kg_l3981_398148


namespace NUMINAMATH_CALUDE_median_salary_is_worker_salary_l3981_398130

/-- Represents a position in the company -/
inductive Position
  | CEO
  | GeneralManager
  | Manager
  | Supervisor
  | Worker

/-- Information about a position: number of employees and salary -/
structure PositionInfo where
  count : Nat
  salary : Nat

/-- Company salary data -/
def companySalaries : List (Position × PositionInfo) :=
  [(Position.CEO, ⟨1, 150000⟩),
   (Position.GeneralManager, ⟨3, 100000⟩),
   (Position.Manager, ⟨12, 80000⟩),
   (Position.Supervisor, ⟨8, 55000⟩),
   (Position.Worker, ⟨35, 30000⟩)]

/-- Total number of employees -/
def totalEmployees : Nat :=
  companySalaries.foldr (fun (_, info) acc => acc + info.count) 0

/-- Theorem: The median salary of the company is $30,000 -/
theorem median_salary_is_worker_salary :
  let salaries := companySalaries.map (fun (_, info) => info.salary)
  let counts := companySalaries.map (fun (_, info) => info.count)
  let medianIndex := (totalEmployees + 1) / 2
  ∃ (i : Nat), i < salaries.length ∧
    (counts.take i).sum < medianIndex ∧
    medianIndex ≤ (counts.take (i + 1)).sum ∧
    salaries[i]! = 30000 := by
  sorry

#eval totalEmployees -- Should output 59

end NUMINAMATH_CALUDE_median_salary_is_worker_salary_l3981_398130


namespace NUMINAMATH_CALUDE_equation_solution_set_l3981_398143

theorem equation_solution_set : ∀ (x y : ℝ), 
  ((x = 1 ∧ y = 3/2) ∨ 
   (x = 1 ∧ y = -1/2) ∨ 
   (x = -1 ∧ y = 3/2) ∨ 
   (x = -1 ∧ y = -1/2)) ↔ 
  4 * x^2 * y^2 = 4 * x * y + 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_set_l3981_398143


namespace NUMINAMATH_CALUDE_complex_simplification_l3981_398157

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- The statement to prove -/
theorem complex_simplification : 7 * (2 - 3 * i) + 4 * i * (3 - 2 * i) = 22 - 9 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l3981_398157


namespace NUMINAMATH_CALUDE_cases_in_1990_l3981_398167

/-- Calculates the number of cases in a given year assuming linear decrease --/
def casesInYear (initialCases : ℕ) (finalCases : ℕ) (initialYear : ℕ) (finalYear : ℕ) (targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let yearlyDecrease := totalDecrease / totalYears
  let yearsFromInitial := targetYear - initialYear
  initialCases - (yearlyDecrease * yearsFromInitial)

/-- The number of cases in 1990 given linear decrease from 1970 to 2000 --/
theorem cases_in_1990 : 
  casesInYear 600000 200 1970 2000 1990 = 200133 := by
  sorry

end NUMINAMATH_CALUDE_cases_in_1990_l3981_398167


namespace NUMINAMATH_CALUDE_real_estate_investment_l3981_398184

theorem real_estate_investment
  (total_investment : ℝ)
  (mutual_funds : ℝ)
  (real_estate : ℝ)
  (h1 : total_investment = 200000)
  (h2 : real_estate = 5 * mutual_funds)
  (h3 : total_investment = mutual_funds + real_estate) :
  real_estate = 166666.65 := by
sorry

end NUMINAMATH_CALUDE_real_estate_investment_l3981_398184


namespace NUMINAMATH_CALUDE_prime_sums_count_l3981_398111

/-- Sequence of prime numbers -/
def primes : List Nat := sorry

/-- Function to generate sums by adding primes and skipping every third -/
def generateSums (n : Nat) : List Nat :=
  sorry

/-- Check if a number is prime -/
def isPrime (n : Nat) : Bool :=
  sorry

/-- Count prime sums in the first n generated sums -/
def countPrimeSums (n : Nat) : Nat :=
  sorry

/-- Main theorem: The number of prime sums among the first 12 generated sums is 5 -/
theorem prime_sums_count : countPrimeSums 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_sums_count_l3981_398111


namespace NUMINAMATH_CALUDE_max_strips_from_sheet_l3981_398118

/-- Represents a rectangular sheet of paper --/
structure Sheet where
  length : ℕ
  width : ℕ

/-- Represents a rectangular strip of paper --/
structure Strip where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of strips that can be cut from a sheet --/
def maxStrips (sheet : Sheet) (strip : Strip) : ℕ :=
  max
    ((sheet.length / strip.length) * (sheet.width / strip.width))
    ((sheet.length / strip.width) * (sheet.width / strip.length))

theorem max_strips_from_sheet :
  let sheet := Sheet.mk 14 11
  let strip := Strip.mk 4 1
  maxStrips sheet strip = 33 := by sorry

end NUMINAMATH_CALUDE_max_strips_from_sheet_l3981_398118


namespace NUMINAMATH_CALUDE_initial_shoe_collection_l3981_398101

theorem initial_shoe_collection (initial_collection : ℕ) : 
  (initial_collection : ℝ) * 0.7 + 6 = 62 → initial_collection = 80 :=
by sorry

end NUMINAMATH_CALUDE_initial_shoe_collection_l3981_398101


namespace NUMINAMATH_CALUDE_perimeter_difference_l3981_398109

/-- The perimeter of a rectangle --/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- The perimeter of the cross-shaped figure --/
def cross_perimeter (center_side : ℕ) : ℕ :=
  4 * center_side

/-- The positive difference between two natural numbers --/
def positive_difference (a b : ℕ) : ℕ :=
  max a b - min a b

theorem perimeter_difference :
  positive_difference (rectangle_perimeter 3 2) (cross_perimeter 3) = 2 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_difference_l3981_398109


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l3981_398113

/-- The sampling interval for systematic sampling. -/
def sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The sampling interval for a systematic sampling of 30 students
    from a population of 1200 students is 40. -/
theorem systematic_sampling_interval :
  sampling_interval 1200 30 = 40 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l3981_398113


namespace NUMINAMATH_CALUDE_yadav_expenditure_l3981_398181

theorem yadav_expenditure (monthly_salary : ℝ) : 
  monthly_salary > 0 →
  (0.6 * monthly_salary) + (0.5 * (0.4 * monthly_salary)) + (0.2 * monthly_salary) = monthly_salary →
  (0.2 * monthly_salary) * 12 = 24624 →
  0.5 * (0.4 * monthly_salary) = 2052 := by
sorry

end NUMINAMATH_CALUDE_yadav_expenditure_l3981_398181


namespace NUMINAMATH_CALUDE_one_diagonal_implies_four_sides_l3981_398122

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  sides : ℕ
  sides_pos : sides > 0

/-- A diagonal in a polygon is a line segment that connects two non-adjacent vertices. -/
def has_one_diagonal (p : Polygon) : Prop :=
  ∃ (v : ℕ), v < p.sides ∧ (p.sides - v - 2 = 1)

/-- Theorem: A polygon with exactly one diagonal that can be drawn from one vertex has 4 sides. -/
theorem one_diagonal_implies_four_sides (p : Polygon) (h : has_one_diagonal p) : p.sides = 4 := by
  sorry

end NUMINAMATH_CALUDE_one_diagonal_implies_four_sides_l3981_398122


namespace NUMINAMATH_CALUDE_circle_equation_through_points_on_line_l3981_398196

/-- The standard equation of a circle passing through two points with its center on a given line -/
theorem circle_equation_through_points_on_line (x y : ℝ) : 
  -- The circle passes through (0, 4) and (4, 6)
  ((x - 0)^2 + (y - 4)^2 = (x - 4)^2 + (y - 6)^2) →
  -- The center of the circle is on the line x - 2y - 2 = 0
  (∃ (a b : ℝ), x = a ∧ y = b ∧ a - 2*b - 2 = 0) →
  -- The standard equation of the circle
  (x - 4)^2 + (y - 1)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_through_points_on_line_l3981_398196


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l3981_398144

def total_players : ℕ := 18
def quintuplets : ℕ := 5
def lineup_size : ℕ := 7
def quintuplets_in_lineup : ℕ := 2

theorem basketball_lineup_combinations :
  (Nat.choose quintuplets quintuplets_in_lineup) *
  (Nat.choose (total_players - quintuplets) (lineup_size - quintuplets_in_lineup)) = 12870 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l3981_398144


namespace NUMINAMATH_CALUDE_root_product_equality_l3981_398195

theorem root_product_equality (α β : ℝ) : 
  α^2 + 2017*α + 1 = 0 →
  β^2 + 2017*β + 1 = 0 →
  (1 + 2020*α + α^2) * (1 + 2020*β + β^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equality_l3981_398195


namespace NUMINAMATH_CALUDE_problem_solution_l3981_398131

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → a * b ≤ m) ∧ 
   (∀ m' : ℝ, m' < m → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ a * b > m') ∧
   m = 1/4) ∧
  (∀ x : ℝ, (4/a + 1/b ≥ |2*x - 1| - |x + 2|) ↔ -2 ≤ x ∧ x ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3981_398131


namespace NUMINAMATH_CALUDE_equation_roots_l3981_398141

theorem equation_roots :
  let f : ℝ → ℝ := λ x => (x^3 - 3*x^2 + x - 2)*(x^3 - x^2 - 4*x + 7) + 6*x^2 - 15*x + 18
  ∃ (a b c d e : ℝ), 
    (a = 1 ∧ b = 2 ∧ c = -2 ∧ d = 1 + Real.sqrt 2 ∧ e = 1 - Real.sqrt 2) ∧
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l3981_398141


namespace NUMINAMATH_CALUDE_students_in_cars_l3981_398138

def total_students : ℕ := 375
def num_buses : ℕ := 7
def students_per_bus : ℕ := 53

theorem students_in_cars : 
  total_students - (num_buses * students_per_bus) = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_in_cars_l3981_398138


namespace NUMINAMATH_CALUDE_fraction_sum_equals_percentage_l3981_398177

theorem fraction_sum_equals_percentage (y : ℝ) (h : y > 0) :
  (7 * y) / 20 + (3 * y) / 10 = 0.65 * y := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_percentage_l3981_398177


namespace NUMINAMATH_CALUDE_steel_rusting_not_LeChatelier_l3981_398175

/-- Le Chatelier's principle states that if a change in conditions is imposed on a system at equilibrium, 
    the equilibrium will shift in a direction that tends to reduce that change. -/
def LeChatelier_principle : Prop := sorry

/-- Rusting of steel in humid air -/
def steel_rusting : Prop := sorry

/-- A chemical process that can be explained by Le Chatelier's principle -/
def explainable_by_LeChatelier (process : Prop) : Prop := sorry

theorem steel_rusting_not_LeChatelier : 
  ¬(explainable_by_LeChatelier steel_rusting) := by sorry

end NUMINAMATH_CALUDE_steel_rusting_not_LeChatelier_l3981_398175


namespace NUMINAMATH_CALUDE_sine_squared_equality_l3981_398152

theorem sine_squared_equality (α β : ℝ) 
  (h : (Real.cos α)^4 / (Real.cos β)^2 + (Real.sin α)^4 / (Real.sin β)^2 = 1) :
  (Real.sin α)^2 = (Real.sin β)^2 := by
  sorry

end NUMINAMATH_CALUDE_sine_squared_equality_l3981_398152


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l3981_398149

theorem abs_inequality_solution_set (x : ℝ) : 
  |3*x + 1| > 2 ↔ x > 1/3 ∨ x < -1 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l3981_398149


namespace NUMINAMATH_CALUDE_georgia_green_buttons_l3981_398108

/-- The number of green buttons Georgia has -/
def green_buttons : ℕ := sorry

/-- The number of yellow buttons Georgia has -/
def yellow_buttons : ℕ := 4

/-- The number of black buttons Georgia has -/
def black_buttons : ℕ := 2

/-- The number of buttons Georgia gave away -/
def buttons_given_away : ℕ := 4

/-- The number of buttons Georgia has left -/
def buttons_left : ℕ := 5

theorem georgia_green_buttons :
  yellow_buttons + black_buttons + green_buttons = buttons_given_away + buttons_left :=
sorry

end NUMINAMATH_CALUDE_georgia_green_buttons_l3981_398108


namespace NUMINAMATH_CALUDE_exam_average_l3981_398123

theorem exam_average (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 15 →
  n2 = 10 →
  avg1 = 70 / 100 →
  avg2 = 95 / 100 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 80 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l3981_398123


namespace NUMINAMATH_CALUDE_polynomial_expansion_properties_l3981_398158

theorem polynomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (x - 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ = 16 ∧ a₂ = 24 ∧ a₁ + a₂ + a₃ + a₄ = -15) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_properties_l3981_398158


namespace NUMINAMATH_CALUDE_smarties_leftover_l3981_398132

theorem smarties_leftover (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_smarties_leftover_l3981_398132


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l3981_398153

/-- Given a sequence {aₙ} with sum Sn = (a₁(4ⁿ - 1)) / 3 and a₄ = 32, prove a₁ = 1/2 -/
theorem sequence_sum_problem (a : ℕ → ℚ) (S : ℕ → ℚ) 
  (h1 : ∀ n, S n = (a 1 * (4^n - 1)) / 3)
  (h2 : a 4 = 32) :
  a 1 = 1/2 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_problem_l3981_398153


namespace NUMINAMATH_CALUDE_cookie_ratio_l3981_398124

/-- Proves that the ratio of Glenn's cookies to Kenny's cookies is 4:1 given the problem conditions --/
theorem cookie_ratio (kenny : ℕ) (glenn : ℕ) (chris : ℕ) : 
  chris = kenny / 2 → 
  glenn = 24 → 
  chris + kenny + glenn = 33 → 
  glenn / kenny = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_l3981_398124


namespace NUMINAMATH_CALUDE_hiker_route_length_l3981_398110

theorem hiker_route_length (rate_up : ℝ) (days_up : ℝ) (rate_down_factor : ℝ) : 
  rate_up = 7 →
  days_up = 2 →
  rate_down_factor = 1.5 →
  (rate_up * days_up) * rate_down_factor = 21 := by
  sorry

end NUMINAMATH_CALUDE_hiker_route_length_l3981_398110


namespace NUMINAMATH_CALUDE_max_cube_volume_in_pyramid_l3981_398169

/-- Represents a pyramid with a triangular equilateral base -/
structure Pyramid where
  base_side_length : ℝ
  height : ℝ

/-- Represents a cube -/
structure Cube where
  side_length : ℝ

/-- Theorem: Maximum cube volume in a specific pyramid -/
theorem max_cube_volume_in_pyramid (p : Pyramid) (c : Cube) :
  p.base_side_length = 2 →
  c.side_length = (4 * Real.sqrt 3) / 3 →
  c.side_length ^ 3 = 64 * Real.sqrt 3 / 9 := by
  sorry

#check max_cube_volume_in_pyramid

end NUMINAMATH_CALUDE_max_cube_volume_in_pyramid_l3981_398169


namespace NUMINAMATH_CALUDE_water_rise_in_vessel_l3981_398114

/-- Represents the rise in water level when a cubical box is immersed in a rectangular vessel -/
theorem water_rise_in_vessel 
  (vessel_length : ℝ) 
  (vessel_breadth : ℝ) 
  (box_edge : ℝ) 
  (h : vessel_length = 60 ∧ vessel_breadth = 30 ∧ box_edge = 30) : 
  (box_edge ^ 3) / (vessel_length * vessel_breadth) = 15 := by
  sorry

#check water_rise_in_vessel

end NUMINAMATH_CALUDE_water_rise_in_vessel_l3981_398114


namespace NUMINAMATH_CALUDE_largest_divisor_of_product_l3981_398168

theorem largest_divisor_of_product (n : ℕ) (h : Even n) (h2 : n > 0) :
  ∃ (k : ℕ), k ≤ 15 ∧ 
  (∀ (m : ℕ), m ≤ k → (m ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+11))) ∧
  (∀ (m : ℕ), m > 15 → ∃ (p : ℕ), Even p ∧ p > 0 ∧ ¬(m ∣ (p+1)*(p+3)*(p+5)*(p+7)*(p+11))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_product_l3981_398168


namespace NUMINAMATH_CALUDE_greatest_value_problem_l3981_398183

theorem greatest_value_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hy : 0 < y₁ ∧ y₁ < y₂) 
  (hsum : x₁ + x₂ = 1 ∧ y₁ + y₂ = 1) : 
  max (x₁*y₁ + x₂*y₂) (max (x₁*x₂ + y₁*y₂) (max (x₁*y₂ + x₂*y₁) (1/2))) = x₁*y₁ + x₂*y₂ := by
  sorry

end NUMINAMATH_CALUDE_greatest_value_problem_l3981_398183
