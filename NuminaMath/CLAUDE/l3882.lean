import Mathlib

namespace max_prime_angle_in_isosceles_triangle_l3882_388249

def IsIsosceles (a b c : ℕ) : Prop := a + b + c = 180 ∧ a = b

def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem max_prime_angle_in_isosceles_triangle :
  ∀ x : ℕ,
    IsIsosceles x x (180 - 2*x) →
    IsPrime x →
    x ≤ 7 :=
sorry

end max_prime_angle_in_isosceles_triangle_l3882_388249


namespace polynomial_division_remainder_l3882_388265

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (2 * X^4 + 10 * X^3 - 45 * X^2 - 52 * X + 63) = 
  (X^2 + 6 * X - 7) * q + (48 * X - 70) := by
  sorry

end polynomial_division_remainder_l3882_388265


namespace function_range_l3882_388284

theorem function_range (f : ℝ → ℝ) : 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x = Real.sin x - Real.sqrt 3 * Real.cos x) →
  Set.range f = Set.Icc (-Real.sqrt 3) 1 := by
sorry

end function_range_l3882_388284


namespace increase_then_decrease_l3882_388247

theorem increase_then_decrease (x p q : ℝ) (hx : x = 80) (hp : p = 150) (hq : q = 30) :
  x * (1 + p / 100) * (1 - q / 100) = 140 := by
  sorry

end increase_then_decrease_l3882_388247


namespace households_with_car_l3882_388286

theorem households_with_car (total : Nat) (without_car_or_bike : Nat) (with_both : Nat) (with_bike_only : Nat)
  (h1 : total = 90)
  (h2 : without_car_or_bike = 11)
  (h3 : with_both = 18)
  (h4 : with_bike_only = 35) :
  total - without_car_or_bike - with_bike_only + with_both = 62 := by
sorry

end households_with_car_l3882_388286


namespace range_of_y_over_x_l3882_388221

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom inequality_condition : ∀ x y : ℝ, f (x^2 - 2*x) ≤ -f (2*y - y^2)
axiom symmetry_condition : ∀ x : ℝ, f (x - 1) = f (1 - x)

-- Define the theorem
theorem range_of_y_over_x :
  (∀ x y : ℝ, 1 ≤ x → x ≤ 4 → f x = y → -1/2 ≤ y/x ∧ y/x ≤ 1) :=
sorry

end range_of_y_over_x_l3882_388221


namespace swimming_speed_in_still_water_l3882_388242

/-- Proves that a person's swimming speed in still water is 4 km/h given the conditions -/
theorem swimming_speed_in_still_water 
  (water_speed : ℝ) 
  (swimming_time : ℝ) 
  (swimming_distance : ℝ) 
  (h1 : water_speed = 2)
  (h2 : swimming_time = 5)
  (h3 : swimming_distance = 10)
  : ∃ (still_water_speed : ℝ), 
    swimming_distance = (still_water_speed - water_speed) * swimming_time ∧ 
    still_water_speed = 4 := by
  sorry

end swimming_speed_in_still_water_l3882_388242


namespace range_of_distance_from_origin_l3882_388206

theorem range_of_distance_from_origin : ∀ x y : ℝ,
  x + y = 10 →
  -5 ≤ x - y →
  x - y ≤ 5 →
  5 * Real.sqrt 2 ≤ Real.sqrt (x^2 + y^2) ∧
  Real.sqrt (x^2 + y^2) ≤ (5 * Real.sqrt 10) / 2 :=
by sorry

end range_of_distance_from_origin_l3882_388206


namespace bowling_ball_weight_is_16_l3882_388244

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 16

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 24

/-- Theorem stating the weight of one bowling ball is 16 pounds -/
theorem bowling_ball_weight_is_16 : 
  (9 * bowling_ball_weight = 6 * canoe_weight) → 
  (5 * canoe_weight = 120) → 
  bowling_ball_weight = 16 := by
  sorry

end bowling_ball_weight_is_16_l3882_388244


namespace acid_dilution_l3882_388229

theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 50 →
  initial_concentration = 0.4 →
  final_concentration = 0.25 →
  water_added = 30 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check acid_dilution

end acid_dilution_l3882_388229


namespace alcohol_concentration_after_dilution_l3882_388290

theorem alcohol_concentration_after_dilution
  (original_volume : ℝ)
  (original_concentration : ℝ)
  (added_water : ℝ)
  (h1 : original_volume = 24)
  (h2 : original_concentration = 0.9)
  (h3 : added_water = 16) :
  let alcohol_volume := original_volume * original_concentration
  let new_volume := original_volume + added_water
  let new_concentration := alcohol_volume / new_volume
  new_concentration = 0.54 := by
sorry

end alcohol_concentration_after_dilution_l3882_388290


namespace max_inscribed_circle_radius_l3882_388275

/-- The maximum radius of an inscribed circle centered at (0,0) in the curve |y| = 1 - a x^2 where |x| ≤ 1/√a -/
noncomputable def f (a : ℝ) : ℝ :=
  if a ≤ 1/2 then 1 else Real.sqrt (4*a - 1) / (2*a)

/-- The curve C defined by |y| = 1 - a x^2 where |x| ≤ 1/√a -/
def C (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.2| = 1 - a * p.1^2 ∧ |p.1| ≤ 1/Real.sqrt a}

theorem max_inscribed_circle_radius (a : ℝ) (ha : a > 0) :
  ∀ r : ℝ, r > 0 → (∀ p : ℝ × ℝ, p ∈ C a → (p.1^2 + p.2^2 ≥ r^2)) → r ≤ f a :=
sorry

end max_inscribed_circle_radius_l3882_388275


namespace five_ruble_coins_l3882_388234

/-- Represents the number of coins of each denomination -/
structure CoinCount where
  one : ℕ
  two : ℕ
  five : ℕ
  ten : ℕ

/-- The total number of coins -/
def total_coins : ℕ := 25

/-- The number of coins that are not of each denomination -/
def not_two_coins : ℕ := 19
def not_ten_coins : ℕ := 20
def not_one_coins : ℕ := 16

/-- Theorem stating the number of five-ruble coins -/
theorem five_ruble_coins (c : CoinCount) : c.five = 5 :=
  by
    have h1 : c.one + c.two + c.five + c.ten = total_coins := sorry
    have h2 : c.two = total_coins - not_two_coins := sorry
    have h3 : c.ten = total_coins - not_ten_coins := sorry
    have h4 : c.one = total_coins - not_one_coins := sorry
    sorry

end five_ruble_coins_l3882_388234


namespace hex_to_decimal_conversion_l3882_388220

/-- Given that the hexadecimal number (3m502_(16)) is equal to 4934 in decimal,
    prove that m = 4. -/
theorem hex_to_decimal_conversion (m : ℕ) : 
  (3 * 16^4 + m * 16^3 + 5 * 16^2 + 0 * 16^1 + 2 * 16^0 = 4934) → m = 4 := by
  sorry

end hex_to_decimal_conversion_l3882_388220


namespace cost_per_patch_l3882_388260

/-- Proves that the cost per patch is $1.25 given the order quantity, selling price, and net profit. -/
theorem cost_per_patch (order_quantity : ℕ) (selling_price : ℚ) (net_profit : ℚ) :
  order_quantity = 100 →
  selling_price = 12 →
  net_profit = 1075 →
  (order_quantity : ℚ) * selling_price - (order_quantity : ℚ) * (selling_price - net_profit / (order_quantity : ℚ)) = net_profit :=
by sorry

end cost_per_patch_l3882_388260


namespace range_of_a_l3882_388251

/-- The range of a given the conditions in the problem -/
theorem range_of_a (a : ℝ) : 
  (a < 0) → 
  (∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0) → (x^2 + 2*x - 8 > 0)) →
  (∃ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0) ∧ (x^2 + 2*x - 8 ≤ 0)) →
  a ≤ -4 :=
by sorry

end range_of_a_l3882_388251


namespace inequality_proof_l3882_388278

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 7/4 ≥ a*b + 2*a + b/2 := by
  sorry

end inequality_proof_l3882_388278


namespace prop_a_neither_sufficient_nor_necessary_l3882_388287

-- Define propositions A and B
def PropA (a b : ℝ) : Prop := a + b ≠ 4
def PropB (a b : ℝ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem stating that Prop A is neither sufficient nor necessary for Prop B
theorem prop_a_neither_sufficient_nor_necessary :
  (∃ a b : ℝ, PropA a b ∧ ¬PropB a b) ∧
  (∃ a b : ℝ, PropB a b ∧ ¬PropA a b) :=
sorry

end prop_a_neither_sufficient_nor_necessary_l3882_388287


namespace complex_operations_l3882_388241

theorem complex_operations (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 2 + 3*I) (h₂ : z₂ = 5 - 7*I) : 
  (z₁ + z₂ = 7 - 4*I) ∧ 
  (z₁ - z₂ = -3 + 10*I) ∧ 
  (z₁ * z₂ = 31 + I) := by
  sorry

#check complex_operations

end complex_operations_l3882_388241


namespace average_first_20_even_numbers_l3882_388233

theorem average_first_20_even_numbers : 
  let first_20_even : List ℕ := List.range 20 |>.map (fun i => 2 * (i + 1))
  (first_20_even.sum / first_20_even.length : ℚ) = 21 := by
  sorry

end average_first_20_even_numbers_l3882_388233


namespace total_plums_picked_l3882_388271

theorem total_plums_picked (melanie_plums dan_plums sally_plums : ℕ) 
  (h1 : melanie_plums = 4)
  (h2 : dan_plums = 9)
  (h3 : sally_plums = 3) :
  melanie_plums + dan_plums + sally_plums = 16 := by
  sorry

end total_plums_picked_l3882_388271


namespace quadratic_two_distinct_roots_l3882_388296

/-- 
Given a quadratic equation 2kx^2 + (8k+1)x + 8k = 0 with real coefficient k,
the equation has two distinct real roots if and only if k > -1/16 and k ≠ 0.
-/
theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 2 * k * x₁^2 + (8 * k + 1) * x₁ + 8 * k = 0 ∧
                          2 * k * x₂^2 + (8 * k + 1) * x₂ + 8 * k = 0) ↔
  (k > -1/16 ∧ k ≠ 0) :=
by sorry

end quadratic_two_distinct_roots_l3882_388296


namespace sum_of_three_numbers_l3882_388257

theorem sum_of_three_numbers (a b c : ℤ) 
  (sum_ab : a + b = 35)
  (sum_bc : b + c = 52)
  (sum_ca : c + a = 61) :
  a + b + c = 74 := by
sorry

end sum_of_three_numbers_l3882_388257


namespace integer_roots_conditions_l3882_388235

theorem integer_roots_conditions (p q : ℤ) : 
  (∃ (a b c d : ℤ), (∀ x : ℤ, x^4 + 2*p*x^2 + q*x + p^2 - 36 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) ∧
  (a + b + c + d = 0) ∧
  (a*b + a*c + a*d + b*c + b*d + c*d = 2*p) ∧
  (a*b*c*d = p^2 - 36)) →
  ∃ (x y z : ℕ), 18 = 2*x^2 + y^2 + z^2 ∧ 
  ((x = 0 ∧ y = 3 ∧ z = 3) ∨
   (x = 1 ∧ y = 4 ∧ z = 0) ∨
   (x = 1 ∧ y = 0 ∧ z = 4) ∨
   (x = 2 ∧ y = 3 ∧ z = 1) ∨
   (x = 2 ∧ y = 1 ∧ z = 3) ∨
   (x = 3 ∧ y = 0 ∧ z = 0)) :=
by sorry

end integer_roots_conditions_l3882_388235


namespace expression_simplification_l3882_388205

theorem expression_simplification (a : ℝ) (h : a = -2) :
  (1 - a / (a + 1)) / (1 / (1 - a^2)) = 1 / 3 := by
  sorry

end expression_simplification_l3882_388205


namespace smallest_b_value_l3882_388217

/-- The second smallest positive integer with exactly 3 factors -/
def a : ℕ := 9

/-- A function that returns the number of factors of a positive integer -/
def num_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_b_value :
  ∃ b : ℕ,
    b > 0 ∧
    num_factors b = a ∧
    a ∣ b ∧
    ∀ c : ℕ, c > 0 → num_factors c = a → a ∣ c → b ≤ c ∧
    b = 30 :=
sorry

end smallest_b_value_l3882_388217


namespace min_value_implies_a_l3882_388254

def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

theorem min_value_implies_a (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = 5 ∧ ∀ x : ℝ, f a x ≥ 5) → a = 6 := by
  sorry

end min_value_implies_a_l3882_388254


namespace probability_is_one_third_l3882_388209

/-- Right triangle XYZ with XY = 10 and XZ = 6 -/
structure RightTriangle where
  xy : ℝ
  xz : ℝ
  xy_eq : xy = 10
  xz_eq : xz = 6

/-- Random point Q in the interior of triangle XYZ -/
def RandomPoint (t : RightTriangle) : Type := Unit

/-- Area of triangle QYZ -/
def AreaQYZ (t : RightTriangle) (q : RandomPoint t) : ℝ := sorry

/-- Area of triangle XYZ -/
def AreaXYZ (t : RightTriangle) : ℝ := sorry

/-- Probability that area of QYZ is less than one-third of area of XYZ -/
def Probability (t : RightTriangle) : ℝ := sorry

/-- Theorem: The probability is equal to 1/3 -/
theorem probability_is_one_third (t : RightTriangle) :
  Probability t = 1 / 3 := by sorry

end probability_is_one_third_l3882_388209


namespace equidistant_points_on_axes_l3882_388208

/-- Given points A(1, 5) and B(2, 4), this theorem states that (0, 3) and (-3, 0) are the only points
    on the coordinate axes that are equidistant from A and B. -/
theorem equidistant_points_on_axes (A B P : ℝ × ℝ) : 
  A = (1, 5) → B = (2, 4) → 
  (P.1 = 0 ∨ P.2 = 0) →  -- P is on a coordinate axis
  (dist A P = dist B P) →  -- P is equidistant from A and B
  (P = (0, 3) ∨ P = (-3, 0)) :=
by sorry

#check equidistant_points_on_axes

end equidistant_points_on_axes_l3882_388208


namespace equilateral_triangle_circumcircle_area_l3882_388255

/-- The area of the circumcircle of an equilateral triangle with side length 4√3 is 16π -/
theorem equilateral_triangle_circumcircle_area :
  let side_length : ℝ := 4 * Real.sqrt 3
  let triangle_area : ℝ := (side_length ^ 2 * Real.sqrt 3) / 4
  let circumradius : ℝ := side_length / Real.sqrt 3
  circumradius ^ 2 * Real.pi = 16 * Real.pi := by
  sorry

end equilateral_triangle_circumcircle_area_l3882_388255


namespace prob_five_shots_expected_shots_l3882_388216

-- Define the probability of hitting a target
variable (p : ℝ) (hp : 0 < p) (hp1 : p < 1)

-- Define the number of targets
def num_targets : ℕ := 3

-- Theorem for part (a)
theorem prob_five_shots : 
  (6 : ℝ) * p^3 * (1 - p)^2 = 
  (num_targets.choose 2) * p^3 * (1 - p)^2 := by sorry

-- Theorem for part (b)
theorem expected_shots : 
  (3 : ℝ) / p = num_targets / p := by sorry

end prob_five_shots_expected_shots_l3882_388216


namespace solve_equation_l3882_388256

theorem solve_equation : ∃ x : ℝ, 0.3 * x + 0.1 * 0.5 = 0.29 ∧ x = 0.8 := by
  sorry

end solve_equation_l3882_388256


namespace smallest_multiple_divisible_by_all_up_to_20_l3882_388227

/-- The smallest positive integer divisible by all numbers from 1 to 20 -/
def smallestMultiple : Nat := 232792560

/-- Checks if a number is divisible by all integers from 1 to 20 -/
def divisibleByAllUpTo20 (n : Nat) : Prop :=
  ∀ i : Nat, 1 ≤ i ∧ i ≤ 20 → n % i = 0

theorem smallest_multiple_divisible_by_all_up_to_20 :
  divisibleByAllUpTo20 smallestMultiple ∧
  ∀ n : Nat, n > 0 ∧ n < smallestMultiple → ¬(divisibleByAllUpTo20 n) := by
  sorry

#eval smallestMultiple

end smallest_multiple_divisible_by_all_up_to_20_l3882_388227


namespace zero_not_in_empty_set_l3882_388223

theorem zero_not_in_empty_set : 0 ∉ (∅ : Set ℕ) := by
  sorry

end zero_not_in_empty_set_l3882_388223


namespace trajectory_of_point_B_l3882_388207

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of a parallelogram ABCD -/
def is_parallelogram (A B C D : Point) : Prop := sorry

/-- Definition of a point lying on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Theorem: Trajectory of point B in parallelogram ABCD -/
theorem trajectory_of_point_B 
  (A B C D : Point)
  (h_parallelogram : is_parallelogram A B C D)
  (h_A : A = ⟨3, -1⟩)
  (h_C : C = ⟨2, -3⟩)
  (l : Line)
  (h_l : l = ⟨3, -1, 1⟩)
  (h_D_on_l : point_on_line D l) :
  point_on_line B ⟨3, -1, -20⟩ := by
    sorry

end trajectory_of_point_B_l3882_388207


namespace probability_at_least_one_six_l3882_388211

theorem probability_at_least_one_six (n : ℕ) (p : ℚ) : 
  n = 3 → p = 1/6 → (1 - (1 - p)^n) = 91/216 := by
  sorry

end probability_at_least_one_six_l3882_388211


namespace stream_speed_prove_stream_speed_l3882_388280

/-- The speed of a stream given a swimmer's still water speed and relative upstream/downstream times -/
theorem stream_speed (still_speed : ℝ) (time_ratio : ℝ) : ℝ :=
  let stream_speed := (still_speed * (time_ratio - 1)) / (time_ratio + 1)
  stream_speed

/-- Proves that the speed of the stream is 3 km/h given the conditions -/
theorem prove_stream_speed :
  stream_speed 9 2 = 3 := by
  sorry

end stream_speed_prove_stream_speed_l3882_388280


namespace odd_difference_of_even_and_odd_l3882_388292

theorem odd_difference_of_even_and_odd (a b : ℤ) 
  (ha : Even a) (hb : Odd b) : Odd (a - b) := by
  sorry

end odd_difference_of_even_and_odd_l3882_388292


namespace tangent_slope_points_l3882_388238

theorem tangent_slope_points (x y : ℝ) : 
  y = x^3 ∧ (3 * x^2 = 3) ↔ (x = -1 ∧ y = -1) ∨ (x = 1 ∧ y = 1) :=
sorry

end tangent_slope_points_l3882_388238


namespace count_integers_satisfying_inequality_l3882_388215

theorem count_integers_satisfying_inequality :
  (Finset.filter (fun n : ℤ => (n - 1) * (n + 3) * (n + 7) < 0)
    (Finset.Icc (-10 : ℤ) 12)).card = 6 := by
  sorry

end count_integers_satisfying_inequality_l3882_388215


namespace largest_number_l3882_388274

theorem largest_number (a b c d e : ℝ) : 
  a = 15467 + 3 / 5791 → 
  b = 15467 - 3 / 5791 → 
  c = 15467 * 3 / 5791 → 
  d = 15467 / (3 / 5791) → 
  e = 15467.5791 → 
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end largest_number_l3882_388274


namespace rectangle_area_l3882_388263

theorem rectangle_area (square_area : ℝ) (rectangle_width rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end rectangle_area_l3882_388263


namespace combined_mean_of_sets_l3882_388225

theorem combined_mean_of_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) 
  (new_set1_count : ℕ) (new_set1_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 21 →
  new_set1_count = set1_count + 1 →
  new_set1_mean = 16 →
  let total_count := new_set1_count + set2_count
  let total_sum := new_set1_mean * new_set1_count + set2_mean * set2_count
  (total_sum / total_count : ℚ) = 37/2 := by
sorry

end combined_mean_of_sets_l3882_388225


namespace quadratic_form_decomposition_l3882_388226

theorem quadratic_form_decomposition (x y z : ℝ) : 
  x^2 + 2*x*y + 5*y^2 - 6*x*z - 22*y*z + 16*z^2 = 
  (x + (y - 3*z))^2 + (2*y - 4*z)^2 - (3*z)^2 := by
  sorry

end quadratic_form_decomposition_l3882_388226


namespace rebecca_camping_items_l3882_388268

/-- The number of items Rebecca bought for her camping trip -/
def total_items (tent_stakes drink_mix water : ℕ) : ℕ :=
  tent_stakes + drink_mix + water

/-- Theorem stating the total number of items Rebecca bought -/
theorem rebecca_camping_items : ∃ (tent_stakes drink_mix water : ℕ),
  tent_stakes = 4 ∧
  drink_mix = 3 * tent_stakes ∧
  water = tent_stakes + 2 ∧
  total_items tent_stakes drink_mix water = 22 := by
  sorry

end rebecca_camping_items_l3882_388268


namespace extra_fruits_calculation_l3882_388231

theorem extra_fruits_calculation (red_ordered green_ordered oranges_ordered : ℕ)
                                 (red_chosen green_chosen oranges_chosen : ℕ)
                                 (h1 : red_ordered = 43)
                                 (h2 : green_ordered = 32)
                                 (h3 : oranges_ordered = 25)
                                 (h4 : red_chosen = 7)
                                 (h5 : green_chosen = 5)
                                 (h6 : oranges_chosen = 4) :
  (red_ordered - red_chosen) + (green_ordered - green_chosen) + (oranges_ordered - oranges_chosen) = 84 :=
by sorry

end extra_fruits_calculation_l3882_388231


namespace arithmetic_sequence_terms_l3882_388250

/-- An arithmetic sequence with given parameters has 13 terms -/
theorem arithmetic_sequence_terms (a d l : ℤ) (h1 : a = -5) (h2 : d = 5) (h3 : l = 55) :
  ∃ n : ℕ, n = 13 ∧ l = a + (n - 1) * d :=
sorry

end arithmetic_sequence_terms_l3882_388250


namespace parallelogram_height_l3882_388201

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) 
  (h_area : area = 216) 
  (h_base : base = 12) 
  (h_formula : area = base * height) : 
  height = 18 := by
  sorry

end parallelogram_height_l3882_388201


namespace cubic_roots_theorem_l3882_388239

theorem cubic_roots_theorem (a b c : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x - c = 0 ↔ x = a ∨ x = b ∨ x = c) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a = 1 ∧ b = -2 ∧ c = 0 := by
sorry

end cubic_roots_theorem_l3882_388239


namespace quadratic_equation_result_l3882_388272

theorem quadratic_equation_result (y : ℝ) (h : 7 * y^2 + 6 = 5 * y + 14) : 
  (14 * y - 2)^2 = 258 := by
  sorry

end quadratic_equation_result_l3882_388272


namespace circle_equation_l3882_388288

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 2}

-- Define the line x-y-1=0
def line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - 1 = 0}

-- Define points A and B
def point_A : ℝ × ℝ := (4, 1)
def point_B : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem circle_equation :
  (point_A ∈ circle_C) ∧
  (point_B ∈ line) ∧
  (∃ (t : ℝ), ∀ (p : ℝ × ℝ), p ∈ circle_C → (p.1 - point_B.1) * 1 + (p.2 - point_B.2) * (-1) = t * ((p.1 - point_B.1)^2 + (p.2 - point_B.2)^2)) →
  ∀ (x y : ℝ), (x, y) ∈ circle_C ↔ (x - 3)^2 + y^2 = 2 :=
by sorry

end circle_equation_l3882_388288


namespace sin_plus_cos_alpha_l3882_388214

theorem sin_plus_cos_alpha (α : ℝ) 
  (h1 : Real.cos (α + π/4) = 7 * Real.sqrt 2 / 10)
  (h2 : Real.cos (2 * α) = 7/25) : 
  Real.sin α + Real.cos α = 1/5 := by
  sorry

end sin_plus_cos_alpha_l3882_388214


namespace largest_negative_integer_and_abs_property_l3882_388294

theorem largest_negative_integer_and_abs_property :
  (∀ n : ℤ, n < 0 → n ≤ -1) ∧
  (∀ x : ℝ, |x| = x → x ≥ 0) :=
by sorry

end largest_negative_integer_and_abs_property_l3882_388294


namespace fred_remaining_cards_l3882_388246

def initial_cards : ℕ := 40
def purchase_percentage : ℚ := 375 / 1000

theorem fred_remaining_cards :
  initial_cards - (purchase_percentage * initial_cards).floor = 25 := by
  sorry

end fred_remaining_cards_l3882_388246


namespace midpoint_locus_l3882_388270

-- Define the line l
def line_l (x y : ℝ) : Prop := x / 12 + y / 8 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 24 + y^2 / 16 = 1

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define tangent points A and B on ellipse C
def tangent_point (x y : ℝ) : Prop := ellipse_C x y

-- Define the midpoint M of AB
def midpoint_M (x y x1 y1 x2 y2 : ℝ) : Prop :=
  x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2

-- Theorem statement
theorem midpoint_locus
  (P_x P_y A_x A_y B_x B_y M_x M_y : ℝ)
  (h_P : point_P P_x P_y)
  (h_A : tangent_point A_x A_y)
  (h_B : tangent_point B_x B_y)
  (h_M : midpoint_M M_x M_y A_x A_y B_x B_y) :
  (M_x - 1)^2 / (5/2) + (M_y - 1)^2 / (5/3) = 1 :=
sorry

end midpoint_locus_l3882_388270


namespace second_year_increase_is_25_percent_l3882_388267

/-- Calculates the percentage increase in the second year given the initial population,
    first year increase percentage, and final population after two years. -/
def second_year_increase (initial_population : ℕ) (first_year_increase : ℚ) (final_population : ℕ) : ℚ :=
  let population_after_first_year := initial_population * (1 + first_year_increase)
  let second_year_factor := final_population / population_after_first_year
  (second_year_factor - 1) * 100

theorem second_year_increase_is_25_percent :
  second_year_increase 800 (22/100) 1220 = 25 := by
  sorry

#eval second_year_increase 800 (22/100) 1220

end second_year_increase_is_25_percent_l3882_388267


namespace profit_percentage_is_10_percent_l3882_388259

def cost_price : ℚ := 340
def selling_price : ℚ := 374

theorem profit_percentage_is_10_percent :
  (selling_price - cost_price) / cost_price * 100 = 10 := by
  sorry

end profit_percentage_is_10_percent_l3882_388259


namespace function_is_identity_l3882_388245

def is_valid_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) - f x - f y ∈ ({0, 1} : Set ℝ)) ∧
  (∀ x : ℝ, ⌊f x⌋ = ⌊x⌋)

theorem function_is_identity (f : ℝ → ℝ) (h : is_valid_function f) :
  ∀ x : ℝ, f x = x := by
  sorry

end function_is_identity_l3882_388245


namespace division_remainder_l3882_388224

-- Define the dividend polynomial
def dividend (x : ℝ) : ℝ := 3*x^5 + 2*x^4 - 5*x^3 + 6*x - 8

-- Define the divisor polynomial
def divisor (x : ℝ) : ℝ := x^2 + 3*x + 2

-- Define the remainder polynomial
def remainder (x : ℝ) : ℝ := 34*x + 24

-- Theorem statement
theorem division_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, dividend x = divisor x * q x + remainder x :=
sorry

end division_remainder_l3882_388224


namespace percent_relation_l3882_388230

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.2 * x := by
sorry

end percent_relation_l3882_388230


namespace expression_evaluation_l3882_388269

theorem expression_evaluation (a b : ℝ) (h1 : a = -1) (h2 : b = -4) :
  ((a - 2*b)^2 + (a - 2*b)*(a + 2*b) + 2*a*(2*a - b)) / (2*a) = 9 := by
sorry

end expression_evaluation_l3882_388269


namespace combined_capacity_after_transfer_l3882_388273

/-- Represents the capacity and fill level of a drum --/
structure Drum where
  capacity : ℝ
  fillLevel : ℝ

/-- Theorem stating the combined capacity of three drums --/
theorem combined_capacity_after_transfer
  (drumX : Drum)
  (drumY : Drum)
  (drumZ : Drum)
  (hX : drumX.capacity = A ∧ drumX.fillLevel = 1/2)
  (hY : drumY.capacity = 2*A ∧ drumY.fillLevel = 1/5)
  (hZ : drumZ.capacity = B ∧ drumZ.fillLevel = 1/4)
  : drumX.capacity + drumY.capacity + drumZ.capacity = 3*A + B :=
by
  sorry

#check combined_capacity_after_transfer

end combined_capacity_after_transfer_l3882_388273


namespace perfect_square_identity_l3882_388277

theorem perfect_square_identity (x y : ℝ) : x^2 + 2*x*y + y^2 = (x + y)^2 := by
  sorry

end perfect_square_identity_l3882_388277


namespace triangle_perimeter_impossibility_l3882_388228

theorem triangle_perimeter_impossibility (a b x : ℝ) (h1 : a = 13) (h2 : b = 24) :
  (a + b + x = 78) → ¬(a + b > x ∧ a + x > b ∧ b + x > a) :=
by sorry

end triangle_perimeter_impossibility_l3882_388228


namespace ratio_is_two_l3882_388240

/-- Three integers a, b, and c where a < b < c and a = 0 -/
def IntegerTriple := {abc : ℤ × ℤ × ℤ // abc.1 < abc.2.1 ∧ abc.2.1 < abc.2.2 ∧ abc.1 = 0}

/-- Three integers p, q, r where p < q < r and r ≠ 0 -/
def GeometricTriple := {pqr : ℤ × ℤ × ℤ // pqr.1 < pqr.2.1 ∧ pqr.2.1 < pqr.2.2 ∧ pqr.2.2 ≠ 0}

/-- The mean of three integers is half the median -/
def MeanHalfMedian (abc : IntegerTriple) : Prop :=
  (abc.val.1 + abc.val.2.1 + abc.val.2.2) / 3 = abc.val.2.1 / 2

/-- The product of three integers is 0 -/
def ProductZero (abc : IntegerTriple) : Prop :=
  abc.val.1 * abc.val.2.1 * abc.val.2.2 = 0

/-- Three integers are in geometric progression -/
def GeometricProgression (pqr : GeometricTriple) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ r ≠ 1 ∧ pqr.val.2.1 = pqr.val.1 * r ∧ pqr.val.2.2 = pqr.val.2.1 * r

/-- Sum of squares equals square of sum -/
def SumSquaresEqualSquareSum (abc : IntegerTriple) (pqr : GeometricTriple) : Prop :=
  abc.val.1^2 + abc.val.2.1^2 + abc.val.2.2^2 = (pqr.val.1 + pqr.val.2.1 + pqr.val.2.2)^2

theorem ratio_is_two (abc : IntegerTriple) (pqr : GeometricTriple)
  (h1 : MeanHalfMedian abc)
  (h2 : ProductZero abc)
  (h3 : GeometricProgression pqr)
  (h4 : SumSquaresEqualSquareSum abc pqr) :
  abc.val.2.2 / abc.val.2.1 = 2 := by sorry

end ratio_is_two_l3882_388240


namespace percentage_problem_l3882_388285

theorem percentage_problem (X : ℝ) : 
  (28 / 100) * 400 + (45 / 100) * X = 224.5 → X = 250 := by
  sorry

end percentage_problem_l3882_388285


namespace probability_at_least_one_man_l3882_388293

theorem probability_at_least_one_man (total : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total = men + women →
  men = 10 →
  women = 5 →
  selected = 5 →
  (1 : ℚ) - (Nat.choose women selected : ℚ) / (Nat.choose total selected : ℚ) = 3002 / 3003 :=
by sorry

end probability_at_least_one_man_l3882_388293


namespace leap_year_53_sundays_5_feb_sundays_probability_l3882_388276

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a leap year -/
structure LeapYear where
  days : Fin 366
  sundays : Nat
  februarySundays : Nat

/-- The probability of a specific configuration of extra days in a leap year -/
def extraDaysProbability : ℚ := 1 / 7

/-- The probability of a leap year having 53 Sundays -/
def prob53Sundays : ℚ := 2 / 7

/-- The probability of February in a leap year having 5 Sundays -/
def probFeb5Sundays : ℚ := 1 / 7

/-- 
Theorem: The probability of a randomly selected leap year having 53 Sundays, 
with exactly 5 of those Sundays falling in February, is 2/49.
-/
theorem leap_year_53_sundays_5_feb_sundays_probability : 
  prob53Sundays * probFeb5Sundays = 2 / 49 := by
  sorry

end leap_year_53_sundays_5_feb_sundays_probability_l3882_388276


namespace smallest_x_value_l3882_388252

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (178 + x)) : 
  ∃ (x_min : ℕ+), x_min ≤ x ∧ ∃ (y_min : ℕ+), (3 : ℚ) / 4 = y_min / (178 + x_min) ∧ x_min = 2 :=
sorry

end smallest_x_value_l3882_388252


namespace original_room_width_l3882_388204

/-- Proves that the original width of the room is 18 feet given the problem conditions -/
theorem original_room_width (length : ℝ) (increased_size : ℝ) (total_area : ℝ) : 
  length = 13 →
  increased_size = 2 →
  total_area = 1800 →
  ∃ w : ℝ, 
    (4 * ((length + increased_size) * (w + increased_size)) + 
     2 * ((length + increased_size) * (w + increased_size))) = total_area ∧
    w = 18 := by
  sorry

end original_room_width_l3882_388204


namespace translation_of_A_l3882_388222

-- Define the points A, B, and C
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (5, 2)
def C : ℝ × ℝ := (3, -1)

-- Define the translation function
def translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + (C.1 - B.1), p.2 + (C.2 - B.2))

-- Theorem statement
theorem translation_of_A :
  translate A = (0, 1) := by sorry

end translation_of_A_l3882_388222


namespace car_travel_distance_l3882_388266

/-- Proves that Car X travels 294 miles from when Car Y starts until both cars stop -/
theorem car_travel_distance (speed_x speed_y : ℝ) (head_start : ℝ) : 
  speed_x = 35 →
  speed_y = 40 →
  head_start = 1.2 →
  (speed_y * (head_start + (294 / speed_x))) = (speed_x * (294 / speed_x) + speed_x * head_start) →
  294 = speed_x * (294 / speed_x) :=
by
  sorry

#check car_travel_distance

end car_travel_distance_l3882_388266


namespace mary_nickels_problem_l3882_388236

theorem mary_nickels_problem (initial : ℕ) (given : ℕ) (total : ℕ) : 
  given = 5 → total = 12 → initial + given = total → initial = 7 := by
  sorry

end mary_nickels_problem_l3882_388236


namespace train_length_problem_l3882_388219

theorem train_length_problem (faster_speed slower_speed : ℝ) 
  (passing_time : ℝ) (h1 : faster_speed = 47) (h2 : slower_speed = 36) 
  (h3 : passing_time = 36) : ∃ (train_length : ℝ), train_length = 55 := by
  sorry

end train_length_problem_l3882_388219


namespace unbiased_scale_impossible_biased_scale_possible_l3882_388298

/-- Represents the result of a weighing -/
inductive WeighResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing strategy -/
def WeighStrategy := List WeighResult → WeighResult

/-- Represents a set of weights -/
def Weights := List Nat

/-- Represents a balance scale -/
structure Balance where
  bias : Int  -- Positive means left pan is lighter

/-- Function to perform a weighing -/
def weigh (b : Balance) (left right : Weights) : WeighResult :=
  sorry

/-- Function to determine if a set of weights can be uniquely identified -/
def canIdentifyWeights (w : Weights) (b : Balance) (n : Nat) : Prop :=
  sorry

/-- The main theorem for the unbiased scale -/
theorem unbiased_scale_impossible 
  (w : Weights) 
  (h1 : w = [1000, 1002, 1004, 1005]) 
  (b : Balance) 
  (h2 : b.bias = 0) : 
  ¬ (canIdentifyWeights w b 4) :=
sorry

/-- The main theorem for the biased scale -/
theorem biased_scale_possible 
  (w : Weights) 
  (h1 : w = [1000, 1002, 1004, 1005]) 
  (b : Balance) 
  (h2 : b.bias = 1) : 
  canIdentifyWeights w b 4 :=
sorry

end unbiased_scale_impossible_biased_scale_possible_l3882_388298


namespace white_balls_added_l3882_388258

theorem white_balls_added (m : ℕ) : 
  (10 + m : ℚ) / (16 + m) = 4/5 → m = 14 := by
  sorry

end white_balls_added_l3882_388258


namespace parabola_directrix_through_ellipse_focus_l3882_388202

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/5 = 1

-- Define the focus of the ellipse
def ellipse_focus : ℝ × ℝ := (2, 0)

-- Define the directrix of the parabola
def parabola_directrix (p : ℝ) : ℝ → Prop := λ x ↦ x = -p/2

-- Theorem statement
theorem parabola_directrix_through_ellipse_focus :
  ∀ p : ℝ, (∃ x y : ℝ, parabola p x y ∧ ellipse x y ∧ 
    parabola_directrix p (ellipse_focus.1)) →
  parabola_directrix p = λ x ↦ x = -2 := by sorry

end parabola_directrix_through_ellipse_focus_l3882_388202


namespace min_people_like_both_tea_and_coffee_l3882_388200

theorem min_people_like_both_tea_and_coffee
  (total : ℕ)
  (tea_lovers : ℕ)
  (coffee_lovers : ℕ)
  (h1 : total = 150)
  (h2 : tea_lovers = 120)
  (h3 : coffee_lovers = 100) :
  (tea_lovers + coffee_lovers - total : ℤ) ≥ 70 :=
sorry

end min_people_like_both_tea_and_coffee_l3882_388200


namespace max_value_trigonometric_function_l3882_388237

open Real

theorem max_value_trigonometric_function :
  ∃ (max : ℝ), max = 6 - 4 * Real.sqrt 2 ∧
  ∀ θ : ℝ, θ ∈ Set.Ioo 0 (π / 2) →
    (2 * sin θ * cos θ) / ((sin θ + 1) * (cos θ + 1)) ≤ max :=
by sorry

end max_value_trigonometric_function_l3882_388237


namespace expression_value_l3882_388248

theorem expression_value (a b : ℝ) 
  (h1 : |a| ≠ |b|) 
  (h2 : (a + b) / (a - b) + (a - b) / (a + b) = 6) : 
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 18/7 := by
  sorry

end expression_value_l3882_388248


namespace two_distinct_prime_factors_l3882_388210

def append_threes (n : ℕ) : ℕ :=
  12320 * 4^(10*n + 1) + (4^(10*n + 1) - 1) / 3

theorem two_distinct_prime_factors (n : ℕ) : 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ 
   append_threes n = p * q) ↔ n = 0 :=
sorry

end two_distinct_prime_factors_l3882_388210


namespace sophomore_sample_count_l3882_388264

/-- Represents a stratified sampling scenario in a high school. -/
structure HighSchoolSampling where
  total_students : ℕ
  sophomore_count : ℕ
  sample_size : ℕ

/-- Calculates the number of sophomores in a stratified sample. -/
def sophomores_in_sample (h : HighSchoolSampling) : ℕ :=
  (h.sophomore_count * h.sample_size) / h.total_students

/-- Theorem: The number of sophomores in the sample is 93 given the specific scenario. -/
theorem sophomore_sample_count (h : HighSchoolSampling) 
  (h_total : h.total_students = 2800)
  (h_sophomores : h.sophomore_count = 930)
  (h_sample : h.sample_size = 280) : 
  sophomores_in_sample h = 93 := by
sorry

end sophomore_sample_count_l3882_388264


namespace lines_concurrent_l3882_388232

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the incidence relation
variable (lies_on : Point → Line → Prop)

-- Define the intersection of two lines
variable (intersect : Line → Line → Point)

-- Define the line passing through two points
variable (line_through : Point → Point → Line)

variable (A B C D E F P X Y Z W Q : Point)

-- Define the quadrilateral ABCD
variable (is_quadrilateral : Prop)

-- Define the conditions for E, F, P, X, Y, Z, W
variable (E_def : E = intersect (line_through A B) (line_through C D))
variable (F_def : F = intersect (line_through B C) (line_through D A))
variable (not_on_EF : ¬ lies_on P (line_through E F))
variable (X_def : X = intersect (line_through P A) (line_through E F))
variable (Y_def : Y = intersect (line_through P B) (line_through E F))
variable (Z_def : Z = intersect (line_through P C) (line_through E F))
variable (W_def : W = intersect (line_through P D) (line_through E F))

-- The theorem to prove
theorem lines_concurrent :
  ∃ Q : Point,
    lies_on Q (line_through A Z) ∧
    lies_on Q (line_through B W) ∧
    lies_on Q (line_through C X) ∧
    lies_on Q (line_through D Y) :=
sorry

end lines_concurrent_l3882_388232


namespace hex_1F4B_equals_8011_l3882_388279

-- Define the hexadecimal digits and their decimal equivalents
def hex_to_dec (c : Char) : Nat :=
  match c with
  | '0' => 0 | '1' => 1 | '2' => 2 | '3' => 3 | '4' => 4 | '5' => 5 | '6' => 6 | '7' => 7
  | '8' => 8 | '9' => 9 | 'A' => 10 | 'B' => 11 | 'C' => 12 | 'D' => 13 | 'E' => 14 | 'F' => 15
  | _ => 0

-- Define the conversion function from hexadecimal to decimal
def hex_to_decimal (s : String) : Nat :=
  s.foldl (fun acc c => 16 * acc + hex_to_dec c) 0

-- Theorem statement
theorem hex_1F4B_equals_8011 :
  hex_to_decimal "1F4B" = 8011 := by
  sorry

end hex_1F4B_equals_8011_l3882_388279


namespace fenced_area_calculation_l3882_388282

/-- The area of a rectangular yard with a square cut out -/
def fenced_area (length width cut_size : ℝ) : ℝ :=
  length * width - cut_size * cut_size

/-- Theorem: The area of a 20-foot by 18-foot rectangular region with a 4-foot by 4-foot square cut out is 344 square feet -/
theorem fenced_area_calculation :
  fenced_area 20 18 4 = 344 := by
  sorry

end fenced_area_calculation_l3882_388282


namespace x_minus_y_power_2007_l3882_388203

theorem x_minus_y_power_2007 (x y : ℝ) :
  5 * x^2 - 4 * x * y + y^2 - 2 * x + 1 = 0 →
  (x - y)^2007 = -1 := by sorry

end x_minus_y_power_2007_l3882_388203


namespace lines_can_coincide_by_rotation_l3882_388262

/-- Given two lines l₁ and l₂ in the xy-plane, prove that they can coincide
    by rotating l₂ around a point on l₁. -/
theorem lines_can_coincide_by_rotation (α c : ℝ) :
  ∃ (x₀ y₀ θ : ℝ), 
    (y₀ = x₀ * Real.sin α) ∧  -- Point (x₀, y₀) is on l₁
    (∀ x y : ℝ,
      y = 2*x + c →  -- Original equation of l₂
      ∃ x' y' : ℝ,
        x' = (x - x₀) * Real.cos θ - (y - y₀) * Real.sin θ + x₀ ∧
        y' = (x - x₀) * Real.sin θ + (y - y₀) * Real.cos θ + y₀ ∧
        y' = x' * Real.sin α) -- Rotated l₂ coincides with l₁
  := by sorry

end lines_can_coincide_by_rotation_l3882_388262


namespace larger_integer_value_l3882_388281

theorem larger_integer_value (a b : ℕ+) (h1 : (a : ℚ) / (b : ℚ) = 3 / 2) (h2 : (a : ℕ) * b = 108) : 
  a = ⌊9 * Real.sqrt 2⌋ := by
sorry

end larger_integer_value_l3882_388281


namespace correct_mean_calculation_l3882_388213

theorem correct_mean_calculation (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 20 ∧ original_mean = 150 ∧ incorrect_value = 135 ∧ correct_value = 160 →
  (n : ℚ) * original_mean - incorrect_value + correct_value = n * (151.25 : ℚ) := by
  sorry

end correct_mean_calculation_l3882_388213


namespace fathers_age_is_32_l3882_388295

/-- The son's current age -/
def sons_age : ℕ := 16

/-- The father's current age -/
def fathers_age : ℕ := 32

/-- Theorem stating that the father's age is 32 -/
theorem fathers_age_is_32 :
  (fathers_age - sons_age = sons_age) ∧ 
  (sons_age = 11 + 5) →
  fathers_age = 32 := by sorry

end fathers_age_is_32_l3882_388295


namespace cos_double_angle_from_series_sum_l3882_388212

theorem cos_double_angle_from_series_sum (θ : ℝ) 
  (h : ∑' n, (Real.cos θ) ^ (2 * n) = 9) : 
  Real.cos (2 * θ) = 7 / 9 := by
  sorry

end cos_double_angle_from_series_sum_l3882_388212


namespace share_A_is_240_l3882_388299

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_share_A (initial_A initial_B : ℕ) (withdraw_A advance_B : ℕ) (months : ℕ) (total_profit : ℕ) : ℕ :=
  let investment_months_A := initial_A * months + (initial_A - withdraw_A) * (12 - months)
  let investment_months_B := initial_B * months + (initial_B + advance_B) * (12 - months)
  let total_investment_months := investment_months_A + investment_months_B
  (investment_months_A * total_profit) / total_investment_months

theorem share_A_is_240 :
  calculate_share_A 3000 4000 1000 1000 8 630 = 240 := by
  sorry

#eval calculate_share_A 3000 4000 1000 1000 8 630

end share_A_is_240_l3882_388299


namespace even_sum_of_even_sum_of_squares_l3882_388289

theorem even_sum_of_even_sum_of_squares (n m : ℤ) (h : Even (n^2 + m^2)) : Even (n + m) := by
  sorry

end even_sum_of_even_sum_of_squares_l3882_388289


namespace unknown_number_in_set_l3882_388261

theorem unknown_number_in_set (x : ℝ) : 
  ((14 + 32 + 53) / 3 = (21 + x + 22) / 3 + 3) → x = 47 := by
  sorry

end unknown_number_in_set_l3882_388261


namespace chocolate_milk_probability_l3882_388243

theorem chocolate_milk_probability : 
  let n : ℕ := 7  -- number of days
  let k : ℕ := 4  -- number of successes (chocolate milk days)
  let p : ℚ := 2/3  -- probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 :=
by sorry

end chocolate_milk_probability_l3882_388243


namespace abc_product_l3882_388297

theorem abc_product (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 30)
  (h_eq : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 672 / (a * b * c) = 1) :
  a * b * c = 2808 := by
  sorry

end abc_product_l3882_388297


namespace pure_imaginary_solutions_of_polynomial_l3882_388283

theorem pure_imaginary_solutions_of_polynomial (x : ℂ) :
  (x^4 - 4*x^3 + 6*x^2 - 40*x - 48 = 0) ∧ (∃ k : ℝ, x = k * Complex.I) ↔
  x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10 := by
  sorry

end pure_imaginary_solutions_of_polynomial_l3882_388283


namespace no_valid_rope_net_with_2001_knots_l3882_388218

/-- A rope net is a structure where knots are connected by ropes. -/
structure RopeNet where
  knots : ℕ
  ropes_per_knot : ℕ

/-- A valid rope net has a positive number of knots and exactly 3 ropes per knot. -/
def is_valid_rope_net (net : RopeNet) : Prop :=
  net.knots > 0 ∧ net.ropes_per_knot = 3

/-- The total number of rope ends in a rope net. -/
def total_rope_ends (net : RopeNet) : ℕ :=
  net.knots * net.ropes_per_knot

/-- The number of distinct ropes in a rope net. -/
def distinct_ropes (net : RopeNet) : ℚ :=
  (total_rope_ends net : ℚ) / 2

/-- Theorem: It is impossible for a valid rope net to have exactly 2001 knots. -/
theorem no_valid_rope_net_with_2001_knots :
  ¬ ∃ (net : RopeNet), is_valid_rope_net net ∧ net.knots = 2001 :=
sorry

end no_valid_rope_net_with_2001_knots_l3882_388218


namespace solve_equation_l3882_388253

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 4 / 3 → x = -27 / 11 := by
  sorry

end solve_equation_l3882_388253


namespace smallest_four_digit_multiplication_result_l3882_388291

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Digit
  hundreds : Digit
  tens : Digit
  ones : Digit

/-- Converts a FourDigitNumber to a natural number -/
def FourDigitNumber.toNat (n : FourDigitNumber) : Nat :=
  1000 * (n.thousands.val + 1) + 100 * (n.hundreds.val + 1) + 10 * (n.tens.val + 1) + (n.ones.val + 1)

/-- The theorem statement -/
theorem smallest_four_digit_multiplication_result :
  ∀ (a b c d e f g h : Digit),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h →
    ((a.val + 1) * 10 + (b.val + 1)) * ((c.val + 1) * 10 + (d.val + 1)) =
      (e.val + 1) * 1000 + (f.val + 1) * 100 + (g.val + 1) * 10 + (h.val + 1) →
    ∀ (n : FourDigitNumber),
      n.toNat ≥ 4396 :=
by sorry

#check smallest_four_digit_multiplication_result

end smallest_four_digit_multiplication_result_l3882_388291
