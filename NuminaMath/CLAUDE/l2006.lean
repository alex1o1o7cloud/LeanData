import Mathlib

namespace dark_integer_characterization_l2006_200694

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer is of the form a999...999 -/
def isA999Form (n : ℕ+) : Prop := sorry

/-- A positive integer is shiny if it can be written as the sum of two integers
    with the same sum of digits -/
def isShiny (n : ℕ+) : Prop :=
  ∃ a b : ℕ, n = a + b ∧ sumOfDigits ⟨a, sorry⟩ = sumOfDigits ⟨b, sorry⟩

theorem dark_integer_characterization (n : ℕ+) :
  ¬isShiny n ↔ isA999Form n ∧ Odd (sumOfDigits n) := by sorry

end dark_integer_characterization_l2006_200694


namespace square_divisibility_l2006_200630

theorem square_divisibility (n : ℕ) (h1 : n > 0) (h2 : ∀ d : ℕ, d ∣ n → d ≤ 6) :
  36 ∣ n^2 := by
sorry

end square_divisibility_l2006_200630


namespace polar_to_cartesian_l2006_200624

/-- The polar equation of the curve -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ ^ 2 - Real.sin θ ^ 2) = 0

/-- The Cartesian equation of two intersecting straight lines -/
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 = y^2

/-- Theorem stating that the polar equation represents two intersecting straight lines -/
theorem polar_to_cartesian :
  ∀ x y : ℝ, ∃ ρ θ : ℝ, polar_equation ρ θ ↔ cartesian_equation x y :=
sorry

end polar_to_cartesian_l2006_200624


namespace base7_equals_base10_l2006_200618

/-- Converts a number from base 7 to base 10 -/
def base7To10 (n : ℕ) : ℕ := sorry

/-- Represents a base-10 digit (0-9) -/
def Digit := {d : ℕ // d < 10}

theorem base7_equals_base10 (c d : Digit) :
  base7To10 764 = 400 + 10 * c.val + d.val →
  (c.val * d.val) / 20 = 6 / 5 := by sorry

end base7_equals_base10_l2006_200618


namespace number_of_divisors_36_l2006_200615

/-- The number of positive divisors of 36 is 9. -/
theorem number_of_divisors_36 : (Finset.filter (· ∣ 36) (Finset.range 37)).card = 9 := by
  sorry

end number_of_divisors_36_l2006_200615


namespace probability_select_one_coastal_l2006_200628

/-- Represents a city that can be either coastal or inland -/
inductive City
| coastal : City
| inland : City

/-- The set of all cities -/
def allCities : Finset City := sorry

/-- The set of coastal cities -/
def coastalCities : Finset City := sorry

theorem probability_select_one_coastal :
  (2 : ℕ) = Finset.card coastalCities →
  (4 : ℕ) = Finset.card allCities →
  (1 : ℚ) / 2 = Finset.card coastalCities / Finset.card allCities := by
  sorry

end probability_select_one_coastal_l2006_200628


namespace sum_of_x_and_y_l2006_200679

/-- The smallest positive integer x such that 420x is a perfect square -/
def x : ℕ := 105

/-- The smallest positive integer y such that 420y is a perfect cube -/
def y : ℕ := 22050

/-- 420 * x is a perfect square -/
axiom x_square : ∃ n : ℕ, 420 * x = n * n

/-- 420 * y is a perfect cube -/
axiom y_cube : ∃ n : ℕ, 420 * y = n * n * n

/-- x is the smallest positive integer such that 420x is a perfect square -/
axiom x_smallest : ∀ z : ℕ, z > 0 → z < x → ¬∃ n : ℕ, 420 * z = n * n

/-- y is the smallest positive integer such that 420y is a perfect cube -/
axiom y_smallest : ∀ z : ℕ, z > 0 → z < y → ¬∃ n : ℕ, 420 * z = n * n * n

theorem sum_of_x_and_y : x + y = 22155 := by sorry

end sum_of_x_and_y_l2006_200679


namespace bookshop_inventory_problem_l2006_200644

/-- Represents the bookshop inventory problem -/
theorem bookshop_inventory_problem (S : ℕ) : 
  743 - (S + 128 + 2*S + (128 + 34)) + 160 = 502 → S = 37 := by
  sorry

end bookshop_inventory_problem_l2006_200644


namespace quadratic_root_problem_l2006_200685

theorem quadratic_root_problem (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - m * x + 6 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 2 * y^2 - m * y + 6 = 0 ∧ y = 3) :=
by sorry

end quadratic_root_problem_l2006_200685


namespace count_integers_satisfying_conditions_l2006_200631

theorem count_integers_satisfying_conditions : 
  ∃! (S : Finset ℤ), 
    (∀ x ∈ S, ⌊Real.sqrt x⌋ = 8 ∧ x % 5 = 3) ∧ 
    (∀ x : ℤ, ⌊Real.sqrt x⌋ = 8 ∧ x % 5 = 3 → x ∈ S) ∧
    S.card = 3 := by
  sorry

end count_integers_satisfying_conditions_l2006_200631


namespace eva_patch_area_l2006_200649

/-- Represents a rectangular vegetable patch -/
structure VegetablePatch where
  short_side : ℕ  -- Number of posts on the shorter side
  long_side : ℕ   -- Number of posts on the longer side
  post_spacing : ℕ -- Distance between posts in yards

/-- Properties of Eva's vegetable patch -/
def eva_patch : VegetablePatch where
  short_side := 3
  long_side := 9
  post_spacing := 6

/-- Total number of posts -/
def total_posts (p : VegetablePatch) : ℕ :=
  2 * (p.short_side + p.long_side) - 4

/-- Relationship between short and long sides -/
def side_relationship (p : VegetablePatch) : Prop :=
  p.long_side = 3 * p.short_side

/-- Calculate the area of the vegetable patch -/
def patch_area (p : VegetablePatch) : ℕ :=
  (p.short_side - 1) * (p.long_side - 1) * p.post_spacing * p.post_spacing

/-- Theorem stating the area of Eva's vegetable patch -/
theorem eva_patch_area :
  total_posts eva_patch = 24 ∧
  side_relationship eva_patch ∧
  patch_area eva_patch = 576 := by
  sorry

#eval patch_area eva_patch

end eva_patch_area_l2006_200649


namespace derivative_sum_positive_l2006_200608

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * log x - b * x - 1 / x

theorem derivative_sum_positive (a : ℝ) (h_a : a > 0) (x₁ x₂ : ℝ) 
  (h_x₁ : x₁ > 0) (h_x₂ : x₂ > 0) (h_neq : x₁ ≠ x₂) :
  ∃ b : ℝ, f a b x₁ = f a b x₂ → 
    (deriv (f a b) x₁ + deriv (f a b) x₂ > 0) := by
  sorry

end derivative_sum_positive_l2006_200608


namespace perfect_square_m_l2006_200622

theorem perfect_square_m (n : ℕ) (m : ℤ) 
  (h1 : m = 2 + 2 * Int.sqrt (44 * n^2 + 1)) : 
  ∃ k : ℤ, m = k^2 := by
sorry

end perfect_square_m_l2006_200622


namespace range_of_a_l2006_200601

/-- Line l: 3x + 4y + a = 0 -/
def line_l (a : ℝ) (x y : ℝ) : Prop := 3*x + 4*y + a = 0

/-- Circle C: (x-2)² + y² = 2 -/
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

/-- Point M is on line l -/
def M_on_line_l (a : ℝ) (M : ℝ × ℝ) : Prop := line_l a M.1 M.2

/-- Tangent condition: �angle PMQ = 90° -/
def tangent_condition (M : ℝ × ℝ) : Prop := 
  ∃ (P Q : ℝ × ℝ), circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧ 
    (M.1 - P.1) * (M.1 - Q.1) + (M.2 - P.2) * (M.2 - Q.2) = 0

/-- Main theorem -/
theorem range_of_a (a : ℝ) : 
  (∃ M : ℝ × ℝ, M_on_line_l a M ∧ tangent_condition M) → 
  -16 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l2006_200601


namespace string_cheese_cost_is_10_cents_l2006_200607

/-- The cost of each piece of string cheese in cents -/
def string_cheese_cost (num_packs : ℕ) (cheeses_per_pack : ℕ) (total_cost : ℚ) : ℚ :=
  (total_cost * 100) / (num_packs * cheeses_per_pack)

/-- Theorem: The cost of each piece of string cheese is 10 cents -/
theorem string_cheese_cost_is_10_cents :
  string_cheese_cost 3 20 6 = 10 := by
  sorry

end string_cheese_cost_is_10_cents_l2006_200607


namespace total_students_is_47_l2006_200676

/-- The number of students supposed to be in Miss Smith's English class -/
def total_students : ℕ :=
  let tables : ℕ := 6
  let students_per_table : ℕ := 3
  let bathroom_students : ℕ := 3
  let canteen_multiplier : ℕ := 3
  let new_groups : ℕ := 2
  let students_per_new_group : ℕ := 4
  let german_students : ℕ := 3
  let french_students : ℕ := 3
  let norwegian_students : ℕ := 3

  let current_students := tables * students_per_table
  let missing_students := bathroom_students + (canteen_multiplier * bathroom_students)
  let new_group_students := new_groups * students_per_new_group
  let exchange_students := german_students + french_students + norwegian_students

  current_students + missing_students + new_group_students + exchange_students

theorem total_students_is_47 : total_students = 47 := by
  sorry

end total_students_is_47_l2006_200676


namespace one_third_complex_point_l2006_200691

theorem one_third_complex_point (z₁ z₂ z : ℂ) :
  z₁ = -5 + 6*I →
  z₂ = 7 - 4*I →
  z = (1 - 1/3) * z₁ + 1/3 * z₂ →
  z = -1 + 8/3 * I :=
by sorry

end one_third_complex_point_l2006_200691


namespace coffee_purchase_l2006_200647

theorem coffee_purchase (gift_card : ℝ) (coffee_price : ℝ) (remaining : ℝ) 
  (h1 : gift_card = 70)
  (h2 : coffee_price = 8.58)
  (h3 : remaining = 35.68) :
  (gift_card - remaining) / coffee_price = 4 := by
  sorry

end coffee_purchase_l2006_200647


namespace star_difference_l2006_200640

def star (x y : ℝ) : ℝ := x * y - 3 * x + y

theorem star_difference : (star 5 8) - (star 8 5) = 12 := by
  sorry

end star_difference_l2006_200640


namespace total_spots_l2006_200654

def cow_spots (left_spots : ℕ) (right_spots : ℕ) : Prop :=
  (left_spots = 16) ∧ 
  (right_spots = 3 * left_spots + 7)

theorem total_spots : ∀ left_spots right_spots : ℕ, 
  cow_spots left_spots right_spots → left_spots + right_spots = 71 := by
sorry

end total_spots_l2006_200654


namespace smallest_m_is_30_l2006_200678

def probability_condition (m : ℕ) : Prop :=
  (1 / 6) * ((m - 4) ^ 3 : ℚ) / (m ^ 3 : ℚ) > 3 / 5

theorem smallest_m_is_30 :
  ∀ k : ℕ, k > 0 → (probability_condition k → k ≥ 30) ∧
  probability_condition 30 := by sorry

end smallest_m_is_30_l2006_200678


namespace simplify_expression_l2006_200689

theorem simplify_expression : (9 * 10^10) / (3 * 10^3 - 2 * 10^3) = 9 * 10^7 := by
  sorry

end simplify_expression_l2006_200689


namespace absolute_difference_of_numbers_l2006_200611

theorem absolute_difference_of_numbers (x y : ℝ) 
  (sum_condition : x + y = 34) 
  (product_condition : x * y = 240) : 
  |x - y| = 14 := by
sorry

end absolute_difference_of_numbers_l2006_200611


namespace student_count_l2006_200629

theorem student_count (x : ℕ) (h : x > 0) :
  (Nat.choose x 4 : ℚ) / (x * (x - 1)) = 13 / 2 → x = 15 := by
  sorry

end student_count_l2006_200629


namespace total_animals_l2006_200614

theorem total_animals (num_pigs num_giraffes : ℕ) : 
  num_pigs = 7 → num_giraffes = 6 → num_pigs + num_giraffes = 13 := by
  sorry

end total_animals_l2006_200614


namespace max_discount_rate_l2006_200600

theorem max_discount_rate (cost : ℝ) (original_price : ℝ) 
  (h1 : cost = 4) (h2 : original_price = 5) : 
  ∃ (max_discount : ℝ), 
    (∀ (discount : ℝ), discount ≤ max_discount → 
      (original_price * (1 - discount / 100) - cost) / cost ≥ 0.1) ∧
    (∀ (discount : ℝ), discount > max_discount → 
      (original_price * (1 - discount / 100) - cost) / cost < 0.1) ∧
    max_discount = 12 :=
sorry

end max_discount_rate_l2006_200600


namespace intersection_equality_implies_m_equals_five_l2006_200690

def A (m : ℝ) : Set ℝ := {-1, 3, m}
def B : Set ℝ := {3, 5}

theorem intersection_equality_implies_m_equals_five (m : ℝ) :
  B ∩ A m = B → m = 5 := by
  sorry

end intersection_equality_implies_m_equals_five_l2006_200690


namespace difference_C_D_l2006_200634

def C : ℤ := (Finset.range 20).sum (fun i => (2*i + 2) * (2*i + 3)) + 42

def D : ℤ := 2 + (Finset.range 20).sum (fun i => (2*i + 3) * (2*i + 4))

theorem difference_C_D : |C - D| = 400 := by sorry

end difference_C_D_l2006_200634


namespace quadratic_roots_relation_l2006_200605

theorem quadratic_roots_relation (b c p q r s : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 + p*x + q = 0 ↔ x = r^2 ∨ x = s^2) →
  p = 2*c - b^2 :=
by sorry

end quadratic_roots_relation_l2006_200605


namespace triangle_problem_l2006_200668

/-- Given a triangle ABC with tanA = 1/4, tanB = 3/5, and AB = √17,
    prove that the measure of angle C is 3π/4 and the smallest side length is √2 -/
theorem triangle_problem (A B C : Real) (AB : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π →
  Real.tan A = 1/4 →
  Real.tan B = 3/5 →
  AB = Real.sqrt 17 →
  C = 3*π/4 ∧ 
  (min AB (min (AB * Real.sin A / Real.sin C) (AB * Real.sin B / Real.sin C)) = Real.sqrt 2) :=
by sorry

end triangle_problem_l2006_200668


namespace binary_1010_is_10_l2006_200638

def binary_to_decimal (b : List Bool) : ℕ :=
  List.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0 b

theorem binary_1010_is_10 :
  binary_to_decimal [true, false, true, false] = 10 := by
  sorry

end binary_1010_is_10_l2006_200638


namespace arithmetic_sum_odd_numbers_l2006_200665

theorem arithmetic_sum_odd_numbers : ∀ (a₁ aₙ d n : ℕ),
  a₁ = 1 →
  aₙ = 99 →
  d = 2 →
  aₙ = a₁ + (n - 1) * d →
  n * (a₁ + aₙ) / 2 = 2500 :=
by
  sorry

end arithmetic_sum_odd_numbers_l2006_200665


namespace markeesha_friday_sales_l2006_200657

/-- Proves that Markeesha sold 30 boxes on Friday given the conditions of the problem -/
theorem markeesha_friday_sales : ∀ (friday : ℕ), 
  (∃ (saturday sunday : ℕ),
    saturday = 2 * friday ∧
    sunday = saturday - 15 ∧
    friday + saturday + sunday = 135) →
  friday = 30 := by
sorry

end markeesha_friday_sales_l2006_200657


namespace twentieth_term_is_400_l2006_200645

/-- A second-order arithmetic sequence -/
def SecondOrderArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → (a n - a (n-1)) - (a (n-1) - a (n-2)) = 2

/-- The sequence starts with 1, 4, 9, 16 -/
def SequenceStart (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 4 ∧ a 3 = 9 ∧ a 4 = 16

theorem twentieth_term_is_400 (a : ℕ → ℕ) 
  (h1 : SecondOrderArithmeticSequence a) 
  (h2 : SequenceStart a) : 
  a 20 = 400 := by
  sorry

end twentieth_term_is_400_l2006_200645


namespace largest_multiple_of_seven_less_than_negative_thirty_l2006_200625

theorem largest_multiple_of_seven_less_than_negative_thirty :
  ∃ (n : ℤ), n * 7 = -35 ∧ 
  n * 7 < -30 ∧ 
  ∀ (m : ℤ), m * 7 < -30 → m * 7 ≤ -35 := by
sorry

end largest_multiple_of_seven_less_than_negative_thirty_l2006_200625


namespace length_of_segment_AB_is_10_l2006_200661

/-- Given point A with coordinates (2, -3, 5) and point B symmetrical to A with respect to the xy-plane,
    prove that the length of line segment AB is 10. -/
theorem length_of_segment_AB_is_10 :
  let A : ℝ × ℝ × ℝ := (2, -3, 5)
  let B : ℝ × ℝ × ℝ := (2, -3, -5)  -- B is symmetrical to A with respect to xy-plane
  ‖A - B‖ = 10 := by
  sorry

end length_of_segment_AB_is_10_l2006_200661


namespace total_ways_is_2531_l2006_200697

/-- The number of different types of cookies -/
def num_cookie_types : ℕ := 6

/-- The number of different types of milk -/
def num_milk_types : ℕ := 4

/-- The total number of product types -/
def total_product_types : ℕ := num_cookie_types + num_milk_types

/-- The number of products they purchase collectively -/
def total_purchases : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of ways Charlie and Delta can leave the store with 4 products collectively -/
def total_ways : ℕ :=
  -- Charlie 4, Delta 0
  choose total_product_types 4 +
  -- Charlie 3, Delta 1
  choose total_product_types 3 * num_cookie_types +
  -- Charlie 2, Delta 2
  choose total_product_types 2 * (choose num_cookie_types 2 + num_cookie_types) +
  -- Charlie 1, Delta 3
  total_product_types * (choose num_cookie_types 3 + num_cookie_types * (num_cookie_types - 1) + num_cookie_types) +
  -- Charlie 0, Delta 4
  (choose num_cookie_types 4 + num_cookie_types * (num_cookie_types - 1) + choose num_cookie_types 2 * 3 + num_cookie_types)

theorem total_ways_is_2531 : total_ways = 2531 := by
  sorry

end total_ways_is_2531_l2006_200697


namespace driver_hourly_wage_l2006_200627

/-- Calculates the hourly wage of a driver after fuel costs --/
theorem driver_hourly_wage
  (speed : ℝ)
  (time : ℝ)
  (fuel_efficiency : ℝ)
  (income_per_mile : ℝ)
  (fuel_cost_per_gallon : ℝ)
  (h1 : speed = 60)
  (h2 : time = 2)
  (h3 : fuel_efficiency = 30)
  (h4 : income_per_mile = 0.5)
  (h5 : fuel_cost_per_gallon = 2)
  : (income_per_mile * speed * time - (speed * time / fuel_efficiency) * fuel_cost_per_gallon) / time = 26 := by
  sorry

end driver_hourly_wage_l2006_200627


namespace quadratic_always_nonnegative_l2006_200604

theorem quadratic_always_nonnegative (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 2) * x + (1/4 : ℝ) ≥ 0) ↔ 1 ≤ a ∧ a ≤ 3 := by
  sorry

end quadratic_always_nonnegative_l2006_200604


namespace least_k_for_inequality_l2006_200653

theorem least_k_for_inequality (k : ℤ) : 
  (∀ m : ℤ, m < k → (0.0010101 * (10 : ℝ) ^ m ≤ 100)) ∧ 
  (0.0010101 * (10 : ℝ) ^ k > 100) → 
  k = 6 := by sorry

end least_k_for_inequality_l2006_200653


namespace peters_age_l2006_200698

theorem peters_age (P Q : ℝ) 
  (h1 : Q - P = P / 2)
  (h2 : P + Q = 35) :
  Q = 21 := by sorry

end peters_age_l2006_200698


namespace sin_phi_value_l2006_200692

theorem sin_phi_value (φ : ℝ) : 
  (∀ x, 2 * Real.sin x + Real.cos x = 2 * Real.sin (x - φ) - Real.cos (x - φ)) →
  Real.sin φ = 4/5 := by
sorry

end sin_phi_value_l2006_200692


namespace total_sand_donation_l2006_200660

-- Define the amounts of sand for each city
def city_A : ℚ := 16 + 1/2
def city_B : ℕ := 26
def city_C : ℚ := 24 + 1/2
def city_D : ℕ := 28

-- Theorem statement
theorem total_sand_donation :
  city_A + city_B + city_C + city_D = 95 := by
  sorry

end total_sand_donation_l2006_200660


namespace product_expansion_sum_l2006_200621

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + c + d = 19 := by
  sorry

end product_expansion_sum_l2006_200621


namespace f_2_3_4_equals_59_l2006_200695

def f (x y z : ℝ) : ℝ := 2 * x^3 + 3 * y^2 + z^2

theorem f_2_3_4_equals_59 : f 2 3 4 = 59 := by
  sorry

end f_2_3_4_equals_59_l2006_200695


namespace storage_unit_area_l2006_200623

theorem storage_unit_area :
  let total_units : ℕ := 42
  let total_area : ℕ := 5040
  let known_units : ℕ := 20
  let known_unit_length : ℕ := 8
  let known_unit_width : ℕ := 4
  let remaining_units := total_units - known_units
  let known_units_area := known_units * known_unit_length * known_unit_width
  let remaining_area := total_area - known_units_area
  remaining_area / remaining_units = 200 := by
sorry

end storage_unit_area_l2006_200623


namespace range_of_a_l2006_200658

def p (a : ℝ) : Prop := ∀ m ∈ Set.Icc (-1 : ℝ) 1, a^2 - 5*a - 3 ≥ Real.sqrt (m^2 + 8)

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 2 < 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) →
  a ∈ Set.Icc (-Real.sqrt 8) (-1) ∪ Set.Ioo (Real.sqrt 8) 6 :=
sorry

end range_of_a_l2006_200658


namespace students_not_enrolled_l2006_200609

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 87) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : both = 9) : 
  total - (french + german - both) = 33 :=
by sorry

end students_not_enrolled_l2006_200609


namespace linear_system_determinant_l2006_200663

/-- 
Given integers a, b, c, d such that the system of equations
  ax + by = m
  cx + dy = n
has integer solutions for all integer values of m and n,
prove that |ad - bc| = 1
-/
theorem linear_system_determinant (a b c d : ℤ) 
  (h : ∀ (m n : ℤ), ∃ (x y : ℤ), a * x + b * y = m ∧ c * x + d * y = n) :
  |a * d - b * c| = 1 :=
sorry

end linear_system_determinant_l2006_200663


namespace triple_transmission_better_for_zero_l2006_200632

-- Define the channel parameters
variable (α β : ℝ)

-- Define the conditions
variable (h1 : 0 < α)
variable (h2 : α < 0.5)
variable (h3 : 0 < β)
variable (h4 : β < 1)

-- Define the probabilities for single and triple transmission
def P_single_0 : ℝ := 1 - α
def P_triple_0 : ℝ := 3 * α * (1 - α)^2 + (1 - α)^3

-- State the theorem
theorem triple_transmission_better_for_zero :
  P_triple_0 α > P_single_0 α := by sorry

end triple_transmission_better_for_zero_l2006_200632


namespace P_in_xoz_plane_l2006_200652

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoz plane in 3D space -/
def xoz_plane : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- Point P with coordinates (-2, 0, 3) -/
def P : Point3D :=
  ⟨-2, 0, 3⟩

theorem P_in_xoz_plane : P ∈ xoz_plane := by
  sorry


end P_in_xoz_plane_l2006_200652


namespace school_trip_speed_l2006_200671

/-- Proves that the speed for the first half of a round trip is 6 km/hr,
    given the specified conditions. -/
theorem school_trip_speed
  (total_distance : ℝ)
  (return_speed : ℝ)
  (total_time : ℝ)
  (h1 : total_distance = 48)
  (h2 : return_speed = 4)
  (h3 : total_time = 10)
  : ∃ (going_speed : ℝ),
    going_speed = 6 ∧
    total_time = (total_distance / 2) / going_speed + (total_distance / 2) / return_speed :=
sorry

end school_trip_speed_l2006_200671


namespace subtract_like_terms_l2006_200620

theorem subtract_like_terms (a : ℝ) : 7 * a^2 - 4 * a^2 = 3 * a^2 := by
  sorry

end subtract_like_terms_l2006_200620


namespace unique_equation_solution_l2006_200637

/-- A function that checks if a list of integers contains exactly the digits 1 to 9 --/
def isValidDigitList (lst : List Int) : Prop :=
  lst.length = 9 ∧ (∀ n, n ∈ lst → 1 ≤ n ∧ n ≤ 9) ∧ lst.toFinset.card = 9

/-- A function that converts a list of three integers to a three-digit number --/
def toThreeDigitNumber (a b c : Int) : Int :=
  100 * a + 10 * b + c

/-- A function that converts a list of two integers to a two-digit number --/
def toTwoDigitNumber (a b : Int) : Int :=
  10 * a + b

theorem unique_equation_solution :
  ∃! (digits : List Int),
    isValidDigitList digits ∧
    7 ∈ digits ∧
    let abc := toThreeDigitNumber (digits[0]!) (digits[1]!) (digits[2]!)
    let de := toTwoDigitNumber (digits[3]!) (digits[4]!)
    let f := digits[5]!
    let h := digits[8]!
    abc / de = f ∧ f = h - 7 := by
  sorry

end unique_equation_solution_l2006_200637


namespace inequality_proof_l2006_200667

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0)
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) :
  x + y/2 + z/3 ≤ 1/3 := by
sorry

end inequality_proof_l2006_200667


namespace discriminant_neither_sufficient_nor_necessary_l2006_200635

/-- The function f(x) = ax^2 + bx + c --/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that the graph of f is always above the x-axis --/
def always_above (a b c : ℝ) : Prop :=
  ∀ x, f a b c x > 0

/-- The discriminant condition --/
def discriminant_condition (a b c : ℝ) : Prop :=
  b^2 - 4*a*c < 0

/-- The main theorem --/
theorem discriminant_neither_sufficient_nor_necessary :
  ¬(∀ a b c : ℝ, discriminant_condition a b c → always_above a b c) ∧
  ¬(∀ a b c : ℝ, always_above a b c → discriminant_condition a b c) := by
  sorry

end discriminant_neither_sufficient_nor_necessary_l2006_200635


namespace min_shift_for_symmetry_l2006_200687

theorem min_shift_for_symmetry (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x)) →
  φ > 0 →
  (∀ x, f (x - φ) = f (π / 6 - x)) →
  φ ≥ 5 * π / 12 :=
by sorry

end min_shift_for_symmetry_l2006_200687


namespace sum_f_positive_l2006_200642

def f (x : ℝ) : ℝ := x + x^3

theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end sum_f_positive_l2006_200642


namespace simplify_radical_sum_l2006_200617

theorem simplify_radical_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end simplify_radical_sum_l2006_200617


namespace cos_120_degrees_l2006_200641

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1/2 := by
  sorry

end cos_120_degrees_l2006_200641


namespace probability_is_three_twentyfifths_l2006_200651

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_max : ℝ
  y_max : ℝ

/-- A point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point satisfies the condition x > 5y --/
def satisfies_condition (p : Point) : Prop :=
  p.x > 5 * p.y

/-- Calculate the probability of a randomly chosen point satisfying the condition --/
def probability_satisfies_condition (r : Rectangle) : ℝ :=
  sorry

/-- The main theorem --/
theorem probability_is_three_twentyfifths :
  let r := Rectangle.mk 3000 2500
  probability_satisfies_condition r = 3 / 25 := by
  sorry

end probability_is_three_twentyfifths_l2006_200651


namespace age_difference_l2006_200674

/-- Given the ages of three people A, B, and C, prove that A is 2 years older than B. -/
theorem age_difference (A B C : ℕ) : 
  B = 18 →
  B = 2 * C →
  A + B + C = 47 →
  A = B + 2 := by
  sorry

end age_difference_l2006_200674


namespace train_speed_problem_l2006_200606

theorem train_speed_problem (train_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  crossing_time = 4 →
  ∃ (speed : ℝ),
    speed * crossing_time = 2 * train_length ∧
    speed * 3.6 = 108 :=
by sorry

end train_speed_problem_l2006_200606


namespace tan_half_product_l2006_200626

theorem tan_half_product (a b : Real) :
  3 * (Real.sin a + Real.sin b) + 2 * (Real.sin a * Real.sin b + 1) = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = -4 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -1 := by
  sorry

end tan_half_product_l2006_200626


namespace number_count_l2006_200648

theorem number_count (average : ℝ) (avg1 avg2 avg3 : ℝ) : 
  average = 6.40 →
  avg1 = 6.2 →
  avg2 = 6.1 →
  avg3 = 6.9 →
  (2 * avg1 + 2 * avg2 + 2 * avg3) / 6 = average →
  6 = (2 * avg1 + 2 * avg2 + 2 * avg3) / average :=
by sorry

end number_count_l2006_200648


namespace min_sum_of_squares_l2006_200613

theorem min_sum_of_squares (x y : ℝ) (h : (x + 5) * (y - 5) = 0) :
  ∃ (min : ℝ), min = 50 ∧ ∀ (a b : ℝ), (a + 5) * (b - 5) = 0 → a^2 + b^2 ≥ min :=
sorry

end min_sum_of_squares_l2006_200613


namespace solution_is_two_lines_l2006_200664

-- Define the equation
def equation (x y : ℝ) : Prop := (x - 2*y)^2 = x^2 - 4*y^2

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | equation p.1 p.2}

-- Define the two lines
def x_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}
def diagonal_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 * p.2}

-- Theorem statement
theorem solution_is_two_lines :
  solution_set = x_axis ∪ diagonal_line :=
sorry

end solution_is_two_lines_l2006_200664


namespace geologists_probability_l2006_200662

/-- Represents a circular field with radial roads -/
structure CircularField where
  numRoads : ℕ
  radius : ℝ

/-- Represents a geologist's position -/
structure GeologistPosition where
  road : ℕ
  distance : ℝ

/-- Calculates the distance between two geologists -/
def distanceBetweenGeologists (field : CircularField) (pos1 pos2 : GeologistPosition) : ℝ :=
  sorry

/-- Calculates the probability of two geologists being at least a certain distance apart -/
def probabilityOfDistance (field : CircularField) (speed time minDistance : ℝ) : ℝ :=
  sorry

theorem geologists_probability (field : CircularField) :
  field.numRoads = 6 →
  probabilityOfDistance field 4 1 6 = 0.5 := by
  sorry

end geologists_probability_l2006_200662


namespace star_identity_l2006_200616

/-- The binary operation * on pairs of real numbers -/
def star (p q : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * q.1, p.1 * q.2 + p.2 * q.1)

/-- The identity element for the star operation -/
def identity_element : ℝ × ℝ := (1, 0)

/-- Theorem stating that (1, 0) is the unique identity element for the star operation -/
theorem star_identity :
  ∀ p : ℝ × ℝ, star p identity_element = p ∧
  (∀ q : ℝ × ℝ, (∀ p : ℝ × ℝ, star p q = p) → q = identity_element) := by
  sorry

end star_identity_l2006_200616


namespace magnitude_of_z_l2006_200619

theorem magnitude_of_z (z : ℂ) (h : (Complex.I - 1) * z = (Complex.I + 1)^2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end magnitude_of_z_l2006_200619


namespace triangle_inequality_squared_l2006_200682

theorem triangle_inequality_squared (a b c : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
by sorry

end triangle_inequality_squared_l2006_200682


namespace extreme_value_cubic_l2006_200639

/-- Given a cubic function f(x) with an extreme value at x = 1, prove that a + b = -7 -/
theorem extreme_value_cubic (a b : ℝ) : 
  let f := fun x : ℝ ↦ x^3 + a*x^2 + b*x + a^2
  (∃ (ε : ℝ), ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∧ 
  (∃ (ε : ℝ), ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  f 1 = 10 →
  a + b = -7 :=
by sorry

end extreme_value_cubic_l2006_200639


namespace min_colors_needed_l2006_200670

/-- Represents a coloring of a 2023 x 2023 grid --/
def Coloring := Fin 2023 → Fin 2023 → ℕ

/-- Checks if a coloring satisfies the given condition --/
def valid_coloring (c : Coloring) : Prop :=
  ∀ color row i,
    (∀ j ∈ Finset.range 6, c row (i + j) = color) →
    (∀ k < i, ∀ l > i + 5, ∀ m, c m k ≠ color ∧ c m l ≠ color)

/-- The main theorem stating the smallest number of colors needed --/
theorem min_colors_needed :
  (∃ (c : Coloring) (n : ℕ), n = 338 ∧ (∀ i j, c i j < n) ∧ valid_coloring c) ∧
  (∀ (c : Coloring) (n : ℕ), (∀ i j, c i j < n) ∧ valid_coloring c → n ≥ 338) :=
sorry

end min_colors_needed_l2006_200670


namespace chord_length_implies_a_values_point_m_existence_implies_a_range_l2006_200684

-- Define the circle C
def C (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - a - 1)^2 = 9

-- Define the line l
def l (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the point A
def A : ℝ × ℝ := (3, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Part 1
theorem chord_length_implies_a_values (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, C a x₁ y₁ ∧ C a x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4) →
  a = -1 ∨ a = 3 :=
sorry

-- Part 2
theorem point_m_existence_implies_a_range (a : ℝ) :
  (∃ x y : ℝ, C a x y ∧ (x - 3)^2 + y^2 = 4 * (x^2 + y^2)) →
  (-1 - 5 * Real.sqrt 2 / 2 ≤ a ∧ a ≤ -1 - Real.sqrt 2 / 2) ∨
  (-1 + Real.sqrt 2 / 2 ≤ a ∧ a ≤ -1 + 5 * Real.sqrt 2 / 2) :=
sorry

end chord_length_implies_a_values_point_m_existence_implies_a_range_l2006_200684


namespace border_area_is_144_l2006_200693

/-- The area of the border of a framed rectangular photograph -/
def border_area (photo_height photo_width border_width : ℝ) : ℝ :=
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - photo_height * photo_width

/-- Theorem: The area of the border of a framed rectangular photograph is 144 square inches -/
theorem border_area_is_144 :
  border_area 8 10 3 = 144 := by
  sorry

end border_area_is_144_l2006_200693


namespace unique_intersection_point_l2006_200683

theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, 
    (3 * p.1 - 2 * p.2 - 9 = 0) ∧
    (6 * p.1 + 4 * p.2 - 12 = 0) ∧
    (p.1 = 3) ∧
    (p.2 = -1) := by
  sorry

end unique_intersection_point_l2006_200683


namespace stacy_paper_completion_time_l2006_200686

/-- The number of days Stacy has to complete her paper -/
def days_to_complete : ℕ := 66 / 11

/-- The total number of pages in Stacy's paper -/
def total_pages : ℕ := 66

/-- The number of pages Stacy has to write per day -/
def pages_per_day : ℕ := 11

theorem stacy_paper_completion_time :
  days_to_complete = 6 :=
by sorry

end stacy_paper_completion_time_l2006_200686


namespace ellipse_eccentricity_l2006_200633

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) : Type :=
  (h : a > b)
  (h' : b > 0)

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse a b) : ℝ := sorry

/-- A point on the ellipse -/
def Point_on_ellipse (e : Ellipse a b) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The right vertex of the ellipse -/
def right_vertex (e : Ellipse a b) : ℝ × ℝ := (a, 0)

/-- The left focus of the ellipse -/
def left_focus (e : Ellipse a b) : ℝ × ℝ := sorry

/-- Predicate to check if a vector is twice another vector -/
def is_twice (v w : ℝ × ℝ) : Prop := v = 2 • w

/-- Theorem stating the eccentricity of the ellipse under given conditions -/
theorem ellipse_eccentricity (a b : ℝ) (e : Ellipse a b) 
  (B : ℝ × ℝ) (P : ℝ × ℝ) 
  (h_B : Point_on_ellipse e B.1 B.2)
  (h_BF_perp : (B.1 - (left_focus e).1) = 0)
  (h_P_on_y : P.1 = 0)
  (h_AP_PB : is_twice (right_vertex e - P) (P - B)) :
  eccentricity e = 1/2 := sorry

end ellipse_eccentricity_l2006_200633


namespace exists_solution_in_interval_l2006_200603

theorem exists_solution_in_interval : 
  ∃ z : ℝ, -10 ≤ z ∧ z ≤ 10 ∧ Real.exp (2 * z) = (z - 2) / (z + 2) := by
  sorry

end exists_solution_in_interval_l2006_200603


namespace lyle_percentage_l2006_200610

/-- Given a total number of chips and a ratio for division, 
    calculate the percentage of chips the second person receives. -/
def calculate_percentage (total_chips : ℕ) (ratio1 ratio2 : ℕ) : ℚ :=
  let total_parts := ratio1 + ratio2
  let chips_per_part := total_chips / total_parts
  let second_person_chips := ratio2 * chips_per_part
  (second_person_chips : ℚ) / total_chips * 100

/-- Theorem stating that given 100 chips divided in a 4:6 ratio, 
    the person with the larger share has 60% of the total chips. -/
theorem lyle_percentage : calculate_percentage 100 4 6 = 60 := by
  sorry

end lyle_percentage_l2006_200610


namespace converse_x_gt_abs_y_implies_x_gt_y_l2006_200677

theorem converse_x_gt_abs_y_implies_x_gt_y : ∀ x y : ℝ, x > |y| → x > y := by
  sorry

end converse_x_gt_abs_y_implies_x_gt_y_l2006_200677


namespace hanas_stamp_collection_value_l2006_200602

theorem hanas_stamp_collection_value :
  ∀ (total_value : ℚ),
    (4 / 7 : ℚ) * total_value +  -- Amount sold at garage sale
    (1 / 3 : ℚ) * ((3 / 7 : ℚ) * total_value) = 28 →  -- Amount sold at auction
    total_value = 196 := by
  sorry

end hanas_stamp_collection_value_l2006_200602


namespace min_even_integers_l2006_200636

theorem min_even_integers (a b c d e f g h : ℤ) : 
  a + b + c = 36 →
  a + b + c + d + e + f = 60 →
  a + b + c + d + e + f + g + h = 76 →
  g * h = 48 →
  ∃ (count : ℕ), count ≥ 1 ∧ 
    count = (if Even a then 1 else 0) + 
            (if Even b then 1 else 0) + 
            (if Even c then 1 else 0) + 
            (if Even d then 1 else 0) + 
            (if Even e then 1 else 0) + 
            (if Even f then 1 else 0) + 
            (if Even g then 1 else 0) + 
            (if Even h then 1 else 0) :=
by
  sorry

end min_even_integers_l2006_200636


namespace geometric_sequence_ratio_l2006_200675

/-- Given a number c that forms a geometric sequence when added to 20, 50, and 100, 
    the common ratio of this sequence is 5/3 -/
theorem geometric_sequence_ratio (c : ℝ) : 
  (∃ r : ℝ, (50 + c) / (20 + c) = r ∧ (100 + c) / (50 + c) = r) → 
  (50 + c) / (20 + c) = 5/3 :=
by sorry

end geometric_sequence_ratio_l2006_200675


namespace coin_difference_l2006_200646

def coin_values : List ℕ := [5, 10, 25]

def total_amount : ℕ := 35

def min_coins (values : List ℕ) (amount : ℕ) : ℕ :=
  sorry

def max_coins (values : List ℕ) (amount : ℕ) : ℕ :=
  sorry

theorem coin_difference :
  max_coins coin_values total_amount - min_coins coin_values total_amount = 5 :=
sorry

end coin_difference_l2006_200646


namespace permutations_count_l2006_200666

def original_number : List Nat := [1, 1, 2, 3, 4, 5, 6, 7]
def target_number : List Nat := [4, 6, 7, 5, 3, 2, 1, 1]

def count_permutations_less_than_or_equal (original : List Nat) (target : List Nat) : Nat :=
  sorry

theorem permutations_count :
  count_permutations_less_than_or_equal original_number target_number = 12240 :=
sorry

end permutations_count_l2006_200666


namespace unique_divisible_by_eight_l2006_200655

theorem unique_divisible_by_eight : ∃! n : ℕ, 70 < n ∧ n < 80 ∧ n % 8 = 0 := by
  sorry

end unique_divisible_by_eight_l2006_200655


namespace quadratic_root_implies_k_l2006_200659

theorem quadratic_root_implies_k (k : ℝ) : 
  ((k - 1) * 1^2 + k^2 - k = 0) → 
  (k - 1 ≠ 0) → 
  (k = -1) := by
  sorry

end quadratic_root_implies_k_l2006_200659


namespace min_coins_blind_pew_l2006_200681

/-- Represents the pirate's trunk with chests, boxes, and gold coins. -/
structure PirateTrunk where
  num_chests : Nat
  boxes_per_chest : Nat
  coins_per_box : Nat
  num_locks_opened : Nat

/-- Calculates the minimum number of gold coins that can be taken. -/
def min_coins_taken (trunk : PirateTrunk) : Nat :=
  let remaining_locks := trunk.num_locks_opened - 1 - trunk.num_chests
  remaining_locks * trunk.coins_per_box

/-- Theorem stating the minimum number of gold coins Blind Pew could take. -/
theorem min_coins_blind_pew :
  let trunk : PirateTrunk := {
    num_chests := 5,
    boxes_per_chest := 4,
    coins_per_box := 10,
    num_locks_opened := 9
  }
  min_coins_taken trunk = 30 := by
  sorry


end min_coins_blind_pew_l2006_200681


namespace remainder_divisibility_l2006_200650

theorem remainder_divisibility (n : ℤ) : 
  (2 * n) % 7 = 4 → n % 7 = 2 := by
sorry

end remainder_divisibility_l2006_200650


namespace f_is_quadratic_l2006_200680

/-- A quadratic equation in one variable x is of the form ax^2 + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation 3x^2 + 2x + 4 = 0 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 4

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l2006_200680


namespace purely_imaginary_Z_implies_m_equals_two_l2006_200699

-- Define the complex number Z as a function of m
def Z (m : ℝ) : ℂ := Complex.mk (m^2 - m - 2) (m^2 - 2*m - 3)

-- State the theorem
theorem purely_imaginary_Z_implies_m_equals_two :
  ∀ m : ℝ, (Z m).re = 0 ∧ (Z m).im ≠ 0 → m = 2 :=
by sorry

end purely_imaginary_Z_implies_m_equals_two_l2006_200699


namespace smallest_sum_with_lcm_2012_l2006_200696

theorem smallest_sum_with_lcm_2012 (a b c d e f g : ℕ) : 
  (Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d (Nat.lcm e (Nat.lcm f g)))))) = 2012 → 
  a + b + c + d + e + f + g ≥ 512 := by
  sorry

end smallest_sum_with_lcm_2012_l2006_200696


namespace flag_arrangement_modulo_l2006_200612

/-- The number of distinguishable arrangements of flags on two poles -/
def M : ℕ :=
  let total_flags := 17
  let blue_flags := 9
  let red_flags := 8
  let slots_for_red := blue_flags + 1
  let ways_to_place_red := Nat.choose slots_for_red red_flags
  let initial_arrangements := (blue_flags + 1) * ways_to_place_red
  let invalid_cases := 2 * Nat.choose blue_flags red_flags
  initial_arrangements - invalid_cases

/-- Theorem stating that M is congruent to 432 modulo 1000 -/
theorem flag_arrangement_modulo :
  M % 1000 = 432 := by sorry

end flag_arrangement_modulo_l2006_200612


namespace total_selection_schemes_l2006_200673

/-- The number of elective courses in each category (physical education and art) -/
def num_courses_per_category : ℕ := 4

/-- The minimum number of courses a student can choose -/
def min_courses : ℕ := 2

/-- The maximum number of courses a student can choose -/
def max_courses : ℕ := 3

/-- The number of categories (physical education and art) -/
def num_categories : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The total number of different course selection schemes is 64 -/
theorem total_selection_schemes : 
  (choose num_courses_per_category 1 * choose num_courses_per_category 1) + 
  (choose num_courses_per_category 2 * choose num_courses_per_category 1 + 
   choose num_courses_per_category 1 * choose num_courses_per_category 2) = 64 := by
  sorry

end total_selection_schemes_l2006_200673


namespace cylinder_lateral_area_l2006_200672

/-- Given a cylinder with height 4 and base area 9π, its lateral area is 24π. -/
theorem cylinder_lateral_area (h : ℝ) (base_area : ℝ) :
  h = 4 → base_area = 9 * Real.pi → 2 * Real.pi * (Real.sqrt (base_area / Real.pi)) * h = 24 * Real.pi := by
  sorry

end cylinder_lateral_area_l2006_200672


namespace sum_of_roots_l2006_200643

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a = 1) 
  (hb : b^3 - 3*b^2 + 5*b = 5) : 
  a + b = 2 := by sorry

end sum_of_roots_l2006_200643


namespace min_value_of_expression_l2006_200669

theorem min_value_of_expression (b : ℝ) (h : 8 * b^2 + 7 * b + 6 = 5) :
  ∃ (m : ℝ), (∀ b', 8 * b'^2 + 7 * b' + 6 = 5 → 3 * b' + 2 ≥ m) ∧ (3 * b + 2 = m) ∧ m = -1 := by
  sorry

end min_value_of_expression_l2006_200669


namespace intersection_of_M_and_N_l2006_200688

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 2}

theorem intersection_of_M_and_N : M ∩ N = {(2, 0)} := by
  sorry

end intersection_of_M_and_N_l2006_200688


namespace monotone_xfx_l2006_200656

open Real

theorem monotone_xfx (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, HasDerivAt f (f' x) x) 
  (h_ineq : ∀ x, x * f' x > -f x) (x₁ x₂ : ℝ) (h_lt : x₁ < x₂) : 
  x₁ * f x₁ < x₂ * f x₂ := by
  sorry

end monotone_xfx_l2006_200656
