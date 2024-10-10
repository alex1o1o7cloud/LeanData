import Mathlib

namespace remainder_46_pow_925_mod_21_l16_1643

theorem remainder_46_pow_925_mod_21 : 46^925 % 21 = 4 := by
  sorry

end remainder_46_pow_925_mod_21_l16_1643


namespace largest_prime_factors_difference_l16_1645

/-- The positive difference between the two largest prime factors of 159137 is 14 -/
theorem largest_prime_factors_difference (n : Nat) : n = 159137 → 
  ∃ (p q : Nat), Prime p ∧ Prime q ∧ p ∣ n ∧ q ∣ n ∧ 
  (∀ (r : Nat), Prime r → r ∣ n → r ≤ p) ∧
  (∀ (r : Nat), Prime r → r ∣ n → r ≠ p → r ≤ q) ∧
  p - q = 14 := by
sorry

end largest_prime_factors_difference_l16_1645


namespace max_value_when_a_is_one_a_values_when_max_is_two_l16_1664

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

-- Part 1
theorem max_value_when_a_is_one :
  ∃ (max : ℝ), (∀ x, f 1 x ≤ max) ∧ (∃ x, f 1 x = max) ∧ max = 1 := by sorry

-- Part 2
theorem a_values_when_max_is_two :
  (∃ (max : ℝ), (∀ x ∈ Set.Icc 0 1, f a x ≤ max) ∧ 
   (∃ x ∈ Set.Icc 0 1, f a x = max) ∧ max = 2) → (a = -1 ∨ a = 2) := by sorry

end max_value_when_a_is_one_a_values_when_max_is_two_l16_1664


namespace largest_angle_in_triangle_l16_1651

/-- Given a triangle DEF with side lengths d, e, and f satisfying certain conditions,
    prove that its largest angle is 120°. -/
theorem largest_angle_in_triangle (d e f : ℝ) (h1 : d + 2*e + 2*f = d^2) (h2 : d + 2*e - 2*f = -9) :
  ∃ (D E F : ℝ), D + E + F = 180 ∧ max D (max E F) = 120 := by
  sorry

end largest_angle_in_triangle_l16_1651


namespace max_value_theorem_l16_1677

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x*y - 1)^2 = (5*y + 2)*(y - 2)) : 
  x + 1/(2*y) ≤ 3/2 * Real.sqrt 2 - 1 :=
sorry

end max_value_theorem_l16_1677


namespace isosceles_trapezoid_dimensions_l16_1647

/-- An isosceles trapezoid with legs intersecting at a right angle -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- Length of the shorter base -/
  shorterBase : ℝ
  /-- Height of the trapezoid -/
  height : ℝ
  /-- Area of the trapezoid -/
  area : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The legs intersect at a right angle -/
  legsRightAngle : True
  /-- The area is calculated correctly -/
  areaEq : area = (longerBase + shorterBase) * height / 2

/-- Theorem about the dimensions of a specific isosceles trapezoid -/
theorem isosceles_trapezoid_dimensions (t : IsoscelesTrapezoid) 
  (h_area : t.area = 12)
  (h_height : t.height = 2) :
  t.longerBase = 8 ∧ t.shorterBase = 4 := by
  sorry


end isosceles_trapezoid_dimensions_l16_1647


namespace chicken_price_per_pound_l16_1621

/-- Given John's food order for a restaurant, prove the price per pound of chicken --/
theorem chicken_price_per_pound (beef_quantity : ℕ) (beef_price : ℚ) 
  (total_cost : ℚ) (chicken_quantity : ℕ) (chicken_price : ℚ) : chicken_price = 3 :=
by
  have h1 : beef_quantity = 1000 := by sorry
  have h2 : beef_price = 8 := by sorry
  have h3 : chicken_quantity = 2 * beef_quantity := by sorry
  have h4 : total_cost = 14000 := by sorry
  have h5 : total_cost = beef_quantity * beef_price + chicken_quantity * chicken_price := by sorry
  sorry

end chicken_price_per_pound_l16_1621


namespace min_money_required_l16_1629

/-- Represents the number of candies of each type -/
structure CandyCounts where
  apple : ℕ
  orange : ℕ
  strawberry : ℕ
  grape : ℕ

/-- Represents the vending machine with given conditions -/
def VendingMachine (c : CandyCounts) : Prop :=
  c.apple = 2 * c.orange ∧
  c.strawberry = 2 * c.grape ∧
  c.apple = 2 * c.strawberry ∧
  c.apple + c.orange + c.strawberry + c.grape = 90

/-- The cost of a single candy -/
def candy_cost : ℚ := 1/10

/-- The minimum number of candies to buy -/
def min_candies_to_buy (c : CandyCounts) : ℕ :=
  min c.grape 10 + 3 + 3 + 3

/-- The theorem to prove -/
theorem min_money_required (c : CandyCounts) (h : VendingMachine c) :
  (min_candies_to_buy c : ℚ) * candy_cost = 19/10 := by
  sorry


end min_money_required_l16_1629


namespace circle_equation_k_value_l16_1615

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def is_circle_equation (h k r : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The given equation -/
def given_equation (k : ℝ) (x y : ℝ) : ℝ :=
  x^2 + 14*x + y^2 + 8*y - k

theorem circle_equation_k_value :
  ∃! k : ℝ, is_circle_equation (-7) (-4) 5 (given_equation k) ∧ k = -40 := by
sorry

end circle_equation_k_value_l16_1615


namespace ethans_work_hours_l16_1696

/-- Proves that Ethan works 8 hours per day given his earnings and work schedule --/
theorem ethans_work_hours 
  (hourly_rate : ℝ) 
  (days_per_week : ℕ) 
  (total_earnings : ℝ) 
  (total_weeks : ℕ) 
  (h1 : hourly_rate = 18)
  (h2 : days_per_week = 5)
  (h3 : total_earnings = 3600)
  (h4 : total_weeks = 5) :
  (total_earnings / total_weeks) / days_per_week / hourly_rate = 8 := by
  sorry

#check ethans_work_hours

end ethans_work_hours_l16_1696


namespace product_coefficient_equality_l16_1626

theorem product_coefficient_equality (m : ℝ) : 
  (∃ a b c d : ℝ, (x^2 - m*x + 2) * (2*x + 1) = a*x^3 + b*x^2 + b*x + d) → m = -3 := by
  sorry

end product_coefficient_equality_l16_1626


namespace triangle_problem_l16_1686

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  A > 0 ∧ B > 0 ∧ C > 0 →  -- Positive angles
  A + B + C = π →  -- Sum of angles in a triangle
  a * Real.cos B = 3 →
  b * Real.cos A = 1 →
  A - B = π / 6 →
  c = 4 ∧ B = π / 6 := by
sorry

end triangle_problem_l16_1686


namespace sufficient_not_necessary_l16_1667

theorem sufficient_not_necessary :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → y / x + x / y ≥ 2) ∧
  (∃ x y : ℝ, y / x + x / y ≥ 2 ∧ ¬(x > 0 ∧ y > 0)) := by
  sorry

end sufficient_not_necessary_l16_1667


namespace angle_A_is_60_degrees_sides_b_c_are_2_l16_1689

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0

def has_area_sqrt_3 (t : Triangle) : Prop :=
  1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3

-- Theorem 1
theorem angle_A_is_60_degrees (t : Triangle) 
  (h : satisfies_condition t) : t.A = Real.pi / 3 := by
  sorry

-- Theorem 2
theorem sides_b_c_are_2 (t : Triangle) 
  (h1 : satisfies_condition t)
  (h2 : t.a = 2)
  (h3 : has_area_sqrt_3 t) : t.b = 2 ∧ t.c = 2 := by
  sorry

end angle_A_is_60_degrees_sides_b_c_are_2_l16_1689


namespace complement_of_A_in_U_l16_1639

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {0, 3, 4} := by sorry

end complement_of_A_in_U_l16_1639


namespace similar_triangles_perimeter_ratio_l16_1658

-- Define the Triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the similarity relation between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the similarity ratio between triangles
def similarityRatio (t1 t2 : Triangle) : ℝ := sorry

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem similar_triangles_perimeter_ratio 
  (ABC DEF : Triangle) 
  (h_similar : similar ABC DEF) 
  (h_ratio : similarityRatio ABC DEF = 1 / 2) : 
  perimeter ABC / perimeter DEF = 1 / 2 := by sorry

end similar_triangles_perimeter_ratio_l16_1658


namespace one_dollar_bills_count_l16_1611

/-- Represents the number of bills of each denomination -/
structure WalletContent where
  ones : ℕ
  twos : ℕ
  fives : ℕ

/-- Calculates the total number of bills -/
def total_bills (w : WalletContent) : ℕ :=
  w.ones + w.twos + w.fives

/-- Calculates the total amount of money -/
def total_money (w : WalletContent) : ℕ :=
  w.ones + 2 * w.twos + 5 * w.fives

/-- Theorem stating that given the conditions, the number of one dollar bills is 20 -/
theorem one_dollar_bills_count (w : WalletContent) 
  (h1 : total_bills w = 60) 
  (h2 : total_money w = 120) : 
  w.ones = 20 := by
  sorry

end one_dollar_bills_count_l16_1611


namespace f_passes_through_origin_l16_1654

def f (x : ℝ) : ℝ := -2 * x

theorem f_passes_through_origin : f 0 = 0 := by
  sorry

end f_passes_through_origin_l16_1654


namespace complex_number_in_first_quadrant_l16_1613

/-- Given a complex number z = i(1-i), prove that it corresponds to a point in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := Complex.I * (1 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_first_quadrant_l16_1613


namespace cos_beta_value_l16_1685

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 2) (h4 : Real.sin (α + β) = Real.sqrt 2 / 2) :
  Real.cos β = Real.sqrt 10 / 10 := by
sorry

end cos_beta_value_l16_1685


namespace smallest_distance_between_circles_l16_1657

theorem smallest_distance_between_circles (z w : ℂ) 
  (hz : Complex.abs (z + 2 + 4*I) = 2)
  (hw : Complex.abs (w - 6 - 7*I) = 4) :
  ∃ (z' w' : ℂ), 
    Complex.abs (z' + 2 + 4*I) = 2 ∧ 
    Complex.abs (w' - 6 - 7*I) = 4 ∧
    Complex.abs (z' - w') = Real.sqrt 185 - 6 ∧
    ∀ (z'' w'' : ℂ), 
      Complex.abs (z'' + 2 + 4*I) = 2 → 
      Complex.abs (w'' - 6 - 7*I) = 4 → 
      Complex.abs (z'' - w'') ≥ Real.sqrt 185 - 6 :=
by sorry

end smallest_distance_between_circles_l16_1657


namespace pure_imaginary_value_l16_1690

theorem pure_imaginary_value (x : ℝ) : 
  (((x^2 - 1) : ℂ) + (x + 1) * Complex.I).re = 0 ∧ 
  (((x^2 - 1) : ℂ) + (x + 1) * Complex.I).im ≠ 0 → 
  x = 1 := by sorry

end pure_imaginary_value_l16_1690


namespace complex_square_equality_l16_1622

theorem complex_square_equality (a b : ℝ) : 
  (a + Complex.I = 2 - b * Complex.I) → (a + b * Complex.I)^2 = 3 - 4 * Complex.I :=
by sorry

end complex_square_equality_l16_1622


namespace unique_solution_to_diophantine_equation_l16_1640

theorem unique_solution_to_diophantine_equation :
  ∃! (a b c : ℕ+), 11^(a:ℕ) + 3^(b:ℕ) = (c:ℕ)^2 ∧ a = 4 ∧ b = 5 ∧ c = 122 := by
  sorry

end unique_solution_to_diophantine_equation_l16_1640


namespace digit_150_is_5_l16_1662

/-- The decimal representation of 5/13 as a list of digits -/
def decimal_rep_5_13 : List Nat := [3, 8, 4, 6, 1, 5]

/-- The length of the repeating sequence in the decimal representation of 5/13 -/
def repeat_length : Nat := 6

/-- The 150th digit after the decimal point in the decimal representation of 5/13 -/
def digit_150 : Nat :=
  decimal_rep_5_13[(150 - 1) % repeat_length]

theorem digit_150_is_5 : digit_150 = 5 := by sorry

end digit_150_is_5_l16_1662


namespace quadratic_increasing_condition_l16_1628

/-- A quadratic function f(x) = x^2 + 2mx + 10 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 10

/-- The function is increasing on [2, +∞) -/
def increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, 2 ≤ x → x < y → f m x < f m y

theorem quadratic_increasing_condition (m : ℝ) :
  increasing_on_interval m → m ≥ -2 := by
  sorry

end quadratic_increasing_condition_l16_1628


namespace matrix_product_sum_l16_1699

def A (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; y, 4]
def B (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, 6; 7, 8]

theorem matrix_product_sum (x y : ℝ) :
  A y * B x = !![19, 22; 43, 50] →
  x + y = 8 := by sorry

end matrix_product_sum_l16_1699


namespace first_expedition_duration_l16_1668

theorem first_expedition_duration (total_days : ℕ) 
  (h1 : total_days = 126) : ∃ (x : ℕ), 
  x * 7 + (x + 2) * 7 + 2 * (x + 2) * 7 = total_days ∧ x = 3 := by
  sorry

end first_expedition_duration_l16_1668


namespace total_wax_needed_l16_1675

def wax_already_has : ℕ := 28
def wax_still_needs : ℕ := 260

theorem total_wax_needed : wax_already_has + wax_still_needs = 288 := by
  sorry

end total_wax_needed_l16_1675


namespace original_cost_of_dvd_pack_l16_1608

theorem original_cost_of_dvd_pack (discount : ℕ) (price_after_discount : ℕ) 
  (h1 : discount = 25)
  (h2 : price_after_discount = 51) :
  discount + price_after_discount = 76 := by
sorry

end original_cost_of_dvd_pack_l16_1608


namespace linear_function_point_relation_l16_1692

/-- Given a linear function y = -x + 6 and two points A(-1, y₁) and B(2, y₂) on its graph, prove that y₁ > y₂ -/
theorem linear_function_point_relation (y₁ y₂ : ℝ) : 
  (∀ x : ℝ, -x + 6 = y₁ → x = -1) →  -- Point A(-1, y₁) is on the graph
  (∀ x : ℝ, -x + 6 = y₂ → x = 2) →   -- Point B(2, y₂) is on the graph
  y₁ > y₂ := by
sorry

end linear_function_point_relation_l16_1692


namespace smallest_square_area_l16_1665

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a square given its side length -/
def square_area (side : ℝ) : ℝ := side * side

/-- Checks if two rectangles can fit in a square of given side length without overlapping -/
def can_fit_in_square (r1 r2 : Rectangle) (side : ℝ) : Prop :=
  (min r1.width r1.height + min r2.width r2.height ≤ side) ∧
  (max r1.width r1.height + max r2.width r2.height ≤ side)

theorem smallest_square_area (r1 r2 : Rectangle) : 
  r1.width = 3 ∧ r1.height = 4 ∧ r2.width = 4 ∧ r2.height = 5 →
  ∃ (side : ℝ), 
    can_fit_in_square r1 r2 side ∧ 
    square_area side = 49 ∧
    ∀ (s : ℝ), can_fit_in_square r1 r2 s → square_area s ≥ 49 := by
  sorry

end smallest_square_area_l16_1665


namespace rachels_age_l16_1612

/-- Given that Rachel is 4 years older than Leah and the sum of their ages is 34,
    prove that Rachel is 19 years old. -/
theorem rachels_age (rachel_age leah_age : ℕ) 
    (h1 : rachel_age = leah_age + 4)
    (h2 : rachel_age + leah_age = 34) : 
  rachel_age = 19 := by
  sorry

end rachels_age_l16_1612


namespace simplify_sqrt_expression_l16_1620

theorem simplify_sqrt_expression (m : ℝ) (h : m < 1) : 
  Real.sqrt (m^2 - 2*m + 1) = 1 - m := by
  sorry

end simplify_sqrt_expression_l16_1620


namespace square_perimeter_l16_1694

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (2 * s + 2 * (s / 5) = 36) → (4 * s = 60) := by
  sorry

end square_perimeter_l16_1694


namespace trigonometric_calculations_l16_1637

theorem trigonometric_calculations :
  (2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1/2) ∧
  ((-1)^2023 + 2 * Real.sin (45 * π / 180) - Real.cos (30 * π / 180) + Real.sin (60 * π / 180) + (Real.tan (60 * π / 180))^2 = 2 + Real.sqrt 2) :=
by sorry

end trigonometric_calculations_l16_1637


namespace greatest_distance_between_circle_centers_l16_1648

/-- The greatest possible distance between the centers of two circles in a rectangle -/
theorem greatest_distance_between_circle_centers 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (circle_diameter : ℝ) 
  (h1 : rectangle_width = 20) 
  (h2 : rectangle_height = 15) 
  (h3 : circle_diameter = 10) :
  ∃ (d : ℝ), d = 5 * Real.sqrt 5 ∧ 
  ∀ (d' : ℝ), d' ≤ d ∧ 
  ∃ (x1 y1 x2 y2 : ℝ), 
    0 ≤ x1 ∧ x1 ≤ rectangle_width ∧
    0 ≤ y1 ∧ y1 ≤ rectangle_height ∧
    0 ≤ x2 ∧ x2 ≤ rectangle_width ∧
    0 ≤ y2 ∧ y2 ≤ rectangle_height ∧
    circle_diameter / 2 ≤ x1 ∧ x1 ≤ rectangle_width - circle_diameter / 2 ∧
    circle_diameter / 2 ≤ y1 ∧ y1 ≤ rectangle_height - circle_diameter / 2 ∧
    circle_diameter / 2 ≤ x2 ∧ x2 ≤ rectangle_width - circle_diameter / 2 ∧
    circle_diameter / 2 ≤ y2 ∧ y2 ≤ rectangle_height - circle_diameter / 2 ∧
    d' = Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) :=
by sorry

end greatest_distance_between_circle_centers_l16_1648


namespace power_2014_of_abs_one_l16_1635

theorem power_2014_of_abs_one (a : ℝ) : |a| = 1 → a^2014 = 1 := by sorry

end power_2014_of_abs_one_l16_1635


namespace power_equation_l16_1602

theorem power_equation (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(2*m + n) = 12 := by
  sorry

end power_equation_l16_1602


namespace pencil_sales_theorem_l16_1601

/-- The number of pencils initially sold for a rupee when losing 30% --/
def initial_pencils : ℝ := 20

/-- The number of pencils sold for a rupee when gaining 30% --/
def gain_pencils : ℝ := 10.77

/-- The percentage of cost price when losing 30% --/
def loss_percentage : ℝ := 0.7

/-- The percentage of cost price when gaining 30% --/
def gain_percentage : ℝ := 1.3

theorem pencil_sales_theorem :
  initial_pencils * loss_percentage = gain_pencils * gain_percentage := by
  sorry

#check pencil_sales_theorem

end pencil_sales_theorem_l16_1601


namespace completing_square_l16_1656

theorem completing_square (x : ℝ) : x^2 - 4*x + 2 = 0 ↔ (x - 2)^2 = 2 := by
  sorry

end completing_square_l16_1656


namespace number_problem_l16_1636

theorem number_problem (n : ℝ) : (0.4 * (3/5) * n = 36) → n = 150 := by
  sorry

end number_problem_l16_1636


namespace smaller_number_problem_l16_1617

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 8) : 
  min x y = 5 := by
sorry

end smaller_number_problem_l16_1617


namespace point_N_coordinates_l16_1673

-- Define the points and lines
def M : ℝ × ℝ := (0, -1)
def N : ℝ × ℝ := (2, 3)

def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the perpendicular property
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem point_N_coordinates :
  line1 N.1 N.2 ∧
  perpendicular 
    ((N.2 - M.2) / (N.1 - M.1)) 
    (-(1 / 2)) →
  N = (2, 3) := by
  sorry

end point_N_coordinates_l16_1673


namespace triangle_area_bound_l16_1683

-- Define a triangle with integer coordinates
structure IntTriangle where
  A : ℤ × ℤ
  B : ℤ × ℤ
  C : ℤ × ℤ

-- Define a function to count integer points inside a triangle
def countInteriorPoints (t : IntTriangle) : ℕ := sorry

-- Define a function to count integer points on the edges of a triangle
def countBoundaryPoints (t : IntTriangle) : ℕ := sorry

-- Define a function to calculate the area of a triangle
def triangleArea (t : IntTriangle) : ℚ := sorry

-- Theorem statement
theorem triangle_area_bound (t : IntTriangle) :
  countInteriorPoints t = 1 → triangleArea t ≤ 9/2 :=
sorry

end triangle_area_bound_l16_1683


namespace fraction_simplification_l16_1681

theorem fraction_simplification :
  (36 : ℚ) / 19 * 57 / 40 * 95 / 171 = 3 / 2 := by
  sorry

end fraction_simplification_l16_1681


namespace parallel_lines_slope_parallel_line_k_value_l16_1633

/-- A line through two points is parallel to another line if and only if their slopes are equal -/
theorem parallel_lines_slope (x1 y1 x2 y2 a b c : ℝ) :
  (∀ x y, a * x + b * y = c → y = (-a/b) * x + c/b) →
  (y2 - y1) / (x2 - x1) = -a/b ↔ 
  (∀ x y, y - y1 = ((y2 - y1) / (x2 - x1)) * (x - x1) → a * x + b * y = c) :=
sorry

/-- The value of k for which the line through (4, 3) and (k, -5) is parallel to 3x - 2y = 6 -/
theorem parallel_line_k_value : 
  (∃! k : ℝ, ((-5) - 3) / (k - 4) = (-3) / (-2) ∧ 
              ∀ x y : ℝ, y - 3 = ((-5) - 3) / (k - 4) * (x - 4) → 
                3 * x + (-2) * y = 6) ∧
  (∃! k : ℝ, ((-5) - 3) / (k - 4) = (-3) / (-2) ∧ 
              ∀ x y : ℝ, y - 3 = ((-5) - 3) / (k - 4) * (x - 4) → 
                3 * x + (-2) * y = 6) → k = -4/3 :=
sorry

end parallel_lines_slope_parallel_line_k_value_l16_1633


namespace point_P_coordinates_l16_1676

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of point P -/
def P (m : ℝ) : Point :=
  { x := 3 * m - 6, y := m + 1 }

/-- Definition of point A -/
def A : Point :=
  { x := 1, y := -2 }

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def lies_on_y_axis (p : Point) : Prop :=
  p.x = 0

/-- Two points form a line parallel to the x-axis if they have the same y-coordinate -/
def parallel_to_x_axis (p1 p2 : Point) : Prop :=
  p1.y = p2.y

theorem point_P_coordinates :
  (∃ m : ℝ, lies_on_y_axis (P m) ∧ P m = { x := 0, y := 3 }) ∧
  (∃ m : ℝ, parallel_to_x_axis (P m) A ∧ P m = { x := -15, y := -2 }) :=
sorry

end point_P_coordinates_l16_1676


namespace perpendicular_line_proof_l16_1698

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Theorem statement
theorem perpendicular_line_proof :
  -- The perpendicular line passes through point P
  perpendicular_line point_P.1 point_P.2 ∧
  -- The perpendicular line is indeed perpendicular to the given line
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    given_line x₁ y₁ → given_line x₂ y₂ →
    perpendicular_line x₁ y₁ → perpendicular_line x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (1) + (y₂ - y₁) * (-2)) * ((x₂ - x₁) * (2) + (y₂ - y₁) * (1)) = 0 :=
by sorry

end perpendicular_line_proof_l16_1698


namespace arithmetic_progression_first_term_l16_1682

def is_arithmetic_progression (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def is_increasing (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (List.range n).map a |>.sum

theorem arithmetic_progression_first_term (a : ℕ → ℤ) :
  is_arithmetic_progression a →
  is_increasing a →
  let S := sum_first_n_terms a 10
  (a 6 * a 12 > S + 1) →
  (a 7 * a 11 < S + 17) →
  a 1 ∈ ({-6, -5, -4, -2, -1, 0} : Set ℤ) := by
  sorry

end arithmetic_progression_first_term_l16_1682


namespace denise_spending_l16_1672

/-- Represents the types of dishes available --/
inductive Dish
| Simple
| Meat
| Fish

/-- Represents the types of vitamins available --/
inductive Vitamin
| Milk
| Fruit
| Special

/-- Returns the price of a dish --/
def dishPrice (d : Dish) : ℕ :=
  match d with
  | Dish.Simple => 7
  | Dish.Meat => 11
  | Dish.Fish => 14

/-- Returns the price of a vitamin --/
def vitaminPrice (v : Vitamin) : ℕ :=
  match v with
  | Vitamin.Milk => 6
  | Vitamin.Fruit => 7
  | Vitamin.Special => 9

/-- Calculates the total price of a meal (dish + vitamin) --/
def mealPrice (d : Dish) (v : Vitamin) : ℕ :=
  dishPrice d + vitaminPrice v

/-- Represents a person's meal choice --/
structure MealChoice where
  dish : Dish
  vitamin : Vitamin

/-- The main theorem to prove --/
theorem denise_spending (julio_choice denise_choice : MealChoice)
  (h : mealPrice julio_choice.dish julio_choice.vitamin = 
       mealPrice denise_choice.dish denise_choice.vitamin + 6) :
  mealPrice denise_choice.dish denise_choice.vitamin = 14 ∨
  mealPrice denise_choice.dish denise_choice.vitamin = 17 := by
  sorry


end denise_spending_l16_1672


namespace equation_represents_hyperbola_l16_1670

/-- Given the equation (x+y)^2 = x^2 + y^2 + 2x + 2y, prove it represents a hyperbola -/
theorem equation_represents_hyperbola (x y : ℝ) :
  (x + y)^2 = x^2 + y^2 + 2*x + 2*y ↔ (x - 1) * (y - 1) = 1 :=
by sorry

end equation_represents_hyperbola_l16_1670


namespace simplify_expressions_l16_1616

theorem simplify_expressions :
  ((-2.48 + 4.33 + (-7.52) + (-4.33)) = -10) ∧
  ((7/13 * (-9) + 7/13 * (-18) + 7/13) = -14) ∧
  (-20 * (1/19) * 38 = -762) := by
  sorry

end simplify_expressions_l16_1616


namespace marble_probability_difference_l16_1659

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1500

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1500

/-- The number of white marbles in the box -/
def white_marbles : ℕ := 1

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles + white_marbles

/-- The probability of drawing two marbles of the same color (including white pairings) -/
def Ps : ℚ := (red_marbles * (red_marbles - 1) + black_marbles * (black_marbles - 1) + 2 * (red_marbles + black_marbles) * white_marbles) / (total_marbles * (total_marbles - 1))

/-- The probability of drawing two marbles of different colors (excluding white pairings) -/
def Pd : ℚ := (2 * red_marbles * black_marbles) / (total_marbles * (total_marbles - 1))

/-- The theorem stating that the absolute difference between Ps and Pd is 1/3 -/
theorem marble_probability_difference : |Ps - Pd| = 1 / 3 := by
  sorry

end marble_probability_difference_l16_1659


namespace systematic_sampling_theorem_l16_1618

/-- Systematic sampling function -/
def systematicSample (totalEmployees : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : ℕ → ℕ :=
  fun n => (n - 1) * (totalEmployees / sampleSize) + firstSample

/-- Theorem: In a systematic sampling of 40 samples from 200 employees, 
    if the 5th sample is 22, then the 10th sample is 47 -/
theorem systematic_sampling_theorem 
  (totalEmployees : ℕ) (sampleSize : ℕ) (groupSize : ℕ) (fifthSample : ℕ) :
  totalEmployees = 200 →
  sampleSize = 40 →
  groupSize = 5 →
  fifthSample = 22 →
  systematicSample totalEmployees sampleSize (fifthSample - (5 - 1) * groupSize) 10 = 47 := by
  sorry

#check systematic_sampling_theorem

end systematic_sampling_theorem_l16_1618


namespace geometric_sequence_sum_l16_1607

/-- 
For a geometric sequence {a_n}, if a_2 + a_4 = 2, 
then a_1a_3 + 2a_2a_4 + a_3a_5 = 4
-/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_sum : a 2 + a 4 = 2) : 
  a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 4 := by
  sorry

end geometric_sequence_sum_l16_1607


namespace souvenir_optimal_price_l16_1652

/-- Represents the optimization problem for a souvenir's selling price --/
def SouvenirOptimization (a : ℝ) : Prop :=
  ∃ (x : ℝ),
    0 < x ∧ x < 1 ∧
    (∀ (z : ℝ), 0 < z ∧ z < 1 →
      5 * a * (1 + 4 * x - x^2 - 4 * x^3) ≥ 5 * a * (1 + 4 * z - z^2 - 4 * z^3)) ∧
    20 * (1 + x) = 30

theorem souvenir_optimal_price (a : ℝ) (h : a > 0) : SouvenirOptimization a := by
  sorry

end souvenir_optimal_price_l16_1652


namespace candy_bar_problem_l16_1634

theorem candy_bar_problem (F : ℕ) : 
  (∃ (J : ℕ), 
    J = 10 * (2 * F + 6) ∧ 
    (2 * F + 6) = F + (F + 6) ∧
    (40 * J) / 100 = 120) → 
  F = 12 := by
sorry

end candy_bar_problem_l16_1634


namespace min_value_of_f_l16_1627

/-- The quadratic function f(x) = 8x^2 - 32x + 2023 -/
def f (x : ℝ) : ℝ := 8 * x^2 - 32 * x + 2023

/-- Theorem stating that the minimum value of f(x) is 1991 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = 1991 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end min_value_of_f_l16_1627


namespace all_graphs_different_l16_1638

-- Define the three equations
def eq_I (x y : ℝ) : Prop := y = x - 3
def eq_II (x y : ℝ) : Prop := y = (x^2 - 9) / (x + 3)
def eq_III (x y : ℝ) : Prop := (x + 3) * y = x^2 - 9

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq1 x y ↔ eq2 x y

-- Theorem stating that all graphs are different
theorem all_graphs_different :
  ¬(same_graph eq_I eq_II) ∧ 
  ¬(same_graph eq_I eq_III) ∧ 
  ¬(same_graph eq_II eq_III) :=
sorry

end all_graphs_different_l16_1638


namespace investment_calculation_l16_1650

theorem investment_calculation (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) (total_dividend : ℝ) :
  face_value = 100 →
  premium_rate = 0.2 →
  dividend_rate = 0.07 →
  total_dividend = 840.0000000000001 →
  (total_dividend / (face_value * dividend_rate)) * (face_value * (1 + premium_rate)) = 14400 :=
by sorry

end investment_calculation_l16_1650


namespace unique_solution_l16_1624

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ 
  (∃ (a b c d e : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e)

theorem unique_solution :
  ∃! (n : ℕ), is_valid_number n ∧ n * 3 = 100000 * n + n + 1 :=
by sorry

end unique_solution_l16_1624


namespace min_socks_for_eight_pairs_l16_1663

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (yellow : ℕ)
  (green : ℕ)
  (purple : ℕ)

/-- The minimum number of socks needed to guarantee at least n pairs -/
def minSocksForPairs (drawer : SockDrawer) (n : ℕ) : ℕ :=
  sorry

/-- The specific drawer configuration in the problem -/
def problemDrawer : SockDrawer :=
  { red := 50, yellow := 100, green := 70, purple := 30 }

theorem min_socks_for_eight_pairs :
  minSocksForPairs problemDrawer 8 = 28 :=
sorry

end min_socks_for_eight_pairs_l16_1663


namespace seconds_in_week_scientific_correct_l16_1687

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number of seconds in a week -/
def seconds_in_week : ℕ := 604800

/-- The scientific notation representation of the number of seconds in a week -/
def seconds_in_week_scientific : ScientificNotation :=
  { coefficient := 6.048
    exponent := 5
    is_valid := by sorry }

/-- Theorem stating that the scientific notation representation is correct -/
theorem seconds_in_week_scientific_correct :
  (seconds_in_week_scientific.coefficient * (10 : ℝ) ^ seconds_in_week_scientific.exponent) = seconds_in_week := by
  sorry

end seconds_in_week_scientific_correct_l16_1687


namespace arcsin_arccos_equation_solution_l16_1631

theorem arcsin_arccos_equation_solution :
  ∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧
  Real.arcsin x + Real.arcsin (2*x) = Real.arccos x + Real.arccos (2*x) ∧
  x = Real.sqrt 5 / 5 := by
sorry

end arcsin_arccos_equation_solution_l16_1631


namespace fifteen_point_figures_l16_1619

def points : ℕ := 15

-- Define a function to calculate combinations
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Number of quadrilaterals
def quadrilaterals : ℕ := choose points 4

-- Number of triangles
def triangles : ℕ := choose points 3

-- Total number of figures
def total_figures : ℕ := quadrilaterals + triangles

-- Theorem statement
theorem fifteen_point_figures : total_figures = 1820 := by
  sorry

end fifteen_point_figures_l16_1619


namespace log_equation_sum_l16_1678

theorem log_equation_sum (A B C : ℕ+) : 
  (Nat.gcd A.val (Nat.gcd B.val C.val) = 1) →
  (A : ℝ) * (Real.log 5 / Real.log 100) + (B : ℝ) * (Real.log 2 / Real.log 100) = C →
  A + B + C = 5 := by
  sorry

end log_equation_sum_l16_1678


namespace arithmetic_progression_ratio_l16_1646

/-- The sum of the first n terms of an arithmetic progression -/
def arithmeticSum (a d : ℚ) (n : ℕ) : ℚ := n / 2 * (2 * a + (n - 1) * d)

/-- Theorem: In an arithmetic progression where the sum of the first 15 terms
    is three times the sum of the first 8 terms, the ratio of the first term
    to the common difference is 7:3 -/
theorem arithmetic_progression_ratio (a d : ℚ) :
  arithmeticSum a d 15 = 3 * arithmeticSum a d 8 → a / d = 7 / 3 := by
  sorry

end arithmetic_progression_ratio_l16_1646


namespace smallest_abs_z_l16_1666

theorem smallest_abs_z (z : ℂ) (h : Complex.abs (z - 15) + Complex.abs (z + 3*I) = 20) :
  ∃ (min_abs : ℝ), min_abs = 2.25 ∧ ∀ w : ℂ, Complex.abs (w - 15) + Complex.abs (w + 3*I) = 20 → Complex.abs w ≥ min_abs :=
sorry

end smallest_abs_z_l16_1666


namespace car_trip_mpg_l16_1684

/-- Represents the miles per gallon for a car trip -/
structure MPG where
  ab : ℝ  -- Miles per gallon from A to B
  bc : ℝ  -- Miles per gallon from B to C
  total : ℝ  -- Overall miles per gallon for the entire trip

/-- Represents the distance for a car trip -/
structure Distance where
  ab : ℝ  -- Distance from A to B
  bc : ℝ  -- Distance from B to C

theorem car_trip_mpg (d : Distance) (mpg : MPG) :
  d.bc = d.ab / 2 →  -- Distance from B to C is half of A to B
  mpg.ab = 40 →  -- MPG from A to B is 40
  mpg.total = 300 / 7 →  -- Overall MPG is 300/7 (approx. 42.857142857142854)
  d.ab > 0 →  -- Distance from A to B is positive
  mpg.bc = 100 / 9 :=  -- MPG from B to C is 100/9 (approx. 11.11)
by sorry

end car_trip_mpg_l16_1684


namespace nine_sevenths_to_fourth_l16_1604

theorem nine_sevenths_to_fourth (x : ℚ) : x = 9 * (1 / 7)^4 → x = 9 / 2401 := by
  sorry

end nine_sevenths_to_fourth_l16_1604


namespace solve_for_y_l16_1655

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end solve_for_y_l16_1655


namespace triangle_sum_property_l16_1669

theorem triangle_sum_property : ∃ (a b c d e f : ℤ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a = b + c + d ∧
  b = a + c + e ∧
  c = a + b + f :=
by sorry

end triangle_sum_property_l16_1669


namespace square_mod_nine_not_five_l16_1697

theorem square_mod_nine_not_five (n : ℤ) : n^2 % 9 ≠ 5 := by
  sorry

end square_mod_nine_not_five_l16_1697


namespace inverse_true_converse_false_l16_1614

-- Define the universe of shapes
variable (Shape : Type)

-- Define predicates for being a circle and having corners
variable (is_circle : Shape → Prop)
variable (has_corners : Shape → Prop)

-- Given statement
axiom circle_no_corners : ∀ s : Shape, is_circle s → ¬(has_corners s)

-- Theorem to prove
theorem inverse_true_converse_false :
  (∀ s : Shape, ¬(is_circle s) → has_corners s) ∧
  ¬(∀ s : Shape, ¬(has_corners s) → is_circle s) :=
sorry

end inverse_true_converse_false_l16_1614


namespace max_students_l16_1660

theorem max_students (n : ℕ) : n < 100 ∧ n % 9 = 4 ∧ n % 7 = 3 → n ≤ 94 := by
  sorry

end max_students_l16_1660


namespace tom_stock_profit_l16_1610

/-- Calculate Tom's overall profit from stock transactions -/
theorem tom_stock_profit : 
  let stock_a_initial_shares : ℕ := 20
  let stock_a_initial_price : ℚ := 3
  let stock_b_initial_shares : ℕ := 30
  let stock_b_initial_price : ℚ := 5
  let stock_c_initial_shares : ℕ := 15
  let stock_c_initial_price : ℚ := 10
  let commission_rate : ℚ := 2 / 100
  let stock_a_sold_shares : ℕ := 10
  let stock_a_sell_price : ℚ := 4
  let stock_b_sold_shares : ℕ := 20
  let stock_b_sell_price : ℚ := 7
  let stock_c_sold_shares : ℕ := 5
  let stock_c_sell_price : ℚ := 12
  let stock_a_value_increase : ℚ := 2
  let stock_b_value_increase : ℚ := 1.2
  let stock_c_value_decrease : ℚ := 0.9

  let initial_cost := (stock_a_initial_shares * stock_a_initial_price + 
                       stock_b_initial_shares * stock_b_initial_price + 
                       stock_c_initial_shares * stock_c_initial_price) * (1 + commission_rate)

  let sales_revenue := (stock_a_sold_shares * stock_a_sell_price + 
                        stock_b_sold_shares * stock_b_sell_price + 
                        stock_c_sold_shares * stock_c_sell_price) * (1 - commission_rate)

  let remaining_value := (stock_a_initial_shares - stock_a_sold_shares) * stock_a_initial_price * stock_a_value_increase + 
                         (stock_b_initial_shares - stock_b_sold_shares) * stock_b_initial_price * stock_b_value_increase + 
                         (stock_c_initial_shares - stock_c_sold_shares) * stock_c_initial_price * stock_c_value_decrease

  let profit := sales_revenue + remaining_value - initial_cost

  profit = 78
  := by sorry

end tom_stock_profit_l16_1610


namespace equation_solutions_l16_1674

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x : ℝ, x^2 - 2*x = 3 ↔ x = -1 ∨ x = 3) := by
sorry

end equation_solutions_l16_1674


namespace unique_solution_l16_1693

/-- A single digit is a natural number from 0 to 9. -/
def SingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- The equation that Θ must satisfy. -/
def SatisfiesEquation (Θ : ℕ) : Prop := 504 * Θ = 40 + Θ + Θ^2

theorem unique_solution :
  ∃! Θ : ℕ, SingleDigit Θ ∧ SatisfiesEquation Θ ∧ Θ = 9 :=
sorry

end unique_solution_l16_1693


namespace cubic_odd_and_increasing_l16_1695

def f (x : ℝ) : ℝ := x^3

theorem cubic_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end cubic_odd_and_increasing_l16_1695


namespace arithmetic_geometric_sequence_l16_1625

/-- Given an arithmetic sequence {a_n} with common difference 3,
    where a_1, a_2, a_5 form a geometric sequence, prove that a_10 = 57/2 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = 3) →  -- arithmetic sequence with common difference 3
  (a 2)^2 = a 1 * a 5 →         -- a_1, a_2, a_5 form a geometric sequence
  a 10 = 57/2 := by
sorry

end arithmetic_geometric_sequence_l16_1625


namespace ellipse_triangle_perimeter_l16_1680

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 169 + y^2 / 144 = 1

-- Define a point on the ellipse
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : is_on_ellipse x y

-- Define the foci of the ellipse
structure Foci where
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  are_foci : ∀ (p : PointOnEllipse), 
    Real.sqrt ((p.x - f1.1)^2 + (p.y - f1.2)^2) + 
    Real.sqrt ((p.x - f2.1)^2 + (p.y - f2.2)^2) = 26

-- The theorem to prove
theorem ellipse_triangle_perimeter 
  (p : PointOnEllipse) (f : Foci) : 
  Real.sqrt ((p.x - f.f1.1)^2 + (p.y - f.f1.2)^2) +
  Real.sqrt ((p.x - f.f2.1)^2 + (p.y - f.f2.2)^2) +
  Real.sqrt ((f.f1.1 - f.f2.1)^2 + (f.f1.2 - f.f2.2)^2) = 36 := by
  sorry

end ellipse_triangle_perimeter_l16_1680


namespace sum_of_coefficients_l16_1600

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
sorry

end sum_of_coefficients_l16_1600


namespace B_subset_A_l16_1609

-- Define set A
def A : Set ℝ := {x : ℝ | |2*x - 3| > 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 + x - 6 > 0}

-- Theorem to prove
theorem B_subset_A : B ⊆ A := by
  sorry

end B_subset_A_l16_1609


namespace irreducibility_of_polynomial_l16_1623

theorem irreducibility_of_polynomial :
  ¬∃ (p q : Polynomial ℤ), (Polynomial.degree p ≥ 1) ∧ (Polynomial.degree q ≥ 1) ∧ (p * q = X^5 + 2*X + 1) :=
by sorry

end irreducibility_of_polynomial_l16_1623


namespace division_problem_l16_1630

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 12 → 
  divisor = 17 → 
  remainder = 7 → 
  dividend = divisor * quotient + remainder →
  quotient = 0 := by
sorry

end division_problem_l16_1630


namespace square_on_circle_radius_l16_1653

theorem square_on_circle_radius (S : ℝ) (x : ℝ) (R : ℝ) : 
  S = 256 →  -- Area of the square
  x^2 = S →  -- Side length of the square
  (x - R)^2 = R^2 - (x/2)^2 →  -- Pythagorean theorem relation
  R = 10 := by
  sorry

end square_on_circle_radius_l16_1653


namespace tangent_circles_radius_l16_1642

/-- Two circles are tangent if the distance between their centers equals the sum of their radii -/
def are_tangent (c1_center c2_center : ℝ × ℝ) (r : ℝ) : Prop :=
  Real.sqrt ((c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2) = 2 * r

theorem tangent_circles_radius (r : ℝ) (h : r > 0) :
  are_tangent (0, 0) (3, -1) r → r = Real.sqrt 10 / 2 := by
  sorry

#check tangent_circles_radius

end tangent_circles_radius_l16_1642


namespace boys_in_class_l16_1679

theorem boys_in_class (total : ℕ) (girls_more : ℕ) (boys : ℕ) 
  (h1 : total = 485) 
  (h2 : girls_more = 69) 
  (h3 : total = boys + (boys + girls_more)) : 
  boys = 208 := by
  sorry

end boys_in_class_l16_1679


namespace only_one_statement_correct_l16_1606

theorem only_one_statement_correct : 
  ¬(∀ (a b : ℤ), a < b → a^2 < b^2) ∧ 
  ¬(∀ (a : ℤ), a^2 > 0) ∧ 
  ¬(∀ (a : ℤ), -a < 0) ∧ 
  (∀ (a b c : ℤ), a * c^2 < b * c^2 → a < b) :=
by sorry

end only_one_statement_correct_l16_1606


namespace intersection_trisection_l16_1605

/-- A line y = mx + b intersecting a circle and a hyperbola -/
structure IntersectingLine where
  m : ℝ
  b : ℝ
  h_m : |m| < 1
  h_b : |b| < 1

/-- Points of intersection with the circle x^2 + y^2 = 1 -/
def circle_intersection (l : IntersectingLine) : Set (ℝ × ℝ) :=
  {(x, y) | y = l.m * x + l.b ∧ x^2 + y^2 = 1}

/-- Points of intersection with the hyperbola x^2 - y^2 = 1 -/
def hyperbola_intersection (l : IntersectingLine) : Set (ℝ × ℝ) :=
  {(x, y) | y = l.m * x + l.b ∧ x^2 - y^2 = 1}

/-- Trisection property of the intersection points -/
def trisects (P Q R S : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ∈ Set.Icc (0 : ℝ) 1 ∧
    P = (1 - t) • R + t • S ∧
    Q = (1 - t) • S + t • R ∧
    t = 1/3 ∨ t = 2/3

/-- Main theorem: Intersection points trisect implies specific values for m and b -/
theorem intersection_trisection (l : IntersectingLine)
  (hP : P ∈ circle_intersection l) (hQ : Q ∈ circle_intersection l)
  (hR : R ∈ hyperbola_intersection l) (hS : S ∈ hyperbola_intersection l)
  (h_trisect : trisects P Q R S) :
  (l.m = 0 ∧ l.b = 2/5 * Real.sqrt 5) ∨
  (l.m = 0 ∧ l.b = -2/5 * Real.sqrt 5) ∨
  (l.m = 2/5 * Real.sqrt 5 ∧ l.b = 0) ∨
  (l.m = -2/5 * Real.sqrt 5 ∧ l.b = 0) :=
sorry

end intersection_trisection_l16_1605


namespace solve_shelves_problem_l16_1641

def shelves_problem (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : Prop :=
  let remaining_books := initial_stock - books_sold
  remaining_books / books_per_shelf = 5

theorem solve_shelves_problem :
  shelves_problem 40 20 4 :=
by
  sorry

end solve_shelves_problem_l16_1641


namespace quadratic_function_properties_l16_1688

/-- Represents a quadratic function y = ax² + bx + 3 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  h : a ≠ 0

/-- The axis of symmetry of the quadratic function is x = 1 -/
def axis_of_symmetry (f : QuadraticFunction) : Prop :=
  -f.b / (2 * f.a) = 1

/-- 3 is a root of the quadratic equation ax² + bx + 3 = 0 -/
def is_root_three (f : QuadraticFunction) : Prop :=
  f.a * 3^2 + f.b * 3 + 3 = 0

/-- The maximum value of the quadratic function is 4 -/
def max_value_is_four (f : QuadraticFunction) : Prop :=
  ∀ x, f.a * x^2 + f.b * x + 3 ≤ 4

/-- When x = 2, y = 5 -/
def y_is_five_at_two (f : QuadraticFunction) : Prop :=
  f.a * 2^2 + f.b * 2 + 3 = 5

/-- The main theorem -/
theorem quadratic_function_properties (f : QuadraticFunction) :
  axis_of_symmetry f → is_root_three f → max_value_is_four f →
  ¬(y_is_five_at_two f) := by sorry

end quadratic_function_properties_l16_1688


namespace plan1_greater_loss_l16_1691

/-- Probability of minor flooding -/
def p_minor : ℝ := 0.2

/-- Probability of major flooding -/
def p_major : ℝ := 0.05

/-- Cost of building a protective wall -/
def wall_cost : ℝ := 4000

/-- Loss due to major flooding -/
def major_flood_loss : ℝ := 30000

/-- Loss due to minor flooding in Plan 2 -/
def minor_flood_loss : ℝ := 15000

/-- Expected loss for Plan 1 -/
def expected_loss_plan1 : ℝ := major_flood_loss * p_major + wall_cost * p_minor + wall_cost

/-- Expected loss for Plan 2 -/
def expected_loss_plan2 : ℝ := major_flood_loss * p_major + minor_flood_loss * p_minor

/-- Theorem stating that the expected loss of Plan 1 is greater than the expected loss of Plan 2 -/
theorem plan1_greater_loss : expected_loss_plan1 > expected_loss_plan2 :=
  sorry

end plan1_greater_loss_l16_1691


namespace triangle_perimeter_l16_1632

-- Define the triangle type
structure Triangle where
  inradius : ℝ
  area : ℝ

-- Theorem statement
theorem triangle_perimeter (t : Triangle) (h1 : t.inradius = 2.5) (h2 : t.area = 75) :
  2 * t.area / t.inradius = 60 := by
  sorry

#check triangle_perimeter

end triangle_perimeter_l16_1632


namespace no_integer_solutions_l16_1644

theorem no_integer_solutions :
  ∀ x : ℤ, x^5 - 31*x + 2015 ≠ 0 := by
  sorry

end no_integer_solutions_l16_1644


namespace tan_alpha_value_l16_1649

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 2) : Real.tan α = 1/3 := by
  sorry

end tan_alpha_value_l16_1649


namespace rectangular_plot_breadth_l16_1671

theorem rectangular_plot_breadth (length breadth : ℝ) 
  (h1 : length * breadth = 15 * breadth) 
  (h2 : length - breadth = 10) : 
  breadth = 5 := by
sorry

end rectangular_plot_breadth_l16_1671


namespace log_inequality_equiv_solution_set_l16_1661

def log_inequality (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x ≠ 1 ∧ x ≠ y ∧ Real.log y / Real.log x ≥ (Real.log x + Real.log y) / (Real.log x - Real.log y)

def solution_set (x y : ℝ) : Prop :=
  (0 < x ∧ x < 1 ∧ 0 < y ∧ y < x) ∨ (x > 1 ∧ y > x)

theorem log_inequality_equiv_solution_set :
  ∀ x y : ℝ, log_inequality x y ↔ solution_set x y :=
sorry

end log_inequality_equiv_solution_set_l16_1661


namespace square_difference_equals_648_l16_1603

theorem square_difference_equals_648 : (36 + 9)^2 - (9^2 + 36^2) = 648 := by
  sorry

end square_difference_equals_648_l16_1603
