import Mathlib

namespace NUMINAMATH_CALUDE_brown_mms_problem_l841_84108

theorem brown_mms_problem (bag1 bag2 bag3 bag4 bag5 : ℕ) 
  (h1 : bag1 = 9)
  (h2 : bag2 = 12)
  (h5 : bag5 = 3)
  (h_avg : (bag1 + bag2 + bag3 + bag4 + bag5) / 5 = 8) :
  bag3 + bag4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_brown_mms_problem_l841_84108


namespace NUMINAMATH_CALUDE_total_people_zoo_and_amusement_park_l841_84113

theorem total_people_zoo_and_amusement_park : 
  let cars_to_zoo : Float := 7.0
  let people_per_car_zoo : Float := 45.0
  let cars_to_amusement_park : Float := 5.0
  let people_per_car_amusement_park : Float := 56.0
  
  cars_to_zoo * people_per_car_zoo + cars_to_amusement_park * people_per_car_amusement_park = 595.0 := by
  sorry

end NUMINAMATH_CALUDE_total_people_zoo_and_amusement_park_l841_84113


namespace NUMINAMATH_CALUDE_staircase_climbing_l841_84156

/-- Number of ways to ascend n steps by jumping 1 or 2 steps at a time -/
def ascend (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n+2 => ascend (n+1) + ascend n

/-- Number of ways to descend n steps with option to skip steps -/
def descend (n : ℕ) : ℕ := 2^(n-1)

theorem staircase_climbing :
  (ascend 10 = 89) ∧ (descend 10 = 512) := by
  sorry


end NUMINAMATH_CALUDE_staircase_climbing_l841_84156


namespace NUMINAMATH_CALUDE_zongzi_pricing_solution_l841_84136

/-- Represents the purchase and sales scenario of zongzi -/
structure ZongziScenario where
  egg_yolk_price : ℝ
  red_bean_price : ℝ
  first_purchase_egg : ℕ
  first_purchase_red : ℕ
  first_purchase_total : ℝ
  second_purchase_egg : ℕ
  second_purchase_red : ℕ
  second_purchase_total : ℝ
  initial_selling_price : ℝ
  initial_daily_sales : ℕ
  price_reduction : ℝ
  sales_increase : ℕ
  target_daily_profit : ℝ

/-- Theorem stating the solution to the zongzi pricing problem -/
theorem zongzi_pricing_solution (s : ZongziScenario)
  (h1 : s.first_purchase_egg * s.egg_yolk_price + s.first_purchase_red * s.red_bean_price = s.first_purchase_total)
  (h2 : s.second_purchase_egg * s.egg_yolk_price + s.second_purchase_red * s.red_bean_price = s.second_purchase_total)
  (h3 : s.first_purchase_egg = 60 ∧ s.first_purchase_red = 90 ∧ s.first_purchase_total = 4800)
  (h4 : s.second_purchase_egg = 40 ∧ s.second_purchase_red = 80 ∧ s.second_purchase_total = 3600)
  (h5 : s.initial_selling_price = 70 ∧ s.initial_daily_sales = 20)
  (h6 : s.price_reduction = 1 ∧ s.sales_increase = 5)
  (h7 : s.target_daily_profit = 220) :
  s.egg_yolk_price = 50 ∧ s.red_bean_price = 20 ∧
  ∃ (selling_price : ℝ),
    selling_price = 52 ∧
    (selling_price - s.egg_yolk_price) * (s.initial_daily_sales + s.sales_increase * (s.initial_selling_price - selling_price)) = s.target_daily_profit :=
by sorry

end NUMINAMATH_CALUDE_zongzi_pricing_solution_l841_84136


namespace NUMINAMATH_CALUDE_lottery_profit_lilys_profit_l841_84145

/-- Calculates the profit from selling lottery tickets -/
theorem lottery_profit (n : ℕ) (first_price : ℕ) (prize : ℕ) : ℕ :=
  let total_revenue := n * (2 * first_price + (n - 1)) / 2
  total_revenue - prize

/-- Proves that Lily's profit is $4 given the specified conditions -/
theorem lilys_profit :
  lottery_profit 5 1 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_lottery_profit_lilys_profit_l841_84145


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_equality_l841_84134

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (1 / x + 1 / y) ≥ 3 / 2 + Real.sqrt 2 :=
by sorry

theorem min_reciprocal_sum_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (1 / x + 1 / y = 3 / 2 + Real.sqrt 2) ↔ (x = 2 / 3 ∧ y = 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_equality_l841_84134


namespace NUMINAMATH_CALUDE_solve_employee_pay_l841_84131

def employee_pay_problem (pay_B : ℝ) (percent_A : ℝ) : Prop :=
  let pay_A : ℝ := percent_A * pay_B
  let total_pay : ℝ := pay_A + pay_B
  pay_B = 228 ∧ percent_A = 1.5 → total_pay = 570

theorem solve_employee_pay : employee_pay_problem 228 1.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_employee_pay_l841_84131


namespace NUMINAMATH_CALUDE_discount_store_purchase_l841_84162

/-- Represents the number of items of each type bought by one person -/
structure ItemCounts where
  typeA : ℕ
  typeB : ℕ

/-- Represents the prices of items -/
structure Prices where
  typeA : ℕ
  typeB : ℕ

def total_spent (counts : ItemCounts) (prices : Prices) : ℕ :=
  counts.typeA * prices.typeA + counts.typeB * prices.typeB

theorem discount_store_purchase : ∃ (counts : ItemCounts),
  let prices : Prices := ⟨8, 9⟩
  total_spent counts prices = 172 ∧
  counts.typeA + counts.typeB = counts.typeA + counts.typeB ∧
  counts.typeA = 4 ∧
  counts.typeB = 6 := by
  sorry

end NUMINAMATH_CALUDE_discount_store_purchase_l841_84162


namespace NUMINAMATH_CALUDE_power_mod_nineteen_l841_84165

theorem power_mod_nineteen : 11^2048 ≡ 16 [MOD 19] := by
  sorry

end NUMINAMATH_CALUDE_power_mod_nineteen_l841_84165


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_isosceles_triangle_l841_84155

/-- The radius of the inscribed circle in an isosceles triangle -/
theorem inscribed_circle_radius_isosceles_triangle (DE DF EF : ℝ) (h1 : DE = 8) (h2 : DF = 8) (h3 : EF = 10) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  r = 5 * Real.sqrt 39 / 13 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_isosceles_triangle_l841_84155


namespace NUMINAMATH_CALUDE_ellipse_k_values_l841_84169

-- Define the eccentricity
def eccentricity : ℝ := 6

-- Define the ellipse equation
def is_on_ellipse (x y k : ℝ) : Prop :=
  x^2 / 20 + y^2 / k = 1

-- Define the relationship between eccentricity, semi-major axis, and semi-minor axis
def eccentricity_relation (a b : ℝ) : Prop :=
  eccentricity^2 = 1 - (b^2 / a^2)

-- Theorem statement
theorem ellipse_k_values :
  ∃ k : ℝ, (k = 11 ∨ k = 29) ∧
  (∀ x y : ℝ, is_on_ellipse x y k) ∧
  ((eccentricity_relation 20 k ∧ 20 > k) ∨ (eccentricity_relation k 20 ∧ k > 20)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_values_l841_84169


namespace NUMINAMATH_CALUDE_sum_of_three_odd_implies_one_odd_l841_84122

theorem sum_of_three_odd_implies_one_odd (a b c : ℤ) : 
  Odd (a + b + c) → Odd a ∨ Odd b ∨ Odd c := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_odd_implies_one_odd_l841_84122


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l841_84125

/-- The area of a square with adjacent vertices at (0,3) and (4,0) is 25. -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (0, 3)
  let p2 : ℝ × ℝ := (4, 0)
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  d^2 = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l841_84125


namespace NUMINAMATH_CALUDE_square_of_six_y_minus_four_l841_84198

theorem square_of_six_y_minus_four (y : ℝ) (h : 3 * y^2 + 6 = 5 * y + 15) : 
  (6 * y - 4)^2 = 134 := by
  sorry

end NUMINAMATH_CALUDE_square_of_six_y_minus_four_l841_84198


namespace NUMINAMATH_CALUDE_sequences_properties_l841_84178

def sequence1 (n : ℕ) : ℤ := (-3)^n
def sequence2 (n : ℕ) : ℤ := -2 * (-3)^n
def sequence3 (n : ℕ) : ℤ := (-3)^n + 2

theorem sequences_properties :
  (∃ k : ℕ, sequence2 k + sequence2 (k+1) + sequence2 (k+2) = 378) ∧
  (sequence1 2024 + sequence2 2024 + sequence3 2024 = 2) := by
  sorry

end NUMINAMATH_CALUDE_sequences_properties_l841_84178


namespace NUMINAMATH_CALUDE_maria_water_bottles_l841_84189

theorem maria_water_bottles (initial_bottles : ℝ) (sister_drank : ℝ) (bottles_left : ℝ) :
  initial_bottles = 45.0 →
  sister_drank = 8.0 →
  bottles_left = 23 →
  initial_bottles - sister_drank - bottles_left = 14.0 :=
by
  sorry

end NUMINAMATH_CALUDE_maria_water_bottles_l841_84189


namespace NUMINAMATH_CALUDE_ice_cream_consumption_l841_84167

/-- The total amount of ice cream consumed over five days -/
def total_ice_cream (friday saturday sunday monday : ℝ) : ℝ :=
  friday + saturday + sunday + monday + 2 * monday

/-- Theorem stating that the total ice cream consumption is 9 pints -/
theorem ice_cream_consumption : 
  total_ice_cream 3.25 2.5 1.75 0.5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_consumption_l841_84167


namespace NUMINAMATH_CALUDE_additional_money_needed_l841_84158

/-- The amount of money Michael has initially -/
def michaels_money : ℕ := 50

/-- The cost of the cake -/
def cake_cost : ℕ := 20

/-- The cost of the bouquet -/
def bouquet_cost : ℕ := 36

/-- The cost of the balloons -/
def balloon_cost : ℕ := 5

/-- The total cost of all items -/
def total_cost : ℕ := cake_cost + bouquet_cost + balloon_cost

/-- The theorem stating how much more money Michael needs -/
theorem additional_money_needed : total_cost - michaels_money = 11 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l841_84158


namespace NUMINAMATH_CALUDE_multiply_polynomials_l841_84130

theorem multiply_polynomials (x : ℝ) : 
  (x^4 + 10*x^2 + 25) * (x^2 - 25) = x^4 + 10*x^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomials_l841_84130


namespace NUMINAMATH_CALUDE_ellipse_conditions_l841_84102

-- Define what it means for an equation to represent an ellipse
def represents_ellipse (a b : ℝ) : Prop :=
  ∃ (h k : ℝ) (A B : ℝ), A ≠ B ∧ A > 0 ∧ B > 0 ∧
    ∀ (x y : ℝ), a * (x - h)^2 + b * (y - k)^2 = 1 ↔ 
      ((x - h)^2 / A^2) + ((y - k)^2 / B^2) = 1

-- State the theorem
theorem ellipse_conditions (a b : ℝ) :
  (a > 0 ∧ b > 0 ∧ represents_ellipse a b) ∧
  ¬(a > 0 ∧ b > 0 → represents_ellipse a b) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_conditions_l841_84102


namespace NUMINAMATH_CALUDE_solve_for_m_l841_84124

theorem solve_for_m (Q t h m : ℝ) (hQ : Q > 0) (ht : t > 0) (hh : h ≥ 0) :
  Q = t / (1 + Real.sqrt h)^m ↔ m = Real.log (t / Q) / Real.log (1 + Real.sqrt h) :=
sorry

end NUMINAMATH_CALUDE_solve_for_m_l841_84124


namespace NUMINAMATH_CALUDE_f_properties_l841_84140

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |2/x - a*x + 5|

theorem f_properties :
  ∀ a : ℝ,
  (∃ x : ℝ, f a x = 0) ∧
  (a = 3 → ∀ x y : ℝ, x < y → y < -1 → f a x > f a y) ∧
  (a > 0 → ∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 2 ∧ f a x₀ = 8/3 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≤ 8/3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l841_84140


namespace NUMINAMATH_CALUDE_ralphs_purchase_cost_l841_84150

/-- Calculates the final cost of Ralph's purchase given the initial conditions --/
theorem ralphs_purchase_cost
  (initial_total : ℝ)
  (discounted_item_price : ℝ)
  (item_discount_rate : ℝ)
  (total_discount_rate : ℝ)
  (h1 : initial_total = 54)
  (h2 : discounted_item_price = 20)
  (h3 : item_discount_rate = 0.2)
  (h4 : total_discount_rate = 0.1)
  : ∃ (final_cost : ℝ), final_cost = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_ralphs_purchase_cost_l841_84150


namespace NUMINAMATH_CALUDE_inverse_power_of_two_l841_84175

theorem inverse_power_of_two : 2⁻¹ = (1 : ℚ) / 2 := by sorry

end NUMINAMATH_CALUDE_inverse_power_of_two_l841_84175


namespace NUMINAMATH_CALUDE_propane_tank_burner_cost_is_14_l841_84143

def propane_tank_burner_cost (total_money sheet_cost rope_cost helium_cost_per_oz flight_height_per_oz max_height : ℚ) : ℚ :=
  let remaining_money := total_money - sheet_cost - rope_cost
  let helium_oz_needed := max_height / flight_height_per_oz
  let helium_cost := helium_oz_needed * helium_cost_per_oz
  remaining_money - helium_cost

theorem propane_tank_burner_cost_is_14 :
  propane_tank_burner_cost 200 42 18 1.5 113 9492 = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_propane_tank_burner_cost_is_14_l841_84143


namespace NUMINAMATH_CALUDE_power_two_gt_square_plus_one_l841_84144

theorem power_two_gt_square_plus_one (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_gt_square_plus_one_l841_84144


namespace NUMINAMATH_CALUDE_minimum_students_in_class_l841_84120

theorem minimum_students_in_class (b g : ℕ) : 
  (b ≠ 0 ∧ g ≠ 0) →  -- Ensure non-zero numbers of boys and girls
  (2 * (b / 2) = 3 * (g / 3)) →  -- Half of boys equals two-thirds of girls who passed
  (b / 2 = 2 * (g / 3)) →  -- Boys who failed is twice girls who failed
  7 ≤ b + g  -- The total number of students is at least 7
  ∧ ∃ (b' g' : ℕ), b' ≠ 0 ∧ g' ≠ 0 
     ∧ (2 * (b' / 2) = 3 * (g' / 3))
     ∧ (b' / 2 = 2 * (g' / 3))
     ∧ b' + g' = 7  -- There exists a solution with exactly 7 students
  := by sorry

end NUMINAMATH_CALUDE_minimum_students_in_class_l841_84120


namespace NUMINAMATH_CALUDE_solve_linear_equation_l841_84123

theorem solve_linear_equation :
  ∃ x : ℚ, 4 * (2 * x - 1) = 1 - 3 * (x + 2) ∧ x = -1/11 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l841_84123


namespace NUMINAMATH_CALUDE_log_equation_solution_l841_84132

theorem log_equation_solution (k x : ℝ) (h : k > 0) (h' : x > 0) :
  Real.log x / Real.log k * Real.log k / Real.log 5 = 3 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l841_84132


namespace NUMINAMATH_CALUDE_sally_total_spent_l841_84141

/-- The total amount Sally spent on peaches and cherries -/
def total_spent (peach_price_after_coupon : ℚ) (cherry_price : ℚ) : ℚ :=
  peach_price_after_coupon + cherry_price

/-- Theorem stating that Sally spent $23.86 in total -/
theorem sally_total_spent : 
  total_spent 12.32 11.54 = 23.86 := by
  sorry

end NUMINAMATH_CALUDE_sally_total_spent_l841_84141


namespace NUMINAMATH_CALUDE_rounding_approximation_less_than_exact_l841_84160

theorem rounding_approximation_less_than_exact (x y z : ℕ+) :
  (↑(Int.floor (x : ℝ)) / ↑(Int.ceil (y : ℝ)) : ℝ) - ↑(Int.ceil (z : ℝ)) < (x : ℝ) / (y : ℝ) - (z : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_rounding_approximation_less_than_exact_l841_84160


namespace NUMINAMATH_CALUDE_recycling_theorem_l841_84151

def recycle (n : ℕ) : ℕ :=
  if n < 5 then 0 else n / 5 + recycle (n / 5)

theorem recycling_theorem :
  recycle 3125 = 781 :=
by
  sorry

end NUMINAMATH_CALUDE_recycling_theorem_l841_84151


namespace NUMINAMATH_CALUDE_positive_integer_pairs_l841_84104

theorem positive_integer_pairs (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (∃ k : ℤ, (a^3 * b - 1 : ℤ) = k * (a + 1)) ∧
  (∃ m : ℤ, (b^3 * a + 1 : ℤ) = m * (b - 1)) →
  ((a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_pairs_l841_84104


namespace NUMINAMATH_CALUDE_reflect_point_across_x_axis_l841_84121

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

theorem reflect_point_across_x_axis :
  let P : Point := ⟨2, -3⟩
  let P' : Point := reflectAcrossXAxis P
  P'.x = 2 ∧ P'.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_reflect_point_across_x_axis_l841_84121


namespace NUMINAMATH_CALUDE_more_mashed_potatoes_than_bacon_l841_84149

theorem more_mashed_potatoes_than_bacon (mashed_potatoes bacon : ℕ) 
  (h1 : mashed_potatoes = 457) 
  (h2 : bacon = 394) : 
  mashed_potatoes - bacon = 63 := by
sorry

end NUMINAMATH_CALUDE_more_mashed_potatoes_than_bacon_l841_84149


namespace NUMINAMATH_CALUDE_three_numbers_sum_l841_84157

theorem three_numbers_sum (x y z M : ℚ) : 
  x + y + z = 48 ∧ 
  x - 5 = M ∧ 
  y + 9 = M ∧ 
  z / 5 = M → 
  M = 52 / 7 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l841_84157


namespace NUMINAMATH_CALUDE_will_hero_count_l841_84106

/-- Represents the number of heroes drawn on a sheet of paper -/
structure HeroCount where
  front : Nat
  back : Nat
  third : Nat

/-- Calculates the total number of heroes drawn -/
def totalHeroes (h : HeroCount) : Nat :=
  h.front + h.back + h.third

/-- Theorem: Given the specific hero counts, the total is 19 -/
theorem will_hero_count :
  ∃ (h : HeroCount), h.front = 4 ∧ h.back = 9 ∧ h.third = 6 ∧ totalHeroes h = 19 :=
by sorry

end NUMINAMATH_CALUDE_will_hero_count_l841_84106


namespace NUMINAMATH_CALUDE_function_property_l841_84116

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) + x = x * f y + f x)
  (h2 : f (-1) = 9) : 
  f (-500) = 1007 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l841_84116


namespace NUMINAMATH_CALUDE_kite_AC_length_l841_84115

-- Define the kite ABCD
structure Kite :=
  (A B C D : ℝ × ℝ)
  (diagonals_perpendicular : (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0)
  (BD_length : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 10)
  (AB_equals_BC : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 
                  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2))
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13)
  (AD_equals_DC : Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 
                  Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2))

-- Theorem statement
theorem kite_AC_length (k : Kite) : 
  Real.sqrt ((k.A.1 - k.C.1)^2 + (k.A.2 - k.C.2)^2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_kite_AC_length_l841_84115


namespace NUMINAMATH_CALUDE_some_number_value_l841_84195

theorem some_number_value (x : ℝ) (h : (55 + 113 / x) * x = 4403) : x = 78 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l841_84195


namespace NUMINAMATH_CALUDE_abc_inequality_l841_84172

theorem abc_inequality (a b c : ℝ) 
  (ha : a = 2 * Real.sqrt 7)
  (hb : b = 3 * Real.sqrt 5)
  (hc : c = 5 * Real.sqrt 2) : 
  c > b ∧ b > a :=
sorry

end NUMINAMATH_CALUDE_abc_inequality_l841_84172


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l841_84119

theorem divisibility_implies_equality (a b : ℕ+) 
  (h : ∀ n : ℕ+, (a.val^n.val + n.val) ∣ (b.val^n.val + n.val)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l841_84119


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l841_84147

/-- A line in the 2D plane represented by its slope-intercept form y = mx + b -/
structure Line where
  slope : ℚ
  intercept : ℚ

def Line.through_point (l : Line) (x y : ℚ) : Prop :=
  y = l.slope * x + l.intercept

def Line.parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem parallel_line_through_point (given_line target_line : Line) 
    (h_parallel : given_line.parallel target_line)
    (h_through_point : target_line.through_point 3 0) :
  target_line = Line.mk (1/2) (-3/2) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l841_84147


namespace NUMINAMATH_CALUDE_schedule_arrangements_proof_l841_84103

/-- Represents the number of periods in the morning -/
def morning_periods : Nat := 4

/-- Represents the number of periods in the afternoon -/
def afternoon_periods : Nat := 3

/-- Represents the total number of periods in a day -/
def total_periods : Nat := morning_periods + afternoon_periods

/-- Represents the number of subjects that require two consecutive periods -/
def two_period_subjects : Nat := 2

/-- Represents the number of subjects that require one period -/
def one_period_subjects : Nat := 4

/-- Represents the total number of subjects to be scheduled -/
def total_subjects : Nat := two_period_subjects + one_period_subjects

/-- Calculates the number of possible arrangements for the schedule -/
def schedule_arrangements : Nat := 336

theorem schedule_arrangements_proof :
  schedule_arrangements = 336 := by sorry

end NUMINAMATH_CALUDE_schedule_arrangements_proof_l841_84103


namespace NUMINAMATH_CALUDE_bunnies_given_away_l841_84176

theorem bunnies_given_away (initial_bunnies : ℕ) (kittens_per_bunny : ℕ) (final_total : ℕ) :
  initial_bunnies = 30 →
  kittens_per_bunny = 2 →
  final_total = 54 →
  (initial_bunnies - (final_total - initial_bunnies) / kittens_per_bunny) / initial_bunnies = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_bunnies_given_away_l841_84176


namespace NUMINAMATH_CALUDE_binomial_18_10_l841_84109

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 32318 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l841_84109


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l841_84174

/-- The line y = a is tangent to the circle x^2 + y^2 - 2y = 0 if and only if a = 0 or a = 2 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, y = a → x^2 + y^2 - 2*y = 0 → (x = 0 ∧ (y = a + 1 ∨ y = a - 1))) ↔ (a = 0 ∨ a = 2) := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l841_84174


namespace NUMINAMATH_CALUDE_sum_of_divisors_450_has_three_prime_factors_l841_84173

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_has_three_prime_factors : 
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_450_has_three_prime_factors_l841_84173


namespace NUMINAMATH_CALUDE_distance_on_line_l841_84163

/-- The distance between two points (p, q) and (r, s) on the line y = 2x + 3, where s = 2r + 6 -/
theorem distance_on_line (p r : ℝ) : 
  let q := 2 * p + 3
  let s := 2 * r + 6
  Real.sqrt ((r - p)^2 + (s - q)^2) = Real.sqrt (5 * (r - p)^2 + 12 * (r - p) + 9) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l841_84163


namespace NUMINAMATH_CALUDE_a_range_l841_84188

-- Define the statements p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Theorem statement
theorem a_range (a : ℝ) : 
  (a > 0) → 
  (∀ x, q x → p x a) → 
  (∃ x, p x a ∧ ¬q x) → 
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l841_84188


namespace NUMINAMATH_CALUDE_marias_gum_count_l841_84139

/-- 
Given:
- Maria initially had 25 pieces of gum
- Tommy gave her 16 more pieces
- Luis gave her 20 more pieces

Prove that Maria now has 61 pieces of gum
-/
theorem marias_gum_count (initial : ℕ) (tommy : ℕ) (luis : ℕ) 
  (h1 : initial = 25)
  (h2 : tommy = 16)
  (h3 : luis = 20) :
  initial + tommy + luis = 61 := by
  sorry

end NUMINAMATH_CALUDE_marias_gum_count_l841_84139


namespace NUMINAMATH_CALUDE_min_value_product_l841_84168

theorem min_value_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 9/b = 6) :
  ∀ x y, x > 0 → y > 0 → 1/x + 9/y = 6 → (a + 1) * (b + 9) ≤ (x + 1) * (y + 9) ∧
  ∃ a b, a > 0 ∧ b > 0 ∧ 1/a + 9/b = 6 ∧ (a + 1) * (b + 9) = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l841_84168


namespace NUMINAMATH_CALUDE_expression_value_l841_84128

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : abs m = 3)  -- |m| = 3
  : m + c * d - (a + b) / (m^2) = 4 ∨ m + c * d - (a + b) / (m^2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l841_84128


namespace NUMINAMATH_CALUDE_work_completion_days_l841_84166

/-- Calculates the number of days needed for the remaining workers to complete a job -/
def daysToComplete (originalWorkers : ℕ) (plannedDays : ℕ) (absentWorkers : ℕ) : ℕ :=
  (originalWorkers * plannedDays) / (originalWorkers - absentWorkers)

/-- Proves that given the original conditions, the remaining workers complete the job in 21 days -/
theorem work_completion_days :
  daysToComplete 42 17 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_days_l841_84166


namespace NUMINAMATH_CALUDE_quadratic_point_value_l841_84194

/-- If the point (1,a) lies on the graph of y = 2x^2, then a = 2 -/
theorem quadratic_point_value (a : ℝ) : (2 : ℝ) * (1 : ℝ)^2 = a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_value_l841_84194


namespace NUMINAMATH_CALUDE_sqrt_negative_one_squared_l841_84138

theorem sqrt_negative_one_squared (x : ℝ) : Real.sqrt ((-1) * (-1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_negative_one_squared_l841_84138


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l841_84170

theorem fraction_sum_simplification : 
  5 / (1/(1*2) + 1/(2*3) + 1/(3*4) + 1/(4*5) + 1/(5*6)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l841_84170


namespace NUMINAMATH_CALUDE_hotel_flat_fee_calculation_l841_84179

/-- A hotel charging system with a flat fee for the first night and a separate rate for additional nights. -/
structure HotelCharges where
  flatFee : ℝ  -- Flat fee for the first night
  nightlyRate : ℝ  -- Rate for each additional night

/-- Calculate the total cost for a given number of nights -/
def totalCost (h : HotelCharges) (nights : ℕ) : ℝ :=
  h.flatFee + h.nightlyRate * (nights - 1)

/-- Theorem stating the flat fee for the first night given the conditions -/
theorem hotel_flat_fee_calculation (h : HotelCharges) :
  totalCost h 2 = 120 ∧ totalCost h 5 = 255 → h.flatFee = 75 := by
  sorry

#check hotel_flat_fee_calculation

end NUMINAMATH_CALUDE_hotel_flat_fee_calculation_l841_84179


namespace NUMINAMATH_CALUDE_sufficient_conditions_for_x_squared_less_than_one_l841_84105

theorem sufficient_conditions_for_x_squared_less_than_one :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 < 1) ∧
  (∀ x : ℝ, -1 < x ∧ x < 0 → x^2 < 1) ∧
  (∀ x : ℝ, -1 < x ∧ x < 1 → x^2 < 1) ∧
  (∃ x : ℝ, x < 1 ∧ x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_conditions_for_x_squared_less_than_one_l841_84105


namespace NUMINAMATH_CALUDE_lottery_probability_theorem_l841_84181

def megaBallCount : ℕ := 30
def winnerBallsTotal : ℕ := 50
def winnerBallsPicked : ℕ := 5
def bonusBallCount : ℕ := 15

def lotteryProbability : ℚ :=
  1 / (megaBallCount * (Nat.choose winnerBallsTotal winnerBallsPicked) * bonusBallCount)

theorem lottery_probability_theorem :
  lotteryProbability = 1 / 95673600 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_theorem_l841_84181


namespace NUMINAMATH_CALUDE_calculate_y_investment_y_investment_proof_l841_84110

/-- Calculates the investment amount of partner y in a business partnership --/
theorem calculate_y_investment (x_investment : ℕ) (total_profit : ℕ) (x_profit_share : ℕ) : ℕ :=
  let y_profit_share := total_profit - x_profit_share
  let y_investment := (y_profit_share * x_investment) / x_profit_share
  y_investment

/-- Proves that y's investment is 15000 given the problem conditions --/
theorem y_investment_proof :
  calculate_y_investment 5000 1600 400 = 15000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_y_investment_y_investment_proof_l841_84110


namespace NUMINAMATH_CALUDE_joanne_weekly_earnings_l841_84171

def main_job_hours : ℝ := 8
def main_job_rate : ℝ := 16
def part_time_hours : ℝ := 2
def part_time_rate : ℝ := 13.5
def days_per_week : ℝ := 5

def weekly_earnings : ℝ := (main_job_hours * main_job_rate + part_time_hours * part_time_rate) * days_per_week

theorem joanne_weekly_earnings : weekly_earnings = 775 := by
  sorry

end NUMINAMATH_CALUDE_joanne_weekly_earnings_l841_84171


namespace NUMINAMATH_CALUDE_closest_multiple_of_15_to_2021_l841_84112

theorem closest_multiple_of_15_to_2021 : ∃ (n : ℤ), 
  15 * n = 2025 ∧ 
  ∀ (m : ℤ), m ≠ n → 15 * m ≠ 2025 → |2021 - 15 * n| ≤ |2021 - 15 * m| := by
  sorry

end NUMINAMATH_CALUDE_closest_multiple_of_15_to_2021_l841_84112


namespace NUMINAMATH_CALUDE_convertible_count_l841_84190

theorem convertible_count (total : ℕ) (regular_percent : ℚ) (truck_percent : ℚ) :
  total = 125 →
  regular_percent = 64 / 100 →
  truck_percent = 8 / 100 →
  (total : ℚ) * regular_percent + (total : ℚ) * truck_percent + 35 = total :=
by sorry

end NUMINAMATH_CALUDE_convertible_count_l841_84190


namespace NUMINAMATH_CALUDE_proposition_truth_l841_84193

theorem proposition_truth (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬¬q) : 
  (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l841_84193


namespace NUMINAMATH_CALUDE_equation_solution_l841_84199

theorem equation_solution (x : ℝ) : 
  (Real.sqrt (x + 12) - 8 / Real.sqrt (x + 12) = 4) ↔ (x = 4 + 8 * Real.sqrt 3 ∨ x = 4 - 8 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l841_84199


namespace NUMINAMATH_CALUDE_percentage_of_indian_children_l841_84183

theorem percentage_of_indian_children (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (percent_indian_men : ℚ) (percent_indian_women : ℚ) (percent_not_indian : ℚ)
  (h1 : total_men = 700)
  (h2 : total_women = 500)
  (h3 : total_children = 800)
  (h4 : percent_indian_men = 20 / 100)
  (h5 : percent_indian_women = 40 / 100)
  (h6 : percent_not_indian = 79 / 100) :
  (((1 - percent_not_indian) * (total_men + total_women + total_children) -
    percent_indian_men * total_men - percent_indian_women * total_women) /
    total_children : ℚ) = 10 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_indian_children_l841_84183


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l841_84146

theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  a 1 = 1 →                                         -- first term
  a 2 = 5 →                                         -- second term
  a 3 = 9 →                                         -- third term
  ∀ n, a n = 4 * n - 3 :=                           -- general formula
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l841_84146


namespace NUMINAMATH_CALUDE_vector_sum_parallel_l841_84191

/-- Given two parallel vectors a and b in R², prove that their linear combination results in (-4, -8) -/
theorem vector_sum_parallel (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  (2 • a + 3 • b : Fin 2 → ℝ) = ![-4, -8] := by
sorry

end NUMINAMATH_CALUDE_vector_sum_parallel_l841_84191


namespace NUMINAMATH_CALUDE_complement_of_A_l841_84154

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x ≥ 1} ∪ {x : ℝ | x < 0}

theorem complement_of_A : Set.compl A = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l841_84154


namespace NUMINAMATH_CALUDE_length_BC_in_triangle_l841_84196

/-- Parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- Triangle ABC -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Theorem: Length of BC in triangle ABC -/
theorem length_BC_in_triangle (t : Triangle) : 
  (t.A.1 = 0 ∧ t.A.2 = 0) →  -- A is at origin
  (t.B.2 = parabola t.B.1) →  -- B is on parabola
  (t.C.2 = parabola t.C.1) →  -- C is on parabola
  (t.B.2 = t.C.2) →  -- BC is parallel to x-axis
  (1/2 * (t.C.1 - t.B.1) * t.B.2 = 128) →  -- Area of triangle is 128
  (t.C.1 - t.B.1 = 8) :=  -- Length of BC is 8
by sorry

end NUMINAMATH_CALUDE_length_BC_in_triangle_l841_84196


namespace NUMINAMATH_CALUDE_sum_of_squares_l841_84133

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 20)
  (eq2 : y^2 + 5*z = -20)
  (eq3 : z^2 + 7*x = -34) :
  x^2 + y^2 + z^2 = 20.75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l841_84133


namespace NUMINAMATH_CALUDE_external_tangency_intersection_two_points_l841_84137

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0
def C₂ (x y r : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = r^2

-- Define the center and radius of C₁
def center_C₁ : ℝ × ℝ := (1, 1)
def radius_C₁ : ℝ := 1

-- Define the center of C₂
def center_C₂ : ℝ × ℝ := (4, 5)

-- Define the distance between centers
def distance_between_centers : ℝ := 5

-- Theorem for external tangency
theorem external_tangency (r : ℝ) (hr : r > 0) :
  (∀ x y, C₁ x y → C₂ x y r → (x - 1)^2 + (y - 1)^2 = 1 ∧ (x - 4)^2 + (y - 5)^2 = r^2) →
  distance_between_centers = radius_C₁ + r →
  r = 4 :=
sorry

-- Theorem for intersection at two points
theorem intersection_two_points (r : ℝ) (hr : r > 0) :
  (∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ C₂ x₁ y₁ r ∧ C₂ x₂ y₂ r) →
  4 < r ∧ r < 6 :=
sorry

end NUMINAMATH_CALUDE_external_tangency_intersection_two_points_l841_84137


namespace NUMINAMATH_CALUDE_A_minus_3B_equals_x_cubed_plus_y_cubed_l841_84192

variable (x y : ℝ)

def A : ℝ := x^3 + 3*x^2*y + y^3 - 3*x*y^2
def B : ℝ := x^2*y - x*y^2

theorem A_minus_3B_equals_x_cubed_plus_y_cubed :
  A x y - 3 * B x y = x^3 + y^3 := by sorry

end NUMINAMATH_CALUDE_A_minus_3B_equals_x_cubed_plus_y_cubed_l841_84192


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l841_84114

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2) : 
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l841_84114


namespace NUMINAMATH_CALUDE_basketball_club_girls_l841_84148

theorem basketball_club_girls (total_members : ℕ) (attendance : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_members = 30 →
  attendance = 18 →
  boys + girls = total_members →
  boys + (1/3 : ℚ) * girls = attendance →
  girls = 18 :=
by sorry

end NUMINAMATH_CALUDE_basketball_club_girls_l841_84148


namespace NUMINAMATH_CALUDE_B_grazed_five_months_l841_84118

/-- Represents the number of months B grazed his cows -/
def B_months : ℕ := sorry

/-- Total rent of the field in rupees -/
def total_rent : ℕ := 3250

/-- A's share of rent in rupees -/
def A_rent : ℕ := 720

/-- Number of cows grazed by each milkman -/
def cows : Fin 4 → ℕ
| 0 => 24  -- A
| 1 => 10  -- B
| 2 => 35  -- C
| 3 => 21  -- D

/-- Number of months each milkman grazed their cows -/
def months : Fin 4 → ℕ
| 0 => 3         -- A
| 1 => B_months  -- B
| 2 => 4         -- C
| 3 => 3         -- D

/-- Total cow-months for all milkmen -/
def total_cow_months : ℕ := 
  (cows 0 * months 0) + (cows 1 * months 1) + (cows 2 * months 2) + (cows 3 * months 3)

theorem B_grazed_five_months : B_months = 5 := by
  sorry

end NUMINAMATH_CALUDE_B_grazed_five_months_l841_84118


namespace NUMINAMATH_CALUDE_complex_equation_result_l841_84153

theorem complex_equation_result (z : ℂ) 
  (h : 15 * Complex.normSq z = 3 * Complex.normSq (z + 3) + Complex.normSq (z^2 + 4) + 25) : 
  z + 8 / z = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l841_84153


namespace NUMINAMATH_CALUDE_number_of_team_formations_l841_84159

def male_athletes : ℕ := 5
def female_athletes : ℕ := 5
def team_size : ℕ := 6
def ma_long_selected : Prop := true
def ding_ning_selected : Prop := true

def remaining_male_athletes : ℕ := male_athletes - 1
def remaining_female_athletes : ℕ := female_athletes - 1
def remaining_slots : ℕ := team_size - 2

theorem number_of_team_formations :
  (Nat.choose remaining_male_athletes (remaining_slots / 2))^2 *
  (Nat.factorial remaining_slots) =
  number_of_ways_to_form_teams :=
sorry

end NUMINAMATH_CALUDE_number_of_team_formations_l841_84159


namespace NUMINAMATH_CALUDE_net_error_is_24x_l841_84129

/-- The net error in cents due to the cashier's miscounting -/
def net_error (x : ℕ) : ℤ :=
  let penny_value : ℤ := 1
  let nickel_value : ℤ := 5
  let dime_value : ℤ := 10
  let quarter_value : ℤ := 25
  let penny_to_nickel_error := x * (nickel_value - penny_value)
  let nickel_to_dime_error := x * (dime_value - nickel_value)
  let dime_to_quarter_error := x * (quarter_value - dime_value)
  penny_to_nickel_error + nickel_to_dime_error + dime_to_quarter_error

theorem net_error_is_24x (x : ℕ) : net_error x = 24 * x :=
sorry

end NUMINAMATH_CALUDE_net_error_is_24x_l841_84129


namespace NUMINAMATH_CALUDE_parabola_line_intersection_right_angle_l841_84111

/-- Parabola represented by the equation y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  isParabola : equation = fun x y => y^2 = 4*x

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line represented by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Angle between two vectors -/
def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_line_intersection_right_angle 
  (E : Parabola) 
  (M N : Point)
  (MN : Line)
  (A B : Point)
  (h1 : M.x = 1 ∧ M.y = -3)
  (h2 : N.x = 5 ∧ N.y = 1)
  (h3 : MN.p1 = M ∧ MN.p2 = N)
  (h4 : E.equation A.x A.y ∧ E.equation B.x B.y)
  (h5 : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
        A.x = M.x + t * (N.x - M.x) ∧ 
        A.y = M.y + t * (N.y - M.y))
  (h6 : ∃ s : ℝ, 0 < s ∧ s < 1 ∧ 
        B.x = M.x + s * (N.x - M.x) ∧ 
        B.y = M.y + s * (N.y - M.y))
  : angle (A.x, A.y) (B.x, B.y) = π / 2 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_right_angle_l841_84111


namespace NUMINAMATH_CALUDE_employee_pay_l841_84180

theorem employee_pay (x y : ℝ) (h1 : x + y = 616) (h2 : x = 1.2 * y) : y = 280 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l841_84180


namespace NUMINAMATH_CALUDE_oldies_requests_l841_84184

/-- Represents the number of song requests for each genre --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of oldies requests given the conditions --/
theorem oldies_requests (sr : SongRequests) : sr.oldies = 5 :=
  by
  have h1 : sr.total = 30 := by sorry
  have h2 : sr.electropop = sr.total / 2 := by sorry
  have h3 : sr.dance = sr.electropop / 3 := by sorry
  have h4 : sr.rock = 5 := by sorry
  have h5 : sr.dj_choice = sr.oldies / 2 := by sorry
  have h6 : sr.rap = 2 := by sorry
  have h7 : sr.total = sr.electropop + sr.rock + sr.oldies + sr.dj_choice + sr.rap := by sorry
  sorry

end NUMINAMATH_CALUDE_oldies_requests_l841_84184


namespace NUMINAMATH_CALUDE_x_total_time_is_20_l841_84186

-- Define the work as a fraction of the total job
def Work := ℚ

-- Define the time y needs to finish the entire work
def y_total_time : ℕ := 16

-- Define the time y worked before leaving
def y_worked_time : ℕ := 12

-- Define the time x needed to finish the remaining work
def x_remaining_time : ℕ := 5

-- Theorem to prove
theorem x_total_time_is_20 : 
  ∃ (x_total_time : ℕ), 
    (y_worked_time : ℚ) / y_total_time + 
    (x_remaining_time : ℚ) / x_total_time = 1 ∧ 
    x_total_time = 20 := by sorry

end NUMINAMATH_CALUDE_x_total_time_is_20_l841_84186


namespace NUMINAMATH_CALUDE_car_speed_l841_84142

/-- Proves that a car traveling 810 km in 5 hours has a speed of 162 km/hour -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 810)
  (h2 : time = 5)
  (h3 : speed = distance / time) :
  speed = 162 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l841_84142


namespace NUMINAMATH_CALUDE_football_season_length_l841_84152

/-- The number of months in a football season -/
def season_length (total_games : ℕ) (games_per_month : ℕ) : ℕ :=
  total_games / games_per_month

/-- Proof that the football season lasts 17 months -/
theorem football_season_length : season_length 323 19 = 17 := by
  sorry

end NUMINAMATH_CALUDE_football_season_length_l841_84152


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_l841_84135

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * Real.cos x

theorem tangent_line_at_zero (x y : ℝ) :
  (x - y + 3 = 0) ↔ 
  (∃ (m : ℝ), y - f 0 = m * (x - 0) ∧ 
               m = (deriv f) 0 ∧
               y = f 0 + m * x) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_l841_84135


namespace NUMINAMATH_CALUDE_election_win_margin_l841_84177

theorem election_win_margin :
  ∀ (total_votes : ℕ) (winner_votes loser_votes : ℕ),
    (winner_votes : ℚ) / total_votes = 3/5 →
    winner_votes = 720 →
    total_votes = winner_votes + loser_votes →
    winner_votes - loser_votes = 240 := by
  sorry

end NUMINAMATH_CALUDE_election_win_margin_l841_84177


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l841_84182

theorem polar_to_rectangular_conversion :
  let r : ℝ := 7
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 3.5 ∧ y = 7 * Real.sqrt 3 / 2) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l841_84182


namespace NUMINAMATH_CALUDE_ellipse_focal_coordinates_specific_ellipse_focal_coordinates_l841_84117

/-- The focal coordinates of an ellipse with equation x²/a² + y²/b² = 1 are (±c, 0) where c² = a² - b² -/
theorem ellipse_focal_coordinates (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let c := Real.sqrt (a^2 - b^2)
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  (∃ x : ℝ, x = c ∨ x = -c) ∧ (∀ x : ℝ, x^2 = c^2 → x = c ∨ x = -c) :=
by sorry

/-- The focal coordinates of the ellipse x²/5 + y²/4 = 1 are (±1, 0) -/
theorem specific_ellipse_focal_coordinates :
  let c := Real.sqrt (5 - 4)
  (∀ x y : ℝ, x^2 / 5 + y^2 / 4 = 1) →
  (∃ x : ℝ, x = 1 ∨ x = -1) ∧ (∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_coordinates_specific_ellipse_focal_coordinates_l841_84117


namespace NUMINAMATH_CALUDE_complement_A_in_U_l841_84197

def U : Set ℝ := {x | x^2 ≤ 4}
def A : Set ℝ := {x | |x + 1| ≤ 1}

theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l841_84197


namespace NUMINAMATH_CALUDE_quadratic_roots_l841_84107

theorem quadratic_roots (a : ℝ) : 
  (2 : ℝ)^2 + 2 - a = 0 → (-3 : ℝ)^2 + (-3) - a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l841_84107


namespace NUMINAMATH_CALUDE_jills_bus_journey_ratio_l841_84164

/-- Represents the time in minutes for various parts of Jill's bus journey -/
structure BusJourney where
  first_bus_wait : ℕ
  first_bus_ride : ℕ
  second_bus_ride : ℕ

/-- Calculates the ratio of the second bus ride time to the combined wait and trip time of the first bus -/
def bus_time_ratio (journey : BusJourney) : ℚ :=
  journey.second_bus_ride / (journey.first_bus_wait + journey.first_bus_ride)

/-- Theorem stating that for Jill's specific journey, the bus time ratio is 1/2 -/
theorem jills_bus_journey_ratio :
  let journey : BusJourney := { first_bus_wait := 12, first_bus_ride := 30, second_bus_ride := 21 }
  bus_time_ratio journey = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_jills_bus_journey_ratio_l841_84164


namespace NUMINAMATH_CALUDE_unique_solution_l841_84100

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y) = f (x * y^2) - 2 * x^2 * f y - f x - 1

theorem unique_solution (f : ℝ → ℝ) (h : functional_equation f) :
  ∀ y : ℝ, f y = y^2 - 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l841_84100


namespace NUMINAMATH_CALUDE_find_number_l841_84126

theorem find_number (A B : ℕ) (hA : A > 0) (hB : B > 0) : 
  Nat.gcd A B = 15 → Nat.lcm A B = 312 → B = 195 → A = 24 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l841_84126


namespace NUMINAMATH_CALUDE_data_transmission_time_l841_84127

theorem data_transmission_time (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) :
  blocks = 100 →
  chunks_per_block = 800 →
  transmission_rate = 200 →
  (blocks * chunks_per_block : ℝ) / transmission_rate / 60 = 6.666666666666667 :=
by sorry

end NUMINAMATH_CALUDE_data_transmission_time_l841_84127


namespace NUMINAMATH_CALUDE_min_value_t_l841_84161

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    satisfying certain conditions, the minimum value of t is 4√2/3. -/
theorem min_value_t (a b c : ℝ) (A B C : ℝ) (S : ℝ) (t : ℝ) :
  a - b = c / 3 →
  3 * Real.sin B = 2 * Real.sin A →
  2 ≤ a * c + c^2 →
  a * c + c^2 ≤ 32 →
  S = (1 / 2) * a * b * Real.sin C →
  t = (S + 2 * Real.sqrt 2) / a →
  t ≥ 4 * Real.sqrt 2 / 3 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ) (A₀ B₀ C₀ : ℝ) (S₀ : ℝ),
    a₀ - b₀ = c₀ / 3 ∧
    3 * Real.sin B₀ = 2 * Real.sin A₀ ∧
    2 ≤ a₀ * c₀ + c₀^2 ∧
    a₀ * c₀ + c₀^2 ≤ 32 ∧
    S₀ = (1 / 2) * a₀ * b₀ * Real.sin C₀ ∧
    (S₀ + 2 * Real.sqrt 2) / a₀ = 4 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_t_l841_84161


namespace NUMINAMATH_CALUDE_mine_locations_determinable_l841_84187

/-- Represents the state of a cell in the grid -/
inductive CellState
  | Empty
  | Mine

/-- Represents the grid of cells -/
def Grid (n : ℕ) := Fin n → Fin n → CellState

/-- The number displayed in a cell, which is the count of mines in the cell and its surroundings -/
def CellNumber (n : ℕ) (grid : Grid n) (i j : Fin n) : Fin 10 :=
  sorry

/-- Checks if it's possible to uniquely determine mine locations given cell numbers -/
def CanDetermineMineLocations (n : ℕ) (cellNumbers : Fin n → Fin n → Fin 10) : Prop :=
  ∃! (grid : Grid n), ∀ (i j : Fin n), CellNumber n grid i j = cellNumbers i j

/-- Theorem stating that mine locations can be determined for n = 2009 and n = 2007 -/
theorem mine_locations_determinable :
  (∀ (cellNumbers : Fin 2009 → Fin 2009 → Fin 10), CanDetermineMineLocations 2009 cellNumbers) ∧
  (∀ (cellNumbers : Fin 2007 → Fin 2007 → Fin 10), CanDetermineMineLocations 2007 cellNumbers) :=
sorry

end NUMINAMATH_CALUDE_mine_locations_determinable_l841_84187


namespace NUMINAMATH_CALUDE_quadratic_from_means_l841_84185

theorem quadratic_from_means (a b : ℝ) (h_am : (a + b) / 2 = 7.5) (h_gm : Real.sqrt (a * b) = 12) :
  ∀ x, x ^ 2 - 15 * x + 144 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_from_means_l841_84185


namespace NUMINAMATH_CALUDE_average_playing_time_l841_84101

/-- The average playing time for children playing table tennis -/
theorem average_playing_time
  (num_children : ℕ)
  (total_time : ℝ)
  (h_num_children : num_children = 5)
  (h_total_time : total_time = 15)
  : total_time / num_children = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_playing_time_l841_84101
