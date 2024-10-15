import Mathlib

namespace NUMINAMATH_CALUDE_unique_base_solution_l4118_411831

/-- Convert a number from base b to decimal --/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Check if the equation holds for a given base --/
def equationHolds (b : Nat) : Prop :=
  toDecimal [1, 7, 2] b + toDecimal [1, 5, 6] b = toDecimal [3, 4, 0] b

/-- The main theorem stating that 10 is the unique solution --/
theorem unique_base_solution :
  ∃! b : Nat, b > 1 ∧ equationHolds b :=
  sorry

end NUMINAMATH_CALUDE_unique_base_solution_l4118_411831


namespace NUMINAMATH_CALUDE_inequality_solution_l4118_411870

theorem inequality_solution (x m : ℝ) : 
  (x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0) → 
  (∀ x, x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0 → 2*x^2 - 9*x + m < 0) → 
  m < 9 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4118_411870


namespace NUMINAMATH_CALUDE_num_kittens_is_eleven_l4118_411842

/-- The number of kittens -/
def num_kittens : ℕ := 11

/-- The weight of the two lightest kittens -/
def weight_lightest : ℕ := 80

/-- The weight of the four heaviest kittens -/
def weight_heaviest : ℕ := 200

/-- The total weight of all kittens -/
def total_weight : ℕ := 500

/-- Theorem stating that the number of kittens is 11 given the weight conditions -/
theorem num_kittens_is_eleven :
  (weight_lightest = 80) →
  (weight_heaviest = 200) →
  (total_weight = 500) →
  (num_kittens = 11) :=
by
  sorry

#check num_kittens_is_eleven

end NUMINAMATH_CALUDE_num_kittens_is_eleven_l4118_411842


namespace NUMINAMATH_CALUDE_sqrt_neg_a_rational_implies_a_opposite_perfect_square_l4118_411849

theorem sqrt_neg_a_rational_implies_a_opposite_perfect_square (a : ℝ) :
  (∃ q : ℚ, q^2 = -a) → ∃ n : ℕ, a = -(n^2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_a_rational_implies_a_opposite_perfect_square_l4118_411849


namespace NUMINAMATH_CALUDE_monthly_sales_fraction_l4118_411898

theorem monthly_sales_fraction (december_sales : ℝ) (monthly_sales : ℝ) (total_sales : ℝ) :
  december_sales = 6 * monthly_sales →
  december_sales = 0.35294117647058826 * total_sales →
  monthly_sales = (1 / 17) * total_sales := by
sorry

end NUMINAMATH_CALUDE_monthly_sales_fraction_l4118_411898


namespace NUMINAMATH_CALUDE_share_ratio_a_to_b_l4118_411867

/-- Proof of the ratio of shares between A and B --/
theorem share_ratio_a_to_b (total amount : ℕ) (a_share b_share c_share : ℕ) :
  amount = 510 →
  a_share = 360 →
  b_share = 90 →
  c_share = 60 →
  b_share = c_share / 4 →
  a_share / b_share = 4 :=
by sorry

end NUMINAMATH_CALUDE_share_ratio_a_to_b_l4118_411867


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_34020_l4118_411811

def largest_perfect_square_factor (n : ℕ) : ℕ := 
  sorry

theorem largest_perfect_square_factor_34020 :
  largest_perfect_square_factor 34020 = 324 := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_34020_l4118_411811


namespace NUMINAMATH_CALUDE_solve_equation_l4118_411869

theorem solve_equation : ∃ x : ℝ, (5 - x = 8) ∧ (x = -3) := by sorry

end NUMINAMATH_CALUDE_solve_equation_l4118_411869


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l4118_411878

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Point on a parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : para.eq x y

/-- Circle passing through two points -/
def circle_eq (P₁ P₂ : ℝ × ℝ) (x y : ℝ) : Prop :=
  (x - P₁.1) * (x - P₂.1) + (y - P₁.2) * (y - P₂.2) = 0

/-- Main theorem -/
theorem parabola_circle_tangency (para : Parabola) (P₁ P₂ : ParabolaPoint para)
    (h : |P₁.y - P₂.y| = 4 * para.p) :
    ∃! (P : ℝ × ℝ), P ≠ (P₁.x, P₁.y) ∧ P ≠ (P₂.x, P₂.y) ∧
      para.eq P.1 P.2 ∧ circle_eq (P₁.x, P₁.y) (P₂.x, P₂.y) P.1 P.2 :=
  sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l4118_411878


namespace NUMINAMATH_CALUDE_ore_without_alloy_percentage_l4118_411801

/-- Represents the composition of an ore -/
structure Ore where
  alloy_percentage : Real
  iron_in_alloy : Real
  total_ore : Real
  pure_iron : Real

/-- Theorem: The percentage of ore not containing the alloy with iron is 75% -/
theorem ore_without_alloy_percentage (ore : Ore)
  (h1 : ore.alloy_percentage = 0.25)
  (h2 : ore.iron_in_alloy = 0.90)
  (h3 : ore.total_ore = 266.6666666666667)
  (h4 : ore.pure_iron = 60) :
  1 - ore.alloy_percentage = 0.75 := by
  sorry

#check ore_without_alloy_percentage

end NUMINAMATH_CALUDE_ore_without_alloy_percentage_l4118_411801


namespace NUMINAMATH_CALUDE_licorice_probability_l4118_411836

def n : ℕ := 7
def k : ℕ := 5
def p : ℚ := 3/5

theorem licorice_probability :
  Nat.choose n k * p^k * (1 - p)^(n - k) = 20412/78125 := by
  sorry

end NUMINAMATH_CALUDE_licorice_probability_l4118_411836


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l4118_411854

theorem square_plus_reciprocal_square (a : ℝ) (h : a + 1/a = Real.sqrt 5) :
  a^2 + 1/a^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l4118_411854


namespace NUMINAMATH_CALUDE_cyclist_distance_l4118_411864

theorem cyclist_distance (x t : ℝ) 
  (h1 : (x + 1/3) * (3*t/4) = x * t)
  (h2 : (x - 1/3) * (t + 3) = x * t) :
  x * t = 132 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_distance_l4118_411864


namespace NUMINAMATH_CALUDE_power_product_l4118_411852

theorem power_product (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 8) : a^m * a^n = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_product_l4118_411852


namespace NUMINAMATH_CALUDE_largest_coefficient_term_l4118_411893

theorem largest_coefficient_term (n : ℕ+) :
  ∀ k : ℕ, k ≠ n + 1 →
    Nat.choose (2 * n) (n + 1) ≥ Nat.choose (2 * n) k := by
  sorry

end NUMINAMATH_CALUDE_largest_coefficient_term_l4118_411893


namespace NUMINAMATH_CALUDE_arithmetic_computation_l4118_411820

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 7 * 2 / 2 * 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l4118_411820


namespace NUMINAMATH_CALUDE_min_distance_complex_unit_circle_l4118_411853

theorem min_distance_complex_unit_circle (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ w : ℂ, Complex.abs w = 1 → Complex.abs (w + 4*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_unit_circle_l4118_411853


namespace NUMINAMATH_CALUDE_sales_tax_theorem_l4118_411818

/-- Calculates the sales tax paid given total purchase, tax rate, and cost of tax-free items -/
def calculate_sales_tax (total_purchase : ℝ) (tax_rate : ℝ) (tax_free_cost : ℝ) : ℝ :=
  let taxable_cost := total_purchase - tax_free_cost
  tax_rate * taxable_cost

/-- Theorem stating that under the given conditions, the sales tax paid is 0.3 -/
theorem sales_tax_theorem (total_purchase tax_rate tax_free_cost : ℝ) 
  (h1 : total_purchase = 25)
  (h2 : tax_rate = 0.06)
  (h3 : tax_free_cost = 19.7) :
  calculate_sales_tax total_purchase tax_rate tax_free_cost = 0.3 := by
  sorry

#eval calculate_sales_tax 25 0.06 19.7

end NUMINAMATH_CALUDE_sales_tax_theorem_l4118_411818


namespace NUMINAMATH_CALUDE_glasses_in_smaller_box_l4118_411822

theorem glasses_in_smaller_box :
  ∀ (x : ℕ) (s l : ℕ),
  -- There are two different-sized boxes
  s = 1 →
  -- There are 16 more larger boxes than smaller boxes
  l = s + 16 →
  -- One box (smaller) contains x glasses, the other (larger) contains 16 glasses
  -- The total number of glasses is 480
  x * s + 16 * l = 480 →
  -- Prove that the number of glasses in the smaller box is 208
  x = 208 := by
sorry

end NUMINAMATH_CALUDE_glasses_in_smaller_box_l4118_411822


namespace NUMINAMATH_CALUDE_system_solution_ratio_l4118_411892

theorem system_solution_ratio (k x y z : ℝ) : 
  x + k*y + 2*z = 0 →
  2*x + k*y + 3*z = 0 →
  3*x + 5*y + 4*z = 0 →
  x ≠ 0 →
  y ≠ 0 →
  z ≠ 0 →
  x*z / (y^2) = -25 := by sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l4118_411892


namespace NUMINAMATH_CALUDE_total_ways_eq_600_l4118_411808

/-- Represents the number of cards in the left pocket -/
def left_cards : ℕ := 30

/-- Represents the number of cards in the right pocket -/
def right_cards : ℕ := 20

/-- Represents the total number of ways to select one card from each pocket -/
def total_ways : ℕ := left_cards * right_cards

/-- Theorem stating that the total number of ways to select one card from each pocket is 600 -/
theorem total_ways_eq_600 : total_ways = 600 := by sorry

end NUMINAMATH_CALUDE_total_ways_eq_600_l4118_411808


namespace NUMINAMATH_CALUDE_shift_theorem_l4118_411859

/-- Given two functions f and g, where f is shifted by φ to obtain g, prove that sinφ = 24/25 -/
theorem shift_theorem (f g : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = 3 * Real.sin x + 4 * Real.cos x) →
  (∀ x, g x = 3 * Real.sin x - 4 * Real.cos x) →
  (∀ x, g x = f (x - φ)) →
  Real.sin φ = 24 / 25 := by
sorry

end NUMINAMATH_CALUDE_shift_theorem_l4118_411859


namespace NUMINAMATH_CALUDE_man_sold_portion_l4118_411846

theorem man_sold_portion (lot_value : ℝ) (sold_amount : ℝ) : 
  lot_value = 9200 → 
  sold_amount = 460 → 
  sold_amount / (lot_value / 2) = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_man_sold_portion_l4118_411846


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l4118_411857

theorem factorial_difference_quotient : (Nat.factorial 13 - Nat.factorial 12) / Nat.factorial 10 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l4118_411857


namespace NUMINAMATH_CALUDE_proportion_equality_l4118_411840

theorem proportion_equality (x : ℚ) : 
  (3 : ℚ) / 5 = 12 / 20 ∧ x / 10 = 16 / 40 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l4118_411840


namespace NUMINAMATH_CALUDE_inequality_proof_l4118_411850

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b * c / a + c * a / b + a * b / c ≥ a + b + c) ∧
  (a + b + c = 1 → (1 - a) / a + (1 - b) / b + (1 - c) / c ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4118_411850


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l4118_411841

/-- Given a rectangular grid and two unshaded shapes within it, calculate the area of the shaded region. -/
theorem shaded_area_calculation (grid_width grid_height : ℝ)
  (triangle_base triangle_height : ℝ)
  (trapezoid_height trapezoid_top_base trapezoid_bottom_base : ℝ) :
  grid_width = 10 ∧ 
  grid_height = 5 ∧ 
  triangle_base = 3 ∧ 
  triangle_height = 2 ∧ 
  trapezoid_height = 3 ∧ 
  trapezoid_top_base = 3 ∧ 
  trapezoid_bottom_base = 6 →
  grid_width * grid_height - 
  (1/2 * triangle_base * triangle_height) - 
  (1/2 * (trapezoid_top_base + trapezoid_bottom_base) * trapezoid_height) = 33.5 := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_calculation_l4118_411841


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l4118_411888

theorem cube_surface_area_increase (s : ℝ) (h : s > 0) :
  let original_surface_area := 6 * s^2
  let new_edge_length := 1.8 * s
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 2.24 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l4118_411888


namespace NUMINAMATH_CALUDE_abs_neg_two_equals_two_l4118_411851

theorem abs_neg_two_equals_two : abs (-2 : ℤ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_equals_two_l4118_411851


namespace NUMINAMATH_CALUDE_only_A_in_first_quadrant_l4118_411868

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def point_A : ℝ × ℝ := (3, 2)
def point_B : ℝ × ℝ := (-3, 2)
def point_C : ℝ × ℝ := (3, -2)
def point_D : ℝ × ℝ := (-3, -2)

theorem only_A_in_first_quadrant :
  first_quadrant point_A.1 point_A.2 ∧
  ¬first_quadrant point_B.1 point_B.2 ∧
  ¬first_quadrant point_C.1 point_C.2 ∧
  ¬first_quadrant point_D.1 point_D.2 := by
  sorry

end NUMINAMATH_CALUDE_only_A_in_first_quadrant_l4118_411868


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l4118_411824

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 200 →
  a * b + b * c + c * d ≤ 10000 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l4118_411824


namespace NUMINAMATH_CALUDE_emily_phone_bill_l4118_411899

/-- Calculates the total cost of a cell phone plan based on usage --/
def calculate_phone_bill (base_cost : ℚ) (text_cost : ℚ) (extra_minute_cost : ℚ) 
  (extra_data_cost : ℚ) (texts_sent : ℕ) (hours_talked : ℕ) (data_used : ℕ) : ℚ :=
  let text_charge := text_cost * texts_sent
  let extra_minutes := max (hours_talked - 25) 0 * 60
  let minute_charge := extra_minute_cost * extra_minutes
  let extra_data := max (data_used - 15) 0
  let data_charge := extra_data_cost * extra_data
  base_cost + text_charge + minute_charge + data_charge

/-- Theorem stating that Emily's phone bill is $59.00 --/
theorem emily_phone_bill : 
  calculate_phone_bill 30 0.1 0.15 5 150 26 16 = 59 := by
  sorry

end NUMINAMATH_CALUDE_emily_phone_bill_l4118_411899


namespace NUMINAMATH_CALUDE_divisibility_in_base_greater_than_six_l4118_411835

theorem divisibility_in_base_greater_than_six (a : ℕ) (h : a > 6) :
  ∃ k : ℕ, a^10 + 2*a^9 + 3*a^8 + 4*a^7 + 5*a^6 + 6*a^5 + 5*a^4 + 4*a^3 + 3*a^2 + 2*a + 1
         = k * (a^4 + 2*a^3 + 3*a^2 + 2*a + 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_in_base_greater_than_six_l4118_411835


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l4118_411871

/-- A bag containing balls of two colors -/
structure Bag where
  black : ℕ
  white : ℕ

/-- The event of drawing balls from the bag -/
structure Draw where
  total : ℕ
  black : ℕ
  white : ℕ

/-- Definition of the specific bag in the problem -/
def problem_bag : Bag := { black := 2, white := 2 }

/-- Definition of drawing two balls -/
def two_ball_draw (b : Bag) : Set Draw := 
  {d | d.total = 2 ∧ d.black + d.white = d.total ∧ d.black ≤ b.black ∧ d.white ≤ b.white}

/-- Event: At least one black ball is drawn -/
def at_least_one_black (d : Draw) : Prop := d.black ≥ 1

/-- Event: All drawn balls are white -/
def all_white (d : Draw) : Prop := d.white = d.total

/-- Theorem: The events are mutually exclusive and complementary -/
theorem events_mutually_exclusive_and_complementary :
  let draw_set := two_ball_draw problem_bag
  ∀ d ∈ draw_set, (at_least_one_black d ↔ ¬ all_white d) ∧ 
                  (at_least_one_black d ∨ all_white d) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l4118_411871


namespace NUMINAMATH_CALUDE_fred_car_wash_earnings_l4118_411823

/-- Fred's earnings from washing the family car -/
def car_wash_earnings (weekly_allowance : ℕ) (final_amount : ℕ) : ℕ :=
  final_amount - weekly_allowance / 2

/-- Proof that Fred earned $6 from washing the family car -/
theorem fred_car_wash_earnings :
  car_wash_earnings 16 14 = 6 :=
by sorry

end NUMINAMATH_CALUDE_fred_car_wash_earnings_l4118_411823


namespace NUMINAMATH_CALUDE_original_hourly_wage_l4118_411817

/-- Given a worker's daily wage, increased wage, bonus, total new wage, and hours worked per day,
    calculate the original hourly wage. -/
theorem original_hourly_wage (W : ℝ) (h1 : 1.60 * W + 10 = 45) (h2 : 8 > 0) :
  W / 8 = (45 - 10) / (1.60 * 8) := by sorry

end NUMINAMATH_CALUDE_original_hourly_wage_l4118_411817


namespace NUMINAMATH_CALUDE_number_calculation_l4118_411845

theorem number_calculation : 
  let x : Float := 0.17999999999999997
  let number : Float := x * 0.05
  number / x = 0.05 ∧ number = 0.009 := by
sorry

end NUMINAMATH_CALUDE_number_calculation_l4118_411845


namespace NUMINAMATH_CALUDE_trebled_resultant_l4118_411834

theorem trebled_resultant (initial_number : ℕ) : initial_number = 5 → 
  3 * (2 * initial_number + 15) = 75 := by
  sorry

end NUMINAMATH_CALUDE_trebled_resultant_l4118_411834


namespace NUMINAMATH_CALUDE_books_count_proof_l4118_411875

/-- Given a ratio of items and a total count, calculates the number of items for a specific part of the ratio. -/
def calculate_items (ratio : List Nat) (total_items : Nat) (part_index : Nat) : Nat :=
  let total_parts := ratio.sum
  let items_per_part := total_items / total_parts
  items_per_part * (ratio.get! part_index)

/-- Proves that given the ratio 7:3:2 for books, pens, and notebooks, and a total of 600 items, the number of books is 350. -/
theorem books_count_proof :
  let ratio := [7, 3, 2]
  let total_items := 600
  let books_index := 0
  calculate_items ratio total_items books_index = 350 := by
  sorry

end NUMINAMATH_CALUDE_books_count_proof_l4118_411875


namespace NUMINAMATH_CALUDE_power_gt_one_iff_diff_times_b_gt_zero_l4118_411814

theorem power_gt_one_iff_diff_times_b_gt_zero
  (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  a^b > 1 ↔ (a - 1) * b > 0 := by
  sorry

end NUMINAMATH_CALUDE_power_gt_one_iff_diff_times_b_gt_zero_l4118_411814


namespace NUMINAMATH_CALUDE_sector_central_angle_l4118_411880

/-- Given a circular sector with perimeter 10 and area 4, prove that its central angle is 1/2 radians -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 10) (h2 : 1/2 * l * r = 4) :
  l / r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l4118_411880


namespace NUMINAMATH_CALUDE_tangent_circles_area_ratio_l4118_411804

/-- Regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- Circle tangent to three sides of a regular hexagon -/
structure TangentCircle (h : RegularHexagon) :=
  (radius : ℝ)
  (tangent_to_parallel_sides : True)
  (tangent_to_other_side : True)

/-- The ratio of areas of two tangent circles to a regular hexagon is 1 -/
theorem tangent_circles_area_ratio (h : RegularHexagon) 
  (c1 c2 : TangentCircle h) : 
  (c1.radius^2) / (c2.radius^2) = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_circles_area_ratio_l4118_411804


namespace NUMINAMATH_CALUDE_cubic_root_sum_l4118_411896

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 →
  q^3 - 8*q^2 + 10*q - 3 = 0 →
  r^3 - 8*r^2 + 10*r - 3 = 0 →
  p/(q*r + 2) + q/(p*r + 2) + r/(p*q + 2) = 4 + 9/20 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l4118_411896


namespace NUMINAMATH_CALUDE_unfactorable_quadratic_l4118_411821

/-- A quadratic trinomial that cannot be factored into linear binomials with integer coefficients -/
theorem unfactorable_quadratic (a b c : ℕ+) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_eval : a * 1991^2 + b * 1991 + c = p) :
  ¬ ∃ (d₁ d₂ e₁ e₂ : ℤ), ∀ x, a * x^2 + b * x + c = (d₁ * x + e₁) * (d₂ * x + e₂) :=
by sorry

end NUMINAMATH_CALUDE_unfactorable_quadratic_l4118_411821


namespace NUMINAMATH_CALUDE_goods_train_speed_l4118_411856

/-- The speed of a goods train passing a man in an opposite moving train -/
theorem goods_train_speed
  (man_train_speed : ℝ)
  (goods_train_length : ℝ)
  (passing_time : ℝ)
  (h1 : man_train_speed = 50)
  (h2 : goods_train_length = 280 / 1000)  -- Convert to km
  (h3 : passing_time = 9 / 3600)  -- Convert to hours
  : ∃ (goods_train_speed : ℝ),
    goods_train_speed = 62 ∧
    (man_train_speed + goods_train_speed) * passing_time = goods_train_length :=
by sorry


end NUMINAMATH_CALUDE_goods_train_speed_l4118_411856


namespace NUMINAMATH_CALUDE_min_value_theorem_l4118_411807

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 2/y = 1 → a*(b - 1) ≤ x*(y - 1) ∧ 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 2/y = 1 ∧ x*(y - 1) = 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4118_411807


namespace NUMINAMATH_CALUDE_problem_statement_l4118_411894

theorem problem_statement :
  (∃ n : ℤ, 15 = 3 * n) ∧
  (∃ m : ℤ, 121 = 11 * m) ∧ (¬ ∃ k : ℤ, 60 = 11 * k) ∧
  (∃ p : ℤ, 63 = 7 * p) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l4118_411894


namespace NUMINAMATH_CALUDE_inequality_of_positive_reals_l4118_411832

theorem inequality_of_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  b^2 / a + c^2 / b + a^2 / c ≥ Real.sqrt (3 * (a^2 + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_reals_l4118_411832


namespace NUMINAMATH_CALUDE_dog_food_cans_per_package_l4118_411860

/-- Given the following conditions:
  * Chad bought 6 packages of cat food, each containing 9 cans
  * Chad bought 2 packages of dog food
  * The total number of cat food cans is 48 more than the total number of dog food cans
  Prove that each package of dog food contains 3 cans -/
theorem dog_food_cans_per_package :
  let cat_packages : ℕ := 6
  let cat_cans_per_package : ℕ := 9
  let dog_packages : ℕ := 2
  let total_cat_cans : ℕ := cat_packages * cat_cans_per_package
  let dog_cans_per_package : ℕ := total_cat_cans / dog_packages - 24
  dog_cans_per_package = 3 := by sorry

end NUMINAMATH_CALUDE_dog_food_cans_per_package_l4118_411860


namespace NUMINAMATH_CALUDE_prism_18_edges_has_8_faces_l4118_411872

/-- Represents a prism -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ :=
  let base_sides := p.edges / 3
  base_sides + 2

/-- Theorem: A prism with 18 edges has 8 faces -/
theorem prism_18_edges_has_8_faces :
  ∀ p : Prism, p.edges = 18 → num_faces p = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_18_edges_has_8_faces_l4118_411872


namespace NUMINAMATH_CALUDE_company_employees_l4118_411895

/-- The number of employees in a company satisfying certain conditions. -/
theorem company_employees : 
  ∀ (total_females : ℕ) 
    (advanced_degrees : ℕ) 
    (males_college_only : ℕ) 
    (females_advanced : ℕ),
  total_females = 110 →
  advanced_degrees = 90 →
  males_college_only = 35 →
  females_advanced = 55 →
  ∃ (total_employees : ℕ),
    total_employees = 180 ∧
    total_employees = advanced_degrees + (males_college_only + (total_females - females_advanced)) :=
by sorry

end NUMINAMATH_CALUDE_company_employees_l4118_411895


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l4118_411813

/-- The cost of tickets at a circus --/
theorem circus_ticket_cost (cost_per_ticket : ℕ) (num_tickets : ℕ) (total_cost : ℕ) : 
  cost_per_ticket = 44 → num_tickets = 7 → total_cost = cost_per_ticket * num_tickets → total_cost = 308 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l4118_411813


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l4118_411861

/-- The complex number z = i(1+i) -/
def z : ℂ := Complex.I * (1 + Complex.I)

/-- The real part of z -/
def real_part : ℝ := z.re

/-- The imaginary part of z -/
def imag_part : ℝ := z.im

/-- Theorem: z is in the second quadrant -/
theorem z_in_second_quadrant : real_part < 0 ∧ imag_part > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l4118_411861


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l4118_411874

theorem arithmetic_mean_problem (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 89 → a = 34 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l4118_411874


namespace NUMINAMATH_CALUDE_mnp_nmp_difference_implies_mmp_nnp_difference_l4118_411876

/-- Represents a three-digit number in base 10 -/
def ThreeDigitNumber (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem mnp_nmp_difference_implies_mmp_nnp_difference
  (m n p : ℕ)
  (h : ThreeDigitNumber m n p - ThreeDigitNumber n m p = 180) :
  ThreeDigitNumber m m p - ThreeDigitNumber n n p = 220 := by
  sorry

end NUMINAMATH_CALUDE_mnp_nmp_difference_implies_mmp_nnp_difference_l4118_411876


namespace NUMINAMATH_CALUDE_quadratic_function_m_value_l4118_411882

/-- A quadratic function g(x) with integer coefficients -/
def g (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

/-- The theorem stating that under given conditions, m = 5 -/
theorem quadratic_function_m_value (a b c m : ℤ) :
  g a b c 2 = 0 →
  70 < g a b c 6 →
  g a b c 6 < 80 →
  110 < g a b c 7 →
  g a b c 7 < 120 →
  2000 * m < g a b c 50 →
  g a b c 50 < 2000 * (m + 1) →
  m = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_value_l4118_411882


namespace NUMINAMATH_CALUDE_square_function_not_property_P_l4118_411829

/-- Property P for a function f --/
def has_property_P (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f ((x₁ + x₂) / 2) = (f x₁ + f x₂) / 2

/-- The square function --/
def square_function (x : ℝ) : ℝ := x^2

/-- Theorem: The square function does not have property P --/
theorem square_function_not_property_P : ¬(has_property_P square_function) := by
  sorry

end NUMINAMATH_CALUDE_square_function_not_property_P_l4118_411829


namespace NUMINAMATH_CALUDE_new_person_weight_l4118_411891

/-- Given a group of 8 people where one person weighing 55 kg is replaced by a new person,
    and the average weight of the group increases by 2.5 kg, prove that the weight of the new person is 75 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 55 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 75 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l4118_411891


namespace NUMINAMATH_CALUDE_sequence_properties_l4118_411885

def sequence_term (n : ℕ) : ℕ :=
  3 * (n^2 - n + 1)

theorem sequence_properties : 
  (∃ k, sequence_term k = 48 ∧ sequence_term (k + 1) = 63) ∧ 
  sequence_term 8 = 168 ∧
  sequence_term 2013 = 9120399 := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l4118_411885


namespace NUMINAMATH_CALUDE_original_number_proof_l4118_411800

theorem original_number_proof :
  ∃ N : ℕ, 
    (∃ k : ℤ, Odd (N * k) ∧ (N * k) % 9 = 0) ∧
    N * 4 = 108 ∧
    N = 27 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l4118_411800


namespace NUMINAMATH_CALUDE_two_year_increase_l4118_411848

/-- Given an initial amount that increases by 1/8th of itself each year,
    calculate the amount after a given number of years. -/
def amount_after_years (initial_amount : ℚ) (years : ℕ) : ℚ :=
  initial_amount * (1 + 1/8) ^ years

/-- Theorem: If an initial amount of 1600 increases by 1/8th of itself each year for two years,
    the final amount will be 2025. -/
theorem two_year_increase : amount_after_years 1600 2 = 2025 := by
  sorry

#eval amount_after_years 1600 2

end NUMINAMATH_CALUDE_two_year_increase_l4118_411848


namespace NUMINAMATH_CALUDE_ivan_share_increase_l4118_411827

theorem ivan_share_increase (p v s i : ℝ) 
  (h1 : p + v + s + i > 0)
  (h2 : 2*p + v + s + i = 1.3*(p + v + s + i))
  (h3 : p + 2*v + s + i = 1.25*(p + v + s + i))
  (h4 : p + v + 3*s + i = 1.5*(p + v + s + i)) :
  ∃ k : ℝ, k > 6 ∧ k*i > 0.6*(p + v + s + k*i) := by
  sorry

end NUMINAMATH_CALUDE_ivan_share_increase_l4118_411827


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l4118_411879

theorem binomial_coefficient_problem (m : ℕ+) 
  (a b : ℕ) 
  (ha : a = Nat.choose (2 * m) m)
  (hb : b = Nat.choose (2 * m + 1) m)
  (h_eq : 13 * a = 7 * b) : 
  m = 6 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l4118_411879


namespace NUMINAMATH_CALUDE_greatest_multiple_less_than_700_l4118_411897

theorem greatest_multiple_less_than_700 : ∃ n : ℕ, n = 680 ∧ 
  (∀ m : ℕ, m < 700 ∧ 5 ∣ m ∧ 4 ∣ m → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_less_than_700_l4118_411897


namespace NUMINAMATH_CALUDE_planes_through_skew_line_l4118_411890

/-- A structure representing a 3D space with lines and planes -/
structure Space3D where
  Line : Type
  Plane : Type
  in_plane : Line → Plane → Prop
  parallel : Plane → Line → Prop
  perpendicular : Plane → Plane → Prop
  skew : Line → Line → Prop

/-- The theorem statement -/
theorem planes_through_skew_line (S : Space3D) 
  (l m : S.Line) (α : S.Plane) 
  (h1 : S.skew l m) 
  (h2 : S.in_plane l α) : 
  (∃ (P : S.Plane), S.parallel P l ∧ ∃ (x : S.Line), S.in_plane x P ∧ x = m) ∧ 
  (∃ (Q : S.Plane), S.perpendicular Q α ∧ ∃ (y : S.Line), S.in_plane y Q ∧ y = m) := by
  sorry

end NUMINAMATH_CALUDE_planes_through_skew_line_l4118_411890


namespace NUMINAMATH_CALUDE_inequality_solution_l4118_411819

theorem inequality_solution : 
  {x : ℝ | (x - 2)^2 < 3*x + 4} = {x : ℝ | 0 ≤ x ∧ x < 7} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4118_411819


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l4118_411844

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 1), radius := 3 }
  let c2 : Circle := { center := (7, 0), radius := 2 }
  ∀ l : Line,
    (∃ p1 p2 : ℝ × ℝ,
      isTangent l c1 ∧
      isTangent l c2 ∧
      isInFirstQuadrant p1 ∧
      isInFirstQuadrant p2 ∧
      (p1.1 - 3)^2 + (p1.2 - 1)^2 = 3^2 ∧
      (p2.1 - 7)^2 + p2.2^2 = 2^2) →
    l.yIntercept = 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l4118_411844


namespace NUMINAMATH_CALUDE_divisibility_condition_l4118_411825

theorem divisibility_condition (m : ℕ) (h1 : m > 2022) 
  (h2 : (2022 + m) ∣ (2022 * m)) : m = 1011 ∨ m = 2022 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l4118_411825


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4118_411803

theorem expression_simplification_and_evaluation (a : ℚ) (h : a = 1/2) :
  (a - 1) / (a - 2) * ((a^2 - 4) / (a^2 - 2*a + 1)) - 2 / (a - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4118_411803


namespace NUMINAMATH_CALUDE_f_extrema_half_f_extrema_sum_gt_zero_l4118_411810

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + a * x) - 2 * x / (x + 2)

-- Theorem for part (1)
theorem f_extrema_half :
  let a : ℝ := 1/2
  ∃ (min_val : ℝ), (∀ x, x > -2 → f a x ≥ min_val) ∧
                   (∃ x, x > -2 ∧ f a x = min_val) ∧
                   min_val = Real.log 2 - 1 ∧
                   (∀ M, ∃ x, x > -2 ∧ f a x > M) :=
sorry

-- Theorem for part (2)
theorem f_extrema_sum_gt_zero (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : 1/2 < a ∧ a < 1) 
  (hx₁ : x₁ > -1/a ∧ (∀ y, y > -1/a → f a y ≤ f a x₁))
  (hx₂ : x₂ > -1/a ∧ (∀ y, y > -1/a → f a y ≤ f a x₂))
  (hd : x₁ ≠ x₂) :
  f a x₁ + f a x₂ > f a 0 :=
sorry

end NUMINAMATH_CALUDE_f_extrema_half_f_extrema_sum_gt_zero_l4118_411810


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l4118_411828

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total : Nat
  sample_size : Nat
  interval : Nat
  elements : Finset Nat

/-- Checks if a number is in the systematic sample -/
def in_sample (n : Nat) (s : SystematicSample) : Prop :=
  n ∈ s.elements

theorem systematic_sample_theorem (s : SystematicSample) 
  (h_total : s.total = 52)
  (h_sample_size : s.sample_size = 4)
  (h_interval : s.interval = 13)
  (h_6 : in_sample 6 s)
  (h_32 : in_sample 32 s)
  (h_45 : in_sample 45 s) :
  in_sample 19 s := by
  sorry

#check systematic_sample_theorem

end NUMINAMATH_CALUDE_systematic_sample_theorem_l4118_411828


namespace NUMINAMATH_CALUDE_production_days_calculation_l4118_411826

/-- Proves that the number of days is 4, given the conditions from the problem -/
theorem production_days_calculation (n : ℕ) : 
  (∀ (average_past : ℝ) (production_today : ℝ) (new_average : ℝ),
    average_past = 50 ∧
    production_today = 90 ∧
    new_average = 58 ∧
    (n : ℝ) * average_past + production_today = (n + 1 : ℝ) * new_average) →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_production_days_calculation_l4118_411826


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4118_411881

/-- Given that the solution set of ax² + bx + c > 0 is (-1, 3), 
    prove that the solution set of ax² - bx + c > 0 is (-3, 1) -/
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 3 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  Set.Ioo (-3 : ℝ) 1 = {x : ℝ | a * x^2 - b * x + c > 0} := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4118_411881


namespace NUMINAMATH_CALUDE_piggy_bank_coins_l4118_411858

/-- The number of dimes in a piggy bank containing quarters and dimes -/
def num_dimes : ℕ := sorry

/-- The number of quarters in the piggy bank -/
def num_quarters : ℕ := sorry

/-- The total number of coins in the piggy bank -/
def total_coins : ℕ := 100

/-- The total value of coins in the piggy bank in cents -/
def total_value : ℕ := 1975

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

theorem piggy_bank_coins :
  num_dimes = 35 ∧
  num_quarters + num_dimes = total_coins ∧
  num_quarters * quarter_value + num_dimes * dime_value = total_value :=
sorry

end NUMINAMATH_CALUDE_piggy_bank_coins_l4118_411858


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_rational_equation_l4118_411863

theorem negation_of_existence (P : ℚ → Prop) : 
  (¬ ∃ x : ℚ, P x) ↔ (∀ x : ℚ, ¬ P x) := by sorry

theorem negation_of_rational_equation : 
  (¬ ∃ x : ℚ, x - 2 = 0) ↔ (∀ x : ℚ, x - 2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_rational_equation_l4118_411863


namespace NUMINAMATH_CALUDE_f_seven_halves_eq_neg_sqrt_two_l4118_411838

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_seven_halves_eq_neg_sqrt_two
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_exp : ∀ x ∈ Set.Ioo 0 1, f x = 2^x) :
  f (7/2) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_seven_halves_eq_neg_sqrt_two_l4118_411838


namespace NUMINAMATH_CALUDE_larger_number_proof_l4118_411815

theorem larger_number_proof (S L : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 10) : 
  L = 1636 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l4118_411815


namespace NUMINAMATH_CALUDE_root_relation_l4118_411866

theorem root_relation (k : ℤ) : 
  (∃ x₁ x₂ : ℝ, x₁ = x₂ / 3 ∧ 
   4 * x₁^2 - (3*k + 2) * x₁ + (k^2 - 1) = 0 ∧
   4 * x₂^2 - (3*k + 2) * x₂ + (k^2 - 1) = 0) ↔ 
  k = 2 :=
sorry

end NUMINAMATH_CALUDE_root_relation_l4118_411866


namespace NUMINAMATH_CALUDE_total_soaking_time_l4118_411806

/-- Calculates the total soaking time for clothes with grass and marinara stains. -/
theorem total_soaking_time
  (grass_stain_time : ℕ)
  (marinara_stain_time : ℕ)
  (grass_stains : ℕ)
  (marinara_stains : ℕ)
  (h1 : grass_stain_time = 4)
  (h2 : marinara_stain_time = 7)
  (h3 : grass_stains = 3)
  (h4 : marinara_stains = 1) :
  grass_stain_time * grass_stains + marinara_stain_time * marinara_stains = 19 :=
by sorry

#check total_soaking_time

end NUMINAMATH_CALUDE_total_soaking_time_l4118_411806


namespace NUMINAMATH_CALUDE_second_machine_rate_l4118_411847

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Given two copy machines where the first makes 40 copies per minute,
    and together they make 2850 copies in half an hour,
    prove that the second machine makes 55 copies per minute -/
theorem second_machine_rate 
  (machine1 machine2 : CopyMachine)
  (h1 : machine1.copies_per_minute = 40)
  (h2 : machine1.copies_per_minute * 30 + machine2.copies_per_minute * 30 = 2850) :
  machine2.copies_per_minute = 55 := by
sorry

end NUMINAMATH_CALUDE_second_machine_rate_l4118_411847


namespace NUMINAMATH_CALUDE_jerry_walking_distance_l4118_411887

theorem jerry_walking_distance (monday_miles tuesday_miles : ℝ) 
  (h1 : monday_miles = tuesday_miles)
  (h2 : monday_miles + tuesday_miles = 18) : 
  monday_miles = 9 :=
by sorry

end NUMINAMATH_CALUDE_jerry_walking_distance_l4118_411887


namespace NUMINAMATH_CALUDE_store_distance_ratio_l4118_411884

/-- Represents the distances between locations in Jason's commute --/
structure CommuteDistances where
  house_to_first : ℝ
  first_to_second : ℝ
  second_to_third : ℝ
  third_to_work : ℝ

/-- Theorem stating the ratio of distances between stores --/
theorem store_distance_ratio (d : CommuteDistances) :
  d.house_to_first = 4 ∧
  d.first_to_second = 6 ∧
  d.third_to_work = 4 ∧
  d.second_to_third > d.first_to_second ∧
  d.house_to_first + d.first_to_second + d.second_to_third + d.third_to_work = 24 →
  d.second_to_third / d.first_to_second = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_store_distance_ratio_l4118_411884


namespace NUMINAMATH_CALUDE_quadratic_factorization_l4118_411805

theorem quadratic_factorization (c d : ℕ) (hc : c > d) :
  (∀ x, x^2 - 18*x + 72 = (x - c)*(x - d)) →
  4*d - c = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l4118_411805


namespace NUMINAMATH_CALUDE_equation_solution_l4118_411873

theorem equation_solution :
  let f : ℝ → ℝ := λ x => 2 * (x - 2)^2 - (6 - 3*x)
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 1/2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4118_411873


namespace NUMINAMATH_CALUDE_polynomial_square_decomposition_l4118_411833

theorem polynomial_square_decomposition (P : Polynomial ℝ) 
  (R : Polynomial ℝ) (h : P^2 = R.comp (Polynomial.X^2)) :
  ∃ Q : Polynomial ℝ, P = Q.comp (Polynomial.X^2) ∨ 
    P = Polynomial.X * Q.comp (Polynomial.X^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_square_decomposition_l4118_411833


namespace NUMINAMATH_CALUDE_about_set_S_l4118_411809

def S : Set ℤ := {x | ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2}

theorem about_set_S :
  (∀ x ∈ S, ¬(3 ∣ x)) ∧ (∃ x ∈ S, 11 ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_about_set_S_l4118_411809


namespace NUMINAMATH_CALUDE_set_operations_l4118_411855

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | x^2 - 2*x ≥ 0}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | -1 ≤ x ∧ x ≤ 0}) ∧
  (A ∪ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x < 2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l4118_411855


namespace NUMINAMATH_CALUDE_row_swap_matrix_l4118_411862

theorem row_swap_matrix (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]
  N * A = !![c, d; a, b] := by sorry

end NUMINAMATH_CALUDE_row_swap_matrix_l4118_411862


namespace NUMINAMATH_CALUDE_red_then_black_combinations_l4118_411883

def standard_deck : ℕ := 52
def red_cards : ℕ := 26
def black_cards : ℕ := 26

theorem red_then_black_combinations : 
  standard_deck = red_cards + black_cards →
  red_cards * black_cards = 676 := by
  sorry

end NUMINAMATH_CALUDE_red_then_black_combinations_l4118_411883


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l4118_411843

/-- The solution set of a quadratic inequality is empty iff the coefficient of x^2 is positive and the discriminant is non-positive -/
theorem quadratic_inequality_empty_solution_set 
  (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4*a*c ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l4118_411843


namespace NUMINAMATH_CALUDE_exponential_linear_critical_point_l4118_411865

/-- A function with a positive critical point -/
def has_positive_critical_point (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (deriv f) x = 0

/-- The main theorem -/
theorem exponential_linear_critical_point (a : ℝ) :
  has_positive_critical_point (fun x => Real.exp x + a * x) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_linear_critical_point_l4118_411865


namespace NUMINAMATH_CALUDE_not_twenty_percent_less_l4118_411877

theorem not_twenty_percent_less (a b : ℝ) (h : a = b * 1.2) : 
  ¬(b = a * 0.8) := by
  sorry

end NUMINAMATH_CALUDE_not_twenty_percent_less_l4118_411877


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4118_411889

theorem negation_of_universal_proposition (a : ℝ) :
  (¬ ∀ x > 0, Real.log x = a) ↔ (∃ x > 0, Real.log x ≠ a) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4118_411889


namespace NUMINAMATH_CALUDE_bryans_precious_stones_l4118_411830

theorem bryans_precious_stones (price_per_stone : ℕ) (total_amount : ℕ) (h1 : price_per_stone = 1785) (h2 : total_amount = 14280) :
  total_amount / price_per_stone = 8 := by
  sorry

end NUMINAMATH_CALUDE_bryans_precious_stones_l4118_411830


namespace NUMINAMATH_CALUDE_sugar_solution_replacement_l4118_411839

theorem sugar_solution_replacement (initial_sugar_percent : ℝ) 
                                   (final_sugar_percent : ℝ) 
                                   (second_sugar_percent : ℝ) 
                                   (replaced_portion : ℝ) : 
  initial_sugar_percent = 10 →
  final_sugar_percent = 16 →
  second_sugar_percent = 34 →
  (100 - replaced_portion) * initial_sugar_percent / 100 + 
    replaced_portion * second_sugar_percent / 100 = 
    final_sugar_percent →
  replaced_portion = 25 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_replacement_l4118_411839


namespace NUMINAMATH_CALUDE_correct_calculation_l4118_411816

theorem correct_calculation (a b : ℝ) : 4 * a^2 * b - 3 * b * a^2 = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l4118_411816


namespace NUMINAMATH_CALUDE_log_inequality_l4118_411812

theorem log_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  let a : ℝ := (Real.sqrt 2 + 1) / 2
  let f : ℝ → ℝ := fun x ↦ Real.log x / Real.log a
  f m > f n → m > n := by sorry

end NUMINAMATH_CALUDE_log_inequality_l4118_411812


namespace NUMINAMATH_CALUDE_acute_angle_between_l1_l2_l4118_411837

/-- The acute angle formed by the intersection of two lines in a 2D plane. -/
def acuteAngleBetweenLines (l1 l2 : ℝ → ℝ → Prop) : ℝ := sorry

/-- Line l1: √3x - y + 1 = 0 -/
def l1 (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0

/-- Line l2: x + 5 = 0 -/
def l2 (x y : ℝ) : Prop := x + 5 = 0

/-- The acute angle formed by the intersection of l1 and l2 is 30° -/
theorem acute_angle_between_l1_l2 : acuteAngleBetweenLines l1 l2 = 30 * Real.pi / 180 := by sorry

end NUMINAMATH_CALUDE_acute_angle_between_l1_l2_l4118_411837


namespace NUMINAMATH_CALUDE_ava_mia_difference_l4118_411886

/-- The number of shells each person has -/
structure ShellCounts where
  david : ℕ
  mia : ℕ
  ava : ℕ
  alice : ℕ

/-- The conditions of the problem -/
def problem_conditions (counts : ShellCounts) : Prop :=
  counts.david = 15 ∧
  counts.mia = 4 * counts.david ∧
  counts.ava > counts.mia ∧
  counts.alice = counts.ava / 2 ∧
  counts.david + counts.mia + counts.ava + counts.alice = 195

/-- The theorem to prove -/
theorem ava_mia_difference (counts : ShellCounts) :
  problem_conditions counts → counts.ava - counts.mia = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_ava_mia_difference_l4118_411886


namespace NUMINAMATH_CALUDE_stock_dividend_rate_l4118_411802

/-- Given a stock with a certain yield and price, calculate its dividend rate. -/
def dividend_rate (yield : ℝ) (price : ℝ) : ℝ :=
  yield * price

/-- Theorem: The dividend rate of a stock yielding 8% quoted at 150 is 12. -/
theorem stock_dividend_rate :
  let yield : ℝ := 0.08
  let price : ℝ := 150
  dividend_rate yield price = 12 := by
  sorry

end NUMINAMATH_CALUDE_stock_dividend_rate_l4118_411802
