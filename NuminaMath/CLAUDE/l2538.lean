import Mathlib

namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l2538_253813

theorem smallest_m_for_integral_solutions : 
  (∃ m : ℕ, m > 0 ∧ 
   (∃ x y : ℤ, 10 * x^2 - m * x + 1980 = 0 ∧ 10 * y^2 - m * y + 1980 = 0 ∧ x ≠ y) ∧
   (∀ k : ℕ, k > 0 ∧ k < m → 
     ¬∃ x y : ℤ, 10 * x^2 - k * x + 1980 = 0 ∧ 10 * y^2 - k * y + 1980 = 0 ∧ x ≠ y)) ∧
  (∀ m : ℕ, m > 0 ∧ 
   (∃ x y : ℤ, 10 * x^2 - m * x + 1980 = 0 ∧ 10 * y^2 - m * y + 1980 = 0 ∧ x ≠ y) ∧
   (∀ k : ℕ, k > 0 ∧ k < m → 
     ¬∃ x y : ℤ, 10 * x^2 - k * x + 1980 = 0 ∧ 10 * y^2 - k * y + 1980 = 0 ∧ x ≠ y) →
   m = 290) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l2538_253813


namespace NUMINAMATH_CALUDE_problem_solution_l2538_253810

theorem problem_solution (x : ℝ) : x * 120 = 346 → x = 346 / 120 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2538_253810


namespace NUMINAMATH_CALUDE_cone_volume_l2538_253857

/-- Given a cone with slant height 15 cm and height 9 cm, its volume is 432π cubic centimeters -/
theorem cone_volume (s h r : ℝ) (hs : s = 15) (hh : h = 9) 
  (hr : r^2 = s^2 - h^2) : (1/3 : ℝ) * π * r^2 * h = 432 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l2538_253857


namespace NUMINAMATH_CALUDE_johns_breakfast_calories_l2538_253842

/-- Represents the number of calories in John's breakfast -/
def breakfast_calories : ℝ := 500

/-- Represents the number of calories in John's lunch -/
def lunch_calories : ℝ := 1.25 * breakfast_calories

/-- Represents the number of calories in John's dinner -/
def dinner_calories : ℝ := 2 * lunch_calories

/-- Represents the total number of calories from shakes -/
def shake_calories : ℝ := 3 * 300

/-- Represents the total number of calories John consumes in a day -/
def total_calories : ℝ := 3275

/-- Theorem stating that given the conditions, John's breakfast contains 500 calories -/
theorem johns_breakfast_calories :
  breakfast_calories + lunch_calories + dinner_calories + shake_calories = total_calories :=
by sorry

end NUMINAMATH_CALUDE_johns_breakfast_calories_l2538_253842


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2538_253831

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x > 5}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 2 < x ∧ x ≤ 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2538_253831


namespace NUMINAMATH_CALUDE_plywood_perimeter_l2538_253811

theorem plywood_perimeter (length width perimeter : ℝ) : 
  length = 6 → width = 5 → perimeter = 2 * (length + width) → perimeter = 22 := by
  sorry

end NUMINAMATH_CALUDE_plywood_perimeter_l2538_253811


namespace NUMINAMATH_CALUDE_mitch_spare_candy_bars_l2538_253886

/-- Proves that Mitch wants to have 10 spare candy bars --/
theorem mitch_spare_candy_bars : 
  let bars_per_friend : ℕ := 2
  let total_bars : ℕ := 24
  let num_friends : ℕ := 7
  let spare_bars : ℕ := total_bars - (bars_per_friend * num_friends)
  spare_bars = 10 := by sorry

end NUMINAMATH_CALUDE_mitch_spare_candy_bars_l2538_253886


namespace NUMINAMATH_CALUDE_max_a_when_a_squared_plus_100a_prime_l2538_253852

theorem max_a_when_a_squared_plus_100a_prime (a : ℕ+) :
  Nat.Prime (a^2 + 100*a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_a_when_a_squared_plus_100a_prime_l2538_253852


namespace NUMINAMATH_CALUDE_seventh_term_ratio_l2538_253891

/-- Two arithmetic sequences with sums S and T for their first n terms. -/
def ArithmeticSequences (S T : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n / T n = (5 * n + 10) / (2 * n - 1)

/-- The 7th term of an arithmetic sequence. -/
def seventhTerm (seq : ℕ → ℚ) : ℚ := seq 7

theorem seventh_term_ratio (S T : ℕ → ℚ) (h : ArithmeticSequences S T) :
  seventhTerm S / seventhTerm T = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_ratio_l2538_253891


namespace NUMINAMATH_CALUDE_inverse_mod_101_l2538_253869

theorem inverse_mod_101 (h : (7⁻¹ : ZMod 101) = 55) : (49⁻¹ : ZMod 101) = 96 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_101_l2538_253869


namespace NUMINAMATH_CALUDE_marias_test_scores_l2538_253880

/-- Maria's test scores problem -/
theorem marias_test_scores 
  (score1 score2 score3 : ℝ) 
  (h1 : (score1 + score2 + score3 + 100) / 4 = 85) :
  score1 + score2 + score3 = 240 := by
  sorry

end NUMINAMATH_CALUDE_marias_test_scores_l2538_253880


namespace NUMINAMATH_CALUDE_solution_count_l2538_253817

/-- The number of pairs of positive integers (x, y) that satisfy x^2 - y^2 = 45 -/
def count_solutions : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let (x, y) := p
    x > 0 ∧ y > 0 ∧ x^2 - y^2 = 45
  ) (Finset.product (Finset.range 46) (Finset.range 46))).card

theorem solution_count : count_solutions = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_l2538_253817


namespace NUMINAMATH_CALUDE_unique_line_intersection_l2538_253895

theorem unique_line_intersection (m b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! k : ℝ, ∃ y1 y2 : ℝ, 
    y1 = k^2 - 2*k + 3 ∧ 
    y2 = m*k + b ∧ 
    |y1 - y2| = 4)
  (h3 : m * 2 + b = 8) : 
  m = 0 ∧ b = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_line_intersection_l2538_253895


namespace NUMINAMATH_CALUDE_expression_evaluation_l2538_253839

theorem expression_evaluation (a : ℝ) (h : a = Real.sqrt 5 + 1) :
  a / (a^2 - 2*a + 1) / (1 + 1/(a - 1)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2538_253839


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2538_253812

theorem sphere_surface_area (d : ℝ) (h : d = 2) : 
  4 * Real.pi * (d / 2)^2 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2538_253812


namespace NUMINAMATH_CALUDE_segment_division_problem_l2538_253888

/-- The problem of determining the number of parts a unit segment is divided into -/
theorem segment_division_problem (min_distance : ℚ) (h1 : min_distance = 0.02857142857142857) : 
  (1 : ℚ) / min_distance = 35 := by
  sorry

end NUMINAMATH_CALUDE_segment_division_problem_l2538_253888


namespace NUMINAMATH_CALUDE_remainder_of_3_power_100_plus_5_mod_11_l2538_253871

theorem remainder_of_3_power_100_plus_5_mod_11 : (3^100 + 5) % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_power_100_plus_5_mod_11_l2538_253871


namespace NUMINAMATH_CALUDE_floor_neg_seven_fourths_l2538_253834

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_seven_fourths_l2538_253834


namespace NUMINAMATH_CALUDE_multiples_of_five_l2538_253845

theorem multiples_of_five (a b : ℤ) (ha : 5 ∣ a) (hb : 10 ∣ b) : 5 ∣ b ∧ 5 ∣ (a - b) := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_five_l2538_253845


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_points_l2538_253854

/-- The ellipse on which points A and B lie -/
def Ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The condition that OA is perpendicular to OB -/
def Perpendicular (A B : ℝ × ℝ) : Prop := A.1 * B.1 + A.2 * B.2 = 0

/-- The condition that P is on segment AB -/
def OnSegment (P A B : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

/-- The condition that OP is perpendicular to AB -/
def Perpendicular_OP_AB (O P A B : ℝ × ℝ) : Prop := 
  (P.1 - O.1) * (B.1 - A.1) + (P.2 - O.2) * (B.2 - A.2) = 0

/-- The main theorem -/
theorem ellipse_perpendicular_points 
  (A B : ℝ × ℝ) 
  (hA : Ellipse A.1 A.2) 
  (hB : Ellipse B.1 B.2) 
  (hPerp : Perpendicular A B)
  (P : ℝ × ℝ)
  (hP : OnSegment P A B)
  (hPPerp : Perpendicular_OP_AB (0, 0) P A B) :
  (1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) = 13/36) ∧
  (P.1^2 + P.2^2 = (6 * Real.sqrt 13 / 13)^2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_points_l2538_253854


namespace NUMINAMATH_CALUDE_large_number_arithmetic_l2538_253815

theorem large_number_arithmetic : 
  999999999999 - 888888888888 + 111111111111 = 222222222222 := by
  sorry

end NUMINAMATH_CALUDE_large_number_arithmetic_l2538_253815


namespace NUMINAMATH_CALUDE_max_floor_product_sum_l2538_253836

theorem max_floor_product_sum (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → x + y + z = 1399 →
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1399 →
  ⌊x⌋ * y + ⌊y⌋ * z + ⌊z⌋ * x ≤ ⌊a⌋ * b + ⌊b⌋ * c + ⌊c⌋ * a →
  ⌊a⌋ * b + ⌊b⌋ * c + ⌊c⌋ * a ≤ 652400 :=
by sorry

#check max_floor_product_sum

end NUMINAMATH_CALUDE_max_floor_product_sum_l2538_253836


namespace NUMINAMATH_CALUDE_line_segment_length_l2538_253853

/-- Given points P, Q, R, and S arranged in order on a line segment,
    with PQ = 1, QR = 2PQ, and RS = 3QR, prove that the length of PS is 9. -/
theorem line_segment_length (P Q R S : ℝ) : 
  P < Q ∧ Q < R ∧ R < S →  -- Points are arranged in order
  Q - P = 1 →              -- PQ = 1
  R - Q = 2 * (Q - P) →    -- QR = 2PQ
  S - R = 3 * (R - Q) →    -- RS = 3QR
  S - P = 9 :=             -- PS = 9
by sorry

end NUMINAMATH_CALUDE_line_segment_length_l2538_253853


namespace NUMINAMATH_CALUDE_divisibility_by_36_l2538_253819

theorem divisibility_by_36 : ∃! n : ℕ, n < 10 ∧ (6130 + n) % 36 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_divisibility_by_36_l2538_253819


namespace NUMINAMATH_CALUDE_product_purely_imaginary_l2538_253862

theorem product_purely_imaginary (x : ℝ) : 
  (Complex.I : ℂ).im * ((x + Complex.I) * ((x^2 + 1 : ℝ) + Complex.I) * ((x^2 + 2 : ℝ) + Complex.I)).re = 0 ↔ 
  x^4 + x^3 + x^2 + 2*x - 2 = 0 :=
by sorry

#check product_purely_imaginary

end NUMINAMATH_CALUDE_product_purely_imaginary_l2538_253862


namespace NUMINAMATH_CALUDE_cone_lateral_area_l2538_253897

/-- The lateral area of a cone with base radius 6 cm and height 8 cm is 60π cm². -/
theorem cone_lateral_area : 
  let r : ℝ := 6  -- base radius in cm
  let h : ℝ := 8  -- height in cm
  let l : ℝ := (r^2 + h^2).sqrt  -- slant height
  let lateral_area : ℝ := π * r * l  -- formula for lateral area
  lateral_area = 60 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l2538_253897


namespace NUMINAMATH_CALUDE_ac_length_is_18_l2538_253865

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- AB length is 12 -/
  ab_length : dist A B = 12
  /-- AD length is 8 -/
  ad_length : dist A D = 8
  /-- DC length is 18 -/
  dc_length : dist D C = 18
  /-- AD is perpendicular to AB -/
  ad_perp_ab : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0
  /-- ABCD is symmetric about AC -/
  symmetric_about_ac : ∃ (m : ℝ) (b : ℝ), 
    (C.2 - A.2) = m * (C.1 - A.1) ∧
    B.2 - A.2 = m * (B.1 - A.1) + b ∧
    D.2 - A.2 = -(m * (D.1 - A.1) + b)

/-- The length of AC in a SpecialQuadrilateral is 18 -/
theorem ac_length_is_18 (q : SpecialQuadrilateral) : dist q.A q.C = 18 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_is_18_l2538_253865


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l2538_253883

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℝ  -- first term
  y : ℝ  -- second term
  b : ℝ  -- third term
  d : ℝ  -- common difference
  h1 : y = a + d  -- relation between y, a, and d
  h2 : b = a + 3 * d  -- relation between b, a, and d
  h3 : y / 2 = a + 3 * d  -- fourth term equals y/2

/-- The ratio of a to b in the given arithmetic sequence is 3/4 -/
theorem ratio_a_to_b (seq : ArithmeticSequence) : seq.a / seq.b = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l2538_253883


namespace NUMINAMATH_CALUDE_cos_10_cos_20_minus_sin_10_sin_20_l2538_253846

theorem cos_10_cos_20_minus_sin_10_sin_20 :
  Real.cos (10 * π / 180) * Real.cos (20 * π / 180) -
  Real.sin (10 * π / 180) * Real.sin (20 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_10_cos_20_minus_sin_10_sin_20_l2538_253846


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l2538_253859

/-- Given a grocery store inventory, calculate the number of diet soda bottles. -/
theorem diet_soda_bottles (total_bottles regular_soda_bottles : ℕ) 
  (h1 : total_bottles = 38) 
  (h2 : regular_soda_bottles = 30) : 
  total_bottles - regular_soda_bottles = 8 := by
  sorry

end NUMINAMATH_CALUDE_diet_soda_bottles_l2538_253859


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2538_253803

theorem division_remainder_problem (larger smaller : ℕ) : 
  larger - smaller = 1515 →
  larger = 1600 →
  larger / smaller = 16 →
  larger % smaller = 240 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2538_253803


namespace NUMINAMATH_CALUDE_product_with_9999_l2538_253850

theorem product_with_9999 (n : ℕ) : n * 9999 = 4691130840 → n = 469200 := by
  sorry

end NUMINAMATH_CALUDE_product_with_9999_l2538_253850


namespace NUMINAMATH_CALUDE_square_ratio_sum_l2538_253889

theorem square_ratio_sum (area_ratio : ℚ) (a b c : ℕ) : 
  area_ratio = 300 / 75 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt area_ratio →
  a + b + c = 4 := by
sorry

end NUMINAMATH_CALUDE_square_ratio_sum_l2538_253889


namespace NUMINAMATH_CALUDE_cats_sold_during_sale_l2538_253821

/-- Represents the number of cats sold during a sale at a pet store. -/
def cats_sold (siamese_initial : ℕ) (house_initial : ℕ) (cats_left : ℕ) : ℕ :=
  siamese_initial + house_initial - cats_left

/-- Theorem stating that 19 cats were sold during the sale. -/
theorem cats_sold_during_sale :
  cats_sold 15 49 45 = 19 := by
  sorry

end NUMINAMATH_CALUDE_cats_sold_during_sale_l2538_253821


namespace NUMINAMATH_CALUDE_factor_y6_minus_64_l2538_253866

theorem factor_y6_minus_64 (y : ℝ) : 
  y^6 - 64 = (y - 2) * (y + 2) * (y^2 + 2*y + 4) * (y^2 - 2*y + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_y6_minus_64_l2538_253866


namespace NUMINAMATH_CALUDE_min_total_cost_l2538_253802

/-- Represents a dish with its price and quantity -/
structure Dish where
  price : ℕ
  quantity : ℕ

/-- Calculates the total price of a dish -/
def dishTotal (d : Dish) : ℕ := d.price * d.quantity

/-- Applies discount to an order based on its total -/
def applyDiscount (total : ℕ) : ℕ :=
  if total > 100 then total - 45
  else if total > 60 then total - 30
  else if total > 30 then total - 12
  else total

/-- Calculates the final cost of an order including delivery fee -/
def orderCost (total : ℕ) : ℕ := applyDiscount total + 3

/-- Theorem: The minimum total cost for Xiaoyu's order is 54 -/
theorem min_total_cost (dishes : List Dish) 
  (h1 : dishes = [
    ⟨30, 1⟩, -- Boiled Beef
    ⟨12, 1⟩, -- Vinegar Potatoes
    ⟨30, 1⟩, -- Spare Ribs in Black Bean Sauce
    ⟨12, 1⟩, -- Hand-Torn Cabbage
    ⟨3, 2⟩   -- Rice
  ]) :
  (dishes.map dishTotal).sum = 90 →
  ∃ (order1 order2 : ℕ), 
    order1 + order2 = 90 ∧ 
    orderCost order1 + orderCost order2 = 54 ∧
    ∀ (split1 split2 : ℕ), 
      split1 + split2 = 90 → 
      orderCost split1 + orderCost split2 ≥ 54 := by
  sorry

end NUMINAMATH_CALUDE_min_total_cost_l2538_253802


namespace NUMINAMATH_CALUDE_laptop_price_difference_l2538_253877

/-- The list price of Laptop Y -/
def list_price : ℝ := 69.80

/-- The discount percentage at Tech Giant -/
def tech_giant_discount : ℝ := 0.15

/-- The fixed discount amount at EconoTech -/
def econotech_discount : ℝ := 10

/-- The sale price at Tech Giant -/
def tech_giant_price : ℝ := list_price * (1 - tech_giant_discount)

/-- The sale price at EconoTech -/
def econotech_price : ℝ := list_price - econotech_discount

/-- The price difference between EconoTech and Tech Giant in dollars -/
def price_difference : ℝ := econotech_price - tech_giant_price

theorem laptop_price_difference :
  ⌊price_difference * 100⌋ = 47 := by sorry

end NUMINAMATH_CALUDE_laptop_price_difference_l2538_253877


namespace NUMINAMATH_CALUDE_positive_real_properties_l2538_253838

theorem positive_real_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 * b - a * b = 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 4 * y - x * y = 0 ∧ x + 4 * y < a + 4 * b) →
  (a + 2 * b ≥ 6 + 4 * Real.sqrt 2) ∧
  (16 / a^2 + 1 / b^2 ≥ 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_positive_real_properties_l2538_253838


namespace NUMINAMATH_CALUDE_larger_number_given_hcf_lcm_factors_l2538_253851

theorem larger_number_given_hcf_lcm_factors (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  Nat.gcd a b = 63 ∧ 
  ∃ (k : ℕ), Nat.lcm a b = 63 * 11 * 17 * k →
  max a b = 1071 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_given_hcf_lcm_factors_l2538_253851


namespace NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l2538_253873

theorem pythagorean_triple_divisibility (x y z : ℕ) (h : x^2 + y^2 = z^2) :
  3 ∣ x ∨ 3 ∣ y ∨ 3 ∣ z := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l2538_253873


namespace NUMINAMATH_CALUDE_line_properties_l2538_253833

-- Define the line l₁
def l₁ (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y - 8 = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (2, 2)

-- Define the line l₂
def l₂ (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y = 0

-- Define the maximized distance line
def max_distance_line (x y : ℝ) : Prop :=
  x + y = 0

theorem line_properties :
  (∀ m : ℝ, l₁ m (fixed_point.1) (fixed_point.2)) ∧
  (∃ m : ℝ, ∀ x y : ℝ, l₂ m x y ↔ max_distance_line x y) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l2538_253833


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2538_253881

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + 3 * a 6 + a 11 = 10) →
  a 5 + a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2538_253881


namespace NUMINAMATH_CALUDE_exists_x0_sin_minus_tan_negative_l2538_253843

open Real

theorem exists_x0_sin_minus_tan_negative :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < π/2 ∧ sin x₀ - tan x₀ < 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x0_sin_minus_tan_negative_l2538_253843


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l2538_253837

/-- A point on a parabola with a specific distance to the focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y ^ 2 = 12 * x
  distance_to_focus : 6 = |x + 3| -- The focus is at (3, 0)

/-- The coordinates of a point on the parabola y² = 12x with distance 6 to the focus -/
theorem parabola_point_coordinates (p : ParabolaPoint) : 
  (p.x = 3 ∧ p.y = 6) ∨ (p.x = 3 ∧ p.y = -6) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l2538_253837


namespace NUMINAMATH_CALUDE_equation_is_linear_l2538_253800

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants -/
def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y + c

/-- The equation x - 3y = -15 -/
def equation (x y : ℝ) : ℝ := x - 3 * y + 15

theorem equation_is_linear :
  is_linear_equation_in_two_variables equation := by
sorry

end NUMINAMATH_CALUDE_equation_is_linear_l2538_253800


namespace NUMINAMATH_CALUDE_visibility_time_correct_l2538_253830

/-- Represents a person walking along a straight path -/
structure Walker where
  speed : ℝ
  initial_position : ℝ × ℝ

/-- Represents the circular building -/
structure Building where
  center : ℝ × ℝ
  radius : ℝ

/-- The scenario of Jenny and Kenny walking -/
def walking_scenario : Building × Walker × Walker := 
  let building := { center := (0, 0), radius := 50 }
  let jenny := { speed := 2, initial_position := (-150, 100) }
  let kenny := { speed := 4, initial_position := (-150, -100) }
  (building, jenny, kenny)

/-- The time when Jenny and Kenny can see each other again -/
noncomputable def visibility_time (scenario : Building × Walker × Walker) : ℝ := 
  200  -- This is the value we want to prove

/-- The theorem stating that the visibility time is correct -/
theorem visibility_time_correct :
  let (building, jenny, kenny) := walking_scenario
  let t := visibility_time walking_scenario
  
  -- At time t, the line connecting Jenny and Kenny is tangent to the building
  ∃ (x y : ℝ),
    (x^2 + y^2 = building.radius^2) ∧
    ((jenny.initial_position.1 + jenny.speed * t - x) * (kenny.initial_position.2 - y) =
     (kenny.initial_position.1 + kenny.speed * t - x) * (jenny.initial_position.2 - y)) ∧
    (x * (jenny.initial_position.2 - y) + y * (x - jenny.initial_position.1 - jenny.speed * t) = 0) :=
by sorry

end NUMINAMATH_CALUDE_visibility_time_correct_l2538_253830


namespace NUMINAMATH_CALUDE_other_root_is_one_l2538_253814

/-- Given a quadratic function f(x) = x^2 + 2x - a with a root of -3, 
    prove that the other root is 1. -/
theorem other_root_is_one (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 + 2*x - a) 
    (h2 : f (-3) = 0) : 
  ∃ x, x ≠ -3 ∧ f x = 0 ∧ x = 1 := by
sorry

end NUMINAMATH_CALUDE_other_root_is_one_l2538_253814


namespace NUMINAMATH_CALUDE_overtime_rate_increase_l2538_253822

def regular_rate : ℚ := 16
def regular_hours : ℕ := 40
def total_compensation : ℚ := 1340
def total_hours : ℕ := 65

theorem overtime_rate_increase :
  let overtime_hours : ℕ := total_hours - regular_hours
  let regular_pay : ℚ := regular_rate * regular_hours
  let overtime_pay : ℚ := total_compensation - regular_pay
  let overtime_rate : ℚ := overtime_pay / overtime_hours
  let rate_increase : ℚ := (overtime_rate - regular_rate) / regular_rate
  rate_increase = 3/4 := by sorry

end NUMINAMATH_CALUDE_overtime_rate_increase_l2538_253822


namespace NUMINAMATH_CALUDE_position_of_2010_l2538_253816

/-- The row number for a given positive integer in the arrangement -/
def row (n : ℕ) : ℕ := 
  (n.sqrt : ℕ) + (if n > (n.sqrt : ℕ)^2 then 1 else 0)

/-- The column number for a given positive integer in the arrangement -/
def column (n : ℕ) : ℕ := 
  n - (row n - 1)^2

/-- The theorem stating that 2010 appears in row 45 and column 74 -/
theorem position_of_2010 : row 2010 = 45 ∧ column 2010 = 74 := by
  sorry

end NUMINAMATH_CALUDE_position_of_2010_l2538_253816


namespace NUMINAMATH_CALUDE_existence_of_special_real_l2538_253861

theorem existence_of_special_real : ∃ A : ℝ, ∀ n : ℕ, ∃ m : ℕ, (⌊A^n⌋ : ℤ) + 2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_real_l2538_253861


namespace NUMINAMATH_CALUDE_cheryl_material_problem_l2538_253863

theorem cheryl_material_problem (x : ℚ) :
  -- Cheryl buys x square yards of first material and 1/3 of second
  -- After project, 15/40 square yards left unused
  -- Total amount used is 1/3 square yards
  (x + 1/3 - 15/40 = 1/3) →
  -- The amount of first material needed is 3/8 square yards
  x = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_material_problem_l2538_253863


namespace NUMINAMATH_CALUDE_number_ordering_l2538_253868

theorem number_ordering : (5 : ℝ) / 2 < 3 ∧ 3 < Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l2538_253868


namespace NUMINAMATH_CALUDE_equation_solution_l2538_253860

theorem equation_solution : ∃! x : ℝ, (1 / 6 + 6 / x = 10 / x + 1 / 15) ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2538_253860


namespace NUMINAMATH_CALUDE_knitting_productivity_comparison_l2538_253872

/-- Represents a knitter with their working time and break time -/
structure Knitter where
  workTime : ℕ
  breakTime : ℕ

/-- Calculates the total cycle time for a knitter -/
def cycleTime (k : Knitter) : ℕ := k.workTime + k.breakTime

/-- Calculates the number of complete cycles in a given time -/
def completeCycles (k : Knitter) (totalTime : ℕ) : ℕ :=
  totalTime / cycleTime k

/-- Calculates the total working time within a given time period -/
def totalWorkTime (k : Knitter) (totalTime : ℕ) : ℕ :=
  completeCycles k totalTime * k.workTime

theorem knitting_productivity_comparison : 
  let girl1 : Knitter := ⟨5, 1⟩
  let girl2 : Knitter := ⟨7, 1⟩
  let commonBreakTime := lcm (cycleTime girl1) (cycleTime girl2)
  totalWorkTime girl1 commonBreakTime * 21 = totalWorkTime girl2 commonBreakTime * 20 := by
  sorry

end NUMINAMATH_CALUDE_knitting_productivity_comparison_l2538_253872


namespace NUMINAMATH_CALUDE_coordinates_wrt_y_axis_l2538_253856

/-- Given a point A(x,y) in a Cartesian coordinate system, 
    its coordinates with respect to the y-axis are (-x,y) -/
theorem coordinates_wrt_y_axis (x y : ℝ) : 
  let A : ℝ × ℝ := (x, y)
  let A_wrt_y_axis : ℝ × ℝ := (-x, y)
  A_wrt_y_axis = (- (A.1), A.2) :=
by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_y_axis_l2538_253856


namespace NUMINAMATH_CALUDE_total_donation_l2538_253892

def cassandra_pennies : ℕ := 5000
def james_pennies : ℕ := cassandra_pennies - 276
def stephanie_pennies : ℕ := 2 * james_pennies

theorem total_donation :
  cassandra_pennies + james_pennies + stephanie_pennies = 19172 :=
by sorry

end NUMINAMATH_CALUDE_total_donation_l2538_253892


namespace NUMINAMATH_CALUDE_line_equation_correctness_l2538_253870

/-- A line passing through a point with a given direction vector -/
structure DirectedLine (n : ℕ) where
  point : Fin n → ℝ
  direction : Fin n → ℝ

/-- The equation of a line in 2D space -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfiesEquation (p : Fin 2 → ℝ) (eq : LineEquation) : Prop :=
  eq.a * p 0 + eq.b * p 1 + eq.c = 0

/-- Check if a vector is parallel to a line equation -/
def isParallel (v : Fin 2 → ℝ) (eq : LineEquation) : Prop :=
  eq.a * v 0 + eq.b * v 1 = 0

theorem line_equation_correctness (l : DirectedLine 2) (eq : LineEquation) :
  (l.point 0 = -3 ∧ l.point 1 = 1) →
  (l.direction 0 = 2 ∧ l.direction 1 = -3) →
  (eq.a = 3 ∧ eq.b = 2 ∧ eq.c = -11) →
  satisfiesEquation l.point eq ∧ isParallel l.direction eq :=
sorry

end NUMINAMATH_CALUDE_line_equation_correctness_l2538_253870


namespace NUMINAMATH_CALUDE_min_pizzas_cover_expenses_l2538_253890

/-- Represents the minimum number of pizzas John must deliver to cover his expenses -/
def min_pizzas : ℕ := 1063

/-- Represents the cost of the used car -/
def car_cost : ℕ := 8000

/-- Represents the upfront maintenance cost -/
def maintenance_cost : ℕ := 500

/-- Represents the earnings per pizza delivered -/
def earnings_per_pizza : ℕ := 12

/-- Represents the gas cost per delivery -/
def gas_cost_per_delivery : ℕ := 4

/-- Represents the net earnings per pizza (earnings minus gas cost) -/
def net_earnings_per_pizza : ℕ := earnings_per_pizza - gas_cost_per_delivery

theorem min_pizzas_cover_expenses :
  (min_pizzas : ℝ) * net_earnings_per_pizza ≥ car_cost + maintenance_cost :=
sorry

end NUMINAMATH_CALUDE_min_pizzas_cover_expenses_l2538_253890


namespace NUMINAMATH_CALUDE_smallest_n_with_properties_l2538_253823

theorem smallest_n_with_properties : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (a : ℕ), 3 * n = a^2) ∧ 
  (∃ (b : ℕ), 2 * n = b^3) ∧ 
  (∃ (c : ℕ), 5 * n = c^5) ∧ 
  (∀ (m : ℕ), m > 0 → 
    ((∃ (x : ℕ), 3 * m = x^2) ∧ 
     (∃ (y : ℕ), 2 * m = y^3) ∧ 
     (∃ (z : ℕ), 5 * m = z^5)) → 
    m ≥ 7500) ∧
  n = 7500 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_properties_l2538_253823


namespace NUMINAMATH_CALUDE_prob_white_ball_l2538_253884

/-- Represents an urn with a certain number of black and white balls -/
structure Urn :=
  (black : ℕ)
  (white : ℕ)

/-- The probability of choosing each urn -/
def urn_choice_prob : ℚ := 1/2

/-- The two urns in the problem -/
def urn1 : Urn := ⟨2, 3⟩
def urn2 : Urn := ⟨2, 1⟩

/-- The probability of drawing a white ball from a given urn -/
def prob_white (u : Urn) : ℚ :=
  u.white / (u.black + u.white)

/-- The theorem stating the probability of drawing a white ball -/
theorem prob_white_ball : 
  urn_choice_prob * prob_white urn1 + urn_choice_prob * prob_white urn2 = 7/15 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_ball_l2538_253884


namespace NUMINAMATH_CALUDE_f_properties_l2538_253825

noncomputable def f (a b x : ℝ) : ℝ := Real.log x - a * x + b

theorem f_properties (a b : ℝ) (ha : a > 0) 
  (hf : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a b y ≤ f a b x) ∧
  (∀ (x₁ x₂ : ℝ), f a b x₁ = 0 → f a b x₂ = 0 → x₁ * x₂ < 1 / (a^2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2538_253825


namespace NUMINAMATH_CALUDE_larger_number_proof_l2538_253864

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1365) (h3 : L = 6 * S + 5) : L = 1637 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2538_253864


namespace NUMINAMATH_CALUDE_percentage_difference_l2538_253824

theorem percentage_difference (x y : ℝ) (P : ℝ) (h1 : x = y - (P / 100) * y) (h2 : y = 2 * x) :
  P = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l2538_253824


namespace NUMINAMATH_CALUDE_peter_calories_l2538_253885

/-- Represents the number of calories Peter wants to eat -/
def calories_wanted (chip_calories : ℕ) (chips_per_bag : ℕ) (bag_cost : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent / bag_cost) * chips_per_bag * chip_calories

/-- Proves that Peter wants to eat 480 calories worth of chips -/
theorem peter_calories : calories_wanted 10 24 2 4 = 480 := by
  sorry

end NUMINAMATH_CALUDE_peter_calories_l2538_253885


namespace NUMINAMATH_CALUDE_ball_count_difference_l2538_253887

theorem ball_count_difference (total : ℕ) (white : ℕ) : 
  total = 100 →
  white = 16 →
  ∃ (blue red : ℕ),
    blue > white ∧
    red = 2 * blue ∧
    red + blue + white = total ∧
    blue - white = 12 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_difference_l2538_253887


namespace NUMINAMATH_CALUDE_swimming_club_members_l2538_253874

theorem swimming_club_members :
  ∃ (j s v : ℕ),
    j > 0 ∧ s > 0 ∧ v > 0 ∧
    3 * s = 2 * j ∧
    5 * v = 2 * s ∧
    j + s + v = 58 :=
by sorry

end NUMINAMATH_CALUDE_swimming_club_members_l2538_253874


namespace NUMINAMATH_CALUDE_mixture_capacity_l2538_253808

/-- Represents the capacity and alcohol percentage of a vessel -/
structure Vessel where
  capacity : ℝ
  alcoholPercentage : ℝ

/-- Represents the mixture of two vessels -/
def Mixture (v1 v2 : Vessel) : ℝ × ℝ :=
  (v1.capacity + v2.capacity, v1.capacity * v1.alcoholPercentage + v2.capacity * v2.alcoholPercentage)

theorem mixture_capacity (v1 v2 : Vessel) (newConcentration : ℝ) :
  v1.capacity = 3 →
  v1.alcoholPercentage = 0.25 →
  v2.capacity = 5 →
  v2.alcoholPercentage = 0.40 →
  (Mixture v1 v2).1 = 8 →
  newConcentration = 0.275 →
  (Mixture v1 v2).2 / newConcentration = 10 := by
  sorry

#check mixture_capacity

end NUMINAMATH_CALUDE_mixture_capacity_l2538_253808


namespace NUMINAMATH_CALUDE_divisibility_congruence_l2538_253867

theorem divisibility_congruence (n : ℤ) :
  (6 ∣ (n - 4)) → (10 ∣ (n - 8)) → n ≡ -2 [ZMOD 30] := by
  sorry

end NUMINAMATH_CALUDE_divisibility_congruence_l2538_253867


namespace NUMINAMATH_CALUDE_steves_book_earnings_l2538_253809

/-- The amount Steve gets for each copy of the book sold -/
def amount_per_copy : ℝ := 2

theorem steves_book_earnings :
  let total_copies : ℕ := 1000000
  let advance_copies : ℕ := 100000
  let agent_percentage : ℝ := 0.1
  let earnings_after_advance : ℝ := 1620000
  
  (total_copies - advance_copies : ℝ) * (1 - agent_percentage) * amount_per_copy = earnings_after_advance :=
by
  sorry

#check steves_book_earnings

end NUMINAMATH_CALUDE_steves_book_earnings_l2538_253809


namespace NUMINAMATH_CALUDE_median_salary_proof_l2538_253840

/-- Represents a position in the company with its count and salary -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- Calculates the median salary given a list of positions -/
def medianSalary (positions : List Position) : Nat :=
  sorry

theorem median_salary_proof (positions : List Position) :
  positions = [
    ⟨"CEO", 1, 140000⟩,
    ⟨"Senior Vice-President", 4, 95000⟩,
    ⟨"Manager", 12, 80000⟩,
    ⟨"Team Leader", 8, 55000⟩,
    ⟨"Office Assistant", 38, 25000⟩
  ] →
  (positions.map (λ p => p.count)).sum = 63 →
  medianSalary positions = 25000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_proof_l2538_253840


namespace NUMINAMATH_CALUDE_range_of_a_l2538_253879

theorem range_of_a (p q : Prop) 
  (hp : ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0)
  (hq : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) :
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2538_253879


namespace NUMINAMATH_CALUDE_our_triangle_can_be_right_or_obtuse_l2538_253841

/-- A triangle with given perimeter and inradius -/
structure Triangle where
  perimeter : ℝ
  inradius : ℝ

/-- Definition of our specific triangle -/
def our_triangle : Triangle := { perimeter := 12, inradius := 1 }

/-- A function to determine if a triangle can be right-angled or obtuse-angled -/
def can_be_right_or_obtuse (t : Triangle) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = t.perimeter ∧
    (a * b * c) / (a + b + c) = 2 * t.inradius * t.perimeter ∧
    (a^2 + b^2 ≥ c^2 ∨ b^2 + c^2 ≥ a^2 ∨ c^2 + a^2 ≥ b^2)

/-- Theorem stating that our triangle can be right-angled or obtuse-angled -/
theorem our_triangle_can_be_right_or_obtuse :
  can_be_right_or_obtuse our_triangle := by
  sorry

end NUMINAMATH_CALUDE_our_triangle_can_be_right_or_obtuse_l2538_253841


namespace NUMINAMATH_CALUDE_intersection_of_planes_l2538_253878

-- Define the two planes
def plane1 (x y z : ℝ) : Prop := 3*x + 4*y - 2*z = 5
def plane2 (x y z : ℝ) : Prop := 2*x + 3*y - z = 3

-- Define the line of intersection
def intersection_line (x y z : ℝ) : Prop :=
  (x - 3) / 2 = (y + 1) / (-1) ∧ (y + 1) / (-1) = z / 1

-- Theorem statement
theorem intersection_of_planes :
  ∀ x y z : ℝ, plane1 x y z ∧ plane2 x y z → intersection_line x y z :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_planes_l2538_253878


namespace NUMINAMATH_CALUDE_series_sum_l2538_253835

-- Define the series
def series_term (n : ℕ) : ℚ := n / 5^n

-- State the theorem
theorem series_sum :
  (∑' n, series_term n) = 5/16 := by sorry

end NUMINAMATH_CALUDE_series_sum_l2538_253835


namespace NUMINAMATH_CALUDE_fair_coin_three_heads_probability_l2538_253829

theorem fair_coin_three_heads_probability :
  let n : ℕ := 7  -- number of coin tosses
  let k : ℕ := 3  -- number of heads we're looking for
  let total_outcomes : ℕ := 2^n  -- total number of possible outcomes
  let favorable_outcomes : ℕ := Nat.choose n k  -- number of ways to choose k heads from n tosses
  (favorable_outcomes : ℚ) / total_outcomes = 35 / 128 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_three_heads_probability_l2538_253829


namespace NUMINAMATH_CALUDE_rope_cut_theorem_l2538_253818

theorem rope_cut_theorem (total_length : ℝ) (ratio_short : ℕ) (ratio_long : ℕ) 
  (h1 : total_length = 40)
  (h2 : ratio_short = 2)
  (h3 : ratio_long = 3) :
  (total_length * ratio_short) / (ratio_short + ratio_long) = 16 := by
  sorry

end NUMINAMATH_CALUDE_rope_cut_theorem_l2538_253818


namespace NUMINAMATH_CALUDE_even_function_m_value_l2538_253858

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x in ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- The function f(x) = x^2 + (m + 2)x + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m + 2) * x + 3

theorem even_function_m_value :
  ∀ m : ℝ, IsEven (f m) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_m_value_l2538_253858


namespace NUMINAMATH_CALUDE_equation_solution_l2538_253826

theorem equation_solution : 
  ∀ x : ℝ, (Real.sqrt (5 * x - 2) + 12 / Real.sqrt (5 * x - 2) = 8) ↔ 
  (x = 38 / 5 ∨ x = 6 / 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2538_253826


namespace NUMINAMATH_CALUDE_benjamins_speed_l2538_253899

/-- Given a distance of 800 kilometers and a time of 10 hours, prove that the speed is 80 kilometers per hour. -/
theorem benjamins_speed (distance : ℝ) (time : ℝ) (h1 : distance = 800) (h2 : time = 10) :
  distance / time = 80 := by
  sorry

end NUMINAMATH_CALUDE_benjamins_speed_l2538_253899


namespace NUMINAMATH_CALUDE_unique_solution_for_inequalities_l2538_253801

theorem unique_solution_for_inequalities :
  ∀ (x y z : ℝ),
    (1 + x^4 ≤ 2*(y - z)^2) ∧
    (1 + y^4 ≤ 2*(z - x)^2) ∧
    (1 + z^4 ≤ 2*(x - y)^2) →
    ((x = 1 ∧ y = 0 ∧ z = -1) ∨
     (x = 1 ∧ y = -1 ∧ z = 0) ∨
     (x = 0 ∧ y = 1 ∧ z = -1) ∨
     (x = 0 ∧ y = -1 ∧ z = 1) ∨
     (x = -1 ∧ y = 1 ∧ z = 0) ∨
     (x = -1 ∧ y = 0 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_inequalities_l2538_253801


namespace NUMINAMATH_CALUDE_dining_table_original_price_l2538_253807

theorem dining_table_original_price (discount_percentage : ℝ) (sale_price : ℝ) (original_price : ℝ) : 
  discount_percentage = 10 →
  sale_price = 450 →
  sale_price = original_price * (1 - discount_percentage / 100) →
  original_price = 500 := by
sorry

end NUMINAMATH_CALUDE_dining_table_original_price_l2538_253807


namespace NUMINAMATH_CALUDE_right_triangle_median_ratio_bound_l2538_253898

theorem right_triangle_median_ratio_bound (a b c s_a s_b s_c : ℝ) 
  (h_right : c^2 = a^2 + b^2)
  (h_s_a : s_a^2 = a^2/4 + b^2)
  (h_s_b : s_b^2 = b^2/4 + a^2)
  (h_s_c : s_c = c/2)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  (s_a + s_b) / s_c ≤ Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_median_ratio_bound_l2538_253898


namespace NUMINAMATH_CALUDE_coin_value_calculation_l2538_253820

theorem coin_value_calculation (total_coins : ℕ) (dimes : ℕ) (nickels : ℕ) : 
  total_coins = 36 → 
  dimes = 26 → 
  nickels = total_coins - dimes → 
  (dimes * 10 + nickels * 5 : ℚ) / 100 = 3.1 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_calculation_l2538_253820


namespace NUMINAMATH_CALUDE_jerry_throws_before_office_l2538_253805

def penalty_system (interrupt : ℕ) (insult : ℕ) (throw : ℕ) : ℕ :=
  5 * interrupt + 10 * insult + 25 * throw

def jerry_current_points : ℕ :=
  penalty_system 2 4 0

theorem jerry_throws_before_office : 
  ∃ (n : ℕ), 
    n = 2 ∧ 
    jerry_current_points + 25 * n < 100 ∧
    jerry_current_points + 25 * (n + 1) ≥ 100 :=
by sorry

end NUMINAMATH_CALUDE_jerry_throws_before_office_l2538_253805


namespace NUMINAMATH_CALUDE_alternative_basis_l2538_253848

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (e₁ e₂ : V)

/-- Given that e₁ and e₂ form a basis for a plane, prove that e₁ + e₂ and e₁ - e₂ also form a basis for the same plane. -/
theorem alternative_basis (h : LinearIndependent ℝ ![e₁, e₂]) :
  LinearIndependent ℝ ![e₁ + e₂, e₁ - e₂] ∧ 
  Submodule.span ℝ {e₁, e₂} = Submodule.span ℝ {e₁ + e₂, e₁ - e₂} := by
  sorry

end NUMINAMATH_CALUDE_alternative_basis_l2538_253848


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2538_253875

theorem inequality_equivalence (x : ℝ) : 
  1 / (x - 2) < 4 ↔ x < 2 ∨ x > 9/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2538_253875


namespace NUMINAMATH_CALUDE_figure2_total_length_l2538_253882

/-- A rectangle-like shape composed of perpendicular line segments -/
structure RectangleShape :=
  (left : ℝ)
  (bottom : ℝ)
  (right : ℝ)
  (top : ℝ)

/-- Calculate the total length of segments in the shape -/
def total_length (shape : RectangleShape) : ℝ :=
  shape.left + shape.bottom + shape.right + shape.top

/-- The theorem stating that the total length of segments in Figure 2 is 23 units -/
theorem figure2_total_length :
  let figure2 : RectangleShape := {
    left := 10,
    bottom := 5,
    right := 7,
    top := 1
  }
  total_length figure2 = 23 := by sorry

end NUMINAMATH_CALUDE_figure2_total_length_l2538_253882


namespace NUMINAMATH_CALUDE_penguin_fish_distribution_l2538_253847

theorem penguin_fish_distribution (days : ℕ) (fish_eaten_by_first_chick : ℕ) : 
  fish_eaten_by_first_chick = 44 →
  (days * 12 - fish_eaten_by_first_chick = 52) := by sorry

end NUMINAMATH_CALUDE_penguin_fish_distribution_l2538_253847


namespace NUMINAMATH_CALUDE_sets_intersection_union_and_subset_l2538_253832

def A (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 4}
def B : Set ℝ := {x | x < -5 ∨ x > 3}

theorem sets_intersection_union_and_subset :
  (∀ x, x ∈ A 1 ∩ B ↔ 3 < x ∧ x ≤ 5) ∧
  (∀ x, x ∈ A 1 ∪ B ↔ x < -5 ∨ x ≥ 1) ∧
  (∀ m, A m ⊆ B ↔ m < -9 ∨ m > 3) :=
sorry

end NUMINAMATH_CALUDE_sets_intersection_union_and_subset_l2538_253832


namespace NUMINAMATH_CALUDE_solve_for_a_l2538_253849

theorem solve_for_a (a b d : ℤ) 
  (eq1 : a + b = d) 
  (eq2 : b + d = 7) 
  (eq3 : d = 4) : 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l2538_253849


namespace NUMINAMATH_CALUDE_regular_working_hours_is_eight_l2538_253806

/-- Represents the problem of finding regular working hours per day --/
def RegularWorkingHours :=
  {H : ℝ // 
    (20 * H * 2.40 + (175 - 20 * H) * 3.20 = 432) ∧ 
    (H > 0) ∧ 
    (H ≤ 24)}

/-- Theorem stating that the regular working hours per day is 8 --/
theorem regular_working_hours_is_eight : 
  ∃ (h : RegularWorkingHours), h.val = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_working_hours_is_eight_l2538_253806


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_value_l2538_253827

theorem mean_equality_implies_x_value : ∃ x : ℝ,
  (7 + 9 + 23) / 3 = (16 + x) / 2 → x = 10 := by sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_value_l2538_253827


namespace NUMINAMATH_CALUDE_expression_simplification_expression_evaluation_l2538_253894

theorem expression_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) (h3 : x ≠ 0) :
  ((x - 2) / (x + 2) + 4 * x / (x^2 - 4)) / (4 * x / (x^2 - 4)) = (x^2 + 4) / (4 * x) :=
by sorry

theorem expression_evaluation :
  let x : ℝ := 1
  ((x - 2) / (x + 2) + 4 * x / (x^2 - 4)) / (4 * x / (x^2 - 4)) = 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_expression_evaluation_l2538_253894


namespace NUMINAMATH_CALUDE_expand_expression_l2538_253876

theorem expand_expression (x y : ℝ) : 
  5 * (3 * x^2 * y - 4 * x * y^2 + 2 * y^3) = 15 * x^2 * y - 20 * x * y^2 + 10 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2538_253876


namespace NUMINAMATH_CALUDE_intersection_A_B_l2538_253844

def A : Set ℝ := {-1, 0, 2, 3, 5}

def B : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_A_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2538_253844


namespace NUMINAMATH_CALUDE_consecutive_zeros_count_is_3719_l2538_253855

/-- Sequence of numbers with no two consecutive zeros -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => a (n+1) + a n

/-- The number of 12-digit positive integers with digits 0 or 1 
    that have at least two consecutive 0's -/
def consecutive_zeros_count : ℕ := 2^12 - a 12

theorem consecutive_zeros_count_is_3719 : consecutive_zeros_count = 3719 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_zeros_count_is_3719_l2538_253855


namespace NUMINAMATH_CALUDE_mean_temperature_l2538_253893

def temperatures : List ℚ := [-8, -5, -5, -6, 0, 4]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : ℚ) = -10/3 := by
sorry

end NUMINAMATH_CALUDE_mean_temperature_l2538_253893


namespace NUMINAMATH_CALUDE_derivative_sin_at_pi_half_l2538_253896

noncomputable def f (x : ℝ) : ℝ := Real.sin x

theorem derivative_sin_at_pi_half :
  deriv f (π / 2) = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_sin_at_pi_half_l2538_253896


namespace NUMINAMATH_CALUDE_b6_b8_value_l2538_253828

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b6_b8_value (a b : ℕ → ℝ) 
    (ha : arithmetic_sequence a)
    (hb : geometric_sequence b)
    (h1 : a 3 + a 11 = 8)
    (h2 : b 7 = a 7) :
  b 6 * b 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_b6_b8_value_l2538_253828


namespace NUMINAMATH_CALUDE_pond_area_l2538_253804

/-- Given a square garden with a perimeter of 48 meters and an area not occupied by a pond of 124 square meters, the area of the pond is 20 square meters. -/
theorem pond_area (garden_perimeter : ℝ) (non_pond_area : ℝ) : 
  garden_perimeter = 48 →
  non_pond_area = 124 →
  (garden_perimeter / 4)^2 - non_pond_area = 20 := by
sorry

end NUMINAMATH_CALUDE_pond_area_l2538_253804
