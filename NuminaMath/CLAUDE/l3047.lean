import Mathlib

namespace disjunction_and_negation_implication_l3047_304784

theorem disjunction_and_negation_implication (p q : Prop) :
  (p ∨ q) → ¬p → (¬p ∧ q) := by
  sorry

end disjunction_and_negation_implication_l3047_304784


namespace expenses_calculation_l3047_304720

/-- Represents the revenue allocation ratio -/
structure RevenueRatio :=
  (employee_salaries : ℕ)
  (stock_purchases : ℕ)
  (rent : ℕ)
  (marketing_costs : ℕ)

/-- Calculates the total amount spent on employee salaries, rent, and marketing costs -/
def calculate_expenses (revenue : ℕ) (ratio : RevenueRatio) : ℕ :=
  let total_ratio := ratio.employee_salaries + ratio.stock_purchases + ratio.rent + ratio.marketing_costs
  let unit_value := revenue / total_ratio
  (ratio.employee_salaries + ratio.rent + ratio.marketing_costs) * unit_value

/-- Theorem stating that the calculated expenses for the given revenue and ratio equal $7,800 -/
theorem expenses_calculation (revenue : ℕ) (ratio : RevenueRatio) :
  revenue = 10800 ∧ 
  ratio = { employee_salaries := 3, stock_purchases := 5, rent := 2, marketing_costs := 8 } →
  calculate_expenses revenue ratio = 7800 :=
by sorry

end expenses_calculation_l3047_304720


namespace solution_check_unique_non_solution_l3047_304746

theorem solution_check : ℝ → ℝ → Prop :=
  fun x y => x + y = 5

theorem unique_non_solution :
  (solution_check 2 3 ∧ 
   solution_check (-2) 7 ∧ 
   solution_check 0 5) ∧ 
  ¬(solution_check 1 6) := by
  sorry

end solution_check_unique_non_solution_l3047_304746


namespace subtraction_of_fractions_l3047_304733

theorem subtraction_of_fractions : 
  (2 + 1/4) - 2/3 = 1 + 7/12 := by sorry

end subtraction_of_fractions_l3047_304733


namespace weight_of_larger_square_l3047_304738

/-- Represents the properties of a square metal piece -/
structure MetalSquare where
  side : ℝ  -- side length in inches
  weight : ℝ  -- weight in ounces

/-- Theorem stating the relationship between two metal squares -/
theorem weight_of_larger_square 
  (small : MetalSquare) 
  (large : MetalSquare) 
  (h1 : small.side = 4) 
  (h2 : small.weight = 16) 
  (h3 : large.side = 6) 
  (h_uniform : ∀ (s1 s2 : MetalSquare), s1.weight / (s1.side ^ 2) = s2.weight / (s2.side ^ 2)) :
  large.weight = 36 := by
  sorry

end weight_of_larger_square_l3047_304738


namespace problem_solution_l3047_304785

theorem problem_solution : ∃ x : ℚ, ((15 - 2 + 4 / 1) / x) * 8 = 77 ∧ x = 136 / 77 := by
  sorry

end problem_solution_l3047_304785


namespace complex_power_modulus_l3047_304768

theorem complex_power_modulus : 
  Complex.abs ((2/3 : ℂ) + (1/3 : ℂ) * Complex.I) ^ 8 = 625/6561 := by
  sorry

end complex_power_modulus_l3047_304768


namespace boat_speed_in_still_water_l3047_304774

/-- The speed of a boat in still water, given downstream travel information and current speed. -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 3 →
  downstream_distance = 3.6 →
  downstream_time = 1/5 →
  ∃ (boat_speed : ℝ), boat_speed = 15 ∧ downstream_distance = (boat_speed + current_speed) * downstream_time :=
by
  sorry


end boat_speed_in_still_water_l3047_304774


namespace base7_subtraction_l3047_304786

/-- Represents a number in base 7 as a list of digits (least significant first) -/
def Base7 := List Nat

/-- Converts a base 7 number to its decimal representation -/
def toDecimal (n : Base7) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The difference between two base 7 numbers -/
def base7Difference (a b : Base7) : Base7 :=
  sorry -- Implementation not required for the statement

/-- Statement: The difference between 4512₇ and 2345₇ in base 7 is 2144₇ -/
theorem base7_subtraction :
  base7Difference [2, 1, 5, 4] [5, 4, 3, 2] = [4, 4, 1, 2] :=
sorry

end base7_subtraction_l3047_304786


namespace trigonometric_expression_simplification_l3047_304710

theorem trigonometric_expression_simplification :
  let original_expression := (Real.sin (20 * π / 180) + Real.sin (40 * π / 180) + 
                              Real.sin (60 * π / 180) + Real.sin (80 * π / 180)) / 
                             (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * 
                              Real.cos (30 * π / 180) * Real.cos (40 * π / 180))
  let simplified_expression := (4 * Real.sin (50 * π / 180)) / 
                               (Real.cos (30 * π / 180) * Real.cos (40 * π / 180))
  original_expression = simplified_expression := by
sorry

end trigonometric_expression_simplification_l3047_304710


namespace sum_of_reciprocals_positive_l3047_304789

theorem sum_of_reciprocals_positive (a b c d : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1) (hd : |d| > 1)
  (h_eq : a * b * (c + d) + d * c * (a + b) + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 :=
by sorry

end sum_of_reciprocals_positive_l3047_304789


namespace area_of_smaller_circle_l3047_304794

/-- Given two externally tangent circles with common external tangents,
    prove that the area of the smaller circle is π(625 + 200√2) / 49 -/
theorem area_of_smaller_circle (P A B A' B' S L : ℝ × ℝ) : 
  let r := Real.sqrt ((5 + 10 * Real.sqrt 2) ^ 2 / 49)
  -- Two circles are externally tangent
  (∃ T : ℝ × ℝ, ‖S - T‖ = r ∧ ‖L - T‖ = 2*r) →
  -- PAB and PA'B' are common external tangents
  (‖P - A‖ = 5 ∧ ‖A - B‖ = 5 ∧ ‖P - A'‖ = 5 ∧ ‖A' - B'‖ = 5) →
  -- A and A' are on the smaller circle
  (‖S - A‖ = r ∧ ‖S - A'‖ = r) →
  -- B and B' are on the larger circle
  (‖L - B‖ = 2*r ∧ ‖L - B'‖ = 2*r) →
  -- Area of the smaller circle
  π * r^2 = π * (625 + 200 * Real.sqrt 2) / 49 :=
by sorry

end area_of_smaller_circle_l3047_304794


namespace fraction_simplification_l3047_304777

theorem fraction_simplification :
  (3 / 7 + 4 / 5) / (5 / 12 + 2 / 3) = 516 / 455 := by
  sorry

end fraction_simplification_l3047_304777


namespace odd_integers_sum_13_to_45_l3047_304721

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  (a₁ + aₙ) * n / 2

theorem odd_integers_sum_13_to_45 :
  arithmetic_sum 13 45 2 = 493 := by
  sorry

end odd_integers_sum_13_to_45_l3047_304721


namespace total_time_is_8_days_l3047_304705

-- Define the problem parameters
def plow_rate : ℝ := 10  -- acres per day
def mow_rate : ℝ := 12   -- acres per day
def farmland_area : ℝ := 55  -- acres
def grassland_area : ℝ := 30  -- acres

-- Theorem statement
theorem total_time_is_8_days : 
  (farmland_area / plow_rate) + (grassland_area / mow_rate) = 8 := by
  sorry

end total_time_is_8_days_l3047_304705


namespace remove_five_blocks_count_l3047_304796

/-- Represents the number of exposed blocks after removing n blocks -/
def E (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3 * n + 1

/-- Represents the number of blocks in the k-th layer from the top -/
def blocks_in_layer (k : ℕ) : ℕ := 4^(k-1)

/-- The total number of ways to remove 5 blocks from the stack -/
def remove_five_blocks : ℕ := 
  (E 0) * (E 1) * (E 2) * (E 3) * (E 4) - (E 0) * (blocks_in_layer 2) * (blocks_in_layer 2) * (blocks_in_layer 2) * (blocks_in_layer 2)

theorem remove_five_blocks_count : remove_five_blocks = 3384 := by
  sorry

end remove_five_blocks_count_l3047_304796


namespace thirteen_thousand_one_hundred_twenty_one_obtainable_twelve_thousand_one_hundred_thirty_one_not_obtainable_l3047_304778

/-- The set of numbers that can be written on the blackboard -/
inductive BoardNumber : ℕ → Prop where
  | one : BoardNumber 1
  | two : BoardNumber 2
  | add (m n : ℕ) : BoardNumber m → BoardNumber n → BoardNumber (m + n + m * n)

/-- A number is obtainable if it's in the set of BoardNumbers -/
def Obtainable (n : ℕ) : Prop := BoardNumber n

theorem thirteen_thousand_one_hundred_twenty_one_obtainable :
  Obtainable 13121 :=
sorry

theorem twelve_thousand_one_hundred_thirty_one_not_obtainable :
  ¬ Obtainable 12131 :=
sorry

end thirteen_thousand_one_hundred_twenty_one_obtainable_twelve_thousand_one_hundred_thirty_one_not_obtainable_l3047_304778


namespace harvard_applicants_l3047_304735

/-- The number of students who choose to attend Harvard University -/
def students_attending : ℕ := 900

/-- The acceptance rate for Harvard University applicants -/
def acceptance_rate : ℚ := 5 / 100

/-- The percentage of accepted students who choose to attend Harvard University -/
def attendance_rate : ℚ := 90 / 100

/-- The number of students who applied to Harvard University -/
def applicants : ℕ := 20000

theorem harvard_applicants :
  (↑students_attending : ℚ) = (↑applicants : ℚ) * acceptance_rate * attendance_rate := by
  sorry

end harvard_applicants_l3047_304735


namespace rectangle_dimensions_l3047_304798

theorem rectangle_dimensions (w : ℝ) (h1 : w > 0) : 
  (6 * w = 3 * (2 * w^2)) → w = 1 ∧ 2 * w = 2 := by
  sorry

end rectangle_dimensions_l3047_304798


namespace inequality_proof_l3047_304724

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end inequality_proof_l3047_304724


namespace centroid_equal_areas_l3047_304748

/-- The centroid of a triangle divides it into three equal-area triangles -/
theorem centroid_equal_areas (P Q R S : ℝ × ℝ) : 
  P = (-1, 3) → Q = (2, 7) → R = (4, 0) → 
  S.1 = (P.1 + Q.1 + R.1) / 3 → 
  S.2 = (P.2 + Q.2 + R.2) / 3 → 
  8 * S.1 + 3 * S.2 = 70 / 3 := by
  sorry

#check centroid_equal_areas

end centroid_equal_areas_l3047_304748


namespace same_expected_defects_l3047_304772

/-- Represents a worker's probability distribution of defective products -/
structure Worker where
  p0 : ℝ  -- Probability of 0 defective products
  p1 : ℝ  -- Probability of 1 defective product
  p2 : ℝ  -- Probability of 2 defective products
  p3 : ℝ  -- Probability of 3 defective products
  sum_to_one : p0 + p1 + p2 + p3 = 1
  non_negative : p0 ≥ 0 ∧ p1 ≥ 0 ∧ p2 ≥ 0 ∧ p3 ≥ 0

/-- Calculate the expected number of defective products for a worker -/
def expected_defects (w : Worker) : ℝ :=
  0 * w.p0 + 1 * w.p1 + 2 * w.p2 + 3 * w.p3

/-- Worker A's probability distribution -/
def worker_A : Worker := {
  p0 := 0.4
  p1 := 0.3
  p2 := 0.2
  p3 := 0.1
  sum_to_one := by norm_num
  non_negative := by norm_num
}

/-- Worker B's probability distribution -/
def worker_B : Worker := {
  p0 := 0.4
  p1 := 0.2
  p2 := 0.4
  p3 := 0
  sum_to_one := by norm_num
  non_negative := by norm_num
}

/-- Theorem stating that the expected number of defective products is the same for both workers -/
theorem same_expected_defects : expected_defects worker_A = expected_defects worker_B := by
  sorry

end same_expected_defects_l3047_304772


namespace imaginary_part_of_complex_division_l3047_304747

theorem imaginary_part_of_complex_division (z₁ z₂ : ℂ) :
  z₁.re = 1 →
  z₁.im = 1 →
  z₂.re = 0 →
  z₂.im = 1 →
  Complex.im (z₁ / z₂) = -1 := by
  sorry

end imaginary_part_of_complex_division_l3047_304747


namespace negative_three_inequality_l3047_304756

theorem negative_three_inequality (a b : ℝ) : a < b → -3 * a > -3 * b := by
  sorry

end negative_three_inequality_l3047_304756


namespace divisors_not_mult_6_l3047_304767

/-- The smallest integer satisfying the given conditions -/
def n : ℕ := 2^30 * 3^15 * 5^25

/-- n/2 is a perfect square -/
axiom n_div_2_is_square : ∃ k : ℕ, n / 2 = k^2

/-- n/4 is a perfect cube -/
axiom n_div_4_is_cube : ∃ j : ℕ, n / 4 = j^3

/-- n/5 is a perfect fifth -/
axiom n_div_5_is_fifth : ∃ m : ℕ, n / 5 = m^5

/-- The number of divisors of n -/
def total_divisors : ℕ := (30 + 1) * (15 + 1) * (25 + 1)

/-- The number of divisors of n that are multiples of 2 -/
def divisors_mult_2 : ℕ := (15 + 1) * (25 + 1)

/-- The number of divisors of n that are multiples of 3 -/
def divisors_mult_3 : ℕ := (29 + 1) * (25 + 1)

/-- Theorem: The number of divisors of n that are not multiples of 6 is 11740 -/
theorem divisors_not_mult_6 : total_divisors - divisors_mult_2 - divisors_mult_3 = 11740 := by
  sorry

end divisors_not_mult_6_l3047_304767


namespace no_integer_pairs_with_square_diff_150_l3047_304782

theorem no_integer_pairs_with_square_diff_150 :
  ¬∃ (m n : ℕ), m ≥ n ∧ m^2 - n^2 = 150 := by
  sorry

end no_integer_pairs_with_square_diff_150_l3047_304782


namespace inequality_proof_l3047_304766

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x < y) :
  x + Real.sqrt (y^2 + 2) < y + Real.sqrt (x^2 + 2) := by
  sorry

end inequality_proof_l3047_304766


namespace train_speed_calculation_l3047_304704

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 90)
  (h2 : bridge_length = 200)
  (h3 : crossing_time = 36) :
  ∃ (speed : ℝ), 
    (speed ≥ 28.9 ∧ speed ≤ 29.1) ∧ 
    speed = (train_length + bridge_length) / crossing_time * 3.6 := by
  sorry

end train_speed_calculation_l3047_304704


namespace coins_problem_l3047_304799

theorem coins_problem (A B C D : ℕ) : 
  A = 21 →
  B = A - 9 →
  C = B + 17 →
  A + B + 5 = C + D →
  D = 9 := by
sorry

end coins_problem_l3047_304799


namespace gcf_lcm_sum_8_12_l3047_304760

theorem gcf_lcm_sum_8_12 : 
  Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcf_lcm_sum_8_12_l3047_304760


namespace down_payment_proof_l3047_304797

-- Define the number of people
def num_people : ℕ := 3

-- Define the individual payment amount
def individual_payment : ℚ := 1166.67

-- Function to round to nearest dollar
def round_to_dollar (x : ℚ) : ℕ := 
  (x + 0.5).floor.toNat

-- Define the total down payment
def total_down_payment : ℕ := num_people * round_to_dollar individual_payment

-- Theorem statement
theorem down_payment_proof : total_down_payment = 3501 := by
  sorry

end down_payment_proof_l3047_304797


namespace divisibility_theorem_l3047_304753

def group_digits (n : ℕ) : List ℕ :=
  sorry

def alternating_sum (groups : List ℕ) : ℤ :=
  sorry

theorem divisibility_theorem (A : ℕ) :
  let groups := group_digits A
  let B := alternating_sum groups
  (7 ∣ (A - B) ∧ 11 ∣ (A - B) ∧ 13 ∣ (A - B)) ↔ (7 ∣ A ∧ 11 ∣ A ∧ 13 ∣ A) :=
by sorry

end divisibility_theorem_l3047_304753


namespace square_partition_exists_equilateral_triangle_partition_exists_l3047_304706

-- Define a structure for a triangle
structure Triangle :=
  (a b c : ℝ)

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define what it means for two triangles to be congruent
def areCongruent (t1 t2 : Triangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

-- Define a structure for a square
structure Square :=
  (side : ℝ)

-- Define a structure for an equilateral triangle
structure EquilateralTriangle :=
  (side : ℝ)

-- Theorem for the square partition
theorem square_partition_exists (s : Square) : 
  ∃ (t1 t2 t3 t4 : Triangle), 
    (isIsosceles t1 ∧ isIsosceles t2 ∧ isIsosceles t3 ∧ isIsosceles t4) ∧
    (¬ areCongruent t1 t2 ∧ ¬ areCongruent t1 t3 ∧ ¬ areCongruent t1 t4 ∧
     ¬ areCongruent t2 t3 ∧ ¬ areCongruent t2 t4 ∧ ¬ areCongruent t3 t4) :=
sorry

-- Theorem for the equilateral triangle partition
theorem equilateral_triangle_partition_exists (et : EquilateralTriangle) : 
  ∃ (t1 t2 t3 t4 : Triangle), 
    (isIsosceles t1 ∧ isIsosceles t2 ∧ isIsosceles t3 ∧ isIsosceles t4) ∧
    (¬ areCongruent t1 t2 ∧ ¬ areCongruent t1 t3 ∧ ¬ areCongruent t1 t4 ∧
     ¬ areCongruent t2 t3 ∧ ¬ areCongruent t2 t4 ∧ ¬ areCongruent t3 t4) :=
sorry

end square_partition_exists_equilateral_triangle_partition_exists_l3047_304706


namespace no_set_M_exists_l3047_304716

theorem no_set_M_exists : ¬ ∃ (M : Set ℕ),
  (∀ m : ℕ, m > 1 → ∃ a b : ℕ, a ∈ M ∧ b ∈ M ∧ a + b = m) ∧
  (∀ a b c d : ℕ, a ∈ M → b ∈ M → c ∈ M → d ∈ M → 
    a > 10 → b > 10 → c > 10 → d > 10 → 
    a + b = c + d → (a = c ∨ a = d)) :=
by sorry

end no_set_M_exists_l3047_304716


namespace linda_expenditure_l3047_304744

def notebook_price : ℝ := 1.20
def notebook_quantity : ℕ := 3
def pencil_box_price : ℝ := 1.50
def pen_box_price : ℝ := 1.70
def marker_pack_price : ℝ := 2.80
def calculator_price : ℝ := 12.50
def item_discount_rate : ℝ := 0.15
def coupon_discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

def total_expenditure : ℝ := 19.52

theorem linda_expenditure :
  let discountable_items_total := notebook_price * notebook_quantity + pencil_box_price + pen_box_price + marker_pack_price
  let discounted_items_total := discountable_items_total * (1 - item_discount_rate)
  let total_after_item_discount := discounted_items_total + calculator_price
  let total_after_coupon := total_after_item_discount * (1 - coupon_discount_rate)
  let final_total := total_after_coupon * (1 + sales_tax_rate)
  final_total = total_expenditure := by
sorry

end linda_expenditure_l3047_304744


namespace sum_of_park_areas_l3047_304727

theorem sum_of_park_areas :
  let park1_side : ℝ := 11
  let park2_side : ℝ := 5
  let park1_area := park1_side * park1_side
  let park2_area := park2_side * park2_side
  park1_area + park2_area = 146 := by
sorry

end sum_of_park_areas_l3047_304727


namespace smallest_norm_w_l3047_304736

theorem smallest_norm_w (w : ℝ × ℝ) (h : ‖w + (4, 2)‖ = 10) :
  ∃ (w_min : ℝ × ℝ), (∀ w' : ℝ × ℝ, ‖w' + (4, 2)‖ = 10 → ‖w_min‖ ≤ ‖w'‖) ∧ ‖w_min‖ = 10 - 2 * Real.sqrt 5 :=
sorry

end smallest_norm_w_l3047_304736


namespace percentage_calculation_l3047_304742

theorem percentage_calculation (x : ℝ) (h : 0.30 * 0.40 * x = 24) : 0.20 * 0.60 * x = 24 := by
  sorry

end percentage_calculation_l3047_304742


namespace simplify_expression_l3047_304783

theorem simplify_expression (a : ℝ) (h1 : a ≠ -3) (h2 : a ≠ 1) :
  (1 - 4 / (a + 3)) / ((a^2 - 2*a + 1) / (2*a + 6)) = 2 / (a - 1) := by
  sorry

end simplify_expression_l3047_304783


namespace cubic_polynomial_sum_l3047_304762

/-- A cubic polynomial with specific values at 0, 1, and -1 -/
structure CubicPolynomial (k : ℝ) where
  P : ℝ → ℝ
  is_cubic : ∃ a b c : ℝ, ∀ x, P x = a * x^3 + b * x^2 + c * x + k
  value_at_zero : P 0 = k
  value_at_one : P 1 = 2 * k
  value_at_neg_one : P (-1) = 3 * k

/-- The sum of the polynomial evaluated at 2 and -2 equals 14k -/
theorem cubic_polynomial_sum (k : ℝ) (p : CubicPolynomial k) :
  p.P 2 + p.P (-2) = 14 * k := by
  sorry

end cubic_polynomial_sum_l3047_304762


namespace repeating_decimal_sum_diff_prod_l3047_304722

theorem repeating_decimal_sum_diff_prod : 
  let repeating_decimal (n : ℕ) := n / 9
  (repeating_decimal 6) + (repeating_decimal 2) - (repeating_decimal 4) * (repeating_decimal 3) = 20 / 27 := by
  sorry

end repeating_decimal_sum_diff_prod_l3047_304722


namespace possible_values_of_a_l3047_304707

-- Define the sets P and M
def P : Set ℝ := {x | x^2 = 1}
def M (a : ℝ) : Set ℝ := {x | a * x = 1}

-- Define the set of possible values for a
def A : Set ℝ := {1, -1, 0}

-- Statement to prove
theorem possible_values_of_a (a : ℝ) : M a ⊆ P → a ∈ A := by
  sorry

end possible_values_of_a_l3047_304707


namespace cuboid_breadth_l3047_304755

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboidSurfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: The breadth of a cuboid with given length, height, and surface area -/
theorem cuboid_breadth (l h area : ℝ) (hl : l = 8) (hh : h = 9) (harea : area = 432) :
  ∃ b : ℝ, cuboidSurfaceArea l b h = area ∧ b = 144 / 17 := by
  sorry

end cuboid_breadth_l3047_304755


namespace count_five_digit_palindromes_l3047_304773

/-- A five-digit palindrome is a number of the form abcba where a, b, c are digits and a ≠ 0. -/
def FiveDigitPalindrome (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    a ≥ 1 ∧ a ≤ 9 ∧
    b ≥ 0 ∧ b ≤ 9 ∧
    c ≥ 0 ∧ c ≤ 9 ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

/-- The count of five-digit palindromes. -/
def CountFiveDigitPalindromes : ℕ := 
  (Finset.range 9).card * (Finset.range 10).card * (Finset.range 10).card

theorem count_five_digit_palindromes :
  CountFiveDigitPalindromes = 900 :=
sorry

end count_five_digit_palindromes_l3047_304773


namespace triangle_two_solutions_range_l3047_304734

theorem triangle_two_solutions_range (a b : ℝ) (B : ℝ) (h1 : b = 2) (h2 : B = 45 * π / 180) :
  (∃ (A C : ℝ), 0 < A ∧ 0 < C ∧ A + B + C = π ∧ 
   a * Real.sin B < b ∧ b < a) ↔ 
  (2 < a ∧ a < 2 * Real.sqrt 2) :=
sorry

end triangle_two_solutions_range_l3047_304734


namespace expression_evaluation_l3047_304737

theorem expression_evaluation : 101^3 + 3*(101^2) + 3*101 + 1 = 1061208 := by
  sorry

end expression_evaluation_l3047_304737


namespace perfect_square_expression_l3047_304793

theorem perfect_square_expression : ∃ y : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + 0.02 * 0.02) = y^2 := by
  sorry

end perfect_square_expression_l3047_304793


namespace alcohol_mixture_percentage_l3047_304757

theorem alcohol_mixture_percentage (x : ℝ) : 
  (8 * 0.25 + 2 * (x / 100)) / (8 + 2) = 0.224 → x = 12 := by
  sorry

end alcohol_mixture_percentage_l3047_304757


namespace octal_sum_l3047_304763

/-- Converts a base-8 number to base-10 --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- The sum of 444₈, 44₈, and 4₈ in base 8 is 514₈ --/
theorem octal_sum : 
  decimal_to_octal (octal_to_decimal 444 + octal_to_decimal 44 + octal_to_decimal 4) = 514 := by
  sorry

end octal_sum_l3047_304763


namespace fraction_to_decimal_l3047_304752

theorem fraction_to_decimal :
  (53 : ℚ) / (4 * 5^7) = (1325 : ℚ) / 10^7 := by sorry

end fraction_to_decimal_l3047_304752


namespace midpoint_property_l3047_304702

/-- Given two points A and B in a 2D plane, prove that if C is the midpoint of AB,
    then 2x - 4y = -15, where (x, y) are the coordinates of C. -/
theorem midpoint_property (A B C : ℝ × ℝ) : 
  A = (17, 10) → B = (-2, 5) → C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  2 * C.1 - 4 * C.2 = -15 := by
  sorry

end midpoint_property_l3047_304702


namespace range_of_a_range_of_x_l3047_304749

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Part 1
theorem range_of_a (a b : ℝ) :
  (f a b 1 = 1) →
  (∀ x ∈ Set.Ioo 2 5, f a b x > 0) →
  a ∈ Set.Ioi (3 - 2 * Real.sqrt 2) :=
sorry

-- Part 2
theorem range_of_x (a b x : ℝ) :
  (f a b 1 = 1) →
  (∀ a ∈ Set.Icc (-2) (-1), f a b x > 0) →
  x ∈ Set.Ioo ((1 - Real.sqrt 17) / 4) ((1 + Real.sqrt 17) / 4) :=
sorry

end range_of_a_range_of_x_l3047_304749


namespace t_range_for_inequality_l3047_304770

theorem t_range_for_inequality (t : ℝ) : 
  (∀ x : ℝ, abs x ≤ 1 → t + 1 > (t^2 - 4) * x) ↔ 
  (t > (Real.sqrt 13 - 1) / 2 ∧ t < (Real.sqrt 21 + 1) / 2) :=
sorry

end t_range_for_inequality_l3047_304770


namespace max_min_difference_z_l3047_304745

theorem max_min_difference_z (x y z : ℝ) 
  (sum_condition : x + y + z = 3)
  (sum_squares_condition : x^2 + y^2 + z^2 = 18) :
  ∃ (z_max z_min : ℝ),
    (∀ z' : ℝ, (∃ x' y' : ℝ, x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≤ z_max) ∧
    (∀ z' : ℝ, (∃ x' y' : ℝ, x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≥ z_min) ∧
    z_max - z_min = 6.5 :=
by sorry

end max_min_difference_z_l3047_304745


namespace geometric_sequence_product_constant_geometric_sequence_product_16_l3047_304754

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of two terms equidistant from the beginning and end of the sequence is constant -/
theorem geometric_sequence_product_constant {a : ℕ → ℝ} (h : GeometricSequence a) :
  ∀ m n k : ℕ, m < n → a m * a n = a (m + k) * a (n - k) := by sorry

theorem geometric_sequence_product_16 (a : ℕ → ℝ) (h : GeometricSequence a) 
  (h2 : a 4 * a 8 = 16) : a 2 * a 10 = 16 := by sorry

end geometric_sequence_product_constant_geometric_sequence_product_16_l3047_304754


namespace inequality_system_solution_set_l3047_304731

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 1 ≥ -3 ∧ -2 * (x + 3) > 0}
  S = {x : ℝ | -4 ≤ x ∧ x < -3} := by
  sorry

end inequality_system_solution_set_l3047_304731


namespace rhombus_area_l3047_304743

/-- The area of a rhombus with longer diagonal 30 units and angle 60° between diagonals is 225√3 square units -/
theorem rhombus_area (d₁ : ℝ) (θ : ℝ) (h₁ : d₁ = 30) (h₂ : θ = Real.pi / 3) :
  let d₂ := d₁ * Real.sin θ
  d₁ * d₂ / 2 = 225 * Real.sqrt 3 := by
  sorry

end rhombus_area_l3047_304743


namespace congruence_solution_l3047_304739

theorem congruence_solution (n : ℕ) : n ≡ 40 [ZMOD 43] ↔ 11 * n ≡ 10 [ZMOD 43] ∧ n ≤ 42 := by
  sorry

end congruence_solution_l3047_304739


namespace greatest_three_digit_number_l3047_304709

theorem greatest_three_digit_number : ∃ n : ℕ, 
  n = 978 ∧ 
  n < 1000 ∧ 
  n > 99 ∧
  ∃ k : ℕ, n = 8 * k + 2 ∧
  ∃ m : ℕ, n = 7 * m + 4 ∧
  ∀ x : ℕ, x < 1000 ∧ x > 99 ∧ (∃ a : ℕ, x = 8 * a + 2) ∧ (∃ b : ℕ, x = 7 * b + 4) → x ≤ n :=
by sorry

end greatest_three_digit_number_l3047_304709


namespace todds_initial_gum_pieces_todds_initial_gum_pieces_proof_l3047_304758

theorem todds_initial_gum_pieces : ℝ → Prop :=
  fun x =>
    let additional_pieces : ℝ := 150
    let percentage_increase : ℝ := 0.25
    let final_total : ℝ := 890
    (x + additional_pieces = final_total) ∧
    (additional_pieces = percentage_increase * x) →
    x = 712

-- The proof is omitted
theorem todds_initial_gum_pieces_proof : todds_initial_gum_pieces 712 := by
  sorry

end todds_initial_gum_pieces_todds_initial_gum_pieces_proof_l3047_304758


namespace angle_A_range_l3047_304701

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the convexity property
def is_convex (q : Quadrilateral) : Prop := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle measure function
def angle_measure (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_A_range (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_AB : distance q.A q.B = 8)
  (h_BC : distance q.B q.C = 4)
  (h_CD : distance q.C q.D = 6)
  (h_DA : distance q.D q.A = 6) :
  0 < angle_measure q.B q.A q.D ∧ angle_measure q.B q.A q.D < Real.pi / 2 := by
  sorry

end angle_A_range_l3047_304701


namespace imaginary_unit_power_l3047_304726

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2014 = -1 := by
  sorry

end imaginary_unit_power_l3047_304726


namespace power_order_l3047_304761

theorem power_order : 
  let p := (2 : ℕ) ^ 3009
  let q := (3 : ℕ) ^ 2006
  let r := (5 : ℕ) ^ 1003
  r < p ∧ p < q := by sorry

end power_order_l3047_304761


namespace unique_solution_l3047_304764

theorem unique_solution : ∃! (x : ℝ), x ≥ 0 ∧ x + 10 * Real.sqrt x = 39 := by sorry

end unique_solution_l3047_304764


namespace average_lifespan_of_sampled_products_l3047_304781

/-- Represents a factory producing electronic products -/
structure Factory where
  production_ratio : ℚ
  average_lifespan : ℚ

/-- Calculates the weighted average lifespan of products from multiple factories -/
def weighted_average_lifespan (factories : List Factory) (total_samples : ℕ) : ℚ :=
  let total_ratio := factories.map (λ f => f.production_ratio) |>.sum
  let weighted_sum := factories.map (λ f => f.production_ratio * f.average_lifespan) |>.sum
  weighted_sum / total_ratio

/-- The main theorem proving the average lifespan of sampled products -/
theorem average_lifespan_of_sampled_products : 
  let factories := [
    { production_ratio := 1, average_lifespan := 980 },
    { production_ratio := 2, average_lifespan := 1020 },
    { production_ratio := 1, average_lifespan := 1032 }
  ]
  let total_samples := 100
  weighted_average_lifespan factories total_samples = 1013 := by
  sorry

end average_lifespan_of_sampled_products_l3047_304781


namespace only_b_opens_upwards_l3047_304765

def quadratic_a (x : ℝ) : ℝ := 1 - x - 6*x^2
def quadratic_b (x : ℝ) : ℝ := -8*x + x^2 + 1
def quadratic_c (x : ℝ) : ℝ := (1 - x)*(x + 5)
def quadratic_d (x : ℝ) : ℝ := 2 - (5 - x)^2

def opens_upwards (f : ℝ → ℝ) : Prop :=
  ∃ a > 0, ∃ b c : ℝ, ∀ x, f x = a*x^2 + b*x + c

theorem only_b_opens_upwards :
  opens_upwards quadratic_b ∧
  ¬opens_upwards quadratic_a ∧
  ¬opens_upwards quadratic_c ∧
  ¬opens_upwards quadratic_d :=
by sorry

end only_b_opens_upwards_l3047_304765


namespace floor_of_4_8_l3047_304718

theorem floor_of_4_8 : ⌊(4.8 : ℝ)⌋ = 4 := by sorry

end floor_of_4_8_l3047_304718


namespace train_length_l3047_304700

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length. -/
theorem train_length (train_speed : ℝ) (time_to_pass : ℝ) (bridge_length : ℝ) :
  train_speed = 45 →
  time_to_pass = 36 →
  bridge_length = 140 →
  (train_speed * 1000 / 3600) * time_to_pass - bridge_length = 310 := by
  sorry

#check train_length

end train_length_l3047_304700


namespace part_one_part_two_l3047_304759

-- Part I
theorem part_one (t : ℝ) (h1 : t^2 - 5*t + 4 < 0) (h2 : (t-2)*(t-6) < 0) : 
  2 < t ∧ t < 4 := by sorry

-- Part II
theorem part_two (a : ℝ) (h : a ≠ 0) 
  (h_suff : ∀ t : ℝ, 2 < t ∧ t < 6 → t^2 - 5*a*t + 4*a^2 < 0) : 
  3/2 ≤ a ∧ a ≤ 2 := by sorry

end part_one_part_two_l3047_304759


namespace machine_working_time_yesterday_l3047_304715

/-- The total working time of an industrial machine, including downtime -/
def total_working_time (shirts_produced : ℕ) (production_rate : ℕ) (downtime : ℕ) : ℕ :=
  shirts_produced * production_rate + downtime

/-- Proof that the machine worked for 38 minutes yesterday -/
theorem machine_working_time_yesterday :
  total_working_time 9 2 20 = 38 := by
  sorry

end machine_working_time_yesterday_l3047_304715


namespace local_minimum_implies_m_eq_one_l3047_304730

/-- The function f(x) = x(x-m)^2 has a local minimum at x = 1 -/
def has_local_minimum_at_one (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1

/-- The main theorem: if f(x) = x(x-m)^2 has a local minimum at x = 1, then m = 1 -/
theorem local_minimum_implies_m_eq_one (m : ℝ) :
  has_local_minimum_at_one (fun x => x * (x - m)^2) m → m = 1 :=
by sorry

end local_minimum_implies_m_eq_one_l3047_304730


namespace min_square_area_is_49_l3047_304740

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a circle with diameter -/
structure Circle where
  diameter : ℝ

/-- Calculates the minimum square side length to contain given shapes -/
def minSquareSideLength (rect1 rect2 : Rectangle) (circle : Circle) : ℝ :=
  sorry

/-- Theorem: The minimum area of the square containing the given shapes is 49 -/
theorem min_square_area_is_49 : 
  let rect1 : Rectangle := ⟨2, 4⟩
  let rect2 : Rectangle := ⟨3, 5⟩
  let circle : Circle := ⟨3⟩
  (minSquareSideLength rect1 rect2 circle) ^ 2 = 49 := by
  sorry

end min_square_area_is_49_l3047_304740


namespace allocation_methods_for_three_schools_l3047_304787

/-- The number of ways to allocate doctors and nurses to schools. -/
def allocation_methods (num_doctors num_nurses num_schools : ℕ) : ℕ :=
  (num_doctors.factorial) * (num_nurses.choose 2 * (num_nurses - 2).choose 2)

/-- Theorem stating that there are 540 different allocation methods for 3 doctors and 6 nurses to 3 schools. -/
theorem allocation_methods_for_three_schools :
  allocation_methods 3 6 3 = 540 := by
  sorry

#eval allocation_methods 3 6 3

end allocation_methods_for_three_schools_l3047_304787


namespace zero_point_condition_sufficient_condition_not_necessary_condition_l3047_304719

def f (a : ℝ) (x : ℝ) := a * x + 3

theorem zero_point_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 2, f a x ≠ 0) ∨
  (∃ x ∈ Set.Ioo (-1 : ℝ) 2, f a x = 0) :=
by sorry

theorem sufficient_condition (a : ℝ) (h : a < -3) :
  ∃ x ∈ Set.Ioo (-1 : ℝ) 2, f a x = 0 :=
by sorry

theorem not_necessary_condition :
  ∃ a ≥ -3, ∃ x ∈ Set.Ioo (-1 : ℝ) 2, f a x = 0 :=
by sorry

end zero_point_condition_sufficient_condition_not_necessary_condition_l3047_304719


namespace match_probabilities_and_expectation_l3047_304725

/-- Represents the outcome of a single game -/
inductive GameOutcome
| A_wins
| B_wins

/-- Represents the state of the match after the first two games -/
structure MatchState :=
  (A_wins : Nat)
  (B_wins : Nat)

/-- The probability of A winning a single game -/
def p_A_win : ℝ := 0.6

/-- The probability of B winning a single game -/
def p_B_win : ℝ := 0.4

/-- The initial state of the match after two games -/
def initial_state : MatchState := ⟨1, 1⟩

/-- The number of wins required to win the match -/
def wins_required : Nat := 3

/-- Calculates the probability of A winning the match given the current state -/
def prob_A_wins_match (state : MatchState) : ℝ :=
  sorry

/-- Calculates the expected number of additional games played -/
def expected_additional_games (state : MatchState) : ℝ :=
  sorry

theorem match_probabilities_and_expectation :
  prob_A_wins_match initial_state = 0.648 ∧
  expected_additional_games initial_state = 2.48 := by
  sorry

end match_probabilities_and_expectation_l3047_304725


namespace min_t_value_l3047_304795

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

-- Define the interval
def I : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem min_t_value : 
  (∃ (t : ℝ), ∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ t) ∧ 
  (∀ (s : ℝ), (∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ s) → s ≥ 20) :=
by sorry

end min_t_value_l3047_304795


namespace existence_implies_upper_bound_l3047_304708

theorem existence_implies_upper_bound (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 3 ∧ x^2 - 3*x - a > 0) → a < 4 := by
  sorry

end existence_implies_upper_bound_l3047_304708


namespace lisa_marbles_l3047_304741

/-- The number of marbles each person has -/
structure Marbles where
  connie : ℕ
  juan : ℕ
  mark : ℕ
  lisa : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : Marbles) : Prop :=
  m.connie = 323 ∧
  m.juan = m.connie + 175 ∧
  m.mark = 3 * m.juan ∧
  m.lisa = m.mark / 2 - 200

/-- The theorem stating that Lisa has 547 marbles -/
theorem lisa_marbles (m : Marbles) (h : marble_problem m) : m.lisa = 547 := by
  sorry

end lisa_marbles_l3047_304741


namespace average_transformation_l3047_304750

theorem average_transformation (x₁ x₂ x₃ : ℝ) (h : (x₁ + x₂ + x₃) / 3 = 8) :
  ((2 * x₁ - 1) + (2 * x₂ - 1) + (2 * x₃ - 1)) / 3 = 15 := by
  sorry

end average_transformation_l3047_304750


namespace sin_cos_lt_cos_sin_acute_l3047_304780

theorem sin_cos_lt_cos_sin_acute (x : ℝ) (h : 0 < x ∧ x < π / 2) : 
  Real.sin (Real.cos x) < Real.cos (Real.sin x) := by
  sorry

end sin_cos_lt_cos_sin_acute_l3047_304780


namespace sin_over_two_minus_cos_max_value_l3047_304788

theorem sin_over_two_minus_cos_max_value (x : ℝ) : 
  (Real.sin x) / (2 - Real.cos x) ≤ Real.sqrt 3 / 3 := by
  sorry

end sin_over_two_minus_cos_max_value_l3047_304788


namespace complex_modulus_inequality_l3047_304711

open Complex

theorem complex_modulus_inequality (z : ℂ) (h : abs z = 1) :
  abs ((z + 1) + Complex.I * (7 - z)) ≠ 5 * Real.sqrt 3 := by
  sorry

end complex_modulus_inequality_l3047_304711


namespace at_most_two_protocols_l3047_304703

/-- Represents a skier in the race -/
structure Skier :=
  (number : Nat)
  (startPosition : Nat)
  (finishPosition : Nat)
  (overtakes : Nat)
  (overtakenBy : Nat)

/-- Represents the race conditions -/
structure RaceConditions :=
  (skiers : List Skier)
  (totalSkiers : Nat)
  (h_totalSkiers : totalSkiers = 7)
  (h_startSequence : ∀ s : Skier, s ∈ skiers → s.number = s.startPosition)
  (h_constantSpeed : ∀ s : Skier, s ∈ skiers → s.overtakes + s.overtakenBy = 2)
  (h_uniqueFinish : ∀ s1 s2 : Skier, s1 ∈ skiers → s2 ∈ skiers → s1.finishPosition = s2.finishPosition → s1 = s2)

/-- The theorem to be proved -/
theorem at_most_two_protocols (rc : RaceConditions) : 
  (∃ p1 p2 : List Nat, 
    (∀ p : List Nat, p.length = rc.totalSkiers ∧ (∀ s : Skier, s ∈ rc.skiers → s.finishPosition = p.indexOf s.number + 1) → p = p1 ∨ p = p2) ∧
    p1 ≠ p2) :=
sorry

end at_most_two_protocols_l3047_304703


namespace soccer_team_wins_solution_l3047_304723

def soccer_team_wins (total_games wins losses draws : ℕ) : Prop :=
  total_games = wins + losses + draws ∧
  losses = 2 ∧
  3 * wins + draws = 46

theorem soccer_team_wins_solution :
  ∃ (wins losses draws : ℕ),
    soccer_team_wins 20 wins losses draws ∧ wins = 14 := by
  sorry

end soccer_team_wins_solution_l3047_304723


namespace marbles_left_l3047_304791

def initial_marbles : ℕ := 143
def marbles_given : ℕ := 73

theorem marbles_left : initial_marbles - marbles_given = 70 := by
  sorry

end marbles_left_l3047_304791


namespace trapezoid_diagonal_length_l3047_304776

/-- Represents a trapezoid ABCD with diagonal AC -/
structure Trapezoid where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Lengths
  AB : ℝ
  DC : ℝ
  AD : ℝ
  -- Properties
  is_trapezoid : (B.2 = C.2) ∧ (A.2 = D.2) -- BC parallel to AD
  AB_length : dist A B = AB
  DC_length : dist D C = DC
  AD_length : dist A D = AD

/-- The length of AC in the trapezoid is approximately 30.1 -/
theorem trapezoid_diagonal_length (t : Trapezoid) (h1 : t.AB = 15) (h2 : t.DC = 24) (h3 : t.AD = 7) :
  ∃ ε > 0, abs (dist t.A t.C - 30.1) < ε :=
sorry

end trapezoid_diagonal_length_l3047_304776


namespace officers_on_duty_l3047_304790

theorem officers_on_duty (total_female : ℕ) (on_duty : ℕ) 
  (h1 : total_female = 1000)
  (h2 : on_duty / 2 = total_female / 4) : 
  on_duty = 500 := by
  sorry

end officers_on_duty_l3047_304790


namespace associativity_of_mul_l3047_304714

-- Define the set S and its binary operation
variable {S : Type}
variable (add : S → S → S)

-- Define the properties of the set S
variable (h1 : ∀ (a b c : S), add (add a c) (add b c) = add a b)
variable (h2 : ∃ (e : S), (∀ (a : S), add a e = a ∧ add a a = e))

-- Define the * operation
def mul (add : S → S → S) (e : S) (a b : S) : S := add a (add e b)

-- State the theorem
theorem associativity_of_mul 
  (add : S → S → S) 
  (h1 : ∀ (a b c : S), add (add a c) (add b c) = add a b)
  (h2 : ∃ (e : S), (∀ (a : S), add a e = a ∧ add a a = e)) :
  ∀ (a b c : S), mul add (Classical.choose h2) (mul add (Classical.choose h2) a b) c = 
                 mul add (Classical.choose h2) a (mul add (Classical.choose h2) b c) :=
by
  sorry


end associativity_of_mul_l3047_304714


namespace smallest_prime_congruence_l3047_304712

theorem smallest_prime_congruence : 
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p = 71 ∧ 
    ∀ (q : ℕ), Nat.Prime q → q < p → 
      ¬(∃ (q_inv : ℕ), (q * q_inv) % 143 = 1 ∧ (q + q_inv) % 143 = 25) ∧
    ∃ (p_inv : ℕ), (p * p_inv) % 143 = 1 ∧ (p + p_inv) % 143 = 25 := by
  sorry

end smallest_prime_congruence_l3047_304712


namespace calculation_1_calculation_2_calculation_3_calculation_4_calculation_5_calculation_6_l3047_304792

theorem calculation_1 : 238 + 45 * 5 = 463 := by sorry

theorem calculation_2 : 65 * 4 - 128 = 132 := by sorry

theorem calculation_3 : 900 - 108 * 4 = 468 := by sorry

theorem calculation_4 : 369 + (512 - 215) = 666 := by sorry

theorem calculation_5 : 758 - 58 * 9 = 236 := by sorry

theorem calculation_6 : 105 * (81 / 9 - 3) = 630 := by sorry

end calculation_1_calculation_2_calculation_3_calculation_4_calculation_5_calculation_6_l3047_304792


namespace H_function_iff_non_decreasing_l3047_304732

/-- A function f: ℝ → ℝ is an H function if for any x₁ ≠ x₂, x₁f(x₁) + x₂f(x₂) ≥ x₁f(x₂) + x₂f(x₁) -/
def is_H_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ ≥ x₁ * f x₂ + x₂ * f x₁

/-- A function f: ℝ → ℝ is non-decreasing if for any x₁ < x₂, f(x₁) ≤ f(x₂) -/
def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

theorem H_function_iff_non_decreasing (f : ℝ → ℝ) :
  is_H_function f ↔ is_non_decreasing f := by
  sorry

end H_function_iff_non_decreasing_l3047_304732


namespace mapping_count_l3047_304775

-- Define the sets P and Q
variable (P Q : Type)

-- Define the conditions
variable (h1 : Fintype Q)
variable (h2 : Fintype.card Q = 3)
variable (h3 : Fintype P)
variable (h4 : (Fintype.card P) ^ (Fintype.card Q) = 81)

-- The theorem to prove
theorem mapping_count : (Fintype.card Q) ^ (Fintype.card P) = 64 := by
  sorry

end mapping_count_l3047_304775


namespace mod_inverse_five_mod_thirtythree_l3047_304713

theorem mod_inverse_five_mod_thirtythree :
  ∃ x : ℕ, x < 33 ∧ (5 * x) % 33 = 1 ∧ x = 20 := by
  sorry

end mod_inverse_five_mod_thirtythree_l3047_304713


namespace electric_vehicle_analysis_l3047_304729

-- Define the variables
variable (x : ℝ) -- Number of vehicles a skilled worker can install per month
variable (y : ℝ) -- Number of vehicles a new worker can install per month
variable (m : ℝ) -- Average cost per kilometer of the electric vehicle
variable (a : ℝ) -- Annual mileage

-- Define the theorem
theorem electric_vehicle_analysis :
  -- Part 1: Installation capacity
  (2 * x + y = 10 ∧ x + 3 * y = 10) →
  (x = 4 ∧ y = 2) ∧
  -- Part 2: Cost per kilometer
  (200 / m = 4 * (200 / (m + 0.6))) →
  m = 0.2 ∧
  -- Part 3: Annual cost comparison
  (0.2 * a + 6400 < 0.8 * a + 4000) →
  a > 4000 :=
by sorry

end electric_vehicle_analysis_l3047_304729


namespace solve_star_equation_l3047_304771

-- Define the operation ★
def star (a b : ℝ) : ℝ := a * b + 2 * b - a

-- Theorem statement
theorem solve_star_equation :
  ∀ x : ℝ, star 5 x = 37 → x = 6 := by
  sorry

end solve_star_equation_l3047_304771


namespace sachin_age_l3047_304717

/-- Proves that Sachin's age is 14 years given the conditions -/
theorem sachin_age (sachin rahul : ℕ) 
  (h1 : rahul = sachin + 4)
  (h2 : (sachin : ℚ) / rahul = 7 / 9) : 
  sachin = 14 := by
  sorry

end sachin_age_l3047_304717


namespace triple_equation_solutions_l3047_304751

theorem triple_equation_solutions :
  ∀ a b c : ℝ, 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
    a^2 + a*b = c ∧ 
    b^2 + b*c = a ∧ 
    c^2 + c*a = b →
    (a = 0 ∧ b = 0 ∧ c = 0) ∨ 
    (a = 1/2 ∧ b = 1/2 ∧ c = 1/2) := by
  sorry

end triple_equation_solutions_l3047_304751


namespace no_solution_for_floor_equation_l3047_304769

theorem no_solution_for_floor_equation :
  ¬ ∃ s : ℝ, (⌊s⌋ : ℝ) + s = 15.6 := by
  sorry

end no_solution_for_floor_equation_l3047_304769


namespace cupboard_cost_price_correct_l3047_304779

/-- The cost price of a cupboard satisfying the given conditions -/
def cupboard_cost_price : ℝ :=
  let below_cost_percentage : ℝ := 0.12
  let profit_percentage : ℝ := 0.12
  let additional_amount : ℝ := 1650
  
  -- Define the selling price as a function of the cost price
  let selling_price (cost : ℝ) : ℝ := cost * (1 - below_cost_percentage)
  
  -- Define the new selling price (with profit) as a function of the cost price
  let new_selling_price (cost : ℝ) : ℝ := cost * (1 + profit_percentage)
  
  -- The cost price that satisfies the conditions
  6875

/-- Theorem stating that the calculated cost price satisfies the given conditions -/
theorem cupboard_cost_price_correct : 
  let cost := cupboard_cost_price
  let below_cost_percentage : ℝ := 0.12
  let profit_percentage : ℝ := 0.12
  let additional_amount : ℝ := 1650
  let selling_price := cost * (1 - below_cost_percentage)
  let new_selling_price := cost * (1 + profit_percentage)
  (new_selling_price - selling_price = additional_amount) ∧ 
  (cost = 6875) :=
by sorry

#eval cupboard_cost_price

end cupboard_cost_price_correct_l3047_304779


namespace square_of_negative_three_x_squared_y_l3047_304728

theorem square_of_negative_three_x_squared_y (x y : ℝ) :
  (-3 * x^2 * y)^2 = 9 * x^4 * y^2 := by
  sorry

end square_of_negative_three_x_squared_y_l3047_304728
