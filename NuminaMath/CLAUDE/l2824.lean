import Mathlib

namespace NUMINAMATH_CALUDE_cube_sum_problem_l2824_282419

theorem cube_sum_problem (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a * b + a * c + b * c = 1) 
  (h3 : a * b * c = -2) : 
  a^3 + b^3 + c^3 = -6 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l2824_282419


namespace NUMINAMATH_CALUDE_retreat_speed_l2824_282479

theorem retreat_speed (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) :
  total_distance = 600 →
  total_time = 10 →
  return_speed = 75 →
  ∃ outbound_speed : ℝ,
    outbound_speed = 50 ∧
    total_time = (total_distance / 2) / outbound_speed + (total_distance / 2) / return_speed :=
by sorry

end NUMINAMATH_CALUDE_retreat_speed_l2824_282479


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l2824_282463

theorem relationship_between_exponents (a c e f : ℝ) (x y q z : ℝ) 
  (h1 : a^(3*x) = c^(4*q))
  (h2 : a^(3*x) = e)
  (h3 : c^(4*q) = e)
  (h4 : c^(2*y) = a^(5*z))
  (h5 : c^(2*y) = f)
  (h6 : a^(5*z) = f)
  (h7 : a ≠ 0)
  (h8 : c ≠ 0)
  (h9 : e > 0)
  (h10 : f > 0) :
  3*y = 10*q := by
sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l2824_282463


namespace NUMINAMATH_CALUDE_expression_evaluation_l2824_282426

theorem expression_evaluation : 
  (-2/3)^2023 * (3/2)^2022 = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2824_282426


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2824_282473

theorem infinite_series_sum : 
  let series := fun n : ℕ => (n : ℝ) / 8^n
  ∑' n, series n = 8 / 49 := by
sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2824_282473


namespace NUMINAMATH_CALUDE_oil_leaked_during_fix_correct_l2824_282407

/-- The amount of oil leaked while engineers were fixing the pipe -/
def oil_leaked_during_fix (total_leaked : ℕ) (leaked_before : ℕ) : ℕ :=
  total_leaked - leaked_before

/-- Theorem: The amount of oil leaked during fix is correct -/
theorem oil_leaked_during_fix_correct 
  (total_leaked : ℕ) 
  (leaked_before : ℕ) 
  (h1 : total_leaked = 6206)
  (h2 : leaked_before = 2475) :
  oil_leaked_during_fix total_leaked leaked_before = 3731 :=
by sorry

end NUMINAMATH_CALUDE_oil_leaked_during_fix_correct_l2824_282407


namespace NUMINAMATH_CALUDE_min_pencils_in_box_l2824_282478

theorem min_pencils_in_box (total_boxes : Nat) (total_pencils : Nat) (max_capacity : Nat)
  (h1 : total_boxes = 13)
  (h2 : total_pencils = 74)
  (h3 : max_capacity = 6) :
  ∃ (min_pencils : Nat), min_pencils = 2 ∧
    (∀ (box : Nat), box ≤ total_boxes → ∃ (pencils_in_box : Nat),
      pencils_in_box ≥ min_pencils ∧ pencils_in_box ≤ max_capacity) ∧
    (∃ (box : Nat), box ≤ total_boxes ∧ ∃ (pencils_in_box : Nat), pencils_in_box = min_pencils) :=
by
  sorry

end NUMINAMATH_CALUDE_min_pencils_in_box_l2824_282478


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_25_l2824_282456

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Checks if a FourDigitNumber is valid (each digit is less than 10) -/
def isValidFourDigitNumber (n : FourDigitNumber) : Prop :=
  n.1 < 10 ∧ n.2.1 < 10 ∧ n.2.2.1 < 10 ∧ n.2.2.2 < 10

/-- Calculates the sum of digits of a FourDigitNumber -/
def digitSum (n : FourDigitNumber) : Nat :=
  n.1 + n.2.1 + n.2.2.1 + n.2.2.2

/-- Converts a FourDigitNumber to its numerical value -/
def toNumber (n : FourDigitNumber) : Nat :=
  1000 * n.1 + 100 * n.2.1 + 10 * n.2.2.1 + n.2.2.2

/-- The main theorem stating that 9970 is the largest four-digit number with digit sum 25 -/
theorem largest_four_digit_sum_25 :
  ∀ n : FourDigitNumber,
    isValidFourDigitNumber n →
    digitSum n = 25 →
    toNumber n ≤ 9970 := by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_25_l2824_282456


namespace NUMINAMATH_CALUDE_license_plate_combinations_l2824_282481

/-- The number of consonants available for the license plate. -/
def num_consonants : ℕ := 21

/-- The number of vowels available for the license plate. -/
def num_vowels : ℕ := 5

/-- The number of digits available for the license plate. -/
def num_digits : ℕ := 10

/-- The total number of possible license plate combinations. -/
def total_combinations : ℕ := num_consonants^2 * num_vowels^2 * num_digits

/-- Theorem stating that the total number of license plate combinations is 110,250. -/
theorem license_plate_combinations : total_combinations = 110250 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l2824_282481


namespace NUMINAMATH_CALUDE_expression_evaluation_l2824_282464

theorem expression_evaluation : (50 - (3050 - 501))^2 + (3050 - (501 - 50)) = 6251600 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2824_282464


namespace NUMINAMATH_CALUDE_negation_of_implication_l2824_282484

theorem negation_of_implication (x y : ℝ) :
  ¬(x + y ≤ 0 → x ≤ 0 ∨ y ≤ 0) ↔ (x + y > 0 → x > 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2824_282484


namespace NUMINAMATH_CALUDE_equal_areas_imply_equal_dimensions_l2824_282415

theorem equal_areas_imply_equal_dimensions (square_side : ℝ) (rect_width : ℝ) (tri_base : ℝ) 
  (h1 : square_side = 4)
  (h2 : rect_width = 4)
  (h3 : tri_base = 8)
  (h4 : square_side ^ 2 = rect_width * (square_side ^ 2 / rect_width))
  (h5 : square_side ^ 2 = (tri_base * (2 * square_side ^ 2 / tri_base)) / 2) :
  square_side ^ 2 / rect_width = 4 ∧ 2 * square_side ^ 2 / tri_base = 4 :=
by sorry

end NUMINAMATH_CALUDE_equal_areas_imply_equal_dimensions_l2824_282415


namespace NUMINAMATH_CALUDE_square_side_length_of_unit_area_l2824_282498

/-- The side length of a square with area 1 is 1. -/
theorem square_side_length_of_unit_area : 
  ∀ s : ℝ, s > 0 → s * s = 1 → s = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_of_unit_area_l2824_282498


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_B_in_A_union_A_l2824_282475

-- Define set A as positive integers less than 9
def A : Set ℕ := {x | x > 0 ∧ x < 9}

-- Define set B
def B : Set ℕ := {1, 2, 3}

-- Define set A for the second part
def A' : Set ℝ := {x | -3 < x ∧ x < 1}

-- Define set B for the second part
def B' : Set ℝ := {x | 2 < x ∧ x < 10}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {1, 2, 3} := by sorry

-- Theorem for complement of B in A
theorem complement_B_in_A : A \ B = {4, 5, 6, 7, 8} := by sorry

-- Theorem for A' ∪ B'
theorem union_A'_B' : A' ∪ B' = {x | -3 < x ∧ x < 1 ∨ 2 < x ∧ x < 10} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_B_in_A_union_A_l2824_282475


namespace NUMINAMATH_CALUDE_max_profit_theorem_l2824_282453

/-- Represents the profit function for a mobile phone store -/
def profit_function (x : ℝ) : ℝ := -200 * x + 140000

/-- Represents the constraint on the number of type B phones -/
def constraint (x : ℝ) : Prop := 100 - x ≤ 3 * x

/-- Theorem stating the maximum profit and optimal purchase strategy -/
theorem max_profit_theorem :
  ∃ (x : ℝ),
    x ≥ 0 ∧
    x ≤ 100 ∧
    constraint x ∧
    profit_function x = 135000 ∧
    (∀ y : ℝ, y ≥ 0 → y ≤ 100 → constraint y → profit_function y ≤ profit_function x) ∧
    x = 25 :=
  sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l2824_282453


namespace NUMINAMATH_CALUDE_scaled_cylinder_volume_l2824_282496

/-- Theorem: Scaling a cylindrical container -/
theorem scaled_cylinder_volume (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  π * r^2 * h = 3 →
  π * (2*r)^2 * (4*h) = 48 := by
  sorry

end NUMINAMATH_CALUDE_scaled_cylinder_volume_l2824_282496


namespace NUMINAMATH_CALUDE_equation_a_l2824_282412

theorem equation_a (a : ℝ) (x : ℝ) : 
  (x + a) * (x + 2*a) * (x + 3*a) * (x + 4*a) = 3*a^4 ↔ 
  x = (-5*a + a*Real.sqrt 37)/2 ∨ x = (-5*a - a*Real.sqrt 37)/2 :=
sorry


end NUMINAMATH_CALUDE_equation_a_l2824_282412


namespace NUMINAMATH_CALUDE_worker_room_arrangement_l2824_282483

/-- The number of rooms and workers -/
def n : ℕ := 5

/-- The number of unchosen rooms -/
def k : ℕ := 2

/-- Represents whether each room choice is equally likely -/
def equal_probability : Prop := sorry

/-- Represents the condition that unchosen rooms are not adjacent -/
def non_adjacent_unchosen : Prop := sorry

/-- The number of ways to arrange workers in rooms with given conditions -/
def arrangement_count : ℕ := sorry

theorem worker_room_arrangement :
  arrangement_count = 900 :=
sorry

end NUMINAMATH_CALUDE_worker_room_arrangement_l2824_282483


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2824_282466

/-- Three points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Collinearity of three points -/
def collinear (p q r : Point3D) : Prop :=
  ∃ (t s : ℝ), q.x - p.x = t * (r.x - p.x) ∧ 
                q.y - p.y = t * (r.y - p.y) ∧
                q.z - p.z = t * (r.z - p.z) ∧
                q.x - p.x = s * (r.x - q.x) ∧
                q.y - p.y = s * (r.y - q.y) ∧
                q.z - p.z = s * (r.z - q.z)

theorem collinear_points_sum (a b : ℝ) :
  collinear (Point3D.mk 2 a b) (Point3D.mk a 3 b) (Point3D.mk a b 4) →
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2824_282466


namespace NUMINAMATH_CALUDE_inverse_g_sum_l2824_282406

-- Define the function g
def g (x : ℝ) : ℝ := x * |x| + 3 * x

-- State the theorem
theorem inverse_g_sum : 
  ∃ (a b : ℝ), g a = 9 ∧ g b = -121 ∧ a + b = (3 * Real.sqrt 5 - 23) / 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l2824_282406


namespace NUMINAMATH_CALUDE_parabola_vertex_l2824_282472

/-- The parabola equation -/
def parabola_equation (x : ℝ) : ℝ := x^2 - 4*x + 7

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 3)

/-- Theorem: The vertex of the parabola y = x^2 - 4x + 7 is at the point (2, 3) -/
theorem parabola_vertex :
  let (h, k) := vertex
  (∀ x, parabola_equation x = (x - h)^2 + k) ∧
  (∀ x, parabola_equation x ≥ parabola_equation h) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2824_282472


namespace NUMINAMATH_CALUDE_expected_value_specific_coin_l2824_282411

/-- A three-sided coin with probabilities and winnings for each outcome -/
structure ThreeSidedCoin where
  prob_heads : ℚ
  prob_tails : ℚ
  prob_edge : ℚ
  win_heads : ℚ
  win_tails : ℚ
  win_edge : ℚ

/-- The expected value of winnings for a three-sided coin flip -/
def expectedValue (coin : ThreeSidedCoin) : ℚ :=
  coin.prob_heads * coin.win_heads +
  coin.prob_tails * coin.win_tails +
  coin.prob_edge * coin.win_edge

/-- Theorem stating the expected value of winnings for a specific three-sided coin -/
theorem expected_value_specific_coin :
  ∃ (coin : ThreeSidedCoin),
    coin.prob_heads = 1/4 ∧
    coin.prob_tails = 3/4 - 1/20 ∧
    coin.prob_edge = 1/20 ∧
    coin.win_heads = 4 ∧
    coin.win_tails = -3 ∧
    coin.win_edge = -1 ∧
    coin.prob_heads + coin.prob_tails + coin.prob_edge = 1 ∧
    expectedValue coin = -23/20 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_specific_coin_l2824_282411


namespace NUMINAMATH_CALUDE_ice_cream_scoops_l2824_282413

/-- Proves that Aaron and Carson each bought 8 scoops of ice cream given the problem conditions --/
theorem ice_cream_scoops (aaron_savings : ℚ) (carson_savings : ℚ) 
  (restaurant_bill_fraction : ℚ) (service_charge : ℚ) (ice_cream_cost : ℚ) 
  (leftover_money : ℚ) :
  aaron_savings = 150 →
  carson_savings = 150 →
  restaurant_bill_fraction = 3/4 →
  service_charge = 15/100 →
  ice_cream_cost = 4 →
  leftover_money = 4 →
  ∃ (scoops : ℕ), scoops = 8 ∧ 
    (aaron_savings + carson_savings) * restaurant_bill_fraction + 
    2 * scoops * ice_cream_cost + 2 * leftover_money = 
    aaron_savings + carson_savings :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_scoops_l2824_282413


namespace NUMINAMATH_CALUDE_alice_oranges_sold_l2824_282477

/-- Given that Alice sold twice as many oranges as Emily, and they sold 180 oranges in total,
    prove that Alice sold 120 oranges. -/
theorem alice_oranges_sold (emily : ℕ) (h1 : emily + 2 * emily = 180) : 2 * emily = 120 := by
  sorry

end NUMINAMATH_CALUDE_alice_oranges_sold_l2824_282477


namespace NUMINAMATH_CALUDE_cindys_calculation_l2824_282439

theorem cindys_calculation (x : ℤ) : (x - 7) / 5 = 37 → (x - 5) / 7 = 26 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l2824_282439


namespace NUMINAMATH_CALUDE_twenty_is_eighty_percent_of_twentyfive_l2824_282467

theorem twenty_is_eighty_percent_of_twentyfive (x : ℝ) : 20 = 0.8 * x → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_twenty_is_eighty_percent_of_twentyfive_l2824_282467


namespace NUMINAMATH_CALUDE_discount_percentage_is_ten_percent_l2824_282431

/-- Calculates the discount percentage on a retail price given the wholesale price, retail price, and profit percentage. -/
def discount_percentage (wholesale_price retail_price profit_percentage : ℚ) : ℚ :=
  let profit := wholesale_price * profit_percentage / 100
  let selling_price := wholesale_price + profit
  let discount_amount := retail_price - selling_price
  (discount_amount / retail_price) * 100

/-- Proves that the discount percentage is 10% given the problem conditions. -/
theorem discount_percentage_is_ten_percent :
  discount_percentage 90 120 20 = 10 := by
  sorry

#eval discount_percentage 90 120 20

end NUMINAMATH_CALUDE_discount_percentage_is_ten_percent_l2824_282431


namespace NUMINAMATH_CALUDE_ends_with_k_zeros_l2824_282465

/-- A p-adic integer with a nonzero last digit -/
def NonZeroLastDigitPAdicInteger (p : ℕ) (a : ℕ) : Prop :=
  Nat.Prime p ∧ a % p ≠ 0

theorem ends_with_k_zeros (p k : ℕ) (a : ℕ) 
  (h_p : Nat.Prime p) 
  (h_a : NonZeroLastDigitPAdicInteger p a) 
  (h_k : k > 0) :
  (a^(p^(k-1) * (p-1)) - 1) % p^k = 0 := by
sorry

end NUMINAMATH_CALUDE_ends_with_k_zeros_l2824_282465


namespace NUMINAMATH_CALUDE_fraction_comparison_l2824_282490

theorem fraction_comparison : 
  (14/10 : ℚ) = 7/5 ∧ 
  (1 + 2/5 : ℚ) = 7/5 ∧ 
  (1 + 14/35 : ℚ) = 7/5 ∧ 
  (1 + 4/20 : ℚ) ≠ 7/5 ∧ 
  (1 + 3/15 : ℚ) ≠ 7/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2824_282490


namespace NUMINAMATH_CALUDE_contrapositive_statement_l2824_282489

theorem contrapositive_statement (a b : ℝ) :
  (a > 0 ∧ a + b < 0) → b < 0 := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_statement_l2824_282489


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2824_282494

theorem trigonometric_inequality (φ : Real) (h : φ ∈ Set.Ioo 0 (Real.pi / 2)) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2824_282494


namespace NUMINAMATH_CALUDE_cubic_sum_equals_one_l2824_282485

theorem cubic_sum_equals_one (a b : ℝ) (h : a + b = 1) : a^3 + 3*a*b + b^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_one_l2824_282485


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l2824_282420

theorem x_range_for_quadratic_inequality :
  (∀ m : ℝ, |m| ≤ 2 → ∀ x : ℝ, m * x^2 - 2 * x - m + 1 < 0) →
  ∃ a b : ℝ, a = (-1 + Real.sqrt 7) / 2 ∧ b = (1 + Real.sqrt 3) / 2 ∧
    ∀ x : ℝ, (a < x ∧ x < b) ↔ (∀ m : ℝ, |m| ≤ 2 → m * x^2 - 2 * x - m + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l2824_282420


namespace NUMINAMATH_CALUDE_cos_180_degrees_l2824_282417

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l2824_282417


namespace NUMINAMATH_CALUDE_sweet_potatoes_remaining_l2824_282436

def sweet_potatoes_problem (initial : ℕ) (sold_adams : ℕ) (sold_lenon : ℕ) (traded : ℕ) (donated : ℕ) : ℕ :=
  initial - (sold_adams + sold_lenon + traded + donated)

theorem sweet_potatoes_remaining :
  sweet_potatoes_problem 80 20 15 10 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sweet_potatoes_remaining_l2824_282436


namespace NUMINAMATH_CALUDE_count_theorem_l2824_282458

/-- Count of numbers between 100 and 799 with digits in strictly increasing order -/
def strictlyIncreasingCount : Nat := Nat.choose 7 3

/-- Count of numbers between 100 and 799 with last two digits equal -/
def lastTwoEqualCount : Nat := Nat.choose 7 2

/-- Total count of numbers between 100 and 799 with digits in strictly increasing order or equal to the last digit -/
def totalCount : Nat := strictlyIncreasingCount + lastTwoEqualCount

theorem count_theorem : totalCount = 56 := by sorry

end NUMINAMATH_CALUDE_count_theorem_l2824_282458


namespace NUMINAMATH_CALUDE_triangle_type_l2824_282471

theorem triangle_type (A B C : ℝ) (a b c : ℝ) 
  (h : a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.sin C) :
  A = π / 4 ∧ B = π / 4 ∧ C = π / 2 := by
  sorry

#check triangle_type

end NUMINAMATH_CALUDE_triangle_type_l2824_282471


namespace NUMINAMATH_CALUDE_same_monotonicity_implies_phi_value_l2824_282405

open Real

theorem same_monotonicity_implies_phi_value (φ : Real) :
  (∀ x ∈ Set.Icc 0 (π / 2), 
    (∀ y ∈ Set.Icc 0 (π / 2), x < y → cos (2 * x) > cos (2 * y)) ↔ 
    (∀ y ∈ Set.Icc 0 (π / 2), x < y → sin (x + φ) > sin (y + φ))) →
  φ = π / 2 := by
sorry

end NUMINAMATH_CALUDE_same_monotonicity_implies_phi_value_l2824_282405


namespace NUMINAMATH_CALUDE_parabola_b_value_l2824_282424

/-- Given a parabola y = x^2 + ax + b passing through (2, 5) and (-2, -11), prove b = -7 -/
theorem parabola_b_value (a b : ℝ) : 
  (5 = 2^2 + 2*a + b) ∧ (-11 = (-2)^2 + (-2)*a + b) → b = -7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_b_value_l2824_282424


namespace NUMINAMATH_CALUDE_jenny_tim_age_difference_l2824_282470

/-- Represents the ages of family members --/
structure FamilyAges where
  tim : ℕ
  rommel : ℕ
  jenny : ℕ
  uncle : ℕ
  aunt : ℚ

/-- Defines the relationships between family members' ages --/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.tim = 5 ∧
  ages.rommel = 3 * ages.tim ∧
  ages.jenny = ages.rommel + 2 ∧
  ages.uncle = 2 * (ages.rommel + ages.jenny) ∧
  ages.aunt = (ages.uncle + ages.jenny) / 2

/-- Theorem stating the age difference between Jenny and Tim --/
theorem jenny_tim_age_difference (ages : FamilyAges) 
  (h : validFamilyAges ages) : ages.jenny - ages.tim = 12 := by
  sorry

end NUMINAMATH_CALUDE_jenny_tim_age_difference_l2824_282470


namespace NUMINAMATH_CALUDE_school_absence_percentage_l2824_282474

theorem school_absence_percentage (total_students boys girls : ℕ) 
  (h_total : total_students = 180)
  (h_boys : boys = 100)
  (h_girls : girls = 80)
  (h_sum : total_students = boys + girls)
  (absent_boys : ℕ := boys / 5)
  (absent_girls : ℕ := girls / 4)
  (total_absent : ℕ := absent_boys + absent_girls) :
  (total_absent : ℚ) / total_students * 100 = 40 / 180 * 100 := by
sorry

end NUMINAMATH_CALUDE_school_absence_percentage_l2824_282474


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2824_282448

/-- Represents the state of the game -/
inductive GameState
  | WinningPosition
  | LosingPosition

/-- Determines if a move is valid according to the game rules -/
def validMove (n : ℕ) (k : ℕ) : Prop :=
  k ≥ 1 ∧ 
  ((n % 2 = 0 ∧ k % 2 = 0 ∧ k ≤ n / 2) ∨ 
   (n % 2 = 1 ∧ k % 2 = 1 ∧ n / 2 ≤ k ∧ k ≤ n))

/-- Determines the game state for a given number of marbles -/
def gameState (n : ℕ) : GameState :=
  if n = 2^17 - 2 then GameState.LosingPosition else GameState.WinningPosition

/-- The main theorem to prove -/
theorem smallest_winning_number : 
  (∀ n, 100000 ≤ n ∧ n < 131070 → gameState n = GameState.WinningPosition) ∧
  gameState 131070 = GameState.LosingPosition :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2824_282448


namespace NUMINAMATH_CALUDE_factor_sum_l2824_282416

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 - 4*X + 8) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 64 :=
by sorry

end NUMINAMATH_CALUDE_factor_sum_l2824_282416


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l2824_282488

/-- Given a train and platform with specific lengths, and the time taken to cross a post,
    calculate the time taken to cross the platform. -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_cross_post : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 350)
  (h3 : time_to_cross_post = 18) :
  (train_length + platform_length) / (train_length / time_to_cross_post) = 39 :=
sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l2824_282488


namespace NUMINAMATH_CALUDE_equation_solution_l2824_282421

theorem equation_solution (x : ℝ) (h : x ≠ 0) : 4 / x^2 = x / 16 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2824_282421


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2824_282432

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a = Real.sqrt 2 ∧
  b = 2 ∧
  Real.sin B + Real.cos B = Real.sqrt 2 →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2824_282432


namespace NUMINAMATH_CALUDE_digit_equation_solutions_l2824_282450

theorem digit_equation_solutions (n : ℕ) (x y z : ℕ) :
  n ≥ 2 →
  let a : ℚ := x * (10^n - 1) / 9
  let b : ℚ := y * (10^n - 1) / 9
  let c : ℚ := z * (10^(2*n) - 1) / 9
  a^2 + b = c →
  ((x = 3 ∧ y = 2 ∧ z = 1) ∨
   (x = 6 ∧ y = 8 ∧ z = 4) ∨
   (x = 8 ∧ y = 3 ∧ z = 7 ∧ n = 2)) :=
by sorry

end NUMINAMATH_CALUDE_digit_equation_solutions_l2824_282450


namespace NUMINAMATH_CALUDE_thermometer_distribution_methods_l2824_282449

/-- The number of ways to distribute thermometers among classes. -/
def distribute_thermometers (total_thermometers : ℕ) (num_classes : ℕ) (min_per_class : ℕ) : ℕ :=
  Nat.choose num_classes 1 + 
  2 * Nat.choose num_classes 2 + 
  Nat.choose num_classes 3

/-- Theorem stating the number of distribution methods for the given problem. -/
theorem thermometer_distribution_methods : 
  distribute_thermometers 23 10 2 = 220 := by
  sorry

#eval distribute_thermometers 23 10 2

end NUMINAMATH_CALUDE_thermometer_distribution_methods_l2824_282449


namespace NUMINAMATH_CALUDE_non_negative_iff_geq_zero_l2824_282442

theorem non_negative_iff_geq_zero (a b : ℝ) :
  (a ≥ 0 ∧ b ≥ 0) ↔ (a ≥ 0 ∧ b ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_non_negative_iff_geq_zero_l2824_282442


namespace NUMINAMATH_CALUDE_journey_time_change_l2824_282423

/-- Proves that for a journey of 40 km, if increasing the speed by 3 kmph reduces
    the time by 40 minutes, then decreasing the speed by 2 kmph from the original
    speed increases the time by 40 minutes. -/
theorem journey_time_change (v : ℝ) (h1 : v > 0) : 
  (40 / v - 40 / (v + 3) = 2 / 3) → 
  (40 / (v - 2) - 40 / v = 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_journey_time_change_l2824_282423


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l2824_282457

def initial_amount : ℚ := 120

def sandwich_fraction : ℚ := 1/5
def museum_fraction : ℚ := 1/6
def book_fraction : ℚ := 1/2

def remaining_amount : ℚ := 
  initial_amount - (initial_amount * sandwich_fraction + 
                    initial_amount * museum_fraction + 
                    initial_amount * book_fraction)

theorem jennifer_remaining_money : remaining_amount = 16 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l2824_282457


namespace NUMINAMATH_CALUDE_cindy_calculation_l2824_282480

theorem cindy_calculation (x : ℝ) : (2 * (x - 9)) / 6 = 36 → (x - 12) / 8 = 13.125 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l2824_282480


namespace NUMINAMATH_CALUDE_max_ab_value_l2824_282444

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a * log x + (a + 1) * x - (1/2) * x^2

theorem max_ab_value (a : ℝ) (h_a : a > 0) :
  (∀ x > 0, f a x ≥ -(1/2) * x^2 + a * x + b) →
  (∃ c : ℝ, c = Real.exp 1 / 2 ∧ ∀ b : ℝ, a * b ≤ c) :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l2824_282444


namespace NUMINAMATH_CALUDE_solve_equation_l2824_282445

theorem solve_equation (y : ℚ) (h : (1 : ℚ) / 3 - (1 : ℚ) / 4 = 1 / y) : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2824_282445


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2824_282462

/-- A quadratic function with vertex (1,16) and roots 8 units apart -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 15

/-- The function g(x) defined in terms of f(x) and a parameter a -/
def g (a : ℝ) (x : ℝ) : ℝ := (2 - 2*a)*x - f x

theorem quadratic_function_properties :
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ |x₁ - x₂| = 8) ∧
  (∀ x : ℝ, f x ≤ f 1) ∧
  f 1 = 16 ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc 0 2, Monotone (g a)) ↔ a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2824_282462


namespace NUMINAMATH_CALUDE_translation_right_5_units_l2824_282491

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_right_5_units :
  let P : Point := { x := -2, y := -3 }
  let P' : Point := translateRight P 5
  P'.x = 3 ∧ P'.y = -3 := by
  sorry

end NUMINAMATH_CALUDE_translation_right_5_units_l2824_282491


namespace NUMINAMATH_CALUDE_copper_percentage_in_second_alloy_l2824_282487

/-- Given a mixture of two alloys, prove the copper percentage in the second alloy -/
theorem copper_percentage_in_second_alloy
  (total_mixture : ℝ)
  (desired_copper_percentage : ℝ)
  (first_alloy_amount : ℝ)
  (first_alloy_copper_percentage : ℝ)
  (h_total_mixture : total_mixture = 100)
  (h_desired_copper_percentage : desired_copper_percentage = 24.9)
  (h_first_alloy_amount : first_alloy_amount = 30)
  (h_first_alloy_copper_percentage : first_alloy_copper_percentage = 20)
  : ∃ (second_alloy_copper_percentage : ℝ),
    second_alloy_copper_percentage = 27 ∧
    (first_alloy_amount * first_alloy_copper_percentage / 100 +
     (total_mixture - first_alloy_amount) * second_alloy_copper_percentage / 100 =
     total_mixture * desired_copper_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_copper_percentage_in_second_alloy_l2824_282487


namespace NUMINAMATH_CALUDE_power_division_seventeen_l2824_282497

theorem power_division_seventeen : (17 : ℕ)^9 / (17 : ℕ)^7 = 289 := by sorry

end NUMINAMATH_CALUDE_power_division_seventeen_l2824_282497


namespace NUMINAMATH_CALUDE_book_pages_calculation_l2824_282403

theorem book_pages_calculation (pages_remaining : ℕ) (percentage_read : ℚ) : 
  pages_remaining = 320 ∧ percentage_read = 1/5 → 
  pages_remaining / (1 - percentage_read) = 400 := by
sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l2824_282403


namespace NUMINAMATH_CALUDE_variance_of_sick_cows_l2824_282451

/-- The variance of a binomial distribution with n trials and probability p --/
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- The number of cows in the pasture --/
def num_cows : ℕ := 10

/-- The incidence rate of the disease --/
def incidence_rate : ℝ := 0.02

/-- Theorem stating that the variance of the number of sick cows is 0.196 --/
theorem variance_of_sick_cows :
  binomial_variance num_cows incidence_rate = 0.196 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_sick_cows_l2824_282451


namespace NUMINAMATH_CALUDE_divisibility_by_three_l2824_282438

theorem divisibility_by_three (a b : ℕ) : 
  (3 ∣ (a * b)) → ¬(¬(3 ∣ a) ∧ ¬(3 ∣ b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l2824_282438


namespace NUMINAMATH_CALUDE_min_sum_intercepts_l2824_282441

/-- The minimum sum of intercepts for a line passing through (1, 2) -/
theorem min_sum_intercepts : 
  ∀ a b : ℝ, a > 0 → b > 0 → 
  (1 : ℝ) / a + (2 : ℝ) / b = 1 → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → (1 : ℝ) / a' + (2 : ℝ) / b' = 1 → a + b ≤ a' + b') → 
  a + b = 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_intercepts_l2824_282441


namespace NUMINAMATH_CALUDE_coords_of_A_wrt_origin_l2824_282469

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin of the Cartesian coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- The coordinates of a point with respect to the origin -/
def coordsWrtOrigin (p : Point) : ℝ × ℝ := (p.x, p.y)

/-- Theorem: The coordinates of point A(-1,3) with respect to the origin are (-1,3) -/
theorem coords_of_A_wrt_origin :
  let A : Point := ⟨-1, 3⟩
  coordsWrtOrigin A = (-1, 3) := by sorry

end NUMINAMATH_CALUDE_coords_of_A_wrt_origin_l2824_282469


namespace NUMINAMATH_CALUDE_marquita_garden_length_l2824_282440

-- Define the number of gardens for each person
def mancino_gardens : ℕ := 3
def marquita_gardens : ℕ := 2

-- Define the dimensions of Mancino's gardens
def mancino_garden_length : ℕ := 16
def mancino_garden_width : ℕ := 5

-- Define the width of Marquita's gardens
def marquita_garden_width : ℕ := 4

-- Define the total area of all gardens
def total_area : ℕ := 304

-- Theorem to prove
theorem marquita_garden_length :
  ∃ (l : ℕ), 
    mancino_gardens * mancino_garden_length * mancino_garden_width +
    marquita_gardens * l * marquita_garden_width = total_area ∧
    l = 8 := by
  sorry

end NUMINAMATH_CALUDE_marquita_garden_length_l2824_282440


namespace NUMINAMATH_CALUDE_ax5_plus_by5_l2824_282495

theorem ax5_plus_by5 (a b x y : ℝ) 
  (h1 : a * x + b * y = 4)
  (h2 : a * x^2 + b * y^2 = 10)
  (h3 : a * x^3 + b * y^3 = 28)
  (h4 : a * x^4 + b * y^4 = 60) :
  a * x^5 + b * y^5 = 229 + 1/3 := by
sorry

end NUMINAMATH_CALUDE_ax5_plus_by5_l2824_282495


namespace NUMINAMATH_CALUDE_polynomial_value_at_one_l2824_282476

-- Define the polynomial P(x)
def P (r : ℝ) (x : ℝ) : ℝ := x^3 + x^2 - r^2*x - 2020

-- Define the roots of P(x)
variable (r s t : ℝ)

-- State the theorem
theorem polynomial_value_at_one (hr : P r r = 0) (hs : P r s = 0) (ht : P r t = 0) :
  P r 1 = -4038 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_one_l2824_282476


namespace NUMINAMATH_CALUDE_company_workforce_l2824_282454

theorem company_workforce (initial_total : ℕ) : 
  (initial_total * 3 / 5 : ℚ) = initial_total * 0.6 →
  (initial_total * 3 / 5 : ℚ) / (initial_total + 28) = 0.55 →
  initial_total + 28 = 336 := by
sorry

end NUMINAMATH_CALUDE_company_workforce_l2824_282454


namespace NUMINAMATH_CALUDE_kids_difference_l2824_282437

theorem kids_difference (monday tuesday : ℕ) 
  (h1 : monday = 18) 
  (h2 : tuesday = 10) : 
  monday - tuesday = 8 := by
sorry

end NUMINAMATH_CALUDE_kids_difference_l2824_282437


namespace NUMINAMATH_CALUDE_max_ab_value_l2824_282422

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * (x - 1)

theorem max_ab_value (a : ℝ) (h : a > 0) :
  (∃ b : ℝ, ∀ x : ℝ, f a x ≥ b) →
  (∃ M : ℝ, M = (exp 3) / 2 ∧ ∀ b : ℝ, (∀ x : ℝ, f a x ≥ b) → a * b ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l2824_282422


namespace NUMINAMATH_CALUDE_sandbox_volume_l2824_282434

def sandbox_length : ℝ := 312
def sandbox_width : ℝ := 146
def sandbox_depth : ℝ := 56

theorem sandbox_volume :
  sandbox_length * sandbox_width * sandbox_depth = 2555520 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_volume_l2824_282434


namespace NUMINAMATH_CALUDE_tom_helicopter_rental_cost_l2824_282468

/-- The total cost for renting a helicopter -/
def helicopter_rental_cost (hours_per_day : ℕ) (days : ℕ) (hourly_rate : ℕ) : ℕ :=
  hours_per_day * days * hourly_rate

/-- Theorem stating the total cost for Tom's helicopter rental -/
theorem tom_helicopter_rental_cost :
  helicopter_rental_cost 2 3 75 = 450 := by
  sorry

end NUMINAMATH_CALUDE_tom_helicopter_rental_cost_l2824_282468


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l2824_282433

-- Define the total number of people
def total_people : ℕ := 12

-- Define the number of men
def num_men : ℕ := 8

-- Define the number of women
def num_women : ℕ := 4

-- Define the number of people to be selected
def num_selected : ℕ := 4

-- Define the probability of selecting at least one woman
def prob_at_least_one_woman : ℚ := 85 / 99

-- Theorem statement
theorem probability_at_least_one_woman :
  (1 : ℚ) - (num_men.choose num_selected : ℚ) / (total_people.choose num_selected : ℚ) = prob_at_least_one_woman :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l2824_282433


namespace NUMINAMATH_CALUDE_problem_solution_l2824_282428

theorem problem_solution (a b : ℕ) (h1 : a > b) (h2 : (a + b) + (3 * a + a * b - b) + 4 * a / b = 64) :
  a = 8 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2824_282428


namespace NUMINAMATH_CALUDE_probability_at_least_one_white_l2824_282435

/-- The probability of drawing at least one white ball from a box -/
theorem probability_at_least_one_white (total : ℕ) (white : ℕ) (red : ℕ) (draw : ℕ) :
  total = white + red →
  white = 8 →
  red = 2 →
  draw = 2 →
  (Nat.choose white 1 * Nat.choose red 1 + Nat.choose white 2 * Nat.choose red 0) / Nat.choose total draw = 44 / 45 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_white_l2824_282435


namespace NUMINAMATH_CALUDE_prob_at_least_one_contract_l2824_282460

/-- The probability of getting at least one contract given specific probabilities for hardware and software contracts -/
theorem prob_at_least_one_contract 
  (p_hardware : ℝ) 
  (p_not_software : ℝ) 
  (p_both : ℝ) 
  (h1 : p_hardware = 4/5)
  (h2 : p_not_software = 3/5)
  (h3 : p_both = 0.3) :
  p_hardware + (1 - p_not_software) - p_both = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_contract_l2824_282460


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_equals_one_fourth_more_than_fifty_l2824_282404

theorem thirty_percent_less_than_ninety_equals_one_fourth_more_than_fifty : 
  (90 : ℝ) * (1 - 0.3) = 50 * (1 + 0.25) := by sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_equals_one_fourth_more_than_fifty_l2824_282404


namespace NUMINAMATH_CALUDE_triangle_formation_l2824_282418

/-- Check if three lengths can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given two stick lengths -/
def stick1 : ℝ := 3
def stick2 : ℝ := 5

theorem triangle_formation :
  ¬(canFormTriangle stick1 stick2 2) ∧
  (canFormTriangle stick1 stick2 3) ∧
  (canFormTriangle stick1 stick2 4) ∧
  (canFormTriangle stick1 stick2 6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_formation_l2824_282418


namespace NUMINAMATH_CALUDE_system_equations_solution_system_inequalities_solution_l2824_282455

-- Part 1: System of equations
theorem system_equations_solution (x y : ℝ) : 
  (x = 4*y + 1 ∧ 2*x - 5*y = 8) → (x = 9 ∧ y = 2) := by sorry

-- Part 2: System of inequalities
theorem system_inequalities_solution (x : ℝ) :
  (4*x - 5 ≤ 3 ∧ (x - 1) / 3 < (2*x + 1) / 5) ↔ (-8 < x ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_system_equations_solution_system_inequalities_solution_l2824_282455


namespace NUMINAMATH_CALUDE_gcd_75_360_l2824_282425

theorem gcd_75_360 : Nat.gcd 75 360 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_75_360_l2824_282425


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2824_282427

-- Define the sets M and N
def M : Set Nat := {1, 2}
def N : Set Nat := {2, 3}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2824_282427


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_squared_l2824_282459

theorem isosceles_right_triangle_hypotenuse_squared 
  (u v w : ℂ) (s t : ℝ) (k : ℝ) : 
  (∀ z : ℂ, z^3 + 2*z^2 + s*z + t = 0 ↔ z = u ∨ z = v ∨ z = w) → 
  Complex.abs u^2 + Complex.abs v^2 + Complex.abs w^2 = 350 →
  ∃ (x y : ℝ), 
    (Complex.abs (u - v))^2 = x^2 + y^2 ∧ 
    (Complex.abs (v - w))^2 = x^2 + y^2 ∧
    (Complex.abs (w - u))^2 = x^2 + y^2 ∧
    k^2 = (Complex.abs (w - u))^2 →
  k^2 = 525 := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_squared_l2824_282459


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2824_282402

theorem sqrt_equation_solution (y : ℝ) :
  y > 2 → (Real.sqrt (7 * y) / Real.sqrt (4 * (y - 2)) = 3) → y = 72 / 29 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2824_282402


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2824_282493

theorem regular_polygon_sides (central_angle : ℝ) (h : central_angle = 36) :
  (360 : ℝ) / central_angle = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2824_282493


namespace NUMINAMATH_CALUDE_school_students_count_l2824_282492

theorem school_students_count (blue_percent : ℚ) (red_percent : ℚ) (green_percent : ℚ) (other_count : ℕ) :
  blue_percent = 44/100 →
  red_percent = 28/100 →
  green_percent = 10/100 →
  other_count = 162 →
  ∃ (total : ℕ), 
    (blue_percent + red_percent + green_percent < 1) ∧
    (1 - (blue_percent + red_percent + green_percent)) * total = other_count ∧
    total = 900 :=
by
  sorry

end NUMINAMATH_CALUDE_school_students_count_l2824_282492


namespace NUMINAMATH_CALUDE_product_remainder_mod_seven_l2824_282401

theorem product_remainder_mod_seven : (1233 * 1984 * 2006 * 2021) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_seven_l2824_282401


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l2824_282429

def f (x : ℝ) : ℝ := x^2

theorem intersection_x_coordinate 
  (A B C E : ℝ × ℝ) 
  (hA : A = (2, f 2)) 
  (hB : B = (8, f 8)) 
  (hC : C.1 = (A.1 + B.1) / 2 ∧ C.2 = (A.2 + B.2) / 2) 
  (hE : E.1^2 = E.2 ∧ E.2 = C.2) : 
  E.1 = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l2824_282429


namespace NUMINAMATH_CALUDE_smallest_x_value_l2824_282410

theorem smallest_x_value (x : ℝ) : 
  (((15 * x^2 - 40 * x + 20) / (4 * x - 3)) + 7 * x = 8 * x - 3) →
  x ≥ (25 - Real.sqrt 141) / 22 ∧
  ∃ y, y = (25 - Real.sqrt 141) / 22 ∧ 
     ((15 * y^2 - 40 * y + 20) / (4 * y - 3)) + 7 * y = 8 * y - 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2824_282410


namespace NUMINAMATH_CALUDE_max_sum_x_y_is_seven_l2824_282414

theorem max_sum_x_y_is_seven (x y : ℕ+) (h : x.val^4 = (x.val - 1) * (y.val^3 - 23) - 1) :
  x.val + y.val ≤ 7 ∧ ∃ (x₀ y₀ : ℕ+), x₀.val^4 = (x₀.val - 1) * (y₀.val^3 - 23) - 1 ∧ x₀.val + y₀.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_x_y_is_seven_l2824_282414


namespace NUMINAMATH_CALUDE_friday_night_revenue_l2824_282486

/-- Represents the revenue calculation for a movie theater --/
def theater_revenue (matinee_price evening_price opening_price popcorn_price : ℕ)
                    (matinee_customers evening_customers opening_customers : ℕ) : ℕ :=
  let total_customers := matinee_customers + evening_customers + opening_customers
  let popcorn_sales := total_customers / 2
  (matinee_price * matinee_customers) +
  (evening_price * evening_customers) +
  (opening_price * opening_customers) +
  (popcorn_price * popcorn_sales)

/-- Theorem stating the total revenue of the theater on Friday night --/
theorem friday_night_revenue :
  theater_revenue 5 7 10 10 32 40 58 = 1670 := by
  sorry

end NUMINAMATH_CALUDE_friday_night_revenue_l2824_282486


namespace NUMINAMATH_CALUDE_systematic_sampling_20_4_l2824_282430

def is_systematic_sample (n : ℕ) (k : ℕ) (sample : List ℕ) : Prop :=
  sample.length = k ∧
  ∀ i, i ∈ sample → i ≤ n ∧
  ∀ i j, i < j → i ∈ sample → j ∈ sample → (j - i) = n / k

theorem systematic_sampling_20_4 :
  is_systematic_sample 20 4 [5, 10, 15, 20] := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_20_4_l2824_282430


namespace NUMINAMATH_CALUDE_sequence_sum_l2824_282452

theorem sequence_sum (P Q R S T U V : ℝ) : 
  S = 7 ∧ 
  P + Q + R = 27 ∧ 
  Q + R + S = 27 ∧ 
  R + S + T = 27 ∧ 
  S + T + U = 27 ∧ 
  T + U + V = 27 → 
  P + V = 0 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_l2824_282452


namespace NUMINAMATH_CALUDE_emily_chocolate_sales_l2824_282443

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (bars_left : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - bars_left) * price_per_bar

/-- Proves that Emily makes $20 from selling chocolate bars -/
theorem emily_chocolate_sales : money_made 8 3 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_emily_chocolate_sales_l2824_282443


namespace NUMINAMATH_CALUDE_cats_count_l2824_282400

/-- Represents the number of animals in a wildlife refuge --/
structure WildlifeRefuge where
  total_animals : ℕ
  birds : ℕ
  mammals : ℕ
  cats : ℕ
  dogs : ℕ

/-- The conditions of the wildlife refuge problem --/
def wildlife_refuge_conditions (w : WildlifeRefuge) : Prop :=
  w.total_animals = 1200 ∧
  w.birds = w.mammals + 145 ∧
  w.cats = w.dogs + 75 ∧
  w.mammals = w.cats + w.dogs ∧
  w.total_animals = w.birds + w.mammals

/-- The theorem stating that under the given conditions, the number of cats is 301 --/
theorem cats_count (w : WildlifeRefuge) :
  wildlife_refuge_conditions w → w.cats = 301 := by
  sorry


end NUMINAMATH_CALUDE_cats_count_l2824_282400


namespace NUMINAMATH_CALUDE_sunglasses_hat_probability_l2824_282408

theorem sunglasses_hat_probability (total_sunglasses : ℕ) (total_hats : ℕ) 
  (prob_hat_and_sunglasses : ℚ) :
  total_sunglasses = 80 →
  total_hats = 60 →
  prob_hat_and_sunglasses = 1/3 →
  (prob_hat_and_sunglasses * total_hats) / total_sunglasses = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_hat_probability_l2824_282408


namespace NUMINAMATH_CALUDE_classroom_notebooks_l2824_282447

theorem classroom_notebooks (total_students : ℕ) 
  (h1 : total_students = 28)
  (group1_notebooks : ℕ) (h2 : group1_notebooks = 5)
  (group2_notebooks : ℕ) (h3 : group2_notebooks = 3)
  (group3_notebooks : ℕ) (h4 : group3_notebooks = 7) :
  (total_students / 3) * group1_notebooks +
  (total_students / 3) * group2_notebooks +
  (total_students - 2 * (total_students / 3)) * group3_notebooks = 142 :=
by sorry

end NUMINAMATH_CALUDE_classroom_notebooks_l2824_282447


namespace NUMINAMATH_CALUDE_basketball_substitutions_remainder_l2824_282482

/-- Number of ways to make substitutions in a basketball game -/
def substitution_ways (total_players starters max_substitutions : ℕ) : ℕ :=
  let substitutes := total_players - starters
  let a0 := 1  -- No substitutions
  let a1 := starters * substitutes  -- One substitution
  let a2 := a1 * (starters - 1) * (substitutes - 1)  -- Two substitutions
  let a3 := a2 * (starters - 2) * (substitutes - 2)  -- Three substitutions
  let a4 := a3 * (starters - 3) * (substitutes - 3)  -- Four substitutions
  a0 + a1 + a2 + a3 + a4

/-- Theorem stating the remainder when the number of substitution ways is divided by 1000 -/
theorem basketball_substitutions_remainder :
  substitution_ways 14 5 4 % 1000 = 606 := by
  sorry

end NUMINAMATH_CALUDE_basketball_substitutions_remainder_l2824_282482


namespace NUMINAMATH_CALUDE_toy_store_shelves_l2824_282461

/-- The number of shelves needed to display bears in a toy store. -/
def shelves_needed (initial_stock new_shipment bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

/-- Theorem stating that the toy store used 2 shelves to display the bears. -/
theorem toy_store_shelves :
  shelves_needed 5 7 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l2824_282461


namespace NUMINAMATH_CALUDE_function_with_same_length_image_l2824_282409

-- Define the property for f
def HasSameLengthImage (f : ℝ → ℝ) : Prop :=
  ∀ (a b : ℝ), a < b → ∃ (c d : ℝ), c < d ∧ 
    (Set.Ioo c d = f '' Set.Ioo a b) ∧ 
    (d - c = b - a)

-- State the theorem
theorem function_with_same_length_image (f : ℝ → ℝ) 
  (h : HasSameLengthImage f) : 
  ∃ (C : ℝ), (∀ x, f x = x + C) ∨ (∀ x, f x = -x + C) := by
  sorry

end NUMINAMATH_CALUDE_function_with_same_length_image_l2824_282409


namespace NUMINAMATH_CALUDE_product_equals_zero_l2824_282499

theorem product_equals_zero (a : ℤ) (h : a = 11) :
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l2824_282499


namespace NUMINAMATH_CALUDE_painting_efficiency_theorem_l2824_282446

/-- Represents the efficiency of painting classrooms -/
structure PaintingEfficiency where
  groups : ℕ
  workers_per_group : ℕ
  hours_per_day : ℕ
  classrooms : ℕ
  days : ℚ

/-- The initial painting scenario -/
def initial_scenario : PaintingEfficiency :=
  { groups := 6
  , workers_per_group := 6
  , hours_per_day := 6
  , classrooms := 6
  , days := 6 }

/-- The new painting scenario -/
def new_scenario : PaintingEfficiency :=
  { groups := 8
  , workers_per_group := 8
  , hours_per_day := 8
  , classrooms := 8
  , days := 27/8 }

/-- Calculates the painting rate (classrooms per worker-hour) -/
def painting_rate (p : PaintingEfficiency) : ℚ :=
  p.classrooms / (p.groups * p.workers_per_group * p.hours_per_day * p.days)

theorem painting_efficiency_theorem :
  painting_rate initial_scenario = painting_rate new_scenario := by
  sorry

#check painting_efficiency_theorem

end NUMINAMATH_CALUDE_painting_efficiency_theorem_l2824_282446
