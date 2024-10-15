import Mathlib

namespace NUMINAMATH_CALUDE_jerry_lawsuit_amount_correct_l730_73064

/-- Calculates the amount Jerry gets from his lawsuit --/
def jerryLawsuitAmount (annualSalary : ℕ) (years : ℕ) (medicalBills : ℕ) (punitiveMultiplier : ℕ) (awardedPercentage : ℚ) : ℚ :=
  let totalSalary := annualSalary * years
  let directDamages := totalSalary + medicalBills
  let punitiveDamages := directDamages * punitiveMultiplier
  let totalAsked := directDamages + punitiveDamages
  totalAsked * awardedPercentage

theorem jerry_lawsuit_amount_correct :
  jerryLawsuitAmount 50000 30 200000 3 (4/5) = 5440000 := by
  sorry

#eval jerryLawsuitAmount 50000 30 200000 3 (4/5)

end NUMINAMATH_CALUDE_jerry_lawsuit_amount_correct_l730_73064


namespace NUMINAMATH_CALUDE_max_value_of_expression_l730_73029

theorem max_value_of_expression (x y z : Real) 
  (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) (hz : 0 < z ∧ z ≤ 1) :
  let A := (Real.sqrt (8 * x^4 + y) + Real.sqrt (8 * y^4 + z) + Real.sqrt (8 * z^4 + x) - 3) / (x + y + z)
  A ≤ 2 ∧ ∃ x y z, (0 < x ∧ x ≤ 1) ∧ (0 < y ∧ y ≤ 1) ∧ (0 < z ∧ z ≤ 1) ∧ A = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l730_73029


namespace NUMINAMATH_CALUDE_line_angle_inclination_l730_73094

/-- The angle of inclination of a line given its equation and a point it passes through -/
def angleOfInclination (a m : ℝ) (h1 : m ≠ 0) (h2 : a + m - 2*a = 0) : ℝ :=
  135

/-- Theorem: The angle of inclination of the line ax + my - 2a = 0 (m ≠ 0) passing through (1, 1) is 135° -/
theorem line_angle_inclination (a m : ℝ) (h1 : m ≠ 0) (h2 : a + m - 2*a = 0) :
  angleOfInclination a m h1 h2 = 135 := by
  sorry

end NUMINAMATH_CALUDE_line_angle_inclination_l730_73094


namespace NUMINAMATH_CALUDE_park_fencing_cost_l730_73037

/-- Proves that the fencing cost per meter is 50 paise for a rectangular park with given conditions -/
theorem park_fencing_cost
  (ratio : ℚ) -- Ratio of length to width
  (area : ℝ) -- Area of the park in square meters
  (total_cost : ℝ) -- Total cost of fencing
  (h_ratio : ratio = 3 / 2) -- The sides are in the ratio 3:2
  (h_area : area = 7350) -- The area is 7350 sq m
  (h_total_cost : total_cost = 175) -- The total cost of fencing is 175
  : ℝ :=
by
  -- Proof goes here
  sorry

#check park_fencing_cost

end NUMINAMATH_CALUDE_park_fencing_cost_l730_73037


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l730_73063

theorem quadratic_inequality_solution_set (a : ℝ) (h : a > 0) :
  let S := {x : ℝ | a * x^2 - (a + 1) * x + 1 < 0}
  (0 < a ∧ a < 1 → S = {x : ℝ | 1 < x ∧ x < 1/a}) ∧
  (a = 1 → S = ∅) ∧
  (a > 1 → S = {x : ℝ | 1/a < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l730_73063


namespace NUMINAMATH_CALUDE_bread_slice_cost_l730_73016

/-- Calculates the cost per slice of bread in cents -/
def cost_per_slice (num_loaves : ℕ) (slices_per_loaf : ℕ) (amount_paid : ℕ) (change : ℕ) : ℕ :=
  let total_cost := amount_paid - change
  let total_slices := num_loaves * slices_per_loaf
  (total_cost * 100) / total_slices

/-- Proves that the cost per slice is 40 cents given the problem conditions -/
theorem bread_slice_cost :
  cost_per_slice 3 20 40 16 = 40 := by
  sorry

#eval cost_per_slice 3 20 40 16

end NUMINAMATH_CALUDE_bread_slice_cost_l730_73016


namespace NUMINAMATH_CALUDE_cafeteria_shirts_l730_73049

theorem cafeteria_shirts (total : Nat) (vertical : Nat) 
  (h1 : total = 40)
  (h2 : vertical = 5)
  (h3 : ∃ (checkered : Nat), total = checkered + 4 * checkered + vertical) :
  ∃ (checkered : Nat), checkered = 7 ∧ 
    total = checkered + 4 * checkered + vertical := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_shirts_l730_73049


namespace NUMINAMATH_CALUDE_range_of_fraction_l730_73022

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∃ (z : ℝ), z = x / y ∧ 1/6 ≤ z ∧ z ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l730_73022


namespace NUMINAMATH_CALUDE_parallel_line_slope_l730_73000

theorem parallel_line_slope (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  let m := -a / b
  (∀ x y, a * x + b * y = c) → m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l730_73000


namespace NUMINAMATH_CALUDE_cookies_eaten_l730_73096

theorem cookies_eaten (initial : ℕ) (remaining : ℕ) (h1 : initial = 28) (h2 : remaining = 7) :
  initial - remaining = 21 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_l730_73096


namespace NUMINAMATH_CALUDE_stock_order_l730_73095

def initial_investment : ℝ := 100

def apple_year1 : ℝ := 1.50
def apple_year2 : ℝ := 0.75
def banana_year1 : ℝ := 0.50
def banana_year2 : ℝ := 2.00
def cherry_year1 : ℝ := 1.30
def cherry_year2 : ℝ := 1.10
def date_year1 : ℝ := 1.00
def date_year2 : ℝ := 0.80

def final_value (year1 : ℝ) (year2 : ℝ) : ℝ :=
  initial_investment * year1 * year2

theorem stock_order :
  let A := final_value apple_year1 apple_year2
  let B := final_value banana_year1 banana_year2
  let C := final_value cherry_year1 cherry_year2
  let D := final_value date_year1 date_year2
  D < B ∧ B < A ∧ A < C := by
  sorry

end NUMINAMATH_CALUDE_stock_order_l730_73095


namespace NUMINAMATH_CALUDE_number_2018_in_group_27_l730_73035

/-- The sum of even numbers up to the k-th group -/
def S (k : ℕ) : ℕ := (3 * k^2 - k) / 2

/-- The proposition that 2018 belongs to the 27th group -/
theorem number_2018_in_group_27 : 
  S 26 < 1009 ∧ 1009 ≤ S 27 := by sorry

end NUMINAMATH_CALUDE_number_2018_in_group_27_l730_73035


namespace NUMINAMATH_CALUDE_group_purchase_equation_l730_73044

/-- Represents a group purchase scenario where:
    - x is the number of people
    - p is the price of the item in coins
    - If each person contributes 8 coins, there's an excess of 3 coins
    - If each person contributes 7 coins, there's a shortage of 4 coins -/
structure GroupPurchase where
  x : ℕ  -- number of people
  p : ℕ  -- price of the item in coins
  excess_condition : 8 * x = p + 3
  shortage_condition : 7 * x + 4 = p

/-- Theorem stating that in a valid GroupPurchase scenario, 
    the number of people satisfies the equation 8x - 3 = 7x + 4 -/
theorem group_purchase_equation (gp : GroupPurchase) : 8 * gp.x - 3 = 7 * gp.x + 4 := by
  sorry


end NUMINAMATH_CALUDE_group_purchase_equation_l730_73044


namespace NUMINAMATH_CALUDE_abs_comparison_negative_numbers_l730_73050

theorem abs_comparison_negative_numbers (x y : ℝ) 
  (hx_neg : x < 0) (hy_neg : y < 0) (hxy : x < y) : 
  |x| > |y| := by
  sorry

end NUMINAMATH_CALUDE_abs_comparison_negative_numbers_l730_73050


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_41_l730_73026

/-- Represents the pricing model of a caterer -/
structure CatererPricing where
  basicFee : ℕ
  pricePerPerson : ℕ → ℕ

/-- The pricing model for the first caterer -/
def firstCaterer : CatererPricing :=
  { basicFee := 150,
    pricePerPerson := fun _ => 17 }

/-- The pricing model for the second caterer -/
def secondCaterer : CatererPricing :=
  { basicFee := 250,
    pricePerPerson := fun x => if x ≤ 40 then 15 else 13 }

/-- Calculate the total price for a caterer given the number of people -/
def totalPrice (c : CatererPricing) (people : ℕ) : ℕ :=
  c.basicFee + c.pricePerPerson people * people

/-- The theorem stating that 41 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_41 :
  (∀ n < 41, totalPrice firstCaterer n ≤ totalPrice secondCaterer n) ∧
  (totalPrice secondCaterer 41 < totalPrice firstCaterer 41) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_41_l730_73026


namespace NUMINAMATH_CALUDE_min_value_expression_l730_73089

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1 / x + 4 / y) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l730_73089


namespace NUMINAMATH_CALUDE_suraj_innings_l730_73015

/-- 
Proves that the number of innings Suraj played before the last one is 16,
given the conditions of the problem.
-/
theorem suraj_innings : 
  ∀ (n : ℕ) (A : ℚ),
  (A + 4 = 28) →                             -- New average after increase
  (n * A + 92 = (n + 1) * 28) →              -- Total runs equation
  (n = 16) := by
sorry

end NUMINAMATH_CALUDE_suraj_innings_l730_73015


namespace NUMINAMATH_CALUDE_expression_factorization_l730_73034

theorem expression_factorization (x : ℝ) : 
  (8 * x^4 + 34 * x^3 - 120 * x + 150) - (-2 * x^4 + 12 * x^3 - 5 * x + 10) = 
  5 * x * (2 * x^3 + (22/5) * x^2 - 23 * x + 28) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l730_73034


namespace NUMINAMATH_CALUDE_outfit_combinations_l730_73062

/-- The number of shirts -/
def num_shirts : ℕ := 4

/-- The number of pants -/
def num_pants : ℕ := 5

/-- The number of items (shirts or pants) that have a unique color -/
def num_unique_colors : ℕ := num_shirts + num_pants - 1

/-- The number of different outfits that can be created -/
def num_outfits : ℕ := num_shirts * num_pants - 1

theorem outfit_combinations : num_outfits = 19 := by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l730_73062


namespace NUMINAMATH_CALUDE_total_points_scored_l730_73055

theorem total_points_scored (team_a team_b team_c : ℕ) 
  (h1 : team_a = 2) 
  (h2 : team_b = 9) 
  (h3 : team_c = 4) : 
  team_a + team_b + team_c = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_points_scored_l730_73055


namespace NUMINAMATH_CALUDE_min_value_theorem_l730_73036

theorem min_value_theorem (x y : ℝ) 
  (h1 : x > -1) 
  (h2 : y > 0) 
  (h3 : x + y = 1) : 
  (∀ x' y' : ℝ, x' > -1 → y' > 0 → x' + y' = 1 → 
    1 / (x' + 1) + 4 / y' ≥ 1 / (x + 1) + 4 / y) ∧ 
  1 / (x + 1) + 4 / y = 9/2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l730_73036


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l730_73070

def q (x : ℚ) : ℚ := (17 * x^3 - 30 * x^2 + x + 12) / 6

theorem cubic_polynomial_satisfies_conditions :
  q (-1) = -6 ∧ q 2 = 5 ∧ q 0 = 2 ∧ q 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l730_73070


namespace NUMINAMATH_CALUDE_miquel_point_midpoint_l730_73077

-- Define the points
variable (A B C D O M T S : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define O as the intersection of diagonals
def is_diagonal_intersection (O A B C D : EuclideanPlane) : Prop := sorry

-- Define the circumcircles
def on_circumcircle (P Q R S : EuclideanPlane) : Prop := sorry

-- Define M as the intersection of circumcircles OAD and OBC
def is_circumcircle_intersection (M O A B C D : EuclideanPlane) : Prop := sorry

-- Define T and S on the line OM and on their respective circumcircles
def on_line_and_circumcircle (P Q R S T : EuclideanPlane) : Prop := sorry

-- Define the midpoint
def is_midpoint (M S T : EuclideanPlane) : Prop := sorry

-- The theorem
theorem miquel_point_midpoint 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_diagonal_intersection O A B C D)
  (h3 : is_circumcircle_intersection M O A B C D)
  (h4 : on_line_and_circumcircle O M A B T)
  (h5 : on_line_and_circumcircle O M C D S) :
  is_midpoint M S T := by sorry

end NUMINAMATH_CALUDE_miquel_point_midpoint_l730_73077


namespace NUMINAMATH_CALUDE_nursing_home_medicine_boxes_l730_73013

/-- The total number of boxes of medicine received by the nursing home -/
def total_boxes (vitamin_boxes supplement_boxes : ℕ) : ℕ :=
  vitamin_boxes + supplement_boxes

/-- Theorem stating that the nursing home received 760 boxes of medicine -/
theorem nursing_home_medicine_boxes : 
  total_boxes 472 288 = 760 := by
  sorry

end NUMINAMATH_CALUDE_nursing_home_medicine_boxes_l730_73013


namespace NUMINAMATH_CALUDE_p_iff_q_l730_73091

theorem p_iff_q (a b : ℝ) :
  (a > 2 ∧ b > 3) ↔ (a + b > 5 ∧ (a - 2) * (b - 3) > 0) := by
  sorry

end NUMINAMATH_CALUDE_p_iff_q_l730_73091


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l730_73038

/-- Given a geometric sequence with common ratio 2, prove that S_4 / a_2 = 15/2 --/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- Common ratio is 2
  (∀ n, S n = (a 1) * (1 - 2^n) / (1 - 2)) →  -- Sum formula for geometric sequence
  S 4 / a 2 = 15 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l730_73038


namespace NUMINAMATH_CALUDE_unique_solution_l730_73052

/-- Jessica's work hours as a function of t -/
def jessica_hours (t : ℤ) : ℤ := 3 * t - 10

/-- Jessica's hourly rate as a function of t -/
def jessica_rate (t : ℤ) : ℤ := 4 * t - 9

/-- Bob's work hours as a function of t -/
def bob_hours (t : ℤ) : ℤ := t + 12

/-- Bob's hourly rate as a function of t -/
def bob_rate (t : ℤ) : ℤ := 2 * t + 1

/-- Predicate to check if t satisfies the equation -/
def satisfies_equation (t : ℤ) : Prop :=
  jessica_hours t * jessica_rate t = bob_hours t * bob_rate t

theorem unique_solution :
  ∃! t : ℤ, t > 3 ∧ satisfies_equation t := by sorry

end NUMINAMATH_CALUDE_unique_solution_l730_73052


namespace NUMINAMATH_CALUDE_prom_couples_count_l730_73008

theorem prom_couples_count (total_students : ℕ) (solo_students : ℕ) (couples : ℕ) : 
  total_students = 123 → 
  solo_students = 3 → 
  couples = (total_students - solo_students) / 2 → 
  couples = 60 := by
  sorry

end NUMINAMATH_CALUDE_prom_couples_count_l730_73008


namespace NUMINAMATH_CALUDE_square_of_three_times_sqrt_two_l730_73030

theorem square_of_three_times_sqrt_two : (3 * Real.sqrt 2) ^ 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_of_three_times_sqrt_two_l730_73030


namespace NUMINAMATH_CALUDE_number_manipulation_l730_73011

theorem number_manipulation (x : ℝ) : (x - 5) / 7 = 7 → (x - 6) / 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_manipulation_l730_73011


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l730_73001

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  sum_property : a 2 + a 8 = 15
  product_property : a 3 * a 7 = 36

/-- The theorem stating the possible values of a_19 / a_13 -/
theorem geometric_sequence_ratio 
  (seq : GeometricSequence) : 
  seq.a 19 / seq.a 13 = 1/4 ∨ seq.a 19 / seq.a 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l730_73001


namespace NUMINAMATH_CALUDE_alf3_weight_l730_73098

/-- The molecular weight of a compound -/
def molecularWeight (alWeight fWeight : ℝ) : ℝ := alWeight + 3 * fWeight

/-- The total weight of a given number of moles of a compound -/
def totalWeight (molWeight : ℝ) (moles : ℝ) : ℝ := molWeight * moles

/-- Theorem stating the total weight of 10 moles of aluminum fluoride -/
theorem alf3_weight : 
  let alWeight : ℝ := 26.98
  let fWeight : ℝ := 19.00
  let moles : ℝ := 10
  totalWeight (molecularWeight alWeight fWeight) moles = 839.8 := by
sorry

end NUMINAMATH_CALUDE_alf3_weight_l730_73098


namespace NUMINAMATH_CALUDE_combined_height_of_tamara_and_kim_l730_73021

/-- Given Tamara's height is 3 times Kim's height less 4 inches and Tamara is 68 inches tall,
    prove that the combined height of Tamara and Kim is 92 inches. -/
theorem combined_height_of_tamara_and_kim (kim_height : ℕ) : 
  (3 * kim_height - 4 = 68) → (68 + kim_height = 92) := by
  sorry

end NUMINAMATH_CALUDE_combined_height_of_tamara_and_kim_l730_73021


namespace NUMINAMATH_CALUDE_same_solution_k_value_l730_73092

theorem same_solution_k_value : ∃ (k : ℝ), 
  (∀ (x : ℝ), (2 * x + 4 = 4 * (x - 2)) ↔ (-x + k = 2 * x - 1)) → k = 17 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_k_value_l730_73092


namespace NUMINAMATH_CALUDE_inequality_proof_l730_73027

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : 1/a + 1/b + 1/c = 1) :
  Real.sqrt (a + b*c) + Real.sqrt (b + c*a) + Real.sqrt (c + a*b) ≥ 
  Real.sqrt (a*b*c) + Real.sqrt a + Real.sqrt b + Real.sqrt c :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l730_73027


namespace NUMINAMATH_CALUDE_triangle_side_length_l730_73005

theorem triangle_side_length (A B C : ℝ × ℝ) :
  let AC := Real.sqrt 3
  let AB := 2
  let angle_B := 60 * Real.pi / 180
  let BC := Real.sqrt ((AC^2 + AB^2) - 2 * AC * AB * Real.cos angle_B)
  AC = Real.sqrt 3 ∧ AB = 2 ∧ angle_B = 60 * Real.pi / 180 →
  BC = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l730_73005


namespace NUMINAMATH_CALUDE_unique_common_root_existence_l730_73086

theorem unique_common_root_existence :
  ∃! m : ℝ, ∃! x : ℝ, 
    (x^2 + m*x + 2 = 0 ∧ x^2 + 2*x + m = 0) ∧
    (∀ y : ℝ, y^2 + m*y + 2 = 0 ∧ y^2 + 2*y + m = 0 → y = x) ∧
    m = -3 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_common_root_existence_l730_73086


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l730_73074

theorem unique_solution_sqrt_equation :
  ∃! x : ℝ, Real.sqrt (2 * x + 6) - Real.sqrt (2 * x - 2) = 2 :=
by
  -- The unique solution is x = 1.5
  use (3/2)
  constructor
  · -- Prove that x = 1.5 satisfies the equation
    sorry
  · -- Prove that any solution must equal 1.5
    sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l730_73074


namespace NUMINAMATH_CALUDE_star_equation_solution_l730_73079

def star (a b : ℝ) : ℝ := a^2 * b + 2 * b - a

theorem star_equation_solution :
  ∀ x : ℝ, star 7 x = 85 → x = 92 / 51 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l730_73079


namespace NUMINAMATH_CALUDE_i_times_one_minus_i_squared_eq_two_l730_73010

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem stating that i(1-i)² = 2 -/
theorem i_times_one_minus_i_squared_eq_two : i * (1 - i)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_i_times_one_minus_i_squared_eq_two_l730_73010


namespace NUMINAMATH_CALUDE_sequence_properties_l730_73073

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (b : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n + d

def c (a b : ℕ → ℝ) (n : ℕ) : ℝ := a n + b n

theorem sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ) (q d : ℝ)
  (h_geom : geometric_sequence a q)
  (h_arith : arithmetic_sequence b d)
  (h_q : q ≠ 1)
  (h_d : d ≠ 0) :
  (¬ arithmetic_sequence (c a b) ((c a b 2) - (c a b 1))) ∧
  (a 1 = 1 ∧ q = 2 → 
    ∃ f : ℝ → ℝ, f d = (c a b 2) ∧ 
    (∀ x : ℝ, x ≠ -1 ∧ x ≠ -2 ∧ x ≠ 0 → f x = x^2 + 3*x)) ∧
  (¬ geometric_sequence (c a b) ((c a b 2) / (c a b 1))) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l730_73073


namespace NUMINAMATH_CALUDE_line_through_points_l730_73017

/-- Given a line x = 3y + 5 passing through points (m, n) and (m + 2, n + q), prove that q = 2/3 -/
theorem line_through_points (m n : ℝ) : 
  (∃ q : ℝ, m = 3 * n + 5 ∧ m + 2 = 3 * (n + q) + 5) → 
  (∃ q : ℝ, q = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l730_73017


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l730_73002

/-- A quadratic equation with reciprocal roots whose sum is four times their product -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  roots_reciprocal : ∃ (r : ℝ), r ≠ 0 ∧ r + 1/r = -b/a
  sum_four_times_product : -b/a = 4 * (c/a)

/-- The coefficients of the quadratic equation satisfy a = c and b = -4a -/
theorem quadratic_equation_coefficients (eq : QuadraticEquation) : eq.a = eq.c ∧ eq.b = -4 * eq.a := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l730_73002


namespace NUMINAMATH_CALUDE_probability_two_teams_play_l730_73057

/-- The probability that two specific teams play each other in a single-elimination tournament -/
theorem probability_two_teams_play (n : ℕ) (h : n = 16) : 
  (2 : ℚ) / ((n : ℚ) * (n - 1)) = 1 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_two_teams_play_l730_73057


namespace NUMINAMATH_CALUDE_two_rotational_homotheties_l730_73060

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rotational homothety -/
structure RotationalHomothety where
  center : ℝ × ℝ
  angle : ℝ
  scale : ℝ

/-- Applies a rotational homothety to a circle -/
def applyRotationalHomothety (h : RotationalHomothety) (c : Circle) : Circle :=
  sorry

/-- Checks if two circles are equal -/
def circlesEqual (c1 c2 : Circle) : Prop :=
  sorry

/-- Main theorem -/
theorem two_rotational_homotheties 
  (S₁ S₂ : Circle) 
  (h : S₁.center ≠ S₂.center) : 
  ∃! (pair : (RotationalHomothety × RotationalHomothety)),
    (circlesEqual (applyRotationalHomothety pair.1 S₁) S₂) ∧
    (circlesEqual (applyRotationalHomothety pair.2 S₁) S₂) ∧
    (pair.1.angle = π/2) ∧ (pair.2.angle = π/2) ∧
    (pair.1.center ≠ pair.2.center) :=
  sorry

end NUMINAMATH_CALUDE_two_rotational_homotheties_l730_73060


namespace NUMINAMATH_CALUDE_convex_quadrilateral_triangle_angles_l730_73006

theorem convex_quadrilateral_triangle_angles 
  (α β γ θ₁ θ₂ θ₃ θ₄ : Real) : 
  (α + β + γ = Real.pi) →  -- Sum of angles in a triangle is π radians (180°)
  (θ₁ + θ₂ + θ₃ + θ₄ = 2 * Real.pi) →  -- Sum of angles in a quadrilateral is 2π radians (360°)
  (θ₁ = α) → (θ₂ = β) → (θ₃ = γ) →  -- Three angles of quadrilateral equal to triangle angles
  ¬(θ₁ < Real.pi ∧ θ₂ < Real.pi ∧ θ₃ < Real.pi ∧ θ₄ < Real.pi)  -- Negation of convexity condition
  := by sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_triangle_angles_l730_73006


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l730_73045

theorem polynomial_division_theorem (x : ℝ) :
  (x - 3) * (x^5 - 2*x^4 + 18*x^3 + 42*x^2 + 135*x + 387) + 1221 =
  x^6 - 5*x^5 + 24*x^4 - 12*x^3 + 9*x^2 - 18*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l730_73045


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_linear_equation_solutions_l730_73069

theorem quadratic_equation_solutions (x : ℝ) :
  (x^2 + 5*x - 1 = 0) ↔ (x = (-5 + Real.sqrt 29) / 2 ∨ x = (-5 - Real.sqrt 29) / 2) :=
sorry

theorem linear_equation_solutions (x : ℝ) :
  (7*x*(5*x + 2) = 6*(5*x + 2)) ↔ (x = -2/5 ∨ x = 6/7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_linear_equation_solutions_l730_73069


namespace NUMINAMATH_CALUDE_quadratic_radicals_combination_l730_73080

theorem quadratic_radicals_combination (x : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ x + 1 = k * (2 * x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radicals_combination_l730_73080


namespace NUMINAMATH_CALUDE_max_value_abc_max_value_abc_attained_l730_73046

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  a^3 * b^2 * c^2 ≤ 432 / 7^7 := by
  sorry

theorem max_value_abc_attained (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 1 ∧ a₀^3 * b₀^2 * c₀^2 = 432 / 7^7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_max_value_abc_attained_l730_73046


namespace NUMINAMATH_CALUDE_f_inequality_iff_a_condition_l730_73084

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - (a + 1) * Real.log x

-- State the theorem
theorem f_inequality_iff_a_condition (a : ℝ) :
  (∀ x > 0, f a x ≤ x) ↔ a ≥ -1 / (Real.exp 1 - 1) := by sorry

end NUMINAMATH_CALUDE_f_inequality_iff_a_condition_l730_73084


namespace NUMINAMATH_CALUDE_solution_of_system_l730_73065

theorem solution_of_system (x₁ x₂ x₃ : ℝ) : 
  (2 * x₁^2 / (1 + x₁^2) = x₂) ∧ 
  (2 * x₂^2 / (1 + x₂^2) = x₃) ∧ 
  (2 * x₃^2 / (1 + x₃^2) = x₁) → 
  ((x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0) ∨ (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1)) := by
  sorry

end NUMINAMATH_CALUDE_solution_of_system_l730_73065


namespace NUMINAMATH_CALUDE_set_union_equality_l730_73031

-- Define the sets M and N
def M : Set ℝ := {x | x^2 < 4*x}
def N : Set ℝ := {x | |x - 1| ≥ 3}

-- Define the union set
def unionSet : Set ℝ := {x | x ≤ -2 ∨ x > 0}

-- Theorem statement
theorem set_union_equality : M ∪ N = unionSet := by
  sorry

end NUMINAMATH_CALUDE_set_union_equality_l730_73031


namespace NUMINAMATH_CALUDE_intersection_M_N_l730_73018

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 2 < 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l730_73018


namespace NUMINAMATH_CALUDE_hyperbola_focus_a_value_l730_73071

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2 / 9 - y^2 / a = 1

-- Define the right focus
def right_focus (x y : ℝ) : Prop := x = Real.sqrt 13 ∧ y = 0

-- Theorem statement
theorem hyperbola_focus_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, hyperbola x y a → right_focus x y) → a = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_a_value_l730_73071


namespace NUMINAMATH_CALUDE_pascal_triangle_elements_l730_73048

/-- The number of elements in a single row of Pascal's Triangle -/
def elementsInRow (n : ℕ) : ℕ := n + 1

/-- The sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of elements in the first n rows of Pascal's Triangle -/
def totalElementsPascal (n : ℕ) : ℕ := triangularNumber n

theorem pascal_triangle_elements :
  totalElementsPascal 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_elements_l730_73048


namespace NUMINAMATH_CALUDE_cube_sum_equation_l730_73053

/-- Given real numbers p, q, and r satisfying certain conditions, 
    their cubes sum to 181. -/
theorem cube_sum_equation (p q r : ℝ) 
  (h1 : p + q + r = 7)
  (h2 : p * q + p * r + q * r = 10)
  (h3 : p * q * r = -20) :
  p^3 + q^3 + r^3 = 181 := by
  sorry


end NUMINAMATH_CALUDE_cube_sum_equation_l730_73053


namespace NUMINAMATH_CALUDE_tire_circumference_l730_73059

/-- The circumference of a tire given its rotation speed and the car's velocity -/
theorem tire_circumference (rotations_per_minute : ℝ) (car_speed_kmh : ℝ) : 
  rotations_per_minute = 400 →
  car_speed_kmh = 96 →
  (car_speed_kmh * 1000 / 60) / rotations_per_minute = 4 := by
sorry

end NUMINAMATH_CALUDE_tire_circumference_l730_73059


namespace NUMINAMATH_CALUDE_unique_p_for_equal_roots_l730_73066

/-- The quadratic equation x^2 - px + p^2 = 0 has equal roots for exactly one real value of p -/
theorem unique_p_for_equal_roots :
  ∃! p : ℝ, ∀ x : ℝ, x^2 - p*x + p^2 = 0 → (∃! y : ℝ, y^2 - p*y + p^2 = 0) := by sorry

end NUMINAMATH_CALUDE_unique_p_for_equal_roots_l730_73066


namespace NUMINAMATH_CALUDE_solve_equation_l730_73033

theorem solve_equation (C : ℝ) (h : 5 * C - 6 = 34) : C = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l730_73033


namespace NUMINAMATH_CALUDE_rhombus_area_l730_73032

/-- The area of a rhombus with side length √125 and diagonal difference 8 is 60.5 -/
theorem rhombus_area (side : ℝ) (diag_diff : ℝ) (area : ℝ) : 
  side = Real.sqrt 125 →
  diag_diff = 8 →
  area = (side^2 * Real.sqrt (4 - (diag_diff / side)^2)) / 2 →
  area = 60.5 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l730_73032


namespace NUMINAMATH_CALUDE_simplify_expression_l730_73047

theorem simplify_expression (w : ℝ) : 3*w + 4 - 2*w - 5 + 6*w + 7 - 3*w - 9 = 4*w - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l730_73047


namespace NUMINAMATH_CALUDE_angle_between_vectors_l730_73068

theorem angle_between_vectors (a b : ℝ × ℝ) : 
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  a = (1, 2) ∧ b = (3, 1) → angle = π / 4 := by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l730_73068


namespace NUMINAMATH_CALUDE_angle_B_not_right_angle_sin_C_over_sin_A_range_l730_73020

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  2 * Real.sin t.C * Real.sin (t.B - t.A) = 2 * Real.sin t.A * Real.sin t.C - Real.sin t.B ^ 2

-- Theorem 1: Angle B cannot be a right angle
theorem angle_B_not_right_angle (t : Triangle) (h : satisfies_condition t) : 
  t.B ≠ π / 2 := by sorry

-- Theorem 2: Range of sin(C)/sin(A) for acute triangles
theorem sin_C_over_sin_A_range (t : Triangle) (h1 : satisfies_condition t) 
  (h2 : t.A < π / 2 ∧ t.B < π / 2 ∧ t.C < π / 2) : 
  1 / 3 < Real.sin t.C / Real.sin t.A ∧ Real.sin t.C / Real.sin t.A < 5 / 3 := by sorry

end NUMINAMATH_CALUDE_angle_B_not_right_angle_sin_C_over_sin_A_range_l730_73020


namespace NUMINAMATH_CALUDE_no_solution_equation_l730_73028

theorem no_solution_equation :
  ¬ ∃ x : ℝ, x - 9 / (x - 5) = 5 - 9 / (x - 5) :=
sorry

end NUMINAMATH_CALUDE_no_solution_equation_l730_73028


namespace NUMINAMATH_CALUDE_y_intersection_is_six_l730_73078

/-- The quadratic function f(x) = -2(x-1)(x+3) -/
def f (x : ℝ) : ℝ := -2 * (x - 1) * (x + 3)

/-- The y-coordinate of the intersection point with the y-axis is 6 -/
theorem y_intersection_is_six : f 0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_y_intersection_is_six_l730_73078


namespace NUMINAMATH_CALUDE_decreasing_quadratic_condition_l730_73024

-- Define the function f(x) = x^2 + mx + 1
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

-- Define the property of f being decreasing on an interval
def isDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Theorem statement
theorem decreasing_quadratic_condition (m : ℝ) :
  isDecreasingOn (f m) 0 5 → m ≤ -10 :=
sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_condition_l730_73024


namespace NUMINAMATH_CALUDE_range_of_a_l730_73003

theorem range_of_a (x a : ℝ) : 
  (∀ x, x - a ≥ 1 → x ≥ 1) ∧ 
  (1 - a ≥ 1) ∧ 
  ¬(-1 - a ≥ 1) → 
  -2 < a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l730_73003


namespace NUMINAMATH_CALUDE_dog_reachable_area_l730_73090

/-- The area outside a regular pentagon that a tethered dog can reach -/
theorem dog_reachable_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 → rope_length = 3 → 
  ∃ (area : ℝ), area = 7.6 * Real.pi ∧ 
  area = (rope_length^2 * Real.pi * (288 / 360)) + 
         (2 * (side_length^2 * Real.pi * (72 / 360))) :=
sorry

end NUMINAMATH_CALUDE_dog_reachable_area_l730_73090


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l730_73088

theorem quadratic_one_solution (m : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + m = 0) → m = 49 / 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l730_73088


namespace NUMINAMATH_CALUDE_expression_equals_six_l730_73042

theorem expression_equals_six :
  (Real.sqrt 27 + Real.sqrt 48) / Real.sqrt 3 - (Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 3 + Real.sqrt 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_six_l730_73042


namespace NUMINAMATH_CALUDE_cubic_root_product_l730_73025

theorem cubic_root_product (x : ℝ) : 
  (∃ p q r : ℝ, x^3 - 15*x^2 + 60*x - 36 = (x - p) * (x - q) * (x - r)) → 
  (∃ p q r : ℝ, x^3 - 15*x^2 + 60*x - 36 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 36) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_product_l730_73025


namespace NUMINAMATH_CALUDE_abcg_over_defh_value_l730_73041

theorem abcg_over_defh_value (a b c d e f g h : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6)
  (h6 : f / g = 5 / 2)
  (h7 : g / h = 3 / 4)
  (h8 : b ≠ 0)
  (h9 : c ≠ 0)
  (h10 : d ≠ 0)
  (h11 : e ≠ 0)
  (h12 : f ≠ 0)
  (h13 : g ≠ 0)
  (h14 : h ≠ 0) :
  a * b * c * g / (d * e * f * h) = 5 / 48 := by
  sorry

end NUMINAMATH_CALUDE_abcg_over_defh_value_l730_73041


namespace NUMINAMATH_CALUDE_trees_planted_per_cut_l730_73004

/-- Proves that the number of new trees planted for each tree cut is 5 --/
theorem trees_planted_per_cut (initial_trees : ℕ) (cut_percentage : ℚ) (final_trees : ℕ) : 
  initial_trees = 400 → 
  cut_percentage = 1/5 →
  final_trees = 720 →
  (final_trees - (initial_trees - initial_trees * cut_percentage)) / (initial_trees * cut_percentage) = 5 := by
  sorry

end NUMINAMATH_CALUDE_trees_planted_per_cut_l730_73004


namespace NUMINAMATH_CALUDE_parabola_perpendicular_chords_theorem_l730_73012

/-- A parabola with vertex at the origin and focus on the positive x-axis -/
structure Parabola where
  p : ℝ
  equation : ℝ × ℝ → Prop := fun (x, y) ↦ y^2 = 2 * p * x

/-- A line passing through two points -/
def Line (A B : ℝ × ℝ) : ℝ × ℝ → Prop :=
  fun P ↦ (P.2 - A.2) * (B.1 - A.1) = (P.1 - A.1) * (B.2 - A.2)

/-- Two lines are perpendicular -/
def Perpendicular (L₁ L₂ : (ℝ × ℝ → Prop)) : Prop :=
  ∃ A B C D, L₁ A ∧ L₁ B ∧ L₂ C ∧ L₂ D ∧
    (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

/-- The projection of a point onto a line -/
def Projection (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  fun H ↦ L H ∧ Perpendicular (Line P H) L

theorem parabola_perpendicular_chords_theorem (Γ : Parabola) :
  ∀ A B, Γ.equation A ∧ Γ.equation B ∧ 
         Perpendicular (Line (0, 0) A) (Line (0, 0) B) →
  (∃ M₀, M₀ = (2 * Γ.p, 0) ∧ Line A B M₀) ∧
  (∀ H, Projection (0, 0) (Line A B) H → 
        H.1^2 + H.2^2 - 2 * Γ.p * H.1 = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_chords_theorem_l730_73012


namespace NUMINAMATH_CALUDE_quadratic_circle_theorem_l730_73085

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + b

-- Define the condition that f intersects the axes at three points
def intersects_axes_at_three_points (b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f b x₁ = 0 ∧ f b x₂ = 0 ∧ f b 0 ≠ 0

-- Define the equation of the circle
def circle_equation (b : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + 2*x - (b + 1)*y + b

-- Main theorem
theorem quadratic_circle_theorem (b : ℝ) 
  (h : intersects_axes_at_three_points b) :
  (b < 1 ∧ b ≠ 0) ∧
  (∀ x y : ℝ, circle_equation b x y = 0 ↔ 
    (x = 0 ∧ f b y = 0) ∨ (y = 0 ∧ f b x = 0) ∨ (x = 0 ∧ y = b)) ∧
  (circle_equation b 0 1 = 0 ∧ circle_equation b (-2) 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_circle_theorem_l730_73085


namespace NUMINAMATH_CALUDE_stratified_sampling_group_b_l730_73043

-- Define the total number of cities and the number in Group B
def total_cities : ℕ := 48
def group_b_cities : ℕ := 18

-- Define the total sample size
def sample_size : ℕ := 16

-- Define the function to calculate the number of cities to sample from Group B
def cities_to_sample_from_b : ℕ := 
  (group_b_cities * sample_size) / total_cities

-- Theorem statement
theorem stratified_sampling_group_b :
  cities_to_sample_from_b = 6 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_b_l730_73043


namespace NUMINAMATH_CALUDE_rectangular_hall_dimensions_l730_73081

theorem rectangular_hall_dimensions (length width : ℝ) : 
  width = length / 2 → 
  length * width = 450 → 
  length - width = 15 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimensions_l730_73081


namespace NUMINAMATH_CALUDE_decimal_expansion_three_eighths_no_repeat_l730_73075

/-- The length of the smallest repeating block in the decimal expansion of 3/8 is 0. -/
theorem decimal_expansion_three_eighths_no_repeat : 
  (∃ n : ℕ, ∃ k : ℕ, (3 : ℚ) / 8 = (n : ℚ) / 10^k ∧ k > 0) := by sorry

end NUMINAMATH_CALUDE_decimal_expansion_three_eighths_no_repeat_l730_73075


namespace NUMINAMATH_CALUDE_train_speed_l730_73099

/-- The speed of a train given its length, time to cross a man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) :
  train_length = 300 →
  crossing_time = 17.998560115190784 →
  man_speed = 3 →
  ∃ (train_speed : ℝ), abs (train_speed - 63.00468) < 0.00001 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l730_73099


namespace NUMINAMATH_CALUDE_missing_number_equation_l730_73056

theorem missing_number_equation : ∃ x : ℤ, 1234562 - 12 * 3 * x = 1234490 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l730_73056


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l730_73093

theorem quadratic_inequality_solution_set 
  (a b c m n : ℝ) 
  (h1 : a ≠ 0)
  (h2 : m > 0)
  (h3 : Set.Ioo m n = {x | a * x^2 + b * x + c > 0}) :
  {x | c * x^2 + b * x + a < 0} = Set.Iic (1/n) ∪ Set.Ioi (1/m) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l730_73093


namespace NUMINAMATH_CALUDE_integers_abs_le_two_l730_73040

theorem integers_abs_le_two : 
  {x : ℤ | |x| ≤ 2} = {-2, -1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_integers_abs_le_two_l730_73040


namespace NUMINAMATH_CALUDE_total_football_games_l730_73019

theorem total_football_games (games_attended : ℕ) (games_missed : ℕ) : 
  games_attended = 3 → games_missed = 4 → games_attended + games_missed = 7 :=
by
  sorry

#check total_football_games

end NUMINAMATH_CALUDE_total_football_games_l730_73019


namespace NUMINAMATH_CALUDE_cars_clearing_time_l730_73058

/-- Calculates the time for two cars to be clear of each other from the moment they meet -/
theorem cars_clearing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 120)
  (h2 : length2 = 280)
  (h3 : speed1 = 42)
  (h4 : speed2 = 30) : 
  (length1 + length2) / ((speed1 + speed2) * (1000 / 3600)) = 20 := by
  sorry

#check cars_clearing_time

end NUMINAMATH_CALUDE_cars_clearing_time_l730_73058


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l730_73054

theorem smallest_factorization_coefficient (b : ℕ) : 
  (∃ r s : ℤ, ∀ x : ℤ, x^2 + b*x + 1800 = (x + r) * (x + s)) →
  b ≥ 85 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l730_73054


namespace NUMINAMATH_CALUDE_sum_of_angles_l730_73061

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := z^5 = -32 * i

-- Define the form of solutions
def solution_form (z : ℂ) (s : ℝ) (α : ℝ) : Prop :=
  z = s * (Complex.cos α + i * Complex.sin α)

-- Define the conditions on s and α
def valid_solution (s : ℝ) (α : ℝ) : Prop :=
  s > 0 ∧ 0 ≤ α ∧ α < 2 * Real.pi

-- Theorem statement
theorem sum_of_angles (z₁ z₂ z₃ z₄ z₅ : ℂ) (s₁ s₂ s₃ s₄ s₅ α₁ α₂ α₃ α₄ α₅ : ℝ) :
  equation z₁ ∧ equation z₂ ∧ equation z₃ ∧ equation z₄ ∧ equation z₅ ∧
  solution_form z₁ s₁ α₁ ∧ solution_form z₂ s₂ α₂ ∧ solution_form z₃ s₃ α₃ ∧
  solution_form z₄ s₄ α₄ ∧ solution_form z₅ s₅ α₅ ∧
  valid_solution s₁ α₁ ∧ valid_solution s₂ α₂ ∧ valid_solution s₃ α₃ ∧
  valid_solution s₄ α₄ ∧ valid_solution s₅ α₅ →
  α₁ + α₂ + α₃ + α₄ + α₅ = 5.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_l730_73061


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l730_73097

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 1)^2 + 4 - a

-- State the theorem
theorem quadratic_function_inequality (a x₁ x₂ : ℝ) 
  (ha : 0 < a ∧ a < 3) 
  (hx : x₁ > x₂) 
  (hsum : x₁ + x₂ = 1 - a) : 
  f a x₁ > f a x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l730_73097


namespace NUMINAMATH_CALUDE_min_box_height_is_ten_l730_73067

/-- Represents the side length of the square base of the box -/
def base_side : ℝ → ℝ := λ x => x

/-- Represents the height of the box -/
def box_height : ℝ → ℝ := λ x => x + 5

/-- Calculates the surface area of the box -/
def surface_area : ℝ → ℝ := λ x => 2 * x^2 + 4 * x * (x + 5)

/-- Theorem: The minimum height of the box satisfying the given conditions is 10 units -/
theorem min_box_height_is_ten :
  ∃ (x : ℝ), x > 0 ∧ 
             surface_area x ≥ 130 ∧ 
             box_height x = 10 ∧
             ∀ (y : ℝ), y > 0 ∧ surface_area y ≥ 130 → box_height y ≥ box_height x :=
by sorry


end NUMINAMATH_CALUDE_min_box_height_is_ten_l730_73067


namespace NUMINAMATH_CALUDE_logarithm_sum_equals_three_logarithm_base_25_144_l730_73051

-- Part 1
theorem logarithm_sum_equals_three :
  (Real.log 2) ^ 2 + (Real.log 20 + 2) * Real.log 5 + Real.log 4 = 3 := by sorry

-- Part 2
theorem logarithm_base_25_144 (a b : ℝ) (h1 : Real.log 3 / Real.log 5 = a) (h2 : Real.log 4 / Real.log 5 = b) :
  Real.log 144 / Real.log 25 = a + b := by sorry

end NUMINAMATH_CALUDE_logarithm_sum_equals_three_logarithm_base_25_144_l730_73051


namespace NUMINAMATH_CALUDE_cos_negative_480_deg_l730_73009

theorem cos_negative_480_deg : Real.cos (-(480 * π / 180)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_480_deg_l730_73009


namespace NUMINAMATH_CALUDE_valerie_skipping_rate_l730_73039

/-- Roberto's skipping rate in skips per hour -/
def roberto_rate : ℕ := 4200

/-- Total skips for Roberto and Valerie in 15 minutes -/
def total_skips : ℕ := 2250

/-- Duration of skipping in minutes -/
def duration : ℕ := 15

/-- Valerie's skipping rate in skips per minute -/
def valerie_rate : ℕ := 80

theorem valerie_skipping_rate :
  (roberto_rate * duration / 60 + valerie_rate * duration = total_skips) ∧
  (valerie_rate = (total_skips - roberto_rate * duration / 60) / duration) :=
sorry

end NUMINAMATH_CALUDE_valerie_skipping_rate_l730_73039


namespace NUMINAMATH_CALUDE_max_value_inequality_l730_73023

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a^2 + b^2 + 2*c^2 = 1) : 
  Real.sqrt 2 * a * b + 2 * b * c + 7 * a * c ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l730_73023


namespace NUMINAMATH_CALUDE_green_percentage_approx_l730_73087

/-- Represents the count of people preferring each color --/
structure ColorPreferences where
  red : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ
  purple : ℕ
  orange : ℕ

/-- Calculates the percentage of people who preferred green --/
def greenPercentage (prefs : ColorPreferences) : ℚ :=
  (prefs.green : ℚ) / (prefs.red + prefs.blue + prefs.green + prefs.yellow + prefs.purple + prefs.orange) * 100

/-- Theorem stating that the percentage of people who preferred green is approximately 16.67% --/
theorem green_percentage_approx (prefs : ColorPreferences)
  (h1 : prefs.red = 70)
  (h2 : prefs.blue = 80)
  (h3 : prefs.green = 50)
  (h4 : prefs.yellow = 40)
  (h5 : prefs.purple = 30)
  (h6 : prefs.orange = 30) :
  ∃ ε > 0, |greenPercentage prefs - 50/3| < ε ∧ ε < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_green_percentage_approx_l730_73087


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l730_73014

/-- Calculates the total amount to pay after applying discounts to two bills -/
def total_amount_after_discount (bill1 bill2 discount1 discount2 : ℚ) : ℚ :=
  (bill1 * (1 - discount1 / 100)) + (bill2 * (1 - discount2 / 100))

/-- Theorem stating that the total amount Bob and Kate pay after discounts is $53 -/
theorem restaurant_bill_theorem :
  total_amount_after_discount 30 25 5 2 = 53 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l730_73014


namespace NUMINAMATH_CALUDE_max_distance_on_C_l730_73082

noncomputable section

open Real

-- Define the curve in polar coordinates
def C (θ : ℝ) : ℝ := 4 * sin θ

-- Define a point on the curve
def point_on_C (θ : ℝ) : ℝ × ℝ := (C θ * cos θ, C θ * sin θ)

-- Define the distance between two points on the curve
def distance_on_C (θ₁ θ₂ : ℝ) : ℝ :=
  let (x₁, y₁) := point_on_C θ₁
  let (x₂, y₂) := point_on_C θ₂
  sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem max_distance_on_C :
  ∃ (M : ℝ), M = 4 ∧ ∀ (θ₁ θ₂ : ℝ), distance_on_C θ₁ θ₂ ≤ M :=
sorry

end

end NUMINAMATH_CALUDE_max_distance_on_C_l730_73082


namespace NUMINAMATH_CALUDE_final_price_after_discounts_l730_73072

/-- Calculates the final price of an item after two successive discounts --/
theorem final_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 200 ∧ discount1 = 0.4 ∧ discount2 = 0.25 →
  original_price * (1 - discount1) * (1 - discount2) = 90 := by
  sorry

#check final_price_after_discounts

end NUMINAMATH_CALUDE_final_price_after_discounts_l730_73072


namespace NUMINAMATH_CALUDE_problem_1_l730_73083

theorem problem_1 : (1) - 8 + 12 - 16 - 23 = -35 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l730_73083


namespace NUMINAMATH_CALUDE_hypotenuse_length_l730_73007

/-- Given a right-angled triangle with sides a, b, and c (hypotenuse),
    where the sum of squares of all sides is 2500,
    prove that the length of the hypotenuse is 25√2. -/
theorem hypotenuse_length (a b c : ℝ) 
  (right_angle : a^2 + b^2 = c^2)  -- right-angled triangle condition
  (sum_squares : a^2 + b^2 + c^2 = 2500)  -- sum of squares condition
  : c = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l730_73007


namespace NUMINAMATH_CALUDE_units_digit_of_sum_cubes_l730_73076

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sum_cubes : units_digit (24^3 + 42^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_cubes_l730_73076
