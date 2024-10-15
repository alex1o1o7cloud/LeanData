import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_width_l2461_246148

/-- Given a rectangle with perimeter 150 cm and length 15 cm greater than width, prove the width is 30 cm. -/
theorem rectangle_width (w l : ℝ) (h1 : l = w + 15) (h2 : 2 * l + 2 * w = 150) : w = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l2461_246148


namespace NUMINAMATH_CALUDE_tshirt_count_l2461_246188

/-- The price of a pant in rupees -/
def pant_price : ℝ := sorry

/-- The price of a t-shirt in rupees -/
def tshirt_price : ℝ := sorry

/-- The total cost of 3 pants and 6 t-shirts in rupees -/
def total_cost_1 : ℝ := 750

/-- The total cost of 1 pant and 12 t-shirts in rupees -/
def total_cost_2 : ℝ := 750

/-- The amount to be spent on t-shirts in rupees -/
def tshirt_budget : ℝ := 400

theorem tshirt_count : 
  3 * pant_price + 6 * tshirt_price = total_cost_1 →
  pant_price + 12 * tshirt_price = total_cost_2 →
  (tshirt_budget / tshirt_price : ℝ) = 8 := by
sorry

end NUMINAMATH_CALUDE_tshirt_count_l2461_246188


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2461_246166

/-- The area of a square inscribed in the ellipse x²/4 + y²/8 = 1, 
    with its sides parallel to the coordinate axes, is 32/3 -/
theorem inscribed_square_area (x y : ℝ) :
  (∃ s : ℝ, s > 0 ∧ 
    (x^2 / 4 + y^2 / 8 = 1) ∧ 
    (x = s ∨ x = -s) ∧ 
    (y = s ∨ y = -s)) →
  (4 * s^2 = 32 / 3) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2461_246166


namespace NUMINAMATH_CALUDE_a_3_eq_35_l2461_246122

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℕ := 5 * n ^ 2 + 10 * n

/-- The n-th term of the sequence -/
def a (n : ℕ+) : ℤ := S n - S (n - 1)

/-- Theorem: The third term of the sequence is 35 -/
theorem a_3_eq_35 : a 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_a_3_eq_35_l2461_246122


namespace NUMINAMATH_CALUDE_sunflower_height_in_meters_l2461_246198

-- Define constants
def sister_height_feet : ℝ := 4.15
def sister_additional_height_cm : ℝ := 37
def sunflower_height_difference_inches : ℝ := 63

-- Define conversion factors
def inches_per_foot : ℝ := 12
def cm_per_inch : ℝ := 2.54
def cm_per_meter : ℝ := 100

-- Theorem statement
theorem sunflower_height_in_meters :
  let sister_height_cm := sister_height_feet * inches_per_foot * cm_per_inch + sister_additional_height_cm
  let sunflower_height_cm := sister_height_cm + sunflower_height_difference_inches * cm_per_inch
  sunflower_height_cm / cm_per_meter = 3.23512 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_height_in_meters_l2461_246198


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l2461_246104

/-- The difference between the larger and smaller x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference : ∃ (a c : ℝ),
  (∀ x : ℝ, 3 * x^2 - 6 * x + 6 = -x^2 - 4 * x + 6 → (x = a ∨ x = c)) ∧
  c ≥ a ∧
  c - a = (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l2461_246104


namespace NUMINAMATH_CALUDE_janes_flower_bed_area_l2461_246194

/-- A rectangular flower bed with fence posts -/
structure FlowerBed where
  total_posts : ℕ
  post_spacing : ℝ
  long_side_post_ratio : ℕ

/-- Calculate the area of a flower bed given its specifications -/
def flowerBedArea (fb : FlowerBed) : ℝ :=
  let short_side_posts := (fb.total_posts + 4) / (2 * (fb.long_side_post_ratio + 1))
  let long_side_posts := short_side_posts * fb.long_side_post_ratio
  let short_side_length := (short_side_posts - 1) * fb.post_spacing
  let long_side_length := (long_side_posts - 1) * fb.post_spacing
  short_side_length * long_side_length

/-- Theorem: The area of Jane's flower bed is 144 square feet -/
theorem janes_flower_bed_area :
  let fb : FlowerBed := {
    total_posts := 24,
    post_spacing := 3,
    long_side_post_ratio := 3
  }
  flowerBedArea fb = 144 := by sorry

end NUMINAMATH_CALUDE_janes_flower_bed_area_l2461_246194


namespace NUMINAMATH_CALUDE_problem_solution_l2461_246118

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (h1 : x/2 = y^2) (h2 : x/4 = 4*y) : x = 128 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2461_246118


namespace NUMINAMATH_CALUDE_prob_at_least_one_six_given_different_outcomes_prob_at_least_one_six_is_one_third_l2461_246128

/-- The probability of rolling at least one 6 given two fair dice with different outcomes -/
theorem prob_at_least_one_six_given_different_outcomes : ℝ :=
let total_outcomes := 30  -- 6 * 5, as outcomes are different
let favorable_outcomes := 10  -- 5 (first die is 6) + 5 (second die is 6)
favorable_outcomes / total_outcomes

/-- Proof that the probability is 1/3 -/
theorem prob_at_least_one_six_is_one_third :
  prob_at_least_one_six_given_different_outcomes = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_six_given_different_outcomes_prob_at_least_one_six_is_one_third_l2461_246128


namespace NUMINAMATH_CALUDE_small_circle_radius_l2461_246105

/-- Given a configuration of circles where:
    - There is one large circle with radius 10 meters
    - There are six congruent smaller circles
    - The smaller circles are aligned in a straight line
    - The smaller circles touch each other and the perimeter of the larger circle
    This theorem proves that the radius of each smaller circle is 5/3 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) (h1 : R = 10) (h2 : 6 * (2 * r) = 2 * R) :
  r = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_small_circle_radius_l2461_246105


namespace NUMINAMATH_CALUDE_square_eq_four_implies_x_values_l2461_246133

theorem square_eq_four_implies_x_values (x : ℝ) :
  (x - 1)^2 = 4 → x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_eq_four_implies_x_values_l2461_246133


namespace NUMINAMATH_CALUDE_correct_statements_l2461_246127

-- Define the statements
inductive Statement
| Synthesis1
| Synthesis2
| Analysis1
| Analysis2
| Contradiction

-- Define a function to check if a statement is correct
def is_correct (s : Statement) : Prop :=
  match s with
  | Statement.Synthesis1 => True  -- Synthesis is a method of cause and effect
  | Statement.Synthesis2 => True  -- Synthesis is a forward reasoning method
  | Statement.Analysis1 => True   -- Analysis is a method of seeking cause from effect
  | Statement.Analysis2 => False  -- Analysis is NOT an indirect proof method
  | Statement.Contradiction => False  -- Contradiction is NOT a backward reasoning method

-- Theorem to prove
theorem correct_statements :
  (is_correct Statement.Synthesis1) ∧
  (is_correct Statement.Synthesis2) ∧
  (is_correct Statement.Analysis1) ∧
  ¬(is_correct Statement.Analysis2) ∧
  ¬(is_correct Statement.Contradiction) :=
by sorry

#check correct_statements

end NUMINAMATH_CALUDE_correct_statements_l2461_246127


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2461_246183

/-- Proves that the cost price of an article is 1200, given that it was sold at a 40% profit for 1680. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
    (h1 : selling_price = 1680)
    (h2 : profit_percentage = 40) : 
  selling_price / (1 + profit_percentage / 100) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2461_246183


namespace NUMINAMATH_CALUDE_chord_equation_l2461_246124

/-- Given a circle with equation x² + y² = 9 and a chord PQ with midpoint (1,2),
    the equation of line PQ is x + 2y - 5 = 0 -/
theorem chord_equation (P Q : ℝ × ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 9 → (x - 1)^2 + (y - 2)^2 = ((P.1 - 1)^2 + (P.2 - 2)^2)) →
  ((P.1 + Q.1) / 2 = 1 ∧ (P.2 + Q.2) / 2 = 2) →
  ∃ (k : ℝ), ∀ (x y : ℝ), (x = P.1 ∧ y = P.2) ∨ (x = Q.1 ∧ y = Q.2) → x + 2*y - 5 = k := by
  sorry

end NUMINAMATH_CALUDE_chord_equation_l2461_246124


namespace NUMINAMATH_CALUDE_base4_division_l2461_246176

/-- Converts a base 4 number to base 10 -/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Theorem stating that the quotient of 1012₄ ÷ 12₄ is 23₄ -/
theorem base4_division :
  (base4ToBase10 [2, 1, 0, 1]) / (base4ToBase10 [2, 1]) = base4ToBase10 [3, 2] := by
  sorry

#eval base4ToBase10 [2, 1, 0, 1]  -- Should output 70
#eval base4ToBase10 [2, 1]        -- Should output 6
#eval base4ToBase10 [3, 2]        -- Should output 11
#eval base10ToBase4 23            -- Should output [2, 3]

end NUMINAMATH_CALUDE_base4_division_l2461_246176


namespace NUMINAMATH_CALUDE_square_area_probability_square_area_probability_proof_l2461_246146

/-- The probability of a randomly chosen point P on a line segment AB of length 10 cm
    resulting in a square with side length AP having an area between 25 cm² and 49 cm² -/
theorem square_area_probability : ℝ :=
  let AB : ℝ := 10
  let lower_bound : ℝ := 25
  let upper_bound : ℝ := 49
  1 / 5

/-- Proof of the theorem -/
theorem square_area_probability_proof :
  square_area_probability = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_area_probability_square_area_probability_proof_l2461_246146


namespace NUMINAMATH_CALUDE_problem_distribution_l2461_246171

def num_problems : ℕ := 5
def num_friends : ℕ := 12

theorem problem_distribution :
  (num_friends ^ num_problems : ℕ) = 248832 :=
by sorry

end NUMINAMATH_CALUDE_problem_distribution_l2461_246171


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2461_246163

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) : 
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 := by
  sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) : 
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2461_246163


namespace NUMINAMATH_CALUDE_toys_per_rabbit_l2461_246168

def monday_toys : ℕ := 8
def num_rabbits : ℕ := 34

def total_toys : ℕ :=
  monday_toys +  -- Monday
  (3 * monday_toys) +  -- Tuesday
  (2 * 3 * monday_toys) +  -- Wednesday
  monday_toys +  -- Thursday
  (5 * monday_toys) +  -- Friday
  (2 * 3 * monday_toys) / 2  -- Saturday

theorem toys_per_rabbit : total_toys / num_rabbits = 4 := by
  sorry

end NUMINAMATH_CALUDE_toys_per_rabbit_l2461_246168


namespace NUMINAMATH_CALUDE_salt_concentration_after_dilution_l2461_246135

/-- Calculates the final salt concentration after adding water to a salt solution -/
theorem salt_concentration_after_dilution
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (water_added : ℝ)
  (h1 : initial_volume = 56)
  (h2 : initial_concentration = 0.1)
  (h3 : water_added = 14) :
  let salt_amount := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added
  let final_concentration := salt_amount / final_volume
  final_concentration = 0.08 := by sorry

end NUMINAMATH_CALUDE_salt_concentration_after_dilution_l2461_246135


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2461_246147

/-- 
Given three consecutive terms of an arithmetic sequence with common difference 6,
prove that if their sum is 342, then the terms are 108, 114, and 120.
-/
theorem arithmetic_sequence_sum (a b c : ℕ) : 
  (b = a + 6 ∧ c = b + 6) →  -- consecutive terms with common difference 6
  (a + b + c = 342) →        -- sum is 342
  (a = 108 ∧ b = 114 ∧ c = 120) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2461_246147


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2461_246151

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) : x^3 + y^3 = -10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2461_246151


namespace NUMINAMATH_CALUDE_candy_division_theorem_l2461_246158

/-- Represents the share of candy each person takes -/
structure CandyShare where
  al : Rat
  bert : Rat
  carl : Rat
  dana : Rat

/-- The function that calculates the remaining candy fraction -/
def remainingCandy (shares : CandyShare) : Rat :=
  1 - (shares.al + shares.bert + shares.carl + shares.dana)

/-- The theorem stating the correct remaining candy fraction -/
theorem candy_division_theorem (x : Rat) (shares : CandyShare) :
  shares.al = 3/7 ∧
  shares.bert = 2/7 * (1 - 3/7) ∧
  shares.carl = 1/7 * (1 - 3/7 - 2/7 * (1 - 3/7)) ∧
  shares.dana = 1/7 * (1 - 3/7 - 2/7 * (1 - 3/7) - 1/7 * (1 - 3/7 - 2/7 * (1 - 3/7))) →
  remainingCandy shares = 584/2401 := by
  sorry

#check candy_division_theorem

end NUMINAMATH_CALUDE_candy_division_theorem_l2461_246158


namespace NUMINAMATH_CALUDE_position_from_front_l2461_246136

theorem position_from_front (total : ℕ) (position_from_back : ℕ) (h1 : total = 22) (h2 : position_from_back = 13) :
  total - position_from_back + 1 = 10 := by
sorry

end NUMINAMATH_CALUDE_position_from_front_l2461_246136


namespace NUMINAMATH_CALUDE_average_marks_of_combined_classes_l2461_246131

theorem average_marks_of_combined_classes 
  (class1_size : ℕ) (class1_avg : ℝ) 
  (class2_size : ℕ) (class2_avg : ℝ) : 
  let total_students := class1_size + class2_size
  let total_marks := class1_size * class1_avg + class2_size * class2_avg
  total_marks / total_students = (35 * 45 + 55 * 65) / (35 + 55) :=
by
  sorry

#eval (35 * 45 + 55 * 65) / (35 + 55)

end NUMINAMATH_CALUDE_average_marks_of_combined_classes_l2461_246131


namespace NUMINAMATH_CALUDE_parabola_passes_through_origin_l2461_246138

/-- A parabola defined by y = 3x^2 passes through the point (0, 0) -/
theorem parabola_passes_through_origin :
  let f : ℝ → ℝ := λ x ↦ 3 * x^2
  f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_passes_through_origin_l2461_246138


namespace NUMINAMATH_CALUDE_polygon_sides_l2461_246132

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 = 4 * 360 - 180) →
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2461_246132


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l2461_246185

/-- The y-coordinate of a point on the y-axis equidistant from (5, 0) and (3, 6) is 5/3 -/
theorem equidistant_point_y_coordinate :
  let A : ℝ × ℝ := (5, 0)
  let B : ℝ × ℝ := (3, 6)
  let P : ℝ → ℝ × ℝ := fun y ↦ (0, y)
  ∃ y : ℝ, (dist (P y) A)^2 = (dist (P y) B)^2 ∧ y = 5/3 :=
by sorry


end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l2461_246185


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l2461_246192

theorem modulus_of_complex_number : 
  let z : ℂ := Complex.I * (1 + 2 * Complex.I)
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l2461_246192


namespace NUMINAMATH_CALUDE_knights_probability_l2461_246191

/-- The number of knights seated at the round table -/
def total_knights : ℕ := 30

/-- The number of knights chosen randomly -/
def chosen_knights : ℕ := 4

/-- The probability that at least two of the chosen knights were sitting next to each other -/
def Q : ℚ := 4456 / 4701

theorem knights_probability :
  Q = 1 - (total_knights * (total_knights - 4) * (total_knights - 8) * (total_knights - 12)) /
      (total_knights * (total_knights - 1) * (total_knights - 2) * (total_knights - 3)) :=
sorry

end NUMINAMATH_CALUDE_knights_probability_l2461_246191


namespace NUMINAMATH_CALUDE_gcd_problem_l2461_246169

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = (2*k + 1) * 7767) :
  Int.gcd (6*a^2 + 5*a + 108) (3*a + 9) = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2461_246169


namespace NUMINAMATH_CALUDE_z_imaginary_and_fourth_quadrant_l2461_246174

def z (m : ℝ) : ℂ := m * (m + 2) + (m^2 + m - 2) * Complex.I

theorem z_imaginary_and_fourth_quadrant (m : ℝ) :
  (z m = Complex.I * Complex.im (z m) → m = 0) ∧
  (Complex.re (z m) > 0 ∧ Complex.im (z m) < 0 → 0 < m ∧ m < 1) :=
sorry

end NUMINAMATH_CALUDE_z_imaginary_and_fourth_quadrant_l2461_246174


namespace NUMINAMATH_CALUDE_find_P_value_l2461_246159

theorem find_P_value (P Q R B C y z : ℝ) 
  (eq1 : P = Q + R + 32)
  (eq2 : y = B + C + P + z)
  (eq3 : z = Q - R)
  (eq4 : B = 1/3 * P)
  (eq5 : C = 1/3 * P) :
  P = 64 := by
  sorry

end NUMINAMATH_CALUDE_find_P_value_l2461_246159


namespace NUMINAMATH_CALUDE_yellow_light_probability_l2461_246154

theorem yellow_light_probability (red_duration green_duration yellow_duration : ℕ) :
  red_duration = 30 →
  yellow_duration = 5 →
  green_duration = 45 →
  (yellow_duration : ℚ) / (red_duration + yellow_duration + green_duration) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_yellow_light_probability_l2461_246154


namespace NUMINAMATH_CALUDE_company_p_employee_count_l2461_246115

theorem company_p_employee_count (jan_employees : ℝ) : 
  jan_employees * 1.10 * 1.15 * 1.20 = 470 →
  ⌊jan_employees⌋ = 310 := by
  sorry

end NUMINAMATH_CALUDE_company_p_employee_count_l2461_246115


namespace NUMINAMATH_CALUDE_linear_function_slope_l2461_246114

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem linear_function_slope (k b : ℝ) :
  (∀ x : ℝ, linear_function k b (x + 3) = linear_function k b x - 2) →
  k = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_slope_l2461_246114


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l2461_246197

/-- Given a line and a circle in 2D space, prove that the sum of distances from a specific point to the intersection points of the line and circle is √6. -/
theorem intersection_distance_sum (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) (P A B : ℝ × ℝ) :
  l = {(x, y) : ℝ × ℝ | x + y = 1} →
  C = {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x + 2*y = 0} →
  P = (1, 0) →
  A ∈ l ∩ C →
  B ∈ l ∩ C →
  A ≠ B →
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) + Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l2461_246197


namespace NUMINAMATH_CALUDE_petya_stickers_l2461_246161

/-- Calculates the final number of stickers after a series of trades -/
def final_stickers (initial : ℕ) (trade_in : ℕ) (trade_out : ℕ) (num_trades : ℕ) : ℕ :=
  initial + num_trades * (trade_out - trade_in)

/-- Theorem: Petya will have 121 stickers after 30 trades -/
theorem petya_stickers :
  final_stickers 1 1 5 30 = 121 := by
  sorry

end NUMINAMATH_CALUDE_petya_stickers_l2461_246161


namespace NUMINAMATH_CALUDE_square_tile_area_l2461_246178

theorem square_tile_area (side_length : ℝ) (h : side_length = 7) :
  side_length * side_length = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_tile_area_l2461_246178


namespace NUMINAMATH_CALUDE_largest_undefined_value_l2461_246126

theorem largest_undefined_value (x : ℝ) :
  let f (x : ℝ) := (x + 2) / (9 * x^2 - 74 * x + 9)
  let roots := { x | 9 * x^2 - 74 * x + 9 = 0 }
  ∃ (max_root : ℝ), max_root ∈ roots ∧ ∀ (y : ℝ), y ∈ roots → y ≤ max_root ∧
  ∀ (z : ℝ), z > max_root → f z ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_undefined_value_l2461_246126


namespace NUMINAMATH_CALUDE_power_function_through_point_l2461_246116

/-- A power function that passes through the point (2, √2) -/
def f (x : ℝ) : ℝ := x ^ (1/2)

/-- Theorem: The power function f(x) that passes through (2, √2) satisfies f(8) = 2√2 -/
theorem power_function_through_point (x : ℝ) :
  f 2 = Real.sqrt 2 → f 8 = 2 * Real.sqrt 2 := by
  sorry

#check power_function_through_point

end NUMINAMATH_CALUDE_power_function_through_point_l2461_246116


namespace NUMINAMATH_CALUDE_x_cubed_greater_y_squared_l2461_246144

theorem x_cubed_greater_y_squared (x y : ℝ) 
  (h1 : x^5 > y^4) (h2 : y^5 > x^4) : x^3 > y^2 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_greater_y_squared_l2461_246144


namespace NUMINAMATH_CALUDE_solve_for_y_l2461_246140

theorem solve_for_y (x y : ℝ) (h1 : x^(2*y) = 64) (h2 : x = 8) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2461_246140


namespace NUMINAMATH_CALUDE_airplane_seating_l2461_246106

/-- A proof problem about airplane seating --/
theorem airplane_seating (first_class business_class economy_class : ℕ) 
  (h1 : first_class = 10)
  (h2 : business_class = 30)
  (h3 : economy_class = 50)
  (h4 : economy_class / 2 = first_class + (business_class - (business_class - x)))
  (h5 : first_class - 7 = 3)
  (x : ℕ) :
  x = 8 := by sorry

end NUMINAMATH_CALUDE_airplane_seating_l2461_246106


namespace NUMINAMATH_CALUDE_otimes_difference_l2461_246121

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := a^3 / b^2

-- State the theorem
theorem otimes_difference : 
  (otimes (otimes 2 4) 6) - (otimes 2 (otimes 4 6)) = -23327/288 := by
  sorry

end NUMINAMATH_CALUDE_otimes_difference_l2461_246121


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2461_246134

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (∃ c : ℝ, (x - 2/x)^4 = c*x^2 + (terms_without_x_squared : ℝ)) → 
  (∃ c : ℝ, (x - 2/x)^4 = 8*x^2 + (terms_without_x_squared : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2461_246134


namespace NUMINAMATH_CALUDE_base_flavors_count_l2461_246172

/-- The number of variations for each base flavor of pizza -/
def variations : ℕ := 4

/-- The total number of pizza varieties available -/
def total_varieties : ℕ := 16

/-- The number of base flavors of pizza -/
def base_flavors : ℕ := total_varieties / variations

theorem base_flavors_count : base_flavors = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_flavors_count_l2461_246172


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2461_246195

theorem opposite_of_negative_two : 
  (∀ x : ℤ, x + (-x) = 0) → (-2 + 2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2461_246195


namespace NUMINAMATH_CALUDE_jack_head_circumference_l2461_246141

theorem jack_head_circumference :
  ∀ (J C B : ℝ),
  C = J / 2 + 9 →
  B = 2 / 3 * C →
  B = 10 →
  J = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_head_circumference_l2461_246141


namespace NUMINAMATH_CALUDE_cone_angle_and_ratio_l2461_246199

/-- For a cone with ratio k of total surface area to axial cross-section area,
    prove the angle between height and slant height, and permissible k values. -/
theorem cone_angle_and_ratio (k : ℝ) (α : ℝ) : k > π ∧ α = π/2 - 2 * Real.arctan (π/k) → 
  (π * (Real.sin α + 1)) / Real.cos α = k := by
  sorry

end NUMINAMATH_CALUDE_cone_angle_and_ratio_l2461_246199


namespace NUMINAMATH_CALUDE_button_probability_l2461_246143

/-- Represents the number of buttons of each color in a jar -/
structure JarContents where
  red : ℕ
  blue : ℕ

/-- Represents the action of removing buttons from one jar to another -/
structure ButtonRemoval where
  removed : ℕ

theorem button_probability (initial_jar_a : JarContents) 
  (removal : ButtonRemoval) (final_jar_a : JarContents) :
  initial_jar_a.red = 4 →
  initial_jar_a.blue = 8 →
  removal.removed + removal.removed = initial_jar_a.red + initial_jar_a.blue - (final_jar_a.red + final_jar_a.blue) →
  3 * (final_jar_a.red + final_jar_a.blue) = 2 * (initial_jar_a.red + initial_jar_a.blue) →
  (final_jar_a.red / (final_jar_a.red + final_jar_a.blue : ℚ)) * 
  (removal.removed / ((initial_jar_a.red + initial_jar_a.blue - (final_jar_a.red + final_jar_a.blue)) : ℚ)) = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_button_probability_l2461_246143


namespace NUMINAMATH_CALUDE_class_average_calculation_l2461_246181

theorem class_average_calculation (total_students : ℕ) (monday_students : ℕ) (tuesday_students : ℕ)
  (monday_average : ℚ) (tuesday_average : ℚ) :
  total_students = 28 →
  monday_students = 24 →
  tuesday_students = 4 →
  monday_average = 82/100 →
  tuesday_average = 90/100 →
  let overall_average := (monday_students * monday_average + tuesday_students * tuesday_average) / total_students
  ∃ ε > 0, |overall_average - 83/100| < ε :=
by sorry

end NUMINAMATH_CALUDE_class_average_calculation_l2461_246181


namespace NUMINAMATH_CALUDE_range_of_f_l2461_246177

noncomputable def f (x : ℝ) : ℝ := (3 * x + 4) / (x - 5)

theorem range_of_f :
  Set.range f = {y : ℝ | y < 3 ∨ y > 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2461_246177


namespace NUMINAMATH_CALUDE_trader_profit_above_goal_l2461_246123

theorem trader_profit_above_goal 
  (total_profit : ℕ) 
  (goal_amount : ℕ) 
  (donation_amount : ℕ) 
  (h1 : total_profit = 960)
  (h2 : goal_amount = 610)
  (h3 : donation_amount = 310) :
  (total_profit / 2 + donation_amount) - goal_amount = 180 :=
by sorry

end NUMINAMATH_CALUDE_trader_profit_above_goal_l2461_246123


namespace NUMINAMATH_CALUDE_set_equality_implies_a_plus_minus_one_l2461_246101

theorem set_equality_implies_a_plus_minus_one (a : ℝ) :
  ({0, -1, 2*a} : Set ℝ) = {a-1, -abs a, a+1} →
  (a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_plus_minus_one_l2461_246101


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l2461_246184

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l2461_246184


namespace NUMINAMATH_CALUDE_sin_cos_product_tan_plus_sec_second_quadrant_tan_plus_sec_fourth_quadrant_l2461_246149

/-- The angle α with vertex at the origin and initial side on positive x-axis -/
structure Angle (α : ℝ) : Prop where
  vertex_origin : True
  initial_side_positive_x : True

/-- Point P on the terminal side of angle α -/
structure TerminalPoint (α : ℝ) (x y : ℝ) : Prop where
  on_terminal_side : True

/-- The terminal side of angle α lies on the line y = mx -/
structure TerminalLine (α : ℝ) (m : ℝ) : Prop where
  on_line : True

theorem sin_cos_product (α : ℝ) (h : Angle α) (p : TerminalPoint α (-1) 2) :
  Real.sin α * Real.cos α = -2/5 := by sorry

theorem tan_plus_sec_second_quadrant (α : ℝ) (h : Angle α) (l : TerminalLine α (-3)) 
  (q : 0 < α ∧ α < π) :
  Real.tan α + 3 / Real.cos α = -3 - 3 * Real.sqrt 10 := by sorry

theorem tan_plus_sec_fourth_quadrant (α : ℝ) (h : Angle α) (l : TerminalLine α (-3)) 
  (q : -π/2 < α ∧ α < 0) :
  Real.tan α + 3 / Real.cos α = -3 + 3 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_sin_cos_product_tan_plus_sec_second_quadrant_tan_plus_sec_fourth_quadrant_l2461_246149


namespace NUMINAMATH_CALUDE_line_y_intercept_l2461_246193

/-- A line with slope -3 and x-intercept (4, 0) has y-intercept (0, 12) -/
theorem line_y_intercept (line : ℝ → ℝ) (slope : ℝ) (x_intercept : ℝ × ℝ) :
  slope = -3 →
  x_intercept = (4, 0) →
  (∀ x, line x = slope * x + line 0) →
  line 4 = 0 →
  line 0 = 12 := by
sorry

end NUMINAMATH_CALUDE_line_y_intercept_l2461_246193


namespace NUMINAMATH_CALUDE_election_result_l2461_246137

/-- Represents an election with three candidates -/
structure Election where
  total_votes : ℕ
  votes_a : ℕ
  votes_b : ℕ
  votes_c : ℕ

/-- Conditions for the specific election scenario -/
def election_conditions (e : Election) : Prop :=
  e.votes_a = (32 * e.total_votes) / 100 ∧
  e.votes_b = (42 * e.total_votes) / 100 ∧
  e.votes_c = e.votes_b - 1908 ∧
  e.total_votes = e.votes_a + e.votes_b + e.votes_c

/-- The theorem to be proved -/
theorem election_result (e : Election) (h : election_conditions e) :
  e.votes_c = (26 * e.total_votes) / 100 ∧ e.total_votes = 11925 := by
  sorry

#check election_result

end NUMINAMATH_CALUDE_election_result_l2461_246137


namespace NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l2461_246113

theorem sum_smallest_largest_prime_1_to_50 : 
  ∃ (p q : ℕ), 
    (1 < p) ∧ (p ≤ 50) ∧ Nat.Prime p ∧
    (1 < q) ∧ (q ≤ 50) ∧ Nat.Prime q ∧
    (∀ r : ℕ, (1 < r) ∧ (r ≤ 50) ∧ Nat.Prime r → p ≤ r ∧ r ≤ q) ∧
    p + q = 49 :=
by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l2461_246113


namespace NUMINAMATH_CALUDE_hens_and_cows_problem_l2461_246107

theorem hens_and_cows_problem (total_animals : ℕ) (total_feet : ℕ) (hens : ℕ) (cows : ℕ) :
  total_animals = 46 →
  total_feet = 136 →
  total_animals = hens + cows →
  total_feet = 2 * hens + 4 * cows →
  hens = 24 := by
  sorry

end NUMINAMATH_CALUDE_hens_and_cows_problem_l2461_246107


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l2461_246108

theorem consecutive_odd_numbers_sum (n k : ℕ) : n > 0 ∧ k > 0 → 
  (∃ (seq : List ℕ), 
    (∀ i ∈ seq, ∃ j, i = n + 2 * j ∧ j ≤ k) ∧ 
    (seq.length = k + 1) ∧
    (seq.sum = 20 * (n + 2 * k)) ∧
    (seq.sum = 60 * n)) →
  n = 29 ∧ k = 29 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l2461_246108


namespace NUMINAMATH_CALUDE_lynne_spent_75_l2461_246145

/-- The total amount Lynne spent on books and magazines -/
def total_spent (cat_books solar_books magazines book_price magazine_price : ℕ) : ℕ :=
  cat_books * book_price + solar_books * book_price + magazines * magazine_price

/-- Theorem stating that Lynne spent $75 in total -/
theorem lynne_spent_75 :
  total_spent 7 2 3 7 4 = 75 := by
  sorry

end NUMINAMATH_CALUDE_lynne_spent_75_l2461_246145


namespace NUMINAMATH_CALUDE_investment_comparison_l2461_246129

def initial_AA : ℝ := 200
def initial_BB : ℝ := 150
def initial_CC : ℝ := 100

def year1_AA_change : ℝ := 1.30
def year1_BB_change : ℝ := 0.80
def year1_CC_change : ℝ := 1.10

def year2_AA_change : ℝ := 0.85
def year2_BB_change : ℝ := 1.30
def year2_CC_change : ℝ := 0.95

def final_A : ℝ := initial_AA * year1_AA_change * year2_AA_change
def final_B : ℝ := initial_BB * year1_BB_change * year2_BB_change
def final_C : ℝ := initial_CC * year1_CC_change * year2_CC_change

theorem investment_comparison : final_A > final_B ∧ final_B > final_C := by
  sorry

end NUMINAMATH_CALUDE_investment_comparison_l2461_246129


namespace NUMINAMATH_CALUDE_provisions_after_reinforcement_provisions_last_20_days_l2461_246110

theorem provisions_after_reinforcement 
  (initial_garrison : ℕ) 
  (initial_provisions : ℕ) 
  (days_before_reinforcement : ℕ) 
  (reinforcement : ℕ) : ℕ :=
  let remaining_provisions := initial_garrison * (initial_provisions - days_before_reinforcement)
  let total_men := initial_garrison + reinforcement
  remaining_provisions / total_men

theorem provisions_last_20_days 
  (initial_garrison : ℕ) 
  (initial_provisions : ℕ) 
  (days_before_reinforcement : ℕ) 
  (reinforcement : ℕ) :
  initial_garrison = 1000 →
  initial_provisions = 60 →
  days_before_reinforcement = 15 →
  reinforcement = 1250 →
  provisions_after_reinforcement initial_garrison initial_provisions days_before_reinforcement reinforcement = 20 :=
by sorry

end NUMINAMATH_CALUDE_provisions_after_reinforcement_provisions_last_20_days_l2461_246110


namespace NUMINAMATH_CALUDE_final_salary_matches_expected_l2461_246175

/-- Calculates the final take-home salary after a raise, pay cut, and tax --/
def finalSalary (initialSalary : ℝ) (raisePercent : ℝ) (cutPercent : ℝ) (taxPercent : ℝ) : ℝ :=
  let salaryAfterRaise := initialSalary * (1 + raisePercent)
  let salaryAfterCut := salaryAfterRaise * (1 - cutPercent)
  salaryAfterCut * (1 - taxPercent)

/-- Theorem stating that the final salary matches the expected value --/
theorem final_salary_matches_expected :
  finalSalary 2500 0.25 0.15 0.10 = 2390.63 := by
  sorry

#eval finalSalary 2500 0.25 0.15 0.10

end NUMINAMATH_CALUDE_final_salary_matches_expected_l2461_246175


namespace NUMINAMATH_CALUDE_smallest_four_digit_solution_l2461_246117

theorem smallest_four_digit_solution (x : ℕ) : x = 1094 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y, y ≥ 1000 ∧ y < 10000 →
    (9 * y ≡ 27 [ZMOD 18] ∧
     3 * y + 5 ≡ 11 [ZMOD 7] ∧
     -3 * y + 2 ≡ 2 * y [ZMOD 16]) →
    x ≤ y) ∧
  (9 * x ≡ 27 [ZMOD 18]) ∧
  (3 * x + 5 ≡ 11 [ZMOD 7]) ∧
  (-3 * x + 2 ≡ 2 * x [ZMOD 16]) := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_solution_l2461_246117


namespace NUMINAMATH_CALUDE_max_roses_for_680_l2461_246162

/-- Represents the price of roses for different quantities -/
structure RosePrices where
  individual : ℝ
  dozen : ℝ
  two_dozen : ℝ

/-- Calculates the maximum number of roses that can be purchased given a budget and price structure -/
def max_roses (budget : ℝ) (prices : RosePrices) : ℕ :=
  sorry

/-- The theorem stating the maximum number of roses that can be purchased with $680 -/
theorem max_roses_for_680 :
  let prices : RosePrices := {
    individual := 4.5,
    dozen := 36,
    two_dozen := 50
  }
  max_roses 680 prices = 318 := by
  sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l2461_246162


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_and_evaluation_l2461_246187

theorem algebraic_expression_simplification_and_evaluation :
  let x : ℝ := 4 * Real.sin (45 * π / 180) - 2
  let original_expression := (1 / (x - 1)) / ((x + 2) / (x^2 - 2*x + 1)) - x / (x + 2)
  let simplified_expression := -1 / (x + 2)
  original_expression = simplified_expression ∧ simplified_expression = -Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_and_evaluation_l2461_246187


namespace NUMINAMATH_CALUDE_seventeen_in_sample_l2461_246130

/-- Systematic sampling function -/
def systematicSample (populationSize sampleSize : ℕ) (first : ℕ) : List ℕ :=
  let interval := populationSize / sampleSize
  List.range sampleSize |>.map (fun i => first + i * interval)

/-- Theorem: In a systematic sample of size 4 from a population of 56, 
    if 3 is the first sampled number, then 17 will also be in the sample -/
theorem seventeen_in_sample :
  let sample := systematicSample 56 4 3
  17 ∈ sample := by
  sorry

end NUMINAMATH_CALUDE_seventeen_in_sample_l2461_246130


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l2461_246139

theorem necessary_and_sufficient_condition : 
  (∀ x : ℝ, x^2 - 2*x + 1 = 0 ↔ x = 1) := by sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l2461_246139


namespace NUMINAMATH_CALUDE_josh_remaining_money_l2461_246150

/-- Calculates the remaining money after spending two amounts -/
def remaining_money (initial : ℚ) (spent1 : ℚ) (spent2 : ℚ) : ℚ :=
  initial - (spent1 + spent2)

/-- Theorem: Given Josh's initial $9 and his spending of $1.75 and $1.25, he has $6 left -/
theorem josh_remaining_money :
  remaining_money 9 (175/100) (125/100) = 6 := by
  sorry

end NUMINAMATH_CALUDE_josh_remaining_money_l2461_246150


namespace NUMINAMATH_CALUDE_payment_calculation_l2461_246152

theorem payment_calculation (payment_rate : ℚ) (rooms_cleaned : ℚ) :
  payment_rate = 13 / 3 →
  rooms_cleaned = 8 / 5 →
  payment_rate * rooms_cleaned = 104 / 15 := by
  sorry

end NUMINAMATH_CALUDE_payment_calculation_l2461_246152


namespace NUMINAMATH_CALUDE_fruit_arrangements_proof_l2461_246155

def numFruitArrangements (apples oranges bananas totalDays : ℕ) : ℕ :=
  let bananasAsBlock := 1
  let nonBananaDays := totalDays - bananas + 1
  let arrangements := (Nat.factorial nonBananaDays) / 
    (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananasAsBlock)
  arrangements * nonBananaDays

theorem fruit_arrangements_proof :
  numFruitArrangements 4 3 3 10 = 2240 :=
by sorry

end NUMINAMATH_CALUDE_fruit_arrangements_proof_l2461_246155


namespace NUMINAMATH_CALUDE_expected_votes_for_candidate_a_l2461_246164

theorem expected_votes_for_candidate_a (total_voters : ℕ) 
  (dem_percent : ℝ) (rep_percent : ℝ) (dem_vote_a : ℝ) (rep_vote_a : ℝ) :
  dem_percent = 0.6 →
  rep_percent = 0.4 →
  dem_vote_a = 0.75 →
  rep_vote_a = 0.3 →
  dem_percent + rep_percent = 1 →
  (dem_percent * dem_vote_a + rep_percent * rep_vote_a) * 100 = 57 := by
  sorry

end NUMINAMATH_CALUDE_expected_votes_for_candidate_a_l2461_246164


namespace NUMINAMATH_CALUDE_equation_solution_l2461_246189

theorem equation_solution : 
  let x : ℝ := 405 / 8
  (2 * x - 60) / 3 = (2 * x - 5) / 7 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2461_246189


namespace NUMINAMATH_CALUDE_possible_values_of_u_l2461_246167

theorem possible_values_of_u (u v : ℝ) (hu : u ≠ 0) (hv : v ≠ 0)
  (eq1 : u + 1/v = 8) (eq2 : v + 1/u = 16/3) :
  u = 4 + Real.sqrt 232 / 4 ∨ u = 4 - Real.sqrt 232 / 4 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_u_l2461_246167


namespace NUMINAMATH_CALUDE_triangle_problem_l2461_246196

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) ∧
  -- Given conditions
  b = 2 ∧
  (1/2) * a * c * (Real.sin B) = Real.sqrt 3 →
  -- Conclusion
  B = π/3 ∧ a = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2461_246196


namespace NUMINAMATH_CALUDE_angles_on_axes_l2461_246100

def TerminalSideOnAxes (α : Real) : Prop :=
  ∃ k : ℤ, α = k * (Real.pi / 2)

theorem angles_on_axes :
  {α : Real | TerminalSideOnAxes α} = {α : Real | ∃ k : ℤ, α = k * (Real.pi / 2)} := by
  sorry

end NUMINAMATH_CALUDE_angles_on_axes_l2461_246100


namespace NUMINAMATH_CALUDE_bottle_capacity_correct_l2461_246160

/-- The capacity of Madeline's water bottle in ounces -/
def bottle_capacity : ℕ := 12

/-- The number of times Madeline refills her water bottle -/
def refills : ℕ := 7

/-- The amount of water Madeline needs to drink after refills in ounces -/
def remaining_water : ℕ := 16

/-- The total amount of water Madeline wants to drink in a day in ounces -/
def total_water : ℕ := 100

/-- Theorem stating that the bottle capacity is correct given the conditions -/
theorem bottle_capacity_correct :
  bottle_capacity * refills + remaining_water = total_water :=
by sorry

end NUMINAMATH_CALUDE_bottle_capacity_correct_l2461_246160


namespace NUMINAMATH_CALUDE_remaining_wire_length_l2461_246182

/-- Given a wire of length 60 cm and a square with side length 9 cm made from this wire,
    the remaining wire length is 24 cm. -/
theorem remaining_wire_length (total_wire : ℝ) (square_side : ℝ) (remaining_wire : ℝ) :
  total_wire = 60 ∧ square_side = 9 →
  remaining_wire = total_wire - 4 * square_side →
  remaining_wire = 24 := by
sorry

end NUMINAMATH_CALUDE_remaining_wire_length_l2461_246182


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l2461_246120

def f (x : ℝ) : ℝ := -x^3

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l2461_246120


namespace NUMINAMATH_CALUDE_complex_equation_implication_l2461_246125

theorem complex_equation_implication (a b : ℝ) :
  let z : ℂ := a + b * Complex.I
  (z * (z + 2 * Complex.I) * (z + 4 * Complex.I) = 5000 * Complex.I) →
  (a^3 - a * (b^2 + 6*b + 8) - (b+6) * (b^2 + 6*b + 8) = 0 ∧
   a * (b+6) - b * (b^2 + 6*b + 8) = 5000) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_implication_l2461_246125


namespace NUMINAMATH_CALUDE_certain_number_proof_l2461_246186

theorem certain_number_proof :
  ∃! x : ℝ, 0.65 * x = (4 / 5 : ℝ) * 25 + 6 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2461_246186


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2461_246103

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and eccentricity e = √5/2, prove that its asymptotes are y = ±(1/2)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (he : Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 / 2) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = (1/2) * x ∨ f x = -(1/2) * x) ∧
  (∀ ε > 0, ∃ M > 0, ∀ x y, x^2/a^2 - y^2/b^2 = 1 → abs x > M →
    abs (y - f x) < ε * abs x) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2461_246103


namespace NUMINAMATH_CALUDE_first_team_cups_l2461_246112

theorem first_team_cups (total : ℕ) (second : ℕ) (third : ℕ) 
  (h1 : total = 280)
  (h2 : second = 120)
  (h3 : third = 70) :
  ∃ first : ℕ, first + second + third = total ∧ first = 90 := by
  sorry

end NUMINAMATH_CALUDE_first_team_cups_l2461_246112


namespace NUMINAMATH_CALUDE_debbys_water_consumption_l2461_246170

/-- Given Debby's beverage consumption pattern, prove the number of water bottles she drank per day. -/
theorem debbys_water_consumption 
  (total_soda : ℕ) 
  (total_water : ℕ) 
  (soda_per_day : ℕ) 
  (soda_days : ℕ) 
  (water_days : ℕ) 
  (h1 : total_soda = 360)
  (h2 : total_water = 162)
  (h3 : soda_per_day = 9)
  (h4 : soda_days = 40)
  (h5 : water_days = 30)
  (h6 : total_soda = soda_per_day * soda_days) :
  (total_water : ℚ) / water_days = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_debbys_water_consumption_l2461_246170


namespace NUMINAMATH_CALUDE_least_integer_divisible_by_three_primes_l2461_246142

theorem least_integer_divisible_by_three_primes : 
  ∃ n : ℕ, n > 0 ∧ 
  (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ n % p = 0 ∧ n % q = 0 ∧ n % r = 0) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ m % p = 0 ∧ m % q = 0 ∧ m % r = 0) → 
    m ≥ 30) :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_divisible_by_three_primes_l2461_246142


namespace NUMINAMATH_CALUDE_sum_of_integers_l2461_246109

theorem sum_of_integers (a b c d e : ℤ) 
  (eq1 : a - b + c - e = 7)
  (eq2 : b - c + d + e = 8)
  (eq3 : c - d + a - e = 4)
  (eq4 : d - a + b + e = 3) :
  a + b + c + d + e = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2461_246109


namespace NUMINAMATH_CALUDE_no_winning_strategy_l2461_246153

/-- Represents a strategy for deciding when to stop in the card game. -/
def Strategy : Type := List Bool → Bool

/-- Represents the state of the game at any point. -/
structure GameState :=
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- The initial state of the game with a standard deck. -/
def initial_state : GameState :=
  { red_cards := 26, black_cards := 26 }

/-- Calculates the probability of winning given a game state and a strategy. -/
def winning_probability (state : GameState) (strategy : Strategy) : ℚ :=
  sorry

/-- Theorem stating that no strategy can have a winning probability greater than 0.5. -/
theorem no_winning_strategy :
  ∀ (strategy : Strategy),
    winning_probability initial_state strategy ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_no_winning_strategy_l2461_246153


namespace NUMINAMATH_CALUDE_ball_probability_l2461_246102

/-- Given a bag of balls with the specified conditions, prove the probability of choosing a ball that is neither red nor purple -/
theorem ball_probability (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h_total : total = 60)
  (h_red : red = 5)
  (h_purple : purple = 7) :
  (total - (red + purple)) / total = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l2461_246102


namespace NUMINAMATH_CALUDE_amount_per_painting_l2461_246165

/-- Hallie's art earnings -/
def total_earnings : ℕ := 300

/-- Number of paintings sold -/
def paintings_sold : ℕ := 3

/-- Theorem: The amount earned per painting is $100 -/
theorem amount_per_painting :
  total_earnings / paintings_sold = 100 := by sorry

end NUMINAMATH_CALUDE_amount_per_painting_l2461_246165


namespace NUMINAMATH_CALUDE_books_owned_by_three_l2461_246156

/-- The number of books owned by Harry, Flora, and Gary -/
def total_books (harry_books : ℕ) : ℕ :=
  let flora_books := 2 * harry_books
  let gary_books := harry_books / 2
  harry_books + flora_books + gary_books

/-- Theorem stating that the total number of books is 175 when Harry has 50 books -/
theorem books_owned_by_three (harry_books : ℕ) 
  (h : harry_books = 50) : total_books harry_books = 175 := by
  sorry

end NUMINAMATH_CALUDE_books_owned_by_three_l2461_246156


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l2461_246180

theorem largest_multiple_of_8_under_100 : ∃ n : ℕ, n * 8 = 96 ∧ 
  (∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l2461_246180


namespace NUMINAMATH_CALUDE_area_of_ABCD_l2461_246119

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The problem statement -/
theorem area_of_ABCD (r1 r2 r3 : Rectangle) : 
  r1.area + r2.area + r3.area = 8 ∧ r1.area = 2 → 
  ∃ (ABCD : Rectangle), ABCD.area = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_of_ABCD_l2461_246119


namespace NUMINAMATH_CALUDE_joan_book_revenue_l2461_246190

def total_revenue (total_books : ℕ) (books_at_4 : ℕ) (books_at_7 : ℕ) (price_4 : ℕ) (price_7 : ℕ) (price_10 : ℕ) : ℕ :=
  let remaining_books := total_books - books_at_4 - books_at_7
  books_at_4 * price_4 + books_at_7 * price_7 + remaining_books * price_10

theorem joan_book_revenue :
  total_revenue 33 15 6 4 7 10 = 222 := by
  sorry

end NUMINAMATH_CALUDE_joan_book_revenue_l2461_246190


namespace NUMINAMATH_CALUDE_total_diagonals_two_polygons_l2461_246111

/-- Number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The first polygon has 100 sides -/
def polygon1_sides : ℕ := 100

/-- The second polygon has 150 sides -/
def polygon2_sides : ℕ := 150

/-- Theorem: The total number of diagonals in a 100-sided polygon and a 150-sided polygon is 15875 -/
theorem total_diagonals_two_polygons : 
  diagonals polygon1_sides + diagonals polygon2_sides = 15875 := by
  sorry

end NUMINAMATH_CALUDE_total_diagonals_two_polygons_l2461_246111


namespace NUMINAMATH_CALUDE_bottles_per_player_first_break_l2461_246157

/-- Proves that each player took 2 bottles during the first break of a soccer match --/
theorem bottles_per_player_first_break :
  let total_bottles : ℕ := 4 * 12  -- 4 dozen
  let num_players : ℕ := 11
  let bottles_remaining : ℕ := 15
  let bottles_end_game : ℕ := num_players * 1  -- each player takes 1 bottle at the end

  let bottles_first_break : ℕ := total_bottles - bottles_end_game - bottles_remaining

  (bottles_first_break / num_players : ℚ) = 2 := by
sorry

end NUMINAMATH_CALUDE_bottles_per_player_first_break_l2461_246157


namespace NUMINAMATH_CALUDE_marcie_coffee_cups_l2461_246173

theorem marcie_coffee_cups (sandra_cups : ℕ) (total_cups : ℕ) (marcie_cups : ℕ) : 
  sandra_cups = 6 → total_cups = 8 → marcie_cups = total_cups - sandra_cups → marcie_cups = 2 := by
  sorry

end NUMINAMATH_CALUDE_marcie_coffee_cups_l2461_246173


namespace NUMINAMATH_CALUDE_computer_table_price_l2461_246179

/-- Calculates the selling price given the cost price and markup percentage -/
def selling_price (cost_price : ℚ) (markup_percent : ℚ) : ℚ :=
  cost_price * (1 + markup_percent / 100)

/-- Proves that the selling price of a computer table with cost price 4480 and 25% markup is 5600 -/
theorem computer_table_price : selling_price 4480 25 = 5600 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_price_l2461_246179
