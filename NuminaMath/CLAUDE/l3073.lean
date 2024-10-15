import Mathlib

namespace NUMINAMATH_CALUDE_inequality_equivalence_l3073_307331

theorem inequality_equivalence (x : ℝ) : 
  (x - 3) / 3 < (2 * x + 1) / 2 - 1 ↔ 2 * (x - 3) < 3 * (2 * x + 1) - 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3073_307331


namespace NUMINAMATH_CALUDE_bruce_grape_purchase_l3073_307324

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The quantity of mangoes purchased in kg -/
def mango_quantity : ℕ := 9

/-- The price of mangoes per kg -/
def mango_price : ℕ := 55

/-- The total amount paid -/
def total_paid : ℕ := 985

/-- The quantity of grapes purchased in kg -/
def grape_quantity : ℕ := (total_paid - mango_quantity * mango_price) / grape_price

theorem bruce_grape_purchase :
  grape_quantity * grape_price + mango_quantity * mango_price = total_paid ∧ grape_quantity = 7 := by
  sorry

end NUMINAMATH_CALUDE_bruce_grape_purchase_l3073_307324


namespace NUMINAMATH_CALUDE_robin_pieces_count_l3073_307373

theorem robin_pieces_count (gum_packages : ℕ) (candy_packages : ℕ) (pieces_per_package : ℕ) : 
  gum_packages = 28 → candy_packages = 14 → pieces_per_package = 6 →
  gum_packages * pieces_per_package + candy_packages * pieces_per_package = 252 := by
sorry

end NUMINAMATH_CALUDE_robin_pieces_count_l3073_307373


namespace NUMINAMATH_CALUDE_kim_cousins_count_l3073_307371

theorem kim_cousins_count (total_gum : ℕ) (gum_per_cousin : ℕ) (cousin_count : ℕ) : 
  total_gum = 20 → gum_per_cousin = 5 → total_gum = gum_per_cousin * cousin_count → cousin_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_kim_cousins_count_l3073_307371


namespace NUMINAMATH_CALUDE_stating_production_constraint_equations_l3073_307336

/-- Represents the daily production capacity for type A toys -/
def type_A_production : ℕ := 200

/-- Represents the daily production capacity for type B toys -/
def type_B_production : ℕ := 100

/-- Represents the number of type A parts required for one complete toy -/
def type_A_parts_per_toy : ℕ := 1

/-- Represents the number of type B parts required for one complete toy -/
def type_B_parts_per_toy : ℕ := 2

/-- Represents the total number of production days -/
def total_days : ℕ := 30

/-- 
Theorem stating that the given system of equations correctly represents 
the production constraints for maximizing toy assembly within 30 days
-/
theorem production_constraint_equations 
  (x y : ℕ) : 
  (x + y = total_days ∧ 
   type_A_production * type_A_parts_per_toy * x = type_B_production * y) ↔ 
  (x + y = 30 ∧ 400 * x = 100 * y) :=
sorry

end NUMINAMATH_CALUDE_stating_production_constraint_equations_l3073_307336


namespace NUMINAMATH_CALUDE_minimum_value_of_expression_minimum_value_achieved_l3073_307369

theorem minimum_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (((a^2 + b^2) * (4 * a^2 + b^2)).sqrt) / (a * b) ≥ Real.sqrt 6 :=
sorry

theorem minimum_value_achieved (a : ℝ) (ha : a > 0) :
  (((a^2 + a^2) * (4 * a^2 + a^2)).sqrt) / (a * a) = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_expression_minimum_value_achieved_l3073_307369


namespace NUMINAMATH_CALUDE_prob_different_colors_is_three_fourths_l3073_307304

/-- The set of colors for shorts -/
inductive ShortsColor
| Red
| Blue
| Green

/-- The set of colors for jerseys -/
inductive JerseyColor
| Red
| Blue
| Green
| Yellow

/-- The probability of choosing different colors for shorts and jersey -/
def prob_different_colors : ℚ := 3/4

/-- Theorem stating that the probability of choosing different colors for shorts and jersey is 3/4 -/
theorem prob_different_colors_is_three_fourths :
  prob_different_colors = 3/4 := by sorry

end NUMINAMATH_CALUDE_prob_different_colors_is_three_fourths_l3073_307304


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3073_307326

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → n ≤ 986 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3073_307326


namespace NUMINAMATH_CALUDE_base7_perfect_square_last_digit_l3073_307397

def is_base7_perfect_square (x y z : ℕ) : Prop :=
  x ≠ 0 ∧ z < 7 ∧ ∃ k : ℕ, k^2 = x * 7^3 + y * 7^2 + 5 * 7 + z

theorem base7_perfect_square_last_digit 
  (x y z : ℕ) (h : is_base7_perfect_square x y z) : z = 1 ∨ z = 6 := by
  sorry

end NUMINAMATH_CALUDE_base7_perfect_square_last_digit_l3073_307397


namespace NUMINAMATH_CALUDE_system_solution_l3073_307378

theorem system_solution :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (x * y = 500 ∧ x^(Real.log y) = 25) ↔
  ((x = 100 ∧ y = 5) ∨ (x = 5 ∧ y = 100)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3073_307378


namespace NUMINAMATH_CALUDE_intersection_volume_of_reflected_tetrahedron_l3073_307356

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ
  is_regular : Bool

/-- The intersection of two regular tetrahedra -/
def tetrahedra_intersection (t1 t2 : RegularTetrahedron) : ℝ := sorry

/-- Reflection of a regular tetrahedron through its center -/
def reflect_through_center (t : RegularTetrahedron) : RegularTetrahedron := sorry

theorem intersection_volume_of_reflected_tetrahedron (t : RegularTetrahedron) 
  (h1 : t.volume = 1)
  (h2 : t.is_regular = true) :
  tetrahedra_intersection t (reflect_through_center t) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_volume_of_reflected_tetrahedron_l3073_307356


namespace NUMINAMATH_CALUDE_probability_two_non_defective_10_2_l3073_307370

/-- Given a box of pens, calculates the probability of selecting two non-defective pens. -/
def probability_two_non_defective (total_pens : ℕ) (defective_pens : ℕ) : ℚ :=
  let non_defective := total_pens - defective_pens
  (non_defective : ℚ) / total_pens * (non_defective - 1) / (total_pens - 1)

/-- Theorem stating that the probability of selecting two non-defective pens
    from a box of 10 pens with 2 defective pens is 28/45. -/
theorem probability_two_non_defective_10_2 :
  probability_two_non_defective 10 2 = 28 / 45 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_non_defective_10_2_l3073_307370


namespace NUMINAMATH_CALUDE_medical_team_selection_l3073_307309

theorem medical_team_selection (nurses : ℕ) (doctors : ℕ) : 
  nurses = 3 → doctors = 6 → 
  (Nat.choose (nurses + doctors) 5 - Nat.choose doctors 5) = 120 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_selection_l3073_307309


namespace NUMINAMATH_CALUDE_tshirt_sales_optimization_l3073_307301

/-- Represents the profit function for T-shirt sales -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 3000

/-- Represents the sales volume function based on price increase -/
def sales_volume (x : ℝ) : ℝ := 300 - 10 * x

theorem tshirt_sales_optimization :
  let initial_price : ℝ := 40
  let purchase_price : ℝ := 30
  let target_profit : ℝ := 3360
  let optimal_increase : ℝ := 2
  let max_profit_price : ℝ := 50
  let max_profit : ℝ := 4000
  
  -- Part 1: Prove that increasing the price by 2 yuan yields the target profit
  (∃ x : ℝ, x ≥ 0 ∧ profit_function x = target_profit ∧
    ∀ y : ℝ, y ≥ 0 ∧ profit_function y = target_profit → x ≤ y) ∧
  profit_function optimal_increase = target_profit ∧
  
  -- Part 2: Prove that setting the price to 50 yuan maximizes profit
  (∀ x : ℝ, profit_function x ≤ max_profit) ∧
  profit_function (max_profit_price - initial_price) = max_profit := by
  sorry

end NUMINAMATH_CALUDE_tshirt_sales_optimization_l3073_307301


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3073_307365

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i^2 = -1 → Complex.im (i^5 / (1 - i)) = 1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3073_307365


namespace NUMINAMATH_CALUDE_prob_same_color_l3073_307396

/-- Represents the contents of a bag of colored balls -/
structure BagContents where
  white : ℕ
  red : ℕ
  black : ℕ

/-- Calculates the total number of balls in a bag -/
def BagContents.total (bag : BagContents) : ℕ :=
  bag.white + bag.red + bag.black

/-- Represents the two bags in the problem -/
def bagA : BagContents := { white := 1, red := 2, black := 3 }
def bagB : BagContents := { white := 2, red := 3, black := 1 }

/-- Calculates the probability of drawing a specific color from a bag -/
def probColor (bag : BagContents) (color : ℕ) : ℚ :=
  color / bag.total

/-- The main theorem: probability of drawing same color from both bags -/
theorem prob_same_color :
  (probColor bagA bagA.white * probColor bagB bagB.white) +
  (probColor bagA bagA.red * probColor bagB bagB.red) +
  (probColor bagA bagA.black * probColor bagB bagB.black) = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_l3073_307396


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3073_307315

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), n = 15 ∧ 
  (∀ (m : ℕ), m < n → ¬(23 ∣ (78721 - m))) ∧ (23 ∣ (78721 - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3073_307315


namespace NUMINAMATH_CALUDE_thirty_day_month_equal_sundays_tuesdays_l3073_307375

/-- Represents the days of the week -/
inductive DayOfWeek
| sunday
| monday
| tuesday
| wednesday
| thursday
| friday
| saturday

/-- Counts the occurrences of a specific day in a 30-day month starting from a given day -/
def countDay (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Checks if Sundays and Tuesdays are equal in a 30-day month starting from a given day -/
def hasSameSundaysAndTuesdays (startDay : DayOfWeek) : Bool :=
  countDay startDay DayOfWeek.sunday = countDay startDay DayOfWeek.tuesday

/-- Counts the number of possible starting days for a 30-day month with equal Sundays and Tuesdays -/
def countValidStartDays : Nat :=
  sorry

theorem thirty_day_month_equal_sundays_tuesdays :
  countValidStartDays = 3 :=
sorry

end NUMINAMATH_CALUDE_thirty_day_month_equal_sundays_tuesdays_l3073_307375


namespace NUMINAMATH_CALUDE_simplest_fraction_l3073_307343

variable (x : ℝ)

-- Define the fractions
def f1 : ℚ → ℚ := λ x => 4 / (2 * x)
def f2 : ℚ → ℚ := λ x => (x - 1) / (x^2 - 1)
def f3 : ℚ → ℚ := λ x => 1 / (x + 1)
def f4 : ℚ → ℚ := λ x => (1 - x) / (x - 1)

-- Define what it means for a fraction to be simplest
def is_simplest (f : ℚ → ℚ) : Prop :=
  ∀ g : ℚ → ℚ, (∀ x, f x = g x) → f = g

-- Theorem statement
theorem simplest_fraction :
  is_simplest f3 ∧ ¬is_simplest f1 ∧ ¬is_simplest f2 ∧ ¬is_simplest f4 := by
  sorry

end NUMINAMATH_CALUDE_simplest_fraction_l3073_307343


namespace NUMINAMATH_CALUDE_sum_of_divisors_24_l3073_307310

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_24 : sum_of_divisors 24 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_24_l3073_307310


namespace NUMINAMATH_CALUDE_hcf_of_ratio_numbers_l3073_307399

def ratio_numbers (x : ℕ) : Fin 4 → ℕ
  | 0 => 2 * x
  | 1 => 3 * x
  | 2 => 4 * x
  | 3 => 5 * x

theorem hcf_of_ratio_numbers (x : ℕ) (h1 : Nat.lcm (ratio_numbers x 0) (Nat.lcm (ratio_numbers x 1) (Nat.lcm (ratio_numbers x 2) (ratio_numbers x 3))) = 3600)
  (h2 : Nat.gcd (ratio_numbers x 2) (ratio_numbers x 3) = 4) :
  Nat.gcd (ratio_numbers x 0) (Nat.gcd (ratio_numbers x 1) (Nat.gcd (ratio_numbers x 2) (ratio_numbers x 3))) = 4 :=
by sorry

end NUMINAMATH_CALUDE_hcf_of_ratio_numbers_l3073_307399


namespace NUMINAMATH_CALUDE_min_difference_triangle_sides_l3073_307398

theorem min_difference_triangle_sides (a b c : ℕ) : 
  a < b → b < c → a + b + c = 2509 → 
  (∀ x y z : ℕ, x < y ∧ y < z ∧ x + y + z = 2509 → y - x ≥ b - a) → 
  b - a = 1 :=
sorry

end NUMINAMATH_CALUDE_min_difference_triangle_sides_l3073_307398


namespace NUMINAMATH_CALUDE_work_left_fraction_l3073_307351

theorem work_left_fraction (a_days : ℕ) (b_days : ℕ) (work_days : ℕ) :
  a_days = 15 →
  b_days = 20 →
  work_days = 4 →
  1 - (work_days * (1 / a_days + 1 / b_days)) = 8 / 15 :=
by sorry

end NUMINAMATH_CALUDE_work_left_fraction_l3073_307351


namespace NUMINAMATH_CALUDE_circle_through_points_with_center_on_y_axis_l3073_307376

/-- The circle passing through points A (-1, 4) and B (3, 2) with its center on the y-axis -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 10

/-- Point A coordinates -/
def point_A : ℝ × ℝ := (-1, 4)

/-- Point B coordinates -/
def point_B : ℝ × ℝ := (3, 2)

/-- The center of the circle is on the y-axis -/
def center_on_y_axis (h k : ℝ) : Prop :=
  h = 0

theorem circle_through_points_with_center_on_y_axis :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  ∃ k, center_on_y_axis 0 k ∧
    ∀ x y, circle_equation x y ↔ (x - 0)^2 + (y - k)^2 = (0 - point_A.1)^2 + (k - point_A.2)^2 :=
sorry

end NUMINAMATH_CALUDE_circle_through_points_with_center_on_y_axis_l3073_307376


namespace NUMINAMATH_CALUDE_parabola_latus_rectum_p_l3073_307360

/-- Given a parabola with equation x^2 = 2py (p > 0) and latus rectum equation y = -3,
    prove that the value of p is 6. -/
theorem parabola_latus_rectum_p (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, x^2 = 2*p*y) → (∃ x : ℝ, x^2 = 2*p*(-3)) → p = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_latus_rectum_p_l3073_307360


namespace NUMINAMATH_CALUDE_seventh_term_is_13_4_l3073_307346

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference of the sequence
  d : ℝ
  -- The sum of the first four terms is 14
  sum_first_four : a + (a + d) + (a + 2*d) + (a + 3*d) = 14
  -- The fifth term is 9
  fifth_term : a + 4*d = 9

/-- The seventh term of the arithmetic sequence is 13.4 -/
theorem seventh_term_is_13_4 (seq : ArithmeticSequence) :
  seq.a + 6*seq.d = 13.4 := by
  sorry


end NUMINAMATH_CALUDE_seventh_term_is_13_4_l3073_307346


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l3073_307321

/-- Given a marked price and cost price, where the cost price is 25% of the marked price,
    and a discount percentage such that the selling price after discount is equal to twice
    the cost price, prove that the discount percentage is 50%. -/
theorem discount_percentage_proof (MP CP : ℝ) (D : ℝ) 
    (h1 : CP = 0.25 * MP) 
    (h2 : MP * (1 - D / 100) = 2 * CP) : 
  D = 50 := by
  sorry

#check discount_percentage_proof

end NUMINAMATH_CALUDE_discount_percentage_proof_l3073_307321


namespace NUMINAMATH_CALUDE_no_good_tetrahedron_in_good_parallelepiped_l3073_307344

/-- A polyhedron is considered "good" if its volume equals its surface area -/
def isGoodPolyhedron (volume : ℝ) (surfaceArea : ℝ) : Prop :=
  volume = surfaceArea

/-- Properties of a tetrahedron -/
structure Tetrahedron where
  volume : ℝ
  surfaceArea : ℝ
  inscribedSphereRadius : ℝ

/-- Properties of a parallelepiped -/
structure Parallelepiped where
  volume : ℝ
  faceAreas : Fin 3 → ℝ
  heights : Fin 3 → ℝ

/-- Theorem stating the impossibility of fitting a good tetrahedron inside a good parallelepiped -/
theorem no_good_tetrahedron_in_good_parallelepiped :
  ∀ (t : Tetrahedron) (p : Parallelepiped),
    isGoodPolyhedron t.volume t.surfaceArea →
    isGoodPolyhedron p.volume (2 * (p.faceAreas 0 + p.faceAreas 1 + p.faceAreas 2)) →
    t.inscribedSphereRadius = 3 →
    ¬(∃ (h : ℝ), h = p.heights 0 ∧ h > 2 * t.inscribedSphereRadius) :=
by sorry

end NUMINAMATH_CALUDE_no_good_tetrahedron_in_good_parallelepiped_l3073_307344


namespace NUMINAMATH_CALUDE_trevor_dropped_eggs_l3073_307358

/-- The number of eggs Trevor collected from each chicken and the number left after dropping some -/
structure EggCollection where
  gertrude : Nat
  blanche : Nat
  nancy : Nat
  martha : Nat
  left : Nat

/-- The total number of eggs collected -/
def total_eggs (e : EggCollection) : Nat :=
  e.gertrude + e.blanche + e.nancy + e.martha

/-- The number of eggs Trevor dropped -/
def dropped_eggs (e : EggCollection) : Nat :=
  total_eggs e - e.left

theorem trevor_dropped_eggs (e : EggCollection) 
  (h1 : e.gertrude = 4)
  (h2 : e.blanche = 3)
  (h3 : e.nancy = 2)
  (h4 : e.martha = 2)
  (h5 : e.left = 9) :
  dropped_eggs e = 2 := by
  sorry

#check trevor_dropped_eggs

end NUMINAMATH_CALUDE_trevor_dropped_eggs_l3073_307358


namespace NUMINAMATH_CALUDE_triangle_expression_range_l3073_307313

theorem triangle_expression_range (A B C a b c : ℝ) : 
  0 < A → A < 3 * π / 4 →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  c * Real.sin A = a * Real.cos C →
  1 < Real.sqrt 3 * Real.sin A - Real.cos (B + π / 4) ∧ 
  Real.sqrt 3 * Real.sin A - Real.cos (B + π / 4) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_expression_range_l3073_307313


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3073_307350

theorem division_remainder_proof (dividend : ℕ) (divisor : ℚ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 13698 →
  divisor = 153.75280898876406 →
  quotient = 89 →
  dividend = (divisor * quotient).floor + remainder →
  remainder = 14 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3073_307350


namespace NUMINAMATH_CALUDE_projection_a_onto_b_l3073_307349

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 1)

theorem projection_a_onto_b :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  (dot_product / magnitude_b) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_projection_a_onto_b_l3073_307349


namespace NUMINAMATH_CALUDE_least_number_with_remainder_one_l3073_307352

theorem least_number_with_remainder_one (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < 386 → (m % 35 ≠ 1 ∨ m % 11 ≠ 1)) ∧ 
  386 % 35 = 1 ∧ 
  386 % 11 = 1 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_one_l3073_307352


namespace NUMINAMATH_CALUDE_boys_without_calculators_l3073_307303

theorem boys_without_calculators (total_boys : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ)
  (h1 : total_boys = 20)
  (h2 : students_with_calculators = 26)
  (h3 : girls_with_calculators = 15) :
  total_boys - (students_with_calculators - girls_with_calculators) = 9 := by
sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l3073_307303


namespace NUMINAMATH_CALUDE_point_on_line_l3073_307300

/-- Given five points O, A, B, C, D on a straight line with specified distances,
    and points P and Q satisfying certain ratio conditions, prove that OQ has the given value. -/
theorem point_on_line (a b c d : ℝ) :
  let O := 0
  let A := 2 * a
  let B := 4 * b
  let C := 5 * c
  let D := 7 * d
  let P := (14 * b * d - 10 * a * c) / (2 * a - 4 * b + 7 * d - 5 * c)
  let Q := (14 * c * d - 10 * b * c) / (5 * c - 7 * d)
  (A - P) / (P - D) = (B - P) / (P - C) →
  (Q - C) / (D - Q) = (B - C) / (D - C) →
  Q = (14 * c * d - 10 * b * c) / (5 * c - 7 * d) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_l3073_307300


namespace NUMINAMATH_CALUDE_investment_ratio_l3073_307332

theorem investment_ratio (P Q : ℝ) (h : P > 0 ∧ Q > 0) :
  (P * 5) / (Q * 9) = 7 / 9 → P / Q = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l3073_307332


namespace NUMINAMATH_CALUDE_basketball_score_l3073_307388

/-- Calculates the total points scored in a basketball game given the number of 2-point and 3-point shots made. -/
def totalPoints (twoPointShots threePointShots : ℕ) : ℕ :=
  2 * twoPointShots + 3 * threePointShots

/-- Proves that 7 two-point shots and 3 three-point shots result in a total of 23 points. -/
theorem basketball_score : totalPoints 7 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_l3073_307388


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3073_307323

/-- Represents a rectangular figure composed of square tiles -/
structure TiledRectangle where
  length : ℕ
  width : ℕ
  extra_tiles : ℕ

/-- Calculates the perimeter of a TiledRectangle -/
def perimeter (rect : TiledRectangle) : ℕ :=
  2 * (rect.length + rect.width)

/-- The initial rectangular figure -/
def initial_rectangle : TiledRectangle :=
  { length := 5, width := 2, extra_tiles := 1 }

theorem perimeter_after_adding_tiles :
  ∃ (final_rect : TiledRectangle),
    perimeter initial_rectangle = 16 ∧
    final_rect.length + final_rect.width = initial_rectangle.length + initial_rectangle.width + 2 ∧
    final_rect.extra_tiles = initial_rectangle.extra_tiles + 2 ∧
    perimeter final_rect = 18 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3073_307323


namespace NUMINAMATH_CALUDE_zoo_visitors_ratio_l3073_307337

theorem zoo_visitors_ratio :
  let friday_visitors : ℕ := 1250
  let saturday_visitors : ℕ := 3750
  (saturday_visitors : ℚ) / (friday_visitors : ℚ) = 3 := by
sorry

end NUMINAMATH_CALUDE_zoo_visitors_ratio_l3073_307337


namespace NUMINAMATH_CALUDE_range_of_a_range_of_g_l3073_307386

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 2*a + 12

-- Define the function g
def g (a : ℝ) : ℝ := (a + 1) * (|a - 1| + 2)

-- Theorem 1: Range of a
theorem range_of_a (h : ∀ x : ℝ, f a x ≥ 0) : a ∈ Set.Icc (-3/2) 2 :=
sorry

-- Theorem 2: Range of g(a)
theorem range_of_g : Set.range g = Set.Icc (-9/4) 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_g_l3073_307386


namespace NUMINAMATH_CALUDE_possible_integer_roots_l3073_307314

def polynomial (x b₂ b₁ : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 30

def is_root (x b₂ b₁ : ℤ) : Prop := polynomial x b₂ b₁ = 0

def divisors_of_30 : Set ℤ := {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30}

theorem possible_integer_roots (b₂ b₁ : ℤ) :
  {x : ℤ | ∃ (b₂ b₁ : ℤ), is_root x b₂ b₁} = divisors_of_30 := by sorry

end NUMINAMATH_CALUDE_possible_integer_roots_l3073_307314


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3073_307380

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def first_digit (n : ℕ) : ℕ := n / 10
def second_digit (n : ℕ) : ℕ := n % 10

def reverse_number (n : ℕ) : ℕ := 10 * (second_digit n) + (first_digit n)

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_two_digit n ∧ 
    (n : ℚ) / ((first_digit n * second_digit n) : ℚ) = 8 / 3 ∧
    n - reverse_number n = 18 ∧
    n = 64 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3073_307380


namespace NUMINAMATH_CALUDE_square_grid_15_toothpicks_l3073_307305

/-- Calculates the total number of toothpicks in a square grid -/
def toothpicks_in_square_grid (side_length : ℕ) : ℕ :=
  2 * side_length * (side_length + 1)

/-- Theorem: A square grid with sides of 15 toothpicks uses 480 toothpicks in total -/
theorem square_grid_15_toothpicks :
  toothpicks_in_square_grid 15 = 480 := by
  sorry

end NUMINAMATH_CALUDE_square_grid_15_toothpicks_l3073_307305


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3073_307372

theorem cubic_root_sum_cubes (x y z : ℝ) : 
  (x^3 - 5*x - 3 = 0) → 
  (y^3 - 5*y - 3 = 0) → 
  (z^3 - 5*z - 3 = 0) → 
  x^3 * y^3 + x^3 * z^3 + y^3 * z^3 = 99 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3073_307372


namespace NUMINAMATH_CALUDE_tile_count_l3073_307342

def room_length : ℕ := 18
def room_width : ℕ := 24
def border_width : ℕ := 2
def small_tile_size : ℕ := 1
def large_tile_size : ℕ := 3

def border_tiles : ℕ := 
  2 * (room_length + room_width - 2 * border_width) * border_width

def interior_length : ℕ := room_length - 2 * border_width
def interior_width : ℕ := room_width - 2 * border_width

def interior_tiles : ℕ := 
  (interior_length * interior_width) / (large_tile_size * large_tile_size)

def total_tiles : ℕ := border_tiles + interior_tiles

theorem tile_count : total_tiles = 167 := by
  sorry

end NUMINAMATH_CALUDE_tile_count_l3073_307342


namespace NUMINAMATH_CALUDE_proportional_relationship_l3073_307330

theorem proportional_relationship (x y z : ℝ) (k₁ k₂ : ℝ) :
  (∃ k₁ > 0, x = k₁ * y^3) →
  (∃ k₂ > 0, y = k₂ / z^2) →
  (x = 8 ∧ z = 16) →
  (z = 64 → x = 1/256) :=
by sorry

end NUMINAMATH_CALUDE_proportional_relationship_l3073_307330


namespace NUMINAMATH_CALUDE_sam_nickels_count_l3073_307393

/-- Calculates the final number of nickels Sam has -/
def final_nickels (initial : ℕ) (added : ℕ) (taken : ℕ) : ℕ :=
  initial + added - taken

theorem sam_nickels_count : final_nickels 29 24 13 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sam_nickels_count_l3073_307393


namespace NUMINAMATH_CALUDE_fraction_simplification_l3073_307383

theorem fraction_simplification : 
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3073_307383


namespace NUMINAMATH_CALUDE_fruit_difference_is_eight_l3073_307361

/-- Represents the number of fruits in a basket -/
structure FruitBasket where
  redPeaches : ℕ
  yellowPeaches : ℕ
  greenPeaches : ℕ
  blueApples : ℕ
  purpleBananas : ℕ
  orangeKiwis : ℕ

/-- Calculates the difference between peaches and other fruits -/
def peachDifference (basket : FruitBasket) : ℕ :=
  (basket.greenPeaches + basket.yellowPeaches) - (basket.blueApples + basket.purpleBananas)

/-- The theorem to be proved -/
theorem fruit_difference_is_eight :
  ∃ (basket : FruitBasket),
    basket.redPeaches = 2 ∧
    basket.yellowPeaches = 6 ∧
    basket.greenPeaches = 14 ∧
    basket.blueApples = 4 ∧
    basket.purpleBananas = 8 ∧
    basket.orangeKiwis = 12 ∧
    peachDifference basket = 8 := by
  sorry

end NUMINAMATH_CALUDE_fruit_difference_is_eight_l3073_307361


namespace NUMINAMATH_CALUDE_circle_rolling_inside_square_l3073_307312

/-- The distance traveled by the center of a circle rolling inside a square -/
theorem circle_rolling_inside_square
  (circle_radius : ℝ)
  (square_side : ℝ)
  (h1 : circle_radius = 1)
  (h2 : square_side = 5) :
  (square_side - 2 * circle_radius) * 4 = 12 :=
by sorry

end NUMINAMATH_CALUDE_circle_rolling_inside_square_l3073_307312


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3073_307374

-- Define the solution set type
def SolutionSet := Set ℝ

-- Define the inequalities
def Inequality1 (k a b c : ℝ) (x : ℝ) : Prop :=
  k / (x + a) + (x + b) / (x + c) < 0

def Inequality2 (k a b c : ℝ) (x : ℝ) : Prop :=
  (k * x) / (a * x + 1) + (b * x + 1) / (c * x + 1) < 0

-- State the theorem
theorem solution_set_equivalence 
  (k a b c : ℝ) 
  (S1 : SolutionSet) 
  (h1 : S1 = {x | x ∈ (Set.Ioo (-3) (-1)) ∪ (Set.Ioo 1 2) ∧ Inequality1 k a b c x}) :
  {x | Inequality2 k a b c x} = 
    {x | x ∈ (Set.Ioo (-1) (-1/3)) ∪ (Set.Ioo (1/2) 1)} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3073_307374


namespace NUMINAMATH_CALUDE_doughnuts_remaining_l3073_307329

/-- The number of doughnuts in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of doughnuts initially in the box -/
def initial_dozens : ℕ := 2

/-- The number of doughnuts eaten by the family -/
def eaten_doughnuts : ℕ := 8

/-- The number of doughnuts left in the box -/
def doughnuts_left : ℕ := initial_dozens * dozen - eaten_doughnuts

theorem doughnuts_remaining : doughnuts_left = 16 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_remaining_l3073_307329


namespace NUMINAMATH_CALUDE_degree_of_sum_polynomials_l3073_307391

-- Define the polynomials f and g
def f (z : ℂ) (c₃ c₂ c₁ c₀ : ℂ) : ℂ := c₃ * z^3 + c₂ * z^2 + c₁ * z + c₀
def g (z : ℂ) (d₂ d₁ d₀ : ℂ) : ℂ := d₂ * z^2 + d₁ * z + d₀

-- Define the degree of a polynomial
def degree (p : ℂ → ℂ) : ℕ := sorry

-- Theorem statement
theorem degree_of_sum_polynomials 
  (c₃ c₂ c₁ c₀ d₂ d₁ d₀ : ℂ) 
  (h₁ : c₃ ≠ 0) 
  (h₂ : d₂ ≠ 0) 
  (h₃ : c₃ + d₂ ≠ 0) : 
  degree (fun z ↦ f z c₃ c₂ c₁ c₀ + g z d₂ d₁ d₀) = 3 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_sum_polynomials_l3073_307391


namespace NUMINAMATH_CALUDE_function_composition_l3073_307339

/-- Given a function f(x) = (x(x-2))/2, prove that f(x+2) = ((x+2)x)/2 -/
theorem function_composition (x : ℝ) : 
  let f : ℝ → ℝ := λ x => (x * (x - 2)) / 2
  f (x + 2) = ((x + 2) * x) / 2 := by
sorry

end NUMINAMATH_CALUDE_function_composition_l3073_307339


namespace NUMINAMATH_CALUDE_rectangles_with_at_least_three_cells_l3073_307362

/-- The number of rectangles containing at least three cells in a 6x6 grid -/
def rectanglesWithAtLeastThreeCells : ℕ := 345

/-- The size of the grid -/
def gridSize : ℕ := 6

/-- Total number of rectangles in an n x n grid -/
def totalRectangles (n : ℕ) : ℕ := (n + 1).choose 2 * (n + 1).choose 2

/-- Number of 1x1 rectangles in an n x n grid -/
def oneByOneRectangles (n : ℕ) : ℕ := n * n

/-- Number of 1x2 and 2x1 rectangles in an n x n grid -/
def oneBytwoRectangles (n : ℕ) : ℕ := 2 * n * (n - 1)

theorem rectangles_with_at_least_three_cells :
  rectanglesWithAtLeastThreeCells = 
    totalRectangles gridSize - oneByOneRectangles gridSize - oneBytwoRectangles gridSize :=
by sorry

end NUMINAMATH_CALUDE_rectangles_with_at_least_three_cells_l3073_307362


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3073_307392

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 2*(m-1)*x + 4 = (x + a)^2) → 
  (m = 3 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3073_307392


namespace NUMINAMATH_CALUDE_girls_count_in_classroom_l3073_307307

theorem girls_count_in_classroom (ratio_girls : ℕ) (ratio_boys : ℕ) 
  (total_count : ℕ) (h1 : ratio_girls = 4) (h2 : ratio_boys = 3) 
  (h3 : total_count = 43) :
  (ratio_girls * total_count - ratio_girls) / (ratio_girls + ratio_boys) = 24 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_in_classroom_l3073_307307


namespace NUMINAMATH_CALUDE_school_students_problem_l3073_307366

theorem school_students_problem (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 150 →
  total = boys + girls →
  girls = (boys : ℚ) / 100 * total →
  boys = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_school_students_problem_l3073_307366


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l3073_307387

theorem unique_solution_to_equation : 
  ∃! (x y : ℕ+), (x.val : ℝ)^4 * (y.val : ℝ)^4 - 16 * (x.val : ℝ)^2 * (y.val : ℝ)^2 + 15 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l3073_307387


namespace NUMINAMATH_CALUDE_energy_usage_is_219_l3073_307316

/-- Calculates the total energy usage given the energy consumption and duration of each light. -/
def total_energy_usage (bedroom_watts_per_hour : ℝ) 
                       (bedroom_hours : ℝ)
                       (office_multiplier : ℝ)
                       (office_hours : ℝ)
                       (living_room_multiplier : ℝ)
                       (living_room_hours : ℝ)
                       (kitchen_multiplier : ℝ)
                       (kitchen_hours : ℝ)
                       (bathroom_multiplier : ℝ)
                       (bathroom_hours : ℝ) : ℝ :=
  bedroom_watts_per_hour * bedroom_hours +
  (office_multiplier * bedroom_watts_per_hour) * office_hours +
  (living_room_multiplier * bedroom_watts_per_hour) * living_room_hours +
  (kitchen_multiplier * bedroom_watts_per_hour) * kitchen_hours +
  (bathroom_multiplier * bedroom_watts_per_hour) * bathroom_hours

/-- Theorem stating that the total energy usage is 219 watts given the specified conditions. -/
theorem energy_usage_is_219 :
  total_energy_usage 6 2 3 3 4 4 2 1 5 1.5 = 219 := by
  sorry

end NUMINAMATH_CALUDE_energy_usage_is_219_l3073_307316


namespace NUMINAMATH_CALUDE_tangent_line_and_decreasing_condition_l3073_307340

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 + (1-2*a)*x + a

-- State the theorem
theorem tangent_line_and_decreasing_condition (a : ℝ) :
  -- The tangent line at x = 1 has equation 2x + y - 2 = 0
  (∃ m b : ℝ, ∀ x y : ℝ, y = f a x → (x = 1 → y = m*x + b) ∧ m = -2 ∧ b = 2) ∧
  -- f is strictly decreasing on ℝ iff a ∈ (3-√6, 3+√6)
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) ↔ (a > 3 - Real.sqrt 6 ∧ a < 3 + Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_decreasing_condition_l3073_307340


namespace NUMINAMATH_CALUDE_digit_150_of_75_over_625_l3073_307341

theorem digit_150_of_75_over_625 : ∃ (d : ℕ), d = 2 ∧ 
  (∃ (a b : ℕ), (75 : ℚ) / 625 = ↑a + (↑b / 100) ∧ 
  (∀ n : ℕ, (75 * 10^(n+2)) % 625 = (75 * 10^(n+150)) % 625) ∧
  d = ((75 * 10^150) / 625) % 10) :=
sorry

end NUMINAMATH_CALUDE_digit_150_of_75_over_625_l3073_307341


namespace NUMINAMATH_CALUDE_trip_time_difference_l3073_307319

def speed : ℝ := 60
def distance1 : ℝ := 360
def distance2 : ℝ := 420

theorem trip_time_difference : 
  (distance2 / speed - distance1 / speed) * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l3073_307319


namespace NUMINAMATH_CALUDE_train_length_calculation_train_B_length_l3073_307381

/-- Given two trains running in opposite directions, calculate the length of the second train. -/
theorem train_length_calculation (length_A : ℝ) (speed_A : ℝ) (speed_B : ℝ) (time : ℝ) : ℝ :=
  let relative_speed := (speed_A + speed_B) * 1000 / 3600
  let total_distance := relative_speed * time
  total_distance - length_A

/-- Prove that the length of Train B is approximately 219.95 meters. -/
theorem train_B_length :
  ∃ (length_B : ℝ), abs (length_B - train_length_calculation 280 120 80 9) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_train_B_length_l3073_307381


namespace NUMINAMATH_CALUDE_parabola_sum_coefficients_l3073_307318

-- Define the parabola equation
def parabola_eq (a b c : ℝ) (x y : ℝ) : Prop := x = a * y^2 + b * y + c

-- State the theorem
theorem parabola_sum_coefficients :
  ∀ a b c : ℝ,
  (parabola_eq a b c 6 (-5)) →
  (parabola_eq a b c 2 (-1)) →
  a + b + c = -3.25 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_coefficients_l3073_307318


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3073_307390

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3073_307390


namespace NUMINAMATH_CALUDE_magnitude_of_c_for_four_distinct_roots_l3073_307338

-- Define the polynomial Q(x)
def Q (c : ℂ) (x : ℂ) : ℂ := (x^2 - 3*x + 3) * (x^2 - c*x + 9) * (x^2 - 6*x + 18)

-- Theorem statement
theorem magnitude_of_c_for_four_distinct_roots (c : ℂ) :
  (∃ (s : Finset ℂ), s.card = 4 ∧ (∀ x ∈ s, Q c x = 0) ∧ (∀ x, Q c x = 0 → x ∈ s)) →
  Complex.abs c = Real.sqrt 35.25 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_c_for_four_distinct_roots_l3073_307338


namespace NUMINAMATH_CALUDE_base6_multiplication_l3073_307379

-- Define a function to convert from base 6 to base 10
def base6ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 6
def base10ToBase6 (n : ℕ) : ℕ := sorry

-- Define the multiplication operation in base 6
def multBase6 (a b : ℕ) : ℕ := 
  base10ToBase6 (base6ToBase10 a * base6ToBase10 b)

-- Theorem statement
theorem base6_multiplication :
  multBase6 132 14 = 1332 := by sorry

end NUMINAMATH_CALUDE_base6_multiplication_l3073_307379


namespace NUMINAMATH_CALUDE_gasoline_price_increase_l3073_307382

theorem gasoline_price_increase (initial_price initial_quantity : ℝ) 
  (h_price_increase : ℝ) (h_spending_increase : ℝ) (h_quantity_decrease : ℝ) :
  h_price_increase > 0 →
  h_spending_increase = 0.1 →
  h_quantity_decrease = 0.12 →
  initial_price * initial_quantity * (1 + h_spending_increase) = 
    initial_price * (1 + h_price_increase) * initial_quantity * (1 - h_quantity_decrease) →
  h_price_increase = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_l3073_307382


namespace NUMINAMATH_CALUDE_max_distance_on_circle_l3073_307385

open Complex

theorem max_distance_on_circle (z : ℂ) :
  Complex.abs (z - I) = 1 →
  (∀ w : ℂ, Complex.abs (w - I) = 1 → Complex.abs (z + 2 + I) ≥ Complex.abs (w + 2 + I)) →
  Complex.abs (z + 2 + I) = Real.sqrt 2 * 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_on_circle_l3073_307385


namespace NUMINAMATH_CALUDE_admission_cost_proof_l3073_307306

/-- Calculates the total cost of admission tickets for a group -/
def total_cost (adult_price child_price : ℕ) (num_children : ℕ) : ℕ :=
  let num_adults := num_children + 25
  let adult_cost := num_adults * adult_price
  let child_cost := num_children * child_price
  adult_cost + child_cost

/-- Proves that the total cost for the given group is $720 -/
theorem admission_cost_proof :
  total_cost 15 8 15 = 720 := by
  sorry

end NUMINAMATH_CALUDE_admission_cost_proof_l3073_307306


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3073_307355

theorem quadratic_minimum (x : ℝ) : ∃ (min : ℝ), min = -29 ∧ ∀ y : ℝ, x^2 + 14*x + 20 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3073_307355


namespace NUMINAMATH_CALUDE_leftmost_digit_of_12_to_37_l3073_307384

def log_2_lower : ℝ := 0.3010
def log_2_upper : ℝ := 0.3011
def log_3_lower : ℝ := 0.4771
def log_3_upper : ℝ := 0.4772

theorem leftmost_digit_of_12_to_37 
  (h1 : log_2_lower < Real.log 2)
  (h2 : Real.log 2 < log_2_upper)
  (h3 : log_3_lower < Real.log 3)
  (h4 : Real.log 3 < log_3_upper) :
  (12^37 : ℝ) ≥ 8 * 10^39 ∧ (12^37 : ℝ) < 9 * 10^39 :=
sorry

end NUMINAMATH_CALUDE_leftmost_digit_of_12_to_37_l3073_307384


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3073_307377

/-- A line is tangent to a circle if the distance from the center of the circle to the line is equal to the radius of the circle. -/
def is_tangent_line (a b c : ℝ) (r : ℝ) : Prop :=
  |c| / Real.sqrt (a^2 + b^2) = r

theorem tangent_line_to_circle (b : ℝ) :
  is_tangent_line 2 (-1) b (Real.sqrt 5) ↔ b = 5 ∨ b = -5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3073_307377


namespace NUMINAMATH_CALUDE_min_sheets_theorem_l3073_307308

/-- The minimum number of sheets in a pad of paper -/
def min_sheets_in_pad : ℕ := 36

/-- The number of weekdays -/
def weekdays : ℕ := 5

/-- The number of days Evelyn takes off per week -/
def days_off : ℕ := 2

/-- The number of sheets Evelyn uses per working day -/
def sheets_per_day : ℕ := 12

/-- Theorem stating that the minimum number of sheets in a pad of paper is 36 -/
theorem min_sheets_theorem : 
  min_sheets_in_pad = (weekdays - days_off) * sheets_per_day :=
by sorry

end NUMINAMATH_CALUDE_min_sheets_theorem_l3073_307308


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l3073_307302

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (m : Line) (n : Line) (α β γ : Plane) :
  parallel m α → perpendicular m β → perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l3073_307302


namespace NUMINAMATH_CALUDE_lower_price_proof_l3073_307311

/-- Given a book with cost C and two selling prices P and H, where H yields 5% more gain than P, 
    this function calculates the lower selling price P. -/
def calculate_lower_price (C H : ℚ) : ℚ :=
  H / (1 + 0.05)

theorem lower_price_proof (C H : ℚ) (hC : C = 200) (hH : H = 350) :
  let P := calculate_lower_price C H
  ∃ ε > 0, |P - 368.42| < ε := by sorry

end NUMINAMATH_CALUDE_lower_price_proof_l3073_307311


namespace NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l3073_307394

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice thrown -/
def numDice : ℕ := 4

/-- The minimum possible sum when throwing the dice -/
def minSum : ℕ := numDice

/-- The maximum possible sum when throwing the dice -/
def maxSum : ℕ := numDice * numFaces

/-- The number of possible distinct sums -/
def numDistinctSums : ℕ := maxSum - minSum + 1

/-- The minimum number of throws required to guarantee a repeated sum -/
def minThrows : ℕ := numDistinctSums + 1

theorem min_throws_for_repeated_sum :
  minThrows = 22 := by sorry

end NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l3073_307394


namespace NUMINAMATH_CALUDE_sum_in_base6_l3073_307348

/-- Converts a number from base 6 to base 10 -/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a number from base 10 to base 6 -/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec go (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else go (m / 6) ((m % 6) :: acc)
  go n []

/-- The main theorem to prove -/
theorem sum_in_base6 :
  let a := toBase10 [4, 4, 4]
  let b := toBase10 [6, 6]
  let c := toBase10 [4]
  toBase6 (a + b + c) = [6, 0, 2] := by sorry

end NUMINAMATH_CALUDE_sum_in_base6_l3073_307348


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3073_307354

theorem log_equality_implies_golden_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 8 = Real.log q / Real.log 18) ∧
  (Real.log q / Real.log 18 = Real.log (p + q) / Real.log 32) →
  q / p = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3073_307354


namespace NUMINAMATH_CALUDE_quadratic_equation_with_zero_sum_coefficients_l3073_307325

theorem quadratic_equation_with_zero_sum_coefficients :
  ∃ (a b c : ℝ), a ≠ 0 ∧ a + b + c = 0 ∧ ∀ x, a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_zero_sum_coefficients_l3073_307325


namespace NUMINAMATH_CALUDE_smallest_n_value_l3073_307317

def is_not_divisible_by_ten (m : ℕ) : Prop := ∀ k : ℕ, m ≠ 10 * k

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a ≥ b → b ≥ c →
  a + b + c = 2010 →
  a * b * c = m * (10 ^ n) →
  is_not_divisible_by_ten m →
  (∀ k : ℕ, k < n → ∃ (m' : ℕ), a * b * c = m' * (10 ^ k) → ¬(is_not_divisible_by_ten m')) →
  n = 500 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_value_l3073_307317


namespace NUMINAMATH_CALUDE_square_side_length_l3073_307395

theorem square_side_length (s : ℝ) (h : s > 0) :
  s^2 = 3 * (4 * s) → s = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3073_307395


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3073_307359

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → a * 1 - b * (-1) = 1 → (1 / a + 1 / b ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x * 1 - y * (-1) = 1 ∧ 1 / x + 1 / y = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3073_307359


namespace NUMINAMATH_CALUDE_race_distance_l3073_307364

theorem race_distance (a : ℝ) (r : ℝ) (S_n : ℝ) (n : ℕ) :
  a = 10 ∧ r = 2 ∧ S_n = 310 ∧ S_n = a * (r^n - 1) / (r - 1) →
  2^n = 32 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l3073_307364


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3073_307345

theorem fraction_to_decimal : 49 / 160 = 0.30625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3073_307345


namespace NUMINAMATH_CALUDE_square_difference_262_258_l3073_307367

theorem square_difference_262_258 : 262^2 - 258^2 = 2080 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_262_258_l3073_307367


namespace NUMINAMATH_CALUDE_min_value_expression_l3073_307328

theorem min_value_expression (x : ℝ) (h : x > 1) :
  (x + 4) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 5 ∧
  ((x + 4) / Real.sqrt (x - 1) = 2 * Real.sqrt 5 ↔ x = 6) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3073_307328


namespace NUMINAMATH_CALUDE_constant_k_equality_l3073_307347

theorem constant_k_equality (k : ℝ) : 
  (∀ x : ℝ, -x^2 - (k + 9)*x - 8 = -(x - 2)*(x - 4)) → k = -15 := by
  sorry

end NUMINAMATH_CALUDE_constant_k_equality_l3073_307347


namespace NUMINAMATH_CALUDE_part1_part2_l3073_307335

-- Define the points
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-2, 3)
def C : ℝ × ℝ := (8, -5)

-- Define vectors
def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Part 1
theorem part1 (x y : ℝ) : 
  OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2) → x = 2 ∧ y = -3 := by sorry

-- Part 2
theorem part2 (m : ℝ) :
  ∃ (k : ℝ), k ≠ 0 ∧ AB = (k * (m * OA.1 + OC.1), k * (m * OA.2 + OC.2)) → m = 1 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3073_307335


namespace NUMINAMATH_CALUDE_equation_solutions_l3073_307322

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 - 16 = 0 ↔ x = 6 ∨ x = -2) ∧
  (∀ x : ℝ, (x + 3)^3 = -27 ↔ x = -6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3073_307322


namespace NUMINAMATH_CALUDE_rectangle_area_equality_l3073_307353

theorem rectangle_area_equality (x y : ℝ) : 
  x * y = (x + 4) * (y - 3) ∧ 
  x * y = (x + 8) * (y - 4) → 
  x + y = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_equality_l3073_307353


namespace NUMINAMATH_CALUDE_divisibility_criterion_a_divisibility_criterion_b_l3073_307389

-- Part a
theorem divisibility_criterion_a (n : ℕ) : 
  (∃ q : Polynomial ℚ, X^(2*n) + X^n + 1 = (X^2 + X + 1) * q) ↔ 
  (n % 3 = 1 ∨ n % 3 = 2) :=
sorry

-- Part b
theorem divisibility_criterion_b (n : ℕ) : 
  (∃ q : Polynomial ℚ, X^(2*n) - X^n + 1 = (X^2 - X + 1) * q) ↔ 
  (n % 6 = 1 ∨ n % 6 = 5) :=
sorry

end NUMINAMATH_CALUDE_divisibility_criterion_a_divisibility_criterion_b_l3073_307389


namespace NUMINAMATH_CALUDE_millet_exceeds_60_percent_on_day_4_l3073_307320

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  millet : Float
  other_seeds : Float

/-- Calculates the next day's feeder state -/
def next_day_state (state : FeederState) : FeederState :=
  { millet := state.millet * 0.7 + 0.3,
    other_seeds := state.other_seeds * 0.5 + 0.7 }

/-- Calculates the proportion of millet in the feeder -/
def millet_proportion (state : FeederState) : Float :=
  state.millet / (state.millet + state.other_seeds)

/-- Initial state of the feeder -/
def initial_state : FeederState := { millet := 0.3, other_seeds := 0.7 }

theorem millet_exceeds_60_percent_on_day_4 :
  let day1 := initial_state
  let day2 := next_day_state day1
  let day3 := next_day_state day2
  let day4 := next_day_state day3
  (millet_proportion day1 ≤ 0.6) ∧
  (millet_proportion day2 ≤ 0.6) ∧
  (millet_proportion day3 ≤ 0.6) ∧
  (millet_proportion day4 > 0.6) :=
by sorry

end NUMINAMATH_CALUDE_millet_exceeds_60_percent_on_day_4_l3073_307320


namespace NUMINAMATH_CALUDE_total_cookies_l3073_307357

theorem total_cookies (num_bags : ℕ) (cookies_per_bag : ℕ) : 
  num_bags = 7 → cookies_per_bag = 2 → num_bags * cookies_per_bag = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l3073_307357


namespace NUMINAMATH_CALUDE_abc_product_l3073_307327

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 152)
  (h2 : b * (c + a) = 162)
  (h3 : c * (a + b) = 170) :
  a * b * c = 720 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l3073_307327


namespace NUMINAMATH_CALUDE_coefficient_x_squared_sum_powers_l3073_307334

/-- The sum of the first n natural numbers -/
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n triangular numbers -/
def sum_triangular (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem coefficient_x_squared_sum_powers (n : ℕ) (h : n = 10) : 
  sum_triangular n = 165 := by
  sorry

#eval sum_triangular 10

end NUMINAMATH_CALUDE_coefficient_x_squared_sum_powers_l3073_307334


namespace NUMINAMATH_CALUDE_canada_sqft_per_person_approx_l3073_307368

/-- The population of Canada in 2020 -/
def canada_population : ℕ := 38005238

/-- The total area of Canada in square miles -/
def canada_area : ℕ := 3855100

/-- The number of square feet in one square mile -/
def sqft_per_sqmile : ℕ := 5280^2

/-- Theorem stating that the average number of square feet per person in Canada
    is approximately 3,000,000 -/
theorem canada_sqft_per_person_approx :
  let total_sqft := canada_area * sqft_per_sqmile
  let avg_sqft_per_person := total_sqft / canada_population
  ∃ (ε : ℝ), ε > 0 ∧ ε < 200000 ∧ 
    (avg_sqft_per_person : ℝ) ≥ 3000000 - ε ∧ 
    (avg_sqft_per_person : ℝ) ≤ 3000000 + ε :=
sorry

end NUMINAMATH_CALUDE_canada_sqft_per_person_approx_l3073_307368


namespace NUMINAMATH_CALUDE_park_area_is_525_l3073_307363

/-- Represents a rectangular park with given perimeter and length-width relationship. -/
structure RectangularPark where
  perimeter : ℝ
  length : ℝ
  width : ℝ
  perimeter_eq : perimeter = 2 * (length + width)
  length_eq : length = 3 * width - 10

/-- Calculates the area of a rectangular park. -/
def parkArea (park : RectangularPark) : ℝ := park.length * park.width

/-- Theorem stating that a rectangular park with perimeter 100 meters and length equal to
    three times the width minus 10 meters has an area of 525 square meters. -/
theorem park_area_is_525 (park : RectangularPark) 
    (h_perimeter : park.perimeter = 100) : parkArea park = 525 := by
  sorry

end NUMINAMATH_CALUDE_park_area_is_525_l3073_307363


namespace NUMINAMATH_CALUDE_square_product_extension_l3073_307333

theorem square_product_extension (a b : ℕ) 
  (h1 : ∃ x : ℕ, a * b = x ^ 2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y ^ 2) :
  ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_product_extension_l3073_307333
