import Mathlib

namespace NUMINAMATH_CALUDE_existence_of_two_integers_l2278_227897

theorem existence_of_two_integers (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ q₁ q₂ : ℕ, q₁ ≠ q₂ ∧
    1 ≤ q₁ ∧ q₁ ≤ p - 1 ∧
    1 ≤ q₂ ∧ q₂ ≤ p - 1 ∧
    (q₁^(p-1) : ℤ) % p^2 = 1 ∧
    (q₂^(p-1) : ℤ) % p^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_two_integers_l2278_227897


namespace NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l2278_227862

theorem proposition_false_iff_a_in_range (a : ℝ) : 
  (¬ ∃ x : ℝ, |x - a| + |x - 1| ≤ 2) ↔ (a < -1 ∨ a > 3) := by
  sorry

end NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l2278_227862


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l2278_227894

theorem weekend_rain_probability (p_sat p_sun : ℝ) 
  (h_sat : p_sat = 0.6) 
  (h_sun : p_sun = 0.7) 
  (h_independent : True) -- We don't need to express independence in the statement
  : 1 - (1 - p_sat) * (1 - p_sun) = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l2278_227894


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2278_227827

/-- For positive real numbers a, b, c ≤ √2 with abc = 2, 
    prove √2 ∑(ab + 3c)/(3ab + c) ≥ a + b + c -/
theorem cyclic_sum_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha2 : a ≤ Real.sqrt 2) (hb2 : b ≤ Real.sqrt 2) (hc2 : c ≤ Real.sqrt 2)
  (habc : a * b * c = 2) :
  Real.sqrt 2 * (((a * b + 3 * c) / (3 * a * b + c)) +
                 ((b * c + 3 * a) / (3 * b * c + a)) +
                 ((c * a + 3 * b) / (3 * c * a + b))) ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2278_227827


namespace NUMINAMATH_CALUDE_pizzeria_small_pizzas_sold_l2278_227835

/-- Calculates the number of small pizzas sold given the prices, total sales, and number of large pizzas sold. -/
def small_pizzas_sold (small_price large_price total_sales : ℕ) (large_pizzas_sold : ℕ) : ℕ :=
  (total_sales - large_price * large_pizzas_sold) / small_price

/-- Theorem stating that the number of small pizzas sold is 8 under the given conditions. -/
theorem pizzeria_small_pizzas_sold :
  small_pizzas_sold 2 8 40 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizzeria_small_pizzas_sold_l2278_227835


namespace NUMINAMATH_CALUDE_compass_leg_swap_impossible_l2278_227837

/-- Represents a point on the integer grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents the state of the compass -/
structure CompassState where
  leg1 : GridPoint
  leg2 : GridPoint

/-- Calculates the squared distance between two grid points -/
def squaredDistance (p1 p2 : GridPoint) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Defines a valid move of the compass -/
def isValidMove (s1 s2 : CompassState) : Prop :=
  (s1.leg1 = s2.leg1 ∧ squaredDistance s1.leg1 s1.leg2 = squaredDistance s2.leg1 s2.leg2) ∨
  (s1.leg2 = s2.leg2 ∧ squaredDistance s1.leg1 s1.leg2 = squaredDistance s2.leg1 s2.leg2)

/-- Defines a sequence of valid moves -/
def isValidMoveSequence : List CompassState → Prop
  | [] => True
  | [_] => True
  | s1 :: s2 :: rest => isValidMove s1 s2 ∧ isValidMoveSequence (s2 :: rest)

theorem compass_leg_swap_impossible (start finish : CompassState) 
  (h_start_distance : squaredDistance start.leg1 start.leg2 = squaredDistance finish.leg1 finish.leg2)
  (h_swap : start.leg1 = finish.leg2 ∧ start.leg2 = finish.leg1) :
  ¬∃ (moves : List CompassState), isValidMoveSequence (start :: moves ++ [finish]) :=
sorry

end NUMINAMATH_CALUDE_compass_leg_swap_impossible_l2278_227837


namespace NUMINAMATH_CALUDE_right_triangle_area_l2278_227870

theorem right_triangle_area (h : ℝ) (α : ℝ) (area : ℝ) : 
  h = 12 →
  α = 30 * π / 180 →
  area = 18 * Real.sqrt 3 →
  area = (1 / 2) * h * h * Real.sin α * Real.cos α :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2278_227870


namespace NUMINAMATH_CALUDE_heroes_total_l2278_227891

theorem heroes_total (front back : ℕ) (h1 : front = 2) (h2 : back = 7) :
  front + back = 9 := by sorry

end NUMINAMATH_CALUDE_heroes_total_l2278_227891


namespace NUMINAMATH_CALUDE_grape_rate_proof_l2278_227841

/-- The rate of grapes per kilogram -/
def grape_rate : ℝ := 70

/-- The amount of grapes purchased in kilograms -/
def grape_amount : ℝ := 8

/-- The rate of mangoes per kilogram -/
def mango_rate : ℝ := 60

/-- The amount of mangoes purchased in kilograms -/
def mango_amount : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 1100

theorem grape_rate_proof : 
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_proof_l2278_227841


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_P_l2278_227887

/-- Given a point P in 3D space, this function returns its symmetric point with respect to the y-axis -/
def symmetricPointYAxis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := P
  (-x, y, z)

/-- Theorem stating that the symmetric point of P(1,2,-1) with respect to the y-axis is (-1,2,1) -/
theorem symmetric_point_y_axis_P :
  symmetricPointYAxis (1, 2, -1) = (-1, 2, -1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_P_l2278_227887


namespace NUMINAMATH_CALUDE_triangle_side_length_l2278_227802

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  A = 30 * π / 180 →
  B = 45 * π / 180 →
  b = 8 →
  a / Real.sin A = b / Real.sin B →
  a = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2278_227802


namespace NUMINAMATH_CALUDE_group_50_properties_l2278_227821

def last_number (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2)

def first_number (n : ℕ) : ℕ := last_number n - 2 * (n - 1)

def sum_of_group (n : ℕ) : ℕ := n * (first_number n + last_number n) / 2

theorem group_50_properties :
  last_number 50 = 2550 ∧
  first_number 50 = 2452 ∧
  sum_of_group 50 = 50 * 2501 := by
  sorry

end NUMINAMATH_CALUDE_group_50_properties_l2278_227821


namespace NUMINAMATH_CALUDE_inequality_solution_l2278_227834

noncomputable def f (x : ℝ) : ℝ := x^2 / ((x - 2)^2 * (x + 1))

theorem inequality_solution :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -1 →
  (f x ≥ 0 ↔ x ∈ Set.Iio (-1) ∪ {0} ∪ Set.Ioi 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2278_227834


namespace NUMINAMATH_CALUDE_coin_combination_difference_l2278_227861

def coin_values : List Nat := [5, 10, 20, 25]

def total_amount : Nat := 45

def is_valid_combination (combination : List Nat) : Prop :=
  combination.all (λ x => x ∈ coin_values) ∧
  combination.sum = total_amount

def num_coins (combination : List Nat) : Nat :=
  combination.length

theorem coin_combination_difference :
  ∃ (min_combination max_combination : List Nat),
    is_valid_combination min_combination ∧
    is_valid_combination max_combination ∧
    (∀ c, is_valid_combination c → 
      num_coins min_combination ≤ num_coins c ∧
      num_coins c ≤ num_coins max_combination) ∧
    num_coins max_combination - num_coins min_combination = 7 :=
  sorry

end NUMINAMATH_CALUDE_coin_combination_difference_l2278_227861


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2278_227892

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 3 - 3 * Complex.I) * (2 * Real.sqrt 2 + 2 * Complex.I)) = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2278_227892


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2278_227801

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 5 }

/-- Theorem: The maximum number of soap boxes that can fit in the carton is 300 -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 300 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2278_227801


namespace NUMINAMATH_CALUDE_paint_mixture_theorem_l2278_227817

/-- Proves that mixing 5 gallons of 20% yellow paint with 5/3 gallons of 40% yellow paint 
    results in a mixture that is 25% yellow -/
theorem paint_mixture_theorem (x : ℝ) :
  let light_green_volume : ℝ := 5
  let light_green_yellow_percent : ℝ := 0.2
  let dark_green_yellow_percent : ℝ := 0.4
  let target_yellow_percent : ℝ := 0.25
  x = 5/3 →
  (light_green_volume * light_green_yellow_percent + x * dark_green_yellow_percent) / 
  (light_green_volume + x) = target_yellow_percent :=
by sorry

end NUMINAMATH_CALUDE_paint_mixture_theorem_l2278_227817


namespace NUMINAMATH_CALUDE_permutations_5_3_l2278_227855

/-- The number of permutations of k elements chosen from n elements -/
def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- Theorem: The number of permutations A_5^3 equals 60 -/
theorem permutations_5_3 : permutations 5 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_permutations_5_3_l2278_227855


namespace NUMINAMATH_CALUDE_nine_power_comparison_l2278_227805

theorem nine_power_comparison : 9^(10^10) > 9^20 := by
  sorry

end NUMINAMATH_CALUDE_nine_power_comparison_l2278_227805


namespace NUMINAMATH_CALUDE_toy_donation_difference_l2278_227840

def leila_bags : ℕ := 2
def leila_toys_per_bag : ℕ := 25
def mohamed_bags : ℕ := 3
def mohamed_toys_per_bag : ℕ := 19

theorem toy_donation_difference : 
  mohamed_bags * mohamed_toys_per_bag - leila_bags * leila_toys_per_bag = 7 := by
  sorry

end NUMINAMATH_CALUDE_toy_donation_difference_l2278_227840


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l2278_227872

theorem sum_reciprocals_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) :
  1/x + 1/y ≥ 8 ∧ ∀ ε > 0, ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x'^2 + y'^2 = 1 ∧ 1/x' + 1/y' > 1/ε :=
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l2278_227872


namespace NUMINAMATH_CALUDE_four_digit_decimal_problem_l2278_227883

theorem four_digit_decimal_problem :
  ∃ (x : ℕ), 
    (1000 ≤ x ∧ x < 10000) ∧
    ((x : ℝ) - (x : ℝ) / 10 = 2059.2 ∨ (x : ℝ) - (x : ℝ) / 100 = 2059.2) ∧
    (x = 2288 ∨ x = 2080) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_decimal_problem_l2278_227883


namespace NUMINAMATH_CALUDE_remainder_theorem_l2278_227880

theorem remainder_theorem (n : ℤ) : n % 7 = 3 → (5 * n - 12) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2278_227880


namespace NUMINAMATH_CALUDE_root_values_l2278_227899

theorem root_values (a b c d k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * k^4 + b * k^3 + c * k^2 + d * k + a = 0)
  (h2 : a * k^3 + b * k^2 + c * k + d = 0) :
  k = Complex.I^(1/4) ∨ k = -Complex.I^(1/4) ∨ k = Complex.I^(3/4) ∨ k = -Complex.I^(3/4) :=
sorry

end NUMINAMATH_CALUDE_root_values_l2278_227899


namespace NUMINAMATH_CALUDE_cloth_length_proof_l2278_227848

/-- The length of a piece of cloth satisfying given cost conditions -/
def cloth_length : ℝ := 10

/-- The cost of the cloth -/
def total_cost : ℝ := 35

/-- The additional length in the hypothetical scenario -/
def additional_length : ℝ := 4

/-- The price reduction per meter in the hypothetical scenario -/
def price_reduction : ℝ := 1

theorem cloth_length_proof :
  cloth_length > 0 ∧
  total_cost = cloth_length * (total_cost / cloth_length) ∧
  total_cost = (cloth_length + additional_length) * (total_cost / cloth_length - price_reduction) :=
by sorry

end NUMINAMATH_CALUDE_cloth_length_proof_l2278_227848


namespace NUMINAMATH_CALUDE_smallest_base_for_62_l2278_227852

theorem smallest_base_for_62 : 
  ∃ (b : ℕ), b = 4 ∧ 
  (∀ (x : ℕ), x < b → ¬(b^2 ≤ 62 ∧ 62 < b^3)) ∧
  (b^2 ≤ 62 ∧ 62 < b^3) := by
sorry

end NUMINAMATH_CALUDE_smallest_base_for_62_l2278_227852


namespace NUMINAMATH_CALUDE_two_digit_perfect_square_conditions_l2278_227814

theorem two_digit_perfect_square_conditions : ∃! n : ℕ, 
  10 ≤ n ∧ n ≤ 99 ∧ 
  (∃ m : ℕ, 2 * n + 1 = m * m) ∧ 
  (∃ k : ℕ, 3 * n + 1 = k * k) ∧ 
  n = 40 := by
sorry

end NUMINAMATH_CALUDE_two_digit_perfect_square_conditions_l2278_227814


namespace NUMINAMATH_CALUDE_triangle_translation_l2278_227831

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation2D where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (t : Translation2D) (p : Point2D) : Point2D :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem triangle_translation :
  let A : Point2D := { x := 0, y := 2 }
  let B : Point2D := { x := 2, y := -1 }
  let A' : Point2D := { x := -1, y := 0 }
  let t : Translation2D := { dx := A'.x - A.x, dy := A'.y - A.y }
  let B' : Point2D := applyTranslation t B
  B'.x = 1 ∧ B'.y = -3 := by sorry

end NUMINAMATH_CALUDE_triangle_translation_l2278_227831


namespace NUMINAMATH_CALUDE_wood_length_problem_l2278_227874

theorem wood_length_problem (first_set second_set : ℝ) :
  second_set = 5 * first_set →
  second_set = 20 →
  first_set = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_wood_length_problem_l2278_227874


namespace NUMINAMATH_CALUDE_sine_product_identity_l2278_227844

theorem sine_product_identity : 
  (Real.sin (10 * π / 180)) * (Real.sin (30 * π / 180)) * 
  (Real.sin (50 * π / 180)) * (Real.sin (70 * π / 180)) = 1/16 := by
sorry

end NUMINAMATH_CALUDE_sine_product_identity_l2278_227844


namespace NUMINAMATH_CALUDE_book_price_l2278_227867

theorem book_price (price : ℝ) : price = 1 + (1/3) * price → price = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_book_price_l2278_227867


namespace NUMINAMATH_CALUDE_hypotenuse_length_hypotenuse_is_four_l2278_227859

-- Define a right triangle with one angle of 15 degrees and altitude to hypotenuse of 1 cm
structure RightTriangle where
  -- One angle is 15 degrees (π/12 radians)
  angle : Real
  angle_eq : angle = Real.pi / 12
  -- The altitude to the hypotenuse is 1 cm
  altitude : Real
  altitude_eq : altitude = 1
  -- It's a right triangle (one angle is 90 degrees)
  is_right : Bool
  is_right_eq : is_right = true

-- Theorem: The hypotenuse of this triangle is 4 cm
theorem hypotenuse_length (t : RightTriangle) : Real :=
  4

-- The proof
theorem hypotenuse_is_four (t : RightTriangle) : hypotenuse_length t = 4 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_hypotenuse_is_four_l2278_227859


namespace NUMINAMATH_CALUDE_proposition_analysis_l2278_227807

-- Define the propositions p and q
def p : Prop := ∀ x y : ℝ, x > y → -x < -y
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Theorem statement
theorem proposition_analysis :
  p ∧ 
  ¬q ∧ 
  (p ∨ q) ∧ 
  (p ∧ ¬q) ∧ 
  ¬(p ∧ q) ∧ 
  ¬(¬p ∨ q) :=
sorry

end NUMINAMATH_CALUDE_proposition_analysis_l2278_227807


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2278_227824

theorem fraction_equation_solution : 
  let x : ℚ := 24
  (1 : ℚ) / 3 + (1 : ℚ) / 2 + (1 : ℚ) / x = (7 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2278_227824


namespace NUMINAMATH_CALUDE_race_time_A_l2278_227882

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  runnerA : Runner
  runnerB : Runner
  timeDiff : ℝ
  distanceDiff : ℝ

/-- The main theorem that proves the race time for runner A -/
theorem race_time_A (race : Race) (h1 : race.distance = 1000) 
    (h2 : race.timeDiff = 10) (h3 : race.distanceDiff = 25) : 
    race.distance / race.runnerA.speed = 390 := by
  sorry

#check race_time_A

end NUMINAMATH_CALUDE_race_time_A_l2278_227882


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2278_227830

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_with_complement : P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2278_227830


namespace NUMINAMATH_CALUDE_tom_younger_than_bob_by_three_l2278_227858

/-- Represents the ages of four siblings -/
structure SiblingAges where
  susan : ℕ
  arthur : ℕ
  tom : ℕ
  bob : ℕ

/-- The age difference between Bob and Tom -/
def ageDifference (ages : SiblingAges) : ℕ :=
  ages.bob - ages.tom

theorem tom_younger_than_bob_by_three (ages : SiblingAges) 
  (susan_age : ages.susan = 15)
  (arthur_age : ages.arthur = ages.susan + 2)
  (bob_age : ages.bob = 11)
  (total_age : ages.susan + ages.arthur + ages.tom + ages.bob = 51) :
  ageDifference ages = 3 := by
sorry

end NUMINAMATH_CALUDE_tom_younger_than_bob_by_three_l2278_227858


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_l2278_227812

theorem smallest_four_digit_multiple : ∃ (n : ℕ), 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  (2 ∣ n) ∧ (3 ∣ n) ∧ (8 ∣ n) ∧ (9 ∣ n) ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ (2 ∣ m) ∧ (3 ∣ m) ∧ (8 ∣ m) ∧ (9 ∣ m) → m ≥ n) ∧
  n = 1008 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_l2278_227812


namespace NUMINAMATH_CALUDE_maria_sheets_problem_l2278_227845

/-- The number of sheets in Maria's desk -/
def sheets_in_desk : ℕ := sorry

/-- The number of sheets in Maria's backpack -/
def sheets_in_backpack : ℕ := sorry

/-- The total number of sheets Maria has -/
def total_sheets : ℕ := 91

theorem maria_sheets_problem :
  (sheets_in_backpack = sheets_in_desk + 41) →
  (total_sheets = sheets_in_desk + sheets_in_backpack) →
  sheets_in_desk = 25 := by sorry

end NUMINAMATH_CALUDE_maria_sheets_problem_l2278_227845


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2278_227815

/-- Given a geometric sequence {a_n} with positive terms and common ratio q,
    if 3a_1, (1/2)a_3, and 2a_2 form an arithmetic sequence, then q = 3. -/
theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence with ratio q
  (3 * a 1 - (1/2) * a 3) = ((1/2) * a 3 - 2 * a 2) →  -- Arithmetic sequence condition
  q = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2278_227815


namespace NUMINAMATH_CALUDE_tree_increase_l2278_227847

theorem tree_increase (initial_trees : ℕ) (increase_percentage : ℚ) : 
  initial_trees = 120 →
  increase_percentage = 5.5 / 100 →
  initial_trees + ⌊(increase_percentage * initial_trees : ℚ)⌋ = 126 := by
sorry

end NUMINAMATH_CALUDE_tree_increase_l2278_227847


namespace NUMINAMATH_CALUDE_uncovered_area_of_rectangles_l2278_227865

theorem uncovered_area_of_rectangles (small_length small_width large_length large_width : ℝ) 
  (h1 : small_length = 4)
  (h2 : small_width = 2)
  (h3 : large_length = 10)
  (h4 : large_width = 6)
  (h5 : small_length ≤ large_length)
  (h6 : small_width ≤ large_width) :
  large_length * large_width - small_length * small_width = 52 := by
sorry

end NUMINAMATH_CALUDE_uncovered_area_of_rectangles_l2278_227865


namespace NUMINAMATH_CALUDE_f_geq_a_implies_a_leq_2_l2278_227866

/-- The function f(x) = x^2 - ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

/-- The theorem stating that if f(x) ≥ a for all x ∈ [-1, +∞), then a ≤ 2 -/
theorem f_geq_a_implies_a_leq_2 (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → f a x ≥ a) → a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_f_geq_a_implies_a_leq_2_l2278_227866


namespace NUMINAMATH_CALUDE_distance_between_points_l2278_227822

def point1 : ℝ × ℝ := (0, 3)
def point2 : ℝ × ℝ := (4, -5)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2278_227822


namespace NUMINAMATH_CALUDE_range_of_a_l2278_227804

-- Define the function representing |x-2|+|x+3|
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Define the condition that the solution set is ℝ
def solution_set_is_reals (a : ℝ) : Prop :=
  ∀ x : ℝ, f x ≥ a

-- Theorem statement
theorem range_of_a (a : ℝ) :
  solution_set_is_reals a ↔ a ∈ Set.Iic 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2278_227804


namespace NUMINAMATH_CALUDE_average_of_eleven_numbers_l2278_227868

theorem average_of_eleven_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (sixth_number : ℝ) :
  first_six_avg = 88 →
  last_six_avg = 65 →
  sixth_number = 258 →
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / 11 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_of_eleven_numbers_l2278_227868


namespace NUMINAMATH_CALUDE_stamp_solution_l2278_227851

/-- Represents the stamp collection and sale problem --/
def stamp_problem (red_count blue_count : ℕ) (red_price blue_price yellow_price : ℚ) (total_goal : ℚ) : Prop :=
  let red_earnings := red_count * red_price
  let blue_earnings := blue_count * blue_price
  let remaining_earnings := total_goal - (red_earnings + blue_earnings)
  ∃ yellow_count : ℕ, yellow_count * yellow_price = remaining_earnings

/-- Theorem stating the solution to the stamp problem --/
theorem stamp_solution :
  stamp_problem 20 80 1.1 0.8 2 100 → ∃ yellow_count : ℕ, yellow_count = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_stamp_solution_l2278_227851


namespace NUMINAMATH_CALUDE_ellipse_equation_correct_l2278_227895

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if two points are foci of an ellipse -/
def areFoci (f1 f2 : Point) (e : Ellipse) : Prop :=
  (f2.x - f1.x)^2 / 4 = e.a^2 - e.b^2

theorem ellipse_equation_correct (P A B : Point) (E : Ellipse) :
  P.x = 5/2 ∧ P.y = -3/2 ∧
  A.x = -2 ∧ A.y = 0 ∧
  B.x = 2 ∧ B.y = 0 ∧
  E.a^2 = 10 ∧ E.b^2 = 6 →
  pointOnEllipse P E ∧ areFoci A B E := by
  sorry

#check ellipse_equation_correct

end NUMINAMATH_CALUDE_ellipse_equation_correct_l2278_227895


namespace NUMINAMATH_CALUDE_product_plus_one_is_square_l2278_227863

theorem product_plus_one_is_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_is_square_l2278_227863


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2278_227833

theorem simplify_trig_expression (x : ℝ) :
  (Real.sin x + Real.sin (3 * x)) / (1 + Real.cos x + Real.cos (3 * x)) =
  4 * (Real.sin x - Real.sin x ^ 3) / (1 - 2 * Real.cos x + 4 * Real.cos x ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2278_227833


namespace NUMINAMATH_CALUDE_opposite_pairs_l2278_227842

theorem opposite_pairs :
  (∀ x : ℝ, -|x| = -x ∧ -(-x) = x) ∧
  (-|-3| = -(-(-3))) ∧
  (3 ≠ -|-3|) ∧
  (-3 ≠ -(-1/3)) ∧
  (-3 ≠ -(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_opposite_pairs_l2278_227842


namespace NUMINAMATH_CALUDE_train_length_calculation_l2278_227839

/-- Represents the speed of the train in km/hr -/
def train_speed : ℝ := 180

/-- Represents the time taken to cross the platform in minutes -/
def crossing_time : ℝ := 1

/-- Theorem stating that under given conditions, the train length is 1500 meters -/
theorem train_length_calculation (train_length platform_length : ℝ) 
  (h1 : train_length = platform_length) 
  (h2 : train_speed * (1000 / 60) * crossing_time = 2 * train_length) : 
  train_length = 1500 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2278_227839


namespace NUMINAMATH_CALUDE_bob_school_year_hours_l2278_227893

/-- Calculates the hours per week Bob needs to work during the school year --/
def school_year_hours_per_week (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_earnings : ℕ) : ℕ :=
  let hourly_wage := summer_earnings / (summer_weeks * summer_hours_per_week)
  let total_hours_needed := school_year_earnings / hourly_wage
  total_hours_needed / school_year_weeks

/-- Theorem stating that Bob needs to work 15 hours per week during the school year --/
theorem bob_school_year_hours : 
  school_year_hours_per_week 8 45 3600 24 3600 = 15 := by sorry

end NUMINAMATH_CALUDE_bob_school_year_hours_l2278_227893


namespace NUMINAMATH_CALUDE_x_lt_2_necessary_not_sufficient_for_x_lt_0_l2278_227819

theorem x_lt_2_necessary_not_sufficient_for_x_lt_0 :
  (∀ x : ℝ, x < 0 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_x_lt_2_necessary_not_sufficient_for_x_lt_0_l2278_227819


namespace NUMINAMATH_CALUDE_fundraising_total_l2278_227871

def total_donations (initial_donors : ℕ) (initial_average : ℕ) (days : ℕ) : ℕ :=
  let donor_counts := List.range days |>.map (fun i => initial_donors * 2^i)
  let daily_averages := List.range days |>.map (fun i => initial_average + 5 * i)
  (List.zip donor_counts daily_averages).map (fun (d, a) => d * a) |>.sum

theorem fundraising_total :
  total_donations 10 10 5 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_total_l2278_227871


namespace NUMINAMATH_CALUDE_equal_output_day_l2278_227864

def initial_output_A : ℝ := 200
def daily_output_A : ℝ := 20
def daily_output_B : ℝ := 30

def total_output_A (days : ℝ) : ℝ := initial_output_A + daily_output_A * days
def total_output_B (days : ℝ) : ℝ := daily_output_B * days

theorem equal_output_day : 
  ∃ (day : ℝ), day > 0 ∧ total_output_A day = total_output_B day ∧ day = 20 :=
sorry

end NUMINAMATH_CALUDE_equal_output_day_l2278_227864


namespace NUMINAMATH_CALUDE_tangent_slope_at_x_one_l2278_227881

noncomputable def f (x : ℝ) := x^2 / 4 - Real.log x + 1

theorem tangent_slope_at_x_one (x : ℝ) (h : x > 0) :
  (deriv f x = -1/2) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_x_one_l2278_227881


namespace NUMINAMATH_CALUDE_fred_remaining_cards_l2278_227857

/-- Given that Fred initially had 40 baseball cards and Keith bought 22 of them,
    prove that Fred now has 18 baseball cards. -/
theorem fred_remaining_cards (initial_cards : ℕ) (cards_bought : ℕ) (h1 : initial_cards = 40) (h2 : cards_bought = 22) :
  initial_cards - cards_bought = 18 := by
  sorry

end NUMINAMATH_CALUDE_fred_remaining_cards_l2278_227857


namespace NUMINAMATH_CALUDE_base7_to_base10_5326_l2278_227878

def base7ToBase10 (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

theorem base7_to_base10_5326 : base7ToBase10 5 3 2 6 = 1882 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_5326_l2278_227878


namespace NUMINAMATH_CALUDE_hexagon_diagonal_theorem_l2278_227811

/-- A convex hexagon in a 2D plane -/
structure ConvexHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_convex : sorry

/-- The area of a convex hexagon -/
def area (h : ConvexHexagon) : ℝ := sorry

/-- A diagonal of a convex hexagon -/
def diagonal (h : ConvexHexagon) (i j : Fin 6) : ℝ × ℝ → ℝ × ℝ := sorry

/-- The area of a triangle formed by a diagonal and two adjacent vertices -/
def triangle_area (h : ConvexHexagon) (i j : Fin 6) : ℝ := sorry

/-- Main theorem: In any convex hexagon, there exists a diagonal that separates
    a triangle with area no more than 1/6 of the hexagon's area -/
theorem hexagon_diagonal_theorem (h : ConvexHexagon) :
  ∃ (i j : Fin 6), triangle_area h i j ≤ (1 / 6) * area h := by sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_theorem_l2278_227811


namespace NUMINAMATH_CALUDE_line_properties_l2278_227820

/-- Point type representing a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing a parametric line -/
structure Line where
  p : Point
  α : ℝ

/-- Function to calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Function to get the x-intercept of a line -/
def xIntercept (l : Line) : Point := sorry

/-- Function to get the y-intercept of a line -/
def yIntercept (l : Line) : Point := sorry

/-- Function to convert a line to its polar form -/
def toPolarForm (l : Line) : ℝ → ℝ := sorry

theorem line_properties (P : Point) (l : Line) (h1 : P.x = 2 ∧ P.y = 1) 
    (h2 : l.p = P) 
    (h3 : ∀ t : ℝ, ∃ x y : ℝ, x = 2 + t * Real.cos l.α ∧ y = 1 + t * Real.sin l.α)
    (h4 : distance P (xIntercept l) * distance P (yIntercept l) = 4) :
  l.α = 3 * Real.pi / 4 ∧ 
  ∀ θ : ℝ, toPolarForm l θ * (Real.cos θ + Real.sin θ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_properties_l2278_227820


namespace NUMINAMATH_CALUDE_expression_bounds_l2278_227877

theorem expression_bounds (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) 
    (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) : 
  4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ∧
  Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l2278_227877


namespace NUMINAMATH_CALUDE_new_person_weight_l2278_227836

theorem new_person_weight (n : ℕ) (initial_weight replaced_weight avg_increase : ℝ) :
  n = 8 ∧ 
  replaced_weight = 65 ∧ 
  avg_increase = 2.5 →
  replaced_weight + n * avg_increase = 85 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2278_227836


namespace NUMINAMATH_CALUDE_jake_sausage_cost_l2278_227849

/-- Calculates the total cost of sausages given the weight per package, number of packages, and price per pound -/
def total_cost (weight_per_package : ℕ) (num_packages : ℕ) (price_per_pound : ℕ) : ℕ :=
  weight_per_package * num_packages * price_per_pound

/-- Theorem: The total cost of Jake's sausage purchase is $24 -/
theorem jake_sausage_cost : total_cost 2 3 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_jake_sausage_cost_l2278_227849


namespace NUMINAMATH_CALUDE_sixth_piggy_bank_coins_l2278_227816

def coin_sequence (n : ℕ) : ℕ := 72 + 9 * (n - 1)

theorem sixth_piggy_bank_coins :
  coin_sequence 6 = 117 := by
  sorry

end NUMINAMATH_CALUDE_sixth_piggy_bank_coins_l2278_227816


namespace NUMINAMATH_CALUDE_x_gt_y_necessary_not_sufficient_l2278_227896

theorem x_gt_y_necessary_not_sufficient (x y : ℝ) (hx : x > 0) :
  (∀ y, x > |y| → x > y) ∧ 
  (∃ y, x > y ∧ ¬(x > |y|)) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_y_necessary_not_sufficient_l2278_227896


namespace NUMINAMATH_CALUDE_radical_sum_product_l2278_227828

theorem radical_sum_product (x y : ℝ) : 
  (x + Real.sqrt y) + (x - Real.sqrt y) = 6 →
  (x + Real.sqrt y) * (x - Real.sqrt y) = 4 →
  x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_radical_sum_product_l2278_227828


namespace NUMINAMATH_CALUDE_cannot_be_equation_l2278_227800

-- Define the linear function
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the condition that the function passes through (-1, -3)
def passes_through_point (k b : ℝ) : Prop :=
  linear_function k b (-1) = -3

-- Define the condition that the distances from intercepts to origin are equal
def equal_intercept_distances (k b : ℝ) : Prop :=
  abs (b / k) = abs b

-- Theorem statement
theorem cannot_be_equation (k b : ℝ) 
  (h1 : passes_through_point k b) 
  (h2 : equal_intercept_distances k b) :
  ¬(k = -3 ∧ b = -6) :=
sorry

end NUMINAMATH_CALUDE_cannot_be_equation_l2278_227800


namespace NUMINAMATH_CALUDE_betting_game_result_l2278_227856

theorem betting_game_result (initial_amount : ℚ) (num_bets num_wins num_losses : ℕ) 
  (h1 : initial_amount = 64)
  (h2 : num_bets = 6)
  (h3 : num_wins = 3)
  (h4 : num_losses = 3)
  (h5 : num_wins + num_losses = num_bets) :
  let final_amount := initial_amount * (3/2)^num_wins * (1/2)^num_losses
  final_amount = 27 ∧ initial_amount - final_amount = 37 := by
  sorry

#eval (64 : ℚ) * (3/2)^3 * (1/2)^3

end NUMINAMATH_CALUDE_betting_game_result_l2278_227856


namespace NUMINAMATH_CALUDE_sum_bound_l2278_227826

theorem sum_bound (k a b c : ℝ) (h1 : k > 1) (h2 : a ≥ 0) (h3 : b ≥ 0) (h4 : c ≥ 0)
  (h5 : a ≤ k * c) (h6 : b ≤ k * c) (h7 : a * b ≤ c^2) :
  a + b ≤ (k + 1/k) * c := by
  sorry

end NUMINAMATH_CALUDE_sum_bound_l2278_227826


namespace NUMINAMATH_CALUDE_cubic_sum_l2278_227810

theorem cubic_sum (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (h : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_l2278_227810


namespace NUMINAMATH_CALUDE_circle_ratio_l2278_227898

theorem circle_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) 
  (h_area : π * r₂^2 - π * r₁^2 = 4 * (π * r₁^2)) :
  r₁ / r₂ = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l2278_227898


namespace NUMINAMATH_CALUDE_roots_squared_relation_l2278_227825

def f (x : ℝ) : ℝ := 2 * x^3 - x^2 + 4 * x - 3

def g (b c d x : ℝ) : ℝ := x^3 + b * x^2 + c * x + d

theorem roots_squared_relation (b c d : ℝ) :
  (∀ r : ℝ, f r = 0 → g b c d (r^2) = 0) →
  b = 15/4 ∧ c = 5/2 ∧ d = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_roots_squared_relation_l2278_227825


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2278_227890

/-- Proves that in a class of 27 students with 15 girls, the ratio of boys to girls is 4:5 -/
theorem boys_to_girls_ratio (total_students : Nat) (girls : Nat) 
  (h1 : total_students = 27) 
  (h2 : girls = 15) : 
  (total_students - girls) * 5 = girls * 4 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2278_227890


namespace NUMINAMATH_CALUDE_nice_number_characterization_l2278_227843

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_digit (n : ℕ) : ℕ := n / 1000
def second_digit (n : ℕ) : ℕ := (n / 100) % 10
def third_digit (n : ℕ) : ℕ := (n / 10) % 10
def fourth_digit (n : ℕ) : ℕ := n % 10

def digit_product (n : ℕ) : ℕ :=
  (first_digit n) * (second_digit n) * (third_digit n) * (fourth_digit n)

def is_nice (n : ℕ) : Prop :=
  is_four_digit n ∧
  first_digit n = third_digit n ∧
  second_digit n = fourth_digit n ∧
  (n * n) % (digit_product n) = 0

def nice_numbers : List ℕ := [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1212, 2424, 3636, 4848, 1515]

theorem nice_number_characterization (n : ℕ) :
  is_nice n ↔ n ∈ nice_numbers := by sorry

end NUMINAMATH_CALUDE_nice_number_characterization_l2278_227843


namespace NUMINAMATH_CALUDE_min_lcm_a_c_l2278_227823

theorem min_lcm_a_c (a b c : ℕ+) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 21) :
  ∃ (a' c' : ℕ+), Nat.lcm a' c' = 420 ∧ ∀ (x y : ℕ+), Nat.lcm x b = 20 → Nat.lcm b y = 21 → Nat.lcm a' c' ≤ Nat.lcm x y :=
sorry

end NUMINAMATH_CALUDE_min_lcm_a_c_l2278_227823


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l2278_227888

-- Define the quadrilateral ABCD and point P
structure Quadrilateral :=
  (A B C D P : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def area (q : Quadrilateral) : ℝ := sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

def diagonals_orthogonal (q : Quadrilateral) : Prop := sorry

def perimeter (q : Quadrilateral) : ℝ := sorry

-- State the theorem
theorem quadrilateral_perimeter 
  (q : Quadrilateral)
  (h_convex : is_convex q)
  (h_area : area q = 2601)
  (h_PA : distance q.P q.A = 25)
  (h_PB : distance q.P q.B = 35)
  (h_PC : distance q.P q.C = 30)
  (h_PD : distance q.P q.D = 50)
  (h_ortho : diagonals_orthogonal q) :
  perimeter q = Real.sqrt 1850 + Real.sqrt 2125 + Real.sqrt 3400 + Real.sqrt 3125 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l2278_227888


namespace NUMINAMATH_CALUDE_james_total_matches_l2278_227886

/-- The number of boxes in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of boxes James has -/
def james_dozens : ℕ := 5

/-- The number of matches in each box -/
def matches_per_box : ℕ := 20

/-- Theorem: James has 1200 matches in total -/
theorem james_total_matches : james_dozens * dozen * matches_per_box = 1200 := by
  sorry

end NUMINAMATH_CALUDE_james_total_matches_l2278_227886


namespace NUMINAMATH_CALUDE_exists_non_prime_combination_l2278_227838

-- Define a function to check if a number is prime
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number has distinct digits (excluding 7)
def hasDistinctDigitsNo7 (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = 9 ∧ 
  7 ∉ digits ∧
  digits.toFinset.card = 9

-- Define a function to check if any three-digit combination is prime
def anyThreeDigitsPrime (n : Nat) : Prop :=
  ∀ i j k, 0 ≤ i ∧ i < j ∧ j < k ∧ k < 9 →
    isPrime (100 * (n.digits 10).get ⟨i, by sorry⟩ + 
             10 * (n.digits 10).get ⟨j, by sorry⟩ + 
             (n.digits 10).get ⟨k, by sorry⟩)

-- The main theorem
theorem exists_non_prime_combination :
  ∃ n : Nat, hasDistinctDigitsNo7 n ∧ ¬(anyThreeDigitsPrime n) :=
sorry

end NUMINAMATH_CALUDE_exists_non_prime_combination_l2278_227838


namespace NUMINAMATH_CALUDE_smallest_value_l2278_227803

theorem smallest_value (y : ℝ) (h1 : 0 < y) (h2 : y < 1) :
  y^3 < 3*y ∧ y^3 < y^(1/2) ∧ y^3 < 1/y ∧ y^3 < Real.exp y := by
  sorry

#check smallest_value

end NUMINAMATH_CALUDE_smallest_value_l2278_227803


namespace NUMINAMATH_CALUDE_unique_modular_equivalence_l2278_227884

theorem unique_modular_equivalence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_equivalence_l2278_227884


namespace NUMINAMATH_CALUDE_expected_squares_under_attack_l2278_227876

/-- The number of squares on a chessboard -/
def board_size : ℕ := 64

/-- The number of rooks placed on the board -/
def num_rooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook -/
def prob_not_attacked_by_one : ℚ := 49 / 64

/-- The expected number of squares under attack by three randomly placed rooks on a chessboard -/
theorem expected_squares_under_attack :
  let prob_attacked := 1 - prob_not_attacked_by_one ^ num_rooks
  (board_size : ℚ) * prob_attacked = 64 * (1 - (49/64)^3) :=
sorry

end NUMINAMATH_CALUDE_expected_squares_under_attack_l2278_227876


namespace NUMINAMATH_CALUDE_M_remainder_l2278_227809

/-- Number of red flags -/
def red_flags : ℕ := 13

/-- Number of yellow flags -/
def yellow_flags : ℕ := 12

/-- Total number of flags -/
def total_flags : ℕ := red_flags + yellow_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Function to calculate the number of distinguishable arrangements -/
noncomputable def M : ℕ := sorry

/-- Theorem stating the remainder when M is divided by 1000 -/
theorem M_remainder : M % 1000 = 188 := by sorry

end NUMINAMATH_CALUDE_M_remainder_l2278_227809


namespace NUMINAMATH_CALUDE_negation_equivalence_l2278_227813

theorem negation_equivalence (a b x : ℝ) :
  ¬(x ≠ a ∧ x ≠ b → x^2 - (a+b)*x + a*b ≠ 0) ↔ (x = a ∨ x = b → x^2 - (a+b)*x + a*b = 0) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2278_227813


namespace NUMINAMATH_CALUDE_article_pricing_theorem_l2278_227818

/-- Represents the price and profit relationship for an article -/
structure ArticlePricing where
  cost_price : ℝ
  profit_price : ℝ
  loss_price : ℝ
  desired_profit_price : ℝ

/-- The main theorem about the article pricing -/
theorem article_pricing_theorem (a : ArticlePricing) 
  (h1 : a.profit_price - a.cost_price = a.cost_price - a.loss_price)
  (h2 : a.profit_price = 832)
  (h3 : a.desired_profit_price = 896) :
  a.cost_price * 1.4 = a.desired_profit_price :=
sorry

#check article_pricing_theorem

end NUMINAMATH_CALUDE_article_pricing_theorem_l2278_227818


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2278_227879

theorem child_ticket_cost (total_seats : ℕ) (adult_ticket_cost : ℚ) 
  (num_children : ℕ) (total_revenue : ℚ) :
  total_seats = 250 →
  adult_ticket_cost = 6 →
  num_children = 188 →
  total_revenue = 1124 →
  ∃ (child_ticket_cost : ℚ),
    child_ticket_cost * num_children + 
    adult_ticket_cost * (total_seats - num_children) = total_revenue ∧
    child_ticket_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2278_227879


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2278_227846

theorem necessary_not_sufficient_condition :
  ∃ (x : ℝ), x ≠ 0 ∧ ¬(|2*x + 5| ≥ 7) ∧
  ∀ (y : ℝ), |2*y + 5| ≥ 7 → y ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2278_227846


namespace NUMINAMATH_CALUDE_algebraic_identities_l2278_227853

theorem algebraic_identities (a b x : ℝ) : 
  ((3 * a * b^3)^2 = 9 * a^2 * b^6) ∧ 
  (x * x^3 + x^2 * x^2 = 2 * x^4) ∧ 
  ((12 * x^4 - 6 * x^3) / (3 * x^2) = 4 * x^2 - 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l2278_227853


namespace NUMINAMATH_CALUDE_coffee_shrink_theorem_l2278_227860

/-- Represents the shrink ray effect on volume --/
def shrinkEffect : ℝ := 0.5

/-- Number of coffee cups --/
def numCups : ℕ := 5

/-- Initial volume of coffee in each cup (in ounces) --/
def initialVolume : ℝ := 8

/-- Calculates the total volume of coffee after shrinking --/
def totalVolumeAfterShrink (shrinkEffect : ℝ) (numCups : ℕ) (initialVolume : ℝ) : ℝ :=
  (shrinkEffect * initialVolume) * numCups

/-- Theorem stating that the total volume of coffee after shrinking is 20 ounces --/
theorem coffee_shrink_theorem : 
  totalVolumeAfterShrink shrinkEffect numCups initialVolume = 20 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shrink_theorem_l2278_227860


namespace NUMINAMATH_CALUDE_storm_rainfall_calculation_l2278_227854

/-- Represents the rainfall during a storm -/
structure StormRainfall where
  first_30min : ℝ
  second_30min : ℝ
  last_hour : ℝ
  average_total : ℝ
  duration : ℝ

/-- Theorem about the rainfall during a specific storm -/
theorem storm_rainfall_calculation (storm : StormRainfall) 
  (h1 : storm.first_30min = 5)
  (h2 : storm.second_30min = storm.first_30min / 2)
  (h3 : storm.duration = 2)
  (h4 : storm.average_total = 4) :
  storm.last_hour = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_storm_rainfall_calculation_l2278_227854


namespace NUMINAMATH_CALUDE_base_conversion_568_to_octal_l2278_227808

theorem base_conversion_568_to_octal :
  (1 * 8^3 + 0 * 8^2 + 7 * 8^1 + 0 * 8^0 : ℕ) = 568 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_568_to_octal_l2278_227808


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2278_227832

theorem rectangle_perimeter (l w : ℝ) : 
  l + w = 7 → 
  2 * l + w = 9.5 → 
  2 * (l + w) = 14 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2278_227832


namespace NUMINAMATH_CALUDE_shooting_probabilities_l2278_227806

/-- Probability of shooter A hitting the target -/
def prob_A : ℝ := 0.7

/-- Probability of shooter B hitting the target -/
def prob_B : ℝ := 0.6

/-- Probability of shooter C hitting the target -/
def prob_C : ℝ := 0.5

/-- Probability that at least one person hits the target -/
def prob_at_least_one : ℝ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

/-- Probability that exactly two people hit the target -/
def prob_exactly_two : ℝ := 
  prob_A * prob_B * (1 - prob_C) + 
  prob_A * (1 - prob_B) * prob_C + 
  (1 - prob_A) * prob_B * prob_C

theorem shooting_probabilities : 
  prob_at_least_one = 0.94 ∧ prob_exactly_two = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l2278_227806


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2278_227829

-- Define the lines and circle
def line_l2 (x y : ℝ) : Prop := x + 3*y + 1 = 0
def line_l1 (x y : ℝ) : Prop := 3*x - y = 0
def circle_C (x y a : ℝ) : Prop := x^2 + y^2 - 2*a*x - 2*a*y = 1 - 2*a^2

-- Define the theorem
theorem circle_center_coordinates (a : ℝ) :
  a > 0 →
  (∃ M N : ℝ × ℝ, line_l1 M.1 M.2 ∧ line_l1 N.1 N.2 ∧ circle_C M.1 M.2 a ∧ circle_C N.1 N.2 a) →
  (∀ x y : ℝ, line_l2 x y → (∀ u v : ℝ, line_l1 u v → u*x + v*y = 0)) →
  (∃ C : ℝ × ℝ, C.1 = a ∧ C.2 = a ∧ circle_C C.1 C.2 a) →
  (∃ M N : ℝ × ℝ, line_l1 M.1 M.2 ∧ line_l1 N.1 N.2 ∧ circle_C M.1 M.2 a ∧ circle_C N.1 N.2 a ∧
    (M.1 - a) * (N.1 - a) + (M.2 - a) * (N.2 - a) = 0) →
  a = Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2278_227829


namespace NUMINAMATH_CALUDE_complex_magnitude_inequality_l2278_227873

theorem complex_magnitude_inequality (a : ℝ) : 
  let z₁ : ℂ := a + 2*Complex.I
  let z₂ : ℂ := -2 + Complex.I
  Complex.abs z₁ < Complex.abs z₂ → -1 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_inequality_l2278_227873


namespace NUMINAMATH_CALUDE_pencil_notebook_cost_l2278_227885

/-- The cost of pencils and notebooks -/
theorem pencil_notebook_cost : 
  ∀ (p n : ℝ), 
  3 * p + 4 * n = 60 →
  p + n = 15.512820512820513 →
  96 * p + 24 * n = 520 := by
sorry

end NUMINAMATH_CALUDE_pencil_notebook_cost_l2278_227885


namespace NUMINAMATH_CALUDE_video_game_points_l2278_227889

/-- The number of points earned for defeating one enemy in a video game -/
def points_per_enemy (total_enemies : ℕ) (enemies_defeated : ℕ) (total_points : ℕ) : ℚ :=
  total_points / enemies_defeated

theorem video_game_points :
  let total_enemies : ℕ := 7
  let enemies_defeated : ℕ := total_enemies - 2
  let total_points : ℕ := 40
  points_per_enemy total_enemies enemies_defeated total_points = 8 := by
  sorry

end NUMINAMATH_CALUDE_video_game_points_l2278_227889


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2278_227869

theorem pie_eating_contest (first_student_total : ℚ) (second_student_total : ℚ) 
  (third_student_first_pie : ℚ) (third_student_second_pie : ℚ) :
  first_student_total = 7/6 ∧ 
  second_student_total = 4/3 ∧ 
  third_student_first_pie = 1/2 ∧ 
  third_student_second_pie = 1/3 →
  (first_student_total - third_student_first_pie * first_student_total = 2/3) ∧
  (second_student_total - third_student_second_pie * second_student_total = 1) ∧
  (third_student_first_pie * first_student_total + third_student_second_pie * second_student_total = 5/6) :=
by sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2278_227869


namespace NUMINAMATH_CALUDE_bianca_carrots_l2278_227875

/-- The number of carrots Bianca picked on the first day -/
def first_day_carrots : ℕ := 23

/-- The number of carrots Bianca threw out after the first day -/
def thrown_out_carrots : ℕ := 10

/-- The number of carrots Bianca picked on the second day -/
def second_day_carrots : ℕ := 47

/-- The total number of carrots Bianca has at the end -/
def total_carrots : ℕ := 60

theorem bianca_carrots :
  first_day_carrots - thrown_out_carrots + second_day_carrots = total_carrots :=
by sorry

end NUMINAMATH_CALUDE_bianca_carrots_l2278_227875


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l2278_227850

/-- The line y - kx - 1 = 0 always has a common point with the ellipse x²/5 + y²/m = 1 
    for all real k if and only if m ∈ [1,5) ∪ (5,+∞) -/
theorem line_ellipse_intersection (m : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y - k*x - 1 = 0 ∧ x^2/5 + y^2/m = 1) ↔ 
  (m ∈ Set.Icc 1 5 ∪ Set.Ioi 5) ∧ m ≠ 5 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l2278_227850
