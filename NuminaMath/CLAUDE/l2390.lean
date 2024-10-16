import Mathlib

namespace NUMINAMATH_CALUDE_fraction_unchanged_l2390_239091

theorem fraction_unchanged (x y : ℝ) (h : y ≠ 0) :
  (3 * (2 * x)) / (2 * (2 * y)) = (3 * x) / (2 * y) := by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l2390_239091


namespace NUMINAMATH_CALUDE_lcm_of_4_8_9_10_l2390_239012

theorem lcm_of_4_8_9_10 : Nat.lcm 4 (Nat.lcm 8 (Nat.lcm 9 10)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_4_8_9_10_l2390_239012


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l2390_239022

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b * b = a * c

theorem arithmetic_geometric_sequence_problem (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l2390_239022


namespace NUMINAMATH_CALUDE_cone_volume_from_sector_l2390_239046

/-- The volume of a right circular cone formed by rolling up a two-third sector of a circle -/
theorem cone_volume_from_sector (r : ℝ) (h : r = 6) :
  let sector_angle : ℝ := 2 * π * (2/3)
  let base_circumference : ℝ := sector_angle * r / (2 * π)
  let base_radius : ℝ := base_circumference / (2 * π)
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let cone_volume : ℝ := (1/3) * π * base_radius^2 * cone_height
  cone_volume = (32/3) * π * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_sector_l2390_239046


namespace NUMINAMATH_CALUDE_pascal_row20_sum_l2390_239010

theorem pascal_row20_sum : Nat.choose 20 4 + Nat.choose 20 5 = 20349 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row20_sum_l2390_239010


namespace NUMINAMATH_CALUDE_box_volume_less_than_500_l2390_239020

def box_volume (x : ℕ) : ℕ := (x + 3) * (x - 3) * (x^2 + 9)

theorem box_volume_less_than_500 :
  ∀ x : ℕ, x > 0 → (box_volume x < 500 ↔ x = 4 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_box_volume_less_than_500_l2390_239020


namespace NUMINAMATH_CALUDE_contractor_absent_days_l2390_239047

/-- Proves that given the contract conditions, the number of days absent is 10 --/
theorem contractor_absent_days
  (total_days : ℕ)
  (daily_pay : ℚ)
  (daily_fine : ℚ)
  (total_received : ℚ)
  (h1 : total_days = 30)
  (h2 : daily_pay = 25)
  (h3 : daily_fine = 7.5)
  (h4 : total_received = 425)
  : ∃ (days_absent : ℕ),
    days_absent = 10 ∧
    days_absent ≤ total_days ∧
    (total_days - days_absent) * daily_pay - days_absent * daily_fine = total_received :=
by
  sorry


end NUMINAMATH_CALUDE_contractor_absent_days_l2390_239047


namespace NUMINAMATH_CALUDE_eliminate_denominators_l2390_239018

theorem eliminate_denominators (x : ℝ) : 
  (3 * x + (2 * x - 1) / 3 = 3 - (x + 1) / 2) ↔ 
  (18 * x + 2 * (2 * x - 1) = 18 - 3 * (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l2390_239018


namespace NUMINAMATH_CALUDE_connie_marble_count_l2390_239062

def marble_problem (connie_marbles juan_marbles : ℕ) : Prop :=
  juan_marbles = connie_marbles + 175 ∧ juan_marbles = 498

theorem connie_marble_count :
  ∀ connie_marbles juan_marbles,
    marble_problem connie_marbles juan_marbles →
    connie_marbles = 323 :=
by
  sorry

end NUMINAMATH_CALUDE_connie_marble_count_l2390_239062


namespace NUMINAMATH_CALUDE_dexter_total_cards_l2390_239058

/-- Calculates the total number of cards Dexter has given the following conditions:
  * Dexter filled 3 fewer plastic boxes with football cards than basketball cards
  * He filled 9 boxes with basketball cards
  * Each basketball card box has 15 cards
  * Each football card box has 20 cards
-/
def totalCards (basketballBoxes : Nat) (basketballCardsPerBox : Nat) 
               (footballCardsPerBox : Nat) (boxDifference : Nat) : Nat :=
  let basketballCards := basketballBoxes * basketballCardsPerBox
  let footballBoxes := basketballBoxes - boxDifference
  let footballCards := footballBoxes * footballCardsPerBox
  basketballCards + footballCards

/-- Theorem stating that given the problem conditions, Dexter has 255 cards in total -/
theorem dexter_total_cards : 
  totalCards 9 15 20 3 = 255 := by
  sorry

end NUMINAMATH_CALUDE_dexter_total_cards_l2390_239058


namespace NUMINAMATH_CALUDE_ellipse_slope_theorem_l2390_239008

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line BC
def line_BC (k t : ℝ) (x : ℝ) : ℝ := k * x + t

-- Define the centroid condition
def is_centroid (xA yA xB yB xC yC : ℝ) : Prop :=
  xA + xB + xC = 0 ∧ yA + yB + yC = 0

-- Define the area ratio condition
def area_ratio_condition (xB yB xC yC : ℝ) : Prop :=
  2 * (yB - line_BC (-3*Real.sqrt 3/2) (line_BC (-3*Real.sqrt 3/2) 0 xB) 0) =
    yC - line_BC (-3*Real.sqrt 3/2) (line_BC (-3*Real.sqrt 3/2) 0 xC) 0 ∨
  2 * (yB - line_BC (-Real.sqrt 3/6) (line_BC (-Real.sqrt 3/6) 0 xB) 0) =
    yC - line_BC (-Real.sqrt 3/6) (line_BC (-Real.sqrt 3/6) 0 xC) 0

theorem ellipse_slope_theorem (xA yA xB yB xC yC : ℝ) :
  is_on_ellipse xA yA →
  is_on_ellipse xB yB →
  is_on_ellipse xC yC →
  is_centroid xA yA xB yB xC yC →
  area_ratio_condition xB yB xC yC →
  (∃ (k : ℝ), k < 0 ∧ (k = -3*Real.sqrt 3/2 ∨ k = -Real.sqrt 3/6)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_slope_theorem_l2390_239008


namespace NUMINAMATH_CALUDE_angle_sum_in_special_polygon_l2390_239036

theorem angle_sum_in_special_polygon (x y : ℝ) : 
  34 + 80 + 90 + (360 - x) + (360 - y) = 540 → x + y = 144 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_special_polygon_l2390_239036


namespace NUMINAMATH_CALUDE_average_of_quadratic_solutions_l2390_239000

theorem average_of_quadratic_solutions (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 4 * x₁ + 1 = 0) → 
  (3 * x₂^2 - 4 * x₂ + 1 = 0) → 
  x₁ ≠ x₂ → 
  (x₁ + x₂) / 2 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_average_of_quadratic_solutions_l2390_239000


namespace NUMINAMATH_CALUDE_alex_jogging_speed_l2390_239016

/-- Given the jogging speeds of Eugene, Brianna, Katie, and Alex, prove that Alex jogs at 2.4 miles per hour. -/
theorem alex_jogging_speed 
  (eugene_speed : ℝ) 
  (brianna_speed : ℝ) 
  (katie_speed : ℝ) 
  (alex_speed : ℝ) 
  (h1 : eugene_speed = 5)
  (h2 : brianna_speed = 4/5 * eugene_speed)
  (h3 : katie_speed = 6/5 * brianna_speed)
  (h4 : alex_speed = 1/2 * katie_speed) :
  alex_speed = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_alex_jogging_speed_l2390_239016


namespace NUMINAMATH_CALUDE_quadratic_form_and_sum_l2390_239024

theorem quadratic_form_and_sum (x : ℝ) : ∃! (a b c : ℝ),
  (6 * x^2 + 48 * x + 300 = a * (x + b)^2 + c) ∧
  (a + b + c = 214) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_and_sum_l2390_239024


namespace NUMINAMATH_CALUDE_new_person_weight_l2390_239003

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem new_person_weight (initial_people : ℕ) (replaced_weight : ℕ) (avg_increase : ℕ) (total_weight : ℕ) :
  initial_people = 4 →
  replaced_weight = 70 →
  avg_increase = 3 →
  total_weight = 390 →
  ∃ (new_weight : ℕ), 
    is_prime new_weight ∧
    new_weight = total_weight - (initial_people * replaced_weight + initial_people * avg_increase) :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2390_239003


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2390_239045

theorem unique_solution_for_equation : 
  ∃! x y : ℕ+, x^2 - 2 * Nat.factorial y.val = 2021 ∧ x = 45 ∧ y = 2 := by
  sorry

#check unique_solution_for_equation

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2390_239045


namespace NUMINAMATH_CALUDE_fault_line_movement_l2390_239014

theorem fault_line_movement (total_movement : ℝ) (past_year_movement : ℝ) 
  (h1 : total_movement = 6.5)
  (h2 : past_year_movement = 1.25) :
  total_movement - past_year_movement = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_movement_l2390_239014


namespace NUMINAMATH_CALUDE_circle_tangency_theorem_l2390_239015

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- Represents the length of a chord as m√n/p -/
structure ChordLength where
  m : ℕ
  n : ℕ
  p : ℕ

/-- Checks if two numbers are relatively prime -/
def are_relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_prime_square (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

theorem circle_tangency_theorem (C1 C2 C3 : Circle) (chord : ChordLength) :
  are_externally_tangent C1 C2 ∧
  is_internally_tangent C1 C3 ∧
  is_internally_tangent C2 C3 ∧
  are_collinear C1.center C2.center C3.center ∧
  C1.radius = 5 ∧
  C2.radius = 13 ∧
  are_relatively_prime chord.m chord.p ∧
  not_divisible_by_prime_square chord.n →
  chord.m + chord.n + chord.p = 455 := by
  sorry


end NUMINAMATH_CALUDE_circle_tangency_theorem_l2390_239015


namespace NUMINAMATH_CALUDE_translator_team_formations_l2390_239021

/-- Represents the number of translators for each language category -/
structure TranslatorCounts where
  english : Nat
  japanese : Nat
  versatile : Nat

/-- Represents the requirements for the translation teams -/
structure TeamRequirements where
  total_selected : Nat
  english_team : Nat
  japanese_team : Nat

/-- Calculates the number of ways to form translation teams -/
def count_team_formations (counts : TranslatorCounts) (reqs : TeamRequirements) : Nat :=
  sorry

/-- The main theorem stating the number of possible team formations -/
theorem translator_team_formations :
  let counts : TranslatorCounts := ⟨5, 4, 2⟩
  let reqs : TeamRequirements := ⟨8, 4, 4⟩
  count_team_formations counts reqs = 185 := by
  sorry

end NUMINAMATH_CALUDE_translator_team_formations_l2390_239021


namespace NUMINAMATH_CALUDE_nectar_water_percentage_l2390_239061

theorem nectar_water_percentage (nectar_mass honey_mass honey_water_percentage : ℝ) :
  nectar_mass = 1.2 →
  honey_mass = 1 →
  honey_water_percentage = 0.4 →
  (nectar_mass * (nectar_mass.sqrt / nectar_mass) - honey_mass * honey_water_percentage) / (nectar_mass - honey_mass) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_nectar_water_percentage_l2390_239061


namespace NUMINAMATH_CALUDE_fifth_element_row_20_l2390_239006

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The element at position k in row n of Pascal's triangle -/
def pascal_element (n k : ℕ) : ℕ :=
  binomial n (k - 1)

/-- The fifth element in row 20 of Pascal's triangle is 4845 -/
theorem fifth_element_row_20 : pascal_element 20 5 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_20_l2390_239006


namespace NUMINAMATH_CALUDE_burger_slices_l2390_239042

theorem burger_slices (total_burgers : ℕ) (friend1_slices : ℕ) (friend2_slices : ℕ) (friend3_slices : ℕ) (friend4_slices : ℕ) (era_slices : ℕ) :
  total_burgers = 5 →
  friend1_slices = 1 →
  friend2_slices = 2 →
  friend3_slices = 3 →
  friend4_slices = 3 →
  era_slices = 1 →
  (friend1_slices + friend2_slices + friend3_slices + friend4_slices + era_slices) / total_burgers = 2 :=
by sorry

end NUMINAMATH_CALUDE_burger_slices_l2390_239042


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2390_239001

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 1

-- Define the point of tangency
def P : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_line_equation :
  let m := (deriv f) P.1  -- Slope of the tangent line
  let b := P.2 - m * P.1  -- y-intercept of the tangent line
  (∀ x y, y = m * x + b ↔ 3 * x - y - 3 = 0) ∧ 
  (f P.1 = P.2) ∧  -- The point P lies on the curve
  (∀ x, (deriv f) x = 3 * x^2) -- The derivative of f is correct
  :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2390_239001


namespace NUMINAMATH_CALUDE_perfect_square_theorem_l2390_239023

/-- A function that checks if a number is a 3-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A function that represents the 6-digit number formed by n and its successor -/
def sixDigitNumber (n : ℕ) : ℕ := 1001 * n + 1

/-- The set of valid 3-digit numbers satisfying the condition -/
def validNumbers : Set ℕ := {183, 328, 528, 715}

/-- Theorem stating that the set of 3-digit numbers n such that 1001n + 1 
    is a perfect square is exactly the set {183, 328, 528, 715} -/
theorem perfect_square_theorem :
  ∀ n : ℕ, isThreeDigit n ∧ (∃ m : ℕ, sixDigitNumber n = m ^ 2) ↔ n ∈ validNumbers := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_theorem_l2390_239023


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2390_239081

/-- Given a geometric sequence {aₙ}, prove that if a₇ × a₉ = 4 and a₄ = 1, then a₁₂ = 16 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∃ (r : ℝ), ∀ n, a (n + 1) = a n * r) →  -- {aₙ} is a geometric sequence
  a 7 * a 9 = 4 →                         -- a₇ × a₉ = 4
  a 4 = 1 →                               -- a₄ = 1
  a 12 = 16 :=                            -- a₁₂ = 16
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l2390_239081


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l2390_239074

theorem greatest_x_with_lcm (x : ℕ+) : 
  (Nat.lcm x (Nat.lcm 15 21) = 105) → x ≤ 105 ∧ ∃ y : ℕ+, y = 105 ∧ Nat.lcm y (Nat.lcm 15 21) = 105 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l2390_239074


namespace NUMINAMATH_CALUDE_p_2017_equals_14_l2390_239093

/-- Function that calculates the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ := sorry

/-- Function that calculates the number of digits of a positive integer -/
def numberOfDigits (n : ℕ+) : ℕ := sorry

/-- Function P(n) as defined in the problem -/
def P (n : ℕ+) : ℕ := sumOfDigits n + numberOfDigits n

/-- Theorem stating that P(2017) = 14 -/
theorem p_2017_equals_14 : P 2017 = 14 := by sorry

end NUMINAMATH_CALUDE_p_2017_equals_14_l2390_239093


namespace NUMINAMATH_CALUDE_box_volume_increase_l2390_239082

/-- Given a rectangular box with dimensions l, w, h satisfying certain conditions,
    prove that increasing each dimension by 2 results in a specific new volume -/
theorem box_volume_increase (l w h : ℝ) 
  (hv : l * w * h = 5184)
  (hs : 2 * (l * w + w * h + h * l) = 1944)
  (he : 4 * (l + w + h) = 216) :
  (l + 2) * (w + 2) * (h + 2) = 7352 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2390_239082


namespace NUMINAMATH_CALUDE_sue_shoe_probability_l2390_239039

/-- Represents the number of pairs of shoes of a specific color -/
structure ShoeCount where
  pairs : ℕ

/-- Represents the total shoe collection -/
structure ShoeCollection where
  black : ShoeCount
  brown : ShoeCount
  gray : ShoeCount

def sue_shoes : ShoeCollection :=
  { black := { pairs := 7 },
    brown := { pairs := 4 },
    gray  := { pairs := 3 } }

def total_shoes (sc : ShoeCollection) : ℕ :=
  2 * (sc.black.pairs + sc.brown.pairs + sc.gray.pairs)

/-- The probability of picking two shoes of the same color,
    one left and one right, from Sue's shoe collection -/
def same_color_diff_foot_prob (sc : ShoeCollection) : ℚ :=
  let total := total_shoes sc
  let prob_black := (2 * sc.black.pairs : ℚ) / total * (sc.black.pairs : ℚ) / (total - 1)
  let prob_brown := (2 * sc.brown.pairs : ℚ) / total * (sc.brown.pairs : ℚ) / (total - 1)
  let prob_gray := (2 * sc.gray.pairs : ℚ) / total * (sc.gray.pairs : ℚ) / (total - 1)
  prob_black + prob_brown + prob_gray

theorem sue_shoe_probability :
  same_color_diff_foot_prob sue_shoes = 37 / 189 := by sorry

end NUMINAMATH_CALUDE_sue_shoe_probability_l2390_239039


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2390_239088

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47)
  (eq2 : 4 * a + 3 * b = 39) :
  a + b = 82 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2390_239088


namespace NUMINAMATH_CALUDE_equation_solution_l2390_239026

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = -3 ∧ x₂ = 2/3 ∧
  (∀ x : ℝ, 3*x*(x+3) = 2*(x+3) ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2390_239026


namespace NUMINAMATH_CALUDE_elise_cab_ride_cost_l2390_239095

/-- Calculates the total cost of a cab ride -/
def cab_ride_cost (base_price : ℝ) (per_mile_cost : ℝ) (distance : ℝ) : ℝ :=
  base_price + per_mile_cost * distance

/-- Proves that Elise's cab ride cost $23 -/
theorem elise_cab_ride_cost :
  cab_ride_cost 3 4 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_elise_cab_ride_cost_l2390_239095


namespace NUMINAMATH_CALUDE_function_inequality_condition_l2390_239084

theorem function_inequality_condition (k m : ℝ) (hk : k > 0) (hm : m > 0) :
  (∀ x : ℝ, |x - 1| < m → |5 * x - 5| < k) ↔ m ≤ k / 5 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l2390_239084


namespace NUMINAMATH_CALUDE_triangular_grid_properties_l2390_239077

/-- Represents a labeled vertex in the triangular grid -/
structure LabeledVertex where
  x : ℕ
  y : ℕ
  label : ℝ

/-- Represents the triangular grid -/
structure TriangularGrid where
  n : ℕ
  vertices : List LabeledVertex
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for adjacent triangles -/
def adjacent_condition (grid : TriangularGrid) : Prop :=
  ∀ A B C D : LabeledVertex,
    A ∈ grid.vertices → B ∈ grid.vertices → C ∈ grid.vertices → D ∈ grid.vertices →
    (A.x + 1 = B.x ∧ A.y = B.y) →
    (B.x = C.x ∧ B.y + 1 = C.y) →
    (C.x - 1 = D.x ∧ C.y = D.y) →
    A.label + D.label = B.label + C.label

/-- The main theorem -/
theorem triangular_grid_properties (grid : TriangularGrid)
    (h1 : grid.vertices.length = grid.n * (grid.n + 1) / 2)
    (h2 : adjacent_condition grid) :
    (∃ v1 v2 : LabeledVertex,
      v1 ∈ grid.vertices ∧ v2 ∈ grid.vertices ∧
      (∀ v : LabeledVertex, v ∈ grid.vertices → v1.label ≤ v.label ∧ v.label ≤ v2.label) ∧
      ((v1.x - v2.x)^2 + (v1.y - v2.y)^2 : ℝ) = grid.n^2) ∧
    (grid.vertices.map (λ v : LabeledVertex => v.label)).sum =
      (grid.n + 1) * (grid.n + 2) * (grid.a + grid.b + grid.c) / 6 :=
sorry

end NUMINAMATH_CALUDE_triangular_grid_properties_l2390_239077


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2390_239004

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 5 = Real.sqrt (5^a * 5^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2390_239004


namespace NUMINAMATH_CALUDE_outfit_combinations_l2390_239069

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (ties : ℕ) (jackets : ℕ) :
  shirts = 8 →
  pants = 5 →
  ties = 4 →
  jackets = 3 →
  shirts * pants * (ties + 1) * (jackets + 1) = 800 :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2390_239069


namespace NUMINAMATH_CALUDE_shortest_player_height_l2390_239099

/-- Given the heights of four players, prove the height of the shortest player. -/
theorem shortest_player_height (T S P Q : ℝ)
  (h1 : T = 77.75)
  (h2 : T = S + 9.5)
  (h3 : P = S + 5)
  (h4 : Q = P - 3) :
  S = 68.25 := by
  sorry

end NUMINAMATH_CALUDE_shortest_player_height_l2390_239099


namespace NUMINAMATH_CALUDE_difference_max_min_all_three_l2390_239038

/-- The total number of students in the school --/
def total_students : ℕ := 1500

/-- The minimum number of students studying English --/
def min_english : ℕ := 1050

/-- The maximum number of students studying English --/
def max_english : ℕ := 1125

/-- The minimum number of students studying Spanish --/
def min_spanish : ℕ := 750

/-- The maximum number of students studying Spanish --/
def max_spanish : ℕ := 900

/-- The minimum number of students studying German --/
def min_german : ℕ := 300

/-- The maximum number of students studying German --/
def max_german : ℕ := 450

/-- The function that calculates the number of students studying all three languages --/
def students_all_three (e s g : ℕ) : ℤ :=
  e + s + g - total_students

/-- The theorem stating the difference between the maximum and minimum number of students studying all three languages --/
theorem difference_max_min_all_three :
  (max_german - (max 0 (students_all_three min_english min_spanish min_german))) = 450 :=
sorry

end NUMINAMATH_CALUDE_difference_max_min_all_three_l2390_239038


namespace NUMINAMATH_CALUDE_probability_of_double_l2390_239043

/-- The number of integers on the dominoes (from 0 to 12, inclusive) -/
def n : ℕ := 12

/-- The total number of dominoes in the set -/
def total_dominoes : ℕ := (n + 1) * (n + 2) / 2

/-- The number of doubles in the set -/
def num_doubles : ℕ := n + 1

/-- The probability of selecting a double -/
def prob_double : ℚ := num_doubles / total_dominoes

/-- Theorem stating the probability of selecting a double -/
theorem probability_of_double : prob_double = 13 / 91 := by sorry

end NUMINAMATH_CALUDE_probability_of_double_l2390_239043


namespace NUMINAMATH_CALUDE_min_colors_needed_l2390_239005

/-- Represents a color assignment for hats and ribbons --/
structure ColorAssignment (n : ℕ) where
  hatColors : Fin n → Fin n
  ribbonColors : Fin n → Fin n → Fin n

/-- A valid color assignment satisfies the problem constraints --/
def isValidColorAssignment (n : ℕ) (ca : ColorAssignment n) : Prop :=
  (∀ i j : Fin n, i ≠ j → ca.ribbonColors i j ≠ ca.hatColors i) ∧
  (∀ i j : Fin n, i ≠ j → ca.ribbonColors i j ≠ ca.hatColors j) ∧
  (∀ i j k : Fin n, i ≠ j → i ≠ k → j ≠ k → ca.ribbonColors i j ≠ ca.ribbonColors i k)

/-- The main theorem: n colors are sufficient and necessary --/
theorem min_colors_needed (n : ℕ) (h : n ≥ 2) :
  (∃ ca : ColorAssignment n, isValidColorAssignment n ca) ∧
  (∀ m : ℕ, m < n → ¬∃ ca : ColorAssignment m, isValidColorAssignment m ca) :=
sorry

end NUMINAMATH_CALUDE_min_colors_needed_l2390_239005


namespace NUMINAMATH_CALUDE_alok_mixed_veg_plates_l2390_239049

/-- Represents the order and pricing information for a restaurant bill --/
structure RestaurantBill where
  chapatis : ℕ
  rice : ℕ
  iceCream : ℕ
  chapatiPrice : ℕ
  ricePrice : ℕ
  mixedVegPrice : ℕ
  iceCreamPrice : ℕ
  totalPaid : ℕ

/-- Calculates the number of mixed vegetable plates ordered --/
def mixedVegPlates (bill : RestaurantBill) : ℕ :=
  (bill.totalPaid - (bill.chapatis * bill.chapatiPrice + bill.rice * bill.ricePrice + bill.iceCream * bill.iceCreamPrice)) / bill.mixedVegPrice

/-- Theorem stating that Alok ordered 7 plates of mixed vegetable --/
theorem alok_mixed_veg_plates :
  let bill : RestaurantBill := {
    chapatis := 16,
    rice := 5,
    iceCream := 6,
    chapatiPrice := 6,
    ricePrice := 45,
    mixedVegPrice := 70,
    iceCreamPrice := 40,
    totalPaid := 1051
  }
  mixedVegPlates bill = 7 := by
  sorry

end NUMINAMATH_CALUDE_alok_mixed_veg_plates_l2390_239049


namespace NUMINAMATH_CALUDE_binomial_sum_identity_l2390_239060

theorem binomial_sum_identity (p q n : ℕ+) :
  (∑' k : ℕ, (Nat.choose (p.val + k) p.val) * (Nat.choose (q.val + n.val - k) q.val)) =
  Nat.choose (p.val + q.val + n.val + 1) (p.val + q.val + 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_sum_identity_l2390_239060


namespace NUMINAMATH_CALUDE_divisible_by_fifteen_l2390_239089

theorem divisible_by_fifteen (n : ℤ) : 15 ∣ (7*n + 5*n^3 + 3*n^5) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_fifteen_l2390_239089


namespace NUMINAMATH_CALUDE_stating_max_coins_three_weighings_l2390_239007

/-- Represents the number of weighings available. -/
def num_weighings : ℕ := 3

/-- Represents the number of possible outcomes for each weighing. -/
def outcomes_per_weighing : ℕ := 3

/-- Calculates the total number of possible outcomes for all weighings. -/
def total_outcomes : ℕ := outcomes_per_weighing ^ num_weighings

/-- Represents the maximum number of coins that can be determined. -/
def max_coins : ℕ := 12

/-- 
Theorem stating that the maximum number of coins that can be determined
with three weighings, identifying both the counterfeit coin and whether
it's lighter or heavier, is 12.
-/
theorem max_coins_three_weighings :
  (2 * max_coins ≤ total_outcomes) ∧
  (2 * (max_coins + 1) > total_outcomes) :=
sorry

end NUMINAMATH_CALUDE_stating_max_coins_three_weighings_l2390_239007


namespace NUMINAMATH_CALUDE_value_calculation_l2390_239083

/-- If 0.5% of a value equals 65 paise, then the value is 130 rupees -/
theorem value_calculation (a : ℝ) : (0.005 * a = 65 / 100) → a = 130 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l2390_239083


namespace NUMINAMATH_CALUDE_odd_function_properties_l2390_239025

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3 * x + m) / (x^2 + 1)

theorem odd_function_properties :
  ∃ (m : ℝ),
    (∀ x, f m x = -f m (-x)) ∧
    (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f m x < f m y) ∧
    (∀ x y, 1 ≤ x ∧ x < y → f m x > f m y) ∧
    (∀ x y, 0 ≤ x ∧ 0 ≤ y → f m x - f m y ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l2390_239025


namespace NUMINAMATH_CALUDE_grocer_banana_profit_l2390_239009

/-- Represents the profit calculation for a grocer selling bananas -/
theorem grocer_banana_profit : 
  let purchase_rate : ℚ := 3 / 0.5  -- 3 pounds per $0.50
  let sell_rate : ℚ := 4 / 1        -- 4 pounds per $1.00
  let total_pounds : ℚ := 108       -- Total pounds purchased
  let cost_price := total_pounds / purchase_rate
  let sell_price := total_pounds / sell_rate
  let profit := sell_price - cost_price
  profit = 9 := by sorry

end NUMINAMATH_CALUDE_grocer_banana_profit_l2390_239009


namespace NUMINAMATH_CALUDE_tablet_cash_savings_l2390_239002

/-- Represents the cost and payment structure for a tablet purchase -/
structure TabletPurchase where
  cash_price : ℕ
  down_payment : ℕ
  first_four_months_payment : ℕ
  middle_four_months_payment : ℕ
  last_four_months_payment : ℕ

/-- Calculates the total amount paid through installments -/
def total_installment_cost (tp : TabletPurchase) : ℕ :=
  tp.down_payment + 4 * tp.first_four_months_payment + 4 * tp.middle_four_months_payment + 4 * tp.last_four_months_payment

/-- Calculates the savings when buying in cash versus installments -/
def cash_savings (tp : TabletPurchase) : ℕ :=
  total_installment_cost tp - tp.cash_price

/-- Theorem stating the savings when buying the tablet in cash -/
theorem tablet_cash_savings :
  ∃ (tp : TabletPurchase),
    tp.cash_price = 450 ∧
    tp.down_payment = 100 ∧
    tp.first_four_months_payment = 40 ∧
    tp.middle_four_months_payment = 35 ∧
    tp.last_four_months_payment = 30 ∧
    cash_savings tp = 70 := by
  sorry

end NUMINAMATH_CALUDE_tablet_cash_savings_l2390_239002


namespace NUMINAMATH_CALUDE_toy_factory_daily_production_l2390_239070

/-- A factory produces toys with the following conditions:
    - The factory produces 5500 toys per week
    - Workers work 5 days a week
    - The same number of toys is made every day -/
def ToyFactory (weekly_production : ℕ) (work_days : ℕ) (daily_production : ℕ) : Prop :=
  weekly_production = 5500 ∧ work_days = 5 ∧ daily_production * work_days = weekly_production

/-- Theorem: Given the conditions of the toy factory, the daily production is 1100 toys -/
theorem toy_factory_daily_production :
  ∀ (weekly_production work_days daily_production : ℕ),
  ToyFactory weekly_production work_days daily_production →
  daily_production = 1100 := by
  sorry

end NUMINAMATH_CALUDE_toy_factory_daily_production_l2390_239070


namespace NUMINAMATH_CALUDE_largest_unorderable_number_l2390_239034

def is_orderable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 6 * a + 9 * b + 20 * c

theorem largest_unorderable_number : 
  (∀ m > 43, is_orderable m) ∧ ¬(is_orderable 43) := by
  sorry

end NUMINAMATH_CALUDE_largest_unorderable_number_l2390_239034


namespace NUMINAMATH_CALUDE_f_properties_l2390_239033

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x > 0 then x^2 - 3
  else if x < 0 then -x^2 + 3
  else 0

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = x^2 - 3) ∧  -- given condition for x > 0
  (∀ x < 0, f x = -x^2 + 3) ∧  -- prove this for x < 0
  (f 0 = 0) ∧  -- prove f(0) = 0
  ({x : ℝ | f x = 2 * x} = {-3, 0, 3}) :=  -- prove the solution set
by sorry

end NUMINAMATH_CALUDE_f_properties_l2390_239033


namespace NUMINAMATH_CALUDE_square_of_difference_l2390_239063

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 25) :
  (5 - Real.sqrt (y^2 - 25))^2 = y^2 - 10 * Real.sqrt (y^2 - 25) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l2390_239063


namespace NUMINAMATH_CALUDE_greater_l_conference_teams_l2390_239080

/-- The number of teams in the GREATER L conference -/
def n : ℕ := sorry

/-- The total number of games played in the season -/
def total_games : ℕ := 90

/-- The formula for the total number of games when each team plays every other team twice -/
def games_formula (x : ℕ) : ℕ := x * (x - 1)

theorem greater_l_conference_teams :
  n = 10 ∧ games_formula n = total_games := by sorry

end NUMINAMATH_CALUDE_greater_l_conference_teams_l2390_239080


namespace NUMINAMATH_CALUDE_encryption_correspondence_unique_decryption_l2390_239050

/-- Encryption function that maps a plaintext to a ciphertext -/
def encrypt (p : ℕ × ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let (a, b, c, d) := p
  (a + 2*b, b + c, 2*c + 3*d, 4*d)

/-- Theorem stating that the plaintext (6, 4, 1, 7) corresponds to the ciphertext (14, 9, 23, 28) -/
theorem encryption_correspondence :
  encrypt (6, 4, 1, 7) = (14, 9, 23, 28) := by
  sorry

/-- Theorem stating that the plaintext (6, 4, 1, 7) is the unique solution -/
theorem unique_decryption :
  ∀ p : ℕ × ℕ × ℕ × ℕ, encrypt p = (14, 9, 23, 28) → p = (6, 4, 1, 7) := by
  sorry

end NUMINAMATH_CALUDE_encryption_correspondence_unique_decryption_l2390_239050


namespace NUMINAMATH_CALUDE_smallest_bob_number_l2390_239040

def alice_number : ℕ := 36

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ n → p ∣ m)

theorem smallest_bob_number :
  ∃ k : ℕ, k > 0 ∧ has_all_prime_factors alice_number k ∧
  ∀ m : ℕ, m > 0 → has_all_prime_factors alice_number m → k ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l2390_239040


namespace NUMINAMATH_CALUDE_sugar_solution_replacement_l2390_239090

/-- Calculates the sugar percentage of a final solution after replacing part of an original solution with a new solution. -/
def final_sugar_percentage (original_percentage : ℚ) (replaced_fraction : ℚ) (new_percentage : ℚ) : ℚ :=
  (1 - replaced_fraction) * original_percentage + replaced_fraction * new_percentage

/-- Theorem stating that replacing 1/4 of a 10% sugar solution with a 38% sugar solution results in a 17% sugar solution. -/
theorem sugar_solution_replacement :
  final_sugar_percentage (10 / 100) (1 / 4) (38 / 100) = 17 / 100 := by
  sorry

#eval final_sugar_percentage (10 / 100) (1 / 4) (38 / 100)

end NUMINAMATH_CALUDE_sugar_solution_replacement_l2390_239090


namespace NUMINAMATH_CALUDE_infinite_representable_elements_l2390_239065

def is_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, a (i + 1) > a i

theorem infinite_representable_elements 
  (a : ℕ → ℕ) 
  (h_increasing : is_increasing_sequence a) :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧
    ∃ (x y h k : ℕ), 
      0 < h ∧ h < k ∧ k < m ∧
      a m = x * a h + y * a k :=
sorry

end NUMINAMATH_CALUDE_infinite_representable_elements_l2390_239065


namespace NUMINAMATH_CALUDE_quadratic_equation_prime_roots_l2390_239019

theorem quadratic_equation_prime_roots (p q : ℕ) 
  (h1 : ∃ (x y : ℕ), Prime x ∧ Prime y ∧ p * x^2 - q * x + 1985 = 0 ∧ p * y^2 - q * y + 1985 = 0) :
  12 * p^2 + q = 414 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_prime_roots_l2390_239019


namespace NUMINAMATH_CALUDE_actual_quarterly_earnings_l2390_239073

/-- Calculates the actual quarterly earnings per share given the dividend paid for 400 shares -/
theorem actual_quarterly_earnings
  (expected_earnings : ℝ)
  (expected_dividend_ratio : ℝ)
  (additional_dividend_rate : ℝ)
  (additional_earnings_threshold : ℝ)
  (shares : ℕ)
  (total_dividend : ℝ)
  (h1 : expected_earnings = 0.80)
  (h2 : expected_dividend_ratio = 0.5)
  (h3 : additional_dividend_rate = 0.04)
  (h4 : additional_earnings_threshold = 0.10)
  (h5 : shares = 400)
  (h6 : total_dividend = 208) :
  ∃ (actual_earnings : ℝ), actual_earnings = 1.10 ∧
  total_dividend = shares * (expected_earnings * expected_dividend_ratio +
    (actual_earnings - expected_earnings) * (additional_dividend_rate / additional_earnings_threshold)) :=
by sorry

end NUMINAMATH_CALUDE_actual_quarterly_earnings_l2390_239073


namespace NUMINAMATH_CALUDE_tax_increase_proof_l2390_239072

/-- Calculates the tax for a given income and tax brackets -/
def calculateTax (income : ℝ) (brackets : List (ℝ × ℝ)) : ℝ :=
  sorry

/-- Calculates the total tax including additional incomes -/
def calculateTotalTax (mainIncome : ℝ) (rentalIncome : ℝ) (investmentIncome : ℝ) (selfEmploymentIncome : ℝ) (brackets : List (ℝ × ℝ)) : ℝ :=
  sorry

def oldBrackets : List (ℝ × ℝ) := [(500000, 0.20), (500000, 0.25), (0, 0.30)]
def newBrackets : List (ℝ × ℝ) := [(500000, 0.30), (500000, 0.35), (0, 0.40)]

theorem tax_increase_proof :
  let oldMainIncome : ℝ := 1000000
  let newMainIncome : ℝ := 1500000
  let rentalIncome : ℝ := 100000
  let rentalDeduction : ℝ := 0.1
  let investmentIncome : ℝ := 50000
  let investmentTaxRate : ℝ := 0.25
  let selfEmploymentIncome : ℝ := 25000
  let selfEmploymentTaxRate : ℝ := 0.15

  calculateTotalTax newMainIncome (rentalIncome * (1 - rentalDeduction)) investmentIncome selfEmploymentIncome newBrackets -
  calculateTax oldMainIncome oldBrackets +
  investmentIncome * investmentTaxRate +
  selfEmploymentIncome * selfEmploymentTaxRate = 352250 :=
by sorry


end NUMINAMATH_CALUDE_tax_increase_proof_l2390_239072


namespace NUMINAMATH_CALUDE_sum_of_coefficients_of_expanded_f_l2390_239066

-- Define the polynomial expression
def f (c : ℝ) : ℝ := 2 * (c - 2) * (c^2 + c * (4 - c))

-- Define the sum of coefficients function
def sumOfCoefficients (p : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem sum_of_coefficients_of_expanded_f :
  sumOfCoefficients f = -8 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_of_expanded_f_l2390_239066


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2390_239031

theorem least_common_multiple_first_ten : ∃ n : ℕ, n > 0 ∧ (∀ i : ℕ, i > 0 ∧ i ≤ 10 → i ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ i : ℕ, i > 0 ∧ i ≤ 10 → i ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2390_239031


namespace NUMINAMATH_CALUDE_sum_y_equals_375_l2390_239085

variable (x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ : ℝ)

-- Define the sum of x values
def sum_x : ℝ := x₁ + x₂ + x₃ + x₄ + x₅

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 0.67 * x + 54.9

-- State the theorem
theorem sum_y_equals_375 
  (h_sum_x : sum_x = 150) : 
  y₁ + y₂ + y₃ + y₄ + y₅ = 375 := by
  sorry

end NUMINAMATH_CALUDE_sum_y_equals_375_l2390_239085


namespace NUMINAMATH_CALUDE_system_solutions_l2390_239075

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x * y^2 - 2 * y + 3 * x^2 = 0
def equation2 (x y : ℝ) : Prop := y^2 + x^2 * y + 2 * x = 0

-- Define the set of solutions
def solutions : Set (ℝ × ℝ) := {(-1, 1), (-2 / Real.rpow 3 (1/3), -2 * Real.rpow 3 (1/3)), (0, 0)}

-- Theorem statement
theorem system_solutions :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l2390_239075


namespace NUMINAMATH_CALUDE_calculate_expression_l2390_239086

theorem calculate_expression : 18 - (-16) / (2^3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2390_239086


namespace NUMINAMATH_CALUDE_share_of_a_l2390_239094

theorem share_of_a (total : ℝ) (a b c : ℝ) : 
  total = 600 →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a + b + c = total →
  a = 240 := by sorry

end NUMINAMATH_CALUDE_share_of_a_l2390_239094


namespace NUMINAMATH_CALUDE_scout_troop_profit_scout_troop_profit_is_200_l2390_239028

/-- Calculate the profit for a scout troop selling candy bars -/
theorem scout_troop_profit (num_bars : ℕ) (buy_price : ℚ) (sell_price : ℚ) : ℚ :=
  let cost := (num_bars : ℚ) * buy_price / 6
  let revenue := (num_bars : ℚ) * sell_price / 3
  revenue - cost

/-- Prove that the scout troop's profit is $200 -/
theorem scout_troop_profit_is_200 :
  scout_troop_profit 1200 (3 : ℚ) (2 : ℚ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_scout_troop_profit_scout_troop_profit_is_200_l2390_239028


namespace NUMINAMATH_CALUDE_lily_bouquet_cost_l2390_239056

/-- The cost of a bouquet of lilies given the number of lilies -/
def bouquet_cost (n : ℕ) : ℚ :=
  sorry

/-- The property that the price is directly proportional to the number of lilies -/
axiom price_proportional (n m : ℕ) :
  n ≠ 0 → m ≠ 0 → bouquet_cost n / n = bouquet_cost m / m

theorem lily_bouquet_cost :
  bouquet_cost 18 = 30 →
  bouquet_cost 45 = 75 :=
by sorry

end NUMINAMATH_CALUDE_lily_bouquet_cost_l2390_239056


namespace NUMINAMATH_CALUDE_non_monotonic_values_l2390_239052

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -2 * x^2 + a * x

-- Define the interval
def interval : Set ℝ := Set.Ioo (-1) 2

-- Define the property of non-monotonicity
def is_non_monotonic (g : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∃ (x y z : ℝ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x < y ∧ y < z ∧
    ((g x < g y ∧ g y > g z) ∨ (g x > g y ∧ g y < g z))

-- State the theorem
theorem non_monotonic_values :
  {a : ℝ | is_non_monotonic (f a) interval} = {-2, 4} := by sorry

end NUMINAMATH_CALUDE_non_monotonic_values_l2390_239052


namespace NUMINAMATH_CALUDE_point_on_line_product_of_y_coords_l2390_239098

theorem point_on_line_product_of_y_coords :
  ∀ y₁ y₂ : ℝ,
  ((-3 - 7)^2 + (y₁ - (-3))^2 = 15^2) →
  ((-3 - 7)^2 + (y₂ - (-3))^2 = 15^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -116 :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_product_of_y_coords_l2390_239098


namespace NUMINAMATH_CALUDE_pet_snake_cost_l2390_239017

def initial_amount : ℕ := 73
def amount_left : ℕ := 18

theorem pet_snake_cost : initial_amount - amount_left = 55 := by sorry

end NUMINAMATH_CALUDE_pet_snake_cost_l2390_239017


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l2390_239071

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Theorem: There are no positive integers n that satisfy n + S(n) + S(S(n)) = 2105 -/
theorem no_solution_for_equation :
  ¬ ∃ (n : ℕ+), (n : ℕ) + sumOfDigits n + sumOfDigits (sumOfDigits n) = 2105 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l2390_239071


namespace NUMINAMATH_CALUDE_sum_m_n_equals_34_l2390_239011

theorem sum_m_n_equals_34 (m n : ℕ+) (p : ℚ) : 
  m + 15 < n + 5 →
  (m + (m + 5) + (m + 15) + (n + 5) + (n + 6) + (2 * n - 1)) / 6 = p →
  ((m + 15) + (n + 5)) / 2 = p →
  m + n = 34 := by sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_34_l2390_239011


namespace NUMINAMATH_CALUDE_no_linear_term_implies_a_value_l2390_239079

/-- 
Given two polynomials (y + 2a) and (5 - y), if their product does not contain 
a linear term of y, then a = 5/2.
-/
theorem no_linear_term_implies_a_value (a : ℚ) : 
  (∀ y : ℚ, ∃ k m : ℚ, (y + 2*a) * (5 - y) = k*y^2 + m) → a = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_a_value_l2390_239079


namespace NUMINAMATH_CALUDE_f_at_2_l2390_239030

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_at_2 (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8) 
  (h2 : f (-2) = 10) : f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l2390_239030


namespace NUMINAMATH_CALUDE_prom_expenses_james_prom_expenses_l2390_239076

theorem prom_expenses (num_people : ℕ) (ticket_cost dinner_cost : ℚ) 
  (tip_percentage : ℚ) (limo_hours : ℕ) (limo_cost_per_hour : ℚ) 
  (tuxedo_rental : ℚ) : ℚ :=
  let total_ticket_cost := num_people * ticket_cost
  let total_dinner_cost := num_people * dinner_cost
  let dinner_tip := total_dinner_cost * tip_percentage
  let total_limo_cost := limo_hours * limo_cost_per_hour
  total_ticket_cost + total_dinner_cost + dinner_tip + total_limo_cost + tuxedo_rental

theorem james_prom_expenses : 
  prom_expenses 4 100 120 0.3 8 80 150 = 1814 := by
  sorry

end NUMINAMATH_CALUDE_prom_expenses_james_prom_expenses_l2390_239076


namespace NUMINAMATH_CALUDE_missing_number_proof_l2390_239041

theorem missing_number_proof (x : ℝ) (y : ℝ) : 
  (12 + x + 42 + 78 + y) / 5 = 62 →
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 →
  y = 104 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2390_239041


namespace NUMINAMATH_CALUDE_writer_productivity_l2390_239051

/-- Calculates the average words per hour for a writer given total words, total hours, and break hours. -/
def averageWordsPerHour (totalWords : ℕ) (totalHours : ℕ) (breakHours : ℕ) : ℚ :=
  totalWords / (totalHours - breakHours)

/-- Theorem stating that for a writer completing 60,000 words in 100 hours with 20 hours of breaks,
    the average words per hour when actually working is 750. -/
theorem writer_productivity : averageWordsPerHour 60000 100 20 = 750 := by
  sorry

end NUMINAMATH_CALUDE_writer_productivity_l2390_239051


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2390_239044

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (h4 : a^2 + b^2 = c^2) : (a + b) / c ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2390_239044


namespace NUMINAMATH_CALUDE_mod_sum_xyz_l2390_239048

theorem mod_sum_xyz (x y z : ℕ) : 
  x < 11 → y < 11 → z < 11 → x > 0 → y > 0 → z > 0 →
  (x * y * z) % 11 = 3 →
  (7 * z) % 11 = 4 →
  (9 * y) % 11 = (5 + y) % 11 →
  (x + y + z) % 11 = 5 := by
sorry

end NUMINAMATH_CALUDE_mod_sum_xyz_l2390_239048


namespace NUMINAMATH_CALUDE_expression_1_equality_expression_2_equality_expression_3_equality_l2390_239092

-- Expression 1
theorem expression_1_equality : (-4)^2 - 6 * (4/3) + 2 * (-1)^3 / (-1/2) = 12 := by sorry

-- Expression 2
theorem expression_2_equality : -1^4 - 1/6 * |2 - (-3)^2| = -13/6 := by sorry

-- Expression 3
theorem expression_3_equality (x y : ℝ) (h : |x+2| + (y-1)^2 = 0) :
  2*(3*x^2*y + x*y^2) - 3*(2*x^2*y - x*y) - 2*x*y^2 + 1 = -5 := by sorry

end NUMINAMATH_CALUDE_expression_1_equality_expression_2_equality_expression_3_equality_l2390_239092


namespace NUMINAMATH_CALUDE_inequality_proof_l2390_239035

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a / (b * (1 + c)) + b / (c * (1 + a)) + c / (a * (1 + b)) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2390_239035


namespace NUMINAMATH_CALUDE_smallest_constant_term_l2390_239032

theorem smallest_constant_term (a b c d e : ℤ) :
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = 1 ∨ x = -3 ∨ x = 7 ∨ x = -2/5) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 → 
    (∃ a' b' c' d' : ℤ, ∀ x : ℚ, a' * x^4 + b' * x^3 + c' * x^2 + d' * x + e' = 0 ↔ 
      x = 1 ∨ x = -3 ∨ x = 7 ∨ x = -2/5) →
    e ≤ e') →
  e = 42 := by
sorry

end NUMINAMATH_CALUDE_smallest_constant_term_l2390_239032


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l2390_239097

theorem square_difference_of_sum_and_diff (x y : ℝ) 
  (h1 : x + y = 5) (h2 : x - y = 10) : x^2 - y^2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l2390_239097


namespace NUMINAMATH_CALUDE_range_of_a_l2390_239027

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Define the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = B a → a ≤ -1 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2390_239027


namespace NUMINAMATH_CALUDE_local_tax_deduction_l2390_239013

/-- Alicia's hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- Local tax rate as a decimal -/
def tax_rate : ℝ := 0.024

/-- Conversion rate from dollars to cents -/
def dollars_to_cents : ℝ := 100

theorem local_tax_deduction :
  hourly_wage * tax_rate * dollars_to_cents = 60 := by
  sorry

end NUMINAMATH_CALUDE_local_tax_deduction_l2390_239013


namespace NUMINAMATH_CALUDE_crosswalk_stripe_distance_l2390_239054

/-- Given a street with parallel curbs and a crosswalk, prove the distance between stripes. -/
theorem crosswalk_stripe_distance
  (curb_distance : ℝ)
  (curb_length : ℝ)
  (stripe_length : ℝ)
  (h_curb_distance : curb_distance = 60)
  (h_curb_length : curb_length = 20)
  (h_stripe_length : stripe_length = 75) :
  curb_length * curb_distance / stripe_length = 16 := by
  sorry

end NUMINAMATH_CALUDE_crosswalk_stripe_distance_l2390_239054


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l2390_239057

theorem fahrenheit_to_celsius (C F : ℝ) : 
  C = (4 / 7) * (F - 40) → C = 35 → F = 101.25 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l2390_239057


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l2390_239067

/-- Perimeter of a trapezoid EFGH with given properties -/
theorem trapezoid_perimeter (EF GH EG FH : ℝ) (h1 : EF = 40) (h2 : GH = 20) 
  (h3 : EG = 30) (h4 : FH = 45) : 
  EF + GH + Real.sqrt (EF ^ 2 - EG ^ 2) + Real.sqrt (FH ^ 2 - GH ^ 2) = 60 + 10 * Real.sqrt 7 + 5 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l2390_239067


namespace NUMINAMATH_CALUDE_monday_sales_proof_l2390_239059

/-- Represents the daily pastry sales for a week -/
structure WeeklySales :=
  (monday : ℕ)
  (increase_per_day : ℕ)
  (days_per_week : ℕ)

/-- Calculates the total sales for the week -/
def total_sales (s : WeeklySales) : ℕ :=
  s.days_per_week * s.monday + (s.days_per_week * (s.days_per_week - 1) * s.increase_per_day) / 2

/-- Theorem: If daily sales increase by 1 for 7 days and average 5 per day, Monday's sales were 2 -/
theorem monday_sales_proof (s : WeeklySales) 
  (h1 : s.increase_per_day = 1)
  (h2 : s.days_per_week = 7)
  (h3 : total_sales s / s.days_per_week = 5) :
  s.monday = 2 := by
  sorry


end NUMINAMATH_CALUDE_monday_sales_proof_l2390_239059


namespace NUMINAMATH_CALUDE_composite_sum_of_powers_l2390_239029

theorem composite_sum_of_powers (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ (a^1984 + b^1984 + c^1984 + d^1984 = m * n) := by
  sorry

end NUMINAMATH_CALUDE_composite_sum_of_powers_l2390_239029


namespace NUMINAMATH_CALUDE_power_of_five_cube_l2390_239078

theorem power_of_five_cube (n : ℤ) : 
  (∃ a : ℕ, n^3 - 3*n^2 + n + 2 = 5^a) ↔ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_power_of_five_cube_l2390_239078


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l2390_239064

theorem arithmetic_sequence_count : 
  let a₁ : ℝ := 2.6
  let aₙ : ℝ := 52.1
  let d : ℝ := 4.5
  let n := (aₙ - a₁) / d + 1
  n = 12 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l2390_239064


namespace NUMINAMATH_CALUDE_count_common_divisors_l2390_239055

/-- The number of positive divisors that 9240 and 10080 have in common -/
def common_divisors_count : ℕ := 32

/-- The first given number -/
def n1 : ℕ := 9240

/-- The second given number -/
def n2 : ℕ := 10080

/-- Theorem stating that the number of positive divisors that n1 and n2 have in common is equal to common_divisors_count -/
theorem count_common_divisors : 
  (Finset.filter (λ d => d ∣ n1 ∧ d ∣ n2) (Finset.range (min n1 n2 + 1))).card = common_divisors_count := by
  sorry


end NUMINAMATH_CALUDE_count_common_divisors_l2390_239055


namespace NUMINAMATH_CALUDE_reflection_distance_l2390_239037

/-- The distance between a point (3, 2) and its reflection over the y-axis is 6. -/
theorem reflection_distance : 
  let D : ℝ × ℝ := (3, 2)
  let D' : ℝ × ℝ := (-3, 2)  -- Reflection of D over y-axis
  Real.sqrt ((D'.1 - D.1)^2 + (D'.2 - D.2)^2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_reflection_distance_l2390_239037


namespace NUMINAMATH_CALUDE_modified_square_perimeter_l2390_239053

/-- The perimeter of a modified square with an isosceles right triangle repositioned -/
theorem modified_square_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 64) :
  let side_length := square_perimeter / 4
  let hypotenuse := Real.sqrt (2 * side_length ^ 2)
  square_perimeter + hypotenuse = 80 + 16 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_modified_square_perimeter_l2390_239053


namespace NUMINAMATH_CALUDE_abs_negative_2023_l2390_239068

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l2390_239068


namespace NUMINAMATH_CALUDE_license_plate_count_l2390_239096

/-- Represents the number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- Represents the number of possible letters (A-Z) -/
def letter_choices : ℕ := 26

/-- Represents the number of digits in a license plate -/
def num_digits : ℕ := 6

/-- Represents the number of adjacent letters in a license plate -/
def num_adjacent_letters : ℕ := 2

/-- Represents the number of positions for the adjacent letter pair -/
def adjacent_letter_positions : ℕ := 7

/-- Represents the number of positions for the optional letter -/
def optional_letter_positions : ℕ := 2

/-- Calculates the total number of distinct license plates -/
def total_license_plates : ℕ :=
  adjacent_letter_positions * 
  optional_letter_positions * 
  digit_choices^num_digits * 
  letter_choices^(num_adjacent_letters + 1)

/-- Theorem stating that the total number of distinct license plates is 936,520,000 -/
theorem license_plate_count : total_license_plates = 936520000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2390_239096


namespace NUMINAMATH_CALUDE_matrix_equation_l2390_239087

theorem matrix_equation (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = !![5, 2; -3, 9]) : 
  B * A = !![5, 2; -3, 9] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l2390_239087
