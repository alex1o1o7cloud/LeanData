import Mathlib

namespace NUMINAMATH_CALUDE_stack_map_front_view_l14_1468

/-- Represents a column of stacked cubes -/
def Column := List Nat

/-- Represents the top view of a stack map -/
structure StackMap :=
  (column1 : Column)
  (column2 : Column)
  (column3 : Column)

/-- Returns the maximum height of a column -/
def maxHeight (c : Column) : Nat :=
  c.foldl max 0

/-- Returns the front view of a stack map -/
def frontView (s : StackMap) : List Nat :=
  [maxHeight s.column1, maxHeight s.column2, maxHeight s.column3]

/-- The given stack map -/
def givenStackMap : StackMap :=
  { column1 := [3, 2]
  , column2 := [2, 4, 2]
  , column3 := [5, 2] }

theorem stack_map_front_view :
  frontView givenStackMap = [3, 4, 5] := by
  sorry

end NUMINAMATH_CALUDE_stack_map_front_view_l14_1468


namespace NUMINAMATH_CALUDE_sum_solution_equations_find_a_value_l14_1434

/-- Definition of a "sum solution equation" -/
def is_sum_solution_equation (a b : ℚ) : Prop :=
  (b / a) = b + a

/-- Theorem for the given equations -/
theorem sum_solution_equations :
  is_sum_solution_equation (-3) (9/4) ∧
  ¬is_sum_solution_equation (2/3) (-2/3) ∧
  ¬is_sum_solution_equation 5 (-2) :=
sorry

/-- Theorem for finding the value of a -/
theorem find_a_value (a : ℚ) :
  is_sum_solution_equation 3 (2*a - 10) → a = 11/4 :=
sorry

end NUMINAMATH_CALUDE_sum_solution_equations_find_a_value_l14_1434


namespace NUMINAMATH_CALUDE_parabola_directrix_parameter_l14_1435

/-- Given a parabola with equation x^2 = ay and directrix y = 1, prove that a = -4 -/
theorem parabola_directrix_parameter (a : ℝ) : 
  (∀ x y : ℝ, x^2 = a*y) →  -- Equation of the parabola
  (1 : ℝ) = -a/4 →          -- Equation of the directrix (y = 1 is equivalent to 1 = -a/4 for a parabola)
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_parameter_l14_1435


namespace NUMINAMATH_CALUDE_projection_of_a_onto_b_l14_1492

def a : Fin 2 → ℚ := ![1, 2]
def b : Fin 2 → ℚ := ![-2, 4]

def dot_product (v w : Fin 2 → ℚ) : ℚ :=
  (v 0) * (w 0) + (v 1) * (w 1)

def magnitude_squared (v : Fin 2 → ℚ) : ℚ :=
  dot_product v v

def scalar_mult (c : ℚ) (v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  fun i => c * (v i)

def projection (v w : Fin 2 → ℚ) : Fin 2 → ℚ :=
  scalar_mult ((dot_product v w) / (magnitude_squared w)) w

theorem projection_of_a_onto_b :
  projection a b = ![-(3/5), 6/5] := by
  sorry

end NUMINAMATH_CALUDE_projection_of_a_onto_b_l14_1492


namespace NUMINAMATH_CALUDE_jam_distribution_and_consumption_l14_1448

/-- Represents the amount of jam and consumption rate for each person -/
structure JamConsumption where
  amount : ℝ
  rate : ℝ

/-- Proves the correct distribution and consumption rates of jam for Ponchik and Syropchik -/
theorem jam_distribution_and_consumption 
  (total_jam : ℝ)
  (ponchik_hypothetical_days : ℝ)
  (syropchik_hypothetical_days : ℝ)
  (h_total : total_jam = 100)
  (h_ponchik : ponchik_hypothetical_days = 45)
  (h_syropchik : syropchik_hypothetical_days = 20)
  : ∃ (ponchik syropchik : JamConsumption),
    ponchik.amount + syropchik.amount = total_jam ∧
    ponchik.amount / ponchik.rate = syropchik.amount / syropchik.rate ∧
    syropchik.amount / ponchik_hypothetical_days = ponchik.rate ∧
    ponchik.amount / syropchik_hypothetical_days = syropchik.rate ∧
    ponchik.amount = 40 ∧
    syropchik.amount = 60 ∧
    ponchik.rate = 4/3 ∧
    syropchik.rate = 2 := by
  sorry


end NUMINAMATH_CALUDE_jam_distribution_and_consumption_l14_1448


namespace NUMINAMATH_CALUDE_equation_solution_l14_1483

theorem equation_solution (b : ℝ) (hb : b ≠ 0) :
  (0 : ℝ)^2 + 9*b^2 = (3*b - 0)^2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l14_1483


namespace NUMINAMATH_CALUDE_greatest_common_factor_of_palindromes_l14_1496

def is_three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ a b : ℕ, n = 100*a + 10*b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def is_multiple_of_three (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3 * k

def set_of_valid_palindromes : Set ℕ :=
  {n : ℕ | is_three_digit_palindrome n ∧ is_multiple_of_three n}

theorem greatest_common_factor_of_palindromes :
  ∃ g : ℕ, g > 0 ∧ 
    (∀ n ∈ set_of_valid_palindromes, g ∣ n) ∧
    (∀ d : ℕ, d > 0 → (∀ n ∈ set_of_valid_palindromes, d ∣ n) → d ≤ g) ∧
    g = 3 :=
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_of_palindromes_l14_1496


namespace NUMINAMATH_CALUDE_two_tetrahedra_in_cube_l14_1411

/-- A cube with edge length a -/
structure Cube (a : ℝ) where
  edge_length : a > 0

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- Represents the placement of a tetrahedron within a cube -/
def TetrahedronPlacement (a : ℝ) := Cube a → RegularTetrahedron a → Prop

/-- Two tetrahedra do not overlap -/
def NonOverlapping (a : ℝ) (t1 t2 : RegularTetrahedron a) : Prop := sorry

/-- Theorem stating that two non-overlapping regular tetrahedra can be inscribed in a cube -/
theorem two_tetrahedra_in_cube (a : ℝ) (h : a > 0) :
  ∃ (c : Cube a) (t1 t2 : RegularTetrahedron a) (p1 p2 : TetrahedronPlacement a),
    p1 c t1 ∧ p2 c t2 ∧ NonOverlapping a t1 t2 :=
  sorry

end NUMINAMATH_CALUDE_two_tetrahedra_in_cube_l14_1411


namespace NUMINAMATH_CALUDE_largest_two_prime_product_digit_product_l14_1454

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def digit_product (n : ℕ) : ℕ := 
  if n < 10 then n
  else (n % 10) * digit_product (n / 10)

theorem largest_two_prime_product_digit_product : 
  ∃ m d e : ℕ,
    is_prime d ∧ 
    2 ≤ d ∧ d ≤ 5 ∧
    e = d + 10 ∧
    is_prime e ∧
    m = d * e ∧
    (∀ m' d' e' : ℕ, 
      is_prime d' ∧ 
      2 ≤ d' ∧ d' ≤ 5 ∧ 
      e' = d' + 10 ∧ 
      is_prime e' ∧ 
      m' = d' * e' → 
      m' ≤ m) ∧
    digit_product m = 27 :=
sorry

end NUMINAMATH_CALUDE_largest_two_prime_product_digit_product_l14_1454


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l14_1478

/-- 
Given a rectangle with length L and width W, and a triangle with one side of the rectangle as its base 
and a vertex on the opposite side of the rectangle, the ratio of the area of the rectangle to the area 
of the triangle is 2:1.
-/
theorem rectangle_triangle_area_ratio 
  (L W : ℝ) 
  (hL : L > 0) 
  (hW : W > 0) : 
  (L * W) / ((1/2) * L * W) = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l14_1478


namespace NUMINAMATH_CALUDE_find_y_l14_1404

theorem find_y (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l14_1404


namespace NUMINAMATH_CALUDE_equation_solutions_l14_1459

def solution_set : Set (ℤ × ℤ) :=
  {(3, 2), (2, 3), (1, -1), (-1, 1), (0, -1), (-1, 0)}

def satisfies_equation (p : ℤ × ℤ) : Prop :=
  (p.1)^3 + (p.2)^3 + 1 = (p.1)^2 * (p.2)^2

theorem equation_solutions :
  ∀ (x y : ℤ), satisfies_equation (x, y) ↔ (x, y) ∈ solution_set := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l14_1459


namespace NUMINAMATH_CALUDE_gcd_204_85_l14_1419

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l14_1419


namespace NUMINAMATH_CALUDE_wendy_initial_bags_l14_1480

/-- The number of points earned per recycled bag -/
def points_per_bag : ℕ := 5

/-- The number of bags Wendy didn't recycle -/
def unrecycled_bags : ℕ := 2

/-- The total points Wendy would have earned if she recycled all bags -/
def total_possible_points : ℕ := 45

/-- The number of bags Wendy initially had -/
def initial_bags : ℕ := 11

theorem wendy_initial_bags :
  points_per_bag * (initial_bags - unrecycled_bags) = total_possible_points :=
by sorry

end NUMINAMATH_CALUDE_wendy_initial_bags_l14_1480


namespace NUMINAMATH_CALUDE_greatest_x_value_l14_1416

theorem greatest_x_value (x : ℤ) : 
  (2.13 * (10 : ℝ)^(x : ℝ) < 2100) ∧ 
  (∀ y : ℤ, y > x → 2.13 * (10 : ℝ)^(y : ℝ) ≥ 2100) → 
  x = 2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l14_1416


namespace NUMINAMATH_CALUDE_eggs_used_for_crepes_l14_1412

theorem eggs_used_for_crepes 
  (total_eggs : ℕ) 
  (eggs_left : ℕ) 
  (h1 : total_eggs = 3 * 12)
  (h2 : eggs_left = 9)
  (h3 : ∃ remaining_after_crepes : ℕ, 
    remaining_after_crepes ≤ total_eggs ∧ 
    eggs_left = remaining_after_crepes - (2 * remaining_after_crepes / 3)) :
  (total_eggs - (total_eggs - eggs_left * 3)) / total_eggs = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_eggs_used_for_crepes_l14_1412


namespace NUMINAMATH_CALUDE_john_total_skateboard_distance_l14_1409

/-- The total distance John skateboarded, given his journey to and from the park -/
def total_skateboarded_distance (initial_skate : ℕ) (walk : ℕ) : ℕ :=
  2 * initial_skate

/-- Theorem stating that John skateboarded 20 miles in total -/
theorem john_total_skateboard_distance :
  total_skateboarded_distance 10 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_total_skateboard_distance_l14_1409


namespace NUMINAMATH_CALUDE_students_taking_german_german_count_l14_1407

theorem students_taking_german (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  let students_taking_language := total - neither
  let only_french := french - both
  let only_german := students_taking_language - only_french - both
  only_german + both
  
theorem german_count :
  students_taking_german 94 41 9 40 = 22 := by sorry

end NUMINAMATH_CALUDE_students_taking_german_german_count_l14_1407


namespace NUMINAMATH_CALUDE_worm_length_difference_l14_1420

theorem worm_length_difference (long_worm short_worm : Real) 
  (h1 : long_worm = 0.8)
  (h2 : short_worm = 0.1) :
  long_worm - short_worm = 0.7 := by
sorry

end NUMINAMATH_CALUDE_worm_length_difference_l14_1420


namespace NUMINAMATH_CALUDE_handshakes_in_room_l14_1431

/-- Represents the number of handshakes in a room with specific friendship conditions -/
def number_of_handshakes (total_people : ℕ) (friends : ℕ) (strangers : ℕ) : ℕ :=
  -- Handshakes between friends and strangers
  friends * strangers +
  -- Handshakes among strangers who know no one
  (strangers - 5).choose 2 +
  -- Handshakes between strangers who know one person and those who know no one
  5 * (strangers - 5)

/-- Theorem stating the number of handshakes in the given scenario -/
theorem handshakes_in_room (total_people : ℕ) (friends : ℕ) (strangers : ℕ) 
  (h1 : total_people = 40)
  (h2 : friends = 25)
  (h3 : strangers = 15)
  (h4 : friends + strangers = total_people) :
  number_of_handshakes total_people friends strangers = 345 := by
  sorry

#eval number_of_handshakes 40 25 15

end NUMINAMATH_CALUDE_handshakes_in_room_l14_1431


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l14_1466

theorem min_value_reciprocal_sum (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h_sum : m + n = 2) :
  1/m + 1/n ≥ 2 ∧ (1/m + 1/n = 2 ↔ m = 1 ∧ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l14_1466


namespace NUMINAMATH_CALUDE_greatest_y_value_l14_1436

theorem greatest_y_value (x y : ℤ) (h : x * y + 7 * x + 2 * y = -8) :
  y ≤ -1 ∧ ∃ (x₀ y₀ : ℤ), x₀ * y₀ + 7 * x₀ + 2 * y₀ = -8 ∧ y₀ = -1 := by
  sorry

end NUMINAMATH_CALUDE_greatest_y_value_l14_1436


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_l14_1481

theorem mark_and_carolyn_money : 
  3/4 + 3/10 = 21/20 := by sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_l14_1481


namespace NUMINAMATH_CALUDE_acclimation_time_is_one_year_l14_1423

/-- Represents the time spent on different phases of PhD study -/
structure PhDTime where
  acclimation : ℝ
  basics : ℝ
  research : ℝ
  dissertation : ℝ

/-- Conditions for John's PhD timeline -/
def johnPhDConditions (t : PhDTime) : Prop :=
  t.basics = 2 ∧
  t.research = 1.75 * t.basics ∧
  t.dissertation = 0.5 * t.acclimation ∧
  t.acclimation + t.basics + t.research + t.dissertation = 7

/-- Theorem stating that under the given conditions, the acclimation time is 1 year -/
theorem acclimation_time_is_one_year (t : PhDTime) 
  (h : johnPhDConditions t) : t.acclimation = 1 := by
  sorry


end NUMINAMATH_CALUDE_acclimation_time_is_one_year_l14_1423


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l14_1442

theorem gcd_of_polynomial_and_multiple : ∀ x : ℤ, 
  18432 ∣ x → 
  Nat.gcd (Int.natAbs ((3*x+5)*(7*x+2)*(13*x+7)*(2*x+10))) (Int.natAbs x) = 28 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l14_1442


namespace NUMINAMATH_CALUDE_d_equals_25_l14_1476

theorem d_equals_25 (x : ℝ) (h : x^2 - 2*x - 5 = 0) : 
  x^4 - 2*x^3 + x^2 - 12*x - 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_d_equals_25_l14_1476


namespace NUMINAMATH_CALUDE_three_numbers_sum_l14_1406

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ascending order
  b = 10 →  -- Median is 10
  (a + b + c) / 3 = a + 20 →  -- Mean is 20 more than least
  (a + b + c) / 3 = c - 25 →  -- Mean is 25 less than greatest
  a + b + c = 45 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l14_1406


namespace NUMINAMATH_CALUDE_abraham_shower_gels_l14_1413

def shower_gel_problem (budget : ℕ) (shower_gel_cost : ℕ) (toothpaste_cost : ℕ) (detergent_cost : ℕ) (remaining : ℕ) : Prop :=
  let total_spent : ℕ := budget - remaining
  let non_gel_cost : ℕ := toothpaste_cost + detergent_cost
  let gel_cost : ℕ := total_spent - non_gel_cost
  gel_cost / shower_gel_cost = 4

theorem abraham_shower_gels :
  shower_gel_problem 60 4 3 11 30 := by
  sorry

end NUMINAMATH_CALUDE_abraham_shower_gels_l14_1413


namespace NUMINAMATH_CALUDE_trajectory_of_c_l14_1414

/-- The trajectory of point C in a triangle ABC, where A(-5, 0) and B(5, 0) are fixed points,
    and the product of slopes of AC and BC is -1/2 --/
theorem trajectory_of_c (x y : ℝ) (h : x ≠ 5 ∧ x ≠ -5) :
  (y / (x + 5)) * (y / (x - 5)) = -1/2 →
  x^2 / 25 + y^2 / (25/2) = 1 := by sorry

end NUMINAMATH_CALUDE_trajectory_of_c_l14_1414


namespace NUMINAMATH_CALUDE_smallest_divisible_by_9_l14_1446

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def insert_digit (a b d : ℕ) : ℕ := a * 10 + d * 10 + b

theorem smallest_divisible_by_9 :
  ∀ d : ℕ, d ≥ 3 →
    is_divisible_by_9 (insert_digit 761 829 d) →
    insert_digit 761 829 3 ≤ insert_digit 761 829 d :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_9_l14_1446


namespace NUMINAMATH_CALUDE_f_even_and_periodic_l14_1403

-- Define a non-constant function f on ℝ
variable (f : ℝ → ℝ)

-- Condition: f is non-constant
axiom f_non_constant : ∃ x y, f x ≠ f y

-- Condition: f(10 + x) is an even function
axiom f_10_even : ∀ x, f (10 + x) = f (10 - x)

-- Condition: f(5 - x) = f(5 + x)
axiom f_5_symmetric : ∀ x, f (5 - x) = f (5 + x)

-- Theorem to prove
theorem f_even_and_periodic :
  (∀ x, f x = f (-x)) ∧ (∃ T > 0, ∀ x, f (x + T) = f x) :=
sorry

end NUMINAMATH_CALUDE_f_even_and_periodic_l14_1403


namespace NUMINAMATH_CALUDE_total_leaves_count_l14_1428

/-- The number of basil pots planted. -/
def basil_pots : ℕ := 3

/-- The number of rosemary pots planted. -/
def rosemary_pots : ℕ := 9

/-- The number of thyme pots planted. -/
def thyme_pots : ℕ := 6

/-- The number of leaves per basil plant. -/
def basil_leaves : ℕ := 4

/-- The number of leaves per rosemary plant. -/
def rosemary_leaves : ℕ := 18

/-- The number of leaves per thyme plant. -/
def thyme_leaves : ℕ := 30

/-- The total number of leaves from all plants. -/
def total_leaves : ℕ := basil_pots * basil_leaves + rosemary_pots * rosemary_leaves + thyme_pots * thyme_leaves

theorem total_leaves_count : total_leaves = 354 := by
  sorry

end NUMINAMATH_CALUDE_total_leaves_count_l14_1428


namespace NUMINAMATH_CALUDE_total_bars_is_300_l14_1475

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 15

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 20

/-- The total number of chocolate bars in the large box -/
def total_chocolate_bars : ℕ := num_small_boxes * bars_per_small_box

/-- Theorem: The total number of chocolate bars in the large box is 300 -/
theorem total_bars_is_300 : total_chocolate_bars = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_bars_is_300_l14_1475


namespace NUMINAMATH_CALUDE_parallel_angle_theorem_l14_1410

theorem parallel_angle_theorem (α β : Real) :
  (α = 60 ∨ β = 60) →  -- One angle is 60°
  (α = β ∨ α + β = 180) →  -- Angles are either equal or supplementary (parallel sides condition)
  (α = 60 ∧ β = 60) ∨ (α = 60 ∧ β = 120) ∨ (α = 120 ∧ β = 60) :=
by sorry

end NUMINAMATH_CALUDE_parallel_angle_theorem_l14_1410


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_l14_1467

theorem sin_cos_equation_solution (x : ℝ) :
  Real.sin x ^ 6 + Real.cos x ^ 6 = 1 / 4 →
  ∃ k : ℤ, x = π / 4 + k * π ∨ x = 3 * π / 4 + k * π :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_l14_1467


namespace NUMINAMATH_CALUDE_max_rows_with_unique_letters_l14_1453

theorem max_rows_with_unique_letters : ∃ (m : ℕ),
  (∀ (n : ℕ), n > m → ¬∃ (table : Fin n → Fin 8 → Fin 4),
    ∀ (i j : Fin n), i ≠ j →
      (∃! (k : Fin 8), table i k = table j k) ∨
      (∀ (k : Fin 8), table i k ≠ table j k)) ∧
  (∃ (table : Fin m → Fin 8 → Fin 4),
    ∀ (i j : Fin m), i ≠ j →
      (∃! (k : Fin 8), table i k = table j k) ∨
      (∀ (k : Fin 8), table i k ≠ table j k)) ∧
  m = 28 :=
by sorry


end NUMINAMATH_CALUDE_max_rows_with_unique_letters_l14_1453


namespace NUMINAMATH_CALUDE_product_of_two_digit_numbers_is_8670_l14_1427

theorem product_of_two_digit_numbers_is_8670 : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8670 ∧
  8670 = 8670 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_digit_numbers_is_8670_l14_1427


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l14_1482

/-- Given a line passing through points (1, 3) and (3, 7) with equation y = mx + b, 
    the sum of m and b is equal to 3. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (3 = m * 1 + b) → (7 = m * 3 + b) → m + b = 3 := by
  sorry


end NUMINAMATH_CALUDE_line_slope_intercept_sum_l14_1482


namespace NUMINAMATH_CALUDE_max_areas_theorem_l14_1486

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk :=
  (n : ℕ)  -- number of pairs of radii

/-- The maximum number of non-overlapping areas in a divided disk -/
def max_areas (disk : DividedDisk) : ℕ :=
  4 * disk.n + 4

/-- Theorem stating the maximum number of non-overlapping areas -/
theorem max_areas_theorem (disk : DividedDisk) :
  max_areas disk = 4 * disk.n + 4 :=
sorry

end NUMINAMATH_CALUDE_max_areas_theorem_l14_1486


namespace NUMINAMATH_CALUDE_sequence_is_increasing_l14_1495

theorem sequence_is_increasing (a : ℕ → ℝ) (h : ∀ n, a (n + 1) - a n - 3 = 0) :
  ∀ n, a (n + 1) > a n :=
sorry

end NUMINAMATH_CALUDE_sequence_is_increasing_l14_1495


namespace NUMINAMATH_CALUDE_rectangle_dimension_difference_l14_1426

theorem rectangle_dimension_difference (L B D : ℝ) : 
  L - B = D →
  2 * (L + B) = 246 →
  L * B = 3650 →
  D^2 = 29729 := by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_difference_l14_1426


namespace NUMINAMATH_CALUDE_sandys_age_l14_1489

/-- Proves that Sandy's age is 42 given the conditions -/
theorem sandys_age (sandy_age molly_age : ℕ) : 
  molly_age = sandy_age + 12 → 
  sandy_age * 9 = molly_age * 7 →
  sandy_age = 42 := by
sorry

end NUMINAMATH_CALUDE_sandys_age_l14_1489


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l14_1422

/-- A regular polygon with side length 6 units and exterior angle 90 degrees has a perimeter of 24 units. -/
theorem regular_polygon_perimeter (n : ℕ) (s : ℝ) (E : ℝ) : 
  n > 0 → 
  s = 6 → 
  E = 90 → 
  E = 360 / n → 
  n * s = 24 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l14_1422


namespace NUMINAMATH_CALUDE_unique_intersection_values_l14_1469

-- Define the set of complex numbers that satisfy |z - 2| = 3|z + 2|
def S : Set ℂ := {z : ℂ | Complex.abs (z - 2) = 3 * Complex.abs (z + 2)}

-- Define a function that returns the set of intersection points between S and |z| = k
def intersection (k : ℝ) : Set ℂ := S ∩ {z : ℂ | Complex.abs z = k}

-- State the theorem
theorem unique_intersection_values :
  ∀ k : ℝ, (∃! z : ℂ, z ∈ intersection k) ↔ (k = 1 ∨ k = 4) :=
by sorry

end NUMINAMATH_CALUDE_unique_intersection_values_l14_1469


namespace NUMINAMATH_CALUDE_circle_area_after_radius_multiplication_area_of_new_circle_l14_1457

/-- Theorem: Area of a circle after radius multiplication -/
theorem circle_area_after_radius_multiplication (A : ℝ) (k : ℝ) :
  A > 0 → k > 0 → (k * (A / Real.pi).sqrt)^2 * Real.pi = k^2 * A := by
  sorry

/-- The area of a circle with radius multiplied by 5 -/
theorem area_of_new_circle (original_area : ℝ) (new_area : ℝ) :
  original_area = 30 →
  new_area = (5 * (original_area / Real.pi).sqrt)^2 * Real.pi →
  new_area = 750 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_after_radius_multiplication_area_of_new_circle_l14_1457


namespace NUMINAMATH_CALUDE_multiple_in_selection_l14_1471

theorem multiple_in_selection (S : Finset ℕ) : 
  S ⊆ Finset.range 100 → S.card = 51 → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ∃ (k : ℕ), b = k * a :=
sorry

end NUMINAMATH_CALUDE_multiple_in_selection_l14_1471


namespace NUMINAMATH_CALUDE_jellybean_difference_l14_1472

/-- The number of jellybeans each person has -/
structure JellybeanCount where
  tino : ℕ
  lee : ℕ
  arnold : ℕ

/-- The conditions of the jellybean problem -/
def jellybean_problem (j : JellybeanCount) : Prop :=
  j.tino > j.lee ∧
  j.arnold = j.lee / 2 ∧
  j.arnold = 5 ∧
  j.tino = 34

/-- The theorem stating the difference between Tino's and Lee's jellybean counts -/
theorem jellybean_difference (j : JellybeanCount) 
  (h : jellybean_problem j) : j.tino - j.lee = 24 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_difference_l14_1472


namespace NUMINAMATH_CALUDE_remainder_of_741147_div_6_l14_1494

theorem remainder_of_741147_div_6 : 741147 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_741147_div_6_l14_1494


namespace NUMINAMATH_CALUDE_four_points_plane_count_l14_1405

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a function to count the number of planes determined by four points
def countPlanesFromFourPoints (A B C D : Point3D) : Nat :=
  sorry

-- Theorem statement
theorem four_points_plane_count (A B C D : Point3D) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) : 
  countPlanesFromFourPoints A B C D = 1 ∨ countPlanesFromFourPoints A B C D = 4 :=
sorry

end NUMINAMATH_CALUDE_four_points_plane_count_l14_1405


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_geometric_sequence_l14_1443

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_mean_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_q : q = -2)
  (h_condition : a 3 * a 7 = 4 * a 4) :
  (a 8 + a 11) / 2 = -56 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_geometric_sequence_l14_1443


namespace NUMINAMATH_CALUDE_apple_pie_problem_l14_1462

def max_pies (total_apples unripe_apples apples_per_pie : ℕ) : ℕ :=
  (total_apples - unripe_apples) / apples_per_pie

theorem apple_pie_problem :
  max_pies 34 6 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_problem_l14_1462


namespace NUMINAMATH_CALUDE_orthic_similarity_condition_l14_1441

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- The orthic triangle of a given triangle -/
def orthicTriangle (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

/-- The sequence of orthic triangles starting from an initial triangle -/
def orthicSequence (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℕ → (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)
| 0 => t
| n + 1 => orthicTriangle (orthicSequence t n)

/-- Two triangles are similar -/
def areSimilar (t1 t2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

/-- The main theorem -/
theorem orthic_similarity_condition (n : ℕ) (p : RegularPolygon n) :
  (∃ (v1 v2 v3 : Fin n) (k : ℕ),
    areSimilar
      (p.vertices v1, p.vertices v2, p.vertices v3)
      (orthicSequence (p.vertices v1, p.vertices v2, p.vertices v3) k))
  ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_orthic_similarity_condition_l14_1441


namespace NUMINAMATH_CALUDE_line_direction_vector_l14_1437

/-- The direction vector of a parameterized line -/
def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

/-- The point on the line at t = 0 -/
def initial_point (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

theorem line_direction_vector :
  let line (t : ℝ) : ℝ × ℝ := 
    (7 - 25 / Real.sqrt 41 * t, 3 - 20 / Real.sqrt 41 * t)
  let y (x : ℝ) : ℝ := (4 * x - 7) / 5
  ∀ x ≤ 7, 
    let point := (x, y x)
    let distance := Real.sqrt ((x - 7)^2 + (y x - 3)^2)
    (∃ t, point = line t ∧ distance = t) →
    direction_vector line = (-25 / Real.sqrt 41, -20 / Real.sqrt 41) :=
by sorry

end NUMINAMATH_CALUDE_line_direction_vector_l14_1437


namespace NUMINAMATH_CALUDE_square_area_error_l14_1488

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * 1.05
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.1025 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l14_1488


namespace NUMINAMATH_CALUDE_remainder_problem_l14_1499

theorem remainder_problem (x : ℤ) (h : x % 82 = 5) : (x + 17) % 41 = 22 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l14_1499


namespace NUMINAMATH_CALUDE_thomas_blocks_total_l14_1473

/-- The height of Thomas' wooden block stacks -/
def ThomasBlockStacks : ℕ → ℕ
| 1 => 7  -- First stack is 7 blocks tall
| 2 => ThomasBlockStacks 1 + 3  -- Second stack is 3 blocks taller than the first
| 3 => ThomasBlockStacks 2 - 6  -- Third stack is 6 blocks shorter than the second
| 4 => ThomasBlockStacks 3 + 10  -- Fourth stack is 10 blocks taller than the third
| 5 => ThomasBlockStacks 2 * 2  -- Fifth stack has twice as many blocks as the second
| _ => 0  -- For completeness, though we only care about the first 5 stacks

/-- The total number of blocks Thomas used -/
def TotalBlocks : ℕ :=
  ThomasBlockStacks 1 + ThomasBlockStacks 2 + ThomasBlockStacks 3 + ThomasBlockStacks 4 + ThomasBlockStacks 5

theorem thomas_blocks_total : TotalBlocks = 55 := by
  sorry

end NUMINAMATH_CALUDE_thomas_blocks_total_l14_1473


namespace NUMINAMATH_CALUDE_margo_walk_l14_1490

theorem margo_walk (outbound_speed return_speed : ℝ) (total_time : ℝ) 
  (h1 : outbound_speed = 5)
  (h2 : return_speed = 3)
  (h3 : total_time = 1) :
  let distance := (outbound_speed * return_speed * total_time) / (outbound_speed + return_speed)
  2 * distance = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_margo_walk_l14_1490


namespace NUMINAMATH_CALUDE_f_opens_upwards_f_passes_through_origin_f_satisfies_conditions_l14_1461

/-- A quadratic function that opens upwards and passes through (0,1) -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The graph of f opens upwards -/
theorem f_opens_upwards : ∀ x y : ℝ, x < y → f ((x + y) / 2) < (f x + f y) / 2 := by sorry

/-- f passes through the point (0,1) -/
theorem f_passes_through_origin : f 0 = 1 := by sorry

/-- f is a quadratic function satisfying the given conditions -/
theorem f_satisfies_conditions : 
  (∀ x y : ℝ, x < y → f ((x + y) / 2) < (f x + f y) / 2) ∧ 
  f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_opens_upwards_f_passes_through_origin_f_satisfies_conditions_l14_1461


namespace NUMINAMATH_CALUDE_min_good_pairs_l14_1455

/-- A circular arrangement of integers from 1 to 100 -/
def CircularArrangement := Fin 100 → ℕ

/-- Property that each number is either greater than both neighbors or less than both neighbors -/
def ValidArrangement (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 100, (arr i > arr (i - 1) ∧ arr i > arr (i + 1)) ∨ 
                 (arr i < arr (i - 1) ∧ arr i < arr (i + 1))

/-- Definition of a "good" pair -/
def GoodPair (arr : CircularArrangement) (i : Fin 100) : Prop :=
  ValidArrangement (Function.update (Function.update arr i (arr (i + 1))) (i + 1) (arr i))

/-- The main theorem stating that any valid arrangement has at least 51 good pairs -/
theorem min_good_pairs (arr : CircularArrangement) (h : ValidArrangement arr) :
  ∃ (s : Finset (Fin 100)), s.card ≥ 51 ∧ ∀ i ∈ s, GoodPair arr i :=
sorry

end NUMINAMATH_CALUDE_min_good_pairs_l14_1455


namespace NUMINAMATH_CALUDE_zain_coin_count_l14_1421

/-- Represents the number of coins Emerie has of each type -/
structure EmerieCoins where
  quarters : Nat
  dimes : Nat
  nickels : Nat

/-- Calculates the total number of coins Zain has given Emerie's coin counts -/
def zainTotalCoins (e : EmerieCoins) : Nat :=
  (e.quarters + 10) + (e.dimes + 10) + (e.nickels + 10)

theorem zain_coin_count (e : EmerieCoins) 
  (hq : e.quarters = 6) 
  (hd : e.dimes = 7) 
  (hn : e.nickels = 5) : 
  zainTotalCoins e = 48 := by
  sorry

end NUMINAMATH_CALUDE_zain_coin_count_l14_1421


namespace NUMINAMATH_CALUDE_cyclic_inequality_l14_1429

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b) / (a * b + a^5 + b^5) + (b * c) / (b * c + b^5 + c^5) + (c * a) / (c * a + c^5 + a^5) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l14_1429


namespace NUMINAMATH_CALUDE_maggie_earnings_proof_l14_1402

/-- Calculates Maggie's earnings from magazine subscriptions -/
def maggieEarnings (pricePerSubscription : ℕ) 
                   (parentsSubscriptions : ℕ)
                   (grandfatherSubscriptions : ℕ)
                   (nextDoorNeighborSubscriptions : ℕ) : ℕ :=
  let otherNeighborSubscriptions := 2 * nextDoorNeighborSubscriptions
  let totalSubscriptions := parentsSubscriptions + grandfatherSubscriptions + 
                            nextDoorNeighborSubscriptions + otherNeighborSubscriptions
  pricePerSubscription * totalSubscriptions

/-- Proves that Maggie's earnings are $55.00 -/
theorem maggie_earnings_proof : 
  maggieEarnings 5 4 1 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_maggie_earnings_proof_l14_1402


namespace NUMINAMATH_CALUDE_quadratic_equation_completion_l14_1425

theorem quadratic_equation_completion (x k ℓ : ℝ) : 
  (13 * x^2 + 39 * x - 91 = 0) ∧ 
  ((x + k)^2 - |ℓ| = 0) →
  |k + ℓ| = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_completion_l14_1425


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l14_1424

def sequence_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (List.range n).map a |>.sum

theorem sequence_sum_formula (a : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → (sequence_sum a n - 1)^2 - a n * (sequence_sum a n - 1) - a n = 0) →
  (∀ n : ℕ, n > 0 → sequence_sum a n = n / (n + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l14_1424


namespace NUMINAMATH_CALUDE_floor_abs_negative_l14_1491

theorem floor_abs_negative : ⌊|(-57.8 : ℝ)|⌋ = 57 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_l14_1491


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l14_1447

theorem number_exceeding_percentage : ∃ x : ℝ, x = 0.16 * x + 105 ∧ x = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l14_1447


namespace NUMINAMATH_CALUDE_basketball_team_enrollment_l14_1465

theorem basketball_team_enrollment (total_players : ℕ) 
  (physics_enrollment : ℕ) (both_enrollment : ℕ) :
  total_players = 15 →
  physics_enrollment = 9 →
  both_enrollment = 3 →
  physics_enrollment + (total_players - physics_enrollment) ≥ total_players →
  total_players - physics_enrollment + both_enrollment = 9 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_enrollment_l14_1465


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l14_1440

/-- The original plane equation -/
def plane_equation (x y z : ℝ) : Prop := x - 3*y + 5*z - 1 = 0

/-- The similarity transformation coefficient -/
def k : ℝ := -1

/-- The point A -/
def point_A : ℝ × ℝ × ℝ := (2, 0, -1)

/-- The transformed plane equation -/
def transformed_plane_equation (x y z : ℝ) : Prop := x - 3*y + 5*z + 1 = 0

/-- Theorem stating that point A does not belong to the transformed plane -/
theorem point_not_on_transformed_plane :
  ¬(transformed_plane_equation point_A.1 point_A.2.1 point_A.2.2) :=
sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l14_1440


namespace NUMINAMATH_CALUDE_max_tickets_jane_can_buy_l14_1445

theorem max_tickets_jane_can_buy (ticket_price : ℚ) (budget : ℚ) : 
  ticket_price = 27/2 → budget = 100 → 
  (∃ n : ℕ, n * ticket_price ≤ budget ∧ 
    ∀ m : ℕ, m * ticket_price ≤ budget → m ≤ n) → 
  (∃ n : ℕ, n * ticket_price ≤ budget ∧ 
    ∀ m : ℕ, m * ticket_price ≤ budget → m ≤ n) ∧ n = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_max_tickets_jane_can_buy_l14_1445


namespace NUMINAMATH_CALUDE_computer_price_theorem_l14_1484

/-- The sticker price of the computer -/
def sticker_price : ℝ := 750

/-- The price at store A after discount and rebate -/
def price_A (x : ℝ) : ℝ := 0.8 * x - 100

/-- The price at store B after discount -/
def price_B (x : ℝ) : ℝ := 0.7 * x

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem computer_price_theorem :
  price_A sticker_price = price_B sticker_price - 25 :=
by sorry

end NUMINAMATH_CALUDE_computer_price_theorem_l14_1484


namespace NUMINAMATH_CALUDE_system_solution_difference_l14_1432

theorem system_solution_difference (a b x y : ℝ) : 
  (2 * x + y = b) → 
  (x - b * y = a) → 
  (x = 1) → 
  (y = 0) → 
  (a - b = -1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_difference_l14_1432


namespace NUMINAMATH_CALUDE_group_size_proof_l14_1415

theorem group_size_proof : 
  ∀ n : ℕ, 
  (n : ℝ) * (n : ℝ) = 9801 → 
  n = 99 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l14_1415


namespace NUMINAMATH_CALUDE_max_top_young_men_l14_1474

/-- Represents a person with height and weight -/
structure Person where
  height : ℝ
  weight : ℝ

/-- Checks if person a is not inferior to person b -/
def notInferiorTo (a b : Person) : Prop :=
  a.height > b.height ∨ a.weight > b.weight

/-- Checks if a person is a top young man among a group of people -/
def isTopYoungMan (p : Person) (group : List Person) : Prop :=
  ∀ q ∈ group, p ≠ q → notInferiorTo p q

/-- The main theorem stating the maximum number of top young men -/
theorem max_top_young_men (youngMen : List Person) 
    (h : youngMen.length = 100) :
    ∃ (topYoungMen : List Person), 
      (∀ p ∈ topYoungMen, p ∈ youngMen ∧ isTopYoungMan p youngMen) ∧
      topYoungMen.length = 100 :=
  sorry

end NUMINAMATH_CALUDE_max_top_young_men_l14_1474


namespace NUMINAMATH_CALUDE_card_game_remainder_l14_1485

def deck_size : ℕ := 60
def hand_size : ℕ := 12

def possible_remainders : List ℕ := [20, 40, 60, 80, 0]

theorem card_game_remainder :
  ∃ (r : ℕ), r ∈ possible_remainders ∧ 
  (Nat.choose deck_size hand_size) % 100 = r :=
sorry

end NUMINAMATH_CALUDE_card_game_remainder_l14_1485


namespace NUMINAMATH_CALUDE_prime_iff_binomial_congruence_l14_1451

theorem prime_iff_binomial_congruence (n : ℕ) (hn : n > 0) :
  Nat.Prime n ↔ ∀ k : ℕ, k < n → (Nat.choose (n - 1) k) % n = ((-1 : ℤ) ^ k).toNat % n :=
sorry

end NUMINAMATH_CALUDE_prime_iff_binomial_congruence_l14_1451


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_21_mod_30_l14_1456

theorem smallest_five_digit_congruent_to_21_mod_30 :
  ∀ n : ℕ, 
    n ≥ 10000 ∧ n ≤ 99999 ∧ n ≡ 21 [MOD 30] → 
    n ≥ 10011 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_21_mod_30_l14_1456


namespace NUMINAMATH_CALUDE_tan_theta_value_l14_1470

theorem tan_theta_value (θ : Real) 
  (h : (1 + Real.sin (2 * θ)) / (Real.cos θ ^ 2 - Real.sin θ ^ 2) = -3) : 
  Real.tan θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l14_1470


namespace NUMINAMATH_CALUDE_tree_growth_l14_1452

theorem tree_growth (initial_circumference : ℝ) (annual_increase : ℝ) (target_circumference : ℝ) :
  initial_circumference = 10 ∧ 
  annual_increase = 3 ∧ 
  target_circumference = 90 →
  ∃ x : ℝ, x > 80 / 3 ∧ initial_circumference + x * annual_increase > target_circumference :=
by sorry

end NUMINAMATH_CALUDE_tree_growth_l14_1452


namespace NUMINAMATH_CALUDE_phone_answer_probability_l14_1430

/-- The probability of answering the phone on the first ring -/
def p1 : ℝ := 0.1

/-- The probability of answering the phone on the second ring -/
def p2 : ℝ := 0.2

/-- The probability of answering the phone on the third ring -/
def p3 : ℝ := 0.25

/-- The probability of answering the phone on the fourth ring -/
def p4 : ℝ := 0.25

/-- The theorem stating that the probability of answering the phone before the fifth ring is 0.8 -/
theorem phone_answer_probability : p1 + p2 + p3 + p4 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_phone_answer_probability_l14_1430


namespace NUMINAMATH_CALUDE_characterize_functions_l14_1493

def is_valid_function (f : ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, m > 0 ∧ n > 0 → ⌊(f (m * n) : ℚ) / n⌋ = f m

theorem characterize_functions :
  ∀ f : ℕ → ℤ, is_valid_function f →
    ∃ r : ℝ, (∀ n : ℕ, n > 0 → f n = ⌊n * r⌋) ∨
              (∀ n : ℕ, n > 0 → f n = ⌈n * r⌉ - 1) :=
sorry

end NUMINAMATH_CALUDE_characterize_functions_l14_1493


namespace NUMINAMATH_CALUDE_second_expression_proof_l14_1497

theorem second_expression_proof (a : ℝ) (x : ℝ) :
  a = 34 →
  ((2 * a + 16) + x) / 2 = 89 →
  x = 94 := by
sorry

end NUMINAMATH_CALUDE_second_expression_proof_l14_1497


namespace NUMINAMATH_CALUDE_triangle_area_rational_l14_1460

theorem triangle_area_rational
  (m n p q : ℚ)
  (hm : m > 0)
  (hn : n > 0)
  (hp : p > 0)
  (hq : q > 0)
  (hmn : m > n)
  (hpq : p > q)
  (a : ℚ := m * n * (p^2 + q^2))
  (b : ℚ := p * q * (m^2 + n^2))
  (c : ℚ := (m * q + n * p) * (m * p - n * q)) :
  ∃ (t : ℚ), t = m * n * p * q * (m * q + n * p) * (m * p - n * q) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_rational_l14_1460


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l14_1464

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | x ≥ 2}
def T : Set ℝ := {x : ℝ | x ≤ 5}

-- State the theorem
theorem intersection_of_S_and_T : S ∩ T = Set.Icc 2 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l14_1464


namespace NUMINAMATH_CALUDE_bicycle_journey_initial_time_l14_1444

theorem bicycle_journey_initial_time 
  (speed : ℝ) 
  (additional_distance : ℝ) 
  (rest_time : ℝ) 
  (final_distance : ℝ) 
  (total_time : ℝ) :
  speed = 10 →
  additional_distance = 15 →
  rest_time = 30 →
  final_distance = 20 →
  total_time = 270 →
  ∃ (initial_time : ℝ), 
    initial_time * 60 + additional_distance / speed * 60 + rest_time + final_distance / speed * 60 = total_time ∧ 
    initial_time * 60 = 30 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_journey_initial_time_l14_1444


namespace NUMINAMATH_CALUDE_line_intersection_l14_1487

theorem line_intersection :
  ∃! p : ℚ × ℚ, 8 * p.1 - 3 * p.2 = 20 ∧ 9 * p.1 + 2 * p.2 = 17 :=
by
  use (91/43, 61/43)
  sorry

end NUMINAMATH_CALUDE_line_intersection_l14_1487


namespace NUMINAMATH_CALUDE_percentage_of_S_grades_l14_1498

def grading_scale (score : ℕ) : String :=
  if 95 ≤ score ∧ score ≤ 100 then "S"
  else if 88 ≤ score ∧ score < 95 then "A"
  else if 80 ≤ score ∧ score < 88 then "B"
  else if 72 ≤ score ∧ score < 80 then "C"
  else if 65 ≤ score ∧ score < 72 then "D"
  else "F"

def scores : List ℕ := [95, 88, 70, 100, 75, 90, 80, 77, 67, 78, 85, 65, 72, 82, 96]

theorem percentage_of_S_grades :
  (scores.filter (λ score => grading_scale score = "S")).length / scores.length * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_S_grades_l14_1498


namespace NUMINAMATH_CALUDE_line_passes_through_center_l14_1418

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x + y + 1 = 0

-- Define the center of a circle
def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 2

-- Theorem statement
theorem line_passes_through_center :
  ∃ h k : ℝ, is_center h k ∧ line_equation h k :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_center_l14_1418


namespace NUMINAMATH_CALUDE_arithmetic_sum_l14_1463

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 3 = 13 →
  a 1 = 2 →
  a 4 + a 5 + a 6 = 42 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sum_l14_1463


namespace NUMINAMATH_CALUDE_function_inequality_l14_1479

noncomputable def f (x : ℝ) : ℝ := (Real.exp 2 * x^2 + 1) / x

noncomputable def g (x : ℝ) : ℝ := (Real.exp 2 * x^2) / Real.exp x

theorem function_inequality (k : ℝ) (hk : k > 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → g x₁ / k ≤ f x₂ / (k + 1)) →
  k ≥ 4 / (2 * Real.exp 1 - 4) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l14_1479


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l14_1439

/-- The length of a bridge that a train can cross, given the train's length, speed, and time to cross. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proves that the length of the bridge is 230 meters given the specified conditions. -/
theorem bridge_length_proof :
  bridge_length 145 45 30 = 230 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l14_1439


namespace NUMINAMATH_CALUDE_complex_equation_solution_l14_1417

theorem complex_equation_solution :
  ∃ (x : ℂ), 5 - 2 * I * x = 4 - 5 * I * x ∧ x = I / 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l14_1417


namespace NUMINAMATH_CALUDE_candy_distribution_l14_1438

theorem candy_distribution (num_children : ℕ) : 
  (3 * num_children + 12 = 5 * num_children - 10) → num_children = 11 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l14_1438


namespace NUMINAMATH_CALUDE_carrie_harvest_money_l14_1458

/-- Calculates the total money earned from selling tomatoes and carrots -/
def totalMoney (numTomatoes : ℕ) (numCarrots : ℕ) (priceTomato : ℚ) (priceCarrot : ℚ) : ℚ :=
  numTomatoes * priceTomato + numCarrots * priceCarrot

/-- Proves that the total money earned is correct for Carrie's harvest -/
theorem carrie_harvest_money :
  totalMoney 200 350 1 (3/2) = 725 := by
  sorry

end NUMINAMATH_CALUDE_carrie_harvest_money_l14_1458


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l14_1401

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l14_1401


namespace NUMINAMATH_CALUDE_high_school_science_club_payment_l14_1477

theorem high_school_science_club_payment (B C : Nat) : 
  B < 10 → C < 10 → 
  (100 * B + 40 + C) % 15 = 0 → 
  (100 * B + 40 + C) % 5 = 0 → 
  B = 5 := by
sorry

end NUMINAMATH_CALUDE_high_school_science_club_payment_l14_1477


namespace NUMINAMATH_CALUDE_equation_solution_l14_1400

theorem equation_solution :
  ∃ x : ℝ, (x + 1 = 5) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l14_1400


namespace NUMINAMATH_CALUDE_zero_function_theorem_l14_1433

theorem zero_function_theorem (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h1 : ∀ x : ℤ, deriv f x = 0)
  (h2 : ∀ x : ℝ, deriv f x = 0 → f x = 0) :
  ∀ x : ℝ, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_zero_function_theorem_l14_1433


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_achievable_l14_1449

theorem min_value_quadratic (x y : ℝ) : x^2 + y^2 - 8*x + 6*y + x*y + 20 ≥ -88/3 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, x^2 + y^2 - 8*x + 6*y + x*y + 20 = -88/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_achievable_l14_1449


namespace NUMINAMATH_CALUDE_model2_best_fit_l14_1408

-- Define the coefficient of determination for each model
def R2_model1 : ℝ := 0.78
def R2_model2 : ℝ := 0.85
def R2_model3 : ℝ := 0.61
def R2_model4 : ℝ := 0.31

-- Define a function to calculate the distance from 1
def distance_from_one (x : ℝ) : ℝ := |1 - x|

-- Theorem stating that Model 2 has the best fitting effect
theorem model2_best_fit :
  distance_from_one R2_model2 < distance_from_one R2_model1 ∧
  distance_from_one R2_model2 < distance_from_one R2_model3 ∧
  distance_from_one R2_model2 < distance_from_one R2_model4 :=
by sorry


end NUMINAMATH_CALUDE_model2_best_fit_l14_1408


namespace NUMINAMATH_CALUDE_calculation_proof_l14_1450

theorem calculation_proof : ((0.15 * 320 + 0.12 * 480) / (2/5)) * (3/4) = 198 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l14_1450
