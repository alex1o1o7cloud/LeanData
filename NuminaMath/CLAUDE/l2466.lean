import Mathlib

namespace NUMINAMATH_CALUDE_expand_expression_l2466_246665

theorem expand_expression (x : ℝ) : (x - 3) * (4 * x + 8) = 4 * x^2 - 4 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2466_246665


namespace NUMINAMATH_CALUDE_max_a_value_l2466_246662

/-- A lattice point in an xy-coordinate system -/
def LatticePoint (x y : ℤ) : Prop := True

/-- The line equation y = mx + 3 -/
def LineEquation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

/-- Predicate to check if a point satisfies the line equation -/
def SatisfiesEquation (m : ℚ) (x y : ℤ) : Prop :=
  LineEquation m x = y

/-- The main theorem -/
theorem max_a_value : 
  ∃ (a : ℚ), a = 101 / 151 ∧ 
  (∀ (m : ℚ), 2/3 < m → m < a → 
    ∀ (x y : ℤ), 0 < x → x ≤ 150 → LatticePoint x y → ¬SatisfiesEquation m x y) ∧
  (∀ (a' : ℚ), a' > a → 
    ∃ (m : ℚ), 2/3 < m ∧ m < a' ∧
      ∃ (x y : ℤ), 0 < x ∧ x ≤ 150 ∧ LatticePoint x y ∧ SatisfiesEquation m x y) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2466_246662


namespace NUMINAMATH_CALUDE_orthogonal_vectors_sum_l2466_246628

theorem orthogonal_vectors_sum (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (hx : x₁ + x₂ + x₃ = 0)
  (hy : y₁ + y₂ + y₃ = 0)
  (hxy : x₁*y₁ + x₂*y₂ + x₃*y₃ = 0) :
  x₁^2 / (x₁^2 + x₂^2 + x₃^2) + y₁^2 / (y₁^2 + y₂^2 + y₃^2) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_sum_l2466_246628


namespace NUMINAMATH_CALUDE_allie_wildflowers_l2466_246605

/-- The number of wildflowers Allie picked -/
def total_flowers : ℕ := 44

/-- The number of yellow and white flowers -/
def yellow_white : ℕ := 13

/-- The number of red and yellow flowers -/
def red_yellow : ℕ := 17

/-- The number of red and white flowers -/
def red_white : ℕ := 14

/-- The difference between red and white flowers -/
def red_white_diff : ℕ := 4

theorem allie_wildflowers :
  total_flowers = yellow_white + red_yellow + red_white :=
by sorry

end NUMINAMATH_CALUDE_allie_wildflowers_l2466_246605


namespace NUMINAMATH_CALUDE_fathers_age_is_32_l2466_246666

/-- The present age of the father -/
def father_age : ℕ := 32

/-- The present age of the older son -/
def older_son_age : ℕ := 22

/-- The present age of the younger son -/
def younger_son_age : ℕ := 18

/-- The average age of the father and his two sons is 24 years -/
axiom average_age : (father_age + older_son_age + younger_son_age) / 3 = 24

/-- 5 years ago, the average age of the two sons was 15 years -/
axiom sons_average_age_5_years_ago : (older_son_age - 5 + younger_son_age - 5) / 2 = 15

/-- The difference between the ages of the two sons is 4 years -/
axiom sons_age_difference : older_son_age - younger_son_age = 4

/-- Theorem: Given the conditions, the father's present age is 32 years -/
theorem fathers_age_is_32 : father_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_is_32_l2466_246666


namespace NUMINAMATH_CALUDE_power_equation_solution_l2466_246671

theorem power_equation_solution (m n : ℕ) (h1 : (1/5)^m * (1/4)^n = 1/(10^4)) (h2 : m = 4) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2466_246671


namespace NUMINAMATH_CALUDE_grunters_win_probability_l2466_246690

/-- The probability of the Grunters winning a single game -/
def p : ℚ := 3/5

/-- The number of games played -/
def n : ℕ := 5

/-- The probability of winning all games -/
def win_all : ℚ := p^n

theorem grunters_win_probability : win_all = 243/3125 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l2466_246690


namespace NUMINAMATH_CALUDE_range_of_m_l2466_246680

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 2 then 2^(x - m) else (m * x) / (4 * x^2 + 16)

theorem range_of_m (m : ℝ) :
  (∀ x₁ ≥ 2, ∃ x₂ ≤ 2, f m x₁ = f m x₂) →
  m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2466_246680


namespace NUMINAMATH_CALUDE_y_range_for_x_condition_l2466_246656

theorem y_range_for_x_condition (x y : ℝ) : 
  (4 * x + y = 1) → ((-1 < x ∧ x ≤ 2) ↔ (-7 ≤ y ∧ y < -3)) := by
  sorry

end NUMINAMATH_CALUDE_y_range_for_x_condition_l2466_246656


namespace NUMINAMATH_CALUDE_apple_basket_count_l2466_246651

theorem apple_basket_count (rotten_percent : ℝ) (spotted_percent : ℝ) (insect_percent : ℝ) (varying_rot_percent : ℝ) (perfect_count : ℕ) : 
  rotten_percent = 0.12 →
  spotted_percent = 0.07 →
  insect_percent = 0.05 →
  varying_rot_percent = 0.03 →
  perfect_count = 66 →
  ∃ (total : ℕ), total = 90 ∧ 
    (1 - (rotten_percent + spotted_percent + insect_percent + varying_rot_percent)) * (total : ℝ) = perfect_count :=
by
  sorry

end NUMINAMATH_CALUDE_apple_basket_count_l2466_246651


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l2466_246683

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n = 15625) ∧ 
  (∀ m : ℕ, m < n → m < 10000 ∨ m > 99999 ∨ ¬∃ a : ℕ, m = a^2 ∨ ¬∃ b : ℕ, m = b^3) ∧
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n = b^3) :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l2466_246683


namespace NUMINAMATH_CALUDE_product_of_numbers_l2466_246678

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2466_246678


namespace NUMINAMATH_CALUDE_cheries_sparklers_l2466_246615

/-- Represents the number of fireworks in a box -/
structure FireworksBox where
  sparklers : ℕ
  whistlers : ℕ

/-- The total number of fireworks in a box -/
def FireworksBox.total (box : FireworksBox) : ℕ := box.sparklers + box.whistlers

theorem cheries_sparklers (koby_box : FireworksBox) 
                          (cherie_box : FireworksBox) 
                          (h1 : koby_box.sparklers = 3)
                          (h2 : koby_box.whistlers = 5)
                          (h3 : cherie_box.whistlers = 9)
                          (h4 : 2 * koby_box.total + cherie_box.total = 33) :
  cherie_box.sparklers = 8 := by
  sorry

#check cheries_sparklers

end NUMINAMATH_CALUDE_cheries_sparklers_l2466_246615


namespace NUMINAMATH_CALUDE_courier_delivery_patterns_l2466_246676

/-- Represents the number of acceptable delivery patterns for n offices -/
def P : ℕ → ℕ
| 0 => 1  -- Base case: only one way to deliver to 0 offices
| 1 => 2  -- Can either deliver or not deliver to 1 office
| 2 => 4  -- All combinations for 2 offices
| 3 => 8  -- All combinations for 3 offices
| 4 => 15 -- All combinations for 4 offices, excluding all non-deliveries
| n + 5 => P (n + 4) + P (n + 3) + P (n + 2) + P (n + 1)

theorem courier_delivery_patterns :
  P 12 = 927 := by
  sorry


end NUMINAMATH_CALUDE_courier_delivery_patterns_l2466_246676


namespace NUMINAMATH_CALUDE_girls_in_classroom_l2466_246624

theorem girls_in_classroom (total_students : ℕ) (girls_ratio boys_ratio : ℕ) : 
  total_students = 28 → 
  girls_ratio = 3 → 
  boys_ratio = 4 → 
  (girls_ratio + boys_ratio) * (total_students / (girls_ratio + boys_ratio)) = girls_ratio * (total_students / (girls_ratio + boys_ratio)) + boys_ratio * (total_students / (girls_ratio + boys_ratio)) →
  girls_ratio * (total_students / (girls_ratio + boys_ratio)) = 12 := by
sorry

end NUMINAMATH_CALUDE_girls_in_classroom_l2466_246624


namespace NUMINAMATH_CALUDE_eleven_by_seven_max_squares_l2466_246698

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of squares that can be cut from a rectangle --/
def maxSquares (rect : Rectangle) : ℕ :=
  sorry

/-- The theorem stating the maximum number of squares for an 11x7 rectangle --/
theorem eleven_by_seven_max_squares :
  maxSquares ⟨11, 7⟩ = 6 := by
  sorry

end NUMINAMATH_CALUDE_eleven_by_seven_max_squares_l2466_246698


namespace NUMINAMATH_CALUDE_paper_fold_sum_l2466_246677

-- Define the fold line
def fold_line (x y : ℝ) : Prop := y = x

-- Define the mapping of points
def maps_to (x1 y1 x2 y2 : ℝ) : Prop :=
  fold_line ((x1 + x2) / 2) ((y1 + y2) / 2) ∧
  (y2 - y1) = -(x2 - x1)

-- Main theorem
theorem paper_fold_sum (m n : ℝ) :
  maps_to 0 5 5 0 →  -- (0,5) maps to (5,0)
  maps_to 8 4 m n →  -- (8,4) maps to (m,n)
  m + n = 12 := by
sorry

end NUMINAMATH_CALUDE_paper_fold_sum_l2466_246677


namespace NUMINAMATH_CALUDE_sara_movie_expenses_l2466_246640

/-- The total amount Sara spent on movies -/
def total_spent (ticket_price : ℚ) (num_tickets : ℕ) (rental_price : ℚ) (purchase_price : ℚ) : ℚ :=
  ticket_price * num_tickets + rental_price + purchase_price

/-- Theorem stating the total amount Sara spent on movies -/
theorem sara_movie_expenses :
  let ticket_price : ℚ := 10.62
  let num_tickets : ℕ := 2
  let rental_price : ℚ := 1.59
  let purchase_price : ℚ := 13.95
  total_spent ticket_price num_tickets rental_price purchase_price = 36.78 := by
  sorry

end NUMINAMATH_CALUDE_sara_movie_expenses_l2466_246640


namespace NUMINAMATH_CALUDE_max_area_of_cut_triangle_l2466_246610

/-- Triangle ABC with side lengths -/
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)

/-- The given triangle -/
def givenTriangle : Triangle :=
  { AB := 13, BC := 14, CA := 15 }

/-- A line cutting the triangle -/
structure CuttingLine :=
  (intersectsSide1 : ℝ)
  (intersectsSide2 : ℝ)

/-- The area of the triangle formed by the cutting line -/
def areaOfCutTriangle (t : Triangle) (l : CuttingLine) : ℝ :=
  sorry

/-- The perimeter of the triangle formed by the cutting line -/
def perimeterOfCutTriangle (t : Triangle) (l : CuttingLine) : ℝ :=
  sorry

/-- The perimeter of the quadrilateral formed by the cutting line -/
def perimeterOfCutQuadrilateral (t : Triangle) (l : CuttingLine) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem max_area_of_cut_triangle (t : Triangle) :
  t = givenTriangle →
  ∃ (l : CuttingLine),
    perimeterOfCutTriangle t l = perimeterOfCutQuadrilateral t l ∧
    ∀ (l' : CuttingLine),
      perimeterOfCutTriangle t l' = perimeterOfCutQuadrilateral t l' →
      areaOfCutTriangle t l' ≤ 1323 / 26 :=
sorry

end NUMINAMATH_CALUDE_max_area_of_cut_triangle_l2466_246610


namespace NUMINAMATH_CALUDE_factor_x4_minus_64_l2466_246636

theorem factor_x4_minus_64 (x : ℝ) : x^4 - 64 = (x^2 + 8) * (x^2 - 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x4_minus_64_l2466_246636


namespace NUMINAMATH_CALUDE_expression_factorization_l2466_246611

theorem expression_factorization (a b c d p q r s : ℝ) :
  (a * p + b * q + c * r + d * s)^2 +
  (a * q - b * p + c * s - d * r)^2 +
  (a * r - b * s - c * p + d * q)^2 +
  (a * s + b * r - c * q - d * p)^2 =
  (a^2 + b^2 + c^2 + d^2) * (p^2 + q^2 + r^2 + s^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2466_246611


namespace NUMINAMATH_CALUDE_arithmetic_sequence_b_l2466_246645

def arithmetic_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d

theorem arithmetic_sequence_b (b : ℝ) 
  (h₁ : arithmetic_sequence 120 b (1/5))
  (h₂ : b > 0) : 
  b = 60.1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_b_l2466_246645


namespace NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_l2466_246612

theorem polygon_sides_from_interior_angle (interior_angle : ℝ) : 
  interior_angle = 140 → (360 / (180 - interior_angle) : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_l2466_246612


namespace NUMINAMATH_CALUDE_college_student_count_l2466_246635

theorem college_student_count (boys girls : ℕ) (h1 : boys = 2 * girls) (h2 : girls = 200) :
  boys + girls = 600 := by
  sorry

end NUMINAMATH_CALUDE_college_student_count_l2466_246635


namespace NUMINAMATH_CALUDE_unique_prime_power_sum_l2466_246696

theorem unique_prime_power_sum (p q : ℕ) : 
  Prime p → Prime q → Prime (p^q + q^p) → (p = 2 ∧ q = 3) := by
sorry

end NUMINAMATH_CALUDE_unique_prime_power_sum_l2466_246696


namespace NUMINAMATH_CALUDE_expected_value_of_winnings_l2466_246658

def fair_10_sided_die : Finset ℕ := Finset.range 10

def winnings (roll : ℕ) : ℚ :=
  if roll % 2 = 0 then roll else 0

theorem expected_value_of_winnings :
  (Finset.sum fair_10_sided_die (λ roll => (1 : ℚ) / 10 * winnings roll)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_winnings_l2466_246658


namespace NUMINAMATH_CALUDE_shaded_area_is_60_l2466_246604

/-- Represents a point in a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a rectangular grid -/
structure Grid where
  width : ℕ
  height : ℕ

/-- Represents the shaded region in the grid -/
structure ShadedRegion where
  grid : Grid
  points : List Point

/-- Calculates the area of the shaded region -/
def shadedArea (region : ShadedRegion) : ℕ :=
  sorry

/-- The specific grid and shaded region from the problem -/
def problemGrid : Grid :=
  { width := 15, height := 5 }

def problemShadedRegion : ShadedRegion :=
  { grid := problemGrid,
    points := [
      { x := 0, y := 0 },   -- bottom left corner
      { x := 4, y := 3 },   -- first point
      { x := 9, y := 5 },   -- second point
      { x := 15, y := 5 }   -- top right corner
    ] }

/-- The main theorem to prove -/
theorem shaded_area_is_60 :
  shadedArea problemShadedRegion = 60 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_is_60_l2466_246604


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2466_246689

theorem repeating_decimal_sum : 
  (2 : ℚ) / 9 + (2 : ℚ) / 99 = (8 : ℚ) / 33 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2466_246689


namespace NUMINAMATH_CALUDE_derivative_at_pi_over_four_l2466_246641

open Real

theorem derivative_at_pi_over_four :
  let f (x : ℝ) := cos x * (sin x - cos x)
  let f' := deriv f
  f' (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_over_four_l2466_246641


namespace NUMINAMATH_CALUDE_prob_not_sunny_l2466_246647

/-- Given that the probability of a sunny day is 5/7, 
    prove that the probability of a not sunny day is 2/7 -/
theorem prob_not_sunny (prob_sunny : ℚ) (h : prob_sunny = 5 / 7) :
  1 - prob_sunny = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_sunny_l2466_246647


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_b_l2466_246608

/-- Two 2D vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- The direction vector of the first line -/
def v : ℝ × ℝ := (4, -9)

/-- The direction vector of the second line -/
def w (b : ℝ) : ℝ × ℝ := (b, 3)

/-- Theorem: If the direction vectors v and w(b) are perpendicular, then b = 27/4 -/
theorem perpendicular_vectors_imply_b (b : ℝ) :
  perpendicular v (w b) → b = 27/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_b_l2466_246608


namespace NUMINAMATH_CALUDE_car_trip_duration_l2466_246684

theorem car_trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) : 
  initial_speed = 30 →
  initial_time = 6 →
  additional_speed = 46 →
  average_speed = 34 →
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) / total_time = average_speed ∧
    total_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_duration_l2466_246684


namespace NUMINAMATH_CALUDE_go_out_to_sea_is_better_l2466_246691

/-- Represents the decision to go out to sea or not -/
inductive Decision
| GoOut
| StayIn

/-- Represents the weather condition -/
inductive Weather
| Good
| Bad

/-- The profit or loss for each scenario -/
def profit (d : Decision) (w : Weather) : ℤ :=
  match d, w with
  | Decision.GoOut, Weather.Good => 6000
  | Decision.GoOut, Weather.Bad => -8000
  | Decision.StayIn, _ => -1000

/-- The probability of each weather condition -/
def weather_prob (w : Weather) : ℚ :=
  match w with
  | Weather.Good => 1/2
  | Weather.Bad => 4/10

/-- The expected value of a decision -/
def expected_value (d : Decision) : ℚ :=
  (weather_prob Weather.Good * profit d Weather.Good) +
  (weather_prob Weather.Bad * profit d Weather.Bad)

/-- Theorem stating that going out to sea has a higher expected value -/
theorem go_out_to_sea_is_better :
  expected_value Decision.GoOut > expected_value Decision.StayIn :=
by sorry

end NUMINAMATH_CALUDE_go_out_to_sea_is_better_l2466_246691


namespace NUMINAMATH_CALUDE_largest_four_digit_number_l2466_246694

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_two_digits (n : ℕ) : ℕ := n / 100

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem largest_four_digit_number (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : n % 10 ≠ 0)
  (h3 : 2014 % (first_two_digits n) = 0)
  (h4 : 2014 % ((first_two_digits n) * (last_two_digits n)) = 0) :
  n ≤ 5376 ∧ ∃ m : ℕ, m = 5376 ∧ 
    is_four_digit m ∧ 
    m % 10 ≠ 0 ∧ 
    2014 % (first_two_digits m) = 0 ∧ 
    2014 % ((first_two_digits m) * (last_two_digits m)) = 0 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_number_l2466_246694


namespace NUMINAMATH_CALUDE_union_complement_equals_set_l2466_246619

def U : Set ℤ := {x | -3 < x ∧ x < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_complement_equals_set : A ∪ (U \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_set_l2466_246619


namespace NUMINAMATH_CALUDE_jerry_throwing_points_l2466_246644

/-- Represents the point system in Mrs. Carlton's class -/
structure PointSystem where
  interrupt_points : ℕ
  insult_points : ℕ
  office_threshold : ℕ

/-- Represents Jerry's behavior -/
structure JerryBehavior where
  interrupts : ℕ
  insults : ℕ
  throws : ℕ

/-- Calculates the points Jerry has accumulated so far -/
def accumulated_points (ps : PointSystem) (jb : JerryBehavior) : ℕ :=
  ps.interrupt_points * jb.interrupts + ps.insult_points * jb.insults

/-- Theorem stating that Jerry gets 25 points for throwing things -/
theorem jerry_throwing_points (ps : PointSystem) (jb : JerryBehavior) :
    ps.interrupt_points = 5 →
    ps.insult_points = 10 →
    ps.office_threshold = 100 →
    jb.interrupts = 2 →
    jb.insults = 4 →
    jb.throws = 2 →
    (ps.office_threshold - accumulated_points ps jb) / jb.throws = 25 := by
  sorry

end NUMINAMATH_CALUDE_jerry_throwing_points_l2466_246644


namespace NUMINAMATH_CALUDE_notebook_dispatch_l2466_246637

theorem notebook_dispatch (x y : ℕ) 
  (h1 : x * (y + 5) = x * y + 1250) 
  (h2 : (x + 7) * y = x * y + 3150) : 
  x + y = 700 := by
  sorry

end NUMINAMATH_CALUDE_notebook_dispatch_l2466_246637


namespace NUMINAMATH_CALUDE_rectangle_area_18_pairs_l2466_246668

theorem rectangle_area_18_pairs : 
  {p : ℕ × ℕ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 18} = 
  {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_18_pairs_l2466_246668


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l2466_246695

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l2466_246695


namespace NUMINAMATH_CALUDE_sin_product_equality_l2466_246667

theorem sin_product_equality : (1 - Real.sin (π / 6)) * (1 - Real.sin (5 * π / 6)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equality_l2466_246667


namespace NUMINAMATH_CALUDE_paper_I_maximum_mark_l2466_246669

theorem paper_I_maximum_mark :
  ∃ (M : ℕ),
    (M : ℚ) * (55 : ℚ) / (100 : ℚ) = (65 : ℚ) + (35 : ℚ) ∧
    M = 182 := by
  sorry

end NUMINAMATH_CALUDE_paper_I_maximum_mark_l2466_246669


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_min_mn_l2466_246670

/-- Given two positive real numbers m and n satisfying 1/m + 2/n = 1,
    the eccentricity of the ellipse x²/m² + y²/n² = 1 is √3/2
    when mn takes its minimum value. -/
theorem ellipse_eccentricity_min_mn (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (h : 1/m + 2/n = 1) :
  let e := Real.sqrt (1 - (min m n)^2 / (max m n)^2)
  ∃ (min_mn : ℝ), (∀ m' n' : ℝ, m' > 0 → n' > 0 → 1/m' + 2/n' = 1 → m' * n' ≥ min_mn) ∧
    (m * n = min_mn → e = Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_min_mn_l2466_246670


namespace NUMINAMATH_CALUDE_pond_volume_calculation_l2466_246618

/-- The volume of a rectangular prism with given dimensions -/
def pond_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

/-- Theorem stating that the volume of the pond is 1200 cubic meters -/
theorem pond_volume_calculation :
  pond_volume 20 12 5 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_pond_volume_calculation_l2466_246618


namespace NUMINAMATH_CALUDE_find_k_angle_90_degrees_l2466_246650

-- Define vectors in R^2
def a : Fin 2 → ℝ := ![3, -1]
def b (k : ℝ) : Fin 2 → ℝ := ![1, k]

-- Define dot product for 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define perpendicularity for 2D vectors
def perpendicular (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

-- Theorem 1: Find the value of k
theorem find_k : ∃ k : ℝ, perpendicular a (b k) ∧ k = 3 := by sorry

-- Define vector addition and subtraction
def add_vectors (v w : Fin 2 → ℝ) : Fin 2 → ℝ := ![v 0 + w 0, v 1 + w 1]
def sub_vectors (v w : Fin 2 → ℝ) : Fin 2 → ℝ := ![v 0 - w 0, v 1 - w 1]

-- Theorem 2: Prove the angle between a + b and a - b is 90°
theorem angle_90_degrees : 
  let b' := b 3
  let sum := add_vectors a b'
  let diff := sub_vectors a b'
  perpendicular sum diff := by sorry

end NUMINAMATH_CALUDE_find_k_angle_90_degrees_l2466_246650


namespace NUMINAMATH_CALUDE_shower_tasks_count_l2466_246600

/-- The number of tasks to clean the house -/
def clean_house_tasks : ℕ := 7

/-- The number of tasks to make dinner -/
def make_dinner_tasks : ℕ := 4

/-- The time each task takes in minutes -/
def time_per_task : ℕ := 10

/-- The total time to complete all tasks in hours -/
def total_time_hours : ℕ := 2

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem shower_tasks_count : 
  (clean_house_tasks + make_dinner_tasks + 1) * time_per_task = total_time_hours * minutes_per_hour := by
  sorry

end NUMINAMATH_CALUDE_shower_tasks_count_l2466_246600


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l2466_246653

-- Define the sets A, B, and C
def A : Set ℝ := {-6, -5, -4, -3}
def B : Set ℝ := {2/3, 3/4, 7/9, 2.5}
def C : Set ℝ := {5, 5.5, 6, 6.5}

-- Define the theorem
theorem greatest_integer_difference (a b c : ℝ) 
  (ha : a ∈ A) (hb : b ∈ B) (hc : c ∈ C) :
  ∃ (d : ℤ), d = 5 ∧ 
  ∀ (a' b' c' : ℝ), a' ∈ A → b' ∈ B → c' ∈ C → 
  (Int.floor (|c' - Real.sqrt b' - (a' + Real.sqrt b')|) : ℤ) ≤ d :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l2466_246653


namespace NUMINAMATH_CALUDE_unique_two_digit_multiple_l2466_246659

theorem unique_two_digit_multiple : ∃! s : ℕ, 
  10 ≤ s ∧ s < 100 ∧ (13 * s) % 100 = 52 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_multiple_l2466_246659


namespace NUMINAMATH_CALUDE_increasing_sin_plus_linear_range_of_a_l2466_246606

/-- A function f : ℝ → ℝ is increasing if for all x₁ x₂, x₁ < x₂ implies f x₁ < f x₂ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- The main theorem: if y = sin x + ax is an increasing function on ℝ, then a ≥ 1 -/
theorem increasing_sin_plus_linear (a : ℝ) :
  IsIncreasing (fun x => Real.sin x + a * x) → a ≥ 1 := by
  sorry

/-- The range of a is [1, +∞) -/
theorem range_of_a (a : ℝ) :
  (IsIncreasing (fun x => Real.sin x + a * x) ↔ a ∈ Set.Ici 1) := by
  sorry

end NUMINAMATH_CALUDE_increasing_sin_plus_linear_range_of_a_l2466_246606


namespace NUMINAMATH_CALUDE_handshakes_in_specific_gathering_l2466_246623

/-- Represents a gathering of people with specific knowledge relationships -/
structure Gathering where
  total : Nat
  group1 : Nat
  group2 : Nat
  group2_with_connections : Nat
  group2_without_connections : Nat

/-- Calculates the number of handshakes in the gathering -/
def count_handshakes (g : Gathering) : Nat :=
  let group2_no_connections_handshakes := g.group2_without_connections * (g.total - 1)
  let group2_with_connections_handshakes := g.group2_with_connections * (g.total - 11)
  (group2_no_connections_handshakes + group2_with_connections_handshakes) / 2

/-- Theorem stating the number of handshakes in the specific gathering -/
theorem handshakes_in_specific_gathering :
  let g : Gathering := {
    total := 40,
    group1 := 25,
    group2 := 15,
    group2_with_connections := 5,
    group2_without_connections := 10
  }
  count_handshakes g = 305 := by
  sorry

#eval count_handshakes {
  total := 40,
  group1 := 25,
  group2 := 15,
  group2_with_connections := 5,
  group2_without_connections := 10
}

end NUMINAMATH_CALUDE_handshakes_in_specific_gathering_l2466_246623


namespace NUMINAMATH_CALUDE_parallel_iff_plane_intersects_parallel_transitive_l2466_246679

-- Define the concept of a line in 3D space
def Line : Type := ℝ × ℝ × ℝ → Prop

-- Define the concept of a plane in 3D space
def Plane : Type := ℝ × ℝ × ℝ → Prop

-- Define parallelism for lines
def parallel (a b : Line) : Prop := sorry

-- Define intersection between a plane and a line
def intersects (p : Plane) (l : Line) : Prop := sorry

-- Define unique intersection
def uniqueIntersection (p : Plane) (l : Line) : Prop := sorry

theorem parallel_iff_plane_intersects (a b : Line) : 
  parallel a b ↔ ∀ (p : Plane), intersects p a → uniqueIntersection p b := by sorry

theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end NUMINAMATH_CALUDE_parallel_iff_plane_intersects_parallel_transitive_l2466_246679


namespace NUMINAMATH_CALUDE_friendly_numbers_solution_l2466_246664

/-- Two rational numbers are friendly if their sum is 66 -/
def friendly (m n : ℚ) : Prop := m + n = 66

/-- Given that 7x and -18 are friendly numbers, prove that x = 12 -/
theorem friendly_numbers_solution : 
  ∀ x : ℚ, friendly (7 * x) (-18) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_friendly_numbers_solution_l2466_246664


namespace NUMINAMATH_CALUDE_flagpole_height_l2466_246655

/-- Given a flagpole that breaks and folds over in half, with its tip 2 feet above the ground
    and the break point 7 feet from the base, prove that its original height was 16 feet. -/
theorem flagpole_height (H : ℝ) : 
  (H - 7 - 2 = 7) →  -- The folded part equals the standing part
  (H = 16) :=
by sorry

end NUMINAMATH_CALUDE_flagpole_height_l2466_246655


namespace NUMINAMATH_CALUDE_chips_price_calculation_l2466_246620

/-- Given a discount and a final price, calculate the original price --/
def original_price (discount : ℝ) (final_price : ℝ) : ℝ :=
  discount + final_price

theorem chips_price_calculation :
  let discount : ℝ := 17
  let final_price : ℝ := 18
  original_price discount final_price = 35 := by
sorry

end NUMINAMATH_CALUDE_chips_price_calculation_l2466_246620


namespace NUMINAMATH_CALUDE_arm_wrestling_tournament_rounds_l2466_246688

/-- Represents the rules and structure of the arm wrestling tournament. -/
structure TournamentRules where
  num_athletes : ℕ
  max_point_diff : ℕ

/-- Calculates the minimum number of rounds required to determine a sole leader. -/
def min_rounds_required (rules : TournamentRules) : ℕ :=
  sorry

/-- Theorem stating that for a tournament with 510 athletes and the given rules,
    the minimum number of rounds required is 9. -/
theorem arm_wrestling_tournament_rounds 
  (rules : TournamentRules) 
  (h1 : rules.num_athletes = 510) 
  (h2 : rules.max_point_diff = 1) : 
  min_rounds_required rules = 9 := by
  sorry

end NUMINAMATH_CALUDE_arm_wrestling_tournament_rounds_l2466_246688


namespace NUMINAMATH_CALUDE_football_team_handedness_l2466_246638

theorem football_team_handedness (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 31)
  (h3 : right_handed = 57)
  (h4 : throwers ≤ right_handed) : 
  (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_football_team_handedness_l2466_246638


namespace NUMINAMATH_CALUDE_supermarket_spending_l2466_246672

theorem supermarket_spending (total : ℚ) : 
  (2/5 : ℚ) * total + (1/4 : ℚ) * total + (1/10 : ℚ) * total + 
  (1/8 : ℚ) * total + (1/20 : ℚ) * total + 12 = total → 
  total = 160 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l2466_246672


namespace NUMINAMATH_CALUDE_equation_not_equivalent_to_expression_with_unknown_l2466_246626

-- Define what an expression is
def Expression : Type := Unit

-- Define what an unknown is
def Unknown : Type := Unit

-- Define what an equation is
def Equation : Type := Unit

-- Define a property for expressions that contain unknowns
def contains_unknown (e : Expression) : Prop := sorry

-- Define the property that an equation contains unknowns
axiom equation_contains_unknown : ∀ (eq : Equation), ∃ (u : Unknown), contains_unknown eq

-- Theorem to prove
theorem equation_not_equivalent_to_expression_with_unknown : 
  ¬(∀ (e : Expression), contains_unknown e → ∃ (eq : Equation), e = eq) :=
sorry

end NUMINAMATH_CALUDE_equation_not_equivalent_to_expression_with_unknown_l2466_246626


namespace NUMINAMATH_CALUDE_cistern_filling_time_l2466_246607

theorem cistern_filling_time (capacity : ℝ) (fill_time : ℝ) (empty_time : ℝ) :
  fill_time = 10 →
  empty_time = 15 →
  (capacity / fill_time - capacity / empty_time) * (fill_time * empty_time / (empty_time - fill_time)) = capacity :=
by
  sorry

#check cistern_filling_time

end NUMINAMATH_CALUDE_cistern_filling_time_l2466_246607


namespace NUMINAMATH_CALUDE_solar_project_profit_l2466_246685

/-- Represents the net profit of a solar power generation project -/
def net_profit (n : ℕ+) : ℤ :=
  n - (4 * n^2 + 20 * n) - 144

/-- Theorem stating the net profit expression and when the project starts to make profit -/
theorem solar_project_profit :
  (∀ n : ℕ+, net_profit n = -4 * n^2 + 80 * n - 144) ∧
  (∀ n : ℕ+, net_profit n > 0 ↔ n ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_solar_project_profit_l2466_246685


namespace NUMINAMATH_CALUDE_units_digit_of_five_consecutive_integers_l2466_246642

theorem units_digit_of_five_consecutive_integers (n : ℕ) : 
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10 = 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_five_consecutive_integers_l2466_246642


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l2466_246692

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Predicate to check if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola a b) (p : ℝ × ℝ) : Prop :=
  (p.1^2 / a^2) - (p.2^2 / b^2) = 1

/-- Predicate to check if four points form a parallelogram -/
def is_parallelogram (p q r s : ℝ × ℝ) : Prop := sorry

/-- The area of a quadrilateral given by four points -/
def quadrilateral_area (p q r s : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity_theorem (a b c : ℝ) (h : Hyperbola a b) 
  (m n : ℝ × ℝ) (hm : on_hyperbola h m) (hn : on_hyperbola h n)
  (hpara : is_parallelogram (0, 0) (right_focus h) m n)
  (harea : quadrilateral_area (0, 0) (right_focus h) m n = Real.sqrt 3 * b * c) :
  eccentricity h = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l2466_246692


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2466_246663

theorem complex_on_imaginary_axis (z : ℂ) : 
  Complex.abs (z - 1) = Complex.abs (z + 1) → z.re = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2466_246663


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2466_246609

theorem other_root_of_quadratic (x : ℚ) :
  (7 * x^2 - 3 * x = 10) ∧ (7 * (-2)^2 - 3 * (-2) = 10) →
  (7 * (5/7)^2 - 3 * (5/7) = 10) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2466_246609


namespace NUMINAMATH_CALUDE_medicine_price_reduction_l2466_246627

theorem medicine_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 50)
  (h2 : final_price = 32)
  (h3 : initial_price > 0)
  (h4 : final_price > 0)
  (h5 : final_price < initial_price) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price :=
by
  sorry

end NUMINAMATH_CALUDE_medicine_price_reduction_l2466_246627


namespace NUMINAMATH_CALUDE_parabola_equation_l2466_246686

/-- A parabola with focus F and point A on the curve, where |FA| is the radius of a circle
    intersecting the parabola's axis at B and C, forming an equilateral triangle FBC. -/
structure ParabolaWithTriangle where
  -- The parameter of the parabola
  p : ℝ
  -- The coordinates of points A, B, C, and F
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  F : ℝ × ℝ

/-- Properties of the parabola and associated triangle -/
def ParabolaProperties (P : ParabolaWithTriangle) : Prop :=
  -- A lies on the parabola y^2 = 2px
  P.A.2^2 = 2 * P.p * P.A.1 ∧
  -- F is the focus (p/2, 0)
  P.F = (P.p/2, 0) ∧
  -- B and C are on the x-axis
  P.B.2 = 0 ∧ P.C.2 = 0 ∧
  -- |FA| = |FB| = |FC|
  (P.A.1 - P.F.1)^2 + (P.A.2 - P.F.2)^2 = 
  (P.B.1 - P.F.1)^2 + (P.B.2 - P.F.2)^2 ∧
  (P.A.1 - P.F.1)^2 + (P.A.2 - P.F.2)^2 = 
  (P.C.1 - P.F.1)^2 + (P.C.2 - P.F.2)^2 ∧
  -- Area of triangle ABC is 128/3
  abs ((P.A.1 - P.C.1) * (P.B.2 - P.C.2) - (P.B.1 - P.C.1) * (P.A.2 - P.C.2)) / 2 = 128/3

theorem parabola_equation (P : ParabolaWithTriangle) 
  (h : ParabolaProperties P) : P.p = 8 ∧ ∀ (x y : ℝ), y^2 = 16*x ↔ y^2 = 2*P.p*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2466_246686


namespace NUMINAMATH_CALUDE_intersection_line_proof_l2466_246682

/-- Given two lines in the plane and a slope, prove that a certain line passes through their intersection point with the given slope. -/
theorem intersection_line_proof (x y : ℝ) : 
  (3 * x + 4 * y = 5) →  -- First line equation
  (3 * x - 4 * y = 13) →  -- Second line equation
  (∃ (x₀ y₀ : ℝ), (3 * x₀ + 4 * y₀ = 5) ∧ (3 * x₀ - 4 * y₀ = 13) ∧ (2 * x₀ - y₀ = 7)) ∧  -- Intersection point exists and satisfies all equations
  (∀ (x₁ y₁ : ℝ), (2 * x₁ - y₁ = 7) → (y₁ - y) / (x₁ - x) = 2 ∨ x₁ = x)  -- Slope of the line 2x - y - 7 = 0 is 2
  := by sorry

end NUMINAMATH_CALUDE_intersection_line_proof_l2466_246682


namespace NUMINAMATH_CALUDE_prob_each_class_one_student_prob_at_least_one_empty_class_prob_exactly_one_empty_class_l2466_246649

/-- The number of newly transferred students -/
def num_students : ℕ := 4

/-- The number of designated classes -/
def num_classes : ℕ := 4

/-- The total number of ways to distribute students into classes -/
def total_distributions : ℕ := num_classes ^ num_students

/-- The number of ways to distribute students such that each class receives one student -/
def each_class_one_student : ℕ := Nat.factorial num_classes

/-- The probability that each class receives one student -/
theorem prob_each_class_one_student :
  (each_class_one_student : ℚ) / total_distributions = 3 / 32 := by sorry

/-- The probability that at least one class does not receive any students -/
theorem prob_at_least_one_empty_class :
  1 - (each_class_one_student : ℚ) / total_distributions = 29 / 32 := by sorry

/-- The number of ways to distribute students such that exactly one class is empty -/
def exactly_one_empty_class : ℕ :=
  (num_classes.choose 1) * (num_classes.choose 2) * ((num_classes - 1).choose 1) * ((num_classes - 2).choose 1)

/-- The probability that exactly one class does not receive any students -/
theorem prob_exactly_one_empty_class :
  (exactly_one_empty_class : ℚ) / total_distributions = 9 / 16 := by sorry

end NUMINAMATH_CALUDE_prob_each_class_one_student_prob_at_least_one_empty_class_prob_exactly_one_empty_class_l2466_246649


namespace NUMINAMATH_CALUDE_homogeneous_polynomial_terms_l2466_246675

/-- The number of distinct terms in a homogeneous polynomial -/
def num_distinct_terms (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: The number of distinct terms in a homogeneous polynomial of degree 6 with 5 variables is 210 -/
theorem homogeneous_polynomial_terms :
  num_distinct_terms 6 5 = 210 := by sorry

end NUMINAMATH_CALUDE_homogeneous_polynomial_terms_l2466_246675


namespace NUMINAMATH_CALUDE_equal_probability_wsw_more_advantageous_l2466_246673

-- Define the probabilities of winning against strong and weak players
variable (Ps Pw : ℝ)

-- Define the condition that Ps < Pw
variable (h : Ps < Pw)

-- Define the probability of winning two consecutive games in the sequence Strong, Weak, Strong
def prob_sws : ℝ := Ps * Pw

-- Define the probability of winning two consecutive games in the sequence Weak, Strong, Weak
def prob_wsw : ℝ := Pw * Ps

-- Theorem stating that both sequences have equal probability
theorem equal_probability : prob_sws Ps Pw = prob_wsw Ps Pw := by
  sorry

-- Theorem stating that Weak, Strong, Weak is more advantageous
theorem wsw_more_advantageous (h : Ps < Pw) : prob_wsw Ps Pw ≥ prob_sws Ps Pw := by
  sorry

end NUMINAMATH_CALUDE_equal_probability_wsw_more_advantageous_l2466_246673


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l2466_246657

theorem polynomial_sum_theorem (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^10 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  (a₀ + a₁) + (a₀ + a₂) + (a₀ + a₃) + (a₀ + a₄) + (a₀ + a₅) + (a₀ + a₆) + (a₀ + a₇) + (a₀ + a₈) + (a₀ + a₉) + (a₀ + a₁₀) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l2466_246657


namespace NUMINAMATH_CALUDE_james_savings_l2466_246661

/-- Proves that James saved for 4 weeks given the problem conditions --/
theorem james_savings (w : ℕ) : 
  (10 : ℚ) * w - ((10 : ℚ) * w / 2) / 4 = 15 → w = 4 := by
  sorry

end NUMINAMATH_CALUDE_james_savings_l2466_246661


namespace NUMINAMATH_CALUDE_socks_theorem_l2466_246681

/-- The number of pairs of socks Niko bought -/
def total_socks : ℕ := 9

/-- The cost of each pair of socks in dollars -/
def cost_per_pair : ℚ := 2

/-- The number of pairs with 25% profit -/
def pairs_with_25_percent : ℕ := 4

/-- The number of pairs with $0.2 profit -/
def pairs_with_20_cents : ℕ := 5

/-- The total profit in dollars -/
def total_profit : ℚ := 3

/-- The profit percentage for the first group of socks -/
def profit_percentage : ℚ := 25 / 100

/-- The profit amount for the second group of socks in dollars -/
def profit_amount : ℚ := 1 / 5

theorem socks_theorem :
  total_socks = pairs_with_25_percent + pairs_with_20_cents ∧
  total_profit = pairs_with_25_percent * (cost_per_pair * profit_percentage) +
                 pairs_with_20_cents * profit_amount :=
by sorry

end NUMINAMATH_CALUDE_socks_theorem_l2466_246681


namespace NUMINAMATH_CALUDE_solution_set_a_neg_one_range_of_a_l2466_246639

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |x + 3|

-- Part 1: Solution set when a = -1
theorem solution_set_a_neg_one :
  {x : ℝ | f (-1) x ≤ 1} = {x : ℝ | x ≥ -5/2} := by sorry

-- Part 2: Range of a when f(x) ≤ 4 for all x ∈ [0,3]
theorem range_of_a :
  {a : ℝ | ∀ x ∈ Set.Icc 0 3, f a x ≤ 4} = Set.Icc (-7) 7 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_neg_one_range_of_a_l2466_246639


namespace NUMINAMATH_CALUDE_det_of_specific_matrix_l2466_246697

theorem det_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![5, -4; 2, 3]
  Matrix.det A = 23 := by
sorry

end NUMINAMATH_CALUDE_det_of_specific_matrix_l2466_246697


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_19_l2466_246614

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is four-digit -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- Theorem stating that 9730 is the largest four-digit number whose digits add up to 19 -/
theorem largest_four_digit_sum_19 : 
  (∀ n : ℕ, is_four_digit n → sum_of_digits n = 19 → n ≤ 9730) ∧ 
  is_four_digit 9730 ∧ 
  sum_of_digits 9730 = 19 := by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_19_l2466_246614


namespace NUMINAMATH_CALUDE_line_plane_relationship_l2466_246621

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using two points or a point and a direction vector
  -- This is a simplified representation
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  -- This is a simplified representation
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  -- Definition of parallel lines
  sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Definition of a line parallel to a plane
  sorry

/-- A line is a subset of a plane -/
def line_subset_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Definition of a line being a subset of a plane
  sorry

theorem line_plane_relationship (m n : Line3D) (α : Plane3D) 
  (h1 : parallel_lines m n) (h2 : line_parallel_plane m α) :
  line_parallel_plane n α ∨ line_subset_plane n α := by
  sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l2466_246621


namespace NUMINAMATH_CALUDE_parabola_directrix_l2466_246617

/-- Given a parabola with equation x^2 = -1/8 * y, its directrix equation is y = 1/32 -/
theorem parabola_directrix (x y : ℝ) : 
  (x^2 = -1/8 * y) → (∃ (k : ℝ), k = 1/32 ∧ k = y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2466_246617


namespace NUMINAMATH_CALUDE_duck_percentage_among_non_swans_l2466_246616

theorem duck_percentage_among_non_swans 
  (duck_percent : ℝ) 
  (swan_percent : ℝ) 
  (eagle_percent : ℝ) 
  (sparrow_percent : ℝ) 
  (h1 : duck_percent = 40)
  (h2 : swan_percent = 20)
  (h3 : eagle_percent = 15)
  (h4 : sparrow_percent = 25)
  (h5 : duck_percent + swan_percent + eagle_percent + sparrow_percent = 100) :
  (duck_percent / (100 - swan_percent)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_duck_percentage_among_non_swans_l2466_246616


namespace NUMINAMATH_CALUDE_fish_problem_l2466_246632

/-- The number of fish Ken and Kendra brought home -/
def total_fish_brought_home (ken_caught : ℕ) (ken_released : ℕ) (kendra_caught : ℕ) : ℕ :=
  (ken_caught - ken_released) + kendra_caught

/-- Theorem stating the total number of fish brought home by Ken and Kendra -/
theorem fish_problem :
  ∀ (ken_caught : ℕ) (kendra_caught : ℕ),
    ken_caught = 2 * kendra_caught →
    kendra_caught = 30 →
    total_fish_brought_home ken_caught 3 kendra_caught = 87 := by
  sorry


end NUMINAMATH_CALUDE_fish_problem_l2466_246632


namespace NUMINAMATH_CALUDE_square_pyramid_frustum_volume_ratio_is_correct_l2466_246648

def square_pyramid_frustum_volume_ratio : ℚ :=
  let base_edge : ℚ := 24
  let altitude : ℚ := 10
  let small_pyramid_altitude_ratio : ℚ := 1/3
  
  let original_volume : ℚ := (1/3) * base_edge^2 * altitude
  let small_pyramid_base_edge : ℚ := base_edge * small_pyramid_altitude_ratio
  let small_pyramid_volume : ℚ := (1/3) * small_pyramid_base_edge^2 * (altitude * small_pyramid_altitude_ratio)
  let frustum_volume : ℚ := original_volume - small_pyramid_volume
  
  frustum_volume / original_volume

theorem square_pyramid_frustum_volume_ratio_is_correct :
  square_pyramid_frustum_volume_ratio = 924/960 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_frustum_volume_ratio_is_correct_l2466_246648


namespace NUMINAMATH_CALUDE_original_group_size_l2466_246634

theorem original_group_size (original_days : ℕ) (absent_men : ℕ) (new_days : ℕ) :
  original_days = 6 →
  absent_men = 4 →
  new_days = 12 →
  ∃ (total_men : ℕ), 
    total_men > absent_men ∧
    (1 : ℚ) / (original_days * total_men) = (1 : ℚ) / (new_days * (total_men - absent_men)) ∧
    total_men = 8 := by
  sorry

end NUMINAMATH_CALUDE_original_group_size_l2466_246634


namespace NUMINAMATH_CALUDE_total_writing_instruments_l2466_246625

theorem total_writing_instruments (pens pencils markers : ℕ) : 
  (5 * pens = 6 * pencils - 54) →  -- Ratio of pens to pencils is 5:6, and 9 more pencils
  (4 * pencils = 3 * markers) →    -- Ratio of markers to pencils is 4:3
  pens + pencils + markers = 171   -- Total number of writing instruments
  := by sorry

end NUMINAMATH_CALUDE_total_writing_instruments_l2466_246625


namespace NUMINAMATH_CALUDE_marble_241_is_blue_l2466_246654

/-- Represents the color of a marble -/
inductive MarbleColor
| Blue
| Red
| Green

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 14 with
  | 0 | 1 | 2 | 3 | 4 | 5 => MarbleColor.Blue
  | 6 | 7 | 8 | 9 | 10 => MarbleColor.Red
  | _ => MarbleColor.Green

/-- Theorem: The 241st marble in the sequence is blue -/
theorem marble_241_is_blue : marbleColor 241 = MarbleColor.Blue := by
  sorry

end NUMINAMATH_CALUDE_marble_241_is_blue_l2466_246654


namespace NUMINAMATH_CALUDE_first_team_cups_l2466_246629

theorem first_team_cups (total required : ℕ) (second_team : ℕ) (third_team : ℕ) 
  (h1 : total = 280)
  (h2 : second_team = 120)
  (h3 : third_team = 70)
  : total - (second_team + third_team) = 90 := by
  sorry

end NUMINAMATH_CALUDE_first_team_cups_l2466_246629


namespace NUMINAMATH_CALUDE_infinite_squares_in_progression_l2466_246687

/-- An arithmetic progression with positive integer members -/
structure ArithmeticProgression where
  a : ℕ+  -- First term
  d : ℕ+  -- Common difference

/-- Predicate to check if a number is in the arithmetic progression -/
def inProgression (ap : ArithmeticProgression) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = ap.a + k * ap.d

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem infinite_squares_in_progression (ap : ArithmeticProgression) :
  (∃ n : ℕ, inProgression ap n ∧ isPerfectSquare n) →
  (∀ N : ℕ, ∃ n : ℕ, n > N ∧ inProgression ap n ∧ isPerfectSquare n) :=
sorry

end NUMINAMATH_CALUDE_infinite_squares_in_progression_l2466_246687


namespace NUMINAMATH_CALUDE_fifteen_shaded_cubes_l2466_246630

/-- Represents a 3x3x3 cube constructed from smaller cubes -/
structure LargeCube where
  size : Nat
  total_cubes : Nat
  shaded_per_face : Nat

/-- Calculates the number of uniquely shaded cubes in the large cube -/
def count_shaded_cubes (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating that the number of uniquely shaded cubes is 15 -/
theorem fifteen_shaded_cubes (cube : LargeCube) 
  (h1 : cube.size = 3) 
  (h2 : cube.total_cubes = 27) 
  (h3 : cube.shaded_per_face = 3) : 
  count_shaded_cubes cube = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_shaded_cubes_l2466_246630


namespace NUMINAMATH_CALUDE_geometric_series_relation_l2466_246613

/-- Given real numbers x and y satisfying an infinite geometric series equation,
    prove that another related infinite geometric series has a specific value. -/
theorem geometric_series_relation (x y : ℝ) 
  (h : (x / y) / (1 - 1 / y) = 3) :
  (x / (x + 2 * y)) / (1 - 1 / (x + 2 * y)) = 3 * (y - 1) / (5 * y - 4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l2466_246613


namespace NUMINAMATH_CALUDE_monotonic_quadratic_condition_l2466_246693

/-- A function f is monotonic on an interval [a, b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The quadratic function f(x) = x^2 - 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem monotonic_quadratic_condition :
  ∀ a : ℝ, IsMonotonic (f a) 2 3 ↔ (a ≤ 2 ∨ a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_condition_l2466_246693


namespace NUMINAMATH_CALUDE_rectangle_height_from_square_perimeter_l2466_246602

theorem rectangle_height_from_square_perimeter (square_side : ℝ) (rect_width : ℝ) :
  square_side = 20 →
  rect_width = 14 →
  4 * square_side = 2 * (rect_width + (80 - 2 * rect_width) / 2) →
  (80 - 2 * rect_width) / 2 = 26 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_height_from_square_perimeter_l2466_246602


namespace NUMINAMATH_CALUDE_equation_solution_l2466_246603

theorem equation_solution (x : ℚ) (h : x ≠ 3) : (x + 5) / (x - 3) = 4 ↔ x = 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2466_246603


namespace NUMINAMATH_CALUDE_ali_baba_treasure_max_value_l2466_246652

/-- The maximum value problem for Ali Baba's treasure --/
theorem ali_baba_treasure_max_value :
  let f : ℝ → ℝ → ℝ := λ x y => 20 * x + 60 * y
  let S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 ≤ 100 ∧ p.1 + 5 * p.2 ≤ 200}
  ∃ (x y : ℝ), (x, y) ∈ S ∧ f x y = 3000 ∧ ∀ (x' y' : ℝ), (x', y') ∈ S → f x' y' ≤ 3000 :=
by sorry


end NUMINAMATH_CALUDE_ali_baba_treasure_max_value_l2466_246652


namespace NUMINAMATH_CALUDE_negation_of_union_membership_l2466_246643

theorem negation_of_union_membership (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B :=
by sorry

end NUMINAMATH_CALUDE_negation_of_union_membership_l2466_246643


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2466_246601

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 2

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - 3*y + 3 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (A B : ℝ × ℝ),
    C A.1 A.2 ∧ C B.1 B.2 ∧
    (∀ (x y : ℝ), C x y → (x - P.1)*(y - P.2) = (A.1 - P.1)*(A.2 - P.2) ∨ (x - P.1)*(y - P.2) = (B.1 - P.1)*(B.2 - P.2)) →
    (∀ (x y : ℝ), line_equation x y ↔ (∃ t : ℝ, x = A.1 + t*(B.1 - A.1) ∧ y = A.2 + t*(B.2 - A.2))) :=
sorry


end NUMINAMATH_CALUDE_tangent_line_equation_l2466_246601


namespace NUMINAMATH_CALUDE_total_curve_length_is_6pi_l2466_246633

/-- Regular tetrahedron with edge length 4 -/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (is_regular : edge_length = 4)

/-- Point on the surface of the tetrahedron -/
structure SurfacePoint (t : RegularTetrahedron) :=
  (distance_from_vertex : ℝ)
  (on_surface : distance_from_vertex = 3)

/-- Total length of curve segments -/
def total_curve_length (t : RegularTetrahedron) (p : SurfacePoint t) : ℝ := sorry

/-- Theorem: The total length of curve segments is 6π -/
theorem total_curve_length_is_6pi (t : RegularTetrahedron) (p : SurfacePoint t) :
  total_curve_length t p = 6 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_total_curve_length_is_6pi_l2466_246633


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2466_246622

def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ StrictMono f := by sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2466_246622


namespace NUMINAMATH_CALUDE_vasya_petya_notebooks_different_l2466_246674

theorem vasya_petya_notebooks_different (S : Finset ℝ) (h : S.card = 10) :
  let vasya_set := Finset.image (fun (p : ℝ × ℝ) => (p.1 - p.2)^2) (S.product S)
  let petya_set := Finset.image (fun (p : ℝ × ℝ) => |p.1^2 - p.2^2|) (S.product S)
  vasya_set ≠ petya_set :=
by sorry

end NUMINAMATH_CALUDE_vasya_petya_notebooks_different_l2466_246674


namespace NUMINAMATH_CALUDE_total_savings_calculation_l2466_246699

-- Define the original prices and discount rates
def chlorine_price : ℝ := 10
def chlorine_discount : ℝ := 0.20
def soap_price : ℝ := 16
def soap_discount : ℝ := 0.25

-- Define the quantities
def chlorine_quantity : ℕ := 3
def soap_quantity : ℕ := 5

-- Theorem statement
theorem total_savings_calculation :
  let chlorine_savings := chlorine_price * chlorine_discount * chlorine_quantity
  let soap_savings := soap_price * soap_discount * soap_quantity
  chlorine_savings + soap_savings = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_calculation_l2466_246699


namespace NUMINAMATH_CALUDE_bookcase_sum_l2466_246660

theorem bookcase_sum (a₁ : ℕ) (d : ℤ) (n : ℕ) (aₙ : ℕ) : 
  a₁ = 32 → 
  d = -3 → 
  aₙ > 0 → 
  aₙ = a₁ + (n - 1) * d → 
  n * (a₁ + aₙ) = 374 → 
  (n : ℤ) * (2 * a₁ + (n - 1) * d) = 374 :=
by sorry

end NUMINAMATH_CALUDE_bookcase_sum_l2466_246660


namespace NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l2466_246631

theorem negation_of_all_squares_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l2466_246631


namespace NUMINAMATH_CALUDE_solve_laptop_battery_problem_l2466_246646

def laptop_battery_problem (standby_capacity : ℝ) (gaming_capacity : ℝ) 
  (standby_used : ℝ) (gaming_used : ℝ) : Prop :=
  standby_capacity = 10 ∧ 
  gaming_capacity = 2 ∧ 
  standby_used = 4 ∧ 
  gaming_used = 1 ∧ 
  (1 - (standby_used / standby_capacity + gaming_used / gaming_capacity)) * standby_capacity = 1

theorem solve_laptop_battery_problem :
  ∀ standby_capacity gaming_capacity standby_used gaming_used,
  laptop_battery_problem standby_capacity gaming_capacity standby_used gaming_used := by
  sorry

end NUMINAMATH_CALUDE_solve_laptop_battery_problem_l2466_246646
