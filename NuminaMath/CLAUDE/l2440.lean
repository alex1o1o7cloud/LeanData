import Mathlib

namespace NUMINAMATH_CALUDE_valid_arrangements_count_l2440_244032

/-- The number of ways to arrange plates around a circular table. -/
def arrange_plates (blue red green orange : ℕ) : ℕ :=
  sorry

/-- The number of valid arrangements of plates. -/
def valid_arrangements : ℕ :=
  arrange_plates 5 3 2 1

/-- Theorem stating the correct number of valid arrangements. -/
theorem valid_arrangements_count : valid_arrangements = 361 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l2440_244032


namespace NUMINAMATH_CALUDE_father_age_three_times_l2440_244046

/-- Marika's birth year -/
def marika_birth_year : ℕ := 1996

/-- The year when Marika's father's age was five times her age -/
def reference_year : ℕ := 2006

/-- Marika's father's age is five times her age in the reference year -/
axiom father_age_five_times (y : ℕ) : y = reference_year → 
  5 * (y - marika_birth_year) = y - (marika_birth_year - 50)

/-- The year we're looking for -/
def target_year : ℕ := 2016

/-- Theorem: In the target year, Marika's father's age will be three times her age -/
theorem father_age_three_times : 
  3 * (target_year - marika_birth_year) = target_year - (marika_birth_year - 50) :=
sorry

end NUMINAMATH_CALUDE_father_age_three_times_l2440_244046


namespace NUMINAMATH_CALUDE_square_side_length_l2440_244002

theorem square_side_length (diagonal : ℝ) (h : diagonal = Real.sqrt 2) : 
  ∃ (side : ℝ), side * side + side * side = diagonal * diagonal ∧ side = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2440_244002


namespace NUMINAMATH_CALUDE_paintable_area_is_1572_l2440_244039

/-- Calculate the total paintable wall area for multiple identical bedrooms -/
def total_paintable_area (num_rooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let total_wall_area := 2 * (length * height + width * height)
  let paintable_area_per_room := total_wall_area - unpaintable_area
  num_rooms * paintable_area_per_room

/-- Theorem stating that the total paintable wall area for the given conditions is 1572 square feet -/
theorem paintable_area_is_1572 :
  total_paintable_area 4 15 11 9 75 = 1572 := by
  sorry

#eval total_paintable_area 4 15 11 9 75

end NUMINAMATH_CALUDE_paintable_area_is_1572_l2440_244039


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2440_244023

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 → 
  (∃ m : ℕ, m > 12 ∧ ∀ k : ℕ, k > 0 → m ∣ (k * (k + 1) * (k + 2) * (k + 3))) → False :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2440_244023


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2440_244005

theorem complex_on_imaginary_axis (a : ℝ) :
  let z : ℂ := (a^2 - 2*a) + (a^2 - a - 2)*I
  (z.re = 0) ↔ (a = 2 ∨ a = 0) := by sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2440_244005


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l2440_244034

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two : 
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l2440_244034


namespace NUMINAMATH_CALUDE_james_distance_traveled_l2440_244060

/-- Calculates the distance traveled given speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: James' distance traveled -/
theorem james_distance_traveled :
  distance_traveled 80.0 16.0 = 1280.0 := by
  sorry

end NUMINAMATH_CALUDE_james_distance_traveled_l2440_244060


namespace NUMINAMATH_CALUDE_congruence_solution_l2440_244006

theorem congruence_solution (n : Int) : n ≡ 26 [ZMOD 47] ↔ 13 * n ≡ 9 [ZMOD 47] ∧ 0 ≤ n ∧ n < 47 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2440_244006


namespace NUMINAMATH_CALUDE_quadratic_max_l2440_244014

/-- The function f(x) = -2x^2 + 8x - 6 achieves its maximum value when x = 2 -/
theorem quadratic_max (x : ℝ) : 
  ∀ y : ℝ, -2 * x^2 + 8 * x - 6 ≥ -2 * y^2 + 8 * y - 6 ↔ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_l2440_244014


namespace NUMINAMATH_CALUDE_special_function_at_one_third_l2440_244027

/-- A function satisfying the given properties -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 1 = 1 ∧ ∀ x y : ℝ, g (x * y + g x) = x * g y + g x

/-- The main theorem -/
theorem special_function_at_one_third {g : ℝ → ℝ} (hg : special_function g) : 
  g (1/3) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_one_third_l2440_244027


namespace NUMINAMATH_CALUDE_problem_statement_l2440_244018

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- State the theorem
theorem problem_statement (m : ℝ) (a b c : ℝ) :
  (∀ x, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-1) 1) →
  a > 0 → b > 0 → c > 0 →
  1 / a + 1 / (2 * b) + 1 / (3 * c) = m →
  (m = 1 ∧ a + 2 * b + 3 * c ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2440_244018


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l2440_244040

def male_teachers : ℕ := 5
def female_teachers : ℕ := 4
def total_teachers : ℕ := male_teachers + female_teachers
def head_teachers_needed : ℕ := 3

def valid_arrangements : ℕ := 
  Nat.factorial total_teachers / Nat.factorial (total_teachers - head_teachers_needed) -
  (Nat.factorial male_teachers / Nat.factorial (male_teachers - head_teachers_needed) +
   Nat.factorial female_teachers / Nat.factorial (female_teachers - head_teachers_needed))

theorem valid_arrangements_count : valid_arrangements = 420 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l2440_244040


namespace NUMINAMATH_CALUDE_jeff_ninja_stars_l2440_244021

/-- The number of ninja throwing stars each person has -/
structure NinjaStars where
  eric : ℕ
  chad : ℕ
  jeff : ℕ

/-- The conditions of the problem -/
def ninja_star_problem (stars : NinjaStars) : Prop :=
  stars.eric = 4 ∧
  stars.chad = 2 * stars.eric ∧
  stars.eric + stars.chad + stars.jeff = 16 ∧
  stars.chad = (2 * stars.eric) - 2

theorem jeff_ninja_stars :
  ∃ (stars : NinjaStars), ninja_star_problem stars ∧ stars.jeff = 6 := by
  sorry

end NUMINAMATH_CALUDE_jeff_ninja_stars_l2440_244021


namespace NUMINAMATH_CALUDE_arrangement_count_l2440_244016

theorem arrangement_count :
  let teachers : ℕ := 3
  let students : ℕ := 6
  let groups : ℕ := 3
  let teachers_per_group : ℕ := 1
  let students_per_group : ℕ := 2
  
  (teachers.factorial * (students.choose students_per_group) * 
   ((students - students_per_group).choose students_per_group) * 
   ((students - 2 * students_per_group).choose students_per_group)) = 540 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l2440_244016


namespace NUMINAMATH_CALUDE_tan_45_degrees_l2440_244098

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l2440_244098


namespace NUMINAMATH_CALUDE_square_area_ratio_l2440_244013

/-- The ratio of the areas of two squares, where one has a side length 5 times the other, is 1/25. -/
theorem square_area_ratio (y : ℝ) (h : y > 0) : 
  (y^2) / ((5*y)^2) = 1 / 25 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2440_244013


namespace NUMINAMATH_CALUDE_existence_of_m_satisfying_inequality_l2440_244038

theorem existence_of_m_satisfying_inequality (a t : ℝ) 
  (ha : a ∈ Set.Icc (-1 : ℝ) 1)
  (ht : t ∈ Set.Icc (-1 : ℝ) 1) :
  ∃ m : ℝ, (∀ x₁ x₂ : ℝ, 
    x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ 
    (4 * x₁ + a * x₁^2 - (2/3) * x₁^3 = 2 * x₁ + (1/3) * x₁^3) ∧
    (4 * x₂ + a * x₂^2 - (2/3) * x₂^3 = 2 * x₂ + (1/3) * x₂^3) →
    m^2 + t * m + 1 ≥ |x₁ - x₂|) ∧
  (m ≥ 2 ∨ m ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_satisfying_inequality_l2440_244038


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l2440_244008

/-- The distance between the vertices of the hyperbola x^2/16 - y^2/9 = 1 is 8 -/
theorem hyperbola_vertices_distance : 
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/16 - y^2/9 = 1
  ∃ (v₁ v₂ : ℝ × ℝ), 
    (h v₁.1 v₁.2 ∧ h v₂.1 v₂.2) ∧ 
    (v₁.2 = 0 ∧ v₂.2 = 0) ∧
    (v₁.1 = -v₂.1) ∧
    abs (v₁.1 - v₂.1) = 8 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l2440_244008


namespace NUMINAMATH_CALUDE_complex_on_real_axis_l2440_244010

theorem complex_on_real_axis (a : ℝ) : 
  let z : ℂ := (a - Complex.I) * (1 + Complex.I)
  (z.im = 0) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_on_real_axis_l2440_244010


namespace NUMINAMATH_CALUDE_digit_sum_of_predecessor_l2440_244099

/-- Represents a natural number as a list of its digits -/
def Digits := List Nat

/-- Returns true if all elements in the list are distinct -/
def allDistinct (l : Digits) : Prop := ∀ i j, i ≠ j → l.get? i ≠ l.get? j

/-- Calculates the sum of all elements in the list -/
def digitSum (l : Digits) : Nat := l.sum

/-- Converts a natural number to its digit representation -/
def toDigits (n : Nat) : Digits := sorry

/-- Converts a digit representation back to a natural number -/
def fromDigits (d : Digits) : Nat := sorry

theorem digit_sum_of_predecessor (n : Nat) :
  (∃ d : Digits, fromDigits d = n ∧ allDistinct d ∧ digitSum d = 44) →
  (∃ d' : Digits, fromDigits d' = n - 1 ∧ (digitSum d' = 43 ∨ digitSum d' = 52)) := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_of_predecessor_l2440_244099


namespace NUMINAMATH_CALUDE_inequality_proof_l2440_244064

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2440_244064


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2440_244078

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) →  -- perimeter is 40 units
  (l * w ≤ 100) -- area is at most 100 square units
:= by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2440_244078


namespace NUMINAMATH_CALUDE_solve_for_q_l2440_244004

theorem solve_for_q (p q : ℝ) (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 1) (h4 : p*q = 9) :
  q = (9 + 3*Real.sqrt 5) / 2 := by sorry

end NUMINAMATH_CALUDE_solve_for_q_l2440_244004


namespace NUMINAMATH_CALUDE_election_majority_proof_l2440_244036

theorem election_majority_proof (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 450 → 
  winning_percentage = 70 / 100 → 
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 180 := by
sorry

end NUMINAMATH_CALUDE_election_majority_proof_l2440_244036


namespace NUMINAMATH_CALUDE_sam_fish_count_l2440_244019

/-- Represents the number of fish a person has -/
structure FishCount where
  goldfish : ℕ
  guppies : ℕ
  angelfish : ℕ

def Lilly : FishCount :=
  { goldfish := 10, guppies := 15, angelfish := 0 }

def Rosy : FishCount :=
  { goldfish := 12, guppies := 8, angelfish := 5 }

def Sam : FishCount :=
  { goldfish := Rosy.goldfish - 3, guppies := 2 * Lilly.guppies, angelfish := 0 }

def guppiesTransferred : ℕ := Lilly.guppies / 2

def LillyAfterTransfer : FishCount :=
  { Lilly with guppies := Lilly.guppies - guppiesTransferred }

def SamAfterTransfer : FishCount :=
  { Sam with guppies := Sam.guppies + guppiesTransferred }

def totalFish (fc : FishCount) : ℕ :=
  fc.goldfish + fc.guppies + fc.angelfish

theorem sam_fish_count :
  totalFish SamAfterTransfer = 46 := by sorry

end NUMINAMATH_CALUDE_sam_fish_count_l2440_244019


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l2440_244053

/-- Proves that if 20% of students took more than $100 on a camping trip, and 75% of students who went on the trip took $100 or less, then 80% of all students went on the camping trip. -/
theorem camping_trip_percentage (total_students : ℕ) (students_over_100 : ℕ) (students_on_trip : ℕ) :
  students_over_100 = (20 : ℕ) * total_students / 100 →
  (75 : ℕ) * students_on_trip / 100 = students_on_trip - students_over_100 →
  students_on_trip = (80 : ℕ) * total_students / 100 :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l2440_244053


namespace NUMINAMATH_CALUDE_prob_three_correct_five_l2440_244057

/-- The number of houses and packages --/
def n : ℕ := 5

/-- The probability of exactly 3 out of n packages being delivered to the correct houses --/
def prob_three_correct (n : ℕ) : ℚ :=
  (n.choose 3 : ℚ) * (1 / n) * (1 / (n - 1)) * (1 / (n - 2)) * (1 / 2)

/-- Theorem stating that the probability of exactly 3 out of 5 packages 
    being delivered to the correct houses is 1/12 --/
theorem prob_three_correct_five : prob_three_correct n = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_correct_five_l2440_244057


namespace NUMINAMATH_CALUDE_vector_sum_l2440_244065

def vector_a : ℝ × ℝ := (2, 0)
def vector_b : ℝ × ℝ := (-1, -2)

theorem vector_sum : vector_a + vector_b = (1, -2) := by sorry

end NUMINAMATH_CALUDE_vector_sum_l2440_244065


namespace NUMINAMATH_CALUDE_circle_center_first_quadrant_l2440_244058

theorem circle_center_first_quadrant (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*m*x + (2*m - 2)*y + 2*m^2 = 0 →
    ∃ r : ℝ, (x - m)^2 + (y - (1 - m))^2 = r^2) →
  (m > 0 ∧ 1 - m > 0) →
  0 < m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_first_quadrant_l2440_244058


namespace NUMINAMATH_CALUDE_number_of_divisors_180_l2440_244063

theorem number_of_divisors_180 : ∃ (n : ℕ), n = 18 ∧ 
  (∀ d : ℕ, d > 0 ∧ (180 % d = 0) ↔ d ∈ Finset.range n) :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_180_l2440_244063


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2440_244030

theorem imaginary_part_of_z (z : ℂ) (h : (z - Complex.I) * (1 + 2 * Complex.I) = Complex.I ^ 3) :
  z.im = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2440_244030


namespace NUMINAMATH_CALUDE_focus_coordinates_l2440_244085

/-- A parabola with equation x^2 = 2py where p > 0 -/
structure Parabola where
  p : ℝ
  p_pos : p > 0

/-- The directrix of a parabola -/
def directrix (par : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y = -2

/-- The focus of a parabola -/
def focus (par : Parabola) : ℝ × ℝ :=
  (0, par.p)

/-- Theorem stating that if the directrix of a parabola passes through (0, -2),
    then its focus is at (0, 2) -/
theorem focus_coordinates (par : Parabola) :
  directrix par 0 (-2) → focus par = (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_focus_coordinates_l2440_244085


namespace NUMINAMATH_CALUDE_sum_first_six_multiples_of_twelve_l2440_244070

theorem sum_first_six_multiples_of_twelve : 
  (Finset.range 6).sum (fun i => 12 * (i + 1)) = 252 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_multiples_of_twelve_l2440_244070


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2440_244029

theorem complex_equation_sum (a b : ℝ) : 
  (a - Complex.I) * Complex.I = -b + 2 * Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2440_244029


namespace NUMINAMATH_CALUDE_ones_digit_of_35_power_ones_digit_of_35_large_power_l2440_244043

theorem ones_digit_of_35_power (n : ℕ) : n > 0 → (35^n) % 10 = 5 := by sorry

theorem ones_digit_of_35_large_power : (35^(35*(17^17))) % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_35_power_ones_digit_of_35_large_power_l2440_244043


namespace NUMINAMATH_CALUDE_calculate_expression_l2440_244033

theorem calculate_expression : 5 * 405 + 4 * 405 - 3 * 405 + 404 = 2834 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2440_244033


namespace NUMINAMATH_CALUDE_max_value_on_interval_l2440_244042

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem max_value_on_interval : 
  ∃ (c : ℝ), c ∈ Set.Icc 0 4 ∧ 
  (∀ x, x ∈ Set.Icc 0 4 → f x ≤ f c) ∧
  f c = 10 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l2440_244042


namespace NUMINAMATH_CALUDE_solution_of_quadratic_equation_l2440_244074

theorem solution_of_quadratic_equation :
  ∀ x : ℝ, 2 * x^2 = 4 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_quadratic_equation_l2440_244074


namespace NUMINAMATH_CALUDE_pen_distribution_l2440_244062

theorem pen_distribution (num_students : ℕ) (red_pens : ℕ) (black_pens : ℕ) 
  (month1 : ℕ) (month2 : ℕ) (month3 : ℕ) (month4 : ℕ) : 
  num_students = 6 →
  red_pens = 85 →
  black_pens = 92 →
  month1 = 77 →
  month2 = 89 →
  month3 = 102 →
  month4 = 68 →
  (num_students * (red_pens + black_pens) - (month1 + month2 + month3 + month4)) / num_students = 121 := by
  sorry

#check pen_distribution

end NUMINAMATH_CALUDE_pen_distribution_l2440_244062


namespace NUMINAMATH_CALUDE_probability_no_adjacent_standing_is_correct_l2440_244044

/-- Represents the number of valid arrangements for n people where no two adjacent people stand. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The number of people sitting around the circular table. -/
def numPeople : ℕ := 10

/-- The probability of no two adjacent people standing when numPeople flip fair coins. -/
def probabilityNoAdjacentStanding : ℚ :=
  validArrangements numPeople / 2^numPeople

theorem probability_no_adjacent_standing_is_correct :
  probabilityNoAdjacentStanding = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_standing_is_correct_l2440_244044


namespace NUMINAMATH_CALUDE_total_adding_schemes_l2440_244020

/-- Represents the number of available raw materials -/
def total_materials : ℕ := 5

/-- Represents the number of materials to be added sequentially -/
def materials_to_add : ℕ := 2

/-- Represents the number of ways to add material A first -/
def ways_with_A_first : ℕ := 3

/-- Represents the number of ways to add material B first -/
def ways_with_B_first : ℕ := 6

/-- Represents the number of ways to add materials without A or B -/
def ways_without_A_or_B : ℕ := 6

/-- Theorem stating the total number of different adding schemes -/
theorem total_adding_schemes :
  ways_with_A_first + ways_with_B_first + ways_without_A_or_B = 15 :=
by sorry

end NUMINAMATH_CALUDE_total_adding_schemes_l2440_244020


namespace NUMINAMATH_CALUDE_second_cart_travel_distance_l2440_244061

/-- Distance traveled by the first cart in n seconds -/
def first_cart_distance (n : ℕ) : ℕ := n * (6 + (n - 1) * 4)

/-- Distance traveled by the second cart in n seconds -/
def second_cart_distance (n : ℕ) : ℕ := n * (7 + (n - 1) * 9 / 2)

/-- Time taken by the first cart to reach the bottom -/
def total_time : ℕ := 35

/-- Time difference between the start of the two carts -/
def start_delay : ℕ := 2

theorem second_cart_travel_distance :
  second_cart_distance (total_time - start_delay) = 4983 := by
  sorry

end NUMINAMATH_CALUDE_second_cart_travel_distance_l2440_244061


namespace NUMINAMATH_CALUDE_distinct_points_on_curve_l2440_244068

theorem distinct_points_on_curve : ∃ (a b : ℝ), 
  a ≠ b ∧ 
  (a^3 + Real.sqrt e^4 = 2 * (Real.sqrt e)^2 * a + 1) ∧
  (b^3 + Real.sqrt e^4 = 2 * (Real.sqrt e)^2 * b + 1) ∧
  |a - b| = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_points_on_curve_l2440_244068


namespace NUMINAMATH_CALUDE_equation_solution_l2440_244075

theorem equation_solution (a : ℝ) : (2 * a * 1 - 2 = a + 3) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2440_244075


namespace NUMINAMATH_CALUDE_dollar_three_neg_one_l2440_244024

def dollar (x y : Int) : Int :=
  x * (y + 2) + x * y - 5

theorem dollar_three_neg_one : dollar 3 (-1) = -5 := by
  sorry

end NUMINAMATH_CALUDE_dollar_three_neg_one_l2440_244024


namespace NUMINAMATH_CALUDE_diamond_calculation_l2440_244087

-- Define the diamond operation
def diamond (X Y : ℚ) : ℚ := (2 * X + 3 * Y) / 5

-- Theorem statement
theorem diamond_calculation : diamond (diamond 3 15) 6 = 192 / 25 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l2440_244087


namespace NUMINAMATH_CALUDE_equal_angles_same_terminal_side_l2440_244011

/-- Represents an angle in the coordinate system -/
structure Angle where
  value : ℝ

/-- Represents the terminal side of an angle -/
structure TerminalSide where
  x : ℝ
  y : ℝ

/-- Returns the terminal side of an angle -/
noncomputable def terminalSide (a : Angle) : TerminalSide :=
  { x := Real.cos a.value, y := Real.sin a.value }

/-- Theorem: Equal angles have the same terminal side -/
theorem equal_angles_same_terminal_side (a b : Angle) :
  a = b → terminalSide a = terminalSide b := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_same_terminal_side_l2440_244011


namespace NUMINAMATH_CALUDE_system_solution_proof_l2440_244041

theorem system_solution_proof (x y z : ℝ) : 
  x = 0.38 ∧ y = 0.992 ∧ z = -0.7176 →
  4 * x - 6 * y + 2 * z = -3 ∧
  8 * x + 3 * y - z = 5.3 ∧
  -x + 4 * y + 5 * z = 0 := by
sorry

end NUMINAMATH_CALUDE_system_solution_proof_l2440_244041


namespace NUMINAMATH_CALUDE_train_crossing_time_l2440_244073

/-- Given a train and platform with specified lengths and crossing time, 
    calculate the time taken for the train to cross a signal pole. -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 285)
  (h3 : platform_crossing_time = 39)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 20 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2440_244073


namespace NUMINAMATH_CALUDE_equation_solution_l2440_244079

theorem equation_solution : ∃ (x y : ℕ), 1984 * x - 1983 * y = 1985 ∧ x = 27764 ∧ y = 27777 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2440_244079


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2440_244056

theorem quadratic_roots_relation (b c : ℚ) : 
  (∃ r s : ℚ, (4 * r^2 - 6 * r - 8 = 0) ∧ 
               (4 * s^2 - 6 * s - 8 = 0) ∧ 
               ((r + 3)^2 + b * (r + 3) + c = 0) ∧ 
               ((s + 3)^2 + b * (s + 3) + c = 0)) →
  c = 23 / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2440_244056


namespace NUMINAMATH_CALUDE_distribute_6_3_l2440_244090

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_6_3 : distribute 6 3 = 729 := by sorry

end NUMINAMATH_CALUDE_distribute_6_3_l2440_244090


namespace NUMINAMATH_CALUDE_fifteen_star_positive_integer_count_l2440_244031

def star (a b : ℤ) : ℚ := a^3 / b

theorem fifteen_star_positive_integer_count :
  (∃ (S : Finset ℤ), (∀ x ∈ S, x > 0 ∧ (star 15 x).isInt) ∧ S.card = 16) :=
sorry

end NUMINAMATH_CALUDE_fifteen_star_positive_integer_count_l2440_244031


namespace NUMINAMATH_CALUDE_nine_digit_divisible_by_11_l2440_244096

def is_divisible_by_11 (n : ℕ) : Prop :=
  ∃ k : ℤ, n = 11 * k

def digit_sum_odd (n : ℕ) : ℕ := 
  (n / 100000000) + ((n / 1000000) % 10) + ((n / 10000) % 10) + ((n / 100) % 10) + (n % 10)

def digit_sum_even (n : ℕ) : ℕ := 
  ((n / 10000000) % 10) + ((n / 100000) % 10) + ((n / 1000) % 10) + ((n / 10) % 10)

theorem nine_digit_divisible_by_11 (m : ℕ) :
  m < 10 →
  is_divisible_by_11 (8542 * 100000 + m * 10000 + 7618) →
  m = 0 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_divisible_by_11_l2440_244096


namespace NUMINAMATH_CALUDE_interior_angles_sum_l2440_244076

theorem interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 1800) → (180 * ((n + 4) - 2) = 2520) := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l2440_244076


namespace NUMINAMATH_CALUDE_decimal_51_to_binary_l2440_244037

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_51_to_binary :
  decimal_to_binary 51 = [1, 1, 0, 0, 1, 1] :=
by sorry

end NUMINAMATH_CALUDE_decimal_51_to_binary_l2440_244037


namespace NUMINAMATH_CALUDE_chocolate_bunny_value_is_100_l2440_244091

/-- The number of points needed to win the Nintendo Switch -/
def total_points_needed : ℕ := 2000

/-- The number of chocolate bunnies already sold -/
def chocolate_bunnies_sold : ℕ := 8

/-- The number of points earned per Snickers bar -/
def points_per_snickers : ℕ := 25

/-- The number of Snickers bars needed to win the Nintendo Switch -/
def snickers_bars_needed : ℕ := 48

/-- The value of each chocolate bunny in points -/
def chocolate_bunny_value : ℕ := (total_points_needed - (points_per_snickers * snickers_bars_needed)) / chocolate_bunnies_sold

theorem chocolate_bunny_value_is_100 : chocolate_bunny_value = 100 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bunny_value_is_100_l2440_244091


namespace NUMINAMATH_CALUDE_jellybean_probability_l2440_244082

def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 2
def yellow_jellybeans : ℕ := 5
def picked_jellybeans : ℕ := 4

theorem jellybean_probability :
  (Nat.choose red_jellybeans 3 * Nat.choose (blue_jellybeans + yellow_jellybeans) 1) /
  Nat.choose total_jellybeans picked_jellybeans = 14 / 99 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_probability_l2440_244082


namespace NUMINAMATH_CALUDE_leanna_cd_purchase_l2440_244094

/-- Represents the number of CDs Leanna can buy -/
def max_cds (total : ℕ) (cd_price : ℕ) (cassette_price : ℕ) : ℕ :=
  (total - cassette_price) / cd_price

/-- The cassette price satisfies the given condition -/
def cassette_price_condition (cd_price : ℕ) (cassette_price : ℕ) : Prop :=
  cd_price + 2 * cassette_price + 5 = 37

theorem leanna_cd_purchase :
  ∀ (total : ℕ) (cd_price : ℕ) (cassette_price : ℕ),
    total = 37 →
    cd_price = 14 →
    cassette_price_condition cd_price cassette_price →
    max_cds total cd_price cassette_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_leanna_cd_purchase_l2440_244094


namespace NUMINAMATH_CALUDE_share_ratio_l2440_244007

/-- Represents the shares of money for three individuals -/
structure Shares where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The problem setup -/
def problem_setup (s : Shares) : Prop :=
  s.a = 80 ∧                          -- a's share is $80
  s.a + s.b + s.c = 200 ∧             -- Total amount is $200
  s.a = (2/3) * (s.b + s.c) ∧         -- a gets 2/3 as much as b and c together
  ∃ x, s.b = x * (s.a + s.c)          -- b gets some fraction of a and c together

/-- The theorem to be proved -/
theorem share_ratio (s : Shares) (h : problem_setup s) : 
  s.b / (s.a + s.c) = 2/3 := by sorry

end NUMINAMATH_CALUDE_share_ratio_l2440_244007


namespace NUMINAMATH_CALUDE_group_b_forms_triangle_group_a_not_triangle_group_c_not_triangle_group_d_not_triangle_only_group_b_forms_triangle_l2440_244048

/-- A function that checks if three numbers can form a triangle based on the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the group (3, 4, 6) can form a triangle -/
theorem group_b_forms_triangle :
  can_form_triangle 3 4 6 := by sorry

/-- Theorem stating that the group (3, 4, 7) cannot form a triangle -/
theorem group_a_not_triangle :
  ¬ can_form_triangle 3 4 7 := by sorry

/-- Theorem stating that the group (5, 7, 12) cannot form a triangle -/
theorem group_c_not_triangle :
  ¬ can_form_triangle 5 7 12 := by sorry

/-- Theorem stating that the group (2, 3, 6) cannot form a triangle -/
theorem group_d_not_triangle :
  ¬ can_form_triangle 2 3 6 := by sorry

/-- Main theorem stating that only group B (3, 4, 6) can form a triangle among the given groups -/
theorem only_group_b_forms_triangle :
  can_form_triangle 3 4 6 ∧
  ¬ can_form_triangle 3 4 7 ∧
  ¬ can_form_triangle 5 7 12 ∧
  ¬ can_form_triangle 2 3 6 := by sorry

end NUMINAMATH_CALUDE_group_b_forms_triangle_group_a_not_triangle_group_c_not_triangle_group_d_not_triangle_only_group_b_forms_triangle_l2440_244048


namespace NUMINAMATH_CALUDE_symmetric_circle_correct_l2440_244026

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  (x + 5)^2 + (y - 6)^2 = 16

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x - y = 0

-- Define the symmetric circle C
def symmetric_circle (x y : ℝ) : Prop :=
  (x - 6)^2 + (y + 5)^2 = 16

-- Theorem stating that the symmetric circle C is correct
theorem symmetric_circle_correct :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧ 
   symmetry_line ((x + x₀) / 2) ((y + y₀) / 2)) →
  symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_symmetric_circle_correct_l2440_244026


namespace NUMINAMATH_CALUDE_equation_solution_l2440_244054

theorem equation_solution : 
  {x : ℝ | x^2 + 6*x + 11 = |2*x + 5 - 5*x|} = {-6, -1} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2440_244054


namespace NUMINAMATH_CALUDE_prob_two_tails_proof_l2440_244045

/-- The probability of getting exactly 2 tails when tossing 3 fair coins -/
def prob_two_tails : ℚ := 3 / 8

/-- A fair coin has a probability of 1/2 for each outcome -/
def fair_coin (outcome : Bool) : ℚ := 1 / 2

/-- The number of possible outcomes when tossing 3 coins -/
def total_outcomes : ℕ := 2^3

/-- The number of outcomes with exactly 2 tails when tossing 3 coins -/
def favorable_outcomes : ℕ := 3

theorem prob_two_tails_proof :
  prob_two_tails = favorable_outcomes / total_outcomes :=
sorry

end NUMINAMATH_CALUDE_prob_two_tails_proof_l2440_244045


namespace NUMINAMATH_CALUDE_system_solution_l2440_244051

theorem system_solution :
  let eq1 (x y : ℚ) := x * y^2 - 2 * y^2 + 3 * x = 18
  let eq2 (x y : ℚ) := 3 * x * y + 5 * x - 6 * y = 24
  (eq1 3 3 ∧ eq2 3 3) ∧
  (eq1 (75/13) (-3/7) ∧ eq2 (75/13) (-3/7)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2440_244051


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equation_l2440_244052

/-- The rectangular coordinate equation of the curve ρ = sin θ - 3cos θ -/
theorem polar_to_rectangular_equation :
  ∀ (x y ρ θ : ℝ),
  (ρ = Real.sin θ - 3 * Real.cos θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x^2 - 3*x + y^2 - y = 0) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equation_l2440_244052


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_negative_two_l2440_244077

theorem sum_of_powers_equals_negative_two :
  -1^2010 + (-1)^2011 + 1^2012 - 1^2013 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_negative_two_l2440_244077


namespace NUMINAMATH_CALUDE_cricket_target_runs_l2440_244080

/-- Calculates the target number of runs in a cricket game -/
def targetRuns (totalOvers runRateFirst8 runRateRemaining : ℕ) : ℕ :=
  let runsFirst8 := (runRateFirst8 * 8) / 10
  let runsRemaining := (runRateRemaining * 20) / 10
  runsFirst8 + runsRemaining

/-- Theorem stating the target number of runs for the given conditions -/
theorem cricket_target_runs :
  targetRuns 28 23 120 = 259 := by
  sorry

#eval targetRuns 28 23 120

end NUMINAMATH_CALUDE_cricket_target_runs_l2440_244080


namespace NUMINAMATH_CALUDE_ben_cards_l2440_244081

theorem ben_cards (B : ℕ) (tim_cards : ℕ) : 
  tim_cards = 20 → B + 3 = 2 * tim_cards → B = 37 := by sorry

end NUMINAMATH_CALUDE_ben_cards_l2440_244081


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l2440_244088

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b * (a + b) = 4) :
  2 * a + b ≥ 2 * Real.sqrt 3 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ a * b * (a + b) = 4 ∧ 2 * a + b = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l2440_244088


namespace NUMINAMATH_CALUDE_exists_x_less_than_zero_l2440_244001

theorem exists_x_less_than_zero : ∃ x : ℝ, x^2 - 4*x + 3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_less_than_zero_l2440_244001


namespace NUMINAMATH_CALUDE_larger_number_of_sum_and_difference_l2440_244000

theorem larger_number_of_sum_and_difference (x y : ℝ) : 
  x + y = 40 → x - y = 4 → max x y = 22 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_sum_and_difference_l2440_244000


namespace NUMINAMATH_CALUDE_can_transport_machines_l2440_244017

/-- Given three machines with masses in kg and a truck's capacity in kg,
    prove that the truck can transport all machines at once. -/
theorem can_transport_machines (m1 m2 m3 truck_capacity : ℕ) 
  (h1 : m1 = 800)
  (h2 : m2 = 500)
  (h3 : m3 = 600)
  (h4 : truck_capacity = 2000) :
  m1 + m2 + m3 ≤ truck_capacity := by
  sorry

#check can_transport_machines

end NUMINAMATH_CALUDE_can_transport_machines_l2440_244017


namespace NUMINAMATH_CALUDE_complex_modulus_one_l2440_244035

theorem complex_modulus_one (a : ℝ) :
  let z : ℂ := (a - 1) + a * Complex.I
  Complex.abs z = 1 → a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l2440_244035


namespace NUMINAMATH_CALUDE_cans_first_day_correct_l2440_244025

/-- The number of cans collected on the first day, given the conditions of the problem -/
def cans_first_day : ℕ := 20

/-- The number of days cans are collected -/
def collection_days : ℕ := 5

/-- The daily increase in the number of cans collected -/
def daily_increase : ℕ := 5

/-- The total number of cans collected over the collection period -/
def total_cans : ℕ := 150

/-- Theorem stating that the number of cans collected on the first day is correct -/
theorem cans_first_day_correct : 
  cans_first_day * collection_days + 
  (daily_increase * (collection_days - 1) * collection_days / 2) = total_cans := by
  sorry

end NUMINAMATH_CALUDE_cans_first_day_correct_l2440_244025


namespace NUMINAMATH_CALUDE_last_problem_number_l2440_244050

theorem last_problem_number 
  (start : ℕ) 
  (total : ℕ) 
  (h1 : start = 75) 
  (h2 : total = 51) : 
  start + total - 1 = 125 := by
sorry

end NUMINAMATH_CALUDE_last_problem_number_l2440_244050


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l2440_244009

theorem fraction_sum_difference (a b c d e f : ℤ) :
  (a : ℚ) / b + (c : ℚ) / d - (e : ℚ) / f = (53 : ℚ) / 72 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l2440_244009


namespace NUMINAMATH_CALUDE_train_length_l2440_244086

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 180 → time = 7 → speed * time * (5 / 18) = 350 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2440_244086


namespace NUMINAMATH_CALUDE_janet_pill_count_l2440_244097

/-- Calculates the total number of pills Janet takes in a month -/
def total_pills_in_month (multivitamins_per_day : ℕ) 
                         (calcium_first_half : ℕ) 
                         (calcium_second_half : ℕ) 
                         (days_in_month : ℕ) : ℕ :=
  let first_half := days_in_month / 2
  let second_half := days_in_month - first_half
  (multivitamins_per_day * days_in_month) + 
  (calcium_first_half * first_half) + 
  (calcium_second_half * second_half)

theorem janet_pill_count :
  total_pills_in_month 2 3 1 28 = 112 := by
  sorry

#eval total_pills_in_month 2 3 1 28

end NUMINAMATH_CALUDE_janet_pill_count_l2440_244097


namespace NUMINAMATH_CALUDE_anthony_transaction_percentage_l2440_244066

/-- Proves that Anthony handled 10% more transactions than Mabel given the conditions in the problem. -/
theorem anthony_transaction_percentage (mabel_transactions cal_transactions jade_transactions : ℕ) 
  (anthony_transactions : ℕ) (anthony_percentage : ℚ) :
  mabel_transactions = 90 →
  cal_transactions = (2 : ℚ) / 3 * anthony_transactions →
  jade_transactions = cal_transactions + 19 →
  jade_transactions = 85 →
  anthony_transactions = mabel_transactions * (1 + anthony_percentage / 100) →
  anthony_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_anthony_transaction_percentage_l2440_244066


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2440_244095

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧ 
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2440_244095


namespace NUMINAMATH_CALUDE_opposite_of_neg_three_squared_l2440_244092

theorem opposite_of_neg_three_squared : -(-(3^2)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_three_squared_l2440_244092


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2440_244047

theorem cubic_equation_roots (a b : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x + 6 = 0 ↔ x = 2 ∨ x = 3 ∨ x = -1) →
  a = -4 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2440_244047


namespace NUMINAMATH_CALUDE_socks_cost_prove_socks_cost_l2440_244055

def initial_amount : ℕ := 100
def shirt_cost : ℕ := 24
def final_amount : ℕ := 65

theorem socks_cost : ℕ :=
  initial_amount - shirt_cost - final_amount

theorem prove_socks_cost : socks_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_socks_cost_prove_socks_cost_l2440_244055


namespace NUMINAMATH_CALUDE_decorative_window_area_ratio_l2440_244059

-- Define the window structure
structure DecorativeWindow where
  ab : ℝ  -- Width of the rectangle (diameter of semicircles)
  ad : ℝ  -- Length of the rectangle
  h_ab_positive : ab > 0
  h_ratio : ad / ab = 4 / 3

-- Define the theorem
theorem decorative_window_area_ratio (w : DecorativeWindow) (h_ab : w.ab = 36) :
  (w.ad * w.ab) / (π * (w.ab / 2)^2) = 16 / (3 * π) := by
  sorry

end NUMINAMATH_CALUDE_decorative_window_area_ratio_l2440_244059


namespace NUMINAMATH_CALUDE_min_sum_of_primes_l2440_244071

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem min_sum_of_primes (a b c d n : ℕ) : 
  (∃ k : ℕ, a * 1000 + b * 100 + c * 10 + d = 3 * 3 * 11 * (n + 49)) →
  is_prime a → is_prime b → is_prime c → is_prime d →
  (∀ a' b' c' d' n' : ℕ, 
    (∃ k' : ℕ, a' * 1000 + b' * 100 + c' * 10 + d' = 3 * 3 * 11 * (n' + 49)) →
    is_prime a' → is_prime b' → is_prime c' → is_prime d' →
    a + b + c + d ≤ a' + b' + c' + d') →
  a + b + c + d = 70 := 
sorry

end NUMINAMATH_CALUDE_min_sum_of_primes_l2440_244071


namespace NUMINAMATH_CALUDE_employee_count_l2440_244072

theorem employee_count (initial_avg : ℝ) (new_avg : ℝ) (manager_salary : ℝ) : 
  initial_avg = 1500 →
  new_avg = 1900 →
  manager_salary = 11500 →
  ∃ n : ℕ, (n : ℝ) * initial_avg + manager_salary = new_avg * ((n : ℝ) + 1) ∧ n = 24 := by
sorry

end NUMINAMATH_CALUDE_employee_count_l2440_244072


namespace NUMINAMATH_CALUDE_bus_stop_problem_l2440_244028

/-- The number of students who got off the bus at the first stop -/
def students_who_got_off (initial_students : ℕ) (remaining_students : ℕ) : ℕ :=
  initial_students - remaining_students

theorem bus_stop_problem (initial_students remaining_students : ℕ) 
  (h1 : initial_students = 10)
  (h2 : remaining_students = 7) :
  students_who_got_off initial_students remaining_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_problem_l2440_244028


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2440_244093

theorem consecutive_odd_integers_sum (x y : ℤ) : 
  (Odd x ∧ Odd y) →  -- x and y are odd
  y = x + 4 →        -- y is the next consecutive odd integer after x
  y = 5 * x →        -- y is five times x
  x + y = 6 :=       -- their sum is 6
by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2440_244093


namespace NUMINAMATH_CALUDE_point_set_classification_l2440_244015

-- Define the type for 2D points
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance squared between two points
def distanceSquared (p q : Point2D) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

-- Define the equation
def satisfiesEquation (X : Point2D) (A : List Point2D) (k : List ℝ) (c : ℝ) : Prop :=
  (List.zip k A).foldl (λ sum (kᵢ, Aᵢ) => sum + kᵢ * distanceSquared Aᵢ X) 0 = c

-- State the theorem
theorem point_set_classification 
  (A : List Point2D) (k : List ℝ) (c : ℝ) 
  (h_length : A.length = k.length) :
  (k.sum ≠ 0 → 
    (∃ center : Point2D, ∃ radius : ℝ, 
      ∀ X, satisfiesEquation X A k c ↔ distanceSquared center X = radius^2) ∨
    (∀ X, ¬satisfiesEquation X A k c)) ∧
  (k.sum = 0 → 
    (∃ a b d : ℝ, ∀ X, satisfiesEquation X A k c ↔ a * X.x + b * X.y = d) ∨
    (∀ X, ¬satisfiesEquation X A k c)) :=
sorry

end NUMINAMATH_CALUDE_point_set_classification_l2440_244015


namespace NUMINAMATH_CALUDE_students_left_is_30_percent_l2440_244049

/-- The percentage of students left in a classroom after some students leave for activities -/
def students_left_percentage (total : ℕ) (painting : ℚ) (playing : ℚ) (workshop : ℚ) : ℚ :=
  (1 - (painting + playing + workshop)) * 100

/-- Theorem: Given the conditions, the percentage of students left in the classroom is 30% -/
theorem students_left_is_30_percent :
  students_left_percentage 250 (3/10) (2/10) (1/5) = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_left_is_30_percent_l2440_244049


namespace NUMINAMATH_CALUDE_rhombus_area_l2440_244012

/-- A rhombus with side length √113 and diagonals differing by 8 units has an area of 194 square units. -/
theorem rhombus_area (side : ℝ) (diag_diff : ℝ) (area : ℝ) : 
  side = Real.sqrt 113 → 
  diag_diff = 8 → 
  area = 194 → 
  ∃ (d₁ d₂ : ℝ), d₁ > 0 ∧ d₂ > 0 ∧ d₂ - d₁ = diag_diff ∧ d₁ * d₂ / 2 = area ∧ 
    d₁^2 / 4 + d₂^2 / 4 = side^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_l2440_244012


namespace NUMINAMATH_CALUDE_oil_depth_theorem_l2440_244003

/-- Represents a horizontal cylindrical oil tank -/
structure OilTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the depth of oil in the tank given the surface area -/
def oilDepth (tank : OilTank) (surfaceArea : ℝ) : Set ℝ :=
  { h : ℝ | ∃ (c : ℝ), 
    c = surfaceArea / tank.length ∧
    c = 2 * Real.sqrt (tank.diameter * h - h^2) ∧
    0 < h ∧ h < tank.diameter }

/-- The main theorem about oil depth in a cylindrical tank -/
theorem oil_depth_theorem (tank : OilTank) (surfaceArea : ℝ) :
  tank.length = 12 →
  tank.diameter = 8 →
  surfaceArea = 60 →
  oilDepth tank surfaceArea = {4 - Real.sqrt 39 / 2, 4 + Real.sqrt 39 / 2} := by
  sorry

end NUMINAMATH_CALUDE_oil_depth_theorem_l2440_244003


namespace NUMINAMATH_CALUDE_smallest_aab_value_l2440_244067

theorem smallest_aab_value (A B : ℕ) : 
  (1 ≤ A ∧ A ≤ 9) →  -- A is a digit from 1 to 9
  (1 ≤ B ∧ B ≤ 9) →  -- B is a digit from 1 to 9
  A + 1 = B →        -- A and B are consecutive digits
  (10 * A + B : ℕ) = (110 * A + B) / 7 →  -- AB = AAB / 7
  (∀ A' B' : ℕ, 
    (1 ≤ A' ∧ A' ≤ 9) → 
    (1 ≤ B' ∧ B' ≤ 9) → 
    A' + 1 = B' → 
    (10 * A' + B' : ℕ) = (110 * A' + B') / 7 → 
    110 * A + B ≤ 110 * A' + B') →
  110 * A + B = 889 := by
sorry

end NUMINAMATH_CALUDE_smallest_aab_value_l2440_244067


namespace NUMINAMATH_CALUDE_train_speed_l2440_244089

/-- The speed of a train given specific conditions -/
theorem train_speed (t_pole : ℝ) (t_stationary : ℝ) (l_stationary : ℝ) :
  t_pole = 10 →
  t_stationary = 30 →
  l_stationary = 600 →
  ∃ v : ℝ, v * t_pole = v * t_stationary - l_stationary ∧ v * 3.6 = 108 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2440_244089


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l2440_244083

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l2440_244083


namespace NUMINAMATH_CALUDE_danivan_drugstore_inventory_l2440_244022

def calculate_final_inventory (starting_inventory : ℕ) (daily_sales : List ℕ) (deliveries : List ℕ) : ℕ :=
  let daily_changes := List.zipWith (λ s d => d - s) daily_sales deliveries
  starting_inventory + daily_changes.sum

theorem danivan_drugstore_inventory : 
  let starting_inventory : ℕ := 4500
  let daily_sales : List ℕ := [1277, 2124, 679, 854, 535, 1073, 728]
  let deliveries : List ℕ := [2250, 0, 980, 750, 0, 1345, 0]
  calculate_final_inventory starting_inventory daily_sales deliveries = 2555 := by
  sorry

#eval calculate_final_inventory 4500 [1277, 2124, 679, 854, 535, 1073, 728] [2250, 0, 980, 750, 0, 1345, 0]

end NUMINAMATH_CALUDE_danivan_drugstore_inventory_l2440_244022


namespace NUMINAMATH_CALUDE_sum_congruence_l2440_244084

theorem sum_congruence : (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l2440_244084


namespace NUMINAMATH_CALUDE_intersection_empty_at_m_zero_l2440_244069

theorem intersection_empty_at_m_zero :
  ∃ m : ℝ, m = 0 ∧ (Set.Icc 0 1 : Set ℝ) ∩ {x : ℝ | x^2 - 2*x + m > 0} = ∅ :=
by sorry

end NUMINAMATH_CALUDE_intersection_empty_at_m_zero_l2440_244069
