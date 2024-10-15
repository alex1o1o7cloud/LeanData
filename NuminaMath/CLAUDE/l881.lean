import Mathlib

namespace NUMINAMATH_CALUDE_intersection_implies_a_values_l881_88118

theorem intersection_implies_a_values (a : ℝ) : 
  let M : Set ℝ := {5, a^2 - 3*a + 5}
  let N : Set ℝ := {1, 3}
  (M ∩ N).Nonempty → a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_l881_88118


namespace NUMINAMATH_CALUDE_jerry_video_games_l881_88158

theorem jerry_video_games (initial_games new_games : ℕ) : 
  initial_games = 7 → new_games = 2 → initial_games + new_games = 9 :=
by sorry

end NUMINAMATH_CALUDE_jerry_video_games_l881_88158


namespace NUMINAMATH_CALUDE_chemists_self_receipts_l881_88140

/-- Represents a chemist in the laboratory -/
structure Chemist where
  id : Nat
  reagents : Finset Nat

/-- Represents the state of the laboratory -/
structure Laboratory where
  chemists : Finset Chemist
  num_chemists : Nat

/-- Checks if a chemist has received all reagents -/
def has_all_reagents (c : Chemist) (lab : Laboratory) : Prop :=
  c.reagents.card = lab.num_chemists

/-- Checks if no chemist has received any reagent more than once -/
def no_double_receipts (lab : Laboratory) : Prop :=
  ∀ c ∈ lab.chemists, ∀ r ∈ c.reagents, (c.reagents.filter (λ x => x = r)).card ≤ 1

/-- Counts the number of chemists who received their own reagent -/
def count_self_receipts (lab : Laboratory) : Nat :=
  (lab.chemists.filter (λ c => c.id ∈ c.reagents)).card

/-- The main theorem to be proved -/
theorem chemists_self_receipts (lab : Laboratory) 
  (h1 : ∀ c ∈ lab.chemists, has_all_reagents c lab)
  (h2 : no_double_receipts lab) :
  count_self_receipts lab ≥ lab.num_chemists - 1 :=
sorry

end NUMINAMATH_CALUDE_chemists_self_receipts_l881_88140


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l881_88182

/-- Theorem stating the relationship between k, a, m, and n for a parabola and line intersection -/
theorem parabola_line_intersection
  (a m n k b : ℝ)
  (ha : a ≠ 0)
  (h_intersect : ∃ (y₁ y₂ : ℝ),
    a * (1 - m) * (1 - n) = k * 1 + b ∧
    a * (6 - m) * (6 - n) = k * 6 + b) :
  k = a * (7 - m - n) := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l881_88182


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l881_88175

theorem nested_fraction_equality : 1 + (1 / (1 + (1 / (1 + (1 / (1 + 2)))))) = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l881_88175


namespace NUMINAMATH_CALUDE_equation_solutions_l881_88128

theorem equation_solutions : 
  ∀ x : ℝ, (x^2 + x)^2 + (x^2 + x) - 6 = 0 ↔ x = -2 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l881_88128


namespace NUMINAMATH_CALUDE_hundred_day_previous_year_is_saturday_l881_88194

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (day : ℕ) : DayOfWeek := sorry

/-- Returns true if the year is a leap year, false otherwise -/
def isLeapYear (y : Year) : Bool := sorry

/-- The number of days in a year -/
def daysInYear (y : Year) : ℕ :=
  if isLeapYear y then 366 else 365

theorem hundred_day_previous_year_is_saturday 
  (N : Year)
  (h1 : dayOfWeek N 400 = DayOfWeek.Friday)
  (h2 : dayOfWeek (Year.mk (N.value + 1)) 300 = DayOfWeek.Friday) :
  dayOfWeek (Year.mk (N.value - 1)) 100 = DayOfWeek.Saturday := by
  sorry

end NUMINAMATH_CALUDE_hundred_day_previous_year_is_saturday_l881_88194


namespace NUMINAMATH_CALUDE_nth_row_equation_l881_88141

theorem nth_row_equation (n : ℕ) : 2 * n + 1 = (n + 1)^2 - n^2 := by
  sorry

end NUMINAMATH_CALUDE_nth_row_equation_l881_88141


namespace NUMINAMATH_CALUDE_range_of_a_l881_88142

-- Define the conditions P and Q
def P (a : ℝ) : Prop := ∀ x y : ℝ, ∃ k : ℝ, k > 0 ∧ x^2 / (3 - a) + y^2 / (1 + a) = k

def Q (a : ℝ) : Prop := ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

-- Theorem statement
theorem range_of_a (a : ℝ) (hP : P a) (hQ : Q a) : 
  -1 < a ∧ a ≤ 2 ∧ a ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l881_88142


namespace NUMINAMATH_CALUDE_blue_jelly_bean_probability_l881_88192

/-- The probability of selecting a blue jelly bean from a bag -/
theorem blue_jelly_bean_probability :
  let red : ℕ := 5
  let green : ℕ := 6
  let yellow : ℕ := 7
  let blue : ℕ := 8
  let total : ℕ := red + green + yellow + blue
  (blue : ℚ) / total = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_blue_jelly_bean_probability_l881_88192


namespace NUMINAMATH_CALUDE_max_book_price_l881_88185

theorem max_book_price (total_money : ℕ) (num_books : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) :
  total_money = 200 →
  num_books = 20 →
  entrance_fee = 5 →
  tax_rate = 7 / 100 →
  ∃ (max_price : ℕ),
    (max_price ≤ (total_money - entrance_fee) / (num_books * (1 + tax_rate))) ∧
    (∀ (price : ℕ), price > max_price →
      price * num_books * (1 + tax_rate) > (total_money - entrance_fee)) ∧
    max_price = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_book_price_l881_88185


namespace NUMINAMATH_CALUDE_work_completion_time_l881_88184

/-- The number of days it takes for worker a to complete the work alone -/
def days_a : ℝ := 4

/-- The number of days it takes for worker b to complete the work alone -/
def days_b : ℝ := 9

/-- The number of days it takes for workers a, b, and c to complete the work together -/
def days_together : ℝ := 2

/-- The number of days it takes for worker c to complete the work alone -/
def days_c : ℝ := 7.2

theorem work_completion_time :
  (1 / days_a) + (1 / days_b) + (1 / days_c) = (1 / days_together) :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l881_88184


namespace NUMINAMATH_CALUDE_equation_solution_l881_88193

theorem equation_solution : ∃! x : ℝ, (1/4 : ℝ)^(2*x+8) = 16^(2*x+5) :=
  by
    use -3
    constructor
    · -- Prove that x = -3 satisfies the equation
      sorry
    · -- Prove uniqueness
      sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l881_88193


namespace NUMINAMATH_CALUDE_difference_one_third_and_decimal_l881_88167

theorem difference_one_third_and_decimal : (1 : ℚ) / 3 - 333 / 1000 = 1 / 3000 := by sorry

end NUMINAMATH_CALUDE_difference_one_third_and_decimal_l881_88167


namespace NUMINAMATH_CALUDE_league_games_count_l881_88165

/-- Calculates the number of games in a round-robin tournament. -/
def numGames (n : ℕ) (k : ℕ) : ℕ := n * (n - 1) / 2 * k

/-- Proves that in a league with 20 teams, where each team plays every other team 4 times, 
    the total number of games played in the season is 760. -/
theorem league_games_count : numGames 20 4 = 760 := by
  sorry

end NUMINAMATH_CALUDE_league_games_count_l881_88165


namespace NUMINAMATH_CALUDE_units_digit_of_large_product_l881_88199

theorem units_digit_of_large_product : ∃ n : ℕ, n < 10 ∧ 2^1007 * 6^1008 * 14^1009 ≡ n [ZMOD 10] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_large_product_l881_88199


namespace NUMINAMATH_CALUDE_brick_wall_theorem_l881_88189

/-- Represents a brick wall with a given number of rows, total bricks, and bricks in the bottom row. -/
structure BrickWall where
  rows : ℕ
  totalBricks : ℕ
  bottomRowBricks : ℕ

/-- Calculates the number of bricks in a given row of the wall. -/
def bricksInRow (wall : BrickWall) (rowNumber : ℕ) : ℕ :=
  wall.bottomRowBricks - (rowNumber - 1)

theorem brick_wall_theorem (wall : BrickWall) 
    (h1 : wall.rows = 5)
    (h2 : wall.totalBricks = 100)
    (h3 : wall.bottomRowBricks = 18) :
    ∀ (r : ℕ), 1 < r ∧ r ≤ wall.rows → 
    bricksInRow wall r = bricksInRow wall (r - 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_brick_wall_theorem_l881_88189


namespace NUMINAMATH_CALUDE_inequality_proof_l881_88111

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) :
  (a^5 - 1) / (a^4 - 1) * (b^5 - 1) / (b^4 - 1) > 25/64 * (a + 1) * (b + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l881_88111


namespace NUMINAMATH_CALUDE_conors_potato_chopping_l881_88164

/-- The number of potatoes Conor can chop in a day -/
def potatoes_per_day : ℕ := sorry

/-- The number of eggplants Conor can chop in a day -/
def eggplants_per_day : ℕ := 12

/-- The number of carrots Conor can chop in a day -/
def carrots_per_day : ℕ := 9

/-- The number of days Conor works per week -/
def work_days_per_week : ℕ := 4

/-- The total number of vegetables Conor chops in a week -/
def total_vegetables_per_week : ℕ := 116

theorem conors_potato_chopping :
  potatoes_per_day = 8 ∧
  work_days_per_week * (eggplants_per_day + carrots_per_day + potatoes_per_day) = total_vegetables_per_week :=
by sorry

end NUMINAMATH_CALUDE_conors_potato_chopping_l881_88164


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l881_88133

/-- For non-zero real numbers a, b, c, if they form a geometric sequence,
    then their reciprocals and their squares also form geometric sequences. -/
theorem geometric_sequence_properties (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
    (h_geometric : b^2 = a * c) : 
  (1 / b)^2 = (1 / a) * (1 / c) ∧ (b^2)^2 = a^2 * c^2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_properties_l881_88133


namespace NUMINAMATH_CALUDE_calculation_1_l881_88177

theorem calculation_1 : (1 * (-1/9)) - (1/2) = -11/18 := by sorry

end NUMINAMATH_CALUDE_calculation_1_l881_88177


namespace NUMINAMATH_CALUDE_quadratic_inequality_l881_88146

theorem quadratic_inequality (y : ℝ) : y^2 - 6*y - 16 > 0 ↔ y < -2 ∨ y > 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l881_88146


namespace NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l881_88129

/-- The function f(x) defined as x^2 + bx + 5 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 5

/-- Theorem stating that -3 is not in the range of f(x) if and only if b is in the open interval (-4√2, 4√2) -/
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, f b x ≠ -3) ↔ b ∈ Set.Ioo (-4 * Real.sqrt 2) (4 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l881_88129


namespace NUMINAMATH_CALUDE_smallest_factorial_with_1987_zeros_l881_88134

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The smallest natural number n such that n! ends with exactly 1987 zeros -/
def smallestFactorialWith1987Zeros : ℕ := 7960

theorem smallest_factorial_with_1987_zeros :
  (∀ m < smallestFactorialWith1987Zeros, trailingZeros m < 1987) ∧
  trailingZeros smallestFactorialWith1987Zeros = 1987 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factorial_with_1987_zeros_l881_88134


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_50_5005_l881_88123

theorem gcd_lcm_sum_50_5005 : 
  Nat.gcd 50 5005 + Nat.lcm 50 5005 = 50055 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_50_5005_l881_88123


namespace NUMINAMATH_CALUDE_angle_F_measure_l881_88112

-- Define a triangle DEF
structure Triangle :=
  (D E F : ℝ)

-- Define the properties of the triangle
def validTriangle (t : Triangle) : Prop :=
  t.D > 0 ∧ t.E > 0 ∧ t.F > 0 ∧ t.D + t.E + t.F = 180

-- Theorem statement
theorem angle_F_measure (t : Triangle) 
  (h1 : validTriangle t) 
  (h2 : t.D = 3 * t.E) 
  (h3 : t.E = 18) : 
  t.F = 108 := by
  sorry

end NUMINAMATH_CALUDE_angle_F_measure_l881_88112


namespace NUMINAMATH_CALUDE_new_person_weight_l881_88169

def group_weight_change (initial_count : ℕ) (leaving_weight : ℝ) (average_increase : ℝ) : ℝ :=
  let final_count : ℕ := initial_count
  let intermediate_count : ℕ := initial_count - 1
  (final_count : ℝ) * average_increase + leaving_weight

theorem new_person_weight 
  (initial_count : ℕ) 
  (leaving_weight : ℝ) 
  (average_increase : ℝ) 
  (h1 : initial_count = 15) 
  (h2 : leaving_weight = 90) 
  (h3 : average_increase = 3.7) : 
  group_weight_change initial_count leaving_weight average_increase = 55.5 := by
sorry

#eval group_weight_change 15 90 3.7

end NUMINAMATH_CALUDE_new_person_weight_l881_88169


namespace NUMINAMATH_CALUDE_min_value_quadratic_l881_88121

theorem min_value_quadratic (x : ℝ) : 
  ∃ (min_y : ℝ), ∀ (y : ℝ), y = 5*x^2 + 10*x + 15 → y ≥ min_y ∧ ∃ (x₀ : ℝ), 5*x₀^2 + 10*x₀ + 15 = min_y :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l881_88121


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_A_in_U_l881_88162

-- Define the universal set U
def U : Set ℝ := {x | 1 < x ∧ x < 7}

-- Define set A
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 5} := by sorry

-- Theorem for complement of A in U
theorem complement_A_in_U : (U \ A) = {x | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 7)} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_A_in_U_l881_88162


namespace NUMINAMATH_CALUDE_fraction_value_proof_l881_88173

theorem fraction_value_proof (a b c : ℚ) (h1 : a = 5) (h2 : b = -3) (h3 : c = 4) :
  2 * c / (a + b) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_proof_l881_88173


namespace NUMINAMATH_CALUDE_work_time_calculation_l881_88116

theorem work_time_calculation (a_time b_time : ℝ) (b_fraction : ℝ) : 
  a_time = 6 →
  b_time = 3 →
  b_fraction = 1/9 →
  (1 - b_fraction) / (1 / a_time) = 16/3 :=
by sorry

end NUMINAMATH_CALUDE_work_time_calculation_l881_88116


namespace NUMINAMATH_CALUDE_division_of_composite_products_l881_88179

-- Define the first six positive composite integers
def first_six_composites : List Nat := [4, 6, 8, 9, 10, 12]

-- Define the product of the first three composite integers
def product_first_three : Nat := (first_six_composites.take 3).prod

-- Define the product of the next three composite integers
def product_next_three : Nat := (first_six_composites.drop 3).prod

-- Theorem to prove
theorem division_of_composite_products :
  (product_first_three : ℚ) / product_next_three = 8 / 45 := by
  sorry

end NUMINAMATH_CALUDE_division_of_composite_products_l881_88179


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l881_88171

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + y = 9) 
  (eq2 : x + 3 * y = 10) : 
  10 * x^2 + 19 * x * y + 10 * y^2 = 181 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l881_88171


namespace NUMINAMATH_CALUDE_solution_set_characterization_l881_88166

def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  x^2 - 2*a*x + a + 2 ≤ 0

def solution_set (a : ℝ) : Set ℝ :=
  {x : ℝ | quadratic_inequality a x}

theorem solution_set_characterization (a : ℝ) :
  (solution_set a ⊆ Set.Icc 1 3) ↔ a ∈ Set.Ioo (-1) (11/5) :=
sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l881_88166


namespace NUMINAMATH_CALUDE_parallelogram_area_l881_88117

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 and 12 is 60 square units. -/
theorem parallelogram_area (a b : ℝ) (angle : ℝ) (h1 : a = 10) (h2 : b = 12) (h3 : angle = 150) :
  a * b * Real.sin (angle * π / 180) = 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l881_88117


namespace NUMINAMATH_CALUDE_buttons_problem_l881_88137

theorem buttons_problem (sue kendra mari : ℕ) : 
  sue = kendra / 2 →
  sue = 6 →
  mari = 5 * kendra + 4 →
  mari = 64 := by
sorry

end NUMINAMATH_CALUDE_buttons_problem_l881_88137


namespace NUMINAMATH_CALUDE_right_to_left_evaluation_l881_88127

-- Define a custom operation that evaluates from right to left
noncomputable def rightToLeftEval (a b c d : ℝ) : ℝ :=
  a * (b / (c + d^2))

-- Theorem statement
theorem right_to_left_evaluation (a b c d : ℝ) :
  rightToLeftEval a b c d = (a * b) / (c + d^2) :=
by sorry

end NUMINAMATH_CALUDE_right_to_left_evaluation_l881_88127


namespace NUMINAMATH_CALUDE_max_negative_integers_in_equation_l881_88156

theorem max_negative_integers_in_equation (a b c d : ℤ) 
  (eq : (2 : ℝ)^a + (2 : ℝ)^b = (5 : ℝ)^c + (5 : ℝ)^d) : 
  ∀ (n : ℕ), n ≤ (if a < 0 then 1 else 0) + 
              (if b < 0 then 1 else 0) + 
              (if c < 0 then 1 else 0) + 
              (if d < 0 then 1 else 0) → n = 0 :=
sorry

end NUMINAMATH_CALUDE_max_negative_integers_in_equation_l881_88156


namespace NUMINAMATH_CALUDE_perpendicular_polygon_perimeter_l881_88191

/-- A polygon with adjacent sides perpendicular to each other -/
structure PerpendicularPolygon where
  a : ℝ  -- Sum of all vertical sides
  b : ℝ  -- Sum of all horizontal sides

/-- The perimeter of a perpendicular polygon -/
def perimeter (p : PerpendicularPolygon) : ℝ := 2 * (p.a + p.b)

/-- Theorem: The perimeter of a perpendicular polygon is 2(a+b) -/
theorem perpendicular_polygon_perimeter (p : PerpendicularPolygon) :
  perimeter p = 2 * (p.a + p.b) := by sorry

end NUMINAMATH_CALUDE_perpendicular_polygon_perimeter_l881_88191


namespace NUMINAMATH_CALUDE_grid_intersection_sum_zero_l881_88157

/-- Represents a cell in the grid -/
inductive CellValue
  | Plus : CellValue
  | Minus : CellValue
  | Zero : CellValue

/-- Represents the grid -/
def Grid := Matrix (Fin 1980) (Fin 1981) CellValue

/-- The sum of all numbers in the grid is zero -/
def sumIsZero (g : Grid) : Prop := sorry

/-- The sum of four numbers at the intersections of two rows and two columns -/
def intersectionSum (g : Grid) (r1 r2 : Fin 1980) (c1 c2 : Fin 1981) : Int := sorry

theorem grid_intersection_sum_zero (g : Grid) (h : sumIsZero g) :
  ∃ (r1 r2 : Fin 1980) (c1 c2 : Fin 1981), intersectionSum g r1 r2 c1 c2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_grid_intersection_sum_zero_l881_88157


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l881_88125

theorem necessary_but_not_sufficient (a b : ℝ) :
  (∀ x y : ℝ, x * y ≠ 0 → x ≠ 0) ∧
  ¬(∀ x y : ℝ, x ≠ 0 → x * y ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l881_88125


namespace NUMINAMATH_CALUDE_prob_all_boys_prob_two_boys_one_girl_l881_88109

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of people to be selected -/
def select_num : ℕ := 3

/-- The probability of selecting 3 boys out of the total 6 people -/
theorem prob_all_boys : 
  (Nat.choose num_boys select_num : ℚ) / (Nat.choose total_people select_num) = 1 / 5 := by
  sorry

/-- The probability of selecting 2 boys and 1 girl out of the total 6 people -/
theorem prob_two_boys_one_girl : 
  ((Nat.choose num_boys 2 * Nat.choose num_girls 1) : ℚ) / (Nat.choose total_people select_num) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_boys_prob_two_boys_one_girl_l881_88109


namespace NUMINAMATH_CALUDE_simplify_expression_l881_88131

theorem simplify_expression : 
  2 - (2 / (2 + Real.sqrt 5)) + (2 / (2 - Real.sqrt 5)) = 2 + 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l881_88131


namespace NUMINAMATH_CALUDE_pauls_crayons_left_l881_88149

/-- Represents the number of crayons Paul had left at the end of the school year. -/
def crayons_left (initial_erasers initial_crayons : ℕ) (extra_crayons : ℕ) : ℕ :=
  initial_erasers + extra_crayons

/-- Theorem stating that Paul had 523 crayons left at the end of the school year. -/
theorem pauls_crayons_left :
  crayons_left 457 617 66 = 523 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_left_l881_88149


namespace NUMINAMATH_CALUDE_unique_perfect_square_and_cube_factor_of_1800_l881_88122

/-- A number is a perfect square if it's equal to some integer squared. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

/-- A number is a perfect cube if it's equal to some integer cubed. -/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

/-- A number is both a perfect square and a perfect cube. -/
def is_perfect_square_and_cube (n : ℕ) : Prop :=
  is_perfect_square n ∧ is_perfect_cube n

/-- The set of positive factors of a natural number. -/
def positive_factors (n : ℕ) : Set ℕ :=
  {k : ℕ | k > 0 ∧ n % k = 0}

/-- There is exactly one positive factor of 1800 that is both a perfect square and a perfect cube. -/
theorem unique_perfect_square_and_cube_factor_of_1800 :
  ∃! x : ℕ, x ∈ positive_factors 1800 ∧ is_perfect_square_and_cube x :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_and_cube_factor_of_1800_l881_88122


namespace NUMINAMATH_CALUDE_reaching_penglai_sufficient_for_immortal_l881_88170

/-- Reaching Penglai implies becoming an immortal -/
def reaching_penglai_implies_immortal (reaching_penglai becoming_immortal : Prop) : Prop :=
  reaching_penglai → becoming_immortal

/-- Not reaching Penglai implies not becoming an immortal -/
axiom not_reaching_penglai_implies_not_immortal {reaching_penglai becoming_immortal : Prop} :
  ¬reaching_penglai → ¬becoming_immortal

/-- Prove that reaching Penglai is a sufficient condition for becoming an immortal -/
theorem reaching_penglai_sufficient_for_immortal
  {reaching_penglai becoming_immortal : Prop}
  (h : ¬reaching_penglai → ¬becoming_immortal) :
  reaching_penglai_implies_immortal reaching_penglai becoming_immortal :=
by sorry

end NUMINAMATH_CALUDE_reaching_penglai_sufficient_for_immortal_l881_88170


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l881_88197

theorem polynomial_evaluation (x : ℝ) (hx_pos : x > 0) (hx_eq : x^2 - 3*x - 9 = 0) :
  x^4 - 3*x^3 - 9*x^2 + 27*x - 8 = -8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l881_88197


namespace NUMINAMATH_CALUDE_m_3_sufficient_m_3_not_necessary_l881_88135

/-- Represents an ellipse with equation x²/4 + y²/m = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ (x y : ℝ), x^2/4 + y^2/m = 1

/-- The focal length of an ellipse -/
def focal_length (e : Ellipse m) : ℝ := 
  sorry

/-- Theorem stating that m = 3 is sufficient for focal length 2 -/
theorem m_3_sufficient (e : Ellipse 3) : focal_length e = 2 :=
  sorry

/-- Theorem stating that m = 3 is not necessary for focal length 2 -/
theorem m_3_not_necessary : ∃ (m : ℝ), m ≠ 3 ∧ ∃ (e : Ellipse m), focal_length e = 2 :=
  sorry

end NUMINAMATH_CALUDE_m_3_sufficient_m_3_not_necessary_l881_88135


namespace NUMINAMATH_CALUDE_max_value_of_function_sum_of_powers_greater_than_one_l881_88101

-- Part 1
theorem max_value_of_function (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∃ M : ℝ, M = 1 ∧ ∀ x > -1, (1 + x)^a - a * x ≤ M :=
sorry

-- Part 2
theorem sum_of_powers_greater_than_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^b + b^a > 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_sum_of_powers_greater_than_one_l881_88101


namespace NUMINAMATH_CALUDE_josh_marbles_l881_88144

def marble_problem (initial_marbles found_marbles : ℕ) : Prop :=
  initial_marbles + found_marbles = 28

theorem josh_marbles : marble_problem 21 7 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l881_88144


namespace NUMINAMATH_CALUDE_clothes_cost_l881_88138

def total_spending : ℕ := 10000

def adidas_cost : ℕ := 800

def nike_cost : ℕ := 2 * adidas_cost

def skechers_cost : ℕ := 4 * adidas_cost

def puma_cost : ℕ := nike_cost / 2

def total_sneakers_cost : ℕ := adidas_cost + nike_cost + skechers_cost + puma_cost

theorem clothes_cost : total_spending - total_sneakers_cost = 3600 := by
  sorry

end NUMINAMATH_CALUDE_clothes_cost_l881_88138


namespace NUMINAMATH_CALUDE_function_domain_implies_m_range_l881_88155

/-- Given a function f(x) = 1 / √(mx² + mx + 1) with domain R, 
    prove that m must be in the range [0, 4) -/
theorem function_domain_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, mx^2 + mx + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_function_domain_implies_m_range_l881_88155


namespace NUMINAMATH_CALUDE_taller_tree_height_taller_tree_height_proof_l881_88181

/-- Given two trees where one is 20 feet taller than the other and their heights are in the ratio 2:3,
    the height of the taller tree is 60 feet. -/
theorem taller_tree_height : ℝ → ℝ → Prop :=
  fun h₁ h₂ => (h₁ = h₂ + 20 ∧ h₂ / h₁ = 2 / 3) → h₁ = 60

/-- Proof of the theorem -/
theorem taller_tree_height_proof : taller_tree_height 60 40 := by
  sorry

end NUMINAMATH_CALUDE_taller_tree_height_taller_tree_height_proof_l881_88181


namespace NUMINAMATH_CALUDE_cafeteria_green_apples_l881_88104

theorem cafeteria_green_apples :
  ∀ (red_apples students_wanting_fruit extra_apples green_apples : ℕ),
    red_apples = 25 →
    students_wanting_fruit = 10 →
    extra_apples = 32 →
    red_apples + green_apples - students_wanting_fruit = extra_apples →
    green_apples = 17 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_green_apples_l881_88104


namespace NUMINAMATH_CALUDE_same_heads_probability_l881_88154

-- Define the probability of getting a specific number of heads when tossing two coins
def prob_heads (n : Nat) : ℚ :=
  if n = 0 then 1/4
  else if n = 1 then 1/2
  else if n = 2 then 1/4
  else 0

-- Define the probability of both people getting the same number of heads
def prob_same_heads : ℚ :=
  (prob_heads 0)^2 + (prob_heads 1)^2 + (prob_heads 2)^2

-- Theorem statement
theorem same_heads_probability : prob_same_heads = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_same_heads_probability_l881_88154


namespace NUMINAMATH_CALUDE_at_most_one_greater_than_one_l881_88126

theorem at_most_one_greater_than_one (x y : ℝ) (h : x + y < 2) :
  ¬(x > 1 ∧ y > 1) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_greater_than_one_l881_88126


namespace NUMINAMATH_CALUDE_abc_inequality_and_reciprocal_sum_l881_88139

theorem abc_inequality_and_reciprocal_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 2) : 
  a * b * c ≤ 8 / 27 ∧ 1 / a + 1 / b + 1 / c ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_and_reciprocal_sum_l881_88139


namespace NUMINAMATH_CALUDE_polynomial_intersection_theorem_l881_88130

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- Define the theorem
theorem polynomial_intersection_theorem (a b c d : ℝ) : 
  -- f and g are distinct polynomials
  (∃ x, f a b x ≠ g c d x) →
  -- The x-coordinate of the vertex of f is a root of g
  g c d (-a/2) = 0 →
  -- The x-coordinate of the vertex of g is a root of f
  f a b (-c/2) = 0 →
  -- The graphs intersect at (50, -200)
  f a b 50 = -200 ∧ g c d 50 = -200 →
  -- The minimum value of f is 50 less than the minimum value of g
  (-a^2/4 + b) = (-c^2/4 + d - 50) →
  -- There exists a unique value for a + c
  ∃! x, x = a + c :=
by sorry

end NUMINAMATH_CALUDE_polynomial_intersection_theorem_l881_88130


namespace NUMINAMATH_CALUDE_xavier_yasmin_age_ratio_l881_88120

/-- Represents a person's age -/
structure Person where
  age : ℕ

/-- Xavier is older than Yasmin -/
def xavier_older (x y : Person) : Prop :=
  x.age > y.age

/-- Xavier will be 30 in 6 years -/
def xavier_future_age (x : Person) : Prop :=
  x.age + 6 = 30

/-- The sum of Xavier and Yasmin's ages is 36 -/
def total_age (x y : Person) : Prop :=
  x.age + y.age = 36

/-- The ratio of Xavier's age to Yasmin's age is 2:1 -/
def age_ratio (x y : Person) : Prop :=
  2 * y.age = x.age

theorem xavier_yasmin_age_ratio (x y : Person) 
  (h1 : xavier_older x y) 
  (h2 : xavier_future_age x) 
  (h3 : total_age x y) : 
  age_ratio x y := by
  sorry

end NUMINAMATH_CALUDE_xavier_yasmin_age_ratio_l881_88120


namespace NUMINAMATH_CALUDE_james_work_hours_l881_88198

/-- Calculates the number of hours James works at his main job --/
theorem james_work_hours (main_rate : ℝ) (second_rate_reduction : ℝ) (total_earnings : ℝ) :
  main_rate = 20 →
  second_rate_reduction = 0.2 →
  total_earnings = 840 →
  ∃ h : ℝ, h = 30 ∧ 
    main_rate * h + (main_rate * (1 - second_rate_reduction)) * (h / 2) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_james_work_hours_l881_88198


namespace NUMINAMATH_CALUDE_inequality_system_solution_l881_88119

def inequality_system (x : ℝ) : Prop :=
  x > -6 - 2*x ∧ x ≤ (3 + x) / 4

theorem inequality_system_solution :
  ∀ x : ℝ, inequality_system x ↔ -2 < x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l881_88119


namespace NUMINAMATH_CALUDE_power_of_64_equals_128_l881_88163

theorem power_of_64_equals_128 : (64 : ℝ) ^ (7/6) = 128 := by
  have h : 64 = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_64_equals_128_l881_88163


namespace NUMINAMATH_CALUDE_distance_traveled_l881_88148

theorem distance_traveled (speed1 speed2 distance_diff : ℝ) (h1 : speed1 = 10)
  (h2 : speed2 = 20) (h3 : distance_diff = 40) :
  let time := distance_diff / (speed2 - speed1)
  let actual_distance := speed1 * time
  actual_distance = 40 :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_l881_88148


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_c_value_l881_88102

theorem polynomial_equality_implies_c_value (a c : ℚ) 
  (h : ∀ x : ℚ, (x + 3) * (x + a) = x^2 + c*x + 8) : 
  c = 17/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_c_value_l881_88102


namespace NUMINAMATH_CALUDE_line_segment_theorem_l881_88168

/-- Represents a line segment on a straight line -/
structure LineSegment where
  left : ℝ
  right : ℝ
  h : left ≤ right

/-- Given a list of line segments, returns true if there exists a point common to at least n of them -/
def has_common_point (segments : List LineSegment) (n : ℕ) : Prop :=
  ∃ p : ℝ, (segments.filter (λ s => s.left ≤ p ∧ p ≤ s.right)).length ≥ n

/-- Given a list of line segments, returns true if there exist n pairwise disjoint segments -/
def has_disjoint_segments (segments : List LineSegment) (n : ℕ) : Prop :=
  ∃ disjoint : List LineSegment, disjoint.length = n ∧
    ∀ i j, i < j → j < disjoint.length →
      (disjoint.get ⟨i, by sorry⟩).right < (disjoint.get ⟨j, by sorry⟩).left

/-- The main theorem -/
theorem line_segment_theorem (segments : List LineSegment) 
    (h : segments.length = 50) :
    has_common_point segments 8 ∨ has_disjoint_segments segments 8 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_theorem_l881_88168


namespace NUMINAMATH_CALUDE_max_prob_highest_second_l881_88106

/-- Represents a player in the chess game -/
structure Player where
  winProb : ℝ
  winProb_pos : winProb > 0

/-- Represents the chess game with three players -/
structure ChessGame where
  p₁ : Player
  p₂ : Player
  p₃ : Player
  prob_order : p₃.winProb > p₂.winProb ∧ p₂.winProb > p₁.winProb

/-- Calculates the probability of winning two consecutive games given the order of players -/
def probTwoConsecutiveWins (game : ChessGame) (second : Player) : ℝ :=
  2 * (second.winProb * (game.p₁.winProb + game.p₂.winProb + game.p₃.winProb - second.winProb) - 
       2 * game.p₁.winProb * game.p₂.winProb * game.p₃.winProb)

/-- Theorem stating that the probability of winning two consecutive games is maximized 
    when the player with the highest winning probability is played second -/
theorem max_prob_highest_second (game : ChessGame) :
  probTwoConsecutiveWins game game.p₃ ≥ probTwoConsecutiveWins game game.p₂ ∧
  probTwoConsecutiveWins game game.p₃ ≥ probTwoConsecutiveWins game game.p₁ := by
  sorry


end NUMINAMATH_CALUDE_max_prob_highest_second_l881_88106


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l881_88132

theorem inscribed_circle_radius (PQ PR QR : ℝ) (h_PQ : PQ = 30) (h_PR : PR = 26) (h_QR : QR = 28) :
  let s := (PQ + PR + QR) / 2
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  area / s = 8 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l881_88132


namespace NUMINAMATH_CALUDE_ellipse_k_value_l881_88136

/-- Theorem: For an ellipse with equation 5x^2 - ky^2 = 5 and one focus at (0, 2), the value of k is -1. -/
theorem ellipse_k_value (k : ℝ) : 
  (∃ (x y : ℝ), 5 * x^2 - k * y^2 = 5) → -- Ellipse equation
  (∃ (c : ℝ), c = 2 ∧ c^2 = 5 - (-5/k)) → -- Focus at (0, 2) and standard form relation
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_k_value_l881_88136


namespace NUMINAMATH_CALUDE_tournament_participants_l881_88110

theorem tournament_participants : ∃ (n : ℕ), n > 0 ∧ 
  (n * (n - 1) / 2 : ℚ) = 90 + (n - 10) * (n - 11) ∧ 
  (∀ k : ℕ, k ≠ n → (k * (k - 1) / 2 : ℚ) ≠ 90 + (k - 10) * (k - 11)) := by
  sorry

end NUMINAMATH_CALUDE_tournament_participants_l881_88110


namespace NUMINAMATH_CALUDE_scarves_per_box_l881_88100

theorem scarves_per_box (num_boxes : ℕ) (total_pieces : ℕ) : 
  num_boxes = 6 → 
  total_pieces = 60 → 
  ∃ (scarves_per_box : ℕ), 
    scarves_per_box * num_boxes * 2 = total_pieces ∧ 
    scarves_per_box = 5 :=
by sorry

end NUMINAMATH_CALUDE_scarves_per_box_l881_88100


namespace NUMINAMATH_CALUDE_root_sum_fraction_l881_88186

theorem root_sum_fraction (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 →
  q^3 - 8*q^2 + 10*q - 3 = 0 →
  r^3 - 8*r^2 + 10*r - 3 = 0 →
  p/(q*r + 1) + q/(p*r + 1) + r/(p*q + 1) = 59/22 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l881_88186


namespace NUMINAMATH_CALUDE_units_digit_of_sum_units_digit_of_power_units_digit_of_expression_l881_88183

theorem units_digit_of_sum (a b : ℕ) : ∃ (x y : ℕ), 
  x = a % 10 ∧ 
  y = b % 10 ∧ 
  (a + b) % 10 = (x + y) % 10 :=
by sorry

theorem units_digit_of_power (base exp : ℕ) : 
  (base ^ exp) % 10 = (base % 10 ^ (exp % 4 + 4)) % 10 :=
by sorry

theorem units_digit_of_expression : (5^12 + 4^2) % 10 = 1 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_units_digit_of_power_units_digit_of_expression_l881_88183


namespace NUMINAMATH_CALUDE_smallest_n_equal_l881_88151

/-- Geometric series C_n -/
def C (n : ℕ) : ℚ :=
  352 * (1 - (1/2)^n) / (1 - 1/2)

/-- Geometric series D_n -/
def D (n : ℕ) : ℚ :=
  992 * (1 - (1/(-2))^n) / (1 + 1/2)

/-- The smallest n ≥ 1 for which C_n = D_n is 1 -/
theorem smallest_n_equal (n : ℕ) (h : n ≥ 1) : (C n = D n) → n = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_equal_l881_88151


namespace NUMINAMATH_CALUDE_quadratic_root_square_relation_l881_88172

theorem quadratic_root_square_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = x^2) →
  b^2 = 3 * a * c + c^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_square_relation_l881_88172


namespace NUMINAMATH_CALUDE_yunas_grandfather_age_l881_88113

/-- Calculates the age of Yuna's grandfather given the ages and age differences of family members. -/
def grandfatherAge (yunaAge : ℕ) (fatherAgeDiff : ℕ) (grandfatherAgeDiff : ℕ) : ℕ :=
  yunaAge + fatherAgeDiff + grandfatherAgeDiff

/-- Proves that Yuna's grandfather is 59 years old given the provided conditions. -/
theorem yunas_grandfather_age :
  grandfatherAge 9 27 23 = 59 := by
  sorry

#eval grandfatherAge 9 27 23

end NUMINAMATH_CALUDE_yunas_grandfather_age_l881_88113


namespace NUMINAMATH_CALUDE_f_min_value_l881_88160

/-- The quadratic function f(x) = 5x^2 - 15x - 2 -/
def f (x : ℝ) : ℝ := 5 * x^2 - 15 * x - 2

/-- The minimum value of f(x) is -13.25 -/
theorem f_min_value : ∃ (x : ℝ), f x = -13.25 ∧ ∀ (y : ℝ), f y ≥ -13.25 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_l881_88160


namespace NUMINAMATH_CALUDE_orange_packing_l881_88178

/-- Given a fruit farm that packs oranges in boxes with a variable capacity,
    this theorem proves the relationship between the number of boxes used,
    the total number of oranges, and the capacity of each box. -/
theorem orange_packing (x : ℕ+) :
  (5623 : ℕ) / x.val = (5623 : ℕ) / x.val := by sorry

end NUMINAMATH_CALUDE_orange_packing_l881_88178


namespace NUMINAMATH_CALUDE_angle_measure_l881_88188

theorem angle_measure : ∃ x : ℝ, 
  (x + (5 * x + 12) = 180) ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l881_88188


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l881_88105

/-- A circle that passes through a fixed point and is tangent to a line -/
structure MovingCircle where
  center : ℝ × ℝ
  passes_through : center.1^2 + (center.2 - 1)^2 = (center.2 + 1)^2

/-- The trajectory of the center of the moving circle -/
def trajectory (c : MovingCircle) : Prop :=
  c.center.1^2 = 4 * c.center.2

/-- Theorem stating that the trajectory of the center is x^2 = 4y -/
theorem trajectory_is_parabola (c : MovingCircle) : trajectory c := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l881_88105


namespace NUMINAMATH_CALUDE_welch_distance_before_pie_l881_88153

/-- The distance Mr. Welch drove before buying a pie -/
def distance_before_pie (total_distance : ℕ) (distance_after_pie : ℕ) : ℕ :=
  total_distance - distance_after_pie

/-- Theorem: Mr. Welch drove 35 miles before buying a pie -/
theorem welch_distance_before_pie :
  distance_before_pie 78 43 = 35 := by
  sorry

end NUMINAMATH_CALUDE_welch_distance_before_pie_l881_88153


namespace NUMINAMATH_CALUDE_tank_capacity_l881_88176

/-- Represents a tank with a leak and an inlet pipe -/
structure Tank where
  capacity : ℝ
  leakRate : ℝ
  inletRate : ℝ

/-- The conditions of the problem -/
def tankProblem (t : Tank) : Prop :=
  t.leakRate = t.capacity / 6 ∧ 
  t.inletRate = 3.5 * 60 ∧ 
  t.inletRate - t.leakRate = t.capacity / 8

/-- The theorem stating that under the given conditions, the tank's capacity is 720 liters -/
theorem tank_capacity (t : Tank) : tankProblem t → t.capacity = 720 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l881_88176


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_digit_sum_l881_88115

/-- Function to create a number with n digits of 1 -/
def oneDigits (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Function to calculate the sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sumOfDigits (n / 10)

/-- Theorem stating that there are infinitely many integers divisible by the sum of their digits -/
theorem infinitely_many_divisible_by_digit_sum :
  ∀ n : ℕ, ∃ k : ℕ,
    k > 0 ∧
    (∀ d : ℕ, d > 0 → d < 10 → k % d ≠ 0) ∧ 
    (k % (sumOfDigits k) = 0) :=
by
  intro n
  use oneDigits (3^n)
  sorry

/-- Lemma: The number created by oneDigits(3^n) has exactly 3^n digits, all of which are 1 -/
lemma oneDigits_all_ones (n : ℕ) :
  ∀ d : ℕ, d > 0 → d < 10 → (oneDigits (3^n)) % d ≠ 0 :=
by sorry

/-- Lemma: The sum of digits of oneDigits(3^n) is equal to 3^n -/
lemma sum_of_digits_oneDigits (n : ℕ) :
  sumOfDigits (oneDigits (3^n)) = 3^n :=
by sorry

/-- Lemma: oneDigits(3^n) is divisible by 3^n -/
lemma oneDigits_divisible (n : ℕ) :
  (oneDigits (3^n)) % (3^n) = 0 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_digit_sum_l881_88115


namespace NUMINAMATH_CALUDE_common_chord_length_l881_88145

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y - 4 = 0
def circle_C2 (x y : ℝ) : Prop := (x + 3/2)^2 + (y - 3/2)^2 = 11/2

-- Theorem statement
theorem common_chord_length :
  ∃ (a b c d : ℝ),
    (circle_C1 a b ∧ circle_C1 c d) ∧
    (circle_C2 a b ∧ circle_C2 c d) ∧
    (a ≠ c ∨ b ≠ d) ∧
    Real.sqrt ((a - c)^2 + (b - d)^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l881_88145


namespace NUMINAMATH_CALUDE_peters_height_l881_88124

theorem peters_height (tree_height : ℝ) (tree_shadow : ℝ) (peter_shadow : ℝ) :
  tree_height = 100 → 
  tree_shadow = 25 → 
  peter_shadow = 1.5 → 
  (tree_height / tree_shadow) * peter_shadow * 12 = 72 := by
  sorry

end NUMINAMATH_CALUDE_peters_height_l881_88124


namespace NUMINAMATH_CALUDE_razorback_shop_revenue_l881_88159

theorem razorback_shop_revenue :
  let tshirt_price : ℕ := 98
  let hat_price : ℕ := 45
  let scarf_price : ℕ := 60
  let tshirts_sold : ℕ := 42
  let hats_sold : ℕ := 32
  let scarves_sold : ℕ := 15
  tshirt_price * tshirts_sold + hat_price * hats_sold + scarf_price * scarves_sold = 6456 :=
by sorry

end NUMINAMATH_CALUDE_razorback_shop_revenue_l881_88159


namespace NUMINAMATH_CALUDE_square_difference_39_40_square_41_from_40_l881_88174

theorem square_difference_39_40 :
  (40 : ℕ)^2 - (39 : ℕ)^2 = 79 :=
by
  sorry

-- Additional theorem to represent the given condition
theorem square_41_from_40 :
  (41 : ℕ)^2 = (40 : ℕ)^2 + 81 :=
by
  sorry

end NUMINAMATH_CALUDE_square_difference_39_40_square_41_from_40_l881_88174


namespace NUMINAMATH_CALUDE_possible_m_values_l881_88107

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem possible_m_values : 
  {m : ℝ | B m ⊆ A} = {-1/2, 0, 1/3} := by sorry

end NUMINAMATH_CALUDE_possible_m_values_l881_88107


namespace NUMINAMATH_CALUDE_book_pages_proof_l881_88143

/-- Proves that a book has 72 pages given the reading conditions -/
theorem book_pages_proof (total_days : ℕ) (fraction_per_day : ℚ) (extra_pages : ℕ) : 
  total_days = 3 → 
  fraction_per_day = 1/4 → 
  extra_pages = 6 → 
  (total_days : ℚ) * (fraction_per_day * (72 : ℚ) + extra_pages) = 72 := by
  sorry

#check book_pages_proof

end NUMINAMATH_CALUDE_book_pages_proof_l881_88143


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l881_88108

theorem trigonometric_equation_solution (k : ℤ) :
  (∃ x : ℝ, 4 - Real.sin x ^ 2 + Real.cos (4 * x) + Real.cos (2 * x) + 
   2 * Real.sin (3 * x) * Real.sin (7 * x) - Real.cos (7 * x) ^ 2 = 
   Real.cos (Real.pi * k / 2021) ^ 2) ↔ 
  (∃ m : ℤ, k = 2021 * m) ∧ 
  (∀ x : ℝ, 4 - Real.sin x ^ 2 + Real.cos (4 * x) + Real.cos (2 * x) + 
   2 * Real.sin (3 * x) * Real.sin (7 * x) - Real.cos (7 * x) ^ 2 = 
   Real.cos (Real.pi * k / 2021) ^ 2 → 
   ∃ n : ℤ, x = Real.pi / 4 + n * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l881_88108


namespace NUMINAMATH_CALUDE_triangle_properties_l881_88152

noncomputable section

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The triangle satisfies the given condition -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a + 2 * t.c = t.b * Real.cos t.C + Real.sqrt 3 * t.b * Real.sin t.C

theorem triangle_properties (t : Triangle) 
  (h : satisfiesCondition t) : 
  t.B = 2 * Real.pi / 3 ∧ 
  (t.b = 3 → 
    6 < t.a + t.b + t.c ∧ 
    t.a + t.b + t.c ≤ 3 + 2 * Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l881_88152


namespace NUMINAMATH_CALUDE_sin_A_in_special_triangle_l881_88150

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem sin_A_in_special_triangle (t : Triangle) (h1 : t.a = 8) (h2 : t.b = 7) (h3 : t.B = 30 * π / 180) :
  Real.sin t.A = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sin_A_in_special_triangle_l881_88150


namespace NUMINAMATH_CALUDE_children_count_l881_88195

theorem children_count (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 6) (h2 : total_pencils = 12) :
  total_pencils / pencils_per_child = 2 :=
by sorry

end NUMINAMATH_CALUDE_children_count_l881_88195


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l881_88103

/-- Represents the dimensions of a rectangular roof --/
structure RoofDimensions where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular roof --/
def area (r : RoofDimensions) : ℝ := r.width * r.length

/-- Theorem: For a rectangular roof with length 4 times its width and an area of 1024 square feet,
    the difference between the length and width is 48 feet. --/
theorem roof_dimension_difference (r : RoofDimensions) 
    (h1 : r.length = 4 * r.width) 
    (h2 : area r = 1024) : 
    r.length - r.width = 48 := by
  sorry


end NUMINAMATH_CALUDE_roof_dimension_difference_l881_88103


namespace NUMINAMATH_CALUDE_triangle_inequality_l881_88114

theorem triangle_inequality (a b c : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : a + b + c = 1) (h8 : n ≥ 2) :
  (a^n + b^n)^(1/n) + (b^n + c^n)^(1/n) + (c^n + a^n)^(1/n) < 1 + (2^(1/n))/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l881_88114


namespace NUMINAMATH_CALUDE_greatest_k_value_l881_88147

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 85) →
  k ≤ Real.sqrt 117 := by
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l881_88147


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l881_88196

/-- A geometric sequence with common ratio q -/
def geometric_sequence (q : ℝ) : ℕ → ℝ := fun n => q ^ (n - 1)

/-- The common ratio of a geometric sequence where a₄ = 27 and a₇ = -729 -/
theorem geometric_sequence_ratio : ∃ q : ℝ, 
  geometric_sequence q 4 = 27 ∧ 
  geometric_sequence q 7 = -729 ∧ 
  q = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l881_88196


namespace NUMINAMATH_CALUDE_equal_value_nickels_l881_88190

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters in the first set -/
def quarters_set1 : ℕ := 30

/-- The number of nickels in the first set -/
def nickels_set1 : ℕ := 15

/-- The number of quarters in the second set -/
def quarters_set2 : ℕ := 15

theorem equal_value_nickels : 
  ∃ n : ℕ, 
    quarters_set1 * quarter_value + nickels_set1 * nickel_value = 
    quarters_set2 * quarter_value + n * nickel_value ∧ 
    n = 90 := by
  sorry

end NUMINAMATH_CALUDE_equal_value_nickels_l881_88190


namespace NUMINAMATH_CALUDE_cloth_cost_price_theorem_l881_88180

/-- Calculates the cost price per meter of cloth given the total length,
    selling price, and profit per meter. -/
def costPricePerMeter (totalLength : ℕ) (sellingPrice : ℕ) (profitPerMeter : ℕ) : ℚ :=
  (sellingPrice - totalLength * profitPerMeter) / totalLength

/-- Theorem stating that for the given conditions, the cost price per meter is 86. -/
theorem cloth_cost_price_theorem (totalLength : ℕ) (sellingPrice : ℕ) (profitPerMeter : ℕ)
    (h1 : totalLength = 45)
    (h2 : sellingPrice = 4500)
    (h3 : profitPerMeter = 14) :
    costPricePerMeter totalLength sellingPrice profitPerMeter = 86 := by
  sorry

#eval costPricePerMeter 45 4500 14

end NUMINAMATH_CALUDE_cloth_cost_price_theorem_l881_88180


namespace NUMINAMATH_CALUDE_condition_D_not_sufficient_condition_A_sufficient_condition_B_sufficient_condition_C_sufficient_l881_88187

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles

-- Define similarity relation between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the four conditions
def condition_A (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β

def condition_B (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.γ = t2.γ

def condition_C (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

def condition_D (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b

-- Theorem stating that condition D is not sufficient for similarity
theorem condition_D_not_sufficient :
  ∃ t1 t2 : Triangle, condition_D t1 t2 ∧ ¬(similar t1 t2) := by sorry

-- Theorems stating that the other conditions are sufficient for similarity
theorem condition_A_sufficient (t1 t2 : Triangle) :
  condition_A t1 t2 → similar t1 t2 := by sorry

theorem condition_B_sufficient (t1 t2 : Triangle) :
  condition_B t1 t2 → similar t1 t2 := by sorry

theorem condition_C_sufficient (t1 t2 : Triangle) :
  condition_C t1 t2 → similar t1 t2 := by sorry

end NUMINAMATH_CALUDE_condition_D_not_sufficient_condition_A_sufficient_condition_B_sufficient_condition_C_sufficient_l881_88187


namespace NUMINAMATH_CALUDE_bill_calculation_l881_88161

/-- Given an initial bill amount, calculate the final amount after applying two successive late charges -/
def final_bill_amount (initial_amount : ℝ) (first_charge_rate : ℝ) (second_charge_rate : ℝ) : ℝ :=
  initial_amount * (1 + first_charge_rate) * (1 + second_charge_rate)

/-- Theorem: The final bill amount after applying late charges is $525.30 -/
theorem bill_calculation : 
  final_bill_amount 500 0.02 0.03 = 525.30 := by
  sorry

#eval final_bill_amount 500 0.02 0.03

end NUMINAMATH_CALUDE_bill_calculation_l881_88161
