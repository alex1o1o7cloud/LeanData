import Mathlib

namespace NUMINAMATH_CALUDE_prime_difference_theorem_l815_81541

theorem prime_difference_theorem (m n : ℕ) : 
  Nat.Prime m → Nat.Prime n → m - n^2 = 2007 → m * n = 4022 := by
  sorry

end NUMINAMATH_CALUDE_prime_difference_theorem_l815_81541


namespace NUMINAMATH_CALUDE_l_shaped_count_is_even_l815_81510

/-- A centrally symmetric figure on a grid --/
structure CentrallySymmetricFigure where
  n : ℕ  -- number of "L-shaped" figures
  k : ℕ  -- number of 1 × 4 rectangles

/-- Theorem: The number of "L-shaped" figures in a centrally symmetric figure is even --/
theorem l_shaped_count_is_even (figure : CentrallySymmetricFigure) : Even figure.n := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_count_is_even_l815_81510


namespace NUMINAMATH_CALUDE_remainder_sum_equals_27_l815_81552

theorem remainder_sum_equals_27 (a : ℕ) (h : a > 0) : 
  (50 % a + 72 % a + 157 % a = 27) → a = 21 :=
by sorry

end NUMINAMATH_CALUDE_remainder_sum_equals_27_l815_81552


namespace NUMINAMATH_CALUDE_right_angled_triangle_k_values_l815_81507

theorem right_angled_triangle_k_values (A B C : ℝ × ℝ) :
  let AB := B - A
  let AC := C - A
  AB = (2, 3) →
  AC = (1, k) →
  (AB.1 * AC.1 + AB.2 * AC.2 = 0 ∨
   AB.1 * (AC.1 - AB.1) + AB.2 * (AC.2 - AB.2) = 0 ∨
   AC.1 * (AB.1 - AC.1) + AC.2 * (AB.2 - AC.2) = 0) →
  k = -2/3 ∨ k = (3 + Real.sqrt 3)/2 ∨ k = (3 - Real.sqrt 3)/2 ∨ k = 11/3 :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_k_values_l815_81507


namespace NUMINAMATH_CALUDE_similar_triangles_area_l815_81581

/-- Given two similar triangles with corresponding sides of 1 cm and 2 cm, 
    and a total area of 25 cm², the area of the larger triangle is 20 cm². -/
theorem similar_triangles_area (A B : ℝ) : 
  A > 0 → B > 0 →  -- Areas are positive
  A + B = 25 →     -- Sum of areas is 25 cm²
  B / A = 4 →      -- Ratio of areas is 4 (square of the ratio of sides)
  B = 20 := by 
sorry

end NUMINAMATH_CALUDE_similar_triangles_area_l815_81581


namespace NUMINAMATH_CALUDE_max_profit_is_270000_l815_81595

/-- Represents the production quantities of products A and B -/
structure Production where
  a : ℝ
  b : ℝ

/-- Represents the constraints and profit calculation for the company -/
def Company :=
  { p : Production //
    p.a ≥ 0 ∧
    p.b ≥ 0 ∧
    3 * p.a + p.b ≤ 13 ∧
    2 * p.a + 3 * p.b ≤ 18 }

/-- Calculates the profit for a given production -/
def profit (p : Production) : ℝ := 50000 * p.a + 30000 * p.b

/-- Theorem stating that the maximum profit is 270,000 yuan -/
theorem max_profit_is_270000 :
  ∃ (p : Company), ∀ (q : Company), profit p.val ≥ profit q.val ∧ profit p.val = 270000 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_is_270000_l815_81595


namespace NUMINAMATH_CALUDE_leap_year_classification_l815_81589

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

theorem leap_year_classification :
  let leap_years : Set ℕ := {1992, 2040}
  let common_years : Set ℕ := {1800, 1994}
  (∀ y ∈ leap_years, is_leap_year y) ∧
  (∀ y ∈ common_years, ¬is_leap_year y) ∧
  (leap_years ∪ common_years = {1800, 1992, 1994, 2040}) :=
by sorry

end NUMINAMATH_CALUDE_leap_year_classification_l815_81589


namespace NUMINAMATH_CALUDE_combined_cost_price_is_430_95_l815_81532

-- Define the parameters for each stock
def stock1_face_value : ℝ := 100
def stock1_discount_rate : ℝ := 0.04
def stock1_brokerage_rate : ℝ := 0.002

def stock2_face_value : ℝ := 200
def stock2_discount_rate : ℝ := 0.06
def stock2_brokerage_rate : ℝ := 0.0025

def stock3_face_value : ℝ := 150
def stock3_discount_rate : ℝ := 0.03
def stock3_brokerage_rate : ℝ := 0.005

-- Define a function to calculate the cost price of a stock
def cost_price (face_value discount_rate brokerage_rate : ℝ) : ℝ :=
  (face_value - face_value * discount_rate) + face_value * brokerage_rate

-- Define the total cost price
def total_cost_price : ℝ :=
  cost_price stock1_face_value stock1_discount_rate stock1_brokerage_rate +
  cost_price stock2_face_value stock2_discount_rate stock2_brokerage_rate +
  cost_price stock3_face_value stock3_discount_rate stock3_brokerage_rate

-- Theorem statement
theorem combined_cost_price_is_430_95 :
  total_cost_price = 430.95 := by
  sorry

end NUMINAMATH_CALUDE_combined_cost_price_is_430_95_l815_81532


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l815_81524

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x^2 - x - 2 < 0 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ ¬(x^2 - x - 2 < 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l815_81524


namespace NUMINAMATH_CALUDE_solution_set_ln_inequality_l815_81563

theorem solution_set_ln_inequality :
  {x : ℝ | x > 0 ∧ 2 - Real.log x ≥ 0} = Set.Ioo 0 (Real.exp 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_ln_inequality_l815_81563


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l815_81569

/-- A geometric sequence is defined by its first term and common ratio -/
structure GeometricSequence where
  first_term : ℚ
  common_ratio : ℚ

/-- Get the nth term of a geometric sequence -/
def GeometricSequence.nth_term (seq : GeometricSequence) (n : ℕ) : ℚ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

/-- Theorem: In a geometric sequence where the 5th term is 48 and the 6th term is 72, the 2nd term is 1152/81 -/
theorem geometric_sequence_second_term
  (seq : GeometricSequence)
  (h5 : seq.nth_term 5 = 48)
  (h6 : seq.nth_term 6 = 72) :
  seq.nth_term 2 = 1152 / 81 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_second_term_l815_81569


namespace NUMINAMATH_CALUDE_divisor_expression_l815_81575

theorem divisor_expression (N D Y : ℕ) : 
  N = 45 * D + 13 → N = 6 * Y + 4 → D = (2 * Y - 3) / 15 := by
  sorry

end NUMINAMATH_CALUDE_divisor_expression_l815_81575


namespace NUMINAMATH_CALUDE_alice_max_plates_l815_81515

/-- Represents the shopping problem with pans, pots, and plates. -/
structure Shopping where
  pan_price : ℕ
  pot_price : ℕ
  plate_price : ℕ
  total_budget : ℕ
  min_pans : ℕ
  min_pots : ℕ

/-- Calculates the maximum number of plates that can be bought. -/
def max_plates (s : Shopping) : ℕ :=
  sorry

/-- The shopping problem instance as described in the question. -/
def alice_shopping : Shopping :=
  { pan_price := 3
  , pot_price := 5
  , plate_price := 11
  , total_budget := 100
  , min_pans := 2
  , min_pots := 2
  }

/-- Theorem stating that the maximum number of plates Alice can buy is 7. -/
theorem alice_max_plates :
  max_plates alice_shopping = 7 := by
  sorry

end NUMINAMATH_CALUDE_alice_max_plates_l815_81515


namespace NUMINAMATH_CALUDE_four_numbers_sum_l815_81572

theorem four_numbers_sum (a b c d T : ℝ) (h : a + b + c + d = T) :
  3 * ((a + 1) + (b + 1) + (c + 1) + (d + 1)) = 3 * T + 12 := by
  sorry

end NUMINAMATH_CALUDE_four_numbers_sum_l815_81572


namespace NUMINAMATH_CALUDE_added_amount_l815_81599

theorem added_amount (x : ℝ) (y : ℝ) (h1 : x = 6) (h2 : 2 / 3 * x + y = 10) : y = 6 := by
  sorry

end NUMINAMATH_CALUDE_added_amount_l815_81599


namespace NUMINAMATH_CALUDE_sum_even_odd_is_odd_l815_81578

theorem sum_even_odd_is_odd (a b : ℤ) (h1 : Even a) (h2 : Odd b) : Odd (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_even_odd_is_odd_l815_81578


namespace NUMINAMATH_CALUDE_circle_symmetry_l815_81553

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 7 = 0

-- Define a line in the plane
def line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define symmetry with respect to a line
def symmetric_wrt_line (C1 C2 : (ℝ → ℝ → Prop)) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), C1 x1 y1 ∧ C2 x2 y2 → 
    ∃ (x y : ℝ), l x y ∧ 
      (x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2

-- Theorem statement
theorem circle_symmetry :
  symmetric_wrt_line circle1 circle2 (line 1 (-1) 2) := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l815_81553


namespace NUMINAMATH_CALUDE_triangle_count_l815_81508

/-- The number of small triangles in the first section -/
def first_section_small : ℕ := 6

/-- The number of small triangles in the additional section -/
def additional_section_small : ℕ := 5

/-- The number of triangles made by combining 2 small triangles in the first section -/
def first_section_combined_2 : ℕ := 4

/-- The number of triangles made by combining 4 small triangles in the first section -/
def first_section_combined_4 : ℕ := 1

/-- The number of combined triangles in the additional section -/
def additional_section_combined : ℕ := 0

/-- The total number of triangles in the figure -/
def total_triangles : ℕ := 16

theorem triangle_count :
  first_section_small + additional_section_small +
  first_section_combined_2 + first_section_combined_4 +
  additional_section_combined = total_triangles := by sorry

end NUMINAMATH_CALUDE_triangle_count_l815_81508


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l815_81582

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l815_81582


namespace NUMINAMATH_CALUDE_complex_equality_l815_81550

/-- Given a real number b, if the real part is equal to the imaginary part
    for the complex number (1+i)/(1-i) + (1/2)b, then b = 2 -/
theorem complex_equality (b : ℝ) : 
  (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I) + (1 / 2 : ℂ) * b).re = 
  (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I) + (1 / 2 : ℂ) * b).im → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l815_81550


namespace NUMINAMATH_CALUDE_jonathan_weekly_caloric_deficit_l815_81574

/-- Jonathan's daily caloric intake on regular days -/
def regular_daily_intake : ℕ := 2500

/-- Jonathan's extra caloric intake on Saturday -/
def saturday_extra_intake : ℕ := 1000

/-- Jonathan's daily caloric burn -/
def daily_burn : ℕ := 3000

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Number of regular eating days in a week -/
def regular_days : ℕ := 6

/-- Calculate Jonathan's weekly caloric deficit -/
theorem jonathan_weekly_caloric_deficit :
  daily_burn * days_in_week - (regular_daily_intake * regular_days + (regular_daily_intake + saturday_extra_intake)) = 2500 := by
  sorry

end NUMINAMATH_CALUDE_jonathan_weekly_caloric_deficit_l815_81574


namespace NUMINAMATH_CALUDE_solution_count_l815_81587

/-- The number of positive integer solutions for a system of equations involving a prime number -/
def num_solutions (p : ℕ) : ℕ :=
  if p = 2 then 5
  else if p % 4 = 1 then 11
  else 3

/-- The main theorem stating the number of solutions for the given system of equations -/
theorem solution_count (p : ℕ) (hp : Nat.Prime p) :
  (∃ (n : ℕ), n = (Finset.filter (fun (quad : ℕ × ℕ × ℕ × ℕ) =>
    let (a, b, c, d) := quad
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a * c + b * d = p * (a + c) ∧
    b * c - a * d = p * (b - d))
    (Finset.product (Finset.range (p^3 + 1)) (Finset.product (Finset.range (p^3 + 1))
      (Finset.product (Finset.range (p^3 + 1)) (Finset.range (p^3 + 1)))))).card) ∧
  n = num_solutions p :=
sorry

end NUMINAMATH_CALUDE_solution_count_l815_81587


namespace NUMINAMATH_CALUDE_distinct_convex_polygons_l815_81547

/-- The number of points marked on the circle -/
def num_points : ℕ := 12

/-- The total number of subsets of the points -/
def total_subsets : ℕ := 2^num_points

/-- The number of subsets with 0 members -/
def subsets_0 : ℕ := (num_points.choose 0)

/-- The number of subsets with 1 member -/
def subsets_1 : ℕ := (num_points.choose 1)

/-- The number of subsets with 2 members -/
def subsets_2 : ℕ := (num_points.choose 2)

/-- The number of distinct convex polygons with three or more sides -/
def num_polygons : ℕ := total_subsets - subsets_0 - subsets_1 - subsets_2

theorem distinct_convex_polygons :
  num_polygons = 4017 :=
by sorry

end NUMINAMATH_CALUDE_distinct_convex_polygons_l815_81547


namespace NUMINAMATH_CALUDE_equal_sundays_tuesdays_count_l815_81514

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a 30-day month -/
structure Month30 where
  firstDay : DayOfWeek

/-- Function to check if a 30-day month has equal Sundays and Tuesdays -/
def hasEqualSundaysAndTuesdays (m : Month30) : Prop :=
  -- Implementation details omitted
  sorry

/-- The number of possible starting days for a 30-day month with equal Sundays and Tuesdays -/
theorem equal_sundays_tuesdays_count :
  (∃ (days : Finset DayOfWeek),
    (∀ d : DayOfWeek, d ∈ days ↔ hasEqualSundaysAndTuesdays ⟨d⟩) ∧
    Finset.card days = 6) :=
  sorry

end NUMINAMATH_CALUDE_equal_sundays_tuesdays_count_l815_81514


namespace NUMINAMATH_CALUDE_existence_of_indistinguishable_arrangements_l815_81544

/-- Represents the type of a tree -/
inductive TreeType
| Oak
| Baobab

/-- Represents a row of trees -/
def TreeRow := List TreeType

/-- Counts the number of oaks in a group of three adjacent trees -/
def countOaks (trees : TreeRow) (index : Nat) : Nat :=
  match trees.get? index, trees.get? (index + 1), trees.get? (index + 2) with
  | some TreeType.Oak, _, _ => 1
  | _, some TreeType.Oak, _ => 1
  | _, _, some TreeType.Oak => 1
  | _, _, _ => 0

/-- Generates the sequence of tag numbers for a given row of trees -/
def generateTags (trees : TreeRow) : List Nat :=
  List.range trees.length |>.map (countOaks trees)

/-- Theorem stating that there exist two different arrangements of trees
    with the same tag sequence -/
theorem existence_of_indistinguishable_arrangements :
  ∃ (row1 row2 : TreeRow),
    row1.length = 2000 ∧
    row2.length = 2000 ∧
    row1 ≠ row2 ∧
    generateTags row1 = generateTags row2 :=
sorry

end NUMINAMATH_CALUDE_existence_of_indistinguishable_arrangements_l815_81544


namespace NUMINAMATH_CALUDE_prime_pairs_square_sum_l815_81579

theorem prime_pairs_square_sum (p q : ℕ) : 
  Prime p → Prime q → (∃ n : ℕ, p^2 + 5*p*q + 4*q^2 = n^2) → 
  ((p = 13 ∧ q = 3) ∨ (p = 7 ∧ q = 5) ∨ (p = 5 ∧ q = 11)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_square_sum_l815_81579


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l815_81580

theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.mk (m^2 - 8*m + 15) (m^2 - 9*m + 18)).im ≠ 0 ∧ 
  (Complex.mk (m^2 - 8*m + 15) (m^2 - 9*m + 18)).re = 0 → 
  m = 5 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l815_81580


namespace NUMINAMATH_CALUDE_excellent_scorers_l815_81566

-- Define the set of students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

-- Define a function to represent whether a student scores excellent
def scores_excellent : Student → Prop := sorry

-- Define the statements made by each student
def statement_A : Prop := scores_excellent Student.A → scores_excellent Student.B
def statement_B : Prop := scores_excellent Student.B → scores_excellent Student.C
def statement_C : Prop := scores_excellent Student.C → scores_excellent Student.D
def statement_D : Prop := scores_excellent Student.D → scores_excellent Student.E

-- Define a function to count the number of students scoring excellent
def count_excellent : (Student → Prop) → Nat := sorry

-- Theorem statement
theorem excellent_scorers :
  (statement_A ∧ statement_B ∧ statement_C ∧ statement_D) →
  (count_excellent scores_excellent = 3) →
  (scores_excellent Student.C ∧ scores_excellent Student.D ∧ scores_excellent Student.E ∧
   ¬scores_excellent Student.A ∧ ¬scores_excellent Student.B) :=
sorry

end NUMINAMATH_CALUDE_excellent_scorers_l815_81566


namespace NUMINAMATH_CALUDE_race_time_comparison_l815_81526

theorem race_time_comparison 
  (a : ℝ) (V : ℝ) 
  (h1 : a > 0) (h2 : V > 0) : 
  let planned_time := a / V
  let first_half_time := a / (2 * 1.25 * V)
  let second_half_time := a / (2 * 0.8 * V)
  let actual_time := first_half_time + second_half_time
  actual_time > planned_time :=
by sorry

end NUMINAMATH_CALUDE_race_time_comparison_l815_81526


namespace NUMINAMATH_CALUDE_expression_equals_one_l815_81512

theorem expression_equals_one : 
  (121^2 - 13^2) / (91^2 - 17^2) * ((91-17)*(91+17)) / ((121-13)*(121+13)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l815_81512


namespace NUMINAMATH_CALUDE_intersection_N_complement_M_l815_81568

-- Define the universe U
def U : Finset ℕ := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset ℕ := {1, 4}

-- Define set N
def N : Finset ℕ := {1, 3, 5}

-- Theorem statement
theorem intersection_N_complement_M : N ∩ (U \ M) = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_N_complement_M_l815_81568


namespace NUMINAMATH_CALUDE_power_calculation_l815_81519

theorem power_calculation : 2^345 - 8^3 / 8^2 + 3^2 = 2^345 + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l815_81519


namespace NUMINAMATH_CALUDE_circle_through_ellipse_vertices_l815_81517

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point lies on an ellipse -/
def Point.onEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Check if a point lies on a circle -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The main theorem to prove -/
theorem circle_through_ellipse_vertices (e : Ellipse) (c : Circle) : 
  e.a = 4 ∧ e.b = 2 ∧ 
  c.center.x = 3/2 ∧ c.center.y = 0 ∧ c.radius = 5/2 →
  (∃ (p1 p2 p3 : Point), 
    p1.onEllipse e ∧ p2.onEllipse e ∧ p3.onEllipse e ∧
    p1.onCircle c ∧ p2.onCircle c ∧ p3.onCircle c) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_through_ellipse_vertices_l815_81517


namespace NUMINAMATH_CALUDE_complex_equation_solution_l815_81585

theorem complex_equation_solution (m n : ℝ) (i : ℂ) 
  (h1 : i * i = -1)
  (h2 : m / (1 + i) = 1 - n * i) : 
  m - n = 1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l815_81585


namespace NUMINAMATH_CALUDE_cut_prism_surface_area_l815_81560

/-- Represents a rectangular prism with a cube cut out from one corner. -/
structure CutPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  cutSize : ℝ

/-- Calculates the surface area of a CutPrism. -/
def surfaceArea (p : CutPrism) : ℝ :=
  2 * (p.length * p.width + p.width * p.height + p.length * p.height)

/-- Theorem: The surface area of a 4 by 2 by 2 rectangular prism with a 1 by 1 by 1 cube
    cut out from one corner is equal to 40 square units. -/
theorem cut_prism_surface_area :
  let p : CutPrism := { length := 4, width := 2, height := 2, cutSize := 1 }
  surfaceArea p = 40 := by
  sorry

end NUMINAMATH_CALUDE_cut_prism_surface_area_l815_81560


namespace NUMINAMATH_CALUDE_isabella_trip_l815_81577

def exchange_rate : ℚ := 8 / 5

def spent_aud : ℕ := 80

def remaining_aud (e : ℕ) : ℕ := e + 20

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.repr.toList.map (fun c => c.toNat - '0'.toNat)
  digits.sum

theorem isabella_trip (e : ℕ) : 
  (exchange_rate * e : ℚ) - spent_aud = remaining_aud e →
  e = 167 ∧ sum_of_digits e = 14 := by
  sorry

end NUMINAMATH_CALUDE_isabella_trip_l815_81577


namespace NUMINAMATH_CALUDE_apples_in_crate_l815_81564

/-- The number of apples in a crate -/
def apples_per_crate : ℕ := sorry

/-- The number of crates delivered -/
def crates_delivered : ℕ := 12

/-- The number of rotten apples -/
def rotten_apples : ℕ := 4

/-- The number of apples that fit in each box -/
def apples_per_box : ℕ := 10

/-- The number of boxes filled with good apples -/
def filled_boxes : ℕ := 50

theorem apples_in_crate :
  apples_per_crate * crates_delivered = filled_boxes * apples_per_box + rotten_apples ∧
  apples_per_crate = 42 := by sorry

end NUMINAMATH_CALUDE_apples_in_crate_l815_81564


namespace NUMINAMATH_CALUDE_inequality_equivalence_l815_81597

def inequality_solution (x : ℝ) : Prop :=
  (x - 1) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0

def solution_set (x : ℝ) : Prop :=
  x < 1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 4) ∨ (4 < x ∧ x < 5) ∨ 7 < x

theorem inequality_equivalence :
  ∀ x : ℝ, inequality_solution x ↔ solution_set x := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l815_81597


namespace NUMINAMATH_CALUDE_parabola_constant_term_l815_81594

theorem parabola_constant_term (p q : ℝ) : 
  (∀ x y : ℝ, y = x^2 + p*x + q → 
    ((x = 3 ∧ y = 4) ∨ (x = 5 ∧ y = 4))) → 
  q = 19 := by
sorry

end NUMINAMATH_CALUDE_parabola_constant_term_l815_81594


namespace NUMINAMATH_CALUDE_earnings_difference_main_theorem_l815_81558

/-- Given investment ratios, return ratios, and total earnings, 
    calculate the difference between earnings of b and a -/
theorem earnings_difference 
  (invest_a invest_b invest_c : ℚ) 
  (return_a return_b return_c : ℚ) 
  (total_earnings : ℚ) : ℚ :=
  let earnings_a := invest_a * return_a
  let earnings_b := invest_b * return_b
  let earnings_c := invest_c * return_c
  by
    have h1 : invest_b / invest_a = 4 / 3 := by sorry
    have h2 : invest_c / invest_a = 5 / 3 := by sorry
    have h3 : return_b / return_a = 5 / 6 := by sorry
    have h4 : return_c / return_a = 4 / 6 := by sorry
    have h5 : earnings_a + earnings_b + earnings_c = total_earnings := by sorry
    have h6 : total_earnings = 4350 := by sorry
    exact 150

/-- The main theorem stating the difference in earnings -/
theorem main_theorem : earnings_difference 3 4 5 6 5 4 4350 = 150 := by sorry

end NUMINAMATH_CALUDE_earnings_difference_main_theorem_l815_81558


namespace NUMINAMATH_CALUDE_nine_times_2010_equals_201_l815_81548

-- Define the operation
def diamond (a b : ℚ) : ℚ := (a * b) / (a + b)

-- Define a function that applies the operation n times
def apply_n_times (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => diamond (apply_n_times n x) x

-- Theorem statement
theorem nine_times_2010_equals_201 :
  apply_n_times 9 2010 = 201 := by sorry

end NUMINAMATH_CALUDE_nine_times_2010_equals_201_l815_81548


namespace NUMINAMATH_CALUDE_negation_false_l815_81503

theorem negation_false : ¬∃ (x y : ℝ), x > 2 ∧ y > 3 ∧ x + y ≤ 5 := by sorry

end NUMINAMATH_CALUDE_negation_false_l815_81503


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l815_81533

def f (x : ℝ) := -x^2 + 2

theorem max_min_values_of_f :
  let a := -1
  let b := 3
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 2 ∧ f x_min = -7 :=
sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l815_81533


namespace NUMINAMATH_CALUDE_total_pushups_count_l815_81570

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 44

/-- The additional number of push-ups David did compared to Zachary -/
def david_extra_pushups : ℕ := 58

/-- The total number of push-ups done by Zachary and David -/
def total_pushups : ℕ := zachary_pushups + (zachary_pushups + david_extra_pushups)

/-- Theorem stating the total number of push-ups done by Zachary and David -/
theorem total_pushups_count : total_pushups = 146 := by
  sorry

end NUMINAMATH_CALUDE_total_pushups_count_l815_81570


namespace NUMINAMATH_CALUDE_unique_solution_l815_81554

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- Checks if a number has the pattern 1*1 -/
def hasPattern1x1 (n : ThreeDigitNumber) : Prop :=
  n.val / 100 = 1 ∧ n.val % 10 = 1

theorem unique_solution :
  ∀ (ab cd : TwoDigitNumber) (n : ThreeDigitNumber),
    ab.val * cd.val = n.val ∧ hasPattern1x1 n →
    ab.val = 11 ∧ cd.val = 11 ∧ n.val = 121 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l815_81554


namespace NUMINAMATH_CALUDE_class_fraction_proof_l815_81504

/-- 
Given a class of students where:
1) The ratio of boys to girls is 2
2) Half the number of girls is equal to some fraction of the total number of students
This theorem proves that the fraction in condition 2 is 1/6
-/
theorem class_fraction_proof (G : ℚ) (h1 : G > 0) : 
  let B := 2 * G
  let total := G + B
  ∃ (x : ℚ), (1/2) * G = x * total ∧ x = 1/6 := by
sorry

end NUMINAMATH_CALUDE_class_fraction_proof_l815_81504


namespace NUMINAMATH_CALUDE_unique_f_l815_81527

def is_valid_f (f : Nat → Nat) : Prop :=
  ∀ n m : Nat, n > 1 → m > 1 → n ≠ m → f n * f m = f ((n * m) ^ 2021)

theorem unique_f : 
  ∀ f : Nat → Nat, is_valid_f f → (∀ x : Nat, x > 1 → f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_f_l815_81527


namespace NUMINAMATH_CALUDE_equation_satisfied_l815_81583

theorem equation_satisfied (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l815_81583


namespace NUMINAMATH_CALUDE_expression_value_l815_81521

theorem expression_value (a b : ℚ) (ha : a = -1) (hb : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l815_81521


namespace NUMINAMATH_CALUDE_rectangle_tromino_subdivision_l815_81551

theorem rectangle_tromino_subdivision (a b c d : ℕ) : 
  a = 1961 ∧ b = 1963 ∧ c = 1963 ∧ d = 1965 → 
  (¬(a * b % 3 = 0) ∧ c * d % 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_tromino_subdivision_l815_81551


namespace NUMINAMATH_CALUDE_average_rate_of_change_f_l815_81562

-- Define the function f(x) = x^2 + 2
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the interval [1, 3]
def a : ℝ := 1
def b : ℝ := 3

-- Theorem: The average rate of change of f(x) on [1, 3] is 4
theorem average_rate_of_change_f : (f b - f a) / (b - a) = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_of_change_f_l815_81562


namespace NUMINAMATH_CALUDE_express_x_in_terms_of_y_l815_81596

theorem express_x_in_terms_of_y (x y : ℝ) (h : 2 * x - 3 * y = 7) : 
  x = 7 / 2 + 3 / 2 * y := by
  sorry

end NUMINAMATH_CALUDE_express_x_in_terms_of_y_l815_81596


namespace NUMINAMATH_CALUDE_youngest_child_age_l815_81501

/-- Given 5 children born at intervals of 3 years, if the sum of their ages is 50 years,
    then the age of the youngest child is 4 years. -/
theorem youngest_child_age (children : ℕ) (interval : ℕ) (total_age : ℕ) :
  children = 5 →
  interval = 3 →
  total_age = 50 →
  total_age = (children - 1) * children / 2 * interval + children * (youngest_age : ℕ) →
  youngest_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_l815_81501


namespace NUMINAMATH_CALUDE_unit_vector_of_difference_l815_81542

/-- Given vectors a and b in ℝ², prove that the unit vector of a - b is (-4/5, 3/5) -/
theorem unit_vector_of_difference (a b : ℝ × ℝ) (ha : a = (3, 1)) (hb : b = (7, -2)) :
  let diff := a - b
  let norm := Real.sqrt ((diff.1)^2 + (diff.2)^2)
  (diff.1 / norm, diff.2 / norm) = (-4/5, 3/5) := by
sorry

end NUMINAMATH_CALUDE_unit_vector_of_difference_l815_81542


namespace NUMINAMATH_CALUDE_remainder_two_power_33_minus_one_mod_9_l815_81518

theorem remainder_two_power_33_minus_one_mod_9 : 2^33 - 1 ≡ 7 [ZMOD 9] := by
  sorry

end NUMINAMATH_CALUDE_remainder_two_power_33_minus_one_mod_9_l815_81518


namespace NUMINAMATH_CALUDE_exponential_comparison_l815_81528

theorem exponential_comparison (h1 : 1.5 > 1) (h2 : 2.3 < 3.2) :
  1.5^2.3 < 1.5^3.2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_comparison_l815_81528


namespace NUMINAMATH_CALUDE_triangle_problem_l815_81588

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and vectors m and n, prove that B = π/3 and the maximum area is √3 --/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  let m : ℝ × ℝ := (2 * Real.sin B, -Real.sqrt 3)
  let n : ℝ × ℝ := (Real.cos (2 * B), 2 * (Real.cos (B / 2))^2 - 1)
  b = 2 →
  (∃ (k : ℝ), m.1 = k * n.1 ∧ m.2 = k * n.2) →
  B = π / 3 ∧
  (∀ (S : ℝ), S = 1/2 * a * c * Real.sin B → S ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l815_81588


namespace NUMINAMATH_CALUDE_softball_opponent_score_l815_81537

theorem softball_opponent_score :
  let team_scores : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let games_lost_by_one : Nat := 5
  let other_games_score_ratio : Nat := 2
  let opponent_scores : List Nat := 
    team_scores.map (fun score => 
      if score % 2 = 1 then score + 1
      else score / other_games_score_ratio)
  opponent_scores.sum = 45 := by
  sorry

end NUMINAMATH_CALUDE_softball_opponent_score_l815_81537


namespace NUMINAMATH_CALUDE_amp_five_two_squared_l815_81586

/-- The & operation defined for real numbers -/
def amp (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Theorem stating that (5 & 2)^2 = 441 -/
theorem amp_five_two_squared : (amp 5 2)^2 = 441 := by
  sorry

end NUMINAMATH_CALUDE_amp_five_two_squared_l815_81586


namespace NUMINAMATH_CALUDE_ellipse_intersection_length_l815_81584

-- Define the ellipse (C)
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Define the line passing through (0, 2) with slope 1
def line_l (x y : ℝ) : Prop :=
  y = x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem ellipse_intersection_length :
  -- Given conditions
  let F₁ : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
  let F₂ : ℝ × ℝ := (2 * Real.sqrt 2, 0)
  let major_axis_length : ℝ := 6
  
  -- Prove the following
  ∀ A B : ℝ × ℝ, intersection_points A B →
    -- 1. The standard equation of the ellipse is correct
    (∀ x y : ℝ, (x^2 / 9 + y^2 = 1) ↔ ellipse_C x y) ∧
    -- 2. The length of AB is 6√3/5
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 * Real.sqrt 3 / 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_intersection_length_l815_81584


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l815_81549

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : Nat, p.Prime ∧ p ∣ (36^2 + 49^2) ∧ ∀ q : Nat, q.Prime → q ∣ (36^2 + 49^2) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l815_81549


namespace NUMINAMATH_CALUDE_cookie_sales_proof_l815_81522

/-- Represents the total number of boxes of cookies sold -/
def total_boxes (chocolate_chip_boxes : ℝ) (plain_boxes : ℝ) : ℝ :=
  chocolate_chip_boxes + plain_boxes

/-- Represents the total sales value -/
def total_sales (chocolate_chip_boxes : ℝ) (plain_boxes : ℝ) : ℝ :=
  1.25 * chocolate_chip_boxes + 0.75 * plain_boxes

theorem cookie_sales_proof :
  ∀ (chocolate_chip_boxes : ℝ) (plain_boxes : ℝ),
    plain_boxes = 793.375 →
    total_sales chocolate_chip_boxes plain_boxes = 1586.75 →
    total_boxes chocolate_chip_boxes plain_boxes = 1586.75 :=
by
  sorry

#check cookie_sales_proof

end NUMINAMATH_CALUDE_cookie_sales_proof_l815_81522


namespace NUMINAMATH_CALUDE_vector_properties_l815_81592

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (a b : V)

-- Define the conditions
def non_collinear (a b : V) : Prop := ¬ ∃ (k : ℝ), a = k • b
def same_starting_point (a b : V) : Prop := True  -- This is implicitly assumed in the vector space
def equal_magnitude (a b : V) : Prop := ‖a‖ = ‖b‖
def angle_60_degrees (a b : V) : Prop := inner a b = (1/2 : ℝ) * ‖a‖ * ‖b‖

-- Define the theorem
theorem vector_properties
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : non_collinear V a b)
  (h4 : same_starting_point V a b)
  (h5 : equal_magnitude V a b)
  (h6 : angle_60_degrees V a b) :
  (∃ (k : ℝ), k • ((1/2 : ℝ) • b - a) = (1/3 : ℝ) • b - (2/3 : ℝ) • a) ∧
  (∀ (t : ℝ), ‖a - (1/2 : ℝ) • b‖ ≤ ‖a - t • b‖) :=
sorry

end NUMINAMATH_CALUDE_vector_properties_l815_81592


namespace NUMINAMATH_CALUDE_power_zero_minus_one_equals_zero_l815_81590

theorem power_zero_minus_one_equals_zero : 2^0 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_minus_one_equals_zero_l815_81590


namespace NUMINAMATH_CALUDE_f_derivative_l815_81505

def f (x : ℝ) : ℝ := -3 * x - 1

theorem f_derivative : 
  deriv f = λ x => -3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_l815_81505


namespace NUMINAMATH_CALUDE_tony_cheese_purchase_l815_81500

theorem tony_cheese_purchase (initial_amount : ℕ) (cheese_cost : ℕ) (beef_cost : ℕ) (remaining_amount : ℕ)
  (h1 : initial_amount = 87)
  (h2 : cheese_cost = 7)
  (h3 : beef_cost = 5)
  (h4 : remaining_amount = 61) :
  (initial_amount - remaining_amount - beef_cost) / cheese_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_tony_cheese_purchase_l815_81500


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l815_81530

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the half-open interval [1, 2)
def interval : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- Theorem statement
theorem intersection_equals_interval : M ∩ N = interval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l815_81530


namespace NUMINAMATH_CALUDE_hiker_distance_hiker_distance_proof_l815_81523

/-- The straight-line distance a hiker travels after walking 8 miles east,
    turning 45 degrees north, and walking another 8 miles. -/
theorem hiker_distance : ℝ :=
  let initial_east_distance : ℝ := 8
  let turn_angle : ℝ := 45
  let second_walk_distance : ℝ := 8
  let final_distance : ℝ := 4 * Real.sqrt (6 + 4 * Real.sqrt 2)
  final_distance

/-- Proof that the hiker's final straight-line distance from the starting point
    is 4√(6 + 4√2) miles. -/
theorem hiker_distance_proof :
  hiker_distance = 4 * Real.sqrt (6 + 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_hiker_distance_proof_l815_81523


namespace NUMINAMATH_CALUDE_sum_f_positive_l815_81520

noncomputable def f (x : ℝ) : ℝ := x^3 / Real.cos x

theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : |x₁| < π/2) (h₂ : |x₂| < π/2) (h₃ : |x₃| < π/2)
  (h₄ : x₁ + x₂ > 0) (h₅ : x₂ + x₃ > 0) (h₆ : x₁ + x₃ > 0) :
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l815_81520


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l815_81502

theorem roots_sum_and_product (a b : ℝ) : 
  a^4 - 6*a^3 + 11*a^2 - 6*a - 1 = 0 →
  b^4 - 6*b^3 + 11*b^2 - 6*b - 1 = 0 →
  a + b + a*b = 4 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l815_81502


namespace NUMINAMATH_CALUDE_restricted_photo_arrangements_l815_81539

/-- The number of ways to arrange n people in a line -/
def lineArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line where one specific person is restricted -/
def restrictedArrangements (n : ℕ) : ℕ := (n - 2) * Nat.factorial (n - 1)

/-- Theorem stating that for 5 people, with one person restricted from ends, there are 72 arrangements -/
theorem restricted_photo_arrangements :
  restrictedArrangements 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_restricted_photo_arrangements_l815_81539


namespace NUMINAMATH_CALUDE_jose_share_of_profit_l815_81546

/-- Calculates the share of profit for an investor given the total profit and investment ratios. -/
def calculate_share_of_profit (total_profit : ℚ) (investor_ratio : ℚ) (total_ratio : ℚ) : ℚ :=
  (investor_ratio / total_ratio) * total_profit

/-- Represents the problem of calculating Jose's share of profit in a business partnership. -/
theorem jose_share_of_profit 
  (tom_investment : ℚ) (tom_duration : ℕ) 
  (jose_investment : ℚ) (jose_duration : ℕ) 
  (total_profit : ℚ) : 
  tom_investment = 30000 → 
  tom_duration = 12 → 
  jose_investment = 45000 → 
  jose_duration = 10 → 
  total_profit = 54000 → 
  calculate_share_of_profit total_profit (jose_investment * jose_duration) 
    (tom_investment * tom_duration + jose_investment * jose_duration) = 30000 := by
  sorry

end NUMINAMATH_CALUDE_jose_share_of_profit_l815_81546


namespace NUMINAMATH_CALUDE_ellipse_chord_theorem_l815_81576

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with center at origin -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Defines if a point lies on an ellipse -/
def Point.onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Defines if a point is the midpoint of two other points -/
def isMidpoint (m p1 p2 : Point) : Prop :=
  m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2

/-- Defines if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem ellipse_chord_theorem (e : Ellipse) (p1 p2 m : Point) :
  e.a = 6 ∧ e.b = 3 →
  p1.onEllipse e ∧ p2.onEllipse e →
  isMidpoint m p1 p2 →
  m = Point.mk 4 2 →
  areCollinear p1 p2 (Point.mk 0 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_chord_theorem_l815_81576


namespace NUMINAMATH_CALUDE_divisibility_problem_l815_81511

theorem divisibility_problem (x y : ℤ) (h : 5 ∣ (x + 9*y)) : 5 ∣ (8*x + 7*y) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l815_81511


namespace NUMINAMATH_CALUDE_subtraction_for_complex_equality_l815_81545

theorem subtraction_for_complex_equality : ∃ (z : ℂ), (7 - 3*I) - z = 3 * ((2 + I) + (4 - 2*I)) ∧ z = -11 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_for_complex_equality_l815_81545


namespace NUMINAMATH_CALUDE_power_ranger_stickers_l815_81531

theorem power_ranger_stickers (total : ℕ) (first_box : ℕ) : 
  total = 58 → first_box = 23 → (total - first_box) - first_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_ranger_stickers_l815_81531


namespace NUMINAMATH_CALUDE_baker_cake_difference_l815_81525

/-- Given the initial number of cakes, the number of cakes sold, and the number of cakes bought,
    prove that the difference between cakes sold and cakes bought is 47. -/
theorem baker_cake_difference (initial : ℕ) (sold : ℕ) (bought : ℕ)
    (h1 : initial = 170)
    (h2 : sold = 78)
    (h3 : bought = 31) :
  sold - bought = 47 := by
  sorry

end NUMINAMATH_CALUDE_baker_cake_difference_l815_81525


namespace NUMINAMATH_CALUDE_f_decreasing_range_l815_81535

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else a/x

/-- Theorem stating the range of a for f(x) to be decreasing -/
theorem f_decreasing_range (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) →
  1/6 ≤ a ∧ a < 1/3 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_range_l815_81535


namespace NUMINAMATH_CALUDE_crate_stack_probability_l815_81540

-- Define the dimensions of a crate
def CrateDimensions : Fin 3 → ℕ
  | 0 => 4
  | 1 => 5
  | 2 => 7

-- Define the number of crates
def NumCrates : ℕ := 15

-- Define the target height
def TargetHeight : ℕ := 50

-- Define the total number of possible arrangements
def TotalArrangements : ℕ := 3^NumCrates

-- Define the number of favorable arrangements
def FavorableArrangements : ℕ := 560

theorem crate_stack_probability :
  (FavorableArrangements : ℚ) / TotalArrangements = 560 / 14348907 := by
  sorry

#eval FavorableArrangements -- Should output 560

end NUMINAMATH_CALUDE_crate_stack_probability_l815_81540


namespace NUMINAMATH_CALUDE_fraction_simplification_l815_81561

theorem fraction_simplification :
  (30 : ℚ) / 45 * 75 / 128 * 256 / 150 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l815_81561


namespace NUMINAMATH_CALUDE_remainder_of_power_minus_digit_l815_81509

theorem remainder_of_power_minus_digit (x : ℕ) : 
  x < 10 → (Nat.pow 2 200 - x) % 7 = 1 → x = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_of_power_minus_digit_l815_81509


namespace NUMINAMATH_CALUDE_M_values_l815_81534

theorem M_values (a b : ℚ) (h : a * b ≠ 0) :
  let M := |a| / a + b / |b|
  M = 0 ∨ M = 2 ∨ M = -2 := by
sorry

end NUMINAMATH_CALUDE_M_values_l815_81534


namespace NUMINAMATH_CALUDE_masha_comb_teeth_l815_81571

/-- Represents a comb with teeth --/
structure Comb where
  numTeeth : ℕ
  numGaps : ℕ
  numSegments : ℕ

/-- The relationship between teeth and gaps in a comb --/
axiom comb_structure (c : Comb) : c.numGaps = c.numTeeth - 1

/-- The total number of segments in a comb --/
axiom comb_segments (c : Comb) : c.numSegments = c.numTeeth + c.numGaps

/-- Katya's comb --/
def katya_comb : Comb := { numTeeth := 11, numGaps := 10, numSegments := 21 }

/-- Masha's comb --/
def masha_comb : Comb := { numTeeth := 53, numGaps := 52, numSegments := 105 }

/-- The relationship between Katya's and Masha's combs --/
axiom comb_relationship : masha_comb.numSegments = 5 * katya_comb.numSegments

theorem masha_comb_teeth : masha_comb.numTeeth = 53 := by
  sorry

end NUMINAMATH_CALUDE_masha_comb_teeth_l815_81571


namespace NUMINAMATH_CALUDE_inequality_solution_set_l815_81513

theorem inequality_solution_set (x : ℝ) : (1 - x > x - 1) ↔ (x < 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l815_81513


namespace NUMINAMATH_CALUDE_lcm_48_75_l815_81516

theorem lcm_48_75 : Nat.lcm 48 75 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_75_l815_81516


namespace NUMINAMATH_CALUDE_abs_is_even_and_increasing_l815_81529

-- Define the absolute value function
def f (x : ℝ) := abs x

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define what it means for a function to be increasing on an interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem abs_is_even_and_increasing :
  is_even f ∧ is_increasing_on f 0 1 :=
sorry

end NUMINAMATH_CALUDE_abs_is_even_and_increasing_l815_81529


namespace NUMINAMATH_CALUDE_sum_of_products_l815_81559

theorem sum_of_products (a b c d : ℝ) 
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 5)
  (eq3 : a + c + d = 14)
  (eq4 : b + c + d = 9) :
  a * b + c * d = 338 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l815_81559


namespace NUMINAMATH_CALUDE_team_not_lose_prob_l815_81565

structure PlayerStats where
  cf_rate : ℝ
  winger_rate : ℝ
  am_rate : ℝ
  cf_lose_prob : ℝ
  winger_lose_prob : ℝ
  am_lose_prob : ℝ

def not_lose_prob (stats : PlayerStats) : ℝ :=
  stats.cf_rate * (1 - stats.cf_lose_prob) +
  stats.winger_rate * (1 - stats.winger_lose_prob) +
  stats.am_rate * (1 - stats.am_lose_prob)

theorem team_not_lose_prob (stats : PlayerStats)
  (h1 : stats.cf_rate = 0.2)
  (h2 : stats.winger_rate = 0.5)
  (h3 : stats.am_rate = 0.3)
  (h4 : stats.cf_lose_prob = 0.4)
  (h5 : stats.winger_lose_prob = 0.2)
  (h6 : stats.am_lose_prob = 0.2) :
  not_lose_prob stats = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_team_not_lose_prob_l815_81565


namespace NUMINAMATH_CALUDE_time_BC_is_five_hours_l815_81556

/-- Represents the train's journey between stations A, B, and C -/
structure TrainJourney where
  M : ℝ  -- Distance unit
  speed : ℝ  -- Constant speed of the train
  time_AB : ℝ  -- Time from A to B
  dist_AC : ℝ  -- Total distance from A to C

/-- The theorem stating the time taken from B to C -/
theorem time_BC_is_five_hours (journey : TrainJourney) 
  (h1 : journey.time_AB = 7)
  (h2 : journey.dist_AC = 6 * journey.M)
  : ∃ (time_BC : ℝ), time_BC = 5 := by
  sorry

end NUMINAMATH_CALUDE_time_BC_is_five_hours_l815_81556


namespace NUMINAMATH_CALUDE_probability_convex_quadrilateral_l815_81557

-- Define the number of points on the circle
def num_points : ℕ := 6

-- Define the number of chords to be selected
def num_chords : ℕ := 4

-- Define the total number of possible chords
def total_chords : ℕ := num_points.choose 2

-- Define the number of ways to select chords
def ways_to_select_chords : ℕ := total_chords.choose num_chords

-- Define the number of ways to form a convex quadrilateral
def convex_quadrilaterals : ℕ := num_points.choose 4

-- State the theorem
theorem probability_convex_quadrilateral :
  (convex_quadrilaterals : ℚ) / ways_to_select_chords = 1 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_convex_quadrilateral_l815_81557


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l815_81598

theorem floor_abs_negative_real : ⌊|(-57.6 : ℝ)|⌋ = 57 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l815_81598


namespace NUMINAMATH_CALUDE_square_difference_equals_two_l815_81593

theorem square_difference_equals_two (x y : ℝ) 
  (h1 : 1/x + 1/y = 2) 
  (h2 : x*y + x - y = 6) : 
  x^2 - y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_two_l815_81593


namespace NUMINAMATH_CALUDE_pet_calculation_l815_81555

theorem pet_calculation (taylor_pets : ℕ) (total_pets : ℕ) : 
  taylor_pets = 4 → 
  total_pets = 32 → 
  ∃ (other_friends_pets : ℕ),
    total_pets = taylor_pets + 3 * (2 * taylor_pets) + 2 * other_friends_pets ∧ 
    other_friends_pets = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_calculation_l815_81555


namespace NUMINAMATH_CALUDE_sara_earnings_l815_81536

/-- Sara's cake-making and selling scenario --/
def sara_cake_scenario (weekdays_per_week : ℕ) (cakes_per_day : ℕ) (price_per_cake : ℕ) (num_weeks : ℕ) : ℕ :=
  weekdays_per_week * cakes_per_day * price_per_cake * num_weeks

/-- Theorem: Sara's earnings over 4 weeks --/
theorem sara_earnings : sara_cake_scenario 5 4 8 4 = 640 := by
  sorry

end NUMINAMATH_CALUDE_sara_earnings_l815_81536


namespace NUMINAMATH_CALUDE_max_value_inequality_l815_81591

theorem max_value_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + 2*c = 2) :
  (a*b)/(a+b) + (a*c)/(a+c) + (b*c)/(b+c) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l815_81591


namespace NUMINAMATH_CALUDE_robot_center_not_necessarily_on_line_l815_81538

/-- Represents a circular robot -/
structure CircularRobot where
  center : ℝ × ℝ
  radius : ℝ
  deriving Inhabited

/-- Represents a movement of the robot -/
def RobotMovement := ℝ → CircularRobot

/-- A point remains on a line throughout the movement -/
def PointRemainsOnLine (p : ℝ × ℝ) (m : RobotMovement) : Prop :=
  ∃ (a b c : ℝ), ∀ t, a * (m t).center.1 + b * (m t).center.2 + c = 0

/-- The theorem statement -/
theorem robot_center_not_necessarily_on_line :
  ∃ (m : RobotMovement),
    (∀ θ : ℝ, PointRemainsOnLine ((m 0).center.1 + (m 0).radius * Real.cos θ,
                                  (m 0).center.2 + (m 0).radius * Real.sin θ) m) ∧
    ¬ PointRemainsOnLine (m 0).center m :=
  sorry


end NUMINAMATH_CALUDE_robot_center_not_necessarily_on_line_l815_81538


namespace NUMINAMATH_CALUDE_derivative_value_at_five_l815_81543

theorem derivative_value_at_five (f : ℝ → ℝ) (hf : ∀ x, f x = 3 * x^2 + 2 * x * (deriv f 2)) :
  deriv f 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_derivative_value_at_five_l815_81543


namespace NUMINAMATH_CALUDE_divisor_property_implies_prime_l815_81573

theorem divisor_property_implies_prime (n : ℕ) 
  (h1 : n > 1)
  (h2 : ∀ d : ℕ, d > 0 → d ∣ n → (d + 1) ∣ (n + 1)) : 
  Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_divisor_property_implies_prime_l815_81573


namespace NUMINAMATH_CALUDE_power_equation_solutions_l815_81506

theorem power_equation_solutions : 
  {(a, b, c) : ℕ × ℕ × ℕ | 2^a * 3^b = 7^c - 1} = {(1, 1, 1), (4, 1, 2)} := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solutions_l815_81506


namespace NUMINAMATH_CALUDE_bounce_count_is_seven_l815_81567

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The number of bounces a ball makes before returning to a vertex -/
def num_bounces (t : Triangle) (Y : ℝ × ℝ) : ℕ := sorry

/-- The theorem stating that the number of bounces is 7 for the given triangle and point -/
theorem bounce_count_is_seven :
  let t : Triangle := { A := (0, 0), B := (7, 0), C := (7/2, 3*Real.sqrt 3/2) }
  let Y : ℝ × ℝ := (7/2, 3*Real.sqrt 3/2)
  num_bounces t Y = 7 := by sorry

end NUMINAMATH_CALUDE_bounce_count_is_seven_l815_81567
