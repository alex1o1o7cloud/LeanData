import Mathlib

namespace NUMINAMATH_CALUDE_compound_interest_problem_l400_40039

def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r) ^ t

theorem compound_interest_problem : 
  ∃ (P : ℝ) (r : ℝ), 
    compound_interest P r 2 = 8880 ∧ 
    compound_interest P r 3 = 9261 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l400_40039


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l400_40071

theorem polynomial_division_theorem (x : ℝ) :
  ∃ R : ℝ, 5 * x^3 + 4 * x^2 - 6 * x - 9 = (x - 1) * (5 * x^2 + 9 * x + 3) + R :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l400_40071


namespace NUMINAMATH_CALUDE_absolute_difference_l400_40095

/-- Given a set of five numbers {m, n, 9, 8, 10} with an average of 9 and a variance of 2, |m - n| = 4 -/
theorem absolute_difference (m n : ℝ) 
  (h_avg : (m + n + 9 + 8 + 10) / 5 = 9)
  (h_var : ((m - 9)^2 + (n - 9)^2 + (9 - 9)^2 + (8 - 9)^2 + (10 - 9)^2) / 5 = 2) :
  |m - n| = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_l400_40095


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l400_40034

/-- The area of an isosceles right triangle with perimeter 2s -/
theorem isosceles_right_triangle_area (s : ℝ) : 
  ∃ (area : ℝ), 
    (∃ (a b c : ℝ), 
      a = b ∧                   -- Two sides are equal (isosceles)
      c^2 = a^2 + b^2 ∧         -- Right triangle (Pythagorean theorem)
      a + b + c = 2*s ∧         -- Perimeter is 2s
      area = (1/2) * a * b) ∧   -- Area formula
    area = (3 - 2*Real.sqrt 2) * s^2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l400_40034


namespace NUMINAMATH_CALUDE_trajectory_and_line_equations_l400_40029

-- Define the points
def A : ℝ × ℝ := (0, 3)
def O : ℝ × ℝ := (0, 0)
def N : ℝ × ℝ := (-1, 3)

-- Define the moving point M
def M : ℝ × ℝ → Prop := fun (x, y) ↦ 
  (x - A.1)^2 + (y - A.2)^2 = 4 * ((x - O.1)^2 + (y - O.2)^2)

-- Define the trajectory
def Trajectory : ℝ × ℝ → Prop := fun (x, y) ↦ 
  x^2 + (y + 1)^2 = 4

-- Define the line equations
def Line1 : ℝ × ℝ → Prop := fun (x, y) ↦ x = -1
def Line2 : ℝ × ℝ → Prop := fun (x, y) ↦ 15*x + 8*y - 9 = 0

theorem trajectory_and_line_equations :
  (∀ p, M p ↔ Trajectory p) ∧
  (∃ l, (l = Line1 ∨ l = Line2) ∧
        (l N) ∧
        (∃ p q : ℝ × ℝ, p ≠ q ∧ Trajectory p ∧ Trajectory q ∧ l p ∧ l q ∧
          (p.1 - q.1)^2 + (p.2 - q.2)^2 = 12)) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_line_equations_l400_40029


namespace NUMINAMATH_CALUDE_hyperbola_proof_l400_40044

/-- Given hyperbola -/
def given_hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

/-- Given ellipse -/
def given_ellipse (x y : ℝ) : Prop := y^2 / 8 + x^2 / 2 = 1

/-- Desired hyperbola -/
def desired_hyperbola (x y : ℝ) : Prop := y^2 / 2 - x^2 / 4 = 1

/-- The theorem to be proved -/
theorem hyperbola_proof :
  ∀ x y : ℝ,
  (∃ k : ℝ, k ≠ 0 ∧ given_hyperbola (k*x) (k*y)) ∧  -- Same asymptotes condition
  (∃ fx fy : ℝ, given_ellipse fx fy ∧ desired_hyperbola fx fy) →  -- Shared focus condition
  desired_hyperbola x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_proof_l400_40044


namespace NUMINAMATH_CALUDE_twentyFifth_is_221_l400_40072

/-- Converts a natural number to its base-3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The 25th number in base-3 counting sequence -/
def twentyFifthBase3 : List ℕ := toBase3 25

theorem twentyFifth_is_221 : twentyFifthBase3 = [2, 2, 1] := by
  sorry

#eval twentyFifthBase3

end NUMINAMATH_CALUDE_twentyFifth_is_221_l400_40072


namespace NUMINAMATH_CALUDE_largest_common_term_l400_40010

def isInFirstSequence (x : ℕ) : Prop := ∃ n : ℕ, x = 3 + 8 * n

def isInSecondSequence (x : ℕ) : Prop := ∃ m : ℕ, x = 5 + 9 * m

theorem largest_common_term : 
  (∃ x : ℕ, x ≤ 200 ∧ isInFirstSequence x ∧ isInSecondSequence x ∧ 
    ∀ y : ℕ, y ≤ 200 → isInFirstSequence y → isInSecondSequence y → y ≤ x) ∧
  (∃ x : ℕ, x = 131 ∧ x ≤ 200 ∧ isInFirstSequence x ∧ isInSecondSequence x) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l400_40010


namespace NUMINAMATH_CALUDE_special_rectangle_area_l400_40052

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  x : ℝ  -- Length
  y : ℝ  -- Width
  perimeter_eq : x + y = 30  -- Half perimeter equals 30
  side_diff : x = y + 3

/-- The area of a SpecialRectangle -/
def area (r : SpecialRectangle) : ℝ := r.x * r.y

/-- Theorem stating the area of the SpecialRectangle -/
theorem special_rectangle_area :
  ∀ r : SpecialRectangle, area r = 222.75 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_area_l400_40052


namespace NUMINAMATH_CALUDE_min_diff_is_one_l400_40049

-- Define the functions
def f (x : ℤ) : ℝ := 2 * (abs x)
def g (x : ℤ) : ℝ := -(x^2) - 4*x - 1

-- Define the difference function
def diff (x : ℤ) : ℝ := f x - g x

-- Theorem statement
theorem min_diff_is_one :
  ∃ (x : ℤ), diff x = 1 ∧ ∀ (y : ℤ), diff y ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_diff_is_one_l400_40049


namespace NUMINAMATH_CALUDE_inversion_number_reverse_l400_40050

/-- An array of 8 distinct integers -/
def Array8 := Fin 8 → ℤ

/-- The inversion number of an array -/
def inversionNumber (A : Array8) : ℕ :=
  sorry

/-- Theorem: Given an array of 8 distinct integers with inversion number 2,
    the inversion number of its reverse (excluding the last element) is at least 19 -/
theorem inversion_number_reverse (A : Array8) 
  (h_distinct : ∀ i j, i ≠ j → A i ≠ A j)
  (h_inv_num : inversionNumber A = 2) :
  inversionNumber (fun i => A (⟨7 - i.val, sorry⟩ : Fin 8)) ≥ 19 :=
sorry

end NUMINAMATH_CALUDE_inversion_number_reverse_l400_40050


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l400_40066

/-- A geometric sequence with first term 1 and third term 4 has a common ratio of ±2 -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℝ), 
  (∀ n : ℕ, a (n + 1) = a n * a 1) → -- Geometric sequence condition
  a 1 = 1 →
  a 3 = 4 →
  ∃ q : ℝ, a 1 * q^2 = a 3 ∧ q = 2 ∨ q = -2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l400_40066


namespace NUMINAMATH_CALUDE_unique_special_number_l400_40067

/-- A four-digit number with specific properties -/
def special_number : ℕ → Prop := λ n =>
  -- The number is four-digit
  1000 ≤ n ∧ n < 10000 ∧
  -- The unit digit is 2
  n % 10 = 2 ∧
  -- Moving the last digit to the front results in a number 108 less than the original
  (2000 + n / 10) = n - 108

theorem unique_special_number :
  ∃! n : ℕ, special_number n ∧ n = 2342 :=
sorry

end NUMINAMATH_CALUDE_unique_special_number_l400_40067


namespace NUMINAMATH_CALUDE_graph_shift_l400_40030

/-- Given a function f: ℝ → ℝ, prove that f(x - 2) + 1 is equivalent to
    shifting the graph of f(x) right by 2 units and up by 1 unit. -/
theorem graph_shift (f : ℝ → ℝ) (x : ℝ) :
  f (x - 2) + 1 = (fun y ↦ f (y - 2)) (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_graph_shift_l400_40030


namespace NUMINAMATH_CALUDE_max_profit_theorem_profit_range_theorem_l400_40036

/-- The daily sales volume as a function of the selling price -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 1000

/-- The daily profit as a function of the selling price -/
def profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The theorem stating the maximum profit and the price at which it occurs -/
theorem max_profit_theorem :
  ∃ (max_profit : ℝ) (optimal_price : ℝ),
    (∀ x : ℝ, 50 ≤ x ∧ x ≤ 65 → profit x ≤ max_profit) ∧
    profit optimal_price = max_profit ∧
    optimal_price = 65 ∧
    max_profit = 8750 :=
sorry

/-- The theorem stating the range of prices for which the profit is at least 8000 -/
theorem profit_range_theorem :
  ∀ x : ℝ, (60 ≤ x ∧ x ≤ 65) ↔ profit x ≥ 8000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_theorem_profit_range_theorem_l400_40036


namespace NUMINAMATH_CALUDE_equation_solution_l400_40091

theorem equation_solution : ∃! x : ℝ, 5 * 5^x + Real.sqrt (25 * 25^x) = 50 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l400_40091


namespace NUMINAMATH_CALUDE_red_shirt_pairs_l400_40001

theorem red_shirt_pairs 
  (total_students : ℕ) 
  (green_students : ℕ) 
  (red_students : ℕ) 
  (total_pairs : ℕ) 
  (green_green_pairs : ℕ) : 
  total_students = 132 →
  green_students = 64 →
  red_students = 68 →
  total_pairs = 66 →
  green_green_pairs = 28 →
  ∃ (red_red_pairs : ℕ), red_red_pairs = 30 :=
by sorry

end NUMINAMATH_CALUDE_red_shirt_pairs_l400_40001


namespace NUMINAMATH_CALUDE_inequality_proof_l400_40069

theorem inequality_proof (a b : ℝ) : a^2 + a*b + b^2 - 3*(a + b - 1) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l400_40069


namespace NUMINAMATH_CALUDE_machine_B_efficiency_l400_40017

/-- The number of sprockets produced by each machine -/
def total_sprockets : ℕ := 440

/-- The rate at which Machine A produces sprockets (sprockets per hour) -/
def rate_A : ℚ := 4

/-- The time difference between Machine A and Machine B to produce the total sprockets -/
def time_difference : ℕ := 10

/-- Calculates the percentage increase of rate B compared to rate A -/
def percentage_increase (rate_A rate_B : ℚ) : ℚ :=
  (rate_B - rate_A) / rate_A * 100

theorem machine_B_efficiency :
  let time_A := total_sprockets / rate_A
  let time_B := time_A - time_difference
  let rate_B := total_sprockets / time_B
  percentage_increase rate_A rate_B = 10 := by sorry

end NUMINAMATH_CALUDE_machine_B_efficiency_l400_40017


namespace NUMINAMATH_CALUDE_regular_polygon_nine_sides_l400_40057

-- Define a regular polygon
structure RegularPolygon where
  n : ℕ  -- number of sides
  a : ℝ  -- side length
  b : ℝ  -- longest diagonal
  c : ℝ  -- shortest diagonal
  h1 : n > 2  -- n must be greater than 2 for a polygon
  h2 : a > 0  -- side length must be positive
  h3 : b > c  -- longest diagonal is greater than shortest diagonal
  h4 : a = b - c  -- given condition

-- Theorem statement
theorem regular_polygon_nine_sides (p : RegularPolygon) : p.n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_nine_sides_l400_40057


namespace NUMINAMATH_CALUDE_schedule_five_courses_nine_periods_l400_40045

/-- The number of ways to schedule courses -/
def schedule_ways (n_courses n_periods : ℕ) : ℕ :=
  Nat.choose n_periods n_courses * Nat.factorial n_courses

/-- Theorem stating the number of ways to schedule 5 courses in 9 periods -/
theorem schedule_five_courses_nine_periods :
  schedule_ways 5 9 = 15120 := by
  sorry

end NUMINAMATH_CALUDE_schedule_five_courses_nine_periods_l400_40045


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l400_40019

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 4) :
  x^2 + 1/x^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l400_40019


namespace NUMINAMATH_CALUDE_man_in_dark_probability_l400_40025

/-- The number of revolutions per minute made by the searchlight -/
def revolutions_per_minute : ℝ := 3

/-- The probability that a man will stay in the dark for at least some seconds -/
def probability_in_dark : ℝ := 0.25

/-- Theorem stating the probability of a man staying in the dark -/
theorem man_in_dark_probability :
  probability_in_dark = 0.25 := by sorry

end NUMINAMATH_CALUDE_man_in_dark_probability_l400_40025


namespace NUMINAMATH_CALUDE_eugene_model_house_l400_40021

/-- The number of toothpicks Eugene uses per card -/
def toothpicks_per_card : ℕ := 75

/-- The total number of cards in the deck -/
def total_cards : ℕ := 52

/-- The number of cards Eugene did not use -/
def unused_cards : ℕ := 16

/-- The number of toothpicks in one box -/
def toothpicks_per_box : ℕ := 450

/-- The number of boxes of toothpicks Eugene used -/
def boxes_used : ℕ := 6

theorem eugene_model_house :
  boxes_used = (total_cards - unused_cards) * toothpicks_per_card / toothpicks_per_box :=
by sorry

end NUMINAMATH_CALUDE_eugene_model_house_l400_40021


namespace NUMINAMATH_CALUDE_arithmetic_sum_specific_l400_40037

/-- Sum of arithmetic sequence with given parameters -/
def arithmetic_sum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℤ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic sequence with first term -45, last term 0, and common difference 3 is -360 -/
theorem arithmetic_sum_specific : arithmetic_sum (-45) 0 3 = -360 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_specific_l400_40037


namespace NUMINAMATH_CALUDE_triangle_base_length_l400_40028

theorem triangle_base_length (area height : ℝ) (h1 : area = 16) (h2 : height = 4) :
  (2 * area) / height = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l400_40028


namespace NUMINAMATH_CALUDE_nissa_cat_grooming_time_l400_40016

/-- Represents the time in seconds for various cat grooming activities -/
structure CatGroomingTime where
  clip_claw : ℕ
  clean_ear : ℕ
  shampoo : ℕ
  brush_fur : ℕ
  give_treat : ℕ
  trim_fur : ℕ

/-- Calculates the total grooming time for a cat -/
def total_grooming_time (t : CatGroomingTime) : ℕ :=
  t.clip_claw * 16 + t.clean_ear * 2 + t.shampoo + t.brush_fur + t.give_treat + t.trim_fur

/-- Theorem stating that the total grooming time for Nissa's cat is 970 seconds -/
theorem nissa_cat_grooming_time :
  ∃ (t : CatGroomingTime),
    t.clip_claw = 10 ∧
    t.clean_ear = 90 ∧
    t.shampoo = 300 ∧
    t.brush_fur = 120 ∧
    t.give_treat = 30 ∧
    t.trim_fur = 180 ∧
    total_grooming_time t = 970 :=
by
  sorry


end NUMINAMATH_CALUDE_nissa_cat_grooming_time_l400_40016


namespace NUMINAMATH_CALUDE_no_such_function_l400_40093

theorem no_such_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1987 := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_l400_40093


namespace NUMINAMATH_CALUDE_sports_books_count_l400_40094

/-- Given the total number of books and the number of school books,
    prove that the number of sports books is 39. -/
theorem sports_books_count (total_books school_books : ℕ)
    (h1 : total_books = 58)
    (h2 : school_books = 19) :
    total_books - school_books = 39 := by
  sorry

end NUMINAMATH_CALUDE_sports_books_count_l400_40094


namespace NUMINAMATH_CALUDE_carpet_breadth_calculation_l400_40051

theorem carpet_breadth_calculation (b : ℝ) : 
  let first_length := 1.44 * b
  let second_length := 1.4 * first_length
  let second_breadth := 1.25 * b
  let second_area := second_length * second_breadth
  let cost_per_sqm := 45
  let total_cost := 4082.4
  second_area = total_cost / cost_per_sqm →
  b = 6.08 := by
sorry

end NUMINAMATH_CALUDE_carpet_breadth_calculation_l400_40051


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l400_40047

/-- Given two rectangles of equal area, where one rectangle measures 4.5 inches by 19.25 inches
    and the other rectangle has a length of 3.75 inches, prove that the width of the second
    rectangle is 23.1 inches. -/
theorem equal_area_rectangles_width (carol_length carol_width jordan_length : ℝ)
    (h1 : carol_length = 4.5)
    (h2 : carol_width = 19.25)
    (h3 : jordan_length = 3.75)
    (h4 : carol_length * carol_width = jordan_length * (carol_length * carol_width / jordan_length)) :
  carol_length * carol_width / jordan_length = 23.1 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l400_40047


namespace NUMINAMATH_CALUDE_batsman_average_increase_l400_40073

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the average runs per innings -/
def calculateAverage (totalRuns : ℕ) (innings : ℕ) : ℚ :=
  (totalRuns : ℚ) / (innings : ℚ)

theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 11 →
    let newTotalRuns := b.totalRuns + 55
    let newInnings := b.innings + 1
    let newAverage := calculateAverage newTotalRuns newInnings
    newAverage = 44 →
    newAverage - b.average = 1 := by
  sorry

#check batsman_average_increase

end NUMINAMATH_CALUDE_batsman_average_increase_l400_40073


namespace NUMINAMATH_CALUDE_max_difference_is_five_point_five_l400_40014

/-- A structure representing a set of segments on the ray (0, +∞) -/
structure SegmentSet where
  /-- The left end of the leftmost segment -/
  a : ℝ
  /-- The right end of the rightmost segment -/
  b : ℝ
  /-- The number of segments (more than two) -/
  n : ℕ
  /-- n > 2 -/
  h_n : n > 2
  /-- a > 0 -/
  h_a : a > 0
  /-- b > a -/
  h_b : b > a
  /-- For any two different segments, there exist numbers that differ by a factor of 2 -/
  factor_of_two : ∀ i j, i ≠ j → i < n → j < n → ∃ x y, x ∈ Set.Icc (a + i) (a + i + 1) ∧ y ∈ Set.Icc (a + j) (a + j + 1) ∧ (x = 2 * y ∨ y = 2 * x)

/-- The theorem stating that the maximum value of b - a is 5.5 -/
theorem max_difference_is_five_point_five (s : SegmentSet) : 
  (∃ (s' : SegmentSet), s'.b - s'.a ≥ s.b - s.a) → s.b - s.a ≤ 5.5 := by
  sorry

end NUMINAMATH_CALUDE_max_difference_is_five_point_five_l400_40014


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l400_40013

theorem sum_of_roots_squared_equation (x : ℝ) :
  (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a + b = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l400_40013


namespace NUMINAMATH_CALUDE_students_representing_x_percent_of_boys_l400_40058

theorem students_representing_x_percent_of_boys 
  (total_population : ℝ) 
  (boys_percentage : ℝ) 
  (x : ℝ) 
  (h1 : total_population = 113.38934190276818)
  (h2 : boys_percentage = 70) :
  (x / 100) * (boys_percentage / 100 * total_population) = 
  (x / 100) * 79.37253933173772 :=
by
  sorry

end NUMINAMATH_CALUDE_students_representing_x_percent_of_boys_l400_40058


namespace NUMINAMATH_CALUDE_equilateral_triangle_cd_l400_40048

/-- An equilateral triangle with vertices at (0,0), (c,14), and (d,41) has cd = -2208 -/
theorem equilateral_triangle_cd (c d : ℝ) : 
  (Complex.abs (Complex.I * 14 - c - Complex.I * 14) = Complex.abs (Complex.I * 41 - c - Complex.I * 14)) ∧
  (Complex.abs (Complex.I * 41 - 0) = Complex.abs (c + Complex.I * 14 - 0)) ∧
  (Complex.abs (c + Complex.I * 14 - 0) = Complex.abs (Complex.I * 14 - 0)) →
  c * d = -2208 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_cd_l400_40048


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l400_40020

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a2 : a 2 = 1)
  (h_a8 : a 8 = a 6 + 2 * a 4) :
  a 6 = 4 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l400_40020


namespace NUMINAMATH_CALUDE_master_bath_size_l400_40089

theorem master_bath_size (bedroom_size new_room_size : ℝ) 
  (h1 : bedroom_size = 309)
  (h2 : new_room_size = 918)
  (h3 : new_room_size = 2 * (bedroom_size + bathroom_size)) :
  bathroom_size = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_master_bath_size_l400_40089


namespace NUMINAMATH_CALUDE_M_greater_than_N_l400_40056

theorem M_greater_than_N : ∀ x : ℝ, (x - 3) * (x - 7) > (x - 2) * (x - 8) := by
  sorry

end NUMINAMATH_CALUDE_M_greater_than_N_l400_40056


namespace NUMINAMATH_CALUDE_circplus_assoc_l400_40031

/-- The custom operation ⊕ on real numbers -/
def circplus (x y : ℝ) : ℝ := x + y - x * y

/-- Theorem stating that the ⊕ operation is associative -/
theorem circplus_assoc :
  ∀ (x y z : ℝ), circplus (circplus x y) z = circplus x (circplus y z) := by
  sorry

end NUMINAMATH_CALUDE_circplus_assoc_l400_40031


namespace NUMINAMATH_CALUDE_quadratic_minimum_l400_40022

theorem quadratic_minimum (x : ℝ) : x^2 + 6*x ≥ -9 ∧ ∃ y : ℝ, y^2 + 6*y = -9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l400_40022


namespace NUMINAMATH_CALUDE_tablecloth_extension_theorem_l400_40099

/-- Represents a circular table with a square tablecloth placed on it. -/
structure TableWithCloth where
  /-- Diameter of the circular table in meters -/
  table_diameter : ℝ
  /-- Side length of the square tablecloth in meters -/
  cloth_side_length : ℝ
  /-- Extension of one corner beyond the table edge in meters -/
  corner1_extension : ℝ
  /-- Extension of an adjacent corner beyond the table edge in meters -/
  corner2_extension : ℝ

/-- Calculates the extensions of the remaining two corners of the tablecloth. -/
def calculate_remaining_extensions (t : TableWithCloth) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct extensions for the given table and tablecloth configuration. -/
theorem tablecloth_extension_theorem (t : TableWithCloth) 
  (h1 : t.table_diameter = 0.6)
  (h2 : t.cloth_side_length = 1)
  (h3 : t.corner1_extension = 0.5)
  (h4 : t.corner2_extension = 0.3) :
  calculate_remaining_extensions t = (0.33, 0.52) :=
by sorry

end NUMINAMATH_CALUDE_tablecloth_extension_theorem_l400_40099


namespace NUMINAMATH_CALUDE_matthew_crackers_l400_40015

theorem matthew_crackers (friends : ℕ) (cakes : ℕ) (eaten_crackers : ℕ) :
  friends = 4 →
  cakes = 98 →
  eaten_crackers = 8 →
  ∃ (initial_crackers : ℕ),
    initial_crackers = 128 ∧
    ∃ (given_per_friend : ℕ),
      given_per_friend * friends ≤ cakes ∧
      given_per_friend * friends ≤ initial_crackers ∧
      initial_crackers = given_per_friend * friends + eaten_crackers * friends :=
by
  sorry

end NUMINAMATH_CALUDE_matthew_crackers_l400_40015


namespace NUMINAMATH_CALUDE_problem_statement_l400_40059

theorem problem_statement (x y : ℚ) 
  (h1 : 3 * x + 4 * y = 0)
  (h2 : x = y + 3) :
  5 * y = -45 / 7 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l400_40059


namespace NUMINAMATH_CALUDE_total_puppies_eq_sum_l400_40098

/-- The number of puppies Alyssa's dog had -/
def total_puppies : ℕ := 23

/-- The number of puppies Alyssa gave to her friends -/
def puppies_given_away : ℕ := 15

/-- The number of puppies Alyssa kept for herself -/
def puppies_kept : ℕ := 8

/-- Theorem stating that the total number of puppies is the sum of puppies given away and kept -/
theorem total_puppies_eq_sum : total_puppies = puppies_given_away + puppies_kept := by
  sorry

end NUMINAMATH_CALUDE_total_puppies_eq_sum_l400_40098


namespace NUMINAMATH_CALUDE_pentagon_area_l400_40038

/-- The area of a specific pentagon -/
theorem pentagon_area : 
  ∀ (s₁ s₂ s₃ s₄ s₅ : ℝ) (θ : ℝ),
  s₁ = 18 → s₂ = 20 → s₃ = 27 → s₄ = 24 → s₅ = 20 →
  θ = Real.pi / 2 →
  ∃ (A : ℝ),
  A = (1/2 * s₁ * s₂) + (1/2 * (s₃ + s₄) * s₅) ∧
  A = 690 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_area_l400_40038


namespace NUMINAMATH_CALUDE_map_scale_l400_40007

/-- Given a map where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale (scale : ℝ → ℝ) (h1 : scale 15 = 90) : scale 20 = 120 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l400_40007


namespace NUMINAMATH_CALUDE_calculation_problems_l400_40075

theorem calculation_problems :
  (∀ a : ℝ, a^3 * a + (-a^2)^3 / a^2 = 0) ∧
  (Real.sqrt 5 - Real.sqrt 2) * (Real.sqrt 5 + Real.sqrt 2) + (Real.sqrt 3 - 1)^2 = 7 - 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_calculation_problems_l400_40075


namespace NUMINAMATH_CALUDE_prob_spade_first_ace_last_value_l400_40006

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of aces in a standard deck -/
def NumAces : ℕ := 4

/-- Probability of drawing three cards from a standard 52-card deck,
    where the first card is a spade and the last card is an ace -/
def prob_spade_first_ace_last : ℚ :=
  (NumSpades * NumAces + NumAces - 1) / (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2))

theorem prob_spade_first_ace_last_value :
  prob_spade_first_ace_last = 51 / 2600 := by
  sorry

end NUMINAMATH_CALUDE_prob_spade_first_ace_last_value_l400_40006


namespace NUMINAMATH_CALUDE_apple_preference_percentage_l400_40065

-- Define the fruit categories
inductive Fruit
| Apple
| Banana
| Cherry
| Orange
| Pear

-- Define the function that gives the frequency for each fruit
def frequency (f : Fruit) : ℕ :=
  match f with
  | Fruit.Apple => 75
  | Fruit.Banana => 80
  | Fruit.Cherry => 45
  | Fruit.Orange => 100
  | Fruit.Pear => 50

-- Define the total number of responses
def total_responses : ℕ := 
  frequency Fruit.Apple + frequency Fruit.Banana + frequency Fruit.Cherry + 
  frequency Fruit.Orange + frequency Fruit.Pear

-- Theorem: The percentage of people who preferred apples is 21%
theorem apple_preference_percentage : 
  (frequency Fruit.Apple : ℚ) / (total_responses : ℚ) * 100 = 21 := by
  sorry

end NUMINAMATH_CALUDE_apple_preference_percentage_l400_40065


namespace NUMINAMATH_CALUDE_trajectory_equation_l400_40000

/-- The equation of the trajectory of the center of a circle that passes through point A(2,0) 
    and is internally tangent to the circle x^2 + 4x + y^2 - 32 = 0 is x^2/9 + y^2/5 = 1 -/
theorem trajectory_equation : 
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), (x - 2)^2 + y^2 = r^2) ∧ 
  (∃ (t : ℝ), (x - (-2))^2 + y^2 = (6 + t)^2 ∧ x^2 + 4*x + y^2 - 32 = 0) →
  x^2/9 + y^2/5 = 1 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l400_40000


namespace NUMINAMATH_CALUDE_expression_evaluation_l400_40088

theorem expression_evaluation :
  (3^102 + 7^103)^2 - (3^102 - 7^103)^2 = 240 * 10^206 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l400_40088


namespace NUMINAMATH_CALUDE_parabola_vertex_l400_40018

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = (x - 2)^2 + 5

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 5)

/-- Theorem: The vertex of the parabola y = (x-2)^2 + 5 is at the point (2,5) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l400_40018


namespace NUMINAMATH_CALUDE_james_total_toys_l400_40082

/-- The minimum number of toy cars needed to get a discount -/
def discount_threshold : ℕ := 25

/-- The initial number of toy cars James buys -/
def initial_cars : ℕ := 20

/-- The ratio of toy soldiers to toy cars -/
def soldier_to_car_ratio : ℕ := 2

/-- The total number of toys James buys to maximize his discount -/
def total_toys : ℕ := 78

/-- Theorem stating that the total number of toys James buys is 78 -/
theorem james_total_toys :
  let additional_cars := discount_threshold + 1 - initial_cars
  let total_cars := initial_cars + additional_cars
  let total_soldiers := soldier_to_car_ratio * total_cars
  total_cars + total_soldiers = total_toys :=
by sorry

end NUMINAMATH_CALUDE_james_total_toys_l400_40082


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l400_40032

/-- Given two lines L1 and L2 in a 2D plane, where:
    - L1 has equation mx - m²y = 1
    - L2 is perpendicular to L1
    - L1 and L2 intersect at point P(2,1)
    Prove that the equation of L2 is x + y - 3 = 0 -/
theorem perpendicular_line_equation (m : ℝ) :
  (∀ x y, m * x - m^2 * y = 1 → x = 2 ∧ y = 1) →
  (∃ k : ℝ, k * m = -1) →
  ∀ x y, x + y - 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l400_40032


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l400_40079

theorem solution_implies_a_value (a : ℝ) : 
  (3 * a + 4 = 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l400_40079


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_91_l400_40070

theorem no_primes_divisible_by_91 :
  ¬∃ p : ℕ, Nat.Prime p ∧ 91 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_91_l400_40070


namespace NUMINAMATH_CALUDE_functional_equation_solution_l400_40008

theorem functional_equation_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x - f y) = f (f y) + x * f x + x^2) ↔ 
  (∀ x : ℝ, f x = 1 - x^2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l400_40008


namespace NUMINAMATH_CALUDE_average_score_theorem_l400_40040

/-- The average score of a class on a test with given score distribution -/
theorem average_score_theorem (num_questions : ℕ) (num_students : ℕ) 
  (prop_score_3 : ℚ) (prop_score_2 : ℚ) (prop_score_1 : ℚ) (prop_score_0 : ℚ) :
  num_questions = 3 →
  num_students = 50 →
  prop_score_3 = 30 / 100 →
  prop_score_2 = 50 / 100 →
  prop_score_1 = 10 / 100 →
  prop_score_0 = 10 / 100 →
  prop_score_3 + prop_score_2 + prop_score_1 + prop_score_0 = 1 →
  3 * prop_score_3 + 2 * prop_score_2 + 1 * prop_score_1 + 0 * prop_score_0 = 2 := by
  sorry

#check average_score_theorem

end NUMINAMATH_CALUDE_average_score_theorem_l400_40040


namespace NUMINAMATH_CALUDE_costume_cost_is_660_l400_40005

/-- Represents the cost of materials for Jenna's costume --/
def costume_cost : ℝ :=
  let velvet_price := 3
  let silk_price := 6
  let lace_price := 10
  let satin_price := 4
  let leather_price := 5
  let wool_price := 8
  let ribbon_price := 2

  let skirt_area := 12 * 4
  let skirts_count := 3
  let bodice_silk_area := 2
  let bodice_lace_area := 5 * 2
  let bonnet_area := 2.5 * 1.5
  let shoe_cover_area := 1 * 1.5 * 2
  let cape_area := 5 * 2
  let ribbon_length := 3

  let velvet_cost := velvet_price * skirt_area * skirts_count
  let bodice_cost := silk_price * bodice_silk_area + lace_price * bodice_lace_area
  let bonnet_cost := satin_price * bonnet_area
  let shoe_covers_cost := leather_price * shoe_cover_area
  let cape_cost := wool_price * cape_area
  let ribbon_cost := ribbon_price * ribbon_length

  velvet_cost + bodice_cost + bonnet_cost + shoe_covers_cost + cape_cost + ribbon_cost

/-- Theorem stating that the total cost of Jenna's costume materials is $660 --/
theorem costume_cost_is_660 : costume_cost = 660 := by
  sorry

end NUMINAMATH_CALUDE_costume_cost_is_660_l400_40005


namespace NUMINAMATH_CALUDE_harolds_car_payment_l400_40084

def monthly_income : ℚ := 2500
def rent : ℚ := 700
def groceries : ℚ := 50
def remaining_money : ℚ := 650

def car_payment (x : ℚ) : Prop :=
  let utilities := x / 2
  let total_expenses := rent + x + utilities + groceries
  let retirement_contribution := (monthly_income - total_expenses) / 2
  monthly_income - total_expenses - retirement_contribution = remaining_money

theorem harolds_car_payment :
  ∃ (x : ℚ), car_payment x ∧ x = 300 :=
sorry

end NUMINAMATH_CALUDE_harolds_car_payment_l400_40084


namespace NUMINAMATH_CALUDE_f_upper_bound_implies_a_bound_l400_40063

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - x) * Real.exp x + a * (x - 1)^2

theorem f_upper_bound_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, f a x ≤ 2 * Real.exp x) →
  a ≤ ((1 - Real.sqrt 2) * Real.exp (1 - Real.sqrt 2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_f_upper_bound_implies_a_bound_l400_40063


namespace NUMINAMATH_CALUDE_expression_simplification_l400_40004

theorem expression_simplification :
  ((3 + 5 + 7 + 9) / 3) - ((4 * 6 + 13) / 5) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l400_40004


namespace NUMINAMATH_CALUDE_propositions_3_and_4_are_true_l400_40077

theorem propositions_3_and_4_are_true :
  (∃ n : ℝ, ∀ m : ℝ, m * n = m) ∧
  (∀ n : ℝ, ∃ m : ℝ, m^2 < n) :=
by sorry

end NUMINAMATH_CALUDE_propositions_3_and_4_are_true_l400_40077


namespace NUMINAMATH_CALUDE_brock_cookies_proof_l400_40002

/-- Represents the number of cookies Brock bought -/
def brock_cookies : ℕ := 7

theorem brock_cookies_proof (total_cookies : ℕ) (stone_cookies : ℕ) (remaining_cookies : ℕ) 
  (h1 : total_cookies = 5 * 12)
  (h2 : stone_cookies = 2 * 12)
  (h3 : remaining_cookies = 15)
  (h4 : total_cookies = stone_cookies + 3 * brock_cookies + remaining_cookies) :
  brock_cookies = 7 := by
  sorry

end NUMINAMATH_CALUDE_brock_cookies_proof_l400_40002


namespace NUMINAMATH_CALUDE_fraction_of_three_fourths_that_is_one_fifth_l400_40043

theorem fraction_of_three_fourths_that_is_one_fifth (x : ℚ) : x * (3/4 : ℚ) = (1/5 : ℚ) ↔ x = (4/15 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_three_fourths_that_is_one_fifth_l400_40043


namespace NUMINAMATH_CALUDE_correct_operation_l400_40023

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l400_40023


namespace NUMINAMATH_CALUDE_max_value_3xy_plus_yz_l400_40012

theorem max_value_3xy_plus_yz (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  3*x*y + y*z ≤ Real.sqrt 10 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_3xy_plus_yz_l400_40012


namespace NUMINAMATH_CALUDE_quiz_answer_key_combinations_l400_40074

def num_true_false_questions : ℕ := 10
def num_multiple_choice_questions : ℕ := 6
def num_multiple_choice_options : ℕ := 6

theorem quiz_answer_key_combinations : 
  (Nat.choose num_true_false_questions (num_true_false_questions / 2)) * 
  (Nat.factorial num_multiple_choice_questions) = 181440 := by
  sorry

end NUMINAMATH_CALUDE_quiz_answer_key_combinations_l400_40074


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_l400_40003

-- Define the original proposition
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 3*x + m = 0

-- Define the contrapositive
def contrapositive (P Q : Prop) : Prop := ¬Q → ¬P

-- Theorem statement
theorem contrapositive_real_roots :
  contrapositive (m < 0) (has_real_roots m) ↔ (¬(has_real_roots m) → m ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_l400_40003


namespace NUMINAMATH_CALUDE_algebra_test_average_l400_40042

theorem algebra_test_average (male_count : ℕ) (female_count : ℕ) 
  (male_avg : ℝ) (female_avg : ℝ) :
  male_count = 8 →
  female_count = 12 →
  male_avg = 87 →
  female_avg = 92 →
  let total_count := male_count + female_count
  let total_sum := male_count * male_avg + female_count * female_avg
  total_sum / total_count = 90 := by
sorry

end NUMINAMATH_CALUDE_algebra_test_average_l400_40042


namespace NUMINAMATH_CALUDE_theater_attendance_l400_40041

theorem theater_attendance 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (total_attendance : ℕ) 
  (total_revenue : ℚ) 
  (h1 : adult_price = 60 / 100)
  (h2 : child_price = 25 / 100)
  (h3 : total_attendance = 280)
  (h4 : total_revenue = 140) :
  ∃ (adults children : ℕ),
    adults + children = total_attendance ∧
    adult_price * adults + child_price * children = total_revenue ∧
    children = 80 := by
sorry

end NUMINAMATH_CALUDE_theater_attendance_l400_40041


namespace NUMINAMATH_CALUDE_slope_movement_l400_40061

theorem slope_movement (hypotenuse : ℝ) (ratio : ℝ) : 
  hypotenuse = 100 * Real.sqrt 5 →
  ratio = 1 / 2 →
  ∃ (x : ℝ), x^2 + (ratio * x)^2 = hypotenuse^2 ∧ x = 100 :=
by sorry

end NUMINAMATH_CALUDE_slope_movement_l400_40061


namespace NUMINAMATH_CALUDE_infinite_square_free_sequences_l400_40062

def x_seq (a b n : ℕ) : ℕ := a * n + b
def y_seq (c d n : ℕ) : ℕ := c * n + d

def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p * p) ∣ n → False

theorem infinite_square_free_sequences
  (a b c d : ℕ) 
  (h1 : Nat.gcd a b = 1) 
  (h2 : Nat.gcd c d = 1) :
  ∃ S : Set ℕ, Set.Infinite S ∧ 
    ∀ n ∈ S, is_square_free (x_seq a b n) ∧ is_square_free (y_seq c d n) := by
  sorry

end NUMINAMATH_CALUDE_infinite_square_free_sequences_l400_40062


namespace NUMINAMATH_CALUDE_expression_evaluation_l400_40085

theorem expression_evaluation (a b : ℝ) (h : a^2 + b^2 - 2*a + 4*b = -5) :
  (a - 2*b)*(a^2 + 2*a*b + 4*b^2) - a*(a - 5*b)*(a + 3*b) = 120 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l400_40085


namespace NUMINAMATH_CALUDE_aardvark_path_distance_l400_40046

/-- The total distance an aardvark runs along a specific path between two concentric circles -/
theorem aardvark_path_distance (r₁ r₂ : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 30) :
  let path_length := π * r₂ + 2 * (r₂ - r₁) + π * r₁
  path_length = 45 * π + 30 :=
by sorry

end NUMINAMATH_CALUDE_aardvark_path_distance_l400_40046


namespace NUMINAMATH_CALUDE_car_profit_percentage_l400_40054

/-- Given a car with an original price, calculate the profit percentage when bought at a discount and sold at an increase. -/
theorem car_profit_percentage (P : ℝ) (discount : ℝ) (increase : ℝ)
  (h_discount : discount = 0.4)
  (h_increase : increase = 0.8) :
  let buying_price := P * (1 - discount)
  let selling_price := buying_price * (1 + increase)
  let profit := selling_price - P
  profit / P * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l400_40054


namespace NUMINAMATH_CALUDE_function_always_positive_l400_40081

theorem function_always_positive (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x + (x - 1) * (deriv f x) > 0) : 
  ∀ x, f x > 0 := by
sorry

end NUMINAMATH_CALUDE_function_always_positive_l400_40081


namespace NUMINAMATH_CALUDE_area_of_three_arc_region_sum_of_coefficients_l400_40083

/-- The area of a region bounded by three circular arcs --/
theorem area_of_three_arc_region :
  let r : ℝ := 6  -- radius of each circle
  let θ : ℝ := 90  -- central angle in degrees
  let area_sector : ℝ := (θ / 360) * π * r^2
  let area_triangle : ℝ := (1 / 2) * r^2
  let area_segment : ℝ := area_sector - area_triangle
  let total_area : ℝ := 3 * area_segment
  total_area = 27 * π - 54 :=
by
  sorry

/-- The sum of a, b, and c in the expression a√b + cπ --/
theorem sum_of_coefficients :
  let a : ℝ := 0
  let b : ℝ := 1
  let c : ℝ := 27
  a + b + c = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_area_of_three_arc_region_sum_of_coefficients_l400_40083


namespace NUMINAMATH_CALUDE_lucas_initial_money_l400_40076

/-- Proves that Lucas' initial amount of money is $20 given the problem conditions --/
theorem lucas_initial_money :
  ∀ (initial_money : ℕ) 
    (avocado_count : ℕ) 
    (avocado_price : ℕ) 
    (change : ℕ),
  avocado_count = 3 →
  avocado_price = 2 →
  change = 14 →
  initial_money = avocado_count * avocado_price + change →
  initial_money = 20 := by
sorry

end NUMINAMATH_CALUDE_lucas_initial_money_l400_40076


namespace NUMINAMATH_CALUDE_expected_steps_l400_40080

/-- Represents a point on the coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- The probability of moving in any direction -/
def moveProbability : ℚ := 1/4

/-- Roger's starting point -/
def startPoint : Point := ⟨0, 0⟩

/-- Function to determine if a point can be reached more quickly by a different route -/
def canReachQuicker (path : List Point) : Bool :=
  sorry

/-- The expected number of additional steps after the initial step -/
def e₁ : ℚ := 8/3

/-- The expected number of additional steps after moving perpendicular -/
def e₂ : ℚ := 2

/-- The main theorem: The expected number of steps Roger takes before he stops is 11/3 -/
theorem expected_steps :
  let totalSteps := 1 + e₁
  totalSteps = 11/3 := by sorry

end NUMINAMATH_CALUDE_expected_steps_l400_40080


namespace NUMINAMATH_CALUDE_equation_solution_l400_40033

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (27 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 :=
by
  use 9
  sorry

end NUMINAMATH_CALUDE_equation_solution_l400_40033


namespace NUMINAMATH_CALUDE_compute_expression_l400_40086

theorem compute_expression : 3 * 3^3 - 9^50 / 9^48 = 0 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l400_40086


namespace NUMINAMATH_CALUDE_infinite_pairs_l400_40078

/-- The set of prime divisors of a natural number -/
def primeDivisors (n : ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ p ∣ n}

/-- The condition for part (a) -/
def conditionA (a b : ℕ) : Prop :=
  primeDivisors a = primeDivisors b ∧
  primeDivisors (a + 1) = primeDivisors (b + 1) ∧
  a ≠ b

/-- The condition for part (b) -/
def conditionB (a b : ℕ) : Prop :=
  primeDivisors a = primeDivisors (b + 1) ∧
  primeDivisors (a + 1) = primeDivisors b

/-- The main theorem -/
theorem infinite_pairs :
  (∃ f : ℕ → ℕ × ℕ, Function.Injective f ∧ (∀ n, conditionA (f n).1 (f n).2)) ∧
  (∃ g : ℕ → ℕ × ℕ, Function.Injective g ∧ (∀ n, conditionB (g n).1 (g n).2)) :=
sorry

end NUMINAMATH_CALUDE_infinite_pairs_l400_40078


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l400_40068

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i^2 = -1 →
  Complex.im (2 * i^3 / (i - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l400_40068


namespace NUMINAMATH_CALUDE_inequality_proof_l400_40092

theorem inequality_proof (a b c : ℝ) : 
  a = Real.log 5 - Real.log 3 →
  b = (2 / 5) * Real.exp (2 / 3) →
  c = 2 / 3 →
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l400_40092


namespace NUMINAMATH_CALUDE_percentage_failed_both_subjects_l400_40096

theorem percentage_failed_both_subjects
  (failed_hindi : Real)
  (failed_english : Real)
  (passed_both : Real)
  (h1 : failed_hindi = 30)
  (h2 : failed_english = 42)
  (h3 : passed_both = 56) :
  100 - passed_both = failed_hindi + failed_english - (failed_hindi + failed_english - (100 - passed_both)) :=
by sorry

end NUMINAMATH_CALUDE_percentage_failed_both_subjects_l400_40096


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l400_40053

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 25 / 100 →
  germination_rate2 = 35 / 100 →
  (((seeds_plot1 : ℚ) * germination_rate1 + (seeds_plot2 : ℚ) * germination_rate2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) * 100 = 29 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l400_40053


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l400_40060

/-- The volume of a tetrahedron given its edge lengths -/
def tetrahedron_volume (ab ac ad cd bd bc : ℝ) : ℝ :=
  -- Definition to be filled
  sorry

/-- Theorem: The volume of the specific tetrahedron is 48 cubic units -/
theorem specific_tetrahedron_volume :
  tetrahedron_volume 6 7 8 9 10 11 = 48 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l400_40060


namespace NUMINAMATH_CALUDE_evaluate_expression_l400_40035

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 = 286 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l400_40035


namespace NUMINAMATH_CALUDE_sqrt_calculation_problems_l400_40011

theorem sqrt_calculation_problems :
  (∃ (x : ℝ), x = Real.sqrt 18 - Real.sqrt 8 - Real.sqrt 2 ∧ x = 0) ∧
  (∃ (y : ℝ), y = 6 * Real.sqrt 2 * Real.sqrt 3 + 3 * Real.sqrt 30 / Real.sqrt 5 ∧ y = 9 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_problems_l400_40011


namespace NUMINAMATH_CALUDE_pattern_equality_l400_40090

theorem pattern_equality (n : ℕ+) : (2*n + 2)^2 - 4*n^2 = 8*n + 4 := by
  sorry

end NUMINAMATH_CALUDE_pattern_equality_l400_40090


namespace NUMINAMATH_CALUDE_additive_inverse_of_zero_l400_40009

theorem additive_inverse_of_zero : (0 : ℤ) + (0 : ℤ) = (0 : ℤ) := by sorry

end NUMINAMATH_CALUDE_additive_inverse_of_zero_l400_40009


namespace NUMINAMATH_CALUDE_max_rope_length_l400_40097

theorem max_rope_length (a b c d : ℕ) 
  (ha : a = 48) (hb : b = 72) (hc : c = 108) (hd : d = 120) : 
  Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_rope_length_l400_40097


namespace NUMINAMATH_CALUDE_f_composition_value_l400_40024

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) else Real.tan x

theorem f_composition_value : f (f (3 * Real.pi / 4)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l400_40024


namespace NUMINAMATH_CALUDE_min_zeros_odd_periodic_function_l400_40026

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem min_zeros_odd_periodic_function 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : has_period f (2 * Real.pi)) 
  (h_zero_3 : f 3 = 0) 
  (h_zero_4 : f 4 = 0) : 
  ∃ (zeros : Finset ℝ), 
    (∀ x ∈ zeros, x ∈ Set.Icc 0 10 ∧ f x = 0) ∧ 
    Finset.card zeros ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_min_zeros_odd_periodic_function_l400_40026


namespace NUMINAMATH_CALUDE_frustum_volume_ratio_l400_40027

/-- Given a right prism with square base of side length L₁ and height H, 
    and a frustum of a pyramid extracted from it with square bases of 
    side lengths L₁ (lower) and L₂ (upper) and height H, 
    if the volume of the frustum is 2/3 of the total volume of the prism, 
    then L₁/L₂ = (1 + √5) / 2 -/
theorem frustum_volume_ratio (L₁ L₂ H : ℝ) (h_positive : L₁ > 0 ∧ L₂ > 0 ∧ H > 0) :
  (H / 3 * (L₁^2 + L₁*L₂ + L₂^2)) = (2 / 3 * L₁^2 * H) → 
  L₁ / L₂ = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_frustum_volume_ratio_l400_40027


namespace NUMINAMATH_CALUDE_abs_neg_two_l400_40055

theorem abs_neg_two : |(-2 : ℤ)| = 2 := by sorry

end NUMINAMATH_CALUDE_abs_neg_two_l400_40055


namespace NUMINAMATH_CALUDE_daisies_per_bouquet_l400_40087

/-- Represents a flower shop selling bouquets of roses and daisies. -/
structure FlowerShop where
  roses_per_bouquet : ℕ
  total_bouquets : ℕ
  rose_bouquets : ℕ
  daisy_bouquets : ℕ
  total_flowers : ℕ

/-- Theorem stating the number of daisies in each bouquet. -/
theorem daisies_per_bouquet (shop : FlowerShop)
  (h1 : shop.roses_per_bouquet = 12)
  (h2 : shop.total_bouquets = 20)
  (h3 : shop.rose_bouquets = 10)
  (h4 : shop.daisy_bouquets = 10)
  (h5 : shop.total_flowers = 190)
  (h6 : shop.total_bouquets = shop.rose_bouquets + shop.daisy_bouquets) :
  (shop.total_flowers - shop.roses_per_bouquet * shop.rose_bouquets) / shop.daisy_bouquets = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_daisies_per_bouquet_l400_40087


namespace NUMINAMATH_CALUDE_gretchens_earnings_l400_40064

/-- The amount Gretchen charges per drawing -/
def price_per_drawing : ℕ := 20

/-- The number of drawings sold on Saturday -/
def saturday_sales : ℕ := 24

/-- The number of drawings sold on Sunday -/
def sunday_sales : ℕ := 16

/-- Gretchen's total earnings over the weekend -/
def total_earnings : ℕ := price_per_drawing * (saturday_sales + sunday_sales)

/-- Theorem stating that Gretchen's total earnings are $800 -/
theorem gretchens_earnings : total_earnings = 800 := by
  sorry

end NUMINAMATH_CALUDE_gretchens_earnings_l400_40064
