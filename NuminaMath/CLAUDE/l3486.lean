import Mathlib

namespace NUMINAMATH_CALUDE_english_only_enrollment_l3486_348626

theorem english_only_enrollment (total : ℕ) (eg ef gf egf : ℕ) (g f : ℕ) :
  total = 50 ∧
  eg = 12 ∧
  g = 22 ∧
  f = 18 ∧
  ef = 10 ∧
  gf = 8 ∧
  egf = 4 →
  ∃ (e g_only f_only : ℕ),
    e + g_only + f_only + eg + ef + gf - egf = total ∧
    g_only = g - (eg + gf - egf) ∧
    f_only = f - (ef + gf - egf) ∧
    e = 14 :=
by sorry

#check english_only_enrollment

end NUMINAMATH_CALUDE_english_only_enrollment_l3486_348626


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3486_348678

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 * x + 9) = 13 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3486_348678


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3486_348602

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : 
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  (360 : ℝ) / exterior_angle = n →
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3486_348602


namespace NUMINAMATH_CALUDE_double_average_l3486_348605

theorem double_average (n : ℕ) (original_avg : ℝ) (h1 : n = 25) (h2 : original_avg = 70) :
  let total_marks := n * original_avg
  let doubled_marks := 2 * total_marks
  let new_avg := doubled_marks / n
  new_avg = 140 := by
sorry

end NUMINAMATH_CALUDE_double_average_l3486_348605


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3486_348630

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line = Line.mk 2 (-3) 4 →
  point = Point.mk (-1) 2 →
  ∃ (result_line : Line),
    result_line.perpendicular given_line ∧
    point.liesOn result_line ∧
    result_line = Line.mk 3 2 (-1) := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l3486_348630


namespace NUMINAMATH_CALUDE_french_fries_cooking_time_l3486_348613

/-- Calculates the remaining cooking time in seconds given the recommended time in minutes and the actual cooking time in seconds. -/
def remaining_cooking_time (recommended_minutes : ℕ) (actual_seconds : ℕ) : ℕ :=
  recommended_minutes * 60 - actual_seconds

/-- Theorem stating that for a recommended cooking time of 5 minutes and an actual cooking time of 45 seconds, the remaining cooking time is 255 seconds. -/
theorem french_fries_cooking_time : remaining_cooking_time 5 45 = 255 := by
  sorry

end NUMINAMATH_CALUDE_french_fries_cooking_time_l3486_348613


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l3486_348681

theorem four_digit_number_problem :
  ∃ (n : ℕ),
    1000 ≤ n ∧ n < 10000 ∧  -- four-digit number
    (n / 1000 = 2) ∧  -- thousand's place is 2
    (((n % 1000) * 10 + 2) = 2 * n + 66) ∧  -- condition for moving 2 to unit's place
    n = 2508 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_problem_l3486_348681


namespace NUMINAMATH_CALUDE_tire_circumference_l3486_348690

-- Define the given conditions
def car_speed : Real := 168 -- km/h
def tire_revolutions : Real := 400 -- revolutions per minute

-- Define the conversion factors
def km_to_m : Real := 1000 -- 1 km = 1000 m
def hour_to_minute : Real := 60 -- 1 hour = 60 minutes

-- Theorem statement
theorem tire_circumference :
  let speed_m_per_minute : Real := car_speed * km_to_m / hour_to_minute
  let circumference : Real := speed_m_per_minute / tire_revolutions
  circumference = 7 := by sorry

end NUMINAMATH_CALUDE_tire_circumference_l3486_348690


namespace NUMINAMATH_CALUDE_vector_problem_l3486_348679

/-- Given vectors in 2D space -/
def a : ℝ × ℝ := (5, 6)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def c (y : ℝ) : ℝ × ℝ := (2, y)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Perpendicular vectors have zero dot product -/
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

/-- Parallel vectors are scalar multiples of each other -/
def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), w = (k * v.1, k * v.2)

/-- Main theorem -/
theorem vector_problem :
  ∃ (x y : ℝ),
    perpendicular a (b x) ∧
    parallel a (c y) ∧
    x = -18/5 ∧
    y = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3486_348679


namespace NUMINAMATH_CALUDE_consecutive_eight_product_divisible_by_ten_l3486_348683

theorem consecutive_eight_product_divisible_by_ten (n : ℕ+) : 
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7)) := by
  sorry

#check consecutive_eight_product_divisible_by_ten

end NUMINAMATH_CALUDE_consecutive_eight_product_divisible_by_ten_l3486_348683


namespace NUMINAMATH_CALUDE_unique_arrangement_l3486_348644

-- Define the characters
inductive Character
| GrayHorse
| GrayMare
| BearCub

-- Define the positions
inductive Position
| Left
| Center
| Right

-- Define the arrangement as a function from Position to Character
def Arrangement := Position → Character

-- Define the property of always lying
def alwaysLies (c : Character) : Prop :=
  c = Character.GrayHorse

-- Define the property of never lying
def neverLies (c : Character) : Prop :=
  c = Character.BearCub

-- Define the statements made by each position
def leftStatement (arr : Arrangement) : Prop :=
  arr Position.Center = Character.BearCub

def rightStatement (arr : Arrangement) : Prop :=
  arr Position.Left = Character.GrayMare

def centerStatement (arr : Arrangement) : Prop :=
  arr Position.Left = Character.GrayHorse

-- Define the correctness of a statement based on who said it
def isCorrectStatement (arr : Arrangement) (pos : Position) (stmt : Prop) : Prop :=
  (alwaysLies (arr pos) ∧ ¬stmt) ∨
  (neverLies (arr pos) ∧ stmt) ∨
  (¬alwaysLies (arr pos) ∧ ¬neverLies (arr pos))

-- Main theorem
theorem unique_arrangement :
  ∃! arr : Arrangement,
    (arr Position.Left = Character.GrayMare) ∧
    (arr Position.Center = Character.GrayHorse) ∧
    (arr Position.Right = Character.BearCub) ∧
    isCorrectStatement arr Position.Left (leftStatement arr) ∧
    isCorrectStatement arr Position.Right (rightStatement arr) ∧
    isCorrectStatement arr Position.Center (centerStatement arr) :=
  sorry


end NUMINAMATH_CALUDE_unique_arrangement_l3486_348644


namespace NUMINAMATH_CALUDE_parallel_lines_sum_l3486_348659

/-- Two parallel lines with a given distance between them -/
structure ParallelLines where
  m : ℝ
  n : ℝ
  h_m_pos : m > 0
  h_parallel : 1 / 2 = -2 / n
  h_distance : |m + 3| / Real.sqrt 5 = Real.sqrt 5

/-- The sum of m and n for the parallel lines is -2 -/
theorem parallel_lines_sum (l : ParallelLines) : l.m + l.n = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_sum_l3486_348659


namespace NUMINAMATH_CALUDE_arithmetic_sequence_specific_term_l3486_348608

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_specific_term
  (seq : ArithmeticSequence)
  (m : ℕ)
  (h1 : seq.S (m - 2) = -4)
  (h2 : seq.S m = 0)
  (h3 : seq.S (m + 2) = 12) :
  seq.a m = 3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_specific_term_l3486_348608


namespace NUMINAMATH_CALUDE_square_to_circle_ratio_l3486_348641

-- Define the sector and its properties
structure RectangularSector where
  R : ℝ  -- Radius of the sector
  a : ℝ  -- Side length of the inscribed square

-- Define the circle touching the chord, arc, and square side
def TouchingCircle (sector : RectangularSector) :=
  { r : ℝ // r > 0 }

-- State the theorem
theorem square_to_circle_ratio
  (sector : RectangularSector)
  (circle : TouchingCircle sector) :
  sector.a / circle.val =
    ((Real.sqrt 5 + Real.sqrt 2) * (3 + Real.sqrt 5)) / (6 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_square_to_circle_ratio_l3486_348641


namespace NUMINAMATH_CALUDE_cement_tess_is_5_1_l3486_348692

/-- The amount of cement used for Tess's street -/
def cement_tess : ℝ := 15.1 - 10

/-- Proof that the amount of cement used for Tess's street is 5.1 tons -/
theorem cement_tess_is_5_1 : cement_tess = 5.1 := by
  sorry

end NUMINAMATH_CALUDE_cement_tess_is_5_1_l3486_348692


namespace NUMINAMATH_CALUDE_first_term_of_special_arithmetic_sequence_l3486_348675

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem first_term_of_special_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n, a (n + 1) > a n) →
  a 1 + a 2 + a 3 = 12 →
  a 1 * a 2 * a 3 = 48 →
  a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_special_arithmetic_sequence_l3486_348675


namespace NUMINAMATH_CALUDE_inverse_proportionality_fraction_l3486_348693

theorem inverse_proportionality_fraction (k : ℝ) (x y : ℝ) (h : x > 0 ∧ y > 0) :
  (k = x * y) → (∃ c : ℝ, c > 0 ∧ y = c / x) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportionality_fraction_l3486_348693


namespace NUMINAMATH_CALUDE_cyclic_equation_solution_l3486_348633

def cyclic_index (n i : ℕ) : ℕ :=
  (i - 1) % n + 1

theorem cyclic_equation_solution (n : ℕ) (x : ℕ → ℝ) :
  (∀ i, 0 ≤ x i) →
  (∀ k, x k + x (cyclic_index n (k + 1)) = (x (cyclic_index n (k + 2)))^2) →
  (∀ i, x i = 0 ∨ x i = 2) :=
sorry

end NUMINAMATH_CALUDE_cyclic_equation_solution_l3486_348633


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3486_348628

theorem linear_equation_solution :
  ∃ (x y : ℝ), 2 * x - 3 * y = 5 ∧ x = 1 ∧ y = -1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3486_348628


namespace NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l3486_348658

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_six_terms :
  geometric_sum (1/4) (1/4) 6 = 4095/12288 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l3486_348658


namespace NUMINAMATH_CALUDE_total_students_correct_l3486_348635

/-- The number of students who tried out for the trivia teams -/
def total_students : ℕ := 17

/-- The number of students who didn't get picked -/
def not_picked : ℕ := 5

/-- The number of groups formed -/
def num_groups : ℕ := 3

/-- The number of students in each group -/
def students_per_group : ℕ := 4

/-- Theorem stating that the total number of students who tried out is correct -/
theorem total_students_correct : 
  total_students = not_picked + num_groups * students_per_group := by
  sorry

end NUMINAMATH_CALUDE_total_students_correct_l3486_348635


namespace NUMINAMATH_CALUDE_semicircle_area_shaded_area_calculation_l3486_348655

theorem semicircle_area (r : ℝ) (h : r = 2.5) : 
  (π * r^2) / 2 = 3.125 * π := by
  sorry

/- Definitions based on problem conditions -/
def semicircle_ADB_radius : ℝ := 2
def semicircle_BEC_radius : ℝ := 1
def point_D : ℝ × ℝ := (1, 2)  -- midpoint of arc ADB
def point_E : ℝ × ℝ := (3, 1)  -- midpoint of arc BEC
def point_F : ℝ × ℝ := (3, 2.5)  -- midpoint of arc DFE

/- Main theorem -/
theorem shaded_area_calculation : 
  let r : ℝ := semicircle_ADB_radius + semicircle_BEC_radius / 2
  (π * r^2) / 2 = 3.125 * π := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_shaded_area_calculation_l3486_348655


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3486_348610

/-- An arithmetic sequence with common difference -2 and S_5 = 10 has a_100 = -192 -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, a (n + 1) - a n = -2) →  -- arithmetic sequence with common difference -2
  (S 5 = 10) →                   -- sum of first 5 terms is 10
  (∀ n, S n = n * a 1 + n * (n - 1) * (-1)) →  -- formula for sum of arithmetic sequence
  (a 100 = -192) :=              -- a_100 = -192
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3486_348610


namespace NUMINAMATH_CALUDE_power_of_81_five_sixths_l3486_348660

theorem power_of_81_five_sixths :
  (81 : ℝ) ^ (5/6) = 27 * (3 : ℝ) ^ (1/3) := by sorry

end NUMINAMATH_CALUDE_power_of_81_five_sixths_l3486_348660


namespace NUMINAMATH_CALUDE_hole_filling_proof_l3486_348656

/-- The amount of water initially in the hole -/
def initial_water : ℕ := 676

/-- The additional amount of water needed to fill the hole -/
def additional_water : ℕ := 147

/-- The total amount of water needed to fill the hole -/
def total_water : ℕ := initial_water + additional_water

theorem hole_filling_proof : total_water = 823 := by
  sorry

end NUMINAMATH_CALUDE_hole_filling_proof_l3486_348656


namespace NUMINAMATH_CALUDE_quilt_remaining_squares_l3486_348611

/-- Given a quilt with 16 squares on each side and 25% of it already sewn,
    prove that the number of remaining squares to sew is 24. -/
theorem quilt_remaining_squares (squares_per_side : ℕ) (percent_sewn : ℚ) : 
  squares_per_side = 16 →
  percent_sewn = 1/4 →
  (2 * squares_per_side : ℕ) - (percent_sewn * (2 * squares_per_side : ℕ) : ℚ).num = 24 := by
  sorry

end NUMINAMATH_CALUDE_quilt_remaining_squares_l3486_348611


namespace NUMINAMATH_CALUDE_zero_in_interval_l3486_348616

/-- The function f(x) = log_a x + x - b -/
noncomputable def f (a b x : ℝ) : ℝ := (Real.log x) / (Real.log a) + x - b

/-- The theorem stating that the zero of f(x) lies in (2, 3) -/
theorem zero_in_interval (a b : ℝ) (ha : 0 < a) (ha' : a ≠ 1) 
  (hab : 2 < a ∧ a < 3 ∧ 3 < b ∧ b < 4) :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 2 3 ∧ f a b x₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3486_348616


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_one_l3486_348672

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_negative_one :
  log10 (5/2) + 2 * log10 2 - (1/2)⁻¹ = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_one_l3486_348672


namespace NUMINAMATH_CALUDE_marble_density_l3486_348622

/-- Density of a rectangular prism made of marble -/
theorem marble_density (height : ℝ) (base_side : ℝ) (weight : ℝ) :
  height = 8 →
  base_side = 2 →
  weight = 86400 →
  weight / (base_side * base_side * height) = 2700 := by
  sorry

end NUMINAMATH_CALUDE_marble_density_l3486_348622


namespace NUMINAMATH_CALUDE_donnas_truck_weight_l3486_348642

-- Define the given weights and quantities
def bridge_limit : ℕ := 20000
def empty_truck_weight : ℕ := 12000
def soda_crates : ℕ := 20
def soda_crate_weight : ℕ := 50
def dryers : ℕ := 3
def dryer_weight : ℕ := 3000

-- Define the theorem
theorem donnas_truck_weight :
  let soda_weight := soda_crates * soda_crate_weight
  let dryers_weight := dryers * dryer_weight
  let produce_weight := 2 * soda_weight
  empty_truck_weight + soda_weight + dryers_weight + produce_weight = 24000 := by
  sorry

end NUMINAMATH_CALUDE_donnas_truck_weight_l3486_348642


namespace NUMINAMATH_CALUDE_duke_dvd_count_l3486_348620

/-- Represents the number of DVDs Duke found in the first box -/
def first_box_count : ℕ := sorry

/-- Represents the price of each DVD in the first box -/
def first_box_price : ℚ := 2

/-- Represents the number of DVDs Duke found in the second box -/
def second_box_count : ℕ := 5

/-- Represents the price of each DVD in the second box -/
def second_box_price : ℚ := 5

/-- Represents the average price of all DVDs bought -/
def average_price : ℚ := 3

theorem duke_dvd_count : first_box_count = 5 := by
  sorry

end NUMINAMATH_CALUDE_duke_dvd_count_l3486_348620


namespace NUMINAMATH_CALUDE_integer_pair_divisibility_l3486_348615

theorem integer_pair_divisibility (x y : ℕ+) : 
  (((x : ℤ) * y - 6)^2 ∣ (x : ℤ)^2 + y^2) ↔ 
  ((x = 7 ∧ y = 1) ∨ (x = 4 ∧ y = 2) ∨ (x = 3 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_divisibility_l3486_348615


namespace NUMINAMATH_CALUDE_total_cost_price_is_60_2_l3486_348682

/-- Calculates the cost price given the selling price and loss ratio -/
def costPrice (sellingPrice : ℚ) (lossRatio : ℚ) : ℚ :=
  sellingPrice / (1 - lossRatio)

/-- The total cost price of an apple, an orange, and a banana -/
def totalCostPrice : ℚ :=
  costPrice 16 (1/6) + costPrice 20 (1/5) + costPrice 12 (1/4)

theorem total_cost_price_is_60_2 :
  totalCostPrice = 60.2 := by sorry

end NUMINAMATH_CALUDE_total_cost_price_is_60_2_l3486_348682


namespace NUMINAMATH_CALUDE_odd_function_properties_l3486_348603

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + (a - 1)*x^2 + a*x + b

-- Define the property of f being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Main theorem
theorem odd_function_properties (a b : ℝ) 
  (h : is_odd_function (f a b)) : 
  (a + b = 1) ∧ 
  (∃ m c : ℝ, m = 4 ∧ c = -2 ∧ 
    ∀ x y : ℝ, y = f a b x → (y - f a b 1 = m * (x - 1) ↔ m*x - y + c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l3486_348603


namespace NUMINAMATH_CALUDE_inequality_proof_l3486_348617

theorem inequality_proof (a b c A α : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hα : α > 0)
  (hA : a + b + c = A) (hA_bound : A ≤ 1) : 
  (1/a - a)^α + (1/b - b)^α + (1/c - c)^α ≥ 3 * ((3/A) - (A/3))^α := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3486_348617


namespace NUMINAMATH_CALUDE_x_complements_c_l3486_348687

/-- Represents a date in a month --/
structure Date :=
  (value : ℕ)
  (h : value > 0 ∧ value ≤ 31)

/-- Represents letters on the calendar --/
inductive Letter
| A | B | C | X

/-- A calendar is a function that assigns a date to each letter --/
def Calendar := Letter → Date

/-- The condition that B is two weeks after A --/
def twoWeeksAfter (cal : Calendar) : Prop :=
  (cal Letter.B).value = (cal Letter.A).value + 14

/-- The condition that the sum of dates behind C and X equals the sum of dates behind A and B --/
def sumEqual (cal : Calendar) : Prop :=
  (cal Letter.C).value + (cal Letter.X).value = (cal Letter.A).value + (cal Letter.B).value

/-- The main theorem --/
theorem x_complements_c (cal : Calendar) 
  (h1 : twoWeeksAfter cal) 
  (h2 : sumEqual cal) : 
  (cal Letter.X).value = (cal Letter.C).value + 18 :=
sorry

end NUMINAMATH_CALUDE_x_complements_c_l3486_348687


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l3486_348612

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 8000 → 
  candidate_percentage = 35 / 100 → 
  (total_votes : ℚ) * candidate_percentage + 
  (total_votes : ℚ) * (1 - candidate_percentage) = total_votes → 
  (total_votes : ℚ) * (1 - candidate_percentage) - 
  (total_votes : ℚ) * candidate_percentage = 2400 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l3486_348612


namespace NUMINAMATH_CALUDE_pig_price_l3486_348691

/-- Given 5 pigs and 15 hens with a total cost of 2100 currency units,
    and an average price of 30 currency units per hen,
    prove that the average price of a pig is 330 currency units. -/
theorem pig_price (num_pigs : ℕ) (num_hens : ℕ) (total_cost : ℕ) (hen_price : ℕ) :
  num_pigs = 5 →
  num_hens = 15 →
  total_cost = 2100 →
  hen_price = 30 →
  (total_cost - num_hens * hen_price) / num_pigs = 330 := by
  sorry

end NUMINAMATH_CALUDE_pig_price_l3486_348691


namespace NUMINAMATH_CALUDE_wage_increase_percentage_l3486_348643

theorem wage_increase_percentage (initial_wage : ℝ) (final_wage : ℝ) : 
  initial_wage = 10 →
  final_wage = 9 →
  final_wage = 0.75 * (initial_wage * (1 + x/100)) →
  x = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_wage_increase_percentage_l3486_348643


namespace NUMINAMATH_CALUDE_pieces_present_l3486_348650

/-- The number of pieces in a standard chess set -/
def standard_chess_pieces : ℕ := 32

/-- The number of missing pieces -/
def missing_pieces : ℕ := 4

/-- Theorem: The number of pieces present in an incomplete chess set -/
theorem pieces_present (standard : ℕ) (missing : ℕ) 
  (h1 : standard = standard_chess_pieces) 
  (h2 : missing = missing_pieces) : 
  standard - missing = 28 := by
  sorry

end NUMINAMATH_CALUDE_pieces_present_l3486_348650


namespace NUMINAMATH_CALUDE_binomial_20_17_l3486_348666

theorem binomial_20_17 : (Nat.choose 20 17) = 1140 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_17_l3486_348666


namespace NUMINAMATH_CALUDE_alloy_ratio_theorem_l3486_348667

/-- Represents an alloy with zinc and copper -/
structure Alloy where
  zinc : ℚ
  copper : ℚ

/-- The first alloy with zinc:copper ratio of 1:2 -/
def alloy1 : Alloy := { zinc := 1, copper := 2 }

/-- The second alloy with zinc:copper ratio of 2:3 -/
def alloy2 : Alloy := { zinc := 2, copper := 3 }

/-- The desired third alloy with zinc:copper ratio of 17:27 -/
def alloy3 : Alloy := { zinc := 17, copper := 27 }

/-- Theorem stating the ratio of alloys needed to create the third alloy -/
theorem alloy_ratio_theorem :
  ∃ (x y : ℚ),
    x > 0 ∧ y > 0 ∧
    (x * alloy1.zinc + y * alloy2.zinc) / (x * alloy1.copper + y * alloy2.copper) = alloy3.zinc / alloy3.copper ∧
    x / y = 9 / 35 := by
  sorry


end NUMINAMATH_CALUDE_alloy_ratio_theorem_l3486_348667


namespace NUMINAMATH_CALUDE_mack_writes_sixteen_pages_l3486_348636

/-- Calculates the total number of pages Mack writes from Monday to Thursday -/
def total_pages (T1 R1 T2 R2 P3 T4 T5 R3 R4 : ℕ) : ℕ :=
  let monday_pages := T1 / R1
  let tuesday_pages := T2 / R2
  let wednesday_pages := P3
  let thursday_first_part := T5 / R3
  let thursday_second_part := (T4 - T5) / R4
  let thursday_pages := thursday_first_part + thursday_second_part
  monday_pages + tuesday_pages + wednesday_pages + thursday_pages

/-- Theorem stating that given the specified conditions, Mack writes 16 pages in total -/
theorem mack_writes_sixteen_pages :
  total_pages 60 30 45 15 5 90 30 10 20 = 16 := by
  sorry

end NUMINAMATH_CALUDE_mack_writes_sixteen_pages_l3486_348636


namespace NUMINAMATH_CALUDE_decimal_density_between_half_and_seven_tenths_l3486_348680

theorem decimal_density_between_half_and_seven_tenths :
  ∃ (x y : ℚ), 0.5 < x ∧ x < y ∧ y < 0.7 :=
sorry

end NUMINAMATH_CALUDE_decimal_density_between_half_and_seven_tenths_l3486_348680


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3486_348648

def binomial_coeff (n k : ℕ) : ℕ := sorry

def general_term (r : ℕ) : ℤ :=
  (binomial_coeff 5 r : ℤ) * (-1)^r

theorem constant_term_expansion :
  (general_term 1) + (general_term 3) + (general_term 5) = -51 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3486_348648


namespace NUMINAMATH_CALUDE_lunks_for_dozen_apples_l3486_348698

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (lunks : ℚ) : ℚ := (3 / 5) * lunks

/-- Exchange rate between kunks and apples -/
def kunks_to_apples (kunks : ℚ) : ℚ := 2 * kunks

/-- The number of lunks required to purchase a given number of apples -/
def lunks_for_apples (apples : ℚ) : ℚ :=
  (5 / 3) * (apples / 2)

theorem lunks_for_dozen_apples :
  lunks_for_apples 12 = 10 := by sorry

end NUMINAMATH_CALUDE_lunks_for_dozen_apples_l3486_348698


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3486_348601

-- Problem 1
theorem problem_1 (a b : ℤ) (h1 : a = 6) (h2 : b = -1) :
  2*a + 3*b - 2*a*b - a - 4*b - a*b = 25 := by sorry

-- Problem 2
theorem problem_2 (m n : ℤ) (h1 : |m| = 3) (h2 : |n| = 2) (h3 : m < n) :
  m^2 + 2*m*n + n^2 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3486_348601


namespace NUMINAMATH_CALUDE_remainder_3005_div_99_l3486_348695

theorem remainder_3005_div_99 : 3005 % 99 = 35 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3005_div_99_l3486_348695


namespace NUMINAMATH_CALUDE_rose_price_l3486_348623

/-- The price of roses given Hanna's budget and distribution to friends -/
theorem rose_price (total_budget : ℚ) (jenna_fraction : ℚ) (imma_fraction : ℚ) (friends_roses : ℕ) : 
  total_budget = 300 →
  jenna_fraction = 1/3 →
  imma_fraction = 1/2 →
  friends_roses = 125 →
  total_budget / ((friends_roses : ℚ) / (jenna_fraction + imma_fraction)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_rose_price_l3486_348623


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3486_348677

/-- Given an arithmetic sequence with common ratio q ≠ 0, where S_n is the sum of first n terms,
    if S_3, S_9, and S_6 form an arithmetic sequence, then q^3 = 3/2 -/
theorem arithmetic_sequence_ratio (q : ℝ) (a₁ : ℝ) (S : ℕ → ℝ) : 
  q ≠ 0 ∧ 
  (∀ n, S n = a₁ * (1 - q^n) / (1 - q)) ∧ 
  (2 * S 9 = S 3 + S 6) →
  q^3 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3486_348677


namespace NUMINAMATH_CALUDE_number_equation_solution_l3486_348668

theorem number_equation_solution :
  ∃ x : ℝ, 5.4 * x + 0.6 = 108.45000000000003 ∧ x = 19.97222222222222 :=
by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3486_348668


namespace NUMINAMATH_CALUDE_circle_center_is_two_one_l3486_348686

/-- A line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- A circle defined by its center and a point on its circumference -/
structure Circle where
  center : ℝ × ℝ
  point_on_circle : ℝ × ℝ

/-- The line l passing through (2, 1) and (6, 3) -/
def l : Line := { point1 := (2, 1), point2 := (6, 3) }

/-- The circle C with center on line l and tangent to x-axis at (2, 0) -/
noncomputable def C : Circle :=
  { center := sorry,  -- To be proved
    point_on_circle := (2, 0) }

theorem circle_center_is_two_one :
  C.center = (2, 1) := by sorry

end NUMINAMATH_CALUDE_circle_center_is_two_one_l3486_348686


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l3486_348639

theorem quadratic_equation_root (a b c : ℝ) (h1 : a - b + c = 0) (h2 : a ≠ 0) :
  a * (-1)^2 + b * (-1) + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l3486_348639


namespace NUMINAMATH_CALUDE_sin_squared_sum_range_l3486_348606

theorem sin_squared_sum_range (α β : ℝ) :
  3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 = 2 * Real.sin α →
  ∃ (x : ℝ), x = (Real.sin α)^2 + (Real.sin β)^2 ∧ 0 ≤ x ∧ x ≤ 4/9 :=
by sorry

end NUMINAMATH_CALUDE_sin_squared_sum_range_l3486_348606


namespace NUMINAMATH_CALUDE_shortest_distance_circle_to_line_l3486_348665

/-- The shortest distance from a point on a circle to a line -/
theorem shortest_distance_circle_to_line :
  let center : ℝ × ℝ := (3, -3)
  let radius : ℝ := 3
  let line := {p : ℝ × ℝ | p.1 = p.2}
  ∃ (shortest : ℝ), 
    shortest = 3 * (Real.sqrt 2 - 1) ∧
    ∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 →
      shortest ≤ Real.sqrt ((p.1 - p.2)^2 + (p.2 - p.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_circle_to_line_l3486_348665


namespace NUMINAMATH_CALUDE_election_votes_l3486_348663

theorem election_votes (V : ℕ) : 
  (60 * V / 100 - 40 * V / 100 = 1380) → V = 6900 := by sorry

end NUMINAMATH_CALUDE_election_votes_l3486_348663


namespace NUMINAMATH_CALUDE_adam_students_count_l3486_348600

/-- The number of students Adam teaches per year (except for the first year) -/
def studentsPerYear : ℕ := 50

/-- The number of students Adam teaches in the first year -/
def studentsFirstYear : ℕ := 40

/-- The total number of years Adam teaches -/
def totalYears : ℕ := 10

/-- The total number of students Adam teaches over the given period -/
def totalStudents : ℕ := studentsFirstYear + studentsPerYear * (totalYears - 1)

theorem adam_students_count : totalStudents = 490 := by
  sorry

end NUMINAMATH_CALUDE_adam_students_count_l3486_348600


namespace NUMINAMATH_CALUDE_domain_of_f_l3486_348689

noncomputable def f (x : ℝ) := (2 * x - 3) / (x^2 - 5 * x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 2 ∨ (2 < x ∧ x < 3) ∨ 3 < x} := by
  sorry

end NUMINAMATH_CALUDE_domain_of_f_l3486_348689


namespace NUMINAMATH_CALUDE_solve_for_t_l3486_348674

theorem solve_for_t (s t : ℚ) (eq1 : 8 * s + 6 * t = 160) (eq2 : s = t + 3) : t = 68 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l3486_348674


namespace NUMINAMATH_CALUDE_min_value_theorem_l3486_348632

theorem min_value_theorem (a b : ℝ) (hb : b > 0) :
  (1/2 * Real.exp a - Real.log (2*b))^2 + (a - b)^2 ≥ 2 * (1 - Real.log 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3486_348632


namespace NUMINAMATH_CALUDE_jackie_sleep_time_l3486_348684

theorem jackie_sleep_time (total_hours work_hours exercise_hours free_hours : ℕ) 
  (h1 : total_hours = 24)
  (h2 : work_hours = 8)
  (h3 : exercise_hours = 3)
  (h4 : free_hours = 5) :
  total_hours - (work_hours + exercise_hours + free_hours) = 8 := by
  sorry

end NUMINAMATH_CALUDE_jackie_sleep_time_l3486_348684


namespace NUMINAMATH_CALUDE_inverse_proportion_example_l3486_348688

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (p q : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, p x * q x = k

theorem inverse_proportion_example :
  ∀ p q : ℝ → ℝ,
  InverselyProportional p q →
  p 6 = 25 →
  p 15 = 10 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_example_l3486_348688


namespace NUMINAMATH_CALUDE_parabola_properties_l3486_348670

def parabola (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem parabola_properties :
  (∀ x, parabola x ≥ parabola 1) ∧
  (∀ x₁ x₂, x₁ > 1 ∧ x₂ > 1 ∧ x₂ > x₁ → parabola x₂ > parabola x₁) ∧
  (parabola 1 = -2) ∧
  (∀ x, parabola x = parabola (2 - x)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3486_348670


namespace NUMINAMATH_CALUDE_expression_value_l3486_348640

theorem expression_value (m n : ℝ) (h : n - m = 2) : 
  (m^2 - n^2) / m * (2 * m) / (m + n) = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3486_348640


namespace NUMINAMATH_CALUDE_range_of_m_l3486_348609

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem range_of_m (m : ℝ) : (A ∩ B m = B m) → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3486_348609


namespace NUMINAMATH_CALUDE_min_strikes_to_defeat_dragon_l3486_348651

/-- Represents the state of the dragon -/
structure DragonState where
  heads : Nat
  tails : Nat

/-- Represents a strike against the dragon -/
inductive Strike
  | CutOneHead
  | CutOneTail
  | CutTwoHeads
  | CutTwoTails

/-- Applies a strike to the dragon state -/
def applyStrike (state : DragonState) (strike : Strike) : DragonState :=
  match strike with
  | Strike.CutOneHead => ⟨state.heads, state.tails⟩
  | Strike.CutOneTail => ⟨state.heads, state.tails + 1⟩
  | Strike.CutTwoHeads => ⟨state.heads - 2, state.tails⟩
  | Strike.CutTwoTails => ⟨state.heads + 1, state.tails - 2⟩

/-- Checks if the dragon is defeated (no heads and tails) -/
def isDragonDefeated (state : DragonState) : Prop :=
  state.heads = 0 ∧ state.tails = 0

/-- Theorem: The minimum number of strikes to defeat the dragon is 9 -/
theorem min_strikes_to_defeat_dragon :
  ∃ (strikes : List Strike),
    strikes.length = 9 ∧
    isDragonDefeated (strikes.foldl applyStrike ⟨3, 3⟩) ∧
    ∀ (otherStrikes : List Strike),
      otherStrikes.length < 9 →
      ¬isDragonDefeated (otherStrikes.foldl applyStrike ⟨3, 3⟩) :=
by
  sorry

end NUMINAMATH_CALUDE_min_strikes_to_defeat_dragon_l3486_348651


namespace NUMINAMATH_CALUDE_parabola_intersection_condition_l3486_348661

theorem parabola_intersection_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 2*(m+2)*x₁ + m^2 - 1 = 0 ∧ 
    x₂^2 - 2*(m+2)*x₂ + m^2 - 1 = 0) 
  ↔ m > -5/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_condition_l3486_348661


namespace NUMINAMATH_CALUDE_cloth_squares_theorem_l3486_348631

/-- Calculates the maximum number of small squares that can be cut from a rectangle -/
def max_squares (length width square_side : ℕ) : ℕ :=
  (length / square_side) * (width / square_side)

/-- Proves that the maximum number of 2x2 cm squares from a 40x27 cm cloth is 260 -/
theorem cloth_squares_theorem :
  max_squares 40 27 2 = 260 := by
  sorry

end NUMINAMATH_CALUDE_cloth_squares_theorem_l3486_348631


namespace NUMINAMATH_CALUDE_denny_followers_after_one_year_l3486_348649

/-- Calculates the number of followers after one year --/
def followers_after_one_year (initial_followers : ℕ) (daily_new_followers : ℕ) (unfollows_per_year : ℕ) : ℕ :=
  initial_followers + daily_new_followers * 365 - unfollows_per_year

/-- Theorem stating that Denny will have 445,000 followers after one year --/
theorem denny_followers_after_one_year :
  followers_after_one_year 100000 1000 20000 = 445000 := by
  sorry

#eval followers_after_one_year 100000 1000 20000

end NUMINAMATH_CALUDE_denny_followers_after_one_year_l3486_348649


namespace NUMINAMATH_CALUDE_cos_120_degrees_l3486_348624

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l3486_348624


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3486_348664

-- Problem 1
theorem problem_1 (a : ℝ) : -2 * a^3 * 3 * a^2 = -6 * a^5 := by sorry

-- Problem 2
theorem problem_2 (m : ℝ) (hm : m ≠ 0) : m^4 * (m^2)^3 / m^8 = m^2 := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) : (-2*x - 1) * (2*x - 1) = 1 - 4*x^2 := by sorry

-- Problem 4
theorem problem_4 (x : ℝ) : (-3*x + 2)^2 = 9*x^2 - 12*x + 4 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3486_348664


namespace NUMINAMATH_CALUDE_train_passing_time_l3486_348653

/-- The time taken for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 120 →
  train_speed = 50 * (1000 / 3600) →
  man_speed = 4 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l3486_348653


namespace NUMINAMATH_CALUDE_function_symmetry_implies_a_equals_four_l3486_348607

/-- Given a quadratic function f(x) = 2x^2 - ax + 3, 
    if f(1-x) = f(1+x) for all real x, then a = 4 -/
theorem function_symmetry_implies_a_equals_four (a : ℝ) : 
  (∀ x : ℝ, 2*(1-x)^2 - a*(1-x) + 3 = 2*(1+x)^2 - a*(1+x) + 3) → 
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_implies_a_equals_four_l3486_348607


namespace NUMINAMATH_CALUDE_min_value_f_range_a_when_f_equals_one_range_a_when_f_geq_f_inv_l3486_348627

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

-- Theorem 1: Minimum value of f when a = 1
theorem min_value_f (x : ℝ) (h : x > 0) :
  ∃ (min : ℝ), min = 1 ∧ ∀ y > 0, f 1 y ≥ min :=
sorry

-- Theorem 2: Range of a when f(x) = 1 for some x in [1/e, e]
theorem range_a_when_f_equals_one (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ f a x = 1) →
  a ∈ Set.Icc 0 1 :=
sorry

-- Theorem 3: Range of a when f(x) ≥ f(1/x) for all x ≥ 1
theorem range_a_when_f_geq_f_inv (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ f a (1 / x)) →
  a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_range_a_when_f_equals_one_range_a_when_f_geq_f_inv_l3486_348627


namespace NUMINAMATH_CALUDE_gum_cost_theorem_l3486_348685

/-- Calculates the discounted cost in dollars for a bulk purchase of gum -/
def discounted_gum_cost (quantity : ℕ) (price_per_piece : ℚ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let total_cost := quantity * price_per_piece
  let discount := if quantity > discount_threshold then discount_rate * total_cost else 0
  (total_cost - discount) / 100

theorem gum_cost_theorem :
  discounted_gum_cost 1500 2 (1/10) 1000 = 27 := by
  sorry

end NUMINAMATH_CALUDE_gum_cost_theorem_l3486_348685


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l3486_348699

theorem smaller_circle_radius (R : ℝ) (r : ℝ) : 
  R = 10 → -- The radius of the largest circle is 10 meters
  4 * r = 2 * R → -- The diameter of the larger circle equals 4 times the radius of smaller circles
  r = 5 := by sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l3486_348699


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3486_348646

theorem solve_linear_equation (x : ℚ) :
  3 * x - 5 * x + 6 * x = 150 → x = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3486_348646


namespace NUMINAMATH_CALUDE_factories_unchecked_l3486_348669

theorem factories_unchecked (total : ℕ) (first_group : ℕ) (second_group : ℕ)
  (h1 : total = 169)
  (h2 : first_group = 69)
  (h3 : second_group = 52) :
  total - (first_group + second_group) = 48 := by
  sorry

end NUMINAMATH_CALUDE_factories_unchecked_l3486_348669


namespace NUMINAMATH_CALUDE_jacob_needs_18_marshmallows_l3486_348662

/-- Calculates the number of additional marshmallows needed for s'mores -/
def additional_marshmallows_needed (graham_crackers : ℕ) (marshmallows : ℕ) : ℕ :=
  let max_smores := graham_crackers / 2
  max_smores - marshmallows

/-- Proves that Jacob needs 18 more marshmallows -/
theorem jacob_needs_18_marshmallows :
  additional_marshmallows_needed 48 6 = 18 := by
  sorry

#eval additional_marshmallows_needed 48 6

end NUMINAMATH_CALUDE_jacob_needs_18_marshmallows_l3486_348662


namespace NUMINAMATH_CALUDE_pet_store_dogs_l3486_348625

theorem pet_store_dogs (initial_dogs : ℕ) (sunday_dogs : ℕ) (monday_dogs : ℕ) (final_dogs : ℕ)
  (h1 : initial_dogs = 2)
  (h2 : monday_dogs = 3)
  (h3 : final_dogs = 10)
  (h4 : initial_dogs + sunday_dogs + monday_dogs = final_dogs) :
  sunday_dogs = 5 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l3486_348625


namespace NUMINAMATH_CALUDE_sheet_length_is_48_l3486_348629

/-- Represents the dimensions and volume of a box made from a rectangular sheet. -/
structure BoxDimensions where
  sheet_length : ℝ
  sheet_width : ℝ
  cut_length : ℝ
  box_volume : ℝ

/-- Calculates the volume of a box given its dimensions. -/
def calculate_box_volume (d : BoxDimensions) : ℝ :=
  (d.sheet_length - 2 * d.cut_length) * (d.sheet_width - 2 * d.cut_length) * d.cut_length

/-- Theorem stating that given the specified conditions, the sheet length must be 48 meters. -/
theorem sheet_length_is_48 (d : BoxDimensions) 
  (h_width : d.sheet_width = 36)
  (h_cut : d.cut_length = 7)
  (h_volume : d.box_volume = 5236)
  (h_vol_calc : d.box_volume = calculate_box_volume d) : 
  d.sheet_length = 48 := by
  sorry

#check sheet_length_is_48

end NUMINAMATH_CALUDE_sheet_length_is_48_l3486_348629


namespace NUMINAMATH_CALUDE_prob_friends_same_group_l3486_348654

/-- The total number of students -/
def total_students : ℕ := 900

/-- The number of lunch groups -/
def num_groups : ℕ := 5

/-- The number of friends we're considering -/
def num_friends : ℕ := 4

/-- Represents a random assignment of students to lunch groups -/
def random_assignment : Type := Fin total_students → Fin num_groups

/-- The probability of a specific student being assigned to a specific group -/
def prob_single_assignment : ℚ := 1 / num_groups

/-- 
The probability that all friends are assigned to the same group
given a random assignment of students to groups
-/
def prob_all_friends_same_group (assignment : random_assignment) : ℚ :=
  prob_single_assignment ^ (num_friends - 1)

theorem prob_friends_same_group :
  ∀ (assignment : random_assignment),
    prob_all_friends_same_group assignment = 1 / 125 :=
by sorry

end NUMINAMATH_CALUDE_prob_friends_same_group_l3486_348654


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l3486_348647

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l3486_348647


namespace NUMINAMATH_CALUDE_paul_reading_books_l3486_348696

/-- The number of books Paul reads per week -/
def books_per_week : ℕ := 7

/-- The number of weeks Paul reads -/
def weeks : ℕ := 12

/-- The total number of books Paul reads -/
def total_books : ℕ := books_per_week * weeks

theorem paul_reading_books : total_books = 84 := by
  sorry

end NUMINAMATH_CALUDE_paul_reading_books_l3486_348696


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l3486_348619

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ :=
  sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nth_number_with_digit_sum_13 (n : ℕ+) : ℕ+ :=
  sorry

/-- Theorem stating that the 11th number with digit sum 13 is 145 -/
theorem eleventh_number_with_digit_sum_13 :
  nth_number_with_digit_sum_13 11 = 145 :=
sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l3486_348619


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3486_348697

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3486_348697


namespace NUMINAMATH_CALUDE_range_of_f_l3486_348657

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | 0 ≤ y ∧ y ≤ 4} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3486_348657


namespace NUMINAMATH_CALUDE_no_x_satisfying_conditions_l3486_348621

theorem no_x_satisfying_conditions : ¬∃ x : ℝ, 
  250 ≤ x ∧ x ≤ 350 ∧ 
  ⌊Real.sqrt (x - 50)⌋ = 14 ∧ 
  ⌊Real.sqrt (50 * x)⌋ = 256 := by
  sorry

#check no_x_satisfying_conditions

end NUMINAMATH_CALUDE_no_x_satisfying_conditions_l3486_348621


namespace NUMINAMATH_CALUDE_permutations_count_l3486_348637

theorem permutations_count (n : ℕ) : Nat.factorial n = 6227020800 → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_permutations_count_l3486_348637


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3486_348645

theorem two_numbers_difference (a b : ℕ) 
  (h1 : a + b = 12390)
  (h2 : b = 2 * a + 18) : 
  b - a = 4142 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3486_348645


namespace NUMINAMATH_CALUDE_ellipse_equation_1_ellipse_equation_2_l3486_348638

-- Define the type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the type for ellipses
structure Ellipse where
  a : ℝ
  b : ℝ

def is_on_ellipse (e : Ellipse) (p : Point2D) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

def has_common_focus (e1 e2 : Ellipse) : Prop :=
  ∃ (f : Point2D), (f.x^2 = e1.a^2 - e1.b^2) ∧ (f.x^2 = e2.a^2 - e2.b^2)

def has_foci_on_axes (e : Ellipse) : Prop :=
  ∃ (f : ℝ), (f^2 = e.a^2 - e.b^2) ∧ (f ≠ 0)

theorem ellipse_equation_1 :
  ∃ (e : Ellipse),
    has_common_focus e (Ellipse.mk 3 2) ∧
    is_on_ellipse e (Point2D.mk 3 (-2)) ∧
    has_foci_on_axes e ∧
    e.a^2 = 15 ∧ e.b^2 = 10 := by sorry

theorem ellipse_equation_2 :
  ∃ (e : Ellipse),
    has_foci_on_axes e ∧
    is_on_ellipse e (Point2D.mk (Real.sqrt 3) (-2)) ∧
    is_on_ellipse e (Point2D.mk (-2 * Real.sqrt 3) 1) ∧
    e.a^2 = 15 ∧ e.b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_1_ellipse_equation_2_l3486_348638


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3486_348614

theorem coin_flip_probability (p : ℝ) (n : ℕ) (h_p : p = 1 / 2) (h_n : n = 5) :
  p ^ 4 * (1 - p) = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3486_348614


namespace NUMINAMATH_CALUDE_max_prob_with_highest_prob_second_l3486_348671

/-- Represents a chess player with a given win probability -/
structure Player where
  winProb : ℝ

/-- Represents the chess player's opponents -/
structure Opponents where
  A : Player
  B : Player
  C : Player

/-- Calculates the probability of winning two consecutive games given the order of opponents -/
def probTwoConsecutiveWins (opponents : Opponents) (first second : Player) : ℝ :=
  2 * (first.winProb * second.winProb - 2 * opponents.A.winProb * opponents.B.winProb * opponents.C.winProb)

/-- Theorem stating that playing against the opponent with the highest win probability in the second game maximizes the probability of winning two consecutive games -/
theorem max_prob_with_highest_prob_second (opponents : Opponents)
    (h1 : opponents.C.winProb > opponents.B.winProb)
    (h2 : opponents.B.winProb > opponents.A.winProb)
    (h3 : opponents.A.winProb > 0) :
    ∀ (first : Player),
      probTwoConsecutiveWins opponents first opponents.C ≥ probTwoConsecutiveWins opponents first opponents.B ∧
      probTwoConsecutiveWins opponents first opponents.C ≥ probTwoConsecutiveWins opponents first opponents.A :=
  sorry


end NUMINAMATH_CALUDE_max_prob_with_highest_prob_second_l3486_348671


namespace NUMINAMATH_CALUDE_u_value_l3486_348676

theorem u_value : 
  let u : ℝ := 1 / (2 - Real.rpow 3 (1/3))
  u = 2 + Real.rpow 3 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_u_value_l3486_348676


namespace NUMINAMATH_CALUDE_cookie_distribution_l3486_348604

/-- The number of ways to distribute n identical items into k distinct groups -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of friends -/
def numFriends : ℕ := 4

/-- The total number of cookies -/
def totalCookies : ℕ := 10

/-- The minimum number of cookies each friend must have -/
def minCookies : ℕ := 2

/-- The number of ways to distribute the cookies -/
def numWays : ℕ := starsAndBars (totalCookies - minCookies * numFriends) numFriends

theorem cookie_distribution :
  numWays = 10 := by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l3486_348604


namespace NUMINAMATH_CALUDE_prime_factorization_sum_l3486_348634

theorem prime_factorization_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 945 → 2*w + 3*x + 5*y + 7*z = 21 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_sum_l3486_348634


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3486_348652

theorem expression_simplification_and_evaluation :
  let x : ℚ := 3
  let expr := (1 / (x - 1) + 1) / ((x^2 - 1) / (x^2 - 2*x + 1))
  expr = 3/4 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3486_348652


namespace NUMINAMATH_CALUDE_combined_work_time_l3486_348618

/-- Given the time taken by individuals A, B, and C to complete a task,
    this theorem calculates the time taken when they work together. -/
theorem combined_work_time (time_A time_B time_C : ℚ) :
  time_A = 12 →
  time_B = 14 →
  time_C = 16 →
  (1 / time_A + 1 / time_B + 1 / time_C)⁻¹ = 336 / 73 := by
  sorry

end NUMINAMATH_CALUDE_combined_work_time_l3486_348618


namespace NUMINAMATH_CALUDE_largest_k_for_g_range_three_l3486_348694

/-- The function g(x) = 2x^2 - 8x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 8 * x + k

/-- Theorem stating that 11 is the largest value of k such that 3 is in the range of g(x) -/
theorem largest_k_for_g_range_three :
  ∀ k : ℝ, (∃ x : ℝ, g k x = 3) ↔ k ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_g_range_three_l3486_348694


namespace NUMINAMATH_CALUDE_unique_solution_m_n_l3486_348673

theorem unique_solution_m_n : ∃! (m n : ℕ+), (m + n : ℕ)^(m : ℕ) = n^(m : ℕ) + 1413 :=
  sorry

end NUMINAMATH_CALUDE_unique_solution_m_n_l3486_348673
