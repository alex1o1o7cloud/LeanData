import Mathlib

namespace repeating_decimal_equality_l3381_338132

theorem repeating_decimal_equality (a : ℕ) : 
  1 ≤ a ∧ a ≤ 9 → (0.1 * a : ℚ) = 1 / a → a = 6 := by
  sorry

end repeating_decimal_equality_l3381_338132


namespace vector_equation_solution_l3381_338191

-- Define points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, 4)

-- Define vector a as a function of x
def a (x : ℝ) : ℝ × ℝ := (2*x - 1, x^2 + 3*x - 3)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem statement
theorem vector_equation_solution :
  ∃ x : ℝ, a x = AB ∧ x = 1 := by sorry

end vector_equation_solution_l3381_338191


namespace greatest_integer_difference_l3381_338138

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 7) :
  ∃ (n : ℕ), n = 3 ∧ ∀ (m : ℕ), (∃ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 7 ∧ m = ⌊b - a⌋) → m ≤ n :=
sorry

end greatest_integer_difference_l3381_338138


namespace arithmetic_sequence_sum_n_arithmetic_sequence_sum_17_arithmetic_sequence_sum_13_l3381_338189

-- Problem 1
theorem arithmetic_sequence_sum_n (a : ℕ → ℤ) (n : ℕ) :
  (∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) →  -- arithmetic sequence
  a 4 = 10 →
  a 10 = -2 →
  (n : ℤ) * (a 1 + a n) / 2 = 60 →
  n = 5 ∨ n = 6 := by sorry

-- Problem 2
theorem arithmetic_sequence_sum_17 (a : ℕ → ℤ) :
  a 1 = -7 →
  (∀ n, a (n + 1) = a n + 2) →
  (17 : ℤ) * (a 1 + a 17) / 2 = 153 := by sorry

-- Problem 3
theorem arithmetic_sequence_sum_13 (a : ℕ → ℤ) :
  (∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) →  -- arithmetic sequence
  a 2 + a 7 + a 12 = 24 →
  (13 : ℤ) * (a 1 + a 13) / 2 = 104 := by sorry

end arithmetic_sequence_sum_n_arithmetic_sequence_sum_17_arithmetic_sequence_sum_13_l3381_338189


namespace cd_purchase_total_l3381_338150

/-- The total cost of purchasing 3 copies each of three different CDs -/
def total_cost (price1 price2 price3 : ℕ) : ℕ :=
  3 * (price1 + price2 + price3)

theorem cd_purchase_total : total_cost 100 50 85 = 705 := by
  sorry

end cd_purchase_total_l3381_338150


namespace mean_score_of_all_students_l3381_338114

/-- Calculates the mean score of all students given the mean scores of two classes and the ratio of students in those classes. -/
theorem mean_score_of_all_students
  (morning_mean : ℝ)
  (afternoon_mean : ℝ)
  (morning_students : ℕ)
  (afternoon_students : ℕ)
  (h1 : morning_mean = 90)
  (h2 : afternoon_mean = 75)
  (h3 : morning_students = 2 * afternoon_students / 5) :
  let total_students := morning_students + afternoon_students
  let total_score := morning_mean * morning_students + afternoon_mean * afternoon_students
  total_score / total_students = 79 :=
by
  sorry

end mean_score_of_all_students_l3381_338114


namespace bread_butter_price_ratio_l3381_338194

/-- Proves that the ratio of bread price to butter price is 1:2 given the problem conditions --/
theorem bread_butter_price_ratio : 
  ∀ (butter bread cheese tea : ℝ),
  butter + bread + cheese + tea = 21 →
  butter = 0.8 * cheese →
  tea = 2 * cheese →
  tea = 10 →
  bread / butter = 1 / 2 := by
sorry

end bread_butter_price_ratio_l3381_338194


namespace dress_cost_calculation_l3381_338172

/-- The cost of a dress in dinars -/
def dress_cost : ℚ := 10/9

/-- The monthly pay in dinars (excluding the dress) -/
def monthly_pay : ℚ := 10

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The number of days worked to earn a dress -/
def days_worked : ℕ := 3

/-- Theorem stating the cost of the dress -/
theorem dress_cost_calculation :
  dress_cost = (monthly_pay + dress_cost) * days_worked / days_in_month :=
by sorry

end dress_cost_calculation_l3381_338172


namespace polygon_sides_l3381_338168

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 + 360 = 1800) → 
  n = 10 := by
sorry

end polygon_sides_l3381_338168


namespace floretta_balloon_count_l3381_338117

/-- The number of water balloons Floretta is left with after Milly takes extra -/
def florettas_balloons (total_packs : ℕ) (balloons_per_pack : ℕ) (extra_taken : ℕ) : ℕ :=
  (total_packs * balloons_per_pack) / 2 - extra_taken

/-- Theorem stating the number of balloons Floretta is left with -/
theorem floretta_balloon_count :
  florettas_balloons 5 6 7 = 8 := by
  sorry

end floretta_balloon_count_l3381_338117


namespace hyperbola_equation_l3381_338199

/-- Given a hyperbola with focal length 2c = 26 and a²/c = 25/13, 
    its standard equation is x²/25 - y²/144 = 1 or y²/25 - x²/144 = 1 -/
theorem hyperbola_equation (c : ℝ) (a : ℝ) (h1 : 2 * c = 26) (h2 : a^2 / c = 25 / 13) :
  (∃ x y : ℝ, x^2 / 25 - y^2 / 144 = 1) ∨ (∃ x y : ℝ, y^2 / 25 - x^2 / 144 = 1) :=
sorry

end hyperbola_equation_l3381_338199


namespace a_4_equals_11_l3381_338187

def S (n : ℕ+) : ℤ := 2 * n.val ^ 2 - 3 * n.val

def a (n : ℕ+) : ℤ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem a_4_equals_11 : a 4 = 11 := by
  sorry

end a_4_equals_11_l3381_338187


namespace fraction_addition_l3381_338159

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l3381_338159


namespace paint_time_together_l3381_338186

-- Define the rates of work for Harish and Ganpat
def harish_rate : ℚ := 1 / 3
def ganpat_rate : ℚ := 1 / 6

-- Define the total rate when working together
def total_rate : ℚ := harish_rate + ganpat_rate

-- Theorem to prove
theorem paint_time_together : (1 : ℚ) / total_rate = 2 := by
  sorry

end paint_time_together_l3381_338186


namespace f_of_one_eq_two_l3381_338115

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + |x - 2|

-- State the theorem
theorem f_of_one_eq_two : f 1 = 2 := by
  sorry

end f_of_one_eq_two_l3381_338115


namespace base_10_to_base_8_conversion_l3381_338128

theorem base_10_to_base_8_conversion : 
  (2 * 8^2 + 3 * 8^1 + 5 * 8^0 : ℕ) = 157 := by sorry

end base_10_to_base_8_conversion_l3381_338128


namespace average_speed_calculation_toms_trip_average_speed_l3381_338152

theorem average_speed_calculation (total_distance : Real) (first_part_distance : Real) 
  (first_part_speed : Real) (second_part_speed : Real) : Real :=
  let second_part_distance := total_distance - first_part_distance
  let first_part_time := first_part_distance / first_part_speed
  let second_part_time := second_part_distance / second_part_speed
  let total_time := first_part_time + second_part_time
  total_distance / total_time

theorem toms_trip_average_speed : 
  average_speed_calculation 60 12 24 48 = 40 := by
  sorry

end average_speed_calculation_toms_trip_average_speed_l3381_338152


namespace taxi_count_2008_l3381_338148

/-- Represents the number of taxis (in thousands) at the end of a given year -/
def taxiCount : ℕ → ℝ
| 0 => 100  -- End of 2005
| n + 1 => taxiCount n * 1.1 - 20  -- Subsequent years

/-- The year we're interested in (2008 is 3 years after 2005) -/
def targetYear : ℕ := 3

theorem taxi_count_2008 :
  12 ≤ taxiCount targetYear ∧ taxiCount targetYear < 13 := by
  sorry

end taxi_count_2008_l3381_338148


namespace perpendicular_lines_parallel_l3381_338111

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (l m : Line) (α : Plane) :
  l ≠ m →  -- l and m are different lines
  perpendicular l α →
  perpendicular m α →
  parallel l m :=
sorry

end perpendicular_lines_parallel_l3381_338111


namespace yan_journey_ratio_l3381_338141

/-- Represents a point on a line --/
structure Point :=
  (position : ℝ)

/-- Represents the scenario of Yan's journey --/
structure Journey :=
  (home : Point)
  (stadium : Point)
  (yan : Point)
  (walking_speed : ℝ)
  (cycling_speed : ℝ)

/-- The conditions of the journey --/
def journey_conditions (j : Journey) : Prop :=
  j.home.position < j.yan.position ∧
  j.yan.position < j.stadium.position ∧
  j.cycling_speed = 5 * j.walking_speed ∧
  (j.stadium.position - j.yan.position) / j.walking_speed =
    (j.yan.position - j.home.position) / j.walking_speed +
    (j.stadium.position - j.home.position) / j.cycling_speed

/-- The theorem to be proved --/
theorem yan_journey_ratio (j : Journey) (h : journey_conditions j) :
  (j.yan.position - j.home.position) / (j.stadium.position - j.yan.position) = 2 / 3 :=
sorry

end yan_journey_ratio_l3381_338141


namespace triangle_side_b_l3381_338108

theorem triangle_side_b (a b c : ℝ) (A B C : ℝ) : 
  a = 8 → B = π/3 → C = 5*π/12 → b = 4 * Real.sqrt 6 := by
  sorry

end triangle_side_b_l3381_338108


namespace three_lines_intersect_once_l3381_338190

/-- A parabola defined by y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is outside a parabola -/
def is_outside (pt : Point) (par : Parabola) : Prop :=
  pt.y^2 > 2 * par.p * pt.x

/-- Predicate to check if a line intersects a parabola at exactly one point -/
def intersects_once (l : Line) (par : Parabola) : Prop :=
  sorry -- Definition of intersection at exactly one point

/-- The main theorem -/
theorem three_lines_intersect_once (par : Parabola) (M : Point) 
  (h_outside : is_outside M par) : 
  ∃ (l₁ l₂ l₃ : Line), 
    (∀ l : Line, (intersects_once l par ∧ l.a * M.x + l.b * M.y + l.c = 0) ↔ 
      (l = l₁ ∨ l = l₂ ∨ l = l₃)) :=
sorry

end three_lines_intersect_once_l3381_338190


namespace orange_harvest_theorem_l3381_338120

/-- Calculates the total number of orange sacks kept after a given number of harvest days. -/
def total_sacks_kept (daily_harvest : ℕ) (daily_discard : ℕ) (harvest_days : ℕ) : ℕ :=
  (daily_harvest - daily_discard) * harvest_days

/-- Proves that given the specified harvest conditions, the total number of sacks kept is 1425. -/
theorem orange_harvest_theorem :
  total_sacks_kept 150 135 95 = 1425 := by
  sorry

end orange_harvest_theorem_l3381_338120


namespace quadratic_inequality_solution_l3381_338125

theorem quadratic_inequality_solution (z : ℝ) :
  z^2 - 42*z + 350 ≤ 4 ↔ 21 - Real.sqrt 95 ≤ z ∧ z ≤ 21 + Real.sqrt 95 := by
  sorry

end quadratic_inequality_solution_l3381_338125


namespace exponent_multiplication_l3381_338162

theorem exponent_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end exponent_multiplication_l3381_338162


namespace garden_area_l3381_338127

/-- The area of a rectangular garden with dimensions 90 cm and 4.5 meters is 4.05 square meters. -/
theorem garden_area : 
  let length_cm : ℝ := 90
  let width_m : ℝ := 4.5
  let length_m : ℝ := length_cm / 100
  let area_m2 : ℝ := length_m * width_m
  area_m2 = 4.05 := by
sorry

end garden_area_l3381_338127


namespace factor_implies_coefficients_l3381_338154

/-- If (x + 5) is a factor of x^4 - mx^3 + nx^2 - px + q, then m = 0, n = 0, p = 0, and q = -625 -/
theorem factor_implies_coefficients (m n p q : ℝ) : 
  (∀ x : ℝ, (x + 5) ∣ (x^4 - m*x^3 + n*x^2 - p*x + q)) →
  (m = 0 ∧ n = 0 ∧ p = 0 ∧ q = -625) := by
  sorry

end factor_implies_coefficients_l3381_338154


namespace jane_sequins_count_l3381_338196

/-- The number of rows of blue sequins -/
def blue_rows : ℕ := 6

/-- The number of blue sequins in each row -/
def blue_per_row : ℕ := 8

/-- The number of rows of purple sequins -/
def purple_rows : ℕ := 5

/-- The number of purple sequins in each row -/
def purple_per_row : ℕ := 12

/-- The number of rows of green sequins -/
def green_rows : ℕ := 9

/-- The number of green sequins in each row -/
def green_per_row : ℕ := 6

/-- The total number of sequins Jane adds to her costume -/
def total_sequins : ℕ := blue_rows * blue_per_row + purple_rows * purple_per_row + green_rows * green_per_row

theorem jane_sequins_count : total_sequins = 162 := by
  sorry

end jane_sequins_count_l3381_338196


namespace max_area_PAB_l3381_338181

-- Define the fixed points A and B
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 3)

-- Define the lines passing through A and B
def line1 (m : ℝ) (x y : ℝ) : Prop := x + m * y = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := m * x - y - m + 3 = 0

-- Define the intersection point P
def P (m : ℝ) : ℝ × ℝ := sorry

-- Define the area of triangle PAB
def area_PAB (m : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_area_PAB :
  ∃ (max_area : ℝ), 
    (∀ m : ℝ, area_PAB m ≤ max_area) ∧ 
    (∃ m : ℝ, area_PAB m = max_area) ∧
    max_area = 5/2 :=
sorry

end max_area_PAB_l3381_338181


namespace ages_solution_l3381_338195

/-- Represents the ages of three individuals -/
structure Ages where
  shekhar : ℚ
  shobha : ℚ
  kapil : ℚ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The ratio of ages is 4:3:2
  ages.shekhar / ages.shobha = 4 / 3 ∧
  ages.shekhar / ages.kapil = 2 ∧
  -- In 10 years, Kapil's age will equal Shekhar's present age
  ages.kapil + 10 = ages.shekhar ∧
  -- Shekhar's age will be 30 in 8 years
  ages.shekhar + 8 = 30

/-- The theorem to prove -/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧
    ages.shekhar = 22 ∧ ages.shobha = 33/2 ∧ ages.kapil = 10 := by
  sorry


end ages_solution_l3381_338195


namespace simplify_fraction_expression_l3381_338126

theorem simplify_fraction_expression (x y : ℝ) (hx : x ≠ 0) (hxy : x + y ≠ 0) :
  1 / (2 * x) - 1 / (x + y) * ((x + y) / (2 * x) - x - y) = 1 := by
  sorry

end simplify_fraction_expression_l3381_338126


namespace negation_of_universal_proposition_l3381_338166

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - 3*x + 2 < 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 2 ≥ 0) :=
by sorry

end negation_of_universal_proposition_l3381_338166


namespace gas_cost_per_gallon_l3381_338193

theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) :
  miles_per_gallon = 32 →
  total_miles = 432 →
  total_cost = 54 →
  (total_cost / (total_miles / miles_per_gallon)) = 4 :=
by sorry

end gas_cost_per_gallon_l3381_338193


namespace arithmetic_progression_x_value_l3381_338178

theorem arithmetic_progression_x_value :
  ∀ (x : ℚ),
  let a₁ := 2 * x - 4
  let a₂ := 2 * x + 2
  let a₃ := 4 * x + 1
  (a₂ - a₁ = a₃ - a₂) → x = 7/2 :=
by sorry

end arithmetic_progression_x_value_l3381_338178


namespace quadratic_inequality_solution_l3381_338130

/-- Given a quadratic inequality ax^2 + bx - 2 > 0 with solution set (1,4), prove a + b = 2 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x - 2 > 0 ↔ 1 < x ∧ x < 4) → 
  a + b = 2 := by sorry

end quadratic_inequality_solution_l3381_338130


namespace is_671st_term_l3381_338136

/-- The arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

/-- 2011 is the 671st term in the arithmetic sequence -/
theorem is_671st_term : arithmetic_sequence 671 = 2011 := by sorry

end is_671st_term_l3381_338136


namespace proposition_analysis_l3381_338105

-- Define proposition P
def P (x y : ℝ) : Prop := x ≠ y → abs x ≠ abs y

-- Define the converse of P
def P_converse (x y : ℝ) : Prop := abs x ≠ abs y → x ≠ y

-- Define the negation of P
def P_negation (x y : ℝ) : Prop := ¬(x ≠ y → abs x ≠ abs y)

-- Define the contrapositive of P
def P_contrapositive (x y : ℝ) : Prop := abs x = abs y → x = y

theorem proposition_analysis :
  (∃ x y : ℝ, ¬(P x y)) ∧
  (∀ x y : ℝ, P_converse x y) ∧
  (∀ x y : ℝ, P_negation x y) ∧
  (∃ x y : ℝ, ¬(P_contrapositive x y)) :=
sorry

end proposition_analysis_l3381_338105


namespace sample_size_calculation_l3381_338165

/-- Given a population of 1000 people and a simple random sampling method where
    the probability of each person being selected is 0.2, prove that the sample size is 200. -/
theorem sample_size_calculation (population : ℕ) (prob : ℝ) (sample_size : ℕ) :
  population = 1000 →
  prob = 0.2 →
  sample_size = population * prob →
  sample_size = 200 := by
  sorry

end sample_size_calculation_l3381_338165


namespace no_real_roots_implications_l3381_338161

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem no_real_roots_implications
  (a b c : ℝ)
  (h_a_nonzero : a ≠ 0)
  (h_no_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a < 0 → ∃ x : ℝ, f a b c (f a b c x) > x) ∧
  (a + b + c = 0 → ∀ x : ℝ, f a b c (f a b c x) < x) :=
by sorry

end no_real_roots_implications_l3381_338161


namespace circle_radius_zero_l3381_338151

theorem circle_radius_zero (x y : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 - 10*y + 41 = 0) → 
  ∃ h k r, (∀ x y, (x - h)^2 + (y - k)^2 = r^2) ∧ r = 0 :=
by sorry

end circle_radius_zero_l3381_338151


namespace triangle_side_length_l3381_338185

/-- Given a triangle ABC with area 3√3/4, side a = 3, and angle B = π/3, prove that side b = √7 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  (1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) →  -- Area formula
  (a = 3) →  -- Given side length
  (B = π/3) →  -- Given angle
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →  -- Law of cosines
  (b = Real.sqrt 7) := by sorry

end triangle_side_length_l3381_338185


namespace m_range_theorem_l3381_338113

-- Define proposition p
def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 - m * x + 1 < 0

-- Define proposition q
def q (m : ℝ) : Prop := (m - 1) * (3 - m) < 0

-- Define the range of m
def m_range (m : ℝ) : Prop := (0 < m ∧ m ≤ 1) ∨ (3 ≤ m ∧ m < 4)

-- Theorem statement
theorem m_range_theorem (m : ℝ) : 
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) → m_range m :=
sorry

end m_range_theorem_l3381_338113


namespace meal_prep_combinations_l3381_338147

def total_people : Nat := 6
def meal_preparers : Nat := 3

theorem meal_prep_combinations :
  Nat.choose total_people meal_preparers = 20 := by
  sorry

end meal_prep_combinations_l3381_338147


namespace pages_read_difference_l3381_338110

/-- Given a book with 270 pages, prove that reading 2/3 of it results in 90 more pages read than left to read. -/
theorem pages_read_difference (total_pages : ℕ) (fraction_read : ℚ) : 
  total_pages = 270 → 
  fraction_read = 2/3 →
  (fraction_read * total_pages : ℚ) - (total_pages - fraction_read * total_pages : ℚ) = 90 :=
by
  sorry

#check pages_read_difference

end pages_read_difference_l3381_338110


namespace transformation_D_not_always_valid_transformation_A_valid_transformation_B_valid_transformation_C_valid_l3381_338131

-- Define the transformations
def transformation_A (x y : ℝ) : Prop := x = y → x + 3 = y + 3
def transformation_B (x y : ℝ) : Prop := -2 * x = -2 * y → x = y
def transformation_C (x y m : ℝ) : Prop := x / m = y / m → x = y
def transformation_D (x y m : ℝ) : Prop := x = y → x / m = y / m

-- Define a property that checks if a transformation satisfies equation properties
def satisfies_equation_properties (t : (ℝ → ℝ → Prop)) : Prop :=
  ∀ x y : ℝ, t x y ↔ x = y

-- Theorem stating that transformation D does not always satisfy equation properties
theorem transformation_D_not_always_valid :
  ¬(∀ m : ℝ, satisfies_equation_properties (transformation_D · · m)) :=
sorry

-- Theorems stating that transformations A, B, and C satisfy equation properties
theorem transformation_A_valid :
  satisfies_equation_properties transformation_A :=
sorry

theorem transformation_B_valid :
  satisfies_equation_properties transformation_B :=
sorry

theorem transformation_C_valid :
  ∀ m : ℝ, m ≠ 0 → satisfies_equation_properties (transformation_C · · m) :=
sorry

end transformation_D_not_always_valid_transformation_A_valid_transformation_B_valid_transformation_C_valid_l3381_338131


namespace andrew_expenses_l3381_338146

def game_night_expenses (game_count : Nat) 
  (game_cost_1 : Nat) (game_count_1 : Nat)
  (game_cost_2 : Nat) (game_count_2 : Nat)
  (game_cost_3 : Nat) (game_count_3 : Nat)
  (snack_cost : Nat) (drink_cost : Nat) : Nat :=
  game_cost_1 * game_count_1 + game_cost_2 * game_count_2 + game_cost_3 * game_count_3 + snack_cost + drink_cost

theorem andrew_expenses : 
  game_night_expenses 7 900 3 1250 2 1500 2 2500 2000 = 12700 := by
  sorry

end andrew_expenses_l3381_338146


namespace valid_fraction_l3381_338116

theorem valid_fraction (x : ℝ) (h : x ≠ 3) : ∃ (f : ℝ → ℝ), f x = 1 / (x - 3) := by
  sorry

end valid_fraction_l3381_338116


namespace candy_problem_l3381_338107

theorem candy_problem (n : ℕ) (x : ℕ) (h1 : n > 1) (h2 : x > 1) 
  (h3 : ∀ i : ℕ, i < n → x = (n - 1) * x - 7) : 
  n * x = 21 := by
sorry

end candy_problem_l3381_338107


namespace age_ratio_proof_l3381_338167

def age_problem (A_current B_current : ℕ) : Prop :=
  B_current = 37 ∧
  A_current = B_current + 7 ∧
  (A_current + 10) / (B_current - 10) = 2

theorem age_ratio_proof :
  ∃ A_current B_current : ℕ, age_problem A_current B_current :=
by
  sorry

end age_ratio_proof_l3381_338167


namespace factors_of_2012_l3381_338139

theorem factors_of_2012 : Finset.card (Nat.divisors 2012) = 6 := by
  sorry

end factors_of_2012_l3381_338139


namespace opposite_of_neg_2023_l3381_338104

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- Theorem: The opposite of -2023 is 2023. -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l3381_338104


namespace arithmetic_sequence_sum_l3381_338197

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 6 = 2 →
  a 8 = 4 →
  a 10 + a 4 = 6 := by
sorry

end arithmetic_sequence_sum_l3381_338197


namespace rectangle_19_65_parts_l3381_338164

/-- Calculates the number of parts a rectangle is divided into when split into unit squares and crossed by a diagonal -/
def rectangle_parts (width : ℕ) (height : ℕ) : ℕ :=
  let unit_squares := width * height
  let diagonal_crossings := width + height - Nat.gcd width height
  unit_squares + diagonal_crossings

/-- The number of parts a 19 cm by 65 cm rectangle is divided into when split into 1 cm squares and crossed by a diagonal -/
theorem rectangle_19_65_parts : rectangle_parts 19 65 = 1318 := by
  sorry

end rectangle_19_65_parts_l3381_338164


namespace root_exists_in_interval_l3381_338140

-- Define the function f(x) = x^2 + 3x - 1
def f (x : ℝ) : ℝ := x^2 + 3*x - 1

-- State the theorem
theorem root_exists_in_interval :
  Continuous f →
  f 0 < 0 →
  f 0.5 > 0 →
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 0 0.5 ∧ f x₀ = 0 :=
by
  sorry

#check root_exists_in_interval

end root_exists_in_interval_l3381_338140


namespace problem_statement_l3381_338176

noncomputable def f (x : ℝ) : ℝ := 3^x + 2 / (1 - x)

theorem problem_statement 
  (x₀ x₁ x₂ : ℝ) 
  (h_root : f x₀ = 0)
  (h_x₁ : 1 < x₁ ∧ x₁ < x₀)
  (h_x₂ : x₀ < x₂) :
  f x₁ < 0 ∧ f x₂ > 0 := by
  sorry

end problem_statement_l3381_338176


namespace fort_sixty_percent_complete_l3381_338198

/-- Calculates the percentage of fort completion given the required sticks, 
    sticks collected per week, and number of weeks collecting. -/
def fort_completion_percentage 
  (required_sticks : ℕ) 
  (sticks_per_week : ℕ) 
  (weeks_collecting : ℕ) : ℚ :=
  (sticks_per_week * weeks_collecting : ℚ) / required_sticks * 100

/-- Theorem stating that given the specific conditions, 
    the fort completion percentage is 60%. -/
theorem fort_sixty_percent_complete : 
  fort_completion_percentage 400 3 80 = 60 := by
  sorry

end fort_sixty_percent_complete_l3381_338198


namespace purely_imaginary_complex_equation_l3381_338170

theorem purely_imaginary_complex_equation (z : ℂ) (a : ℝ) : 
  (z.re = 0) → ((2 - I) * z = a + I) → (a = 1/2) := by
  sorry

end purely_imaginary_complex_equation_l3381_338170


namespace town_population_problem_l3381_338155

theorem town_population_problem (original_population : ℝ) : 
  (original_population * 1.15 * 0.87 = original_population - 50) → 
  original_population = 100000 := by
  sorry

end town_population_problem_l3381_338155


namespace candy_division_l3381_338175

theorem candy_division (total_candy : ℚ) (num_piles : ℕ) (piles_for_carlos : ℕ) :
  total_candy = 75 / 7 →
  num_piles = 5 →
  piles_for_carlos = 2 →
  piles_for_carlos * (total_candy / num_piles) = 30 / 7 := by
  sorry

end candy_division_l3381_338175


namespace vector_combination_equals_result_l3381_338134

/-- Given vectors a, b, and c in ℝ³, prove that 2a - 3b + 4c equals the specified result -/
theorem vector_combination_equals_result (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (3, 5, -1)) 
  (hb : b = (2, 2, 3)) 
  (hc : c = (4, -1, -3)) : 
  (2 : ℝ) • a - (3 : ℝ) • b + (4 : ℝ) • c = (16, 0, -23) := by
  sorry

end vector_combination_equals_result_l3381_338134


namespace fixed_point_of_exponential_shift_l3381_338174

theorem fixed_point_of_exponential_shift (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 4 + a^(x - 1)
  f 1 = 5 := by sorry

end fixed_point_of_exponential_shift_l3381_338174


namespace gcd_1237_1849_l3381_338102

theorem gcd_1237_1849 : Nat.gcd 1237 1849 = 1 := by
  sorry

end gcd_1237_1849_l3381_338102


namespace x₃_value_l3381_338145

noncomputable def x₃ (x₁ x₂ : ℝ) : ℝ :=
  let y₁ := Real.exp x₁
  let y₂ := Real.exp x₂
  let yC := (2/3) * y₁ + (1/3) * y₂
  Real.log ((2/3) + (1/3) * Real.exp 2)

theorem x₃_value :
  let x₁ : ℝ := 0
  let x₂ : ℝ := 2
  let f : ℝ → ℝ := Real.exp
  x₃ x₁ x₂ = Real.log ((2/3) + (1/3) * Real.exp 2) :=
by sorry

end x₃_value_l3381_338145


namespace tire_price_proof_l3381_338109

theorem tire_price_proof :
  let regular_price : ℝ := 90
  let third_tire_price : ℝ := 5
  let total_cost : ℝ := 185
  (2 * regular_price + third_tire_price = total_cost) →
  regular_price = 90 := by
sorry

end tire_price_proof_l3381_338109


namespace lcm_gcf_problem_l3381_338153

theorem lcm_gcf_problem (n : ℕ+) : 
  Nat.lcm n 14 = 56 → Nat.gcd n 14 = 12 → n = 48 := by
  sorry

end lcm_gcf_problem_l3381_338153


namespace age_ratio_is_two_l3381_338169

/-- The age difference between Yuan and David -/
def age_difference : ℕ := 7

/-- David's age -/
def david_age : ℕ := 7

/-- Yuan's age -/
def yuan_age : ℕ := david_age + age_difference

/-- The ratio of Yuan's age to David's age -/
def age_ratio : ℚ := yuan_age / david_age

theorem age_ratio_is_two : age_ratio = 2 := by
  sorry

end age_ratio_is_two_l3381_338169


namespace factory_work_hours_l3381_338133

/-- Calculates the number of hours a factory works per day given its production rates and total output. -/
theorem factory_work_hours 
  (refrigerators_per_hour : ℕ)
  (extra_coolers : ℕ)
  (total_products : ℕ)
  (days : ℕ)
  (h : refrigerators_per_hour = 90)
  (h' : extra_coolers = 70)
  (h'' : total_products = 11250)
  (h''' : days = 5) :
  (total_products / (days * (refrigerators_per_hour + (refrigerators_per_hour + extra_coolers)))) = 9 :=
by sorry

end factory_work_hours_l3381_338133


namespace chord_midpoint_line_l3381_338156

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem chord_midpoint_line :
  ∀ (A B : ℝ × ℝ),
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 →  -- A and B are on the ellipse
  P = ((A.1 + B.1)/2, (A.2 + B.2)/2) →  -- P is the midpoint of AB
  line_equation A.1 A.2 ∧ line_equation B.1 B.2  -- A and B satisfy the line equation
  := by sorry

end chord_midpoint_line_l3381_338156


namespace sine_product_equality_l3381_338135

theorem sine_product_equality : 
  3.438 * Real.sin (84 * π / 180) * Real.sin (24 * π / 180) * 
  Real.sin (48 * π / 180) * Real.sin (12 * π / 180) = 1/16 := by
  sorry

end sine_product_equality_l3381_338135


namespace set_operations_and_range_l3381_338179

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_range :
  (∃ a : ℝ,
    (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
    ((Set.univ \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) ∧
    (A ∩ C a = A → a ≥ 7)) :=
by sorry

end set_operations_and_range_l3381_338179


namespace sin_150_degrees_l3381_338100

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l3381_338100


namespace tree_ratio_l3381_338177

/-- The number of streets in the neighborhood -/
def num_streets : ℕ := 18

/-- The number of plum trees planted -/
def num_plum_trees : ℕ := 3

/-- The number of pear trees planted -/
def num_pear_trees : ℕ := 3

/-- The number of apricot trees planted -/
def num_apricot_trees : ℕ := 3

/-- Theorem stating that the ratio of plum trees to pear trees to apricot trees is 1:1:1 -/
theorem tree_ratio : 
  num_plum_trees = num_pear_trees ∧ num_pear_trees = num_apricot_trees :=
sorry

end tree_ratio_l3381_338177


namespace owen_daily_chores_hours_l3381_338122

/-- 
Given that:
- There are 24 hours in a day
- Owen spends 6 hours at work
- Owen sleeps for 11 hours

Prove that Owen spends 7 hours on other daily chores.
-/
theorem owen_daily_chores_hours : 
  let total_hours : ℕ := 24
  let work_hours : ℕ := 6
  let sleep_hours : ℕ := 11
  total_hours - work_hours - sleep_hours = 7 := by sorry

end owen_daily_chores_hours_l3381_338122


namespace initial_limes_count_l3381_338112

def limes_given_to_sara : ℕ := 4
def limes_dan_has_now : ℕ := 5

theorem initial_limes_count : 
  limes_given_to_sara + limes_dan_has_now = 9 := by sorry

end initial_limes_count_l3381_338112


namespace diagonal_length_16_12_rectangle_l3381_338180

/-- The length of a diagonal in a 16 cm by 12 cm rectangle is 20 cm -/
theorem diagonal_length_16_12_rectangle : 
  ∀ (a b : ℝ), a = 16 ∧ b = 12 → Real.sqrt (a^2 + b^2) = 20 := by
  sorry

end diagonal_length_16_12_rectangle_l3381_338180


namespace concert_expense_l3381_338137

def ticket_price : ℕ := 6
def tickets_for_friends : ℕ := 8
def extra_tickets : ℕ := 2

theorem concert_expense : 
  ticket_price * (tickets_for_friends + extra_tickets) = 60 := by
  sorry

end concert_expense_l3381_338137


namespace chinese_chess_draw_probability_l3381_338163

theorem chinese_chess_draw_probability 
  (p_xiao_ming_not_lose : ℝ)
  (p_xiao_dong_lose : ℝ)
  (h1 : p_xiao_ming_not_lose = 3/4)
  (h2 : p_xiao_dong_lose = 1/2) :
  p_xiao_ming_not_lose - p_xiao_dong_lose = 1/4 := by
sorry

end chinese_chess_draw_probability_l3381_338163


namespace project_distribution_theorem_l3381_338149

def number_of_arrangements (total_projects : ℕ) 
                            (company_a_projects : ℕ) 
                            (company_b_projects : ℕ) 
                            (company_c_projects : ℕ) 
                            (company_d_projects : ℕ) : ℕ :=
  (Nat.choose total_projects company_a_projects) * 
  (Nat.choose (total_projects - company_a_projects) company_b_projects) * 
  (Nat.choose (total_projects - company_a_projects - company_b_projects) company_c_projects)

theorem project_distribution_theorem :
  number_of_arrangements 8 3 1 2 2 = 1680 := by
  sorry

end project_distribution_theorem_l3381_338149


namespace company_reduction_l3381_338157

/-- The original number of employees before reductions -/
def original_employees : ℕ := 344

/-- The number of employees after both reductions -/
def final_employees : ℕ := 263

/-- The reduction factor after the first quarter -/
def first_reduction : ℚ := 9/10

/-- The reduction factor after the second quarter -/
def second_reduction : ℚ := 85/100

theorem company_reduction :
  ⌊(second_reduction * first_reduction * original_employees : ℚ)⌋ = final_employees := by
  sorry

end company_reduction_l3381_338157


namespace stone_falling_in_water_l3381_338129

/-- Stone falling in water problem -/
theorem stone_falling_in_water
  (stone_density : ℝ)
  (lake_depth : ℝ)
  (gravity : ℝ)
  (water_density : ℝ)
  (h_stone_density : stone_density = 2.1)
  (h_lake_depth : lake_depth = 8.5)
  (h_gravity : gravity = 980.8)
  (h_water_density : water_density = 1.0) :
  ∃ (time velocity : ℝ),
    (abs (time - 1.82) < 0.01) ∧
    (abs (velocity - 935) < 1) ∧
    time = Real.sqrt ((2 * lake_depth * 100) / ((stone_density - water_density) / stone_density * gravity)) ∧
    velocity = ((stone_density - water_density) / stone_density * gravity) * time :=
  sorry


end stone_falling_in_water_l3381_338129


namespace number_exceeds_fraction_l3381_338101

theorem number_exceeds_fraction : 
  ∀ x : ℚ, x = (3/8) * x + 25 → x = 40 := by
  sorry

end number_exceeds_fraction_l3381_338101


namespace milk_left_l3381_338123

theorem milk_left (initial_milk : ℚ) (given_milk : ℚ) (remaining_milk : ℚ) : 
  initial_milk = 5 → given_milk = 18 / 7 → remaining_milk = initial_milk - given_milk → 
  remaining_milk = 17 / 7 := by
  sorry

end milk_left_l3381_338123


namespace divisor_pairs_count_l3381_338183

theorem divisor_pairs_count (n : ℕ) (h : n = 2^6 * 3^3) :
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range (n+1)) (Finset.range (n+1)))).card = 28 :=
by sorry

end divisor_pairs_count_l3381_338183


namespace parabola_tangent_ellipse_l3381_338171

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := 4*x - 4

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

theorem parabola_tangent_ellipse :
  -- Conditions
  (∀ x, parabola x = x^2) →
  (parabola 2 = 4) →
  (tangent_line 2 = 4) →
  (tangent_line 1 = 0) →
  (tangent_line 0 = -4) →
  -- Conclusion
  ellipse (Real.sqrt 17) 4 1 0 ∧ ellipse (Real.sqrt 17) 4 0 (-4) :=
by sorry

end parabola_tangent_ellipse_l3381_338171


namespace min_overlap_percentage_l3381_338182

theorem min_overlap_percentage (laptop_users smartphone_users : ℚ) 
  (h1 : laptop_users = 90/100) 
  (h2 : smartphone_users = 80/100) : 
  (laptop_users + smartphone_users - 1 : ℚ) ≥ 70/100 := by
  sorry

end min_overlap_percentage_l3381_338182


namespace least_k_squared_divisible_by_240_l3381_338142

theorem least_k_squared_divisible_by_240 : 
  ∃ k : ℕ+, k.val = 60 ∧ 
  (∀ m : ℕ+, m.val < k.val → ¬(240 ∣ m.val^2)) ∧
  (240 ∣ k.val^2) := by
  sorry

end least_k_squared_divisible_by_240_l3381_338142


namespace custom_mul_four_three_l3381_338160

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := a^2 + a*b + b^2

/-- Theorem stating that 4 * 3 = 37 under the custom multiplication -/
theorem custom_mul_four_three : custom_mul 4 3 = 37 := by sorry

end custom_mul_four_three_l3381_338160


namespace trip_length_proof_average_efficiency_proof_l3381_338143

/-- The total length of the trip in miles -/
def trip_length : ℝ := 180

/-- The distance the car ran on battery -/
def battery_distance : ℝ := 60

/-- The rate of gasoline consumption in gallons per mile -/
def gasoline_rate : ℝ := 0.03

/-- The average fuel efficiency for the entire trip in miles per gallon -/
def average_efficiency : ℝ := 50

/-- Theorem stating that the trip length satisfies the given conditions -/
theorem trip_length_proof :
  trip_length = battery_distance + 
  (trip_length - battery_distance) * gasoline_rate * average_efficiency :=
by sorry

/-- Theorem stating that the average efficiency is correct -/
theorem average_efficiency_proof :
  average_efficiency = trip_length / (gasoline_rate * (trip_length - battery_distance)) :=
by sorry

end trip_length_proof_average_efficiency_proof_l3381_338143


namespace girls_in_class_l3381_338124

theorem girls_in_class (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 35 →
  boys + girls = total →
  girls = (2 / 5 : ℚ) * boys →
  girls = 10 := by
sorry

end girls_in_class_l3381_338124


namespace cost_per_serving_l3381_338158

/-- The cost per serving of a meal given the costs of ingredients and number of servings -/
theorem cost_per_serving 
  (pasta_cost : ℚ) 
  (sauce_cost : ℚ) 
  (meatballs_cost : ℚ) 
  (num_servings : ℕ) 
  (h1 : pasta_cost = 1)
  (h2 : sauce_cost = 2)
  (h3 : meatballs_cost = 5)
  (h4 : num_servings = 8) :
  (pasta_cost + sauce_cost + meatballs_cost) / num_servings = 1 := by
  sorry

end cost_per_serving_l3381_338158


namespace lebesgue_stieltjes_countable_zero_l3381_338188

-- Define the Lebesgue-Stieltjes measure
def LebesgueStieltjesMeasure (ν : Set ℝ → ℝ) : Prop :=
  -- Add properties of Lebesgue-Stieltjes measure here
  sorry

-- Define continuous generalized distribution function
def ContinuousGeneralizedDistributionFunction (F : ℝ → ℝ) : Prop :=
  -- Add properties of continuous generalized distribution function here
  sorry

-- Define the correspondence between ν and F
def CorrespondsTo (ν : Set ℝ → ℝ) (F : ℝ → ℝ) : Prop :=
  -- Add the correspondence condition here
  sorry

-- Theorem statement
theorem lebesgue_stieltjes_countable_zero 
  (ν : Set ℝ → ℝ) 
  (F : ℝ → ℝ) 
  (A : Set ℝ) :
  LebesgueStieltjesMeasure ν →
  ContinuousGeneralizedDistributionFunction F →
  CorrespondsTo ν F →
  (Set.Countable A ∨ A = ∅) →
  ν A = 0 :=
sorry

end lebesgue_stieltjes_countable_zero_l3381_338188


namespace product_of_roots_l3381_338184

theorem product_of_roots (x : ℝ) (hx : x + 16 / x = 12) : 
  ∃ y : ℝ, y + 16 / y = 12 ∧ x * y = 32 := by
sorry

end product_of_roots_l3381_338184


namespace area_of_triangle_PF₁F₂_l3381_338144

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point on the ellipse
def P : ℝ × ℝ := sorry

-- Assert that P is on the ellipse
axiom P_on_ellipse : is_on_ellipse P.1 P.2

-- Define the distances
def PF₁ : ℝ := sorry
def PF₂ : ℝ := sorry

-- Assert the ratio of distances
axiom distance_ratio : PF₁ / PF₂ = 2

-- Theorem to prove
theorem area_of_triangle_PF₁F₂ : 
  let triangle_area := sorry
  triangle_area = 4 := by sorry

end area_of_triangle_PF₁F₂_l3381_338144


namespace perpendicular_lines_in_parallel_planes_l3381_338106

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (lies_in : Line → Plane → Prop)
variable (not_lies_on : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_in_parallel_planes
  (α β : Plane) (l m : Line)
  (h1 : lies_in l α)
  (h2 : not_lies_on m α)
  (h3 : parallel α β)
  (h4 : perpendicular m β) :
  line_perpendicular l m :=
sorry

end perpendicular_lines_in_parallel_planes_l3381_338106


namespace div_by_eleven_iff_alternating_sum_div_by_eleven_l3381_338192

/-- Calculates the alternating sum of digits of a natural number -/
def alternatingDigitSum (n : ℕ) : ℤ :=
  sorry

/-- Proves the equivalence of divisibility by 11 and divisibility of alternating digit sum by 11 -/
theorem div_by_eleven_iff_alternating_sum_div_by_eleven (n : ℕ) :
  11 ∣ n ↔ 11 ∣ (alternatingDigitSum n) :=
sorry

end div_by_eleven_iff_alternating_sum_div_by_eleven_l3381_338192


namespace ellipse_curve_l3381_338121

-- Define the set of points (x,y) parametrized by t
def ellipse_points : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p.1 = 2 * Real.cos t ∧ p.2 = 3 * Real.sin t}

-- Define the standard form equation of an ellipse
def is_ellipse (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ p ∈ S, (p.1 / a)^2 + (p.2 / b)^2 = 1

-- Theorem statement
theorem ellipse_curve : is_ellipse ellipse_points := by
  sorry

end ellipse_curve_l3381_338121


namespace remaining_grain_l3381_338103

theorem remaining_grain (original : ℕ) (spilled : ℕ) (remaining : ℕ) : 
  original = 50870 → spilled = 49952 → remaining = original - spilled → remaining = 918 := by
  sorry

end remaining_grain_l3381_338103


namespace trig_product_equality_l3381_338119

theorem trig_product_equality : 
  Real.sin (-15 * Real.pi / 6) * Real.cos (20 * Real.pi / 3) * Real.tan (-7 * Real.pi / 6) = Real.sqrt 3 / 6 := by
  sorry

end trig_product_equality_l3381_338119


namespace problem_statement_l3381_338118

theorem problem_statement (x y : ℝ) (h : |x - 3| + (y + 4)^2 = 0) : (x + y)^2023 = -1 := by
  sorry

end problem_statement_l3381_338118


namespace function_determination_l3381_338173

theorem function_determination (f : ℝ → ℝ) 
  (h0 : f 0 = 1) 
  (h1 : ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) : 
  ∀ x : ℝ, f x = x + 1 := by
sorry

end function_determination_l3381_338173
