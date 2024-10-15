import Mathlib

namespace NUMINAMATH_CALUDE_solution_in_interval_monotonic_decreasing_range_two_roots_range_l1270_127082

-- Define the function f(x)
def f (x k : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

-- Theorem 1
theorem solution_in_interval (k : ℝ) :
  ∃ x : ℝ, x ∈ Set.Ioo 0 2 ∧ f x k = k*x + 3 → x = Real.sqrt 2 :=
sorry

-- Theorem 2
theorem monotonic_decreasing_range (k : ℝ) :
  (∀ x y : ℝ, x ∈ Set.Ioo 0 2 → y ∈ Set.Ioo 0 2 → x < y → f x k > f y k) →
  k ∈ Set.Iic (-8) :=
sorry

-- Theorem 3
theorem two_roots_range (k : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Ioo 0 2 ∧ y ∈ Set.Ioo 0 2 ∧ x ≠ y ∧ f x k = 0 ∧ f y k = 0) →
  k ∈ Set.Ioo (-7/2) (-1) :=
sorry

end NUMINAMATH_CALUDE_solution_in_interval_monotonic_decreasing_range_two_roots_range_l1270_127082


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1270_127061

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | (x - 1)^2 < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1270_127061


namespace NUMINAMATH_CALUDE_triangle_area_l1270_127095

/-- The area of a triangle with perimeter 32 and inradius 2.5 is 40 -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
  (h1 : perimeter = 32) 
  (h2 : inradius = 2.5) 
  (h3 : area = inradius * (perimeter / 2)) : 
  area = 40 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1270_127095


namespace NUMINAMATH_CALUDE_school_club_profit_l1270_127065

/-- Calculates the profit for a school club selling granola bars -/
theorem school_club_profit : 
  ∀ (total_bars : ℕ) 
    (buy_price : ℚ) 
    (buy_quantity : ℕ) 
    (sell_price : ℚ) 
    (sell_quantity : ℕ),
  total_bars = 1200 →
  buy_price = 3/2 →
  buy_quantity = 3 →
  sell_price = 12/5 →
  sell_quantity = 4 →
  (total_bars : ℚ) * (sell_price / sell_quantity) - 
  (total_bars : ℚ) * (buy_price / buy_quantity) = 120 := by
sorry


end NUMINAMATH_CALUDE_school_club_profit_l1270_127065


namespace NUMINAMATH_CALUDE_pen_purchase_problem_l1270_127023

/-- The problem of calculating the total number of pens purchased --/
theorem pen_purchase_problem (price_x price_y total_spent : ℚ) (num_x : ℕ) : 
  price_x = 4 → 
  price_y = (14/5 : ℚ) → 
  total_spent = 40 → 
  num_x = 8 → 
  ∃ (num_y : ℕ), num_x * price_x + num_y * price_y = total_spent ∧ num_x + num_y = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_pen_purchase_problem_l1270_127023


namespace NUMINAMATH_CALUDE_y_value_l1270_127043

theorem y_value (x y : ℝ) (h1 : x^2 = y - 7) (h2 : x = 6) : y = 43 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1270_127043


namespace NUMINAMATH_CALUDE_cloth_worth_calculation_l1270_127060

/-- Represents the commission rate as a percentage -/
def commission_rate : ℚ := 25/10

/-- Represents the commission earned in rupees -/
def commission_earned : ℚ := 15

/-- Represents the worth of cloth sold -/
def cloth_worth : ℚ := 600

/-- Theorem stating that given the commission rate and earned commission, 
    the worth of cloth sold is 600 rupees -/
theorem cloth_worth_calculation : 
  commission_earned = (commission_rate / 100) * cloth_worth :=
sorry

end NUMINAMATH_CALUDE_cloth_worth_calculation_l1270_127060


namespace NUMINAMATH_CALUDE_factorization_x3_plus_5x_l1270_127027

theorem factorization_x3_plus_5x (x : ℂ) : x^3 + 5*x = x * (x - Complex.I * Real.sqrt 5) * (x + Complex.I * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x3_plus_5x_l1270_127027


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_four_l1270_127034

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_negative_four :
  (lg 8 + lg 125 - lg 2 - lg 5) / (lg (Real.sqrt 10) * lg 0.1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_four_l1270_127034


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l1270_127005

/-- A function that returns true if a number is a five-digit number -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The greatest five-digit number whose digits have a product of 72 -/
def M : ℕ := sorry

theorem sum_of_digits_M :
  is_five_digit M ∧
  digit_product M = 72 ∧
  (∀ n : ℕ, is_five_digit n → digit_product n = 72 → n ≤ M) →
  digit_sum M = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l1270_127005


namespace NUMINAMATH_CALUDE_sum_sqrt_inequality_l1270_127094

theorem sum_sqrt_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  Real.sqrt (a / (a + 8)) + Real.sqrt (b / (b + 8)) + Real.sqrt (c / (c + 8)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_sqrt_inequality_l1270_127094


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1270_127096

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ
  base : ℕ

/-- Checks if two isosceles triangles are noncongruent -/
def noncongruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.side ≠ t2.side ∨ t1.base ≠ t2.base

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ :=
  2 * t.side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 4 : ℝ) * Real.sqrt (4 * t.side^2 - t.base^2)

/-- Theorem: Minimum perimeter of two specific isosceles triangles -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    noncongruent t1 t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t2.base = 4 * t1.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      noncongruent s1 s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s2.base = 4 * s1.base →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 1180 :=
  sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1270_127096


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1270_127099

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 - 8*x + 12 = 0 ↔ x = 6 ∨ x = 2) ∧
  (∀ x : ℝ, (x - 3)^2 = 2*x*(x - 3) ↔ x = 3 ∨ x = -3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1270_127099


namespace NUMINAMATH_CALUDE_complex_modulus_range_l1270_127078

theorem complex_modulus_range (z : ℂ) (a : ℝ) :
  z = 3 + a * Complex.I ∧ Complex.abs z < 4 →
  a ∈ Set.Ioo (-Real.sqrt 7) (Real.sqrt 7) := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l1270_127078


namespace NUMINAMATH_CALUDE_quadrilateral_equal_sides_is_rhombus_l1270_127068

-- Define a quadrilateral
structure Quadrilateral :=
  (a b c d : ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

-- Theorem: A quadrilateral with all sides equal is a rhombus
theorem quadrilateral_equal_sides_is_rhombus (q : Quadrilateral) :
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d → is_rhombus q :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_equal_sides_is_rhombus_l1270_127068


namespace NUMINAMATH_CALUDE_john_walking_speed_l1270_127045

/-- The walking speed of John in km/h -/
def john_speed : ℝ := 6

/-- The biking speed of Joan in km/h -/
def joan_speed (js : ℝ) : ℝ := 2 * js

/-- The distance between home and school in km -/
def distance : ℝ := 3

/-- The time difference between John's and Joan's departure in hours -/
def time_difference : ℝ := 0.25

theorem john_walking_speed :
  ∃ (js : ℝ),
    js = john_speed ∧
    joan_speed js = 2 * js ∧
    distance / js = distance / (joan_speed js) + time_difference :=
by sorry

end NUMINAMATH_CALUDE_john_walking_speed_l1270_127045


namespace NUMINAMATH_CALUDE_total_amount_is_correct_l1270_127097

def grapes_quantity : ℕ := 10
def grapes_rate : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_rate : ℕ := 55
def apples_quantity : ℕ := 12
def apples_rate : ℕ := 80
def papayas_quantity : ℕ := 7
def papayas_rate : ℕ := 45
def oranges_quantity : ℕ := 15
def oranges_rate : ℕ := 30
def bananas_quantity : ℕ := 5
def bananas_rate : ℕ := 25

def total_amount : ℕ := 
  grapes_quantity * grapes_rate +
  mangoes_quantity * mangoes_rate +
  apples_quantity * apples_rate +
  papayas_quantity * papayas_rate +
  oranges_quantity * oranges_rate +
  bananas_quantity * bananas_rate

theorem total_amount_is_correct : total_amount = 3045 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_correct_l1270_127097


namespace NUMINAMATH_CALUDE_remainder_theorem_l1270_127036

theorem remainder_theorem (x y u v : ℤ) : 
  x > 0 → y > 0 → x = u * y + v → 0 ≤ v → v < y → 
  (x - u * y + 3 * v) % y = 4 * v % y := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1270_127036


namespace NUMINAMATH_CALUDE_betty_calculation_l1270_127092

theorem betty_calculation : ∀ (x y : ℚ),
  x = 8/100 →
  y = 325/100 →
  (x * y : ℚ) = 26/100 :=
by
  sorry

end NUMINAMATH_CALUDE_betty_calculation_l1270_127092


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1270_127041

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 3rd term is 5 and the 6th term is 11, the 9th term is 17. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third : a 3 = 5)
  (h_sixth : a 6 = 11) :
  a 9 = 17 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1270_127041


namespace NUMINAMATH_CALUDE_molecular_weight_3_moles_HBrO3_l1270_127057

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- Atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Number of Hydrogen atoms in HBrO3 -/
def num_H : ℕ := 1

/-- Number of Bromine atoms in HBrO3 -/
def num_Br : ℕ := 1

/-- Number of Oxygen atoms in HBrO3 -/
def num_O : ℕ := 3

/-- Number of moles of HBrO3 -/
def num_moles : ℝ := 3

/-- Calculates the molecular weight of HBrO3 in g/mol -/
def molecular_weight_HBrO3 : ℝ := 
  num_H * atomic_weight_H + num_Br * atomic_weight_Br + num_O * atomic_weight_O

/-- Theorem: The molecular weight of 3 moles of HBrO3 is 386.73 grams -/
theorem molecular_weight_3_moles_HBrO3 : 
  num_moles * molecular_weight_HBrO3 = 386.73 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_3_moles_HBrO3_l1270_127057


namespace NUMINAMATH_CALUDE_area_BPQ_is_six_l1270_127044

/-- Rectangle ABCD with length 8 and width 6, diagonal AC divided into 4 equal segments by P, Q, R -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (diagonal_segments : ℕ)

/-- The area of triangle BPQ in the given rectangle -/
def area_BPQ (rect : Rectangle) : ℝ :=
  sorry

/-- Theorem stating that the area of triangle BPQ is 6 square inches -/
theorem area_BPQ_is_six (rect : Rectangle) 
  (h1 : rect.length = 8)
  (h2 : rect.width = 6)
  (h3 : rect.diagonal_segments = 4) : 
  area_BPQ rect = 6 :=
sorry

end NUMINAMATH_CALUDE_area_BPQ_is_six_l1270_127044


namespace NUMINAMATH_CALUDE_quadratic_roots_l1270_127055

/-- The quadratic equation kx^2 - (2k-3)x + k-2 = 0 -/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 - (2*k - 3) * x + (k - 2) = 0

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ :=
  9 - 4*k

theorem quadratic_roots :
  (∃! x : ℝ, quadratic_equation 0 x) ∧
  (∀ k : ℝ, 0 < k → k ≤ 9/4 → ∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x ∧ quadratic_equation k y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1270_127055


namespace NUMINAMATH_CALUDE_terms_before_zero_l1270_127087

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem terms_before_zero (a : ℤ) (d : ℤ) (h1 : a = 102) (h2 : d = -6) :
  ∃ n : ℕ, n = 17 ∧ arithmetic_sequence a d (n + 1) = 0 :=
sorry

end NUMINAMATH_CALUDE_terms_before_zero_l1270_127087


namespace NUMINAMATH_CALUDE_speeds_satisfy_conditions_l1270_127040

/-- The speed of person A in km/h -/
def speed_A : ℝ := 3.6

/-- The speed of person B in km/h -/
def speed_B : ℝ := 6

/-- The total distance between the starting points of person A and person B in km -/
def total_distance : ℝ := 36

/-- Theorem stating that the given speeds satisfy the conditions of the problem -/
theorem speeds_satisfy_conditions :
  (5 * speed_A + 3 * speed_B = total_distance) ∧
  (2.5 * speed_A + 4.5 * speed_B = total_distance) :=
by sorry

end NUMINAMATH_CALUDE_speeds_satisfy_conditions_l1270_127040


namespace NUMINAMATH_CALUDE_digit_squaring_l1270_127038

theorem digit_squaring (A B C : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ B ≠ C →
  (A + 1 > 1) →
  (A * (A + 1)^3 + A * (A + 1)^2 + A * (A + 1) + A)^2 = 
    A * (A + 1)^7 + A * (A + 1)^6 + A * (A + 1)^5 + B * (A + 1)^4 + C * (A + 1)^3 + C * (A + 1)^2 + C * (A + 1) + B →
  A = 2 ∧ B = 1 ∧ C = 0 := by
sorry

end NUMINAMATH_CALUDE_digit_squaring_l1270_127038


namespace NUMINAMATH_CALUDE_medical_team_selection_l1270_127090

/-- The number of ways to select k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of internists available. -/
def num_internists : ℕ := 5

/-- The number of surgeons available. -/
def num_surgeons : ℕ := 6

/-- The total number of doctors needed for the team. -/
def team_size : ℕ := 4

/-- The number of ways to select the medical team. -/
def select_team : ℕ :=
  choose num_internists 1 * choose num_surgeons 3 +
  choose num_internists 2 * choose num_surgeons 2 +
  choose num_internists 3 * choose num_surgeons 1

theorem medical_team_selection :
  select_team = 310 := by sorry

end NUMINAMATH_CALUDE_medical_team_selection_l1270_127090


namespace NUMINAMATH_CALUDE_total_shells_l1270_127077

/-- The amount of shells in Jovana's bucket -/
def shells_in_bucket (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating the total amount of shells in Jovana's bucket -/
theorem total_shells : shells_in_bucket 5 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_l1270_127077


namespace NUMINAMATH_CALUDE_class_size_problem_l1270_127059

/-- Given information about class sizes, prove the size of Class C -/
theorem class_size_problem (class_b : ℕ) (h1 : class_b = 20) : ∃ (class_c : ℕ), class_c = 170 ∧
  ∃ (class_a class_d : ℕ),
    class_a = 2 * class_b ∧
    3 * class_a = class_c ∧
    class_d = 4 * class_a ∧
    class_c = class_d + 10 := by
  sorry

end NUMINAMATH_CALUDE_class_size_problem_l1270_127059


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1270_127021

theorem unique_triple_solution :
  ∃! (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y = 3 ∧ x * y - z^2 = 4 ∧ x = 1 ∧ y = 2 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1270_127021


namespace NUMINAMATH_CALUDE_quadratic_real_roots_implies_m_leq_3_l1270_127016

theorem quadratic_real_roots_implies_m_leq_3 (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_implies_m_leq_3_l1270_127016


namespace NUMINAMATH_CALUDE_sum_digits_1944_base9_l1270_127086

/-- Converts a natural number from base 10 to base 9 -/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list of natural numbers -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The sum of the digits of 1944 in base 9 is 8 -/
theorem sum_digits_1944_base9 : sumDigits (toBase9 1944) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_1944_base9_l1270_127086


namespace NUMINAMATH_CALUDE_A_intersect_B_l1270_127054

def A : Set ℝ := {-1, 0, 1}

def B : Set ℝ := {y | ∃ x ∈ A, y = Real.exp x}

theorem A_intersect_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1270_127054


namespace NUMINAMATH_CALUDE_expression_equals_one_l1270_127020

theorem expression_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) :
  let x := b^2 + c^2 - b - c + 1 + b*c
  (a^2*b^2) / (x^2) + (a^2*c^2) / (x^2) + (b^2*c^2) / (x^2) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1270_127020


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1270_127079

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - Real.cos α) / (3 * Real.sin α + Real.cos α) = 1/7) : 
  Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1270_127079


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l1270_127012

theorem cyclist_speed_problem (distance : ℝ) (speed_diff : ℝ) (time_diff : ℝ) :
  distance = 195 →
  speed_diff = 4 →
  time_diff = 1 →
  ∃ (v : ℝ),
    v > 0 ∧
    distance / v = distance / (v - speed_diff) - time_diff ∧
    v = 30 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l1270_127012


namespace NUMINAMATH_CALUDE_jellybean_guess_difference_l1270_127062

/-- The jellybean guessing problem -/
theorem jellybean_guess_difference :
  ∀ (guess1 guess2 guess3 guess4 : ℕ),
  guess1 = 100 →
  guess2 = 8 * guess1 →
  guess3 < guess2 →
  guess4 = (guess1 + guess2 + guess3) / 3 + 25 →
  guess4 = 525 →
  guess2 - guess3 = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_jellybean_guess_difference_l1270_127062


namespace NUMINAMATH_CALUDE_guitar_price_theorem_l1270_127032

-- Define the suggested retail price
variable (P : ℝ)

-- Define the prices at Guitar Center and Sweetwater
def guitar_center_price (P : ℝ) : ℝ := 0.85 * P + 100
def sweetwater_price (P : ℝ) : ℝ := 0.90 * P

-- State the theorem
theorem guitar_price_theorem (h : abs (guitar_center_price P - sweetwater_price P) = 50) : 
  P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_guitar_price_theorem_l1270_127032


namespace NUMINAMATH_CALUDE_container_max_volume_l1270_127064

theorem container_max_volume :
  let total_length : ℝ := 24
  let volume (x : ℝ) : ℝ := x^2 * (total_length / 4 - x / 2)
  ∀ x > 0, x < total_length / 4 → volume x ≤ 8 ∧
  ∃ x > 0, x < total_length / 4 ∧ volume x = 8 :=
by sorry

end NUMINAMATH_CALUDE_container_max_volume_l1270_127064


namespace NUMINAMATH_CALUDE_experiments_to_target_reduction_l1270_127014

/-- The factor by which the range is reduced after each experiment -/
def reduction_factor : ℝ := 0.618

/-- The target reduction of the range -/
def target_reduction : ℝ := 0.618^4

/-- The number of experiments needed to reach the target reduction -/
def num_experiments : ℕ := 4

/-- Theorem stating that the number of experiments needed to reach the target reduction is correct -/
theorem experiments_to_target_reduction :
  (reduction_factor ^ num_experiments) = target_reduction :=
by sorry

end NUMINAMATH_CALUDE_experiments_to_target_reduction_l1270_127014


namespace NUMINAMATH_CALUDE_stating_cat_purchase_possible_l1270_127013

/-- Represents the available denominations of rubles --/
def denominations : List ℕ := [1, 5, 10, 50, 100, 500, 1000]

/-- Represents the total amount of money available --/
def total_money : ℕ := 1999

/-- 
Theorem stating that for any price of the cat, 
the buyer can make the purchase and receive correct change
--/
theorem cat_purchase_possible :
  ∀ (price : ℕ), price ≤ total_money →
  ∃ (buyer_money seller_money : List ℕ),
    (buyer_money.sum = price) ∧
    (seller_money.sum = total_money - price) ∧
    (∀ x ∈ buyer_money ∪ seller_money, x ∈ denominations) :=
by sorry

end NUMINAMATH_CALUDE_stating_cat_purchase_possible_l1270_127013


namespace NUMINAMATH_CALUDE_max_quotient_value_l1270_127071

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 1200 ≤ b ∧ b ≤ 2400) :
  (∀ x y, 100 ≤ x ∧ x ≤ 300 → 1200 ≤ y ∧ y ≤ 2400 → y / x ≤ b / a) →
  b / a = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l1270_127071


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l1270_127074

theorem max_students_equal_distribution (pens pencils : ℕ) (h1 : pens = 1048) (h2 : pencils = 828) :
  (Nat.gcd pens pencils : ℕ) = 4 :=
sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l1270_127074


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1270_127024

theorem circle_equation_proof (x y : ℝ) : 
  let equation := x^2 + y^2 - 10*y
  let center := (0, 5)
  let radius := 5
  -- The circle's equation
  (equation = 0) →
  -- Center is on the y-axis
  (center.1 = 0) ∧
  -- Circle is tangent to x-axis (distance from center to x-axis equals radius)
  (center.2 = radius) ∧
  -- Circle passes through (3, 1)
  ((3 - center.1)^2 + (1 - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1270_127024


namespace NUMINAMATH_CALUDE_arianna_work_hours_l1270_127025

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents the number of hours Arianna spends on chores -/
def hours_on_chores : ℕ := 5

/-- Represents the number of hours Arianna spends sleeping -/
def hours_sleeping : ℕ := 13

/-- Theorem stating that Arianna spends 6 hours at work -/
theorem arianna_work_hours :
  hours_in_day - (hours_on_chores + hours_sleeping) = 6 := by
  sorry

end NUMINAMATH_CALUDE_arianna_work_hours_l1270_127025


namespace NUMINAMATH_CALUDE_cubic_fraction_simplification_l1270_127075

theorem cubic_fraction_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a + b + 2 * c = 0) : 
  (a^3 + b^3 + 2 * c^3) / (a * b * c) = -3 * (a^2 - a * b + b^2) / (2 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_simplification_l1270_127075


namespace NUMINAMATH_CALUDE_circles_have_three_common_tangents_l1270_127048

/-- Circle C₁ with equation x² + y² + 2x + 4y + 1 = 0 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0

/-- Circle C₂ with equation x² + y² - 4x - 4y - 1 = 0 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 1 = 0

/-- The number of common tangents between C₁ and C₂ -/
def num_common_tangents : ℕ := 3

theorem circles_have_three_common_tangents :
  ∃ (n : ℕ), n = num_common_tangents ∧ 
  (∀ (x y : ℝ), C₁ x y ∨ C₂ x y → n = 3) :=
sorry

end NUMINAMATH_CALUDE_circles_have_three_common_tangents_l1270_127048


namespace NUMINAMATH_CALUDE_solution_set_l1270_127089

-- Define the variables
variable (a b x : ℝ)

-- Define the conditions
def condition1 : Prop := -Real.sqrt (1 / ((a - b)^2)) * (b - a) = 1
def condition2 : Prop := 3*x - 4*a ≤ a - 2*x
def condition3 : Prop := (3*x + 2*b) / 5 > b

-- State the theorem
theorem solution_set (h1 : condition1 a b) (h2 : condition2 a x) (h3 : condition3 b x) :
  b < x ∧ x ≤ a :=
sorry

end NUMINAMATH_CALUDE_solution_set_l1270_127089


namespace NUMINAMATH_CALUDE_chipped_marbles_are_36_l1270_127053

def marble_bags : List Nat := [16, 18, 22, 24, 26, 30, 36]

structure MarbleDistribution where
  jane_bags : List Nat
  george_bags : List Nat
  chipped_bag : Nat

def is_valid_distribution (d : MarbleDistribution) : Prop :=
  d.jane_bags.length = 4 ∧
  d.george_bags.length = 2 ∧
  d.chipped_bag ∈ marble_bags ∧
  d.jane_bags.sum = 3 * d.george_bags.sum ∧
  (∀ b ∈ d.jane_bags ++ d.george_bags, b ≠ d.chipped_bag) ∧
  (∀ b ∈ marble_bags, b ∉ d.jane_bags → b ∉ d.george_bags → b = d.chipped_bag)

theorem chipped_marbles_are_36 :
  ∀ d : MarbleDistribution, is_valid_distribution d → d.chipped_bag = 36 := by
  sorry

end NUMINAMATH_CALUDE_chipped_marbles_are_36_l1270_127053


namespace NUMINAMATH_CALUDE_product_of_divisors_1024_l1270_127083

/-- The product of divisors of a positive integer -/
def product_of_divisors (n : ℕ+) : ℕ := sorry

/-- Theorem: If the product of divisors of n is 1024, then n = 16 -/
theorem product_of_divisors_1024 (n : ℕ+) :
  product_of_divisors n = 1024 → n = 16 := by sorry

end NUMINAMATH_CALUDE_product_of_divisors_1024_l1270_127083


namespace NUMINAMATH_CALUDE_sum_of_digits_of_3_to_17_l1270_127072

/-- The sum of the tens digit and the ones digit of (7-4)^17 is 9. -/
theorem sum_of_digits_of_3_to_17 : 
  (((7 - 4)^17 / 10) % 10 + (7 - 4)^17 % 10) = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_3_to_17_l1270_127072


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1270_127042

theorem complex_equation_solution (z : ℂ) (h : (1 + Complex.I) * z = 2) : z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1270_127042


namespace NUMINAMATH_CALUDE_quadratic_equation_root_zero_l1270_127063

theorem quadratic_equation_root_zero (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 + x + a^2 - 1 = 0) ∧
  ((a - 1) * 0^2 + 0 + a^2 - 1 = 0) ∧
  (a - 1 ≠ 0) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_zero_l1270_127063


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l1270_127052

theorem least_three_digit_multiple_of_eight : 
  (∀ n : ℕ, 100 ≤ n ∧ n < 104 → ¬(n % 8 = 0)) ∧ 
  104 % 8 = 0 ∧ 
  104 ≥ 100 ∧ 
  104 < 1000 := by
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l1270_127052


namespace NUMINAMATH_CALUDE_juan_number_puzzle_l1270_127047

theorem juan_number_puzzle (n : ℝ) : ((n + 3) * 3 - 3) * 2 / 3 = 10 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_juan_number_puzzle_l1270_127047


namespace NUMINAMATH_CALUDE_carly_swimming_time_l1270_127076

/-- Calculates the total swimming practice time in a month -/
def monthly_swimming_time (butterfly_hours_per_day : ℕ) (butterfly_days_per_week : ℕ)
                          (backstroke_hours_per_day : ℕ) (backstroke_days_per_week : ℕ)
                          (weeks_in_month : ℕ) : ℕ :=
  ((butterfly_hours_per_day * butterfly_days_per_week) +
   (backstroke_hours_per_day * backstroke_days_per_week)) * weeks_in_month

/-- Proves that Carly spends 96 hours practicing swimming in a month -/
theorem carly_swimming_time :
  monthly_swimming_time 3 4 2 6 4 = 96 :=
by sorry

end NUMINAMATH_CALUDE_carly_swimming_time_l1270_127076


namespace NUMINAMATH_CALUDE_hair_cut_first_day_l1270_127098

/-- The amount of hair cut off on the first day, given the total amount cut off and the amount cut off on the second day. -/
theorem hair_cut_first_day (total : ℚ) (second_day : ℚ) (h1 : total = 0.875) (h2 : second_day = 0.5) :
  total - second_day = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_first_day_l1270_127098


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1270_127085

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) := True

-- Define the properties of the triangle
def TriangleProperties (A B C : ℝ × ℝ) :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  BC = AC - 1 ∧ AC = AB - 1 ∧ (AB^2 + AC^2 - BC^2) / (2 * AB * AC) = 3/5

-- Theorem statement
theorem triangle_perimeter (A B C : ℝ × ℝ) 
  (h : Triangle A B C) 
  (hp : TriangleProperties A B C) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB + BC + AC = 42 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1270_127085


namespace NUMINAMATH_CALUDE_samantha_birthday_next_monday_l1270_127007

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Determines if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Calculates the day of the week for June 18 in a given year, 
    given the day of the week for June 18 in the previous year -/
def nextJune18 (prevDay : DayOfWeek) (year : Nat) : DayOfWeek :=
  sorry

/-- Finds the next year when June 18 falls on a Monday, given a starting year and day -/
def nextMondayJune18 (startYear : Nat) (startDay : DayOfWeek) : Nat :=
  sorry

theorem samantha_birthday_next_monday (startYear : Nat) (startDay : DayOfWeek) :
  startYear = 2009 →
  startDay = DayOfWeek.Friday →
  ¬isLeapYear startYear →
  nextMondayJune18 startYear startDay = 2017 :=
sorry

end NUMINAMATH_CALUDE_samantha_birthday_next_monday_l1270_127007


namespace NUMINAMATH_CALUDE_picture_books_count_l1270_127006

theorem picture_books_count (total : ℕ) (fiction : ℕ) (non_fiction : ℕ) (autobiographies : ℕ) (picture : ℕ) : 
  total = 35 →
  fiction = 5 →
  non_fiction = fiction + 4 →
  autobiographies = 2 * fiction →
  total = fiction + non_fiction + autobiographies + picture →
  picture = 11 := by
sorry

end NUMINAMATH_CALUDE_picture_books_count_l1270_127006


namespace NUMINAMATH_CALUDE_sin_300_degrees_l1270_127009

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l1270_127009


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_explicit_l1270_127088

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 3)⁻¹ + (b + 3)⁻¹ = 1/4) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (x + 3)⁻¹ + (y + 3)⁻¹ = 1/4 → a + 3*b ≤ x + 3*y :=
by sorry

theorem min_value_explicit (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 3)⁻¹ + (b + 3)⁻¹ = 1/4) : 
  a + 3*b = 4 + 8*Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_explicit_l1270_127088


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l1270_127015

theorem unique_solution_power_equation :
  ∃! (n k l m : ℕ), l > 1 ∧ (1 + n^k)^l = 1 + n^m ∧ n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l1270_127015


namespace NUMINAMATH_CALUDE_base8_to_base7_conversion_l1270_127031

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- The given number in base 8 -/
def givenNumber : ℕ := 653

/-- The expected result in base 7 -/
def expectedResult : ℕ := 1150

theorem base8_to_base7_conversion :
  base10ToBase7 (base8ToBase10 givenNumber) = expectedResult := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base7_conversion_l1270_127031


namespace NUMINAMATH_CALUDE_money_distribution_l1270_127070

/-- Given that p, q, and r have $9000 among themselves, and r has two-thirds of the total amount with p and q, prove that r has $3600. -/
theorem money_distribution (p q r : ℝ) 
  (total : p + q + r = 9000)
  (r_proportion : r = (2/3) * (p + q)) :
  r = 3600 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l1270_127070


namespace NUMINAMATH_CALUDE_train_crossing_contradiction_l1270_127091

theorem train_crossing_contradiction (V₁ V₂ L₁ L₂ T₂ : ℝ) : 
  V₁ > 0 → V₂ > 0 → L₁ > 0 → L₂ > 0 → T₂ > 0 →
  (L₁ / V₁ = 20) →  -- First train crosses man in 20 seconds
  (L₂ / V₂ = T₂) →  -- Second train crosses man in T₂ seconds
  ((L₁ + L₂) / (V₁ + V₂) = 19) →  -- Trains cross each other in 19 seconds
  (V₁ = V₂) →  -- Ratio of speeds is 1
  False :=  -- This leads to a contradiction
by
  sorry

#check train_crossing_contradiction

end NUMINAMATH_CALUDE_train_crossing_contradiction_l1270_127091


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_conditions_l1270_127028

def is_prime (n : ℕ) : Prop := sorry

def is_consecutive (a b : ℕ) : Prop := b = a + 1 ∨ a = b + 1

theorem least_integer_with_divisibility_conditions (N : ℕ) : 
  (∀ k ∈ Finset.range 31, k ≠ 0 → ∃ (a b : ℕ), a ≠ b ∧ is_consecutive a b ∧ 
    (is_prime a ∨ is_prime b) ∧ 
    (∀ i ∈ Finset.range 31, i ≠ 0 ∧ i ≠ a ∧ i ≠ b → N % i = 0) ∧
    N % a ≠ 0 ∧ N % b ≠ 0) →
  N ≥ 8923714800 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_divisibility_conditions_l1270_127028


namespace NUMINAMATH_CALUDE_problem_statement_l1270_127093

theorem problem_statement (x y : ℝ) 
  (h : |x - 1/2| + Real.sqrt (y^2 - 1) = 0) :
  |x| + |y| = 3/2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1270_127093


namespace NUMINAMATH_CALUDE_rs_equals_240_l1270_127073

-- Define the triangle DEF
structure Triangle (DE EF : ℝ) where
  de_positive : DE > 0
  ef_positive : EF > 0

-- Define points Q, R, S, N
structure Points (D E F Q R S N : ℝ × ℝ) where
  q_on_de : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • D + t • E
  r_on_df : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (1 - t) • D + t • F
  s_on_fq : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (1 - t) • F + t • Q
  s_on_er : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (1 - t) • E + t • R
  n_on_fq : ∃ t : ℝ, t > 1 ∧ N = (1 - t) • F + t • Q

-- Define the conditions
def Conditions (D E F Q R S N : ℝ × ℝ) (triangle : Triangle 600 400) (points : Points D E F Q R S N) : Prop :=
  let de := ‖E - D‖
  let dq := ‖Q - D‖
  let qe := ‖E - Q‖
  let dn := ‖N - D‖
  let sn := ‖N - S‖
  let sq := ‖Q - S‖
  de = 600 ∧ dq = qe ∧ dn = 240 ∧ sn = sq

-- Theorem statement
theorem rs_equals_240 (D E F Q R S N : ℝ × ℝ) 
  (triangle : Triangle 600 400) (points : Points D E F Q R S N) 
  (h : Conditions D E F Q R S N triangle points) : 
  ‖R - S‖ = 240 := by sorry

end NUMINAMATH_CALUDE_rs_equals_240_l1270_127073


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l1270_127084

theorem probability_at_least_one_correct (n : ℕ) (choices : ℕ) : 
  n = 6 → choices = 6 → 
  1 - (1 - 1 / choices : ℚ) ^ n = 31031 / 46656 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l1270_127084


namespace NUMINAMATH_CALUDE_seokjin_position_relative_to_jungkook_l1270_127056

/-- Given the positions of Jungkook, Yoojeong, and Seokjin on a staircase,
    prove that Seokjin stands 3 steps above Jungkook. -/
theorem seokjin_position_relative_to_jungkook 
  (jungkook_stair : ℕ) 
  (yoojeong_above_jungkook : ℕ) 
  (seokjin_below_yoojeong : ℕ) 
  (h1 : jungkook_stair = 19)
  (h2 : yoojeong_above_jungkook = 8)
  (h3 : seokjin_below_yoojeong = 5) :
  (jungkook_stair + yoojeong_above_jungkook - seokjin_below_yoojeong) - jungkook_stair = 3 :=
by sorry

end NUMINAMATH_CALUDE_seokjin_position_relative_to_jungkook_l1270_127056


namespace NUMINAMATH_CALUDE_sum_75_odd_numbers_l1270_127080

-- Define a function for the sum of first n odd numbers
def sum_odd_numbers (n : ℕ) : ℕ := n^2

-- State the theorem
theorem sum_75_odd_numbers :
  (sum_odd_numbers 50 = 2500) → (sum_odd_numbers 75 = 5625) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_75_odd_numbers_l1270_127080


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1270_127030

theorem rationalize_denominator : 
  1 / (Real.sqrt 3 - 2) = -Real.sqrt 3 - 2 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1270_127030


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1270_127029

theorem absolute_value_inequality (x : ℝ) : 3 ≤ |x + 2| ∧ |x + 2| ≤ 7 ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1270_127029


namespace NUMINAMATH_CALUDE_triangle_height_l1270_127022

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) :
  area = 576 →
  base = 32 →
  area = (base * height) / 2 →
  height = 36 := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l1270_127022


namespace NUMINAMATH_CALUDE_function_inequality_relation_l1270_127035

theorem function_inequality_relation (f : ℝ → ℝ) (a b : ℝ) :
  (f = λ x => 3 * x + 1) →
  (a > 0 ∧ b > 0) →
  (∀ x, |x - 1| < b → |f x - 4| < a) →
  a - 3 * b ≥ 0 := by sorry

end NUMINAMATH_CALUDE_function_inequality_relation_l1270_127035


namespace NUMINAMATH_CALUDE_x_value_l1270_127039

theorem x_value : ∃ x : ℝ, (0.25 * x = 0.1 * 500 - 5) ∧ (x = 180) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1270_127039


namespace NUMINAMATH_CALUDE_digital_earth_capabilities_l1270_127004

-- Define the capabilities of Digital Earth
def can_simulate_environmental_impact : Prop := True
def can_monitor_crop_pests : Prop := True
def can_predict_submerged_areas : Prop := True
def can_simulate_past_environments : Prop := True

-- Define the statement to be proven false
def incorrect_statement : Prop :=
  ∃ (can_predict_future : Prop),
    can_predict_future ∧ ¬can_simulate_past_environments

-- Theorem statement
theorem digital_earth_capabilities :
  can_simulate_environmental_impact →
  can_monitor_crop_pests →
  can_predict_submerged_areas →
  can_simulate_past_environments →
  ¬incorrect_statement :=
by
  sorry

end NUMINAMATH_CALUDE_digital_earth_capabilities_l1270_127004


namespace NUMINAMATH_CALUDE_parallel_vectors_iff_m_values_l1270_127011

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- The vector a as a function of m -/
def a (m : ℝ) : ℝ × ℝ := (2*m + 1, 3)

/-- The vector b as a function of m -/
def b (m : ℝ) : ℝ × ℝ := (2, m)

/-- Theorem stating that vectors a and b are parallel if and only if m = 3/2 or m = -2 -/
theorem parallel_vectors_iff_m_values :
  ∀ m : ℝ, are_parallel (a m) (b m) ↔ m = 3/2 ∨ m = -2 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_iff_m_values_l1270_127011


namespace NUMINAMATH_CALUDE_set_relationship_l1270_127018

/-- Definition of set E -/
def E : Set ℚ := { e | ∃ m : ℤ, e = m + 1/6 }

/-- Definition of set F -/
def F : Set ℚ := { f | ∃ n : ℤ, f = n/2 - 1/3 }

/-- Definition of set G -/
def G : Set ℚ := { g | ∃ p : ℤ, g = p/2 + 1/6 }

/-- Theorem stating the relationship among sets E, F, and G -/
theorem set_relationship : E ⊆ F ∧ F = G := by
  sorry

end NUMINAMATH_CALUDE_set_relationship_l1270_127018


namespace NUMINAMATH_CALUDE_equation_solutions_l1270_127069

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = 5/2 ∧ x₂ = 4 ∧ 
  (∀ x : ℝ, 3*(2*x - 5) = (2*x - 5)^2 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1270_127069


namespace NUMINAMATH_CALUDE_problem_solution_l1270_127033

theorem problem_solution (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 4)^5 + (Real.log y / Real.log 5)^5 + 10 = 
       10 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x^2 + y^2 = 4^(2*5^(1/5)) + 5^(2*5^(1/5)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1270_127033


namespace NUMINAMATH_CALUDE_min_tiles_to_pave_courtyard_l1270_127046

def courtyard_length : ℕ := 378
def courtyard_width : ℕ := 525

def tile_side_length : ℕ := Nat.gcd courtyard_length courtyard_width

def courtyard_area : ℕ := courtyard_length * courtyard_width
def tile_area : ℕ := tile_side_length * tile_side_length

def number_of_tiles : ℕ := courtyard_area / tile_area

theorem min_tiles_to_pave_courtyard :
  number_of_tiles = 450 := by sorry

end NUMINAMATH_CALUDE_min_tiles_to_pave_courtyard_l1270_127046


namespace NUMINAMATH_CALUDE_first_divisor_of_square_plus_164_l1270_127003

theorem first_divisor_of_square_plus_164 : 
  ∀ n ∈ [3, 4, 5, 6, 7, 8, 9, 10, 11], 
    (n ∣ (166^2 + 164)) → 
    n = 3 := by sorry

end NUMINAMATH_CALUDE_first_divisor_of_square_plus_164_l1270_127003


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l1270_127058

theorem perfect_square_polynomial (k : ℝ) : 
  (∀ a : ℝ, ∃ b : ℝ, a^2 + 2*k*a + 1 = b^2) → (k = 1 ∨ k = -1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l1270_127058


namespace NUMINAMATH_CALUDE_power_equality_l1270_127049

theorem power_equality (x y : ℕ) (h1 : x - y = 12) (h2 : x = 12) :
  3^x * 4^y = 531441 := by
sorry

end NUMINAMATH_CALUDE_power_equality_l1270_127049


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1270_127000

theorem arithmetic_mean_of_fractions : 
  (3 / 8 + 5 / 12) / 2 = 19 / 48 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1270_127000


namespace NUMINAMATH_CALUDE_total_spent_is_correct_l1270_127051

def original_cost : ℝ := 1200
def discount_rate : ℝ := 0.20
def tax_rate : ℝ := 0.08

def total_spent : ℝ :=
  let discounted_cost := original_cost * (1 - discount_rate)
  let other_toys_with_tax := discounted_cost * (1 + tax_rate)
  let lightsaber_cost := 2 * original_cost
  let lightsaber_with_tax := lightsaber_cost * (1 + tax_rate)
  other_toys_with_tax + lightsaber_with_tax

theorem total_spent_is_correct :
  total_spent = 3628.80 := by sorry

end NUMINAMATH_CALUDE_total_spent_is_correct_l1270_127051


namespace NUMINAMATH_CALUDE_comparison_inequality_l1270_127010

theorem comparison_inequality : ∀ x : ℝ, (x - 2) * (x + 3) > x^2 + x - 7 := by
  sorry

end NUMINAMATH_CALUDE_comparison_inequality_l1270_127010


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l1270_127067

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def base7Number : List Nat := [4, 6, 5, 7, 3]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 9895 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l1270_127067


namespace NUMINAMATH_CALUDE_polynomial_invariant_under_increment_l1270_127008

def P (x : ℝ) : ℝ := x^3 - 5*x^2 + 8*x

theorem polynomial_invariant_under_increment :
  ∀ x : ℝ, P x = P (x + 1) ↔ x = 1 ∨ x = 4/3 := by sorry

end NUMINAMATH_CALUDE_polynomial_invariant_under_increment_l1270_127008


namespace NUMINAMATH_CALUDE_equation_is_linear_l1270_127017

/-- A linear equation in two variables has the form ax + by = c, where a, b, and c are constants, and x and y are variables. -/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The equation 4x - y = 3 -/
def equation (x y : ℝ) : ℝ := 4 * x - y - 3

theorem equation_is_linear : IsLinearEquationInTwoVariables equation := by
  sorry


end NUMINAMATH_CALUDE_equation_is_linear_l1270_127017


namespace NUMINAMATH_CALUDE_parabola_directrix_l1270_127019

/-- Definition of a parabola with equation y^2 = 6x -/
def parabola (x y : ℝ) : Prop := y^2 = 6*x

/-- Definition of the directrix of a parabola -/
def directrix (x : ℝ) : Prop := x = -3/2

/-- Theorem: The directrix of the parabola y^2 = 6x is x = -3/2 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → directrix x :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1270_127019


namespace NUMINAMATH_CALUDE_blue_marble_probability_l1270_127002

theorem blue_marble_probability : 
  ∀ (total yellow green red blue : ℕ),
    total = 60 →
    yellow = 20 →
    green = yellow / 2 →
    red = blue →
    total = yellow + green + red + blue →
    (blue : ℚ) / total * 100 = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_marble_probability_l1270_127002


namespace NUMINAMATH_CALUDE_selling_price_after_markup_and_discount_l1270_127037

/-- The selling price of a commodity after markup and discount -/
theorem selling_price_after_markup_and_discount (a : ℝ) : 
  let markup_rate : ℝ := 0.5
  let discount_rate : ℝ := 0.3
  let marked_price : ℝ := a * (1 + markup_rate)
  let final_price : ℝ := marked_price * (1 - discount_rate)
  final_price = 1.05 * a :=
by sorry

end NUMINAMATH_CALUDE_selling_price_after_markup_and_discount_l1270_127037


namespace NUMINAMATH_CALUDE_max_individual_points_is_23_l1270_127050

/-- Represents a basketball team -/
structure BasketballTeam where
  players : Nat
  totalPoints : Nat
  minPointsPerPlayer : Nat

/-- Calculates the maximum points a single player could have scored -/
def maxIndividualPoints (team : BasketballTeam) : Nat :=
  team.totalPoints - (team.players - 1) * team.minPointsPerPlayer

/-- Theorem: The maximum points an individual player could have scored is 23 -/
theorem max_individual_points_is_23 (team : BasketballTeam) 
  (h1 : team.players = 12)
  (h2 : team.totalPoints = 100)
  (h3 : team.minPointsPerPlayer = 7) :
  maxIndividualPoints team = 23 := by
  sorry

#eval maxIndividualPoints ⟨12, 100, 7⟩

end NUMINAMATH_CALUDE_max_individual_points_is_23_l1270_127050


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1270_127066

theorem inequality_equivalence (x : ℝ) :
  (1 / Real.sqrt (1 - x) - 1 / Real.sqrt (1 + x) ≥ 1) ↔ 
  (Real.sqrt (2 * Real.sqrt 3 - 3) ≤ x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1270_127066


namespace NUMINAMATH_CALUDE_min_value_of_expression_existence_of_minimum_l1270_127001

theorem min_value_of_expression (x : ℝ) : 
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2040 ≥ 2039 :=
sorry

theorem existence_of_minimum : 
  ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2040 = 2039 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_existence_of_minimum_l1270_127001


namespace NUMINAMATH_CALUDE_total_sheets_is_114_l1270_127026

/-- The number of bundles of colored paper -/
def coloredBundles : ℕ := 3

/-- The number of bunches of white paper -/
def whiteBunches : ℕ := 2

/-- The number of heaps of scrap paper -/
def scrapHeaps : ℕ := 5

/-- The number of sheets in a bunch -/
def sheetsPerBunch : ℕ := 4

/-- The number of sheets in a bundle -/
def sheetsPerBundle : ℕ := 2

/-- The number of sheets in a heap -/
def sheetsPerHeap : ℕ := 20

/-- The total number of sheets of paper removed from the chest of drawers -/
def totalSheets : ℕ := coloredBundles * sheetsPerBundle + whiteBunches * sheetsPerBunch + scrapHeaps * sheetsPerHeap

theorem total_sheets_is_114 : totalSheets = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_sheets_is_114_l1270_127026


namespace NUMINAMATH_CALUDE_max_take_home_pay_l1270_127081

-- Define the income function
def income (y : ℝ) : ℝ := 100 * y^2

-- Define the tax function
def tax (y : ℝ) : ℝ := y^3

-- Define the take-home pay function
def takeHomePay (y : ℝ) : ℝ := income y - tax y

-- Theorem statement
theorem max_take_home_pay :
  ∃ y : ℝ, y > 0 ∧ 
    (∀ z : ℝ, z > 0 → takeHomePay z ≤ takeHomePay y) ∧
    income y = 250000 := by sorry

end NUMINAMATH_CALUDE_max_take_home_pay_l1270_127081
