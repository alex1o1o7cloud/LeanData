import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_angle_chord_length_l1670_167004

/-- Given a circle with radius R and an inscribed angle α that subtends a chord of length a,
    prove that a = 2R sin α. -/
theorem inscribed_angle_chord_length (R : ℝ) (α : ℝ) (a : ℝ) 
    (h_circle : R > 0) 
    (h_angle : 0 < α ∧ α < π) 
    (h_chord : a > 0) : 
  a = 2 * R * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_inscribed_angle_chord_length_l1670_167004


namespace NUMINAMATH_CALUDE_second_number_calculation_l1670_167008

theorem second_number_calculation (x y z : ℚ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 5 / 8) :
  y = 2400 / 67 := by
sorry

end NUMINAMATH_CALUDE_second_number_calculation_l1670_167008


namespace NUMINAMATH_CALUDE_field_area_l1670_167027

/-- A rectangular field with specific properties -/
structure RectangularField where
  breadth : ℝ
  length : ℝ
  length_relation : length = breadth + 30
  perimeter : ℝ
  perimeter_formula : perimeter = 2 * (length + breadth)
  perimeter_value : perimeter = 540

/-- The area of the rectangular field is 18000 square metres -/
theorem field_area (field : RectangularField) : field.length * field.breadth = 18000 := by
  sorry

end NUMINAMATH_CALUDE_field_area_l1670_167027


namespace NUMINAMATH_CALUDE_eldoria_license_plates_l1670_167058

/-- The number of vowels available for the first letter of a license plate. -/
def numVowels : ℕ := 5

/-- The number of letters in the alphabet. -/
def numLetters : ℕ := 26

/-- The number of digits (0-9). -/
def numDigits : ℕ := 10

/-- The number of characters in a valid license plate. -/
def licensePlateLength : ℕ := 5

/-- Calculates the number of valid license plates in Eldoria. -/
def numValidLicensePlates : ℕ :=
  numVowels * numLetters * numDigits * numDigits * numDigits

/-- Theorem stating the number of valid license plates in Eldoria. -/
theorem eldoria_license_plates :
  numValidLicensePlates = 130000 := by
  sorry

end NUMINAMATH_CALUDE_eldoria_license_plates_l1670_167058


namespace NUMINAMATH_CALUDE_no_tetrahedron_with_heights_1_2_3_6_l1670_167025

/-- Represents a tetrahedron with face heights -/
structure Tetrahedron where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ

/-- The theorem stating that a tetrahedron with heights 1, 2, 3, and 6 cannot exist -/
theorem no_tetrahedron_with_heights_1_2_3_6 :
  ¬ ∃ (t : Tetrahedron), t.h₁ = 1 ∧ t.h₂ = 2 ∧ t.h₃ = 3 ∧ t.h₄ = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_tetrahedron_with_heights_1_2_3_6_l1670_167025


namespace NUMINAMATH_CALUDE_board_numbers_l1670_167021

theorem board_numbers (a b : ℕ) (h1 : a > b) (h2 : a = 1580) :
  (((a - b) : ℚ) / (2^10 : ℚ)).isInt → b = 556 := by
  sorry

end NUMINAMATH_CALUDE_board_numbers_l1670_167021


namespace NUMINAMATH_CALUDE_gertrude_has_ten_fleas_l1670_167023

/-- The number of fleas on Gertrude's chicken -/
def gertrude_fleas : ℕ := sorry

/-- The number of fleas on Maud's chicken -/
def maud_fleas : ℕ := sorry

/-- The number of fleas on Olive's chicken -/
def olive_fleas : ℕ := sorry

/-- Maud has 5 times the amount of fleas as Olive -/
axiom maud_olive_relation : maud_fleas = 5 * olive_fleas

/-- Olive has half the amount of fleas as Gertrude -/
axiom olive_gertrude_relation : olive_fleas * 2 = gertrude_fleas

/-- The total number of fleas is 40 -/
axiom total_fleas : gertrude_fleas + maud_fleas + olive_fleas = 40

/-- Theorem: Gertrude has 10 fleas -/
theorem gertrude_has_ten_fleas : gertrude_fleas = 10 := by sorry

end NUMINAMATH_CALUDE_gertrude_has_ten_fleas_l1670_167023


namespace NUMINAMATH_CALUDE_remainder_problem_l1670_167016

theorem remainder_problem (N : ℕ) : 
  (∃ R, N = 7 * 5 + R ∧ R < 7) → 
  (∃ Q, N = 11 * Q + 2) → 
  N % 7 = 4 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l1670_167016


namespace NUMINAMATH_CALUDE_no_multiple_of_four_l1670_167017

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_form_1C34 (n : ℕ) : Prop :=
  ∃ C : ℕ, C < 10 ∧ n = 1000 + 100 * C + 34

theorem no_multiple_of_four :
  ¬∃ n : ℕ, is_four_digit n ∧ has_form_1C34 n ∧ 4 ∣ n :=
sorry

end NUMINAMATH_CALUDE_no_multiple_of_four_l1670_167017


namespace NUMINAMATH_CALUDE_waiting_time_is_correct_l1670_167078

/-- The total waiting time in minutes for Mark's vaccine appointments -/
def total_waiting_time : ℕ :=
  let days_first_vaccine := 4
  let days_second_vaccine := 20
  let days_first_secondary := 30 + 10  -- 1 month and 10 days
  let days_second_secondary := 14 + 3  -- 2 weeks and 3 days
  let days_full_effectiveness := 3 * 7 -- 3 weeks
  let total_days := days_first_vaccine + days_second_vaccine + days_first_secondary +
                    days_second_secondary + days_full_effectiveness
  let minutes_per_day := 24 * 60
  total_days * minutes_per_day

/-- Theorem stating that the total waiting time is 146,880 minutes -/
theorem waiting_time_is_correct : total_waiting_time = 146880 := by
  sorry

end NUMINAMATH_CALUDE_waiting_time_is_correct_l1670_167078


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1670_167018

/-- A regular polygon with an exterior angle of 12 degrees has 30 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 12 → (n : ℝ) * exterior_angle = 360 → n = 30 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1670_167018


namespace NUMINAMATH_CALUDE_garrison_size_l1670_167044

/-- The number of men initially in the garrison -/
def initial_men : ℕ := 2000

/-- The number of days the initial provisions would last -/
def initial_days : ℕ := 54

/-- The number of days after which reinforcements arrive -/
def days_before_reinforcement : ℕ := 21

/-- The number of men that arrive as reinforcement -/
def reinforcement : ℕ := 1300

/-- The number of days the provisions last after reinforcement -/
def remaining_days : ℕ := 20

theorem garrison_size :
  initial_men * initial_days = 
  (initial_men + reinforcement) * remaining_days + 
  initial_men * days_before_reinforcement := by
  sorry

end NUMINAMATH_CALUDE_garrison_size_l1670_167044


namespace NUMINAMATH_CALUDE_crop_planting_problem_l1670_167049

/-- Cost function for planting crops -/
def cost_function (x : ℝ) : ℝ := x^2 + 5*x + 10

/-- Revenue function for planting crops -/
def revenue_function (x : ℝ) : ℝ := 15*x

/-- Profit function for planting crops -/
def profit_function (x : ℝ) : ℝ := revenue_function x - cost_function x

theorem crop_planting_problem :
  (cost_function 1 = 16 ∧ cost_function 3 = 34) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ cost_function x₁ / x₁ = 12 ∧ cost_function x₂ / x₂ = 12) ∧
  (∃ x_max : ℝ, x_max = 5 ∧ 
    ∀ x : ℝ, profit_function x ≤ profit_function x_max ∧ 
    profit_function x_max = 15) :=
by sorry

#check crop_planting_problem

end NUMINAMATH_CALUDE_crop_planting_problem_l1670_167049


namespace NUMINAMATH_CALUDE_fifty_seventh_pair_l1670_167032

def pair_sequence : ℕ → ℕ × ℕ
| n => sorry

theorem fifty_seventh_pair :
  pair_sequence 57 = (2, 10) := by sorry

end NUMINAMATH_CALUDE_fifty_seventh_pair_l1670_167032


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l1670_167094

theorem sqrt_sum_comparison : Real.sqrt 2 + Real.sqrt 10 < 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l1670_167094


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l1670_167092

theorem smallest_n_for_candy_purchase : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 → (20 * m) % 12 = 0 ∧ (20 * m) % 14 = 0 ∧ (20 * m) % 15 = 0 → m ≥ n) ∧
  (20 * n) % 12 = 0 ∧ (20 * n) % 14 = 0 ∧ (20 * n) % 15 = 0 ∧ n = 21 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l1670_167092


namespace NUMINAMATH_CALUDE_power_of_three_mod_eight_l1670_167006

theorem power_of_three_mod_eight : 3^1234 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eight_l1670_167006


namespace NUMINAMATH_CALUDE_min_value_smallest_at_a_l1670_167020

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

/-- Theorem stating that the minimum value of f(x) is smallest when a = 82/43 -/
theorem min_value_smallest_at_a (a : ℝ) :
  (∀ x : ℝ, f (82/43) x ≤ f a x) :=
sorry

end NUMINAMATH_CALUDE_min_value_smallest_at_a_l1670_167020


namespace NUMINAMATH_CALUDE_kyle_gas_and_maintenance_amount_l1670_167011

/-- Calculates the amount left for gas and maintenance given Kyle's income and expenses --/
def amount_for_gas_and_maintenance (monthly_income : ℝ) (rent : ℝ) (utilities : ℝ) 
  (retirement_savings : ℝ) (groceries : ℝ) (insurance : ℝ) (miscellaneous : ℝ) 
  (car_payment : ℝ) : ℝ :=
  monthly_income - (rent + utilities + retirement_savings + groceries + insurance + miscellaneous + car_payment)

/-- Theorem stating that Kyle's amount left for gas and maintenance is $350 --/
theorem kyle_gas_and_maintenance_amount :
  amount_for_gas_and_maintenance 3200 1250 150 400 300 200 200 350 = 350 := by
  sorry

end NUMINAMATH_CALUDE_kyle_gas_and_maintenance_amount_l1670_167011


namespace NUMINAMATH_CALUDE_boys_average_weight_l1670_167030

/-- Proves that given a group of 10 students with 5 girls and 5 boys, where the average weight of
    the girls is 45 kg and the average weight of all students is 50 kg, then the average weight
    of the boys is 55 kg. -/
theorem boys_average_weight 
  (num_students : Nat) 
  (num_girls : Nat) 
  (num_boys : Nat) 
  (girls_avg_weight : ℝ) 
  (total_avg_weight : ℝ) : ℝ :=
by
  have h1 : num_students = 10 := by sorry
  have h2 : num_girls = 5 := by sorry
  have h3 : num_boys = 5 := by sorry
  have h4 : girls_avg_weight = 45 := by sorry
  have h5 : total_avg_weight = 50 := by sorry

  -- The average weight of the boys
  let boys_avg_weight : ℝ := 55

  -- Proof that boys_avg_weight = 55
  sorry

end NUMINAMATH_CALUDE_boys_average_weight_l1670_167030


namespace NUMINAMATH_CALUDE_number_ordering_l1670_167000

theorem number_ordering (a b c : ℝ) (ha : a = (-0.3)^0) (hb : b = 0.32) (hc : c = 20.3) :
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_number_ordering_l1670_167000


namespace NUMINAMATH_CALUDE_myPolygonArea_l1670_167095

/-- A point in a 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A polygon defined by a list of points -/
def Polygon := List Point

/-- The polygon in question -/
def myPolygon : Polygon := [
  {x := 0, y := 0},
  {x := 0, y := 30},
  {x := 30, y := 30},
  {x := 30, y := 0}
]

/-- Calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℤ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the area of myPolygon is 15 square units -/
theorem myPolygonArea : calculateArea myPolygon = 15 := by
  sorry

end NUMINAMATH_CALUDE_myPolygonArea_l1670_167095


namespace NUMINAMATH_CALUDE_reading_materials_cost_l1670_167098

/-- The total cost of purchasing reading materials -/
def total_cost (a b : ℕ) : ℕ := 10 * a + 8 * b

/-- Theorem: The total cost of purchasing 'a' copies of type A reading materials
    at 10 yuan per copy and 'b' copies of type B reading materials at 8 yuan
    per copy is equal to 10a + 8b yuan. -/
theorem reading_materials_cost (a b : ℕ) :
  total_cost a b = 10 * a + 8 * b := by
  sorry

end NUMINAMATH_CALUDE_reading_materials_cost_l1670_167098


namespace NUMINAMATH_CALUDE_doll_count_sum_l1670_167040

/-- The number of dolls each person has -/
structure DollCounts where
  vera : ℕ
  lisa : ℕ
  sophie : ℕ
  aida : ℕ

/-- The conditions of the doll counting problem -/
def doll_problem (d : DollCounts) : Prop :=
  d.aida = 3 * d.sophie ∧
  d.sophie = 2 * d.vera ∧
  d.vera = d.lisa / 3 ∧
  d.lisa = d.vera + 10 ∧
  d.vera = 15

theorem doll_count_sum (d : DollCounts) : 
  doll_problem d → d.aida + d.sophie + d.vera + d.lisa = 160 := by
  sorry

end NUMINAMATH_CALUDE_doll_count_sum_l1670_167040


namespace NUMINAMATH_CALUDE_sum_product_nonpositive_l1670_167096

theorem sum_product_nonpositive (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_nonpositive_l1670_167096


namespace NUMINAMATH_CALUDE_triangle_area_is_15_5_l1670_167052

/-- Triangle ABC inscribed in a rectangle --/
structure TriangleInRectangle where
  -- Rectangle dimensions
  width : ℝ
  height : ℝ
  -- Vertex positions
  a_height : ℝ
  b_distance : ℝ
  c_distance : ℝ
  -- Conditions
  width_positive : width > 0
  height_positive : height > 0
  a_height_valid : 0 < a_height ∧ a_height < height
  b_distance_valid : 0 < b_distance ∧ b_distance < width
  c_distance_valid : 0 < c_distance ∧ c_distance < height

/-- The area of triangle ABC --/
def triangleArea (t : TriangleInRectangle) : ℝ :=
  t.width * t.height - (0.5 * t.width * t.c_distance + 0.5 * (t.height - t.a_height) * t.width + 0.5 * t.b_distance * t.a_height)

/-- Theorem: The area of triangle ABC is 15.5 square units --/
theorem triangle_area_is_15_5 (t : TriangleInRectangle) 
    (h_width : t.width = 6)
    (h_height : t.height = 4)
    (h_a_height : t.a_height = 1)
    (h_b_distance : t.b_distance = 3)
    (h_c_distance : t.c_distance = 1) : 
  triangleArea t = 15.5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_is_15_5_l1670_167052


namespace NUMINAMATH_CALUDE_book_sale_result_l1670_167029

/-- Represents the book sale scenario -/
structure BookSale where
  initial_fiction : ℕ
  initial_nonfiction : ℕ
  fiction_sold : ℕ
  fiction_remaining : ℕ
  total_earnings : ℕ
  fiction_price : ℕ
  nonfiction_price : ℕ

/-- Theorem stating the results of the book sale -/
theorem book_sale_result (sale : BookSale)
  (h1 : sale.fiction_sold = 137)
  (h2 : sale.fiction_remaining = 105)
  (h3 : sale.total_earnings = 685)
  (h4 : sale.fiction_price = 3)
  (h5 : sale.nonfiction_price = 5)
  (h6 : sale.initial_fiction = sale.fiction_sold + sale.fiction_remaining) :
  sale.initial_fiction = 242 ∧
  (sale.total_earnings - sale.fiction_sold * sale.fiction_price) / sale.nonfiction_price = 54 := by
  sorry


end NUMINAMATH_CALUDE_book_sale_result_l1670_167029


namespace NUMINAMATH_CALUDE_function_positivity_implies_m_range_l1670_167080

theorem function_positivity_implies_m_range 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (m : ℝ) 
  (h_f : ∀ x, f x = 2 * m * x^2 - 2 * (4 - m) * x + 1) 
  (h_g : ∀ x, g x = m * x) 
  (h_pos : ∀ x, f x > 0 ∨ g x > 0) : 
  0 < m ∧ m < 8 := by
sorry

end NUMINAMATH_CALUDE_function_positivity_implies_m_range_l1670_167080


namespace NUMINAMATH_CALUDE_range_of_a_l1670_167028

-- Define the conditions
def condition_p (x : ℝ) : Prop := x^2 + x - 2 > 0
def condition_q (x a : ℝ) : Prop := x > a

-- Define the sufficient but not necessary relationship
def sufficient_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬(q x)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, 
    (sufficient_not_necessary (condition_p) (condition_q a)) → 
    a ≥ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1670_167028


namespace NUMINAMATH_CALUDE_special_triangle_angles_l1670_167047

/-- A triangle with excircle radii and circumradius satisfying certain conditions -/
structure SpecialTriangle where
  /-- Excircle radius opposite to side a -/
  r_a : ℝ
  /-- Excircle radius opposite to side b -/
  r_b : ℝ
  /-- Excircle radius opposite to side c -/
  r_c : ℝ
  /-- Circumradius of the triangle -/
  R : ℝ
  /-- First condition: r_a + r_b = 3R -/
  cond1 : r_a + r_b = 3 * R
  /-- Second condition: r_b + r_c = 2R -/
  cond2 : r_b + r_c = 2 * R

/-- The angles of a SpecialTriangle are 30°, 60°, and 90° -/
theorem special_triangle_angles (t : SpecialTriangle) :
  ∃ (A B C : Real),
    A = 30 * π / 180 ∧
    B = 60 * π / 180 ∧
    C = 90 * π / 180 ∧
    A + B + C = π :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_angles_l1670_167047


namespace NUMINAMATH_CALUDE_total_expenditure_nine_persons_l1670_167051

/-- Given 9 persons, where 8 spend 30 Rs each and the 9th spends 20 Rs more than the average,
    prove that the total expenditure is 292.5 Rs -/
theorem total_expenditure_nine_persons :
  let num_persons : ℕ := 9
  let num_regular_spenders : ℕ := 8
  let regular_expenditure : ℚ := 30
  let extra_expenditure : ℚ := 20
  let total_expenditure : ℚ := num_regular_spenders * regular_expenditure +
    (((num_regular_spenders * regular_expenditure) / num_persons) + extra_expenditure)
  total_expenditure = 292.5 := by
sorry

end NUMINAMATH_CALUDE_total_expenditure_nine_persons_l1670_167051


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1670_167042

theorem complex_equation_solution (z : ℂ) :
  (Complex.I - 1) * z = 2 → z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1670_167042


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1670_167072

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1670_167072


namespace NUMINAMATH_CALUDE_equidistant_points_on_line_in_quadrants_I_II_l1670_167012

/-- A point (x, y) is in the first quadrant if both x and y are positive -/
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- A point (x, y) is in the second quadrant if x is negative and y is positive -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- A point (x, y) is equidistant from the coordinate axes if |x| = |y| -/
def equidistant_from_axes (x y : ℝ) : Prop := abs x = abs y

/-- A point (x, y) is on the line 4x + 6y = 24 -/
def on_line (x y : ℝ) : Prop := 4*x + 6*y = 24

theorem equidistant_points_on_line_in_quadrants_I_II :
  ∃ x y : ℝ, on_line x y ∧ equidistant_from_axes x y ∧ (in_first_quadrant x y ∨ in_second_quadrant x y) ∧
  ∀ x' y' : ℝ, on_line x' y' ∧ equidistant_from_axes x' y' → (in_first_quadrant x' y' ∨ in_second_quadrant x' y') :=
sorry

end NUMINAMATH_CALUDE_equidistant_points_on_line_in_quadrants_I_II_l1670_167012


namespace NUMINAMATH_CALUDE_line_vector_proof_l1670_167093

def line_vector (t : ℝ) : ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 0 = (2, 3)) →
  (line_vector 5 = (12, -37)) →
  (line_vector (-3) = (-4, 27)) :=
by sorry

end NUMINAMATH_CALUDE_line_vector_proof_l1670_167093


namespace NUMINAMATH_CALUDE_incorrect_statements_l1670_167059

theorem incorrect_statements : 
  let statement1 := (∃ a b : ℚ, a + b = 5 ∧ a + b = -3)
  let statement2 := (∀ x : ℝ, ∃ q : ℚ, x = q)
  let statement3 := (∀ x : ℝ, |x| > 0)
  let statement4 := (∀ x : ℝ, x * x = x → (x = 0 ∨ x = 1))
  let statement5 := (∀ a b : ℚ, a + b = 0 → (a > 0 ∨ b > 0))
  (¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4 ∧ ¬statement5) := by sorry

end NUMINAMATH_CALUDE_incorrect_statements_l1670_167059


namespace NUMINAMATH_CALUDE_line_equations_correct_l1670_167041

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define a line by its slope and a point it passes through
structure Line1 where
  slope : ℝ
  point : Point

-- Define a line by two points it passes through
structure Line2 where
  point1 : Point
  point2 : Point

-- Function to get the equation of a line given slope and point
def lineEquation1 (l : Line1) : ℝ → ℝ → Prop :=
  fun x y => y - l.point.2 = l.slope * (x - l.point.1)

-- Function to get the equation of a line given two points
def lineEquation2 (l : Line2) : ℝ → ℝ → Prop :=
  fun x y => (y - l.point1.2) * (l.point2.1 - l.point1.1) = 
             (l.point2.2 - l.point1.2) * (x - l.point1.1)

theorem line_equations_correct :
  let line1 := Line1.mk (-1/2) (8, -2)
  let line2 := Line2.mk (3, -2) (5, -4)
  (∀ x y, lineEquation1 line1 x y ↔ x + 2*y - 4 = 0) ∧
  (∀ x y, lineEquation2 line2 x y ↔ x + y - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_equations_correct_l1670_167041


namespace NUMINAMATH_CALUDE_income_comparison_l1670_167007

theorem income_comparison (juan tim mart : ℝ) 
  (h1 : mart = 1.6 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mart = 0.96 * juan := by
  sorry

end NUMINAMATH_CALUDE_income_comparison_l1670_167007


namespace NUMINAMATH_CALUDE_cousin_distribution_count_l1670_167019

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 cousins -/
def num_cousins : ℕ := 5

/-- There are 5 rooms -/
def num_rooms : ℕ := 5

/-- The number of ways to distribute the cousins among the rooms -/
def num_distributions : ℕ := distribute num_cousins num_rooms

theorem cousin_distribution_count : num_distributions = 137 := by sorry

end NUMINAMATH_CALUDE_cousin_distribution_count_l1670_167019


namespace NUMINAMATH_CALUDE_star_equation_has_two_distinct_real_roots_l1670_167050

-- Define the ☆ operation
def star (m n : ℝ) : ℝ := m * n^2 - m * n - 1

-- Theorem statement
theorem star_equation_has_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ star 1 x₁ = 0 ∧ star 1 x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_star_equation_has_two_distinct_real_roots_l1670_167050


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1670_167086

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/8 = 1

-- Define a square inscribed in the ellipse
structure InscribedSquare where
  side : ℝ
  vertex_on_ellipse : ellipse (side/2) (side/2)

-- Theorem statement
theorem inscribed_square_area (s : InscribedSquare) : s.side^2 = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1670_167086


namespace NUMINAMATH_CALUDE_stating_fencers_count_correct_l1670_167090

/-- The number of fencers participating in the championship. -/
def num_fencers : ℕ := 9

/-- The number of possibilities for awarding first and second place medals. -/
def num_possibilities : ℕ := 72

/-- 
Theorem stating that the number of fencers is correct given the number of possibilities 
for awarding first and second place medals.
-/
theorem fencers_count_correct : 
  num_fencers * (num_fencers - 1) = num_possibilities := by
  sorry

#check fencers_count_correct

end NUMINAMATH_CALUDE_stating_fencers_count_correct_l1670_167090


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l1670_167071

/-- Represents a 3x3 grid of natural numbers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Checks if two positions in the grid are adjacent -/
def isAdjacent (x1 y1 x2 y2 : Fin 3) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y2 = y1 + 1)) ∨
  (y1 = y2 ∧ (x1 = x2 + 1 ∨ x2 = x1 + 1))

/-- Checks if a grid arrangement is valid according to the problem conditions -/
def isValidArrangement (g : Grid) : Prop :=
  (∀ x y : Fin 3, g x y ∈ Finset.range 9) ∧
  (∀ x1 y1 x2 y2 : Fin 3, isAdjacent x1 y1 x2 y2 → isPrime (g x1 y1 + g x2 y2)) ∧
  (∀ n : Fin 9, ∃ x y : Fin 3, g x y = n + 1)

/-- The main theorem stating that no valid arrangement exists -/
theorem no_valid_arrangement : ¬∃ g : Grid, isValidArrangement g := by
  sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l1670_167071


namespace NUMINAMATH_CALUDE_min_value_expression_l1670_167010

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  2 * x^2 + 8 * x * y + 6 * y^2 + 16 * y * z + 3 * z^2 ≥ 24 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧
    2 * a^2 + 8 * a * b + 6 * b^2 + 16 * b * c + 3 * c^2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1670_167010


namespace NUMINAMATH_CALUDE_parking_lot_levels_l1670_167038

/-- Represents a multi-story parking lot -/
structure ParkingLot where
  totalCapacity : ℕ
  levelCapacity : ℕ
  additionalCars : ℕ
  initialCars : ℕ

/-- Calculates the number of levels in the parking lot -/
def ParkingLot.levels (p : ParkingLot) : ℕ :=
  p.totalCapacity / p.levelCapacity

/-- Theorem: The specific parking lot has 5 levels -/
theorem parking_lot_levels :
  ∀ (p : ParkingLot),
    p.totalCapacity = 425 →
    p.additionalCars = 62 →
    p.initialCars = 23 →
    p.levelCapacity = p.additionalCars + p.initialCars →
    p.levels = 5 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_levels_l1670_167038


namespace NUMINAMATH_CALUDE_meal_combinations_l1670_167054

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The number of dishes Camille avoids -/
def avoided_dishes : ℕ := 2

/-- The number of dishes Camille can choose from -/
def camille_choices : ℕ := menu_items - avoided_dishes

theorem meal_combinations : menu_items * camille_choices = 195 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_l1670_167054


namespace NUMINAMATH_CALUDE_election_votes_l1670_167003

theorem election_votes (total_votes : ℕ) 
  (h1 : (70 : ℚ) / 100 * total_votes - (30 : ℚ) / 100 * total_votes = 182) : 
  total_votes = 455 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l1670_167003


namespace NUMINAMATH_CALUDE_max_value_theorem_l1670_167053

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  2 * x^2 * y * Real.sqrt 6 + 8 * y^2 * z ≤ Real.sqrt (144/35) + Real.sqrt (88/35) := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1670_167053


namespace NUMINAMATH_CALUDE_cube_root_abs_square_sum_l1670_167091

theorem cube_root_abs_square_sum : ∃ (x : ℝ), x^3 = -8 ∧ x + |(-6)| - 2^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_abs_square_sum_l1670_167091


namespace NUMINAMATH_CALUDE_triangle_inequality_l1670_167070

/-- Given a triangle with circumradius R, inradius r, side lengths a, b, c, and semiperimeter p,
    prove that 20Rr - 4r^2 ≤ ab + bc + ca ≤ 4(R + r)^2 -/
theorem triangle_inequality (R r a b c p : ℝ) (hR : R > 0) (hr : r > 0)
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hp : p = (a + b + c) / 2)
    (hcirc : R = a * b * c / (4 * p * r)) (hinr : r = p * (p - a) * (p - b) * (p - c) / (a * b * c)) :
    20 * R * r - 4 * r^2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 4 * (R + r)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1670_167070


namespace NUMINAMATH_CALUDE_pyramid_volume_approx_l1670_167009

-- Define the pyramid
structure Pyramid where
  baseArea : ℝ
  face1Area : ℝ
  face2Area : ℝ

-- Define the volume function
def pyramidVolume (p : Pyramid) : ℝ :=
  sorry

-- Theorem statement
theorem pyramid_volume_approx (p : Pyramid) 
  (h1 : p.baseArea = 256)
  (h2 : p.face1Area = 120)
  (h3 : p.face2Area = 104) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |pyramidVolume p - 1163| < ε :=
sorry

end NUMINAMATH_CALUDE_pyramid_volume_approx_l1670_167009


namespace NUMINAMATH_CALUDE_email_sending_combinations_l1670_167099

theorem email_sending_combinations (num_addresses : ℕ) (num_emails : ℕ) : 
  num_addresses = 3 → num_emails = 5 → num_addresses ^ num_emails = 243 :=
by sorry

end NUMINAMATH_CALUDE_email_sending_combinations_l1670_167099


namespace NUMINAMATH_CALUDE_base_number_proof_l1670_167001

theorem base_number_proof (n : ℕ) (x : ℕ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = x^22)
  (h2 : n = 21) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l1670_167001


namespace NUMINAMATH_CALUDE_correct_balance_amount_l1670_167022

/-- The amount Carlos must give LeRoy to balance their adjusted shares -/
def balance_amount (A B C : ℝ) : ℝ := 0.35 * A - 0.65 * B + 0.35 * C

/-- Theorem stating the correct amount Carlos must give LeRoy -/
theorem correct_balance_amount (A B C : ℝ) (hB_lt_A : B < A) (hB_lt_C : B < C) :
  balance_amount A B C = (0.35 * (A + B + C) - B) := by sorry

end NUMINAMATH_CALUDE_correct_balance_amount_l1670_167022


namespace NUMINAMATH_CALUDE_sphere_quarter_sphere_radius_l1670_167085

theorem sphere_quarter_sphere_radius (r : ℝ) (h : r = 4 * Real.rpow 2 (1/3)) :
  ∃ R : ℝ, (4/3 * Real.pi * R^3 = 1/3 * Real.pi * r^3) ∧ R = 2 * Real.rpow 2 (1/3) :=
sorry

end NUMINAMATH_CALUDE_sphere_quarter_sphere_radius_l1670_167085


namespace NUMINAMATH_CALUDE_product_equals_fraction_l1670_167082

/-- The repeating decimal 0.456̄ -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of 0.456̄ and 7 -/
def product : ℚ := repeating_decimal * 7

/-- Theorem stating that the product of 0.456̄ and 7 is equal to 1064/333 -/
theorem product_equals_fraction : product = 1064 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l1670_167082


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1670_167031

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 999 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1670_167031


namespace NUMINAMATH_CALUDE_early_winner_emerges_l1670_167002

/-- The number of participants in the tournament -/
def n : ℕ := 10

/-- The number of matches each participant plays -/
def matches_per_participant : ℕ := n - 1

/-- The total number of matches in the tournament -/
def total_matches : ℕ := n * matches_per_participant / 2

/-- The number of matches per round -/
def matches_per_round : ℕ := n / 2

/-- The maximum points a participant can score in one round -/
def max_points_per_round : ℚ := 1

/-- The minimum number of rounds required for an early winner to emerge -/
def min_rounds_for_winner : ℕ := 7

theorem early_winner_emerges (
  winner_points : ℚ → ℚ → Prop) 
  (other_max_points : ℚ → ℚ → Prop) : 
  (∀ r : ℕ, r < min_rounds_for_winner → 
    ¬(winner_points r > other_max_points r)) ∧
  (winner_points min_rounds_for_winner > 
    other_max_points min_rounds_for_winner) := by
  sorry

end NUMINAMATH_CALUDE_early_winner_emerges_l1670_167002


namespace NUMINAMATH_CALUDE_overall_gain_calculation_l1670_167035

def flat1_purchase : ℝ := 675958
def flat1_gain_percent : ℝ := 0.14

def flat2_purchase : ℝ := 848592
def flat2_loss_percent : ℝ := 0.10

def flat3_purchase : ℝ := 940600
def flat3_gain_percent : ℝ := 0.07

def calculate_selling_price (purchase : ℝ) (gain_percent : ℝ) : ℝ :=
  purchase * (1 + gain_percent)

theorem overall_gain_calculation :
  let flat1_selling := calculate_selling_price flat1_purchase flat1_gain_percent
  let flat2_selling := calculate_selling_price flat2_purchase (-flat2_loss_percent)
  let flat3_selling := calculate_selling_price flat3_purchase flat3_gain_percent
  let total_purchase := flat1_purchase + flat2_purchase + flat3_purchase
  let total_selling := flat1_selling + flat2_selling + flat3_selling
  total_selling - total_purchase = 75617.92 := by
  sorry

end NUMINAMATH_CALUDE_overall_gain_calculation_l1670_167035


namespace NUMINAMATH_CALUDE_product_increase_l1670_167063

theorem product_increase (a b : ℕ) (h1 : a * b = 72) : a * (10 * b) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_increase_l1670_167063


namespace NUMINAMATH_CALUDE_practice_time_is_three_l1670_167065

/-- Calculates the practice time per minute of singing given the performance duration,
    tantrum time per minute of singing, and total time. -/
def practice_time_per_minute (performance_duration : ℕ) (tantrum_time_per_minute : ℕ) (total_time : ℕ) : ℕ :=
  ((total_time - performance_duration) / performance_duration) - tantrum_time_per_minute

/-- Proves that given a 6-minute performance, 5 minutes of tantrums per minute of singing,
    and a total time of 54 minutes, the practice time per minute of singing is 3 minutes. -/
theorem practice_time_is_three :
  practice_time_per_minute 6 5 54 = 3 := by
  sorry

#eval practice_time_per_minute 6 5 54

end NUMINAMATH_CALUDE_practice_time_is_three_l1670_167065


namespace NUMINAMATH_CALUDE_converse_proposition_l1670_167048

theorem converse_proposition (a : ℝ) : a > 2 → a^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_converse_proposition_l1670_167048


namespace NUMINAMATH_CALUDE_bucket_capacity_l1670_167036

theorem bucket_capacity (tank_capacity : ℝ) (first_scenario_buckets : ℕ) (second_scenario_buckets : ℕ) (second_scenario_capacity : ℝ) :
  first_scenario_buckets = 30 →
  second_scenario_buckets = 45 →
  second_scenario_capacity = 9 →
  tank_capacity = first_scenario_buckets * (tank_capacity / first_scenario_buckets) →
  tank_capacity = second_scenario_buckets * second_scenario_capacity →
  tank_capacity / first_scenario_buckets = 13.5 := by
sorry

end NUMINAMATH_CALUDE_bucket_capacity_l1670_167036


namespace NUMINAMATH_CALUDE_munchausen_polygon_theorem_l1670_167043

/-- A polygon in 2D space -/
structure Polygon :=
  (vertices : Set (ℝ × ℝ))

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- A line in 2D space -/
structure Line :=
  (a b c : ℝ)

/-- Predicate to check if a point is inside a polygon -/
def is_inside (p : Point) (poly : Polygon) : Prop :=
  sorry

/-- Predicate to check if a line intersects a polygon at exactly two points -/
def intersects_at_two_points (l : Line) (poly : Polygon) : Prop :=
  sorry

/-- Predicate to check if a line passes through a point -/
def passes_through (l : Line) (p : Point) : Prop :=
  sorry

/-- Predicate to check if a line divides a polygon into three smaller polygons -/
def divides_into_three (l : Line) (poly : Polygon) : Prop :=
  sorry

/-- Theorem stating that there exists a polygon and a point inside it
    such that any line passing through this point divides the polygon into three smaller polygons -/
theorem munchausen_polygon_theorem :
  ∃ (poly : Polygon) (p : Point),
    is_inside p poly ∧
    ∀ (l : Line),
      passes_through l p →
      intersects_at_two_points l poly ∧
      divides_into_three l poly :=
sorry

end NUMINAMATH_CALUDE_munchausen_polygon_theorem_l1670_167043


namespace NUMINAMATH_CALUDE_range_of_a_l1670_167056

def p (x : ℝ) : Prop := (4 * x - 3)^2 ≤ 1

def q (a x : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

theorem range_of_a : 
  (∀ a : ℝ, (∀ x : ℝ, p x → q a x) ∧ 
  (∃ x : ℝ, ¬p x ∧ q a x)) → 
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1/2) ∧ 
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1/2 → 
    (∀ x : ℝ, p x → q a x) ∧ 
    (∃ x : ℝ, ¬p x ∧ q a x)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1670_167056


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l1670_167013

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l1670_167013


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l1670_167005

theorem unique_root_quadratic (X Y Z : ℝ) (hX : X ≠ 0) (hY : Y ≠ 0) (hZ : Z ≠ 0) :
  (∀ t : ℝ, X * t^2 - Y * t + Z = 0 ↔ t = Y) → X = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l1670_167005


namespace NUMINAMATH_CALUDE_divisor_problem_l1670_167015

theorem divisor_problem (D : ℚ) : D ≠ 0 → (72 / D + 5 = 17) → D = 6 := by sorry

end NUMINAMATH_CALUDE_divisor_problem_l1670_167015


namespace NUMINAMATH_CALUDE_product_of_fractions_l1670_167097

theorem product_of_fractions : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1670_167097


namespace NUMINAMATH_CALUDE_x_intercepts_count_l1670_167034

/-- The number of x-intercepts of y = sin(1/x) in the interval (0.00005, 0.0005) -/
theorem x_intercepts_count : 
  (⌊(20000 : ℝ) / Real.pi⌋ - ⌊(2000 : ℝ) / Real.pi⌋ : ℤ) = 5729 := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l1670_167034


namespace NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_line_l1670_167062

/-- The equation of a line passing through the center of a circle and parallel to another line -/
theorem line_through_circle_center_parallel_to_line :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 2*x + 2*y = 0
  let parallel_line_eq : ℝ → ℝ → Prop := λ x y => 2*x - y = 0
  let result_line_eq : ℝ → ℝ → Prop := λ x y => 2*x - y - 3 = 0
  ∃ (center_x center_y : ℝ),
    (∀ x y, circle_eq x y ↔ (x - center_x)^2 + (y - center_y)^2 = (center_x^2 + center_y^2)) →
    (∀ x y, result_line_eq x y ↔ y - center_y = 2 * (x - center_x)) →
    (∀ x₁ y₁ x₂ y₂, parallel_line_eq x₁ y₁ ∧ parallel_line_eq x₂ y₂ → y₂ - y₁ = 2 * (x₂ - x₁)) →
    result_line_eq center_x center_y :=
by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_line_l1670_167062


namespace NUMINAMATH_CALUDE_perimeter_of_specific_hexagon_l1670_167073

-- Define the hexagon ABCDEF
structure RightAngledHexagon where
  AB : ℝ
  BC : ℝ
  EF : ℝ

-- Define the perimeter function
def perimeter (h : RightAngledHexagon) : ℝ :=
  2 * (h.AB + h.EF) + 2 * h.BC

-- Theorem statement
theorem perimeter_of_specific_hexagon :
  ∃ (h : RightAngledHexagon), h.AB = 8 ∧ h.BC = 15 ∧ h.EF = 5 ∧ perimeter h = 56 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_specific_hexagon_l1670_167073


namespace NUMINAMATH_CALUDE_cos_40_plus_sqrt3_tan_10_eq_1_l1670_167079

theorem cos_40_plus_sqrt3_tan_10_eq_1 : 
  Real.cos (40 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_40_plus_sqrt3_tan_10_eq_1_l1670_167079


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1670_167039

theorem linear_equation_solution (a : ℝ) : 
  (∃ x y : ℝ, x = 2 ∧ y = 3 ∧ a * x - 3 * y = 3) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1670_167039


namespace NUMINAMATH_CALUDE_number_of_boys_l1670_167060

theorem number_of_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 100 →
  boys + girls = total →
  girls = boys * total / 100 →
  boys = 50 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l1670_167060


namespace NUMINAMATH_CALUDE_corners_removed_cube_edges_l1670_167075

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Represents the solid formed by removing smaller cubes from corners of a larger cube -/
structure CornersRemovedCube where
  originalCube : Cube
  removedCubeSideLength : ℝ
  removedCubeSideLength_pos : removedCubeSideLength > 0
  validRemoval : removedCubeSideLength < originalCube.sideLength / 2

/-- Calculates the number of edges in the resulting solid after removing smaller cubes from corners -/
def edgesAfterRemoval (c : CornersRemovedCube) : ℕ :=
  sorry

/-- Theorem stating that removing cubes of side length 2 from corners of a cube with side length 6 results in a solid with 36 edges -/
theorem corners_removed_cube_edges :
  let originalCube : Cube := ⟨6, by norm_num⟩
  let cornersRemovedCube : CornersRemovedCube := ⟨originalCube, 2, by norm_num, by norm_num⟩
  edgesAfterRemoval cornersRemovedCube = 36 :=
sorry

end NUMINAMATH_CALUDE_corners_removed_cube_edges_l1670_167075


namespace NUMINAMATH_CALUDE_complex_point_location_l1670_167026

theorem complex_point_location (x y : ℝ) (h : x / (1 + Complex.I) = 1 - y * Complex.I) :
  x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_location_l1670_167026


namespace NUMINAMATH_CALUDE_five_digit_palindromes_count_l1670_167055

/-- A function that counts the number of 5-digit palindromes -/
def count_five_digit_palindromes : ℕ :=
  9 * 10 * 10

/-- Theorem stating that the number of 5-digit palindromes is 900 -/
theorem five_digit_palindromes_count :
  count_five_digit_palindromes = 900 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_palindromes_count_l1670_167055


namespace NUMINAMATH_CALUDE_xyz_value_l1670_167087

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 4 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1670_167087


namespace NUMINAMATH_CALUDE_converse_x_squared_greater_than_one_l1670_167061

theorem converse_x_squared_greater_than_one (x : ℝ) :
  x^2 > 1 → (x < -1 ∨ x > 1) :=
sorry

end NUMINAMATH_CALUDE_converse_x_squared_greater_than_one_l1670_167061


namespace NUMINAMATH_CALUDE_employed_females_percentage_l1670_167084

theorem employed_females_percentage (total_employed_percent : ℝ) (employed_males_percent : ℝ)
  (h1 : total_employed_percent = 64)
  (h2 : employed_males_percent = 48) :
  (total_employed_percent - employed_males_percent) / total_employed_percent * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l1670_167084


namespace NUMINAMATH_CALUDE_crayon_purchase_worth_l1670_167074

/-- Calculates the total worth of crayons after a discounted purchase -/
theorem crayon_purchase_worth
  (initial_packs : ℕ)
  (additional_packs : ℕ)
  (regular_price : ℝ)
  (discount_percent : ℝ)
  (h1 : initial_packs = 4)
  (h2 : additional_packs = 2)
  (h3 : regular_price = 2.5)
  (h4 : discount_percent = 15)
  : ℝ := by
  sorry

#check crayon_purchase_worth

end NUMINAMATH_CALUDE_crayon_purchase_worth_l1670_167074


namespace NUMINAMATH_CALUDE_quadratic_polynomial_discriminant_l1670_167014

-- Define a quadratic polynomial
def QuadraticPolynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic polynomial
def Discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_polynomial_discriminant 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∃! x, QuadraticPolynomial a b c x = x - 2) ∧ 
  (∃! x, QuadraticPolynomial a b c x = 1 - x/2) →
  Discriminant a b c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_discriminant_l1670_167014


namespace NUMINAMATH_CALUDE_triangle_angles_from_radii_relations_l1670_167067

/-- Given a triangle with excircle radii r_a, r_b, r_c, and circumcircle radius R,
    if r_a + r_b = 3R and r_b + r_c = 2R, then the angles of the triangle are 90°, 60°, and 30°. -/
theorem triangle_angles_from_radii_relations (r_a r_b r_c R : ℝ) 
    (h1 : r_a + r_b = 3 * R) (h2 : r_b + r_c = 2 * R) :
    ∃ (α β γ : ℝ),
      α = π / 2 ∧ β = π / 6 ∧ γ = π / 3 ∧
      α + β + γ = π ∧
      0 < α ∧ 0 < β ∧ 0 < γ :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_from_radii_relations_l1670_167067


namespace NUMINAMATH_CALUDE_digit_sum_10_2017_position_l1670_167046

/-- A sequence of positive integers whose digits sum to 10, arranged in ascending order -/
def digit_sum_10_sequence : ℕ → ℕ := sorry

/-- Predicate to check if a natural number's digits sum to 10 -/
def digits_sum_to_10 (n : ℕ) : Prop := sorry

/-- The sequence digit_sum_10_sequence contains all and only the numbers whose digits sum to 10 -/
axiom digit_sum_10_sequence_property :
  ∀ n : ℕ, digits_sum_to_10 (digit_sum_10_sequence n) ∧
  (∀ m : ℕ, digits_sum_to_10 m → ∃ k : ℕ, digit_sum_10_sequence k = m)

/-- The sequence digit_sum_10_sequence is strictly increasing -/
axiom digit_sum_10_sequence_increasing :
  ∀ n m : ℕ, n < m → digit_sum_10_sequence n < digit_sum_10_sequence m

theorem digit_sum_10_2017_position :
  ∃ n : ℕ, digit_sum_10_sequence n = 2017 ∧ n = 110 := by sorry

end NUMINAMATH_CALUDE_digit_sum_10_2017_position_l1670_167046


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1670_167045

/-- The eccentricity of the ellipse x^2 + 4y^2 = 4 is √3/2 -/
theorem ellipse_eccentricity : 
  let equation := fun (x y : ℝ) => x^2 + 4*y^2 = 4
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 - b^2)
  let e := c / a
  equation 0 1 ∧ e = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1670_167045


namespace NUMINAMATH_CALUDE_prime_quadratic_roots_l1670_167083

theorem prime_quadratic_roots (p : ℕ) : 
  Nat.Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 444*p = 0 ∧ y^2 + p*y - 444*p = 0) → 
  p = 37 :=
sorry

end NUMINAMATH_CALUDE_prime_quadratic_roots_l1670_167083


namespace NUMINAMATH_CALUDE_no_solution_for_specific_k_l1670_167088

theorem no_solution_for_specific_k (p : ℕ) (hp : Prime p) (hp_mod : p % 4 = 3) :
  ¬ ∃ (n m : ℕ+), (n.val^2 + m.val^2 : ℚ) / (m.val^4 + n.val) = p^2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_specific_k_l1670_167088


namespace NUMINAMATH_CALUDE_total_cost_is_49_l1670_167037

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def discount_threshold : ℕ := 35
def discount_amount : ℕ := 5
def num_sandwiches : ℕ := 6
def num_sodas : ℕ := 10

def total_cost : ℕ :=
  let pre_discount := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  if pre_discount > discount_threshold then
    pre_discount - discount_amount
  else
    pre_discount

theorem total_cost_is_49 : total_cost = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_49_l1670_167037


namespace NUMINAMATH_CALUDE_bicycle_price_l1670_167077

theorem bicycle_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) :
  upfront_payment = 200 →
  upfront_percentage = 0.20 →
  upfront_payment = upfront_percentage * total_price →
  total_price = 1000 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_l1670_167077


namespace NUMINAMATH_CALUDE_rachel_apples_remaining_l1670_167089

/-- The number of apples remaining on trees after picking -/
def apples_remaining (num_trees : ℕ) (apples_per_tree : ℕ) (initial_total : ℕ) : ℕ :=
  initial_total - (num_trees * apples_per_tree)

/-- Theorem: The number of apples remaining on Rachel's trees is 9 -/
theorem rachel_apples_remaining :
  apples_remaining 3 8 33 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rachel_apples_remaining_l1670_167089


namespace NUMINAMATH_CALUDE_remainder_is_perfect_square_l1670_167081

theorem remainder_is_perfect_square (n : ℕ+) : ∃ k : ℤ, (10^n.val - 1) % 37 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_is_perfect_square_l1670_167081


namespace NUMINAMATH_CALUDE_initial_amount_proof_l1670_167033

theorem initial_amount_proof (P : ℚ) : 
  (P * (1 + 1/8) * (1 + 1/8) = 105300) → P = 83200 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l1670_167033


namespace NUMINAMATH_CALUDE_peters_horse_food_l1670_167069

/-- Calculates the total amount of food needed to feed horses for a given number of days. -/
def total_food_needed (num_horses : ℕ) (oats_per_feeding : ℕ) (oats_feedings_per_day : ℕ) 
                      (grain_per_day : ℕ) (num_days : ℕ) : ℕ :=
  num_horses * (oats_per_feeding * oats_feedings_per_day + grain_per_day) * num_days

/-- Proves that Peter needs 132 pounds of food to feed his horses for 3 days. -/
theorem peters_horse_food : total_food_needed 4 4 2 3 3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_peters_horse_food_l1670_167069


namespace NUMINAMATH_CALUDE_a_range_l1670_167076

def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {1, 3, a}

theorem a_range (a : ℝ) : (A ∩ B a).Nonempty → a ∈ A := by
  sorry

end NUMINAMATH_CALUDE_a_range_l1670_167076


namespace NUMINAMATH_CALUDE_largest_number_with_property_l1670_167064

/-- Checks if a four-digit number satisfies the property that each of the last two digits
    is equal to the sum of the two preceding digits. -/
def satisfiesProperty (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n % 100 = (n / 100 % 10 + n / 10 % 10) % 10) ∧
  (n / 10 % 10 = (n / 1000 + n / 100 % 10) % 10)

/-- Theorem stating that 9099 is the largest four-digit number satisfying the property. -/
theorem largest_number_with_property :
  satisfiesProperty 9099 ∧ ∀ m : ℕ, satisfiesProperty m → m ≤ 9099 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_property_l1670_167064


namespace NUMINAMATH_CALUDE_basketball_shooting_averages_l1670_167066

/-- Represents the average number of successful shots -/
structure ShootingAverage where
  male : ℝ
  female : ℝ

/-- Represents the number of students -/
structure StudentCount where
  male : ℝ
  female : ℝ

/-- The theorem stating the average number of successful shots for male and female students -/
theorem basketball_shooting_averages 
  (avg : ShootingAverage) 
  (count : StudentCount) 
  (h1 : avg.male = 1.25 * avg.female) 
  (h2 : count.female = 1.25 * count.male) 
  (h3 : (avg.male * count.male + avg.female * count.female) / (count.male + count.female) = 4) :
  avg.male = 4.5 ∧ avg.female = 3.6 := by
  sorry

#check basketball_shooting_averages

end NUMINAMATH_CALUDE_basketball_shooting_averages_l1670_167066


namespace NUMINAMATH_CALUDE_last_season_episodes_l1670_167057

/-- The number of seasons before the announcement -/
def previous_seasons : ℕ := 9

/-- The number of episodes in each regular season -/
def episodes_per_season : ℕ := 22

/-- The duration of each episode in hours -/
def episode_duration : ℚ := 1/2

/-- The total watch time for all seasons in hours -/
def total_watch_time : ℚ := 112

/-- The additional episodes in the last season compared to regular seasons -/
def additional_episodes : ℕ := 4

theorem last_season_episodes (last_season_episodes : ℕ) :
  last_season_episodes = episodes_per_season + additional_episodes ∧
  (previous_seasons * episodes_per_season + last_season_episodes) * episode_duration = total_watch_time :=
by sorry

end NUMINAMATH_CALUDE_last_season_episodes_l1670_167057


namespace NUMINAMATH_CALUDE_contract_completion_problem_l1670_167068

/-- Represents the contract completion problem -/
theorem contract_completion_problem (total_days : ℕ) (initial_hours_per_day : ℕ) 
  (days_worked : ℕ) (work_completed_fraction : ℚ) (additional_men : ℕ) 
  (new_hours_per_day : ℕ) :
  total_days = 46 →
  initial_hours_per_day = 8 →
  days_worked = 33 →
  work_completed_fraction = 4/7 →
  additional_men = 81 →
  new_hours_per_day = 9 →
  ∃ (initial_men : ℕ), 
    (initial_men * days_worked * initial_hours_per_day : ℚ) / (total_days * initial_hours_per_day) = work_completed_fraction ∧
    ((initial_men + additional_men) * (total_days - days_worked) * new_hours_per_day : ℚ) / (total_days * initial_hours_per_day) = 1 - work_completed_fraction ∧
    initial_men = 117 :=
by sorry

end NUMINAMATH_CALUDE_contract_completion_problem_l1670_167068


namespace NUMINAMATH_CALUDE_phone_rep_hourly_wage_l1670_167024

/-- Calculates the hourly wage for phone reps given the number of reps, hours worked per day, days worked, and total payment -/
def hourly_wage (num_reps : ℕ) (hours_per_day : ℕ) (days_worked : ℕ) (total_payment : ℕ) : ℚ :=
  total_payment / (num_reps * hours_per_day * days_worked)

/-- Proves that the hourly wage for phone reps is $14 given the specified conditions -/
theorem phone_rep_hourly_wage :
  hourly_wage 50 8 5 28000 = 14 := by
  sorry

#eval hourly_wage 50 8 5 28000

end NUMINAMATH_CALUDE_phone_rep_hourly_wage_l1670_167024
