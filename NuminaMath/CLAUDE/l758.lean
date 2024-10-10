import Mathlib

namespace least_three_digit_multiple_of_13_l758_75801

theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), n = 104 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 13 ∣ m → n ≤ m) ∧ 
  100 ≤ n ∧ n < 1000 ∧ 13 ∣ n :=
sorry

end least_three_digit_multiple_of_13_l758_75801


namespace no_integer_function_satisfies_condition_l758_75840

theorem no_integer_function_satisfies_condition :
  ¬ ∃ (f : ℤ → ℤ), ∀ (x y : ℤ), f (x + f y) = f x - y :=
by sorry

end no_integer_function_satisfies_condition_l758_75840


namespace sarah_ate_one_apple_l758_75884

/-- The number of apples Sarah ate while walking home -/
def apples_eaten (total : ℕ) (to_teachers : ℕ) (to_friends : ℕ) (left : ℕ) : ℕ :=
  total - (to_teachers + to_friends) - left

/-- Theorem stating that Sarah ate 1 apple while walking home -/
theorem sarah_ate_one_apple :
  apples_eaten 25 16 5 3 = 1 := by
  sorry

end sarah_ate_one_apple_l758_75884


namespace m_less_than_n_l758_75804

theorem m_less_than_n (x : ℝ) : (x + 2) * (x + 3) < 2 * x^2 + 5 * x + 9 := by
  sorry

end m_less_than_n_l758_75804


namespace inclination_angle_range_l758_75808

/-- The range of inclination angles for a line passing through (1, 1) and (2, m²) -/
theorem inclination_angle_range (m : ℝ) : 
  let α := Real.arctan (m^2 - 1)
  0 ≤ α ∧ α < π/2 ∨ 3*π/4 ≤ α ∧ α < π := by
  sorry

end inclination_angle_range_l758_75808


namespace a_greater_than_b_l758_75820

theorem a_greater_than_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := by
  sorry

end a_greater_than_b_l758_75820


namespace inequality_proof_l758_75850

-- Define the function f(x) = |x-1|
def f (x : ℝ) : ℝ := |x - 1|

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) (ha_neq_zero : a ≠ 0) :
  f (a * b) / |a| > f (b / a) := by
  sorry


end inequality_proof_l758_75850


namespace area_comparison_l758_75835

-- Define a polygon as a list of points in 2D space
def Polygon := List (Real × Real)

-- Function to calculate the area of a polygon
noncomputable def area (p : Polygon) : Real := sorry

-- Function to check if a polygon is convex
def isConvex (p : Polygon) : Prop := sorry

-- Function to check if two polygons have equal corresponding sides
def equalSides (p1 p2 : Polygon) : Prop := sorry

-- Function to check if a polygon is inscribed in a circle
def isInscribed (p : Polygon) : Prop := sorry

-- Theorem statement
theorem area_comparison 
  (A B : Polygon) 
  (h1 : isConvex A) 
  (h2 : isConvex B) 
  (h3 : equalSides A B) 
  (h4 : isInscribed B) : 
  area B ≥ area A := by sorry

end area_comparison_l758_75835


namespace sanchez_rope_theorem_l758_75854

/-- Represents the number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Represents the number of feet of rope bought last week -/
def last_week_feet : ℕ := 6

/-- Represents the difference in feet between last week's and this week's purchase -/
def difference_feet : ℕ := 4

/-- Calculates the total inches of rope bought by Mr. Sanchez -/
def total_inches : ℕ := 
  (last_week_feet * inches_per_foot) + ((last_week_feet - difference_feet) * inches_per_foot)

/-- Theorem stating that the total inches of rope bought is 96 -/
theorem sanchez_rope_theorem : total_inches = 96 := by
  sorry

end sanchez_rope_theorem_l758_75854


namespace complex_equation_sum_l758_75825

theorem complex_equation_sum (x y : ℝ) : (2*x - y : ℂ) + (x + 3)*I = 0 → x + y = -9 := by
  sorry

end complex_equation_sum_l758_75825


namespace min_tablets_for_both_types_l758_75852

/-- Given a box with tablets of two types of medicine, this theorem proves
    the minimum number of tablets needed to ensure at least one of each type
    when extracting a specific total number. -/
theorem min_tablets_for_both_types 
  (total_A : ℕ) 
  (total_B : ℕ) 
  (extract_total : ℕ) 
  (h1 : total_A = 10) 
  (h2 : total_B = 16) 
  (h3 : extract_total = 18) :
  extract_total = min (total_A + total_B) extract_total := by
sorry

end min_tablets_for_both_types_l758_75852


namespace fraction_sum_simplification_l758_75839

theorem fraction_sum_simplification :
  8 / 19 - 5 / 57 + 1 / 3 = 2 / 3 := by
  sorry

end fraction_sum_simplification_l758_75839


namespace function_properties_l758_75871

noncomputable def f (a b x : ℝ) := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem function_properties (a b : ℝ) (h1 : a * b ≠ 0) 
  (h2 : ∀ x : ℝ, f a b x ≤ |f a b (π/6)|) : 
  (f a b (11*π/12) = 0) ∧ 
  (|f a b (7*π/12)| < |f a b (π/5)|) ∧ 
  (∀ x : ℝ, f a b (-x) ≠ f a b x ∧ f a b (-x) ≠ -f a b x) ∧
  (∀ k m : ℝ, ∃ x : ℝ, k * x + m = f a b x) :=
by sorry

end function_properties_l758_75871


namespace pentagon_angle_measure_l758_75858

/-- Given a pentagon STARS where four of its angles are congruent and two of these are equal, 
    prove that the measure of one of these angles is 108°. -/
theorem pentagon_angle_measure (S T A R : ℝ) : 
  (S + T + A + R + S = 540) → -- Sum of angles in a pentagon
  (S = T) → (T = A) → (A = R) → -- Four angles are congruent
  (A = S) → -- Two of these angles are equal
  R = 108 := by sorry

end pentagon_angle_measure_l758_75858


namespace quadratic_inequality_solution_l758_75896

theorem quadratic_inequality_solution (a : ℝ) (x : ℝ) :
  a * x^2 - 2 ≥ 2 * x - a * x ↔
    (a = 0 ∧ x ≤ -1) ∨
    (a > 0 ∧ (x ≥ 2 / a ∨ x ≤ -1)) ∨
    (-2 < a ∧ a < 0 ∧ 2 / a ≤ x ∧ x ≤ -1) ∨
    (a = -2 ∧ x = -1) ∨
    (a < -2 ∧ -1 ≤ x ∧ x ≤ 2 / a) :=
by sorry

end quadratic_inequality_solution_l758_75896


namespace mosquito_lethal_feedings_l758_75889

/-- The number of mosquito feedings required to reach lethal blood loss -/
def lethal_feedings (drops_per_feeding : ℕ) (drops_per_liter : ℕ) (lethal_liters : ℕ) : ℕ :=
  (lethal_liters * drops_per_liter) / drops_per_feeding

theorem mosquito_lethal_feedings :
  lethal_feedings 20 5000 3 = 750 := by
  sorry

#eval lethal_feedings 20 5000 3

end mosquito_lethal_feedings_l758_75889


namespace custom_operation_theorem_l758_75842

-- Define the custom operation *
def star (a b : ℝ) : ℝ := (a + b)^2

-- State the theorem
theorem custom_operation_theorem (x y : ℝ) : 
  star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by sorry

end custom_operation_theorem_l758_75842


namespace gcd_876543_765432_l758_75814

theorem gcd_876543_765432 : Nat.gcd 876543 765432 = 1 := by
  sorry

end gcd_876543_765432_l758_75814


namespace james_muffins_count_l758_75860

def arthur_muffins : ℕ := 115
def james_multiplier : ℚ := 12.5

theorem james_muffins_count :
  ⌈(arthur_muffins : ℚ) * james_multiplier⌉ = 1438 := by
  sorry

end james_muffins_count_l758_75860


namespace circle_equation_l758_75888

/-- The equation of a circle with center (0, 4) passing through (3, 0) is x² + (y - 4)² = 25 -/
theorem circle_equation (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ x^2 + (y - 4)^2 = r^2) ∧ 
  (3^2 + (0 - 4)^2 = x^2 + (y - 4)^2) → 
  x^2 + (y - 4)^2 = 25 := by
sorry

end circle_equation_l758_75888


namespace car_cost_difference_l758_75870

/-- Represents the cost and characteristics of a car --/
structure Car where
  initialCost : ℕ
  fuelConsumption : ℕ
  annualInsurance : ℕ
  annualMaintenance : ℕ
  resaleValue : ℕ

/-- Calculates the total cost of owning a car for 5 years --/
def totalCost (c : Car) (annualDistance : ℕ) (fuelCost : ℕ) (years : ℕ) : ℕ :=
  c.initialCost +
  (annualDistance * c.fuelConsumption * fuelCost * years) / 10000 +
  c.annualInsurance * years +
  c.annualMaintenance * years -
  c.resaleValue

/-- The statement to be proved --/
theorem car_cost_difference :
  let carA : Car := {
    initialCost := 900000,
    fuelConsumption := 9,
    annualInsurance := 35000,
    annualMaintenance := 25000,
    resaleValue := 500000
  }
  let carB : Car := {
    initialCost := 600000,
    fuelConsumption := 10,
    annualInsurance := 32000,
    annualMaintenance := 20000,
    resaleValue := 350000
  }
  let annualDistance := 15000
  let fuelCost := 40
  let years := 5
  totalCost carA annualDistance fuelCost years - totalCost carB annualDistance fuelCost years = 160000 := by
  sorry

end car_cost_difference_l758_75870


namespace triangle_area_l758_75859

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) :
  (1/2) * a * b = 54 := by
  sorry

end triangle_area_l758_75859


namespace graph_translation_l758_75807

/-- Translating the graph of f(x) = cos(2x - π/3) to the left by π/6 units results in y = cos(2x) -/
theorem graph_translation (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.cos (2 * x - π / 3)
  let g : ℝ → ℝ := λ x => Real.cos (2 * x)
  let h : ℝ → ℝ := λ x => f (x + π / 6)
  h x = g x := by sorry

end graph_translation_l758_75807


namespace min_sum_power_mod_l758_75886

theorem min_sum_power_mod (m n : ℕ) : 
  n > m → 
  m > 1 → 
  (1978^m) % 1000 = (1978^n) % 1000 → 
  ∃ (m₀ n₀ : ℕ), m₀ + n₀ = 106 ∧ 
    ∀ (m' n' : ℕ), n' > m' → m' > 1 → 
      (1978^m') % 1000 = (1978^n') % 1000 → 
      m' + n' ≥ m₀ + n₀ :=
by sorry

end min_sum_power_mod_l758_75886


namespace total_problems_l758_75865

def daily_record : List Int := [-3, 5, -4, 2, -1, 1, 0, -3, 8, 7]

theorem total_problems (record : List Int) (h : record = daily_record) :
  (List.sum record + 60 : Int) = 72 := by
  sorry

end total_problems_l758_75865


namespace equation_solution_l758_75878

theorem equation_solution : ∃ x : ℚ, x ≠ 1 ∧ (x^2 - 2*x + 3) / (x - 1) = x + 4 ∧ x = 7/5 := by
  sorry

end equation_solution_l758_75878


namespace equation_has_solution_in_interval_l758_75831

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 3*x + 5

-- State the theorem
theorem equation_has_solution_in_interval :
  (Continuous f) → ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  sorry


end equation_has_solution_in_interval_l758_75831


namespace fourth_vertex_of_parallelogram_l758_75881

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Addition of a point and a vector -/
def Point2D.add (p : Point2D) (v : Vector2D) : Point2D :=
  ⟨p.x + v.x, p.y + v.y⟩

/-- Subtraction of two points to get a vector -/
def Point2D.sub (p q : Point2D) : Vector2D :=
  ⟨p.x - q.x, p.y - q.y⟩

/-- The given points of the parallelogram -/
def Q : Point2D := ⟨1, -1⟩
def R : Point2D := ⟨-1, 0⟩
def S : Point2D := ⟨0, 1⟩

/-- The theorem stating that the fourth vertex of the parallelogram is (-2, 2) -/
theorem fourth_vertex_of_parallelogram :
  let V := S.add (R.sub Q)
  V = Point2D.mk (-2) 2 := by
  sorry

end fourth_vertex_of_parallelogram_l758_75881


namespace six_digit_permutations_count_l758_75800

/-- The number of different positive, six-digit integers that can be formed using the digits 1, 2, 2, 5, 9, and 9 -/
def six_digit_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of such integers is 180 -/
theorem six_digit_permutations_count : six_digit_permutations = 180 := by
  sorry

end six_digit_permutations_count_l758_75800


namespace inscribed_quadrilateral_with_geometric_sides_is_square_l758_75861

/-- A quadrilateral inscribed around a circle with sides in geometric progression is a square -/
theorem inscribed_quadrilateral_with_geometric_sides_is_square
  (R : ℝ) -- radius of the inscribed circle
  (a : ℝ) -- first term of the geometric progression
  (r : ℝ) -- common ratio of the geometric progression
  (h1 : R > 0)
  (h2 : a > 0)
  (h3 : r > 0)
  (h4 : a + a * r^3 = a * r + a * r^2) -- Pitot's theorem
  : 
  r = 1 ∧ -- all sides are equal
  R = a / 2 ∧ -- radius is half the side length
  a^2 = 4 * R^2 -- area of the quadrilateral
  := by sorry

#check inscribed_quadrilateral_with_geometric_sides_is_square

end inscribed_quadrilateral_with_geometric_sides_is_square_l758_75861


namespace least_perimeter_of_triangle_l758_75826

theorem least_perimeter_of_triangle (a b x : ℕ) : 
  a = 33 → b = 42 → x > 0 → 
  x + a > b → x + b > a → a + b > x →
  ∀ y : ℕ, y > 0 → y + a > b → y + b > a → a + b > y → x ≤ y →
  a + b + x = 85 := by
sorry

end least_perimeter_of_triangle_l758_75826


namespace population_difference_after_two_years_l758_75819

/-- The difference in population between city A and city C after 2 years -/
def population_difference (A B C : ℝ) : ℝ :=
  A * (1 + 0.03)^2 - C * (1 + 0.02)^2

/-- Theorem stating the difference in population after 2 years -/
theorem population_difference_after_two_years (A B C : ℝ) 
  (h : A + B = B + C + 5000) :
  population_difference A B C = 0.0205 * A + 5202 := by
  sorry

end population_difference_after_two_years_l758_75819


namespace quadratic_factorization_l758_75821

theorem quadratic_factorization (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end quadratic_factorization_l758_75821


namespace min_value_expressions_l758_75856

theorem min_value_expressions (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (max x (1/y) + max y (2/x) ≥ 2 * Real.sqrt 2) ∧
  (max x (1/y) + max y (2/z) + max z (3/x) ≥ 2 * Real.sqrt 5) := by
  sorry

end min_value_expressions_l758_75856


namespace decimal_sum_and_subtraction_l758_75883

theorem decimal_sum_and_subtraction :
  (0.5 + 0.003 + 0.070) - 0.008 = 0.565 := by
  sorry

end decimal_sum_and_subtraction_l758_75883


namespace company_size_proof_l758_75867

/-- The total number of employees in the company -/
def total_employees : ℕ := 100

/-- The number of employees in group C -/
def group_C_employees : ℕ := 10

/-- The ratio of employees in levels A:B:C -/
def employee_ratio : Fin 3 → ℕ
| 0 => 5  -- Level A
| 1 => 4  -- Level B
| 2 => 1  -- Level C

/-- The size of the stratified sample -/
def sample_size : ℕ := 20

/-- The probability of selecting both people from group C in the sample -/
def prob_both_from_C : ℚ := 1 / 45

theorem company_size_proof :
  (total_employees = 100) ∧
  (group_C_employees = 10) ∧
  (∀ i : Fin 3, employee_ratio i = [5, 4, 1].get i) ∧
  (sample_size = 20) ∧
  (prob_both_from_C = 1 / 45) ∧
  (group_C_employees.choose 2 = prob_both_from_C * total_employees.choose 2) ∧
  (group_C_employees * (employee_ratio 0 + employee_ratio 1 + employee_ratio 2) = total_employees) :=
by sorry

#check company_size_proof

end company_size_proof_l758_75867


namespace g_inequality_range_l758_75805

noncomputable def g (x : ℝ) : ℝ := 2^x + 2^(-x) + |x|

theorem g_inequality_range : 
  {x : ℝ | g (2*x - 1) < g 3} = Set.Ioo (-1) 2 := by sorry

end g_inequality_range_l758_75805


namespace power_inequality_l758_75891

theorem power_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^2 > x + y) (h2 : x^4 > x^3 + y) : x^3 > x^2 + y := by
  sorry

end power_inequality_l758_75891


namespace tims_books_l758_75829

theorem tims_books (mike_books : ℕ) (total_books : ℕ) (h1 : mike_books = 20) (h2 : total_books = 42) :
  total_books - mike_books = 22 := by
sorry

end tims_books_l758_75829


namespace quadratic_equation_condition_l758_75855

theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, ∃ p q r : ℝ, (a + 4) * x^(a^2 - 14) - 3 * x + 8 = p * x^2 + q * x + r) ∧ 
  (a + 4 ≠ 0) → 
  a = 4 := by
sorry

end quadratic_equation_condition_l758_75855


namespace students_suggesting_bacon_l758_75834

theorem students_suggesting_bacon (total : ℕ) (mashed_potatoes : ℕ) (tomatoes : ℕ) 
  (h1 : total = 826)
  (h2 : mashed_potatoes = 324)
  (h3 : tomatoes = 128) :
  total - (mashed_potatoes + tomatoes) = 374 := by
  sorry

end students_suggesting_bacon_l758_75834


namespace isosceles_triangle_perimeter_l758_75882

/-- An isosceles triangle with sides 4 and 6 has a perimeter of either 14 or 16. -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  (a = 4 ∧ b = 4 ∧ c = 6) ∨ (a = 4 ∧ b = 6 ∧ c = 6) →  -- possible configurations
  a + b > c ∧ b + c > a ∧ c + a > b →  -- triangle inequality
  a + b + c = 14 ∨ a + b + c = 16 :=
by sorry


end isosceles_triangle_perimeter_l758_75882


namespace min_colors_is_23_l758_75838

/-- A coloring scheme for boxes of balls -/
structure ColoringScheme where
  n : ℕ  -- number of colors
  boxes : Fin 8 → Fin 6 → Fin n  -- coloring function

/-- Predicate to check if a coloring scheme is valid -/
def is_valid_coloring (c : ColoringScheme) : Prop :=
  -- No two balls in the same box have the same color
  (∀ i : Fin 8, ∀ j k : Fin 6, j ≠ k → c.boxes i j ≠ c.boxes i k) ∧
  -- No two colors occur together in more than one box
  (∀ i j : Fin 8, i ≠ j → ∀ c1 c2 : Fin c.n, c1 ≠ c2 →
    (∃ k : Fin 6, c.boxes i k = c1 ∧ ∃ l : Fin 6, c.boxes i l = c2) →
    ¬(∃ m : Fin 6, c.boxes j m = c1 ∧ ∃ n : Fin 6, c.boxes j n = c2))

/-- The main theorem: the minimum number of colors is 23 -/
theorem min_colors_is_23 :
  (∃ c : ColoringScheme, c.n = 23 ∧ is_valid_coloring c) ∧
  (∀ c : ColoringScheme, c.n < 23 → ¬is_valid_coloring c) := by
  sorry

end min_colors_is_23_l758_75838


namespace solution_set_equivalence_l758_75813

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_increasing : ∀ x y, x < y → f x < f y)
variable (h_f_0 : f 0 = -1)
variable (h_f_3 : f 3 = 1)

-- Define the solution set
def solution_set := {x : ℝ | |f (x + 1)| < 1}

-- State the theorem
theorem solution_set_equivalence : 
  solution_set f = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end solution_set_equivalence_l758_75813


namespace geometric_series_sum_l758_75815

/-- Given real numbers a, b, and c such that the infinite geometric series
    a/b + a/b^2 + a/b^3 + ... equals 3, prove that the sum of the series
    ca/(a+b) + ca/(a+b)^2 + ca/(a+b)^3 + ... equals 3c/4 -/
theorem geometric_series_sum (a b c : ℝ) 
  (h : ∑' n, a / b^n = 3) : 
  ∑' n, c * a / (a + b)^n = 3/4 * c := by
  sorry

end geometric_series_sum_l758_75815


namespace five_balls_three_boxes_l758_75894

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 243 ways to put 5 distinguishable balls in 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 243 := by
  sorry

end five_balls_three_boxes_l758_75894


namespace parallelogram_area_60_16_l758_75837

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 60 cm and height 16 cm is 960 square centimeters -/
theorem parallelogram_area_60_16 : 
  parallelogram_area 60 16 = 960 := by sorry

end parallelogram_area_60_16_l758_75837


namespace expression_value_l758_75816

theorem expression_value (x y : ℝ) (h1 : x + y = 4) (h2 : x * y = -2) :
  ∃ ε > 0, |x + x^3/y^2 + y^3/x^2 + y - 440| < ε :=
sorry

end expression_value_l758_75816


namespace radio_loss_percentage_l758_75857

/-- Calculate the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that the loss percentage for a radio with
    cost price 1500 and selling price 1290 is 14% -/
theorem radio_loss_percentage :
  loss_percentage 1500 1290 = 14 := by sorry

end radio_loss_percentage_l758_75857


namespace manager_percentage_l758_75818

theorem manager_percentage (total_employees : ℕ) (initial_percentage : ℚ) 
  (managers_leaving : ℚ) (final_percentage : ℚ) : 
  total_employees = 300 →
  initial_percentage = 99/100 →
  managers_leaving = 149.99999999999986 →
  final_percentage = 49/100 →
  (↑total_employees * initial_percentage - managers_leaving) / ↑total_employees = final_percentage :=
by sorry

end manager_percentage_l758_75818


namespace same_color_probability_l758_75803

def red_plates : ℕ := 7
def blue_plates : ℕ := 5
def green_plates : ℕ := 3

def total_plates : ℕ := red_plates + blue_plates + green_plates

def same_color_pairs : ℕ := (red_plates.choose 2) + (blue_plates.choose 2) + (green_plates.choose 2)
def total_pairs : ℕ := total_plates.choose 2

theorem same_color_probability :
  (same_color_pairs : ℚ) / total_pairs = 34 / 105 :=
by sorry

end same_color_probability_l758_75803


namespace multiplication_commutativity_certainty_l758_75868

theorem multiplication_commutativity_certainty :
  ∀ (a b : ℝ), a * b = b * a := by
  sorry

end multiplication_commutativity_certainty_l758_75868


namespace soccer_team_enrollment_l758_75806

theorem soccer_team_enrollment (total : ℕ) (physics : ℕ) (both : ℕ) (mathematics : ℕ)
  (h1 : total = 15)
  (h2 : physics = 9)
  (h3 : both = 3)
  (h4 : physics + mathematics - both = total) :
  mathematics = 9 := by
  sorry

end soccer_team_enrollment_l758_75806


namespace sufficient_but_not_necessary_l758_75848

theorem sufficient_but_not_necessary (a b : ℝ) :
  (((2 : ℝ) ^ a > (2 : ℝ) ^ b ∧ (2 : ℝ) ^ b > 1) → (a ^ (1/3) > b ^ (1/3))) ∧
  ¬(∀ a b : ℝ, a ^ (1/3) > b ^ (1/3) → ((2 : ℝ) ^ a > (2 : ℝ) ^ b ∧ (2 : ℝ) ^ b > 1)) :=
by sorry

end sufficient_but_not_necessary_l758_75848


namespace factorization_of_cubic_l758_75809

theorem factorization_of_cubic (b : ℝ) : 2*b^3 - 4*b^2 + 2*b = 2*b*(b-1)^2 := by
  sorry

end factorization_of_cubic_l758_75809


namespace total_travel_time_l758_75822

def luke_bus_time : ℕ := 70
def paula_bus_time : ℕ := (3 * luke_bus_time) / 5
def jane_train_time : ℕ := 120
def michael_cycle_time : ℕ := jane_train_time / 4

def luke_total_time : ℕ := luke_bus_time + 5 * luke_bus_time
def paula_total_time : ℕ := 2 * paula_bus_time
def jane_total_time : ℕ := jane_train_time + 2 * jane_train_time
def michael_total_time : ℕ := 2 * michael_cycle_time

theorem total_travel_time :
  luke_total_time + paula_total_time + jane_total_time + michael_total_time = 924 :=
by sorry

end total_travel_time_l758_75822


namespace x_factor_change_l758_75833

/-- Given a function q defined in terms of e, x, and z, prove that when e is quadrupled,
    z is tripled, and q is multiplied by 0.2222222222222222, x is doubled. -/
theorem x_factor_change (e x z : ℝ) (h : x ≠ 0) (hz : z ≠ 0) :
  let q := 5 * e / (4 * x * z^2)
  let q' := 0.2222222222222222 * (5 * (4 * e) / (4 * x * (3 * z)^2))
  ∃ x' : ℝ, x' = 2 * x ∧ q' = 5 * (4 * e) / (4 * x' * (3 * z)^2) :=
by sorry

end x_factor_change_l758_75833


namespace sum_odd_and_even_integers_l758_75830

def sum_odd_integers (n : ℕ) : ℕ := 
  (n^2 + n) / 2

def sum_even_integers (n : ℕ) : ℕ := 
  n * (n + 1)

theorem sum_odd_and_even_integers : 
  sum_odd_integers 111 + sum_even_integers 25 = 3786 := by
  sorry

end sum_odd_and_even_integers_l758_75830


namespace triangle_area_13_13_24_l758_75827

/-- The area of a triangle with side lengths 13, 13, and 24 is 60 square units. -/
theorem triangle_area_13_13_24 : ∃ (A : ℝ), 
  A = (1/2) * 24 * Real.sqrt (13^2 - 12^2) ∧ A = 60 := by sorry

end triangle_area_13_13_24_l758_75827


namespace intersection_polygon_exists_and_unique_l758_75864

/- Define the cube and points -/
def cube_edge_length : ℝ := 30

/- Define points on cube edges -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨0, 0, 0⟩
def B : Point3D := ⟨cube_edge_length, 0, 0⟩
def C : Point3D := ⟨cube_edge_length, 0, cube_edge_length⟩
def D : Point3D := ⟨cube_edge_length, cube_edge_length, cube_edge_length⟩

def P : Point3D := ⟨10, 0, 0⟩
def Q : Point3D := ⟨cube_edge_length, 0, 10⟩
def R : Point3D := ⟨cube_edge_length, 15, cube_edge_length⟩

/- Define the plane PQR -/
def plane_PQR (x y z : ℝ) : Prop := 2*x + y - 2*z = 15

/- Define the cube -/
def in_cube (p : Point3D) : Prop :=
  0 ≤ p.x ∧ p.x ≤ cube_edge_length ∧
  0 ≤ p.y ∧ p.y ≤ cube_edge_length ∧
  0 ≤ p.z ∧ p.z ≤ cube_edge_length

/- Theorem statement -/
theorem intersection_polygon_exists_and_unique :
  ∃! polygon : Set Point3D,
    (∀ p ∈ polygon, in_cube p ∧ plane_PQR p.x p.y p.z) ∧
    (∀ p, in_cube p ∧ plane_PQR p.x p.y p.z → p ∈ polygon) :=
sorry

end intersection_polygon_exists_and_unique_l758_75864


namespace rosy_fish_count_l758_75877

/-- Given that Lilly has 10 fish and the total number of fish is 22,
    prove that Rosy has 12 fish. -/
theorem rosy_fish_count (lilly_fish : ℕ) (total_fish : ℕ) (h1 : lilly_fish = 10) (h2 : total_fish = 22) :
  total_fish - lilly_fish = 12 := by
  sorry

end rosy_fish_count_l758_75877


namespace no_such_polyhedron_l758_75893

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ
  is_convex : Bool
  no_triangular_faces : Bool
  no_three_valent_vertices : Bool

/-- Euler's formula for polyhedra -/
def euler_formula (p : ConvexPolyhedron) : Prop :=
  p.faces + p.vertices - p.edges = 2

/-- Theorem: A convex polyhedron with no triangular faces and no three-valent vertices violates Euler's formula -/
theorem no_such_polyhedron (p : ConvexPolyhedron) 
  (h_convex : p.is_convex = true) 
  (h_no_tri : p.no_triangular_faces = true) 
  (h_no_three : p.no_three_valent_vertices = true) : 
  ¬(euler_formula p) := by
  sorry

end no_such_polyhedron_l758_75893


namespace quadratic_function_bounds_l758_75817

/-- Given a quadratic function f(x) = ax² + bx with certain constraints on f(-1) and f(1),
    prove that f(-2) is bounded between 6 and 10. -/
theorem quadratic_function_bounds (a b : ℝ) :
  let f := fun (x : ℝ) => a * x^2 + b * x
  (1 ≤ f (-1) ∧ f (-1) ≤ 2) →
  (3 ≤ f 1 ∧ f 1 ≤ 4) →
  (6 ≤ f (-2) ∧ f (-2) ≤ 10) :=
by sorry

end quadratic_function_bounds_l758_75817


namespace wire_parts_used_l758_75887

theorem wire_parts_used (total_length : ℝ) (total_parts : ℕ) (unused_length : ℝ) : 
  total_length = 50 →
  total_parts = 5 →
  unused_length = 20 →
  (total_parts : ℝ) - (unused_length / (total_length / total_parts)) = 3 := by
  sorry

end wire_parts_used_l758_75887


namespace cos_sin_sum_equals_sqrt2_over_2_l758_75844

theorem cos_sin_sum_equals_sqrt2_over_2 : 
  Real.cos (80 * π / 180) * Real.cos (35 * π / 180) + 
  Real.sin (80 * π / 180) * Real.cos (55 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end cos_sin_sum_equals_sqrt2_over_2_l758_75844


namespace min_value_trig_function_l758_75823

theorem min_value_trig_function (x : ℝ) : 
  Real.sin x ^ 4 + Real.cos x ^ 4 + (1 / Real.cos x) ^ 4 + (1 / Real.sin x) ^ 4 ≥ 8.5 := by
  sorry

end min_value_trig_function_l758_75823


namespace divisibility_of_expression_l758_75892

theorem divisibility_of_expression (n : ℕ) (h : Odd n) (h' : n > 0) :
  ∃ k : ℤ, n^4 - n^2 - n = n * k :=
sorry

end divisibility_of_expression_l758_75892


namespace cycle_selling_price_l758_75841

def cost_price : ℝ := 2800
def loss_percentage : ℝ := 25

theorem cycle_selling_price :
  let loss := (loss_percentage / 100) * cost_price
  let selling_price := cost_price - loss
  selling_price = 2100 := by sorry

end cycle_selling_price_l758_75841


namespace line_inclination_trig_identity_l758_75879

/-- Given a line with equation x - 2y + 1 = 0 and inclination angle α, 
    prove that cos²α + sin(2α) = 8/5 -/
theorem line_inclination_trig_identity (α : ℝ) : 
  (∃ x y : ℝ, x - 2*y + 1 = 0 ∧ Real.tan α = 1/2) → 
  Real.cos α ^ 2 + Real.sin (2 * α) = 8/5 :=
by sorry

end line_inclination_trig_identity_l758_75879


namespace expression_simplification_l758_75895

theorem expression_simplification (a₁ a₂ a₃ a₄ : ℝ) :
  1 + a₁ / (1 - a₁) + a₂ / ((1 - a₁) * (1 - a₂)) + 
  a₃ / ((1 - a₁) * (1 - a₂) * (1 - a₃)) + 
  (a₄ - a₁) / ((1 - a₁) * (1 - a₂) * (1 - a₃) * (1 - a₄)) = 
  1 / ((1 - a₂) * (1 - a₃) * (1 - a₄)) :=
by sorry

end expression_simplification_l758_75895


namespace gecko_sale_price_l758_75874

/-- The amount Brandon sold the geckos for -/
def brandon_sale_price : ℝ := 100

/-- The pet store's selling price -/
def pet_store_price (x : ℝ) : ℝ := 3 * x + 5

/-- The pet store's profit -/
def pet_store_profit : ℝ := 205

theorem gecko_sale_price :
  pet_store_price brandon_sale_price - brandon_sale_price = pet_store_profit :=
by sorry

end gecko_sale_price_l758_75874


namespace proportional_relationship_l758_75846

/-- The constant of proportionality -/
def k : ℝ := 3

/-- The functional relationship between y and x -/
def f (x : ℝ) : ℝ := -k * x + 10

theorem proportional_relationship (x y : ℝ) :
  (y + 2 = k * (4 - x)) ∧ (f 3 = 1) →
  (∀ x, f x = -3 * x + 10) ∧
  (∀ y, -2 < y → y < 1 → ∃ x, 3 < x ∧ x < 4 ∧ f x = y) :=
by sorry

end proportional_relationship_l758_75846


namespace jolyn_older_than_clarisse_l758_75843

/-- Represents an age difference in months and days -/
structure AgeDifference where
  months : ℕ
  days : ℕ

/-- Adds two age differences -/
def addAgeDifference (ad1 ad2 : AgeDifference) : AgeDifference :=
  { months := ad1.months + ad2.months + (ad1.days + ad2.days) / 30,
    days := (ad1.days + ad2.days) % 30 }

/-- Subtracts two age differences -/
def subtractAgeDifference (ad1 ad2 : AgeDifference) : AgeDifference :=
  { months := ad1.months - ad2.months - (if ad1.days < ad2.days then 1 else 0),
    days := if ad1.days < ad2.days then ad1.days + 30 - ad2.days else ad1.days - ad2.days }

theorem jolyn_older_than_clarisse
  (jolyn_therese : AgeDifference)
  (therese_aivo : AgeDifference)
  (leon_aivo : AgeDifference)
  (clarisse_leon : AgeDifference)
  (h1 : jolyn_therese = { months := 2, days := 10 })
  (h2 : therese_aivo = { months := 5, days := 15 })
  (h3 : leon_aivo = { months := 2, days := 25 })
  (h4 : clarisse_leon = { months := 3, days := 20 })
  : subtractAgeDifference (addAgeDifference jolyn_therese therese_aivo)
                          (addAgeDifference clarisse_leon leon_aivo)
    = { months := 1, days := 10 } := by
  sorry


end jolyn_older_than_clarisse_l758_75843


namespace quadratic_equation_properties_l758_75828

theorem quadratic_equation_properties (x y : ℝ) 
  (h : (x - y)^2 - 2*(x + y) + 1 = 0) : 
  (x ≥ 0 ∧ y ≥ 0) ∧ 
  (x > 1 ∧ y < x → Real.sqrt x - Real.sqrt y = 1) ∧
  (x < 1 ∧ y < 1 → Real.sqrt x + Real.sqrt y = 1) := by
  sorry

end quadratic_equation_properties_l758_75828


namespace polyhedron_has_triangle_l758_75812

/-- A polyhedron with edges of non-increasing lengths -/
structure Polyhedron where
  n : ℕ
  edges : Fin n → ℝ
  edges_decreasing : ∀ i j, i ≤ j → edges i ≥ edges j

/-- Three edges can form a triangle if the sum of any two is greater than the third -/
def CanFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- In any polyhedron, there exist three edges that can form a triangle -/
theorem polyhedron_has_triangle (P : Polyhedron) :
  ∃ i j k, i < j ∧ j < k ∧ CanFormTriangle (P.edges i) (P.edges j) (P.edges k) := by
  sorry

end polyhedron_has_triangle_l758_75812


namespace jerichos_remaining_money_l758_75836

def jerichos_money_problem (jerichos_money : ℚ) (debt_to_annika : ℚ) : Prop :=
  2 * jerichos_money = 60 ∧
  debt_to_annika = 14 ∧
  let debt_to_manny := debt_to_annika / 2
  let remaining_money := jerichos_money - debt_to_annika - debt_to_manny
  remaining_money = 9

theorem jerichos_remaining_money :
  ∀ (jerichos_money : ℚ) (debt_to_annika : ℚ),
  jerichos_money_problem jerichos_money debt_to_annika :=
by
  sorry

end jerichos_remaining_money_l758_75836


namespace right_triangle_shorter_leg_l758_75810

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a < b →            -- a is the shorter leg
  a ≤ b →            -- Ensure a is not equal to b
  a = 16 :=          -- The shorter leg is 16 units
by sorry

end right_triangle_shorter_leg_l758_75810


namespace line_satisfies_conditions_l758_75853

theorem line_satisfies_conditions : ∃! k : ℝ,
  let f (x : ℝ) := x^2 + 8*x + 7
  let g (x : ℝ) := 19.5*x - 32
  let p1 := (k, f k)
  let p2 := (k, g k)
  (g 2 = 7) ∧
  (abs (f k - g k) = 6) ∧
  (-32 ≠ 0) := by
  sorry

end line_satisfies_conditions_l758_75853


namespace max_uncovered_corridor_length_l758_75824

theorem max_uncovered_corridor_length 
  (corridor_length : ℝ) 
  (num_rugs : ℕ) 
  (total_rug_length : ℝ) 
  (h1 : corridor_length = 100)
  (h2 : num_rugs = 20)
  (h3 : total_rug_length = 1000) :
  (corridor_length - (total_rug_length - corridor_length)) ≤ 50 := by
sorry

end max_uncovered_corridor_length_l758_75824


namespace kenny_basketball_time_l758_75845

def trumpet_practice : ℕ := 40

theorem kenny_basketball_time (run_time trumpet_time basketball_time : ℕ) 
  (h1 : trumpet_time = trumpet_practice)
  (h2 : trumpet_time = 2 * run_time)
  (h3 : run_time = 2 * basketball_time) : 
  basketball_time = 10 := by
  sorry

end kenny_basketball_time_l758_75845


namespace oil_leak_total_l758_75811

/-- The total amount of oil leaked from four pipes -/
def total_oil_leaked (pipe_a_before pipe_a_during pipe_b_before pipe_b_during pipe_c_first pipe_c_second pipe_d_first pipe_d_second pipe_d_third : ℕ) : ℕ :=
  pipe_a_before + pipe_a_during + 
  pipe_b_before + pipe_b_during + 
  pipe_c_first + pipe_c_second + 
  pipe_d_first + pipe_d_second + pipe_d_third

/-- Theorem stating the total amount of oil leaked from the four pipes -/
theorem oil_leak_total : 
  total_oil_leaked 6522 5165 4378 3250 2897 7562 1789 3574 5110 = 40247 := by
  sorry

end oil_leak_total_l758_75811


namespace factors_of_1320_l758_75898

/-- The number of distinct, positive factors of 1320 -/
def num_factors_1320 : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem stating that the number of distinct, positive factors of 1320 is 32 -/
theorem factors_of_1320 : num_factors_1320 = 32 := by
  sorry

end factors_of_1320_l758_75898


namespace intersection_point_l758_75849

theorem intersection_point (x y : ℚ) : 
  (5 * x - 3 * y = 8) ∧ (4 * x + 2 * y = 20) ↔ x = 38/11 ∧ y = 34/11 := by
  sorry

end intersection_point_l758_75849


namespace arithmetic_sequence_problem_l758_75847

-- Define the arithmetic sequence
def a (n : ℕ) : ℝ := sorry

-- Define the sum of the first n terms
def S (n : ℕ) : ℝ := sorry

-- Theorem statement
theorem arithmetic_sequence_problem :
  (a 1 + a 2 = 10) ∧ (a 5 = a 3 + 4) →
  (∀ n : ℕ, a n = 2 * n + 2) ∧
  (∃! k : ℕ, k > 0 ∧ S (k + 1) < 2 * a k + a 2 ∧ k = 1) :=
by sorry

end arithmetic_sequence_problem_l758_75847


namespace prob_at_least_six_heads_in_eight_flips_prob_at_least_six_heads_in_eight_flips_proof_l758_75872

/-- The probability of getting at least 6 heads in 8 fair coin flips -/
theorem prob_at_least_six_heads_in_eight_flips : ℚ :=
  37 / 256

/-- Proof that the probability of getting at least 6 heads in 8 fair coin flips is 37/256 -/
theorem prob_at_least_six_heads_in_eight_flips_proof :
  prob_at_least_six_heads_in_eight_flips = 37 / 256 := by
  sorry

end prob_at_least_six_heads_in_eight_flips_prob_at_least_six_heads_in_eight_flips_proof_l758_75872


namespace odd_divisibility_l758_75873

def sum_of_powers (n : ℕ) : ℕ := (Finset.range (n - 1)).sum (λ k => k^n)

theorem odd_divisibility (n : ℕ) (h : n > 1) :
  n ∣ sum_of_powers n ↔ Odd n :=
sorry

end odd_divisibility_l758_75873


namespace tan_graph_product_l758_75863

theorem tan_graph_product (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * x) = 3 → x = π / 8) →
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + π / 2))) →
  a * b = 6 := by
sorry

end tan_graph_product_l758_75863


namespace arithmetic_sequence_properties_l758_75876

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  sum_1_5 : a 1 + a 5 = -20
  sum_3_8 : a 3 + a 8 = -10

/-- The general term of the sequence -/
def general_term (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  2 * n - 16

/-- The sum of the first n terms of the sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = general_term seq n) ∧
  (∃ n : ℕ, sum_n_terms seq n = -56 ∧ (n = 7 ∨ n = 8)) ∧
  (∀ m : ℕ, sum_n_terms seq m ≥ -56) :=
  sorry

end arithmetic_sequence_properties_l758_75876


namespace inequality_solution_l758_75885

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 1) + 10 / (x + 4) ≥ 3 / (x + 2)) ↔ 
  (x ∈ Set.Ioc (-4) (-1) ∪ Set.Ioi (-4/3)) :=
by sorry

end inequality_solution_l758_75885


namespace initial_amount_80_leads_to_128_each_l758_75869

/-- Represents the amount of money each person has at each stage -/
structure Money where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Performs the first transaction where A gives to B and C -/
def transaction1 (m : Money) : Money :=
  { a := m.a - m.b - m.c,
    b := 2 * m.b,
    c := 2 * m.c }

/-- Performs the second transaction where B gives to A and C -/
def transaction2 (m : Money) : Money :=
  { a := 2 * m.a,
    b := m.b - m.a - m.c,
    c := 2 * m.c }

/-- Performs the third transaction where C gives to A and B -/
def transaction3 (m : Money) : Money :=
  { a := 2 * m.a,
    b := 2 * m.b,
    c := m.c - m.a - m.b }

/-- The main theorem stating that if the initial amount for A is 80,
    after all transactions, each person will have 128 cents -/
theorem initial_amount_80_leads_to_128_each (m : Money)
    (h_total : m.a + m.b + m.c = 128 + 128 + 128)
    (h_initial_a : m.a = 80) :
    let m1 := transaction1 m
    let m2 := transaction2 m1
    let m3 := transaction3 m2
    m3.a = 128 ∧ m3.b = 128 ∧ m3.c = 128 := by
  sorry


end initial_amount_80_leads_to_128_each_l758_75869


namespace product_ab_l758_75899

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l758_75899


namespace intercepts_correct_l758_75862

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-intercept of the line -/
def y_intercept : ℝ := -5

/-- Theorem stating that the x-intercept and y-intercept are correct for the given line equation -/
theorem intercepts_correct : 
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept :=
sorry

end intercepts_correct_l758_75862


namespace puppies_given_away_l758_75866

theorem puppies_given_away (initial_puppies : ℝ) (current_puppies : ℕ) : 
  initial_puppies = 6.0 →
  current_puppies = 4 →
  initial_puppies - current_puppies = 2 := by
sorry

end puppies_given_away_l758_75866


namespace keith_attended_games_l758_75880

def total_games : ℕ := 8
def missed_games : ℕ := 4

theorem keith_attended_games :
  total_games - missed_games = 4 := by sorry

end keith_attended_games_l758_75880


namespace quadratic_symmetry_solution_set_l758_75851

theorem quadratic_symmetry_solution_set 
  (a b c m n p : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hm : m ≠ 0) 
  (hn : n ≠ 0) 
  (hp : p ≠ 0) : 
  let f := fun (x : ℝ) ↦ a * x^2 + b * x + c
  let solution_set := {x : ℝ | m * (f x)^2 + n * (f x) + p = 0}
  solution_set ≠ {1, 4, 16, 64} := by
  sorry

end quadratic_symmetry_solution_set_l758_75851


namespace max_log_product_l758_75897

theorem max_log_product (x y : ℝ) (hx : x > 1) (hy : y > 1) (h_sum : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) :
  (Real.log x / Real.log 10) * (Real.log y / Real.log 10) ≤ 4 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 1 ∧ y₀ > 1 ∧
    Real.log x₀ / Real.log 10 + Real.log y₀ / Real.log 10 = 4 ∧
    (Real.log x₀ / Real.log 10) * (Real.log y₀ / Real.log 10) = 4 :=
by sorry

end max_log_product_l758_75897


namespace pencils_per_row_l758_75832

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (h1 : total_pencils = 30) (h2 : num_rows = 6) :
  total_pencils / num_rows = 5 := by
  sorry

end pencils_per_row_l758_75832


namespace max_full_books_read_l758_75875

def pages_per_hour : ℕ := 120
def pages_per_book : ℕ := 360
def available_hours : ℕ := 8

def books_read : ℕ := available_hours * pages_per_hour / pages_per_book

theorem max_full_books_read :
  books_read = 2 :=
sorry

end max_full_books_read_l758_75875


namespace first_group_count_l758_75890

theorem first_group_count (avg_first : ℝ) (avg_second : ℝ) (count_second : ℕ) (avg_all : ℝ)
  (h1 : avg_first = 20)
  (h2 : avg_second = 30)
  (h3 : count_second = 20)
  (h4 : avg_all = 24) :
  ∃ (count_first : ℕ), 
    (count_first : ℝ) * avg_first + (count_second : ℝ) * avg_second = 
    (count_first + count_second : ℝ) * avg_all ∧ count_first = 30 := by
  sorry

end first_group_count_l758_75890


namespace diamond_self_not_always_zero_l758_75802

-- Define the diamond operator
def diamond (x y : ℝ) : ℝ := |x - 2*y|

-- Theorem stating that the statement "For all real x, x ◇ x = 0" is false
theorem diamond_self_not_always_zero : ¬ ∀ x : ℝ, diamond x x = 0 := by
  sorry

end diamond_self_not_always_zero_l758_75802
