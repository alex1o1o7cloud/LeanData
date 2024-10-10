import Mathlib

namespace triangle_count_is_38_l3938_393879

/-- Represents a rectangle with internal divisions as described in the problem -/
structure DividedRectangle where
  -- Add necessary fields to represent the rectangle and its divisions
  -- This is a simplified representation
  height : ℕ
  width : ℕ

/-- Counts the number of triangles in a DividedRectangle -/
def countTriangles (rect : DividedRectangle) : ℕ := sorry

/-- The specific rectangle from the problem -/
def problemRectangle : DividedRectangle := {
  height := 20,
  width := 30
}

/-- Theorem stating that the number of triangles in the problem rectangle is 38 -/
theorem triangle_count_is_38 : countTriangles problemRectangle = 38 := by sorry

end triangle_count_is_38_l3938_393879


namespace person_B_age_l3938_393809

theorem person_B_age (a b c d e f g : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  c = d / 2 →
  d = e - 3 →
  f = a * d →
  g = b + e →
  a + b + c + d + e + f + g = 292 →
  b = 14 := by
sorry

end person_B_age_l3938_393809


namespace imaginary_part_of_complex_fraction_l3938_393845

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i^2 = -1 →
  (2 * i / (1 - i)).im = 1 := by sorry

end imaginary_part_of_complex_fraction_l3938_393845


namespace complex_equation_solution_l3938_393893

theorem complex_equation_solution (z : ℂ) :
  z * Complex.I = 1 → z = -Complex.I :=
by sorry

end complex_equation_solution_l3938_393893


namespace triangle_side_ratio_sum_l3938_393885

theorem triangle_side_ratio_sum (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle : c^2 = a^2 + b^2 - a*b) :
  (a / (b + c) + b / (a + c)) = 1 := by
  sorry

end triangle_side_ratio_sum_l3938_393885


namespace mod_seventeen_problem_l3938_393859

theorem mod_seventeen_problem (n : ℕ) (h1 : n < 17) (h2 : (2 * n) % 17 = 1) :
  (3^n)^2 % 17 - 3 % 17 = 13 % 17 := by
  sorry

end mod_seventeen_problem_l3938_393859


namespace sufficient_condition_implies_a_range_l3938_393897

/-- Proposition p: A real number x satisfies the given inequalities -/
def p (x : ℝ) : Prop := 2 < x ∧ x < 3

/-- Proposition q: A real number x satisfies the given inequality -/
def q (x a : ℝ) : Prop := 2 * x^2 - 9 * x + a < 0

/-- Theorem stating that if p is a sufficient condition for q, then 7 ≤ a ≤ 8 -/
theorem sufficient_condition_implies_a_range (a : ℝ) : 
  (∀ x, p x → q x a) → 7 ≤ a ∧ a ≤ 8 := by
  sorry

end sufficient_condition_implies_a_range_l3938_393897


namespace painted_cubes_count_l3938_393826

/-- Represents the number of smaller cubes with a given number of painted faces -/
structure PaintedCubes :=
  (three : ℕ)
  (two : ℕ)
  (one : ℕ)

/-- Calculates the number of smaller cubes with different numbers of painted faces
    when a large cube is cut into smaller cubes -/
def countPaintedCubes (large_edge : ℕ) (small_edge : ℕ) : PaintedCubes :=
  sorry

/-- Theorem stating the correct number of painted smaller cubes for the given problem -/
theorem painted_cubes_count :
  countPaintedCubes 8 2 = PaintedCubes.mk 8 24 24 := by
  sorry

end painted_cubes_count_l3938_393826


namespace taehyung_current_age_l3938_393869

/-- Taehyung's age this year -/
def taehyung_age : ℕ := 9

/-- Taehyung's uncle's age this year -/
def uncle_age : ℕ := taehyung_age + 17

/-- The sum of Taehyung's and his uncle's ages four years later -/
def sum_ages_later : ℕ := (taehyung_age + 4) + (uncle_age + 4)

theorem taehyung_current_age :
  taehyung_age = 9 ∧ uncle_age = taehyung_age + 17 ∧ sum_ages_later = 43 :=
by sorry

end taehyung_current_age_l3938_393869


namespace range_of_a_l3938_393833

open Set Real

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | -2 - a < x ∧ x < a}

-- Define propositions p and q
def p (a : ℝ) : Prop := (1 : ℝ) ∈ A a
def q (a : ℝ) : Prop := (2 : ℝ) ∈ A a

-- Theorem statement
theorem range_of_a (a : ℝ) 
  (h1 : a > 0) 
  (h2 : p a ∨ q a) 
  (h3 : ¬(p a ∧ q a)) : 
  1 < a ∧ a ≤ 2 := by sorry

end range_of_a_l3938_393833


namespace polynomial_division_remainder_l3938_393851

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  (X^5 + 3*X^3 + 1 : Polynomial ℝ) = (X + 1)^2 * q + r ∧
  r.degree < 2 ∧
  r = 5*X + 9 := by
sorry

end polynomial_division_remainder_l3938_393851


namespace abs_nonnegative_rational_l3938_393848

theorem abs_nonnegative_rational (x : ℚ) : |x| ≥ 0 := by
  sorry

end abs_nonnegative_rational_l3938_393848


namespace median_of_dataset2_with_X_l3938_393801

def dataset1 : List ℕ := [15, 9, 11, 7]
def dataset2 : List ℕ := [10, 11, 14, 8]

def mode (l : List ℕ) : ℕ := sorry
def median (l : List ℕ) : ℚ := sorry

theorem median_of_dataset2_with_X (X : ℕ) : 
  mode (X :: dataset1) = 11 → median (X :: dataset2) = 11 := by sorry

end median_of_dataset2_with_X_l3938_393801


namespace thalassa_population_2050_l3938_393899

/-- The population growth factor for Thalassa every 30 years -/
def growth_factor : ℕ := 3

/-- The initial population of Thalassa in 1990 -/
def initial_population : ℕ := 300

/-- The number of 30-year periods between 1990 and 2050 -/
def num_periods : ℕ := 2

/-- The population of Thalassa in 2050 -/
def population_2050 : ℕ := initial_population * growth_factor ^ num_periods

theorem thalassa_population_2050 : population_2050 = 2700 := by
  sorry

end thalassa_population_2050_l3938_393899


namespace gcd_factorial_8_factorial_6_squared_l3938_393805

theorem gcd_factorial_8_factorial_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 2880 := by
  sorry

end gcd_factorial_8_factorial_6_squared_l3938_393805


namespace cups_per_girl_l3938_393818

/-- Given a class with students, boys, and girls, prove the number of cups each girl brought. -/
theorem cups_per_girl (total_students : ℕ) (num_boys : ℕ) (cups_per_boy : ℕ) (total_cups : ℕ)
  (h1 : total_students = 30)
  (h2 : num_boys = 10)
  (h3 : cups_per_boy = 5)
  (h4 : total_cups = 90)
  (h5 : total_students = num_boys + 2 * num_boys) :
  (total_cups - num_boys * cups_per_boy) / (total_students - num_boys) = 2 := by
  sorry

end cups_per_girl_l3938_393818


namespace ae_length_l3938_393858

-- Define the points
variable (A B C D E : Point)

-- Define the shapes
def is_isosceles_trapezoid (A B C E : Point) : Prop := sorry

def is_rectangle (A C D E : Point) : Prop := sorry

-- Define the lengths
def length (P Q : Point) : ℝ := sorry

-- State the theorem
theorem ae_length 
  (h1 : is_isosceles_trapezoid A B C E)
  (h2 : is_rectangle A C D E)
  (h3 : length A B = 10)
  (h4 : length E C = 20) :
  length A E = 20 := by sorry

end ae_length_l3938_393858


namespace no_nonzero_triple_sum_zero_l3938_393888

theorem no_nonzero_triple_sum_zero :
  ¬∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a = b + c ∧ b = c + a ∧ c = a + b ∧
    a + b + c = 0 := by
  sorry

end no_nonzero_triple_sum_zero_l3938_393888


namespace expression_evaluation_l3938_393844

theorem expression_evaluation :
  |(-Real.sqrt 2)| + (-2023)^(0 : ℕ) - 2 * Real.sin (45 * π / 180) - (1/2)⁻¹ = -1 := by
  sorry

end expression_evaluation_l3938_393844


namespace min_value_and_points_l3938_393847

theorem min_value_and_points (x y : ℝ) :
  (y - 1)^2 + (x + y - 3)^2 + (2*x + y - 6)^2 ≥ 1/6 ∧
  (∃ x y : ℝ, (y - 1)^2 + (x + y - 3)^2 + (2*x + y - 6)^2 = 1/6 ∧ 
   x = 5/2 ∧ y = 5/6) := by
  sorry

end min_value_and_points_l3938_393847


namespace largest_number_problem_l3938_393877

theorem largest_number_problem (A B C : ℝ) 
  (sum_eq : A + B + C = 50)
  (first_eq : A = 2 * B - 43)
  (third_eq : C = (1/2) * A + 5) :
  max A (max B C) = B ∧ B = 27.375 := by
  sorry

end largest_number_problem_l3938_393877


namespace min_value_expression_l3938_393803

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1 / x^2 + 1 / y^2 + 1 / (x * y) ≥ 3 := by
  sorry

end min_value_expression_l3938_393803


namespace negation_of_universal_proposition_l3938_393822

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + 1 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l3938_393822


namespace sum_of_roots_of_special_quadratic_l3938_393864

/-- A real quadratic trinomial -/
def QuadraticTrinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_roots_of_special_quadratic 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x : ℝ, QuadraticTrinomial a b c (x^3 + x) ≥ QuadraticTrinomial a b c (x^2 + 1)) →
  (-b / a = 4) :=
by sorry

end sum_of_roots_of_special_quadratic_l3938_393864


namespace red_lucky_stars_count_l3938_393890

theorem red_lucky_stars_count 
  (blue_count : ℕ) 
  (yellow_count : ℕ) 
  (red_count : ℕ) 
  (total_count : ℕ) 
  (pick_probability : ℚ) :
  blue_count = 20 →
  yellow_count = 15 →
  total_count = blue_count + yellow_count + red_count →
  pick_probability = 1/2 →
  (red_count : ℚ) / (total_count : ℚ) = pick_probability →
  red_count = 35 := by
sorry

end red_lucky_stars_count_l3938_393890


namespace page_lines_increase_l3938_393881

theorem page_lines_increase (original_lines : ℕ) 
  (h1 : (110 : ℝ) = 0.8461538461538461 * original_lines) 
  (h2 : original_lines + 110 = 240) : 
  (original_lines + 110 : ℕ) = 240 := by
  sorry

end page_lines_increase_l3938_393881


namespace waiter_earnings_proof_l3938_393892

/-- Calculates the waiter's earnings from tips given the total number of customers,
    number of non-tipping customers, and the tip amount from each tipping customer. -/
def waiterEarnings (totalCustomers nonTippingCustomers tipAmount : ℕ) : ℕ :=
  (totalCustomers - nonTippingCustomers) * tipAmount

/-- Proves that the waiter's earnings are $27 given the specific conditions -/
theorem waiter_earnings_proof :
  waiterEarnings 7 4 9 = 27 := by
  sorry

end waiter_earnings_proof_l3938_393892


namespace expression_evaluation_l3938_393820

theorem expression_evaluation (x : ℝ) (h : 2*x - 9 ≥ 0) :
  (3 + Real.sqrt (2*x - 9))^2 - 3*x = -x + 6*Real.sqrt (2*x - 9) :=
by sorry

end expression_evaluation_l3938_393820


namespace range_of_a_l3938_393802

-- Define the propositions p and q
def p (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), ∀ (m : ℝ),
    x₁^2 - m*x₁ - 1 = 0 ∧
    x₂^2 - m*x₂ - 1 = 0 ∧
    a^2 + 4*a - 3 ≤ |x₁ - x₂|

def q (a : ℝ) : Prop :=
  ∃ (x : ℝ), x^2 + 2*x + a < 0

-- Define the theorem
theorem range_of_a :
  ∀ (a : ℝ), (p a ∨ q a) ∧ ¬(p a ∧ q a) → a = 1 ∨ a < -5 := by
  sorry

end range_of_a_l3938_393802


namespace total_money_l3938_393856

theorem total_money (brad_money : ℚ) (josh_money : ℚ) (doug_money : ℚ) : 
  josh_money = 2 * brad_money →
  josh_money = (3 / 4) * doug_money →
  doug_money = 32 →
  brad_money + josh_money + doug_money = 68 := by
sorry

end total_money_l3938_393856


namespace bridge_lamps_l3938_393883

/-- The number of lamps on a bridge -/
def numLamps (bridgeLength : ℕ) (lampSpacing : ℕ) : ℕ :=
  bridgeLength / lampSpacing + 1

theorem bridge_lamps :
  let bridgeLength : ℕ := 30
  let lampSpacing : ℕ := 5
  numLamps bridgeLength lampSpacing = 7 := by
  sorry

end bridge_lamps_l3938_393883


namespace no_all_permutations_perfect_squares_l3938_393821

/-- A function that checks if a natural number has all non-zero digits -/
def allDigitsNonZero (n : ℕ) : Prop := sorry

/-- A function that generates all permutations of digits of a natural number -/
def digitPermutations (n : ℕ) : Set ℕ := sorry

/-- A function that checks if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := sorry

theorem no_all_permutations_perfect_squares :
  ∀ n : ℕ, n ≥ 10 → allDigitsNonZero n →
    ∃ m ∈ digitPermutations n, ¬ isPerfectSquare m :=
sorry

end no_all_permutations_perfect_squares_l3938_393821


namespace fourth_power_plus_64_solutions_l3938_393857

theorem fourth_power_plus_64_solutions :
  let solutions : Set ℂ := {2 + 2*I, -2 - 2*I, -2 + 2*I, 2 - 2*I}
  ∀ z : ℂ, z^4 + 64 = 0 ↔ z ∈ solutions :=
by
  sorry

end fourth_power_plus_64_solutions_l3938_393857


namespace solution_set_reciprocal_inequality_l3938_393868

theorem solution_set_reciprocal_inequality (x : ℝ) :
  {x : ℝ | 1 / x > 3} = Set.Ioo 0 (1 / 3) := by sorry

end solution_set_reciprocal_inequality_l3938_393868


namespace tangent_fifteen_degree_ratio_l3938_393876

theorem tangent_fifteen_degree_ratio :
  (1 - Real.tan (15 * π / 180)) / (1 + Real.tan (15 * π / 180)) = Real.sqrt 3 / 3 := by
sorry

end tangent_fifteen_degree_ratio_l3938_393876


namespace line_perpendicular_theorem_l3938_393867

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_theorem
  (a b : Line) (α β : Plane)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : contained a α)
  (h4 : perpendicularLP b β)
  (h5 : parallel α β) :
  perpendicular a b :=
sorry

end line_perpendicular_theorem_l3938_393867


namespace isosceles_obtuse_triangle_smallest_angle_l3938_393875

/-- An isosceles, obtuse triangle with one angle 80% larger than a right angle has two smallest angles measuring 9 degrees each. -/
theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →  -- sum of angles in a triangle
  a = b →  -- isosceles condition
  c = 90 + 0.8 * 90 →  -- largest angle is 80% larger than right angle
  a = 9 :=
by sorry

end isosceles_obtuse_triangle_smallest_angle_l3938_393875


namespace xanths_are_yelps_and_wicks_l3938_393839

-- Define the sets
variable (U : Type) -- Universe set
variable (Zorb Yelp Xanth Wick : Set U)

-- State the given conditions
variable (h1 : Zorb ⊆ Yelp)
variable (h2 : Xanth ⊆ Zorb)
variable (h3 : Xanth ⊆ Wick)

-- State the theorem to be proved
theorem xanths_are_yelps_and_wicks : Xanth ⊆ Yelp ∩ Wick := by
  sorry

end xanths_are_yelps_and_wicks_l3938_393839


namespace unique_prime_sum_10003_l3938_393819

/-- A function that returns the number of ways to write a given natural number as the sum of two primes -/
def countPrimeSumWays (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there is exactly one way to write 10003 as the sum of two primes -/
theorem unique_prime_sum_10003 : countPrimeSumWays 10003 = 1 := by
  sorry

end unique_prime_sum_10003_l3938_393819


namespace fifth_term_is_nine_l3938_393874

-- Define the sequence and its sum
def S (n : ℕ) : ℕ := n^2

-- Define the sequence term
def a (n : ℕ) : ℕ := S n - S (n-1)

-- Theorem statement
theorem fifth_term_is_nine : a 5 = 9 := by
  sorry

end fifth_term_is_nine_l3938_393874


namespace exponent_simplification_l3938_393814

theorem exponent_simplification (x : ℝ) : (x^5 * x^3) * x^2 = x^10 := by
  sorry

end exponent_simplification_l3938_393814


namespace problem_solution_l3938_393872

theorem problem_solution (x : ℝ) : 3 ∈ ({x + 2, x^2 + 2*x} : Set ℝ) → x = -3 := by
  sorry

end problem_solution_l3938_393872


namespace pentagon_fraction_sum_l3938_393840

theorem pentagon_fraction_sum (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) (h₅ : a₅ > 0) : 
  let s := a₁ + a₂ + a₃ + a₄ + a₅
  (a₁ / (s - a₁)) + (a₂ / (s - a₂)) + (a₃ / (s - a₃)) + (a₄ / (s - a₄)) + (a₅ / (s - a₅)) < 2 :=
by sorry

end pentagon_fraction_sum_l3938_393840


namespace hexagon_intersection_area_l3938_393825

-- Define the hexagon
structure Hexagon where
  area : ℝ
  is_regular : Prop

-- Define a function to calculate the expected value
def expected_intersection_area (H : Hexagon) : ℝ :=
  -- The actual calculation of the expected value
  12

-- The theorem to be proved
theorem hexagon_intersection_area (H : Hexagon) 
  (h1 : H.area = 360) 
  (h2 : H.is_regular) : 
  expected_intersection_area H = 12 := by
  sorry


end hexagon_intersection_area_l3938_393825


namespace owner_short_percentage_l3938_393866

/-- Calculates the percentage of tank price the owner is short of after selling goldfish --/
def percentage_short_of_tank_price (goldfish_buy_price goldfish_sell_price tank_cost : ℚ) 
                                   (goldfish_sold : ℕ) : ℚ :=
  let profit_per_goldfish := goldfish_sell_price - goldfish_buy_price
  let total_profit := profit_per_goldfish * goldfish_sold
  let amount_short := tank_cost - total_profit
  (amount_short / tank_cost) * 100

/-- Proves that the owner is short of 45% of the tank price --/
theorem owner_short_percentage (goldfish_buy_price goldfish_sell_price tank_cost : ℚ) 
                               (goldfish_sold : ℕ) :
  goldfish_buy_price = 25/100 →
  goldfish_sell_price = 75/100 →
  tank_cost = 100 →
  goldfish_sold = 110 →
  percentage_short_of_tank_price goldfish_buy_price goldfish_sell_price tank_cost goldfish_sold = 45 :=
by
  sorry

#eval percentage_short_of_tank_price (25/100) (75/100) 100 110

end owner_short_percentage_l3938_393866


namespace equation_solutions_l3938_393849

theorem equation_solutions (x : ℝ) : 
  1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8 ↔ 
  x = 7 ∨ x = -2 := by
sorry

end equation_solutions_l3938_393849


namespace xy_value_l3938_393896

theorem xy_value (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 - y^2 = 20) : x * y = -56/9 := by
  sorry

end xy_value_l3938_393896


namespace sqrt_equation_solution_l3938_393824

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end sqrt_equation_solution_l3938_393824


namespace equation_solutions_l3938_393873

theorem equation_solutions :
  ∀ x : ℝ, 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := by
  sorry

end equation_solutions_l3938_393873


namespace gmat_question_percentages_l3938_393863

theorem gmat_question_percentages
  (first_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : first_correct = 85)
  (h2 : neither_correct = 5)
  (h3 : both_correct = 60)
  : ∃ (second_correct : ℝ), second_correct = 70 :=
by sorry

end gmat_question_percentages_l3938_393863


namespace angle_terminal_side_problem_tan_equation_problem_l3938_393831

-- Problem 1
theorem angle_terminal_side_problem (α : Real) 
  (h : Real.tan α = -3/4 ∧ Real.sin α = 3/5) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (2019*π/2 - α) * Real.tan (9*π/2 + α)) = 9/20 := by sorry

-- Problem 2
theorem tan_equation_problem (x : Real) (h : Real.tan (π/4 + x) = 2018) :
  1 / Real.cos (2*x) + Real.tan (2*x) = 2018 := by sorry

end angle_terminal_side_problem_tan_equation_problem_l3938_393831


namespace range_of_m_value_of_m_l3938_393846

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - (2*m + 3)*x + m^2

-- Define the condition for distinct real roots
def has_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0

-- Part 1: Range of m
theorem range_of_m (m : ℝ) (h : has_distinct_real_roots m) : m > -3/4 := by
  sorry

-- Part 2: Value of m when 1/x₁ + 1/x₂ = 1
theorem value_of_m (m : ℝ) (h1 : has_distinct_real_roots m)
  (h2 : ∃ x₁ x₂ : ℝ, quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ 1/x₁ + 1/x₂ = 1) :
  m = 3 := by
  sorry

end range_of_m_value_of_m_l3938_393846


namespace isosceles_triangle_ratio_l3938_393816

/-- Two isosceles triangles with equal perimeters -/
structure IsoscelesTrianglePair :=
  (base₁ : ℝ)
  (leg₁ : ℝ)
  (base₂ : ℝ)
  (leg₂ : ℝ)
  (base₁_pos : 0 < base₁)
  (leg₁_pos : 0 < leg₁)
  (base₂_pos : 0 < base₂)
  (leg₂_pos : 0 < leg₂)
  (equal_perimeters : base₁ + 2 * leg₁ = base₂ + 2 * leg₂)
  (base_relation : base₂ = 1.15 * base₁)
  (leg_relation : leg₂ = 0.95 * leg₁)

/-- The ratio of the base to the leg of the first triangle is 2:3 -/
theorem isosceles_triangle_ratio (t : IsoscelesTrianglePair) : t.base₁ / t.leg₁ = 2 / 3 :=
sorry

end isosceles_triangle_ratio_l3938_393816


namespace expression_simplification_l3938_393813

theorem expression_simplification (x : ℝ) : 
  ((3 * x - 6) - 5 * x) / 3 = -2/3 * x - 2 := by sorry

end expression_simplification_l3938_393813


namespace sqrt_sum_to_fraction_l3938_393811

theorem sqrt_sum_to_fraction : 
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by sorry

end sqrt_sum_to_fraction_l3938_393811


namespace purse_wallet_cost_difference_l3938_393815

theorem purse_wallet_cost_difference (wallet_cost purse_cost : ℕ) : 
  wallet_cost = 22 →
  purse_cost < 4 * wallet_cost →
  wallet_cost + purse_cost = 107 →
  4 * wallet_cost - purse_cost = 3 :=
by
  sorry

end purse_wallet_cost_difference_l3938_393815


namespace smaller_pack_size_l3938_393835

/-- Represents the number of eggs in a package -/
structure EggPackage where
  size : ℕ

/-- Represents a purchase of eggs -/
structure EggPurchase where
  totalEggs : ℕ
  largePacks : ℕ
  smallPacks : ℕ
  largePackSize : ℕ
  smallPackSize : ℕ

/-- Defines a valid egg purchase -/
def isValidPurchase (p : EggPurchase) : Prop :=
  p.totalEggs = p.largePacks * p.largePackSize + p.smallPacks * p.smallPackSize

/-- Theorem: Given the conditions, the size of the smaller pack must be 24 eggs -/
theorem smaller_pack_size (p : EggPurchase) 
    (h1 : p.totalEggs = 79)
    (h2 : p.largePacks = 5)
    (h3 : p.largePackSize = 11)
    (h4 : isValidPurchase p) :
    p.smallPackSize = 24 := by
  sorry

#check smaller_pack_size

end smaller_pack_size_l3938_393835


namespace kindergarten_group_divisibility_l3938_393871

theorem kindergarten_group_divisibility (n : ℕ) (a : ℕ) (h1 : n = 3 * a / 2) 
  (h2 : a % 2 = 0) (h3 : a % 4 = 0) : n % 8 = 0 := by
  sorry

end kindergarten_group_divisibility_l3938_393871


namespace high_school_student_count_l3938_393889

theorem high_school_student_count :
  let total_students : ℕ := 325
  let glasses_percentage : ℚ := 40 / 100
  let non_glasses_count : ℕ := 195
  (1 - glasses_percentage) * total_students = non_glasses_count :=
by
  sorry

end high_school_student_count_l3938_393889


namespace triangle_area_l3938_393807

/-- The area of the triangle bounded by the x-axis and two lines -/
theorem triangle_area (line1 line2 : ℝ × ℝ → ℝ) : 
  (line1 = fun (x, y) ↦ x - 2*y - 4) →
  (line2 = fun (x, y) ↦ 2*x + y - 5) →
  (∃ x₁ x₂ y : ℝ, 
    line1 (x₁, 0) = 0 ∧ 
    line2 (x₂, 0) = 0 ∧ 
    line1 (x₁, y) = 0 ∧ 
    line2 (x₁, y) = 0 ∧ 
    y > 0) →
  (1/2 * (x₁ - x₂) * y = 9/20) :=
by sorry

end triangle_area_l3938_393807


namespace fourth_to_sixth_ratio_l3938_393812

structure MathClasses where
  fourth_level : ℕ
  sixth_level : ℕ
  seventh_level : ℕ
  total_students : ℕ

def MathClasses.valid (c : MathClasses) : Prop :=
  c.fourth_level = c.sixth_level ∧
  c.seventh_level = 2 * c.fourth_level ∧
  c.sixth_level = 40 ∧
  c.total_students = 520

theorem fourth_to_sixth_ratio (c : MathClasses) (h : c.valid) :
  c.fourth_level = c.sixth_level :=
by sorry

end fourth_to_sixth_ratio_l3938_393812


namespace inscribed_circles_radii_l3938_393817

/-- Three circles inscribed in a corner -/
structure InscribedCircles where
  r : ℝ  -- radius of small circle
  a : ℝ  -- distance from center of small circle to corner vertex
  x : ℝ  -- radius of medium circle
  y : ℝ  -- radius of large circle

/-- Conditions for the inscribed circles -/
def valid_inscribed_circles (c : InscribedCircles) : Prop :=
  c.r > 0 ∧ c.a > c.r ∧ c.x > c.r ∧ c.y > c.x

/-- Theorem stating the radii of medium and large circles -/
theorem inscribed_circles_radii (c : InscribedCircles) 
  (h : valid_inscribed_circles c) : 
  c.x = c.a * c.r / (c.a - c.r) ∧ 
  c.y = c.a^2 * c.r / (c.a - c.r)^2 :=
by sorry

end inscribed_circles_radii_l3938_393817


namespace cos_alpha_values_l3938_393895

theorem cos_alpha_values (α : Real) (h : Real.sin (Real.pi + α) = -3/5) :
  Real.cos α = 4/5 ∨ Real.cos α = -4/5 := by
  sorry

end cos_alpha_values_l3938_393895


namespace bodies_of_water_is_six_l3938_393837

/-- Represents the aquatic reserve scenario -/
structure AquaticReserve where
  total_fish : ℕ
  fish_per_body : ℕ
  h_total : total_fish = 1050
  h_per_body : fish_per_body = 175

/-- The number of bodies of water in the aquatic reserve -/
def bodies_of_water (reserve : AquaticReserve) : ℕ :=
  reserve.total_fish / reserve.fish_per_body

/-- Theorem stating that the number of bodies of water is 6 -/
theorem bodies_of_water_is_six (reserve : AquaticReserve) : 
  bodies_of_water reserve = 6 := by
  sorry


end bodies_of_water_is_six_l3938_393837


namespace log_equality_l3938_393843

theorem log_equality : Real.log 16 / Real.log 4096 = Real.log 4 / Real.log 64 := by sorry

end log_equality_l3938_393843


namespace ravi_money_l3938_393884

theorem ravi_money (ravi giri kiran : ℚ) : 
  (ravi / giri = 6 / 7) →
  (giri / kiran = 6 / 15) →
  (kiran = 105) →
  ravi = 36 := by
sorry

end ravi_money_l3938_393884


namespace soda_cans_problem_l3938_393882

/-- The number of cans Tim initially had -/
def initial_cans : ℕ := 22

/-- The number of cans Jeff took -/
def cans_taken : ℕ := 6

/-- The number of cans Tim had after Jeff took some -/
def cans_after_taken : ℕ := initial_cans - cans_taken

/-- The number of cans Tim bought -/
def cans_bought : ℕ := cans_after_taken / 2

/-- The final number of cans Tim had -/
def final_cans : ℕ := 24

theorem soda_cans_problem :
  cans_after_taken + cans_bought = final_cans :=
by sorry

end soda_cans_problem_l3938_393882


namespace paula_tickets_needed_l3938_393854

/-- Represents the number of times Paula wants to ride each attraction -/
structure RideFrequencies where
  goKarts : Nat
  bumperCars : Nat
  rollerCoaster : Nat
  ferrisWheel : Nat

/-- Represents the ticket cost for each attraction -/
structure TicketCosts where
  goKarts : Nat
  bumperCars : Nat
  rollerCoaster : Nat
  ferrisWheel : Nat

/-- Calculates the total number of tickets needed based on ride frequencies and ticket costs -/
def totalTicketsNeeded (freq : RideFrequencies) (costs : TicketCosts) : Nat :=
  freq.goKarts * costs.goKarts +
  freq.bumperCars * costs.bumperCars +
  freq.rollerCoaster * costs.rollerCoaster +
  freq.ferrisWheel * costs.ferrisWheel

/-- Theorem stating that Paula needs 52 tickets in total -/
theorem paula_tickets_needed :
  let frequencies : RideFrequencies := {
    goKarts := 2,
    bumperCars := 4,
    rollerCoaster := 3,
    ferrisWheel := 1
  }
  let costs : TicketCosts := {
    goKarts := 4,
    bumperCars := 5,
    rollerCoaster := 7,
    ferrisWheel := 3
  }
  totalTicketsNeeded frequencies costs = 52 := by
  sorry

end paula_tickets_needed_l3938_393854


namespace trailing_zeros_of_nine_to_999_plus_one_l3938_393865

theorem trailing_zeros_of_nine_to_999_plus_one :
  ∃ n : ℕ, (9^999 + 1 : ℕ) = 10 * n ∧ (9^999 + 1 : ℕ) % 100 ≠ 0 :=
by sorry

end trailing_zeros_of_nine_to_999_plus_one_l3938_393865


namespace line_tangent_to_ellipse_l3938_393808

theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 1 ∧ x^2 + 4 * y^2 = 1 → 
    ∃! p : ℝ × ℝ, p.1^2 + 4 * p.2^2 = 1 ∧ p.2 = m * p.1 + 1) →
  m^2 = 3/4 := by
sorry

end line_tangent_to_ellipse_l3938_393808


namespace contest_ranking_l3938_393891

theorem contest_ranking (A B C D : ℝ) 
  (sum_equal : A + B = C + D)
  (interchange : C + A > D + B)
  (bob_highest : B > A + D)
  (nonnegative : A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ D ≥ 0) :
  B > A ∧ A > C ∧ C > D := by
  sorry

end contest_ranking_l3938_393891


namespace job_applicant_age_range_l3938_393850

/-- The maximum number of different integer ages within a range defined by
    an average age and a number of standard deviations. -/
def max_different_ages (average_age : ℕ) (std_dev : ℕ) (num_std_devs : ℕ) : ℕ :=
  2 * num_std_devs * std_dev + 1

/-- Theorem stating that for the given problem parameters, 
    the maximum number of different ages is 41. -/
theorem job_applicant_age_range : 
  max_different_ages 40 10 2 = 41 := by
  sorry

end job_applicant_age_range_l3938_393850


namespace compare_complex_fractions_l3938_393827

theorem compare_complex_fractions : 
  1 / ((123^2 - 4) * 1375) > (7 / (5 * 9150625)) - (1 / (605 * 125^2)) := by
  sorry

end compare_complex_fractions_l3938_393827


namespace triangle_properties_l3938_393855

/-- Triangle ABC with given properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating properties of the specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.B = π/3)
  (h3 : Real.cos t.A = 2 * Real.sqrt 7 / 7) :
  t.c = 3 ∧ (1/2 * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 2) := by
  sorry


end triangle_properties_l3938_393855


namespace equal_area_necessary_not_sufficient_l3938_393898

-- Define a triangle type
structure Triangle where
  -- You might add more specific properties here, but for this problem we only need area
  area : ℝ

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem statement
theorem equal_area_necessary_not_sufficient :
  (∀ t1 t2 : Triangle, congruent t1 t2 → t1.area = t2.area) ∧
  (∃ t1 t2 : Triangle, t1.area = t2.area ∧ ¬congruent t1 t2) := by
  sorry

end equal_area_necessary_not_sufficient_l3938_393898


namespace concert_theorem_l3938_393829

def concert_probability (n : ℕ) (sure : ℕ) (unsure : ℕ) (p : ℚ) : ℚ :=
  sorry

theorem concert_theorem :
  let n : ℕ := 8
  let sure : ℕ := 4
  let unsure : ℕ := 4
  let p : ℚ := 1/3
  concert_probability n sure unsure p = 1 := by sorry

end concert_theorem_l3938_393829


namespace shifted_roots_polynomial_l3938_393838

theorem shifted_roots_polynomial (a b c : ℂ) : 
  (a^3 - 4*a^2 + 6*a - 3 = 0) →
  (b^3 - 4*b^2 + 6*b - 3 = 0) →
  (c^3 - 4*c^2 + 6*c - 3 = 0) →
  ∀ x, (x - (a + 3)) * (x - (b + 3)) * (x - (c + 3)) = x^3 - 13*x^2 + 57*x - 84 :=
by sorry

end shifted_roots_polynomial_l3938_393838


namespace birds_in_tree_l3938_393878

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29) 
  (h2 : final_birds = 42) : 
  final_birds - initial_birds = 13 := by
  sorry

end birds_in_tree_l3938_393878


namespace journey_speed_proof_l3938_393834

/-- Proves that the speed of the first half of a journey is 21 km/hr, given the total journey conditions -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 224 →
  total_time = 10 →
  second_half_speed = 24 →
  let first_half_distance : ℝ := total_distance / 2
  let second_half_time : ℝ := first_half_distance / second_half_speed
  let first_half_time : ℝ := total_time - second_half_time
  let first_half_speed : ℝ := first_half_distance / first_half_time
  first_half_speed = 21 := by
  sorry

end journey_speed_proof_l3938_393834


namespace cube_root_26_approximation_l3938_393828

theorem cube_root_26_approximation (ε : ℝ) (h : ε > 0) : 
  ∃ (x : ℝ), |x - (3 - 1/27)| < ε ∧ x^3 = 26 :=
sorry

end cube_root_26_approximation_l3938_393828


namespace whitewashing_cost_l3938_393862

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12
def door_height : ℝ := 6
def door_width : ℝ := 3
def window_height : ℝ := 4
def window_width : ℝ := 3
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 3

theorem whitewashing_cost :
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_height * door_width
  let window_area := num_windows * (window_height * window_width)
  let whitewash_area := wall_area - door_area - window_area
  whitewash_area * cost_per_sqft = 2718 :=
by sorry

end whitewashing_cost_l3938_393862


namespace valid_lineup_count_l3938_393870

def total_players : ℕ := 16
def num_starters : ℕ := 7
def num_triplets : ℕ := 3
def num_twins : ℕ := 2

def valid_lineups : ℕ := 8778

theorem valid_lineup_count :
  (Nat.choose total_players num_starters) -
  ((Nat.choose (total_players - num_triplets) (num_starters - num_triplets)) +
   (Nat.choose (total_players - num_twins) (num_starters - num_twins)) -
   (Nat.choose (total_players - num_triplets - num_twins) (num_starters - num_triplets - num_twins)))
  = valid_lineups := by sorry

end valid_lineup_count_l3938_393870


namespace polynomial_root_theorem_l3938_393830

theorem polynomial_root_theorem (a b : ℚ) :
  let f : ℝ → ℝ := fun x ↦ x^3 + a*x + b
  (f (4 - 2*Real.sqrt 5) = 0) →
  (∃ r : ℤ, f r = 0) →
  (∃ r : ℤ, f r = 0 ∧ r = -8) := by
sorry

end polynomial_root_theorem_l3938_393830


namespace money_distribution_l3938_393804

theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 500)
  (h2 : A + C = 200)
  (h3 : B + C = 310) :
  C = 10 := by
  sorry

end money_distribution_l3938_393804


namespace max_volume_at_one_cm_l3938_393823

/-- The side length of the original square sheet -/
def sheet_side : ℝ := 6

/-- The side length of the small square cut from each corner -/
def cut_side : ℝ := 1

/-- The volume of the box as a function of the cut side length -/
def box_volume (x : ℝ) : ℝ := x * (sheet_side - 2 * x)^2

theorem max_volume_at_one_cm :
  ∀ x, 0 < x → x < sheet_side / 2 → box_volume cut_side ≥ box_volume x :=
sorry

end max_volume_at_one_cm_l3938_393823


namespace odd_function_complete_expression_l3938_393894

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_positive (x : ℝ) : ℝ :=
  -x^2 + x + 1

theorem odd_function_complete_expression 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x, f x = if x > 0 then f_positive x
             else if x = 0 then 0
             else x^2 + x - 1 := by
  sorry

end odd_function_complete_expression_l3938_393894


namespace sixth_root_unity_product_l3938_393841

theorem sixth_root_unity_product (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 6 := by
  sorry

end sixth_root_unity_product_l3938_393841


namespace choose_one_book_from_specific_shelf_l3938_393842

/-- Represents a bookshelf with Chinese books on the upper shelf and math books on the lower shelf -/
structure Bookshelf :=
  (chinese_books : ℕ)
  (math_books : ℕ)

/-- Calculates the number of ways to choose one book from the bookshelf -/
def ways_to_choose_one_book (shelf : Bookshelf) : ℕ :=
  shelf.chinese_books + shelf.math_books

/-- Theorem stating that for a bookshelf with 5 Chinese books and 4 math books,
    the number of ways to choose one book is 9 -/
theorem choose_one_book_from_specific_shelf :
  let shelf : Bookshelf := ⟨5, 4⟩
  ways_to_choose_one_book shelf = 9 := by sorry

end choose_one_book_from_specific_shelf_l3938_393842


namespace special_function_unique_special_function_at_3_l3938_393853

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y)

/-- Theorem stating that any function satisfying the conditions must be f(x) = 2x -/
theorem special_function_unique (f : ℝ → ℝ) (h : special_function f) : 
  ∀ x : ℝ, f x = 2 * x :=
sorry

/-- Corollary: f(3) = 6 for any function satisfying the conditions -/
theorem special_function_at_3 (f : ℝ → ℝ) (h : special_function f) : 
  f 3 = 6 :=
sorry

end special_function_unique_special_function_at_3_l3938_393853


namespace g_composition_of_three_l3938_393836

def g (n : ℤ) : ℤ :=
  if n < 5 then n^2 + 2*n - 1 else 2*n + 3

theorem g_composition_of_three : g (g (g 3)) = 65 := by
  sorry

end g_composition_of_three_l3938_393836


namespace kenny_basketball_time_l3938_393886

-- Define variables for time spent on each activity
def trumpet_time : ℕ := 40
def running_time : ℕ := trumpet_time / 2
def basketball_time : ℕ := running_time / 2

-- Theorem to prove
theorem kenny_basketball_time : basketball_time = 10 := by
  sorry

end kenny_basketball_time_l3938_393886


namespace total_cookies_eq_eaten_plus_left_l3938_393880

/-- The number of cookies Mom made initially -/
def total_cookies : ℕ := sorry

/-- The number of cookies eaten by Julie and Matt -/
def cookies_eaten : ℕ := 9

/-- The number of cookies left after Julie and Matt ate -/
def cookies_left : ℕ := 23

/-- Theorem stating that the total number of cookies is the sum of eaten and left cookies -/
theorem total_cookies_eq_eaten_plus_left : 
  total_cookies = cookies_eaten + cookies_left := by sorry

end total_cookies_eq_eaten_plus_left_l3938_393880


namespace perpendicular_bisector_of_AB_l3938_393810

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Assume the circles intersect at A and B
axiom intersect_at_A : circle1 A.1 A.2 ∧ circle2 A.1 A.2
axiom intersect_at_B : circle1 B.1 B.2 ∧ circle2 B.1 B.2

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_of_AB :
  ∀ x y : ℝ, perpendicular_bisector x y ↔ 
  (x - A.1) * (B.1 - A.1) + (y - A.2) * (B.2 - A.2) = 0 ∧
  (x - (A.1 + B.1) / 2)^2 + (y - (A.2 + B.2) / 2)^2 = 
  ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4 :=
sorry

end perpendicular_bisector_of_AB_l3938_393810


namespace sum_of_special_sequence_l3938_393861

/-- Given positive real numbers a and b that form an arithmetic sequence with -2,
    and can also form a geometric sequence after rearrangement, prove their sum is 5 -/
theorem sum_of_special_sequence (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ d : ℝ, (a = b + d ∧ b = -2 + d) ∨ (b = a + d ∧ a = -2 + d) ∨ (a = -2 + d ∧ -2 = b + d)) →
  (∃ r : ℝ, r ≠ 0 ∧ ((a = b * r ∧ b = -2 * r) ∨ (b = a * r ∧ a = -2 * r) ∨ (a = -2 * r ∧ -2 = b * r))) →
  a + b = 5 := by
  sorry

end sum_of_special_sequence_l3938_393861


namespace negation_of_exists_ln_positive_l3938_393800

theorem negation_of_exists_ln_positive :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x > 0) ↔ (∀ x : ℝ, x > 0 → Real.log x ≤ 0) := by
  sorry

end negation_of_exists_ln_positive_l3938_393800


namespace price_reduction_doubles_profit_l3938_393860

-- Define the initial conditions
def initial_purchase_price : ℝ := 210
def initial_selling_price : ℝ := 270
def initial_daily_sales : ℝ := 30
def sales_increase_per_yuan : ℝ := 3

-- Define the profit function
def profit (price_reduction : ℝ) : ℝ :=
  let new_price := initial_selling_price - price_reduction
  let new_sales := initial_daily_sales + sales_increase_per_yuan * price_reduction
  (new_price - initial_purchase_price) * new_sales

-- Theorem statement
theorem price_reduction_doubles_profit :
  ∃ (price_reduction : ℝ), 
    price_reduction = 30 ∧ 
    profit price_reduction = 2 * profit 0 := by
  sorry

end price_reduction_doubles_profit_l3938_393860


namespace chord_bisected_at_P_l3938_393852

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 5 = 1

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define the chord equation
def chord_equation (x y : ℝ) : Prop := 5*x - 3*y - 13 = 0

-- Theorem statement
theorem chord_bisected_at_P :
  ∀ (A B : ℝ × ℝ),
  is_on_ellipse A.1 A.2 →
  is_on_ellipse B.1 B.2 →
  chord_equation A.1 A.2 →
  chord_equation B.1 B.2 →
  chord_equation P.1 P.2 →
  (A.1 + B.1) / 2 = P.1 ∧
  (A.2 + B.2) / 2 = P.2 :=
sorry

end chord_bisected_at_P_l3938_393852


namespace cubic_three_roots_l3938_393806

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x - a

/-- The derivative of f with respect to x -/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The second derivative of f with respect to x -/
def f'' (x : ℝ) : ℝ := 6*x

/-- The value of f at x = 1 -/
def f_at_1 (a : ℝ) : ℝ := -2 - a

/-- The value of f at x = -1 -/
def f_at_neg_1 (a : ℝ) : ℝ := 2 - a

/-- Theorem: The cubic function f(x) = x^3 - 3x - a has three distinct real roots 
    if and only if a is in the open interval (-2, 2) -/
theorem cubic_three_roots (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ 
  -2 < a ∧ a < 2 :=
sorry

end cubic_three_roots_l3938_393806


namespace fractional_exponent_simplification_l3938_393887

theorem fractional_exponent_simplification (a : ℝ) (ha : a > 0) :
  a^2 * Real.sqrt a = a^(5/2) := by
  sorry

end fractional_exponent_simplification_l3938_393887


namespace tangent_point_coordinates_l3938_393832

def f (x : ℝ) := x^4 - x

theorem tangent_point_coordinates :
  ∀ m n : ℝ,
  (∃ k : ℝ, f m = n ∧ 4 * m^3 - 1 = 3) →
  m = 1 ∧ n = 0 := by
sorry

end tangent_point_coordinates_l3938_393832
