import Mathlib

namespace last_digit_is_two_l1839_183995

/-- Represents a 2000-digit integer as a list of natural numbers -/
def LongInteger := List Nat

/-- Checks if two consecutive digits are divisible by 17 or 23 -/
def validPair (a b : Nat) : Prop := (a * 10 + b) % 17 = 0 ∨ (a * 10 + b) % 23 = 0

/-- Defines the properties of our specific 2000-digit integer -/
def SpecialInteger (n : LongInteger) : Prop :=
  n.length = 2000 ∧
  n.head? = some 3 ∧
  ∀ i, i < 1999 → validPair (n.get! i) (n.get! (i + 1))

theorem last_digit_is_two (n : LongInteger) (h : SpecialInteger n) : 
  n.getLast? = some 2 := by
  sorry

#check last_digit_is_two

end last_digit_is_two_l1839_183995


namespace books_per_continent_l1839_183952

theorem books_per_continent 
  (total_books : ℕ) 
  (num_continents : ℕ) 
  (h1 : total_books = 488) 
  (h2 : num_continents = 4) 
  (h3 : total_books % num_continents = 0) : 
  total_books / num_continents = 122 := by
sorry

end books_per_continent_l1839_183952


namespace ninth_grade_students_l1839_183936

/-- Proves that given a total of 50 students from three grades, with the seventh grade having 2x - 1 students and the eighth grade having x students, the number of students in the ninth grade is 51 - 3x. -/
theorem ninth_grade_students (x : ℕ) : 
  (50 : ℕ) = (2 * x - 1) + x + (51 - 3 * x) := by
  sorry

end ninth_grade_students_l1839_183936


namespace residual_analysis_characteristics_l1839_183996

/-- Represents a residual in a statistical model. -/
structure Residual where
  value : ℝ

/-- Represents a statistical analysis method. -/
structure AnalysisMethod where
  name : String
  uses_residuals : Bool
  judges_model_fitting : Bool
  identifies_suspicious_data : Bool

/-- Definition of residual analysis based on its characteristics. -/
def residual_analysis : AnalysisMethod :=
  { name := "residual analysis",
    uses_residuals := true,
    judges_model_fitting := true,
    identifies_suspicious_data := true }

/-- Theorem stating that the analysis method using residuals to judge model fitting
    and identify suspicious data is residual analysis. -/
theorem residual_analysis_characteristics :
  ∀ (method : AnalysisMethod),
    method.uses_residuals ∧
    method.judges_model_fitting ∧
    method.identifies_suspicious_data →
    method = residual_analysis :=
by sorry

end residual_analysis_characteristics_l1839_183996


namespace quadratic_inequality_solution_set_l1839_183994

theorem quadratic_inequality_solution_set (a b : ℝ) :
  (∀ x, x^2 + a*x + b > 0 ↔ x ∈ Set.Iio (-3) ∪ Set.Ioi 1) →
  (∀ x, a*x^2 + b*x - 2 < 0 ↔ x ∈ Set.Ioo (-1/2) 2) :=
by sorry

end quadratic_inequality_solution_set_l1839_183994


namespace class_grade_average_l1839_183916

theorem class_grade_average (n : ℕ) (h : n > 0) :
  let first_quarter := n / 4
  let remaining := n - first_quarter
  let first_quarter_avg := 92
  let remaining_avg := 76
  let total_sum := first_quarter * first_quarter_avg + remaining * remaining_avg
  (total_sum : ℚ) / n = 80 := by
sorry

end class_grade_average_l1839_183916


namespace distance_to_karasuk_is_210_l1839_183956

/-- The distance from Novosibirsk to Karasuk --/
def distance_to_karasuk : ℝ := 210

/-- The initial distance between the bus and the car --/
def initial_distance : ℝ := 70

/-- The distance the car travels after catching up with the bus --/
def car_distance_after_catchup : ℝ := 40

/-- The distance the bus travels after the car catches up --/
def bus_distance_after_catchup : ℝ := 20

/-- The speed of the bus --/
def bus_speed : ℝ := sorry

/-- The speed of the car --/
def car_speed : ℝ := sorry

/-- The time taken for the car to catch up with the bus --/
def catchup_time : ℝ := sorry

theorem distance_to_karasuk_is_210 :
  distance_to_karasuk = initial_distance + car_speed * catchup_time :=
by sorry

end distance_to_karasuk_is_210_l1839_183956


namespace loan_principal_calculation_l1839_183939

/-- Represents the loan with varying interest rates over time -/
structure Loan where
  principal : ℝ
  rate1 : ℝ
  rate2 : ℝ
  rate3 : ℝ
  period1 : ℝ
  period2 : ℝ
  period3 : ℝ

/-- Calculates the total interest for a given loan -/
def totalInterest (loan : Loan) : ℝ :=
  loan.principal * (loan.rate1 * loan.period1 + loan.rate2 * loan.period2 + loan.rate3 * loan.period3)

/-- Theorem stating that given the specific interest rates and periods, 
    if the total interest is 11400, then the principal is 12000 -/
theorem loan_principal_calculation (loan : Loan) 
  (h1 : loan.rate1 = 0.06)
  (h2 : loan.rate2 = 0.09)
  (h3 : loan.rate3 = 0.14)
  (h4 : loan.period1 = 2)
  (h5 : loan.period2 = 3)
  (h6 : loan.period3 = 4)
  (h7 : totalInterest loan = 11400) :
  loan.principal = 12000 := by
  sorry

#check loan_principal_calculation

end loan_principal_calculation_l1839_183939


namespace uncovered_area_of_box_l1839_183903

/-- Given a rectangular box with dimensions 4 inches by 6 inches and a square block with side length 4 inches placed inside, the uncovered area of the box is 8 square inches. -/
theorem uncovered_area_of_box (box_length : ℕ) (box_width : ℕ) (block_side : ℕ) : 
  box_length = 4 → box_width = 6 → block_side = 4 → 
  (box_length * box_width) - (block_side * block_side) = 8 := by
sorry

end uncovered_area_of_box_l1839_183903


namespace sum_after_2015_iterations_l1839_183920

/-- The process of adding digits and appending the sum -/
def process (n : ℕ) : ℕ := sorry

/-- The result of applying the process n times to the initial number -/
def iterate_process (initial : ℕ) (n : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_after_2015_iterations :
  sum_of_digits (iterate_process 2015 2015) = 8065 := by sorry

end sum_after_2015_iterations_l1839_183920


namespace sum_of_digits_square_of_nine_twos_l1839_183917

/-- The sum of digits of the square of a number consisting of n twos -/
def sum_of_digits_square_of_twos (n : ℕ) : ℕ := 2 * n^2

/-- The number of twos in our specific case -/
def num_twos : ℕ := 9

/-- Theorem: The sum of the digits of the square of a number consisting of 9 twos is 162 -/
theorem sum_of_digits_square_of_nine_twos :
  sum_of_digits_square_of_twos num_twos = 162 := by
  sorry

end sum_of_digits_square_of_nine_twos_l1839_183917


namespace students_who_got_off_l1839_183969

/-- Given a school bus scenario where some students get off at a stop, 
    this theorem proves the number of students who got off. -/
theorem students_who_got_off (initial : ℕ) (remaining : ℕ) 
  (h1 : initial = 10) (h2 : remaining = 7) : initial - remaining = 3 := by
  sorry

#check students_who_got_off

end students_who_got_off_l1839_183969


namespace same_terminal_side_l1839_183912

/-- Proves that 375° has the same terminal side as α = π/12 + 2kπ, where k is an integer -/
theorem same_terminal_side (k : ℤ) : ∃ (n : ℤ), 375 * π / 180 = π / 12 + 2 * k * π + 2 * n * π := by
  sorry

end same_terminal_side_l1839_183912


namespace matrix_inverse_proof_l1839_183992

theorem matrix_inverse_proof :
  let A : Matrix (Fin 4) (Fin 4) ℝ := !![2, -3, 0, 0;
                                       -4, 6, 0, 0;
                                        0, 0, 3, -5;
                                        0, 0, 1, -2]
  let M : Matrix (Fin 4) (Fin 4) ℝ := !![0, 0, 0.5, -0.5;
                                        0, 0, 0.5, -0.5;
                                        0, 0, 0.5, -0.5;
                                        0, 0, 0.5, -0.5]
  M * A = (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  sorry

end matrix_inverse_proof_l1839_183992


namespace exactly_one_divisible_by_3_5_7_l1839_183958

theorem exactly_one_divisible_by_3_5_7 :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 200 ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n :=
by sorry

end exactly_one_divisible_by_3_5_7_l1839_183958


namespace prob_sum_less_than_9_l1839_183966

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The maximum sum we're considering -/
def maxSum : ℕ := 9

/-- The set of possible outcomes when rolling two dice -/
def outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range sides) (Finset.range sides)

/-- The favorable outcomes (sum less than maxSum) -/
def favorableOutcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun p => p.1 + p.2 < maxSum)

/-- Probability of rolling a sum less than maxSum with two fair dice -/
theorem prob_sum_less_than_9 :
  (favorableOutcomes.card : ℚ) / (outcomes.card : ℚ) = 7 / 9 := by
  sorry

end prob_sum_less_than_9_l1839_183966


namespace symmetry_of_regular_polygons_l1839_183940

-- Define the types of polygons we're considering
inductive RegularPolygon
  | EquilateralTriangle
  | Square
  | RegularPentagon
  | RegularHexagon

-- Define the properties of symmetry
def isAxiSymmetric (p : RegularPolygon) : Prop :=
  match p with
  | RegularPolygon.EquilateralTriangle => true
  | RegularPolygon.Square => true
  | RegularPolygon.RegularPentagon => true
  | RegularPolygon.RegularHexagon => true

def isCentrallySymmetric (p : RegularPolygon) : Prop :=
  match p with
  | RegularPolygon.EquilateralTriangle => false
  | RegularPolygon.Square => true
  | RegularPolygon.RegularPentagon => false
  | RegularPolygon.RegularHexagon => true

-- Theorem statement
theorem symmetry_of_regular_polygons :
  ∀ p : RegularPolygon, 
    (isAxiSymmetric p ∧ isCentrallySymmetric p) ↔ 
    (p = RegularPolygon.Square ∨ p = RegularPolygon.RegularHexagon) :=
by sorry

end symmetry_of_regular_polygons_l1839_183940


namespace sin_300_degrees_l1839_183979

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l1839_183979


namespace cell_growth_l1839_183949

/-- The number of hours in 3 days and nights -/
def total_hours : ℕ := 72

/-- The number of hours required for one cell division -/
def division_time : ℕ := 12

/-- The initial number of cells -/
def initial_cells : ℕ := 2^10

/-- The number of cell divisions that occur in the given time period -/
def num_divisions : ℕ := total_hours / division_time

/-- The final number of cells after the given time period -/
def final_cells : ℕ := initial_cells * 2^num_divisions

theorem cell_growth :
  final_cells = 2^16 := by sorry

end cell_growth_l1839_183949


namespace solve_for_y_l1839_183933

theorem solve_for_y (x y : ℝ) (h1 : 3 * (x - y) = 18) (h2 : x + y = 20) : y = 7 := by
  sorry

end solve_for_y_l1839_183933


namespace last_digit_of_fraction_l1839_183962

/-- The last digit of the decimal expansion of 1 / (3^15 * 2^5) is 5 -/
theorem last_digit_of_fraction : ∃ (n : ℕ), (1 : ℚ) / (3^15 * 2^5) = n / 10 + 5 / 10^(n + 1) := by
  sorry

end last_digit_of_fraction_l1839_183962


namespace ellipse_parameter_sum_l1839_183991

-- Define the ellipse parameters
def F₁ : ℝ × ℝ := (0, 0)
def F₂ : ℝ × ℝ := (8, 0)
def distance_sum : ℝ := 10

-- Define the ellipse equation
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ∃ (h k a b : ℝ), 
    (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ∧
    (x - F₁.1)^2 + (y - F₁.2)^2 + (x - F₂.1)^2 + (y - F₂.2)^2 = distance_sum^2

-- Theorem statement
theorem ellipse_parameter_sum :
  ∃ (h k a b : ℝ),
    (∀ P, is_on_ellipse P → 
      (P.1 - h)^2 / a^2 + (P.2 - k)^2 / b^2 = 1) ∧
    h + k + a + b = 12 :=
sorry

end ellipse_parameter_sum_l1839_183991


namespace sodium_sulfate_decahydrate_weight_sodium_sulfate_decahydrate_weight_is_966_75_l1839_183972

/-- The molecular weight of 3 moles of Na2SO4·10H2O -/
theorem sodium_sulfate_decahydrate_weight : ℝ → ℝ → ℝ → ℝ → ℝ := 
  fun (na_weight : ℝ) (s_weight : ℝ) (o_weight : ℝ) (h_weight : ℝ) =>
  let mw := 2 * na_weight + s_weight + 14 * o_weight + 20 * h_weight
  3 * mw

/-- The molecular weight of 3 moles of Na2SO4·10H2O is 966.75 grams -/
theorem sodium_sulfate_decahydrate_weight_is_966_75 :
  sodium_sulfate_decahydrate_weight 22.99 32.07 16.00 1.01 = 966.75 := by
  sorry

end sodium_sulfate_decahydrate_weight_sodium_sulfate_decahydrate_weight_is_966_75_l1839_183972


namespace isosceles_triangle_base_length_l1839_183913

/-- An isosceles triangle with congruent sides of 8 cm and perimeter of 25 cm has a base of 9 cm -/
theorem isosceles_triangle_base_length : 
  ∀ (base congruent_side : ℝ),
  congruent_side = 8 →
  base + 2 * congruent_side = 25 →
  base = 9 := by
sorry

end isosceles_triangle_base_length_l1839_183913


namespace fifteen_percent_of_a_minus_70_l1839_183938

theorem fifteen_percent_of_a_minus_70 (a : ℝ) : (0.15 * a) - 70 = 0.15 * a - 70 := by
  sorry

end fifteen_percent_of_a_minus_70_l1839_183938


namespace complex_magnitude_l1839_183980

theorem complex_magnitude (x y : ℝ) (h : x * (1 + Complex.I) = 1 + y * Complex.I) : 
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
  sorry

end complex_magnitude_l1839_183980


namespace function_property_l1839_183910

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 2) + f x = 3) 
  (h2 : f 1 = 0) : 
  f 2023 = 3 := by
  sorry

end function_property_l1839_183910


namespace line_circle_intersection_l1839_183968

theorem line_circle_intersection (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = k * A.1 + 2) ∧ 
    (B.2 = k * B.1 + 2) ∧ 
    ((A.1 - 3)^2 + (A.2 - 1)^2 = 9) ∧ 
    ((B.1 - 3)^2 + (B.2 - 1)^2 = 9) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 32)) →
  (k = 0 ∨ k = -3/4) :=
sorry

end line_circle_intersection_l1839_183968


namespace problem_statement_l1839_183974

theorem problem_statement (a b : ℝ) (h1 : a * b = 3) (h2 : a - 2 * b = 5) :
  a^2 * b - 2 * a * b^2 = 15 := by
  sorry

end problem_statement_l1839_183974


namespace total_cost_calculation_l1839_183907

def shirt_price : ℝ := 50
def pants_price : ℝ := 40
def shoes_price : ℝ := 60
def shirt_discount : ℝ := 0.2
def shoes_discount : ℝ := 0.5
def sales_tax : ℝ := 0.08

def total_cost : ℝ :=
  let shirt_cost := 6 * shirt_price * (1 - shirt_discount)
  let pants_cost := 2 * pants_price
  let shoes_cost := 2 * shoes_price + shoes_price * (1 - shoes_discount)
  let subtotal := shirt_cost + pants_cost + shoes_cost
  subtotal * (1 + sales_tax)

theorem total_cost_calculation :
  total_cost = 507.60 := by sorry

end total_cost_calculation_l1839_183907


namespace stratified_sampling_group_D_l1839_183928

/-- Represents the number of districts in each group -/
structure GroupSizes :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (D : ℕ)

/-- Calculates the total number of districts -/
def total_districts (g : GroupSizes) : ℕ := g.A + g.B + g.C + g.D

/-- Calculates the number of districts to be selected from a group in stratified sampling -/
def stratified_sample (group_size : ℕ) (total : ℕ) (sample_size : ℕ) : ℚ :=
  (group_size : ℚ) / (total : ℚ) * (sample_size : ℚ)

theorem stratified_sampling_group_D :
  let groups : GroupSizes := ⟨4, 10, 16, 8⟩
  let total := total_districts groups
  let sample_size := 9
  stratified_sample groups.D total sample_size = 2 := by
  sorry

end stratified_sampling_group_D_l1839_183928


namespace units_digit_of_k_cubed_plus_five_to_k_l1839_183990

theorem units_digit_of_k_cubed_plus_five_to_k (k : ℕ) : 
  k = 2024^2 + 3^2024 → (k^3 + 5^k) % 10 = 8 := by
  sorry

end units_digit_of_k_cubed_plus_five_to_k_l1839_183990


namespace quadratic_equation_roots_l1839_183999

theorem quadratic_equation_roots (x : ℝ) :
  (x^2 - 2*x - 1 = 0) ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
sorry

end quadratic_equation_roots_l1839_183999


namespace angle_measure_l1839_183918

theorem angle_measure (x : ℝ) : 
  (180 - x = 3 * x - 10) → x = 47.5 := by
  sorry

end angle_measure_l1839_183918


namespace A_finish_work_l1839_183948

/-- The number of days it takes A to finish the work -/
def days_A : ℝ := 12

/-- The number of days it takes B to finish the work -/
def days_B : ℝ := 15

/-- The number of days B worked before leaving -/
def days_B_worked : ℝ := 10

/-- The number of days it takes A to finish the remaining work after B left -/
def days_A_remaining : ℝ := 4

/-- Theorem stating that A can finish the work in 12 days -/
theorem A_finish_work : 
  days_A = 12 :=
by sorry

end A_finish_work_l1839_183948


namespace no_solution_3a_squared_equals_b_squared_plus_1_l1839_183955

theorem no_solution_3a_squared_equals_b_squared_plus_1 :
  ¬ ∃ (a b : ℕ), 3 * a^2 = b^2 + 1 := by
  sorry

end no_solution_3a_squared_equals_b_squared_plus_1_l1839_183955


namespace min_value_on_line_l1839_183935

/-- Given a point A(m,n) on the line x + 2y = 1 where m > 0 and n > 0,
    the minimum value of 2/m + 1/n is 8 -/
theorem min_value_on_line (m n : ℝ) (h1 : m + 2*n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∀ (x y : ℝ), x + 2*y = 1 → x > 0 → y > 0 → 2/m + 1/n ≤ 2/x + 1/y :=
by sorry

end min_value_on_line_l1839_183935


namespace units_digit_17_pow_2023_l1839_183963

theorem units_digit_17_pow_2023 :
  ∃ (n : ℕ), n < 10 ∧ 17^2023 ≡ n [ZMOD 10] ∧ n = 3 := by sorry

end units_digit_17_pow_2023_l1839_183963


namespace article_original_price_l1839_183902

/-- Given an article with a discounted price after a 24% decrease, 
    prove that its original price was Rs. 1400. -/
theorem article_original_price (discounted_price : ℝ) : 
  discounted_price = 1064 → 
  ∃ (original_price : ℝ), 
    original_price * (1 - 0.24) = discounted_price ∧ 
    original_price = 1400 := by
  sorry

end article_original_price_l1839_183902


namespace bowl_water_percentage_l1839_183934

theorem bowl_water_percentage (x : ℝ) (h1 : x > 0) (h2 : x / 2 + 4 = 14) : 
  (14 / x) * 100 = 70 :=
sorry

end bowl_water_percentage_l1839_183934


namespace exam_average_l1839_183986

theorem exam_average (x : ℝ) : 
  (15 * x + 10 * 90) / 25 = 81 → x = 75 := by sorry

end exam_average_l1839_183986


namespace points_on_circle_l1839_183900

-- Define the points
variable (A B C X Y A' : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (acute_triangle : IsAcute A B C)
(X_side : DifferentSide X C (Line.throughPoints A B))
(Y_side : DifferentSide Y B (Line.throughPoints A C))
(BX_eq_AC : dist B X = dist A C)
(CY_eq_AB : dist C Y = dist A B)
(AX_eq_AY : dist A X = dist A Y)
(A'_reflection : IsReflection A A' (Perp.bisector B C))
(XY_diff_sides : DifferentSide X Y (Line.throughPoints A A'))

-- State the theorem
theorem points_on_circle :
  ∃ (O : EuclideanSpace ℝ (Fin 2)) (r : ℝ), 
    dist O A = r ∧ dist O A' = r ∧ dist O X = r ∧ dist O Y = r :=
sorry

end points_on_circle_l1839_183900


namespace count_multiples_of_three_l1839_183926

/-- An arithmetic sequence with first term 9 and 8th term 12 -/
structure ArithmeticSequence where
  a₁ : ℕ
  a₈ : ℕ
  h₁ : a₁ = 9
  h₈ : a₈ = 12

/-- The number of terms among the first 2015 that are multiples of 3 -/
def multiples_of_three (seq : ArithmeticSequence) : ℕ :=
  sorry

/-- The main theorem -/
theorem count_multiples_of_three (seq : ArithmeticSequence) :
  multiples_of_three seq = 288 := by
  sorry

end count_multiples_of_three_l1839_183926


namespace least_positive_integer_with_remainders_l1839_183981

theorem least_positive_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 2 ∧ 
  n % 6 = 3 ∧
  ∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → n ≤ m :=
by
  -- The proof goes here
  sorry

end least_positive_integer_with_remainders_l1839_183981


namespace x_minus_y_values_l1839_183927

theorem x_minus_y_values (x y : ℤ) (hx : x = -3) (hy : |y| = 2) : 
  x - y = -5 ∨ x - y = -1 := by sorry

end x_minus_y_values_l1839_183927


namespace expression_evaluation_l1839_183937

theorem expression_evaluation :
  let a : ℚ := 2
  let b : ℚ := -1/2
  let c : ℚ := -1
  a * b * c - (2 * a * b - (3 * a * b * c - b * c) + 4 * a * b * c) = 3/2 := by
sorry

end expression_evaluation_l1839_183937


namespace popsicle_sticks_remaining_l1839_183959

theorem popsicle_sticks_remaining (initial : Real) (given_away : Real) :
  initial = 63.0 →
  given_away = 50.0 →
  initial - given_away = 13.0 := by sorry

end popsicle_sticks_remaining_l1839_183959


namespace expression_simplification_l1839_183982

theorem expression_simplification (y : ℝ) :
  3 * y - 2 * y^2 + 4 - (5 - 3 * y + 2 * y^2 - y^3) = y^3 + 6 * y - 4 * y^2 - 1 := by
  sorry

end expression_simplification_l1839_183982


namespace triangle_problem_l1839_183908

theorem triangle_problem (A B C : ℝ) (m n : ℝ × ℝ) (AC : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  m = (Real.cos (A + π / 3), Real.sin (A + π / 3)) →
  n = (Real.cos B, Real.sin B) →
  m.1 * n.1 + m.2 * n.2 = 0 →
  Real.cos B = 3 / 5 →
  AC = 8 →
  A - B = π / 6 ∧ Real.sqrt ((4 * Real.sqrt 3 + 3) ^ 2) = 4 * Real.sqrt 3 + 3 := by
  sorry

end triangle_problem_l1839_183908


namespace gcd_3869_6497_l1839_183965

theorem gcd_3869_6497 : Nat.gcd 3869 6497 = 73 := by
  sorry

end gcd_3869_6497_l1839_183965


namespace unique_positive_integer_solution_l1839_183925

theorem unique_positive_integer_solution : 
  ∃! (z : ℕ), z > 0 ∧ (4 * z)^2 - z = 2345 :=
by
  use 7
  sorry

end unique_positive_integer_solution_l1839_183925


namespace inequality_sum_l1839_183953

theorem inequality_sum (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) : a + c > b + d := by
  sorry

end inequality_sum_l1839_183953


namespace cakes_baked_yesterday_prove_cakes_baked_yesterday_l1839_183961

def cakes_baked_today : ℕ := 5
def cakes_sold_dinner : ℕ := 6
def cakes_left : ℕ := 2

theorem cakes_baked_yesterday : ℕ :=
  cakes_sold_dinner - cakes_baked_today + cakes_left

theorem prove_cakes_baked_yesterday :
  cakes_baked_yesterday = 3 := by
  sorry

end cakes_baked_yesterday_prove_cakes_baked_yesterday_l1839_183961


namespace projection_periodicity_l1839_183906

/-- Regular n-gon with vertices A₁, A₂, ..., Aₙ -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Point on a side of the n-gon -/
structure PointOnSide (n : ℕ) where
  ngon : RegularNGon n
  side : Fin n
  point : ℝ × ℝ

/-- Projection function that maps Mᵢ to Mᵢ₊₁ -/
def project (n : ℕ) (m : PointOnSide n) : PointOnSide n :=
  sorry

/-- The k-th projection of a point -/
def kthProjection (n k : ℕ) (m : PointOnSide n) : PointOnSide n :=
  sorry

theorem projection_periodicity (n : ℕ) (m : PointOnSide n) :
  (n = 4 → kthProjection n 13 m = m) ∧
  (n = 6 → kthProjection n 13 m = m) ∧
  (n = 10 → kthProjection n 11 m = m) :=
sorry

end projection_periodicity_l1839_183906


namespace quadratic_equation_nonnegative_solutions_l1839_183983

theorem quadratic_equation_nonnegative_solutions :
  ∃! (n : ℕ), n^2 + 3*n - 18 = 0 :=
by sorry

end quadratic_equation_nonnegative_solutions_l1839_183983


namespace clown_balloons_l1839_183904

theorem clown_balloons (initial_balloons : ℕ) (additional_balloons : ℕ) 
  (h1 : initial_balloons = 47) 
  (h2 : additional_balloons = 13) : 
  initial_balloons + additional_balloons = 60 := by
  sorry

end clown_balloons_l1839_183904


namespace profit_percentage_is_30_percent_l1839_183942

/-- Calculate the percentage of profit given the cost price and selling price --/
def percentage_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that the percentage profit is 30% for the given prices --/
theorem profit_percentage_is_30_percent :
  percentage_profit 350 455 = 30 := by
  sorry

end profit_percentage_is_30_percent_l1839_183942


namespace lego_airplane_model_l1839_183909

theorem lego_airplane_model (total_legos : ℕ) (additional_legos : ℕ) (num_models : ℕ) :
  total_legos = 400 →
  additional_legos = 80 →
  num_models = 2 →
  (total_legos + additional_legos) / num_models = 240 :=
by sorry

end lego_airplane_model_l1839_183909


namespace x_condition_l1839_183915

theorem x_condition (x : ℝ) : |x - 1| + |x - 5| = 4 → 1 ≤ x ∧ x ≤ 5 := by
  sorry

end x_condition_l1839_183915


namespace complex_product_real_l1839_183957

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 1 + a * Complex.I
  let z₂ : ℂ := 3 + 2 * Complex.I
  (z₁ * z₂).im = 0 → a = -2/3 := by
sorry

end complex_product_real_l1839_183957


namespace quadratic_function_form_l1839_183989

/-- A quadratic function with two equal real roots and derivative 2x + 2 -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
  equal_roots : ∃ (r : ℝ), (∀ x, f x = 0 ↔ x = r)
  derivative : ∀ x, deriv f x = 2 * x + 2

/-- The quadratic function with the given properties is x^2 + 2x + 1 -/
theorem quadratic_function_form (qf : QuadraticFunction) : 
  ∀ x, qf.f x = x^2 + 2*x + 1 := by
  sorry

end quadratic_function_form_l1839_183989


namespace average_equation_l1839_183954

theorem average_equation (x y : ℚ) : 
  x = 50 / 11399 ∧ y = -11275 / 151 →
  (List.sum (List.range 150) + x + y) / 152 = 75 * x + y := by
  sorry

end average_equation_l1839_183954


namespace cylinder_surface_area_l1839_183984

/-- Given a rectangle with length 4π cm and width 2 cm that is rolled into a cylinder
    using the longer side as the circumference of the base, prove that the total
    surface area of the resulting cylinder is 16π cm². -/
theorem cylinder_surface_area (π : ℝ) (h : π > 0) :
  let rectangle_length : ℝ := 4 * π
  let rectangle_width : ℝ := 2
  let base_circumference : ℝ := rectangle_length
  let base_radius : ℝ := base_circumference / (2 * π)
  let cylinder_height : ℝ := rectangle_width
  let total_surface_area : ℝ := 2 * π * base_radius^2 + 2 * π * base_radius * cylinder_height
  total_surface_area = 16 * π :=
by sorry

end cylinder_surface_area_l1839_183984


namespace tangent_line_equation_l1839_183970

/-- The function f(x) = x³ - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The point A through which the tangent line passes --/
def A : ℝ × ℝ := (0, 16)

/-- The point of tangency M --/
def M : ℝ × ℝ := (-2, f (-2))

theorem tangent_line_equation :
  ∀ x y : ℝ, (9:ℝ)*x - y + 16 = 0 ↔ 
  (y - M.2 = f' M.1 * (x - M.1) ∧ f M.1 = M.2 ∧ A.2 - M.2 = f' M.1 * (A.1 - M.1)) :=
sorry

end tangent_line_equation_l1839_183970


namespace james_chores_time_l1839_183922

/-- Given James spends 3 hours vacuuming and 3 times as long on other chores,
    prove that he spends 12 hours in total on his chores. -/
theorem james_chores_time :
  let vacuuming_time : ℝ := 3
  let other_chores_factor : ℝ := 3
  let other_chores_time : ℝ := vacuuming_time * other_chores_factor
  let total_time : ℝ := vacuuming_time + other_chores_time
  total_time = 12 := by sorry

end james_chores_time_l1839_183922


namespace isosceles_right_triangle_property_l1839_183997

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  is_isosceles : (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

/-- Distance squared between two points -/
def dist_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

/-- The theorem to be proved -/
theorem isosceles_right_triangle_property (triangle : IsoscelesRightTriangle) :
  ∀ P : ℝ × ℝ, (P.2 = triangle.A.2 ∧ P.2 = triangle.B.2) →
    dist_squared P triangle.A + dist_squared P triangle.B = 2 * dist_squared P triangle.C :=
by sorry

end isosceles_right_triangle_property_l1839_183997


namespace equal_even_odd_probability_l1839_183919

/-- The number of dice being rolled -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def sides_per_die : ℕ := 6

/-- The probability of rolling an even number on a single die -/
def prob_even : ℚ := 1/2

/-- The probability of rolling an odd number on a single die -/
def prob_odd : ℚ := 1/2

/-- The number of ways to choose half the dice to show even numbers -/
def ways_to_choose_half : ℕ := Nat.choose num_dice (num_dice / 2)

/-- Theorem: The probability of rolling 8 six-sided dice and getting an equal number of even and odd results is 35/128 -/
theorem equal_even_odd_probability : 
  (ways_to_choose_half : ℚ) * prob_even^num_dice = 35/128 := by sorry

end equal_even_odd_probability_l1839_183919


namespace rectangle_square_ratio_l1839_183941

theorem rectangle_square_ratio (s a b : ℝ) (h1 : a * b = 2 * s ^ 2) (h2 : a = 2 * b) :
  a / s = 2 := by sorry

end rectangle_square_ratio_l1839_183941


namespace initial_milk_amount_l1839_183930

/-- Proves that the initial amount of milk is 10 liters given the conditions of the problem -/
theorem initial_milk_amount (initial_water_content : Real) 
                             (target_water_content : Real)
                             (pure_milk_added : Real) :
  initial_water_content = 0.05 →
  target_water_content = 0.02 →
  pure_milk_added = 15 →
  ∃ (initial_milk : Real),
    initial_milk * initial_water_content = 
      (initial_milk + pure_milk_added) * target_water_content ∧
    initial_milk = 10 := by
  sorry

end initial_milk_amount_l1839_183930


namespace uncovered_side_length_l1839_183971

/-- Represents a rectangular field with three sides fenced -/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing : ℝ

/-- The uncovered side of a fenced field is 20 feet given the conditions -/
theorem uncovered_side_length (field : FencedField)
  (h_area : field.area = 80)
  (h_fencing : field.fencing = 28)
  (h_rect_area : field.area = field.length * field.width)
  (h_fencing_sum : field.fencing = 2 * field.width + field.length) :
  field.length = 20 := by
  sorry

end uncovered_side_length_l1839_183971


namespace rectangle_area_l1839_183946

theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) :
  length = 4 * width →
  perimeter = 2 * (length + width) →
  perimeter = 200 →
  width * length = 1600 := by
sorry

end rectangle_area_l1839_183946


namespace customers_per_table_l1839_183911

theorem customers_per_table 
  (initial_customers : ℕ) 
  (left_customers : ℕ) 
  (num_tables : ℕ) 
  (h1 : initial_customers = 21)
  (h2 : left_customers = 12)
  (h3 : num_tables = 3)
  (h4 : num_tables > 0)
  : (initial_customers - left_customers) / num_tables = 3 := by
sorry

end customers_per_table_l1839_183911


namespace arithmetic_sequence_length_l1839_183988

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 6 4 n = 206 ∧ n = 51 := by
  sorry

end arithmetic_sequence_length_l1839_183988


namespace line_through_parabola_vertex_l1839_183923

/-- The number of real values of b for which the line y = 2x + b passes through the vertex of the parabola y = x^2 + b^2 + 1 is zero. -/
theorem line_through_parabola_vertex (b : ℝ) : ¬∃ b, 2 * 0 + b = 0^2 + b^2 + 1 := by
  sorry

end line_through_parabola_vertex_l1839_183923


namespace eagles_winning_percentage_min_additional_games_is_minimum_l1839_183998

/-- The minimum number of additional games needed for the Eagles to win at least 90% of all games -/
def min_additional_games : ℕ := 26

/-- The initial number of games played -/
def initial_games : ℕ := 4

/-- The initial number of games won by the Eagles -/
def initial_eagles_wins : ℕ := 1

/-- The minimum winning percentage required for the Eagles -/
def min_winning_percentage : ℚ := 9/10

theorem eagles_winning_percentage (M : ℕ) :
  (initial_eagles_wins + M : ℚ) / (initial_games + M) ≥ min_winning_percentage ↔ M ≥ min_additional_games :=
sorry

theorem min_additional_games_is_minimum :
  ∀ M : ℕ, M < min_additional_games →
    (initial_eagles_wins + M : ℚ) / (initial_games + M) < min_winning_percentage :=
sorry

end eagles_winning_percentage_min_additional_games_is_minimum_l1839_183998


namespace complex_modulus_theorem_l1839_183993

theorem complex_modulus_theorem (r : ℝ) (z : ℂ) 
  (h1 : |r| < 3) 
  (h2 : r ≠ 2) 
  (h3 : z + r * z⁻¹ = 2) : 
  Complex.abs z = 3 := by
  sorry

end complex_modulus_theorem_l1839_183993


namespace trapezoid_longer_base_l1839_183967

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  shorter_base : ℝ
  altitude : ℝ
  longer_base : ℝ
  area : ℝ

/-- The trapezoid satisfies the given conditions -/
def satisfies_conditions (t : Trapezoid) : Prop :=
  t.shorter_base = 5 ∧
  t.altitude = 7 ∧
  t.area = 63 ∧
  ∃ (d : ℝ), t.shorter_base = t.altitude - d ∧ t.longer_base = t.altitude + d

theorem trapezoid_longer_base (t : Trapezoid) 
  (h : satisfies_conditions t) : t.longer_base = 13 := by
  sorry

end trapezoid_longer_base_l1839_183967


namespace uncovered_side_length_l1839_183973

/-- A rectangular field with three sides fenced -/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing : ℝ

/-- The uncovered side of the field is 20 feet long -/
theorem uncovered_side_length (field : FencedField) 
  (h_area : field.area = 120)
  (h_fencing : field.fencing = 32)
  (h_rectangle : field.area = field.length * field.width)
  (h_fence_sides : field.fencing = field.length + 2 * field.width) :
  field.length = 20 := by
  sorry

end uncovered_side_length_l1839_183973


namespace problem_one_problem_two_l1839_183951

-- Problem 1
theorem problem_one : (9/4)^(3/2) - (-9.6)^0 - (27/8)^(2/3) + (3/2)^(-2) = 1/2 := by
  sorry

-- Problem 2
theorem problem_two (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end problem_one_problem_two_l1839_183951


namespace pure_imaginary_complex_number_l1839_183950

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 - 4*m + 3)
  (z.re = 0 ∧ z.im ≠ 0) → m = -1 := by
  sorry

end pure_imaginary_complex_number_l1839_183950


namespace unique_triple_sum_l1839_183976

theorem unique_triple_sum (x y z : ℕ) : 
  x ≤ y ∧ y ≤ z ∧ x^x + y^y + z^z = 3382 ↔ (x, y, z) = (1, 4, 5) :=
by sorry

end unique_triple_sum_l1839_183976


namespace hyperbola_line_intersection_l1839_183921

-- Define the hyperbola C
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the line l
def line (t : ℝ) (x y : ℝ) : Prop := x = t * y + 2

-- Define the condition for the circle with diameter MN passing through A(2,-2)
def circle_condition (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 - 2) * (x2 - 2) + (y1 + 2) * (y2 + 2) = 0

theorem hyperbola_line_intersection :
  (hyperbola 3 4) →
  (hyperbola (Real.sqrt 2) (Real.sqrt 2)) →
  ∀ t : ℝ,
    (∃ x1 y1 x2 y2 : ℝ,
      x1 ≠ x2 ∧
      hyperbola x1 y1 ∧
      hyperbola x2 y2 ∧
      line t x1 y1 ∧
      line t x2 y2 ∧
      circle_condition x1 y1 x2 y2) →
    (t = 1 ∨ t = 1/7) :=
by sorry

end hyperbola_line_intersection_l1839_183921


namespace circledTimes_calculation_l1839_183929

-- Define the ⊗ operation
def circledTimes (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem circledTimes_calculation :
  circledTimes (circledTimes 5 7) (circledTimes 4 2) = 1/3 := by
  sorry

end circledTimes_calculation_l1839_183929


namespace rectangular_box_surface_area_l1839_183914

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : a + b + c = 45) 
  (h2 : a^2 + b^2 + c^2 = 625) : 
  2 * (a * b + b * c + c * a) = 1400 := by
sorry

end rectangular_box_surface_area_l1839_183914


namespace equation_implication_l1839_183978

theorem equation_implication (x y : ℝ) :
  x^2 - 3*x*y + 2*y^2 + x - y = 0 →
  x^2 - 2*x*y + y^2 - 5*x + 7*y = 0 →
  x*y - 12*x + 15*y = 0 := by
  sorry

end equation_implication_l1839_183978


namespace range_of_a_l1839_183931

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a^2 - 1)*x + (a - 2)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ -2 < a ∧ a < 1 := by
  sorry

end range_of_a_l1839_183931


namespace cat_food_sale_calculation_l1839_183960

/-- Theorem: Cat Food Sale Calculation
Given:
- 20 people bought cat food
- First 8 customers bought 3 cases each
- Next 4 customers bought 2 cases each
- Last 8 customers bought 1 case each

Prove: The total number of cases of cat food sold is 40.
-/
theorem cat_food_sale_calculation (total_customers : Nat) 
  (first_group_size : Nat) (first_group_cases : Nat)
  (second_group_size : Nat) (second_group_cases : Nat)
  (third_group_size : Nat) (third_group_cases : Nat)
  (h1 : total_customers = 20)
  (h2 : first_group_size = 8)
  (h3 : first_group_cases = 3)
  (h4 : second_group_size = 4)
  (h5 : second_group_cases = 2)
  (h6 : third_group_size = 8)
  (h7 : third_group_cases = 1)
  (h8 : total_customers = first_group_size + second_group_size + third_group_size) :
  first_group_size * first_group_cases + 
  second_group_size * second_group_cases + 
  third_group_size * third_group_cases = 40 := by
  sorry

end cat_food_sale_calculation_l1839_183960


namespace area_of_trapezoid_l1839_183944

structure Triangle where
  area : ℝ

structure Trapezoid where
  area : ℝ

def isosceles_triangle (t : Triangle) : Prop := sorry

theorem area_of_trapezoid (PQR : Triangle) (smallest : Triangle) (QSTM : Trapezoid) :
  isosceles_triangle PQR →
  PQR.area = 100 →
  smallest.area = 2 →
  QSTM.area = 90 := by
  sorry

end area_of_trapezoid_l1839_183944


namespace no_x_with_both_rational_l1839_183985

theorem no_x_with_both_rational : ¬∃ x : ℝ, ∃ p q : ℚ, 
  (Real.sin x + Real.sqrt 2 = ↑p) ∧ (Real.cos x - Real.sqrt 2 = ↑q) := by
  sorry

end no_x_with_both_rational_l1839_183985


namespace average_weight_increase_l1839_183905

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 65 →
  new_weight = 98.6 →
  (new_weight - old_weight) / initial_count = 4.2 :=
by sorry

end average_weight_increase_l1839_183905


namespace collinear_necessary_not_sufficient_l1839_183947

/-- Four points in 3D space -/
structure FourPoints where
  p1 : ℝ × ℝ × ℝ
  p2 : ℝ × ℝ × ℝ
  p3 : ℝ × ℝ × ℝ
  p4 : ℝ × ℝ × ℝ

/-- Predicate: three of the four points lie on the same straight line -/
def threePointsCollinear (points : FourPoints) : Prop :=
  sorry

/-- Predicate: all four points lie on the same plane -/
def fourPointsCoplanar (points : FourPoints) : Prop :=
  sorry

/-- Theorem: Three points collinear is necessary but not sufficient for four points coplanar -/
theorem collinear_necessary_not_sufficient :
  (∀ points : FourPoints, fourPointsCoplanar points → threePointsCollinear points) ∧
  (∃ points : FourPoints, threePointsCollinear points ∧ ¬fourPointsCoplanar points) :=
sorry

end collinear_necessary_not_sufficient_l1839_183947


namespace not_divisible_by_seven_l1839_183943

theorem not_divisible_by_seven (a b : ℕ) : 
  ¬(7 ∣ (a * b)) → ¬(7 ∣ a ∨ 7 ∣ b) := by
  sorry

end not_divisible_by_seven_l1839_183943


namespace f_power_of_two_divides_l1839_183945

/-- f(d) is the smallest possible integer that has exactly d positive divisors -/
def f (d : ℕ) : ℕ := sorry

/-- Theorem: For every non-negative integer k, f(2^k) divides f(2^(k+1)) -/
theorem f_power_of_two_divides (k : ℕ) : 
  (f (2^k)) ∣ (f (2^(k+1))) := by sorry

end f_power_of_two_divides_l1839_183945


namespace pencil_box_theorems_l1839_183901

/-- Represents the number of pencils of each color in the box -/
structure PencilBox where
  blue : Nat
  red : Nat
  green : Nat
  yellow : Nat

/-- The initial state of the pencil box -/
def initialBox : PencilBox := {
  blue := 5,
  red := 9,
  green := 6,
  yellow := 4
}

/-- The minimum number of pencils to ensure at least one of each color -/
def minPencilsForAllColors (box : PencilBox) : Nat :=
  box.blue + box.red + box.green + box.yellow - 3

/-- The maximum number of pencils to ensure at least one of each color remains -/
def maxPencilsLeaveAllColors (box : PencilBox) : Nat :=
  min box.blue box.red |> min box.green |> min box.yellow |> (· - 1)

/-- The maximum number of pencils to ensure at least five red pencils remain -/
def maxPencilsLeaveFiveRed (box : PencilBox) : Nat :=
  max (box.red - 5) 0

theorem pencil_box_theorems (box : PencilBox := initialBox) :
  (minPencilsForAllColors box = 21) ∧
  (maxPencilsLeaveAllColors box = 3) ∧
  (maxPencilsLeaveFiveRed box = 4) := by
  sorry

end pencil_box_theorems_l1839_183901


namespace part_one_part_two_l1839_183932

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧ Real.cos t.B = 3/5

-- Part 1
theorem part_one (t : Triangle) (h : triangle_conditions t) (h_b : t.b = 4) :
  Real.sin t.A = 2/5 := by sorry

-- Part 2
theorem part_two (t : Triangle) (h : triangle_conditions t) 
  (h_area : (1/2) * t.a * t.c * Real.sin t.B = 4) :
  t.b = Real.sqrt 17 ∧ t.c = 5 := by sorry

end part_one_part_two_l1839_183932


namespace line_through_points_with_45_degree_inclination_l1839_183964

/-- Given a line passing through points P(-2, m) and Q(m, 4) with an inclination angle of 45°, prove that m = 1. -/
theorem line_through_points_with_45_degree_inclination (m : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    ((-2, m) ∈ line) ∧ 
    ((m, 4) ∈ line) ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → (y - m) = (x + 2))) → 
  m = 1 := by
sorry

end line_through_points_with_45_degree_inclination_l1839_183964


namespace students_in_all_classes_l1839_183987

theorem students_in_all_classes (total_students : ℕ) (drama_students : ℕ) (music_students : ℕ) (dance_students : ℕ) (students_in_two_plus : ℕ) :
  total_students = 25 →
  drama_students = 15 →
  music_students = 17 →
  dance_students = 11 →
  students_in_two_plus = 13 →
  ∃ (students_all_three : ℕ), students_all_three = 4 ∧
    students_all_three ≤ students_in_two_plus ∧
    students_all_three ≤ drama_students ∧
    students_all_three ≤ music_students ∧
    students_all_three ≤ dance_students :=
by
  sorry

end students_in_all_classes_l1839_183987


namespace picture_position_l1839_183924

theorem picture_position (wall_width picture_width shift : ℝ) 
  (hw : wall_width = 25)
  (hp : picture_width = 4)
  (hs : shift = 1) :
  let center := wall_width / 2
  let picture_center := center + shift
  let left_edge := picture_center - picture_width / 2
  left_edge = 11.5 := by sorry

end picture_position_l1839_183924


namespace inequality_system_solution_l1839_183977

theorem inequality_system_solution (x : ℝ) :
  (4 * x - 2 ≥ 3 * (x - 1)) ∧ ((x - 5) / 2 > x - 4) → -1 ≤ x ∧ x < 3 := by
  sorry

end inequality_system_solution_l1839_183977


namespace mean_home_runs_l1839_183975

def player_count : ℕ := 13
def total_home_runs : ℕ := 80

def home_run_distribution : List (ℕ × ℕ) :=
  [(5, 5), (5, 6), (1, 7), (1, 8), (1, 10)]

theorem mean_home_runs :
  (total_home_runs : ℚ) / player_count = 80 / 13 := by
  sorry

end mean_home_runs_l1839_183975
