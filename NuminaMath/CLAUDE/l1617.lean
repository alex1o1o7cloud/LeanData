import Mathlib

namespace binomial_30_3_l1617_161717

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l1617_161717


namespace max_abs_z_is_one_l1617_161716

theorem max_abs_z_is_one (a b c d z : ℂ) 
  (h1 : Complex.abs a = Complex.abs b)
  (h2 : Complex.abs b = Complex.abs c)
  (h3 : Complex.abs c = Complex.abs d)
  (h4 : Complex.abs a > 0)
  (h5 : a * z^3 + b * z^2 + c * z + d = 0) :
  Complex.abs z ≤ 1 := by
  sorry

end max_abs_z_is_one_l1617_161716


namespace length_of_AB_is_8_l1617_161798

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = -4*x ∧ x < 0

-- Define point P
def P : ℝ × ℝ := (-3, -2)

-- Define the line l passing through P
def l (x y : ℝ) : Prop := y + 2 = x + 3

-- Define the property of P being the midpoint of AB
def is_midpoint (A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Main theorem
theorem length_of_AB_is_8 :
  ∀ A B : ℝ × ℝ,
  C A.1 A.2 → C B.1 B.2 →
  l A.1 A.2 → l B.1 B.2 →
  is_midpoint A B →
  ‖A - B‖ = 8 :=
sorry

end length_of_AB_is_8_l1617_161798


namespace sum_sequence_equality_l1617_161767

theorem sum_sequence_equality (M : ℤ) : 
  1499 + 1497 + 1495 + 1493 + 1491 = 7500 - M → M = 25 := by
  sorry

end sum_sequence_equality_l1617_161767


namespace evolute_parabola_evolute_ellipse_l1617_161791

-- Part 1: Parabola
theorem evolute_parabola (x y X Y : ℝ) :
  x^2 = 2 * (1 - y) →
  27 * X^2 = -8 * Y^3 :=
sorry

-- Part 2: Ellipse
theorem evolute_ellipse (a b c t X Y : ℝ) :
  c^2 = a^2 - b^2 →
  X = -(c^2 / a) * (Real.cos t)^3 ∧
  Y = -(c^2 / b) * (Real.sin t)^3 :=
sorry

end evolute_parabola_evolute_ellipse_l1617_161791


namespace min_value_expression_l1617_161753

theorem min_value_expression (k : ℕ) (hk : k > 0) : 
  (10 : ℝ) / 3 + 32 / 10 ≤ (k : ℝ) / 3 + 32 / k :=
sorry

end min_value_expression_l1617_161753


namespace general_term_is_arithmetic_l1617_161789

-- Define the sequence a_n and its sum S_n
def S (n : ℕ) : ℕ := n^2 + 2*n

def a : ℕ → ℕ := fun n => S n - S (n-1)

-- Theorem 1: The general term of the sequence
theorem general_term : ∀ n : ℕ, n > 0 → a n = 2*n + 1 :=
sorry

-- Definition of arithmetic sequence
def is_arithmetic_sequence (f : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, n > 1 → f n - f (n-1) = d

-- Theorem 2: The sequence is arithmetic
theorem is_arithmetic : is_arithmetic_sequence a :=
sorry

end general_term_is_arithmetic_l1617_161789


namespace correct_scientific_notation_l1617_161792

/-- Scientific notation representation of a positive real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10
  normalize : coefficient * (10 : ℝ) ^ exponent = number

/-- The number we want to represent in scientific notation -/
def number : ℕ := 37600

/-- The scientific notation of the number -/
def scientific_notation : ScientificNotation where
  coefficient := 3.76
  exponent := 4
  coeff_range := by sorry
  normalize := by sorry

/-- Theorem stating that the given scientific notation is correct for the number -/
theorem correct_scientific_notation :
  scientific_notation.coefficient * (10 : ℝ) ^ scientific_notation.exponent = number := by sorry

end correct_scientific_notation_l1617_161792


namespace solve_equation_l1617_161706

theorem solve_equation : ∃ x : ℚ, (5*x + 8*x = 350 - 9*(x+8)) ∧ (x = 12 + 7/11) := by
  sorry

end solve_equation_l1617_161706


namespace exists_m_divides_sum_powers_l1617_161756

theorem exists_m_divides_sum_powers (n : ℕ+) :
  ∃ m : ℕ+, (7^n.val : ℤ) ∣ (3^m.val + 5^m.val - 1) := by sorry

end exists_m_divides_sum_powers_l1617_161756


namespace colonization_ways_l1617_161747

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 5

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := 6

/-- Represents the units required to colonize an Earth-like planet -/
def earth_like_units : ℕ := 2

/-- Represents the units required to colonize a Mars-like planet -/
def mars_like_units : ℕ := 1

/-- Represents the total available units for colonization -/
def total_units : ℕ := 14

/-- Theorem stating that there are exactly 20 different ways to occupy the planets -/
theorem colonization_ways : 
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 ≤ earth_like_planets ∧ 
    p.2 ≤ mars_like_planets ∧ 
    p.1 * earth_like_units + p.2 * mars_like_units = total_units)
  (Finset.product (Finset.range (earth_like_planets + 1)) (Finset.range (mars_like_planets + 1)))).card = 20 :=
sorry

end colonization_ways_l1617_161747


namespace probability_of_different_homes_l1617_161784

def num_volunteers : ℕ := 5
def num_homes : ℕ := 2

def probability_different_homes : ℚ := 8/15

theorem probability_of_different_homes :
  let total_arrangements := (2^num_volunteers - 2)
  let arrangements_same_home := (2^(num_volunteers - 2) - 1) * 2
  (total_arrangements - arrangements_same_home : ℚ) / total_arrangements = probability_different_homes :=
sorry

end probability_of_different_homes_l1617_161784


namespace average_of_quadratic_roots_l1617_161738

theorem average_of_quadratic_roots (p q : ℝ) (h : p ≠ 0) :
  let f : ℝ → ℝ := λ x => 3 * p * x^2 - 6 * p * x + q
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) → (x₁ + x₂) / 2 = 1 := by
  sorry

end average_of_quadratic_roots_l1617_161738


namespace darryl_has_twenty_books_l1617_161731

/-- Represents the number of books each friend has -/
structure BookCount where
  darryl : ℕ
  lamont : ℕ
  loris : ℕ

/-- The conditions of the problem -/
def BookProblem (bc : BookCount) : Prop :=
  bc.lamont = 2 * bc.darryl ∧
  bc.loris = bc.lamont - 3 ∧
  bc.darryl + bc.lamont + bc.loris = 97

/-- The theorem stating that Darryl has 20 books -/
theorem darryl_has_twenty_books :
  ∃ (bc : BookCount), BookProblem bc ∧ bc.darryl = 20 := by
  sorry

end darryl_has_twenty_books_l1617_161731


namespace waynes_blocks_l1617_161705

theorem waynes_blocks (initial_blocks additional_blocks : ℕ) 
  (h1 : initial_blocks = 9)
  (h2 : additional_blocks = 6) :
  initial_blocks + additional_blocks = 15 := by
  sorry

end waynes_blocks_l1617_161705


namespace jane_start_babysitting_age_l1617_161761

/-- Represents the age at which Jane started babysitting -/
def start_age : ℕ := 8

/-- Jane's current age -/
def current_age : ℕ := 32

/-- Years since Jane stopped babysitting -/
def years_since_stopped : ℕ := 10

/-- Current age of the oldest person Jane could have babysat -/
def oldest_babysat_current_age : ℕ := 24

/-- Theorem stating that Jane started babysitting at age 8 -/
theorem jane_start_babysitting_age :
  (start_age + years_since_stopped < current_age) ∧
  (∀ (jane_age : ℕ) (child_age : ℕ),
    jane_age ≥ start_age →
    jane_age ≤ current_age - years_since_stopped →
    child_age ≤ oldest_babysat_current_age - (current_age - jane_age) →
    child_age ≤ jane_age / 2) ∧
  (oldest_babysat_current_age = current_age - (start_age + 8)) :=
by sorry

#check jane_start_babysitting_age

end jane_start_babysitting_age_l1617_161761


namespace scientific_notation_correct_scientific_notation_format_l1617_161710

/-- Represents the value in billions -/
def billion_value : ℝ := 57.44

/-- Represents the coefficient in scientific notation -/
def scientific_coefficient : ℝ := 5.744

/-- Represents the exponent in scientific notation -/
def scientific_exponent : ℤ := 9

/-- Asserts that the scientific notation is correct for the given value -/
theorem scientific_notation_correct :
  billion_value * 10^9 = scientific_coefficient * 10^scientific_exponent :=
by sorry

/-- Asserts that the coefficient in scientific notation is between 1 and 10 -/
theorem scientific_notation_format :
  1 ≤ scientific_coefficient ∧ scientific_coefficient < 10 :=
by sorry

end scientific_notation_correct_scientific_notation_format_l1617_161710


namespace complex_number_forms_l1617_161783

theorem complex_number_forms (z : ℂ) : 
  z = 4 * (Complex.cos (4 * Real.pi / 3) + Complex.I * Complex.sin (4 * Real.pi / 3)) →
  (z = -2 - 2 * Complex.I * Real.sqrt 3) ∧ 
  (z = 4 * Complex.exp (Complex.I * (4 * Real.pi / 3))) := by
  sorry

end complex_number_forms_l1617_161783


namespace tan_alpha_two_implies_fraction_equals_four_l1617_161735

theorem tan_alpha_two_implies_fraction_equals_four (α : Real) 
  (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 := by
  sorry

end tan_alpha_two_implies_fraction_equals_four_l1617_161735


namespace f_not_monotonic_iff_t_in_range_l1617_161755

open Set
open Real

noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 4*x - 3 * Real.log x

def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a ≤ x ∧ x < y ∧ y ≤ b ∧ (f x < f y ∧ ∃ z, x < z ∧ z < y ∧ f z < f x) ∨
                               (f x > f y ∧ ∃ z, x < z ∧ z < y ∧ f z > f x)

theorem f_not_monotonic_iff_t_in_range (t : ℝ) :
  not_monotonic f t (t + 1) ↔ t ∈ Ioo 0 1 ∪ Ioo 2 3 := by
  sorry

end f_not_monotonic_iff_t_in_range_l1617_161755


namespace vehicle_value_last_year_l1617_161787

theorem vehicle_value_last_year 
  (value_this_year : ℝ)
  (ratio : ℝ)
  (h1 : value_this_year = 16000)
  (h2 : ratio = 0.8)
  (h3 : value_this_year = ratio * value_last_year) :
  value_last_year = 20000 := by
  sorry

end vehicle_value_last_year_l1617_161787


namespace intersection_point_is_solution_l1617_161745

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (3.5, -1.25)

/-- The first line equation -/
def line1 (x y : ℝ) : Prop := 5 * x - 2 * y = 20

/-- The second line equation -/
def line2 (x y : ℝ) : Prop := 3 * x + 2 * y = 8

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y :=
by sorry

end intersection_point_is_solution_l1617_161745


namespace gold_cube_buying_price_l1617_161775

/-- Proves that the buying price per gram of gold is $60 given the specified conditions --/
theorem gold_cube_buying_price (side_length : ℝ) (density : ℝ) (selling_factor : ℝ) (profit : ℝ) :
  side_length = 6 →
  density = 19 →
  selling_factor = 1.5 →
  profit = 123120 →
  let volume := side_length ^ 3
  let mass := density * volume
  let buying_price := profit / (selling_factor * mass - mass)
  buying_price = 60 := by
  sorry

end gold_cube_buying_price_l1617_161775


namespace window_screen_sales_l1617_161734

/-- Represents the monthly sales of window screens -/
structure MonthlySales where
  january : ℕ
  february : ℕ
  march : ℕ
  april : ℕ

/-- Calculates the total sales from January to April -/
def totalSales (sales : MonthlySales) : ℕ :=
  sales.january + sales.february + sales.march + sales.april

/-- Theorem stating the total sales given the conditions -/
theorem window_screen_sales : ∃ (sales : MonthlySales),
  sales.february = 2 * sales.january ∧
  sales.march = (5 / 4 : ℚ) * sales.february ∧
  sales.april = (9 / 10 : ℚ) * sales.march ∧
  sales.march = 12100 ∧
  totalSales sales = 37510 := by
  sorry


end window_screen_sales_l1617_161734


namespace solve_nested_equation_l1617_161713

theorem solve_nested_equation : ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 58 ∧ x = 19 := by
  sorry

end solve_nested_equation_l1617_161713


namespace tax_increase_l1617_161794

/-- Calculates the tax amount based on income and tax rates -/
def calculate_tax (income : ℕ) (rate1 : ℚ) (rate2 : ℚ) : ℚ :=
  if income ≤ 500000 then
    (income : ℚ) * rate1
  else if income ≤ 1000000 then
    500000 * rate1 + ((income - 500000) : ℚ) * rate2
  else
    500000 * rate1 + 500000 * rate2 + ((income - 1000000) : ℚ) * rate2

/-- Represents the tax system change and income increase -/
theorem tax_increase :
  let old_tax := calculate_tax 1000000 (20/100) (25/100)
  let new_main_tax := calculate_tax 1500000 (30/100) (35/100)
  let rental_income : ℚ := 100000
  let rental_deduction : ℚ := 10/100
  let taxable_rental := rental_income * (1 - rental_deduction)
  let rental_tax := taxable_rental * (35/100)
  let new_total_tax := new_main_tax + rental_tax
  new_total_tax - old_tax = 306500 := by sorry

end tax_increase_l1617_161794


namespace situps_total_l1617_161771

/-- The number of sit-ups Barney can do in one minute -/
def barney_situps : ℕ := 45

/-- The number of sit-ups Carrie can do in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can do in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The number of minutes Barney performs sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie performs sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie performs sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := barney_situps * barney_minutes + 
                        carrie_situps * carrie_minutes + 
                        jerrie_situps * jerrie_minutes

theorem situps_total : total_situps = 510 := by
  sorry

end situps_total_l1617_161771


namespace larger_screen_diagonal_l1617_161726

theorem larger_screen_diagonal (d : ℝ) : 
  d ^ 2 = 17 ^ 2 + 36 → d = 5 * Real.sqrt 13 := by
  sorry

end larger_screen_diagonal_l1617_161726


namespace total_ribbons_used_l1617_161757

def dresses_per_day_first_week : ℕ := 2
def days_first_week : ℕ := 7
def dresses_per_day_second_week : ℕ := 3
def days_second_week : ℕ := 2
def ribbons_per_dress : ℕ := 2

theorem total_ribbons_used : 
  (dresses_per_day_first_week * days_first_week + 
   dresses_per_day_second_week * days_second_week) * 
  ribbons_per_dress = 40 := by sorry

end total_ribbons_used_l1617_161757


namespace paper_tray_height_l1617_161708

/-- The height of a paper tray formed from a square sheet -/
theorem paper_tray_height (side_length : ℝ) (cut_start : ℝ) : 
  side_length = 120 →
  cut_start = Real.sqrt 20 →
  2 * Real.sqrt 5 = 
    (Real.sqrt 2 * cut_start) / Real.sqrt 2 :=
by sorry

end paper_tray_height_l1617_161708


namespace cubic_expansion_sum_l1617_161793

theorem cubic_expansion_sum (a a₁ a₂ a₃ : ℝ) :
  (∀ x, (2*x + 1)^3 = a + a₁*x + a₂*x^2 + a₃*x^3) →
  -a + a₁ - a₂ + a₃ = 1 := by
  sorry

end cubic_expansion_sum_l1617_161793


namespace string_length_problem_l1617_161720

/-- Given three strings A, B, and C, where the length of A is 6 times the length of C
    and 5 times the length of B, and the length of B is 12 meters,
    prove that the length of C is 10 meters. -/
theorem string_length_problem (A B C : ℝ) 
    (h1 : A = 6 * C) 
    (h2 : A = 5 * B) 
    (h3 : B = 12) : 
  C = 10 := by
  sorry

end string_length_problem_l1617_161720


namespace smallest_divisor_after_subtraction_l1617_161748

theorem smallest_divisor_after_subtraction (n m k : ℕ) (h1 : n = 899830) (h2 : m = 6) (h3 : k = 8) :
  k > m ∧
  (n - m) % k = 0 ∧
  ∀ d : ℕ, m < d ∧ d < k → (n - m) % d ≠ 0 :=
by sorry

end smallest_divisor_after_subtraction_l1617_161748


namespace complex_fourth_quadrant_l1617_161741

theorem complex_fourth_quadrant (a : ℝ) : 
  let z₁ : ℂ := 3 - a * Complex.I
  let z₂ : ℂ := 1 + 2 * Complex.I
  (z₁ / z₂).re > 0 ∧ (z₁ / z₂).im < 0 ↔ -6 < a ∧ a < 3/2 := by
sorry

end complex_fourth_quadrant_l1617_161741


namespace circle_tangency_distance_ratio_l1617_161799

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the distance function
variable (dist : Point → Point → ℝ)

-- Define the four circles
variable (A₁ A₂ A₃ A₄ : Circle)

-- Define the points
variable (P T₁ T₂ T₃ T₄ : Point)

-- Define the tangency and intersection relations
variable (tangent : Circle → Circle → Point → Prop)
variable (intersect : Circle → Circle → Point → Prop)

-- State the theorem
theorem circle_tangency_distance_ratio
  (h1 : tangent A₁ A₃ P)
  (h2 : tangent A₂ A₄ P)
  (h3 : intersect A₁ A₂ T₁)
  (h4 : intersect A₂ A₃ T₂)
  (h5 : intersect A₃ A₄ T₃)
  (h6 : intersect A₄ A₁ T₄)
  (h7 : T₁ ≠ P ∧ T₂ ≠ P ∧ T₃ ≠ P ∧ T₄ ≠ P) :
  (dist T₁ T₂ * dist T₂ T₃) / (dist T₁ T₄ * dist T₃ T₄) = (dist P T₂)^2 / (dist P T₄)^2 :=
sorry

end circle_tangency_distance_ratio_l1617_161799


namespace negative_three_inverse_l1617_161737

theorem negative_three_inverse : (-3 : ℚ)⁻¹ = -1/3 := by sorry

end negative_three_inverse_l1617_161737


namespace special_pair_example_special_pair_with_three_special_pair_negation_l1617_161721

/-- Definition of a special rational number pair -/
def is_special_pair (a b : ℚ) : Prop := a + b = a * b - 1

/-- Theorem 1: (5, 3/2) is a special rational number pair -/
theorem special_pair_example : is_special_pair 5 (3/2) := by sorry

/-- Theorem 2: If (a, 3) is a special rational number pair, then a = 2 -/
theorem special_pair_with_three (a : ℚ) : is_special_pair a 3 → a = 2 := by sorry

/-- Theorem 3: If (m, n) is a special rational number pair, then (-n, -m) is not a special rational number pair -/
theorem special_pair_negation (m n : ℚ) : is_special_pair m n → ¬ is_special_pair (-n) (-m) := by sorry

end special_pair_example_special_pair_with_three_special_pair_negation_l1617_161721


namespace guitar_picks_l1617_161760

theorem guitar_picks (total : ℕ) (red blue yellow : ℕ) : 
  2 * red = total →
  3 * blue = total →
  red + blue + yellow = total →
  blue = 12 →
  yellow = 6 := by
sorry

end guitar_picks_l1617_161760


namespace expected_points_is_16_l1617_161718

/-- The probability of a successful free throw -/
def free_throw_probability : ℝ := 0.8

/-- The number of free throw opportunities in a game -/
def opportunities : ℕ := 10

/-- The number of attempts per free throw opportunity -/
def attempts_per_opportunity : ℕ := 2

/-- The number of points awarded for each successful hit -/
def points_per_hit : ℕ := 1

/-- The expected number of points scored in a game -/
def expected_points : ℝ :=
  (opportunities : ℝ) * (attempts_per_opportunity : ℝ) * free_throw_probability * (points_per_hit : ℝ)

theorem expected_points_is_16 : expected_points = 16 := by sorry

end expected_points_is_16_l1617_161718


namespace students_not_in_chorus_or_band_l1617_161709

theorem students_not_in_chorus_or_band 
  (total : ℕ) (chorus : ℕ) (band : ℕ) (both : ℕ) 
  (h1 : total = 50) 
  (h2 : chorus = 18) 
  (h3 : band = 26) 
  (h4 : both = 2) : 
  total - (chorus + band - both) = 8 := by
  sorry

end students_not_in_chorus_or_band_l1617_161709


namespace parallel_vectors_k_value_l1617_161704

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

theorem parallel_vectors_k_value :
  ∀ (k : ℝ),
  (∃ (c : ℝ), c ≠ 0 ∧ (k * a.1 + b.1, k * a.2 + b.2) = c • (a.1 - 2 * b.1, a.2 - 2 * b.2)) →
  k = -1/2 := by
sorry

end parallel_vectors_k_value_l1617_161704


namespace solution_set_inequality_l1617_161768

theorem solution_set_inequality (a b : ℝ) :
  (∀ x, x^2 + a*x + b < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 + a*x + 1 > 0 ↔ x < 1/3 ∨ x > 1/2) :=
by sorry

end solution_set_inequality_l1617_161768


namespace probability_three_consecutive_beliy_naliv_l1617_161740

/-- The probability of selecting 3 "Beliy Naliv" bushes consecutively -/
def probability_three_consecutive (beliy_naliv : ℕ) (verlioka : ℕ) : ℚ :=
  (beliy_naliv / (beliy_naliv + verlioka)) *
  ((beliy_naliv - 1) / (beliy_naliv + verlioka - 1)) *
  ((beliy_naliv - 2) / (beliy_naliv + verlioka - 2))

/-- Theorem stating the probability of selecting 3 "Beliy Naliv" bushes consecutively is 1/8 -/
theorem probability_three_consecutive_beliy_naliv :
  probability_three_consecutive 9 7 = 1 / 8 := by
  sorry

end probability_three_consecutive_beliy_naliv_l1617_161740


namespace jose_join_time_l1617_161754

/-- Proves that Jose joined 2 months after Tom opened the shop given the investment and profit information -/
theorem jose_join_time (tom_investment : ℕ) (jose_investment : ℕ) (total_profit : ℕ) (jose_profit : ℕ) :
  tom_investment = 30000 →
  jose_investment = 45000 →
  total_profit = 36000 →
  jose_profit = 20000 →
  ∃ x : ℕ, 
    (tom_investment * 12) / (jose_investment * (12 - x)) = (total_profit - jose_profit) / jose_profit ∧
    x = 2 := by
  sorry

end jose_join_time_l1617_161754


namespace union_of_A_and_B_l1617_161772

def A : Set ℕ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℕ := {1, 2, 3}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by
  sorry

end union_of_A_and_B_l1617_161772


namespace three_common_tangents_implies_a_equals_9_l1617_161758

/-- Circle M with equation x^2 + y^2 - 4x + 3 = 0 -/
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

/-- Another circle with equation x^2 + y^2 - 4x - 6y + a = 0 -/
def other_circle (x y a : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + a = 0

/-- Theorem stating that if circle M has exactly three common tangent lines
    with the other circle, then a = 9 -/
theorem three_common_tangents_implies_a_equals_9 :
  ∀ a : ℝ, (∃! (l₁ l₂ l₃ : ℝ → ℝ → Prop),
    (∀ x y, circle_M x y → (l₁ x y ∨ l₂ x y ∨ l₃ x y)) ∧
    (∀ x y, other_circle x y a → (l₁ x y ∨ l₂ x y ∨ l₃ x y))) →
  a = 9 :=
sorry

end three_common_tangents_implies_a_equals_9_l1617_161758


namespace tommy_balloons_l1617_161711

/-- Given that Tommy had 26 balloons initially and received 34 more from his mom,
    prove that he ended up with 60 balloons in total. -/
theorem tommy_balloons (initial_balloons : ℕ) (mom_gift : ℕ) : 
  initial_balloons = 26 → mom_gift = 34 → initial_balloons + mom_gift = 60 := by
sorry

end tommy_balloons_l1617_161711


namespace ammonia_formed_l1617_161769

/-- Represents a chemical compound in the reaction -/
inductive Compound
| NH4NO3
| NaOH
| NH3
| H2O
| NaNO3

/-- Represents the stoichiometric coefficients in the balanced equation -/
def reaction_coefficients : Compound → ℕ
| Compound.NH4NO3 => 1
| Compound.NaOH => 1
| Compound.NH3 => 1
| Compound.H2O => 1
| Compound.NaNO3 => 1

/-- The number of moles of each reactant available -/
def available_moles : Compound → ℕ
| Compound.NH4NO3 => 2
| Compound.NaOH => 2
| _ => 0

/-- Theorem stating that 2 moles of NH3 are formed in the reaction -/
theorem ammonia_formed :
  let limiting_reactant := min (available_moles Compound.NH4NO3) (available_moles Compound.NaOH)
  limiting_reactant * (reaction_coefficients Compound.NH3) = 2 := by
  sorry

end ammonia_formed_l1617_161769


namespace absolute_value_equals_negative_l1617_161719

theorem absolute_value_equals_negative (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end absolute_value_equals_negative_l1617_161719


namespace combined_tax_rate_l1617_161739

/-- Calculates the combined tax rate for Mork and Mindy -/
theorem combined_tax_rate (mork_rate : ℚ) (mindy_rate : ℚ) (income_ratio : ℚ) : 
  mork_rate = 40 / 100 →
  mindy_rate = 30 / 100 →
  income_ratio = 3 →
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 325 / 1000 := by
  sorry

#eval (40 / 100 + 30 / 100 * 3) / (1 + 3)

end combined_tax_rate_l1617_161739


namespace breakfast_consumption_l1617_161788

/-- Represents the number of slices of bread each member consumes during breakfast -/
def breakfast_slices : ℕ := 3

/-- Represents the number of members in the household -/
def household_members : ℕ := 4

/-- Represents the number of slices each member consumes for snacks -/
def snack_slices : ℕ := 2

/-- Represents the number of slices in a loaf of bread -/
def slices_per_loaf : ℕ := 12

/-- Represents the number of loaves that last for 3 days -/
def loaves_for_three_days : ℕ := 5

/-- Represents the number of days the loaves last -/
def days_lasted : ℕ := 3

theorem breakfast_consumption :
  breakfast_slices = 3 ∧
  household_members * (breakfast_slices + snack_slices) * days_lasted = 
  loaves_for_three_days * slices_per_loaf := by
  sorry

#check breakfast_consumption

end breakfast_consumption_l1617_161788


namespace sum_of_special_system_l1617_161728

theorem sum_of_special_system (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (a + c)) :
  a + b + c = 1128 / 35 := by
  sorry

end sum_of_special_system_l1617_161728


namespace total_amount_proof_l1617_161730

/-- The total amount of money shared by Debby, Maggie, and Alex -/
def total : ℝ := 22500

/-- Debby's share percentage -/
def debby_share : ℝ := 0.30

/-- Maggie's share percentage -/
def maggie_share : ℝ := 0.40

/-- Alex's share percentage -/
def alex_share : ℝ := 0.30

/-- Maggie's actual share amount -/
def maggie_amount : ℝ := 9000

theorem total_amount_proof :
  maggie_share * total = maggie_amount ∧
  debby_share + maggie_share + alex_share = 1 :=
sorry

end total_amount_proof_l1617_161730


namespace arithmetic_geometric_sequence_problem_l1617_161725

/-- An arithmetic-geometric sequence -/
def ArithGeomSeq (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_geometric_sequence_problem (a : ℕ → ℝ) 
    (h_seq : ArithGeomSeq a)
    (h_first : a 1 = 3)
    (h_sum : a 1 + a 3 + a 5 = 21) :
    a 2 * a 6 = 72 := by
  sorry

end arithmetic_geometric_sequence_problem_l1617_161725


namespace bank_savings_exceed_target_l1617_161723

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

def initial_deposit := 5
def daily_ratio := 2
def target_amount := 5000  -- 50 dollars in cents

theorem bank_savings_exceed_target :
  ∃ n : ℕ, 
    n = 10 ∧ 
    geometric_sum initial_deposit daily_ratio n ≥ target_amount ∧
    ∀ m : ℕ, m < n → geometric_sum initial_deposit daily_ratio m < target_amount :=
by sorry

end bank_savings_exceed_target_l1617_161723


namespace greater_number_is_84_l1617_161727

theorem greater_number_is_84 (x y : ℝ) (h1 : x * y = 2688) (h2 : (x + y) - (x - y) = 64) : max x y = 84 := by
  sorry

end greater_number_is_84_l1617_161727


namespace lina_sticker_collection_l1617_161765

/-- The sum of an arithmetic sequence with first term a, common difference d, and n terms -/
def arithmeticSequenceSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Lina's sticker collection problem -/
theorem lina_sticker_collection :
  arithmeticSequenceSum 3 2 10 = 120 := by
  sorry

end lina_sticker_collection_l1617_161765


namespace possible_numbers_correct_l1617_161764

/-- Represents a digit on a seven-segment display -/
inductive Digit
| Zero | One | Two | Three | Four | Five | Six | Seven | Eight | Nine

/-- Represents a three-digit number -/
structure ThreeDigitNumber :=
(hundreds : Digit)
(tens : Digit)
(ones : Digit)

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  match n.hundreds, n.tens, n.ones with
  | Digit.Three, Digit.Five, Digit.One => 351
  | Digit.Three, Digit.Five, Digit.Four => 354
  | Digit.Three, Digit.Five, Digit.Seven => 357
  | Digit.Three, Digit.Six, Digit.One => 361
  | Digit.Three, Digit.Six, Digit.Seven => 367
  | Digit.Three, Digit.Eight, Digit.One => 381
  | Digit.Three, Digit.Nine, Digit.One => 391
  | Digit.Three, Digit.Nine, Digit.Seven => 397
  | Digit.Eight, Digit.Five, Digit.One => 851
  | Digit.Nine, Digit.Five, Digit.One => 951
  | Digit.Nine, Digit.Five, Digit.Seven => 957
  | Digit.Nine, Digit.Six, Digit.One => 961
  | Digit.Nine, Digit.Nine, Digit.One => 991
  | _, _, _ => 0

/-- The set of all possible original numbers -/
def possibleNumbers : Set Nat :=
  {351, 354, 357, 361, 367, 381, 391, 397, 851, 951, 957, 961, 991}

/-- Function to check if a number can be displayed as 351 with two malfunctioning segments -/
def canBeDisplayedAs351WithTwoMalfunctions (n : ThreeDigitNumber) : Prop :=
  ∃ (seg1 seg2 : Nat), seg1 ≠ seg2 ∧ seg1 < 7 ∧ seg2 < 7 ∧
    (n.toNat ∈ possibleNumbers)

/-- Theorem stating that the set of possible numbers is correct -/
theorem possible_numbers_correct :
  ∀ n : ThreeDigitNumber, canBeDisplayedAs351WithTwoMalfunctions n ↔ n.toNat ∈ possibleNumbers :=
sorry

end possible_numbers_correct_l1617_161764


namespace periodic_decimal_difference_l1617_161762

theorem periodic_decimal_difference : (4 : ℚ) / 11 - (9 : ℚ) / 25 = (1 : ℚ) / 275 := by sorry

end periodic_decimal_difference_l1617_161762


namespace water_flow_speed_equation_l1617_161722

/-- The speed of water flow in a river where two boats meet under specific conditions -/
def water_flow_speed : ℝ → Prop := λ V =>
  -- Speed of boat A in still water
  let speed_A : ℝ := 44
  -- Speed of boat B in still water
  let speed_B : ℝ := V^2
  -- Normal meeting time
  let normal_time : ℝ := 11
  -- Delayed meeting time
  let delayed_time : ℝ := 11.25
  -- Delay of boat B
  let delay : ℝ := 2/3
  -- Equation representing the scenario
  5 * V^2 - 8 * V - 132 = 0

theorem water_flow_speed_equation : ∃ V : ℝ, water_flow_speed V := by
  sorry

end water_flow_speed_equation_l1617_161722


namespace cat_litter_cost_l1617_161744

/-- Proves the cost of a cat litter container given specific conditions --/
theorem cat_litter_cost 
  (container_size : ℕ) 
  (litter_box_capacity : ℕ) 
  (change_frequency : ℕ) 
  (total_cost : ℕ) 
  (total_days : ℕ) 
  (h1 : container_size = 45)
  (h2 : litter_box_capacity = 15)
  (h3 : change_frequency = 7)
  (h4 : total_cost = 210)
  (h5 : total_days = 210) :
  total_cost / (total_days / change_frequency * litter_box_capacity / container_size) = 21 := by
  sorry


end cat_litter_cost_l1617_161744


namespace stones_division_impossible_l1617_161773

theorem stones_division_impossible (stones : List Nat) : 
  stones.length = 31 ∧ stones.sum = 660 → 
  ∃ (a b : Nat), a ∈ stones ∧ b ∈ stones ∧ a > 2 * b :=
by sorry

end stones_division_impossible_l1617_161773


namespace sum_equation_solution_l1617_161743

theorem sum_equation_solution (x : ℤ) : 
  (1 + 2 + 3 + 4 + 5 + x = 21 + 22 + 23 + 24 + 25) → x = 100 := by
  sorry

end sum_equation_solution_l1617_161743


namespace hannah_dolls_multiplier_l1617_161742

theorem hannah_dolls_multiplier (x : ℝ) : 
  x > 0 → -- Hannah has a positive number of times as many dolls
  8 * x + 8 = 48 → -- Total dolls equation
  x = 5 := by sorry

end hannah_dolls_multiplier_l1617_161742


namespace line_passes_through_fixed_point_l1617_161729

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (3*m + 2)*(-1) - (2*m - 1)*1 + 5*m + 1 = 0 := by
  sorry

end line_passes_through_fixed_point_l1617_161729


namespace min_xy_value_l1617_161777

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y + 6 = x*y) :
  x * y ≥ 18 :=
by sorry

end min_xy_value_l1617_161777


namespace factorization_validity_l1617_161781

theorem factorization_validity (x : ℝ) : x^2 - x - 6 = (x - 3) * (x + 2) := by
  sorry

#check factorization_validity

end factorization_validity_l1617_161781


namespace parallel_equal_segment_construction_l1617_161750

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a vector on a 2D grid -/
structure GridVector where
  dx : ℤ
  dy : ℤ

/-- Calculates the vector between two grid points -/
def vectorBetween (a b : GridPoint) : GridVector :=
  { dx := b.x - a.x, dy := b.y - a.y }

/-- Translates a point by a vector -/
def translatePoint (p : GridPoint) (v : GridVector) : GridPoint :=
  { x := p.x + v.dx, y := p.y + v.dy }

/-- Calculates the squared length of a vector -/
def vectorLengthSquared (v : GridVector) : ℤ :=
  v.dx * v.dx + v.dy * v.dy

theorem parallel_equal_segment_construction 
  (a b c : GridPoint) : 
  let v := vectorBetween a b
  let d := translatePoint c v
  (vectorBetween c d = v) ∧ 
  (vectorLengthSquared (vectorBetween a b) = vectorLengthSquared (vectorBetween c d)) :=
by sorry


end parallel_equal_segment_construction_l1617_161750


namespace parabola_rhombus_theorem_l1617_161746

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the rhombus
def rhombus (O B F C : ℝ × ℝ) : Prop :=
  let (xo, yo) := O
  let (xb, yb) := B
  let (xf, yf) := F
  let (xc, yc) := C
  (xf - xo)^2 + (yf - yo)^2 = (xb - xc)^2 + (yb - yc)^2 ∧
  (xb - xo)^2 + (yb - yo)^2 = (xc - xo)^2 + (yc - yo)^2

-- Define the theorem
theorem parabola_rhombus_theorem (p : ℝ) (O B F C : ℝ × ℝ) :
  parabola p B.1 B.2 →
  parabola p C.1 C.2 →
  rhombus O B F C →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 4 →
  p = Real.sqrt 2 := by sorry

end parabola_rhombus_theorem_l1617_161746


namespace sum_of_a_and_c_l1617_161796

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 48)
  (h2 : b + d = 6) : 
  a + c = 8 := by sorry

end sum_of_a_and_c_l1617_161796


namespace work_completion_proof_l1617_161776

/-- The number of days it takes for a and b to finish the work together -/
def combined_days : ℝ := 30

/-- The number of days it takes for a to finish the work alone -/
def a_alone_days : ℝ := 60

/-- The number of days a worked alone after b left -/
def a_remaining_days : ℝ := 20

/-- The number of days a and b worked together before b left -/
def days_worked_together : ℝ := 20

theorem work_completion_proof :
  days_worked_together * (1 / combined_days) + a_remaining_days * (1 / a_alone_days) = 1 :=
sorry

end work_completion_proof_l1617_161776


namespace tangent_slope_at_point_one_l1617_161733

def f (x : ℝ) : ℝ := 2 * x^3

theorem tangent_slope_at_point_one (x : ℝ) :
  HasDerivAt f 6 1 :=
sorry

end tangent_slope_at_point_one_l1617_161733


namespace bd_length_is_ten_l1617_161797

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the right angle at C
def isRightAngleAtC (t : Triangle) : Prop :=
  let (xa, ya) := t.A
  let (xb, yb) := t.B
  let (xc, yc) := t.C
  (xc - xa) * (xc - xb) + (yc - ya) * (yc - yb) = 0

-- Define the lengths of AC and BC
def AC_length (t : Triangle) : ℝ := 5
def BC_length (t : Triangle) : ℝ := 12

-- Define points D, E, F
def D (t : Triangle) : ℝ × ℝ := sorry
def E (t : Triangle) : ℝ × ℝ := sorry
def F (t : Triangle) : ℝ × ℝ := sorry

-- Define the right angle at FED
def isRightAngleAtFED (t : Triangle) : Prop :=
  let (xd, yd) := D t
  let (xe, ye) := E t
  let (xf, yf) := F t
  (xf - xd) * (xe - xd) + (yf - yd) * (ye - yd) = 0

-- Define the lengths of DE and DF
def DE_length (t : Triangle) : ℝ := 5
def DF_length (t : Triangle) : ℝ := 3

-- Define the length of BD
def BD_length (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem bd_length_is_ten (t : Triangle) :
  isRightAngleAtC t →
  isRightAngleAtFED t →
  BD_length t = 10 := by sorry

end bd_length_is_ten_l1617_161797


namespace base_n_representation_l1617_161780

theorem base_n_representation (n : ℕ) (a b : ℤ) : 
  n > 8 → 
  n^2 - a*n + b = 0 → 
  a = n + 8 → 
  b = 8*n := by
sorry

end base_n_representation_l1617_161780


namespace notebook_cost_proof_l1617_161790

theorem notebook_cost_proof :
  ∀ (s c n : ℕ),
    s ≤ 36 →                     -- number of students who bought notebooks
    s > 36 / 2 →                 -- at least half of the students
    n > 2 →                      -- more than 2 notebooks per student
    c > n →                      -- cost in cents greater than number of notebooks
    s * c * n = 3969 →           -- total cost in cents
    c = 27 :=                    -- cost per notebook is 27 cents
by sorry

end notebook_cost_proof_l1617_161790


namespace noemi_blackjack_loss_l1617_161707

/-- Calculates the amount lost on blackjack given initial amount, amount lost on roulette, and final amount -/
def blackjack_loss (initial : ℕ) (roulette_loss : ℕ) (final : ℕ) : ℕ :=
  initial - roulette_loss - final

/-- Proves that Noemi lost $500 on blackjack -/
theorem noemi_blackjack_loss :
  let initial := 1700
  let roulette_loss := 400
  let final := 800
  blackjack_loss initial roulette_loss final = 500 := by
  sorry

end noemi_blackjack_loss_l1617_161707


namespace arithmetic_expression_equality_l1617_161785

theorem arithmetic_expression_equality : 15 - 14 * 3 + 11 / 2 - 9 * 4 + 18 = -39.5 := by
  sorry

end arithmetic_expression_equality_l1617_161785


namespace sqrt_division_property_l1617_161702

theorem sqrt_division_property (x : ℝ) (hx : x > 0) : 2 * Real.sqrt x / Real.sqrt x = 2 := by
  sorry

end sqrt_division_property_l1617_161702


namespace investment_ratio_l1617_161770

/-- Given three investors A, B, and C with the following conditions:
  1. A invests the same amount as B
  2. A invests 2/3 of what C invests
  3. Total profit is 11000
  4. C's share of the profit is 3000
Prove that the ratio of A's investment to B's investment is 1:1 -/
theorem investment_ratio (a b c : ℝ) (h1 : a = b) (h2 : a = (2/3) * c)
  (total_profit : ℝ) (h3 : total_profit = 11000)
  (c_share : ℝ) (h4 : c_share = 3000) :
  a / b = 1 := by
  sorry

end investment_ratio_l1617_161770


namespace chicken_price_per_pound_l1617_161700

/-- The price per pound of chicken given the conditions of Alice's grocery shopping --/
theorem chicken_price_per_pound (min_spend : ℝ) (amount_needed : ℝ)
  (chicken_weight : ℝ) (lettuce_price : ℝ) (tomatoes_price : ℝ)
  (sweet_potato_price : ℝ) (sweet_potato_count : ℕ)
  (broccoli_price : ℝ) (broccoli_count : ℕ)
  (brussels_sprouts_price : ℝ) :
  min_spend = 35 →
  amount_needed = 11 →
  chicken_weight = 1.5 →
  lettuce_price = 3 →
  tomatoes_price = 2.5 →
  sweet_potato_price = 0.75 →
  sweet_potato_count = 4 →
  broccoli_price = 2 →
  broccoli_count = 2 →
  brussels_sprouts_price = 2.5 →
  (min_spend - amount_needed - (lettuce_price + tomatoes_price +
    sweet_potato_price * sweet_potato_count + broccoli_price * broccoli_count +
    brussels_sprouts_price)) / chicken_weight = 6 := by
  sorry

end chicken_price_per_pound_l1617_161700


namespace calculate_tax_rate_l1617_161732

/-- Given a total purchase amount, percentage of total spent on sales tax,
    and the cost of tax-free items, calculate the tax rate on taxable purchases. -/
theorem calculate_tax_rate (total_purchase : ℝ) (tax_percentage : ℝ) (tax_free_cost : ℝ) :
  total_purchase = 40 →
  tax_percentage = 30 →
  tax_free_cost = 34.7 →
  ∃ (tax_rate : ℝ), abs (tax_rate - 226.42) < 0.01 ∧
    tax_rate = (tax_percentage / 100 * total_purchase) / (total_purchase - tax_free_cost) * 100 :=
by sorry

end calculate_tax_rate_l1617_161732


namespace lemons_for_drinks_l1617_161736

/-- The number of lemons needed to make a certain amount of lemonade and lemon tea -/
def lemons_needed (lemonade_ratio : ℚ) (tea_ratio : ℚ) (lemonade_gallons : ℚ) (tea_gallons : ℚ) : ℚ :=
  lemonade_ratio * lemonade_gallons + tea_ratio * tea_gallons

/-- Theorem stating the number of lemons needed for 6 gallons of lemonade and 5 gallons of lemon tea -/
theorem lemons_for_drinks : 
  let lemonade_ratio : ℚ := 36 / 48
  let tea_ratio : ℚ := 20 / 10
  lemons_needed lemonade_ratio tea_ratio 6 5 = 29/2 := by
  sorry

#eval (29 : ℚ) / 2  -- To verify that 29/2 is indeed equal to 14.5

end lemons_for_drinks_l1617_161736


namespace proposition_equivalence_l1617_161778

theorem proposition_equivalence (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x - 4*a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) :=
by sorry

end proposition_equivalence_l1617_161778


namespace g_sum_property_l1617_161795

def g (x : ℝ) : ℝ := 3 * x^6 + 5 * x^4 - 6 * x^2 + 7

theorem g_sum_property : g 2 + g (-2) = 8 :=
by
  have h1 : g 2 = 4 := by sorry
  sorry

end g_sum_property_l1617_161795


namespace cubic_root_sum_inverse_squares_l1617_161714

theorem cubic_root_sum_inverse_squares : 
  ∀ (a b c : ℝ), 
  (a^3 - 6*a^2 - a + 3 = 0) → 
  (b^3 - 6*b^2 - b + 3 = 0) → 
  (c^3 - 6*c^2 - c + 3 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1/a^2 + 1/b^2 + 1/c^2 = 37/9) := by
sorry

end cubic_root_sum_inverse_squares_l1617_161714


namespace same_solution_implies_a_equals_four_l1617_161763

theorem same_solution_implies_a_equals_four (a : ℝ) : 
  (∃ x : ℝ, 2 * x + 1 = 3) →
  (∃ x : ℝ, 2 - (a - x) / 3 = 1) →
  (∀ x : ℝ, 2 * x + 1 = 3 ↔ 2 - (a - x) / 3 = 1) →
  a = 4 := by
sorry

end same_solution_implies_a_equals_four_l1617_161763


namespace cubic_polynomial_inequality_l1617_161759

/-- A cubic polynomial with real coefficients and three non-zero real roots satisfies the inequality 6a^3 + 10(a^2 - 2b)^(3/2) - 12ab ≥ 27c. -/
theorem cubic_polynomial_inequality (a b c : ℝ) (P : ℝ → ℝ) (h1 : P = fun x ↦ x^3 + a*x^2 + b*x + c) 
  (h2 : ∃ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ P x = 0 ∧ P y = 0 ∧ P z = 0) :
  6 * a^3 + 10 * (a^2 - 2*b)^(3/2) - 12 * a * b ≥ 27 * c := by
  sorry

end cubic_polynomial_inequality_l1617_161759


namespace rectangle_path_ratio_l1617_161715

/-- Represents a rectangle on a lattice grid --/
structure LatticeRectangle where
  width : ℕ
  height : ℕ

/-- Calculates the number of shortest paths between opposite corners of a rectangle --/
def shortestPaths (rect : LatticeRectangle) : ℕ :=
  Nat.choose (rect.width + rect.height) rect.width

/-- Theorem: For a rectangle with height = k * width, the number of paths starting vertically
    is k times the number of paths starting horizontally --/
theorem rectangle_path_ratio {k : ℕ} (rect : LatticeRectangle) 
    (h : rect.height = k * rect.width) :
  shortestPaths ⟨rect.height, rect.width⟩ = k * shortestPaths ⟨rect.width, rect.height⟩ := by
  sorry

#check rectangle_path_ratio

end rectangle_path_ratio_l1617_161715


namespace sandy_initial_money_l1617_161703

def sandy_shopping (initial_money : ℝ) : Prop :=
  let watch_price : ℝ := 50
  let shirt_price : ℝ := 30
  let shoes_price : ℝ := 70
  let shirt_discount : ℝ := 0.1
  let shoes_discount : ℝ := 0.2
  let spent_percentage : ℝ := 0.3
  let money_left : ℝ := 210
  
  let total_cost : ℝ := watch_price + 
    shirt_price * (1 - shirt_discount) + 
    shoes_price * (1 - shoes_discount)
  
  (initial_money * spent_percentage = total_cost) ∧
  (initial_money * (1 - spent_percentage) = money_left)

theorem sandy_initial_money :
  sandy_shopping 300 := by sorry

end sandy_initial_money_l1617_161703


namespace cubic_square_inequality_l1617_161724

theorem cubic_square_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^3 + b^3) / (a^2 + b^2) ≥ Real.sqrt (a * b) := by
  sorry

end cubic_square_inequality_l1617_161724


namespace polygon_exterior_interior_sum_equal_l1617_161712

theorem polygon_exterior_interior_sum_equal (n : ℕ) (h : n > 2) :
  (n - 2) * 180 = 360 → n = 4 := by
  sorry

end polygon_exterior_interior_sum_equal_l1617_161712


namespace ellipse_major_axis_length_l1617_161749

/-- Given an ellipse mx^2 + y^2 = 1 with eccentricity √3/2, its major axis length is either 2 or 4 -/
theorem ellipse_major_axis_length (m : ℝ) :
  (∃ (x y : ℝ), m * x^2 + y^2 = 1) →  -- Ellipse equation
  (∃ (a b : ℝ), a > b ∧ a^2 * m = b^2 ∧ (a^2 - b^2) / a^2 = 3/4) →  -- Eccentricity condition
  (∃ (l : ℝ), l = 2 ∨ l = 4 ∧ l = 2 * a) :=  -- Major axis length
by sorry

end ellipse_major_axis_length_l1617_161749


namespace max_value_of_trig_expression_l1617_161779

theorem max_value_of_trig_expression :
  ∀ α : Real, 0 ≤ α ∧ α ≤ π / 2 →
    (∀ β : Real, 0 ≤ β ∧ β ≤ π / 2 → 
      1 / (Real.sin β ^ 6 + Real.cos β ^ 6) ≤ 1 / (Real.sin α ^ 6 + Real.cos α ^ 6)) →
    1 / (Real.sin α ^ 6 + Real.cos α ^ 6) = 4 :=
by sorry

end max_value_of_trig_expression_l1617_161779


namespace subtraction_of_decimals_l1617_161774

theorem subtraction_of_decimals : 3.57 - 1.14 - 0.23 = 2.20 := by
  sorry

end subtraction_of_decimals_l1617_161774


namespace range_when_a_b_one_values_of_a_b_for_range_zero_one_max_min_a_squared_plus_b_squared_l1617_161751

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Part I(i)
theorem range_when_a_b_one (x : ℝ) (h : x ∈ Set.Icc 0 1) :
  f 1 1 x ∈ Set.Icc 1 3 :=
sorry

-- Part I(ii)
theorem values_of_a_b_for_range_zero_one :
  (∀ x ∈ Set.Icc 0 1, f a b x ∈ Set.Icc 0 1) →
  ((a = 0 ∧ b = 0) ∨ (a = -2 ∧ b = 1)) :=
sorry

-- Part II
theorem max_min_a_squared_plus_b_squared 
  (h1 : ∀ x : ℝ, |x| ≥ 2 → f a b x ≥ 0)
  (h2 : ∃ x ∈ Set.Ioo 2 3, ∀ y ∈ Set.Ioo 2 3, f a b x ≥ f a b y)
  (h3 : ∃ x ∈ Set.Ioo 2 3, f a b x = 1) :
  (a^2 + b^2 ≥ 32 ∧ a^2 + b^2 ≤ 74) :=
sorry

end range_when_a_b_one_values_of_a_b_for_range_zero_one_max_min_a_squared_plus_b_squared_l1617_161751


namespace prime_8p_plus_1_square_cube_l1617_161701

theorem prime_8p_plus_1_square_cube (p : ℕ) : 
  Prime p → 
  ((∃ n : ℕ, 8 * p + 1 = n^2) ↔ p = 3) ∧ 
  (¬∃ n : ℕ, 8 * p + 1 = n^3) := by
sorry

end prime_8p_plus_1_square_cube_l1617_161701


namespace roots_eighth_power_sum_l1617_161766

theorem roots_eighth_power_sum (x y : ℝ) : 
  x^2 - 2*x*Real.sqrt 2 + 1 = 0 ∧ 
  y^2 - 2*y*Real.sqrt 2 + 1 = 0 ∧ 
  x ≠ y → 
  x^8 + y^8 = 1154 := by sorry

end roots_eighth_power_sum_l1617_161766


namespace bills_divisible_by_101_l1617_161782

theorem bills_divisible_by_101 
  (a b : ℕ) 
  (h_not_cong : a % 101 ≠ b % 101) 
  (h_total : ℕ) 
  (h_total_eq : h_total = 100) :
  ∃ (subset : Finset ℕ), subset.card ≤ h_total ∧ 
    (∃ (k₁ k₂ : ℕ), k₁ + k₂ = subset.card ∧ (k₁ * a + k₂ * b) % 101 = 0) :=
sorry

end bills_divisible_by_101_l1617_161782


namespace circle_radius_is_5_l1617_161786

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (-1, 1)

-- Define the tangent line
def TangentLine (x y : ℝ) : Prop := 3 * x - 4 * y + 7 = 0

-- State the theorem
theorem circle_radius_is_5 :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    -- Circle C passes through point A
    A ∈ Circle center radius ∧
    -- Circle C is tangent to the line 3x-4y+7=0 at point B
    B ∈ Circle center radius ∧
    TangentLine B.1 B.2 ∧
    -- The radius of circle C is 5
    radius = 5 := by
  sorry

end circle_radius_is_5_l1617_161786


namespace quadratic_root_problem_l1617_161752

theorem quadratic_root_problem (a : ℝ) : 
  (3 : ℝ)^2 - (a + 2) * 3 + 2 * a = 0 → 
  ∃ x : ℝ, x^2 - (a + 2) * x + 2 * a = 0 ∧ x ≠ 3 ∧ x = 2 := by
sorry

end quadratic_root_problem_l1617_161752
