import Mathlib

namespace certain_number_divisor_l1067_106718

theorem certain_number_divisor (n : Nat) (h1 : n = 1020) : 
  ∃ x : Nat, x > 0 ∧ 
  (n - 12) % x = 0 ∧
  (n - 12) % 12 = 0 ∧ 
  (n - 12) % 24 = 0 ∧ 
  (n - 12) % 36 = 0 ∧ 
  (n - 12) % 48 = 0 ∧
  x = 7 ∧
  x ∉ Nat.divisors (Nat.lcm 12 (Nat.lcm 24 (Nat.lcm 36 48))) ∧
  ∀ y : Nat, y > x → (n - 12) % y ≠ 0 ∨ 
    y ∈ Nat.divisors (Nat.lcm 12 (Nat.lcm 24 (Nat.lcm 36 48))) :=
by sorry

end certain_number_divisor_l1067_106718


namespace value_of_q_l1067_106737

theorem value_of_q (a q : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * q) : q = 49 := by
  sorry

end value_of_q_l1067_106737


namespace ticket_sales_total_l1067_106762

/-- Calculates the total money collected from ticket sales -/
def total_money_collected (student_price general_price : ℕ) (total_tickets general_tickets : ℕ) : ℕ :=
  let student_tickets := total_tickets - general_tickets
  student_tickets * student_price + general_tickets * general_price

/-- Theorem stating that the total money collected is 2876 given the specific conditions -/
theorem ticket_sales_total :
  total_money_collected 4 6 525 388 = 2876 := by
  sorry

end ticket_sales_total_l1067_106762


namespace complex_modulus_l1067_106774

theorem complex_modulus (z : ℂ) : z = -6 + (3 - 5/3*I)*I → Complex.abs z = 5*Real.sqrt 10/3 := by
  sorry

end complex_modulus_l1067_106774


namespace sum_of_solutions_l1067_106751

theorem sum_of_solutions (x : ℝ) : 
  (Real.sqrt x + Real.sqrt (9 / x) + Real.sqrt (x + 9 / x) = 7) → 
  (∃ y : ℝ, x^2 - (49/4) * x + 9 = 0 ∧ y^2 - (49/4) * y + 9 = 0 ∧ x + y = 49/4) :=
by sorry

end sum_of_solutions_l1067_106751


namespace monotone_increasing_condition_l1067_106746

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + a| + 3

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x y, 1 < x ∧ x < y → f a x < f a y) → a ≥ -2 := by
  sorry

end monotone_increasing_condition_l1067_106746


namespace tax_difference_is_correct_l1067_106772

-- Define the item price
def item_price : ℝ := 50

-- Define the tax rates
def high_tax_rate : ℝ := 0.075
def low_tax_rate : ℝ := 0.05

-- Define the tax difference function
def tax_difference (price : ℝ) (high_rate : ℝ) (low_rate : ℝ) : ℝ :=
  price * high_rate - price * low_rate

-- Theorem statement
theorem tax_difference_is_correct : 
  tax_difference item_price high_tax_rate low_tax_rate = 1.25 := by
  sorry

end tax_difference_is_correct_l1067_106772


namespace faulty_balance_inequality_l1067_106790

/-- A faulty balance with unequal arm lengths -/
structure FaultyBalance where
  m : ℝ  -- Length of one arm
  n : ℝ  -- Length of the other arm
  h_positive_m : m > 0
  h_positive_n : n > 0
  h_unequal : m ≠ n

/-- Measurements obtained from weighing an object on a faulty balance -/
structure Measurements (fb : FaultyBalance) where
  a : ℝ  -- Measurement on one side
  b : ℝ  -- Measurement on the other side
  G : ℝ  -- True weight of the object
  h_positive_a : a > 0
  h_positive_b : b > 0
  h_positive_G : G > 0
  h_relation_a : fb.m * a = fb.n * G
  h_relation_b : fb.n * b = fb.m * G

/-- The arithmetic mean of the measurements is greater than the true weight -/
theorem faulty_balance_inequality (fb : FaultyBalance) (m : Measurements fb) :
  (m.a + m.b) / 2 > m.G := by
  sorry

end faulty_balance_inequality_l1067_106790


namespace binomial_60_2_l1067_106741

theorem binomial_60_2 : Nat.choose 60 2 = 1770 := by
  sorry

end binomial_60_2_l1067_106741


namespace tube_length_doubles_pressure_l1067_106725

/-- The length of the tube that doubles the pressure at the bottom of a water-filled barrel. -/
theorem tube_length_doubles_pressure (h₁ : ℝ) (m : ℝ) (ρ : ℝ) (g : ℝ) :
  h₁ = 1.5 →  -- height of the barrel in meters
  m = 1000 →  -- mass of water in the barrel in kg
  ρ = 1000 →  -- density of water in kg/m³
  g = 9.8 →   -- acceleration due to gravity in m/s²
  ∃ h₂ : ℝ,   -- height of water in the tube
    h₂ = 1.5 ∧ ρ * g * (h₁ + h₂) = 2 * (ρ * g * h₁) :=
by sorry

end tube_length_doubles_pressure_l1067_106725


namespace octal_subtraction_l1067_106780

/-- Represents a number in base 8 --/
def OctalNum := ℕ

/-- Converts a natural number to its octal representation --/
def toOctal (n : ℕ) : OctalNum :=
  sorry

/-- Performs subtraction in base 8 --/
def octalSub (a b : OctalNum) : OctalNum :=
  sorry

theorem octal_subtraction :
  octalSub (toOctal 126) (toOctal 57) = toOctal 47 :=
sorry

end octal_subtraction_l1067_106780


namespace parameterized_line_matches_equation_l1067_106739

/-- A line parameterized by a point and a direction vector -/
structure ParametricLine (n : Type*) [NormedAddCommGroup n] where
  point : n
  direction : n

/-- The equation of a line in slope-intercept form -/
structure SlopeInterceptLine (α : Type*) [Field α] where
  slope : α
  intercept : α

def line_equation (l : SlopeInterceptLine ℝ) (x : ℝ) : ℝ :=
  l.slope * x + l.intercept

theorem parameterized_line_matches_equation 
  (r k : ℝ) 
  (param_line : ParametricLine (Fin 2 → ℝ))
  (slope_intercept_line : SlopeInterceptLine ℝ) :
  param_line.point = ![r, 2] ∧ 
  param_line.direction = ![3, k] ∧
  slope_intercept_line.slope = 2 ∧
  slope_intercept_line.intercept = -5 →
  r = 7/2 ∧ k = 6 := by
  sorry

end parameterized_line_matches_equation_l1067_106739


namespace six_digit_divisibility_difference_l1067_106742

def six_digit_lower_bound : Nat := 100000
def six_digit_upper_bound : Nat := 999999

def count_divisible (n : Nat) : Nat :=
  (six_digit_upper_bound / n) - (six_digit_lower_bound / n)

def a : Nat := count_divisible 13 - count_divisible (13 * 17)
def b : Nat := count_divisible 17 - count_divisible (13 * 17)

theorem six_digit_divisibility_difference : a - b = 16290 := by
  sorry

end six_digit_divisibility_difference_l1067_106742


namespace geometric_sequence_sum_l1067_106756

theorem geometric_sequence_sum (n : ℕ) : 
  let a : ℚ := 1/3
  let r : ℚ := 2/3
  let sum : ℚ := a * (1 - r^n) / (1 - r)
  sum = 80/243 → n = 5 := by
sorry

end geometric_sequence_sum_l1067_106756


namespace quadratic_equation_solution_l1067_106786

theorem quadratic_equation_solution (m n : ℝ) : 
  (∀ x, x^2 + m*x - 15 = (x + 5)*(x + n)) → m = 2 ∧ n = -3 := by
  sorry

end quadratic_equation_solution_l1067_106786


namespace tank_unoccupied_volume_l1067_106773

/-- Calculates the unoccupied volume in a cube-shaped tank --/
def unoccupied_volume (tank_side : ℝ) (water_fraction : ℝ) (ice_cube_side : ℝ) (num_ice_cubes : ℕ) : ℝ :=
  let tank_volume := tank_side ^ 3
  let water_volume := water_fraction * tank_volume
  let ice_cube_volume := ice_cube_side ^ 3
  let total_ice_volume := (num_ice_cubes : ℝ) * ice_cube_volume
  let occupied_volume := water_volume + total_ice_volume
  tank_volume - occupied_volume

/-- Theorem stating the unoccupied volume in the tank --/
theorem tank_unoccupied_volume :
  unoccupied_volume 12 (1/3) 1.5 15 = 1101.375 := by
  sorry

end tank_unoccupied_volume_l1067_106773


namespace sqrt_72_plus_24sqrt6_l1067_106766

theorem sqrt_72_plus_24sqrt6 :
  ∃ (a b c : ℤ), (c > 0) ∧ 
  (∀ (n : ℕ), n > 1 → ¬(∃ (k : ℕ), c = n^2 * k)) ∧
  Real.sqrt (72 + 24 * Real.sqrt 6) = a + b * Real.sqrt c ∧
  a = 6 ∧ b = 3 ∧ c = 6 :=
by sorry

end sqrt_72_plus_24sqrt6_l1067_106766


namespace largest_increase_2011_2012_l1067_106716

/-- Represents the number of students participating in AMC 12 for each year from 2010 to 2016 --/
def amc_participants : Fin 7 → ℕ
  | 0 => 120  -- 2010
  | 1 => 130  -- 2011
  | 2 => 150  -- 2012
  | 3 => 155  -- 2013
  | 4 => 160  -- 2014
  | 5 => 140  -- 2015
  | 6 => 150  -- 2016

/-- Calculates the percentage increase between two consecutive years --/
def percentage_increase (year : Fin 6) : ℚ :=
  (amc_participants (year.succ) - amc_participants year : ℚ) / amc_participants year * 100

/-- Theorem stating that the percentage increase between 2011 and 2012 is the largest --/
theorem largest_increase_2011_2012 :
  ∀ year : Fin 6, percentage_increase 1 ≥ percentage_increase year :=
by sorry

#eval percentage_increase 1  -- Should output the largest percentage increase

end largest_increase_2011_2012_l1067_106716


namespace cannot_determine_read_sonnets_l1067_106767

/-- Represents the number of lines in a sonnet -/
def lines_per_sonnet : ℕ := 14

/-- Represents the number of unread lines -/
def unread_lines : ℕ := 70

/-- Represents the number of sonnets not read -/
def unread_sonnets : ℕ := unread_lines / lines_per_sonnet

theorem cannot_determine_read_sonnets (total_sonnets : ℕ) :
  ∀ n : ℕ, n < total_sonnets → n ≥ unread_sonnets →
  ∃ m : ℕ, m ≠ n ∧ m < total_sonnets ∧ m ≥ unread_sonnets :=
sorry

end cannot_determine_read_sonnets_l1067_106767


namespace equation_solutions_l1067_106748

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 81 = 0 ↔ x = 9/2 ∨ x = -9/2) ∧
  (∀ x l : ℝ, 64 * (x + l)^3 = 27 → x = -1/4) :=
by sorry

end equation_solutions_l1067_106748


namespace dimes_per_quarter_l1067_106750

/-- Represents the number of coins traded for a quarter -/
structure TradeRatio :=
  (dimes : ℚ)
  (nickels : ℚ)

/-- Calculates the total value of coins traded -/
def totalValue (ratio : TradeRatio) : ℚ :=
  20 * (ratio.dimes * (1/10) + ratio.nickels * (1/20))

/-- Theorem: The number of dimes traded for each quarter is 4 -/
theorem dimes_per_quarter :
  ∃ (ratio : TradeRatio),
    totalValue ratio = 10 + 3 ∧
    ratio.nickels = 5 ∧
    ratio.dimes = 4 := by
  sorry

end dimes_per_quarter_l1067_106750


namespace g_sum_lower_bound_l1067_106704

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2

noncomputable def g (x : ℝ) : ℝ := f x + 3 * x + 1

theorem g_sum_lower_bound (x₁ x₂ : ℝ) (h : x₁ + x₂ ≥ 0) :
  g x₁ + g x₂ ≥ 4 := by sorry

end g_sum_lower_bound_l1067_106704


namespace x_minus_y_equals_ten_l1067_106768

theorem x_minus_y_equals_ten (x y : ℝ) 
  (h1 : 2 = 0.10 * x) 
  (h2 : 2 = 0.20 * y) : 
  x - y = 10 := by
  sorry

end x_minus_y_equals_ten_l1067_106768


namespace smallest_number_with_conditions_l1067_106702

/-- Given a natural number, returns true if it ends with 56 -/
def ends_with_56 (n : ℕ) : Prop :=
  n % 100 = 56

/-- Given a natural number, returns the sum of its digits -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem smallest_number_with_conditions :
  ∃ (n : ℕ), 
    ends_with_56 n ∧ 
    n % 56 = 0 ∧ 
    digit_sum n = 56 ∧
    (∀ m : ℕ, m < n → ¬(ends_with_56 m ∧ m % 56 = 0 ∧ digit_sum m = 56)) ∧
    n = 29899856 :=
by sorry

end smallest_number_with_conditions_l1067_106702


namespace transformation_of_point_l1067_106778

/-- Given a point A and a transformation φ, prove that the transformed point A' has specific coordinates -/
theorem transformation_of_point (x y x' y' : ℚ) : 
  x = 1/3 ∧ y = -2 ∧ x' = 3*x ∧ 2*y' = y → x' = 1 ∧ y' = -1 := by
  sorry

end transformation_of_point_l1067_106778


namespace petr_ivanovich_insurance_contract_l1067_106711

/-- Represents an insurance tool --/
inductive InsuranceTool
| AggregateInsuranceAmount
| Deductible

/-- Represents an insurance document --/
inductive InsuranceDocument
| InsuranceRules

/-- Represents a person --/
structure Person where
  name : String

/-- Represents an insurance contract --/
structure InsuranceContract where
  owner : Person
  tools : List InsuranceTool
  appendix : InsuranceDocument

/-- Theorem stating the correct insurance tools and document for Petr Ivanovich's contract --/
theorem petr_ivanovich_insurance_contract :
  ∃ (contract : InsuranceContract),
    contract.owner = Person.mk "Petr Ivanovich" ∧
    contract.tools = [InsuranceTool.AggregateInsuranceAmount, InsuranceTool.Deductible] ∧
    contract.appendix = InsuranceDocument.InsuranceRules :=
by sorry

end petr_ivanovich_insurance_contract_l1067_106711


namespace paco_cookies_theorem_l1067_106792

/-- The number of cookies Paco ate -/
def cookies_eaten (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

theorem paco_cookies_theorem (initial : ℕ) (remaining : ℕ) 
  (h1 : initial = 28) 
  (h2 : remaining = 7) : 
  cookies_eaten initial remaining = 21 := by
  sorry

end paco_cookies_theorem_l1067_106792


namespace intersection_A_B_l1067_106788

def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 3*x < 0}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end intersection_A_B_l1067_106788


namespace exists_k_for_1001_free_ends_l1067_106765

/-- Represents the number of free ends after k iterations of drawing segments -/
def freeEnds (k : ℕ) : ℕ := 2 + 4 * k

/-- Theorem stating that there exists a positive integer k such that
    the number of free ends after k iterations is 1001 -/
theorem exists_k_for_1001_free_ends :
  ∃ k : ℕ, k > 0 ∧ freeEnds k = 1001 :=
sorry

end exists_k_for_1001_free_ends_l1067_106765


namespace closest_fraction_to_37_57_l1067_106700

theorem closest_fraction_to_37_57 :
  ∀ n : ℤ, n ≠ 15 → |37/57 - 15/23| < |37/57 - n/23| := by
sorry

end closest_fraction_to_37_57_l1067_106700


namespace algebraic_expression_value_l1067_106736

theorem algebraic_expression_value (m n : ℝ) (h : 2*m - 3*n = -2) :
  4*m - 6*n + 1 = -3 := by
  sorry

end algebraic_expression_value_l1067_106736


namespace ellipse_k_range_l1067_106775

/-- The equation of an ellipse with parameter k -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (3 + k) + y^2 / (2 - k) = 1 ∧
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

/-- The range of k for which the equation represents an ellipse -/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ k ∈ Set.Ioo (-3 : ℝ) (-1/2) ∪ Set.Ioo (-1/2 : ℝ) 2 :=
sorry

end ellipse_k_range_l1067_106775


namespace average_after_exclusion_l1067_106769

theorem average_after_exclusion (numbers : Finset ℕ) (sum : ℕ) (excluded : ℕ) :
  numbers.card = 5 →
  sum / numbers.card = 27 →
  excluded ∈ numbers →
  excluded = 35 →
  (sum - excluded) / (numbers.card - 1) = 25 := by
  sorry

end average_after_exclusion_l1067_106769


namespace triangle_area_theorem_l1067_106771

/-- The area of a triangle given its three altitudes --/
def triangle_area_from_altitudes (h₁ h₂ h₃ : ℝ) : ℝ := sorry

/-- A triangle with altitudes 36.4, 39, and 42 has an area of 3549/4 --/
theorem triangle_area_theorem :
  triangle_area_from_altitudes 36.4 39 42 = 3549 / 4 := by sorry

end triangle_area_theorem_l1067_106771


namespace total_kids_l1067_106728

theorem total_kids (girls : ℕ) (boys : ℕ) (h1 : girls = 3) (h2 : boys = 6) :
  girls + boys = 9 := by
  sorry

end total_kids_l1067_106728


namespace three_digit_reverse_double_l1067_106722

theorem three_digit_reverse_double (g : ℕ) (a b c : ℕ) : 
  (0 < g) → 
  (a < g) → (b < g) → (c < g) →
  (a * g^2 + b * g + c = 2 * (c * g^2 + b * g + a)) →
  ∃ k : ℕ, (k > 0) ∧ (g = 3 * k + 2) := by
sorry


end three_digit_reverse_double_l1067_106722


namespace squirrels_and_nuts_l1067_106701

theorem squirrels_and_nuts (squirrels : ℕ) (nuts : ℕ) : 
  squirrels = 4 → squirrels = nuts + 2 → nuts = 2 := by
  sorry

end squirrels_and_nuts_l1067_106701


namespace triangle_inequality_l1067_106710

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > a^3 + b^3 + c^3 := by
sorry

end triangle_inequality_l1067_106710


namespace shaded_area_sum_l1067_106732

/-- Represents the shaded area between a circle and an inscribed equilateral triangle --/
structure ShadedArea where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the shaded area for a given circle and inscribed equilateral triangle --/
def calculateShadedArea (sideLength : ℝ) : ShadedArea :=
  { a := 18.75,
    b := 21,
    c := 3 }

/-- Theorem stating the sum of a, b, and c for the given problem --/
theorem shaded_area_sum (sideLength : ℝ) :
  sideLength = 15 →
  let area := calculateShadedArea sideLength
  area.a + area.b + area.c = 42.75 := by
  sorry

#check shaded_area_sum

end shaded_area_sum_l1067_106732


namespace distinct_roots_condition_l1067_106787

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := k * x^2 - 2 * x - 1

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := 4 + 4 * k

-- Theorem statement
theorem distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x = 0 ∧ quadratic_equation k y = 0) ↔
  (k > -1 ∧ k ≠ 0) :=
sorry

end distinct_roots_condition_l1067_106787


namespace no_common_values_under_180_l1067_106759

theorem no_common_values_under_180 : 
  ¬ ∃ x : ℕ, x < 180 ∧ x % 13 = 2 ∧ x % 8 = 5 := by
sorry

end no_common_values_under_180_l1067_106759


namespace number_2009_in_group_31_l1067_106719

/-- The sum of squares of the first n odd numbers -/
def O (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 3

/-- The number we're looking for -/
def target : ℕ := 2009

/-- The group number we're proving -/
def group_number : ℕ := 31

theorem number_2009_in_group_31 :
  O (group_number - 1) < target ∧ target ≤ O group_number :=
sorry

end number_2009_in_group_31_l1067_106719


namespace angle_equality_l1067_106764

-- Define the types for points and angles
variable (Point Angle : Type)

-- Define the triangle ABC
variable (A B C : Point)

-- Define the points on the sides of the triangle
variable (P₁ P₂ Q₁ Q₂ R S M : Point)

-- Define the necessary geometric predicates
variable (lies_on : Point → Point → Point → Prop)
variable (is_midpoint : Point → Point → Point → Prop)
variable (angle : Point → Point → Point → Angle)
variable (length_eq : Point → Point → Point → Point → Prop)
variable (intersects : Point → Point → Point → Point → Point → Prop)
variable (on_circumcircle : Point → Point → Point → Point → Prop)
variable (inside_triangle : Point → Point → Point → Point → Prop)

-- State the theorem
theorem angle_equality 
  (h1 : lies_on P₁ A B) (h2 : lies_on P₂ A B) (h3 : lies_on P₂ B P₁)
  (h4 : length_eq A P₁ B P₂)
  (h5 : lies_on Q₁ B C) (h6 : lies_on Q₂ B C) (h7 : lies_on Q₂ B Q₁)
  (h8 : length_eq B Q₁ C Q₂)
  (h9 : intersects P₁ Q₂ P₂ Q₁ R)
  (h10 : on_circumcircle S P₁ P₂ R) (h11 : on_circumcircle S Q₁ Q₂ R)
  (h12 : inside_triangle S P₁ Q₁ R)
  (h13 : is_midpoint M A C) :
  angle P₁ R S = angle Q₁ R M :=
by sorry

end angle_equality_l1067_106764


namespace books_in_series_l1067_106789

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 59

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 61

/-- There are 2 more movies than books in the series -/
axiom movie_book_difference : num_movies = num_books + 2

theorem books_in_series : num_books = 59 := by sorry

end books_in_series_l1067_106789


namespace mountain_bike_price_l1067_106723

theorem mountain_bike_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 → 
  upfront_percentage = 20 → 
  upfront_payment = (upfront_percentage / 100) * total_price → 
  total_price = 1200 := by
sorry

end mountain_bike_price_l1067_106723


namespace stating_adjacent_probability_in_grid_l1067_106720

/-- The number of students -/
def num_students : ℕ := 8

/-- The number of rows in the seating arrangement -/
def num_rows : ℕ := 2

/-- The number of columns in the seating arrangement -/
def num_columns : ℕ := 4

/-- The probability of two specific students being adjacent -/
def adjacent_probability : ℚ := 5/14

/-- 
Theorem stating that the probability of two specific students 
being adjacent in a random seating arrangement is 5/14
-/
theorem adjacent_probability_in_grid : 
  let total_arrangements := Nat.factorial num_students
  let row_adjacent_pairs := num_rows * (num_columns - 1)
  let column_adjacent_pairs := num_columns
  let ways_to_arrange_pair := 2
  let remaining_arrangements := Nat.factorial (num_students - 2)
  let favorable_outcomes := (row_adjacent_pairs + column_adjacent_pairs) * 
                            ways_to_arrange_pair * 
                            remaining_arrangements
  (favorable_outcomes : ℚ) / total_arrangements = adjacent_probability := by
  sorry

end stating_adjacent_probability_in_grid_l1067_106720


namespace total_steps_in_week_l1067_106745

/-- Represents the number of steps taken to school and back for each day -/
structure DailySteps where
  toSchool : ℕ
  fromSchool : ℕ

/-- Calculates the total steps for a given day -/
def totalSteps (day : DailySteps) : ℕ := day.toSchool + day.fromSchool

/-- Represents Raine's walking data for the week -/
structure WeeklyWalk where
  monday : DailySteps
  tuesday : DailySteps
  wednesday : DailySteps
  thursday : DailySteps
  friday : DailySteps

/-- The actual walking data for Raine's week -/
def rainesWeek : WeeklyWalk := {
  monday := { toSchool := 150, fromSchool := 170 }
  tuesday := { toSchool := 140, fromSchool := 170 }  -- 140 + 30 rest stop
  wednesday := { toSchool := 160, fromSchool := 210 }
  thursday := { toSchool := 150, fromSchool := 170 }  -- 140 + 30 rest stop
  friday := { toSchool := 180, fromSchool := 200 }
}

/-- Theorem: The total number of steps Raine takes in five days is 1700 -/
theorem total_steps_in_week (w : WeeklyWalk := rainesWeek) :
  totalSteps w.monday + totalSteps w.tuesday + totalSteps w.wednesday +
  totalSteps w.thursday + totalSteps w.friday = 1700 := by
  sorry

end total_steps_in_week_l1067_106745


namespace range_of_a_l1067_106758

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → 
  -1 ≤ a ∧ a ≤ 3 := by
sorry

end range_of_a_l1067_106758


namespace henry_returned_half_l1067_106749

/-- The portion of catch Henry returned -/
def henryReturnedPortion (willCatfish : ℕ) (willEels : ℕ) (henryTroutPerCatfish : ℕ) (totalFishAfterReturn : ℕ) : ℚ :=
  let willTotal := willCatfish + willEels
  let henryTotal := willCatfish * henryTroutPerCatfish
  let totalBeforeReturn := willTotal + henryTotal
  let returnedFish := totalBeforeReturn - totalFishAfterReturn
  returnedFish / henryTotal

/-- Theorem stating that Henry returned half of his catch -/
theorem henry_returned_half :
  henryReturnedPortion 16 10 3 50 = 1/2 := by
  sorry

end henry_returned_half_l1067_106749


namespace histogram_frequency_l1067_106729

theorem histogram_frequency (sample_size : ℕ) (num_groups : ℕ) (class_interval : ℕ) (rectangle_height : ℝ) : 
  sample_size = 100 →
  num_groups = 10 →
  class_interval = 10 →
  rectangle_height = 0.03 →
  (rectangle_height * class_interval * sample_size : ℝ) = 30 :=
by
  sorry

end histogram_frequency_l1067_106729


namespace point_move_result_l1067_106776

def point_move (initial_position : ℤ) (move_distance : ℤ) : Set ℤ :=
  {initial_position - move_distance, initial_position + move_distance}

theorem point_move_result :
  point_move (-5) 3 = {-8, -2} := by sorry

end point_move_result_l1067_106776


namespace inequality_system_solution_set_l1067_106721

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by
sorry

end inequality_system_solution_set_l1067_106721


namespace valid_word_count_mod_2000_l1067_106735

/-- Represents a letter in Zuminglish --/
inductive ZuminglishLetter
| M
| O
| P

/-- Represents whether a letter is a vowel or consonant --/
def isVowel : ZuminglishLetter → Bool
| ZuminglishLetter.O => true
| _ => false

/-- A Zuminglish word is a list of Zuminglish letters --/
def ZuminglishWord := List ZuminglishLetter

/-- Checks if a Zuminglish word is valid (no two O's are adjacent without at least two consonants in between) --/
def isValidWord : ZuminglishWord → Bool := sorry

/-- Counts the number of valid 12-letter Zuminglish words --/
def countValidWords : Nat := sorry

/-- The main theorem: The number of valid 12-letter Zuminglish words is congruent to 192 modulo 2000 --/
theorem valid_word_count_mod_2000 : countValidWords % 2000 = 192 := by sorry

end valid_word_count_mod_2000_l1067_106735


namespace career_preference_theorem_l1067_106783

/-- Represents the degrees in a circle graph for a career preference -/
def career_preference_degrees (male_ratio female_ratio : ℚ) 
  (male_preference female_preference : ℚ) : ℚ :=
  ((male_ratio * male_preference + female_ratio * female_preference) / 
   (male_ratio + female_ratio)) * 360

/-- Theorem: The degrees for the given career preference -/
theorem career_preference_theorem : 
  career_preference_degrees 2 3 (1/4) (3/4) = 198 := by
  sorry

end career_preference_theorem_l1067_106783


namespace daves_weight_l1067_106743

/-- Proves Dave's weight given the conditions from the problem -/
theorem daves_weight (dave_weight : ℝ) (dave_bench : ℝ) (craig_bench : ℝ) (mark_bench : ℝ) :
  dave_bench = 3 * dave_weight →
  craig_bench = 0.2 * dave_bench →
  mark_bench = craig_bench - 50 →
  mark_bench = 55 →
  dave_weight = 175 := by
sorry

end daves_weight_l1067_106743


namespace lg_sum_equals_zero_l1067_106734

-- Define lg as the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_zero : lg 5 + lg 0.2 = 0 := by sorry

end lg_sum_equals_zero_l1067_106734


namespace sum_f_negative_l1067_106730

def f (x : ℝ) := -x - x^3

theorem sum_f_negative (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end sum_f_negative_l1067_106730


namespace water_volume_ratio_l1067_106763

/-- Represents a location with rainfall and area data -/
structure Location where
  rainfall : ℝ  -- rainfall in cm
  area : ℝ      -- area in hectares

/-- Calculates the volume of water collected at a location -/
def waterVolume (loc : Location) : ℝ :=
  loc.rainfall * loc.area * 10

/-- Theorem stating the ratio of water volumes collected at locations A, B, and C -/
theorem water_volume_ratio 
  (locA locB locC : Location)
  (hA : locA = { rainfall := 7, area := 2 })
  (hB : locB = { rainfall := 11, area := 3.5 })
  (hC : locC = { rainfall := 15, area := 5 }) :
  ∃ (k : ℝ), k > 0 ∧ 
    waterVolume locA = 140 * k ∧ 
    waterVolume locB = 385 * k ∧ 
    waterVolume locC = 750 * k :=
sorry

end water_volume_ratio_l1067_106763


namespace maria_reading_capacity_l1067_106795

/-- The number of books Maria can read given her reading speed, book length, and available time -/
def books_read (reading_speed : ℕ) (pages_per_book : ℕ) (available_hours : ℕ) : ℕ :=
  (reading_speed * available_hours) / pages_per_book

/-- Theorem: Maria can read 3 books of 360 pages each in 9 hours at a speed of 120 pages per hour -/
theorem maria_reading_capacity :
  books_read 120 360 9 = 3 := by
  sorry

end maria_reading_capacity_l1067_106795


namespace decreasing_function_condition_l1067_106724

-- Define the function f(x)
def f (k x : ℝ) : ℝ := k * x^3 + 3 * (k - 1) * x^2 - k^2 + 1

-- Define the derivative of f(x)
def f_derivative (k x : ℝ) : ℝ := 3 * k * x^2 + 6 * (k - 1) * x

-- Theorem statement
theorem decreasing_function_condition (k : ℝ) :
  (∀ x ∈ Set.Ioo 0 4, f_derivative k x ≤ 0) ↔ k ≤ 1/3 :=
by sorry

end decreasing_function_condition_l1067_106724


namespace fraction_of_special_number_in_list_l1067_106791

theorem fraction_of_special_number_in_list (l : List ℝ) (n : ℝ) :
  l.length = 21 →
  l.Nodup →
  n ∈ l →
  n = 4 * ((l.sum - n) / 20) →
  n = (1 / 6) * l.sum := by
sorry

end fraction_of_special_number_in_list_l1067_106791


namespace consecutive_integers_average_l1067_106744

theorem consecutive_integers_average (c d : ℤ) : 
  (c > 0) →
  (d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7) →
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = c + 6) :=
by sorry

end consecutive_integers_average_l1067_106744


namespace population_ratio_x_to_z_l1067_106713

/-- Represents the population of a city. -/
structure CityPopulation where
  value : ℕ

/-- Represents the ratio between two city populations. -/
structure PopulationRatio where
  numerator : ℕ
  denominator : ℕ

/-- Given three cities X, Y, and Z, where X's population is 8 times Y's,
    and Y's population is twice Z's, prove that the ratio of X's population
    to Z's population is 16:1. -/
theorem population_ratio_x_to_z
  (pop_x pop_y pop_z : CityPopulation)
  (h1 : pop_x.value = 8 * pop_y.value)
  (h2 : pop_y.value = 2 * pop_z.value) :
  PopulationRatio.mk 16 1 = PopulationRatio.mk (pop_x.value / pop_z.value) 1 := by
  sorry

end population_ratio_x_to_z_l1067_106713


namespace scissors_in_drawer_l1067_106781

theorem scissors_in_drawer (initial_scissors : ℕ) : initial_scissors = 54 →
  initial_scissors + 22 = 76 := by
  sorry

end scissors_in_drawer_l1067_106781


namespace sequence_formula_correct_l1067_106709

/-- The general term formula for the sequence -1/2, 1/4, -1/8, 1/16, ... -/
def sequence_formula (n : ℕ) : ℚ := (-1)^(n+1) / (2^n)

/-- The nth term of the sequence -1/2, 1/4, -1/8, 1/16, ... -/
def sequence_term (n : ℕ) : ℚ := 
  if n % 2 = 1 
  then -1 / (2^n) 
  else 1 / (2^n)

theorem sequence_formula_correct : 
  ∀ n : ℕ, n > 0 → sequence_formula n = sequence_term n :=
sorry

end sequence_formula_correct_l1067_106709


namespace rectangle_y_value_l1067_106798

/-- Given a rectangle with vertices at (-2, y), (6, y), (-2, 2), and (6, 2),
    if the area is 80 square units, then y = 12 -/
theorem rectangle_y_value (y : ℝ) : 
  let vertices : List (ℝ × ℝ) := [(-2, y), (6, y), (-2, 2), (6, 2)]
  let width : ℝ := 6 - (-2)
  let height : ℝ := y - 2
  let area : ℝ := width * height
  (∀ v ∈ vertices, v.1 = -2 ∨ v.1 = 6) ∧
  (∀ v ∈ vertices, v.2 = y ∨ v.2 = 2) ∧
  area = 80 →
  y = 12 := by
sorry

end rectangle_y_value_l1067_106798


namespace bing_duan_duan_properties_l1067_106753

/-- Represents the production and sales of "Bing Duan Duan" mascots --/
structure BingDuanDuan where
  feb_production : ℕ
  apr_production : ℕ
  daily_sales : ℕ
  profit_per_item : ℕ
  sales_increase : ℕ
  max_price_reduction : ℕ
  target_daily_profit : ℕ

/-- Calculates the monthly growth rate given February and April production --/
def monthly_growth_rate (b : BingDuanDuan) : ℚ :=
  ((b.apr_production : ℚ) / b.feb_production) ^ (1/2) - 1

/-- Calculates the optimal price reduction --/
def optimal_price_reduction (b : BingDuanDuan) : ℕ :=
  sorry -- The actual calculation would go here

/-- Theorem stating the properties of BingDuanDuan production and sales --/
theorem bing_duan_duan_properties (b : BingDuanDuan) 
  (h1 : b.feb_production = 500)
  (h2 : b.apr_production = 720)
  (h3 : b.daily_sales = 20)
  (h4 : b.profit_per_item = 40)
  (h5 : b.sales_increase = 5)
  (h6 : b.max_price_reduction = 10)
  (h7 : b.target_daily_profit = 1440) :
  monthly_growth_rate b = 1/5 ∧ 
  optimal_price_reduction b = 4 ∧ 
  optimal_price_reduction b ≤ b.max_price_reduction :=
by sorry


end bing_duan_duan_properties_l1067_106753


namespace water_bottle_pricing_l1067_106794

theorem water_bottle_pricing (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive (price can't be negative or zero)
  (h2 : x > 10) -- Ensure x-10 is positive (price of type B can't be negative)
  : 700 / x = 500 / (x - 10) := by
  sorry

end water_bottle_pricing_l1067_106794


namespace equation_system_solutions_l1067_106712

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(1, 5, 2, 3), (1, 5, 3, 2), (5, 1, 2, 3), (5, 1, 3, 2),
   (2, 3, 1, 5), (2, 3, 5, 1), (3, 2, 1, 5), (3, 2, 5, 1),
   (2, 2, 2, 2)}

theorem equation_system_solutions :
  ∀ x y z t : ℕ,
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 →
    x + y = z * t ∧ z + t = x * y ↔ (x, y, z, t) ∈ solution_set :=
sorry

end equation_system_solutions_l1067_106712


namespace pet_store_cages_l1067_106706

def total_cages (num_snakes num_parrots num_rabbits : ℕ)
                (snakes_per_cage parrots_per_cage rabbits_per_cage : ℕ) : ℕ :=
  (num_snakes / snakes_per_cage) + (num_parrots / parrots_per_cage) + (num_rabbits / rabbits_per_cage)

theorem pet_store_cages :
  total_cages 4 6 8 2 3 4 = 6 :=
by sorry

end pet_store_cages_l1067_106706


namespace shortest_distance_to_circle_l1067_106703

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 6*y + 9 = 0

/-- The shortest distance from the origin to the circle -/
def shortest_distance : ℝ := 1

/-- Theorem: The shortest distance from the origin to the circle defined by
    x^2 - 8x + y^2 + 6y + 9 = 0 is equal to 1 -/
theorem shortest_distance_to_circle :
  ∀ (x y : ℝ), circle_equation x y →
  ∃ (p : ℝ × ℝ), p ∈ {(x, y) | circle_equation x y} ∧
  ∀ (q : ℝ × ℝ), q ∈ {(x, y) | circle_equation x y} →
  Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≤ Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) ∧
  Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) = shortest_distance :=
by sorry

end shortest_distance_to_circle_l1067_106703


namespace min_distance_b_to_c_l1067_106726

/-- Calculates the minimum distance between points B and C given boat and river conditions -/
theorem min_distance_b_to_c 
  (boat_speed : ℝ) 
  (downstream_current : ℝ) 
  (upstream_current : ℝ) 
  (time_a_to_b : ℝ) 
  (max_time_b_to_c : ℝ) 
  (h1 : boat_speed = 42) 
  (h2 : downstream_current = 5) 
  (h3 : upstream_current = 7) 
  (h4 : time_a_to_b = 1 + 10/60) 
  (h5 : max_time_b_to_c = 2.5) : 
  ∃ (min_distance : ℝ), min_distance = 87.5 := by
  sorry

#check min_distance_b_to_c

end min_distance_b_to_c_l1067_106726


namespace intersection_distance_approx_l1067_106705

-- Define the centers of the circles
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (4, 0)
def D : ℝ × ℝ := (5, 0)

-- Define the radii of the circles
def radius_A : ℝ := 2
def radius_B : ℝ := 2
def radius_C : ℝ := 3
def radius_D : ℝ := 3

-- Define the equations of the circles
def circle_A (x y : ℝ) : Prop := x^2 + y^2 = radius_A^2
def circle_C (x y : ℝ) : Prop := (x - C.1)^2 + y^2 = radius_C^2
def circle_D (x y : ℝ) : Prop := (x - D.1)^2 + y^2 = radius_D^2

-- Define the intersection points
def B' : ℝ × ℝ := sorry
def D' : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_distance_approx :
  ∃ ε > 0, abs (Real.sqrt ((B'.1 - D'.1)^2 + (B'.2 - D'.2)^2) - 0.8) < ε :=
sorry

end intersection_distance_approx_l1067_106705


namespace logans_father_cartons_l1067_106757

/-- The number of cartons Logan's father usually receives -/
def usual_cartons : ℕ := 50

/-- The number of jars in each carton -/
def jars_per_carton : ℕ := 20

/-- The number of cartons received in the particular week -/
def received_cartons : ℕ := usual_cartons - 20

/-- The number of damaged jars from partially damaged cartons -/
def partially_damaged_jars : ℕ := 5 * 3

/-- The number of damaged jars from the totally damaged carton -/
def totally_damaged_jars : ℕ := jars_per_carton

/-- The total number of damaged jars -/
def total_damaged_jars : ℕ := partially_damaged_jars + totally_damaged_jars

/-- The number of jars good for sale in the particular week -/
def good_jars : ℕ := 565

theorem logans_father_cartons :
  jars_per_carton * received_cartons - total_damaged_jars = good_jars :=
by sorry

end logans_father_cartons_l1067_106757


namespace line_properties_l1067_106754

/-- A line passing through a point with given conditions -/
structure Line where
  P : ℝ × ℝ
  α : ℝ
  intersects_positive_axes : Bool
  PA_PB_product : ℝ

/-- The main theorem stating the properties of the line -/
theorem line_properties (l : Line) 
  (h1 : l.P = (2, 1))
  (h2 : l.intersects_positive_axes = true)
  (h3 : l.PA_PB_product = 4) :
  (l.α = 3 * Real.pi / 4) ∧ 
  (∃ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 3) := by
  sorry

#check line_properties

end line_properties_l1067_106754


namespace orange_balls_count_l1067_106779

theorem orange_balls_count (total green red blue yellow pink orange purple : ℕ) :
  total = 120 ∧
  green = 5 ∧
  red = 30 ∧
  blue = 20 ∧
  yellow = 10 ∧
  pink = 2 * green ∧
  orange = 3 * pink ∧
  purple = orange - pink ∧
  total = red + blue + yellow + green + pink + orange + purple →
  orange = 30 := by
sorry

end orange_balls_count_l1067_106779


namespace morley_theorem_l1067_106770

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Represents a ray (half-line) -/
structure Ray :=
  (origin : Point) (direction : Point)

/-- Defines a trisector of an angle -/
def is_trisector (r : Ray) (A B C : Point) : Prop := sorry

/-- Defines the intersection point of two rays -/
def intersection (r1 r2 : Ray) : Point := sorry

/-- Morley's theorem -/
theorem morley_theorem (T : Triangle) :
  let A := T.A
  let B := T.B
  let C := T.C
  let trisector_B1 := Ray.mk B (sorry : Point)
  let trisector_B2 := Ray.mk B (sorry : Point)
  let trisector_C1 := Ray.mk C (sorry : Point)
  let trisector_C2 := Ray.mk C (sorry : Point)
  let trisector_A1 := Ray.mk A (sorry : Point)
  let trisector_A2 := Ray.mk A (sorry : Point)
  let A1 := intersection trisector_B1 trisector_C1
  let B1 := intersection trisector_C2 trisector_A1
  let C1 := intersection trisector_A2 trisector_B2
  is_trisector trisector_B1 B A C ∧
  is_trisector trisector_B2 B A C ∧
  is_trisector trisector_C1 C B A ∧
  is_trisector trisector_C2 C B A ∧
  is_trisector trisector_A1 A C B ∧
  is_trisector trisector_A2 A C B →
  -- A1B1 = B1C1 = C1A1
  (A1.x - B1.x)^2 + (A1.y - B1.y)^2 =
  (B1.x - C1.x)^2 + (B1.y - C1.y)^2 ∧
  (B1.x - C1.x)^2 + (B1.y - C1.y)^2 =
  (C1.x - A1.x)^2 + (C1.y - A1.y)^2 :=
sorry

end morley_theorem_l1067_106770


namespace tan_seven_pi_fourths_l1067_106731

theorem tan_seven_pi_fourths : Real.tan (7 * π / 4) = -1 := by
  sorry

end tan_seven_pi_fourths_l1067_106731


namespace average_marks_combined_classes_l1067_106761

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 12) (h2 : n2 = 28) (h3 : avg1 = 40) (h4 : avg2 = 60) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 54 := by
  sorry

end average_marks_combined_classes_l1067_106761


namespace toy_production_on_time_l1067_106796

/-- Proves that the toy production can be completed on time --/
theorem toy_production_on_time (total_toys : ℕ) (first_three_days_avg : ℕ) (remaining_days_avg : ℕ) 
  (available_days : ℕ) (h1 : total_toys = 3000) (h2 : first_three_days_avg = 250) 
  (h3 : remaining_days_avg = 375) (h4 : available_days = 11) : 
  (3 + ((total_toys - 3 * first_three_days_avg) / remaining_days_avg : ℕ)) ≤ available_days := by
  sorry

#check toy_production_on_time

end toy_production_on_time_l1067_106796


namespace point_coordinates_l1067_106752

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (p : Point) : ℝ := |p.y|

/-- The distance of a point from the y-axis -/
def distanceFromYAxis (p : Point) : ℝ := |p.x|

/-- Determines if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop := p.x < 0 ∧ p.y > 0

/-- Theorem: Given the conditions, the point M has coordinates (-2, 3) -/
theorem point_coordinates (M : Point)
    (h1 : distanceFromXAxis M = 3)
    (h2 : distanceFromYAxis M = 2)
    (h3 : isInSecondQuadrant M) :
    M = Point.mk (-2) 3 := by
  sorry

end point_coordinates_l1067_106752


namespace glass_bowl_problem_l1067_106782

/-- The initial rate per bowl given the conditions of the glass bowl sales problem -/
def initial_rate_per_bowl (total_bowls : ℕ) (sold_bowls : ℕ) (selling_price : ℚ) (percentage_gain : ℚ) : ℚ :=
  (sold_bowls * selling_price) / (total_bowls * (1 + percentage_gain / 100))

theorem glass_bowl_problem :
  let total_bowls : ℕ := 114
  let sold_bowls : ℕ := 108
  let selling_price : ℚ := 17
  let percentage_gain : ℚ := 23.88663967611336
  abs (initial_rate_per_bowl total_bowls sold_bowls selling_price percentage_gain - 13) < 0.01 := by
  sorry

#eval initial_rate_per_bowl 114 108 17 23.88663967611336

end glass_bowl_problem_l1067_106782


namespace power_of_two_divisibility_l1067_106717

theorem power_of_two_divisibility (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ r : ℕ, n.val = 2^r :=
sorry

end power_of_two_divisibility_l1067_106717


namespace specific_l_shape_area_l1067_106799

/-- The area of an "L" shape formed by removing a smaller rectangle from a larger rectangle --/
def l_shape_area (length width subtract_length subtract_width : ℕ) : ℕ :=
  length * width - (length - subtract_length) * (width - subtract_width)

/-- Theorem: The area of the specific "L" shape is 42 square units --/
theorem specific_l_shape_area : l_shape_area 10 7 3 3 = 42 := by
  sorry

end specific_l_shape_area_l1067_106799


namespace phase_shift_cosine_l1067_106747

/-- The phase shift of y = 3 cos(4x - π/4) is π/16 -/
theorem phase_shift_cosine (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 3 * Real.cos (4 * x - π / 4)
  ∃ (shift : ℝ), shift = π / 16 ∧ 
    ∀ (t : ℝ), f (t + shift) = 3 * Real.cos (4 * t) := by
  sorry

end phase_shift_cosine_l1067_106747


namespace det_special_matrix_l1067_106738

theorem det_special_matrix (x : ℝ) : 
  Matrix.det (![![x + 3, x, x], ![x, x + 3, x], ![x, x, x + 3]]) = 27 * x + 27 := by
  sorry

end det_special_matrix_l1067_106738


namespace invest_in_good_B_l1067_106793

def expected_profit (p1 p2 p3 : ℝ) (v1 v2 v3 : ℝ) : ℝ :=
  p1 * v1 + p2 * v2 + p3 * v3

theorem invest_in_good_B (capital : ℝ) 
  (a_p1 a_p2 a_p3 : ℝ) (a_v1 a_v2 a_v3 : ℝ)
  (b_p1 b_p2 b_p3 : ℝ) (b_v1 b_v2 b_v3 : ℝ)
  (ha1 : a_p1 = 0.4) (ha2 : a_p2 = 0.3) (ha3 : a_p3 = 0.3)
  (ha4 : a_v1 = 20000) (ha5 : a_v2 = 30000) (ha6 : a_v3 = -10000)
  (hb1 : b_p1 = 0.6) (hb2 : b_p2 = 0.2) (hb3 : b_p3 = 0.2)
  (hb4 : b_v1 = 20000) (hb5 : b_v2 = 40000) (hb6 : b_v3 = -20000)
  (hcap : capital = 100000) :
  expected_profit b_p1 b_p2 b_p3 b_v1 b_v2 b_v3 > 
  expected_profit a_p1 a_p2 a_p3 a_v1 a_v2 a_v3 := by
  sorry

#check invest_in_good_B

end invest_in_good_B_l1067_106793


namespace propositions_are_false_l1067_106733

-- Define a type for planes
def Plane : Type := Unit

-- Define a relation for "is in"
def is_in (α β : Plane) : Prop := sorry

-- Define a relation for "is parallel to"
def is_parallel (α β : Plane) : Prop := sorry

-- Define a type for points
def Point : Type := Unit

-- Define a property for three points being non-collinear
def non_collinear (p q r : Point) : Prop := sorry

-- Define a property for a point being on a plane
def on_plane (p : Point) (α : Plane) : Prop := sorry

-- Define a property for a point being equidistant from a plane
def equidistant_from_plane (p : Point) (β : Plane) : Prop := sorry

theorem propositions_are_false :
  (∃ α β γ : Plane, is_in α β ∧ is_in β γ ∧ ¬is_parallel α γ) ∧
  (∃ α β : Plane, ∃ p q r : Point,
    non_collinear p q r ∧
    on_plane p α ∧ on_plane q α ∧ on_plane r α ∧
    equidistant_from_plane p β ∧ equidistant_from_plane q β ∧ equidistant_from_plane r β ∧
    ¬is_parallel α β) :=
by sorry

end propositions_are_false_l1067_106733


namespace circumcircle_diameter_l1067_106784

/-- Given a triangle ABC with side length a = 2 and angle A = 60°,
    prove that the diameter of its circumcircle is 4√3/3 -/
theorem circumcircle_diameter (a : ℝ) (A : ℝ) (h1 : a = 2) (h2 : A = π/3) :
  (2 * a) / Real.sin A = 4 * Real.sqrt 3 / 3 := by
  sorry

end circumcircle_diameter_l1067_106784


namespace tic_tac_toe_rounds_l1067_106755

/-- Given that William won 10 rounds of tic-tac-toe and 5 more rounds than Harry,
    prove that the total number of rounds played is 15. -/
theorem tic_tac_toe_rounds (william_rounds harry_rounds total_rounds : ℕ) 
  (h1 : william_rounds = 10)
  (h2 : william_rounds = harry_rounds + 5) : 
  total_rounds = 15 := by
  sorry

end tic_tac_toe_rounds_l1067_106755


namespace min_value_of_f_l1067_106727

def f (x : ℝ) : ℝ := |2*x + 1| + |x - 1|

theorem min_value_of_f :
  ∃ (min_val : ℝ), min_val = 3/2 ∧ ∀ (x : ℝ), f x ≥ min_val :=
sorry

end min_value_of_f_l1067_106727


namespace train_passing_jogger_l1067_106760

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger (v_jogger v_train : ℝ) (train_length initial_distance : ℝ) :
  v_jogger = 10 * 1000 / 3600 →
  v_train = 46 * 1000 / 3600 →
  train_length = 120 →
  initial_distance = 340 →
  (initial_distance + train_length) / (v_train - v_jogger) = 46 := by
  sorry

#check train_passing_jogger

end train_passing_jogger_l1067_106760


namespace dot_product_sum_l1067_106777

theorem dot_product_sum (a b : ℝ × ℝ × ℝ) (h1 : a = (0, 2, 0)) (h2 : b = (1, 0, -1)) :
  (a.1 + b.1, a.2.1 + b.2.1, a.2.2 + b.2.2) • b = 2 := by
  sorry

end dot_product_sum_l1067_106777


namespace colored_integers_theorem_l1067_106740

def ColoredInteger := ℤ → Bool

theorem colored_integers_theorem (color : ColoredInteger) 
  (h1 : color 1 = true)
  (h2 : ∀ a b : ℤ, color a = true → color b = true → color (a + b) ≠ color (a - b)) :
  color 2011 = true := by sorry

end colored_integers_theorem_l1067_106740


namespace trapezium_area_l1067_106708

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 15) (hh : h = 14) :
  (a + b) * h / 2 = 245 :=
by sorry

end trapezium_area_l1067_106708


namespace sphere_cylinder_volume_difference_l1067_106785

/-- The volume of the space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 7) (h_cylinder : r_cylinder = 4) :
  let h_cylinder := 2 * Real.sqrt 33
  let v_sphere := (4 / 3) * π * r_sphere ^ 3
  let v_cylinder := π * r_cylinder ^ 2 * h_cylinder
  v_sphere - v_cylinder = (1372 / 3 - 32 * Real.sqrt 33) * π :=
by sorry

end sphere_cylinder_volume_difference_l1067_106785


namespace problem_solution_l1067_106707

theorem problem_solution (a b c : ℕ) 
  (ha : a > 0 ∧ a < 10) 
  (hb : b > 0 ∧ b < 10) 
  (hc : c > 0 ∧ c < 10) 
  (h_prob : (1/a + 1/b + 1/c) - (1/a * 1/b + 1/a * 1/c + 1/b * 1/c) + (1/a * 1/b * 1/c) = 7/15) : 
  (1 - 1/a) * (1 - 1/b) * (1 - 1/c) = 8/15 := by
sorry

end problem_solution_l1067_106707


namespace equal_area_rectangle_width_l1067_106714

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangle_width (r1 r2 : Rectangle) 
  (h1 : r1.length = 12)
  (h2 : r1.width = 10)
  (h3 : r2.length = 24)
  (h4 : area r1 = area r2) :
  r2.width = 5 := by
  sorry

end equal_area_rectangle_width_l1067_106714


namespace line_equation_l1067_106715

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a line has equal intercepts on both axes -/
def Line.equalIntercepts (l : Line) : Prop :=
  l.a * l.c = -l.b * l.c

theorem line_equation (P : Point) (l : Line) :
  P.x = 2 ∧ P.y = 1 ∧
  P.onLine l ∧
  l.perpendicular { a := 1, b := -1, c := 1 } ∧
  l.equalIntercepts →
  (l = { a := 1, b := 1, c := -3 } ∨ l = { a := 1, b := -2, c := 0 }) :=
sorry

end line_equation_l1067_106715


namespace opposite_of_negative_seven_l1067_106797

theorem opposite_of_negative_seven :
  ∃ x : ℤ, ((-7 : ℤ) + x = 0) ∧ x = 7 := by
  sorry

end opposite_of_negative_seven_l1067_106797
