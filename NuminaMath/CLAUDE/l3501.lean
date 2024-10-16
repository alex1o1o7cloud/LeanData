import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_max_value_l3501_350180

/-- Given a quadratic x^2 - tx + q with roots α and β, where 
    α + β = α^2 + β^2 = α^3 + β^3 = ⋯ = α^2010 + β^2010,
    the maximum value of 1/α^2012 + 1/β^2012 is 2. -/
theorem quadratic_roots_max_value (t q α β : ℝ) : 
  α^2 - t*α + q = 0 →
  β^2 - t*β + q = 0 →
  (∀ n : ℕ, n ≤ 2010 → α^n + β^n = α + β) →
  (∃ M : ℝ, M = 2 ∧ 
    ∀ t' q' α' β' : ℝ, 
      α'^2 - t'*α' + q' = 0 →
      β'^2 - t'*β' + q' = 0 →
      (∀ n : ℕ, n ≤ 2010 → α'^n + β'^n = α' + β') →
      1/α'^2012 + 1/β'^2012 ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_max_value_l3501_350180


namespace NUMINAMATH_CALUDE_total_carrots_grown_l3501_350157

/-- The total number of carrots grown by Joan, Jessica, and Michael is 77. -/
theorem total_carrots_grown (joan_carrots : ℕ) (jessica_carrots : ℕ) (michael_carrots : ℕ)
  (h1 : joan_carrots = 29)
  (h2 : jessica_carrots = 11)
  (h3 : michael_carrots = 37) :
  joan_carrots + jessica_carrots + michael_carrots = 77 :=
by sorry

end NUMINAMATH_CALUDE_total_carrots_grown_l3501_350157


namespace NUMINAMATH_CALUDE_xia_shared_hundred_stickers_l3501_350101

/-- The number of stickers Xia shared with her friends -/
def shared_stickers (total : ℕ) (sheets_left : ℕ) (stickers_per_sheet : ℕ) : ℕ :=
  total - (sheets_left * stickers_per_sheet)

/-- Theorem stating that Xia shared 100 stickers with her friends -/
theorem xia_shared_hundred_stickers :
  shared_stickers 150 5 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_xia_shared_hundred_stickers_l3501_350101


namespace NUMINAMATH_CALUDE_julie_reading_ratio_l3501_350137

theorem julie_reading_ratio : 
  ∀ (total_pages pages_yesterday pages_tomorrow : ℕ) (pages_today : ℕ),
    total_pages = 120 →
    pages_yesterday = 12 →
    pages_tomorrow = 42 →
    2 * pages_tomorrow = total_pages - pages_yesterday - pages_today →
    pages_today / pages_yesterday = 2 := by
  sorry

end NUMINAMATH_CALUDE_julie_reading_ratio_l3501_350137


namespace NUMINAMATH_CALUDE_f_inequalities_l3501_350116

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x - a

theorem f_inequalities (a : ℝ) :
  (a < -1 → {x : ℝ | f a x < 0} = Set.Ioo a (-1)) ∧
  (a = -1 → {x : ℝ | f a x < 0} = ∅) ∧
  (a > -1 → {x : ℝ | f a x < 0} = Set.Ioo (-1) a) ∧
  ({x : ℝ | x^3 * f 2 x > 0} = Set.Ioo (-1) 0 ∪ Set.Ioi 2) :=
by sorry


end NUMINAMATH_CALUDE_f_inequalities_l3501_350116


namespace NUMINAMATH_CALUDE_car_license_combinations_l3501_350167

def letter_choices : ℕ := 2
def digit_choices : ℕ := 10
def num_digits : ℕ := 6

def total_license_combinations : ℕ := letter_choices * digit_choices ^ num_digits

theorem car_license_combinations :
  total_license_combinations = 2000000 := by
  sorry

end NUMINAMATH_CALUDE_car_license_combinations_l3501_350167


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3501_350126

theorem fractional_equation_solution : 
  ∃ x : ℝ, (x * (x - 2) ≠ 0) ∧ (5 / (x - 2) = 3 / x) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3501_350126


namespace NUMINAMATH_CALUDE_fraction_equality_l3501_350133

theorem fraction_equality (m : ℝ) (h : (m - 1) / m = 3) : (m^2 + 1) / m^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3501_350133


namespace NUMINAMATH_CALUDE_product_of_numbers_l3501_350149

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 18) (sum_squares_eq : x^2 + y^2 = 180) : x * y = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3501_350149


namespace NUMINAMATH_CALUDE_stratified_sampling_proportional_l3501_350130

/-- Represents the number of employees in each title category -/
structure TitleCounts where
  senior : ℕ
  intermediate : ℕ
  junior : ℕ

/-- Represents the number of employees selected in each title category -/
structure SelectedCounts where
  senior : ℕ
  intermediate : ℕ
  junior : ℕ

/-- Checks if the selected counts are proportional to the total counts -/
def isProportionalSelection (total : TitleCounts) (selected : SelectedCounts) (sampleSize : ℕ) : Prop :=
  selected.senior * (total.senior + total.intermediate + total.junior) = 
    total.senior * sampleSize ∧
  selected.intermediate * (total.senior + total.intermediate + total.junior) = 
    total.intermediate * sampleSize ∧
  selected.junior * (total.senior + total.intermediate + total.junior) = 
    total.junior * sampleSize

theorem stratified_sampling_proportional 
  (total : TitleCounts)
  (selected : SelectedCounts)
  (h1 : total.senior = 15)
  (h2 : total.intermediate = 45)
  (h3 : total.junior = 90)
  (h4 : selected.senior = 3)
  (h5 : selected.intermediate = 9)
  (h6 : selected.junior = 18) :
  isProportionalSelection total selected 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportional_l3501_350130


namespace NUMINAMATH_CALUDE_hens_and_cows_l3501_350198

/-- Given a total number of animals and feet, calculates the number of hens -/
def number_of_hens (total_animals : ℕ) (total_feet : ℕ) : ℕ :=
  total_animals - (total_feet - 2 * total_animals) / 2

/-- Theorem stating that given 50 animals with 140 feet, where hens have 2 feet
    and cows have 4 feet, the number of hens is 30 -/
theorem hens_and_cows (total_animals : ℕ) (total_feet : ℕ) 
  (h1 : total_animals = 50)
  (h2 : total_feet = 140) :
  number_of_hens total_animals total_feet = 30 := by
  sorry

#eval number_of_hens 50 140  -- Should output 30

end NUMINAMATH_CALUDE_hens_and_cows_l3501_350198


namespace NUMINAMATH_CALUDE_nancys_contribution_is_36_l3501_350170

/-- The number of bottle caps Marilyn had initially -/
def initial_caps : ℝ := 51.0

/-- The number of bottle caps Marilyn had after Nancy's contribution -/
def final_caps : ℝ := 87.0

/-- The number of bottle caps Nancy gave to Marilyn -/
def nancys_contribution : ℝ := final_caps - initial_caps

theorem nancys_contribution_is_36 : nancys_contribution = 36 := by
  sorry

end NUMINAMATH_CALUDE_nancys_contribution_is_36_l3501_350170


namespace NUMINAMATH_CALUDE_shopping_remaining_amount_l3501_350165

def initial_amount : ℚ := 74
def sweater_cost : ℚ := 9
def tshirt_cost : ℚ := 11
def shoes_cost : ℚ := 30
def refund_percentage : ℚ := 90 / 100

theorem shopping_remaining_amount :
  initial_amount - (sweater_cost + tshirt_cost + shoes_cost * (1 - refund_percentage)) = 51 := by
  sorry

end NUMINAMATH_CALUDE_shopping_remaining_amount_l3501_350165


namespace NUMINAMATH_CALUDE_total_outstanding_credit_l3501_350145

/-- The total outstanding consumer installment credit in billions of dollars -/
def total_credit : ℝ := 416.67

/-- The percentage of automobile installment credit in total consumer installment credit -/
def auto_credit_percentage : ℝ := 36

/-- The amount of credit extended by automobile finance companies in billions of dollars -/
def auto_finance_credit : ℝ := 75

/-- Theorem stating the total outstanding consumer installment credit -/
theorem total_outstanding_credit : 
  total_credit = (2 * auto_finance_credit) / (auto_credit_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_total_outstanding_credit_l3501_350145


namespace NUMINAMATH_CALUDE_interior_angles_sum_l3501_350110

/-- The sum of interior angles of a triangle in degrees -/
def triangle_angle_sum : ℝ := 180

/-- The number of triangles a quadrilateral can be divided into -/
def quadrilateral_triangles : ℕ := 2

/-- The number of triangles a pentagon can be divided into -/
def pentagon_triangles : ℕ := 3

/-- The number of triangles a convex n-gon can be divided into -/
def n_gon_triangles (n : ℕ) : ℕ := n - 2

/-- The sum of interior angles of a quadrilateral -/
def quadrilateral_angle_sum : ℝ := triangle_angle_sum * quadrilateral_triangles

/-- The sum of interior angles of a convex pentagon -/
def pentagon_angle_sum : ℝ := triangle_angle_sum * pentagon_triangles

/-- The sum of interior angles of a convex n-gon -/
def n_gon_angle_sum (n : ℕ) : ℝ := triangle_angle_sum * n_gon_triangles n

theorem interior_angles_sum :
  (quadrilateral_angle_sum = 360) ∧
  (pentagon_angle_sum = 540) ∧
  (∀ n : ℕ, n_gon_angle_sum n = 180 * (n - 2)) :=
sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l3501_350110


namespace NUMINAMATH_CALUDE_no_valid_formations_l3501_350195

/-- Represents a rectangular formation for a marching band -/
structure Formation where
  rows : ℕ
  cols : ℕ

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Checks if a formation is valid according to the given conditions -/
def isValidFormation (f : Formation) : Prop :=
  f.rows * f.cols = 240 ∧
  isPrime f.rows ∧
  8 ≤ f.cols ∧ f.cols ≤ 30

/-- The set of all valid formations -/
def validFormations : Set Formation :=
  {f : Formation | isValidFormation f}

/-- Theorem stating that there are no valid formations -/
theorem no_valid_formations : validFormations = ∅ := by
  sorry

#check no_valid_formations

end NUMINAMATH_CALUDE_no_valid_formations_l3501_350195


namespace NUMINAMATH_CALUDE_exists_linear_bound_l3501_350125

def Color := Bool

def is_valid_coloring (coloring : ℕ+ → Color) : Prop :=
  ∀ n : ℕ+, coloring n = true ∨ coloring n = false

structure ColoredIntegerFunction where
  f : ℕ+ → ℕ+
  coloring : ℕ+ → Color
  is_valid_coloring : is_valid_coloring coloring
  monotone : ∀ x y : ℕ+, x ≤ y → f x ≤ f y
  color_additive : ∀ x y z : ℕ+, 
    coloring x = coloring y ∧ coloring y = coloring z → 
    x + y = z → f x + f y = f z

theorem exists_linear_bound (cf : ColoredIntegerFunction) : 
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℕ+, (cf.f x : ℝ) ≤ a * x :=
sorry

end NUMINAMATH_CALUDE_exists_linear_bound_l3501_350125


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3501_350162

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the point corresponding to i + i^2
def point : ℂ := i + i^2

-- Theorem stating that the point is in the second quadrant
theorem point_in_second_quadrant : 
  Complex.re point < 0 ∧ Complex.im point > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3501_350162


namespace NUMINAMATH_CALUDE_garage_door_properties_l3501_350160

/-- Represents a garage door mechanism -/
structure GarageDoor where
  AC : ℝ
  BC : ℝ
  CY : ℝ
  AX : ℝ
  BD : ℝ

/-- Properties of the garage door mechanism -/
def isValidGarageDoor (door : GarageDoor) : Prop :=
  door.AC = 0.5 ∧ door.BC = 0.5 ∧ door.CY = 0.5 ∧ door.AX = 1 ∧ door.BD = 2

/-- Calculate CR given XS -/
def calculateCR (door : GarageDoor) (XS : ℝ) : ℝ := sorry

/-- Check if Y's height remains constant -/
def isYHeightConstant (door : GarageDoor) : Prop := sorry

/-- Calculate DT given XT -/
def calculateDT (door : GarageDoor) (XT : ℝ) : ℝ := sorry

/-- Main theorem about the garage door mechanism -/
theorem garage_door_properties (door : GarageDoor) 
  (h : isValidGarageDoor door) : 
  calculateCR door 0.2 = 0.1 ∧ 
  isYHeightConstant door ∧ 
  calculateDT door 0.4 = 0.6 := by sorry

end NUMINAMATH_CALUDE_garage_door_properties_l3501_350160


namespace NUMINAMATH_CALUDE_dividend_calculation_l3501_350131

/-- Calculate the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (share_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : share_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : dividend_rate = 0.05) :
  let actual_share_price := share_value * (1 + premium_rate)
  let num_shares := investment / actual_share_price
  let dividend_per_share := share_value * dividend_rate
  num_shares * dividend_per_share = 600 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3501_350131


namespace NUMINAMATH_CALUDE_isosceles_triangle_trajectory_l3501_350140

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: The trajectory of the third vertex of an isosceles triangle -/
theorem isosceles_triangle_trajectory (C : Point) : 
  let A : Point := ⟨2, 4⟩
  let B : Point := ⟨2, 8⟩
  (squaredDistance A B = squaredDistance A C) → 
  C.x ≠ 2 →
  (C.x - 2)^2 + (C.y - 4)^2 = 16 :=
by
  sorry

#check isosceles_triangle_trajectory

end NUMINAMATH_CALUDE_isosceles_triangle_trajectory_l3501_350140


namespace NUMINAMATH_CALUDE_odd_integers_between_fractions_l3501_350182

theorem odd_integers_between_fractions : 
  let lower_bound : ℚ := 17 / 4
  let upper_bound : ℚ := 35 / 2
  ∃ (S : Finset ℤ), (∀ n ∈ S, (n : ℚ) > lower_bound ∧ (n : ℚ) < upper_bound ∧ Odd n) ∧ 
                    (∀ n : ℤ, (n : ℚ) > lower_bound ∧ (n : ℚ) < upper_bound ∧ Odd n → n ∈ S) ∧
                    Finset.card S = 7 :=
by sorry

end NUMINAMATH_CALUDE_odd_integers_between_fractions_l3501_350182


namespace NUMINAMATH_CALUDE_no_common_points_implies_b_range_l3501_350177

theorem no_common_points_implies_b_range 
  (f g : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x = x^2 - a*x) 
  (k : ∀ x : ℝ, g x = b + a * Real.log (x - 1)) 
  (a_ge_one : a ≥ 1) 
  (no_common_points : ∀ x : ℝ, f x ≠ g x) : 
  b < 3/4 + Real.log 2 := by
sorry

end NUMINAMATH_CALUDE_no_common_points_implies_b_range_l3501_350177


namespace NUMINAMATH_CALUDE_largest_factorable_m_l3501_350113

/-- A quadratic expression of the form 3x^2 + mx - 60 -/
def quadratic (m : ℤ) (x : ℤ) : ℤ := 3 * x^2 + m * x - 60

/-- Checks if a quadratic expression can be factored into two linear factors with integer coefficients -/
def is_factorable (m : ℤ) : Prop :=
  ∃ (a b c d : ℤ), ∀ x, quadratic m x = (a * x + b) * (c * x + d)

/-- The largest value of m for which the quadratic is factorable -/
def largest_m : ℤ := 57

theorem largest_factorable_m :
  (is_factorable largest_m) ∧
  (∀ m : ℤ, m > largest_m → ¬(is_factorable m)) := by sorry

end NUMINAMATH_CALUDE_largest_factorable_m_l3501_350113


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_five_l3501_350120

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem unique_square_divisible_by_five :
  ∃! x : ℕ, is_square x ∧ x % 5 = 0 ∧ 50 < x ∧ x < 120 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_five_l3501_350120


namespace NUMINAMATH_CALUDE_marbles_lost_calculation_l3501_350103

def initial_marbles : ℕ := 15
def marbles_found : ℕ := 9
def extra_marbles_lost : ℕ := 14

theorem marbles_lost_calculation :
  marbles_found + extra_marbles_lost = 23 :=
by sorry

end NUMINAMATH_CALUDE_marbles_lost_calculation_l3501_350103


namespace NUMINAMATH_CALUDE_parabola_solution_l3501_350168

/-- The parabola C: y^2 = 6x with focus F, and a point M(x,y) on C where |MF| = 5/2 and y > 0 -/
def parabola_problem (x y : ℝ) : Prop :=
  y^2 = 6*x ∧ y > 0 ∧ (x + 3/2)^2 + y^2 = (5/2)^2

/-- The coordinates of point M are (1, √6) -/
theorem parabola_solution :
  ∀ x y : ℝ, parabola_problem x y → x = 1 ∧ y = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_parabola_solution_l3501_350168


namespace NUMINAMATH_CALUDE_greatest_perfect_square_under_1000_l3501_350132

theorem greatest_perfect_square_under_1000 : 
  ∀ n : ℕ, n < 1000 → n ≤ 961 ∨ ¬∃ m : ℕ, n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_perfect_square_under_1000_l3501_350132


namespace NUMINAMATH_CALUDE_sweet_potatoes_remaining_l3501_350109

def sweet_potatoes_problem (initial : ℕ) (sold_adams : ℕ) (sold_lenon : ℕ) (traded : ℕ) (donated : ℕ) : ℕ :=
  initial - (sold_adams + sold_lenon + traded + donated)

theorem sweet_potatoes_remaining :
  sweet_potatoes_problem 80 20 15 10 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sweet_potatoes_remaining_l3501_350109


namespace NUMINAMATH_CALUDE_triangle_perimeter_not_88_l3501_350166

theorem triangle_perimeter_not_88 (a b x : ℝ) (h1 : a = 18) (h2 : b = 25) (h3 : a + b > x) (h4 : a + x > b) (h5 : b + x > a) : a + b + x ≠ 88 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_not_88_l3501_350166


namespace NUMINAMATH_CALUDE_iphone_price_decrease_l3501_350197

theorem iphone_price_decrease (initial_price : ℝ) (second_month_decrease : ℝ) (final_price : ℝ) :
  initial_price = 1000 →
  second_month_decrease = 20 →
  final_price = 720 →
  ∃ (first_month_decrease : ℝ),
    first_month_decrease = 10 ∧
    final_price = initial_price * (1 - first_month_decrease / 100) * (1 - second_month_decrease / 100) :=
by sorry


end NUMINAMATH_CALUDE_iphone_price_decrease_l3501_350197


namespace NUMINAMATH_CALUDE_largest_integer_a_l3501_350102

theorem largest_integer_a : ∃ (a : ℤ), 
  (∀ x : ℝ, -π/2 < x ∧ x < π/2 → 
    a^2 - 15*a - (Real.tan x - 1)*(Real.tan x + 2)*(Real.tan x + 5)*(Real.tan x + 8) < 35) ∧ 
  (∀ b : ℤ, b > a → 
    ∃ x : ℝ, -π/2 < x ∧ x < π/2 ∧ 
      b^2 - 15*b - (Real.tan x - 1)*(Real.tan x + 2)*(Real.tan x + 5)*(Real.tan x + 8) ≥ 35) ∧
  a = 10 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_a_l3501_350102


namespace NUMINAMATH_CALUDE_cannon_probability_l3501_350144

theorem cannon_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.5) (h2 : p2 = 0.8) (h3 : p3 = 0.7) : 
  p1 * p2 * p3 = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_cannon_probability_l3501_350144


namespace NUMINAMATH_CALUDE_train_length_calculation_l3501_350124

theorem train_length_calculation (train_speed_kmph : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed_kmph = 72 →
  platform_length = 350 →
  crossing_time = 26 →
  let train_speed_mps := train_speed_kmph * (5/18)
  let total_distance := train_speed_mps * crossing_time
  let train_length := total_distance - platform_length
  train_length = 170 := by
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3501_350124


namespace NUMINAMATH_CALUDE_competition_problem_l3501_350114

theorem competition_problem : ((7^2 - 3^2)^4) = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_competition_problem_l3501_350114


namespace NUMINAMATH_CALUDE_point_always_on_line_l3501_350163

theorem point_always_on_line (m b : ℝ) (h : m * b < 0) :
  0 = m * 2003 + b := by sorry

end NUMINAMATH_CALUDE_point_always_on_line_l3501_350163


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l3501_350176

theorem largest_of_three_consecutive_odds (a b c : ℤ) : 
  (∃ n : ℤ, a = 2*n + 1 ∧ b = 2*n + 3 ∧ c = 2*n + 5) →  -- consecutive odd integers
  (a + b + c = -147) →                                  -- sum is -147
  (max a (max b c) = -47) :=                            -- largest is -47
by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l3501_350176


namespace NUMINAMATH_CALUDE_cd_length_is_nine_l3501_350158

/-- A tetrahedron with specific edge lengths -/
structure Tetrahedron where
  edges : Finset ℝ
  edge_count : edges.card = 6
  edge_values : edges = {9, 15, 22, 35, 40, 44}
  ab_length : 44 ∈ edges

/-- The length of CD in the tetrahedron -/
def cd_length (t : Tetrahedron) : ℝ := 9

/-- Theorem stating that CD length is 9 in the given tetrahedron -/
theorem cd_length_is_nine (t : Tetrahedron) : cd_length t = 9 := by
  sorry

end NUMINAMATH_CALUDE_cd_length_is_nine_l3501_350158


namespace NUMINAMATH_CALUDE_constants_are_like_terms_l3501_350181

/-- Two terms are considered like terms if they have the same variables raised to the same powers, or if they are both constants. -/
def like_terms (t1 t2 : ℝ) : Prop :=
  (∃ (c1 c2 : ℝ), t1 = c1 ∧ t2 = c2) ∨ 
  (∃ (c1 c2 : ℝ) (f : ℝ → ℝ), t1 = c1 * f 0 ∧ t2 = c2 * f 0)

/-- Constants 0 and π are like terms. -/
theorem constants_are_like_terms : like_terms 0 π := by
  sorry

end NUMINAMATH_CALUDE_constants_are_like_terms_l3501_350181


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3501_350159

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : x + |y| - y = 12) : 
  x + y = 3.6 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3501_350159


namespace NUMINAMATH_CALUDE_overlap_probability_4x4_on_8x8_l3501_350123

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Represents a square on the chessboard -/
structure Square :=
  (side : ℕ)

/-- The number of possible placements for a square on a chessboard -/
def num_placements (b : Chessboard) (s : Square) : ℕ :=
  (b.size - s.side + 1) * (b.size - s.side + 1)

/-- The number of non-overlapping placements for two squares -/
def num_non_overlapping (b : Chessboard) (s : Square) : ℕ :=
  2 * (b.size - s.side + 1) * (b.size - s.side + 1) - 4

/-- The probability of two squares overlapping -/
def overlap_probability (b : Chessboard) (s : Square) : ℚ :=
  1 - (num_non_overlapping b s : ℚ) / ((num_placements b s * num_placements b s) : ℚ)

theorem overlap_probability_4x4_on_8x8 :
  overlap_probability (Chessboard.mk 8) (Square.mk 4) = 529 / 625 := by
  sorry

end NUMINAMATH_CALUDE_overlap_probability_4x4_on_8x8_l3501_350123


namespace NUMINAMATH_CALUDE_inequality_proof_l3501_350192

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3501_350192


namespace NUMINAMATH_CALUDE_miss_aisha_height_l3501_350108

/-- Miss Aisha's height calculation -/
theorem miss_aisha_height :
  ∀ (h : ℝ),
  h > 0 →
  h / 3 + h / 4 + 25 = h →
  h = 60 := by
sorry

end NUMINAMATH_CALUDE_miss_aisha_height_l3501_350108


namespace NUMINAMATH_CALUDE_inequality_chain_l3501_350173

theorem inequality_chain (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l3501_350173


namespace NUMINAMATH_CALUDE_three_gorges_dam_capacity_scientific_notation_l3501_350104

-- Define the original number
def original_number : ℝ := 18200000

-- Define the scientific notation components
def coefficient : ℝ := 1.82
def exponent : ℤ := 7

-- Theorem statement
theorem three_gorges_dam_capacity_scientific_notation :
  original_number = coefficient * (10 : ℝ) ^ exponent :=
by sorry

end NUMINAMATH_CALUDE_three_gorges_dam_capacity_scientific_notation_l3501_350104


namespace NUMINAMATH_CALUDE_number_of_girls_l3501_350186

def number_of_boys : ℕ := 5
def committee_size : ℕ := 4
def boys_in_committee : ℕ := 2
def girls_in_committee : ℕ := 2
def total_committees : ℕ := 150

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem number_of_girls : 
  ∃ g : ℕ, 
    choose number_of_boys boys_in_committee * choose g girls_in_committee = total_committees ∧ 
    g = 6 :=
sorry

end NUMINAMATH_CALUDE_number_of_girls_l3501_350186


namespace NUMINAMATH_CALUDE_volunteers_needed_l3501_350139

/-- Represents the number of volunteers needed for the school Christmas play --/
def total_volunteers_needed : ℕ := 100

/-- Represents the number of math classes --/
def math_classes : ℕ := 5

/-- Represents the number of students volunteering from each math class --/
def students_per_class : ℕ := 4

/-- Represents the total number of teachers volunteering --/
def teachers_volunteering : ℕ := 10

/-- Represents the number of teachers skilled in carpentry --/
def teachers_carpentry : ℕ := 3

/-- Represents the total number of parents volunteering --/
def parents_volunteering : ℕ := 15

/-- Represents the number of parents experienced with lighting and sound --/
def parents_lighting_sound : ℕ := 6

/-- Represents the additional number of volunteers needed with carpentry skills --/
def additional_carpentry_needed : ℕ := 8

/-- Represents the additional number of volunteers needed with lighting and sound experience --/
def additional_lighting_sound_needed : ℕ := 10

/-- Theorem stating that 9 more volunteers are needed to meet the requirements --/
theorem volunteers_needed : 
  (math_classes * students_per_class + teachers_volunteering + parents_volunteering) +
  (additional_carpentry_needed - teachers_carpentry) + 
  (additional_lighting_sound_needed - parents_lighting_sound) = 9 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_needed_l3501_350139


namespace NUMINAMATH_CALUDE_ice_cream_scoops_l3501_350188

/-- Proves that Aaron and Carson each bought 8 scoops of ice cream given the problem conditions --/
theorem ice_cream_scoops (aaron_savings : ℚ) (carson_savings : ℚ) 
  (restaurant_bill_fraction : ℚ) (service_charge : ℚ) (ice_cream_cost : ℚ) 
  (leftover_money : ℚ) :
  aaron_savings = 150 →
  carson_savings = 150 →
  restaurant_bill_fraction = 3/4 →
  service_charge = 15/100 →
  ice_cream_cost = 4 →
  leftover_money = 4 →
  ∃ (scoops : ℕ), scoops = 8 ∧ 
    (aaron_savings + carson_savings) * restaurant_bill_fraction + 
    2 * scoops * ice_cream_cost + 2 * leftover_money = 
    aaron_savings + carson_savings :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_scoops_l3501_350188


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3501_350194

theorem polynomial_coefficient_sum (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₁ - 2*a₂ + 3*a₃ - 4*a₄ = -216 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3501_350194


namespace NUMINAMATH_CALUDE_area_pda_equals_sqrt_vw_l3501_350174

-- Define the rectangular pyramid
structure RectangularPyramid where
  -- Lengths of edges
  a : ℝ
  b : ℝ
  h : ℝ
  -- Areas of triangles
  u : ℝ
  v : ℝ
  w : ℝ
  -- Conditions
  pos_a : 0 < a
  pos_b : 0 < b
  pos_h : 0 < h
  area_pab : u = (1/2) * a * h
  area_pbc : v = (1/2) * a * b
  area_pcd : w = (1/2) * b * h

-- Theorem statement
theorem area_pda_equals_sqrt_vw (pyramid : RectangularPyramid) :
  (1/2) * pyramid.b * pyramid.h = Real.sqrt (pyramid.v * pyramid.w) :=
sorry

end NUMINAMATH_CALUDE_area_pda_equals_sqrt_vw_l3501_350174


namespace NUMINAMATH_CALUDE_carlos_summer_reading_l3501_350143

/-- Carlos' summer reading challenge -/
theorem carlos_summer_reading 
  (july_books august_books total_goal : ℕ) 
  (h1 : july_books = 28)
  (h2 : august_books = 30)
  (h3 : total_goal = 100) :
  total_goal - (july_books + august_books) = 42 := by
  sorry

end NUMINAMATH_CALUDE_carlos_summer_reading_l3501_350143


namespace NUMINAMATH_CALUDE_sterilization_tank_capacity_l3501_350153

/-- The total capacity of the sterilization tank in gallons -/
def tank_capacity : ℝ := 100

/-- The initial concentration of bleach in the tank as a decimal -/
def initial_concentration : ℝ := 0.02

/-- The target concentration of bleach in the tank as a decimal -/
def target_concentration : ℝ := 0.05

/-- The amount of solution drained and replaced with pure bleach in gallons -/
def drained_amount : ℝ := 3.0612244898

theorem sterilization_tank_capacity :
  let initial_bleach := initial_concentration * tank_capacity
  let drained_bleach := initial_concentration * drained_amount
  let added_bleach := drained_amount
  let final_bleach := initial_bleach - drained_bleach + added_bleach
  final_bleach = target_concentration * tank_capacity := by
  sorry

end NUMINAMATH_CALUDE_sterilization_tank_capacity_l3501_350153


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l3501_350121

-- Define the income and savings
def income : ℕ := 40000
def savings : ℕ := 5000

-- Define the expenditure
def expenditure : ℕ := income - savings

-- Define the ratio of income to expenditure
def ratio : ℕ × ℕ := (income / (income.gcd expenditure), expenditure / (income.gcd expenditure))

-- Theorem statement
theorem income_expenditure_ratio :
  ratio = (8, 7) := by sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l3501_350121


namespace NUMINAMATH_CALUDE_escalator_time_l3501_350115

/-- Time taken to cover the length of an escalator -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (length : ℝ) 
  (h1 : escalator_speed = 11)
  (h2 : person_speed = 3)
  (h3 : length = 126) :
  length / (escalator_speed + person_speed) = 9 := by
sorry

end NUMINAMATH_CALUDE_escalator_time_l3501_350115


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l3501_350128

theorem smallest_gcd_multiple (a b : ℕ+) (h : Nat.gcd a b = 10) :
  (∃ (a' b' : ℕ+), Nat.gcd a' b' = 10 ∧ Nat.gcd (12 * a') (20 * b') = 40) ∧
  (∀ (a'' b'' : ℕ+), Nat.gcd a'' b'' = 10 → Nat.gcd (12 * a'') (20 * b'') ≥ 40) :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l3501_350128


namespace NUMINAMATH_CALUDE_polynomial_standard_form_l3501_350169

theorem polynomial_standard_form :
  ∀ (a b x : ℝ),
  (a - b) * (a + b) * (a^2 + a*b + b^2) * (a^2 - a*b + b^2) = a^6 - b^6 ∧
  (x - 1)^3 * (x + 1)^2 * (x^2 + 1) * (x^2 + x + 1) = x^9 - x^7 - x^8 - x^5 + x^4 + x^3 + x^2 - 1 ∧
  (x^4 - x^2 + 1) * (x^2 - x + 1) * (x^2 + x + 1) = x^8 + x^4 + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_standard_form_l3501_350169


namespace NUMINAMATH_CALUDE_coefficient_a2_value_l3501_350107

/-- Given a complex number z and a polynomial expansion of (x-z)^4,
    prove that the coefficient of x^2 is -3 + 3√3i. -/
theorem coefficient_a2_value (z : ℂ) (a₀ a₁ a₂ a₃ a₄ : ℂ) :
  z = (1/2 : ℂ) + (Complex.I * Real.sqrt 3) / 2 →
  (fun x : ℂ ↦ (x - z)^4) = (fun x : ℂ ↦ a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) →
  a₂ = -3 + Complex.I * (3 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_coefficient_a2_value_l3501_350107


namespace NUMINAMATH_CALUDE_intersection_of_planes_and_line_l3501_350185

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relation for intersection between planes
variable (intersect : Plane → Plane → Prop)

-- Define the relation for perpendicularity between planes
variable (perpendicular : Plane → Plane → Prop)

-- Define the relation for a line lying in a plane
variable (lies_in : Line → Plane → Prop)

-- Define the relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Define the relation for perpendicular lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the theorem
theorem intersection_of_planes_and_line 
  (α β : Plane) (m : Line)
  (h1 : intersect α β)
  (h2 : ¬ perpendicular α β)
  (h3 : lies_in m α) :
  (∃ (n : Line), lies_in n β ∧ ¬ (∀ (n : Line), lies_in n β → parallel m n)) ∧
  (∃ (p : Line), lies_in p β ∧ perpendicular_lines m p) :=
sorry

end NUMINAMATH_CALUDE_intersection_of_planes_and_line_l3501_350185


namespace NUMINAMATH_CALUDE_initial_group_size_l3501_350100

theorem initial_group_size (initial_avg : ℝ) (new_avg : ℝ) (weight1 : ℝ) (weight2 : ℝ) :
  initial_avg = 48 →
  new_avg = 51 →
  weight1 = 78 →
  weight2 = 93 →
  ∃ n : ℕ, n * initial_avg + weight1 + weight2 = (n + 2) * new_avg ∧ n = 23 :=
by sorry

end NUMINAMATH_CALUDE_initial_group_size_l3501_350100


namespace NUMINAMATH_CALUDE_floor_x_times_x_equals_90_l3501_350106

theorem floor_x_times_x_equals_90 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 90 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_x_times_x_equals_90_l3501_350106


namespace NUMINAMATH_CALUDE_max_sum_x_y_is_seven_l3501_350189

theorem max_sum_x_y_is_seven (x y : ℕ+) (h : x.val^4 = (x.val - 1) * (y.val^3 - 23) - 1) :
  x.val + y.val ≤ 7 ∧ ∃ (x₀ y₀ : ℕ+), x₀.val^4 = (x₀.val - 1) * (y₀.val^3 - 23) - 1 ∧ x₀.val + y₀.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_x_y_is_seven_l3501_350189


namespace NUMINAMATH_CALUDE_rocket_arrangements_l3501_350184

def word : String := "ROCKET"

theorem rocket_arrangements : 
  (∃ (c : Char), c ∈ word.data ∧ 
    (word.data.count c = 2) ∧ 
    (∀ (d : Char), d ∈ word.data ∧ d ≠ c → word.data.count d = 1)) →
  (Nat.factorial (word.length + 1) / 2 = 2520) :=
by sorry

end NUMINAMATH_CALUDE_rocket_arrangements_l3501_350184


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3501_350117

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-2, -1, 0}
def B : Set Int := {0, 1, 2}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3501_350117


namespace NUMINAMATH_CALUDE_consecutive_sum_fifteen_l3501_350161

theorem consecutive_sum_fifteen (n : ℤ) : n + (n + 1) + (n + 2) = 15 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_fifteen_l3501_350161


namespace NUMINAMATH_CALUDE_french_books_count_l3501_350193

/-- The number of English books -/
def num_english_books : ℕ := 11

/-- The total number of arrangement ways -/
def total_arrangements : ℕ := 220

/-- The number of French books -/
def num_french_books : ℕ := 3

/-- The number of slots for French books -/
def num_slots : ℕ := num_english_books + 1

theorem french_books_count :
  (Nat.choose num_slots num_french_books = total_arrangements) ∧
  (∀ k : ℕ, k ≠ num_french_books → Nat.choose num_slots k ≠ total_arrangements) :=
sorry

end NUMINAMATH_CALUDE_french_books_count_l3501_350193


namespace NUMINAMATH_CALUDE_equal_areas_imply_equal_dimensions_l3501_350190

theorem equal_areas_imply_equal_dimensions (square_side : ℝ) (rect_width : ℝ) (tri_base : ℝ) 
  (h1 : square_side = 4)
  (h2 : rect_width = 4)
  (h3 : tri_base = 8)
  (h4 : square_side ^ 2 = rect_width * (square_side ^ 2 / rect_width))
  (h5 : square_side ^ 2 = (tri_base * (2 * square_side ^ 2 / tri_base)) / 2) :
  square_side ^ 2 / rect_width = 4 ∧ 2 * square_side ^ 2 / tri_base = 4 :=
by sorry

end NUMINAMATH_CALUDE_equal_areas_imply_equal_dimensions_l3501_350190


namespace NUMINAMATH_CALUDE_max_vertices_with_unique_distances_five_vertices_with_unique_distances_exist_l3501_350178

/-- Represents a regular polygon with n sides inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A function to calculate the distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Checks if all pairwise distances in a list of points are unique -/
def allDistancesUnique (points : List (ℝ × ℝ)) : Prop := sorry

/-- The main theorem -/
theorem max_vertices_with_unique_distances
  (polygon : RegularPolygon 21)
  (selectedVertices : List (Fin 21)) :
  (allDistancesUnique (selectedVertices.map polygon.vertices)) →
  selectedVertices.length ≤ 5 := sorry

/-- The maximum number of vertices with unique distances is indeed achievable -/
theorem five_vertices_with_unique_distances_exist
  (polygon : RegularPolygon 21) :
  ∃ (selectedVertices : List (Fin 21)),
    selectedVertices.length = 5 ∧
    allDistancesUnique (selectedVertices.map polygon.vertices) := sorry

end NUMINAMATH_CALUDE_max_vertices_with_unique_distances_five_vertices_with_unique_distances_exist_l3501_350178


namespace NUMINAMATH_CALUDE_area_of_four_triangles_l3501_350118

/-- The combined area of four right triangles with legs of 4 and 3 units is 24 square units. -/
theorem area_of_four_triangles :
  let triangle_area := (1 / 2) * 4 * 3
  4 * triangle_area = 24 := by sorry

end NUMINAMATH_CALUDE_area_of_four_triangles_l3501_350118


namespace NUMINAMATH_CALUDE_line_through_point_l3501_350199

theorem line_through_point (k : ℚ) :
  (1 - 3 * k * (1/2) = 10 * 3) ↔ (k = -58/3) := by sorry

end NUMINAMATH_CALUDE_line_through_point_l3501_350199


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3501_350179

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ≠ 0 ∧ a.1 = t * b.1 ∧ a.2 = t * b.2

/-- Given vectors a and b, if they are parallel, then k = 1/2 -/
theorem parallel_vectors_k_value (k : ℝ) :
  let a : ℝ × ℝ := (1, k)
  let b : ℝ × ℝ := (2, 1)
  are_parallel a b → k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3501_350179


namespace NUMINAMATH_CALUDE_line_y_coordinate_l3501_350155

/-- Given a line passing through points (10, y₁) and (x₂, -8), with an x-intercept at (4, 0), prove that y₁ = -8 -/
theorem line_y_coordinate (y₁ x₂ : ℝ) : 
  (∃ m b : ℝ, 
    (y₁ = m * 10 + b) ∧ 
    (-8 = m * x₂ + b) ∧ 
    (0 = m * 4 + b)) → 
  y₁ = -8 :=
by sorry

end NUMINAMATH_CALUDE_line_y_coordinate_l3501_350155


namespace NUMINAMATH_CALUDE_gcd_diff_is_square_l3501_350156

theorem gcd_diff_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_gcd_diff_is_square_l3501_350156


namespace NUMINAMATH_CALUDE_total_shirts_count_l3501_350154

/-- Proves that the total number of shirts is 5 given the conditions -/
theorem total_shirts_count : 
  ∀ (total_cost : ℕ) (cheap_shirt_cost : ℕ) (expensive_shirt_cost : ℕ) 
    (cheap_shirt_count : ℕ) (total_shirt_count : ℕ),
  total_cost = 85 →
  cheap_shirt_cost = 15 →
  expensive_shirt_cost = 20 →
  cheap_shirt_count = 3 →
  total_cost = cheap_shirt_cost * cheap_shirt_count + 
               expensive_shirt_cost * (total_shirt_count - cheap_shirt_count) →
  total_shirt_count = 5 := by
sorry

end NUMINAMATH_CALUDE_total_shirts_count_l3501_350154


namespace NUMINAMATH_CALUDE_pizza_slices_per_pizza_l3501_350129

theorem pizza_slices_per_pizza (num_people : ℕ) (slices_per_person : ℕ) (num_pizzas : ℕ) : 
  num_people = 18 → slices_per_person = 3 → num_pizzas = 6 →
  (num_people * slices_per_person) / num_pizzas = 9 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_pizza_l3501_350129


namespace NUMINAMATH_CALUDE_set_A_equals_one_two_l3501_350141

def A : Set ℕ := {x | x^2 - 3*x < 0 ∧ x > 0}

theorem set_A_equals_one_two : A = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_A_equals_one_two_l3501_350141


namespace NUMINAMATH_CALUDE_dividend_calculation_l3501_350148

/-- Calculates the total dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : face_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : dividend_rate = 0.07) :
  let actual_price := face_value * (1 + premium_rate)
  let num_shares := investment / actual_price
  let dividend_per_share := face_value * dividend_rate
  num_shares * dividend_per_share = 840 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3501_350148


namespace NUMINAMATH_CALUDE_calculation_result_solution_set_l3501_350122

-- Problem 1
theorem calculation_result : (Real.pi - 2023) ^ 0 + |-Real.sqrt 3| - 2 * Real.sin (π / 3) = 1 := by sorry

-- Problem 2
def system_of_inequalities (x : ℝ) : Prop :=
  2 * (x + 3) ≥ 8 ∧ x < (x + 4) / 2

theorem solution_set :
  ∀ x : ℝ, system_of_inequalities x ↔ 1 ≤ x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_calculation_result_solution_set_l3501_350122


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_in_U_l3501_350191

def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

theorem complement_of_A_union_B_in_U :
  (U \ (A ∪ B)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_in_U_l3501_350191


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l3501_350171

/-- Represents a 4-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Checks if a tuple represents a valid 4-digit number -/
def isValidFourDigitNumber (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9

/-- Checks if a 4-digit number satisfies the given conditions -/
def satisfiesConditions (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  b = 3 * a ∧ c = a + b ∧ d = 3 * b

theorem unique_four_digit_number :
  ∃! (n : FourDigitNumber), isValidFourDigitNumber n ∧ satisfiesConditions n ∧ n = (1, 3, 4, 9) :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l3501_350171


namespace NUMINAMATH_CALUDE_tax_calculation_l3501_350164

/-- Given a monthly income and a tax rate, calculates the amount paid in taxes -/
def calculate_tax (income : ℝ) (tax_rate : ℝ) : ℝ :=
  income * tax_rate

/-- Proves that for a monthly income of 2120 dollars and a tax rate of 0.4, 
    the amount paid in taxes is 848 dollars -/
theorem tax_calculation :
  calculate_tax 2120 0.4 = 848 := by
sorry

end NUMINAMATH_CALUDE_tax_calculation_l3501_350164


namespace NUMINAMATH_CALUDE_solve_equation_l3501_350196

theorem solve_equation : ∃ x : ℝ, 25 - (4 + 3) = 5 + x ∧ x = 13 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3501_350196


namespace NUMINAMATH_CALUDE_unique_number_between_cube_roots_l3501_350135

theorem unique_number_between_cube_roots : ∃! (n : ℕ), 
  n > 0 ∧ 
  24 ∣ n ∧ 
  (9 : ℝ) < (n : ℝ) ^ (1/3 : ℝ) ∧ 
  (n : ℝ) ^ (1/3 : ℝ) < (9.1 : ℝ) ∧
  n = 744 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_between_cube_roots_l3501_350135


namespace NUMINAMATH_CALUDE_parallel_lines_l3501_350183

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m : ℝ) : Prop :=
  -m = -(3*m - 2)/m

/-- The first line equation: mx + y + 3 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop :=
  m*x + y + 3 = 0

/-- The second line equation: (3m - 2)x + my + 2 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop :=
  (3*m - 2)*x + m*y + 2 = 0

/-- The main theorem: lines are parallel iff m = 1 or m = 2 -/
theorem parallel_lines (m : ℝ) : parallel m ↔ (m = 1 ∨ m = 2) := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_l3501_350183


namespace NUMINAMATH_CALUDE_product_equals_328185_l3501_350150

theorem product_equals_328185 :
  let product := 9 * 11 * 13 * 15 * 17
  ∃ n : ℕ, n < 10 ∧ product = 300000 + n * 10000 + 8000 + 100 + 80 + 5 :=
by
  sorry

end NUMINAMATH_CALUDE_product_equals_328185_l3501_350150


namespace NUMINAMATH_CALUDE_variance_of_X_l3501_350111

/-- A random variable X with two possible values -/
def X : Fin 2 → ℝ
  | 0 => 0
  | 1 => 1

/-- The probability mass function of X -/
def P : Fin 2 → ℝ
  | 0 => 0.4
  | 1 => 0.6

/-- The expected value of X -/
def E : ℝ := X 0 * P 0 + X 1 * P 1

/-- The variance of X -/
def D : ℝ := (X 0 - E)^2 * P 0 + (X 1 - E)^2 * P 1

/-- Theorem: The variance of X is 0.24 -/
theorem variance_of_X : D = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_X_l3501_350111


namespace NUMINAMATH_CALUDE_total_weight_calculation_l3501_350146

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight : ℝ := 1184

/-- The number of moles of the compound -/
def number_of_moles : ℝ := 4

/-- The total weight of the compound in grams -/
def total_weight : ℝ := number_of_moles * molecular_weight

theorem total_weight_calculation : total_weight = 4736 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_calculation_l3501_350146


namespace NUMINAMATH_CALUDE_no_positive_abc_with_all_roots_l3501_350127

theorem no_positive_abc_with_all_roots : ¬ ∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (b^2 - 4*a*c ≥ 0) ∧ 
  (c^2 - 4*b*a ≥ 0) ∧ 
  (a^2 - 4*b*c ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_abc_with_all_roots_l3501_350127


namespace NUMINAMATH_CALUDE_max_sum_product_l3501_350138

theorem max_sum_product (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 200 → 
  a ≥ 50 → 
  a * b + b * c + c * d + d * a ≤ 5000 := by
sorry

end NUMINAMATH_CALUDE_max_sum_product_l3501_350138


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_ratio_l3501_350119

/-- For an isosceles right triangle with legs of length a, 
    the ratio of twice a leg to the hypotenuse is √2 -/
theorem isosceles_right_triangle_ratio (a : ℝ) (h : a > 0) : 
  (2 * a) / Real.sqrt (a^2 + a^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_ratio_l3501_350119


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l3501_350187

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (p : ℝ → ℝ),
    (∀ x, p x = (14 * x^2 + 4 * x + 12) / 15) ∧
    p (-2) = 4 ∧
    p 1 = 2 ∧
    p 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l3501_350187


namespace NUMINAMATH_CALUDE_triangle_area_double_l3501_350151

theorem triangle_area_double (halved_area : ℝ) :
  halved_area = 7 → 2 * halved_area = 14 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_double_l3501_350151


namespace NUMINAMATH_CALUDE_alloy_mixture_theorem_l3501_350112

/-- The amount of the first alloy used to create the third alloy -/
def first_alloy_amount : ℝ := 15

/-- The percentage of chromium in the first alloy -/
def first_alloy_chromium_percent : ℝ := 0.10

/-- The percentage of chromium in the second alloy -/
def second_alloy_chromium_percent : ℝ := 0.06

/-- The amount of the second alloy used to create the third alloy -/
def second_alloy_amount : ℝ := 35

/-- The percentage of chromium in the resulting third alloy -/
def third_alloy_chromium_percent : ℝ := 0.072

theorem alloy_mixture_theorem :
  first_alloy_amount * first_alloy_chromium_percent +
  second_alloy_amount * second_alloy_chromium_percent =
  (first_alloy_amount + second_alloy_amount) * third_alloy_chromium_percent :=
by sorry

end NUMINAMATH_CALUDE_alloy_mixture_theorem_l3501_350112


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3501_350147

theorem decimal_to_fraction : 
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ (0.4 + (3 : ℚ) / 99) = (n : ℚ) / d := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3501_350147


namespace NUMINAMATH_CALUDE_roadway_deck_concrete_amount_l3501_350175

/-- The amount of concrete needed for the roadway deck of a bridge -/
def roadway_deck_concrete (total_concrete : ℕ) (anchor_concrete : ℕ) (pillar_concrete : ℕ) : ℕ :=
  total_concrete - (2 * anchor_concrete + pillar_concrete)

/-- Theorem stating that the roadway deck needs 1600 tons of concrete -/
theorem roadway_deck_concrete_amount :
  roadway_deck_concrete 4800 700 1800 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_roadway_deck_concrete_amount_l3501_350175


namespace NUMINAMATH_CALUDE_cut_cylinder_volume_l3501_350172

/-- Represents a right cylinder with a vertical planar cut -/
structure CutCylinder where
  height : ℝ
  baseRadius : ℝ
  cutArea : ℝ

/-- The volume of the larger piece of a cut cylinder -/
def largerPieceVolume (c : CutCylinder) : ℝ := sorry

theorem cut_cylinder_volume 
  (c : CutCylinder) 
  (h_height : c.height = 20)
  (h_radius : c.baseRadius = 5)
  (h_cut_area : c.cutArea = 100 * Real.sqrt 2) :
  largerPieceVolume c = 250 + 375 * Real.pi := by sorry

end NUMINAMATH_CALUDE_cut_cylinder_volume_l3501_350172


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3501_350134

/-- A quadratic function with leading coefficient a -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The solution set of f(x) > -2x is (1,3) -/
def solution_set (a b c : ℝ) : Prop :=
  ∀ x, (1 < x ∧ x < 3) ↔ f a b c x > -2 * x

/-- The equation f(x) + 6a = 0 has two equal real roots -/
def equal_roots (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x, f a b c x + 6 * a = 0 ↔ x = r

theorem quadratic_function_theorem (a b c : ℝ) 
  (h1 : solution_set a b c)
  (h2 : equal_roots a b c)
  (h3 : a < 0) :
  ∀ x, f a b c x = -x^2 - x - 3/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3501_350134


namespace NUMINAMATH_CALUDE_factor_implies_b_value_l3501_350105

/-- Q is a polynomial of degree 4 with a parameter b -/
def Q (b : ℝ) (x : ℝ) : ℝ := x^4 - 3*x^3 + b*x^2 - 12*x + 24

/-- Theorem: If x+2 is a factor of Q, then b = -22 -/
theorem factor_implies_b_value (b : ℝ) : 
  (∀ x, Q b x = 0 → x + 2 = 0) → b = -22 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l3501_350105


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3501_350142

theorem smallest_solution_of_equation : 
  ∃ x : ℝ, x = -15 ∧ 
  (∀ y : ℝ, 3 * y^2 + 39 * y - 75 = y * (y + 16) → y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3501_350142


namespace NUMINAMATH_CALUDE_secret_code_count_l3501_350152

/-- The number of colors available -/
def num_colors : ℕ := 7

/-- The number of slots to fill -/
def num_slots : ℕ := 5

/-- The number of possible secret codes -/
def num_codes : ℕ := 2520

/-- Theorem: The number of ways to arrange 5 colors chosen from 7 distinct colors is 2520 -/
theorem secret_code_count :
  (Finset.card (Finset.range num_colors)).factorial / 
  (Finset.card (Finset.range (num_colors - num_slots))).factorial = num_codes :=
by sorry

end NUMINAMATH_CALUDE_secret_code_count_l3501_350152


namespace NUMINAMATH_CALUDE_billboard_area_l3501_350136

/-- The area of a rectangular billboard with perimeter 44 feet and width 9 feet is 117 square feet. -/
theorem billboard_area (perimeter width : ℝ) (h1 : perimeter = 44) (h2 : width = 9) :
  let length := (perimeter - 2 * width) / 2
  width * length = 117 :=
by sorry

end NUMINAMATH_CALUDE_billboard_area_l3501_350136
