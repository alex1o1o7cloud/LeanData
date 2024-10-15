import Mathlib

namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l480_48083

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 * 1 ≤ 45 * n := by sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l480_48083


namespace NUMINAMATH_CALUDE_existence_of_close_ratios_l480_48021

theorem existence_of_close_ratios (S : Finset ℝ) (h : S.card = 2000) :
  ∃ (a b c d : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
  a > b ∧ c > d ∧ (a ≠ c ∨ b ≠ d) ∧
  |((a - b) / (c - d)) - 1| < (1 : ℝ) / 100000 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_close_ratios_l480_48021


namespace NUMINAMATH_CALUDE_pancake_covers_center_l480_48072

-- Define the pan
def Pan : Type := Unit

-- Define the area of the pan
def pan_area : ℝ := 1

-- Define the pancake
def Pancake : Type := Unit

-- Define the property of the pancake being convex
def is_convex (p : Pancake) : Prop := sorry

-- Define the area of the pancake
def pancake_area (p : Pancake) : ℝ := sorry

-- Define the center of the pan
def pan_center (pan : Pan) : Set ℝ := sorry

-- Define the region covered by the pancake
def pancake_region (p : Pancake) : Set ℝ := sorry

-- The theorem to be proved
theorem pancake_covers_center (pan : Pan) (p : Pancake) :
  is_convex p →
  pancake_area p > 1/2 →
  pan_center pan ⊆ pancake_region p :=
sorry

end NUMINAMATH_CALUDE_pancake_covers_center_l480_48072


namespace NUMINAMATH_CALUDE_post_office_mailing_l480_48022

def total_cost : ℚ := 449/100
def letter_cost : ℚ := 37/100
def package_cost : ℚ := 88/100
def num_letters : ℕ := 5

theorem post_office_mailing :
  ∃ (num_packages : ℕ),
    letter_cost * num_letters + package_cost * num_packages = total_cost ∧
    num_letters - num_packages = 2 :=
by sorry

end NUMINAMATH_CALUDE_post_office_mailing_l480_48022


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l480_48013

def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℚ)
  (h_geometric : is_geometric_sequence a)
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = -54) :
  a 1 = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l480_48013


namespace NUMINAMATH_CALUDE_polynomial_expansion_l480_48015

theorem polynomial_expansion (z : R) [CommRing R] :
  (3 * z^2 + 4 * z - 7) * (4 * z^3 - 3 * z + 2) =
  12 * z^5 + 16 * z^4 - 37 * z^3 - 6 * z^2 + 29 * z - 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l480_48015


namespace NUMINAMATH_CALUDE_intersection_distance_l480_48038

/-- The distance between the intersection points of the line x - y + 1 = 0 and the circle x² + y² = 2 is equal to √6. -/
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ), 
    (A.1 - A.2 + 1 = 0) ∧ (A.1^2 + A.2^2 = 2) ∧
    (B.1 - B.2 + 1 = 0) ∧ (B.1^2 + B.2^2 = 2) ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l480_48038


namespace NUMINAMATH_CALUDE_max_badminton_rackets_l480_48097

theorem max_badminton_rackets 
  (table_tennis_price badminton_price : ℕ)
  (total_rackets : ℕ)
  (max_expenditure : ℕ)
  (h1 : 2 * table_tennis_price + badminton_price = 220)
  (h2 : 3 * table_tennis_price + 2 * badminton_price = 380)
  (h3 : total_rackets = 30)
  (h4 : ∀ m : ℕ, m ≤ total_rackets → 
        (total_rackets - m) * table_tennis_price + m * badminton_price ≤ max_expenditure) :
  ∃ max_badminton : ℕ, 
    max_badminton ≤ total_rackets ∧
    (total_rackets - max_badminton) * table_tennis_price + max_badminton * badminton_price ≤ max_expenditure ∧
    ∀ n : ℕ, n > max_badminton → 
      (total_rackets - n) * table_tennis_price + n * badminton_price > max_expenditure :=
by
  sorry

end NUMINAMATH_CALUDE_max_badminton_rackets_l480_48097


namespace NUMINAMATH_CALUDE_ian_money_left_l480_48029

/-- Calculates the amount of money Ian has left after paying off debts --/
def money_left (lottery_win : ℕ) (colin_payment : ℕ) : ℕ :=
  let helen_payment := 2 * colin_payment
  let benedict_payment := helen_payment / 2
  lottery_win - (colin_payment + helen_payment + benedict_payment)

/-- Theorem stating that Ian has $20 left after paying off debts --/
theorem ian_money_left : money_left 100 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ian_money_left_l480_48029


namespace NUMINAMATH_CALUDE_units_digit_of_first_four_composites_product_l480_48070

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def first_four_composites : List ℕ := [4, 6, 8, 9]

theorem units_digit_of_first_four_composites_product :
  (first_four_composites.prod % 10 = 8) ∧
  (∀ n ∈ first_four_composites, is_composite n) ∧
  (∀ m, is_composite m → m ≥ 4 → ∃ n ∈ first_four_composites, n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_first_four_composites_product_l480_48070


namespace NUMINAMATH_CALUDE_page_number_digit_difference_l480_48066

/-- Counts the occurrences of a digit in a range of numbers -/
def countDigit (d : Nat) (start finish : Nat) : Nat :=
  sorry

/-- The difference between the count of 5's and 3's in page numbers from 1 to 512 -/
theorem page_number_digit_difference :
  let pages := 512
  let start_page := 1
  let end_page := pages
  let digit_five := 5
  let digit_three := 3
  (countDigit digit_five start_page end_page) - (countDigit digit_three start_page end_page) = 22 :=
sorry

end NUMINAMATH_CALUDE_page_number_digit_difference_l480_48066


namespace NUMINAMATH_CALUDE_window_dimensions_l480_48093

/-- Represents the dimensions of a rectangular glass pane -/
structure PaneDimensions where
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a rectangular window -/
structure WindowDimensions where
  width : ℝ
  height : ℝ

/-- Calculates the dimensions of a window given the pane dimensions and border widths -/
def calculateWindowDimensions (pane : PaneDimensions) (topBorderWidth sideBorderWidth : ℝ) : WindowDimensions :=
  { width := 3 * pane.width + 4 * sideBorderWidth,
    height := 2 * pane.height + 2 * topBorderWidth + sideBorderWidth }

theorem window_dimensions (y : ℝ) :
  let pane : PaneDimensions := { width := 4 * y, height := 3 * y }
  let window := calculateWindowDimensions pane 3 1
  window.width = 12 * y + 4 ∧ window.height = 6 * y + 7 := by
  sorry

#check window_dimensions

end NUMINAMATH_CALUDE_window_dimensions_l480_48093


namespace NUMINAMATH_CALUDE_complex_equation_sum_l480_48007

theorem complex_equation_sum (a b : ℝ) : 
  (a - 2 * Complex.I) * Complex.I = b - Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l480_48007


namespace NUMINAMATH_CALUDE_cafe_combinations_l480_48031

/-- The number of drinks on the menu -/
def menu_size : ℕ := 8

/-- Whether Yann orders coffee -/
def yann_orders_coffee : Bool := sorry

/-- The number of options available to Camille -/
def camille_options : ℕ :=
  if yann_orders_coffee then menu_size - 1 else menu_size

/-- The number of combinations when Yann orders coffee -/
def coffee_combinations : ℕ := 1 * (menu_size - 1)

/-- The number of combinations when Yann doesn't order coffee -/
def non_coffee_combinations : ℕ := (menu_size - 1) * menu_size

/-- The total number of different combinations of drinks Yann and Camille can order -/
def total_combinations : ℕ := coffee_combinations + non_coffee_combinations

theorem cafe_combinations : total_combinations = 63 := by sorry

end NUMINAMATH_CALUDE_cafe_combinations_l480_48031


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l480_48004

/-- Given a geometric sequence {a_n} where a_{2013} + a_{2015} = π, 
    prove that a_{2014}(a_{2012} + 2a_{2014} + a_{2016}) = π^2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h1 : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) 
    (h2 : a 2013 + a 2015 = π) : 
  a 2014 * (a 2012 + 2 * a 2014 + a 2016) = π^2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l480_48004


namespace NUMINAMATH_CALUDE_add_three_people_to_two_rows_l480_48054

/-- The number of ways to add three people to two rows of people -/
def add_people_ways (front_row : ℕ) (back_row : ℕ) (people_to_add : ℕ) : ℕ :=
  (people_to_add) * (front_row + 1) * (back_row + 1) * (back_row + 2)

/-- Theorem: The number of ways to add three people to two rows with 3 in front and 4 in back is 360 -/
theorem add_three_people_to_two_rows :
  add_people_ways 3 4 3 = 360 := by
  sorry

end NUMINAMATH_CALUDE_add_three_people_to_two_rows_l480_48054


namespace NUMINAMATH_CALUDE_white_square_area_l480_48048

theorem white_square_area (cube_edge : ℝ) (green_paint_area : ℝ) (white_square_area : ℝ) : 
  cube_edge = 12 →
  green_paint_area = 432 →
  white_square_area = (cube_edge ^ 2) - (green_paint_area / 6) →
  white_square_area = 72 := by
sorry

end NUMINAMATH_CALUDE_white_square_area_l480_48048


namespace NUMINAMATH_CALUDE_distance_between_locations_l480_48079

theorem distance_between_locations (s : ℝ) : 
  (s > 0) →  -- Distance is positive
  ((s/2 + 12) / (s/2 - 12) = s / (s - 40)) →  -- Condition when cars meet
  (s = 120) :=  -- Distance to prove
by sorry

end NUMINAMATH_CALUDE_distance_between_locations_l480_48079


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l480_48010

theorem trigonometric_expression_equality : 
  (Real.sin (7 * π / 180) + Real.sin (8 * π / 180) * Real.cos (15 * π / 180)) / 
  (Real.cos (7 * π / 180) - Real.sin (8 * π / 180) * Real.sin (15 * π / 180)) = 
  2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l480_48010


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l480_48099

theorem sqrt_difference_equality : 
  Real.sqrt (121 + 81) - Real.sqrt (49 - 36) = Real.sqrt 202 - Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l480_48099


namespace NUMINAMATH_CALUDE_five_digit_diff_last_two_count_l480_48009

/-- The number of five-digit numbers -/
def total_five_digit_numbers : ℕ := 90000

/-- The number of five-digit numbers where the last two digits are the same -/
def five_digit_same_last_two : ℕ := 9000

/-- The number of five-digit numbers where at least the last two digits are different -/
def five_digit_diff_last_two : ℕ := total_five_digit_numbers - five_digit_same_last_two

theorem five_digit_diff_last_two_count : five_digit_diff_last_two = 81000 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_diff_last_two_count_l480_48009


namespace NUMINAMATH_CALUDE_points_on_opposite_sides_l480_48014

def plane_equation (x y z : ℝ) : ℝ := x + 2*y + 3*z

def point1 : ℝ × ℝ × ℝ := (1, 2, -2)
def point2 : ℝ × ℝ × ℝ := (2, 1, -1)

theorem points_on_opposite_sides :
  (plane_equation point1.1 point1.2.1 point1.2.2) * (plane_equation point2.1 point2.2.1 point2.2.2) < 0 := by
  sorry

end NUMINAMATH_CALUDE_points_on_opposite_sides_l480_48014


namespace NUMINAMATH_CALUDE_difference_greater_than_one_l480_48006

theorem difference_greater_than_one : 19^91 - (999991:ℕ)^19 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_greater_than_one_l480_48006


namespace NUMINAMATH_CALUDE_cubic_sum_equals_nine_l480_48019

theorem cubic_sum_equals_nine (a b : ℝ) 
  (h1 : a^5 - a^4*b - a^4 + a - b - 1 = 0)
  (h2 : 2*a - 3*b = 1) : 
  a^3 + b^3 = 9 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_nine_l480_48019


namespace NUMINAMATH_CALUDE_mMobileCheaperByEleven_l480_48061

-- Define the cost structure for T-Mobile
def tMobileBaseCost : ℕ := 50
def tMobileAdditionalLineCost : ℕ := 16

-- Define the cost structure for M-Mobile
def mMobileBaseCost : ℕ := 45
def mMobileAdditionalLineCost : ℕ := 14

-- Define the number of lines needed
def totalLines : ℕ := 5

-- Define the function to calculate the total cost for a given plan
def calculateTotalCost (baseCost additionalLineCost : ℕ) : ℕ :=
  baseCost + (totalLines - 2) * additionalLineCost

-- Theorem statement
theorem mMobileCheaperByEleven :
  calculateTotalCost tMobileBaseCost tMobileAdditionalLineCost -
  calculateTotalCost mMobileBaseCost mMobileAdditionalLineCost = 11 := by
  sorry

end NUMINAMATH_CALUDE_mMobileCheaperByEleven_l480_48061


namespace NUMINAMATH_CALUDE_vasya_irrational_sequence_l480_48003

theorem vasya_irrational_sequence (r : ℚ) (hr : 0 < r) : 
  ∃ n : ℕ, ¬ (∃ q : ℚ, q = (λ x => Real.sqrt (x + 1))^[n] r) :=
sorry

end NUMINAMATH_CALUDE_vasya_irrational_sequence_l480_48003


namespace NUMINAMATH_CALUDE_three_team_leads_per_supervisor_l480_48055

/-- Represents the organizational structure of a company -/
structure Company where
  workers : ℕ
  team_leads : ℕ
  supervisors : ℕ
  worker_to_lead_ratio : ℕ

/-- Calculates the number of team leads per supervisor -/
def team_leads_per_supervisor (c : Company) : ℚ :=
  c.team_leads / c.supervisors

/-- Theorem: The number of team leads per supervisor is 3 -/
theorem three_team_leads_per_supervisor (c : Company) 
  (h1 : c.worker_to_lead_ratio = 10)
  (h2 : c.supervisors = 13)
  (h3 : c.workers = 390) :
  team_leads_per_supervisor c = 3 := by
  sorry

#check three_team_leads_per_supervisor

end NUMINAMATH_CALUDE_three_team_leads_per_supervisor_l480_48055


namespace NUMINAMATH_CALUDE_decimal_equivalent_of_one_fifth_squared_l480_48040

theorem decimal_equivalent_of_one_fifth_squared :
  (1 / 5 : ℚ) ^ 2 = (4 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_of_one_fifth_squared_l480_48040


namespace NUMINAMATH_CALUDE_carrot_to_green_bean_ratio_l480_48043

/-- Given a grocery bag with a maximum capacity and known weights of items,
    prove that the ratio of carrots to green beans is 1:2. -/
theorem carrot_to_green_bean_ratio
  (bag_capacity : ℕ)
  (green_beans : ℕ)
  (milk : ℕ)
  (remaining_capacity : ℕ)
  (h1 : bag_capacity = 20)
  (h2 : green_beans = 4)
  (h3 : milk = 6)
  (h4 : remaining_capacity = 2)
  (h5 : green_beans + milk + remaining_capacity = bag_capacity) :
  (bag_capacity - green_beans - milk) / green_beans = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_carrot_to_green_bean_ratio_l480_48043


namespace NUMINAMATH_CALUDE_corveus_weekly_lack_of_sleep_l480_48098

/-- Calculates the weekly lack of sleep given actual and recommended daily sleep hours -/
def weeklyLackOfSleep (actualSleep recommendedSleep : ℕ) : ℕ :=
  (recommendedSleep - actualSleep) * 7

/-- Proves that Corveus lacks 14 hours of sleep in a week -/
theorem corveus_weekly_lack_of_sleep :
  weeklyLackOfSleep 4 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_corveus_weekly_lack_of_sleep_l480_48098


namespace NUMINAMATH_CALUDE_max_profit_l480_48081

noncomputable section

def fixed_cost : ℝ := 2.5

def variable_cost (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10*x^2 + 100*x
  else if x ≥ 40 then 701*x + 10000/x - 9450
  else 0

def selling_price : ℝ := 0.7

def profit (x : ℝ) : ℝ :=
  selling_price * x - (fixed_cost + variable_cost x)

def production_quantity : ℝ := 100

theorem max_profit :
  profit production_quantity = 9000 ∧
  ∀ x > 0, profit x ≤ profit production_quantity :=
sorry

end

end NUMINAMATH_CALUDE_max_profit_l480_48081


namespace NUMINAMATH_CALUDE_line_passes_first_and_fourth_quadrants_l480_48095

/-- A line passes through the first quadrant if there exists a point (x, y) on the line where both x and y are positive. -/
def passes_through_first_quadrant (k b : ℝ) : Prop :=
  ∃ x > 0, k * x + b > 0

/-- A line passes through the fourth quadrant if there exists a point (x, y) on the line where x is positive and y is negative. -/
def passes_through_fourth_quadrant (k b : ℝ) : Prop :=
  ∃ x > 0, k * x + b < 0

/-- If bk < 0, then the line y = kx + b passes through both the first and fourth quadrants. -/
theorem line_passes_first_and_fourth_quadrants (k b : ℝ) (h : b * k < 0) :
  passes_through_first_quadrant k b ∧ passes_through_fourth_quadrant k b :=
sorry

end NUMINAMATH_CALUDE_line_passes_first_and_fourth_quadrants_l480_48095


namespace NUMINAMATH_CALUDE_three_digit_difference_divisible_by_nine_l480_48023

theorem three_digit_difference_divisible_by_nine :
  ∀ (a b c : ℕ), 
  0 ≤ a ∧ a ≤ 9 →
  0 ≤ b ∧ b ≤ 9 →
  0 ≤ c ∧ c ≤ 9 →
  ∃ (k : ℤ), (100 * a + 10 * b + c) - (a + b + c) = 9 * k :=
by sorry

end NUMINAMATH_CALUDE_three_digit_difference_divisible_by_nine_l480_48023


namespace NUMINAMATH_CALUDE_factorial_divisibility_theorem_l480_48039

def factorial (n : ℕ) : ℕ := Nat.factorial n

def sum_factorials (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => factorial (i + 1))

theorem factorial_divisibility_theorem :
  ∀ n : ℕ, n > 2 → ¬(factorial (n + 1) ∣ sum_factorials n) ∧
  (factorial 2 ∣ sum_factorials 1) ∧
  (factorial 3 ∣ sum_factorials 2) ∧
  ∀ m : ℕ, m ≠ 1 ∧ m ≠ 2 → ¬(factorial (m + 1) ∣ sum_factorials m) :=
by sorry

end NUMINAMATH_CALUDE_factorial_divisibility_theorem_l480_48039


namespace NUMINAMATH_CALUDE_cupcake_dozens_correct_l480_48067

/-- The number of dozens of cupcakes Jose needs to make -/
def cupcake_dozens : ℕ := 3

/-- The number of tablespoons of lemon juice needed for one dozen cupcakes -/
def juice_per_dozen : ℕ := 12

/-- The number of tablespoons of lemon juice provided by one lemon -/
def juice_per_lemon : ℕ := 4

/-- The number of lemons Jose needs -/
def lemons_needed : ℕ := 9

/-- Theorem stating that the number of dozens of cupcakes Jose needs to make is correct -/
theorem cupcake_dozens_correct : 
  cupcake_dozens = (lemons_needed * juice_per_lemon) / juice_per_dozen :=
by sorry

end NUMINAMATH_CALUDE_cupcake_dozens_correct_l480_48067


namespace NUMINAMATH_CALUDE_arrangement_count_is_correct_l480_48044

/-- The number of ways to arrange 5 people in a row with one person between A and B -/
def arrangement_count : ℕ := 36

/-- The total number of people in the arrangement -/
def total_people : ℕ := 5

/-- The number of people between A and B -/
def people_between : ℕ := 1

theorem arrangement_count_is_correct :
  arrangement_count = 
    2 * (total_people - 2) * (Nat.factorial (total_people - 2)) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_correct_l480_48044


namespace NUMINAMATH_CALUDE_median_is_212_l480_48053

/-- Represents the list where each integer n from 1 to 300 appears n times -/
def special_list : List ℕ := sorry

/-- The sum of all elements in the special list -/
def total_elements : ℕ := (300 * (300 + 1)) / 2

/-- The position of the median in the special list -/
def median_position : ℕ × ℕ := (total_elements / 2, total_elements / 2 + 1)

/-- Theorem stating that the median of the special list is 212 -/
theorem median_is_212 : 
  ∃ (median : ℕ), median = 212 ∧ 
  (∃ (l1 l2 : List ℕ), special_list = l1 ++ [median] ++ [median] ++ l2 ∧ 
   l1.length = median_position.1 - 1 ∧
   l2.length = special_list.length - median_position.2) :=
sorry

end NUMINAMATH_CALUDE_median_is_212_l480_48053


namespace NUMINAMATH_CALUDE_urn_probability_l480_48018

theorem urn_probability (N : ℝ) : N = 21 →
  (3 / 8) * (9 / (9 + N)) + (5 / 8) * (N / (9 + N)) = 0.55 := by
  sorry

#check urn_probability

end NUMINAMATH_CALUDE_urn_probability_l480_48018


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l480_48012

theorem complex_fraction_equality : Complex.I * 4 / (Real.sqrt 3 + Complex.I) = 1 + Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l480_48012


namespace NUMINAMATH_CALUDE_special_triangle_properties_l480_48035

/-- A triangle with an inscribed circle of radius 2, where one side is divided into segments of 4 and 6 by the point of tangency -/
structure SpecialTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  a : ℝ
  /-- The length of the second segment of the divided side -/
  b : ℝ
  /-- The radius is 2 -/
  h_r : r = 2
  /-- The first segment is 4 -/
  h_a : a = 4
  /-- The second segment is 6 -/
  h_b : b = 6

/-- The area of the triangle -/
def area (t : SpecialTriangle) : ℝ := 24

/-- The triangle is right-angled -/
def is_right_triangle (t : SpecialTriangle) : Prop :=
  ∃ (x y z : ℝ), x^2 + y^2 = z^2 ∧ 
    ((x = t.a + t.b ∧ y = 2 * t.r ∧ z = t.a + t.b + 2 * t.r) ∨
     (x = t.a + t.b ∧ y = t.a + t.b + 2 * t.r ∧ z = 2 * t.r) ∨
     (x = 2 * t.r ∧ y = t.a + t.b + 2 * t.r ∧ z = t.a + t.b))

theorem special_triangle_properties (t : SpecialTriangle) :
  is_right_triangle t ∧ area t = 24 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l480_48035


namespace NUMINAMATH_CALUDE_smallest_with_18_divisors_l480_48000

/-- Number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- n has exactly 18 positive integer divisors -/
def has_18_divisors (n : ℕ) : Prop := num_divisors n = 18

theorem smallest_with_18_divisors :
  ∃ (n : ℕ), has_18_divisors n ∧ ∀ m : ℕ, has_18_divisors m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_with_18_divisors_l480_48000


namespace NUMINAMATH_CALUDE_quadratic_function_range_l480_48069

/-- Given a quadratic function y = -x^2 + 2ax + a + 1, if y > a + 1 for all x in (-1, a),
    then -1 < a ≤ -1/2 -/
theorem quadratic_function_range (a : ℝ) :
  (∀ x, -1 < x ∧ x < a → -x^2 + 2*a*x + a + 1 > a + 1) →
  -1 < a ∧ a ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l480_48069


namespace NUMINAMATH_CALUDE_different_course_choices_l480_48059

theorem different_course_choices (n : ℕ) (k : ℕ) : n = 4 → k = 2 →
  (Nat.choose n k)^2 - (Nat.choose n k) = 30 := by
  sorry

end NUMINAMATH_CALUDE_different_course_choices_l480_48059


namespace NUMINAMATH_CALUDE_rectangle_area_l480_48020

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) :
  square_area = 2500 →
  rectangle_breadth = 10 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area = 200 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l480_48020


namespace NUMINAMATH_CALUDE_sine_function_omega_values_l480_48063

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f is symmetric about a point (a, b) if f(a + x) = f(a - x) for all x -/
def IsSymmetricAbout (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

/-- A function f is monotonic on an interval [a, b] if it is either increasing or decreasing on that interval -/
def IsMonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y)

theorem sine_function_omega_values 
  (ω φ : ℝ) 
  (f : ℝ → ℝ)
  (h1 : f = fun x ↦ Real.sin (ω * x + φ))
  (h2 : ω > 0)
  (h3 : 0 ≤ φ ∧ φ ≤ π)
  (h4 : IsEven f)
  (h5 : IsSymmetricAbout f (3 * π / 4) 0)
  (h6 : IsMonotonicOn f 0 (π / 2)) :
  ω = 2/3 ∨ ω = 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_omega_values_l480_48063


namespace NUMINAMATH_CALUDE_other_endpoint_coordinate_sum_l480_48082

/-- Given a line segment with midpoint (5, -8) and one endpoint at (9, -6),
    the sum of the coordinates of the other endpoint is -9. -/
theorem other_endpoint_coordinate_sum :
  ∀ (x y : ℝ),
  (5 = (9 + x) / 2) →
  (-8 = (-6 + y) / 2) →
  x + y = -9 :=
by sorry

end NUMINAMATH_CALUDE_other_endpoint_coordinate_sum_l480_48082


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l480_48024

theorem complex_number_quadrant (z : ℂ) (m : ℝ) :
  z * Complex.I = Complex.I + m →
  z.im = 1 →
  z.re > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l480_48024


namespace NUMINAMATH_CALUDE_dispatch_three_male_two_female_dispatch_at_least_two_male_l480_48032

def male_drivers : ℕ := 5
def female_drivers : ℕ := 4
def team_size : ℕ := 5

/-- The number of ways to choose 3 male drivers and 2 female drivers -/
theorem dispatch_three_male_two_female : 
  (Nat.choose male_drivers 3) * (Nat.choose female_drivers 2) = 60 := by sorry

/-- The number of ways to dispatch with at least two male drivers -/
theorem dispatch_at_least_two_male : 
  (Nat.choose male_drivers 2) * (Nat.choose female_drivers 3) +
  (Nat.choose male_drivers 3) * (Nat.choose female_drivers 2) +
  (Nat.choose male_drivers 4) * (Nat.choose female_drivers 1) +
  (Nat.choose male_drivers 5) * (Nat.choose female_drivers 0) = 121 := by sorry

end NUMINAMATH_CALUDE_dispatch_three_male_two_female_dispatch_at_least_two_male_l480_48032


namespace NUMINAMATH_CALUDE_smaller_rectangle_perimeter_l480_48049

/-- Given a rectangle with dimensions a × b that is divided into a smaller rectangle 
    with dimensions c × b and two squares with side length c, 
    the perimeter of the smaller rectangle is 2(c + b). -/
theorem smaller_rectangle_perimeter 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : a = 3 * c) : 
  2 * (c + b) = 2 * c + 2 * b := by
sorry

end NUMINAMATH_CALUDE_smaller_rectangle_perimeter_l480_48049


namespace NUMINAMATH_CALUDE_complex_power_sum_l480_48076

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (Real.pi/4)) :
  z^8 + 1/z^8 = 2 := by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l480_48076


namespace NUMINAMATH_CALUDE_max_cities_in_network_l480_48002

/-- Represents a city in the airline network -/
structure City where
  id : Nat

/-- Represents the airline network -/
structure AirlineNetwork where
  cities : Finset City
  connections : City → Finset City

/-- The maximum number of direct connections a city can have -/
def maxDirectConnections : Nat := 3

/-- Defines a valid airline network based on the given conditions -/
def isValidNetwork (network : AirlineNetwork) : Prop :=
  ∀ c ∈ network.cities,
    (network.connections c).card ≤ maxDirectConnections ∧
    ∀ d ∈ network.cities, 
      c ≠ d → (d ∈ network.connections c ∨ 
               ∃ e ∈ network.cities, e ∈ network.connections c ∧ d ∈ network.connections e)

/-- The theorem stating the maximum number of cities in a valid network -/
theorem max_cities_in_network (network : AirlineNetwork) 
  (h : isValidNetwork network) : network.cities.card ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_max_cities_in_network_l480_48002


namespace NUMINAMATH_CALUDE_special_house_profit_calculation_l480_48033

def special_house_profit (extra_cost : ℝ) (price_multiplier : ℝ) (standard_house_price : ℝ) : ℝ :=
  price_multiplier * standard_house_price - standard_house_price - extra_cost

theorem special_house_profit_calculation :
  special_house_profit 100000 1.5 320000 = 60000 := by
  sorry

end NUMINAMATH_CALUDE_special_house_profit_calculation_l480_48033


namespace NUMINAMATH_CALUDE_scout_troop_profit_l480_48034

-- Define the problem parameters
def candy_bars : ℕ := 1500
def buy_price : ℚ := 1 / 3
def transport_cost : ℕ := 50
def sell_price : ℚ := 3 / 5

-- Define the net profit calculation
def net_profit : ℚ :=
  candy_bars * sell_price - (candy_bars * buy_price + transport_cost)

-- Theorem statement
theorem scout_troop_profit :
  net_profit = 350 := by
  sorry

end NUMINAMATH_CALUDE_scout_troop_profit_l480_48034


namespace NUMINAMATH_CALUDE_rectangle_subdivision_l480_48025

/-- A rectangle can be subdivided into n pairwise noncongruent rectangles similar to the original -/
def can_subdivide (n : ℕ) : Prop :=
  ∃ (r : ℝ) (h : r > 1), ∃ (rectangles : Fin n → ℝ × ℝ),
    (∀ i j, i ≠ j → rectangles i ≠ rectangles j) ∧
    (∀ i, (rectangles i).1 / (rectangles i).2 = r)

theorem rectangle_subdivision (n : ℕ) (h : n > 1) :
  can_subdivide n ↔ n ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_rectangle_subdivision_l480_48025


namespace NUMINAMATH_CALUDE_max_blue_points_max_blue_points_2016_l480_48001

/-- Given a set of spheres, some red and some green, with blue points at each red-green contact,
    the maximum number of blue points is achieved when there are equal numbers of red and green spheres. -/
theorem max_blue_points (total_spheres : ℕ) (h_total : total_spheres = 2016) :
  ∃ (red_spheres green_spheres : ℕ),
    red_spheres + green_spheres = total_spheres ∧
    red_spheres * green_spheres ≤ (total_spheres / 2) ^ 2 :=
by sorry

/-- The maximum number of blue points for 2016 spheres is 1008^2. -/
theorem max_blue_points_2016 :
  ∃ (red_spheres green_spheres : ℕ),
    red_spheres + green_spheres = 2016 ∧
    red_spheres * green_spheres = 1008 ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_blue_points_max_blue_points_2016_l480_48001


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l480_48045

/-- Given that g(x) = ax^2 + bx + c divides f(x) = x^3 + px^2 + qx + r, 
    where a ≠ 0, b ≠ 0, c ≠ 0, prove that (ap - b) / a = (aq - c) / b = ar / c -/
theorem polynomial_division_theorem 
  (a b c p q r : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_divides : ∃ k, x^3 + p*x^2 + q*x + r = (a*x^2 + b*x + c) * k) : 
  (a*p - b) / a = (a*q - c) / b ∧ (a*q - c) / b = a*r / c := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l480_48045


namespace NUMINAMATH_CALUDE_octagon_diagonals_l480_48075

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l480_48075


namespace NUMINAMATH_CALUDE_distance_between_points_l480_48058

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-2, 4)
  let p2 : ℝ × ℝ := (3, -8)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l480_48058


namespace NUMINAMATH_CALUDE_inequality_solution_and_abc_inequality_l480_48090

theorem inequality_solution_and_abc_inequality :
  let solution_set := {x : ℝ | -1/2 < x ∧ x < 7/2}
  let p : ℝ := -3
  let q : ℝ := -7/4
  (∀ x, x ∈ solution_set ↔ |2*x - 3| < 4) →
  (∀ x, x ∈ solution_set ↔ x^2 + p*x + q < 0) →
  ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c →
    a + b + c = 2*p - 4*q →
    Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_abc_inequality_l480_48090


namespace NUMINAMATH_CALUDE_chord_equation_l480_48047

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Main theorem -/
theorem chord_equation (e : Ellipse) (p : Point) (l : Line) : 
  e.a^2 = 8 ∧ e.b^2 = 4 ∧ p.x = 2 ∧ p.y = -1 ∧ 
  (∃ p1 p2 : Point, 
    pointOnEllipse p1 e ∧ 
    pointOnEllipse p2 e ∧
    p.x = (p1.x + p2.x) / 2 ∧ 
    p.y = (p1.y + p2.y) / 2 ∧
    pointOnLine p1 l ∧ 
    pointOnLine p2 l) →
  l.slope = 1 ∧ l.intercept = -3 := by
  sorry

end NUMINAMATH_CALUDE_chord_equation_l480_48047


namespace NUMINAMATH_CALUDE_supply_lasts_18_months_l480_48037

/-- Represents the number of pills in a supply -/
def supply : ℕ := 60

/-- Represents the fraction of a pill taken per dose -/
def dose : ℚ := 1 / 3

/-- Represents the number of days between doses -/
def days_between_doses : ℕ := 3

/-- Represents the average number of days in a month -/
def days_per_month : ℕ := 30

/-- Calculates the number of months a supply of medicine will last -/
def months_supply_lasts : ℚ :=
  (supply : ℚ) * (days_between_doses : ℚ) / dose / days_per_month

theorem supply_lasts_18_months :
  months_supply_lasts = 18 := by sorry

end NUMINAMATH_CALUDE_supply_lasts_18_months_l480_48037


namespace NUMINAMATH_CALUDE_parabola_vertex_l480_48017

/-- The vertex of a parabola defined by y = a(x+1)^2 - 2 is at (-1, -2) --/
theorem parabola_vertex (a : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * (x + 1)^2 - 2
  ∃! p : ℝ × ℝ, p.1 = -1 ∧ p.2 = -2 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l480_48017


namespace NUMINAMATH_CALUDE_wendy_albums_l480_48050

theorem wendy_albums (total_pictures : ℕ) (first_album : ℕ) (pictures_per_album : ℕ) 
  (h1 : total_pictures = 79)
  (h2 : first_album = 44)
  (h3 : pictures_per_album = 7) :
  (total_pictures - first_album) / pictures_per_album = 5 :=
by sorry

end NUMINAMATH_CALUDE_wendy_albums_l480_48050


namespace NUMINAMATH_CALUDE_square_difference_l480_48084

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l480_48084


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l480_48030

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (c d : ℝ), x^2 + 12*x + 4 = 0 ↔ (x + c)^2 = d ∧ d = 32 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l480_48030


namespace NUMINAMATH_CALUDE_sum_reciprocal_products_equals_three_eighths_l480_48046

theorem sum_reciprocal_products_equals_three_eighths :
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) +
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_products_equals_three_eighths_l480_48046


namespace NUMINAMATH_CALUDE_expression_evaluation_l480_48068

theorem expression_evaluation :
  let x : ℚ := 7/6
  (x - 1) / x / (x - (2*x - 1) / x) = 6 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l480_48068


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l480_48026

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = (1 : ℝ) / 4)
  (h_S : S = 80)
  (h_sum : S = a / (1 - r))
  : a = 60 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l480_48026


namespace NUMINAMATH_CALUDE_complement_M_union_N_when_a_is_2_M_union_N_equals_M_iff_a_in_range_l480_48078

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x ≤ 10}
def N (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for the first part of the problem
theorem complement_M_union_N_when_a_is_2 :
  (Set.univ \ M) ∪ N 2 = {x | x > 5 ∨ x < -2 ∨ (1 ≤ x ∧ x ≤ 5)} := by sorry

-- Theorem for the second part of the problem
theorem M_union_N_equals_M_iff_a_in_range (a : ℝ) :
  M ∪ N a = M ↔ a < -1 ∨ (-1 ≤ a ∧ a ≤ 2) := by sorry

end NUMINAMATH_CALUDE_complement_M_union_N_when_a_is_2_M_union_N_equals_M_iff_a_in_range_l480_48078


namespace NUMINAMATH_CALUDE_edwards_earnings_l480_48042

/-- Edward's lawn mowing earnings problem -/
theorem edwards_earnings (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ) :
  rate = 4 →
  total_lawns = 17 →
  forgotten_lawns = 9 →
  rate * (total_lawns - forgotten_lawns) = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_edwards_earnings_l480_48042


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l480_48091

/-- A complex number is pure imaginary if its real part is zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem a_equals_one_sufficient_not_necessary :
  ∃ (a : ℝ),
    (a = 1 → is_pure_imaginary ((a - 1) * (a + 2) + (a + 3) * Complex.I)) ∧
    (∃ (b : ℝ), b ≠ 1 ∧ is_pure_imaginary ((b - 1) * (b + 2) + (b + 3) * Complex.I)) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l480_48091


namespace NUMINAMATH_CALUDE_no_such_function_exists_l480_48052

theorem no_such_function_exists :
  ¬ ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, (f^[n.val] n = n + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l480_48052


namespace NUMINAMATH_CALUDE_inequality_solution_set_l480_48064

theorem inequality_solution_set : 
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l480_48064


namespace NUMINAMATH_CALUDE_original_balance_l480_48062

/-- Proves that if a balance incurs a 2% finance charge and results in a total of $153, then the original balance was $150. -/
theorem original_balance (total : ℝ) (finance_charge_rate : ℝ) (h1 : finance_charge_rate = 0.02) (h2 : total = 153) :
  ∃ (original : ℝ), original * (1 + finance_charge_rate) = total ∧ original = 150 :=
by sorry

end NUMINAMATH_CALUDE_original_balance_l480_48062


namespace NUMINAMATH_CALUDE_greatest_whole_number_inequality_l480_48074

theorem greatest_whole_number_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 3 * x + 2 < 5 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_inequality_l480_48074


namespace NUMINAMATH_CALUDE_slope_angle_range_l480_48071

/-- The range of slope angles for a line passing through (2,1) and (1,m²) where m ∈ ℝ -/
theorem slope_angle_range :
  ∀ m : ℝ, ∃ θ : ℝ, 
    (θ ∈ Set.Icc 0 (π/2) ∪ Set.Ioo (π/2) π) ∧ 
    (θ = Real.arctan ((m^2 - 1) / (2 - 1)) ∨ θ = Real.arctan ((m^2 - 1) / (2 - 1)) + π) :=
sorry

end NUMINAMATH_CALUDE_slope_angle_range_l480_48071


namespace NUMINAMATH_CALUDE_common_roots_product_l480_48080

/-- Given two cubic equations with two common roots, prove their product is 8 -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (p q r s : ℝ), 
    (p^3 + C*p + 20 = 0) ∧ 
    (q^3 + C*q + 20 = 0) ∧ 
    (r^3 + C*r + 20 = 0) ∧
    (p^3 + D*p^2 + 80 = 0) ∧ 
    (q^3 + D*q^2 + 80 = 0) ∧ 
    (s^3 + D*s^2 + 80 = 0) ∧
    (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧
    (p ≠ s) ∧ (q ≠ s) →
    p * q = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_roots_product_l480_48080


namespace NUMINAMATH_CALUDE_minimum_pages_per_day_l480_48094

theorem minimum_pages_per_day (total_pages : ℕ) (days : ℕ) (pages_per_day : ℕ) : 
  total_pages = 220 → days = 7 → 
  (pages_per_day * days ≥ total_pages ∧ 
   ∀ n : ℕ, n * days ≥ total_pages → n ≥ pages_per_day) →
  pages_per_day = 32 := by
sorry

end NUMINAMATH_CALUDE_minimum_pages_per_day_l480_48094


namespace NUMINAMATH_CALUDE_remaining_digits_count_l480_48087

theorem remaining_digits_count (total : ℕ) (avg_total : ℚ) (subset : ℕ) (avg_subset : ℚ) (avg_remaining : ℚ)
  (h1 : total = 10)
  (h2 : avg_total = 80)
  (h3 : subset = 6)
  (h4 : avg_subset = 58)
  (h5 : avg_remaining = 113) :
  total - subset = 4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_digits_count_l480_48087


namespace NUMINAMATH_CALUDE_triangle_angles_l480_48096

theorem triangle_angles (a b c : ℝ) (h1 : a = 2) (h2 : b = 2) (h3 : c = Real.sqrt 6 - Real.sqrt 2) :
  ∃ (α β γ : ℝ),
    α = 30 * π / 180 ∧
    β = 75 * π / 180 ∧
    γ = 75 * π / 180 ∧
    (Real.cos α = (a^2 + b^2 - c^2) / (2 * a * b)) ∧
    (Real.cos β = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
    (Real.cos γ = (b^2 + c^2 - a^2) / (2 * b * c)) ∧
    α + β + γ = π := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_l480_48096


namespace NUMINAMATH_CALUDE_investment_rate_proof_l480_48077

-- Define the given values
def total_investment : ℚ := 12000
def first_investment : ℚ := 5000
def first_rate : ℚ := 3 / 100
def second_investment : ℚ := 3000
def second_return : ℚ := 90
def desired_income : ℚ := 480

-- Define the theorem
theorem investment_rate_proof :
  let remaining_investment := total_investment - first_investment - second_investment
  let known_income := first_investment * first_rate + second_return
  let required_income := desired_income - known_income
  let rate := required_income / remaining_investment
  rate = 6 / 100 := by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l480_48077


namespace NUMINAMATH_CALUDE_complex_equation_solution_l480_48073

/-- Given the complex equation (1+2i)a + b = 2i, where a and b are real numbers, prove that a = 1 and b = -1. -/
theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I : ℂ) * 2 + 1 * (a : ℂ) + (b : ℂ) = (Complex.I : ℂ) * 2 → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l480_48073


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l480_48088

theorem existence_of_special_sequence :
  ∃ (a : Fin 2013 → ℕ), 
    (∀ i j : Fin 2013, i ≠ j → a i ≠ a j) ∧ 
    (∀ k m : Fin 2013, k < m → (a m + a k) % (a m - a k) = 0) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l480_48088


namespace NUMINAMATH_CALUDE_measure_one_kg_cereal_l480_48041

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | Balanced
  | Unbalanced

/-- Represents a weighing operation -/
def weighing (left right : ℕ) : WeighingResult :=
  if left = right then WeighingResult.Balanced else WeighingResult.Unbalanced

/-- Represents the process of measuring cereal -/
def measureCereal (totalCereal weight : ℕ) (maxWeighings : ℕ) : Prop :=
  ∃ (firstLeft firstRight secondLeft secondRight : ℕ),
    firstLeft + firstRight = totalCereal ∧
    secondLeft + secondRight ≤ firstLeft ∧
    weighing (firstLeft - secondLeft) (firstRight + weight) = WeighingResult.Balanced ∧
    weighing secondLeft weight = WeighingResult.Balanced ∧
    secondRight = 1 ∧
    2 ≤ maxWeighings

/-- Theorem stating that it's possible to measure 1 kg of cereal from 11 kg using a 3 kg weight in two weighings -/
theorem measure_one_kg_cereal :
  measureCereal 11 3 2 := by sorry

end NUMINAMATH_CALUDE_measure_one_kg_cereal_l480_48041


namespace NUMINAMATH_CALUDE_slope_of_l₃_l480_48051

-- Define the lines and points
def l₁ (x y : ℝ) : Prop := 2 * x + 3 * y = 6
def l₂ (y : ℝ) : Prop := y = 2
def A : ℝ × ℝ := (0, -3)

-- Define the properties of the lines and points
axiom l₁_through_A : l₁ A.1 A.2
axiom l₂_meets_l₁ : ∃ B : ℝ × ℝ, l₁ B.1 B.2 ∧ l₂ B.2
axiom l₃_positive_slope : ∃ m : ℝ, m > 0 ∧ ∀ x y : ℝ, y - A.2 = m * (x - A.1)
axiom l₃_through_A : ∀ x y : ℝ, y - A.2 = (y - A.2) / (x - A.1) * (x - A.1)
axiom l₃_meets_l₂ : ∃ C : ℝ × ℝ, l₂ C.2 ∧ C.2 - A.2 = (C.2 - A.2) / (C.1 - A.1) * (C.1 - A.1)

-- Define the area of triangle ABC
axiom triangle_area : ∃ B C : ℝ × ℝ, 
  l₁ B.1 B.2 ∧ l₂ B.2 ∧ l₂ C.2 ∧ C.2 - A.2 = (C.2 - A.2) / (C.1 - A.1) * (C.1 - A.1) ∧
  1/2 * |(B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)| = 10

-- Theorem statement
theorem slope_of_l₃ : 
  ∃ m : ℝ, m = 5/4 ∧ ∀ x y : ℝ, y - A.2 = m * (x - A.1) :=
sorry

end NUMINAMATH_CALUDE_slope_of_l₃_l480_48051


namespace NUMINAMATH_CALUDE_group_dynamics_index_l480_48089

theorem group_dynamics_index (n : ℕ) (female_count : ℕ) : 
  n = 25 →
  female_count ≤ n →
  (n - female_count : ℚ) / n - (n - (n - female_count) : ℚ) / n = 9 / 25 →
  female_count = 8 := by
sorry

end NUMINAMATH_CALUDE_group_dynamics_index_l480_48089


namespace NUMINAMATH_CALUDE_ella_video_game_spending_l480_48028

/-- Proves that Ella spends 40% of her salary on video games given the conditions -/
theorem ella_video_game_spending (
  last_year_spending : ℝ)
  (new_salary : ℝ)
  (raise_percentage : ℝ)
  (h1 : last_year_spending = 100)
  (h2 : new_salary = 275)
  (h3 : raise_percentage = 0.1)
  : (last_year_spending / (new_salary / (1 + raise_percentage))) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ella_video_game_spending_l480_48028


namespace NUMINAMATH_CALUDE_students_who_like_basketball_l480_48086

/-- Given a class of students where some play basketball and/or cricket, 
    this theorem proves the number of students who like basketball. -/
theorem students_who_like_basketball 
  (cricket : ℕ)
  (both : ℕ)
  (basketball_or_cricket : ℕ)
  (h1 : cricket = 8)
  (h2 : both = 4)
  (h3 : basketball_or_cricket = 14) :
  basketball_or_cricket = cricket + (basketball_or_cricket - cricket - both) - both :=
by sorry

end NUMINAMATH_CALUDE_students_who_like_basketball_l480_48086


namespace NUMINAMATH_CALUDE_clark_discount_clark_discount_proof_l480_48005

/-- Calculates the discount given to Clark for purchasing auto parts -/
theorem clark_discount (original_price : ℕ) (quantity : ℕ) (total_paid : ℕ) : ℕ :=
  let total_without_discount := original_price * quantity
  let discount := total_without_discount - total_paid
  discount

/-- Proves that Clark's discount is $121 given the problem conditions -/
theorem clark_discount_proof :
  clark_discount 80 7 439 = 121 := by
  sorry

end NUMINAMATH_CALUDE_clark_discount_clark_discount_proof_l480_48005


namespace NUMINAMATH_CALUDE_halloween_houses_per_hour_l480_48056

theorem halloween_houses_per_hour 
  (num_children : ℕ) 
  (num_hours : ℕ) 
  (treats_per_child_per_house : ℕ) 
  (total_treats : ℕ) 
  (h1 : num_children = 3)
  (h2 : num_hours = 4)
  (h3 : treats_per_child_per_house = 3)
  (h4 : total_treats = 180) :
  total_treats / (num_children * num_hours * treats_per_child_per_house) = 5 := by
  sorry

end NUMINAMATH_CALUDE_halloween_houses_per_hour_l480_48056


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l480_48008

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 6*x + 5 = 2*x - 8) → 
  (∃ x₁ x₂ : ℝ, (x₁^2 - 6*x₁ + 5 = 2*x₁ - 8) ∧ 
                (x₂^2 - 6*x₂ + 5 = 2*x₂ - 8) ∧ 
                (x₁ + x₂ = 8)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l480_48008


namespace NUMINAMATH_CALUDE_product_of_sum_of_four_squares_l480_48011

theorem product_of_sum_of_four_squares (a b : ℤ)
  (ha : ∃ x₁ x₂ x₃ x₄ : ℤ, a = x₁^2 + x₂^2 + x₃^2 + x₄^2)
  (hb : ∃ y₁ y₂ y₃ y₄ : ℤ, b = y₁^2 + y₂^2 + y₃^2 + y₄^2) :
  ∃ z₁ z₂ z₃ z₄ : ℤ, a * b = z₁^2 + z₂^2 + z₃^2 + z₄^2 :=
by sorry

end NUMINAMATH_CALUDE_product_of_sum_of_four_squares_l480_48011


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l480_48092

/-- For an arithmetic sequence {a_n} with S_n as the sum of its first n terms, 
    if S_9 = 27, then a_4 + a_6 = 6 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_S9 : S 9 = 27) : 
  a 4 + a 6 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l480_48092


namespace NUMINAMATH_CALUDE_f_inequality_and_abs_inequality_l480_48065

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

-- Define the set M
def M : Set ℝ := {x | 1 < x ∧ x < 4}

-- Theorem statement
theorem f_inequality_and_abs_inequality :
  (∀ x, f x < 3 ↔ x ∈ M) ∧
  (∀ a b, a ∈ M → b ∈ M → |a + b| < |1 + a * b|) := by sorry

end NUMINAMATH_CALUDE_f_inequality_and_abs_inequality_l480_48065


namespace NUMINAMATH_CALUDE_angle_trigonometry_l480_48027

theorem angle_trigonometry (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α * Real.tan β = 13/7) (h4 : Real.sin (α - β) = Real.sqrt 5 / 3) :
  Real.cos (α - β) = 2/3 ∧ Real.cos (α + β) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_trigonometry_l480_48027


namespace NUMINAMATH_CALUDE_printer_problem_l480_48057

/-- Calculates the time needed to print a given number of pages with breaks -/
def print_time (pages_per_minute : ℕ) (total_pages : ℕ) (pages_before_break : ℕ) (break_duration : ℕ) : ℕ :=
  let full_segments := total_pages / pages_before_break
  let remaining_pages := total_pages % pages_before_break
  let printing_time := (full_segments * pages_before_break + remaining_pages) / pages_per_minute
  let break_time := full_segments * break_duration
  printing_time + break_time

theorem printer_problem :
  print_time 25 350 150 5 = 24 := by
sorry

end NUMINAMATH_CALUDE_printer_problem_l480_48057


namespace NUMINAMATH_CALUDE_emily_beads_count_l480_48060

/-- The number of beads per necklace -/
def beads_per_necklace : ℕ := 5

/-- The number of necklaces Emily made -/
def necklaces_made : ℕ := 4

/-- The total number of beads Emily used -/
def total_beads : ℕ := beads_per_necklace * necklaces_made

theorem emily_beads_count : total_beads = 20 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l480_48060


namespace NUMINAMATH_CALUDE_sector_radius_proof_l480_48016

/-- The area of a circular sector -/
def sectorArea : ℝ := 51.54285714285714

/-- The central angle of the sector in degrees -/
def centralAngle : ℝ := 41

/-- The radius of the circle -/
def radius : ℝ := 12

/-- Theorem stating that the given sector area and central angle result in the specified radius -/
theorem sector_radius_proof : 
  abs (sectorArea - (centralAngle / 360) * Real.pi * radius^2) < 1e-6 := by sorry

end NUMINAMATH_CALUDE_sector_radius_proof_l480_48016


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l480_48036

/-- The area of a circle with diameter 10 meters is 25π square meters. -/
theorem circle_area_with_diameter_10 :
  let d : ℝ := 10
  let r : ℝ := d / 2
  let area : ℝ := π * r^2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l480_48036


namespace NUMINAMATH_CALUDE_bert_profit_is_one_l480_48085

/-- Calculates the profit from a sale given the sale price, markup, and tax rate. -/
def calculate_profit (sale_price markup tax_rate : ℚ) : ℚ :=
  let purchase_price := sale_price - markup
  let tax := sale_price * tax_rate
  sale_price - purchase_price - tax

/-- Proves that the profit is $1 given the specific conditions of Bert's sale. -/
theorem bert_profit_is_one :
  calculate_profit 90 10 (1/10) = 1 := by
  sorry

end NUMINAMATH_CALUDE_bert_profit_is_one_l480_48085
