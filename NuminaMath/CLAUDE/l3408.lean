import Mathlib

namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3408_340898

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = s → 3 * s = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3408_340898


namespace NUMINAMATH_CALUDE_mooncake_sales_properties_l3408_340858

/-- Represents the mooncake sales scenario -/
structure MooncakeSales where
  initial_purchase : ℕ
  purchase_price : ℕ
  initial_selling_price : ℕ
  price_reduction : ℕ
  sales_increase_per_yuan : ℕ

/-- Calculates the profit per box in the second sale -/
def profit_per_box (s : MooncakeSales) : ℤ :=
  s.initial_selling_price - s.purchase_price - s.price_reduction

/-- Calculates the expected sales volume in the second sale -/
def expected_sales_volume (s : MooncakeSales) : ℕ :=
  s.initial_purchase + s.sales_increase_per_yuan * s.price_reduction

/-- Theorem stating the properties of the mooncake sales scenario -/
theorem mooncake_sales_properties (s : MooncakeSales) 
  (h1 : s.initial_purchase = 180)
  (h2 : s.purchase_price = 40)
  (h3 : s.initial_selling_price = 52)
  (h4 : s.sales_increase_per_yuan = 10) :
  (∃ a : ℕ, 
    profit_per_box { initial_purchase := s.initial_purchase,
                     purchase_price := s.purchase_price,
                     initial_selling_price := s.initial_selling_price,
                     price_reduction := a,
                     sales_increase_per_yuan := s.sales_increase_per_yuan } = 12 - a ∧
    expected_sales_volume { initial_purchase := s.initial_purchase,
                            purchase_price := s.purchase_price,
                            initial_selling_price := s.initial_selling_price,
                            price_reduction := a,
                            sales_increase_per_yuan := s.sales_increase_per_yuan } = 180 + 10 * a ∧
    (profit_per_box { initial_purchase := s.initial_purchase,
                      purchase_price := s.purchase_price,
                      initial_selling_price := s.initial_selling_price,
                      price_reduction := a,
                      sales_increase_per_yuan := s.sales_increase_per_yuan } *
     expected_sales_volume { initial_purchase := s.initial_purchase,
                             purchase_price := s.purchase_price,
                             initial_selling_price := s.initial_selling_price,
                             price_reduction := a,
                             sales_increase_per_yuan := s.sales_increase_per_yuan } = 2000 →
     a = 2)) :=
by sorry

end NUMINAMATH_CALUDE_mooncake_sales_properties_l3408_340858


namespace NUMINAMATH_CALUDE_scientific_notation_of_1300000000_l3408_340807

/-- Expresses 1300000000 in scientific notation -/
theorem scientific_notation_of_1300000000 :
  (1300000000 : ℝ) = 1.3 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1300000000_l3408_340807


namespace NUMINAMATH_CALUDE_inverse_function_difference_l3408_340852

-- Define a function f and its inverse
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- Define the property that f and f_inv are inverse functions
def is_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Define the property that f(x+2) and f^(-1)(x-1) are inverse functions
def special_inverse_property (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv (x - 1) + 2) = x ∧ f_inv (f (x + 2) - 1) = x

-- Theorem statement
theorem inverse_function_difference
  (h1 : is_inverse f f_inv)
  (h2 : special_inverse_property f f_inv) :
  f_inv 2004 - f_inv 1 = 4006 :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_difference_l3408_340852


namespace NUMINAMATH_CALUDE_balloon_count_l3408_340812

def total_balloons (joan_initial : ℕ) (popped : ℕ) (jessica : ℕ) : ℕ :=
  (joan_initial - popped) + jessica

theorem balloon_count : total_balloons 9 5 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l3408_340812


namespace NUMINAMATH_CALUDE_find_x_l3408_340871

theorem find_x : ∃ x : ℝ, 3 * x = (26 - x) + 26 ∧ x = 13 := by sorry

end NUMINAMATH_CALUDE_find_x_l3408_340871


namespace NUMINAMATH_CALUDE_vector_magnitude_l3408_340808

/-- Given two vectors a and b in ℝ², prove that |2a + b| = 2√21 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  a = (3, -4) → 
  ‖b‖ = 2 → 
  a • b = -5 → 
  ‖2 • a + b‖ = 2 * Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3408_340808


namespace NUMINAMATH_CALUDE_zero_exponent_rule_l3408_340816

theorem zero_exponent_rule (a : ℝ) (h : a ≠ 0) : a ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_rule_l3408_340816


namespace NUMINAMATH_CALUDE_average_cost_theorem_l3408_340810

/-- The average cost per marker in cents, rounded to the nearest whole number -/
def average_cost_per_marker (num_markers : ℕ) (package_cost : ℚ) (handling_fee : ℚ) : ℕ :=
  let total_cost_cents := (package_cost + handling_fee) * 100
  let avg_cost_cents := total_cost_cents / num_markers
  (avg_cost_cents + 1/2).floor.toNat

theorem average_cost_theorem (num_markers : ℕ) (package_cost : ℚ) (handling_fee : ℚ)
  (h1 : num_markers = 150)
  (h2 : package_cost = 24.75)
  (h3 : handling_fee = 5.25) :
  average_cost_per_marker num_markers package_cost handling_fee = 20 := by
  sorry

end NUMINAMATH_CALUDE_average_cost_theorem_l3408_340810


namespace NUMINAMATH_CALUDE_statement_contrapositive_and_negation_l3408_340845

theorem statement_contrapositive_and_negation (x y : ℝ) :
  (((x - 1) * (y + 2) = 0 → x = 1 ∨ y = -2) ↔
   (x ≠ 1 ∧ y ≠ -2 → (x - 1) * (y + 2) ≠ 0)) ∧
  (¬((x - 1) * (y + 2) = 0 → x = 1 ∨ y = -2) ↔
   ((x - 1) * (y + 2) = 0 → x ≠ 1 ∧ y ≠ -2)) :=
by sorry

end NUMINAMATH_CALUDE_statement_contrapositive_and_negation_l3408_340845


namespace NUMINAMATH_CALUDE_water_containers_capacity_l3408_340846

/-- The problem of calculating the combined capacity of three water containers -/
theorem water_containers_capacity :
  ∀ (A B C : ℝ),
    (0.35 * A + 48 = 0.75 * A) →
    (0.45 * B + 36 = 0.95 * B) →
    (0.20 * C - 24 = 0.10 * C) →
    A + B + C = 432 :=
by sorry

end NUMINAMATH_CALUDE_water_containers_capacity_l3408_340846


namespace NUMINAMATH_CALUDE_sum_of_seventh_terms_l3408_340874

/-- First sequence defined by a_n = n^2 + n - 1 -/
def sequence_a (n : ℕ) : ℕ := n^2 + n - 1

/-- Second sequence defined by b_n = n(n+1)/2 -/
def sequence_b (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the 7th terms of both sequences is 83 -/
theorem sum_of_seventh_terms :
  sequence_a 7 + sequence_b 7 = 83 := by sorry

end NUMINAMATH_CALUDE_sum_of_seventh_terms_l3408_340874


namespace NUMINAMATH_CALUDE_fence_painting_ways_l3408_340862

/-- Represents the number of colors available for painting --/
def num_colors : ℕ := 3

/-- Represents the number of boards in the fence --/
def num_boards : ℕ := 10

/-- Calculates the total number of ways to paint the fence with any two adjacent boards having different colors --/
def total_ways : ℕ := num_colors * (2^(num_boards - 1))

/-- Calculates the number of ways to paint the fence using only two colors --/
def two_color_ways : ℕ := num_colors * (num_colors - 1)

/-- Theorem: The number of ways to paint a fence of 10 boards with 3 colors, 
    such that any two adjacent boards are of different colors and all three colors are used, 
    is equal to 1530 --/
theorem fence_painting_ways : 
  total_ways - two_color_ways = 1530 := by sorry

end NUMINAMATH_CALUDE_fence_painting_ways_l3408_340862


namespace NUMINAMATH_CALUDE_rocket_max_height_l3408_340853

/-- The height function of the rocket -/
def h (t : ℝ) : ℝ := -12 * t^2 + 72 * t + 36

/-- Theorem stating that the maximum height of the rocket is 144 feet -/
theorem rocket_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 144 := by
  sorry

end NUMINAMATH_CALUDE_rocket_max_height_l3408_340853


namespace NUMINAMATH_CALUDE_angle_of_inclination_of_line_l3408_340899

theorem angle_of_inclination_of_line (x y : ℝ) :
  x + Real.sqrt 3 * y - 1 = 0 →
  ∃ α : ℝ, α = 5 * π / 6 ∧ Real.tan α = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_of_line_l3408_340899


namespace NUMINAMATH_CALUDE_class_size_difference_l3408_340891

theorem class_size_difference (A B : ℕ) (h : A - 4 = B + 4) : A - B = 8 := by
  sorry

end NUMINAMATH_CALUDE_class_size_difference_l3408_340891


namespace NUMINAMATH_CALUDE_arithmetic_sequence_y_value_l3408_340872

/-- 
Given an arithmetic sequence with the first three terms 2/3, y-2, and 4y+1,
prove that y = -17/6.
-/
theorem arithmetic_sequence_y_value :
  ∀ y : ℚ,
  let a₁ : ℚ := 2/3
  let a₂ : ℚ := y - 2
  let a₃ : ℚ := 4*y + 1
  (a₂ - a₁ = a₃ - a₂) →
  y = -17/6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_y_value_l3408_340872


namespace NUMINAMATH_CALUDE_floor_area_calculation_l3408_340840

/-- The total area of a floor covered by square stone slabs -/
def floor_area (num_slabs : ℕ) (slab_side_length : ℝ) : ℝ :=
  (num_slabs : ℝ) * slab_side_length * slab_side_length

/-- Theorem: The total area of a floor covered by 50 square stone slabs, 
    each with a side length of 140 cm, is 980000 cm² -/
theorem floor_area_calculation :
  floor_area 50 140 = 980000 := by
  sorry

end NUMINAMATH_CALUDE_floor_area_calculation_l3408_340840


namespace NUMINAMATH_CALUDE_book_costs_18_l3408_340884

-- Define the cost of the album
def album_cost : ℝ := 20

-- Define the discount percentage for the CD
def cd_discount_percentage : ℝ := 0.30

-- Define the cost difference between the book and CD
def book_cd_difference : ℝ := 4

-- Calculate the cost of the CD
def cd_cost : ℝ := album_cost * (1 - cd_discount_percentage)

-- Calculate the cost of the book
def book_cost : ℝ := cd_cost + book_cd_difference

-- Theorem to prove
theorem book_costs_18 : book_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_book_costs_18_l3408_340884


namespace NUMINAMATH_CALUDE_problem_solution_l3408_340829

theorem problem_solution (m a b c d : ℚ) 
  (h1 : |m + 1| = 4)
  (h2 : a + b = 0)
  (h3 : a ≠ 0)
  (h4 : c * d = 1) :
  2*a + 2*b + (a + b - 3*c*d) - m = 2 ∨ 2*a + 2*b + (a + b - 3*c*d) - m = -6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3408_340829


namespace NUMINAMATH_CALUDE_range_of_m_l3408_340894

-- Define the propositions p and q
def p (x : ℝ) : Prop := |1 - (x-1)/3| < 2
def q (x m : ℝ) : Prop := (x-1)^2 < m^2

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬q x

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (sufficient_not_necessary (p) (q m)) ↔ (-5 < m ∧ m < 5) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3408_340894


namespace NUMINAMATH_CALUDE_not_perfect_power_probability_l3408_340819

/-- A function that determines if a number is a perfect power --/
def isPerfectPower (n : ℕ) : Prop :=
  ∃ (x y : ℕ), y > 1 ∧ x^y = n

/-- The count of numbers from 1 to 200 that are perfect powers --/
def perfectPowerCount : ℕ := 22

/-- The probability of selecting a number that is not a perfect power --/
def probabilityNotPerfectPower : ℚ := 89 / 100

theorem not_perfect_power_probability :
  (200 - perfectPowerCount : ℚ) / 200 = probabilityNotPerfectPower :=
sorry

end NUMINAMATH_CALUDE_not_perfect_power_probability_l3408_340819


namespace NUMINAMATH_CALUDE_sixteen_point_sphere_half_circumscribed_sphere_l3408_340803

/-- A tetrahedron with its associated spheres -/
structure Tetrahedron where
  /-- The radius of the circumscribed sphere -/
  R : ℝ
  /-- The radius of the sixteen-point sphere -/
  r : ℝ

/-- Theorem: There exists a tetrahedron for which the radius of its sixteen-point sphere 
    is equal to half the radius of its circumscribed sphere -/
theorem sixteen_point_sphere_half_circumscribed_sphere : 
  ∃ (t : Tetrahedron), t.r = t.R / 2 := by
  sorry


end NUMINAMATH_CALUDE_sixteen_point_sphere_half_circumscribed_sphere_l3408_340803


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l3408_340851

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l3408_340851


namespace NUMINAMATH_CALUDE_yard_length_18_trees_l3408_340801

/-- The length of a yard with equally spaced trees -/
def yardLength (numTrees : ℕ) (distanceBetweenTrees : ℝ) : ℝ :=
  (numTrees - 1 : ℝ) * distanceBetweenTrees

/-- Theorem: The length of a yard with 18 equally spaced trees,
    where the distance between consecutive trees is 15 meters, is 255 meters -/
theorem yard_length_18_trees : yardLength 18 15 = 255 := by
  sorry

#eval yardLength 18 15

end NUMINAMATH_CALUDE_yard_length_18_trees_l3408_340801


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3408_340890

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  1/a + 1/b + 1/c ≥ 3 ∧ (1/a + 1/b + 1/c = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3408_340890


namespace NUMINAMATH_CALUDE_gp_special_term_l3408_340880

def geometric_progression (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

theorem gp_special_term (b : ℕ → ℝ) (α : ℝ) :
  geometric_progression b →
  (0 < α) ∧ (α < Real.pi / 2) →
  b 25 = 2 * Real.tan α →
  b 31 = 2 * Real.sin α →
  b 37 = Real.sin (2 * α) :=
by sorry

end NUMINAMATH_CALUDE_gp_special_term_l3408_340880


namespace NUMINAMATH_CALUDE_class_average_mark_l3408_340804

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) (remaining_students : ℕ)
  (excluded_avg : ℚ) (remaining_avg : ℚ) :
  total_students = 25 →
  excluded_students = 5 →
  remaining_students = 20 →
  excluded_avg = 40 →
  remaining_avg = 90 →
  (total_students : ℚ) * ((excluded_students : ℚ) * excluded_avg + 
    (remaining_students : ℚ) * remaining_avg) / total_students = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l3408_340804


namespace NUMINAMATH_CALUDE_jason_final_cards_l3408_340886

def pokemon_card_transactions (initial_cards : ℕ) 
  (benny_trade_out benny_trade_in : ℕ) 
  (sean_trade_out sean_trade_in : ℕ) 
  (given_to_brother : ℕ) : ℕ :=
  initial_cards - benny_trade_out + benny_trade_in - sean_trade_out + sean_trade_in - given_to_brother

theorem jason_final_cards : 
  pokemon_card_transactions 5 2 3 3 4 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jason_final_cards_l3408_340886


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3408_340828

theorem absolute_value_inequality (x : ℝ) : 
  |((3 * x - 2) / (2 * x - 3))| > 3 ↔ 11/9 < x ∧ x < 7/3 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3408_340828


namespace NUMINAMATH_CALUDE_percentage_relation_l3408_340837

theorem percentage_relation (a b c P : ℝ) : 
  (P / 100) * a = 12 →
  (12 / 100) * b = 6 →
  c = b / a →
  c = P / 24 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3408_340837


namespace NUMINAMATH_CALUDE_unique_prime_sum_digits_l3408_340882

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Primality test -/
def isPrime (n : ℕ) : Prop := sorry

theorem unique_prime_sum_digits : 
  ∃! (n : ℕ), isPrime n ∧ n + S n + S (S n) = 3005 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_digits_l3408_340882


namespace NUMINAMATH_CALUDE_school_ratio_problem_l3408_340834

/-- Given a school with 300 students, where the ratio of boys to girls is x : y,
    prove that if the number of boys is increased by z such that the number of girls
    becomes x% of the total, then z = 300 - 3x - 300x / (x + y). -/
theorem school_ratio_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  ∃ z : ℝ, z = 300 - 3*x - 300*x / (x + y) := by
  sorry


end NUMINAMATH_CALUDE_school_ratio_problem_l3408_340834


namespace NUMINAMATH_CALUDE_thursday_steps_l3408_340821

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The target average steps per day -/
def target_average : ℕ := 9000

/-- Steps walked on Sunday -/
def sunday_steps : ℕ := 9400

/-- Steps walked on Monday -/
def monday_steps : ℕ := 9100

/-- Steps walked on Tuesday -/
def tuesday_steps : ℕ := 8300

/-- Steps walked on Wednesday -/
def wednesday_steps : ℕ := 9200

/-- Average steps for Friday and Saturday -/
def friday_saturday_average : ℕ := 9050

/-- Theorem: Given the conditions, Toby must have walked 8900 steps on Thursday to meet his weekly goal -/
theorem thursday_steps : 
  (days_in_week * target_average) - 
  (sunday_steps + monday_steps + tuesday_steps + wednesday_steps + 2 * friday_saturday_average) = 8900 := by
  sorry

end NUMINAMATH_CALUDE_thursday_steps_l3408_340821


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l3408_340889

-- Part 1
def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a :
  (∀ x, ¬(q x a) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

-- Part 2
def s (m : ℝ) : Prop :=
  ∃ x y, x^2 + (m - 3) * x + m = 0 ∧
         y^2 + (m - 3) * y + m = 0 ∧
         0 < x ∧ x < 1 ∧ 2 < y ∧ y < 3

def t (m : ℝ) : Prop :=
  ∀ x, m * x^2 - 2 * x + 1 > 0

theorem range_of_m :
  s m ∨ t m →
  (0 < m ∧ m < 2/3) ∨ m > 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l3408_340889


namespace NUMINAMATH_CALUDE_expand_product_l3408_340817

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3408_340817


namespace NUMINAMATH_CALUDE_estate_distribution_valid_l3408_340842

/-- Represents the estate distribution problem with twins --/
structure EstateDistribution :=
  (total : ℚ)
  (son_share : ℚ)
  (daughter_share : ℚ)
  (mother_share : ℚ)

/-- Checks if the distribution is valid according to the will's conditions --/
def is_valid_distribution (d : EstateDistribution) : Prop :=
  d.total = 210 ∧
  d.son_share + d.daughter_share + d.mother_share = d.total ∧
  d.son_share = (2/3) * d.total ∧
  d.daughter_share = (1/2) * d.mother_share

/-- Theorem stating that the given distribution is valid --/
theorem estate_distribution_valid :
  is_valid_distribution ⟨210, 140, 70/3, 140/3⟩ := by
  sorry

#check estate_distribution_valid

end NUMINAMATH_CALUDE_estate_distribution_valid_l3408_340842


namespace NUMINAMATH_CALUDE_book_club_meeting_lcm_l3408_340887

/-- The least common multiple of 5, 6, 8, 9, and 10 is 360 -/
theorem book_club_meeting_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 9 10))) = 360 := by
  sorry

end NUMINAMATH_CALUDE_book_club_meeting_lcm_l3408_340887


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3408_340826

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ∧ 
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3408_340826


namespace NUMINAMATH_CALUDE_unique_five_digit_square_last_five_l3408_340841

theorem unique_five_digit_square_last_five : ∃! (A : ℕ), 
  10000 ≤ A ∧ A < 100000 ∧ A^2 % 100000 = A :=
by
  use 90625
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_square_last_five_l3408_340841


namespace NUMINAMATH_CALUDE_balloon_difference_l3408_340875

theorem balloon_difference (your_balloons friend_balloons : ℕ) 
  (h1 : your_balloons = 7) 
  (h2 : friend_balloons = 5) : 
  your_balloons - friend_balloons = 2 := by
sorry

end NUMINAMATH_CALUDE_balloon_difference_l3408_340875


namespace NUMINAMATH_CALUDE_G_of_4_f_2_l3408_340870

-- Define the functions f and G
def f (a : ℝ) : ℝ := a^2 - 3
def G (a b : ℝ) : ℝ := b^2 - a

-- State the theorem
theorem G_of_4_f_2 : G 4 (f 2) = -3 := by sorry

end NUMINAMATH_CALUDE_G_of_4_f_2_l3408_340870


namespace NUMINAMATH_CALUDE_remainder_of_n_l3408_340857

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 7 = 1) (h2 : n^3 % 7 = 6) : n % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_l3408_340857


namespace NUMINAMATH_CALUDE_equation_solution_l3408_340825

theorem equation_solution (x : ℝ) : x ≠ 1 →
  ((3 * x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1)) ↔ (x = -4 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3408_340825


namespace NUMINAMATH_CALUDE_committee_formation_l3408_340802

theorem committee_formation (n m : ℕ) (hn : n = 8) (hm : m = 4) :
  Nat.choose n m = 70 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_l3408_340802


namespace NUMINAMATH_CALUDE_star_equation_roots_l3408_340881

-- Define the "★" operation
def star (a b : ℝ) : ℝ := a^2 - b^2

-- Theorem statement
theorem star_equation_roots :
  let x₁ : ℝ := 4
  let x₂ : ℝ := -4
  (star (star 2 3) x₁ = 9) ∧ (star (star 2 3) x₂ = 9) ∧
  (∀ x : ℝ, star (star 2 3) x = 9 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_star_equation_roots_l3408_340881


namespace NUMINAMATH_CALUDE_nancy_coffee_days_l3408_340861

/-- Represents Nancy's coffee buying habits and expenses -/
structure CoffeeExpense where
  double_espresso_price : ℚ
  iced_coffee_price : ℚ
  total_spent : ℚ

/-- Calculates the number of days Nancy has been buying coffee -/
def days_buying_coffee (expense : CoffeeExpense) : ℚ :=
  expense.total_spent / (expense.double_espresso_price + expense.iced_coffee_price)

/-- Theorem stating that Nancy has been buying coffee for 20 days -/
theorem nancy_coffee_days :
  let expense : CoffeeExpense := {
    double_espresso_price := 3,
    iced_coffee_price := 5/2,
    total_spent := 110
  }
  days_buying_coffee expense = 20 := by
  sorry

end NUMINAMATH_CALUDE_nancy_coffee_days_l3408_340861


namespace NUMINAMATH_CALUDE_customer_income_proof_l3408_340897

/-- Proves that given a group of 50 customers with an average income of $45,000, 
    where 10 of these customers have an average income of $55,000, 
    the average income of the remaining 40 customers is $42,500. -/
theorem customer_income_proof (total_customers : Nat) (wealthy_customers : Nat)
  (remaining_customers : Nat) (total_avg_income : ℝ) (wealthy_avg_income : ℝ) :
  total_customers = 50 →
  wealthy_customers = 10 →
  remaining_customers = total_customers - wealthy_customers →
  total_avg_income = 45000 →
  wealthy_avg_income = 55000 →
  (total_customers * total_avg_income - wealthy_customers * wealthy_avg_income) / remaining_customers = 42500 :=
by sorry

end NUMINAMATH_CALUDE_customer_income_proof_l3408_340897


namespace NUMINAMATH_CALUDE_perpendicular_distance_to_adjacent_plane_l3408_340885

/-- A rectangular parallelepiped with dimensions 5 × 5 × 4 -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  length_eq : length = 5
  width_eq : width = 5
  height_eq : height = 4

/-- A vertex of the parallelepiped -/
structure Vertex where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The perpendicular distance from a vertex to a plane -/
def perpendicularDistance (v : Vertex) (plane : Vertex → Vertex → Vertex → Prop) : ℝ :=
  sorry

theorem perpendicular_distance_to_adjacent_plane (p : Parallelepiped) 
  (h v1 v2 v3 : Vertex) 
  (adj1 : h.z = 0)
  (adj2 : v1.z = 0 ∧ v1.y = 0 ∧ v1.x = p.length)
  (adj3 : v2.z = 0 ∧ v2.x = 0 ∧ v2.y = p.width)
  (adj4 : v3.x = 0 ∧ v3.y = 0 ∧ v3.z = p.height) :
  perpendicularDistance h (fun a b c => True) = 4 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_distance_to_adjacent_plane_l3408_340885


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3408_340800

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (m, -1)
  are_parallel a b → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3408_340800


namespace NUMINAMATH_CALUDE_even_function_value_l3408_340850

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

-- Define the property of being an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the domain of the function
def domain (b : ℝ) : Set ℝ := Set.Ioo (-b) (2*b - 2)

-- State the theorem
theorem even_function_value (a b : ℝ) :
  is_even_function (f a b) ∧ (∀ x ∈ domain b, f a b x = f a b x) →
  f a b (b/2) = 2 :=
sorry

end NUMINAMATH_CALUDE_even_function_value_l3408_340850


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l3408_340844

theorem average_of_remaining_numbers
  (total_count : Nat)
  (total_average : ℝ)
  (first_pair_average : ℝ)
  (second_pair_average : ℝ)
  (h_total_count : total_count = 6)
  (h_total_average : total_average = 4.60)
  (h_first_pair_average : first_pair_average = 3.4)
  (h_second_pair_average : second_pair_average = 3.8) :
  (total_count : ℝ) * total_average - 2 * first_pair_average - 2 * second_pair_average = 2 * 6.6 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l3408_340844


namespace NUMINAMATH_CALUDE_abc_inequality_l3408_340827

noncomputable def e : ℝ := Real.exp 1

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1)
  (eqa : a^2 - 2 * Real.log a + 1 = e)
  (eqb : b^2 - 2 * Real.log b + 2 = e^2)
  (eqc : c^2 - 2 * Real.log c + 3 = e^3) : 
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3408_340827


namespace NUMINAMATH_CALUDE_gcd_5039_3427_l3408_340833

theorem gcd_5039_3427 : Nat.gcd 5039 3427 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5039_3427_l3408_340833


namespace NUMINAMATH_CALUDE_non_congruent_squares_count_l3408_340854

/-- A lattice point on a 2D grid --/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A square on a lattice grid --/
structure LatticeSquare where
  vertices : Finset LatticePoint
  size : ℕ

/-- The size of the grid --/
def gridSize : ℕ := 6

/-- Function to count standard squares of a given size --/
def countStandardSquares (k : ℕ) : ℕ :=
  (gridSize - k + 1) ^ 2

/-- Function to count 45-degree tilted squares with diagonal of a given size --/
def countTiltedSquares (k : ℕ) : ℕ :=
  (gridSize - k) ^ 2

/-- Function to count 45-degree tilted squares with diagonal of a rectangle --/
def countRectangleDiagonalSquares (w h : ℕ) : ℕ :=
  2 * (gridSize - w) * (gridSize - h)

/-- The total number of non-congruent squares on the grid --/
def totalNonCongruentSquares : ℕ :=
  (countStandardSquares 1) + (countStandardSquares 2) + (countStandardSquares 3) +
  (countStandardSquares 4) + (countStandardSquares 5) +
  (countTiltedSquares 1) + (countTiltedSquares 2) +
  (countRectangleDiagonalSquares 1 2) + (countRectangleDiagonalSquares 1 3)

/-- Theorem: The number of non-congruent squares on a 6x6 grid is 201 --/
theorem non_congruent_squares_count :
  totalNonCongruentSquares = 201 := by
  sorry

end NUMINAMATH_CALUDE_non_congruent_squares_count_l3408_340854


namespace NUMINAMATH_CALUDE_some_value_theorem_l3408_340888

theorem some_value_theorem (w x y : ℝ) (h1 : (w + x) / 2 = 0.5) (h2 : w * x = y) :
  ∃ some_value : ℝ, 3 / w + some_value = 3 / y ∧ some_value = 6 := by
  sorry

end NUMINAMATH_CALUDE_some_value_theorem_l3408_340888


namespace NUMINAMATH_CALUDE_x_minus_y_equals_twelve_l3408_340883

theorem x_minus_y_equals_twelve (x y : ℕ) : 
  3^x * 4^y = 531441 → x = 12 → x - y = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_twelve_l3408_340883


namespace NUMINAMATH_CALUDE_smallest_divisor_after_361_l3408_340893

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem smallest_divisor_after_361 (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m ≤ 9999)  -- m is a 4-digit number
  (h2 : is_even m)             -- m is even
  (h3 : m % 361 = 0)           -- m is divisible by 361
  : (∃ d : ℕ, d ∣ m ∧ 361 < d ∧ ∀ d' : ℕ, d' ∣ m → 361 < d' → d ≤ d') → 
    (∃ d : ℕ, d ∣ m ∧ 361 < d ∧ ∀ d' : ℕ, d' ∣ m → 361 < d' → d ≤ d' ∧ d = 380) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_361_l3408_340893


namespace NUMINAMATH_CALUDE_sarah_apples_to_teachers_l3408_340895

/-- Calculates the number of apples given to teachers -/
def apples_to_teachers (initial : ℕ) (to_friends : ℕ) (eaten : ℕ) (left : ℕ) : ℕ :=
  initial - to_friends - eaten - left

/-- Theorem stating that Sarah gave 16 apples to teachers -/
theorem sarah_apples_to_teachers :
  apples_to_teachers 25 5 1 3 = 16 := by
  sorry

#eval apples_to_teachers 25 5 1 3

end NUMINAMATH_CALUDE_sarah_apples_to_teachers_l3408_340895


namespace NUMINAMATH_CALUDE_triangle_problem_l3408_340839

theorem triangle_problem (a b c A B C : Real) (R : Real) :
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- Angles are positive
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Sides are positive
  a + b + c = π ∧  -- Sum of angles in a triangle
  2 * B * Real.cos A = C * Real.cos a + A * Real.cos c ∧  -- Given equation
  A + B + C = 8 ∧  -- Perimeter is 8
  R = Real.sqrt 3 ∧  -- Radius of circumscribed circle is √3
  2 * R * Real.sin (a / 2) = A ∧  -- Relation between side and circumradius
  2 * R * Real.sin (b / 2) = B ∧
  2 * R * Real.sin (c / 2) = C →
  a = π / 3 ∧  -- Angle A is 60°
  A * B * Real.sin c / 2 = 4 * Real.sqrt 3 / 3  -- Area of triangle
  :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3408_340839


namespace NUMINAMATH_CALUDE_range_of_a_l3408_340805

-- Define the sets A, B, and C
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 3) - 1 / Real.sqrt (7 - x)}
def B : Set ℝ := {y | ∃ x, y = -x^2 + 2*x + 8}
def C (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- Define the theorem
theorem range_of_a (a : ℝ) : A ∪ C a = C a → a ≥ 7 ∨ a + 1 < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3408_340805


namespace NUMINAMATH_CALUDE_equal_angles_l3408_340831

-- Define the basic structures
variable (Circle₁ Circle₂ : Set (ℝ × ℝ))
variable (K M A B C D : ℝ × ℝ)

-- Define the conditions
variable (h1 : K ∈ Circle₁ ∩ Circle₂)
variable (h2 : M ∈ Circle₁ ∩ Circle₂)
variable (h3 : A ∈ Circle₁)
variable (h4 : B ∈ Circle₂)
variable (h5 : C ∈ Circle₁)
variable (h6 : D ∈ Circle₂)
variable (h7 : ∃ ray₁ : Set (ℝ × ℝ), K ∈ ray₁ ∧ A ∈ ray₁ ∧ B ∈ ray₁)
variable (h8 : ∃ ray₂ : Set (ℝ × ℝ), K ∈ ray₂ ∧ C ∈ ray₂ ∧ D ∈ ray₂)

-- Define the angle function
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem equal_angles : angle M A B = angle M C D := sorry

end NUMINAMATH_CALUDE_equal_angles_l3408_340831


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l3408_340830

/-- Represents the price and volume of orangeade on a given day -/
structure OrangeadeDay where
  juice : ℝ
  water : ℝ
  price : ℝ

/-- Calculates the revenue for a given day -/
def revenue (day : OrangeadeDay) : ℝ :=
  (day.juice + day.water) * day.price

theorem orangeade_price_day2 (day1 day2 : OrangeadeDay) :
  day1.juice = day1.water →
  day2.juice = day1.juice →
  day2.water = 2 * day1.water →
  day1.price = 0.9 →
  revenue day1 = revenue day2 →
  day2.price = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_price_day2_l3408_340830


namespace NUMINAMATH_CALUDE_homework_check_probability_l3408_340879

/-- Represents the days of the week when math lessons occur -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- The probability space for the homework checking scenario -/
structure HomeworkProbability where
  /-- The probability that the teacher will not check homework at all during the week -/
  p_no_check : ℝ
  /-- The probability that the teacher will check homework exactly once during the week -/
  p_check_once : ℝ
  /-- The number of math lessons per week -/
  num_lessons : ℕ
  /-- Assumption: probabilities sum to 1 -/
  sum_to_one : p_no_check + p_check_once = 1
  /-- Assumption: probabilities are non-negative -/
  non_negative : 0 ≤ p_no_check ∧ 0 ≤ p_check_once
  /-- Assumption: there are 5 math lessons per week -/
  five_lessons : num_lessons = 5

/-- The main theorem to prove -/
theorem homework_check_probability (hp : HomeworkProbability) :
  hp.p_no_check = 1/2 →
  hp.p_check_once = 1/2 →
  (1/hp.num_lessons : ℝ) * hp.p_check_once / (hp.p_no_check + (1/hp.num_lessons) * hp.p_check_once) = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_homework_check_probability_l3408_340879


namespace NUMINAMATH_CALUDE_three_intersections_iff_l3408_340822

/-- The number of intersection points between the curves x^2 + y^2 = a^2 and y = x^2 + a -/
def num_intersections (a : ℝ) : ℕ :=
  sorry

/-- Theorem stating the condition for exactly 3 intersection points -/
theorem three_intersections_iff (a : ℝ) : 
  num_intersections a = 3 ↔ a < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_three_intersections_iff_l3408_340822


namespace NUMINAMATH_CALUDE_range_of_a_l3408_340832

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∃ a : ℝ, (∀ x : ℝ, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x : ℝ, ¬(q x a) ∧ p x)) →
  (∀ a : ℝ, (∀ x : ℝ, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x : ℝ, ¬(q x a) ∧ p x) → a ≥ 1) :=
by sorry

-- The range of a is [1, +∞)

end NUMINAMATH_CALUDE_range_of_a_l3408_340832


namespace NUMINAMATH_CALUDE_cd_length_ratio_l3408_340838

/-- Given three CDs, where two have the same length and the total length of all CDs is known,
    this theorem proves the ratio of the length of the third CD to one of the first two. -/
theorem cd_length_ratio (length_first_two : ℝ) (total_length : ℝ) : 
  length_first_two > 0 →
  total_length > 2 * length_first_two →
  (total_length - 2 * length_first_two) / length_first_two = 2 := by
  sorry

#check cd_length_ratio

end NUMINAMATH_CALUDE_cd_length_ratio_l3408_340838


namespace NUMINAMATH_CALUDE_tangent_circles_m_values_l3408_340856

-- Define the equations of the circles
def C₁ (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x + 4*y + m^2 - 5 = 0
def C₂ (x y m : ℝ) : Prop := x^2 + y^2 + 2*x - 2*m*y + m^2 - 3 = 0

-- Define the condition for circles being tangent
def are_tangent (m : ℝ) : Prop :=
  ∃ x y, C₁ x y m ∧ C₂ x y m ∧
  (∀ x' y', C₁ x' y' m ∧ C₂ x' y' m → (x', y') = (x, y))

-- Theorem statement
theorem tangent_circles_m_values :
  {m : ℝ | are_tangent m} = {-5, -2, -1, 2} :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_m_values_l3408_340856


namespace NUMINAMATH_CALUDE_grid_toothpicks_l3408_340818

/-- Calculates the number of toothpicks required for a rectangular grid. -/
def toothpicks_in_grid (height width : ℕ) : ℕ :=
  (height + 1) * width + (width + 1) * height

/-- Theorem: A rectangular grid with height 15 and width 12 requires 387 toothpicks. -/
theorem grid_toothpicks : toothpicks_in_grid 15 12 = 387 := by
  sorry

#eval toothpicks_in_grid 15 12

end NUMINAMATH_CALUDE_grid_toothpicks_l3408_340818


namespace NUMINAMATH_CALUDE_min_boxes_for_muffins_l3408_340873

/-- Represents the number of muffins that can be packed in each box type -/
structure BoxCapacity where
  large : Nat
  medium : Nat
  small : Nat

/-- Represents the number of boxes used for each type -/
structure BoxCount where
  large : Nat
  medium : Nat
  small : Nat

def total_muffins : Nat := 250

def box_capacity : BoxCapacity := ⟨12, 8, 4⟩

def box_count : BoxCount := ⟨20, 1, 1⟩

/-- Calculates the total number of muffins that can be packed in the given boxes -/
def muffins_packed (capacity : BoxCapacity) (count : BoxCount) : Nat :=
  capacity.large * count.large + capacity.medium * count.medium + capacity.small * count.small

/-- Calculates the total number of boxes used -/
def total_boxes (count : BoxCount) : Nat :=
  count.large + count.medium + count.small

theorem min_boxes_for_muffins :
  muffins_packed box_capacity box_count = total_muffins ∧
  total_boxes box_count = 22 ∧
  ∀ (other_count : BoxCount),
    muffins_packed box_capacity other_count ≥ total_muffins →
    total_boxes other_count ≥ total_boxes box_count :=
by
  sorry

end NUMINAMATH_CALUDE_min_boxes_for_muffins_l3408_340873


namespace NUMINAMATH_CALUDE_system_solution_l3408_340814

theorem system_solution (a b c m n k : ℚ) :
  (∃ x y : ℚ, a * x + b * y = c ∧ m * x - n * y = k ∧ x = -3 ∧ y = 4) →
  (∃ x y : ℚ, a * (x + y) + b * (x - y) = c ∧ m * (x + y) - n * (x - y) = k ∧ x = 1/2 ∧ y = -7/2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3408_340814


namespace NUMINAMATH_CALUDE_duck_flying_days_l3408_340896

/-- The number of days a duck spends flying during winter, summer, and spring -/
def total_flying_days (south_days : ℕ) (east_days : ℕ) : ℕ :=
  south_days + 2 * south_days + east_days

/-- Theorem: The duck spends 180 days flying during winter, summer, and spring -/
theorem duck_flying_days : total_flying_days 40 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_duck_flying_days_l3408_340896


namespace NUMINAMATH_CALUDE_molecular_weight_CCl4_is_152_l3408_340868

/-- The molecular weight of Carbon tetrachloride -/
def molecular_weight_CCl4 : ℝ := 152

/-- The number of moles in the given sample -/
def num_moles : ℝ := 9

/-- The total molecular weight of the given sample -/
def total_weight : ℝ := 1368

/-- Theorem stating that the molecular weight of Carbon tetrachloride is 152 g/mol -/
theorem molecular_weight_CCl4_is_152 :
  molecular_weight_CCl4 = total_weight / num_moles :=
by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_CCl4_is_152_l3408_340868


namespace NUMINAMATH_CALUDE_boxes_with_neither_l3408_340876

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (erasers : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : markers = 8)
  (h3 : erasers = 5)
  (h4 : both = 4)
  : total - (markers + erasers - both) = 6 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l3408_340876


namespace NUMINAMATH_CALUDE_triangle_properties_l3408_340835

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b^2 + t.c^2 = 3 * t.b * t.c * Real.cos t.A)
  (h2 : t.B = t.C)
  (h3 : t.a = 2) :
  (1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 5) ∧ 
  (Real.tan t.A / Real.tan t.B + Real.tan t.A / Real.tan t.C = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3408_340835


namespace NUMINAMATH_CALUDE_largest_two_digit_satisfying_property_l3408_340892

/-- Given a two-digit number n = 10a + b, where a and b are single digits,
    we define a function that switches the digits and adds 5. -/
def switchAndAdd5 (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a + 5

/-- The property that switching digits and adding 5 results in 3n -/
def satisfiesProperty (n : ℕ) : Prop :=
  switchAndAdd5 n = 3 * n

theorem largest_two_digit_satisfying_property :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ satisfiesProperty n ∧
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ satisfiesProperty m → m ≤ n :=
by
  use 13
  sorry

end NUMINAMATH_CALUDE_largest_two_digit_satisfying_property_l3408_340892


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_plus_c_l3408_340869

theorem min_value_2a_plus_b_plus_c (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a * (a + b + c) + b * c = 4) : 
  ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * (x + y + z) + y * z = 4 → 2*a + b + c ≤ 2*x + y + z :=
by sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_plus_c_l3408_340869


namespace NUMINAMATH_CALUDE_example_linear_equation_l3408_340867

/-- Represents a linear equation with two variables -/
structure LinearEquationTwoVars where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ a * x + b * y = c

/-- The equation 5x + y = 2 is a linear equation with two variables -/
theorem example_linear_equation : ∃ e : LinearEquationTwoVars, e.a = 5 ∧ e.b = 1 ∧ e.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_example_linear_equation_l3408_340867


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3408_340823

/-- The distance between foci of an ellipse with given semi-major and semi-minor axes -/
theorem ellipse_foci_distance (a b : ℝ) (ha : a = 10) (hb : b = 4) :
  2 * Real.sqrt (a^2 - b^2) = 4 * Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3408_340823


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3408_340855

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 * x - 6) = 10 → x = 106 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3408_340855


namespace NUMINAMATH_CALUDE_latest_sixty_degree_time_l3408_340847

/-- Temperature model as a function of time -/
def T (t : ℝ) : ℝ := -2 * t^2 + 16 * t + 40

/-- The statement to prove -/
theorem latest_sixty_degree_time :
  ∃ t_max : ℝ, t_max = 5 ∧ 
  T t_max = 60 ∧ 
  ∀ t : ℝ, T t = 60 → t ≤ t_max :=
sorry

end NUMINAMATH_CALUDE_latest_sixty_degree_time_l3408_340847


namespace NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_3_between_1_and_2_l3408_340863

theorem sqrt_12_minus_sqrt_3_between_1_and_2 : 1 < Real.sqrt 12 - Real.sqrt 3 ∧ Real.sqrt 12 - Real.sqrt 3 < 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_3_between_1_and_2_l3408_340863


namespace NUMINAMATH_CALUDE_regression_line_equation_l3408_340849

theorem regression_line_equation (slope : ℝ) (center_x center_y : ℝ) :
  slope = 2.03 →
  center_x = 5 →
  center_y = 11 →
  ∀ x y : ℝ, y = slope * x + (center_y - slope * center_x) ↔ y = 2.03 * x + 0.85 :=
by sorry

end NUMINAMATH_CALUDE_regression_line_equation_l3408_340849


namespace NUMINAMATH_CALUDE_monday_miles_proof_l3408_340864

def weekly_miles : ℕ := 30
def wednesday_miles : ℕ := 12

theorem monday_miles_proof (monday_miles : ℕ) 
  (h1 : monday_miles + wednesday_miles + 2 * monday_miles = weekly_miles) : 
  monday_miles = 6 := by
  sorry

end NUMINAMATH_CALUDE_monday_miles_proof_l3408_340864


namespace NUMINAMATH_CALUDE_george_red_marbles_l3408_340824

/-- The number of red marbles in George's collection --/
def red_marbles (total yellow white green : ℕ) : ℕ :=
  total - (yellow + white + green)

/-- Theorem stating the number of red marbles in George's collection --/
theorem george_red_marbles :
  ∀ (total yellow white green : ℕ),
    total = 50 →
    yellow = 12 →
    white = total / 2 →
    green = yellow / 2 →
    red_marbles total yellow white green = 7 := by
  sorry

end NUMINAMATH_CALUDE_george_red_marbles_l3408_340824


namespace NUMINAMATH_CALUDE_three_tenths_of_number_l3408_340843

theorem three_tenths_of_number (n : ℚ) (h : (1/3) * (1/4) * n = 15) : (3/10) * n = 54 := by
  sorry

end NUMINAMATH_CALUDE_three_tenths_of_number_l3408_340843


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3408_340811

theorem consecutive_integers_sum :
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ y = x + 1 ∧ z = y + 1 ∧ x + y + z = 48) ∧
  (∃ (x y z w : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ 
    Even x ∧ Even y ∧ Even z ∧ Even w ∧
    y = x + 2 ∧ z = y + 2 ∧ w = z + 2 ∧ x + y + z + w = 52) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3408_340811


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l3408_340865

theorem no_solution_absolute_value_equation :
  (∀ x : ℝ, (x - 4)^2 ≠ 0 → ∃ y : ℝ, (y - 4)^2 = 0) ∧
  (∀ x : ℝ, |(-5 : ℝ) * x| + 10 ≠ 0) ∧
  (∀ x : ℝ, Real.sqrt (-x) - 3 ≠ 0 → ∃ y : ℝ, Real.sqrt (-y) - 3 = 0) ∧
  (∀ x : ℝ, Real.sqrt x - 7 ≠ 0 → ∃ y : ℝ, Real.sqrt y - 7 = 0) ∧
  (∀ x : ℝ, |(-5 : ℝ) * x| - 6 ≠ 0 → ∃ y : ℝ, |(-5 : ℝ) * y| - 6 = 0) :=
by sorry


end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l3408_340865


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3408_340878

theorem sufficient_but_not_necessary (p q : Prop) :
  (p ∧ q → ¬(¬p)) ∧ ¬(¬(¬p) → p ∧ q) := by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3408_340878


namespace NUMINAMATH_CALUDE_triangle_property_l3408_340859

-- Define a triangle
structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_angles : α + β + γ = Real.pi

-- Define the condition
def condition (t : Triangle) : Prop :=
  Real.tan t.β * (Real.sin t.γ)^2 = Real.tan t.γ * (Real.sin t.β)^2

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.α = t.β ∨ t.β = t.γ ∨ t.γ = t.α

-- Define right-angled triangle
def is_right_angled (t : Triangle) : Prop :=
  t.α = Real.pi/2 ∨ t.β = Real.pi/2 ∨ t.γ = Real.pi/2

-- Theorem statement
theorem triangle_property (t : Triangle) :
  condition t → is_isosceles t ∨ is_right_angled t :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l3408_340859


namespace NUMINAMATH_CALUDE_eighth_term_of_specific_sequence_l3408_340836

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  first_term : ℝ
  common_diff : ℝ

/-- The nth term of an arithmetic sequence. -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + seq.common_diff * (n - 1 : ℝ)

theorem eighth_term_of_specific_sequence :
  ∃ (seq : ArithmeticSequence),
    nth_term seq 4 = 23 ∧
    nth_term seq 6 = 47 ∧
    nth_term seq 8 = 71 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_specific_sequence_l3408_340836


namespace NUMINAMATH_CALUDE_mode_better_representation_l3408_340813

/-- Represents the salary distribution of employees in a company -/
structure SalaryDistribution where
  manager_salary : ℕ
  deputy_manager_salary : ℕ
  employee_salary : ℕ
  manager_count : ℕ
  deputy_manager_count : ℕ
  employee_count : ℕ

/-- Calculates the mean salary -/
def mean_salary (sd : SalaryDistribution) : ℚ :=
  (sd.manager_salary * sd.manager_count +
   sd.deputy_manager_salary * sd.deputy_manager_count +
   sd.employee_salary * sd.employee_count) /
  (sd.manager_count + sd.deputy_manager_count + sd.employee_count)

/-- Finds the mode salary -/
def mode_salary (sd : SalaryDistribution) : ℕ :=
  if sd.employee_count > sd.manager_count ∧ sd.employee_count > sd.deputy_manager_count then
    sd.employee_salary
  else if sd.deputy_manager_count > sd.manager_count then
    sd.deputy_manager_salary
  else
    sd.manager_salary

/-- Represents how well a measure describes the concentration trend -/
def concentration_measure (salary : ℕ) (sd : SalaryDistribution) : ℚ :=
  (sd.manager_count * (if salary = sd.manager_salary then 1 else 0) +
   sd.deputy_manager_count * (if salary = sd.deputy_manager_salary then 1 else 0) +
   sd.employee_count * (if salary = sd.employee_salary then 1 else 0)) /
  (sd.manager_count + sd.deputy_manager_count + sd.employee_count)

/-- Theorem stating that the mode better represents the concentration trend than the mean -/
theorem mode_better_representation (sd : SalaryDistribution)
  (h1 : sd.manager_salary = 12000)
  (h2 : sd.deputy_manager_salary = 8000)
  (h3 : sd.employee_salary = 3000)
  (h4 : sd.manager_count = 1)
  (h5 : sd.deputy_manager_count = 1)
  (h6 : sd.employee_count = 8) :
  concentration_measure (mode_salary sd) sd > concentration_measure (Nat.floor (mean_salary sd)) sd :=
  sorry

end NUMINAMATH_CALUDE_mode_better_representation_l3408_340813


namespace NUMINAMATH_CALUDE_girls_in_college_l3408_340848

theorem girls_in_college (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) :
  total_students = 455 →
  ratio_boys = 8 →
  ratio_girls = 5 →
  ∃ (num_girls : ℕ), num_girls * (ratio_boys + ratio_girls) = total_students * ratio_girls ∧ num_girls = 175 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_in_college_l3408_340848


namespace NUMINAMATH_CALUDE_zebra_stripes_l3408_340860

theorem zebra_stripes (w n b : ℕ) : 
  w + n = b + 1 →  -- Total black stripes is one more than white stripes
  b = w + 7 →      -- White stripes are 7 more than wide black stripes
  n = 8            -- Number of narrow black stripes is 8
:= by sorry

end NUMINAMATH_CALUDE_zebra_stripes_l3408_340860


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l3408_340866

theorem line_tangent_to_parabola :
  ∃! (x y : ℝ), 4 * x + 3 * y + 18 = 0 ∧ y^2 = 32 * x := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l3408_340866


namespace NUMINAMATH_CALUDE_room_tiles_count_l3408_340877

/-- Represents the dimensions of a room in centimeters -/
structure RoomDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a room given its dimensions -/
def roomVolume (d : RoomDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Finds the greatest common divisor of three natural numbers -/
def gcd3 (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

/-- Calculates the number of cubic tiles needed to fill a room -/
def numTiles (d : RoomDimensions) : ℕ :=
  let tileSize := gcd3 d.length d.width d.height
  roomVolume d / (tileSize * tileSize * tileSize)

/-- The main theorem stating the number of tiles needed for the given room -/
theorem room_tiles_count (room : RoomDimensions) 
    (h1 : room.length = 624)
    (h2 : room.width = 432)
    (h3 : room.height = 356) : 
  numTiles room = 1493952 := by
  sorry

end NUMINAMATH_CALUDE_room_tiles_count_l3408_340877


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l3408_340815

/-- The shortest distance between a point on the parabola y = x^2 - 9x + 25 
    and a point on the line y = x - 8 is 4√2. -/
theorem shortest_distance_parabola_to_line :
  let parabola := {(x, y) : ℝ × ℝ | y = x^2 - 9*x + 25}
  let line := {(x, y) : ℝ × ℝ | y = x - 8}
  ∃ (d : ℝ), d = 4 * Real.sqrt 2 ∧ 
    ∀ (A : ℝ × ℝ) (B : ℝ × ℝ), A ∈ parabola → B ∈ line → 
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l3408_340815


namespace NUMINAMATH_CALUDE_bird_feet_count_l3408_340809

theorem bird_feet_count (num_birds : ℕ) (feet_per_bird : ℕ) (h1 : num_birds = 46) (h2 : feet_per_bird = 2) :
  num_birds * feet_per_bird = 92 := by
  sorry

end NUMINAMATH_CALUDE_bird_feet_count_l3408_340809


namespace NUMINAMATH_CALUDE_line_intercept_l3408_340820

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line is the x-coordinate where the line crosses the x-axis -/
def x_intercept (l : Line) : ℝ := sorry

/-- Theorem: The line passing through (7, -3) and (3, 1) intersects the x-axis at (4, 0) -/
theorem line_intercept : 
  let l : Line := { x₁ := 7, y₁ := -3, x₂ := 3, y₂ := 1 }
  x_intercept l = 4 := by sorry

end NUMINAMATH_CALUDE_line_intercept_l3408_340820


namespace NUMINAMATH_CALUDE_total_discount_savings_l3408_340806

def mangoes_per_box : ℕ := 10 * 12 -- 10 dozen

def prices_per_dozen : List ℕ := [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def total_boxes : ℕ := 36

def discount_rate (boxes : ℕ) : ℚ :=
  if boxes ≥ 30 then 15 / 100
  else if boxes ≥ 20 then 10 / 100
  else if boxes ≥ 10 then 5 / 100
  else 0

theorem total_discount_savings : 
  let total_cost := (prices_per_dozen.map (· * mangoes_per_box)).sum * total_boxes
  let discounted_cost := total_cost * (1 - discount_rate total_boxes)
  total_cost - discounted_cost = 5090 := by
  sorry

end NUMINAMATH_CALUDE_total_discount_savings_l3408_340806
