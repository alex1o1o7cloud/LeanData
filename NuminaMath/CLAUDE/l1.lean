import Mathlib

namespace min_sum_of_weights_l1_129

theorem min_sum_of_weights (S : ℕ) : 
  S > 280 ∧ S % 70 = 30 → S ≥ 310 :=
by sorry

end min_sum_of_weights_l1_129


namespace child_attraction_fee_is_two_l1_170

/-- Represents the cost of various tickets and the family composition --/
structure ParkCosts where
  entrance_fee : ℕ
  adult_attraction_fee : ℕ
  child_attraction_fee : ℕ
  num_children : ℕ
  num_parents : ℕ
  num_grandparents : ℕ
  total_cost : ℕ

/-- Theorem stating that given the conditions, the child attraction fee is $2 --/
theorem child_attraction_fee_is_two (c : ParkCosts)
  (h1 : c.entrance_fee = 5)
  (h2 : c.adult_attraction_fee = 4)
  (h3 : c.num_children = 4)
  (h4 : c.num_parents = 2)
  (h5 : c.num_grandparents = 1)
  (h6 : c.total_cost = 55)
  (h7 : c.total_cost = (c.num_children + c.num_parents + c.num_grandparents) * c.entrance_fee +
                       (c.num_parents + c.num_grandparents) * c.adult_attraction_fee +
                       c.num_children * c.child_attraction_fee) :
  c.child_attraction_fee = 2 :=
by sorry

end child_attraction_fee_is_two_l1_170


namespace davids_english_marks_l1_182

/-- Given David's marks in 4 subjects and the average of all 5 subjects, 
    prove that his marks in English are 70. -/
theorem davids_english_marks 
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (h1 : math_marks = 63)
  (h2 : physics_marks = 80)
  (h3 : chemistry_marks = 63)
  (h4 : biology_marks = 65)
  (h5 : average_marks = 68.2)
  (h6 : (math_marks + physics_marks + chemistry_marks + biology_marks + english_marks : ℚ) / 5 = average_marks) :
  english_marks = 70 :=
by sorry

end davids_english_marks_l1_182


namespace battle_treaty_day_l1_199

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Calculates the day of the week for a given date -/
def dayOfWeek (date : Date) : DayOfWeek :=
  sorry

/-- Calculates the date after adding a number of days to a given date -/
def addDays (date : Date) (days : Nat) : Date :=
  sorry

/-- The statement of the theorem -/
theorem battle_treaty_day :
  let battleStart : Date := ⟨1800, 3, 3⟩
  let battleStartDay : DayOfWeek := DayOfWeek.Monday
  let treatyDate : Date := addDays battleStart 1000
  dayOfWeek treatyDate = DayOfWeek.Thursday :=
sorry

end battle_treaty_day_l1_199


namespace greatest_common_multiple_9_15_under_100_l1_111

def is_common_multiple (m n k : ℕ) : Prop := k % m = 0 ∧ k % n = 0

theorem greatest_common_multiple_9_15_under_100 :
  ∃ (k : ℕ), k < 100 ∧ is_common_multiple 9 15 k ∧
  ∀ (m : ℕ), m < 100 → is_common_multiple 9 15 m → m ≤ k :=
by
  use 90
  sorry

end greatest_common_multiple_9_15_under_100_l1_111


namespace radical_conjugate_sum_product_l1_136

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = -6 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 1 → 
  a + b = 5 := by
  sorry

end radical_conjugate_sum_product_l1_136


namespace sqrt_of_sqrt_81_l1_119

theorem sqrt_of_sqrt_81 : ∃ (x : ℝ), x^2 = Real.sqrt 81 ↔ x = 3 ∨ x = -3 := by sorry

end sqrt_of_sqrt_81_l1_119


namespace sector_angle_l1_126

/-- Given a sector with area 1 and perimeter 4, its central angle in radians is 2 -/
theorem sector_angle (r : ℝ) (α : ℝ) 
  (h_area : (1/2) * α * r^2 = 1) 
  (h_perim : 2*r + α*r = 4) : 
  α = 2 := by sorry

end sector_angle_l1_126


namespace part_one_part_two_l1_117

-- Define α
variable (α : Real)

-- Given condition
axiom tan_alpha : Real.tan α = 3

-- Theorem for part (I)
theorem part_one : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5/7 := by
  sorry

-- Theorem for part (II)
theorem part_two :
  (Real.sin α + Real.cos α)^2 = 8/5 := by
  sorry

end part_one_part_two_l1_117


namespace womens_doubles_handshakes_l1_197

/-- The number of handshakes in a women's doubles tennis tournament -/
theorem womens_doubles_handshakes (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 2) : 
  let total_women := n * k
  let handshakes_per_woman := total_women - k
  (total_women * handshakes_per_woman) / 2 = 24 := by
sorry

end womens_doubles_handshakes_l1_197


namespace smallest_four_digit_with_product_512_l1_109

def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

def digit_product (n : ℕ) : ℕ :=
  (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

theorem smallest_four_digit_with_product_512 :
  ∀ n : ℕ, is_four_digit n → digit_product n = 512 → n ≥ 1888 :=
by sorry

end smallest_four_digit_with_product_512_l1_109


namespace max_value_when_a_zero_range_of_a_for_nonpositive_f_l1_149

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * x) / Real.exp x + a * Real.log (x + 1)

/-- Theorem for the maximum value of f when a = 0 -/
theorem max_value_when_a_zero :
  ∃ (x : ℝ), ∀ (y : ℝ), f 0 y ≤ f 0 x ∧ f 0 x = 2 / Real.exp 1 :=
sorry

/-- Theorem for the range of a when f(x) ≤ 0 for x ∈ [0, +∞) -/
theorem range_of_a_for_nonpositive_f :
  ∀ (a : ℝ), (∀ (x : ℝ), x ≥ 0 → f a x ≤ 0) ↔ a ≤ -2 :=
sorry

end max_value_when_a_zero_range_of_a_for_nonpositive_f_l1_149


namespace other_triangle_rectangle_area_ratio_l1_179

/-- Represents a right triangle with a point on its hypotenuse -/
structure RightTriangleWithPoint where
  /-- Length of the side of the rectangle along the hypotenuse -/
  side_along_hypotenuse : ℝ
  /-- Length of the side of the rectangle perpendicular to the hypotenuse -/
  side_perpendicular : ℝ
  /-- Ratio of the area of one small right triangle to the area of the rectangle -/
  area_ratio : ℝ
  /-- Condition: The side along the hypotenuse has length 1 -/
  hypotenuse_side_length : side_along_hypotenuse = 1
  /-- Condition: The area of one small right triangle is n times the area of the rectangle -/
  area_ratio_condition : area_ratio > 0

/-- Theorem: The ratio of the area of the other small right triangle to the area of the rectangle -/
theorem other_triangle_rectangle_area_ratio 
  (t : RightTriangleWithPoint) : 
  ∃ (ratio : ℝ), ratio = t.side_perpendicular / t.area_ratio := by
  sorry

end other_triangle_rectangle_area_ratio_l1_179


namespace exists_k_for_all_m_unique_k_characterization_l1_174

/-- The number of elements in {k+1, k+2, ..., 2k} with exactly 3 ones in binary representation -/
def f (k : ℕ+) : ℕ := sorry

/-- There exists a k for every m such that f(k) = m -/
theorem exists_k_for_all_m (m : ℕ+) : ∃ k : ℕ+, f k = m := by sorry

/-- Characterization of m for which there's exactly one k satisfying f(k) = m -/
theorem unique_k_characterization (m : ℕ+) : 
  (∃! k : ℕ+, f k = m) ↔ ∃ n : ℕ, n ≥ 2 ∧ m = n * (n - 1) / 2 + 1 := by sorry

end exists_k_for_all_m_unique_k_characterization_l1_174


namespace point_in_second_quadrant_l1_195

def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

theorem point_in_second_quadrant : 
  let z : ℂ := (complex_number 1 2) / (complex_number 1 (-1))
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end point_in_second_quadrant_l1_195


namespace largest_tray_size_l1_106

theorem largest_tray_size (tim_sweets peter_sweets : ℕ) 
  (h1 : tim_sweets = 36) 
  (h2 : peter_sweets = 44) : 
  Nat.gcd tim_sweets peter_sweets = 4 := by
  sorry

end largest_tray_size_l1_106


namespace coin_order_l1_114

/-- Represents the relative position of coins -/
inductive Position
| Above
| Below
| Same
| Unknown

/-- Represents a coin -/
inductive Coin
| F
| A
| B
| C
| D
| E

/-- Defines the relative position between two coins -/
def relative_position (c1 c2 : Coin) : Position := sorry

/-- Defines whether a coin is directly above another -/
def is_directly_above (c1 c2 : Coin) : Prop := 
  relative_position c1 c2 = Position.Above ∧ 
  ∀ c, c ≠ c1 ∧ c ≠ c2 → relative_position c1 c = Position.Above ∨ relative_position c c2 = Position.Above

/-- The main theorem to prove -/
theorem coin_order :
  (∀ c, c ≠ Coin.F → relative_position Coin.F c = Position.Above) ∧
  (is_directly_above Coin.A Coin.B) ∧
  (is_directly_above Coin.A Coin.C) ∧
  (relative_position Coin.A Coin.D = Position.Unknown) ∧
  (relative_position Coin.A Coin.E = Position.Unknown) ∧
  (is_directly_above Coin.D Coin.E) ∧
  (is_directly_above Coin.E Coin.B) ∧
  (∀ c, c ≠ Coin.F ∧ c ≠ Coin.A → relative_position c Coin.C = Position.Below ∨ relative_position c Coin.C = Position.Unknown) →
  (relative_position Coin.F Coin.A = Position.Above) ∧
  (relative_position Coin.A Coin.D = Position.Above) ∧
  (relative_position Coin.D Coin.E = Position.Above) ∧
  (relative_position Coin.E Coin.C = Position.Above ∨ relative_position Coin.E Coin.C = Position.Unknown) ∧
  (relative_position Coin.C Coin.B = Position.Above) := by
  sorry

end coin_order_l1_114


namespace grid_division_theorem_l1_191

/-- A grid division is valid if it satisfies the given conditions -/
def is_valid_division (n : ℕ) : Prop :=
  ∃ (m : ℕ), n^2 = 4 + 5*m ∧ 
  ∃ (square_pos : ℕ × ℕ), square_pos.1 < n ∧ square_pos.2 < n-1 ∧
  (square_pos.1 = 0 ∨ square_pos.1 = n-2 ∨ square_pos.2 = 0 ∨ square_pos.2 = n-2)

/-- The main theorem stating the condition for valid grid division -/
theorem grid_division_theorem (n : ℕ) : 
  is_valid_division n ↔ n % 5 = 2 :=
sorry

end grid_division_theorem_l1_191


namespace intersection_of_A_and_B_l1_137

-- Define the sets A and B
def A : Set ℝ := {x | -5 < x ∧ x < 2}
def B : Set ℝ := {x | |x| < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -3 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l1_137


namespace inequality_region_l1_138

theorem inequality_region (x y : ℝ) : 
  ((x*y + 1) / (x + y))^2 < 1 ↔ 
  ((-1 < x ∧ x < 1 ∧ (y < -1 ∨ y > 1)) ∨ ((x < -1 ∨ x > 1) ∧ -1 < y ∧ y < 1)) :=
by sorry

end inequality_region_l1_138


namespace two_point_distribution_properties_l1_177

/-- A random variable following a two-point distribution -/
structure TwoPointDistribution where
  X : ℝ → ℝ
  prob_zero : ℝ
  prob_one : ℝ
  sum_to_one : prob_zero + prob_one = 1
  only_two_points : ∀ x, X x ≠ 0 → X x = 1

/-- Expected value of a two-point distribution -/
def expected_value (dist : TwoPointDistribution) : ℝ :=
  0 * dist.prob_zero + 1 * dist.prob_one

/-- Variance of a two-point distribution -/
def variance (dist : TwoPointDistribution) : ℝ :=
  dist.prob_zero * (0 - expected_value dist)^2 + 
  dist.prob_one * (1 - expected_value dist)^2

/-- Theorem: Expected value and variance for a specific two-point distribution -/
theorem two_point_distribution_properties (dist : TwoPointDistribution)
  (h : dist.prob_zero = 1/4) :
  expected_value dist = 3/4 ∧ variance dist = 3/16 := by
  sorry

end two_point_distribution_properties_l1_177


namespace binary_1101_equals_13_l1_162

/-- Represents a binary digit (0 or 1) -/
inductive BinaryDigit
| zero : BinaryDigit
| one : BinaryDigit

/-- Represents a binary number as a list of binary digits -/
def BinaryNumber := List BinaryDigit

/-- Converts a binary number to its decimal equivalent -/
def binaryToDecimal (bin : BinaryNumber) : ℕ :=
  bin.enum.foldl (fun acc (i, digit) =>
    acc + match digit with
      | BinaryDigit.zero => 0
      | BinaryDigit.one => 2^i
  ) 0

/-- The binary representation of 1101 -/
def bin1101 : BinaryNumber :=
  [BinaryDigit.one, BinaryDigit.one, BinaryDigit.zero, BinaryDigit.one]

theorem binary_1101_equals_13 :
  binaryToDecimal bin1101 = 13 := by
  sorry

end binary_1101_equals_13_l1_162


namespace total_frogs_is_18_l1_135

/-- The number of frogs inside the pond -/
def frogs_inside : ℕ := 12

/-- The number of frogs outside the pond -/
def frogs_outside : ℕ := 6

/-- The total number of frogs -/
def total_frogs : ℕ := frogs_inside + frogs_outside

/-- Theorem stating that the total number of frogs is 18 -/
theorem total_frogs_is_18 : total_frogs = 18 := by sorry

end total_frogs_is_18_l1_135


namespace soap_scrap_parts_l1_130

/-- The number of parts used to manufacture one soap -/
def soap_parts : ℕ := 11

/-- The total number of scraps at the end of the day -/
def total_scraps : ℕ := 251

/-- The number of additional soaps that can be manufactured from the scraps -/
def additional_soaps : ℕ := 25

/-- The number of scrap parts obtained for making one soap -/
def scrap_parts_per_soap : ℕ := 10

theorem soap_scrap_parts :
  scrap_parts_per_soap * additional_soaps = total_scraps ∧
  scrap_parts_per_soap < soap_parts :=
by sorry

end soap_scrap_parts_l1_130


namespace buckingham_palace_visitors_l1_128

/-- The number of visitors to Buckingham Palace on different days --/
structure PalaceVisitors where
  dayOfVisit : ℕ
  previousDay : ℕ
  twoDaysPrior : ℕ

/-- Calculates the difference in visitors between the day of visit and the sum of the previous two days --/
def visitorDifference (v : PalaceVisitors) : ℕ :=
  v.dayOfVisit - (v.previousDay + v.twoDaysPrior)

/-- Theorem stating the difference in visitors for the given data --/
theorem buckingham_palace_visitors :
  ∃ (v : PalaceVisitors),
    v.dayOfVisit = 8333 ∧
    v.previousDay = 3500 ∧
    v.twoDaysPrior = 2500 ∧
    visitorDifference v = 2333 := by
  sorry

end buckingham_palace_visitors_l1_128


namespace overall_gain_percentage_l1_113

theorem overall_gain_percentage (cost_A cost_B cost_C gain_A gain_B gain_C : ℚ) :
  cost_A = 700 ∧ cost_B = 500 ∧ cost_C = 300 ∧
  gain_A = 70 ∧ gain_B = 50 ∧ gain_C = 30 →
  (gain_A + gain_B + gain_C) / (cost_A + cost_B + cost_C) * 100 = 10 := by
  sorry

end overall_gain_percentage_l1_113


namespace complex_root_cubic_equation_l1_187

theorem complex_root_cubic_equation 
  (a b q r : ℝ) 
  (h_b : b ≠ 0) 
  (h_root : ∃ (z : ℂ), z^3 + q * z + r = 0 ∧ z = a + b * Complex.I) :
  q = b^2 - 3 * a^2 := by
sorry

end complex_root_cubic_equation_l1_187


namespace pool_filling_rate_l1_180

/-- Given a pool filled by four hoses, this theorem proves the rate of two unknown hoses. -/
theorem pool_filling_rate 
  (pool_volume : ℝ) 
  (fill_time : ℝ) 
  (known_hose_rate : ℝ) 
  (h_volume : pool_volume = 15000)
  (h_time : fill_time = 25 * 60)  -- Convert hours to minutes
  (h_known_rate : known_hose_rate = 2)
  : ∃ (unknown_hose_rate : ℝ), 
    2 * known_hose_rate + 2 * unknown_hose_rate = pool_volume / fill_time ∧ 
    unknown_hose_rate = 3 :=
by sorry

end pool_filling_rate_l1_180


namespace max_value_abs_sum_l1_116

theorem max_value_abs_sum (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) : 
  ∃ (m : ℝ), m = 5 ∧ ∀ x, 0 ≤ x ∧ x ≤ 4 → |x - 2| + |3 - x| ≤ m :=
by sorry

end max_value_abs_sum_l1_116


namespace typing_speed_ratio_l1_154

/-- Represents the typing speeds of Tim and Tom -/
structure TypingSpeed where
  tim : ℝ
  tom : ℝ

/-- The total pages typed by Tim and Tom in one hour -/
def totalPages (speed : TypingSpeed) : ℝ := speed.tim + speed.tom

/-- The total pages typed when Tom increases his speed by 25% -/
def increasedTotalPages (speed : TypingSpeed) : ℝ := speed.tim + 1.25 * speed.tom

theorem typing_speed_ratio 
  (speed : TypingSpeed) 
  (h1 : totalPages speed = 12)
  (h2 : increasedTotalPages speed = 14) :
  speed.tom / speed.tim = 2 := by
  sorry

#check typing_speed_ratio

end typing_speed_ratio_l1_154


namespace least_integer_square_75_more_than_double_l1_198

theorem least_integer_square_75_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 75 ∧ ∀ y : ℤ, y^2 = 2*y + 75 → x ≤ y :=
by sorry

end least_integer_square_75_more_than_double_l1_198


namespace shift_down_three_units_l1_156

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := 2 * x

theorem shift_down_three_units (x : ℝ) : f x - 3 = g x := by
  sorry

end shift_down_three_units_l1_156


namespace line_y_coordinate_l1_146

/-- Given a line in a rectangular coordinate system passing through points (-2, y), (10, 3),
    and having an x-intercept of 4, prove that the y-coordinate of the point with x-coordinate -2 is -3. -/
theorem line_y_coordinate (y : ℝ) : 
  ∃ (m b : ℝ), 
    (∀ x, y = m * x + b) ∧  -- Line equation
    (y = m * (-2) + b) ∧    -- Line passes through (-2, y)
    (3 = m * 10 + b) ∧      -- Line passes through (10, 3)
    (0 = m * 4 + b) →       -- Line has x-intercept at 4
  y = -3 := by sorry

end line_y_coordinate_l1_146


namespace largest_number_with_given_hcf_lcm_factors_l1_142

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_number_with_given_hcf_lcm_factors 
  (a b : ℕ) 
  (hcf_prime : is_prime 31) 
  (hcf_val : Nat.gcd a b = 31) 
  (lcm_factors : Nat.lcm a b = 31 * 13 * 14 * 17) :
  max a b = 95914 := by
  sorry

end largest_number_with_given_hcf_lcm_factors_l1_142


namespace leo_commute_cost_l1_164

theorem leo_commute_cost (total_cost : ℕ) (working_days : ℕ) (trips_per_day : ℕ) 
  (h1 : total_cost = 960)
  (h2 : working_days = 20)
  (h3 : trips_per_day = 2) :
  total_cost / (working_days * trips_per_day) = 24 := by
sorry

end leo_commute_cost_l1_164


namespace projection_matrix_values_l1_141

/-- A 2x2 matrix is a projection matrix if and only if P² = P -/
def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P ^ 2 = P

/-- The specific 2x2 matrix we're working with -/
def P (b d : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![b, 12/25; d, 13/25]

/-- The theorem stating the values of b and d for the projection matrix -/
theorem projection_matrix_values :
  ∀ b d : ℚ, is_projection_matrix (P b d) → b = 37/50 ∧ d = 19/50 := by
  sorry

end projection_matrix_values_l1_141


namespace mean_temperature_and_humidity_l1_190

def temperatures : List Int := [-6, -2, -2, -3, 2, 4, 3]
def humidities : List Int := [70, 65, 65, 72, 80, 75, 77]

theorem mean_temperature_and_humidity :
  (temperatures.sum : ℚ) / temperatures.length = -4/7 ∧
  (humidities.sum : ℚ) / humidities.length = 72 := by
  sorry

end mean_temperature_and_humidity_l1_190


namespace harry_joe_fish_ratio_l1_122

/-- Proves that Harry has 4 times as many fish as Joe given the conditions -/
theorem harry_joe_fish_ratio :
  ∀ (harry joe sam : ℕ),
  joe = 8 * sam →
  sam = 7 →
  harry = 224 →
  harry = 4 * joe :=
by
  sorry

end harry_joe_fish_ratio_l1_122


namespace smallest_value_of_a_l1_168

def a (n : ℕ+) : ℤ := 2 * n.val ^ 2 - 10 * n.val + 3

theorem smallest_value_of_a (n : ℕ+) :
  a n ≥ a 2 ∧ a n ≥ a 3 ∧ (a 2 = a 3) :=
sorry

end smallest_value_of_a_l1_168


namespace length_BC_l1_163

-- Define the centers and radii of the circles
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def radius_A : ℝ := 7
def radius_B : ℝ := 4

-- Define the distance between centers A and B
def AB : ℝ := radius_A + radius_B

-- Define point C
def C : ℝ × ℝ := sorry

-- Define the distance AC
def AC : ℝ := AB + 2

-- Theorem to prove
theorem length_BC : ∃ (BC : ℝ), BC = 52 / 7 := by
  sorry

end length_BC_l1_163


namespace complex_fraction_simplification_l1_157

theorem complex_fraction_simplification :
  (I : ℂ) / (Real.sqrt 7 + 3 * I) = (3 : ℂ) / 16 + (Real.sqrt 7 / 16) * I :=
by sorry

end complex_fraction_simplification_l1_157


namespace domain_f_l1_152

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_plus_one : Set ℝ := Set.Icc (-2) 3

-- Theorem statement
theorem domain_f (h : ∀ x, f (x + 1) ∈ domain_f_plus_one ↔ x ∈ Set.Icc (-2) 3) :
  ∀ x, f x ∈ Set.Icc (-3) 2 ↔ x ∈ Set.Icc (-3) 2 :=
sorry

end domain_f_l1_152


namespace fence_building_time_l1_143

/-- The time in minutes to build one fence -/
def time_per_fence : ℕ := 30

/-- The number of fences initially built -/
def initial_fences : ℕ := 10

/-- The total number of fences after additional work -/
def total_fences : ℕ := 26

/-- The additional work time in hours -/
def additional_work_time : ℕ := 8

/-- Theorem stating that the time per fence is 30 minutes -/
theorem fence_building_time :
  time_per_fence = 30 ∧
  initial_fences = 10 ∧
  total_fences = 26 ∧
  additional_work_time = 8 ∧
  (total_fences - initial_fences) * time_per_fence = additional_work_time * 60 :=
by sorry

end fence_building_time_l1_143


namespace arccos_difference_equals_negative_pi_sixth_l1_108

theorem arccos_difference_equals_negative_pi_sixth : 
  Real.arccos ((Real.sqrt 6 + 1) / (2 * Real.sqrt 3)) - Real.arccos (Real.sqrt (2/3)) = -π/6 := by
  sorry

end arccos_difference_equals_negative_pi_sixth_l1_108


namespace function_decreasing_interval_l1_144

/-- Given a function f(x) = kx³ - 3(k+1)x² - k² + 1 where k > 0,
    if the decreasing interval of f(x) is (0, 4), then k = 1. -/
theorem function_decreasing_interval (k : ℝ) (h₁ : k > 0) :
  let f : ℝ → ℝ := λ x => k * x^3 - 3 * (k + 1) * x^2 - k^2 + 1
  (∀ x ∈ Set.Ioo 0 4, ∀ y ∈ Set.Ioo 0 4, x < y → f x > f y) →
  k = 1 := by
  sorry

end function_decreasing_interval_l1_144


namespace vowel_word_count_l1_184

/-- The number of vowels available (including Y) -/
def num_vowels : ℕ := 6

/-- The number of times each vowel appears, except A -/
def vowel_count : ℕ := 5

/-- The number of times A appears -/
def a_count : ℕ := 3

/-- The length of each word -/
def word_length : ℕ := 5

/-- The number of five-letter words that can be formed using vowels A, E, I, O, U, Y,
    where each vowel appears 5 times except A which appears 3 times -/
def num_words : ℕ := 7750

theorem vowel_word_count : 
  (vowel_count ^ word_length) + 
  (word_length.choose 1 * vowel_count ^ (word_length - 1)) +
  (word_length.choose 2 * vowel_count ^ (word_length - 2)) +
  (word_length.choose 3 * vowel_count ^ (word_length - 3)) = num_words :=
sorry

end vowel_word_count_l1_184


namespace wallpaper_removal_time_l1_147

/-- Time to remove wallpaper from one wall in hours -/
def time_per_wall : ℕ := 2

/-- Number of walls in the dining room -/
def dining_room_walls : ℕ := 4

/-- Number of walls in the living room -/
def living_room_walls : ℕ := 4

/-- Number of walls already completed in the dining room -/
def completed_walls : ℕ := 1

/-- Calculates the total time to remove remaining wallpaper -/
def total_time : ℕ :=
  time_per_wall * (dining_room_walls - completed_walls) +
  time_per_wall * living_room_walls

theorem wallpaper_removal_time : total_time = 14 := by
  sorry

end wallpaper_removal_time_l1_147


namespace eight_divided_by_repeating_third_l1_145

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1 / 3

/-- The result of 8 divided by the repeating decimal 0.333... --/
def result : ℚ := 8 / repeating_third

theorem eight_divided_by_repeating_third :
  result = 24 := by sorry

end eight_divided_by_repeating_third_l1_145


namespace special_polynomial_characterization_l1_150

/-- A polynomial that satisfies the given functional equation -/
structure SpecialPolynomial where
  P : Polynomial ℝ
  eq : ∀ (X : ℝ), 16 * (P.eval (X^2)) = (P.eval (2*X))^2

/-- The characterization of polynomials satisfying the functional equation -/
theorem special_polynomial_characterization (sp : SpecialPolynomial) :
  ∃ (n : ℕ), sp.P = Polynomial.monomial n (16 * (1/4)^n) := by
  sorry

end special_polynomial_characterization_l1_150


namespace not_necessarily_similar_remaining_parts_l1_183

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define a function to split a triangle into two smaller triangles
def split (t : Triangle) : Triangle × Triangle := sorry

theorem not_necessarily_similar_remaining_parts 
  (T1 T2 : Triangle) 
  (h_similar : similar T1 T2) 
  (T1_split : Triangle × Triangle) 
  (T2_split : Triangle × Triangle)
  (h_T1_split : T1_split = split T1)
  (h_T2_split : T2_split = split T2)
  (h_part_similar : similar T1_split.1 T2_split.1) :
  ¬ (∀ (T1 T2 : Triangle) (h_similar : similar T1 T2) 
      (T1_split T2_split : Triangle × Triangle)
      (h_T1_split : T1_split = split T1)
      (h_T2_split : T2_split = split T2)
      (h_part_similar : similar T1_split.1 T2_split.1),
    similar T1_split.2 T2_split.2) :=
by sorry

end not_necessarily_similar_remaining_parts_l1_183


namespace middle_building_height_l1_159

/-- The height of the middle building in feet -/
def middle_height : ℝ := sorry

/-- The height of the left building in feet -/
def left_height : ℝ := 0.8 * middle_height

/-- The height of the right building in feet -/
def right_height : ℝ := middle_height + left_height - 20

/-- The total height of all three buildings in feet -/
def total_height : ℝ := 340

theorem middle_building_height :
  middle_height + left_height + right_height = total_height →
  middle_height = 340 / 5.2 :=
by sorry

end middle_building_height_l1_159


namespace cars_meeting_time_l1_181

/-- Given two cars driving toward each other, prove that they meet in 4 hours -/
theorem cars_meeting_time (speed1 : ℝ) (speed2 : ℝ) (distance : ℝ) : 
  speed1 = 100 →
  speed1 = 1.25 * speed2 →
  distance = 720 →
  distance / (speed1 + speed2) = 4 := by
sorry


end cars_meeting_time_l1_181


namespace sum_of_numbers_l1_148

theorem sum_of_numbers (t a : ℝ) 
  (h1 : t = a + 12) 
  (h2 : t^2 + a^2 = 169/2) 
  (h3 : t^4 = a^4 + 5070) : 
  t + a = 5 := by
  sorry

end sum_of_numbers_l1_148


namespace gcd_consecutive_triple_product_l1_121

theorem gcd_consecutive_triple_product (i : ℕ) (h : i ≥ 1) :
  ∃ (g : ℕ), g = Nat.gcd i ((i + 1) * (i + 2)) ∧ g = 6 :=
sorry

end gcd_consecutive_triple_product_l1_121


namespace model_A_better_fit_l1_193

-- Define the R² values for models A and B
def R_squared_A : ℝ := 0.96
def R_squared_B : ℝ := 0.85

-- Define a function to compare fitting effects based on R²
def better_fit (r1 r2 : ℝ) : Prop := r1 > r2

-- Theorem statement
theorem model_A_better_fit :
  better_fit R_squared_A R_squared_B :=
by sorry

end model_A_better_fit_l1_193


namespace sine_of_supplementary_angles_l1_188

theorem sine_of_supplementary_angles (VPQ VPS : Real) 
  (h1 : VPS + VPQ = Real.pi)  -- Supplementary angles
  (h2 : Real.sin VPQ = 3/5) : 
  Real.sin VPS = 3/5 := by
  sorry

end sine_of_supplementary_angles_l1_188


namespace product_parity_two_numbers_product_even_three_numbers_l1_134

-- Definition for two numbers
def sum_is_even (a b : ℤ) : Prop := ∃ k : ℤ, a + b = 2 * k

-- Theorem for two numbers
theorem product_parity_two_numbers (a b : ℤ) (h : sum_is_even a b) :
  (∃ m : ℤ, a * b = 2 * m) ∨ (∃ n : ℤ, a * b = 2 * n + 1) :=
sorry

-- Theorem for three numbers
theorem product_even_three_numbers (a b c : ℤ) :
  ∃ k : ℤ, a * b * c = 2 * k :=
sorry

end product_parity_two_numbers_product_even_three_numbers_l1_134


namespace shortest_distance_between_circles_l1_186

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles :
  let circle1 := (fun (x y : ℝ) => x^2 - 2*x + y^2 + 6*y + 2 = 0)
  let circle2 := (fun (x y : ℝ) => x^2 + 6*x + y^2 - 2*y + 9 = 0)
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 - 1 ∧
  ∀ (p1 p2 : ℝ × ℝ),
    circle1 p1.1 p1.2 → circle2 p2.1 p2.2 →
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) ≥ d :=
by sorry

end shortest_distance_between_circles_l1_186


namespace trivia_team_points_per_member_l1_175

theorem trivia_team_points_per_member 
  (total_members : ℕ) 
  (absent_members : ℕ) 
  (total_points : ℕ) 
  (h1 : total_members = 9) 
  (h2 : absent_members = 3) 
  (h3 : total_points = 12) : 
  (total_points / (total_members - absent_members) : ℚ) = 2 := by
sorry

#eval (12 : ℚ) / 6  -- This should evaluate to 2

end trivia_team_points_per_member_l1_175


namespace total_earnings_is_4350_l1_140

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  invest_ratio_A : ℕ
  invest_ratio_B : ℕ
  invest_ratio_C : ℕ
  return_ratio_A : ℕ
  return_ratio_B : ℕ
  return_ratio_C : ℕ

/-- Calculates the total earnings given investment data and the earnings difference between B and A -/
def calculate_total_earnings (data : InvestmentData) (earnings_diff_B_A : ℕ) : ℕ :=
  let earnings_A := data.invest_ratio_A * data.return_ratio_A
  let earnings_B := data.invest_ratio_B * data.return_ratio_B
  let earnings_C := data.invest_ratio_C * data.return_ratio_C
  let total_ratio := earnings_A + earnings_B + earnings_C
  (total_ratio * earnings_diff_B_A) / (earnings_B - earnings_A)

/-- Theorem stating that given the specific investment ratios and conditions, the total earnings is 4350 -/
theorem total_earnings_is_4350 : 
  let data : InvestmentData := {
    invest_ratio_A := 3,
    invest_ratio_B := 4,
    invest_ratio_C := 5,
    return_ratio_A := 6,
    return_ratio_B := 5,
    return_ratio_C := 4
  }
  calculate_total_earnings data 150 = 4350 := by
  sorry

end total_earnings_is_4350_l1_140


namespace milk_fraction_problem_l1_131

theorem milk_fraction_problem (V : ℝ) (h : V > 0) :
  let x := (3 : ℝ) / 5
  let second_cup_milk := (4 : ℝ) / 5 * V
  let second_cup_water := V - second_cup_milk
  let total_milk := x * V + second_cup_milk
  let total_water := (1 - x) * V + second_cup_water
  (total_water / total_milk = 3 / 7) → x = 3 / 5 := by
sorry

end milk_fraction_problem_l1_131


namespace rowing_time_ratio_l1_161

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 given boat and stream speeds -/
theorem rowing_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 60) 
  (h2 : stream_speed = 20) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
  sorry

#check rowing_time_ratio

end rowing_time_ratio_l1_161


namespace smallest_integer_fraction_six_is_solution_six_is_smallest_l1_178

theorem smallest_integer_fraction (x : ℤ) : x > 5 ∧ (x^2 - 4*x + 13) % (x - 5) = 0 → x ≥ 6 := by
  sorry

theorem six_is_solution : (6^2 - 4*6 + 13) % (6 - 5) = 0 := by
  sorry

theorem six_is_smallest : ∀ (y : ℤ), y > 5 ∧ y < 6 → (y^2 - 4*y + 13) % (y - 5) ≠ 0 := by
  sorry

end smallest_integer_fraction_six_is_solution_six_is_smallest_l1_178


namespace product_of_roots_rational_l1_103

-- Define the polynomial
def polynomial (a b c d e : ℤ) (z : ℂ) : ℂ :=
  a * z^4 + b * z^3 + c * z^2 + d * z + e

-- Define the theorem
theorem product_of_roots_rational
  (a b c d e : ℤ)
  (r₁ r₂ r₃ r₄ : ℂ)
  (h₁ : a ≠ 0)
  (h₂ : polynomial a b c d e r₁ = 0)
  (h₃ : polynomial a b c d e r₂ = 0)
  (h₄ : polynomial a b c d e r₃ = 0)
  (h₅ : polynomial a b c d e r₄ = 0)
  (h₆ : ∃ q : ℚ, r₁ + r₂ = q)
  (h₇ : r₃ + r₄ ≠ r₁ + r₂)
  : ∃ q : ℚ, r₁ * r₂ = q :=
sorry

end product_of_roots_rational_l1_103


namespace binomial_1500_1_l1_160

theorem binomial_1500_1 : Nat.choose 1500 1 = 1500 := by
  sorry

end binomial_1500_1_l1_160


namespace polyhedron_sum_l1_125

/-- A convex polyhedron with triangular and hexagonal faces -/
structure ConvexPolyhedron where
  faces : ℕ
  triangles : ℕ
  hexagons : ℕ
  vertices : ℕ
  edges : ℕ
  T : ℕ  -- number of triangular faces meeting at each vertex
  H : ℕ  -- number of hexagonal faces meeting at each vertex
  faces_sum : faces = triangles + hexagons
  faces_20 : faces = 20
  edges_formula : edges = (3 * triangles + 6 * hexagons) / 2
  euler_formula : vertices - edges + faces = 2

/-- The theorem to be proved -/
theorem polyhedron_sum (p : ConvexPolyhedron) : 100 * p.H + 10 * p.T + p.vertices = 227 := by
  sorry

end polyhedron_sum_l1_125


namespace partnership_profit_l1_104

/-- 
Given a business partnership between Mary and Mike:
- Mary invests $700
- Mike invests $300
- 1/3 of profit is divided equally for efforts
- 2/3 of profit is divided in ratio of investments (7:3)
- Mary received $800 more than Mike

This theorem proves that the total profit P satisfies the equation:
[(P/6) + (7/10) * (2P/3)] - [(P/6) + (3/10) * (2P/3)] = 800
-/
theorem partnership_profit (P : ℝ) : 
  (P / 6 + 7 / 10 * (2 * P / 3)) - (P / 6 + 3 / 10 * (2 * P / 3)) = 800 → 
  P = 3000 := by
sorry


end partnership_profit_l1_104


namespace colored_pencils_count_l1_173

def number_of_packs : ℕ := 7
def pencils_per_pack : ℕ := 10
def difference : ℕ := 3

theorem colored_pencils_count :
  let total_pencils := number_of_packs * pencils_per_pack
  let colored_pencils := total_pencils + difference
  colored_pencils = 73 := by
  sorry

end colored_pencils_count_l1_173


namespace equation_solution_l1_172

theorem equation_solution :
  ∃! x : ℚ, (4 * x^2 + 3 * x + 1) / (x - 2) = 4 * x + 5 :=
by
  use -11/6
  sorry

end equation_solution_l1_172


namespace pencil_count_l1_158

theorem pencil_count (people notebooks_per_person pencil_multiplier : ℕ) 
  (h1 : people = 6)
  (h2 : notebooks_per_person = 9)
  (h3 : pencil_multiplier = 6) :
  people * notebooks_per_person * pencil_multiplier = 324 :=
by sorry

end pencil_count_l1_158


namespace repair_cost_is_correct_l1_153

/-- Calculates the total cost of car repair given the following conditions:
  * Two mechanics work on the car
  * First mechanic: $60/hour, 8 hours/day, 14 days
  * Second mechanic: $75/hour, 6 hours/day, 10 days
  * 15% discount on first mechanic's labor cost
  * 10% discount on second mechanic's labor cost
  * Parts cost: $3,200
  * 7% sales tax on final bill after discounts
-/
def totalRepairCost (
  mechanic1_rate : ℝ)
  (mechanic1_hours : ℝ)
  (mechanic1_days : ℝ)
  (mechanic2_rate : ℝ)
  (mechanic2_hours : ℝ)
  (mechanic2_days : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (parts_cost : ℝ)
  (sales_tax_rate : ℝ) : ℝ :=
  let mechanic1_cost := mechanic1_rate * mechanic1_hours * mechanic1_days
  let mechanic2_cost := mechanic2_rate * mechanic2_hours * mechanic2_days
  let discounted_mechanic1_cost := mechanic1_cost * (1 - discount1)
  let discounted_mechanic2_cost := mechanic2_cost * (1 - discount2)
  let total_before_tax := discounted_mechanic1_cost + discounted_mechanic2_cost + parts_cost
  total_before_tax * (1 + sales_tax_rate)

/-- Theorem stating that the total repair cost is $13,869.34 given the specific conditions -/
theorem repair_cost_is_correct :
  totalRepairCost 60 8 14 75 6 10 0.15 0.10 3200 0.07 = 13869.34 := by
  sorry

end repair_cost_is_correct_l1_153


namespace x_intercept_of_perpendicular_line_l1_107

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℚ := -l.y_intercept / l.slope

/-- Two lines are perpendicular if their slopes are negative reciprocals -/
def perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

theorem x_intercept_of_perpendicular_line (given_line perp_line : Line) :
  given_line.slope = -5/3 →
  perpendicular given_line perp_line →
  perp_line.y_intercept = -4 →
  x_intercept perp_line = 20/3 := by
  sorry

end x_intercept_of_perpendicular_line_l1_107


namespace greatest_divisor_with_remainders_l1_151

theorem greatest_divisor_with_remainders : Nat.gcd (1442 - 12) (1816 - 6) = 10 := by
  sorry

end greatest_divisor_with_remainders_l1_151


namespace function_extrema_condition_l1_155

def f (a x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

theorem function_extrema_condition (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ f a max ∧ f a min ≤ f a x) →
  (a < 1/3 ∧ a ≠ 0) :=
by sorry

end function_extrema_condition_l1_155


namespace four_distinct_roots_iff_q_16_l1_169

/-- The function f(x) = x^2 + 8x + q -/
def f (q : ℝ) (x : ℝ) : ℝ := x^2 + 8*x + q

/-- The composition of f with itself -/
def f_comp (q : ℝ) (x : ℝ) : ℝ := f q (f q x)

/-- The number of distinct real roots of f(f(x)) -/
noncomputable def num_distinct_roots (q : ℝ) : ℕ := sorry

theorem four_distinct_roots_iff_q_16 :
  ∀ q : ℝ, num_distinct_roots q = 4 ↔ q = 16 :=
sorry

end four_distinct_roots_iff_q_16_l1_169


namespace quadratic_real_roots_condition_l1_171

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + 2*m = 0) → m ≤ 9/8 := by
  sorry

end quadratic_real_roots_condition_l1_171


namespace count_four_digit_divisible_by_13_l1_102

theorem count_four_digit_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 9000)).card = 693 :=
by sorry

end count_four_digit_divisible_by_13_l1_102


namespace expression_evaluation_l1_118

theorem expression_evaluation : (3^2 - 5) / (0.08 * 7 + 2) = 1.5625 := by
  sorry

end expression_evaluation_l1_118


namespace boys_on_playground_l1_189

theorem boys_on_playground (total_children girls : ℕ) 
  (h1 : total_children = 62) 
  (h2 : girls = 35) : 
  total_children - girls = 27 := by
sorry

end boys_on_playground_l1_189


namespace sum_of_roots_l1_110

theorem sum_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y z : ℝ, x^2 - x - 34 = (x - y) * (x - z) ∧ y + z = 1 := by
  sorry

end sum_of_roots_l1_110


namespace day_before_yesterday_is_sunday_l1_192

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the previous day
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

theorem day_before_yesterday_is_sunday 
  (h : nextDay (nextDay DayOfWeek.Sunday) = DayOfWeek.Monday) : 
  prevDay (prevDay DayOfWeek.Sunday) = DayOfWeek.Sunday := by
  sorry


end day_before_yesterday_is_sunday_l1_192


namespace complex_division_fourth_quadrant_l1_132

theorem complex_division_fourth_quadrant : 
  let i : ℂ := Complex.I
  let z₁ : ℂ := 1 + i
  let z₂ : ℂ := 1 + 2*i
  (z₁ / z₂).re > 0 ∧ (z₁ / z₂).im < 0 :=
by sorry

end complex_division_fourth_quadrant_l1_132


namespace jean_trips_l1_100

theorem jean_trips (total : ℕ) (extra : ℕ) (h1 : total = 40) (h2 : extra = 6) :
  ∃ (bill : ℕ) (jean : ℕ), bill + jean = total ∧ jean = bill + extra ∧ jean = 23 :=
by sorry

end jean_trips_l1_100


namespace opposite_seven_is_nine_or_eleven_l1_166

def DieNumbers : Finset ℕ := {6, 7, 8, 9, 10, 11}

def isValidOpposite (n : ℕ) : Prop :=
  n ∈ DieNumbers ∧ n ≠ 7 ∧
  ∃ (a b c d e : ℕ),
    {a, b, c, d, e, 7} = DieNumbers ∧
    (a + b + c + d = 33 ∨ a + b + c + d = 35) ∧
    (e + 7 = 16 ∨ e + 7 = 17 ∨ e + 7 = 18)

theorem opposite_seven_is_nine_or_eleven :
  ∀ n, isValidOpposite n → n = 9 ∨ n = 11 := by sorry

end opposite_seven_is_nine_or_eleven_l1_166


namespace marble_remainder_l1_123

theorem marble_remainder (r p : ℕ) 
  (hr : r % 8 = 5) 
  (hp : p % 8 = 6) : 
  (r + p) % 8 = 3 := by sorry

end marble_remainder_l1_123


namespace ice_cream_flavors_count_l1_127

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ice cream flavors that can be created -/
def ice_cream_flavors : ℕ := distribute 5 4

theorem ice_cream_flavors_count : ice_cream_flavors = 56 := by
  sorry

end ice_cream_flavors_count_l1_127


namespace inequality_proof_l1_112

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c ≥ 1) :
  1 / (2 + a) + 1 / (2 + b) + 1 / (2 + c) ≤ 1 ∧
  (1 / (2 + a) + 1 / (2 + b) + 1 / (2 + c) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end inequality_proof_l1_112


namespace lcm_gcd_difference_even_nonnegative_l1_105

theorem lcm_gcd_difference_even_nonnegative (a b : ℕ+) :
  let n := Nat.lcm a b + Nat.gcd a b - a - b
  n % 2 = 0 ∧ n ≥ 0 := by sorry

end lcm_gcd_difference_even_nonnegative_l1_105


namespace ratio_problem_l1_124

theorem ratio_problem (x y : ℤ) : 
  (y = 4 * x) →  -- The two integers are in the ratio of 1 to 4
  (x + 12 = y) →  -- Adding 12 to the smaller number makes the ratio 1 to 1
  y = 16 :=  -- The larger integer is 16
by sorry

end ratio_problem_l1_124


namespace consecutive_integers_average_l1_101

theorem consecutive_integers_average (n : ℤ) : 
  (n * (n + 6) = 391) → 
  (((n + n + 1 + n + 2 + n + 3 + n + 4 + n + 5 + n + 6) : ℚ) / 7 = 20) :=
by sorry

end consecutive_integers_average_l1_101


namespace function_is_zero_l1_139

/-- A function from natural numbers to natural numbers. -/
def NatFunction := ℕ → ℕ

/-- The property that a function satisfies the given conditions. -/
def SatisfiesConditions (f : NatFunction) : Prop :=
  f 0 = 0 ∧
  ∀ x y : ℕ, x > y → f (x^2 - y^2) = f x * f y

/-- Theorem stating that any function satisfying the conditions must be identically zero. -/
theorem function_is_zero (f : NatFunction) (h : SatisfiesConditions f) : 
  ∀ x : ℕ, f x = 0 := by
  sorry


end function_is_zero_l1_139


namespace point_on_transformed_plane_l1_196

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

theorem point_on_transformed_plane (A : Point3D) (a : Plane) (k : ℝ) :
  A.x = 2 ∧ A.y = 5 ∧ A.z = 1 ∧
  a.a = 5 ∧ a.b = -2 ∧ a.c = 1 ∧ a.d = -3 ∧
  k = 1/3 →
  pointOnPlane A (transformPlane a k) :=
by sorry

end point_on_transformed_plane_l1_196


namespace potato_distribution_ratio_l1_194

/-- Represents the number of people who were served potatoes -/
def num_people : ℕ := 3

/-- Represents the number of potatoes each person received -/
def potatoes_per_person : ℕ := 8

/-- Represents the ratio of potatoes served to each person -/
def potato_ratio : List ℕ := [1, 1, 1]

/-- Theorem stating that the ratio of potatoes served to each person is 1:1:1 -/
theorem potato_distribution_ratio :
  (List.length potato_ratio = num_people) ∧
  (∀ n ∈ potato_ratio, n = 1) ∧
  (List.sum potato_ratio * potatoes_per_person = num_people * potatoes_per_person) := by
  sorry

end potato_distribution_ratio_l1_194


namespace polygon_sides_l1_133

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 3420 → ∃ n : ℕ, n = 21 ∧ sum_interior_angles = 180 * (n - 2) := by
  sorry

#check polygon_sides

end polygon_sides_l1_133


namespace partial_fraction_decomposition_l1_165

theorem partial_fraction_decomposition :
  let f (x : ℝ) := (2*x + 7) / (x^2 - 2*x - 63)
  let g (x : ℝ) := 25 / (16 * (x - 9)) + 7 / (16 * (x + 7))
  ∀ x : ℝ, x ≠ 9 ∧ x ≠ -7 → f x = g x :=
by
  sorry

end partial_fraction_decomposition_l1_165


namespace triangle_bisector_product_l1_176

/-- Given a triangle ABC with sides a, b, and c, internal angle bisectors of lengths fa, fb, and fc,
    and segments of internal angle bisectors on the circumcircle ta, tb, and tc,
    the product of the squares of the sides equals the product of all bisector lengths
    and their segments on the circumcircle. -/
theorem triangle_bisector_product (a b c fa fb fc ta tb tc : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (hfa : fa > 0) (hfb : fb > 0) (hfc : fc > 0)
    (hta : ta > 0) (htb : tb > 0) (htc : tc > 0) :
  a^2 * b^2 * c^2 = fa * fb * fc * ta * tb * tc := by
  sorry

end triangle_bisector_product_l1_176


namespace min_value_reciprocal_sum_l1_115

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : Real.log (a + b) = 0) :
  (1 / a + 1 / b) ≥ 4 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ Real.log (x + y) = 0 ∧ 1 / x + 1 / y = 4 := by
sorry

end min_value_reciprocal_sum_l1_115


namespace triangle_area_and_fixed_point_l1_185

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the area of the triangle
def triangle_area : ℝ := 8

-- Define the family of lines
def family_of_lines (m x y : ℝ) : Prop := m * x + y + m = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (-1, 0)

theorem triangle_area_and_fixed_point :
  (∀ x y, line_equation x y → 
    (x = 0 ∨ y = 0) → triangle_area = 8) ∧
  (∀ m x y, family_of_lines m x y → 
    (x, y) = fixed_point) :=
sorry

end triangle_area_and_fixed_point_l1_185


namespace largest_square_area_l1_167

-- Define the triangle and its properties
structure RightTriangle where
  XY : ℝ
  YZ : ℝ
  XZ : ℝ
  right_angle : XZ^2 = XY^2 + YZ^2
  hypotenuse_relation : XZ^2 = 2 * XY^2

-- Define the theorem
theorem largest_square_area
  (triangle : RightTriangle)
  (total_area : ℝ)
  (h_total_area : XY^2 + YZ^2 + XZ^2 = total_area)
  (h_total_area_value : total_area = 450) :
  XZ^2 = 225 := by
  sorry

#check largest_square_area

end largest_square_area_l1_167


namespace bus_riders_l1_120

theorem bus_riders (initial_riders : ℕ) : 
  (initial_riders + 40 - 60 = 2) → initial_riders = 22 := by
  sorry

end bus_riders_l1_120
