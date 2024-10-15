import Mathlib

namespace NUMINAMATH_CALUDE_dog_bones_problem_l1461_146128

theorem dog_bones_problem (buried_bones initial_bones final_bones : ℚ) : 
  buried_bones = 367.5 ∧ 
  final_bones = -860 ∧ 
  initial_bones - buried_bones = final_bones → 
  initial_bones = 367.5 := by
sorry


end NUMINAMATH_CALUDE_dog_bones_problem_l1461_146128


namespace NUMINAMATH_CALUDE_man_rowing_speed_l1461_146133

/-- Calculates the speed of a man rowing upstream given his speed in still water and downstream speed -/
def speed_upstream (speed_still : ℝ) (speed_downstream : ℝ) : ℝ :=
  2 * speed_still - speed_downstream

/-- Theorem stating that given a man's speed in still water is 32 kmph and his speed downstream is 42 kmph, his speed upstream is 22 kmph -/
theorem man_rowing_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 32) 
  (h2 : speed_downstream = 42) : 
  speed_upstream speed_still speed_downstream = 22 := by
  sorry

#eval speed_upstream 32 42

end NUMINAMATH_CALUDE_man_rowing_speed_l1461_146133


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l1461_146151

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_nonprimes (start : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start ∧ k < start + 6 → ¬(is_prime k)

theorem smallest_prime_after_six_nonprimes :
  ∃ n : ℕ, consecutive_nonprimes n ∧ is_prime (n + 6) ∧
  ∀ m : ℕ, m < n → ¬(consecutive_nonprimes m ∧ is_prime (m + 6)) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l1461_146151


namespace NUMINAMATH_CALUDE_polynomial_equality_l1461_146196

theorem polynomial_equality : 102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 100406401 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1461_146196


namespace NUMINAMATH_CALUDE_reflect_F_final_coords_l1461_146111

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def F : ℝ × ℝ := (1, 3)

theorem reflect_F_final_coords :
  (reflect_y_eq_x ∘ reflect_y ∘ reflect_x) F = (-3, -1) := by
  sorry

end NUMINAMATH_CALUDE_reflect_F_final_coords_l1461_146111


namespace NUMINAMATH_CALUDE_max_blocks_fit_l1461_146194

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- Represents the box and block dimensions -/
def box : Dimensions := ⟨40, 60, 80⟩
def block : Dimensions := ⟨20, 30, 40⟩

/-- Calculates the maximum number of blocks that can fit in the box based on volume -/
def max_blocks_by_volume : ℕ := volume box / volume block

/-- Checks if the blocks can be arranged to fit in the box -/
def can_arrange (n : ℕ) : Prop :=
  ∃ (l w h : ℕ), l * block.length ≤ box.length ∧
                 w * block.width ≤ box.width ∧
                 h * block.height ≤ box.height ∧
                 l * w * h = n

/-- The main theorem to prove -/
theorem max_blocks_fit :
  max_blocks_by_volume = 8 ∧ can_arrange 8 ∧
  ∀ n > 8, ¬can_arrange n :=
sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l1461_146194


namespace NUMINAMATH_CALUDE_unique_number_with_nine_divisors_and_special_property_l1461_146118

def has_exactly_nine_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 9

theorem unique_number_with_nine_divisors_and_special_property :
  ∃! n : ℕ, has_exactly_nine_divisors n ∧
  ∃ (a b c : ℕ), a ∣ n ∧ b ∣ n ∧ c ∣ n ∧
  a + b + c = 79 ∧ a * a = b * c :=
by
  use 441
  sorry

end NUMINAMATH_CALUDE_unique_number_with_nine_divisors_and_special_property_l1461_146118


namespace NUMINAMATH_CALUDE_number_of_women_at_event_l1461_146181

/-- Proves that the number of women at an event is 20, given the specified dancing conditions -/
theorem number_of_women_at_event (num_men : ℕ) (men_dance_count : ℕ) (women_dance_count : ℕ) 
  (h1 : num_men = 15)
  (h2 : men_dance_count = 4)
  (h3 : women_dance_count = 3)
  : (num_men * men_dance_count) / women_dance_count = 20 := by
  sorry

#check number_of_women_at_event

end NUMINAMATH_CALUDE_number_of_women_at_event_l1461_146181


namespace NUMINAMATH_CALUDE_correct_group_sizes_l1461_146167

/-- Represents the pricing structure and group information for a scenic area in Xi'an --/
structure ScenicAreaPricing where
  regularPrice : ℕ
  nonHolidayDiscount : ℚ
  holidayDiscountThreshold : ℕ
  holidayDiscount : ℚ
  totalPeople : ℕ
  totalCost : ℕ

/-- Calculates the cost for a group visiting on a non-holiday --/
def nonHolidayCost (pricing : ScenicAreaPricing) (people : ℕ) : ℚ :=
  pricing.regularPrice * (1 - pricing.nonHolidayDiscount) * people

/-- Calculates the cost for a group visiting on a holiday --/
def holidayCost (pricing : ScenicAreaPricing) (people : ℕ) : ℚ :=
  if people ≤ pricing.holidayDiscountThreshold then
    pricing.regularPrice * people
  else
    pricing.regularPrice * pricing.holidayDiscountThreshold +
    pricing.regularPrice * (1 - pricing.holidayDiscount) * (people - pricing.holidayDiscountThreshold)

/-- Theorem stating the correct number of people in each group --/
theorem correct_group_sizes (pricing : ScenicAreaPricing)
  (h1 : pricing.regularPrice = 50)
  (h2 : pricing.nonHolidayDiscount = 0.4)
  (h3 : pricing.holidayDiscountThreshold = 10)
  (h4 : pricing.holidayDiscount = 0.2)
  (h5 : pricing.totalPeople = 50)
  (h6 : pricing.totalCost = 1840) :
  ∃ (groupA groupB : ℕ),
    groupA + groupB = pricing.totalPeople ∧
    holidayCost pricing groupA + nonHolidayCost pricing groupB = pricing.totalCost ∧
    groupA = 24 ∧ groupB = 26 := by
  sorry


end NUMINAMATH_CALUDE_correct_group_sizes_l1461_146167


namespace NUMINAMATH_CALUDE_rationality_of_given_numbers_l1461_146137

theorem rationality_of_given_numbers :
  (∃ (a b : ℚ), a^2 = 4 ∧ b ≠ 0) ∧  -- √4 is rational
  (∀ (a b : ℚ), a^3 ≠ 0.5 * b^3) ∧  -- ∛0.5 is irrational
  (∃ (a b : ℚ), a^4 = 0.0625 * b^4 ∧ b ≠ 0) ∧  -- ∜0.0625 is rational
  (∃ (a b : ℚ), a^3 = -8 ∧ b^2 = 4 ∧ b ≠ 0) :=  -- ∛(-8) * √((0.25)^(-1)) is rational
by sorry

end NUMINAMATH_CALUDE_rationality_of_given_numbers_l1461_146137


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1461_146129

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the theorem
theorem solution_set_of_inequality
  (h_even : ∀ x, f (-x) = f x)  -- f is even
  (h_derivative : ∀ x, HasDerivAt f (f' x) x)  -- f' is the derivative of f
  (h_condition : ∀ x, x < 0 → x * f' x - f x > 0)  -- condition for x < 0
  (h_f_1 : f 1 = 0)  -- f(1) = 0
  : {x : ℝ | f x / x < 0} = {x | x < -1 ∨ (0 < x ∧ x < 1)} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1461_146129


namespace NUMINAMATH_CALUDE_part_one_part_two_l1461_146175

-- Part 1
theorem part_one (α : Real) (h : Real.tan α = 2) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = -3 := by sorry

-- Part 2
theorem part_two (α : Real) :
  (Real.sin (α - π/2) * Real.cos (π/2 - α) * Real.tan (π - α)) / 
  (Real.tan (π + α) * Real.sin (π + α)) = -Real.cos α := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1461_146175


namespace NUMINAMATH_CALUDE_tan_product_from_cos_sum_diff_l1461_146154

theorem tan_product_from_cos_sum_diff (α β : Real) 
  (h1 : Real.cos (α + β) = 1/5)
  (h2 : Real.cos (α - β) = 3/5) : 
  Real.tan α * Real.tan β = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_from_cos_sum_diff_l1461_146154


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1461_146174

/-- Given a square with perimeter 240 units divided into 4 congruent rectangles,
    where each rectangle's width is half the side length of the square,
    the perimeter of one rectangle is 180 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (rectangle_count : ℕ) 
  (h1 : square_perimeter = 240)
  (h2 : rectangle_count = 4) : ℝ :=
by
  -- Define the side length of the square
  let square_side := square_perimeter / 4

  -- Define the dimensions of each rectangle
  let rectangle_width := square_side / 2
  let rectangle_length := square_side

  -- Calculate the perimeter of one rectangle
  let rectangle_perimeter := 2 * (rectangle_width + rectangle_length)

  -- Prove that the rectangle_perimeter equals 180
  sorry

#check rectangle_perimeter

end NUMINAMATH_CALUDE_rectangle_perimeter_l1461_146174


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l1461_146143

theorem inequality_and_minimum_value {a b x : ℝ} (ha : a > 0) (hb : b > 0) (hx : 0 < x ∧ x < 1) :
  (a^2 / b + b^2 / a ≥ a + b) ∧
  (∀ y, y = (1 - x)^2 / x + x^2 / (1 - x) → y ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l1461_146143


namespace NUMINAMATH_CALUDE_no_fourth_power_sum_1599_l1461_146156

theorem no_fourth_power_sum_1599 :
  ¬ ∃ (s : Finset ℕ), (∀ n ∈ s, ∃ k, n = k^4) ∧ s.card ≤ 14 ∧ s.sum id = 1599 := by
  sorry

end NUMINAMATH_CALUDE_no_fourth_power_sum_1599_l1461_146156


namespace NUMINAMATH_CALUDE_line_point_at_t_4_l1461_146161

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  point_at : ℝ → ℝ × ℝ × ℝ

/-- Given a parameterized line with known points at t = 1 and t = -1, 
    prove that the point at t = 4 is (-27, 57, 27) -/
theorem line_point_at_t_4 
  (line : ParameterizedLine)
  (h1 : line.point_at 1 = (-3, 9, 12))
  (h2 : line.point_at (-1) = (4, -4, 2)) :
  line.point_at 4 = (-27, 57, 27) := by
sorry


end NUMINAMATH_CALUDE_line_point_at_t_4_l1461_146161


namespace NUMINAMATH_CALUDE_gcd_problem_l1461_146142

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 2 * 947 * k) : 
  Nat.gcd (Int.natAbs (3 * a^2 + 47 * a + 101)) (Int.natAbs (a + 19)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1461_146142


namespace NUMINAMATH_CALUDE_same_solution_value_l1461_146153

theorem same_solution_value (c : ℝ) : 
  (∃ x : ℝ, 3 * x + 5 = 2 ∧ c * x + 4 = 1) ↔ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_same_solution_value_l1461_146153


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1461_146120

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp x + 1

theorem derivative_f_at_zero :
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1461_146120


namespace NUMINAMATH_CALUDE_sum_abc_equals_three_l1461_146192

theorem sum_abc_equals_three (a b c : ℝ) 
  (eq1 : a^2 + 2*b = 7)
  (eq2 : b^2 - 2*c = -1)
  (eq3 : c^2 - 6*a = -17) : 
  a + b + c = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_three_l1461_146192


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1461_146179

theorem complex_magnitude_product : Complex.abs ((7 - 4*I) * (3 + 11*I)) = Real.sqrt 8450 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1461_146179


namespace NUMINAMATH_CALUDE_triangle_inequality_violation_l1461_146159

/-- Theorem: A triangle cannot be formed with side lengths 9, 4, and 3. -/
theorem triangle_inequality_violation (a b c : ℝ) 
  (ha : a = 9) (hb : b = 4) (hc : c = 3) : 
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_violation_l1461_146159


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1461_146172

theorem floor_negative_seven_fourths : ⌊(-7/4 : ℚ)⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1461_146172


namespace NUMINAMATH_CALUDE_target_number_is_294_l1461_146130

/-- Represents the list of numbers starting with digit 2 in increasing order -/
def digit2List : List ℕ := sorry

/-- Returns the nth digit in the concatenated representation of digit2List -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- The three-digit number formed by the 1498th, 1499th, and 1500th digits -/
def targetNumber : ℕ := 100 * (nthDigit 1498) + 10 * (nthDigit 1499) + (nthDigit 1500)

theorem target_number_is_294 : targetNumber = 294 := by sorry

end NUMINAMATH_CALUDE_target_number_is_294_l1461_146130


namespace NUMINAMATH_CALUDE_sean_cricket_theorem_l1461_146100

def sean_cricket_problem (total_days : ℕ) (total_minutes : ℕ) (indira_minutes : ℕ) : Prop :=
  let sean_total_minutes := total_minutes - indira_minutes
  sean_total_minutes / total_days = 50

theorem sean_cricket_theorem :
  sean_cricket_problem 14 1512 812 := by
  sorry

end NUMINAMATH_CALUDE_sean_cricket_theorem_l1461_146100


namespace NUMINAMATH_CALUDE_min_value_zero_implies_t_l1461_146117

/-- The function f(x) defined in the problem -/
def f (t : ℝ) (x : ℝ) : ℝ := 4 * x^4 - 6 * t * x^3 + (2 * t + 6) * x^2 - 3 * t * x + 1

/-- The theorem statement -/
theorem min_value_zero_implies_t (t : ℝ) :
  (∀ x > 0, f t x ≥ 0) ∧ 
  (∃ x > 0, f t x = 0) →
  t = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_zero_implies_t_l1461_146117


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1461_146134

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 64 + y^2 / 28 = 1

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 0

-- Define the standard form of a hyperbola
def hyperbola_standard_form (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Theorem statement
theorem hyperbola_equation (Γ : Set (ℝ × ℝ)) :
  (∃ F₁ F₂ : ℝ × ℝ, (∀ x y, ellipse x y ↔ (x - F₁.1)^2 + (y - F₁.2)^2 + (x - F₂.1)^2 + (y - F₂.2)^2 = 2 * Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) * Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2)) ∧
                     (∀ x y, (x, y) ∈ Γ ↔ |(x - F₁.1)^2 + (y - F₁.2)^2 - (x - F₂.1)^2 - (y - F₂.2)^2| = 2 * Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) * Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2))) →
  (∃ x y, (x, y) ∈ Γ ∧ asymptote x y) →
  ∃ x y, (x, y) ∈ Γ ↔ hyperbola_standard_form 27 9 x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1461_146134


namespace NUMINAMATH_CALUDE_equation_roots_l1461_146185

theorem equation_roots (c d : ℝ) : 
  (∀ x, (x + c) * (x + d) * (x - 5) / ((x + 4)^2) = 0 → x = -c ∨ x = -d ∨ x = 5) ∧
  (∀ x, x ≠ -4 → (x + c) * (x + d) * (x - 5) / ((x + 4)^2) ≠ 0) ∧
  (∀ x, (x + 2*c) * (x + 6) * (x + 9) / ((x + d) * (x - 5)) = 0 ↔ x = -4) →
  c = 1 ∧ d ≠ -6 ∧ d ≠ -9 ∧ 100 * c + d = 93 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l1461_146185


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_fraction_multiplication_l1461_146158

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem fraction_multiplication :
  (1 / 5 : ℚ) * (1 / 3 : ℚ) * (1 / 4 : ℚ) * 120 = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_fraction_multiplication_l1461_146158


namespace NUMINAMATH_CALUDE_expected_lotus_seed_is_three_l1461_146146

/-- The total number of zongzi -/
def total_zongzi : ℕ := 180

/-- The number of lotus seed zongzi -/
def lotus_seed_zongzi : ℕ := 54

/-- The size of the random sample -/
def sample_size : ℕ := 10

/-- The expected number of lotus seed zongzi in the sample -/
def expected_lotus_seed : ℚ := (sample_size : ℚ) * (lotus_seed_zongzi : ℚ) / (total_zongzi : ℚ)

theorem expected_lotus_seed_is_three :
  expected_lotus_seed = 3 := by sorry

end NUMINAMATH_CALUDE_expected_lotus_seed_is_three_l1461_146146


namespace NUMINAMATH_CALUDE_calculation_proof_l1461_146126

theorem calculation_proof : (42 / (12 - 10 + 3)) ^ 2 * 7 = 493.92 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1461_146126


namespace NUMINAMATH_CALUDE_five_balls_two_boxes_l1461_146147

/-- The number of ways to distribute distinguishable objects into distinguishable containers -/
def distribute_objects (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 32 ways to distribute 5 distinguishable balls into 2 distinguishable boxes -/
theorem five_balls_two_boxes : distribute_objects 5 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_two_boxes_l1461_146147


namespace NUMINAMATH_CALUDE_copper_alloy_percentage_l1461_146166

/-- Proves that the percentage of copper in the alloy that we need 32 kg of is 43.75% --/
theorem copper_alloy_percentage :
  ∀ (x : ℝ),
  -- Total mass of the final alloy
  let total_mass : ℝ := 40
  -- Percentage of copper in the final alloy
  let final_copper_percentage : ℝ := 45
  -- Mass of the alloy with unknown copper percentage
  let mass_unknown : ℝ := 32
  -- Mass of the alloy with 50% copper
  let mass_known : ℝ := 8
  -- Percentage of copper in the known alloy
  let known_copper_percentage : ℝ := 50
  -- The equation representing the mixture of alloys
  (mass_unknown * x / 100 + mass_known * known_copper_percentage / 100 = total_mass * final_copper_percentage / 100) →
  x = 43.75 := by
sorry

end NUMINAMATH_CALUDE_copper_alloy_percentage_l1461_146166


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l1461_146168

theorem smallest_solution_floor_equation : 
  ∀ x : ℝ, (x ≥ Real.sqrt 119 ∧ ⌊x^2⌋ - ⌊x⌋^2 = 19) ∨ (x < Real.sqrt 119 ∧ ⌊x^2⌋ - ⌊x⌋^2 ≠ 19) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l1461_146168


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l1461_146104

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x < -1 ∨ x > 2}

-- Theorem 1
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a ∈ Set.Icc 0 1 :=
sorry

-- Theorem 2
theorem union_equals_B_iff_a_in_range (a : ℝ) :
  A a ∪ B = B ↔ a ∈ Set.Iic (-2) ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l1461_146104


namespace NUMINAMATH_CALUDE_savings_calculation_l1461_146127

/-- Given a person's income and expenditure ratio, and their total income, calculate their savings. -/
def calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (total_income : ℕ) : ℕ :=
  let total_ratio := income_ratio + expenditure_ratio
  let expenditure := (expenditure_ratio * total_income) / total_ratio
  total_income - expenditure

/-- Theorem stating that given the specific income-expenditure ratio and total income, 
    the savings amount to 7000. -/
theorem savings_calculation :
  calculate_savings 3 2 21000 = 7000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l1461_146127


namespace NUMINAMATH_CALUDE_cosine_period_l1461_146116

theorem cosine_period (ω : ℝ) (h1 : ω > 0) : 
  (∃ y : ℝ → ℝ, y = λ x => Real.cos (ω * x - π / 6)) →
  (π / 5 = 2 * π / ω) →
  ω = 10 := by
sorry

end NUMINAMATH_CALUDE_cosine_period_l1461_146116


namespace NUMINAMATH_CALUDE_sin_angle_DAE_sin_angle_DAE_value_l1461_146124

/-- An equilateral triangle with side length 9 -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Points D and E on side BC -/
structure PointsOnBC where
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Main theorem: sin ∠DAE in the given configuration -/
theorem sin_angle_DAE (triangle : EquilateralTriangle) (points : PointsOnBC) : ℝ :=
  sorry

/-- The value of sin ∠DAE is √3/2 -/
theorem sin_angle_DAE_value (triangle : EquilateralTriangle) (points : PointsOnBC) :
    sin_angle_DAE triangle points = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_angle_DAE_sin_angle_DAE_value_l1461_146124


namespace NUMINAMATH_CALUDE_chessboard_ratio_l1461_146195

/-- The number of squares on an n x n chessboard -/
def num_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The number of rectangles on a chessboard with m horizontal and n vertical lines -/
def num_rectangles (m n : ℕ) : ℕ := (m.choose 2) * (n.choose 2)

theorem chessboard_ratio :
  (num_squares 9 : ℚ) / (num_rectangles 10 10 : ℚ) = 19 / 135 := by sorry

end NUMINAMATH_CALUDE_chessboard_ratio_l1461_146195


namespace NUMINAMATH_CALUDE_fathers_age_l1461_146136

/-- Proves that given the conditions, the father's age is 70 years. -/
theorem fathers_age (man_age : ℕ) (father_age : ℕ) : 
  man_age = (2 / 5 : ℚ) * father_age →
  man_age + 14 = (1 / 2 : ℚ) * (father_age + 14) →
  father_age = 70 :=
by sorry

end NUMINAMATH_CALUDE_fathers_age_l1461_146136


namespace NUMINAMATH_CALUDE_abc_inequality_l1461_146132

theorem abc_inequality (a b c : ℝ) (sum_eq_one : a + b + c = 1) (prod_pos : a * b * c > 0) :
  a * b + b * c + c * a < Real.sqrt (a * b * c) / 2 + 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1461_146132


namespace NUMINAMATH_CALUDE_dice_sum_probability_l1461_146125

theorem dice_sum_probability : 
  let die := Finset.range 6
  let outcomes := die.product die
  let favorable_outcomes := outcomes.filter (fun (x, y) => x + y + 2 ≥ 10)
  (favorable_outcomes.card : ℚ) / outcomes.card = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_probability_l1461_146125


namespace NUMINAMATH_CALUDE_jean_calories_consumed_l1461_146139

/-- Calculates the total calories consumed by Jean while writing her paper. -/
def total_calories (pages : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  let donuts := (pages + pages_per_donut - 1) / pages_per_donut
  donuts * calories_per_donut

/-- Proves that Jean consumes 1260 calories while writing her paper. -/
theorem jean_calories_consumed :
  total_calories 20 3 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_jean_calories_consumed_l1461_146139


namespace NUMINAMATH_CALUDE_dice_sum_divisibility_probability_l1461_146197

theorem dice_sum_divisibility_probability (n : ℕ) (p q r : ℝ) : 
  p ≥ 0 → q ≥ 0 → r ≥ 0 → p + q + r = 1 → 
  p^3 + q^3 + r^3 + 6*p*q*r ≥ 1/4 := by sorry

end NUMINAMATH_CALUDE_dice_sum_divisibility_probability_l1461_146197


namespace NUMINAMATH_CALUDE_man_son_age_difference_l1461_146114

/-- Represents the age difference between a man and his son -/
def ageDifference (manAge sonAge : ℕ) : ℕ := manAge - sonAge

/-- Theorem stating the age difference between a man and his son -/
theorem man_son_age_difference :
  ∀ (manAge sonAge : ℕ),
    sonAge = 18 →
    manAge + 2 = 2 * (sonAge + 2) →
    ageDifference manAge sonAge = 20 := by
  sorry


end NUMINAMATH_CALUDE_man_son_age_difference_l1461_146114


namespace NUMINAMATH_CALUDE_remainder_of_polynomial_l1461_146163

theorem remainder_of_polynomial (r : ℝ) : 
  (r^15 - r^3 + 1) % (r - 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_polynomial_l1461_146163


namespace NUMINAMATH_CALUDE_third_term_in_hundredth_group_l1461_146171

/-- The sequence term for a given index -/
def sequenceTerm (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of terms before the nth group -/
def termsBeforeGroup (n : ℕ) : ℕ := triangularNumber (n - 1)

/-- The last term in the nth group -/
def lastTermInGroup (n : ℕ) : ℕ := sequenceTerm (termsBeforeGroup (n + 1))

/-- The kth term in the nth group -/
def termInGroup (n k : ℕ) : ℕ := lastTermInGroup n - 2 * (n - k)

theorem third_term_in_hundredth_group :
  termInGroup 100 3 = 9905 := by sorry

end NUMINAMATH_CALUDE_third_term_in_hundredth_group_l1461_146171


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1461_146122

theorem quadratic_equation_solution (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 + 7 * x - 20 = 0) : x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1461_146122


namespace NUMINAMATH_CALUDE_inner_circle_radius_l1461_146160

/-- The radius of the inner tangent circle in a rectangle with semicircles --/
theorem inner_circle_radius (length width : ℝ) (h_length : length = 4) (h_width : width = 2) :
  let semicircle_radius := length / 8
  let center_to_semicircle := (3 * length / 8)^2 + (width / 2)^2
  (Real.sqrt center_to_semicircle / 2) - semicircle_radius = (Real.sqrt 10 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_inner_circle_radius_l1461_146160


namespace NUMINAMATH_CALUDE_distinct_words_count_l1461_146115

def first_digit (n : ℕ) : ℕ := 
  -- Definition of the function that returns the first digit of 2^n
  sorry

def word_sequence (start : ℕ) : List ℕ := 
  -- Definition of a list of 13 consecutive terms starting from 'start'
  (List.range 13).map (λ i => first_digit (start + i))

def distinct_words : Finset (List ℕ) :=
  -- Set of all distinct words in the sequence
  sorry

theorem distinct_words_count : Finset.card distinct_words = 57 := by
  sorry

end NUMINAMATH_CALUDE_distinct_words_count_l1461_146115


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l1461_146157

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 1

-- State the theorem
theorem even_function_implies_a_zero :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l1461_146157


namespace NUMINAMATH_CALUDE_ampersand_composition_l1461_146109

-- Define the & operation for the case &=9-x
def ampersand1 (x : ℤ) : ℤ := 9 - x

-- Define the & operation for the case &x = x - 9
def ampersand2 (x : ℤ) : ℤ := x - 9

-- Theorem to prove
theorem ampersand_composition : ampersand1 (ampersand2 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_composition_l1461_146109


namespace NUMINAMATH_CALUDE_max_value_of_a_minus_b_squared_l1461_146119

theorem max_value_of_a_minus_b_squared (a b : ℝ) (h : a^2 + b^2 = 4) :
  (∀ x y : ℝ, x^2 + y^2 = 4 → (x - y)^2 ≤ 8) ∧ 
  (∃ x y : ℝ, x^2 + y^2 = 4 ∧ (x - y)^2 = 8) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_minus_b_squared_l1461_146119


namespace NUMINAMATH_CALUDE_chess_tournament_rounds_l1461_146198

/-- The number of rounds needed for a chess tournament -/
theorem chess_tournament_rounds (n : ℕ) (games_per_round : ℕ) 
  (h1 : n = 20) 
  (h2 : games_per_round = 10) : 
  (n * (n - 1)) / games_per_round = 38 := by
  sorry

#check chess_tournament_rounds

end NUMINAMATH_CALUDE_chess_tournament_rounds_l1461_146198


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1461_146173

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    ∀ x : ℝ, x^3 - 9*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c) ∧
    a + b + c = 9) →
  p + q = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1461_146173


namespace NUMINAMATH_CALUDE_line_projection_onto_plane_line_projection_onto_plane_ratio_form_l1461_146169

/-- Given a line in 3D space defined by two equations and a plane, 
    this theorem states the equation of the projection of the line onto the plane. -/
theorem line_projection_onto_plane :
  ∀ (x y z : ℝ),
  (3*x - 2*y - z + 4 = 0 ∧ x - 4*y - 3*z - 2 = 0) →  -- Line equations
  (5*x + 2*y + 2*z - 7 = 0) →                        -- Plane equation
  ∃ (t : ℝ), 
    x = -2*t + 1 ∧                                   -- Parametric form of
    y = -14*t + 1 ∧                                  -- the projected line
    z = 19*t :=
by sorry

/-- An alternative formulation of the projection theorem using ratios. -/
theorem line_projection_onto_plane_ratio_form :
  ∀ (x y z : ℝ),
  (3*x - 2*y - z + 4 = 0 ∧ x - 4*y - 3*z - 2 = 0) →  -- Line equations
  (5*x + 2*y + 2*z - 7 = 0) →                        -- Plane equation
  (x - 1) / (-2) = (y - 1) / (-14) ∧ (x - 1) / (-2) = z / 19 :=
by sorry

end NUMINAMATH_CALUDE_line_projection_onto_plane_line_projection_onto_plane_ratio_form_l1461_146169


namespace NUMINAMATH_CALUDE_max_stores_visited_l1461_146101

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ)
  (two_store_visitors : ℕ) (three_store_visitors : ℕ) (four_store_visitors : ℕ)
  (h1 : total_stores = 15)
  (h2 : total_visits = 60)
  (h3 : total_shoppers = 30)
  (h4 : two_store_visitors = 12)
  (h5 : three_store_visitors = 6)
  (h6 : four_store_visitors = 4)
  (h7 : two_store_visitors * 2 + three_store_visitors * 3 + four_store_visitors * 4 < total_visits)
  (h8 : ∀ n : ℕ, n ≤ total_shoppers → n > 0) :
  ∃ (max_visited : ℕ), max_visited = 4 ∧ 
  ∀ (individual_visits : ℕ), individual_visits ≤ max_visited :=
sorry

end NUMINAMATH_CALUDE_max_stores_visited_l1461_146101


namespace NUMINAMATH_CALUDE_number_with_special_average_l1461_146131

theorem number_with_special_average (x : ℝ) (h1 : x ≠ 0) 
  (h2 : (x + x^2) / 2 = 5 * x) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_with_special_average_l1461_146131


namespace NUMINAMATH_CALUDE_reading_ratio_is_two_l1461_146107

/-- The minimum number of pages assigned for reading -/
def min_assigned : ℕ := 25

/-- The number of extra pages Harrison read -/
def harrison_extra : ℕ := 10

/-- The number of extra pages Pam read compared to Harrison -/
def pam_extra : ℕ := 15

/-- The number of pages Sam read -/
def sam_pages : ℕ := 100

/-- Calculate the number of pages Harrison read -/
def harrison_pages : ℕ := min_assigned + harrison_extra

/-- Calculate the number of pages Pam read -/
def pam_pages : ℕ := harrison_pages + pam_extra

/-- The ratio of pages Sam read to pages Pam read -/
def reading_ratio : ℚ := sam_pages / pam_pages

theorem reading_ratio_is_two : reading_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_reading_ratio_is_two_l1461_146107


namespace NUMINAMATH_CALUDE_fraction_problem_l1461_146187

theorem fraction_problem (x : ℚ) (h : 75 * x = 37.5) : x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1461_146187


namespace NUMINAMATH_CALUDE_arithmetic_sequence_smallest_negative_smallest_n_is_minimal_l1461_146178

/-- The smallest positive integer n such that 2009 - 7n < 0 -/
def smallest_n : ℕ := 288

theorem arithmetic_sequence_smallest_negative (n : ℕ) :
  n ≥ smallest_n ↔ 2009 - 7 * n < 0 :=
by
  sorry

theorem smallest_n_is_minimal :
  ∀ k : ℕ, k < smallest_n → 2009 - 7 * k ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_smallest_negative_smallest_n_is_minimal_l1461_146178


namespace NUMINAMATH_CALUDE_distance_from_focus_to_line_l1461_146110

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Define the right focus
def right_focus : ℝ × ℝ := (3, 0)

-- State the theorem
theorem distance_from_focus_to_line :
  let (x₀, y₀) := right_focus
  ∃ d : ℝ, d = |x₀ + 2*y₀ - 8| / Real.sqrt (1^2 + 2^2) ∧ d = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_distance_from_focus_to_line_l1461_146110


namespace NUMINAMATH_CALUDE_positive_expression_l1461_146112

theorem positive_expression (x : ℝ) (h : x > 0) : x^2 + π*x + (15*π/2)*Real.sin x > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l1461_146112


namespace NUMINAMATH_CALUDE_truncated_cone_radius_theorem_l1461_146183

/-- Represents a cone with its base radius -/
structure Cone where
  baseRadius : ℝ

/-- Represents a truncated cone with its smaller base radius -/
structure TruncatedCone where
  smallerBaseRadius : ℝ

/-- Given three cones touching each other and a truncated cone sharing
    a common generatrix with each, compute the smaller base radius of the truncated cone -/
def computeTruncatedConeRadius (c1 c2 c3 : Cone) : ℝ :=
  6

theorem truncated_cone_radius_theorem (c1 c2 c3 : Cone) (tc : TruncatedCone) 
    (h1 : c1.baseRadius = 23)
    (h2 : c2.baseRadius = 46)
    (h3 : c3.baseRadius = 69)
    (h4 : tc.smallerBaseRadius = computeTruncatedConeRadius c1 c2 c3) :
  tc.smallerBaseRadius = 6 := by
  sorry

end NUMINAMATH_CALUDE_truncated_cone_radius_theorem_l1461_146183


namespace NUMINAMATH_CALUDE_odd_function_tangent_line_sum_l1461_146145

def f (a b x : ℝ) : ℝ := a * x^3 + x + b

theorem odd_function_tangent_line_sum (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →  -- f is odd
  (∃ m c : ℝ, ∀ x, m * x + c = f a b 1 + (3 * a * 1^2 + 1) * (x - 1) ∧ 
              m * 2 + c = 6) →  -- tangent line passes through (2, 6)
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_tangent_line_sum_l1461_146145


namespace NUMINAMATH_CALUDE_moon_speed_km_per_hour_l1461_146184

/-- The speed of the moon around the earth in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.05

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_sec_to_km_per_hour (speed_km_per_sec : ℝ) : ℝ :=
  speed_km_per_sec * seconds_per_hour

/-- Theorem stating that the moon's speed in kilometers per hour is 3780 -/
theorem moon_speed_km_per_hour :
  km_per_sec_to_km_per_hour moon_speed_km_per_sec = 3780 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_km_per_hour_l1461_146184


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1461_146103

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is the arithmetic sequence
  (h1 : a 2 = 1)  -- given: a2 = 1
  (h2 : a 6 = 13)  -- given: a6 = 13
  : ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1461_146103


namespace NUMINAMATH_CALUDE_carriage_problem_representation_l1461_146155

/-- Represents the problem of people sharing carriages -/
structure CarriageProblem where
  x : ℕ  -- Total number of people
  y : ℕ  -- Total number of carriages

/-- The conditions of the carriage problem are satisfied -/
def satisfies_conditions (p : CarriageProblem) : Prop :=
  (p.x / 3 = p.y + 2) ∧ ((p.x - 9) / 2 = p.y)

/-- The system of equations correctly represents the carriage problem -/
theorem carriage_problem_representation (p : CarriageProblem) :
  satisfies_conditions p ↔ 
    (p.x / 3 = p.y + 2) ∧ ((p.x - 9) / 2 = p.y) :=
by sorry


end NUMINAMATH_CALUDE_carriage_problem_representation_l1461_146155


namespace NUMINAMATH_CALUDE_parabola_with_directrix_neg_seven_l1461_146140

/-- Represents a parabola with a vertical axis of symmetry -/
structure Parabola where
  /-- The distance from the vertex to the focus or directrix -/
  p : ℝ
  /-- Indicates whether the parabola opens to the right (true) or left (false) -/
  opensRight : Bool

/-- The standard equation of a parabola -/
def standardEquation (par : Parabola) : ℝ → ℝ → Prop :=
  if par.opensRight then
    fun y x => y^2 = 4 * par.p * x
  else
    fun y x => y^2 = -4 * par.p * x

/-- The equation of the directrix of a parabola -/
def directrixEquation (par : Parabola) : ℝ → Prop :=
  if par.opensRight then
    fun x => x = -par.p
  else
    fun x => x = par.p

theorem parabola_with_directrix_neg_seven (par : Parabola) :
  directrixEquation par = fun x => x = -7 →
  standardEquation par = fun y x => y^2 = 28 * x :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_with_directrix_neg_seven_l1461_146140


namespace NUMINAMATH_CALUDE_citizenship_test_study_time_l1461_146193

/-- Represents the time in minutes to learn each fill-in-the-blank question -/
def time_per_blank_question (total_questions : ℕ) (multiple_choice : ℕ) (fill_blank : ℕ) 
  (time_per_mc : ℕ) (total_study_time : ℕ) : ℕ :=
  ((total_study_time * 60) - (multiple_choice * time_per_mc)) / fill_blank

/-- Theorem stating that given the conditions, the time to learn each fill-in-the-blank question is 25 minutes -/
theorem citizenship_test_study_time :
  time_per_blank_question 60 30 30 15 20 = 25 := by
  sorry

end NUMINAMATH_CALUDE_citizenship_test_study_time_l1461_146193


namespace NUMINAMATH_CALUDE_three_digit_subtraction_result_zero_l1461_146162

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value ≤ 999

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Subtracts the sum of digits from a number -/
def subtract_sum_of_digits (n : ℕ) : ℕ :=
  n - sum_of_digits n

/-- Applies the subtraction process n times -/
def apply_n_times (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => apply_n_times n (subtract_sum_of_digits x)

/-- The main theorem to be proved -/
theorem three_digit_subtraction_result_zero (x : ThreeDigitNumber) :
  apply_n_times 100 x.value = 0 :=
sorry

end NUMINAMATH_CALUDE_three_digit_subtraction_result_zero_l1461_146162


namespace NUMINAMATH_CALUDE_triangle_shape_l1461_146188

/-- Given a triangle ABC where BC⋅cos A = AC⋅cos B, prove that the triangle is either isosceles or right-angled -/
theorem triangle_shape (A B C : Real) (BC AC : Real) 
  (h : BC * Real.cos A = AC * Real.cos B) :
  (A = B) ∨ (A + B = Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l1461_146188


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1461_146148

/-- Acme's pricing function -/
def acme_price (x : ℕ) : ℚ := 30 + 7 * x

/-- Gamma's pricing function -/
def gamma_price (x : ℕ) : ℚ := 11 * x

/-- The minimum number of shirts for which Acme is cheaper -/
def min_shirts_acme_cheaper : ℕ := 8

theorem acme_cheaper_at_min_shirts :
  acme_price min_shirts_acme_cheaper < gamma_price min_shirts_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_acme_cheaper → acme_price n ≥ gamma_price n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1461_146148


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l1461_146191

theorem arcsin_equation_solution :
  let f (x : ℝ) := Real.arcsin (x * Real.sqrt 5 / 3) + Real.arcsin (x * Real.sqrt 5 / 6) - Real.arcsin (7 * x * Real.sqrt 5 / 18)
  ∀ x : ℝ, 
    (abs x ≤ 18 / (7 * Real.sqrt 5)) →
    (f x = 0 ↔ x = 0 ∨ x = 8/7 ∨ x = -8/7) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l1461_146191


namespace NUMINAMATH_CALUDE_digit_125_of_1_17_l1461_146182

/-- The decimal representation of 1/17 -/
def decimal_rep_1_17 : List ℕ := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating sequence in the decimal representation of 1/17 -/
def repeat_length : ℕ := 16

/-- The 125th digit after the decimal point in the decimal representation of 1/17 is 4 -/
theorem digit_125_of_1_17 : 
  (decimal_rep_1_17[(125 - 1) % repeat_length]) = 4 := by sorry

end NUMINAMATH_CALUDE_digit_125_of_1_17_l1461_146182


namespace NUMINAMATH_CALUDE_bridge_length_l1461_146149

/-- The length of a bridge that a train can cross, given the train's length, speed, and time to cross. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross : ℝ) :
  train_length = 256 →
  train_speed_kmh = 72 →
  time_to_cross = 20 →
  (train_speed_kmh * 1000 / 3600 * time_to_cross) - train_length = 144 := by
  sorry

#check bridge_length

end NUMINAMATH_CALUDE_bridge_length_l1461_146149


namespace NUMINAMATH_CALUDE_max_sum_abc_l1461_146152

def An (a n : ℕ) : ℕ := a * (8^n - 1) / 7
def Bn (b n : ℕ) : ℕ := b * (6^n - 1) / 5
def Cn (c n : ℕ) : ℕ := c * (10^(3*n) - 1) / 9

theorem max_sum_abc :
  ∃ (a b c : ℕ),
    (0 < a ∧ a ≤ 9) ∧
    (0 < b ∧ b ≤ 9) ∧
    (0 < c ∧ c ≤ 9) ∧
    (∃ (n₁ n₂ : ℕ), n₁ ≠ n₂ ∧ 
      Cn c n₁ - Bn b n₁ = (An a n₁)^3 ∧
      Cn c n₂ - Bn b n₂ = (An a n₂)^3) ∧
    (∀ (a' b' c' : ℕ),
      (0 < a' ∧ a' ≤ 9) →
      (0 < b' ∧ b' ≤ 9) →
      (0 < c' ∧ c' ≤ 9) →
      (∃ (n₁ n₂ : ℕ), n₁ ≠ n₂ ∧ 
        Cn c' n₁ - Bn b' n₁ = (An a' n₁)^3 ∧
        Cn c' n₂ - Bn b' n₂ = (An a' n₂)^3) →
      a + b + c ≥ a' + b' + c') ∧
    a + b + c = 21 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_abc_l1461_146152


namespace NUMINAMATH_CALUDE_bakery_order_cost_is_54_l1461_146123

/-- Calculates the final cost of a bakery order with a possible discount --/
def bakery_order_cost (quiche_price croissant_price biscuit_price : ℚ) 
  (quiche_quantity croissant_quantity biscuit_quantity : ℕ)
  (discount_rate : ℚ) (discount_threshold : ℚ) : ℚ :=
  let total_before_discount := 
    quiche_price * quiche_quantity + 
    croissant_price * croissant_quantity + 
    biscuit_price * biscuit_quantity
  let discount := 
    if total_before_discount > discount_threshold 
    then total_before_discount * discount_rate 
    else 0
  total_before_discount - discount

/-- Theorem stating that the bakery order cost is $54.00 given the specified conditions --/
theorem bakery_order_cost_is_54 : 
  bakery_order_cost 15 3 2 2 6 6 (1/10) 50 = 54 := by
  sorry

end NUMINAMATH_CALUDE_bakery_order_cost_is_54_l1461_146123


namespace NUMINAMATH_CALUDE_b_over_a_range_l1461_146108

-- Define an acute triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π

-- Define the function f
def f (x a b c : ℝ) : ℝ := x^2 + c^2 - a^2 - a*b

-- State the theorem
theorem b_over_a_range (t : AcuteTriangle) 
  (h : ∃! x, f x t.a t.b t.c = 0) : 
  1 < t.b / t.a ∧ t.b / t.a < 2 := by
  sorry

end NUMINAMATH_CALUDE_b_over_a_range_l1461_146108


namespace NUMINAMATH_CALUDE_total_cans_is_twelve_l1461_146180

/-- Represents the ratio of chili beans to tomato soup -/
def chili_to_tomato_ratio : ℚ := 2

/-- Represents the number of chili bean cans ordered -/
def chili_beans_ordered : ℕ := 8

/-- Calculates the total number of cans ordered -/
def total_cans_ordered : ℕ :=
  chili_beans_ordered + (chili_beans_ordered / chili_to_tomato_ratio.num).toNat

/-- Proves that the total number of cans ordered is 12 -/
theorem total_cans_is_twelve : total_cans_ordered = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_is_twelve_l1461_146180


namespace NUMINAMATH_CALUDE_cats_remaining_after_sale_l1461_146150

/-- The number of cats remaining after a sale at a pet store -/
theorem cats_remaining_after_sale 
  (siamese_cats : ℕ) 
  (house_cats : ℕ) 
  (cats_sold : ℕ) 
  (h1 : siamese_cats = 13)
  (h2 : house_cats = 5)
  (h3 : cats_sold = 10) :
  siamese_cats + house_cats - cats_sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_after_sale_l1461_146150


namespace NUMINAMATH_CALUDE_triangle_area_l1461_146105

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is √3/2 when c = √2, b = √6, and B = 120°. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 2 →
  b = Real.sqrt 6 →
  B = 2 * π / 3 →  -- 120° in radians
  (1/2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1461_146105


namespace NUMINAMATH_CALUDE_blue_balloons_count_l1461_146135

def total_balloons : ℕ := 200
def red_percentage : ℚ := 35 / 100
def green_percentage : ℚ := 25 / 100
def purple_percentage : ℚ := 15 / 100

theorem blue_balloons_count :
  (total_balloons : ℚ) * (1 - (red_percentage + green_percentage + purple_percentage)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_blue_balloons_count_l1461_146135


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1461_146190

theorem linear_equation_solution (V : ℝ → ℝ) (p q : ℝ) 
  (h1 : ∀ t, V t = p * t + q)
  (h2 : V 0 = 100)
  (h3 : V 10 = 103.5) :
  p = 0.35 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1461_146190


namespace NUMINAMATH_CALUDE_cistern_fill_time_with_leak_l1461_146138

/-- The additional time required to fill a cistern with a leak -/
theorem cistern_fill_time_with_leak 
  (normal_fill_time : ℝ) 
  (leak_empty_time : ℝ) 
  (h1 : normal_fill_time = 8) 
  (h2 : leak_empty_time = 40.00000000000001) : 
  (1 / (1 / normal_fill_time - 1 / leak_empty_time)) - normal_fill_time = 2.000000000000003 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_with_leak_l1461_146138


namespace NUMINAMATH_CALUDE_square_difference_value_l1461_146177

theorem square_difference_value (x y : ℝ) 
  (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) : 
  (x - y)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_square_difference_value_l1461_146177


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l1461_146176

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (x + 3) / (x - 2)) ↔ x ≥ -3 ∧ x ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l1461_146176


namespace NUMINAMATH_CALUDE_officers_selection_count_l1461_146102

/-- The number of ways to select 3 distinct individuals from a group of 6 people to fill 3 distinct positions -/
def selectOfficers (n : ℕ) : ℕ :=
  if n < 3 then 0
  else n * (n - 1) * (n - 2)

/-- Theorem stating that selecting 3 officers from 6 people results in 120 possibilities -/
theorem officers_selection_count : selectOfficers 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_officers_selection_count_l1461_146102


namespace NUMINAMATH_CALUDE_final_fish_count_l1461_146165

/- Define the initial number of fish -/
variable (F : ℚ)

/- Define the number of fish after each day's operations -/
def fish_count (day : ℕ) : ℚ :=
  match day with
  | 0 => F
  | 1 => 2 * F
  | 2 => 4 * F * (2/3)
  | 3 => 8 * F * (2/3)
  | 4 => 16 * F * (2/3) * (3/4)
  | 5 => 32 * F * (2/3) * (3/4)
  | 6 => 64 * F * (2/3) * (3/4)
  | _ => 128 * F * (2/3) * (3/4) + 15

/- Theorem stating that the final count is 207 if and only if F = 6 -/
theorem final_fish_count (F : ℚ) : fish_count F 7 = 207 ↔ F = 6 := by
  sorry

end NUMINAMATH_CALUDE_final_fish_count_l1461_146165


namespace NUMINAMATH_CALUDE_range_of_a_l1461_146106

-- Define a decreasing function on (-1, 1)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f y < f x

-- Define the theorem
theorem range_of_a (f : ℝ → ℝ) (h_decreasing : DecreasingFunction f) :
  (∀ a, f (2 * a - 1) < f (1 - a)) → 
  (∀ a, (2/3 : ℝ) < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1461_146106


namespace NUMINAMATH_CALUDE_kimberly_skittles_l1461_146141

/-- Calculates the total number of Skittles Kimberly has -/
def total_skittles (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Proves that Kimberly has 12 Skittles in total -/
theorem kimberly_skittles : total_skittles 5 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_skittles_l1461_146141


namespace NUMINAMATH_CALUDE_klinker_daughter_age_l1461_146113

/-- Proves that given Mr. Klinker is 35 years old and in 15 years he will be twice as old as his daughter, his daughter's current age is 10 years. -/
theorem klinker_daughter_age (klinker_age : ℕ) (daughter_age : ℕ) : 
  klinker_age = 35 →
  klinker_age + 15 = 2 * (daughter_age + 15) →
  daughter_age = 10 := by
sorry

end NUMINAMATH_CALUDE_klinker_daughter_age_l1461_146113


namespace NUMINAMATH_CALUDE_coffee_decaf_percentage_l1461_146164

def initial_stock : ℝ := 800
def type_a_percent : ℝ := 0.40
def type_b_percent : ℝ := 0.35
def type_c_percent : ℝ := 0.25
def type_a_decaf : ℝ := 0.20
def type_b_decaf : ℝ := 0.50
def type_c_decaf : ℝ := 0

def additional_purchase : ℝ := 300
def additional_type_a_percent : ℝ := 0.50
def additional_type_b_percent : ℝ := 0.30
def additional_type_c_percent : ℝ := 0.20

theorem coffee_decaf_percentage :
  let total_stock := initial_stock + additional_purchase
  let initial_decaf := 
    initial_stock * (type_a_percent * type_a_decaf + 
                     type_b_percent * type_b_decaf + 
                     type_c_percent * type_c_decaf)
  let additional_decaf := 
    additional_purchase * (additional_type_a_percent * type_a_decaf + 
                           additional_type_b_percent * type_b_decaf + 
                           additional_type_c_percent * type_c_decaf)
  let total_decaf := initial_decaf + additional_decaf
  (total_decaf / total_stock) * 100 = (279 / 1100) * 100 := by
sorry

end NUMINAMATH_CALUDE_coffee_decaf_percentage_l1461_146164


namespace NUMINAMATH_CALUDE_rachelle_pennies_l1461_146170

/-- The number of pennies thrown by Rachelle, Gretchen, and Rocky -/
structure PennyThrowers where
  rachelle : ℕ
  gretchen : ℕ
  rocky : ℕ

/-- The conditions of the penny-throwing problem -/
def PennyConditions (p : PennyThrowers) : Prop :=
  p.gretchen = p.rachelle / 2 ∧
  p.rocky = p.gretchen / 3 ∧
  p.rachelle + p.gretchen + p.rocky = 300

/-- Theorem stating that under the given conditions, Rachelle threw 180 pennies -/
theorem rachelle_pennies (p : PennyThrowers) (h : PennyConditions p) : p.rachelle = 180 := by
  sorry

end NUMINAMATH_CALUDE_rachelle_pennies_l1461_146170


namespace NUMINAMATH_CALUDE_digit_product_property_l1461_146121

theorem digit_product_property :
  ∃! (count : Nat), count = (Finset.filter 
    (fun pair : Nat × Nat => 
      let x := pair.1
      let y := pair.2
      x ≠ y ∧ 
      x < 10 ∧ 
      y < 10 ∧ 
      1000 ≤ x * 1111 ∧ 
      x * 1111 < 10000 ∧
      1000 ≤ y * 1111 ∧ 
      y * 1111 < 10000 ∧
      1000000 ≤ (x * 1111) * (y * 1111) ∧ 
      (x * 1111) * (y * 1111) < 10000000 ∧
      (x * 1111) * (y * 1111) % 10 = x ∧
      ((x * 1111) * (y * 1111) / 1000000) % 10 = x)
    (Finset.product (Finset.range 10) (Finset.range 10))).card ∧
  count = 3 := by
sorry

end NUMINAMATH_CALUDE_digit_product_property_l1461_146121


namespace NUMINAMATH_CALUDE_x_sixth_minus_six_x_squared_l1461_146189

theorem x_sixth_minus_six_x_squared (x : ℝ) (h : x = 3) : x^6 - 6*x^2 = 675 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_minus_six_x_squared_l1461_146189


namespace NUMINAMATH_CALUDE_randy_house_blocks_l1461_146144

/-- The number of blocks Randy used to build the house -/
def blocks_for_house : ℕ := 20

/-- The total number of blocks Randy has -/
def total_blocks : ℕ := 95

/-- The number of blocks Randy used to build the tower -/
def blocks_for_tower : ℕ := 50

theorem randy_house_blocks :
  blocks_for_house = 20 ∧
  total_blocks = 95 ∧
  blocks_for_tower = 50 ∧
  blocks_for_tower = blocks_for_house + 30 :=
sorry

end NUMINAMATH_CALUDE_randy_house_blocks_l1461_146144


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l1461_146186

-- Define the quadratic function
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 1

-- Define monotonicity in an open interval
def MonotonicIn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ f y < f x)

-- Theorem statement
theorem quadratic_monotonicity (t : ℝ) :
  MonotonicIn (f t) 1 3 → t ≤ 1 ∨ t ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l1461_146186


namespace NUMINAMATH_CALUDE_unattainable_y_l1461_146199

theorem unattainable_y (x : ℝ) (y : ℝ) (h1 : x ≠ -3/2) (h2 : y = (1-x)/(2*x+3)) :
  y ≠ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_y_l1461_146199
