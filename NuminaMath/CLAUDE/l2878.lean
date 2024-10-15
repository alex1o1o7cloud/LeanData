import Mathlib

namespace NUMINAMATH_CALUDE_shape_ratios_l2878_287890

/-- Given three shapes (cube A, cube B, and cylinder C) with specific volume ratios
    and height relationships, this theorem proves the ratios of their dimensions. -/
theorem shape_ratios (a b r : ℝ) (h : ℝ) :
  a > 0 ∧ b > 0 ∧ r > 0 ∧ h > 0 →
  h = a →
  a^3 / b^3 = 81 / 25 →
  a^3 / (π * r^2 * h) = 81 / 40 →
  (a / b = 3 / 5) ∧ (a / r = 9 * Real.sqrt π / Real.sqrt 40) := by
  sorry

#check shape_ratios

end NUMINAMATH_CALUDE_shape_ratios_l2878_287890


namespace NUMINAMATH_CALUDE_parallelogram_area_18_10_l2878_287896

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 18 cm and height 10 cm is 180 cm² -/
theorem parallelogram_area_18_10 : parallelogram_area 18 10 = 180 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_18_10_l2878_287896


namespace NUMINAMATH_CALUDE_necklace_cost_calculation_l2878_287802

/-- The cost of a single necklace -/
def necklace_cost : ℕ := sorry

/-- The number of necklaces sold -/
def necklaces_sold : ℕ := 4

/-- The number of rings sold -/
def rings_sold : ℕ := 8

/-- The cost of a single ring -/
def ring_cost : ℕ := 4

/-- The total sales amount -/
def total_sales : ℕ := 80

theorem necklace_cost_calculation :
  necklace_cost = 12 :=
by
  sorry

#check necklace_cost_calculation

end NUMINAMATH_CALUDE_necklace_cost_calculation_l2878_287802


namespace NUMINAMATH_CALUDE_sharp_four_times_25_l2878_287894

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- State the theorem
theorem sharp_four_times_25 : sharp (sharp (sharp (sharp 25))) = 7.592 := by
  sorry

end NUMINAMATH_CALUDE_sharp_four_times_25_l2878_287894


namespace NUMINAMATH_CALUDE_greatest_mpn_l2878_287899

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  tens_nonzero : tens ≠ 0
  tens_single_digit : tens < 10
  ones_single_digit : ones < 10

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_nonzero : hundreds ≠ 0
  hundreds_single_digit : hundreds < 10
  tens_single_digit : tens < 10
  ones_single_digit : ones < 10

def is_valid_mpn (m n : Nat) (mpn : ThreeDigitNumber) : Prop :=
  m ≠ n ∧
  m < 10 ∧
  n < 10 ∧
  mpn.hundreds = m ∧
  mpn.ones = m ∧
  (10 * m + n) * m = 100 * mpn.hundreds + 10 * mpn.tens + mpn.ones

theorem greatest_mpn :
  ∀ m n : Nat,
  ∀ mpn : ThreeDigitNumber,
  is_valid_mpn m n mpn →
  mpn.hundreds * 100 + mpn.tens * 10 + mpn.ones ≤ 898 :=
sorry

end NUMINAMATH_CALUDE_greatest_mpn_l2878_287899


namespace NUMINAMATH_CALUDE_expand_equality_l2878_287834

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the third term of the expansion
def third_term (x y : ℝ) : ℝ := (binomial 10 2 : ℝ) * x^8 * y^2

-- Define the fourth term of the expansion
def fourth_term (x y : ℝ) : ℝ := (binomial 10 3 : ℝ) * x^7 * y^3

-- Main theorem
theorem expand_equality (p q : ℝ) 
  (h_pos_p : p > 0) 
  (h_pos_q : q > 0) 
  (h_sum : p + q = 2) 
  (h_equal : third_term p q = fourth_term p q) : 
  p = 16/11 := by
sorry

end NUMINAMATH_CALUDE_expand_equality_l2878_287834


namespace NUMINAMATH_CALUDE_min_value_of_f_l2878_287871

def f (x : ℝ) : ℝ := (x - 1)^2 + 3

theorem min_value_of_f :
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2878_287871


namespace NUMINAMATH_CALUDE_single_digit_addition_l2878_287873

theorem single_digit_addition (A : ℕ) : 
  A < 10 → -- A is a single digit number
  10 * A + A + 10 * A + A = 132 → -- AA + AA = 132
  A = 6 := by sorry

end NUMINAMATH_CALUDE_single_digit_addition_l2878_287873


namespace NUMINAMATH_CALUDE_z1_in_first_quadrant_l2878_287867

def z1 (a : ℝ) : ℂ := a + Complex.I
def z2 : ℂ := 1 - Complex.I

theorem z1_in_first_quadrant (a : ℝ) :
  (z1 a / z2).im ≠ 0 ∧ (z1 a / z2).re = 0 →
  0 < (z1 a).re ∧ 0 < (z1 a).im :=
by sorry

end NUMINAMATH_CALUDE_z1_in_first_quadrant_l2878_287867


namespace NUMINAMATH_CALUDE_ariels_age_multiplier_l2878_287800

theorem ariels_age_multiplier :
  let current_age : ℕ := 5
  let years_passed : ℕ := 15
  let future_age : ℕ := current_age + years_passed
  ∃ (multiplier : ℕ), future_age = multiplier * current_age ∧ multiplier = 4 :=
by sorry

end NUMINAMATH_CALUDE_ariels_age_multiplier_l2878_287800


namespace NUMINAMATH_CALUDE_company_female_employees_l2878_287880

theorem company_female_employees :
  ∀ (total_employees : ℕ) 
    (advanced_degrees : ℕ) 
    (males_college_only : ℕ) 
    (females_advanced : ℕ),
  total_employees = 180 →
  advanced_degrees = 90 →
  males_college_only = 35 →
  females_advanced = 55 →
  ∃ (female_employees : ℕ),
    female_employees = 110 ∧
    female_employees = (total_employees - advanced_degrees - males_college_only) + females_advanced :=
by
  sorry

end NUMINAMATH_CALUDE_company_female_employees_l2878_287880


namespace NUMINAMATH_CALUDE_treasure_chest_gems_l2878_287879

theorem treasure_chest_gems (diamonds : ℕ) (rubies : ℕ) : 
  diamonds = 45 → rubies = 5110 → diamonds + rubies = 5155 := by
  sorry

end NUMINAMATH_CALUDE_treasure_chest_gems_l2878_287879


namespace NUMINAMATH_CALUDE_solution_set_equiv_solution_values_l2878_287809

-- Part I
def solution_set (x : ℝ) : Prop := |x + 3| < 2*x + 1

theorem solution_set_equiv : ∀ x : ℝ, solution_set x ↔ x > 2 := by sorry

-- Part II
def has_solution (t : ℝ) : Prop := 
  t ≠ 0 ∧ ∃ x : ℝ, |x - t| + |x + 1/t| = 2

theorem solution_values : ∀ t : ℝ, has_solution t → t = 1 ∨ t = -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_equiv_solution_values_l2878_287809


namespace NUMINAMATH_CALUDE_area_of_overlapping_squares_area_of_overlapping_squares_is_252_l2878_287888

/-- The area of the region covered by two congruent squares with side length 12 units,
    where one corner of one square coincides with a corner of the other square. -/
theorem area_of_overlapping_squares : ℝ :=
  let square_side_length : ℝ := 12
  let single_square_area : ℝ := square_side_length ^ 2
  let total_area_without_overlap : ℝ := 2 * single_square_area
  let overlap_area : ℝ := single_square_area / 4
  total_area_without_overlap - overlap_area

/-- The area of the region covered by the two squares is 252 square units. -/
theorem area_of_overlapping_squares_is_252 :
  area_of_overlapping_squares = 252 := by sorry

end NUMINAMATH_CALUDE_area_of_overlapping_squares_area_of_overlapping_squares_is_252_l2878_287888


namespace NUMINAMATH_CALUDE_exclusive_albums_count_l2878_287822

/-- The number of albums that are in either Andrew's or Bella's collection, but not both. -/
def exclusive_albums (shared : ℕ) (andrew_total : ℕ) (bella_unique : ℕ) : ℕ :=
  (andrew_total - shared) + bella_unique

/-- Theorem stating that the number of exclusive albums is 17 given the problem conditions. -/
theorem exclusive_albums_count :
  exclusive_albums 15 23 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_exclusive_albums_count_l2878_287822


namespace NUMINAMATH_CALUDE_circle_center_transformation_l2878_287811

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

theorem circle_center_transformation :
  let S : ℝ × ℝ := (-2, 6)
  let reflected := reflect_x S
  let final := translate_up reflected 10
  final = (-2, 4) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l2878_287811


namespace NUMINAMATH_CALUDE_cube_iff_diagonal_perpendicular_l2878_287801

/-- A rectangular parallelepiped -/
structure RectangularParallelepiped where
  -- Add necessary fields and properties here

/-- Predicate for a rectangular parallelepiped being a cube -/
def is_cube (S : RectangularParallelepiped) : Prop :=
  sorry

/-- Predicate for the diagonal perpendicularity property -/
def diagonal_perpendicular_property (S : RectangularParallelepiped) : Prop :=
  sorry

/-- Theorem stating the equivalence of the cube property and the diagonal perpendicularity property -/
theorem cube_iff_diagonal_perpendicular (S : RectangularParallelepiped) :
  is_cube S ↔ diagonal_perpendicular_property S :=
sorry

end NUMINAMATH_CALUDE_cube_iff_diagonal_perpendicular_l2878_287801


namespace NUMINAMATH_CALUDE_michaels_birds_l2878_287886

/-- Given Michael's pets distribution, prove he has 12 birds -/
theorem michaels_birds (total_pets : ℕ) (dog_percent cat_percent bunny_percent : ℚ) : 
  total_pets = 120 →
  dog_percent = 30 / 100 →
  cat_percent = 40 / 100 →
  bunny_percent = 20 / 100 →
  (↑total_pets * (1 - dog_percent - cat_percent - bunny_percent) : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_michaels_birds_l2878_287886


namespace NUMINAMATH_CALUDE_dog_weight_gain_l2878_287823

/-- Given a golden retriever that:
    - Is 8 years old
    - Currently weighs 88 pounds
    - Weighed 40 pounds at 1 year old
    Prove that the average yearly weight gain is 6 pounds. -/
theorem dog_weight_gain (current_weight : ℕ) (age : ℕ) (initial_weight : ℕ) 
  (h1 : current_weight = 88)
  (h2 : age = 8)
  (h3 : initial_weight = 40) :
  (current_weight - initial_weight) / (age - 1) = 6 :=
sorry

end NUMINAMATH_CALUDE_dog_weight_gain_l2878_287823


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2878_287882

theorem cube_volume_problem (a : ℝ) : 
  a > 0 →
  (a + 2) * (a + 2) * a - a^3 = 12 →
  a^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2878_287882


namespace NUMINAMATH_CALUDE_floor_product_equals_45_l2878_287821

theorem floor_product_equals_45 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 45 ↔ x ∈ Set.Ico 7.5 (46 / 6) :=
sorry

end NUMINAMATH_CALUDE_floor_product_equals_45_l2878_287821


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l2878_287837

-- Define the room dimensions
def room_length : ℝ := 10
def room_width : ℝ := 4.75

-- Define the paving rate
def paving_rate : ℝ := 900

-- Calculate the area of the room
def room_area : ℝ := room_length * room_width

-- Calculate the total cost of paving
def paving_cost : ℝ := room_area * paving_rate

-- Theorem to prove
theorem paving_cost_calculation : paving_cost = 42750 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l2878_287837


namespace NUMINAMATH_CALUDE_sin_alpha_value_l2878_287813

theorem sin_alpha_value (α : Real) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 4) 
  (h3 : Real.sin α * Real.cos α = 3 * Real.sqrt 7 / 16) : 
  Real.sin α = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l2878_287813


namespace NUMINAMATH_CALUDE_problem_solution_l2878_287826

/-- Check if a number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Check if a number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k, n = k * k

/-- Get the last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- Get the hundreds digit of a number -/
def hundredsDigit (n : ℕ) : ℕ := (n / 100) % 10

/-- Create a new number by placing one digit in front of another number -/
def placeDigitInFront (digit : ℕ) (n : ℕ) : ℕ := digit * 100 + n

theorem problem_solution :
  let a : ℕ := 3
  let b : ℕ := 44
  let c : ℕ := 149
  (a < 10 ∧ b < 100 ∧ b ≥ 10 ∧ c < 1000 ∧ c ≥ 100) ∧
  (isOdd a ∧ isEven b ∧ isOdd c) ∧
  (lastTwoDigits (a * b * c) = 68) ∧
  (isPerfectSquare (a + b + c)) ∧
  (isPerfectSquare (placeDigitInFront (hundredsDigit c) b) ∧ isPerfectSquare (c % 100)) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l2878_287826


namespace NUMINAMATH_CALUDE_expense_settlement_proof_l2878_287869

def expense_settlement (alice_paid bob_paid charlie_paid : ℚ) : Prop :=
  let total_paid := alice_paid + bob_paid + charlie_paid
  let share_per_person := total_paid / 3
  let alice_owes := share_per_person - alice_paid
  let bob_owes := share_per_person - bob_paid
  let charlie_owed := charlie_paid - share_per_person
  ∃ a b : ℚ, 
    a = alice_owes ∧ 
    b = bob_owes ∧ 
    a - b = 30

theorem expense_settlement_proof :
  expense_settlement 130 160 210 := by
  sorry

end NUMINAMATH_CALUDE_expense_settlement_proof_l2878_287869


namespace NUMINAMATH_CALUDE_room_extension_ratio_l2878_287851

/-- Given a room with original length, width, and an extension to the length,
    prove that the ratio of the new total length to the new perimeter is 35:100. -/
theorem room_extension_ratio (original_length width extension : ℕ) 
  (h1 : original_length = 25)
  (h2 : width = 15)
  (h3 : extension = 10) :
  (original_length + extension) * 100 = 35 * (2 * (original_length + extension + width)) :=
by sorry

end NUMINAMATH_CALUDE_room_extension_ratio_l2878_287851


namespace NUMINAMATH_CALUDE_triple_sum_of_45_2_and_quarter_l2878_287877

theorem triple_sum_of_45_2_and_quarter (x : ℝ) (h : x = 45.2 + (1 / 4)) :
  3 * x = 136.35 := by
  sorry

end NUMINAMATH_CALUDE_triple_sum_of_45_2_and_quarter_l2878_287877


namespace NUMINAMATH_CALUDE_regular_star_points_l2878_287804

/-- An n-pointed regular star polygon -/
structure RegularStar where
  n : ℕ
  angle_A : ℝ
  angle_B : ℝ
  h1 : angle_A = angle_B - 15
  h2 : n * angle_A = n * angle_B - 180

theorem regular_star_points (star : RegularStar) : star.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_star_points_l2878_287804


namespace NUMINAMATH_CALUDE_harry_age_l2878_287814

/-- Represents the ages of the people in the problem -/
structure Ages where
  kiarra : ℕ
  bea : ℕ
  job : ℕ
  figaro : ℕ
  harry : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.kiarra = 2 * ages.bea ∧
  ages.job = 3 * ages.bea ∧
  ages.figaro = ages.job + 7 ∧
  2 * ages.harry = ages.figaro ∧
  ages.kiarra = 30

/-- The theorem stating that under the given conditions, Harry's age is 26 -/
theorem harry_age (ages : Ages) :
  problem_conditions ages → ages.harry = 26 := by
  sorry

end NUMINAMATH_CALUDE_harry_age_l2878_287814


namespace NUMINAMATH_CALUDE_fourth_power_sum_l2878_287846

theorem fourth_power_sum (a b t : ℝ) 
  (h1 : a + b = t) 
  (h2 : a^2 + b^2 = t) 
  (h3 : a^3 + b^3 = t) : 
  a^4 + b^4 = t := by sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l2878_287846


namespace NUMINAMATH_CALUDE_original_integer_is_45_l2878_287849

theorem original_integer_is_45 (a b c d : ℤ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (eq1 : (b + c + d) / 3 + 10 = 37)
  (eq2 : (a + c + d) / 3 + 10 = 31)
  (eq3 : (a + b + d) / 3 + 10 = 25)
  (eq4 : (a + b + c) / 3 + 10 = 19) :
  a = 45 ∨ b = 45 ∨ c = 45 ∨ d = 45 :=
sorry

end NUMINAMATH_CALUDE_original_integer_is_45_l2878_287849


namespace NUMINAMATH_CALUDE_homework_problem_ratio_l2878_287853

theorem homework_problem_ratio : 
  ∀ (total_problems : ℕ) 
    (martha_problems : ℕ) 
    (angela_problems : ℕ) 
    (jenna_problems : ℕ),
  total_problems = 20 →
  martha_problems = 2 →
  angela_problems = 9 →
  jenna_problems + martha_problems + (jenna_problems / 2) + angela_problems = total_problems →
  (jenna_problems : ℚ) / martha_problems = 3 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_ratio_l2878_287853


namespace NUMINAMATH_CALUDE_claire_photos_l2878_287891

theorem claire_photos (lisa robert claire : ℕ) 
  (h1 : lisa = robert) 
  (h2 : lisa = 3 * claire) 
  (h3 : robert = claire + 16) : 
  claire = 8 := by
sorry

end NUMINAMATH_CALUDE_claire_photos_l2878_287891


namespace NUMINAMATH_CALUDE_original_number_of_people_l2878_287872

theorem original_number_of_people (x : ℕ) : 
  (x / 2 : ℚ) = 18 → x = 36 := by sorry

end NUMINAMATH_CALUDE_original_number_of_people_l2878_287872


namespace NUMINAMATH_CALUDE_acid_concentration_theorem_l2878_287893

def acid_concentration_problem (acid1 acid2 acid3 : ℝ) 
  (conc1 conc2 : ℝ) : Prop :=
  let water1 := acid1 / conc1 - acid1
  let water2 := acid2 / conc2 - acid2
  let total_water := water1 + water2
  let conc3 := acid3 / (acid3 + total_water)
  acid1 = 10 ∧ 
  acid2 = 20 ∧ 
  acid3 = 30 ∧ 
  conc1 = 0.05 ∧ 
  conc2 = 70/300 ∧ 
  conc3 = 0.105

theorem acid_concentration_theorem : 
  acid_concentration_problem 10 20 30 0.05 (70/300) :=
sorry

end NUMINAMATH_CALUDE_acid_concentration_theorem_l2878_287893


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_five_l2878_287852

/-- Given two real-valued functions f and h, where f is linear and h is affine,
    and a condition relating their composition to a linear function,
    prove that the sum of the coefficients of f is 5. -/
theorem sum_of_coefficients_is_five
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h : ℝ → ℝ)
  (h_def : ∀ x, h x = 3 * x - 6)
  (f_def : ∀ x, f x = a * x + b)
  (composition_condition : ∀ x, h (f x) = 4 * x + 5) :
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_five_l2878_287852


namespace NUMINAMATH_CALUDE_expression_equality_l2878_287854

theorem expression_equality : (8 * 10^10) / (2 * 10^5 * 4) = 100000 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2878_287854


namespace NUMINAMATH_CALUDE_x_squared_minus_five_is_quadratic_l2878_287824

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² - 5 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 5

/-- Theorem: x² - 5 = 0 is a quadratic equation -/
theorem x_squared_minus_five_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_five_is_quadratic_l2878_287824


namespace NUMINAMATH_CALUDE_gold_cube_profit_l2878_287845

-- Define the cube's side length in cm
def cube_side : ℝ := 6

-- Define the density of gold in g/cm³
def gold_density : ℝ := 19

-- Define the buying price per gram in dollars
def buying_price : ℝ := 60

-- Define the selling price multiplier
def selling_multiplier : ℝ := 1.5

-- Theorem statement
theorem gold_cube_profit :
  let volume : ℝ := cube_side ^ 3
  let mass : ℝ := gold_density * volume
  let cost : ℝ := mass * buying_price
  let selling_price : ℝ := cost * selling_multiplier
  selling_price - cost = 123120 := by
  sorry

end NUMINAMATH_CALUDE_gold_cube_profit_l2878_287845


namespace NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_a_range_l2878_287857

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 2|

-- Theorem for part 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f a x ≥ 3) ∧ (∃ x, f a x = 3) → a = 1 ∨ a = -5 :=
sorry

-- Theorem for part 2
theorem inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_a_range_l2878_287857


namespace NUMINAMATH_CALUDE_story_problem_solution_l2878_287876

/-- Represents the story writing problem with given parameters -/
structure StoryProblem where
  total_words : ℕ
  num_chapters : ℕ
  total_vocab_terms : ℕ
  vocab_distribution : Fin 4 → ℕ
  words_per_line : ℕ
  lines_per_page : ℕ
  pages_filled : ℚ

/-- Calculates the number of words left to write given a StoryProblem -/
def words_left_to_write (problem : StoryProblem) : ℕ :=
  problem.total_words - (problem.words_per_line * problem.lines_per_page * problem.pages_filled.num / problem.pages_filled.den).toNat

/-- Theorem stating that given the specific problem conditions, 100 words are left to write -/
theorem story_problem_solution (problem : StoryProblem) 
  (h1 : problem.total_words = 400)
  (h2 : problem.num_chapters = 4)
  (h3 : problem.total_vocab_terms = 20)
  (h4 : problem.vocab_distribution 0 = 8)
  (h5 : problem.vocab_distribution 1 = 4)
  (h6 : problem.vocab_distribution 2 = 6)
  (h7 : problem.vocab_distribution 3 = 2)
  (h8 : problem.words_per_line = 10)
  (h9 : problem.lines_per_page = 20)
  (h10 : problem.pages_filled = 3/2) :
  words_left_to_write problem = 100 := by
  sorry


end NUMINAMATH_CALUDE_story_problem_solution_l2878_287876


namespace NUMINAMATH_CALUDE_factorization_problem_1_l2878_287843

theorem factorization_problem_1 (a : ℝ) :
  3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l2878_287843


namespace NUMINAMATH_CALUDE_range_of_a_l2878_287883

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x | |x + a| < 1}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (B a ⊂ A) ∧ (B a ≠ A) → 0 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2878_287883


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2878_287850

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_incr : is_increasing_sequence a)
  (h_pos : a 1 > 0)
  (h_eq : ∀ n : ℕ, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2878_287850


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2878_287836

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

-- Define the derivative of f
noncomputable def f' (a b x : ℝ) : ℝ := -3*x^2 + 2*a*x + b

theorem cubic_function_properties (a b c : ℝ) :
  (f' a b (-1) = 0 ∧ f' a b 3 = 0) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x = 20 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a b c y ≤ 20) →
  (a = 3 ∧ b = 9) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x = 13 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a b c y ≥ 13) ∨
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x = -7 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a b c y ≥ -7) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2878_287836


namespace NUMINAMATH_CALUDE_organization_member_count_organization_has_ten_members_l2878_287898

/-- Represents an organization with committees and members -/
structure Organization :=
  (num_committees : ℕ)
  (num_members : ℕ)
  (member_committee_count : ℕ)
  (shared_member_count : ℕ)

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating the required number of members in the organization -/
theorem organization_member_count (org : Organization) 
  (h1 : org.num_committees = 5)
  (h2 : org.member_committee_count = 2)
  (h3 : org.shared_member_count = 1) :
  org.num_members = choose_two org.num_committees :=
by sorry

/-- The main theorem proving the organization must have 10 members -/
theorem organization_has_ten_members (org : Organization) 
  (h1 : org.num_committees = 5)
  (h2 : org.member_committee_count = 2)
  (h3 : org.shared_member_count = 1) :
  org.num_members = 10 :=
by sorry

end NUMINAMATH_CALUDE_organization_member_count_organization_has_ten_members_l2878_287898


namespace NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_sum_inequality_l2878_287892

theorem arithmetic_sequence_increasing_iff_sum_inequality 
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, a n = a 1 + (n - 1) * d) →
  (∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * d) →
  (∀ n : ℕ, n ≥ 2 → S n < n * a n) ↔ d > 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_sum_inequality_l2878_287892


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2878_287805

theorem infinite_series_sum : 
  ∑' (n : ℕ), (1 : ℝ) / (n * (n + 3)) = 1/3 + 1/6 + 1/9 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2878_287805


namespace NUMINAMATH_CALUDE_sqrt_product_equals_three_l2878_287859

theorem sqrt_product_equals_three : Real.sqrt (1/2) * Real.sqrt 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_three_l2878_287859


namespace NUMINAMATH_CALUDE_geometric_sequence_205th_term_l2878_287887

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

theorem geometric_sequence_205th_term :
  let a₁ : ℝ := 6
  let r : ℝ := -1
  geometric_sequence a₁ r 205 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_205th_term_l2878_287887


namespace NUMINAMATH_CALUDE_internet_service_duration_l2878_287862

/-- Calculates the number of days of internet service given the specified parameters. -/
def internetServiceDays (initialBalance : ℚ) (dailyCost : ℚ) (debtLimit : ℚ) (payment : ℚ) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that given the specified parameters, the number of days of internet service is 14. -/
theorem internet_service_duration :
  internetServiceDays 0 (1/2) 5 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_internet_service_duration_l2878_287862


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2878_287812

/-- Given two real numbers a and b that are inversely proportional,
    prove that if a + b = 30 and a - b = 8, then when a = 6, b = 209/6 -/
theorem inverse_proportion_problem (a b : ℝ) (h1 : ∃ k : ℝ, a * b = k) 
    (h2 : a + b = 30) (h3 : a - b = 8) : 
    (a = 6) → (b = 209 / 6) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2878_287812


namespace NUMINAMATH_CALUDE_mother_age_is_36_l2878_287827

/-- Petra's age -/
def petra_age : ℕ := 11

/-- The sum of Petra's and her mother's ages -/
def age_sum : ℕ := 47

/-- Petra's mother's age -/
def mother_age : ℕ := age_sum - petra_age

/-- Theorem: Petra's mother is 36 years old -/
theorem mother_age_is_36 : mother_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_mother_age_is_36_l2878_287827


namespace NUMINAMATH_CALUDE_integer_pair_property_l2878_287806

theorem integer_pair_property (a b : ℤ) :
  (∃ d : ℤ, d ≥ 2 ∧ ∀ n : ℕ, n > 0 → (d ∣ a^n + b^n + 1)) ↔
  ((a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0)) ∨
  ((a % 3 = 1 ∧ b % 3 = 1) ∨ (a % 3 = 2 ∧ b % 3 = 2)) ∨
  ((a % 6 = 1 ∧ b % 6 = 4) ∨ (a % 6 = 4 ∧ b % 6 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_property_l2878_287806


namespace NUMINAMATH_CALUDE_line_through_ellipse_focus_l2878_287889

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := 10 * x^2 + y^2 = 10

/-- The line equation -/
def line (x y b : ℝ) : Prop := 2 * x + b * y + 3 = 0

/-- Theorem: The value of b for a line passing through a focus of the given ellipse is either -1 or 1 -/
theorem line_through_ellipse_focus (b : ℝ) : 
  (∃ x y : ℝ, ellipse x y ∧ line x y b) → b = -1 ∨ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_ellipse_focus_l2878_287889


namespace NUMINAMATH_CALUDE_clothes_to_earnings_ratio_l2878_287848

/-- Proves that the ratio of clothes spending to initial earnings is 1:2 given the conditions --/
theorem clothes_to_earnings_ratio 
  (initial_earnings : ℚ)
  (clothes_spending : ℚ)
  (book_spending : ℚ)
  (remaining : ℚ)
  (h1 : initial_earnings = 600)
  (h2 : book_spending = (initial_earnings - clothes_spending) / 2)
  (h3 : remaining = initial_earnings - clothes_spending - book_spending)
  (h4 : remaining = 150) :
  clothes_spending / initial_earnings = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_clothes_to_earnings_ratio_l2878_287848


namespace NUMINAMATH_CALUDE_correct_calculation_l2878_287830

theorem correct_calculation : ∃ x : ℝ, 5 * x = 40 ∧ 2 * x = 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2878_287830


namespace NUMINAMATH_CALUDE_square_side_increase_l2878_287878

theorem square_side_increase (s : ℝ) (h : s > 0) :
  ∃ p : ℝ, (s * (1 + p / 100))^2 = 1.69 * s^2 → p = 30 := by
  sorry

end NUMINAMATH_CALUDE_square_side_increase_l2878_287878


namespace NUMINAMATH_CALUDE_linear_function_proof_l2878_287874

theorem linear_function_proof (k b : ℝ) :
  (1 * k + b = -2) →
  (-1 * k + b = -4) →
  (3 * k + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_proof_l2878_287874


namespace NUMINAMATH_CALUDE_average_b_c_l2878_287833

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 115)
  (h2 : a - c = 90) : 
  (b + c) / 2 = 70 := by
sorry

end NUMINAMATH_CALUDE_average_b_c_l2878_287833


namespace NUMINAMATH_CALUDE_oil_containers_per_box_l2878_287840

theorem oil_containers_per_box :
  let trucks_with_20_boxes : ℕ := 7
  let boxes_per_truck_20 : ℕ := 20
  let trucks_with_12_boxes : ℕ := 5
  let boxes_per_truck_12 : ℕ := 12
  let total_trucks_after_redistribution : ℕ := 10
  let containers_per_truck_after_redistribution : ℕ := 160

  let total_boxes : ℕ := trucks_with_20_boxes * boxes_per_truck_20 + trucks_with_12_boxes * boxes_per_truck_12
  let total_containers : ℕ := total_trucks_after_redistribution * containers_per_truck_after_redistribution

  (total_containers / total_boxes : ℚ) = 8 := by sorry

end NUMINAMATH_CALUDE_oil_containers_per_box_l2878_287840


namespace NUMINAMATH_CALUDE_dealer_net_profit_dealer_net_profit_is_97_20_l2878_287829

/-- Calculates the dealer's net profit from selling a desk --/
theorem dealer_net_profit (purchase_price : ℝ) (markup_rate : ℝ) (discount_rate : ℝ) 
  (sales_tax_rate : ℝ) (commission_rate : ℝ) : ℝ :=
  let selling_price := purchase_price / (1 - markup_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let total_payment := discounted_price * (1 + sales_tax_rate)
  let commission := discounted_price * commission_rate
  total_payment - purchase_price - commission

/-- Proves that the dealer's net profit is $97.20 under the given conditions --/
theorem dealer_net_profit_is_97_20 :
  dealer_net_profit 150 0.5 0.2 0.05 0.02 = 97.20 := by
  sorry

end NUMINAMATH_CALUDE_dealer_net_profit_dealer_net_profit_is_97_20_l2878_287829


namespace NUMINAMATH_CALUDE_tan_three_implies_sum_l2878_287844

theorem tan_three_implies_sum (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ + Real.sin θ / (1 + Real.cos θ) = 2 * (Real.sqrt 10 - 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_implies_sum_l2878_287844


namespace NUMINAMATH_CALUDE_no_real_roots_composition_l2878_287841

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem no_real_roots_composition (a b c : ℝ) :
  (∀ x : ℝ, QuadraticPolynomial a b c x ≠ x) →
  (∀ x : ℝ, QuadraticPolynomial a b c (QuadraticPolynomial a b c x) ≠ x) :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_composition_l2878_287841


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2878_287868

/-- The quadratic equation with coefficient m, (1/3), and 1 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 + (1/3) * x + 1 = 0

theorem quadratic_root_relation (m₁ m₂ x₁ x₂ x₃ x₄ : ℝ) :
  quadratic_equation m₁ x₁ →
  quadratic_equation m₁ x₂ →
  quadratic_equation m₂ x₃ →
  quadratic_equation m₂ x₄ →
  x₁ < x₃ →
  x₃ < x₄ →
  x₄ < x₂ →
  x₂ < 0 →
  m₂ > m₁ ∧ m₁ > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2878_287868


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2878_287816

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^3 + y^3 = 85/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2878_287816


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l2878_287835

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 18)
  (h_sum2 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l2878_287835


namespace NUMINAMATH_CALUDE_equation_solution_l2878_287866

theorem equation_solution : ∃ x : ℚ, (2*x - 1)/3 - (x - 2)/6 = 2 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2878_287866


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l2878_287870

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 34 →
  (n * original_mean - n * decrement) / n = 166 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l2878_287870


namespace NUMINAMATH_CALUDE_faye_age_l2878_287861

/-- Represents the ages of Diana, Eduardo, Chad, and Faye -/
structure Ages where
  diana : ℕ
  eduardo : ℕ
  chad : ℕ
  faye : ℕ

/-- Defines the age relationships between Diana, Eduardo, Chad, and Faye -/
def valid_ages (ages : Ages) : Prop :=
  ages.diana + 3 = ages.eduardo ∧
  ages.eduardo = ages.chad + 4 ∧
  ages.faye = ages.chad + 3 ∧
  ages.diana = 14

/-- Theorem stating that given the age relationships and Diana's age, Faye's age is 18 -/
theorem faye_age (ages : Ages) (h : valid_ages ages) : ages.faye = 18 := by
  sorry

end NUMINAMATH_CALUDE_faye_age_l2878_287861


namespace NUMINAMATH_CALUDE_expression_value_l2878_287858

theorem expression_value : 
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 1
  x^2 + y^2 - 2*z^2 + 3*x*y = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2878_287858


namespace NUMINAMATH_CALUDE_correct_lunch_bill_l2878_287856

/-- The cost of Sara's lunch items and the total bill -/
def lunch_bill (hotdog_cost salad_cost : ℚ) : Prop :=
  hotdog_cost = 5.36 ∧ salad_cost = 5.10 ∧ hotdog_cost + salad_cost = 10.46

/-- Theorem stating that the total lunch bill is correct -/
theorem correct_lunch_bill :
  ∃ (hotdog_cost salad_cost : ℚ), lunch_bill hotdog_cost salad_cost :=
sorry

end NUMINAMATH_CALUDE_correct_lunch_bill_l2878_287856


namespace NUMINAMATH_CALUDE_saltwater_animals_per_aquarium_l2878_287839

theorem saltwater_animals_per_aquarium :
  ∀ (num_aquariums : ℕ) (total_animals : ℕ) (animals_per_aquarium : ℕ),
    num_aquariums = 26 →
    total_animals = 52 →
    total_animals = num_aquariums * animals_per_aquarium →
    animals_per_aquarium = 2 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_animals_per_aquarium_l2878_287839


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l2878_287860

/-- The rate of a man rowing in still water, given his speeds with and against a stream. -/
theorem mans_rate_in_still_water
  (speed_with_stream : ℝ)
  (speed_against_stream : ℝ)
  (h_with : speed_with_stream = 20)
  (h_against : speed_against_stream = 4) :
  (speed_with_stream + speed_against_stream) / 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l2878_287860


namespace NUMINAMATH_CALUDE_noah_yearly_call_cost_l2878_287865

/-- The cost of Noah's calls to his Grammy for a year -/
def yearly_call_cost (calls_per_week : ℕ) (minutes_per_call : ℕ) (cost_per_minute : ℚ) (weeks_per_year : ℕ) : ℚ :=
  (calls_per_week * minutes_per_call * cost_per_minute * weeks_per_year : ℚ)

/-- Theorem stating that Noah's yearly call cost to his Grammy is $78 -/
theorem noah_yearly_call_cost :
  yearly_call_cost 1 30 (5/100) 52 = 78 := by
  sorry

end NUMINAMATH_CALUDE_noah_yearly_call_cost_l2878_287865


namespace NUMINAMATH_CALUDE_negation_of_implication_l2878_287863

theorem negation_of_implication (x : ℝ) :
  ¬(x > 2 → x^2 - 3*x + 2 > 0) ↔ (x ≤ 2 → x^2 - 3*x + 2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2878_287863


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_l2878_287819

-- Define the sets P and Q
def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {3, 4, 5}

-- Define the universal set U as the set of real numbers
def U : Type := ℝ

-- Theorem statement
theorem intersection_P_complement_Q :
  P ∩ (Set.univ \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_l2878_287819


namespace NUMINAMATH_CALUDE_water_bottle_problem_l2878_287881

def water_bottle_capacity (initial_capacity : ℝ) : Prop :=
  let remaining_after_first_drink := (3/4) * initial_capacity
  let remaining_after_second_drink := (1/3) * remaining_after_first_drink
  remaining_after_second_drink = 1

theorem water_bottle_problem : ∃ (c : ℝ), water_bottle_capacity c ∧ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_problem_l2878_287881


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2878_287838

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, foci at (-c, 0) and (c, 0),
    and an isosceles right triangle with hypotenuse connecting the foci,
    if the midpoint of the legs of this triangle lies on the hyperbola,
    then c/a = (√10 + √2)/2 -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (∃ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ∧ 
              x = -c/2 ∧ y = c/2) →
  c/a = (Real.sqrt 10 + Real.sqrt 2)/2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2878_287838


namespace NUMINAMATH_CALUDE_initial_distance_is_point_eight_l2878_287884

/-- Two boats moving towards each other with given speeds and a known distance before collision -/
structure BoatProblem where
  speed1 : ℝ  -- Speed of boat 1 in miles/hr
  speed2 : ℝ  -- Speed of boat 2 in miles/hr
  distance_before_collision : ℝ  -- Distance between boats 1 minute before collision in miles
  time_before_collision : ℝ  -- Time before collision in hours

/-- The initial distance between the boats given the problem parameters -/
def initial_distance (p : BoatProblem) : ℝ :=
  p.distance_before_collision + (p.speed1 + p.speed2) * p.time_before_collision

/-- Theorem stating that the initial distance is 0.8 miles given the specific problem conditions -/
theorem initial_distance_is_point_eight :
  let p : BoatProblem := {
    speed1 := 4,
    speed2 := 20,
    distance_before_collision := 0.4,
    time_before_collision := 1 / 60
  }
  initial_distance p = 0.8 := by sorry

end NUMINAMATH_CALUDE_initial_distance_is_point_eight_l2878_287884


namespace NUMINAMATH_CALUDE_pizza_distribution_l2878_287825

/-- Calculates the number of slices each person gets in a group pizza order -/
def slices_per_person (num_people : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  (num_pizzas * slices_per_pizza) / num_people

/-- Proves that given 18 people, 6 pizzas with 9 slices each, each person gets 3 slices -/
theorem pizza_distribution :
  slices_per_person 18 6 9 = 3 := by
  sorry

#eval slices_per_person 18 6 9

end NUMINAMATH_CALUDE_pizza_distribution_l2878_287825


namespace NUMINAMATH_CALUDE_no_maximum_b_plus_c_l2878_287842

/-- A cubic function f(x) = x^3 + bx^2 + cx + d -/
def cubic_function (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- The derivative of the cubic function -/
def cubic_derivative (b c : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*b*x + c

theorem no_maximum_b_plus_c :
  ∀ b c d : ℝ,
  (∀ x ∈ Set.Icc (-1) 2, cubic_derivative b c x ≤ 0) →
  ¬∃ M : ℝ, ∀ b' c' : ℝ, 
    (∀ x ∈ Set.Icc (-1) 2, cubic_derivative b' c' x ≤ 0) →
    b' + c' ≤ M :=
by sorry

end NUMINAMATH_CALUDE_no_maximum_b_plus_c_l2878_287842


namespace NUMINAMATH_CALUDE_bill_after_30_days_l2878_287820

/-- The amount owed after applying late charges -/
def amount_owed (initial_bill : ℝ) (late_charge_rate : ℝ) (days : ℕ) : ℝ :=
  initial_bill * (1 + late_charge_rate) ^ (days / 10)

/-- Theorem stating the amount owed after 30 days -/
theorem bill_after_30_days (initial_bill : ℝ) (late_charge_rate : ℝ) :
  initial_bill = 500 →
  late_charge_rate = 0.02 →
  amount_owed initial_bill late_charge_rate 30 = 530.604 :=
by
  sorry

#eval amount_owed 500 0.02 30

end NUMINAMATH_CALUDE_bill_after_30_days_l2878_287820


namespace NUMINAMATH_CALUDE_average_hours_worked_per_month_l2878_287808

def hours_per_day_april : ℕ := 6
def hours_per_day_june : ℕ := 5
def hours_per_day_september : ℕ := 8
def days_per_month : ℕ := 30
def num_months : ℕ := 3

theorem average_hours_worked_per_month :
  (hours_per_day_april * days_per_month +
   hours_per_day_june * days_per_month +
   hours_per_day_september * days_per_month) / num_months = 190 := by
  sorry

end NUMINAMATH_CALUDE_average_hours_worked_per_month_l2878_287808


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l2878_287815

/-- A geometric sequence with real terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  GeometricSequence a → a 2 = 9 → a 6 = 1 → a 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l2878_287815


namespace NUMINAMATH_CALUDE_evaluate_expression_l2878_287864

theorem evaluate_expression : 6 - 9 * (1 / 2 - 3^3) * 2 = 483 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2878_287864


namespace NUMINAMATH_CALUDE_volume_increase_when_radius_doubled_l2878_287832

/-- The volume increase of a right circular cylinder when its radius is doubled -/
theorem volume_increase_when_radius_doubled (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 6 → 
  π * (2*r)^2 * h - π * r^2 * h = 18 := by
  sorry

end NUMINAMATH_CALUDE_volume_increase_when_radius_doubled_l2878_287832


namespace NUMINAMATH_CALUDE_friend_age_order_l2878_287885

-- Define the set of friends
inductive Friend : Type
  | David : Friend
  | Emma : Friend
  | Fiona : Friend

-- Define the age ordering relation
def AgeOrder : Friend → Friend → Prop := sorry

-- Define the property of being the oldest
def IsOldest (f : Friend) : Prop := ∀ g : Friend, g ≠ f → AgeOrder f g

-- Define the property of being the youngest
def IsYoungest (f : Friend) : Prop := ∀ g : Friend, g ≠ f → AgeOrder g f

-- State the theorem
theorem friend_age_order :
  -- Exactly one of the following statements is true
  (IsOldest Friend.Emma ∧ ¬IsYoungest Friend.Fiona ∧ IsOldest Friend.David) ∨
  (¬IsOldest Friend.Emma ∧ IsYoungest Friend.Fiona ∧ IsOldest Friend.David) ∨
  (¬IsOldest Friend.Emma ∧ ¬IsYoungest Friend.Fiona ∧ ¬IsOldest Friend.David) →
  -- The age order is David (oldest), Emma (middle), Fiona (youngest)
  AgeOrder Friend.David Friend.Emma ∧ AgeOrder Friend.Emma Friend.Fiona :=
by sorry

end NUMINAMATH_CALUDE_friend_age_order_l2878_287885


namespace NUMINAMATH_CALUDE_green_face_prob_five_eighths_l2878_287818

/-- A regular octahedron with colored faces -/
structure ColoredOctahedron where
  green_faces : ℕ
  purple_faces : ℕ
  total_faces : ℕ
  face_sum : green_faces + purple_faces = total_faces
  is_octahedron : total_faces = 8

/-- The probability of rolling a green face on a colored octahedron -/
def green_face_probability (o : ColoredOctahedron) : ℚ :=
  o.green_faces / o.total_faces

/-- Theorem: The probability of rolling a green face on a regular octahedron 
    with 5 green faces and 3 purple faces is 5/8 -/
theorem green_face_prob_five_eighths :
  ∀ (o : ColoredOctahedron), 
    o.green_faces = 5 → 
    o.purple_faces = 3 → 
    green_face_probability o = 5/8 :=
by
  sorry

end NUMINAMATH_CALUDE_green_face_prob_five_eighths_l2878_287818


namespace NUMINAMATH_CALUDE_simplify_expression_l2878_287895

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (x*y*z)⁻¹ * (x + y + z)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2878_287895


namespace NUMINAMATH_CALUDE_soccer_lineup_combinations_l2878_287828

def total_players : ℕ := 16
def rookie_players : ℕ := 4
def goalkeeper_count : ℕ := 1
def defender_count : ℕ := 4
def midfielder_count : ℕ := 4
def forward_count : ℕ := 3

def lineup_combinations : ℕ := 
  total_players * 
  (Nat.choose (total_players - 1) defender_count) * 
  (Nat.choose (total_players - 1 - defender_count) midfielder_count) * 
  (Nat.choose rookie_players 2 * Nat.choose (total_players - rookie_players - goalkeeper_count - defender_count - midfielder_count) 1)

theorem soccer_lineup_combinations : 
  lineup_combinations = 21508800 := by sorry

end NUMINAMATH_CALUDE_soccer_lineup_combinations_l2878_287828


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2878_287803

theorem quadratic_roots_relation (a b n r s : ℝ) : 
  (a^2 - n*a + 3 = 0) →
  (b^2 - n*b + 3 = 0) →
  ((a + 1/b)^2 - r*(a + 1/b) + s = 0) →
  ((b + 1/a)^2 - r*(b + 1/a) + s = 0) →
  s = 16/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2878_287803


namespace NUMINAMATH_CALUDE_max_distance_line_l2878_287875

/-- The point through which the line passes -/
def point : ℝ × ℝ := (1, 2)

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := x + 2*y - 5 = 0

/-- Theorem stating that the given line equation represents the line with maximum distance from the origin passing through the specified point -/
theorem max_distance_line :
  line_equation point.1 point.2 ∧
  ∀ (a b c : ℝ), (a*point.1 + b*point.2 + c = 0) →
    (a^2 + b^2 ≤ 1^2 + 2^2) :=
sorry

end NUMINAMATH_CALUDE_max_distance_line_l2878_287875


namespace NUMINAMATH_CALUDE_unique_prime_satisfying_conditions_l2878_287807

theorem unique_prime_satisfying_conditions :
  ∃! (n : ℕ), n.Prime ∧ 
    (n^2 + 10).Prime ∧ 
    (n^2 - 2).Prime ∧ 
    (n^3 + 6).Prime ∧ 
    (n^5 + 36).Prime ∧ 
    n = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_satisfying_conditions_l2878_287807


namespace NUMINAMATH_CALUDE_find_m_l2878_287810

theorem find_m (w x y z m : ℝ) 
  (h : 9 / (w + x + y) = m / (w + z) ∧ m / (w + z) = 15 / (z - x - y)) : 
  m = 24 := by
sorry

end NUMINAMATH_CALUDE_find_m_l2878_287810


namespace NUMINAMATH_CALUDE_power_plus_mod_five_l2878_287847

theorem power_plus_mod_five : (2^2018 + 2019) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_plus_mod_five_l2878_287847


namespace NUMINAMATH_CALUDE_russian_pairing_probability_l2878_287831

def total_players : ℕ := 10
def russian_players : ℕ := 4

theorem russian_pairing_probability :
  let remaining_players := total_players - 1
  let remaining_russian_players := russian_players - 1
  let first_pair_prob := remaining_russian_players / remaining_players
  let second_pair_prob := 1 / (remaining_players - 1)
  first_pair_prob * second_pair_prob = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_russian_pairing_probability_l2878_287831


namespace NUMINAMATH_CALUDE_number_problem_l2878_287817

theorem number_problem (x : ℝ) : (0.5 * x - 10 = 25) → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2878_287817


namespace NUMINAMATH_CALUDE_bank_coins_l2878_287897

/-- Given a total of 11 coins, including 2 dimes and 2 nickels, prove that the number of quarters is 7. -/
theorem bank_coins (total : ℕ) (dimes : ℕ) (nickels : ℕ) (quarters : ℕ)
  (h_total : total = 11)
  (h_dimes : dimes = 2)
  (h_nickels : nickels = 2)
  (h_sum : total = dimes + nickels + quarters) :
  quarters = 7 := by
  sorry

end NUMINAMATH_CALUDE_bank_coins_l2878_287897


namespace NUMINAMATH_CALUDE_concert_ticket_price_l2878_287855

theorem concert_ticket_price 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (total_revenue : ℚ) :
  child_price = adult_price / 2 →
  num_adults = 183 →
  num_children = 28 →
  total_revenue = 5122 →
  num_adults * adult_price + num_children * child_price = total_revenue →
  adult_price = 26 := by
sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l2878_287855
