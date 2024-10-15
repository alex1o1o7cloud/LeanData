import Mathlib

namespace NUMINAMATH_CALUDE_particle_position_after_3045_minutes_l3571_357151

/-- Represents the position of a particle -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Calculates the time taken for n rectangles -/
def timeForNRectangles (n : ℕ) : ℕ :=
  (n + 1)^2 - 1

/-- Calculates the position after n complete rectangles -/
def positionAfterNRectangles (n : ℕ) : Position :=
  if n % 2 = 0 then
    ⟨0, n⟩
  else
    ⟨0, n⟩

/-- Calculates the final position after given time -/
def finalPosition (time : ℕ) : Position :=
  let n := (Nat.sqrt (time + 1) : ℕ) - 1
  let remainingTime := time - timeForNRectangles n
  let basePosition := positionAfterNRectangles n
  if n % 2 = 0 then
    ⟨basePosition.x + remainingTime, basePosition.y⟩
  else
    ⟨basePosition.x + remainingTime, basePosition.y⟩

theorem particle_position_after_3045_minutes :
  finalPosition 3045 = ⟨21, 54⟩ := by
  sorry


end NUMINAMATH_CALUDE_particle_position_after_3045_minutes_l3571_357151


namespace NUMINAMATH_CALUDE_constant_for_max_n_l3571_357134

theorem constant_for_max_n (c : ℝ) : 
  (∀ n : ℤ, c * n^2 ≤ 3600) ∧ 
  (∃ n : ℤ, n > 5 ∧ c * n^2 > 3600) ∧
  c * 5^2 ≤ 3600 →
  c = 144 := by sorry

end NUMINAMATH_CALUDE_constant_for_max_n_l3571_357134


namespace NUMINAMATH_CALUDE_halfway_fraction_l3571_357165

theorem halfway_fraction : 
  let a := (1 : ℚ) / 2
  let b := (3 : ℚ) / 4
  (a + b) / 2 = (5 : ℚ) / 8 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l3571_357165


namespace NUMINAMATH_CALUDE_secretary_project_time_l3571_357101

theorem secretary_project_time (t1 t2 t3 : ℕ) : 
  t1 + t2 + t3 = 120 →
  t2 = 2 * t1 →
  t3 = 5 * t1 →
  t3 = 75 := by
sorry

end NUMINAMATH_CALUDE_secretary_project_time_l3571_357101


namespace NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l3571_357118

theorem max_area_inscribed_rectangle (d : ℝ) (h : d = 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → x^2 + y^2 = d^2 → x * y ≤ d^2 / 2 :=
by
  sorry

#check max_area_inscribed_rectangle

end NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l3571_357118


namespace NUMINAMATH_CALUDE_johns_cloth_cost_l3571_357105

/-- The total cost of cloth purchased by John -/
def total_cost (length : ℝ) (price_per_metre : ℝ) : ℝ :=
  length * price_per_metre

/-- Theorem stating the total cost of John's cloth purchase -/
theorem johns_cloth_cost :
  let length : ℝ := 9.25
  let price_per_metre : ℝ := 46
  total_cost length price_per_metre = 425.50 := by
  sorry

end NUMINAMATH_CALUDE_johns_cloth_cost_l3571_357105


namespace NUMINAMATH_CALUDE_rent_increase_problem_l3571_357199

theorem rent_increase_problem (initial_average : ℝ) (new_average : ℝ) (num_friends : ℕ) 
  (increase_percentage : ℝ) (h1 : initial_average = 800) (h2 : new_average = 880) 
  (h3 : num_friends = 4) (h4 : increase_percentage = 0.2) : 
  ∃ (original_rent : ℝ), 
    (num_friends * new_average - num_friends * initial_average) / increase_percentage = original_rent ∧ 
    original_rent = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_problem_l3571_357199


namespace NUMINAMATH_CALUDE_expression_can_be_any_real_l3571_357150

theorem expression_can_be_any_real (x : ℝ) : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  a + b + c = 1 ∧ 
  (a^4 + b^4 + c^4) / (a*b + b*c + c*a) = x :=
sorry

end NUMINAMATH_CALUDE_expression_can_be_any_real_l3571_357150


namespace NUMINAMATH_CALUDE_velocity_zero_at_two_l3571_357196

-- Define the displacement function
def s (t : ℝ) : ℝ := -2 * t^2 + 8 * t

-- Define the velocity function (derivative of displacement)
def v (t : ℝ) : ℝ := -4 * t + 8

-- Theorem: The time when velocity is 0 is equal to 2
theorem velocity_zero_at_two :
  ∃ t : ℝ, v t = 0 ∧ t = 2 :=
sorry

end NUMINAMATH_CALUDE_velocity_zero_at_two_l3571_357196


namespace NUMINAMATH_CALUDE_bricks_to_paint_theorem_l3571_357126

/-- Represents a stack of bricks -/
structure BrickStack :=
  (height : ℕ)
  (width : ℕ)
  (depth : ℕ)
  (total_bricks : ℕ)
  (sides_against_wall : ℕ)

/-- Calculates the number of bricks that need to be painted on their exposed surfaces -/
def bricks_to_paint (stack : BrickStack) : ℕ :=
  let front_face := stack.height * stack.width + stack.depth
  let top_face := stack.width * stack.depth
  front_face * stack.height + top_face * (4 - stack.sides_against_wall)

theorem bricks_to_paint_theorem (stack : BrickStack) :
  stack.height = 4 ∧ 
  stack.width = 3 ∧ 
  stack.depth = 15 ∧ 
  stack.total_bricks = 180 ∧ 
  stack.sides_against_wall = 2 →
  bricks_to_paint stack = 96 :=
by sorry

end NUMINAMATH_CALUDE_bricks_to_paint_theorem_l3571_357126


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3571_357170

theorem polynomial_expansion (z : ℂ) : 
  (3 * z^3 + 4 * z^2 - 5 * z + 1) * (2 * z^2 - 3 * z + 4) * (z - 1) = 
  3 * z^6 - 3 * z^5 + 5 * z^4 + 2 * z^3 - 5 * z^2 + 4 * z - 16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3571_357170


namespace NUMINAMATH_CALUDE_octagon_area_l3571_357184

theorem octagon_area (r : ℝ) (h : r = 3) : 
  let s := 2 * r * Real.sin (π / 8)
  let triangle_area := (1 / 2) * s^2 * Real.sin (π / 4)
  8 * triangle_area = 8 * (1 / 2) * (6 * Real.sin (π / 8))^2 * Real.sin (π / 4) := by
sorry

end NUMINAMATH_CALUDE_octagon_area_l3571_357184


namespace NUMINAMATH_CALUDE_digit_sum_equals_78331_l3571_357155

/-- A function that generates all possible natural numbers from a given list of digits,
    where each digit can be used no more than once. -/
def generateNumbers (digits : List Nat) : List Nat :=
  sorry

/-- The sum of all numbers generated from the digits 2, 0, 1, 8. -/
def digitSum : Nat :=
  (generateNumbers [2, 0, 1, 8]).sum

/-- Theorem stating that the sum of all possible natural numbers formed from digits 2, 0, 1, 8,
    where each digit is used no more than once, is equal to 78331. -/
theorem digit_sum_equals_78331 : digitSum = 78331 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_equals_78331_l3571_357155


namespace NUMINAMATH_CALUDE_tank_full_time_l3571_357189

/-- Represents the state of a water tank system -/
structure TankSystem where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℤ

/-- Calculates the time needed to fill the tank -/
def time_to_fill (system : TankSystem) : ℕ :=
  sorry

/-- Theorem stating that the tank will be full after 56 minutes -/
theorem tank_full_time (system : TankSystem) 
  (h1 : system.capacity = 950)
  (h2 : system.pipeA_rate = 40)
  (h3 : system.pipeB_rate = 30)
  (h4 : system.pipeC_rate = -20) : 
  time_to_fill system = 56 :=
  sorry

end NUMINAMATH_CALUDE_tank_full_time_l3571_357189


namespace NUMINAMATH_CALUDE_number_operation_l3571_357180

theorem number_operation (x : ℝ) : (x - 5) / 7 = 7 → (x - 34) / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_l3571_357180


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3571_357147

theorem quadratic_no_real_roots (c : ℤ) : 
  c < 3 → 
  (∀ x : ℝ, x^2 + 2*x + c ≠ 0) → 
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3571_357147


namespace NUMINAMATH_CALUDE_square_sum_inequality_square_sum_equality_l3571_357164

theorem square_sum_inequality {a b c d : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
by sorry

theorem square_sum_equality {a b c d : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_square_sum_inequality_square_sum_equality_l3571_357164


namespace NUMINAMATH_CALUDE_volume_of_specific_open_box_l3571_357148

/-- Calculates the volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
def openBoxVolume (sheetLength sheetWidth cutLength : ℝ) : ℝ :=
  (sheetLength - 2 * cutLength) * (sheetWidth - 2 * cutLength) * cutLength

/-- Theorem stating that the volume of the open box is 5440 m³ given the specified dimensions. -/
theorem volume_of_specific_open_box :
  openBoxVolume 50 36 8 = 5440 := by
  sorry

#eval openBoxVolume 50 36 8

end NUMINAMATH_CALUDE_volume_of_specific_open_box_l3571_357148


namespace NUMINAMATH_CALUDE_base_prime_441_l3571_357142

/-- Definition of base prime representation for a natural number -/
def basePrimeRepresentation (n : ℕ) : List ℕ :=
  sorry

/-- Theorem stating that the base prime representation of 441 is [0, 2, 2, 0] -/
theorem base_prime_441 : basePrimeRepresentation 441 = [0, 2, 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_441_l3571_357142


namespace NUMINAMATH_CALUDE_valid_grid_probability_l3571_357131

/-- Represents a 3x3 grid filled with numbers 1 to 9 --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if a number is odd --/
def isOdd (n : Fin 9) : Bool :=
  n.val % 2 ≠ 0

/-- Checks if the sum of numbers in a row is odd --/
def isRowSumOdd (g : Grid) (row : Fin 3) : Bool :=
  isOdd (g row 0 + g row 1 + g row 2)

/-- Checks if the sum of numbers in a column is odd --/
def isColumnSumOdd (g : Grid) (col : Fin 3) : Bool :=
  isOdd (g 0 col + g 1 col + g 2 col)

/-- Checks if all rows and columns have odd sums --/
def isValidGrid (g : Grid) : Bool :=
  (∀ row, isRowSumOdd g row) ∧ (∀ col, isColumnSumOdd g col)

/-- The total number of possible 3x3 grids filled with numbers 1 to 9 --/
def totalGrids : Nat :=
  Nat.factorial 9

/-- The number of valid grids where all rows and columns have odd sums --/
def validGrids : Nat :=
  9

/-- The main theorem stating the probability of a valid grid --/
theorem valid_grid_probability :
  (validGrids : ℚ) / totalGrids = 1 / 14 :=
sorry

end NUMINAMATH_CALUDE_valid_grid_probability_l3571_357131


namespace NUMINAMATH_CALUDE_same_color_probability_l3571_357127

def total_plates : ℕ := 11
def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def selected_plates : ℕ := 3

theorem same_color_probability : 
  (Nat.choose red_plates selected_plates : ℚ) / (Nat.choose total_plates selected_plates) = 4 / 33 := by
sorry

end NUMINAMATH_CALUDE_same_color_probability_l3571_357127


namespace NUMINAMATH_CALUDE_average_brown_mms_l3571_357157

def brown_mms : List Nat := [9, 12, 8, 8, 3]

theorem average_brown_mms :
  (brown_mms.sum / brown_mms.length : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_brown_mms_l3571_357157


namespace NUMINAMATH_CALUDE_x_equals_one_ninth_l3571_357102

theorem x_equals_one_ninth (x : ℚ) (h : x - 1/10 = x/10) : x = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_ninth_l3571_357102


namespace NUMINAMATH_CALUDE_part_one_part_two_l3571_357136

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}

-- Define set C with parameter a
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

-- Theorem for part (1)
theorem part_one :
  B = {x : ℝ | x ≥ 2} ∧
  (A ∩ B)ᶜ = {x : ℝ | x < 2 ∨ x ≥ 3} :=
sorry

-- Theorem for part (2)
theorem part_two :
  {a : ℝ | ∀ x, x ∈ B → x ∈ C a} = {a : ℝ | a > -4} :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3571_357136


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l3571_357146

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x, x > a → (x - 1) / x > 0) → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l3571_357146


namespace NUMINAMATH_CALUDE_picture_coverage_percentage_l3571_357178

theorem picture_coverage_percentage (poster_width poster_height picture_width picture_height : ℝ) 
  (hw_poster : poster_width = 50 ∧ poster_height = 100)
  (hw_picture : picture_width = 20 ∧ picture_height = 40) :
  (picture_width * picture_height) / (poster_width * poster_height) * 100 = 16 := by
  sorry

end NUMINAMATH_CALUDE_picture_coverage_percentage_l3571_357178


namespace NUMINAMATH_CALUDE_chicken_fried_steak_cost_l3571_357122

theorem chicken_fried_steak_cost (steak_egg_cost : ℝ) (james_payment : ℝ) 
  (tip_percentage : ℝ) (chicken_fried_steak_cost : ℝ) :
  steak_egg_cost = 16 →
  james_payment = 21 →
  tip_percentage = 0.20 →
  james_payment = (steak_egg_cost + chicken_fried_steak_cost) / 2 + 
    tip_percentage * (steak_egg_cost + chicken_fried_steak_cost) →
  chicken_fried_steak_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_chicken_fried_steak_cost_l3571_357122


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3571_357138

open Set

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = Ioo (-1 : ℝ) 1 ∪ {1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3571_357138


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l3571_357161

theorem real_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  (Complex.re ((1 : ℂ) + i) / i) = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l3571_357161


namespace NUMINAMATH_CALUDE_product_of_powers_l3571_357163

theorem product_of_powers : 3^2 * 5^2 * 7 * 11^2 = 190575 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_l3571_357163


namespace NUMINAMATH_CALUDE_middle_number_proof_l3571_357198

theorem middle_number_proof (a b c : ℤ) : 
  a < b ∧ b < c ∧ 
  a + b = 18 ∧ 
  a + c = 23 ∧ 
  b + c = 27 → 
  b = 11 := by
sorry

end NUMINAMATH_CALUDE_middle_number_proof_l3571_357198


namespace NUMINAMATH_CALUDE_coffee_consumption_l3571_357173

def vacation_duration : ℕ := 40
def pods_per_box : ℕ := 30
def cost_per_box : ℚ := 8
def total_spent : ℚ := 32

def cups_per_day : ℚ := total_spent / cost_per_box * pods_per_box / vacation_duration

theorem coffee_consumption : cups_per_day = 3 := by sorry

end NUMINAMATH_CALUDE_coffee_consumption_l3571_357173


namespace NUMINAMATH_CALUDE_product_equals_10000_l3571_357115

theorem product_equals_10000 : ∃ x : ℕ, 469160 * x = 4691130840 ∧ x = 10000 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_10000_l3571_357115


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_min_value_achieved_l3571_357168

/-- The function f(x) = |2x-1| - m -/
def f (x m : ℝ) : ℝ := |2*x - 1| - m

/-- The theorem stating the minimum value of a + b -/
theorem min_value_a_plus_b (m : ℝ) (h1 : Set.Icc 0 1 = {x | f x m ≤ 0}) 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h2 : 1/a + 1/(2*b) = m) : 
  a + b ≥ 3/2 + Real.sqrt 2 := by
  sorry

/-- The theorem stating that the minimum value is achieved -/
theorem min_value_achieved (m : ℝ) (h1 : Set.Icc 0 1 = {x | f x m ≤ 0}) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1/a + 1/(2*b) = m ∧ a + b = 3/2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_min_value_achieved_l3571_357168


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3571_357195

theorem purely_imaginary_complex_number (m : ℝ) :
  (3 * m ^ 2 - 8 * m - 3 : ℂ) + (m ^ 2 - 4 * m + 3 : ℂ) * Complex.I = Complex.I * ((m ^ 2 - 4 * m + 3 : ℝ) : ℂ) →
  m = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3571_357195


namespace NUMINAMATH_CALUDE_f_properties_l3571_357140

noncomputable def f : ℝ → ℝ := fun x => if x ≤ 0 then x^2 + 2*x else x^2 - 2*x

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x : ℝ, x > 0 → f x = x^2 - 2*x) ∧
  (StrictMonoOn f (Set.Ioo (-1) 0)) ∧
  (StrictMonoOn f (Set.Ioi 1)) ∧
  (StrictAntiOn f (Set.Iic (-1))) ∧
  (StrictAntiOn f (Set.Ioo 0 1)) ∧
  (Set.range f = Set.Ici (-1)) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l3571_357140


namespace NUMINAMATH_CALUDE_initial_water_amount_l3571_357133

/-- Given a bucket of water, prove that the initial amount is 0.8 gallons when 0.2 gallons are poured out and 0.6 gallons remain. -/
theorem initial_water_amount (poured_out : ℝ) (remaining : ℝ) : poured_out = 0.2 → remaining = 0.6 → poured_out + remaining = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_amount_l3571_357133


namespace NUMINAMATH_CALUDE_set_equality_l3571_357113

open Set Real

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | Real.log (x - 2) ≤ 0}

theorem set_equality : (Aᶜ ∪ B) = Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_set_equality_l3571_357113


namespace NUMINAMATH_CALUDE_salary_increase_l3571_357153

theorem salary_increase (last_year_salary : ℝ) (last_year_savings_rate : ℝ) 
  (this_year_savings_rate : ℝ) (salary_increase_rate : ℝ) :
  last_year_savings_rate = 0.06 →
  this_year_savings_rate = 0.05 →
  this_year_savings_rate * (last_year_salary * (1 + salary_increase_rate)) = 
    last_year_savings_rate * last_year_salary →
  salary_increase_rate = 0.2 := by
  sorry

#check salary_increase

end NUMINAMATH_CALUDE_salary_increase_l3571_357153


namespace NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l3571_357141

theorem sqrt_product_equals_sqrt_of_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l3571_357141


namespace NUMINAMATH_CALUDE_directory_page_numbering_l3571_357191

/-- Calculate the total number of digits needed to number pages in a directory --/
def totalDigits (totalPages : ℕ) : ℕ :=
  let singleDigitPages := min totalPages 9
  let doubleDigitPages := min (max (totalPages - 9) 0) 90
  let tripleDigitPages := max (totalPages - 99) 0
  singleDigitPages * 1 + doubleDigitPages * 2 + tripleDigitPages * 3

/-- Theorem: A directory with 710 pages requires 2022 digits to number all pages --/
theorem directory_page_numbering :
  totalDigits 710 = 2022 := by sorry

end NUMINAMATH_CALUDE_directory_page_numbering_l3571_357191


namespace NUMINAMATH_CALUDE_problem_statement_l3571_357171

theorem problem_statement (x y P Q : ℝ) 
  (h1 : x + y = P)
  (h2 : x^2 + y^2 = Q)
  (h3 : x^3 + y^3 = P^2) :
  Q = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3571_357171


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l3571_357188

theorem remainder_of_large_number (M : ℕ) (d : ℕ) (h : M = 123456789012 ∧ d = 252) :
  M % d = 228 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l3571_357188


namespace NUMINAMATH_CALUDE_john_buys_three_boxes_l3571_357185

/-- The number of times John plays paintball per month. -/
def plays_per_month : ℕ := 3

/-- The cost of one box of paintballs in dollars. -/
def cost_per_box : ℕ := 25

/-- The total amount John spends on paintballs per month in dollars. -/
def monthly_spending : ℕ := 225

/-- The number of boxes of paintballs John buys each time he plays. -/
def boxes_per_play : ℚ := monthly_spending / (plays_per_month * cost_per_box)

theorem john_buys_three_boxes : boxes_per_play = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_buys_three_boxes_l3571_357185


namespace NUMINAMATH_CALUDE_function_inequalities_l3571_357179

theorem function_inequalities (p q r s : ℝ) (h : p * s - q * r < 0) :
  let f := fun x => (p * x + q) / (r * x + s)
  ∀ x₁ x₂ ε : ℝ,
    ε > 0 →
    (x₁ < x₂ ∧ x₂ < -s/r → f x₁ > f x₂) ∧
    (-s/r < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
    (x₁ < x₂ ∧ x₂ < -s/r → f (x₁ - ε) - f x₁ < f (x₂ - ε) - f x₂) ∧
    (-s/r < x₁ ∧ x₁ < x₂ → f x₁ - f (x₁ + ε) > f x₂ - f (x₂ + ε)) :=
by sorry

end NUMINAMATH_CALUDE_function_inequalities_l3571_357179


namespace NUMINAMATH_CALUDE_eagles_score_l3571_357192

/-- Given the total points and margin of victory in a basketball game, prove the losing team's score. -/
theorem eagles_score (total_points margin : ℕ) (h1 : total_points = 82) (h2 : margin = 18) :
  (total_points - margin) / 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_eagles_score_l3571_357192


namespace NUMINAMATH_CALUDE_ball_returns_after_15_throws_l3571_357103

/-- Represents the number of girls to skip in each throw -/
def skip_pattern : ℕ → ℕ
  | n => if n % 2 = 0 then 3 else 4

/-- Calculates the position of the girl who receives the ball after n throws -/
def ball_position (n : ℕ) : Fin 15 :=
  (List.range n).foldl (fun pos _ => 
    (pos + skip_pattern pos + 1 : Fin 15)) 0

theorem ball_returns_after_15_throws :
  ball_position 15 = 0 := by sorry

end NUMINAMATH_CALUDE_ball_returns_after_15_throws_l3571_357103


namespace NUMINAMATH_CALUDE_lines_are_parallel_l3571_357114

-- Define the lines
def line1 (a : ℝ) (θ : ℝ) : Prop := θ = a
def line2 (p a θ : ℝ) : Prop := p * Real.sin (θ - a) = 1

-- Theorem statement
theorem lines_are_parallel (a p : ℝ) : 
  ∀ θ, ¬(line1 a θ ∧ line2 p a θ) :=
sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l3571_357114


namespace NUMINAMATH_CALUDE_greatest_possible_award_l3571_357166

theorem greatest_possible_award (total_prize : ℕ) (num_winners : ℕ) (min_award : ℕ) :
  total_prize = 600 →
  num_winners = 15 →
  min_award = 15 →
  (2 : ℚ) / 5 * total_prize = (3 : ℚ) / 5 * num_winners * min_award →
  ∃ (max_award : ℕ), max_award = 390 ∧
    max_award ≤ total_prize ∧
    max_award ≥ min_award ∧
    ∃ (other_awards : List ℕ),
      other_awards.length = num_winners - 1 ∧
      (∀ x ∈ other_awards, min_award ≤ x) ∧
      max_award + other_awards.sum = total_prize :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_award_l3571_357166


namespace NUMINAMATH_CALUDE_chess_tournament_games_14_l3571_357117

/-- The number of games played in a chess tournament -/
def chess_tournament_games (n : ℕ) : ℕ := n.choose 2

/-- Theorem: In a chess tournament with 14 players where each player plays every other player once,
    the total number of games played is 91. -/
theorem chess_tournament_games_14 :
  chess_tournament_games 14 = 91 := by
  sorry

#eval chess_tournament_games 14  -- This should output 91

end NUMINAMATH_CALUDE_chess_tournament_games_14_l3571_357117


namespace NUMINAMATH_CALUDE_wednesday_sales_proof_l3571_357182

def initial_stock : ℕ := 1300
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135
def unsold_percentage : ℚ := 69.07692307692308

theorem wednesday_sales_proof :
  ∃ (wednesday_sales : ℕ),
    (initial_stock : ℚ) * (1 - unsold_percentage / 100) =
    (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales : ℚ) ∧
    wednesday_sales = 64 := by sorry

end NUMINAMATH_CALUDE_wednesday_sales_proof_l3571_357182


namespace NUMINAMATH_CALUDE_tens_digit_of_6_pow_19_l3571_357183

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem
theorem tens_digit_of_6_pow_19 : tens_digit (6^19) = 9 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_pow_19_l3571_357183


namespace NUMINAMATH_CALUDE_third_sibling_age_difference_l3571_357193

/-- Represents the ages of four siblings --/
structure SiblingAges where
  youngest : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ

/-- The conditions of the sibling age problem --/
def siblingAgeProblem (ages : SiblingAges) : Prop :=
  ages.youngest = 25.75 ∧
  ages.second = ages.youngest + 3 ∧
  ages.third = ages.youngest + 6 ∧
  (ages.youngest + ages.second + ages.third + ages.fourth) / 4 = 30

/-- The theorem stating that the third sibling is 6 years older than the youngest --/
theorem third_sibling_age_difference (ages : SiblingAges) :
  siblingAgeProblem ages → ages.third - ages.youngest = 6 := by
  sorry

end NUMINAMATH_CALUDE_third_sibling_age_difference_l3571_357193


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l3571_357167

theorem quadratic_root_ratio (c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (5 * x₁^2 - 2 * x₁ + c = 0) ∧ 
    (5 * x₂^2 - 2 * x₂ + c = 0) ∧ 
    (x₁ / x₂ = -3/5)) → 
  c = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l3571_357167


namespace NUMINAMATH_CALUDE_two_colonies_limit_time_l3571_357110

/-- Represents the growth of a bacteria colony -/
structure BacteriaColony where
  initialSize : ℕ
  doubleTime : ℕ
  habitatLimit : ℕ

/-- The time it takes for a single colony to reach the habitat limit -/
def singleColonyLimitTime (colony : BacteriaColony) : ℕ := 20

/-- The size of a colony after a given number of days -/
def colonySize (colony : BacteriaColony) (days : ℕ) : ℕ :=
  colony.initialSize * 2^days

/-- Predicate to check if a colony has reached the habitat limit -/
def hasReachedLimit (colony : BacteriaColony) (days : ℕ) : Prop :=
  colonySize colony days ≥ colony.habitatLimit

/-- Theorem: Two colonies reach the habitat limit in the same time as a single colony -/
theorem two_colonies_limit_time (colony1 colony2 : BacteriaColony) :
  (∃ t : ℕ, hasReachedLimit colony1 t ∧ hasReachedLimit colony2 t) →
  (∃ t : ℕ, t = singleColonyLimitTime colony1 ∧ hasReachedLimit colony1 t ∧ hasReachedLimit colony2 t) :=
sorry

end NUMINAMATH_CALUDE_two_colonies_limit_time_l3571_357110


namespace NUMINAMATH_CALUDE_carrot_count_l3571_357139

theorem carrot_count (olivia_carrots : ℕ) (mom_carrots : ℕ) : 
  olivia_carrots = 20 → mom_carrots = 14 → olivia_carrots + mom_carrots = 34 := by
sorry

end NUMINAMATH_CALUDE_carrot_count_l3571_357139


namespace NUMINAMATH_CALUDE_trig_identity_l3571_357187

theorem trig_identity (θ : Real) (h : θ ≠ 0) (h2 : θ ≠ π/2) : 
  (Real.tan θ)^2 - (Real.sin θ)^2 = (Real.tan θ)^2 * (Real.sin θ)^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3571_357187


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l3571_357128

theorem smallest_sum_of_squares (x y : ℕ) : x^2 - y^2 = 221 → x^2 + y^2 ≥ 229 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l3571_357128


namespace NUMINAMATH_CALUDE_range_of_a_l3571_357159

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : is_monotone_decreasing f (-2) 4)
  (h2 : f (a + 1) > f (2 * a)) :
  1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3571_357159


namespace NUMINAMATH_CALUDE_payment_difference_l3571_357107

/-- Represents the pizza with its properties and how it was shared -/
structure Pizza :=
  (total_slices : ℕ)
  (plain_cost : ℚ)
  (pepperoni_cost : ℚ)
  (mushroom_cost : ℚ)
  (bob_slices : ℕ)
  (charlie_slices : ℕ)
  (alice_slices : ℕ)

/-- Calculates the total cost of the pizza -/
def total_cost (p : Pizza) : ℚ :=
  p.plain_cost + p.pepperoni_cost + p.mushroom_cost

/-- Calculates the cost per slice -/
def cost_per_slice (p : Pizza) : ℚ :=
  total_cost p / p.total_slices

/-- Calculates how much Bob paid -/
def bob_payment (p : Pizza) : ℚ :=
  cost_per_slice p * p.bob_slices

/-- Calculates how much Alice paid -/
def alice_payment (p : Pizza) : ℚ :=
  cost_per_slice p * p.alice_slices

/-- The main theorem stating the difference in payment between Bob and Alice -/
theorem payment_difference (p : Pizza) 
  (h1 : p.total_slices = 12)
  (h2 : p.plain_cost = 12)
  (h3 : p.pepperoni_cost = 3)
  (h4 : p.mushroom_cost = 2)
  (h5 : p.bob_slices = 6)
  (h6 : p.charlie_slices = 5)
  (h7 : p.alice_slices = 3) :
  bob_payment p - alice_payment p = 4.26 := by
  sorry


end NUMINAMATH_CALUDE_payment_difference_l3571_357107


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3571_357172

def total_sum : ℝ := 2743
def second_part : ℝ := 1688
def second_rate : ℝ := 0.05
def first_time : ℝ := 8
def second_time : ℝ := 3

theorem interest_rate_calculation (first_rate : ℝ) : 
  (total_sum - second_part) * first_rate * first_time = second_part * second_rate * second_time →
  first_rate = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3571_357172


namespace NUMINAMATH_CALUDE_tree_purchase_solution_l3571_357194

/-- Represents the unit prices and purchasing schemes for tree seedlings -/
structure TreePurchase where
  osmanthus_price : ℕ
  camphor_price : ℕ
  schemes : List (ℕ × ℕ)

/-- Defines the conditions of the tree purchasing problem -/
def tree_purchase_problem (p : TreePurchase) : Prop :=
  -- First purchase condition
  10 * p.osmanthus_price + 20 * p.camphor_price = 3000 ∧
  -- Second purchase condition
  8 * p.osmanthus_price + 24 * p.camphor_price = 2800 ∧
  -- Next purchase conditions
  (∀ (o c : ℕ), (o, c) ∈ p.schemes →
    o + c = 40 ∧
    o * p.osmanthus_price + c * p.camphor_price ≤ 3800 ∧
    c ≤ 3 * o) ∧
  -- All possible schemes are included
  (∀ (o c : ℕ), o + c = 40 →
    o * p.osmanthus_price + c * p.camphor_price ≤ 3800 →
    c ≤ 3 * o →
    (o, c) ∈ p.schemes)

/-- Theorem stating the solution to the tree purchasing problem -/
theorem tree_purchase_solution :
  ∃ (p : TreePurchase),
    tree_purchase_problem p ∧
    p.osmanthus_price = 200 ∧
    p.camphor_price = 50 ∧
    p.schemes = [(10, 30), (11, 29), (12, 28)] :=
  sorry

end NUMINAMATH_CALUDE_tree_purchase_solution_l3571_357194


namespace NUMINAMATH_CALUDE_average_of_r_s_t_l3571_357100

theorem average_of_r_s_t (r s t : ℝ) (h : (5 / 4) * (r + s + t - 2) = 15) : 
  (r + s + t) / 3 = 14 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_r_s_t_l3571_357100


namespace NUMINAMATH_CALUDE_square_diff_fourth_power_l3571_357132

theorem square_diff_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_fourth_power_l3571_357132


namespace NUMINAMATH_CALUDE_marble_selection_probability_l3571_357129

theorem marble_selection_probability : 
  let total_marbles : ℕ := 12
  let marbles_per_color : ℕ := 3
  let colors : ℕ := 4
  let selected_marbles : ℕ := 4

  let total_ways : ℕ := Nat.choose total_marbles selected_marbles
  let favorable_ways : ℕ := marbles_per_color ^ colors

  (favorable_ways : ℚ) / total_ways = 9 / 55 :=
by sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l3571_357129


namespace NUMINAMATH_CALUDE_circle_area_not_covered_l3571_357120

theorem circle_area_not_covered (outer_diameter inner_diameter : ℝ) 
  (h1 : outer_diameter = 30) 
  (h2 : inner_diameter = 24) : 
  (outer_diameter^2 - inner_diameter^2) / outer_diameter^2 = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_not_covered_l3571_357120


namespace NUMINAMATH_CALUDE_a_plus_b_equals_two_l3571_357145

theorem a_plus_b_equals_two (a b : ℝ) : 
  (∀ x y : ℝ, y = a + b / x) →
  (2 = a + b / 1) →
  (4 = a + b / 4) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_two_l3571_357145


namespace NUMINAMATH_CALUDE_xyz_inequality_l3571_357137

theorem xyz_inequality : ∃ c : ℝ, ∀ x y z : ℝ, -|x*y*z| > c * (|x| + |y| + |z|) := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l3571_357137


namespace NUMINAMATH_CALUDE_tetradecagon_side_length_l3571_357144

/-- A regular tetradecagon is a polygon with 14 sides of equal length -/
def RegularTetradecagon := { n : ℕ // n = 14 }

/-- The perimeter of the tetradecagon table in centimeters -/
def perimeter : ℝ := 154

/-- Theorem: In a regular tetradecagon with a perimeter of 154 cm, the length of each side is 11 cm -/
theorem tetradecagon_side_length (t : RegularTetradecagon) :
  perimeter / t.val = 11 := by sorry

end NUMINAMATH_CALUDE_tetradecagon_side_length_l3571_357144


namespace NUMINAMATH_CALUDE_sallys_pears_l3571_357121

theorem sallys_pears (sara_pears : ℕ) (total_pears : ℕ) (sally_pears : ℕ) :
  sara_pears = 45 →
  total_pears = 56 →
  sally_pears = total_pears - sara_pears →
  sally_pears = 11 := by
  sorry

end NUMINAMATH_CALUDE_sallys_pears_l3571_357121


namespace NUMINAMATH_CALUDE_min_price_with_profit_margin_l3571_357152

theorem min_price_with_profit_margin (marked_price : ℝ) (markup_percentage : ℝ) (min_profit_margin : ℝ) : 
  marked_price = 240 →
  markup_percentage = 0.6 →
  min_profit_margin = 0.1 →
  let cost_price := marked_price / (1 + markup_percentage)
  let min_reduced_price := cost_price * (1 + min_profit_margin)
  min_reduced_price = 165 :=
by sorry

end NUMINAMATH_CALUDE_min_price_with_profit_margin_l3571_357152


namespace NUMINAMATH_CALUDE_max_value_of_product_l3571_357130

theorem max_value_of_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 2) :
  a^2 * b^3 * c^4 ≤ 143327232 / 386989855 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_product_l3571_357130


namespace NUMINAMATH_CALUDE_smallest_distance_to_i_l3571_357197

theorem smallest_distance_to_i (w : ℂ) (h : Complex.abs (w^2 - 3) = Complex.abs (w * (2*w + 3*Complex.I))) :
  ∃ (min_dist : ℝ), 
    (∀ w', Complex.abs (w'^2 - 3) = Complex.abs (w' * (2*w' + 3*Complex.I)) → 
      Complex.abs (w' - Complex.I) ≥ min_dist) ∧
    min_dist = Complex.abs ((Real.sqrt 3 - Real.sqrt 6) / 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_to_i_l3571_357197


namespace NUMINAMATH_CALUDE_algorithm_structure_logical_judgment_l3571_357158

-- Define the basic algorithm structures
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

-- Define a property for structures requiring logical judgment and different processing
def RequiresLogicalJudgment (s : AlgorithmStructure) : Prop :=
  match s with
  | AlgorithmStructure.Conditional => true
  | AlgorithmStructure.Loop => true
  | _ => false

-- Theorem statement
theorem algorithm_structure_logical_judgment :
  ∀ (s : AlgorithmStructure),
    RequiresLogicalJudgment s ↔ (s = AlgorithmStructure.Conditional ∨ s = AlgorithmStructure.Loop) :=
by sorry

end NUMINAMATH_CALUDE_algorithm_structure_logical_judgment_l3571_357158


namespace NUMINAMATH_CALUDE_village_population_l3571_357181

theorem village_population (partial_population : ℕ) (partial_percentage : ℚ) (total_population : ℕ) :
  partial_percentage = 9/10 →
  partial_population = 36000 →
  total_population * partial_percentage = partial_population →
  total_population = 40000 := by
sorry

end NUMINAMATH_CALUDE_village_population_l3571_357181


namespace NUMINAMATH_CALUDE_coloring_scheme_satisfies_conditions_l3571_357125

/-- Represents the three colors used in the coloring scheme. -/
inductive Color
  | White
  | Red
  | Blue

/-- The coloring function that assigns a color to each integral point in the plane. -/
def f : ℤ × ℤ → Color :=
  sorry

/-- Represents an infinite set of integers. -/
def InfiniteSet (s : Set ℤ) : Prop :=
  ∀ n : ℤ, ∃ m ∈ s, m > n

theorem coloring_scheme_satisfies_conditions :
  (∀ c : Color, InfiniteSet {k : ℤ | InfiniteSet {n : ℤ | f (n, k) = c}}) ∧
  (∀ a b c : ℤ × ℤ,
    f a = Color.White → f b = Color.Red → f c = Color.Blue →
    ∃ d : ℤ × ℤ, f d = Color.Red ∧ d = (a.1 + c.1 - b.1, a.2 + c.2 - b.2)) :=
by
  sorry

end NUMINAMATH_CALUDE_coloring_scheme_satisfies_conditions_l3571_357125


namespace NUMINAMATH_CALUDE_rachel_book_count_l3571_357104

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves for mystery books -/
def mystery_shelves : ℕ := 6

/-- The number of shelves for picture books -/
def picture_shelves : ℕ := 2

/-- The total number of books Rachel has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem rachel_book_count : total_books = 72 := by
  sorry

end NUMINAMATH_CALUDE_rachel_book_count_l3571_357104


namespace NUMINAMATH_CALUDE_brian_stones_l3571_357109

theorem brian_stones (total : ℕ) (grey : ℕ) (green : ℕ) (white : ℕ) (black : ℕ) : 
  total = 100 →
  grey = 40 →
  green = 60 →
  grey + green = total →
  white + black = total →
  (white : ℚ) / total = (green : ℚ) / total →
  white > black →
  white = 60 := by
sorry

end NUMINAMATH_CALUDE_brian_stones_l3571_357109


namespace NUMINAMATH_CALUDE_quarterly_insurance_payment_l3571_357111

theorem quarterly_insurance_payment 
  (annual_payment : ℕ) 
  (quarters_per_year : ℕ) 
  (h1 : annual_payment = 1512) 
  (h2 : quarters_per_year = 4) : 
  annual_payment / quarters_per_year = 378 := by
sorry

end NUMINAMATH_CALUDE_quarterly_insurance_payment_l3571_357111


namespace NUMINAMATH_CALUDE_sine_graph_transformation_l3571_357169

/-- Given two sine functions with different periods, prove that the graph of one
    can be obtained by transforming the graph of the other. -/
theorem sine_graph_transformation (x : ℝ) :
  let f (x : ℝ) := Real.sin (x + π/4)
  let g (x : ℝ) := Real.sin (3*x + π/4)
  ∃ (h : ℝ → ℝ), (∀ x, g x = f (h x)) ∧ (∀ x, h x = x/3) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_transformation_l3571_357169


namespace NUMINAMATH_CALUDE_unknown_number_value_l3571_357116

theorem unknown_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * x * 315 * 7) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l3571_357116


namespace NUMINAMATH_CALUDE_factorization_equality_l3571_357175

theorem factorization_equality (x y : ℝ) : 6 * x^3 * y^2 + 3 * x^2 * y^2 = 3 * x^2 * y^2 * (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3571_357175


namespace NUMINAMATH_CALUDE_clock_ticks_theorem_l3571_357177

/-- Represents the number of ticks and time between first and last ticks for a clock -/
structure ClockTicks where
  num_ticks : ℕ
  time_between : ℕ

/-- Calculates the number of ticks given the time between first and last ticks -/
def calculate_ticks (reference : ClockTicks) (time : ℕ) : ℕ :=
  let interval := reference.time_between / (reference.num_ticks - 1)
  (time / interval) + 1

theorem clock_ticks_theorem (reference : ClockTicks) (time : ℕ) :
  reference.num_ticks = 8 ∧ reference.time_between = 42 ∧ time = 30 →
  calculate_ticks reference time = 6 :=
by
  sorry

#check clock_ticks_theorem

end NUMINAMATH_CALUDE_clock_ticks_theorem_l3571_357177


namespace NUMINAMATH_CALUDE_flight_passenger_distribution_l3571_357154

/-- Proof of the flight passenger distribution problem -/
theorem flight_passenger_distribution
  (total_passengers : ℕ)
  (female_percentage : ℚ)
  (first_class_male_ratio : ℚ)
  (coach_females : ℕ)
  (h1 : total_passengers = 120)
  (h2 : female_percentage = 30 / 100)
  (h3 : first_class_male_ratio = 1 / 3)
  (h4 : coach_females = 28)
  : ∃ (first_class_percentage : ℚ), first_class_percentage = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_flight_passenger_distribution_l3571_357154


namespace NUMINAMATH_CALUDE_circle_polar_equation_l3571_357149

/-- The polar equation ρ = 2a cos θ represents a circle with center C(a, 0) and radius a -/
theorem circle_polar_equation (a : ℝ) :
  ∀ ρ θ : ℝ, ρ = 2 * a * Real.cos θ ↔ 
  ∃ x y : ℝ, (x - a)^2 + y^2 = a^2 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_circle_polar_equation_l3571_357149


namespace NUMINAMATH_CALUDE_triangle_area_l3571_357119

-- Define the lines that bound the triangle
def line1 (x : ℝ) : ℝ := x
def line2 (x : ℝ) : ℝ := -x
def line3 : ℝ := 8

-- Theorem statement
theorem triangle_area : 
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := |A.1 - B.1|
  let height := |line3 - O.2|
  (1/2 : ℝ) * base * height = 64 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3571_357119


namespace NUMINAMATH_CALUDE_factors_of_20160_l3571_357106

theorem factors_of_20160 : (Finset.filter (· ∣ 20160) (Finset.range 20161)).card = 72 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_20160_l3571_357106


namespace NUMINAMATH_CALUDE_prob_at_least_eight_sixes_l3571_357135

/-- The probability of rolling a six on a fair die -/
def prob_six : ℚ := 1/6

/-- The number of rolls -/
def num_rolls : ℕ := 10

/-- The minimum number of sixes required -/
def min_sixes : ℕ := 8

/-- Calculates the probability of rolling exactly k sixes in n rolls -/
def prob_exact_sixes (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (prob_six ^ k) * ((1 - prob_six) ^ (n - k))

/-- The probability of rolling at least 8 sixes in 10 rolls of a fair die -/
theorem prob_at_least_eight_sixes : 
  (prob_exact_sixes num_rolls min_sixes + 
   prob_exact_sixes num_rolls (min_sixes + 1) + 
   prob_exact_sixes num_rolls (min_sixes + 2)) = 3/15504 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_eight_sixes_l3571_357135


namespace NUMINAMATH_CALUDE_two_part_journey_average_speed_l3571_357123

/-- Calculates the average speed of a two-part journey -/
theorem two_part_journey_average_speed 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) :
  distance1 = 360 →
  speed1 = 60 →
  distance2 = 120 →
  speed2 = 40 →
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 480 / 9 :=
by
  sorry

#eval (480 : ℚ) / 9

end NUMINAMATH_CALUDE_two_part_journey_average_speed_l3571_357123


namespace NUMINAMATH_CALUDE_coordinates_not_on_C_do_not_satisfy_F_l3571_357186

-- Define the curve C as a set of points in R²
def C : Set (ℝ × ℝ) := sorry

-- Define the function F
def F : ℝ → ℝ → ℝ := sorry

-- Theorem statement
theorem coordinates_not_on_C_do_not_satisfy_F :
  (∀ x y, F x y = 0 → (x, y) ∈ C) →
  ∀ x y, (x, y) ∉ C → F x y ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_not_on_C_do_not_satisfy_F_l3571_357186


namespace NUMINAMATH_CALUDE_sector_area_l3571_357162

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 12) (h2 : central_angle = 2) :
  let radius := perimeter / (2 + central_angle)
  (1/2) * central_angle * radius^2 = 9 := by sorry

end NUMINAMATH_CALUDE_sector_area_l3571_357162


namespace NUMINAMATH_CALUDE_exterior_angle_measure_l3571_357124

-- Define the nonagon's interior angle
def nonagon_interior_angle : ℝ := 140

-- Define the nonagon's exterior angle
def nonagon_exterior_angle : ℝ := 360 - nonagon_interior_angle

-- Define the square's interior angle
def square_interior_angle : ℝ := 90

-- Theorem statement
theorem exterior_angle_measure :
  nonagon_exterior_angle - square_interior_angle = 130 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_measure_l3571_357124


namespace NUMINAMATH_CALUDE_fraction_change_theorem_l3571_357160

theorem fraction_change_theorem (a b c d e f x y : ℚ) 
  (h1 : a ≠ b) (h2 : b ≠ 0) 
  (h3 : (a + x) / (b + y) = c / d) 
  (h4 : (a + 2*x) / (b + 2*y) = e / f) 
  (h5 : d ≠ c) (h6 : f ≠ e) : 
  x = (b*c - a*d) / (d - c) ∧ 
  y = (b*e - a*f) / (2*f - 2*e) := by
sorry

end NUMINAMATH_CALUDE_fraction_change_theorem_l3571_357160


namespace NUMINAMATH_CALUDE_population_growth_model_l3571_357143

/-- World population growth model from 1992 to 2000 -/
theorem population_growth_model 
  (initial_population : ℝ) 
  (growth_rate : ℝ) 
  (years : ℕ) 
  (final_population : ℝ) :
  initial_population = 5.48 →
  years = 8 →
  final_population = initial_population * (1 + growth_rate / 100) ^ years :=
by sorry

end NUMINAMATH_CALUDE_population_growth_model_l3571_357143


namespace NUMINAMATH_CALUDE_average_annual_growth_rate_l3571_357156

theorem average_annual_growth_rate 
  (p q : ℝ) 
  (hp : p > -1) 
  (hq : q > -1) :
  ∃ x : ℝ, x > -1 ∧ (1 + x)^2 = (1 + p) * (1 + q) ∧ 
  x = Real.sqrt ((1 + p) * (1 + q)) - 1 :=
sorry

end NUMINAMATH_CALUDE_average_annual_growth_rate_l3571_357156


namespace NUMINAMATH_CALUDE_fermat_like_equation_implies_power_l3571_357174

theorem fermat_like_equation_implies_power (n p x y k : ℕ) : 
  Odd n → 
  n > 1 → 
  Nat.Prime p → 
  Odd p → 
  x^n + y^n = p^k → 
  ∃ t : ℕ, n = p^t := by
sorry

end NUMINAMATH_CALUDE_fermat_like_equation_implies_power_l3571_357174


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_in_interval_l3571_357176

theorem inequality_holds_iff_p_in_interval (p : ℝ) :
  (∀ x : ℝ, -9 < (3 * x^2 + p * x - 6) / (x^2 - x + 1) ∧ 
             (3 * x^2 + p * x - 6) / (x^2 - x + 1) < 6) ↔ 
  -3 < p ∧ p < 6 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_in_interval_l3571_357176


namespace NUMINAMATH_CALUDE_distributive_property_l3571_357190

theorem distributive_property (x y : ℝ) : x * (1 + y) = x + x * y := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_l3571_357190


namespace NUMINAMATH_CALUDE_one_by_one_tile_position_l3571_357108

/-- Represents a tile with width and height -/
structure Tile where
  width : ℕ
  height : ℕ

/-- Represents a square grid -/
structure Square where
  side_length : ℕ

/-- Represents the position of a tile in the square -/
structure TilePosition where
  row : ℕ
  col : ℕ

/-- Checks if a position is in the center or adjacent to the boundary of the square -/
def is_center_or_adjacent_boundary (pos : TilePosition) (square : Square) : Prop :=
  (pos.row = square.side_length / 2 + 1 ∧ pos.col = square.side_length / 2 + 1) ∨
  (pos.row = 1 ∨ pos.row = square.side_length ∨ pos.col = 1 ∨ pos.col = square.side_length)

/-- Theorem: In a 7x7 square formed by sixteen 1x3 tiles and one 1x1 tile,
    the 1x1 tile must be either in the center or adjacent to the boundary -/
theorem one_by_one_tile_position
  (square : Square)
  (large_tiles : Finset Tile)
  (small_tile : Tile)
  (tile_arrangement : Square → Finset Tile → Tile → TilePosition) :
  square.side_length = 7 →
  large_tiles.card = 16 →
  (∀ t ∈ large_tiles, t.width = 1 ∧ t.height = 3) →
  small_tile.width = 1 ∧ small_tile.height = 1 →
  is_center_or_adjacent_boundary (tile_arrangement square large_tiles small_tile) square :=
by sorry

end NUMINAMATH_CALUDE_one_by_one_tile_position_l3571_357108


namespace NUMINAMATH_CALUDE_stone_skipping_l3571_357112

theorem stone_skipping (throw1 throw2 throw3 throw4 throw5 : ℕ) : 
  throw5 = 8 ∧ 
  throw2 = throw1 + 2 ∧ 
  throw3 = 2 * throw2 ∧ 
  throw4 = throw3 - 3 ∧ 
  throw5 = throw4 + 1 →
  throw1 + throw2 + throw3 + throw4 + throw5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_stone_skipping_l3571_357112
