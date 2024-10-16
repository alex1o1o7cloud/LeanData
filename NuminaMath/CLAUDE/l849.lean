import Mathlib

namespace NUMINAMATH_CALUDE_sin_product_equation_l849_84984

theorem sin_product_equation : 
  256 * Real.sin (10 * π / 180) * Real.sin (30 * π / 180) * 
  Real.sin (50 * π / 180) * Real.sin (70 * π / 180) = 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equation_l849_84984


namespace NUMINAMATH_CALUDE_dave_initial_apps_l849_84933

/-- The number of apps Dave initially had on his phone -/
def initial_apps : ℕ := 15

/-- The number of apps Dave added -/
def added_apps : ℕ := 71

/-- The number of apps Dave had left after deleting some -/
def remaining_apps : ℕ := 14

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := added_apps + 1

theorem dave_initial_apps : 
  initial_apps + added_apps - deleted_apps = remaining_apps :=
by sorry

end NUMINAMATH_CALUDE_dave_initial_apps_l849_84933


namespace NUMINAMATH_CALUDE_min_throws_for_repeated_sum_min_throws_is_22_l849_84930

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice being thrown -/
def num_dice : ℕ := 4

/-- The minimum possible sum when rolling the dice -/
def min_sum : ℕ := num_dice

/-- The maximum possible sum when rolling the dice -/
def max_sum : ℕ := num_dice * sides

/-- The number of distinct possible sums -/
def distinct_sums : ℕ := max_sum - min_sum + 1

/-- 
The minimum number of throws required to guarantee a repeated sum 
when rolling four fair six-sided dice
-/
theorem min_throws_for_repeated_sum : ℕ := distinct_sums + 1

/-- The main theorem to prove -/
theorem min_throws_is_22 : min_throws_for_repeated_sum = 22 := by sorry

end NUMINAMATH_CALUDE_min_throws_for_repeated_sum_min_throws_is_22_l849_84930


namespace NUMINAMATH_CALUDE_committee_meeting_attendance_l849_84958

theorem committee_meeting_attendance :
  ∀ (associate_profs assistant_profs : ℕ),
    2 * associate_profs + assistant_profs = 7 →
    associate_profs + 2 * assistant_profs = 11 →
    associate_profs + assistant_profs = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_committee_meeting_attendance_l849_84958


namespace NUMINAMATH_CALUDE_triangle_problem_l849_84940

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (b-2a)cos C + c cos B = 0, c = √7, and b = 3a, then the measure of angle C is π/3
    and the area of the triangle is 3√3/4. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (b - 2*a) * Real.cos C + c * Real.cos B = 0 →
  c = Real.sqrt 7 →
  b = 3*a →
  C = π/3 ∧ (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l849_84940


namespace NUMINAMATH_CALUDE_composition_is_rotation_l849_84973

-- Define a rotation
def Rotation (center : Point) (angle : ℝ) : Point → Point :=
  sorry

-- Define the composition of two rotations
def ComposeRotations (A B : Point) (α β : ℝ) : Point → Point :=
  Rotation B β ∘ Rotation A α

-- Theorem statement
theorem composition_is_rotation (A B : Point) (α β : ℝ) 
  (h1 : A ≠ B) 
  (h2 : ¬ (∃ k : ℤ, α + β = 2 * π * k)) :
  ∃ (O : Point) (γ : ℝ), ComposeRotations A B α β = Rotation O γ ∧ γ = α + β :=
sorry

end NUMINAMATH_CALUDE_composition_is_rotation_l849_84973


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l849_84937

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The problem statement -/
theorem base_conversion_subtraction :
  let base_7_num := [3, 0, 1, 2, 5]  -- 52103 in base 7 (least significant digit first)
  let base_5_num := [0, 2, 1, 3, 4]  -- 43120 in base 5 (least significant digit first)
  to_base_10 base_7_num 7 - to_base_10 base_5_num 5 = 9833 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l849_84937


namespace NUMINAMATH_CALUDE_parabola_properties_l849_84941

/-- Parabola with equation y = ax(x-6) + 1 where a ≠ 0 -/
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x * (x - 6) + 1

theorem parabola_properties (a : ℝ) (h : a ≠ 0) :
  /- Point (0,1) lies on the parabola -/
  (parabola a 0 = 1) ∧
  /- If the distance from the vertex to the x-axis is 5, then a = 2/3 or a = -4/9 -/
  (∃ (x : ℝ), (∀ y : ℝ, parabola a y ≥ parabola a x) →
    |parabola a x| = 5 → (a = 2/3 ∨ a = -4/9)) ∧
  /- If the length of the segment formed by the intersection of the parabola with the x-axis
     is less than or equal to 4, then 1/9 < a ≤ 1/5 -/
  ((∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ parabola a x₁ = 0 ∧ parabola a x₂ = 0 ∧ x₂ - x₁ ≤ 4) →
    1/9 < a ∧ a ≤ 1/5) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l849_84941


namespace NUMINAMATH_CALUDE_total_book_price_l849_84951

theorem total_book_price (total_books : ℕ) (math_books : ℕ) (math_price : ℕ) (history_price : ℕ) :
  total_books = 90 →
  math_books = 54 →
  math_price = 4 →
  history_price = 5 →
  (math_books * math_price + (total_books - math_books) * history_price) = 396 :=
by sorry

end NUMINAMATH_CALUDE_total_book_price_l849_84951


namespace NUMINAMATH_CALUDE_marlas_errand_time_l849_84955

/-- The total time Marla spends on her errand activities -/
def total_time (driving_time grocery_time gas_time parent_teacher_time coffee_time : ℕ) : ℕ :=
  2 * driving_time + grocery_time + gas_time + parent_teacher_time + coffee_time

/-- Theorem stating the total time Marla spends on her errand activities -/
theorem marlas_errand_time : 
  total_time 20 15 5 70 30 = 160 := by sorry

end NUMINAMATH_CALUDE_marlas_errand_time_l849_84955


namespace NUMINAMATH_CALUDE_special_rhombus_perimeter_l849_84974

/-- A rhombus with integer side lengths where the area equals the perimeter -/
structure SpecialRhombus where
  side_length : ℕ
  area_eq_perimeter : (side_length ^ 2 * Real.sin (π / 6)) = (4 * side_length)

/-- The perimeter of a SpecialRhombus is 32 -/
theorem special_rhombus_perimeter (r : SpecialRhombus) : 4 * r.side_length = 32 := by
  sorry

#check special_rhombus_perimeter

end NUMINAMATH_CALUDE_special_rhombus_perimeter_l849_84974


namespace NUMINAMATH_CALUDE_average_salary_calculation_l849_84934

/-- Calculates the average salary of all employees in an office --/
theorem average_salary_calculation (officer_salary : ℕ) (non_officer_salary : ℕ) 
  (officer_count : ℕ) (non_officer_count : ℕ) :
  officer_salary = 470 →
  non_officer_salary = 110 →
  officer_count = 15 →
  non_officer_count = 525 →
  (officer_salary * officer_count + non_officer_salary * non_officer_count) / 
    (officer_count + non_officer_count) = 120 := by
  sorry

#check average_salary_calculation

end NUMINAMATH_CALUDE_average_salary_calculation_l849_84934


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_f_l849_84978

noncomputable def f (x : ℝ) : ℝ := (Real.cos x + Real.sin x) / (Real.cos x - Real.sin x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬is_periodic f q

theorem smallest_positive_period_of_f :
  is_smallest_positive_period f Real.pi := by sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_f_l849_84978


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l849_84900

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |2*x - 5|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 2 x ≥ 5} = {x : ℝ | x ≤ 2 ∨ x ≥ 8/3} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | a > 0 ∧ ∀ x ∈ Set.Icc a (2*a - 2), f a x ≤ |x + 4|} = Set.Ioo 2 (13/5) := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l849_84900


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l849_84923

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : Nat
  sampleSize : Nat
  startingNumber : Nat

/-- Generates the sequence of selected student numbers. -/
def generateSequence (s : SystematicSampling) : List Nat :=
  List.range s.sampleSize |>.map (fun i => s.startingNumber + i * (s.totalStudents / s.sampleSize))

/-- Theorem stating the properties of systematic sampling for the given problem. -/
theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.totalStudents = 50)
  (h2 : s.sampleSize = 5)
  (h3 : 1 ≤ s.startingNumber)
  (h4 : s.startingNumber ≤ 10) :
  ∃ (a : Nat), 1 ≤ a ∧ a ≤ 10 ∧ 
  generateSequence s = [a, a + 10, a + 20, a + 30, a + 40] :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l849_84923


namespace NUMINAMATH_CALUDE_log_difference_l849_84926

theorem log_difference (a b c d : ℕ+) 
  (h1 : (Real.log b) / (Real.log a) = 3/2)
  (h2 : (Real.log d) / (Real.log c) = 5/4)
  (h3 : a - c = 9) :
  b - d = 93 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_l849_84926


namespace NUMINAMATH_CALUDE_chess_tournament_games_l849_84982

theorem chess_tournament_games (n : ℕ) (h : n = 50) : 
  (n * (n - 1)) / 2 = 1225 :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l849_84982


namespace NUMINAMATH_CALUDE_sequence_properties_l849_84928

def sequence_a (n : ℕ) : ℕ := 2 * n - 1

def sequence_b (n : ℕ) : ℕ := 2^(n - 1)

def sum_sequence_a (n : ℕ) : ℕ := n^2

def sum_sequence_ab (n : ℕ) : ℕ := (2 * n - 3) * 2^n + 3

theorem sequence_properties :
  (∀ n, sum_sequence_a n = n^2) →
  sequence_b 2 = 2 →
  sequence_b 5 = 16 →
  (∀ n, sequence_a n = 2 * n - 1) ∧
  (∀ n, sequence_b n = 2^(n - 1)) ∧
  (∀ n, sum_sequence_ab n = (2 * n - 3) * 2^n + 3) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l849_84928


namespace NUMINAMATH_CALUDE_workshop_average_salary_l849_84979

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_salary : ℕ)
  (other_salary : ℕ)
  (h1 : total_workers = 14)
  (h2 : technicians = 7)
  (h3 : technician_salary = 12000)
  (h4 : other_salary = 6000) :
  (technicians * technician_salary + (total_workers - technicians) * other_salary) / total_workers = 9000 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l849_84979


namespace NUMINAMATH_CALUDE_no_rational_multiples_of_pi_l849_84953

theorem no_rational_multiples_of_pi (x y : ℚ) : 
  (∃ (m n : ℚ), x = m * Real.pi ∧ y = n * Real.pi) →
  0 < x → x < y → y < Real.pi / 2 →
  Real.tan x + Real.tan y = 2 →
  False :=
sorry

end NUMINAMATH_CALUDE_no_rational_multiples_of_pi_l849_84953


namespace NUMINAMATH_CALUDE_donut_theorem_l849_84911

def donut_problem (initial : ℕ) (eaten : ℕ) (taken : ℕ) : ℕ :=
  let remaining_after_eaten := initial - eaten
  let remaining_after_taken := remaining_after_eaten - taken
  remaining_after_taken - remaining_after_taken / 2

theorem donut_theorem : donut_problem 50 2 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_donut_theorem_l849_84911


namespace NUMINAMATH_CALUDE_apple_sales_remaining_fraction_l849_84909

/-- Proves that the fraction of money remaining after repairs is 1/5 --/
theorem apple_sales_remaining_fraction (apple_price : ℚ) (bike_cost : ℚ) (repair_percentage : ℚ) (apples_sold : ℕ) :
  apple_price = 5/4 →
  bike_cost = 80 →
  repair_percentage = 1/4 →
  apples_sold = 20 →
  let total_earnings := apple_price * apples_sold
  let repair_cost := repair_percentage * bike_cost
  let remaining := total_earnings - repair_cost
  remaining / total_earnings = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_apple_sales_remaining_fraction_l849_84909


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_7_l849_84988

/-- The product of the first 7 positive integers -/
def product_7 : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

/-- A function to check if a number is a five-digit integer -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- A function to calculate the product of digits of a number -/
def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

/-- The theorem stating that 98752 is the largest five-digit integer
    whose digits have a product equal to (7)(6)(5)(4)(3)(2)(1) -/
theorem largest_five_digit_with_product_7 :
  (is_five_digit 98752) ∧ 
  (digit_product 98752 = product_7) ∧ 
  (∀ n : ℕ, is_five_digit n → digit_product n = product_7 → n ≤ 98752) :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_7_l849_84988


namespace NUMINAMATH_CALUDE_right_triangle_sine_cosine_l849_84945

theorem right_triangle_sine_cosine (P Q R : Real) (h1 : 3 * Real.sin P = 4 * Real.cos P) :
  Real.sin P = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sine_cosine_l849_84945


namespace NUMINAMATH_CALUDE_blocks_used_in_structure_l849_84966

/-- Represents the dimensions of a rectangular structure --/
structure StructureDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the number of blocks used in a rectangular structure --/
def blocksUsed (dimensions : StructureDimensions) (floorThickness : ℝ) (wallThickness : ℝ) : ℝ :=
  let totalVolume := dimensions.length * dimensions.width * dimensions.height
  let internalLength := dimensions.length - 2 * wallThickness
  let internalWidth := dimensions.width - 2 * wallThickness
  let internalHeight := dimensions.height - 2 * floorThickness
  let internalVolume := internalLength * internalWidth * internalHeight
  totalVolume - internalVolume

/-- Theorem stating that the number of blocks used in the given structure is 1068 --/
theorem blocks_used_in_structure :
  let dimensions : StructureDimensions := { length := 16, width := 12, height := 8 }
  let floorThickness := 2
  let wallThickness := 1.5
  blocksUsed dimensions floorThickness wallThickness = 1068 := by
  sorry

end NUMINAMATH_CALUDE_blocks_used_in_structure_l849_84966


namespace NUMINAMATH_CALUDE_complex_subtraction_l849_84980

/-- Given complex numbers c and d, prove that c - 3d = 2 + 6i -/
theorem complex_subtraction (c d : ℂ) (hc : c = 5 + 3*I) (hd : d = 1 - I) :
  c - 3*d = 2 + 6*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l849_84980


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_43_l849_84905

theorem least_positive_integer_multiple_of_43 :
  ∃ (x : ℕ+), 
    (∀ (y : ℕ+), y < x → ¬(43 ∣ (2*y)^2 + 2*33*(2*y) + 33^2)) ∧ 
    (43 ∣ (2*x)^2 + 2*33*(2*x) + 33^2) ∧
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_43_l849_84905


namespace NUMINAMATH_CALUDE_erica_saw_three_warthogs_l849_84959

/-- Represents the number of animals Erica saw on each day of her safari --/
structure SafariCount where
  saturday : Nat
  sunday : Nat
  monday_rhinos : Nat
  monday_warthogs : Nat

/-- The total number of animals seen during the safari --/
def total_animals : Nat := 20

/-- The number of animals Erica saw on Saturday --/
def saturday_count : Nat := 3 + 2

/-- The number of animals Erica saw on Sunday --/
def sunday_count : Nat := 2 + 5

/-- The number of rhinos Erica saw on Monday --/
def monday_rhinos : Nat := 5

/-- Theorem stating that Erica saw 3 warthogs on Monday --/
theorem erica_saw_three_warthogs (safari : SafariCount) :
  safari.saturday = saturday_count →
  safari.sunday = sunday_count →
  safari.monday_rhinos = monday_rhinos →
  safari.saturday + safari.sunday + safari.monday_rhinos + safari.monday_warthogs = total_animals →
  safari.monday_warthogs = 3 := by
  sorry


end NUMINAMATH_CALUDE_erica_saw_three_warthogs_l849_84959


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_396_l849_84986

theorem six_digit_divisible_by_396 : ∃ (x y z : ℕ), 
  x < 10 ∧ y < 10 ∧ z < 10 ∧ 
  (243000 + 100 * x + 10 * y + z) % 396 = 0 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_396_l849_84986


namespace NUMINAMATH_CALUDE_lawrence_walking_days_l849_84987

/-- Given Lawrence's walking data, prove the number of days he walked. -/
theorem lawrence_walking_days (daily_distance : ℝ) (total_distance : ℝ) 
  (h1 : daily_distance = 4.0)
  (h2 : total_distance = 12) : 
  total_distance / daily_distance = 3 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_walking_days_l849_84987


namespace NUMINAMATH_CALUDE_complex_magnitude_l849_84947

theorem complex_magnitude (z : ℂ) (h : (2 + Complex.I) * z = 4 - (1 + Complex.I)^2) : 
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l849_84947


namespace NUMINAMATH_CALUDE_range_of_e_l849_84915

theorem range_of_e (a b c d e : ℝ) 
  (sum_eq : a + b + c + d + e = 8) 
  (sum_squares_eq : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  0 ≤ e ∧ e ≤ 16/5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_e_l849_84915


namespace NUMINAMATH_CALUDE_f_is_odd_iff_l849_84975

/-- A function f is odd if f(-x) = -f(x) for all x in the domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- The function f(x) = x|x + a| + b -/
def f (a b : ℝ) : ℝ → ℝ := fun x ↦ x * |x + a| + b

/-- Theorem: f is an odd function if and only if a = 0 and b = 0 -/
theorem f_is_odd_iff (a b : ℝ) :
  IsOdd (f a b) ↔ a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_iff_l849_84975


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l849_84917

-- Define a sequence
def Sequence := ℕ → ℝ

-- Define the property of being a geometric sequence
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the given condition
def Condition (a : Sequence) : Prop :=
  ∀ n : ℕ, n > 1 → a n ^ 2 = a (n - 1) * a (n + 1)

-- State the theorem
theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → Condition a) ∧
  (∃ a : Sequence, Condition a ∧ ¬IsGeometric a) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l849_84917


namespace NUMINAMATH_CALUDE_ellipse_equation_l849_84902

/-- The standard equation of an ellipse with given properties -/
theorem ellipse_equation (a b c : ℝ) (h1 : a = 2) (h2 : c = Real.sqrt 3) (h3 : b^2 = a^2 - c^2) :
  ∀ (x y : ℝ), x^2 / 4 + y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l849_84902


namespace NUMINAMATH_CALUDE_flower_basket_problem_l849_84948

theorem flower_basket_problem (o y p : ℕ) 
  (h1 : y + p = 7)   -- All but 7 are orange
  (h2 : o + p = 10)  -- All but 10 are yellow
  (h3 : o + y = 5)   -- All but 5 are purple
  : o + y + p = 11 := by
  sorry

#check flower_basket_problem

end NUMINAMATH_CALUDE_flower_basket_problem_l849_84948


namespace NUMINAMATH_CALUDE_rectangular_garden_diagonal_ratio_l849_84946

theorem rectangular_garden_diagonal_ratio (b : ℝ) (h : b > 0) :
  let a := 3 * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let perimeter := 2 * (a + b)
  diagonal / perimeter = Real.sqrt 10 / 8 ∧ perimeter - diagonal = b :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_diagonal_ratio_l849_84946


namespace NUMINAMATH_CALUDE_first_half_speed_l849_84913

theorem first_half_speed (total_distance : ℝ) (first_half_distance : ℝ) (second_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 20 →
  first_half_distance = 10 →
  second_half_speed = 10 →
  average_speed = 10.909090909090908 →
  (total_distance / (first_half_distance / (total_distance / average_speed - first_half_distance / second_half_speed) + first_half_distance / second_half_speed)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_first_half_speed_l849_84913


namespace NUMINAMATH_CALUDE_area_BCD_l849_84997

-- Define the points A, B, C, D
variable (A B C D : ℝ × ℝ)

-- Define the conditions
variable (area_ABC : Real)
variable (length_AC : Real)
variable (length_CD : Real)

-- Axioms
axiom area_ABC_value : area_ABC = 45
axiom AC_length : length_AC = 10
axiom CD_length : length_CD = 30
axiom B_perpendicular_AD : (B.2 - A.2) * (D.1 - A.1) = (B.1 - A.1) * (D.2 - A.2)

-- Theorem to prove
theorem area_BCD (h : ℝ) : 
  area_ABC = 1/2 * length_AC * h → 
  1/2 * length_CD * h = 135 :=
sorry

end NUMINAMATH_CALUDE_area_BCD_l849_84997


namespace NUMINAMATH_CALUDE_y2_greater_than_y1_l849_84922

-- Define the linear function
def f (x : ℝ) : ℝ := -2 * x + 1

-- Define the points A and B
def A : ℝ × ℝ := (-1, f (-1))
def B : ℝ × ℝ := (-2, f (-2))

-- Theorem statement
theorem y2_greater_than_y1 : A.2 < B.2 := by
  sorry

end NUMINAMATH_CALUDE_y2_greater_than_y1_l849_84922


namespace NUMINAMATH_CALUDE_neds_video_games_l849_84924

theorem neds_video_games (non_working : ℕ) (price_per_game : ℕ) (total_earned : ℕ) :
  non_working = 6 →
  price_per_game = 7 →
  total_earned = 63 →
  non_working + (total_earned / price_per_game) = 15 :=
by sorry

end NUMINAMATH_CALUDE_neds_video_games_l849_84924


namespace NUMINAMATH_CALUDE_roots_product_plus_one_l849_84994

theorem roots_product_plus_one (p q r : ℂ) : 
  p^3 - 15*p^2 + 25*p - 10 = 0 →
  q^3 - 15*q^2 + 25*q - 10 = 0 →
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  (1+p)*(1+q)*(1+r) = 51 := by
sorry

end NUMINAMATH_CALUDE_roots_product_plus_one_l849_84994


namespace NUMINAMATH_CALUDE_bread_calculation_l849_84957

def initial_bread : ℕ := 200

def day1_fraction : ℚ := 1/4
def day2_fraction : ℚ := 2/5
def day3_fraction : ℚ := 1/2

def remaining_bread : ℕ := 45

theorem bread_calculation :
  (initial_bread - (day1_fraction * initial_bread).floor) -
  (day2_fraction * (initial_bread - (day1_fraction * initial_bread).floor)).floor -
  (day3_fraction * ((initial_bread - (day1_fraction * initial_bread).floor) -
    (day2_fraction * (initial_bread - (day1_fraction * initial_bread).floor)).floor)).floor = remaining_bread := by
  sorry

end NUMINAMATH_CALUDE_bread_calculation_l849_84957


namespace NUMINAMATH_CALUDE_rainfall_problem_l849_84919

/-- Rainfall problem -/
theorem rainfall_problem (total_rainfall : ℝ) (ratio : ℝ) :
  total_rainfall = 35 →
  ratio = 1.5 →
  ∃ (first_week : ℝ),
    first_week + ratio * first_week = total_rainfall ∧
    ratio * first_week = 21 := by
  sorry


end NUMINAMATH_CALUDE_rainfall_problem_l849_84919


namespace NUMINAMATH_CALUDE_radius_of_circle_from_spherical_coords_l849_84938

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/3) is √3/2 -/
theorem radius_of_circle_from_spherical_coords :
  let r : ℝ := Real.sqrt 3 / 2
  ∀ θ : ℝ,
  let x : ℝ := (1 : ℝ) * Real.sin (π / 3) * Real.cos θ
  let y : ℝ := (1 : ℝ) * Real.sin (π / 3) * Real.sin θ
  Real.sqrt (x^2 + y^2) = r :=
by sorry

end NUMINAMATH_CALUDE_radius_of_circle_from_spherical_coords_l849_84938


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l849_84991

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x - 3 = 0 ∧ y^2 + m*y - 3 = 0) ∧
  (3^2 + m*3 - 3 = 0 → (-1)^2 + m*(-1) - 3 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l849_84991


namespace NUMINAMATH_CALUDE_solve_equation_l849_84969

theorem solve_equation (m : ℝ) : (m - 4)^2 = (1/16)⁻¹ → m = 8 ∨ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l849_84969


namespace NUMINAMATH_CALUDE_cantaloupes_total_l849_84963

/-- The number of cantaloupes grown by Fred -/
def fred_cantaloupes : ℕ := 38

/-- The number of cantaloupes grown by Tim -/
def tim_cantaloupes : ℕ := 44

/-- The total number of cantaloupes grown by Fred and Tim -/
def total_cantaloupes : ℕ := fred_cantaloupes + tim_cantaloupes

theorem cantaloupes_total : total_cantaloupes = 82 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupes_total_l849_84963


namespace NUMINAMATH_CALUDE_total_earnings_is_18_56_l849_84912

/-- Represents the total number of marbles -/
def total_marbles : ℕ := 150

/-- Represents the percentage of white marbles -/
def white_percent : ℚ := 20 / 100

/-- Represents the percentage of black marbles -/
def black_percent : ℚ := 25 / 100

/-- Represents the percentage of blue marbles -/
def blue_percent : ℚ := 30 / 100

/-- Represents the percentage of green marbles -/
def green_percent : ℚ := 15 / 100

/-- Represents the percentage of red marbles -/
def red_percent : ℚ := 10 / 100

/-- Represents the price of a white marble in dollars -/
def white_price : ℚ := 5 / 100

/-- Represents the price of a black marble in dollars -/
def black_price : ℚ := 10 / 100

/-- Represents the price of a blue marble in dollars -/
def blue_price : ℚ := 15 / 100

/-- Represents the price of a green marble in dollars -/
def green_price : ℚ := 12 / 100

/-- Represents the price of a red marble in dollars -/
def red_price : ℚ := 25 / 100

/-- Theorem stating that the total earnings from selling all marbles is $18.56 -/
theorem total_earnings_is_18_56 : 
  (↑total_marbles * white_percent * white_price) +
  (↑total_marbles * black_percent * black_price) +
  (↑total_marbles * blue_percent * blue_price) +
  (↑total_marbles * green_percent * green_price) +
  (↑total_marbles * red_percent * red_price) = 1856 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_is_18_56_l849_84912


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l849_84976

theorem absolute_value_inequality (x : ℝ) :
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ ((1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l849_84976


namespace NUMINAMATH_CALUDE_bernoulli_misplacement_6_letters_l849_84983

/-- Bernoulli's misplacement number for n letters -/
def D : ℕ → ℕ
  | 0 => 1
  | 1 => 0
  | n + 2 => (n + 1) * (D (n + 1) + D n)

/-- Number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Number of permutations of n items -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

theorem bernoulli_misplacement_6_letters :
  -- Probability of exactly 3 letters placed correctly
  (choose 6 3 * D 3) / permutations 6 = 1 / 18 ∧
  -- Probability of exactly 4 letters placed correctly, given first 2 are correct
  ((choose 4 2) / (2 * permutations 4)) = 1 / 4 ∧
  -- Probability of at least 3 letters placed correctly
  (choose 6 3 * D 3 + choose 6 4 * D 2 + choose 6 6) / permutations 6 = 7 / 90 := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_misplacement_6_letters_l849_84983


namespace NUMINAMATH_CALUDE_vector_addition_scalar_multiplication_l849_84907

theorem vector_addition_scalar_multiplication :
  let v1 : Fin 3 → ℝ := ![-3, 2, -5]
  let v2 : Fin 3 → ℝ := ![1, 7, -3]
  v1 + 2 • v2 = ![-1, 16, -11] := by sorry

end NUMINAMATH_CALUDE_vector_addition_scalar_multiplication_l849_84907


namespace NUMINAMATH_CALUDE_min_value_shifted_sine_l849_84927

theorem min_value_shifted_sine (φ : ℝ) (h_φ : |φ| < π/2) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2*x - π/3)
  ∃ x₀ ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f x₀ ≤ f x ∧ f x₀ = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_shifted_sine_l849_84927


namespace NUMINAMATH_CALUDE_symmetric_latin_square_diagonal_property_l849_84901

/-- A square matrix with odd size, filled with numbers 1 to n, where each row and column contains all numbers exactly once, and which is symmetric about the main diagonal. -/
structure SymmetricLatinSquare (n : ℕ) :=
  (matrix : Fin n → Fin n → Fin n)
  (odd : Odd n)
  (latin_square : ∀ (i j : Fin n), ∃! (k : Fin n), matrix i k = j ∧ ∃! (k : Fin n), matrix k j = i)
  (symmetric : ∀ (i j : Fin n), matrix i j = matrix j i)

/-- The main diagonal of a square matrix contains all numbers from 1 to n exactly once. -/
def diagonal_contains_all (n : ℕ) (matrix : Fin n → Fin n → Fin n) : Prop :=
  ∀ (k : Fin n), ∃! (i : Fin n), matrix i i = k

/-- If a SymmetricLatinSquare exists, then its main diagonal contains all numbers from 1 to n exactly once. -/
theorem symmetric_latin_square_diagonal_property {n : ℕ} (sls : SymmetricLatinSquare n) :
  diagonal_contains_all n sls.matrix :=
sorry

end NUMINAMATH_CALUDE_symmetric_latin_square_diagonal_property_l849_84901


namespace NUMINAMATH_CALUDE_cone_surface_area_l849_84921

/-- The surface area of a cone with slant height 2 and base radius 1 is 3π -/
theorem cone_surface_area :
  let slant_height : ℝ := 2
  let base_radius : ℝ := 1
  let lateral_area := π * base_radius * slant_height
  let base_area := π * base_radius^2
  lateral_area + base_area = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l849_84921


namespace NUMINAMATH_CALUDE_fair_distribution_exists_l849_84996

/-- Represents a piece of ham with its value according to the store scale -/
structure HamPiece where
  value : ℕ

/-- Represents a woman and her belief about the ham's value -/
inductive Woman
  | TrustsHomeScales
  | TrustsStoreScales
  | BelievesEqual

/-- Represents the distribution of ham pieces to women -/
def Distribution := Woman → HamPiece

/-- Checks if a distribution is fair according to each woman's belief -/
def is_fair_distribution (d : Distribution) : Prop :=
  (d Woman.TrustsHomeScales).value ≥ 15 ∧
  (d Woman.TrustsStoreScales).value ≥ 15 ∧
  (d Woman.BelievesEqual).value > 0

/-- The main theorem stating that a fair distribution exists -/
theorem fair_distribution_exists : ∃ (d : Distribution), is_fair_distribution d := by
  sorry

end NUMINAMATH_CALUDE_fair_distribution_exists_l849_84996


namespace NUMINAMATH_CALUDE_alissa_picked_16_flowers_l849_84972

/-- The number of flowers Alissa picked -/
def alissa_flowers : ℕ := sorry

/-- The number of flowers Melissa picked -/
def melissa_flowers : ℕ := sorry

/-- The number of flowers given to their mother -/
def flowers_to_mother : ℕ := 18

/-- The number of flowers left after giving to their mother -/
def flowers_left : ℕ := 14

theorem alissa_picked_16_flowers :
  (alissa_flowers = melissa_flowers) ∧
  (alissa_flowers + melissa_flowers = flowers_to_mother + flowers_left) ∧
  (flowers_to_mother = 18) ∧
  (flowers_left = 14) →
  alissa_flowers = 16 := by sorry

end NUMINAMATH_CALUDE_alissa_picked_16_flowers_l849_84972


namespace NUMINAMATH_CALUDE_stock_fall_amount_l849_84916

/-- Represents the daily change in stock value -/
structure StockChange where
  morning_rise : ℚ
  afternoon_fall : ℚ

/-- Models the stock behavior over time -/
def stock_value (initial_value : ℚ) (daily_change : StockChange) (days : ℕ) : ℚ :=
  initial_value + (daily_change.morning_rise - daily_change.afternoon_fall) * days

/-- Theorem stating the condition for the stock to reach a specific value -/
theorem stock_fall_amount (initial_value target_value : ℚ) (days : ℕ) :
  let morning_rise := 2
  ∀ afternoon_fall : ℚ,
    stock_value initial_value ⟨morning_rise, afternoon_fall⟩ (days - 1) < target_value ∧
    stock_value initial_value ⟨morning_rise, afternoon_fall⟩ days ≥ target_value →
    afternoon_fall = 98 / 99 :=
by sorry

end NUMINAMATH_CALUDE_stock_fall_amount_l849_84916


namespace NUMINAMATH_CALUDE_pentagon_y_coordinate_l849_84962

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The area of a rectangle given its width and height -/
def rectangleArea (width height : ℝ) : ℝ := width * height

/-- The area of a triangle given its base and height -/
def triangleArea (base height : ℝ) : ℝ := 0.5 * base * height

/-- The total area of the pentagon -/
def pentagonArea (p : Pentagon) : ℝ :=
  let rectangleABDE := rectangleArea 4 3
  let triangleBCD := triangleArea 4 (p.C.2 - 3)
  rectangleABDE + triangleBCD

theorem pentagon_y_coordinate (p : Pentagon) 
  (h1 : p.A = (0, 0))
  (h2 : p.B = (0, 3))
  (h3 : p.C = (2, p.C.2))
  (h4 : p.D = (4, 3))
  (h5 : p.E = (4, 0))
  (h6 : pentagonArea p = 35) :
  p.C.2 = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_y_coordinate_l849_84962


namespace NUMINAMATH_CALUDE_suma_work_time_l849_84981

/-- Proves the time taken by Suma to complete the work alone -/
theorem suma_work_time (renu_time suma_renu_time : ℝ) 
  (h1 : renu_time = 8)
  (h2 : suma_renu_time = 3)
  (h3 : renu_time > 0)
  (h4 : suma_renu_time > 0) :
  ∃ (suma_time : ℝ), 
    suma_time > 0 ∧ 
    1 / renu_time + 1 / suma_time = 1 / suma_renu_time ∧ 
    suma_time = 24 / 5 := by
  sorry

end NUMINAMATH_CALUDE_suma_work_time_l849_84981


namespace NUMINAMATH_CALUDE_team_score_proof_l849_84914

def team_size : ℕ := 15
def absent_members : ℕ := 5
def present_members : ℕ := team_size - absent_members
def scores : List ℕ := [4, 6, 2, 8, 3, 5, 10, 3, 7]

theorem team_score_proof :
  present_members = scores.length ∧ scores.sum = 48 := by sorry

end NUMINAMATH_CALUDE_team_score_proof_l849_84914


namespace NUMINAMATH_CALUDE_sum_of_xyz_l849_84944

/-- An arithmetic sequence with six terms where the first term is 4 and the last term is 31 -/
def arithmetic_sequence (x y z : ℝ) : Prop :=
  let d := (31 - 4) / 5
  (y - x = d) ∧ (16 - y = d) ∧ (z - 16 = d)

/-- The theorem stating that the sum of x, y, and z in the given arithmetic sequence is 45.6 -/
theorem sum_of_xyz (x y z : ℝ) (h : arithmetic_sequence x y z) : x + y + z = 45.6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l849_84944


namespace NUMINAMATH_CALUDE_luke_candy_purchase_luke_candy_purchase_result_l849_84931

/-- The number of candy pieces Luke can buy given his tickets and candy cost -/
theorem luke_candy_purchase (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) : ℕ :=
  by
  have h1 : whack_a_mole_tickets = 2 := by sorry
  have h2 : skee_ball_tickets = 13 := by sorry
  have h3 : candy_cost = 3 := by sorry
  
  have total_tickets : ℕ := whack_a_mole_tickets + skee_ball_tickets
  
  exact total_tickets / candy_cost

/-- Proof that Luke can buy 5 pieces of candy -/
theorem luke_candy_purchase_result : luke_candy_purchase 2 13 3 = 5 := by sorry

end NUMINAMATH_CALUDE_luke_candy_purchase_luke_candy_purchase_result_l849_84931


namespace NUMINAMATH_CALUDE_rectangle_opposite_vertex_l849_84990

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Predicate to check if four points form a rectangle --/
def is_rectangle (r : Rectangle) : Prop :=
  let midpoint1 := ((r.v1.1 + r.v3.1) / 2, (r.v1.2 + r.v3.2) / 2)
  let midpoint2 := ((r.v2.1 + r.v4.1) / 2, (r.v2.2 + r.v4.2) / 2)
  midpoint1 = midpoint2

/-- The theorem to be proved --/
theorem rectangle_opposite_vertex 
  (r : Rectangle)
  (h1 : r.v1 = (5, 10))
  (h2 : r.v3 = (15, -6))
  (h3 : r.v2 = (11, 2))
  (h4 : is_rectangle r) :
  r.v4 = (9, 2) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_opposite_vertex_l849_84990


namespace NUMINAMATH_CALUDE_quiz_probability_l849_84943

/-- The probability of answering a multiple-choice question with 5 options correctly -/
def prob_multiple_choice : ℚ := 1 / 5

/-- The probability of answering a true/false question correctly -/
def prob_true_false : ℚ := 1 / 2

/-- The number of true/false questions in the quiz -/
def num_true_false : ℕ := 4

/-- The probability of answering all questions in the quiz correctly -/
def prob_all_correct : ℚ := prob_multiple_choice * prob_true_false ^ num_true_false

theorem quiz_probability :
  prob_all_correct = 1 / 80 := by sorry

end NUMINAMATH_CALUDE_quiz_probability_l849_84943


namespace NUMINAMATH_CALUDE_lcm_gcd_product_24_36_l849_84910

theorem lcm_gcd_product_24_36 : Nat.lcm 24 36 * Nat.gcd 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_24_36_l849_84910


namespace NUMINAMATH_CALUDE_paving_rate_per_sq_meter_l849_84920

/-- Given a rectangular room with length 5.5 m and width 4 m, 
    and a total paving cost of Rs. 16500, 
    prove that the paving rate per square meter is Rs. 750. -/
theorem paving_rate_per_sq_meter 
  (length : ℝ) 
  (width : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 4)
  (h3 : total_cost = 16500) :
  total_cost / (length * width) = 750 := by
  sorry

end NUMINAMATH_CALUDE_paving_rate_per_sq_meter_l849_84920


namespace NUMINAMATH_CALUDE_company_fund_distribution_l849_84925

/-- Represents the company fund distribution problem -/
theorem company_fund_distribution (n : ℕ) 
  (h1 : 50 * n + 130 = 60 * n - 10) : 
  60 * n - 10 = 830 :=
by
  sorry

#check company_fund_distribution

end NUMINAMATH_CALUDE_company_fund_distribution_l849_84925


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l849_84954

def M : Set ℕ := {0, 1, 3}

def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem intersection_of_M_and_N : M ∩ N = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l849_84954


namespace NUMINAMATH_CALUDE_no_good_filling_for_1399_l849_84967

theorem no_good_filling_for_1399 :
  ¬ ∃ (f : Fin 1399 → Fin 2798), 
    (∀ i : Fin 1399, f i ≠ f (i + 1)) ∧ 
    (∀ i j : Fin 1399, i ≠ j → f i ≠ f j) ∧
    (∀ i : Fin 1399, (f i.succ - f i) % 2798 = i.val + 1) :=
by
  sorry

#check no_good_filling_for_1399

end NUMINAMATH_CALUDE_no_good_filling_for_1399_l849_84967


namespace NUMINAMATH_CALUDE_coin_count_proof_l849_84977

/-- Represents the number of nickels -/
def n : ℕ := 7

/-- Represents the number of dimes -/
def d : ℕ := 3 * n

/-- Represents the number of quarters -/
def q : ℕ := 9 * n

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The total value of all coins in cents -/
def total_value : ℕ := 1820

theorem coin_count_proof :
  (n * nickel_value + d * dime_value + q * quarter_value = total_value) →
  (n + d + q = 91) := by
  sorry

end NUMINAMATH_CALUDE_coin_count_proof_l849_84977


namespace NUMINAMATH_CALUDE_tip_percentage_is_22_percent_l849_84903

/-- Calculates the tip percentage given the total amount spent, food price, and sales tax rate. -/
def calculate_tip_percentage (total_spent : ℚ) (food_price : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let sales_tax := food_price * sales_tax_rate
  let tip := total_spent - (food_price + sales_tax)
  (tip / food_price) * 100

/-- Theorem stating that under the given conditions, the tip percentage is 22%. -/
theorem tip_percentage_is_22_percent :
  let total_spent : ℚ := 132
  let food_price : ℚ := 100
  let sales_tax_rate : ℚ := 10 / 100
  calculate_tip_percentage total_spent food_price sales_tax_rate = 22 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_is_22_percent_l849_84903


namespace NUMINAMATH_CALUDE_percentage_good_fruits_is_87_point_6_percent_l849_84904

/-- Calculates the percentage of fruits in good condition given the quantities and spoilage rates --/
def percentageGoodFruits (oranges bananas apples pears : ℕ) 
  (orangesSpoilage bananaSpoilage appleSpoilage pearSpoilage : ℚ) : ℚ :=
  let totalFruits := oranges + bananas + apples + pears
  let goodOranges := oranges - (oranges * orangesSpoilage).floor
  let goodBananas := bananas - (bananas * bananaSpoilage).floor
  let goodApples := apples - (apples * appleSpoilage).floor
  let goodPears := pears - (pears * pearSpoilage).floor
  let totalGoodFruits := goodOranges + goodBananas + goodApples + goodPears
  (totalGoodFruits : ℚ) / (totalFruits : ℚ) * 100

/-- Theorem stating that the percentage of good fruits is 87.6% given the problem conditions --/
theorem percentage_good_fruits_is_87_point_6_percent :
  percentageGoodFruits 600 400 800 200 (15/100) (3/100) (12/100) (25/100) = 876/10 := by
  sorry


end NUMINAMATH_CALUDE_percentage_good_fruits_is_87_point_6_percent_l849_84904


namespace NUMINAMATH_CALUDE_solution_of_system_l849_84950

theorem solution_of_system (x y : ℝ) : 
  (x^2 - x*y + y^2 = 49*(x - y) ∧ x^2 + x*y + y^2 = 76*(x + y)) ↔ 
  ((x = 0 ∧ y = 0) ∨ (x = 40 ∧ y = -24)) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l849_84950


namespace NUMINAMATH_CALUDE_number_problem_l849_84929

theorem number_problem (x : ℤ) (h : x + 1015 = 3016) : x = 2001 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l849_84929


namespace NUMINAMATH_CALUDE_circles_intersect_l849_84936

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Theorem statement
theorem circles_intersect : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l849_84936


namespace NUMINAMATH_CALUDE_one_hundred_ten_billion_scientific_notation_l849_84964

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem one_hundred_ten_billion_scientific_notation :
  toScientificNotation 110000000000 = ScientificNotation.mk 1.1 11 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_one_hundred_ten_billion_scientific_notation_l849_84964


namespace NUMINAMATH_CALUDE_money_division_l849_84995

/-- Represents the share ratios of five individuals over five weeks -/
structure ShareRatios :=
  (a b c d e : Fin 5 → ℚ)

/-- Calculates the total ratio for a given week -/
def totalRatio (sr : ShareRatios) (week : Fin 5) : ℚ :=
  sr.a week + sr.b week + sr.c week + sr.d week + sr.e week

/-- Defines the initial ratios and weekly changes -/
def initialRatios : ShareRatios :=
  { a := λ _ => 1,
    b := λ w => 75/100 - w.val * 5/100,
    c := λ w => 60/100 - w.val * 5/100,
    d := λ w => 45/100 - w.val * 5/100,
    e := λ w => 30/100 + w.val * 15/100 }

/-- Theorem statement -/
theorem money_division (sr : ShareRatios) (h1 : sr = initialRatios) 
    (h2 : sr.e 4 * (totalRatio sr 0 / sr.e 4) = 413.33) : 
  sr.e 4 = 120 → totalRatio sr 0 = 413.33 := by
  sorry


end NUMINAMATH_CALUDE_money_division_l849_84995


namespace NUMINAMATH_CALUDE_quadratic_equation_standard_form_l849_84998

theorem quadratic_equation_standard_form :
  ∀ x : ℝ, (2*x - 1)^2 = (x + 1)*(3*x + 4) ↔ x^2 - 11*x - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_standard_form_l849_84998


namespace NUMINAMATH_CALUDE_fraction_simplification_l849_84949

theorem fraction_simplification : 
  (4 * 6) / (12 * 18) * (9 * 12 * 18) / (4 * 6 * 9^2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l849_84949


namespace NUMINAMATH_CALUDE_base_conversion_1729_to_base7_l849_84999

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 7^3 + d₂ * 7^2 + d₁ * 7^1 + d₀ * 7^0

/-- States that 1729 in base 10 is equal to 5020 in base 7 --/
theorem base_conversion_1729_to_base7 :
  1729 = base7ToBase10 5 0 2 0 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1729_to_base7_l849_84999


namespace NUMINAMATH_CALUDE_even_function_inequality_l849_84935

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

theorem even_function_inequality (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  f m (Real.sqrt 3) < f m (-Real.sqrt 2) ∧ f m (-Real.sqrt 2) < f m (-1) :=
by sorry

end NUMINAMATH_CALUDE_even_function_inequality_l849_84935


namespace NUMINAMATH_CALUDE_sachins_age_l849_84932

theorem sachins_age (sachin_age rahul_age : ℝ) 
  (h1 : rahul_age = sachin_age + 9)
  (h2 : sachin_age / rahul_age = 7 / 9) :
  sachin_age = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_sachins_age_l849_84932


namespace NUMINAMATH_CALUDE_increase_by_percentage_l849_84939

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 700 ∧ percentage = 85 ∧ final = initial * (1 + percentage / 100) →
  final = 1295 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l849_84939


namespace NUMINAMATH_CALUDE_animal_shelter_count_l849_84952

/-- The number of cats received by the animal shelter -/
def num_cats : ℕ := 40

/-- The difference between the number of cats and dogs -/
def cat_dog_difference : ℕ := 20

/-- The total number of animals received by the shelter -/
def total_animals : ℕ := num_cats + (num_cats - cat_dog_difference)

theorem animal_shelter_count : total_animals = 60 := by
  sorry

end NUMINAMATH_CALUDE_animal_shelter_count_l849_84952


namespace NUMINAMATH_CALUDE_gcd_repeating_even_three_digit_l849_84989

theorem gcd_repeating_even_three_digit : 
  ∃ g : ℕ, ∀ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ Even n → 
    g = Nat.gcd (1001 * n) (Nat.gcd (1001 * (n + 2)) (1001 * (n + 4))) ∧ 
    g = 2002 := by
  sorry

end NUMINAMATH_CALUDE_gcd_repeating_even_three_digit_l849_84989


namespace NUMINAMATH_CALUDE_problem_statement_l849_84993

theorem problem_statement (a b c : ℤ) 
  (h1 : 0 < c) (h2 : c < 90) 
  (h3 : Real.sqrt (9 - 8 * Real.sin (50 * π / 180)) = a + b * Real.sin (c * π / 180)) :
  (a + b) / c = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l849_84993


namespace NUMINAMATH_CALUDE_min_value_theorem_l849_84961

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * Real.sqrt x + 2 / x^2 ≥ 5 ∧
  (3 * Real.sqrt x + 2 / x^2 = 5 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l849_84961


namespace NUMINAMATH_CALUDE_log_relation_l849_84971

theorem log_relation (x k : ℝ) (h1 : Real.log 3 / Real.log 4 = x) (h2 : Real.log 64 / Real.log 2 = k * x) : k = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l849_84971


namespace NUMINAMATH_CALUDE_right_triangle_area_rational_l849_84970

/-- A right-angled triangle with integer coordinates -/
structure RightTriangle where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The area of a right-angled triangle with integer coordinates -/
def area (t : RightTriangle) : ℚ :=
  (|t.a * t.d - t.b * t.c| : ℚ) / 2

/-- Theorem: The area of a right-angled triangle with integer coordinates is always rational -/
theorem right_triangle_area_rational (t : RightTriangle) : 
  ∃ (q : ℚ), area t = q :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_rational_l849_84970


namespace NUMINAMATH_CALUDE_sport_water_amount_l849_84956

/-- Represents a flavored drink formulation -/
structure Formulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the drink -/
def standard : Formulation :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport : Formulation :=
  { flavoring := 1, corn_syrup := 4, water := 60 }

theorem sport_water_amount (corn_syrup_amount : ℚ) :
  corn_syrup_amount = 5 →
  sport.water / sport.corn_syrup * corn_syrup_amount = 75 := by
sorry

end NUMINAMATH_CALUDE_sport_water_amount_l849_84956


namespace NUMINAMATH_CALUDE_total_dolls_count_l849_84985

/-- The number of dolls in a big box -/
def dolls_per_big_box : ℕ := 7

/-- The number of dolls in a small box -/
def dolls_per_small_box : ℕ := 4

/-- The number of big boxes -/
def num_big_boxes : ℕ := 5

/-- The number of small boxes -/
def num_small_boxes : ℕ := 9

/-- The total number of dolls in all boxes -/
def total_dolls : ℕ := dolls_per_big_box * num_big_boxes + dolls_per_small_box * num_small_boxes

theorem total_dolls_count : total_dolls = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_count_l849_84985


namespace NUMINAMATH_CALUDE_algae_coverage_day_18_and_19_l849_84942

/-- Represents the coverage of algae on the pond on a given day -/
def algaeCoverage (day : ℕ) : ℚ :=
  (1 : ℚ) / 3^(20 - day)

/-- The problem statement -/
theorem algae_coverage_day_18_and_19 :
  algaeCoverage 18 < (1 : ℚ) / 4 ∧ (1 : ℚ) / 4 < algaeCoverage 19 := by
  sorry

#eval algaeCoverage 18  -- Expected: 1/9
#eval algaeCoverage 19  -- Expected: 1/3

end NUMINAMATH_CALUDE_algae_coverage_day_18_and_19_l849_84942


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l849_84960

theorem min_value_squared_sum (x y z : ℝ) (h : 2*x + 3*y + z = 7) :
  x^2 + y^2 + z^2 ≥ 7/2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l849_84960


namespace NUMINAMATH_CALUDE_total_stickers_is_36_l849_84918

/-- The number of stickers Elizabeth uses on her water bottles -/
def total_stickers : ℕ :=
  let initial_bottles : ℕ := 20
  let lost_school : ℕ := 5
  let found_park : ℕ := 3
  let stolen_dance : ℕ := 4
  let misplaced_library : ℕ := 2
  let acquired_friend : ℕ := 6
  let stickers_school : ℕ := 4
  let stickers_dance : ℕ := 3
  let stickers_library : ℕ := 2

  let school_stickers := lost_school * stickers_school
  let dance_stickers := stolen_dance * stickers_dance
  let library_stickers := misplaced_library * stickers_library

  school_stickers + dance_stickers + library_stickers

theorem total_stickers_is_36 : total_stickers = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_is_36_l849_84918


namespace NUMINAMATH_CALUDE_contrapositive_odd_product_l849_84908

theorem contrapositive_odd_product (a b : ℤ) :
  (¬(Odd (a * b)) → ¬(Odd a ∧ Odd b)) ↔
  ((Odd a ∧ Odd b) → Odd (a * b)) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_odd_product_l849_84908


namespace NUMINAMATH_CALUDE_cos_minus_sin_value_l849_84965

theorem cos_minus_sin_value (α : Real) 
  (h1 : π/4 < α) (h2 : α < π/2) (h3 : Real.sin (2 * α) = 24/25) : 
  Real.cos α - Real.sin α = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_value_l849_84965


namespace NUMINAMATH_CALUDE_num_different_selections_eq_six_l849_84992

/-- Represents the set of attractions -/
inductive Attraction : Type
  | A : Attraction
  | B : Attraction
  | C : Attraction

/-- Represents a selection of two attractions -/
def Selection := Finset Attraction

/-- The set of all possible selections -/
def all_selections : Finset Selection :=
  sorry

/-- Predicate to check if two selections are different -/
def different_selections (s1 s2 : Selection) : Prop :=
  s1 ≠ s2

/-- The number of ways two people can choose different selections -/
def num_different_selections : ℕ :=
  sorry

/-- Theorem: The number of ways two people can choose two out of three attractions,
    such that their choices are different, is equal to 6 -/
theorem num_different_selections_eq_six :
  num_different_selections = 6 :=
sorry

end NUMINAMATH_CALUDE_num_different_selections_eq_six_l849_84992


namespace NUMINAMATH_CALUDE_juan_speed_l849_84968

/-- Given a distance of 80 miles and a time of 8 hours, prove that the speed is 10 miles per hour. -/
theorem juan_speed (distance : ℝ) (time : ℝ) (h1 : distance = 80) (h2 : time = 8) :
  distance / time = 10 := by
  sorry

end NUMINAMATH_CALUDE_juan_speed_l849_84968


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l849_84906

/-- The parabola function -/
def f (x : ℝ) : ℝ := -(x + 2)^2 + 6

/-- The y-axis -/
def y_axis : Set ℝ := {x | x = 0}

/-- Theorem: The intersection point of the parabola and the y-axis is (0, 2) -/
theorem parabola_y_axis_intersection :
  ∃! p : ℝ × ℝ, p.1 ∈ y_axis ∧ p.2 = f p.1 ∧ p = (0, 2) := by
sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l849_84906
