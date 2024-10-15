import Mathlib

namespace NUMINAMATH_CALUDE_data_average_problem_l548_54864

theorem data_average_problem (x : ℝ) : 
  (6 + x + 2 + 4) / 4 = 5 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_data_average_problem_l548_54864


namespace NUMINAMATH_CALUDE_us_stripes_count_l548_54817

/-- The number of stars on the US flag -/
def us_stars : ℕ := 50

/-- The number of circles on Pete's flag -/
def pete_circles : ℕ := us_stars / 2 - 3

/-- The number of squares on Pete's flag as a function of US flag stripes -/
def pete_squares (s : ℕ) : ℕ := 2 * s + 6

/-- The total number of shapes on Pete's flag -/
def pete_total_shapes : ℕ := 54

/-- Theorem: The number of stripes on the US flag is 13 -/
theorem us_stripes_count : 
  ∃ (s : ℕ), s = 13 ∧ pete_circles + pete_squares s = pete_total_shapes :=
sorry

end NUMINAMATH_CALUDE_us_stripes_count_l548_54817


namespace NUMINAMATH_CALUDE_square_gt_of_abs_lt_l548_54827

theorem square_gt_of_abs_lt (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_gt_of_abs_lt_l548_54827


namespace NUMINAMATH_CALUDE_unique_solution_in_interval_l548_54889

theorem unique_solution_in_interval (x : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  ((2 - Real.sin (2 * x)) * Real.sin (x + Real.pi / 4) = 1) ↔
  (x = Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_in_interval_l548_54889


namespace NUMINAMATH_CALUDE_consecutive_even_sum_42_square_diff_l548_54808

theorem consecutive_even_sum_42_square_diff (n m : ℤ) : 
  (Even n) → (Even m) → (m = n + 2) → (n + m = 42) → 
  (m ^ 2 - n ^ 2 = 84) := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_42_square_diff_l548_54808


namespace NUMINAMATH_CALUDE_x1_range_proof_l548_54898

theorem x1_range_proof (f : ℝ → ℝ) (h_incr : Monotone f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 1 → f x₁ + f 0 > f x₂ + f 1) →
  ∀ x₁ : ℝ, (∃ x₂ : ℝ, x₁ + x₂ = 1 ∧ f x₁ + f 0 > f x₂ + f 1) → x₁ > 1 :=
by sorry

end NUMINAMATH_CALUDE_x1_range_proof_l548_54898


namespace NUMINAMATH_CALUDE_demand_analysis_l548_54807

def f (x : ℕ) : ℚ := (1 / 150) * x * (x + 1) * (35 - 2 * x)

def g (x : ℕ) : ℚ := (1 / 25) * x * (12 - x)

theorem demand_analysis (x : ℕ) (h : x ≤ 12) :
  -- 1. The demand in the x-th month
  g x = f x - f (x - 1) ∧
  -- 2. The maximum monthly demand occurs when x = 6 and is equal to 36/25
  (∀ y : ℕ, y ≤ 12 → g y ≤ g 6) ∧ g 6 = 36 / 25 ∧
  -- 3. The total demand for the first 6 months is 161/25
  f 6 = 161 / 25 :=
sorry

end NUMINAMATH_CALUDE_demand_analysis_l548_54807


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l548_54822

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.03 * L) (h2 : B' = B * (1 + 0.06)) :
  L' * B' = 1.0918 * (L * B) :=
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l548_54822


namespace NUMINAMATH_CALUDE_sum_1_to_50_base6_l548_54811

/-- Converts a base 10 number to base 6 --/
def toBase6 (n : ℕ) : ℕ := sorry

/-- Converts a base 6 number to base 10 --/
def fromBase6 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of integers from 1 to n in base 6 --/
def sumInBase6 (n : ℕ) : ℕ := sorry

theorem sum_1_to_50_base6 :
  sumInBase6 (fromBase6 50) = toBase6 55260 := by sorry

end NUMINAMATH_CALUDE_sum_1_to_50_base6_l548_54811


namespace NUMINAMATH_CALUDE_bus_left_seats_l548_54869

/-- Represents the seating configuration of a bus -/
structure BusSeating where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeatCapacity : ℕ
  seatCapacity : ℕ
  totalCapacity : ℕ

/-- The bus seating configuration satisfies the given conditions -/
def validBusSeating (bus : BusSeating) : Prop :=
  bus.rightSeats = bus.leftSeats - 3 ∧
  bus.backSeatCapacity = 12 ∧
  bus.seatCapacity = 3 ∧
  bus.totalCapacity = 93 ∧
  bus.totalCapacity = bus.seatCapacity * (bus.leftSeats + bus.rightSeats) + bus.backSeatCapacity

theorem bus_left_seats (bus : BusSeating) (h : validBusSeating bus) : bus.leftSeats = 15 := by
  sorry

end NUMINAMATH_CALUDE_bus_left_seats_l548_54869


namespace NUMINAMATH_CALUDE_tuesday_rainfall_amount_l548_54828

/-- The amount of rainfall on Monday in inches -/
def monday_rain : ℝ := 0.9

/-- The difference in rainfall between Monday and Tuesday in inches -/
def rain_difference : ℝ := 0.7

/-- The amount of rainfall on Tuesday in inches -/
def tuesday_rain : ℝ := monday_rain - rain_difference

/-- Theorem stating that the amount of rain on Tuesday is 0.2 inches -/
theorem tuesday_rainfall_amount : tuesday_rain = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_rainfall_amount_l548_54828


namespace NUMINAMATH_CALUDE_binary_1111111111_equals_1023_l548_54867

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1111111111 -/
def binary_1111111111 : List Bool :=
  [true, true, true, true, true, true, true, true, true, true]

theorem binary_1111111111_equals_1023 :
  binary_to_decimal binary_1111111111 = 1023 := by
  sorry

end NUMINAMATH_CALUDE_binary_1111111111_equals_1023_l548_54867


namespace NUMINAMATH_CALUDE_melody_reading_pages_l548_54862

theorem melody_reading_pages (english : ℕ) (civics : ℕ) (chinese : ℕ) (science : ℕ) :
  english = 20 →
  civics = 8 →
  chinese = 12 →
  (english / 4 + civics / 4 + chinese / 4 + science / 4 : ℚ) = 14 →
  science = 16 := by
sorry

end NUMINAMATH_CALUDE_melody_reading_pages_l548_54862


namespace NUMINAMATH_CALUDE_slope_range_l548_54895

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

-- Define the line equation
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x - 3

-- Define the intersection condition
def intersects (m : ℝ) : Prop := ∃ x y : ℝ, ellipse x y ∧ line m x y

-- State the theorem
theorem slope_range (m : ℝ) : 
  intersects m ↔ m ≤ -Real.sqrt (1/5) ∨ m ≥ Real.sqrt (1/5) :=
sorry

end NUMINAMATH_CALUDE_slope_range_l548_54895


namespace NUMINAMATH_CALUDE_cube_root_implies_value_l548_54849

theorem cube_root_implies_value (x : ℝ) : 
  (2 * x - 14) ^ (1/3 : ℝ) = -2 → 2 * x + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_implies_value_l548_54849


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l548_54847

/-- Given that y and x are inversely proportional -/
def inversely_proportional (y x : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y * x = k

/-- The theorem to prove -/
theorem inverse_proportion_problem (y₁ y₂ : ℝ) :
  inversely_proportional y₁ 4 ∧ y₁ = 30 →
  inversely_proportional y₂ 10 →
  y₂ = 12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l548_54847


namespace NUMINAMATH_CALUDE_female_students_count_l548_54816

theorem female_students_count (total_average : ℝ) (male_count : ℕ) (male_average : ℝ) (female_average : ℝ)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 83)
  (h4 : female_average = 92) :
  ∃ (female_count : ℕ),
    female_count = 28 ∧
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average :=
by sorry

end NUMINAMATH_CALUDE_female_students_count_l548_54816


namespace NUMINAMATH_CALUDE_geometric_series_sum_l548_54891

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  let S := (a * (1 - r^n)) / (1 - r)
  a = 1 → r = 1/4 → n = 6 → S = 1365/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l548_54891


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l548_54861

/-- A right triangle with a point on its hypotenuse and parallel lines dividing it -/
structure DividedRightTriangle where
  -- The rectangle formed by the parallel lines
  rectangle_area : ℝ
  -- The area of one of the smaller right triangles
  small_triangle_area : ℝ
  -- The condition that the area of one small triangle is n times the rectangle area
  area_condition : ∃ n : ℝ, small_triangle_area = n * rectangle_area

/-- The theorem stating the ratio of areas -/
theorem area_ratio_theorem (t : DividedRightTriangle) : 
  ∃ n : ℝ, t.small_triangle_area = n * t.rectangle_area → 
  ∃ other_triangle_area : ℝ, other_triangle_area / t.rectangle_area = 1 / (4 * n) :=
sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l548_54861


namespace NUMINAMATH_CALUDE_solve_equation_l548_54826

theorem solve_equation : ∃ x : ℚ, (3 * x - 4) / 6 = 15 ∧ x = 94 / 3 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l548_54826


namespace NUMINAMATH_CALUDE_probability_A_and_B_selected_is_three_tenths_l548_54875

def total_students : ℕ := 5
def selected_students : ℕ := 3

def probability_A_and_B_selected : ℚ :=
  (Nat.choose (total_students - 2) (selected_students - 2)) /
  (Nat.choose total_students selected_students)

theorem probability_A_and_B_selected_is_three_tenths :
  probability_A_and_B_selected = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_A_and_B_selected_is_three_tenths_l548_54875


namespace NUMINAMATH_CALUDE_greatest_prime_factor_f_28_l548_54837

def f (m : ℕ) : ℕ := Finset.prod (Finset.range (m/2 + 1)) (λ i => 2 * i)

theorem greatest_prime_factor_f_28 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ f 28 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ f 28 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_f_28_l548_54837


namespace NUMINAMATH_CALUDE_orthocenter_diameter_bisection_l548_54863

/-- A point in a 2D plane. -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A triangle defined by three points. -/
structure Triangle :=
  (A B C : Point)

/-- The orthocenter of a triangle. -/
def orthocenter (t : Triangle) : Point := sorry

/-- The circumcircle of a triangle. -/
def circumcircle (t : Triangle) : Set Point := sorry

/-- A diameter of a circle. -/
def is_diameter (A A' : Point) (circle : Set Point) : Prop := sorry

/-- A segment bisects another segment. -/
def bisects (P Q : Point) (R S : Point) : Prop := sorry

/-- Main theorem: If H is the orthocenter of triangle ABC and AA' is a diameter
    of its circumcircle, then A'H bisects the side BC. -/
theorem orthocenter_diameter_bisection
  (t : Triangle) (A' : Point) (H : Point) :
  H = orthocenter t →
  is_diameter t.A A' (circumcircle t) →
  bisects A' H t.B t.C :=
sorry

end NUMINAMATH_CALUDE_orthocenter_diameter_bisection_l548_54863


namespace NUMINAMATH_CALUDE_tom_annual_cost_l548_54886

/-- Calculates the annual cost of medication and doctor visits for Tom --/
def annual_cost (pills_per_day : ℕ) (doctor_visit_interval_months : ℕ) (doctor_visit_cost : ℕ) 
  (pill_cost : ℕ) (insurance_coverage_percent : ℕ) : ℕ :=
  let daily_medication_cost := pills_per_day * (pill_cost * (100 - insurance_coverage_percent) / 100)
  let annual_medication_cost := daily_medication_cost * 365
  let annual_doctor_visits := 12 / doctor_visit_interval_months
  let annual_doctor_cost := annual_doctor_visits * doctor_visit_cost
  annual_medication_cost + annual_doctor_cost

/-- Theorem stating that Tom's annual cost is $1530 --/
theorem tom_annual_cost : 
  annual_cost 2 6 400 5 80 = 1530 := by
  sorry

end NUMINAMATH_CALUDE_tom_annual_cost_l548_54886


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l548_54809

/-- Given that the solution set of ax² + 5x - 2 > 0 is {x | 1/2 < x < 2},
    prove that a = -2 and the solution set of ax² + 5x + a² - 1 > 0 is {x | -1/2 < x < 3} -/
theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) → 
  (a = -2 ∧ ∀ x : ℝ, a*x^2 + 5*x + a^2 - 1 > 0 ↔ -1/2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l548_54809


namespace NUMINAMATH_CALUDE_symmetrical_line_slope_range_l548_54880

/-- Given a line l: y = kx - 1 intersecting with x + y - 1 = 0,
    the range of k for which a symmetrical line can be derived is (1, +∞) -/
theorem symmetrical_line_slope_range (k : ℝ) : 
  (∃ (x y : ℝ), y = k * x - 1 ∧ x + y - 1 = 0) →
  (∃ (m : ℝ), m ≠ k ∧ ∃ (x₀ y₀ : ℝ), (∀ (x y : ℝ), y - y₀ = m * (x - x₀) ↔ y = k * x - 1)) ↔
  k > 1 :=
sorry

end NUMINAMATH_CALUDE_symmetrical_line_slope_range_l548_54880


namespace NUMINAMATH_CALUDE_min_value_X_l548_54868

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- Converts a four-digit number to its integer representation -/
def fourDigitToInt (a b c d : Digit) : ℕ :=
  1000 * (a.val + 1) + 100 * (b.val + 1) + 10 * (c.val + 1) + (d.val + 1)

/-- Converts a two-digit number to its integer representation -/
def twoDigitToInt (e f : Digit) : ℕ :=
  10 * (e.val + 1) + (f.val + 1)

theorem min_value_X (a b c d e f g h i : Digit) 
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i)
  (h2 : b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i)
  (h3 : c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i)
  (h4 : d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i)
  (h5 : e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i)
  (h6 : f ≠ g ∧ f ≠ h ∧ f ≠ i)
  (h7 : g ≠ h ∧ g ≠ i)
  (h8 : h ≠ i) :
  ∃ (x : ℕ), x = fourDigitToInt a b c d + twoDigitToInt e f * twoDigitToInt g h - (i.val + 1) ∧
    x ≥ 2369 ∧
    (∀ (a' b' c' d' e' f' g' h' i' : Digit),
      (a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ a' ≠ e' ∧ a' ≠ f' ∧ a' ≠ g' ∧ a' ≠ h' ∧ a' ≠ i') →
      (b' ≠ c' ∧ b' ≠ d' ∧ b' ≠ e' ∧ b' ≠ f' ∧ b' ≠ g' ∧ b' ≠ h' ∧ b' ≠ i') →
      (c' ≠ d' ∧ c' ≠ e' ∧ c' ≠ f' ∧ c' ≠ g' ∧ c' ≠ h' ∧ c' ≠ i') →
      (d' ≠ e' ∧ d' ≠ f' ∧ d' ≠ g' ∧ d' ≠ h' ∧ d' ≠ i') →
      (e' ≠ f' ∧ e' ≠ g' ∧ e' ≠ h' ∧ e' ≠ i') →
      (f' ≠ g' ∧ f' ≠ h' ∧ f' ≠ i') →
      (g' ≠ h' ∧ g' ≠ i') →
      (h' ≠ i') →
      x ≤ fourDigitToInt a' b' c' d' + twoDigitToInt e' f' * twoDigitToInt g' h' - (i'.val + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_X_l548_54868


namespace NUMINAMATH_CALUDE_molar_mass_calculation_l548_54840

/-- Given a chemical compound where 10 moles weigh 2070 grams, 
    prove that its molar mass is 207 grams/mole. -/
theorem molar_mass_calculation (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 2070)
  (h2 : num_moles = 10) :
  total_weight / num_moles = 207 := by
  sorry

end NUMINAMATH_CALUDE_molar_mass_calculation_l548_54840


namespace NUMINAMATH_CALUDE_fruit_eating_arrangements_l548_54879

def num_apples : ℕ := 4
def num_oranges : ℕ := 2
def num_bananas : ℕ := 2

def total_fruits : ℕ := num_apples + num_oranges + num_bananas

theorem fruit_eating_arrangements :
  (total_fruits.factorial) / (num_oranges.factorial * num_bananas.factorial) = 6 :=
by sorry

end NUMINAMATH_CALUDE_fruit_eating_arrangements_l548_54879


namespace NUMINAMATH_CALUDE_miss_darlington_blueberries_l548_54833

/-- The number of blueberries in Miss Darlington's basket problem -/
theorem miss_darlington_blueberries 
  (initial_basket : ℕ) 
  (additional_baskets : ℕ) 
  (h1 : initial_basket = 20)
  (h2 : additional_baskets = 9) : 
  initial_basket + additional_baskets * initial_basket = 200 := by
  sorry

end NUMINAMATH_CALUDE_miss_darlington_blueberries_l548_54833


namespace NUMINAMATH_CALUDE_pentagon_fifth_angle_l548_54831

/-- A pentagon with four known angles -/
structure Pentagon where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 + angle4 + angle5 = 540

/-- The theorem to prove -/
theorem pentagon_fifth_angle (p : Pentagon) 
  (h1 : p.angle1 = 270)
  (h2 : p.angle2 = 70)
  (h3 : p.angle3 = 60)
  (h4 : p.angle4 = 90) :
  p.angle5 = 50 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_fifth_angle_l548_54831


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l548_54812

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := fun x ↦ 2 * x^2 - x - 1
  {x : ℝ | f x > 0} = {x : ℝ | x < -1/2 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l548_54812


namespace NUMINAMATH_CALUDE_simplify_expression_l548_54800

theorem simplify_expression (x : ℝ) : 4*x + 6*x^3 + 8 - (3 - 6*x^3 - 4*x) = 12*x^3 + 8*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l548_54800


namespace NUMINAMATH_CALUDE_solve_y_l548_54852

theorem solve_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 10) : y = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_y_l548_54852


namespace NUMINAMATH_CALUDE_final_sum_is_212_l548_54838

/-- Represents a person in the debt settlement problem -/
inductive Person
| Earl
| Fred
| Greg
| Hannah

/-- Represents the initial amount of money each person has -/
def initial_amount (p : Person) : Int :=
  match p with
  | Person.Earl => 90
  | Person.Fred => 48
  | Person.Greg => 36
  | Person.Hannah => 72

/-- Represents the amount one person owes to another -/
def debt (debtor receiver : Person) : Int :=
  match debtor, receiver with
  | Person.Earl, Person.Fred => 28
  | Person.Earl, Person.Hannah => 30
  | Person.Fred, Person.Greg => 32
  | Person.Fred, Person.Hannah => 10
  | Person.Greg, Person.Earl => 40
  | Person.Greg, Person.Hannah => 20
  | Person.Hannah, Person.Greg => 15
  | Person.Hannah, Person.Earl => 25
  | _, _ => 0

/-- Calculates the final amount a person has after settling all debts -/
def final_amount (p : Person) : Int :=
  initial_amount p
  + (debt Person.Earl p + debt Person.Fred p + debt Person.Greg p + debt Person.Hannah p)
  - (debt p Person.Earl + debt p Person.Fred + debt p Person.Greg + debt p Person.Hannah)

/-- Theorem stating that the sum of Greg's, Earl's, and Hannah's money after settling debts is $212 -/
theorem final_sum_is_212 :
  final_amount Person.Greg + final_amount Person.Earl + final_amount Person.Hannah = 212 :=
by sorry

end NUMINAMATH_CALUDE_final_sum_is_212_l548_54838


namespace NUMINAMATH_CALUDE_octagon_area_in_circle_l548_54815

theorem octagon_area_in_circle (circle_area : ℝ) (octagon_area : ℝ) :
  circle_area = 256 * Real.pi →
  octagon_area = 8 * (1 / 2 * (Real.sqrt (circle_area / Real.pi))^2 * Real.sin (Real.pi / 4)) →
  octagon_area = 256 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_octagon_area_in_circle_l548_54815


namespace NUMINAMATH_CALUDE_john_nails_count_l548_54876

/-- Calculates the total number of nails used in John's house wall construction --/
def total_nails (nails_per_plank : ℕ) (additional_nails : ℕ) (num_planks : ℕ) : ℕ :=
  nails_per_plank * num_planks + additional_nails

/-- Proves that John used 11 nails in total for his house wall construction --/
theorem john_nails_count :
  let nails_per_plank : ℕ := 3
  let additional_nails : ℕ := 8
  let num_planks : ℕ := 1
  total_nails nails_per_plank additional_nails num_planks = 11 := by
  sorry

end NUMINAMATH_CALUDE_john_nails_count_l548_54876


namespace NUMINAMATH_CALUDE_tangent_line_at_neg_one_l548_54823

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x - 1

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Theorem statement
theorem tangent_line_at_neg_one (a : ℝ) :
  f_derivative a 1 = 1 →
  ∃ y : ℝ, 9 * (-1) - y + 3 = 0 ∧
  y = f a (-1) ∧
  f_derivative a (-1) = 9 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_neg_one_l548_54823


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l548_54856

theorem largest_inscribed_triangle_area (r : ℝ) (h : r = 8) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_height := r
  let triangle_area := (1/2) * diameter * max_height
  triangle_area = 64 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l548_54856


namespace NUMINAMATH_CALUDE_ede_viv_properties_l548_54871

theorem ede_viv_properties :
  let ede : ℕ := 242
  let viv : ℕ := 303
  (100 ≤ ede ∧ ede < 1000) ∧  -- EDE is a three-digit number
  (100 ≤ viv ∧ viv < 1000) ∧  -- VIV is a three-digit number
  (ede ≠ viv) ∧               -- EDE and VIV are distinct
  (Nat.gcd ede viv = 1) ∧     -- EDE and VIV are relatively prime
  (ede / viv = 242 / 303) ∧   -- The fraction is correct
  (∃ n : ℕ, (1000 * ede) / viv = 798 + n * 999) -- The decimal repeats as 0.798679867...
  := by sorry

end NUMINAMATH_CALUDE_ede_viv_properties_l548_54871


namespace NUMINAMATH_CALUDE_initial_eggs_count_l548_54874

/-- Given a person shares eggs among 8 friends, with each friend receiving 2 eggs,
    prove that the initial number of eggs is 16. -/
theorem initial_eggs_count (num_friends : ℕ) (eggs_per_friend : ℕ) 
  (h1 : num_friends = 8) (h2 : eggs_per_friend = 2) : 
  num_friends * eggs_per_friend = 16 := by
  sorry

#check initial_eggs_count

end NUMINAMATH_CALUDE_initial_eggs_count_l548_54874


namespace NUMINAMATH_CALUDE_probability_increases_l548_54842

/-- The probability of player A winning a game of 2n rounds -/
noncomputable def P (n : ℕ) : ℝ :=
  1/2 * (1 - (Nat.choose (2*n) n : ℝ) / 2^(2*n))

/-- Theorem stating that the probability of winning increases with the number of rounds -/
theorem probability_increases (n : ℕ) : P (n+1) > P n := by
  sorry

end NUMINAMATH_CALUDE_probability_increases_l548_54842


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l548_54894

theorem shirt_price_calculation (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  final_price = 105 ∧ 
  discount1 = 19.954259576901087 ∧ 
  discount2 = 12.55 →
  ∃ (original_price : ℝ), 
    original_price = 150 ∧ 
    final_price = original_price * (1 - discount1 / 100) * (1 - discount2 / 100) :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l548_54894


namespace NUMINAMATH_CALUDE_tank_fill_time_l548_54836

/-- Represents a machine that can fill or empty a tank -/
structure Machine where
  fillRate : ℚ  -- Rate at which the machine fills the tank (fraction per minute)
  emptyRate : ℚ -- Rate at which the machine empties the tank (fraction per minute)

/-- Calculates the net rate of a machine that alternates between filling and emptying -/
def alternatingRate (fillTime emptyTime cycleTime : ℚ) : ℚ :=
  (fillTime / cycleTime) * (1 / fillTime) + (emptyTime / cycleTime) * (-1 / emptyTime)

/-- The main theorem stating the time to fill the tank -/
theorem tank_fill_time :
  let machineA : Machine := ⟨1/25, 0⟩
  let machineB : Machine := ⟨0, 1/50⟩
  let machineC : Machine := ⟨alternatingRate 5 5 10, 0⟩
  let combinedRate := machineA.fillRate - machineB.emptyRate + machineC.fillRate
  let remainingVolume := 1/2
  ⌈remainingVolume / combinedRate⌉ = 20 := by
  sorry


end NUMINAMATH_CALUDE_tank_fill_time_l548_54836


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_square_sum_l548_54882

theorem cube_sum_given_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_square_sum_l548_54882


namespace NUMINAMATH_CALUDE_star_equation_solution_l548_54830

/-- Custom binary operation -/
def star (a b : ℝ) : ℝ := a * b + a - 2 * b

/-- Theorem stating that if 3 star m = 17, then m = 14 -/
theorem star_equation_solution :
  ∀ m : ℝ, star 3 m = 17 → m = 14 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l548_54830


namespace NUMINAMATH_CALUDE_new_homes_theorem_l548_54818

/-- The number of original trailer homes -/
def original_homes : ℕ := 30

/-- The initial average age of original trailer homes 5 years ago -/
def initial_avg_age : ℚ := 15

/-- The current average age of all trailer homes -/
def current_avg_age : ℚ := 12

/-- The number of years that have passed -/
def years_passed : ℕ := 5

/-- Function to calculate the number of new trailer homes added -/
def new_homes_added : ℚ :=
  (original_homes * (initial_avg_age + years_passed) - original_homes * current_avg_age) /
  (current_avg_age - years_passed)

theorem new_homes_theorem :
  new_homes_added = 240 / 7 :=
sorry

end NUMINAMATH_CALUDE_new_homes_theorem_l548_54818


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_l548_54873

/-- For a parabola with equation y² = ax, if the distance from its focus to its directrix is 2, then a = ±4 -/
theorem parabola_focus_directrix (a : ℝ) : 
  (∃ (y x : ℝ), y^2 = a*x) →  -- parabola equation
  (∃ (p : ℝ), p = 2) →        -- distance from focus to directrix
  (a = 4 ∨ a = -4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_l548_54873


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l548_54890

/-- Given a school with 1200 students, where 875 play football, 450 play cricket, 
    and 100 neither play football nor cricket, prove that 225 students play both sports. -/
theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) :
  total = 1200 →
  football = 875 →
  cricket = 450 →
  neither = 100 →
  total - neither = football + cricket - 225 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l548_54890


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l548_54851

noncomputable def hypotenuse_length (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem right_triangle_hypotenuse (a b : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (vol1 : (1/3) * Real.pi * b^2 * a = 1250 * Real.pi)
  (vol2 : (1/3) * Real.pi * a^2 * b = 2700 * Real.pi) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (hypotenuse_length a b - 21.33) < ε :=
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l548_54851


namespace NUMINAMATH_CALUDE_scientific_notation_5690_l548_54854

theorem scientific_notation_5690 : 
  5690 = 5.69 * (10 : ℝ)^3 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_5690_l548_54854


namespace NUMINAMATH_CALUDE_sixth_score_achieves_target_mean_l548_54866

def test_scores : List ℝ := [78, 84, 76, 82, 88]
def sixth_score : ℝ := 102
def target_mean : ℝ := 85

theorem sixth_score_achieves_target_mean :
  (List.sum test_scores + sixth_score) / (test_scores.length + 1) = target_mean := by
  sorry

end NUMINAMATH_CALUDE_sixth_score_achieves_target_mean_l548_54866


namespace NUMINAMATH_CALUDE_constant_term_zero_l548_54857

theorem constant_term_zero (x : ℝ) (x_pos : x > 0) : 
  (∃ k : ℕ, k ≤ 10 ∧ (10 - k) / 2 - k = 0) → False :=
sorry

end NUMINAMATH_CALUDE_constant_term_zero_l548_54857


namespace NUMINAMATH_CALUDE_reflection_line_sum_l548_54846

/-- Given a line y = mx + b, if the reflection of point (2, 2) across this line is (8, 6), then m + b = 10 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∀ (x y : ℝ), (x - 2)^2 + (y - 2)^2 = (8 - x)^2 + (6 - y)^2 → y = m*x + b) →
  m + b = 10 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l548_54846


namespace NUMINAMATH_CALUDE_probability_one_red_ball_l548_54829

theorem probability_one_red_ball (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) 
  (h1 : total_balls = red_balls + yellow_balls)
  (h2 : red_balls = 3)
  (h3 : yellow_balls = 2)
  (h4 : total_balls ≥ 2) :
  (red_balls.choose 1 * yellow_balls.choose 1 : ℚ) / total_balls.choose 2 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_red_ball_l548_54829


namespace NUMINAMATH_CALUDE_number_problem_l548_54896

theorem number_problem : 
  ∃ (n : ℝ), n - (102 / 20.4) = 5095 ∧ n = 5100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l548_54896


namespace NUMINAMATH_CALUDE_common_point_tangent_line_l548_54877

theorem common_point_tangent_line (a : ℝ) (h_a : a > 0) :
  ∃ x : ℝ, x > 0 ∧ 
    a * Real.sqrt x = Real.log (Real.sqrt x) ∧
    (a / (2 * Real.sqrt x) = 1 / (2 * x)) →
  a = Real.exp (-1) := by
  sorry

end NUMINAMATH_CALUDE_common_point_tangent_line_l548_54877


namespace NUMINAMATH_CALUDE_range_of_a_complete_theorem_l548_54865

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | 0 < x ∧ x < a}
def Q : Set ℝ := {x | -5 < x ∧ x < 1}

-- State the theorem
theorem range_of_a (a : ℝ) (ha : 0 < a) (h_union : P a ∪ Q = Q) : a ≤ 1 := by
  sorry

-- The complete theorem combining all conditions
theorem complete_theorem :
  ∃ a : ℝ, 0 < a ∧ P a ∪ Q = Q ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_complete_theorem_l548_54865


namespace NUMINAMATH_CALUDE_special_sequence_2011_l548_54845

/-- A sequence satisfying the given conditions -/
def special_sequence (a : ℕ → ℤ) : Prop :=
  a 201 = 2 ∧ ∀ n : ℕ, n > 0 → a n + a (n + 1) = 0

/-- The 2011th term of the special sequence is 2 -/
theorem special_sequence_2011 (a : ℕ → ℤ) (h : special_sequence a) : a 2011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_2011_l548_54845


namespace NUMINAMATH_CALUDE_G_simplification_l548_54813

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((2 * x - x^2) / (1 + 2 * x + x^2))

theorem G_simplification (x : ℝ) (h : x ≠ -1/2) : G x = Real.log (1 + 4 * x) - Real.log (1 + 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_G_simplification_l548_54813


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_6_with_digit_sum_15_l548_54885

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_6_with_digit_sum_15 :
  ∀ n : ℕ, is_three_digit n → n % 6 = 0 → digit_sum n = 15 → n ≤ 690 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_6_with_digit_sum_15_l548_54885


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l548_54841

theorem cos_two_theta_value (θ : Real) 
  (h : Real.sin (θ / 2) + Real.cos (θ / 2) = 2 * Real.sqrt 2 / 3) : 
  Real.cos (2 * θ) = 79 / 81 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l548_54841


namespace NUMINAMATH_CALUDE_smallest_x_for_1680x_perfect_cube_l548_54899

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_x_for_1680x_perfect_cube : 
  (∀ x : ℕ, x > 0 ∧ x < 44100 → ¬(is_perfect_cube (1680 * x))) ∧
  (is_perfect_cube (1680 * 44100)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_1680x_perfect_cube_l548_54899


namespace NUMINAMATH_CALUDE_not_perfect_square_property_l548_54832

def S : Set ℕ := {2, 5, 13}

theorem not_perfect_square_property (d : ℕ) (h1 : d ∉ S) (h2 : d > 0) :
  ∃ a b : ℕ, a ∈ S ∪ {d} ∧ b ∈ S ∪ {d} ∧ a ≠ b ∧ ¬∃ k : ℕ, a * b - 1 = k^2 :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_property_l548_54832


namespace NUMINAMATH_CALUDE_range_of_m_l548_54803

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x + 1 > 0) → 0 ≤ m ∧ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l548_54803


namespace NUMINAMATH_CALUDE_last_three_average_l548_54870

theorem last_three_average (list : List ℝ) : 
  list.length = 6 →
  list.sum / 6 = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 3 = 65 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l548_54870


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_is_nine_l548_54819

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a * b > 0) (h2 : a + 4 * b = 1) :
  ∀ x y : ℝ, x * y > 0 ∧ x + 4 * y = 1 → 1 / x + 1 / y ≥ 1 / a + 1 / b :=
by sorry

theorem min_value_is_nine (a b : ℝ) (h1 : a * b > 0) (h2 : a + 4 * b = 1) :
  1 / a + 1 / b = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_is_nine_l548_54819


namespace NUMINAMATH_CALUDE_prob_calculations_l548_54860

/-- Represents a box containing balls of two colors -/
structure Box where
  white : ℕ
  red : ℕ

/-- Calculates the probability of drawing two red balls without replacement -/
def prob_two_red (b : Box) : ℚ :=
  (b.red * (b.red - 1)) / ((b.white + b.red) * (b.white + b.red - 1))

/-- Calculates the probability of drawing a red ball after transferring two balls -/
def prob_red_after_transfer (b1 b2 : Box) : ℚ :=
  let total_ways := (b1.white + b1.red) * (b1.white + b1.red - 1) / 2
  let p_two_red := (b1.red * (b1.red - 1) / 2) / total_ways
  let p_one_each := (b1.red * b1.white) / total_ways
  let p_two_white := (b1.white * (b1.white - 1) / 2) / total_ways
  
  let p_red_given_two_red := (b2.red + 2) / (b2.white + b2.red + 2)
  let p_red_given_one_each := (b2.red + 1) / (b2.white + b2.red + 2)
  let p_red_given_two_white := b2.red / (b2.white + b2.red + 2)
  
  p_two_red * p_red_given_two_red + p_one_each * p_red_given_one_each + p_two_white * p_red_given_two_white

theorem prob_calculations (b1 b2 : Box) 
  (h1 : b1.white = 2) (h2 : b1.red = 4) (h3 : b2.white = 5) (h4 : b2.red = 3) :
  prob_two_red b1 = 2/5 ∧ prob_red_after_transfer b1 b2 = 13/30 := by
  sorry

end NUMINAMATH_CALUDE_prob_calculations_l548_54860


namespace NUMINAMATH_CALUDE_set_C_characterization_l548_54824

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

def C : Set ℝ := {0, 1, 2}

theorem set_C_characterization :
  ∀ a : ℝ, (A ∪ B a = A) ↔ a ∈ C :=
sorry

end NUMINAMATH_CALUDE_set_C_characterization_l548_54824


namespace NUMINAMATH_CALUDE_paint_distribution_l548_54843

theorem paint_distribution (total_paint : ℝ) (num_colors : ℕ) (paint_per_color : ℝ) :
  total_paint = 15 →
  num_colors = 3 →
  paint_per_color * num_colors = total_paint →
  paint_per_color = 5 := by
  sorry

end NUMINAMATH_CALUDE_paint_distribution_l548_54843


namespace NUMINAMATH_CALUDE_sin_plus_2cos_period_l548_54878

open Real

/-- The function f(x) = sin x + 2cos x has a period of 2π. -/
theorem sin_plus_2cos_period : ∃ (k : ℝ), k > 0 ∧ ∀ x, sin x + 2 * cos x = sin (x + k) + 2 * cos (x + k) := by
  use 2 * π
  constructor
  · exact two_pi_pos
  · intro x
    sorry


end NUMINAMATH_CALUDE_sin_plus_2cos_period_l548_54878


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l548_54821

open Real

theorem function_inequality_implies_parameter_bound 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ) :
  (∀ x, x ∈ Set.Icc (1/2 : ℝ) 2 → f x = a / x + x * log x) →
  (∀ x, x ∈ Set.Icc (1/2 : ℝ) 2 → g x = x^3 - x^2 - 5) →
  (∀ x₁ x₂, x₁ ∈ Set.Icc (1/2 : ℝ) 2 → x₂ ∈ Set.Icc (1/2 : ℝ) 2 → f x₁ - g x₂ ≥ 2) →
  a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l548_54821


namespace NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_l548_54872

/-- Pascal's triangle function -/
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.choose n k

/-- Predicate for a number being in Pascal's triangle -/
def inPascalTriangle (m : ℕ) : Prop :=
  ∃ (n : ℕ) (k : ℕ), pascal n k = m

theorem smallest_four_digit_in_pascal : 
  (∀ m : ℕ, m < 1000 → ¬(inPascalTriangle m ∧ m ≥ 1000)) ∧ 
  inPascalTriangle 1000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_l548_54872


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l548_54820

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℕ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℕ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℕ) 
  (h1 : total_students = 15) 
  (h2 : avg_age_all = 15) 
  (h3 : group1_size = 5) 
  (h4 : avg_age_group1 = 14) 
  (h5 : group2_size = 9) 
  (h6 : avg_age_group2 = 16) :
  total_students * avg_age_all = 
    group1_size * avg_age_group1 + 
    group2_size * avg_age_group2 + 11 :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l548_54820


namespace NUMINAMATH_CALUDE_intersection_probability_is_four_sevenths_l548_54883

/-- A rectangular prism with dimensions 3, 4, and 5 units -/
structure RectangularPrism where
  length : ℕ := 3
  width : ℕ := 4
  height : ℕ := 5

/-- The probability that a plane determined by three randomly chosen distinct vertices
    intersects the interior of the prism -/
def intersection_probability (prism : RectangularPrism) : ℚ :=
  4/7

/-- Theorem stating that the probability of intersection is 4/7 -/
theorem intersection_probability_is_four_sevenths (prism : RectangularPrism) :
  intersection_probability prism = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_probability_is_four_sevenths_l548_54883


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l548_54893

theorem cubic_minus_linear_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l548_54893


namespace NUMINAMATH_CALUDE_maria_piggy_bank_theorem_l548_54859

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "dime" => 10
  | "quarter" => 25
  | "nickel" => 5
  | _ => 0

/-- Calculates the total value of coins in dollars -/
def total_value (dimes quarters nickels additional_quarters : ℕ) : ℚ :=
  (dimes * coin_value "dime" +
   (quarters + additional_quarters) * coin_value "quarter" +
   nickels * coin_value "nickel") / 100

theorem maria_piggy_bank_theorem (dimes quarters nickels additional_quarters : ℕ)
  (h1 : dimes = 4)
  (h2 : quarters = 4)
  (h3 : nickels = 7)
  (h4 : additional_quarters = 5) :
  total_value dimes quarters nickels additional_quarters = 3 :=
sorry

end NUMINAMATH_CALUDE_maria_piggy_bank_theorem_l548_54859


namespace NUMINAMATH_CALUDE_sarah_candy_duration_l548_54801

/-- The number of days Sarah's candy will last -/
def candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (daily_consumption : ℕ) : ℕ :=
  (neighbors_candy + sister_candy) / daily_consumption

/-- Proof that Sarah's candy will last 9 days -/
theorem sarah_candy_duration :
  candy_duration 66 15 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sarah_candy_duration_l548_54801


namespace NUMINAMATH_CALUDE_candy_bar_cost_l548_54844

/-- The cost of a candy bar given initial and final amounts --/
theorem candy_bar_cost (initial : ℕ) (final : ℕ) (h : initial = 4) (h' : final = 3) :
  initial - final = 1 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l548_54844


namespace NUMINAMATH_CALUDE_square_roots_problem_l548_54802

theorem square_roots_problem (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2*a + 6)^2 = x ∧ (3 - a)^2 = x) → a = -9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l548_54802


namespace NUMINAMATH_CALUDE_pyramid_volume_change_l548_54853

theorem pyramid_volume_change (s h : ℝ) : 
  s > 0 → h > 0 → (1/3 : ℝ) * s^2 * h = 60 → 
  (1/3 : ℝ) * (3*s)^2 * (2*h) = 1080 := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_change_l548_54853


namespace NUMINAMATH_CALUDE_largest_square_factor_of_10_factorial_l548_54835

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_square_factor_of_10_factorial :
  ∀ n : ℕ, n ≤ 10 → (factorial n)^2 ≤ factorial 10 →
  (factorial n)^2 ≤ (factorial 6)^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_factor_of_10_factorial_l548_54835


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l548_54887

/-- The ratio of the area to the square of the perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio : 
  let side_length : ℝ := 10
  let perimeter : ℝ := 3 * side_length
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  area / (perimeter ^ 2) = Real.sqrt 3 / 36 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l548_54887


namespace NUMINAMATH_CALUDE_decode_1236_is_rand_l548_54888

/-- Represents a coding scheme for words -/
structure CodeScheme where
  range_code : String
  random_code : String

/-- Decodes a given code based on the coding scheme -/
def decode (scheme : CodeScheme) (code : String) : String :=
  sorry

/-- The specific coding scheme used in the problem -/
def problem_scheme : CodeScheme :=
  { range_code := "12345", random_code := "123678" }

/-- The theorem stating that 1236 decodes to "rand" under the given scheme -/
theorem decode_1236_is_rand :
  decode problem_scheme "1236" = "rand" :=
sorry

end NUMINAMATH_CALUDE_decode_1236_is_rand_l548_54888


namespace NUMINAMATH_CALUDE_total_paintable_area_l548_54848

/-- Calculate the total paintable area for four bedrooms --/
theorem total_paintable_area (
  num_bedrooms : ℕ)
  (length width height : ℝ)
  (window_area : ℝ) :
  num_bedrooms = 4 →
  length = 14 →
  width = 11 →
  height = 9 →
  window_area = 70 →
  (num_bedrooms : ℝ) * ((2 * (length * height + width * height)) - window_area) = 1520 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_l548_54848


namespace NUMINAMATH_CALUDE_smallest_angle_is_three_l548_54810

/-- Represents a polygon divided into sectors with central angles forming an arithmetic sequence -/
structure PolygonSectors where
  num_sectors : ℕ
  angle_sum : ℕ
  is_arithmetic_sequence : Bool
  all_angles_integer : Bool

/-- The smallest possible sector angle for a polygon with given properties -/
def smallest_sector_angle (p : PolygonSectors) : ℕ :=
  sorry

/-- Theorem stating the smallest possible sector angle for a specific polygon configuration -/
theorem smallest_angle_is_three :
  ∀ (p : PolygonSectors),
    p.num_sectors = 16 ∧
    p.angle_sum = 360 ∧
    p.is_arithmetic_sequence = true ∧
    p.all_angles_integer = true →
    smallest_sector_angle p = 3 :=
  sorry

end NUMINAMATH_CALUDE_smallest_angle_is_three_l548_54810


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l548_54805

theorem complex_modulus_problem (z : ℂ) : z = (-1 + I) / (1 + I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l548_54805


namespace NUMINAMATH_CALUDE_sum_of_products_is_negative_one_l548_54825

-- Define the polynomial Q(x)
def Q (x : ℝ) : ℝ := x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

-- Define the theorem
theorem sum_of_products_is_negative_one 
  (d₁ d₂ d₃ d₄ e₁ e₂ e₃ e₄ : ℝ) 
  (h : ∀ x : ℝ, Q x = (x^2 + d₁*x + e₁) * (x^2 + d₂*x + e₂) * (x^2 + d₃*x + e₃) * (x^2 + d₄*x + e₄)) : 
  d₁*e₁ + d₂*e₂ + d₃*e₃ + d₄*e₄ = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_is_negative_one_l548_54825


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l548_54806

theorem unique_integer_satisfying_conditions (x : ℤ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : 0 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + 2 < 4) : 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l548_54806


namespace NUMINAMATH_CALUDE_frustum_volume_l548_54855

/-- The volume of a frustum of a right square pyramid inscribed in a sphere -/
theorem frustum_volume (R : ℝ) (β : ℝ) : 
  (R > 0) → (π/4 < β) → (β < π/2) →
  ∃ V : ℝ, V = (2/3) * R^3 * Real.sin (2*β) * (1 + Real.cos (2*β)^2 - Real.cos (2*β)) :=
by sorry

end NUMINAMATH_CALUDE_frustum_volume_l548_54855


namespace NUMINAMATH_CALUDE_average_payment_is_657_l548_54881

/-- Represents the payment structure for a debt over a year -/
structure DebtPayment where
  base : ℕ  -- Base payment amount
  increment1 : ℕ  -- Increment for second segment
  increment2 : ℕ  -- Increment for third segment
  increment3 : ℕ  -- Increment for fourth segment
  increment4 : ℕ  -- Increment for fifth segment

/-- Calculates the average payment given the debt payment structure -/
def averagePayment (dp : DebtPayment) : ℚ :=
  let total := 
    20 * dp.base + 
    30 * (dp.base + dp.increment1) + 
    40 * (dp.base + dp.increment1 + dp.increment2) + 
    50 * (dp.base + dp.increment1 + dp.increment2 + dp.increment3) + 
    60 * (dp.base + dp.increment1 + dp.increment2 + dp.increment3 + dp.increment4)
  total / 200

/-- Theorem stating that the average payment for the given structure is $657 -/
theorem average_payment_is_657 (dp : DebtPayment) 
    (h1 : dp.base = 450)
    (h2 : dp.increment1 = 80)
    (h3 : dp.increment2 = 65)
    (h4 : dp.increment3 = 105)
    (h5 : dp.increment4 = 95) : 
  averagePayment dp = 657 := by
  sorry

end NUMINAMATH_CALUDE_average_payment_is_657_l548_54881


namespace NUMINAMATH_CALUDE_tea_mixture_price_l548_54839

/-- Given two types of tea mixed in a 1:1 ratio, where one tea costs 62 rupees per kg
    and the mixture is worth 67 rupees per kg, prove that the price of the second tea
    is 72 rupees per kg. -/
theorem tea_mixture_price (price_tea1 price_mixture : ℚ) (ratio : ℚ × ℚ) :
  price_tea1 = 62 →
  price_mixture = 67 →
  ratio = (1, 1) →
  ∃ price_tea2 : ℚ, price_tea2 = 72 ∧
    (price_tea1 * ratio.1 + price_tea2 * ratio.2) / (ratio.1 + ratio.2) = price_mixture :=
by sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l548_54839


namespace NUMINAMATH_CALUDE_movie_original_length_l548_54897

/-- The original length of a movie, given the length of a cut scene and the final length -/
def original_length (cut_scene_length final_length : ℕ) : ℕ :=
  final_length + cut_scene_length

/-- Theorem: The original length of the movie is 60 minutes -/
theorem movie_original_length : original_length 6 54 = 60 := by
  sorry

end NUMINAMATH_CALUDE_movie_original_length_l548_54897


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l548_54804

-- Define the start point of the line segment
def start_point : ℝ × ℝ := (1, 3)

-- Define the end point of the line segment
def end_point (x : ℝ) : ℝ × ℝ := (x, 7)

-- Define the length of the line segment
def segment_length : ℝ := 15

-- Theorem statement
theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  Real.sqrt ((x - 1)^2 + (7 - 3)^2) = segment_length → 
  x = 1 - Real.sqrt 209 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l548_54804


namespace NUMINAMATH_CALUDE_negation_of_proposition_l548_54858

theorem negation_of_proposition (p : ℝ → Prop) :
  (∀ x : ℝ, x ≥ 2 → p x) ↔ ¬(∃ x : ℝ, x < 2 ∧ ¬(p x)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l548_54858


namespace NUMINAMATH_CALUDE_roots_condition_inequality_condition_max_value_condition_l548_54814

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a + 2

-- Part 1
theorem roots_condition (a : ℝ) :
  (∃ x y, x ≠ y ∧ x < 2 ∧ y < 2 ∧ f a x = 0 ∧ f a y = 0) → a < -1 := by sorry

-- Part 2
theorem inequality_condition (a : ℝ) :
  (∀ x, f a x ≥ -1 - a*x) → -2 ≤ a ∧ a ≤ 6 := by sorry

-- Part 3
theorem max_value_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x ≤ 4) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 4) → a = 2/3 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_roots_condition_inequality_condition_max_value_condition_l548_54814


namespace NUMINAMATH_CALUDE_rectangle_area_l548_54892

/-- A rectangle with perimeter 100 meters and length three times the width has an area of 468.75 square meters. -/
theorem rectangle_area (l w : ℝ) (h1 : 2 * l + 2 * w = 100) (h2 : l = 3 * w) : l * w = 468.75 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l548_54892


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l548_54834

theorem ratio_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (16 * x - 3 * y) = 5 / 7) : 
  x / y = 5 / 1 := by sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l548_54834


namespace NUMINAMATH_CALUDE_shirt_cost_l548_54850

/-- Proves that the cost of each shirt is $50 given the sales and commission information --/
theorem shirt_cost (commission_rate : ℝ) (suit_price : ℝ) (suit_count : ℕ)
  (shirt_count : ℕ) (loafer_price : ℝ) (loafer_count : ℕ) (total_commission : ℝ) :
  commission_rate = 0.15 →
  suit_price = 700 →
  suit_count = 2 →
  shirt_count = 6 →
  loafer_price = 150 →
  loafer_count = 2 →
  total_commission = 300 →
  ∃ (shirt_price : ℝ), 
    total_commission = commission_rate * (suit_price * suit_count + shirt_price * shirt_count + loafer_price * loafer_count) ∧
    shirt_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l548_54850


namespace NUMINAMATH_CALUDE_vector_dot_product_theorem_l548_54884

def vector_a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def vector_b (y : ℝ) : Fin 2 → ℝ := ![1, y]
def vector_c : Fin 2 → ℝ := ![3, -6]

def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

def perpendicular (u v : Fin 2 → ℝ) : Prop := dot_product u v = 0

def parallel (u v : Fin 2 → ℝ) : Prop := ∃ (k : ℝ), ∀ (i : Fin 2), u i = k * (v i)

theorem vector_dot_product_theorem (x y : ℝ) :
  perpendicular (vector_a x) vector_c →
  parallel (vector_b y) vector_c →
  dot_product (vector_a x + vector_b y) vector_c = 15 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_theorem_l548_54884
