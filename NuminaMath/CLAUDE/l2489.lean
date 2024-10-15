import Mathlib

namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2489_248935

def set_A : Set ℝ := {x | x - 2 > 0}
def set_B : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem union_of_A_and_B :
  set_A ∪ set_B = Set.Ici (1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2489_248935


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2489_248937

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2489_248937


namespace NUMINAMATH_CALUDE_train_speed_ratio_l2489_248911

theorem train_speed_ratio (v1 v2 : ℝ) (t1 t2 : ℝ) (h1 : t1 = 4) (h2 : t2 = 36) :
  v1 * t1 / (v2 * t2) = 1 / 9 → v1 = v2 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_ratio_l2489_248911


namespace NUMINAMATH_CALUDE_line_intersection_l2489_248976

def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + (a + 2) * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 = 0

def not_parallel (a : ℝ) : Prop :=
  ¬ (∃ k : ℝ, k ≠ 0 ∧ a = k * a ∧ (a + 2) = -k ∧ 1 = k * 2)

theorem line_intersection (a : ℝ) :
  not_parallel a → a = 0 ∨ a = -3 := by sorry

end NUMINAMATH_CALUDE_line_intersection_l2489_248976


namespace NUMINAMATH_CALUDE_rectangle_diagonal_intersection_l2489_248952

/-- Given a rectangle with opposite vertices at (2,-3) and (14,9),
    the point where the diagonals intersect has coordinates (8, 3) -/
theorem rectangle_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (14, 9)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 3) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_intersection_l2489_248952


namespace NUMINAMATH_CALUDE_bus_driver_overtime_limit_l2489_248905

/-- Represents the problem of determining overtime limit for a bus driver --/
theorem bus_driver_overtime_limit 
  (regular_rate : ℝ) 
  (overtime_rate : ℝ) 
  (total_compensation : ℝ) 
  (total_hours : ℝ) 
  (h1 : regular_rate = 16)
  (h2 : overtime_rate = regular_rate * 1.75)
  (h3 : total_compensation = 864)
  (h4 : total_hours = 48) :
  ∃ (limit : ℝ), 
    limit = 40 ∧ 
    total_compensation = limit * regular_rate + (total_hours - limit) * overtime_rate :=
by sorry

end NUMINAMATH_CALUDE_bus_driver_overtime_limit_l2489_248905


namespace NUMINAMATH_CALUDE_f_increasing_range_l2489_248938

/-- The function f(x) = 2x^2 + mx - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + m * x - 1

/-- The theorem stating the range of m for which f is increasing on (1, +∞) -/
theorem f_increasing_range (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > 1 ∧ x₁ ≠ x₂ → (f m x₁ - f m x₂) / (x₁ - x₂) > 0) →
  m ≥ -4 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_range_l2489_248938


namespace NUMINAMATH_CALUDE_min_days_correct_l2489_248995

/-- Represents the problem of scheduling warriors for duty --/
structure WarriorSchedule where
  total_warriors : ℕ
  min_duty : ℕ
  max_duty : ℕ
  min_days : ℕ

/-- The specific instance of the problem --/
def warrior_problem : WarriorSchedule :=
  { total_warriors := 33
  , min_duty := 9
  , max_duty := 10
  , min_days := 7 }

/-- Theorem stating that the minimum number of days is correct --/
theorem min_days_correct (w : WarriorSchedule) (h1 : w = warrior_problem) :
  ∃ (k l m : ℕ),
    k + l = w.min_days ∧
    w.min_duty * k + w.max_duty * l = w.total_warriors * m ∧
    (∀ (k' l' : ℕ), k' + l' < w.min_days →
      ¬∃ (m' : ℕ), w.min_duty * k' + w.max_duty * l' = w.total_warriors * m') :=
by sorry

end NUMINAMATH_CALUDE_min_days_correct_l2489_248995


namespace NUMINAMATH_CALUDE_binomial_10_3_l2489_248946

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by sorry

end NUMINAMATH_CALUDE_binomial_10_3_l2489_248946


namespace NUMINAMATH_CALUDE_rotation_180_maps_points_l2489_248964

def rotation_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem rotation_180_maps_points :
  let C : ℝ × ℝ := (3, -2)
  let D : ℝ × ℝ := (2, -5)
  let C' : ℝ × ℝ := (-3, 2)
  let D' : ℝ × ℝ := (-2, 5)
  rotation_180 C = C' ∧ rotation_180 D = D' :=
by sorry

end NUMINAMATH_CALUDE_rotation_180_maps_points_l2489_248964


namespace NUMINAMATH_CALUDE_missile_time_equation_l2489_248934

/-- Represents the speed of the missile in Mach -/
def missile_speed : ℝ := 26

/-- Represents the conversion factor from Mach to meters per second -/
def mach_to_mps : ℝ := 340

/-- Represents the distance to the target in kilometers -/
def target_distance : ℝ := 12000

/-- Represents the time taken to reach the target in minutes -/
def time_to_target : ℝ → ℝ := λ x => x

/-- Theorem stating the equation for the time taken by the missile to reach the target -/
theorem missile_time_equation :
  ∀ x : ℝ, (missile_speed * mach_to_mps * 60 * time_to_target x) / 1000 = target_distance * 1000 :=
by sorry

end NUMINAMATH_CALUDE_missile_time_equation_l2489_248934


namespace NUMINAMATH_CALUDE_equation_solutions_l2489_248991

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 = 9 ↔ x = 5 ∨ x = -1) ∧
  (∀ x : ℝ, 2*x^2 - 3*x - 1 = 0 ↔ x = (3 + Real.sqrt 17) / 4 ∨ x = (3 - Real.sqrt 17) / 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2489_248991


namespace NUMINAMATH_CALUDE_media_group_arrangement_count_l2489_248904

/-- Represents the number of domestic media groups -/
def domestic_groups : ℕ := 6

/-- Represents the number of foreign media groups -/
def foreign_groups : ℕ := 3

/-- Represents the total number of media groups to be selected -/
def selected_groups : ℕ := 4

/-- Calculates the number of ways to select and arrange media groups -/
def media_group_arrangements (d : ℕ) (f : ℕ) (s : ℕ) : ℕ :=
  -- Implementation details are omitted as per the instructions
  sorry

/-- Theorem stating that the number of valid arrangements is 684 -/
theorem media_group_arrangement_count :
  media_group_arrangements domestic_groups foreign_groups selected_groups = 684 := by
  sorry

end NUMINAMATH_CALUDE_media_group_arrangement_count_l2489_248904


namespace NUMINAMATH_CALUDE_min_distance_inverse_curves_l2489_248915

/-- The minimum distance between points on two inverse curves -/
theorem min_distance_inverse_curves :
  let f (x : ℝ) := (1/2) * Real.exp x
  let g (x : ℝ) := Real.log (2 * x)
  ∀ (x y : ℝ), x > 0 → y > 0 →
  let P := (x, f x)
  let Q := (y, g y)
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 * (1 - Real.log 2) ∧
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 →
    let P' := (x', f x')
    let Q' := (y', g y')
    Real.sqrt ((x' - y')^2 + (f x' - g y')^2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_inverse_curves_l2489_248915


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2489_248948

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 3 * x - y^2

-- State the theorem
theorem diamond_equation_solution :
  ∀ x : ℝ, diamond x 7 = 20 → x = 23 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2489_248948


namespace NUMINAMATH_CALUDE_probability_female_wears_glasses_l2489_248969

/-- Given a class with female and male students, some wearing glasses, prove the probability of a randomly selected female student wearing glasses. -/
theorem probability_female_wears_glasses 
  (total_female : ℕ) 
  (total_male : ℕ) 
  (female_no_glasses : ℕ) 
  (male_with_glasses : ℕ) 
  (h1 : total_female = 18) 
  (h2 : total_male = 20) 
  (h3 : female_no_glasses = 8) 
  (h4 : male_with_glasses = 11) : 
  (total_female - female_no_glasses : ℚ) / total_female = 5 / 9 := by
  sorry

#check probability_female_wears_glasses

end NUMINAMATH_CALUDE_probability_female_wears_glasses_l2489_248969


namespace NUMINAMATH_CALUDE_total_tickets_sold_l2489_248931

theorem total_tickets_sold (adult_price student_price total_amount student_count : ℕ) 
  (h1 : adult_price = 12)
  (h2 : student_price = 6)
  (h3 : total_amount = 16200)
  (h4 : student_count = 300) :
  ∃ (adult_count : ℕ), 
    adult_count * adult_price + student_count * student_price = total_amount ∧
    adult_count + student_count = 1500 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l2489_248931


namespace NUMINAMATH_CALUDE_book_pricing_deduction_percentage_l2489_248994

theorem book_pricing_deduction_percentage
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (list_price : ℝ)
  (h1 : cost_price = 47.50)
  (h2 : profit_percentage = 25)
  (h3 : list_price = 69.85) :
  let selling_price := cost_price * (1 + profit_percentage / 100)
  let deduction_percentage := (list_price - selling_price) / list_price * 100
  deduction_percentage = 15 := by
sorry

end NUMINAMATH_CALUDE_book_pricing_deduction_percentage_l2489_248994


namespace NUMINAMATH_CALUDE_existence_of_symmetry_axes_l2489_248922

/-- A bounded planar figure. -/
structure BoundedPlanarFigure where
  -- Define properties of a bounded planar figure
  is_bounded : Bool
  is_planar : Bool

/-- An axis of symmetry for a bounded planar figure. -/
structure AxisOfSymmetry (F : BoundedPlanarFigure) where
  -- Define properties of an axis of symmetry

/-- The number of axes of symmetry for a bounded planar figure. -/
def num_axes_of_symmetry (F : BoundedPlanarFigure) : Nat :=
  sorry

/-- Theorem: There exist bounded planar figures with exactly three axes of symmetry,
    and there exist bounded planar figures with more than three axes of symmetry. -/
theorem existence_of_symmetry_axes :
  (∃ F : BoundedPlanarFigure, F.is_bounded ∧ F.is_planar ∧ num_axes_of_symmetry F = 3) ∧
  (∃ G : BoundedPlanarFigure, G.is_bounded ∧ G.is_planar ∧ num_axes_of_symmetry G > 3) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_symmetry_axes_l2489_248922


namespace NUMINAMATH_CALUDE_sticker_distribution_theorem_l2489_248961

/-- Number of stickers -/
def n : ℕ := 10

/-- Number of sheets -/
def k : ℕ := 5

/-- Number of color options for each sheet -/
def c : ℕ := 2

/-- The number of ways to distribute n identical objects into k distinct boxes -/
def ways_to_distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- The total number of distinct arrangements considering both sticker counts and colors -/
def total_arrangements (n k c : ℕ) : ℕ := (ways_to_distribute n k) * (c^k)

/-- The main theorem stating the total number of distinct arrangements -/
theorem sticker_distribution_theorem : total_arrangements n k c = 32032 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_theorem_l2489_248961


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l2489_248950

/-- Given a quadratic equation ax^2 + bx + c = 0 where a ≠ 0 and c ≠ 0,
    if one root is 4 times the other root, then b^2 / (ac) = 25/4 -/
theorem quadratic_root_ratio (a b c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  (∃ x y : ℝ, x = 4 * y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  b^2 / (a * c) = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l2489_248950


namespace NUMINAMATH_CALUDE_car_speed_l2489_248970

/-- Given a car that travels 275 miles in 5 hours, its speed is 55 miles per hour. -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 275) 
  (h2 : time = 5) 
  (h3 : speed = distance / time) : 
  speed = 55 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l2489_248970


namespace NUMINAMATH_CALUDE_smallest_square_addition_l2489_248985

theorem smallest_square_addition (n : ℕ) (h : n = 2019) : 
  ∃ m : ℕ, (n - 1) * n * (n + 1) * (n + 2) + 1 = m^2 ∧ 
  ∀ k : ℕ, k < 1 → ¬∃ m : ℕ, (n - 1) * n * (n + 1) * (n + 2) + k = m^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_addition_l2489_248985


namespace NUMINAMATH_CALUDE_range_of_a_l2489_248917

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2489_248917


namespace NUMINAMATH_CALUDE_largest_value_l2489_248962

theorem largest_value : 
  let a := 3 + 1 + 2 + 8
  let b := 3 * 1 + 2 + 8
  let c := 3 + 1 * 2 + 8
  let d := 3 + 1 + 2 * 8
  let e := 3 * 1 * 2 * 8
  (e ≥ a) ∧ (e ≥ b) ∧ (e ≥ c) ∧ (e ≥ d) :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l2489_248962


namespace NUMINAMATH_CALUDE_a2_times_a6_eq_68_l2489_248989

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℤ := 4 * n^2 - 10 * n

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℤ := S n - S (n-1)

/-- Theorem stating that a_2 * a_6 = 68 -/
theorem a2_times_a6_eq_68 : a 2 * a 6 = 68 := by
  sorry

end NUMINAMATH_CALUDE_a2_times_a6_eq_68_l2489_248989


namespace NUMINAMATH_CALUDE_existence_of_special_number_l2489_248949

theorem existence_of_special_number : ∃ A : ℕ, 
  (100000 ≤ A ∧ A < 1000000) ∧ 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 500000 → 
    ¬(∃ d : ℕ, d < 10 ∧ (k * A) % 1000000 = d * 111111) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l2489_248949


namespace NUMINAMATH_CALUDE_sector_central_angle_l2489_248951

/-- Given a circular sector with radius 2 and area 8, 
    the radian measure of its central angle is 4. -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (angle : ℝ) : 
  r = 2 → area = 8 → area = (1/2) * r^2 * angle → angle = 4 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2489_248951


namespace NUMINAMATH_CALUDE_max_xyz_value_l2489_248966

theorem max_xyz_value (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum : x + y + z = 2) (h_sq_sum : x^2 + y^2 + z^2 = x*z + y*z + x*y) :
  x*y*z ≤ 8/27 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b + c = 2 ∧ a^2 + b^2 + c^2 = a*c + b*c + a*b ∧ a*b*c = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_max_xyz_value_l2489_248966


namespace NUMINAMATH_CALUDE_geometric_locus_definition_l2489_248975

-- Define a type for points in a space
variable {Point : Type*}

-- Define a predicate for the condition that points must satisfy
variable (condition : Point → Prop)

-- Define a predicate for points being on the locus
variable (on_locus : Point → Prop)

-- Statement A
def statement_A : Prop :=
  (∀ p, on_locus p → condition p) ∧ (∀ p, condition p → on_locus p)

-- Statement B
def statement_B : Prop :=
  (∀ p, ¬condition p → ¬on_locus p) ∧ ¬(∀ p, condition p → on_locus p)

-- Statement C
def statement_C : Prop :=
  ∀ p, on_locus p ↔ condition p

-- Statement D
def statement_D : Prop :=
  (∀ p, ¬on_locus p → ¬condition p) ∧ (∀ p, condition p → on_locus p)

-- Statement E
def statement_E : Prop :=
  (∀ p, on_locus p → condition p) ∧ ¬(∀ p, condition p → on_locus p)

theorem geometric_locus_definition :
  (statement_A condition on_locus ∧ 
   statement_C condition on_locus ∧ 
   statement_D condition on_locus) ∧
  (¬statement_B condition on_locus ∧ 
   ¬statement_E condition on_locus) :=
sorry

end NUMINAMATH_CALUDE_geometric_locus_definition_l2489_248975


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2489_248984

/-- The standard equation of a hyperbola passing through specific points and sharing asymptotes with another hyperbola -/
theorem hyperbola_equation : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ (x y : ℝ), y^2 / a - x^2 / b = 1 ↔ 
    ((x = -3 ∧ y = 2 * Real.sqrt 7) ∨ 
     (x = -6 * Real.sqrt 2 ∧ y = -7) ∨ 
     (x = 2 ∧ y = 2 * Real.sqrt 3)) ∧
    (∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), y^2 / a - x^2 / b = 1 ↔ x^2 / 4 - y^2 / 3 = k)) ∧
  a = 9 ∧ b = 12 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2489_248984


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2489_248924

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is 2√(a² + b²) -/
theorem hyperbola_focal_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let focal_length := 2 * Real.sqrt (a^2 + b^2)
  focal_length = 2 * Real.sqrt 7 ↔ a^2 = 4 ∧ b^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2489_248924


namespace NUMINAMATH_CALUDE_enthalpy_combustion_10_moles_glucose_l2489_248965

/-- The standard enthalpy of combustion for glucose (C6H12O6) in kJ/mol -/
def standard_enthalpy_combustion_glucose : ℝ := -2800

/-- The number of moles of glucose -/
def moles_glucose : ℝ := 10

/-- The enthalpy of combustion for a given number of moles of glucose -/
def enthalpy_combustion (moles : ℝ) : ℝ :=
  standard_enthalpy_combustion_glucose * moles

/-- Theorem: The enthalpy of combustion for 10 moles of C6H12O6 is -28000 kJ -/
theorem enthalpy_combustion_10_moles_glucose :
  enthalpy_combustion moles_glucose = -28000 := by
  sorry

end NUMINAMATH_CALUDE_enthalpy_combustion_10_moles_glucose_l2489_248965


namespace NUMINAMATH_CALUDE_five_sixteenths_decimal_l2489_248903

theorem five_sixteenths_decimal : (5 : ℚ) / 16 = (3125 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_five_sixteenths_decimal_l2489_248903


namespace NUMINAMATH_CALUDE_initial_female_percent_calculation_l2489_248971

/-- Represents a company's workforce statistics -/
structure Workforce where
  initial_total : ℕ
  initial_female_percent : ℚ
  hired_male : ℕ
  final_total : ℕ
  final_female_percent : ℚ

/-- Theorem stating the conditions and the result to be proved -/
theorem initial_female_percent_calculation (w : Workforce) 
  (h1 : w.hired_male = 30)
  (h2 : w.final_total = 360)
  (h3 : w.final_female_percent = 55/100)
  (h4 : w.initial_total * w.initial_female_percent = w.final_total * w.final_female_percent) :
  w.initial_female_percent = 60/100 := by
  sorry

end NUMINAMATH_CALUDE_initial_female_percent_calculation_l2489_248971


namespace NUMINAMATH_CALUDE_cos_225_degrees_l2489_248960

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l2489_248960


namespace NUMINAMATH_CALUDE_square_sum_ge_product_sum_l2489_248928

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_product_sum_l2489_248928


namespace NUMINAMATH_CALUDE_sum_a_b_equals_seven_l2489_248987

/-- Represents a four-digit number in the form 3a72 -/
def fourDigitNum (a : ℕ) : ℕ := 3000 + 100 * a + 72

/-- Checks if a number is divisible by 11 -/
def divisibleBy11 (n : ℕ) : Prop := n % 11 = 0

theorem sum_a_b_equals_seven :
  ∀ a b : ℕ,
  (a < 10) →
  (b < 10) →
  (fourDigitNum a + 895 = 4000 + 100 * b + 67) →
  divisibleBy11 (4000 + 100 * b + 67) →
  a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_seven_l2489_248987


namespace NUMINAMATH_CALUDE_inkblot_area_bound_l2489_248979

/-- Represents an inkblot on a square sheet of paper -/
structure Inkblot where
  area : ℝ
  x_extent : ℝ
  y_extent : ℝ

/-- The theorem stating that the total area of inkblots does not exceed the side length of the square paper -/
theorem inkblot_area_bound (a : ℝ) (inkblots : List Inkblot) : a > 0 →
  (∀ i ∈ inkblots, i.area ≤ 1) →
  (∀ i ∈ inkblots, i.x_extent ≤ a ∧ i.y_extent ≤ a) →
  (∀ x : ℝ, x ≥ 0 ∧ x ≤ a → (inkblots.filter (fun i => i.x_extent > x)).length ≤ 1) →
  (∀ y : ℝ, y ≥ 0 ∧ y ≤ a → (inkblots.filter (fun i => i.y_extent > y)).length ≤ 1) →
  (inkblots.map (fun i => i.area)).sum ≤ a :=
by sorry

end NUMINAMATH_CALUDE_inkblot_area_bound_l2489_248979


namespace NUMINAMATH_CALUDE_fraction_simplification_l2489_248940

theorem fraction_simplification (x y z : ℝ) 
  (h1 : x > z) (h2 : z > y) (h3 : y > 0) :
  (x^z * z^y * y^x) / (z^z * y^y * x^x) = x^(z-x) * z^(y-z) * y^(x-y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2489_248940


namespace NUMINAMATH_CALUDE_sum_of_rationals_l2489_248919

theorem sum_of_rationals (a b : ℚ) (h : a + Real.sqrt 3 * b = Real.sqrt (4 + 2 * Real.sqrt 3)) : 
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rationals_l2489_248919


namespace NUMINAMATH_CALUDE_smallest_x_y_sum_l2489_248993

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_x_y_sum (x y : ℕ) : 
  (x > 0 ∧ y > 0) →
  (is_square (450 * x)) →
  (is_cube (450 * y)) →
  (∀ x' : ℕ, x' > 0 → x' < x → ¬(is_square (450 * x'))) →
  (∀ y' : ℕ, y' > 0 → y' < y → ¬(is_cube (450 * y'))) →
  x = 2 ∧ y = 60 ∧ x + y = 62 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_y_sum_l2489_248993


namespace NUMINAMATH_CALUDE_fabric_order_calculation_l2489_248967

/-- The conversion factor from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- David's waist size in inches -/
def waist_size : ℝ := 38

/-- The extra allowance for waistband sewing in centimeters -/
def waistband_allowance : ℝ := 2

/-- The total length of fabric David should order in centimeters -/
def total_fabric_length : ℝ := waist_size * inch_to_cm + waistband_allowance

theorem fabric_order_calculation :
  total_fabric_length = 98.52 :=
by sorry

end NUMINAMATH_CALUDE_fabric_order_calculation_l2489_248967


namespace NUMINAMATH_CALUDE_vector_operation_result_l2489_248913

theorem vector_operation_result :
  let v1 : Fin 2 → ℝ := ![5, -3]
  let v2 : Fin 2 → ℝ := ![0, 4]
  let v3 : Fin 2 → ℝ := ![-2, 1]
  let result : Fin 2 → ℝ := ![3, -14]
  v1 - 3 • v2 + v3 = result :=
by
  sorry

end NUMINAMATH_CALUDE_vector_operation_result_l2489_248913


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l2489_248909

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 15.625

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 25

theorem bowling_ball_weight_proof :
  (8 * bowling_ball_weight = 5 * canoe_weight) ∧
  (4 * canoe_weight = 100) →
  bowling_ball_weight = 15.625 := by sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l2489_248909


namespace NUMINAMATH_CALUDE_find_number_l2489_248942

theorem find_number (x : ℚ) : (x + 113 / 78) * 78 = 4403 → x = 55 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2489_248942


namespace NUMINAMATH_CALUDE_temperature_conversion_l2489_248980

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → k = 95 → t = 35 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2489_248980


namespace NUMINAMATH_CALUDE_distribute_balls_count_l2489_248945

/-- The number of ways to distribute 6 balls into 3 boxes -/
def distribute_balls : ℕ :=
  3 * (Nat.choose 4 2)

/-- Theorem stating that the number of ways to distribute the balls is 18 -/
theorem distribute_balls_count : distribute_balls = 18 := by
  sorry

end NUMINAMATH_CALUDE_distribute_balls_count_l2489_248945


namespace NUMINAMATH_CALUDE_two_equal_real_roots_l2489_248977

def quadratic_equation (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem two_equal_real_roots (a b c : ℝ) (ha : a ≠ 0) :
  a = 4 ∧ b = -4 ∧ c = 1 →
  ∃ x : ℝ, quadratic_equation a b c x ∧
    ∀ y : ℝ, quadratic_equation a b c y → y = x :=
by
  sorry

end NUMINAMATH_CALUDE_two_equal_real_roots_l2489_248977


namespace NUMINAMATH_CALUDE_josh_final_wallet_amount_l2489_248998

-- Define the initial conditions
def initial_wallet_amount : ℝ := 300
def initial_investment : ℝ := 2000
def stock_price_increase : ℝ := 0.30

-- Define the function to calculate the final amount
def final_wallet_amount : ℝ :=
  initial_wallet_amount + initial_investment * (1 + stock_price_increase)

-- Theorem to prove
theorem josh_final_wallet_amount :
  final_wallet_amount = 2900 := by
  sorry

end NUMINAMATH_CALUDE_josh_final_wallet_amount_l2489_248998


namespace NUMINAMATH_CALUDE_sara_movie_spending_l2489_248944

def movie_spending (theater_ticket_price : ℚ) (num_tickets : ℕ) (rental_price : ℚ) (purchase_price : ℚ) : ℚ :=
  theater_ticket_price * num_tickets + rental_price + purchase_price

theorem sara_movie_spending :
  let theater_ticket_price : ℚ := 10.62
  let num_tickets : ℕ := 2
  let rental_price : ℚ := 1.59
  let purchase_price : ℚ := 13.95
  movie_spending theater_ticket_price num_tickets rental_price purchase_price = 36.78 := by
sorry

end NUMINAMATH_CALUDE_sara_movie_spending_l2489_248944


namespace NUMINAMATH_CALUDE_garden_size_l2489_248999

theorem garden_size (garden_size fruit_size vegetable_size strawberry_size : ℝ) : 
  fruit_size = vegetable_size →
  garden_size = fruit_size + vegetable_size →
  strawberry_size = fruit_size / 4 →
  strawberry_size = 8 →
  garden_size = 64 := by
sorry

end NUMINAMATH_CALUDE_garden_size_l2489_248999


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l2489_248974

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the property of a line intersecting another line
variable (intersects : Line → Line → Prop)

theorem line_intersection_theorem 
  (a b m : Line) (α β : Plane)
  (h1 : skew a b)
  (h2 : contained_in a α)
  (h3 : contained_in b β)
  (h4 : intersect α β = m) :
  intersects m a ∨ intersects m b :=
sorry

end NUMINAMATH_CALUDE_line_intersection_theorem_l2489_248974


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2489_248933

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 10 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2489_248933


namespace NUMINAMATH_CALUDE_tetrahedron_side_length_l2489_248927

/-- The side length of a regular tetrahedron given its square shadow area -/
theorem tetrahedron_side_length (shadow_area : ℝ) (h : shadow_area = 16) :
  ∃ (side_length : ℝ), side_length = 4 * Real.sqrt 2 ∧
  side_length * side_length = 2 * shadow_area :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_side_length_l2489_248927


namespace NUMINAMATH_CALUDE_digit_150_of_11_13_l2489_248932

/-- The decimal representation of 11/13 has a repeating sequence of 6 digits. -/
def decimal_rep_11_13 : List Nat := [8, 4, 6, 1, 5, 3]

/-- The 150th digit after the decimal point in the decimal representation of 11/13 is 3. -/
theorem digit_150_of_11_13 : 
  decimal_rep_11_13[150 % decimal_rep_11_13.length] = 3 := by sorry

end NUMINAMATH_CALUDE_digit_150_of_11_13_l2489_248932


namespace NUMINAMATH_CALUDE_initial_potatoes_count_l2489_248912

/-- The number of potatoes Dan initially had in the garden --/
def initial_potatoes : ℕ := sorry

/-- The number of potatoes eaten by rabbits --/
def eaten_potatoes : ℕ := 4

/-- The number of potatoes Dan has now --/
def remaining_potatoes : ℕ := 3

/-- Theorem stating the initial number of potatoes --/
theorem initial_potatoes_count : initial_potatoes = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_potatoes_count_l2489_248912


namespace NUMINAMATH_CALUDE_base_10_1234_equals_base_7_3412_l2489_248902

def base_10_to_base_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec to_digits (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else to_digits (m / 7) ((m % 7) :: acc)
  to_digits n []

theorem base_10_1234_equals_base_7_3412 :
  base_10_to_base_7 1234 = [3, 4, 1, 2] := by sorry

end NUMINAMATH_CALUDE_base_10_1234_equals_base_7_3412_l2489_248902


namespace NUMINAMATH_CALUDE_max_third_side_length_l2489_248914

theorem max_third_side_length (a b : ℝ) (ha : a = 6) (hb : b = 10) :
  ∃ (s : ℕ), s ≤ 15 ∧ 
  (∀ (t : ℕ), (t : ℝ) < a + b ∧ a < (t : ℝ) + b ∧ b < a + (t : ℝ) → t ≤ s) ∧
  ((15 : ℝ) < a + b ∧ a < 15 + b ∧ b < a + 15) :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_length_l2489_248914


namespace NUMINAMATH_CALUDE_chord_of_contact_ellipse_l2489_248929

/-- Given an ellipse and a point outside it, the chord of contact has a specific equation. -/
theorem chord_of_contact_ellipse (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_outside : (x₀^2 / a^2) + (y₀^2 / b^2) > 1) :
  ∃ (A B : ℝ × ℝ), 
    (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧ 
    (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
    (∀ (x y : ℝ), ((x₀ * x) / a^2 + (y₀ * y) / b^2 = 1) ↔ 
      ∃ (t : ℝ), x = A.1 + t * (B.1 - A.1) ∧ y = A.2 + t * (B.2 - A.2)) := by
  sorry

end NUMINAMATH_CALUDE_chord_of_contact_ellipse_l2489_248929


namespace NUMINAMATH_CALUDE_divisibility_rule_2701_l2489_248916

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  tens * tens + ones * ones

theorem divisibility_rule_2701 :
  ∀ x : ℕ, is_two_digit x →
    (2701 % x = 0 ↔ sum_of_squares_of_digits x = 58) := by sorry

end NUMINAMATH_CALUDE_divisibility_rule_2701_l2489_248916


namespace NUMINAMATH_CALUDE_doctors_lawyers_ratio_l2489_248963

theorem doctors_lawyers_ratio (d l : ℕ) (h_group_avg : (40 * d + 55 * l) / (d + l) = 45) : d = 2 * l := by
  sorry

end NUMINAMATH_CALUDE_doctors_lawyers_ratio_l2489_248963


namespace NUMINAMATH_CALUDE_fruit_basket_total_cost_l2489_248957

def fruit_basket_cost (banana_count : ℕ) (apple_count : ℕ) (strawberry_count : ℕ) (avocado_count : ℕ) (grape_bunch_count : ℕ) : ℕ :=
  let banana_price := 1
  let apple_price := 2
  let strawberry_price_per_12 := 4
  let avocado_price := 3
  let grape_half_bunch_price := 2

  banana_count * banana_price +
  apple_count * apple_price +
  (strawberry_count / 12) * strawberry_price_per_12 +
  avocado_count * avocado_price +
  grape_bunch_count * (2 * grape_half_bunch_price)

theorem fruit_basket_total_cost :
  fruit_basket_cost 4 3 24 2 1 = 28 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_total_cost_l2489_248957


namespace NUMINAMATH_CALUDE_line_slope_intercept_l2489_248986

/-- The line equation in vector form -/
def line_equation (x y : ℝ) : Prop :=
  (2 : ℝ) * (x - 1) + (-1 : ℝ) * (y - 5) = 0

/-- The slope-intercept form of a line -/
def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem line_slope_intercept :
  ∃ (m b : ℝ), (∀ x y : ℝ, line_equation x y ↔ slope_intercept_form m b x y) ∧ m = 2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_l2489_248986


namespace NUMINAMATH_CALUDE_measure_nine_kg_from_twentyfour_l2489_248921

/-- Represents a pile of nails with a given weight in kg. -/
structure NailPile :=
  (weight : ℚ)

/-- Represents the state of our nails, divided into at most four piles. -/
structure NailState :=
  (pile1 : NailPile)
  (pile2 : Option NailPile)
  (pile3 : Option NailPile)
  (pile4 : Option NailPile)

/-- Divides a pile into two equal piles. -/
def dividePile (p : NailPile) : NailPile × NailPile :=
  (⟨p.weight / 2⟩, ⟨p.weight / 2⟩)

/-- Combines two piles into one. -/
def combinePiles (p1 p2 : NailPile) : NailPile :=
  ⟨p1.weight + p2.weight⟩

/-- The theorem stating that we can measure out 9 kg from 24 kg using only division. -/
theorem measure_nine_kg_from_twentyfour :
  ∃ (final : NailState),
    (final.pile1.weight = 9 ∨ 
     (∃ p, final.pile2 = some p ∧ p.weight = 9) ∨
     (∃ p, final.pile3 = some p ∧ p.weight = 9) ∨
     (∃ p, final.pile4 = some p ∧ p.weight = 9)) ∧
    final.pile1.weight + 
    (final.pile2.map (λ p => p.weight) |>.getD 0) +
    (final.pile3.map (λ p => p.weight) |>.getD 0) +
    (final.pile4.map (λ p => p.weight) |>.getD 0) = 24 :=
sorry

end NUMINAMATH_CALUDE_measure_nine_kg_from_twentyfour_l2489_248921


namespace NUMINAMATH_CALUDE_laptop_sticker_price_l2489_248941

/-- The sticker price of a laptop satisfying certain discount conditions -/
theorem laptop_sticker_price : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (0.7 * x - (0.8 * x - 120) = 30) ∧ 
  (x = 900) := by
  sorry

end NUMINAMATH_CALUDE_laptop_sticker_price_l2489_248941


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l2489_248997

def coin_flips : ℕ := 12
def heads_count : ℕ := 9

-- Define the probability of getting exactly k heads in n flips of a fair coin
def probability_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

theorem probability_nine_heads_in_twelve_flips :
  probability_k_heads coin_flips heads_count = 55 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l2489_248997


namespace NUMINAMATH_CALUDE_tank_capacity_l2489_248955

theorem tank_capacity (initial_buckets : ℕ) (initial_capacity : ℚ) (new_buckets : ℕ) :
  initial_buckets = 26 →
  initial_capacity = 13.5 →
  new_buckets = 39 →
  (initial_buckets : ℚ) * initial_capacity / (new_buckets : ℚ) = 9 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l2489_248955


namespace NUMINAMATH_CALUDE_trig_identity_l2489_248982

theorem trig_identity (x y : ℝ) : 
  Real.sin (x + y) * Real.sin (x - y) - Real.cos (x + y) * Real.cos (x - y) = -Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2489_248982


namespace NUMINAMATH_CALUDE_half_plus_five_equals_eleven_l2489_248978

theorem half_plus_five_equals_eleven (n : ℝ) : (1/2) * n + 5 = 11 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_eleven_l2489_248978


namespace NUMINAMATH_CALUDE_f_properties_l2489_248900

def f (x : ℝ) : ℝ := x^3 - 3*x^2

theorem f_properties : 
  (∀ x y, x < y ∧ ((x ≤ 0 ∧ y ≤ 0) ∨ (x ≥ 2 ∧ y ≥ 2)) → f x < f y) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x > f y) ∧
  (∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → f x < f 0) ∧
  (∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → f x > f 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2489_248900


namespace NUMINAMATH_CALUDE_expression_value_l2489_248918

theorem expression_value (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 1) :
  (2*a + 3*b - 2*a*b) - (a + 4*b + a*b) - (3*a*b + 2*b - 2*a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2489_248918


namespace NUMINAMATH_CALUDE_m_range_l2489_248926

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0

def q (m : ℝ) : Prop := ∀ x : ℝ, m*x^2 - x + (1/16)*m > 0

-- Define the set of m satisfying the conditions
def S : Set ℝ := {m | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- Theorem statement
theorem m_range : S = {m | 1 < m ∧ m ≤ 2} := by sorry

end NUMINAMATH_CALUDE_m_range_l2489_248926


namespace NUMINAMATH_CALUDE_square_root_computation_l2489_248968

theorem square_root_computation : (3 * Real.sqrt 15625 - 5)^2 = 136900 := by
  sorry

end NUMINAMATH_CALUDE_square_root_computation_l2489_248968


namespace NUMINAMATH_CALUDE_claire_earnings_l2489_248988

def total_flowers : ℕ := 400
def tulips : ℕ := 120
def white_roses : ℕ := 80
def price_per_red_rose : ℚ := 3/4

def total_roses : ℕ := total_flowers - tulips
def red_roses : ℕ := total_roses - white_roses
def red_roses_to_sell : ℕ := red_roses / 2

theorem claire_earnings :
  (red_roses_to_sell : ℚ) * price_per_red_rose = 75 := by
  sorry

end NUMINAMATH_CALUDE_claire_earnings_l2489_248988


namespace NUMINAMATH_CALUDE_square_difference_ben_subtraction_l2489_248901

theorem square_difference (n : ℕ) : (n - 1)^2 = n^2 - (2*n - 1) := by sorry

theorem ben_subtraction : 49^2 = 50^2 - 99 := by sorry

end NUMINAMATH_CALUDE_square_difference_ben_subtraction_l2489_248901


namespace NUMINAMATH_CALUDE_additional_space_needed_l2489_248956

theorem additional_space_needed (available_space backup_size software_size : ℕ) : 
  available_space = 28 → 
  backup_size = 26 → 
  software_size = 4 → 
  (backup_size + software_size) - available_space = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_additional_space_needed_l2489_248956


namespace NUMINAMATH_CALUDE_square_ratio_side_length_l2489_248958

theorem square_ratio_side_length (area_ratio : ℚ) : 
  area_ratio = 250 / 98 →
  ∃ (a b c : ℕ), 
    (a^2 * b : ℚ) / c^2 = area_ratio ∧
    a = 5 ∧ b = 5 ∧ c = 7 ∧
    a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_l2489_248958


namespace NUMINAMATH_CALUDE_set_union_problem_l2489_248992

theorem set_union_problem (M N : Set ℕ) (x : ℕ) : 
  M = {0, x} → N = {1, 2} → M ∩ N = {2} → M ∪ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l2489_248992


namespace NUMINAMATH_CALUDE_division_remainder_division_remainder_is_200000_l2489_248947

theorem division_remainder : ℤ → Prop :=
  fun r => ((8 * 10^9) / (4 * 10^4)) % (10^6) = r

theorem division_remainder_is_200000 : division_remainder 200000 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_division_remainder_is_200000_l2489_248947


namespace NUMINAMATH_CALUDE_amusement_park_capacity_l2489_248943

/-- Represents the capacity of an amusement park ride -/
structure RideCapacity where
  people_per_unit : ℕ
  units : ℕ

/-- Calculates the total capacity of a ride -/
def total_capacity (ride : RideCapacity) : ℕ :=
  ride.people_per_unit * ride.units

/-- Theorem: The total capacity of three specific rides is 248 people -/
theorem amusement_park_capacity (whirling_wonderland sky_high_swings roaring_rapids : RideCapacity)
  (h1 : whirling_wonderland = ⟨12, 15⟩)
  (h2 : sky_high_swings = ⟨1, 20⟩)
  (h3 : roaring_rapids = ⟨6, 8⟩) :
  total_capacity whirling_wonderland + total_capacity sky_high_swings + total_capacity roaring_rapids = 248 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_capacity_l2489_248943


namespace NUMINAMATH_CALUDE_inequality_proof_l2489_248972

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a^2 - a*b + b^2) ≥ (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2489_248972


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2489_248954

/-- A three-digit number satisfying specific conditions -/
def three_digit_number (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  a + b + c = 10 ∧
  b = a + c ∧
  100 * c + 10 * b + a = 100 * a + 10 * b + c + 99

theorem unique_three_digit_number :
  ∃! (a b c : ℕ), three_digit_number a b c ∧ 100 * a + 10 * b + c = 253 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2489_248954


namespace NUMINAMATH_CALUDE_checkerboard_covering_l2489_248936

/-- Represents an L-shaped piece that can cover three squares on a checkerboard -/
inductive LPiece
| mk : LPiece

/-- Represents a square on the checkerboard -/
structure Square where
  x : Nat
  y : Nat

/-- Represents a checkerboard with one square removed -/
structure Checkerboard (n : Nat) where
  sideLength : Nat
  removedSquare : Square
  validSideLength : sideLength = 2^n
  validRemovedSquare : removedSquare.x < sideLength ∧ removedSquare.y < sideLength

/-- Represents a covering of the checkerboard with L-shaped pieces -/
def Covering (n : Nat) := List (Square × Square × Square)

/-- Checks if a covering is valid for a given checkerboard -/
def isValidCovering (n : Nat) (board : Checkerboard n) (covering : Covering n) : Prop :=
  -- Each L-piece covers exactly three squares
  -- No gaps or overlaps in the covering
  -- The removed square is not covered
  sorry

/-- Main theorem: Any checkerboard with one square removed can be covered by L-shaped pieces -/
theorem checkerboard_covering (n : Nat) (h : n > 0) (board : Checkerboard n) :
  ∃ (covering : Covering n), isValidCovering n board covering :=
sorry

end NUMINAMATH_CALUDE_checkerboard_covering_l2489_248936


namespace NUMINAMATH_CALUDE_c_constant_when_n_doubled_l2489_248983

/-- Given positive constants e, R, and r, and a positive variable n,
    the function C(n) remains constant when n is doubled. -/
theorem c_constant_when_n_doubled
  (e R r : ℝ) (n : ℝ) 
  (he : e > 0) (hR : R > 0) (hr : r > 0) (hn : n > 0) :
  let C : ℝ → ℝ := fun n => (e^2 * n) / (R + n * r^2)
  C n = C (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_c_constant_when_n_doubled_l2489_248983


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l2489_248906

/-- Represents the number of employees to be sampled from each category -/
structure StratifiedSample where
  business : ℕ
  management : ℕ
  logistics : ℕ

/-- Calculates the stratified sample given total employees and sample size -/
def calculateStratifiedSample (totalEmployees business management logistics sampleSize : ℕ) : StratifiedSample :=
  { business := (business * sampleSize) / totalEmployees,
    management := (management * sampleSize) / totalEmployees,
    logistics := (logistics * sampleSize) / totalEmployees }

theorem correct_stratified_sample :
  let totalEmployees : ℕ := 160
  let business : ℕ := 120
  let management : ℕ := 16
  let logistics : ℕ := 24
  let sampleSize : ℕ := 20
  let sample := calculateStratifiedSample totalEmployees business management logistics sampleSize
  sample.business = 15 ∧ sample.management = 2 ∧ sample.logistics = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l2489_248906


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2489_248990

theorem repeating_decimal_sum (a b : ℕ) : 
  (a ≤ 9 ∧ b ≤ 9) →
  (3 : ℚ) / 13 = 
    (a : ℚ) / 10 + (b : ℚ) / 100 + 
    (a : ℚ) / 1000 + (b : ℚ) / 10000 + 
    (a : ℚ) / 100000 + (b : ℚ) / 1000000 + 
    (a : ℚ) / 10000000 + (b : ℚ) / 100000000 + 
    (a : ℚ) / 1000000000 + (b : ℚ) / 10000000000 →
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2489_248990


namespace NUMINAMATH_CALUDE_ruffy_orlie_age_difference_l2489_248959

/-- Proves that given Ruffy's current age is 9 and Ruffy is three-fourths as old as Orlie,
    the difference between Ruffy's age and half of Orlie's age four years ago is 1 year. -/
theorem ruffy_orlie_age_difference : ∀ (ruffy_age orlie_age : ℕ),
  ruffy_age = 9 →
  ruffy_age = (3 * orlie_age) / 4 →
  (ruffy_age - 4) - ((orlie_age - 4) / 2) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ruffy_orlie_age_difference_l2489_248959


namespace NUMINAMATH_CALUDE_no_solution_when_k_equals_five_l2489_248907

theorem no_solution_when_k_equals_five :
  ∀ x : ℝ, x ≠ 2 → x ≠ 6 → (x - 1) / (x - 2) ≠ (x - 5) / (x - 6) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_when_k_equals_five_l2489_248907


namespace NUMINAMATH_CALUDE_rectangular_plot_length_breadth_difference_l2489_248939

theorem rectangular_plot_length_breadth_difference 
  (area length breadth : ℝ)
  (h1 : area = length * breadth)
  (h2 : area = 15 * breadth)
  (h3 : breadth = 5) :
  length - breadth = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_breadth_difference_l2489_248939


namespace NUMINAMATH_CALUDE_perfectSquareFactors_360_l2489_248923

/-- A function that returns the number of perfect square factors of a natural number -/
def perfectSquareFactors (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of perfect square factors of 360 is 4 -/
theorem perfectSquareFactors_360 : perfectSquareFactors 360 = 4 := by sorry

end NUMINAMATH_CALUDE_perfectSquareFactors_360_l2489_248923


namespace NUMINAMATH_CALUDE_cosine_sine_ratio_equals_sqrt_three_l2489_248953

theorem cosine_sine_ratio_equals_sqrt_three : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_ratio_equals_sqrt_three_l2489_248953


namespace NUMINAMATH_CALUDE_set_c_not_right_triangle_set_a_right_triangle_set_b_right_triangle_set_d_right_triangle_l2489_248910

/-- A function to check if three numbers can form a right triangle -/
def can_form_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that (7, 8, 9) cannot form a right triangle -/
theorem set_c_not_right_triangle : ¬(can_form_right_triangle 7 8 9) := by sorry

/-- Theorem stating that (1, 1, √2) can form a right triangle -/
theorem set_a_right_triangle : can_form_right_triangle 1 1 (Real.sqrt 2) := by sorry

/-- Theorem stating that (5, 12, 13) can form a right triangle -/
theorem set_b_right_triangle : can_form_right_triangle 5 12 13 := by sorry

/-- Theorem stating that (1.5, 2, 2.5) can form a right triangle -/
theorem set_d_right_triangle : can_form_right_triangle 1.5 2 2.5 := by sorry

end NUMINAMATH_CALUDE_set_c_not_right_triangle_set_a_right_triangle_set_b_right_triangle_set_d_right_triangle_l2489_248910


namespace NUMINAMATH_CALUDE_angle_315_same_terminal_side_as_negative_45_l2489_248930

-- Define a function to represent angles with the same terminal side
def sameTerminalSide (θ : ℝ) (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + θ

-- State the theorem
theorem angle_315_same_terminal_side_as_negative_45 :
  sameTerminalSide 315 (-45) := by
  sorry

end NUMINAMATH_CALUDE_angle_315_same_terminal_side_as_negative_45_l2489_248930


namespace NUMINAMATH_CALUDE_correct_answers_for_given_exam_l2489_248925

/-- Represents an exam with a fixed number of questions and scoring rules. -/
structure Exam where
  totalQuestions : ℕ
  correctScore : ℕ
  wrongScore : ℤ

/-- Represents a student's exam attempt. -/
structure ExamAttempt where
  exam : Exam
  correctAnswers : ℕ
  wrongAnswers : ℕ
  totalScore : ℤ

/-- Calculates the total score for an exam attempt. -/
def calculateScore (attempt : ExamAttempt) : ℤ :=
  (attempt.correctAnswers : ℤ) * attempt.exam.correctScore - attempt.wrongAnswers * (-attempt.exam.wrongScore)

/-- Theorem stating the correct number of answers for the given exam conditions. -/
theorem correct_answers_for_given_exam :
  ∀ (attempt : ExamAttempt),
    attempt.exam.totalQuestions = 75 →
    attempt.exam.correctScore = 4 →
    attempt.exam.wrongScore = -1 →
    attempt.correctAnswers + attempt.wrongAnswers = attempt.exam.totalQuestions →
    calculateScore attempt = 125 →
    attempt.correctAnswers = 40 := by
  sorry


end NUMINAMATH_CALUDE_correct_answers_for_given_exam_l2489_248925


namespace NUMINAMATH_CALUDE_problem_statement_l2489_248996

theorem problem_statement : 103^4 - 4*103^3 + 6*103^2 - 4*103 + 1 = 108243216 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2489_248996


namespace NUMINAMATH_CALUDE_f_at_two_l2489_248973

/-- The polynomial function f(x) = x^6 - 2x^5 + 3x^3 + 4x^2 - 6x + 5 -/
def f (x : ℝ) : ℝ := x^6 - 2*x^5 + 3*x^3 + 4*x^2 - 6*x + 5

/-- Theorem: The value of f(2) is 29 -/
theorem f_at_two : f 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_f_at_two_l2489_248973


namespace NUMINAMATH_CALUDE_rectangular_section_properties_l2489_248981

/-- A regular tetrahedron with unit edge length -/
structure UnitTetrahedron where
  -- Add necessary fields here

/-- A rectangular section of a tetrahedron -/
structure RectangularSection (T : UnitTetrahedron) where
  -- Add necessary fields here

/-- The perimeter of a rectangular section -/
def perimeter (T : UnitTetrahedron) (S : RectangularSection T) : ℝ :=
  sorry

/-- The area of a rectangular section -/
def area (T : UnitTetrahedron) (S : RectangularSection T) : ℝ :=
  sorry

theorem rectangular_section_properties (T : UnitTetrahedron) :
  (∀ S : RectangularSection T, perimeter T S = 2) ∧
  (∀ S : RectangularSection T, 0 ≤ area T S ∧ area T S ≤ 1/4) :=
sorry

end NUMINAMATH_CALUDE_rectangular_section_properties_l2489_248981


namespace NUMINAMATH_CALUDE_independence_test_smoking_lung_disease_l2489_248920

-- Define the variables and constants
variable (K : ℝ)
variable (confidence_level : ℝ)
variable (error_rate : ℝ)

-- Define the relationship between smoking and lung disease
def smoking_related_to_lung_disease : Prop := sorry

-- Define the critical value for K^2
def critical_value : ℝ := 6.635

-- Define the theorem
theorem independence_test_smoking_lung_disease :
  K ≥ critical_value →
  confidence_level = 0.99 →
  error_rate = 1 - confidence_level →
  smoking_related_to_lung_disease ∧
  (smoking_related_to_lung_disease → error_rate = 0.01) :=
by sorry

end NUMINAMATH_CALUDE_independence_test_smoking_lung_disease_l2489_248920


namespace NUMINAMATH_CALUDE_exactly_one_solution_l2489_248908

-- Define the function g₀
def g₀ (x : ℝ) : ℝ := x + |x - 50| - |x + 150|

-- Define the function gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 1

-- Theorem statement
theorem exactly_one_solution :
  ∃! x, g 100 x = 0 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_solution_l2489_248908
