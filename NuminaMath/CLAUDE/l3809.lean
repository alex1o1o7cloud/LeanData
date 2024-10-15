import Mathlib

namespace NUMINAMATH_CALUDE_sequence_fifth_term_l3809_380984

/-- Given a sequence {aₙ} with the following properties:
  1) a₁ = 1
  2) aₙ - aₙ₋₁ = 2 for n ≥ 2, n ∈ ℕ*
  Prove that a₅ = 9 -/
theorem sequence_fifth_term (a : ℕ+ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ+, n ≥ 2 → a n - a (n-1) = 2) :
  a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sequence_fifth_term_l3809_380984


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_five_l3809_380905

theorem sqrt_sum_equals_five (x y : ℝ) (h : y = Real.sqrt (x - 9) - Real.sqrt (9 - x) + 4) :
  Real.sqrt x + Real.sqrt y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_five_l3809_380905


namespace NUMINAMATH_CALUDE_pencils_purchased_l3809_380919

/-- The number of pens purchased -/
def num_pens : ℕ := 30

/-- The total cost of pens and pencils -/
def total_cost : ℚ := 630

/-- The average price of a pencil -/
def pencil_price : ℚ := 2

/-- The average price of a pen -/
def pen_price : ℚ := 16

/-- The number of pencils purchased -/
def num_pencils : ℕ := 75

theorem pencils_purchased : 
  (num_pens : ℚ) * pen_price + (num_pencils : ℚ) * pencil_price = total_cost := by
  sorry

end NUMINAMATH_CALUDE_pencils_purchased_l3809_380919


namespace NUMINAMATH_CALUDE_kelly_vacation_days_at_sisters_house_l3809_380931

/-- Represents Kelly's vacation schedule --/
structure VacationSchedule where
  totalDays : ℕ
  planeTravelDays : ℕ
  grandparentsDays : ℕ
  trainTravelDays : ℕ
  brotherDays : ℕ
  carToSisterDays : ℕ
  busToSisterDays : ℕ
  timeZoneExtraDays : ℕ
  busBackDays : ℕ
  carBackDays : ℕ

/-- Calculates the number of days Kelly spent at her sister's house --/
def daysAtSistersHouse (schedule : VacationSchedule) : ℕ :=
  schedule.totalDays -
  (schedule.planeTravelDays +
   schedule.grandparentsDays +
   schedule.trainTravelDays +
   schedule.brotherDays +
   schedule.carToSisterDays +
   schedule.busToSisterDays +
   schedule.timeZoneExtraDays +
   schedule.busBackDays +
   schedule.carBackDays)

/-- Theorem stating that Kelly spent 3 days at her sister's house --/
theorem kelly_vacation_days_at_sisters_house :
  ∀ (schedule : VacationSchedule),
    schedule.totalDays = 21 ∧
    schedule.planeTravelDays = 2 ∧
    schedule.grandparentsDays = 5 ∧
    schedule.trainTravelDays = 1 ∧
    schedule.brotherDays = 5 ∧
    schedule.carToSisterDays = 1 ∧
    schedule.busToSisterDays = 1 ∧
    schedule.timeZoneExtraDays = 1 ∧
    schedule.busBackDays = 1 ∧
    schedule.carBackDays = 1 →
    daysAtSistersHouse schedule = 3 := by
  sorry

end NUMINAMATH_CALUDE_kelly_vacation_days_at_sisters_house_l3809_380931


namespace NUMINAMATH_CALUDE_circles_intersect_l3809_380988

/-- The equation of circle C₁ -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y - 2 = 0

/-- The equation of circle C₂ -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The circles C₁ and C₂ intersect -/
theorem circles_intersect : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y :=
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l3809_380988


namespace NUMINAMATH_CALUDE_expression_factorization_l3809_380995

theorem expression_factorization (x : ℝ) :
  (12 * x^4 - 27 * x^2 + 9) - (-3 * x^4 - 9 * x^2 + 6) = 3 * (5 * x^2 - 1) * (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3809_380995


namespace NUMINAMATH_CALUDE_abnormal_segregation_in_secondary_spermatocyte_l3809_380924

-- Define the alleles
inductive Allele
| E  -- normal eye
| e  -- eyeless

-- Define a genotype as a list of alleles
def Genotype := List Allele

-- Define the parents' genotypes
def male_parent : Genotype := [Allele.E, Allele.e]
def female_parent : Genotype := [Allele.e, Allele.e]

-- Define the offspring's genotype
def offspring : Genotype := [Allele.E, Allele.E, Allele.e]

-- Define the possible cell types where segregation could have occurred abnormally
inductive CellType
| PrimarySpermatocyte
| PrimaryOocyte
| SecondarySpermatocyte
| SecondaryOocyte

-- Define the property of no crossing over
def no_crossing_over : Prop := sorry

-- Define the dominance of E over e
def E_dominant_over_e : Prop := sorry

-- Theorem statement
theorem abnormal_segregation_in_secondary_spermatocyte :
  E_dominant_over_e →
  no_crossing_over →
  (∃ (abnormal_cell : CellType), 
    abnormal_cell = CellType.SecondarySpermatocyte ∧
    (∀ (other_cell : CellType), 
      other_cell ≠ CellType.SecondarySpermatocyte → 
      ¬(offspring = [Allele.E, Allele.E, Allele.e]))) :=
sorry

end NUMINAMATH_CALUDE_abnormal_segregation_in_secondary_spermatocyte_l3809_380924


namespace NUMINAMATH_CALUDE_hardware_store_earnings_l3809_380955

/-- Represents the sales data for a single item -/
structure ItemSales where
  quantity : Nat
  price : Nat
  discount_percent : Nat
  returns : Nat

/-- Calculates the total earnings from the hardware store sales -/
def calculate_earnings (sales_data : List ItemSales) : Nat :=
  sales_data.foldl (fun acc item =>
    let gross_sales := item.quantity * item.price
    let discount := gross_sales * item.discount_percent / 100
    let returns := item.returns * item.price
    acc + gross_sales - discount - returns
  ) 0

/-- Theorem stating that the total earnings of the hardware store are $11740 -/
theorem hardware_store_earnings : 
  let sales_data : List ItemSales := [
    { quantity := 10, price := 600, discount_percent := 10, returns := 0 },  -- Graphics cards
    { quantity := 14, price := 80,  discount_percent := 0,  returns := 0 },  -- Hard drives
    { quantity := 8,  price := 200, discount_percent := 0,  returns := 2 },  -- CPUs
    { quantity := 4,  price := 60,  discount_percent := 0,  returns := 0 },  -- RAM
    { quantity := 12, price := 90,  discount_percent := 0,  returns := 0 },  -- Power supply units
    { quantity := 6,  price := 250, discount_percent := 0,  returns := 0 },  -- Monitors
    { quantity := 18, price := 40,  discount_percent := 0,  returns := 0 },  -- Keyboards
    { quantity := 24, price := 20,  discount_percent := 0,  returns := 0 }   -- Mice
  ]
  calculate_earnings sales_data = 11740 := by
  sorry


end NUMINAMATH_CALUDE_hardware_store_earnings_l3809_380955


namespace NUMINAMATH_CALUDE_arithmetic_mean_greater_than_harmonic_mean_l3809_380962

theorem arithmetic_mean_greater_than_harmonic_mean 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (a + b) / 2 > 2 * a * b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_greater_than_harmonic_mean_l3809_380962


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3809_380926

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 + 3*x - 9 = 0) :
  x^3 + 3*x^2 - 9*x - 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3809_380926


namespace NUMINAMATH_CALUDE_trapezoid_area_is_72_l3809_380968

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- One of the base angles in radians -/
  baseAngle : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The trapezoid is circumscribed around a circle -/
  isCircumscribed : True

/-- Calculate the area of the isosceles trapezoid -/
def areaOfTrapezoid (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem trapezoid_area_is_72 (t : IsoscelesTrapezoid) 
  (h1 : t.longerBase = 20)
  (h2 : t.baseAngle = Real.arcsin 0.6) :
  areaOfTrapezoid t = 72 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_72_l3809_380968


namespace NUMINAMATH_CALUDE_probability_one_ball_in_last_box_l3809_380954

theorem probability_one_ball_in_last_box (n : ℕ) (h : n = 100) :
  let p := 1 / n
  (n : ℝ) * p * (1 - p)^(n - 1) = ((n - 1 : ℝ) / n)^(n - 1) :=
by sorry

end NUMINAMATH_CALUDE_probability_one_ball_in_last_box_l3809_380954


namespace NUMINAMATH_CALUDE_tangent_problem_l3809_380932

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (1 + Real.tan α) / (1 - Real.tan α) = 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l3809_380932


namespace NUMINAMATH_CALUDE_triangle_area_l3809_380974

/-- Given a triangle ABC with side lengths a, b, c, prove that its area is √3/4 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  b = 1 →
  b * Real.cos C = c * Real.cos B →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3809_380974


namespace NUMINAMATH_CALUDE_base3_20112_equals_176_l3809_380942

/-- Converts a base-3 digit to its base-10 equivalent --/
def base3ToBase10Digit (d : Nat) : Nat :=
  if d < 3 then d else 0

/-- Converts a base-3 number represented as a list of digits to its base-10 equivalent --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + base3ToBase10Digit d * 3^i) 0

theorem base3_20112_equals_176 :
  base3ToBase10 [2, 1, 1, 0, 2] = 176 := by
  sorry

end NUMINAMATH_CALUDE_base3_20112_equals_176_l3809_380942


namespace NUMINAMATH_CALUDE_intersecting_circles_properties_l3809_380999

/-- Two circles intersecting at two distinct points -/
structure IntersectingCircles where
  r : ℝ
  a : ℝ
  b : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  r_pos : r > 0
  on_C₁_A : x₁^2 + y₁^2 = r^2
  on_C₁_B : x₂^2 + y₂^2 = r^2
  on_C₂_A : (x₁ - a)^2 + (y₁ - b)^2 = r^2
  on_C₂_B : (x₂ - a)^2 + (y₂ - b)^2 = r^2
  distinct : (x₁, y₁) ≠ (x₂, y₂)

/-- Properties of intersecting circles -/
theorem intersecting_circles_properties (c : IntersectingCircles) :
  (c.a * (c.x₁ - c.x₂) + c.b * (c.y₁ - c.y₂) = 0) ∧
  (2 * c.a * c.x₁ + 2 * c.b * c.y₁ = c.a^2 + c.b^2) ∧
  (c.x₁ + c.x₂ = c.a ∧ c.y₁ + c.y₂ = c.b) := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_properties_l3809_380999


namespace NUMINAMATH_CALUDE_line_circle_intersection_slope_range_l3809_380966

/-- Given a line intersecting a circle, prove the range of its slope. -/
theorem line_circle_intersection_slope_range :
  ∀ (a : ℝ),
  (∃ (x y : ℝ), x + a * y + 2 = 0 ∧ x^2 + y^2 + 2*x - 2*y + 1 = 0) →
  a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_slope_range_l3809_380966


namespace NUMINAMATH_CALUDE_sin_decreasing_interval_l3809_380933

/-- The function f(x) = sin(π/6 - x) is strictly decreasing on the interval [0, 2π/3] -/
theorem sin_decreasing_interval (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi / 3) →
  StrictMonoOn (fun x => Real.sin (Real.pi / 6 - x)) (Set.Icc 0 (2 * Real.pi / 3)) := by
  sorry

end NUMINAMATH_CALUDE_sin_decreasing_interval_l3809_380933


namespace NUMINAMATH_CALUDE_contract_schemes_count_l3809_380960

def projects : ℕ := 6
def company_a_projects : ℕ := 3
def company_b_projects : ℕ := 2
def company_c_projects : ℕ := 1

theorem contract_schemes_count :
  (Nat.choose projects company_a_projects) *
  (Nat.choose (projects - company_a_projects) company_b_projects) *
  (Nat.choose (projects - company_a_projects - company_b_projects) company_c_projects) = 60 := by
  sorry

end NUMINAMATH_CALUDE_contract_schemes_count_l3809_380960


namespace NUMINAMATH_CALUDE_solve_for_m_l3809_380963

/-- Given that the solution set of mx + 2 > 0 is {x | x < 2}, prove that m = -1 -/
theorem solve_for_m (m : ℝ) 
  (h : ∀ x, mx + 2 > 0 ↔ x < 2) : 
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3809_380963


namespace NUMINAMATH_CALUDE_triangle_sum_formula_l3809_380904

def triangleSum (n : ℕ) : ℕ := 8 * 2^n - 4

theorem triangle_sum_formula (n : ℕ) : 
  n ≥ 1 → 
  (∀ k, k ≥ 2 → triangleSum k = 2 * triangleSum (k-1) + 4) → 
  triangleSum 1 = 4 → 
  triangleSum n = 8 * 2^n - 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_formula_l3809_380904


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3809_380979

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (0 < x ∧ x < 5) → (-5 < x - 2 ∧ x - 2 < 5)) ∧
  (∃ x : ℝ, (-5 < x - 2 ∧ x - 2 < 5) ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3809_380979


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3809_380978

theorem sqrt_equation_solution :
  ∃ (x : ℝ), x = (1225 : ℝ) / 36 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3809_380978


namespace NUMINAMATH_CALUDE_jenny_jellybeans_proof_l3809_380959

/-- The original number of jellybeans in Jenny's jar -/
def original_jellybeans : ℝ := 85

/-- The fraction of jellybeans remaining after each day -/
def daily_remaining_fraction : ℝ := 0.7

/-- The number of days Jenny eats jellybeans -/
def days : ℕ := 3

/-- The number of jellybeans remaining after 'days' days -/
def remaining_jellybeans : ℝ := 29.16

/-- Theorem stating that the original number of jellybeans is correct -/
theorem jenny_jellybeans_proof :
  original_jellybeans * daily_remaining_fraction ^ days = remaining_jellybeans := by
  sorry

#eval original_jellybeans -- Should output 85

end NUMINAMATH_CALUDE_jenny_jellybeans_proof_l3809_380959


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_plus_i_l3809_380916

theorem imaginary_part_of_i_over_one_plus_i : Complex.im (Complex.I / (1 + Complex.I)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_plus_i_l3809_380916


namespace NUMINAMATH_CALUDE_inequality_solution_l3809_380985

-- Define the polynomial function
def f (x : ℝ) := x^3 - 4*x^2 - x + 20

-- Define the set of x satisfying the inequality
def S : Set ℝ := {x | f x > 0}

-- State the theorem
theorem inequality_solution : S = Set.Ioi (-4) ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3809_380985


namespace NUMINAMATH_CALUDE_projection_property_l3809_380911

/-- A projection that takes [4, 4] to [60/13, 12/13] -/
def projection (v : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem projection_property : 
  projection (4, 4) = (60/13, 12/13) ∧ 
  projection (-2, 2) = (-20/13, -4/13) := by
  sorry

end NUMINAMATH_CALUDE_projection_property_l3809_380911


namespace NUMINAMATH_CALUDE_third_year_percentage_l3809_380956

theorem third_year_percentage
  (total : ℝ)
  (third_year : ℝ)
  (second_year : ℝ)
  (h1 : second_year = 0.1 * total)
  (h2 : second_year / (total - third_year) = 1 / 7)
  : third_year = 0.3 * total :=
by sorry

end NUMINAMATH_CALUDE_third_year_percentage_l3809_380956


namespace NUMINAMATH_CALUDE_right_triangle_area_l3809_380997

theorem right_triangle_area (base hypotenuse : ℝ) (h_right_angle : base > 0 ∧ hypotenuse > 0 ∧ hypotenuse > base) :
  base = 12 → hypotenuse = 13 → 
  ∃ height : ℝ, height > 0 ∧ height ^ 2 + base ^ 2 = hypotenuse ^ 2 ∧ 
  (1 / 2) * base * height = 30 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3809_380997


namespace NUMINAMATH_CALUDE_octagon_side_length_l3809_380912

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  w : Point
  x : Point
  y : Point
  z : Point

/-- Represents an octagon -/
structure Octagon where
  a : Point
  b : Point
  c : Point
  d : Point
  e : Point
  f : Point
  g : Point
  h : Point

def is_on_line (p q r : Point) : Prop := sorry

def is_equilateral (oct : Octagon) : Prop := sorry

def is_convex (oct : Octagon) : Prop := sorry

def side_length (oct : Octagon) : ℝ := sorry

theorem octagon_side_length 
  (rect : Rectangle)
  (oct : Octagon)
  (h1 : rect.z.x - rect.w.x = 10)
  (h2 : rect.y.y - rect.z.y = 8)
  (h3 : is_on_line rect.w oct.a rect.z)
  (h4 : is_on_line rect.w oct.b rect.z)
  (h5 : is_on_line rect.z oct.c rect.y)
  (h6 : is_on_line rect.z oct.d rect.y)
  (h7 : is_on_line rect.y oct.e rect.w)
  (h8 : is_on_line rect.y oct.f rect.w)
  (h9 : is_on_line rect.x oct.g rect.w)
  (h10 : is_on_line rect.x oct.h rect.w)
  (h11 : oct.a.x - rect.w.x = rect.z.x - oct.b.x)
  (h12 : oct.a.x - rect.w.x ≤ 5)
  (h13 : is_equilateral oct)
  (h14 : is_convex oct) :
  side_length oct = -9 + Real.sqrt 652 := by sorry

end NUMINAMATH_CALUDE_octagon_side_length_l3809_380912


namespace NUMINAMATH_CALUDE_original_savings_calculation_l3809_380969

theorem original_savings_calculation (savings : ℝ) : 
  (5/6 : ℝ) * savings + 500 = savings → savings = 3000 := by
  sorry

end NUMINAMATH_CALUDE_original_savings_calculation_l3809_380969


namespace NUMINAMATH_CALUDE_certain_triangle_angle_sum_l3809_380971

/-- A triangle is a shape with three sides -/
structure Triangle where
  sides : Fin 3 → ℝ
  positive : ∀ i, sides i > 0

/-- The sum of interior angles of a triangle is 180° -/
axiom triangle_angle_sum (t : Triangle) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

/-- For any triangle, the sum of its interior angles is always 180° -/
theorem certain_triangle_angle_sum (t : Triangle) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180 := by
  sorry

end NUMINAMATH_CALUDE_certain_triangle_angle_sum_l3809_380971


namespace NUMINAMATH_CALUDE_enclosed_area_of_special_curve_l3809_380951

/-- The area enclosed by a curve consisting of 9 congruent circular arcs, 
    each of length 2π/3, with centers on the vertices of a regular hexagon 
    with side length 3, is equal to 13.5√3 + π. -/
theorem enclosed_area_of_special_curve (
  n : ℕ) (arc_length : ℝ) (hexagon_side : ℝ) (enclosed_area : ℝ) : 
  n = 9 → 
  arc_length = 2 * Real.pi / 3 → 
  hexagon_side = 3 → 
  enclosed_area = 13.5 * Real.sqrt 3 + Real.pi → 
  enclosed_area = 
    (3 * Real.sqrt 3 / 2 * hexagon_side^2) + (n * arc_length * (arc_length / (2 * Real.pi))) :=
by sorry

end NUMINAMATH_CALUDE_enclosed_area_of_special_curve_l3809_380951


namespace NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_5_8_2_l3809_380972

theorem smallest_three_digit_divisible_by_5_8_2 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 5 ∣ n ∧ 8 ∣ n ∧ 2 ∣ n → n ≥ 120 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_5_8_2_l3809_380972


namespace NUMINAMATH_CALUDE_order_of_numbers_l3809_380970

theorem order_of_numbers (x y z : ℝ) (h1 : 0.9 < x) (h2 : x < 1) 
  (h3 : y = x^x) (h4 : z = x^(x^x)) : x < z ∧ z < y := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3809_380970


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l3809_380949

def A : Set ℝ := {3, 5}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem subset_implies_a_values (a : ℝ) (h : B a ⊆ A) : a = 0 ∨ a = 1/3 ∨ a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l3809_380949


namespace NUMINAMATH_CALUDE_workshop_workers_l3809_380998

theorem workshop_workers (total_average : ℕ) (technician_average : ℕ) (other_average : ℕ) 
  (technician_count : ℕ) :
  total_average = 8000 →
  technician_average = 16000 →
  other_average = 6000 →
  technician_count = 7 →
  ∃ (total_workers : ℕ), 
    total_workers * total_average = 
      technician_count * technician_average + (total_workers - technician_count) * other_average ∧
    total_workers = 35 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l3809_380998


namespace NUMINAMATH_CALUDE_exclusive_multiples_of_6_or_8_less_than_151_l3809_380913

def count_multiples (n m : ℕ) : ℕ := (n - 1) / m

def count_exclusive_multiples (upper bound1 bound2 : ℕ) : ℕ :=
  let lcm := Nat.lcm bound1 bound2
  (count_multiples upper bound1) + (count_multiples upper bound2) - 2 * (count_multiples upper lcm)

theorem exclusive_multiples_of_6_or_8_less_than_151 :
  count_exclusive_multiples 151 6 8 = 31 := by sorry

end NUMINAMATH_CALUDE_exclusive_multiples_of_6_or_8_less_than_151_l3809_380913


namespace NUMINAMATH_CALUDE_m_greater_than_n_l3809_380946

/-- Given two quadratic functions M and N, prove that M > N for all real x. -/
theorem m_greater_than_n : ∀ x : ℝ, (x^2 - 3*x + 7) > (-x^2 + x + 1) := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l3809_380946


namespace NUMINAMATH_CALUDE_fraction_equality_proof_l3809_380917

theorem fraction_equality_proof (a b z : ℕ+) (h : a * b = z^2 + 1) :
  ∃ (x y : ℕ+), (a : ℚ) / b = ((x^2 : ℚ) + 1) / ((y^2 : ℚ) + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_proof_l3809_380917


namespace NUMINAMATH_CALUDE_blue_tickets_per_red_l3809_380980

/-- The number of yellow tickets needed to win a Bible -/
def yellow_tickets_needed : ℕ := 10

/-- The number of red tickets needed for one yellow ticket -/
def red_per_yellow : ℕ := 10

/-- The number of yellow tickets Tom has -/
def tom_yellow : ℕ := 8

/-- The number of red tickets Tom has -/
def tom_red : ℕ := 3

/-- The number of blue tickets Tom has -/
def tom_blue : ℕ := 7

/-- The number of additional blue tickets Tom needs to win a Bible -/
def additional_blue_needed : ℕ := 163

/-- The number of blue tickets required to obtain one red ticket -/
def blue_per_red : ℕ := 10

theorem blue_tickets_per_red : 
  yellow_tickets_needed = 10 ∧ 
  red_per_yellow = 10 ∧ 
  tom_yellow = 8 ∧ 
  tom_red = 3 ∧ 
  tom_blue = 7 ∧ 
  additional_blue_needed = 163 → 
  blue_per_red = 10 := by sorry

end NUMINAMATH_CALUDE_blue_tickets_per_red_l3809_380980


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3809_380940

theorem inequality_system_solution (x : ℝ) :
  x - 2 ≤ 0 ∧ (x - 1) / 2 < x → -1 < x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3809_380940


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l3809_380950

theorem floor_abs_negative_real : ⌊|(-58.7 : ℝ)|⌋ = 58 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l3809_380950


namespace NUMINAMATH_CALUDE_max_value_at_two_a_is_max_point_l3809_380918

-- Define the function f(x) = -x^3 + 12x
def f (x : ℝ) : ℝ := -x^3 + 12*x

-- State the theorem
theorem max_value_at_two : 
  ∀ x : ℝ, f x ≤ f 2 := by
sorry

-- Define a as the point where f reaches its maximum value
def a : ℝ := 2

-- State that a is indeed the point of maximum value
theorem a_is_max_point : 
  ∀ x : ℝ, f x ≤ f a := by
sorry

end NUMINAMATH_CALUDE_max_value_at_two_a_is_max_point_l3809_380918


namespace NUMINAMATH_CALUDE_arrangements_not_adjacent_l3809_380947

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n distinct objects, treating two objects as a single unit -/
def arrangements_with_pair (n : ℕ) : ℕ := 2 * factorial (n - 1)

/-- The number of ways to arrange 4 distinct people such that two specific people are not next to each other -/
theorem arrangements_not_adjacent : 
  factorial 4 - arrangements_with_pair 4 = 12 := by sorry

end NUMINAMATH_CALUDE_arrangements_not_adjacent_l3809_380947


namespace NUMINAMATH_CALUDE_cupcakes_left_l3809_380992

/-- The number of cupcakes in a dozen -/
def dozen : ℕ := 12

/-- The number of cupcakes Dani brings -/
def cupcakes_brought : ℕ := 2 * dozen + dozen / 2

/-- The total number of students -/
def total_students : ℕ := 27

/-- The number of teachers -/
def teachers : ℕ := 1

/-- The number of teacher's aids -/
def teacher_aids : ℕ := 1

/-- The number of students who called in sick -/
def sick_students : ℕ := 3

/-- Theorem: The number of cupcakes left after distribution -/
theorem cupcakes_left : 
  cupcakes_brought - (total_students - sick_students + teachers + teacher_aids) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_left_l3809_380992


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l3809_380945

theorem simplify_and_ratio : 
  (∀ m : ℝ, (6*m + 12) / 3 = 2*m + 4) ∧ (2 / 4 : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l3809_380945


namespace NUMINAMATH_CALUDE_number_operation_result_l3809_380909

theorem number_operation_result (x : ℝ) : x + 7 = 27 → ((x / 5) + 5) * 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_result_l3809_380909


namespace NUMINAMATH_CALUDE_stanley_walk_distance_l3809_380930

theorem stanley_walk_distance (run_distance walk_distance : ℝ) :
  run_distance = 0.4 →
  run_distance = walk_distance + 0.2 →
  walk_distance = 0.2 := by
sorry

end NUMINAMATH_CALUDE_stanley_walk_distance_l3809_380930


namespace NUMINAMATH_CALUDE_arithmetic_sum_equals_square_l3809_380908

theorem arithmetic_sum_equals_square (n : ℕ) :
  let first_term := 1
  let last_term := 2*n + 3
  let num_terms := n + 2
  (num_terms * (first_term + last_term)) / 2 = (n + 2)^2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sum_equals_square_l3809_380908


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l3809_380983

theorem square_perimeter_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_area_ratio : a^2 / b^2 = 16 / 25) :
  a / b = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l3809_380983


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_42_l3809_380929

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the factors of 42
def factorsOf42 : List ℕ := [2, 3, 7]

-- Theorem statement
theorem no_primes_divisible_by_42 : 
  ∀ p : ℕ, isPrime p → ¬(42 ∣ p) :=
sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_42_l3809_380929


namespace NUMINAMATH_CALUDE_debt_average_payment_l3809_380923

/-- Proves that the average payment for a debt with specific conditions is $465 --/
theorem debt_average_payment 
  (total_installments : ℕ) 
  (first_payment_count : ℕ) 
  (first_payment_amount : ℚ) 
  (additional_amount : ℚ) : 
  total_installments = 52 →
  first_payment_count = 8 →
  first_payment_amount = 410 →
  additional_amount = 65 →
  let remaining_payment_count := total_installments - first_payment_count
  let remaining_payment_amount := first_payment_amount + additional_amount
  let total_amount := 
    (first_payment_count * first_payment_amount) + 
    (remaining_payment_count * remaining_payment_amount)
  total_amount / total_installments = 465 := by
  sorry

end NUMINAMATH_CALUDE_debt_average_payment_l3809_380923


namespace NUMINAMATH_CALUDE_sum_of_mean_and_median_l3809_380944

def number_set : List ℕ := [1, 2, 3, 0, 1]

def median (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

theorem sum_of_mean_and_median :
  median number_set + mean number_set = 12/5 := by sorry

end NUMINAMATH_CALUDE_sum_of_mean_and_median_l3809_380944


namespace NUMINAMATH_CALUDE_class_average_weight_l3809_380961

/-- The average weight of a group of children given their total weight and count -/
def average_weight (total_weight : ℚ) (count : ℕ) : ℚ :=
  total_weight / count

/-- The total weight of a group of children given their average weight and count -/
def total_weight (avg_weight : ℚ) (count : ℕ) : ℚ :=
  avg_weight * count

theorem class_average_weight 
  (boys_count : ℕ) 
  (girls_count : ℕ) 
  (boys_avg_weight : ℚ) 
  (girls_avg_weight : ℚ) 
  (h1 : boys_count = 8)
  (h2 : girls_count = 6)
  (h3 : boys_avg_weight = 140)
  (h4 : girls_avg_weight = 130) :
  average_weight 
    (total_weight boys_avg_weight boys_count + total_weight girls_avg_weight girls_count) 
    (boys_count + girls_count) = 135 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l3809_380961


namespace NUMINAMATH_CALUDE_product_not_fifty_l3809_380928

theorem product_not_fifty : ∃! (a b : ℚ), (a = 5 ∧ b = 11) ∧ a * b ≠ 50 ∧
  ((a = 1/2 ∧ b = 100) ∨ (a = -5 ∧ b = -10) ∨ (a = 2 ∧ b = 25) ∨ (a = 5/2 ∧ b = 20)) → a * b = 50 :=
by sorry

end NUMINAMATH_CALUDE_product_not_fifty_l3809_380928


namespace NUMINAMATH_CALUDE_top_square_is_one_l3809_380993

/-- Represents a 4x4 grid of squares --/
def Grid := Fin 4 → Fin 4 → ℕ

/-- Initial configuration of the grid --/
def initial_grid : Grid :=
  λ i j => 4 * i.val + j.val + 1

/-- Fold right half over left half --/
def fold_right_left (g : Grid) : Grid :=
  λ i j => g i (Fin.ofNat (3 - j.val))

/-- Fold left half over right half --/
def fold_left_right (g : Grid) : Grid :=
  λ i j => g i j

/-- Fold top half over bottom half --/
def fold_top_bottom (g : Grid) : Grid :=
  λ i j => g (Fin.ofNat (3 - i.val)) j

/-- Fold bottom half over top half --/
def fold_bottom_top (g : Grid) : Grid :=
  λ i j => g i j

/-- Perform all folds in sequence --/
def perform_folds (g : Grid) : Grid :=
  fold_bottom_top ∘ fold_top_bottom ∘ fold_left_right ∘ fold_right_left $ g

theorem top_square_is_one :
  (perform_folds initial_grid) 0 0 = 1 := by sorry

end NUMINAMATH_CALUDE_top_square_is_one_l3809_380993


namespace NUMINAMATH_CALUDE_beavers_swimming_l3809_380991

theorem beavers_swimming (initial_beavers : ℕ) (remaining_beavers : ℕ) : 
  initial_beavers = 2 → remaining_beavers = 1 → initial_beavers - remaining_beavers = 1 := by
  sorry

end NUMINAMATH_CALUDE_beavers_swimming_l3809_380991


namespace NUMINAMATH_CALUDE_oliver_final_balance_l3809_380939

def olivers_money (initial_amount savings chores_earnings frisbee_cost puzzle_cost stickers_cost
                   movie_ticket_cost snack_cost birthday_gift : ℤ) : ℤ :=
  initial_amount + savings + chores_earnings - frisbee_cost - puzzle_cost - stickers_cost -
  movie_ticket_cost - snack_cost + birthday_gift

theorem oliver_final_balance :
  olivers_money 9 5 6 4 3 2 7 3 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_oliver_final_balance_l3809_380939


namespace NUMINAMATH_CALUDE_oreo_shop_combinations_l3809_380922

/-- Represents the number of flavors for each product type -/
structure Flavors where
  oreos : Nat
  milk : Nat
  cookies : Nat

/-- Represents the purchasing rules for Alpha and Gamma -/
structure PurchaseRules where
  alpha_max_items : Nat
  alpha_allows_repeats : Bool
  gamma_allowed_products : List String
  gamma_allows_repeats : Bool

/-- Calculates the number of ways to purchase items given the rules and flavors -/
def purchase_combinations (flavors : Flavors) (rules : PurchaseRules) (total_items : Nat) : Nat :=
  sorry

/-- The main theorem stating the number of purchase combinations -/
theorem oreo_shop_combinations :
  let flavors := Flavors.mk 5 3 2
  let rules := PurchaseRules.mk 2 false ["oreos", "cookies"] true
  purchase_combinations flavors rules 4 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_oreo_shop_combinations_l3809_380922


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l3809_380994

theorem arithmetic_series_sum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = -300 →
  aₙ = 309 →
  d = 3 →
  n = (aₙ - a₁) / d + 1 →
  (n : ℤ) * (a₁ + aₙ) / 2 = 918 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l3809_380994


namespace NUMINAMATH_CALUDE_kids_on_other_days_l3809_380903

/-- 
Given that Julia played tag with some kids from Monday to Friday,
prove that the number of kids she played with on Monday, Thursday, and Friday combined
is equal to the total number of kids minus the number of kids on Tuesday and Wednesday.
-/
theorem kids_on_other_days 
  (total_kids : ℕ) 
  (tuesday_wednesday_kids : ℕ) 
  (h1 : total_kids = 75) 
  (h2 : tuesday_wednesday_kids = 36) : 
  total_kids - tuesday_wednesday_kids = 39 := by
sorry

end NUMINAMATH_CALUDE_kids_on_other_days_l3809_380903


namespace NUMINAMATH_CALUDE_min_sum_with_product_constraint_l3809_380981

theorem min_sum_with_product_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : a * b = a + b + 3) : 
  6 ≤ a + b ∧ ∀ (x y : ℝ), 0 < x → 0 < y → x * y = x + y + 3 → a + b ≤ x + y := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_product_constraint_l3809_380981


namespace NUMINAMATH_CALUDE_last_box_weight_l3809_380996

theorem last_box_weight (box1_weight box2_weight total_weight : ℕ) : 
  box1_weight = 2 → 
  box2_weight = 11 → 
  total_weight = 18 → 
  ∃ last_box_weight : ℕ, last_box_weight = total_weight - (box1_weight + box2_weight) ∧ 
                           last_box_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_box_weight_l3809_380996


namespace NUMINAMATH_CALUDE_eight_possible_pairs_l3809_380990

/-- Represents a seating arrangement at a round table -/
structure RoundTable :=
  (people : Finset (Fin 5))
  (girls : Finset (Fin 5))
  (boys : Finset (Fin 5))
  (all_seated : people = Finset.univ)
  (girls_boys_partition : girls ∪ boys = people ∧ girls ∩ boys = ∅)

/-- The number of people sitting next to at least one girl -/
def g (table : RoundTable) : ℕ := sorry

/-- The number of people sitting next to at least one boy -/
def b (table : RoundTable) : ℕ := sorry

/-- The set of all possible (g,b) pairs for a given round table -/
def possible_pairs (table : RoundTable) : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 = g table ∧ p.2 = b table) (Finset.product (Finset.range 6) (Finset.range 6))

/-- The theorem stating that there are exactly 8 possible (g,b) pairs -/
theorem eight_possible_pairs :
  ∀ table : RoundTable, Finset.card (possible_pairs table) = 8 := sorry

end NUMINAMATH_CALUDE_eight_possible_pairs_l3809_380990


namespace NUMINAMATH_CALUDE_education_allocation_l3809_380906

def town_budget : ℕ := 32000000

theorem education_allocation :
  let policing : ℕ := town_budget / 2
  let public_spaces : ℕ := 4000000
  let education : ℕ := town_budget - (policing + public_spaces)
  education = 12000000 := by sorry

end NUMINAMATH_CALUDE_education_allocation_l3809_380906


namespace NUMINAMATH_CALUDE_fibonacci_rectangle_division_l3809_380953

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- A rectangle that can be divided into squares -/
structure DivisibleRectangle where
  width : ℕ
  height : ℕ
  num_squares : ℕ
  max_identical_squares : ℕ

/-- Proposition: For every natural number n, there exists a rectangle with 
    dimensions Fn × Fn+1 that can be divided into exactly n squares, 
    with no more than two squares of the same size -/
theorem fibonacci_rectangle_division (n : ℕ) : 
  ∃ (rect : DivisibleRectangle), 
    rect.width = fib n ∧ 
    rect.height = fib (n + 1) ∧ 
    rect.num_squares = n ∧ 
    rect.max_identical_squares ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_rectangle_division_l3809_380953


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l3809_380987

theorem nested_fraction_equality : 
  (((((3 + 2)⁻¹ + 2)⁻¹ + 1)⁻¹ + 2)⁻¹ + 1 : ℚ) = 59 / 43 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l3809_380987


namespace NUMINAMATH_CALUDE_divisibility_implies_r_value_l3809_380965

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := 10 * x^3 - 5 * x^2 - 52 * x + 56

/-- Divisibility condition -/
def is_divisible_by_square (r : ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - r)^2 * q x

theorem divisibility_implies_r_value :
  ∀ r : ℝ, is_divisible_by_square r → r = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_r_value_l3809_380965


namespace NUMINAMATH_CALUDE_sequence_ratio_proof_l3809_380927

theorem sequence_ratio_proof (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a n > 0) →
  (∀ n, (a (n + 1))^2 = (a n) * (a (n + 2))) →
  (S 3 = 13) →
  (a 1 = 1) →
  ((a 3 + a 4) / (a 1 + a 2) = 9) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_proof_l3809_380927


namespace NUMINAMATH_CALUDE_min_value_of_function_l3809_380967

open Real

theorem min_value_of_function (θ : ℝ) (h1 : sin θ ≠ 0) (h2 : cos θ ≠ 0) :
  ∃ (min_val : ℝ), min_val = 5 * sqrt 5 ∧ 
  ∀ θ', sin θ' ≠ 0 → cos θ' ≠ 0 → 1 / sin θ' + 8 / cos θ' ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3809_380967


namespace NUMINAMATH_CALUDE_x_plus_y_equals_30_l3809_380920

theorem x_plus_y_equals_30 (x y : ℝ) 
  (h1 : |x| - x + y = 6) 
  (h2 : x + |y| + y = 8) : 
  x + y = 30 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_30_l3809_380920


namespace NUMINAMATH_CALUDE_triangle_properties_l3809_380925

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a + b + c = 10 →
  Real.sin B + Real.sin C = 4 * Real.sin A →
  b * c = 16 →
  (a = 2 ∧ Real.cos A = 7/8) := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3809_380925


namespace NUMINAMATH_CALUDE_sequence_general_term_l3809_380989

def S (n : ℕ+) : ℤ := n.val^2 - 2

def a : ℕ+ → ℤ
  | ⟨1, _⟩ => -1
  | ⟨n+2, _⟩ => 2*(n+2) - 1

theorem sequence_general_term (n : ℕ+) : 
  a n = if n = 1 then -1 else S n - S (n-1) := by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3809_380989


namespace NUMINAMATH_CALUDE_parallel_transitivity_l3809_380958

-- Define the type for planes
def Plane : Type := Unit

-- Define the parallel relation between planes
def parallel (p q : Plane) : Prop := sorry

-- State the theorem
theorem parallel_transitivity (α β γ : Plane) 
  (h1 : α ≠ β) (h2 : β ≠ γ) (h3 : α ≠ γ)
  (h4 : parallel α β) (h5 : parallel β γ) : 
  parallel α γ := by sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l3809_380958


namespace NUMINAMATH_CALUDE_tom_bonus_percentage_l3809_380935

theorem tom_bonus_percentage :
  let points_per_enemy : ℕ := 10
  let enemies_killed : ℕ := 150
  let total_score : ℕ := 2250
  let score_without_bonus : ℕ := points_per_enemy * enemies_killed
  let bonus : ℕ := total_score - score_without_bonus
  let bonus_percentage : ℚ := (bonus : ℚ) / (score_without_bonus : ℚ) * 100
  bonus_percentage = 50 := by
sorry

end NUMINAMATH_CALUDE_tom_bonus_percentage_l3809_380935


namespace NUMINAMATH_CALUDE_fraction_sum_product_equality_l3809_380975

theorem fraction_sum_product_equality : (1 : ℚ) / 2 + ((1 : ℚ) / 2 * (1 : ℚ) / 2) = (3 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_product_equality_l3809_380975


namespace NUMINAMATH_CALUDE_two_in_A_implies_a_eq_two_A_eq_B_implies_a_eq_one_A_intersection_B_eq_A_implies_a_eq_one_or_four_l3809_380934

-- Define set A
def A (a : ℝ) : Set ℝ := {x | x^2 + 4*a = (a + 4)*x}

-- Define set B
def B : Set ℝ := {x | x^2 + 4 = 5*x}

-- Theorem 1
theorem two_in_A_implies_a_eq_two :
  ∀ a : ℝ, 2 ∈ A a → a = 2 := by sorry

-- Theorem 2
theorem A_eq_B_implies_a_eq_one :
  ∀ a : ℝ, A a = B → a = 1 := by sorry

-- Theorem 3
theorem A_intersection_B_eq_A_implies_a_eq_one_or_four :
  ∀ a : ℝ, A a ∩ B = A a → a = 1 ∨ a = 4 := by sorry

end NUMINAMATH_CALUDE_two_in_A_implies_a_eq_two_A_eq_B_implies_a_eq_one_A_intersection_B_eq_A_implies_a_eq_one_or_four_l3809_380934


namespace NUMINAMATH_CALUDE_road_trip_gas_cost_jennas_road_trip_cost_l3809_380900

/-- Calculates the cost of a road trip given driving times, speeds, and gas efficiency --/
theorem road_trip_gas_cost 
  (time1 : ℝ) (speed1 : ℝ) (time2 : ℝ) (speed2 : ℝ) 
  (gas_efficiency : ℝ) (gas_price : ℝ) : ℝ :=
  let distance1 := time1 * speed1
  let distance2 := time2 * speed2
  let total_distance := distance1 + distance2
  let gas_used := total_distance / gas_efficiency
  let total_cost := gas_used * gas_price
  total_cost

/-- Proves that Jenna's road trip gas cost is $18 --/
theorem jennas_road_trip_cost : 
  road_trip_gas_cost 2 60 3 50 30 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_gas_cost_jennas_road_trip_cost_l3809_380900


namespace NUMINAMATH_CALUDE_harmonic_series_term_count_l3809_380976

theorem harmonic_series_term_count (k : ℕ) (h : k ≥ 2) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end NUMINAMATH_CALUDE_harmonic_series_term_count_l3809_380976


namespace NUMINAMATH_CALUDE_cases_in_2007_l3809_380902

/-- Calculates the number of disease cases in a given year, assuming a linear decrease --/
def diseaseCases (initialYear initialCases finalYear finalCases targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let annualDecrease := totalDecrease / totalYears
  let targetYearsSinceInitial := targetYear - initialYear
  initialCases - (annualDecrease * targetYearsSinceInitial)

/-- The number of disease cases in 2007, given the conditions --/
theorem cases_in_2007 :
  diseaseCases 1980 300000 2016 1000 2007 = 75738 := by
  sorry

#eval diseaseCases 1980 300000 2016 1000 2007

end NUMINAMATH_CALUDE_cases_in_2007_l3809_380902


namespace NUMINAMATH_CALUDE_student_lecture_choices_l3809_380986

/-- The number of different choices when n students can each independently
    choose one of m lectures to attend -/
def number_of_choices (n m : ℕ) : ℕ := m^n

/-- Theorem: Given 5 students and 3 lectures, where each student can independently
    choose one lecture to attend, the total number of different possible choices is 3^5 -/
theorem student_lecture_choices :
  number_of_choices 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_student_lecture_choices_l3809_380986


namespace NUMINAMATH_CALUDE_polygon_with_two_diagonals_has_five_sides_l3809_380914

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  sides : ℕ
  sides_pos : sides > 0

/-- The number of diagonals from any vertex in a polygon. -/
def diagonals_from_vertex (p : Polygon) : ℕ := p.sides - 3

/-- Theorem: A polygon with 2 diagonals from any vertex has 5 sides. -/
theorem polygon_with_two_diagonals_has_five_sides (p : Polygon) 
  (h : diagonals_from_vertex p = 2) : p.sides = 5 := by
  sorry

#check polygon_with_two_diagonals_has_five_sides

end NUMINAMATH_CALUDE_polygon_with_two_diagonals_has_five_sides_l3809_380914


namespace NUMINAMATH_CALUDE_square_property_necessary_not_sufficient_l3809_380952

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The property a_{n+1}^2 = a_n * a_{n+2} for all n -/
def HasSquareProperty (a : Sequence) : Prop :=
  ∀ n : ℕ, (a (n + 1))^2 = a n * a (n + 2)

/-- Definition of a geometric sequence -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem: HasSquareProperty is necessary but not sufficient for IsGeometric -/
theorem square_property_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → HasSquareProperty a) ∧
  ¬(∀ a : Sequence, HasSquareProperty a → IsGeometric a) := by
  sorry


end NUMINAMATH_CALUDE_square_property_necessary_not_sufficient_l3809_380952


namespace NUMINAMATH_CALUDE_middle_circle_radius_l3809_380915

/-- A configuration of five circles tangent to each other and two parallel lines -/
structure CircleConfiguration where
  /-- The radii of the five circles, from smallest to largest -/
  radii : Fin 5 → ℝ
  /-- The radii are positive -/
  radii_pos : ∀ i, 0 < radii i
  /-- The radii are in ascending order -/
  radii_ascending : ∀ i j, i < j → radii i < radii j

/-- The theorem stating that if the smallest and largest radii are 8 and 18, 
    then the middle radius is 12 -/
theorem middle_circle_radius (c : CircleConfiguration)
    (h_smallest : c.radii 0 = 8)
    (h_largest : c.radii 4 = 18) :
    c.radii 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_middle_circle_radius_l3809_380915


namespace NUMINAMATH_CALUDE_pin_combinations_l3809_380938

/-- The number of distinct digits in the PIN -/
def n : ℕ := 4

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem: The number of permutations of 4 distinct objects is 24 -/
theorem pin_combinations : permutations n = 24 := by
  sorry

end NUMINAMATH_CALUDE_pin_combinations_l3809_380938


namespace NUMINAMATH_CALUDE_calen_current_pencils_l3809_380907

-- Define the number of pencils for each person
def candy_pencils : ℕ := 9
def caleb_pencils : ℕ := 2 * candy_pencils - 3
def calen_original_pencils : ℕ := caleb_pencils + 5
def calen_lost_pencils : ℕ := 10

-- Theorem to prove
theorem calen_current_pencils :
  calen_original_pencils - calen_lost_pencils = 10 := by
  sorry

end NUMINAMATH_CALUDE_calen_current_pencils_l3809_380907


namespace NUMINAMATH_CALUDE_sum_and_product_of_primes_l3809_380937

theorem sum_and_product_of_primes :
  ∀ p q : ℕ, Prime p → Prime q → p + q = 85 → p * q = 166 := by
sorry

end NUMINAMATH_CALUDE_sum_and_product_of_primes_l3809_380937


namespace NUMINAMATH_CALUDE_all_expressions_correct_l3809_380982

theorem all_expressions_correct (x y : ℝ) (h : x / y = 5 / 6) :
  (x + 2*y) / y = 17 / 6 ∧
  (2*x) / (3*y) = 5 / 9 ∧
  (y - x) / (2*y) = 1 / 12 ∧
  (x + y) / (2*y) = 11 / 12 ∧
  x / (y + x) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_all_expressions_correct_l3809_380982


namespace NUMINAMATH_CALUDE_max_valid_triples_l3809_380936

/-- A function that checks if four positive integers can be arranged in a circle
    with all neighbors being coprime -/
def can_arrange_coprime (a₁ a₂ a₃ a₄ : ℕ+) : Prop :=
  (Nat.gcd a₁.val a₂.val = 1 ∧ Nat.gcd a₂.val a₃.val = 1 ∧ Nat.gcd a₃.val a₄.val = 1 ∧ Nat.gcd a₄.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₂.val = 1 ∧ Nat.gcd a₂.val a₄.val = 1 ∧ Nat.gcd a₄.val a₃.val = 1 ∧ Nat.gcd a₃.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₃.val = 1 ∧ Nat.gcd a₃.val a₂.val = 1 ∧ Nat.gcd a₂.val a₄.val = 1 ∧ Nat.gcd a₄.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₃.val = 1 ∧ Nat.gcd a₃.val a₄.val = 1 ∧ Nat.gcd a₄.val a₂.val = 1 ∧ Nat.gcd a₂.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₄.val = 1 ∧ Nat.gcd a₄.val a₂.val = 1 ∧ Nat.gcd a₂.val a₃.val = 1 ∧ Nat.gcd a₃.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₄.val = 1 ∧ Nat.gcd a₄.val a₃.val = 1 ∧ Nat.gcd a₃.val a₂.val = 1 ∧ Nat.gcd a₂.val a₁.val = 1)

/-- A function that counts the number of valid triples (i,j,k) where (gcd(aᵢ,a_j))² | a_k -/
def count_valid_triples (a₁ a₂ a₃ a₄ : ℕ+) : ℕ :=
  let check (i j k : ℕ+) : Bool :=
    i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ (Nat.gcd i.val j.val)^2 ∣ k.val
  (if check a₁ a₂ a₃ then 1 else 0) +
  (if check a₁ a₂ a₄ then 1 else 0) +
  (if check a₁ a₃ a₂ then 1 else 0) +
  (if check a₁ a₃ a₄ then 1 else 0) +
  (if check a₁ a₄ a₂ then 1 else 0) +
  (if check a₁ a₄ a₃ then 1 else 0) +
  (if check a₂ a₃ a₁ then 1 else 0) +
  (if check a₂ a₃ a₄ then 1 else 0) +
  (if check a₂ a₄ a₁ then 1 else 0) +
  (if check a₂ a₄ a₃ then 1 else 0) +
  (if check a₃ a₄ a₁ then 1 else 0) +
  (if check a₃ a₄ a₂ then 1 else 0)

theorem max_valid_triples (a₁ a₂ a₃ a₄ : ℕ+) :
  ¬(can_arrange_coprime a₁ a₂ a₃ a₄) → count_valid_triples a₁ a₂ a₃ a₄ ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_valid_triples_l3809_380936


namespace NUMINAMATH_CALUDE_smallest_marble_count_l3809_380957

theorem smallest_marble_count : ∃ (M : ℕ), 
  M > 1 ∧
  M % 5 = 1 ∧
  M % 7 = 1 ∧
  M % 11 = 1 ∧
  M % 4 = 2 ∧
  (∀ (N : ℕ), N > 1 ∧ N % 5 = 1 ∧ N % 7 = 1 ∧ N % 11 = 1 ∧ N % 4 = 2 → M ≤ N) ∧
  M = 386 := by
  sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l3809_380957


namespace NUMINAMATH_CALUDE_paint_time_per_room_l3809_380941

theorem paint_time_per_room 
  (total_rooms : ℕ) 
  (painted_rooms : ℕ) 
  (remaining_time : ℕ) 
  (h1 : total_rooms = 11) 
  (h2 : painted_rooms = 2) 
  (h3 : remaining_time = 63) : 
  remaining_time / (total_rooms - painted_rooms) = 7 := by
  sorry

end NUMINAMATH_CALUDE_paint_time_per_room_l3809_380941


namespace NUMINAMATH_CALUDE_problem_proof_l3809_380964

theorem problem_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + x*y - 3 = 0) :
  (0 < x*y ∧ x*y ≤ 1) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b + a*b - 3 = 0 → x + 2*y ≤ a + 2*b) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b + a*b - 3 = 0 ∧ a + 2*b = 4*Real.sqrt 2 - 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_proof_l3809_380964


namespace NUMINAMATH_CALUDE_complex_sum_problem_l3809_380948

theorem complex_sum_problem (p r s t u : ℝ) : 
  (∃ q : ℝ, q = 4 ∧ 
   t = -p - r ∧ 
   Complex.I * (q + s + u) = Complex.I * 3) →
  s + u = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l3809_380948


namespace NUMINAMATH_CALUDE_row_col_product_equality_l3809_380901

theorem row_col_product_equality 
  (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h_row_col_sum : 
    a₁ + a₂ + a₃ = b₁ + b₂ + b₃ ∧ 
    b₁ + b₂ + b₃ = c₁ + c₂ + c₃ ∧ 
    c₁ + c₂ + c₃ = a₁ + b₁ + c₁ ∧ 
    a₁ + b₁ + c₁ = a₂ + b₂ + c₂ ∧ 
    a₂ + b₂ + c₂ = a₃ + b₃ + c₃) : 
  a₁*b₁*c₁ + a₂*b₂*c₂ + a₃*b₃*c₃ = a₁*a₂*a₃ + b₁*b₂*b₃ + c₁*c₂*c₃ :=
by
  sorry

end NUMINAMATH_CALUDE_row_col_product_equality_l3809_380901


namespace NUMINAMATH_CALUDE_translate_linear_function_l3809_380921

/-- A linear function in the Cartesian coordinate system. -/
def LinearFunction (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

/-- Vertical translation of a function. -/
def VerticalTranslate (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x ↦ f x - k

theorem translate_linear_function :
  let f := LinearFunction 5 0
  let g := VerticalTranslate f 5
  ∀ x, g x = 5 * x - 5 := by sorry

end NUMINAMATH_CALUDE_translate_linear_function_l3809_380921


namespace NUMINAMATH_CALUDE_train_crossing_time_l3809_380910

/-- A train crosses a platform and an electric pole -/
theorem train_crossing_time (train_speed : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed = 10 →
  platform_length = 320 →
  platform_crossing_time = 44 →
  (platform_length + train_speed * platform_crossing_time) / train_speed = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3809_380910


namespace NUMINAMATH_CALUDE_f_derivative_at_pi_third_l3809_380943

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x

theorem f_derivative_at_pi_third : 
  (deriv f) (π / 3) = 0 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_pi_third_l3809_380943


namespace NUMINAMATH_CALUDE_special_gp_ratio_is_one_l3809_380977

/-- A geometric progression with positive terms where any term is the product of the next two -/
structure SpecialGP where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  term_product : ∀ n : ℕ, a * r^n = (a * r^(n+1)) * (a * r^(n+2))

/-- The common ratio of a SpecialGP is 1 -/
theorem special_gp_ratio_is_one (gp : SpecialGP) : gp.r = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_gp_ratio_is_one_l3809_380977


namespace NUMINAMATH_CALUDE_log_difference_sqrt_l3809_380973

theorem log_difference_sqrt (x : ℝ) : 
  x = Real.sqrt (Real.log 8 / Real.log 4 - Real.log 16 / Real.log 8) → x = 1 / Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_sqrt_l3809_380973
