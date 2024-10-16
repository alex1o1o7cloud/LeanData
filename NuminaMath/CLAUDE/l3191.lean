import Mathlib

namespace NUMINAMATH_CALUDE_term_degree_le_poly_degree_l3191_319192

/-- A polynomial of degree 6 -/
def Polynomial6 : Type := ℕ → ℚ

/-- The degree of a polynomial -/
def degree (p : Polynomial6) : ℕ := 6

/-- A term of a polynomial -/
def Term : Type := ℕ × ℚ

/-- The degree of a term -/
def termDegree (t : Term) : ℕ := t.1

theorem term_degree_le_poly_degree (p : Polynomial6) (t : Term) : 
  termDegree t ≤ degree p := by sorry

end NUMINAMATH_CALUDE_term_degree_le_poly_degree_l3191_319192


namespace NUMINAMATH_CALUDE_total_germs_count_l3191_319194

/-- The number of petri dishes in the biology lab -/
def num_dishes : ℕ := 10800

/-- The number of germs in a single petri dish -/
def germs_per_dish : ℕ := 500

/-- The total number of germs in the biology lab -/
def total_germs : ℕ := num_dishes * germs_per_dish

/-- Theorem stating that the total number of germs is 5,400,000 -/
theorem total_germs_count : total_germs = 5400000 := by
  sorry

end NUMINAMATH_CALUDE_total_germs_count_l3191_319194


namespace NUMINAMATH_CALUDE_james_car_rental_days_l3191_319110

/-- Calculates the number of days James rents his car per week -/
def days_rented_per_week (hourly_rate : ℕ) (hours_per_day : ℕ) (weekly_earnings : ℕ) : ℕ :=
  weekly_earnings / (hourly_rate * hours_per_day)

/-- Theorem stating that James rents his car for 4 days per week -/
theorem james_car_rental_days :
  days_rented_per_week 20 8 640 = 4 := by
  sorry

end NUMINAMATH_CALUDE_james_car_rental_days_l3191_319110


namespace NUMINAMATH_CALUDE_arctan_sum_three_seven_l3191_319123

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_seven_l3191_319123


namespace NUMINAMATH_CALUDE_additional_grazing_area_l3191_319117

/-- The additional grassy ground area a calf can graze after increasing rope length -/
theorem additional_grazing_area (initial_length new_length obstacle_length obstacle_width : ℝ) 
  (h1 : initial_length = 12)
  (h2 : new_length = 18)
  (h3 : obstacle_length = 4)
  (h4 : obstacle_width = 3) :
  (π * new_length^2 - obstacle_length * obstacle_width) - π * initial_length^2 = 180 * π - 12 :=
by sorry

end NUMINAMATH_CALUDE_additional_grazing_area_l3191_319117


namespace NUMINAMATH_CALUDE_add_3_15_base6_l3191_319133

/-- Represents a number in base 6 --/
def Base6 := Nat

/-- Converts a base 6 number to its decimal representation --/
def toDecimal (n : Base6) : Nat :=
  sorry

/-- Converts a decimal number to its base 6 representation --/
def toBase6 (n : Nat) : Base6 :=
  sorry

/-- Addition in base 6 --/
def addBase6 (a b : Base6) : Base6 :=
  toBase6 (toDecimal a + toDecimal b)

theorem add_3_15_base6 :
  addBase6 (toBase6 3) (toBase6 15) = toBase6 22 := by
  sorry

end NUMINAMATH_CALUDE_add_3_15_base6_l3191_319133


namespace NUMINAMATH_CALUDE_triangle_properties_l3191_319182

open Real

theorem triangle_properties 
  (a b c A B C : ℝ) 
  (h1 : sin A / (sin B + sin C) = 1 - (a - b) / (a - c))
  (h2 : b = Real.sqrt 3)
  (h3 : 0 < A ∧ A < 2 * π / 3) :
  ∃ (area : ℝ) (range_lower range_upper : ℝ),
    (∀ perimeter, perimeter ≤ 3 * Real.sqrt 3 → 
      area * 2 ≤ perimeter * Real.sqrt (perimeter * (perimeter - 2*a) * (perimeter - 2*b) * (perimeter - 2*c)) / 4) ∧
    (area = 3 * Real.sqrt 3 / 4) ∧
    (∀ m_dot_n : ℝ, 
      (∃ A', 0 < A' ∧ A' < 2 * π / 3 ∧ 
        m_dot_n = 6 * sin A' * cos B + cos (2 * A')) → 
      (range_lower < m_dot_n ∧ m_dot_n ≤ range_upper)) ∧
    (range_lower = 1 ∧ range_upper = 17/8) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3191_319182


namespace NUMINAMATH_CALUDE_monk_problem_l3191_319115

theorem monk_problem (total_mantou total_monks : ℕ) 
  (big_monk_consumption small_monk_consumption : ℚ) :
  total_mantou = 100 →
  total_monks = 100 →
  big_monk_consumption = 1 →
  small_monk_consumption = 1/3 →
  ∃ (big_monks small_monks : ℕ),
    big_monks + small_monks = total_monks ∧
    big_monks * big_monk_consumption + small_monks * small_monk_consumption = total_mantou ∧
    big_monks = 25 ∧
    small_monks = 75 := by
  sorry

end NUMINAMATH_CALUDE_monk_problem_l3191_319115


namespace NUMINAMATH_CALUDE_sum_in_special_base_l3191_319144

theorem sum_in_special_base (b : ℕ) (h : b > 1) :
  (b + 3) * (b + 4) * (b + 5) = 2 * b^3 + 3 * b^2 + 2 * b + 5 →
  (b + 3) + (b + 4) + (b + 5) = 4 * b + 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_in_special_base_l3191_319144


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3191_319174

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 100) : 
  a * b = -3 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3191_319174


namespace NUMINAMATH_CALUDE_fourth_root_of_33177600_l3191_319148

theorem fourth_root_of_33177600 : (33177600 : ℝ) ^ (1/4 : ℝ) = 576 := by sorry

end NUMINAMATH_CALUDE_fourth_root_of_33177600_l3191_319148


namespace NUMINAMATH_CALUDE_total_balloons_proof_l3191_319153

def sam_initial_balloons : ℝ := 6.0
def sam_given_balloons : ℝ := 5.0
def mary_balloons : ℝ := 7.0

theorem total_balloons_proof :
  sam_initial_balloons - sam_given_balloons + mary_balloons = 8 :=
by sorry

end NUMINAMATH_CALUDE_total_balloons_proof_l3191_319153


namespace NUMINAMATH_CALUDE_min_value_a_a_range_l3191_319103

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |x - 3|

-- Theorem 1
theorem min_value_a (a : ℝ) :
  (∀ x, f x a ≥ 5) ∧ (∃ x, f x a = 5) → a = 2 ∨ a = -8 := by
  sorry

-- Theorem 2
theorem a_range (a : ℝ) :
  (∀ x, -1 ≤ x → x ≤ 0 → f x a ≤ |x - 4|) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_a_range_l3191_319103


namespace NUMINAMATH_CALUDE_coordinate_axis_angles_characterization_l3191_319134

-- Define the set of angles whose terminal sides lie on the coordinate axes
def CoordinateAxisAngles : Set ℝ :=
  {α | ∃ n : ℤ, α = n * Real.pi / 2}

-- Theorem stating that the set of angles whose terminal sides lie on the coordinate axes
-- is equal to the set {α | α = nπ/2, n ∈ ℤ}
theorem coordinate_axis_angles_characterization :
  CoordinateAxisAngles = {α | ∃ n : ℤ, α = n * Real.pi / 2} := by
  sorry

end NUMINAMATH_CALUDE_coordinate_axis_angles_characterization_l3191_319134


namespace NUMINAMATH_CALUDE_cheesecakes_sold_l3191_319136

theorem cheesecakes_sold (display : ℕ) (fridge : ℕ) (left : ℕ) : 
  display + fridge - left = display - (display + fridge - left - fridge) :=
by sorry

#check cheesecakes_sold 10 15 18

end NUMINAMATH_CALUDE_cheesecakes_sold_l3191_319136


namespace NUMINAMATH_CALUDE_perimeter_difference_l3191_319170

/-- Represents a figure made of unit squares -/
structure UnitSquareFigure where
  perimeter : ℕ

/-- The first figure in the problem -/
def figure1 : UnitSquareFigure :=
  { perimeter := 24 }

/-- The second figure in the problem -/
def figure2 : UnitSquareFigure :=
  { perimeter := 33 }

/-- The theorem stating the difference between the perimeters of the two figures -/
theorem perimeter_difference :
  (figure2.perimeter - figure1.perimeter : ℤ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_l3191_319170


namespace NUMINAMATH_CALUDE_ellipse_properties_l3191_319163

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the line l -/
def line_l (x : ℝ) : Prop := x = -3

theorem ellipse_properties :
  ∀ a b : ℝ,
  a > b ∧ b > 0 ∧
  2 * Real.sqrt 3 = 2 * b ∧
  Real.sqrt 2 / 2 = Real.sqrt (a^2 - b^2) / a →
  (∀ x y : ℝ, ellipse_C x y a b ↔ x^2 / 6 + y^2 / 3 = 1) ∧
  (∃ min_value : ℝ,
    min_value = 0 ∧
    ∀ x y : ℝ,
    ellipse_C x y a b ∧ y > 0 →
    (x + 3)^2 - y^2 ≥ min_value) ∧
  (∃ m : ℝ,
    m = 9/8 ∧
    ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_C x₁ y₁ a b ∧
    ellipse_C x₂ y₂ a b ∧
    y₁ = 1/4 * x₁ + m ∧
    y₂ = 1/4 * x₂ + m ∧
    ∃ xg yg : ℝ,
    ellipse_C xg yg a b ∧
    xg - (-3) = x₂ - x₁ ∧
    yg - 3 = y₂ - y₁) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3191_319163


namespace NUMINAMATH_CALUDE_larger_number_in_ratio_l3191_319109

theorem larger_number_in_ratio (a b : ℚ) : 
  a / b = 8 / 3 → a + b = 143 → max a b = 104 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_in_ratio_l3191_319109


namespace NUMINAMATH_CALUDE_probability_divisible_by_five_l3191_319171

/- Define the spinner outcomes -/
def spinner : Finset ℕ := {1, 2, 4, 5}

/- Define a function to check if a number is divisible by 5 -/
def divisible_by_five (n : ℕ) : Bool :=
  n % 5 = 0

/- Define a function to create a three-digit number from three spins -/
def make_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

/- Main theorem -/
theorem probability_divisible_by_five :
  (Finset.filter (fun n => divisible_by_five (make_number n.1 n.2.1 n.2.2))
    (spinner.product (spinner.product spinner))).card /
  (spinner.product (spinner.product spinner)).card = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_divisible_by_five_l3191_319171


namespace NUMINAMATH_CALUDE_pig_farm_fence_length_l3191_319125

/-- Represents a rectangular pig farm with specific dimensions -/
structure PigFarm where
  /-- Length of the shorter sides of the rectangle -/
  short_side : ℝ
  /-- Ensures the short side is positive -/
  short_side_pos : short_side > 0

/-- Calculates the area of the pig farm -/
def PigFarm.area (farm : PigFarm) : ℝ :=
  2 * farm.short_side * farm.short_side

/-- Calculates the total fence length of the pig farm -/
def PigFarm.fence_length (farm : PigFarm) : ℝ :=
  4 * farm.short_side

/-- Theorem stating the fence length for a pig farm with area 1250 sq ft -/
theorem pig_farm_fence_length :
  ∃ (farm : PigFarm), farm.area = 1250 ∧ farm.fence_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_pig_farm_fence_length_l3191_319125


namespace NUMINAMATH_CALUDE_carbon_monoxide_weight_l3191_319124

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (c o : ℝ) : ℝ := c + o

/-- Theorem: The molecular weight of Carbon monoxide (CO) is 28.01 g/mol -/
theorem carbon_monoxide_weight : molecular_weight carbon_weight oxygen_weight = 28.01 := by
  sorry

end NUMINAMATH_CALUDE_carbon_monoxide_weight_l3191_319124


namespace NUMINAMATH_CALUDE_kara_book_count_l3191_319107

/-- The number of books read by each person in the Book Tournament --/
structure BookCount where
  candice : ℕ
  amanda : ℕ
  kara : ℕ
  patricia : ℕ

/-- The conditions of the Book Tournament --/
def BookTournament (bc : BookCount) : Prop :=
  bc.candice = 18 ∧
  bc.candice = 3 * bc.amanda ∧
  bc.kara = bc.amanda / 2

theorem kara_book_count (bc : BookCount) (h : BookTournament bc) : bc.kara = 3 := by
  sorry

end NUMINAMATH_CALUDE_kara_book_count_l3191_319107


namespace NUMINAMATH_CALUDE_parallel_planes_from_skew_parallel_lines_l3191_319187

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the skew relation for lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_from_skew_parallel_lines 
  (m n : Line) (α β : Plane) :
  skew m n →
  parallel_line_plane m α →
  parallel_line_plane n α →
  parallel_line_plane m β →
  parallel_line_plane n β →
  parallel_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_skew_parallel_lines_l3191_319187


namespace NUMINAMATH_CALUDE_house_painting_time_l3191_319197

theorem house_painting_time (a b c : ℚ) 
  (hab : a + b = 1/3)
  (hbc : b + c = 1/4)
  (hca : c + a = 1/6) :
  1 / (a + b + c) = 8/3 := by sorry

end NUMINAMATH_CALUDE_house_painting_time_l3191_319197


namespace NUMINAMATH_CALUDE_earl_floor_problem_l3191_319178

theorem earl_floor_problem (total_floors : ℕ) (initial_floor : ℕ) (first_up : ℕ) (second_up : ℕ) (floors_from_top : ℕ) (floors_down : ℕ) :
  total_floors = 20 →
  initial_floor = 1 →
  first_up = 5 →
  second_up = 7 →
  floors_from_top = 9 →
  initial_floor + first_up - floors_down + second_up = total_floors - floors_from_top →
  floors_down = 2 := by
sorry

end NUMINAMATH_CALUDE_earl_floor_problem_l3191_319178


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3191_319147

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, (x + 1) * (x - 3) < 0 → x > -1) ∧
  ¬(∀ x : ℝ, x > -1 → (x + 1) * (x - 3) < 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3191_319147


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_root_two_implies_k_value_l3191_319150

/-- The quadratic equation k^2*x^2 + 2*(k-1)*x + 1 = 0 -/
def quadratic_equation (k x : ℝ) : Prop :=
  k^2 * x^2 + 2*(k-1)*x + 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ :=
  4*(k-1)^2 - 4*k^2

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x ∧ quadratic_equation k y) ↔
  (k < 1/2 ∧ k ≠ 0) :=
sorry

theorem root_two_implies_k_value :
  ∀ k : ℝ, quadratic_equation k 2 → k = -3/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_root_two_implies_k_value_l3191_319150


namespace NUMINAMATH_CALUDE_function_properties_l3191_319165

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem function_properties (a b : ℝ) 
  (h : (a - 1)^2 - 4*b < 0) : 
  (∀ x, f a b x > x) ∧ 
  (∀ x, f a b (f a b x) > x) ∧ 
  (a + b > 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3191_319165


namespace NUMINAMATH_CALUDE_square_sum_from_diff_and_product_l3191_319131

theorem square_sum_from_diff_and_product (p q : ℝ) 
  (h1 : p - q = 4) 
  (h2 : p * q = -2) : 
  p^2 + q^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_diff_and_product_l3191_319131


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3191_319119

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3191_319119


namespace NUMINAMATH_CALUDE_not_prime_qt_plus_q_plus_t_l3191_319140

theorem not_prime_qt_plus_q_plus_t (q t : ℕ+) (h : q > 1 ∨ t > 1) : 
  ¬ Nat.Prime (q * t + q + t) := by
sorry

end NUMINAMATH_CALUDE_not_prime_qt_plus_q_plus_t_l3191_319140


namespace NUMINAMATH_CALUDE_coefficient_x_squared_sum_binomials_l3191_319145

theorem coefficient_x_squared_sum_binomials : 
  let f (n : ℕ) := (1 + X : Polynomial ℚ)^n
  let sum := (f 4) + (f 5) + (f 6) + (f 7) + (f 8) + (f 9)
  (sum.coeff 2 : ℚ) = 116 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_sum_binomials_l3191_319145


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3191_319155

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - 2*y + 2*Real.log 2 - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3191_319155


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3191_319158

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3191_319158


namespace NUMINAMATH_CALUDE_arc_length_of_sector_l3191_319173

/-- The arc length of a sector with radius π cm and central angle 120° is 2π²/3 cm. -/
theorem arc_length_of_sector (r : Real) (θ : Real) : 
  r = π → θ = 120 * π / 180 → r * θ = 2 * π^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_of_sector_l3191_319173


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l3191_319129

/-- The diameter of the inscribed circle in a triangle with side lengths 13, 14, and 15 is 8 -/
theorem inscribed_circle_diameter (DE DF EF : ℝ) (h1 : DE = 13) (h2 : DF = 14) (h3 : EF = 15) :
  let s := (DE + DF + EF) / 2
  let A := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  2 * A / s = 8 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l3191_319129


namespace NUMINAMATH_CALUDE_adams_change_l3191_319164

/-- Given that Adam has $5 and an airplane costs $4.28, prove that he will receive $0.72 in change. -/
theorem adams_change (adams_money : ℚ) (airplane_cost : ℚ) (h1 : adams_money = 5) (h2 : airplane_cost = 4.28) :
  adams_money - airplane_cost = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_adams_change_l3191_319164


namespace NUMINAMATH_CALUDE_hotel_flat_fee_calculation_l3191_319175

/-- A hotel charges a flat fee for the first night and a fixed amount for subsequent nights. -/
structure HotelPricing where
  flatFee : ℝ
  subsequentNightFee : ℝ

/-- Calculate the total cost for a given number of nights -/
def totalCost (pricing : HotelPricing) (nights : ℕ) : ℝ :=
  pricing.flatFee + pricing.subsequentNightFee * (nights - 1)

theorem hotel_flat_fee_calculation (pricing : HotelPricing) :
  totalCost pricing 4 = 185 ∧ totalCost pricing 8 = 350 → pricing.flatFee = 61.25 := by
  sorry

end NUMINAMATH_CALUDE_hotel_flat_fee_calculation_l3191_319175


namespace NUMINAMATH_CALUDE_B_necessary_not_sufficient_l3191_319139

def A (x : ℝ) : Prop := 0 < x ∧ x < 5

def B (x : ℝ) : Prop := |x - 2| < 3

theorem B_necessary_not_sufficient :
  (∀ x, A x → B x) ∧ (∃ x, B x ∧ ¬A x) := by
  sorry

end NUMINAMATH_CALUDE_B_necessary_not_sufficient_l3191_319139


namespace NUMINAMATH_CALUDE_line_property_l3191_319122

/-- Given two points on a line, prove that m - 2b equals 21 --/
theorem line_property (x₁ y₁ x₂ y₂ m b : ℝ) 
  (h₁ : y₁ = m * x₁ + b) 
  (h₂ : y₂ = m * x₂ + b) 
  (h₃ : x₁ = 2) 
  (h₄ : y₁ = -3) 
  (h₅ : x₂ = 6) 
  (h₆ : y₂ = 9) : 
  m - 2 * b = 21 := by
  sorry

#check line_property

end NUMINAMATH_CALUDE_line_property_l3191_319122


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l3191_319162

theorem greatest_q_minus_r : ∃ (q r : ℕ+), 
  975 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 975 = 23 * q' + r' → (q - r : ℤ) ≥ (q' - r' : ℤ) ∧
  (q - r : ℤ) = 33 :=
sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l3191_319162


namespace NUMINAMATH_CALUDE_first_class_average_mark_l3191_319113

theorem first_class_average_mark (x : ℝ) : 
  (25 * x + 30 * 60) / 55 = 50.90909090909091 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_first_class_average_mark_l3191_319113


namespace NUMINAMATH_CALUDE_second_number_is_255_l3191_319185

def first_set (x : ℝ) : List ℝ := [28, x, 42, 78, 104]
def second_set (x y : ℝ) : List ℝ := [128, y, 511, 1023, x]

theorem second_number_is_255 
  (x : ℝ)
  (h1 : (first_set x).sum / (first_set x).length = 90)
  (h2 : ∃ y, (second_set x y).sum / (second_set x y).length = 423) :
  ∃ y, (second_set x y).sum / (second_set x y).length = 423 ∧ y = 255 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_255_l3191_319185


namespace NUMINAMATH_CALUDE_money_relation_l3191_319193

theorem money_relation (a b : ℝ) 
  (h1 : 8 * a - b = 98) 
  (h2 : 2 * a + b > 36) : 
  a > 13.4 ∧ b > 9.2 := by
sorry

end NUMINAMATH_CALUDE_money_relation_l3191_319193


namespace NUMINAMATH_CALUDE_total_packs_is_108_l3191_319186

/-- The number of people buying baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person bought -/
def cards_per_person : ℕ := 540

/-- The number of cards in each pack -/
def cards_per_pack : ℕ := 20

/-- Theorem: The total number of packs for all people is 108 -/
theorem total_packs_is_108 : 
  (num_people * cards_per_person) / cards_per_pack = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_is_108_l3191_319186


namespace NUMINAMATH_CALUDE_zongzi_price_calculation_l3191_319141

theorem zongzi_price_calculation (pork_total red_bean_total : ℕ) 
  (h1 : pork_total = 8000)
  (h2 : red_bean_total = 6000)
  (h3 : ∃ (n : ℕ), n ≠ 0 ∧ pork_total = n * 40 ∧ red_bean_total = n * 30) :
  ∃ (pork_price red_bean_price : ℕ),
    pork_price = 40 ∧
    red_bean_price = 30 ∧
    pork_price = red_bean_price + 10 ∧
    pork_total = red_bean_total + 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_zongzi_price_calculation_l3191_319141


namespace NUMINAMATH_CALUDE_distinct_plants_count_l3191_319143

/-- Represents a flower bed -/
structure FlowerBed where
  plants : Finset ℕ

/-- The total number of distinct plants in three intersecting flower beds -/
def total_distinct_plants (X Y Z : FlowerBed) : ℕ :=
  (X.plants ∪ Y.plants ∪ Z.plants).card

/-- The theorem stating the total number of distinct plants in the given scenario -/
theorem distinct_plants_count (X Y Z : FlowerBed)
  (hX : X.plants.card = 600)
  (hY : Y.plants.card = 500)
  (hZ : Z.plants.card = 400)
  (hXY : (X.plants ∩ Y.plants).card = 100)
  (hYZ : (Y.plants ∩ Z.plants).card = 80)
  (hXZ : (X.plants ∩ Z.plants).card = 120)
  (hXYZ : (X.plants ∩ Y.plants ∩ Z.plants).card = 30) :
  total_distinct_plants X Y Z = 1230 := by
  sorry


end NUMINAMATH_CALUDE_distinct_plants_count_l3191_319143


namespace NUMINAMATH_CALUDE_smallest_a_for_minimum_l3191_319105

noncomputable def f (a x : ℝ) : ℝ := -Real.log x / x + Real.exp (a * x - 1)

theorem smallest_a_for_minimum (a : ℝ) : 
  (∀ x > 0, f a x ≥ a) ∧ (∃ x > 0, f a x = a) ↔ a = -Real.exp (-2) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_minimum_l3191_319105


namespace NUMINAMATH_CALUDE_triangle_problem_l3191_319101

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 1) 
  (h2 : t.b = 2 * Real.sqrt 3) 
  (h3 : t.B - t.A = π / 6) 
  (h4 : t.A + t.B + t.C = π) -- Triangle angle sum
  (h5 : t.a / Real.sin t.A = t.b / Real.sin t.B) -- Law of sines
  (h6 : t.b / Real.sin t.B = t.c / Real.sin t.C) -- Law of sines
  : Real.sin t.A = Real.sqrt 7 / 14 ∧ t.c = (11 / 7) * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3191_319101


namespace NUMINAMATH_CALUDE_average_tv_sets_is_48_l3191_319181

/-- The average number of TV sets in 5 electronic shops -/
def average_tv_sets : ℚ :=
  let shops := 5
  let tv_sets := [20, 30, 60, 80, 50]
  (tv_sets.sum : ℚ) / shops

/-- Theorem: The average number of TV sets in the 5 electronic shops is 48 -/
theorem average_tv_sets_is_48 : average_tv_sets = 48 := by
  sorry

end NUMINAMATH_CALUDE_average_tv_sets_is_48_l3191_319181


namespace NUMINAMATH_CALUDE_monotonicity_f_range_of_a_l3191_319198

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -a * x^2 + Real.log x

-- State the theorems
theorem monotonicity_f (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a > 0 → ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → x₂ < 1 / Real.sqrt (2 * a) → f a x₁ < f a x₂) ∧
  (a > 0 → ∀ x₁ x₂, 1 / Real.sqrt (2 * a) < x₁ → x₁ < x₂ → f a x₁ > f a x₂) :=
sorry

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, x > 1 ∧ f a x > -a) ↔ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_f_range_of_a_l3191_319198


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l3191_319157

theorem existence_of_special_sequence :
  ∃ (a : Fin 100 → ℕ),
    (∀ i j, i < j → a i < a j) ∧
    (∀ i : Fin 98, Nat.gcd (a i) (a (i + 1)) > Nat.gcd (a (i + 1)) (a (i + 2))) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l3191_319157


namespace NUMINAMATH_CALUDE_problem_statements_l3191_319195

theorem problem_statements :
  (∀ (p q : Prop), (p ∧ q) → ¬(¬p)) ∧
  (∃ (x : ℝ), x^2 - x - 1 < 0) ↔ ¬(∀ (x : ℝ), x^2 - x - 1 ≥ 0) ∧
  (∃ (a b : ℝ), (a + b > 0) ∧ ¬(a > 5 ∧ b > -5)) ∧
  (∀ (α : ℝ), α < 0 → ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ < x₂ → x₁^α > x₂^α) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l3191_319195


namespace NUMINAMATH_CALUDE_first_day_next_year_monday_l3191_319152

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : Nat
  is_leap : Bool

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Theorem: If a non-leap year has 53 Sundays, then the first day of the following year is a Monday -/
theorem first_day_next_year_monday 
  (year : Year) 
  (h1 : year.is_leap = false) 
  (h2 : ∃ (sundays : Nat), sundays = 53) : 
  nextDay DayOfWeek.Sunday = DayOfWeek.Monday := by
  sorry

#check first_day_next_year_monday

end NUMINAMATH_CALUDE_first_day_next_year_monday_l3191_319152


namespace NUMINAMATH_CALUDE_limit_of_a_sequence_l3191_319190

def a (n : ℕ) : ℚ := n / (3 * n - 1)

theorem limit_of_a_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 1/3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_a_sequence_l3191_319190


namespace NUMINAMATH_CALUDE_sheep_with_only_fleas_l3191_319199

theorem sheep_with_only_fleas (total : ℕ) (lice : ℕ) (both : ℕ) (only_fleas : ℕ) : 
  total = 2 * lice →
  both = 84 →
  lice = 94 →
  total = only_fleas + (lice - both) + both →
  only_fleas = 94 := by
sorry

end NUMINAMATH_CALUDE_sheep_with_only_fleas_l3191_319199


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3191_319184

/-- The cost of items in dollars -/
structure ItemCost where
  mango : ℝ
  rice : ℝ
  flour : ℝ

/-- Given conditions and the theorem to prove -/
theorem total_cost_calculation (c : ItemCost) 
  (h1 : 10 * c.mango = 24 * c.rice)
  (h2 : 6 * c.flour = 2 * c.rice)
  (h3 : c.flour = 23) :
  4 * c.mango + 3 * c.rice + 5 * c.flour = 984.4 := by
  sorry


end NUMINAMATH_CALUDE_total_cost_calculation_l3191_319184


namespace NUMINAMATH_CALUDE_linda_original_money_l3191_319118

/-- The amount of money Lucy originally had -/
def lucy_original : ℕ := 20

/-- The amount of money Linda originally had -/
def linda_original : ℕ := 10

/-- The amount Lucy would give to Linda -/
def transfer_amount : ℕ := 5

theorem linda_original_money :
  linda_original = 10 :=
by
  have h1 : lucy_original - transfer_amount = linda_original + transfer_amount :=
    sorry
  sorry

end NUMINAMATH_CALUDE_linda_original_money_l3191_319118


namespace NUMINAMATH_CALUDE_bacteria_growth_without_offset_bacteria_growth_non_negative_without_offset_l3191_319108

/-- The number of bacteria after 60 minutes of doubling every minute, starting with 10 bacteria -/
def final_bacteria_count : ℕ := 10240

/-- The initial number of bacteria -/
def initial_bacteria_count : ℕ := 10

/-- The number of minutes in one hour -/
def minutes_in_hour : ℕ := 60

/-- Theorem stating that the initial number of bacteria without offset would be 0 -/
theorem bacteria_growth_without_offset :
  ∀ n : ℤ, (n + initial_bacteria_count) * 2^minutes_in_hour = final_bacteria_count → n = -initial_bacteria_count :=
by sorry

/-- Corollary stating that the non-negative initial number of bacteria without offset is 0 -/
theorem bacteria_growth_non_negative_without_offset :
  ∀ n : ℕ, (n + initial_bacteria_count) * 2^minutes_in_hour = final_bacteria_count → n = 0 :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_without_offset_bacteria_growth_non_negative_without_offset_l3191_319108


namespace NUMINAMATH_CALUDE_parallelepiped_has_twelve_edges_l3191_319120

/-- A parallelepiped is a three-dimensional figure formed by six parallelograms. -/
structure Parallelepiped where
  faces : Fin 6 → Parallelogram
  -- Additional properties ensuring the faces form a valid parallelepiped could be added here

/-- The number of edges in a geometric figure. -/
def numEdges (figure : Type) : ℕ := sorry

/-- Theorem stating that a parallelepiped has 12 edges. -/
theorem parallelepiped_has_twelve_edges (P : Parallelepiped) : numEdges Parallelepiped = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_has_twelve_edges_l3191_319120


namespace NUMINAMATH_CALUDE_seating_theorem_l3191_319137

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (n : ℕ) (no_adjacent_pair : ℕ) (no_adjacent_triple : ℕ) : ℕ :=
  factorial n - (factorial (n - 1) * factorial 2 + factorial (n - 2) * factorial 3) + 
  factorial (n - 3) * factorial 2 * factorial 3

theorem seating_theorem : seating_arrangements 8 2 3 = 25360 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l3191_319137


namespace NUMINAMATH_CALUDE_smaller_number_problem_l3191_319114

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 8) : 
  min x y = 5 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3191_319114


namespace NUMINAMATH_CALUDE_certain_number_exists_l3191_319130

theorem certain_number_exists : ∃ x : ℝ, 
  5.4 * x - (0.6 * 10) / 1.2 = 31.000000000000004 ∧ 
  abs (x - 6.666666666666667) < 1e-15 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l3191_319130


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3191_319100

theorem smallest_four_digit_divisible_by_53 :
  ∃ n : ℕ, n = 1007 ∧ 
  (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ 53 ∣ m → n ≤ m) ∧
  1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3191_319100


namespace NUMINAMATH_CALUDE_inverse_function_property_l3191_319135

theorem inverse_function_property (f : ℝ → ℝ) (h_inv : Function.Injective f) :
  f 1 = 0 → (Function.invFun f 0) + 1 = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_function_property_l3191_319135


namespace NUMINAMATH_CALUDE_parabola_intersection_comparison_l3191_319179

theorem parabola_intersection_comparison (m n a b : ℝ) : 
  (∀ x, m * x^2 + x ≥ 0 → x ≤ a) →  -- A(a,0) is the rightmost intersection of y = mx^2 + x with x-axis
  (∀ x, n * x^2 + x ≥ 0 → x ≤ b) →  -- B(b,0) is the rightmost intersection of y = nx^2 + x with x-axis
  m * a^2 + a = 0 →                  -- A(a,0) is on the parabola y = mx^2 + x
  n * b^2 + b = 0 →                  -- B(b,0) is on the parabola y = nx^2 + x
  a > b →                            -- A is to the right of B
  a > 0 →                            -- A is in the positive half of x-axis
  b > 0 →                            -- B is in the positive half of x-axis
  m > n :=                           -- Conclusion: m > n
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_comparison_l3191_319179


namespace NUMINAMATH_CALUDE_original_number_is_107_l3191_319127

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def increase_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  (hundreds + 3) * 100 + (tens + 2) * 10 + (units + 1)

theorem original_number_is_107 :
  is_three_digit_number 107 ∧ increase_digits 107 = 4 * 107 :=
sorry

end NUMINAMATH_CALUDE_original_number_is_107_l3191_319127


namespace NUMINAMATH_CALUDE_remainder_problem_l3191_319196

theorem remainder_problem : (56 * 67 * 78) % 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3191_319196


namespace NUMINAMATH_CALUDE_cubic_odd_and_increasing_l3191_319116

def f (x : ℝ) : ℝ := x^3

theorem cubic_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_odd_and_increasing_l3191_319116


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l3191_319146

theorem square_plus_inverse_square (a : ℝ) (h : a - (1 / a) = 5) : a^2 + (1 / a^2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l3191_319146


namespace NUMINAMATH_CALUDE_doll_collection_problem_l3191_319151

theorem doll_collection_problem (original_count : ℕ) : 
  (original_count + 2 : ℚ) = original_count * (1 + 1/4) → 
  original_count + 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_doll_collection_problem_l3191_319151


namespace NUMINAMATH_CALUDE_max_value_problem_min_value_problem_l3191_319160

-- Problem 1
theorem max_value_problem (x : ℝ) (h : 0 < x ∧ x < 2) : x * (4 - 2*x) ≤ 2 := by
  sorry

-- Problem 2
theorem min_value_problem (x : ℝ) (h : x > 3/2) : x + 8 / (2*x - 3) ≥ 11/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_min_value_problem_l3191_319160


namespace NUMINAMATH_CALUDE_total_books_in_class_l3191_319189

theorem total_books_in_class (num_tables : ℕ) (books_per_table_ratio : ℚ) : 
  num_tables = 500 →
  books_per_table_ratio = 2 / 5 →
  (num_tables : ℚ) * books_per_table_ratio * num_tables = 100000 :=
by sorry

end NUMINAMATH_CALUDE_total_books_in_class_l3191_319189


namespace NUMINAMATH_CALUDE_reservoir_duration_l3191_319166

theorem reservoir_duration (x y z : ℝ) 
  (h1 : 40 * (y - x) = z)
  (h2 : 40 * (1.1 * y - 1.2 * x) = z)
  : z / (y - 1.2 * x) = 50 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_duration_l3191_319166


namespace NUMINAMATH_CALUDE_inequality_range_real_inequality_range_unit_interval_l3191_319188

-- Define the inequality function
def inequality (k x : ℝ) : Prop :=
  (k * x^2 + k * x + 4) / (x^2 + x + 1) > 1

-- Theorem for the first part of the problem
theorem inequality_range_real : 
  (∀ x : ℝ, inequality k x) ↔ k ∈ Set.Icc 1 13 := by sorry

-- Theorem for the second part of the problem
theorem inequality_range_unit_interval :
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → inequality k x) ↔ k ∈ Set.Ioi (-1/2) := by sorry

end NUMINAMATH_CALUDE_inequality_range_real_inequality_range_unit_interval_l3191_319188


namespace NUMINAMATH_CALUDE_largest_fraction_l3191_319172

theorem largest_fraction : 
  let fractions : List ℚ := [2/5, 1/3, 5/15, 4/10, 7/21]
  ∀ x ∈ fractions, x ≤ 2/5 ∧ x ≤ 4/10 :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l3191_319172


namespace NUMINAMATH_CALUDE_mothers_day_discount_percentage_l3191_319159

/-- Calculates the discount percentage for a Mother's day special at a salon -/
theorem mothers_day_discount_percentage 
  (regular_price : ℝ) 
  (num_services : ℕ) 
  (discounted_total : ℝ) 
  (h1 : regular_price = 40)
  (h2 : num_services = 5)
  (h3 : discounted_total = 150) : 
  (1 - discounted_total / (regular_price * num_services)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_mothers_day_discount_percentage_l3191_319159


namespace NUMINAMATH_CALUDE_delores_purchase_shortage_delores_specific_shortage_l3191_319111

/-- Calculates the amount Delores is short by after attempting to purchase a computer, printer, and table -/
theorem delores_purchase_shortage (initial_amount : ℝ) (computer_price : ℝ) (computer_discount : ℝ)
  (printer_price : ℝ) (printer_tax : ℝ) (table_price_euros : ℝ) (exchange_rate : ℝ) : ℝ :=
  let computer_cost := computer_price * (1 - computer_discount)
  let printer_cost := printer_price * (1 + printer_tax)
  let table_cost := table_price_euros * exchange_rate
  let total_cost := computer_cost + printer_cost + table_cost
  total_cost - initial_amount

/-- Proves that Delores is short by $605 given the specific conditions -/
theorem delores_specific_shortage : 
  delores_purchase_shortage 450 1000 0.3 100 0.15 200 1.2 = 605 := by
  sorry


end NUMINAMATH_CALUDE_delores_purchase_shortage_delores_specific_shortage_l3191_319111


namespace NUMINAMATH_CALUDE_ducks_in_lake_l3191_319154

theorem ducks_in_lake (initial_ducks joining_ducks : ℕ) 
  (h1 : initial_ducks = 13) 
  (h2 : joining_ducks = 20) : 
  initial_ducks + joining_ducks = 33 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_lake_l3191_319154


namespace NUMINAMATH_CALUDE_vaccine_effectiveness_l3191_319176

-- Define the contingency table data
def a : ℕ := 10  -- Injected and Infected
def b : ℕ := 40  -- Injected and Not Infected
def c : ℕ := 20  -- Not Injected and Infected
def d : ℕ := 30  -- Not Injected and Not Infected
def n : ℕ := 100 -- Total number of observations

-- Define the K² formula
def K_squared : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the thresholds
def lower_threshold : ℚ := 3841 / 1000
def upper_threshold : ℚ := 5024 / 1000

-- Theorem statement
theorem vaccine_effectiveness :
  lower_threshold < K_squared ∧ K_squared < upper_threshold :=
sorry

end NUMINAMATH_CALUDE_vaccine_effectiveness_l3191_319176


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_l3191_319180

/-- Represents different sampling methods -/
inductive SamplingMethod
| Random
| Systematic
| Stratified

/-- Represents income levels -/
inductive IncomeLevel
| High
| Middle
| Low

/-- Represents a community with different income levels -/
structure Community where
  total_households : ℕ
  high_income : ℕ
  middle_income : ℕ
  low_income : ℕ

/-- Represents a sampling scenario -/
structure SamplingScenario where
  population_size : ℕ
  sample_size : ℕ
  has_distinct_strata : Bool

/-- Determines the optimal sampling method for a given scenario -/
def optimal_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- The community described in the problem -/
def problem_community : Community :=
  { total_households := 1000
  , high_income := 250
  , middle_income := 560
  , low_income := 190 }

/-- The household study sampling scenario -/
def household_study : SamplingScenario :=
  { population_size := 1000
  , sample_size := 200
  , has_distinct_strata := true }

/-- The discussion forum sampling scenario -/
def discussion_forum : SamplingScenario :=
  { population_size := 20
  , sample_size := 6
  , has_distinct_strata := false }

theorem optimal_sampling_methods :
  optimal_sampling_method household_study = SamplingMethod.Stratified ∧
  optimal_sampling_method discussion_forum = SamplingMethod.Random :=
sorry

end NUMINAMATH_CALUDE_optimal_sampling_methods_l3191_319180


namespace NUMINAMATH_CALUDE_cubic_meter_to_cubic_centimeters_l3191_319149

/-- Prove that one cubic meter is equal to 1,000,000 cubic centimeters -/
theorem cubic_meter_to_cubic_centimeters :
  (∀ m cm : ℕ, m = 100 * cm → m^3 = 1000000 * cm^3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_meter_to_cubic_centimeters_l3191_319149


namespace NUMINAMATH_CALUDE_distinct_primes_dividing_P_l3191_319128

def P : ℕ := (List.range 10).foldl (· * ·) 1

theorem distinct_primes_dividing_P :
  (Finset.filter (fun p => Nat.Prime p ∧ P % p = 0) (Finset.range 11)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_distinct_primes_dividing_P_l3191_319128


namespace NUMINAMATH_CALUDE_simplify_expression_l3191_319191

theorem simplify_expression (x : ℝ) : 120 * x - 55 * x = 65 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3191_319191


namespace NUMINAMATH_CALUDE_roots_equation_relation_l3191_319126

theorem roots_equation_relation (p q a b c : ℝ) : 
  (a^2 + p*a + 1 = 0) → 
  (b^2 + p*b + 1 = 0) → 
  (b^2 + q*b + 2 = 0) → 
  (c^2 + q*c + 2 = 0) → 
  (b-a)*(b-c) = p*q - 6 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_relation_l3191_319126


namespace NUMINAMATH_CALUDE_problem_solution_l3191_319167

def f (x : ℝ) := |2*x - 3| + |2*x + 3|

theorem problem_solution :
  (∃ (M : ℝ),
    (∀ x, f x ≥ M) ∧
    (∃ x, f x = M) ∧
    (M = 6)) ∧
  ({x : ℝ | f x ≤ 8} = {x : ℝ | -2 ≤ x ∧ x ≤ 2}) ∧
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    1/a + 1/(2*b) + 1/(3*c) = 1 →
    a + 2*b + 3*c ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3191_319167


namespace NUMINAMATH_CALUDE_chess_tournament_l3191_319132

theorem chess_tournament (n : ℕ) : 
  (n ≥ 3) →  -- Ensure at least 3 players (2 who withdraw + 1 more)
  ((n - 2) * (n - 3) / 2 + 3 = 81) →  -- Total games equation
  (n = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_l3191_319132


namespace NUMINAMATH_CALUDE_transform_f_to_g_l3191_319106

-- Define the original function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f ((1 - x) / 2) + 1

-- Theorem stating the transformations
theorem transform_f_to_g :
  ∀ x : ℝ,
  -- Reflection across y-axis
  g (-x) = f ((1 + x) / 2) + 1 ∧
  -- Horizontal stretch by factor 2
  g (2 * x) = f ((1 - 2*x) / 2) + 1 ∧
  -- Horizontal shift right by 0.5 units
  g (x - 0.5) = f ((1 - (x - 0.5)) / 2) + 1 ∧
  -- Vertical shift up by 1 unit
  g x = f ((1 - x) / 2) + 1 :=
by sorry

end NUMINAMATH_CALUDE_transform_f_to_g_l3191_319106


namespace NUMINAMATH_CALUDE_triangle_area_symmetric_point_correct_l3191_319169

-- Define the line x-y-2=0
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the symmetric point with respect to a line
def symmetric_point (p q : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  let midpoint := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)
  l midpoint.1 = midpoint.2

-- Theorem for the area of the triangle
theorem triangle_area : 
  ∃ (a b : ℝ), line a 0 ∧ line 0 b ∧ (1/2 * |a| * |b| = 2) :=
sorry

-- Theorem for the symmetric point
theorem symmetric_point_correct :
  symmetric_point (0, 2) (1, 1) (λ x => x + 1) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_symmetric_point_correct_l3191_319169


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l3191_319112

theorem trig_expression_equals_one : 
  (Real.sin (15 * π / 180) * Real.cos (10 * π / 180) + Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) /
  (Real.sin (25 * π / 180) * Real.cos (5 * π / 180) + Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l3191_319112


namespace NUMINAMATH_CALUDE_points_per_game_l3191_319161

theorem points_per_game (total_points : ℕ) (num_games : ℕ) (points_per_game : ℕ) : 
  total_points = 81 → 
  num_games = 3 → 
  total_points = num_games * points_per_game → 
  points_per_game = 27 := by
sorry

end NUMINAMATH_CALUDE_points_per_game_l3191_319161


namespace NUMINAMATH_CALUDE_smallest_x_prime_factorization_l3191_319177

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2
def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3
def is_fifth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m^5

def satisfies_conditions (x : ℕ) : Prop :=
  is_square (2 * x) ∧ is_cube (3 * x) ∧ is_fifth_power (5 * x)

theorem smallest_x_prime_factorization :
  ∃ x : ℕ, 
    satisfies_conditions x ∧ 
    (∀ y : ℕ, satisfies_conditions y → x ≤ y) ∧
    x = 2^15 * 3^20 * 5^24 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_prime_factorization_l3191_319177


namespace NUMINAMATH_CALUDE_evaluate_expression_l3191_319183

theorem evaluate_expression : (10^8) / (2 * (10^5) * (1/2)) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3191_319183


namespace NUMINAMATH_CALUDE_system_solution_l3191_319142

/-- Given a system of equations x + y = 2a and xy(x^2 + y^2) = 2b^4,
    this theorem states the condition for real solutions and
    provides the solutions for specific values of a and b. -/
theorem system_solution (a b : ℝ) (h : b^4 = 9375) :
  (∀ x y : ℝ, x + y = 2*a ∧ x*y*(x^2 + y^2) = 2*b^4 → a^2 ≥ b^2) ∧
  (a = 10 → ∃ x y : ℝ, (x = 15 ∧ y = 5 ∨ x = 5 ∧ y = 15) ∧
                       x + y = 2*a ∧ x*y*(x^2 + y^2) = 2*b^4) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3191_319142


namespace NUMINAMATH_CALUDE_cube_edge_sum_exists_l3191_319138

/-- Represents the edges of a cube --/
def CubeEdges := Fin 12

/-- Represents the faces of a cube --/
def CubeFaces := Fin 6

/-- A function that assigns numbers to the edges of a cube --/
def EdgeAssignment := CubeEdges → Fin 12

/-- A function that returns the edges that make up a face --/
def FaceEdges : CubeFaces → Finset CubeEdges := sorry

/-- The sum of numbers on a face given an edge assignment --/
def FaceSum (assignment : EdgeAssignment) (face : CubeFaces) : ℕ :=
  (FaceEdges face).sum (fun edge => (assignment edge).val + 1)

/-- Theorem stating that there exists an assignment of numbers from 1 to 12
    to the edges of a cube such that the sum of numbers on each face is equal --/
theorem cube_edge_sum_exists : 
  ∃ (assignment : EdgeAssignment), 
    (∀ (face1 face2 : CubeFaces), FaceSum assignment face1 = FaceSum assignment face2) ∧ 
    (∀ (edge1 edge2 : CubeEdges), edge1 ≠ edge2 → assignment edge1 ≠ assignment edge2) := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_sum_exists_l3191_319138


namespace NUMINAMATH_CALUDE_art_project_markers_l3191_319121

/-- Calculates the total number of markers needed for an art project given the distribution of markers among student groups. -/
theorem art_project_markers (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ) 
  (group1_markers_per_student : ℕ) (group2_markers_per_student : ℕ) (group3_markers_per_student : ℕ) :
  total_students = 30 →
  group1_students = 10 →
  group2_students = 15 →
  group1_markers_per_student = 2 →
  group2_markers_per_student = 4 →
  group3_markers_per_student = 6 →
  (group1_students * group1_markers_per_student + 
   group2_students * group2_markers_per_student + 
   (total_students - group1_students - group2_students) * group3_markers_per_student) = 110 :=
by sorry


end NUMINAMATH_CALUDE_art_project_markers_l3191_319121


namespace NUMINAMATH_CALUDE_sum_of_ratios_l3191_319168

theorem sum_of_ratios (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_l3191_319168


namespace NUMINAMATH_CALUDE_greatest_c_for_non_range_greatest_integer_c_l3191_319104

theorem greatest_c_for_non_range (c : ℤ) : 
  (∀ x : ℝ, x^2 + c*x + 20 ≠ 5) ↔ c^2 < 60 :=
sorry

theorem greatest_integer_c : 
  ∃ c : ℤ, c = 7 ∧ (∀ x : ℝ, x^2 + c*x + 20 ≠ 5) ∧ 
  (∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 20 = 5) :=
sorry

end NUMINAMATH_CALUDE_greatest_c_for_non_range_greatest_integer_c_l3191_319104


namespace NUMINAMATH_CALUDE_shopping_tax_rate_l3191_319102

def shopping_problem (clothing_percent : ℝ) (food_percent : ℝ) (other_percent : ℝ) 
                     (other_tax_rate : ℝ) (total_tax_rate : ℝ) : Prop :=
  clothing_percent + food_percent + other_percent = 100 ∧
  clothing_percent = 50 ∧
  food_percent = 20 ∧
  other_percent = 30 ∧
  other_tax_rate = 10 ∧
  total_tax_rate = 5 ∧
  ∃ (clothing_tax_rate : ℝ),
    clothing_tax_rate * clothing_percent + other_tax_rate * other_percent = 
    total_tax_rate * 100 ∧
    clothing_tax_rate = 4

theorem shopping_tax_rate :
  ∀ (clothing_percent food_percent other_percent other_tax_rate total_tax_rate : ℝ),
  shopping_problem clothing_percent food_percent other_percent other_tax_rate total_tax_rate →
  ∃ (clothing_tax_rate : ℝ), clothing_tax_rate = 4 :=
by
  sorry

#check shopping_tax_rate

end NUMINAMATH_CALUDE_shopping_tax_rate_l3191_319102


namespace NUMINAMATH_CALUDE_composite_polynomial_l3191_319156

theorem composite_polynomial (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^3 + 6*n^2 + 12*n + 16 = a * b :=
by sorry

end NUMINAMATH_CALUDE_composite_polynomial_l3191_319156
