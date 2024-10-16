import Mathlib

namespace NUMINAMATH_CALUDE_smallest_positive_root_of_g_l149_14910

open Real

theorem smallest_positive_root_of_g : ∃ s : ℝ,
  s > 0 ∧
  sin s + 3 * cos s + 4 * tan s = 0 ∧
  (∀ x, 0 < x → x < s → sin x + 3 * cos x + 4 * tan x ≠ 0) ∧
  ⌊s⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_root_of_g_l149_14910


namespace NUMINAMATH_CALUDE_quadratic_factorability_l149_14940

theorem quadratic_factorability : ∃ (a b c p q : ℤ),
  (∀ x : ℝ, 3 * (x - 3)^2 = x^2 - 9 ↔ a * x^2 + b * x + c = 0) ∧
  (a * x^2 + b * x + c = (x - p) * (x - q)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_factorability_l149_14940


namespace NUMINAMATH_CALUDE_quadratic_property_l149_14920

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_property (a b c : ℝ) :
  (∀ x, f a b c x ≥ 10) ∧  -- minimum value is 10
  (f a b c (-2) = 10) ∧    -- minimum occurs at x = -2
  (f a b c 0 = 6) →        -- passes through (0, 6)
  f a b c 5 = -39 :=       -- f(5) = -39
by sorry

end NUMINAMATH_CALUDE_quadratic_property_l149_14920


namespace NUMINAMATH_CALUDE_board_number_generation_l149_14949

theorem board_number_generation (target : ℕ := 2020) : ∃ a b : ℕ, 20 * a + 21 * b = target := by
  sorry

end NUMINAMATH_CALUDE_board_number_generation_l149_14949


namespace NUMINAMATH_CALUDE_average_equation_solution_l149_14922

theorem average_equation_solution (x : ℝ) : 
  ((x + 8) + (5 * x + 4) + (2 * x + 7)) / 3 = 3 * x - 10 → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l149_14922


namespace NUMINAMATH_CALUDE_green_blue_difference_after_borders_l149_14974

/-- Represents the number of tiles in a hexagonal figure -/
structure HexagonalFigure where
  blue : ℕ
  green : ℕ

/-- Calculates the number of green tiles added by one border -/
def greenTilesPerBorder : ℕ := 6 * 3

/-- Theorem: The difference between green and blue tiles after adding two borders -/
theorem green_blue_difference_after_borders (initial : HexagonalFigure) :
  let newFigure := HexagonalFigure.mk
    initial.blue
    (initial.green + 2 * greenTilesPerBorder)
  newFigure.green - newFigure.blue = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_after_borders_l149_14974


namespace NUMINAMATH_CALUDE_product_of_primes_summing_to_91_l149_14914

theorem product_of_primes_summing_to_91 (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 91 → p * q = 178 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_summing_to_91_l149_14914


namespace NUMINAMATH_CALUDE_coefficient_not_fifty_l149_14985

theorem coefficient_not_fifty :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 →
  (Nat.choose 5 k) * (2^(5-k)) ≠ 50 := by
sorry

end NUMINAMATH_CALUDE_coefficient_not_fifty_l149_14985


namespace NUMINAMATH_CALUDE_square_perimeter_l149_14966

theorem square_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 675 → perimeter = 60 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l149_14966


namespace NUMINAMATH_CALUDE_sum_reciprocals_inequality_l149_14905

theorem sum_reciprocals_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 + 1 / b^2 ≥ 8) ∧ (1 / a + 1 / b + 1 / (a * b) ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_inequality_l149_14905


namespace NUMINAMATH_CALUDE_parallel_planes_intersection_theorem_l149_14992

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection relation for planes and lines
variable (intersects : Plane → Plane → Line → Prop)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_intersection_theorem 
  (α β γ : Plane) (m n : Line) :
  parallel_planes α β →
  intersects α γ m →
  intersects β γ n →
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_intersection_theorem_l149_14992


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l149_14952

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 25 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let c := 4
  F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ellipse x y

-- Theorem statement
theorem ellipse_triangle_perimeter
  (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ)
  (h_ellipse : point_on_ellipse P)
  (h_foci : foci F₁ F₂) :
  let perimeter := dist P F₁ + dist P F₂ + dist F₁ F₂
  perimeter = 18 :=
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l149_14952


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l149_14980

theorem trigonometric_equation_solutions (x : ℝ) :
  1 + Real.sin x - Real.cos (5 * x) - Real.sin (7 * x) = 2 * (Real.cos (3 * x / 2))^2 ↔
  (∃ k : ℤ, x = π / 8 * (2 * k + 1)) ∨ (∃ n : ℤ, x = π / 4 * (4 * n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l149_14980


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l149_14970

open Set Real

theorem negation_of_universal_proposition (f : ℝ → ℝ) :
  (¬ (∀ x ∈ Ioo 0 (π / 2), f x < 0)) ↔ (∃ x ∈ Ioo 0 (π / 2), f x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l149_14970


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l149_14958

theorem quadratic_inequality_solution_set (x : ℝ) :
  {x | x^2 - 4*x > 44} = {x | x < -4 ∨ x > 11} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l149_14958


namespace NUMINAMATH_CALUDE_bagel_store_expenditure_l149_14979

theorem bagel_store_expenditure (B D : ℝ) : 
  D = B / 2 →
  B = D + 15 →
  B + D = 45 := by sorry

end NUMINAMATH_CALUDE_bagel_store_expenditure_l149_14979


namespace NUMINAMATH_CALUDE_dinner_time_calculation_l149_14942

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, sorry⟩

theorem dinner_time_calculation (start : Time) (commute grocery drycleaning groomer cooking : ℕ) :
  commute = 30 →
  grocery = 30 →
  drycleaning = 10 →
  groomer = 20 →
  cooking = 90 →
  start = ⟨16, 0, sorry⟩ →
  addMinutes start (commute + grocery + drycleaning + groomer + cooking) = ⟨19, 0, sorry⟩ := by
  sorry


end NUMINAMATH_CALUDE_dinner_time_calculation_l149_14942


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l149_14999

theorem complex_equation_solutions :
  ∃! (s : Finset ℂ), s.card = 4 ∧
  (∀ c ∈ s, ∃ u v w : ℂ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    ∀ z : ℂ, (z - u) * (z - v) * (z - w) = (z - c*u) * (z - c*v) * (z - c*w)) ∧
  (∀ c : ℂ, c ∉ s →
    ¬∃ u v w : ℂ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    ∀ z : ℂ, (z - u) * (z - v) * (z - w) = (z - c*u) * (z - c*v) * (z - c*w)) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l149_14999


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l149_14969

/-- The minimum value of 2x^2 + 4xy + 5y^2 - 8x - 6y over all real numbers x and y is 3 -/
theorem min_value_quadratic_expression :
  ∀ x y : ℝ, 2 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ 3 ∧
  ∃ x₀ y₀ : ℝ, 2 * x₀^2 + 4 * x₀ * y₀ + 5 * y₀^2 - 8 * x₀ - 6 * y₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l149_14969


namespace NUMINAMATH_CALUDE_project_cans_total_l149_14990

theorem project_cans_total (martha_cans : ℕ) (diego_extra : ℕ) (additional_cans : ℕ) : 
  martha_cans = 90 →
  diego_extra = 10 →
  additional_cans = 5 →
  martha_cans + (martha_cans / 2 + diego_extra) + additional_cans = 150 :=
by sorry

end NUMINAMATH_CALUDE_project_cans_total_l149_14990


namespace NUMINAMATH_CALUDE_number_puzzle_l149_14934

theorem number_puzzle :
  ∀ (a b : ℤ),
  a + b = 72 →
  a = b + 12 →
  (a = 30 ∨ b = 30) →
  (a = 18 ∨ b = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l149_14934


namespace NUMINAMATH_CALUDE_smaller_circle_area_smaller_circle_radius_l149_14972

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  R : ℝ  -- radius of larger circle
  r : ℝ  -- radius of smaller circle
  tangent_length : ℝ  -- length of common tangent segment (PA and AB)
  circles_tangent : R = 2 * r  -- condition for external tangency
  common_tangent : tangent_length = 4  -- given PA = AB = 4

/-- The area of the smaller circle in a TangentCircles configuration is 2π -/
theorem smaller_circle_area (tc : TangentCircles) : 
  Real.pi * tc.r^2 = 2 * Real.pi := by
  sorry

/-- Alternative formulation using Real.sqrt -/
theorem smaller_circle_radius (tc : TangentCircles) :
  tc.r = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_area_smaller_circle_radius_l149_14972


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l149_14901

/-- Family of lines parameterized by t -/
def C (t : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x * Real.cos t + (y + 1) * Real.sin t = 2

/-- Predicate for three lines from C forming an equilateral triangle -/
def forms_equilateral_triangle (t₁ t₂ t₃ : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ x₃ y₃,
    C t₁ x₁ y₁ ∧ C t₂ x₂ y₂ ∧ C t₃ x₃ y₃ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2 ∧
    (x₂ - x₃)^2 + (y₂ - y₃)^2 = (x₃ - x₁)^2 + (y₃ - y₁)^2

/-- The area of the equilateral triangle formed by three lines from C -/
def triangle_area (t₁ t₂ t₃ : ℝ) : ℝ := sorry

theorem equilateral_triangle_area :
  ∀ t₁ t₂ t₃, forms_equilateral_triangle t₁ t₂ t₃ →
  triangle_area t₁ t₂ t₃ = 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l149_14901


namespace NUMINAMATH_CALUDE_smallest_next_divisor_l149_14983

theorem smallest_next_divisor (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : Even m) (h3 : 171 ∣ m) : 
  ∃ (d : ℕ), d ∣ m ∧ 171 < d ∧ d = 190 ∧ ∀ (x : ℕ), x ∣ m → 171 < x → x ≥ 190 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_l149_14983


namespace NUMINAMATH_CALUDE_remainder_sum_l149_14973

theorem remainder_sum (n : ℤ) : n % 20 = 13 → (n % 4 + n % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l149_14973


namespace NUMINAMATH_CALUDE_product_of_roots_l149_14978

theorem product_of_roots (x : ℝ) : 
  (∃ α β : ℝ, α * β = -10 ∧ -20 = -2 * x^2 - 6 * x ↔ (x = α ∨ x = β)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l149_14978


namespace NUMINAMATH_CALUDE_count_seven_digit_phone_numbers_l149_14943

/-- The number of different seven-digit phone numbers where the first digit cannot be zero -/
def seven_digit_phone_numbers : ℕ :=
  9 * 10^6

/-- Theorem stating that the number of different seven-digit phone numbers
    where the first digit cannot be zero is equal to 9 × 10^6 -/
theorem count_seven_digit_phone_numbers :
  seven_digit_phone_numbers = 9 * 10^6 := by
  sorry

end NUMINAMATH_CALUDE_count_seven_digit_phone_numbers_l149_14943


namespace NUMINAMATH_CALUDE_unique_solution_mn_l149_14995

theorem unique_solution_mn : ∃! (m n : ℕ+), 10 * m * n = 45 - 5 * m - 3 * n := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mn_l149_14995


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l149_14998

/-- The perpendicular bisector of a line segment connecting two points -/
theorem perpendicular_bisector_equation (A B : ℝ × ℝ) :
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let m_AB : ℝ := (B.2 - A.2) / (B.1 - A.1)
  let m_perp : ℝ := -1 / m_AB
  A = (0, 1) →
  B = (4, 3) →
  (λ (x y : ℝ) => 2 * x + y - 6 = 0) =
    (λ (x y : ℝ) => y - M.2 = m_perp * (x - M.1)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l149_14998


namespace NUMINAMATH_CALUDE_questionnaires_15_16_l149_14967

/-- Represents the number of questionnaires collected for each age group -/
structure QuestionnaireData where
  age_8_10 : ℕ
  age_11_12 : ℕ
  age_13_14 : ℕ
  age_15_16 : ℕ

/-- Represents the sampling data -/
structure SamplingData where
  total_sample : ℕ
  sample_11_12 : ℕ

/-- Theorem stating the number of questionnaires drawn from the 15-16 years old group -/
theorem questionnaires_15_16 (data : QuestionnaireData) (sampling : SamplingData) :
  data.age_8_10 = 120 →
  data.age_11_12 = 180 →
  data.age_13_14 = 240 →
  sampling.total_sample = 300 →
  sampling.sample_11_12 = 60 →
  (data.age_8_10 + data.age_11_12 + data.age_13_14 + data.age_15_16) * sampling.sample_11_12 = 
    sampling.total_sample * data.age_11_12 →
  (sampling.total_sample * data.age_15_16) / (data.age_8_10 + data.age_11_12 + data.age_13_14 + data.age_15_16) = 120 :=
by sorry

end NUMINAMATH_CALUDE_questionnaires_15_16_l149_14967


namespace NUMINAMATH_CALUDE_forty_eggs_not_eaten_l149_14971

/-- Represents the number of eggs in a problem about weekly egg consumption --/
structure EggProblem where
  trays_per_week : ℕ
  eggs_per_tray : ℕ
  children_eggs_per_day : ℕ
  parents_eggs_per_day : ℕ
  days_per_week : ℕ

/-- Calculates the number of eggs not eaten in a week --/
def eggs_not_eaten (p : EggProblem) : ℕ :=
  p.trays_per_week * p.eggs_per_tray - 
  (p.children_eggs_per_day + p.parents_eggs_per_day) * p.days_per_week

/-- Theorem stating that given the problem conditions, 40 eggs are not eaten in a week --/
theorem forty_eggs_not_eaten (p : EggProblem) 
  (h1 : p.trays_per_week = 2)
  (h2 : p.eggs_per_tray = 24)
  (h3 : p.children_eggs_per_day = 4)
  (h4 : p.parents_eggs_per_day = 4)
  (h5 : p.days_per_week = 7) :
  eggs_not_eaten p = 40 := by
  sorry

end NUMINAMATH_CALUDE_forty_eggs_not_eaten_l149_14971


namespace NUMINAMATH_CALUDE_license_plate_count_l149_14991

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits available -/
def num_digits : ℕ := 10

/-- A valid license plate configuration -/
structure LicensePlate where
  first : Fin num_letters
  second : Fin (num_letters + num_digits - 2)
  third : Fin num_letters
  fourth : Fin num_digits

/-- The total number of valid license plates -/
def total_license_plates : ℕ := num_letters * (num_letters + num_digits - 2) * num_letters * num_digits

theorem license_plate_count :
  total_license_plates = 236600 := by
  sorry

#eval total_license_plates

end NUMINAMATH_CALUDE_license_plate_count_l149_14991


namespace NUMINAMATH_CALUDE_root_product_theorem_l149_14961

theorem root_product_theorem (n r : ℝ) (c d : ℝ) : 
  (c^2 - n*c + 3 = 0) →
  (d^2 - n*d + 3 = 0) →
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) →
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) →
  s = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l149_14961


namespace NUMINAMATH_CALUDE_fourth_vertex_not_in_third_quadrant_l149_14933

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The main theorem -/
theorem fourth_vertex_not_in_third_quadrant :
  ∀ (p : Parallelogram),
    p.A = ⟨2, 0⟩ →
    p.B = ⟨-1/2, 0⟩ →
    p.C = ⟨0, 1⟩ →
    ¬(isInThirdQuadrant p.D) :=
by sorry

end NUMINAMATH_CALUDE_fourth_vertex_not_in_third_quadrant_l149_14933


namespace NUMINAMATH_CALUDE_set_intersection_example_l149_14931

theorem set_intersection_example : 
  let A : Set ℕ := {1, 3, 9}
  let B : Set ℕ := {1, 5, 9}
  A ∩ B = {1, 9} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l149_14931


namespace NUMINAMATH_CALUDE_top_view_area_is_eight_l149_14986

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the area of the top view of a rectangular prism -/
def topViewArea (d : PrismDimensions) : ℝ := d.length * d.width

/-- Theorem: The area of the top view of the given rectangular prism is 8 square units -/
theorem top_view_area_is_eight :
  let d : PrismDimensions := { length := 4, width := 2, height := 3 }
  topViewArea d = 8 := by
  sorry

end NUMINAMATH_CALUDE_top_view_area_is_eight_l149_14986


namespace NUMINAMATH_CALUDE_price_reduction_equation_l149_14959

/-- Represents the price reduction scenario for an item -/
structure PriceReduction where
  initial_price : ℝ
  final_price : ℝ
  reduction_percentage : ℝ
  num_reductions : ℕ

/-- Theorem stating the relationship between initial price, final price, and reduction percentage -/
theorem price_reduction_equation (pr : PriceReduction) 
  (h1 : pr.initial_price = 150)
  (h2 : pr.final_price = 96)
  (h3 : pr.num_reductions = 2) :
  pr.initial_price * (1 - pr.reduction_percentage)^pr.num_reductions = pr.final_price := by
  sorry

#check price_reduction_equation

end NUMINAMATH_CALUDE_price_reduction_equation_l149_14959


namespace NUMINAMATH_CALUDE_car_overtake_time_l149_14927

/-- The time it takes for a car to overtake a motorcyclist by 36 km -/
theorem car_overtake_time (v_motorcycle : ℝ) (v_car : ℝ) (head_start : ℝ) (overtake_distance : ℝ) :
  v_motorcycle = 45 →
  v_car = 60 →
  head_start = 2/3 →
  overtake_distance = 36 →
  ∃ t : ℝ, t = 4.4 ∧ 
    v_car * t = v_motorcycle * (t + head_start) + overtake_distance :=
by sorry

end NUMINAMATH_CALUDE_car_overtake_time_l149_14927


namespace NUMINAMATH_CALUDE_inverse_inequality_for_negative_reals_l149_14928

theorem inverse_inequality_for_negative_reals (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  1 / a > 1 / b := by
sorry

end NUMINAMATH_CALUDE_inverse_inequality_for_negative_reals_l149_14928


namespace NUMINAMATH_CALUDE_adding_five_increases_value_l149_14955

theorem adding_five_increases_value (x : ℝ) : x + 5 > x := by
  sorry

end NUMINAMATH_CALUDE_adding_five_increases_value_l149_14955


namespace NUMINAMATH_CALUDE_snack_cost_l149_14902

/-- Given the following conditions:
    - There are 4 people
    - Each ticket costs $18
    - The total cost for tickets and snacks for all 4 people is $92
    Prove that the cost of a set of snacks is $5. -/
theorem snack_cost (num_people : ℕ) (ticket_price : ℕ) (total_cost : ℕ) :
  num_people = 4 →
  ticket_price = 18 →
  total_cost = 92 →
  (total_cost - num_people * ticket_price) / num_people = 5 := by
  sorry

end NUMINAMATH_CALUDE_snack_cost_l149_14902


namespace NUMINAMATH_CALUDE_infinitely_many_2024_endings_l149_14930

/-- The sequence (x_n) defined by the given recurrence relation -/
def x : ℕ → ℕ
  | 0 => 0
  | 1 => 2024
  | (n + 2) => x (n + 1) + x n

/-- The set of natural numbers n where x_n ends with 2024 -/
def ends_with_2024 : Set ℕ := {n | x n % 10000 = 2024}

/-- The main theorem stating that there are infinitely many terms in the sequence ending with 2024 -/
theorem infinitely_many_2024_endings : Set.Infinite ends_with_2024 := by
  sorry


end NUMINAMATH_CALUDE_infinitely_many_2024_endings_l149_14930


namespace NUMINAMATH_CALUDE_complex_roots_of_equation_l149_14965

theorem complex_roots_of_equation : ∃ (z₁ z₂ : ℂ),
  z₁ = -1 + 2 * Real.sqrt 5 + (2 * Real.sqrt 5 / 5) * Complex.I ∧
  z₂ = -1 - 2 * Real.sqrt 5 - (2 * Real.sqrt 5 / 5) * Complex.I ∧
  z₁^2 + 2*z₁ = 16 + 8*Complex.I ∧
  z₂^2 + 2*z₂ = 16 + 8*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_of_equation_l149_14965


namespace NUMINAMATH_CALUDE_expression_factorization_l149_14926

theorem expression_factorization (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 6 * x^4 + 5) = 2 * (6 * x^6 + 21 * x^4 - 7) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l149_14926


namespace NUMINAMATH_CALUDE_interchange_relation_l149_14916

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The original number satisfies the given condition -/
def satisfies_condition (n : TwoDigitNumber) (c : Nat) : Prop :=
  10 * n.tens + n.ones = c * (n.tens + n.ones) + 3

/-- The number formed by interchanging digits -/
def interchange_digits (n : TwoDigitNumber) : Nat :=
  10 * n.ones + n.tens

/-- The main theorem to prove -/
theorem interchange_relation (n : TwoDigitNumber) (c : Nat) 
  (h : satisfies_condition n c) :
  interchange_digits n = (11 - c) * (n.tens + n.ones) := by
  sorry


end NUMINAMATH_CALUDE_interchange_relation_l149_14916


namespace NUMINAMATH_CALUDE_speaking_brother_is_tryalya_l149_14921

-- Define the brothers
inductive Brother
| Tralyalya
| Tryalya

-- Define the card suits
inductive Suit
| Black
| Red

-- Define the statement about having a black suit card
def claims_black_suit (b : Brother) : Prop :=
  match b with
  | Brother.Tralyalya => true
  | Brother.Tryalya => true

-- Define the rule that the brother with the black suit card cannot tell the truth
axiom black_suit_rule : ∀ (b : Brother), 
  (∃ (s : Suit), s = Suit.Black ∧ claims_black_suit b) → ¬(claims_black_suit b)

-- Theorem: The speaking brother must be Tryalya and he must have the black suit card
theorem speaking_brother_is_tryalya : 
  ∃ (b : Brother) (s : Suit), 
    b = Brother.Tryalya ∧ 
    s = Suit.Black ∧ 
    claims_black_suit b :=
sorry

end NUMINAMATH_CALUDE_speaking_brother_is_tryalya_l149_14921


namespace NUMINAMATH_CALUDE_probability_three_out_of_five_cured_l149_14913

theorem probability_three_out_of_five_cured (p : ℝ) (h : p = 0.9) :
  Nat.choose 5 3 * p^3 * (1 - p)^2 = Nat.choose 5 3 * 0.9^3 * 0.1^2 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_out_of_five_cured_l149_14913


namespace NUMINAMATH_CALUDE_trig_identity_l149_14919

theorem trig_identity (α β : ℝ) :
  1 - Real.cos (β - α) + Real.cos α - Real.cos β =
  4 * Real.cos (α / 2) * Real.sin (β / 2) * Real.sin ((β - α) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l149_14919


namespace NUMINAMATH_CALUDE_teresa_spent_forty_l149_14987

/-- The total amount spent by Teresa at the local shop -/
def total_spent (sandwich_price : ℚ) (sandwich_quantity : ℕ)
  (salami_price : ℚ) (olive_price_per_pound : ℚ) (olive_quantity : ℚ)
  (feta_price_per_pound : ℚ) (feta_quantity : ℚ) (bread_price : ℚ) : ℚ :=
  sandwich_price * sandwich_quantity +
  salami_price +
  3 * salami_price +
  olive_price_per_pound * olive_quantity +
  feta_price_per_pound * feta_quantity +
  bread_price

/-- Theorem: Teresa spends $40.00 at the local shop -/
theorem teresa_spent_forty : 
  total_spent 7.75 2 4 10 (1/4) 8 (1/2) 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_teresa_spent_forty_l149_14987


namespace NUMINAMATH_CALUDE_apricot_tea_calories_l149_14938

/-- Represents the composition of the apricot tea -/
structure ApricotTea where
  apricot_juice : ℝ
  honey : ℝ
  water : ℝ

/-- Calculates the total calories in the apricot tea mixture -/
def total_calories (tea : ApricotTea) : ℝ :=
  tea.apricot_juice * 0.3 + tea.honey * 3.04

/-- Calculates the total weight of the apricot tea mixture -/
def total_weight (tea : ApricotTea) : ℝ :=
  tea.apricot_juice + tea.honey + tea.water

/-- Theorem: 250g of Nathan's apricot tea contains 98.5 calories -/
theorem apricot_tea_calories :
  let tea : ApricotTea := { apricot_juice := 150, honey := 50, water := 300 }
  let caloric_density : ℝ := total_calories tea / total_weight tea
  250 * caloric_density = 98.5 := by
  sorry

#check apricot_tea_calories

end NUMINAMATH_CALUDE_apricot_tea_calories_l149_14938


namespace NUMINAMATH_CALUDE_sum_of_cubes_l149_14950

theorem sum_of_cubes (a b c : ℝ) 
  (h1 : a + b + c = 4) 
  (h2 : a * b + a * c + b * c = 7) 
  (h3 : a * b * c = -10) : 
  a^3 + b^3 + c^3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l149_14950


namespace NUMINAMATH_CALUDE_simplify_expression_l149_14963

theorem simplify_expression (x : ℝ) : (3*x)^3 - (4*x^2)*(2*x^3) = 27*x^3 - 8*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l149_14963


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l149_14945

/-- 
Given a point P in a Cartesian coordinate system, this theorem states that 
its coordinates with respect to the origin are the negatives of its original coordinates.
-/
theorem point_coordinates_wrt_origin (x y : ℝ) : 
  let P : ℝ × ℝ := (x, y)
  let P_wrt_origin : ℝ × ℝ := (-x, -y)
  P_wrt_origin = (-(P.1), -(P.2)) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l149_14945


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l149_14909

theorem solution_set_implies_a_equals_one (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x < 0 ↔ 0 < x ∧ x < 1) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l149_14909


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l149_14962

theorem simplify_complex_fraction (m : ℝ) (h1 : m ≠ 2) (h2 : m ≠ -3) :
  (m - (4*m - 9) / (m - 2)) / ((m^2 - 9) / (m - 2)) = (m - 3) / (m + 3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l149_14962


namespace NUMINAMATH_CALUDE_partnership_profit_share_l149_14964

/-- Given a partnership where A invests 3 times as much as B and 2/3 of what C invests,
    and the total profit is 55000, prove that C's share of the profit is (9/17) * 55000. -/
theorem partnership_profit_share (a b c : ℝ) (total_profit : ℝ) : 
  a = 3 * b ∧ a = (2/3) * c ∧ total_profit = 55000 → 
  c * total_profit / (a + b + c) = (9/17) * 55000 := by
sorry

end NUMINAMATH_CALUDE_partnership_profit_share_l149_14964


namespace NUMINAMATH_CALUDE_probability_of_two_triples_l149_14912

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (ranks : Nat)
  (cards_per_rank : Nat)
  (h_total : total_cards = ranks * cards_per_rank)

/-- Represents the specific hand we're looking for -/
structure TargetHand :=
  (total_cards : Nat)
  (sets : Nat)
  (cards_per_set : Nat)
  (h_total : total_cards = sets * cards_per_set)

def probability_of_target_hand (d : Deck) (h : TargetHand) : ℚ :=
  (d.ranks.choose h.sets) * (d.cards_per_rank.choose h.cards_per_set)^h.sets /
  d.total_cards.choose h.total_cards

theorem probability_of_two_triples (d : Deck) (h : TargetHand) :
  d.total_cards = 52 →
  d.ranks = 13 →
  d.cards_per_rank = 4 →
  h.total_cards = 6 →
  h.sets = 2 →
  h.cards_per_set = 3 →
  probability_of_target_hand d h = 13 / 106470 :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_triples_l149_14912


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l149_14977

theorem greatest_whole_number_satisfying_inequality :
  ∀ n : ℤ, (∀ x : ℤ, x ≤ n → 4 * x - 3 < 2 - x) → n ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l149_14977


namespace NUMINAMATH_CALUDE_no_solutions_power_equation_l149_14923

theorem no_solutions_power_equation (x n r : ℕ) (hx : x > 1) :
  x^(2*n + 1) ≠ 2^r + 1 ∧ x^(2*n + 1) ≠ 2^r - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_power_equation_l149_14923


namespace NUMINAMATH_CALUDE_sin_value_for_specific_tan_l149_14917

/-- Prove that for an acute angle α, if tan(π - α) + 3 = 0, then sinα = 3√10 / 10 -/
theorem sin_value_for_specific_tan (α : Real) : 
  0 < α ∧ α < π / 2 →  -- α is an acute angle
  Real.tan (π - α) + 3 = 0 → 
  Real.sin α = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_value_for_specific_tan_l149_14917


namespace NUMINAMATH_CALUDE_lesser_solution_quadratic_l149_14932

theorem lesser_solution_quadratic (x : ℝ) : 
  x^2 + 10*x - 24 = 0 ∧ ∀ y, y^2 + 10*y - 24 = 0 → x ≤ y → x = -12 :=
by sorry

end NUMINAMATH_CALUDE_lesser_solution_quadratic_l149_14932


namespace NUMINAMATH_CALUDE_area_of_triangle_AFK_l149_14925

/-- Parabola with equation y² = 8x, focus F(2, 0), and directrix intersecting x-axis at K(-2, 0) -/
structure Parabola where
  F : ℝ × ℝ := (2, 0)
  K : ℝ × ℝ := (-2, 0)

/-- Point on the parabola -/
structure PointOnParabola (p : Parabola) where
  A : ℝ × ℝ
  on_parabola : A.2^2 = 8 * A.1
  distance_condition : (A.1 + 2)^2 + A.2^2 = 2 * ((A.1 - 2)^2 + A.2^2)

/-- The area of triangle AFK is 8 -/
theorem area_of_triangle_AFK (p : Parabola) (point : PointOnParabola p) :
  (1 / 2 : ℝ) * 4 * |point.A.2| = 8 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AFK_l149_14925


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l149_14929

def scores : List ℝ := [90, 93.5, 87, 96, 92, 89.5]

theorem arithmetic_mean_of_scores :
  (scores.sum / scores.length : ℝ) = 91.333333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l149_14929


namespace NUMINAMATH_CALUDE_range_of_a_l149_14976

def f (x a : ℝ) : ℝ := |2*x - a| + a

def g (x : ℝ) : ℝ := |2*x - 1|

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a + g x ≥ 2*a^2 - 13) → a ∈ Set.Icc (-Real.sqrt 7) 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l149_14976


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l149_14954

theorem smallest_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 3457) % 15 = 1537 % 15 ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 3457) % 15 = 1537 % 15 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l149_14954


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l149_14948

theorem nested_fraction_evaluation :
  1 + 1 / (2 + 1 / (3 + 1 / (3 + 3))) = 63 / 44 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l149_14948


namespace NUMINAMATH_CALUDE_square_area_error_l149_14989

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.01)
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.0201 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l149_14989


namespace NUMINAMATH_CALUDE_M_divisible_by_40_l149_14957

/-- M is the number formed by concatenating integers from 1 to 39 -/
def M : ℕ := sorry

/-- Theorem stating that M is divisible by 40 -/
theorem M_divisible_by_40 : 40 ∣ M := by sorry

end NUMINAMATH_CALUDE_M_divisible_by_40_l149_14957


namespace NUMINAMATH_CALUDE_max_revenue_theorem_l149_14936

/-- Represents the advertising allocation problem for a company --/
structure AdvertisingProblem where
  totalTime : ℝ
  totalBudget : ℝ
  rateA : ℝ
  rateB : ℝ
  revenueA : ℝ
  revenueB : ℝ

/-- Represents a solution to the advertising allocation problem --/
structure AdvertisingSolution where
  timeA : ℝ
  timeB : ℝ
  revenue : ℝ

/-- Checks if a solution is valid for a given problem --/
def isValidSolution (p : AdvertisingProblem) (s : AdvertisingSolution) : Prop :=
  s.timeA ≥ 0 ∧ s.timeB ≥ 0 ∧
  s.timeA + s.timeB ≤ p.totalTime ∧
  s.timeA * p.rateA + s.timeB * p.rateB ≤ p.totalBudget ∧
  s.revenue = s.timeA * p.revenueA + s.timeB * p.revenueB

/-- Theorem stating that the given solution maximizes revenue --/
theorem max_revenue_theorem (p : AdvertisingProblem)
  (h1 : p.totalTime = 300)
  (h2 : p.totalBudget = 90000)
  (h3 : p.rateA = 500)
  (h4 : p.rateB = 200)
  (h5 : p.revenueA = 0.3)
  (h6 : p.revenueB = 0.2) :
  ∃ (s : AdvertisingSolution),
    isValidSolution p s ∧
    s.timeA = 100 ∧
    s.timeB = 200 ∧
    s.revenue = 70 ∧
    ∀ (s' : AdvertisingSolution), isValidSolution p s' → s'.revenue ≤ s.revenue :=
sorry

end NUMINAMATH_CALUDE_max_revenue_theorem_l149_14936


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l149_14946

open Real

-- Define the triangle
def Triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem acute_triangle_properties
  (A B C a b c : ℝ)
  (h_triangle : Triangle A B C a b c)
  (h_acute : A < π/2 ∧ B < π/2 ∧ C < π/2)
  (h_relation : c - b = 2 * b * cos A) :
  A = 2 * B ∧
  π/6 < B ∧ B < π/4 ∧
  sqrt 2 < a/b ∧ a/b < sqrt 3 ∧
  5 * sqrt 3 / 3 < 1 / tan B - 1 / tan A + 2 * sin A ∧
  1 / tan B - 1 / tan A + 2 * sin A < 3 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l149_14946


namespace NUMINAMATH_CALUDE_intersection_sum_l149_14907

/-- Given two lines y = nx + 5 and y = 4x + c that intersect at (8, 9),
    prove that n + c = -22.5 -/
theorem intersection_sum (n c : ℝ) : 
  (∀ x y : ℝ, y = n * x + 5 ∨ y = 4 * x + c) →
  9 = n * 8 + 5 →
  9 = 4 * 8 + c →
  n + c = -22.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l149_14907


namespace NUMINAMATH_CALUDE_two_primes_not_congruent_to_one_l149_14975

theorem two_primes_not_congruent_to_one (p : Nat) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ (q r : Nat), q ≠ r ∧ q.Prime ∧ r.Prime ∧ 2 ≤ q ∧ q ≤ p - 2 ∧ 2 ≤ r ∧ r ≤ p - 2 ∧
  ¬(q^(p-1) ≡ 1 [MOD p^2]) ∧ ¬(r^(p-1) ≡ 1 [MOD p^2]) := by
  sorry

end NUMINAMATH_CALUDE_two_primes_not_congruent_to_one_l149_14975


namespace NUMINAMATH_CALUDE_square_difference_equality_l149_14918

theorem square_difference_equality : (19 + 12)^2 - (12^2 + 19^2) = 456 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l149_14918


namespace NUMINAMATH_CALUDE_range_of_x_l149_14984

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)

-- Define the theorem
theorem range_of_x (x : ℝ) (h : f (Real.log x) > f 1) :
  (1 / 10 : ℝ) < x ∧ x < 10 := by sorry

end NUMINAMATH_CALUDE_range_of_x_l149_14984


namespace NUMINAMATH_CALUDE_eight_b_plus_one_composite_l149_14997

theorem eight_b_plus_one_composite (a b : ℕ) (h1 : a > b) (h2 : a - b = 5 * b^2 - 4 * a^2) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ 8 * b + 1 = x * y :=
sorry

end NUMINAMATH_CALUDE_eight_b_plus_one_composite_l149_14997


namespace NUMINAMATH_CALUDE_fraction_evaluation_l149_14935

theorem fraction_evaluation : 
  (10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l149_14935


namespace NUMINAMATH_CALUDE_function_inequality_l149_14956

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h1 : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) : 
  Real.exp a * f 0 < f a := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l149_14956


namespace NUMINAMATH_CALUDE_sum_of_odd_powers_l149_14968

theorem sum_of_odd_powers (x y z : ℝ) (n : ℕ) (h1 : x + y + z = 1) 
  (h2 : Real.arctan x + Real.arctan y + Real.arctan z = π / 4) : 
  x^(2*n + 1) + y^(2*n + 1) + z^(2*n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_powers_l149_14968


namespace NUMINAMATH_CALUDE_unique_function_solution_l149_14982

theorem unique_function_solution (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (1 + x * y) - f (x + y) = f x * f y) 
  (h2 : f (-1) ≠ 0) : 
  ∀ x : ℝ, f x = x - 1 := by
sorry

end NUMINAMATH_CALUDE_unique_function_solution_l149_14982


namespace NUMINAMATH_CALUDE_similar_triangles_proportion_l149_14904

/-- Two triangles PQR and STU with given side lengths and angles -/
structure Triangles where
  PR : ℝ
  QR : ℝ
  SU : ℝ
  ST : ℝ
  TU : ℝ
  angle_PQR : ℝ
  angle_STU : ℝ

/-- The triangles are similar -/
def are_similar (t : Triangles) : Prop :=
  t.angle_PQR = t.angle_STU ∧ t.PR / t.SU = t.QR / t.TU

/-- The theorem to prove -/
theorem similar_triangles_proportion (t : Triangles) 
  (h1 : t.PR = 21)
  (h2 : t.QR = 15)
  (h3 : t.SU = 9)
  (h4 : t.ST = 4.5)
  (h5 : t.TU = 7.5)
  (h6 : t.angle_PQR = 2 * π / 3)
  (h7 : t.angle_STU = 2 * π / 3)
  (h8 : are_similar t) :
  t.PR * t.ST / t.SU = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_proportion_l149_14904


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l149_14953

theorem cube_volume_from_surface_area :
  ∀ s : ℝ,
  s > 0 →
  6 * s^2 = 864 →
  s^3 = 1728 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l149_14953


namespace NUMINAMATH_CALUDE_power_comparison_l149_14906

theorem power_comparison : 2^1997 > 5^850 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l149_14906


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l149_14944

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2016 = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l149_14944


namespace NUMINAMATH_CALUDE_subtracted_number_l149_14937

theorem subtracted_number (x : ℝ) : 3889 + 12.808 - x = 3854.002 → x = 47.806 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l149_14937


namespace NUMINAMATH_CALUDE_function_analysis_l149_14994

def f (x : ℝ) := x^3 - 3*x^2 - 9*x + 11

theorem function_analysis :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≤ f (-1)) ∧
  f (-1) = 16 ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3 - ε) (3 + ε), f x ≥ f 3) ∧
  f 3 = -16 ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) 1, f x < f 1) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo 1 (1 + ε), f x > f 1) ∧
  f 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_function_analysis_l149_14994


namespace NUMINAMATH_CALUDE_simplify_expression_l149_14993

theorem simplify_expression : 4^4 * 9^4 * 4^9 * 9^9 = 36^13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l149_14993


namespace NUMINAMATH_CALUDE_notebook_difference_l149_14947

theorem notebook_difference (tara_spent lea_spent : ℚ) 
  (h1 : tara_spent = 5.20)
  (h2 : lea_spent = 7.80)
  (h3 : ∃ (price : ℚ), price > 1 ∧ 
    ∃ (tara_count lea_count : ℕ), 
      tara_count * price = tara_spent ∧ 
      lea_count * price = lea_spent) :
  ∃ (price : ℚ) (tara_count lea_count : ℕ), 
    price > 1 ∧
    tara_count * price = tara_spent ∧
    lea_count * price = lea_spent ∧
    lea_count = tara_count + 2 :=
by sorry

end NUMINAMATH_CALUDE_notebook_difference_l149_14947


namespace NUMINAMATH_CALUDE_fruit_store_problem_l149_14981

-- Define the types of fruits
inductive FruitType
| A
| B

-- Define the purchase data
structure PurchaseData where
  typeA : ℕ
  typeB : ℕ
  totalCost : ℕ

-- Define the problem parameters
def firstPurchase : PurchaseData := ⟨60, 40, 1520⟩
def secondPurchase : PurchaseData := ⟨30, 50, 1360⟩
def thirdPurchaseTotal : ℕ := 200
def thirdPurchaseMaxCost : ℕ := 3360
def typeASellingPrice : ℕ := 17
def typeBSellingPrice : ℕ := 30
def minProfit : ℕ := 800

-- Define the theorem
theorem fruit_store_problem :
  ∃ (priceA priceB : ℕ) (m : ℕ),
    -- Conditions for the first two purchases
    priceA * firstPurchase.typeA + priceB * firstPurchase.typeB = firstPurchase.totalCost ∧
    priceA * secondPurchase.typeA + priceB * secondPurchase.typeB = secondPurchase.totalCost ∧
    -- Conditions for the third purchase
    ∀ (x : ℕ),
      x ≤ thirdPurchaseTotal →
      priceA * x + priceB * (thirdPurchaseTotal - x) ≤ thirdPurchaseMaxCost →
      (typeASellingPrice - priceA) * (x - m) + (typeBSellingPrice - priceB) * (thirdPurchaseTotal - x - 3 * m) ≥ minProfit →
      -- Conclusion
      priceA = 12 ∧ priceB = 20 ∧ m ≤ 22 ∧
      ∀ (m' : ℕ), m' > m → 
        ¬(∃ (x : ℕ),
          x ≤ thirdPurchaseTotal ∧
          priceA * x + priceB * (thirdPurchaseTotal - x) ≤ thirdPurchaseMaxCost ∧
          (typeASellingPrice - priceA) * (x - m') + (typeBSellingPrice - priceB) * (thirdPurchaseTotal - x - 3 * m') ≥ minProfit) :=
by sorry

end NUMINAMATH_CALUDE_fruit_store_problem_l149_14981


namespace NUMINAMATH_CALUDE_shifted_parabola_equation_l149_14903

/-- Represents a vertical shift of a parabola -/
def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  fun x => f x - shift

/-- The original parabola function -/
def original_parabola : ℝ → ℝ :=
  fun x => 2 * x^2

/-- The amount of downward shift -/
def shift_amount : ℝ := 4

theorem shifted_parabola_equation :
  vertical_shift original_parabola shift_amount =
  fun x => 2 * x^2 - 4 := by sorry

end NUMINAMATH_CALUDE_shifted_parabola_equation_l149_14903


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l149_14900

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l149_14900


namespace NUMINAMATH_CALUDE_badminton_cost_equality_l149_14960

/-- Represents the cost calculation for two stores selling badminton equipment -/
theorem badminton_cost_equality (x : ℝ) : x ≥ 5 → (125 + 5*x = 135 + 4.5*x ↔ x = 20) :=
by
  sorry

#check badminton_cost_equality

end NUMINAMATH_CALUDE_badminton_cost_equality_l149_14960


namespace NUMINAMATH_CALUDE_integer_root_condition_l149_14939

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, x^3 + 3*x^2 + a*x + 11 = 0

theorem integer_root_condition (a : ℤ) :
  has_integer_root a ↔ a = -155 ∨ a = -15 ∨ a = 13 ∨ a = 87 :=
sorry

end NUMINAMATH_CALUDE_integer_root_condition_l149_14939


namespace NUMINAMATH_CALUDE_orange_cells_theorem_l149_14924

/-- Represents the possible outcomes of orange cells on the board -/
inductive OrangeCellsOutcome
  | lower  : OrangeCellsOutcome  -- represents 2021 * 2020
  | higher : OrangeCellsOutcome  -- represents 2022 * 2020

/-- The size of one side of the square board -/
def boardSize : Nat := 2022

/-- The size of one side of the paintable square -/
def squareSize : Nat := 2

/-- Represents the game rules and outcomes -/
structure GameBoard where
  size : Nat
  squareSize : Nat
  possibleOutcomes : List OrangeCellsOutcome

/-- The main theorem to prove -/
theorem orange_cells_theorem (board : GameBoard) 
  (h1 : board.size = boardSize) 
  (h2 : board.squareSize = squareSize) 
  (h3 : board.possibleOutcomes = [OrangeCellsOutcome.lower, OrangeCellsOutcome.higher]) : 
  ∃ (n : Nat), (n = 2021 * 2020 ∨ n = 2022 * 2020) ∧ 
  (∀ (m : Nat), m = 2021 * 2020 ∨ m = 2022 * 2020 → 
    ∃ (outcome : OrangeCellsOutcome), outcome ∈ board.possibleOutcomes ∧
    (outcome = OrangeCellsOutcome.lower → m = 2021 * 2020) ∧
    (outcome = OrangeCellsOutcome.higher → m = 2022 * 2020)) :=
  sorry


end NUMINAMATH_CALUDE_orange_cells_theorem_l149_14924


namespace NUMINAMATH_CALUDE_count_with_zero_1000_l149_14996

def count_with_zero (n : ℕ) : ℕ :=
  (n + 1) - (9 * 9 * 10)

theorem count_with_zero_1000 : count_with_zero 1000 = 181 := by
  sorry

end NUMINAMATH_CALUDE_count_with_zero_1000_l149_14996


namespace NUMINAMATH_CALUDE_polynomial_simplification_l149_14908

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + x^3 - 6 * x^2 + 9 * x - 5) + 
  (-x^4 + 2 * x^3 - 3 * x^2 + 4 * x - 2) + 
  (3 * x^4 - 3 * x^3 + x^2 - x + 1) = 
  4 * x^4 - 8 * x^2 + 12 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l149_14908


namespace NUMINAMATH_CALUDE_find_n_l149_14951

theorem find_n : ∃ n : ℕ, 2^3 * 8 = 4^n ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l149_14951


namespace NUMINAMATH_CALUDE_rectangular_field_area_l149_14988

theorem rectangular_field_area (width length perimeter area : ℝ) : 
  width = (1/3) * length →
  perimeter = 2 * (width + length) →
  perimeter = 60 →
  area = width * length →
  area = 168.75 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l149_14988


namespace NUMINAMATH_CALUDE_league_games_count_l149_14941

/-- The number of games played in a season for a league with a given number of teams and games per pair of teams. -/
def games_in_season (num_teams : ℕ) (games_per_pair : ℕ) : ℕ :=
  (num_teams * (num_teams - 1) / 2) * games_per_pair

/-- Theorem: In a league with 20 teams where each pair of teams plays 10 games, 
    the total number of games in the season is 1900. -/
theorem league_games_count : games_in_season 20 10 = 1900 := by
  sorry

#eval games_in_season 20 10

end NUMINAMATH_CALUDE_league_games_count_l149_14941


namespace NUMINAMATH_CALUDE_lines_are_parallel_l149_14915

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

theorem lines_are_parallel : 
  let line1 : Line := { a := 1, b := -1, c := 2 }
  let line2 : Line := { a := 1, b := -1, c := 1 }
  parallel line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l149_14915


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l149_14911

/-- The ratio of the volume of a cone to the volume of a cylinder, given specific proportions -/
theorem cone_cylinder_volume_ratio 
  (r h : ℝ) 
  (h_pos : h > 0) 
  (r_pos : r > 0) : 
  (1 / 3 * π * (r / 2)^2 * (h / 3)) / (π * r^2 * h) = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l149_14911
