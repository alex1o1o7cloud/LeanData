import Mathlib

namespace NUMINAMATH_CALUDE_not_always_parallel_to_intersection_l3733_373323

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- State the theorem
theorem not_always_parallel_to_intersection
  (α β : Plane) (m n : Line) :
  ¬(∀ (α β : Plane) (m n : Line),
    line_parallel_plane m α ∧ intersect α β n → parallel m n) :=
by sorry

end NUMINAMATH_CALUDE_not_always_parallel_to_intersection_l3733_373323


namespace NUMINAMATH_CALUDE_total_pencils_after_adding_l3733_373386

def initial_pencils : ℕ := 115
def added_pencils : ℕ := 100

theorem total_pencils_after_adding :
  initial_pencils + added_pencils = 215 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_after_adding_l3733_373386


namespace NUMINAMATH_CALUDE_clients_using_all_three_l3733_373399

def total_clients : ℕ := 180
def tv_clients : ℕ := 115
def radio_clients : ℕ := 110
def magazine_clients : ℕ := 130
def tv_and_magazine : ℕ := 85
def tv_and_radio : ℕ := 75
def radio_and_magazine : ℕ := 95

theorem clients_using_all_three :
  tv_clients + radio_clients + magazine_clients -
  tv_and_magazine - tv_and_radio - radio_and_magazine +
  (total_clients - (tv_clients + radio_clients + magazine_clients -
  tv_and_magazine - tv_and_radio - radio_and_magazine)) = 80 :=
by sorry

end NUMINAMATH_CALUDE_clients_using_all_three_l3733_373399


namespace NUMINAMATH_CALUDE_turtleneck_discount_l3733_373302

theorem turtleneck_discount (C : ℝ) (C_pos : C > 0) : 
  let initial_price := 1.2 * C
  let marked_up_price := 1.25 * initial_price
  let final_price := (1 - 0.08) * marked_up_price
  final_price = 1.38 * C := by sorry

end NUMINAMATH_CALUDE_turtleneck_discount_l3733_373302


namespace NUMINAMATH_CALUDE_arc_length_sector_l3733_373397

/-- The arc length of a sector with central angle 36° and radius 15 is 3π. -/
theorem arc_length_sector (centralAngle : Real) (radius : Real) : 
  centralAngle = 36 → radius = 15 → 
  (centralAngle * π * radius) / 180 = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_arc_length_sector_l3733_373397


namespace NUMINAMATH_CALUDE_apple_sale_percentage_l3733_373383

/-- The percentage of apples sold by a fruit seller -/
theorem apple_sale_percentage (original : Real) (remaining : Real) 
  (h1 : original = 2499.9987500006246)
  (h2 : remaining = 500) :
  let sold := original - remaining
  let percentage := (sold / original) * 100
  ∃ ε > 0, abs (percentage - 80) < ε :=
by sorry

end NUMINAMATH_CALUDE_apple_sale_percentage_l3733_373383


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_10_l3733_373365

theorem binomial_coefficient_20_10 
  (h1 : Nat.choose 18 8 = 43758)
  (h2 : Nat.choose 18 9 = 48620)
  (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_10_l3733_373365


namespace NUMINAMATH_CALUDE_permutations_of_seven_distinct_objects_l3733_373337

theorem permutations_of_seven_distinct_objects : Nat.factorial 7 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_seven_distinct_objects_l3733_373337


namespace NUMINAMATH_CALUDE_modulus_of_z_squared_l3733_373387

theorem modulus_of_z_squared (z : ℂ) (h : z^2 = 3 + 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_squared_l3733_373387


namespace NUMINAMATH_CALUDE_cistern_filling_fraction_l3733_373379

theorem cistern_filling_fraction (fill_time : ℝ) (fraction : ℝ) : 
  (fill_time = 25) → 
  (fraction * fill_time = 25) → 
  (fraction = 1 / 25) :=
by sorry

end NUMINAMATH_CALUDE_cistern_filling_fraction_l3733_373379


namespace NUMINAMATH_CALUDE_unique_k_value_l3733_373353

/-- The equation has infinitely many solutions when the coefficients of x are equal on both sides -/
def has_infinitely_many_solutions (k : ℝ) : Prop :=
  3 * k = 15

/-- The value of k for which the equation has infinitely many solutions -/
def k_value : ℝ := 5

/-- Theorem stating that k_value is the unique solution -/
theorem unique_k_value :
  has_infinitely_many_solutions k_value ∧
  ∀ k : ℝ, has_infinitely_many_solutions k → k = k_value :=
by sorry

end NUMINAMATH_CALUDE_unique_k_value_l3733_373353


namespace NUMINAMATH_CALUDE_smaller_angle_measure_l3733_373356

-- Define a parallelogram
structure Parallelogram where
  -- Smaller angle
  angle1 : ℝ
  -- Larger angle
  angle2 : ℝ
  -- Condition: angle2 exceeds angle1 by 70 degrees
  angle_diff : angle2 = angle1 + 70
  -- Condition: adjacent angles are supplementary
  supplementary : angle1 + angle2 = 180

-- Theorem statement
theorem smaller_angle_measure (p : Parallelogram) : p.angle1 = 55 := by
  sorry

end NUMINAMATH_CALUDE_smaller_angle_measure_l3733_373356


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l3733_373322

/-- A function f is decreasing on ℝ if for all x, y ∈ ℝ, x < y implies f(x) > f(y) -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The set of all real numbers x satisfying f(1/|x|) < f(1) for a decreasing function f -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | f (1 / |x|) < f 1}

theorem solution_set_is_open_interval (f : ℝ → ℝ) (h : DecreasingOn f) :
  SolutionSet f = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l3733_373322


namespace NUMINAMATH_CALUDE_sum_of_exponents_is_eight_l3733_373360

/-- Represents the exponents of variables in a simplified cube root expression -/
structure SimplifiedCubeRootExponents where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Simplifies the cube root of 40a^6b^7c^14 and returns the exponents of variables outside the radical -/
def simplify_cube_root : SimplifiedCubeRootExponents := {
  a := 2,
  b := 2,
  c := 4
}

/-- The sum of exponents outside the radical after simplifying ∛(40a^6b^7c^14) is 8 -/
theorem sum_of_exponents_is_eight :
  (simplify_cube_root.a + simplify_cube_root.b + simplify_cube_root.c) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_is_eight_l3733_373360


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3733_373370

theorem right_triangle_hypotenuse : 
  ∀ (a : ℝ), a > 0 → a^2 = 8^2 + 15^2 → a = 17 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3733_373370


namespace NUMINAMATH_CALUDE_sea_creatures_lost_l3733_373378

-- Define the initial number of items collected
def sea_stars : ℕ := 34
def seashells : ℕ := 21
def snails : ℕ := 29

-- Define the number of items left at the end
def items_left : ℕ := 59

-- Define the total number of items collected
def total_collected : ℕ := sea_stars + seashells + snails

-- Define the number of items lost
def items_lost : ℕ := total_collected - items_left

-- Theorem statement
theorem sea_creatures_lost : items_lost = 25 := by
  sorry

end NUMINAMATH_CALUDE_sea_creatures_lost_l3733_373378


namespace NUMINAMATH_CALUDE_triangle_problem_l3733_373395

/-- Given an acute triangle ABC with collinear vectors m and n, prove B = π/6 and a + c = 2 + √3 -/
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) : 
  0 < B → B < π/2 →  -- B is acute
  (2 * Real.sin (A + C)) * (2 * Real.cos (B/2)^2 - 1) = Real.sqrt 3 * Real.cos (2*B) →  -- m and n are collinear
  b = 1 →  -- given condition
  a * c * Real.sin B / 2 = Real.sqrt 3 / 2 →  -- area condition
  (B = π/6 ∧ a + c = 2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3733_373395


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3733_373332

/-- The number of distinct points common to the circle x^2 + y^2 = 16 and the vertical line x = 4 is one. -/
theorem circle_line_intersection :
  ∃! p : ℝ × ℝ, (p.1^2 + p.2^2 = 16) ∧ (p.1 = 4) := by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3733_373332


namespace NUMINAMATH_CALUDE_second_graders_count_l3733_373329

theorem second_graders_count (kindergartners : ℕ) (first_graders : ℕ) (total_students : ℕ) 
  (h1 : kindergartners = 34)
  (h2 : first_graders = 48)
  (h3 : total_students = 120) :
  total_students - (kindergartners + first_graders) = 38 := by
  sorry

end NUMINAMATH_CALUDE_second_graders_count_l3733_373329


namespace NUMINAMATH_CALUDE_scientific_notation_of_120_l3733_373380

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_120 :
  toScientificNotation 120 = ScientificNotation.mk 1.2 2 (by norm_num) (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_120_l3733_373380


namespace NUMINAMATH_CALUDE_intersecting_line_circle_isosceles_right_triangle_l3733_373334

/-- Given a line and a circle that intersect at two points forming an isosceles right triangle with a third point, prove the value of the parameter a. -/
theorem intersecting_line_circle_isosceles_right_triangle (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 + a * A.2 - 1 = 0 ∧ (A.1 + a)^2 + (A.2 - 1)^2 = 1) ∧
    (B.1 + a * B.2 - 1 = 0 ∧ (B.1 + a)^2 + (B.2 - 1)^2 = 1) ∧
    A ≠ B) →
  (∃ C : ℝ × ℝ, 
    (C.1 + a * C.2 - 1 ≠ 0 ∨ (C.1 + a)^2 + (C.2 - 1)^2 ≠ 1) ∧
    (dist A C = dist B C ∧ dist A B = dist A C * Real.sqrt 2)) →
  a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_intersecting_line_circle_isosceles_right_triangle_l3733_373334


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_22_5_l3733_373392

/-- The area of a quadrilateral with vertices at (1, 2), (1, -1), (4, -1), and (7, 8) -/
def quadrilateral_area : ℝ :=
  let A := (1, 2)
  let B := (1, -1)
  let C := (4, -1)
  let D := (7, 8)
  -- Area calculation goes here
  0 -- Placeholder

theorem quadrilateral_area_is_22_5 : quadrilateral_area = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_22_5_l3733_373392


namespace NUMINAMATH_CALUDE_digit_count_problem_l3733_373320

theorem digit_count_problem (n : ℕ) 
  (h1 : (n : ℝ) * 500 = 14 * 390 + 6 * 756.67)
  (h2 : n > 0) : 
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_digit_count_problem_l3733_373320


namespace NUMINAMATH_CALUDE_largest_angle_in_three_three_four_triangle_l3733_373359

/-- A triangle with interior angles in the ratio 3:3:4 has its largest angle measuring 72° -/
theorem largest_angle_in_three_three_four_triangle : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  a / 3 = b / 3 ∧ b / 3 = c / 4 →
  max a (max b c) = 72 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_three_three_four_triangle_l3733_373359


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l3733_373391

def trailing_zeros (n : ℕ) : ℕ := sorry

theorem product_trailing_zeros : trailing_zeros (45 * 160 * 7) = 2 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l3733_373391


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_negative_three_l3733_373345

def expression (x : ℝ) : ℝ :=
  4 * (x^3 - 2*x^4) + 3 * (x^2 - 3*x^3 + 4*x^6) - (5*x^4 - 2*x^3)

theorem coefficient_of_x_cubed_is_negative_three :
  (deriv (deriv (deriv expression))) 0 / 6 = -3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_negative_three_l3733_373345


namespace NUMINAMATH_CALUDE_max_value_fraction_l3733_373352

theorem max_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 2 / 2 ∧
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    (x' * y' + y' * z') / (x'^2 + y'^2 + z'^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3733_373352


namespace NUMINAMATH_CALUDE_total_non_defective_engines_l3733_373319

/-- Represents a batch of engines with their total count and defect rate -/
structure Batch where
  total : ℕ
  defect_rate : ℚ
  defect_rate_valid : 0 ≤ defect_rate ∧ defect_rate ≤ 1

/-- Calculates the number of non-defective engines in a batch -/
def non_defective (b : Batch) : ℚ :=
  b.total * (1 - b.defect_rate)

/-- The list of batches with their respective data -/
def batches : List Batch := [
  ⟨140, 12/100, by norm_num⟩,
  ⟨150, 18/100, by norm_num⟩,
  ⟨170, 22/100, by norm_num⟩,
  ⟨180, 28/100, by norm_num⟩,
  ⟨190, 32/100, by norm_num⟩,
  ⟨210, 36/100, by norm_num⟩,
  ⟨220, 41/100, by norm_num⟩
]

/-- The theorem stating the total number of non-defective engines -/
theorem total_non_defective_engines :
  Int.floor (batches.map non_defective).sum = 902 := by
  sorry

end NUMINAMATH_CALUDE_total_non_defective_engines_l3733_373319


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l3733_373339

theorem sqrt_sum_fractions : Real.sqrt ((1 : ℝ) / 8 + (1 : ℝ) / 18) = (Real.sqrt 26) / 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l3733_373339


namespace NUMINAMATH_CALUDE_system_solution_l3733_373351

theorem system_solution :
  ∃ (x y : ℝ), 
    y * (x + y)^2 = 9 ∧
    y * (x^3 - y^3) = 7 ∧
    x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3733_373351


namespace NUMINAMATH_CALUDE_equation_solution_l3733_373312

theorem equation_solution : ∃ (x₁ x₂ : ℚ), x₁ = 7/4 ∧ x₂ = 1/4 ∧
  (16 * (x₁ - 1)^2 - 9 = 0) ∧ (16 * (x₂ - 1)^2 - 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3733_373312


namespace NUMINAMATH_CALUDE_unique_integer_set_l3733_373381

theorem unique_integer_set : ∃! (a b c : ℕ+), 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  (∃ k : ℕ+, 2 * a - 1 = k * b) ∧
  (∃ l : ℕ+, 2 * b - 1 = l * c) ∧
  (∃ m : ℕ+, 2 * c - 1 = m * a) ∧
  a = 13 ∧ b = 25 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_set_l3733_373381


namespace NUMINAMATH_CALUDE_speedster_fraction_l3733_373326

/-- Represents the inventory of vehicles -/
structure Inventory where
  speedsters : ℕ
  nonSpeedsters : ℕ

/-- The fraction of Speedsters that are convertibles -/
def convertibleFraction : ℚ := 3/5

/-- The number of Speedster convertibles -/
def speedsterConvertibles : ℕ := 54

/-- The number of non-Speedster vehicles -/
def nonSpeedsterCount : ℕ := 30

/-- Theorem: The fraction of Speedsters in the inventory is 3/4 -/
theorem speedster_fraction (inv : Inventory) 
  (h1 : inv.speedsters * convertibleFraction = speedsterConvertibles)
  (h2 : inv.nonSpeedsters = nonSpeedsterCount) :
  (inv.speedsters : ℚ) / (inv.speedsters + inv.nonSpeedsters) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_speedster_fraction_l3733_373326


namespace NUMINAMATH_CALUDE_sum_of_two_squares_l3733_373344

theorem sum_of_two_squares (n m : ℕ) (h : 2 * m = n^2 + 1) :
  ∃ k : ℕ, m = k^2 + (k - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_l3733_373344


namespace NUMINAMATH_CALUDE_quadratic_function_m_range_l3733_373300

theorem quadratic_function_m_range
  (a : ℝ) (m : ℝ) (y₁ y₂ : ℝ)
  (h_a_neg : a < 0)
  (h_y₁ : y₁ = a * m^2 - 4 * a * m)
  (h_y₂ : y₂ = a * (2*m)^2 - 4 * a * (2*m))
  (h_above_line : y₁ > -3*a ∧ y₂ > -3*a)
  (h_y₁_gt_y₂ : y₁ > y₂) :
  4/3 < m ∧ m < 3/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_m_range_l3733_373300


namespace NUMINAMATH_CALUDE_chipmunk_families_went_away_l3733_373350

theorem chipmunk_families_went_away (original : ℕ) (left : ℕ) (h1 : original = 86) (h2 : left = 21) :
  original - left = 65 := by
  sorry

end NUMINAMATH_CALUDE_chipmunk_families_went_away_l3733_373350


namespace NUMINAMATH_CALUDE_vector_problem_l3733_373390

def a : ℝ × ℝ := (1, 2)

theorem vector_problem (b : ℝ × ℝ) (θ : ℝ) :
  (b.1 ^ 2 + b.2 ^ 2 = 20) →
  (∃ (k : ℝ), b = k • a) →
  (b = (2, 4) ∨ b = (-2, -4)) ∧
  ((2 * a.1 - 3 * b.1) * (2 * a.1 + b.1) + (2 * a.2 - 3 * b.2) * (2 * a.2 + b.2) = -20) →
  θ = 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l3733_373390


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3733_373308

-- Define the universal set U
def U : Set ℝ := {x | x^2 ≤ 4}

-- Define set A
def A : Set ℝ := {x | |x + 1| ≤ 1}

-- State the theorem
theorem complement_of_A_in_U : 
  {x ∈ U | x ∉ A} = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3733_373308


namespace NUMINAMATH_CALUDE_money_collection_l3733_373311

theorem money_collection (households_per_day : ℕ) (days : ℕ) (total_amount : ℕ) :
  households_per_day = 20 →
  days = 5 →
  total_amount = 2000 →
  (households_per_day * days) / 2 * (total_amount / ((households_per_day * days) / 2)) = 40 :=
by sorry

end NUMINAMATH_CALUDE_money_collection_l3733_373311


namespace NUMINAMATH_CALUDE_first_shipment_cost_l3733_373306

/-- Represents the cost of a clothing shipment -/
def shipment_cost (num_sweaters num_jackets : ℕ) (sweater_price jacket_price : ℚ) : ℚ :=
  num_sweaters * sweater_price + num_jackets * jacket_price

theorem first_shipment_cost (sweater_price jacket_price : ℚ) :
  shipment_cost 5 15 sweater_price jacket_price = 550 →
  shipment_cost 10 20 sweater_price jacket_price = 1100 := by
  sorry

end NUMINAMATH_CALUDE_first_shipment_cost_l3733_373306


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l3733_373362

def n : ℕ := 245700

theorem sum_of_distinct_prime_factors :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (n + 1))) id : ℕ)
    = 30 := by sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l3733_373362


namespace NUMINAMATH_CALUDE_triangle_properties_l3733_373346

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  t.c^2 + t.a * t.b = t.c * (t.a * Real.cos t.B - t.b * Real.cos t.A) + 2 * t.b^2

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : t.c = 2 * Real.sqrt 3) : 
  t.C = π / 3 ∧ 
  ∃ (x : ℝ), -2 * Real.sqrt 3 < x ∧ x < 2 * Real.sqrt 3 ∧ x = 4 * Real.sin t.B - t.a :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3733_373346


namespace NUMINAMATH_CALUDE_jenny_investment_l3733_373375

/-- Jenny's investment problem -/
theorem jenny_investment (total : ℝ) (real_estate : ℝ) (mutual_funds : ℝ) 
  (h1 : total = 200000)
  (h2 : real_estate = 3 * mutual_funds)
  (h3 : total = real_estate + mutual_funds) :
  real_estate = 150000 := by
  sorry

end NUMINAMATH_CALUDE_jenny_investment_l3733_373375


namespace NUMINAMATH_CALUDE_quadratic_solution_l3733_373384

theorem quadratic_solution :
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ x = 2 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3733_373384


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l3733_373372

/-- For a quadratic equation x^2 + 2x + 4c = 0 to have two distinct real roots, c must be less than 1/4 -/
theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) →
  c < (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l3733_373372


namespace NUMINAMATH_CALUDE_units_digit_G_100_l3733_373340

-- Define G_n
def G (n : ℕ) : ℕ := 2^(5^n) + 1

-- Define the units digit function
def units_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_G_100 : units_digit (G 100) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_100_l3733_373340


namespace NUMINAMATH_CALUDE_tangent_fraction_equality_l3733_373354

theorem tangent_fraction_equality (α β : Real) 
  (h1 : Real.tan (α - β) = 2) 
  (h2 : Real.tan β = 4) : 
  (7 * Real.sin α - Real.cos α) / (7 * Real.sin α + Real.cos α) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_fraction_equality_l3733_373354


namespace NUMINAMATH_CALUDE_car_selection_average_l3733_373333

theorem car_selection_average (num_cars : ℕ) (num_clients : ℕ) (selections_per_client : ℕ) 
  (h1 : num_cars = 18) 
  (h2 : num_clients = 18) 
  (h3 : selections_per_client = 3) : 
  (num_clients * selections_per_client) / num_cars = 3 := by
  sorry

end NUMINAMATH_CALUDE_car_selection_average_l3733_373333


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3733_373305

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (72 - 18*x - x^2 = 0) → (∃ r s : ℝ, (72 - 18*r - r^2 = 0) ∧ (72 - 18*s - s^2 = 0) ∧ (r + s = 18)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3733_373305


namespace NUMINAMATH_CALUDE_arccos_cos_eq_two_thirds_x_l3733_373328

theorem arccos_cos_eq_two_thirds_x (x : Real) :
  0 ≤ x ∧ x ≤ (3 * Real.pi / 2) →
  (Real.arccos (Real.cos x) = 2 * x / 3) ↔ (x = 0 ∨ x = 6 * Real.pi / 5 ∨ x = 12 * Real.pi / 5) :=
by sorry

end NUMINAMATH_CALUDE_arccos_cos_eq_two_thirds_x_l3733_373328


namespace NUMINAMATH_CALUDE_bottles_per_case_l3733_373341

/-- Represents the number of bottles produced in a day -/
def total_bottles : ℕ := 120000

/-- Represents the number of cases required for one day's production -/
def total_cases : ℕ := 10000

/-- Theorem stating that the number of bottles per case is 12 -/
theorem bottles_per_case :
  total_bottles / total_cases = 12 := by
  sorry

end NUMINAMATH_CALUDE_bottles_per_case_l3733_373341


namespace NUMINAMATH_CALUDE_winner_received_55_percent_l3733_373330

/-- Represents an election with two candidates -/
structure Election where
  winner_votes : ℕ
  margin : ℕ

/-- Calculates the percentage of votes received by the winner -/
def winner_percentage (e : Election) : ℚ :=
  (e.winner_votes : ℚ) / ((e.winner_votes + (e.winner_votes - e.margin)) : ℚ) * 100

/-- Theorem stating that in the given election scenario, the winner received 55% of the votes -/
theorem winner_received_55_percent (e : Election) 
  (h1 : e.winner_votes = 550) 
  (h2 : e.margin = 100) : 
  winner_percentage e = 55 := by
  sorry

#eval winner_percentage ⟨550, 100⟩

end NUMINAMATH_CALUDE_winner_received_55_percent_l3733_373330


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3733_373373

/-- Given 2x + 3y + 5z = 29, the maximum value of √(2x+1) + √(3y+4) + √(5z+6) is 2√30 -/
theorem max_value_sqrt_sum (x y z : ℝ) (h : 2*x + 3*y + 5*z = 29) :
  (∀ a b c : ℝ, 2*a + 3*b + 5*c = 29 →
    Real.sqrt (2*a + 1) + Real.sqrt (3*b + 4) + Real.sqrt (5*c + 6) ≤
    Real.sqrt (2*x + 1) + Real.sqrt (3*y + 4) + Real.sqrt (5*z + 6)) →
  Real.sqrt (2*x + 1) + Real.sqrt (3*y + 4) + Real.sqrt (5*z + 6) = 2 * Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3733_373373


namespace NUMINAMATH_CALUDE_lydia_apple_eating_age_l3733_373394

/-- The age at which Lydia will eat an apple from her tree for the first time -/
def apple_eating_age (planting_age : ℕ) (years_to_bear_fruit : ℕ) : ℕ :=
  planting_age + years_to_bear_fruit

/-- Theorem stating Lydia's age when she first eats an apple from her tree -/
theorem lydia_apple_eating_age :
  apple_eating_age 4 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_lydia_apple_eating_age_l3733_373394


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3733_373376

theorem quadratic_no_real_roots (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + m ≠ 0) ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3733_373376


namespace NUMINAMATH_CALUDE_range_of_positive_integers_in_list_l3733_373361

def consecutive_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List ℤ) : List ℤ :=
  l.filter (λ x => x > 0)

def range_of_list (l : List ℤ) : ℤ :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem range_of_positive_integers_in_list :
  let d := consecutive_integers (-4) 12
  let positives := positive_integers d
  range_of_list positives = 6 := by
sorry

end NUMINAMATH_CALUDE_range_of_positive_integers_in_list_l3733_373361


namespace NUMINAMATH_CALUDE_dimas_age_l3733_373317

theorem dimas_age (dima_age brother_age sister_age : ℕ) : 
  dima_age = 2 * brother_age →
  dima_age = 3 * sister_age →
  (dima_age + brother_age + sister_age) / 3 = 11 →
  dima_age = 18 := by
sorry

end NUMINAMATH_CALUDE_dimas_age_l3733_373317


namespace NUMINAMATH_CALUDE_quadruple_solution_l3733_373382

-- Define the condition function
def condition (a b c d : ℝ) : Prop :=
  a + b * c * d = b + c * d * a ∧
  a + b * c * d = c + d * a * b ∧
  a + b * c * d = d + a * b * c

-- Define the solution set
def solution_set (a b c d : ℝ) : Prop :=
  (a = b ∧ b = c ∧ c = d) ∨
  (a = b ∧ c = d ∧ c = 1 / a ∧ a ≠ 0) ∨
  (a = 1 ∧ b = 1 ∧ c = 1) ∨
  (a = -1 ∧ b = -1 ∧ c = -1)

-- Theorem statement
theorem quadruple_solution (a b c d : ℝ) :
  condition a b c d → solution_set a b c d :=
sorry

end NUMINAMATH_CALUDE_quadruple_solution_l3733_373382


namespace NUMINAMATH_CALUDE_zhu_shijie_wine_problem_l3733_373368

/-- The amount of wine in the jug after visiting n taverns and meeting n friends -/
def wine_amount (initial : ℝ) (n : ℕ) : ℝ :=
  (2^n) * initial - (2^n - 1)

theorem zhu_shijie_wine_problem :
  ∃ (initial : ℝ), initial > 0 ∧ wine_amount initial 3 = 0 ∧ initial = 0.875 := by
  sorry

end NUMINAMATH_CALUDE_zhu_shijie_wine_problem_l3733_373368


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_plus_one_l3733_373398

theorem root_sum_reciprocal_plus_one (a b c : ℂ) : 
  (a^3 - a - 2 = 0) → (b^3 - b - 2 = 0) → (c^3 - c - 2 = 0) →
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 2) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_plus_one_l3733_373398


namespace NUMINAMATH_CALUDE_binomial_18_4_l3733_373342

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_4_l3733_373342


namespace NUMINAMATH_CALUDE_expression_equality_l3733_373309

theorem expression_equality : 150 * (150 - 4) - (150 * 150 - 8 + 2^3) = -600 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3733_373309


namespace NUMINAMATH_CALUDE_complex_difference_magnitude_l3733_373303

theorem complex_difference_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1)
  (h2 : Complex.abs z₂ = 1)
  (h3 : Complex.abs (z₁ + z₂) = Real.sqrt 3) :
  Complex.abs (z₁ - z₂) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_difference_magnitude_l3733_373303


namespace NUMINAMATH_CALUDE_greatest_power_of_three_l3733_373369

def v : ℕ := (List.range 30).foldl (· * ·) 1

theorem greatest_power_of_three (a : ℕ) : 
  (∀ k : ℕ, k ≤ 30 → k > 0 → v % 3^k = 0) → 
  (∀ m : ℕ, m > a → ¬(v % 3^m = 0)) → 
  a = 14 := by sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_l3733_373369


namespace NUMINAMATH_CALUDE_wendy_chocolate_sales_l3733_373301

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (unsold_bars : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Proves that Wendy made $18 from selling chocolate bars -/
theorem wendy_chocolate_sales : money_made 9 3 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_wendy_chocolate_sales_l3733_373301


namespace NUMINAMATH_CALUDE_point_on_line_l3733_373349

/-- A line in the xy-plane defined by two points -/
structure Line where
  x1 : ℚ
  y1 : ℚ
  x2 : ℚ
  y2 : ℚ

/-- Check if a point (x, y) lies on the given line -/
def Line.contains (l : Line) (x y : ℚ) : Prop :=
  (y - l.y1) * (l.x2 - l.x1) = (x - l.x1) * (l.y2 - l.y1)

theorem point_on_line (l : Line) (x : ℚ) :
  l.x1 = 1 ∧ l.y1 = 9 ∧ l.x2 = -2 ∧ l.y2 = -1 →
  l.contains x 2 →
  x = -11/10 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l3733_373349


namespace NUMINAMATH_CALUDE_tan_alpha_on_xy_eq_0_line_l3733_373324

-- Define the line x + y = 0
def line_xy_eq_0 (x y : ℝ) : Prop := x + y = 0

-- Define the terminal side of an angle
def terminal_side (α : ℝ) (x y : ℝ) : Prop := 
  ∃ (t : ℝ), t > 0 ∧ x = t * Real.cos α ∧ y = t * Real.sin α

-- Theorem statement
theorem tan_alpha_on_xy_eq_0_line (α : ℝ) : 
  (∃ (x y : ℝ), line_xy_eq_0 x y ∧ terminal_side α x y) → Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_on_xy_eq_0_line_l3733_373324


namespace NUMINAMATH_CALUDE_least_integer_in_ratio_l3733_373374

theorem least_integer_in_ratio (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = (3 * a) / 2 →
  c = (5 * a) / 2 →
  a + b + c = 60 →
  a = 12 :=
sorry

end NUMINAMATH_CALUDE_least_integer_in_ratio_l3733_373374


namespace NUMINAMATH_CALUDE_annas_money_l3733_373389

theorem annas_money (original spent remaining : ℚ) : 
  spent = (1 : ℚ) / 4 * original →
  remaining = (3 : ℚ) / 4 * original →
  remaining = 24 →
  original = 32 := by
sorry

end NUMINAMATH_CALUDE_annas_money_l3733_373389


namespace NUMINAMATH_CALUDE_milton_books_total_l3733_373385

/-- The number of zoology books Milton has -/
def zoology_books : ℕ := 16

/-- The number of botany books Milton has -/
def botany_books : ℕ := 4 * zoology_books

/-- The total number of books Milton has -/
def total_books : ℕ := zoology_books + botany_books

theorem milton_books_total : total_books = 80 := by
  sorry

end NUMINAMATH_CALUDE_milton_books_total_l3733_373385


namespace NUMINAMATH_CALUDE_elaine_rent_percentage_l3733_373313

/-- Elaine's earnings last year -/
def last_year_earnings : ℝ := 1

/-- Percentage of earnings spent on rent last year -/
def last_year_rent_percentage : ℝ := 20

/-- Earnings increase percentage this year -/
def earnings_increase : ℝ := 20

/-- Percentage of earnings spent on rent this year -/
def this_year_rent_percentage : ℝ := 30

/-- Increase in rent amount from last year to this year -/
def rent_increase : ℝ := 180

theorem elaine_rent_percentage :
  last_year_rent_percentage = 20 :=
by
  sorry

#check elaine_rent_percentage

end NUMINAMATH_CALUDE_elaine_rent_percentage_l3733_373313


namespace NUMINAMATH_CALUDE_polynomial_equality_l3733_373318

/-- Given that 2x^5 + 4x^3 + 3x + 4 + g(x) = x^4 - 2x^3 + 3,
    prove that g(x) = -2x^5 + x^4 - 6x^3 - 3x - 1 -/
theorem polynomial_equality (x : ℝ) (g : ℝ → ℝ) 
    (h : ∀ x, 2 * x^5 + 4 * x^3 + 3 * x + 4 + g x = x^4 - 2 * x^3 + 3) :
  g x = -2 * x^5 + x^4 - 6 * x^3 - 3 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3733_373318


namespace NUMINAMATH_CALUDE_max_cos_sin_sum_l3733_373366

open Real

theorem max_cos_sin_sum (α β γ : ℝ) (h1 : 0 < α ∧ α < π)
                                   (h2 : 0 < β ∧ β < π)
                                   (h3 : 0 < γ ∧ γ < π)
                                   (h4 : α + β + 2 * γ = π) :
  (∀ a b c, 0 < a ∧ a < π ∧ 0 < b ∧ b < π ∧ 0 < c ∧ c < π ∧ a + b + 2 * c = π →
    cos α + cos β + sin (2 * γ) ≥ cos a + cos b + sin (2 * c)) ∧
  cos α + cos β + sin (2 * γ) = 3 * sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_cos_sin_sum_l3733_373366


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3733_373336

theorem polynomial_simplification (w : ℝ) : 
  2 * w^2 + 3 - 4 * w^2 + 2 * w - 6 * w + 4 = -2 * w^2 - 4 * w + 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3733_373336


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3733_373315

/-- Given an inequality tx^2 - 6x + t^2 < 0 with solution set (-∞,a)∪(1,+∞), prove that a = -3 -/
theorem inequality_solution_set (t : ℝ) (a : ℝ) :
  (∀ x : ℝ, (t * x^2 - 6 * x + t^2 < 0) ↔ (x < a ∨ x > 1)) →
  a = -3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3733_373315


namespace NUMINAMATH_CALUDE_theorem_3_squeeze_theorem_l3733_373347

-- Theorem 3
theorem theorem_3 (v u : ℕ → ℝ) (n_0 : ℕ) 
  (h_v : ∀ ε > 0, ∃ N, ∀ n ≥ N, |v n| ≤ ε) 
  (h_u : ∀ n ≥ n_0, |u n| ≤ |v n|) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n| ≤ ε :=
sorry

-- Squeeze Theorem
theorem squeeze_theorem (u v w : ℕ → ℝ) (l : ℝ) (n_0 : ℕ)
  (h_u : ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - l| ≤ ε)
  (h_w : ∀ ε > 0, ∃ N, ∀ n ≥ N, |w n - l| ≤ ε)
  (h_v : ∀ n ≥ n_0, u n ≤ v n ∧ v n ≤ w n) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |v n - l| ≤ ε :=
sorry

end NUMINAMATH_CALUDE_theorem_3_squeeze_theorem_l3733_373347


namespace NUMINAMATH_CALUDE_function_upper_bound_l3733_373396

theorem function_upper_bound 
  (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ Set.Icc 0 1, f x ≥ 0)
  (h2 : f 1 = 1)
  (h3 : ∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≤ f x₁ + f x₂) :
  ∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x :=
by
  sorry


end NUMINAMATH_CALUDE_function_upper_bound_l3733_373396


namespace NUMINAMATH_CALUDE_attendants_with_both_tools_l3733_373357

theorem attendants_with_both_tools (pencil_users : ℕ) (pen_users : ℕ) (single_tool_users : ℕ) : 
  pencil_users = 25 →
  pen_users = 15 →
  single_tool_users = 20 →
  pencil_users + pen_users - single_tool_users = 10 := by
sorry

end NUMINAMATH_CALUDE_attendants_with_both_tools_l3733_373357


namespace NUMINAMATH_CALUDE_cards_thrown_away_l3733_373338

theorem cards_thrown_away (cards_per_deck : ℕ) (half_full_decks : ℕ) (full_decks : ℕ) (remaining_cards : ℕ) : 
  cards_per_deck = 52 →
  half_full_decks = 3 →
  full_decks = 3 →
  remaining_cards = 200 →
  (cards_per_deck * full_decks + (cards_per_deck / 2) * half_full_decks) - remaining_cards = 34 :=
by sorry

end NUMINAMATH_CALUDE_cards_thrown_away_l3733_373338


namespace NUMINAMATH_CALUDE_children_events_count_l3733_373363

theorem children_events_count (cupcakes_per_event : ℝ) (total_cupcakes : ℕ) 
  (h1 : cupcakes_per_event = 96.0)
  (h2 : total_cupcakes = 768) :
  (total_cupcakes : ℝ) / cupcakes_per_event = 8 := by
  sorry

end NUMINAMATH_CALUDE_children_events_count_l3733_373363


namespace NUMINAMATH_CALUDE_apps_files_difference_l3733_373364

/-- Given Dave's initial and final numbers of apps and files on his phone, prove that he has 7 more apps than files left. -/
theorem apps_files_difference (initial_apps initial_files final_apps final_files : ℕ) :
  initial_apps = 24 →
  initial_files = 9 →
  final_apps = 12 →
  final_files = 5 →
  final_apps - final_files = 7 := by
  sorry

end NUMINAMATH_CALUDE_apps_files_difference_l3733_373364


namespace NUMINAMATH_CALUDE_complex_triplet_theorem_l3733_373367

theorem complex_triplet_theorem (a b c : ℂ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  Complex.abs a = Complex.abs b → 
  Complex.abs b = Complex.abs c → 
  a / b + b / c + c / a = -1 → 
  ((a = b ∧ c = -a) ∨ (b = c ∧ a = -b) ∨ (c = a ∧ b = -c)) := by
sorry

end NUMINAMATH_CALUDE_complex_triplet_theorem_l3733_373367


namespace NUMINAMATH_CALUDE_angle_equivalence_l3733_373327

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
variable (h1 : A.2 = B.2 ∧ B.2 = C.2 ∧ C.2 = D.2)  -- A, B, C, D are on the same line
variable (h2 : A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1)  -- A, B, C, D are in that order
variable (h3 : dist A B = dist C D)  -- AB = CD
variable (h4 : E.2 ≠ A.2)  -- E is off the line
variable (h5 : dist C E = dist D E)  -- CE = DE

-- Define the angle function
def angle (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem angle_equivalence :
  angle C E D = 2 * angle A E B ↔ dist A C = dist E C :=
sorry

end NUMINAMATH_CALUDE_angle_equivalence_l3733_373327


namespace NUMINAMATH_CALUDE_quadratic_equation_satisfaction_l3733_373316

theorem quadratic_equation_satisfaction (p q : ℝ) : 
  p^2 + 9*q^2 + 3*p - p*q = 30 ∧ p - 5*q - 8 = 0 → p^2 - p - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_satisfaction_l3733_373316


namespace NUMINAMATH_CALUDE_carnation_percentage_l3733_373393

/-- Represents a bouquet of flowers -/
structure Bouquet where
  total : ℝ
  pink : ℝ
  red : ℝ
  pink_roses : ℝ
  pink_carnations : ℝ
  red_roses : ℝ
  red_carnations : ℝ

/-- The theorem stating the percentage of carnations in the bouquet -/
theorem carnation_percentage (b : Bouquet) : 
  b.pink + b.red = b.total →
  b.pink_roses + b.pink_carnations = b.pink →
  b.red_roses + b.red_carnations = b.red →
  b.pink_roses = b.pink / 2 →
  b.red_carnations = b.red * 2 / 3 →
  b.pink = b.total * 7 / 10 →
  (b.pink_carnations + b.red_carnations) / b.total = 11 / 20 := by
sorry

end NUMINAMATH_CALUDE_carnation_percentage_l3733_373393


namespace NUMINAMATH_CALUDE_modulus_of_z_l3733_373348

theorem modulus_of_z (z : ℂ) : z = Complex.I * (2 - Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3733_373348


namespace NUMINAMATH_CALUDE_max_clerks_results_l3733_373371

theorem max_clerks_results (initial_count : ℕ) (operation_count : ℕ) 
  (h1 : initial_count = 100)
  (h2 : operation_count = initial_count - 1) :
  ∃ (max_results : ℕ), max_results = operation_count / 2 + 1 ∧ 
  max_results = 51 := by
  sorry

end NUMINAMATH_CALUDE_max_clerks_results_l3733_373371


namespace NUMINAMATH_CALUDE_alpha_beta_range_l3733_373355

-- Define the curve E
def curve_E (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the line l1
def line_l1 (k x y : ℝ) : Prop := y = k * (x + 2)

-- Define the intersection points A and B
def intersection_points (x1 y1 x2 y2 k : ℝ) : Prop :=
  curve_E x1 y1 ∧ curve_E x2 y2 ∧ 
  line_l1 k x1 y1 ∧ line_l1 k x2 y2 ∧
  x1 ≠ x2

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the relationship between α, β, and the points
def alpha_beta_relation (α β x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : Prop :=
  α = (1 - x1) / (x3 - 1) ∧
  β = (1 - x2) / (x4 - 1) ∧
  curve_E x3 y3 ∧ curve_E x4 y4

-- Main theorem
theorem alpha_beta_range :
  ∀ (k x1 y1 x2 y2 x3 y3 x4 y4 α β : ℝ),
    0 < k^2 ∧ k^2 < 1/2 →
    intersection_points x1 y1 x2 y2 k →
    alpha_beta_relation α β x1 y1 x2 y2 x3 y3 x4 y4 →
    6 < α + β ∧ α + β < 10 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_range_l3733_373355


namespace NUMINAMATH_CALUDE_greatest_piece_length_l3733_373343

theorem greatest_piece_length (rope1 rope2 rope3 max_length : ℕ) 
  (h1 : rope1 = 48)
  (h2 : rope2 = 72)
  (h3 : rope3 = 120)
  (h4 : max_length = 24) : 
  (Nat.gcd rope1 (Nat.gcd rope2 rope3) ≤ max_length ∧ 
   Nat.gcd rope1 (Nat.gcd rope2 rope3) = max_length) := by
  sorry

#eval Nat.gcd 48 (Nat.gcd 72 120)

end NUMINAMATH_CALUDE_greatest_piece_length_l3733_373343


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3733_373304

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = 2017 * 2016^n - 2018 * t) →
  (∀ n, S (n + 1) - S n = a (n + 1)) →
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →
  t = 2017 / 2018 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3733_373304


namespace NUMINAMATH_CALUDE_average_leaves_per_hour_l3733_373335

/-- Represents the leaf fall pattern of a tree over 3 hours -/
structure TreeLeafFall where
  hour1 : ℕ
  hour2 : ℕ
  hour3 : ℕ

/-- Calculates the total number of leaves that fell from a tree -/
def totalLeaves (tree : TreeLeafFall) : ℕ :=
  tree.hour1 + tree.hour2 + tree.hour3

/-- Represents the leaf fall patterns of two trees in Rylee's backyard -/
def ryleesBackyard : (TreeLeafFall × TreeLeafFall) :=
  (⟨7, 12, 9⟩, ⟨4, 4, 6⟩)

/-- The number of hours of observation -/
def observationHours : ℕ := 3

/-- Theorem: The average number of leaves falling per hour across both trees is 14 -/
theorem average_leaves_per_hour :
  (totalLeaves ryleesBackyard.1 + totalLeaves ryleesBackyard.2) / observationHours = 14 :=
by sorry

end NUMINAMATH_CALUDE_average_leaves_per_hour_l3733_373335


namespace NUMINAMATH_CALUDE_tangent_line_at_one_a_range_when_f_negative_l3733_373331

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem tangent_line_at_one (h : ℝ → ℝ := f 2) :
  ∃ (m b : ℝ), ∀ x y, y = m * (x - 1) + h 1 ↔ x + y + 1 = 0 :=
sorry

theorem a_range_when_f_negative (a : ℝ) :
  (∀ x > 0, f a x < 0) → a > Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_a_range_when_f_negative_l3733_373331


namespace NUMINAMATH_CALUDE_system_of_equations_l3733_373314

theorem system_of_equations (y : ℝ) :
  ∃ (x z : ℝ),
    (19 * (x + y) + 17 = 19 * (-x + y) - 21) ∧
    (5 * x - 3 * z = 11 * y - 7) ∧
    (x = -1) ∧
    (z = -11 * y / 3 + 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l3733_373314


namespace NUMINAMATH_CALUDE_wire_ratio_l3733_373377

theorem wire_ratio (x y : ℝ) : 
  x > 0 → y > 0 → 
  (4 * (x / 4) = 5 * (y / 5)) → 
  x / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_l3733_373377


namespace NUMINAMATH_CALUDE_pythagorean_equivalent_l3733_373307

theorem pythagorean_equivalent (t : ℝ) : 
  (∃ (a b : ℚ), (2 * t) / (1 + t^2) = a ∧ (1 - t^2) / (1 + t^2) = b) → 
  ∃ (q : ℚ), (t : ℝ) = q :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_equivalent_l3733_373307


namespace NUMINAMATH_CALUDE_follower_point_coords_follower_on_axis_follower_distance_l3733_373321

-- Define a-level follower point
def a_level_follower (a : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (x + a * y, a * x + y)

-- Statement 1
theorem follower_point_coords : a_level_follower 3 (-3, 5) = (12, -4) := by sorry

-- Statement 2
theorem follower_on_axis (c : ℝ) : 
  (∃ x y, a_level_follower (-3) (c, 2*c + 2) = (x, y) ∧ (x = 0 ∨ y = 0)) →
  a_level_follower (-3) (c, 2*c + 2) = (-16, 0) ∨ 
  a_level_follower (-3) (c, 2*c + 2) = (0, 16/5) := by sorry

-- Statement 3
theorem follower_distance (x : ℝ) (a : ℝ) :
  x > 0 →
  let P : ℝ × ℝ := (x, 0)
  let P3 := a_level_follower a P
  let PP3_length := Real.sqrt ((P3.1 - P.1)^2 + (P3.2 - P.2)^2)
  let OP_length := Real.sqrt (P.1^2 + P.2^2)
  PP3_length = 2 * OP_length →
  a = 2 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_follower_point_coords_follower_on_axis_follower_distance_l3733_373321


namespace NUMINAMATH_CALUDE_prob_at_least_75_cents_is_correct_l3733_373388

-- Define the coin types and their quantities
structure CoinBox :=
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)
  (quarters : Nat)

-- Define the function to calculate the total number of coins
def totalCoins (box : CoinBox) : Nat :=
  box.pennies + box.nickels + box.dimes + box.quarters

-- Define the function to calculate the number of ways to choose 7 coins
def waysToChoose7 (box : CoinBox) : Nat :=
  Nat.choose (totalCoins box) 7

-- Define the probability of drawing coins worth at least 75 cents
def probAtLeast75Cents (box : CoinBox) : Rat :=
  2450 / waysToChoose7 box

-- State the theorem
theorem prob_at_least_75_cents_is_correct (box : CoinBox) :
  box.pennies = 4 ∧ box.nickels = 5 ∧ box.dimes = 7 ∧ box.quarters = 3 →
  probAtLeast75Cents box = 2450 / 50388 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_75_cents_is_correct_l3733_373388


namespace NUMINAMATH_CALUDE_exactly_four_pairs_l3733_373358

/-- A line is tangent to a circle if and only if the distance from the center
    of the circle to the line equals the radius of the circle. -/
def is_tangent_line (m : ℕ) (n : ℕ) : Prop :=
  2^m = 2*n

/-- The condition that n and m are positive integers with n - m < 5 -/
def satisfies_condition (m : ℕ) (n : ℕ) : Prop :=
  0 < m ∧ 0 < n ∧ n < m + 5

/-- The main theorem stating that there are exactly 4 pairs (m, n) satisfying
    both the tangency condition and the inequality condition -/
theorem exactly_four_pairs :
  ∃! (s : Finset (ℕ × ℕ)),
    s.card = 4 ∧
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      (is_tangent_line p.1 p.2 ∧ satisfies_condition p.1 p.2)) := by
  sorry

end NUMINAMATH_CALUDE_exactly_four_pairs_l3733_373358


namespace NUMINAMATH_CALUDE_chess_team_arrangements_l3733_373325

def chess_team_size : ℕ := 6
def num_boys : ℕ := 3
def num_girls : ℕ := 3

def arrangements_count : ℕ := sorry

theorem chess_team_arrangements :
  chess_team_size = num_boys + num_girls →
  arrangements_count = 144 := by sorry

end NUMINAMATH_CALUDE_chess_team_arrangements_l3733_373325


namespace NUMINAMATH_CALUDE_line_translation_l3733_373310

/-- Given a line y = -2x + 1, translating it upwards by 2 units results in y = -2x + 3 -/
theorem line_translation (x y : ℝ) : 
  (y = -2*x + 1) → (y + 2 = -2*x + 3) := by sorry

end NUMINAMATH_CALUDE_line_translation_l3733_373310
