import Mathlib

namespace NUMINAMATH_CALUDE_no_power_of_two_solution_l3177_317761

theorem no_power_of_two_solution : ¬∃ (a b c k : ℕ), 
  a + b + c = 1001 ∧ 27 * a + 14 * b + c = 2^k :=
by sorry

end NUMINAMATH_CALUDE_no_power_of_two_solution_l3177_317761


namespace NUMINAMATH_CALUDE_total_glows_is_569_l3177_317704

/-- The number of seconds between 1:57:58 am and 3:20:47 am -/
def time_duration : ℕ := 4969

/-- The interval at which Light A glows, in seconds -/
def light_a_interval : ℕ := 16

/-- The interval at which Light B glows, in seconds -/
def light_b_interval : ℕ := 35

/-- The interval at which Light C glows, in seconds -/
def light_c_interval : ℕ := 42

/-- The number of times Light A glows -/
def light_a_glows : ℕ := time_duration / light_a_interval

/-- The number of times Light B glows -/
def light_b_glows : ℕ := time_duration / light_b_interval

/-- The number of times Light C glows -/
def light_c_glows : ℕ := time_duration / light_c_interval

/-- The total number of glows for all light sources combined -/
def total_glows : ℕ := light_a_glows + light_b_glows + light_c_glows

theorem total_glows_is_569 : total_glows = 569 := by
  sorry

end NUMINAMATH_CALUDE_total_glows_is_569_l3177_317704


namespace NUMINAMATH_CALUDE_trapezium_height_l3177_317773

theorem trapezium_height (a b h : ℝ) : 
  a > 0 → b > 0 → h > 0 →
  a = 20 → b = 18 → 
  (1/2) * (a + b) * h = 190 →
  h = 10 := by
sorry

end NUMINAMATH_CALUDE_trapezium_height_l3177_317773


namespace NUMINAMATH_CALUDE_polynomial_form_theorem_l3177_317708

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that the polynomial P satisfies for all real a, b, c -/
def SatisfiesCondition (P : RealPolynomial) : Prop :=
  ∀ (a b c : ℝ), a*b + b*c + c*a = 0 → 
    P (a-b) + P (b-c) + P (c-a) = 2 * P (a+b+c)

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem polynomial_form_theorem (P : RealPolynomial) 
  (h : SatisfiesCondition P) : 
  ∃ (α β : ℝ), ∀ (x : ℝ), P x = α * x^4 + β * x^2 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_form_theorem_l3177_317708


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3177_317734

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 4) (h3 : x ≠ -2) :
  (x^2 + 4*x + 11) / ((x - 1)*(x - 4)*(x + 2)) = 
  (-16/9) / (x - 1) + (35/18) / (x - 4) + (1/6) / (x + 2) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3177_317734


namespace NUMINAMATH_CALUDE_treats_per_day_l3177_317760

def treat_cost : ℚ := 1 / 10
def total_cost : ℚ := 6
def days_in_month : ℕ := 30

theorem treats_per_day :
  (total_cost / treat_cost) / days_in_month = 2 := by sorry

end NUMINAMATH_CALUDE_treats_per_day_l3177_317760


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3177_317737

def f (x : ℝ) : ℝ := 8*x^5 - 10*x^4 + 3*x^3 + 5*x^2 - 7*x - 35

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := by sorry

theorem polynomial_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, f x = (x - 5) * q x + 19180 := by
  have h := remainder_theorem f 5
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3177_317737


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_is_correct_l3177_317711

def unfair_die_expected_value (p1 p2 p3 p4 p5 p6 p7 p8 : ℚ) : Prop :=
  p1 = 1/15 ∧ p2 = 1/15 ∧ p3 = 1/15 ∧ p4 = 1/15 ∧ p5 = 1/15 ∧ p6 = 1/15 ∧ p7 = 1/6 ∧ p8 = 1/3 →
  1 * p1 + 2 * p2 + 3 * p3 + 4 * p4 + 5 * p5 + 6 * p6 + 7 * p7 + 8 * p8 = 157/30

theorem unfair_die_expected_value_is_correct :
  unfair_die_expected_value (1/15) (1/15) (1/15) (1/15) (1/15) (1/15) (1/6) (1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_unfair_die_expected_value_is_correct_l3177_317711


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l3177_317749

theorem pure_imaginary_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ k : ℝ, (3 - 5*Complex.I) * (a + b*Complex.I) * (1 + 2*Complex.I) = k * Complex.I) →
  a / b = -1 / 7 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l3177_317749


namespace NUMINAMATH_CALUDE_pete_calculation_l3177_317768

theorem pete_calculation (x y z : ℕ+) : 
  (x + y) * z = 14 ∧ 
  x * y + z = 14 → 
  ∃ (s : Finset ℕ+), s.card = 4 ∧ ∀ a : ℕ+, a ∈ s ↔ 
    ∃ (b c : ℕ+), ((a + b) * c = 14 ∧ a * b + c = 14) := by
  sorry

end NUMINAMATH_CALUDE_pete_calculation_l3177_317768


namespace NUMINAMATH_CALUDE_range_of_m_l3177_317743

/-- The set of x satisfying the condition p -/
def P : Set ℝ := {x | (x + 2) / (x - 10) ≤ 0}

/-- The set of x satisfying the condition q for a given m -/
def Q (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 < 0}

/-- p is a necessary but not sufficient condition for q -/
def NecessaryNotSufficient (m : ℝ) : Prop :=
  (∀ x, x ∈ Q m → x ∈ P) ∧ (∃ x, x ∈ P ∧ x ∉ Q m)

/-- The main theorem stating the range of m -/
theorem range_of_m :
  ∀ m, m > 0 → (NecessaryNotSufficient m ↔ m < 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3177_317743


namespace NUMINAMATH_CALUDE_total_apples_l3177_317726

/-- Represents the number of apples Tessa has -/
def tessas_apples : ℕ := 4

/-- Represents the number of apples Anita gave to Tessa -/
def anitas_gift : ℕ := 5

/-- Theorem stating that Tessa's total apples is the sum of her initial apples and Anita's gift -/
theorem total_apples : tessas_apples + anitas_gift = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l3177_317726


namespace NUMINAMATH_CALUDE_fast_food_cost_l3177_317703

/-- The cost of items at a fast food restaurant -/
theorem fast_food_cost (H M F : ℝ) : 
  (3 * H + 5 * M + F = 23.50) → 
  (5 * H + 9 * M + F = 39.50) → 
  (2 * H + 2 * M + 2 * F = 15.00) :=
by sorry

end NUMINAMATH_CALUDE_fast_food_cost_l3177_317703


namespace NUMINAMATH_CALUDE_eighth_row_interior_sum_l3177_317740

/-- Sum of all elements in the n-th row of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^(n-1)

/-- Sum of interior numbers in the n-th row of Pascal's Triangle -/
def pascal_interior_sum (n : ℕ) : ℕ := pascal_row_sum n - 2

theorem eighth_row_interior_sum :
  pascal_interior_sum 8 = 126 := by sorry

end NUMINAMATH_CALUDE_eighth_row_interior_sum_l3177_317740


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_sum_l3177_317790

theorem largest_divisor_of_consecutive_sum (a : ℤ) : 
  ∃ (d : ℤ), d > 0 ∧ d ∣ (a - 1 + a + a + 1) ∧ 
  ∀ (k : ℤ), k > d → ∃ (n : ℤ), ¬(k ∣ (n - 1 + n + n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_sum_l3177_317790


namespace NUMINAMATH_CALUDE_total_fruits_picked_l3177_317744

theorem total_fruits_picked (joan_oranges sara_oranges carlos_oranges 
                             alyssa_pears ben_pears vanessa_pears 
                             tim_apples linda_apples : ℕ) 
                            (h1 : joan_oranges = 37)
                            (h2 : sara_oranges = 10)
                            (h3 : carlos_oranges = 25)
                            (h4 : alyssa_pears = 30)
                            (h5 : ben_pears = 40)
                            (h6 : vanessa_pears = 20)
                            (h7 : tim_apples = 15)
                            (h8 : linda_apples = 10) :
  joan_oranges + sara_oranges + carlos_oranges + 
  alyssa_pears + ben_pears + vanessa_pears + 
  tim_apples + linda_apples = 187 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_picked_l3177_317744


namespace NUMINAMATH_CALUDE_grouping_theorem_l3177_317784

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to divide 4 men and 3 women into a group of five 
    (with at least two men and two women) and a group of two -/
def groupingWays : ℕ :=
  choose 4 2 * choose 3 2 * choose 3 1

theorem grouping_theorem : groupingWays = 54 := by sorry

end NUMINAMATH_CALUDE_grouping_theorem_l3177_317784


namespace NUMINAMATH_CALUDE_min_buses_required_l3177_317727

theorem min_buses_required (total_students : ℕ) (bus_capacity : ℕ) (h1 : total_students = 325) (h2 : bus_capacity = 45) :
  ∃ (n : ℕ), n * bus_capacity ≥ total_students ∧ ∀ m : ℕ, m * bus_capacity ≥ total_students → m ≥ n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_buses_required_l3177_317727


namespace NUMINAMATH_CALUDE_b_value_when_a_is_4_l3177_317778

/-- The inverse relationship between a^3 and √b -/
def inverse_relation (a b : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a^3 * Real.sqrt b = k

theorem b_value_when_a_is_4 (a b : ℝ) :
  inverse_relation 3 64 →
  inverse_relation a b →
  a = 4 →
  a * Real.sqrt b = 24 →
  b = 11.390625 := by
  sorry

end NUMINAMATH_CALUDE_b_value_when_a_is_4_l3177_317778


namespace NUMINAMATH_CALUDE_first_condition_second_condition_l3177_317741

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem for the first condition
theorem first_condition (a : ℝ) : 
  (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) → a = -2 := by sorry

-- Theorem for the second condition
theorem second_condition (a : ℝ) :
  (A a ∩ B = A a ∩ C) ∧ (A a ∩ B ≠ ∅) → a = -3 := by sorry

end NUMINAMATH_CALUDE_first_condition_second_condition_l3177_317741


namespace NUMINAMATH_CALUDE_min_non_red_surface_fraction_for_specific_cube_l3177_317720

/-- Represents a cube with given edge length and colored subcubes -/
structure ColoredCube where
  edge_length : ℕ
  red_cubes : ℕ
  white_cubes : ℕ
  blue_cubes : ℕ

/-- Calculate the minimum non-red surface area fraction of a ColoredCube -/
def min_non_red_surface_fraction (c : ColoredCube) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem min_non_red_surface_fraction_for_specific_cube :
  let c := ColoredCube.mk 4 48 12 4
  min_non_red_surface_fraction c = 1/8 := by sorry

end NUMINAMATH_CALUDE_min_non_red_surface_fraction_for_specific_cube_l3177_317720


namespace NUMINAMATH_CALUDE_min_value_sum_product_l3177_317724

theorem min_value_sum_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l3177_317724


namespace NUMINAMATH_CALUDE_set_union_problem_l3177_317717

theorem set_union_problem (S T : Set ℕ) (h1 : S = {0, 1}) (h2 : T = {0}) :
  S ∪ T = {0, 1} := by sorry

end NUMINAMATH_CALUDE_set_union_problem_l3177_317717


namespace NUMINAMATH_CALUDE_negation_of_implication_l3177_317798

theorem negation_of_implication (p q : Prop) : ¬(p → q) ↔ (p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3177_317798


namespace NUMINAMATH_CALUDE_set_intersection_implies_values_l3177_317742

theorem set_intersection_implies_values (a b : ℤ) : 
  let A : Set ℤ := {1, b, a + b}
  let B : Set ℤ := {a - b, a * b}
  A ∩ B = {-1, 0} →
  a = -1 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_implies_values_l3177_317742


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3177_317701

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, 7) and (4, -3) is 9 -/
theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ := (10, 7)
  let p₂ : ℝ × ℝ := (4, -3)
  let midpoint : ℝ × ℝ := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint.1 + midpoint.2 : ℝ) = 9 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3177_317701


namespace NUMINAMATH_CALUDE_impossible_all_tails_l3177_317745

/-- Represents a 4x4 grid of binary values -/
def Grid := Matrix (Fin 4) (Fin 4) Bool

/-- Represents the possible flip operations -/
inductive FlipOperation
| Row : Fin 4 → FlipOperation
| Column : Fin 4 → FlipOperation
| Diagonal : Bool → Fin 4 → FlipOperation

/-- Initial configuration of the grid -/
def initialGrid : Grid :=
  Matrix.of (fun i j => if i = 0 ∧ j < 2 then true else false)

/-- Applies a flip operation to the grid -/
def applyFlip (g : Grid) (op : FlipOperation) : Grid :=
  sorry

/-- Checks if all values in the grid are false (tails) -/
def allTails (g : Grid) : Prop :=
  ∀ i j, g i j = false

/-- Main theorem: It's impossible to reach all tails from the initial configuration -/
theorem impossible_all_tails :
  ¬∃ (ops : List FlipOperation), allTails (ops.foldl applyFlip initialGrid) :=
  sorry

end NUMINAMATH_CALUDE_impossible_all_tails_l3177_317745


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l3177_317723

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length breadth cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Theorem: The total cost of fencing a rectangular plot with given dimensions -/
theorem fencing_cost_theorem (length breadth cost_per_meter : ℝ) 
  (h1 : length = 62)
  (h2 : breadth = length - 24)
  (h3 : cost_per_meter = 26.5) :
  total_fencing_cost length breadth cost_per_meter = 5300 :=
by
  sorry

#check fencing_cost_theorem

end NUMINAMATH_CALUDE_fencing_cost_theorem_l3177_317723


namespace NUMINAMATH_CALUDE_at_least_one_side_not_exceeding_double_l3177_317716

-- Define a structure for a parallelogram
structure Parallelogram :=
  (side1 : ℝ)
  (side2 : ℝ)
  (area : ℝ)

-- Define the problem setup
def parallelogram_inscriptions (P1 P2 P3 : Parallelogram) : Prop :=
  -- P2 is inscribed in P1
  P2.area < P1.area ∧
  -- P3 is inscribed in P2
  P3.area < P2.area ∧
  -- The sides of P3 are parallel to the sides of P1
  (P3.side1 < P1.side1 ∧ P3.side2 < P1.side2)

-- Theorem statement
theorem at_least_one_side_not_exceeding_double :
  ∀ (P1 P2 P3 : Parallelogram),
  parallelogram_inscriptions P1 P2 P3 →
  (P1.side1 ≤ 2 * P3.side1 ∨ P1.side2 ≤ 2 * P3.side2) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_side_not_exceeding_double_l3177_317716


namespace NUMINAMATH_CALUDE_complementary_angle_theorem_l3177_317758

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary_angles (α β : ℝ) : Prop := α + β = 90

/-- Given complementary angles α and β where α = 40°, prove that β = 50° -/
theorem complementary_angle_theorem (α β : ℝ) 
  (h1 : complementary_angles α β) (h2 : α = 40) : β = 50 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_theorem_l3177_317758


namespace NUMINAMATH_CALUDE_even_quadratic_sum_l3177_317752

/-- A quadratic function f(x) = ax^2 + bx defined on [-1, 2] -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

/-- The property of f being an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The theorem stating that if f is even, then a + b = 1/3 -/
theorem even_quadratic_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f a b x = f a b (-x)) →
  a + b = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_even_quadratic_sum_l3177_317752


namespace NUMINAMATH_CALUDE_pie_remainder_l3177_317794

/-- Proves that given Carlos took 60% of a whole pie and Sophia took a quarter of the remainder, 
    the portion of the whole pie left is 30%. -/
theorem pie_remainder (whole_pie : ℝ) (carlos_share sophia_share remainder : ℝ) : 
  carlos_share = 0.6 * whole_pie →
  sophia_share = 0.25 * (whole_pie - carlos_share) →
  remainder = whole_pie - carlos_share - sophia_share →
  remainder = 0.3 * whole_pie :=
by sorry

end NUMINAMATH_CALUDE_pie_remainder_l3177_317794


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3177_317797

/-- Quadratic function passing through (2,3) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 2) * x + 3

theorem quadratic_function_properties :
  ∃ a : ℝ,
  (f a 2 = 3) ∧
  (∀ x : ℝ, 0 < x → x < 3 → 2 ≤ f 0 x ∧ f 0 x < 6) ∧
  (∀ m y₁ y₂ : ℝ, f 0 (m - 1) = y₁ → f 0 m = y₂ → y₁ > y₂ → m < 3/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3177_317797


namespace NUMINAMATH_CALUDE_john_business_venture_result_l3177_317738

structure Currency where
  name : String
  exchange_rate : ℚ

structure Item where
  name : String
  currency : Currency
  purchase_price : ℚ
  sale_percentage : ℚ
  tax_rate : ℚ

def calculate_profit_or_loss (items : List Item) : ℚ :=
  sorry

theorem john_business_venture_result 
  (grinder : Item)
  (mobile_phone : Item)
  (refrigerator : Item)
  (television : Item)
  (h_grinder : grinder = { 
    name := "Grinder", 
    currency := { name := "INR", exchange_rate := 1 },
    purchase_price := 15000,
    sale_percentage := -4/100,
    tax_rate := 5/100
  })
  (h_mobile_phone : mobile_phone = {
    name := "Mobile Phone",
    currency := { name := "USD", exchange_rate := 75 },
    purchase_price := 100,
    sale_percentage := 10/100,
    tax_rate := 7/100
  })
  (h_refrigerator : refrigerator = {
    name := "Refrigerator",
    currency := { name := "GBP", exchange_rate := 101 },
    purchase_price := 200,
    sale_percentage := 8/100,
    tax_rate := 6/100
  })
  (h_television : television = {
    name := "Television",
    currency := { name := "EUR", exchange_rate := 90 },
    purchase_price := 300,
    sale_percentage := -6/100,
    tax_rate := 9/100
  }) :
  calculate_profit_or_loss [grinder, mobile_phone, refrigerator, television] = -346/100 :=
sorry

end NUMINAMATH_CALUDE_john_business_venture_result_l3177_317738


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3177_317739

theorem fraction_subtraction (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3 * x^2 - 2 * x + 1) / ((x + 2) * (x - 3)) - (x^2 - 5 * x + 6) / ((x + 2) * (x - 3)) =
  (2 * x^2 + 3 * x - 5) / ((x + 2) * (x - 3)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3177_317739


namespace NUMINAMATH_CALUDE_specific_trapezoid_dimensions_l3177_317735

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedIsoscelesTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The angle at the base of the trapezoid -/
  baseAngle : ℝ
  /-- The length of the shorter base -/
  shorterBase : ℝ
  /-- The length of the longer base -/
  longerBase : ℝ
  /-- The length of the legs (equal for isosceles trapezoid) -/
  legLength : ℝ

/-- The theorem about the specific trapezoid -/
theorem specific_trapezoid_dimensions (t : CircumscribedIsoscelesTrapezoid) 
  (h_area : t.area = 8)
  (h_angle : t.baseAngle = π / 6) :
  t.shorterBase = 4 - 2 * Real.sqrt 3 ∧ 
  t.longerBase = 4 + 2 * Real.sqrt 3 ∧ 
  t.legLength = 4 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_dimensions_l3177_317735


namespace NUMINAMATH_CALUDE_triangle_equilateral_if_angles_arithmetic_and_geometric_l3177_317769

theorem triangle_equilateral_if_angles_arithmetic_and_geometric :
  ∀ (a b c : ℝ),
  -- The angles form an arithmetic sequence
  (∃ d : ℝ, b = a + d ∧ c = b + d) →
  -- The angles form a geometric sequence
  (∃ r : ℝ, b = a * r ∧ c = b * r) →
  -- The sum of angles is 180°
  a + b + c = 180 →
  -- The triangle is equilateral (all angles are equal)
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_triangle_equilateral_if_angles_arithmetic_and_geometric_l3177_317769


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3177_317777

theorem trigonometric_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.cos α - Real.sin α = -Real.sqrt 5 / 5) : 
  (Real.sin α * Real.cos α = 2 / 5) ∧ 
  (Real.sin α + Real.cos α = 3 * Real.sqrt 5 / 5) ∧ 
  ((2 * Real.sin α * Real.cos α - Real.cos α + 1) / (1 - Real.tan α) = (-9 + Real.sqrt 5) / 5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3177_317777


namespace NUMINAMATH_CALUDE_promotion_savings_difference_l3177_317774

/-- Represents the cost of a pair of shoes in dollars -/
def shoe_cost : ℝ := 50

/-- Calculates the cost of two pairs of shoes using Promotion X -/
def cost_promotion_x : ℝ :=
  shoe_cost + (shoe_cost * (1 - 0.4))

/-- Calculates the cost of two pairs of shoes using Promotion Y -/
def cost_promotion_y : ℝ :=
  shoe_cost + (shoe_cost - 15)

/-- Theorem: The difference in cost between Promotion Y and Promotion X is $5 -/
theorem promotion_savings_difference :
  cost_promotion_y - cost_promotion_x = 5 := by
  sorry

end NUMINAMATH_CALUDE_promotion_savings_difference_l3177_317774


namespace NUMINAMATH_CALUDE_cubic_integer_root_l3177_317764

theorem cubic_integer_root 
  (b c : ℚ) 
  (h1 : ∃ x : ℤ, x^3 + b*x + c = 0) 
  (h2 : (5 - Real.sqrt 11)^3 + b*(5 - Real.sqrt 11) + c = 0) : 
  ∃ x : ℤ, x^3 + b*x + c = 0 ∧ x = -10 := by
sorry

end NUMINAMATH_CALUDE_cubic_integer_root_l3177_317764


namespace NUMINAMATH_CALUDE_smallest_integer_price_with_tax_l3177_317722

theorem smallest_integer_price_with_tax (n : ℕ) : n = 53 ↔ 
  n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬ ∃ x : ℕ, (106 * x : ℚ) / 100 = m) ∧
  (∃ x : ℕ, (106 * x : ℚ) / 100 = n) :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_price_with_tax_l3177_317722


namespace NUMINAMATH_CALUDE_linear_system_solution_l3177_317714

theorem linear_system_solution (x y : ℝ) 
  (eq1 : x + 3*y = 20) 
  (eq2 : x + y = 10) : 
  x = 5 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l3177_317714


namespace NUMINAMATH_CALUDE_valid_sequence_count_l3177_317789

def word : String := "EQUALS"

def valid_sequence (s : String) : Prop :=
  s.length = 4 ∧
  s.toList.toFinset ⊆ word.toList.toFinset ∧
  s.front = 'L' ∧
  s.back = 'Q' ∧
  s.toList.toFinset.card = 4

def count_valid_sequences : ℕ :=
  (word.toList.toFinset.filter (λ c => c ≠ 'L' ∧ c ≠ 'Q')).card *
  ((word.toList.toFinset.filter (λ c => c ≠ 'L' ∧ c ≠ 'Q')).card - 1)

theorem valid_sequence_count :
  count_valid_sequences = 12 :=
sorry

end NUMINAMATH_CALUDE_valid_sequence_count_l3177_317789


namespace NUMINAMATH_CALUDE_equation_solution_l3177_317759

theorem equation_solution : ∃ x : ℝ, 2 * x - 3 = 7 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3177_317759


namespace NUMINAMATH_CALUDE_line_parabola_intersection_condition_l3177_317705

/-- Parabola C with equation x² = 1/2 * y -/
def parabola_C (x y : ℝ) : Prop := x^2 = 1/2 * y

/-- Line passing through points (0, -4) and (t, 0) -/
def line_AB (t x y : ℝ) : Prop := 4*x - t*y - 4*t = 0

/-- The line does not intersect the parabola -/
def no_intersection (t : ℝ) : Prop :=
  ∀ x y : ℝ, parabola_C x y ∧ line_AB t x y → False

/-- The range of t for which the line does not intersect the parabola -/
theorem line_parabola_intersection_condition (t : ℝ) :
  no_intersection t ↔ t < -Real.sqrt 2 ∨ t > Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_condition_l3177_317705


namespace NUMINAMATH_CALUDE_min_value_a2b_l3177_317754

-- Define the function f
def f (x : ℝ) : ℝ := |x^2 - 6|

-- State the theorem
theorem min_value_a2b (a b : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : f a = f b) :
  ∃ (m : ℝ), m = -4 ∧ ∀ (x y : ℝ), x < y ∧ y < 0 ∧ f x = f y → m ≤ x^2 * y :=
sorry

end NUMINAMATH_CALUDE_min_value_a2b_l3177_317754


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l3177_317788

theorem rectangle_area_perimeter_relation (x : ℝ) :
  let length : ℝ := 4 * x
  let width : ℝ := x + 10
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (area = 2 * perimeter) → x = (Real.sqrt 41 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l3177_317788


namespace NUMINAMATH_CALUDE_sequence_range_l3177_317719

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem sequence_range (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, 2 * a (n + 1) * a n + a (n + 1) - 3 * a n = 0)
  (h2 : a 1 > 0)
  (h3 : is_increasing a) :
  0 < a 1 ∧ a 1 < 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_range_l3177_317719


namespace NUMINAMATH_CALUDE_ages_sum_l3177_317756

theorem ages_sum (a b s : ℕ+) : 
  (3 * a + 5 + b = s) →
  (6 * s^2 = 2 * a^2 + 10 * b^2) →
  (Nat.gcd (Nat.gcd a.val b.val) s.val = 1) →
  (a + b + s = 19) := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l3177_317756


namespace NUMINAMATH_CALUDE_factorial_8_divisors_l3177_317709

theorem factorial_8_divisors : Nat.card (Nat.divisors (Nat.factorial 8)) = 96 := by sorry

end NUMINAMATH_CALUDE_factorial_8_divisors_l3177_317709


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l3177_317721

theorem unique_four_digit_number :
  ∃! (a b c d : ℕ),
    0 < a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    0 ≤ d ∧ d ≤ 9 ∧
    a + b = c + d ∧
    b + d = 2 * (a + c) ∧
    a + d = c ∧
    b + c - a = 3 * d :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l3177_317721


namespace NUMINAMATH_CALUDE_cos_five_pi_thirds_l3177_317747

theorem cos_five_pi_thirds : Real.cos (5 * π / 3) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_five_pi_thirds_l3177_317747


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l3177_317750

theorem simplify_product_of_square_roots (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l3177_317750


namespace NUMINAMATH_CALUDE_prob_even_sum_is_11_20_l3177_317702

/-- Represents a wheel with a certain number of even and odd sections -/
structure Wheel where
  total : ℕ
  even : ℕ
  odd : ℕ
  valid : total = even + odd

/-- The probability of getting an even number on a wheel -/
def prob_even (w : Wheel) : ℚ :=
  w.even / w.total

/-- The probability of getting an odd number on a wheel -/
def prob_odd (w : Wheel) : ℚ :=
  w.odd / w.total

/-- The two wheels in the game -/
def wheel1 : Wheel := ⟨5, 2, 3, rfl⟩
def wheel2 : Wheel := ⟨4, 1, 3, rfl⟩

/-- The theorem to be proved -/
theorem prob_even_sum_is_11_20 :
  prob_even wheel1 * prob_even wheel2 + prob_odd wheel1 * prob_odd wheel2 = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_11_20_l3177_317702


namespace NUMINAMATH_CALUDE_lucky_325th_number_l3177_317732

/-- A positive integer is "lucky" if the sum of its digits is 7. -/
def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ (Nat.digits 10 n).sum = 7

/-- The sequence of "lucky" numbers in ascending order. -/
def lucky_seq : ℕ → ℕ := sorry

theorem lucky_325th_number : lucky_seq 325 = 52000 := by sorry

end NUMINAMATH_CALUDE_lucky_325th_number_l3177_317732


namespace NUMINAMATH_CALUDE_area_of_bounded_region_l3177_317707

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 30*|x| = 500

/-- The bounded region created by the curve -/
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve_equation p.1 p.2}

/-- The area of the bounded region -/
noncomputable def area : ℝ := sorry

/-- Theorem stating that the area of the bounded region is 5000/3 -/
theorem area_of_bounded_region : area = 5000/3 := by sorry

end NUMINAMATH_CALUDE_area_of_bounded_region_l3177_317707


namespace NUMINAMATH_CALUDE_sebastian_orchestra_size_l3177_317783

/-- Represents the number of musicians in each section of the orchestra -/
structure OrchestraSection :=
  (percussion : ℕ)
  (brass : ℕ)
  (strings : ℕ)
  (woodwinds : ℕ)
  (keyboardsAndHarp : ℕ)
  (conductor : ℕ)

/-- Calculates the total number of musicians in the orchestra -/
def totalMusicians (o : OrchestraSection) : ℕ :=
  o.percussion + o.brass + o.strings + o.woodwinds + o.keyboardsAndHarp + o.conductor

/-- The specific orchestra composition as described in the problem -/
def sebastiansOrchestra : OrchestraSection :=
  { percussion := 4
  , brass := 13
  , strings := 18
  , woodwinds := 10
  , keyboardsAndHarp := 3
  , conductor := 1 }

/-- Theorem stating that the total number of musicians in Sebastian's orchestra is 49 -/
theorem sebastian_orchestra_size :
  totalMusicians sebastiansOrchestra = 49 := by
  sorry


end NUMINAMATH_CALUDE_sebastian_orchestra_size_l3177_317783


namespace NUMINAMATH_CALUDE_trapezoidal_formation_relationship_l3177_317736

/-- Represents the trapezoidal formation of people in rows. -/
structure TrapezoidalFormation where
  total_rows : ℕ
  first_row_count : ℕ
  row_increment : ℕ

/-- The function that calculates the number of people in a given row. -/
def people_in_row (f : TrapezoidalFormation) (x : ℕ) : ℕ :=
  f.first_row_count + (x - 1) * f.row_increment

/-- Theorem stating the relationship between row number and number of people. -/
theorem trapezoidal_formation_relationship (f : TrapezoidalFormation) 
  (h1 : f.total_rows = 60)
  (h2 : f.first_row_count = 40)
  (h3 : f.row_increment = 1)
  (x : ℕ) 
  (hx : x > 0 ∧ x ≤ f.total_rows) :
  people_in_row f x = 39 + x := by
  sorry

end NUMINAMATH_CALUDE_trapezoidal_formation_relationship_l3177_317736


namespace NUMINAMATH_CALUDE_train_speed_theorem_l3177_317753

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_theorem (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 120)
  (h2 : bridge_length = 255)
  (h3 : crossing_time = 30)
  : (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_theorem

end NUMINAMATH_CALUDE_train_speed_theorem_l3177_317753


namespace NUMINAMATH_CALUDE_new_students_count_l3177_317782

theorem new_students_count (initial_students : ℕ) (left_students : ℕ) (final_students : ℕ) :
  initial_students = 10 →
  left_students = 4 →
  final_students = 48 →
  final_students - (initial_students - left_students) = 42 :=
by sorry

end NUMINAMATH_CALUDE_new_students_count_l3177_317782


namespace NUMINAMATH_CALUDE_pens_to_pencils_ratio_l3177_317730

/-- Represents the contents of Tommy's pencil case -/
structure PencilCase where
  total : Nat
  pencils : Nat
  eraser : Nat
  pens : Nat

/-- Theorem stating the ratio of pens to pencils in Tommy's pencil case -/
theorem pens_to_pencils_ratio (case : PencilCase) 
  (h_total : case.total = 13)
  (h_pencils : case.pencils = 4)
  (h_eraser : case.eraser = 1)
  (h_sum : case.total = case.pencils + case.pens + case.eraser)
  (h_multiple : ∃ k : Nat, case.pens = k * case.pencils) :
  case.pens / case.pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_pens_to_pencils_ratio_l3177_317730


namespace NUMINAMATH_CALUDE_items_sold_l3177_317718

/-- Given the following conditions:
  1. A grocery store ordered 4458 items to restock.
  2. They have 575 items in the storeroom.
  3. They have 3,472 items left in the whole store.
  Prove that the number of items sold that day is 1561. -/
theorem items_sold (restocked : ℕ) (in_storeroom : ℕ) (left_in_store : ℕ) 
  (h1 : restocked = 4458)
  (h2 : in_storeroom = 575)
  (h3 : left_in_store = 3472) :
  restocked + in_storeroom - left_in_store = 1561 :=
by sorry

end NUMINAMATH_CALUDE_items_sold_l3177_317718


namespace NUMINAMATH_CALUDE_power_of_power_l3177_317766

theorem power_of_power (a : ℝ) : (a^3)^3 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3177_317766


namespace NUMINAMATH_CALUDE_minimum_value_complex_l3177_317748

theorem minimum_value_complex (z : ℂ) (h : Complex.abs (z - 3 + Complex.I) = 3) :
  (Complex.abs (z + 2 - 3 * Complex.I))^2 + (Complex.abs (z - 6 + 2 * Complex.I))^2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_complex_l3177_317748


namespace NUMINAMATH_CALUDE_inequality_solution_l3177_317728

theorem inequality_solution (a b : ℝ) :
  (∀ x, b - a * x > 0 ↔ 
    ((a > 0 ∧ x < b / a) ∨ 
     (a < 0 ∧ x > b / a) ∨ 
     (a = 0 ∧ False))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3177_317728


namespace NUMINAMATH_CALUDE_rectangle_max_regions_l3177_317775

/-- The maximum number of regions a rectangle can be divided into with n line segments --/
def max_regions (n : ℕ) : ℕ :=
  if n = 0 then 1
  else max_regions (n - 1) + n

/-- Theorem: A rectangle with 5 line segments can be divided into at most 16 regions --/
theorem rectangle_max_regions :
  max_regions 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_regions_l3177_317775


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l3177_317746

theorem ratio_of_sum_and_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h_sum_diff : a + b = 4 * (a - b)) : a / b = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l3177_317746


namespace NUMINAMATH_CALUDE_square_properties_l3177_317762

/-- Properties of a square with side length 30 cm -/
theorem square_properties :
  let s : ℝ := 30
  let area : ℝ := s^2
  let diagonal : ℝ := s * Real.sqrt 2
  (area = 900 ∧ diagonal = 30 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_square_properties_l3177_317762


namespace NUMINAMATH_CALUDE_cube_intersection_area_ratio_l3177_317780

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a polygon -/
structure Polygon where
  vertices : List Point3D

/-- The ratio of the area of polygon P to the area of triangle ABC -/
def areaRatio (cube : Cube) (A B C : Point3D) (P : Polygon) : ℚ :=
  11/6

/-- Theorem: The ratio of the area of polygon P to the area of triangle ABC is 11/6 -/
theorem cube_intersection_area_ratio 
  (cube : Cube) 
  (A : Point3D) 
  (B : Point3D) 
  (C : Point3D) 
  (P : Polygon)
  (h1 : A.x = 0 ∧ A.y = 0 ∧ A.z = 0)  -- A is a corner of the cube
  (h2 : B.x = cube.sideLength ∧ B.y = cube.sideLength/2 ∧ B.z = 0)  -- B is a midpoint of an edge
  (h3 : C.x = 0 ∧ C.y = cube.sideLength/2 ∧ C.z = cube.sideLength)  -- C is a midpoint of an edge
  (h4 : P.vertices.length = 5)  -- P is a pentagon
  : areaRatio cube A B C P = 11/6 := by
  sorry


end NUMINAMATH_CALUDE_cube_intersection_area_ratio_l3177_317780


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l3177_317770

-- Define the total number of people and the number of math majors
def total_people : ℕ := 12
def math_majors : ℕ := 5

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the probability of math majors sitting consecutively
def prob_consecutive_math_majors : ℚ := (total_people : ℚ) / (choose total_people math_majors : ℚ)

-- State the theorem
theorem math_majors_consecutive_probability :
  prob_consecutive_math_majors = 1 / 66 :=
sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l3177_317770


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3177_317763

def A : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1 + 1}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1 + 3}

theorem intersection_of_A_and_B : ∃! a : ℝ × ℝ, a ∈ A ∧ a ∈ B ∧ a = (2, 5) := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3177_317763


namespace NUMINAMATH_CALUDE_expression_equals_zero_l3177_317793

theorem expression_equals_zero (x y : ℝ) : 
  (5 * x^2 - 3 * x + 2) * (107 - 107) + (7 * y^2 + 4 * y - 1) * (93 - 93) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l3177_317793


namespace NUMINAMATH_CALUDE_min_abs_b_minus_c_l3177_317733

/-- Given real numbers a, b, c satisfying (a - 2b - 1)² + (a - c - ln c)² = 0,
    the minimum value of |b - c| is 1. -/
theorem min_abs_b_minus_c (a b c : ℝ) 
    (h : (a - 2*b - 1)^2 + (a - c - Real.log c)^2 = 0) :
    ∀ x : ℝ, |b - c| ≤ x → 1 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_min_abs_b_minus_c_l3177_317733


namespace NUMINAMATH_CALUDE_water_formation_l3177_317731

-- Define the molecules and their quantities
def HCl_moles : ℕ := 1
def NaHCO3_moles : ℕ := 1

-- Define the reaction equation
def reaction_equation : String := "HCl + NaHCO3 → NaCl + H2O + CO2"

-- Define the function to calculate water moles produced
def water_moles_produced (hcl : ℕ) (nahco3 : ℕ) : ℕ :=
  min hcl nahco3

-- Theorem statement
theorem water_formation :
  water_moles_produced HCl_moles NaHCO3_moles = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_water_formation_l3177_317731


namespace NUMINAMATH_CALUDE_milk_dilution_l3177_317751

/-- Proves that adding 15 liters of pure milk to 10 liters of milk with 5% water content
    results in a final water content of 2% -/
theorem milk_dilution (initial_milk : ℝ) (pure_milk : ℝ) (initial_water_percent : ℝ) :
  initial_milk = 10 →
  pure_milk = 15 →
  initial_water_percent = 5 →
  let total_milk := initial_milk + pure_milk
  let water_volume := initial_milk * (initial_water_percent / 100)
  let final_water_percent := (water_volume / total_milk) * 100
  final_water_percent = 2 := by
sorry

end NUMINAMATH_CALUDE_milk_dilution_l3177_317751


namespace NUMINAMATH_CALUDE_runners_meeting_time_l3177_317796

/-- Represents a runner with their start time (in minutes after 7:00 AM) and lap duration -/
structure Runner where
  startTime : ℕ
  lapDuration : ℕ

/-- The earliest time (in minutes after 7:00 AM) when all runners meet at the starting point -/
def earliestMeetingTime (runners : List Runner) : ℕ :=
  sorry

/-- The problem statement -/
theorem runners_meeting_time :
  let kevin := Runner.mk 45 5
  let laura := Runner.mk 50 8
  let neil := Runner.mk 55 10
  let runners := [kevin, laura, neil]
  earliestMeetingTime runners = 95
  := by sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l3177_317796


namespace NUMINAMATH_CALUDE_parabola_c_value_l3177_317791

/-- A parabola with equation y = 2x^2 + bx + c passing through (-2, 20) and (2, 28) has c = 16 -/
theorem parabola_c_value (b c : ℝ) : 
  (∀ x y : ℝ, y = 2 * x^2 + b * x + c → 
    ((x = -2 ∧ y = 20) ∨ (x = 2 ∧ y = 28))) → 
  c = 16 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3177_317791


namespace NUMINAMATH_CALUDE_sam_winning_probability_l3177_317771

theorem sam_winning_probability :
  let hit_probability : ℚ := 2/5
  let miss_probability : ℚ := 3/5
  let p : ℚ := p -- p represents the probability of Sam winning
  (hit_probability = 2/5) →
  (miss_probability = 3/5) →
  (p = hit_probability + miss_probability * miss_probability * p) →
  p = 5/8 := by
sorry

end NUMINAMATH_CALUDE_sam_winning_probability_l3177_317771


namespace NUMINAMATH_CALUDE_cornbread_pieces_l3177_317786

theorem cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 24)
  (h2 : pan_width = 20)
  (h3 : piece_length = 3)
  (h4 : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 80 := by
  sorry

end NUMINAMATH_CALUDE_cornbread_pieces_l3177_317786


namespace NUMINAMATH_CALUDE_expand_product_l3177_317767

theorem expand_product (x y : ℝ) : (3*x + 4*y)*(2*x - 5*y + 7) = 6*x^2 - 7*x*y + 21*x - 20*y^2 + 28*y := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3177_317767


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3177_317715

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    (5 * x - 3) / (x^2 - 3*x - 18) = C / (x - 6) + D / (x + 3)) →
  C = 3 ∧ D = 2 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3177_317715


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3177_317725

theorem divisibility_by_five (x y : ℤ) :
  (∃ k : ℤ, x^2 - 2*x*y + 2*y^2 = 5*k ∨ x^2 + 2*x*y + 2*y^2 = 5*k) ↔ 
  (∃ a b : ℤ, x = 5*a ∧ y = 5*b) ∨ 
  (∀ k : ℤ, x ≠ 5*k ∧ y ≠ 5*k) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l3177_317725


namespace NUMINAMATH_CALUDE_subtraction_problem_l3177_317700

theorem subtraction_problem : 
  (7000 / 10) - (7000 * (1 / 10) / 100) = 693 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3177_317700


namespace NUMINAMATH_CALUDE_max_defective_items_l3177_317795

theorem max_defective_items 
  (N M n : ℕ) 
  (h1 : M ≤ N) 
  (h2 : n ≤ N) : 
  ∃ X : ℕ, X ≤ min M n ∧ 
  ∀ Y : ℕ, Y ≤ M ∧ Y ≤ n → Y ≤ X :=
sorry

end NUMINAMATH_CALUDE_max_defective_items_l3177_317795


namespace NUMINAMATH_CALUDE_second_bus_ride_duration_l3177_317757

def first_bus_wait : ℕ := 12
def first_bus_ride : ℕ := 30

def total_first_bus_time : ℕ := first_bus_wait + first_bus_ride

def second_bus_time : ℕ := total_first_bus_time / 2

theorem second_bus_ride_duration : second_bus_time = 21 := by
  sorry

end NUMINAMATH_CALUDE_second_bus_ride_duration_l3177_317757


namespace NUMINAMATH_CALUDE_fraction_zero_value_l3177_317712

theorem fraction_zero_value (x : ℝ) : 
  (x^2 - 4) / (x - 2) = 0 ∧ x - 2 ≠ 0 → x = -2 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_value_l3177_317712


namespace NUMINAMATH_CALUDE_lenora_points_scored_l3177_317776

-- Define the types of shots
inductive ShotType
| ThreePoint
| FreeThrow

-- Define the game parameters
def total_shots : ℕ := 40
def three_point_success_rate : ℚ := 1/4
def free_throw_success_rate : ℚ := 1/2

-- Define the point values for each shot type
def point_value (shot : ShotType) : ℕ :=
  match shot with
  | ShotType.ThreePoint => 3
  | ShotType.FreeThrow => 1

-- Define the function to calculate points scored
def points_scored (three_point_attempts : ℕ) (free_throw_attempts : ℕ) : ℚ :=
  (three_point_attempts : ℚ) * three_point_success_rate * (point_value ShotType.ThreePoint) +
  (free_throw_attempts : ℚ) * free_throw_success_rate * (point_value ShotType.FreeThrow)

-- Theorem statement
theorem lenora_points_scored :
  ∀ (three_point_attempts free_throw_attempts : ℕ),
    three_point_attempts + free_throw_attempts = total_shots →
    points_scored three_point_attempts free_throw_attempts = 30 :=
by sorry

end NUMINAMATH_CALUDE_lenora_points_scored_l3177_317776


namespace NUMINAMATH_CALUDE_hundred_decomposition_l3177_317713

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def isPerfectCube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def isValidDecomposition (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  isPerfectSquare a ∧ isPerfectSquare b ∧ isPerfectCube c

theorem hundred_decomposition :
  ∃! (a b c : ℕ), a + b + c = 100 ∧ isValidDecomposition a b c :=
sorry

end NUMINAMATH_CALUDE_hundred_decomposition_l3177_317713


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3177_317799

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3177_317799


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_two_one_l3177_317729

/-- 
Given two positive integers r and s with r > s, and two distinct non-constant polynomials P and Q 
with real coefficients such that P(x)^r - P(x)^s = Q(x)^r - Q(x)^s for all real x, 
prove that r = 2 and s = 1.
-/
theorem polynomial_equality_implies_two_one (r s : ℕ) (P Q : ℝ → ℝ) : 
  r > s → 
  s > 0 →
  (∀ x : ℝ, P x ≠ Q x) → 
  (∃ a b c d : ℝ, a ≠ 0 ∧ c ≠ 0 ∧ ∀ x : ℝ, P x = a * x + b ∧ Q x = c * x + d) →
  (∀ x : ℝ, (P x)^r - (P x)^s = (Q x)^r - (Q x)^s) →
  r = 2 ∧ s = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_two_one_l3177_317729


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_open_zero_one_l3177_317785

def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem M_intersect_N_equals_open_zero_one : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_open_zero_one_l3177_317785


namespace NUMINAMATH_CALUDE_sister_age_l3177_317781

theorem sister_age (B S : ℕ) (h : B = B * S) : S = 1 := by
  sorry

end NUMINAMATH_CALUDE_sister_age_l3177_317781


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_twice_l3177_317710

/-- A quadratic function parameterized by k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 - (2*k - 1) * x + k

/-- The discriminant of the quadratic function f -/
def discriminant (k : ℝ) : ℝ := (2*k - 1)^2 - 4*k*(k - 2)

/-- The condition for f to have two distinct real roots -/
def has_two_distinct_roots (k : ℝ) : Prop :=
  discriminant k > 0 ∧ k ≠ 2

theorem quadratic_intersects_x_axis_twice (k : ℝ) :
  has_two_distinct_roots k ↔ k > -1/4 ∧ k ≠ 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_twice_l3177_317710


namespace NUMINAMATH_CALUDE_sum_of_cubes_over_product_l3177_317792

theorem sum_of_cubes_over_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hsum : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_over_product_l3177_317792


namespace NUMINAMATH_CALUDE_distance_point_to_line_l3177_317755

/-- Given a line in polar form and a point in polar coordinates, 
    calculate the distance from the point to the line. -/
theorem distance_point_to_line 
  (ρ θ : ℝ) -- polar coordinates of the point
  (h_line : ∀ (ρ' θ' : ℝ), 2 * ρ' * Real.sin (θ' - π/4) = Real.sqrt 2) -- line equation
  (h_point : ρ = 2 * Real.sqrt 2 ∧ θ = 7 * π/4) -- point coordinates
  : let x := ρ * Real.cos θ
    let y := ρ * Real.sin θ
    (y - x - 1) / Real.sqrt 2 = 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l3177_317755


namespace NUMINAMATH_CALUDE_system_solution_l3177_317779

theorem system_solution (x y a : ℝ) : 
  x + 2*y = 2*a - 1 →
  x - y = 6 →
  x = -y →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3177_317779


namespace NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l3177_317772

/-- The total surface area of a shape consisting of a hemisphere attached to a cylindrical segment -/
theorem hemisphere_cylinder_surface_area (r : ℝ) (h : r = 10) :
  let hemisphere_area := 2 * π * r^2
  let cylinder_base_area := π * r^2
  let cylinder_lateral_area := 2 * π * r * (r / 2)
  hemisphere_area + cylinder_base_area + cylinder_lateral_area = 40 * π * r^2 :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l3177_317772


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3177_317787

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 4 → (x > 3 ∨ x < -1)) ∧ 
  (∃ x : ℝ, (x > 3 ∨ x < -1) ∧ ¬(x > 4)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3177_317787


namespace NUMINAMATH_CALUDE_investment_percentage_problem_l3177_317765

theorem investment_percentage_problem (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ) 
  (second_rate : ℝ) (third_rate : ℝ) (desired_income : ℝ) (x : ℝ) :
  total_investment = 10000 ∧ 
  first_investment = 4000 ∧ 
  second_investment = 3500 ∧ 
  second_rate = 0.04 ∧ 
  third_rate = 0.064 ∧ 
  desired_income = 500 ∧
  first_investment * (x / 100) + second_investment * second_rate + 
    (total_investment - first_investment - second_investment) * third_rate = desired_income →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_investment_percentage_problem_l3177_317765


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l3177_317706

def f (x : ℝ) := 6 - 12 * x + x^3

theorem f_max_min_on_interval :
  let a := -1/3
  let b := 1
  ∃ (x_max x_min : ℝ),
    x_max ∈ Set.Icc a b ∧
    x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    x_max = a ∧
    x_min = b ∧
    f x_max = 269/27 ∧
    f x_min = -5 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l3177_317706
