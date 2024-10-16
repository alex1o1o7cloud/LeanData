import Mathlib

namespace NUMINAMATH_CALUDE_max_value_on_unit_circle_l3680_368042

def unitCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem max_value_on_unit_circle (x₁ y₁ x₂ y₂ : ℝ) :
  unitCircle x₁ y₁ →
  unitCircle x₂ y₂ →
  (x₁, y₁) ≠ (x₂, y₂) →
  x₁ * y₂ = x₂ * y₁ →
  ∀ t, 2*x₁ + x₂ + 2*y₁ + y₂ ≤ t →
  t = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_unit_circle_l3680_368042


namespace NUMINAMATH_CALUDE_fraction_of_25_l3680_368006

theorem fraction_of_25 : 
  ∃ x : ℚ, x * 25 = (80 / 100) * 40 - 22 ∧ x = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_25_l3680_368006


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l3680_368010

/-- The cost of Grandfather Zhao's ticket -/
def grandfather_ticket_cost : ℝ := 10

/-- The discount rate for Grandfather Zhao's ticket -/
def grandfather_discount_rate : ℝ := 0.2

/-- The number of minor tickets -/
def num_minor_tickets : ℕ := 3

/-- The discount rate for minor tickets -/
def minor_discount_rate : ℝ := 0.4

/-- The number of regular tickets -/
def num_regular_tickets : ℕ := 2

/-- The number of senior tickets (excluding Grandfather Zhao) -/
def num_senior_tickets : ℕ := 1

/-- The discount rate for senior tickets (excluding Grandfather Zhao) -/
def senior_discount_rate : ℝ := 0.3

/-- The total cost of all tickets -/
def total_cost : ℝ := 66.25

theorem concert_ticket_cost :
  let regular_ticket_cost := grandfather_ticket_cost / (1 - grandfather_discount_rate)
  let minor_ticket_cost := regular_ticket_cost * (1 - minor_discount_rate)
  let senior_ticket_cost := regular_ticket_cost * (1 - senior_discount_rate)
  total_cost = num_minor_tickets * minor_ticket_cost +
               num_regular_tickets * regular_ticket_cost +
               num_senior_tickets * senior_ticket_cost +
               grandfather_ticket_cost :=
by sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l3680_368010


namespace NUMINAMATH_CALUDE_last_two_digits_product_l3680_368099

theorem last_two_digits_product (n : ℤ) : 
  (n % 100 ≥ 10) →  -- Ensure n has at least two digits
  (n % 8 = 0) →     -- n is divisible by 8
  ((n % 100) / 10 + n % 10 = 12) →  -- Sum of last two digits is 12
  ((n % 100) / 10 * (n % 10) = 32) :=  -- Product of last two digits is 32
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l3680_368099


namespace NUMINAMATH_CALUDE_min_distinct_sums_products_l3680_368066

theorem min_distinct_sums_products (a b c d : ℤ) (h1 : a < b) (h2 : b < c) (h3 : c < d) :
  let sums := {a + b, a + c, a + d, b + c, b + d, c + d}
  let products := {a * b, a * c, a * d, b * c, b * d, c * d}
  Finset.card (sums ∪ products) ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_distinct_sums_products_l3680_368066


namespace NUMINAMATH_CALUDE_x_value_proof_l3680_368080

theorem x_value_proof (x : ℝ) (h1 : x ≠ 0) (h2 : Real.sqrt ((5 * x) / 3) = x) : x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3680_368080


namespace NUMINAMATH_CALUDE_factorization_equality_l3680_368026

theorem factorization_equality (a b : ℝ) : 2*a - 8*a*b^2 = 2*a*(1-2*b)*(1+2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3680_368026


namespace NUMINAMATH_CALUDE_ch4_formation_and_consumption_l3680_368093

/-- Represents a chemical compound with its coefficient in a reaction --/
structure Compound where
  name : String
  coefficient : ℚ

/-- Represents a chemical reaction --/
structure Reaction where
  reactants : List Compound
  products : List Compound

/-- Represents the initial conditions of the problem --/
structure InitialConditions where
  be2c : ℚ
  h2o : ℚ
  o2 : ℚ
  temperature : ℚ
  pressure : ℚ

/-- The first reaction: Be2C + 4H2O → 2Be(OH)2 + CH4 --/
def reaction1 : Reaction := {
  reactants := [⟨"Be2C", 1⟩, ⟨"H2O", 4⟩],
  products := [⟨"Be(OH)2", 2⟩, ⟨"CH4", 1⟩]
}

/-- The second reaction: CH4 + 2O2 → CO2 + 2H2O --/
def reaction2 : Reaction := {
  reactants := [⟨"CH4", 1⟩, ⟨"O2", 2⟩],
  products := [⟨"CO2", 1⟩, ⟨"H2O", 2⟩]
}

/-- The initial conditions of the problem --/
def initialConditions : InitialConditions := {
  be2c := 3,
  h2o := 15,
  o2 := 6,
  temperature := 350,
  pressure := 2
}

/-- Theorem stating the amount of CH4 formed and remaining --/
theorem ch4_formation_and_consumption 
  (r1 : Reaction)
  (r2 : Reaction)
  (ic : InitialConditions)
  (h1 : r1 = reaction1)
  (h2 : r2 = reaction2)
  (h3 : ic = initialConditions) :
  ∃ (ch4_formed : ℚ) (ch4_remaining : ℚ),
    ch4_formed = 3 ∧ ch4_remaining = 0 :=
  sorry


end NUMINAMATH_CALUDE_ch4_formation_and_consumption_l3680_368093


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l3680_368030

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the centers of the circles
def center_C1 : ℝ × ℝ := (0, 0)
def center_C2 : ℝ × ℝ := (2, 0)

-- Define the radii of the circles
def radius_C1 : ℝ := 1
def radius_C2 : ℝ := 1

-- Define the distance between centers
def distance_between_centers : ℝ := 2

-- Theorem: The circles are externally tangent
theorem circles_externally_tangent :
  distance_between_centers = radius_C1 + radius_C2 :=
by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l3680_368030


namespace NUMINAMATH_CALUDE_square_side_ratio_l3680_368048

theorem square_side_ratio (area_ratio : ℚ) :
  area_ratio = 45 / 64 →
  ∃ (a b c : ℕ), (a * Real.sqrt b) / c = Real.sqrt (area_ratio) ∧
                  a = 3 ∧ b = 5 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_ratio_l3680_368048


namespace NUMINAMATH_CALUDE_sector_area_l3680_368043

/-- The area of a circular sector with central angle π/3 and radius 4 is 8π/3 -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 4) :
  (1 / 2) * r * r * θ = (8 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3680_368043


namespace NUMINAMATH_CALUDE_max_product_sum_2020_l3680_368032

theorem max_product_sum_2020 :
  (∃ (x y : ℤ), x + y = 2020 ∧ x * y = 1020100) ∧
  (∀ (a b : ℤ), a + b = 2020 → a * b ≤ 1020100) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2020_l3680_368032


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3680_368079

theorem arithmetic_expression_evaluation : 8 + 15 / 3 - 2^3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3680_368079


namespace NUMINAMATH_CALUDE_area_product_eq_volume_squared_l3680_368055

/-- Represents a rectangular box with dimensions x, y, and z, and diagonal d. -/
structure RectBox where
  x : ℝ
  y : ℝ
  z : ℝ
  d : ℝ
  h_positive : x > 0 ∧ y > 0 ∧ z > 0
  h_diagonal : d^2 = x^2 + y^2 + z^2

/-- The volume of a rectangular box. -/
def volume (box : RectBox) : ℝ := box.x * box.y * box.z

/-- The product of the areas of the bottom, side, and front of a rectangular box. -/
def areaProduct (box : RectBox) : ℝ := (box.x * box.y) * (box.y * box.z) * (box.z * box.x)

/-- Theorem stating that the product of the areas is equal to the square of the volume. -/
theorem area_product_eq_volume_squared (box : RectBox) :
  areaProduct box = (volume box)^2 := by sorry

end NUMINAMATH_CALUDE_area_product_eq_volume_squared_l3680_368055


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_fourth_l3680_368098

theorem sin_thirteen_pi_fourth : Real.sin (13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_fourth_l3680_368098


namespace NUMINAMATH_CALUDE_optimal_purchase_plan_l3680_368007

/-- Represents a machine model with its cost and production capacity -/
structure MachineModel where
  cost : ℕ
  production : ℕ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  modelA : ℕ
  modelB : ℕ

def modelA : MachineModel := ⟨60000, 15⟩
def modelB : MachineModel := ⟨40000, 10⟩

def totalMachines : ℕ := 10
def budgetLimit : ℕ := 440000
def requiredProduction : ℕ := 102

def isValidPlan (plan : PurchasePlan) : Prop :=
  plan.modelA + plan.modelB = totalMachines ∧
  plan.modelA * modelA.cost + plan.modelB * modelB.cost ≤ budgetLimit ∧
  plan.modelA * modelA.production + plan.modelB * modelB.production ≥ requiredProduction

def isOptimalPlan (plan : PurchasePlan) : Prop :=
  isValidPlan plan ∧
  ∀ (otherPlan : PurchasePlan), 
    isValidPlan otherPlan → 
    plan.modelA * modelA.cost + plan.modelB * modelB.cost ≤ 
    otherPlan.modelA * modelA.cost + otherPlan.modelB * modelB.cost

theorem optimal_purchase_plan :
  ∃ (plan : PurchasePlan), isOptimalPlan plan ∧ plan.modelA = 1 ∧ plan.modelB = 9 := by
  sorry

end NUMINAMATH_CALUDE_optimal_purchase_plan_l3680_368007


namespace NUMINAMATH_CALUDE_fraction_simplification_l3680_368072

theorem fraction_simplification :
  (1/2 + 1/5) / (3/7 - 1/14) = 49/25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3680_368072


namespace NUMINAMATH_CALUDE_sequence_formula_l3680_368047

def S (n : ℕ) : ℕ := n^2 + 3*n

def a (n : ℕ) : ℕ := 2*n + 2

theorem sequence_formula (n : ℕ) : 
  (∀ k : ℕ, S k = k^2 + 3*k) → 
  a n = S n - S (n-1) :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_l3680_368047


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l3680_368076

theorem point_movement_on_number_line (m : ℝ) : 
  (|m - 3 + 5| = 6) → (m = -8 ∨ m = 4) := by
  sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l3680_368076


namespace NUMINAMATH_CALUDE_parabola_symmetry_l3680_368005

/-- Given a parabola y = x^2 + 3x + m in the Cartesian coordinate system,
    prove that when translated 5 units to the right,
    the original and translated parabolas are symmetric about the line x = 1 -/
theorem parabola_symmetry (m : ℝ) :
  let f (x : ℝ) := x^2 + 3*x + m
  let g (x : ℝ) := f (x - 5)
  ∀ (x y : ℝ), f (1 - (x - 1)) = g (1 + (x - 1)) := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l3680_368005


namespace NUMINAMATH_CALUDE_increasing_function_implies_m_range_l3680_368020

/-- The function f(x) = 2x³ - 3mx² + 6x --/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

/-- f is increasing on the interval (2, +∞) --/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂

/-- The theorem stating that if f is increasing on (2, +∞), then m ∈ (-∞, 5/2] --/
theorem increasing_function_implies_m_range (m : ℝ) :
  is_increasing_on_interval m → m ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_implies_m_range_l3680_368020


namespace NUMINAMATH_CALUDE_problem_statement_l3680_368002

theorem problem_statement : 2006 * ((Real.sqrt 8 - Real.sqrt 2) / Real.sqrt 2) = 2006 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3680_368002


namespace NUMINAMATH_CALUDE_compute_expression_l3680_368018

theorem compute_expression : 6 * (2/3)^4 - 1/6 = 55/54 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3680_368018


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3680_368083

theorem sqrt_equation_solution : ∃ x : ℝ, x = 2401 / 100 ∧ Real.sqrt x + Real.sqrt (x + 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3680_368083


namespace NUMINAMATH_CALUDE_fraction_value_l3680_368053

theorem fraction_value (x : ℤ) : 
  (∃ (n : ℕ+), (2 : ℚ) / (x + 1 : ℚ) = n) → x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3680_368053


namespace NUMINAMATH_CALUDE_lily_milk_problem_l3680_368057

theorem lily_milk_problem (initial_milk : ℚ) (given_milk : ℚ) (remaining_milk : ℚ) :
  initial_milk = 5 ∧ given_milk = 18 / 7 ∧ remaining_milk = initial_milk - given_milk →
  remaining_milk = 17 / 7 := by
  sorry

end NUMINAMATH_CALUDE_lily_milk_problem_l3680_368057


namespace NUMINAMATH_CALUDE_min_value_expression_l3680_368071

theorem min_value_expression (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  m + (n^2 - m*n + 4) / (m - n) ≥ 4 ∧
  (m + (n^2 - m*n + 4) / (m - n) = 4 ↔ m - n = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3680_368071


namespace NUMINAMATH_CALUDE_find_other_number_l3680_368097

theorem find_other_number (x y : ℤ) : 
  3 * x + 4 * y = 161 → (x = 17 ∨ y = 17) → (x = 31 ∨ y = 31) := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l3680_368097


namespace NUMINAMATH_CALUDE_log_product_equals_four_l3680_368039

theorem log_product_equals_four : Real.log 9 / Real.log 2 * (Real.log 4 / Real.log 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_four_l3680_368039


namespace NUMINAMATH_CALUDE_unique_quadratic_trinomial_l3680_368051

theorem unique_quadratic_trinomial :
  ∃! (a b c : ℝ), 
    (∀ x : ℝ, (a + 1) * x^2 + b * x + c = 0 → (∃! y : ℝ, y = x)) ∧
    (∀ x : ℝ, a * x^2 + (b + 1) * x + c = 0 → (∃! y : ℝ, y = x)) ∧
    (∀ x : ℝ, a * x^2 + b * x + (c + 1) = 0 → (∃! y : ℝ, y = x)) ∧
    a = 1/8 ∧ b = -3/4 ∧ c = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_trinomial_l3680_368051


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3680_368052

theorem complex_equation_solution (z : ℂ) : (3 + 4*I)*z = 1 - 2*I → z = -1/5 - 2/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3680_368052


namespace NUMINAMATH_CALUDE_part_one_part_two_l3680_368045

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Part I
theorem part_one : 
  ∀ x : ℝ, (|x + 1| + 2 * |x - 1| > 5) ↔ (x < -4/3 ∨ x > 2) :=
sorry

-- Part II
theorem part_two :
  (∃ a : ℝ, ∀ x : ℝ, f x a ≤ a * |x + 3|) ∧
  (∀ b : ℝ, (∀ x : ℝ, f x b ≤ b * |x + 3|) → b ≥ 1/2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3680_368045


namespace NUMINAMATH_CALUDE_shaded_area_is_36_l3680_368033

/-- Given a rectangle and a right triangle with the following properties:
    - Rectangle: width 12, height 12, lower right vertex at (12, 0)
    - Triangle: base 12, height 12, lower left vertex at (12, 0)
    - Line passing through (0, 12) and (24, 0)
    Prove that the area of the triangle formed by this line, the vertical line x = 12,
    and the x-axis is 36 square units. -/
theorem shaded_area_is_36 (rectangle_width rectangle_height triangle_base triangle_height : ℝ)
  (h_rect_width : rectangle_width = 12)
  (h_rect_height : rectangle_height = 12)
  (h_tri_base : triangle_base = 12)
  (h_tri_height : triangle_height = 12) :
  let line := fun x => -1/2 * x + 12
  let intersection_x := 12
  let intersection_y := line intersection_x
  let shaded_area := 1/2 * intersection_y * triangle_base
  shaded_area = 36 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_36_l3680_368033


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_volume_l3680_368049

/-- The volume of a cone with the same radius and height as a cylinder with volume 81π cm³ is 27π cm³ -/
theorem cone_volume_from_cylinder_volume (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  π * r^2 * h = 81 * π → (1/3) * π * r^2 * h = 27 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_volume_l3680_368049


namespace NUMINAMATH_CALUDE_not_all_triangles_form_square_l3680_368046

/-- A triangle is a set of three points in a plane. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A partition of a triangle is a division of the triangle into a finite number of smaller parts. -/
def Partition (T : Triangle) (n : ℕ) := Set (Set (ℝ × ℝ))

/-- A square is a regular quadrilateral with four equal sides and four right angles. -/
structure Square where
  side : ℝ

/-- A function that checks if a partition of a triangle can be reassembled into a square. -/
def can_form_square (T : Triangle) (p : Partition T 1000) (S : Square) : Prop :=
  sorry

/-- Theorem stating that not all triangles can be divided into 1000 parts to form a square. -/
theorem not_all_triangles_form_square :
  ∃ T : Triangle, ¬∃ (p : Partition T 1000) (S : Square), can_form_square T p S := by
  sorry

end NUMINAMATH_CALUDE_not_all_triangles_form_square_l3680_368046


namespace NUMINAMATH_CALUDE_thirty_ninth_beautiful_time_l3680_368013

/-- A time is represented by its hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- A time is beautiful if the sum of its hours and minutes is divisible by 12 -/
def is_beautiful (t : Time) : Prop :=
  (t.hours + t.minutes) % 12 = 0

/-- Returns the next beautiful time after a given time -/
def next_beautiful (t : Time) : Time :=
  sorry

/-- Returns the nth beautiful time after a given time -/
def nth_beautiful (t : Time) (n : Nat) : Time :=
  sorry

/-- The main theorem to be proved -/
theorem thirty_ninth_beautiful_time :
  let start_time : Time := ⟨7, 49, sorry, sorry⟩
  nth_beautiful start_time 39 = ⟨15, 45, sorry, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_thirty_ninth_beautiful_time_l3680_368013


namespace NUMINAMATH_CALUDE_sort_table_in_99_moves_l3680_368094

/-- Represents a 10x10 table of integers -/
def Table := Fin 10 → Fin 10 → ℤ

/-- Checks if a table is sorted in ascending order both row-wise and column-wise -/
def is_sorted (t : Table) : Prop :=
  ∀ i j k, i < j → t i k < t j k ∧
  ∀ i j k, i < j → t k i < t k j

/-- Represents a rectangular rotation operation on the table -/
def rotate (t : Table) (i j k l : Fin 10) : Table :=
  sorry

/-- The main theorem stating that any table can be sorted in 99 or fewer moves -/
theorem sort_table_in_99_moves (t : Table) :
  (∀ i j k l, t i j ≠ t k l) →  -- All numbers are distinct
  ∃ (moves : List (Fin 10 × Fin 10 × Fin 10 × Fin 10)),
    moves.length ≤ 99 ∧
    is_sorted (moves.foldl (λ acc m => rotate acc m.1 m.2.1 m.2.2.1 m.2.2.2) t) :=
sorry

end NUMINAMATH_CALUDE_sort_table_in_99_moves_l3680_368094


namespace NUMINAMATH_CALUDE_gabriel_has_35_boxes_l3680_368021

-- Define the number of boxes for each person
def stan_boxes : ℕ := 120

-- Define relationships between box counts
def joseph_boxes : ℕ := (stan_boxes * 20) / 100
def jules_boxes : ℕ := joseph_boxes + 5
def john_boxes : ℕ := (jules_boxes * 120) / 100
def martin_boxes : ℕ := (jules_boxes * 150) / 100
def alice_boxes : ℕ := (john_boxes * 75) / 100
def gabriel_boxes : ℕ := (martin_boxes + alice_boxes) / 2

-- Theorem to prove
theorem gabriel_has_35_boxes : gabriel_boxes = 35 := by
  sorry

end NUMINAMATH_CALUDE_gabriel_has_35_boxes_l3680_368021


namespace NUMINAMATH_CALUDE_alex_initial_silk_amount_l3680_368086

/-- The amount of silk Alex had in storage initially -/
def initial_silk_amount (num_friends : ℕ) (silk_per_friend : ℕ) (num_dresses : ℕ) (silk_per_dress : ℕ) : ℕ :=
  num_friends * silk_per_friend + num_dresses * silk_per_dress

/-- Theorem stating that Alex had 600 meters of silk initially -/
theorem alex_initial_silk_amount :
  initial_silk_amount 5 20 100 5 = 600 := by sorry

end NUMINAMATH_CALUDE_alex_initial_silk_amount_l3680_368086


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l3680_368068

theorem multiply_mixed_number : (7 : ℚ) * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l3680_368068


namespace NUMINAMATH_CALUDE_range_of_G_l3680_368000

-- Define the function G
def G (x : ℝ) : ℝ := |x + 2| - 2 * |x - 2|

-- State the theorem about the range of G
theorem range_of_G :
  Set.range G = Set.Icc (-8 : ℝ) 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_G_l3680_368000


namespace NUMINAMATH_CALUDE_lauren_mail_problem_l3680_368027

theorem lauren_mail_problem (x : ℕ) 
  (h : x + (x + 10) + (x + 5) + (x + 20) = 295) : 
  x = 65 := by
  sorry

end NUMINAMATH_CALUDE_lauren_mail_problem_l3680_368027


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l3680_368078

theorem restaurant_bill_proof (total_friends : Nat) (paying_friends : Nat) (extra_payment : ℝ) :
  total_friends = 10 →
  paying_friends = 9 →
  extra_payment = 3 →
  ∃ (bill : ℝ), (paying_friends : ℝ) * ((bill / total_friends) + extra_payment) = bill ∧ bill = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l3680_368078


namespace NUMINAMATH_CALUDE_star_op_greater_star_op_commutative_l3680_368069

-- Define the new operation ※ for rational numbers
def star_op (a b : ℚ) : ℚ := (a + b + abs (a - b)) / 2

-- Theorem for part (2)
theorem star_op_greater (a b : ℚ) (h : a > b) : star_op a b = a := by sorry

-- Theorem for part (3)
theorem star_op_commutative (a b : ℚ) : star_op a b = star_op b a := by sorry

-- Examples for part (1)
example : star_op 2 3 = 3 := by sorry
example : star_op 3 3 = 3 := by sorry
example : star_op (-2) (-3) = -2 := by sorry

end NUMINAMATH_CALUDE_star_op_greater_star_op_commutative_l3680_368069


namespace NUMINAMATH_CALUDE_nancy_apples_l3680_368056

def mike_apples : ℕ := 7
def keith_apples : ℕ := 6
def total_apples : ℕ := 16

theorem nancy_apples :
  total_apples - (mike_apples + keith_apples) = 3 :=
by sorry

end NUMINAMATH_CALUDE_nancy_apples_l3680_368056


namespace NUMINAMATH_CALUDE_company_workforce_l3680_368058

/-- Proves the number of employees after hiring, given initial conditions and hiring information -/
theorem company_workforce (initial_female_percentage : ℚ) 
                          (final_female_percentage : ℚ)
                          (additional_male_workers : ℕ) : ℕ :=
  let initial_female_percentage : ℚ := 60 / 100
  let final_female_percentage : ℚ := 55 / 100
  let additional_male_workers : ℕ := 30
  360

#check company_workforce

end NUMINAMATH_CALUDE_company_workforce_l3680_368058


namespace NUMINAMATH_CALUDE_water_speed_proof_l3680_368082

/-- Proves that the speed of the water is 2 km/h, given the conditions of the swimming problem. -/
theorem water_speed_proof (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : still_water_speed = 4) (h2 : distance = 10) (h3 : time = 5) :
  ∃ water_speed : ℝ, water_speed = 2 ∧ still_water_speed - water_speed = distance / time :=
by sorry

end NUMINAMATH_CALUDE_water_speed_proof_l3680_368082


namespace NUMINAMATH_CALUDE_total_followers_count_l3680_368023

def instagram_followers : ℕ := 240
def facebook_followers : ℕ := 500

def twitter_followers : ℕ := (instagram_followers + facebook_followers) / 2
def tiktok_followers : ℕ := 3 * twitter_followers
def youtube_followers : ℕ := tiktok_followers + 510

def total_followers : ℕ := instagram_followers + facebook_followers + twitter_followers + tiktok_followers + youtube_followers

theorem total_followers_count : total_followers = 3840 := by
  sorry

end NUMINAMATH_CALUDE_total_followers_count_l3680_368023


namespace NUMINAMATH_CALUDE_amoeba_count_10_days_l3680_368034

/-- The number of amoebas in the petri dish after n days -/
def amoeba_count (n : ℕ) : ℕ := 3^n

/-- Theorem stating that the number of amoebas after 10 days is 59049 -/
theorem amoeba_count_10_days : amoeba_count 10 = 59049 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_10_days_l3680_368034


namespace NUMINAMATH_CALUDE_third_number_in_proportion_l3680_368067

theorem third_number_in_proportion (x : ℝ) (h : x = 3) : 
  ∃ y : ℝ, (x + 1) / (x + 5) = (x + 5) / (x + y) → y = 13 := by
  sorry

end NUMINAMATH_CALUDE_third_number_in_proportion_l3680_368067


namespace NUMINAMATH_CALUDE_parabola_translation_l3680_368025

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b - 2 * p.a * h
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk 1 (-6) 5
  let translated := translate original 1 2
  translated = Parabola.mk 1 (-8) 14 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3680_368025


namespace NUMINAMATH_CALUDE_daniels_noodles_l3680_368074

def noodles_problem (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : Prop :=
  initial = given_away + remaining

theorem daniels_noodles : ∃ initial : ℕ, noodles_problem initial 12 54 ∧ initial = 66 := by
  sorry

end NUMINAMATH_CALUDE_daniels_noodles_l3680_368074


namespace NUMINAMATH_CALUDE_estate_distribution_l3680_368009

-- Define the estate distribution function
def distribute (total : ℕ) (n : ℕ) : ℕ → ℕ
| 0 => 0  -- Base case: no children
| (i+1) => 
  let fixed := 1000 * i
  let remaining := total - fixed
  fixed + remaining / 10

-- Theorem statement
theorem estate_distribution (total : ℕ) :
  (∃ n : ℕ, n > 0 ∧ 
    (∀ i j : ℕ, i > 0 → j > 0 → i ≤ n → j ≤ n → 
      distribute total n i = distribute total n j) ∧
    (∀ i : ℕ, i > 0 → i ≤ n → distribute total n i > 0)) →
  (∃ n : ℕ, n = 9 ∧
    (∀ i j : ℕ, i > 0 → j > 0 → i ≤ n → j ≤ n → 
      distribute total n i = distribute total n j) ∧
    (∀ i : ℕ, i > 0 → i ≤ n → distribute total n i > 0)) :=
by sorry


end NUMINAMATH_CALUDE_estate_distribution_l3680_368009


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3680_368061

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 2 / (2^x + 1)

theorem inequality_solution_set 
  (a : ℝ)
  (h1 : ∀ x y : ℝ, x < y → f a x > f a y)
  (h2 : ∀ x : ℝ, f a (-x) = -(f a x)) :
  {t : ℝ | f a (2*t + 1) + f a (t - 5) ≤ 0} = {t : ℝ | t ≥ 4/3} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3680_368061


namespace NUMINAMATH_CALUDE_gcd_306_522_l3680_368028

theorem gcd_306_522 : Nat.gcd 306 522 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_306_522_l3680_368028


namespace NUMINAMATH_CALUDE_max_value_and_inequality_l3680_368038

noncomputable def f (x : ℝ) := Real.log (x + 1)

noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem max_value_and_inequality :
  (∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x) ∧
  (g (3 : ℝ) = 2 * Real.log 2 - 7 / 4) ∧
  (∀ (x : ℝ), x > 0 → f x < (Real.exp x - 1) / (x^2)) := by sorry

end NUMINAMATH_CALUDE_max_value_and_inequality_l3680_368038


namespace NUMINAMATH_CALUDE_distance_between_points_l3680_368088

theorem distance_between_points : 
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (-4, 7)
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3680_368088


namespace NUMINAMATH_CALUDE_fence_rods_count_l3680_368044

/-- Calculates the total number of metal rods needed for a fence --/
def total_rods (panels : ℕ) (sheets_per_panel : ℕ) (beams_per_panel : ℕ) 
                (rods_per_sheet : ℕ) (rods_per_beam : ℕ) : ℕ :=
  panels * (sheets_per_panel * rods_per_sheet + beams_per_panel * rods_per_beam)

/-- Proves that the total number of metal rods needed for the fence is 380 --/
theorem fence_rods_count : total_rods 10 3 2 10 4 = 380 := by
  sorry

end NUMINAMATH_CALUDE_fence_rods_count_l3680_368044


namespace NUMINAMATH_CALUDE_xiao_ming_score_l3680_368062

/-- Calculates the comprehensive score based on individual scores and weights -/
def comprehensive_score (written_score practical_score publicity_score : ℝ)
  (written_weight practical_weight publicity_weight : ℝ) : ℝ :=
  written_score * written_weight + practical_score * practical_weight + publicity_score * publicity_weight

/-- Theorem stating that Xiao Ming's comprehensive score is 97 -/
theorem xiao_ming_score :
  let written_score : ℝ := 96
  let practical_score : ℝ := 98
  let publicity_score : ℝ := 96
  let written_weight : ℝ := 0.30
  let practical_weight : ℝ := 0.50
  let publicity_weight : ℝ := 0.20
  comprehensive_score written_score practical_score publicity_score
    written_weight practical_weight publicity_weight = 97 :=
by sorry


end NUMINAMATH_CALUDE_xiao_ming_score_l3680_368062


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3680_368070

theorem quadratic_inequality (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3680_368070


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_collinearity_l3680_368089

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

-- Define the line l
def l (m q y : ℝ) : ℝ := m * y + q

-- Define the right focus of the ellipse
def F : ℝ × ℝ := (1, 0)

-- Define the condition for A₁, F, and B to be collinear
def collinear (A₁ F B : ℝ × ℝ) : Prop :=
  (F.2 - A₁.2) * (B.1 - F.1) = (B.2 - F.2) * (F.1 - A₁.1)

-- Main theorem
theorem ellipse_line_intersection_collinearity 
  (m q : ℝ) 
  (hm : m ≠ 0) 
  (A B : ℝ × ℝ) 
  (hA : Γ A.1 A.2 ∧ A.1 = l m q A.2)
  (hB : Γ B.1 B.2 ∧ B.1 = l m q B.2)
  (hAB : A ≠ B)
  (A₁ : ℝ × ℝ)
  (hA₁ : A₁ = (A.1, -A.2)) :
  (collinear A₁ F B ↔ q = 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_collinearity_l3680_368089


namespace NUMINAMATH_CALUDE_combined_yellow_ratio_approx_32_percent_l3680_368077

/-- Represents a bag of jelly beans -/
structure JellyBeanBag where
  total : ℕ
  yellow_ratio : ℚ

/-- Calculates the ratio of yellow jelly beans to all beans when multiple bags are combined -/
def combined_yellow_ratio (bags : List JellyBeanBag) : ℚ :=
  let total_beans := bags.map (λ bag => bag.total) |>.sum
  let total_yellow := bags.map (λ bag => (bag.total : ℚ) * bag.yellow_ratio) |>.sum
  total_yellow / total_beans

/-- The theorem to be proved -/
theorem combined_yellow_ratio_approx_32_percent : 
  let bags := [
    JellyBeanBag.mk 24 (2/5),
    JellyBeanBag.mk 32 (3/10),
    JellyBeanBag.mk 34 (1/4)
  ]
  abs (combined_yellow_ratio bags - 32222/100000) < 1/10000 := by
  sorry

end NUMINAMATH_CALUDE_combined_yellow_ratio_approx_32_percent_l3680_368077


namespace NUMINAMATH_CALUDE_g_of_3_eq_125_l3680_368060

def g (x : ℝ) : ℝ := 3 * x^4 - 5 * x^3 + 4 * x^2 - 7 * x + 2

theorem g_of_3_eq_125 : g 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_125_l3680_368060


namespace NUMINAMATH_CALUDE_square_paper_side_length_l3680_368001

/-- The length of a cube's edge in centimeters -/
def cube_edge : ℝ := 12

/-- The number of square paper pieces covering the cube -/
def num_squares : ℕ := 54

/-- The surface area of a cube given its edge length -/
def cube_surface_area (edge : ℝ) : ℝ := 6 * edge^2

/-- The theorem stating that the side length of each square paper is 4 cm -/
theorem square_paper_side_length :
  ∃ (side : ℝ),
    side > 0 ∧
    side^2 * num_squares = cube_surface_area cube_edge ∧
    side = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_paper_side_length_l3680_368001


namespace NUMINAMATH_CALUDE_pages_difference_l3680_368073

/-- The number of pages Juwella read over four nights -/
def total_pages : ℕ := 100

/-- The number of pages Juwella will read tonight -/
def pages_tonight : ℕ := 20

/-- The number of pages Juwella read three nights ago -/
def pages_three_nights_ago : ℕ := 15

/-- The number of pages Juwella read two nights ago -/
def pages_two_nights_ago : ℕ := 2 * pages_three_nights_ago

/-- The number of pages Juwella read last night -/
def pages_last_night : ℕ := total_pages - pages_tonight - pages_three_nights_ago - pages_two_nights_ago

theorem pages_difference : pages_last_night - pages_two_nights_ago = 5 := by
  sorry

end NUMINAMATH_CALUDE_pages_difference_l3680_368073


namespace NUMINAMATH_CALUDE_scientific_notation_of_13976000_l3680_368050

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Function to convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_13976000 :
  toScientificNotation 13976000 = ScientificNotation.mk 1.3976 7 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_13976000_l3680_368050


namespace NUMINAMATH_CALUDE_olympic_quiz_probability_l3680_368081

theorem olympic_quiz_probability (A B C : ℝ) 
  (hA : A = 3/4)
  (hAC : (1 - A) * (1 - C) = 1/12)
  (hBC : B * C = 1/4) :
  A * B * (1 - C) + A * (1 - B) * C + (1 - A) * B * C = 15/32 := by
  sorry

end NUMINAMATH_CALUDE_olympic_quiz_probability_l3680_368081


namespace NUMINAMATH_CALUDE_min_blocks_for_specific_wall_l3680_368091

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  height : ℕ
  length : List ℕ

/-- Represents the constraints for block placement --/
structure WallConstraints where
  staggeredJoins : Bool
  evenEnds : Bool
  maxSmallBlocksPerEvenRow : ℕ

/-- Calculates the minimum number of blocks needed to build a wall --/
def minBlocksNeeded (wall : WallDimensions) (block : BlockDimensions) (constraints : WallConstraints) : ℕ :=
  sorry

/-- Theorem stating the minimum number of blocks needed for the specific wall --/
theorem min_blocks_for_specific_wall :
  let wall := WallDimensions.mk 150 8
  let block := BlockDimensions.mk 1 [3, 1]
  let constraints := WallConstraints.mk true true 4
  minBlocksNeeded wall block constraints = 400 :=
by sorry

end NUMINAMATH_CALUDE_min_blocks_for_specific_wall_l3680_368091


namespace NUMINAMATH_CALUDE_polygon_properties_l3680_368090

/-- The number of diagonals from a vertex in a polygon with n sides -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- The number of triangles formed by diagonals in a polygon with n sides -/
def triangles_formed (n : ℕ) : ℕ := n - 2

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- The sum of exterior angles of any polygon -/
def sum_exterior_angles : ℕ := 360

theorem polygon_properties :
  (diagonals_from_vertex 6 = 3) ∧
  (triangles_formed 6 = 4) ∧
  (sum_interior_angles 6 = 720) ∧
  (∃ n : ℕ, sum_interior_angles n = 2 * sum_exterior_angles - 180 ∧ n = 5) :=
sorry

end NUMINAMATH_CALUDE_polygon_properties_l3680_368090


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3680_368024

/-- The vertex of the parabola y = x^2 - 9 has coordinates (0, -9) -/
theorem parabola_vertex_coordinates :
  let f : ℝ → ℝ := λ x => x^2 - 9
  ∃ (x y : ℝ), (∀ t, f t ≥ f x) ∧ y = f x ∧ x = 0 ∧ y = -9 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3680_368024


namespace NUMINAMATH_CALUDE_number_problem_l3680_368092

theorem number_problem (x : ℝ) : 
  (∃ k : ℝ, 3 * x - 1 = k * x) ∧ 
  ((3 * x - 1) / x = 2 * x) ↔ 
  x = 1 :=
sorry

end NUMINAMATH_CALUDE_number_problem_l3680_368092


namespace NUMINAMATH_CALUDE_least_product_of_distinct_primes_l3680_368063

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def distinct_primes_product (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ 50 < p ∧ p < 100 ∧ 50 < q ∧ q < 100

theorem least_product_of_distinct_primes :
  ∃ p q : ℕ, distinct_primes_product p q ∧
    p * q = 3127 ∧
    ∀ r s : ℕ, distinct_primes_product r s → p * q ≤ r * s :=
sorry

end NUMINAMATH_CALUDE_least_product_of_distinct_primes_l3680_368063


namespace NUMINAMATH_CALUDE_complement_of_union_equals_singleton_l3680_368004

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_equals_singleton :
  (U \ (A ∪ B)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_singleton_l3680_368004


namespace NUMINAMATH_CALUDE_pole_length_problem_l3680_368040

theorem pole_length_problem (original_length : ℝ) (cut_length : ℝ) : 
  cut_length = 0.7 * original_length →
  cut_length = 14 →
  original_length = 20 := by
sorry

end NUMINAMATH_CALUDE_pole_length_problem_l3680_368040


namespace NUMINAMATH_CALUDE_y_value_at_50_l3680_368008

/-- A line passing through given points -/
structure Line where
  -- Define the line using two points
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- Theorem: Y-coordinate when X is 50 on a specific line -/
theorem y_value_at_50 (l : Line) (u : ℝ) : 
  l.x1 = 10 ∧ l.y1 = 30 ∧ 
  l.x2 = 15 ∧ l.y2 = 45 ∧ 
  (∃ y3 : ℝ, y3 = 3 * 20 ∧ Line.mk 10 30 20 y3 = l) ∧
  (∃ y4 : ℝ, y4 = u ∧ Line.mk 10 30 40 y4 = l) →
  (∃ y : ℝ, y = 150 ∧ Line.mk 10 30 50 y = l) :=
by sorry

end NUMINAMATH_CALUDE_y_value_at_50_l3680_368008


namespace NUMINAMATH_CALUDE_binomial_product_factorial_equals_l3680_368065

theorem binomial_product_factorial_equals : (
  Nat.choose 10 3 * Nat.choose 8 3 * (Nat.factorial 7 / Nat.factorial 4)
) = 235200 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_factorial_equals_l3680_368065


namespace NUMINAMATH_CALUDE_min_sum_of_product_2010_l3680_368003

theorem min_sum_of_product_2010 :
  ∃ (min : ℕ), min = 78 ∧
  ∀ (a b c : ℕ), 
    a > 0 → b > 0 → c > 0 →
    a * b * c = 2010 →
    a + b + c ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2010_l3680_368003


namespace NUMINAMATH_CALUDE_f_neg_five_halves_l3680_368087

-- Define the function f
def f : ℝ → ℝ := sorry

-- f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- f has a period of 2
axiom f_periodic : ∀ x, f (x + 2) = f x

-- f(x) = 2x(1-x) when 0 ≤ x ≤ 1
axiom f_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

-- Theorem to prove
theorem f_neg_five_halves : f (-5/2) = -1/2 := sorry

end NUMINAMATH_CALUDE_f_neg_five_halves_l3680_368087


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3680_368075

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ ∃ x : ℝ, x ≥ 0 ∧ x^3 + x < 0 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3680_368075


namespace NUMINAMATH_CALUDE_minimum_N_l3680_368022

theorem minimum_N (k : ℕ) (h : k > 0) :
  let N := k * (2 * k^2 + 3 * k + 3)
  ∃ (S : Finset ℕ),
    (S.card = 2 * k + 1) ∧
    (∀ x ∈ S, x > 0) ∧
    (S.sum id > N) ∧
    (∀ T ⊆ S, T.card = k → T.sum id ≤ N / 2) ∧
    (∀ M < N, ¬∃ (S' : Finset ℕ),
      (S'.card = 2 * k + 1) ∧
      (∀ x ∈ S', x > 0) ∧
      (S'.sum id > M) ∧
      (∀ T ⊆ S', T.card = k → T.sum id ≤ M / 2)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_N_l3680_368022


namespace NUMINAMATH_CALUDE_equidistant_complex_function_l3680_368037

theorem equidistant_complex_function (a b : ℝ) :
  (∀ z : ℂ, ‖(a + b * I) * z^2 - z^2‖ = ‖(a + b * I) * z^2‖) →
  ‖(a + b * I)‖ = 10 →
  b^2 = 99.75 := by sorry

end NUMINAMATH_CALUDE_equidistant_complex_function_l3680_368037


namespace NUMINAMATH_CALUDE_largest_integer_solution_2x_plus_3_lt_0_l3680_368017

theorem largest_integer_solution_2x_plus_3_lt_0 :
  ∀ x : ℤ, 2 * x + 3 < 0 → x ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_solution_2x_plus_3_lt_0_l3680_368017


namespace NUMINAMATH_CALUDE_student_community_selection_l3680_368011

/-- The number of ways to select communities for students. -/
def ways_to_select (num_students : ℕ) (num_communities : ℕ) : ℕ :=
  num_communities ^ num_students

/-- Theorem: Given 4 students and 3 communities, where each student chooses 1 community,
    the number of different ways of selection is 3^4. -/
theorem student_community_selection :
  ways_to_select 4 3 = 3^4 := by
  sorry

#eval ways_to_select 4 3  -- Should output 81

end NUMINAMATH_CALUDE_student_community_selection_l3680_368011


namespace NUMINAMATH_CALUDE_sum_side_lengths_eq_66_l3680_368035

/-- Represents a convex quadrilateral with specific properties -/
structure ConvexQuadrilateral where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  -- Angle A
  angle_a : ℝ
  -- Parallel sides condition
  parallel_ab_cd : Prop
  -- Arithmetic progression condition
  arithmetic_progression : Prop
  -- AB is maximum length
  ab_max : Prop

/-- The sum of all possible values for a side length other than AB -/
def sum_possible_side_lengths (q : ConvexQuadrilateral) : ℝ := sorry

/-- Main theorem statement -/
theorem sum_side_lengths_eq_66 (q : ConvexQuadrilateral) 
  (h1 : q.ab = 18)
  (h2 : q.angle_a = 60 * π / 180)
  (h3 : q.parallel_ab_cd)
  (h4 : q.arithmetic_progression)
  (h5 : q.ab_max) :
  sum_possible_side_lengths q = 66 := by sorry

end NUMINAMATH_CALUDE_sum_side_lengths_eq_66_l3680_368035


namespace NUMINAMATH_CALUDE_min_value_a_l3680_368036

theorem min_value_a (a : ℝ) : (∀ x ∈ Set.Ioc (0 : ℝ) (1/2), x^2 + a*x + 1 ≥ 0) ↔ a ≥ -5/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l3680_368036


namespace NUMINAMATH_CALUDE_apple_difference_apple_problem_solution_l3680_368012

/-- Proves that Mark has 13 fewer apples than Susan given the conditions of the problem -/
theorem apple_difference : ℕ → Prop := fun total_apples =>
  ∀ (greg_sarah_apples susan_apples mark_apples mom_pie_apples mom_leftover_apples : ℕ),
    greg_sarah_apples = 18 →
    susan_apples = 2 * (greg_sarah_apples / 2) →
    mom_pie_apples = 40 →
    mom_leftover_apples = 9 →
    total_apples = mom_pie_apples + mom_leftover_apples →
    mark_apples = total_apples - susan_apples →
    susan_apples - mark_apples = 13

/-- The main theorem stating that there exists a total number of apples satisfying the conditions -/
theorem apple_problem_solution : ∃ total_apples : ℕ, apple_difference total_apples := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_apple_problem_solution_l3680_368012


namespace NUMINAMATH_CALUDE_right_triangle_BD_length_l3680_368084

-- Define the triangle and its properties
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  D : ℝ  -- Represents the position of D on BC
  hAB : AB = 45
  hAC : AC = 60
  hBC : BC^2 = AB^2 + AC^2
  hD : 0 < D ∧ D < BC

-- Define the theorem
theorem right_triangle_BD_length (t : RightTriangle) : 
  let AD := (t.AB * t.AC) / t.BC
  let BD := Real.sqrt (t.AB^2 - AD^2)
  BD = 27 := by sorry

end NUMINAMATH_CALUDE_right_triangle_BD_length_l3680_368084


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3680_368019

theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ k : ℕ, (Nat.choose 7 k) * (2^(7-k)) * (a^k) * (1^(7-2*k)) = -70 ∧ 7 - 2*k = 1) → 
  a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3680_368019


namespace NUMINAMATH_CALUDE_coefficient_x2y4_is_30_l3680_368031

/-- The coefficient of x^2y^4 in the expansion of (1+x+y^2)^5 -/
def coefficient_x2y4 : ℕ :=
  (Nat.choose 5 2) * (Nat.choose 3 2)

/-- Theorem stating that the coefficient of x^2y^4 in (1+x+y^2)^5 is 30 -/
theorem coefficient_x2y4_is_30 : coefficient_x2y4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y4_is_30_l3680_368031


namespace NUMINAMATH_CALUDE_bracelet_profit_l3680_368095

/-- Given the following conditions:
    - Total bracelets made
    - Number of bracelets given away
    - Cost of materials
    - Selling price per bracelet
    Prove that the profit equals $8.00 -/
theorem bracelet_profit 
  (total_bracelets : ℕ) 
  (given_away : ℕ) 
  (material_cost : ℚ) 
  (price_per_bracelet : ℚ) 
  (h1 : total_bracelets = 52)
  (h2 : given_away = 8)
  (h3 : material_cost = 3)
  (h4 : price_per_bracelet = 1/4) : 
  (total_bracelets - given_away : ℚ) * price_per_bracelet - material_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_profit_l3680_368095


namespace NUMINAMATH_CALUDE_projections_proportional_to_squares_l3680_368029

/-- In a right triangle, the projections of the legs onto the hypotenuse are proportional to the squares of the legs. -/
theorem projections_proportional_to_squares 
  {a b c a1 b1 : ℝ} 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (right_triangle : a^2 + b^2 = c^2)
  (proj_a : a1 = (a^2) / c)
  (proj_b : b1 = (b^2) / c) :
  a1 / b1 = a^2 / b^2 := by
  sorry

end NUMINAMATH_CALUDE_projections_proportional_to_squares_l3680_368029


namespace NUMINAMATH_CALUDE_cookie_remainder_l3680_368064

theorem cookie_remainder (whole : ℝ) (person_a_fraction : ℝ) (person_b_fraction : ℝ) :
  person_a_fraction = 0.7 →
  person_b_fraction = 1/3 →
  (whole - person_a_fraction * whole) * (1 - person_b_fraction) = 0.2 * whole := by
  sorry

end NUMINAMATH_CALUDE_cookie_remainder_l3680_368064


namespace NUMINAMATH_CALUDE_cubic_root_theorem_l3680_368085

-- Define the cubic root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the polynomial
def f (p q : ℚ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + 45

-- State the theorem
theorem cubic_root_theorem (p q : ℚ) :
  f p q (2 - 3 * cubeRoot 5) = 0 → p = -6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_theorem_l3680_368085


namespace NUMINAMATH_CALUDE_biff_break_even_l3680_368096

/-- The number of hours required for Biff to break even on his bus trip -/
def break_even_hours (ticket_cost drinks_snacks_cost headphones_cost online_earnings wifi_cost : ℚ) : ℚ :=
  (ticket_cost + drinks_snacks_cost + headphones_cost) / (online_earnings - wifi_cost)

/-- Theorem stating that Biff needs 3 hours to break even on his bus trip -/
theorem biff_break_even :
  break_even_hours 11 3 16 12 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_biff_break_even_l3680_368096


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3680_368016

/-- Represents the repeating decimal 0.42̄157 -/
def repeating_decimal : ℚ := 42157 / 100000 + (157 / 100000) / (1 - 1/1000)

/-- The fraction we want to prove equality with -/
def target_fraction : ℚ := 4207359 / 99900

/-- Theorem stating that the repeating decimal 0.42̄157 is equal to 4207359/99900 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3680_368016


namespace NUMINAMATH_CALUDE_monotone_increasing_implies_a_geq_one_l3680_368041

/-- The function f(x) = (1/3)x³ + x² + ax + 1 is monotonically increasing in the interval [-2, a] -/
def is_monotone_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ a → f x < f y

/-- The main theorem stating that if f(x) = (1/3)x³ + x² + ax + 1 is monotonically increasing 
    in the interval [-2, a], then a ≥ 1 -/
theorem monotone_increasing_implies_a_geq_one (a : ℝ) :
  is_monotone_increasing (fun x => (1/3) * x^3 + x^2 + a*x + 1) a → a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_monotone_increasing_implies_a_geq_one_l3680_368041


namespace NUMINAMATH_CALUDE_total_salary_proof_l3680_368015

def salary_B : ℝ := 232

def salary_A : ℝ := 1.5 * salary_B

def total_salary : ℝ := salary_A + salary_B

theorem total_salary_proof : total_salary = 580 := by
  sorry

end NUMINAMATH_CALUDE_total_salary_proof_l3680_368015


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l3680_368059

/-- Theorem: Cylinder Volume Change
  Given a cylinder with an initial volume of 20 cubic feet,
  if its radius is tripled and its height is doubled,
  then its new volume will be 360 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 20 → π * (3*r)^2 * (2*h) = 360 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l3680_368059


namespace NUMINAMATH_CALUDE_target_line_is_correct_l3680_368054

/-- Given two lines in the xy-plane -/
def line1 : ℝ → ℝ → Prop := λ x y => x - y - 2 = 0
def line2 : ℝ → Prop := λ x => x - 2 = 0
def line3 : ℝ → ℝ → Prop := λ x y => x + y - 1 = 0

/-- The intersection point of line2 and line3 -/
def intersection_point : ℝ × ℝ := (2, -1)

/-- The equation of the line we want to prove -/
def target_line : ℝ → ℝ → Prop := λ x y => x - y - 3 = 0

/-- Main theorem -/
theorem target_line_is_correct : 
  (∀ x y, line1 x y ↔ ∃ k, target_line (x + k) (y + k)) ∧ 
  target_line intersection_point.1 intersection_point.2 :=
sorry

end NUMINAMATH_CALUDE_target_line_is_correct_l3680_368054


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l3680_368014

theorem greatest_integer_for_all_real_domain : 
  ∃ (b : ℤ), (∀ (x : ℝ), (x^2 + b*x + 10 ≠ 0)) ∧ 
  (∀ (c : ℤ), c > b → ∃ (x : ℝ), x^2 + c*x + 10 = 0) ∧ 
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l3680_368014
