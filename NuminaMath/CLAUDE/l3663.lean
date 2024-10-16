import Mathlib

namespace NUMINAMATH_CALUDE_diophantine_logarithm_equation_l3663_366325

theorem diophantine_logarithm_equation : ∃ (X Y Z : ℕ+), 
  (Nat.gcd X.val (Nat.gcd Y.val Z.val) = 1) ∧ 
  (X.val : ℝ) * (Real.log 5 / Real.log 100) + (Y.val : ℝ) * (Real.log 4 / Real.log 100) = Z.val ∧
  X.val + Y.val + Z.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_logarithm_equation_l3663_366325


namespace NUMINAMATH_CALUDE_chosen_number_l3663_366362

theorem chosen_number (x : ℝ) : (x / 8) - 160 = 12 → x = 1376 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_l3663_366362


namespace NUMINAMATH_CALUDE_shipwreck_age_conversion_l3663_366305

theorem shipwreck_age_conversion : 
  (7 * 8^2 + 4 * 8^1 + 2 * 8^0 : ℕ) = 482 := by
  sorry

end NUMINAMATH_CALUDE_shipwreck_age_conversion_l3663_366305


namespace NUMINAMATH_CALUDE_table_arrangement_l3663_366393

theorem table_arrangement (total_tables : Nat) (num_rows : Nat) 
  (tables_per_row : Nat) (leftover : Nat) : 
  total_tables = 74 → num_rows = 8 → 
  tables_per_row = total_tables / num_rows →
  leftover = total_tables % num_rows →
  tables_per_row = 9 ∧ leftover = 2 := by
  sorry

end NUMINAMATH_CALUDE_table_arrangement_l3663_366393


namespace NUMINAMATH_CALUDE_carpentry_job_cost_l3663_366315

/-- Calculates the total cost of a carpentry job -/
theorem carpentry_job_cost 
  (hourly_rate : ℕ) 
  (material_cost : ℕ) 
  (estimated_hours : ℕ) 
  (h1 : hourly_rate = 28)
  (h2 : material_cost = 560)
  (h3 : estimated_hours = 15) :
  hourly_rate * estimated_hours + material_cost = 980 := by
sorry

end NUMINAMATH_CALUDE_carpentry_job_cost_l3663_366315


namespace NUMINAMATH_CALUDE_quadratic_product_is_square_l3663_366352

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Derivative of a quadratic polynomial -/
def QuadraticPolynomial.deriv (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

theorem quadratic_product_is_square (f g : QuadraticPolynomial) 
  (h : ∀ x : ℝ, f.deriv x * g.deriv x ≥ |f.eval x| + |g.eval x|) :
  ∃ h : QuadraticPolynomial, ∀ x : ℝ, f.eval x * g.eval x = (h.eval x)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_product_is_square_l3663_366352


namespace NUMINAMATH_CALUDE_dice_probability_l3663_366371

/-- The number of possible outcomes when rolling six standard six-sided dice -/
def total_outcomes : ℕ := 6^6

/-- The number of ways to get at least one pair but not a four-of-a-kind -/
def successful_outcomes : ℕ := 27000

/-- The probability of getting at least one pair but not a four-of-a-kind when rolling six standard six-sided dice -/
theorem dice_probability : 
  (successful_outcomes : ℚ) / total_outcomes = 625 / 1089 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l3663_366371


namespace NUMINAMATH_CALUDE_no_x_exists_rational_l3663_366303

theorem no_x_exists_rational : ¬ ∃ (x : ℝ), (∃ (a b : ℚ), (x + Real.sqrt 2 = a) ∧ (x^3 + Real.sqrt 2 = b)) := by
  sorry

end NUMINAMATH_CALUDE_no_x_exists_rational_l3663_366303


namespace NUMINAMATH_CALUDE_a7_value_in_arithmetic_sequence_l3663_366346

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a7_value_in_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_sum : a 3 + a 5 = 10) : 
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_a7_value_in_arithmetic_sequence_l3663_366346


namespace NUMINAMATH_CALUDE_parabola_properties_l3663_366332

/-- A parabola is defined by its coefficients a, h, and k in the equation y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- A parabola opens downwards if its 'a' coefficient is negative -/
def opens_downwards (p : Parabola) : Prop := p.a < 0

/-- The axis of symmetry of a parabola is the line x = h -/
def axis_of_symmetry (p : Parabola) (x : ℝ) : Prop := x = p.h

theorem parabola_properties (p : Parabola) :
  opens_downwards p ∧ axis_of_symmetry p 3 → p.a < 0 ∧ p.h = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3663_366332


namespace NUMINAMATH_CALUDE_ticket_price_difference_l3663_366301

def prebought_count : ℕ := 20
def prebought_price : ℕ := 155
def gate_count : ℕ := 30
def gate_price : ℕ := 200

theorem ticket_price_difference : 
  gate_count * gate_price - prebought_count * prebought_price = 2900 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_difference_l3663_366301


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l3663_366395

theorem fraction_sum_theorem (a b c d x y z w : ℝ) 
  (h1 : x / a + y / b + z / c + w / d = 4)
  (h2 : a / x + b / y + c / z + d / w = 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 + w^2 / d^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l3663_366395


namespace NUMINAMATH_CALUDE_biggest_measure_for_containers_l3663_366360

theorem biggest_measure_for_containers (a b c : ℕ) 
  (ha : a = 496) (hb : b = 403) (hc : c = 713) : 
  Nat.gcd a (Nat.gcd b c) = 31 := by
  sorry

end NUMINAMATH_CALUDE_biggest_measure_for_containers_l3663_366360


namespace NUMINAMATH_CALUDE_smallest_bounded_area_l3663_366326

/-- The area of the smallest region bounded by y = x^2 and x^2 + y^2 = 9 -/
theorem smallest_bounded_area : 
  let f (x : ℝ) := x^2
  let g (x y : ℝ) := x^2 + y^2 = 9
  let intersection_x := Real.sqrt ((Real.sqrt 37 - 1) / 2)
  let bounded_area := (2/3) * (((-1 + Real.sqrt 37) / 2)^(3/2))
  ∃ (area : ℝ), area = bounded_area ∧ 
    (∀ x y, -intersection_x ≤ x ∧ x ≤ intersection_x ∧ 
            y = f x ∧ g x y → 
            area = ∫ x in -intersection_x..intersection_x, f x) :=
by sorry


end NUMINAMATH_CALUDE_smallest_bounded_area_l3663_366326


namespace NUMINAMATH_CALUDE_rectangle_waste_area_l3663_366313

theorem rectangle_waste_area (x y : ℝ) (h1 : x + 2*y = 7) (h2 : 2*x + 3*y = 11) : 
  let a := Real.sqrt (x^2 + y^2)
  let total_area := 11 * 7
  let waste_area := total_area - 4 * a^2
  let waste_percentage := (waste_area / total_area) * 100
  ∃ ε > 0, abs (waste_percentage - 48) < ε :=
sorry

end NUMINAMATH_CALUDE_rectangle_waste_area_l3663_366313


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3663_366389

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3663_366389


namespace NUMINAMATH_CALUDE_inequality_proof_l3663_366340

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 1) : 
  x * y + y * z + z * x ≤ 2 / 7 + 9 * x * y * z / 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3663_366340


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3663_366324

/-- Given three equally spaced parallel lines intersecting a circle and creating chords of lengths 40, 36, and 32, 
    the distance between two adjacent parallel lines is √(576/31). -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ), 
    chord1 = 40 ∧ 
    chord2 = 36 ∧ 
    chord3 = 32 ∧ 
    400 + (5/4) * d^2 = r^2 ∧ 
    256 + (36/4) * d^2 = r^2) → 
  d = Real.sqrt (576/31) := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l3663_366324


namespace NUMINAMATH_CALUDE_stocking_stuffers_l3663_366388

theorem stocking_stuffers (num_kids : ℕ) (candy_canes_per_stocking : ℕ) (beanie_babies_per_stocking : ℕ) (total_stuffers : ℕ) : 
  num_kids = 3 → 
  candy_canes_per_stocking = 4 → 
  beanie_babies_per_stocking = 2 → 
  total_stuffers = 21 → 
  (total_stuffers - (candy_canes_per_stocking + beanie_babies_per_stocking) * num_kids) / num_kids = 1 :=
by sorry

end NUMINAMATH_CALUDE_stocking_stuffers_l3663_366388


namespace NUMINAMATH_CALUDE_binomial_coefficient_property_l3663_366372

theorem binomial_coefficient_property :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243 :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_property_l3663_366372


namespace NUMINAMATH_CALUDE_integers_less_than_four_abs_value_l3663_366342

theorem integers_less_than_four_abs_value :
  {x : ℤ | |x| < 4} = {-3, -2, -1, 0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_integers_less_than_four_abs_value_l3663_366342


namespace NUMINAMATH_CALUDE_triangle_shape_l3663_366361

theorem triangle_shape (A B C : ℝ) (a b c : ℝ) (S : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a * Real.sin ((A + C) / 2) = b * Real.sin A ∧
  2 * S = Real.sqrt 3 * (b * c * Real.cos A) →
  A = π / 3 ∧ B = π / 3 ∧ C = π / 3 :=
by sorry

#check triangle_shape

end NUMINAMATH_CALUDE_triangle_shape_l3663_366361


namespace NUMINAMATH_CALUDE_national_bank_interest_rate_national_bank_interest_rate_is_five_percent_l3663_366331

theorem national_bank_interest_rate 
  (initial_investment : ℝ) 
  (additional_investment : ℝ) 
  (additional_rate : ℝ) 
  (total_income_rate : ℝ) : ℝ :=
  let total_investment := initial_investment + additional_investment
  let total_income := total_investment * total_income_rate
  let additional_income := additional_investment * additional_rate
  let national_bank_income := total_income - additional_income
  national_bank_income / initial_investment

#check national_bank_interest_rate 2400 600 0.1 0.06 -- Expected output: 0.05

theorem national_bank_interest_rate_is_five_percent :
  national_bank_interest_rate 2400 600 0.1 0.06 = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_national_bank_interest_rate_national_bank_interest_rate_is_five_percent_l3663_366331


namespace NUMINAMATH_CALUDE_circle_theorem_l3663_366398

/-- The circle passing through points A(1, 4) and B(3, 2) with its center on the line y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 4.5)^2 + y^2 = 28.25

/-- Point A -/
def point_A : ℝ × ℝ := (1, 4)

/-- Point B -/
def point_B : ℝ × ℝ := (3, 2)

/-- Point P -/
def point_P : ℝ × ℝ := (2, 4)

/-- A point is inside the circle if the left side of the equation is less than the right side -/
def is_inside_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - 4.5)^2 + p.2^2 < 28.25

theorem circle_theorem :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  is_inside_circle point_P :=
by sorry

end NUMINAMATH_CALUDE_circle_theorem_l3663_366398


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3663_366381

def i : ℂ := Complex.I

theorem complex_number_in_second_quadrant :
  let z : ℂ := (1 + i) * i
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3663_366381


namespace NUMINAMATH_CALUDE_sum_of_roots_special_quadratic_l3663_366341

theorem sum_of_roots_special_quadratic :
  let f : ℝ → ℝ := λ x ↦ (x - 7)^2 - 16
  ∃ r₁ r₂ : ℝ, (f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ ≠ r₂) ∧ r₁ + r₂ = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_special_quadratic_l3663_366341


namespace NUMINAMATH_CALUDE_two_cubic_feet_to_cubic_inches_l3663_366348

-- Define the conversion factor
def inches_per_foot : ℕ := 12

-- Define the volume conversion
def cubic_inches_per_cubic_foot : ℕ := inches_per_foot ^ 3

-- Theorem statement
theorem two_cubic_feet_to_cubic_inches :
  2 * cubic_inches_per_cubic_foot = 3456 := by
  sorry

end NUMINAMATH_CALUDE_two_cubic_feet_to_cubic_inches_l3663_366348


namespace NUMINAMATH_CALUDE_part_one_part_two_l3663_366314

/-- The function f(x) as defined in the problem -/
def f (a c x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + c

/-- Part 1 of the theorem -/
theorem part_one (a : ℝ) :
  f a 19 1 > 0 ↔ -2 < a ∧ a < 8 := by sorry

/-- Part 2 of the theorem -/
theorem part_two (a c : ℝ) :
  (∀ x : ℝ, f a c x > 0 ↔ -1 < x ∧ x < 3) →
  ((a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ c = 9) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3663_366314


namespace NUMINAMATH_CALUDE_regular_square_pyramid_volume_l3663_366343

/-- A regular square pyramid with side edge length 2√3 and angle 60° between side edge and base has volume 6 -/
theorem regular_square_pyramid_volume (side_edge : ℝ) (angle : ℝ) : 
  side_edge = 2 * Real.sqrt 3 →
  angle = π / 3 →
  let height := side_edge * Real.sin angle
  let base_area := (side_edge^2) / 2
  let volume := (1/3) * base_area * height
  volume = 6 := by sorry

end NUMINAMATH_CALUDE_regular_square_pyramid_volume_l3663_366343


namespace NUMINAMATH_CALUDE_wrapping_cost_calculation_l3663_366369

/-- Represents the number of boxes a roll of wrapping paper can wrap -/
structure WrapCapacity where
  shirt : ℕ
  xl : ℕ

/-- Represents the number of boxes to be wrapped -/
structure BoxesToWrap where
  shirt : ℕ
  xl : ℕ

/-- Calculates the total cost of wrapping paper needed -/
def totalCost (capacity : WrapCapacity) (boxes : BoxesToWrap) (price_per_roll : ℚ) : ℚ :=
  let rolls_needed_shirt := (boxes.shirt + capacity.shirt - 1) / capacity.shirt
  let rolls_needed_xl := (boxes.xl + capacity.xl - 1) / capacity.xl
  (rolls_needed_shirt + rolls_needed_xl : ℚ) * price_per_roll

theorem wrapping_cost_calculation 
  (capacity : WrapCapacity) 
  (boxes : BoxesToWrap) 
  (price_per_roll : ℚ) :
  capacity.shirt = 5 →
  capacity.xl = 3 →
  boxes.shirt = 20 →
  boxes.xl = 12 →
  price_per_roll = 4 →
  totalCost capacity boxes price_per_roll = 32 :=
sorry

end NUMINAMATH_CALUDE_wrapping_cost_calculation_l3663_366369


namespace NUMINAMATH_CALUDE_sequence_difference_l3663_366312

/-- The sequence a_n with sum S_n = n^2 + 2n for n ∈ ℕ* -/
def S (n : ℕ+) : ℕ := n.val^2 + 2*n.val

/-- The n-th term of the sequence -/
def a (n : ℕ+) : ℕ := 2*n.val + 1

theorem sequence_difference (n m : ℕ+) (h : m.val - n.val = 5) :
  a m - a n = 10 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l3663_366312


namespace NUMINAMATH_CALUDE_second_metal_cost_l3663_366330

/-- Given two metals mixed in equal proportions, prove the cost of the second metal
    when the cost of the first metal and the resulting alloy are known. -/
theorem second_metal_cost (cost_first : ℝ) (cost_alloy : ℝ) : 
  cost_first = 68 → cost_alloy = 82 → 2 * cost_alloy - cost_first = 96 := by
  sorry

end NUMINAMATH_CALUDE_second_metal_cost_l3663_366330


namespace NUMINAMATH_CALUDE_exists_four_mutual_l3663_366320

-- Define a type for people
def Person : Type := ℕ

-- Define a relation for familiarity
def familiar : Person → Person → Prop := sorry

-- Define a group of 18 people
def group : Finset Person := sorry

-- Axiom: The group has exactly 18 people
axiom group_size : Finset.card group = 18

-- Axiom: Any two people are either familiar or unfamiliar
axiom familiar_or_unfamiliar (p q : Person) : p ∈ group → q ∈ group → p ≠ q → 
  familiar p q ∨ ¬familiar p q

-- Theorem to prove
theorem exists_four_mutual (group : Finset Person) 
  (h₁ : Finset.card group = 18) 
  (h₂ : ∀ p q : Person, p ∈ group → q ∈ group → p ≠ q → familiar p q ∨ ¬familiar p q) :
  ∃ (s : Finset Person), Finset.card s = 4 ∧ s ⊆ group ∧
    (∀ p q : Person, p ∈ s → q ∈ s → p ≠ q → familiar p q) ∨
    (∀ p q : Person, p ∈ s → q ∈ s → p ≠ q → ¬familiar p q) :=
sorry

end NUMINAMATH_CALUDE_exists_four_mutual_l3663_366320


namespace NUMINAMATH_CALUDE_f_fixed_points_l3663_366357

def f (x : ℝ) : ℝ := x^2 - 5*x

theorem f_fixed_points : 
  {x : ℝ | f (f x) = f x} = {0, -2, 5, 6} := by sorry

end NUMINAMATH_CALUDE_f_fixed_points_l3663_366357


namespace NUMINAMATH_CALUDE_problem_solution_l3663_366366

/-- Represents the color of a ball -/
inductive Color
| White
| Black

/-- Represents a pair of drawn balls -/
structure DrawnBalls :=
  (first second : Color)

/-- The sample space of all possible outcomes when drawing two balls without replacement -/
def Ω : Finset DrawnBalls := sorry

/-- Event A: drawing two balls of the same color -/
def A : Set DrawnBalls := {db | db.first = db.second}

/-- Event B: the first ball drawn is white -/
def B : Set DrawnBalls := {db | db.first = Color.White}

/-- Event C: the second ball drawn is white -/
def C : Set DrawnBalls := {db | db.second = Color.White}

/-- Event D: drawing two balls of different colors -/
def D : Set DrawnBalls := {db | db.first ≠ db.second}

/-- The probability measure on the sample space -/
noncomputable def P : Set DrawnBalls → ℝ := sorry

theorem problem_solution :
  (P B = 1/2) ∧
  (P (A ∩ B) = P A * P B) ∧
  (A ∪ D = Set.univ) ∧ (A ∩ D = ∅) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3663_366366


namespace NUMINAMATH_CALUDE_sum_squared_distances_coinciding_centroids_l3663_366376

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  leg_length_pos : leg_length > 0

/-- The sum of squared distances between vertices of two triangles -/
def sum_squared_distances (et : EquilateralTriangle) (irt : IsoscelesRightTriangle) : ℝ := 
  3 * et.side_length^2 + 4 * irt.leg_length^2

theorem sum_squared_distances_coinciding_centroids 
  (et : EquilateralTriangle) 
  (irt : IsoscelesRightTriangle) :
  sum_squared_distances et irt = 3 * et.side_length^2 + 4 * irt.leg_length^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_distances_coinciding_centroids_l3663_366376


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_k_value_l3663_366374

/-- Given vectors a, b, and c in R², prove that if (a - c) is parallel to b, then k = 5 -/
theorem parallel_vectors_imply_k_value (a b c : ℝ × ℝ) (k : ℝ) :
  a = (3, 1) →
  b = (1, 3) →
  c = (k, 7) →
  (∃ (t : ℝ), a - c = t • b) →
  k = 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_k_value_l3663_366374


namespace NUMINAMATH_CALUDE_diamond_three_four_l3663_366367

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem diamond_three_four : diamond 3 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_four_l3663_366367


namespace NUMINAMATH_CALUDE_inequality_1_inequality_2_l3663_366350

-- First inequality
theorem inequality_1 (x : ℝ) : (2*x - 1)/3 - (9*x + 2)/6 ≤ 1 ↔ x ≥ -2 := by sorry

-- Second system of inequalities
theorem inequality_2 (x : ℝ) : 
  (x - 3*(x - 2) ≥ 4 ∧ (2*x - 1)/5 < (x + 1)/2) ↔ -7 < x ∧ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_1_inequality_2_l3663_366350


namespace NUMINAMATH_CALUDE_sticker_cost_theorem_l3663_366329

/-- Calculates the total cost of buying stickers on two days -/
def total_sticker_cost (day1_packs : ℕ) (day1_price : ℚ) (day1_discount : ℚ)
                       (day2_packs : ℕ) (day2_price : ℚ) (day2_tax : ℚ) : ℚ :=
  let day1_cost := day1_packs * day1_price * (1 - day1_discount)
  let day2_cost := day2_packs * day2_price * (1 + day2_tax)
  day1_cost + day2_cost

/-- Theorem stating the total cost of buying stickers on two days -/
theorem sticker_cost_theorem :
  total_sticker_cost 15 (5/2) (1/10) 25 3 (1/20) = 225/2 := by
  sorry

end NUMINAMATH_CALUDE_sticker_cost_theorem_l3663_366329


namespace NUMINAMATH_CALUDE_cistern_fill_time_l3663_366337

/-- The time it takes for pipe p to fill the cistern -/
def p_time : ℝ := 10

/-- The time both pipes are opened together -/
def both_open_time : ℝ := 2

/-- The additional time it takes to fill the cistern after pipe p is turned off -/
def additional_time : ℝ := 10

/-- The time it takes for pipe q to fill the cistern -/
def q_time : ℝ := 15

theorem cistern_fill_time : 
  (both_open_time * (1 / p_time + 1 / q_time)) + 
  (additional_time * (1 / q_time)) = 1 := by sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l3663_366337


namespace NUMINAMATH_CALUDE_third_derivative_y_l3663_366379

open Real

noncomputable def y (x : ℝ) : ℝ := (log (2 * x + 5)) / (2 * x + 5)

theorem third_derivative_y (x : ℝ) :
  (deriv^[3] y) x = (88 - 48 * log (2 * x + 5)) / (2 * x + 5)^4 :=
by sorry

end NUMINAMATH_CALUDE_third_derivative_y_l3663_366379


namespace NUMINAMATH_CALUDE_missing_number_odd_l3663_366394

def set_a : Finset Nat := {11, 44, 55}

def is_odd (n : Nat) : Prop := n % 2 = 1

def probability_even_sum (b : Nat) : Rat :=
  (set_a.filter (fun a => (a + b) % 2 = 0)).card / set_a.card

theorem missing_number_odd (b : Nat) :
  probability_even_sum b = 1/2 → is_odd b := by
  sorry

end NUMINAMATH_CALUDE_missing_number_odd_l3663_366394


namespace NUMINAMATH_CALUDE_age_difference_proof_l3663_366387

theorem age_difference_proof (elder_age younger_age : ℕ) : 
  elder_age > younger_age →
  elder_age - 10 = 5 * (younger_age - 10) →
  elder_age = 35 →
  younger_age = 15 →
  elder_age - younger_age = 20 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3663_366387


namespace NUMINAMATH_CALUDE_f_range_l3663_366377

/-- The function f(x) = (x^2-1)(x^2-12x+35) -/
def f (x : ℝ) : ℝ := (x^2 - 1) * (x^2 - 12*x + 35)

/-- The graph of f(x) is symmetric about the line x=3 -/
axiom f_symmetry (x : ℝ) : f (6 - x) = f x

theorem f_range : Set.range f = Set.Ici (-36) := by sorry

end NUMINAMATH_CALUDE_f_range_l3663_366377


namespace NUMINAMATH_CALUDE_function_sum_property_l3663_366391

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - x + 2

theorem function_sum_property (a : ℝ) (h : a ≠ 0) :
  (∀ x₁ ∈ Set.Icc 1 ℯ, ∃ x₂ ∈ Set.Icc 1 ℯ, f a x₁ + f a x₂ = 4) →
  a = ℯ + 1 := by
sorry

end NUMINAMATH_CALUDE_function_sum_property_l3663_366391


namespace NUMINAMATH_CALUDE_f_passes_through_point_two_zero_l3663_366375

-- Define the function f
def f (x : ℝ) : ℝ := x - 2

-- Theorem statement
theorem f_passes_through_point_two_zero : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_passes_through_point_two_zero_l3663_366375


namespace NUMINAMATH_CALUDE_power_of_81_l3663_366368

theorem power_of_81 : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by sorry

end NUMINAMATH_CALUDE_power_of_81_l3663_366368


namespace NUMINAMATH_CALUDE_correct_operation_l3663_366351

theorem correct_operation (a : ℝ) : -a + 5*a = 4*a := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3663_366351


namespace NUMINAMATH_CALUDE_coat_price_proof_l3663_366382

theorem coat_price_proof (price : ℝ) : 
  (price - 250 = price * 0.5) → price = 500 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_proof_l3663_366382


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l3663_366300

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, -4)

-- Define the theorem
theorem intersection_distance_sum :
  ∃ (M N : ℝ × ℝ),
    curve_C M.1 M.2 ∧
    curve_C N.1 N.2 ∧
    line M.1 M.2 ∧
    line N.1 N.2 ∧
    M ≠ N ∧
    Real.sqrt ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2) +
    Real.sqrt ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2) =
    12 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l3663_366300


namespace NUMINAMATH_CALUDE_dog_food_per_meal_l3663_366302

/-- Calculates the amount of dog food each dog eats per meal given the total amount bought,
    amount left after a week, number of dogs, and number of meals per day. -/
theorem dog_food_per_meal
  (total_food : ℝ)
  (food_left : ℝ)
  (num_dogs : ℕ)
  (meals_per_day : ℕ)
  (days_per_week : ℕ)
  (h1 : total_food = 30)
  (h2 : food_left = 9)
  (h3 : num_dogs = 3)
  (h4 : meals_per_day = 2)
  (h5 : days_per_week = 7)
  : (total_food - food_left) / (num_dogs * meals_per_day * days_per_week) = 0.5 := by
  sorry

#check dog_food_per_meal

end NUMINAMATH_CALUDE_dog_food_per_meal_l3663_366302


namespace NUMINAMATH_CALUDE_basketball_win_rate_l3663_366384

theorem basketball_win_rate (games_won_first_half : ℕ) (total_games : ℕ) (desired_win_rate : ℚ) (games_to_win : ℕ) : 
  games_won_first_half = 30 →
  total_games = 80 →
  desired_win_rate = 3/4 →
  games_to_win = 30 →
  (games_won_first_half + games_to_win : ℚ) / total_games = desired_win_rate :=
by
  sorry

#check basketball_win_rate

end NUMINAMATH_CALUDE_basketball_win_rate_l3663_366384


namespace NUMINAMATH_CALUDE_two_candles_burn_time_l3663_366364

/-- Burning time of candle 1 -/
def burn_time_1 : ℕ := 30

/-- Burning time of candle 2 -/
def burn_time_2 : ℕ := 40

/-- Burning time of candle 3 -/
def burn_time_3 : ℕ := 50

/-- Time all three candles burn simultaneously -/
def time_all_three : ℕ := 10

/-- Time only one candle burns -/
def time_one_candle : ℕ := 20

/-- Theorem stating that exactly two candles burn simultaneously for 35 minutes -/
theorem two_candles_burn_time :
  (burn_time_1 + burn_time_2 + burn_time_3) - (3 * time_all_three + time_one_candle) = 70 :=
by sorry

end NUMINAMATH_CALUDE_two_candles_burn_time_l3663_366364


namespace NUMINAMATH_CALUDE_last_year_ticket_cost_l3663_366335

/-- 
Proves that the ticket cost last year was $85, given that this year's cost 
is $102 and represents a 20% increase from last year.
-/
theorem last_year_ticket_cost : 
  ∀ (last_year_cost : ℝ), 
  (last_year_cost + 0.2 * last_year_cost = 102) → 
  last_year_cost = 85 := by
sorry

end NUMINAMATH_CALUDE_last_year_ticket_cost_l3663_366335


namespace NUMINAMATH_CALUDE_negation_of_conditional_l3663_366321

theorem negation_of_conditional (a : ℝ) :
  ¬(a > 0 → a^2 > 0) ↔ (a ≤ 0 → a^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l3663_366321


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l3663_366365

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l3663_366365


namespace NUMINAMATH_CALUDE_consecutive_factorials_divisible_by_61_l3663_366385

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem consecutive_factorials_divisible_by_61 (k : ℕ) :
  (∃ m : ℕ, factorial (k - 2) + factorial (k - 1) + factorial k = 61 * m) →
  k ≥ 61 := by
sorry

end NUMINAMATH_CALUDE_consecutive_factorials_divisible_by_61_l3663_366385


namespace NUMINAMATH_CALUDE_definite_integral_2x_minus_1_l3663_366327

theorem definite_integral_2x_minus_1 :
  ∫ x in (0:ℝ)..3, (2*x - 1) = 6 := by sorry

end NUMINAMATH_CALUDE_definite_integral_2x_minus_1_l3663_366327


namespace NUMINAMATH_CALUDE_credits_needed_is_84_l3663_366339

/-- The number of credits needed to buy cards for a game -/
def credits_needed : ℕ :=
  let red_card_cost : ℕ := 3
  let blue_card_cost : ℕ := 5
  let total_cards_required : ℕ := 20
  let red_cards_owned : ℕ := 8
  let blue_cards_needed : ℕ := total_cards_required - red_cards_owned
  red_card_cost * red_cards_owned + blue_card_cost * blue_cards_needed

theorem credits_needed_is_84 : credits_needed = 84 := by
  sorry

end NUMINAMATH_CALUDE_credits_needed_is_84_l3663_366339


namespace NUMINAMATH_CALUDE_min_pieces_for_special_l3663_366311

/-- Represents a piece of the pie -/
inductive PieceType
| Empty
| Fish
| Sausage
| Special

/-- Represents the 8x8 pie grid -/
def Pie := Fin 8 → Fin 8 → PieceType

/-- Checks if a 6x6 square in the pie has at least 2 fish pieces -/
def has_two_fish (p : Pie) (i j : Fin 8) : Prop :=
  ∃ (i1 j1 i2 j2 : Fin 8),
    i1 < i + 6 ∧ j1 < j + 6 ∧ i2 < i + 6 ∧ j2 < j + 6 ∧
    (i1 ≠ i2 ∨ j1 ≠ j2) ∧
    p i1 j1 = PieceType.Fish ∧ p i2 j2 = PieceType.Fish

/-- Checks if a 3x3 square in the pie has at most 1 sausage piece -/
def at_most_one_sausage (p : Pie) (i j : Fin 8) : Prop :=
  ∀ (i1 j1 i2 j2 : Fin 8),
    i1 < i + 3 → j1 < j + 3 → i2 < i + 3 → j2 < j + 3 →
    p i1 j1 = PieceType.Sausage → p i2 j2 = PieceType.Sausage →
    i1 = i2 ∧ j1 = j2

/-- Defines a valid pie configuration -/
def valid_pie (p : Pie) : Prop :=
  (∃ (i1 j1 i2 j2 i3 j3 : Fin 8),
     p i1 j1 = PieceType.Fish ∧ p i2 j2 = PieceType.Fish ∧ p i3 j3 = PieceType.Fish ∧
     (i1 ≠ i2 ∨ j1 ≠ j2) ∧ (i1 ≠ i3 ∨ j1 ≠ j3) ∧ (i2 ≠ i3 ∨ j2 ≠ j3)) ∧
  (∃ (i1 j1 i2 j2 : Fin 8),
     p i1 j1 = PieceType.Sausage ∧ p i2 j2 = PieceType.Sausage ∧
     (i1 ≠ i2 ∨ j1 ≠ j2)) ∧
  (∃! (i j : Fin 8), p i j = PieceType.Special) ∧
  (∀ (i j : Fin 8), has_two_fish p i j) ∧
  (∀ (i j : Fin 8), at_most_one_sausage p i j)

/-- Theorem: The minimum number of pieces to guarantee getting the special piece is 5 -/
theorem min_pieces_for_special (p : Pie) (h : valid_pie p) :
  ∀ (s : Finset (Fin 8 × Fin 8)),
    s.card < 5 → ∃ (i j : Fin 8), p i j = PieceType.Special ∧ (i, j) ∉ s :=
sorry

end NUMINAMATH_CALUDE_min_pieces_for_special_l3663_366311


namespace NUMINAMATH_CALUDE_four_digit_greater_than_product_l3663_366309

theorem four_digit_greater_than_product (a b c d : ℕ) : 
  a ≤ 9 → b ≤ 9 → c ≤ 9 → d ≤ 9 → 
  (1000 * a + 100 * b + 10 * c + d > (10 * a + b) * (10 * c + d)) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_greater_than_product_l3663_366309


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l3663_366328

open Set Real

theorem negation_of_existence_inequality :
  (¬ ∃ x : ℝ, x^2 - 5*x - 6 < 0) ↔ (∀ x : ℝ, x^2 - 5*x - 6 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l3663_366328


namespace NUMINAMATH_CALUDE_f_shifted_passes_through_point_one_zero_l3663_366373

-- Define the function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Define the shifted function f(x-1)
def f_shifted (a : ℝ) (x : ℝ) : ℝ := f a (x - 1)

-- Theorem statement
theorem f_shifted_passes_through_point_one_zero (a : ℝ) :
  f_shifted a 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_shifted_passes_through_point_one_zero_l3663_366373


namespace NUMINAMATH_CALUDE_set_of_positive_rationals_l3663_366310

def is_closed_under_addition_and_multiplication (S : Set ℚ) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a * b) ∈ S

def has_trichotomy_property (S : Set ℚ) : Prop :=
  ∀ r : ℚ, (r ∈ S ∧ -r ∉ S ∧ r ≠ 0) ∨ (-r ∈ S ∧ r ∉ S ∧ r ≠ 0) ∨ (r = 0 ∧ r ∉ S ∧ -r ∉ S)

theorem set_of_positive_rationals (S : Set ℚ) 
  (h1 : is_closed_under_addition_and_multiplication S)
  (h2 : has_trichotomy_property S) :
  S = {r : ℚ | 0 < r} :=
sorry

end NUMINAMATH_CALUDE_set_of_positive_rationals_l3663_366310


namespace NUMINAMATH_CALUDE_family_ages_l3663_366349

theorem family_ages :
  ∀ (dad mom kolya tanya : ℕ),
    dad = mom + 4 →
    kolya = tanya + 4 →
    2 * kolya = dad →
    dad + mom + kolya + tanya = 130 →
    dad = 46 ∧ mom = 42 ∧ kolya = 23 ∧ tanya = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_family_ages_l3663_366349


namespace NUMINAMATH_CALUDE_marks_profit_l3663_366397

/-- The profit Mark makes from selling a Magic card -/
def profit (initial_cost : ℝ) (value_multiplier : ℝ) : ℝ :=
  initial_cost * value_multiplier - initial_cost

/-- Theorem stating that Mark's profit is $200 -/
theorem marks_profit : profit 100 3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_marks_profit_l3663_366397


namespace NUMINAMATH_CALUDE_bills_earnings_l3663_366383

/-- Represents the earnings from dairy products -/
def dairy_earnings (total_milk : ℚ) (butter_ratio : ℚ) (sour_cream_ratio : ℚ) 
  (milk_to_butter : ℚ) (milk_to_sour_cream : ℚ) 
  (butter_price : ℚ) (sour_cream_price : ℚ) (milk_price : ℚ) : ℚ :=
  let butter_milk := total_milk * butter_ratio
  let sour_cream_milk := total_milk * sour_cream_ratio
  let whole_milk := total_milk - butter_milk - sour_cream_milk
  let butter_gallons := butter_milk / milk_to_butter
  let sour_cream_gallons := sour_cream_milk / milk_to_sour_cream
  butter_gallons * butter_price + sour_cream_gallons * sour_cream_price + whole_milk * milk_price

/-- Bill's earnings from his dairy products -/
theorem bills_earnings : 
  dairy_earnings 16 (1/4) (1/4) 4 2 5 6 3 = 41 := by
  sorry

end NUMINAMATH_CALUDE_bills_earnings_l3663_366383


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3663_366304

/-- Given two vectors in R², prove that the magnitude of their difference is 5. -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3663_366304


namespace NUMINAMATH_CALUDE_bulb_arrangement_count_l3663_366306

/-- The number of ways to arrange blue and red bulbs -/
def arrange_blue_red : ℕ := Nat.choose 16 8

/-- The number of ways to place white bulbs between blue and red bulbs -/
def place_white : ℕ := Nat.choose 17 11

/-- The total number of blue bulbs -/
def blue_bulbs : ℕ := 8

/-- The total number of red bulbs -/
def red_bulbs : ℕ := 8

/-- The total number of white bulbs -/
def white_bulbs : ℕ := 11

/-- The theorem stating the number of ways to arrange the bulbs -/
theorem bulb_arrangement_count :
  arrange_blue_red * place_white = 159279120 :=
sorry

end NUMINAMATH_CALUDE_bulb_arrangement_count_l3663_366306


namespace NUMINAMATH_CALUDE_max_cube_volume_from_sheet_l3663_366356

/-- Given a rectangular sheet of dimensions 60 cm by 25 cm, 
    prove that the maximum volume of a cube that can be constructed from this sheet is 3375 cm³. -/
theorem max_cube_volume_from_sheet (sheet_length : ℝ) (sheet_width : ℝ) 
  (h_length : sheet_length = 60) (h_width : sheet_width = 25) :
  ∃ (cube_edge : ℝ), 
    cube_edge > 0 ∧
    6 * cube_edge^2 ≤ sheet_length * sheet_width ∧
    ∀ (other_edge : ℝ), 
      other_edge > 0 → 
      6 * other_edge^2 ≤ sheet_length * sheet_width → 
      other_edge^3 ≤ cube_edge^3 ∧
    cube_edge^3 = 3375 := by
  sorry

end NUMINAMATH_CALUDE_max_cube_volume_from_sheet_l3663_366356


namespace NUMINAMATH_CALUDE_no_solution_when_n_negative_one_l3663_366316

-- Define the system of equations
def system (n x y z : ℝ) : Prop :=
  n * x^2 + y = 2 ∧ n * y^2 + z = 2 ∧ n * z^2 + x = 2

-- Theorem stating that the system has no solution when n = -1
theorem no_solution_when_n_negative_one :
  ¬ ∃ (x y z : ℝ), system (-1) x y z :=
sorry

end NUMINAMATH_CALUDE_no_solution_when_n_negative_one_l3663_366316


namespace NUMINAMATH_CALUDE_first_group_size_l3663_366307

/-- The number of days it takes the first group to complete the work -/
def first_group_days : ℕ := 30

/-- The number of days it takes 20 men to complete the work -/
def second_group_days : ℕ := 24

/-- The number of men in the second group -/
def second_group_men : ℕ := 20

/-- The number of men in the first group -/
def first_group_men : ℕ := (second_group_men * second_group_days) / first_group_days

theorem first_group_size :
  first_group_men = 16 :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l3663_366307


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3663_366378

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x^2 + m*x + m - 2
  (f (-2) = 0) →
  (∃ x, x ≠ -2 ∧ f x = 0 ∧ x = 0) ∧
  (∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_equation_properties_l3663_366378


namespace NUMINAMATH_CALUDE_smallest_n_for_no_real_roots_l3663_366322

theorem smallest_n_for_no_real_roots :
  ∀ n : ℤ, (∀ x : ℝ, 3 * x * (n * x + 3) - 2 * x^2 - 9 ≠ 0) →
  n ≥ -1 ∧ ∀ m : ℤ, m < -1 → ∃ x : ℝ, 3 * x * (m * x + 3) - 2 * x^2 - 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_no_real_roots_l3663_366322


namespace NUMINAMATH_CALUDE_wraps_percentage_increase_l3663_366392

/-- Given John's raw squat weight, the additional weight from sleeves, and the difference between wraps and sleeves, 
    calculate the percentage increase wraps provide to his raw squat. -/
theorem wraps_percentage_increase 
  (raw_squat : ℝ) 
  (sleeves_addition : ℝ) 
  (wraps_vs_sleeves_difference : ℝ) 
  (h1 : raw_squat = 600) 
  (h2 : sleeves_addition = 30) 
  (h3 : wraps_vs_sleeves_difference = 120) : 
  (raw_squat + sleeves_addition + wraps_vs_sleeves_difference - raw_squat) / raw_squat * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_wraps_percentage_increase_l3663_366392


namespace NUMINAMATH_CALUDE_problem_statement_l3663_366317

theorem problem_statement (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_condition : (a / (1 + a)) + (b / (1 + b)) = 1) :
  (a / (1 + b^2)) - (b / (1 + a^2)) = a - b := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3663_366317


namespace NUMINAMATH_CALUDE_expression_simplification_l3663_366390

theorem expression_simplification :
  let a := 2016
  let b := 2017
  (a^4 - 2*a^3*b + 3*a^2*b^2 - a*b^3 + 1) / (a^2 * b^2) = 1 - 1 / b^2 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3663_366390


namespace NUMINAMATH_CALUDE_equal_intersection_areas_exist_l3663_366386

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  opposite_edges_perpendicular : Bool
  opposite_edges_horizontal : Bool
  vertical_midline : Bool

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Represents the configuration of a tetrahedron and a sphere -/
structure Configuration where
  tetrahedron : Tetrahedron
  sphere : Sphere
  sphere_centered_on_midline : Bool

/-- Represents a horizontal plane -/
structure HorizontalPlane where
  height : ℝ

/-- Function to calculate the area of intersection between a horizontal plane and the tetrahedron -/
def tetrahedron_intersection_area (t : Tetrahedron) (p : HorizontalPlane) : ℝ := sorry

/-- Function to calculate the area of intersection between a horizontal plane and the sphere -/
def sphere_intersection_area (s : Sphere) (p : HorizontalPlane) : ℝ := sorry

/-- Theorem stating that there exists a configuration where all horizontal plane intersections have equal areas -/
theorem equal_intersection_areas_exist : 
  ∃ (c : Configuration), ∀ (p : HorizontalPlane), 
    tetrahedron_intersection_area c.tetrahedron p = sphere_intersection_area c.sphere p :=
sorry

end NUMINAMATH_CALUDE_equal_intersection_areas_exist_l3663_366386


namespace NUMINAMATH_CALUDE_roof_area_l3663_366355

/-- Calculates the area of a rectangular roof given the conditions --/
theorem roof_area (width : ℝ) (length : ℝ) : 
  length = 4 * width → 
  length - width = 42 → 
  width * length = 784 := by
  sorry

end NUMINAMATH_CALUDE_roof_area_l3663_366355


namespace NUMINAMATH_CALUDE_figures_per_shelf_is_eleven_l3663_366363

/-- The number of shelves in Adam's room -/
def num_shelves : ℕ := 4

/-- The total number of action figures that can fit on all shelves -/
def total_figures : ℕ := 44

/-- The number of action figures that can fit on each shelf -/
def figures_per_shelf : ℕ := total_figures / num_shelves

/-- Theorem: The number of action figures that can fit on each shelf is 11 -/
theorem figures_per_shelf_is_eleven : figures_per_shelf = 11 := by
  sorry

end NUMINAMATH_CALUDE_figures_per_shelf_is_eleven_l3663_366363


namespace NUMINAMATH_CALUDE_janes_profit_is_correct_l3663_366347

/-- Farm data -/
structure FarmData where
  chickenCount : ℕ
  duckCount : ℕ
  quailCount : ℕ
  chickenEggsPerWeek : ℕ
  duckEggsPerWeek : ℕ
  quailEggsPerWeek : ℕ
  chickenEggPrice : ℚ
  duckEggPrice : ℚ
  quailEggPrice : ℚ
  chickenFeedCost : ℚ
  duckFeedCost : ℚ
  quailFeedCost : ℚ

/-- Sales data for a week -/
structure WeeklySales where
  chickenEggsSoldPercent : ℚ
  duckEggsSoldPercent : ℚ
  quailEggsSoldPercent : ℚ

def calculateProfit (farm : FarmData) (sales : List WeeklySales) : ℚ :=
  sorry

def janesFarm : FarmData := {
  chickenCount := 10,
  duckCount := 8,
  quailCount := 12,
  chickenEggsPerWeek := 6,
  duckEggsPerWeek := 4,
  quailEggsPerWeek := 10,
  chickenEggPrice := 2 / 12,
  duckEggPrice := 3 / 12,
  quailEggPrice := 4 / 12,
  chickenFeedCost := 1 / 2,
  duckFeedCost := 3 / 4,
  quailFeedCost := 3 / 5
}

def janesSales : List WeeklySales := [
  { chickenEggsSoldPercent := 1, duckEggsSoldPercent := 1, quailEggsSoldPercent := 1/2 },
  { chickenEggsSoldPercent := 1, duckEggsSoldPercent := 3/4, quailEggsSoldPercent := 1 },
  { chickenEggsSoldPercent := 0, duckEggsSoldPercent := 1, quailEggsSoldPercent := 1 }
]

theorem janes_profit_is_correct :
  calculateProfit janesFarm janesSales = 876 / 10 := by
  sorry

end NUMINAMATH_CALUDE_janes_profit_is_correct_l3663_366347


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3663_366358

theorem geometric_series_sum (a b : ℝ) (h : b ≠ 1) (h2 : b ≠ 0) :
  (∑' n, a / b^n) = 2 →
  (∑' n, a / (2*a + b)^n) = 2/5 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3663_366358


namespace NUMINAMATH_CALUDE_distance_to_focus_l3663_366308

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus (x : ℝ) : 
  x^2 = 16 → -- Point A(x, 4) is on the parabola x^2 = 4y
  ∃ (f : ℝ × ℝ), -- There exists a focus f
    (∀ (p : ℝ × ℝ), p.2 = p.1^2 / 4 → dist p f = p.2 + 1) ∧ -- Definition of parabola
    dist (x, 4) f = 5 -- The distance from A to the focus is 5
:= by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l3663_366308


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3663_366336

theorem rectangular_prism_diagonal (a b c : ℝ) (ha : a = 12) (hb : b = 15) (hc : c = 8) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 433 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3663_366336


namespace NUMINAMATH_CALUDE_dictionary_cost_l3663_366338

def total_cost : ℕ := 8 + 29
def dinosaur_book_cost : ℕ := 19
def cookbook_cost : ℕ := 7
def savings : ℕ := 8
def additional_needed : ℕ := 29

theorem dictionary_cost : 
  total_cost - (dinosaur_book_cost + cookbook_cost) = 11 := by
sorry

end NUMINAMATH_CALUDE_dictionary_cost_l3663_366338


namespace NUMINAMATH_CALUDE_a_7_not_prime_l3663_366359

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Sequence defined by the recursive formula -/
def a : ℕ → ℕ
  | 0 => 1  -- a_1 is a positive integer
  | n + 1 => a n + reverseDigits (a n)

theorem a_7_not_prime : ∃ (k : ℕ), k > 1 ∧ k < a 7 ∧ a 7 % k = 0 := by sorry

end NUMINAMATH_CALUDE_a_7_not_prime_l3663_366359


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l3663_366380

theorem unique_x_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → (r = p ∨ r = q ∨ r = 11))) →
  11 ∣ x →
  x = 59048 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l3663_366380


namespace NUMINAMATH_CALUDE_f_domain_l3663_366318

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + 1 / (2 - x)

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

theorem f_domain : domain f = {x : ℝ | x ≥ -1 ∧ x ≠ 2} := by
  sorry

end NUMINAMATH_CALUDE_f_domain_l3663_366318


namespace NUMINAMATH_CALUDE_function_upper_bound_condition_l3663_366353

theorem function_upper_bound_condition (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → a * x - x^2 ≤ 1) ↔ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_function_upper_bound_condition_l3663_366353


namespace NUMINAMATH_CALUDE_solution_form_l3663_366399

open Real

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 1 → y > 1 → f x - f y = (y - x) * f (x * y)

/-- The theorem stating that any function satisfying the equation must be of the form k/x -/
theorem solution_form (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    ∃ k : ℝ, ∀ x, x > 1 → f x = k / x := by
  sorry

end NUMINAMATH_CALUDE_solution_form_l3663_366399


namespace NUMINAMATH_CALUDE_fish_tank_count_l3663_366334

theorem fish_tank_count : 
  ∀ (n : ℕ) (first_tank : ℕ) (other_tanks : ℕ),
    n = 3 →
    first_tank = 20 →
    other_tanks = 2 * first_tank →
    first_tank + (n - 1) * other_tanks = 100 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_count_l3663_366334


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3663_366345

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  0.000000023 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.3 ∧ n = -8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3663_366345


namespace NUMINAMATH_CALUDE_heesu_has_greatest_sum_l3663_366319

def sora_numbers : List ℕ := [4, 6]
def heesu_numbers : List ℕ := [7, 5]
def jiyeon_numbers : List ℕ := [3, 8]

def sum_list (l : List ℕ) : ℕ := l.sum

theorem heesu_has_greatest_sum :
  sum_list heesu_numbers > sum_list sora_numbers ∧
  sum_list heesu_numbers > sum_list jiyeon_numbers :=
by sorry

end NUMINAMATH_CALUDE_heesu_has_greatest_sum_l3663_366319


namespace NUMINAMATH_CALUDE_house_sale_loss_l3663_366370

theorem house_sale_loss (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : 
  initial_value = 12000 ∧ 
  loss_percent = 0.1 ∧ 
  gain_percent = 0.15 → 
  initial_value * (1 - loss_percent) * (1 + gain_percent) - initial_value = 420 :=
by sorry

end NUMINAMATH_CALUDE_house_sale_loss_l3663_366370


namespace NUMINAMATH_CALUDE_book_recipient_sequences_l3663_366344

theorem book_recipient_sequences (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 3) :
  (n * (n - 1) * (n - 2)) = 2730 :=
by sorry

end NUMINAMATH_CALUDE_book_recipient_sequences_l3663_366344


namespace NUMINAMATH_CALUDE_f_monotonicity_and_max_k_l3663_366396

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + Real.log x) / x

theorem f_monotonicity_and_max_k :
  (∀ m : ℝ, m ≥ 1 → ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f m x₁ > f m x₂) ∧
  (∀ m : ℝ, m < 1 → ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → x₂ ≤ Real.exp (1 - m) → f m x₁ < f m x₂) ∧
  (∀ m : ℝ, m < 1 → ∀ x₁ x₂ : ℝ, Real.exp (1 - m) < x₁ → x₁ < x₂ → f m x₁ > f m x₂) ∧
  (∀ x : ℝ, x > 1 → 6 / (x + 1) < f 4 x) ∧
  (∀ k : ℕ, k > 6 → ∃ x : ℝ, x > 1 ∧ k / (x + 1) ≥ f 4 x) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_max_k_l3663_366396


namespace NUMINAMATH_CALUDE_quadratic_downwards_condition_l3663_366333

/-- A quadratic function of the form y = (2a-6)x^2 + 4 -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (2*a - 6)*x^2 + 4

/-- The condition for a quadratic function to open downwards -/
def opens_downwards (a : ℝ) : Prop := 2*a - 6 < 0

theorem quadratic_downwards_condition (a : ℝ) :
  opens_downwards a → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_downwards_condition_l3663_366333


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l3663_366354

theorem concentric_circles_radii_difference 
  (r R : ℝ) 
  (h_positive : r > 0) 
  (h_ratio : π * R^2 = 4 * π * r^2) : 
  R - r = r := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l3663_366354


namespace NUMINAMATH_CALUDE_sum_of_specific_digits_l3663_366323

/-- A sequence where each positive integer n is repeated n times in increasing order -/
def special_sequence : ℕ → ℕ
  | 0 => 0
  | n + 1 => sorry

/-- The nth digit of the special sequence -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the 4501st and 4052nd digits of the special sequence is 13 -/
theorem sum_of_specific_digits :
  nth_digit 4501 + nth_digit 4052 = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_digits_l3663_366323
