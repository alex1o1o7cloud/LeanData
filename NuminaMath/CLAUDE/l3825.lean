import Mathlib

namespace NUMINAMATH_CALUDE_kth_roots_sum_power_real_l3825_382509

theorem kth_roots_sum_power_real (k : ℕ) (x y : ℂ) 
  (hx : x^k = 1) (hy : y^k = 1) : 
  ∃ (r : ℝ), (x + y)^k = r := by sorry

end NUMINAMATH_CALUDE_kth_roots_sum_power_real_l3825_382509


namespace NUMINAMATH_CALUDE_bucket_capacity_problem_l3825_382586

theorem bucket_capacity_problem (tank_capacity : ℝ) (first_case_buckets : ℕ) (second_case_buckets : ℕ) (second_case_capacity : ℝ) :
  first_case_buckets = 13 →
  second_case_buckets = 39 →
  second_case_capacity = 17 →
  tank_capacity = first_case_buckets * (tank_capacity / first_case_buckets) →
  tank_capacity = second_case_buckets * second_case_capacity →
  tank_capacity / first_case_buckets = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_problem_l3825_382586


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3825_382505

theorem complex_fraction_simplification :
  let a := 6 + 7 / 2015
  let b := 4 + 5 / 2016
  let c := 7 + 2008 / 2015
  let d := 2 + 2011 / 2016
  let expression := a * b - c * d - 7 * (7 / 2015)
  expression = 5 / 144 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3825_382505


namespace NUMINAMATH_CALUDE_train_speed_correct_l3825_382500

/-- The speed of the train in km/hr given the conditions -/
def train_speed : ℝ :=
  let train_length : ℝ := 110  -- meters
  let passing_time : ℝ := 4.399648028157747  -- seconds
  let man_speed : ℝ := 6  -- km/hr
  84  -- km/hr

/-- Theorem stating that the calculated train speed is correct -/
theorem train_speed_correct :
  let train_length : ℝ := 110  -- meters
  let passing_time : ℝ := 4.399648028157747  -- seconds
  let man_speed : ℝ := 6  -- km/hr
  train_speed = 84 := by sorry

end NUMINAMATH_CALUDE_train_speed_correct_l3825_382500


namespace NUMINAMATH_CALUDE_functional_relationship_max_profit_remaining_profit_range_l3825_382501

-- Define the constants and variables
def cost_price : ℝ := 40
def min_selling_price : ℝ := 44
def max_selling_price : ℝ := 52
def initial_sales : ℝ := 300
def price_increase : ℝ := 1
def sales_decrease : ℝ := 10
def donation : ℝ := 200
def min_remaining_profit : ℝ := 2200

-- Define the functional relationship
def sales (x : ℝ) : ℝ := -10 * x + 740

-- Define the profit function
def profit (x : ℝ) : ℝ := (sales x) * (x - cost_price)

-- State the theorems to be proved
theorem functional_relationship (x : ℝ) (h : min_selling_price ≤ x ∧ x ≤ max_selling_price) :
  sales x = -10 * x + 740 := by sorry

theorem max_profit :
  ∃ (max_x : ℝ), max_x = max_selling_price ∧
  ∀ (x : ℝ), min_selling_price ≤ x ∧ x ≤ max_selling_price →
  profit x ≤ profit max_x ∧ profit max_x = 2640 := by sorry

theorem remaining_profit_range :
  ∀ (x : ℝ), 50 ≤ x ∧ x ≤ 52 ↔ profit x - donation ≥ min_remaining_profit := by sorry

end NUMINAMATH_CALUDE_functional_relationship_max_profit_remaining_profit_range_l3825_382501


namespace NUMINAMATH_CALUDE_padic_square_root_solutions_l3825_382596

/-- The number of solutions to x^2 = a in p-adic numbers is either 0 or 2 -/
theorem padic_square_root_solutions (p : ℕ) [Fact (Nat.Prime p)] (a : ℚ_[p]) :
  (∃ x y : ℚ_[p], x ^ 2 = a ∧ y ^ 2 = a ∧ x ≠ y) ∨ (∀ x : ℚ_[p], x ^ 2 ≠ a) :=
sorry

end NUMINAMATH_CALUDE_padic_square_root_solutions_l3825_382596


namespace NUMINAMATH_CALUDE_projection_of_a_onto_b_l3825_382522

def vector_a : ℝ × ℝ := (-1, 1)
def vector_b : ℝ × ℝ := (3, 4)

theorem projection_of_a_onto_b :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) / Real.sqrt (vector_b.1^2 + vector_b.2^2) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_projection_of_a_onto_b_l3825_382522


namespace NUMINAMATH_CALUDE_max_digit_sum_l3825_382512

def DigitalClock := Fin 24 × Fin 60

def digit_sum (time : DigitalClock) : Nat :=
  let (h, m) := time
  let h1 := h.val / 10
  let h2 := h.val % 10
  let m1 := m.val / 10
  let m2 := m.val % 10
  h1 + h2 + m1 + m2

theorem max_digit_sum :
  ∃ (max_time : DigitalClock), ∀ (time : DigitalClock), digit_sum time ≤ digit_sum max_time ∧ digit_sum max_time = 19 := by
  sorry

end NUMINAMATH_CALUDE_max_digit_sum_l3825_382512


namespace NUMINAMATH_CALUDE_import_value_calculation_l3825_382547

theorem import_value_calculation (export_value import_value : ℝ) : 
  export_value = 8.07 ∧ 
  export_value = 1.5 * import_value + 1.11 → 
sorry

end NUMINAMATH_CALUDE_import_value_calculation_l3825_382547


namespace NUMINAMATH_CALUDE_age_difference_value_l3825_382559

/-- Represents the ages of three individuals and their relationships -/
structure AgeRelationship where
  /-- Age of Ramesh -/
  x : ℚ
  /-- Age of Suresh -/
  y : ℚ
  /-- Ratio of Ramesh's age to Suresh's age is 2:y -/
  age_ratio : 2 * x = y
  /-- 20 years later, ratio of Ramesh's age to Suresh's age is 8:3 -/
  future_ratio : (5 * x + 20) / (y + 20) = 8 / 3

/-- The difference between Mahesh's and Suresh's present ages -/
def age_difference (ar : AgeRelationship) : ℚ :=
  5 * ar.x - ar.y

/-- Theorem stating the difference between Mahesh's and Suresh's present ages -/
theorem age_difference_value (ar : AgeRelationship) :
  age_difference ar = 125 / 8 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_value_l3825_382559


namespace NUMINAMATH_CALUDE_least_number_to_add_for_divisibility_l3825_382518

theorem least_number_to_add_for_divisibility (n m : ℕ) (h : n = 1076 ∧ m = 23) :
  ∃ x : ℕ, (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_least_number_to_add_for_divisibility_l3825_382518


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3825_382517

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_cond1 : a 7 - 2 * a 4 = -1)
  (h_cond2 : a 3 = 0) :
  ∃ d : ℚ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -1/2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3825_382517


namespace NUMINAMATH_CALUDE_cassie_water_refills_l3825_382583

/-- Represents the number of cups of water Cassie aims to drink daily -/
def daily_cups : ℕ := 12

/-- Represents the capacity of Cassie's water bottle in ounces -/
def bottle_capacity : ℕ := 16

/-- Represents the number of ounces in a cup -/
def ounces_per_cup : ℕ := 8

/-- Represents the number of times Cassie needs to refill her water bottle -/
def refills : ℕ := 6

/-- Theorem stating that Cassie needs to refill her water bottle 6 times
    to meet her daily water intake goal -/
theorem cassie_water_refills :
  (daily_cups * ounces_per_cup) / bottle_capacity = refills :=
by sorry

end NUMINAMATH_CALUDE_cassie_water_refills_l3825_382583


namespace NUMINAMATH_CALUDE_phone_repair_cost_l3825_382516

theorem phone_repair_cost (laptop_cost computer_cost : ℕ) 
  (phone_repairs laptop_repairs computer_repairs : ℕ) (total_earnings : ℕ) :
  laptop_cost = 15 →
  computer_cost = 18 →
  phone_repairs = 5 →
  laptop_repairs = 2 →
  computer_repairs = 2 →
  total_earnings = 121 →
  ∃ (phone_cost : ℕ), 
    phone_cost * phone_repairs + 
    laptop_cost * laptop_repairs + 
    computer_cost * computer_repairs = total_earnings ∧
    phone_cost = 11 :=
by sorry

end NUMINAMATH_CALUDE_phone_repair_cost_l3825_382516


namespace NUMINAMATH_CALUDE_smallest_number_in_sequence_l3825_382550

theorem smallest_number_in_sequence (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive integers
  (a + b + c) / 3 = 30 →  -- arithmetic mean is 30
  b = 28 →  -- median is 28
  c = b + 6 →  -- largest number is 6 more than median
  a ≤ b ∧ a ≤ c →  -- a is the smallest number
  a = 28 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_in_sequence_l3825_382550


namespace NUMINAMATH_CALUDE_sqrt_7_simplest_l3825_382594

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≥ 0 → (∃ n : ℕ, y = n ^ 2 * x) → y = x

theorem sqrt_7_simplest :
  is_simplest_quadratic_radical 7 ∧
  ¬ is_simplest_quadratic_radical 9 ∧
  ¬ is_simplest_quadratic_radical 12 ∧
  ¬ is_simplest_quadratic_radical (2/3) :=
sorry

end NUMINAMATH_CALUDE_sqrt_7_simplest_l3825_382594


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3825_382567

theorem right_triangle_hypotenuse (a b : ℝ) (ha : a = 24) (hb : b = 32) :
  Real.sqrt (a^2 + b^2) = 40 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3825_382567


namespace NUMINAMATH_CALUDE_cos_210_degrees_l3825_382504

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l3825_382504


namespace NUMINAMATH_CALUDE_range_of_set_A_l3825_382571

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def set_A : Set ℕ := {p | 15 < p ∧ p < 36 ∧ is_prime p}

theorem range_of_set_A : 
  ∃ (min max : ℕ), min ∈ set_A ∧ max ∈ set_A ∧ 
  (∀ x ∈ set_A, min ≤ x ∧ x ≤ max) ∧
  max - min = 14 :=
sorry

end NUMINAMATH_CALUDE_range_of_set_A_l3825_382571


namespace NUMINAMATH_CALUDE_cloth_seller_gain_percentage_l3825_382502

/-- Calculates the gain percentage for a cloth seller -/
theorem cloth_seller_gain_percentage 
  (total_cloth : ℝ) 
  (profit_cloth : ℝ) 
  (total_cloth_positive : total_cloth > 0)
  (profit_ratio : profit_cloth = total_cloth / 3) :
  (profit_cloth / total_cloth) * 100 = 100 / 3 := by
sorry

end NUMINAMATH_CALUDE_cloth_seller_gain_percentage_l3825_382502


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l3825_382521

theorem square_root_equation_solution (x : ℝ) (h1 : x ≠ 0) (h2 : Real.sqrt ((7 * x) / 5) = x) : x = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l3825_382521


namespace NUMINAMATH_CALUDE_no_natural_solution_for_square_difference_2014_l3825_382597

theorem no_natural_solution_for_square_difference_2014 :
  ∀ (m n : ℕ), m^2 ≠ n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_for_square_difference_2014_l3825_382597


namespace NUMINAMATH_CALUDE_evaluate_complex_exponential_l3825_382584

theorem evaluate_complex_exponential : (3^2)^(3^(3^2)) = 9^19683 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_complex_exponential_l3825_382584


namespace NUMINAMATH_CALUDE_last_three_digits_of_5_to_15000_l3825_382585

theorem last_three_digits_of_5_to_15000 (h : 5^500 ≡ 1 [ZMOD 1000]) :
  5^15000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_5_to_15000_l3825_382585


namespace NUMINAMATH_CALUDE_count_denominators_repeating_decimal_l3825_382507

/-- The number of different possible denominators for the fraction representation of a repeating decimal 0.ab̅ in lowest terms, where a and b are digits. -/
theorem count_denominators_repeating_decimal : ∃ (n : ℕ), n = 6 ∧ n = (Finset.image (λ (p : ℕ × ℕ) => (Nat.lcm 99 (10 * p.1 + p.2) / (10 * p.1 + p.2)).gcd 99) (Finset.filter (λ (p : ℕ × ℕ) => p.1 < 10 ∧ p.2 < 10) (Finset.product (Finset.range 10) (Finset.range 10)))).card := by
  sorry

end NUMINAMATH_CALUDE_count_denominators_repeating_decimal_l3825_382507


namespace NUMINAMATH_CALUDE_triangle_theorem_l3825_382545

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c^2 = t.a^2 + t.b^2 - t.a * t.b ∧
  t.b = 2 ∧
  (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 3 ∧ t.a = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3825_382545


namespace NUMINAMATH_CALUDE_points_on_conic_l3825_382543

/-- A point in the Euclidean plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A conic section in the Euclidean plane -/
structure ConicSection where
  -- Define the conic section using its general equation: ax² + bxy + cy² + dx + ey + f = 0
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Check if a point lies on a conic section -/
def lies_on (p : Point) (c : ConicSection) : Prop :=
  c.a * p.x^2 + c.b * p.x * p.y + c.c * p.y^2 + c.d * p.x + c.e * p.y + c.f = 0

/-- Define a convex cyclic octagon -/
structure ConvexCyclicOctagon where
  vertices : Fin 8 → Point
  is_convex : Prop
  is_cyclic : Prop

/-- Define the intersection of two line segments -/
def intersection (p1 p2 p3 p4 : Point) : Point :=
  sorry -- Implementation of line intersection

/-- Define the B points as per the problem statement -/
def B (A : ConvexCyclicOctagon) (i : Fin 8) : Point :=
  intersection (A.vertices i) (A.vertices ((i + 3) % 8)) 
               (A.vertices ((i + 1) % 8)) (A.vertices ((i + 4) % 8))

/-- The main theorem -/
theorem points_on_conic (A : ConvexCyclicOctagon) : 
  ∃ (c : ConicSection), ∀ (i : Fin 8), lies_on (B A i) c := by
  sorry

end NUMINAMATH_CALUDE_points_on_conic_l3825_382543


namespace NUMINAMATH_CALUDE_distributor_cost_distributor_cost_proof_l3825_382553

/-- The cost of an item for a distributor given online store commission, desired profit, and observed price. -/
theorem distributor_cost (commission_rate : ℝ) (profit_rate : ℝ) (observed_price : ℝ) : ℝ :=
  let selling_price := observed_price / (1 - commission_rate)
  let cost := selling_price / (1 + profit_rate)
  cost

/-- Proof that the distributor's cost is $28.125 given the specified conditions. -/
theorem distributor_cost_proof :
  distributor_cost 0.2 0.2 27 = 28.125 :=
by sorry

end NUMINAMATH_CALUDE_distributor_cost_distributor_cost_proof_l3825_382553


namespace NUMINAMATH_CALUDE_initial_water_amount_l3825_382589

/-- Given a bucket of water, prove that the initial amount is 0.8 gallons when 0.2 gallons are poured out and 0.6 gallons remain. -/
theorem initial_water_amount (poured_out : ℝ) (remaining : ℝ) : poured_out = 0.2 → remaining = 0.6 → poured_out + remaining = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_amount_l3825_382589


namespace NUMINAMATH_CALUDE_seven_n_representable_l3825_382592

theorem seven_n_representable (n a b : ℤ) (h : n = a^2 + a*b + b^2) :
  ∃ x y : ℤ, 7*n = x^2 + x*y + y^2 := by sorry

end NUMINAMATH_CALUDE_seven_n_representable_l3825_382592


namespace NUMINAMATH_CALUDE_mary_lamb_count_l3825_382513

/-- The number of lambs Mary has after a series of events -/
def final_lamb_count (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) 
  (lambs_traded : ℕ) (extra_lambs_found : ℕ) : ℕ :=
  initial_lambs + lambs_with_babies * babies_per_lamb - lambs_traded + extra_lambs_found

/-- Theorem stating that Mary ends up with 14 lambs -/
theorem mary_lamb_count : 
  final_lamb_count 6 2 2 3 7 = 14 := by sorry

end NUMINAMATH_CALUDE_mary_lamb_count_l3825_382513


namespace NUMINAMATH_CALUDE_pentagonal_prism_sum_l3825_382577

/-- A pentagonal prism is a three-dimensional geometric shape with pentagonal bases and rectangular lateral faces. -/
structure PentagonalPrism where
  /-- The number of faces in a pentagonal prism -/
  faces : Nat
  /-- The number of edges in a pentagonal prism -/
  edges : Nat
  /-- The number of vertices in a pentagonal prism -/
  vertices : Nat
  /-- The faces of a pentagonal prism consist of 2 pentagonal bases and 5 rectangular lateral faces -/
  faces_def : faces = 7
  /-- The edges of a pentagonal prism consist of 10 edges from the two pentagons and 5 edges connecting them -/
  edges_def : edges = 15
  /-- The vertices of a pentagonal prism are the 5 vertices from each of the two pentagonal bases -/
  vertices_def : vertices = 10

/-- The sum of faces, edges, and vertices of a pentagonal prism is 32 -/
theorem pentagonal_prism_sum (p : PentagonalPrism) : p.faces + p.edges + p.vertices = 32 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_prism_sum_l3825_382577


namespace NUMINAMATH_CALUDE_files_remaining_l3825_382565

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 26)
  (h2 : video_files = 36)
  (h3 : deleted_files = 48) :
  music_files + video_files - deleted_files = 14 := by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l3825_382565


namespace NUMINAMATH_CALUDE_cubic_geometric_roots_property_l3825_382557

/-- A cubic equation with coefficients a, b, c has three nonzero real roots in geometric progression -/
structure CubicWithGeometricRoots (a b c : ℝ) : Prop where
  roots_exist : ∃ (d q : ℝ), d ≠ 0 ∧ q ≠ 0 ∧ q ≠ 1
  root_equation : ∀ (d q : ℝ), d ≠ 0 → q ≠ 0 → q ≠ 1 →
    d^3 + a*d^2 + b*d + c = 0 ∧
    (d*q)^3 + a*(d*q)^2 + b*(d*q) + c = 0 ∧
    (d*q^2)^3 + a*(d*q^2)^2 + b*(d*q^2) + c = 0

/-- The main theorem -/
theorem cubic_geometric_roots_property {a b c : ℝ} (h : CubicWithGeometricRoots a b c) :
  a^3 * c - b^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_geometric_roots_property_l3825_382557


namespace NUMINAMATH_CALUDE_sine_cosine_difference_equals_half_l3825_382563

theorem sine_cosine_difference_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) - 
  Real.cos (43 * π / 180) * Real.sin (13 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_difference_equals_half_l3825_382563


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3825_382599

theorem quadratic_roots_product (n : ℝ) (c d : ℝ) :
  c^2 - n*c + 4 = 0 →
  d^2 - n*d + 4 = 0 →
  ∃ (s : ℝ), (c + 1/d)^2 - s*(c + 1/d) + 25/4 = 0 ∧
             (d + 1/c)^2 - s*(d + 1/c) + 25/4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3825_382599


namespace NUMINAMATH_CALUDE_shirt_price_proof_l3825_382530

-- Define the original price
def original_price : ℝ := 32

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Define the final price
def final_price : ℝ := 18

-- Theorem statement
theorem shirt_price_proof :
  (1 - discount_rate) * (1 - discount_rate) * original_price = final_price :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_proof_l3825_382530


namespace NUMINAMATH_CALUDE_C_equals_46_l3825_382595

/-- Custom operation ⊕ -/
def circplus (a b : ℕ) : ℕ := a * b + 10

/-- Definition of C using the custom operation -/
def C : ℕ := circplus (circplus 1 2) 3

/-- Theorem stating that C equals 46 -/
theorem C_equals_46 : C = 46 := by
  sorry

end NUMINAMATH_CALUDE_C_equals_46_l3825_382595


namespace NUMINAMATH_CALUDE_square_park_fencing_cost_l3825_382579

/-- The total cost of fencing a square-shaped park -/
theorem square_park_fencing_cost (cost_per_side : ℕ) (h : cost_per_side = 72) : 
  cost_per_side * 4 = 288 := by
  sorry

#check square_park_fencing_cost

end NUMINAMATH_CALUDE_square_park_fencing_cost_l3825_382579


namespace NUMINAMATH_CALUDE_divisor_problem_l3825_382511

theorem divisor_problem (n d : ℕ) (h1 : n % d = 3) (h2 : (n^2) % d = 4) : d = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3825_382511


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l3825_382552

/-- Given two rectangles with equal area, where one rectangle measures 5 inches by 24 inches
    and the other is 8 inches long, prove that the width of the second rectangle is 15 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length jordan_width : ℝ) :
  carol_length = 5 →
  carol_width = 24 →
  jordan_length = 8 →
  carol_length * carol_width = jordan_length * jordan_width →
  jordan_width = 15 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l3825_382552


namespace NUMINAMATH_CALUDE_madeline_and_brother_total_l3825_382528

/-- Given Madeline has $48 and her brother has half as much, prove that they have $72 together. -/
theorem madeline_and_brother_total (madeline_amount : ℕ) (brother_amount : ℕ) : 
  madeline_amount = 48 → 
  brother_amount = madeline_amount / 2 → 
  madeline_amount + brother_amount = 72 := by
sorry

end NUMINAMATH_CALUDE_madeline_and_brother_total_l3825_382528


namespace NUMINAMATH_CALUDE_solution_set_inequality_proof_l3825_382555

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 5|

-- Part 1: Solution set of f(x) < 10
theorem solution_set (x : ℝ) : f x < 10 ↔ x ∈ Set.Ioo (-19/3) (-1) := by sorry

-- Part 2: Prove |a+b| + |a-b| < f(x) given |a| < 3 and |b| < 3
theorem inequality_proof (x a b : ℝ) (ha : |a| < 3) (hb : |b| < 3) :
  |a + b| + |a - b| < f x := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_proof_l3825_382555


namespace NUMINAMATH_CALUDE_cylinder_surface_area_minimized_l3825_382556

/-- Theorem: For a cylinder with fixed volume, the surface area is minimized when the height is twice the radius. -/
theorem cylinder_surface_area_minimized (R H V : ℝ) (h_positive : R > 0 ∧ H > 0 ∧ V > 0) 
  (h_volume : π * R^2 * H = V / 2) :
  let A := 2 * π * R^2 + 2 * π * R * H
  ∀ R' H' : ℝ, R' > 0 → H' > 0 → π * R'^2 * H' = V / 2 → 
    2 * π * R'^2 + 2 * π * R' * H' ≥ A → H / R = 2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_minimized_l3825_382556


namespace NUMINAMATH_CALUDE_external_tangent_intercept_l3825_382541

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculates the common external tangent line with positive slope for two circles -/
def commonExternalTangent (c1 c2 : Circle) : Line :=
  sorry

theorem external_tangent_intercept :
  let c1 : Circle := { center := (2, 4), radius := 5 }
  let c2 : Circle := { center := (14, 9), radius := 10 }
  let tangent := commonExternalTangent c1 c2
  tangent.slope > 0 → tangent.intercept = 912 / 119 :=
sorry

end NUMINAMATH_CALUDE_external_tangent_intercept_l3825_382541


namespace NUMINAMATH_CALUDE_min_distance_squared_l3825_382506

/-- The minimum squared distance from a point M(x,y,z) to N(1,1,1), 
    given specific conditions on x, y, and z -/
theorem min_distance_squared (x y z : ℝ) : 
  (∃ r : ℝ, y = x * r ∧ z = y * r) →  -- geometric progression condition
  (y * z = (x * y + x * z) / 2) →    -- arithmetic progression condition
  (z ≥ 1) →                          -- z ≥ 1 condition
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) →          -- distinctness condition
  18 ≤ (x - 1)^2 + (y - 1)^2 + (z - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_squared_l3825_382506


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3825_382551

theorem sum_of_three_numbers (a b c : ℤ) 
  (h1 : a + b = 31) 
  (h2 : b + c = 47) 
  (h3 : c + a = 54) : 
  a + b + c = 66 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3825_382551


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_14_l3825_382510

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ :=
  sorry

/-- A function that returns the nth positive integer whose digits sum to 14 -/
def nthNumberWithDigitSum14 (n : ℕ+) : ℕ+ :=
  sorry

/-- The theorem stating that the 11th number with digit sum 14 is 194 -/
theorem eleventh_number_with_digit_sum_14 : 
  nthNumberWithDigitSum14 11 = 194 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_14_l3825_382510


namespace NUMINAMATH_CALUDE_D_72_equals_81_l3825_382508

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where the order matters. -/
def D (n : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that D(72) = 81 -/
theorem D_72_equals_81 : D 72 = 81 := by
  sorry

end NUMINAMATH_CALUDE_D_72_equals_81_l3825_382508


namespace NUMINAMATH_CALUDE_fraction_multiplication_equals_decimal_l3825_382533

theorem fraction_multiplication_equals_decimal : 
  (1 : ℚ) / 3 * (3 : ℚ) / 7 * (7 : ℚ) / 8 = 0.12499999999999997 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_equals_decimal_l3825_382533


namespace NUMINAMATH_CALUDE_constant_for_max_n_l3825_382590

theorem constant_for_max_n (c : ℝ) : 
  (∀ n : ℤ, c * n^2 ≤ 3600) ∧ 
  (∃ n : ℤ, n > 5 ∧ c * n^2 > 3600) ∧
  c * 5^2 ≤ 3600 →
  c = 144 := by sorry

end NUMINAMATH_CALUDE_constant_for_max_n_l3825_382590


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l3825_382515

theorem max_value_x_plus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : 
  ∃ (max : ℝ), max = 3 ∧ ∀ (a b : ℝ), 3 * (a^2 + b^2) = a + b → a + 2*b ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l3825_382515


namespace NUMINAMATH_CALUDE_number_sum_proof_l3825_382549

theorem number_sum_proof (x : ℤ) : x + 14 = 68 → x + (x + 41) = 149 := by
  sorry

end NUMINAMATH_CALUDE_number_sum_proof_l3825_382549


namespace NUMINAMATH_CALUDE_problem_solution_l3825_382539

theorem problem_solution (x y : ℝ) 
  (hx : x = 2 + Real.sqrt 3) 
  (hy : y = 2 - Real.sqrt 3) : 
  (x^2 + 2*x*y + y^2 = 16) ∧ (x^2 - y^2 = 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3825_382539


namespace NUMINAMATH_CALUDE_work_completion_time_l3825_382598

/-- Given that:
  * A can finish a work in 10 days
  * When A and B work together, A's share of the work is 3/5
Prove that B can finish the work alone in 15 days -/
theorem work_completion_time (a_time : ℝ) (a_share : ℝ) (b_time : ℝ) : 
  a_time = 10 →
  a_share = 3/5 →
  b_time = (a_time * a_share) / (1 - a_share) →
  b_time = 15 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3825_382598


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_six_l3825_382593

theorem no_linear_term_implies_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (2*x + m) * (x - 3) = 2*x^2 - 3*m) → m = 6 := by
sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_six_l3825_382593


namespace NUMINAMATH_CALUDE_negative_two_and_negative_half_reciprocal_l3825_382525

/-- Two non-zero real numbers are reciprocal if their product is 1 -/
def IsReciprocal (a b : ℝ) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ a * b = 1

/-- -2 and -1/2 are reciprocal -/
theorem negative_two_and_negative_half_reciprocal : IsReciprocal (-2) (-1/2) := by
  sorry

end NUMINAMATH_CALUDE_negative_two_and_negative_half_reciprocal_l3825_382525


namespace NUMINAMATH_CALUDE_senior_ticket_price_l3825_382558

/-- Proves that the price of senior citizen tickets is $10 -/
theorem senior_ticket_price
  (total_tickets : ℕ)
  (regular_price : ℕ)
  (total_sales : ℕ)
  (regular_tickets : ℕ)
  (h1 : total_tickets = 65)
  (h2 : regular_price = 15)
  (h3 : total_sales = 855)
  (h4 : regular_tickets = 41)
  : (total_sales - regular_tickets * regular_price) / (total_tickets - regular_tickets) = 10 := by
  sorry

end NUMINAMATH_CALUDE_senior_ticket_price_l3825_382558


namespace NUMINAMATH_CALUDE_day_after_2_pow_20_l3825_382572

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the day of the week after a given number of days from Monday -/
def dayAfter (days : Nat) : DayOfWeek :=
  match (days % 7) with
  | 0 => DayOfWeek.Monday
  | 1 => DayOfWeek.Tuesday
  | 2 => DayOfWeek.Wednesday
  | 3 => DayOfWeek.Thursday
  | 4 => DayOfWeek.Friday
  | 5 => DayOfWeek.Saturday
  | _ => DayOfWeek.Sunday

/-- Theorem: After 2^20 days from Monday, it will be Friday -/
theorem day_after_2_pow_20 : dayAfter (2^20) = DayOfWeek.Friday := by
  sorry


end NUMINAMATH_CALUDE_day_after_2_pow_20_l3825_382572


namespace NUMINAMATH_CALUDE_special_function_value_l3825_382570

/-- A function satisfying f(xy) = f(x)/y for all positive real numbers x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y

theorem special_function_value 
  (f : ℝ → ℝ) 
  (h1 : special_function f) 
  (h2 : f 45 = 15) : 
  f 60 = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l3825_382570


namespace NUMINAMATH_CALUDE_gaming_system_value_proof_l3825_382566

/-- The value of Tom's gaming system -/
def gaming_system_value : ℝ := 150

/-- The percentage of the gaming system's value given as store credit -/
def store_credit_percentage : ℝ := 0.80

/-- The amount Tom pays in cash -/
def cash_paid : ℝ := 80

/-- The change Tom receives -/
def change_received : ℝ := 10

/-- The value of the game Tom receives -/
def game_value : ℝ := 30

/-- The cost of the NES -/
def nes_cost : ℝ := 160

theorem gaming_system_value_proof :
  store_credit_percentage * gaming_system_value + cash_paid - change_received = nes_cost + game_value :=
by sorry

end NUMINAMATH_CALUDE_gaming_system_value_proof_l3825_382566


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3825_382588

/-- A positive integer is a multiple of 3, 5, 7, and 9 if and only if it's a multiple of their LCM -/
axiom multiple_of_3_5_7_9 (n : ℕ) : (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (9 ∣ n) ↔ 315 ∣ n

/-- The theorem stating that 314 is the unique three-digit positive integer
    that is one less than a multiple of 3, 5, 7, and 9 -/
theorem unique_three_digit_number : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ ∃ m : ℕ, n + 1 = 315 * m :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3825_382588


namespace NUMINAMATH_CALUDE_sachins_age_l3825_382538

theorem sachins_age (sachin rahul : ℝ) 
  (h1 : rahul = sachin + 7)
  (h2 : sachin / rahul = 7 / 9) : 
  sachin = 24.5 := by
sorry

end NUMINAMATH_CALUDE_sachins_age_l3825_382538


namespace NUMINAMATH_CALUDE_banana_pancakes_count_l3825_382523

/-- The number of banana pancakes given the total, blueberry, and plain pancake counts. -/
def banana_pancakes (total blueberry plain : ℕ) : ℕ :=
  total - blueberry - plain

/-- Theorem stating that the number of banana pancakes is 24 given the specific counts. -/
theorem banana_pancakes_count :
  banana_pancakes 67 20 23 = 24 := by
  sorry

end NUMINAMATH_CALUDE_banana_pancakes_count_l3825_382523


namespace NUMINAMATH_CALUDE_largest_root_is_six_l3825_382532

/-- Polynomial P(x) = x^6 - 15x^5 + 74x^4 - 130x^3 + ax^2 + bx -/
def P (a b : ℝ) (x : ℝ) : ℝ := x^6 - 15*x^5 + 74*x^4 - 130*x^3 + a*x^2 + b*x

/-- Line L(x) = cx - 24 -/
def L (c : ℝ) (x : ℝ) : ℝ := c*x - 24

/-- The difference between P(x) and L(x) -/
def D (a b c : ℝ) (x : ℝ) : ℝ := P a b x - L c x

theorem largest_root_is_six (a b c : ℝ) : 
  (∃ p q : ℝ, (∀ x : ℝ, D a b c x = (x - p)^3 * (x - q)^2)) →
  (∀ x : ℝ, D a b c x ≥ 0) →
  (∃ x₁ x₂ x₃ : ℝ, D a b c x₁ = 0 ∧ D a b c x₂ = 0 ∧ D a b c x₃ = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
  (∃ x : ℝ, D a b c x = 0 ∧ x = 6 ∧ ∀ y : ℝ, D a b c y = 0 → y ≤ x) :=
sorry

end NUMINAMATH_CALUDE_largest_root_is_six_l3825_382532


namespace NUMINAMATH_CALUDE_composite_solid_surface_area_l3825_382514

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The volume of a rectangular solid is the product of its length, width, and height. -/
def volume (l w h : ℕ) : ℕ := l * w * h

/-- The surface area of a rectangular solid. -/
def surfaceAreaRectangular (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

/-- The surface area of a cube. -/
def surfaceAreaCube (s : ℕ) : ℕ := 6 * s * s

theorem composite_solid_surface_area 
  (l w h : ℕ) 
  (prime_l : isPrime l) 
  (prime_w : isPrime w) 
  (prime_h : isPrime h) 
  (vol : volume l w h = 1001) :
  surfaceAreaRectangular l w h + surfaceAreaCube 13 - 13 * 13 = 1467 := by
  sorry

end NUMINAMATH_CALUDE_composite_solid_surface_area_l3825_382514


namespace NUMINAMATH_CALUDE_solutions_sum_and_product_l3825_382582

theorem solutions_sum_and_product : ∃ (x₁ x₂ : ℝ),
  (x₁ - 6)^2 = 49 ∧
  (x₂ - 6)^2 = 49 ∧
  x₁ + x₂ = 12 ∧
  x₁ * x₂ = -13 :=
by sorry

end NUMINAMATH_CALUDE_solutions_sum_and_product_l3825_382582


namespace NUMINAMATH_CALUDE_shopkeeper_cloth_cost_price_l3825_382564

/-- Given a shopkeeper who sells cloth at a loss, calculate the cost price per metre. -/
theorem shopkeeper_cloth_cost_price
  (total_metres : ℕ)
  (selling_price : ℕ)
  (loss_per_metre : ℕ)
  (h1 : total_metres = 500)
  (h2 : selling_price = 18000)
  (h3 : loss_per_metre = 5) :
  (selling_price + total_metres * loss_per_metre) / total_metres = 41 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_cloth_cost_price_l3825_382564


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3825_382537

theorem arithmetic_expression_equality : 36 + (120 / 15) + (15 * 19) - 150 - (450 / 9) = 129 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3825_382537


namespace NUMINAMATH_CALUDE_intersection_digit_l3825_382578

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def third_digit (n : ℕ) : ℕ := (n / 100) % 10

def four_digit_power_of_2 (m : ℕ) : Prop :=
  ∃ k, is_four_digit (2^k) ∧ m = third_digit (2^k)

def four_digit_power_of_5 (n : ℕ) : Prop :=
  ∃ k, is_four_digit (5^k) ∧ n = third_digit (5^k)

theorem intersection_digit :
  ∃! d, four_digit_power_of_2 d ∧ four_digit_power_of_5 d :=
sorry

end NUMINAMATH_CALUDE_intersection_digit_l3825_382578


namespace NUMINAMATH_CALUDE_tetrahedron_volume_bound_l3825_382562

-- Define a tetrahedron type
structure Tetrahedron :=
  (edges : Fin 6 → ℝ)

-- Define the volume of a tetrahedron
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

-- Define the condition that at least 5 edges are not greater than 2
def at_least_five_short_edges (t : Tetrahedron) : Prop :=
  ∃ (long_edge : Fin 6), ∀ (e : Fin 6), e ≠ long_edge → t.edges e ≤ 2

-- Theorem statement
theorem tetrahedron_volume_bound (t : Tetrahedron) :
  at_least_five_short_edges t → volume t ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_bound_l3825_382562


namespace NUMINAMATH_CALUDE_complex_subtraction_multiplication_l3825_382591

theorem complex_subtraction_multiplication (i : ℂ) : 
  (7 - 3 * i) - 3 * (2 - 5 * i) = 1 + 12 * i := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_multiplication_l3825_382591


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l3825_382531

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- positive terms
  ∃ d : ℝ, a = b - d ∧ c = b + d →  -- arithmetic sequence
  a * b * c = 216 →  -- product condition
  b ≥ 6 ∧ (∀ x : ℝ, x > 0 ∧ (∃ y z : ℝ, y > 0 ∧ z > 0 ∧ 
    (∃ e : ℝ, y = x - e ∧ z = x + e) ∧ 
    y * x * z = 216) → x ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l3825_382531


namespace NUMINAMATH_CALUDE_cone_apex_angle_l3825_382535

theorem cone_apex_angle (R : ℝ) (h : R > 0) :
  let lateral_surface := π * R^2 / 2
  let base_circumference := π * R
  lateral_surface = base_circumference * R / 2 →
  let base_diameter := R
  let apex_angle := 2 * Real.arcsin (base_diameter / (2 * R))
  apex_angle = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_apex_angle_l3825_382535


namespace NUMINAMATH_CALUDE_bill_difference_is_18_l3825_382575

def restaurant_bills (mike_tip_percent joe_tip_percent anna_tip_percent : ℚ)
  (tip_amount : ℚ) : ℚ :=
  let mike_bill := tip_amount / mike_tip_percent
  let joe_bill := tip_amount / joe_tip_percent
  let anna_bill := tip_amount / anna_tip_percent
  max mike_bill (max joe_bill anna_bill) - min mike_bill (min joe_bill anna_bill)

theorem bill_difference_is_18 :
  restaurant_bills (15/100) (25/100) (10/100) 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_bill_difference_is_18_l3825_382575


namespace NUMINAMATH_CALUDE_jack_hand_in_amount_l3825_382534

def calculate_amount_to_hand_in (hundred_bills two_hundred_bills fifty_bills twenty_bills ten_bills five_bills one_bills quarters dimes nickels pennies : ℕ) (amount_to_leave : ℝ) : ℝ :=
  let total_notes := 100 * hundred_bills + 50 * fifty_bills + 20 * twenty_bills + 10 * ten_bills + 5 * five_bills + one_bills
  let amount_to_hand_in := total_notes - amount_to_leave
  amount_to_hand_in

theorem jack_hand_in_amount :
  calculate_amount_to_hand_in 2 1 5 3 7 27 42 19 36 47 300 = 142 := by
  sorry

end NUMINAMATH_CALUDE_jack_hand_in_amount_l3825_382534


namespace NUMINAMATH_CALUDE_expression_evaluation_l3825_382573

theorem expression_evaluation (x : ℝ) (h : x = -2) :
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = -1 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3825_382573


namespace NUMINAMATH_CALUDE_larger_number_problem_l3825_382536

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1345) (h3 : L = 6 * S + 15) : L = 1611 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3825_382536


namespace NUMINAMATH_CALUDE_percentage_relationship_l3825_382554

theorem percentage_relationship (x y : ℝ) (c : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x = 2.5 * y) (h2 : 2 * y = c / 100 * x) : c = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relationship_l3825_382554


namespace NUMINAMATH_CALUDE_sqrt_a_plus_one_real_iff_a_geq_neg_one_l3825_382542

theorem sqrt_a_plus_one_real_iff_a_geq_neg_one (a : ℝ) : 
  (∃ x : ℝ, x^2 = a + 1) ↔ a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_plus_one_real_iff_a_geq_neg_one_l3825_382542


namespace NUMINAMATH_CALUDE_rachels_journey_distance_l3825_382529

/-- The distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Rachel's journey distance -/
theorem rachels_journey_distance :
  let speed := 2 -- miles per hour
  let time := 5 -- hours
  distance speed time = 10 -- miles
:= by sorry

end NUMINAMATH_CALUDE_rachels_journey_distance_l3825_382529


namespace NUMINAMATH_CALUDE_sum_division_problem_l3825_382519

theorem sum_division_problem (share_ratio_a share_ratio_b share_ratio_c : ℕ) 
  (second_person_share : ℚ) (total_amount : ℚ) : 
  share_ratio_a = 100 → 
  share_ratio_b = 45 → 
  share_ratio_c = 30 → 
  second_person_share = 63 → 
  total_amount = (second_person_share / share_ratio_b) * (share_ratio_a + share_ratio_b + share_ratio_c) → 
  total_amount = 245 := by
sorry

end NUMINAMATH_CALUDE_sum_division_problem_l3825_382519


namespace NUMINAMATH_CALUDE_set_A_equals_one_two_l3825_382520

def A : Set ℕ := {x | x^2 - 3*x < 0 ∧ x > 0}

theorem set_A_equals_one_two : A = {1, 2} := by sorry

end NUMINAMATH_CALUDE_set_A_equals_one_two_l3825_382520


namespace NUMINAMATH_CALUDE_winnie_lollipops_l3825_382580

theorem winnie_lollipops (total_lollipops : ℕ) (num_friends : ℕ) (h1 : total_lollipops = 400) (h2 : num_friends = 13) :
  total_lollipops - (num_friends * (total_lollipops / num_friends)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipops_l3825_382580


namespace NUMINAMATH_CALUDE_prob_and_arrangements_correct_l3825_382546

/-- The number of class officers -/
def total_officers : Nat := 6

/-- The number of boys among the officers -/
def num_boys : Nat := 3

/-- The number of girls among the officers -/
def num_girls : Nat := 3

/-- The number of people selected for voluntary labor -/
def num_selected : Nat := 3

/-- The probability of selecting at least 2 girls out of 3 people from a group of 3 boys and 3 girls -/
def prob_at_least_two_girls : ℚ := 1/2

/-- The number of ways to arrange 6 people (3 boys and 3 girls) in a row, 
    where one boy must be at an end and two specific girls must be together -/
def num_arrangements : Nat := 96

theorem prob_and_arrangements_correct : 
  (total_officers = num_boys + num_girls) →
  (prob_at_least_two_girls = 1/2) ∧ 
  (num_arrangements = 96) := by sorry

end NUMINAMATH_CALUDE_prob_and_arrangements_correct_l3825_382546


namespace NUMINAMATH_CALUDE_sum_product_inequalities_l3825_382576

theorem sum_product_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a + b) * (1/a + 1/b) ≥ 4) ∧ ((a + b + c) * (1/a + 1/b + 1/c) ≥ 9) := by sorry

end NUMINAMATH_CALUDE_sum_product_inequalities_l3825_382576


namespace NUMINAMATH_CALUDE_vector_decomposition_l3825_382540

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![8, 9, 4]
def p : Fin 3 → ℝ := ![1, 0, 1]
def q : Fin 3 → ℝ := ![0, -2, 1]
def r : Fin 3 → ℝ := ![1, 3, 0]

/-- Theorem: Vector x can be decomposed as 7p - 3q + r -/
theorem vector_decomposition :
  x = fun i => 7 * p i - 3 * q i + r i :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l3825_382540


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l3825_382544

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l3825_382544


namespace NUMINAMATH_CALUDE_expression_evaluation_l3825_382527

theorem expression_evaluation : 
  let a : ℚ := 7
  let b : ℚ := 11
  let c : ℚ := 13
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3825_382527


namespace NUMINAMATH_CALUDE_root_sum_fourth_power_l3825_382569

theorem root_sum_fourth_power (a b c s : ℝ) : 
  (x^3 - 6*x^2 + 14*x - 6 = 0 → (x = a ∨ x = b ∨ x = c)) →
  s = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  s^4 - 12*s^2 - 24*s = 20 := by
sorry

end NUMINAMATH_CALUDE_root_sum_fourth_power_l3825_382569


namespace NUMINAMATH_CALUDE_orchid_count_l3825_382587

/-- The number of orchid bushes initially in the park -/
def initial_orchids : ℕ := 22

/-- The number of orchid bushes to be planted -/
def planted_orchids : ℕ := 13

/-- The final number of orchid bushes after planting -/
def final_orchids : ℕ := 35

/-- Theorem stating that the initial number of orchid bushes plus the planted ones equals the final number -/
theorem orchid_count : initial_orchids + planted_orchids = final_orchids := by
  sorry

end NUMINAMATH_CALUDE_orchid_count_l3825_382587


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l3825_382524

theorem triangle_max_perimeter (a b c : ℝ) : 
  1 ≤ a ∧ a ≤ 3 ∧ 3 ≤ b ∧ b ≤ 5 ∧ 5 ≤ c ∧ c ≤ 7 →
  ∃ (p : ℝ), p = 8 + Real.sqrt 34 ∧ 
  ∀ (a' b' c' : ℝ), 1 ≤ a' ∧ a' ≤ 3 ∧ 3 ≤ b' ∧ b' ≤ 5 ∧ 5 ≤ c' ∧ c' ≤ 7 →
  (a' + b' + c' ≤ p ∧ 
   ∃ (s : ℝ), s = (a' + b' + c') / 2 ∧ 
   a' * b' * c' / (4 * s * (s - a') * (s - b') * (s - c')).sqrt ≤ 
   3 * 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l3825_382524


namespace NUMINAMATH_CALUDE_angle_expression_proof_l3825_382581

theorem angle_expression_proof (α : Real) (h : Real.tan α = 2) :
  (Real.cos (α - π) - 2 * Real.cos (π / 2 + α)) / (Real.sin (α - 3 * π / 2) - Real.sin α) = -3 :=
by sorry

end NUMINAMATH_CALUDE_angle_expression_proof_l3825_382581


namespace NUMINAMATH_CALUDE_city_outgoing_roads_l3825_382574

/-- Represents a city with squares and roads -/
structure City where
  /-- Number of squares in the city -/
  squares : ℕ
  /-- Number of streets going out of the city -/
  outgoing_streets : ℕ
  /-- Number of avenues going out of the city -/
  outgoing_avenues : ℕ
  /-- Number of crescents going out of the city -/
  outgoing_crescents : ℕ
  /-- Total number of outgoing roads is 3 -/
  outgoing_total : outgoing_streets + outgoing_avenues + outgoing_crescents = 3

/-- Theorem: In a city where exactly three roads meet at every square (one street, one avenue, and one crescent),
    and three roads go outside of the city, there must be exactly one street, one avenue, and one crescent going out of the city -/
theorem city_outgoing_roads (c : City) : 
  c.outgoing_streets = 1 ∧ c.outgoing_avenues = 1 ∧ c.outgoing_crescents = 1 := by
  sorry

end NUMINAMATH_CALUDE_city_outgoing_roads_l3825_382574


namespace NUMINAMATH_CALUDE_walking_problem_l3825_382503

theorem walking_problem (distance : ℝ) (initial_meeting_time : ℝ) 
  (speed_ratio : ℝ) (h1 : distance = 100) (h2 : initial_meeting_time = 3) 
  (h3 : speed_ratio = 4) : 
  ∃ (speed_A speed_B : ℝ) (meeting_times : List ℝ),
    speed_A = 80 / 3 ∧ 
    speed_B = 20 / 3 ∧
    speed_A = speed_ratio * speed_B ∧
    initial_meeting_time * (speed_A + speed_B) = distance ∧
    meeting_times = [3, 5, 9, 15] ∧
    (∀ t ∈ meeting_times, 
      (t ≤ distance / speed_B) ∧ 
      (∃ n : ℕ, t * speed_B = 2 * n * distance - t * speed_A ∨ 
               t * speed_B = (2 * n + 1) * distance - (distance - t * speed_A))) :=
by sorry

end NUMINAMATH_CALUDE_walking_problem_l3825_382503


namespace NUMINAMATH_CALUDE_equation_solution_l3825_382560

theorem equation_solution :
  let f (x : ℝ) := x + 2 - 4 / (x - 3)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ 3 ∧ x₂ ≠ 3 ∧
    x₁ = (1 + Real.sqrt 41) / 2 ∧
    x₂ = (1 - Real.sqrt 41) / 2 ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ (x : ℝ), x ≠ 3 → f x = 0 → (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3825_382560


namespace NUMINAMATH_CALUDE_parallelepiped_surface_area_l3825_382526

/-- Represents a parallelepiped composed of white and black unit cubes -/
structure Parallelepiped where
  white_cubes : ℕ
  black_cubes : ℕ
  length : ℕ
  width : ℕ
  height : ℕ

/-- Conditions for the parallelepiped -/
def valid_parallelepiped (p : Parallelepiped) : Prop :=
  p.white_cubes > 0 ∧
  p.black_cubes = p.white_cubes * 53 / 52 ∧
  p.length > 1 ∧ p.width > 1 ∧ p.height > 1 ∧
  p.length * p.width * p.height = p.white_cubes + p.black_cubes

/-- Surface area of a parallelepiped -/
def surface_area (p : Parallelepiped) : ℕ :=
  2 * (p.length * p.width + p.width * p.height + p.height * p.length)

/-- Theorem stating the surface area of the parallelepiped is 142 -/
theorem parallelepiped_surface_area (p : Parallelepiped) 
  (h : valid_parallelepiped p) : surface_area p = 142 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_surface_area_l3825_382526


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3825_382568

theorem tangent_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 1) (h₂ : r₂ = 2) (h₃ : r₃ = 3) :
  ∃ r : ℝ, r > 0 ∧
  (r₁ + r)^2 + (r₂ + r)^2 = (r₃ - r)^2 + (r₁ + r₂)^2 ∧
  r = 6/7 := by
sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3825_382568


namespace NUMINAMATH_CALUDE_function_inequality_equivalence_l3825_382561

-- Define the functions f and g with domain and range ℝ
variable (f g : ℝ → ℝ)

-- State the theorem
theorem function_inequality_equivalence :
  (∀ x, f x > g x) ↔ (∀ x, x ∉ {x | f x ≤ g x}) :=
sorry

end NUMINAMATH_CALUDE_function_inequality_equivalence_l3825_382561


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3825_382548

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  a₁ ≠ 0 →
  q ≠ 1 →
  let S₂ := a₁ * (1 - q^2) / (1 - q)
  let S₃ := a₁ * (1 - q^3) / (1 - q)
  S₃ + 3 * S₂ = 0 →
  q = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3825_382548
