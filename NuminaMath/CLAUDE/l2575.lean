import Mathlib

namespace NUMINAMATH_CALUDE_min_groups_for_30_students_max_12_l2575_257546

/-- Given a total number of students and a maximum group size, 
    calculate the minimum number of equal-sized groups. -/
def min_groups (total_students : ℕ) (max_group_size : ℕ) : ℕ :=
  let divisors := (Finset.range total_students).filter (λ d => total_students % d = 0)
  let valid_divisors := divisors.filter (λ d => d ≤ max_group_size)
  total_students / valid_divisors.max' (by sorry)

/-- The theorem stating that for 30 students and a maximum group size of 12, 
    the minimum number of equal-sized groups is 3. -/
theorem min_groups_for_30_students_max_12 :
  min_groups 30 12 = 3 := by sorry

end NUMINAMATH_CALUDE_min_groups_for_30_students_max_12_l2575_257546


namespace NUMINAMATH_CALUDE_integer_solutions_exist_l2575_257513

theorem integer_solutions_exist : ∃ (k x : ℤ), (k - 5) * x + 6 = 1 - 5 * x ∧
  ((k = 1 ∧ x = -5) ∨ (k = -1 ∧ x = 5) ∨ (k = 5 ∧ x = -1) ∨ (k = -5 ∧ x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_exist_l2575_257513


namespace NUMINAMATH_CALUDE_incorrect_statement_is_E_l2575_257527

theorem incorrect_statement_is_E :
  -- Statement A
  (∀ (a b c : ℝ), c > 0 → (a > b ↔ a + c > b + c)) ∧
  (∀ (a b c : ℝ), c > 0 → (a > b ↔ a * c > b * c)) ∧
  (∀ (a b c : ℝ), c > 0 → (a > b ↔ a / c > b / c)) ∧
  -- Statement B
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b → (a + b) / 2 > Real.sqrt (a * b)) ∧
  -- Statement C
  (∀ (s : ℝ), s > 0 → ∃ (x : ℝ), x > 0 ∧ x < s ∧
    ∀ (y : ℝ), y > 0 → y < s → x * (s - x) ≥ y * (s - y)) ∧
  -- Statement D
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b →
    (a^2 + b^2) / 2 > ((a + b) / 2)^2) ∧
  -- Statement E (negation)
  (∃ (p : ℝ), p > 0 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = p ∧ x + y > 2 * Real.sqrt p) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_statement_is_E_l2575_257527


namespace NUMINAMATH_CALUDE_max_value_theorem_l2575_257519

theorem max_value_theorem (a b c : ℝ) (h : a * b * c + a + c - b = 0) :
  ∃ (max : ℝ), max = 5/4 ∧ 
  ∀ (x y z : ℝ), x * y * z + x + z - y = 0 →
  (1 / (1 + x^2) - 1 / (1 + y^2) + 1 / (1 + z^2)) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2575_257519


namespace NUMINAMATH_CALUDE_rod_length_theorem_l2575_257576

/-- The length of a rod in meters, given the number of pieces it can be cut into and the length of each piece in centimeters. -/
def rod_length_meters (num_pieces : ℕ) (piece_length_cm : ℕ) : ℚ :=
  (num_pieces * piece_length_cm : ℚ) / 100

/-- Theorem stating that a rod that can be cut into 50 pieces of 85 cm each is 42.5 meters long. -/
theorem rod_length_theorem : rod_length_meters 50 85 = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_theorem_l2575_257576


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_example_l2575_257571

def is_simplest_quadratic_radical (n : ℝ) : Prop :=
  ∃ (a : ℕ), n = a ∧ ¬∃ (b : ℕ), b * b = a ∧ b > 1

theorem simplest_quadratic_radical_example : 
  ∃ (x : ℝ), is_simplest_quadratic_radical (x + 3) ∧ x = 2 := by
  sorry

#check simplest_quadratic_radical_example

end NUMINAMATH_CALUDE_simplest_quadratic_radical_example_l2575_257571


namespace NUMINAMATH_CALUDE_power_tower_mod_2000_l2575_257507

theorem power_tower_mod_2000 : 2^(2^(2^2)) ≡ 536 [ZMOD 2000] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_2000_l2575_257507


namespace NUMINAMATH_CALUDE_sqrt_fraction_equivalence_l2575_257568

theorem sqrt_fraction_equivalence (x : ℝ) (h : x < 0) :
  Real.sqrt (x / (1 - (x - 4) / x)) = -x / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equivalence_l2575_257568


namespace NUMINAMATH_CALUDE_sin_beta_value_l2575_257539

theorem sin_beta_value (α β : Real) (h_acute_α : 0 < α ∧ α < Real.pi / 2)
  (h_acute_β : 0 < β ∧ β < Real.pi / 2) (h_cos_α : Real.cos α = 2 * Real.sqrt 5 / 5)
  (h_sin_diff : Real.sin (α - β) = -3/5) : Real.sin β = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_beta_value_l2575_257539


namespace NUMINAMATH_CALUDE_distribution_ways_l2575_257542

theorem distribution_ways (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  (k : ℕ) ^ n = 243 := by
  sorry

end NUMINAMATH_CALUDE_distribution_ways_l2575_257542


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2575_257572

/-- Given a point P(-2, 3) in a Cartesian coordinate system, 
    its symmetric point with respect to the origin has coordinates (2, -3). -/
theorem symmetric_point_wrt_origin : 
  let P : ℝ × ℝ := (-2, 3)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  symmetric_point P = (2, -3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2575_257572


namespace NUMINAMATH_CALUDE_cubic_equation_roots_progression_l2575_257545

/-- Represents a cubic equation x³ + ax² + bx + c = 0 -/
structure CubicEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- The roots of a cubic equation -/
structure CubicRoots (α : Type*) [Field α] where
  x₁ : α
  x₂ : α
  x₃ : α

/-- Checks if the roots form an arithmetic progression -/
def is_arithmetic_progression {α : Type*} [Field α] (roots : CubicRoots α) : Prop :=
  roots.x₁ - roots.x₂ = roots.x₂ - roots.x₃

/-- Checks if the roots form a geometric progression -/
def is_geometric_progression {α : Type*} [Field α] (roots : CubicRoots α) : Prop :=
  roots.x₂ / roots.x₁ = roots.x₃ / roots.x₂

/-- Checks if the roots form a harmonic sequence -/
def is_harmonic_sequence {α : Type*} [Field α] (roots : CubicRoots α) : Prop :=
  (roots.x₁ - roots.x₂) / (roots.x₂ - roots.x₃) = roots.x₁ / roots.x₃

theorem cubic_equation_roots_progression {α : Type*} [Field α] (eq : CubicEquation α) (roots : CubicRoots α) :
  (is_arithmetic_progression roots ↔ (2 * eq.a^3 + 27 * eq.c) / (9 * eq.a) = eq.b) ∧
  (is_geometric_progression roots ↔ eq.b = eq.a * (eq.c^(1/3))) ∧
  (is_harmonic_sequence roots ↔ eq.a = (2 * eq.b^3 + 27 * eq.c) / (9 * eq.b^2)) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_progression_l2575_257545


namespace NUMINAMATH_CALUDE_marble_percentage_theorem_l2575_257556

/-- Represents a set of marbles -/
structure MarbleSet where
  total : ℕ
  broken : ℕ

/-- The problem setup -/
def marbleProblem : Prop :=
  ∃ (set1 set2 : MarbleSet),
    set1.total = 50 ∧
    set2.total = 60 ∧
    set1.broken = (10 : ℕ) * set1.total / 100 ∧
    set1.broken + set2.broken = 17 ∧
    set2.broken * 100 / set2.total = 20

/-- The theorem to prove -/
theorem marble_percentage_theorem : marbleProblem := by
  sorry

#check marble_percentage_theorem

end NUMINAMATH_CALUDE_marble_percentage_theorem_l2575_257556


namespace NUMINAMATH_CALUDE_lindas_age_multiple_l2575_257532

/-- Given:
  - Linda's age (L) is 3 more than a certain multiple (M) of Jane's age (J)
  - In five years, the sum of their ages will be 28
  - Linda's current age is 13
Prove that the multiple M is equal to 2 -/
theorem lindas_age_multiple (J L M : ℕ) : 
  L = M * J + 3 →
  L = 13 →
  L + 5 + J + 5 = 28 →
  M = 2 := by
sorry

end NUMINAMATH_CALUDE_lindas_age_multiple_l2575_257532


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l2575_257574

/-- Given a triangle ABC with side lengths a, b, c and angle A, prove the length of side a and the area of the triangle. -/
theorem triangle_side_and_area 
  (b c : ℝ) 
  (A : ℝ) 
  (hb : b = 4) 
  (hc : c = 2) 
  (hA : Real.cos A = 1/4) :
  ∃ (a : ℝ), 
    a = 4 ∧ 
    (1/2 * b * c * Real.sin A : ℝ) = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_and_area_l2575_257574


namespace NUMINAMATH_CALUDE_tank_capacity_l2575_257599

/-- Proves that the capacity of a tank is 21600 litres given specific inlet and outlet conditions. -/
theorem tank_capacity : 
  ∀ (outlet_time inlet_rate extended_time : ℝ),
  outlet_time = 10 →
  inlet_rate = 16 * 60 →
  extended_time = outlet_time + 8 →
  ∃ (capacity : ℝ),
  capacity / outlet_time - inlet_rate = capacity / extended_time ∧
  capacity = 21600 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l2575_257599


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l2575_257535

/-- If the solution set of (x-a)(x-b) < 0 is (-1,2), then a+b = 1 -/
theorem solution_set_implies_sum (a b : ℝ) : 
  (∀ x, (x-a)*(x-b) < 0 ↔ -1 < x ∧ x < 2) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l2575_257535


namespace NUMINAMATH_CALUDE_range_of_f_l2575_257577

noncomputable def f (x : ℝ) : ℝ := (Real.arccos x)^4 + (Real.arcsin x)^4

theorem range_of_f :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1,
  ∃ y ∈ Set.Icc 0 (π^4/8),
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc 0 (π^4/8) := by
sorry

end NUMINAMATH_CALUDE_range_of_f_l2575_257577


namespace NUMINAMATH_CALUDE_binomial_18_6_l2575_257579

theorem binomial_18_6 : Nat.choose 18 6 = 18564 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_6_l2575_257579


namespace NUMINAMATH_CALUDE_quadratic_equation_negative_root_l2575_257587

theorem quadratic_equation_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ (0 < a ∧ a ≤ 1) ∨ a < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_negative_root_l2575_257587


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2575_257509

/-- 
Given an arithmetic sequence where:
- a is the first term
- d is the common difference
- The first term (a) is 5/6
- The seventeenth term (a + 16d) is 5/8

Prove that the ninth term (a + 8d) is 15/16
-/
theorem arithmetic_sequence_ninth_term 
  (a d : ℚ) 
  (h1 : a = 5/6) 
  (h2 : a + 16*d = 5/8) : 
  a + 8*d = 15/16 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2575_257509


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2575_257580

-- Define the edge lengths in inches
def edge_length_1 : ℚ := 9
def edge_length_2 : ℚ := 3 * 12

-- Define the volume ratio function
def volume_ratio (a b : ℚ) : ℚ := (a / b) ^ 3

-- Theorem statement
theorem cube_volume_ratio : volume_ratio edge_length_1 edge_length_2 = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2575_257580


namespace NUMINAMATH_CALUDE_square_with_specific_digits_l2575_257521

theorem square_with_specific_digits (S : ℕ) : 
  (∃ (E : ℕ), S^2 = 10 * (10 * (10^100 * E + 2*E) + 2) + 5) →
  (S^2 % 10 = 5 ∧ S = (10^101 + 5) / 3) :=
by sorry

end NUMINAMATH_CALUDE_square_with_specific_digits_l2575_257521


namespace NUMINAMATH_CALUDE_box_volume_l2575_257588

theorem box_volume (l w h : ℝ) (shortest_path : ℝ) : 
  l = 6 → w = 6 → shortest_path = 20 → 
  shortest_path^2 = (l + w + h)^2 + w^2 →
  l * w * h = 576 := by
sorry

end NUMINAMATH_CALUDE_box_volume_l2575_257588


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2575_257538

-- Define a function to get the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_first_six_primes_mod_seventh_prime :
  (nthPrime 1 + nthPrime 2 + nthPrime 3 + nthPrime 4 + nthPrime 5 + nthPrime 6) % (nthPrime 7) = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2575_257538


namespace NUMINAMATH_CALUDE_five_integers_average_l2575_257528

theorem five_integers_average (a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (a₁ + a₂) + (a₁ + a₃) + (a₁ + a₄) + (a₁ + a₅) + 
  (a₂ + a₃) + (a₂ + a₄) + (a₂ + a₅) + 
  (a₃ + a₄) + (a₃ + a₅) + 
  (a₄ + a₅) = 2020 →
  (a₁ + a₂ + a₃ + a₄ + a₅) / 5 = 101 := by
sorry

#eval (2020 : ℤ) / 4  -- To verify that 2020 / 4 = 505

end NUMINAMATH_CALUDE_five_integers_average_l2575_257528


namespace NUMINAMATH_CALUDE_factor_expression_l2575_257524

theorem factor_expression (x : ℝ) : x^2*(x+3) + 2*x*(x+3) + (x+3) = (x+1)^2*(x+3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2575_257524


namespace NUMINAMATH_CALUDE_palindrome_count_is_420_l2575_257567

/-- Represents the count of each digit available -/
def digit_counts : List (Nat × Nat) := [(2, 2), (3, 3), (5, 4)]

/-- The total number of digits available -/
def total_digits : Nat := (digit_counts.map Prod.snd).sum

/-- A function to calculate the number of 9-digit palindromes -/
def count_palindromes (counts : List (Nat × Nat)) : Nat :=
  sorry

theorem palindrome_count_is_420 :
  total_digits = 9 ∧ count_palindromes digit_counts = 420 :=
sorry

end NUMINAMATH_CALUDE_palindrome_count_is_420_l2575_257567


namespace NUMINAMATH_CALUDE_longest_tape_measure_l2575_257581

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 100) 
  (hb : b = 225) 
  (hc : c = 780) : 
  Nat.gcd a (Nat.gcd b c) = 5 := by
  sorry

end NUMINAMATH_CALUDE_longest_tape_measure_l2575_257581


namespace NUMINAMATH_CALUDE_system_solution_l2575_257563

/-- 
Given a system of equations x - y = a and xy = b, 
this theorem proves that the solutions are 
(x, y) = ((a + √(a² + 4b))/2, (-a + √(a² + 4b))/2) and 
(x, y) = ((a - √(a² + 4b))/2, (-a - √(a² + 4b))/2).
-/
theorem system_solution (a b : ℝ) :
  let x₁ := (a + Real.sqrt (a^2 + 4*b)) / 2
  let y₁ := (-a + Real.sqrt (a^2 + 4*b)) / 2
  let x₂ := (a - Real.sqrt (a^2 + 4*b)) / 2
  let y₂ := (-a - Real.sqrt (a^2 + 4*b)) / 2
  (x₁ - y₁ = a ∧ x₁ * y₁ = b) ∧ 
  (x₂ - y₂ = a ∧ x₂ * y₂ = b) := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l2575_257563


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_l2575_257560

/-- The stock price after two years of changes -/
theorem stock_price_after_two_years 
  (initial_price : ℝ) 
  (first_year_increase : ℝ) 
  (second_year_decrease : ℝ) 
  (h1 : initial_price = 120)
  (h2 : first_year_increase = 1.2)
  (h3 : second_year_decrease = 0.3) :
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease) = 184.8 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_after_two_years_l2575_257560


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2575_257533

/-- An isosceles triangle with side lengths a and b satisfying a certain equation has perimeter 10 -/
theorem isosceles_triangle_perimeter (a b : ℝ) : 
  a > 0 → b > 0 → -- Positive side lengths
  (∃ c : ℝ, c > 0 ∧ a + a + c = b + b) → -- Isosceles triangle condition
  2 * Real.sqrt (3 * a - 6) + 3 * Real.sqrt (2 - a) = b - 4 → -- Given equation
  a + a + b = 10 := by -- Perimeter is 10
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2575_257533


namespace NUMINAMATH_CALUDE_sin_cos_relation_l2575_257547

theorem sin_cos_relation (x : Real) (h : Real.sin x = 4 * Real.cos x) :
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_relation_l2575_257547


namespace NUMINAMATH_CALUDE_distance_to_focus_l2575_257551

/-- Given a parabola y^2 = 2x and a point P(m, 2) on the parabola,
    the distance from P to the focus of the parabola is 5/2 -/
theorem distance_to_focus (m : ℝ) (h : 2^2 = 2*m) : 
  let P : ℝ × ℝ := (m, 2)
  let F : ℝ × ℝ := (1/2, 0)
  Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_l2575_257551


namespace NUMINAMATH_CALUDE_consecutive_product_divisibility_l2575_257525

theorem consecutive_product_divisibility (k : ℤ) :
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 8 * m) →
  ¬ (∀ m : ℤ, n = 64 * m) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_divisibility_l2575_257525


namespace NUMINAMATH_CALUDE_parking_lot_revenue_l2575_257502

/-- Given a parking lot with the following properties:
  * Total spaces: 1000
  * Section 1: 320 spaces at $5 per hour
  * Section 2: 200 more spaces than Section 3 at $8 per hour
  * Section 3: Remaining spaces at $4 per hour
  Prove that Section 2 has 440 spaces and the total revenue for 5 hours is $30400 -/
theorem parking_lot_revenue 
  (total_spaces : Nat) 
  (section1_spaces : Nat) 
  (section2_price : Nat) 
  (section3_price : Nat) 
  (section1_price : Nat) 
  (hours : Nat) :
  total_spaces = 1000 →
  section1_spaces = 320 →
  section2_price = 8 →
  section3_price = 4 →
  section1_price = 5 →
  hours = 5 →
  ∃ (section2_spaces section3_spaces : Nat),
    section2_spaces = section3_spaces + 200 ∧
    section1_spaces + section2_spaces + section3_spaces = total_spaces ∧
    section2_spaces = 440 ∧
    section1_spaces * section1_price * hours + 
    section2_spaces * section2_price * hours + 
    section3_spaces * section3_price * hours = 30400 := by
  sorry


end NUMINAMATH_CALUDE_parking_lot_revenue_l2575_257502


namespace NUMINAMATH_CALUDE_elective_course_selection_l2575_257518

def category_A : ℕ := 3
def category_B : ℕ := 4
def total_courses : ℕ := 3

theorem elective_course_selection :
  (Nat.choose category_A 1 * Nat.choose category_B 2) +
  (Nat.choose category_A 2 * Nat.choose category_B 1) = 30 := by
  sorry

end NUMINAMATH_CALUDE_elective_course_selection_l2575_257518


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2575_257584

/-- Given a line L1 defined by 4x + 5y = 20, and a perpendicular line L2 with y-intercept -3,
    the x-intercept of L2 is 12/5 -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y => 4 * x + 5 * y = 20
  let m1 : ℝ := -4 / 5  -- slope of L1
  let m2 : ℝ := 5 / 4   -- slope of L2 (perpendicular to L1)
  let L2 : ℝ → ℝ → Prop := λ x y => y = m2 * x - 3  -- equation of L2
  ∃ x : ℝ, L2 x 0 ∧ x = 12 / 5
  :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2575_257584


namespace NUMINAMATH_CALUDE_largest_factorable_n_l2575_257523

/-- The largest value of n for which 5x^2 + nx + 110 can be factored with integer coefficients -/
def largest_n : ℕ := 551

/-- Predicate to check if a polynomial can be factored with integer coefficients -/
def can_be_factored (n : ℤ) : Prop :=
  ∃ (A B : ℤ), 5 * B + A = n ∧ A * B = 110

theorem largest_factorable_n :
  (∀ m : ℕ, m > largest_n → ¬(can_be_factored m)) ∧
  (can_be_factored largest_n) :=
sorry

end NUMINAMATH_CALUDE_largest_factorable_n_l2575_257523


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l2575_257573

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (heq : x * (x + y) = 5 * x + y) :
  ∃ (m : ℝ), m = 9 ∧ ∀ z, z = 2 * x + y → z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l2575_257573


namespace NUMINAMATH_CALUDE_journey_time_proof_l2575_257543

/-- Proves that given a round trip where the outbound journey is at 60 km/h, 
    the return journey is at 90 km/h, and the total time is 2 hours, 
    the time taken for the outbound journey is 72 minutes. -/
theorem journey_time_proof (distance : ℝ) 
    (h1 : distance / 60 + distance / 90 = 2) : 
    distance / 60 * 60 = 72 := by
  sorry

#check journey_time_proof

end NUMINAMATH_CALUDE_journey_time_proof_l2575_257543


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2575_257585

theorem smallest_integer_with_remainder_one : ∃ m : ℕ, 
  (m > 1) ∧ 
  (m % 5 = 1) ∧ 
  (m % 7 = 1) ∧ 
  (m % 3 = 1) ∧ 
  (∀ n : ℕ, n > 1 → n % 5 = 1 → n % 7 = 1 → n % 3 = 1 → m ≤ n) ∧
  (m = 106) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2575_257585


namespace NUMINAMATH_CALUDE_opposite_of_neg_two_is_two_l2575_257569

/-- The opposite number of a real number x is the number y such that x + y = 0 -/
def opposite (x : ℝ) : ℝ := -x

/-- Theorem: The opposite number of -2 is 2 -/
theorem opposite_of_neg_two_is_two : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_two_is_two_l2575_257569


namespace NUMINAMATH_CALUDE_nabla_computation_l2575_257517

-- Define the operation ∇
def nabla (x y : ℕ) : ℕ := x^3 - 2*y

-- State the theorem
theorem nabla_computation :
  (5^(nabla 7 4)) - 2*(2^(nabla 6 9)) = 5^1005 - 2^199 :=
by sorry

end NUMINAMATH_CALUDE_nabla_computation_l2575_257517


namespace NUMINAMATH_CALUDE_ratio_10_20_percent_l2575_257549

/-- The percent value of a ratio a:b is defined as (a/b) * 100 -/
def percent_value (a b : ℚ) : ℚ := (a / b) * 100

/-- The ratio 10:20 expressed as a percent is 50% -/
theorem ratio_10_20_percent : percent_value 10 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ratio_10_20_percent_l2575_257549


namespace NUMINAMATH_CALUDE_min_value_of_function_l2575_257552

theorem min_value_of_function (x : ℝ) (h : x > 1) : x + 2 / (x - 1) ≥ 2 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2575_257552


namespace NUMINAMATH_CALUDE_exists_sixth_root_of_3_30_sixth_root_of_3_30_correct_l2575_257561

theorem exists_sixth_root_of_3_30 : ∃ n : ℕ, n^6 = 3^30 :=
by
  -- The proof would go here
  sorry

def sixth_root_of_3_30 : ℕ :=
  -- The definition of the actual value would go here
  -- We're not providing the implementation as per the instructions
  sorry

-- This theorem ensures that our defined value actually satisfies the property
theorem sixth_root_of_3_30_correct : (sixth_root_of_3_30)^6 = 3^30 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_exists_sixth_root_of_3_30_sixth_root_of_3_30_correct_l2575_257561


namespace NUMINAMATH_CALUDE_total_cost_of_gifts_l2575_257596

def parents : ℕ := 2
def brothers : ℕ := 4
def sister : ℕ := 1
def brothers_spouses : ℕ := 4
def children_of_brothers : ℕ := 12
def sister_spouse : ℕ := 1
def children_of_sister : ℕ := 2
def grandparents : ℕ := 2
def cousins : ℕ := 3
def cost_per_package : ℕ := 7

theorem total_cost_of_gifts :
  parents + brothers + sister + brothers_spouses + children_of_brothers +
  sister_spouse + children_of_sister + grandparents + cousins = 31 →
  (parents + brothers + sister + brothers_spouses + children_of_brothers +
  sister_spouse + children_of_sister + grandparents + cousins) * cost_per_package = 217 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_of_gifts_l2575_257596


namespace NUMINAMATH_CALUDE_max_value_of_linear_combination_l2575_257500

theorem max_value_of_linear_combination (x y : ℝ) : 
  x^2 + y^2 = 16*x + 8*y + 8 → (∀ a b : ℝ, 4*x + 3*y ≤ 64) ∧ (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 = 16*x₀ + 8*y₀ + 8 ∧ 4*x₀ + 3*y₀ = 64) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_linear_combination_l2575_257500


namespace NUMINAMATH_CALUDE_trig_identity_l2575_257512

theorem trig_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 + 2 * Real.sin x * Real.sin y * Real.sin (x + y) =
  2 - Real.cos x ^ 2 - Real.cos (x + y) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2575_257512


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2575_257583

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n : ℕ, b (n + 1) > b n) →  -- increasing sequence
  (∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d) →  -- arithmetic sequence
  b 5 * b 6 = 35 →
  b 4 * b 7 = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2575_257583


namespace NUMINAMATH_CALUDE_cubic_function_through_point_l2575_257566

/-- Given a function f(x) = ax³ - 3x that passes through the point (-1, 4), prove that a = -1 --/
theorem cubic_function_through_point (a : ℝ) : 
  (fun x : ℝ => a * x^3 - 3*x) (-1) = 4 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_through_point_l2575_257566


namespace NUMINAMATH_CALUDE_ninth_term_is_512_l2575_257541

/-- Given a geometric sequence where:
  * The first term is 2
  * The common ratio is 2
  * n is the term number
  This function calculates the nth term of the sequence -/
def geometricSequenceTerm (n : ℕ) : ℕ := 2 * 2^(n - 1)

/-- Theorem stating that the 9th term of the geometric sequence is 512 -/
theorem ninth_term_is_512 : geometricSequenceTerm 9 = 512 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_512_l2575_257541


namespace NUMINAMATH_CALUDE_main_theorem_l2575_257503

noncomputable section

variable (e : ℝ)
variable (f : ℝ → ℝ)

-- Define the conditions
def non_negative (f : ℝ → ℝ) : Prop := ∀ x ∈ Set.Icc 0 e, f x ≥ 0
def f_e_equals_e : Prop := f e = e
def superadditive (f : ℝ → ℝ) : Prop := 
  ∀ x₁ x₂, x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ + x₂ ≤ e → f (x₁ + x₂) ≥ f x₁ + f x₂

-- Define the inequality condition
def inequality_condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 e, 4 * (f x)^2 - 4 * (2 * e - a) * f x + 4 * e^2 - 4 * e * a + 1 ≥ 0

-- Main theorem
theorem main_theorem (h1 : non_negative e f) (h2 : f_e_equals_e e f) (h3 : superadditive e f) :
  (f 0 = 0) ∧
  (∀ x ∈ Set.Icc 0 e, f x ≤ e) ∧
  (∀ a : ℝ, inequality_condition e f a → a ≤ e) := by
  sorry

end

end NUMINAMATH_CALUDE_main_theorem_l2575_257503


namespace NUMINAMATH_CALUDE_reinforcement_size_l2575_257514

/-- Calculates the size of reinforcement given initial garrison size, initial provision duration,
    days before reinforcement, and remaining provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_provisions : ℕ) 
                            (days_before_reinforcement : ℕ) (remaining_provisions : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_provisions
  let used_provisions := initial_garrison * days_before_reinforcement
  let remaining_total := total_provisions - used_provisions
  (remaining_total / remaining_provisions) - initial_garrison

theorem reinforcement_size :
  let initial_garrison : ℕ := 2000
  let initial_provisions : ℕ := 54
  let days_before_reinforcement : ℕ := 21
  let remaining_provisions : ℕ := 20
  calculate_reinforcement initial_garrison initial_provisions days_before_reinforcement remaining_provisions = 1300 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l2575_257514


namespace NUMINAMATH_CALUDE_town_friendship_theorem_l2575_257522

structure Town where
  inhabitants : Set Nat
  friendship : inhabitants → inhabitants → Prop
  enemy : inhabitants → inhabitants → Prop

def Town.canBecomeFriends (t : Town) : Prop :=
  ∃ (steps : ℕ), ∀ (a b : t.inhabitants), t.friendship a b

theorem town_friendship_theorem (t : Town) 
  (h1 : ∀ (a b : t.inhabitants), t.friendship a b ∨ t.enemy a b)
  (h2 : ∀ (a b c : t.inhabitants), t.friendship a b → t.friendship b c → t.friendship a c)
  (h3 : ∀ (a b c : t.inhabitants), t.friendship a b ∨ t.friendship a c ∨ t.friendship b c)
  (h4 : ∀ (day : ℕ), ∃ (a : t.inhabitants), 
    ∀ (b : t.inhabitants), 
      (t.friendship a b → t.enemy a b) ∧ 
      (t.enemy a b → t.friendship a b)) :
  t.canBecomeFriends :=
sorry

end NUMINAMATH_CALUDE_town_friendship_theorem_l2575_257522


namespace NUMINAMATH_CALUDE_cereal_eating_time_l2575_257559

/-- The time it takes for two people to eat a certain amount of cereal together -/
def eating_time (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- Mr. Fat's eating rate in pounds per minute -/
def mr_fat_rate : ℚ := 1 / 20

/-- Mr. Thin's eating rate in pounds per minute -/
def mr_thin_rate : ℚ := 1 / 30

/-- The amount of cereal to be eaten in pounds -/
def cereal_amount : ℚ := 3

theorem cereal_eating_time :
  eating_time mr_fat_rate mr_thin_rate cereal_amount = 36 := by
  sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l2575_257559


namespace NUMINAMATH_CALUDE_fourth_score_calculation_l2575_257534

theorem fourth_score_calculation (s1 s2 s3 s4 : ℕ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76)
  (h_average : (s1 + s2 + s3 + s4) / 4 = 75) : s4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_fourth_score_calculation_l2575_257534


namespace NUMINAMATH_CALUDE_rectangle_area_with_fixed_dimension_l2575_257511

theorem rectangle_area_with_fixed_dimension (l w : ℕ) : 
  (2 * l + 2 * w = 200) →  -- perimeter is 200 cm
  (w = 30 ∨ l = 30) →      -- one dimension is fixed at 30 cm
  (l * w = 2100)           -- area is 2100 square cm
:= by sorry

end NUMINAMATH_CALUDE_rectangle_area_with_fixed_dimension_l2575_257511


namespace NUMINAMATH_CALUDE_absolute_value_and_exponent_zero_sum_l2575_257548

theorem absolute_value_and_exponent_zero_sum : |-5| + (2 - Real.sqrt 3)^0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponent_zero_sum_l2575_257548


namespace NUMINAMATH_CALUDE_BF_length_is_150_l2575_257578

/-- Square ABCD with side length 500 and points E, F on AB satisfying given conditions -/
structure SquareEF where
  /-- Side length of the square -/
  side_length : ℝ
  /-- Length of EF -/
  EF_length : ℝ
  /-- Angle EOF in degrees -/
  angle_EOF : ℝ
  /-- E is between A and F -/
  E_between_A_F : Prop
  /-- AE is less than BF -/
  AE_less_than_BF : Prop
  /-- Side length is 500 -/
  side_length_eq : side_length = 500
  /-- EF length is 300 -/
  EF_length_eq : EF_length = 300
  /-- Angle EOF is 45 degrees -/
  angle_EOF_eq : angle_EOF = 45

/-- The length of BF in the given square configuration -/
def BF_length (s : SquareEF) : ℝ := sorry

/-- Theorem stating that BF length is 150 -/
theorem BF_length_is_150 (s : SquareEF) : BF_length s = 150 := sorry

end NUMINAMATH_CALUDE_BF_length_is_150_l2575_257578


namespace NUMINAMATH_CALUDE_total_movies_l2575_257520

-- Define the number of movies Timothy watched in 2009
def timothy_2009 : ℕ := 24

-- Define the number of movies Timothy watched in 2010
def timothy_2010 : ℕ := timothy_2009 + 7

-- Define the number of movies Theresa watched in 2009
def theresa_2009 : ℕ := timothy_2009 / 2

-- Define the number of movies Theresa watched in 2010
def theresa_2010 : ℕ := timothy_2010 * 2

-- Theorem to prove
theorem total_movies : timothy_2009 + timothy_2010 + theresa_2009 + theresa_2010 = 129 := by
  sorry

end NUMINAMATH_CALUDE_total_movies_l2575_257520


namespace NUMINAMATH_CALUDE_difference_of_max_min_F_l2575_257557

-- Define the function F
def F (x y : ℝ) : ℝ := 4 * x + y

-- State the theorem
theorem difference_of_max_min_F :
  ∀ x y : ℝ, x > 0 → y > 0 → 4 * x + 1 / x + y + 9 / y = 26 →
  (∃ (max min : ℝ), (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 * x' + 1 / x' + y' + 9 / y' = 26 → F x' y' ≤ max) ∧
                    (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 * x' + 1 / x' + y' + 9 / y' = 26 → F x' y' ≥ min) ∧
                    (max - min = 24)) :=
by sorry

end NUMINAMATH_CALUDE_difference_of_max_min_F_l2575_257557


namespace NUMINAMATH_CALUDE_glutenNutNonVegan_is_65_l2575_257582

/-- Represents the number of cupcakes with specific properties -/
structure Cupcakes where
  total : ℕ
  glutenFree : ℕ
  vegan : ℕ
  nutFree : ℕ
  glutenFreeVegan : ℕ
  veganNutFree : ℕ

/-- The properties of the cupcakes ordered for the birthday party -/
def birthdayCupcakes : Cupcakes where
  total := 120
  glutenFree := 120 / 3
  vegan := 120 / 4
  nutFree := 120 / 5
  glutenFreeVegan := 15
  veganNutFree := 10

/-- Calculates the number of cupcakes that are gluten, nut, and non-vegan -/
def glutenNutNonVegan (c : Cupcakes) : ℕ :=
  c.total - (c.glutenFree + (c.vegan - c.glutenFreeVegan))

/-- Theorem stating that the number of gluten, nut, and non-vegan cupcakes is 65 -/
theorem glutenNutNonVegan_is_65 : glutenNutNonVegan birthdayCupcakes = 65 := by
  sorry

end NUMINAMATH_CALUDE_glutenNutNonVegan_is_65_l2575_257582


namespace NUMINAMATH_CALUDE_arrangement_count_correct_l2575_257530

/-- The number of ways to arrange 5 volunteers and 2 elderly people in a row -/
def arrangementCount : ℕ :=
  let volunteerCount : ℕ := 5
  let elderlyCount : ℕ := 2
  let totalCount : ℕ := volunteerCount + elderlyCount
  let endPositions : ℕ := 2  -- number of end positions
  let intermediatePositions : ℕ := totalCount - endPositions - 1  -- -1 for elderly pair

  -- Choose volunteers for end positions
  let endArrangements : ℕ := volunteerCount * (volunteerCount - 1)
  
  -- Arrange remaining volunteers and elderly pair
  let intermediateArrangements : ℕ := Nat.factorial intermediatePositions
  
  -- Arrange elderly within their pair
  let elderlyArrangements : ℕ := Nat.factorial elderlyCount

  endArrangements * intermediateArrangements * elderlyArrangements

theorem arrangement_count_correct :
  arrangementCount = 960 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_correct_l2575_257530


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2575_257540

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (ha : a = 1657) (hb : b = 2037) (hr1 : r1 = 6) (hr2 : r2 = 5) :
  Nat.gcd (a - r1) (b - r2) = 127 :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2575_257540


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2575_257570

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, |x - 1| < 2 → -3 < x ∧ x < 3) ∧
  (∃ x : ℝ, -3 < x ∧ x < 3 ∧ ¬(|x - 1| < 2)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2575_257570


namespace NUMINAMATH_CALUDE_no_extra_savings_when_combined_l2575_257594

def book_price : ℕ := 120
def alice_books : ℕ := 10
def bob_books : ℕ := 15

def calculate_cost (num_books : ℕ) : ℕ :=
  let free_books := (num_books / 5) * 2
  let paid_books := num_books - free_books
  paid_books * book_price

def calculate_savings (num_books : ℕ) : ℕ :=
  num_books * book_price - calculate_cost num_books

theorem no_extra_savings_when_combined :
  calculate_savings alice_books + calculate_savings bob_books =
  calculate_savings (alice_books + bob_books) :=
by sorry

end NUMINAMATH_CALUDE_no_extra_savings_when_combined_l2575_257594


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2575_257508

-- Define the slopes of the lines
def m1 : ℚ := 3/4
def m2 : ℚ := -3/4
def m3 : ℚ := -3/4
def m4 : ℚ := -4/3

-- Define a function to check if two lines are perpendicular
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines :
  (are_perpendicular m1 m4) ∧
  (¬ are_perpendicular m1 m2) ∧
  (¬ are_perpendicular m1 m3) ∧
  (¬ are_perpendicular m2 m3) ∧
  (¬ are_perpendicular m2 m4) ∧
  (¬ are_perpendicular m3 m4) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2575_257508


namespace NUMINAMATH_CALUDE_machine_probabilities_theorem_l2575_257529

/-- Machine processing probabilities -/
structure MachineProbabilities where
  A : ℝ  -- Probability for machine A
  B : ℝ  -- Probability for machine B
  C : ℝ  -- Probability for machine C

/-- Given conditions -/
def conditions (p : MachineProbabilities) : Prop :=
  p.A * (1 - p.B) = 1/4 ∧
  p.B * (1 - p.C) = 1/12 ∧
  p.A * p.C = 2/9

/-- Theorem statement -/
theorem machine_probabilities_theorem (p : MachineProbabilities) 
  (h : conditions p) :
  p.A = 1/3 ∧ p.B = 1/4 ∧ p.C = 2/3 ∧
  1 - (1 - p.A) * (1 - p.B) * (1 - p.C) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_machine_probabilities_theorem_l2575_257529


namespace NUMINAMATH_CALUDE_smallest_value_w_cube_plus_z_cube_l2575_257504

theorem smallest_value_w_cube_plus_z_cube (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 8) :
  Complex.abs (w^3 + z^3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_w_cube_plus_z_cube_l2575_257504


namespace NUMINAMATH_CALUDE_ant_colony_problem_l2575_257550

theorem ant_colony_problem (x y : ℕ) :
  x + y = 40 →
  64 * x + 729 * y = 8748 →
  64 * x = 1984 :=
by
  sorry

end NUMINAMATH_CALUDE_ant_colony_problem_l2575_257550


namespace NUMINAMATH_CALUDE_creature_probability_l2575_257597

/-- Represents the type of creature on the island -/
inductive Creature
| Hare
| Rabbit

/-- The probability of a creature being mistaken -/
def mistakeProbability (c : Creature) : ℚ :=
  match c with
  | Creature.Hare => 1/4
  | Creature.Rabbit => 1/3

/-- The probability of a creature being correct -/
def correctProbability (c : Creature) : ℚ :=
  1 - mistakeProbability c

/-- The probability of a creature being of a certain type -/
def populationProbability (c : Creature) : ℚ := 1/2

theorem creature_probability (A B C : Prop) :
  let pA := populationProbability Creature.Hare
  let pNotA := populationProbability Creature.Rabbit
  let pBA := mistakeProbability Creature.Hare
  let pCA := correctProbability Creature.Hare
  let pBNotA := correctProbability Creature.Rabbit
  let pCNotA := mistakeProbability Creature.Rabbit
  let pABC := pA * pBA * pCA
  let pNotABC := pNotA * pBNotA * pCNotA
  let pBC := pABC + pNotABC
  pABC / pBC = 27/59 := by sorry

end NUMINAMATH_CALUDE_creature_probability_l2575_257597


namespace NUMINAMATH_CALUDE_product_digits_count_l2575_257562

theorem product_digits_count : ∃ n : ℕ, 
  (1002000000000000000 * 999999999999999999 : ℕ) ≥ 10^37 ∧ 
  (1002000000000000000 * 999999999999999999 : ℕ) < 10^38 :=
by sorry

end NUMINAMATH_CALUDE_product_digits_count_l2575_257562


namespace NUMINAMATH_CALUDE_max_subtract_add_result_l2575_257593

def S : Set Int := {-20, -10, 0, 5, 15, 25}

theorem max_subtract_add_result (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) :
  (a - b + c) ≤ 70 ∧ ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x - y + z = 70 := by
  sorry

end NUMINAMATH_CALUDE_max_subtract_add_result_l2575_257593


namespace NUMINAMATH_CALUDE_remainder_three_divisor_l2575_257589

theorem remainder_three_divisor (n : ℕ) (h : n = 1680) (h9 : n % 9 = 0) :
  ∃ m : ℕ, m = 1677 ∧ n % m = 3 :=
by sorry

end NUMINAMATH_CALUDE_remainder_three_divisor_l2575_257589


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l2575_257564

theorem cubic_polynomial_root (b c : ℚ) :
  (∃ (x : ℝ), x^3 + b*x + c = 0 ∧ x = 5 - 2*Real.sqrt 2) →
  (∃ (y : ℤ), y^3 + b*y + c = 0 ∧ y = -10) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l2575_257564


namespace NUMINAMATH_CALUDE_rectangle_squares_sides_l2575_257510

/-- Represents the side lengths of squares in a rectangle divided into 6 squares. -/
structure SquareSides where
  s1 : ℝ
  s2 : ℝ
  s3 : ℝ
  s4 : ℝ
  s5 : ℝ
  s6 : ℝ

/-- Given a rectangle divided into 6 squares with specific conditions,
    proves that the side lengths of the squares are as calculated. -/
theorem rectangle_squares_sides (sides : SquareSides) 
    (h1 : sides.s1 = 18)
    (h2 : sides.s2 = 3) :
    sides.s3 = 15 ∧ 
    sides.s4 = 12 ∧ 
    sides.s5 = 12 ∧ 
    sides.s6 = 21 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_squares_sides_l2575_257510


namespace NUMINAMATH_CALUDE_sin_squared_alpha_plus_pi_fourth_l2575_257544

theorem sin_squared_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/4)) 
  (h2 : Real.cos (2*α) = 4/5) : 
  Real.sin (α + π/4)^2 = 4/5 := by sorry

end NUMINAMATH_CALUDE_sin_squared_alpha_plus_pi_fourth_l2575_257544


namespace NUMINAMATH_CALUDE_line_slope_is_four_l2575_257565

/-- Given a line passing through points (0, 100) and (50, 300), prove that its slope is 4. -/
theorem line_slope_is_four :
  let x₁ : ℝ := 0
  let y₁ : ℝ := 100
  let x₂ : ℝ := 50
  let y₂ : ℝ := 300
  let slope : ℝ := (y₂ - y₁) / (x₂ - x₁)
  slope = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_is_four_l2575_257565


namespace NUMINAMATH_CALUDE_mandatory_service_effect_l2575_257537

/-- Represents the labor market for doctors --/
structure DoctorLaborMarket where
  state_supply : ℝ → ℝ  -- Supply function for state sector
  state_demand : ℝ → ℝ  -- Demand function for state sector
  commercial_supply : ℝ → ℝ  -- Supply function for commercial sector
  commercial_demand : ℝ → ℝ  -- Demand function for commercial sector

/-- Represents the policy of mandatory service for state-funded graduates --/
structure MandatoryServicePolicy where
  years_required : ℕ  -- Number of years of mandatory service

/-- Equilibrium wage in the state healthcare sector --/
def state_equilibrium_wage (market : DoctorLaborMarket) : ℝ :=
  sorry

/-- Equilibrium price for commercial medical services --/
def commercial_equilibrium_price (market : DoctorLaborMarket) : ℝ :=
  sorry

/-- The effect of the mandatory service policy on the doctor labor market --/
theorem mandatory_service_effect (initial_market : DoctorLaborMarket) 
    (policy : MandatoryServicePolicy) :
  ∃ (final_market : DoctorLaborMarket),
    state_equilibrium_wage final_market > state_equilibrium_wage initial_market ∧
    commercial_equilibrium_price final_market < commercial_equilibrium_price initial_market :=
  sorry

end NUMINAMATH_CALUDE_mandatory_service_effect_l2575_257537


namespace NUMINAMATH_CALUDE_cricket_average_increase_l2575_257590

/-- Proves that the increase in average runs per innings is 5 -/
theorem cricket_average_increase
  (initial_average : ℝ)
  (initial_innings : ℕ)
  (next_innings_runs : ℝ)
  (h1 : initial_average = 32)
  (h2 : initial_innings = 20)
  (h3 : next_innings_runs = 137) :
  let total_runs := initial_average * initial_innings
  let new_innings := initial_innings + 1
  let new_total_runs := total_runs + next_innings_runs
  let new_average := new_total_runs / new_innings
  new_average - initial_average = 5 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_increase_l2575_257590


namespace NUMINAMATH_CALUDE_girls_together_arrangements_girls_separate_arrangements_l2575_257586

/-- The number of boys in the lineup -/
def num_boys : ℕ := 4

/-- The number of girls in the lineup -/
def num_girls : ℕ := 3

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of ways to arrange the lineup with girls together -/
def arrangements_girls_together : ℕ := 720

/-- The number of ways to arrange the lineup with no two girls together -/
def arrangements_girls_separate : ℕ := 1440

/-- Theorem stating the number of arrangements with girls together -/
theorem girls_together_arrangements :
  (num_girls.factorial * (num_boys + 1).factorial) = arrangements_girls_together := by sorry

/-- Theorem stating the number of arrangements with no two girls together -/
theorem girls_separate_arrangements :
  (num_boys.factorial * (Nat.choose (num_boys + 1) num_girls) * num_girls.factorial) = arrangements_girls_separate := by sorry

end NUMINAMATH_CALUDE_girls_together_arrangements_girls_separate_arrangements_l2575_257586


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l2575_257558

theorem sum_of_roots_quadratic_equation (α β : ℝ) : 
  (α^2 - 4*α + 3 = 0) → (β^2 - 4*β + 3 = 0) → α + β = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l2575_257558


namespace NUMINAMATH_CALUDE_ball_height_properties_l2575_257526

/-- The height of a ball as a function of time -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- Theorem stating the maximum height and height at t = 1 -/
theorem ball_height_properties :
  (∀ t, h t ≤ 40) ∧ (h 1 = 40) := by
  sorry

end NUMINAMATH_CALUDE_ball_height_properties_l2575_257526


namespace NUMINAMATH_CALUDE_solution_set_range_of_m_l2575_257506

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for the solution set of |f(x) - 3| ≤ 4
theorem solution_set :
  {x : ℝ | |f x - 3| ≤ 4} = {x : ℝ | -6 ≤ x ∧ x ≤ 8} := by sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, f x + f (x + 3) ≥ m^2 - 2*m} = {m : ℝ | -1 ≤ m ∧ m ≤ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_range_of_m_l2575_257506


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l2575_257595

theorem rectangular_prism_width (l h d w : ℝ) : 
  l = 5 → h = 7 → d = 15 → d^2 = l^2 + w^2 + h^2 → w^2 = 151 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l2575_257595


namespace NUMINAMATH_CALUDE_subtracted_value_l2575_257515

theorem subtracted_value (n : ℕ) (x : ℕ) (h1 : n = 121) (h2 : 2 * n - x = 104) : x = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l2575_257515


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2575_257598

theorem negation_of_proposition (p : ℕ → Prop) : 
  (¬∀ n : ℕ, 3^n ≥ n + 1) ↔ (∃ n : ℕ, 3^n < n + 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2575_257598


namespace NUMINAMATH_CALUDE_max_value_inequality_l2575_257553

theorem max_value_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_condition : a + b + c + d ≤ 4) :
  (a * (b + 2 * c)) ^ (1/4) + (b * (c + 2 * d)) ^ (1/4) + 
  (c * (d + 2 * a)) ^ (1/4) + (d * (a + 2 * b)) ^ (1/4) ≤ 4 * 3 ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2575_257553


namespace NUMINAMATH_CALUDE_student_rank_from_left_l2575_257531

/-- Given a total number of students and a student's rank from the right,
    calculate the student's rank from the left. -/
def rankFromLeft (totalStudents : ℕ) (rankFromRight : ℕ) : ℕ :=
  totalStudents - rankFromRight + 1

/-- Theorem: Given 20 students in total and a student ranked 13th from the right,
    prove that the student's rank from the left is 8th. -/
theorem student_rank_from_left :
  let totalStudents : ℕ := 20
  let rankFromRight : ℕ := 13
  rankFromLeft totalStudents rankFromRight = 8 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_from_left_l2575_257531


namespace NUMINAMATH_CALUDE_sally_carl_owe_amount_l2575_257536

def total_promised : ℝ := 400
def amount_received : ℝ := 285
def amy_owes : ℝ := 30

theorem sally_carl_owe_amount :
  ∃ (s : ℝ), 
    s > 0 ∧
    2 * s + amy_owes + amy_owes / 2 = total_promised - amount_received ∧
    s = 35 := by sorry

end NUMINAMATH_CALUDE_sally_carl_owe_amount_l2575_257536


namespace NUMINAMATH_CALUDE_florist_bouquets_l2575_257592

/-- Calculates the number of bouquets that can be made given the initial number of seeds,
    the number of flowers killed by fungus, and the number of flowers per bouquet. -/
def calculateBouquets (seedsPerColor : ℕ) (redKilled yellowKilled orangeKilled purpleKilled : ℕ) (flowersPerBouquet : ℕ) : ℕ :=
  let redLeft := seedsPerColor - redKilled
  let yellowLeft := seedsPerColor - yellowKilled
  let orangeLeft := seedsPerColor - orangeKilled
  let purpleLeft := seedsPerColor - purpleKilled
  let totalFlowersLeft := redLeft + yellowLeft + orangeLeft + purpleLeft
  totalFlowersLeft / flowersPerBouquet

/-- Theorem stating that given the specific conditions of the problem,
    the florist can make 36 bouquets. -/
theorem florist_bouquets :
  calculateBouquets 125 45 61 30 40 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_florist_bouquets_l2575_257592


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2575_257554

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3*x⌋ - 1/2⌋ = ⌊x + 3⌋ ↔ 2 ≤ x ∧ x < 7/3 := by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2575_257554


namespace NUMINAMATH_CALUDE_zeros_in_99999_cubed_l2575_257505

-- Define a function to count zeros in a number
def count_zeros (n : ℕ) : ℕ := sorry

-- Define a function to count digits in a number
def count_digits (n : ℕ) : ℕ := sorry

-- Define the given conditions
axiom zeros_9 : count_zeros (9^3) = 0
axiom zeros_99 : count_zeros (99^3) = 2
axiom zeros_999 : count_zeros (999^3) = 3

-- Define the pattern continuation
axiom pattern_continuation (n : ℕ) : 
  n > 999 → count_zeros (n^3) = count_digits n

-- The theorem to prove
theorem zeros_in_99999_cubed : 
  count_zeros ((99999 : ℕ)^3) = count_digits 99999 := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_99999_cubed_l2575_257505


namespace NUMINAMATH_CALUDE_ticket_distribution_l2575_257555

/-- The number of ways to distribute identical objects among people --/
def distribution_methods (n : ℕ) (m : ℕ) : ℕ :=
  if n + 1 = m then m else 0

/-- Theorem: There are 5 ways to distribute 4 identical tickets among 5 people --/
theorem ticket_distribution : distribution_methods 4 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ticket_distribution_l2575_257555


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2575_257516

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 2) :
  (1 / a + 1 / b) ≥ 2 + Real.sqrt 3 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2575_257516


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2575_257591

-- Define the sets P and Q
def P : Set ℝ := {x | 1 < x ∧ x < 3}
def Q : Set ℝ := {x | x > 2}

-- Define the open interval (2, 3)
def open_interval_2_3 : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem intersection_P_Q : P ∩ Q = open_interval_2_3 := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2575_257591


namespace NUMINAMATH_CALUDE_difference_between_fractions_l2575_257501

theorem difference_between_fractions (n : ℝ) : n = 100 → (3/5 * n) - (1/2 * n) = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_fractions_l2575_257501


namespace NUMINAMATH_CALUDE_exam_student_count_l2575_257575

theorem exam_student_count (N : ℕ) (average_all : ℝ) (average_excluded : ℝ) (average_remaining : ℝ) 
  (h1 : average_all = 70)
  (h2 : average_excluded = 50)
  (h3 : average_remaining = 90)
  (h4 : N * average_all = 250 + (N - 5) * average_remaining) :
  N = 10 := by sorry

end NUMINAMATH_CALUDE_exam_student_count_l2575_257575
