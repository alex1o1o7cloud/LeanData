import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2208_220804

theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let asymptotes := fun (x y : ℝ) => y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  e = 2 → (∀ x y, hyperbola x y ↔ asymptotes x y) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2208_220804


namespace NUMINAMATH_CALUDE_maximal_regions_quadrilaterals_l2208_220811

/-- The maximal number of regions created by n convex quadrilaterals in a plane -/
def maxRegions (n : ℕ) : ℕ := 4*n^2 - 4*n + 2

/-- Theorem stating that maxRegions gives the maximal number of regions -/
theorem maximal_regions_quadrilaterals (n : ℕ) :
  ∀ (regions : ℕ), regions ≤ maxRegions n :=
by sorry

end NUMINAMATH_CALUDE_maximal_regions_quadrilaterals_l2208_220811


namespace NUMINAMATH_CALUDE_frog_arrangement_problem_l2208_220856

theorem frog_arrangement_problem :
  ∃! (N : ℕ), 
    N > 0 ∧
    N % 2 = 1 ∧
    N % 3 = 1 ∧
    N % 4 = 1 ∧
    N % 5 = 0 ∧
    N < 50 ∧
    N = 25 := by sorry

end NUMINAMATH_CALUDE_frog_arrangement_problem_l2208_220856


namespace NUMINAMATH_CALUDE_sum_geq_sqrt_product_of_sum_products_eq_27_l2208_220842

theorem sum_geq_sqrt_product_of_sum_products_eq_27
  (x y z : ℝ)
  (pos_x : 0 < x)
  (pos_y : 0 < y)
  (pos_z : 0 < z)
  (sum_products : x * y + y * z + z * x = 27) :
  x + y + z ≥ Real.sqrt (3 * x * y * z) ∧
  (x + y + z = Real.sqrt (3 * x * y * z) ↔ x = 3 ∧ y = 3 ∧ z = 3) :=
by sorry

end NUMINAMATH_CALUDE_sum_geq_sqrt_product_of_sum_products_eq_27_l2208_220842


namespace NUMINAMATH_CALUDE_bags_at_end_of_week_l2208_220869

/-- Calculates the total number of bags of cans at the end of the week given daily changes --/
def total_bags_at_end_of_week (
  monday : Real
  ) (tuesday : Real) (wednesday : Real) (thursday : Real) 
    (friday : Real) (saturday : Real) (sunday : Real) : Real :=
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

/-- Theorem stating the total number of bags at the end of the week --/
theorem bags_at_end_of_week : 
  total_bags_at_end_of_week 4 2.5 (-1.25) 0 3.75 (-1.5) 0 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_bags_at_end_of_week_l2208_220869


namespace NUMINAMATH_CALUDE_pages_read_relationship_l2208_220875

/-- Represents the number of pages read on each night --/
structure PagesRead where
  night1 : ℕ
  night2 : ℕ
  night3 : ℕ

/-- Theorem stating the relationship between pages read on night 3 and the other nights --/
theorem pages_read_relationship (p : PagesRead) (total : ℕ) : 
  p.night1 = 30 →
  p.night2 = 2 * p.night1 - 2 →
  total = p.night1 + p.night2 + p.night3 →
  total = 179 →
  p.night3 = total - (p.night1 + p.night2) := by
  sorry

end NUMINAMATH_CALUDE_pages_read_relationship_l2208_220875


namespace NUMINAMATH_CALUDE_right_triangle_sine_cosine_l2208_220885

theorem right_triangle_sine_cosine (D E F : ℝ) : 
  E = 90 → -- angle E is 90 degrees
  3 * Real.sin D = 4 * Real.cos D → -- given condition
  Real.sin D = 4/5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_sine_cosine_l2208_220885


namespace NUMINAMATH_CALUDE_standard_deviation_transformation_l2208_220851

-- Define a sample data type
def SampleData := Fin 10 → ℝ

-- Define standard deviation for a sample
noncomputable def standardDeviation (data : SampleData) : ℝ := sorry

-- Define the transformation function
def transform (x : ℝ) : ℝ := 2 * x - 1

-- Main theorem
theorem standard_deviation_transformation (data : SampleData) :
  standardDeviation data = 8 →
  standardDeviation (fun i => transform (data i)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_transformation_l2208_220851


namespace NUMINAMATH_CALUDE_amber_pieces_count_l2208_220817

theorem amber_pieces_count (green clear : ℕ) (h1 : green = 35) (h2 : clear = 85) 
  (h3 : green = (green + clear + amber) / 4) : amber = 20 := by
  sorry

end NUMINAMATH_CALUDE_amber_pieces_count_l2208_220817


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2208_220854

theorem smallest_n_satisfying_conditions : ∃ N : ℕ, 
  (∀ m : ℕ, m < N → ¬(3 ∣ m ∧ 11 ∣ m ∧ m % 12 = 6)) ∧ 
  (3 ∣ N ∧ 11 ∣ N ∧ N % 12 = 6) ∧
  N = 66 := by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2208_220854


namespace NUMINAMATH_CALUDE_complex_fourth_power_l2208_220863

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l2208_220863


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2208_220843

theorem tangent_line_equation (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.cos x
  let slope : ℝ := -a * Real.sin (π / 6)
  slope = 1 / 2 →
  let x₀ : ℝ := π / 6
  let y₀ : ℝ := f x₀
  ∀ x y : ℝ, (y - y₀ = slope * (x - x₀)) ↔ (x - 2 * y - Real.sqrt 3 - π / 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2208_220843


namespace NUMINAMATH_CALUDE_cubic_equation_natural_roots_l2208_220894

/-- The cubic equation has three natural number roots if and only if p = 76 -/
theorem cubic_equation_natural_roots (p : ℝ) : 
  (∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (5 * (x : ℝ)^3 - 5*(p+1)*(x : ℝ)^2 + (71*p-1)*(x : ℝ) + 1 = 66*p) ∧
    (5 * (y : ℝ)^3 - 5*(p+1)*(y : ℝ)^2 + (71*p-1)*(y : ℝ) + 1 = 66*p) ∧
    (5 * (z : ℝ)^3 - 5*(p+1)*(z : ℝ)^2 + (71*p-1)*(z : ℝ) + 1 = 66*p)) ↔
  p = 76 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_natural_roots_l2208_220894


namespace NUMINAMATH_CALUDE_gcd_sum_product_is_one_l2208_220883

/-- The sum of 1234 and 4321 -/
def sum_numbers : ℕ := 1234 + 4321

/-- The product of 1, 2, 3, and 4 -/
def product_digits : ℕ := 1 * 2 * 3 * 4

/-- Theorem stating that the greatest common divisor of the sum of 1234 and 4321,
    and the product of 1, 2, 3, and 4 is 1 -/
theorem gcd_sum_product_is_one : Nat.gcd sum_numbers product_digits = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_product_is_one_l2208_220883


namespace NUMINAMATH_CALUDE_binomial_distribution_properties_l2208_220827

/-- Represents the probability of success in a single trial -/
def p : ℝ := 0.6

/-- Represents the number of trials -/
def n : ℕ := 5

/-- Expected value of a binomial distribution -/
def expected_value : ℝ := n * p

/-- Variance of a binomial distribution -/
def variance : ℝ := n * p * (1 - p)

theorem binomial_distribution_properties :
  expected_value = 3 ∧ variance = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_properties_l2208_220827


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2208_220853

theorem polynomial_factorization (x : ℝ) :
  5 * (x + 3) * (x + 7) * (x + 9) * (x + 11) - 4 * x^2 =
  (5 * x^2 + 81 * x + 315) * (x + 3) * (x + 213) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2208_220853


namespace NUMINAMATH_CALUDE_means_inequality_l2208_220803

theorem means_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h_max : max b c ≥ (a + b) / 2) : 
  Real.sqrt ((b^2 + c^2) / 2) > (a + b) / 2 ∧ 
  (a + b) / 2 > Real.sqrt (a * b) ∧ 
  Real.sqrt (a * b) > 2 * a * b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_means_inequality_l2208_220803


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2208_220866

theorem complex_fraction_equality (a b : ℂ) 
  (h : (a + b) / (a - b) + (a - b) / (a + b) = 2) :
  (a^4 + b^4) / (a^4 - b^4) + (a^4 - b^4) / (a^4 + b^4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2208_220866


namespace NUMINAMATH_CALUDE_new_person_weight_l2208_220899

def initial_persons : ℕ := 6
def average_weight_increase : ℝ := 2
def replaced_person_weight : ℝ := 75

theorem new_person_weight :
  ∃ (new_weight : ℝ),
    new_weight = replaced_person_weight + initial_persons * average_weight_increase :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2208_220899


namespace NUMINAMATH_CALUDE_infinite_primes_with_solutions_l2208_220876

theorem infinite_primes_with_solutions : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ x y : ℤ, x^2 + x + 1 = p * y} := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_with_solutions_l2208_220876


namespace NUMINAMATH_CALUDE_nell_baseball_cards_l2208_220877

theorem nell_baseball_cards (initial_cards given_to_john given_to_jeff : ℕ) 
  (h1 : initial_cards = 573)
  (h2 : given_to_john = 195)
  (h3 : given_to_jeff = 168) :
  initial_cards - (given_to_john + given_to_jeff) = 210 := by
  sorry

end NUMINAMATH_CALUDE_nell_baseball_cards_l2208_220877


namespace NUMINAMATH_CALUDE_part_one_part_two_l2208_220872

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2 * x^2 - 3 * x * y + y^2 + 2 * x + 2 * y
def B (x y : ℝ) : ℝ := 4 * x^2 - 6 * x * y + 2 * y^2 - 3 * x - y

-- Part 1
theorem part_one : B 2 (-1/5) - 2 * A 2 (-1/5) = -13 := by sorry

-- Part 2
theorem part_two (a : ℝ) : 
  (∃ x y : ℝ, (|x - 2*a| + (y - 3)^2 = 0) ∧ (B x y - 2 * A x y = a)) → a = -1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2208_220872


namespace NUMINAMATH_CALUDE_divisor_remainders_l2208_220847

theorem divisor_remainders (n : ℕ) 
  (h : ∀ i ∈ Finset.range 1012, ∃ (d : ℕ), d ∣ n ∧ d % 2013 = 1001 + i) :
  ∀ k ∈ Finset.range 2012, ∃ (d : ℕ), d ∣ n^2 ∧ d % 2013 = k + 1 := by
sorry

end NUMINAMATH_CALUDE_divisor_remainders_l2208_220847


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2208_220838

/-- Given an arithmetic sequence {a_n} with a_2 = 7 and a_11 = a_9 + 6, prove a_1 = 4 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) → -- arithmetic sequence condition
  a 2 = 7 →
  a 11 = a 9 + 6 →
  a 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2208_220838


namespace NUMINAMATH_CALUDE_vector_simplification_l2208_220870

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_simplification (a b : V) : 
  2 • (a + b) - a = a + 2 • b := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l2208_220870


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_a_l2208_220825

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 1|

-- Theorem for the first part of the problem
theorem solution_set_f (x : ℝ) :
  f x ≤ 2 ↔ x ≤ -1 ∨ x ≥ -1/3 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_a (a : ℝ) :
  (∃ b : ℝ, ∀ x : ℝ, |a + b| - |a - b| ≥ f x) ↔ a ≥ 5/4 ∨ a ≤ -5/4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_a_l2208_220825


namespace NUMINAMATH_CALUDE_rope_ratio_proof_l2208_220805

theorem rope_ratio_proof (total_length longer_part shorter_part : ℕ) 
  (h1 : total_length = 40)
  (h2 : shorter_part = 16)
  (h3 : longer_part = total_length - shorter_part) :
  shorter_part * 3 = longer_part * 2 := by
  sorry

end NUMINAMATH_CALUDE_rope_ratio_proof_l2208_220805


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2208_220857

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∀ x y : ℝ, x + 2*y = r → (x^2 + y^2 = 2*r → (∀ ε > 0, ∃ x' y', x' + 2*y' = r ∧ (x'-x)^2 + (y'-y)^2 < ε^2 ∧ x'^2 + y'^2 ≠ 2*r))) → 
  r = 10 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2208_220857


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2208_220833

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- Definition of the foci -/
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-4, 0) ∧ F₂ = (4, 0)

/-- Theorem: Perimeter of triangle PF₁F₂ is 18 for any point P on the ellipse -/
theorem ellipse_triangle_perimeter 
  (x y : ℝ) 
  (F₁ F₂ : ℝ × ℝ) 
  (h_ellipse : is_on_ellipse x y) 
  (h_foci : foci F₁ F₂) : 
  let P := (x, y)
  ‖P - F₁‖ + ‖P - F₂‖ + ‖F₁ - F₂‖ = 18 :=
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2208_220833


namespace NUMINAMATH_CALUDE_angies_age_l2208_220830

theorem angies_age : ∃ (age : ℕ), 2 * age + 4 = 20 ∧ age = 8 := by sorry

end NUMINAMATH_CALUDE_angies_age_l2208_220830


namespace NUMINAMATH_CALUDE_original_profit_percentage_l2208_220882

theorem original_profit_percentage 
  (cost_price : ℝ) 
  (original_selling_price : ℝ) 
  (h1 : original_selling_price > 0) 
  (h2 : cost_price > 0) 
  (h3 : (2 * original_selling_price - cost_price) / cost_price = 2.6) : 
  (original_selling_price - cost_price) / cost_price = 0.8 := by
sorry

end NUMINAMATH_CALUDE_original_profit_percentage_l2208_220882


namespace NUMINAMATH_CALUDE_triangle_side_value_l2208_220840

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.b^2 - t.c^2 + 2*t.a = 0 ∧ Real.tan t.C / Real.tan t.B = 3

theorem triangle_side_value (t : Triangle) (h : TriangleConditions t) : t.a = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_value_l2208_220840


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2208_220893

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2208_220893


namespace NUMINAMATH_CALUDE_sofa_love_seat_cost_l2208_220887

/-- The cost of a love seat and sofa, where the sofa costs double the love seat -/
def total_cost (love_seat_cost : ℝ) : ℝ :=
  love_seat_cost + 2 * love_seat_cost

/-- Theorem stating that the total cost is $444 when the love seat costs $148 -/
theorem sofa_love_seat_cost : total_cost 148 = 444 := by
  sorry

end NUMINAMATH_CALUDE_sofa_love_seat_cost_l2208_220887


namespace NUMINAMATH_CALUDE_trapezoid_mn_length_l2208_220834

/-- Represents a trapezoid ABCD with point M on diagonal AC and point N on diagonal BD -/
structure Trapezoid where
  /-- Length of base AD -/
  ad : ℝ
  /-- Length of base BC -/
  bc : ℝ
  /-- Ratio of AM to MC on diagonal AC -/
  am_mc_ratio : ℝ × ℝ
  /-- Length of segment MN -/
  mn : ℝ

/-- Theorem stating the length of MN in the given trapezoid configuration -/
theorem trapezoid_mn_length (t : Trapezoid) :
  t.ad = 3 ∧ t.bc = 18 ∧ t.am_mc_ratio = (1, 2) → t.mn = 4 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_mn_length_l2208_220834


namespace NUMINAMATH_CALUDE_triangle_ABC_proof_l2208_220845

theorem triangle_ABC_proof (A B C : Real) (a b c : Real) :
  -- Conditions
  A + B + C = π →
  2 * Real.sin (B + C) ^ 2 - 3 * Real.cos A = 0 →
  B = π / 4 →
  a = 2 * Real.sqrt 3 →
  -- Conclusions
  A = π / 3 ∧ c = Real.sqrt 6 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_ABC_proof_l2208_220845


namespace NUMINAMATH_CALUDE_sqrt_six_greater_than_two_l2208_220812

theorem sqrt_six_greater_than_two : Real.sqrt 6 > 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_greater_than_two_l2208_220812


namespace NUMINAMATH_CALUDE_lateral_edge_length_l2208_220821

/-- A rectangular prism with 8 vertices and a given sum of lateral edge lengths -/
structure RectangularPrism :=
  (vertices : Nat)
  (lateral_edges_sum : ℝ)
  (is_valid : vertices = 8)

/-- The number of lateral edges in a rectangular prism -/
def lateral_edges_count : Nat := 4

/-- Theorem: In a valid rectangular prism, if the sum of lateral edges is 56,
    then each lateral edge has length 14 -/
theorem lateral_edge_length (prism : RectangularPrism)
    (h_sum : prism.lateral_edges_sum = 56) :
    prism.lateral_edges_sum / lateral_edges_count = 14 := by
  sorry

#check lateral_edge_length

end NUMINAMATH_CALUDE_lateral_edge_length_l2208_220821


namespace NUMINAMATH_CALUDE_smallest_n_for_Bn_radius_greater_than_two_l2208_220846

theorem smallest_n_for_Bn_radius_greater_than_two :
  (∃ n : ℕ+, (∀ k : ℕ+, k < n → Real.sqrt k - 1 ≤ 2) ∧ Real.sqrt n - 1 > 2) ∧
  (∀ n : ℕ+, (∀ k : ℕ+, k < n → Real.sqrt k - 1 ≤ 2) ∧ Real.sqrt n - 1 > 2 → n = 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_Bn_radius_greater_than_two_l2208_220846


namespace NUMINAMATH_CALUDE_basketball_prices_l2208_220829

theorem basketball_prices (price_A price_B : ℝ) : 
  price_A = 2 * price_B - 48 →
  9600 / price_A = 7200 / price_B →
  price_A = 96 ∧ price_B = 72 := by
sorry

end NUMINAMATH_CALUDE_basketball_prices_l2208_220829


namespace NUMINAMATH_CALUDE_quaternary_123_equals_27_l2208_220874

/-- Converts a quaternary (base-4) digit to its decimal value --/
def quaternary_to_decimal (digit : Nat) : Nat :=
  if digit < 4 then digit else 0

/-- Represents the quaternary number 123 --/
def quaternary_123 : List Nat := [1, 2, 3]

/-- Converts a list of quaternary digits to its decimal value --/
def quaternary_list_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + quaternary_to_decimal d * (4 ^ i)) 0

theorem quaternary_123_equals_27 :
  quaternary_list_to_decimal quaternary_123 = 27 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_123_equals_27_l2208_220874


namespace NUMINAMATH_CALUDE_orange_ribbons_count_l2208_220820

/-- The number of ribbons in a container with yellow, purple, orange, and black ribbons. -/
def total_ribbons : ℚ :=
  let black_ribbons : ℚ := 45
  let black_fraction : ℚ := 1 - (1/4 + 3/8 + 1/8)
  black_ribbons / black_fraction

/-- The number of orange ribbons in the container. -/
def orange_ribbons : ℚ := (1/8) * total_ribbons

/-- Theorem stating that the number of orange ribbons is 22.5. -/
theorem orange_ribbons_count : orange_ribbons = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_orange_ribbons_count_l2208_220820


namespace NUMINAMATH_CALUDE_opposite_of_three_abs_l2208_220895

theorem opposite_of_three_abs (x : ℝ) : x = -3 → |x + 2| = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_abs_l2208_220895


namespace NUMINAMATH_CALUDE_inverse_sum_simplification_l2208_220889

theorem inverse_sum_simplification (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ - y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (y * z - x * z + x * y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_simplification_l2208_220889


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l2208_220848

-- Define the logarithm function with base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- Define the equation
def equation (x : ℝ) : Prop :=
  2 * log5 x - 3 * log5 4 = 1

-- Theorem statement
theorem solution_satisfies_equation :
  equation (4 * Real.sqrt 5) ∧ equation (-4 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l2208_220848


namespace NUMINAMATH_CALUDE_football_games_indeterminate_l2208_220888

theorem football_games_indeterminate 
  (night_games : ℕ) 
  (keith_missed : ℕ) 
  (keith_attended : ℕ) 
  (h1 : night_games = 4) 
  (h2 : keith_missed = 4) 
  (h3 : keith_attended = 4) :
  ¬ ∃ (total_games : ℕ), 
    (total_games ≥ night_games) ∧ 
    (total_games = keith_missed + keith_attended) :=
by sorry

end NUMINAMATH_CALUDE_football_games_indeterminate_l2208_220888


namespace NUMINAMATH_CALUDE_transformation_is_rotation_after_dilation_l2208_220806

/-- The matrix representing a dilation with scale factor 4 followed by a 90-degree counterclockwise rotation -/
def transformationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -4; 4, 0]

/-- The dilation matrix with scale factor 4 -/
def dilationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![4, 0; 0, 4]

/-- The 90-degree counterclockwise rotation matrix -/
def rotationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -1; 1, 0]

/-- Theorem stating that the transformation matrix is equivalent to 
    the product of the rotation matrix and the dilation matrix -/
theorem transformation_is_rotation_after_dilation :
  transformationMatrix = rotationMatrix * dilationMatrix := by
  sorry

end NUMINAMATH_CALUDE_transformation_is_rotation_after_dilation_l2208_220806


namespace NUMINAMATH_CALUDE_max_integer_inequality_l2208_220892

theorem max_integer_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2*a + b = 1) :
  ∀ m : ℤ, (∀ a b, a > 0 → b > 0 → 2*a + b = 1 → 2/a + 1/b ≥ m) → m ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_integer_inequality_l2208_220892


namespace NUMINAMATH_CALUDE_cylinder_radius_ratio_l2208_220880

/-- Represents a right circular cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The cost of filling a cylinder with gasoline --/
def fillCost (c : Cylinder) (fullness : ℝ) : ℝ := sorry

/-- The problem statement --/
theorem cylinder_radius_ratio 
  (V B : Cylinder) 
  (h_height : V.height = B.height / 2)
  (h_cost_B : fillCost B 0.5 = 4)
  (h_cost_V : fillCost V 1 = 16) :
  V.radius / B.radius = 2 := by 
  sorry


end NUMINAMATH_CALUDE_cylinder_radius_ratio_l2208_220880


namespace NUMINAMATH_CALUDE_larger_number_proof_l2208_220837

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 4 * S + 15) : L = 1815 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2208_220837


namespace NUMINAMATH_CALUDE_odd_function_condition_l2208_220810

def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x, f (-x) a b = -f x a b) ↔ a^2 + b^2 = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_condition_l2208_220810


namespace NUMINAMATH_CALUDE_no_integer_solution_l2208_220858

theorem no_integer_solution : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2208_220858


namespace NUMINAMATH_CALUDE_bicycle_sales_cost_price_l2208_220864

theorem bicycle_sales_cost_price 
  (profit_A_to_B : Real) 
  (profit_B_to_C : Real) 
  (final_price : Real) :
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  final_price = 225 →
  ∃ (initial_cost : Real) (profit_C_to_D : Real),
    initial_cost = 150 ∧
    final_price = initial_cost * (1 + profit_A_to_B) * (1 + profit_B_to_C) * (1 + profit_C_to_D) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_sales_cost_price_l2208_220864


namespace NUMINAMATH_CALUDE_initial_student_count_l2208_220813

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (dropped_score : ℝ) :
  initial_avg = 61.5 →
  new_avg = 64.0 →
  dropped_score = 24 →
  ∃ n : ℕ, n * initial_avg = (n - 1) * new_avg + dropped_score ∧ n = 16 :=
by sorry

end NUMINAMATH_CALUDE_initial_student_count_l2208_220813


namespace NUMINAMATH_CALUDE_inequality_proof_l2208_220860

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^3 + b^3 + c^3 = 3) : 
  1/(a^4 + 3) + 1/(b^4 + 3) + 1/(c^4 + 3) ≥ 3/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2208_220860


namespace NUMINAMATH_CALUDE_probability_of_convex_quadrilateral_l2208_220819

-- Define the number of points on the circle
def num_points : ℕ := 8

-- Define the number of chords to be selected
def num_selected_chords : ℕ := 4

-- Define the total number of possible chords
def total_chords : ℕ := num_points.choose 2

-- Define the number of ways to select the chords
def ways_to_select_chords : ℕ := total_chords.choose num_selected_chords

-- Define the number of ways to form a convex quadrilateral
def convex_quadrilaterals : ℕ := num_points.choose 4

-- State the theorem
theorem probability_of_convex_quadrilateral :
  (convex_quadrilaterals : ℚ) / ways_to_select_chords = 2 / 585 :=
sorry

end NUMINAMATH_CALUDE_probability_of_convex_quadrilateral_l2208_220819


namespace NUMINAMATH_CALUDE_miller_rabin_composite_probability_l2208_220878

/-- Miller-Rabin primality test -/
def miller_rabin (n : ℕ) : Bool := sorry

/-- Probability of Miller-Rabin test correctly identifying a composite number -/
def prob_correct_composite (n : ℕ) : ℝ := sorry

theorem miller_rabin_composite_probability (n : ℕ) (h : ¬ Prime n) :
  prob_correct_composite n ≥ (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_miller_rabin_composite_probability_l2208_220878


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l2208_220879

/-- The product of two specific repeating decimals -/
theorem product_of_repeating_decimals :
  (63 : ℚ) / 99 * (54 : ℚ) / 99 = (14 : ℚ) / 41 := by
  sorry

#check product_of_repeating_decimals

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l2208_220879


namespace NUMINAMATH_CALUDE_sum_of_all_alternating_sums_l2208_220839

-- Define the set of numbers
def S : Finset ℕ := Finset.range 9

-- Define the alternating sum function
noncomputable def alternatingSum (subset : Finset ℕ) : ℤ :=
  sorry

-- Define the modified alternating sum that adds 9 again if present
noncomputable def modifiedAlternatingSum (subset : Finset ℕ) : ℤ :=
  if 9 ∈ subset then alternatingSum subset + 9 else alternatingSum subset

-- Theorem statement
theorem sum_of_all_alternating_sums : 
  (Finset.powerset S).sum modifiedAlternatingSum = 2304 :=
sorry

end NUMINAMATH_CALUDE_sum_of_all_alternating_sums_l2208_220839


namespace NUMINAMATH_CALUDE_linear_congruence_solution_l2208_220868

theorem linear_congruence_solution (x : Int) : 
  (7 * x + 3) % 17 = 2 % 17 ↔ x % 17 = 12 % 17 := by
  sorry

end NUMINAMATH_CALUDE_linear_congruence_solution_l2208_220868


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2208_220808

theorem absolute_value_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  x₁ > x₂ ∧ 
  (|x₁ - 3| = 15) ∧ 
  (|x₂ - 3| = 15) ∧ 
  (x₁ - x₂ = 30) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2208_220808


namespace NUMINAMATH_CALUDE_ratio_problem_l2208_220831

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 6) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2208_220831


namespace NUMINAMATH_CALUDE_sin_alpha_value_l2208_220881

theorem sin_alpha_value (α β : Real) 
  (eq1 : 1 - Real.cos α - Real.cos β + Real.sin α * Real.cos β = 0)
  (eq2 : 1 + Real.cos α - Real.sin β + Real.sin α * Real.cos β = 0) :
  Real.sin α = (1 - Real.sqrt 10) / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l2208_220881


namespace NUMINAMATH_CALUDE_complex_quadrant_l2208_220816

theorem complex_quadrant : ∃ (z : ℂ), z = (1 + Complex.I) * (1 - 2 * Complex.I) ∧ 
  (z.re > 0 ∧ z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l2208_220816


namespace NUMINAMATH_CALUDE_eight_power_twelve_sum_equals_two_power_y_l2208_220801

theorem eight_power_twelve_sum_equals_two_power_y (y : ℕ) : 
  (8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 = 2^y) → y = 39 := by
sorry

end NUMINAMATH_CALUDE_eight_power_twelve_sum_equals_two_power_y_l2208_220801


namespace NUMINAMATH_CALUDE_toms_climbing_time_l2208_220835

/-- Proves that Tom's climbing time is 2 hours given the conditions -/
theorem toms_climbing_time (elizabeth_time : ℕ) (tom_factor : ℕ) :
  elizabeth_time = 30 →
  tom_factor = 4 →
  (elizabeth_time * tom_factor : ℚ) / 60 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_toms_climbing_time_l2208_220835


namespace NUMINAMATH_CALUDE_largest_number_with_6_and_3_l2208_220873

def largest_two_digit_number (d1 d2 : Nat) : Nat :=
  max (10 * d1 + d2) (10 * d2 + d1)

theorem largest_number_with_6_and_3 :
  largest_two_digit_number 6 3 = 63 := by
sorry

end NUMINAMATH_CALUDE_largest_number_with_6_and_3_l2208_220873


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2208_220852

/-- Given a hyperbola with equation x²/16 - y²/25 = 1, 
    its asymptotes have the equation y = ±(5/4)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 / 16 - y^2 / 25 = 1 →
  ∃ (k : ℝ), k = 5/4 ∧ (y = k*x ∨ y = -k*x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2208_220852


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l2208_220898

theorem infinitely_many_solutions (a : ℚ) : 
  (∀ x : ℚ, 4 * (3 * x - 2 * a) = 3 * (4 * x + 18)) ↔ a = -27/4 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l2208_220898


namespace NUMINAMATH_CALUDE_remaining_students_l2208_220865

/-- The number of remaining students in the class -/
def n : ℕ := sorry

/-- The weight of the student who left the class -/
def weight_left : ℝ := 45

/-- The increase in average weight after the student left -/
def weight_increase : ℝ := 0.2

/-- The average weight of the remaining students -/
def avg_weight_remaining : ℝ := 57

/-- Theorem stating that the number of remaining students is 59 -/
theorem remaining_students : n = 59 := by
  sorry

end NUMINAMATH_CALUDE_remaining_students_l2208_220865


namespace NUMINAMATH_CALUDE_least_multiple_first_ten_gt_1000_l2208_220802

theorem least_multiple_first_ten_gt_1000 : ∃ n : ℕ,
  n > 1000 ∧
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧
  (∀ m : ℕ, m > 1000 ∧ (∀ k : ℕ, k ≤ 10 → k > 0 → m % k = 0) → m ≥ n) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_multiple_first_ten_gt_1000_l2208_220802


namespace NUMINAMATH_CALUDE_xiaoqiang_games_l2208_220862

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Jia : Player
| Yi : Player
| Bing : Player
| Ding : Player
| Xiaoqiang : Player

/-- The number of games played by each player -/
def games_played (p : Player) : ℕ :=
  match p with
  | Player.Jia => 4
  | Player.Yi => 3
  | Player.Bing => 2
  | Player.Ding => 1
  | Player.Xiaoqiang => 2  -- This is what we want to prove

/-- The total number of games in a round-robin tournament -/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem xiaoqiang_games :
  games_played Player.Xiaoqiang = 2 :=
by sorry

end NUMINAMATH_CALUDE_xiaoqiang_games_l2208_220862


namespace NUMINAMATH_CALUDE_arrange_digits_eq_sixteen_l2208_220823

/-- The number of ways to arrange the digits of 45,550 to form a 5-digit number, where numbers cannot begin with 0 -/
def arrange_digits : ℕ :=
  let digits : Multiset ℕ := {0, 4, 5, 5, 5}
  let non_zero_positions := 4  -- Number of valid positions for 0 (2nd to 5th)
  let remaining_digits := 4    -- Number of digits to arrange after placing 0
  let repeated_digit := 3      -- Number of 5's
  non_zero_positions * (remaining_digits.factorial / repeated_digit.factorial)

theorem arrange_digits_eq_sixteen : arrange_digits = 16 := by
  sorry

end NUMINAMATH_CALUDE_arrange_digits_eq_sixteen_l2208_220823


namespace NUMINAMATH_CALUDE_quadratic_root_cube_l2208_220809

theorem quadratic_root_cube (A B C : ℝ) (r s : ℝ) (h1 : A ≠ 0) :
  (A * r^2 + B * r + C = 0) →
  (A * s^2 + B * s + C = 0) →
  (r + s = -B / A) →
  (r * s = C / A) →
  let p := (B^3 - 3*A*B*C) / A^3
  ∃ q, (r^3)^2 + p*(r^3) + q = 0 ∧ (s^3)^2 + p*(s^3) + q = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_cube_l2208_220809


namespace NUMINAMATH_CALUDE_scientific_notation_378300_l2208_220815

/-- Proves that 378300 is equal to 3.783 × 10^5 in scientific notation -/
theorem scientific_notation_378300 :
  ∃ (a : ℝ) (n : ℤ), 378300 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.783 ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_378300_l2208_220815


namespace NUMINAMATH_CALUDE_problem_solution_l2208_220800

theorem problem_solution (x n : ℕ) (h1 : x = 9^n - 1) (h2 : Odd n) 
  (h3 : (Nat.factors x).length = 3) (h4 : 61 ∈ Nat.factors x) : x = 59048 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2208_220800


namespace NUMINAMATH_CALUDE_point_on_135_degree_angle_l2208_220896

/-- Given a point (√4, a) on the terminal side of the angle 135°, prove that a = 2 -/
theorem point_on_135_degree_angle (a : ℝ) : 
  (∃ (x y : ℝ), x = Real.sqrt 4 ∧ y = a ∧ 
   x = 2 * Real.cos (135 * π / 180) ∧ 
   y = 2 * Real.sin (135 * π / 180)) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_point_on_135_degree_angle_l2208_220896


namespace NUMINAMATH_CALUDE_excess_of_repeating_over_terminating_l2208_220828

/-- The value of the repeating decimal 0.727272... -/
def repeating_72 : ℚ := 72 / 99

/-- The value of the terminating decimal 0.72 -/
def terminating_72 : ℚ := 72 / 100

/-- The fraction by which 0.727272... exceeds 0.72 -/
def excess_fraction : ℚ := 800 / 1099989

theorem excess_of_repeating_over_terminating :
  repeating_72 - terminating_72 = excess_fraction := by
  sorry

end NUMINAMATH_CALUDE_excess_of_repeating_over_terminating_l2208_220828


namespace NUMINAMATH_CALUDE_sum_inequality_l2208_220855

theorem sum_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2208_220855


namespace NUMINAMATH_CALUDE_function_symmetry_implies_m_range_l2208_220867

theorem function_symmetry_implies_m_range 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h_f : ∀ x, f x = m * 4^x - 2^x) 
  (h_symmetry : ∃ x_0 : ℝ, x_0 ≠ 0 ∧ f (-x_0) = f x_0) : 
  0 < m ∧ m < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_implies_m_range_l2208_220867


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2208_220891

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (5 * y^12 + 4 * y^11 + 6 * y^9 + 3 * y^8) =
  15 * y^13 + 2 * y^12 - 8 * y^11 + 18 * y^10 - 3 * y^9 - 6 * y^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2208_220891


namespace NUMINAMATH_CALUDE_souvenir_spending_difference_l2208_220818

def total_spent : ℚ := 548
def keychain_bracelet_spent : ℚ := 347

theorem souvenir_spending_difference :
  keychain_bracelet_spent - (total_spent - keychain_bracelet_spent) = 146 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_spending_difference_l2208_220818


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2208_220826

theorem arithmetic_calculation : 8 / 4 - 3 - 10 + 3 * 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2208_220826


namespace NUMINAMATH_CALUDE_apple_bag_theorem_l2208_220897

/-- Represents the number of apples in a bag -/
inductive BagSize
  | small : BagSize  -- 6 apples
  | large : BagSize  -- 12 apples

/-- The total number of apples from all bags -/
def totalApples (bags : List BagSize) : Nat :=
  bags.foldl (fun sum bag => sum + match bag with
    | BagSize.small => 6
    | BagSize.large => 12) 0

/-- Theorem stating the possible total numbers of apples -/
theorem apple_bag_theorem (bags : List BagSize) :
  (totalApples bags ≥ 70 ∧ totalApples bags ≤ 80) →
  (totalApples bags = 72 ∨ totalApples bags = 78) := by
  sorry

end NUMINAMATH_CALUDE_apple_bag_theorem_l2208_220897


namespace NUMINAMATH_CALUDE_ryan_overall_percentage_l2208_220871

def total_problems : ℕ := 25 + 40 + 10

def correct_problems : ℕ := 
  (25 * 80 / 100) + (40 * 90 / 100) + (10 * 70 / 100)

theorem ryan_overall_percentage : 
  (correct_problems * 100) / total_problems = 84 := by sorry

end NUMINAMATH_CALUDE_ryan_overall_percentage_l2208_220871


namespace NUMINAMATH_CALUDE_parabola_translation_l2208_220824

/-- Given a parabola y = 2(x+1)^2 - 3, prove that translating it right by 1 unit and up by 3 units results in y = 2x^2 -/
theorem parabola_translation (x y : ℝ) :
  (y = 2 * (x + 1)^2 - 3) →
  (y + 3 = 2 * x^2) := by
sorry

end NUMINAMATH_CALUDE_parabola_translation_l2208_220824


namespace NUMINAMATH_CALUDE_received_a_implies_met_criteria_l2208_220849

/-- Represents the criteria for receiving an A on the exam -/
structure ExamCriteria where
  multiple_choice_correct : ℝ
  extra_credit_completed : Bool

/-- Represents a student's exam performance -/
structure ExamPerformance where
  multiple_choice_correct : ℝ
  extra_credit_completed : Bool
  received_a : Bool

/-- The criteria for receiving an A on the exam -/
def a_criteria : ExamCriteria :=
  { multiple_choice_correct := 90
  , extra_credit_completed := true }

/-- Predicate to check if a student's performance meets the criteria for an A -/
def meets_a_criteria (performance : ExamPerformance) (criteria : ExamCriteria) : Prop :=
  performance.multiple_choice_correct ≥ criteria.multiple_choice_correct ∧
  performance.extra_credit_completed = criteria.extra_credit_completed

/-- Theorem stating that if a student received an A, they must have met the criteria -/
theorem received_a_implies_met_criteria (student : ExamPerformance) :
  student.received_a → meets_a_criteria student a_criteria := by
  sorry

end NUMINAMATH_CALUDE_received_a_implies_met_criteria_l2208_220849


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l2208_220832

/-- Prove that (1, 1, 1) is the solution to the given system of equations -/
theorem solution_satisfies_system :
  let x₁ : ℝ := 1
  let x₂ : ℝ := 1
  let x₃ : ℝ := 1
  (x₁ + 2*x₂ + x₃ = 4) ∧
  (3*x₁ - 5*x₂ + 3*x₃ = 1) ∧
  (2*x₁ + 7*x₂ - x₃ = 8) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l2208_220832


namespace NUMINAMATH_CALUDE_alicia_ran_3350_miles_l2208_220807

/-- The number of steps before the counter resets -/
def max_steps : ℕ := 99999

/-- The number of times the counter reset throughout the year -/
def reset_count : ℕ := 50

/-- The number of steps shown on the counter on the last day of the year -/
def final_steps : ℕ := 25000

/-- The number of steps Alicia takes per mile -/
def steps_per_mile : ℕ := 1500

/-- The total number of steps Alicia took over the year -/
def total_steps : ℕ := (max_steps + 1) * reset_count + final_steps

/-- The approximate number of miles Alicia ran over the year -/
def miles_run : ℕ := total_steps / steps_per_mile

theorem alicia_ran_3350_miles : miles_run = 3350 := by
  sorry

end NUMINAMATH_CALUDE_alicia_ran_3350_miles_l2208_220807


namespace NUMINAMATH_CALUDE_initial_peanuts_count_l2208_220884

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Mary adds to the box -/
def peanuts_added : ℕ := 12

/-- The final number of peanuts in the box after Mary adds more -/
def final_peanuts : ℕ := 16

/-- Theorem stating that the initial number of peanuts is 4 -/
theorem initial_peanuts_count : initial_peanuts = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_peanuts_count_l2208_220884


namespace NUMINAMATH_CALUDE_grade_A_students_over_three_years_l2208_220836

theorem grade_A_students_over_three_years 
  (total : ℕ) 
  (first_year : ℕ) 
  (growth_rate : ℝ) 
  (h1 : total = 728)
  (h2 : first_year = 200)
  (h3 : first_year + first_year * (1 + growth_rate) + first_year * (1 + growth_rate)^2 = total) :
  first_year + first_year * (1 + growth_rate) + first_year * (1 + growth_rate)^2 = 728 := by
sorry

end NUMINAMATH_CALUDE_grade_A_students_over_three_years_l2208_220836


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2208_220822

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence satisfying
    certain conditions, the 9th term equals 0. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_third : a 3 = 6)
  (h_sum : a 1 + a 11 = 6) :
  a 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2208_220822


namespace NUMINAMATH_CALUDE_song_listeners_l2208_220850

theorem song_listeners (total group_size : ℕ) (book_readers : ℕ) (both_listeners : ℕ) : 
  group_size = 100 → book_readers = 50 → both_listeners = 20 → 
  ∃ song_listeners : ℕ, song_listeners = 70 ∧ 
    group_size = book_readers + song_listeners - both_listeners :=
by sorry

end NUMINAMATH_CALUDE_song_listeners_l2208_220850


namespace NUMINAMATH_CALUDE_roots_equation_value_l2208_220844

theorem roots_equation_value (α β : ℝ) : 
  α^2 - α - 1 = 0 → β^2 - β - 1 = 0 → α^4 + 3*β = 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_value_l2208_220844


namespace NUMINAMATH_CALUDE_zero_function_solution_l2208_220861

def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x - f y

theorem zero_function_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_function_solution_l2208_220861


namespace NUMINAMATH_CALUDE_smallest_fourth_lucky_number_l2208_220859

theorem smallest_fourth_lucky_number : 
  ∃ (n : ℕ), 
    n ≥ 10 ∧ n < 100 ∧
    (∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ m < n →
      ¬((57 + 13 + 72 + m) * 5 = 
        (5 + 7 + 1 + 3 + 7 + 2 + (m / 10) + (m % 10)) * 25)) ∧
    (57 + 13 + 72 + n) * 5 = 
      (5 + 7 + 1 + 3 + 7 + 2 + (n / 10) + (n % 10)) * 25 ∧
    n = 38 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fourth_lucky_number_l2208_220859


namespace NUMINAMATH_CALUDE_marble_distribution_l2208_220814

def valid_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (fun m => n % m = 0 ∧ m > 1 ∧ m < n ∧ n / m > 1)

theorem marble_distribution :
  (valid_divisors 420).card = 22 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l2208_220814


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l2208_220841

theorem unique_positive_integer_solution (m : ℤ) : 
  (∃! x : ℤ, x > 0 ∧ 6 * x^2 + 2 * (m - 13) * x + 12 - m = 0) ↔ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l2208_220841


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2208_220890

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 22)
  (sum_squares_eq : a^2 + b^2 + c^2 = 404)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 9346) :
  a * b * c = 446 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2208_220890


namespace NUMINAMATH_CALUDE_sum_of_first_40_digits_of_fraction_l2208_220886

-- Define the fraction
def fraction : ℚ := 1 / 1234

-- Define a function to get the nth digit after the decimal point
def nthDigitAfterDecimal (n : ℕ) : ℕ := sorry

-- Define the sum of the first 40 digits after the decimal point
def sumOfFirst40Digits : ℕ := (List.range 40).map nthDigitAfterDecimal |>.sum

-- Theorem statement
theorem sum_of_first_40_digits_of_fraction :
  sumOfFirst40Digits = 218 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_40_digits_of_fraction_l2208_220886
