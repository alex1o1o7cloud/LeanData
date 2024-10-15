import Mathlib

namespace NUMINAMATH_CALUDE_plans_equal_at_325_miles_unique_intersection_at_325_miles_l1135_113564

/-- Represents a car rental plan with an initial fee and a per-mile rate -/
structure RentalPlan where
  initialFee : ℝ
  perMileRate : ℝ

/-- The two rental plans available -/
def plan1 : RentalPlan := { initialFee := 65, perMileRate := 0.4 }
def plan2 : RentalPlan := { initialFee := 0, perMileRate := 0.6 }

/-- The cost of a rental plan for a given number of miles -/
def rentalCost (plan : RentalPlan) (miles : ℝ) : ℝ :=
  plan.initialFee + plan.perMileRate * miles

/-- The theorem stating that the two plans cost the same at 325 miles -/
theorem plans_equal_at_325_miles :
  rentalCost plan1 325 = rentalCost plan2 325 := by
  sorry

/-- The theorem stating that 325 is the unique point where the plans cost the same -/
theorem unique_intersection_at_325_miles :
  ∀ m : ℝ, rentalCost plan1 m = rentalCost plan2 m → m = 325 := by
  sorry

end NUMINAMATH_CALUDE_plans_equal_at_325_miles_unique_intersection_at_325_miles_l1135_113564


namespace NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l1135_113575

theorem polygon_sides_and_diagonals :
  ∀ n : ℕ,
  (n > 2) →
  (180 * (n - 2) = 3 * 360 - 180) →
  (n = 7 ∧ (n * (n - 3)) / 2 = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l1135_113575


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1135_113519

theorem quadratic_inequality_equivalence (x : ℝ) : 
  x^2 - 50*x + 625 ≤ 25 ↔ 20 ≤ x ∧ x ≤ 30 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1135_113519


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_four_proof_l1135_113511

/-- The probability of rolling two dice and getting a sum greater than four -/
def prob_sum_greater_than_four : ℚ := 5/6

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is less than or equal to four -/
def outcomes_sum_le_four : ℕ := 6

theorem prob_sum_greater_than_four_proof :
  prob_sum_greater_than_four = 1 - (outcomes_sum_le_four / total_outcomes) :=
by sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_four_proof_l1135_113511


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l1135_113538

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b^2 = a * c

theorem arithmetic_geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a 2)
  (h2 : geometric_sequence (a 1) (a 3) (a 4)) :
  a 2 = -6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l1135_113538


namespace NUMINAMATH_CALUDE_first_nonzero_digit_not_eventually_periodic_l1135_113504

/-- The first non-zero digit from the unit's place in the decimal representation of n! -/
def first_nonzero_digit (n : ℕ) : ℕ :=
  sorry

/-- The sequence of first non-zero digits is eventually periodic if there exists an N such that
    the sequence {a_n}_{n>N} is periodic -/
def eventually_periodic (a : ℕ → ℕ) : Prop :=
  ∃ N T : ℕ, T > 0 ∧ ∀ n > N, a (n + T) = a n

theorem first_nonzero_digit_not_eventually_periodic :
  ¬ eventually_periodic first_nonzero_digit :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_not_eventually_periodic_l1135_113504


namespace NUMINAMATH_CALUDE_ten_times_average_letters_l1135_113517

def elida_letters : ℕ := 5

def adrianna_letters : ℕ := 2 * elida_letters - 2

def average_letters : ℚ := (elida_letters + adrianna_letters) / 2

theorem ten_times_average_letters : 10 * average_letters = 65 := by
  sorry

end NUMINAMATH_CALUDE_ten_times_average_letters_l1135_113517


namespace NUMINAMATH_CALUDE_area_ratio_bound_l1135_113596

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  area : ℝ
  area_pos : area > 0

/-- The result of reflecting each vertex of a quadrilateral 
    with respect to the diagonal that does not contain it -/
def reflect_vertices (q : ConvexQuadrilateral) : ℝ := 
  sorry

theorem area_ratio_bound (q : ConvexQuadrilateral) : 
  reflect_vertices q / q.area < 3 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_bound_l1135_113596


namespace NUMINAMATH_CALUDE_largest_quantity_l1135_113500

theorem largest_quantity (a b c d e : ℝ) 
  (h : a - 1 = b + 2 ∧ a - 1 = c - 3 ∧ a - 1 = d + 4 ∧ a - 1 = e - 6) : 
  e = max a (max b (max c d)) :=
sorry

end NUMINAMATH_CALUDE_largest_quantity_l1135_113500


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1135_113505

/-- Given two quadratic equations and a relationship between their roots, 
    prove the condition for the equations to be identical -/
theorem quadratic_equation_equivalence 
  (p q r s : ℝ) 
  (x₁ x₂ y₁ y₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) →
  (x₂^2 + p*x₂ + q = 0) →
  (y₁^2 + r*y₁ + s = 0) →
  (y₂^2 + r*y₂ + s = 0) →
  (y₁ = x₁/(x₁-1)) →
  (y₂ = x₂/(x₂-1)) →
  (x₁ ≠ 1) →
  (x₂ ≠ 1) →
  (p = -r ∧ q = s) →
  (p + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1135_113505


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1135_113581

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1135_113581


namespace NUMINAMATH_CALUDE_tangent_line_to_quartic_curve_l1135_113562

/-- Given that y = 4x + b is a tangent line to y = x^4 - 1, prove that b = -4 -/
theorem tangent_line_to_quartic_curve (b : ℝ) : 
  (∃ x₀ : ℝ, (4 * x₀ + b = x₀^4 - 1) ∧ 
             (∀ x : ℝ, 4 * x + b ≥ x^4 - 1) ∧ 
             (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - x₀| ∧ |x - x₀| < δ → 4 * x + b > x^4 - 1)) → 
  b = -4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_quartic_curve_l1135_113562


namespace NUMINAMATH_CALUDE_parallel_line_slope_l1135_113530

/-- The slope of a line parallel to 3x - 6y = 12 is 1/2 -/
theorem parallel_line_slope (a b c : ℝ) (h : b ≠ 0) :
  (∃ k : ℝ, k ≠ 0 ∧ a = 3 * k ∧ b = -6 * k ∧ c = 12 * k) →
  (a / b : ℝ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l1135_113530


namespace NUMINAMATH_CALUDE_laptops_in_shop_l1135_113534

theorem laptops_in_shop (rows : ℕ) (laptops_per_row : ℕ) 
  (h1 : rows = 5) (h2 : laptops_per_row = 8) : 
  rows * laptops_per_row = 40 := by
  sorry

end NUMINAMATH_CALUDE_laptops_in_shop_l1135_113534


namespace NUMINAMATH_CALUDE_m_range_characterization_l1135_113561

def r (m : ℝ) (x : ℝ) : Prop := Real.sin x + Real.cos x > m

def s (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 > 0

theorem m_range_characterization (m : ℝ) :
  (∀ x : ℝ, (r m x ∧ ¬(s m x)) ∨ (¬(r m x) ∧ s m x)) ↔ 
  (m ≤ -2 ∨ (-Real.sqrt 2 ≤ m ∧ m < 2)) :=
sorry

end NUMINAMATH_CALUDE_m_range_characterization_l1135_113561


namespace NUMINAMATH_CALUDE_evaluate_expression_l1135_113566

theorem evaluate_expression : 
  45 * ((4 + 1/3) - (5 + 1/4)) / ((3 + 1/2) + (2 + 1/5)) = -(7 + 9/38) := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1135_113566


namespace NUMINAMATH_CALUDE_theater_seating_l1135_113541

/-- Represents the number of seats in a given row of the theater. -/
def seats (n : ℕ) : ℕ := 3 * n + 57

theorem theater_seating :
  (seats 6 = 75) ∧
  (seats 8 = 81) ∧
  (∀ n : ℕ, seats n = 3 * n + 57) ∧
  (seats 21 = 120) := by
  sorry

#check theater_seating

end NUMINAMATH_CALUDE_theater_seating_l1135_113541


namespace NUMINAMATH_CALUDE_no_solution_condition_l1135_113594

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ -4 → 1 / (x - 4) + m / (x + 4) ≠ (m + 3) / (x^2 - 16)) ↔ 
  (m = -1 ∨ m = 5 ∨ m = -1/3) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l1135_113594


namespace NUMINAMATH_CALUDE_inequality_problem_l1135_113579

theorem inequality_problem (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (h : a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + c^2*a^2)) :
  (a ≤ b + c ∧ b ≤ a + c ∧ c ≤ a + b) ∧
  (a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + c*a)) ∧
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x^2 + y^2 + z^2 ≤ 2*(x*y + y*z + z*x) ∧
    ¬(x^4 + y^4 + z^4 ≤ 2*(x^2*y^2 + y^2*z^2 + z^2*x^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l1135_113579


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1135_113532

theorem complex_fraction_equality (c d : ℂ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : c^2 - c*d + d^2 = 0) : 
  (c^12 + d^12) / (c + d)^12 = 2 / 81 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1135_113532


namespace NUMINAMATH_CALUDE_three_in_A_l1135_113514

def A : Set ℝ := {x | x ≤ Real.sqrt 13}

theorem three_in_A : (3 : ℝ) ∈ A := by sorry

end NUMINAMATH_CALUDE_three_in_A_l1135_113514


namespace NUMINAMATH_CALUDE_factorization_sum_l1135_113552

theorem factorization_sum (a b : ℤ) : 
  (∀ x : ℝ, 16 * x^2 - 106 * x - 105 = (8 * x + a) * (2 * x + b)) → 
  a + 2 * b = -23 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l1135_113552


namespace NUMINAMATH_CALUDE_twenty_solutions_implies_twenty_or_twentythree_l1135_113588

/-- Given a positive integer n, count_solutions n returns the number of solutions
    to the equation 3x + 3y + 2z = n in positive integers x, y, and z -/
def count_solutions (n : ℕ+) : ℕ :=
  sorry

theorem twenty_solutions_implies_twenty_or_twentythree (n : ℕ+) :
  count_solutions n = 20 → n = 20 ∨ n = 23 := by
  sorry

end NUMINAMATH_CALUDE_twenty_solutions_implies_twenty_or_twentythree_l1135_113588


namespace NUMINAMATH_CALUDE_problem_solution_l1135_113548

theorem problem_solution :
  -- Part 1
  (let a : ℤ := 2
   let b : ℤ := -1
   (3 * a^2 * b + (1/4) * a * b^2 - (3/4) * a * b^2 + a^2 * b) = -17) ∧
  -- Part 2
  (∀ x y : ℝ, ∃ a b : ℝ,
    (2*x^2 + a*x - y + 6) - (2*b*x^2 - 3*x + 5*y - 1) = 0 →
    5*a*b^2 - (a^2*b + 2*(a^2*b - 3*a*b^2)) = -60) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1135_113548


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1135_113545

theorem inequality_solution_set : 
  {x : ℝ | x + 2 < 1} = {x : ℝ | x < -1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1135_113545


namespace NUMINAMATH_CALUDE_sets_equality_implies_coefficients_l1135_113591

def A : Set ℝ := {-1, 3}

def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

theorem sets_equality_implies_coefficients (a b : ℝ) : 
  A = B a b → a = -2 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_sets_equality_implies_coefficients_l1135_113591


namespace NUMINAMATH_CALUDE_h_solutions_l1135_113585

noncomputable def h (x : ℝ) : ℝ :=
  if x < 2 then 4 * x + 10 else 3 * x - 12

theorem h_solutions :
  ∀ x : ℝ, h x = 6 ↔ x = -1 ∨ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_h_solutions_l1135_113585


namespace NUMINAMATH_CALUDE_triangle_properties_l1135_113590

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B = (3 * t.c - t.b) * Real.cos t.A ∧
  t.a = 2 * Real.sqrt 2 ∧
  (1 / 2) * t.b * t.c * Real.sin t.A = Real.sqrt 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.A = (2 * Real.sqrt 2) / 3 ∧ t.b + t.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1135_113590


namespace NUMINAMATH_CALUDE_smallest_other_integer_l1135_113597

theorem smallest_other_integer (m n x : ℕ) : 
  m = 30 →
  x > 0 →
  Nat.gcd m n = x + 1 →
  Nat.lcm m n = x * (x + 1) →
  ∃ (n_min : ℕ), n_min = 6 ∧ ∀ (n' : ℕ), (
    Nat.gcd m n' = x + 1 ∧
    Nat.lcm m n' = x * (x + 1) →
    n' ≥ n_min
  ) := by sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l1135_113597


namespace NUMINAMATH_CALUDE_average_difference_l1135_113565

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 170) : 
  a - c = -120 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l1135_113565


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1135_113520

theorem isosceles_triangle_perimeter : ∀ (a b : ℝ),
  a^2 - 9*a + 18 = 0 →
  b^2 - 9*b + 18 = 0 →
  a ≠ b →
  (∃ (leg base : ℝ), (leg = max a b ∧ base = min a b) ∧
    2*leg + base = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1135_113520


namespace NUMINAMATH_CALUDE_smallest_with_70_divisors_l1135_113529

/-- The number of natural divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- A natural number has exactly 70 divisors -/
def has_70_divisors (n : ℕ) : Prop := num_divisors n = 70

/-- 25920 is the smallest natural number with exactly 70 divisors -/
theorem smallest_with_70_divisors : 
  has_70_divisors 25920 ∧ ∀ m < 25920, ¬has_70_divisors m :=
sorry

end NUMINAMATH_CALUDE_smallest_with_70_divisors_l1135_113529


namespace NUMINAMATH_CALUDE_lowest_degree_is_four_l1135_113586

/-- A polynomial with coefficients in ℤ -/
def IntPolynomial := Polynomial ℤ

/-- The set of coefficients of a polynomial -/
def coefficientSet (P : IntPolynomial) : Set ℤ :=
  {a : ℤ | ∃ (i : ℕ), a = P.coeff i}

/-- The property that a polynomial satisfies the given conditions -/
def satisfiesCondition (P : IntPolynomial) : Prop :=
  ∃ (b : ℤ), 
    (∃ (x y : ℤ), x ∈ coefficientSet P ∧ y ∈ coefficientSet P ∧ x < b ∧ b < y) ∧
    b ∉ coefficientSet P

/-- The theorem stating that the lowest degree of a polynomial satisfying the condition is 4 -/
theorem lowest_degree_is_four :
  ∃ (P : IntPolynomial), satisfiesCondition P ∧ P.degree = 4 ∧
  ∀ (Q : IntPolynomial), satisfiesCondition Q → Q.degree ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_lowest_degree_is_four_l1135_113586


namespace NUMINAMATH_CALUDE_min_cards_is_smallest_l1135_113546

/-- The smallest number of cards needed to represent all integers from 1 to n! as sums of factorials -/
def min_cards (n : ℕ+) : ℕ :=
  n.val * (n.val + 1) / 2 + 1

/-- Theorem stating that min_cards gives the smallest possible number of cards needed -/
theorem min_cards_is_smallest (n : ℕ+) :
  ∀ (t : ℕ), t ≤ n.val.factorial →
  ∃ (S : Finset ℕ),
    (∀ m ∈ S, ∃ k : ℕ+, m = k.val.factorial) ∧
    (S.card ≤ min_cards n) ∧
    (t = S.sum id) :=
sorry

end NUMINAMATH_CALUDE_min_cards_is_smallest_l1135_113546


namespace NUMINAMATH_CALUDE_fraction_simplification_l1135_113556

theorem fraction_simplification (x : ℝ) (hx : x > 0) :
  (x^(3/4) - 25*x^(1/4)) / (x^(1/2) + 5*x^(1/4)) = x^(1/4) - 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1135_113556


namespace NUMINAMATH_CALUDE_product_decomposition_l1135_113525

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def options : List ℕ := [2986, 2858, 2672, 2754]

theorem product_decomposition :
  ∃! (product : ℕ) (a b : ℕ), 
    product ∈ options ∧
    is_two_digit a ∧
    is_three_digit b ∧
    product = a * b :=
sorry

end NUMINAMATH_CALUDE_product_decomposition_l1135_113525


namespace NUMINAMATH_CALUDE_exponential_equation_solutions_l1135_113521

theorem exponential_equation_solutions :
  ∀ x y : ℕ+, (3 : ℕ) ^ x.val = 2 ^ x.val * y.val + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 5) := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solutions_l1135_113521


namespace NUMINAMATH_CALUDE_union_equality_condition_min_value_of_expression_min_value_achieved_l1135_113572

-- Option B
theorem union_equality_condition (A B : Set α) :
  (A ∪ B = B) ↔ (A ∩ B = A) := by sorry

-- Option D
theorem min_value_of_expression {x y : ℝ} (hx : x > 1) (hy : y > 1) (hxy : x + y = x * y) :
  (2 * x / (x - 1) + 4 * y / (y - 1)) ≥ 6 + 4 * Real.sqrt 2 := by sorry

-- Theorem stating that the minimum value is achieved
theorem min_value_achieved {x y : ℝ} (hx : x > 1) (hy : y > 1) (hxy : x + y = x * y) :
  ∃ (x₀ y₀ : ℝ), x₀ > 1 ∧ y₀ > 1 ∧ x₀ + y₀ = x₀ * y₀ ∧
    (2 * x₀ / (x₀ - 1) + 4 * y₀ / (y₀ - 1)) = 6 + 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_union_equality_condition_min_value_of_expression_min_value_achieved_l1135_113572


namespace NUMINAMATH_CALUDE_sum_of_baby_ages_theorem_l1135_113553

/- Define the ages of the animals -/
def lioness_age : ℕ := 12
def hyena_age : ℕ := lioness_age / 2
def leopard_age : ℕ := 3 * hyena_age

/- Define the ages of the babies -/
def lioness_baby_age : ℕ := lioness_age / 2
def hyena_baby_age : ℕ := hyena_age / 2
def leopard_baby_age : ℕ := leopard_age / 2

/- Define the sum of the babies' ages after 5 years -/
def sum_of_baby_ages_after_5_years : ℕ := 
  (lioness_baby_age + 5) + (hyena_baby_age + 5) + (leopard_baby_age + 5)

theorem sum_of_baby_ages_theorem : sum_of_baby_ages_after_5_years = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_baby_ages_theorem_l1135_113553


namespace NUMINAMATH_CALUDE_valid_outfits_count_l1135_113554

/-- The number of shirts available -/
def num_shirts : ℕ := 7

/-- The number of pairs of pants available -/
def num_pants : ℕ := 5

/-- The number of hats available -/
def num_hats : ℕ := 7

/-- The number of colors available for shirts and hats -/
def num_colors : ℕ := 7

/-- Calculate the number of valid outfits -/
def num_valid_outfits : ℕ := num_shirts * num_pants * num_hats - num_colors * num_pants

theorem valid_outfits_count :
  num_valid_outfits = 210 :=
by sorry

end NUMINAMATH_CALUDE_valid_outfits_count_l1135_113554


namespace NUMINAMATH_CALUDE_estimated_students_above_average_l1135_113536

/-- Represents the time intervals for physical exercise --/
inductive TimeInterval
| LessThan30
| Between30And60
| Between60And90
| Between90And120

/-- Represents the data from the survey --/
structure SurveyData where
  sampleSize : Nat
  totalStudents : Nat
  mean : Nat
  studentsPerInterval : TimeInterval → Nat

/-- Theorem: Given the survey data, prove that the estimated number of students
    spending at least the average time on exercise is 130 --/
theorem estimated_students_above_average (data : SurveyData)
  (h1 : data.sampleSize = 20)
  (h2 : data.totalStudents = 200)
  (h3 : data.mean = 60)
  (h4 : data.studentsPerInterval TimeInterval.LessThan30 = 2)
  (h5 : data.studentsPerInterval TimeInterval.Between30And60 = 5)
  (h6 : data.studentsPerInterval TimeInterval.Between60And90 = 10)
  (h7 : data.studentsPerInterval TimeInterval.Between90And120 = 3) :
  (data.totalStudents * (data.studentsPerInterval TimeInterval.Between60And90 +
   data.studentsPerInterval TimeInterval.Between90And120) / data.sampleSize) = 130 := by
  sorry


end NUMINAMATH_CALUDE_estimated_students_above_average_l1135_113536


namespace NUMINAMATH_CALUDE_triangle_c_coordinates_l1135_113516

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the area function
def area (t : Triangle) : ℝ := sorry

-- Define the line equation
def onLine (p : ℝ × ℝ) : Prop :=
  3 * p.1 - p.2 + 3 = 0

-- Theorem statement
theorem triangle_c_coordinates :
  ∀ (t : Triangle),
    t.A = (3, 2) →
    t.B = (-1, 5) →
    onLine t.C →
    area t = 10 →
    (t.C = (-1, 0) ∨ t.C = (5/3, 8)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_c_coordinates_l1135_113516


namespace NUMINAMATH_CALUDE_expression_evaluation_l1135_113539

theorem expression_evaluation : 
  let f (x : ℚ) := (x + 2) / (x - 2)
  let g (x : ℚ) := (f x + 2) / (f x - 2)
  g (1/3) = -31/37 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1135_113539


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2014th_term_l1135_113506

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a_1_eq_1 : a 1 = 1
  d : ℝ
  d_ne_0 : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  is_geometric : (a 2) ^ 2 = a 1 * a 5

/-- The 2014th term of the arithmetic sequence is 4027 -/
theorem arithmetic_sequence_2014th_term (seq : ArithmeticSequence) : seq.a 2014 = 4027 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2014th_term_l1135_113506


namespace NUMINAMATH_CALUDE_x_values_proof_l1135_113568

theorem x_values_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 7) (h2 : y + 1 / x = 7 / 8) :
  x = 1 ∨ x = 8 := by
sorry

end NUMINAMATH_CALUDE_x_values_proof_l1135_113568


namespace NUMINAMATH_CALUDE_fraction_addition_l1135_113582

theorem fraction_addition (y C D : ℚ) : 
  (6 * y - 15) / (3 * y^3 - 13 * y^2 + 4 * y + 12) = C / (y + 3) + D / (3 * y^2 - 10 * y + 4) →
  C = -3/17 ∧ D = 81/17 := by
sorry

end NUMINAMATH_CALUDE_fraction_addition_l1135_113582


namespace NUMINAMATH_CALUDE_expansion_equals_fourth_power_l1135_113567

theorem expansion_equals_fourth_power (x : ℝ) : 
  (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*(x - 1) + 1 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equals_fourth_power_l1135_113567


namespace NUMINAMATH_CALUDE_product_is_solution_quotient_is_solution_l1135_113559

/-- A type representing solutions of the equation x^2 - 5y^2 = 1 -/
structure Solution where
  x : ℝ
  y : ℝ
  property : x^2 - 5*y^2 = 1

/-- The product of two solutions is also a solution -/
theorem product_is_solution (s₁ s₂ : Solution) :
  ∃ (m n : ℝ), m^2 - 5*n^2 = 1 ∧ m + n * Real.sqrt 5 = (s₁.x + s₁.y * Real.sqrt 5) * (s₂.x + s₂.y * Real.sqrt 5) :=
by sorry

/-- The quotient of two solutions can be represented as p + q√5 and is also a solution -/
theorem quotient_is_solution (s₁ s₂ : Solution) (h : s₂.x^2 - 5*s₂.y^2 ≠ 0) :
  ∃ (p q : ℝ), p^2 - 5*q^2 = 1 ∧ p + q * Real.sqrt 5 = (s₁.x + s₁.y * Real.sqrt 5) / (s₂.x + s₂.y * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_product_is_solution_quotient_is_solution_l1135_113559


namespace NUMINAMATH_CALUDE_multiply_three_six_and_quarter_l1135_113547

theorem multiply_three_six_and_quarter : 3.6 * 0.25 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_six_and_quarter_l1135_113547


namespace NUMINAMATH_CALUDE_quartic_polynomial_theorem_l1135_113542

-- Define a quartic polynomial
def is_quartic_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d e : ℝ, ∀ x, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e

-- Define the condition that p(n) = 1/n^2 for n = 1, 2, 3, 4, 5
def satisfies_condition (p : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → p n = 1 / (n^2 : ℝ)

theorem quartic_polynomial_theorem (p : ℝ → ℝ) 
  (h1 : is_quartic_polynomial p) 
  (h2 : satisfies_condition p) : 
  p 6 = -67/180 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_theorem_l1135_113542


namespace NUMINAMATH_CALUDE_hyperbola_range_theorem_l1135_113513

/-- The range of m for which the equation represents a hyperbola -/
def hyperbola_range : Set ℝ := Set.union (Set.Ioo (-1) 1) (Set.Ioi 2)

/-- The equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (|m| - 1) + y^2 / (2 - m) = 1 ∧
  ((|m| - 1 > 0 ∧ 2 - m < 0) ∨ (|m| - 1 < 0 ∧ 2 - m > 0))

/-- Theorem stating the range of m for which the equation represents a hyperbola -/
theorem hyperbola_range_theorem :
  ∀ m : ℝ, is_hyperbola m ↔ m ∈ hyperbola_range :=
sorry

end NUMINAMATH_CALUDE_hyperbola_range_theorem_l1135_113513


namespace NUMINAMATH_CALUDE_points_on_line_procedure_l1135_113555

theorem points_on_line_procedure (x : ℕ) : ∃ x > 0, 9*x - 8 = 82 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_procedure_l1135_113555


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l1135_113512

theorem pure_imaginary_complex_fraction (a : ℝ) :
  let z : ℂ := (a - Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l1135_113512


namespace NUMINAMATH_CALUDE_danny_drive_to_work_l1135_113560

/-- Represents the distance Danny drives between different locations -/
structure DannyDrive where
  x : ℝ  -- Distance from Danny's house to the first friend's house
  first_to_second : ℝ := 0.5 * x
  second_to_third : ℝ := 2 * x
  third_to_fourth : ℝ  -- Will be calculated
  fourth_to_work : ℝ   -- To be proven

/-- Calculates the total distance driven up to the third friend's house -/
def total_to_third (d : DannyDrive) : ℝ :=
  d.x + d.first_to_second + d.second_to_third

/-- Theorem stating the distance Danny drives between the fourth friend's house and work -/
theorem danny_drive_to_work (d : DannyDrive) 
    (h1 : d.third_to_fourth = (1/3) * total_to_third d) 
    (h2 : d.fourth_to_work = 3 * (total_to_third d + d.third_to_fourth)) : 
  d.fourth_to_work = 14 * d.x := by
  sorry


end NUMINAMATH_CALUDE_danny_drive_to_work_l1135_113560


namespace NUMINAMATH_CALUDE_book_price_change_l1135_113515

/-- Given a book with an initial price of $400, prove that after a 15% decrease
    followed by a 40% increase, the final price is $476. -/
theorem book_price_change (initial_price : ℝ) (decrease_percent : ℝ) (increase_percent : ℝ) :
  initial_price = 400 →
  decrease_percent = 15 →
  increase_percent = 40 →
  let price_after_decrease := initial_price * (1 - decrease_percent / 100)
  let final_price := price_after_decrease * (1 + increase_percent / 100)
  final_price = 476 := by
  sorry

#check book_price_change

end NUMINAMATH_CALUDE_book_price_change_l1135_113515


namespace NUMINAMATH_CALUDE_loan_amount_calculation_l1135_113510

/-- Proves that given the initial amount, interest rate, and final amount, 
    the calculated loan amount is correct. -/
theorem loan_amount_calculation 
  (initial_amount : ℝ) 
  (interest_rate : ℝ) 
  (final_amount : ℝ) 
  (loan_amount : ℝ) : 
  initial_amount = 30 ∧ 
  interest_rate = 0.20 ∧ 
  final_amount = 33 ∧
  loan_amount = 2.50 → 
  initial_amount + loan_amount * (1 + interest_rate) = final_amount :=
by sorry

end NUMINAMATH_CALUDE_loan_amount_calculation_l1135_113510


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l1135_113527

theorem power_of_power_at_three :
  (3^3)^(3^3) = 27^27 := by sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l1135_113527


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1135_113509

theorem smallest_integer_with_remainders : ∃! N : ℕ+, 
  (N : ℤ) % 7 = 5 ∧ 
  (N : ℤ) % 8 = 6 ∧ 
  (N : ℤ) % 9 = 7 ∧ 
  ∀ M : ℕ+, 
    ((M : ℤ) % 7 = 5 ∧ (M : ℤ) % 8 = 6 ∧ (M : ℤ) % 9 = 7) → N ≤ M :=
by
  use 502
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1135_113509


namespace NUMINAMATH_CALUDE_complex_magnitude_plus_fraction_l1135_113518

theorem complex_magnitude_plus_fraction :
  Complex.abs (3/4 - 3*Complex.I) + 5/12 = (9*Real.sqrt 17 + 5) / 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_plus_fraction_l1135_113518


namespace NUMINAMATH_CALUDE_max_value_of_f_l1135_113571

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 * Real.exp x

theorem max_value_of_f (h : a ≠ 0) :
  (a > 0 → ∃ M, M = 4 * a * Real.exp (-2) ∧ ∀ x, f a x ≤ M) ∧
  (a < 0 → ∃ M, M = 0 ∧ ∀ x, f a x ≤ M) :=
sorry

end

end NUMINAMATH_CALUDE_max_value_of_f_l1135_113571


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1135_113578

theorem regular_polygon_sides (n : ℕ) (angle : ℝ) : 
  n > 0 → 
  angle > 0 → 
  angle < 180 → 
  (360 : ℝ) / n = angle → 
  angle = 20 → 
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1135_113578


namespace NUMINAMATH_CALUDE_triangle_on_parabola_ef_length_l1135_113501

/-- Parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- Vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 0)

/-- Triangle DEF -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The theorem to be proved -/
theorem triangle_on_parabola_ef_length (t : Triangle) :
  t.D = vertex ∧
  (∀ x, (x, parabola x) = t.D ∨ (x, parabola x) = t.E ∨ (x, parabola x) = t.F) ∧
  t.E.2 = t.F.2 ∧
  (1/2 * (t.F.1 - t.E.1) * (t.E.2 - t.D.2) = 32) →
  t.F.1 - t.E.1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_on_parabola_ef_length_l1135_113501


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l1135_113526

theorem mistaken_multiplication (x : ℚ) : 
  6 * x = 12 → 7 * x = 14 := by
sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l1135_113526


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1135_113584

theorem inequality_solution_set (x : ℝ) :
  |x + 3| - |2*x - 1| < x/2 + 1 ↔ x < -2/5 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1135_113584


namespace NUMINAMATH_CALUDE_propositions_truth_l1135_113502

theorem propositions_truth : 
  (∀ x : ℝ, x < 0 → abs x > x) ∧ 
  (∀ a b : ℝ, a * b < 0 ↔ a / b < 0) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l1135_113502


namespace NUMINAMATH_CALUDE_equation_solutions_l1135_113589

theorem equation_solutions : 
  let f (x : ℝ) := 
    10 / (Real.sqrt (x - 10) - 10) + 
    2 / (Real.sqrt (x - 10) - 5) + 
    14 / (Real.sqrt (x - 10) + 5) + 
    20 / (Real.sqrt (x - 10) + 10)
  ∀ x : ℝ, f x = 0 ↔ (x = 190 / 9 ∨ x = 5060 / 256) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1135_113589


namespace NUMINAMATH_CALUDE_sum_of_smallest_and_largest_l1135_113558

-- Define the property of three consecutive even numbers
def ConsecutiveEvenNumbers (a b c : ℕ) : Prop :=
  ∃ n : ℕ, a = 2 * n ∧ b = 2 * n + 2 ∧ c = 2 * n + 4

theorem sum_of_smallest_and_largest (a b c : ℕ) :
  ConsecutiveEvenNumbers a b c → a + b + c = 1194 → a + c = 796 := by
  sorry

#check sum_of_smallest_and_largest

end NUMINAMATH_CALUDE_sum_of_smallest_and_largest_l1135_113558


namespace NUMINAMATH_CALUDE_symmetric_point_line_equation_l1135_113563

/-- Given points A and M, if B is symmetric to A with respect to M, 
    and line l passes through the origin and point B, 
    then the equation of line l is 7x + 5y = 0 -/
theorem symmetric_point_line_equation 
  (A : ℝ × ℝ) (M : ℝ × ℝ) (B : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  A = (3, 1) →
  M = (4, -3) →
  B.1 = 2 * M.1 - A.1 →
  B.2 = 2 * M.2 - A.2 →
  (0, 0) ∈ l →
  B ∈ l →
  ∀ (x y : ℝ), (x, y) ∈ l ↔ 7 * x + 5 * y = 0 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_line_equation_l1135_113563


namespace NUMINAMATH_CALUDE_system_solution_l1135_113557

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 3 * y = -9) ∧ (5 * x + 4 * y = 14) ∧ (x = 6/31) ∧ (y = 101/31) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1135_113557


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_5_l1135_113573

-- Define a function to calculate the sum of digits in a number
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the property of being the first year after 2020 with digit sum 5
def isFirstYearAfter2020WithDigitSum5 (year : ℕ) : Prop :=
  year > 2020 ∧
  sumOfDigits year = 5 ∧
  ∀ y, 2020 < y ∧ y < year → sumOfDigits y ≠ 5

-- Theorem statement
theorem first_year_after_2020_with_digit_sum_5 :
  sumOfDigits 2020 = 4 →
  isFirstYearAfter2020WithDigitSum5 2021 :=
by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_5_l1135_113573


namespace NUMINAMATH_CALUDE_egg_yolk_count_l1135_113503

theorem egg_yolk_count (total_eggs : ℕ) (double_yolk_eggs : ℕ) : 
  total_eggs = 12 → double_yolk_eggs = 5 → 
  (total_eggs - double_yolk_eggs) + 2 * double_yolk_eggs = 17 := by
  sorry

#check egg_yolk_count

end NUMINAMATH_CALUDE_egg_yolk_count_l1135_113503


namespace NUMINAMATH_CALUDE_peaches_per_box_is_15_l1135_113598

/-- Given the initial number of peaches per basket, the number of baskets,
    the number of peaches eaten by farmers, and the number of smaller boxes,
    calculate the number of peaches in each smaller box. -/
def peaches_per_box (initial_peaches_per_basket : ℕ) (num_baskets : ℕ) 
                    (peaches_eaten : ℕ) (num_smaller_boxes : ℕ) : ℕ :=
  ((initial_peaches_per_basket * num_baskets) - peaches_eaten) / num_smaller_boxes

/-- Theorem stating that given the specific conditions in the problem,
    the number of peaches in each smaller box is 15. -/
theorem peaches_per_box_is_15 :
  peaches_per_box 25 5 5 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_peaches_per_box_is_15_l1135_113598


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1135_113551

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 + a 2 + a 3 = 6) →
  (a 7 + a 8 + a 9 = 24) →
  (a 4 + a 5 + a 6 = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1135_113551


namespace NUMINAMATH_CALUDE_largest_parallelogram_perimeter_l1135_113580

/-- Triangle with sides 13, 13, and 12 -/
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- Parallelogram formed by four copies of a triangle -/
def Parallelogram (t : Triangle) :=
  { p : ℝ // ∃ (a b c d : ℝ), 
    a + b + c + d = p ∧
    a ≤ t.side1 ∧ b ≤ t.side1 ∧ c ≤ t.side2 ∧ d ≤ t.side3 }

/-- The theorem stating the largest possible perimeter of the parallelogram -/
theorem largest_parallelogram_perimeter :
  let t : Triangle := { side1 := 13, side2 := 13, side3 := 12 }
  ∀ p : Parallelogram t, p.val ≤ 76 :=
by sorry

end NUMINAMATH_CALUDE_largest_parallelogram_perimeter_l1135_113580


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1135_113569

theorem floor_equation_solution (x : ℝ) : 
  ⌊(3:ℝ) * x + 4⌋ = ⌊(5:ℝ) * x - 1⌋ ↔ 
  ((11:ℝ)/5 ≤ x ∧ x < 7/3) ∨ 
  ((12:ℝ)/5 ≤ x ∧ x < 13/5) ∨ 
  ((8:ℝ)/3 ≤ x ∧ x < 14/5) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1135_113569


namespace NUMINAMATH_CALUDE_divisibility_problem_l1135_113508

theorem divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    ((a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) →
    ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1135_113508


namespace NUMINAMATH_CALUDE_horner_v3_value_l1135_113543

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 2x^4 + 3.5x^3 - 2.6x^2 + 1.7x - 0.8 -/
def f : ℝ → ℝ := fun x => 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- The coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [-0.8, 1.7, -2.6, 3.5, 2, 5]

/-- Theorem: The third intermediate value (v_3) in Horner's method for f(x) at x=1 is 7.9 -/
theorem horner_v3_value : 
  (horner (coeffs.take 4) 1) = 7.9 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l1135_113543


namespace NUMINAMATH_CALUDE_rice_purchase_difference_l1135_113593

/-- Represents the price and quantity of rice from a supplier -/
structure RiceSupply where
  quantity : ℝ
  price : ℝ

/-- Calculates the total cost of rice supplies -/
def totalCost (supplies : List RiceSupply) : ℝ :=
  supplies.foldl (fun acc supply => acc + supply.quantity * supply.price) 0

/-- Represents the rice purchase scenario -/
structure RicePurchase where
  supplies : List RiceSupply
  keptRatio : ℝ
  conversionRate : ℝ

theorem rice_purchase_difference (purchase : RicePurchase) 
  (h1 : purchase.supplies = [
    ⟨15, 1.2⟩, ⟨10, 1.4⟩, ⟨12, 1.6⟩, ⟨8, 1.9⟩, ⟨5, 2.3⟩
  ])
  (h2 : purchase.keptRatio = 7/10)
  (h3 : purchase.conversionRate = 1.15) :
  let totalCostEuros := totalCost purchase.supplies
  let keptCostDollars := totalCostEuros * purchase.keptRatio * purchase.conversionRate
  let givenCostDollars := totalCostEuros * (1 - purchase.keptRatio) * purchase.conversionRate
  keptCostDollars - givenCostDollars = 35.88 := by
  sorry

end NUMINAMATH_CALUDE_rice_purchase_difference_l1135_113593


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1135_113583

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + 2 * a 6 + a 10 = 120) →
  (a 3 + a 9 = 60) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1135_113583


namespace NUMINAMATH_CALUDE_george_christopher_age_difference_l1135_113507

theorem george_christopher_age_difference :
  ∀ (G C F : ℕ),
    C = 18 →
    F = C - 2 →
    G + C + F = 60 →
    G > C →
    G - C = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_george_christopher_age_difference_l1135_113507


namespace NUMINAMATH_CALUDE_friends_marbles_theorem_l1135_113595

/-- Calculates the number of marbles Reggie's friend arrived with -/
def friends_initial_marbles (total_games : ℕ) (marbles_per_game : ℕ) (reggies_final_marbles : ℕ) (games_lost : ℕ) : ℕ :=
  let games_won := total_games - games_lost
  let marbles_gained := games_won * marbles_per_game
  let reggies_initial_marbles := reggies_final_marbles - marbles_gained
  reggies_initial_marbles + marbles_per_game

theorem friends_marbles_theorem (total_games : ℕ) (marbles_per_game : ℕ) (reggies_final_marbles : ℕ) (games_lost : ℕ)
  (h1 : total_games = 9)
  (h2 : marbles_per_game = 10)
  (h3 : reggies_final_marbles = 90)
  (h4 : games_lost = 1) :
  friends_initial_marbles total_games marbles_per_game reggies_final_marbles games_lost = 20 := by
  sorry

#eval friends_initial_marbles 9 10 90 1

end NUMINAMATH_CALUDE_friends_marbles_theorem_l1135_113595


namespace NUMINAMATH_CALUDE_value_of_a_l1135_113587

theorem value_of_a (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^3 / b = 1) (h2 : b^3 / c = 8) (h3 : c^3 / a = 27) :
  a = (24^(1/8 : ℝ))^(1/3 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l1135_113587


namespace NUMINAMATH_CALUDE_pet_store_combinations_l1135_113522

def num_puppies : ℕ := 12
def num_kittens : ℕ := 8
def num_hamsters : ℕ := 10
def num_birds : ℕ := 5

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * num_birds * 4 * 3 * 2 * 1 = 115200 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l1135_113522


namespace NUMINAMATH_CALUDE_mrs_hilt_pies_l1135_113549

/-- The total number of pies Mrs. Hilt needs to bake for the bigger event -/
def total_pies (pecan_initial : Float) (apple_initial : Float) (cherry_initial : Float)
                (pecan_multiplier : Float) (apple_multiplier : Float) (cherry_multiplier : Float) : Float :=
  pecan_initial * pecan_multiplier + apple_initial * apple_multiplier + cherry_initial * cherry_multiplier

/-- Theorem stating that Mrs. Hilt needs to bake 193.5 pies for the bigger event -/
theorem mrs_hilt_pies : 
  total_pies 16.5 14.25 12.75 4.3 3.5 5.7 = 193.5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pies_l1135_113549


namespace NUMINAMATH_CALUDE_binomial_coefficient_inequality_l1135_113570

theorem binomial_coefficient_inequality (n k : ℕ) (h1 : n > k) (h2 : k > 0) : 
  (1 : ℝ) / (n + 1 : ℝ) * (n^n : ℝ) / ((k^k * (n-k)^(n-k)) : ℝ) < 
  (n.factorial : ℝ) / ((k.factorial * (n-k).factorial) : ℝ) ∧
  (n.factorial : ℝ) / ((k.factorial * (n-k).factorial) : ℝ) < 
  (n^n : ℝ) / ((k^k * (n-k)^(n-k)) : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_inequality_l1135_113570


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1135_113531

/-- Proves that the average speed for a 60-mile trip with specified conditions is 30 mph -/
theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (speed_increase : ℝ) :
  total_distance = 60 →
  first_half_speed = 24 →
  speed_increase = 16 →
  let second_half_speed := first_half_speed + speed_increase
  let first_half_time := (total_distance / 2) / first_half_speed
  let second_half_time := (total_distance / 2) / second_half_speed
  let total_time := first_half_time + second_half_time
  total_distance / total_time = 30 := by
  sorry

#check average_speed_calculation

end NUMINAMATH_CALUDE_average_speed_calculation_l1135_113531


namespace NUMINAMATH_CALUDE_melanie_books_before_l1135_113524

/-- The number of books Melanie had before the yard sale -/
def books_before : ℕ := sorry

/-- The number of books Melanie bought at the yard sale -/
def books_bought : ℕ := 46

/-- The total number of books Melanie has after the yard sale -/
def books_after : ℕ := 87

/-- Theorem stating that Melanie had 41 books before the yard sale -/
theorem melanie_books_before : books_before = 41 := by sorry

end NUMINAMATH_CALUDE_melanie_books_before_l1135_113524


namespace NUMINAMATH_CALUDE_min_score_given_average_l1135_113528

theorem min_score_given_average (x y : ℝ) : 
  x ≥ 0 ∧ x ≤ 100 ∧ 
  y ≥ 0 ∧ y ≤ 100 ∧ 
  (69 + 53 + 69 + 71 + 78 + x + y) / 7 = 66 →
  x ≥ 22 :=
by sorry

end NUMINAMATH_CALUDE_min_score_given_average_l1135_113528


namespace NUMINAMATH_CALUDE_audience_with_envelopes_l1135_113535

theorem audience_with_envelopes (total_audience : ℕ) (winners : ℕ) (winning_percentage : ℚ) :
  total_audience = 100 →
  winners = 8 →
  winning_percentage = 1/5 →
  (winners : ℚ) / (winning_percentage * total_audience) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_audience_with_envelopes_l1135_113535


namespace NUMINAMATH_CALUDE_trapezoid_area_l1135_113523

/-- The area of a trapezoid with height x, one base 3x, and the other base 5x, is 4x² -/
theorem trapezoid_area (x : ℝ) (h : x > 0) : 
  let height := x
  let base1 := 3 * x
  let base2 := 5 * x
  let area := height * (base1 + base2) / 2
  area = 4 * x^2 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1135_113523


namespace NUMINAMATH_CALUDE_major_premise_is_false_l1135_113574

theorem major_premise_is_false : ¬ ∀ (a : ℝ) (n : ℕ), (a^(1/n : ℝ))^n = a := by sorry

end NUMINAMATH_CALUDE_major_premise_is_false_l1135_113574


namespace NUMINAMATH_CALUDE_employee_b_pay_is_220_l1135_113544

/-- Given two employees A and B with a total weekly pay and A's pay as a percentage of B's, 
    calculate B's weekly pay. -/
def calculate_employee_b_pay (total_pay : ℚ) (a_percentage : ℚ) : ℚ :=
  total_pay / (1 + a_percentage)

/-- Theorem stating that given the problem conditions, employee B's pay is 220. -/
theorem employee_b_pay_is_220 :
  calculate_employee_b_pay 550 (3/2) = 220 := by sorry

end NUMINAMATH_CALUDE_employee_b_pay_is_220_l1135_113544


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1135_113577

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g (x : ℚ) := c * x^3 - 8 * x^2 + d * x - 7
  (g 2 = -15) ∧ (g (-3) = -140) → c = 36/7 ∧ d = -109/7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1135_113577


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1135_113537

theorem sufficient_not_necessary_condition (m : ℝ) : 
  (∀ x : ℝ, |x - 4| ≤ 6 → x ≤ 1 + m) ∧ 
  (∃ x : ℝ, x ≤ 1 + m ∧ |x - 4| > 6) ↔ 
  m ≥ 9 := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1135_113537


namespace NUMINAMATH_CALUDE_average_popped_percentage_is_82_l1135_113599

/-- Represents a bag of popcorn kernels -/
structure PopcornBag where
  popped : ℕ
  total : ℕ

/-- Calculates the percentage of popped kernels in a bag -/
def percentPopped (bag : PopcornBag) : ℚ :=
  (bag.popped : ℚ) / (bag.total : ℚ) * 100

theorem average_popped_percentage_is_82 (bag1 bag2 bag3 : PopcornBag)
    (h1 : bag1 = ⟨60, 75⟩)
    (h2 : bag2 = ⟨42, 50⟩)
    (h3 : bag3 = ⟨82, 100⟩) :
    (percentPopped bag1 + percentPopped bag2 + percentPopped bag3) / 3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_average_popped_percentage_is_82_l1135_113599


namespace NUMINAMATH_CALUDE_outfit_combinations_l1135_113550

def num_shirts : ℕ := 8
def num_ties : ℕ := 5
def num_pants : ℕ := 4

theorem outfit_combinations : num_shirts * num_ties * num_pants = 160 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1135_113550


namespace NUMINAMATH_CALUDE_sample_size_theorem_l1135_113540

theorem sample_size_theorem (N : ℕ) (sample_size : ℕ) (prob : ℚ) : 
  sample_size = 30 → prob = 1/4 → N * prob = sample_size → N = 120 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_theorem_l1135_113540


namespace NUMINAMATH_CALUDE_hydrogen_atom_count_l1135_113533

/-- Represents the number of atoms of each element in the compound -/
structure AtomCount where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

/-- Calculates the molecular weight of a compound -/
def molecularWeight (count : AtomCount) (weights : AtomicWeights) : ℝ :=
  count.carbon * weights.carbon + count.hydrogen * weights.hydrogen + count.oxygen * weights.oxygen

/-- The main theorem stating the number of hydrogen atoms in the compound -/
theorem hydrogen_atom_count (weights : AtomicWeights) 
    (h_carbon : weights.carbon = 12)
    (h_hydrogen : weights.hydrogen = 1)
    (h_oxygen : weights.oxygen = 16) : 
  ∃ (count : AtomCount), 
    count.carbon = 3 ∧ 
    count.oxygen = 1 ∧ 
    molecularWeight count weights = 58 ∧ 
    count.hydrogen = 6 := by
  sorry

end NUMINAMATH_CALUDE_hydrogen_atom_count_l1135_113533


namespace NUMINAMATH_CALUDE_peter_twice_harriet_age_l1135_113576

/- Define the current ages and time span -/
def mother_age : ℕ := 60
def harriet_age : ℕ := 13
def years_passed : ℕ := 4

/- Define Peter's current age based on the given condition -/
def peter_age : ℕ := mother_age / 2

/- Define future ages -/
def peter_future_age : ℕ := peter_age + years_passed
def harriet_future_age : ℕ := harriet_age + years_passed

/- Theorem to prove -/
theorem peter_twice_harriet_age : 
  peter_future_age = 2 * harriet_future_age := by
  sorry


end NUMINAMATH_CALUDE_peter_twice_harriet_age_l1135_113576


namespace NUMINAMATH_CALUDE_chalk_per_box_l1135_113592

def total_chalk : ℕ := 3484
def full_boxes : ℕ := 194

theorem chalk_per_box : total_chalk / full_boxes = 18 := by
  sorry

end NUMINAMATH_CALUDE_chalk_per_box_l1135_113592
