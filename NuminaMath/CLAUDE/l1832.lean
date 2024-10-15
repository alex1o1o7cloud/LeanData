import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_discriminant_condition_l1832_183234

theorem quadratic_discriminant_condition (a b c : ℝ) :
  (2 * a ≠ 0) →
  (ac = (9 * b^2 - 25) / 32) ↔ ((3 * b)^2 - 4 * (2 * a) * (4 * c) = 25) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_condition_l1832_183234


namespace NUMINAMATH_CALUDE_expression_value_l1832_183247

theorem expression_value (x : ℝ) (h : x^2 + 3*x - 1 = 0) : 
  (x - 3)^2 - (2*x + 1)*(2*x - 1) - 3*x = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1832_183247


namespace NUMINAMATH_CALUDE_cos_A_from_tan_A_l1832_183253

theorem cos_A_from_tan_A (A : Real) (h : Real.tan A = 2/3) : 
  Real.cos A = 3 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_cos_A_from_tan_A_l1832_183253


namespace NUMINAMATH_CALUDE_sandwich_combinations_l1832_183286

theorem sandwich_combinations (meat_types : ℕ) (cheese_types : ℕ) : 
  meat_types = 12 → cheese_types = 8 → 
  (meat_types.choose 2) * cheese_types = 528 :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1832_183286


namespace NUMINAMATH_CALUDE_circle_inequality_l1832_183212

theorem circle_inequality (c : ℝ) : 
  (∀ x y : ℝ, x^2 + (y-1)^2 = 1 → x + y + c ≥ 0) → 
  c ≥ Real.sqrt 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_circle_inequality_l1832_183212


namespace NUMINAMATH_CALUDE_employment_percentage_l1832_183266

theorem employment_percentage (population : ℝ) (employed : ℝ) 
  (h1 : employed > 0) 
  (h2 : population > 0) 
  (h3 : 0.42 * population = 0.7 * employed) : 
  employed / population = 0.6 := by
sorry

end NUMINAMATH_CALUDE_employment_percentage_l1832_183266


namespace NUMINAMATH_CALUDE_restaurant_hotdogs_l1832_183217

theorem restaurant_hotdogs (hotdogs : ℕ) (pizzas : ℕ) : 
  pizzas = hotdogs + 40 →
  30 * (hotdogs + pizzas) = 4800 →
  hotdogs = 60 := by
sorry

end NUMINAMATH_CALUDE_restaurant_hotdogs_l1832_183217


namespace NUMINAMATH_CALUDE_max_rental_income_l1832_183209

/-- Represents the daily rental income for the construction company. -/
def daily_rental_income (x : ℕ) : ℝ :=
  -200 * x + 80000

/-- The problem statement and proof objective. -/
theorem max_rental_income :
  let total_vehicles : ℕ := 50
  let type_a_vehicles : ℕ := 20
  let type_b_vehicles : ℕ := 30
  let site_a_vehicles : ℕ := 30
  let site_b_vehicles : ℕ := 20
  let site_a_type_a_price : ℝ := 1800
  let site_a_type_b_price : ℝ := 1600
  let site_b_type_a_price : ℝ := 1600
  let site_b_type_b_price : ℝ := 1200
  ∀ x : ℕ, x ≤ type_a_vehicles →
    daily_rental_income x ≤ 80000 ∧
    (∃ x₀ : ℕ, x₀ ≤ type_a_vehicles ∧ daily_rental_income x₀ = 80000) :=
by sorry

#check max_rental_income

end NUMINAMATH_CALUDE_max_rental_income_l1832_183209


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l1832_183258

/-- Proves the number of adult tickets sold given ticket prices and total sales information -/
theorem adult_tickets_sold
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_revenue : ℕ)
  (total_tickets : ℕ)
  (h1 : adult_price = 8)
  (h2 : child_price = 5)
  (h3 : total_revenue = 236)
  (h4 : total_tickets = 34)
  : ∃ (adult_tickets : ℕ) (child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_revenue ∧
    adult_tickets = 22 := by
  sorry


end NUMINAMATH_CALUDE_adult_tickets_sold_l1832_183258


namespace NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l1832_183299

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x^2 - 5*x - 6 < 0
def inequality2 (x : ℝ) : Prop := (x - 1) / (x + 2) ≤ 0

-- Define the solution sets
def solution_set1 : Set ℝ := {x | -1 < x ∧ x < 6}
def solution_set2 : Set ℝ := {x | -2 < x ∧ x ≤ 1}

-- Theorem statements
theorem inequality1_solution : 
  ∀ x : ℝ, inequality1 x ↔ x ∈ solution_set1 :=
sorry

theorem inequality2_solution : 
  ∀ x : ℝ, x ≠ -2 → (inequality2 x ↔ x ∈ solution_set2) :=
sorry

end NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l1832_183299


namespace NUMINAMATH_CALUDE_set_union_problem_l1832_183215

theorem set_union_problem (a b : ℝ) :
  let M : Set ℝ := {a, b}
  let N : Set ℝ := {a + 1, 3}
  M ∩ N = {2} →
  M ∪ N = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l1832_183215


namespace NUMINAMATH_CALUDE_simple_interest_principal_l1832_183293

/-- Simple interest calculation -/
theorem simple_interest_principal (rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ) :
  rate = 0.05 →
  time = 1 →
  interest = 500 →
  principal * rate * time = interest →
  principal = 10000 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l1832_183293


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1832_183264

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- Predicate for an arithmetic sequence. -/
def IsArithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : Sequence) 
  (h1 : IsArithmetic a) 
  (h2 : a 1 + a 6 + a 11 = 3) : 
  a 3 + a 9 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1832_183264


namespace NUMINAMATH_CALUDE_closure_union_M_N_l1832_183200

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x : ℝ | x ≤ -3}

-- State the theorem
theorem closure_union_M_N :
  closure (M ∪ N) = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_closure_union_M_N_l1832_183200


namespace NUMINAMATH_CALUDE_common_terms_is_geometric_l1832_183235

/-- Arithmetic sequence with sum of first n terms S_n = (3n^2 + 5n) / 2 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  3 * n + 1

/-- Geometric sequence with b_3 = 4 and b_6 = 32 -/
def geometric_sequence (n : ℕ) : ℚ :=
  2^(n - 1)

/-- Sequence of common terms between arithmetic_sequence and geometric_sequence -/
def common_terms (n : ℕ) : ℚ :=
  4^n

theorem common_terms_is_geometric :
  ∀ n : ℕ, n > 0 → ∃ k : ℕ, k > 0 ∧ 
    arithmetic_sequence k = geometric_sequence k ∧
    common_terms n = arithmetic_sequence k := by
  sorry

end NUMINAMATH_CALUDE_common_terms_is_geometric_l1832_183235


namespace NUMINAMATH_CALUDE_xfx_nonnegative_set_l1832_183277

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

theorem xfx_nonnegative_set (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_monotone : monotone_decreasing_on f (Set.Iic 0))
  (h_f2 : f 2 = 0) :
  {x : ℝ | x * f x ≥ 0} = Set.Icc (-2) 0 ∪ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_xfx_nonnegative_set_l1832_183277


namespace NUMINAMATH_CALUDE_train_speed_l1832_183237

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 160) (h2 : time = 6) :
  length / time = 26.67 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1832_183237


namespace NUMINAMATH_CALUDE_inequality_solution_l1832_183204

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 5 / (x + 4) ≥ 1) ↔ (x ∈ Set.Icc (-4 : ℝ) (-2) ∨ x ∈ Set.Ico (-2 : ℝ) 5) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1832_183204


namespace NUMINAMATH_CALUDE_chipmunk_families_left_l1832_183218

theorem chipmunk_families_left (original : ℕ) (went_away : ℕ) (h1 : original = 86) (h2 : went_away = 65) :
  original - went_away = 21 := by
  sorry

end NUMINAMATH_CALUDE_chipmunk_families_left_l1832_183218


namespace NUMINAMATH_CALUDE_power_of_54_l1832_183207

theorem power_of_54 (a b : ℕ+) (h : (54 : ℕ) ^ a.val = a.val ^ b.val) :
  ∃ k : ℕ, a.val = (54 : ℕ) ^ k := by
sorry

end NUMINAMATH_CALUDE_power_of_54_l1832_183207


namespace NUMINAMATH_CALUDE_school_seat_cost_l1832_183230

/-- Calculates the total cost of seats with discounts applied --/
def totalCostWithDiscounts (
  rows1 : ℕ) (seats1 : ℕ) (price1 : ℕ) (discount1 : ℚ)
  (rows2 : ℕ) (seats2 : ℕ) (price2 : ℕ) (discount2 : ℚ) (extraDiscount2 : ℚ)
  (rows3 : ℕ) (seats3 : ℕ) (price3 : ℕ) (discount3 : ℚ) : ℚ :=
  let totalSeats1 := rows1 * seats1
  let totalSeats2 := rows2 * seats2
  let totalSeats3 := rows3 * seats3
  let cost1 := totalSeats1 * price1
  let cost2 := totalSeats2 * price2
  let cost3 := totalSeats3 * price3
  let discountedCost1 := cost1 * (1 - discount1 * (totalSeats1 / seats1))
  let discountedCost2 := 
    if totalSeats2 ≥ 30 then
      cost2 * (1 - discount2 * (totalSeats2 / seats2)) * (1 - extraDiscount2)
    else
      cost2 * (1 - discount2 * (totalSeats2 / seats2))
  let discountedCost3 := cost3 * (1 - discount3 * (totalSeats3 / seats3))
  discountedCost1 + discountedCost2 + discountedCost3

/-- Theorem stating the total cost for the school --/
theorem school_seat_cost : 
  totalCostWithDiscounts 10 20 60 (12/100)
                         10 15 50 (10/100) (3/100)
                         5 10 40 (8/100) = 18947.50 := by
  sorry

end NUMINAMATH_CALUDE_school_seat_cost_l1832_183230


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1832_183274

theorem polynomial_factorization (x : ℝ) :
  x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1832_183274


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l1832_183254

/-- The magnitude of the complex number (2+4i)/(1+i) is √10 -/
theorem magnitude_of_complex_fraction :
  let z : ℂ := (2 + 4 * Complex.I) / (1 + Complex.I)
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l1832_183254


namespace NUMINAMATH_CALUDE_allocation_problem_l1832_183275

/-- The number of ways to allocate doctors and nurses to schools --/
def allocations (doctors nurses schools : ℕ) : ℕ :=
  (doctors.factorial * nurses.choose (2 * schools)) / (schools.factorial * (2 ^ schools))

/-- Theorem stating the number of allocations for the given problem --/
theorem allocation_problem :
  allocations 3 6 3 = 540 :=
by sorry

end NUMINAMATH_CALUDE_allocation_problem_l1832_183275


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1832_183249

theorem fraction_subtraction : 
  (2 + 6 + 8) / (1 + 2 + 3) - (1 + 2 + 3) / (2 + 6 + 8) = 55 / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1832_183249


namespace NUMINAMATH_CALUDE_tommy_makes_twelve_loaves_l1832_183276

/-- Represents the number of loaves Tommy can make given the flour costs and his budget --/
def tommys_loaves (flour_per_loaf : ℝ) (small_bag_weight : ℝ) (small_bag_cost : ℝ) 
  (large_bag_weight : ℝ) (large_bag_cost : ℝ) (budget : ℝ) : ℕ :=
  sorry

/-- Theorem stating that Tommy can make 12 loaves of bread --/
theorem tommy_makes_twelve_loaves :
  tommys_loaves 4 10 10 12 13 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_tommy_makes_twelve_loaves_l1832_183276


namespace NUMINAMATH_CALUDE_polyhedron_sum_l1832_183243

/-- Represents a convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  V : ℕ -- number of vertices
  E : ℕ -- number of edges
  F : ℕ -- number of faces
  q : ℕ -- number of quadrilateral faces
  h : ℕ -- number of hexagonal faces
  Q : ℕ -- number of quadrilateral faces meeting at each vertex
  H : ℕ -- number of hexagonal faces meeting at each vertex
  euler_formula : V - E + F = 2
  face_count : F = 24
  face_types : q + h = F
  edge_count : E = 2*q + 3*h
  vertex_degree : Q = 1 ∧ H = 1

/-- The main theorem to be proved -/
theorem polyhedron_sum (p : ConvexPolyhedron) : 100 * p.H + 10 * p.Q + p.V = 136 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_sum_l1832_183243


namespace NUMINAMATH_CALUDE_common_solution_y_value_l1832_183291

theorem common_solution_y_value : 
  ∃! y : ℝ, ∃ x : ℝ, (x^2 + y^2 - 4 = 0) ∧ (x^2 - 4*y + 8 = 0) ∧ (y = 2) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_y_value_l1832_183291


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l1832_183287

/-- Given a cubic polynomial x^3 + ax^2 + bx + c, returns the sum of its roots -/
def sumOfRoots (a b c : ℝ) : ℝ := -a

/-- Given a cubic polynomial x^3 + ax^2 + bx + c, returns the product of its roots -/
def productOfRoots (a b c : ℝ) : ℝ := -c

theorem cubic_roots_problem (p q r u v w : ℝ) :
  (∀ x, x^3 + 2*x^2 + 5*x - 8 = (x - p)*(x - q)*(x - r)) →
  (∀ x, x^3 + u*x^2 + v*x + w = (x - (p + q))*(x - (q + r))*(x - (r + p))) →
  w = 18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l1832_183287


namespace NUMINAMATH_CALUDE_conference_handshakes_l1832_183202

theorem conference_handshakes (total : ℕ) (group_a : ℕ) (group_b : ℕ) :
  total = 50 →
  group_a = 30 →
  group_b = 20 →
  group_a + group_b = total →
  (group_a * group_b) + (group_b * (group_b - 1) / 2) = 790 :=
by sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1832_183202


namespace NUMINAMATH_CALUDE_yard_trees_l1832_183260

/-- Calculates the number of trees in a yard given the yard length and distance between trees. -/
def num_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem stating that in a 325-meter yard with trees 13 meters apart, there are 26 trees. -/
theorem yard_trees : num_trees 325 13 = 26 := by
  sorry

end NUMINAMATH_CALUDE_yard_trees_l1832_183260


namespace NUMINAMATH_CALUDE_abs_cubic_inequality_l1832_183290

theorem abs_cubic_inequality (x : ℝ) : 
  |x| ≤ 2 → |3*x - x^3| ≤ 2 := by sorry

end NUMINAMATH_CALUDE_abs_cubic_inequality_l1832_183290


namespace NUMINAMATH_CALUDE_legs_walking_on_ground_l1832_183206

theorem legs_walking_on_ground (num_horses : ℕ) (num_men : ℕ) : 
  num_horses = 12 →
  num_men = num_horses →
  num_horses * 4 + (num_men / 2) * 2 = 60 :=
by sorry

end NUMINAMATH_CALUDE_legs_walking_on_ground_l1832_183206


namespace NUMINAMATH_CALUDE_smallest_a_value_l1832_183232

-- Define the polynomial
def polynomial (a b x : ℤ) : ℤ := x^3 - a*x^2 + b*x - 2310

-- Define the property of having three positive integer roots
def has_three_positive_integer_roots (a b : ℤ) : Prop :=
  ∃ (x y z : ℤ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    polynomial a b x = 0 ∧ polynomial a b y = 0 ∧ polynomial a b z = 0

-- State the theorem
theorem smallest_a_value :
  ∀ a b : ℤ, has_three_positive_integer_roots a b →
    (∀ a' b' : ℤ, has_three_positive_integer_roots a' b' → a ≤ a') →
    a = 88 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l1832_183232


namespace NUMINAMATH_CALUDE_product_of_large_integers_l1832_183262

theorem product_of_large_integers : ∃ (a b : ℤ), 
  a > 10^2009 ∧ b > 10^2009 ∧ a * b = 3^(4^5) + 4^(5^6) := by
  sorry

end NUMINAMATH_CALUDE_product_of_large_integers_l1832_183262


namespace NUMINAMATH_CALUDE_smallest_number_l1832_183267

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Represents the number 85 in base 9 --/
def num1 : List Nat := [8, 5]

/-- Represents the number 1000 in base 4 --/
def num2 : List Nat := [1, 0, 0, 0]

/-- Represents the number 111111 in base 2 --/
def num3 : List Nat := [1, 1, 1, 1, 1, 1]

theorem smallest_number :
  to_base_10 num3 2 ≤ to_base_10 num1 9 ∧
  to_base_10 num3 2 ≤ to_base_10 num2 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1832_183267


namespace NUMINAMATH_CALUDE_fibonacci_sum_quadruples_l1832_183252

/-- The n-th Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- A predicate that checks if a quadruple (a, b, c, d) satisfies the Fibonacci sum equation -/
def is_valid_quadruple (a b c d : ℕ) : Prop :=
  a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ fib a + fib b = fib c + fib d

/-- The set of all valid quadruples -/
def valid_quadruples : Set (ℕ × ℕ × ℕ × ℕ) :=
  {q | ∃ a b c d, q = (a, b, c, d) ∧ is_valid_quadruple a b c d}

/-- The set of solution quadruples -/
def solution_quadruples : Set (ℕ × ℕ × ℕ × ℕ) :=
  {q | ∃ a b,
    (q = (a, b, a, b) ∨ q = (a, b, b, a) ∨
     q = (a, a-3, a-1, a-1) ∨ q = (a-3, a, a-1, a-1) ∨
     q = (a-1, a-1, a, a-3) ∨ q = (a-1, a-1, a-3, a)) ∧
    a ≥ 2 ∧ b ≥ 2}

/-- The main theorem stating that the valid quadruples are exactly the solution quadruples -/
theorem fibonacci_sum_quadruples : valid_quadruples = solution_quadruples := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_sum_quadruples_l1832_183252


namespace NUMINAMATH_CALUDE_min_value_m_exists_l1832_183225

theorem min_value_m_exists (m : ℝ) : 
  (∃ x₀ : ℝ, |x₀ + 1| + |x₀ - 1| ≤ m) ↔ m ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_m_exists_l1832_183225


namespace NUMINAMATH_CALUDE_factorization_quadratic_l1832_183244

theorem factorization_quadratic (a x y : ℝ) : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_quadratic_l1832_183244


namespace NUMINAMATH_CALUDE_divisor_calculation_l1832_183227

theorem divisor_calculation (dividend quotient remainder : ℕ) (h1 : dividend = 76) (h2 : quotient = 4) (h3 : remainder = 8) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 17 := by
  sorry

end NUMINAMATH_CALUDE_divisor_calculation_l1832_183227


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l1832_183298

/-- Calculates the total weight of a zinc-copper mixture given the ratio and zinc weight -/
theorem zinc_copper_mixture_weight 
  (zinc_ratio : ℚ) 
  (copper_ratio : ℚ) 
  (zinc_weight : ℚ) 
  (h1 : zinc_ratio = 9) 
  (h2 : copper_ratio = 11) 
  (h3 : zinc_weight = 26.1) : 
  zinc_weight + (copper_ratio / zinc_ratio) * zinc_weight = 58 := by
sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l1832_183298


namespace NUMINAMATH_CALUDE_number_wall_solution_l1832_183294

/-- Represents a number wall with four layers --/
structure NumberWall :=
  (bottom_row : Fin 4 → ℕ)
  (second_row : Fin 3 → ℕ)
  (third_row : Fin 2 → ℕ)
  (top : ℕ)

/-- Checks if a number wall follows the addition rule --/
def is_valid_wall (wall : NumberWall) : Prop :=
  (∀ i : Fin 3, wall.second_row i = wall.bottom_row i + wall.bottom_row (i + 1)) ∧
  (∀ i : Fin 2, wall.third_row i = wall.second_row i + wall.second_row (i + 1)) ∧
  (wall.top = wall.third_row 0 + wall.third_row 1)

/-- The theorem to be proved --/
theorem number_wall_solution (m : ℕ) : 
  (∃ wall : NumberWall, 
    wall.bottom_row = ![m, 4, 10, 9] ∧ 
    wall.top = 52 ∧ 
    is_valid_wall wall) → 
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_solution_l1832_183294


namespace NUMINAMATH_CALUDE_two_numbers_puzzle_l1832_183238

def is_two_digit_same_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (n / 10 = n % 10)

def is_three_digit_same_digits (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100 = (n / 10) % 10) ∧ (n / 100 = n % 10)

theorem two_numbers_puzzle :
  ∀ a b : ℕ,
    a > 0 ∧ b > 0 →
    is_two_digit_same_digits (a + b) →
    is_three_digit_same_digits (a * b) →
    ((a = 37 ∧ b = 18) ∨ (a = 18 ∧ b = 37) ∨ (a = 74 ∧ b = 3) ∨ (a = 3 ∧ b = 74)) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_puzzle_l1832_183238


namespace NUMINAMATH_CALUDE_point_A_coordinates_l1832_183255

/-- A point in the second quadrant of the Cartesian coordinate system with coordinates dependent on an integer m -/
def point_A (m : ℤ) : ℝ × ℝ := (7 - 2*m, 5 - m)

/-- Predicate to check if a point is in the second quadrant -/
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- Theorem stating that if A(7-2m, 5-m) is in the second quadrant and m is an integer, then A(-1, 1) is the only solution -/
theorem point_A_coordinates : 
  ∃! m : ℤ, in_second_quadrant (point_A m) ∧ point_A m = (-1, 1) :=
sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l1832_183255


namespace NUMINAMATH_CALUDE_extremum_at_one_min_value_one_l1832_183292

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

-- Theorem 1: If f attains an extremum at x=1, then a = 1
theorem extremum_at_one (a : ℝ) (h : a > 0) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 1 :=
sorry

-- Theorem 2: If the minimum value of f is 1, then a ≥ 2
theorem min_value_one (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f a x ≥ 1) →
  a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_extremum_at_one_min_value_one_l1832_183292


namespace NUMINAMATH_CALUDE_sampling_plans_correct_l1832_183282

/-- Represents a canning factory with its production rate and operating hours. -/
structure CanningFactory where
  production_rate : ℕ  -- cans per hour
  operating_hours : ℕ

/-- Represents a sampling plan with the number of cans sampled and the interval between samples. -/
structure SamplingPlan where
  cans_per_sample : ℕ
  sample_interval : ℕ  -- in minutes

/-- Calculates the total number of cans sampled in a day given a factory and a sampling plan. -/
def total_sampled_cans (factory : CanningFactory) (plan : SamplingPlan) : ℕ :=
  (factory.operating_hours * 60 / plan.sample_interval) * plan.cans_per_sample

/-- Theorem stating that the given sampling plans result in the required number of sampled cans. -/
theorem sampling_plans_correct (factory : CanningFactory) :
  factory.production_rate = 120000 ∧ factory.operating_hours = 12 →
  (∃ plan1200 : SamplingPlan, total_sampled_cans factory plan1200 = 1200 ∧ 
    plan1200.cans_per_sample = 10 ∧ plan1200.sample_interval = 6) ∧
  (∃ plan980 : SamplingPlan, total_sampled_cans factory plan980 = 980 ∧ 
    plan980.cans_per_sample = 49 ∧ plan980.sample_interval = 36) := by
  sorry

end NUMINAMATH_CALUDE_sampling_plans_correct_l1832_183282


namespace NUMINAMATH_CALUDE_teacher_age_l1832_183263

/-- Proves that given a class of 50 students with an average age of 18 years, 
    if including the teacher's age changes the average to 19.5 years, 
    then the teacher's age is 94.5 years. -/
theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 50 →
  student_avg_age = 18 →
  new_avg_age = 19.5 →
  (num_students * student_avg_age + (new_avg_age * (num_students + 1) - num_students * student_avg_age)) = 94.5 :=
by
  sorry

#check teacher_age

end NUMINAMATH_CALUDE_teacher_age_l1832_183263


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1832_183270

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation (1-i)z = 2i
def equation (z : ℂ) : Prop := (1 - i) * z = 2 * i

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ second_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l1832_183270


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_165_l1832_183224

/-- Represents a 5-digit number in the form XX4XY -/
structure FiveDigitNumber where
  x : ℕ
  y : ℕ
  is_valid : x < 10 ∧ y < 10

/-- The 5-digit number as an integer -/
def FiveDigitNumber.to_int (n : FiveDigitNumber) : ℤ :=
  ↑(n.x * 10000 + n.x * 1000 + 400 + n.x * 10 + n.y)

theorem five_digit_divisible_by_165 (n : FiveDigitNumber) :
  n.to_int % 165 = 0 → n.x + n.y = 14 := by
  sorry


end NUMINAMATH_CALUDE_five_digit_divisible_by_165_l1832_183224


namespace NUMINAMATH_CALUDE_simplify_fraction_ratio_l1832_183297

theorem simplify_fraction_ratio (k : ℤ) : 
  ∃ (a b : ℤ), (4*k + 8) / 4 = a*k + b ∧ a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_ratio_l1832_183297


namespace NUMINAMATH_CALUDE_age_problem_l1832_183220

theorem age_problem (A B C : ℕ) : 
  A = B + 2 →
  B = 2 * C →
  A + B + C = 37 →
  B = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_age_problem_l1832_183220


namespace NUMINAMATH_CALUDE_P_greater_than_Q_l1832_183226

theorem P_greater_than_Q : 
  let P : ℝ := Real.sqrt 7 - 1
  let Q : ℝ := Real.sqrt 11 - Real.sqrt 5
  P > Q := by sorry

end NUMINAMATH_CALUDE_P_greater_than_Q_l1832_183226


namespace NUMINAMATH_CALUDE_octahedron_triangles_l1832_183283

/-- The number of vertices in a regular octahedron -/
def octahedron_vertices : ℕ := 8

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles that can be constructed by connecting three different vertices of a regular octahedron -/
def distinct_triangles : ℕ := Nat.choose octahedron_vertices triangle_vertices

theorem octahedron_triangles : distinct_triangles = 56 := by sorry

end NUMINAMATH_CALUDE_octahedron_triangles_l1832_183283


namespace NUMINAMATH_CALUDE_triangle_area_is_two_l1832_183231

/-- The area of the triangle formed by the line x + y - 2 = 0 and the coordinate axes -/
def triangle_area : ℝ := 2

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x + y - 2 = 0

theorem triangle_area_is_two :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_equation x₁ y₁ ∧
    line_equation x₂ y₂ ∧
    x₁ = 0 ∧ y₂ = 0 ∧
    (1/2 : ℝ) * x₂ * y₁ = triangle_area :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_two_l1832_183231


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l1832_183208

/-- Given two planar vectors a and b, prove that the cosine of the angle between them is -3/5. -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) : 
  a = (2, 4) → a - 2 • b = (0, 8) → 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -3/5 := by
  sorry

#check cosine_of_angle_between_vectors

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l1832_183208


namespace NUMINAMATH_CALUDE_pauls_homework_average_l1832_183248

/-- Represents the homework schedule for Paul --/
structure HomeworkSchedule where
  weeknight_hours : ℕ
  weekend_hours : ℕ
  practice_nights : ℕ
  total_nights : ℕ

/-- Calculates the average homework hours per available night --/
def average_homework_hours (schedule : HomeworkSchedule) : ℚ :=
  let total_homework := schedule.weeknight_hours * (schedule.total_nights - 2) + schedule.weekend_hours
  let available_nights := schedule.total_nights - schedule.practice_nights
  (total_homework : ℚ) / available_nights

/-- Theorem stating that Paul's average homework hours per available night is 3 --/
theorem pauls_homework_average (pauls_schedule : HomeworkSchedule) 
  (h1 : pauls_schedule.weeknight_hours = 2)
  (h2 : pauls_schedule.weekend_hours = 5)
  (h3 : pauls_schedule.practice_nights = 2)
  (h4 : pauls_schedule.total_nights = 7) :
  average_homework_hours pauls_schedule = 3 := by
  sorry


end NUMINAMATH_CALUDE_pauls_homework_average_l1832_183248


namespace NUMINAMATH_CALUDE_expected_score_is_correct_l1832_183201

/-- The expected score for a round in the basketball shooting game. -/
def expected_score : ℝ := 6

/-- The probability of making a shot. -/
def shot_probability : ℝ := 0.5

/-- The score for making the first shot. -/
def first_shot_score : ℕ := 8

/-- The score for making the second shot (after missing the first). -/
def second_shot_score : ℕ := 6

/-- The score for making the third shot (after missing the first two). -/
def third_shot_score : ℕ := 4

/-- The score for missing all three shots. -/
def miss_all_score : ℕ := 0

/-- Theorem stating that the expected score is correct given the game rules. -/
theorem expected_score_is_correct :
  expected_score = 
    shot_probability * first_shot_score +
    (1 - shot_probability) * shot_probability * second_shot_score +
    (1 - shot_probability) * (1 - shot_probability) * shot_probability * third_shot_score +
    (1 - shot_probability) * (1 - shot_probability) * (1 - shot_probability) * miss_all_score :=
by sorry

end NUMINAMATH_CALUDE_expected_score_is_correct_l1832_183201


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_7_and_4_l1832_183246

theorem smallest_common_multiple_of_7_and_4 : ∃ (n : ℕ), n > 0 ∧ n % 7 = 0 ∧ n % 4 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 7 = 0 ∧ m % 4 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_7_and_4_l1832_183246


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l1832_183233

theorem simplify_and_ratio (k : ℚ) : ∃ (a b : ℚ), 
  (6 * k + 12) / 3 = a * k + b ∧ a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l1832_183233


namespace NUMINAMATH_CALUDE_product_in_geometric_sequence_l1832_183223

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_in_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 18 = -15) →
  (a 2 * a 18 = 16) →
  a 3 * a 10 * a 17 = -64 := by
  sorry

end NUMINAMATH_CALUDE_product_in_geometric_sequence_l1832_183223


namespace NUMINAMATH_CALUDE_sum_using_splitting_terms_l1832_183279

/-- The sum of (-2017⅔) + 2016¾ + (-2015⅚) + 16½ using the method of splitting terms -/
theorem sum_using_splitting_terms :
  (-2017 - 2/3) + (2016 + 3/4) + (-2015 - 5/6) + (16 + 1/2) = -2000 - 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_using_splitting_terms_l1832_183279


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l1832_183265

theorem intersection_implies_sum (a b : ℝ) : 
  let A : Set ℝ := {3, 2^a}
  let B : Set ℝ := {a, b}
  A ∩ B = {2} → a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l1832_183265


namespace NUMINAMATH_CALUDE_least_whole_number_ratio_l1832_183242

theorem least_whole_number_ratio (x : ℕ) : x ≥ 3 ↔ (6 - x : ℚ) / (7 - x) < 16 / 21 :=
sorry

end NUMINAMATH_CALUDE_least_whole_number_ratio_l1832_183242


namespace NUMINAMATH_CALUDE_pumpkin_pie_cost_pumpkin_pie_cost_proof_l1832_183269

/-- The cost to make a pumpkin pie given the following conditions:
  * 10 pumpkin pies and 12 cherry pies are made
  * Cherry pies cost $5 each to make
  * The total profit is $20
  * Each pie is sold for $5
-/
theorem pumpkin_pie_cost : ℝ :=
  let num_pumpkin_pies : ℕ := 10
  let num_cherry_pies : ℕ := 12
  let cherry_pie_cost : ℝ := 5
  let profit : ℝ := 20
  let selling_price : ℝ := 5
  3

/-- Proof that the cost to make each pumpkin pie is $3 -/
theorem pumpkin_pie_cost_proof :
  let num_pumpkin_pies : ℕ := 10
  let num_cherry_pies : ℕ := 12
  let cherry_pie_cost : ℝ := 5
  let profit : ℝ := 20
  let selling_price : ℝ := 5
  pumpkin_pie_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_pie_cost_pumpkin_pie_cost_proof_l1832_183269


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1832_183219

-- Define the complex number z
def z : ℂ := Complex.I * (2 + Complex.I)

-- Theorem stating that the imaginary part of z is 2
theorem imaginary_part_of_z : z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1832_183219


namespace NUMINAMATH_CALUDE_largest_B_181_l1832_183221

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The sequence B_k as defined in the problem -/
def B (k : ℕ) : ℝ := (binomial 2000 k : ℝ) * (0.1 ^ k)

/-- Theorem stating that B_181 is the largest among all B_k -/
theorem largest_B_181 : ∀ k : ℕ, k ≤ 2000 → B 181 ≥ B k := by sorry

end NUMINAMATH_CALUDE_largest_B_181_l1832_183221


namespace NUMINAMATH_CALUDE_increase_decrease_theorem_l1832_183211

theorem increase_decrease_theorem (k r s N : ℝ) 
  (hk : k > 0) (hr : r > 0) (hs : s > 0) (hN : N > 0) (hr_bound : r < 80) :
  N * (1 + k / 100) * (1 - r / 100) + 10 * s > N ↔ k > 100 * r / (100 - r) := by
sorry

end NUMINAMATH_CALUDE_increase_decrease_theorem_l1832_183211


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1832_183228

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_nonzero_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ 0 ∧ tens ≠ 0 ∧ ones ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem unique_three_digit_number :
  ∃! n : ℕ, is_three_digit n ∧ has_nonzero_digits n ∧ 222 * (sum_of_digits n) - n = 1990 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1832_183228


namespace NUMINAMATH_CALUDE_cost_of_scooter_l1832_183257

def scooter_cost (megan_money tara_money : ℕ) : Prop :=
  (tara_money = megan_money + 4) ∧
  (tara_money = 15) ∧
  (megan_money + tara_money = 26)

theorem cost_of_scooter :
  ∃ (megan_money tara_money : ℕ), scooter_cost megan_money tara_money :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_scooter_l1832_183257


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1832_183222

/-- The distance between the foci of the hyperbola x^2 - 6x - 4y^2 - 8y = 27 is 4√10 -/
theorem hyperbola_foci_distance :
  ∃ (a b c : ℝ),
    (∀ x y : ℝ, x^2 - 6*x - 4*y^2 - 8*y = 27 ↔ (x - 3)^2 / a^2 - (y + 1)^2 / b^2 = 1) ∧
    c^2 = a^2 + b^2 ∧
    2*c = 4 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1832_183222


namespace NUMINAMATH_CALUDE_product_ab_equals_negative_one_l1832_183281

theorem product_ab_equals_negative_one (a b : ℝ) 
  (h : ∀ x : ℝ, x ≥ 0 → 0 ≤ x^4 - x^3 + a*x + b ∧ x^4 - x^3 + a*x + b ≤ (x^2 - 1)^2) : 
  a * b = -1 := by
sorry

end NUMINAMATH_CALUDE_product_ab_equals_negative_one_l1832_183281


namespace NUMINAMATH_CALUDE_negation_equivalence_l1832_183229

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x > 0 ∧ -x^2 + 2*x - 1 > 0) ↔ 
  (∀ x : ℝ, x > 0 → -x^2 + 2*x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1832_183229


namespace NUMINAMATH_CALUDE_fred_card_purchase_l1832_183261

/-- The number of packs of football cards Fred bought -/
def football_packs : ℕ := 2

/-- The cost of one pack of football cards -/
def football_cost : ℚ := 273/100

/-- The cost of the pack of Pokemon cards -/
def pokemon_cost : ℚ := 401/100

/-- The cost of the deck of baseball cards -/
def baseball_cost : ℚ := 895/100

/-- The total amount Fred spent on cards -/
def total_spent : ℚ := 1842/100

theorem fred_card_purchase :
  (football_packs : ℚ) * football_cost + pokemon_cost + baseball_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_fred_card_purchase_l1832_183261


namespace NUMINAMATH_CALUDE_club_officer_selection_l1832_183250

/-- Represents the number of ways to choose club officers under specific conditions -/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  2 * boys * girls * (boys - 1)

/-- Theorem stating the number of ways to choose club officers -/
theorem club_officer_selection :
  let total_members : ℕ := 30
  let boys : ℕ := 15
  let girls : ℕ := 15
  choose_officers total_members boys girls = 6300 := by
  sorry

#eval choose_officers 30 15 15

end NUMINAMATH_CALUDE_club_officer_selection_l1832_183250


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l1832_183213

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_P_intersect_Q :
  (Set.univ \ P) ∩ Q = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l1832_183213


namespace NUMINAMATH_CALUDE_purchase_cost_l1832_183284

theorem purchase_cost (pretzel_cost : ℝ) (chip_cost_percentage : ℝ) : 
  pretzel_cost = 4 →
  chip_cost_percentage = 175 →
  2 * pretzel_cost + 2 * (pretzel_cost * chip_cost_percentage / 100) = 22 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l1832_183284


namespace NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l1832_183214

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio :=
  (flour : ℕ)
  (water : ℕ)
  (sugar : ℕ)

/-- Calculates the new ratio based on the original ratio -/
def new_ratio (original : RecipeRatio) : RecipeRatio :=
  { flour := original.flour,
    water := original.flour / 2,
    sugar := original.sugar * 2 }

/-- Calculates the amount of an ingredient based on the ratio and a known amount -/
def calculate_amount (ratio : RecipeRatio) (known_part : ℕ) (known_amount : ℚ) (target_part : ℕ) : ℚ :=
  (known_amount * target_part) / known_part

theorem sugar_amount_in_new_recipe : 
  let original_ratio := RecipeRatio.mk 8 4 3
  let new_ratio := new_ratio original_ratio
  let water_amount : ℚ := 2
  calculate_amount new_ratio new_ratio.water water_amount new_ratio.sugar = 3 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l1832_183214


namespace NUMINAMATH_CALUDE_system_solution_l1832_183239

theorem system_solution :
  let S : Set (ℝ × ℝ × ℝ) := {(x, y, z) | 
    x + y - z = 4 ∧
    x^2 - y^2 + z^2 = -4 ∧
    x * y * z = 6}
  S = {(2, 3, 1), (-1, 3, -2)} := by sorry

end NUMINAMATH_CALUDE_system_solution_l1832_183239


namespace NUMINAMATH_CALUDE_not_perfect_square_p_squared_plus_q_power_l1832_183280

theorem not_perfect_square_p_squared_plus_q_power (p q : ℕ) (hp : Prime p) (hq : Prime q) (h_perfect_square : ∃ a : ℕ, p + q^2 = a^2) :
  ∀ n : ℕ, ¬∃ b : ℕ, p^2 + q^n = b^2 :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_p_squared_plus_q_power_l1832_183280


namespace NUMINAMATH_CALUDE_stating_holiday_lodge_assignments_l1832_183278

/-- Represents the number of rooms in the holiday lodge -/
def num_rooms : ℕ := 4

/-- Represents the number of friends staying at the lodge -/
def num_friends : ℕ := 6

/-- Represents the maximum number of friends allowed per room -/
def max_friends_per_room : ℕ := 2

/-- Represents the minimum number of empty rooms required -/
def min_empty_rooms : ℕ := 1

/-- 
Calculates the number of ways to assign friends to rooms 
given the constraints
-/
def num_assignments (n_rooms : ℕ) (n_friends : ℕ) 
  (max_per_room : ℕ) (min_empty : ℕ) : ℕ := 
  sorry

/-- 
Theorem stating that the number of assignments for the given problem is 1080
-/
theorem holiday_lodge_assignments : 
  num_assignments num_rooms num_friends max_friends_per_room min_empty_rooms = 1080 := by
  sorry

end NUMINAMATH_CALUDE_stating_holiday_lodge_assignments_l1832_183278


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1832_183256

theorem diophantine_equation_solution (k : ℕ+) : 
  (∃ (x y : ℕ+), x^2 + y^2 = k * x * y - 1) ↔ k = 3 := by
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1832_183256


namespace NUMINAMATH_CALUDE_total_amount_is_200_l1832_183241

/-- Represents the distribution of money among four individuals -/
structure MoneyDistribution where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- The total amount of money distributed -/
def total_amount (d : MoneyDistribution) : ℝ :=
  d.w + d.x + d.y + d.z

/-- Theorem stating the total amount given the conditions -/
theorem total_amount_is_200 (d : MoneyDistribution) 
  (h1 : d.x = 0.75 * d.w)
  (h2 : d.y = 0.45 * d.w)
  (h3 : d.z = 0.30 * d.w)
  (h4 : d.y = 36) :
  total_amount d = 200 := by
  sorry

#check total_amount_is_200

end NUMINAMATH_CALUDE_total_amount_is_200_l1832_183241


namespace NUMINAMATH_CALUDE_locus_empty_near_origin_l1832_183268

/-- Represents a polynomial of degree 3 in two variables -/
structure Polynomial3 (α : Type*) [Ring α] where
  A : α
  B : α
  C : α
  D : α
  E : α
  F : α
  G : α

/-- Evaluates the polynomial at a given point (x, y) -/
def eval_poly (p : Polynomial3 ℝ) (x y : ℝ) : ℝ :=
  p.A * x^2 + p.B * x * y + p.C * y^2 + p.D * x^3 + p.E * x^2 * y + p.F * x * y^2 + p.G * y^3

theorem locus_empty_near_origin (p : Polynomial3 ℝ) (h : p.B^2 - 4 * p.A * p.C < 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x y : ℝ, 0 < x^2 + y^2 ∧ x^2 + y^2 < δ^2 → eval_poly p x y ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_locus_empty_near_origin_l1832_183268


namespace NUMINAMATH_CALUDE_distance_between_points_l1832_183203

theorem distance_between_points (x : ℝ) :
  (x - 2)^2 + (5 - 5)^2 = 5^2 → x = -3 ∨ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1832_183203


namespace NUMINAMATH_CALUDE_path_result_l1832_183259

def move_north (x : ℚ) : ℚ := x + 7
def move_east (x : ℚ) : ℚ := x - 4
def move_south (x : ℚ) : ℚ := x / 2
def move_west (x : ℚ) : ℚ := x * 3

def path (x : ℚ) : ℚ :=
  move_north (move_east (move_south (move_west (move_west (move_south (move_east (move_north x)))))))

theorem path_result : path 21 = 57 := by
  sorry

end NUMINAMATH_CALUDE_path_result_l1832_183259


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1832_183240

-- Define the ellipse type
structure Ellipse where
  endpoints : List (ℝ × ℝ)
  axes_perpendicular : Bool

-- Define the function to calculate the distance between foci
noncomputable def distance_between_foci (e : Ellipse) : ℝ :=
  sorry

-- Theorem statement
theorem ellipse_foci_distance 
  (e : Ellipse) 
  (h1 : e.endpoints = [(1, 3), (7, -5), (1, -5)])
  (h2 : e.axes_perpendicular = true) : 
  distance_between_foci e = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1832_183240


namespace NUMINAMATH_CALUDE_cookies_per_bag_l1832_183271

/-- Proves that the number of cookies in each bag is 20, given the conditions of the problem. -/
theorem cookies_per_bag (bags_per_box : ℕ) (total_calories : ℕ) (calories_per_cookie : ℕ)
  (h1 : bags_per_box = 4)
  (h2 : total_calories = 1600)
  (h3 : calories_per_cookie = 20) :
  total_calories / (bags_per_box * calories_per_cookie) = 20 :=
by sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l1832_183271


namespace NUMINAMATH_CALUDE_triangle_ratio_l1832_183288

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 3 * a →
  b / a = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l1832_183288


namespace NUMINAMATH_CALUDE_repair_shop_earnings_121_l1832_183289

/-- Represents the earnings for a repair shop for a week. -/
def repair_shop_earnings (phone_cost laptop_cost computer_cost : ℕ) 
  (phone_repairs laptop_repairs computer_repairs : ℕ) : ℕ :=
  phone_cost * phone_repairs + laptop_cost * laptop_repairs + computer_cost * computer_repairs

/-- Theorem stating that the repair shop's earnings for the week is $121. -/
theorem repair_shop_earnings_121 :
  repair_shop_earnings 11 15 18 5 2 2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_repair_shop_earnings_121_l1832_183289


namespace NUMINAMATH_CALUDE_fourth_quadrant_condition_l1832_183251

def complex_number (b : ℝ) : ℂ := (1 + b * Complex.I) * (2 + Complex.I)

def in_fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

theorem fourth_quadrant_condition (b : ℝ) : 
  in_fourth_quadrant (complex_number b) ↔ b < -1/2 := by sorry

end NUMINAMATH_CALUDE_fourth_quadrant_condition_l1832_183251


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l1832_183272

/-- The reciprocal of the common fraction form of 0.353535... is 99/35 -/
theorem reciprocal_of_repeating_decimal :
  let x : ℚ := 35 / 99  -- Common fraction form of 0.353535...
  (1 : ℚ) / x = 99 / 35 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l1832_183272


namespace NUMINAMATH_CALUDE_toy_boxes_theorem_l1832_183216

/-- Represents the number of toy cars in each box -/
structure ToyBoxes :=
  (box1 : ℕ)
  (box2 : ℕ)
  (box3 : ℕ)
  (box4 : ℕ)
  (box5 : ℕ)

/-- The initial state of the toy boxes -/
def initial_state : ToyBoxes :=
  { box1 := 21
  , box2 := 31
  , box3 := 19
  , box4 := 45
  , box5 := 27 }

/-- The final state of the toy boxes after moving 12 cars from box1 to box4 -/
def final_state : ToyBoxes :=
  { box1 := 9
  , box2 := 31
  , box3 := 19
  , box4 := 57
  , box5 := 27 }

/-- The number of cars moved from box1 to box4 -/
def cars_moved : ℕ := 12

theorem toy_boxes_theorem (initial : ToyBoxes) (final : ToyBoxes) (moved : ℕ) :
  initial = initial_state →
  moved = cars_moved →
  final.box1 = initial.box1 - moved ∧
  final.box2 = initial.box2 ∧
  final.box3 = initial.box3 ∧
  final.box4 = initial.box4 + moved ∧
  final.box5 = initial.box5 →
  final = final_state :=
by sorry

end NUMINAMATH_CALUDE_toy_boxes_theorem_l1832_183216


namespace NUMINAMATH_CALUDE_unique_solution_iff_b_eq_two_or_six_l1832_183210

/-- The function g(x) = x^2 + bx + 2b -/
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 2*b

/-- The statement that |g(x)| ≤ 3 has exactly one solution -/
def has_unique_solution (b : ℝ) : Prop :=
  ∃! x, |g b x| ≤ 3

/-- Theorem: The inequality |x^2 + bx + 2b| ≤ 3 has exactly one solution
    if and only if b = 2 or b = 6 -/
theorem unique_solution_iff_b_eq_two_or_six :
  ∀ b : ℝ, has_unique_solution b ↔ (b = 2 ∨ b = 6) := by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_b_eq_two_or_six_l1832_183210


namespace NUMINAMATH_CALUDE_complex_fraction_square_l1832_183296

theorem complex_fraction_square (m n : ℝ) (h : m * (1 + Complex.I) = 1 + n * Complex.I) :
  ((m + n * Complex.I) / (m - n * Complex.I))^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_square_l1832_183296


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1832_183285

theorem quadratic_equation_roots : 
  let f : ℝ → ℝ := λ x => x^2 - 1 - 3
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1832_183285


namespace NUMINAMATH_CALUDE_inscribed_circle_equation_l1832_183245

-- Define the line
def line (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define points A and B
def point_A : ℝ × ℝ := (4, 0)
def point_B : ℝ × ℝ := (0, 3)

-- Define origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the inscribed circle equation
def is_inscribed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Theorem statement
theorem inscribed_circle_equation :
  ∀ x y : ℝ,
  is_inscribed_circle x y ↔
  (∃ r : ℝ, r > 0 ∧
    (x - r)^2 + (y - r)^2 = r^2 ∧
    (x - 4)^2 + y^2 = r^2 ∧
    x^2 + (y - 3)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_equation_l1832_183245


namespace NUMINAMATH_CALUDE_equal_spending_dolls_l1832_183205

/-- The number of sisters Tonya is buying gifts for -/
def num_sisters : ℕ := 2

/-- The cost of each doll in dollars -/
def doll_cost : ℕ := 15

/-- The cost of each lego set in dollars -/
def lego_cost : ℕ := 20

/-- The number of lego sets bought for the older sister -/
def num_lego_sets : ℕ := 3

/-- The total amount spent on the older sister in dollars -/
def older_sister_cost : ℕ := num_lego_sets * lego_cost

/-- The number of dolls bought for the younger sister -/
def num_dolls : ℕ := older_sister_cost / doll_cost

theorem equal_spending_dolls : num_dolls = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_spending_dolls_l1832_183205


namespace NUMINAMATH_CALUDE_system_solution_l1832_183273

/-- Given a system of equations, prove that t = 24 -/
theorem system_solution (p t j x y a b c : ℝ) 
  (h1 : j = 0.75 * p)
  (h2 : j = 0.8 * t)
  (h3 : t = p - (t / 100) * p)
  (h4 : x = 0.1 * t)
  (h5 : y = 0.5 * j)
  (h6 : x + y = 12)
  (h7 : a = x + y)
  (h8 : b = 0.15 * a)
  (h9 : c = 2 * b) :
  t = 24 := by sorry

end NUMINAMATH_CALUDE_system_solution_l1832_183273


namespace NUMINAMATH_CALUDE_line_problem_l1832_183236

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem line_problem (m n : ℝ) :
  let l1 : Line := ⟨2, 2, -1⟩
  let l2 : Line := ⟨4, n, 3⟩
  let l3 : Line := ⟨m, 6, 1⟩
  are_parallel l1 l2 ∧ are_perpendicular l1 l3 → m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_problem_l1832_183236


namespace NUMINAMATH_CALUDE_souvenir_shop_optimal_solution_souvenir_shop_max_profit_l1832_183295

/-- Represents the cost and profit structure for souvenir types A and B -/
structure SouvenirShop where
  cost_A : ℝ
  cost_B : ℝ
  profit_A : ℝ
  profit_B : ℝ

/-- Theorem stating the optimal solution for the souvenir shop problem -/
theorem souvenir_shop_optimal_solution (shop : SouvenirShop) 
  (h1 : 7 * shop.cost_A + 8 * shop.cost_B = 380)
  (h2 : 10 * shop.cost_A + 6 * shop.cost_B = 380)
  (h3 : shop.profit_A = 5)
  (h4 : shop.profit_B = 7) : 
  (shop.cost_A = 20 ∧ shop.cost_B = 30) ∧ 
  (∀ a b : ℕ, a + b = 40 → a * shop.cost_A + b * shop.cost_B ≤ 900 → 
    a * shop.profit_A + b * shop.profit_B ≥ 216 → 
    a * shop.profit_A + b * shop.profit_B ≤ 30 * shop.profit_A + 10 * shop.profit_B) :=
sorry

/-- Corollary stating the maximum profit -/
theorem souvenir_shop_max_profit (shop : SouvenirShop) 
  (h : shop.cost_A = 20 ∧ shop.cost_B = 30 ∧ shop.profit_A = 5 ∧ shop.profit_B = 7) :
  30 * shop.profit_A + 10 * shop.profit_B = 220 :=
sorry

end NUMINAMATH_CALUDE_souvenir_shop_optimal_solution_souvenir_shop_max_profit_l1832_183295
