import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l3289_328900

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3289_328900


namespace NUMINAMATH_CALUDE_max_perfect_matchings_20gon_l3289_328948

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Maximum number of perfect matchings for a 2n-gon -/
def max_perfect_matchings (n : ℕ) : ℕ := fib n

/-- A convex polygon -/
structure ConvexPolygon where
  sides : ℕ
  convex : sides > 2

/-- A triangulation of a convex polygon -/
structure Triangulation (p : ConvexPolygon) where
  diagonals : ℕ
  triangles : ℕ
  valid : diagonals = p.sides - 3 ∧ triangles = p.sides - 2

/-- A perfect matching in a triangulation -/
structure PerfectMatching (t : Triangulation p) where
  edges : ℕ
  valid : edges = p.sides / 2

/-- Theorem: Maximum number of perfect matchings for a 20-gon -/
theorem max_perfect_matchings_20gon (p : ConvexPolygon) 
    (h : p.sides = 20) : 
    (∀ t : Triangulation p, ∀ m : PerfectMatching t, 
      ∃ n : ℕ, n ≤ max_perfect_matchings 10) ∧ 
    (∃ t : Triangulation p, ∃ m : PerfectMatching t, 
      max_perfect_matchings 10 = 55) :=
  sorry

end NUMINAMATH_CALUDE_max_perfect_matchings_20gon_l3289_328948


namespace NUMINAMATH_CALUDE_don_profit_l3289_328980

/-- Represents a person's rose bundle transaction -/
structure Transaction where
  bought : ℕ
  sold : ℕ
  profit : ℚ

/-- Represents the prices of rose bundles -/
structure Prices where
  buy : ℚ
  sell : ℚ

/-- The main theorem -/
theorem don_profit 
  (jamie : Transaction)
  (linda : Transaction)
  (don : Transaction)
  (prices : Prices)
  (h1 : jamie.bought = 20)
  (h2 : jamie.sold = 15)
  (h3 : jamie.profit = 60)
  (h4 : linda.bought = 34)
  (h5 : linda.sold = 24)
  (h6 : linda.profit = 69)
  (h7 : don.bought = 40)
  (h8 : don.sold = 36)
  (h9 : prices.sell > prices.buy)
  (h10 : jamie.profit = jamie.sold * prices.sell - jamie.bought * prices.buy)
  (h11 : linda.profit = linda.sold * prices.sell - linda.bought * prices.buy)
  (h12 : don.profit = don.sold * prices.sell - don.bought * prices.buy) :
  don.profit = 252 := by sorry

end NUMINAMATH_CALUDE_don_profit_l3289_328980


namespace NUMINAMATH_CALUDE_justin_tim_games_l3289_328995

theorem justin_tim_games (n : ℕ) (h : n = 12) :
  let total_combinations := Nat.choose n 6
  let games_with_both := Nat.choose (n - 2) 4
  games_with_both = 210 ∧ 
  2 * games_with_both = total_combinations := by
  sorry

end NUMINAMATH_CALUDE_justin_tim_games_l3289_328995


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3289_328902

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geometric : IsGeometricSequence a)
  (h_fourth : a 4 = 16)
  (h_ninth : a 9 = 2) :
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3289_328902


namespace NUMINAMATH_CALUDE_longest_collection_pages_l3289_328928

/-- Represents a book collection --/
structure Collection where
  inches_per_page : ℚ
  total_inches : ℚ

/-- Calculates the total number of pages in a collection --/
def total_pages (c : Collection) : ℚ :=
  c.total_inches / c.inches_per_page

theorem longest_collection_pages (miles daphne : Collection)
  (h1 : miles.inches_per_page = 1/5)
  (h2 : daphne.inches_per_page = 1/50)
  (h3 : miles.total_inches = 240)
  (h4 : daphne.total_inches = 25) :
  max (total_pages miles) (total_pages daphne) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_longest_collection_pages_l3289_328928


namespace NUMINAMATH_CALUDE_root_in_interval_l3289_328938

def f (x : ℝ) := x^5 + x - 3

theorem root_in_interval :
  (f 1 < 0) → (f 2 > 0) → ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l3289_328938


namespace NUMINAMATH_CALUDE_range_of_fraction_l3289_328903

def f (a b c x : ℝ) : ℝ := 3 * a * x^2 - 2 * b * x + c

theorem range_of_fraction (a b c : ℝ) :
  (a - b + c = 0) →
  (f a b c 0 > 0) →
  (f a b c 1 > 0) →
  ∃ (y : ℝ), (4/3 < y ∧ y < 7/2 ∧ y = (a + 3*b + 7*c) / (2*a + b)) ∧
  ∀ (z : ℝ), (z = (a + 3*b + 7*c) / (2*a + b)) → (4/3 < z ∧ z < 7/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l3289_328903


namespace NUMINAMATH_CALUDE_animus_tower_beavers_l3289_328988

/-- The number of beavers hired for the Animus Tower project -/
def num_beavers : ℕ := 862 - 544

/-- The total number of workers hired for the Animus Tower project -/
def total_workers : ℕ := 862

/-- The number of spiders hired for the Animus Tower project -/
def num_spiders : ℕ := 544

theorem animus_tower_beavers : num_beavers = 318 := by
  sorry

end NUMINAMATH_CALUDE_animus_tower_beavers_l3289_328988


namespace NUMINAMATH_CALUDE_senior_junior_ratio_l3289_328915

theorem senior_junior_ratio (j s : ℕ) (hj : j > 0) (hs : s > 0) : 
  (3 * j : ℚ) / 4 = (1 * s : ℚ) / 2 → (s : ℚ) / j = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_senior_junior_ratio_l3289_328915


namespace NUMINAMATH_CALUDE_disjoint_subsets_sum_theorem_l3289_328983

theorem disjoint_subsets_sum_theorem (S : Set ℕ) (M₁ M₂ M₃ : Set ℕ) 
  (h1 : M₁ ⊆ S) (h2 : M₂ ⊆ S) (h3 : M₃ ⊆ S)
  (h4 : M₁ ∩ M₂ = ∅) (h5 : M₁ ∩ M₃ = ∅) (h6 : M₂ ∩ M₃ = ∅) :
  ∃ (X Y : ℕ), (X ∈ M₁ ∧ Y ∈ M₂) ∨ (X ∈ M₁ ∧ Y ∈ M₃) ∨ (X ∈ M₂ ∧ Y ∈ M₃) ∧ 
    (X + Y ∉ M₁ ∨ X + Y ∉ M₂ ∨ X + Y ∉ M₃) :=
by sorry

end NUMINAMATH_CALUDE_disjoint_subsets_sum_theorem_l3289_328983


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2005_l3289_328930

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_2005 :
  ∃ n : ℕ, arithmetic_sequence 1 3 n = 2005 ∧ n = 669 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2005_l3289_328930


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3289_328998

/-- The quadratic equation x^2 + 4x - 4 = 0 has two distinct real roots -/
theorem quadratic_two_distinct_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ - 4 = 0 ∧ x₂^2 + 4*x₂ - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3289_328998


namespace NUMINAMATH_CALUDE_inequality_solution_and_function_property_l3289_328914

def f (x : ℝ) : ℝ := |x + 1|

theorem inequality_solution_and_function_property :
  (∀ x : ℝ, f (x + 8) ≥ 10 - f x ↔ x ≤ -10 ∨ x ≥ 0) ∧
  (∀ x y : ℝ, |x| > 1 → |y| < 1 → f y < |x| * f (y / x^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_function_property_l3289_328914


namespace NUMINAMATH_CALUDE_distance_ratio_theorem_l3289_328931

theorem distance_ratio_theorem (x : ℝ) (h1 : x^2 + (-4)^2 = 8^2) :
  |(-4)| / 8 = 1/2 := by sorry

end NUMINAMATH_CALUDE_distance_ratio_theorem_l3289_328931


namespace NUMINAMATH_CALUDE_sum_of_four_sequential_terms_l3289_328927

theorem sum_of_four_sequential_terms (n : ℝ) : 
  n + (n + 1) + (n + 2) + (n + 3) = 20 → n = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_sequential_terms_l3289_328927


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3289_328985

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 4

-- State the theorem
theorem quadratic_roots_range (a b : ℝ) : 
  a > 0 → 
  (∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) →
  (∃ r : ℝ, 1 < r ∧ r < 2 ∧ f a b r = 0) →
  ∀ s : ℝ, s < 4 ↔ ∃ t : ℝ, a + b = t ∧ t < 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3289_328985


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3289_328945

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3289_328945


namespace NUMINAMATH_CALUDE_valid_perm_count_l3289_328918

/-- 
Given a permutation π of n distinct elements, we define:
inv_count(π, i) = number of elements to the left of π(i) that are greater than π(i) +
                  number of elements to the right of π(i) that are less than π(i)
-/
def inv_count (π : Fin n → Fin n) (i : Fin n) : ℕ := sorry

/-- A permutation is valid if inv_count is even for all elements -/
def is_valid_perm (π : Fin n → Fin n) : Prop :=
  ∀ i, Even (inv_count π i)

/-- The number of valid permutations -/
def count_valid_perms (n : ℕ) : ℕ := sorry

theorem valid_perm_count (n : ℕ) : 
  count_valid_perms n = (Nat.factorial (n / 2)) * (Nat.factorial ((n + 1) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_valid_perm_count_l3289_328918


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l3289_328925

theorem square_perimeter_problem (area_A : ℝ) (prob_not_in_B : ℝ) : 
  area_A = 30 →
  prob_not_in_B = 0.4666666666666667 →
  let area_B := area_A * (1 - prob_not_in_B)
  let side_B := Real.sqrt area_B
  let perimeter_B := 4 * side_B
  perimeter_B = 16 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l3289_328925


namespace NUMINAMATH_CALUDE_square_root_equation_l3289_328913

theorem square_root_equation (x : ℝ) : Real.sqrt (x + 4) = 12 → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l3289_328913


namespace NUMINAMATH_CALUDE_symmetric_circle_correct_l3289_328955

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 11 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-2, 1)

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 5)^2 + y^2 = 16

-- Theorem statement
theorem symmetric_circle_correct :
  ∀ (x y : ℝ),
  symmetric_circle x y ↔
  ∃ (x₀ y₀ : ℝ),
    original_circle x₀ y₀ ∧
    x = 2 * point_P.1 - x₀ ∧
    y = 2 * point_P.2 - y₀ :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_correct_l3289_328955


namespace NUMINAMATH_CALUDE_martha_children_count_l3289_328909

theorem martha_children_count (total_cakes num_cakes_per_child : ℕ) 
  (h1 : total_cakes = 18)
  (h2 : num_cakes_per_child = 6)
  (h3 : total_cakes % num_cakes_per_child = 0) :
  total_cakes / num_cakes_per_child = 3 := by
  sorry

end NUMINAMATH_CALUDE_martha_children_count_l3289_328909


namespace NUMINAMATH_CALUDE_tim_apartment_complexes_l3289_328951

/-- The number of keys Tim needs to make -/
def total_keys : ℕ := 72

/-- The number of keys needed per lock -/
def keys_per_lock : ℕ := 3

/-- The number of apartments in each complex -/
def apartments_per_complex : ℕ := 12

/-- The number of apartment complexes Tim owns -/
def num_complexes : ℕ := total_keys / keys_per_lock / apartments_per_complex

theorem tim_apartment_complexes : num_complexes = 2 := by
  sorry

end NUMINAMATH_CALUDE_tim_apartment_complexes_l3289_328951


namespace NUMINAMATH_CALUDE_cos_pi_half_minus_A_l3289_328937

theorem cos_pi_half_minus_A (A : ℝ) (h : Real.sin (π - A) = 1/2) : 
  Real.cos (π/2 - A) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_half_minus_A_l3289_328937


namespace NUMINAMATH_CALUDE_f_max_value_l3289_328977

/-- A function f(x) that is symmetric about x = -1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := -x^2 * (x^2 + a*x + b)

/-- Symmetry condition: f(x) = f(-2-x) for all x -/
def is_symmetric (a b : ℝ) : Prop := ∀ x, f a b x = f a b (-2-x)

/-- The maximum value of f(x) is 0 -/
theorem f_max_value (a b : ℝ) (h : is_symmetric a b) : 
  ∃ x₀, ∀ x, f a b x ≤ f a b x₀ ∧ f a b x₀ = 0 := by
  sorry

#check f_max_value

end NUMINAMATH_CALUDE_f_max_value_l3289_328977


namespace NUMINAMATH_CALUDE_xyz_sum_value_l3289_328947

theorem xyz_sum_value (x y z : ℝ) 
  (h1 : x^2 - y*z = 2) 
  (h2 : y^2 - z*x = 2) 
  (h3 : z^2 - x*y = 2) : 
  x*y + y*z + z*x = -2 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_value_l3289_328947


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_quadratics_l3289_328936

theorem unique_prime_with_prime_quadratics :
  ∃! p : ℕ, Nat.Prime p ∧ Nat.Prime (4 * p^2 + 1) ∧ Nat.Prime (6 * p^2 + 1) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_quadratics_l3289_328936


namespace NUMINAMATH_CALUDE_polynomial_no_ab_term_l3289_328965

theorem polynomial_no_ab_term (m : ℤ) : 
  (∀ a b : ℤ, 2 * (a^2 - 2*a*b - b^2) - (a^2 + m*a*b + 2*b^2) = a^2 - 4*b^2) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_no_ab_term_l3289_328965


namespace NUMINAMATH_CALUDE_sum_of_digits_63_l3289_328942

theorem sum_of_digits_63 : 
  let n : ℕ := 63
  let tens : ℕ := n / 10
  let ones : ℕ := n % 10
  tens - ones = 3 →
  tens + ones = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_63_l3289_328942


namespace NUMINAMATH_CALUDE_complex_not_purely_imaginary_range_l3289_328966

theorem complex_not_purely_imaginary_range (a : ℝ) : 
  ¬(∃ (y : ℝ), (a^2 - a - 2) + (|a-1| - 1)*I = y*I) → a ≠ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_not_purely_imaginary_range_l3289_328966


namespace NUMINAMATH_CALUDE_typing_sequences_count_l3289_328940

/-- Represents the state of letters in the secretary's inbox -/
structure LetterState where
  letters : List Nat
  typed : List Nat

/-- Calculates the number of possible typing sequences -/
def countSequences (state : LetterState) : Nat :=
  sorry

/-- The initial state of letters -/
def initialState : LetterState :=
  { letters := [1, 2, 3, 4, 5, 6, 7, 8, 9], typed := [] }

/-- The state after typing letters 8 and 5 -/
def stateAfterTyping : LetterState :=
  { letters := [6, 7, 9], typed := [8, 5] }

theorem typing_sequences_count :
  countSequences stateAfterTyping = 32 :=
sorry

end NUMINAMATH_CALUDE_typing_sequences_count_l3289_328940


namespace NUMINAMATH_CALUDE_discount_calculation_l3289_328978

/-- Calculates the discount given the cost price, markup percentage, and loss percentage -/
def calculate_discount (cost_price : ℝ) (markup_percentage : ℝ) (loss_percentage : ℝ) : ℝ :=
  let marked_price := cost_price * (1 + markup_percentage)
  let selling_price := cost_price * (1 - loss_percentage)
  marked_price - selling_price

/-- Theorem stating that for a cost price of 100, a markup of 40%, and a loss of 1%, the discount is 41 -/
theorem discount_calculation :
  calculate_discount 100 0.4 0.01 = 41 := by
  sorry


end NUMINAMATH_CALUDE_discount_calculation_l3289_328978


namespace NUMINAMATH_CALUDE_john_weight_is_250_l3289_328906

/-- The weight bench capacity in pounds -/
def bench_capacity : ℝ := 1000

/-- The safety margin percentage -/
def safety_margin : ℝ := 0.20

/-- The weight John puts on the bar in pounds -/
def bar_weight : ℝ := 550

/-- John's weight in pounds -/
def john_weight : ℝ := bench_capacity * (1 - safety_margin) - bar_weight

theorem john_weight_is_250 : john_weight = 250 := by
  sorry

end NUMINAMATH_CALUDE_john_weight_is_250_l3289_328906


namespace NUMINAMATH_CALUDE_polynomial_property_l3289_328911

/-- Polynomial P(x) = 3x^3 + ax^2 + bx + c satisfying given conditions -/
def P (a b c : ℝ) (x : ℝ) : ℝ := 3 * x^3 + a * x^2 + b * x + c

theorem polynomial_property (a b c : ℝ) :
  (∀ x₁ x₂ x₃ : ℝ, x₁ + x₂ + x₃ = 0 → P a b c x₁ = 0 → P a b c x₂ = 0 → P a b c x₃ = 0 →
    ((x₁ + x₂ + x₃) / 3 = x₁ * x₂ * x₃)) →  -- mean of zeros equals product of zeros
  (∀ x₁ x₂ x₃ : ℝ, x₁ + x₂ + x₃ = 0 → P a b c x₁ = 0 → P a b c x₂ = 0 → P a b c x₃ = 0 →
    (x₁ * x₂ * x₃ = 3 + a + b + c)) →  -- product of zeros equals sum of coefficients
  P a b c 0 = 15 →  -- y-intercept is 15
  b = -38 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l3289_328911


namespace NUMINAMATH_CALUDE_sin_pi_12_plus_theta_l3289_328950

theorem sin_pi_12_plus_theta (θ : ℝ) (h : Real.cos ((5 * Real.pi) / 12 - θ) = 1 / 3) :
  Real.sin (Real.pi / 12 + θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_12_plus_theta_l3289_328950


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3289_328943

/-- Given a line L₁: Ax + By + C = 0 and a point P₀(x₀, y₀), 
    the line L₂ passing through P₀ and perpendicular to L₁ 
    has the equation Bx - Ay - Bx₀ + Ay₀ = 0 -/
theorem perpendicular_line_equation (A B C x₀ y₀ : ℝ) :
  let L₁ := fun (x y : ℝ) ↦ A * x + B * y + C = 0
  let P₀ := (x₀, y₀)
  let L₂ := fun (x y : ℝ) ↦ B * x - A * y - B * x₀ + A * y₀ = 0
  (∀ x y, L₂ x y ↔ (x - x₀) * B = (y - y₀) * A) ∧
  (∀ x₁ y₁ x₂ y₂, L₁ x₁ y₁ ∧ L₁ x₂ y₂ → (x₂ - x₁) * B = -(y₂ - y₁) * A) ∧
  L₂ x₀ y₀ :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3289_328943


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3289_328989

def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3289_328989


namespace NUMINAMATH_CALUDE_rectangle_area_l3289_328949

/-- 
Given a rectangle with length l and width w, where:
1. The length is four times the width (l = 4w)
2. The perimeter is 200 cm (2l + 2w = 200)
Prove that the area of the rectangle is 1600 square centimeters.
-/
theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3289_328949


namespace NUMINAMATH_CALUDE_hyperbola_auxiliary_lines_l3289_328992

/-- Represents a hyperbola with given equation and asymptotes -/
structure Hyperbola where
  a : ℝ
  eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / 16 = 1
  asymptote : ∀ x : ℝ, ∃ y : ℝ, y = 4/3 * x ∨ y = -4/3 * x
  a_pos : a > 0

/-- The auxiliary lines of a hyperbola -/
def auxiliary_lines (h : Hyperbola) : Set ℝ :=
  {x : ℝ | x = 9/5 ∨ x = -9/5}

/-- Theorem stating that the auxiliary lines of the given hyperbola are x = ±9/5 -/
theorem hyperbola_auxiliary_lines (h : Hyperbola) :
  auxiliary_lines h = {x : ℝ | x = 9/5 ∨ x = -9/5} := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_auxiliary_lines_l3289_328992


namespace NUMINAMATH_CALUDE_average_shift_l3289_328933

theorem average_shift (x₁ x₂ x₃ x₄ : ℝ) :
  (x₁ + x₂ + x₃ + x₄) / 4 = 2 →
  ((x₁ + 3) + (x₂ + 3) + (x₃ + 3) + (x₄ + 3)) / 4 = 5 := by
sorry

end NUMINAMATH_CALUDE_average_shift_l3289_328933


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3289_328921

theorem remainder_divisibility (n : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 → n % 30 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3289_328921


namespace NUMINAMATH_CALUDE_savings_account_balance_l3289_328910

theorem savings_account_balance (initial_amount : ℝ) 
  (increase_percentage : ℝ) (decrease_percentage : ℝ) : 
  initial_amount = 125 ∧ 
  increase_percentage = 0.25 ∧ 
  decrease_percentage = 0.20 →
  initial_amount = initial_amount * (1 + increase_percentage) * (1 - decrease_percentage) :=
by sorry

end NUMINAMATH_CALUDE_savings_account_balance_l3289_328910


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l3289_328926

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  ((-1 : ℝ) / (1 + m) = -m / 2) ∧ (m ≠ -2)

/-- The value of m for which the lines x + (1+m)y = 2-m and mx + 2y + 8 = 0 are parallel -/
theorem parallel_lines_m_value : ∃! m : ℝ, parallel_lines m :=
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l3289_328926


namespace NUMINAMATH_CALUDE_anna_remaining_money_l3289_328929

-- Define the given values
def initial_money : ℝ := 50
def gum_price : ℝ := 1.50
def gum_quantity : ℕ := 4
def chocolate_price : ℝ := 2.25
def chocolate_quantity : ℕ := 7
def candy_cane_price : ℝ := 0.75
def candy_cane_quantity : ℕ := 3
def jelly_beans_original_price : ℝ := 3.00
def jelly_beans_discount : ℝ := 0.20
def sales_tax_rate : ℝ := 0.075

-- Calculate the total cost and remaining money
def calculate_remaining_money : ℝ :=
  let gum_cost := gum_price * gum_quantity
  let chocolate_cost := chocolate_price * chocolate_quantity
  let candy_cane_cost := candy_cane_price * candy_cane_quantity
  let jelly_beans_cost := jelly_beans_original_price * (1 - jelly_beans_discount)
  let total_before_tax := gum_cost + chocolate_cost + candy_cane_cost + jelly_beans_cost
  let total_after_tax := total_before_tax * (1 + sales_tax_rate)
  initial_money - total_after_tax

-- Theorem to prove
theorem anna_remaining_money :
  calculate_remaining_money = 21.62 := by sorry

end NUMINAMATH_CALUDE_anna_remaining_money_l3289_328929


namespace NUMINAMATH_CALUDE_walking_cycling_speeds_l3289_328963

/-- Proves that given conditions result in specific walking and cycling speeds -/
theorem walking_cycling_speeds (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 2 →
  speed_ratio = 4 →
  time_difference = 1/3 →
  ∃ (walking_speed cycling_speed : ℝ),
    walking_speed = 4.5 ∧
    cycling_speed = 18 ∧
    cycling_speed = speed_ratio * walking_speed ∧
    distance / walking_speed - distance / cycling_speed = time_difference :=
by sorry

end NUMINAMATH_CALUDE_walking_cycling_speeds_l3289_328963


namespace NUMINAMATH_CALUDE_abc_zero_l3289_328912

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) :
  a = 0 ∨ b = 0 ∨ c = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_zero_l3289_328912


namespace NUMINAMATH_CALUDE_f_x_plus_5_l3289_328969

def f (x : ℝ) := 3 * x + 1

theorem f_x_plus_5 (x : ℝ) : f (x + 5) = 3 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_f_x_plus_5_l3289_328969


namespace NUMINAMATH_CALUDE_point_b_representation_l3289_328946

theorem point_b_representation (a b : ℝ) : 
  a = -2 → (b - a = 3 ∨ a - b = 3) → (b = 1 ∨ b = -5) := by sorry

end NUMINAMATH_CALUDE_point_b_representation_l3289_328946


namespace NUMINAMATH_CALUDE_grey_eyed_black_hair_count_l3289_328973

/-- Represents the number of students with a specific hair color and eye color combination -/
structure StudentCount where
  redHairGreenEyes : ℕ
  redHairGreyEyes : ℕ
  blackHairGreenEyes : ℕ
  blackHairGreyEyes : ℕ

/-- Theorem stating the number of grey-eyed students with black hair -/
theorem grey_eyed_black_hair_count (s : StudentCount) : s.blackHairGreyEyes = 20 :=
  by
  have total_students : s.redHairGreenEyes + s.redHairGreyEyes + s.blackHairGreenEyes + s.blackHairGreyEyes = 60 := by sorry
  have green_eyed_red_hair : s.redHairGreenEyes = 20 := by sorry
  have black_hair_total : s.blackHairGreenEyes + s.blackHairGreyEyes = 36 := by sorry
  have grey_eyes_total : s.redHairGreyEyes + s.blackHairGreyEyes = 24 := by sorry
  sorry

#check grey_eyed_black_hair_count

end NUMINAMATH_CALUDE_grey_eyed_black_hair_count_l3289_328973


namespace NUMINAMATH_CALUDE_zeros_in_repeated_nines_square_l3289_328919

/-- The number of zeros in the decimal representation of n^2 -/
def zeros_in_square (n : ℕ) : ℕ := sorry

/-- The number of nines in the decimal representation of n -/
def count_nines (n : ℕ) : ℕ := sorry

theorem zeros_in_repeated_nines_square (n : ℕ) :
  (∀ k ≤ 3, zeros_in_square (10^k - 1) = k - 1) →
  count_nines 999999 = 6 →
  zeros_in_square 999999 = 5 := by sorry

end NUMINAMATH_CALUDE_zeros_in_repeated_nines_square_l3289_328919


namespace NUMINAMATH_CALUDE_intersection_in_first_quadrant_l3289_328987

/-- The intersection point of two lines is in the first quadrant iff a is in the range (-1, 2) -/
theorem intersection_in_first_quadrant (a : ℝ) :
  (∃ x y : ℝ, ax + y - 4 = 0 ∧ x - y - 2 = 0 ∧ x > 0 ∧ y > 0) ↔ -1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_in_first_quadrant_l3289_328987


namespace NUMINAMATH_CALUDE_sin_105_cos_105_l3289_328984

theorem sin_105_cos_105 : Real.sin (105 * π / 180) * Real.cos (105 * π / 180) = -(1/4) := by
  sorry

end NUMINAMATH_CALUDE_sin_105_cos_105_l3289_328984


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l3289_328982

theorem right_triangle_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (right_triangle : a^2 + b^2 = c^2) (leg : b = 5) (hyp : c = 13) :
  a / c = 12 / 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l3289_328982


namespace NUMINAMATH_CALUDE_lara_flowers_in_vase_l3289_328957

/-- The number of flowers Lara put in the vase -/
def flowers_in_vase (total : ℕ) (to_mom : ℕ) (extra_to_grandma : ℕ) : ℕ :=
  total - (to_mom + (to_mom + extra_to_grandma))

/-- Theorem stating the number of flowers Lara put in the vase -/
theorem lara_flowers_in_vase :
  flowers_in_vase 52 15 6 = 16 := by sorry

end NUMINAMATH_CALUDE_lara_flowers_in_vase_l3289_328957


namespace NUMINAMATH_CALUDE_angle_two_pi_third_in_second_quadrant_l3289_328967

/-- The angle 2π/3 is in the second quadrant -/
theorem angle_two_pi_third_in_second_quadrant :
  let θ : Real := 2 * Real.pi / 3
  0 < θ ∧ θ < Real.pi / 2 → False
  ∧ Real.pi / 2 < θ ∧ θ ≤ Real.pi → True
  ∧ Real.pi < θ ∧ θ ≤ 3 * Real.pi / 2 → False
  ∧ 3 * Real.pi / 2 < θ ∧ θ < 2 * Real.pi → False :=
by sorry

end NUMINAMATH_CALUDE_angle_two_pi_third_in_second_quadrant_l3289_328967


namespace NUMINAMATH_CALUDE_variable_value_l3289_328932

theorem variable_value (w x v : ℝ) 
  (h1 : 5 / w + 5 / x = 5 / v) 
  (h2 : w * x = v)
  (h3 : (w + x) / 2 = 0.5) : 
  v = 0.25 := by
sorry

end NUMINAMATH_CALUDE_variable_value_l3289_328932


namespace NUMINAMATH_CALUDE_frank_reading_speed_l3289_328990

/-- Given a book with a certain number of pages and the number of days to read it,
    calculate the number of pages read per day. -/
def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

/-- Theorem stating that for a book with 249 pages read in 3 days,
    the number of pages read per day is 83. -/
theorem frank_reading_speed :
  pages_per_day 249 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_speed_l3289_328990


namespace NUMINAMATH_CALUDE_james_sold_five_last_week_l3289_328941

/-- The number of chocolate bars sold last week -/
def chocolate_bars_sold_last_week (total : ℕ) (sold_this_week : ℕ) (need_to_sell : ℕ) : ℕ :=
  total - (sold_this_week + need_to_sell)

/-- Theorem stating that James sold 5 chocolate bars last week -/
theorem james_sold_five_last_week :
  chocolate_bars_sold_last_week 18 7 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_james_sold_five_last_week_l3289_328941


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3289_328958

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if both roots of a quadratic equation are greater than 1 -/
def bothRootsGreaterThanOne (eq : QuadraticEquation) : Prop :=
  ∀ x, eq.a * x^2 + eq.b * x + eq.c = 0 → x > 1

/-- The main theorem stating the condition on m -/
theorem quadratic_roots_condition (m : ℝ) :
  let eq : QuadraticEquation := ⟨8, 1 - m, m - 7⟩
  bothRootsGreaterThanOne eq → m ≥ 25 := by
  sorry

#check quadratic_roots_condition

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3289_328958


namespace NUMINAMATH_CALUDE_strawberry_division_l3289_328970

def strawberry_problem (brother_baskets : ℕ) (strawberries_per_basket : ℕ) 
  (kimberly_multiplier : ℕ) (parents_difference : ℕ) (family_size : ℕ) : Prop :=
  let brother_strawberries := brother_baskets * strawberries_per_basket
  let kimberly_strawberries := kimberly_multiplier * brother_strawberries
  let parents_strawberries := kimberly_strawberries - parents_difference
  let total_strawberries := kimberly_strawberries + brother_strawberries + parents_strawberries
  (total_strawberries / family_size = 168)

theorem strawberry_division :
  strawberry_problem 3 15 8 93 4 :=
by
  sorry

#check strawberry_division

end NUMINAMATH_CALUDE_strawberry_division_l3289_328970


namespace NUMINAMATH_CALUDE_no_covalent_bond_IA_VIIA_l3289_328972

/-- Represents an element in the periodic table -/
structure Element where
  group : ℕ
  isHydrogen : Bool

/-- Represents the bonding behavior of an element -/
inductive BondingBehavior
  | LoseElectrons
  | GainElectrons

/-- Determines the bonding behavior of an element based on its group -/
def bondingBehavior (e : Element) : BondingBehavior :=
  if e.group = 1 ∧ ¬e.isHydrogen then BondingBehavior.LoseElectrons
  else if e.group = 17 then BondingBehavior.GainElectrons
  else BondingBehavior.LoseElectrons  -- Default case, not relevant for this problem

/-- Determines if two elements can form a covalent bond -/
def canFormCovalentBond (e1 e2 : Element) : Prop :=
  bondingBehavior e1 = bondingBehavior e2

/-- Theorem stating that elements in Group IA (except H) and Group VIIA cannot form covalent bonds -/
theorem no_covalent_bond_IA_VIIA :
  ∀ (e1 e2 : Element),
    ((e1.group = 1 ∧ ¬e1.isHydrogen) ∨ e1.group = 17) →
    ((e2.group = 1 ∧ ¬e2.isHydrogen) ∨ e2.group = 17) →
    ¬(canFormCovalentBond e1 e2) :=
by
  sorry

end NUMINAMATH_CALUDE_no_covalent_bond_IA_VIIA_l3289_328972


namespace NUMINAMATH_CALUDE_expression_evaluation_l3289_328981

theorem expression_evaluation : (4 * 6) / (12 * 16) * (8 * 12 * 16) / (4 * 6 * 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3289_328981


namespace NUMINAMATH_CALUDE_find_second_number_l3289_328901

/-- Given two positive integers with known HCF and LCM, find the second number -/
theorem find_second_number (A B : ℕ+) (h1 : A = 24) 
  (h2 : Nat.gcd A B = 13) (h3 : Nat.lcm A B = 312) : B = 169 := by
  sorry

end NUMINAMATH_CALUDE_find_second_number_l3289_328901


namespace NUMINAMATH_CALUDE_problem_solution_l3289_328971

theorem problem_solution (x y : ℝ) 
  (h1 : 2*x + 3*y = 9) 
  (h2 : x*y = -12) : 
  4*x^2 + 9*y^2 = 225 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3289_328971


namespace NUMINAMATH_CALUDE_hyperbola_equation_theorem_l3289_328964

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  /-- The focal distance of the hyperbola -/
  focal_distance : ℝ
  /-- The distance from a focus to an asymptote -/
  focus_to_asymptote : ℝ
  /-- Assumption that the foci are on the x-axis -/
  foci_on_x_axis : Bool

/-- The equation of a hyperbola given its properties -/
def hyperbola_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 - y^2 / 3 = 1 ↔ 
    h.focal_distance = 4 ∧ 
    h.focus_to_asymptote = Real.sqrt 3 ∧ 
    h.foci_on_x_axis = true

/-- Theorem stating that a hyperbola with the given properties has the specified equation -/
theorem hyperbola_equation_theorem (h : Hyperbola) :
  h.focal_distance = 4 →
  h.focus_to_asymptote = Real.sqrt 3 →
  h.foci_on_x_axis = true →
  hyperbola_equation h :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_theorem_l3289_328964


namespace NUMINAMATH_CALUDE_four_digit_numbers_with_specific_remainders_l3289_328952

theorem four_digit_numbers_with_specific_remainders :
  ∀ N : ℕ,
  (1000 ≤ N ∧ N ≤ 9999) →
  (N % 2 = 0 ∧ N % 3 = 1 ∧ N % 5 = 3 ∧ N % 7 = 5 ∧ N % 11 = 9) →
  (N = 2308 ∨ N = 4618 ∨ N = 6928 ∨ N = 9238) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_numbers_with_specific_remainders_l3289_328952


namespace NUMINAMATH_CALUDE_sum_floor_equals_n_l3289_328975

/-- For any natural number n, the sum of floor((n + 2^k) / 2^(k+1)) from k = 0 to infinity equals n -/
theorem sum_floor_equals_n (n : ℕ) :
  (∑' k : ℕ, ⌊(n + 2^k : ℝ) / (2^(k+1) : ℝ)⌋) = n :=
sorry

end NUMINAMATH_CALUDE_sum_floor_equals_n_l3289_328975


namespace NUMINAMATH_CALUDE_flood_monitoring_technologies_l3289_328922

-- Define the set of available technologies
inductive GeoTechnology
| RemoteSensing
| GPS
| GIS
| DigitalEarth

-- Define the capabilities of technologies
def canMonitorDisaster (tech : GeoTechnology) : Prop :=
  match tech with
  | GeoTechnology.RemoteSensing => true
  | _ => false

def canManageInfo (tech : GeoTechnology) : Prop :=
  match tech with
  | GeoTechnology.GIS => true
  | _ => false

def isEffectiveForFloodMonitoring (tech : GeoTechnology) : Prop :=
  canMonitorDisaster tech ∨ canManageInfo tech

-- Define the set of effective technologies
def effectiveTechnologies : Set GeoTechnology :=
  {tech | isEffectiveForFloodMonitoring tech}

-- Theorem statement
theorem flood_monitoring_technologies :
  effectiveTechnologies = {GeoTechnology.RemoteSensing, GeoTechnology.GIS} :=
sorry

end NUMINAMATH_CALUDE_flood_monitoring_technologies_l3289_328922


namespace NUMINAMATH_CALUDE_gcd_count_for_product_504_l3289_328996

theorem gcd_count_for_product_504 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 504) :
  ∃! (s : Finset ℕ), s.card = 9 ∧ ∀ d, d ∈ s ↔ ∃ (a' b' : ℕ+), Nat.gcd a' b' * Nat.lcm a' b' = 504 ∧ Nat.gcd a' b' = d :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_504_l3289_328996


namespace NUMINAMATH_CALUDE_range_of_m_plus_n_l3289_328939

/-- The function f(x) = x^2 + nx + m -/
def f (n m x : ℝ) : ℝ := x^2 + n*x + m

/-- The set of roots of f -/
def roots (n m : ℝ) : Set ℝ := {x | f n m x = 0}

/-- The set of roots of f(f(x)) -/
def roots_of_f_of_f (n m : ℝ) : Set ℝ := {x | f n m (f n m x) = 0}

theorem range_of_m_plus_n (n m : ℝ) :
  roots n m = roots_of_f_of_f n m ∧ roots n m ≠ ∅ → 0 < m + n ∧ m + n < 4 := by sorry

end NUMINAMATH_CALUDE_range_of_m_plus_n_l3289_328939


namespace NUMINAMATH_CALUDE_rachel_essay_time_l3289_328953

/-- Calculates the total time spent on an essay given the writing speed, number of pages, research time, and editing time. -/
def total_essay_time (writing_speed : ℝ) (pages : ℕ) (research_time : ℝ) (editing_time : ℝ) : ℝ :=
  (pages : ℝ) * writing_speed + research_time + editing_time

/-- Proves that Rachel spent 5 hours on her essay given the conditions. -/
theorem rachel_essay_time : 
  let writing_speed : ℝ := 30  -- minutes per page
  let pages : ℕ := 6
  let research_time : ℝ := 45  -- minutes
  let editing_time : ℝ := 75   -- minutes
  total_essay_time writing_speed pages research_time editing_time / 60 = 5 := by
sorry


end NUMINAMATH_CALUDE_rachel_essay_time_l3289_328953


namespace NUMINAMATH_CALUDE_profit_ratio_equals_investment_ratio_l3289_328986

/-- The ratio of profits is proportional to the ratio of investments -/
theorem profit_ratio_equals_investment_ratio (p_investment q_investment : ℚ) 
  (hp : p_investment = 50000)
  (hq : q_investment = 66666.67)
  : ∃ (k : ℚ), k * p_investment = 3 ∧ k * q_investment = 4 := by
  sorry

end NUMINAMATH_CALUDE_profit_ratio_equals_investment_ratio_l3289_328986


namespace NUMINAMATH_CALUDE_stratified_sampling_l3289_328974

/-- Represents the number of questionnaires for each unit -/
structure Questionnaires :=
  (a b c d : ℕ)

/-- Represents the sample sizes for each unit -/
structure SampleSizes :=
  (a b c d : ℕ)

/-- Checks if the given questionnaire counts form an arithmetic sequence -/
def is_arithmetic_sequence (q : Questionnaires) : Prop :=
  q.b - q.a = q.c - q.b ∧ q.c - q.b = q.d - q.c

/-- The main theorem statement -/
theorem stratified_sampling
  (q : Questionnaires)
  (s : SampleSizes)
  (h1 : q.a + q.b + q.c + q.d = 1000)
  (h2 : is_arithmetic_sequence q)
  (h3 : s.a + s.b + s.c + s.d = 150)
  (h4 : s.b = 30)
  (h5 : s.b * 1000 = q.b * 150) :
  s.d = 60 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_l3289_328974


namespace NUMINAMATH_CALUDE_bens_previous_salary_l3289_328976

/-- Prove that Ben's previous job's annual salary was $75,000 given the conditions of his new job --/
theorem bens_previous_salary (new_base_salary : ℝ) (commission_rate : ℝ) (sale_price : ℝ) (min_sales : ℝ) :
  new_base_salary = 45000 →
  commission_rate = 0.15 →
  sale_price = 750 →
  min_sales = 266.67 →
  ∃ (previous_salary : ℝ), 
    previous_salary ≥ new_base_salary + commission_rate * sale_price * min_sales ∧
    previous_salary < new_base_salary + commission_rate * sale_price * min_sales + 1 :=
by
  sorry

#eval (45000 : ℝ) + 0.15 * 750 * 266.67

end NUMINAMATH_CALUDE_bens_previous_salary_l3289_328976


namespace NUMINAMATH_CALUDE_vertical_angles_are_congruent_l3289_328962

-- Define what it means for two angles to be vertical
def are_vertical_angles (α β : Angle) : Prop := sorry

-- Define what it means for two angles to be congruent
def are_congruent (α β : Angle) : Prop := sorry

-- Theorem statement
theorem vertical_angles_are_congruent (α β : Angle) : 
  are_vertical_angles α β → are_congruent α β := by
  sorry

end NUMINAMATH_CALUDE_vertical_angles_are_congruent_l3289_328962


namespace NUMINAMATH_CALUDE_potatoes_left_l3289_328956

theorem potatoes_left (initial : ℕ) (salad : ℕ) (mashed : ℕ) (h1 : initial = 52) (h2 : salad = 15) (h3 : mashed = 24) : initial - (salad + mashed) = 13 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_left_l3289_328956


namespace NUMINAMATH_CALUDE_shuttle_speed_km_per_hour_l3289_328907

/-- The speed of a space shuttle orbiting the Earth -/
def shuttle_speed_km_per_sec : ℝ := 2

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: The speed of the space shuttle in kilometers per hour -/
theorem shuttle_speed_km_per_hour :
  shuttle_speed_km_per_sec * (seconds_per_hour : ℝ) = 7200 := by
  sorry

end NUMINAMATH_CALUDE_shuttle_speed_km_per_hour_l3289_328907


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l3289_328993

theorem geometric_progression_ratio (a : ℝ) (r : ℝ) :
  a > 0 ∧ r > 0 ∧ 
  (∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)) →
  r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l3289_328993


namespace NUMINAMATH_CALUDE_regression_satisfies_negative_correlation_l3289_328944

/-- Represents the regression equation for sales volume based on selling price -/
def regression_equation (x : ℝ) : ℝ := -2 * x + 100

/-- Represents the correlation between sales volume and selling price -/
def negative_correlation (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂

theorem regression_satisfies_negative_correlation :
  negative_correlation regression_equation :=
sorry

end NUMINAMATH_CALUDE_regression_satisfies_negative_correlation_l3289_328944


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l3289_328959

-- Define the original function
def f (x : ℝ) : ℝ := |x + 1| - 2

-- Define the translation
def translate_x : ℝ := 3
def translate_y : ℝ := 4

-- Define the translated function
def g (x : ℝ) : ℝ := f (x - translate_x) + translate_y

-- State the theorem
theorem minimum_point_of_translated_graph :
  ∃ (x₀ y₀ : ℝ), x₀ = 2 ∧ y₀ = 2 ∧
  ∀ (x : ℝ), g x ≥ g x₀ :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l3289_328959


namespace NUMINAMATH_CALUDE_maria_paint_cans_l3289_328904

/-- Represents the paint situation for Maria's room painting problem -/
structure PaintSituation where
  initialRooms : ℕ
  finalRooms : ℕ
  lostCans : ℕ

/-- Calculates the number of cans used for the final number of rooms -/
def cansUsed (s : PaintSituation) : ℕ :=
  s.finalRooms / ((s.initialRooms - s.finalRooms) / s.lostCans)

/-- Theorem stating that for Maria's specific situation, 16 cans were used -/
theorem maria_paint_cans :
  let s : PaintSituation := { initialRooms := 40, finalRooms := 32, lostCans := 4 }
  cansUsed s = 16 := by sorry

end NUMINAMATH_CALUDE_maria_paint_cans_l3289_328904


namespace NUMINAMATH_CALUDE_A_power_2023_l3289_328960

def A : Matrix (Fin 3) (Fin 3) ℚ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_2023 : A^2023 = A := by sorry

end NUMINAMATH_CALUDE_A_power_2023_l3289_328960


namespace NUMINAMATH_CALUDE_waiter_customer_count_l3289_328994

theorem waiter_customer_count (initial : Float) (lunch_rush : Float) (later : Float) :
  initial = 29.0 →
  lunch_rush = 20.0 →
  later = 34.0 →
  initial + lunch_rush + later = 83.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customer_count_l3289_328994


namespace NUMINAMATH_CALUDE_system_solution_l3289_328934

theorem system_solution :
  ∀ x y z : ℝ,
  (x^2 + y^2 + 25*z^2 = 6*x*z + 8*y*z) ∧
  (3*x^2 + 2*y^2 + z^2 = 240) →
  ((x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3289_328934


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3289_328924

theorem simplify_trig_expression (α : ℝ) :
  3 - 4 * Real.cos (4 * α) + Real.cos (8 * α) - 8 * (Real.cos (2 * α))^4 = -8 * Real.cos (4 * α) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3289_328924


namespace NUMINAMATH_CALUDE_some_number_divisibility_l3289_328991

theorem some_number_divisibility (x : ℕ) : (1425 * x * 1429) % 12 = 3 ↔ x % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_divisibility_l3289_328991


namespace NUMINAMATH_CALUDE_f_derivative_l3289_328999

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x + 1)^2 - (x + 1)

-- State the theorem
theorem f_derivative : 
  ∀ x : ℝ, deriv f x = 4 * x + 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_l3289_328999


namespace NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l3289_328905

def total_students : ℕ := 5
def students_to_select : ℕ := 3

theorem probability_of_selecting_A_and_B :
  (Nat.choose (total_students - 2) (students_to_select - 2)) / 
  (Nat.choose total_students students_to_select) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l3289_328905


namespace NUMINAMATH_CALUDE_zeros_in_square_of_999999999_l3289_328979

/-- The number of zeros in the decimal expansion of (999,999,999)^2 -/
def zeros_in_square_of_nines : ℕ := 8

/-- The observed pattern: squaring a number with n nines results in n-1 zeros -/
axiom pattern_holds (n : ℕ) : 
  ∀ x : ℕ, x = 10^n - 1 → (∃ k : ℕ, x^2 = k * 10^(n-1) ∧ k % 10 ≠ 0)

/-- Theorem: The number of zeros in the decimal expansion of (999,999,999)^2 is 8 -/
theorem zeros_in_square_of_999999999 : 
  ∃ k : ℕ, (999999999 : ℕ)^2 = k * 10^zeros_in_square_of_nines ∧ k % 10 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zeros_in_square_of_999999999_l3289_328979


namespace NUMINAMATH_CALUDE_reflex_angle_at_H_l3289_328954

-- Define the points
variable (C D F M H : Point)

-- Define the angles
def angle_CDH : ℝ := 150
def angle_HFM : ℝ := 95

-- Define the properties
def collinear (C D F M : Point) : Prop := sorry
def angle (A B C : Point) : ℝ := sorry
def reflex_angle (A : Point) : ℝ := sorry

-- State the theorem
theorem reflex_angle_at_H (h_collinear : collinear C D F M) 
  (h_CDH : angle C D H = angle_CDH)
  (h_HFM : angle H F M = angle_HFM) : 
  reflex_angle H = 180 := by sorry

end NUMINAMATH_CALUDE_reflex_angle_at_H_l3289_328954


namespace NUMINAMATH_CALUDE_barbed_wire_rate_l3289_328923

/-- The rate of drawing barbed wire per meter given the conditions of the problem -/
theorem barbed_wire_rate (field_area : ℝ) (wire_extension : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ)
  (h1 : field_area = 3136)
  (h2 : wire_extension = 3)
  (h3 : gate_width = 1)
  (h4 : num_gates = 2)
  (h5 : total_cost = 732.6) :
  (total_cost / (4 * Real.sqrt field_area + wire_extension - num_gates * gate_width)) = 3.256 := by
  sorry

end NUMINAMATH_CALUDE_barbed_wire_rate_l3289_328923


namespace NUMINAMATH_CALUDE_arrangements_A_not_head_B_not_tail_arrangements_at_least_one_between_A_B_arrangements_A_B_together_C_D_not_together_l3289_328916

-- Define the number of people
def n : ℕ := 5

-- Define the factorial function
def factorial (m : ℕ) : ℕ := (List.range m).foldl (· * ·) 1

-- Define the permutation function
def permutation (m k : ℕ) : ℕ := 
  if k > m then 0
  else factorial m / factorial (m - k)

-- Theorem 1
theorem arrangements_A_not_head_B_not_tail : 
  permutation n n - 2 * permutation (n - 1) (n - 1) + permutation (n - 2) (n - 2) = 78 := by sorry

-- Theorem 2
theorem arrangements_at_least_one_between_A_B :
  permutation n n - permutation (n - 1) (n - 1) * permutation 2 2 = 72 := by sorry

-- Theorem 3
theorem arrangements_A_B_together_C_D_not_together :
  permutation 2 2 * permutation 2 2 * permutation 3 2 = 24 := by sorry

end NUMINAMATH_CALUDE_arrangements_A_not_head_B_not_tail_arrangements_at_least_one_between_A_B_arrangements_A_B_together_C_D_not_together_l3289_328916


namespace NUMINAMATH_CALUDE_cubic_equation_has_real_root_l3289_328908

theorem cubic_equation_has_real_root (a b : ℝ) : 
  ∃ x : ℝ, x^3 + a*x + b = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_has_real_root_l3289_328908


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3289_328961

theorem simplify_square_roots : 
  (Real.sqrt 648 / Real.sqrt 81) - (Real.sqrt 245 / Real.sqrt 49) = 2 * Real.sqrt 2 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3289_328961


namespace NUMINAMATH_CALUDE_polygon_angles_l3289_328935

theorem polygon_angles (n : ℕ) : 
  (n - 2) * 180 = 5 * 360 → n = 12 := by
sorry

end NUMINAMATH_CALUDE_polygon_angles_l3289_328935


namespace NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l3289_328968

theorem gcd_of_powers_minus_one :
  Nat.gcd (2^2100 - 1) (2^1950 - 1) = 2^150 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l3289_328968


namespace NUMINAMATH_CALUDE_composite_expression_l3289_328997

theorem composite_expression (x y : ℕ) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 2022*x^2 + 349*x + 72*x*y + 12*y + 2 = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_expression_l3289_328997


namespace NUMINAMATH_CALUDE_ab_plus_cd_eq_zero_l3289_328917

/-- Given real numbers a, b, c, d satisfying certain conditions, prove that ab + cd = 0 -/
theorem ab_plus_cd_eq_zero 
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : a*c + b*d = 0) : 
  a*b + c*d = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_plus_cd_eq_zero_l3289_328917


namespace NUMINAMATH_CALUDE_andrew_bought_65_planks_l3289_328920

/-- The number of wooden planks Andrew bought initially -/
def total_planks : ℕ :=
  let andrew_bedroom := 8
  let living_room := 20
  let kitchen := 11
  let guest_bedroom := andrew_bedroom - 2
  let hallway := 4
  let num_hallways := 2
  let ruined_per_bedroom := 3
  let num_bedrooms := 2
  let leftover := 6
  andrew_bedroom + living_room + kitchen + guest_bedroom + 
  (hallway * num_hallways) + (ruined_per_bedroom * num_bedrooms) + leftover

/-- Theorem stating that Andrew bought 65 wooden planks initially -/
theorem andrew_bought_65_planks : total_planks = 65 := by
  sorry

end NUMINAMATH_CALUDE_andrew_bought_65_planks_l3289_328920
