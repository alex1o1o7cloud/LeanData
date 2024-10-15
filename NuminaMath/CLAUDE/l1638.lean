import Mathlib

namespace NUMINAMATH_CALUDE_triangle_inequality_and_equality_l1638_163821

theorem triangle_inequality_and_equality (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) 
    ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c) ∧ 
  ((Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) 
    = Real.sqrt a + Real.sqrt b + Real.sqrt c) ↔ (a = b ∧ b = c)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_and_equality_l1638_163821


namespace NUMINAMATH_CALUDE_rational_sum_zero_l1638_163804

theorem rational_sum_zero (a b c : ℚ) 
  (h : (a + b + c) * (a + b - c) = 4 * c^2) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_zero_l1638_163804


namespace NUMINAMATH_CALUDE_cubes_form_larger_cube_l1638_163812

/-- A function that determines if n cubes can form a larger cube -/
def can_form_cube (n : ℕ) : Prop :=
  ∃ (side : ℕ), side^3 = n

/-- The theorem stating that for all natural numbers greater than 70,
    it's possible to select that many cubes to form a larger cube -/
theorem cubes_form_larger_cube :
  ∃ (N : ℕ), ∀ (n : ℕ), n > N → can_form_cube n :=
sorry

end NUMINAMATH_CALUDE_cubes_form_larger_cube_l1638_163812


namespace NUMINAMATH_CALUDE_cos_13_17_minus_sin_17_13_l1638_163854

theorem cos_13_17_minus_sin_17_13 :
  Real.cos (13 * π / 180) * Real.cos (17 * π / 180) - 
  Real.sin (17 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_13_17_minus_sin_17_13_l1638_163854


namespace NUMINAMATH_CALUDE_intersection_union_theorem_complement_intersection_theorem_l1638_163871

def A : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | |x| + a < 0}

theorem intersection_union_theorem :
  ∀ a : ℝ, a = -4 →
    (A ∩ B a = {x : ℝ | 1/2 ≤ x ∧ x ≤ 3}) ∧
    (A ∪ B a = {x : ℝ | -4 < x ∧ x < 4}) :=
by sorry

theorem complement_intersection_theorem :
  ∀ a : ℝ, (Aᶜ ∩ B a = B a) ↔ a ≥ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_union_theorem_complement_intersection_theorem_l1638_163871


namespace NUMINAMATH_CALUDE_intersection_point_polar_coordinates_l1638_163863

/-- The intersection point of ρ = 2sinθ and ρ = 2cosθ in polar coordinates -/
theorem intersection_point_polar_coordinates :
  ∃ (ρ θ : ℝ), ρ > 0 ∧ 0 ≤ θ ∧ θ < π/2 ∧ 
  ρ = 2 * Real.sin θ ∧ ρ = 2 * Real.cos θ ∧
  ρ = Real.sqrt 2 ∧ θ = π/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_polar_coordinates_l1638_163863


namespace NUMINAMATH_CALUDE_orthocenter_locus_is_circle_l1638_163851

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Check if a point lies on a circle -/
def onCircle (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a point lies outside a circle -/
def outsideCircle (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 > c.radius^2

/-- Intersection points of a secant with a circle -/
def secantIntersection (c : Circle) (k : Point) (p q : Point) : Prop :=
  onCircle c p ∧ onCircle c q ∧ ∃ t : ℝ, p = (k.1 + t * (q.1 - k.1), k.2 + t * (q.2 - k.2))

/-- Orthocenter of a triangle -/
def orthocenter (a p q : Point) : Point := sorry

/-- Main theorem -/
theorem orthocenter_locus_is_circle 
  (c : Circle) (a k : Point) 
  (h_a : onCircle c a) 
  (h_k : outsideCircle c k) :
  ∃ c' : Circle, ∀ p q : Point, 
    secantIntersection c k p q → 
    onCircle c' (orthocenter a p q) := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_locus_is_circle_l1638_163851


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1638_163853

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sine law for triangles -/
axiom sine_law (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The cosine law for triangles -/
axiom cosine_law (t : Triangle) : Real.cos t.C = (t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)

theorem triangle_abc_properties (t : Triangle) 
  (ha : t.a = 4)
  (hc : t.c = Real.sqrt 13)
  (hsin : Real.sin t.A = 4 * Real.sin t.B) :
  t.b = 1 ∧ t.C = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l1638_163853


namespace NUMINAMATH_CALUDE_fib_gcd_property_fib_1960_1988_gcd_l1638_163889

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fib_gcd_property (m n : ℕ) : Nat.gcd (fibonacci m) (fibonacci n) = fibonacci (Nat.gcd m n) := by
  sorry

theorem fib_1960_1988_gcd : Nat.gcd (fibonacci 1960) (fibonacci 1988) = fibonacci 28 := by
  sorry

end NUMINAMATH_CALUDE_fib_gcd_property_fib_1960_1988_gcd_l1638_163889


namespace NUMINAMATH_CALUDE_smallest_difference_in_digit_sum_sequence_l1638_163831

/-- The sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Predicate for numbers whose digit sum is divisible by 5 -/
def digitSumDivisibleBy5 (n : ℕ) : Prop := digitSum n % 5 = 0

theorem smallest_difference_in_digit_sum_sequence :
  ∃ (a b : ℕ), a < b ∧ 
    digitSumDivisibleBy5 a ∧ 
    digitSumDivisibleBy5 b ∧ 
    b - a = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_in_digit_sum_sequence_l1638_163831


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l1638_163860

def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 4) ∧
  (∀ x : ℝ, (x - 1) * (x - 2) * (x - 4) = p x) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l1638_163860


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l1638_163898

theorem geometric_to_arithmetic_sequence (a₁ a₂ a₃ a₄ q : ℝ) :
  q > 0 ∧ q ≠ 1 ∧ 
  a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧
  ((2 * a₃ = a₁ + a₄) ∨ (2 * a₂ = a₁ + a₄)) →
  q = ((-1 + Real.sqrt 5) / 2) ∨ q = ((1 + Real.sqrt 5) / 2) := by
sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l1638_163898


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l1638_163858

-- Define the number of rings and fingers
def total_rings : ℕ := 9
def rings_to_arrange : ℕ := 5
def fingers : ℕ := 5

-- Define the function to calculate the number of arrangements
def ring_arrangements (total : ℕ) (arrange : ℕ) (fingers : ℕ) : ℕ :=
  (Nat.choose total arrange) * (Nat.factorial arrange) * (Nat.choose (arrange + fingers - 1) (fingers - 1))

-- Theorem statement
theorem ring_arrangement_count :
  ring_arrangements total_rings rings_to_arrange fingers = 1900800 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l1638_163858


namespace NUMINAMATH_CALUDE_sam_coin_problem_l1638_163887

/-- Represents the number of coins of each type -/
structure CoinCount where
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total number of coins -/
def totalCoins (coins : CoinCount) : ℕ :=
  coins.dimes + coins.nickels + coins.pennies

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Represents the transactions Sam made -/
def samTransactions (initial : CoinCount) : CoinCount :=
  let afterDad := CoinCount.mk (initial.dimes + 7) (initial.nickels - 3) initial.pennies
  CoinCount.mk (afterDad.dimes + 2) afterDad.nickels 2

theorem sam_coin_problem (initial : CoinCount) 
  (h_initial : initial = CoinCount.mk 9 5 12) : 
  let final := samTransactions initial
  totalCoins final = 22 ∧ totalValue final = 192 := by
  sorry

end NUMINAMATH_CALUDE_sam_coin_problem_l1638_163887


namespace NUMINAMATH_CALUDE_john_calorie_burn_l1638_163802

/-- Represents the number of calories John burns per day -/
def calories_burned_per_day : ℕ := 2300

/-- Represents the number of calories John eats per day -/
def calories_eaten_per_day : ℕ := 1800

/-- Represents the number of calories needed to be burned to lose 1 pound -/
def calories_per_pound : ℕ := 4000

/-- Represents the number of days it takes John to lose 10 pounds -/
def days_to_lose_10_pounds : ℕ := 80

/-- Represents the number of pounds John wants to lose -/
def pounds_to_lose : ℕ := 10

theorem john_calorie_burn :
  calories_burned_per_day = 
    calories_eaten_per_day + 
    (pounds_to_lose * calories_per_pound) / days_to_lose_10_pounds :=
by
  sorry

end NUMINAMATH_CALUDE_john_calorie_burn_l1638_163802


namespace NUMINAMATH_CALUDE_duty_shoes_price_l1638_163803

def full_price : ℝ → Prop :=
  λ price => 
    let discount1 := 0.2
    let discount2 := 0.25
    let price_after_discount1 := price * (1 - discount1)
    let price_after_discount2 := price_after_discount1 * (1 - discount2)
    price_after_discount2 = 51

theorem duty_shoes_price : ∃ (price : ℝ), full_price price ∧ price = 85 := by
  sorry

end NUMINAMATH_CALUDE_duty_shoes_price_l1638_163803


namespace NUMINAMATH_CALUDE_total_length_of_T_l1638_163807

/-- The set T of points (x, y) in the Cartesian plane -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ‖‖|p.1| - 3‖ - 2‖ + ‖‖|p.2| - 3‖ - 2‖ = 2}

/-- The total length of all lines forming the set T -/
def total_length (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem stating that the total length of lines forming T is 64√2 -/
theorem total_length_of_T : total_length T = 64 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_total_length_of_T_l1638_163807


namespace NUMINAMATH_CALUDE_band_repertoire_proof_l1638_163800

theorem band_repertoire_proof (total_songs : ℕ) (second_set : ℕ) (encore : ℕ) (avg_third_fourth : ℕ) :
  total_songs = 30 →
  second_set = 7 →
  encore = 2 →
  avg_third_fourth = 8 →
  ∃ (first_set : ℕ), first_set + second_set + encore + 2 * avg_third_fourth = total_songs ∧ first_set = 5 := by
  sorry

end NUMINAMATH_CALUDE_band_repertoire_proof_l1638_163800


namespace NUMINAMATH_CALUDE_rectangle_breadth_l1638_163877

theorem rectangle_breadth (L B : ℝ) (h1 : L / B = 25 / 16) (h2 : L * B = 200 * 200) : B = 160 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l1638_163877


namespace NUMINAMATH_CALUDE_shorter_base_length_l1638_163895

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  midpoint_segment : ℝ

/-- Calculates the length of the shorter base of a trapezoid -/
def shorter_base (t : Trapezoid) : ℝ :=
  t.long_base - 2 * t.midpoint_segment

/-- Theorem stating the length of the shorter base for a specific trapezoid -/
theorem shorter_base_length (t : Trapezoid) 
  (h1 : t.long_base = 102) 
  (h2 : t.midpoint_segment = 5) : 
  shorter_base t = 92 := by
  sorry

end NUMINAMATH_CALUDE_shorter_base_length_l1638_163895


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l1638_163825

/-- The number of dogwood trees in the park after planting and accounting for losses -/
def final_tree_count (initial_trees : ℕ) 
                     (worker_a_trees : ℕ) 
                     (worker_b_trees : ℕ) 
                     (worker_c_trees : ℕ) 
                     (worker_d_trees : ℕ) 
                     (worker_e_trees : ℕ) 
                     (worker_c_losses : ℕ) 
                     (worker_d_losses : ℕ) : ℕ :=
  initial_trees + 
  worker_a_trees + 
  worker_b_trees + 
  (worker_c_trees - worker_c_losses) + 
  (worker_d_trees - worker_d_losses) + 
  worker_e_trees

/-- Theorem stating that the final number of dogwood trees in the park is 80 -/
theorem dogwood_tree_count : 
  final_tree_count 34 12 10 15 8 4 2 1 = 80 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l1638_163825


namespace NUMINAMATH_CALUDE_normal_distribution_estimate_l1638_163864

/-- Represents the normal distribution function -/
noncomputable def normal_dist (μ σ : ℝ) (x : ℝ) : ℝ := sorry

/-- Represents the cumulative distribution function of a normal distribution -/
noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

/-- The probability that a value from N(μ, σ²) falls within μ ± σ -/
axiom normal_prob_within_1sigma (μ σ : ℝ) : 
  normal_cdf μ σ (μ + σ) - normal_cdf μ σ (μ - σ) = 0.6826

/-- The probability that a value from N(μ, σ²) falls within μ ± 2σ -/
axiom normal_prob_within_2sigma (μ σ : ℝ) : 
  normal_cdf μ σ (μ + 2*σ) - normal_cdf μ σ (μ - 2*σ) = 0.9544

/-- The theorem to be proved -/
theorem normal_distribution_estimate : 
  let μ : ℝ := 70
  let σ : ℝ := 5
  let sample_size : ℕ := 100000
  let lower_bound : ℝ := 75
  let upper_bound : ℝ := 80
  ⌊(normal_cdf μ σ upper_bound - normal_cdf μ σ lower_bound) * sample_size⌋ = 13590 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_estimate_l1638_163864


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_perpendicular_lines_from_parallel_planes_l1638_163892

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_lines_from_perpendicular_planes 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  perpendicular n β → 
  perpendicular_planes α β → 
  perpendicular_lines m n :=
sorry

-- Theorem 2
theorem perpendicular_lines_from_parallel_planes 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel n β → 
  parallel_planes α β → 
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_perpendicular_lines_from_parallel_planes_l1638_163892


namespace NUMINAMATH_CALUDE_lasso_probability_l1638_163838

theorem lasso_probability (p : ℝ) (n : ℕ) (h1 : p = 1 / 2) (h2 : n = 4) :
  1 - (1 - p) ^ n = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_lasso_probability_l1638_163838


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1638_163830

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_first_term 
  (seq : ArithmeticSequence)
  (h1 : seq.a 2 * seq.a 3 = seq.a 4 * seq.a 5)
  (h2 : sum_n seq 4 = 27) :
  seq.a 1 = 135 / 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1638_163830


namespace NUMINAMATH_CALUDE_middle_number_proof_l1638_163818

theorem middle_number_proof (x y z : ℕ) : 
  x < y ∧ y < z →
  x + y = 18 →
  x + z = 23 →
  y + z = 27 →
  y = 11 := by
sorry

end NUMINAMATH_CALUDE_middle_number_proof_l1638_163818


namespace NUMINAMATH_CALUDE_no_special_subset_exists_l1638_163862

theorem no_special_subset_exists : ¬∃ M : Set ℕ,
  (∀ n : ℕ, n > 1 → ∃ a b : ℕ, a ∈ M ∧ b ∈ M ∧ n = a + b) ∧
  (∀ a b c d : ℕ, a ∈ M → b ∈ M → c ∈ M → d ∈ M →
    a > 10 → b > 10 → c > 10 → d > 10 →
    ((a + b = c + d) ↔ (a = c ∨ a = d))) :=
by sorry

end NUMINAMATH_CALUDE_no_special_subset_exists_l1638_163862


namespace NUMINAMATH_CALUDE_complement_A_in_U_l1638_163856

-- Define the set U
def U : Set ℝ := {y | ∃ x : ℝ, y = 2^x ∧ x ≥ -1}

-- Define the set A
def A : Set ℝ := {x | (x - 2) * (x - 1) < 0}

-- State the theorem
theorem complement_A_in_U : 
  (U \ A) = {x | x ∈ Set.Icc (1/2 : ℝ) 1 ∨ x ∈ Set.Ici 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l1638_163856


namespace NUMINAMATH_CALUDE_tori_test_score_l1638_163890

theorem tori_test_score (total : ℕ) (arithmetic : ℕ) (algebra : ℕ) (geometry : ℕ)
  (arithmetic_correct : ℚ) (algebra_correct : ℚ) (geometry_correct : ℚ)
  (passing_grade : ℚ) :
  total = 100 →
  arithmetic = 20 →
  algebra = 40 →
  geometry = 40 →
  arithmetic_correct = 4/5 →
  algebra_correct = 1/2 →
  geometry_correct = 7/10 →
  passing_grade = 13/20 →
  ↑⌈passing_grade * total⌉ - (↑⌊arithmetic_correct * arithmetic⌋ + 
    ↑⌊algebra_correct * algebra⌋ + ↑⌊geometry_correct * geometry⌋) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tori_test_score_l1638_163890


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1638_163832

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  sum_formula : ∀ n : ℕ+, S n = (n : ℝ) * (a 1 + a n) / 2
  arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_properties
  (seq : ArithmeticSequence)
  (h : seq.S 5 > seq.S 6 ∧ seq.S 6 > seq.S 4) :
  common_difference seq < 0 ∧ seq.S 10 > 0 ∧ seq.S 11 < 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1638_163832


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l1638_163879

/-- The radius of a sphere tangent to a truncated cone -/
theorem sphere_radius_in_truncated_cone (r₁ r₂ : ℝ) (hr₁ : r₁ = 25) (hr₂ : r₂ = 7) :
  let h := Real.sqrt ((r₁ + r₂)^2 - (r₁ - r₂)^2)
  (h / 2 : ℝ) = 5 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l1638_163879


namespace NUMINAMATH_CALUDE_product_of_numbers_l1638_163806

theorem product_of_numbers (x y : ℝ) : 
  |x - y| = 11 → x^2 + y^2 = 221 → x * y = 60 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1638_163806


namespace NUMINAMATH_CALUDE_inequality_proof_l1638_163817

theorem inequality_proof (a b c d : ℝ) (h : a * d - b * c = 1) :
  a^2 + b^2 + c^2 + d^2 + a * b + c * d ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1638_163817


namespace NUMINAMATH_CALUDE_equivalent_angle_sets_l1638_163894

def angle_set (base : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 360 + base}

theorem equivalent_angle_sets :
  angle_set (-437) = angle_set 283 :=
sorry

end NUMINAMATH_CALUDE_equivalent_angle_sets_l1638_163894


namespace NUMINAMATH_CALUDE_chris_newspaper_collection_l1638_163885

theorem chris_newspaper_collection (chris_newspapers lily_newspapers : ℕ) : 
  lily_newspapers = chris_newspapers + 23 →
  chris_newspapers + lily_newspapers = 65 →
  chris_newspapers = 21 := by
sorry

end NUMINAMATH_CALUDE_chris_newspaper_collection_l1638_163885


namespace NUMINAMATH_CALUDE_partnership_gain_l1638_163808

/-- Represents the investment and profit of a partnership --/
structure Partnership where
  x : ℝ  -- A's investment
  a_share : ℝ  -- A's share of the profit
  total_gain : ℝ  -- Total annual gain

/-- Calculates the total annual gain of the partnership --/
def calculate_total_gain (p : Partnership) : ℝ :=
  3 * p.a_share

/-- Theorem stating that given the investment conditions and A's share, 
    the total annual gain is 12000 --/
theorem partnership_gain (p : Partnership) 
  (h1 : p.x > 0)  -- A's investment is positive
  (h2 : p.a_share = 4000)  -- A's share is 4000
  : p.total_gain = 12000 :=
by
  sorry

#check partnership_gain

end NUMINAMATH_CALUDE_partnership_gain_l1638_163808


namespace NUMINAMATH_CALUDE_borrowed_amounts_proof_l1638_163861

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem borrowed_amounts_proof 
  (interest₁ interest₂ interest₃ : ℝ)
  (rate₁ rate₂ rate₃ : ℝ)
  (time₁ time₂ time₃ : ℝ)
  (h₁ : interest₁ = 1500)
  (h₂ : interest₂ = 1500)
  (h₃ : interest₃ = 1500)
  (hr₁ : rate₁ = 0.12)
  (hr₂ : rate₂ = 0.10)
  (hr₃ : rate₃ = 0.05)
  (ht₁ : time₁ = 1)
  (ht₂ : time₂ = 2)
  (ht₃ : time₃ = 3) :
  ∃ (principal₁ principal₂ principal₃ : ℝ),
    simple_interest principal₁ rate₁ time₁ = interest₁ ∧
    simple_interest principal₂ rate₂ time₂ = interest₂ ∧
    simple_interest principal₃ rate₃ time₃ = interest₃ ∧
    principal₁ = 12500 ∧
    principal₂ = 7500 ∧
    principal₃ = 10000 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_amounts_proof_l1638_163861


namespace NUMINAMATH_CALUDE_winning_team_fourth_quarter_points_l1638_163846

theorem winning_team_fourth_quarter_points :
  ∀ (first_quarter second_quarter third_quarter fourth_quarter : ℕ),
  let losing_team_first_quarter := 10
  let total_points := 80
  first_quarter = 2 * losing_team_first_quarter →
  second_quarter = first_quarter + 10 →
  third_quarter = second_quarter + 20 →
  fourth_quarter = total_points - (first_quarter + second_quarter + third_quarter) →
  fourth_quarter = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_winning_team_fourth_quarter_points_l1638_163846


namespace NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l1638_163852

theorem cos_squared_alpha_minus_pi_fourth (α : Real) 
  (h : Real.sin (2 * α) = 1/3) : 
  Real.cos (α - Real.pi/4)^2 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l1638_163852


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l1638_163896

/-- A rectangular prism with dimensions a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallelEdgePairs (prism : RectangularPrism) : ℕ := 12

/-- Theorem: A rectangular prism has 12 pairs of parallel edges -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  parallelEdgePairs prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l1638_163896


namespace NUMINAMATH_CALUDE_trig_sum_equals_three_halves_l1638_163855

theorem trig_sum_equals_three_halves :
  let α := 5 * π / 24
  let β := 11 * π / 24
  Real.cos α ^ 4 + Real.cos β ^ 4 + Real.sin (π - α) ^ 4 + Real.sin (π - β) ^ 4 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_three_halves_l1638_163855


namespace NUMINAMATH_CALUDE_geometric_sequence_relation_l1638_163865

/-- Given a geometric sequence {a_n} with common ratio q ≠ 1,
    if a_m, a_n, a_p form a geometric sequence in that order,
    then 2n = m + p, where m, n, and p are natural numbers. -/
theorem geometric_sequence_relation (a : ℕ → ℝ) (q : ℝ) (m n p : ℕ) :
  (∀ k, a (k + 1) = q * a k) →  -- geometric sequence condition
  q ≠ 1 →                       -- common ratio ≠ 1
  (a n)^2 = a m * a p →         -- a_m, a_n, a_p form a geometric sequence
  2 * n = m + p :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_relation_l1638_163865


namespace NUMINAMATH_CALUDE_books_loaned_out_l1638_163819

/-- Proves that the number of books loaned out is 30 given the initial count, return rate, and final count -/
theorem books_loaned_out 
  (initial_count : ℕ) 
  (return_rate : ℚ) 
  (final_count : ℕ) 
  (h1 : initial_count = 75)
  (h2 : return_rate = 4/5)
  (h3 : final_count = 69) :
  (initial_count - final_count : ℚ) / (1 - return_rate) = 30 := by
sorry

end NUMINAMATH_CALUDE_books_loaned_out_l1638_163819


namespace NUMINAMATH_CALUDE_loan_sum_calculation_l1638_163886

/-- Proves that a sum P lent at 6% simple interest per annum for 8 years, 
    where the interest is $572 less than P, equals $1100. -/
theorem loan_sum_calculation (P : ℝ) : 
  (P * 0.06 * 8 = P - 572) → P = 1100 := by
  sorry

end NUMINAMATH_CALUDE_loan_sum_calculation_l1638_163886


namespace NUMINAMATH_CALUDE_at_least_one_geq_one_l1638_163829

theorem at_least_one_geq_one (x y : ℝ) (h : x + y ≥ 2) : max x y ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_one_l1638_163829


namespace NUMINAMATH_CALUDE_product_72_difference_sum_l1638_163869

theorem product_72_difference_sum (A B C D : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 →
  A * B = 72 →
  C * D = 72 →
  A - B = C + D + 2 →
  A = 6 := by
sorry

end NUMINAMATH_CALUDE_product_72_difference_sum_l1638_163869


namespace NUMINAMATH_CALUDE_cubic_sum_l1638_163843

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) :
  x^3 + y^3 = 35 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_l1638_163843


namespace NUMINAMATH_CALUDE_least_integer_abs_inequality_l1638_163872

theorem least_integer_abs_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), |3*y + 4| ≤ 25 → x ≤ y) ∧ |3*x + 4| ≤ 25 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_integer_abs_inequality_l1638_163872


namespace NUMINAMATH_CALUDE_girls_without_notebooks_l1638_163815

theorem girls_without_notebooks (total_girls : Nat) (students_with_notebooks : Nat) (boys_with_notebooks : Nat) 
  (h1 : total_girls = 20)
  (h2 : students_with_notebooks = 25)
  (h3 : boys_with_notebooks = 16) :
  total_girls - (students_with_notebooks - boys_with_notebooks) = 11 := by
  sorry

end NUMINAMATH_CALUDE_girls_without_notebooks_l1638_163815


namespace NUMINAMATH_CALUDE_vegetarians_count_l1638_163893

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  only_veg : ℕ
  only_non_veg : ℕ
  both : ℕ

/-- Calculates the total number of people who eat vegetarian food in the family -/
def total_vegetarians (f : FamilyDiet) : ℕ :=
  f.only_veg + f.both

/-- Theorem stating that the number of vegetarians in the given family is 31 -/
theorem vegetarians_count (f : FamilyDiet) 
  (h1 : f.only_veg = 19)
  (h2 : f.only_non_veg = 9)
  (h3 : f.both = 12) :
  total_vegetarians f = 31 := by
  sorry

#eval total_vegetarians ⟨19, 9, 12⟩

end NUMINAMATH_CALUDE_vegetarians_count_l1638_163893


namespace NUMINAMATH_CALUDE_bound_on_expression_l1638_163849

theorem bound_on_expression (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) :
  -15 ≤ x - 2*y + 2*z ∧ x - 2*y + 2*z ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_bound_on_expression_l1638_163849


namespace NUMINAMATH_CALUDE_solution_set_for_a_4_range_of_a_for_f_leq_4_l1638_163824

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for part (1)
theorem solution_set_for_a_4 :
  {x : ℝ | f x 4 ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_f_leq_4 :
  {a : ℝ | ∃ x, f x a ≤ 4} = {a : ℝ | -3 ≤ a ∧ a ≤ 5} := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_4_range_of_a_for_f_leq_4_l1638_163824


namespace NUMINAMATH_CALUDE_alla_boris_meeting_l1638_163801

/-- Represents the meeting point of two people walking along a line of lamps. -/
def meetingPoint (totalLamps : ℕ) (allaPos : ℕ) (borisPos : ℕ) : ℕ :=
  let intervalsCovered := (allaPos - 1) + (totalLamps - borisPos)
  let totalIntervals := totalLamps - 1
  let meetingInterval := intervalsCovered * 3
  1 + meetingInterval

/-- Theorem stating the meeting point of Alla and Boris. -/
theorem alla_boris_meeting :
  meetingPoint 400 55 321 = 163 := by
  sorry

end NUMINAMATH_CALUDE_alla_boris_meeting_l1638_163801


namespace NUMINAMATH_CALUDE_probability_of_queen_in_standard_deck_l1638_163810

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (queens : ℕ)

/-- Calculates the probability of drawing a Queen from a given deck -/
def probability_of_queen (d : Deck) : ℚ :=
  d.queens / d.total_cards

/-- Theorem stating the probability of drawing a Queen from a standard deck -/
theorem probability_of_queen_in_standard_deck :
  ∃ (d : Deck), d.total_cards = 52 ∧ d.queens = 4 ∧ probability_of_queen d = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_queen_in_standard_deck_l1638_163810


namespace NUMINAMATH_CALUDE_friend_ratio_proof_l1638_163882

theorem friend_ratio_proof (julian_total : ℕ) (julian_boy_percent : ℚ) (julian_girl_percent : ℚ)
  (boyd_total : ℕ) (boyd_boy_percent : ℚ) :
  julian_total = 80 →
  julian_boy_percent = 60 / 100 →
  julian_girl_percent = 40 / 100 →
  boyd_total = 100 →
  boyd_boy_percent = 36 / 100 →
  (boyd_total - boyd_total * boyd_boy_percent) / (julian_total * julian_girl_percent) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_friend_ratio_proof_l1638_163882


namespace NUMINAMATH_CALUDE_number_reciprocal_problem_l1638_163899

theorem number_reciprocal_problem : ∃ x : ℚ, (1 + 1 / x = 5 / 2) ∧ (x = 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_number_reciprocal_problem_l1638_163899


namespace NUMINAMATH_CALUDE_fixed_cost_calculation_l1638_163823

/-- Represents the fixed cost to run the molding machine per week -/
def fixed_cost : ℝ := 7640

/-- Represents the cost to mold each handle -/
def variable_cost : ℝ := 0.60

/-- Represents the selling price per handle -/
def selling_price : ℝ := 4.60

/-- Represents the break-even point in number of handles per week -/
def break_even_point : ℝ := 1910

/-- Proves that the fixed cost is correct given the other parameters -/
theorem fixed_cost_calculation :
  fixed_cost = break_even_point * (selling_price - variable_cost) :=
by sorry

end NUMINAMATH_CALUDE_fixed_cost_calculation_l1638_163823


namespace NUMINAMATH_CALUDE_trig_identity_l1638_163847

theorem trig_identity (x y : ℝ) :
  Real.sin (x + y) * Real.cos (2 * y) + Real.cos (x + y) * Real.sin (2 * y) = Real.sin (x + 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1638_163847


namespace NUMINAMATH_CALUDE_pascals_triangle_50th_number_l1638_163891

theorem pascals_triangle_50th_number (n : ℕ) (h : n + 1 = 52) : 
  Nat.choose n 49 = 1275 := by
  sorry

end NUMINAMATH_CALUDE_pascals_triangle_50th_number_l1638_163891


namespace NUMINAMATH_CALUDE_equation_solution_l1638_163836

theorem equation_solution : ∃ x : ℝ, (24 - 6 = 3 * x + 3) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1638_163836


namespace NUMINAMATH_CALUDE_fraction_of_roots_equals_exponents_l1638_163881

theorem fraction_of_roots_equals_exponents (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 * b)^(1/2) / (a * b)^(1/3) = a^(7/6) * b^(1/6) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_roots_equals_exponents_l1638_163881


namespace NUMINAMATH_CALUDE_score_difference_l1638_163884

theorem score_difference (hajar_score : ℕ) (total_score : ℕ) : 
  hajar_score = 24 →
  total_score = 69 →
  ∃ (farah_score : ℕ),
    farah_score > hajar_score ∧
    farah_score + hajar_score = total_score ∧
    farah_score - hajar_score = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_score_difference_l1638_163884


namespace NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l1638_163828

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : 
  a / (1 / b) = a * b :=
by sorry

theorem twelve_divided_by_one_sixth : 
  12 / (1 / 6) = 72 :=
by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l1638_163828


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l1638_163874

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₃ + a₅ + a₇ = 8256 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l1638_163874


namespace NUMINAMATH_CALUDE_sequence_last_term_l1638_163837

theorem sequence_last_term (n : ℕ) : 2^n - 1 = 127 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_sequence_last_term_l1638_163837


namespace NUMINAMATH_CALUDE_computer_from_syllables_l1638_163845

/-- Represents a syllable in a word --/
structure Syllable where
  value : String

/-- Represents a word composed of syllables --/
def Word := List Syllable

/-- Function to combine syllables into a word --/
def combineWord (syllables : Word) : String :=
  String.join (syllables.map (λ s => s.value))

/-- Theorem: Given the specific syllables, the resulting word is "компьютер" --/
theorem computer_from_syllables (s1 s2 s3 : Syllable) 
  (h1 : s1.value = "ком")  -- First syllable: A big piece of a snowman
  (h2 : s2.value = "пьют") -- Second syllable: Something done by elephants at watering hole
  (h3 : s3.value = "ер")   -- Third syllable: Called as the hard sign used to be called
  : combineWord [s1, s2, s3] = "компьютер" := by
  sorry

/-- The word has exactly three syllables --/
axiom word_has_three_syllables (w : Word) : w.length = 3

#check computer_from_syllables
#check word_has_three_syllables

end NUMINAMATH_CALUDE_computer_from_syllables_l1638_163845


namespace NUMINAMATH_CALUDE_unique_divisibility_pair_l1638_163841

theorem unique_divisibility_pair : ∀ (a b : ℕ+),
  (∃ k : ℤ, (4 * b.val - 1) = k * (3 * a.val + 1)) →
  (∃ m : ℤ, (3 * a.val - 1) = m * (2 * b.val + 1)) →
  a.val = 2 ∧ b.val = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_divisibility_pair_l1638_163841


namespace NUMINAMATH_CALUDE_triangle_inequality_l1638_163883

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  A > 0 → B > 0 → C > 0 → 
  A + B + C = π →
  (a + b) * Real.sin (C / 2) + (b + c) * Real.sin (A / 2) + (c + a) * Real.sin (B / 2) ≤ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1638_163883


namespace NUMINAMATH_CALUDE_paper_stack_height_l1638_163842

/-- Given a ream of paper with 500 sheets and 5 cm thickness, 
    prove that a 7.5 cm stack contains 750 sheets -/
theorem paper_stack_height (sheets_per_ream : ℕ) (ream_thickness : ℝ) (stack_height : ℝ) :
  sheets_per_ream = 500 →
  ream_thickness = 5 →
  stack_height = 7.5 →
  (stack_height / ream_thickness * sheets_per_ream : ℝ) = 750 := by
  sorry

#check paper_stack_height

end NUMINAMATH_CALUDE_paper_stack_height_l1638_163842


namespace NUMINAMATH_CALUDE_equation_solution_l1638_163888

theorem equation_solution : ∃ x : ℤ, (158 - x = 59) ∧ (x = 99) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1638_163888


namespace NUMINAMATH_CALUDE_expression_a_equality_l1638_163850

theorem expression_a_equality : 7 * (2/3) + 16 * (5/12) = 11 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_a_equality_l1638_163850


namespace NUMINAMATH_CALUDE_max_value_problem_l1638_163814

theorem max_value_problem (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 2) :
  x^3 * y^2 * z^4 ≤ 13824 / 40353607 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), 0 < x₀ ∧ 0 < y₀ ∧ 0 < z₀ ∧ x₀ + y₀ + z₀ = 2 ∧ x₀^3 * y₀^2 * z₀^4 = 13824 / 40353607 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l1638_163814


namespace NUMINAMATH_CALUDE_vector_at_zero_l1638_163809

/-- A line parameterized by t in 3D space -/
structure ParametricLine where
  point : ℝ → ℝ × ℝ × ℝ

/-- The given line satisfies the conditions -/
def given_line : ParametricLine where
  point := λ t => sorry

theorem vector_at_zero (l : ParametricLine)
  (h1 : l.point (-2) = (2, 6, 16))
  (h2 : l.point 1 = (-1, -4, -10)) :
  l.point 0 = (0, 2/3, 16/3) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_zero_l1638_163809


namespace NUMINAMATH_CALUDE_toys_per_day_l1638_163859

/-- A factory produces toys under the following conditions:
  * The factory produces 3400 toys per week.
  * The workers work 5 days a week.
  * The same number of toys is made every day. -/
def toy_factory (total_toys : ℕ) (work_days : ℕ) (daily_production : ℕ) : Prop :=
  total_toys = 3400 ∧ work_days = 5 ∧ daily_production * work_days = total_toys

/-- The number of toys produced each day is 680. -/
theorem toys_per_day : 
  ∀ (total_toys work_days daily_production : ℕ), 
  toy_factory total_toys work_days daily_production → daily_production = 680 :=
sorry

end NUMINAMATH_CALUDE_toys_per_day_l1638_163859


namespace NUMINAMATH_CALUDE_tom_bonus_points_l1638_163897

/-- The number of bonus points earned by an employee in a day --/
def bonus_points (customers_per_hour : ℕ) (hours_worked : ℕ) : ℕ :=
  (customers_per_hour * hours_worked * 20) / 100

/-- Theorem: Tom earned 16 bonus points on Monday --/
theorem tom_bonus_points : bonus_points 10 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_bonus_points_l1638_163897


namespace NUMINAMATH_CALUDE_two_a_div_a_equals_two_l1638_163826

theorem two_a_div_a_equals_two (a : ℝ) (h : a ≠ 0) : 2 * a / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_a_div_a_equals_two_l1638_163826


namespace NUMINAMATH_CALUDE_f_properties_l1638_163870

def f (x : ℝ) : ℝ := |x| + 2

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1638_163870


namespace NUMINAMATH_CALUDE_range_of_a_l1638_163839

def S : Set ℝ := {x : ℝ | (x - 2)^2 > 9}
def T (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a + 8}

theorem range_of_a : ∀ a : ℝ, (S ∪ T a = Set.univ) ↔ (-3 < a ∧ a < -1) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1638_163839


namespace NUMINAMATH_CALUDE_pairwise_products_sum_l1638_163848

theorem pairwise_products_sum (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 560) 
  (h2 : a + b + c = 24) : 
  a*b + b*c + c*a = 8 := by
sorry

end NUMINAMATH_CALUDE_pairwise_products_sum_l1638_163848


namespace NUMINAMATH_CALUDE_binary_to_base_4_conversion_l1638_163873

-- Define the binary number
def binary_num : ℕ := 110110001

-- Define the base 4 number
def base_4_num : ℕ := 13201

-- Theorem statement
theorem binary_to_base_4_conversion :
  (binary_num : ℕ).digits 2 = [1, 1, 0, 1, 1, 0, 0, 0, 1] ∧
  (base_4_num : ℕ).digits 4 = [1, 3, 2, 0, 1] ∧
  binary_num = base_4_num :=
by sorry

end NUMINAMATH_CALUDE_binary_to_base_4_conversion_l1638_163873


namespace NUMINAMATH_CALUDE_system_solution_l1638_163822

theorem system_solution (x y : ℝ) (h1 : x + 2*y = -1) (h2 : 2*x + y = 3) : x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1638_163822


namespace NUMINAMATH_CALUDE_efficiency_increase_l1638_163834

theorem efficiency_increase (days_sakshi days_tanya : ℝ) 
  (h1 : days_sakshi = 20) 
  (h2 : days_tanya = 16) : 
  (1 / days_tanya - 1 / days_sakshi) / (1 / days_sakshi) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_efficiency_increase_l1638_163834


namespace NUMINAMATH_CALUDE_tiling_polygons_are_3_4_6_l1638_163811

/-- A regular polygon can tile the plane if there exists a positive integer number of polygons that can meet at a vertex to form a complete 360° angle. -/
def can_tile (n : ℕ) : Prop :=
  n > 2 ∧ ∃ x : ℕ, x > 0 ∧ x * ((n - 2) * 180 / n) = 360

/-- The set of numbers of sides for regular polygons that can tile the plane. -/
def tiling_polygons : Set ℕ := {n : ℕ | can_tile n}

/-- Theorem stating that the only regular polygons that can tile the plane have 3, 4, or 6 sides. -/
theorem tiling_polygons_are_3_4_6 : tiling_polygons = {3, 4, 6} := by sorry

end NUMINAMATH_CALUDE_tiling_polygons_are_3_4_6_l1638_163811


namespace NUMINAMATH_CALUDE_quartic_roots_max_value_l1638_163867

theorem quartic_roots_max_value (a b d : ℝ) (x₁ x₂ x₃ x₄ : ℝ) :
  (1/2 ≤ x₁ ∧ x₁ ≤ 2) →
  (1/2 ≤ x₂ ∧ x₂ ≤ 2) →
  (1/2 ≤ x₃ ∧ x₃ ≤ 2) →
  (1/2 ≤ x₄ ∧ x₄ ≤ 2) →
  x₁^4 - a*x₁^3 + b*x₁^2 - a*x₁ + d = 0 →
  x₂^4 - a*x₂^3 + b*x₂^2 - a*x₂ + d = 0 →
  x₃^4 - a*x₃^3 + b*x₃^2 - a*x₃ + d = 0 →
  x₄^4 - a*x₄^3 + b*x₄^2 - a*x₄ + d = 0 →
  (x₁ + x₂) * (x₁ + x₃) * x₄ / ((x₄ + x₂) * (x₄ + x₃) * x₁) ≤ 5/4 :=
by sorry

end NUMINAMATH_CALUDE_quartic_roots_max_value_l1638_163867


namespace NUMINAMATH_CALUDE_equation_solution_l1638_163820

theorem equation_solution (A : ℕ+) : 
  (∃! (x₁ y₁ x₂ y₂ : ℕ+), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ 
    A * x₁ + 10 * y₁ = 100 ∧ A * x₂ + 10 * y₂ = 100) ↔ A = 10 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1638_163820


namespace NUMINAMATH_CALUDE_cube_root_of_5488000_l1638_163813

theorem cube_root_of_5488000 : (5488000 : ℝ) ^ (1/3 : ℝ) = 40 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_5488000_l1638_163813


namespace NUMINAMATH_CALUDE_triangle_side_length_l1638_163805

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  b = 4 →
  B = π / 6 →
  Real.sin A = 1 / 3 →
  a = 8 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1638_163805


namespace NUMINAMATH_CALUDE_existence_of_complex_root_l1638_163835

theorem existence_of_complex_root (n : ℕ) (A : Finset ℕ) (hn : n ≥ 2) (hA : A.card = n) :
  ∃ z : ℂ, Complex.abs z = 1 ∧ Complex.abs (A.sum (λ a => z^a)) = Real.sqrt (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_complex_root_l1638_163835


namespace NUMINAMATH_CALUDE_square_minus_self_divisible_by_two_l1638_163816

theorem square_minus_self_divisible_by_two (a : ℤ) : ∃ k : ℤ, a^2 - a = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_square_minus_self_divisible_by_two_l1638_163816


namespace NUMINAMATH_CALUDE_gus_egg_consumption_l1638_163875

/-- The number of eggs Gus ate for breakfast -/
def breakfast_eggs : ℕ := 2

/-- The number of eggs Gus ate for lunch -/
def lunch_eggs : ℕ := 3

/-- The number of eggs Gus ate for dinner -/
def dinner_eggs : ℕ := 1

/-- The total number of eggs Gus ate throughout the day -/
def total_eggs : ℕ := breakfast_eggs + lunch_eggs + dinner_eggs

theorem gus_egg_consumption : total_eggs = 6 := by
  sorry

end NUMINAMATH_CALUDE_gus_egg_consumption_l1638_163875


namespace NUMINAMATH_CALUDE_one_by_one_square_position_l1638_163833

/-- A square on a grid --/
structure Square where
  size : ℕ
  row : ℕ
  col : ℕ

/-- A decomposition of a large square into smaller squares --/
structure Decomposition where
  grid_size : ℕ
  squares : List Square

/-- Predicate to check if a decomposition is valid --/
def is_valid_decomposition (d : Decomposition) : Prop :=
  d.grid_size = 23 ∧
  (∀ s ∈ d.squares, s.size ∈ [1, 2, 3]) ∧
  (d.squares.filter (λ s => s.size = 1)).length = 1

/-- Predicate to check if a position is valid for the 1x1 square --/
def is_valid_position (row col : ℕ) : Prop :=
  row % 6 = 0 ∧ col % 6 = 0 ∧ row ≤ 18 ∧ col ≤ 18

theorem one_by_one_square_position (d : Decomposition) 
  (h : is_valid_decomposition d) :
  ∃ s ∈ d.squares, s.size = 1 ∧ is_valid_position s.row s.col :=
sorry

end NUMINAMATH_CALUDE_one_by_one_square_position_l1638_163833


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1638_163857

/-- Proves that if an article is sold for 250 Rs. with a 25% profit, its cost price is 200 Rs. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 250)
  (h2 : profit_percentage = 25) :
  selling_price / (1 + profit_percentage / 100) = 200 :=
by
  sorry

#check cost_price_calculation

end NUMINAMATH_CALUDE_cost_price_calculation_l1638_163857


namespace NUMINAMATH_CALUDE_product_of_base8_digits_8670_l1638_163868

/-- The base 8 representation of a natural number -/
def base8Representation (n : ℕ) : List ℕ :=
  sorry

/-- The product of a list of natural numbers -/
def listProduct (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 8 representation of 8670 is 0 -/
theorem product_of_base8_digits_8670 :
  listProduct (base8Representation 8670) = 0 :=
sorry

end NUMINAMATH_CALUDE_product_of_base8_digits_8670_l1638_163868


namespace NUMINAMATH_CALUDE_one_beaver_still_working_l1638_163876

/-- The number of beavers still working on their home -/
def beavers_still_working (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

/-- Proof that 1 beaver is still working on their home -/
theorem one_beaver_still_working :
  beavers_still_working 2 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_beaver_still_working_l1638_163876


namespace NUMINAMATH_CALUDE_exam_failure_rate_l1638_163844

theorem exam_failure_rate (total : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ) :
  total = 100 →
  failed_hindi = 20 →
  failed_both = 10 →
  passed_both = 20 →
  ∃ failed_english : ℝ, failed_english = 70 :=
by sorry

end NUMINAMATH_CALUDE_exam_failure_rate_l1638_163844


namespace NUMINAMATH_CALUDE_lee_lawn_mowing_l1638_163880

/-- The number of lawns Lee mowed last week -/
def num_lawns : ℕ := 16

/-- The price Lee charges for mowing one lawn -/
def price_per_lawn : ℕ := 33

/-- The number of customers who gave Lee a tip -/
def num_tips : ℕ := 3

/-- The amount of each tip -/
def tip_amount : ℕ := 10

/-- Lee's total earnings last week -/
def total_earnings : ℕ := 558

theorem lee_lawn_mowing :
  num_lawns * price_per_lawn + num_tips * tip_amount = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_lee_lawn_mowing_l1638_163880


namespace NUMINAMATH_CALUDE_number_difference_l1638_163866

theorem number_difference (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a / b = 2 / 3) (h4 : a^3 + b^3 = 945) : b - a = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1638_163866


namespace NUMINAMATH_CALUDE_correct_hardbacks_verify_selections_l1638_163878

def total_books : ℕ := 8
def paperbacks : ℕ := 2
def selections_with_paperback : ℕ := 36

def hardbacks : ℕ := total_books - paperbacks

theorem correct_hardbacks : hardbacks = 6 := by sorry

theorem verify_selections :
  Nat.choose total_books 3 - Nat.choose hardbacks 3 = selections_with_paperback := by sorry

end NUMINAMATH_CALUDE_correct_hardbacks_verify_selections_l1638_163878


namespace NUMINAMATH_CALUDE_jack_birth_year_l1638_163827

def first_amc8_year : ℕ := 1990

def jack_age_at_ninth_amc8 : ℕ := 15

def ninth_amc8_year : ℕ := first_amc8_year + 8

theorem jack_birth_year :
  first_amc8_year = 1990 →
  jack_age_at_ninth_amc8 = 15 →
  ninth_amc8_year - jack_age_at_ninth_amc8 = 1983 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_birth_year_l1638_163827


namespace NUMINAMATH_CALUDE_rectangle_ribbon_length_l1638_163840

/-- The length of ribbon needed to form a rectangle -/
def ribbon_length (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: The length of ribbon needed to form a rectangle with length 20 feet and width 15 feet is 70 feet -/
theorem rectangle_ribbon_length : 
  ribbon_length 20 15 = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ribbon_length_l1638_163840
