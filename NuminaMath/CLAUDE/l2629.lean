import Mathlib

namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l2629_262947

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) (girls boys : ℕ) : 
  total = 30 →
  difference = 6 →
  girls = boys + difference →
  total = girls + boys →
  (girls : ℚ) / (boys : ℚ) = 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l2629_262947


namespace NUMINAMATH_CALUDE_cupcake_distribution_l2629_262993

theorem cupcake_distribution (total_cupcakes : ℕ) (num_children : ℕ) 
  (h1 : total_cupcakes = 96) (h2 : num_children = 8) :
  total_cupcakes / num_children = 12 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_distribution_l2629_262993


namespace NUMINAMATH_CALUDE_even_heads_probability_l2629_262973

def probability_even_heads (p1 p2 : ℚ) (n1 n2 : ℕ) : ℚ :=
  let P1 := (1 + ((1 - 2*p1) / (1 - p1))^n1) / 2
  let P2 := (1 + ((1 - 2*p2) / (1 - p2))^n2) / 2
  P1 * P2 + (1 - P1) * (1 - P2)

theorem even_heads_probability :
  probability_even_heads (3/4) (1/2) 40 10 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_even_heads_probability_l2629_262973


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2629_262985

theorem function_inequality_implies_a_bound 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x + a) 
  (hg : ∀ x, g x = x + 4/x) 
  (h : ∀ x₁ ∈ Set.Icc 1 3, ∃ x₂ ∈ Set.Icc 1 4, f x₁ ≥ g x₂) : 
  a ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2629_262985


namespace NUMINAMATH_CALUDE_simplify_sqrt_one_minus_sin_two_alpha_l2629_262987

theorem simplify_sqrt_one_minus_sin_two_alpha (α : Real) 
  (h : π / 4 < α ∧ α < π / 2) : 
  Real.sqrt (1 - Real.sin (2 * α)) = Real.sin α - Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_one_minus_sin_two_alpha_l2629_262987


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l2629_262939

def b (n : ℕ) : ℕ := 2^n * n.factorial + n

theorem max_gcd_consecutive_terms :
  ∀ n : ℕ, ∃ m : ℕ, m ≤ n → Nat.gcd (b m) (b (m + 1)) = 1 ∧
  ∀ k : ℕ, k ≤ n → Nat.gcd (b k) (b (k + 1)) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l2629_262939


namespace NUMINAMATH_CALUDE_flower_beds_count_l2629_262953

theorem flower_beds_count (total_seeds : ℕ) (seeds_per_bed : ℕ) (h1 : total_seeds = 54) (h2 : seeds_per_bed = 6) :
  total_seeds / seeds_per_bed = 9 := by
  sorry

end NUMINAMATH_CALUDE_flower_beds_count_l2629_262953


namespace NUMINAMATH_CALUDE_bananas_to_pears_cost_equivalence_l2629_262981

/-- Given the cost relationships between bananas, apples, and pears at Lucy's Local Market,
    this theorem proves that 25 bananas cost as much as 10 pears. -/
theorem bananas_to_pears_cost_equivalence 
  (banana_apple_ratio : (5 : ℚ) * banana_cost = (3 : ℚ) * apple_cost)
  (apple_pear_ratio : (9 : ℚ) * apple_cost = (6 : ℚ) * pear_cost)
  (banana_cost apple_cost pear_cost : ℚ) :
  (25 : ℚ) * banana_cost = (10 : ℚ) * pear_cost :=
by sorry


end NUMINAMATH_CALUDE_bananas_to_pears_cost_equivalence_l2629_262981


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2629_262943

theorem cubic_equation_solution :
  ∀ (x y : ℤ), y^2 = x^3 - 3*x + 2 ↔ ∃ (k : ℕ), x = k^2 - 2 ∧ (y = k*(k^2 - 3) ∨ y = -k*(k^2 - 3)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2629_262943


namespace NUMINAMATH_CALUDE_problem_solution_l2629_262971

theorem problem_solution (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 72) :
  let y := ((x - 3)^2 * (x + 4)) / (3*x - 4)
  y = 2 ∨ y = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2629_262971


namespace NUMINAMATH_CALUDE_cos_2x_values_l2629_262960

theorem cos_2x_values (x : ℝ) (h : Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - 3 * Real.cos x ^ 2 = 0) :
  Real.cos (2 * x) = -4/5 ∨ Real.cos (2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_values_l2629_262960


namespace NUMINAMATH_CALUDE_yoongi_result_l2629_262932

theorem yoongi_result (x : ℝ) (h : x / 10 = 6) : x - 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_result_l2629_262932


namespace NUMINAMATH_CALUDE_no_isosceles_triangles_in_grid_l2629_262930

/-- Represents a point in a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a triangle in the grid --/
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

/-- Checks if a point is within the 5x5 grid --/
def isInGrid (p : GridPoint) : Prop :=
  1 ≤ p.x ∧ p.x ≤ 5 ∧ 1 ≤ p.y ∧ p.y ≤ 5

/-- Calculates the squared distance between two points --/
def squaredDistance (p1 p2 : GridPoint) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Checks if a triangle is isosceles --/
def isIsosceles (t : GridTriangle) : Prop :=
  squaredDistance t.A t.B = squaredDistance t.A t.C ∨
  squaredDistance t.A t.B = squaredDistance t.B t.C ∨
  squaredDistance t.A t.C = squaredDistance t.B t.C

/-- The main theorem --/
theorem no_isosceles_triangles_in_grid :
  ∀ (A B : GridPoint),
    isInGrid A ∧ isInGrid B ∧
    A.y = B.y ∧ squaredDistance A B = 9 →
    ¬∃ (C : GridPoint), isInGrid C ∧ isIsosceles ⟨A, B, C⟩ := by
  sorry


end NUMINAMATH_CALUDE_no_isosceles_triangles_in_grid_l2629_262930


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_third_l2629_262929

theorem cos_alpha_minus_pi_third (α : ℝ) 
  (h : Real.cos (α - π / 6) + Real.sin α = (4 / 5) * Real.sqrt 3) : 
  Real.cos (α - π / 3) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_third_l2629_262929


namespace NUMINAMATH_CALUDE_cow_spots_multiple_l2629_262980

/-- 
Given a cow with spots on both sides:
* The left side has 16 spots
* The total number of spots is 71
* The right side has 16x + 7 spots, where x is some multiple

Prove that x = 3
-/
theorem cow_spots_multiple (x : ℚ) : 
  16 + (16 * x + 7) = 71 → x = 3 := by sorry

end NUMINAMATH_CALUDE_cow_spots_multiple_l2629_262980


namespace NUMINAMATH_CALUDE_angle_sum_BD_l2629_262994

-- Define the triangle and its angles
structure Triangle (A B C : Type) where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the configuration
structure Configuration where
  angleA : ℝ
  angleAFG : ℝ
  angleAGF : ℝ
  angleB : ℝ
  angleD : ℝ

-- Theorem statement
theorem angle_sum_BD (config : Configuration) 
  (h1 : config.angleA = 30)
  (h2 : config.angleAFG = config.angleAGF) :
  config.angleB + config.angleD = 75 := by
  sorry


end NUMINAMATH_CALUDE_angle_sum_BD_l2629_262994


namespace NUMINAMATH_CALUDE_total_sums_attempted_l2629_262933

/-- Given a student's math problem attempt results, calculate the total number of sums attempted. -/
theorem total_sums_attempted (right_sums wrong_sums : ℕ) 
  (h1 : wrong_sums = 2 * right_sums) 
  (h2 : right_sums = 16) : 
  right_sums + wrong_sums = 48 :=
by sorry

end NUMINAMATH_CALUDE_total_sums_attempted_l2629_262933


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2629_262963

theorem max_value_sqrt_sum (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  Real.sqrt ((2 - a) * (2 - b) * (2 - c)) + Real.sqrt (a * b * c) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2629_262963


namespace NUMINAMATH_CALUDE_sin_sum_product_identity_l2629_262938

theorem sin_sum_product_identity : 
  Real.sin (17 * π / 180) * Real.sin (223 * π / 180) + 
  Real.sin (253 * π / 180) * Real.sin (313 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_product_identity_l2629_262938


namespace NUMINAMATH_CALUDE_geometric_sequence_special_property_l2629_262902

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_special_property 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arithmetic : 2 * ((1/2) * a 3) = a 1 + 2 * a 2) :
  q = 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_property_l2629_262902


namespace NUMINAMATH_CALUDE_sweater_markup_l2629_262948

theorem sweater_markup (wholesale_price : ℝ) (h1 : wholesale_price > 0) :
  let discounted_price := 1.4 * wholesale_price
  let retail_price := 2 * discounted_price
  let markup := (retail_price - wholesale_price) / wholesale_price * 100
  markup = 180 := by
  sorry

end NUMINAMATH_CALUDE_sweater_markup_l2629_262948


namespace NUMINAMATH_CALUDE_sum_of_yellow_and_blue_is_red_l2629_262958

theorem sum_of_yellow_and_blue_is_red (a b : ℕ) :
  ∃ m : ℕ, (4 * a + 3) + (4 * b + 2) = 4 * m + 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_yellow_and_blue_is_red_l2629_262958


namespace NUMINAMATH_CALUDE_average_difference_number_of_elements_averaged_l2629_262992

/-- Given two real numbers with an average of 45, and two real numbers with an average of 90,
    prove that the difference between the third and first number is 90. -/
theorem average_difference (a b c : ℝ) (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 90) :
  c - a = 90 := by
  sorry

/-- The number of elements being averaged in both situations is 2. -/
theorem number_of_elements_averaged (n m : ℕ) 
  (h1 : ∃ (a b : ℝ), (a + b) / n = 45)
  (h2 : ∃ (b c : ℝ), (b + c) / m = 90) :
  n = 2 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_number_of_elements_averaged_l2629_262992


namespace NUMINAMATH_CALUDE_linear_congruence_solvability_and_solutions_l2629_262941

/-- 
For integers a, b, and m > 0, this theorem states:
1. The congruence ax ≡ b (mod m) has solutions if and only if gcd(a,m) | b.
2. If solutions exist, they are of the form x = x₀ + k(m/d) for all integers k, 
   where d = gcd(a,m) and x₀ is a particular solution to (a/d)x ≡ (b/d) (mod m/d).
-/
theorem linear_congruence_solvability_and_solutions 
  (a b m : ℤ) (hm : m > 0) : 
  (∃ x, a * x ≡ b [ZMOD m]) ↔ (gcd a m ∣ b) ∧
  (∀ x, (a * x ≡ b [ZMOD m]) ↔ 
    ∃ (x₀ k : ℤ), x = x₀ + k * (m / gcd a m) ∧ 
    (a / gcd a m) * x₀ ≡ (b / gcd a m) [ZMOD (m / gcd a m)]) :=
by sorry

end NUMINAMATH_CALUDE_linear_congruence_solvability_and_solutions_l2629_262941


namespace NUMINAMATH_CALUDE_modulo_residue_problem_l2629_262966

theorem modulo_residue_problem : (513 + 3 * 68 + 9 * 289 + 2 * 34 - 10) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_problem_l2629_262966


namespace NUMINAMATH_CALUDE_unique_prime_pair_l2629_262998

def f (x : ℕ) : ℕ := x^2 + x + 1

theorem unique_prime_pair : 
  ∃! p q : ℕ, Prime p ∧ Prime q ∧ f p = f q + 242 ∧ p = 61 ∧ q = 59 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l2629_262998


namespace NUMINAMATH_CALUDE_class_size_l2629_262999

theorem class_size (hindi : ℕ) (english : ℕ) (both : ℕ) (total : ℕ) : 
  hindi = 30 → 
  english = 20 → 
  both ≥ 10 → 
  total = hindi + english - both → 
  total = 40 := by
sorry

end NUMINAMATH_CALUDE_class_size_l2629_262999


namespace NUMINAMATH_CALUDE_geometric_and_arithmetic_sequences_l2629_262969

-- Define the geometric sequence a_n
def a (n : ℕ) : ℝ := 2^n

-- Define the arithmetic sequence b_n
def b (n : ℕ) : ℝ := 12 * n - 28

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℝ := 6 * n^2 - 22 * n

theorem geometric_and_arithmetic_sequences :
  (a 1 = 2) ∧ 
  (a 4 = 16) ∧ 
  (∀ n : ℕ, a n = 2^n) ∧
  (b 3 = a 3) ∧
  (b 5 = a 5) ∧
  (∀ n : ℕ, b n = 12 * n - 28) ∧
  (∀ n : ℕ, S n = 6 * n^2 - 22 * n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_and_arithmetic_sequences_l2629_262969


namespace NUMINAMATH_CALUDE_stock_price_increase_percentage_l2629_262904

theorem stock_price_increase_percentage (total_stocks : ℕ) (higher_stocks : ℕ) : 
  total_stocks = 1980 →
  higher_stocks = 1080 →
  higher_stocks > (total_stocks - higher_stocks) →
  (((higher_stocks : ℝ) - (total_stocks - higher_stocks)) / (total_stocks - higher_stocks : ℝ)) * 100 = 20 :=
by sorry

end NUMINAMATH_CALUDE_stock_price_increase_percentage_l2629_262904


namespace NUMINAMATH_CALUDE_percentage_increase_l2629_262913

theorem percentage_increase (x : ℝ) (h : x = 99.9) :
  (x - 90) / 90 * 100 = 11 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2629_262913


namespace NUMINAMATH_CALUDE_area_difference_S_R_l2629_262954

/-- A square with side length 2 -/
def square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- An isosceles right triangle with legs of length 2 -/
def isoscelesRightTriangle : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 ∧ p.1 + p.2 ≤ 2}

/-- Region R: union of the square and 12 isosceles right triangles -/
def R : Set (ℝ × ℝ) := sorry

/-- Region S: smallest convex polygon containing R -/
def S : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (A : Set (ℝ × ℝ)) : ℝ := sorry

theorem area_difference_S_R : area S - area R = 36 := by sorry

end NUMINAMATH_CALUDE_area_difference_S_R_l2629_262954


namespace NUMINAMATH_CALUDE_expected_value_eight_sided_die_l2629_262984

def winnings (n : Nat) : Real := 8 - n

theorem expected_value_eight_sided_die :
  let outcomes := Finset.range 8
  let prob (k : Nat) := 1 / 8
  Finset.sum outcomes (fun k => prob k * winnings (k + 1)) = 3.5 := by sorry

end NUMINAMATH_CALUDE_expected_value_eight_sided_die_l2629_262984


namespace NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l2629_262967

def OddUnitsDigit : Set Nat := {1, 3, 5, 7, 9}
def AllDigits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem smallest_non_odd_units_digit : 
  ∃ (d : Nat), d ∈ AllDigits ∧ d ∉ OddUnitsDigit ∧ 
  ∀ (x : Nat), x ∈ AllDigits ∧ x ∉ OddUnitsDigit → d ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l2629_262967


namespace NUMINAMATH_CALUDE_tangent_line_problem_l2629_262997

theorem tangent_line_problem (a : ℝ) : 
  (∃ (k : ℝ), 
    (∃ (x₀ : ℝ), 
      (x₀^3 = k * (x₀ - 1)) ∧ 
      (a * x₀^2 + 15/4 * x₀ - 9 = k * (x₀ - 1)) ∧
      (3 * x₀^2 = k))) →
  (a = -25/64 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l2629_262997


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2629_262935

theorem unique_prime_solution : 
  ∀ p q r : ℕ, 
    Prime p → Prime q → Prime r → 
    p * q * r = 5 * (p + q + r) → 
    (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 7 ∧ r = 5) ∨ (p = 5 ∧ q = 2 ∧ r = 7) ∨ 
    (p = 5 ∧ q = 7 ∧ r = 2) ∨ (p = 7 ∧ q = 2 ∧ r = 5) ∨ (p = 7 ∧ q = 5 ∧ r = 2) :=
by
  sorry

#check unique_prime_solution

end NUMINAMATH_CALUDE_unique_prime_solution_l2629_262935


namespace NUMINAMATH_CALUDE_isosceles_side_in_equilateral_l2629_262936

/-- The length of a side of an isosceles triangle inscribed in an equilateral triangle -/
theorem isosceles_side_in_equilateral (s : ℝ) (h : s = 2) :
  let equilateral_side := s
  let isosceles_base := equilateral_side / 2
  let isosceles_side := Real.sqrt (7 / 3)
  ∃ (triangle : Set (ℝ × ℝ)),
    (∀ p ∈ triangle, p.1 ≥ 0 ∧ p.1 ≤ equilateral_side ∧ p.2 ≥ 0 ∧ p.2 ≤ equilateral_side * Real.sqrt 3 / 2) ∧
    (∃ (a b c : ℝ × ℝ), a ∈ triangle ∧ b ∈ triangle ∧ c ∈ triangle ∧
      (a.1 - b.1)^2 + (a.2 - b.2)^2 = equilateral_side^2 ∧
      (b.1 - c.1)^2 + (b.2 - c.2)^2 = equilateral_side^2 ∧
      (c.1 - a.1)^2 + (c.2 - a.2)^2 = equilateral_side^2) ∧
    (∃ (p q r : ℝ × ℝ), p ∈ triangle ∧ q ∈ triangle ∧ r ∈ triangle ∧
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = isosceles_side^2 ∧
      (q.1 - r.1)^2 + (q.2 - r.2)^2 = isosceles_side^2 ∧
      (r.1 - p.1)^2 + (r.2 - p.2)^2 = isosceles_base^2) := by
  sorry


end NUMINAMATH_CALUDE_isosceles_side_in_equilateral_l2629_262936


namespace NUMINAMATH_CALUDE_license_plate_increase_l2629_262961

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^3
  let new_plates := 26^4 * 10^4
  new_plates / old_plates = 6760 := by
sorry

end NUMINAMATH_CALUDE_license_plate_increase_l2629_262961


namespace NUMINAMATH_CALUDE_male_avg_is_58_l2629_262903

/-- Represents an association with male and female members selling raffle tickets -/
structure Association where
  total_avg : ℝ
  female_avg : ℝ
  male_female_ratio : ℝ

/-- The average number of tickets sold by male members -/
def male_avg (a : Association) : ℝ :=
  (3 * a.total_avg - 2 * a.female_avg)

/-- Theorem stating the average number of tickets sold by male members -/
theorem male_avg_is_58 (a : Association) 
  (h1 : a.total_avg = 66)
  (h2 : a.female_avg = 70)
  (h3 : a.male_female_ratio = 1/2) :
  male_avg a = 58 := by
  sorry


end NUMINAMATH_CALUDE_male_avg_is_58_l2629_262903


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l2629_262990

theorem least_k_for_inequality : ∃ k : ℤ, k = 5 ∧ 
  (∀ n : ℤ, 0.0010101 * (10 : ℝ)^n > 10 → n ≥ k) ∧
  (0.0010101 * (10 : ℝ)^k > 10) := by
  sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l2629_262990


namespace NUMINAMATH_CALUDE_unique_number_between_30_and_40_with_units_digit_2_l2629_262915

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_units_digit (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem unique_number_between_30_and_40_with_units_digit_2 :
  ∃! n : ℕ, is_two_digit n ∧ 30 < n ∧ n < 40 ∧ has_units_digit n 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_between_30_and_40_with_units_digit_2_l2629_262915


namespace NUMINAMATH_CALUDE_remainder_2_pow_1984_mod_17_l2629_262926

theorem remainder_2_pow_1984_mod_17 (h1 : Prime 17) (h2 : ¬ 17 ∣ 2) :
  2^1984 ≡ 0 [ZMOD 17] := by
  sorry

end NUMINAMATH_CALUDE_remainder_2_pow_1984_mod_17_l2629_262926


namespace NUMINAMATH_CALUDE_surface_area_of_modified_structure_l2629_262951

/-- Represents the dimensions of the large cube -/
def large_cube_dim : ℕ := 12

/-- Represents the dimensions of the small cubes -/
def small_cube_dim : ℕ := 2

/-- The total number of small cubes in the original structure -/
def total_small_cubes : ℕ := 64

/-- The number of small cubes removed from the structure -/
def removed_cubes : ℕ := 7

/-- The surface area of a single 2x2x2 cube before modification -/
def small_cube_surface_area : ℕ := 24

/-- The additional surface area exposed on each small cube after modification -/
def additional_exposed_area : ℕ := 6

/-- The surface area of a modified small cube -/
def modified_small_cube_area : ℕ := small_cube_surface_area + additional_exposed_area

/-- The theorem to be proved -/
theorem surface_area_of_modified_structure :
  (total_small_cubes - removed_cubes) * modified_small_cube_area = 1710 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_modified_structure_l2629_262951


namespace NUMINAMATH_CALUDE_exponent_division_l2629_262905

theorem exponent_division (a : ℝ) : a^10 / a^5 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2629_262905


namespace NUMINAMATH_CALUDE_equation_solution_l2629_262919

theorem equation_solution : 
  ∃! x : ℝ, x ≠ -4 ∧ (7 * x / (x + 4) - 5 / (x + 4) = 2 / (x + 4)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2629_262919


namespace NUMINAMATH_CALUDE_yellow_shirt_pairs_l2629_262910

theorem yellow_shirt_pairs (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) : 
  blue_students = 57 →
  yellow_students = 75 →
  total_students = 132 →
  total_pairs = 66 →
  blue_blue_pairs = 23 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_shirt_pairs_l2629_262910


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2629_262923

theorem complex_expression_equality :
  (7 - 3*Complex.I) - 3*(2 - 5*Complex.I) + (1 + 2*Complex.I) = 2 + 14*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2629_262923


namespace NUMINAMATH_CALUDE_second_graders_borrowed_books_l2629_262952

theorem second_graders_borrowed_books (initial_books borrowed_books : ℕ) : 
  initial_books = 75 → 
  initial_books - borrowed_books = 57 → 
  borrowed_books = 18 := by
sorry

end NUMINAMATH_CALUDE_second_graders_borrowed_books_l2629_262952


namespace NUMINAMATH_CALUDE_union_covers_reals_l2629_262927

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -3 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

-- State the theorem
theorem union_covers_reals (a : ℝ) : A ∪ B a = Set.univ → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_l2629_262927


namespace NUMINAMATH_CALUDE_scalene_triangle_not_unique_l2629_262968

/-- Represents a scalene triangle -/
structure ScaleneTriangle where
  -- We don't need to define the specific properties of a scalene triangle here
  -- as it's not relevant for this particular proof

/-- Represents the circumscribed circle of a triangle -/
structure CircumscribedCircle where
  radius : ℝ

/-- States that a scalene triangle is not uniquely determined by two of its angles
    and the radius of its circumscribed circle -/
theorem scalene_triangle_not_unique (α β : ℝ) (r : CircumscribedCircle) :
  ∃ (t1 t2 : ScaleneTriangle), t1 ≠ t2 ∧
  (∃ (γ1 γ2 : ℝ), α + β + γ1 = π ∧ α + β + γ2 = π) :=
sorry

end NUMINAMATH_CALUDE_scalene_triangle_not_unique_l2629_262968


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l2629_262906

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (|6 * x₁| + 5 = 47 ∧ |6 * x₂| + 5 = 47 ∧ x₁ ≠ x₂) ∧ x₁ * x₂ = -49) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l2629_262906


namespace NUMINAMATH_CALUDE_ab_gt_ac_l2629_262978

theorem ab_gt_ac (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_ab_gt_ac_l2629_262978


namespace NUMINAMATH_CALUDE_sisters_sandcastle_height_is_half_foot_l2629_262996

/-- The height of Miki's sister's sandcastle given Miki's sandcastle height and the height difference -/
def sisters_sandcastle_height (mikis_height : ℝ) (height_difference : ℝ) : ℝ :=
  mikis_height - height_difference

/-- Theorem stating that Miki's sister's sandcastle height is 0.50 foot -/
theorem sisters_sandcastle_height_is_half_foot :
  sisters_sandcastle_height 0.83 0.33 = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_sisters_sandcastle_height_is_half_foot_l2629_262996


namespace NUMINAMATH_CALUDE_sum_of_roots_l2629_262912

theorem sum_of_roots (x : ℝ) : (x + 3) * (x - 4) = 12 → ∃ (x₁ x₂ : ℝ), 
  (x₁ + 3) * (x₁ - 4) = 12 ∧ 
  (x₂ + 3) * (x₂ - 4) = 12 ∧ 
  x₁ + x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2629_262912


namespace NUMINAMATH_CALUDE_right_triangle_area_l2629_262955

/-- A right triangle PQR in the xy-plane with specific properties -/
structure RightTriangle where
  -- P, Q, R are points in ℝ²
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  -- PQR is a right triangle with right angle at R
  right_angle_at_R : (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0
  -- Length of hypotenuse PQ is 50
  hypotenuse_length : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2
  -- Median through P lies along y = x - 2
  median_P : ∃ (t : ℝ), (P.1 + R.1) / 2 = t ∧ (P.2 + R.2) / 2 = t - 2
  -- Median through Q lies along y = 3x + 3
  median_Q : ∃ (t : ℝ), (Q.1 + R.1) / 2 = t ∧ (Q.2 + R.2) / 2 = 3 * t + 3

/-- The area of the right triangle PQR is 290 -/
theorem right_triangle_area (t : RightTriangle) : 
  abs ((t.P.1 - t.R.1) * (t.Q.2 - t.R.2) - (t.Q.1 - t.R.1) * (t.P.2 - t.R.2)) / 2 = 290 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2629_262955


namespace NUMINAMATH_CALUDE_ages_sum_l2629_262925

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * a * c = 162 → a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l2629_262925


namespace NUMINAMATH_CALUDE_norm_scalar_multiple_l2629_262917

theorem norm_scalar_multiple (v : ℝ × ℝ) (h : ‖v‖ = 5) : ‖(5 : ℝ) • v‖ = 25 := by
  sorry

end NUMINAMATH_CALUDE_norm_scalar_multiple_l2629_262917


namespace NUMINAMATH_CALUDE_expression_evaluation_l2629_262975

/-- Evaluates the expression 2x^y + 5y^x - z^2 for given x, y, and z values -/
def evaluate (x y z : ℕ) : ℕ :=
  2 * (x ^ y) + 5 * (y ^ x) - (z ^ 2)

/-- Theorem stating that the expression evaluates to 42 for x=3, y=2, and z=4 -/
theorem expression_evaluation :
  evaluate 3 2 4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2629_262975


namespace NUMINAMATH_CALUDE_inequality_proof_l2629_262986

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b) / (a^5 + a * b + b^5) + (b * c) / (b^5 + b * c + c^5) + (c * a) / (c^5 + c * a + a^5) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2629_262986


namespace NUMINAMATH_CALUDE_marble_probability_l2629_262909

/-- The probability of drawing either a green or black marble from a bag -/
theorem marble_probability (green black white : ℕ) 
  (h_green : green = 4)
  (h_black : black = 3)
  (h_white : white = 6) :
  (green + black : ℚ) / (green + black + white) = 7 / 13 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l2629_262909


namespace NUMINAMATH_CALUDE_age_difference_l2629_262934

/-- Given a father and daughter whose ages sum to 54, and the daughter is 16 years old,
    prove that the difference between their ages is 22 years. -/
theorem age_difference (father_age daughter_age : ℕ) : 
  father_age + daughter_age = 54 →
  daughter_age = 16 →
  father_age - daughter_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2629_262934


namespace NUMINAMATH_CALUDE_problem_solution_l2629_262911

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2629_262911


namespace NUMINAMATH_CALUDE_min_sum_is_twelve_l2629_262928

/-- Represents a 3x3 grid of integers -/
def Grid := Matrix (Fin 3) (Fin 3) ℕ

/-- Checks if a grid contains all numbers from 1 to 9 exactly once -/
def isValidGrid (g : Grid) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → (∃! (i j : Fin 3), g i j = n)

/-- Calculates the sum of a row in the grid -/
def rowSum (g : Grid) (i : Fin 3) : ℕ :=
  (g i 0) + (g i 1) + (g i 2)

/-- Calculates the sum of a column in the grid -/
def colSum (g : Grid) (j : Fin 3) : ℕ :=
  (g 0 j) + (g 1 j) + (g 2 j)

/-- Checks if all rows and columns in the grid have the same sum -/
def hasEqualSums (g : Grid) : Prop :=
  ∃ s : ℕ, (∀ i : Fin 3, rowSum g i = s) ∧ (∀ j : Fin 3, colSum g j = s)

/-- The main theorem: The minimum sum for a valid grid with equal sums is 12 -/
theorem min_sum_is_twelve :
  ∀ g : Grid, isValidGrid g → hasEqualSums g →
  ∃ s : ℕ, (∀ i : Fin 3, rowSum g i = s) ∧ (∀ j : Fin 3, colSum g j = s) ∧ s ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_min_sum_is_twelve_l2629_262928


namespace NUMINAMATH_CALUDE_prob_at_least_one_of_two_is_eight_ninths_l2629_262942

/-- The number of candidates -/
def num_candidates : ℕ := 2

/-- The number of colleges -/
def num_colleges : ℕ := 3

/-- The probability of a candidate choosing any particular college -/
def prob_choose_college : ℚ := 1 / num_colleges

/-- The probability that both candidates choose the third college -/
def prob_both_choose_third : ℚ := prob_choose_college ^ num_candidates

/-- The probability that at least one of the first two colleges is selected -/
def prob_at_least_one_of_two : ℚ := 1 - prob_both_choose_third

theorem prob_at_least_one_of_two_is_eight_ninths :
  prob_at_least_one_of_two = 8 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_of_two_is_eight_ninths_l2629_262942


namespace NUMINAMATH_CALUDE_apples_bought_l2629_262945

/-- The number of apples bought by Cecile, Diane, and Emily -/
def total_apples (cecile diane emily : ℕ) : ℕ := cecile + diane + emily

/-- The theorem stating the total number of apples bought -/
theorem apples_bought (cecile diane emily : ℕ) 
  (h1 : cecile = 15)
  (h2 : diane = cecile + 20)
  (h3 : emily = ((cecile + diane) * 13) / 10) :
  total_apples cecile diane emily = 115 := by
  sorry

#check apples_bought

end NUMINAMATH_CALUDE_apples_bought_l2629_262945


namespace NUMINAMATH_CALUDE_fruits_picked_and_ratio_l2629_262901

/-- Represents the number of fruits picked by a person -/
structure FruitsPicked where
  pears : ℕ
  apples : ℕ

/-- Represents the orchard -/
structure Orchard where
  pear_trees : ℕ
  apple_trees : ℕ

def keith_picked : FruitsPicked := { pears := 6, apples := 4 }
def jason_picked : FruitsPicked := { pears := 9, apples := 8 }
def joan_picked : FruitsPicked := { pears := 4, apples := 12 }

def orchard : Orchard := { pear_trees := 4, apple_trees := 3 }

def total_fruits (keith jason joan : FruitsPicked) : ℕ :=
  keith.pears + keith.apples + jason.pears + jason.apples + joan.pears + joan.apples

def total_apples (keith jason joan : FruitsPicked) : ℕ :=
  keith.apples + jason.apples + joan.apples

def total_pears (keith jason joan : FruitsPicked) : ℕ :=
  keith.pears + jason.pears + joan.pears

theorem fruits_picked_and_ratio 
  (keith jason joan : FruitsPicked) 
  (o : Orchard) 
  (h_keith : keith = keith_picked)
  (h_jason : jason = jason_picked)
  (h_joan : joan = joan_picked)
  (h_orchard : o = orchard) :
  total_fruits keith jason joan = 43 ∧ 
  total_apples keith jason joan = 24 ∧
  total_pears keith jason joan = 19 := by
  sorry

end NUMINAMATH_CALUDE_fruits_picked_and_ratio_l2629_262901


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l2629_262988

-- Equation 1: x^2 + 4x - 1 = 0
theorem equation_one_solutions (x : ℝ) :
  x^2 + 4*x - 1 = 0 ↔ x = Real.sqrt 5 - 2 ∨ x = -Real.sqrt 5 - 2 := by sorry

-- Equation 2: (x-1)^2 = 3(x-1)
theorem equation_two_solutions (x : ℝ) :
  (x - 1)^2 = 3*(x - 1) ↔ x = 1 ∨ x = 4 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l2629_262988


namespace NUMINAMATH_CALUDE_abc_inequality_l2629_262976

theorem abc_inequality (a b c : ℝ) (h : (1/4)*a^2 + (1/4)*b^2 + c^2 = 1) :
  -2 ≤ a*b + 2*b*c + 2*c*a ∧ a*b + 2*b*c + 2*c*a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2629_262976


namespace NUMINAMATH_CALUDE_center_number_l2629_262965

/-- Represents a 3x3 grid with numbers from 1 to 9 --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two positions in the grid share an edge --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Check if the grid satisfies the consecutive number constraint --/
def consecutive_constraint (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ p1 p2 : Fin 3 × Fin 3,
    g p1.1 p1.2 = n ∧ g p2.1 p2.2 = n + 1 ∧ adjacent p1 p2

/-- Sum of corner numbers in the grid --/
def corner_sum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- The main theorem --/
theorem center_number (g : Grid) 
  (unique : ∀ i j k l : Fin 3, g i j = g k l → (i, j) = (k, l))
  (all_numbers : ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n)
  (consec : consecutive_constraint g)
  (corners : corner_sum g = 20) :
  g 1 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_center_number_l2629_262965


namespace NUMINAMATH_CALUDE_tangent_line_equation_intersecting_line_equation_l2629_262931

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

-- Define a line passing through point P(-2, 0)
def line_through_P (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Theorem for tangent line
theorem tangent_line_equation :
  ∀ k : ℝ, (∀ x y : ℝ, circle_C x y → line_through_P k x y → (x = -2 ∨ y = (3/4)*x + 3/2)) ∧
            (∀ x y : ℝ, (x = -2 ∨ y = (3/4)*x + 3/2) → line_through_P k x y → 
             (∃! p : ℝ × ℝ, circle_C p.1 p.2 ∧ line_through_P k p.1 p.2)) :=
sorry

-- Theorem for intersecting line with chord length 2√2
theorem intersecting_line_equation :
  ∀ k : ℝ, (∀ x y : ℝ, circle_C x y → line_through_P k x y → 
            (x - y + 2 = 0 ∨ 7*x - y + 14 = 0)) ∧
           (∀ x y : ℝ, (x - y + 2 = 0 ∨ 7*x - y + 14 = 0) → line_through_P k x y → 
            (∃ A B : ℝ × ℝ, circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ 
             line_through_P k A.1 A.2 ∧ line_through_P k B.1 B.2 ∧
             (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_intersecting_line_equation_l2629_262931


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2629_262983

theorem sum_of_roots_quadratic (z : ℂ) : 
  (∃ z₁ z₂ : ℂ, z₁ + z₂ = 16 ∧ z₁ * z₂ = 15 ∧ z₁ ≠ z₂ ∧ z^2 - 16*z + 15 = 0 → z = z₁ ∨ z = z₂) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2629_262983


namespace NUMINAMATH_CALUDE_smallest_x_value_l2629_262962

theorem smallest_x_value (x : ℝ) : 
  (3 * x^2 + 36 * x - 72 = x * (x + 20) + 8) → x ≥ -10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2629_262962


namespace NUMINAMATH_CALUDE_battleship_detectors_l2629_262989

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a ship with width and height -/
structure Ship :=
  (width : ℕ)
  (height : ℕ)

/-- Function to calculate the minimum number of detectors required -/
def min_detectors (g : Grid) (s : Ship) : ℕ :=
  ((g.width - 1) / 3) * 2 + ((g.width - 1) % 3) + 1

/-- Theorem stating the minimum number of detectors for the Battleship problem -/
theorem battleship_detectors :
  let grid : Grid := ⟨203, 1⟩
  let ship : Ship := ⟨2, 1⟩
  min_detectors grid ship = 134 := by
  sorry

#check battleship_detectors

end NUMINAMATH_CALUDE_battleship_detectors_l2629_262989


namespace NUMINAMATH_CALUDE_derivative_inequality_l2629_262937

theorem derivative_inequality (a : ℝ) (ha : a > 0) (x : ℝ) (hx : x ≥ 1) :
  let f : ℝ → ℝ := λ x => a * Real.log x + x + 2
  (deriv f) x < x^2 + (a + 2) * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_inequality_l2629_262937


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2629_262920

theorem inequality_equivalence (x : ℝ) : 
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + x)) ↔ 
  (x ≥ -12 / 7 ∧ x < -6 / 5) := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2629_262920


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2629_262982

/-- Given a hyperbola and a circle with specific properties, prove the equation of the hyperbola. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (∃ x y : ℝ, x^2 + y^2 - 6*y + 5 = 0) →
  (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 6*y₀ + 5 = 0 ∧ 
    (∀ x y : ℝ, (y - y₀)^2 / a^2 - (x - x₀)^2 / b^2 = 1)) →
  a^2 = 5 ∧ b^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2629_262982


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2629_262924

theorem complex_number_quadrant (z : ℂ) : iz = -1 + I → z.re > 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2629_262924


namespace NUMINAMATH_CALUDE_length_QR_is_4_l2629_262900

-- Define the points and circles
structure Point := (x : ℝ) (y : ℝ)
structure Circle := (center : Point) (radius : ℝ)

-- Define the given conditions
axiom P : Point
axiom Q : Point
axiom circle_P : Circle
axiom circle_Q : Circle
axiom R : Point

-- Radii of circles
axiom radius_P : circle_P.radius = 7
axiom radius_Q : circle_Q.radius = 4

-- Circles are externally tangent
axiom externally_tangent : 
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2) = circle_P.radius + circle_Q.radius

-- R is on the line tangent to both circles
axiom R_on_tangent_line : 
  ∃ (S T : Point),
    (Real.sqrt ((S.x - P.x)^2 + (S.y - P.y)^2) = circle_P.radius) ∧
    (Real.sqrt ((T.x - Q.x)^2 + (T.y - Q.y)^2) = circle_Q.radius) ∧
    (R.x - S.x) * (T.y - S.y) = (R.y - S.y) * (T.x - S.x)

-- R is on ray PQ
axiom R_on_ray_PQ :
  ∃ (t : ℝ), t ≥ 0 ∧ R.x = P.x + t * (Q.x - P.x) ∧ R.y = P.y + t * (Q.y - P.y)

-- Vertical line from Q is tangent to circle at P
axiom vertical_tangent :
  ∃ (U : Point),
    U.x = P.x ∧ 
    Real.sqrt ((U.x - P.x)^2 + (U.y - P.y)^2) = circle_P.radius ∧
    U.x - Q.x = 0

-- Theorem to prove
theorem length_QR_is_4 :
  Real.sqrt ((R.x - Q.x)^2 + (R.y - Q.y)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_length_QR_is_4_l2629_262900


namespace NUMINAMATH_CALUDE_green_paint_amount_l2629_262914

/-- Paint mixture ratios -/
structure PaintMixture where
  blue : ℚ
  green : ℚ
  white : ℚ
  red : ℚ

/-- Theorem: Given a paint mixture with ratio 5:3:4:2 for blue:green:white:red,
    if 10 quarts of blue paint are used, then 6 quarts of green paint should be used. -/
theorem green_paint_amount (mix : PaintMixture) 
  (ratio : mix.blue = 5 ∧ mix.green = 3 ∧ mix.white = 4 ∧ mix.red = 2) 
  (blue_amount : ℚ) (h : blue_amount = 10) : 
  (blue_amount * mix.green / mix.blue) = 6 := by
  sorry

end NUMINAMATH_CALUDE_green_paint_amount_l2629_262914


namespace NUMINAMATH_CALUDE_evaluate_expression_l2629_262918

theorem evaluate_expression : 5000 * (5000 ^ 1000) = 5000 ^ 1001 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2629_262918


namespace NUMINAMATH_CALUDE_correct_distribution_l2629_262907

structure Participant where
  name : String
  initialDeposit : ℕ
  depositTime : ℕ

def totalValue : ℕ := 108000
def secondCarValue : ℕ := 48000
def secondCarSoldValue : ℕ := 42000

def calculateShare (p : Participant) (totalDays : ℕ) : ℚ :=
  (p.initialDeposit * p.depositTime : ℚ) / (totalDays * totalValue : ℚ)

def adjustedShare (share : ℚ) : ℚ :=
  share * (secondCarSoldValue : ℚ) / (secondCarValue : ℚ)

theorem correct_distribution 
  (istvan kalman laszlo miklos : Participant)
  (h1 : istvan.initialDeposit = 5000 + 4000 - 2500)
  (h2 : istvan.depositTime = 90)
  (h3 : kalman.initialDeposit = 4000)
  (h4 : kalman.depositTime = 70)
  (h5 : laszlo.initialDeposit = 2500)
  (h6 : laszlo.depositTime = 40)
  (h7 : miklos.initialDeposit = 2000)
  (h8 : miklos.depositTime = 90)
  : adjustedShare (calculateShare istvan 90) * secondCarValue = 54600 ∧
    adjustedShare (calculateShare kalman 90) * secondCarValue - 
    adjustedShare (calculateShare miklos 90) * secondCarValue = 7800 ∧
    adjustedShare (calculateShare laszlo 90) * secondCarValue = 10500 ∧
    adjustedShare (calculateShare miklos 90) * secondCarValue = 18900 := by
  sorry

#eval totalValue
#eval secondCarValue
#eval secondCarSoldValue

end NUMINAMATH_CALUDE_correct_distribution_l2629_262907


namespace NUMINAMATH_CALUDE_complex_magnitude_l2629_262991

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2629_262991


namespace NUMINAMATH_CALUDE_cat_pictures_count_l2629_262977

/-- Represents the number of photos on Toby's camera roll at different stages -/
structure PhotoCount where
  initial : ℕ
  afterFirstDeletion : ℕ
  final : ℕ

/-- Represents the number of photos deleted or added at different stages -/
structure PhotoChanges where
  firstDeletion : ℕ
  catPictures : ℕ
  friendPhotos : ℕ
  secondDeletion : ℕ

/-- Theorem stating the relationship between cat pictures and friend photos -/
theorem cat_pictures_count (p : PhotoCount) (c : PhotoChanges) :
  p.initial = 63 →
  p.final = 84 →
  c.firstDeletion = 7 →
  c.secondDeletion = 3 →
  p.afterFirstDeletion = p.initial - c.firstDeletion →
  p.final = p.afterFirstDeletion + c.catPictures + c.friendPhotos - c.secondDeletion →
  c.catPictures = 31 - c.friendPhotos := by
  sorry


end NUMINAMATH_CALUDE_cat_pictures_count_l2629_262977


namespace NUMINAMATH_CALUDE_uncovered_side_length_l2629_262957

/-- Represents a rectangular field with three sides fenced -/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing : ℝ

/-- The uncovered side of a fenced field is 40 feet given the conditions -/
theorem uncovered_side_length (field : FencedField)
  (h_area : field.area = 680)
  (h_fencing : field.fencing = 74)
  (h_area_calc : field.area = field.length * field.width)
  (h_fencing_calc : field.fencing = 2 * field.width + field.length) :
  field.length = 40 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_side_length_l2629_262957


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_ratio_l2629_262972

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Length of the smaller base -/
  ab : ℝ
  /-- Length of the larger base -/
  cd : ℝ
  /-- Length of the diagonal AC -/
  ac : ℝ
  /-- Height of the trapezoid (altitude from D to AB) -/
  h : ℝ
  /-- The smaller base is less than the larger base -/
  ab_lt_cd : ab < cd
  /-- The diagonal AC is twice the length of the larger base CD -/
  ac_eq_2cd : ac = 2 * cd
  /-- The smaller base AB equals the height of the trapezoid -/
  ab_eq_h : ab = h

/-- The ratio of the smaller base to the larger base in the specific isosceles trapezoid is 3:1 -/
theorem isosceles_trapezoid_ratio (t : IsoscelesTrapezoid) : t.ab / t.cd = 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_ratio_l2629_262972


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2629_262950

theorem arithmetic_sequence_problem (a b c : ℝ) : 
  (b - a = c - b) →  -- arithmetic sequence condition
  (a + b + c = 9) →  -- sum condition
  (a * b = 6 * c) →  -- product condition
  (a = 4 ∧ b = 3 ∧ c = 2) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2629_262950


namespace NUMINAMATH_CALUDE_set_inclusion_conditions_l2629_262944

def P : Set ℝ := {x | x^2 + 4*x = 0}
def Q (m : ℝ) : Set ℝ := {x | x^2 + 2*(m+1)*x + m^2 - 1 = 0}

theorem set_inclusion_conditions :
  (∀ m : ℝ, P ⊆ Q m ↔ m = 1) ∧
  (∀ m : ℝ, Q m ⊆ P ↔ m ≤ -1 ∨ m = 1) := by sorry

end NUMINAMATH_CALUDE_set_inclusion_conditions_l2629_262944


namespace NUMINAMATH_CALUDE_triangular_number_difference_l2629_262964

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The difference between the 30th and 28th triangular numbers is 59 -/
theorem triangular_number_difference : triangular_number 30 - triangular_number 28 = 59 := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_difference_l2629_262964


namespace NUMINAMATH_CALUDE_fraction_equality_l2629_262922

theorem fraction_equality (m n p r : ℚ) 
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / r = 1 / 7) :
  m / r = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2629_262922


namespace NUMINAMATH_CALUDE_locus_of_circle_centers_is_hyperbola_l2629_262949

/-- The locus of points equidistant from two fixed points forms a hyperbola --/
theorem locus_of_circle_centers_is_hyperbola 
  (M : ℝ × ℝ) -- Point M(x, y)
  (C₁ : ℝ × ℝ := (0, -1)) -- Center of circle C₁
  (C₂ : ℝ × ℝ := (0, 4)) -- Center of circle C₂
  (h : Real.sqrt ((M.1 - C₂.1)^2 + (M.2 - C₂.2)^2) - 
       Real.sqrt ((M.1 - C₁.1)^2 + (M.2 - C₁.2)^2) = 1) :
  -- The statement that M lies on a hyperbola
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (M.1^2 / a^2) - (M.2^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_circle_centers_is_hyperbola_l2629_262949


namespace NUMINAMATH_CALUDE_similar_cube_volume_l2629_262979

theorem similar_cube_volume (original_volume : ℝ) (scale_factor : ℝ) : 
  original_volume = 27 → scale_factor = 2 → 
  (scale_factor^3 * original_volume : ℝ) = 216 := by
  sorry

end NUMINAMATH_CALUDE_similar_cube_volume_l2629_262979


namespace NUMINAMATH_CALUDE_pen_cost_l2629_262995

/-- Given the cost of pens and pencils in two different combinations, 
    prove that the cost of a single pen is 39 cents. -/
theorem pen_cost (x y : ℚ) 
  (eq1 : 3 * x + 2 * y = 183) 
  (eq2 : 5 * x + 4 * y = 327) : 
  x = 39 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_l2629_262995


namespace NUMINAMATH_CALUDE_ceiling_equality_abs_diff_l2629_262921

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- Main theorem
theorem ceiling_equality_abs_diff (x y : ℝ) :
  (∀ x y, ceiling x = ceiling y → |x - y| < 1) ∧
  (∃ x y, |x - y| < 1 ∧ ceiling x ≠ ceiling y) :=
by sorry

end NUMINAMATH_CALUDE_ceiling_equality_abs_diff_l2629_262921


namespace NUMINAMATH_CALUDE_mass_of_iodine_l2629_262946

/-- The mass of 3 moles of I₂ given the atomic mass of I -/
theorem mass_of_iodine (atomic_mass_I : ℝ) (h : atomic_mass_I = 126.90) :
  let molar_mass_I2 := 2 * atomic_mass_I
  3 * molar_mass_I2 = 761.40 := by
  sorry

#check mass_of_iodine

end NUMINAMATH_CALUDE_mass_of_iodine_l2629_262946


namespace NUMINAMATH_CALUDE_angle_counterexample_l2629_262959

theorem angle_counterexample : ∃ (angle1 angle2 : ℝ), 
  angle1 + angle2 = 90 ∧ angle1 = angle2 := by
  sorry

end NUMINAMATH_CALUDE_angle_counterexample_l2629_262959


namespace NUMINAMATH_CALUDE_meaningful_fraction_l2629_262956

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l2629_262956


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l2629_262916

theorem no_solutions_in_interval (x : ℝ) :
  x ∈ Set.Icc (π / 4) (π / 2) →
  ¬(Real.sin (x ^ Real.sin x) = Real.cos (x ^ Real.cos x)) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l2629_262916


namespace NUMINAMATH_CALUDE_combinable_with_sqrt_three_l2629_262974

theorem combinable_with_sqrt_three : ∃! x : ℝ, x > 0 ∧ 
  (x = Real.sqrt (3^2) ∨ x = Real.sqrt 27 ∨ x = Real.sqrt 30 ∨ x = Real.sqrt (2/3)) ∧
  ∃ (r : ℚ), x = r * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_combinable_with_sqrt_three_l2629_262974


namespace NUMINAMATH_CALUDE_balls_given_to_partner_l2629_262940

/-- Represents the number of tennis games played by Bertha -/
def games : ℕ := 20

/-- Represents the number of games after which one ball wears out -/
def wear_out_rate : ℕ := 10

/-- Represents the number of games after which Bertha loses a ball -/
def lose_rate : ℕ := 5

/-- Represents the number of games after which Bertha buys a canister of balls -/
def buy_rate : ℕ := 4

/-- Represents the number of balls in each canister -/
def balls_per_canister : ℕ := 3

/-- Represents the number of balls Bertha started with -/
def initial_balls : ℕ := 2

/-- Represents the number of balls Bertha has after 20 games -/
def final_balls : ℕ := 10

/-- Calculates the number of balls worn out during the games -/
def balls_worn_out : ℕ := games / wear_out_rate

/-- Calculates the number of balls lost during the games -/
def balls_lost : ℕ := games / lose_rate

/-- Calculates the number of balls bought during the games -/
def balls_bought : ℕ := (games / buy_rate) * balls_per_canister

/-- Theorem stating that Bertha gave 1 ball to her partner -/
theorem balls_given_to_partner :
  initial_balls + balls_bought - balls_worn_out - balls_lost - final_balls = 1 := by
  sorry

end NUMINAMATH_CALUDE_balls_given_to_partner_l2629_262940


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l2629_262970

/-- The line equation y = -3x + 4 -/
def line_equation (x y : ℝ) : Prop := y = -3 * x + 4

/-- Point A with coordinates (2, y₁) -/
def point_A (y₁ : ℝ) : ℝ × ℝ := (2, y₁)

/-- Point B with coordinates (-1, y₂) -/
def point_B (y₂ : ℝ) : ℝ × ℝ := (-1, y₂)

theorem y1_less_than_y2 (y₁ y₂ : ℝ) 
  (hA : line_equation (point_A y₁).1 (point_A y₁).2)
  (hB : line_equation (point_B y₂).1 (point_B y₂).2) :
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l2629_262970


namespace NUMINAMATH_CALUDE_inequalities_proof_l2629_262908

theorem inequalities_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^2 > b^2) ∧ (a^3 > b^3) ∧ (Real.sqrt (a - b) > Real.sqrt a - Real.sqrt b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2629_262908
