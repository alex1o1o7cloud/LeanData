import Mathlib

namespace NUMINAMATH_CALUDE_f_42_17_l2221_222106

def is_valid_f (f : ℚ → Int) : Prop :=
  (∀ x y : ℚ, x ≠ y → (x * y = 1 ∨ x + y = 0 ∨ x + y = 1) → f x * f y = -1) ∧
  (∀ x : ℚ, f x = 1 ∨ f x = -1) ∧
  f 0 = 1

theorem f_42_17 (f : ℚ → Int) (h : is_valid_f f) : f (42/17) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_42_17_l2221_222106


namespace NUMINAMATH_CALUDE_oldest_child_age_l2221_222170

def average_age : ℝ := 7
def younger_child1_age : ℝ := 4
def younger_child2_age : ℝ := 7

theorem oldest_child_age :
  ∃ (oldest_age : ℝ),
    (younger_child1_age + younger_child2_age + oldest_age) / 3 = average_age ∧
    oldest_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l2221_222170


namespace NUMINAMATH_CALUDE_at_most_one_lattice_point_on_circle_l2221_222190

/-- A point in a 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The squared distance between two points -/
def squaredDistance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem at_most_one_lattice_point_on_circle 
  (center : ℝ × ℝ) 
  (h_center : center = (Real.sqrt 2, Real.sqrt 3)) 
  (p q : LatticePoint) 
  (r : ℝ) 
  (h_p : squaredDistance (p.x, p.y) center = r^2) 
  (h_q : squaredDistance (q.x, q.y) center = r^2) : 
  p = q :=
sorry

end NUMINAMATH_CALUDE_at_most_one_lattice_point_on_circle_l2221_222190


namespace NUMINAMATH_CALUDE_coffee_consumption_ratio_l2221_222195

/-- Represents the number of coffees John used to buy daily -/
def old_coffee_count : ℕ := 4

/-- Represents the original price of each coffee in dollars -/
def old_coffee_price : ℚ := 2

/-- Represents the percentage increase in coffee price -/
def price_increase_percent : ℚ := 50

/-- Represents the amount John saves daily compared to his old spending in dollars -/
def daily_savings : ℚ := 2

/-- Theorem stating that the ratio of John's current coffee consumption to his previous consumption is 1:2 -/
theorem coffee_consumption_ratio :
  ∃ (new_coffee_count : ℕ),
    new_coffee_count * (old_coffee_price * (1 + price_increase_percent / 100)) = 
      old_coffee_count * old_coffee_price - daily_savings ∧
    new_coffee_count * 2 = old_coffee_count := by
  sorry

end NUMINAMATH_CALUDE_coffee_consumption_ratio_l2221_222195


namespace NUMINAMATH_CALUDE_x_coordinate_of_first_point_l2221_222161

/-- Given two points on a line, prove the x-coordinate of the first point -/
theorem x_coordinate_of_first_point 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  m = 2 * n + 5 := by
sorry

end NUMINAMATH_CALUDE_x_coordinate_of_first_point_l2221_222161


namespace NUMINAMATH_CALUDE_fraction_equality_l2221_222115

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 18)
  (h2 : p / n = 6)
  (h3 : p / q = 1 / 15) :
  m / q = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2221_222115


namespace NUMINAMATH_CALUDE_complex_multiplication_l2221_222127

theorem complex_multiplication (i : ℂ) : i * i = -1 → (-1 + i) * (2 - i) = -1 + 3 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2221_222127


namespace NUMINAMATH_CALUDE_union_subset_iff_m_in_range_l2221_222155

def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def C (m : ℝ) : Set ℝ := {x : ℝ | m * x + 1 > 0}

theorem union_subset_iff_m_in_range :
  ∀ m : ℝ, (A ∪ B) ⊆ C m ↔ m ∈ Set.Icc (-1/2) 1 := by sorry

end NUMINAMATH_CALUDE_union_subset_iff_m_in_range_l2221_222155


namespace NUMINAMATH_CALUDE_expansion_properties_l2221_222114

-- Define the binomial expansion function
def binomial_expansion (x : ℝ) (n : ℕ) : ℝ → ℝ := sorry

-- Define the coefficient function for the expansion
def coefficient (x : ℝ) (n r : ℕ) : ℝ := sorry

-- Define the general term of the expansion
def general_term (x : ℝ) (n r : ℕ) : ℝ := sorry

theorem expansion_properties :
  let f := binomial_expansion x 8
  -- The first three coefficients are in arithmetic sequence
  ∃ (a d : ℝ), coefficient x 8 0 = a ∧ 
               coefficient x 8 1 = a + d ∧ 
               coefficient x 8 2 = a + 2*d →
  -- 1. The term containing x to the first power
  (∃ (r : ℕ), general_term x 8 r = (35/8) * x) ∧
  -- 2. The rational terms involving x
  (∀ (r : ℕ), r ≤ 8 → 
    (∃ (k : ℤ), general_term x 8 r = x^k) ↔ 
    (general_term x 8 r = x^4 ∨ 
     general_term x 8 r = (35/8) * x ∨ 
     general_term x 8 r = 1/(256 * x^2))) ∧
  -- 3. The terms with the largest coefficient
  (∀ (r : ℕ), r ≤ 8 → 
    coefficient x 8 r ≤ 7 ∧
    (coefficient x 8 r = 7 ↔ (r = 2 ∨ r = 3))) :=
sorry

end NUMINAMATH_CALUDE_expansion_properties_l2221_222114


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l2221_222194

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    n ≥ 100 ∧ 
    n < 1000 ∧ 
    75 * n % 345 = 225 ∧ 
    (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 75 * m % 345 = 225 → m ≥ n) ∧
    n = 118 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l2221_222194


namespace NUMINAMATH_CALUDE_weeks_to_save_l2221_222107

def console_cost : ℕ := 282
def initial_savings : ℕ := 42
def weekly_allowance : ℕ := 24

theorem weeks_to_save : 
  (console_cost - initial_savings) / weekly_allowance = 10 :=
sorry

end NUMINAMATH_CALUDE_weeks_to_save_l2221_222107


namespace NUMINAMATH_CALUDE_f_max_at_zero_l2221_222139

-- Define the function f and its derivative
noncomputable def f : ℝ → ℝ := λ x => x^4 - 2*x^2 - 5
def f' : ℝ → ℝ := λ x => 4*x^3 - 4*x

-- State the theorem
theorem f_max_at_zero :
  (∀ x : ℝ, (f' x) = 4*x^3 - 4*x) →
  f 0 = -5 →
  (∀ x : ℝ, f x ≤ -5) ∧ f 0 = -5 :=
by sorry

end NUMINAMATH_CALUDE_f_max_at_zero_l2221_222139


namespace NUMINAMATH_CALUDE_jerome_theorem_l2221_222125

def jerome_problem (initial_money : ℝ) : Prop :=
  let half_money : ℝ := 43
  let meg_amount : ℝ := 8
  let bianca_amount : ℝ := 3 * meg_amount
  let after_meg_bianca : ℝ := initial_money - meg_amount - bianca_amount
  let nathan_amount : ℝ := after_meg_bianca / 2
  let after_nathan : ℝ := after_meg_bianca - nathan_amount
  let charity_percentage : ℝ := 0.2
  let charity_amount : ℝ := charity_percentage * after_nathan
  let final_amount : ℝ := after_nathan - charity_amount

  (initial_money / 2 = half_money) ∧
  (final_amount = 21.60)

theorem jerome_theorem : 
  ∃ (initial_money : ℝ), jerome_problem initial_money :=
by
  sorry

end NUMINAMATH_CALUDE_jerome_theorem_l2221_222125


namespace NUMINAMATH_CALUDE_vector_calculation_l2221_222117

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, -1]

theorem vector_calculation : 
  (-2 • a + 4 • b) = ![-6, -8] := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l2221_222117


namespace NUMINAMATH_CALUDE_partnership_profit_share_l2221_222119

/-- A partnership problem with four partners A, B, C, and D --/
theorem partnership_profit_share
  (capital_A : ℚ) (capital_B : ℚ) (capital_C : ℚ) (capital_D : ℚ) (total_profit : ℕ)
  (h1 : capital_A = 1 / 3)
  (h2 : capital_B = 1 / 4)
  (h3 : capital_C = 1 / 5)
  (h4 : capital_A + capital_B + capital_C + capital_D = 1)
  (h5 : total_profit = 2490) :
  ∃ (share_A : ℕ), share_A = 830 ∧ share_A = (capital_A * total_profit).num :=
sorry

end NUMINAMATH_CALUDE_partnership_profit_share_l2221_222119


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l2221_222141

theorem min_x_prime_factorization_sum (x y : ℕ+) (h : 3 * x ^ 12 = 5 * y ^ 17) :
  ∃ (a b c d : ℕ),
    (∀ (p : ℕ), p.Prime → p ∣ x → p = a ∨ p = b) ∧
    x = a ^ c * b ^ d ∧
    (∀ (x' : ℕ+), 3 * x' ^ 12 = 5 * y ^ 17 → x ≤ x') ∧
    a + b + c + d = 30 :=
by sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l2221_222141


namespace NUMINAMATH_CALUDE_prop_1_false_prop_2_true_prop_3_false_l2221_222128

-- Proposition 1
theorem prop_1_false : ∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
  a * d = b * c ∧ ¬(∃ r : ℝ, (b = a * r ∧ c = b * r ∧ d = c * r) ∨ 
                             (a = b * r ∧ b = c * r ∧ c = d * r)) := by
  sorry

-- Proposition 2
theorem prop_2_true : ∀ (a : ℤ), 2 ∣ a → Even a := by
  sorry

-- Proposition 3
theorem prop_3_false : ∃ (A : ℝ), 
  30 * π / 180 < A ∧ A < π ∧ Real.sin A ≤ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prop_1_false_prop_2_true_prop_3_false_l2221_222128


namespace NUMINAMATH_CALUDE_probability_A_selected_l2221_222132

/-- The number of individuals in the group -/
def n : ℕ := 3

/-- The number of representatives to be chosen -/
def k : ℕ := 2

/-- The probability of selecting A as one of the representatives -/
def prob_A_selected : ℚ := 2/3

/-- Theorem stating that the probability of selecting A as one of two representatives
    from a group of three individuals is 2/3 -/
theorem probability_A_selected :
  prob_A_selected = 2/3 := by sorry

end NUMINAMATH_CALUDE_probability_A_selected_l2221_222132


namespace NUMINAMATH_CALUDE_largest_unrepresentable_amount_is_correct_l2221_222142

/-- Represents the set of coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Finset ℕ :=
  {6*n + 1, 6*n + 4, 6*n + 7, 6*n + 10}

/-- Predicate to check if an amount can be represented using given coin denominations -/
def is_representable (s : ℕ) (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), s = a*(6*n + 1) + b*(6*n + 4) + c*(6*n + 7) + d*(6*n + 10)

/-- The largest amount that cannot be represented using the given coin denominations -/
def largest_unrepresentable_amount (n : ℕ) : ℕ :=
  12*n^2 + 14*n - 1

/-- Theorem stating that the largest_unrepresentable_amount is correct -/
theorem largest_unrepresentable_amount_is_correct (n : ℕ) :
  (∀ k < largest_unrepresentable_amount n, is_representable k n) ∧
  ¬is_representable (largest_unrepresentable_amount n) n :=
by sorry

end NUMINAMATH_CALUDE_largest_unrepresentable_amount_is_correct_l2221_222142


namespace NUMINAMATH_CALUDE_zara_brixton_height_l2221_222180

/-- The heights of four people satisfying certain conditions -/
structure Heights where
  itzayana : ℝ
  zora : ℝ
  brixton : ℝ
  zara : ℝ
  itzayana_taller : itzayana = zora + 4
  zora_shorter : zora = brixton - 8
  zara_equal : zara = brixton
  average_height : (itzayana + zora + brixton + zara) / 4 = 61

/-- Theorem stating that Zara and Brixton's height is 64 inches -/
theorem zara_brixton_height (h : Heights) : h.zara = 64 ∧ h.brixton = 64 := by
  sorry

end NUMINAMATH_CALUDE_zara_brixton_height_l2221_222180


namespace NUMINAMATH_CALUDE_triangle_properties_l2221_222116

/-- Given a triangle ABC with the following properties:
  - m = (sin C, sin B cos A)
  - n = (b, 2c)
  - m · n = 0
  - a = 2√3
  - sin B + sin C = 1
  Prove that:
  1. The measure of angle A is 2π/3
  2. The area of triangle ABC is √3
-/
theorem triangle_properties (a b c A B C : ℝ) 
  (m : ℝ × ℝ) (n : ℝ × ℝ) 
  (hm : m = (Real.sin C, Real.sin B * Real.cos A))
  (hn : n = (b, 2 * c))
  (hdot : m.1 * n.1 + m.2 * n.2 = 0)
  (ha : a = 2 * Real.sqrt 3)
  (hsin : Real.sin B + Real.sin C = 1) :
  A = 2 * Real.pi / 3 ∧ 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2221_222116


namespace NUMINAMATH_CALUDE_omega_sum_l2221_222124

theorem omega_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80 = -ω^2 := by
  sorry

end NUMINAMATH_CALUDE_omega_sum_l2221_222124


namespace NUMINAMATH_CALUDE_choose_four_from_seven_l2221_222140

-- Define the number of available paints
def n : ℕ := 7

-- Define the number of paints to be chosen
def k : ℕ := 4

-- Theorem stating that choosing 4 paints from 7 different ones results in 35 ways
theorem choose_four_from_seven :
  Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_seven_l2221_222140


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l2221_222110

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (3 * x) % 17 = 14 % 17 ∧ ∀ (y : ℕ), y > 0 → (3 * y) % 17 = 14 % 17 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l2221_222110


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2221_222133

theorem expand_and_simplify (x : ℝ) : 2*(x+3)*(x^2 + 2*x + 7) = 2*x^3 + 10*x^2 + 26*x + 42 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2221_222133


namespace NUMINAMATH_CALUDE_parallel_intersections_l2221_222147

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection of a plane with another plane resulting in a line
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Theorem statement
theorem parallel_intersections
  (P1 P2 P3 : Plane) (l1 l2 : Line)
  (h1 : parallel_planes P1 P2)
  (h2 : l1 = intersect P3 P1)
  (h3 : l2 = intersect P3 P2) :
  parallel_lines l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_intersections_l2221_222147


namespace NUMINAMATH_CALUDE_cafe_menu_combinations_l2221_222109

theorem cafe_menu_combinations (n : ℕ) (h : n = 12) : 
  n * (n - 1) = 132 := by
  sorry

end NUMINAMATH_CALUDE_cafe_menu_combinations_l2221_222109


namespace NUMINAMATH_CALUDE_max_gcd_lcm_l2221_222177

theorem max_gcd_lcm (x y z : ℕ) 
  (h : Nat.gcd (Nat.lcm x y) z * Nat.lcm (Nat.gcd x y) z = 1400) : 
  Nat.gcd (Nat.lcm x y) z ≤ 10 ∧ 
  ∃ (a b c : ℕ), Nat.gcd (Nat.lcm a b) c = 10 ∧ 
                 Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 1400 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_lcm_l2221_222177


namespace NUMINAMATH_CALUDE_expression_evaluation_l2221_222171

theorem expression_evaluation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2221_222171


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2221_222123

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_second : a 2 = 4)
  (h_sixth : a 6 = 64) :
  a 4 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2221_222123


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l2221_222159

def digit_sum (n : Nat) : Nat :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : Nat, 100 ≤ n ∧ n < 1000 ∧ n % 9 = 0 ∧ digit_sum n = 27 → n ≤ 999 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l2221_222159


namespace NUMINAMATH_CALUDE_brick_length_proof_l2221_222104

theorem brick_length_proof (w h A : ℝ) (hw : w = 4) (hh : h = 3) (hA : A = 164) :
  let l := (A - 2 * w * h) / (2 * (w + h))
  l = 10 := by sorry

end NUMINAMATH_CALUDE_brick_length_proof_l2221_222104


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_solution_x_l2221_222187

/-- The percentage of alcohol in a solution that, when mixed with another solution,
    results in a specific alcohol concentration. -/
theorem alcohol_percentage_in_solution_x 
  (volume_x : ℝ) 
  (volume_y : ℝ) 
  (percent_y : ℝ) 
  (percent_final : ℝ) 
  (h1 : volume_x = 300)
  (h2 : volume_y = 900)
  (h3 : percent_y = 0.30)
  (h4 : percent_final = 0.25)
  : ∃ (percent_x : ℝ), 
    percent_x = 0.10 ∧ 
    volume_x * percent_x + volume_y * percent_y = (volume_x + volume_y) * percent_final :=
sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_solution_x_l2221_222187


namespace NUMINAMATH_CALUDE_engineer_designer_ratio_l2221_222103

theorem engineer_designer_ratio (e d : ℕ) (h_total : (40 * e + 55 * d) / (e + d) = 45) :
  e = 2 * d := by
  sorry

end NUMINAMATH_CALUDE_engineer_designer_ratio_l2221_222103


namespace NUMINAMATH_CALUDE_range_of_m_l2221_222174

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 2) :
  (∃ m : ℝ, x + y/4 < m^2 - m) ↔ ∃ m : ℝ, m < -1 ∨ m > 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2221_222174


namespace NUMINAMATH_CALUDE_inner_diagonal_sum_bound_l2221_222153

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  /-- Sum of the lengths of the diagonals -/
  diagonalSum : ℝ
  /-- Convexity condition -/
  convex : diagonalSum > 0

/-- Theorem: For any two convex quadrilaterals where one is inside the other,
    the sum of the diagonals of the inner quadrilateral is less than twice
    the sum of the diagonals of the outer quadrilateral -/
theorem inner_diagonal_sum_bound
  (outer inner : ConvexQuadrilateral)
  (h : inner.diagonalSum < outer.diagonalSum) :
  inner.diagonalSum < 2 * outer.diagonalSum :=
by
  sorry


end NUMINAMATH_CALUDE_inner_diagonal_sum_bound_l2221_222153


namespace NUMINAMATH_CALUDE_distance_to_focus_l2221_222175

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a point on the parabola with x-coordinate 4
def point_on_parabola : {P : ℝ × ℝ // parabola P.1 P.2 ∧ P.1 = 4} :=
  sorry

-- Theorem statement
theorem distance_to_focus :
  let P := point_on_parabola.val
  (P.1 - 0)^2 = 4^2 →  -- Distance from P to y-axis is 4
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 6^2  -- Distance from P to focus is 6
:= by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l2221_222175


namespace NUMINAMATH_CALUDE_geometric_sequence_cubic_root_count_l2221_222101

/-- Given a, b, c form a geometric sequence, the equation ax³ + bx² + cx = 0 has exactly one real root -/
theorem geometric_sequence_cubic_root_count 
  (a b c : ℝ) 
  (h_geom : ∃ (r : ℝ), b = a * r ∧ c = b * r ∧ r ≠ 0) :
  (∃! x : ℝ, a * x^3 + b * x^2 + c * x = 0) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_cubic_root_count_l2221_222101


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_ABCD_l2221_222181

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  sideLength : ℝ

/-- Represents the quadrilateral ABCD formed by the intersection of a plane with the cube -/
structure Quadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculate the area of the quadrilateral ABCD -/
def quadrilateralArea (quad : Quadrilateral) : ℝ := sorry

/-- Main theorem: The area of quadrilateral ABCD is 2√3 -/
theorem area_of_quadrilateral_ABCD :
  let cube := Cube.mk 2
  let A := Point3D.mk 0 0 0
  let C := Point3D.mk 2 2 2
  let B := Point3D.mk (2/3) 2 0
  let D := Point3D.mk 2 (4/3) 2
  let quad := Quadrilateral.mk A B C D
  quadrilateralArea quad = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_ABCD_l2221_222181


namespace NUMINAMATH_CALUDE_sector_radius_l2221_222143

/-- Given a circular sector with perimeter 83 cm and central angle 225 degrees,
    prove that the radius of the circle is 332 / (5π + 8) cm. -/
theorem sector_radius (perimeter : ℝ) (central_angle : ℝ) (radius : ℝ) : 
  perimeter = 83 →
  central_angle = 225 →
  radius = 332 / (5 * Real.pi + 8) →
  perimeter = (central_angle / 360) * 2 * Real.pi * radius + 2 * radius :=
by sorry

end NUMINAMATH_CALUDE_sector_radius_l2221_222143


namespace NUMINAMATH_CALUDE_condition_relationship_l2221_222163

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a - b > 0 → a^2 - b^2 > 0) ∧
  (∃ a b, a^2 - b^2 > 0 ∧ a - b ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2221_222163


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l2221_222160

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 294) (h2 : Nat.gcd a c = 1155) :
  (∀ b' c' : ℕ+, Nat.gcd a b' = 294 → Nat.gcd a c' = 1155 → Nat.gcd b c ≤ Nat.gcd b' c') ∧
  Nat.gcd b c = 21 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l2221_222160


namespace NUMINAMATH_CALUDE_paco_salty_cookies_l2221_222157

/-- Prove that Paco initially had 56 salty cookies -/
theorem paco_salty_cookies 
  (initial_sweet : ℕ) 
  (eaten_sweet : ℕ) 
  (eaten_salty : ℕ) 
  (remaining_sweet : ℕ) 
  (h1 : initial_sweet = 34)
  (h2 : eaten_sweet = 15)
  (h3 : eaten_salty = 56)
  (h4 : remaining_sweet = 19)
  (h5 : initial_sweet = eaten_sweet + remaining_sweet) :
  eaten_salty = 56 := by
  sorry

end NUMINAMATH_CALUDE_paco_salty_cookies_l2221_222157


namespace NUMINAMATH_CALUDE_y_value_proof_l2221_222150

theorem y_value_proof (y : ℝ) :
  (y / 5) / 3 = 15 / (y / 3) → y = 15 * Real.sqrt 3 ∨ y = -15 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l2221_222150


namespace NUMINAMATH_CALUDE_green_balloons_l2221_222149

theorem green_balloons (total : ℕ) (red : ℕ) (green : ℕ) : 
  total = 17 → red = 8 → green = total - red → green = 9 := by
  sorry

end NUMINAMATH_CALUDE_green_balloons_l2221_222149


namespace NUMINAMATH_CALUDE_yellow_bead_cost_l2221_222179

/-- The cost of a box of yellow beads, given the following conditions:
  * Red beads cost $1.30 per box
  * 10 boxes of mixed beads cost $1.72 per box
  * 4 boxes of each color (red and yellow) are used to make the 10 mixed boxes
-/
theorem yellow_bead_cost (red_cost : ℝ) (mixed_cost : ℝ) (red_boxes : ℕ) (yellow_boxes : ℕ) :
  red_cost = 1.30 →
  mixed_cost = 1.72 →
  red_boxes = 4 →
  yellow_boxes = 4 →
  red_boxes * red_cost + yellow_boxes * (3 : ℝ) = 10 * mixed_cost :=
by sorry

end NUMINAMATH_CALUDE_yellow_bead_cost_l2221_222179


namespace NUMINAMATH_CALUDE_carol_extra_invitations_l2221_222193

def invitation_problem (packs_bought : ℕ) (invitations_per_pack : ℕ) (friends_to_invite : ℕ) : ℕ :=
  let total_invitations := packs_bought * invitations_per_pack
  let additional_packs_needed := ((friends_to_invite - total_invitations) + invitations_per_pack - 1) / invitations_per_pack
  let final_invitations := total_invitations + additional_packs_needed * invitations_per_pack
  final_invitations - friends_to_invite

theorem carol_extra_invitations :
  invitation_problem 3 5 23 = 2 :=
sorry

end NUMINAMATH_CALUDE_carol_extra_invitations_l2221_222193


namespace NUMINAMATH_CALUDE_existence_of_increasing_pair_l2221_222135

theorem existence_of_increasing_pair {α : Type*} [LinearOrder α] (a b : ℕ → α) :
  ∃ p q : ℕ, p < q ∧ a p ≤ a q ∧ b p ≤ b q :=
sorry

end NUMINAMATH_CALUDE_existence_of_increasing_pair_l2221_222135


namespace NUMINAMATH_CALUDE_simplify_expression_l2221_222178

theorem simplify_expression (a : ℝ) : (36 * a ^ 9) ^ 4 * (63 * a ^ 9) ^ 4 = a ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2221_222178


namespace NUMINAMATH_CALUDE_total_fish_count_l2221_222131

/-- Represents a fish company with tuna and mackerel counts -/
structure FishCompany where
  tuna : ℕ
  mackerel : ℕ

/-- Calculates the total fish count for a company -/
def totalFish (company : FishCompany) : ℕ :=
  company.tuna + company.mackerel

/-- Theorem stating the total fish count for all three companies -/
theorem total_fish_count 
  (jerk_tuna : FishCompany)
  (tall_tuna : FishCompany)
  (swell_tuna : FishCompany)
  (h1 : jerk_tuna.tuna = 144)
  (h2 : jerk_tuna.mackerel = 80)
  (h3 : tall_tuna.tuna = 2 * jerk_tuna.tuna)
  (h4 : tall_tuna.mackerel = jerk_tuna.mackerel + (30 * jerk_tuna.mackerel) / 100)
  (h5 : swell_tuna.tuna = tall_tuna.tuna + (50 * tall_tuna.tuna) / 100)
  (h6 : swell_tuna.mackerel = jerk_tuna.mackerel + (25 * jerk_tuna.mackerel) / 100) :
  totalFish jerk_tuna + totalFish tall_tuna + totalFish swell_tuna = 1148 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l2221_222131


namespace NUMINAMATH_CALUDE_num_arrangements_eq_162_l2221_222137

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n items --/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of different arrangements for dispatching volunteers --/
def num_arrangements : ℕ :=
  let total_volunteers := 5
  let dispatched_volunteers := 4
  let num_communities := 3
  let scenario1 := choose 3 2 * (choose 4 2 - 1) * arrange 3 3
  let scenario2 := choose 2 1 * choose 4 2 * arrange 3 3
  scenario1 + scenario2

theorem num_arrangements_eq_162 : num_arrangements = 162 := by sorry

end NUMINAMATH_CALUDE_num_arrangements_eq_162_l2221_222137


namespace NUMINAMATH_CALUDE_inequality_proof_l2221_222126

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^3 / (a^3 + 2*b^2)) + (b^3 / (b^3 + 2*c^2)) + (c^3 / (c^3 + 2*a^2)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2221_222126


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l2221_222165

-- Define the equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (1 + m) + y^2 / (1 - m) = 1

-- Define what it means for the equation to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, hyperbola_equation x y m ∧ 
  (1 + m > 0 ∧ 1 - m < 0) ∨ (1 + m < 0 ∧ 1 - m > 0)

-- Theorem stating the range of m for which the equation represents a hyperbola
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ m < -1 ∨ m > 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l2221_222165


namespace NUMINAMATH_CALUDE_chord_length_at_135_degrees_chord_equation_when_bisected_l2221_222162

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define a point on the circle
def P : ℝ × ℝ := (-1, 2)

-- Define the chord AB
structure Chord where
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  passes_through_P : A.1 ≠ B.1 ∧ (P.2 - A.2) / (P.1 - A.1) = (B.2 - A.2) / (B.1 - A.1)

-- Theorem 1
theorem chord_length_at_135_degrees (O : ℝ × ℝ) (r : ℝ) (AB : Chord) :
  O = (0, 0) →
  r^2 = 8 →
  P ∈ Circle O r →
  AB.P = P →
  (AB.B.2 - AB.A.2) / (AB.B.1 - AB.A.1) = -1 →
  Real.sqrt ((AB.A.1 - AB.B.1)^2 + (AB.A.2 - AB.B.2)^2) = Real.sqrt 30 :=
sorry

-- Theorem 2
theorem chord_equation_when_bisected (O : ℝ × ℝ) (r : ℝ) (AB : Chord) :
  O = (0, 0) →
  r^2 = 8 →
  P ∈ Circle O r →
  AB.P = P →
  AB.A.1 - P.1 = P.1 - AB.B.1 →
  AB.A.2 - P.2 = P.2 - AB.B.2 →
  ∃ (a b c : ℝ), a * AB.A.1 + b * AB.A.2 + c = 0 ∧
                 a * AB.B.1 + b * AB.B.2 + c = 0 ∧
                 a = 1 ∧ b = -2 ∧ c = 5 :=
sorry

end NUMINAMATH_CALUDE_chord_length_at_135_degrees_chord_equation_when_bisected_l2221_222162


namespace NUMINAMATH_CALUDE_distance_to_origin_l2221_222189

theorem distance_to_origin (x y n : ℝ) : 
  y = 15 → 
  x = 2 + Real.sqrt 105 → 
  x > 2 → 
  n = Real.sqrt (x^2 + y^2) →
  n = Real.sqrt (334 + 4 * Real.sqrt 105) := by
sorry

end NUMINAMATH_CALUDE_distance_to_origin_l2221_222189


namespace NUMINAMATH_CALUDE_building_block_width_l2221_222158

/-- Given a box and building blocks with specified dimensions, prove that the width of the building block is 2 inches. -/
theorem building_block_width (box_height box_width box_length : ℕ)
  (block_height block_length : ℕ) (num_blocks : ℕ) :
  box_height = 8 →
  box_width = 10 →
  box_length = 12 →
  block_height = 3 →
  block_length = 4 →
  num_blocks = 40 →
  (box_height * box_width * box_length) / num_blocks = block_height * 2 * block_length :=
by sorry

end NUMINAMATH_CALUDE_building_block_width_l2221_222158


namespace NUMINAMATH_CALUDE_expected_girls_left_of_boys_l2221_222183

/-- The number of boys in the lineup -/
def num_boys : ℕ := 10

/-- The number of girls in the lineup -/
def num_girls : ℕ := 7

/-- The total number of students in the lineup -/
def total_students : ℕ := num_boys + num_girls

/-- The expected number of girls standing to the left of all boys -/
def expected_girls_left : ℚ := 7 / 11

theorem expected_girls_left_of_boys :
  let random_arrangement := (Finset.range total_students).powerset
  expected_girls_left = (num_girls : ℚ) / (total_students + 1 : ℚ) := by sorry

end NUMINAMATH_CALUDE_expected_girls_left_of_boys_l2221_222183


namespace NUMINAMATH_CALUDE_average_income_l2221_222156

/-- The average monthly income problem -/
theorem average_income (A B C : ℕ) : 
  (B + C) / 2 = 5250 →
  (A + C) / 2 = 4200 →
  A = 3000 →
  (A + B) / 2 = 4050 := by
sorry

end NUMINAMATH_CALUDE_average_income_l2221_222156


namespace NUMINAMATH_CALUDE_spinner_probability_F_l2221_222120

/-- Represents a spinner with three sections -/
structure Spinner :=
  (D : ℚ) (E : ℚ) (F : ℚ)

/-- The probability of landing on each section of the spinner -/
def probability (s : Spinner) : ℚ := s.D + s.E + s.F

theorem spinner_probability_F (s : Spinner) 
  (hD : s.D = 2/5) 
  (hE : s.E = 1/5) 
  (hP : probability s = 1) : 
  s.F = 2/5 := by
  sorry


end NUMINAMATH_CALUDE_spinner_probability_F_l2221_222120


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2221_222172

theorem sqrt_equation_solution : ∃ (a b c : ℕ+), 
  (2 * Real.sqrt (Real.sqrt 4 - Real.sqrt 3) = Real.sqrt a.val - Real.sqrt b.val + Real.sqrt c.val) ∧
  (a.val + b.val + c.val = 22) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2221_222172


namespace NUMINAMATH_CALUDE_max_value_implies_sum_l2221_222100

/-- The function f(x) = x^3 + ax^2 + bx - a^2 - 7a -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem max_value_implies_sum (a b : ℝ) :
  (∀ x, f a b x ≤ f a b 1) ∧ 
  (f a b 1 = 10) ∧ 
  (f' a b 1 = 0) →
  a + b = 3 := by sorry

end NUMINAMATH_CALUDE_max_value_implies_sum_l2221_222100


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l2221_222176

theorem square_of_binomial_constant (b : ℚ) : 
  (∃ (c : ℚ), ∀ (x : ℚ), 9*x^2 + 27*x + b = (3*x + c)^2) → b = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l2221_222176


namespace NUMINAMATH_CALUDE_sum_of_xy_l2221_222122

theorem sum_of_xy (x y : ℕ+) (h : (2 * x - 5) * (2 * y - 5) = 25) :
  x + y = 18 ∨ x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l2221_222122


namespace NUMINAMATH_CALUDE_fraction_simplification_l2221_222151

theorem fraction_simplification (x y : ℝ) : 
  (2*x + y)/4 + (5*y - 4*x)/6 - y/12 = (-x + 6*y)/6 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2221_222151


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2221_222199

/-- A line with slope k passing through a fixed point (x₀, y₀) -/
def line_equation (k : ℝ) (x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = k * (x - x₀)

/-- The fixed point theorem for a family of lines -/
theorem fixed_point_theorem :
  ∃! p : ℝ × ℝ, ∀ k : ℝ, line_equation k p.1 p.2 (-3) 4 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2221_222199


namespace NUMINAMATH_CALUDE_value_of_c_l2221_222152

theorem value_of_c (a b c : ℝ) 
  (h1 : 12 = 0.06 * a) 
  (h2 : 6 = 0.12 * b) 
  (h3 : c = b / a) : 
  c = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l2221_222152


namespace NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l2221_222198

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_distribution_unique_parameters (ξ : BinomialRV) 
  (h_exp : expectation ξ = 12) 
  (h_var : variance ξ = 2.4) : 
  ξ.n = 15 ∧ ξ.p = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l2221_222198


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2221_222191

theorem imaginary_part_of_z (z : ℂ) (h : (Complex.I - 1) * z = 2) : 
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2221_222191


namespace NUMINAMATH_CALUDE_train_length_l2221_222197

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 54 → time_s = 9 → speed_kmh * (1000 / 3600) * time_s = 135 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2221_222197


namespace NUMINAMATH_CALUDE_problem_statement_l2221_222111

theorem problem_statement :
  ∀ (a b x y z : ℝ),
    (a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b)) ∧
    (let c := z^2 + 2*x + Real.pi/6;
     a = x^2 + 2*y + Real.pi/2 ∧
     b = y^2 + 2*z + Real.pi/3 →
     max a (max b c) > 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2221_222111


namespace NUMINAMATH_CALUDE_combination_equality_implies_three_l2221_222196

theorem combination_equality_implies_three (x : ℕ) : 
  (Nat.choose 5 x = Nat.choose 5 (x - 1)) → x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_combination_equality_implies_three_l2221_222196


namespace NUMINAMATH_CALUDE_blakes_change_is_correct_l2221_222144

/-- Calculates the change Blake receives after buying candy with discounts -/
def blakes_change (lollipop_price : ℚ) (gummy_price : ℚ) (candy_bar_price : ℚ) : ℚ :=
  let chocolate_price := 4 * lollipop_price
  let lollipop_cost := 3 * lollipop_price + lollipop_price / 2
  let chocolate_cost := 4 * chocolate_price + 2 * (chocolate_price * 3 / 4)
  let gummy_cost := 3 * gummy_price
  let candy_bar_cost := 5 * candy_bar_price
  let total_cost := lollipop_cost + chocolate_cost + gummy_cost + candy_bar_cost
  let total_given := 4 * 20 + 2 * 5 + 5 * 1
  total_given - total_cost

/-- Theorem stating that Blake's change is $27.50 -/
theorem blakes_change_is_correct :
  blakes_change 2 3 (3/2) = 55/2 := by sorry

end NUMINAMATH_CALUDE_blakes_change_is_correct_l2221_222144


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l2221_222112

theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  4 * Real.pi * radius^2 = 50 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l2221_222112


namespace NUMINAMATH_CALUDE_area_of_triangle_def_is_nine_l2221_222136

/-- A triangle with vertices on the sides of a rectangle -/
structure TriangleInRectangle where
  /-- Width of the rectangle -/
  width : ℝ
  /-- Height of the rectangle -/
  height : ℝ
  /-- x-coordinate of vertex D -/
  dx : ℝ
  /-- y-coordinate of vertex D -/
  dy : ℝ
  /-- x-coordinate of vertex E -/
  ex : ℝ
  /-- y-coordinate of vertex E -/
  ey : ℝ
  /-- x-coordinate of vertex F -/
  fx : ℝ
  /-- y-coordinate of vertex F -/
  fy : ℝ
  /-- Ensure D is on the left side of the rectangle -/
  hd : dx = 0 ∧ 0 ≤ dy ∧ dy ≤ height
  /-- Ensure E is on the bottom side of the rectangle -/
  he : ey = 0 ∧ 0 ≤ ex ∧ ex ≤ width
  /-- Ensure F is on the top side of the rectangle -/
  hf : fy = height ∧ 0 ≤ fx ∧ fx ≤ width

/-- Calculate the area of the triangle DEF -/
def areaOfTriangleDEF (t : TriangleInRectangle) : ℝ :=
  sorry

/-- Theorem stating that the area of triangle DEF is 9 square units -/
theorem area_of_triangle_def_is_nine (t : TriangleInRectangle) 
    (h_width : t.width = 6) 
    (h_height : t.height = 4)
    (h_d : t.dx = 0 ∧ t.dy = 2)
    (h_e : t.ex = 6 ∧ t.ey = 0)
    (h_f : t.fx = 3 ∧ t.fy = 4) : 
  areaOfTriangleDEF t = 9 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_def_is_nine_l2221_222136


namespace NUMINAMATH_CALUDE_stratified_sample_correct_l2221_222105

/-- Represents the number of people to be selected from each group in a stratified sampling --/
structure StratifiedSample where
  regular : ℕ
  middle : ℕ
  senior : ℕ

/-- Calculates the stratified sample given total employees and managers --/
def calculateStratifiedSample (total : ℕ) (middle : ℕ) (senior : ℕ) (toSelect : ℕ) : StratifiedSample :=
  sorry

/-- Theorem stating that the calculated stratified sample is correct --/
theorem stratified_sample_correct :
  let total := 160
  let middle := 30
  let senior := 10
  let toSelect := 20
  let result := calculateStratifiedSample total middle senior toSelect
  result.regular = 16 ∧ result.middle = 3 ∧ result.senior = 1 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_correct_l2221_222105


namespace NUMINAMATH_CALUDE_second_class_average_l2221_222130

theorem second_class_average (students1 : ℕ) (students2 : ℕ) (avg1 : ℝ) (avg_combined : ℝ) :
  students1 = 25 →
  students2 = 30 →
  avg1 = 40 →
  avg_combined = 50.90909090909091 →
  ((students1 * avg1 + students2 * 60) / (students1 + students2) = avg_combined) := by
sorry

end NUMINAMATH_CALUDE_second_class_average_l2221_222130


namespace NUMINAMATH_CALUDE_alan_pine_trees_l2221_222184

/-- The number of pine cones dropped by each tree -/
def pine_cones_per_tree : ℕ := 200

/-- The percentage of pine cones that fall on Alan's roof -/
def roof_percentage : ℚ := 30 / 100

/-- The weight of each pine cone in ounces -/
def pine_cone_weight : ℕ := 4

/-- The total weight of pine cones on Alan's roof in ounces -/
def total_roof_weight : ℕ := 1920

/-- The number of pine trees in Alan's backyard -/
def num_trees : ℕ := 8

theorem alan_pine_trees :
  num_trees * (pine_cones_per_tree * roof_percentage).floor * pine_cone_weight = total_roof_weight :=
sorry

end NUMINAMATH_CALUDE_alan_pine_trees_l2221_222184


namespace NUMINAMATH_CALUDE_ceiling_floor_calculation_l2221_222166

theorem ceiling_floor_calculation : 
  ⌈(12 / 5 : ℚ) * (((-19 : ℚ) / 4) - 3)⌉ - ⌊(12 / 5 : ℚ) * ⌊(-19 : ℚ) / 4⌋⌋ = -6 :=
by sorry

end NUMINAMATH_CALUDE_ceiling_floor_calculation_l2221_222166


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2221_222108

theorem arithmetic_computation : -(12 * 2) - (3 * 2) + (-18 / 3 * -4) = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2221_222108


namespace NUMINAMATH_CALUDE_sunshine_car_rentals_rate_l2221_222102

theorem sunshine_car_rentals_rate (sunshine_daily_rate city_daily_rate city_mile_rate : ℚ)
  (equal_cost_miles : ℕ) :
  sunshine_daily_rate = 17.99 ∧
  city_daily_rate = 18.95 ∧
  city_mile_rate = 0.16 ∧
  equal_cost_miles = 48 →
  ∃ sunshine_mile_rate : ℚ,
    sunshine_mile_rate = 0.18 ∧
    sunshine_daily_rate + sunshine_mile_rate * equal_cost_miles =
    city_daily_rate + city_mile_rate * equal_cost_miles :=
by sorry

end NUMINAMATH_CALUDE_sunshine_car_rentals_rate_l2221_222102


namespace NUMINAMATH_CALUDE_area_of_two_sectors_l2221_222164

/-- The area of a figure formed by two sectors of a circle -/
theorem area_of_two_sectors (r : ℝ) (angle1 angle2 : ℝ) (h1 : r = 10) (h2 : angle1 = 45) (h3 : angle2 = 90) :
  (angle1 / 360) * π * r^2 + (angle2 / 360) * π * r^2 = 37.5 * π := by
  sorry

end NUMINAMATH_CALUDE_area_of_two_sectors_l2221_222164


namespace NUMINAMATH_CALUDE_circle_area_theorem_l2221_222129

theorem circle_area_theorem (r : ℝ) (chord_length : ℝ) (intersection_distance : ℝ)
  (h1 : r = 42)
  (h2 : chord_length = 78)
  (h3 : intersection_distance = 18) :
  ∃ (m n d : ℕ), 
    (m * π - n * Real.sqrt d : ℝ) = 294 * π - 81 * Real.sqrt 3 ∧
    m + n + d = 378 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l2221_222129


namespace NUMINAMATH_CALUDE_complement_characterization_l2221_222121

-- Define the universe of quadrilaterals
def Quadrilateral : Type := sorry

-- Define properties of quadrilaterals
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def has_right_angle (q : Quadrilateral) : Prop := sorry

-- Define sets A and B
def A : Set Quadrilateral := {q | is_rhombus q ∨ is_rectangle q}
def B : Set Quadrilateral := {q | is_rectangle q}

-- Define the complement of B with respect to A
def C_AB : Set Quadrilateral := A \ B

-- Theorem to prove
theorem complement_characterization :
  C_AB = {q : Quadrilateral | is_rhombus q ∧ ¬has_right_angle q} :=
sorry

end NUMINAMATH_CALUDE_complement_characterization_l2221_222121


namespace NUMINAMATH_CALUDE_root_multiplicity_two_l2221_222148

variable (n : ℕ)

def f (A B : ℝ) (x : ℝ) : ℝ := A * x^(n+1) + B * x^n + 1

theorem root_multiplicity_two (A B : ℝ) :
  (f n A B 1 = 0 ∧ (deriv (f n A B)) 1 = 0) ↔ (A = n ∧ B = -(n+1)) := by sorry

end NUMINAMATH_CALUDE_root_multiplicity_two_l2221_222148


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2221_222138

theorem sum_of_fractions : (1 / 1.01) + (1 / 1.1) + (1 / 1) + (1 / 11) + (1 / 101) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2221_222138


namespace NUMINAMATH_CALUDE_geometric_sequence_existence_l2221_222145

theorem geometric_sequence_existence : ∃ (a r : ℝ), 
  a * r = 2 ∧ 
  a * r^3 = 6 ∧ 
  a = -2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_existence_l2221_222145


namespace NUMINAMATH_CALUDE_fifty_percent_greater_than_88_l2221_222167

theorem fifty_percent_greater_than_88 (x : ℝ) : x = 88 * 1.5 → x = 132 := by
  sorry

end NUMINAMATH_CALUDE_fifty_percent_greater_than_88_l2221_222167


namespace NUMINAMATH_CALUDE_water_level_rise_l2221_222168

/-- Given a cube with edge length 15 cm and a rectangular vessel with base dimensions 20 cm × 15 cm,
    prove that the rise in water level when the cube is fully immersed is 11.25 cm. -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length : ℝ) (vessel_width : ℝ) :
  cube_edge = 15 →
  vessel_length = 20 →
  vessel_width = 15 →
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 11.25 := by
  sorry

#check water_level_rise

end NUMINAMATH_CALUDE_water_level_rise_l2221_222168


namespace NUMINAMATH_CALUDE_original_number_exists_and_unique_l2221_222188

theorem original_number_exists_and_unique : 
  ∃! x : ℚ, 4 * (3 * x + 29) = 212 := by sorry

end NUMINAMATH_CALUDE_original_number_exists_and_unique_l2221_222188


namespace NUMINAMATH_CALUDE_floor_nested_expression_l2221_222192

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

theorem floor_nested_expression : floor (-2.3 + floor 1.6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_nested_expression_l2221_222192


namespace NUMINAMATH_CALUDE_frog_eggs_difference_l2221_222134

/-- Represents the number of eggs laid by a frog over 4 days -/
def FrogEggs : Type :=
  { eggs : Fin 4 → ℕ // 
    eggs 0 = 50 ∧ 
    eggs 1 = 2 * eggs 0 ∧ 
    eggs 3 = 2 * (eggs 0 + eggs 1 + eggs 2) ∧
    eggs 0 + eggs 1 + eggs 2 + eggs 3 = 810 }

/-- The difference between eggs laid on the third day and second day is 20 -/
theorem frog_eggs_difference (e : FrogEggs) : e.val 2 - e.val 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_frog_eggs_difference_l2221_222134


namespace NUMINAMATH_CALUDE_oddProbabilityConvergesTo1Third_l2221_222185

/-- Represents the state of the calculator --/
structure CalculatorState where
  display : ℕ
  lastOperation : Option (ℕ → ℕ → ℕ)

/-- Represents a button press on the calculator --/
inductive ButtonPress
  | Digit (d : Fin 10)
  | Add
  | Multiply

/-- The probability of the display showing an odd number after n button presses --/
def oddProbability (n : ℕ) : ℝ := sorry

/-- The limiting probability of the display showing an odd number as n approaches infinity --/
def limitingOddProbability : ℝ := sorry

/-- The main theorem stating that the limiting probability converges to 1/3 --/
theorem oddProbabilityConvergesTo1Third :
  limitingOddProbability = 1/3 := by sorry

end NUMINAMATH_CALUDE_oddProbabilityConvergesTo1Third_l2221_222185


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l2221_222186

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- A line passing through a point (x₀, y₀) -/
structure Line where
  x₀ : ℝ
  y₀ : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The problem statement -/
theorem circle_and_tangent_line 
  (C : Circle) 
  (l : Line) :
  C.h = 2 ∧ 
  C.k = 3 ∧ 
  C.r = 1 ∧
  l.x₀ = 1 ∧ 
  l.y₀ = 0 →
  (∀ x y : ℝ, (x - C.h)^2 + (y - C.k)^2 = C.r^2) ∧
  ((l.a = 1 ∧ l.b = 0 ∧ l.c = -1) ∨
   (l.a = 4 ∧ l.b = -3 ∧ l.c = -4)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l2221_222186


namespace NUMINAMATH_CALUDE_system_solution_l2221_222173

theorem system_solution : 
  ∀ x y z : ℝ, 
    (y * z = 3 * y + 2 * z - 8) ∧ 
    (z * x = 4 * z + 3 * x - 8) ∧ 
    (x * y = 2 * x + y - 1) → 
    ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 5/2 ∧ z = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2221_222173


namespace NUMINAMATH_CALUDE_divisors_of_86400000_l2221_222113

/-- The number of divisors of 86,400,000 -/
def num_divisors : ℕ := 264

/-- The sum of all divisors of 86,400,000 -/
def sum_divisors : ℕ := 319823280

/-- The prime factorization of 86,400,000 -/
def n : ℕ := 2^10 * 3^3 * 5^5

theorem divisors_of_86400000 :
  (∃ (d : Finset ℕ), d.card = num_divisors ∧ 
    (∀ x : ℕ, x ∈ d ↔ x ∣ n) ∧
    d.sum id = sum_divisors) :=
sorry

end NUMINAMATH_CALUDE_divisors_of_86400000_l2221_222113


namespace NUMINAMATH_CALUDE_defective_product_arrangements_l2221_222118

theorem defective_product_arrangements :
  let total_products : ℕ := 7
  let defective_products : ℕ := 4
  let non_defective_products : ℕ := 3
  let third_defective_position : ℕ := 4

  (Nat.choose non_defective_products 1) *
  (Nat.choose defective_products 1) *
  (Nat.choose (defective_products - 1) 1) *
  1 *
  (Nat.choose 2 1) *
  ((total_products - third_defective_position) - (defective_products - 3)) = 288 :=
by sorry

end NUMINAMATH_CALUDE_defective_product_arrangements_l2221_222118


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2221_222169

theorem quadratic_real_roots (n : ℕ+) :
  (∃ x : ℝ, x^2 - 4*x + n.val = 0) ↔ n.val = 1 ∨ n.val = 2 ∨ n.val = 3 ∨ n.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2221_222169


namespace NUMINAMATH_CALUDE_job_completion_time_l2221_222182

theorem job_completion_time (time_a time_b : ℝ) (h1 : time_a = 5) (h2 : time_b = 15) :
  let combined_time := 1 / (1 / time_a + 1 / time_b)
  combined_time = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2221_222182


namespace NUMINAMATH_CALUDE_leonards_age_l2221_222146

theorem leonards_age (leonard nina jerome : ℕ) 
  (h1 : leonard = nina - 4)
  (h2 : nina = jerome / 2)
  (h3 : leonard + nina + jerome = 36) :
  leonard = 6 := by
sorry

end NUMINAMATH_CALUDE_leonards_age_l2221_222146


namespace NUMINAMATH_CALUDE_find_divisor_l2221_222154

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 23) (h2 : quotient = 4) (h3 : remainder = 3) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 5 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2221_222154
