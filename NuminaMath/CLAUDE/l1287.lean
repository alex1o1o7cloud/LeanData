import Mathlib

namespace triangle_inequality_l1287_128752

/-- Given a triangle ABC with sides a ≤ b ≤ c, angle bisectors l_a, l_b, l_c,
    and corresponding medians m_a, m_b, m_c, prove that
    h_n/m_n + h_n/m_h_n + l_c/m_m_p > 1 -/
theorem triangle_inequality (a b c : ℝ) (h_sides : 0 < a ∧ a ≤ b ∧ b ≤ c)
  (l_a l_b l_c : ℝ) (h_bisectors : l_a > 0 ∧ l_b > 0 ∧ l_c > 0)
  (m_a m_b m_c : ℝ) (h_medians : m_a > 0 ∧ m_b > 0 ∧ m_c > 0)
  (h_n m_n m_h_n m_m_p : ℝ) (h_positive : h_n > 0 ∧ m_n > 0 ∧ m_h_n > 0 ∧ m_m_p > 0) :
  h_n / m_n + h_n / m_h_n + l_c / m_m_p > 1 := by
  sorry

end triangle_inequality_l1287_128752


namespace expression_evaluation_l1287_128795

theorem expression_evaluation : 200 * (200 - 7) - (200 * 200 - 7 * 3) = -1379 := by
  sorry

end expression_evaluation_l1287_128795


namespace added_number_proof_l1287_128701

theorem added_number_proof : 
  let n : ℝ := 90
  let x : ℝ := 3
  (1/2 : ℝ) * (1/3 : ℝ) * (1/5 : ℝ) * n + x = (1/15 : ℝ) * n := by
  sorry

end added_number_proof_l1287_128701


namespace max_distance_line_theorem_l1287_128713

/-- The line equation that passes through point A(1, 2) and is at the maximum distance from the origin -/
def max_distance_line : ℝ → ℝ → Prop :=
  fun x y => x + 2 * y - 5 = 0

/-- Point A -/
def point_A : ℝ × ℝ := (1, 2)

/-- The origin -/
def origin : ℝ × ℝ := (0, 0)

theorem max_distance_line_theorem :
  (max_distance_line (point_A.1) (point_A.2)) ∧
  (∀ x y, max_distance_line x y → 
    ∀ a b, (a, b) ≠ origin → 
      (a - origin.1)^2 + (b - origin.2)^2 ≤ (x - origin.1)^2 + (y - origin.2)^2) :=
sorry

end max_distance_line_theorem_l1287_128713


namespace complement_of_A_l1287_128779

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | x < -1 ∨ 2 ≤ x} := by sorry

end complement_of_A_l1287_128779


namespace quadratic_expression_value_l1287_128703

theorem quadratic_expression_value (a : ℝ) (h : 2 * a^2 - a - 3 = 0) :
  (2 * a + 3) * (2 * a - 3) + (2 * a - 1)^2 = 4 := by
  sorry

end quadratic_expression_value_l1287_128703


namespace expression_evaluation_l1287_128770

theorem expression_evaluation : (28 + 48 / 69) * 69 = 1980 := by
  sorry

end expression_evaluation_l1287_128770


namespace elder_sister_age_when_sum_was_twenty_l1287_128745

/-- 
Given:
- The younger sister is currently 18 years old
- The elder sister is currently 26 years old
- At some point in the past, the sum of their ages was 20 years

Prove that when the sum of their ages was 20 years, the elder sister was 14 years old.
-/
theorem elder_sister_age_when_sum_was_twenty 
  (younger_current : ℕ) 
  (elder_current : ℕ) 
  (years_ago : ℕ) 
  (h1 : younger_current = 18) 
  (h2 : elder_current = 26) 
  (h3 : younger_current - years_ago + elder_current - years_ago = 20) : 
  elder_current - years_ago = 14 :=
sorry

end elder_sister_age_when_sum_was_twenty_l1287_128745


namespace infinite_series_sum_l1287_128715

/-- The sum of the infinite series ∑(k=1 to ∞) [6^k / ((3^k - 2^k)(3^(k+1) - 2^(k+1)))] is equal to 2. -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (6:ℝ)^k / ((3:ℝ)^k - (2:ℝ)^k * ((3:ℝ)^(k+1) - (2:ℝ)^(k+1)))) = 2 := by
  sorry

end infinite_series_sum_l1287_128715


namespace vector_equality_conditions_l1287_128734

theorem vector_equality_conditions (n : ℕ) :
  ∃ (a b : Fin n → ℝ),
    (norm a = norm b ∧ norm (a + b) ≠ norm (a - b)) ∧
    ∃ (c d : Fin n → ℝ),
      (norm (c + d) = norm (c - d) ∧ norm c ≠ norm d) :=
by sorry

end vector_equality_conditions_l1287_128734


namespace defective_products_m1_l1287_128786

theorem defective_products_m1 (m1_production m2_production m3_production : ℝ)
  (m2_defective_rate m3_defective_rate total_defective_rate : ℝ) :
  m1_production = 0.4 →
  m2_production = 0.3 →
  m3_production = 0.3 →
  m2_defective_rate = 0.01 →
  m3_defective_rate = 0.07 →
  total_defective_rate = 0.036 →
  ∃ (m1_defective_rate : ℝ),
    m1_defective_rate * m1_production +
    m2_defective_rate * m2_production +
    m3_defective_rate * m3_production = total_defective_rate ∧
    m1_defective_rate = 0.03 := by
  sorry

end defective_products_m1_l1287_128786


namespace solution_existence_l1287_128788

theorem solution_existence (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) 
  (h : x + y + 2 * Real.sqrt (x * y) = 2017) :
  (x = 0 ∧ y = 2017) ∨ (x = 2017 ∧ y = 0) := by
  sorry

end solution_existence_l1287_128788


namespace parallel_postulate_l1287_128740

-- Define a line in a 2D Euclidean plane
structure Line where
  -- You might represent a line using two points or a point and a direction
  -- For simplicity, we'll leave the internal representation abstract
  dummy : Unit

-- Define a point in a 2D Euclidean plane
structure Point where
  -- You might represent a point using x and y coordinates
  -- For simplicity, we'll leave the internal representation abstract
  dummy : Unit

-- Define what it means for a point to not be on a line
def Point.notOn (p : Point) (l : Line) : Prop := sorry

-- Define what it means for two lines to be parallel
def Line.parallel (l1 l2 : Line) : Prop := sorry

-- Define what it means for a line to pass through a point
def Line.passesThroughPoint (l : Line) (p : Point) : Prop := sorry

-- The parallel postulate
theorem parallel_postulate (L : Line) (P : Point) (h : P.notOn L) :
  ∃! L' : Line, L'.parallel L ∧ L'.passesThroughPoint P := by sorry

end parallel_postulate_l1287_128740


namespace no_consistent_solution_l1287_128716

-- Define the types for teams and match results
inductive Team : Type
| Spartak | Dynamo | Zenit | Lokomotiv

structure MatchResult :=
(winner : Team)
(loser : Team)

-- Define the problem setup
def problem_setup (match1 match2 : MatchResult) (fan_count : Team → ℕ) : Prop :=
  match1.winner ≠ match1.loser ∧ 
  match2.winner ≠ match2.loser ∧
  match1.winner ≠ match2.winner ∧
  (fan_count Team.Spartak + fan_count match1.loser + fan_count match2.loser = 200) ∧
  (fan_count Team.Dynamo + fan_count match1.loser + fan_count match2.loser = 300) ∧
  (fan_count Team.Zenit = 500) ∧
  (fan_count Team.Lokomotiv = 600)

-- Theorem statement
theorem no_consistent_solution :
  ∀ (match1 match2 : MatchResult) (fan_count : Team → ℕ),
  problem_setup match1 match2 fan_count → False :=
sorry

end no_consistent_solution_l1287_128716


namespace factorial_250_trailing_zeros_l1287_128731

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 250! ends with 62 zeros -/
theorem factorial_250_trailing_zeros :
  trailingZeros 250 = 62 := by
  sorry

end factorial_250_trailing_zeros_l1287_128731


namespace max_intersections_three_lines_one_circle_l1287_128719

-- Define a type for geometric figures
inductive Figure
| Line : Figure
| Circle : Figure

-- Define a function to count maximum intersections between two figures
def maxIntersections (f1 f2 : Figure) : ℕ :=
  match f1, f2 with
  | Figure.Line, Figure.Line => 1
  | Figure.Line, Figure.Circle => 2
  | Figure.Circle, Figure.Line => 2
  | Figure.Circle, Figure.Circle => 0

-- Theorem statement
theorem max_intersections_three_lines_one_circle :
  ∃ (l1 l2 l3 : Figure) (c : Figure),
    l1 = Figure.Line ∧ l2 = Figure.Line ∧ l3 = Figure.Line ∧ c = Figure.Circle ∧
    l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 ∧
    (maxIntersections l1 l2 + maxIntersections l2 l3 + maxIntersections l1 l3 +
     maxIntersections l1 c + maxIntersections l2 c + maxIntersections l3 c) = 9 :=
by
  sorry

end max_intersections_three_lines_one_circle_l1287_128719


namespace imaginary_unit_multiplication_l1287_128709

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_unit_multiplication :
  i * (1 - i) = 1 + i := by sorry

end imaginary_unit_multiplication_l1287_128709


namespace ceiling_sqrt_twelve_count_l1287_128743

theorem ceiling_sqrt_twelve_count : 
  (Finset.filter (fun x : ℕ => ⌈Real.sqrt x⌉ = 12) (Finset.range 1000)).card = 25 := by
  sorry

end ceiling_sqrt_twelve_count_l1287_128743


namespace apple_piles_l1287_128737

/-- Given two piles of apples, prove the original number in the second pile -/
theorem apple_piles (a : ℕ) : 
  (∃ b : ℕ, (a - 2) * 2 = b + 2) → 
  (∃ b : ℕ, b = 2 * a - 6) :=
by sorry

end apple_piles_l1287_128737


namespace units_digit_of_expression_l1287_128705

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_expression : units_digit (8 * 19 * 1981 + 6^3 - 2^5) = 6 := by
  sorry

end units_digit_of_expression_l1287_128705


namespace probability_both_presidents_selected_l1287_128748

def club_sizes : List Nat := [6, 8, 9, 10]

def probability_both_presidents (n : Nat) : Rat :=
  (Nat.choose (n - 2) 2 : Rat) / (Nat.choose n 4 : Rat)

theorem probability_both_presidents_selected :
  (1 / 4 : Rat) * (club_sizes.map probability_both_presidents).sum = 119 / 700 := by
  sorry

end probability_both_presidents_selected_l1287_128748


namespace cube_root_sum_of_eighth_powers_l1287_128706

theorem cube_root_sum_of_eighth_powers (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 3*a + 1 = 0 →
  b^3 - 3*b + 1 = 0 →
  c^3 - 3*c + 1 = 0 →
  a^8 + b^8 + c^8 = 186 := by
sorry

end cube_root_sum_of_eighth_powers_l1287_128706


namespace polynomial_factorization_l1287_128760

theorem polynomial_factorization :
  ∀ x : ℝ, x^12 + x^6 + 1 = (x^4 - x^2 + 1) * (x^8 + x^4 + 1) := by
  sorry

end polynomial_factorization_l1287_128760


namespace price_change_theorem_l1287_128714

/-- Proves that a price of $100 after a 10% increase followed by a 10% decrease results in $99 -/
theorem price_change_theorem (initial_price : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) : 
  initial_price = 100 ∧ 
  increase_rate = 0.1 ∧ 
  decrease_rate = 0.1 → 
  initial_price * (1 + increase_rate) * (1 - decrease_rate) = 99 :=
by
  sorry

#check price_change_theorem

end price_change_theorem_l1287_128714


namespace prime_product_660_l1287_128754

theorem prime_product_660 (w x y z a b c d : ℕ) : 
  (w.Prime ∧ x.Prime ∧ y.Prime ∧ z.Prime) →
  (w < x ∧ x < y ∧ y < z) →
  ((w^a) * (x^b) * (y^c) * (z^d) = 660) →
  ((a + b) - (c + d) = 1) →
  a = 2 := by
sorry

end prime_product_660_l1287_128754


namespace product_closure_l1287_128780

-- Define the set A
def A : Set ℤ := {z | ∃ (a b k n : ℤ), z = a^2 + k*a*b + n*b^2}

-- State the theorem
theorem product_closure (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := by
  sorry

end product_closure_l1287_128780


namespace expression_simplification_and_evaluation_l1287_128769

theorem expression_simplification_and_evaluation :
  let x : ℝ := 3 * Real.cos (60 * π / 180)
  let original_expression := (2 * x) / (x + 1) - (2 * x - 4) / (x^2 - 1) / ((x - 2) / (x^2 - 2 * x + 1))
  let simplified_expression := 4 / (x + 1)
  original_expression = simplified_expression ∧ simplified_expression = 8 / 5 := by
  sorry

end expression_simplification_and_evaluation_l1287_128769


namespace quadratic_root_in_interval_l1287_128730

theorem quadratic_root_in_interval (a b c : ℝ) (h : 2*a + 3*b + 6*c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end quadratic_root_in_interval_l1287_128730


namespace quadratic_root_implies_m_value_l1287_128781

/-- The quadratic equation mx^2 + x - m^2 + 1 = 0 has -1 as a root if and only if m = 1 -/
theorem quadratic_root_implies_m_value (m : ℝ) : 
  (m * (-1)^2 + (-1) - m^2 + 1 = 0) ↔ (m = 1) := by sorry

end quadratic_root_implies_m_value_l1287_128781


namespace shaded_area_calculation_l1287_128796

/-- Represents a triangle with side lengths and an angle -/
structure Triangle :=
  (sideAB : ℝ)
  (sideAC : ℝ)
  (sideBC : ℝ)
  (angleBAC : ℝ)

/-- Represents a circle with a radius -/
structure Circle :=
  (radius : ℝ)

/-- Calculates the area of two shaded regions in a specific geometric configuration -/
def shadedArea (t : Triangle) (c : Circle) : ℝ :=
  sorry

theorem shaded_area_calculation (t : Triangle) (c : Circle) :
  t.sideAB = 16 ∧ t.sideAC = 16 ∧ t.sideBC = c.radius * 2 ∧ t.angleBAC = 120 * π / 180 →
  shadedArea t c = 43 * π - 128 * Real.sqrt 3 :=
by sorry

end shaded_area_calculation_l1287_128796


namespace property_of_x_l1287_128765

theorem property_of_x (x : ℝ) (h1 : x > 0) :
  (100 - x) / 100 * x = 16 → x = 40 ∨ x = 60 := by
  sorry

end property_of_x_l1287_128765


namespace max_value_theorem_l1287_128708

theorem max_value_theorem (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  (∃ x y z : ℝ, 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 ∧ 8 * x + 3 * y + 5 * z > 8 * a + 3 * b + 5 * c) →
  8 * a + 3 * b + 5 * c ≤ Real.sqrt (373 / 36) :=
sorry

end max_value_theorem_l1287_128708


namespace orange_count_l1287_128797

/-- Represents the number of oranges in a basket -/
structure Basket where
  good : ℕ
  bad : ℕ

/-- Defines the ratio between good and bad oranges -/
def hasRatio (b : Basket) (g : ℕ) (d : ℕ) : Prop :=
  g * b.bad = d * b.good

theorem orange_count (b : Basket) (h1 : hasRatio b 3 1) (h2 : b.bad = 8) : b.good = 24 := by
  sorry

end orange_count_l1287_128797


namespace surface_area_circumscribed_sphere_l1287_128744

/-- The surface area of a sphere circumscribing a cube with edge length 1 is 3π. -/
theorem surface_area_circumscribed_sphere (cube_edge : Real) (sphere_radius : Real) :
  cube_edge = 1 →
  sphere_radius = (Real.sqrt 3) / 2 →
  4 * Real.pi * sphere_radius ^ 2 = 3 * Real.pi :=
by sorry

end surface_area_circumscribed_sphere_l1287_128744


namespace polygon_sides_for_900_degrees_l1287_128728

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180° --/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- For a polygon with n sides and sum of interior angles equal to 900°, n = 7 --/
theorem polygon_sides_for_900_degrees (n : ℕ) :
  sum_interior_angles n = 900 → n = 7 := by
  sorry

end polygon_sides_for_900_degrees_l1287_128728


namespace binomial_expansion_coefficient_l1287_128763

theorem binomial_expansion_coefficient (a b : ℝ) :
  (∃ x, (1 + a*x)^5 = 1 + 10*x + b*x^2 + a^3*x^3 + a^4*x^4 + a^5*x^5) →
  b = 40 := by
  sorry

end binomial_expansion_coefficient_l1287_128763


namespace cabbage_production_increase_cabbage_production_increase_holds_l1287_128757

theorem cabbage_production_increase : ℕ → Prop :=
  fun n =>
    (∃ a : ℕ, a * a = 11236) ∧
    (∀ b : ℕ, b * b < 11236 → b * b ≤ (n - 1) * (n - 1)) ∧
    (n * n < 11236) →
    11236 - (n - 1) * (n - 1) = 211

theorem cabbage_production_increase_holds : cabbage_production_increase 106 := by
  sorry

end cabbage_production_increase_cabbage_production_increase_holds_l1287_128757


namespace function_properties_l1287_128722

open Real

theorem function_properties (e : ℝ) (h_e : e = exp 1) :
  let f (a : ℝ) (x : ℝ) := a * x - log x
  let g (x : ℝ) := (log x) / x
  
  -- Part 1
  (∀ x ∈ Set.Ioo 0 e, |f 1 x| > g x + 1/2) ∧
  (∃ x₀ ∈ Set.Ioo 0 e, ∀ x ∈ Set.Ioo 0 e, f 1 x₀ ≤ f 1 x) ∧
  (∃ x₀ ∈ Set.Ioo 0 e, f 1 x₀ = 1) ∧
  
  -- Part 2
  (∃ a : ℝ, ∃ x₀ ∈ Set.Ioo 0 e, 
    (∀ x ∈ Set.Ioo 0 e, f a x₀ ≤ f a x) ∧
    f a x₀ = 3 ∧
    a = e^2) := by
  sorry

end function_properties_l1287_128722


namespace sqrt_real_iff_nonneg_l1287_128753

theorem sqrt_real_iff_nonneg (a : ℝ) : ∃ (x : ℝ), x ^ 2 = a ↔ a ≥ 0 := by sorry

end sqrt_real_iff_nonneg_l1287_128753


namespace arithmetic_sequence_sum_l1287_128787

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum (seq.a ∘ Nat.succ)

/-- Main theorem -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  S seq 8 - S seq 3 = 10 → S seq 11 = 16 := by
  sorry

end arithmetic_sequence_sum_l1287_128787


namespace unique_factorial_equation_l1287_128727

theorem unique_factorial_equation : ∃! (N : ℕ), N > 0 ∧ ∃ (m : ℕ), m > 0 ∧ (7 : ℕ).factorial * (11 : ℕ).factorial = 20 * m * N.factorial := by
  sorry

end unique_factorial_equation_l1287_128727


namespace tan_3x_eq_sin_x_solutions_l1287_128790

open Real

theorem tan_3x_eq_sin_x_solutions (x : ℝ) :
  ∃ (s : Finset ℝ), s.card = 12 ∧
  (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2*π ∧ tan (3*x) = sin x) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 2*π ∧ tan (3*y) = sin y → y ∈ s) := by
  sorry

end tan_3x_eq_sin_x_solutions_l1287_128790


namespace inequality_range_l1287_128704

theorem inequality_range (a : ℝ) : 
  (∀ x y, x ∈ Set.Icc 0 (π/6) → y ∈ Set.Ioi 0 → 
    y/4 - 2*(Real.cos x)^2 ≥ a*(Real.sin x) - 9/y) → 
  a ≤ 3 := by sorry

end inequality_range_l1287_128704


namespace arithmetic_sequence_quadratic_root_l1287_128751

/-- Given real numbers x, y, z forming an arithmetic sequence with x ≥ y ≥ z ≥ 0,
    and the quadratic equation yx^2 + zx + y = 0 having exactly one root,
    prove that this root is 4. -/
theorem arithmetic_sequence_quadratic_root :
  ∀ (x y z : ℝ),
  (∃ (d : ℝ), y = x - d ∧ z = x - 2*d) →  -- arithmetic sequence condition
  x ≥ y ∧ y ≥ z ∧ z ≥ 0 →                -- ordering condition
  (∀ r : ℝ, y*r^2 + z*r + y = 0 ↔ r = 4) →  -- unique root condition
  ∀ r : ℝ, y*r^2 + z*r + y = 0 → r = 4 :=
by sorry

end arithmetic_sequence_quadratic_root_l1287_128751


namespace amelia_win_probability_l1287_128750

/-- Probability of Amelia's coin landing heads -/
def p_amelia : ℚ := 2/7

/-- Probability of Blaine's coin landing heads -/
def p_blaine : ℚ := 1/3

/-- Probability of Amelia getting two heads in one turn -/
def p_amelia_win_turn : ℚ := p_amelia ^ 2

/-- Probability of Blaine getting two heads in one turn -/
def p_blaine_win_turn : ℚ := p_blaine ^ 2

/-- Probability of neither player winning in one round -/
def p_no_win_round : ℚ := (1 - p_amelia_win_turn) * (1 - p_blaine_win_turn)

/-- The probability that Amelia wins the game -/
theorem amelia_win_probability : 
  (p_amelia_win_turn / (1 - p_no_win_round)) = 4/9 :=
sorry

end amelia_win_probability_l1287_128750


namespace constant_term_product_l1287_128794

-- Define polynomials p, q, and r
variable (p q r : ℝ[X])

-- Define the relationship between r, p, and q
variable (h_prod : r = p * q)

-- Define the constant terms of p and r
variable (h_p_const : p.coeff 0 = 5)
variable (h_r_const : r.coeff 0 = -10)

-- Theorem statement
theorem constant_term_product :
  q.eval 0 = -2 := by sorry

end constant_term_product_l1287_128794


namespace lcm_of_numbers_with_given_hcf_and_product_l1287_128785

theorem lcm_of_numbers_with_given_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 16 → 
  a * b = 2560 → 
  Nat.lcm a b = 160 := by
sorry

end lcm_of_numbers_with_given_hcf_and_product_l1287_128785


namespace regular_pentagon_diagonal_inequality_l1287_128742

/-- A regular pentagon -/
structure RegularPentagon where
  side_length : ℝ
  diagonal_short : ℝ
  diagonal_long : ℝ
  side_length_pos : 0 < side_length

/-- The longer diagonal is greater than the shorter diagonal in a regular pentagon -/
theorem regular_pentagon_diagonal_inequality (p : RegularPentagon) : 
  p.diagonal_long > p.diagonal_short := by
  sorry

end regular_pentagon_diagonal_inequality_l1287_128742


namespace ice_cream_cost_is_7_l1287_128772

/-- The cost of one portion of ice cream in kopecks -/
def ice_cream_cost : ℕ := 7

/-- Fedya's money in kopecks -/
def fedya_money : ℕ := ice_cream_cost - 7

/-- Masha's money in kopecks -/
def masha_money : ℕ := ice_cream_cost - 1

theorem ice_cream_cost_is_7 :
  (fedya_money + masha_money < ice_cream_cost) ∧
  (fedya_money = ice_cream_cost - 7) ∧
  (masha_money = ice_cream_cost - 1) →
  ice_cream_cost = 7 := by
  sorry

end ice_cream_cost_is_7_l1287_128772


namespace total_match_sticks_l1287_128766

/-- Given the number of boxes ordered by Farrah -/
def num_boxes : ℕ := 4

/-- The number of matchboxes in each box -/
def matchboxes_per_box : ℕ := 20

/-- The number of sticks in each matchbox -/
def sticks_per_matchbox : ℕ := 300

/-- Theorem stating the total number of match sticks ordered by Farrah -/
theorem total_match_sticks : 
  num_boxes * matchboxes_per_box * sticks_per_matchbox = 24000 := by
  sorry

end total_match_sticks_l1287_128766


namespace arithmetic_sequence_sum_l1287_128799

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (a 1 + a n) * n / 2

/-- Theorem: For an arithmetic sequence where S₁₅ - S₁₀ = 1, S₂₅ = 5 -/
theorem arithmetic_sequence_sum
  (seq : ArithmeticSequence)
  (h : seq.S 15 - seq.S 10 = 1) :
  seq.S 25 = 5 := by
  sorry

end arithmetic_sequence_sum_l1287_128799


namespace part_one_part_two_l1287_128791

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  -- Add necessary conditions
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  -- Add the given condition
  given_condition : 2 * Real.cos C + 2 * Real.cos A = 5 * b / 2

theorem part_one (t : Triangle) : 2 * (t.a + t.c) = 3 * t.b := by sorry

theorem part_two (t : Triangle) (h1 : Real.cos t.B = 1/4) (h2 : t.S = Real.sqrt 15) : t.b = 4 := by sorry

end part_one_part_two_l1287_128791


namespace percentage_problem_l1287_128736

theorem percentage_problem : ∃ P : ℝ, P * 600 = 50 / 100 * 900 ∧ P = 75 / 100 := by
  sorry

end percentage_problem_l1287_128736


namespace sqrt_sum_equality_l1287_128710

theorem sqrt_sum_equality (a b m n : ℚ) : 
  Real.sqrt a + Real.sqrt b = 1 →
  Real.sqrt a = m + (a - b) / 2 →
  Real.sqrt b = n - (a - b) / 2 →
  m^2 + n^2 = 1/2 := by
  sorry

end sqrt_sum_equality_l1287_128710


namespace no_representation_2023_l1287_128735

theorem no_representation_2023 : ¬∃ (a b c : ℕ), 
  (a + b + c = 2023) ∧ 
  (∃ k : ℕ, a = k * (b + c)) ∧ 
  (∃ m : ℕ, b + c = m * (b - c + 1)) := by
sorry

end no_representation_2023_l1287_128735


namespace line_parameterization_l1287_128777

/-- Given a line y = (3/4)x - 15 parameterized by (x,y) = (f(t), 20t - 10),
    prove that f(t) = (80/3)t + (20/3) -/
theorem line_parameterization (f : ℝ → ℝ) :
  (∀ x y, y = (3/4) * x - 15 ↔ ∃ t, x = f t ∧ y = 20 * t - 10) →
  f = λ t => (80/3) * t + 20/3 := by
  sorry

end line_parameterization_l1287_128777


namespace perpendicular_iff_m_eq_neg_two_thirds_l1287_128721

/-- Two lines in the cartesian plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line in the problem -/
def line1 (m : ℝ) : Line :=
  { a := 1, b := m + 1, c := m - 2 }

/-- The second line in the problem -/
def line2 (m : ℝ) : Line :=
  { a := m, b := 2, c := 8 }

/-- The theorem to be proved -/
theorem perpendicular_iff_m_eq_neg_two_thirds :
  ∀ m : ℝ, perpendicular (line1 m) (line2 m) ↔ m = -2/3 := by
  sorry

end perpendicular_iff_m_eq_neg_two_thirds_l1287_128721


namespace composite_19_8n_17_l1287_128758

theorem composite_19_8n_17 (n : ℕ) (hn : n > 0) : 
  ∃ (k : ℕ), k > 1 ∧ k < 19 * 8^n + 17 ∧ (19 * 8^n + 17) % k = 0 := by
  sorry

end composite_19_8n_17_l1287_128758


namespace consumption_increase_l1287_128746

theorem consumption_increase (T C : ℝ) (h1 : T > 0) (h2 : C > 0) : 
  let new_tax := 0.7 * T
  let new_revenue := 0.77 * (T * C)
  let new_consumption := C * (1 + 10/100)
  new_tax * new_consumption = new_revenue :=
sorry

end consumption_increase_l1287_128746


namespace odd_divides_power_two_minus_one_l1287_128749

theorem odd_divides_power_two_minus_one (a : ℕ) (h : Odd a) :
  ∃ b : ℕ, a ∣ (2^b - 1) := by sorry

end odd_divides_power_two_minus_one_l1287_128749


namespace min_value_of_f_l1287_128798

def f (x : ℝ) : ℝ := x^2 + 14*x + 24

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = -25 := by
  sorry

end min_value_of_f_l1287_128798


namespace baba_yaga_students_l1287_128778

theorem baba_yaga_students (B G : ℕ) : 
  B + G = 33 →
  (2 * G + 2 * B) / 3 = 22 :=
by
  sorry

#check baba_yaga_students

end baba_yaga_students_l1287_128778


namespace even_expression_l1287_128782

theorem even_expression (n : ℕ) (h : n = 101) : Even (2 * n - 2) := by
  sorry

end even_expression_l1287_128782


namespace room_width_is_two_l1287_128784

-- Define the room's properties
def room_area : ℝ := 10
def room_length : ℝ := 5

-- Theorem statement
theorem room_width_is_two : 
  ∃ (width : ℝ), room_area = room_length * width ∧ width = 2 := by
  sorry

end room_width_is_two_l1287_128784


namespace quadratic_inequality_l1287_128712

theorem quadratic_inequality (x : ℝ) : x^2 - 42*x + 400 ≤ 10 ↔ 13 ≤ x ∧ x ≤ 30 := by
  sorry

end quadratic_inequality_l1287_128712


namespace f_explicit_formula_l1287_128776

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_explicit_formula 
  (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_period : has_period f 2)
  (h_known : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| :=
sorry

end f_explicit_formula_l1287_128776


namespace rectangle_area_with_three_squares_l1287_128726

/-- The area of a rectangle containing three non-overlapping squares -/
theorem rectangle_area_with_three_squares 
  (small_square_area : ℝ) 
  (large_square_side_multiplier : ℝ) : 
  small_square_area = 1 →
  large_square_side_multiplier = 3 →
  2 * small_square_area + (large_square_side_multiplier^2 * small_square_area) = 11 :=
by
  sorry

#check rectangle_area_with_three_squares

end rectangle_area_with_three_squares_l1287_128726


namespace gcd_2028_2295_l1287_128718

theorem gcd_2028_2295 : Nat.gcd 2028 2295 = 1 := by
  sorry

end gcd_2028_2295_l1287_128718


namespace range_of_4x_plus_2y_l1287_128773

theorem range_of_4x_plus_2y (x y : ℝ) 
  (h1 : 1 ≤ x + y) (h2 : x + y ≤ 3) 
  (h3 : -1 ≤ x - y) (h4 : x - y ≤ 1) : 
  2 ≤ 4*x + 2*y ∧ 4*x + 2*y ≤ 10 := by
  sorry

end range_of_4x_plus_2y_l1287_128773


namespace smallest_largest_five_digit_reverse_multiple_of_four_l1287_128738

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a five-digit number -/
def isFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem smallest_largest_five_digit_reverse_multiple_of_four :
  ∀ n : ℕ, isFiveDigit n → (reverseDigits n % 4 = 0) →
    21001 ≤ n ∧ n ≤ 88999 ∧
    (∀ m : ℕ, isFiveDigit m → (reverseDigits m % 4 = 0) →
      (m < 21001 ∨ 88999 < m) → False) :=
sorry

end smallest_largest_five_digit_reverse_multiple_of_four_l1287_128738


namespace candy_chocolate_cost_difference_l1287_128707

/-- The cost difference between a candy bar and a chocolate -/
def cost_difference (candy_bar_cost chocolate_cost : ℕ) : ℕ :=
  candy_bar_cost - chocolate_cost

/-- Theorem: The cost difference between a $7 candy bar and a $3 chocolate is $4 -/
theorem candy_chocolate_cost_difference :
  cost_difference 7 3 = 4 := by
  sorry

end candy_chocolate_cost_difference_l1287_128707


namespace sufficient_not_necessary_condition_l1287_128700

theorem sufficient_not_necessary_condition (a b c d : ℝ) :
  (∀ a b c d : ℝ, a > b ∧ c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) := by
  sorry

end sufficient_not_necessary_condition_l1287_128700


namespace greater_number_proof_l1287_128739

theorem greater_number_proof (x y : ℝ) (h1 : 4 * y = 5 * x) (h2 : x + y = 26) : 
  y = 130 / 9 := by
  sorry

end greater_number_proof_l1287_128739


namespace rectangle_x_value_l1287_128702

/-- A rectangle in a 2D plane --/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The area of a rectangle --/
def rectangleArea (r : Rectangle) : ℝ :=
  (r.C.1 - r.A.1) * (r.B.2 - r.A.2)

theorem rectangle_x_value 
  (x : ℝ) 
  (h_pos : x > 0) 
  (rect : Rectangle) 
  (h_vertices : rect = { 
    A := (0, 0), 
    B := (0, 4), 
    C := (x, 4), 
    D := (x, 0) 
  }) 
  (h_area : rectangleArea rect = 28) : 
  x = 7 := by
sorry

end rectangle_x_value_l1287_128702


namespace seventh_triangular_is_28_l1287_128771

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The seventh triangular number is 28 -/
theorem seventh_triangular_is_28 : triangular 7 = 28 := by sorry

end seventh_triangular_is_28_l1287_128771


namespace fraction_zero_implies_x_negative_one_l1287_128783

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (x ≠ 1) →  -- ensure fraction is defined
  ((|x| - 1) / (x - 1) = 0) →
  x = -1 := by
sorry

end fraction_zero_implies_x_negative_one_l1287_128783


namespace de_moivre_and_rationality_l1287_128725

/-- De Moivre's formula and its implication on rationality of trigonometric functions -/
theorem de_moivre_and_rationality (θ : ℝ) (n : ℕ) :
  (Complex.exp (θ * Complex.I))^n = Complex.exp (n * θ * Complex.I) ∧
  (∀ (a b : ℚ), Complex.exp (θ * Complex.I) = ↑a + ↑b * Complex.I →
    ∃ (c d : ℚ), Complex.exp (n * θ * Complex.I) = ↑c + ↑d * Complex.I) := by
  sorry

end de_moivre_and_rationality_l1287_128725


namespace unique_pair_satisfying_inequality_l1287_128768

theorem unique_pair_satisfying_inequality :
  ∃! (a b : ℝ), ∀ x : ℝ, x ∈ Set.Icc 0 1 →
    |Real.sqrt (1 - x^2) - a*x - b| ≤ (Real.sqrt 2 - 1) / 2 ∧ a = 0 ∧ b = 0 := by
  sorry

end unique_pair_satisfying_inequality_l1287_128768


namespace limit_proof_l1287_128759

theorem limit_proof : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ,
  0 < |x - 1/3| ∧ |x - 1/3| < δ →
  |(15*x^2 - 2*x - 1) / (x - 1/3) - 8| < ε := by
  sorry

end limit_proof_l1287_128759


namespace no_adjacent_women_correct_a_not_first_b_not_last_correct_fixed_sequence_correct_a_left_of_b_correct_l1287_128774

def num_men : Nat := 4
def num_women : Nat := 3
def total_people : Nat := num_men + num_women

-- Function to calculate the number of arrangements where no two women are adjacent
def arrangements_no_adjacent_women : Nat :=
  Nat.factorial num_men * Nat.descFactorial (num_men + 1) num_women

-- Function to calculate the number of arrangements where Man A is not first and Man B is not last
def arrangements_a_not_first_b_not_last : Nat :=
  Nat.factorial total_people - 2 * Nat.factorial (total_people - 1) + Nat.factorial (total_people - 2)

-- Function to calculate the number of arrangements where Men A, B, and C are in a fixed sequence
def arrangements_fixed_sequence : Nat :=
  Nat.factorial total_people / Nat.factorial 3

-- Function to calculate the number of arrangements where Man A is to the left of Man B
def arrangements_a_left_of_b : Nat :=
  Nat.factorial total_people / 2

-- Theorems to prove
theorem no_adjacent_women_correct :
  arrangements_no_adjacent_women = 1440 := by sorry

theorem a_not_first_b_not_last_correct :
  arrangements_a_not_first_b_not_last = 3720 := by sorry

theorem fixed_sequence_correct :
  arrangements_fixed_sequence = 840 := by sorry

theorem a_left_of_b_correct :
  arrangements_a_left_of_b = 2520 := by sorry

end no_adjacent_women_correct_a_not_first_b_not_last_correct_fixed_sequence_correct_a_left_of_b_correct_l1287_128774


namespace correct_locus_definition_l1287_128761

-- Define a type for points in a geometric space
variable {Point : Type}

-- Define a predicate for the locus condition
variable (locus_condition : Point → Prop)

-- Define the locus as a set of points
def locus (locus_condition : Point → Prop) : Set Point :=
  {p : Point | locus_condition p}

-- State the theorem
theorem correct_locus_definition (p : Point) :
  p ∈ locus locus_condition ↔ locus_condition p :=
sorry

end correct_locus_definition_l1287_128761


namespace triangle_angle_B_value_l1287_128723

theorem triangle_angle_B_value 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h3 : A + B + C = π) 
  (h4 : (c - b) / (c - a) = Real.sin A / (Real.sin C + Real.sin B)) : 
  B = π / 3 := by
sorry

end triangle_angle_B_value_l1287_128723


namespace number_exceeds_fraction_l1287_128720

theorem number_exceeds_fraction (N : ℚ) (F : ℚ) : 
  N = 56 → N = F + 35 → F = 3/8 := by
  sorry

end number_exceeds_fraction_l1287_128720


namespace water_level_change_notation_l1287_128756

/-- Represents the change in water level -/
def WaterLevelChange : ℤ → ℝ
  | 2 => 2
  | -2 => -2
  | _ => 0  -- Default case, not relevant for this problem

/-- The water level rise notation -/
def WaterLevelRiseNotation : ℝ := 2

/-- The water level drop notation -/
def WaterLevelDropNotation : ℝ := -2

theorem water_level_change_notation :
  WaterLevelChange 2 = WaterLevelRiseNotation ∧
  WaterLevelChange (-2) = WaterLevelDropNotation :=
by sorry

end water_level_change_notation_l1287_128756


namespace ship_length_l1287_128775

/-- The length of a ship given its speed and time to pass a fixed point -/
theorem ship_length (speed : ℝ) (time : ℝ) (h1 : speed = 18) (h2 : time = 20) :
  speed * time * (1000 / 3600) = 100 := by
  sorry

#check ship_length

end ship_length_l1287_128775


namespace compound_molecular_weight_l1287_128747

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (n_Al n_Cl n_O : ℕ) (w_Al w_Cl w_O : ℝ) : ℝ :=
  n_Al * w_Al + n_Cl * w_Cl + n_O * w_O

/-- The molecular weight of a compound with 2 Al, 6 Cl, and 3 O atoms is 314.66 g/mol -/
theorem compound_molecular_weight :
  molecular_weight 2 6 3 26.98 35.45 16.00 = 314.66 := by sorry

end compound_molecular_weight_l1287_128747


namespace smallest_positive_solution_l1287_128733

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x - 2 ∧
  ∀ (y : ℝ), y > 0 → Real.sqrt (3 * y) = 5 * y - 2 → x ≤ y ∧
  x = 4 / 25 :=
by sorry

end smallest_positive_solution_l1287_128733


namespace mandy_book_ratio_l1287_128793

/-- Represents Mandy's book reading progression --/
structure BookReading where
  initial_length : ℕ
  initial_age : ℕ
  current_length : ℕ

/-- Calculates the ratio of book length at twice the starting age to initial book length --/
def length_ratio (r : BookReading) : ℚ :=
  let twice_age_length := r.initial_length * (r.current_length / (4 * 3 * r.initial_length))
  twice_age_length / r.initial_length

/-- Theorem stating the ratio of book length at twice Mandy's starting age to her initial book length --/
theorem mandy_book_ratio : 
  ∀ (r : BookReading), 
  r.initial_length = 8 ∧ 
  r.initial_age = 6 ∧ 
  r.current_length = 480 → 
  length_ratio r = 5 := by
  sorry

#eval length_ratio { initial_length := 8, initial_age := 6, current_length := 480 }

end mandy_book_ratio_l1287_128793


namespace range_of_a_l1287_128789

-- Define set A
def A : Set ℝ := {x | 1 < |x - 2| ∧ |x - 2| < 2}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 1) * x + a < 0}

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ A ∩ B a) ↔ a ∈ Set.Iio 1 ∪ Set.Ioi 3 :=
by sorry

end range_of_a_l1287_128789


namespace tan_sum_product_equals_one_l1287_128717

theorem tan_sum_product_equals_one :
  Real.tan (22 * π / 180) + Real.tan (23 * π / 180) + Real.tan (22 * π / 180) * Real.tan (23 * π / 180) = 1 := by
  sorry

end tan_sum_product_equals_one_l1287_128717


namespace min_value_expression_l1287_128755

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (p + q + r) * (1 / (p + q) + 1 / (p + r) + 1 / (q + r) + 1 / (p + q + r)) ≥ 5 ∧
  (∃ t : ℝ, t > 0 ∧ (t + t + t) * (1 / (t + t) + 1 / (t + t) + 1 / (t + t) + 1 / (t + t + t)) = 5) :=
by sorry

end min_value_expression_l1287_128755


namespace dividend_calculation_l1287_128764

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 36)
  (h2 : quotient = 21)
  (h3 : remainder = 4) :
  divisor * quotient + remainder = 760 := by
sorry

end dividend_calculation_l1287_128764


namespace consecutive_integers_divisibility_l1287_128732

theorem consecutive_integers_divisibility (a₁ a₂ a₃ : ℕ) :
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₂ = a₁ + 1 ∧ a₃ = a₂ + 1 →
  a₂^3 ∣ (a₁ * a₂ * a₃ + a₂) :=
by
  sorry

end consecutive_integers_divisibility_l1287_128732


namespace andrea_reach_time_l1287_128711

/-- The time it takes Andrea to reach Lauren's stop location -/
def time_to_reach (initial_distance : ℝ) (speed_ratio : ℝ) (distance_decrease_rate : ℝ) (lauren_stop_time : ℝ) : ℝ :=
  sorry

/-- Theorem stating the time it takes Andrea to reach Lauren's stop location -/
theorem andrea_reach_time :
  let initial_distance : ℝ := 30
  let speed_ratio : ℝ := 2
  let distance_decrease_rate : ℝ := 90
  let lauren_stop_time : ℝ := 1/6 -- 10 minutes in hours
  time_to_reach initial_distance speed_ratio distance_decrease_rate lauren_stop_time = 25/60 := by
  sorry

end andrea_reach_time_l1287_128711


namespace first_month_sale_l1287_128767

theorem first_month_sale (sales_2 sales_3 sales_4 sales_5 sales_6 : ℕ)
  (h1 : sales_2 = 6927)
  (h2 : sales_3 = 6855)
  (h3 : sales_4 = 7230)
  (h4 : sales_5 = 6562)
  (h5 : sales_6 = 4791)
  (desired_average : ℕ)
  (h6 : desired_average = 6500)
  (num_months : ℕ)
  (h7 : num_months = 6) :
  ∃ (sales_1 : ℕ), sales_1 = 6635 ∧
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = desired_average :=
by
  sorry

end first_month_sale_l1287_128767


namespace shape_to_square_transformation_exists_l1287_128792

/-- A shape on a graph paper --/
structure GraphShape where
  -- Add necessary fields to represent the shape

/-- A triangle on a graph paper --/
structure Triangle where
  -- Add necessary fields to represent a triangle

/-- A square on a graph paper --/
structure Square where
  -- Add necessary fields to represent a square

/-- Function to divide a shape into triangles --/
def divideIntoTriangles (shape : GraphShape) : List Triangle :=
  sorry

/-- Function to check if a list of triangles can form a square --/
def canFormSquare (triangles : List Triangle) : Bool :=
  sorry

/-- Theorem stating that there exists a shape that can be divided into 5 triangles
    which can be reassembled to form a square --/
theorem shape_to_square_transformation_exists :
  ∃ (shape : GraphShape),
    let triangles := divideIntoTriangles shape
    triangles.length = 5 ∧ canFormSquare triangles :=
by
  sorry

end shape_to_square_transformation_exists_l1287_128792


namespace point_in_fourth_quadrant_l1287_128729

theorem point_in_fourth_quadrant (P : ℝ × ℝ) :
  P.1 = Real.tan (2011 * π / 180) →
  P.2 = Real.cos (2011 * π / 180) →
  Real.tan (2011 * π / 180) > 0 →
  Real.cos (2011 * π / 180) < 0 →
  P.1 > 0 ∧ P.2 < 0 := by
sorry

end point_in_fourth_quadrant_l1287_128729


namespace sphere_surface_area_l1287_128762

theorem sphere_surface_area (d : ℝ) (h : d = 12) : 
  4 * Real.pi * (d / 2)^2 = 144 * Real.pi := by
  sorry

end sphere_surface_area_l1287_128762


namespace market_price_calculation_l1287_128741

/-- Given an initial sales tax rate, a reduced sales tax rate, and the difference in tax amount,
    proves that the market price of an article is 6600. -/
theorem market_price_calculation (initial_rate reduced_rate : ℚ) (tax_difference : ℝ) :
  initial_rate = 35 / 1000 →
  reduced_rate = 100 / 3000 →
  tax_difference = 10.999999999999991 →
  ∃ (price : ℕ), price = 6600 ∧ (initial_rate - reduced_rate) * price = tax_difference :=
sorry

end market_price_calculation_l1287_128741


namespace adult_ticket_price_l1287_128724

theorem adult_ticket_price 
  (total_amount : ℕ)
  (child_price : ℕ)
  (total_tickets : ℕ)
  (child_tickets : ℕ)
  (h1 : total_amount = 104)
  (h2 : child_price = 4)
  (h3 : total_tickets = 21)
  (h4 : child_tickets = 11) :
  ∃ (adult_price : ℕ), 
    adult_price * (total_tickets - child_tickets) + child_price * child_tickets = total_amount ∧ 
    adult_price = 6 :=
by sorry

end adult_ticket_price_l1287_128724
