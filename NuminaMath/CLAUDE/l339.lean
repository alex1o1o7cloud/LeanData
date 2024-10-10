import Mathlib

namespace perfect_square_trinomial_condition_l339_33966

/-- A quadratic expression ax^2 + bx + c is a perfect square trinomial if there exists a real number k such that ax^2 + bx + c = (kx + r)^2 for some real r. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ k r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (k * x + r)^2

/-- If 4x^2 + mx + 9 is a perfect square trinomial, then m = 12 or m = -12. -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  IsPerfectSquareTrinomial 4 m 9 → m = 12 ∨ m = -12 := by
  sorry

end perfect_square_trinomial_condition_l339_33966


namespace circuit_current_l339_33945

/-- Given a voltage V and impedance Z as complex numbers,
    prove that the current I = V / Z equals the expected value. -/
theorem circuit_current (V Z : ℂ) (hV : V = 2 + 3*I) (hZ : Z = 4 - 2*I) :
  V / Z = (1 / 10 : ℂ) + (4 / 5 : ℂ) * I :=
by sorry

end circuit_current_l339_33945


namespace expression_evaluation_l339_33982

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/2
  2 * x^2 + (-x^2 - 2*x*y + 2*y^2) - 3*(x^2 - x*y + 2*y^2) = -10 :=
by sorry

end expression_evaluation_l339_33982


namespace fred_age_difference_l339_33933

theorem fred_age_difference (jim fred sam : ℕ) : 
  jim = 2 * fred →
  jim = 46 →
  jim - 6 = 5 * (sam - 6) →
  fred - sam = 9 := by sorry

end fred_age_difference_l339_33933


namespace sum_four_consecutive_composite_sum_three_consecutive_composite_l339_33930

-- Define the sum of four consecutive positive integers
def sum_four_consecutive (n : ℕ) : ℕ := n + (n + 1) + (n + 2) + (n + 3)

-- Define the sum of three consecutive positive integers
def sum_three_consecutive (n : ℕ) : ℕ := n + (n + 1) + (n + 2)

-- Theorem for four consecutive positive integers
theorem sum_four_consecutive_composite (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ sum_four_consecutive n = a * b :=
sorry

-- Theorem for three consecutive positive integers
theorem sum_three_consecutive_composite (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ sum_three_consecutive n = a * b :=
sorry

end sum_four_consecutive_composite_sum_three_consecutive_composite_l339_33930


namespace no_real_solutions_l339_33978

theorem no_real_solutions : ¬∃ (x y : ℝ), x^2 + 3*y^2 - 4*x - 6*y + 9 = 0 := by
  sorry

end no_real_solutions_l339_33978


namespace greatest_ABDBA_div_by_11_l339_33919

/-- Represents a five-digit number in the form AB,DBA -/
structure ABDBA where
  a : Nat
  b : Nat
  d : Nat
  h1 : a < 10
  h2 : b < 10
  h3 : d < 10
  h4 : a ≠ b
  h5 : a ≠ d
  h6 : b ≠ d

/-- Converts ABDBA to its numerical value -/
def ABDBA.toNat (n : ABDBA) : Nat :=
  n.a * 10000 + n.b * 1000 + n.d * 100 + n.b * 10 + n.a

/-- Theorem stating the greatest ABDBA number divisible by 11 -/
theorem greatest_ABDBA_div_by_11 :
  ∀ n : ABDBA, n.toNat ≤ 96569 ∧ n.toNat % 11 = 0 →
  ∃ m : ABDBA, m.toNat = 96569 ∧ m.toNat % 11 = 0 :=
sorry

end greatest_ABDBA_div_by_11_l339_33919


namespace negation_equivalence_l339_33969

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 3*x + 2 ≥ 0) := by
  sorry

end negation_equivalence_l339_33969


namespace five_digit_division_l339_33989

/-- A five-digit number with the first digit not zero -/
def FiveDigitNumber (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A four-digit number formed by removing the middle digit of a five-digit number -/
def FourDigitNumber (m n : ℕ) : Prop :=
  FiveDigitNumber n ∧ 
  ∃ (x y z u v : ℕ), 
    n = x * 10000 + y * 1000 + z * 100 + u * 10 + v ∧
    m = x * 1000 + y * 100 + u * 10 + v ∧
    0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9 ∧ 0 ≤ v ∧ v ≤ 9 ∧
    x ≠ 0

theorem five_digit_division (n m : ℕ) : 
  FiveDigitNumber n → FourDigitNumber m n → (∃ k : ℕ, n = k * m) ↔ 
  ∃ (x y : ℕ), n = (10 * x + y) * 1000 ∧ 10 ≤ 10 * x + y ∧ 10 * x + y ≤ 99 :=
sorry

end five_digit_division_l339_33989


namespace valid_circular_arrangement_exists_l339_33996

/-- A type representing a circular arrangement of 9 numbers -/
def CircularArrangement := Fin 9 → Fin 9

/-- Check if two numbers in the arrangement are adjacent -/
def are_adjacent (arr : CircularArrangement) (i j : Fin 9) : Prop :=
  (j = i + 1) ∨ (i = 8 ∧ j = 0)

/-- Check if a number is valid in the arrangement (1 to 9) -/
def is_valid_number (n : Fin 9) : Prop := n.val + 1 ∈ Finset.range 10

/-- Check if the sum of two numbers is not divisible by 3, 5, or 7 -/
def sum_not_divisible (a b : Fin 9) : Prop :=
  ¬(((a.val + 1) + (b.val + 1)) % 3 = 0) ∧
  ¬(((a.val + 1) + (b.val + 1)) % 5 = 0) ∧
  ¬(((a.val + 1) + (b.val + 1)) % 7 = 0)

/-- The main theorem stating that a valid circular arrangement exists -/
theorem valid_circular_arrangement_exists : ∃ (arr : CircularArrangement),
  (∀ i : Fin 9, is_valid_number (arr i)) ∧
  (∀ i j : Fin 9, are_adjacent arr i j → sum_not_divisible (arr i) (arr j)) ∧
  Function.Injective arr :=
sorry

end valid_circular_arrangement_exists_l339_33996


namespace average_of_three_angles_l339_33994

/-- Given that the average of α and β is 105°, prove that the average of α, β, and γ is 80°. -/
theorem average_of_three_angles (α β γ : ℝ) :
  (α + β) / 2 = 105 → (α + β + γ) / 3 = 80 :=
by sorry

end average_of_three_angles_l339_33994


namespace square_root_fraction_equality_l339_33960

theorem square_root_fraction_equality : 
  Real.sqrt (8^2 + 15^2) / Real.sqrt (25 + 16) = (17 * Real.sqrt 41) / 41 := by
  sorry

end square_root_fraction_equality_l339_33960


namespace power_function_increasing_m_eq_3_l339_33948

/-- A function f(x) = cx^p where c and p are constants and x > 0 -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ c p : ℝ, ∀ x > 0, f x = c * x^p

/-- A function f is increasing on (0, +∞) if for all x, y > 0, x < y implies f(x) < f(y) -/
def isIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

/-- The main theorem stating that m = 3 is the only value satisfying the conditions -/
theorem power_function_increasing_m_eq_3 :
  ∃! m : ℝ, 
    isPowerFunction (fun x => (m^2 - m - 5) * x^(m-1)) ∧ 
    isIncreasing (fun x => (m^2 - m - 5) * x^(m-1)) :=
sorry

end power_function_increasing_m_eq_3_l339_33948


namespace extremum_at_one_implies_a_eq_neg_two_l339_33963

/-- Given a cubic function f(x) = x^3 + ax^2 + x + b with an extremum at x = 1, 
    prove that a = -2. -/
theorem extremum_at_one_implies_a_eq_neg_two (a b : ℝ) :
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + x + b
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) →
  a = -2 := by
sorry

end extremum_at_one_implies_a_eq_neg_two_l339_33963


namespace min_value_expression_l339_33985

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2)) + (y^2 / (x - 2)) ≥ 18 :=
by sorry

end min_value_expression_l339_33985


namespace identity_condition_l339_33901

/-- 
Proves that the equation (3x-a)(2x+5)-x = 6x^2+2(5x-b) is an identity 
for all x if and only if a = 2 and b = 5.
-/
theorem identity_condition (a b : ℝ) : 
  (∀ x : ℝ, (3*x - a)*(2*x + 5) - x = 6*x^2 + 2*(5*x - b)) ↔ (a = 2 ∧ b = 5) := by
  sorry

end identity_condition_l339_33901


namespace even_count_pascal_triangle_l339_33941

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Check if a natural number is even -/
def isEven (n : ℕ) : Bool := sorry

/-- Count even binomial coefficients in a single row of Pascal's Triangle -/
def countEvenInRow (row : ℕ) : ℕ := sorry

/-- Count even binomial coefficients in the first n rows of Pascal's Triangle -/
def countEvenInTriangle (n : ℕ) : ℕ := sorry

/-- The number of even integers in the top 15 rows of Pascal's Triangle is 84 -/
theorem even_count_pascal_triangle : countEvenInTriangle 15 = 84 := by sorry

end even_count_pascal_triangle_l339_33941


namespace cyclist_distance_l339_33970

/-- Proves that a cyclist traveling at 18 km/hr for 2 minutes and 30 seconds covers a distance of 750 meters. -/
theorem cyclist_distance (speed : ℝ) (time_min : ℝ) (time_sec : ℝ) (distance : ℝ) :
  speed = 18 →
  time_min = 2 →
  time_sec = 30 →
  distance = speed * (time_min / 60 + time_sec / 3600) * 1000 →
  distance = 750 := by
sorry


end cyclist_distance_l339_33970


namespace sequence_is_arithmetic_l339_33984

/-- Given a sequence {a_n} where the sum of its first n terms is S_n = 2n^2 - 3n,
    prove that {a_n} is an arithmetic sequence. -/
theorem sequence_is_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = 2 * n^2 - 3 * n) :
    ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d :=
  sorry

end sequence_is_arithmetic_l339_33984


namespace problem_solution_l339_33921

theorem problem_solution (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = 1) :
  x + x^4 / y^3 + y^4 / x^3 + y = 228498 := by
  sorry

end problem_solution_l339_33921


namespace angle_symmetry_l339_33926

/-- Two angles are symmetric about the y-axis if their sum is congruent to 180° modulo 360° -/
def symmetric_about_y_axis (α β : Real) : Prop :=
  ∃ k : ℤ, α + β = k * 360 + 180

theorem angle_symmetry (α β : Real) :
  symmetric_about_y_axis α β →
  ∃ k : ℤ, α + β = k * 360 + 180 := by
  sorry

end angle_symmetry_l339_33926


namespace locus_of_touching_parabolas_l339_33927

/-- Given a parabola y = x^2 with directrix y = -1/4, this theorem describes
    the locus of points P(u, v) for which there exists a line v parallel to
    the directrix and at a distance s from it, such that the parabola with
    directrix v and focus P touches the given parabola. -/
theorem locus_of_touching_parabolas (s : ℝ) (u v : ℝ) :
  (2 * s ≠ 1 → (v = (1 / (1 - 2 * s)) * u^2 + s / 2 ∨
                v = (1 / (1 + 2 * s)) * u^2 - s / 2)) ∧
  (2 * s = 1 → v = u^2 / 2 - 1 / 4) :=
by sorry

end locus_of_touching_parabolas_l339_33927


namespace acme_vowel_soup_sequences_l339_33947

/-- The number of distinct elements in the set -/
def num_elements : ℕ := 5

/-- The length of the sequences to be formed -/
def sequence_length : ℕ := 7

/-- The maximum number of times each element can be used -/
def max_repetitions : ℕ := 4

/-- The number of possible sequences -/
def num_sequences : ℕ := num_elements ^ sequence_length

theorem acme_vowel_soup_sequences :
  num_sequences = 78125 :=
sorry

end acme_vowel_soup_sequences_l339_33947


namespace expression_evaluation_l339_33903

theorem expression_evaluation :
  let a : ℚ := -2
  let b : ℚ := 1/5
  2 * (a^2 * b - 2 * a * b) - 3 * (a^2 * b - 3 * a * b) + a^2 * b = -2 := by
  sorry

end expression_evaluation_l339_33903


namespace union_complement_equality_l339_33920

open Set

def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {1, 4}
def N : Finset ℕ := {2, 5}

theorem union_complement_equality : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_equality_l339_33920


namespace spoiled_fish_fraction_l339_33993

theorem spoiled_fish_fraction (initial_stock sold_fish new_stock final_stock : ℕ) : 
  initial_stock = 200 →
  sold_fish = 50 →
  new_stock = 200 →
  final_stock = 300 →
  (final_stock - new_stock) / (initial_stock - sold_fish) = 2/3 := by
sorry

end spoiled_fish_fraction_l339_33993


namespace sum_of_twos_and_threes_3024_l339_33906

/-- The number of ways to write a positive integer as an unordered sum of 2s and 3s -/
def sumOfTwosAndThrees (n : ℕ) : ℕ := 
  (n / 3 : ℕ) - (n % 3) / 3 + 1

/-- Theorem stating that there are 337 ways to write 3024 as an unordered sum of 2s and 3s -/
theorem sum_of_twos_and_threes_3024 : sumOfTwosAndThrees 3024 = 337 := by
  sorry

end sum_of_twos_and_threes_3024_l339_33906


namespace sqrt_meaningful_range_l339_33908

theorem sqrt_meaningful_range (a : ℝ) : (∃ x : ℝ, x^2 = a - 2) ↔ a ≥ 2 := by sorry

end sqrt_meaningful_range_l339_33908


namespace intersection_point_l339_33924

/-- A parabola defined by x = -3y^2 - 4y + 7 -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

/-- The line x = m -/
def line (m : ℝ) : ℝ := m

/-- The condition for a single intersection point -/
def single_intersection (m : ℝ) : Prop :=
  ∃! y : ℝ, parabola y = line m

theorem intersection_point (m : ℝ) : single_intersection m ↔ m = 25/3 := by
  sorry

end intersection_point_l339_33924


namespace hyperbola_eccentricity_l339_33925

theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b * c / (a^2 + b^2).sqrt = b) →
  (2 * a ≤ b) →
  let e := c / a
  e > Real.sqrt 5 := by sorry

end hyperbola_eccentricity_l339_33925


namespace rice_mixture_cost_l339_33931

/-- Proves that mixing two varieties of rice in the ratio 1:2.4, where one costs 4.5 per kg 
    and the other costs 8.75 per kg, results in a mixture costing 7.50 per kg. -/
theorem rice_mixture_cost 
  (cost1 : ℝ) (cost2 : ℝ) (mixture_cost : ℝ) 
  (ratio1 : ℝ) (ratio2 : ℝ) :
  cost1 = 4.5 →
  cost2 = 8.75 →
  mixture_cost = 7.50 →
  ratio1 = 1 →
  ratio2 = 2.4 →
  (cost1 * ratio1 + cost2 * ratio2) / (ratio1 + ratio2) = mixture_cost :=
by sorry

end rice_mixture_cost_l339_33931


namespace alloy_ratio_l339_33940

/-- Proves that the ratio of tin to copper in alloy B is 3:5 given the specified conditions -/
theorem alloy_ratio : 
  ∀ (lead_A tin_A tin_B copper_B : ℝ),
  -- Alloy A has 170 kg total
  lead_A + tin_A = 170 →
  -- Alloy A has lead and tin in ratio 1:3
  lead_A * 3 = tin_A →
  -- Alloy B has 250 kg total
  tin_B + copper_B = 250 →
  -- Total tin in new alloy is 221.25 kg
  tin_A + tin_B = 221.25 →
  -- Ratio of tin to copper in alloy B is 3:5
  tin_B * 5 = copper_B * 3 := by
sorry


end alloy_ratio_l339_33940


namespace parabolas_no_intersection_l339_33944

/-- The parabolas y = 3x^2 - 6x + 6 and y = -2x^2 + x + 3 do not intersect in the real plane. -/
theorem parabolas_no_intersection : 
  ∀ x y : ℝ, (y = 3*x^2 - 6*x + 6) → (y = -2*x^2 + x + 3) → False :=
by
  sorry

end parabolas_no_intersection_l339_33944


namespace problem_solution_l339_33992

theorem problem_solution : ∃ x : ℤ, x - (28 - (37 - (15 - 16))) = 55 ∧ x = 65 := by
  sorry

end problem_solution_l339_33992


namespace hyperbola_real_axis_length_l339_33910

-- Define the hyperbola equation
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 4 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop :=
  y = 2 * x

-- Define the real axis length
def real_axis_length (a : ℝ) : ℝ :=
  2 * a

-- Theorem statement
theorem hyperbola_real_axis_length :
  ∃ a : ℝ, (∃ x y : ℝ, hyperbola a x y ∧ asymptote x y) →
  real_axis_length a = 2 :=
sorry

end hyperbola_real_axis_length_l339_33910


namespace intersection_complement_equals_l339_33987

def U : Set ℕ := {x | 0 ≤ x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem intersection_complement_equals : S ∩ (U \ T) = {1, 2, 4} := by sorry

end intersection_complement_equals_l339_33987


namespace intersection_line_slope_l339_33949

/-- Given two lines y = 4 - 3x and y = 2x - 1, and a third line y = ax + 7 that passes through 
    their intersection point, prove that a = -6. -/
theorem intersection_line_slope (a : ℝ) : 
  (∃ x y : ℝ, y = 4 - 3*x ∧ y = 2*x - 1 ∧ y = a*x + 7) → a = -6 := by
  sorry

end intersection_line_slope_l339_33949


namespace circle_tangent_line_segment_l339_33980

/-- Given two circles in a plane with radii r₁ and r₂, centered at O₁ and O₂ respectively,
    touching a line at points M₁ and M₂, and lying on the same side of the line,
    if the ratio of M₁M₂ to O₁O₂ is k, then M₁M₂ can be calculated. -/
theorem circle_tangent_line_segment (r₁ r₂ : ℝ) (k : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 7) (h₃ : k = 2 * Real.sqrt 5 / 5) :
  let M₁M₂ := r₁ - r₂
  M₁M₂ * (Real.sqrt (1 - k^2) / k) = 10 :=
by sorry

end circle_tangent_line_segment_l339_33980


namespace fraction_equality_l339_33965

theorem fraction_equality : ∃! (n m : ℕ) (d : ℚ), n > 0 ∧ m > 0 ∧ (n : ℚ) / m = d ∧ d = 2.5 := by
  sorry

end fraction_equality_l339_33965


namespace square_side_length_l339_33946

/-- Given a rectangle formed by three squares and two other rectangles, 
    prove that the middle square has a side length of 651 -/
theorem square_side_length (s₁ s₂ s₃ : ℕ) : 
  s₁ + s₂ + s₃ = 3322 →
  s₁ - s₂ + s₃ = 2020 →
  s₂ = 651 := by
sorry

end square_side_length_l339_33946


namespace equation_solution_l339_33995

theorem equation_solution :
  ∃ (x : ℝ), x ≠ -1 ∧ x ≠ 1 ∧ (x - 1) / (x + 1) - 3 / (x^2 - 1) = 1 ∧ x = -1/2 :=
by sorry

end equation_solution_l339_33995


namespace tan_theta_value_l339_33932

theorem tan_theta_value (θ : Real) 
  (h1 : 2 * Real.sin θ + Real.cos θ = Real.sqrt 2 / 3)
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.tan θ = -(90 + 5 * Real.sqrt 86) / 168 := by
  sorry

end tan_theta_value_l339_33932


namespace periodic_sequence_characterization_l339_33972

def is_periodic_sequence (x : ℕ → ℝ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ ∀ n, x (n + T) = x n

theorem periodic_sequence_characterization
  (x : ℕ → ℝ)
  (h_pos : ∀ n, x n > 0)
  (h_periodic : is_periodic_sequence x)
  (h_recurrence : ∀ n, x (n + 2) = (1 / 2) * (1 / x (n + 1) + x n)) :
  ∃ a : ℝ, a > 0 ∧ ∀ n, x n = if n % 2 = 0 then a else 1 / a :=
sorry

end periodic_sequence_characterization_l339_33972


namespace hyperbola_equation_l339_33950

/-- The hyperbola equation -/
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x^2 = 4 * Real.sqrt 6 * y

/-- The point A lies on the hyperbola -/
def A_on_hyperbola (a b c m n : ℝ) : Prop := hyperbola a b m n

/-- The point B is on the imaginary axis of the hyperbola -/
def B_on_imaginary_axis (b : ℝ) : Prop := b = Real.sqrt 6

/-- The vector relation between BA and AF -/
def vector_relation (c m n : ℝ) : Prop :=
  m - 0 = 2 * (c - m) ∧ n - Real.sqrt 6 = 2 * (0 - n)

/-- The main theorem -/
theorem hyperbola_equation (a b c m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : hyperbola a b m n)
  (h2 : parabola m n)
  (h3 : A_on_hyperbola a b c m n)
  (h4 : B_on_imaginary_axis b)
  (h5 : vector_relation c m n) :
  a = 2 ∧ b = Real.sqrt 6 := by sorry

end hyperbola_equation_l339_33950


namespace loan_principal_calculation_l339_33997

/-- Calculates the principal amount of a loan given the interest rate, time, and total interest. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  interest / (rate * time)

/-- Theorem stating that for a loan with 12% annual simple interest rate,
    where the interest after 3 years is $3600, the principal amount is $10,000. -/
theorem loan_principal_calculation :
  let rate : ℚ := 12 / 100
  let time : ℕ := 3
  let interest : ℚ := 3600
  calculate_principal rate time interest = 10000 := by
  sorry

end loan_principal_calculation_l339_33997


namespace equal_max_attendance_l339_33954

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

-- Define the people
inductive Person
| Anna
| Bill
| Carl
| Dana

-- Define a function that returns whether a person can attend on a given day
def canAttend (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Friday => false
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Thursday => false
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Thursday => false
  | Person.Carl, Day.Friday => false
  | Person.Dana, Day.Wednesday => false
  | _, _ => true

-- Define a function that counts the number of people who can attend on a given day
def attendanceCount (d : Day) : Nat :=
  List.foldl (fun count p => count + if canAttend p d then 1 else 0) 0 [Person.Anna, Person.Bill, Person.Carl, Person.Dana]

-- Statement to prove
theorem equal_max_attendance :
  ∀ d1 d2 : Day, attendanceCount d1 = attendanceCount d2 ∧ attendanceCount d1 = 2 :=
sorry

end equal_max_attendance_l339_33954


namespace exists_perpendicular_line_l339_33961

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define a relation for a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define a relation for two lines being perpendicular
variable (perpendicular : Line → Line → Prop)

-- Theorem statement
theorem exists_perpendicular_line (l : Line) (α : Plane) :
  ∃ (m : Line), in_plane m α ∧ perpendicular m l := by
  sorry

end exists_perpendicular_line_l339_33961


namespace wall_length_proof_l339_33986

/-- Proves that the length of a wall is 800 cm given the dimensions of bricks and wall, and the number of bricks needed. -/
theorem wall_length_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
                          (wall_width : ℝ) (wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 50 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_width = 600 →
  wall_height = 22.5 →
  num_bricks = 3200 →
  ∃ (wall_length : ℝ), wall_length = 800 := by
  sorry

#check wall_length_proof

end wall_length_proof_l339_33986


namespace sand_bucket_calculation_l339_33913

theorem sand_bucket_calculation (bucket_weight : ℕ) (total_weight : ℕ) (h1 : bucket_weight = 2) (h2 : total_weight = 34) :
  total_weight / bucket_weight = 17 := by
  sorry

end sand_bucket_calculation_l339_33913


namespace gordon_jamie_persian_ratio_l339_33929

/-- Represents the number of cats of each type owned by each person -/
structure CatOwnership where
  jamie_persian : ℕ
  jamie_maine_coon : ℕ
  gordon_persian : ℕ
  gordon_maine_coon : ℕ
  hawkeye_persian : ℕ
  hawkeye_maine_coon : ℕ

/-- The theorem stating the ratio of Gordon's Persian cats to Jamie's Persian cats -/
theorem gordon_jamie_persian_ratio (cats : CatOwnership) : 
  cats.jamie_persian = 4 →
  cats.jamie_maine_coon = 2 →
  cats.gordon_maine_coon = cats.jamie_maine_coon + 1 →
  cats.hawkeye_persian = 0 →
  cats.hawkeye_maine_coon = cats.gordon_maine_coon - 1 →
  cats.jamie_persian + cats.jamie_maine_coon + 
  cats.gordon_persian + cats.gordon_maine_coon + 
  cats.hawkeye_persian + cats.hawkeye_maine_coon = 13 →
  2 * cats.gordon_persian = cats.jamie_persian := by
  sorry


end gordon_jamie_persian_ratio_l339_33929


namespace max_books_borrowed_l339_33952

/-- Given a college with the following book borrowing statistics:
    - 200 total students
    - 10 students borrowed 0 books
    - 30 students borrowed 1 book each
    - 40 students borrowed 2 books each
    - 50 students borrowed 3 books each
    - 25 students borrowed 5 books each
    - The average number of books per student is 3

    Prove that the maximum number of books any single student could have borrowed is 215. -/
theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (three_books : ℕ) (five_books : ℕ) (avg_books : ℚ) :
  total_students = 200 →
  zero_books = 10 →
  one_book = 30 →
  two_books = 40 →
  three_books = 50 →
  five_books = 25 →
  avg_books = 3 →
  (zero_books + one_book + two_books + three_books + five_books : ℚ) / total_students = avg_books →
  ∃ (max_books : ℕ), max_books = 215 ∧ 
    ∀ (student_books : ℕ), student_books ≤ max_books :=
by sorry

end max_books_borrowed_l339_33952


namespace tangent_line_at_x_1_l339_33907

noncomputable def f (x : ℝ) : ℝ := 6 * (x^(1/3)) - (16/3) * (x^(1/4))

theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) → y = (2/3) * x :=
by sorry

end tangent_line_at_x_1_l339_33907


namespace constant_is_monomial_l339_33939

/-- A monomial is a constant or a product of variables with non-negative integer exponents. --/
def IsMonomial (x : ℝ) : Prop :=
  x ≠ 0 ∨ ∃ (n : ℕ), x = 1 ∨ x = -1

/-- Theorem: The constant -2010 is a monomial. --/
theorem constant_is_monomial : IsMonomial (-2010) := by
  sorry

end constant_is_monomial_l339_33939


namespace x_range_l339_33938

theorem x_range (x : ℝ) : (Real.sqrt ((5 - x)^2) = x - 5) → x ≥ 5 := by
  sorry

end x_range_l339_33938


namespace range_of_a_l339_33923

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) → -8 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l339_33923


namespace max_trailing_zeros_l339_33973

def trailing_zeros (n : ℕ) : ℕ := sorry

def expr_a : ℕ := 2^5 * 3^4 * 5^6
def expr_b : ℕ := 2^4 * 3^4 * 5^5
def expr_c : ℕ := 4^3 * 5^6 * 6^5
def expr_d : ℕ := 4^2 * 5^4 * 6^3

theorem max_trailing_zeros :
  trailing_zeros expr_c > trailing_zeros expr_a ∧
  trailing_zeros expr_c > trailing_zeros expr_b ∧
  trailing_zeros expr_c > trailing_zeros expr_d :=
sorry

end max_trailing_zeros_l339_33973


namespace grade_assignments_l339_33968

/-- The number of possible grades to assign -/
def num_grades : ℕ := 3

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- Theorem: The number of ways to assign grades to students -/
theorem grade_assignments :
  num_grades ^ num_students = 531441 := by
sorry

end grade_assignments_l339_33968


namespace robertson_seymour_theorem_l339_33905

-- Define a graph type
structure Graph (V : Type) where
  edge : V → V → Prop

-- Define a complete graph
def CompleteGraph (n : ℕ) : Graph (Fin n) where
  edge i j := i ≠ j

-- Define the concept of a minor
def IsMinor {V W : Type} (G : Graph V) (H : Graph W) : Prop := sorry

-- Define tree decomposition
structure TreeDecomposition (V : Type) where
  T : Type
  bags : T → Set V
  -- Other properties of tree decomposition

-- Define k-almost embeddable
def KAlmostEmbeddable (k : ℕ) (G : Graph V) (S : Type) : Prop := sorry

-- Define the concept of a surface where K^n cannot be embedded
def SurfaceWithoutKn (n : ℕ) (S : Type) : Prop := sorry

-- The main theorem
theorem robertson_seymour_theorem {V : Type} (n : ℕ) (hn : n ≥ 5) :
  ∃ k : ℕ, ∀ (G : Graph V),
    ¬IsMinor G (CompleteGraph n) →
    ∃ (td : TreeDecomposition V) (S : Type),
      SurfaceWithoutKn n S ∧
      KAlmostEmbeddable k G S :=
sorry

end robertson_seymour_theorem_l339_33905


namespace star_value_proof_l339_33981

def star (a b : ℤ) : ℤ := a^2 + 2*a*b + b^2

theorem star_value_proof (a b : ℤ) (h : 4 ∣ (a + b)) : 
  a = 3 → b = 5 → star a b = 64 := by
  sorry

end star_value_proof_l339_33981


namespace fifteenth_in_base_8_l339_33942

/-- Converts a decimal number to its representation in base 8 -/
def to_base_8 (n : ℕ) : ℕ := sorry

/-- The fifteenth number in base 10 -/
def fifteenth : ℕ := 15

/-- The representation of the fifteenth number in base 8 -/
def fifteenth_base_8 : ℕ := 17

theorem fifteenth_in_base_8 :
  to_base_8 fifteenth = fifteenth_base_8 := by sorry

end fifteenth_in_base_8_l339_33942


namespace kids_joined_soccer_l339_33964

theorem kids_joined_soccer (initial_kids final_kids : ℕ) (h1 : initial_kids = 14) (h2 : final_kids = 36) :
  final_kids - initial_kids = 22 := by
  sorry

end kids_joined_soccer_l339_33964


namespace all_students_same_room_probability_l339_33983

/-- The number of rooms available for assignment. -/
def num_rooms : ℕ := 4

/-- The number of students being assigned to rooms. -/
def num_students : ℕ := 3

/-- The probability of a student being assigned to any specific room. -/
def prob_per_room : ℚ := 1 / num_rooms

/-- The total number of possible assignment outcomes. -/
def total_outcomes : ℕ := num_rooms ^ num_students

/-- The number of favorable outcomes (all students in the same room). -/
def favorable_outcomes : ℕ := num_rooms

/-- The probability that all students are assigned to the same room. -/
theorem all_students_same_room_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 16 := by
  sorry

end all_students_same_room_probability_l339_33983


namespace base_2_representation_of_123_l339_33914

theorem base_2_representation_of_123 : 
  (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by sorry

end base_2_representation_of_123_l339_33914


namespace probability_of_meeting_l339_33951

def knockout_tournament (n : ℕ) := n > 1

def num_matches (n : ℕ) (h : knockout_tournament n) : ℕ := n - 1

def num_pairs (n : ℕ) : ℕ := n.choose 2

theorem probability_of_meeting (n : ℕ) (h : knockout_tournament n) :
  (num_matches n h : ℚ) / (num_pairs n : ℚ) = 31 / 496 :=
sorry

end probability_of_meeting_l339_33951


namespace square_root_of_four_l339_33904

theorem square_root_of_four : ∃ (x : ℝ), x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

end square_root_of_four_l339_33904


namespace right_triangle_hypotenuse_l339_33971

theorem right_triangle_hypotenuse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (1 / 3 * π * b * a^2 = 1280 * π) → (b / a = 3 / 4) → 
  Real.sqrt (a^2 + b^2) = 20 := by
  sorry

end right_triangle_hypotenuse_l339_33971


namespace remainder_problem_l339_33922

theorem remainder_problem (N : ℤ) (h : N % 133 = 16) : N % 50 = 49 := by
  sorry

end remainder_problem_l339_33922


namespace complex_subtraction_l339_33935

theorem complex_subtraction (z : ℂ) : (5 - 3*I - z = -1 + 4*I) → z = 6 - 7*I := by
  sorry

end complex_subtraction_l339_33935


namespace cube_edge_length_is_sqrt_3_l339_33979

/-- The edge length of a cube inscribed in a sphere with volume 9π/2 --/
def cube_edge_length (s : Real) (c : Real) : Prop :=
  -- All vertices of the cube are on the surface of the sphere
  -- The volume of the sphere is 9π/2
  (4 / 3 * Real.pi * s^3 = 9 * Real.pi / 2) ∧
  -- The space diagonal of the cube is the diameter of the sphere
  (Real.sqrt 3 * c = 2 * s) →
  c = Real.sqrt 3

/-- Theorem stating that the edge length of the cube is √3 --/
theorem cube_edge_length_is_sqrt_3 :
  ∃ (s : Real) (c : Real), cube_edge_length s c :=
by
  sorry

end cube_edge_length_is_sqrt_3_l339_33979


namespace store_pricing_strategy_l339_33937

/-- Calculates the sale price given the cost price and profit percentage -/
def calculateSalePrice (costPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  costPrice * (1 + profitPercentage / 100)

/-- Represents the store's pricing strategy -/
theorem store_pricing_strategy :
  let costA : ℚ := 320
  let costB : ℚ := 480
  let costC : ℚ := 600
  let profitA : ℚ := 50
  let profitB : ℚ := 70
  let profitC : ℚ := 40
  (calculateSalePrice costA profitA = 480) ∧
  (calculateSalePrice costB profitB = 816) ∧
  (calculateSalePrice costC profitC = 840) := by
  sorry

end store_pricing_strategy_l339_33937


namespace sqrt_three_minus_one_over_two_gt_one_third_l339_33957

theorem sqrt_three_minus_one_over_two_gt_one_third : (Real.sqrt 3 - 1) / 2 > 1 / 3 := by
  sorry

end sqrt_three_minus_one_over_two_gt_one_third_l339_33957


namespace total_students_l339_33902

theorem total_students (students_3rd : ℕ) (students_4th : ℕ) (boys_2nd : ℕ) (girls_2nd : ℕ) :
  students_3rd = 19 →
  students_4th = 2 * students_3rd →
  boys_2nd = 10 →
  girls_2nd = 19 →
  students_3rd + students_4th + (boys_2nd + girls_2nd) = 86 :=
by sorry

end total_students_l339_33902


namespace two_std_dev_below_mean_l339_33975

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : 0 < std_dev

/-- The value that is exactly n standard deviations less than the mean -/
def value_n_std_dev_below (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 14.0 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 11.0 -/
theorem two_std_dev_below_mean :
  ∃ (d : NormalDistribution),
    d.mean = 14.0 ∧
    d.std_dev = 1.5 ∧
    value_n_std_dev_below d 2 = 11.0 := by
  sorry

end two_std_dev_below_mean_l339_33975


namespace athlete_heartbeats_l339_33909

/-- Calculates the total number of heartbeats during a race --/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  let race_duration := pace * race_distance
  race_duration * heart_rate

/-- Theorem: The athlete's heart beats 24300 times during the 30-mile race --/
theorem athlete_heartbeats :
  let heart_rate : ℕ := 135  -- beats per minute
  let pace : ℕ := 6          -- minutes per mile
  let race_distance : ℕ := 30 -- miles
  total_heartbeats heart_rate pace race_distance = 24300 :=
by
  sorry

end athlete_heartbeats_l339_33909


namespace darius_age_l339_33955

theorem darius_age (jenna_age darius_age : ℕ) : 
  jenna_age = 13 →
  jenna_age = darius_age + 5 →
  jenna_age + darius_age = 21 →
  darius_age = 8 := by
sorry

end darius_age_l339_33955


namespace dice_throw_outcomes_l339_33956

/-- The number of possible outcomes for a single dice throw -/
def single_throw_outcomes : ℕ := 6

/-- The number of times the dice is thrown -/
def number_of_throws : ℕ := 2

/-- The total number of different outcomes when throwing a dice twice in succession -/
def total_outcomes : ℕ := single_throw_outcomes ^ number_of_throws

theorem dice_throw_outcomes : total_outcomes = 36 := by
  sorry

end dice_throw_outcomes_l339_33956


namespace probability_at_least_two_in_same_class_probability_equals_seven_twentyfifths_l339_33959

/-- The probability that at least 2 out of 3 friends are in the same class, given 10 classes -/
theorem probability_at_least_two_in_same_class : ℝ :=
  let total_classes := 10
  let total_friends := 3
  let prob_all_different := (total_classes * (total_classes - 1) * (total_classes - 2)) / (total_classes ^ total_friends)
  1 - prob_all_different

/-- The probability is equal to 7/25 -/
theorem probability_equals_seven_twentyfifths : probability_at_least_two_in_same_class = 7 / 25 := by
  sorry

end probability_at_least_two_in_same_class_probability_equals_seven_twentyfifths_l339_33959


namespace quadratic_real_roots_discriminant_nonnegative_l339_33918

theorem quadratic_real_roots_discriminant_nonnegative
  (a b c : ℝ) (ha : a ≠ 0)
  (h_real_roots : ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  b^2 - 4*a*c ≥ 0 := by
  sorry

end quadratic_real_roots_discriminant_nonnegative_l339_33918


namespace four_cube_painted_subcubes_l339_33917

/-- Represents a cube with some faces painted -/
structure PaintedCube where
  size : ℕ
  painted_faces : ℕ
  unpainted_faces : ℕ

/-- Calculates the number of subcubes with at least one painted face -/
def subcubes_with_paint (c : PaintedCube) : ℕ :=
  sorry

/-- Theorem stating that a 4x4x4 cube with 4 painted faces has 48 subcubes with paint -/
theorem four_cube_painted_subcubes :
  let c : PaintedCube := { size := 4, painted_faces := 4, unpainted_faces := 2 }
  subcubes_with_paint c = 48 := by
  sorry

end four_cube_painted_subcubes_l339_33917


namespace square_with_equilateral_triangle_l339_33988

/-- Given a square ABCD with side length (R-1) cm and an equilateral triangle AEF 
    where E and F are points on BC and CD respectively, if the area of triangle AEF 
    is (S-3) cm², then S = 2√3. -/
theorem square_with_equilateral_triangle (R S : ℝ) : 
  let square_side := R - 1
  let triangle_area := S - 3
  square_side > 0 →
  triangle_area > 0 →
  S = 2 * Real.sqrt 3 :=
by
  sorry

end square_with_equilateral_triangle_l339_33988


namespace alternating_seating_theorem_l339_33911

/-- The number of ways to arrange n girls and n boys alternately around a round table with 2n seats -/
def alternating_seating_arrangements (n : ℕ) : ℕ :=
  2 * (n.factorial)^2

/-- Theorem stating that the number of alternating seating arrangements
    for n girls and n boys around a round table with 2n seats is 2(n!)^2 -/
theorem alternating_seating_theorem (n : ℕ) :
  alternating_seating_arrangements n = 2 * (n.factorial)^2 := by
  sorry

end alternating_seating_theorem_l339_33911


namespace quadratic_comparison_l339_33900

/-- Given a quadratic function f(x) = a(x-1)^2 + 3 where a < 0,
    if f(-1) = y₁ and f(2) = y₂, then y₁ < y₂ -/
theorem quadratic_comparison (a y₁ y₂ : ℝ) (ha : a < 0) 
    (h1 : y₁ = a * (-1 - 1)^2 + 3)
    (h2 : y₂ = a * (2 - 1)^2 + 3) :
  y₁ < y₂ := by
sorry

end quadratic_comparison_l339_33900


namespace rectangle_area_with_squares_l339_33974

/-- The area of a rectangle containing three non-overlapping squares -/
theorem rectangle_area_with_squares (s : ℝ) (h : s > 0) : 
  let small_square_area := s^2
  let large_square_area := (3*s)^2
  let total_area := 2 * small_square_area + large_square_area
  total_area = 11 * s^2 := by
sorry

end rectangle_area_with_squares_l339_33974


namespace total_earnings_l339_33936

-- Define pizza types
inductive PizzaType
| Margherita
| Pepperoni
| VeggieSupreme
| MeatLovers
| Hawaiian

-- Define pizza prices
def slicePrice (p : PizzaType) : ℚ :=
  match p with
  | .Margherita => 3
  | .Pepperoni => 4
  | .VeggieSupreme => 5
  | .MeatLovers => 6
  | .Hawaiian => 4.5

def wholePizzaPrice (p : PizzaType) : ℚ :=
  match p with
  | .Margherita => 15
  | .Pepperoni => 18
  | .VeggieSupreme => 22
  | .MeatLovers => 25
  | .Hawaiian => 20

-- Define discount and promotion rules
def wholeDiscountRate : ℚ := 0.1
def wholeDiscountThreshold : ℕ := 3
def regularToppingPrice : ℚ := 2
def weekendToppingPrice : ℚ := 1
def happyHourPrice : ℚ := 3

-- Define sales data
structure SalesData where
  margheritaSlices : ℕ
  margheritaHappyHour : ℕ
  pepperoniSlices : ℕ
  pepperoniHappyHour : ℕ
  pepperoniToppings : ℕ
  veggieSupremeWhole : ℕ
  veggieSupremeToppings : ℕ
  margheritaWholePackage : ℕ
  meatLoversSlices : ℕ
  meatLoversHappyHour : ℕ
  hawaiianSlices : ℕ
  hawaiianToppings : ℕ
  pepperoniWholeWeekend : ℕ
  pepperoniWholeToppings : ℕ

def salesData : SalesData := {
  margheritaSlices := 24,
  margheritaHappyHour := 12,
  pepperoniSlices := 16,
  pepperoniHappyHour := 8,
  pepperoniToppings := 6,
  veggieSupremeWhole := 4,
  veggieSupremeToppings := 8,
  margheritaWholePackage := 3,
  meatLoversSlices := 20,
  meatLoversHappyHour := 10,
  hawaiianSlices := 12,
  hawaiianToppings := 4,
  pepperoniWholeWeekend := 1,
  pepperoniWholeToppings := 3
}

-- Theorem statement
theorem total_earnings (data : SalesData) :
  let earnings := 
    (data.margheritaSlices - data.margheritaHappyHour) * slicePrice PizzaType.Margherita +
    data.margheritaHappyHour * happyHourPrice +
    (data.pepperoniSlices - data.pepperoniHappyHour) * slicePrice PizzaType.Pepperoni +
    data.pepperoniHappyHour * happyHourPrice +
    data.pepperoniToppings * weekendToppingPrice +
    data.veggieSupremeWhole * wholePizzaPrice PizzaType.VeggieSupreme +
    data.veggieSupremeToppings * weekendToppingPrice +
    (data.margheritaWholePackage * wholePizzaPrice PizzaType.Margherita) * (1 - wholeDiscountRate) +
    (data.meatLoversSlices - data.meatLoversHappyHour) * slicePrice PizzaType.MeatLovers +
    data.meatLoversHappyHour * happyHourPrice +
    data.hawaiianSlices * slicePrice PizzaType.Hawaiian +
    data.hawaiianToppings * weekendToppingPrice +
    data.pepperoniWholeWeekend * wholePizzaPrice PizzaType.Pepperoni +
    data.pepperoniWholeToppings * weekendToppingPrice
  earnings = 439.5 := by sorry


end total_earnings_l339_33936


namespace geometric_arithmetic_inequality_l339_33990

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n ∧ a n > 0

/-- An arithmetic sequence -/
def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n, b (n + 1) = b n + d

theorem geometric_arithmetic_inequality 
  (a b : ℕ → ℝ) 
  (ha : is_positive_geometric_sequence a) 
  (hb : is_arithmetic_sequence b) 
  (h_eq : a 6 = b 7) : 
  a 3 + a 9 ≥ b 4 + b 10 := by
sorry

end geometric_arithmetic_inequality_l339_33990


namespace complex_magnitude_equality_l339_33962

theorem complex_magnitude_equality (t : ℝ) (h1 : t > 0) :
  Complex.abs (-3 + t * Complex.I) = 3 * Real.sqrt 5 → t = 6 := by
sorry

end complex_magnitude_equality_l339_33962


namespace center_of_gravity_semicircle_semidisk_l339_33953

/-- The center of gravity of a homogeneous semicircle and semi-disk -/
theorem center_of_gravity_semicircle_semidisk (r : ℝ) (hr : r > 0) :
  ∃ (y z : ℝ),
    y = (2 * r) / Real.pi ∧
    z = (4 * r) / (3 * Real.pi) ∧
    y > 0 ∧ z > 0 :=
by sorry

end center_of_gravity_semicircle_semidisk_l339_33953


namespace contrapositive_example_l339_33976

theorem contrapositive_example :
  (∀ x : ℝ, x = 2 → x^2 - 3*x + 2 = 0) ↔
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 2) :=
by sorry

end contrapositive_example_l339_33976


namespace m_plus_p_equals_11_l339_33943

/-- Sum of odd numbers from 1 to n -/
def sumOddNumbers (n : ℕ) : ℕ :=
  (n + 1) * n / 2

/-- Decomposition of p^3 for positive integers p ≥ 2 -/
def decompositionP3 (p : ℕ) : ℕ :=
  2 * p * p - 1

theorem m_plus_p_equals_11 (m p : ℕ) 
  (h1 : m ^ 2 = sumOddNumbers 6)
  (h2 : decompositionP3 p = 21) : 
  m + p = 11 := by
  sorry

#eval sumOddNumbers 6  -- Should output 36
#eval decompositionP3 5  -- Should output 21

end m_plus_p_equals_11_l339_33943


namespace max_intersection_points_four_spheres_l339_33934

/-- A sphere in three-dimensional space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- A line in three-dimensional space -/
structure Line where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- The number of intersection points between a line and a sphere -/
def intersectionPoints (l : Line) (s : Sphere) : ℕ := sorry

/-- The theorem stating the maximum number of intersection points -/
theorem max_intersection_points_four_spheres (s₁ s₂ s₃ s₄ : Sphere) :
  ∃ (l : Line), (intersectionPoints l s₁) + (intersectionPoints l s₂) +
                (intersectionPoints l s₃) + (intersectionPoints l s₄) ≤ 8 :=
sorry

end max_intersection_points_four_spheres_l339_33934


namespace dave_guitar_strings_l339_33912

/-- The number of guitar strings Dave breaks per night -/
def strings_per_night : ℕ := 2

/-- The number of shows Dave performs per week -/
def shows_per_week : ℕ := 6

/-- The number of weeks Dave performs -/
def total_weeks : ℕ := 12

/-- The total number of guitar strings Dave needs to replace -/
def total_strings_replaced : ℕ := strings_per_night * shows_per_week * total_weeks

theorem dave_guitar_strings :
  total_strings_replaced = 144 :=
by sorry

end dave_guitar_strings_l339_33912


namespace abe_age_sum_l339_33999

theorem abe_age_sum : 
  let present_age : ℕ := 29
  let years_ago : ℕ := 7
  let past_age : ℕ := present_age - years_ago
  present_age + past_age = 51
  := by sorry

end abe_age_sum_l339_33999


namespace sum_difference_theorem_l339_33915

def sara_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (x : ℕ) : ℕ :=
  5 * ((x + 2) / 5)

def mike_sum (n : ℕ) : ℕ :=
  List.sum (List.map round_to_nearest_five (List.range n))

theorem sum_difference_theorem :
  sara_sum 120 - mike_sum 120 = 6900 := by
  sorry

end sum_difference_theorem_l339_33915


namespace sin_alpha_cos_beta_value_l339_33928

/-- Given two points on the unit circle representing the terminal sides of angles α and β,
    prove that sin(α) * cos(β) equals a specific value. -/
theorem sin_alpha_cos_beta_value (α β : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 12/13 ∧ y = 5/13 ∧ 
   Real.cos α = x ∧ Real.sin α = y) →
  (∃ (u v : Real), u^2 + v^2 = 1 ∧ u = -3/5 ∧ v = 4/5 ∧ 
   Real.cos β = u ∧ Real.sin β = v) →
  Real.sin α * Real.cos β = -15/65 := by
sorry

end sin_alpha_cos_beta_value_l339_33928


namespace sin_shift_l339_33991

theorem sin_shift (x : ℝ) : Real.sin (2 * x + π / 3) = Real.sin (2 * (x + π / 6)) := by
  sorry

end sin_shift_l339_33991


namespace triangle_problem_l339_33998

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π / 2 →  -- Acute angle A
  0 < B ∧ B < π / 2 →  -- Acute angle B
  0 < C ∧ C < π / 2 →  -- Acute angle C
  A + B + C = π →      -- Sum of angles in a triangle
  b + c = 10 →         -- Given condition
  a = Real.sqrt 10 →   -- Given condition
  5 * b * Real.sin A * Real.cos C + 5 * c * Real.sin A * Real.cos B = 3 * Real.sqrt 10 → -- Given condition
  Real.cos A = 4 / 5 ∧ b = 5 ∧ c = 5 := by
sorry


end triangle_problem_l339_33998


namespace dice_game_probability_l339_33967

def score (roll1 roll2 : Nat) : Nat := max roll1 roll2

def is_favorable (roll1 roll2 : Nat) : Bool :=
  score roll1 roll2 ≤ 3

def total_outcomes : Nat := 36

def favorable_outcomes : Nat := 9

theorem dice_game_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
  sorry

end dice_game_probability_l339_33967


namespace unique_factorial_product_l339_33958

theorem unique_factorial_product (n : ℕ) : (n + 1) * n.factorial = 5040 ↔ n = 6 := by sorry

end unique_factorial_product_l339_33958


namespace arithmetic_sequence_problem_l339_33977

/-- An arithmetic sequence with positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_positive : d > 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 1 + seq.a 2 + seq.a 3 = 15)
  (h2 : seq.a 1 * seq.a 2 * seq.a 3 = 80) :
  seq.a 11 + seq.a 12 + seq.a 13 = 105 := by
  sorry

end arithmetic_sequence_problem_l339_33977


namespace mrs_brown_utility_bill_l339_33916

/-- Calculates the actual utility bill amount given the initial payment and returned amount -/
def actualUtilityBill (initialPayment returnedAmount : ℕ) : ℕ :=
  initialPayment - returnedAmount

/-- Theorem stating that Mrs. Brown's actual utility bill is $710 -/
theorem mrs_brown_utility_bill :
  let initialPayment := 4 * 100 + 5 * 50 + 7 * 20
  let returnedAmount := 3 * 20 + 2 * 10
  actualUtilityBill initialPayment returnedAmount = 710 := by
  sorry

#eval actualUtilityBill (4 * 100 + 5 * 50 + 7 * 20) (3 * 20 + 2 * 10)

end mrs_brown_utility_bill_l339_33916
