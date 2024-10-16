import Mathlib

namespace NUMINAMATH_CALUDE_division_problem_l2418_241850

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) : 
  dividend = divisor + 2016 →
  quotient = 15 →
  dividend = divisor * quotient →
  dividend = 2160 :=
by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2418_241850


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_two_l2418_241844

theorem cube_root_sum_equals_two :
  ∃ n : ℤ, (Real.rpow (2 + 10/9 * Real.sqrt 3) (1/3 : ℝ) + Real.rpow (2 - 10/9 * Real.sqrt 3) (1/3 : ℝ) = n) →
  Real.rpow (2 + 10/9 * Real.sqrt 3) (1/3 : ℝ) + Real.rpow (2 - 10/9 * Real.sqrt 3) (1/3 : ℝ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_two_l2418_241844


namespace NUMINAMATH_CALUDE_modulus_of_complex_expression_l2418_241817

theorem modulus_of_complex_expression :
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I) + 2 * Complex.I
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_expression_l2418_241817


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2418_241863

/-- Given a quadratic inequality ax^2 + b > 0 with solution set (-∞, -1/2) ∪ (1/3, ∞), prove ab = 24 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, a * x^2 + b > 0 ↔ x < -1/2 ∨ x > 1/3) → a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2418_241863


namespace NUMINAMATH_CALUDE_remainder_of_5n_mod_11_l2418_241859

theorem remainder_of_5n_mod_11 (n : ℤ) (h : n % 11 = 1) : (5 * n) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_5n_mod_11_l2418_241859


namespace NUMINAMATH_CALUDE_abc_maximum_l2418_241842

theorem abc_maximum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : 2*a + 4*b + 8*c = 16) : a*b*c ≤ 64/27 := by
  sorry

end NUMINAMATH_CALUDE_abc_maximum_l2418_241842


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2418_241887

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 2 → x^2 > 4) ↔ (∃ x : ℝ, x > 2 ∧ x^2 ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2418_241887


namespace NUMINAMATH_CALUDE_negative_one_to_zero_power_l2418_241848

theorem negative_one_to_zero_power : ((-1 : ℤ) ^ (0 : ℕ)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_to_zero_power_l2418_241848


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_64_l2418_241861

theorem sqrt_49_times_sqrt_64 : Real.sqrt (49 * Real.sqrt 64) = 14 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_64_l2418_241861


namespace NUMINAMATH_CALUDE_exhibit_fish_count_l2418_241828

/-- The number of pufferfish in the exhibit -/
def num_pufferfish : ℕ := 15

/-- The ratio of swordfish to pufferfish -/
def swordfish_ratio : ℕ := 5

/-- The total number of fish in the exhibit -/
def total_fish : ℕ := num_pufferfish + swordfish_ratio * num_pufferfish

theorem exhibit_fish_count : total_fish = 90 := by
  sorry

end NUMINAMATH_CALUDE_exhibit_fish_count_l2418_241828


namespace NUMINAMATH_CALUDE_no_double_square_sum_l2418_241835

theorem no_double_square_sum (x y : ℕ) : 
  ¬(∃ (a b : ℕ), a^2 = x^2 + y ∧ b^2 = y^2 + x) := by
  sorry

end NUMINAMATH_CALUDE_no_double_square_sum_l2418_241835


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2418_241860

-- Define the curve
def f (x : ℝ) : ℝ := 2*x - x^3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 2 - 3*x^2

-- Define the point of tangency
def x₀ : ℝ := -1
def y₀ : ℝ := f x₀

-- Define the slope of the tangent line at x₀
def m : ℝ := f' x₀

-- Statement: The equation of the tangent line is x + y + 2 = 0
theorem tangent_line_equation :
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ x + y + 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2418_241860


namespace NUMINAMATH_CALUDE_divides_totient_power_two_minus_one_l2418_241849

theorem divides_totient_power_two_minus_one (n : ℕ) (hn : n > 0) : 
  n ∣ Nat.totient (2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_divides_totient_power_two_minus_one_l2418_241849


namespace NUMINAMATH_CALUDE_professor_seating_arrangements_l2418_241813

/-- Represents the number of chairs in a row -/
def total_chairs : ℕ := 14

/-- Represents the number of professors -/
def num_professors : ℕ := 4

/-- Represents the number of students -/
def num_students : ℕ := 10

/-- Represents the number of possible positions for professors (excluding first and last chair) -/
def professor_positions : ℕ := total_chairs - 2

/-- Theorem stating the number of ways professors can choose their chairs -/
theorem professor_seating_arrangements :
  (∃ (two_adjacent : ℕ) (three_adjacent : ℕ) (four_adjacent : ℕ),
    two_adjacent = (professor_positions - 1) * (Nat.choose (professor_positions - 2) 2) * (Nat.factorial num_professors / 2) ∧
    three_adjacent = (professor_positions - 2) * (professor_positions - 3) * (Nat.factorial num_professors) ∧
    four_adjacent = (professor_positions - 3) * (Nat.factorial num_professors) ∧
    two_adjacent + three_adjacent + four_adjacent = 5346) :=
by sorry

end NUMINAMATH_CALUDE_professor_seating_arrangements_l2418_241813


namespace NUMINAMATH_CALUDE_veronica_initial_marbles_l2418_241838

/-- Represents the number of marbles each person has -/
structure Marbles where
  dilan : ℕ
  martha : ℕ
  phillip : ℕ
  veronica : ℕ

/-- The initial distribution of marbles -/
def initial_marbles : Marbles where
  dilan := 14
  martha := 20
  phillip := 19
  veronica := 7  -- We'll prove this is correct

/-- The number of people -/
def num_people : ℕ := 4

/-- The number of marbles each person has after redistribution -/
def marbles_after_redistribution : ℕ := 15

theorem veronica_initial_marbles :
  (initial_marbles.dilan +
   initial_marbles.martha +
   initial_marbles.phillip +
   initial_marbles.veronica) =
  (num_people * marbles_after_redistribution) :=
by sorry

end NUMINAMATH_CALUDE_veronica_initial_marbles_l2418_241838


namespace NUMINAMATH_CALUDE_factorization_proof_l2418_241830

theorem factorization_proof (x : ℝ) : -2 * x^2 + 18 = -2 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2418_241830


namespace NUMINAMATH_CALUDE_equation_solution_l2418_241862

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (27 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 33 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2418_241862


namespace NUMINAMATH_CALUDE_sarahs_journey_length_l2418_241865

theorem sarahs_journey_length :
  ∀ (total : ℚ),
    (1 / 4 : ℚ) * total +   -- First part
    30 +                    -- Second part
    (1 / 6 : ℚ) * total =   -- Third part
    total →                 -- Sum of parts equals total
    total = 360 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_journey_length_l2418_241865


namespace NUMINAMATH_CALUDE_complement_intersection_equality_l2418_241896

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x ≥ 2}
def Q : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- Define the open interval (1, 2)
def open_interval : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}

-- State the theorem
theorem complement_intersection_equality :
  (Set.univ \ P) ∩ Q = open_interval :=
sorry

end NUMINAMATH_CALUDE_complement_intersection_equality_l2418_241896


namespace NUMINAMATH_CALUDE_tank_emptying_time_l2418_241891

/-- Proves that a tank with given properties will empty in 12 hours due to a leak --/
theorem tank_emptying_time 
  (tank_capacity : ℝ) 
  (inlet_rate : ℝ) 
  (emptying_time_with_inlet : ℝ) 
  (h1 : tank_capacity = 5760) 
  (h2 : inlet_rate = 4) 
  (h3 : emptying_time_with_inlet = 8) : 
  (tank_capacity / (inlet_rate - tank_capacity / (emptying_time_with_inlet * 60))) = 12 * 60 := by
  sorry

end NUMINAMATH_CALUDE_tank_emptying_time_l2418_241891


namespace NUMINAMATH_CALUDE_power_of_power_l2418_241809

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2418_241809


namespace NUMINAMATH_CALUDE_committee_formation_proof_l2418_241875

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_formation_proof :
  let total_students : ℕ := 8
  let committee_size : ℕ := 5
  let always_included : ℕ := 2
  let remaining_students : ℕ := total_students - always_included
  let students_to_choose : ℕ := committee_size - always_included
  choose remaining_students students_to_choose = 20 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_proof_l2418_241875


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l2418_241836

theorem no_solutions_in_interval (x : ℝ) :
  x ∈ Set.Ioo 0 (π / 6) →
  3 * Real.tan (2 * x) - 4 * Real.tan (3 * x) ≠ Real.tan (3 * x) ^ 2 * Real.tan (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l2418_241836


namespace NUMINAMATH_CALUDE_crocus_bulbs_count_l2418_241823

/-- Represents the number of crocus bulbs that can be bought given the constraints -/
def crocus_bulbs : ℕ := 22

/-- Represents the number of daffodil bulbs that can be bought given the constraints -/
def daffodil_bulbs : ℕ := 55 - crocus_bulbs

/-- The total number of bulbs -/
def total_bulbs : ℕ := 55

/-- The cost of a single crocus bulb in cents -/
def crocus_cost : ℕ := 35

/-- The cost of a single daffodil bulb in cents -/
def daffodil_cost : ℕ := 65

/-- The total budget in cents -/
def total_budget : ℕ := 2915

theorem crocus_bulbs_count : 
  crocus_bulbs = 22 ∧ 
  crocus_bulbs + daffodil_bulbs = total_bulbs ∧ 
  crocus_bulbs * crocus_cost + daffodil_bulbs * daffodil_cost = total_budget := by
  sorry

end NUMINAMATH_CALUDE_crocus_bulbs_count_l2418_241823


namespace NUMINAMATH_CALUDE_modulus_of_z_equals_sqrt_two_l2418_241815

theorem modulus_of_z_equals_sqrt_two :
  let z : ℂ := (Complex.I + 1) / Complex.I
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_equals_sqrt_two_l2418_241815


namespace NUMINAMATH_CALUDE_find_c_l2418_241885

/-- Given two functions p and q, prove that c = 6 when p(q(3)) = 10 -/
theorem find_c (p q : ℝ → ℝ) (c : ℝ) : 
  (∀ x, p x = 3 * x - 8) →
  (∀ x, q x = 4 * x - c) →
  p (q 3) = 10 →
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_find_c_l2418_241885


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2418_241829

theorem negation_of_proposition :
  (¬(∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0)) ↔
  (∀ a b : ℝ, a^2 + b^2 = 0 → a ≠ 0 ∨ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2418_241829


namespace NUMINAMATH_CALUDE_smallest_cube_with_divisor_l2418_241800

theorem smallest_cube_with_divisor (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r →
  (∀ m : ℕ, m < (p * q * r^2)^3 → ¬(∃ k : ℕ, m = k^3 ∧ p^2 * q^3 * r^5 ∣ m)) →
  (p * q * r^2)^3 = (p * q * r^2)^3 ∧ p^2 * q^3 * r^5 ∣ (p * q * r^2)^3 := by
  sorry

#check smallest_cube_with_divisor

end NUMINAMATH_CALUDE_smallest_cube_with_divisor_l2418_241800


namespace NUMINAMATH_CALUDE_test_questions_l2418_241814

theorem test_questions (total_points : ℕ) (four_point_questions : ℕ) 
  (h1 : total_points = 100)
  (h2 : four_point_questions = 10) : 
  ∃ (two_point_questions : ℕ),
    two_point_questions * 2 + four_point_questions * 4 = total_points ∧
    two_point_questions + four_point_questions = 40 := by
  sorry

end NUMINAMATH_CALUDE_test_questions_l2418_241814


namespace NUMINAMATH_CALUDE_polygon_sides_proof_l2418_241807

theorem polygon_sides_proof (n : ℕ) : 
  let sides1 := n
  let sides2 := n + 4
  let sides3 := n + 12
  let sides4 := n + 13
  let diagonals (m : ℕ) := m * (m - 3) / 2
  diagonals sides1 + diagonals sides4 = diagonals sides2 + diagonals sides3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_proof_l2418_241807


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2418_241897

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 2)
  (h_ninth : a 9 = 8) :
  a 5 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2418_241897


namespace NUMINAMATH_CALUDE_function_has_two_zeros_l2418_241802

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x - 3 else -2 + Real.log x

theorem function_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_function_has_two_zeros_l2418_241802


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_f_l2418_241832

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem smallest_positive_period_of_f : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_f_l2418_241832


namespace NUMINAMATH_CALUDE_union_equals_B_exists_union_equals_intersection_l2418_241806

-- Define the sets A, B, and C
def A : Set ℝ := {x | |x - 1| < 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x - 6 < 0}
def C : Set ℝ := {x | x^2 - 2*x - 15 < 0}

-- Statement 1
theorem union_equals_B (a : ℝ) : 
  A ∪ B a = B a ↔ a ∈ Set.Icc (-5 : ℝ) (-1 : ℝ) := by sorry

-- Statement 2
theorem exists_union_equals_intersection :
  ∃ a ∈ Set.Icc (-19/5 : ℝ) (-1 : ℝ), A ∪ B a = B a ∩ C := by sorry

end NUMINAMATH_CALUDE_union_equals_B_exists_union_equals_intersection_l2418_241806


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2418_241818

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 9 / b ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2418_241818


namespace NUMINAMATH_CALUDE_vertical_shift_theorem_l2418_241852

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define a constant for vertical shift
variable (k : ℝ)

-- Define the shifted function
def shifted_f (x : ℝ) : ℝ := f x + k

-- Theorem: The graph of y = f(x) + k is a vertical shift of y = f(x) by k units
theorem vertical_shift_theorem :
  ∀ (x y : ℝ), y = shifted_f f k x ↔ y - k = f x :=
by sorry

end NUMINAMATH_CALUDE_vertical_shift_theorem_l2418_241852


namespace NUMINAMATH_CALUDE_snow_volume_calculation_l2418_241881

/-- Calculates the volume of snow to be shoveled from a partially melted rectangular pathway -/
theorem snow_volume_calculation (length width : ℝ) (depth_full depth_half : ℝ) 
  (h_length : length = 30)
  (h_width : width = 4)
  (h_depth_full : depth_full = 1)
  (h_depth_half : depth_half = 1/2) :
  length * width * depth_full / 2 + length * width * depth_half / 2 = 90 :=
by sorry

end NUMINAMATH_CALUDE_snow_volume_calculation_l2418_241881


namespace NUMINAMATH_CALUDE_isabel_candy_count_l2418_241803

/-- The total number of candy pieces Isabel has -/
def total_candy (initial : ℕ) (from_friend : ℕ) (from_cousin : ℕ) : ℕ :=
  initial + from_friend + from_cousin

/-- Theorem stating the total number of candy pieces Isabel has -/
theorem isabel_candy_count :
  ∀ x : ℕ, total_candy 216 137 x = 353 + x :=
by sorry

end NUMINAMATH_CALUDE_isabel_candy_count_l2418_241803


namespace NUMINAMATH_CALUDE_chess_tournament_l2418_241843

theorem chess_tournament (W M : ℕ) 
  (h1 : W * (W - 1) / 2 = 45)  -- Number of games with both women
  (h2 : W * M = 200)           -- Number of games with one man and one woman
  : M * (M - 1) / 2 = 190 :=   -- Number of games with both men
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_l2418_241843


namespace NUMINAMATH_CALUDE_sum_of_gcd_and_binary_l2418_241882

theorem sum_of_gcd_and_binary : ∃ (a b : ℕ),
  (Nat.gcd 98 63 = a) ∧
  (((1 : ℕ) * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = b) ∧
  (a + b = 58) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_gcd_and_binary_l2418_241882


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l2418_241892

theorem max_sum_of_factors (diamond : ℕ) (delta : ℕ) : 
  diamond * delta = 60 → (∀ x y : ℕ, x * y = 60 → x + y ≤ diamond + delta) → diamond + delta = 61 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l2418_241892


namespace NUMINAMATH_CALUDE_rectangle_min_area_l2418_241878

/-- A rectangle with integer dimensions and perimeter 60 has minimum area 29 when the shorter dimension is minimized. -/
theorem rectangle_min_area : ∀ l w : ℕ,
  l > 0 → w > 0 →
  2 * (l + w) = 60 →
  ∀ l' w' : ℕ, l' > 0 → w' > 0 → 2 * (l' + w') = 60 →
  min l w ≤ min l' w' →
  l * w ≥ 29 := by
sorry

end NUMINAMATH_CALUDE_rectangle_min_area_l2418_241878


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2418_241867

theorem quadratic_solution_sum (m n : ℝ) (h1 : m ≠ 0) :
  m * 1^2 + n * 1 - 1 = 0 → m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2418_241867


namespace NUMINAMATH_CALUDE_henrys_chore_earnings_l2418_241825

/-- The amount of money Henry earned doing chores -/
def henrys_earnings : ℕ := sorry

/-- The amount of money Henry already had -/
def henrys_initial_money : ℕ := 5

/-- The amount of money Henry's friend had -/
def friends_money : ℕ := 13

/-- The total amount when they put their money together -/
def total_money : ℕ := 20

theorem henrys_chore_earnings :
  henrys_earnings = 2 ∧
  henrys_initial_money + henrys_earnings + friends_money = total_money :=
sorry

end NUMINAMATH_CALUDE_henrys_chore_earnings_l2418_241825


namespace NUMINAMATH_CALUDE_quadratic_circle_intersection_l2418_241866

/-- Given a quadratic polynomial ax^2 + bx + c where a ≠ 0, if a circle passes through
    its three intersection points with the coordinate axes and intersects the y-axis
    at a fourth point with ordinate y₀, then y₀ = 1/a -/
theorem quadratic_circle_intersection 
  (a b c : ℝ) (h : a ≠ 0) : 
  ∃ (x₁ x₂ : ℝ), 
    (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
    (∃ y₀ : ℝ, y₀ * c = x₁ * x₂) →
    (∀ y₀ : ℝ, y₀ * c = x₁ * x₂ → y₀ = 1 / a) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_circle_intersection_l2418_241866


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2418_241822

/-- Proves that the ratio of Rommel's age to Tim's age is 3:1 -/
theorem age_ratio_proof (tim_age : ℕ) (rommel_age : ℕ) (jenny_age : ℕ) : 
  tim_age = 5 →
  jenny_age = rommel_age + 2 →
  jenny_age = tim_age + 12 →
  rommel_age / tim_age = 3 := by
  sorry

#check age_ratio_proof

end NUMINAMATH_CALUDE_age_ratio_proof_l2418_241822


namespace NUMINAMATH_CALUDE_degree_of_specific_monomial_l2418_241888

def monomial_degree (coeff : ℤ) (vars : List (Char × ℕ)) : ℕ :=
  (vars.map (·.2)).sum

theorem degree_of_specific_monomial :
  monomial_degree (-5) [('a', 2), ('b', 3)] = 5 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_specific_monomial_l2418_241888


namespace NUMINAMATH_CALUDE_cyclist_speed_ratio_l2418_241894

theorem cyclist_speed_ratio :
  ∀ (v_A v_B : ℝ),
    v_A > 0 →
    v_B > 0 →
    v_A < v_B →
    (v_B - v_A) * 4.5 = 10 →
    v_A + v_B = 10 →
    v_A / v_B = 61 / 29 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_ratio_l2418_241894


namespace NUMINAMATH_CALUDE_square_area_remainder_l2418_241851

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Square in 2D space -/
structure Square where
  center : Point
  side_length : ℝ

/-- Check if a point is on a side of a square -/
def is_on_side (p : Point) (s : Square) : Prop :=
  (abs (p.x - s.center.x) = s.side_length / 2 ∧ abs (p.y - s.center.y) ≤ s.side_length / 2) ∨
  (abs (p.y - s.center.y) = s.side_length / 2 ∧ abs (p.x - s.center.x) ≤ s.side_length / 2)

theorem square_area_remainder (A B C D : Point) (S : Square) :
  A.x = 0 ∧ A.y = 12 ∧
  B.x = 10 ∧ B.y = 9 ∧
  C.x = 8 ∧ C.y = 0 ∧
  D.x = -4 ∧ D.y = 7 ∧
  is_on_side A S ∧ is_on_side B S ∧ is_on_side C S ∧ is_on_side D S ∧
  (∀ S' : Square, is_on_side A S' ∧ is_on_side B S' ∧ is_on_side C S' ∧ is_on_side D S' → S' = S) →
  (10 * S.side_length ^ 2) % 1000 = 936 := by
  sorry

end NUMINAMATH_CALUDE_square_area_remainder_l2418_241851


namespace NUMINAMATH_CALUDE_cubic_function_property_l2418_241847

theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x - 4
  f 2 = 6 → f (-2) = -14 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2418_241847


namespace NUMINAMATH_CALUDE_range_of_a_l2418_241890

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + a > 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a :
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∃ a : ℝ, a ≤ 0 ∨ (1/4 < a ∧ a < 4)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2418_241890


namespace NUMINAMATH_CALUDE_max_candy_consumption_l2418_241870

theorem max_candy_consumption (n : ℕ) (h : n = 45) : 
  (n * (n - 1)) / 2 = 990 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_consumption_l2418_241870


namespace NUMINAMATH_CALUDE_negation_of_conditional_l2418_241837

theorem negation_of_conditional (x : ℝ) :
  (¬(x = 3 → x^2 - 2*x - 3 = 0)) ↔ (x ≠ 3 → x^2 - 2*x - 3 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l2418_241837


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l2418_241884

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l2418_241884


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l2418_241898

/-- Represents a rectangle inscribed in a circular segment -/
structure InscribedRectangle where
  h : ℝ  -- height of the segment
  ab : ℝ  -- length of side AB
  bc : ℝ  -- length of side BC
  arc_angle : ℝ  -- angle of the arc in degrees
  ab_bc_ratio : ab / bc = 1 / 4  -- ratio condition
  bc_on_chord : True  -- BC lies on the chord (represented as a trivial condition)

/-- The area of an inscribed rectangle -/
def area (r : InscribedRectangle) : ℝ := r.ab * r.bc

/-- Theorem stating the area of the inscribed rectangle -/
theorem inscribed_rectangle_area (r : InscribedRectangle) 
  (h_arc : r.arc_angle = 120) : 
  area r = (36 * r.h^2) / 25 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l2418_241898


namespace NUMINAMATH_CALUDE_odd_integers_equality_l2418_241816

theorem odd_integers_equality (a b : ℕ) (ha : Odd a) (hb : Odd b) (hpos_a : 0 < a) (hpos_b : 0 < b)
  (h_div : (2 * a * b + 1) ∣ (a^2 + b^2 + 1)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_odd_integers_equality_l2418_241816


namespace NUMINAMATH_CALUDE_original_number_proof_l2418_241846

theorem original_number_proof (q : ℝ) : 
  (q + 0.125 * q) - (q - 0.25 * q) = 30 → q = 80 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l2418_241846


namespace NUMINAMATH_CALUDE_hannah_strawberry_harvest_l2418_241876

/-- Hannah's strawberry harvest problem -/
theorem hannah_strawberry_harvest :
  let daily_harvest : ℕ := 5
  let days_in_april : ℕ := 30
  let given_away : ℕ := 20
  let stolen : ℕ := 30
  let total_harvested : ℕ := daily_harvest * days_in_april
  let remaining_after_giving : ℕ := total_harvested - given_away
  let final_count : ℕ := remaining_after_giving - stolen
  final_count = 100 := by sorry

end NUMINAMATH_CALUDE_hannah_strawberry_harvest_l2418_241876


namespace NUMINAMATH_CALUDE_displeased_polynomial_at_one_is_zero_l2418_241871

-- Define a polynomial p(x) = x^2 - (m+n)x + mn
def p (m n : ℝ) (x : ℝ) : ℝ := x^2 - (m + n) * x + m * n

-- Define what it means for a polynomial to be displeased
def isDispleased (m n : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
  (∀ x : ℝ, p m n (p m n x) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

-- Define the theorem
theorem displeased_polynomial_at_one_is_zero :
  ∃! (a : ℝ), isDispleased a a ∧
  (∀ m n : ℝ, isDispleased m n → m * n ≤ a * a) ∧
  p a a 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_displeased_polynomial_at_one_is_zero_l2418_241871


namespace NUMINAMATH_CALUDE_average_people_moving_per_hour_l2418_241895

/-- The number of people moving to Florida -/
def people_moving : ℕ := 3000

/-- The number of days -/
def days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculate the average number of people moving per hour -/
def average_per_hour : ℚ :=
  people_moving / (days * hours_per_day)

/-- Round a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem average_people_moving_per_hour :
  round_to_nearest average_per_hour = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_people_moving_per_hour_l2418_241895


namespace NUMINAMATH_CALUDE_equal_numbers_product_l2418_241879

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 12 →
  b = 22 →
  c = 16 →
  d = e →
  d * e = 625 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l2418_241879


namespace NUMINAMATH_CALUDE_black_friday_tv_sales_increase_black_friday_tv_sales_increase_proof_l2418_241827

theorem black_friday_tv_sales_increase : ℕ → Prop :=
  fun increase =>
    ∃ (T : ℕ),
      T + increase = 327 ∧
      T + 3 * increase = 477 ∧
      increase = 75

-- The proof would go here, but we'll use sorry as instructed
theorem black_friday_tv_sales_increase_proof :
  ∃ increase, black_friday_tv_sales_increase increase :=
by sorry

end NUMINAMATH_CALUDE_black_friday_tv_sales_increase_black_friday_tv_sales_increase_proof_l2418_241827


namespace NUMINAMATH_CALUDE_parallel_range_perpendicular_min_abs_product_l2418_241874

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + a^2 * y + 1 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a^2 + 1) * x - b * y + 3 = 0

-- Define parallel lines
def parallel (a b : ℝ) : Prop := ∀ x y, l₁ a x y ↔ l₂ a b x y

-- Define perpendicular lines
def perpendicular (a b : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, 
  l₁ a x₁ y₁ → l₂ a b x₂ y₂ → (x₁ - x₂) * (y₁ - y₂) = 0

-- Statement 1: If l₁ ∥ l₂, then b ∈ (-∞, -6) ∪ (-6, 0]
theorem parallel_range (a b : ℝ) : 
  parallel a b → b < -6 ∨ (-6 < b ∧ b ≤ 0) :=
sorry

-- Statement 2: If l₁ ⟂ l₂, then the minimum value of |ab| is 2
theorem perpendicular_min_abs_product (a b : ℝ) :
  perpendicular a b → |a * b| ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_parallel_range_perpendicular_min_abs_product_l2418_241874


namespace NUMINAMATH_CALUDE_hibiscus_flowers_solution_l2418_241872

/-- The number of flowers on Mario's first hibiscus plant -/
def first_plant_flowers : ℕ := 2

/-- The number of flowers on Mario's second hibiscus plant -/
def second_plant_flowers : ℕ := 2 * first_plant_flowers

/-- The number of flowers on Mario's third hibiscus plant -/
def third_plant_flowers : ℕ := 4 * second_plant_flowers

/-- The total number of flowers on all three plants -/
def total_flowers : ℕ := 22

theorem hibiscus_flowers_solution :
  first_plant_flowers + second_plant_flowers + third_plant_flowers = total_flowers ∧
  first_plant_flowers = 2 := by
  sorry

end NUMINAMATH_CALUDE_hibiscus_flowers_solution_l2418_241872


namespace NUMINAMATH_CALUDE_complement_of_M_l2418_241873

-- Define the universal set U
def U : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define the set M
def M : Set ℝ := {x | x^2 - x ≤ 0}

-- Theorem statement
theorem complement_of_M (x : ℝ) : 
  x ∈ (U \ M) ↔ (1 < x ∧ x ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_complement_of_M_l2418_241873


namespace NUMINAMATH_CALUDE_joe_caramel_probability_l2418_241869

/-- Represents the set of candies in Joe's pocket -/
structure CandySet :=
  (lemon : ℕ)
  (caramel : ℕ)

/-- Calculates the probability of selecting a caramel-flavored candy -/
def probability_caramel (cs : CandySet) : ℚ :=
  cs.caramel / (cs.lemon + cs.caramel)

/-- Theorem stating that the probability of selecting a caramel-flavored candy
    from Joe's set is 3/7 -/
theorem joe_caramel_probability :
  let joe_candies : CandySet := { lemon := 4, caramel := 3 }
  probability_caramel joe_candies = 3 / 7 := by
  sorry


end NUMINAMATH_CALUDE_joe_caramel_probability_l2418_241869


namespace NUMINAMATH_CALUDE_next_three_same_calendar_years_l2418_241821

/-- A function that determines if a given year is a leap year -/
def isLeapYear (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

/-- The number of years between consecutive years with the same calendar -/
def calendarCycle : ℕ := 28

/-- The base year from which we start calculating -/
def baseYear : ℕ := 2024

/-- A function that calculates the nth year with the same calendar as the base year -/
def nthSameCalendarYear (n : ℕ) : ℕ :=
  baseYear + n * calendarCycle

/-- Theorem stating that the next three years following 2024 with the same calendar
    are 2052, 2080, and 2108 -/
theorem next_three_same_calendar_years :
  (nthSameCalendarYear 1 = 2052) ∧
  (nthSameCalendarYear 2 = 2080) ∧
  (nthSameCalendarYear 3 = 2108) ∧
  (isLeapYear baseYear) ∧
  (∀ n : ℕ, isLeapYear (nthSameCalendarYear n)) :=
sorry

end NUMINAMATH_CALUDE_next_three_same_calendar_years_l2418_241821


namespace NUMINAMATH_CALUDE_existence_of_m_l2418_241883

theorem existence_of_m (a b : ℝ) (h : a > b) : ∃ m : ℝ, a * m < b * m := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_l2418_241883


namespace NUMINAMATH_CALUDE_intersection_point_correct_l2418_241855

/-- The line equation y = x + 3 -/
def line_equation (x y : ℝ) : Prop := y = x + 3

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the line y = x + 3 and the y-axis -/
def intersection_point : ℝ × ℝ := (0, 3)

theorem intersection_point_correct :
  let (x, y) := intersection_point
  line_equation x y ∧ on_y_axis x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l2418_241855


namespace NUMINAMATH_CALUDE_function_and_inequality_l2418_241856

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- Define the solution set condition
def solution_set (m : ℝ) : Set ℝ := {x | f m (x + 2) ≥ 0}

-- State the theorem
theorem function_and_inequality (m a b c : ℝ) : 
  (solution_set m = Set.Icc (-1) 1) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b + c = m) →
  (m = 1 ∧ a + 2*b + 3*c ≥ 9) := by
  sorry


end NUMINAMATH_CALUDE_function_and_inequality_l2418_241856


namespace NUMINAMATH_CALUDE_percentage_difference_l2418_241804

theorem percentage_difference : 
  (0.6 * 50 + 0.45 * 30) - (0.4 * 30 + 0.25 * 20) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2418_241804


namespace NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l2418_241889

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle of rotational symmetry for a regular polygon (in degrees) -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_17gon_symmetry_sum :
  ∀ (p : RegularPolygon 17),
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 649 / 17 := by
  sorry

end NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l2418_241889


namespace NUMINAMATH_CALUDE_cube_side_length_l2418_241899

-- Define the constants
def paint_cost_per_kg : ℝ := 60
def area_covered_per_kg : ℝ := 20
def total_paint_cost : ℝ := 1800
def num_cube_sides : ℕ := 6

-- Define the theorem
theorem cube_side_length :
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    side_length^2 * num_cube_sides * paint_cost_per_kg / area_covered_per_kg = total_paint_cost ∧
    side_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_l2418_241899


namespace NUMINAMATH_CALUDE_rides_second_day_l2418_241853

def rides_first_day : ℕ := 4
def total_rides : ℕ := 7

theorem rides_second_day : total_rides - rides_first_day = 3 := by
  sorry

end NUMINAMATH_CALUDE_rides_second_day_l2418_241853


namespace NUMINAMATH_CALUDE_fruits_left_l2418_241834

-- Define the initial quantities of fruits
def initial_bananas : ℕ := 12
def initial_apples : ℕ := 7
def initial_grapes : ℕ := 19

-- Define the quantities of fruits eaten
def eaten_bananas : ℕ := 4
def eaten_apples : ℕ := 2
def eaten_grapes : ℕ := 10

-- Define the function to calculate remaining fruits
def remaining_fruits : ℕ := 
  (initial_bananas - eaten_bananas) + 
  (initial_apples - eaten_apples) + 
  (initial_grapes - eaten_grapes)

-- Theorem statement
theorem fruits_left : remaining_fruits = 22 := by
  sorry

end NUMINAMATH_CALUDE_fruits_left_l2418_241834


namespace NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_or_q_l2418_241880

theorem not_p_or_q_false_implies_p_or_q (p q : Prop) :
  ¬(¬(p ∨ q)) → (p ∨ q) := by
sorry

end NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_or_q_l2418_241880


namespace NUMINAMATH_CALUDE_equation_equality_l2418_241819

theorem equation_equality : ∀ x y : ℝ, 9*x*y - 6*x*y = 3*x*y := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l2418_241819


namespace NUMINAMATH_CALUDE_point_C_complex_number_l2418_241805

/-- Given points A, B, and C in the complex plane, prove that C corresponds to 4-2i -/
theorem point_C_complex_number (A B C : ℂ) : 
  A = 2 + I →
  B - A = 1 + 2*I →
  C - B = 3 - I →
  C = 4 - 2*I := by sorry

end NUMINAMATH_CALUDE_point_C_complex_number_l2418_241805


namespace NUMINAMATH_CALUDE_orchestra_members_count_l2418_241864

theorem orchestra_members_count : ∃! n : ℕ,
  150 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 1 ∧
  n % 8 = 3 ∧
  n % 9 = 5 ∧
  n = 211 := by sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l2418_241864


namespace NUMINAMATH_CALUDE_reseating_women_circular_l2418_241858

-- Define the recurrence relation for reseating women
def R : ℕ → ℕ
  | 0 => 0  -- We define R(0) as 0 for completeness
  | 1 => 1
  | 2 => 2
  | (n + 3) => R (n + 2) + R (n + 1)

-- Theorem statement
theorem reseating_women_circular (n : ℕ) : R 15 = 987 := by
  sorry

-- You can also add additional lemmas to help prove the main theorem
lemma R_recurrence (n : ℕ) : n ≥ 3 → R n = R (n - 1) + R (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_reseating_women_circular_l2418_241858


namespace NUMINAMATH_CALUDE_sum_of_ages_at_milestone_l2418_241801

-- Define the ages of Hans, Josiah, and Julia
def hans_age : ℕ := 15
def josiah_age : ℕ := 3 * hans_age
def julia_age : ℕ := hans_age - 5

-- Define Julia's age when Hans was born
def julia_age_at_hans_birth : ℕ := julia_age / 2

-- Define Josiah's age when Julia was half her current age
def josiah_age_at_milestone : ℕ := josiah_age - hans_age - julia_age_at_hans_birth

-- Theorem statement
theorem sum_of_ages_at_milestone : 
  josiah_age_at_milestone + julia_age_at_hans_birth + 0 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_at_milestone_l2418_241801


namespace NUMINAMATH_CALUDE_second_class_males_count_l2418_241868

/-- Represents the number of students in each class by gender -/
structure ClassComposition where
  males : ℕ
  females : ℕ

/-- Represents the composition of the three classes -/
structure SquareDancingClasses where
  class1 : ClassComposition
  class2 : ClassComposition
  class3 : ClassComposition

def total_males (classes : SquareDancingClasses) : ℕ :=
  classes.class1.males + classes.class2.males + classes.class3.males

def total_females (classes : SquareDancingClasses) : ℕ :=
  classes.class1.females + classes.class2.females + classes.class3.females

theorem second_class_males_count 
  (classes : SquareDancingClasses)
  (h1 : classes.class1 = ⟨17, 13⟩)
  (h2 : classes.class2.females = 18)
  (h3 : classes.class3 = ⟨15, 17⟩)
  (h4 : total_males classes - total_females classes = 2) :
  classes.class2.males = 18 :=
sorry

end NUMINAMATH_CALUDE_second_class_males_count_l2418_241868


namespace NUMINAMATH_CALUDE_more_selected_in_B_l2418_241833

def total_candidates : ℕ := 8000
def selection_rate_A : ℚ := 6 / 100
def selection_rate_B : ℚ := 7 / 100

theorem more_selected_in_B : 
  ⌊(selection_rate_B * total_candidates : ℚ)⌋ - ⌊(selection_rate_A * total_candidates : ℚ)⌋ = 80 := by
  sorry

end NUMINAMATH_CALUDE_more_selected_in_B_l2418_241833


namespace NUMINAMATH_CALUDE_bacteria_growth_l2418_241857

/-- The time interval between bacterial divisions in minutes -/
def division_interval : ℕ := 20

/-- The total observation time in hours -/
def total_time : ℕ := 3

/-- The number of divisions that occur in the total observation time -/
def num_divisions : ℕ := (total_time * 60) / division_interval

/-- The final number of bacteria after the total observation time -/
def final_bacteria_count : ℕ := 2^num_divisions

theorem bacteria_growth :
  final_bacteria_count = 512 :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_l2418_241857


namespace NUMINAMATH_CALUDE_kylie_coins_left_l2418_241812

/-- The number of coins Kylie has after all transactions -/
def coins_left (piggy_bank : ℕ) (from_brother : ℕ) (from_father : ℕ) (given_to_friend : ℕ) : ℕ :=
  piggy_bank + from_brother + from_father - given_to_friend

/-- Theorem stating that Kylie is left with 15 coins -/
theorem kylie_coins_left : 
  coins_left 15 13 8 21 = 15 := by sorry

end NUMINAMATH_CALUDE_kylie_coins_left_l2418_241812


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2418_241840

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2418_241840


namespace NUMINAMATH_CALUDE_chenny_initial_candies_l2418_241893

/-- The number of friends Chenny has -/
def num_friends : ℕ := 7

/-- The number of candies each friend should receive -/
def candies_per_friend : ℕ := 2

/-- The number of additional candies Chenny needs to buy -/
def additional_candies : ℕ := 4

/-- Chenny's initial number of candies -/
def initial_candies : ℕ := num_friends * candies_per_friend - additional_candies

theorem chenny_initial_candies : initial_candies = 10 := by
  sorry

end NUMINAMATH_CALUDE_chenny_initial_candies_l2418_241893


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2418_241854

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2418_241854


namespace NUMINAMATH_CALUDE_total_apples_l2418_241811

def marin_apples : ℕ := 8

def david_apples (m : ℕ) : ℚ := (3 : ℚ) / 4 * m

def amanda_apples (d : ℚ) : ℚ := (3 : ℚ) / 2 * d + 2

theorem total_apples :
  let m := marin_apples
  let d := david_apples m
  let a := amanda_apples d
  ⌊m + d + a⌋ = 25 := by sorry

end NUMINAMATH_CALUDE_total_apples_l2418_241811


namespace NUMINAMATH_CALUDE_cookies_per_person_l2418_241886

theorem cookies_per_person (total_cookies : ℕ) (num_people : ℚ) 
  (h1 : total_cookies = 144) 
  (h2 : num_people = 6.0) : 
  (total_cookies : ℚ) / num_people = 24 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_person_l2418_241886


namespace NUMINAMATH_CALUDE_weight_plates_theorem_l2418_241824

/-- Calculates the effective weight of plates when lowered, considering technology and incline effects -/
def effectiveWeight (numPlates : ℕ) (plateWeight : ℝ) (techIncrease : ℝ) (inclineIncrease : ℝ) : ℝ :=
  let baseWeight := numPlates * plateWeight
  let withTech := baseWeight * (1 + techIncrease)
  withTech * (1 + inclineIncrease)

/-- Theorem: The effective weight of 10 plates of 30 pounds each, with 20% tech increase and 15% incline increase, is 414 pounds -/
theorem weight_plates_theorem :
  effectiveWeight 10 30 0.2 0.15 = 414 := by
  sorry


end NUMINAMATH_CALUDE_weight_plates_theorem_l2418_241824


namespace NUMINAMATH_CALUDE_factorization_difference_l2418_241810

theorem factorization_difference (y : ℝ) (a b : ℤ) : 
  (5 * y^2 + 17 * y + 6 = (5 * y + a) * (y + b)) → (a - b = -1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_l2418_241810


namespace NUMINAMATH_CALUDE_eighth_term_of_specific_geometric_sequence_l2418_241845

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

-- Theorem statement
theorem eighth_term_of_specific_geometric_sequence :
  geometric_sequence 8 2 8 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_specific_geometric_sequence_l2418_241845


namespace NUMINAMATH_CALUDE_smallest_positive_e_value_l2418_241826

theorem smallest_positive_e_value (a b c d e : ℤ) :
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = -3 ∨ x = 4 ∨ x = 8 ∨ x = -1/4) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 → 
    (∃ a' b' c' d' : ℤ, ∀ x : ℚ, a' * x^4 + b' * x^3 + c' * x^2 + d' * x + e' = 0 ↔ 
      x = -3 ∨ x = 4 ∨ x = 8 ∨ x = -1/4) →
    e ≤ e') →
  e = 96 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_e_value_l2418_241826


namespace NUMINAMATH_CALUDE_paper_fold_crease_length_l2418_241808

theorem paper_fold_crease_length :
  ∀ (width : ℝ) (angle : ℝ),
  width = 8 →
  angle = π / 4 →
  ∃ (crease_length : ℝ),
  crease_length = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_paper_fold_crease_length_l2418_241808


namespace NUMINAMATH_CALUDE_percentage_non_mutated_frogs_l2418_241839

def total_frogs : ℕ := 250
def extra_legs : ℕ := 32
def two_heads : ℕ := 21
def bright_red : ℕ := 16
def skin_abnormalities : ℕ := 12
def extra_eyes : ℕ := 7

theorem percentage_non_mutated_frogs :
  let mutated_frogs := extra_legs + two_heads + bright_red + skin_abnormalities + extra_eyes
  let non_mutated_frogs := total_frogs - mutated_frogs
  (non_mutated_frogs : ℚ) / total_frogs * 100 = 648 / 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_non_mutated_frogs_l2418_241839


namespace NUMINAMATH_CALUDE_curve_intersects_all_planes_l2418_241831

/-- A smooth curve in ℝ³ -/
def C : ℝ → ℝ × ℝ × ℝ := fun t ↦ (t, t^3, t^5)

/-- Definition of a plane in ℝ³ -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  not_all_zero : A ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0

/-- The theorem stating that the curve C intersects every plane -/
theorem curve_intersects_all_planes :
  ∀ (p : Plane), ∃ (t : ℝ), 
    let (x, y, z) := C t
    p.A * x + p.B * y + p.C * z + p.D = 0 := by
  sorry


end NUMINAMATH_CALUDE_curve_intersects_all_planes_l2418_241831


namespace NUMINAMATH_CALUDE_sharon_wants_254_supplies_l2418_241877

/-- Calculates the total number of kitchen supplies Sharon wants to buy -/
def sharons_kitchen_supplies (angela_pots : ℕ) : ℕ :=
  let angela_plates := 6 + 3 * angela_pots
  let angela_cutlery := angela_plates / 2
  let sharon_pots := angela_pots / 2
  let sharon_plates := 3 * angela_plates - 20
  let sharon_cutlery := 2 * angela_cutlery
  sharon_pots + sharon_plates + sharon_cutlery

/-- Theorem stating that Sharon wants to buy 254 kitchen supplies -/
theorem sharon_wants_254_supplies : sharons_kitchen_supplies 20 = 254 := by
  sorry

end NUMINAMATH_CALUDE_sharon_wants_254_supplies_l2418_241877


namespace NUMINAMATH_CALUDE_negation_equivalence_l2418_241820

theorem negation_equivalence (x y : ℝ) : 
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 = 0 → x ≠ 0 ∨ y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2418_241820


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2418_241841

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  -- Given conditions
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A ∧
  b = 3 ∧
  c = 2 →
  -- Conclusions
  A = π / 3 ∧ a = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2418_241841
