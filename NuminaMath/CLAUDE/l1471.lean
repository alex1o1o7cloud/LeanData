import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_product_range_l1471_147140

theorem geometric_sequence_product_range (a₁ a₂ a₃ m q : ℝ) (hm : m > 0) (hq : q > 0) :
  (a₁ + a₂ + a₃ = 3 * m) →
  (a₂ = a₁ * q) →
  (a₃ = a₂ * q) →
  let t := a₁ * a₂ * a₃
  0 < t ∧ t ≤ m^3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_range_l1471_147140


namespace NUMINAMATH_CALUDE_alternative_interest_rate_l1471_147159

/-- Proves that given a principal of 4200 invested for 2 years, if the interest at 18% p.a. 
    is 504 more than the interest at an unknown rate p, then p is equal to 12. -/
theorem alternative_interest_rate (principal : ℝ) (time : ℝ) (known_rate : ℝ) (difference : ℝ) (p : ℝ) : 
  principal = 4200 →
  time = 2 →
  known_rate = 18 →
  difference = 504 →
  principal * known_rate / 100 * time - principal * p / 100 * time = difference →
  p = 12 := by
  sorry

#check alternative_interest_rate

end NUMINAMATH_CALUDE_alternative_interest_rate_l1471_147159


namespace NUMINAMATH_CALUDE_small_circle_radius_l1471_147161

/-- Given two circles where the radius of the larger circle is 80 cm and 4 times
    the radius of the smaller circle, prove that the radius of the smaller circle is 20 cm. -/
theorem small_circle_radius (r : ℝ) : 
  r > 0 → 4 * r = 80 → r = 20 := by
  sorry

end NUMINAMATH_CALUDE_small_circle_radius_l1471_147161


namespace NUMINAMATH_CALUDE_solve_for_A_l1471_147182

theorem solve_for_A (x : ℝ) (A : ℝ) (h : (5 : ℝ) / (x + 1) = A - ((2 * x - 3) / (x + 1))) :
  A = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l1471_147182


namespace NUMINAMATH_CALUDE_class_size_problem_l1471_147117

/-- Given information about class sizes, prove the size of Class C -/
theorem class_size_problem (class_b : ℕ) (h1 : class_b = 20) : ∃ (class_c : ℕ), class_c = 170 ∧
  ∃ (class_a class_d : ℕ),
    class_a = 2 * class_b ∧
    3 * class_a = class_c ∧
    class_d = 4 * class_a ∧
    class_c = class_d + 10 := by
  sorry

end NUMINAMATH_CALUDE_class_size_problem_l1471_147117


namespace NUMINAMATH_CALUDE_unique_n_l1471_147100

def is_valid_n (n : ℕ+) : Prop :=
  ∃ (k : ℕ) (d : ℕ → ℕ+),
    k ≥ 6 ∧
    (∀ i ≤ k, d i ∣ n) ∧
    (∀ i j, i < j → d i < d j) ∧
    d 1 = 1 ∧
    d k = n ∧
    n = (d 5)^2 + (d 6)^2

theorem unique_n : ∀ n : ℕ+, is_valid_n n → n = 500 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_l1471_147100


namespace NUMINAMATH_CALUDE_cloth_worth_calculation_l1471_147118

/-- Represents the commission rate as a percentage -/
def commission_rate : ℚ := 25/10

/-- Represents the commission earned in rupees -/
def commission_earned : ℚ := 15

/-- Represents the worth of cloth sold -/
def cloth_worth : ℚ := 600

/-- Theorem stating that given the commission rate and earned commission, 
    the worth of cloth sold is 600 rupees -/
theorem cloth_worth_calculation : 
  commission_earned = (commission_rate / 100) * cloth_worth :=
sorry

end NUMINAMATH_CALUDE_cloth_worth_calculation_l1471_147118


namespace NUMINAMATH_CALUDE_parallelogram_above_x_axis_ratio_l1471_147171

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four points -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculates the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of the part of the parallelogram above the x-axis -/
def areaAboveXAxis (p : Parallelogram) : ℝ := sorry

/-- The main theorem to be proved -/
theorem parallelogram_above_x_axis_ratio 
  (p : Parallelogram) 
  (h1 : p.P = ⟨-1, 1⟩) 
  (h2 : p.Q = ⟨3, -5⟩) 
  (h3 : p.R = ⟨1, -3⟩) 
  (h4 : p.S = ⟨-3, 3⟩) : 
  areaAboveXAxis p / parallelogramArea p = 1/4 := by sorry

end NUMINAMATH_CALUDE_parallelogram_above_x_axis_ratio_l1471_147171


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l1471_147198

theorem complex_magnitude_equation (x : ℝ) :
  x > 0 ∧ Complex.abs (3 + x * Complex.I) = 7 ↔ x = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l1471_147198


namespace NUMINAMATH_CALUDE_league_teams_count_l1471_147173

/-- The number of games in a league where each team plays every other team once -/
def numGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league where each team plays every other team exactly once, 
    if the total number of games played is 36, then the number of teams in the league is 9 -/
theorem league_teams_count : ∃ (n : ℕ), n > 0 ∧ numGames n = 36 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_league_teams_count_l1471_147173


namespace NUMINAMATH_CALUDE_experiments_to_target_reduction_l1471_147156

/-- The factor by which the range is reduced after each experiment -/
def reduction_factor : ℝ := 0.618

/-- The target reduction of the range -/
def target_reduction : ℝ := 0.618^4

/-- The number of experiments needed to reach the target reduction -/
def num_experiments : ℕ := 4

/-- Theorem stating that the number of experiments needed to reach the target reduction is correct -/
theorem experiments_to_target_reduction :
  (reduction_factor ^ num_experiments) = target_reduction :=
by sorry

end NUMINAMATH_CALUDE_experiments_to_target_reduction_l1471_147156


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l1471_147116

theorem fraction_sum_inequality (a b c : ℝ) (h : a * b * c = 1) :
  (1 / (2 * a^2 + b^2 + 3)) + (1 / (2 * b^2 + c^2 + 3)) + (1 / (2 * c^2 + a^2 + 3)) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l1471_147116


namespace NUMINAMATH_CALUDE_existence_of_special_divisor_l1471_147191

theorem existence_of_special_divisor (n k : ℕ) (h1 : n > 1) (h2 : k = (Nat.factors n).card) :
  ∃ a : ℕ, 1 < a ∧ a < n / k + 1 ∧ n ∣ (a^2 - a) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_divisor_l1471_147191


namespace NUMINAMATH_CALUDE_count_distinct_tetrahedrons_l1471_147115

/-- The number of distinct tetrahedrons that can be painted with n colors, 
    where each face is painted with exactly one color. -/
def distinctTetrahedrons (n : ℕ) : ℕ :=
  n * ((n-1)*(n-2)*(n-3)/12 + (n-1)*(n-2)/3 + (n-1) + 1)

/-- Theorem stating the number of distinct tetrahedrons that can be painted 
    with n colors, where n ≥ 4 and each face is painted with exactly one color. -/
theorem count_distinct_tetrahedrons (n : ℕ) (h : n ≥ 4) : 
  distinctTetrahedrons n = n * ((n-1)*(n-2)*(n-3)/12 + (n-1)*(n-2)/3 + (n-1) + 1) :=
by
  sorry

#check count_distinct_tetrahedrons

end NUMINAMATH_CALUDE_count_distinct_tetrahedrons_l1471_147115


namespace NUMINAMATH_CALUDE_paco_cookies_l1471_147146

theorem paco_cookies (cookies_eaten : ℕ) (cookies_given : ℕ) : 
  cookies_eaten = 14 →
  cookies_given = 13 →
  cookies_eaten = cookies_given + 1 →
  cookies_eaten + cookies_given = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l1471_147146


namespace NUMINAMATH_CALUDE_set_equality_proof_l1471_147122

def U : Set Nat := {0, 1, 2}

theorem set_equality_proof (A : Set Nat) (h : (U \ A) = {2}) : A = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_proof_l1471_147122


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l1471_147193

/-- Proves that if a deposit of 5000 is 20% of a person's monthly income, then their monthly income is 25000. -/
theorem monthly_income_calculation (deposit : ℝ) (percentage : ℝ) (monthly_income : ℝ) 
  (h1 : deposit = 5000)
  (h2 : percentage = 20)
  (h3 : deposit = (percentage / 100) * monthly_income) :
  monthly_income = 25000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l1471_147193


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l1471_147107

theorem rectangle_area_perimeter_sum (a b : ℕ+) :
  ∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 114 ∧
  ∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 116 ∧
  ∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 120 ∧
  ∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 122 ∧
  ¬∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 118 :=
by sorry


end NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l1471_147107


namespace NUMINAMATH_CALUDE_triangle_side_length_l1471_147190

theorem triangle_side_length (A B C : ℝ) (AB BC AC : ℝ) : 
  A = π / 3 →
  Real.tan B = 1 / 2 →
  AB = 2 * Real.sqrt 3 + 1 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  AB > 0 ∧ BC > 0 ∧ AC > 0 →
  AC / Real.sin B = AB / Real.sin C →
  BC / Real.sin A = AB / Real.sin C →
  BC = Real.sqrt 15 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1471_147190


namespace NUMINAMATH_CALUDE_polynomial_sum_independence_l1471_147192

theorem polynomial_sum_independence (a b : ℝ) :
  (∀ x y : ℝ, (x^2 + a*x - y + b) + (b*x^2 - 3*x + 6*y - 3) = (5*y + b - 3)) →
  3*(a^2 - 2*a*b + b^2) - (4*a^2 - 2*(1/2*a^2 + a*b - 3/2*b^2)) = 12 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_independence_l1471_147192


namespace NUMINAMATH_CALUDE_first_divisor_of_square_plus_164_l1471_147172

theorem first_divisor_of_square_plus_164 : 
  ∀ n ∈ [3, 4, 5, 6, 7, 8, 9, 10, 11], 
    (n ∣ (166^2 + 164)) → 
    n = 3 := by sorry

end NUMINAMATH_CALUDE_first_divisor_of_square_plus_164_l1471_147172


namespace NUMINAMATH_CALUDE_binomial_p_value_l1471_147180

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

variable (X : BinomialDistribution)

/-- The expected value of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a binomial distribution with E(X) = 4 and D(X) = 3, p = 1/4 -/
theorem binomial_p_value (X : BinomialDistribution) 
  (h2 : expectation X = 4) 
  (h3 : variance X = 3) : 
  X.p = 1/4 := by sorry

end NUMINAMATH_CALUDE_binomial_p_value_l1471_147180


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1471_147133

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h : IsGeometric a) 
    (h3 : a 3 = 3) (h10 : a 10 = 384) : 
  ∀ n : ℕ, a n = 3 * 2^(n - 3) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1471_147133


namespace NUMINAMATH_CALUDE_tigers_home_games_l1471_147110

/-- The number of home games played by the Tigers -/
def total_home_games (losses ties wins : ℕ) : ℕ := losses + ties + wins

/-- The number of losses in Tiger's home games -/
def losses : ℕ := 12

/-- The number of ties in Tiger's home games -/
def ties : ℕ := losses / 2

/-- The number of wins in Tiger's home games -/
def wins : ℕ := 38

theorem tigers_home_games : total_home_games losses ties wins = 56 := by
  sorry

end NUMINAMATH_CALUDE_tigers_home_games_l1471_147110


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1471_147175

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1471_147175


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1471_147119

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | (x - 1)^2 < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1471_147119


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l1471_147176

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) := by sorry

theorem negation_of_inequality :
  (¬ ∀ x : ℝ, Real.exp x - x - 1 > 0) ↔ (∃ x : ℝ, Real.exp x - x - 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l1471_147176


namespace NUMINAMATH_CALUDE_cafeteria_shirt_ratio_l1471_147113

/-- Proves that the ratio of people wearing horizontal stripes to people wearing checkered shirts is 4:1 --/
theorem cafeteria_shirt_ratio 
  (total_people : ℕ) 
  (checkered_shirts : ℕ) 
  (vertical_stripes : ℕ) 
  (h1 : total_people = 40)
  (h2 : checkered_shirts = 7)
  (h3 : vertical_stripes = 5) :
  (total_people - checkered_shirts - vertical_stripes) / checkered_shirts = 4 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_shirt_ratio_l1471_147113


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1471_147129

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1)))
  (h2 : ∀ n, a (n + 1) = a n * (a 2 / a 1))
  (h3 : S 3 = 15)
  (h4 : a 3 = 5) :
  (a 2 / a 1 = -1/2) ∨ (a 2 / a 1 = 1) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1471_147129


namespace NUMINAMATH_CALUDE_examination_statements_l1471_147120

/-- Represents a statistical population -/
structure Population where
  size : ℕ

/-- Represents a sample from a population -/
structure Sample (pop : Population) where
  size : ℕ
  h_size_le : size ≤ pop.size

/-- The given examination scenario -/
def examination_scenario : Prop :=
  ∃ (pop : Population) (sample : Sample pop),
    pop.size = 70000 ∧
    sample.size = 1000 ∧
    (sample.size = 1000 → sample.size = 1000) ∧
    (pop.size = 70000 → pop.size = 70000)

/-- The statements to be proved -/
theorem examination_statements (h : examination_scenario) :
  ∃ (pop : Population) (sample : Sample pop),
    pop.size = 70000 ∧
    sample.size = 1000 ∧
    (Sample pop → True) ∧  -- Statement 1
    (pop.size = 70000 → True) ∧  -- Statement 3
    (sample.size = 1000 → True)  -- Statement 4
    := by sorry

end NUMINAMATH_CALUDE_examination_statements_l1471_147120


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_seventh_term_l1471_147109

/-- Given that the binomial coefficient of the 7th term in the expansion of (a+b)^n is the largest, prove that n = 12. -/
theorem largest_binomial_coefficient_seventh_term (n : ℕ) : 
  (∀ k : ℕ, k ≤ n → Nat.choose n k ≤ Nat.choose n 6) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_binomial_coefficient_seventh_term_l1471_147109


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1471_147108

theorem min_value_trig_expression (x : Real) (h : 0 < x ∧ x < π / 2) :
  (8 / Real.sin x) + (1 / Real.cos x) ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1471_147108


namespace NUMINAMATH_CALUDE_factorization_identity_l1471_147131

theorem factorization_identity (x y : ℝ) : x^2 - 2*x*y + y^2 - 1 = (x - y + 1) * (x - y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identity_l1471_147131


namespace NUMINAMATH_CALUDE_cubic_fraction_simplification_l1471_147124

theorem cubic_fraction_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a + b + 2 * c = 0) : 
  (a^3 + b^3 + 2 * c^3) / (a * b * c) = -3 * (a^2 - a * b + b^2) / (2 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_simplification_l1471_147124


namespace NUMINAMATH_CALUDE_integral_f_equals_five_sixths_l1471_147169

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then x^2
  else if 1 < x ∧ x ≤ 2 then 2 - x
  else 0  -- Define a value for x outside the given ranges

theorem integral_f_equals_five_sixths :
  ∫ x in (0)..(2), f x = 5/6 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_five_sixths_l1471_147169


namespace NUMINAMATH_CALUDE_fraction_calculation_l1471_147101

theorem fraction_calculation : 
  (1 / 5 + 1 / 7) / (3 / 8 - 1 / 9) = 864 / 665 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1471_147101


namespace NUMINAMATH_CALUDE_four_tire_repair_cost_l1471_147121

/-- The total cost for repairing a given number of tires -/
def total_cost (repair_cost : ℚ) (sales_tax : ℚ) (num_tires : ℕ) : ℚ :=
  (repair_cost + sales_tax) * num_tires

/-- Theorem: The total cost for repairing 4 tires is $30 -/
theorem four_tire_repair_cost :
  total_cost 7 0.5 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_four_tire_repair_cost_l1471_147121


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l1471_147189

/-- The function f(x) = x³ + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ f' x = 4 ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) := by
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l1471_147189


namespace NUMINAMATH_CALUDE_annie_village_trick_or_treat_l1471_147149

/-- The number of blocks in Annie's village -/
def num_blocks : ℕ := 9

/-- The number of children on each block -/
def children_per_block : ℕ := 6

/-- The total number of children going trick or treating in Annie's village -/
def total_children : ℕ := num_blocks * children_per_block

theorem annie_village_trick_or_treat : total_children = 54 := by
  sorry

end NUMINAMATH_CALUDE_annie_village_trick_or_treat_l1471_147149


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1471_147184

theorem intersection_of_sets : 
  let A : Set ℕ := {x | ∃ n, x = 2 * n}
  let B : Set ℕ := {x | ∃ n, x = 3 * n}
  let C : Set ℕ := {x | ∃ n, x = n * n}
  A ∩ B ∩ C = {x | ∃ n, x = 36 * n * n} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1471_147184


namespace NUMINAMATH_CALUDE_real_square_properties_l1471_147135

theorem real_square_properties (a b : ℝ) : 
  (a^2 ≠ b^2 → a ≠ b) ∧ (a > |b| → a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_real_square_properties_l1471_147135


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l1471_147157

theorem unique_solution_power_equation :
  ∃! (n k l m : ℕ), l > 1 ∧ (1 + n^k)^l = 1 + n^m ∧ n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l1471_147157


namespace NUMINAMATH_CALUDE_sphere_water_volume_calculation_l1471_147134

/-- The volume of water in a sphere container that can be transferred to 
    a given number of hemisphere containers of a specific volume. -/
def sphere_water_volume (num_hemispheres : ℕ) (hemisphere_volume : ℝ) : ℝ :=
  (num_hemispheres : ℝ) * hemisphere_volume

/-- Theorem stating the volume of water in a sphere container
    given the number of hemisphere containers and their volume. -/
theorem sphere_water_volume_calculation :
  sphere_water_volume 2744 4 = 10976 := by
  sorry

end NUMINAMATH_CALUDE_sphere_water_volume_calculation_l1471_147134


namespace NUMINAMATH_CALUDE_batch_size_proof_l1471_147114

theorem batch_size_proof (n : ℕ) : 
  500 ≤ n ∧ n ≤ 600 ∧ 
  n % 20 = 13 ∧ 
  n % 27 = 20 → 
  n = 533 :=
sorry

end NUMINAMATH_CALUDE_batch_size_proof_l1471_147114


namespace NUMINAMATH_CALUDE_power_equality_l1471_147155

theorem power_equality (x y : ℕ) (h1 : x - y = 12) (h2 : x = 12) :
  3^x * 4^y = 531441 := by
sorry

end NUMINAMATH_CALUDE_power_equality_l1471_147155


namespace NUMINAMATH_CALUDE_unfactorable_quartic_l1471_147152

theorem unfactorable_quartic : ¬∃ (a b c d : ℤ), ∀ (x : ℝ), 
  x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end NUMINAMATH_CALUDE_unfactorable_quartic_l1471_147152


namespace NUMINAMATH_CALUDE_parabola_parameter_l1471_147183

/-- Given a circle C₁ and a parabola C₂ intersecting at two points with a specific chord length,
    prove that the parameter of the parabola has a specific value. -/
theorem parabola_parameter (p : ℝ) (h_p : p > 0) : 
  ∃ A B : ℝ × ℝ,
    (A.1^2 + (A.2 - 2)^2 = 4) ∧ 
    (B.1^2 + (B.2 - 2)^2 = 4) ∧
    (A.2^2 = 2*p*A.1) ∧ 
    (B.2^2 = 2*p*B.1) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (8*Real.sqrt 5/5)^2) →
    p = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_parameter_l1471_147183


namespace NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_conditions_l1471_147188

/-- The analysis method in mathematical proofs --/
structure AnalysisMethod where
  conclusion : Prop
  seek_conditions : Prop → Prop

/-- Definition of sufficient conditions --/
def sufficient_conditions (am : AnalysisMethod) (conditions : Prop) : Prop :=
  conditions → am.conclusion

/-- Theorem stating that the analysis method seeks sufficient conditions --/
theorem analysis_method_seeks_sufficient_conditions (am : AnalysisMethod) :
  ∃ (conditions : Prop), sufficient_conditions am conditions :=
sorry

end NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_conditions_l1471_147188


namespace NUMINAMATH_CALUDE_stamp_selection_l1471_147143

theorem stamp_selection (n k : ℕ) (stamps : Finset ℕ) : 
  0 < n → 
  stamps.card = k → 
  n ≤ stamps.sum id → 
  stamps.sum id < 2 * k → 
  ∃ s : Finset ℕ, s ⊆ stamps ∧ s.sum id = n := by
  sorry

#check stamp_selection

end NUMINAMATH_CALUDE_stamp_selection_l1471_147143


namespace NUMINAMATH_CALUDE_intersection_probability_correct_l1471_147178

/-- Given a positive integer n, this function calculates the probability that
    the intersection of two randomly selected non-empty subsets from {1, 2, ..., n}
    is not empty. -/
def intersection_probability (n : ℕ+) : ℚ :=
  (4^n.val - 3^n.val : ℚ) / (2^n.val - 1)^2

/-- Theorem stating that the probability of non-empty intersection of two randomly
    selected non-empty subsets from {1, 2, ..., n} is given by the function
    intersection_probability. -/
theorem intersection_probability_correct (n : ℕ+) :
  intersection_probability n =
    (4^n.val - 3^n.val : ℚ) / (2^n.val - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_probability_correct_l1471_147178


namespace NUMINAMATH_CALUDE_square_sum_equals_sixteen_l1471_147137

theorem square_sum_equals_sixteen (x y : ℝ) 
  (h1 : (x + y)^2 = 36) 
  (h2 : x * y = 10) : 
  x^2 + y^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_sixteen_l1471_147137


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_l1471_147160

/-- Two lines in a 2D plane -/
structure TwoLines where
  line1 : ℝ → ℝ → ℝ
  line2 : ℝ → ℝ → ℝ

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (-4, 3)

/-- The given two lines -/
def given_lines : TwoLines where
  line1 := fun x y => 3 * x + 2 * y + 6
  line2 := fun x y => 2 * x + 5 * y - 7

theorem intersection_point_satisfies_equations : 
  let (x, y) := intersection_point
  given_lines.line1 x y = 0 ∧ given_lines.line2 x y = 0 := by
  sorry

#check intersection_point_satisfies_equations

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_l1471_147160


namespace NUMINAMATH_CALUDE_correct_outfit_assignment_l1471_147104

-- Define the colors
inductive Color
  | White
  | Red
  | Blue

-- Define a person's outfit
structure Outfit :=
  (dress : Color)
  (shoes : Color)

-- Define the friends
inductive Friend
  | Nadya
  | Valya
  | Masha

def outfit_assignment : Friend → Outfit
  | Friend.Nadya => { dress := Color.Blue, shoes := Color.Blue }
  | Friend.Valya => { dress := Color.Red, shoes := Color.White }
  | Friend.Masha => { dress := Color.White, shoes := Color.Red }

theorem correct_outfit_assignment :
  -- Nadya's shoes match her dress
  (outfit_assignment Friend.Nadya).dress = (outfit_assignment Friend.Nadya).shoes ∧
  -- Valya's dress and shoes are not blue
  (outfit_assignment Friend.Valya).dress ≠ Color.Blue ∧
  (outfit_assignment Friend.Valya).shoes ≠ Color.Blue ∧
  -- Masha wears red shoes
  (outfit_assignment Friend.Masha).shoes = Color.Red ∧
  -- All dresses are different colors
  (outfit_assignment Friend.Nadya).dress ≠ (outfit_assignment Friend.Valya).dress ∧
  (outfit_assignment Friend.Nadya).dress ≠ (outfit_assignment Friend.Masha).dress ∧
  (outfit_assignment Friend.Valya).dress ≠ (outfit_assignment Friend.Masha).dress ∧
  -- All shoes are different colors
  (outfit_assignment Friend.Nadya).shoes ≠ (outfit_assignment Friend.Valya).shoes ∧
  (outfit_assignment Friend.Nadya).shoes ≠ (outfit_assignment Friend.Masha).shoes ∧
  (outfit_assignment Friend.Valya).shoes ≠ (outfit_assignment Friend.Masha).shoes := by
  sorry

end NUMINAMATH_CALUDE_correct_outfit_assignment_l1471_147104


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1471_147177

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_sum1 : a 1 + a 2 = 4/9)
  (h_sum2 : a 3 + a 4 + a 5 + a 6 = 40) :
  (a 7 + a 8 + a 9) / 9 = 117 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1471_147177


namespace NUMINAMATH_CALUDE_quadratic_real_roots_implies_m_leq_3_l1471_147158

theorem quadratic_real_roots_implies_m_leq_3 (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_implies_m_leq_3_l1471_147158


namespace NUMINAMATH_CALUDE_container_max_volume_l1471_147144

theorem container_max_volume :
  let total_length : ℝ := 24
  let volume (x : ℝ) : ℝ := x^2 * (total_length / 4 - x / 2)
  ∀ x > 0, x < total_length / 4 → volume x ≤ 8 ∧
  ∃ x > 0, x < total_length / 4 ∧ volume x = 8 :=
by sorry

end NUMINAMATH_CALUDE_container_max_volume_l1471_147144


namespace NUMINAMATH_CALUDE_area_BPQ_is_six_l1471_147126

/-- Rectangle ABCD with length 8 and width 6, diagonal AC divided into 4 equal segments by P, Q, R -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (diagonal_segments : ℕ)

/-- The area of triangle BPQ in the given rectangle -/
def area_BPQ (rect : Rectangle) : ℝ :=
  sorry

/-- Theorem stating that the area of triangle BPQ is 6 square inches -/
theorem area_BPQ_is_six (rect : Rectangle) 
  (h1 : rect.length = 8)
  (h2 : rect.width = 6)
  (h3 : rect.diagonal_segments = 4) : 
  area_BPQ rect = 6 :=
sorry

end NUMINAMATH_CALUDE_area_BPQ_is_six_l1471_147126


namespace NUMINAMATH_CALUDE_matrix_equality_l1471_147154

theorem matrix_equality (X Y : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : X + Y = X * Y)
  (h2 : X * Y = ![![16/3, 2], ![-10/3, 10/3]]) :
  Y * X = ![![16/3, 2], ![-10/3, 10/3]] := by sorry

end NUMINAMATH_CALUDE_matrix_equality_l1471_147154


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1471_147145

theorem solve_exponential_equation :
  ∃! x : ℝ, (64 : ℝ)^(3*x) = (16 : ℝ)^(4*x - 5) :=
by
  use -10
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1471_147145


namespace NUMINAMATH_CALUDE_equation_solution_l1471_147139

theorem equation_solution (y : ℝ) : 
  y = (13/2)^4 ↔ 3 * y^(1/4) - (3 * y^(1/2)) / y^(1/4) = 13 - 2 * y^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1471_147139


namespace NUMINAMATH_CALUDE_min_tiles_to_pave_courtyard_l1471_147127

def courtyard_length : ℕ := 378
def courtyard_width : ℕ := 525

def tile_side_length : ℕ := Nat.gcd courtyard_length courtyard_width

def courtyard_area : ℕ := courtyard_length * courtyard_width
def tile_area : ℕ := tile_side_length * tile_side_length

def number_of_tiles : ℕ := courtyard_area / tile_area

theorem min_tiles_to_pave_courtyard :
  number_of_tiles = 450 := by sorry

end NUMINAMATH_CALUDE_min_tiles_to_pave_courtyard_l1471_147127


namespace NUMINAMATH_CALUDE_log_125_equals_3_minus_3log2_l1471_147179

theorem log_125_equals_3_minus_3log2 : Real.log 125 = 3 - 3 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_125_equals_3_minus_3log2_l1471_147179


namespace NUMINAMATH_CALUDE_car_distance_proof_l1471_147132

/-- Proves that the distance covered by a car is 540 kilometers under given conditions -/
theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) :
  initial_time = 8 →
  speed = 45 →
  (3 / 2 : ℝ) * initial_time * speed = 540 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l1471_147132


namespace NUMINAMATH_CALUDE_white_balls_count_l1471_147195

theorem white_balls_count (total : ℕ) (p_red p_black : ℚ) (h_total : total = 50)
  (h_red : p_red = 15/100) (h_black : p_black = 45/100) :
  (total : ℚ) * (1 - p_red - p_black) = 20 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l1471_147195


namespace NUMINAMATH_CALUDE_solid_is_cone_l1471_147174

-- Define the properties of the solid
structure Solid where
  front_view_isosceles : Bool
  left_view_isosceles : Bool
  top_view_circle_with_center : Bool

-- Define what it means for a solid to be a cone
def is_cone (s : Solid) : Prop :=
  s.front_view_isosceles ∧ s.left_view_isosceles ∧ s.top_view_circle_with_center

-- Theorem statement
theorem solid_is_cone (s : Solid) 
  (h1 : s.front_view_isosceles = true) 
  (h2 : s.left_view_isosceles = true) 
  (h3 : s.top_view_circle_with_center = true) : 
  is_cone s := by sorry

end NUMINAMATH_CALUDE_solid_is_cone_l1471_147174


namespace NUMINAMATH_CALUDE_arrangements_count_l1471_147106

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 5

/-- Represents the number of venues -/
def num_venues : ℕ := 4

/-- Represents the condition that A is assigned to the badminton venue -/
def a_assigned_to_badminton : Prop := true

/-- Represents the condition that each volunteer goes to only one venue -/
def one_venue_per_volunteer : Prop := true

/-- Represents the condition that each venue has at least one volunteer -/
def at_least_one_volunteer_per_venue : Prop := true

/-- The total number of different arrangements -/
def total_arrangements : ℕ := 60

/-- Theorem stating that the number of arrangements is 60 -/
theorem arrangements_count :
  num_volunteers = 5 ∧
  num_venues = 4 ∧
  a_assigned_to_badminton ∧
  one_venue_per_volunteer ∧
  at_least_one_volunteer_per_venue →
  total_arrangements = 60 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l1471_147106


namespace NUMINAMATH_CALUDE_union_of_sets_l1471_147163

theorem union_of_sets : 
  let A : Set ℕ := {1,2,3,4}
  let B : Set ℕ := {2,4,5}
  A ∪ B = {1,2,3,4,5} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l1471_147163


namespace NUMINAMATH_CALUDE_juan_number_puzzle_l1471_147130

theorem juan_number_puzzle (n : ℝ) : ((n + 3) * 3 - 3) * 2 / 3 = 10 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_juan_number_puzzle_l1471_147130


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l1471_147181

theorem smallest_b_for_factorization : ∃ (b : ℕ), 
  (∀ (r s : ℕ), r > 0 ∧ s > 0 ∧ 8 ∣ s ∧ (∀ x : ℤ, x^2 + b*x + 2016 = (x+r)*(x+s)) → b ≥ 260) ∧
  (∃ (r s : ℕ), r > 0 ∧ s > 0 ∧ 8 ∣ s ∧ (∀ x : ℤ, x^2 + 260*x + 2016 = (x+r)*(x+s))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l1471_147181


namespace NUMINAMATH_CALUDE_randy_quiz_average_l1471_147197

/-- The number of quizzes Randy wants to have the average for -/
def n : ℕ := 5

/-- The sum of Randy's first four quiz scores -/
def initial_sum : ℕ := 374

/-- Randy's desired average -/
def desired_average : ℕ := 94

/-- Randy's next quiz score -/
def next_score : ℕ := 96

theorem randy_quiz_average : 
  (initial_sum + next_score : ℚ) / n = desired_average := by sorry

end NUMINAMATH_CALUDE_randy_quiz_average_l1471_147197


namespace NUMINAMATH_CALUDE_wire_length_between_poles_l1471_147162

/-- Given two vertical poles on flat ground with a distance of 20 feet between their bases
    and a height difference of 10 feet, the length of a wire stretched between their tops
    is 10√5 feet. -/
theorem wire_length_between_poles (distance : ℝ) (height_diff : ℝ) :
  distance = 20 → height_diff = 10 → 
  ∃ (wire_length : ℝ), wire_length = 10 * Real.sqrt 5 ∧ 
  wire_length ^ 2 = distance ^ 2 + height_diff ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_wire_length_between_poles_l1471_147162


namespace NUMINAMATH_CALUDE_local_extremum_properties_l1471_147187

/-- A function with a local extremum -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - 1

/-- The derivative of f -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem local_extremum_properties (a b : ℝ) :
  (f' a b (-1) = 0 ∧ f a b (-1) = 4) →
  (a = -3 ∧ b = -9 ∧
   ∀ x ∈ Set.Icc (0 : ℝ) 4, f (-3) (-9) x ≤ -1 ∧
   ∃ y ∈ Set.Icc (0 : ℝ) 4, f (-3) (-9) y = -28 ∧
   ∀ z ∈ Set.Icc (0 : ℝ) 4, f (-3) (-9) z ≥ -28) := by sorry

end NUMINAMATH_CALUDE_local_extremum_properties_l1471_147187


namespace NUMINAMATH_CALUDE_jackson_metropolitan_population_l1471_147102

theorem jackson_metropolitan_population :
  ∀ (average_population : ℝ),
  3200 ≤ average_population ∧ average_population ≤ 3600 →
  80000 ≤ 25 * average_population ∧ 25 * average_population ≤ 90000 :=
by sorry

end NUMINAMATH_CALUDE_jackson_metropolitan_population_l1471_147102


namespace NUMINAMATH_CALUDE_sum_of_digits_of_3_to_17_l1471_147147

/-- The sum of the tens digit and the ones digit of (7-4)^17 is 9. -/
theorem sum_of_digits_of_3_to_17 : 
  (((7 - 4)^17 / 10) % 10 + (7 - 4)^17 % 10) = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_3_to_17_l1471_147147


namespace NUMINAMATH_CALUDE_flagpole_break_height_l1471_147123

theorem flagpole_break_height (h : ℝ) (d : ℝ) (x : ℝ) :
  h = 10 ∧ d = 2 ∧ x * x + d * d = (h - x) * (h - x) →
  x = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l1471_147123


namespace NUMINAMATH_CALUDE_cat_food_weight_l1471_147168

/-- Given the conditions of Mrs. Anderson's pet food purchase, prove that each bag of cat food weighs 3 pounds. -/
theorem cat_food_weight (cat_bags dog_bags : ℕ) (dog_extra_weight : ℕ) (ounces_per_pound : ℕ) (total_ounces : ℕ) :
  cat_bags = 2 ∧ 
  dog_bags = 2 ∧ 
  dog_extra_weight = 2 ∧
  ounces_per_pound = 16 ∧
  total_ounces = 256 →
  ∃ (cat_weight : ℕ), 
    cat_weight = 3 ∧
    total_ounces = ounces_per_pound * (cat_bags * cat_weight + dog_bags * (cat_weight + dog_extra_weight)) :=
by sorry

end NUMINAMATH_CALUDE_cat_food_weight_l1471_147168


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1471_147165

theorem no_positive_integer_solutions :
  ∀ x : ℕ+, ¬(15 < -3 * (x : ℤ) + 18) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1471_147165


namespace NUMINAMATH_CALUDE_friends_total_distance_l1471_147138

/-- Represents the distance walked by each friend -/
structure FriendDistances where
  lionel : ℕ  -- miles
  esther : ℕ  -- yards
  niklaus : ℕ  -- feet

/-- Converts miles to feet -/
def milesToFeet (miles : ℕ) : ℕ := miles * 5280

/-- Converts yards to feet -/
def yardsToFeet (yards : ℕ) : ℕ := yards * 3

/-- Calculates the total distance walked by all friends in feet -/
def totalDistanceInFeet (distances : FriendDistances) : ℕ :=
  milesToFeet distances.lionel + yardsToFeet distances.esther + distances.niklaus

/-- Theorem stating that the total distance walked by the friends is 26332 feet -/
theorem friends_total_distance (distances : FriendDistances) 
  (h1 : distances.lionel = 4)
  (h2 : distances.esther = 975)
  (h3 : distances.niklaus = 1287) :
  totalDistanceInFeet distances = 26332 := by
  sorry


end NUMINAMATH_CALUDE_friends_total_distance_l1471_147138


namespace NUMINAMATH_CALUDE_quarters_indeterminate_l1471_147103

/-- Represents the number of coins Mike has --/
structure MikeCoins where
  quarters : ℕ
  nickels : ℕ

/-- Represents the state of Mike's coins before and after his dad's borrowing --/
structure CoinState where
  initial : MikeCoins
  borrowed_nickels : ℕ
  current : MikeCoins

/-- Theorem stating that the number of quarters cannot be uniquely determined --/
theorem quarters_indeterminate (state : CoinState) 
    (h1 : state.initial.nickels = 87)
    (h2 : state.borrowed_nickels = 75)
    (h3 : state.current.nickels = 12)
    (h4 : state.initial.nickels = state.borrowed_nickels + state.current.nickels) :
    ∀ q : ℕ, ∃ state' : CoinState, 
      state'.initial.nickels = state.initial.nickels ∧
      state'.borrowed_nickels = state.borrowed_nickels ∧
      state'.current.nickels = state.current.nickels ∧
      state'.initial.quarters = q :=
  sorry

end NUMINAMATH_CALUDE_quarters_indeterminate_l1471_147103


namespace NUMINAMATH_CALUDE_reeyas_average_is_73_l1471_147150

def reeyas_scores : List ℝ := [55, 67, 76, 82, 85]

theorem reeyas_average_is_73 : 
  (reeyas_scores.sum / reeyas_scores.length : ℝ) = 73 := by
  sorry

end NUMINAMATH_CALUDE_reeyas_average_is_73_l1471_147150


namespace NUMINAMATH_CALUDE_crate_missing_dimension_l1471_147141

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  width : Real
  length : Real
  height : Real

/-- Represents a cylindrical tank -/
structure CylindricalTank where
  radius : Real
  height : Real

def fits_in_crate (tank : CylindricalTank) (crate : CrateDimensions) : Prop :=
  2 * tank.radius ≤ min crate.width crate.length ∧
  tank.height ≤ crate.height

def is_max_volume (tank : CylindricalTank) (crate : CrateDimensions) : Prop :=
  fits_in_crate tank crate ∧
  ∀ other_tank : CylindricalTank,
    fits_in_crate other_tank crate →
    tank.radius * tank.radius * tank.height ≥ other_tank.radius * other_tank.radius * other_tank.height

theorem crate_missing_dimension
  (crate : CrateDimensions)
  (h_width : crate.width = 8)
  (h_length : crate.length = 12)
  (tank : CylindricalTank)
  (h_radius : tank.radius = 6)
  (h_max_volume : is_max_volume tank crate) :
  crate.height = 12 :=
sorry

end NUMINAMATH_CALUDE_crate_missing_dimension_l1471_147141


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l1471_147186

/-- Represents the number of accent options for each letter in "cesontoiseaux" --/
def accentOptions : List Nat := [2, 5, 5, 1, 1, 3, 3, 1, 1, 2, 3, 1, 4]

/-- The number of ways to split 12 letters into 3 words --/
def wordSplitOptions : Nat := 66

/-- Calculates the total number of possible phrases --/
def totalPhrases : Nat :=
  wordSplitOptions * (accentOptions.foldl (·*·) 1)

/-- Theorem stating that the number of distinct prime factors of totalPhrases is 4 --/
theorem distinct_prime_factors_count :
  (Nat.factors totalPhrases).toFinset.card = 4 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l1471_147186


namespace NUMINAMATH_CALUDE_john_walking_speed_l1471_147111

/-- The walking speed of John in km/h -/
def john_speed : ℝ := 6

/-- The biking speed of Joan in km/h -/
def joan_speed (js : ℝ) : ℝ := 2 * js

/-- The distance between home and school in km -/
def distance : ℝ := 3

/-- The time difference between John's and Joan's departure in hours -/
def time_difference : ℝ := 0.25

theorem john_walking_speed :
  ∃ (js : ℝ),
    js = john_speed ∧
    joan_speed js = 2 * js ∧
    distance / js = distance / (joan_speed js) + time_difference :=
by sorry

end NUMINAMATH_CALUDE_john_walking_speed_l1471_147111


namespace NUMINAMATH_CALUDE_sets_equality_independent_of_order_l1471_147166

theorem sets_equality_independent_of_order (A B : Set ℕ) : 
  (∀ x, x ∈ A ↔ x ∈ B) → A = B :=
by sorry

end NUMINAMATH_CALUDE_sets_equality_independent_of_order_l1471_147166


namespace NUMINAMATH_CALUDE_mixed_fraction_product_l1471_147112

theorem mixed_fraction_product (X Y : ℕ) (hX : X > 0) (hY : Y > 0) : 
  (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ 5 < 5 + 1 / X ∧ 5 + 1 / X ≤ 11 / 2 → X = 17 ∧ Y = 8 := by
  sorry

end NUMINAMATH_CALUDE_mixed_fraction_product_l1471_147112


namespace NUMINAMATH_CALUDE_max_value_of_vector_sum_l1471_147153

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem max_value_of_vector_sum (a b c : V) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (hc : ‖c‖ = 3) :
  ∃ (max_value : ℝ), max_value = 94 ∧
    ∀ (x y z : V), ‖x‖ = 1 → ‖y‖ = 2 → ‖z‖ = 3 →
      ‖x + 2•y‖^2 + ‖y + 2•z‖^2 + ‖z + 2•x‖^2 ≤ max_value :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_vector_sum_l1471_147153


namespace NUMINAMATH_CALUDE_intersection_A_B_l1471_147105

def f (x : ℝ) : ℝ := x^2 - 12*x + 36

def A : Set ℕ := {a | 1 ≤ a ∧ a ≤ 10}

def B : Set ℕ := {b | ∃ a ∈ A, f a = b}

theorem intersection_A_B : A ∩ B = {1, 4, 9} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1471_147105


namespace NUMINAMATH_CALUDE_rectangle_ellipse_perimeter_l1471_147128

/-- Given a rectangle and an ellipse with specific properties, prove that the perimeter of the rectangle is 450. -/
theorem rectangle_ellipse_perimeter :
  ∀ (x y : ℝ) (a b : ℝ),
  -- Rectangle conditions
  x * y = 2500 ∧
  x / y = 5 / 4 ∧
  -- Ellipse conditions
  π * a * b = 2500 * π ∧
  x + y = 2 * a ∧
  (x^2 + y^2 : ℝ) = 4 * (a^2 - b^2) →
  -- Conclusion
  2 * (x + y) = 450 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ellipse_perimeter_l1471_147128


namespace NUMINAMATH_CALUDE_yoga_to_exercise_ratio_l1471_147164

/-- Represents the time spent on different activities in minutes -/
structure ExerciseTime where
  gym : ℕ
  bike : ℕ
  yoga : ℕ

/-- Calculates the total exercise time -/
def totalExerciseTime (t : ExerciseTime) : ℕ := t.gym + t.bike

/-- Represents the ratio of gym to bike time -/
def gymBikeRatio (t : ExerciseTime) : ℚ := t.gym / t.bike

theorem yoga_to_exercise_ratio (t : ExerciseTime) 
  (h1 : gymBikeRatio t = 2/3)
  (h2 : t.bike = 18) :
  ∃ (y : ℕ), t.yoga = y ∧ y / (totalExerciseTime t) = y / 30 := by
  sorry

end NUMINAMATH_CALUDE_yoga_to_exercise_ratio_l1471_147164


namespace NUMINAMATH_CALUDE_blue_left_handed_fraction_proof_l1471_147151

/-- The fraction of "blue" world participants who are left-handed -/
def blue_left_handed_fraction : ℝ := 0.66

theorem blue_left_handed_fraction_proof :
  let red_to_blue_ratio : ℝ := 2
  let red_left_handed_fraction : ℝ := 1/3
  let total_left_handed_fraction : ℝ := 0.44222222222222224
  blue_left_handed_fraction = 
    (3 * total_left_handed_fraction - 2 * red_left_handed_fraction) / red_to_blue_ratio :=
by sorry

#check blue_left_handed_fraction_proof

end NUMINAMATH_CALUDE_blue_left_handed_fraction_proof_l1471_147151


namespace NUMINAMATH_CALUDE_carly_swimming_time_l1471_147125

/-- Calculates the total swimming practice time in a month -/
def monthly_swimming_time (butterfly_hours_per_day : ℕ) (butterfly_days_per_week : ℕ)
                          (backstroke_hours_per_day : ℕ) (backstroke_days_per_week : ℕ)
                          (weeks_in_month : ℕ) : ℕ :=
  ((butterfly_hours_per_day * butterfly_days_per_week) +
   (backstroke_hours_per_day * backstroke_days_per_week)) * weeks_in_month

/-- Proves that Carly spends 96 hours practicing swimming in a month -/
theorem carly_swimming_time :
  monthly_swimming_time 3 4 2 6 4 = 96 :=
by sorry

end NUMINAMATH_CALUDE_carly_swimming_time_l1471_147125


namespace NUMINAMATH_CALUDE_fraction_of_25_smaller_than_40_percent_of_60_by_4_l1471_147185

theorem fraction_of_25_smaller_than_40_percent_of_60_by_4 : 
  (25 * (40 / 100 * 60 - 4)) / 25 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_25_smaller_than_40_percent_of_60_by_4_l1471_147185


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1471_147199

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola with the given properties is 2 + √3 -/
theorem hyperbola_eccentricity (h : Hyperbola) (F₁ F₂ A B : Point) :
  -- F₁ and F₂ are the left and right foci respectively
  -- A line passes through F₁ at a 60° angle
  -- The line intersects the y-axis at A and the right branch of the hyperbola at B
  -- A is the midpoint of F₁B
  (∃ (θ : ℝ), θ = Real.pi / 3 ∧ 
    A.x = 0 ∧
    B.x > 0 ∧
    (B.x - F₁.x) * Real.cos θ = (B.y - F₁.y) * Real.sin θ ∧
    A.x = (F₁.x + B.x) / 2 ∧
    A.y = (F₁.y + B.y) / 2) →
  -- The eccentricity of the hyperbola is 2 + √3
  h.a / Real.sqrt (h.a^2 + h.b^2) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1471_147199


namespace NUMINAMATH_CALUDE_division_problem_l1471_147194

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/5) : 
  c / a = 5/6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1471_147194


namespace NUMINAMATH_CALUDE_jellybean_theorem_l1471_147136

/-- The number of jellybeans each person has -/
structure JellyBeans where
  arnold : ℕ
  lee : ℕ
  tino : ℕ
  joshua : ℕ

/-- The conditions of the jellybean distribution -/
def jellybean_conditions (j : JellyBeans) : Prop :=
  j.arnold = 5 ∧
  j.lee = 2 * j.arnold ∧
  j.tino = j.lee + 24 ∧
  j.joshua = 3 * j.arnold

/-- The theorem to prove -/
theorem jellybean_theorem (j : JellyBeans) 
  (h : jellybean_conditions j) : 
  j.tino = 34 ∧ j.arnold + j.lee + j.tino + j.joshua = 64 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_theorem_l1471_147136


namespace NUMINAMATH_CALUDE_systematic_sampling_fourth_group_l1471_147167

/-- Systematic sampling function -/
def systematic_sample (class_size : ℕ) (sample_size : ℕ) (start : ℕ) : ℕ → ℕ :=
  fun group => start + (group - 1) * (class_size / sample_size)

theorem systematic_sampling_fourth_group 
  (class_size : ℕ) 
  (sample_size : ℕ) 
  (second_group_number : ℕ) :
  class_size = 72 →
  sample_size = 6 →
  second_group_number = 16 →
  systematic_sample class_size sample_size second_group_number 4 = 40 :=
by
  sorry

#check systematic_sampling_fourth_group

end NUMINAMATH_CALUDE_systematic_sampling_fourth_group_l1471_147167


namespace NUMINAMATH_CALUDE_probability_of_pair_after_removal_l1471_147170

/-- Represents a deck of cards -/
structure Deck :=
  (cards : Finset (Fin 10 × Fin 4))
  (card_count : cards.card = 40)

/-- Represents the deck after removing a matching pair -/
def RemainingDeck (d : Deck) : Finset (Fin 10 × Fin 4) :=
  d.cards.filter (λ x ↦ x.2 ≠ 3)

/-- The probability of selecting a matching pair from the remaining deck -/
def ProbabilityOfPair (d : Deck) : ℚ :=
  55 / 703

theorem probability_of_pair_after_removal (d : Deck) :
  ProbabilityOfPair d = 55 / 703 :=
sorry

end NUMINAMATH_CALUDE_probability_of_pair_after_removal_l1471_147170


namespace NUMINAMATH_CALUDE_sequence_property_l1471_147148

theorem sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n > 0) →
  (∀ n : ℕ, n > 0 → S n = (1 / 2) * (a n + 1 / a n)) →
  ∀ n : ℕ, n > 0 → S n = Real.sqrt n :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l1471_147148


namespace NUMINAMATH_CALUDE_counterexample_exists_l1471_147196

theorem counterexample_exists : ∃ x y : ℝ, x + y = 5 ∧ ¬(x = 1 ∧ y = 4) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1471_147196


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1471_147142

/-- Represents different sampling methods -/
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

/-- Represents a population with different income levels -/
structure Population :=
  (total : ℕ)
  (high_income : ℕ)
  (middle_income : ℕ)
  (low_income : ℕ)

/-- Represents a sampling problem -/
structure SamplingProblem :=
  (population : Population)
  (sample_size : ℕ)

/-- Determines the best sampling method for a given problem -/
def best_sampling_method (problem : SamplingProblem) : SamplingMethod :=
  sorry

/-- The community population for problem 1 -/
def community : Population :=
  { total := 600
  , high_income := 100
  , middle_income := 380
  , low_income := 120 }

/-- Problem 1: Family income study -/
def problem1 : SamplingProblem :=
  { population := community
  , sample_size := 100 }

/-- Problem 2: Student seminar selection -/
def problem2 : SamplingProblem :=
  { population := { total := 15, high_income := 0, middle_income := 0, low_income := 0 }
  , sample_size := 3 }

theorem correct_sampling_methods :
  (best_sampling_method problem1 = SamplingMethod.Stratified) ∧
  (best_sampling_method problem2 = SamplingMethod.SimpleRandom) :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1471_147142
