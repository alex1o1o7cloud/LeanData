import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1618_161820

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 0 → x * (x + 1) > 0) ∧
  (∃ x, x * (x + 1) > 0 ∧ ¬(x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1618_161820


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1618_161885

/-- Given a sphere with surface area 256π cm², its volume is (2048/3)π cm³ -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    4 * Real.pi * r^2 = 256 * Real.pi → 
    (4 / 3) * Real.pi * r^3 = (2048 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1618_161885


namespace NUMINAMATH_CALUDE_vector_same_direction_l1618_161821

open Real

/-- Given two vectors a and b in ℝ², prove that if they have the same direction,
    a = (1, -√3), and |b| = 1, then b = (1/2, -√3/2) -/
theorem vector_same_direction (a b : ℝ × ℝ) :
  (∃ k : ℝ, b = k • a) →  -- same direction
  a = (1, -Real.sqrt 3) →
  Real.sqrt ((b.1)^2 + (b.2)^2) = 1 →
  b = (1/2, -(Real.sqrt 3)/2) := by
sorry

end NUMINAMATH_CALUDE_vector_same_direction_l1618_161821


namespace NUMINAMATH_CALUDE_subset_condition_1_subset_condition_2_l1618_161838

-- Define the sets M, N1, and N2
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N1 (m : ℝ) : Set ℝ := {x | m - 6 ≤ x ∧ x ≤ 2*m - 1}
def N2 (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for part 1
theorem subset_condition_1 :
  ∀ m : ℝ, (M ⊆ N1 m) ↔ (2 ≤ m ∧ m ≤ 3) :=
by sorry

-- Theorem for part 2
theorem subset_condition_2 :
  ∀ m : ℝ, (N2 m ⊆ M) ↔ (m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_subset_condition_1_subset_condition_2_l1618_161838


namespace NUMINAMATH_CALUDE_negate_sum_diff_l1618_161808

theorem negate_sum_diff (a b c : ℝ) : -(a - b + c) = -a + b - c := by sorry

end NUMINAMATH_CALUDE_negate_sum_diff_l1618_161808


namespace NUMINAMATH_CALUDE_arthur_total_distance_l1618_161835

/-- Represents the distance walked in a single direction --/
structure DirectionalDistance :=
  (blocks : ℕ)

/-- Calculates the total number of blocks walked --/
def total_blocks (east west north south : DirectionalDistance) : ℕ :=
  east.blocks + west.blocks + north.blocks + south.blocks

/-- Converts blocks to miles --/
def blocks_to_miles (blocks : ℕ) : ℚ :=
  (blocks : ℚ) * (1 / 4 : ℚ)

/-- Theorem: Arthur's total walking distance is 5.75 miles --/
theorem arthur_total_distance :
  let east := DirectionalDistance.mk 8
  let north := DirectionalDistance.mk 10
  let south := DirectionalDistance.mk 5
  let west := DirectionalDistance.mk 0
  blocks_to_miles (total_blocks east west north south) = 5.75 := by
  sorry

end NUMINAMATH_CALUDE_arthur_total_distance_l1618_161835


namespace NUMINAMATH_CALUDE_linear_congruence_solution_l1618_161856

theorem linear_congruence_solution (x : Int) : 
  (7 * x + 3) % 17 = 2 % 17 ↔ x % 17 = 12 % 17 := by
  sorry

end NUMINAMATH_CALUDE_linear_congruence_solution_l1618_161856


namespace NUMINAMATH_CALUDE_escalator_standing_time_l1618_161851

/-- Represents the time it takes to travel an escalator under different conditions -/
def EscalatorTime (normal_time twice_normal_time : ℝ) : Prop :=
  ∃ (x u : ℝ),
    x > 0 ∧ u > 0 ∧
    (u + x) * normal_time = (u + 2*x) * twice_normal_time ∧
    u * (normal_time * 1.5) = (u + x) * normal_time

theorem escalator_standing_time 
  (h : EscalatorTime 40 30) : 
  ∃ (standing_time : ℝ), standing_time = 60 :=
by sorry

end NUMINAMATH_CALUDE_escalator_standing_time_l1618_161851


namespace NUMINAMATH_CALUDE_china_forex_reserves_scientific_notation_l1618_161882

-- Define the original amount in billions of US dollars
def original_amount : ℚ := 10663

-- Define the number of significant figures to retain
def significant_figures : ℕ := 3

-- Define the function to convert to scientific notation with given significant figures
def to_scientific_notation (x : ℚ) (sig_figs : ℕ) : ℚ × ℤ := sorry

-- Theorem statement
theorem china_forex_reserves_scientific_notation :
  let (mantissa, exponent) := to_scientific_notation (original_amount * 1000000000) significant_figures
  mantissa = 1.07 ∧ exponent = 12 := by sorry

end NUMINAMATH_CALUDE_china_forex_reserves_scientific_notation_l1618_161882


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l1618_161853

/-- Proves that 25x^2 + 40x + 16 is a perfect square binomial -/
theorem perfect_square_binomial : 
  ∃ (p q : ℝ), ∀ x : ℝ, 25*x^2 + 40*x + 16 = (p*x + q)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l1618_161853


namespace NUMINAMATH_CALUDE_function_with_finitely_many_discontinuities_doesnt_satisfy_condition1_l1618_161801

-- Define the function type
def RealFunction (a b : ℝ) := ℝ → ℝ

-- Define the property of having finitely many discontinuities
def HasFinitelyManyDiscontinuities (f : RealFunction a b) : Prop := sorry

-- Define condition (1) (we don't know what it is exactly, so we'll leave it abstract)
def SatisfiesCondition1 (f : RealFunction a b) : Prop := sorry

-- The main theorem
theorem function_with_finitely_many_discontinuities_doesnt_satisfy_condition1 
  {a b : ℝ} (f : RealFunction a b) 
  (h_finite : HasFinitelyManyDiscontinuities f) : 
  ¬(SatisfiesCondition1 f) := by
  sorry


end NUMINAMATH_CALUDE_function_with_finitely_many_discontinuities_doesnt_satisfy_condition1_l1618_161801


namespace NUMINAMATH_CALUDE_part_one_part_two_l1618_161813

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2 * x^2 - 3 * x * y + y^2 + 2 * x + 2 * y
def B (x y : ℝ) : ℝ := 4 * x^2 - 6 * x * y + 2 * y^2 - 3 * x - y

-- Part 1
theorem part_one : B 2 (-1/5) - 2 * A 2 (-1/5) = -13 := by sorry

-- Part 2
theorem part_two (a : ℝ) : 
  (∃ x y : ℝ, (|x - 2*a| + (y - 3)^2 = 0) ∧ (B x y - 2 * A x y = a)) → a = -1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1618_161813


namespace NUMINAMATH_CALUDE_car_cost_sharing_l1618_161849

theorem car_cost_sharing
  (total_cost : ℕ)
  (car_wash_funds : ℕ)
  (initial_friends : ℕ)
  (dropouts : ℕ)
  (h1 : total_cost = 1700)
  (h2 : car_wash_funds = 500)
  (h3 : initial_friends = 6)
  (h4 : dropouts = 1) :
  (total_cost - car_wash_funds) / (initial_friends - dropouts) -
  (total_cost - car_wash_funds) / initial_friends = 40 :=
by sorry

end NUMINAMATH_CALUDE_car_cost_sharing_l1618_161849


namespace NUMINAMATH_CALUDE_sequence_remainder_l1618_161872

def arithmetic_sequence_sum (a₁ : ℤ) (aₙ : ℤ) (n : ℕ) : ℤ :=
  n * (a₁ + aₙ) / 2

theorem sequence_remainder (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 3 →
  aₙ = 315 →
  d = 8 →
  aₙ = a₁ + (n - 1) * d →
  (arithmetic_sequence_sum a₁ aₙ n) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_remainder_l1618_161872


namespace NUMINAMATH_CALUDE_largest_number_with_6_and_3_l1618_161814

def largest_two_digit_number (d1 d2 : Nat) : Nat :=
  max (10 * d1 + d2) (10 * d2 + d1)

theorem largest_number_with_6_and_3 :
  largest_two_digit_number 6 3 = 63 := by
sorry

end NUMINAMATH_CALUDE_largest_number_with_6_and_3_l1618_161814


namespace NUMINAMATH_CALUDE_s_point_implies_a_value_l1618_161850

/-- Definition of an S point for two functions -/
def is_S_point (f g : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = g x₀ ∧ deriv f x₀ = deriv g x₀

/-- The main theorem -/
theorem s_point_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, is_S_point (λ x => a * x^2 - 1) (λ x => Real.log (a * x)) x₀) →
  a = 2 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_s_point_implies_a_value_l1618_161850


namespace NUMINAMATH_CALUDE_arithmetic_sequence_contains_powers_of_four_l1618_161837

theorem arithmetic_sequence_contains_powers_of_four (k : ℕ) :
  ∃ n : ℕ, 3 + 9 * (n - 1) = 3 * 4^k := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_contains_powers_of_four_l1618_161837


namespace NUMINAMATH_CALUDE_gcd_2_exp_1020_minus_1_2_exp_1031_minus_1_l1618_161898

theorem gcd_2_exp_1020_minus_1_2_exp_1031_minus_1 :
  Nat.gcd (2^1020 - 1) (2^1031 - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2_exp_1020_minus_1_2_exp_1031_minus_1_l1618_161898


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1618_161832

theorem smallest_n_satisfying_conditions : ∃ N : ℕ, 
  (∀ m : ℕ, m < N → ¬(3 ∣ m ∧ 11 ∣ m ∧ m % 12 = 6)) ∧ 
  (3 ∣ N ∧ 11 ∣ N ∧ N % 12 = 6) ∧
  N = 66 := by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1618_161832


namespace NUMINAMATH_CALUDE_parabola_translation_l1618_161830

/-- Given a parabola y = 2(x+1)^2 - 3, prove that translating it right by 1 unit and up by 3 units results in y = 2x^2 -/
theorem parabola_translation (x y : ℝ) :
  (y = 2 * (x + 1)^2 - 3) →
  (y + 3 = 2 * x^2) := by
sorry

end NUMINAMATH_CALUDE_parabola_translation_l1618_161830


namespace NUMINAMATH_CALUDE_prime_roots_integer_l1618_161836

theorem prime_roots_integer (p : ℕ) : 
  Prime p ∧ 
  (∃ x y : ℤ, x ≠ y ∧ 
    x^2 + 2*p*x - 240*p = 0 ∧ 
    y^2 + 2*p*y - 240*p = 0) ↔ 
  p = 5 := by
sorry

end NUMINAMATH_CALUDE_prime_roots_integer_l1618_161836


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1618_161809

def sum_of_reciprocals (a b : ℕ+) : ℚ := (a⁻¹ : ℚ) + (b⁻¹ : ℚ)

theorem reciprocal_sum_theorem (a b : ℕ+) 
  (sum_cond : a + b = 45)
  (lcm_cond : Nat.lcm a b = 120)
  (hcf_cond : Nat.gcd a b = 5) :
  sum_of_reciprocals a b = 3/40 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1618_161809


namespace NUMINAMATH_CALUDE_sum_of_gcd_and_lcm_equals_90_l1618_161871

def numbers : List Nat := [18, 36, 72]

theorem sum_of_gcd_and_lcm_equals_90 : 
  (numbers.foldl Nat.gcd numbers.head!) + (numbers.foldl Nat.lcm numbers.head!) = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_gcd_and_lcm_equals_90_l1618_161871


namespace NUMINAMATH_CALUDE_tan_beta_minus_2alpha_l1618_161839

theorem tan_beta_minus_2alpha (α β : Real) 
  (h1 : Real.tan α = 1 / 2) 
  (h2 : Real.tan (α - β) = -1 / 3) : 
  Real.tan (β - 2 * α) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_minus_2alpha_l1618_161839


namespace NUMINAMATH_CALUDE_unbroken_seashells_l1618_161841

theorem unbroken_seashells (total : ℕ) (broken : ℕ) (unbroken : ℕ) 
  (h1 : total = 7)
  (h2 : broken = 4)
  (h3 : unbroken = total - broken) :
  unbroken = 3 := by
  sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l1618_161841


namespace NUMINAMATH_CALUDE_soccer_league_games_l1618_161858

/-- The number of teams in the soccer league -/
def num_teams : ℕ := 10

/-- The number of games each team plays with every other team -/
def games_per_pair : ℕ := 2

/-- The total number of games played in the season -/
def total_games : ℕ := num_teams * (num_teams - 1) * games_per_pair / 2

theorem soccer_league_games :
  total_games = 90 :=
sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1618_161858


namespace NUMINAMATH_CALUDE_pizza_slices_l1618_161855

/-- The number of slices in a whole pizza -/
def total_slices : ℕ := sorry

/-- The number of slices each person ate -/
def slices_per_person : ℚ := 3/2

/-- The number of people who ate pizza -/
def num_people : ℕ := 2

/-- The number of slices left -/
def slices_left : ℕ := 5

/-- Theorem: The original number of slices in the pizza is 8 -/
theorem pizza_slices : total_slices = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l1618_161855


namespace NUMINAMATH_CALUDE_lateral_edge_length_l1618_161843

/-- A rectangular prism with 8 vertices and a given sum of lateral edge lengths -/
structure RectangularPrism :=
  (vertices : Nat)
  (lateral_edges_sum : ℝ)
  (is_valid : vertices = 8)

/-- The number of lateral edges in a rectangular prism -/
def lateral_edges_count : Nat := 4

/-- Theorem: In a valid rectangular prism, if the sum of lateral edges is 56,
    then each lateral edge has length 14 -/
theorem lateral_edge_length (prism : RectangularPrism)
    (h_sum : prism.lateral_edges_sum = 56) :
    prism.lateral_edges_sum / lateral_edges_count = 14 := by
  sorry

#check lateral_edge_length

end NUMINAMATH_CALUDE_lateral_edge_length_l1618_161843


namespace NUMINAMATH_CALUDE_gcd_sum_product_is_one_l1618_161822

/-- The sum of 1234 and 4321 -/
def sum_numbers : ℕ := 1234 + 4321

/-- The product of 1, 2, 3, and 4 -/
def product_digits : ℕ := 1 * 2 * 3 * 4

/-- Theorem stating that the greatest common divisor of the sum of 1234 and 4321,
    and the product of 1, 2, 3, and 4 is 1 -/
theorem gcd_sum_product_is_one : Nat.gcd sum_numbers product_digits = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_product_is_one_l1618_161822


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1618_161893

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (5 * y^12 + 4 * y^11 + 6 * y^9 + 3 * y^8) =
  15 * y^13 + 2 * y^12 - 8 * y^11 + 18 * y^10 - 3 * y^9 - 6 * y^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1618_161893


namespace NUMINAMATH_CALUDE_abcd_not_2012_l1618_161811

theorem abcd_not_2012 (a b c d : ℤ) 
  (h : (a - b) * (c + d) = (a + b) * (c - d)) : 
  a * b * c * d ≠ 2012 := by
sorry

end NUMINAMATH_CALUDE_abcd_not_2012_l1618_161811


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1618_161818

theorem polynomial_factorization (x : ℝ) :
  5 * (x + 3) * (x + 7) * (x + 9) * (x + 11) - 4 * x^2 =
  (5 * x^2 + 81 * x + 315) * (x + 3) * (x + 213) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1618_161818


namespace NUMINAMATH_CALUDE_bags_at_end_of_week_l1618_161857

/-- Calculates the total number of bags of cans at the end of the week given daily changes --/
def total_bags_at_end_of_week (
  monday : Real
  ) (tuesday : Real) (wednesday : Real) (thursday : Real) 
    (friday : Real) (saturday : Real) (sunday : Real) : Real :=
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

/-- Theorem stating the total number of bags at the end of the week --/
theorem bags_at_end_of_week : 
  total_bags_at_end_of_week 4 2.5 (-1.25) 0 3.75 (-1.5) 0 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_bags_at_end_of_week_l1618_161857


namespace NUMINAMATH_CALUDE_remaining_meat_l1618_161827

/-- Given an initial amount of meat and the amounts used for meatballs and spring rolls,
    prove that the remaining amount of meat is 12 kilograms. -/
theorem remaining_meat (initial_meat : ℝ) (meatball_fraction : ℝ) (spring_roll_meat : ℝ)
    (h1 : initial_meat = 20)
    (h2 : meatball_fraction = 1 / 4)
    (h3 : spring_roll_meat = 3) :
    initial_meat - (initial_meat * meatball_fraction) - spring_roll_meat = 12 :=
by sorry

end NUMINAMATH_CALUDE_remaining_meat_l1618_161827


namespace NUMINAMATH_CALUDE_michelle_crayon_count_l1618_161873

/-- The number of crayons in a box of the first type -/
def crayons_in_first_type : ℕ := 5

/-- The number of crayons in a box of the second type -/
def crayons_in_second_type : ℕ := 12

/-- The number of boxes of the first type -/
def boxes_of_first_type : ℕ := 4

/-- The number of boxes of the second type -/
def boxes_of_second_type : ℕ := 3

/-- The number of crayons missing from one box of the first type -/
def missing_crayons : ℕ := 2

/-- The total number of boxes -/
def total_boxes : ℕ := boxes_of_first_type + boxes_of_second_type

theorem michelle_crayon_count : 
  (boxes_of_first_type * crayons_in_first_type - missing_crayons) + 
  (boxes_of_second_type * crayons_in_second_type) = 54 := by
  sorry

#check michelle_crayon_count

end NUMINAMATH_CALUDE_michelle_crayon_count_l1618_161873


namespace NUMINAMATH_CALUDE_rat_value_formula_l1618_161845

/-- The number value of a letter in a shifted alphabet with offset N -/
def letterValue (position : ℕ) (N : ℕ) : ℕ := position + N

/-- The sum of letter values for the word "rat" in a shifted alphabet with offset N -/
def ratSum (N : ℕ) : ℕ := letterValue 18 N + letterValue 1 N + letterValue 20 N

/-- The length of the word "rat" -/
def ratLength : ℕ := 3

/-- The number value of the word "rat" in a shifted alphabet with offset N -/
def ratValue (N : ℕ) : ℕ := ratSum N * ratLength

theorem rat_value_formula (N : ℕ) : ratValue N = 117 + 9 * N := by
  sorry

end NUMINAMATH_CALUDE_rat_value_formula_l1618_161845


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l1618_161868

def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_triangle_interior_sum :
  (∀ k < 7, interior_sum k ≤ 50) ∧
  interior_sum 7 > 50 ∧
  interior_sum 7 = 62 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l1618_161868


namespace NUMINAMATH_CALUDE_total_peaches_l1618_161876

theorem total_peaches (initial_baskets : Nat) (initial_peaches_per_basket : Nat)
                      (additional_baskets : Nat) (additional_peaches_per_basket : Nat) :
  initial_baskets = 5 →
  initial_peaches_per_basket = 20 →
  additional_baskets = 4 →
  additional_peaches_per_basket = 25 →
  initial_baskets * initial_peaches_per_basket +
  additional_baskets * additional_peaches_per_basket = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l1618_161876


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1618_161852

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Given linear function passes through a point -/
def passesThroughPoint (f : LinearFunction) (p : Point) : Prop :=
  p.y = f.m * p.x + f.b

/-- The main theorem to be proved -/
theorem linear_function_not_in_third_quadrant :
  ∀ (p : Point), isInThirdQuadrant p → ¬passesThroughPoint ⟨-5, 2023⟩ p := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1618_161852


namespace NUMINAMATH_CALUDE_soap_brand_usage_l1618_161865

/-- Given a survey of households and their soap usage, prove the number using both brands --/
theorem soap_brand_usage (total : ℕ) (neither : ℕ) (only_A : ℕ) (both : ℕ) :
  total = 300 →
  neither = 80 →
  only_A = 60 →
  total = neither + only_A + both + 3 * both →
  both = 40 := by
sorry

end NUMINAMATH_CALUDE_soap_brand_usage_l1618_161865


namespace NUMINAMATH_CALUDE_three_zeros_implies_a_range_l1618_161861

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

/-- The statement that f has three distinct zeros -/
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0

/-- The main theorem: if f has three distinct zeros, then -2 < a < 2 -/
theorem three_zeros_implies_a_range (a : ℝ) :
  has_three_distinct_zeros a → -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_three_zeros_implies_a_range_l1618_161861


namespace NUMINAMATH_CALUDE_sally_pokemon_cards_l1618_161878

theorem sally_pokemon_cards 
  (initial_cards : ℕ) 
  (dan_cards : ℕ) 
  (total_cards : ℕ) 
  (h1 : initial_cards = 27) 
  (h2 : dan_cards = 41) 
  (h3 : total_cards = 88) : 
  total_cards - (initial_cards + dan_cards) = 20 := by
sorry

end NUMINAMATH_CALUDE_sally_pokemon_cards_l1618_161878


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1618_161862

def quadratic_inequality (a : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0

theorem quadratic_inequality_range :
  {a : ℝ | quadratic_inequality a} = Set.Ici 3 ∪ Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1618_161862


namespace NUMINAMATH_CALUDE_triangle_area_l1618_161886

theorem triangle_area (a b c A B C : Real) : 
  a + b = 3 →
  c = Real.sqrt 3 →
  Real.sin (2 * C - Real.pi / 6) = 1 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1618_161886


namespace NUMINAMATH_CALUDE_problem_statement_l1618_161864

theorem problem_statement (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2014 + b^2013 = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1618_161864


namespace NUMINAMATH_CALUDE_tank_problem_solution_l1618_161825

def tank_problem (capacity : ℝ) (initial_fill : ℝ) (empty_percent : ℝ) (refill_percent : ℝ) : ℝ :=
  let initial_volume := capacity * initial_fill
  let emptied_volume := initial_volume * empty_percent
  let remaining_volume := initial_volume - emptied_volume
  let refilled_volume := remaining_volume * refill_percent
  remaining_volume + refilled_volume

theorem tank_problem_solution :
  tank_problem 8000 (3/4) 0.4 0.3 = 4680 := by
  sorry

end NUMINAMATH_CALUDE_tank_problem_solution_l1618_161825


namespace NUMINAMATH_CALUDE_largest_k_for_real_root_l1618_161826

/-- The quadratic function f(x) parameterized by k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x + (k-1)^2

/-- The discriminant of f(x) as a function of k -/
def discriminant (k : ℝ) : ℝ := (-k)^2 - 4*(k-1)^2

/-- Theorem: The largest possible real value of k such that f has at least one real root is 2 -/
theorem largest_k_for_real_root :
  ∀ k : ℝ, (∃ x : ℝ, f k x = 0) → k ≤ 2 ∧ 
  ∃ x : ℝ, f 2 x = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_real_root_l1618_161826


namespace NUMINAMATH_CALUDE_junsu_is_winner_l1618_161887

-- Define the participants
inductive Participant
| Younghee
| Jimin
| Junsu

-- Define the amount of water drunk by each participant
def water_drunk : Participant → Float
  | Participant.Younghee => 1.4
  | Participant.Jimin => 1.8
  | Participant.Junsu => 2.1

-- Define the winner as the participant who drank the most water
def is_winner (p : Participant) : Prop :=
  ∀ q : Participant, water_drunk p ≥ water_drunk q

-- Theorem stating that Junsu is the winner
theorem junsu_is_winner : is_winner Participant.Junsu := by
  sorry

end NUMINAMATH_CALUDE_junsu_is_winner_l1618_161887


namespace NUMINAMATH_CALUDE_digit_sum_inequalities_l1618_161859

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

theorem digit_sum_inequalities :
  (∀ k : ℕ, sumOfDigits k ≤ 8 * sumOfDigits (8 * k)) ∧
  (∀ N : ℕ, sumOfDigits N ≤ 5 * sumOfDigits (5^5 * N)) := by sorry

end NUMINAMATH_CALUDE_digit_sum_inequalities_l1618_161859


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l1618_161883

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (2, 1) and b = (x, -2) for some x ∈ ℝ, then a + b = (-2, -1) -/
theorem parallel_vectors_sum (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x, -2]
  (∃ (k : ℝ), a = k • b) →
  a + b = ![(-2), (-1)] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l1618_161883


namespace NUMINAMATH_CALUDE_pen_cost_calculation_l1618_161829

def notebook_cost (pen_cost : ℝ) : ℝ := 3 * pen_cost

theorem pen_cost_calculation (total_cost : ℝ) (num_notebooks : ℕ) 
  (h1 : total_cost = 18)
  (h2 : num_notebooks = 4) :
  ∃ (pen_cost : ℝ), 
    pen_cost = 1.5 ∧ 
    total_cost = num_notebooks * (notebook_cost pen_cost) :=
by
  sorry

end NUMINAMATH_CALUDE_pen_cost_calculation_l1618_161829


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1618_161890

theorem absolute_value_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  x₁ > x₂ ∧ 
  (|x₁ - 3| = 15) ∧ 
  (|x₂ - 3| = 15) ∧ 
  (x₁ - x₂ = 30) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1618_161890


namespace NUMINAMATH_CALUDE_quadrant_function_m_range_l1618_161894

/-- A proportional function passing through the second and fourth quadrants -/
structure QuadrantFunction where
  m : ℝ
  passes_through_second_fourth : (∀ x y, y = (1 - m) * x → 
    ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)))

/-- The range of m for a QuadrantFunction -/
theorem quadrant_function_m_range (f : QuadrantFunction) : f.m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_function_m_range_l1618_161894


namespace NUMINAMATH_CALUDE_bran_leftover_amount_l1618_161812

/-- Represents Bran's financial situation for a semester --/
structure BranFinances where
  tuitionFee : ℝ
  additionalExpenses : ℝ
  hourlyWage : ℝ
  weeklyHours : ℝ
  scholarshipPercentage : ℝ
  semesterMonths : ℕ

/-- Calculates the amount left after paying expenses --/
def calculateLeftoverAmount (finances : BranFinances) : ℝ :=
  let scholarshipAmount := finances.tuitionFee * finances.scholarshipPercentage
  let tuitionAfterScholarship := finances.tuitionFee - scholarshipAmount
  let totalExpenses := tuitionAfterScholarship + finances.additionalExpenses
  let weeklyEarnings := finances.hourlyWage * finances.weeklyHours
  let totalEarnings := weeklyEarnings * (finances.semesterMonths * 4 : ℝ)
  totalEarnings - totalExpenses

/-- Theorem stating that Bran will have $1,481 left after expenses --/
theorem bran_leftover_amount :
  let finances : BranFinances := {
    tuitionFee := 2500,
    additionalExpenses := 600,
    hourlyWage := 18,
    weeklyHours := 12,
    scholarshipPercentage := 0.45,
    semesterMonths := 4
  }
  calculateLeftoverAmount finances = 1481 := by
  sorry

end NUMINAMATH_CALUDE_bran_leftover_amount_l1618_161812


namespace NUMINAMATH_CALUDE_orange_ribbons_count_l1618_161842

/-- The number of ribbons in a container with yellow, purple, orange, and black ribbons. -/
def total_ribbons : ℚ :=
  let black_ribbons : ℚ := 45
  let black_fraction : ℚ := 1 - (1/4 + 3/8 + 1/8)
  black_ribbons / black_fraction

/-- The number of orange ribbons in the container. -/
def orange_ribbons : ℚ := (1/8) * total_ribbons

/-- Theorem stating that the number of orange ribbons is 22.5. -/
theorem orange_ribbons_count : orange_ribbons = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_orange_ribbons_count_l1618_161842


namespace NUMINAMATH_CALUDE_train_length_l1618_161869

/-- The length of a train given its speed, the speed of a man running in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) 
  (h1 : train_speed = 50) 
  (h2 : man_speed = 4) 
  (h3 : passing_time = 8) : 
  (train_speed + man_speed) * passing_time * (1000 / 3600) = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1618_161869


namespace NUMINAMATH_CALUDE_inequality_proof_l1618_161840

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ≤ 1 / (a*b*c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1618_161840


namespace NUMINAMATH_CALUDE_parallel_vectors_x_values_l1618_161863

/-- Given two vectors a and b in ℝ², prove that if they are parallel and have the specified components, then x must be 2 or -1. -/
theorem parallel_vectors_x_values (x : ℝ) :
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (x - 1, 2)
  (∃ (k : ℝ), a = k • b) → x = 2 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_values_l1618_161863


namespace NUMINAMATH_CALUDE_find_n_l1618_161834

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) + 3 = 15 ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_find_n_l1618_161834


namespace NUMINAMATH_CALUDE_smallest_square_sides_l1618_161815

/-- Represents the configuration of three squares arranged as described in the problem -/
structure SquareArrangement where
  small_side : ℝ
  mid_side : ℝ
  large_side : ℝ
  mid_is_larger : mid_side = small_side + 8
  large_is_50 : large_side = 50

/-- The theorem stating the possible side lengths of the smallest square -/
theorem smallest_square_sides (arr : SquareArrangement) : 
  (arr.small_side = 2 ∨ arr.small_side = 32) ↔ 
  (∃ (x : ℝ), x * (x + 8) * 8 = x * (42 - x) * (x + 8)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_sides_l1618_161815


namespace NUMINAMATH_CALUDE_certain_instrument_count_l1618_161891

/-- The number of the certain instrument Charlie owns -/
def x : ℕ := sorry

/-- Charlie's flutes -/
def charlie_flutes : ℕ := 1

/-- Charlie's horns -/
def charlie_horns : ℕ := 2

/-- Carli's flutes -/
def carli_flutes : ℕ := 2 * charlie_flutes

/-- Carli's horns -/
def carli_horns : ℕ := charlie_horns / 2

/-- Total number of instruments owned by Charlie and Carli -/
def total_instruments : ℕ := 7

theorem certain_instrument_count : 
  charlie_flutes + charlie_horns + x + carli_flutes + carli_horns = total_instruments ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_instrument_count_l1618_161891


namespace NUMINAMATH_CALUDE_combine_like_terms_l1618_161823

theorem combine_like_terms (a : ℝ) : 3 * a + 2 * a = 5 * a := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l1618_161823


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1618_161828

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1, eccentricity e = 5/4, 
    and right focus F₂(5,0), prove that the equation of C is x²/16 - y²/9 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →  -- Equation of hyperbola C
  (a/b)^2 + 1 = (5/4)^2 →               -- Eccentricity e = 5/4
  5^2 = a^2 + b^2 →                     -- Right focus F₂(5,0)
  a^2 = 16 ∧ b^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1618_161828


namespace NUMINAMATH_CALUDE_opposite_of_ten_l1618_161804

theorem opposite_of_ten : ∃ x : ℝ, (x + 10 = 0) ∧ (x = -10) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_ten_l1618_161804


namespace NUMINAMATH_CALUDE_x_value_l1618_161884

theorem x_value (y : ℝ) (x : ℝ) : 
  y = 125 * (1 + 0.1) → 
  x = y * (1 - 0.1) → 
  x = 123.75 := by
sorry

end NUMINAMATH_CALUDE_x_value_l1618_161884


namespace NUMINAMATH_CALUDE_share_distribution_l1618_161854

theorem share_distribution (total : ℚ) (a b c : ℚ) : 
  total = 392 →
  a = (1 / 2) * b →
  b = (1 / 2) * c →
  total = a + b + c →
  c = 224 := by
sorry

end NUMINAMATH_CALUDE_share_distribution_l1618_161854


namespace NUMINAMATH_CALUDE_basketball_points_per_basket_l1618_161888

theorem basketball_points_per_basket 
  (matthew_points : ℕ) 
  (shawn_points : ℕ) 
  (total_baskets : ℕ) 
  (h1 : matthew_points = 9) 
  (h2 : shawn_points = 6) 
  (h3 : total_baskets = 5) : 
  (matthew_points + shawn_points) / total_baskets = 3 := by
sorry

end NUMINAMATH_CALUDE_basketball_points_per_basket_l1618_161888


namespace NUMINAMATH_CALUDE_combinatorics_identities_l1618_161833

theorem combinatorics_identities :
  (∀ n k : ℕ, Nat.choose n k = Nat.choose n (n - k)) ∧
  (Nat.choose 5 3 = Nat.choose 4 2 + Nat.choose 4 3) ∧
  (5 * Nat.factorial 5 = Nat.factorial 6 - Nat.factorial 5) :=
by sorry

end NUMINAMATH_CALUDE_combinatorics_identities_l1618_161833


namespace NUMINAMATH_CALUDE_pages_needed_is_twelve_l1618_161800

/-- Calculates the number of pages needed to organize sports cards -/
def pages_needed (new_baseball old_baseball new_basketball old_basketball new_football old_football cards_per_page : ℕ) : ℕ :=
  let total_baseball := new_baseball + old_baseball
  let total_basketball := new_basketball + old_basketball
  let total_football := new_football + old_football
  let baseball_pages := (total_baseball + cards_per_page - 1) / cards_per_page
  let basketball_pages := (total_basketball + cards_per_page - 1) / cards_per_page
  let football_pages := (total_football + cards_per_page - 1) / cards_per_page
  baseball_pages + basketball_pages + football_pages

/-- Theorem stating that the number of pages needed is 12 -/
theorem pages_needed_is_twelve :
  pages_needed 3 9 4 6 7 5 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_pages_needed_is_twelve_l1618_161800


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1618_161899

theorem arithmetic_calculation : 8 / 4 - 3 - 10 + 3 * 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1618_161899


namespace NUMINAMATH_CALUDE_problem_statement_l1618_161824

theorem problem_statement (x y z k : ℝ) 
  (h1 : x + 1/y = k)
  (h2 : 2*y + 2/z = k)
  (h3 : 3*z + 3/x = k)
  (h4 : x*y*z = 3) :
  k = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1618_161824


namespace NUMINAMATH_CALUDE_function_composition_property_l1618_161802

theorem function_composition_property (f : ℤ → ℤ) (m : ℕ+) :
  (∀ n : ℤ, (f^[m] n = n + 2017)) → (m = 1 ∨ m = 2017) := by
  sorry

#check function_composition_property

end NUMINAMATH_CALUDE_function_composition_property_l1618_161802


namespace NUMINAMATH_CALUDE_picture_area_l1618_161805

theorem picture_area (x y : ℤ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : (2*x + 4)*(y + 2) - x*y = 56) : 
  x * y = 24 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l1618_161805


namespace NUMINAMATH_CALUDE_product_of_cube_and_square_l1618_161807

theorem product_of_cube_and_square (x : ℝ) : 2 * x^3 * (-3 * x)^2 = 18 * x^5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cube_and_square_l1618_161807


namespace NUMINAMATH_CALUDE_initial_average_marks_l1618_161892

/-- 
Given a class of students with an incorrect average mark, prove that the initial average 
before correcting an error in one student's mark is equal to a specific value.
-/
theorem initial_average_marks 
  (n : ℕ) -- number of students
  (wrong_mark correct_mark : ℕ) -- the wrong and correct marks for one student
  (final_average : ℚ) -- the correct average after fixing the error
  (h1 : n = 25) -- there are 25 students
  (h2 : wrong_mark = 60) -- the wrong mark was 60
  (h3 : correct_mark = 10) -- the correct mark is 10
  (h4 : final_average = 98) -- the final correct average is 98
  : ∃ (initial_average : ℚ), initial_average = 100 ∧ 
    n * initial_average - (wrong_mark - correct_mark) = n * final_average :=
by sorry

end NUMINAMATH_CALUDE_initial_average_marks_l1618_161892


namespace NUMINAMATH_CALUDE_group_purchase_equations_l1618_161877

theorem group_purchase_equations (x y : ℤ) : 
  (∀ (z : ℤ), z * x - y = 5 → z = 9) ∧ 
  (∀ (w : ℤ), y - w * x = 4 → w = 6) → 
  (9 * x - 5 = y ∧ 6 * x + 4 = y) := by
  sorry

end NUMINAMATH_CALUDE_group_purchase_equations_l1618_161877


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l1618_161803

theorem rectangle_area_diagonal (length width diagonal : ℝ) (k : ℝ) : 
  length > 0 → width > 0 → diagonal > 0 →
  length / width = 3 / 2 → 
  diagonal^2 = length^2 + width^2 →
  k = 6 / 13 →
  length * width = k * diagonal^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l1618_161803


namespace NUMINAMATH_CALUDE_kindergarten_ratio_l1618_161819

theorem kindergarten_ratio (boys girls : ℕ) (h1 : boys = 12) (h2 : 2 * girls = 3 * boys) : girls = 18 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_ratio_l1618_161819


namespace NUMINAMATH_CALUDE_eventual_stability_l1618_161860

/-- Represents a line of 2018 natural numbers -/
def Line := Fin 2018 → ℕ

/-- Applies the frequency counting operation to a line -/
def frequency_count (l : Line) : Line := sorry

/-- Predicate to check if two lines are identical -/
def identical (l1 l2 : Line) : Prop := ∀ i, l1 i = l2 i

/-- Theorem stating that repeated frequency counting eventually leads to identical lines -/
theorem eventual_stability (initial : Line) : 
  ∃ n : ℕ, ∀ m ≥ n, identical (frequency_count^[m] initial) (frequency_count^[m+1] initial) := by
  sorry

end NUMINAMATH_CALUDE_eventual_stability_l1618_161860


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l1618_161817

-- Define the vectors m and n
def m : Fin 2 → ℝ := ![1, 3]
def n (t : ℝ) : Fin 2 → ℝ := ![2, t]

-- Define the condition for perpendicularity
def perpendicular (t : ℝ) : Prop :=
  (m 0 + n t 0) * (m 0 - n t 0) + (m 1 + n t 1) * (m 1 - n t 1) = 0

-- State the theorem
theorem vector_perpendicular_condition (t : ℝ) :
  perpendicular t → t = Real.sqrt 6 ∨ t = -Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l1618_161817


namespace NUMINAMATH_CALUDE_inequality_proof_l1618_161870

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (b + c) + 1 / (a + c) + 1 / (a + b) ≥ 9 / (2 * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1618_161870


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1618_161848

theorem sum_of_coefficients (x : ℝ) : 
  let p : ℝ → ℝ := λ x => 2 * (4 * x^8 - 5 * x^3 + 6) + 8 * (x^6 + 3 * x^4 - 4)
  p 1 = 10 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1618_161848


namespace NUMINAMATH_CALUDE_positive_distinct_solution_condition_l1618_161897

theorem positive_distinct_solution_condition 
  (a b x y z : ℝ) 
  (eq1 : x + y + z = a) 
  (eq2 : x^2 + y^2 + z^2 = b^2) 
  (eq3 : x * y = z^2) 
  (pos_x : x > 0) 
  (pos_y : y > 0) 
  (pos_z : z > 0) 
  (distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) : 
  b^2 ≥ a^2 / 2 := by
sorry

end NUMINAMATH_CALUDE_positive_distinct_solution_condition_l1618_161897


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1618_161879

theorem trigonometric_identity (α β : ℝ) 
  (h : (Real.cos α)^4 / (Real.cos β)^2 + (Real.sin α)^4 / (Real.sin β)^2 = 1) :
  (Real.sin β)^4 / (Real.sin α)^2 + (Real.cos β)^4 / (Real.cos α)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1618_161879


namespace NUMINAMATH_CALUDE_quarter_difference_l1618_161806

/-- Represents the number and value of coins in Sally's savings jar. -/
structure CoinJar where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  total_coins : ℕ
  total_value : ℕ

/-- Checks if a CoinJar configuration is valid according to the problem constraints. -/
def is_valid_jar (jar : CoinJar) : Prop :=
  jar.total_coins = 150 ∧
  jar.total_value = 2000 ∧
  jar.total_coins = jar.nickels + jar.dimes + jar.quarters ∧
  jar.total_value = 5 * jar.nickels + 10 * jar.dimes + 25 * jar.quarters

/-- Finds the maximum number of quarters possible in a valid CoinJar. -/
def max_quarters (jar : CoinJar) : ℕ := sorry

/-- Finds the minimum number of quarters possible in a valid CoinJar. -/
def min_quarters (jar : CoinJar) : ℕ := sorry

/-- Theorem stating the difference between max and min quarters is 62. -/
theorem quarter_difference (jar : CoinJar) (h : is_valid_jar jar) :
  max_quarters jar - min_quarters jar = 62 := by sorry

end NUMINAMATH_CALUDE_quarter_difference_l1618_161806


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1618_161874

/-- A line passing through (1,3) perpendicular to 2x-6y-8=0 has equation y+3x-6=0 -/
theorem perpendicular_line_equation :
  let l₁ : Set (ℝ × ℝ) := {p | 2 * p.1 - 6 * p.2 - 8 = 0}
  let l₂ : Set (ℝ × ℝ) := {p | p.2 + 3 * p.1 - 6 = 0}
  let point : ℝ × ℝ := (1, 3)
  (point ∈ l₂) ∧
  (∀ (p₁ p₂ : ℝ × ℝ), p₁ ∈ l₁ → p₂ ∈ l₁ → p₁ ≠ p₂ →
    ∀ (q₁ q₂ : ℝ × ℝ), q₁ ∈ l₂ → q₂ ∈ l₂ → q₁ ≠ q₂ →
      ((p₂.1 - p₁.1) * (q₂.1 - q₁.1) + (p₂.2 - p₁.2) * (q₂.2 - q₁.2) = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1618_161874


namespace NUMINAMATH_CALUDE_correct_calculation_l1618_161816

theorem correct_calculation (y : ℝ) : -8 * y + 3 * y = -5 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1618_161816


namespace NUMINAMATH_CALUDE_min_visible_pairs_l1618_161847

/-- Represents the number of birds on the circle -/
def num_birds : ℕ := 155

/-- Represents the maximum arc length for mutual visibility in degrees -/
def visibility_arc : ℝ := 10

/-- Calculates the number of pairs in a group of n birds -/
def pairs_in_group (n : ℕ) : ℕ := n.choose 2

/-- Represents the optimal grouping of birds -/
def optimal_grouping : List ℕ := [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

theorem min_visible_pairs :
  (List.sum (List.map pairs_in_group optimal_grouping) = 270) ∧
  (List.sum optimal_grouping = num_birds) ∧
  (List.length optimal_grouping * visibility_arc ≥ 360) ∧
  (∀ (grouping : List ℕ), 
    (List.sum grouping = num_birds) →
    (List.length grouping * visibility_arc ≥ 360) →
    (List.sum (List.map pairs_in_group grouping) ≥ 270)) := by
  sorry

end NUMINAMATH_CALUDE_min_visible_pairs_l1618_161847


namespace NUMINAMATH_CALUDE_angle_identity_l1618_161810

/-- If the terminal side of angle α passes through point P(-2, 1) in the rectangular coordinate system, 
    then cos²α - sin(2α) = 8/5 -/
theorem angle_identity (α : ℝ) : 
  (∃ (x y : ℝ), x = -2 ∧ y = 1 ∧ y / x = Real.tan α) → 
  Real.cos α ^ 2 - Real.sin (2 * α) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_identity_l1618_161810


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_a_l1618_161831

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 1|

-- Theorem for the first part of the problem
theorem solution_set_f (x : ℝ) :
  f x ≤ 2 ↔ x ≤ -1 ∨ x ≥ -1/3 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_a (a : ℝ) :
  (∃ b : ℝ, ∀ x : ℝ, |a + b| - |a - b| ≥ f x) ↔ a ≥ 5/4 ∨ a ≤ -5/4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_a_l1618_161831


namespace NUMINAMATH_CALUDE_rectangular_paper_area_l1618_161846

theorem rectangular_paper_area (L W : ℝ) 
  (h1 : L + 2*W = 34) 
  (h2 : 2*L + W = 38) : 
  L * W = 140 := by
sorry

end NUMINAMATH_CALUDE_rectangular_paper_area_l1618_161846


namespace NUMINAMATH_CALUDE_sqrt_seven_decimal_part_l1618_161889

theorem sqrt_seven_decimal_part (a : ℝ) : 
  (2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3) → 
  (a = Real.sqrt 7 - 2) → 
  ((Real.sqrt 7 + 2) * a = 3) := by
sorry

end NUMINAMATH_CALUDE_sqrt_seven_decimal_part_l1618_161889


namespace NUMINAMATH_CALUDE_min_sum_reciprocal_constraint_min_sum_reciprocal_constraint_equality_l1618_161896

theorem min_sum_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 4/y = 1) : x + y ≥ 9 := by
  sorry

theorem min_sum_reciprocal_constraint_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 4/y = 1) : 
  (x + y = 9) ↔ (x = 3 ∧ y = 6) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_reciprocal_constraint_min_sum_reciprocal_constraint_equality_l1618_161896


namespace NUMINAMATH_CALUDE_first_month_sale_is_6400_l1618_161881

/-- Represents the sales data for a grocer over six months -/
structure GrocerSales where
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ
  average : ℕ

/-- Calculates the sale in the first month given the sales data -/
def firstMonthSale (sales : GrocerSales) : ℕ :=
  6 * sales.average - (sales.month2 + sales.month3 + sales.month4 + sales.month5 + sales.month6)

/-- Theorem stating that the first month's sale is 6400 given the specific sales data -/
theorem first_month_sale_is_6400 (sales : GrocerSales) 
  (h1 : sales.month2 = 7000)
  (h2 : sales.month3 = 6800)
  (h3 : sales.month4 = 7200)
  (h4 : sales.month5 = 6500)
  (h5 : sales.month6 = 5100)
  (h6 : sales.average = 6500) :
  firstMonthSale sales = 6400 := by
  sorry


end NUMINAMATH_CALUDE_first_month_sale_is_6400_l1618_161881


namespace NUMINAMATH_CALUDE_excess_of_repeating_over_terminating_l1618_161866

/-- The value of the repeating decimal 0.727272... -/
def repeating_72 : ℚ := 72 / 99

/-- The value of the terminating decimal 0.72 -/
def terminating_72 : ℚ := 72 / 100

/-- The fraction by which 0.727272... exceeds 0.72 -/
def excess_fraction : ℚ := 800 / 1099989

theorem excess_of_repeating_over_terminating :
  repeating_72 - terminating_72 = excess_fraction := by
  sorry

end NUMINAMATH_CALUDE_excess_of_repeating_over_terminating_l1618_161866


namespace NUMINAMATH_CALUDE_bike_savings_time_l1618_161880

/-- The cost of the mountain bike in dollars -/
def bike_cost : ℕ := 600

/-- The total birthday money Chandler received in dollars -/
def birthday_money : ℕ := 60 + 40 + 20

/-- The amount Chandler earns per week from his paper route in dollars -/
def weekly_earnings : ℕ := 20

/-- The number of weeks it takes to save enough money for the bike -/
def weeks_to_save : ℕ := 24

/-- Theorem stating that it takes 24 weeks to save enough money for the bike -/
theorem bike_savings_time :
  birthday_money + weekly_earnings * weeks_to_save = bike_cost := by
  sorry

end NUMINAMATH_CALUDE_bike_savings_time_l1618_161880


namespace NUMINAMATH_CALUDE_basketball_prices_l1618_161867

theorem basketball_prices (price_A price_B : ℝ) : 
  price_A = 2 * price_B - 48 →
  9600 / price_A = 7200 / price_B →
  price_A = 96 ∧ price_B = 72 := by
sorry

end NUMINAMATH_CALUDE_basketball_prices_l1618_161867


namespace NUMINAMATH_CALUDE_alien_eggs_count_l1618_161895

-- Define a function to convert a number from base 7 to base 10
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

-- Theorem statement
theorem alien_eggs_count :
  base7ToBase10 [1, 2, 3] = 162 := by
  sorry

end NUMINAMATH_CALUDE_alien_eggs_count_l1618_161895


namespace NUMINAMATH_CALUDE_negative_square_power_2014_l1618_161875

theorem negative_square_power_2014 : -(-(-1)^2)^2014 = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_power_2014_l1618_161875


namespace NUMINAMATH_CALUDE_infinite_solutions_l1618_161844

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * x - 4 * y = 10
def equation2 (x y : ℝ) : Prop := 6 * x - 8 * y = 20

-- Theorem stating that the system has infinitely many solutions
theorem infinite_solutions :
  ∃ (f : ℝ → ℝ × ℝ), ∀ t : ℝ,
    let (x, y) := f t
    equation1 x y ∧ equation2 x y ∧
    (∀ s : ℝ, s ≠ t → f s ≠ f t) :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_l1618_161844
