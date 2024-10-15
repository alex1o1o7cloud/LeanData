import Mathlib

namespace NUMINAMATH_CALUDE_triangle_inequality_l2020_202058

theorem triangle_inequality (a b c : ℝ) (x y z : ℝ) : 
  a ≥ b ∧ b ≥ c ∧ c > 0 ∧ 
  0 < x ∧ x < π ∧ 0 < y ∧ y < π ∧ 0 < z ∧ z < π ∧ x + y + z = π →
  b * c + c * a - a * b < b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ∧
  b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ≤ (1 / 2) * (a^2 + b^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2020_202058


namespace NUMINAMATH_CALUDE_total_gum_pieces_l2020_202082

/-- The number of gum packages Robin has -/
def num_packages : ℕ := 9

/-- The number of gum pieces in each package -/
def pieces_per_package : ℕ := 15

/-- Theorem: The total number of gum pieces Robin has is 135 -/
theorem total_gum_pieces : num_packages * pieces_per_package = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_gum_pieces_l2020_202082


namespace NUMINAMATH_CALUDE_fraction_simplification_l2020_202074

theorem fraction_simplification (a b c : ℝ) :
  ((a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) / (a^2 - b^2 - c^2 - 2*b*c) = (a + b + c) / (a - b - c)) ∧
  ((a^2 - 3*a*b + a*c + 2*b^2 - 2*b*c) / (a^2 - b^2 + 2*b*c - c^2) = (a - 2*b) / (a + b - c)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2020_202074


namespace NUMINAMATH_CALUDE_shaded_area_is_18_l2020_202016

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a point that divides a line segment -/
structure DivisionPoint where
  x : ℝ
  y : ℝ

/-- Calculate the area of the shaded region in a rectangle -/
def shadedArea (rect : Rectangle) (numDivisions : ℕ) : ℝ :=
  sorry

/-- Theorem stating that the shaded area in the given rectangle is 18 -/
theorem shaded_area_is_18 :
  let rect : Rectangle := { length := 9, width := 5 }
  let numDivisions : ℕ := 5
  shadedArea rect numDivisions = 18 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_18_l2020_202016


namespace NUMINAMATH_CALUDE_find_divisor_l2020_202053

theorem find_divisor (dividend : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 76)
  (h2 : remainder = 8)
  (h3 : quotient = 4)
  : ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 17 :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l2020_202053


namespace NUMINAMATH_CALUDE_investment_interest_rate_l2020_202072

/-- Proves that given the specified investment conditions, the second year's interest rate is 6% -/
theorem investment_interest_rate (initial_investment : ℝ) (first_rate : ℝ) (second_rate : ℝ) (final_value : ℝ) :
  initial_investment = 15000 →
  first_rate = 0.08 →
  final_value = 17160 →
  initial_investment * (1 + first_rate) * (1 + second_rate) = final_value →
  second_rate = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l2020_202072


namespace NUMINAMATH_CALUDE_power_division_l2020_202005

theorem power_division (a : ℝ) : a^7 / a = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l2020_202005


namespace NUMINAMATH_CALUDE_calculator_operations_l2020_202078

def A (n : ℕ) : ℕ := 2 * n
def B (n : ℕ) : ℕ := n + 1

def applyKeys (n : ℕ) (keys : List (ℕ → ℕ)) : ℕ :=
  keys.foldl (fun acc f => f acc) n

theorem calculator_operations :
  (∃ keys : List (ℕ → ℕ), keys.length = 4 ∧ applyKeys 1 keys = 10) ∧
  (∃ keys : List (ℕ → ℕ), keys.length = 6 ∧ applyKeys 1 keys = 15) ∧
  (∃ keys : List (ℕ → ℕ), keys.length = 8 ∧ applyKeys 1 keys = 100) := by
  sorry

end NUMINAMATH_CALUDE_calculator_operations_l2020_202078


namespace NUMINAMATH_CALUDE_min_distance_to_vertex_l2020_202015

/-- A right circular cone with base radius 1 and slant height 3 -/
structure Cone where
  base_radius : ℝ
  slant_height : ℝ
  base_radius_eq : base_radius = 1
  slant_height_eq : slant_height = 3

/-- A point on the shortest path between two points on the base circumference -/
def ShortestPathPoint (c : Cone) := ℝ

/-- The distance from the vertex to a point on the shortest path -/
def distance_to_vertex (c : Cone) (p : ShortestPathPoint c) : ℝ := sorry

/-- The theorem stating the minimum distance from the vertex to a point on the shortest path -/
theorem min_distance_to_vertex (c : Cone) : 
  ∃ (p : ShortestPathPoint c), distance_to_vertex c p = 3/2 ∧ 
  ∀ (q : ShortestPathPoint c), distance_to_vertex c q ≥ 3/2 := by sorry

end NUMINAMATH_CALUDE_min_distance_to_vertex_l2020_202015


namespace NUMINAMATH_CALUDE_a_range_when_p_false_l2020_202027

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Ioo 2 3, x^2 + 5 > a*x

-- Define the range of a
def a_range : Set ℝ := Set.Ici (2 * Real.sqrt 5)

-- Theorem statement
theorem a_range_when_p_false :
  (∃ a : ℝ, ¬(p a)) ↔ ∃ a ∈ a_range, True :=
sorry

end NUMINAMATH_CALUDE_a_range_when_p_false_l2020_202027


namespace NUMINAMATH_CALUDE_lp_has_only_minimum_l2020_202071

/-- A point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The feasible region defined by the constraints -/
def FeasibleRegion (p : Point) : Prop :=
  6 * p.x + 3 * p.y < 15 ∧ p.y ≤ p.x + 1 ∧ p.x - 5 * p.y ≤ 3

/-- The objective function -/
def ObjectiveFunction (p : Point) : ℝ :=
  3 * p.x + 5 * p.y

/-- The statement that the linear programming problem has only a minimum value and no maximum value -/
theorem lp_has_only_minimum :
  (∃ (p : Point), FeasibleRegion p ∧
    ∀ (q : Point), FeasibleRegion q → ObjectiveFunction p ≤ ObjectiveFunction q) ∧
  (¬ ∃ (p : Point), FeasibleRegion p ∧
    ∀ (q : Point), FeasibleRegion q → ObjectiveFunction q ≤ ObjectiveFunction p) :=
sorry

end NUMINAMATH_CALUDE_lp_has_only_minimum_l2020_202071


namespace NUMINAMATH_CALUDE_a_range_l2020_202062

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > a then x + 2 else x^2 + 5*x + 2

-- Define g(x) in terms of f(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 2*x

-- Define a predicate for g having exactly three distinct zeros
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    g a x = 0 ∧ g a y = 0 ∧ g a z = 0 ∧
    ∀ w : ℝ, g a w = 0 → w = x ∨ w = y ∨ w = z

-- The main theorem
theorem a_range (a : ℝ) :
  has_three_distinct_zeros a ↔ a ∈ Set.Icc (-1 : ℝ) 2 ∧ a ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_a_range_l2020_202062


namespace NUMINAMATH_CALUDE_pauls_peaches_l2020_202041

/-- Given that Audrey has 26 peaches and the difference between Audrey's and Paul's peaches is 22,
    prove that Paul has 4 peaches. -/
theorem pauls_peaches (audrey_peaches : ℕ) (peach_difference : ℕ) 
    (h1 : audrey_peaches = 26)
    (h2 : peach_difference = 22)
    (h3 : audrey_peaches - paul_peaches = peach_difference) : 
    paul_peaches = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_pauls_peaches_l2020_202041


namespace NUMINAMATH_CALUDE_sqrt_prime_irrational_l2020_202021

theorem sqrt_prime_irrational (p : ℕ) (h : Prime p) : Irrational (Real.sqrt p) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_prime_irrational_l2020_202021


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l2020_202049

/-- Represents the number of male students in a stratified sample -/
def male_students_in_sample (total_male : ℕ) (total_female : ℕ) (sample_size : ℕ) : ℕ :=
  (total_male * sample_size) / (total_male + total_female)

/-- Theorem: In a school with 560 male students and 420 female students,
    a stratified sample of 140 students will contain 80 male students -/
theorem stratified_sample_theorem :
  male_students_in_sample 560 420 140 = 80 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l2020_202049


namespace NUMINAMATH_CALUDE_sqrt_of_sixteen_l2020_202090

theorem sqrt_of_sixteen : ∃ (x : ℝ), x^2 = 16 ∧ (x = 4 ∨ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sixteen_l2020_202090


namespace NUMINAMATH_CALUDE_ace_diamond_king_probability_l2020_202070

-- Define the structure of a standard deck
def StandardDeck : Type := Fin 52

-- Define the properties of cards
def isAce : StandardDeck → Prop := sorry
def isDiamond : StandardDeck → Prop := sorry
def isKing : StandardDeck → Prop := sorry

-- Define the draw function
def draw : ℕ → (StandardDeck → Prop) → ℚ := sorry

-- Theorem statement
theorem ace_diamond_king_probability :
  draw 1 isAce * draw 2 isDiamond * draw 3 isKing = 1 / 663 := by sorry

end NUMINAMATH_CALUDE_ace_diamond_king_probability_l2020_202070


namespace NUMINAMATH_CALUDE_carols_birthday_invitations_l2020_202088

/-- Given that Carol buys invitation packages with 2 invitations each and needs 5 packs,
    prove that she is inviting 10 friends. -/
theorem carols_birthday_invitations
  (invitations_per_pack : ℕ)
  (packs_needed : ℕ)
  (h1 : invitations_per_pack = 2)
  (h2 : packs_needed = 5) :
  invitations_per_pack * packs_needed = 10 := by
  sorry

end NUMINAMATH_CALUDE_carols_birthday_invitations_l2020_202088


namespace NUMINAMATH_CALUDE_hotel_charge_difference_l2020_202000

theorem hotel_charge_difference (G R P : ℝ) 
  (hR : R = G * 3.0000000000000006)
  (hP : P = G * 0.9) :
  (R - P) / R * 100 = 70 := by sorry

end NUMINAMATH_CALUDE_hotel_charge_difference_l2020_202000


namespace NUMINAMATH_CALUDE_subset_condition_1_subset_condition_2_l2020_202054

-- Define the sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 7}
def B (a : ℝ) : Set ℝ := {x | 3 - 2*a ≤ x ∧ x ≤ 2*a - 5}

-- Theorem for part 1
theorem subset_condition_1 (a : ℝ) : A ⊆ B a ↔ a ≥ 6 := by sorry

-- Theorem for part 2
theorem subset_condition_2 (a : ℝ) : B a ⊆ A ↔ 2 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_1_subset_condition_2_l2020_202054


namespace NUMINAMATH_CALUDE_expansion_binomial_coefficients_l2020_202026

theorem expansion_binomial_coefficients (n : ℕ) : 
  (∃ a d : ℚ, (n.choose 1 : ℚ) = a ∧ 
               (n.choose 2 : ℚ) = a + d ∧ 
               (n.choose 3 : ℚ) = a + 2*d) → 
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_expansion_binomial_coefficients_l2020_202026


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2020_202030

theorem geometric_sequence_seventh_term 
  (a₁ : ℝ) 
  (a₃ : ℝ) 
  (h₁ : a₁ = 3) 
  (h₃ : a₃ = 1/9) : 
  let r := (a₃ / a₁) ^ (1/2)
  a₁ * r^6 = Real.sqrt 3 / 81 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2020_202030


namespace NUMINAMATH_CALUDE_function_value_at_four_l2020_202034

/-- Given a function f where f(2x) = 3x^2 + 1 for all x, prove that f(4) = 13 -/
theorem function_value_at_four (f : ℝ → ℝ) (h : ∀ x, f (2 * x) = 3 * x^2 + 1) : f 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_four_l2020_202034


namespace NUMINAMATH_CALUDE_rabbits_ate_23_pumpkins_l2020_202048

/-- The number of pumpkins Sara initially grew -/
def initial_pumpkins : ℕ := 43

/-- The number of pumpkins Sara has left -/
def remaining_pumpkins : ℕ := 20

/-- The number of pumpkins eaten by rabbits -/
def eaten_pumpkins : ℕ := initial_pumpkins - remaining_pumpkins

theorem rabbits_ate_23_pumpkins : eaten_pumpkins = 23 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_ate_23_pumpkins_l2020_202048


namespace NUMINAMATH_CALUDE_divide_42_problem_l2020_202003

theorem divide_42_problem (x : ℚ) (h : 35 / x = 5) : 42 / x = 6 := by
  sorry

end NUMINAMATH_CALUDE_divide_42_problem_l2020_202003


namespace NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l2020_202047

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -1

-- Define a circle with center at the origin
def circle_at_origin (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Theorem statement
theorem circle_tangent_to_parabola_directrix :
  ∀ (x y r : ℝ),
  (∃ (x_d : ℝ), directrix x_d ∧ 
    (∀ (x_p y_p : ℝ), parabola x_p y_p → x_p ≥ x_d) ∧
    (∃ (x_t y_t : ℝ), parabola x_t y_t ∧ x_t = x_d ∧ 
      circle_at_origin x_t y_t r)) →
  circle_at_origin x y 1 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l2020_202047


namespace NUMINAMATH_CALUDE_rice_purchase_problem_l2020_202076

/-- The problem of determining the amount of rice bought given the prices and quantities of different grains --/
theorem rice_purchase_problem (rice_price corn_price beans_price : ℚ)
  (total_weight total_cost : ℚ) (beans_weight : ℚ) :
  rice_price = 75 / 100 →
  corn_price = 110 / 100 →
  beans_price = 55 / 100 →
  total_weight = 36 →
  total_cost = 2835 / 100 →
  beans_weight = 8 →
  ∃ (rice_weight : ℚ), 
    (rice_weight + (total_weight - rice_weight - beans_weight) + beans_weight = total_weight) ∧
    (rice_price * rice_weight + corn_price * (total_weight - rice_weight - beans_weight) + beans_price * beans_weight = total_cost) ∧
    (abs (rice_weight - 196 / 10) < 1 / 10) :=
by sorry

end NUMINAMATH_CALUDE_rice_purchase_problem_l2020_202076


namespace NUMINAMATH_CALUDE_triangle_properties_l2020_202031

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_A : 0 < A
  pos_B : 0 < B
  pos_C : 0 < C
  sum_angles : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem triangle_properties (t : Triangle) :
  (t.c^2 + t.a*t.b = t.c*(t.a*Real.cos t.B - t.b*Real.cos t.A) + 2*t.b^2 → t.C = π/3) ∧
  (t.C = π/3 ∧ t.c = 2*Real.sqrt 3 → -2*Real.sqrt 3 < 4*Real.sin t.B - t.a ∧ 4*Real.sin t.B - t.a < 2*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2020_202031


namespace NUMINAMATH_CALUDE_leftmost_box_value_l2020_202043

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i, i < n - 2 → a i + a (i + 1) + a (i + 2) = 2005

theorem leftmost_box_value (a : ℕ → ℕ) :
  sequence_sum a 9 →
  a 1 = 888 →
  a 2 = 999 →
  a 0 = 118 :=
by sorry

end NUMINAMATH_CALUDE_leftmost_box_value_l2020_202043


namespace NUMINAMATH_CALUDE_find_original_number_l2020_202010

theorem find_original_number : ∃ x : ℕ, 
  (x : ℚ) / 25 * 85 = x * 67 / 25 + 3390 ∧ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_find_original_number_l2020_202010


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2020_202059

theorem inverse_variation_problem (x y : ℝ) : 
  (x > 0) →
  (y > 0) →
  (∃ k : ℝ, ∀ x y, x^3 * y = k) →
  (2^3 * 8 = x^3 * 512) →
  (y = 512) →
  x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2020_202059


namespace NUMINAMATH_CALUDE_min_width_rectangle_l2020_202086

/-- Given a rectangular area with length 20 ft longer than the width,
    and an area of at least 150 sq. ft, the minimum possible width is 10 ft. -/
theorem min_width_rectangle (w : ℝ) (h1 : w > 0) : 
  w * (w + 20) ≥ 150 → w ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_min_width_rectangle_l2020_202086


namespace NUMINAMATH_CALUDE_negation_equivalence_l2020_202011

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 = 1) ↔ (∀ x : ℝ, x^2 ≠ 1) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2020_202011


namespace NUMINAMATH_CALUDE_unique_solution_system_l2020_202006

theorem unique_solution_system (x y : ℝ) :
  (2 * x + y + 8 ≤ 0) ∧
  (x^4 + 2 * x^2 * y^2 + y^4 + 9 - 10 * x^2 - 10 * y^2 = 8 * x * y) →
  x = -3 ∧ y = -2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2020_202006


namespace NUMINAMATH_CALUDE_expression_value_l2020_202067

theorem expression_value (x y : ℝ) (h : x - 3*y = 4) :
  (x - 3*y)^2 + 2*x - 6*y - 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2020_202067


namespace NUMINAMATH_CALUDE_expression_evaluation_l2020_202051

/-- Proves that the given expression evaluates to 58.51045 -/
theorem expression_evaluation :
  (3.415 * 2.67) + (8.641 - 1.23) / (0.125 * 4.31) + 5.97^2 = 58.51045 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2020_202051


namespace NUMINAMATH_CALUDE_unique_pair_existence_l2020_202065

theorem unique_pair_existence :
  ∃! (c d : ℝ), 0 < c ∧ c < d ∧ d < π / 2 ∧
    Real.sin (Real.cos c) = c ∧ Real.cos (Real.sin d) = d := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_existence_l2020_202065


namespace NUMINAMATH_CALUDE_definite_integral_equals_26_over_3_l2020_202079

theorem definite_integral_equals_26_over_3 : 
  ∫ x in (0)..(2 * Real.arctan (1/2)), (1 + Real.sin x) / ((1 - Real.sin x)^2) = 26/3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_equals_26_over_3_l2020_202079


namespace NUMINAMATH_CALUDE_seventh_twenty_ninth_712th_digit_l2020_202004

def decimal_representation (n d : ℕ) : List ℕ :=
  sorry

theorem seventh_twenty_ninth_712th_digit :
  let repr := decimal_representation 7 29
  let cycle_length := 29
  let digit_position := 712 % cycle_length
  List.get! repr digit_position = 1 := by
  sorry

end NUMINAMATH_CALUDE_seventh_twenty_ninth_712th_digit_l2020_202004


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l2020_202036

theorem roots_of_quadratic (x y : ℝ) : 
  x + y = 10 ∧ |x - y| = 6 → 
  x^2 - 10*x + 16 = 0 ∧ y^2 - 10*y + 16 = 0 := by
sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l2020_202036


namespace NUMINAMATH_CALUDE_cube_decomposition_smallest_l2020_202035

theorem cube_decomposition_smallest (m : ℕ) (h1 : m ≥ 2) : 
  (m^2 - m + 1 = 73) → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_decomposition_smallest_l2020_202035


namespace NUMINAMATH_CALUDE_probability_of_2500_is_6_125_l2020_202066

/-- The number of outcomes on the wheel -/
def num_outcomes : ℕ := 5

/-- The number of spins -/
def num_spins : ℕ := 3

/-- The number of ways to achieve the desired sum -/
def num_successful_combinations : ℕ := 6

/-- The total number of possible outcomes after three spins -/
def total_possibilities : ℕ := num_outcomes ^ num_spins

/-- The probability of earning exactly $2500 in three spins -/
def probability_of_2500 : ℚ := num_successful_combinations / total_possibilities

theorem probability_of_2500_is_6_125 : probability_of_2500 = 6 / 125 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_2500_is_6_125_l2020_202066


namespace NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_3_l2020_202040

theorem infinitely_many_primes_4k_plus_3 :
  ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4 * k + 3) →
  ∃ q, Nat.Prime q ∧ (∃ m, q = 4 * m + 3) ∧ q ∉ S :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_3_l2020_202040


namespace NUMINAMATH_CALUDE_sqrt5_and_sequences_l2020_202087

-- Define p-arithmetic
structure PArithmetic where
  p : ℕ
  -- Add more properties as needed

-- Define the concept of "extracting √5"
def can_extract_sqrt5 (pa : PArithmetic) : Prop := sorry

-- Define a sequence type
def Sequence (α : Type) := ℕ → α

-- Define properties for Fibonacci and geometric sequences
def is_fibonacci {α : Type} [Add α] (seq : Sequence α) : Prop := 
  ∀ n, seq (n + 2) = seq (n + 1) + seq n

def is_geometric {α : Type} [Mul α] (seq : Sequence α) : Prop := 
  ∃ r, ∀ n, seq (n + 1) = r * seq n

-- Main theorem
theorem sqrt5_and_sequences (pa : PArithmetic) :
  (¬ can_extract_sqrt5 pa → 
    ¬ ∃ (seq : Sequence ℚ), is_fibonacci seq ∧ is_geometric seq) ∧
  (can_extract_sqrt5 pa → 
    (∃ (seq : Sequence ℚ), is_fibonacci seq ∧ is_geometric seq) ∧
    (∀ (fib : Sequence ℚ), is_fibonacci fib → 
      ∃ (seq1 seq2 : Sequence ℚ), 
        is_fibonacci seq1 ∧ is_geometric seq1 ∧
        is_fibonacci seq2 ∧ is_geometric seq2 ∧
        (∀ n, fib n = seq1 n + seq2 n))) :=
by sorry

end NUMINAMATH_CALUDE_sqrt5_and_sequences_l2020_202087


namespace NUMINAMATH_CALUDE_total_oranges_in_boxes_l2020_202022

def box1_capacity : ℕ := 80
def box2_capacity : ℕ := 50
def box1_fill_ratio : ℚ := 3/4
def box2_fill_ratio : ℚ := 3/5

theorem total_oranges_in_boxes :
  (↑box1_capacity * box1_fill_ratio).floor + (↑box2_capacity * box2_fill_ratio).floor = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_in_boxes_l2020_202022


namespace NUMINAMATH_CALUDE_exists_six_digit_number_without_identical_endings_l2020_202083

theorem exists_six_digit_number_without_identical_endings : ∃ A : ℕ, 
  (100000 ≤ A ∧ A < 1000000) ∧ 
  ∀ k : ℕ, k ≤ 500000 → ∀ d : ℕ, d < 10 → 
    (k * A) % 1000000 ≠ d * 111111 := by
  sorry

end NUMINAMATH_CALUDE_exists_six_digit_number_without_identical_endings_l2020_202083


namespace NUMINAMATH_CALUDE_min_diff_y_x_l2020_202046

theorem min_diff_y_x (x y z : ℤ) : 
  x < y ∧ y < z ∧ 
  Even x ∧ Odd y ∧ Odd z ∧
  (∀ w, w - x ≥ 9 → z ≤ w) →
  ∃ (d : ℤ), d = y - x ∧ (∀ d' : ℤ, d' = y - x → d ≤ d') ∧ d = 1 := by
sorry

end NUMINAMATH_CALUDE_min_diff_y_x_l2020_202046


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2020_202042

theorem polynomial_factorization (a x y : ℝ) : 3*a*x^2 + 6*a*x*y + 3*a*y^2 = 3*a*(x+y)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2020_202042


namespace NUMINAMATH_CALUDE_system_solution_l2020_202044

theorem system_solution : ∃ (x y z : ℚ), 
  (7 * x + 3 * y = z - 10) ∧ 
  (2 * x - 4 * y = 3 * z + 20) := by
  use 0, -50/13, -20/13
  sorry

end NUMINAMATH_CALUDE_system_solution_l2020_202044


namespace NUMINAMATH_CALUDE_sin_15_minus_sin_75_fourth_power_l2020_202085

theorem sin_15_minus_sin_75_fourth_power :
  Real.sin (15 * π / 180) ^ 4 - Real.sin (75 * π / 180) ^ 4 = -(Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_minus_sin_75_fourth_power_l2020_202085


namespace NUMINAMATH_CALUDE_cost_of_450_candies_l2020_202099

/-- The cost of buying a specific number of chocolate candies -/
def cost_of_candies (candies_per_box : ℕ) (cost_per_box : ℚ) (total_candies : ℕ) : ℚ :=
  (total_candies / candies_per_box) * cost_per_box

/-- Theorem stating the cost of 450 chocolate candies -/
theorem cost_of_450_candies :
  cost_of_candies 30 (7.5 : ℚ) 450 = 112.5 := by
  sorry

#eval cost_of_candies 30 (7.5 : ℚ) 450

end NUMINAMATH_CALUDE_cost_of_450_candies_l2020_202099


namespace NUMINAMATH_CALUDE_field_area_l2020_202052

/-- The area of a rectangular field with specific fencing conditions -/
theorem field_area (L W : ℝ) : 
  L = 20 →  -- One side is 20 feet
  2 * W + L = 41 →  -- Total fencing is 41 feet
  L * W = 210 :=  -- Area of the field
by
  sorry

end NUMINAMATH_CALUDE_field_area_l2020_202052


namespace NUMINAMATH_CALUDE_distinct_selections_count_l2020_202024

/-- Represents the counts of each letter in "MATHEMATICAL" --/
structure LetterCounts where
  a : Nat
  e : Nat
  i : Nat
  m : Nat
  t : Nat
  h : Nat
  c : Nat
  l : Nat

/-- The initial letter counts in "MATHEMATICAL" --/
def initial_counts : LetterCounts := {
  a := 3, e := 1, i := 1, m := 2, t := 2, h := 1, c := 1, l := 1
}

/-- Counts the number of distinct ways to choose 3 vowels and 4 consonants
    from the word "MATHEMATICAL" with indistinguishable T's, M's, and A's --/
def count_distinct_selections (counts : LetterCounts) : Nat :=
  sorry

theorem distinct_selections_count :
  count_distinct_selections initial_counts = 64 := by
  sorry

end NUMINAMATH_CALUDE_distinct_selections_count_l2020_202024


namespace NUMINAMATH_CALUDE_existence_of_prime_q_l2020_202096

theorem existence_of_prime_q (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ ∀ n : ℕ, n > 0 → ¬(q ∣ (n^p - p)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_prime_q_l2020_202096


namespace NUMINAMATH_CALUDE_subtraction_result_l2020_202037

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_valid_arrangement (a b c d e f g h i j : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧ is_valid_digit e ∧
  is_valid_digit f ∧ is_valid_digit g ∧ is_valid_digit h ∧ is_valid_digit i ∧ is_valid_digit j ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

theorem subtraction_result 
  (a b c d e f g h i j : ℕ) 
  (h1 : is_valid_arrangement a b c d e f g h i j)
  (h2 : a = 6)
  (h3 : b = 1) :
  61000 + c * 1000 + d * 100 + e * 10 + f - (g * 10000 + h * 1000 + i * 100 + j * 10 + a) = 59387 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l2020_202037


namespace NUMINAMATH_CALUDE_volume_of_specific_prism_l2020_202098

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculate the volume of a prism formed by slicing a rectangular solid -/
def volumeOfSlicedPrism (solid : RectangularSolid) (plane : Plane3D) (p1 p2 p3 vertex : Point3D) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific prism -/
theorem volume_of_specific_prism :
  let solid := RectangularSolid.mk 4 3 3
  let p1 := Point3D.mk 0 0 3
  let p2 := Point3D.mk 4 0 3
  let p3 := Point3D.mk 0 3 1.5
  let vertex := Point3D.mk 4 3 0
  let plane := Plane3D.mk (-0.75) 0.75 1 (-3)
  volumeOfSlicedPrism solid plane p1 p2 p3 vertex = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_prism_l2020_202098


namespace NUMINAMATH_CALUDE_lcm_eight_fifteen_l2020_202080

theorem lcm_eight_fifteen : Nat.lcm 8 15 = 120 := by sorry

end NUMINAMATH_CALUDE_lcm_eight_fifteen_l2020_202080


namespace NUMINAMATH_CALUDE_only_expr2_same_type_as_reference_l2020_202063

-- Define the structure of a monomial expression
structure Monomial (α : Type*) :=
  (coeff : ℤ)
  (vars : List (α × ℕ))

-- Define a function to check if two monomials have the same type
def same_type {α : Type*} (m1 m2 : Monomial α) : Prop :=
  m1.vars = m2.vars

-- Define the reference monomial -3a²b
def reference : Monomial Char :=
  ⟨-3, [('a', 2), ('b', 1)]⟩

-- Define the given expressions
def expr1 : Monomial Char := ⟨-3, [('a', 1), ('b', 2)]⟩  -- -3ab²
def expr2 : Monomial Char := ⟨-1, [('b', 1), ('a', 2)]⟩  -- -ba²
def expr3 : Monomial Char := ⟨2, [('a', 1), ('b', 2)]⟩   -- 2ab²
def expr4 : Monomial Char := ⟨2, [('a', 3), ('b', 1)]⟩   -- 2a³b

-- Theorem to prove
theorem only_expr2_same_type_as_reference :
  (¬ same_type reference expr1) ∧
  (same_type reference expr2) ∧
  (¬ same_type reference expr3) ∧
  (¬ same_type reference expr4) :=
sorry

end NUMINAMATH_CALUDE_only_expr2_same_type_as_reference_l2020_202063


namespace NUMINAMATH_CALUDE_series_sum_equals_half_l2020_202032

/-- The sum of the series Σ(2^n / (3^(2^n) + 1)) from n = 0 to infinity is equal to 1/2 -/
theorem series_sum_equals_half :
  ∑' n : ℕ, (2 : ℝ)^n / (3^(2^n) + 1) = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_half_l2020_202032


namespace NUMINAMATH_CALUDE_no_natural_solution_l2020_202069

theorem no_natural_solution (x y z : ℕ) : 
  (x : ℚ) / y + (y : ℚ) / z + (z : ℚ) / x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l2020_202069


namespace NUMINAMATH_CALUDE_arccos_of_neg_one_equals_pi_l2020_202093

theorem arccos_of_neg_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_of_neg_one_equals_pi_l2020_202093


namespace NUMINAMATH_CALUDE_min_value_xy_l2020_202012

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y + 6 = x*y) : 
  x*y ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_l2020_202012


namespace NUMINAMATH_CALUDE_frances_pencil_collection_l2020_202017

/-- The number of groups of pencils in Frances's collection -/
def num_groups : ℕ := 5

/-- The number of pencils in each group -/
def pencils_per_group : ℕ := 5

/-- The total number of pencils in Frances's collection -/
def total_pencils : ℕ := num_groups * pencils_per_group

theorem frances_pencil_collection : total_pencils = 25 := by
  sorry

end NUMINAMATH_CALUDE_frances_pencil_collection_l2020_202017


namespace NUMINAMATH_CALUDE_solution_l2020_202055

def problem (x y a : ℚ) : Prop :=
  (1 / 5) * x = (5 / 8) * y ∧
  y = 40 ∧
  x + a = 4 * y

theorem solution : ∃ x y a : ℚ, problem x y a ∧ a = 35 := by
  sorry

end NUMINAMATH_CALUDE_solution_l2020_202055


namespace NUMINAMATH_CALUDE_book_reading_time_l2020_202050

theorem book_reading_time (pages_per_book : ℕ) (pages_per_day : ℕ) (days_to_finish : ℕ) : 
  pages_per_book = 249 → pages_per_day = 83 → days_to_finish = 3 →
  pages_per_book = days_to_finish * pages_per_day :=
by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l2020_202050


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2020_202064

def A : Set (ℝ × ℝ) := {p | p.2 = 3 * p.1 - 2}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1 ^ 2}

theorem intersection_of_A_and_B :
  A ∩ B = {(1, 1), (2, 4)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2020_202064


namespace NUMINAMATH_CALUDE_not_divisible_by_4_8_16_32_l2020_202091

def x : ℕ := 80 + 112 + 144 + 176 + 304 + 368 + 3248 + 17

theorem not_divisible_by_4_8_16_32 : 
  ¬(∃ k : ℕ, x = 4 * k) ∧ 
  ¬(∃ k : ℕ, x = 8 * k) ∧ 
  ¬(∃ k : ℕ, x = 16 * k) ∧ 
  ¬(∃ k : ℕ, x = 32 * k) :=
by sorry

end NUMINAMATH_CALUDE_not_divisible_by_4_8_16_32_l2020_202091


namespace NUMINAMATH_CALUDE_rhombus_common_area_l2020_202033

/-- Represents a rhombus with given diagonal lengths -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Calculates the area of the common part of two rhombuses -/
def commonArea (r : Rhombus) : ℝ :=
  -- Implementation details are omitted
  sorry

/-- Theorem: The area of the common part of two rhombuses is 9.6 cm² -/
theorem rhombus_common_area :
  let r := Rhombus.mk 4 6
  commonArea r = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_common_area_l2020_202033


namespace NUMINAMATH_CALUDE_solve_strawberry_problem_l2020_202094

def strawberry_problem (betty_strawberries : ℕ) (matthew_extra : ℕ) (jar_strawberries : ℕ) (total_money : ℕ) : Prop :=
  let matthew_strawberries := betty_strawberries + matthew_extra
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let num_jars := total_strawberries / jar_strawberries
  let price_per_jar := total_money / num_jars
  price_per_jar = 4

theorem solve_strawberry_problem :
  strawberry_problem 16 20 7 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_strawberry_problem_l2020_202094


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2020_202045

theorem cyclic_sum_inequality (k : ℕ) (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum : x + y + z = 1) : 
  (x^(k+2) / (x^(k+1) + y^k + z^k)) + 
  (y^(k+2) / (y^(k+1) + z^k + x^k)) + 
  (z^(k+2) / (z^(k+1) + x^k + y^k)) ≥ 1/7 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2020_202045


namespace NUMINAMATH_CALUDE_puppy_weight_l2020_202007

/-- Given the weights of a puppy and two cats satisfying certain conditions,
    prove that the puppy weighs 12 pounds. -/
theorem puppy_weight (a b c : ℝ) 
    (h1 : a + b + c = 36)
    (h2 : a + c = 3 * b)
    (h3 : a + b = c + 6) :
    a = 12 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_l2020_202007


namespace NUMINAMATH_CALUDE_complex_conversion_l2020_202014

theorem complex_conversion :
  (2 * Real.sqrt 3) * Complex.exp (Complex.I * (17 * Real.pi / 6)) = -3 + Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_conversion_l2020_202014


namespace NUMINAMATH_CALUDE_x_equals_zero_l2020_202018

theorem x_equals_zero (a : ℝ) (x : ℝ) 
  (h1 : a > 0) 
  (h2 : (10 : ℝ) ^ x = Real.log (10 * a) + Real.log (a⁻¹)) : 
  x = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_zero_l2020_202018


namespace NUMINAMATH_CALUDE_family_trip_eggs_l2020_202008

theorem family_trip_eggs (adults girls : ℕ) (total_eggs : ℕ) : 
  adults = 3 →
  girls = 7 →
  total_eggs = 36 →
  ∃ (boys : ℕ), 
    adults * 3 + girls * 1 + boys * 2 = total_eggs ∧
    boys = 10 :=
by sorry

end NUMINAMATH_CALUDE_family_trip_eggs_l2020_202008


namespace NUMINAMATH_CALUDE_petyas_sum_l2020_202038

theorem petyas_sum (n k : ℕ) : 
  (∀ i ∈ Finset.range (k + 1), Even (n + 2 * i)) →
  ((k + 1) * (n + k) = 30 * (n + 2 * k)) →
  ((k + 1) * (n + k) = 90 * n) →
  n = 44 ∧ k = 44 :=
by sorry

end NUMINAMATH_CALUDE_petyas_sum_l2020_202038


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2020_202068

open Real

theorem necessary_but_not_sufficient 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) : 
  (a > b ∧ b > ℯ → a^b < b^a) ∧
  ¬(a^b < b^a → a > b ∧ b > ℯ) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2020_202068


namespace NUMINAMATH_CALUDE_no_sequences_exist_l2020_202089

theorem no_sequences_exist : ¬ ∃ (a b : ℕ → ℝ), 
  (∀ n : ℕ, (3/2) * Real.pi ≤ a n ∧ a n ≤ b n) ∧
  (∀ n : ℕ, ∀ x : ℝ, 0 < x ∧ x < 1 → Real.cos (a n * x) + Real.cos (b n * x) ≥ -1 / n) := by
  sorry

end NUMINAMATH_CALUDE_no_sequences_exist_l2020_202089


namespace NUMINAMATH_CALUDE_equal_volumes_l2020_202095

-- Define the tetrahedrons
structure Tetrahedron :=
  (a b c d e f : ℝ)

-- Define the volumes of the tetrahedrons
def volume (t : Tetrahedron) : ℝ := sorry

-- Define the specific tetrahedrons
def ABCD : Tetrahedron :=
  { a := 13, b := 5, c := 12, d := 13, e := 6, f := 5 }

def EFGH : Tetrahedron :=
  { a := 13, b := 13, c := 8, d := 5, e := 12, f := 5 }

-- Theorem statement
theorem equal_volumes : volume ABCD = volume EFGH := by
  sorry

end NUMINAMATH_CALUDE_equal_volumes_l2020_202095


namespace NUMINAMATH_CALUDE_no_solution_for_modified_problem_l2020_202061

theorem no_solution_for_modified_problem (r : ℝ) : 
  ¬∃ (a h : ℝ), 
    (0 < r) ∧ 
    (0 < a) ∧ (a ≤ 2*r) ∧ 
    (0 < h) ∧ (h < 2*r) ∧ 
    (a + h = 2*Real.pi*r) := by
  sorry


end NUMINAMATH_CALUDE_no_solution_for_modified_problem_l2020_202061


namespace NUMINAMATH_CALUDE_noa_score_l2020_202056

/-- Proves that Noa scored 30 points given the conditions of the problem -/
theorem noa_score (noa_score : ℕ) (phillip_score : ℕ) : 
  phillip_score = 2 * noa_score →
  noa_score + phillip_score = 90 →
  noa_score = 30 := by
sorry

end NUMINAMATH_CALUDE_noa_score_l2020_202056


namespace NUMINAMATH_CALUDE_average_equation_solution_l2020_202092

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((2*x + 5) + (8*x + 3) + (3*x + 8)) = 5*x - 10 → x = 23 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l2020_202092


namespace NUMINAMATH_CALUDE_caterer_order_conditions_caterer_order_underdetermined_l2020_202057

/-- Represents the order of a caterer -/
structure CatererOrder where
  x : ℝ  -- number of ice-cream bars
  y : ℝ  -- number of sundaes
  z : ℝ  -- number of milkshakes
  m : ℝ  -- price of each milkshake

/-- Theorem stating the conditions of the caterer's order -/
theorem caterer_order_conditions (order : CatererOrder) : Prop :=
  0.60 * order.x + 1.20 * order.y + order.m * order.z = 425 ∧
  order.x + order.y + order.z = 350

/-- Theorem stating that the conditions do not uniquely determine the order -/
theorem caterer_order_underdetermined :
  ∃ (order1 order2 : CatererOrder),
    caterer_order_conditions order1 ∧
    caterer_order_conditions order2 ∧
    order1 ≠ order2 := by
  sorry

end NUMINAMATH_CALUDE_caterer_order_conditions_caterer_order_underdetermined_l2020_202057


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l2020_202029

theorem not_sufficient_nor_necessary : 
  ¬(∀ x : ℝ, x < 0 → Real.log (x + 1) ≤ 0) ∧ 
  ¬(∀ x : ℝ, Real.log (x + 1) ≤ 0 → x < 0) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l2020_202029


namespace NUMINAMATH_CALUDE_smallest_positive_integer_2016m_45000n_l2020_202075

theorem smallest_positive_integer_2016m_45000n :
  ∃ (k : ℕ), k > 0 ∧
  (∃ (m n : ℤ), k = 2016 * m + 45000 * n) ∧
  (∀ (j : ℕ), j > 0 → (∃ (x y : ℤ), j = 2016 * x + 45000 * y) → j ≥ k) ∧
  k = 24 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_2016m_45000n_l2020_202075


namespace NUMINAMATH_CALUDE_base10_729_equals_base7_261_l2020_202009

-- Define a function to convert a base-7 number to base-10
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

-- Define the base-7 representation of 261₇
def base7_261 : List Nat := [1, 6, 2]

-- Theorem statement
theorem base10_729_equals_base7_261 :
  base7ToBase10 base7_261 = 729 := by
  sorry

end NUMINAMATH_CALUDE_base10_729_equals_base7_261_l2020_202009


namespace NUMINAMATH_CALUDE_smallest_lucky_number_unique_lucky_number_divisible_by_seven_l2020_202060

/-- Definition of a lucky number -/
def is_lucky_number (M : ℕ) : Prop :=
  ∃ (A B : ℕ), 
    M % 10 ≠ 0 ∧
    M = A * B ∧
    A ≥ B ∧
    10 ≤ A ∧ A < 100 ∧
    10 ≤ B ∧ B < 100 ∧
    (A / 10) = (B / 10) ∧
    (A % 10) + (B % 10) = 6

/-- The smallest lucky number is 165 -/
theorem smallest_lucky_number : 
  is_lucky_number 165 ∧ ∀ M, is_lucky_number M → M ≥ 165 := by sorry

/-- There exists a unique lucky number M such that (A + B) / (A - B) is divisible by 7, and it equals 3968 -/
theorem unique_lucky_number_divisible_by_seven :
  ∃! M, is_lucky_number M ∧ 
    (∃ A B, M = A * B ∧ ((A + B) / (A - B)) % 7 = 0) ∧
    M = 3968 := by sorry

end NUMINAMATH_CALUDE_smallest_lucky_number_unique_lucky_number_divisible_by_seven_l2020_202060


namespace NUMINAMATH_CALUDE_preston_high_teachers_l2020_202077

/-- Represents the number of students in Preston High School -/
def total_students : ℕ := 1500

/-- Represents the number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- Represents the number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- Represents the number of students in each class -/
def students_per_class : ℕ := 30

/-- Calculates the number of teachers at Preston High School -/
def number_of_teachers : ℕ :=
  (total_students * classes_per_student) / (students_per_class * classes_per_teacher)

/-- Theorem stating that the number of teachers at Preston High School is 60 -/
theorem preston_high_teachers :
  number_of_teachers = 60 := by sorry

end NUMINAMATH_CALUDE_preston_high_teachers_l2020_202077


namespace NUMINAMATH_CALUDE_expand_expression_l2020_202019

theorem expand_expression (x : ℝ) : (x + 2) * (3 * x - 6) = 3 * x^2 - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2020_202019


namespace NUMINAMATH_CALUDE_steves_speed_steves_speed_proof_l2020_202081

/-- Calculates Steve's speed during John's final push in a race --/
theorem steves_speed (john_initial_behind : ℝ) (john_speed : ℝ) (john_time : ℝ) (john_final_ahead : ℝ) : ℝ :=
  let john_distance := john_speed * john_time
  let steve_distance := john_distance - (john_initial_behind + john_final_ahead)
  steve_distance / john_time

/-- Proves that Steve's speed during John's final push was 3.8 m/s --/
theorem steves_speed_proof :
  steves_speed 15 4.2 42.5 2 = 3.8 := by
  sorry

end NUMINAMATH_CALUDE_steves_speed_steves_speed_proof_l2020_202081


namespace NUMINAMATH_CALUDE_sqrt_21_position_l2020_202013

theorem sqrt_21_position (n : ℕ) : 
  (∀ k : ℕ, k > 0 → ∃ a : ℝ, a = Real.sqrt (2 * k - 1)) → 
  Real.sqrt 21 = Real.sqrt (2 * 11 - 1) := by
sorry

end NUMINAMATH_CALUDE_sqrt_21_position_l2020_202013


namespace NUMINAMATH_CALUDE_distance_between_foci_l2020_202039

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y + 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24

-- Define the foci
def focus1 : ℝ × ℝ := (4, -5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem statement
theorem distance_between_foci :
  ∃ (x y : ℝ), ellipse_equation x y →
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = 2 * Real.sqrt 74 :=
sorry

end NUMINAMATH_CALUDE_distance_between_foci_l2020_202039


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_l2020_202020

/-- Given a sphere with surface area 256π cm², prove that the volume of a cylinder
    with the same radius as the sphere and height equal to the sphere's diameter
    is 1024π cm³. -/
theorem sphere_cylinder_volume (r : ℝ) (h : 4 * Real.pi * r^2 = 256 * Real.pi) :
  Real.pi * r^2 * (2 * r) = 1024 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_l2020_202020


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2020_202025

theorem unique_quadratic_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b ≠ 0 ∧ 
   ∃! x : ℝ, x^2 + (b + 3/b) * x + c = 0) ↔ 
  c = 3/2 := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2020_202025


namespace NUMINAMATH_CALUDE_equation_is_parabola_and_ellipse_l2020_202002

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents the equation y^4 - 6x^4 = 3y^2 + 1 -/
def equation (p : Point2D) : Prop :=
  p.y^4 - 6*p.x^4 = 3*p.y^2 + 1

/-- Represents a parabola in 2D space -/
def isParabola (S : Set Point2D) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ p ∈ S, p.y = a*p.x^2 + b*p.x + c

/-- Represents an ellipse in 2D space -/
def isEllipse (S : Set Point2D) : Prop :=
  ∃ h k a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ p ∈ S, 
    ((p.x - h)^2 / a^2) + ((p.y - k)^2 / b^2) = 1

/-- The set of all points satisfying the equation -/
def S : Set Point2D :=
  {p : Point2D | equation p}

/-- Theorem stating that the equation represents the union of a parabola and an ellipse -/
theorem equation_is_parabola_and_ellipse :
  ∃ P E : Set Point2D, isParabola P ∧ isEllipse E ∧ S = P ∪ E :=
sorry

end NUMINAMATH_CALUDE_equation_is_parabola_and_ellipse_l2020_202002


namespace NUMINAMATH_CALUDE_divisibility_properties_l2020_202001

theorem divisibility_properties (a b : ℤ) (k : ℕ) :
  (¬ ((a + b) ∣ (a^(2*k) + b^(2*k))) ∧ ¬ ((a - b) ∣ (a^(2*k) + b^(2*k)))) ∧
  ((a + b) ∣ (a^(2*k) - b^(2*k)) ∧ (a - b) ∣ (a^(2*k) - b^(2*k))) ∧
  ((a + b) ∣ (a^(2*k+1) + b^(2*k+1))) ∧
  ((a - b) ∣ (a^(2*k+1) - b^(2*k+1))) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_properties_l2020_202001


namespace NUMINAMATH_CALUDE_cryptarithm_unique_solution_l2020_202023

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Represents the cryptarithm KIC + KCI = ICK -/
def cryptarithm (K I C : Digit) : Prop :=
  100 * K.val + 10 * I.val + C.val +
  100 * K.val + 10 * C.val + I.val =
  100 * I.val + 10 * C.val + K.val

/-- The cryptarithm has a unique solution -/
theorem cryptarithm_unique_solution :
  ∃! (K I C : Digit), cryptarithm K I C ∧ K ≠ I ∧ K ≠ C ∧ I ≠ C ∧
  K.val = 4 ∧ I.val = 9 ∧ C.val = 5 := by sorry

end NUMINAMATH_CALUDE_cryptarithm_unique_solution_l2020_202023


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2020_202097

-- Define the repeating decimal 4.8̄
def repeating_decimal : ℚ := 4 + 8/9

-- Theorem to prove
theorem repeating_decimal_as_fraction : repeating_decimal = 44/9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2020_202097


namespace NUMINAMATH_CALUDE_hyperbola_t_squared_l2020_202084

-- Define a hyperbola
structure Hyperbola where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ
  vertical : Bool

-- Define a function to check if a point is on the hyperbola
def on_hyperbola (h : Hyperbola) (p : ℝ × ℝ) : Prop :=
  if h.vertical then
    (p.2 - h.center.2)^2 / h.b^2 - (p.1 - h.center.1)^2 / h.a^2 = 1
  else
    (p.1 - h.center.1)^2 / h.a^2 - (p.2 - h.center.2)^2 / h.b^2 = 1

-- Theorem statement
theorem hyperbola_t_squared (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_point1 : on_hyperbola h (4, -3))
  (h_point2 : on_hyperbola h (0, -2))
  (h_point3 : ∃ t : ℝ, on_hyperbola h (2, t)) :
  ∃ t : ℝ, on_hyperbola h (2, t) ∧ t^2 = 21/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_t_squared_l2020_202084


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l2020_202073

theorem sum_of_four_consecutive_even_integers :
  ∀ a : ℤ,
  (∃ b c d : ℤ, 
    b = a + 2 ∧ 
    c = a + 4 ∧ 
    d = a + 6 ∧ 
    a + d = 136) →
  a + (a + 2) + (a + 4) + (a + 6) = 272 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l2020_202073


namespace NUMINAMATH_CALUDE_sum_base4_to_base10_l2020_202028

/-- Converts a base 4 number represented as a list of digits to base 10 -/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The sum of 2213₄, 2703₄, and 1531₄ in base 10 is 309 -/
theorem sum_base4_to_base10 :
  base4ToBase10 [3, 1, 2, 2] + base4ToBase10 [3, 0, 7, 2] + base4ToBase10 [1, 3, 5, 1] = 309 := by
  sorry

#eval base4ToBase10 [3, 1, 2, 2] + base4ToBase10 [3, 0, 7, 2] + base4ToBase10 [1, 3, 5, 1]

end NUMINAMATH_CALUDE_sum_base4_to_base10_l2020_202028
