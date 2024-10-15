import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_by_120_l1098_109844

theorem divisibility_by_120 (n : ℕ) : ∃ k : ℤ, (n ^ 7 : ℤ) - (n ^ 3 : ℤ) = 120 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_120_l1098_109844


namespace NUMINAMATH_CALUDE_total_cost_rounded_to_18_l1098_109816

def item1 : ℚ := 247 / 100
def item2 : ℚ := 625 / 100
def item3 : ℚ := 876 / 100
def item4 : ℚ := 149 / 100

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

def total_cost : ℚ := item1 + item2 + item3 + item4

theorem total_cost_rounded_to_18 :
  round_to_nearest_dollar total_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_rounded_to_18_l1098_109816


namespace NUMINAMATH_CALUDE_marble_game_theorem_l1098_109815

/-- Represents the state of marbles for each player --/
structure MarbleState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Simulates one round of the game where the loser doubles the other players' marbles --/
def playRound (state : MarbleState) (loser : ℕ) : MarbleState :=
  match loser with
  | 1 => MarbleState.mk state.a (state.b * 3) (state.c * 3)
  | 2 => MarbleState.mk (state.a * 3) state.b (state.c * 3)
  | 3 => MarbleState.mk (state.a * 3) (state.b * 3) state.c
  | _ => state

/-- The main theorem statement --/
theorem marble_game_theorem :
  let initial_state := MarbleState.mk 165 57 21
  let after_round1 := playRound initial_state 1
  let after_round2 := playRound after_round1 2
  let final_state := playRound after_round2 3
  (after_round1.c = after_round1.a + 54) ∧
  (final_state.a = final_state.b) ∧
  (final_state.b = final_state.c) := by sorry

end NUMINAMATH_CALUDE_marble_game_theorem_l1098_109815


namespace NUMINAMATH_CALUDE_composition_equality_l1098_109862

theorem composition_equality (a : ℝ) (h1 : a > 1) : 
  let f (x : ℝ) := x^2 + 2
  let g (x : ℝ) := x^2 + 2
  f (g a) = 12 → a = Real.sqrt (Real.sqrt 10 - 2) := by
sorry

end NUMINAMATH_CALUDE_composition_equality_l1098_109862


namespace NUMINAMATH_CALUDE_inheritance_calculation_l1098_109851

theorem inheritance_calculation (inheritance : ℝ) : 
  (0.25 * inheritance + 0.15 * (inheritance - 0.25 * inheritance) = 20000) → 
  inheritance = 55172.41 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l1098_109851


namespace NUMINAMATH_CALUDE_cistern_length_l1098_109824

/-- The length of a cistern with given dimensions and wet surface area -/
theorem cistern_length (width : ℝ) (depth : ℝ) (wet_surface_area : ℝ) 
  (h1 : width = 2)
  (h2 : depth = 1.25)
  (h3 : wet_surface_area = 23) :
  ∃ length : ℝ, 
    wet_surface_area = length * width + 2 * length * depth + 2 * width * depth ∧ 
    length = 4 := by
  sorry

end NUMINAMATH_CALUDE_cistern_length_l1098_109824


namespace NUMINAMATH_CALUDE_cone_rolling_ratio_l1098_109834

/-- Represents a right circular cone. -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- Represents the rolling properties of the cone. -/
structure RollingCone extends RightCircularCone where
  rotations : ℕ
  no_slip : Bool

theorem cone_rolling_ratio (c : RollingCone) 
  (h_positive : c.h > 0)
  (r_positive : c.r > 0)
  (twenty_rotations : c.rotations = 20)
  (no_slip : c.no_slip = true) :
  c.h / c.r = Real.sqrt 399 :=
sorry

end NUMINAMATH_CALUDE_cone_rolling_ratio_l1098_109834


namespace NUMINAMATH_CALUDE_mark_has_six_parking_tickets_l1098_109842

/-- Represents the number of tickets for each person -/
structure Tickets where
  mark_parking : ℕ
  mark_speeding : ℕ
  sarah_parking : ℕ
  sarah_speeding : ℕ
  john_parking : ℕ
  john_speeding : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (t : Tickets) : Prop :=
  t.mark_parking + t.mark_speeding + t.sarah_parking + t.sarah_speeding + t.john_parking + t.john_speeding = 36 ∧
  t.mark_parking = 2 * t.sarah_parking ∧
  t.mark_speeding = t.sarah_speeding ∧
  t.john_parking * 3 = t.mark_parking ∧
  t.john_speeding = 2 * t.sarah_speeding ∧
  t.sarah_speeding = 6

/-- The theorem stating that Mark has 6 parking tickets -/
theorem mark_has_six_parking_tickets (t : Tickets) (h : satisfies_conditions t) : t.mark_parking = 6 := by
  sorry

end NUMINAMATH_CALUDE_mark_has_six_parking_tickets_l1098_109842


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1098_109833

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 9th, 52nd, and 95th terms of the sequence. -/
def sum_terms (a : ℕ → ℝ) : ℝ := a 9 + a 52 + a 95

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 = 4 → a 101 = 36 → sum_terms a = 60 :=
by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1098_109833


namespace NUMINAMATH_CALUDE_cosine_relationship_triangle_area_l1098_109870

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem stating the relationship between cosines in a triangle -/
theorem cosine_relationship (t : Triangle) 
  (h : t.a * Real.cos t.C = (2 * t.b - t.c) * Real.cos t.A) : 
  Real.cos t.A = 1 / 2 := by sorry

/-- Theorem for calculating the area of a specific triangle -/
theorem triangle_area (t : Triangle) 
  (h1 : t.a = 6) 
  (h2 : t.b + t.c = 8) 
  (h3 : Real.cos t.A = 1 / 2) : 
  (1 / 2) * t.a * t.b * Real.sin t.C = 7 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_cosine_relationship_triangle_area_l1098_109870


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l1098_109845

theorem pure_imaginary_product (a b c d : ℝ) :
  (∃ k : ℝ, (a + b * Complex.I) * (c + d * Complex.I) = k * Complex.I) →
  (a * c - b * d = 0 ∧ a * d + b * c ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l1098_109845


namespace NUMINAMATH_CALUDE_problems_per_page_l1098_109843

/-- Given a homework assignment with the following conditions:
  * There are 72 total problems
  * 32 problems have been completed
  * The remaining problems are spread equally across 5 pages
  This theorem proves that there are 8 problems on each remaining page. -/
theorem problems_per_page (total : ℕ) (completed : ℕ) (pages : ℕ) : 
  total = 72 → completed = 32 → pages = 5 → (total - completed) / pages = 8 := by
  sorry

end NUMINAMATH_CALUDE_problems_per_page_l1098_109843


namespace NUMINAMATH_CALUDE_jackson_painting_fraction_l1098_109871

-- Define the time it takes Jackson to paint the entire garage
def total_time : ℚ := 60

-- Define the time we want to calculate the portion for
def partial_time : ℚ := 12

-- Define the fraction of the garage painted in partial_time
def fraction_painted : ℚ := partial_time / total_time

-- Theorem to prove
theorem jackson_painting_fraction :
  fraction_painted = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_jackson_painting_fraction_l1098_109871


namespace NUMINAMATH_CALUDE_almonds_vs_white_sugar_difference_l1098_109807

-- Define the amounts of ingredients used
def brown_sugar : ℝ := 1.28
def white_sugar : ℝ := 0.75
def ground_almonds : ℝ := 1.56
def cocoa_powder : ℝ := 0.49

-- Theorem statement
theorem almonds_vs_white_sugar_difference :
  ground_almonds - white_sugar = 0.81 := by
  sorry

end NUMINAMATH_CALUDE_almonds_vs_white_sugar_difference_l1098_109807


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1098_109800

/-- Given a geometric sequence {aₙ} where the sum of the first n terms
    is given by Sₙ = a·2^(n-1) + 1/6, prove that a = -1/3 -/
theorem geometric_sequence_sum (a : ℝ) : 
  (∀ n : ℕ, ∃ Sn : ℝ, Sn = a * 2^(n-1) + 1/6) → a = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1098_109800


namespace NUMINAMATH_CALUDE_pet_shop_total_l1098_109896

/-- Given a pet shop with dogs, cats, and bunnies in stock, prove that the total number of dogs and bunnies is 375. -/
theorem pet_shop_total (dogs cats bunnies : ℕ) : 
  dogs = 75 →
  dogs / 3 = cats / 7 →
  dogs / 3 = bunnies / 12 →
  dogs + bunnies = 375 := by
sorry


end NUMINAMATH_CALUDE_pet_shop_total_l1098_109896


namespace NUMINAMATH_CALUDE_range_of_f_when_m_eq_1_solution_set_of_f_gt_3x_when_m_eq_neg_1_inequality_equivalence_l1098_109861

-- Define the function f(x) with parameter m
def f (m : ℝ) (x : ℝ) : ℝ := |x + 1| - m * |x - 2|

-- Theorem for the range of f(x) when m = 1
theorem range_of_f_when_m_eq_1 :
  Set.range (f 1) = Set.Icc (-3) 3 := by sorry

-- Theorem for the solution set of f(x) > 3x when m = -1
theorem solution_set_of_f_gt_3x_when_m_eq_neg_1 :
  {x : ℝ | f (-1) x > 3 * x} = Set.Iio 1 := by sorry

-- Additional helper theorem to show the equivalence of the inequality
theorem inequality_equivalence (x : ℝ) :
  f (-1) x > 3 * x ↔ |x + 1| + |x - 2| > 3 * x := by sorry

end NUMINAMATH_CALUDE_range_of_f_when_m_eq_1_solution_set_of_f_gt_3x_when_m_eq_neg_1_inequality_equivalence_l1098_109861


namespace NUMINAMATH_CALUDE_fraction_difference_equals_square_difference_l1098_109826

theorem fraction_difference_equals_square_difference 
  (x y z v : ℚ) (h : x / y + z / v = 1) : 
  x / y - z / v = (x / y)^2 - (z / v)^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_square_difference_l1098_109826


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l1098_109829

theorem triangle_perimeter_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  a = 1 →
  2 * Real.cos C + c = 2 * b →
  a = 2 * Real.sin (B / 2) * Real.sin (C / 2) / Real.sin ((B + C) / 2) →
  b = 2 * Real.sin (A / 2) * Real.sin (C / 2) / Real.sin ((A + C) / 2) →
  c = 2 * Real.sin (A / 2) * Real.sin (B / 2) / Real.sin ((A + B) / 2) →
  let p := a + b + c
  Real.sqrt 3 + 1 < p ∧ p < 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l1098_109829


namespace NUMINAMATH_CALUDE_menelaus_condition_l1098_109876

-- Define the points
variable (A B C D P Q R S O : Point)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define points on sides
def point_on_segment (P A B : Point) : Prop := sorry

-- Define intersection of lines
def lines_intersect (P R Q S O : Point) : Prop := sorry

-- Define quadrilateral with incircle
def has_incircle (A P O S : Point) : Prop := sorry

-- Define the ratio of segments
def segment_ratio (A P B : Point) : ℝ := sorry

-- Main theorem
theorem menelaus_condition 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_P : point_on_segment P A B)
  (h_Q : point_on_segment Q B C)
  (h_R : point_on_segment R C D)
  (h_S : point_on_segment S D A)
  (h_intersect : lines_intersect P R Q S O)
  (h_incircle1 : has_incircle A P O S)
  (h_incircle2 : has_incircle B Q O P)
  (h_incircle3 : has_incircle C R O Q)
  (h_incircle4 : has_incircle D S O R) :
  (segment_ratio A P B) * (segment_ratio B Q C) * 
  (segment_ratio C R D) * (segment_ratio D S A) = 1 := by
  sorry

end NUMINAMATH_CALUDE_menelaus_condition_l1098_109876


namespace NUMINAMATH_CALUDE_intersection_point_a_l1098_109840

/-- A function f(x) = 4x + b where b is an integer -/
def f (b : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + b

/-- The inverse of f -/
noncomputable def f_inv (b : ℤ) : ℝ → ℝ := λ x ↦ (x - b) / 4

theorem intersection_point_a (b : ℤ) (a : ℤ) :
  f b (-4) = a ∧ f_inv b (-4) = a → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_a_l1098_109840


namespace NUMINAMATH_CALUDE_population_1988_l1098_109806

/-- The population growth factor for a 4-year period -/
def growth_factor : ℝ := 2

/-- The number of 4-year periods between 1988 and 2008 -/
def num_periods : ℕ := 5

/-- The population of Arloe in 2008 -/
def population_2008 : ℕ := 3456

/-- The population growth function -/
def population (initial : ℕ) (periods : ℕ) : ℝ :=
  initial * growth_factor ^ periods

theorem population_1988 :
  ∃ p : ℕ, population p num_periods = population_2008 ∧ p = 108 := by
  sorry

end NUMINAMATH_CALUDE_population_1988_l1098_109806


namespace NUMINAMATH_CALUDE_range_of_a_min_value_of_a_l1098_109849

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Statement 1
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f a x ≤ 3) → 0 ≤ a ∧ a ≤ 4 := by sorry

-- Statement 2
theorem min_value_of_a :
  ∃ a : ℝ, a = 1/3 ∧ (∀ x : ℝ, |x - a| + |x + a| ≥ 1 - a) ∧
  (∀ b : ℝ, (∀ x : ℝ, |x - b| + |x + b| ≥ 1 - b) → a ≤ b) := by sorry

end NUMINAMATH_CALUDE_range_of_a_min_value_of_a_l1098_109849


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l1098_109869

theorem lcm_factor_proof (A B : ℕ) (X : ℕ) : 
  A > 0 → B > 0 →
  Nat.gcd A B = 59 →
  Nat.lcm A B = 59 * X * 16 →
  A = 944 →
  X = 1 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l1098_109869


namespace NUMINAMATH_CALUDE_max_sum_arithmetic_progression_l1098_109895

/-- The first term of the arithmetic progression -/
def a₁ : ℤ := 113

/-- The common difference of the arithmetic progression -/
def d : ℤ := -4

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

/-- The n-th term of the arithmetic progression -/
def aₙ (n : ℕ) : ℤ := a₁ + (n - 1) * d

/-- The maximum number of terms before the sequence becomes non-positive -/
def max_n : ℕ := 29

theorem max_sum_arithmetic_progression :
  ∀ n : ℕ, S n ≤ S max_n ∧ S max_n = 1653 :=
sorry

end NUMINAMATH_CALUDE_max_sum_arithmetic_progression_l1098_109895


namespace NUMINAMATH_CALUDE_negation_of_existence_l1098_109872

theorem negation_of_existence (Triangle : Type) (isSymmetrical : Triangle → Prop) :
  (¬ ∃ t : Triangle, isSymmetrical t) ↔ (∀ t : Triangle, ¬ isSymmetrical t) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l1098_109872


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1098_109814

theorem rectangular_field_area (L W : ℝ) : 
  L = 10 →                 -- One side is 10 feet
  2 * W + L = 130 →        -- Total fencing is 130 feet
  L * W = 600 :=           -- Area of the field is 600 square feet
by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1098_109814


namespace NUMINAMATH_CALUDE_point_translation_l1098_109873

/-- Given a point M(-2, 3) in the Cartesian coordinate system,
    prove that after translating it 3 units downwards and then 1 unit to the right,
    the resulting point has coordinates (-1, 0). -/
theorem point_translation (M : ℝ × ℝ) :
  M = (-2, 3) →
  let M' := (M.1, M.2 - 3)  -- Translate 3 units downwards
  let M'' := (M'.1 + 1, M'.2)  -- Translate 1 unit to the right
  M'' = (-1, 0) := by
sorry

end NUMINAMATH_CALUDE_point_translation_l1098_109873


namespace NUMINAMATH_CALUDE_inequality_solution_l1098_109830

theorem inequality_solution (k : ℝ) : 
  (∀ x : ℝ, (k + 2) * x > k + 2 ↔ x < 1) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1098_109830


namespace NUMINAMATH_CALUDE_sqrt_735_simplification_l1098_109848

theorem sqrt_735_simplification : Real.sqrt 735 = 7 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_735_simplification_l1098_109848


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l1098_109803

theorem largest_divisor_of_expression (x : ℤ) (h : Even x) :
  (∃ (k : ℤ), (10*x + 1) * (10*x + 5) * (5*x + 3) = 3 * k) ∧
  (∀ (d : ℤ), d > 3 → ∃ (y : ℤ), Even y ∧ ¬(∃ (k : ℤ), (10*y + 1) * (10*y + 5) * (5*y + 3) = d * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l1098_109803


namespace NUMINAMATH_CALUDE_pyramid_volume_l1098_109853

/-- Given a pyramid with a square base ABCD and vertex P, prove its volume. -/
theorem pyramid_volume (base_area : ℝ) (triangle_ABP_area : ℝ) (triangle_BCP_area : ℝ) (triangle_ADP_area : ℝ)
  (h_base : base_area = 256)
  (h_ABP : triangle_ABP_area = 128)
  (h_BCP : triangle_BCP_area = 80)
  (h_ADP : triangle_ADP_area = 128) :
  ∃ (volume : ℝ), volume = (2048 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l1098_109853


namespace NUMINAMATH_CALUDE_equation_solution_l1098_109877

theorem equation_solution (x : ℝ) (h : 1 - 9 / x + 9 / x^2 = 0) :
  3 / x = (3 - Real.sqrt 5) / 2 ∨ 3 / x = (3 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1098_109877


namespace NUMINAMATH_CALUDE_charlies_weight_l1098_109887

theorem charlies_weight (alice_weight charlie_weight : ℚ) 
  (sum_condition : alice_weight + charlie_weight = 240)
  (difference_condition : charlie_weight - alice_weight = charlie_weight / 3) :
  charlie_weight = 144 := by
  sorry

end NUMINAMATH_CALUDE_charlies_weight_l1098_109887


namespace NUMINAMATH_CALUDE_multiple_value_l1098_109818

-- Define the variables
variable (x : ℝ)
variable (m : ℝ)

-- State the theorem
theorem multiple_value (h1 : m * x + 36 = 48) (h2 : x = 4) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiple_value_l1098_109818


namespace NUMINAMATH_CALUDE_clara_sticker_ratio_l1098_109884

/-- Given Clara's sticker distribution, prove the ratio of stickers given to best friends
    to stickers left after giving to the boy is 1:2 -/
theorem clara_sticker_ratio :
  ∀ (initial stickers_to_boy stickers_left : ℕ),
  initial = 100 →
  stickers_to_boy = 10 →
  stickers_left = 45 →
  (initial - stickers_to_boy - stickers_left) * 2 = initial - stickers_to_boy :=
by sorry

end NUMINAMATH_CALUDE_clara_sticker_ratio_l1098_109884


namespace NUMINAMATH_CALUDE_all_nines_square_l1098_109852

/-- A function that generates a number with n 9's -/
def all_nines (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Theorem: For any positive integer n, (all_nines n)² = (all_nines n + 1)(all_nines n - 1) + 1 -/
theorem all_nines_square (n : ℕ+) :
  (all_nines n)^2 = (all_nines n + 1) * (all_nines n - 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_all_nines_square_l1098_109852


namespace NUMINAMATH_CALUDE_multiply_by_17_equals_493_l1098_109897

theorem multiply_by_17_equals_493 : ∃ x : ℤ, x * 17 = 493 ∧ x = 29 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_17_equals_493_l1098_109897


namespace NUMINAMATH_CALUDE_jeanne_needs_eight_tickets_l1098_109857

/-- The number of tickets needed for the Ferris wheel -/
def ferris_wheel_tickets : ℕ := 5

/-- The number of tickets needed for the roller coaster -/
def roller_coaster_tickets : ℕ := 4

/-- The number of tickets needed for the bumper cars -/
def bumper_cars_tickets : ℕ := 4

/-- The number of tickets Jeanne already has -/
def jeanne_tickets : ℕ := 5

/-- The total number of tickets needed for all three rides -/
def total_tickets_needed : ℕ := ferris_wheel_tickets + roller_coaster_tickets + bumper_cars_tickets

/-- The number of additional tickets Jeanne needs to buy -/
def additional_tickets_needed : ℕ := total_tickets_needed - jeanne_tickets

theorem jeanne_needs_eight_tickets : additional_tickets_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_jeanne_needs_eight_tickets_l1098_109857


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l1098_109811

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation
  (rate : ℝ) (interest : ℝ) (time : ℝ) :
  rate = 5 →
  interest = 160 →
  time = 4 →
  160 = (rate * time / 100) * (interest * 100 / (rate * time)) :=
by
  sorry

#check simple_interest_principal_calculation

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l1098_109811


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1098_109813

theorem solution_set_of_inequality (x : ℝ) :
  (8 * x^2 + 6 * x ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1098_109813


namespace NUMINAMATH_CALUDE_fraction_product_l1098_109855

theorem fraction_product : 
  (4 : ℚ) / 5 * 5 / 6 * 6 / 7 * 7 / 8 * 8 / 9 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1098_109855


namespace NUMINAMATH_CALUDE_age_difference_l1098_109821

/-- Given three people A, B, and C, where C is 14 years younger than A,
    prove that the total age of A and B is 14 years more than the total age of B and C. -/
theorem age_difference (A B C : ℕ) (h : C = A - 14) :
  (A + B) - (B + C) = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1098_109821


namespace NUMINAMATH_CALUDE_inequality_proof_l1098_109875

theorem inequality_proof (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) :
  y * (y - 1) ≤ x^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1098_109875


namespace NUMINAMATH_CALUDE_equation_graph_is_two_parallel_lines_l1098_109891

-- Define the equation
def equation (x y : ℝ) : Prop := x^3 * (x + y + 2) = y^3 * (x + y + 2)

-- Define what it means for two lines to be parallel
def parallel (l₁ l₂ : ℝ → ℝ) : Prop := 
  ∃ (k : ℝ), ∀ x, l₂ x = l₁ x + k

-- Theorem statement
theorem equation_graph_is_two_parallel_lines :
  ∃ (l₁ l₂ : ℝ → ℝ), 
    (∀ x y, equation x y ↔ (y = l₁ x ∨ y = l₂ x)) ∧
    parallel l₁ l₂ :=
sorry

end NUMINAMATH_CALUDE_equation_graph_is_two_parallel_lines_l1098_109891


namespace NUMINAMATH_CALUDE_assignFourFromTwentyFive_eq_303600_l1098_109879

/-- The number of ways to select and assign 4 people from a group of 25 to 4 distinct positions -/
def assignFourFromTwentyFive : ℕ := 25 * 24 * 23 * 22

/-- Theorem stating that the number of ways to select and assign 4 people from a group of 25 to 4 distinct positions is 303600 -/
theorem assignFourFromTwentyFive_eq_303600 : assignFourFromTwentyFive = 303600 := by
  sorry

end NUMINAMATH_CALUDE_assignFourFromTwentyFive_eq_303600_l1098_109879


namespace NUMINAMATH_CALUDE_expression_simplification_l1098_109889

theorem expression_simplification :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 12) / 4) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1098_109889


namespace NUMINAMATH_CALUDE_scale_division_l1098_109812

-- Define the length of the scale in inches
def scale_length : ℕ := 6 * 12 + 8

-- Define the number of parts
def num_parts : ℕ := 4

-- Theorem to prove
theorem scale_division :
  scale_length / num_parts = 20 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l1098_109812


namespace NUMINAMATH_CALUDE_marbles_difference_l1098_109838

/-- The number of marbles Cindy and Lisa have after Cindy gives some to Lisa -/
def marbles_after_giving (cindy_initial : ℕ) (lisa_initial : ℕ) (marbles_given : ℕ) :
  ℕ × ℕ :=
  (cindy_initial - marbles_given, lisa_initial + marbles_given)

/-- The theorem stating the difference in marbles after Cindy gives some to Lisa -/
theorem marbles_difference
  (cindy_initial : ℕ)
  (lisa_initial : ℕ)
  (marbles_given : ℕ)
  (h1 : cindy_initial = 20)
  (h2 : cindy_initial = lisa_initial + 5)
  (h3 : marbles_given = 12) :
  (marbles_after_giving cindy_initial lisa_initial marbles_given).2 -
  (marbles_after_giving cindy_initial lisa_initial marbles_given).1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_marbles_difference_l1098_109838


namespace NUMINAMATH_CALUDE_abc_fraction_value_l1098_109899

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 4)
  (h2 : b * c / (b + c) = 5)
  (h3 : c * a / (c + a) = 7) :
  a * b * c / (a * b + b * c + c * a) = 280 / 83 := by
  sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l1098_109899


namespace NUMINAMATH_CALUDE_range_of_x2_plus_y2_l1098_109885

theorem range_of_x2_plus_y2 (x y : ℝ) (h : x^2 - 2*x*y + 5*y^2 = 4) :
  ∃ (min max : ℝ), min = 3 - Real.sqrt 5 ∧ max = 3 + Real.sqrt 5 ∧
  (min ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ max) ∧
  ∃ (x1 y1 x2 y2 : ℝ), x1^2 - 2*x1*y1 + 5*y1^2 = 4 ∧
                       x2^2 - 2*x2*y2 + 5*y2^2 = 4 ∧
                       x1^2 + y1^2 = min ∧
                       x2^2 + y2^2 = max :=
by sorry

end NUMINAMATH_CALUDE_range_of_x2_plus_y2_l1098_109885


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1098_109835

theorem smallest_solution_of_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ 
  (∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x) →
  x = 4 - Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1098_109835


namespace NUMINAMATH_CALUDE_smaller_bucket_capacity_proof_l1098_109863

/-- The capacity of the smaller bucket in liters -/
def smaller_bucket_capacity : ℝ := 3

/-- The capacity of the medium bucket in liters -/
def medium_bucket_capacity : ℝ := 5

/-- The capacity of the larger bucket in liters -/
def larger_bucket_capacity : ℝ := 6

/-- The amount of water that can be added to the larger bucket after pouring from the medium bucket -/
def remaining_capacity : ℝ := 4

theorem smaller_bucket_capacity_proof :
  smaller_bucket_capacity = medium_bucket_capacity - (larger_bucket_capacity - remaining_capacity) :=
by sorry

end NUMINAMATH_CALUDE_smaller_bucket_capacity_proof_l1098_109863


namespace NUMINAMATH_CALUDE_glove_selection_theorem_l1098_109801

theorem glove_selection_theorem :
  let total_pairs : ℕ := 6
  let gloves_to_select : ℕ := 4
  let same_color_pair : ℕ := 1
  let ways_to_select_pair : ℕ := total_pairs.choose same_color_pair
  let remaining_gloves : ℕ := 2 * (total_pairs - same_color_pair)
  let ways_to_select_others : ℕ := remaining_gloves.choose (gloves_to_select - 2) - (total_pairs - same_color_pair)
  ways_to_select_pair * ways_to_select_others = 240
  := by sorry

end NUMINAMATH_CALUDE_glove_selection_theorem_l1098_109801


namespace NUMINAMATH_CALUDE_sequence_properties_l1098_109823

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, n > 1 → a (n - 1) + a (n + 1) > 2 * a n

theorem sequence_properties (a : ℕ+ → ℝ) (h : sequence_property a) :
  (a 2 > a 1 → ∀ n : ℕ+, n > 1 → a n > a (n - 1)) ∧
  (∃ d : ℝ, ∀ n : ℕ+, a n > a 1 + (n - 1) * d) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1098_109823


namespace NUMINAMATH_CALUDE_no_friendly_triplet_in_small_range_exists_friendly_triplet_in_large_range_l1098_109865

-- Define friendly integers
def friendly (a b c : ℤ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a ∣ b * c ∨ b ∣ a * c ∨ c ∣ a * b)

theorem no_friendly_triplet_in_small_range (n : ℕ) :
  ¬∃ a b c : ℤ, n^2 < a ∧ a < b ∧ b < c ∧ c < n^2 + n ∧ friendly a b c := by
  sorry

theorem exists_friendly_triplet_in_large_range (n : ℕ) :
  ∃ a b c : ℤ, n^2 < a ∧ a < b ∧ b < c ∧ c < n^2 + n + 3 * Real.sqrt n ∧ friendly a b c := by
  sorry

end NUMINAMATH_CALUDE_no_friendly_triplet_in_small_range_exists_friendly_triplet_in_large_range_l1098_109865


namespace NUMINAMATH_CALUDE_bus_passing_theorem_l1098_109802

/-- Represents the time in minutes since midnight -/
def Time := ℕ

/-- Represents the direction of the bus -/
inductive Direction
| Austin2SanAntonio
| SanAntonio2Austin

/-- Represents a bus schedule -/
structure BusSchedule where
  start : Time
  interval : ℕ
  direction : Direction

/-- Calculates the number of buses passed during a journey -/
def count_passed_buses (sa_schedule : BusSchedule) (austin_schedule : BusSchedule) (journey_time : ℕ) : ℕ :=
  sorry

/-- Converts time from hour:minute format to minutes since midnight -/
def time_to_minutes (hour : ℕ) (minute : ℕ) : Time :=
  hour * 60 + minute

theorem bus_passing_theorem (sa_schedule : BusSchedule) (austin_schedule : BusSchedule) :
  sa_schedule.start = time_to_minutes 12 15 ∧
  sa_schedule.interval = 30 ∧
  sa_schedule.direction = Direction.SanAntonio2Austin ∧
  austin_schedule.start = time_to_minutes 12 0 ∧
  austin_schedule.interval = 45 ∧
  austin_schedule.direction = Direction.Austin2SanAntonio →
  count_passed_buses sa_schedule austin_schedule (6 * 60) = 9 :=
sorry

end NUMINAMATH_CALUDE_bus_passing_theorem_l1098_109802


namespace NUMINAMATH_CALUDE_min_dot_product_on_hyperbola_l1098_109805

theorem min_dot_product_on_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let m : ℝ × ℝ := (1, Real.sqrt (a^2 + 1/a^2))
  let B : ℝ × ℝ := (b, 1/b)
  m.1 * B.1 + m.2 * B.2 ≥ 2 * Real.sqrt (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_on_hyperbola_l1098_109805


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l1098_109882

theorem max_imaginary_part_of_roots (z : ℂ) (φ : ℝ) :
  z^6 - z^4 + z^2 - 1 = 0 →
  -π/2 ≤ φ ∧ φ ≤ π/2 →
  z.im = Real.sin φ →
  z.im ≤ Real.sin (π/4) :=
by sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l1098_109882


namespace NUMINAMATH_CALUDE_cube_root_simplification_l1098_109858

theorem cube_root_simplification : 
  (25^3 + 30^3 + 35^3 : ℝ)^(1/3) = 5 * 684^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l1098_109858


namespace NUMINAMATH_CALUDE_inequality_not_always_hold_l1098_109859

theorem inequality_not_always_hold (a b : ℝ) (h : a > b) : 
  ¬ (∀ c : ℝ, a * c > b * c) := by
sorry

end NUMINAMATH_CALUDE_inequality_not_always_hold_l1098_109859


namespace NUMINAMATH_CALUDE_number_puzzle_l1098_109856

theorem number_puzzle (x y : ℝ) : x = 265 → (x / 5) + y = 61 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1098_109856


namespace NUMINAMATH_CALUDE_coin_division_problem_l1098_109841

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (n % 8 = 6) → 
  (n % 7 = 5) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 9 = 0) := by
sorry

end NUMINAMATH_CALUDE_coin_division_problem_l1098_109841


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l1098_109819

theorem subtraction_of_decimals : (3.75 : ℝ) - (1.46 : ℝ) = 2.29 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l1098_109819


namespace NUMINAMATH_CALUDE_time_to_meet_prove_time_to_meet_l1098_109836

/-- The time it takes for Michael to reach Eric given the specified conditions --/
theorem time_to_meet (initial_distance : ℝ) (speed_ratio : ℝ) (closing_rate : ℝ) 
  (initial_time : ℝ) (delay_time : ℝ) : ℝ :=
  65

/-- Proof of the time it takes for Michael to reach Eric --/
theorem prove_time_to_meet :
  time_to_meet 30 4 2 4 6 = 65 := by
  sorry

end NUMINAMATH_CALUDE_time_to_meet_prove_time_to_meet_l1098_109836


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l1098_109878

theorem wire_ratio_proof (total_length longer_length shorter_length : ℚ) : 
  total_length = 80 →
  shorter_length = 30 →
  longer_length = total_length - shorter_length →
  shorter_length / longer_length = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l1098_109878


namespace NUMINAMATH_CALUDE_expansion_terms_count_l1098_109883

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^7 -/
def dissimilar_terms : ℕ := Nat.choose 10 3

/-- Theorem stating that the number of dissimilar terms in (a + b + c + d)^7 is equal to (10 choose 3) -/
theorem expansion_terms_count : dissimilar_terms = 120 := by sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l1098_109883


namespace NUMINAMATH_CALUDE_tomato_plants_per_row_l1098_109825

/-- Proves that the number of plants in each row is 10, given the conditions of the tomato planting problem -/
theorem tomato_plants_per_row :
  ∀ (rows : ℕ) (yield_per_plant : ℕ) (total_yield : ℕ),
    rows = 30 →
    yield_per_plant = 20 →
    total_yield = 6000 →
    total_yield = rows * yield_per_plant * (total_yield / (rows * yield_per_plant)) →
    total_yield / (rows * yield_per_plant) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_tomato_plants_per_row_l1098_109825


namespace NUMINAMATH_CALUDE_alice_bake_time_proof_l1098_109832

/-- The time it takes Alice to bake a pie -/
def alice_bake_time : ℝ := 5

/-- The time it takes Bob to bake a pie -/
def bob_bake_time : ℝ := 6

/-- The total time given in the problem -/
def total_time : ℝ := 60

/-- The number of additional pies Alice can bake compared to Bob in the given time -/
def additional_pies : ℕ := 2

theorem alice_bake_time_proof :
  alice_bake_time = 5 ∧
  (total_time / bob_bake_time + additional_pies) * alice_bake_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_alice_bake_time_proof_l1098_109832


namespace NUMINAMATH_CALUDE_rain_probability_l1098_109817

theorem rain_probability (p : ℝ) (h : p = 3/4) :
  1 - (1 - p)^4 = 255/256 := by sorry

end NUMINAMATH_CALUDE_rain_probability_l1098_109817


namespace NUMINAMATH_CALUDE_x_is_perfect_square_l1098_109874

theorem x_is_perfect_square (x y : ℕ+) (h : (2 * x * y) ∣ (x^2 + y^2 - x)) : 
  ∃ n : ℕ+, x = n^2 := by
sorry

end NUMINAMATH_CALUDE_x_is_perfect_square_l1098_109874


namespace NUMINAMATH_CALUDE_unique_monic_quadratic_l1098_109888

/-- A monic polynomial of degree 2 -/
def MonicQuadratic (g : ℝ → ℝ) : Prop :=
  ∃ b c : ℝ, ∀ x, g x = x^2 + b*x + c

theorem unique_monic_quadratic (g : ℝ → ℝ) 
  (h_monic : MonicQuadratic g) 
  (h_g0 : g 0 = 2) 
  (h_g1 : g 1 = 6) : 
  ∀ x, g x = x^2 + 3*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_monic_quadratic_l1098_109888


namespace NUMINAMATH_CALUDE_first_concert_attendance_calculation_l1098_109868

/-- The number of people attending the second concert -/
def second_concert_attendance : ℕ := 66018

/-- The difference in attendance between the second and first concerts -/
def attendance_difference : ℕ := 119

/-- The number of people attending the first concert -/
def first_concert_attendance : ℕ := second_concert_attendance - attendance_difference

theorem first_concert_attendance_calculation :
  first_concert_attendance = 65899 :=
by sorry

end NUMINAMATH_CALUDE_first_concert_attendance_calculation_l1098_109868


namespace NUMINAMATH_CALUDE_friendship_class_theorem_l1098_109846

/-- Represents the number of students in a class with specific friendship conditions. -/
structure FriendshipClass where
  boys : ℕ
  girls : ℕ

/-- Checks if the friendship conditions are satisfied for a given class. -/
def satisfiesFriendshipConditions (c : FriendshipClass) : Prop :=
  3 * c.boys = 2 * c.girls

/-- Checks if a class with the given total number of students can satisfy the friendship conditions. -/
def canHaveStudents (n : ℕ) : Prop :=
  ∃ c : FriendshipClass, c.boys + c.girls = n ∧ satisfiesFriendshipConditions c

theorem friendship_class_theorem :
  ¬(canHaveStudents 32) ∧ (canHaveStudents 30) := by sorry

end NUMINAMATH_CALUDE_friendship_class_theorem_l1098_109846


namespace NUMINAMATH_CALUDE_emily_quiz_score_theorem_l1098_109831

def emily_scores : List ℕ := [85, 92, 88, 90, 93]
def target_mean : ℕ := 91
def num_quizzes : ℕ := 6
def sixth_score : ℕ := 98

theorem emily_quiz_score_theorem :
  let total_sum := (emily_scores.sum + sixth_score)
  total_sum / num_quizzes = target_mean :=
by sorry

end NUMINAMATH_CALUDE_emily_quiz_score_theorem_l1098_109831


namespace NUMINAMATH_CALUDE_quarter_squared_decimal_l1098_109881

theorem quarter_squared_decimal : (1 / 4 : ℚ) ^ 2 = 0.0625 := by
  sorry

end NUMINAMATH_CALUDE_quarter_squared_decimal_l1098_109881


namespace NUMINAMATH_CALUDE_train_speed_proof_l1098_109822

/-- Proves that a train crossing a 320-meter platform in 34 seconds and passing a stationary man in 18 seconds has a speed of 72 km/h -/
theorem train_speed_proof (platform_length : ℝ) (platform_crossing_time : ℝ) (man_passing_time : ℝ) :
  platform_length = 320 →
  platform_crossing_time = 34 →
  man_passing_time = 18 →
  ∃ (train_speed : ℝ),
    train_speed * man_passing_time = train_speed * platform_crossing_time - platform_length ∧
    train_speed * 3.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_proof_l1098_109822


namespace NUMINAMATH_CALUDE_store_comparison_and_best_plan_l1098_109804

/- Define the prices and quantities -/
def racket_price : ℝ := 50
def ball_price : ℝ := 20
def racket_quantity : ℕ := 10
def ball_quantity : ℕ := 40

/- Define the cost functions for each store -/
def cost_store_a (x : ℝ) : ℝ := 20 * x + 300
def cost_store_b (x : ℝ) : ℝ := 16 * x + 400

/- Define the most cost-effective plan -/
def cost_effective_plan : ℝ := racket_price * racket_quantity + ball_price * (ball_quantity - racket_quantity) * 0.8

/- Theorem statement -/
theorem store_comparison_and_best_plan :
  (cost_store_b ball_quantity < cost_store_a ball_quantity) ∧
  (cost_effective_plan = 980) := by
  sorry


end NUMINAMATH_CALUDE_store_comparison_and_best_plan_l1098_109804


namespace NUMINAMATH_CALUDE_solve_equations_l1098_109828

theorem solve_equations :
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, 3 * x * (x - 1) = 2 * (x - 1) ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 1 ∧ x₂ = 2/3) ∧
  (∃ y₁ y₂ : ℝ, (∀ x : ℝ, x^2 - 6*x + 6 = 0 ↔ x = y₁ ∨ x = y₂) ∧ y₁ = 3 + Real.sqrt 3 ∧ y₂ = 3 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l1098_109828


namespace NUMINAMATH_CALUDE_inequality_solution_l1098_109892

theorem inequality_solution (x : ℝ) : 
  (x + 1) / 2 > 1 - (2 * x - 1) / 3 ↔ x > 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1098_109892


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l1098_109839

theorem student_multiplication_problem (x : ℝ) : 40 * x - 150 = 130 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l1098_109839


namespace NUMINAMATH_CALUDE_convex_ngon_coverage_l1098_109827

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool
  area : Real

/-- Represents a triangle in 2D space -/
structure Triangle where
  vertices : List (Real × Real)
  area : Real

/-- Checks if a polygon is covered by a triangle -/
def is_covered (p : ConvexPolygon) (t : Triangle) : Prop :=
  sorry

/-- Main theorem: A convex n-gon with area 1 (n ≥ 6) can be covered by a triangle with area ≤ 2 -/
theorem convex_ngon_coverage (p : ConvexPolygon) :
  p.is_convex ∧ p.area = 1 ∧ p.vertices.length ≥ 6 →
  ∃ t : Triangle, t.area ≤ 2 ∧ is_covered p t :=
sorry

end NUMINAMATH_CALUDE_convex_ngon_coverage_l1098_109827


namespace NUMINAMATH_CALUDE_brick_width_calculation_l1098_109854

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 11.25

/-- The volume of the wall in cubic centimeters -/
def wall_volume : ℝ := 700 * 600 * 22.5

/-- The number of bricks required -/
def num_bricks : ℕ := 5600

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 25

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 6

theorem brick_width_calculation : 
  wall_volume = (brick_length * brick_width * brick_height) * num_bricks :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l1098_109854


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt26_l1098_109847

theorem consecutive_integers_around_sqrt26 (n m : ℤ) : 
  (n + 1 = m) → (n < Real.sqrt 26) → (Real.sqrt 26 < m) → (m + n = 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt26_l1098_109847


namespace NUMINAMATH_CALUDE_sqrt_two_squared_cubed_l1098_109837

theorem sqrt_two_squared_cubed : (Real.sqrt (Real.sqrt 2)^2)^3 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_cubed_l1098_109837


namespace NUMINAMATH_CALUDE_square_field_area_l1098_109866

/-- Proves that a square field crossed diagonally in 9 seconds by a man walking at 6 km/h has an area of 112.5 square meters. -/
theorem square_field_area (speed_kmh : ℝ) (time_s : ℝ) (area : ℝ) : 
  speed_kmh = 6 → time_s = 9 → area = 112.5 → 
  let speed_ms := speed_kmh * 1000 / 3600
  let diagonal := speed_ms * time_s
  let side := (diagonal^2 / 2).sqrt
  area = side^2 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l1098_109866


namespace NUMINAMATH_CALUDE_tori_classroom_trash_l1098_109809

/-- Represents the number of pieces of trash picked up in various locations --/
structure TrashCount where
  total : ℕ
  outside : ℕ

/-- Calculates the number of pieces of trash picked up in the classrooms --/
def classroom_trash (t : TrashCount) : ℕ :=
  t.total - t.outside

/-- Theorem stating that for Tori's specific trash counts, the classroom trash is 344 --/
theorem tori_classroom_trash :
  let tori_trash : TrashCount := { total := 1576, outside := 1232 }
  classroom_trash tori_trash = 344 := by
  sorry

#eval classroom_trash { total := 1576, outside := 1232 }

end NUMINAMATH_CALUDE_tori_classroom_trash_l1098_109809


namespace NUMINAMATH_CALUDE_eight_digit_non_decreasing_integers_mod_1000_l1098_109864

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- The number of 8-digit positive integers with non-decreasing digits -/
def M : ℕ := stars_and_bars 8 9

theorem eight_digit_non_decreasing_integers_mod_1000 : M % 1000 = 870 := by sorry

end NUMINAMATH_CALUDE_eight_digit_non_decreasing_integers_mod_1000_l1098_109864


namespace NUMINAMATH_CALUDE_range_of_a_l1098_109867

theorem range_of_a (a : ℝ) : 
  (¬∀ x : ℝ, |1 - x| - |x - 5| < a → False) → a > 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1098_109867


namespace NUMINAMATH_CALUDE_find_N_l1098_109898

theorem find_N (a b c N : ℚ) : 
  a + b + c = 120 ∧
  a - 10 = N ∧
  b + 10 = N ∧
  7 * c = N →
  N = 56 := by
sorry

end NUMINAMATH_CALUDE_find_N_l1098_109898


namespace NUMINAMATH_CALUDE_no_polynomial_iteration_fixed_points_l1098_109886

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial) : ℕ := sorry

/-- A function from integers to integers -/
def IntFunction := ℤ → ℤ

/-- n-fold application of a function -/
def iterate (f : IntFunction) (n : ℕ) : IntFunction := sorry

/-- The number of fixed points of a function -/
def fixedPointCount (f : IntFunction) : ℕ := sorry

/-- Main theorem -/
theorem no_polynomial_iteration_fixed_points :
  ¬ ∃ (P : IntPolynomial) (T : IntFunction),
    degree P ≥ 1 ∧
    (∀ n : ℕ, n ≥ 1 → fixedPointCount (iterate T n) = P n) :=
sorry

end NUMINAMATH_CALUDE_no_polynomial_iteration_fixed_points_l1098_109886


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l1098_109894

theorem purely_imaginary_z (z : ℂ) : 
  (∃ b : ℝ, z = Complex.I * b) →  -- z is purely imaginary
  (∃ c : ℝ, (z - 3)^2 + Complex.I * 5 = Complex.I * c) →  -- (z-3)^2+5i is purely imaginary
  z = Complex.I * 3 ∨ z = Complex.I * (-3) :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l1098_109894


namespace NUMINAMATH_CALUDE_max_value_of_a_plus_inverse_l1098_109808

theorem max_value_of_a_plus_inverse (a : ℝ) (h : a < 0) : 
  ∃ (M : ℝ), M = -2 ∧ ∀ (x : ℝ), x < 0 → x + 1/x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_plus_inverse_l1098_109808


namespace NUMINAMATH_CALUDE_three_possible_values_for_d_l1098_109880

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Represents the equation AABC + CBBA = DCCD -/
def satisfies_equation (a b c d : Digit) : Prop :=
  1000 * a.val + 100 * a.val + 10 * b.val + c.val +
  1000 * c.val + 100 * b.val + 10 * b.val + a.val =
  1000 * d.val + 100 * c.val + 10 * c.val + d.val

/-- The main theorem stating there are exactly 3 possible values for D -/
theorem three_possible_values_for_d :
  ∃ (s : Finset Digit),
    s.card = 3 ∧
    (∀ d : Digit, d ∈ s ↔ 
      ∃ (a b c : Digit), distinct a b c d ∧ satisfies_equation a b c d) :=
sorry

end NUMINAMATH_CALUDE_three_possible_values_for_d_l1098_109880


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1098_109850

theorem fraction_sum_equality : (1 : ℚ) / 3 + 5 / 9 - 2 / 9 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1098_109850


namespace NUMINAMATH_CALUDE_apple_pyramid_theorem_l1098_109893

/-- Calculates the number of apples in a layer of the pyramid --/
def apples_in_layer (base_width : ℕ) (base_length : ℕ) (layer : ℕ) : ℕ :=
  (base_width - layer + 1) * (base_length - layer + 1)

/-- Calculates the total number of apples in the pyramid --/
def total_apples (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let max_layers := min base_width base_length
  (List.range max_layers).foldl (fun acc layer => acc + apples_in_layer base_width base_length layer) 0

/-- The theorem stating that a pyramid with a 6x9 base contains 154 apples --/
theorem apple_pyramid_theorem :
  total_apples 6 9 = 154 := by
  sorry

end NUMINAMATH_CALUDE_apple_pyramid_theorem_l1098_109893


namespace NUMINAMATH_CALUDE_divisibility_by_1995_l1098_109890

theorem divisibility_by_1995 (n : ℕ) : 
  1995 ∣ 256^(2*n) * 7^(2*n) - 168^(2*n) - 32^(2*n) + 3^(2*n) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_1995_l1098_109890


namespace NUMINAMATH_CALUDE_product_of_numbers_l1098_109820

theorem product_of_numbers (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 31) : a * b = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1098_109820


namespace NUMINAMATH_CALUDE_average_of_new_sequence_eq_l1098_109860

/-- Given a positive integer a, this function returns the average of seven consecutive integers starting with a. -/
def average_of_seven (a : ℤ) : ℚ :=
  (7 * a + 21) / 7

/-- Given a positive integer a, this function returns the average of seven consecutive integers starting with the average of seven consecutive integers starting with a. -/
def average_of_new_sequence (a : ℤ) : ℚ :=
  let b := average_of_seven a
  (7 * ⌊b⌋ + 21) / 7

/-- Theorem stating that the average of the new sequence is equal to a + 6 -/
theorem average_of_new_sequence_eq (a : ℤ) (h : a > 0) : 
  average_of_new_sequence a = a + 6 := by
  sorry

end NUMINAMATH_CALUDE_average_of_new_sequence_eq_l1098_109860


namespace NUMINAMATH_CALUDE_fabric_theorem_l1098_109810

def fabric_problem (checkered_cost plain_cost yard_cost : ℚ) : Prop :=
  let checkered_yards := checkered_cost / yard_cost
  let plain_yards := plain_cost / yard_cost
  let total_yards := checkered_yards + plain_yards
  total_yards = 16

theorem fabric_theorem :
  fabric_problem 75 45 7.5 := by
  sorry

end NUMINAMATH_CALUDE_fabric_theorem_l1098_109810
