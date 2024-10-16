import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2477_247770

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (1 - 2*I) / (1 + I^3)
  Complex.im z = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2477_247770


namespace NUMINAMATH_CALUDE_root_implies_q_value_l2477_247781

theorem root_implies_q_value (p q : ℝ) : 
  (Complex.I * Real.sqrt 3 + 1) ^ 2 + p * (Complex.I * Real.sqrt 3 + 1) + q = 0 → q = 4 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_q_value_l2477_247781


namespace NUMINAMATH_CALUDE_charlie_seashells_l2477_247739

theorem charlie_seashells (c e : ℕ) : 
  c = e + 10 →  -- Charlie collected 10 more seashells than Emily
  e = c / 3 →   -- Emily collected one-third the number of seashells Charlie collected
  c = 15 :=     -- Charlie collected 15 seashells
by sorry

end NUMINAMATH_CALUDE_charlie_seashells_l2477_247739


namespace NUMINAMATH_CALUDE_constant_x_coordinate_l2477_247730

/-- Definition of the ellipse C -/
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Right focus F -/
def F : ℝ × ℝ := (1, 0)

/-- Left vertex A -/
def A : ℝ × ℝ := (-2, 0)

/-- Right vertex B -/
def B : ℝ × ℝ := (2, 0)

/-- Line l passing through F, not coincident with x-axis -/
def l (k : ℝ) (x y : ℝ) : Prop := y = k*(x - F.1) ∧ k ≠ 0

/-- Intersection points M and N of line l with ellipse C -/
def intersectionPoints (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | C p.1 p.2 ∧ l k p.1 p.2}

/-- Line AM -/
def lineAM (M : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - A.2) * (M.1 - A.1) = (x - A.1) * (M.2 - A.2)

/-- Line BN -/
def lineBN (N : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - B.2) * (N.1 - B.1) = (x - B.1) * (N.2 - B.2)

/-- Theorem: x-coordinate of intersection point T is constant -/
theorem constant_x_coordinate (k : ℝ) (M N : ℝ × ℝ) (h1 : M ∈ intersectionPoints k) (h2 : N ∈ intersectionPoints k) (h3 : M ≠ N) :
  ∃ (T : ℝ × ℝ), lineAM M T.1 T.2 ∧ lineBN N T.1 T.2 ∧ T.1 = 4 := by sorry

end NUMINAMATH_CALUDE_constant_x_coordinate_l2477_247730


namespace NUMINAMATH_CALUDE_lowest_score_jack_l2477_247734

def class_mean : ℝ := 60
def standard_deviation : ℝ := 10
def z_score_90th_percentile : ℝ := 1.28

theorem lowest_score_jack (score : ℝ) : 
  (score ≥ z_score_90th_percentile * standard_deviation + class_mean) →
  (score ≤ class_mean + 2 * standard_deviation) →
  (∀ x : ℝ, x ≥ z_score_90th_percentile * standard_deviation + class_mean →
            x ≤ class_mean + 2 * standard_deviation →
            score ≤ x) →
  score = 73 := by
sorry

end NUMINAMATH_CALUDE_lowest_score_jack_l2477_247734


namespace NUMINAMATH_CALUDE_union_covers_reals_l2477_247782

open Set Real

theorem union_covers_reals (a : ℝ) : 
  let S : Set ℝ := {x | |x - 2| > 3}
  let T : Set ℝ := {x | a < x ∧ x < a + 8}
  (S ∪ T = univ) → (-3 < a ∧ a < -1) :=
by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_l2477_247782


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2477_247775

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 8192; 0, -8192] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2477_247775


namespace NUMINAMATH_CALUDE_sum_due_from_discounts_l2477_247778

/-- The sum due (present value) given banker's discount and true discount -/
theorem sum_due_from_discounts (BD TD : ℝ) (h1 : BD = 42) (h2 : TD = 36) :
  ∃ PV : ℝ, PV = 216 ∧ BD = TD + TD^2 / PV :=
by sorry

end NUMINAMATH_CALUDE_sum_due_from_discounts_l2477_247778


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2477_247746

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4 + a 7 = 19) →
  (a 3 + 5 * a 6 = 57) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2477_247746


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2477_247779

-- Define the sets M and N
def M : Set ℝ := {x | x + 1 ≥ 0}
def N : Set ℝ := {x | x^2 < 4}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | -1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2477_247779


namespace NUMINAMATH_CALUDE_total_candles_in_small_boxes_l2477_247700

theorem total_candles_in_small_boxes 
  (small_boxes_per_big_box : ℕ) 
  (num_big_boxes : ℕ) 
  (candles_per_small_box : ℕ) : 
  small_boxes_per_big_box = 4 → 
  num_big_boxes = 50 → 
  candles_per_small_box = 40 → 
  small_boxes_per_big_box * num_big_boxes * candles_per_small_box = 8000 :=
by sorry

end NUMINAMATH_CALUDE_total_candles_in_small_boxes_l2477_247700


namespace NUMINAMATH_CALUDE_expressions_same_type_l2477_247727

/-- Two expressions are of the same type if they have the same variables with the same exponents -/
def same_type (e1 e2 : ℕ → ℕ → ℕ → ℚ) : Prop :=
  ∀ a b c : ℕ, ∃ k1 k2 : ℚ, e1 a b c = k1 * a * b^3 * c ∧ e2 a b c = k2 * a * b^3 * c

/-- The original expression -/
def original (a b c : ℕ) : ℚ := -↑a * ↑b^3 * ↑c

/-- The expression to compare -/
def to_compare (a b c : ℕ) : ℚ := (1/3) * ↑a * ↑c * ↑b^3

/-- Theorem stating that the two expressions are of the same type -/
theorem expressions_same_type : same_type original to_compare := by
  sorry

end NUMINAMATH_CALUDE_expressions_same_type_l2477_247727


namespace NUMINAMATH_CALUDE_diamond_value_l2477_247724

/-- Given a digit d, this function returns the value of d3 in base 5 -/
def base5_value (d : ℕ) : ℕ := d * 5 + 3

/-- Given a digit d, this function returns the value of d2 in base 6 -/
def base6_value (d : ℕ) : ℕ := d * 6 + 2

/-- The theorem states that the digit d satisfying d3 in base 5 equals d2 in base 6 is 1 -/
theorem diamond_value :
  ∃ (d : ℕ), d < 10 ∧ base5_value d = base6_value d ∧ d = 1 :=
sorry

end NUMINAMATH_CALUDE_diamond_value_l2477_247724


namespace NUMINAMATH_CALUDE_expected_balls_in_original_position_l2477_247709

/-- Represents the number of balls arranged in a circle -/
def numBalls : ℕ := 6

/-- Represents the number of people performing swaps -/
def numSwaps : ℕ := 3

/-- Probability that a specific ball is not involved in a single swap -/
def probNotSwapped : ℚ := 4 / 6

/-- Probability that a ball remains in its original position after all swaps -/
def probInOriginalPosition : ℚ := probNotSwapped ^ numSwaps

/-- Expected number of balls in their original positions after all swaps -/
def expectedBallsInOriginalPosition : ℚ := numBalls * probInOriginalPosition

/-- Theorem stating the expected number of balls in their original positions -/
theorem expected_balls_in_original_position :
  expectedBallsInOriginalPosition = 48 / 27 := by
  sorry

end NUMINAMATH_CALUDE_expected_balls_in_original_position_l2477_247709


namespace NUMINAMATH_CALUDE_four_digit_count_l2477_247787

/-- The count of four-digit numbers -/
def count_four_digit_numbers : ℕ := 9999 - 1000 + 1

/-- The smallest four-digit number -/
def min_four_digit : ℕ := 1000

/-- The largest four-digit number -/
def max_four_digit : ℕ := 9999

/-- Theorem: The count of integers from 1000 to 9999 (inclusive) is equal to 9000 -/
theorem four_digit_count :
  count_four_digit_numbers = 9000 := by sorry

end NUMINAMATH_CALUDE_four_digit_count_l2477_247787


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2477_247702

theorem sum_of_fractions_equals_one (a b c : ℝ) (h : a * b * c = 1) :
  1 / (1 + a + a * b) + 1 / (1 + b + b * c) + 1 / (1 + c + c * a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2477_247702


namespace NUMINAMATH_CALUDE_x_is_perfect_square_l2477_247722

theorem x_is_perfect_square (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x > y)
  (h_div : (x^2019 + x + y^2) % (x*y) = 0) : 
  ∃ (n : ℕ), x = n^2 := by
sorry

end NUMINAMATH_CALUDE_x_is_perfect_square_l2477_247722


namespace NUMINAMATH_CALUDE_sum_and_count_30_to_40_l2477_247762

def sum_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_in_range (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_30_to_40 : 
  sum_range 30 40 + count_even_in_range 30 40 = 391 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_30_to_40_l2477_247762


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2477_247771

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2477_247771


namespace NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l2477_247758

theorem complex_exp_13pi_over_2 : Complex.exp ((13 * Real.pi / 2) * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l2477_247758


namespace NUMINAMATH_CALUDE_acute_angles_inequality_l2477_247733

theorem acute_angles_inequality (α β : Real) (h_α : 0 < α ∧ α < π / 2) (h_β : 0 < β ∧ β < π / 2) :
  Real.cos α * Real.sin (2 * α) * Real.sin (2 * β) ≤ 4 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_inequality_l2477_247733


namespace NUMINAMATH_CALUDE_product_increase_thirteen_times_l2477_247714

theorem product_increase_thirteen_times :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ),
    ((a₁ - 3) * (a₂ - 3) * (a₃ - 3) * (a₄ - 3) * (a₅ - 3) * (a₆ - 3) * (a₇ - 3)) / 
    (a₁ * a₂ * a₃ * a₄ * a₅ * a₆ * a₇ : ℚ) = 13 :=
by sorry

end NUMINAMATH_CALUDE_product_increase_thirteen_times_l2477_247714


namespace NUMINAMATH_CALUDE_work_completion_time_l2477_247726

theorem work_completion_time (b : ℝ) (a_wage_ratio : ℝ) (a : ℝ) : 
  b = 15 →
  a_wage_ratio = 3/5 →
  (1/a) / ((1/a) + (1/b)) = a_wage_ratio →
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2477_247726


namespace NUMINAMATH_CALUDE_triangle_properties_l2477_247708

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

end NUMINAMATH_CALUDE_triangle_properties_l2477_247708


namespace NUMINAMATH_CALUDE_laura_change_l2477_247704

def change_calculation (pants_cost : ℕ) (pants_count : ℕ) (shirt_cost : ℕ) (shirt_count : ℕ) (amount_given : ℕ) : ℕ :=
  amount_given - (pants_cost * pants_count + shirt_cost * shirt_count)

theorem laura_change : change_calculation 54 2 33 4 250 = 10 := by
  sorry

end NUMINAMATH_CALUDE_laura_change_l2477_247704


namespace NUMINAMATH_CALUDE_parallel_lines_k_values_l2477_247743

/-- Definition of Line l₁ -/
def l₁ (k : ℝ) (x y : ℝ) : Prop :=
  (k - 3) * x + (4 - k) * y + 1 = 0

/-- Definition of Line l₂ -/
def l₂ (k : ℝ) (x y : ℝ) : Prop :=
  2 * (k - 3) * x - 2 * y + 3 = 0

/-- Definition of parallel lines -/
def parallel (l₁ l₂ : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ (m : ℝ), ∀ (k x y : ℝ), l₁ k x y ↔ l₂ k (m * x) y

/-- Theorem stating that for l₁ and l₂ to be parallel, k must be 2, 3, or 6 -/
theorem parallel_lines_k_values :
  parallel l₁ l₂ ↔ (∃ k : ℝ, k = 2 ∨ k = 3 ∨ k = 6) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_k_values_l2477_247743


namespace NUMINAMATH_CALUDE_smallest_n_for_B_exceeds_A_l2477_247744

def A (n : ℕ) : ℚ := 490 * n - 10 * n^2

def B (n : ℕ) : ℚ := 500 * n + 400 - 500 / 2^(n-1)

theorem smallest_n_for_B_exceeds_A :
  ∀ k : ℕ, k < 4 → B k ≤ A k ∧ B 4 > A 4 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_B_exceeds_A_l2477_247744


namespace NUMINAMATH_CALUDE_executive_committee_formation_l2477_247716

theorem executive_committee_formation (total_members : ℕ) (experienced_members : ℕ) (committee_size : ℕ) : 
  total_members = 30 →
  experienced_members = 8 →
  committee_size = 5 →
  (Finset.sum (Finset.range (Nat.min committee_size experienced_members + 1))
    (λ k => Nat.choose experienced_members k * Nat.choose (total_members - experienced_members) (committee_size - k))) = 116172 := by
  sorry

end NUMINAMATH_CALUDE_executive_committee_formation_l2477_247716


namespace NUMINAMATH_CALUDE_defective_units_shipped_l2477_247799

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ) :
  defective_rate = 0.1 →
  shipped_rate = 0.05 →
  (defective_rate * shipped_rate * 100) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l2477_247799


namespace NUMINAMATH_CALUDE_jiangxia_is_first_largest_bidirectional_l2477_247780

structure TidalPowerPlant where
  location : String
  year_built : Nat
  is_bidirectional : Bool
  is_largest : Bool

def china_tidal_plants : Nat := 9

def jiangxia_plant : TidalPowerPlant := {
  location := "Jiangxia",
  year_built := 1980,
  is_bidirectional := true,
  is_largest := true
}

theorem jiangxia_is_first_largest_bidirectional :
  ∃ (plant : TidalPowerPlant),
    plant.year_built = 1980 ∧
    plant.is_bidirectional = true ∧
    plant.is_largest = true ∧
    plant.location = "Jiangxia" :=
by
  sorry

#check jiangxia_is_first_largest_bidirectional

end NUMINAMATH_CALUDE_jiangxia_is_first_largest_bidirectional_l2477_247780


namespace NUMINAMATH_CALUDE_product_of_pairs_l2477_247786

/-- Given three pairs of real numbers satisfying specific equations, 
    their product in a certain form equals a specific value -/
theorem product_of_pairs (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (eq1₁ : x₁^3 - 3*x₁*y₁^2 = 2007)
  (eq2₁ : y₁^3 - 3*x₁^2*y₁ = 2006)
  (eq1₂ : x₂^3 - 3*x₂*y₂^2 = 2007)
  (eq2₂ : y₂^3 - 3*x₂^2*y₂ = 2006)
  (eq1₃ : x₃^3 - 3*x₃*y₃^2 = 2007)
  (eq2₃ : y₃^3 - 3*x₃^2*y₃ = 2006) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1/1003.5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_pairs_l2477_247786


namespace NUMINAMATH_CALUDE_divisibility_property_l2477_247755

theorem divisibility_property (a b : ℕ+) 
  (h : ∀ k : ℕ+, k < b → (b + k) ∣ (a + k)) :
  ∀ k : ℕ+, k < b → (b - k) ∣ (a - k) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2477_247755


namespace NUMINAMATH_CALUDE_slope_135_implies_y_negative_four_l2477_247741

/-- Given two points A and B, if the slope of the line passing through them is 135°, then the y-coordinate of A is -4. -/
theorem slope_135_implies_y_negative_four (x_a y_a x_b y_b : ℝ) :
  x_a = 3 →
  x_b = 2 →
  y_b = -3 →
  (y_a - y_b) / (x_a - x_b) = Real.tan (135 * π / 180) →
  y_a = -4 := by
  sorry

#check slope_135_implies_y_negative_four

end NUMINAMATH_CALUDE_slope_135_implies_y_negative_four_l2477_247741


namespace NUMINAMATH_CALUDE_set_equality_implies_a_values_l2477_247754

theorem set_equality_implies_a_values (a : ℝ) : 
  let A : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}
  let B : Set ℝ := {y | ∃ x ∈ A, y = x + 1}
  let C : Set ℝ := {y | ∃ x ∈ A, y = x^2}
  A.Nonempty → B = C → a = 0 ∨ a = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_values_l2477_247754


namespace NUMINAMATH_CALUDE_basketball_substitutions_remainder_l2477_247752

/-- The number of ways to make exactly k substitutions in a basketball game -/
def num_substitutions (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | k + 1 => 12 * (13 - k) * num_substitutions k

/-- The total number of ways to make substitutions in the basketball game -/
def total_substitutions : ℕ :=
  num_substitutions 0 + num_substitutions 1 + num_substitutions 2 + 
  num_substitutions 3 + num_substitutions 4

theorem basketball_substitutions_remainder :
  total_substitutions % 1000 = 953 := by
  sorry

end NUMINAMATH_CALUDE_basketball_substitutions_remainder_l2477_247752


namespace NUMINAMATH_CALUDE_superhero_movie_count_l2477_247794

/-- The number of movies watched by Dalton -/
def dalton_movies : ℕ := 7

/-- The number of movies watched by Hunter -/
def hunter_movies : ℕ := 12

/-- The number of movies watched by Alex -/
def alex_movies : ℕ := 15

/-- The number of movies watched together by all three -/
def movies_watched_together : ℕ := 2

/-- The total number of different movies watched -/
def total_different_movies : ℕ := dalton_movies + hunter_movies + alex_movies - 2 * movies_watched_together

theorem superhero_movie_count : total_different_movies = 32 := by
  sorry

end NUMINAMATH_CALUDE_superhero_movie_count_l2477_247794


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2477_247720

theorem batsman_average_increase (total_runs : ℕ → ℕ) (innings : ℕ) :
  innings = 17 →
  total_runs innings = total_runs (innings - 1) + 74 →
  (total_runs innings : ℚ) / innings = 26 →
  (total_runs innings : ℚ) / innings - (total_runs (innings - 1) : ℚ) / (innings - 1) = 3 := by
  sorry

#check batsman_average_increase

end NUMINAMATH_CALUDE_batsman_average_increase_l2477_247720


namespace NUMINAMATH_CALUDE_midpoint_specific_segment_l2477_247798

/-- Given two points in polar coordinates, returns their midpoint. -/
def polar_midpoint (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ :=
  sorry

theorem midpoint_specific_segment :
  let A : ℝ × ℝ := (5, π/4)
  let B : ℝ × ℝ := (5, 3*π/4)
  let M : ℝ × ℝ := polar_midpoint A.1 A.2 B.1 B.2
  M.1 = 5*Real.sqrt 2/2 ∧ M.2 = 3*π/8 :=
sorry

end NUMINAMATH_CALUDE_midpoint_specific_segment_l2477_247798


namespace NUMINAMATH_CALUDE_vowel_sum_is_twenty_l2477_247711

/-- The number of times each vowel was written on the board -/
structure VowelCounts where
  a : Nat
  e : Nat
  i : Nat
  o : Nat
  u : Nat

/-- The total sum of all the times vowels were written on the board -/
def total_vowel_count (vc : VowelCounts) : Nat :=
  vc.a + vc.e + vc.i + vc.o + vc.u

/-- Theorem: The sum of all vowel counts is 20 -/
theorem vowel_sum_is_twenty :
  ∃ (vc : VowelCounts),
    vc.a = 3 ∧
    vc.e = 5 ∧
    vc.i = 4 ∧
    vc.o = 2 ∧
    vc.u = 6 ∧
    total_vowel_count vc = 20 := by
  sorry

end NUMINAMATH_CALUDE_vowel_sum_is_twenty_l2477_247711


namespace NUMINAMATH_CALUDE_determinant_equals_zy_l2477_247718

def matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  !![1, x, z; 1, x+z, z; 1, y, y+z]

theorem determinant_equals_zy (x y z : ℝ) : 
  Matrix.det (matrix x y z) = z * y := by sorry

end NUMINAMATH_CALUDE_determinant_equals_zy_l2477_247718


namespace NUMINAMATH_CALUDE_tom_read_18_books_l2477_247740

/-- The number of books Tom read in May -/
def may_books : ℕ := 2

/-- The number of books Tom read in June -/
def june_books : ℕ := 6

/-- The number of books Tom read in July -/
def july_books : ℕ := 10

/-- The total number of books Tom read -/
def total_books : ℕ := may_books + june_books + july_books

theorem tom_read_18_books : total_books = 18 := by
  sorry

end NUMINAMATH_CALUDE_tom_read_18_books_l2477_247740


namespace NUMINAMATH_CALUDE_square_divisibility_l2477_247791

theorem square_divisibility (n d : ℕ+) : 
  (n.val % d.val = 0) → 
  ((n.val^2 + d.val^2) % (d.val^2 * n.val + 1) = 0) → 
  n = d^2 := by
sorry

end NUMINAMATH_CALUDE_square_divisibility_l2477_247791


namespace NUMINAMATH_CALUDE_class_composition_l2477_247773

theorem class_composition (total : ℕ) (boys girls : ℕ) : 
  total = 20 →
  boys + girls = total →
  (boys : ℚ) / total = 3/4 * ((girls : ℚ) / total) →
  boys = 12 ∧ girls = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_class_composition_l2477_247773


namespace NUMINAMATH_CALUDE_largest_integral_x_l2477_247769

theorem largest_integral_x : ∃ x : ℤ, x = 4 ∧ 
  (∀ y : ℤ, (1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 7/9 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_integral_x_l2477_247769


namespace NUMINAMATH_CALUDE_fraction_equality_l2477_247789

theorem fraction_equality : (1/5 - 1/6) / (1/3 - 1/4) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2477_247789


namespace NUMINAMATH_CALUDE_train_speed_l2477_247759

theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 160)
  (h2 : bridge_length = 215)
  (h3 : crossing_time = 30) : 
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2477_247759


namespace NUMINAMATH_CALUDE_probability_one_defective_l2477_247701

def total_products : ℕ := 10
def quality_products : ℕ := 7
def defective_products : ℕ := 3
def selected_products : ℕ := 4

theorem probability_one_defective :
  (Nat.choose quality_products (selected_products - 1) * Nat.choose defective_products 1) /
  Nat.choose total_products selected_products = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_defective_l2477_247701


namespace NUMINAMATH_CALUDE_sequence_fourth_term_l2477_247763

theorem sequence_fourth_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = 2 * a n - 2) : a 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sequence_fourth_term_l2477_247763


namespace NUMINAMATH_CALUDE_acme_vowel_soup_words_l2477_247783

/-- Represents the count of each vowel in the alphabet soup -/
structure VowelCount where
  a : Nat
  e : Nat
  i : Nat
  o : Nat
  u : Nat

/-- The modified Acme alphabet soup recipe -/
def acmeVowelSoup : VowelCount :=
  { a := 4, e := 6, i := 5, o := 3, u := 2 }

/-- The length of words to be formed -/
def wordLength : Nat := 5

/-- Calculates the number of five-letter words that can be formed from the given vowel counts -/
def countWords (vc : VowelCount) (len : Nat) : Nat :=
  sorry

/-- Theorem stating that the number of five-letter words from Acme Vowel Soup is 1125 -/
theorem acme_vowel_soup_words :
  countWords acmeVowelSoup wordLength = 1125 := by
  sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_words_l2477_247783


namespace NUMINAMATH_CALUDE_martha_cards_l2477_247721

/-- The number of cards Martha initially had -/
def initial_cards : ℕ := 3

/-- The number of cards Martha received from Emily -/
def cards_from_emily : ℕ := 76

/-- The total number of cards Martha ended up with -/
def total_cards : ℕ := 79

/-- Theorem stating that the initial number of cards plus the cards received equals the total cards -/
theorem martha_cards : initial_cards + cards_from_emily = total_cards := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_l2477_247721


namespace NUMINAMATH_CALUDE_min_paper_toys_l2477_247735

/-- Represents the number of paper toys that can be made from one sheet -/
structure PaperToy where
  boats : Nat
  planes : Nat

/-- The number of paper toys that can be made from one sheet -/
def sheet_capacity : PaperToy := { boats := 8, planes := 6 }

/-- The minimum number of sheets used for boats -/
def min_sheets_for_boats : Nat := 1

/-- Calculates the total number of toys given the number of boats and planes -/
def total_toys (boats planes : Nat) : Nat := boats + planes

/-- Theorem stating the minimum number of paper toys that can be made -/
theorem min_paper_toys :
  ∃ (boats planes : Nat),
    boats = min_sheets_for_boats * sheet_capacity.boats ∧
    planes = 0 ∧
    total_toys boats planes = 8 :=
sorry

end NUMINAMATH_CALUDE_min_paper_toys_l2477_247735


namespace NUMINAMATH_CALUDE_presidency_meeting_arrangements_count_l2477_247767

/- Define the number of schools -/
def num_schools : ℕ := 3

/- Define the number of members per school -/
def members_per_school : ℕ := 6

/- Define the number of representatives from the host school -/
def host_representatives : ℕ := 3

/- Define the number of representatives from each non-host school -/
def non_host_representatives : ℕ := 1

/- Function to calculate the number of ways to arrange the meeting -/
def presidency_meeting_arrangements : ℕ :=
  num_schools * (members_per_school.choose host_representatives) * 
  (members_per_school.choose non_host_representatives) * 
  (members_per_school.choose non_host_representatives)

/- Theorem stating the number of arrangements -/
theorem presidency_meeting_arrangements_count :
  presidency_meeting_arrangements = 2160 := by
  sorry

end NUMINAMATH_CALUDE_presidency_meeting_arrangements_count_l2477_247767


namespace NUMINAMATH_CALUDE_two_vertical_asymptotes_l2477_247795

-- Define the numerator and denominator of the rational function
def numerator (x : ℝ) : ℝ := x + 2
def denominator (x : ℝ) : ℝ := x^2 + 8*x + 15

-- Define a function to check if a given x-value is a vertical asymptote
def is_vertical_asymptote (x : ℝ) : Prop :=
  denominator x = 0 ∧ numerator x ≠ 0

-- Theorem stating that there are exactly 2 vertical asymptotes
theorem two_vertical_asymptotes :
  ∃ (a b : ℝ), a ≠ b ∧
    is_vertical_asymptote a ∧
    is_vertical_asymptote b ∧
    ∀ (x : ℝ), is_vertical_asymptote x → (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_two_vertical_asymptotes_l2477_247795


namespace NUMINAMATH_CALUDE_swimmers_speed_l2477_247712

/-- A swimmer's speed in still water, given downstream and upstream times and distances. -/
theorem swimmers_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 36) 
  (h2 : upstream_distance = 18) (h3 : downstream_time = 6) (h4 : upstream_time = 6) :
  ∃ (v_m v_s : ℝ), v_m + v_s = downstream_distance / downstream_time ∧
                   v_m - v_s = upstream_distance / upstream_time ∧
                   v_m = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_swimmers_speed_l2477_247712


namespace NUMINAMATH_CALUDE_second_concert_proof_l2477_247710

/-- The attendance of the first concert -/
def first_concert_attendance : ℕ := 65899

/-- The additional attendance at the second concert -/
def additional_attendance : ℕ := 119

/-- The attendance of the second concert -/
def second_concert_attendance : ℕ := first_concert_attendance + additional_attendance

theorem second_concert_proof : second_concert_attendance = 66018 := by
  sorry

end NUMINAMATH_CALUDE_second_concert_proof_l2477_247710


namespace NUMINAMATH_CALUDE_parallel_line_implies_a_value_l2477_247766

/-- Two points are on a line parallel to the y-axis if their x-coordinates are equal -/
def parallel_to_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = x₂

/-- The theorem stating that if M(a-3, a+4) and N(5, 9) form a line segment
    parallel to the y-axis, then a = 8 -/
theorem parallel_line_implies_a_value :
  ∀ a : ℝ,
  parallel_to_y_axis (a - 3) (a + 4) 5 9 →
  a = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_line_implies_a_value_l2477_247766


namespace NUMINAMATH_CALUDE_special_rectangle_perimeter_l2477_247776

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  small_perimeter : ℝ
  width_half_length : width = length / 2
  divides_into_three : length = 3 * width
  small_rect_perimeter : small_perimeter = 2 * (width + length / 3)
  small_perimeter_value : small_perimeter = 40

/-- The perimeter of a SpecialRectangle is 72 -/
theorem special_rectangle_perimeter (rect : SpecialRectangle) : 
  2 * (rect.length + rect.width) = 72 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_perimeter_l2477_247776


namespace NUMINAMATH_CALUDE_fractions_product_one_l2477_247731

theorem fractions_product_one :
  ∃ (a b c : ℕ), 
    2 ≤ a ∧ a ≤ 2016 ∧
    2 ≤ b ∧ b ≤ 2016 ∧
    2 ≤ c ∧ c ≤ 2016 ∧
    (a : ℚ) / (2018 - a) * (b : ℚ) / (2018 - b) * (c : ℚ) / (2018 - c) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fractions_product_one_l2477_247731


namespace NUMINAMATH_CALUDE_least_multiple_of_35_greater_than_500_l2477_247764

theorem least_multiple_of_35_greater_than_500 : ∃! n : ℕ, 
  35 * n > 500 ∧ ∀ m : ℕ, 35 * m > 500 → n ≤ m :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_35_greater_than_500_l2477_247764


namespace NUMINAMATH_CALUDE_pet_store_birds_l2477_247788

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 7

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 4

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 3

/-- The number of cockatiels in each cage -/
def cockatiels_per_cage : ℕ := 2

/-- The number of canaries in each cage -/
def canaries_per_cage : ℕ := 1

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage + cockatiels_per_cage + canaries_per_cage)

theorem pet_store_birds : total_birds = 70 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l2477_247788


namespace NUMINAMATH_CALUDE_marks_pond_depth_l2477_247796

/-- Given that Peter's pond is 5 feet deep and Mark's pond is 4 feet deeper than 3 times Peter's pond,
    prove that the depth of Mark's pond is 19 feet. -/
theorem marks_pond_depth (peters_depth : ℕ) (marks_depth : ℕ) 
  (h1 : peters_depth = 5)
  (h2 : marks_depth = 3 * peters_depth + 4) :
  marks_depth = 19 := by
  sorry

end NUMINAMATH_CALUDE_marks_pond_depth_l2477_247796


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l2477_247729

-- Equation 1
theorem equation_one_solutions (x : ℝ) : 
  x * (x - 2) = x - 2 ↔ x = 1 ∨ x = 2 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) : 
  x^2 - 6*x + 5 = 0 ↔ x = 1 ∨ x = 5 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l2477_247729


namespace NUMINAMATH_CALUDE_decagon_flip_impossible_l2477_247737

/-- Represents a point in the decagon configuration -/
structure Point where
  value : Int
  deriving Repr

/-- Represents the decagon configuration -/
structure DecagonConfig where
  points : List Point
  deriving Repr

/-- Represents an operation to flip signs -/
inductive FlipOperation
  | Side
  | Diagonal

/-- Applies a flip operation to the configuration -/
def applyFlip (config : DecagonConfig) (op : FlipOperation) : DecagonConfig :=
  sorry

/-- Checks if all points in the configuration are negative -/
def allNegative (config : DecagonConfig) : Bool :=
  sorry

/-- Theorem: It's impossible to make all points negative in a decagon configuration -/
theorem decagon_flip_impossible (initial : DecagonConfig) :
  ∀ (ops : List FlipOperation), ¬(allNegative (ops.foldl applyFlip initial)) :=
sorry

end NUMINAMATH_CALUDE_decagon_flip_impossible_l2477_247737


namespace NUMINAMATH_CALUDE_triangular_seating_theorem_l2477_247732

/-- Represents a triangular seating arrangement in a cinema -/
structure TriangularSeating where
  /-- The number of the best seat (at the center of the height from the top vertex) -/
  best_seat : ℕ
  /-- The total number of seats in the arrangement -/
  total_seats : ℕ

/-- 
Theorem: In a triangular seating arrangement where the best seat 
(at the center of the height from the top vertex) is numbered 265, 
the total number of seats is 1035.
-/
theorem triangular_seating_theorem (ts : TriangularSeating) 
  (h : ts.best_seat = 265) : ts.total_seats = 1035 := by
  sorry

#check triangular_seating_theorem

end NUMINAMATH_CALUDE_triangular_seating_theorem_l2477_247732


namespace NUMINAMATH_CALUDE_line_slope_points_l2477_247793

/-- Given m > 0 and three points on a line with slope m^2, prove m = √3 --/
theorem line_slope_points (m : ℝ) 
  (h_pos : m > 0)
  (h_line : ∃ (k b : ℝ), k = m^2 ∧ 
    3 = k * m + b ∧ 
    m = k * 1 + b ∧ 
    m^2 = k * 2 + b) : 
  m = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_points_l2477_247793


namespace NUMINAMATH_CALUDE_exists_triangle_from_polygon_with_inscribed_circle_l2477_247784

/-- A polygon with an inscribed circle. -/
structure PolygonWithInscribedCircle where
  /-- The number of sides of the polygon. -/
  n : ℕ
  /-- The lengths of the sides of the polygon. -/
  sides : Fin n → ℝ
  /-- The radius of the inscribed circle. -/
  radius : ℝ
  /-- All sides are positive. -/
  sides_positive : ∀ i, sides i > 0
  /-- The inscribed circle is tangent to all sides. -/
  tangent_to_all_sides : ∀ i, ∃ t, 0 < t ∧ t < sides i ∧ t = radius

/-- Theorem: In a polygon with an inscribed circle, there exist three sides that form a triangle. -/
theorem exists_triangle_from_polygon_with_inscribed_circle
  (p : PolygonWithInscribedCircle)
  (h : p.n ≥ 3) :
  ∃ i j k : Fin p.n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    p.sides i + p.sides j > p.sides k ∧
    p.sides j + p.sides k > p.sides i ∧
    p.sides k + p.sides i > p.sides j :=
  sorry

end NUMINAMATH_CALUDE_exists_triangle_from_polygon_with_inscribed_circle_l2477_247784


namespace NUMINAMATH_CALUDE_ali_seashells_l2477_247751

theorem ali_seashells (initial : ℕ) (given_to_friends : ℕ) (left_after_selling : ℕ) :
  initial = 180 →
  given_to_friends = 40 →
  left_after_selling = 55 →
  ∃ (given_to_brothers : ℕ),
    given_to_brothers = 30 ∧
    2 * left_after_selling = initial - given_to_friends - given_to_brothers :=
by sorry

end NUMINAMATH_CALUDE_ali_seashells_l2477_247751


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2477_247749

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (x - 1) * (x - 2*a + 1) < 0}
  (a = 1 → S = ∅) ∧
  (a > 1 → S = {x : ℝ | 1 < x ∧ x < 2*a - 1}) ∧
  (a < 1 → S = {x : ℝ | 2*a - 1 < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2477_247749


namespace NUMINAMATH_CALUDE_ball_probabilities_l2477_247703

def total_balls : ℕ := 4
def red_balls : ℕ := 2

def prob_two_red : ℚ := 1 / 6
def prob_at_least_one_red : ℚ := 5 / 6

theorem ball_probabilities :
  (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1)) = prob_two_red ∧
  1 - ((total_balls - red_balls) * (total_balls - red_balls - 1)) / (total_balls * (total_balls - 1)) = prob_at_least_one_red :=
by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2477_247703


namespace NUMINAMATH_CALUDE_derivative_symmetry_l2477_247738

/-- Given a function f(x) = ax^4 + bx^2 + c where f'(1) = 2, prove that f'(-1) = -2 -/
theorem derivative_symmetry (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^4 + b * x^2 + c
  (4 * a + 2 * b = 2) → 
  (fun (x : ℝ) => 4 * a * x^3 + 2 * b * x) (-1) = -2 := by
sorry

end NUMINAMATH_CALUDE_derivative_symmetry_l2477_247738


namespace NUMINAMATH_CALUDE_max_sum_of_unknown_pairs_l2477_247756

def pairwise_sums (a b c d : ℕ) : Finset ℕ :=
  {a + b, a + c, a + d, b + c, b + d, c + d}

theorem max_sum_of_unknown_pairs (a b c d : ℕ) :
  let sums := pairwise_sums a b c d
  ∀ x y, x ∈ sums → y ∈ sums →
    {210, 335, 296, 245, x, y} = sums →
    x + y ≤ 717 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_unknown_pairs_l2477_247756


namespace NUMINAMATH_CALUDE_line_perpendicular_transitive_parallel_lines_from_parallel_planes_not_always_parallel_transitive_not_always_parallel_from_intersections_l2477_247725

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Theorem 1
theorem line_perpendicular_transitive 
  (l m : Line) (α : Plane) :
  parallel m l → perpendicular m α → perpendicular l α :=
sorry

-- Theorem 2
theorem parallel_lines_from_parallel_planes 
  (l m : Line) (α β γ : Plane) :
  intersect α γ m → intersect β γ l → plane_parallel α β → parallel m l :=
sorry

-- Theorem 3
theorem not_always_parallel_transitive 
  (l m : Line) (α : Plane) :
  ¬(∀ l m α, parallel m l → parallel m α → parallel l α) :=
sorry

-- Theorem 4
theorem not_always_parallel_from_intersections 
  (l m n : Line) (α β γ : Plane) :
  ¬(∀ l m n α β γ, 
    intersect α β l → intersect β γ m → intersect γ α n → 
    parallel l m ∧ parallel m n ∧ parallel l n) :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_transitive_parallel_lines_from_parallel_planes_not_always_parallel_transitive_not_always_parallel_from_intersections_l2477_247725


namespace NUMINAMATH_CALUDE_substitution_elimination_l2477_247723

theorem substitution_elimination (x y : ℝ) : 
  (y = x - 5 ∧ 3*x - y = 8) → (3*x - x + 5 = 8) := by
  sorry

end NUMINAMATH_CALUDE_substitution_elimination_l2477_247723


namespace NUMINAMATH_CALUDE_volume_ratio_cone_cylinder_l2477_247774

/-- Given a cylinder and a cone with the same radius, where the cone's height is 1/3 of the cylinder's height,
    the ratio of the volume of the cone to the volume of the cylinder is 1/9. -/
theorem volume_ratio_cone_cylinder (r h : ℝ) (h_pos : 0 < r) (h_height : 0 < h) :
  (1 / 3 * π * r^2 * (h / 3)) / (π * r^2 * h) = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_volume_ratio_cone_cylinder_l2477_247774


namespace NUMINAMATH_CALUDE_meeting_time_prove_meeting_time_l2477_247777

/-- The time it takes for a motorcyclist and a cyclist to meet under specific conditions -/
theorem meeting_time : ℝ → Prop := fun t =>
  ∀ (D vm vb : ℝ),
  D > 0 →  -- Total distance between A and B is positive
  vm > 0 →  -- Motorcyclist's speed is positive
  vb > 0 →  -- Cyclist's speed is positive
  (1/3) * vm = D/2 + 2 →  -- Motorcyclist's position after 20 minutes
  (1/2) * vb = D/2 - 3 →  -- Cyclist's position after 30 minutes
  t * (vm + vb) = D →  -- They meet when they cover the total distance
  t = 24/60  -- The meeting time is 24 minutes (converted to hours)

/-- Proof of the meeting time theorem -/
theorem prove_meeting_time : meeting_time (24/60) := by
  sorry

end NUMINAMATH_CALUDE_meeting_time_prove_meeting_time_l2477_247777


namespace NUMINAMATH_CALUDE_am_length_l2477_247768

/-- Given points M, A, and B on a straight line, with AM twice as long as BM and AB = 6,
    the length of AM is either 4 or 12. -/
theorem am_length (M A B : ℝ) : 
  (∃ t : ℝ, M = t * A + (1 - t) * B) →  -- M, A, B are collinear
  abs (A - M) = 2 * abs (B - M) →       -- AM is twice as long as BM
  abs (A - B) = 6 →                     -- AB = 6
  abs (A - M) = 4 ∨ abs (A - M) = 12 := by
sorry


end NUMINAMATH_CALUDE_am_length_l2477_247768


namespace NUMINAMATH_CALUDE_even_periodic_function_range_l2477_247728

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem even_periodic_function_range (f : ℝ → ℝ) (a : ℝ) :
  IsEven f →
  HasPeriod f 3 →
  f 1 < 1 →
  f 5 = (2*a - 3) / (a + 1) →
  -1 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_even_periodic_function_range_l2477_247728


namespace NUMINAMATH_CALUDE_odd_periodic_function_value_l2477_247713

-- Define an odd function with period 2
def is_odd_periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = f x)

-- Theorem statement
theorem odd_periodic_function_value (f : ℝ → ℝ) 
  (h : is_odd_periodic_function f) : f 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_value_l2477_247713


namespace NUMINAMATH_CALUDE_quadratic_positivity_l2477_247772

theorem quadratic_positivity (a : ℝ) : 
  (∀ x, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_positivity_l2477_247772


namespace NUMINAMATH_CALUDE_min_value_theorem_l2477_247760

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * x^2 + 1 / x^2 ≥ 2 * Real.sqrt 3 ∧ ∃ y > 0, 3 * y^2 + 1 / y^2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2477_247760


namespace NUMINAMATH_CALUDE_nancy_file_deletion_l2477_247747

theorem nancy_file_deletion (initial_files : ℕ) (num_folders : ℕ) (files_per_folder : ℕ) : 
  initial_files = 80 → 
  num_folders = 7 → 
  files_per_folder = 7 → 
  initial_files - (num_folders * files_per_folder) = 31 := by
sorry

end NUMINAMATH_CALUDE_nancy_file_deletion_l2477_247747


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l2477_247757

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, n^2 = 900 ∧ 
  (∀ m : ℕ, m > 0 → m^2 < 900 → ¬(2 ∣ m^2 ∧ 3 ∣ m^2 ∧ 5 ∣ m^2)) ∧
  2 ∣ 900 ∧ 3 ∣ 900 ∧ 5 ∣ 900 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l2477_247757


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l2477_247705

/-- Given a cubic polynomial with three distinct real roots, 
    the equation formed by its product with its derivative 
    equals the square of its derivative has exactly two distinct real solutions. -/
theorem cubic_equation_solutions (a b c d : ℝ) (h : ∃ α β γ : ℝ, α < β ∧ β < γ ∧ 
  ∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = α ∨ x = β ∨ x = γ) :
  ∃! (s t : ℝ), s < t ∧ 
    ∀ x, 4 * (a * x^3 + b * x^2 + c * x + d) * (3 * a * x + b) = (3 * a * x^2 + 2 * b * x + c)^2 
    ↔ x = s ∨ x = t :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l2477_247705


namespace NUMINAMATH_CALUDE_fraction_problem_l2477_247785

theorem fraction_problem (x : ℚ) (h1 : x * 180 = 36) (h2 : x < 0.3) : x = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2477_247785


namespace NUMINAMATH_CALUDE_inverse_proposition_is_correct_l2477_247753

/-- The original proposition -/
def original_proposition (n : ℕ) : Prop :=
  n % 10 = 5 → n % 5 = 0

/-- The inverse proposition -/
def inverse_proposition (n : ℕ) : Prop :=
  n % 5 = 0 → n % 10 = 5

/-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition -/
theorem inverse_proposition_is_correct :
  inverse_proposition = λ n => ¬(original_proposition n) → ¬(n % 10 = 5) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_is_correct_l2477_247753


namespace NUMINAMATH_CALUDE_ship_speed_in_still_water_l2477_247706

theorem ship_speed_in_still_water :
  let downstream_distance : ℝ := 81
  let upstream_distance : ℝ := 69
  let water_flow_speed : ℝ := 2
  let ship_speed : ℝ := 25
  (downstream_distance / (ship_speed + water_flow_speed) =
   upstream_distance / (ship_speed - water_flow_speed)) →
  ship_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_ship_speed_in_still_water_l2477_247706


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l2477_247750

/-- A quadratic function with specific properties -/
def p (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_symmetry (d e f : ℝ) :
  (∀ x : ℝ, p d e f (10.5 + x) = p d e f (10.5 - x)) →  -- axis of symmetry at x = 10.5
  p d e f 3 = -5 →                                      -- passes through (3, -5)
  p d e f 12 = -5 :=                                    -- conclusion: p(12) = -5
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l2477_247750


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l2477_247748

theorem binomial_coefficient_equality (x : ℕ+) : 
  (Nat.choose 11 (2 * x.val - 1) = Nat.choose 11 x.val) → (x = 1 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l2477_247748


namespace NUMINAMATH_CALUDE_max_magnitude_c_l2477_247765

open Real

/-- Given vectors a and b, and a vector c satisfying the dot product condition,
    prove that the maximum magnitude of c is √2. -/
theorem max_magnitude_c (a b c : ℝ × ℝ) : 
  a = (1, 0) → 
  b = (0, 1) → 
  (c.1 + a.1, c.2 + a.2) • (c.1 + b.1, c.2 + b.2) = 0 → 
  (∀ c' : ℝ × ℝ, (c'.1 + a.1, c'.2 + a.2) • (c'.1 + b.1, c'.2 + b.2) = 0 → 
    Real.sqrt (c.1^2 + c.2^2) ≥ Real.sqrt (c'.1^2 + c'.2^2)) → 
  Real.sqrt (c.1^2 + c.2^2) = sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_max_magnitude_c_l2477_247765


namespace NUMINAMATH_CALUDE_birds_joining_fence_l2477_247790

/-- Proves that 2 additional birds joined the fence given the initial and final conditions -/
theorem birds_joining_fence :
  let initial_birds : ℕ := 3
  let initial_storks : ℕ := 4
  let additional_birds : ℕ := 2
  let final_birds : ℕ := initial_birds + additional_birds
  let final_storks : ℕ := initial_storks
  final_birds = final_storks + 1 :=
by sorry

end NUMINAMATH_CALUDE_birds_joining_fence_l2477_247790


namespace NUMINAMATH_CALUDE_onion_chopping_difference_l2477_247792

/-- Represents the rate of chopping onions in terms of number of onions and time in minutes -/
structure ChoppingRate where
  onions : ℕ
  minutes : ℕ

/-- Calculates the number of onions chopped in a given time based on a chopping rate -/
def chop_onions (rate : ChoppingRate) (time : ℕ) : ℕ :=
  (rate.onions * time) / rate.minutes

theorem onion_chopping_difference :
  let brittney_rate : ChoppingRate := ⟨15, 5⟩
  let carl_rate : ChoppingRate := ⟨20, 5⟩
  let time : ℕ := 30
  chop_onions carl_rate time - chop_onions brittney_rate time = 30 := by
  sorry

end NUMINAMATH_CALUDE_onion_chopping_difference_l2477_247792


namespace NUMINAMATH_CALUDE_unique_solution_l2477_247736

theorem unique_solution : ∃! x : ℤ, x^2 + 105 = (x - 20)^2 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2477_247736


namespace NUMINAMATH_CALUDE_equation_solution_l2477_247745

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (7 * x / (x - 2) - 5 / (x - 2) = 2 / (x - 2)) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2477_247745


namespace NUMINAMATH_CALUDE_circle_transform_prime_impossibility_l2477_247761

/-- Represents the transformation of four numbers on a circle -/
def circle_transform (a b c d : ℤ) : (ℤ × ℤ × ℤ × ℤ) :=
  (a - b, b - c, c - d, d - a)

/-- Applies the circle transformation n times -/
def iterate_transform (n : ℕ) (a b c d : ℤ) : (ℤ × ℤ × ℤ × ℤ) :=
  match n with
  | 0 => (a, b, c, d)
  | n + 1 =>
    let (a', b', c', d') := iterate_transform n a b c d
    circle_transform a' b' c' d'

/-- Checks if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem circle_transform_prime_impossibility :
  ∀ a b c d : ℤ,
  let (a', b', c', d') := iterate_transform 1996 a b c d
  ¬(is_prime (|b' * c' - a' * d'|.natAbs) ∧
    is_prime (|a' * c' - b' * d'|.natAbs) ∧
    is_prime (|a' * b' - c' * d'|.natAbs)) := by
  sorry

end NUMINAMATH_CALUDE_circle_transform_prime_impossibility_l2477_247761


namespace NUMINAMATH_CALUDE_fraction_equality_l2477_247707

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 18)
  (h2 : p / n = 6)
  (h3 : p / q = 1 / 15) :
  m / q = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2477_247707


namespace NUMINAMATH_CALUDE_remainder_sum_l2477_247717

theorem remainder_sum (n : ℤ) : n % 20 = 13 → (n % 4 + n % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2477_247717


namespace NUMINAMATH_CALUDE_total_pieces_is_4000_l2477_247715

/-- The number of pieces in the first puzzle -/
def first_puzzle_pieces : ℕ := 1000

/-- The number of pieces in the second and third puzzles -/
def other_puzzle_pieces : ℕ := first_puzzle_pieces + first_puzzle_pieces / 2

/-- The total number of pieces in all three puzzles -/
def total_pieces : ℕ := first_puzzle_pieces + 2 * other_puzzle_pieces

/-- Theorem stating that the total number of pieces in all three puzzles is 4000 -/
theorem total_pieces_is_4000 : total_pieces = 4000 := by
  sorry

end NUMINAMATH_CALUDE_total_pieces_is_4000_l2477_247715


namespace NUMINAMATH_CALUDE_f_less_than_g_for_x_greater_than_one_l2477_247719

/-- Given functions f and g with specified properties, f(x) < g(x) for x > 1 -/
theorem f_less_than_g_for_x_greater_than_one 
  (f g : ℝ → ℝ)
  (h_f : ∀ x, f x = Real.log x)
  (h_g : ∃ a b : ℝ, ∀ x, g x = a * x + b / x)
  (h_common_tangent : ∃ x₀, x₀ > 0 ∧ f x₀ = g x₀ ∧ (deriv f) x₀ = (deriv g) x₀)
  (x : ℝ)
  (h_x : x > 1) :
  f x < g x := by
sorry

end NUMINAMATH_CALUDE_f_less_than_g_for_x_greater_than_one_l2477_247719


namespace NUMINAMATH_CALUDE_order_silk_total_l2477_247797

/-- The total yards of silk dyed for an order, given the yards of green and pink silk. -/
def total_silk_dyed (green_silk : ℕ) (pink_silk : ℕ) : ℕ :=
  green_silk + pink_silk

/-- Theorem stating that the total yards of silk dyed for the order is 111421 yards. -/
theorem order_silk_total : 
  total_silk_dyed 61921 49500 = 111421 := by
  sorry

end NUMINAMATH_CALUDE_order_silk_total_l2477_247797


namespace NUMINAMATH_CALUDE_short_story_pages_approx_l2477_247742

/-- Calculates the number of pages in each short story --/
def pages_per_short_story (stories_per_week : ℕ) (weeks : ℕ) (reams : ℕ) 
  (sheets_per_ream : ℕ) (pages_per_sheet : ℕ) : ℚ :=
  let total_sheets := reams * sheets_per_ream
  let total_pages := total_sheets * pages_per_sheet
  let total_stories := stories_per_week * weeks
  (total_pages : ℚ) / total_stories

theorem short_story_pages_approx : 
  let result := pages_per_short_story 3 12 3 500 2
  ∃ ε > 0, |result - 83.33| < ε := by
  sorry

end NUMINAMATH_CALUDE_short_story_pages_approx_l2477_247742
