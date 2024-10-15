import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l130_13034

theorem equation_solution : 
  ∃ (x : ℤ), (1 + 1 / x : ℚ) ^ (x + 1) = (1 + 1 / 2003 : ℚ) ^ 2003 :=
by
  use -2004
  sorry

end NUMINAMATH_CALUDE_equation_solution_l130_13034


namespace NUMINAMATH_CALUDE_complex_fourth_power_l130_13046

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l130_13046


namespace NUMINAMATH_CALUDE_max_value_tangent_double_angle_l130_13067

/-- Given a function f(x) = 3sin(x) + cos(x) that reaches its maximum value at x = α, 
    prove that tan(2α) = -3/4 -/
theorem max_value_tangent_double_angle (f : ℝ → ℝ) (α : ℝ) 
  (h₁ : ∀ x, f x = 3 * Real.sin x + Real.cos x)
  (h₂ : IsLocalMax f α) : 
  Real.tan (2 * α) = -3/4 := by sorry

end NUMINAMATH_CALUDE_max_value_tangent_double_angle_l130_13067


namespace NUMINAMATH_CALUDE_english_test_questions_l130_13073

theorem english_test_questions (math_questions : ℕ) (math_correct_percentage : ℚ)
  (english_correct_percentage : ℚ) (total_correct : ℕ) :
  math_questions = 40 →
  math_correct_percentage = 75 / 100 →
  english_correct_percentage = 98 / 100 →
  total_correct = 79 →
  ∃ (english_questions : ℕ),
    english_questions = 50 ∧
    (math_questions : ℚ) * math_correct_percentage +
    (english_questions : ℚ) * english_correct_percentage = total_correct :=
by sorry

end NUMINAMATH_CALUDE_english_test_questions_l130_13073


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l130_13059

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_fraction :
  (3 - 4 * i) / (5 - 2 * i) = 7 / 29 - (14 / 29) * i :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l130_13059


namespace NUMINAMATH_CALUDE_coffee_shop_multiple_l130_13084

theorem coffee_shop_multiple (x : ℕ) : 
  (32 = x * 6 + 8) → x = 4 := by sorry

end NUMINAMATH_CALUDE_coffee_shop_multiple_l130_13084


namespace NUMINAMATH_CALUDE_lcm_gcd_product_12_15_l130_13091

theorem lcm_gcd_product_12_15 : Nat.lcm 12 15 * Nat.gcd 12 15 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_12_15_l130_13091


namespace NUMINAMATH_CALUDE_golden_section_addition_correct_l130_13025

/-- The 0.618 method for finding the optimal addition amount --/
def golden_section_addition (a b x : ℝ) : ℝ :=
  a + b - x

/-- Theorem stating the correct formula for the addition point in the 0.618 method --/
theorem golden_section_addition_correct (a b x : ℝ) 
  (h_range : a ≤ x ∧ x ≤ b) 
  (h_good_point : x = a + 0.618 * (b - a)) : 
  golden_section_addition a b x = a + b - x :=
by sorry

end NUMINAMATH_CALUDE_golden_section_addition_correct_l130_13025


namespace NUMINAMATH_CALUDE_unique_function_theorem_l130_13023

def is_valid_function (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃! k : ℕ, k > 0 ∧ (f^[k] n ≤ n + k + 1)

theorem unique_function_theorem :
  ∀ f : ℕ → ℕ, is_valid_function f → ∀ n : ℕ, f n = n + 2 := by sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l130_13023


namespace NUMINAMATH_CALUDE_complex_quadrant_problem_l130_13070

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_quadrant_problem (a : ℝ) :
  is_purely_imaginary (Complex.mk (a^2 - 3*a - 4) (a - 4)) →
  a = -1 ∧ a < 0 ∧ -a > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_quadrant_problem_l130_13070


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_15_13_l130_13010

theorem half_abs_diff_squares_15_13 : 
  (1/2 : ℝ) * |15^2 - 13^2| = 28 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_15_13_l130_13010


namespace NUMINAMATH_CALUDE_average_expenditure_feb_to_jul_l130_13036

def average_expenditure_jan_to_jun : ℝ := 4200
def expenditure_january : ℝ := 1200
def expenditure_july : ℝ := 1500
def num_months : ℕ := 6

theorem average_expenditure_feb_to_jul :
  let total_jan_to_jun := average_expenditure_jan_to_jun * num_months
  let total_feb_to_jun := total_jan_to_jun - expenditure_january
  let total_feb_to_jul := total_feb_to_jun + expenditure_july
  total_feb_to_jul / num_months = 4250 := by
sorry

end NUMINAMATH_CALUDE_average_expenditure_feb_to_jul_l130_13036


namespace NUMINAMATH_CALUDE_rehana_age_l130_13009

/-- Represents the ages of Rehana, Phoebe, and Jacob -/
structure Ages where
  rehana : ℕ
  phoebe : ℕ
  jacob : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.jacob = 3 ∧
  ages.jacob = (3 * ages.phoebe) / 5 ∧
  ages.rehana + 5 = 3 * (ages.phoebe + 5)

/-- The theorem stating Rehana's current age -/
theorem rehana_age :
  ∃ (ages : Ages), problem_conditions ages ∧ ages.rehana = 25 := by
  sorry

end NUMINAMATH_CALUDE_rehana_age_l130_13009


namespace NUMINAMATH_CALUDE_conference_lefthandedness_l130_13097

theorem conference_lefthandedness 
  (total : ℕ) 
  (red : ℕ) 
  (blue : ℕ) 
  (h1 : red + blue = total) 
  (h2 : red = 7 * blue / 3) 
  (red_left : ℕ) 
  (blue_left : ℕ) 
  (h3 : red_left = red / 3) 
  (h4 : blue_left = 2 * blue / 3) : 
  (red_left + blue_left : ℚ) / total = 13 / 30 := by
sorry

end NUMINAMATH_CALUDE_conference_lefthandedness_l130_13097


namespace NUMINAMATH_CALUDE_inequality_proof_l130_13083

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + 4*b^2)) + (b^3 / (b^2 + 4*c^2)) + (c^3 / (c^2 + 4*a^2)) ≥ (a + b + c) / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l130_13083


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l130_13026

theorem right_triangle_acute_angles (α β : ℝ) : 
  α = 60 → β = 90 → α + β + (180 - α - β) = 180 → 180 - α - β = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l130_13026


namespace NUMINAMATH_CALUDE_mn_equals_six_l130_13080

/-- Given that -x³yⁿ and 3xᵐy² are like terms, prove that mn = 6 -/
theorem mn_equals_six (x y : ℝ) (m n : ℕ) 
  (h : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → -x^3 * y^n = 3 * x^m * y^2) : 
  m * n = 6 := by
  sorry

end NUMINAMATH_CALUDE_mn_equals_six_l130_13080


namespace NUMINAMATH_CALUDE_power_function_through_2_4_l130_13013

/-- A power function passing through the point (2, 4) has exponent 2 -/
theorem power_function_through_2_4 :
  ∀ a : ℝ, (2 : ℝ) ^ a = 4 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_2_4_l130_13013


namespace NUMINAMATH_CALUDE_f_comp_f_four_roots_l130_13002

/-- A quadratic function f(x) = x^2 + 10x + d -/
def f (d : ℝ) (x : ℝ) : ℝ := x^2 + 10*x + d

/-- The composition of f with itself -/
def f_comp_f (d : ℝ) (x : ℝ) : ℝ := f d (f d x)

/-- The theorem stating the condition for f(f(x)) to have exactly 4 distinct real roots -/
theorem f_comp_f_four_roots (d : ℝ) :
  (∃ (a b c e : ℝ), a < b ∧ b < c ∧ c < e ∧
    (∀ x : ℝ, f_comp_f d x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = e)) ↔
  d < 25 :=
sorry

end NUMINAMATH_CALUDE_f_comp_f_four_roots_l130_13002


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l130_13041

/-- The line (2k-1)x-(k+3)y-(k-11)=0 passes through the point (2, 3) for all values of k. -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l130_13041


namespace NUMINAMATH_CALUDE_required_moles_of_reactants_l130_13014

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String

-- Define the molar ratio
def molarRatio : ℚ := 1

-- Define the desired amount of products
def desiredProduct : ℚ := 3

-- Define the chemical equation
def chemicalEquation : Reaction := {
  reactant1 := "AgNO3"
  reactant2 := "NaOH"
  product1 := "AgOH"
  product2 := "NaNO3"
}

-- Theorem statement
theorem required_moles_of_reactants :
  let requiredReactant1 := desiredProduct * molarRatio
  let requiredReactant2 := desiredProduct * molarRatio
  requiredReactant1 = 3 ∧ requiredReactant2 = 3 :=
sorry

end NUMINAMATH_CALUDE_required_moles_of_reactants_l130_13014


namespace NUMINAMATH_CALUDE_octagon_perimeter_in_cm_l130_13003

/-- Regular octagon with side length in meters -/
structure RegularOctagon where
  side_length : ℝ

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

/-- Sum of all side lengths of a regular octagon in centimeters -/
def sum_side_lengths (octagon : RegularOctagon) : ℝ :=
  8 * octagon.side_length * meters_to_cm

theorem octagon_perimeter_in_cm (octagon : RegularOctagon) 
    (h : octagon.side_length = 2.3) : 
    sum_side_lengths octagon = 1840 := by
  sorry

end NUMINAMATH_CALUDE_octagon_perimeter_in_cm_l130_13003


namespace NUMINAMATH_CALUDE_quadratic_no_roots_l130_13030

/-- Given a quadratic function f(x) = ax^2 + bx + c where b is the geometric mean of a and c,
    prove that f(x) has no real roots. -/
theorem quadratic_no_roots (a b c : ℝ) (h : b^2 = a*c) (h_a : a ≠ 0) (h_c : c ≠ 0) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_l130_13030


namespace NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l130_13044

/-- The number of distinct arrangements of n beads on a necklace,
    considering rotational and reflectional symmetry -/
def necklace_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of distinct arrangements
    of 8 beads on a necklace is 2520 -/
theorem eight_bead_necklace_arrangements :
  necklace_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l130_13044


namespace NUMINAMATH_CALUDE_certain_number_equation_l130_13053

theorem certain_number_equation (x : ℝ) : (40 * 30 + (x + 8) * 3) / 5 = 1212 ↔ x = 1612 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l130_13053


namespace NUMINAMATH_CALUDE_turtle_count_difference_l130_13096

theorem turtle_count_difference (owen_initial : ℕ) (owen_final : ℕ) : 
  owen_initial = 21 →
  owen_final = 50 →
  ∃ (johanna_initial : ℕ),
    johanna_initial < owen_initial ∧
    owen_final = 2 * owen_initial + johanna_initial / 2 ∧
    owen_initial - johanna_initial = 5 :=
by sorry

end NUMINAMATH_CALUDE_turtle_count_difference_l130_13096


namespace NUMINAMATH_CALUDE_linear_system_solution_l130_13071

theorem linear_system_solution (x y : ℝ) : 
  3 * x + 2 * y = 2 → 2 * x + 3 * y = 8 → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l130_13071


namespace NUMINAMATH_CALUDE_union_A_B_intersection_complement_A_B_C_subset_B_iff_l130_13056

-- Define the sets A, B, and C
def A : Set ℝ := {x | (x - 2) / (x - 7) < 0}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- State the theorems to be proved
theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := by sorry

theorem intersection_complement_A_B : (Set.univ \ A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10} := by sorry

theorem C_subset_B_iff (a : ℝ) : C a ⊆ B ↔ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_complement_A_B_C_subset_B_iff_l130_13056


namespace NUMINAMATH_CALUDE_decagon_painting_count_l130_13022

/-- The number of ways to choose 4 colors from 8 available colors -/
def choose_colors : ℕ := Nat.choose 8 4

/-- The number of circular permutations of 4 colors -/
def circular_permutations : ℕ := Nat.factorial 3

/-- The number of distinct colorings of a decagon -/
def decagon_colorings : ℕ := choose_colors * circular_permutations / 2

/-- Theorem stating the number of distinct ways to paint the decagon -/
theorem decagon_painting_count : decagon_colorings = 210 := by
  sorry

#eval decagon_colorings

end NUMINAMATH_CALUDE_decagon_painting_count_l130_13022


namespace NUMINAMATH_CALUDE_geometric_sequence_150th_term_l130_13042

/-- Given a geometric sequence with first term 5 and second term -10,
    the 150th term is equal to -5 * 2^149 -/
theorem geometric_sequence_150th_term :
  let a₁ : ℝ := 5
  let a₂ : ℝ := -10
  let r : ℝ := a₂ / a₁
  let a₁₅₀ : ℝ := a₁ * r^149
  a₁₅₀ = -5 * 2^149 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_150th_term_l130_13042


namespace NUMINAMATH_CALUDE_oranges_per_crate_l130_13029

theorem oranges_per_crate :
  ∀ (num_crates num_boxes nectarines_per_box total_fruit : ℕ),
    num_crates = 12 →
    num_boxes = 16 →
    nectarines_per_box = 30 →
    total_fruit = 2280 →
    total_fruit = num_boxes * nectarines_per_box + num_crates * (total_fruit - num_boxes * nectarines_per_box) / num_crates →
    (total_fruit - num_boxes * nectarines_per_box) / num_crates = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_oranges_per_crate_l130_13029


namespace NUMINAMATH_CALUDE_bellas_to_annes_height_ratio_l130_13043

/-- Proves that given the conditions in the problem, the ratio of Bella's height to Anne's height is 3:1 -/
theorem bellas_to_annes_height_ratio : 
  ∀ (anne_height bella_height sister_height : ℝ),
  anne_height = 2 * sister_height →
  anne_height = 80 →
  bella_height - sister_height = 200 →
  bella_height / anne_height = 3 := by
sorry

end NUMINAMATH_CALUDE_bellas_to_annes_height_ratio_l130_13043


namespace NUMINAMATH_CALUDE_triple_hash_40_l130_13081

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.3 * N + 2

-- State the theorem
theorem triple_hash_40 : hash (hash (hash 40)) = 3.86 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_40_l130_13081


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l130_13051

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_second_term : a 2 = 1/3)
  (h_eighth_term : a 8 = 27) :
  a 5 = 3 ∨ a 5 = -3 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l130_13051


namespace NUMINAMATH_CALUDE_prob_both_red_is_one_ninth_l130_13078

/-- The probability of drawing a red ball from both bags A and B -/
def prob_both_red (red_a white_a red_b white_b : ℕ) : ℚ :=
  (red_a : ℚ) / (red_a + white_a) * (red_b : ℚ) / (red_b + white_b)

/-- Theorem: The probability of drawing a red ball from both Bag A and Bag B is 1/9 -/
theorem prob_both_red_is_one_ninth :
  prob_both_red 4 2 1 5 = 1 / 9 := by
  sorry

#eval prob_both_red 4 2 1 5

end NUMINAMATH_CALUDE_prob_both_red_is_one_ninth_l130_13078


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_eight_l130_13055

theorem sum_of_solutions_is_eight : 
  ∃ (N₁ N₂ : ℝ), N₁ * (N₁ - 8) = 7 ∧ N₂ * (N₂ - 8) = 7 ∧ N₁ + N₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_eight_l130_13055


namespace NUMINAMATH_CALUDE_certain_number_problem_l130_13057

theorem certain_number_problem (n x : ℝ) : 
  (n - 4) / x = 7 + (8 / x) → x = 6 → n = 54 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l130_13057


namespace NUMINAMATH_CALUDE_a_investment_amount_verify_profit_share_l130_13037

/-- Represents the investment scenario described in the problem -/
structure Investment where
  a_amount : ℝ  -- A's investment amount
  b_amount : ℝ := 200  -- B's investment amount
  a_months : ℝ := 12  -- Duration of A's investment in months
  b_months : ℝ := 6  -- Duration of B's investment in months
  total_profit : ℝ := 100  -- Total profit
  a_profit : ℝ := 50  -- A's share of the profit

/-- Theorem stating that A's investment amount must be $100 given the conditions -/
theorem a_investment_amount (inv : Investment) : 
  (inv.a_amount * inv.a_months) / (inv.b_amount * inv.b_months) = 1 →
  inv.a_amount = 100 := by
  sorry

/-- Corollary confirming that the calculated investment satisfies the profit sharing condition -/
theorem verify_profit_share (inv : Investment) : 
  inv.a_amount = 100 →
  (inv.a_amount * inv.a_months) / (inv.b_amount * inv.b_months) = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_investment_amount_verify_profit_share_l130_13037


namespace NUMINAMATH_CALUDE_robot_trajectory_constraint_l130_13016

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The trajectory of the robot -/
def robotTrajectory : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- The line x = -1 -/
def verticalLine : Line :=
  { slope := 0, yIntercept := -1 }

/-- The point F(1, 0) -/
def pointF : Point :=
  { x := 1, y := 0 }

/-- The point P(-1, 0) -/
def pointP : Point :=
  { x := -1, y := 0 }

/-- The line passing through P(-1, 0) with slope k -/
def lineThroughP (k : ℝ) : Line :=
  { slope := k, yIntercept := k }

/-- The robot's trajectory does not intersect the line through P -/
def noIntersection (k : ℝ) : Prop :=
  ∀ p : Point, p ∈ robotTrajectory → p ∉ {p : Point | p.y = (lineThroughP k).slope * (p.x + 1)}

theorem robot_trajectory_constraint (k : ℝ) :
  noIntersection k ↔ k > 1 ∨ k < -1 :=
sorry

end NUMINAMATH_CALUDE_robot_trajectory_constraint_l130_13016


namespace NUMINAMATH_CALUDE_circle_radius_l130_13017

theorem circle_radius (x y : ℝ) : 
  (x^2 - 10*x + y^2 + 4*y + 13 = 0) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 4^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l130_13017


namespace NUMINAMATH_CALUDE_outfit_combinations_l130_13039

theorem outfit_combinations (shirts : ℕ) (hats : ℕ) : shirts = 5 → hats = 3 → shirts * hats = 15 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l130_13039


namespace NUMINAMATH_CALUDE_isabel_math_homework_pages_l130_13088

/-- Proves that Isabel had 2 pages of math homework given the problem conditions -/
theorem isabel_math_homework_pages :
  ∀ (total_pages math_pages reading_pages : ℕ) 
    (problems_per_page total_problems : ℕ),
  reading_pages = 4 →
  problems_per_page = 5 →
  total_problems = 30 →
  total_pages = math_pages + reading_pages →
  total_problems = total_pages * problems_per_page →
  math_pages = 2 := by
sorry


end NUMINAMATH_CALUDE_isabel_math_homework_pages_l130_13088


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l130_13089

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_terms : 
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (b n) (b (n + 1)) ≤ k ∧ 
  ∃ (m : ℕ), Nat.gcd (b m) (b (m + 1)) = k :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l130_13089


namespace NUMINAMATH_CALUDE_picture_frame_dimensions_l130_13001

theorem picture_frame_dimensions (a b : ℕ+) : 
  (a : ℤ) * b = ((a + 2) * (b + 2) : ℤ) - a * b → 
  ((a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4)) :=
by sorry

end NUMINAMATH_CALUDE_picture_frame_dimensions_l130_13001


namespace NUMINAMATH_CALUDE_cara_age_difference_l130_13021

/-- The age difference between Cara and her mom -/
def age_difference (grandmother_age mom_age_difference cara_age : ℕ) : ℕ :=
  grandmother_age - mom_age_difference - cara_age

/-- Proof that Cara is 20 years younger than her mom -/
theorem cara_age_difference :
  age_difference 75 15 40 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cara_age_difference_l130_13021


namespace NUMINAMATH_CALUDE_garden_perimeter_l130_13007

/-- Given a square garden with area q and perimeter p, if q = 2p + 20, then p = 40 -/
theorem garden_perimeter (q p : ℝ) (h1 : q > 0) (h2 : p > 0) (h3 : q = p^2 / 16) (h4 : q = 2*p + 20) : p = 40 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l130_13007


namespace NUMINAMATH_CALUDE_fraction_simplification_l130_13047

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 * x) / (x^2 - 1) - 1 / (x - 1) = 1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l130_13047


namespace NUMINAMATH_CALUDE_shifted_sine_equals_cosine_l130_13035

open Real

theorem shifted_sine_equals_cosine (ω φ : ℝ) (h_ω : ω < 0) :
  (∀ x, sin (ω * (x - π / 12) + φ) = cos (2 * x)) →
  ∃ k : ℤ, φ = π / 3 + 2 * π * ↑k := by sorry

end NUMINAMATH_CALUDE_shifted_sine_equals_cosine_l130_13035


namespace NUMINAMATH_CALUDE_initial_students_on_bus_l130_13061

theorem initial_students_on_bus (students_left_bus : ℕ) (students_remaining : ℕ) 
  (h1 : students_left_bus = 3) 
  (h2 : students_remaining = 7) : 
  students_left_bus + students_remaining = 10 := by
sorry

end NUMINAMATH_CALUDE_initial_students_on_bus_l130_13061


namespace NUMINAMATH_CALUDE_sam_chewing_gums_l130_13069

theorem sam_chewing_gums (total : ℕ) (mary : ℕ) (sue : ℕ) (h1 : total = 30) (h2 : mary = 5) (h3 : sue = 15) : 
  total - mary - sue = 10 := by
  sorry

end NUMINAMATH_CALUDE_sam_chewing_gums_l130_13069


namespace NUMINAMATH_CALUDE_differential_of_y_l130_13049

noncomputable section

open Real

-- Define the function y
def y (x : ℝ) : ℝ := x * (sin (log x) - cos (log x))

-- State the theorem
theorem differential_of_y (x : ℝ) (h : x > 0) :
  deriv y x = 2 * sin (log x) :=
by sorry

end

end NUMINAMATH_CALUDE_differential_of_y_l130_13049


namespace NUMINAMATH_CALUDE_not_divisible_by_four_sum_of_digits_l130_13072

def numbers : List Nat := [3674, 3684, 3694, 3704, 3714, 3722]

theorem not_divisible_by_four_sum_of_digits : 
  ∃ n ∈ numbers, 
    ¬(n % 4 = 0) ∧ 
    (n % 10 + (n / 10) % 10 = 11) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_four_sum_of_digits_l130_13072


namespace NUMINAMATH_CALUDE_expression_value_l130_13052

theorem expression_value (a b c : ℝ) (ha : a = 20) (hb : b = 40) (hc : c = 10) :
  (a - (b - c)) - ((a - b) - c) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l130_13052


namespace NUMINAMATH_CALUDE_parallel_vectors_implies_m_equals_one_l130_13019

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, prove that if they are parallel, then m = 1 -/
theorem parallel_vectors_implies_m_equals_one :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (m, m + 1)
  are_parallel a b → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_implies_m_equals_one_l130_13019


namespace NUMINAMATH_CALUDE_chord_length_l130_13005

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (t : ℝ) : 
  let x := 1 + 2*t
  let y := 2 + t
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 9}
  let line := {(x, y) : ℝ × ℝ | ∃ t, x = 1 + 2*t ∧ y = 2 + t}
  let intersection := circle ∩ line
  ∃ p q : ℝ × ℝ, p ∈ intersection ∧ q ∈ intersection ∧ 
    dist p q = 12/5 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_chord_length_l130_13005


namespace NUMINAMATH_CALUDE_line_equal_intercepts_l130_13031

/-- A line passing through (1,2) with equal x and y intercepts has equation x+y-3=0 or 2x-y=0 -/
theorem line_equal_intercepts :
  ∀ (L : Set (ℝ × ℝ)), 
    ((1, 2) ∈ L) →
    (∃ a : ℝ, a ≠ 0 ∧ (a, 0) ∈ L ∧ (0, a) ∈ L) →
    (∀ x y : ℝ, (x, y) ∈ L ↔ (x + y = 3 ∨ 2*x - y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equal_intercepts_l130_13031


namespace NUMINAMATH_CALUDE_alberts_earnings_l130_13082

-- Define Albert's original earnings
def original_earnings : ℝ := 660

-- Theorem statement
theorem alberts_earnings :
  let scenario1 := original_earnings * 1.14 * 0.9
  let scenario2 := original_earnings * 1.15 * 1.2 * 0.9
  (scenario1 = 678) → (scenario2 = 819.72) := by
  sorry

end NUMINAMATH_CALUDE_alberts_earnings_l130_13082


namespace NUMINAMATH_CALUDE_six_students_solved_only_B_l130_13068

/-- Represents the number of students who solved each combination of problems -/
structure ProblemSolvers where
  a : ℕ  -- only A
  b : ℕ  -- only B
  c : ℕ  -- only C
  d : ℕ  -- A and B
  e : ℕ  -- A and C
  f : ℕ  -- B and C
  g : ℕ  -- A, B, and C

/-- The conditions of the math competition problem -/
def MathCompetitionConditions (s : ProblemSolvers) : Prop :=
  -- Total number of students is 25
  s.a + s.b + s.c + s.d + s.e + s.f + s.g = 25 ∧
  -- Among students who didn't solve A, number solving B is twice the number solving C
  s.b + s.f = 2 * (s.c + s.f) ∧
  -- Number of students solving only A is one more than number of students solving A among remaining students
  s.a = s.d + s.e + s.g + 1 ∧
  -- Among students solving only one problem, half didn't solve A
  s.a = s.b + s.c

/-- The theorem stating that 6 students solved only problem B -/
theorem six_students_solved_only_B (s : ProblemSolvers) 
  (h : MathCompetitionConditions s) : s.b = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_students_solved_only_B_l130_13068


namespace NUMINAMATH_CALUDE_expression_simplification_l130_13033

theorem expression_simplification (x : ℝ) (h : x = 1 + Real.sqrt 3) :
  (x + 3) / (x^2 - 2*x + 1) * (x - 1) / (x^2 + 3*x) + 1 / x = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l130_13033


namespace NUMINAMATH_CALUDE_range_of_m_l130_13004

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, m * x^2 + 1 ≤ 0) ∧ 
  (¬ ∃ x : ℝ, x^2 + m * x + 1 < 0) → 
  0 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l130_13004


namespace NUMINAMATH_CALUDE_power_of_three_plus_five_mod_eight_l130_13092

theorem power_of_three_plus_five_mod_eight :
  (3^100 + 5) % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_plus_five_mod_eight_l130_13092


namespace NUMINAMATH_CALUDE_lowest_dropped_score_l130_13098

theorem lowest_dropped_score (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
  (a + b + c + d) / 4 = 45 →
  d ≤ a ∧ d ≤ b ∧ d ≤ c →
  (a + b + c) / 3 = 50 →
  d = 30 := by
sorry

end NUMINAMATH_CALUDE_lowest_dropped_score_l130_13098


namespace NUMINAMATH_CALUDE_kim_dropped_one_class_l130_13006

/-- Calculates the number of classes dropped given the initial number of classes,
    hours per class, and remaining total hours of classes. -/
def classes_dropped (initial_classes : ℕ) (hours_per_class : ℕ) (remaining_hours : ℕ) : ℕ :=
  (initial_classes * hours_per_class - remaining_hours) / hours_per_class

theorem kim_dropped_one_class :
  classes_dropped 4 2 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_kim_dropped_one_class_l130_13006


namespace NUMINAMATH_CALUDE_point_movement_and_quadrant_l130_13024

/-- Given a point P with coordinates (a - 7, 3 - 2a) in the Cartesian coordinate system,
    which is moved 4 units up and 5 units right to obtain point Q (a - 2, 7 - 2a),
    prove that for Q to be in the first quadrant, 2 < a < 3.5,
    and when a is an integer satisfying this condition, P = (-4, -3) and Q = (1, 1). -/
theorem point_movement_and_quadrant (a : ℝ) :
  let P : ℝ × ℝ := (a - 7, 3 - 2*a)
  let Q : ℝ × ℝ := (a - 2, 7 - 2*a)
  (∀ x y, Q = (x, y) → x > 0 ∧ y > 0) ↔ (2 < a ∧ a < 3.5) ∧
  (∃ n : ℤ, ↑n = a ∧ 2 < a ∧ a < 3.5) →
    P = (-4, -3) ∧ Q = (1, 1) :=
by sorry

end NUMINAMATH_CALUDE_point_movement_and_quadrant_l130_13024


namespace NUMINAMATH_CALUDE_complex_expression_equality_l130_13020

/-- Given complex numbers a and b, prove that 2a - 3bi equals 22 - 12i -/
theorem complex_expression_equality (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + 4*I) :
  2*a - 3*b*I = 22 - 12*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l130_13020


namespace NUMINAMATH_CALUDE_trees_in_yard_l130_13015

theorem trees_in_yard (yard_length : ℕ) (tree_distance : ℕ) (h1 : yard_length = 300) (h2 : tree_distance = 12) : 
  yard_length / tree_distance + 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l130_13015


namespace NUMINAMATH_CALUDE_rectangular_plot_theorem_l130_13058

/-- Represents a rectangular plot with given properties --/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ
  length_breadth_difference : ℝ

/-- Theorem stating the properties of the rectangular plot --/
theorem rectangular_plot_theorem (plot : RectangularPlot) : 
  plot.length = plot.breadth + plot.length_breadth_difference ∧
  plot.total_fencing_cost = plot.fencing_cost_per_meter * (2 * plot.length + 2 * plot.breadth) ∧
  plot.length = 65 ∧
  plot.fencing_cost_per_meter = 26.5 ∧
  plot.total_fencing_cost = 5300 →
  plot.length_breadth_difference = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_theorem_l130_13058


namespace NUMINAMATH_CALUDE_unique_winning_combination_l130_13008

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 60 ∧ ∃ (m n : ℕ), n = 2^m * 3^n

def is_valid_combination (combo : Finset ℕ) : Prop :=
  combo.card = 5 ∧ 
  ∀ n ∈ combo, is_valid_number n ∧
  ∃ k : ℕ, (combo.prod id) = 12^k

theorem unique_winning_combination : 
  ∃! combo : Finset ℕ, is_valid_combination combo :=
sorry

end NUMINAMATH_CALUDE_unique_winning_combination_l130_13008


namespace NUMINAMATH_CALUDE_cookies_sold_l130_13027

theorem cookies_sold (total : ℕ) (ratio_brownies : ℕ) (ratio_cookies : ℕ) (cookies : ℕ) : 
  total = 104 →
  ratio_brownies = 7 →
  ratio_cookies = 6 →
  ratio_brownies * cookies = ratio_cookies * (total - cookies) →
  cookies = 48 := by
sorry

end NUMINAMATH_CALUDE_cookies_sold_l130_13027


namespace NUMINAMATH_CALUDE_bacterial_growth_result_l130_13095

/-- Represents the bacterial population growth model -/
structure BacterialGrowth where
  initial_population : ℕ
  triple_rate : ℕ  -- number of 5-minute intervals where population triples
  double_rate : ℕ  -- number of 10-minute intervals where population doubles

/-- Calculates the final population given a BacterialGrowth model -/
def final_population (model : BacterialGrowth) : ℕ :=
  model.initial_population * (3 ^ model.triple_rate) * (2 ^ model.double_rate)

/-- Theorem stating that under the given conditions, the final population is 16200 -/
theorem bacterial_growth_result :
  let model : BacterialGrowth := {
    initial_population := 50,
    triple_rate := 4,
    double_rate := 2
  }
  final_population model = 16200 := by sorry

end NUMINAMATH_CALUDE_bacterial_growth_result_l130_13095


namespace NUMINAMATH_CALUDE_sum_of_digits_N_l130_13048

/-- The sum of digits function for natural numbers -/
noncomputable def sumOfDigits (n : ℕ) : ℕ := sorry

/-- N is defined as the positive integer whose square is 36^49 * 49^36 * 81^25 -/
noncomputable def N : ℕ := sorry

/-- Theorem stating that the sum of digits of N is 21 -/
theorem sum_of_digits_N : sumOfDigits N = 21 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_N_l130_13048


namespace NUMINAMATH_CALUDE_set_union_problem_l130_13062

theorem set_union_problem (M N : Set ℕ) (x : ℕ) :
  M = {0, x} →
  N = {1, 2} →
  M ∩ N = {2} →
  M ∪ N = {0, 1, 2} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l130_13062


namespace NUMINAMATH_CALUDE_kenneth_earnings_l130_13085

theorem kenneth_earnings (spent_percentage : Real) (remaining_amount : Real) (total_earnings : Real) : 
  spent_percentage = 0.1 →
  remaining_amount = 405 →
  remaining_amount = (1 - spent_percentage) * total_earnings →
  total_earnings = 450 :=
by sorry

end NUMINAMATH_CALUDE_kenneth_earnings_l130_13085


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l130_13063

theorem tan_alpha_plus_pi_fourth (α : Real) (m : Real) (h : m ≠ 0) :
  let P : Real × Real := (m, -2*m)
  (∃ k : Real, k > 0 ∧ P = (k * Real.cos α, k * Real.sin α)) →
  Real.tan (α + π/4) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l130_13063


namespace NUMINAMATH_CALUDE_billy_carnival_tickets_l130_13099

theorem billy_carnival_tickets : ∀ (ferris_rides bumper_rides ticket_per_ride : ℕ),
  ferris_rides = 7 →
  bumper_rides = 3 →
  ticket_per_ride = 5 →
  (ferris_rides + bumper_rides) * ticket_per_ride = 50 := by
  sorry

end NUMINAMATH_CALUDE_billy_carnival_tickets_l130_13099


namespace NUMINAMATH_CALUDE_inner_circle_to_triangle_ratio_l130_13090

/-- The ratio of the area of the innermost circle to the area of the equilateral triangle --/
theorem inner_circle_to_triangle_ratio (s : ℝ) (h : s = 10) :
  let R := s * Real.sqrt 3 / 6
  let a := 2 * R
  let r := a / 2
  let A_triangle := Real.sqrt 3 / 4 * s^2
  let A_circle := Real.pi * r^2
  A_circle / A_triangle = Real.pi * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_inner_circle_to_triangle_ratio_l130_13090


namespace NUMINAMATH_CALUDE_dalton_has_excess_money_l130_13087

def jump_rope_cost : ℝ := 7
def board_game_cost : ℝ := 12
def ball_cost : ℝ := 4
def jump_rope_discount : ℝ := 2
def ball_discount : ℝ := 1
def jump_rope_quantity : ℕ := 3
def board_game_quantity : ℕ := 2
def ball_quantity : ℕ := 4
def allowance_savings : ℝ := 30
def uncle_money : ℝ := 25
def grandma_money : ℝ := 10
def sales_tax_rate : ℝ := 0.08

def total_cost_before_discounts : ℝ :=
  jump_rope_cost * jump_rope_quantity +
  board_game_cost * board_game_quantity +
  ball_cost * ball_quantity

def total_discounts : ℝ :=
  jump_rope_discount * jump_rope_quantity +
  ball_discount * ball_quantity

def total_cost_after_discounts : ℝ :=
  total_cost_before_discounts - total_discounts

def sales_tax : ℝ :=
  total_cost_after_discounts * sales_tax_rate

def final_total_cost : ℝ :=
  total_cost_after_discounts + sales_tax

def total_money_dalton_has : ℝ :=
  allowance_savings + uncle_money + grandma_money

theorem dalton_has_excess_money :
  total_money_dalton_has - final_total_cost = 9.92 := by sorry

end NUMINAMATH_CALUDE_dalton_has_excess_money_l130_13087


namespace NUMINAMATH_CALUDE_original_number_proof_l130_13011

theorem original_number_proof (x : ℝ) : 3 * (2 * x + 5) = 123 ↔ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l130_13011


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l130_13028

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) ≥ 4 :=
by sorry

theorem equality_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) = 4 ↔ a^2 = 2 ∧ b = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l130_13028


namespace NUMINAMATH_CALUDE_sum_squared_l130_13054

theorem sum_squared (x y : ℝ) (h1 : x * (x + y) = 24) (h2 : y * (x + y) = 72) :
  (x + y)^2 = 96 := by
sorry

end NUMINAMATH_CALUDE_sum_squared_l130_13054


namespace NUMINAMATH_CALUDE_lcm_gcd_1560_1040_l130_13040

theorem lcm_gcd_1560_1040 :
  (Nat.lcm 1560 1040 = 1560) ∧ (Nat.gcd 1560 1040 = 520) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_1560_1040_l130_13040


namespace NUMINAMATH_CALUDE_third_smallest_number_indeterminate_l130_13045

theorem third_smallest_number_indeterminate 
  (a b c d : ℕ) 
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (h_sum1 : a + b + c = 21)
  (h_sum2 : a + b + d = 27)
  (h_sum3 : a + c + d = 30) :
  ¬∃(n : ℕ), ∀(x : ℕ), (x = c) ↔ (x = n) :=
sorry

end NUMINAMATH_CALUDE_third_smallest_number_indeterminate_l130_13045


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l130_13012

theorem trigonometric_simplification :
  (Real.sqrt (1 + 2 * Real.sin (610 * π / 180) * Real.cos (430 * π / 180))) /
  (Real.sin (250 * π / 180) + Real.cos (790 * π / 180)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l130_13012


namespace NUMINAMATH_CALUDE_sum_of_integers_l130_13076

theorem sum_of_integers : (47 : ℤ) + (-27 : ℤ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l130_13076


namespace NUMINAMATH_CALUDE_sqrt_five_exists_and_unique_l130_13018

theorem sqrt_five_exists_and_unique :
  ∃! (y : ℝ), y > 0 ∧ y^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_five_exists_and_unique_l130_13018


namespace NUMINAMATH_CALUDE_sum_160_45_base4_l130_13032

/-- Convert a decimal number to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Convert a list of base 4 digits to decimal -/
def fromBase4 (l : List ℕ) : ℕ :=
  sorry

/-- Add two numbers in base 4 -/
def addBase4 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_160_45_base4 :
  addBase4 (toBase4 160) (toBase4 45) = [2, 4, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_sum_160_45_base4_l130_13032


namespace NUMINAMATH_CALUDE_sales_after_three_years_l130_13077

/-- The number of televisions sold initially -/
def initial_sales : ℕ := 327

/-- The annual increase rate as a percentage -/
def increase_rate : ℚ := 20 / 100

/-- The number of years for which the sales increase -/
def years : ℕ := 3

/-- Function to calculate sales after a given number of years -/
def sales_after_years (initial : ℕ) (rate : ℚ) (n : ℕ) : ℚ :=
  initial * (1 + rate) ^ n

/-- Theorem stating that the sales after 3 years is approximately 565 -/
theorem sales_after_three_years :
  ∃ ε > 0, |sales_after_years initial_sales increase_rate years - 565| < ε :=
sorry

end NUMINAMATH_CALUDE_sales_after_three_years_l130_13077


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l130_13000

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m α) : 
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l130_13000


namespace NUMINAMATH_CALUDE_prob_all_red_first_is_half_l130_13038

/-- The number of red chips in the hat -/
def num_red_chips : ℕ := 3

/-- The number of green chips in the hat -/
def num_green_chips : ℕ := 3

/-- The total number of chips in the hat -/
def total_chips : ℕ := num_red_chips + num_green_chips

/-- The probability of drawing all red chips before all green chips -/
def prob_all_red_first : ℚ :=
  (Nat.choose (total_chips - 1) num_green_chips) / (Nat.choose total_chips num_red_chips)

/-- Theorem stating that the probability of drawing all red chips first is 1/2 -/
theorem prob_all_red_first_is_half : prob_all_red_first = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_red_first_is_half_l130_13038


namespace NUMINAMATH_CALUDE_quadrangular_prism_has_12_edges_l130_13066

/-- Number of edges in a prism with n sides -/
def prism_edges (n : ℕ) : ℕ := 3 * n

/-- Number of edges in a pyramid with n sides -/
def pyramid_edges (n : ℕ) : ℕ := 2 * n

theorem quadrangular_prism_has_12_edges :
  prism_edges 4 = 12 ∧
  pyramid_edges 4 ≠ 12 ∧
  pyramid_edges 5 ≠ 12 ∧
  prism_edges 5 ≠ 12 :=
by sorry

end NUMINAMATH_CALUDE_quadrangular_prism_has_12_edges_l130_13066


namespace NUMINAMATH_CALUDE_negation_equivalence_l130_13065

theorem negation_equivalence :
  (¬ ∀ x : ℝ, |x - 2| + |x - 4| > 3) ↔ (∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l130_13065


namespace NUMINAMATH_CALUDE_problem_solution_l130_13060

theorem problem_solution (x y z : ℚ) : 
  x = 2/3 → y = 3/2 → z = 1/3 → (1/3) * x^7 * y^5 * z^4 = 11/600 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l130_13060


namespace NUMINAMATH_CALUDE_john_used_four_quarters_l130_13086

/-- The number of quarters John used to pay for a candy bar -/
def quarters_used (candy_cost dime_value nickel_value quarter_value : ℕ) 
  (num_dimes : ℕ) (change : ℕ) : ℕ :=
  ((candy_cost + change) - (num_dimes * dime_value + nickel_value)) / quarter_value

/-- Theorem stating that John used 4 quarters to pay for the candy bar -/
theorem john_used_four_quarters :
  quarters_used 131 10 5 25 3 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_john_used_four_quarters_l130_13086


namespace NUMINAMATH_CALUDE_win_sector_area_l130_13050

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/7) :
  let total_area := π * r^2
  let win_area := p * total_area
  win_area = 192 * π / 7 := by
sorry

end NUMINAMATH_CALUDE_win_sector_area_l130_13050


namespace NUMINAMATH_CALUDE_ajay_work_days_l130_13075

/-- The number of days it takes Vijay to complete the work alone -/
def vijay_days : ℝ := 24

/-- The number of days it takes Ajay and Vijay to complete the work together -/
def together_days : ℝ := 6

/-- The number of days it takes Ajay to complete the work alone -/
noncomputable def ajay_days : ℝ := 
  (vijay_days * together_days) / (vijay_days - together_days)

theorem ajay_work_days : ajay_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_ajay_work_days_l130_13075


namespace NUMINAMATH_CALUDE_tessellation_theorem_l130_13079

/-- Represents a regular polygon -/
structure RegularPolygon where
  sides : ℕ
  interiorAngle : ℝ

/-- Checks if two regular polygons can tessellate -/
def canTessellate (p1 p2 : RegularPolygon) : Prop :=
  ∃ (n m : ℕ), n * p1.interiorAngle + m * p2.interiorAngle = 360

theorem tessellation_theorem :
  let triangle : RegularPolygon := ⟨3, 60⟩
  let square : RegularPolygon := ⟨4, 90⟩
  let hexagon : RegularPolygon := ⟨6, 120⟩
  let octagon : RegularPolygon := ⟨8, 135⟩
  
  (canTessellate triangle square ∧
   canTessellate triangle hexagon ∧
   canTessellate octagon square) ∧
  ¬(canTessellate hexagon square) :=
by sorry

end NUMINAMATH_CALUDE_tessellation_theorem_l130_13079


namespace NUMINAMATH_CALUDE_equilateral_triangle_division_l130_13064

theorem equilateral_triangle_division : 
  ∃ (k m : ℕ), 2007 = 9 + 3 * k ∧ 2008 = 4 + 3 * m :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_division_l130_13064


namespace NUMINAMATH_CALUDE_triangle_formation_l130_13094

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if a set of three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 4 5 6 ∧
  ¬can_form_triangle 2 2 5 ∧
  ¬can_form_triangle 1 (Real.sqrt 3) 3 ∧
  ¬can_form_triangle 3 4 8 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l130_13094


namespace NUMINAMATH_CALUDE_at_most_one_negative_l130_13074

theorem at_most_one_negative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b - c ≤ 0 → b + c - a > 0 ∧ c + a - b > 0) ∧
  (b + c - a ≤ 0 → a + b - c > 0 ∧ c + a - b > 0) ∧
  (c + a - b ≤ 0 → a + b - c > 0 ∧ b + c - a > 0) :=
sorry

end NUMINAMATH_CALUDE_at_most_one_negative_l130_13074


namespace NUMINAMATH_CALUDE_gathering_women_count_l130_13093

/-- Represents a gathering with men and women dancing --/
structure Gathering where
  num_men : ℕ
  num_women : ℕ
  men_dance_count : ℕ
  women_dance_count : ℕ

/-- Theorem: In a gathering where each man dances with 4 women, each woman dances with 3 men, 
    and there are 15 men, the number of women is 20 --/
theorem gathering_women_count (g : Gathering) 
  (h1 : g.num_men = 15)
  (h2 : g.men_dance_count = 4)
  (h3 : g.women_dance_count = 3)
  : g.num_women = 20 := by
  sorry

end NUMINAMATH_CALUDE_gathering_women_count_l130_13093
