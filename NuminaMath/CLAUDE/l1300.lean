import Mathlib

namespace NUMINAMATH_CALUDE_tracy_book_collection_l1300_130084

theorem tracy_book_collection (first_week : ℕ) (total_books : ℕ) : 
  total_books = 99 → 
  first_week + 5 * (10 * first_week) = total_books →
  first_week = 9 := by
sorry

end NUMINAMATH_CALUDE_tracy_book_collection_l1300_130084


namespace NUMINAMATH_CALUDE_max_n_when_T_less_than_2019_l1300_130063

/-- Define the arithmetic sequence a_n -/
def a (n : ℕ) : ℕ := 2 * n - 1

/-- Define the geometric sequence b_n -/
def b (n : ℕ) : ℕ := 2^(n - 1)

/-- Define the sequence c_n -/
def c (n : ℕ) : ℕ := a (b n)

/-- Define the sum T_n -/
def T (n : ℕ) : ℕ := 2^(n + 1) - n - 2

theorem max_n_when_T_less_than_2019 :
  (∀ n : ℕ, n ≤ 9 → T n < 2019) ∧ T 10 ≥ 2019 := by sorry

end NUMINAMATH_CALUDE_max_n_when_T_less_than_2019_l1300_130063


namespace NUMINAMATH_CALUDE_range_of_a_l1300_130080

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + x - 2 > 0
def q (x a : ℝ) : Prop := x > a

-- Define what it means for q to be sufficient but not necessary for p
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, q x a → p x) ∧ ¬(∀ x, p x → q x a)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary a → a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1300_130080


namespace NUMINAMATH_CALUDE_union_and_complement_l1300_130064

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {2, 7}

theorem union_and_complement :
  (A ∪ B = {2, 4, 5, 7}) ∧ (Aᶜ = {1, 3, 6, 7}) := by
  sorry

end NUMINAMATH_CALUDE_union_and_complement_l1300_130064


namespace NUMINAMATH_CALUDE_free_throw_probabilities_l1300_130072

/-- The probability of player A scoring a free throw -/
def prob_A : ℚ := 1/2

/-- The probability of player B scoring a free throw -/
def prob_B : ℚ := 2/5

/-- The probability of both A and B scoring their free throws -/
def prob_both_score : ℚ := prob_A * prob_B

/-- The probability of at least one of A or B scoring their free throw -/
def prob_at_least_one_scores : ℚ := 1 - (1 - prob_A) * (1 - prob_B)

theorem free_throw_probabilities :
  (prob_both_score = 1/5) ∧ (prob_at_least_one_scores = 7/10) := by
  sorry

end NUMINAMATH_CALUDE_free_throw_probabilities_l1300_130072


namespace NUMINAMATH_CALUDE_rhombus_line_equations_l1300_130051

-- Define the rhombus ABCD
structure Rhombus where
  A : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ

-- Define the rhombus with given coordinates
def given_rhombus : Rhombus := {
  A := (-4, 7)
  C := (2, -3)
  P := (3, -1)
}

-- Define a line equation
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

-- Theorem statement
theorem rhombus_line_equations (ABCD : Rhombus) 
  (h1 : ABCD = given_rhombus) :
  ∃ (line_AD line_BD : LineEquation),
    (line_AD.a = 2 ∧ line_AD.b = -1 ∧ line_AD.c = 15) ∧
    (line_BD.a = 3 ∧ line_BD.b = -5 ∧ line_BD.c = 13) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_line_equations_l1300_130051


namespace NUMINAMATH_CALUDE_scaled_building_height_l1300_130043

/-- Calculates the height of a scaled model building given the original building's height and the volumes of water held in the top portions of both the original and the model. -/
theorem scaled_building_height
  (original_height : ℝ)
  (original_volume : ℝ)
  (model_volume : ℝ)
  (h_original_height : original_height = 120)
  (h_original_volume : original_volume = 30000)
  (h_model_volume : model_volume = 0.03)
  : ∃ (model_height : ℝ), model_height = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_scaled_building_height_l1300_130043


namespace NUMINAMATH_CALUDE_sqrt_2_3_5_not_arithmetic_progression_l1300_130049

theorem sqrt_2_3_5_not_arithmetic_progression : ¬ ∃ (d : ℝ), Real.sqrt 3 = Real.sqrt 2 + d ∧ Real.sqrt 5 = Real.sqrt 2 + 2 * d := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_3_5_not_arithmetic_progression_l1300_130049


namespace NUMINAMATH_CALUDE_x_minus_y_equals_four_l1300_130092

theorem x_minus_y_equals_four (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_four_l1300_130092


namespace NUMINAMATH_CALUDE_earthquake_energy_ratio_l1300_130008

-- Define the Richter scale energy relation
def richter_energy_ratio (x : ℝ) : ℝ := 10

-- Define the frequency function type
def frequency := ℝ → ℝ

-- Theorem statement
theorem earthquake_energy_ratio 
  (f : frequency) 
  (x y : ℝ) 
  (h1 : y - x = 2) 
  (h2 : f y = 2 * f x) :
  (richter_energy_ratio ^ y) / (richter_energy_ratio ^ x) = 200 := by
  sorry

end NUMINAMATH_CALUDE_earthquake_energy_ratio_l1300_130008


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1300_130089

theorem polar_to_rectangular_conversion :
  let r : ℝ := 5
  let θ : ℝ := 5 * Real.pi / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -5 * Real.sqrt 2 / 2) ∧ (y = -5 * Real.sqrt 2 / 2) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1300_130089


namespace NUMINAMATH_CALUDE_divisor_problem_l1300_130083

theorem divisor_problem (D : ℕ) : 
  D > 0 ∧
  242 % D = 11 ∧
  698 % D = 18 ∧
  (242 + 698) % D = 9 →
  D = 20 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l1300_130083


namespace NUMINAMATH_CALUDE_intersection_union_problem_l1300_130073

theorem intersection_union_problem (x : ℝ) : 
  let A : Set ℝ := {1, 3, 5}
  let B : Set ℝ := {1, 2, x^2 - 1}
  (A ∩ B = {1, 3}) → (x = -2 ∧ A ∪ B = {1, 2, 3, 5}) := by
sorry

end NUMINAMATH_CALUDE_intersection_union_problem_l1300_130073


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1300_130002

-- Define the type for dice rolls
def DiceRoll := Fin 6

-- Define the condition for the angle being greater than 90°
def angleGreaterThan90 (m n : DiceRoll) : Prop :=
  (m.val : ℤ) - (n.val : ℤ) > 0

-- Define the probability space
def totalOutcomes : ℕ := 36

-- Define the number of favorable outcomes
def favorableOutcomes : ℕ := 15

-- State the theorem
theorem dice_roll_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1300_130002


namespace NUMINAMATH_CALUDE_inequality_proof_l1300_130097

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) ≥ 
  2 / (1 + a) + 2 / (1 + b) + 2 / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1300_130097


namespace NUMINAMATH_CALUDE_smallest_triple_consecutive_sum_l1300_130061

def sum_of_consecutive (n : ℕ) (k : ℕ) : ℕ := 
  k * n + k * (k - 1) / 2

def is_sum_of_consecutive (x : ℕ) (k : ℕ) : Prop :=
  ∃ n : ℕ, sum_of_consecutive n k = x

theorem smallest_triple_consecutive_sum : 
  (∀ m : ℕ, m < 105 → ¬(is_sum_of_consecutive m 5 ∧ is_sum_of_consecutive m 6 ∧ is_sum_of_consecutive m 7)) ∧ 
  (is_sum_of_consecutive 105 5 ∧ is_sum_of_consecutive 105 6 ∧ is_sum_of_consecutive 105 7) :=
sorry

end NUMINAMATH_CALUDE_smallest_triple_consecutive_sum_l1300_130061


namespace NUMINAMATH_CALUDE_f_sum_eq_two_l1300_130082

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * (Real.sin x) ^ 2 + b * Real.tan x + 1

theorem f_sum_eq_two (a b : ℝ) (h : f a b 2 = 5) : f a b (Real.pi - 2) + f a b Real.pi = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_eq_two_l1300_130082


namespace NUMINAMATH_CALUDE_pigeonhole_principle_buttons_l1300_130045

theorem pigeonhole_principle_buttons : ∀ (r w b : ℕ),
  r ≥ 3 ∧ w ≥ 3 ∧ b ≥ 3 →
  ∀ n : ℕ, n ≥ 7 →
  ∀ f : Fin n → Fin 3,
  ∃ c : Fin 3, ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
  f i = c ∧ f j = c ∧ f k = c :=
by
  sorry

#check pigeonhole_principle_buttons

end NUMINAMATH_CALUDE_pigeonhole_principle_buttons_l1300_130045


namespace NUMINAMATH_CALUDE_net_folds_to_partial_cube_l1300_130095

/-- Represents a net that can be folded into a cube -/
structure Net where
  faces : Finset (Fin 6)
  edges : Finset (Fin 12)
  holes : Finset (Fin 12)

/-- Represents a partial cube -/
structure PartialCube where
  faces : Finset (Fin 6)
  edges : Finset (Fin 12)
  holes : Finset (Fin 12)

/-- A net can be folded into a partial cube -/
def canFoldInto (n : Net) (pc : PartialCube) : Prop :=
  n.faces = pc.faces ∧ n.edges = pc.edges ∧ n.holes = pc.holes

/-- The given partial cube has holes on the edges of four different faces -/
axiom partial_cube_property (pc : PartialCube) :
  ∃ (f1 f2 f3 f4 : Fin 6) (e1 e2 e3 e4 : Fin 12),
    f1 ≠ f2 ∧ f1 ≠ f3 ∧ f1 ≠ f4 ∧ f2 ≠ f3 ∧ f2 ≠ f4 ∧ f3 ≠ f4 ∧
    e1 ∈ pc.holes ∧ e2 ∈ pc.holes ∧ e3 ∈ pc.holes ∧ e4 ∈ pc.holes

/-- Theorem: A net can be folded into the given partial cube if and only if
    it has holes on the edges of four different faces -/
theorem net_folds_to_partial_cube (n : Net) (pc : PartialCube) :
  canFoldInto n pc ↔
    ∃ (f1 f2 f3 f4 : Fin 6) (e1 e2 e3 e4 : Fin 12),
      f1 ≠ f2 ∧ f1 ≠ f3 ∧ f1 ≠ f4 ∧ f2 ≠ f3 ∧ f2 ≠ f4 ∧ f3 ≠ f4 ∧
      e1 ∈ n.holes ∧ e2 ∈ n.holes ∧ e3 ∈ n.holes ∧ e4 ∈ n.holes :=
by sorry

end NUMINAMATH_CALUDE_net_folds_to_partial_cube_l1300_130095


namespace NUMINAMATH_CALUDE_election_win_margin_l1300_130025

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) :
  winner_votes = 1944 →
  (winner_votes : ℚ) / total_votes = 54 / 100 →
  winner_votes - (total_votes - winner_votes) = 288 :=
by
  sorry

end NUMINAMATH_CALUDE_election_win_margin_l1300_130025


namespace NUMINAMATH_CALUDE_no_points_in_circle_l1300_130003

theorem no_points_in_circle (r : ℝ) (A B : ℝ × ℝ) : r = 1 → 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 →
  ¬∃ P : ℝ × ℝ, (P.1 - A.1)^2 + (P.2 - A.2)^2 < r^2 ∧ 
                (P.1 - B.1)^2 + (P.2 - B.2)^2 < r^2 ∧
                (P.1 - A.1)^2 + (P.2 - A.2)^2 + (P.1 - B.1)^2 + (P.2 - B.2)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_no_points_in_circle_l1300_130003


namespace NUMINAMATH_CALUDE_total_seashells_l1300_130060

-- Define the variables
def seashells_given_to_tom : ℕ := 49
def seashells_left_with_mike : ℕ := 13

-- Define the theorem
theorem total_seashells :
  seashells_given_to_tom + seashells_left_with_mike = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l1300_130060


namespace NUMINAMATH_CALUDE_diamond_calculation_l1300_130015

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation : 
  let x := diamond (diamond 2 3) 4
  let y := diamond 2 (diamond 3 4)
  x - y = -29 / 132 := by sorry

end NUMINAMATH_CALUDE_diamond_calculation_l1300_130015


namespace NUMINAMATH_CALUDE_second_derivative_at_one_l1300_130076

-- Define the function f
def f (x : ℝ) : ℝ := (1 - 2 * x^3)^10

-- State the theorem
theorem second_derivative_at_one (x : ℝ) : 
  (deriv (deriv f)) 1 = 60 := by sorry

end NUMINAMATH_CALUDE_second_derivative_at_one_l1300_130076


namespace NUMINAMATH_CALUDE_nth_term_formula_l1300_130054

/-- Represents the coefficient of the nth term in the sequence -/
def coeff (n : ℕ) : ℕ := n + 1

/-- Represents the exponent of 'a' in the nth term of the sequence -/
def exponent (n : ℕ) : ℕ := n

/-- Represents the nth term in the sequence as a function of 'a' -/
def nthTerm (n : ℕ) (a : ℝ) : ℝ := (coeff n : ℝ) * (a ^ exponent n)

/-- The theorem stating that the nth term of the sequence is (n+1)aⁿ -/
theorem nth_term_formula (n : ℕ) (a : ℝ) : nthTerm n a = (n + 1 : ℝ) * a ^ n := by sorry

end NUMINAMATH_CALUDE_nth_term_formula_l1300_130054


namespace NUMINAMATH_CALUDE_remaining_amount_with_taxes_remaining_amount_is_622_54_l1300_130017

/-- Calculates the remaining amount to be paid including taxes for a product purchase --/
theorem remaining_amount_with_taxes (deposit_percent : ℝ) (cash_deposit : ℝ) (reward_points : ℕ) 
  (point_value : ℝ) (tax_rate : ℝ) (luxury_tax_rate : ℝ) : ℝ :=
  let total_deposit := cash_deposit + (reward_points : ℝ) * point_value
  let total_price := total_deposit / deposit_percent
  let remaining_before_taxes := total_price - total_deposit
  let tax := remaining_before_taxes * tax_rate
  let luxury_tax := remaining_before_taxes * luxury_tax_rate
  remaining_before_taxes + tax + luxury_tax

/-- The remaining amount to be paid including taxes is $622.54 --/
theorem remaining_amount_is_622_54 :
  remaining_amount_with_taxes 0.30 150 800 0.10 0.12 0.04 = 622.54 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_with_taxes_remaining_amount_is_622_54_l1300_130017


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1300_130087

theorem complex_equation_solution (z : ℂ) : 
  z * (1 + Complex.I) = Complex.abs (1 - Complex.I * Real.sqrt 3) →
  z = Real.sqrt 2 * (Complex.cos (Real.pi / 4) - Complex.I * Complex.sin (Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1300_130087


namespace NUMINAMATH_CALUDE_lily_siblings_count_l1300_130037

/-- The number of suitcases each sibling brings -/
def suitcases_per_sibling : ℕ := 2

/-- The number of suitcases parents bring -/
def suitcases_parents : ℕ := 6

/-- The total number of suitcases the family brings -/
def total_suitcases : ℕ := 14

/-- The number of Lily's siblings -/
def num_siblings : ℕ := (total_suitcases - suitcases_parents) / suitcases_per_sibling

theorem lily_siblings_count : num_siblings = 4 := by
  sorry

end NUMINAMATH_CALUDE_lily_siblings_count_l1300_130037


namespace NUMINAMATH_CALUDE_dog_treat_cost_l1300_130094

-- Define the given conditions
def treats_per_day : ℕ := 2
def cost_per_treat : ℚ := 1/10
def days_in_month : ℕ := 30

-- Define the theorem to prove
theorem dog_treat_cost :
  (treats_per_day * days_in_month : ℚ) * cost_per_treat = 6 := by sorry

end NUMINAMATH_CALUDE_dog_treat_cost_l1300_130094


namespace NUMINAMATH_CALUDE_sequence_constant_l1300_130034

theorem sequence_constant (a : ℕ → ℤ) (d : ℤ) :
  (∀ n : ℕ, Nat.Prime (Int.natAbs (a n))) →
  (∀ n : ℕ, a (n + 2) = a (n + 1) + a n + d) →
  (∀ n : ℕ, a n = 0) := by
sorry

end NUMINAMATH_CALUDE_sequence_constant_l1300_130034


namespace NUMINAMATH_CALUDE_tree_height_after_four_years_l1300_130081

/-- The height of a tree after n years, given its initial height and growth rate -/
def treeHeight (initialHeight : ℝ) (growthRate : ℝ) (n : ℕ) : ℝ :=
  initialHeight * growthRate^(n - 1)

/-- Theorem stating the height of the tree after 4 years -/
theorem tree_height_after_four_years
  (h1 : treeHeight 2 2 7 = 64)
  (h2 : treeHeight 2 2 1 = 2) :
  treeHeight 2 2 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_four_years_l1300_130081


namespace NUMINAMATH_CALUDE_inequality_implication_l1300_130011

theorem inequality_implication (a b : ℝ) : a < b → -3 * a > -3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1300_130011


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1300_130052

/-- In an arithmetic sequence {aₙ}, if a₄ + a₆ + a₈ + a₁₀ = 28, then a₇ = 7 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence property
  (a 4 + a 6 + a 8 + a 10 = 28) →                   -- given condition
  a 7 = 7 :=                                        -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1300_130052


namespace NUMINAMATH_CALUDE_bryan_has_more_candies_l1300_130068

/-- Given that Bryan has 50 skittles and Ben has 20 M&M's, 
    prove that Bryan has 30 more candies than Ben. -/
theorem bryan_has_more_candies : 
  ∀ (bryan_skittles ben_mms : ℕ), 
    bryan_skittles = 50 → 
    ben_mms = 20 → 
    bryan_skittles - ben_mms = 30 := by
  sorry

end NUMINAMATH_CALUDE_bryan_has_more_candies_l1300_130068


namespace NUMINAMATH_CALUDE_cosine_value_from_tangent_half_l1300_130040

theorem cosine_value_from_tangent_half (α : Real) :
  (1 - Real.cos α) / Real.sin α = 3 → Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_from_tangent_half_l1300_130040


namespace NUMINAMATH_CALUDE_no_rational_q_exists_l1300_130030

theorem no_rational_q_exists : ¬ ∃ (q : ℚ) (b c : ℚ),
  -- f(x) = x^2 + bx + c is a quadratic trinomial
  -- The coefficients 1, b, and c form a geometric progression with common ratio q
  ((1 = b ∧ b = c * q) ∨ (1 = c * q ∧ c * q = b) ∨ (b = 1 * q ∧ 1 * q = c)) ∧
  -- The difference between the roots of f(x) is q
  (b^2 - 4*c).sqrt = q := by
sorry

end NUMINAMATH_CALUDE_no_rational_q_exists_l1300_130030


namespace NUMINAMATH_CALUDE_jennifer_spending_l1300_130070

theorem jennifer_spending (total : ℚ) (sandwich_fraction : ℚ) (ticket_fraction : ℚ) (book_fraction : ℚ) :
  total = 120 →
  sandwich_fraction = 1 / 5 →
  ticket_fraction = 1 / 6 →
  book_fraction = 1 / 2 →
  total - (sandwich_fraction * total + ticket_fraction * total + book_fraction * total) = 16 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_spending_l1300_130070


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l1300_130019

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 3

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements in the arrangement -/
def total_elements : ℕ := num_ones + num_zeros

/-- The probability that two zeros are not adjacent when randomly arranged with three ones in a row -/
def prob_zeros_not_adjacent : ℚ := 3/5

theorem zeros_not_adjacent_probability :
  prob_zeros_not_adjacent = 3/5 :=
sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l1300_130019


namespace NUMINAMATH_CALUDE_stones_per_bracelet_l1300_130028

def total_stones : ℝ := 48.0
def num_bracelets : ℕ := 6

theorem stones_per_bracelet :
  total_stones / num_bracelets = 8 := by sorry

end NUMINAMATH_CALUDE_stones_per_bracelet_l1300_130028


namespace NUMINAMATH_CALUDE_rectangle_with_cut_corners_l1300_130013

/-- Given a rectangle ABCD with identical isosceles right triangles cut off from its corners,
    each having a leg of length a, and the total area cut off is 160 cm²,
    if the longer side of ABCD is 32√2 cm, then the length of PQ is 16√2 cm. -/
theorem rectangle_with_cut_corners (a : ℝ) (l : ℝ) (PQ : ℝ) :
  (4 * (1/2 * a^2) = 160) →  -- Total area cut off
  (l = 32 * Real.sqrt 2) →   -- Longer side of ABCD
  (PQ = l - 2*a) →           -- Definition of PQ
  (PQ = 16 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_with_cut_corners_l1300_130013


namespace NUMINAMATH_CALUDE_wire_oscillation_period_l1300_130024

/-- The period of oscillation for a mass on a wire with small displacements -/
theorem wire_oscillation_period
  (l d g : ℝ) -- wire length, distance between fixed points, gravitational acceleration
  (G : ℝ) -- mass
  (h_l_pos : l > 0)
  (h_d_pos : d > 0)
  (h_g_pos : g > 0)
  (h_G_pos : G > 0)
  (h_l_gt_d : l > d)
  (h_small_displacements : True) -- Assumption for small displacements
  : ∃ (T : ℝ), T = Real.pi * l * Real.sqrt (Real.sqrt 2 / (g * Real.sqrt (l^2 - d^2))) :=
sorry

end NUMINAMATH_CALUDE_wire_oscillation_period_l1300_130024


namespace NUMINAMATH_CALUDE_students_not_in_biology_l1300_130056

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) :
  total_students = 840 →
  biology_percentage = 35 / 100 →
  (total_students : ℚ) * (1 - biology_percentage) = 546 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l1300_130056


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_range_l1300_130067

theorem function_inequality_implies_parameter_range :
  ∀ (a : ℝ),
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (|x + a| + |x - 2| ≤ |x - 3|)) →
  (a ∈ Set.Icc (-1) 0) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_range_l1300_130067


namespace NUMINAMATH_CALUDE_cost_price_of_article_l1300_130012

/-- Proves that the cost price of an article is 800, given the conditions from the problem. -/
theorem cost_price_of_article : ∃ (C : ℝ), 
  (C = 800) ∧ 
  (1.05 * C - (0.95 * C + 0.1 * (0.95 * C)) = 4) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_article_l1300_130012


namespace NUMINAMATH_CALUDE_bucket_capacity_l1300_130018

/-- Represents the number of buckets needed to fill the bathtub to the top -/
def full_bathtub : ℕ := 14

/-- Represents the number of buckets removed to reach the bath level -/
def removed_buckets : ℕ := 3

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Represents the total amount of water used in a week (in ounces) -/
def weekly_water_usage : ℕ := 9240

/-- Calculates the number of buckets used for each bath -/
def buckets_per_bath : ℕ := full_bathtub - removed_buckets

/-- Calculates the number of buckets used in a week -/
def weekly_buckets : ℕ := buckets_per_bath * days_per_week

/-- Theorem: The bucket holds 120 ounces of water -/
theorem bucket_capacity : weekly_water_usage / weekly_buckets = 120 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_l1300_130018


namespace NUMINAMATH_CALUDE_remainder_sum_mod_seven_l1300_130075

theorem remainder_sum_mod_seven (a b c : ℕ) : 
  0 < a ∧ a < 7 ∧ 0 < b ∧ b < 7 ∧ 0 < c ∧ c < 7 →
  (a * b * c) % 7 = 1 →
  (4 * c) % 7 = 5 →
  (5 * b) % 7 = (4 + b) % 7 →
  (a + b + c) % 7 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_seven_l1300_130075


namespace NUMINAMATH_CALUDE_student_ranking_l1300_130058

theorem student_ranking (n : ℕ) 
  (rank_from_right : ℕ) 
  (rank_from_left : ℕ) 
  (h1 : rank_from_right = 17) 
  (h2 : rank_from_left = 5) : 
  n = rank_from_right + rank_from_left - 1 :=
by sorry

end NUMINAMATH_CALUDE_student_ranking_l1300_130058


namespace NUMINAMATH_CALUDE_counterexample_25_l1300_130029

theorem counterexample_25 : 
  ¬(¬(Nat.Prime 25) → Nat.Prime (25 + 3)) := by sorry

end NUMINAMATH_CALUDE_counterexample_25_l1300_130029


namespace NUMINAMATH_CALUDE_min_balls_to_draw_l1300_130057

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- Represents the maximum number of balls that can be drawn for each color -/
structure MaxDrawnBalls where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee the desired outcome -/
def minBallsToGuarantee : Nat := 57

/-- The threshold for a single color to be guaranteed -/
def singleColorThreshold : Nat := 12

/-- Theorem stating the minimum number of balls to be drawn -/
theorem min_balls_to_draw (initial : BallCounts) (max_drawn : MaxDrawnBalls) : 
  initial.red = 30 ∧ 
  initial.green = 25 ∧ 
  initial.yellow = 20 ∧ 
  initial.blue = 15 ∧ 
  initial.white = 10 ∧ 
  initial.black = 5 ∧
  max_drawn.red < singleColorThreshold ∧
  max_drawn.green < singleColorThreshold ∧ 
  max_drawn.yellow < singleColorThreshold ∧
  max_drawn.blue < singleColorThreshold ∧
  max_drawn.white < singleColorThreshold ∧
  max_drawn.black < singleColorThreshold ∧
  max_drawn.green % 2 = 0 ∧
  max_drawn.white % 2 = 0 ∧
  max_drawn.green ≤ initial.green ∧
  max_drawn.white ≤ initial.white →
  minBallsToGuarantee = 
    max_drawn.red + max_drawn.green + max_drawn.yellow + 
    max_drawn.blue + max_drawn.white + max_drawn.black + 1 :=
by sorry

end NUMINAMATH_CALUDE_min_balls_to_draw_l1300_130057


namespace NUMINAMATH_CALUDE_angle_multiplication_l1300_130046

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define multiplication of an angle by a natural number
def multiplyAngle (a : Angle) (n : ℕ) : Angle :=
  let totalMinutes := a.degrees * 60 + a.minutes
  let newTotalMinutes := totalMinutes * n
  ⟨newTotalMinutes / 60, newTotalMinutes % 60⟩

-- Define equality for angles
def angleEq (a b : Angle) : Prop :=
  a.degrees * 60 + a.minutes = b.degrees * 60 + b.minutes

-- Theorem statement
theorem angle_multiplication :
  angleEq (multiplyAngle ⟨21, 17⟩ 5) ⟨106, 25⟩ := by
  sorry

end NUMINAMATH_CALUDE_angle_multiplication_l1300_130046


namespace NUMINAMATH_CALUDE_original_number_proof_l1300_130026

theorem original_number_proof (x y : ℝ) : 
  10 * x + 22 * y = 780 → 
  y = 37.66666666666667 → 
  x + y = 32.7 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l1300_130026


namespace NUMINAMATH_CALUDE_brothers_initial_money_l1300_130077

theorem brothers_initial_money (michael_initial : ℕ) (brother_final : ℕ) (candy_cost : ℕ) :
  michael_initial = 42 →
  brother_final = 35 →
  candy_cost = 3 →
  ∃ (brother_initial : ℕ),
    brother_initial + michael_initial / 2 = brother_final + candy_cost ∧
    brother_initial = 17 :=
by sorry

end NUMINAMATH_CALUDE_brothers_initial_money_l1300_130077


namespace NUMINAMATH_CALUDE_ellipse_m_range_l1300_130078

/-- The equation of the curve in Cartesian coordinates -/
def curve_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

/-- The condition for the curve to be an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), curve_equation m x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)

/-- The theorem stating the range of m for which the curve is an ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m ↔ m > 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l1300_130078


namespace NUMINAMATH_CALUDE_katie_spent_sixty_dollars_l1300_130033

/-- The amount Katie spent on flowers -/
def katies_spending (flower_cost : ℕ) (roses : ℕ) (daisies : ℕ) : ℕ :=
  flower_cost * (roses + daisies)

/-- Theorem: Katie spent 60 dollars on flowers -/
theorem katie_spent_sixty_dollars : katies_spending 6 5 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_katie_spent_sixty_dollars_l1300_130033


namespace NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l1300_130091

/-- 
Proves that in an isosceles, obtuse triangle where one angle is 30% larger than a right angle, 
the measure of one of the two smallest angles is 31.5°.
-/
theorem isosceles_obtuse_triangle_smallest_angle : 
  ∀ (a b c : ℝ), 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Isosceles triangle condition
  c > 90 →           -- Obtuse triangle condition
  c = 1.3 * 90 →     -- One angle is 30% larger than a right angle
  a = 31.5 :=        -- The measure of one of the two smallest angles
by
  sorry

end NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l1300_130091


namespace NUMINAMATH_CALUDE_zyx_syndrome_diagnosis_l1300_130099

/-- Represents the characteristics and diagnostic information for ZYX syndrome --/
structure ZYXSyndromeData where
  total_patients : ℕ
  female_ratio : ℚ
  female_syndrome_ratio : ℚ
  male_syndrome_ratio : ℚ
  female_diagnostic_accuracy : ℚ
  male_diagnostic_accuracy : ℚ
  female_false_negative_rate : ℚ
  male_false_negative_rate : ℚ

/-- Calculates the number of patients diagnosed with ZYX syndrome --/
def diagnosed_patients (data : ZYXSyndromeData) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, 14 patients will be diagnosed with ZYX syndrome --/
theorem zyx_syndrome_diagnosis :
  let data : ZYXSyndromeData := {
    total_patients := 52,
    female_ratio := 3/5,
    female_syndrome_ratio := 1/5,
    male_syndrome_ratio := 3/10,
    female_diagnostic_accuracy := 7/10,
    male_diagnostic_accuracy := 4/5,
    female_false_negative_rate := 1/10,
    male_false_negative_rate := 3/20
  }
  diagnosed_patients data = 14 := by
  sorry


end NUMINAMATH_CALUDE_zyx_syndrome_diagnosis_l1300_130099


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l1300_130074

/-- Represents the possible stripe configurations on a cube face -/
inductive StripeConfig
| DiagonalA
| DiagonalB
| EdgeToEdgeA
| EdgeToEdgeB

/-- Represents a cube with stripes on each face -/
def StripedCube := Fin 6 → StripeConfig

/-- Checks if a given StripedCube has a continuous stripe encircling it -/
def hasContinuousStripe (cube : StripedCube) : Prop := sorry

/-- The total number of possible stripe configurations for a cube -/
def totalConfigurations : ℕ := 4^6

/-- The number of configurations that result in a continuous stripe -/
def continuousStripeConfigurations : ℕ := 48

theorem continuous_stripe_probability :
  (continuousStripeConfigurations : ℚ) / totalConfigurations = 3 / 256 := by sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l1300_130074


namespace NUMINAMATH_CALUDE_num_tables_made_l1300_130098

-- Define the total number of furniture legs
def total_legs : Nat := 40

-- Define the number of chairs
def num_chairs : Nat := 6

-- Define the number of legs per furniture piece
def legs_per_piece : Nat := 4

-- Theorem to prove
theorem num_tables_made : 
  (total_legs - num_chairs * legs_per_piece) / legs_per_piece = 4 := by
  sorry


end NUMINAMATH_CALUDE_num_tables_made_l1300_130098


namespace NUMINAMATH_CALUDE_fraction_to_repeating_decimal_value_of_expression_l1300_130042

def repeating_decimal (n d : ℕ) (a b c d : ℕ) : Prop :=
  (n : ℚ) / d = (a * 1000 + b * 100 + c * 10 + d : ℚ) / 9999

theorem fraction_to_repeating_decimal :
  repeating_decimal 7 26 2 6 9 2 :=
sorry

theorem value_of_expression (a b c d : ℕ) :
  repeating_decimal 7 26 a b c d → 3 * a - b = 0 :=
sorry

end NUMINAMATH_CALUDE_fraction_to_repeating_decimal_value_of_expression_l1300_130042


namespace NUMINAMATH_CALUDE_hands_closest_and_farthest_l1300_130005

/-- Represents a time between 6:30 and 6:35 -/
inductive ClockTime
  | t630
  | t631
  | t632
  | t633
  | t634
  | t635

/-- Calculates the angle between hour and minute hands for a given time -/
def angleBetweenHands (t : ClockTime) : ℝ :=
  match t with
  | ClockTime.t630 => 15
  | ClockTime.t631 => 9.5
  | ClockTime.t632 => 4
  | ClockTime.t633 => 1.5
  | ClockTime.t634 => 7
  | ClockTime.t635 => 12.5

theorem hands_closest_and_farthest :
  (∀ t : ClockTime, angleBetweenHands ClockTime.t633 ≤ angleBetweenHands t) ∧
  (∀ t : ClockTime, angleBetweenHands t ≤ angleBetweenHands ClockTime.t630) :=
by sorry


end NUMINAMATH_CALUDE_hands_closest_and_farthest_l1300_130005


namespace NUMINAMATH_CALUDE_product_of_sum_of_roots_l1300_130020

theorem product_of_sum_of_roots (x : ℝ) :
  (Real.sqrt (5 + x) + Real.sqrt (25 - x) = 8) →
  (5 + x) * (25 - x) = 289 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_roots_l1300_130020


namespace NUMINAMATH_CALUDE_abs_inequality_l1300_130014

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l1300_130014


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l1300_130079

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 150th term of the specific arithmetic sequence is 1046 -/
theorem arithmetic_sequence_150th_term :
  arithmetic_sequence 3 7 150 = 1046 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l1300_130079


namespace NUMINAMATH_CALUDE_stratified_sampling_grade12_l1300_130085

theorem stratified_sampling_grade12 (total_students : ℕ) (grade12_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 3600) 
  (h2 : grade12_students = 1500) 
  (h3 : sample_size = 720) :
  (grade12_students : ℚ) / (total_students : ℚ) * (sample_size : ℚ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_grade12_l1300_130085


namespace NUMINAMATH_CALUDE_fitness_center_member_ratio_l1300_130044

theorem fitness_center_member_ratio 
  (f : ℕ) (m : ℕ) -- f: number of female members, m: number of male members
  (h1 : (35 * f + 30 * m) / (f + m) = 32) : -- average age of all members is 32
  f / m = 2 / 3 := by
sorry


end NUMINAMATH_CALUDE_fitness_center_member_ratio_l1300_130044


namespace NUMINAMATH_CALUDE_coin_problem_l1300_130059

theorem coin_problem (p n : ℕ) : 
  p + n = 32 →  -- Total number of coins
  p + 5 * n = 100 →  -- Total value in cents
  n = 17 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l1300_130059


namespace NUMINAMATH_CALUDE_no_distinct_naturals_satisfying_equation_l1300_130035

theorem no_distinct_naturals_satisfying_equation :
  ¬ ∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (2 * a + Nat.lcm b c = 2 * b + Nat.lcm a c) ∧
    (2 * b + Nat.lcm a c = 2 * c + Nat.lcm a b) :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_naturals_satisfying_equation_l1300_130035


namespace NUMINAMATH_CALUDE_arithmetic_mean_two_digit_multiples_of_eight_l1300_130039

theorem arithmetic_mean_two_digit_multiples_of_eight : 
  let first_multiple := 16
  let last_multiple := 96
  let number_of_multiples := (last_multiple - first_multiple) / 8 + 1
  (first_multiple + last_multiple) / 2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_two_digit_multiples_of_eight_l1300_130039


namespace NUMINAMATH_CALUDE_quilt_shaded_area_is_40_percent_l1300_130038

/-- Represents a square quilt with shaded areas -/
structure Quilt where
  size : Nat
  fully_shaded : Nat
  half_shaded_single : Nat
  half_shaded_double : Nat

/-- Calculates the percentage of shaded area in the quilt -/
def shaded_percentage (q : Quilt) : Rat :=
  let total_squares := q.size * q.size
  let shaded_area := q.fully_shaded + (q.half_shaded_single / 2) + (q.half_shaded_double / 2)
  (shaded_area / total_squares) * 100

/-- Theorem stating that the specific quilt configuration has 40% shaded area -/
theorem quilt_shaded_area_is_40_percent :
  let q : Quilt := {
    size := 5,
    fully_shaded := 4,
    half_shaded_single := 8,
    half_shaded_double := 4
  }
  shaded_percentage q = 40 := by sorry

end NUMINAMATH_CALUDE_quilt_shaded_area_is_40_percent_l1300_130038


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l1300_130000

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 2 * y^2 - 2 * x * y + x + 9 * y - 2 = 0 ↔
    ((x = 9 ∧ y = 1) ∨ (x = 2 ∧ y = 0) ∨ (x = 8 ∧ y = 2) ∨ (x = 3 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l1300_130000


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l1300_130023

theorem number_with_specific_remainders (x : ℤ) :
  x % 7 = 3 →
  x^2 % 49 = 44 →
  x^3 % 343 = 111 →
  x % 343 = 17 := by
sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l1300_130023


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1300_130031

theorem cubic_root_sum (a b c : ℝ) : 
  (a^3 - 2*a^2 - a + 2 = 0) →
  (b^3 - 2*b^2 - b + 2 = 0) →
  (c^3 - 2*c^2 - c + 2 = 0) →
  a*(b-c)^2 + b*(c-a)^2 + c*(a-b)^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1300_130031


namespace NUMINAMATH_CALUDE_largest_common_term_l1300_130032

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def is_common_term (x : ℤ) (a₁ d₁ a₂ d₂ : ℤ) : Prop :=
  ∃ n m : ℕ, arithmetic_sequence a₁ d₁ n = x ∧ arithmetic_sequence a₂ d₂ m = x

theorem largest_common_term :
  ∃ x : ℤ, x ≤ 150 ∧ is_common_term x 1 8 5 9 ∧
    ∀ y : ℤ, y ≤ 150 → is_common_term y 1 8 5 9 → y ≤ x :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l1300_130032


namespace NUMINAMATH_CALUDE_q_value_l1300_130006

-- Define the polynomial Q(x)
def Q (p q d : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + d

-- Define the property that the mean of zeros, twice the product of zeros, and sum of coefficients are equal
def property (p q d : ℝ) : Prop :=
  let sum_of_zeros := -p
  let product_of_zeros := -d
  let sum_of_coefficients := 1 + p + q + d
  (sum_of_zeros / 3 = 2 * product_of_zeros) ∧ (sum_of_zeros / 3 = sum_of_coefficients)

-- Theorem statement
theorem q_value (p q d : ℝ) :
  property p q d → Q p q d 0 = 4 → q = -37 := by
  sorry

end NUMINAMATH_CALUDE_q_value_l1300_130006


namespace NUMINAMATH_CALUDE_f_of_one_eq_zero_l1300_130090

theorem f_of_one_eq_zero (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 2*x) : f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_eq_zero_l1300_130090


namespace NUMINAMATH_CALUDE_geometry_propositions_l1300_130088

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_propositions 
  (m n : Line) (α β : Plane) :
  (∀ m α β, parallel_line_plane m α → perpendicular_line_plane m β → perpendicular_plane α β) ∧
  (∀ m n α, parallel_line m n → perpendicular_line_plane m α → perpendicular_line_plane n α) :=
sorry

end NUMINAMATH_CALUDE_geometry_propositions_l1300_130088


namespace NUMINAMATH_CALUDE_delores_initial_money_l1300_130050

/-- The initial amount of money Delores had --/
def initial_amount : ℕ := sorry

/-- The cost of the computer --/
def computer_cost : ℕ := 400

/-- The cost of the printer --/
def printer_cost : ℕ := 40

/-- The amount of money left after purchases --/
def remaining_money : ℕ := 10

/-- Theorem stating that Delores' initial amount of money was $450 --/
theorem delores_initial_money : 
  initial_amount = computer_cost + printer_cost + remaining_money := by sorry

end NUMINAMATH_CALUDE_delores_initial_money_l1300_130050


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1300_130027

theorem quadratic_inequality_equivalence (x : ℝ) :
  3 * x^2 + 5 * x < 8 ↔ -4 < x ∧ x < 2/3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1300_130027


namespace NUMINAMATH_CALUDE_smallest_b_value_l1300_130096

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 7) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val^2 * b.val) = 12) :
  ∀ k : ℕ+, k.val < b.val → ¬(∃ a' : ℕ+, 
    a'.val - k.val = 7 ∧ 
    Nat.gcd ((a'.val^3 + k.val^3) / (a'.val + k.val)) (a'.val^2 * k.val) = 12) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1300_130096


namespace NUMINAMATH_CALUDE_jeds_speed_l1300_130086

def speed_limit : ℕ := 50
def speeding_fine_rate : ℕ := 16
def red_light_fine : ℕ := 75
def cellphone_fine : ℕ := 120
def total_fine : ℕ := 826

def non_speeding_fines : ℕ := 2 * red_light_fine + cellphone_fine

theorem jeds_speed :
  ∃ (speed : ℕ),
    speed = speed_limit + (total_fine - non_speeding_fines) / speeding_fine_rate ∧
    speed = 84 := by
  sorry

end NUMINAMATH_CALUDE_jeds_speed_l1300_130086


namespace NUMINAMATH_CALUDE_circle_area_tripled_radius_l1300_130009

theorem circle_area_tripled_radius (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := π * (3*r)^2
  A' = 9 * A :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tripled_radius_l1300_130009


namespace NUMINAMATH_CALUDE_hyperbola_center_l1300_130069

/-- The center of a hyperbola is the point (h, k) such that the equation of the hyperbola
    can be written in the form ((y-k)/a)² - ((x-h)/b)² = 1 for some non-zero real numbers a and b. -/
def is_center_of_hyperbola (h k : ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  ∀ (x y : ℝ), ((3 * y + 3)^2 / 7^2) - ((4 * x - 5)^2 / 3^2) = 1 ↔
                ((y - k) / a)^2 - ((x - h) / b)^2 = 1

/-- The center of the hyperbola (3y+3)²/7² - (4x-5)²/3² = 1 is (5/4, -1). -/
theorem hyperbola_center :
  is_center_of_hyperbola (5/4) (-1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_l1300_130069


namespace NUMINAMATH_CALUDE_second_term_of_arithmetic_sequence_l1300_130016

/-- Given an arithmetic sequence where the sum of the first and third terms is 8,
    prove that the second term is 4. -/
theorem second_term_of_arithmetic_sequence (a d : ℝ) 
  (h : a + (a + 2 * d) = 8) : a + d = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_arithmetic_sequence_l1300_130016


namespace NUMINAMATH_CALUDE_ordering_of_expressions_l1300_130036

theorem ordering_of_expressions : 
  Real.exp 0.1 > Real.sqrt 1.2 ∧ Real.sqrt 1.2 > 1 + Real.log 1.1 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_expressions_l1300_130036


namespace NUMINAMATH_CALUDE_remainder_problem_l1300_130022

theorem remainder_problem (n : ℤ) (h : n % 20 = 11) : (2 * n) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1300_130022


namespace NUMINAMATH_CALUDE_complex_trig_identity_l1300_130062

theorem complex_trig_identity (θ : Real) (h : π < θ ∧ θ < (3 * π) / 2) :
  Real.sqrt ((1 / 2) + (1 / 2) * Real.sqrt ((1 / 2) + (1 / 2) * Real.cos (2 * θ))) - 
  Real.sqrt (1 - Real.sin θ) = Real.cos (θ / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_trig_identity_l1300_130062


namespace NUMINAMATH_CALUDE_triangle_area_proof_l1300_130041

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) - Real.sin (ω * x) ^ 2 + 1

theorem triangle_area_proof (ω : ℝ) (A B C : ℝ) (b : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω (x + π / (2 * ω)) = f ω x) →
  b = 2 →
  f ω A = 1 →
  2 * Real.sin A = Real.sqrt 3 * Real.sin C →
  ∃ (a c : ℝ), a * b * Real.sin C / 2 = 2 * Real.sqrt 3 := by
  sorry

#check triangle_area_proof

end NUMINAMATH_CALUDE_triangle_area_proof_l1300_130041


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1300_130007

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 2, 4}
def B : Set Nat := {1, 4}

theorem intersection_complement_equality : A ∩ (U \ B) = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1300_130007


namespace NUMINAMATH_CALUDE_unknown_rate_is_225_l1300_130010

/-- The unknown rate of two blankets given the following conditions:
    - 3 blankets at Rs. 100 each
    - 6 blankets at Rs. 150 each
    - 2 blankets at an unknown rate
    - The average price of all blankets is Rs. 150
-/
def unknown_rate : ℕ := by
  -- Define the known quantities
  let blankets_100 : ℕ := 3
  let price_100 : ℕ := 100
  let blankets_150 : ℕ := 6
  let price_150 : ℕ := 150
  let blankets_unknown : ℕ := 2
  let average_price : ℕ := 150

  -- Calculate the total number of blankets
  let total_blankets : ℕ := blankets_100 + blankets_150 + blankets_unknown

  -- Calculate the total cost of all blankets
  let total_cost : ℕ := average_price * total_blankets

  -- Calculate the cost of known blankets
  let cost_known : ℕ := blankets_100 * price_100 + blankets_150 * price_150

  -- Calculate the cost of unknown blankets
  let cost_unknown : ℕ := total_cost - cost_known

  -- Calculate the rate of each unknown blanket
  exact cost_unknown / blankets_unknown

theorem unknown_rate_is_225 : unknown_rate = 225 := by
  sorry

end NUMINAMATH_CALUDE_unknown_rate_is_225_l1300_130010


namespace NUMINAMATH_CALUDE_sqrt_fifth_power_cubed_l1300_130066

theorem sqrt_fifth_power_cubed : (((5 : ℝ) ^ (1/2)) ^ 4) ^ (1/2) ^ 3 = 125 := by sorry

end NUMINAMATH_CALUDE_sqrt_fifth_power_cubed_l1300_130066


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l1300_130093

theorem fraction_sum_simplification (x : ℝ) (h : x + 1 ≠ 0) :
  x / ((x + 1)^2) + 1 / ((x + 1)^2) = 1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l1300_130093


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l1300_130053

theorem arithmetic_mean_difference (a b c : ℝ) :
  (a + b) / 2 = (a + b + c) / 3 + 5 →
  (a + c) / 2 = (a + b + c) / 3 - 8 →
  (b + c) / 2 = (a + b + c) / 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l1300_130053


namespace NUMINAMATH_CALUDE_square_tiles_count_l1300_130001

theorem square_tiles_count (total_tiles : ℕ) (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 30)
  (h_total_edges : total_edges = 100) :
  ∃ (triangles squares pentagons : ℕ),
    triangles + squares + pentagons = total_tiles ∧
    3 * triangles + 4 * squares + 5 * pentagons = total_edges ∧
    squares = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_tiles_count_l1300_130001


namespace NUMINAMATH_CALUDE_cos_675_degrees_l1300_130071

theorem cos_675_degrees : Real.cos (675 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_675_degrees_l1300_130071


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_equality_condition_l1300_130047

theorem min_value_reciprocal_sum (x y z : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_one : x + y + z = 1) :
  1/x + 4/y + 9/z ≥ 36 := by
  sorry

theorem equality_condition (x y z : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_one : x + y + z = 1) :
  1/x + 4/y + 9/z = 36 ↔ x = 1/6 ∧ y = 1/3 ∧ z = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_equality_condition_l1300_130047


namespace NUMINAMATH_CALUDE_range_of_m_l1300_130048

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def B : Set ℝ := {x | -1 < x ∧ x < 5}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem range_of_m (m : ℝ) :
  (Set.compl (A m) ∩ B).Nonempty → m < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1300_130048


namespace NUMINAMATH_CALUDE_billy_horse_feeding_days_billy_horse_feeding_problem_l1300_130004

theorem billy_horse_feeding_days 
  (num_horses : ℕ) 
  (oats_per_feeding : ℕ) 
  (feedings_per_day : ℕ) 
  (total_oats : ℕ) : ℕ :=
  let daily_oats_per_horse := oats_per_feeding * feedings_per_day
  let total_daily_oats := daily_oats_per_horse * num_horses
  total_oats / total_daily_oats

theorem billy_horse_feeding_problem :
  billy_horse_feeding_days 4 4 2 96 = 3 := by
  sorry

end NUMINAMATH_CALUDE_billy_horse_feeding_days_billy_horse_feeding_problem_l1300_130004


namespace NUMINAMATH_CALUDE_initial_num_pipes_is_three_l1300_130065

-- Define the fill time for the initial number of pipes
def initial_fill_time : ℝ := 8

-- Define the fill time for two pipes
def two_pipes_fill_time : ℝ := 12

-- Define the number of pipes we want to prove
def target_num_pipes : ℕ := 3

-- Theorem statement
theorem initial_num_pipes_is_three :
  ∃ (n : ℕ), n > 0 ∧
  (1 : ℝ) / initial_fill_time = (n : ℝ) * ((1 : ℝ) / two_pipes_fill_time / 2) ∧
  n = target_num_pipes :=
sorry

end NUMINAMATH_CALUDE_initial_num_pipes_is_three_l1300_130065


namespace NUMINAMATH_CALUDE_perpendicular_condition_l1300_130055

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line y = ax + 1 -/
def slope1 (a : ℝ) : ℝ := a

/-- The slope of the second line y = (a-2)x - 1 -/
def slope2 (a : ℝ) : ℝ := a - 2

/-- Theorem: a = 1 is the necessary and sufficient condition for the lines to be perpendicular -/
theorem perpendicular_condition : 
  ∀ a : ℝ, perpendicular (slope1 a) (slope2 a) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l1300_130055


namespace NUMINAMATH_CALUDE_ken_kept_pencils_l1300_130021

def pencil_distribution (total : ℕ) (manny nilo carlos tina rina : ℕ) : Prop :=
  total = 200 ∧
  manny = 20 ∧
  nilo = manny + 10 ∧
  carlos = nilo + 5 ∧
  tina = carlos + 15 ∧
  rina = tina + 5

theorem ken_kept_pencils (total manny nilo carlos tina rina : ℕ) :
  pencil_distribution total manny nilo carlos tina rina →
  total - (manny + nilo + carlos + tina + rina) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ken_kept_pencils_l1300_130021
