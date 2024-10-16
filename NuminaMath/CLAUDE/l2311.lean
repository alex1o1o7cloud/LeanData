import Mathlib

namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_l2311_231103

/-- A function f is monotonic on an interval [a, b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The theorem states that for the function f(x) = x^2 - kx + 1,
    f is monotonic on the interval [1, 2] if and only if
    k is in the set (-∞, 2] ∪ [4, +∞). -/
theorem monotonic_quadratic_function (k : ℝ) :
  IsMonotonic (fun x => x^2 - k*x + 1) 1 2 ↔ k ≤ 2 ∨ k ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_l2311_231103


namespace NUMINAMATH_CALUDE_optimal_rent_and_income_l2311_231137

def daily_net_income (rent : ℕ) : ℤ :=
  if rent ≤ 6 then
    50 * rent - 115
  else
    (50 - 3 * (rent - 6)) * rent - 115

def is_valid_rent (rent : ℕ) : Prop :=
  3 ≤ rent ∧ rent ≤ 20 ∧ daily_net_income rent > 0

theorem optimal_rent_and_income :
  ∃ (optimal_rent : ℕ) (max_income : ℤ),
    is_valid_rent optimal_rent ∧
    max_income = daily_net_income optimal_rent ∧
    optimal_rent = 11 ∧
    max_income = 270 ∧
    ∀ (rent : ℕ), is_valid_rent rent → daily_net_income rent ≤ max_income :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_rent_and_income_l2311_231137


namespace NUMINAMATH_CALUDE_inequality_implies_a_zero_l2311_231121

theorem inequality_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, a * (Real.sin x)^2 + Real.cos x ≥ a^2 - 1) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_zero_l2311_231121


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2311_231177

/-- Represents a repeating decimal with a given numerator and denominator. -/
structure RepeatingDecimal where
  numerator : ℕ
  denominator : ℕ
  denom_nonzero : denominator ≠ 0

/-- Converts a repeating decimal to a rational number. -/
def RepeatingDecimal.toRational (r : RepeatingDecimal) : ℚ :=
  ↑r.numerator / ↑r.denominator

theorem repeating_decimal_subtraction :
  let a : RepeatingDecimal := ⟨845, 999, by norm_num⟩
  let b : RepeatingDecimal := ⟨267, 999, by norm_num⟩
  let c : RepeatingDecimal := ⟨159, 999, by norm_num⟩
  a.toRational - b.toRational - c.toRational = 419 / 999 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2311_231177


namespace NUMINAMATH_CALUDE_sams_adventure_books_l2311_231113

/-- The number of adventure books Sam bought at the school's book fair -/
def adventure_books : ℕ := sorry

/-- The number of mystery books Sam bought -/
def mystery_books : ℕ := 17

/-- The total number of books Sam bought -/
def total_books : ℕ := 30

theorem sams_adventure_books :
  adventure_books = total_books - mystery_books ∧ adventure_books = 13 := by sorry

end NUMINAMATH_CALUDE_sams_adventure_books_l2311_231113


namespace NUMINAMATH_CALUDE_quadratic_tangent_line_l2311_231155

/-- Given a quadratic function f(x) = x^2 + ax + b, prove that if its tangent line
    at (0, b) has the equation x - y + 1 = 0, then a = 1 and b = 1. -/
theorem quadratic_tangent_line (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  let f' : ℝ → ℝ := λ x ↦ 2*x + a
  (∀ x y, y = f x → x - y + 1 = 0 → x = 0) →
  f' 0 = 1 →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_tangent_line_l2311_231155


namespace NUMINAMATH_CALUDE_total_bears_l2311_231157

/-- The number of bears in a national park --/
def bear_population (black white brown : ℕ) : ℕ :=
  black + white + brown

/-- Theorem: Given the conditions, the total bear population is 190 --/
theorem total_bears : ∀ (black white brown : ℕ),
  black = 60 →
  black = 2 * white →
  brown = black + 40 →
  bear_population black white brown = 190 := by
  sorry

end NUMINAMATH_CALUDE_total_bears_l2311_231157


namespace NUMINAMATH_CALUDE_remainder_sum_l2311_231147

theorem remainder_sum (n : ℤ) : n % 15 = 7 → (n % 3 + n % 5 = 3) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2311_231147


namespace NUMINAMATH_CALUDE_sum_of_A_and_C_is_seven_l2311_231173

theorem sum_of_A_and_C_is_seven (A B C D : ℕ) : 
  A ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  B ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  C ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  D ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (A : ℚ) / B + (C : ℚ) / D = 3 →
  A + C = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_A_and_C_is_seven_l2311_231173


namespace NUMINAMATH_CALUDE_treasure_value_proof_l2311_231159

def base7ToBase10 (n : Nat) : Nat :=
  let digits := n.digits 7
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (7 ^ i)) 0

theorem treasure_value_proof :
  let diamonds := 5643
  let silver := 1652
  let spices := 236
  (base7ToBase10 diamonds) + (base7ToBase10 silver) + (base7ToBase10 spices) = 2839 := by
  sorry

end NUMINAMATH_CALUDE_treasure_value_proof_l2311_231159


namespace NUMINAMATH_CALUDE_acidic_concentration_after_water_removal_l2311_231118

/-- Calculates the final concentration of an acidic solution after removing water -/
theorem acidic_concentration_after_water_removal
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (water_removed : ℝ)
  (h1 : initial_volume = 27)
  (h2 : initial_concentration = 0.4)
  (h3 : water_removed = 9)
  : (initial_volume * initial_concentration) / (initial_volume - water_removed) = 0.6 := by
  sorry

#check acidic_concentration_after_water_removal

end NUMINAMATH_CALUDE_acidic_concentration_after_water_removal_l2311_231118


namespace NUMINAMATH_CALUDE_optimal_loquat_variety_l2311_231114

/-- Represents a variety of loquat trees -/
structure LoquatVariety where
  name : String
  average_yield : ℝ
  variance : ℝ

/-- Determines if one variety is better than another based on yield and stability -/
def is_better (v1 v2 : LoquatVariety) : Prop :=
  (v1.average_yield > v2.average_yield) ∨ 
  (v1.average_yield = v2.average_yield ∧ v1.variance < v2.variance)

/-- Determines if a variety is the best among a list of varieties -/
def is_best (v : LoquatVariety) (vs : List LoquatVariety) : Prop :=
  ∀ v' ∈ vs, v ≠ v' → is_better v v'

theorem optimal_loquat_variety (A B C : LoquatVariety)
  (hA : A = { name := "A", average_yield := 42, variance := 1.8 })
  (hB : B = { name := "B", average_yield := 45, variance := 23 })
  (hC : C = { name := "C", average_yield := 45, variance := 1.8 }) :
  is_best C [A, B, C] := by
  sorry

#check optimal_loquat_variety

end NUMINAMATH_CALUDE_optimal_loquat_variety_l2311_231114


namespace NUMINAMATH_CALUDE_eulers_formula_l2311_231136

/-- A connected planar graph -/
structure ConnectedPlanarGraph where
  s : ℕ  -- number of vertices
  f : ℕ  -- number of faces
  a : ℕ  -- number of edges

/-- Euler's formula for connected planar graphs -/
theorem eulers_formula (G : ConnectedPlanarGraph) : G.f = G.a - G.s + 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l2311_231136


namespace NUMINAMATH_CALUDE_f_properties_l2311_231170

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1/2) * (x + a)^2

theorem f_properties (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ 0) →
  (HasDerivAt (f a) 1 0) →
  (∀ x : ℝ, StrictMono (f a)) ∧ 
  (a ≥ -Real.sqrt 2 ∧ a ≤ 2 - Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2311_231170


namespace NUMINAMATH_CALUDE_prob_king_jack_queen_value_l2311_231199

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Jacks in a standard deck -/
def NumJacks : ℕ := 4

/-- Number of Queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Probability of drawing a King, then a Jack, then a Queen from a standard deck without replacement -/
def prob_king_jack_queen : ℚ :=
  (NumKings : ℚ) / StandardDeck *
  (NumJacks : ℚ) / (StandardDeck - 1) *
  (NumQueens : ℚ) / (StandardDeck - 2)

theorem prob_king_jack_queen_value :
  prob_king_jack_queen = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_jack_queen_value_l2311_231199


namespace NUMINAMATH_CALUDE_am_gm_inequality_two_terms_l2311_231105

theorem am_gm_inequality_two_terms (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_am_gm_inequality_two_terms_l2311_231105


namespace NUMINAMATH_CALUDE_paving_cost_l2311_231126

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 400) :
  length * width * rate = 8250 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l2311_231126


namespace NUMINAMATH_CALUDE_angle_range_in_scalene_triangle_l2311_231192

-- Define a scalene triangle
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the theorem
theorem angle_range_in_scalene_triangle (t : ScaleneTriangle) 
  (h_longest : t.a ≥ t.b ∧ t.a ≥ t.c) 
  (h_inequality : t.a^2 < t.b^2 + t.c^2) :
  let A := Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))
  60 * π / 180 < A ∧ A < 90 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_range_in_scalene_triangle_l2311_231192


namespace NUMINAMATH_CALUDE_weight_after_deliveries_l2311_231175

/-- Calculates the remaining weight on a truck after two deliveries with given percentages -/
def remaining_weight (initial_load : ℝ) (first_unload_percent : ℝ) (second_unload_percent : ℝ) : ℝ :=
  let remaining_after_first := initial_load * (1 - first_unload_percent)
  remaining_after_first * (1 - second_unload_percent)

/-- Theorem stating the remaining weight after two deliveries -/
theorem weight_after_deliveries :
  remaining_weight 50000 0.1 0.2 = 36000 := by
  sorry

end NUMINAMATH_CALUDE_weight_after_deliveries_l2311_231175


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2311_231162

/-- A polynomial P(x) with a real parameter r satisfies:
    1) P(x) has remainder 2 when divided by (x-r)
    2) P(x) has remainder (-2x^2 - 3x + 4) when divided by (2x^2 + 7x - 4)(x-r)
    This theorem states that r can only be 1/2 or -2. -/
theorem polynomial_remainder_theorem (P : ℝ → ℝ) (r : ℝ) :
  (∃ Q₁ : ℝ → ℝ, ∀ x, P x = (x - r) * Q₁ x + 2) ∧
  (∃ Q₂ : ℝ → ℝ, ∀ x, P x = (2*x^2 + 7*x - 4)*(x - r) * Q₂ x + (-2*x^2 - 3*x + 4)) →
  r = 1/2 ∨ r = -2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2311_231162


namespace NUMINAMATH_CALUDE_test_score_problem_l2311_231146

theorem test_score_problem (total_questions : ℕ) (score : ℤ) 
  (correct_answers : ℕ) (incorrect_answers : ℕ) : 
  total_questions = 100 →
  score = correct_answers - 2 * incorrect_answers →
  correct_answers + incorrect_answers = total_questions →
  score = 73 →
  correct_answers = 91 := by
sorry

end NUMINAMATH_CALUDE_test_score_problem_l2311_231146


namespace NUMINAMATH_CALUDE_reinforcement_size_is_300_l2311_231172

/-- Calculates the reinforcement size given the initial garrison size, 
    initial provision duration, days passed, and remaining provision duration -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                             (days_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_duration
  let provisions_left := initial_garrison * (initial_duration - days_passed)
  (provisions_left / remaining_duration) - initial_garrison

/-- Theorem stating that the reinforcement size is 300 given the problem conditions -/
theorem reinforcement_size_is_300 : 
  calculate_reinforcement 150 31 16 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_is_300_l2311_231172


namespace NUMINAMATH_CALUDE_polynomial_value_l2311_231108

/-- Given a polynomial G satisfying certain conditions, prove G(8) = 491/3 -/
theorem polynomial_value (G : ℝ → ℝ) : 
  (∀ x, G (4 * x) / G (x + 2) = 4 - (20 * x + 24) / (x^2 + 4 * x + 4)) →
  G 4 = 35 →
  G 8 = 491 / 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l2311_231108


namespace NUMINAMATH_CALUDE_polynomial_root_implies_h_value_l2311_231128

theorem polynomial_root_implies_h_value : 
  ∀ h : ℚ, (3 : ℚ)^3 + h * 3 + 14 = 0 → h = -41/3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_h_value_l2311_231128


namespace NUMINAMATH_CALUDE_sqrt_193_between_13_and_14_l2311_231169

theorem sqrt_193_between_13_and_14 : 13 < Real.sqrt 193 ∧ Real.sqrt 193 < 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_193_between_13_and_14_l2311_231169


namespace NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l2311_231116

/-- Given a geometric sequence with first term 3 and common ratio 5/2, 
    the tenth term is equal to 5859375/512. -/
theorem tenth_term_of_geometric_sequence : 
  let a : ℚ := 3
  let r : ℚ := 5/2
  let n : ℕ := 10
  let a_n := a * r^(n - 1)
  a_n = 5859375/512 := by sorry

end NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l2311_231116


namespace NUMINAMATH_CALUDE_joe_pays_four_more_than_jenny_l2311_231190

/-- Represents the pizza sharing scenario between Jenny and Joe -/
structure PizzaScenario where
  totalSlices : ℕ
  plainPizzaCost : ℚ
  mushroomExtraCost : ℚ
  mushroomSlices : ℕ
  joeMushroomSlices : ℕ
  joePlainSlices : ℕ

/-- Calculates the cost difference between Joe's and Jenny's payments -/
def paymentDifference (scenario : PizzaScenario) : ℚ :=
  let plainSliceCost := scenario.plainPizzaCost / scenario.totalSlices
  let mushroomSliceCost := plainSliceCost + scenario.mushroomExtraCost
  let jennysSlices := scenario.totalSlices - scenario.joeMushroomSlices - scenario.joePlainSlices
  let joePayment := scenario.joeMushroomSlices * mushroomSliceCost + scenario.joePlainSlices * plainSliceCost
  let jennyPayment := jennysSlices * plainSliceCost
  joePayment - jennyPayment

/-- Theorem stating that in the given scenario, Joe pays $4 more than Jenny -/
theorem joe_pays_four_more_than_jenny : 
  let scenario := PizzaScenario.mk 12 12 (1/2) 4 4 3
  paymentDifference scenario = 4 := by sorry

end NUMINAMATH_CALUDE_joe_pays_four_more_than_jenny_l2311_231190


namespace NUMINAMATH_CALUDE_monomial_coefficient_and_degree_l2311_231149

/-- The monomial type -/
structure Monomial (α : Type*) [Ring α] where
  coeff : α
  vars : List (α × Nat)

/-- Definition of the coefficient of a monomial -/
def coefficient (m : Monomial ℚ) : ℚ := m.coeff

/-- Definition of the degree of a monomial -/
def degree (m : Monomial ℚ) : Nat := m.vars.foldr (λ (_, exp) acc => acc + exp) 0

/-- The given monomial -/
def m : Monomial ℚ := ⟨-3/5, [(1, 1), (1, 2)]⟩

theorem monomial_coefficient_and_degree :
  coefficient m = -3/5 ∧ degree m = 3 := by sorry

end NUMINAMATH_CALUDE_monomial_coefficient_and_degree_l2311_231149


namespace NUMINAMATH_CALUDE_factorization_proof_l2311_231148

-- Define the expressions
def expr_A (x y : ℝ) : ℝ := x^2 - 4*y^2
def expr_A_factored (x y : ℝ) : ℝ := (x + 2*y) * (x - 2*y)

def expr_B (x y : ℝ) : ℝ := 2*x*(x - 3*y)
def expr_B_expanded (x y : ℝ) : ℝ := 2*x^2 - 6*x*y

def expr_C (x : ℝ) : ℝ := (x + 2)^2
def expr_C_expanded (x : ℝ) : ℝ := x^2 + 4*x + 4

def expr_D (a b c x : ℝ) : ℝ := a*x + b*x + c
def expr_D_simplified (a b c x : ℝ) : ℝ := x*(a + b) + c

-- Theorem stating that A is a factorization while others are not
theorem factorization_proof :
  (∀ x y, expr_A x y = expr_A_factored x y) ∧
  (∃ x y, expr_B x y ≠ expr_B_expanded x y) ∧
  (∃ x, expr_C x ≠ expr_C_expanded x) ∧
  (∃ a b c x, expr_D a b c x ≠ expr_D_simplified a b c x) :=
sorry

end NUMINAMATH_CALUDE_factorization_proof_l2311_231148


namespace NUMINAMATH_CALUDE_nth_terms_equal_condition_l2311_231104

/-- 
Given two arithmetic progressions with n terms, where the sum of the first progression 
is n^2 + pn and the sum of the second progression is 3n^2 - 2n, this theorem states the 
condition for their n-th terms to be equal.
-/
theorem nth_terms_equal_condition (n : ℕ) (p : ℝ) 
  (sum1 : ℝ → ℝ → ℝ) (sum2 : ℝ → ℝ) 
  (h1 : sum1 n p = n^2 + p*n) 
  (h2 : sum2 n = 3*n^2 - 2*n) : 
  (∃ (a1 b1 d e : ℝ), 
    (∀ k : ℕ, k > 0 → k ≤ n → 
      sum1 n p = (n : ℝ)/2 * (a1 + (a1 + (n - 1)*d))) ∧
    (∀ k : ℕ, k > 0 → k ≤ n → 
      sum2 n = (n : ℝ)/2 * (b1 + (b1 + (n - 1)*e))) ∧
    a1 + (n - 1)*d = b1 + (n - 1)*e) ↔ 
  p = 4*(n - 1) := by
sorry

end NUMINAMATH_CALUDE_nth_terms_equal_condition_l2311_231104


namespace NUMINAMATH_CALUDE_cosine_sine_values_l2311_231168

/-- If the sum of cos²ⁿ(θ) from n=0 to infinity equals 9, 
    then cos(2θ) = 7/9 and sin²(θ) = 1/9 -/
theorem cosine_sine_values (θ : ℝ) 
  (h : ∑' n, (Real.cos θ)^(2*n) = 9) : 
  Real.cos (2*θ) = 7/9 ∧ Real.sin θ^2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_values_l2311_231168


namespace NUMINAMATH_CALUDE_parabola_intersection_locus_l2311_231124

/-- Given a parabola y² = 2px with vertex at the origin, 
    prove that the locus of intersection points forms another parabola -/
theorem parabola_intersection_locus (p : ℝ) (h : p > 0) :
  ∃ (f : ℝ → ℝ), 
    (∀ x y : ℝ, y ^ 2 = 2 * p * x → 
      ∃ (x₁ y₁ : ℝ), 
        y₁ ^ 2 = 2 * p * x₁ ∧ 
        (y - y₁) = -(y₁ / p) * (x - x₁) ∧
        y = (p / y₁) * (x - p / 2) ∧
        f x = y) ∧
    (∀ x : ℝ, (f x) ^ 2 = (p / 2) * (x - p / 2)) := by
  sorry


end NUMINAMATH_CALUDE_parabola_intersection_locus_l2311_231124


namespace NUMINAMATH_CALUDE_probability_diamond_then_ace_l2311_231141

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of diamonds in a standard deck -/
def DiamondCount : ℕ := 13

/-- Represents the number of aces in a standard deck -/
def AceCount : ℕ := 4

/-- Represents the remaining deck after one card (not a diamond ace) has been dealt -/
def RemainingDeck : ℕ := StandardDeck - 1

/-- Represents the number of diamonds (excluding ace) in the remaining deck -/
def RemainingDiamonds : ℕ := DiamondCount - 1

theorem probability_diamond_then_ace :
  (RemainingDiamonds : ℚ) / RemainingDeck * AceCount / (RemainingDeck - 1) = 24 / 1275 := by
  sorry

end NUMINAMATH_CALUDE_probability_diamond_then_ace_l2311_231141


namespace NUMINAMATH_CALUDE_norma_banana_problem_l2311_231181

/-- Norma's banana problem -/
theorem norma_banana_problem (initial_bananas : ℕ) (lost_bananas : ℕ) 
  (h1 : initial_bananas = 47) 
  (h2 : lost_bananas = 45) : 
  initial_bananas - lost_bananas = 2 :=
by sorry

end NUMINAMATH_CALUDE_norma_banana_problem_l2311_231181


namespace NUMINAMATH_CALUDE_money_problem_l2311_231176

theorem money_problem (a b : ℝ) 
  (h1 : 5 * a + b > 51)
  (h2 : 3 * a - b = 21) :
  a > 9 ∧ b > 6 := by
sorry

end NUMINAMATH_CALUDE_money_problem_l2311_231176


namespace NUMINAMATH_CALUDE_original_triangle_area_l2311_231143

theorem original_triangle_area (original_area new_area : ℝ) : 
  (new_area = 32) → 
  (new_area = 4 * original_area) → 
  (original_area = 8) := by
sorry

end NUMINAMATH_CALUDE_original_triangle_area_l2311_231143


namespace NUMINAMATH_CALUDE_magnitude_a_minus_b_l2311_231100

def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (3, -2)

theorem magnitude_a_minus_b : 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_minus_b_l2311_231100


namespace NUMINAMATH_CALUDE_set_operation_result_l2311_231119

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {3, 4, 5}

-- Theorem to prove
theorem set_operation_result :
  ((U \ A) ∩ B) = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_result_l2311_231119


namespace NUMINAMATH_CALUDE_candy_calories_per_serving_l2311_231178

/-- Calculates the number of calories per serving in a package of candy. -/
def calories_per_serving (total_servings : ℕ) (half_package_calories : ℕ) : ℕ :=
  (2 * half_package_calories) / total_servings

/-- Proves that the number of calories per serving is 120, given the problem conditions. -/
theorem candy_calories_per_serving :
  calories_per_serving 3 180 = 120 := by
  sorry

end NUMINAMATH_CALUDE_candy_calories_per_serving_l2311_231178


namespace NUMINAMATH_CALUDE_function_property_l2311_231186

-- Define the function f and its property
def f_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y)) = f (x + y) + f x * f y - x * y

-- Define α
def α (f : ℝ → ℝ) : ℝ := f 0

-- Theorem statement
theorem function_property (f : ℝ → ℝ) (h : f_property f) :
  (f (α f) * f (-(α f)) = 0) ∧
  (α f = 0) ∧
  (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l2311_231186


namespace NUMINAMATH_CALUDE_product_of_three_integers_l2311_231174

theorem product_of_three_integers (A B C : Int) : 
  A < B → B < C → A + B + C = 33 → C = 3 * B → A = C - 23 → A * B * C = 192 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_integers_l2311_231174


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2311_231152

theorem complex_magnitude_problem (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) :
  Complex.abs (z + Complex.I) = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2311_231152


namespace NUMINAMATH_CALUDE_children_share_sum_l2311_231131

theorem children_share_sum (total_money : ℕ) (ratio_a ratio_b ratio_c ratio_d : ℕ) : 
  total_money = 4500 → 
  ratio_a = 2 → 
  ratio_b = 4 → 
  ratio_c = 5 → 
  ratio_d = 4 → 
  (ratio_a + ratio_b) * total_money / (ratio_a + ratio_b + ratio_c + ratio_d) = 1800 := by
sorry

end NUMINAMATH_CALUDE_children_share_sum_l2311_231131


namespace NUMINAMATH_CALUDE_expression_decrease_l2311_231130

theorem expression_decrease (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let x' := 0.9 * x
  let y' := 0.7 * y
  (x' ^ 2 * y' ^ 3) / (x ^ 2 * y ^ 3) = 0.27783 := by
sorry

end NUMINAMATH_CALUDE_expression_decrease_l2311_231130


namespace NUMINAMATH_CALUDE_twirly_tea_cups_capacity_l2311_231127

/-- The number of people that can fit in one teacup -/
def people_per_teacup : ℕ := 9

/-- The number of teacups on the ride -/
def number_of_teacups : ℕ := 7

/-- The total number of people that can ride at a time -/
def total_riders : ℕ := people_per_teacup * number_of_teacups

theorem twirly_tea_cups_capacity :
  total_riders = 63 := by sorry

end NUMINAMATH_CALUDE_twirly_tea_cups_capacity_l2311_231127


namespace NUMINAMATH_CALUDE_willies_stickers_l2311_231179

/-- Willie's sticker problem -/
theorem willies_stickers (initial_stickers given_away : ℕ) 
  (h1 : initial_stickers = 36)
  (h2 : given_away = 7) :
  initial_stickers - given_away = 29 := by
  sorry

end NUMINAMATH_CALUDE_willies_stickers_l2311_231179


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l2311_231101

theorem complex_modulus_equality (n : ℝ) (hn : n > 0) :
  Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 13 → n = 10 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l2311_231101


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2311_231123

/-- An ellipse with axes parallel to the coordinate axes, tangent to the x-axis at (6, 0) and tangent to the y-axis at (0, 2) -/
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ -- semi-major axis
  b : ℝ -- semi-minor axis
  h_tangent_x : center.1 - a = 6
  h_tangent_y : center.2 - b = 2
  h_a_gt_b : a > b
  h_positive : a > 0 ∧ b > 0

/-- The distance between the foci of the ellipse is 8√2 -/
theorem ellipse_foci_distance (e : Ellipse) : Real.sqrt 128 = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2311_231123


namespace NUMINAMATH_CALUDE_min_value_of_sum_reciprocals_l2311_231112

theorem min_value_of_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_reciprocals_l2311_231112


namespace NUMINAMATH_CALUDE_college_choice_probability_l2311_231151

theorem college_choice_probability : 
  let num_examinees : ℕ := 2
  let num_colleges : ℕ := 3
  let prob_choose_college : ℚ := 1 / num_colleges
  
  -- Probability that both examinees choose the third college
  let prob_both_choose_third : ℚ := prob_choose_college ^ num_examinees
  
  -- Probability that at least one of the first two colleges is chosen
  let prob_at_least_one_first_two : ℚ := 1 - prob_both_choose_third
  
  prob_at_least_one_first_two = 8 / 9 :=
by sorry

end NUMINAMATH_CALUDE_college_choice_probability_l2311_231151


namespace NUMINAMATH_CALUDE_system_solution_l2311_231167

theorem system_solution (a b : ℚ) : 
  (a/3 - 1) + 2*(b/5 + 2) = 4 ∧ 
  2*(a/3 - 1) + (b/5 + 2) = 5 → 
  a = 9 ∧ b = -5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2311_231167


namespace NUMINAMATH_CALUDE_function_property_l2311_231189

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a
  else Real.log x / Real.log a

-- State the theorem
theorem function_property (a : ℝ) (h : a ≠ 0) (h1 : a ≠ 1) :
  f a (f a 1) = 2 → a = -2 := by sorry

end

end NUMINAMATH_CALUDE_function_property_l2311_231189


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2311_231180

theorem decimal_sum_to_fraction :
  (0.1 : ℚ) + 0.02 + 0.003 + 0.0004 + 0.00005 = 2469 / 20000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2311_231180


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2311_231182

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that in an arithmetic sequence where the fifth term is 2 and the sixth term is 5, the third term is -4. -/
theorem arithmetic_sequence_third_term
  (a : ℕ → ℤ)
  (h_arithmetic : ArithmeticSequence a)
  (h_fifth : a 5 = 2)
  (h_sixth : a 6 = 5) :
  a 3 = -4 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2311_231182


namespace NUMINAMATH_CALUDE_father_son_meeting_point_father_son_meeting_point_specific_l2311_231132

/-- The meeting point of a father and son in a hallway -/
theorem father_son_meeting_point (hallway_length : ℝ) (speed_ratio : ℝ) : 
  hallway_length > 0 → 
  speed_ratio > 1 → 
  (speed_ratio * hallway_length) / (speed_ratio + 1) = 
    hallway_length - hallway_length / (speed_ratio + 1) :=
by
  sorry

/-- The specific case of a 16m hallway and 3:1 speed ratio -/
theorem father_son_meeting_point_specific : 
  (16 : ℝ) - 16 / (3 + 1) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_meeting_point_father_son_meeting_point_specific_l2311_231132


namespace NUMINAMATH_CALUDE_football_points_sum_l2311_231135

theorem football_points_sum : 
  let zach_points : Float := 42.0
  let ben_points : Float := 21.0
  let sarah_points : Float := 18.5
  let emily_points : Float := 27.5
  zach_points + ben_points + sarah_points + emily_points = 109.0 := by
  sorry

end NUMINAMATH_CALUDE_football_points_sum_l2311_231135


namespace NUMINAMATH_CALUDE_total_cookies_l2311_231120

def num_bags : ℕ := 37
def cookies_per_bag : ℕ := 19

theorem total_cookies : num_bags * cookies_per_bag = 703 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l2311_231120


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l2311_231156

theorem function_inequality_implies_parameter_bound 
  (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 4, f x ∈ Set.Icc 1 2) →
  (∀ x ∈ Set.Icc 0 4, f x ^ 2 - a * f x + 2 < 0) →
  a > 3 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l2311_231156


namespace NUMINAMATH_CALUDE_find_k_l2311_231150

theorem find_k (d m n : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, (3*x^2 - 4*x + 2)*(d*x^3 + k*x^2 + m*x + n) = 6*x^5 - 11*x^4 + 14*x^3 - 4*x^2 + 8*x - 3) → 
  (∃ k : ℝ, k = -11/3 ∧ ∀ x : ℝ, (3*x^2 - 4*x + 2)*(d*x^3 + k*x^2 + m*x + n) = 6*x^5 - 11*x^4 + 14*x^3 - 4*x^2 + 8*x - 3) :=
by sorry

end NUMINAMATH_CALUDE_find_k_l2311_231150


namespace NUMINAMATH_CALUDE_max_of_expression_l2311_231165

theorem max_of_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 20) :
  Real.sqrt (x + 50) + Real.sqrt (20 - x) + 2 * Real.sqrt x ≤ 18.124 ∧
  (x = 16 → Real.sqrt (x + 50) + Real.sqrt (20 - x) + 2 * Real.sqrt x = 18.124) :=
by sorry

end NUMINAMATH_CALUDE_max_of_expression_l2311_231165


namespace NUMINAMATH_CALUDE_total_fish_catch_l2311_231158

def fishing_competition (jackson_daily : ℕ) (jonah_daily : ℕ) (george_catches : List ℕ) 
  (lily_catches : List ℕ) (alex_diff : ℕ) : Prop :=
  george_catches.length = 5 ∧ 
  lily_catches.length = 4 ∧
  ∀ i, i < 5 → List.get? (george_catches) i ≠ none ∧
  ∀ i, i < 4 → List.get? (lily_catches) i ≠ none ∧
  (jackson_daily * 5 + jonah_daily * 5 + george_catches.sum + lily_catches.sum + 
    (george_catches.map (λ x => x - alex_diff)).sum) = 159

theorem total_fish_catch : 
  fishing_competition 6 4 [8, 12, 7, 9, 11] [5, 6, 9, 5] 2 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_catch_l2311_231158


namespace NUMINAMATH_CALUDE_joan_apples_l2311_231111

/-- The number of apples Joan has after picking, receiving, and having some taken away. -/
def final_apples (initial : ℕ) (added : ℕ) (taken : ℕ) : ℕ :=
  initial + added - taken

/-- Theorem stating that Joan's final number of apples is 55 given the problem conditions. -/
theorem joan_apples : final_apples 43 27 15 = 55 := by
  sorry

end NUMINAMATH_CALUDE_joan_apples_l2311_231111


namespace NUMINAMATH_CALUDE_f_inequality_implies_m_bound_l2311_231160

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log (Real.sqrt (x^2 + 1) + x)

theorem f_inequality_implies_m_bound (m : ℝ) :
  (∀ x : ℝ, f (2^x - 4^x) + f (m * 2^x - 3) < 0) →
  m < 2 * Real.sqrt 3 - 1 := by
sorry

end NUMINAMATH_CALUDE_f_inequality_implies_m_bound_l2311_231160


namespace NUMINAMATH_CALUDE_correct_ingredients_l2311_231183

/-- Recipe proportions and banana usage --/
structure RecipeData where
  flour_to_mush : ℚ  -- ratio of flour to banana mush
  sugar_to_mush : ℚ  -- ratio of sugar to banana mush
  milk_to_flour : ℚ  -- ratio of milk to flour
  bananas_per_mush : ℕ  -- number of bananas per cup of mush
  total_bananas : ℕ  -- total number of bananas used

/-- Calculated ingredients based on recipe data --/
def calculate_ingredients (r : RecipeData) : ℚ × ℚ × ℚ :=
  let mush := r.total_bananas / r.bananas_per_mush
  let flour := mush * r.flour_to_mush
  let sugar := mush * r.sugar_to_mush
  let milk := flour * r.milk_to_flour
  (flour, sugar, milk)

/-- Theorem stating the correct amounts of ingredients --/
theorem correct_ingredients (r : RecipeData) 
  (h1 : r.flour_to_mush = 3)
  (h2 : r.sugar_to_mush = 2/3)
  (h3 : r.milk_to_flour = 1/6)
  (h4 : r.bananas_per_mush = 4)
  (h5 : r.total_bananas = 32) :
  calculate_ingredients r = (24, 16/3, 4) := by
  sorry

#eval calculate_ingredients {
  flour_to_mush := 3,
  sugar_to_mush := 2/3,
  milk_to_flour := 1/6,
  bananas_per_mush := 4,
  total_bananas := 32
}

end NUMINAMATH_CALUDE_correct_ingredients_l2311_231183


namespace NUMINAMATH_CALUDE_inequality_for_positive_product_l2311_231188

theorem inequality_for_positive_product (a b : ℝ) (h : a * b > 0) :
  b / a + a / b ≥ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_for_positive_product_l2311_231188


namespace NUMINAMATH_CALUDE_selection_schemes_count_l2311_231102

def total_people : ℕ := 6
def cities : ℕ := 4
def excluded_from_paris : ℕ := 2

theorem selection_schemes_count :
  (total_people.choose cities) *
  (cities.factorial) -
  (excluded_from_paris * (total_people - 1).choose (cities - 1) * (cities - 1).factorial) = 240 :=
sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l2311_231102


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_equality_l2311_231107

/-- Given a square ABCD with side length 2 and a rectangle EFGH within it,
    prove that if AE = x, EB = 1, EF = x, and the areas of EFGH and ABE are equal,
    then x = 3/2 -/
theorem rectangle_triangle_area_equality (x : ℝ) : 
  (∀ (A B C D E F G H : ℝ × ℝ),
    -- ABCD is a square with side length 2
    ‖B - A‖ = 2 ∧ ‖C - B‖ = 2 ∧ ‖D - C‖ = 2 ∧ ‖A - D‖ = 2 ∧
    -- EFGH is a rectangle within the square
    (E.1 ≥ A.1 ∧ E.1 ≤ B.1) ∧ (E.2 ≥ A.2 ∧ E.2 ≤ D.2) ∧
    (F.1 ≥ A.1 ∧ F.1 ≤ B.1) ∧ (F.2 ≥ A.2 ∧ F.2 ≤ D.2) ∧
    (G.1 ≥ A.1 ∧ G.1 ≤ B.1) ∧ (G.2 ≥ A.2 ∧ G.2 ≤ D.2) ∧
    (H.1 ≥ A.1 ∧ H.1 ≤ B.1) ∧ (H.2 ≥ A.2 ∧ H.2 ≤ D.2) ∧
    -- AE = x, EB = 1, EF = x
    ‖E - A‖ = x ∧ ‖B - E‖ = 1 ∧ ‖F - E‖ = x ∧
    -- Areas of rectangle EFGH and triangle ABE are equal
    ‖F - E‖ * ‖G - F‖ = (1/2) * ‖E - A‖ * ‖B - E‖) →
  x = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_triangle_area_equality_l2311_231107


namespace NUMINAMATH_CALUDE_stratified_sample_correct_l2311_231171

/-- Represents the three age groups in the population -/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents the population sizes for each age group -/
def populationSize (group : AgeGroup) : Nat :=
  match group with
  | .Elderly => 25
  | .MiddleAged => 35
  | .Young => 40

/-- The total population size -/
def totalPopulation : Nat := populationSize .Elderly + populationSize .MiddleAged + populationSize .Young

/-- The desired sample size -/
def sampleSize : Nat := 40

/-- Calculates the stratified sample size for a given age group -/
def stratifiedSampleSize (group : AgeGroup) : Nat :=
  (populationSize group * sampleSize) / totalPopulation

/-- Theorem stating that the stratified sample sizes are correct -/
theorem stratified_sample_correct :
  stratifiedSampleSize .Elderly = 10 ∧
  stratifiedSampleSize .MiddleAged = 14 ∧
  stratifiedSampleSize .Young = 16 ∧
  stratifiedSampleSize .Elderly + stratifiedSampleSize .MiddleAged + stratifiedSampleSize .Young = sampleSize :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_correct_l2311_231171


namespace NUMINAMATH_CALUDE_fir_trees_count_l2311_231164

/-- Represents the statements made by each child --/
inductive Statement
  | Anya : Statement
  | Borya : Statement
  | Vera : Statement
  | Gena : Statement

/-- Represents the gender of each child --/
inductive Gender
  | Boy : Gender
  | Girl : Gender

/-- Associates each child with their gender --/
def childGender : Statement → Gender
  | Statement.Anya => Gender.Girl
  | Statement.Borya => Gender.Boy
  | Statement.Vera => Gender.Girl
  | Statement.Gena => Gender.Boy

/-- Checks if a given number satisfies a child's statement --/
def satisfiesStatement (n : ℕ) : Statement → Bool
  | Statement.Anya => n = 15
  | Statement.Borya => n % 11 = 0
  | Statement.Vera => n < 25
  | Statement.Gena => n % 22 = 0

/-- Theorem: The number of fir trees is 11 --/
theorem fir_trees_count : 
  ∃ (n : ℕ) (t₁ t₂ : Statement), 
    n = 11 ∧ 
    childGender t₁ ≠ childGender t₂ ∧
    satisfiesStatement n t₁ ∧ 
    satisfiesStatement n t₂ ∧
    (∀ t : Statement, t ≠ t₁ → t ≠ t₂ → ¬satisfiesStatement n t) :=
  sorry

end NUMINAMATH_CALUDE_fir_trees_count_l2311_231164


namespace NUMINAMATH_CALUDE_original_red_marbles_l2311_231117

-- Define the initial number of red and green marbles
variable (r g : ℚ)

-- Define the conditions
def initial_ratio : Prop := r / g = 3 / 2
def new_ratio : Prop := (r - 15) / (g + 25) = 2 / 5

-- State the theorem
theorem original_red_marbles 
  (h1 : initial_ratio r g) 
  (h2 : new_ratio r g) : 
  r = 375 / 11 := by
  sorry

end NUMINAMATH_CALUDE_original_red_marbles_l2311_231117


namespace NUMINAMATH_CALUDE_max_customers_interviewed_l2311_231163

theorem max_customers_interviewed (total : ℕ) (impulsive : ℕ) (ad_influence_percent : ℚ) (consultant_ratio : ℚ) : 
  total ≤ 50 ∧ 
  impulsive = 7 ∧ 
  ad_influence_percent = 3/4 ∧ 
  consultant_ratio = 1/3 →
  ∃ (max_customers : ℕ), 
    max_customers ≤ 50 ∧
    (∃ (ad_influenced : ℕ) (consultant_advised : ℕ),
      max_customers = impulsive + ad_influenced + consultant_advised ∧
      ad_influenced = ⌊(max_customers - impulsive) * ad_influence_percent⌋ ∧
      consultant_advised = ⌊ad_influenced * consultant_ratio⌋) ∧
    ∀ (n : ℕ), n > max_customers →
      ¬(∃ (ad_influenced : ℕ) (consultant_advised : ℕ),
        n = impulsive + ad_influenced + consultant_advised ∧
        ad_influenced = ⌊(n - impulsive) * ad_influence_percent⌋ ∧
        consultant_advised = ⌊ad_influenced * consultant_ratio⌋) ∧
    max_customers = 47 :=
by sorry

end NUMINAMATH_CALUDE_max_customers_interviewed_l2311_231163


namespace NUMINAMATH_CALUDE_garage_sale_dvd_average_price_l2311_231138

/-- Calculate the average price of DVDs bought at a garage sale --/
theorem garage_sale_dvd_average_price : 
  let box1_count : ℕ := 10
  let box1_price : ℚ := 2
  let box2_count : ℕ := 5
  let box2_price : ℚ := 5
  let box3_count : ℕ := 3
  let box3_price : ℚ := 7
  let box4_count : ℕ := 4
  let box4_price : ℚ := 7/2
  let discount_rate : ℚ := 15/100
  let tax_rate : ℚ := 10/100
  let total_count : ℕ := box1_count + box2_count + box3_count + box4_count
  let total_cost : ℚ := 
    box1_count * box1_price + 
    box2_count * box2_price + 
    box3_count * box3_price + 
    box4_count * box4_price
  let discounted_cost : ℚ := total_cost * (1 - discount_rate)
  let final_cost : ℚ := discounted_cost * (1 + tax_rate)
  let average_price : ℚ := final_cost / total_count
  average_price = 17/5 := by sorry

end NUMINAMATH_CALUDE_garage_sale_dvd_average_price_l2311_231138


namespace NUMINAMATH_CALUDE_area_equality_of_circles_l2311_231145

theorem area_equality_of_circles (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 25) (h₃ : r₃ = 20) :
  π * r₃^2 = π * r₂^2 - π * r₁^2 := by
  sorry

#check area_equality_of_circles

end NUMINAMATH_CALUDE_area_equality_of_circles_l2311_231145


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2311_231198

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if its focus (c, 0) is symmetric about the asymptote y = (b/a)x and 
    the symmetric point of the focus lies on the other asymptote y = -(b/a)x,
    then its eccentricity is 2. -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = (b/a) * x ∧ (x - c)^2 + y^2 = (x + c)^2 + y^2) →
  (∃ x y : ℝ, y = -(b/a) * x ∧ (x + c)^2 + y^2 = 0) →
  c / a = 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2311_231198


namespace NUMINAMATH_CALUDE_digits_zeros_equality_l2311_231196

/-- 
Given a positive integer n, count_digits n returns the sum of all digits in n.
-/
def count_digits (n : ℕ) : ℕ := sorry

/-- 
Given a positive integer n, count_zeros n returns the number of zeros in n.
-/
def count_zeros (n : ℕ) : ℕ := sorry

/-- 
sum_digits_to_n n returns the sum of digits of all numbers from 1 to n.
-/
def sum_digits_to_n (n : ℕ) : ℕ := sorry

/-- 
sum_zeros_to_n n returns the count of zeros in all numbers from 1 to n.
-/
def sum_zeros_to_n (n : ℕ) : ℕ := sorry

/-- 
For any positive integer k, the sum of digits in all numbers from 1 to 10^k
is equal to the count of zeros in all numbers from 1 to 10^(k+1).
-/
theorem digits_zeros_equality (k : ℕ) (h : k > 0) : 
  sum_digits_to_n (10^k) = sum_zeros_to_n (10^(k+1)) := by sorry

end NUMINAMATH_CALUDE_digits_zeros_equality_l2311_231196


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2311_231122

theorem polynomial_simplification (x : ℝ) :
  (3*x - 2) * (6*x^12 + 3*x^11 + 6*x^10 + 3*x^9) =
  18*x^13 - 3*x^12 + 12*x^11 - 3*x^10 - 6*x^9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2311_231122


namespace NUMINAMATH_CALUDE_f_minus_p_div_by_9_l2311_231161

/-- Function that computes the sum of all numbers possible by removing some digits of n -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating that f(p) - p is divisible by 9 for any 2011-digit integer p -/
theorem f_minus_p_div_by_9 (p : ℕ) (h : 10^2010 ≤ p ∧ p < 10^2011) : 
  9 ∣ (f p - p) := by
  sorry

end NUMINAMATH_CALUDE_f_minus_p_div_by_9_l2311_231161


namespace NUMINAMATH_CALUDE_binomial_factorial_product_l2311_231144

theorem binomial_factorial_product : Nat.choose 20 6 * Nat.factorial 6 = 27907200 := by
  sorry

end NUMINAMATH_CALUDE_binomial_factorial_product_l2311_231144


namespace NUMINAMATH_CALUDE_intersection_difference_l2311_231106

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1
def parabola2 (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

-- Define the theorem
theorem intersection_difference (a b c d : ℝ) 
  (h1 : parabola1 a = parabola2 a) 
  (h2 : parabola1 c = parabola2 c) 
  (h3 : c ≥ a) : 
  c - a = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_difference_l2311_231106


namespace NUMINAMATH_CALUDE_matts_plantation_length_l2311_231195

/-- Represents Matt's peanut plantation and its production process -/
structure PeanutPlantation where
  width : ℝ
  length : ℝ
  peanuts_per_sqft : ℝ
  peanuts_to_butter_ratio : ℝ
  butter_price_per_kg : ℝ
  total_revenue : ℝ

/-- Calculates the length of one side of Matt's plantation -/
def calculate_plantation_length (p : PeanutPlantation) : ℝ :=
  p.width

/-- Theorem stating that given the conditions, the length of Matt's plantation is 500 feet -/
theorem matts_plantation_length (p : PeanutPlantation) 
  (h1 : p.length = 500)
  (h2 : p.peanuts_per_sqft = 50)
  (h3 : p.peanuts_to_butter_ratio = 20 / 5)
  (h4 : p.butter_price_per_kg = 10)
  (h5 : p.total_revenue = 31250) :
  calculate_plantation_length p = 500 := by
  sorry

end NUMINAMATH_CALUDE_matts_plantation_length_l2311_231195


namespace NUMINAMATH_CALUDE_ages_sum_l2311_231184

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l2311_231184


namespace NUMINAMATH_CALUDE_apple_basket_problem_l2311_231197

theorem apple_basket_problem :
  ∃ (a b : ℕ), 4 * a + 3 * a + 3 * b + 2 * b = 31 ∧ 3 * a + 2 * b = 13 :=
by sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l2311_231197


namespace NUMINAMATH_CALUDE_profit_percent_from_cost_price_ratio_l2311_231133

/-- Profit percent calculation given cost price as a percentage of selling price -/
theorem profit_percent_from_cost_price_ratio (selling_price : ℝ) (cost_price_ratio : ℝ) 
  (h : cost_price_ratio = 0.8) : 
  (selling_price - cost_price_ratio * selling_price) / (cost_price_ratio * selling_price) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_from_cost_price_ratio_l2311_231133


namespace NUMINAMATH_CALUDE_solution1_composition_l2311_231193

/-- Represents a solution with lemonade and carbonated water -/
structure Solution :=
  (lemonade : ℝ)
  (carbonated_water : ℝ)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution1 : Solution)
  (solution2 : Solution)
  (proportion1 : ℝ)

/-- Theorem stating the composition of Solution 1 given the mixture properties -/
theorem solution1_composition 
  (s1 : Solution)
  (s2 : Solution)
  (m : Mixture)
  (h1 : s1.lemonade = 20)
  (h2 : s2.lemonade = 45)
  (h3 : s2.carbonated_water = 55)
  (h4 : m.solution1 = s1)
  (h5 : m.solution2 = s2)
  (h6 : m.proportion1 = 20)
  (h7 : m.proportion1 * s1.carbonated_water + (100 - m.proportion1) * s2.carbonated_water = 60 * 100) :
  s1.carbonated_water = 80 := by
  sorry

end NUMINAMATH_CALUDE_solution1_composition_l2311_231193


namespace NUMINAMATH_CALUDE_line_arrangement_with_restriction_l2311_231139

def number_of_students : ℕ := 5

def total_permutations (n : ℕ) : ℕ := Nat.factorial n

def restricted_permutations (n : ℕ) : ℕ := 
  Nat.factorial (n - 1) * 2

theorem line_arrangement_with_restriction : 
  total_permutations number_of_students - restricted_permutations number_of_students = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_arrangement_with_restriction_l2311_231139


namespace NUMINAMATH_CALUDE_general_equation_proof_l2311_231191

theorem general_equation_proof (n : ℝ) (h1 : n ≠ 4) (h2 : n ≠ 8) :
  n / (n - 4) + (8 - n) / ((8 - n) - 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_general_equation_proof_l2311_231191


namespace NUMINAMATH_CALUDE_cost_of_three_batches_l2311_231154

/-- Represents the cost and quantity of ingredients for yogurt production -/
structure YogurtProduction where
  milk_price : ℝ
  fruit_price : ℝ
  milk_per_batch : ℝ
  fruit_per_batch : ℝ

/-- Calculates the cost of producing a given number of yogurt batches -/
def cost_of_batches (y : YogurtProduction) (num_batches : ℝ) : ℝ :=
  num_batches * (y.milk_price * y.milk_per_batch + y.fruit_price * y.fruit_per_batch)

/-- Theorem: The cost of producing three batches of yogurt is $63 -/
theorem cost_of_three_batches :
  ∃ (y : YogurtProduction),
    y.milk_price = 1.5 ∧
    y.fruit_price = 2 ∧
    y.milk_per_batch = 10 ∧
    y.fruit_per_batch = 3 ∧
    cost_of_batches y 3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_three_batches_l2311_231154


namespace NUMINAMATH_CALUDE_average_team_size_l2311_231194

theorem average_team_size (boys girls teams : ℕ) (h1 : boys = 83) (h2 : girls = 77) (h3 : teams = 4) :
  (boys + girls) / teams = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_team_size_l2311_231194


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2311_231129

theorem arithmetic_calculation : 2 + 7 * 3 - 4 + 8 * 2 / 4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2311_231129


namespace NUMINAMATH_CALUDE_students_taking_only_history_l2311_231140

theorem students_taking_only_history (total : ℕ) (history : ℕ) (statistics : ℕ) (physics : ℕ) (chemistry : ℕ)
  (hist_stat : ℕ) (hist_phys : ℕ) (hist_chem : ℕ) (stat_phys : ℕ) (stat_chem : ℕ) (phys_chem : ℕ) (all_four : ℕ)
  (h_total : total = 500)
  (h_history : history = 150)
  (h_statistics : statistics = 130)
  (h_physics : physics = 120)
  (h_chemistry : chemistry = 100)
  (h_hist_stat : hist_stat = 60)
  (h_hist_phys : hist_phys = 50)
  (h_hist_chem : hist_chem = 40)
  (h_stat_phys : stat_phys = 35)
  (h_stat_chem : stat_chem = 30)
  (h_phys_chem : phys_chem = 25)
  (h_all_four : all_four = 20) :
  history - hist_stat - hist_phys - hist_chem + all_four = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_only_history_l2311_231140


namespace NUMINAMATH_CALUDE_division_multiplication_result_l2311_231185

theorem division_multiplication_result : (-6) / (-6) * (-1/6 : ℚ) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l2311_231185


namespace NUMINAMATH_CALUDE_division_problem_l2311_231166

theorem division_problem : (62976 : ℕ) / 512 = 123 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2311_231166


namespace NUMINAMATH_CALUDE_intersection_contains_two_elements_l2311_231110

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = 3 * p.1^2}
def N : Set (ℝ × ℝ) := {p | p.2 = 5 * p.1}

-- State the theorem
theorem intersection_contains_two_elements :
  ∃ (a b : ℝ × ℝ), a ≠ b ∧ a ∈ M ∩ N ∧ b ∈ M ∩ N ∧
  ∀ c, c ∈ M ∩ N → c = a ∨ c = b :=
sorry

end NUMINAMATH_CALUDE_intersection_contains_two_elements_l2311_231110


namespace NUMINAMATH_CALUDE_series_sum_equals_81_and_two_fifths_l2311_231187

def series_sum : ℚ :=
  1 + 3 * (1/6) + 5 * (1/12) + 7 * (1/20) + 9 * (1/30) + 11 * (1/42) + 
  13 * (1/56) + 15 * (1/72) + 17 * (1/90)

theorem series_sum_equals_81_and_two_fifths : 
  series_sum = 81 + 2/5 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_81_and_two_fifths_l2311_231187


namespace NUMINAMATH_CALUDE_kenny_sunday_jumping_jacks_l2311_231142

/-- Represents the number of jumping jacks Kenny did on each day of the week -/
structure WeeklyJumpingJacks where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total number of jumping jacks for a week -/
def totalJumpingJacks (week : WeeklyJumpingJacks) : ℕ :=
  week.sunday + week.monday + week.tuesday + week.wednesday + week.thursday + week.friday + week.saturday

theorem kenny_sunday_jumping_jacks 
  (lastWeek : ℕ) 
  (thisWeek : WeeklyJumpingJacks) 
  (h1 : lastWeek = 324)
  (h2 : thisWeek.tuesday = 0)
  (h3 : thisWeek.wednesday = 123)
  (h4 : thisWeek.thursday = 64)
  (h5 : thisWeek.friday = 23)
  (h6 : thisWeek.saturday = 61)
  (h7 : thisWeek.monday = 20 ∨ thisWeek.sunday = 20)
  (h8 : totalJumpingJacks thisWeek > lastWeek) :
  thisWeek.sunday = 33 := by
  sorry

end NUMINAMATH_CALUDE_kenny_sunday_jumping_jacks_l2311_231142


namespace NUMINAMATH_CALUDE_exactly_seven_numbers_with_one_ninth_property_l2311_231134

/-- A four-digit number -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The three-digit number obtained by removing the leftmost digit of a four-digit number -/
def RemoveLeftmostDigit (n : ℕ) : ℕ := n % 1000

/-- The property that the three-digit number obtained by removing the leftmost digit is one ninth of the original number -/
def HasOneNinthProperty (n : ℕ) : Prop :=
  FourDigitNumber n ∧ RemoveLeftmostDigit n = n / 9

/-- The theorem stating that there are exactly 7 numbers satisfying the property -/
theorem exactly_seven_numbers_with_one_ninth_property :
  ∃! (s : Finset ℕ), (∀ n ∈ s, HasOneNinthProperty n) ∧ s.card = 7 := by sorry

end NUMINAMATH_CALUDE_exactly_seven_numbers_with_one_ninth_property_l2311_231134


namespace NUMINAMATH_CALUDE_walking_distance_l2311_231115

theorem walking_distance (x y : ℝ) 
  (h1 : x / 4 + y / 3 + y / 6 + x / 4 = 5) : x + y = 10 ∧ 2 * (x + y) = 20 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_l2311_231115


namespace NUMINAMATH_CALUDE_equation_solution_l2311_231125

theorem equation_solution : ∃ x : ℝ, (x / 3 + (30 - x) / 2 = 5) ∧ (x = 60) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2311_231125


namespace NUMINAMATH_CALUDE_jennifer_grooming_time_l2311_231153

/-- Calculates the total grooming time in hours for a given number of dogs, 
    grooming time per dog, and number of days. -/
def totalGroomingTime (numDogs : ℕ) (groomTimePerDog : ℕ) (numDays : ℕ) : ℚ :=
  (numDogs * groomTimePerDog * numDays : ℚ) / 60

/-- Proves that Jennifer spends 20 hours grooming her dogs in 30 days. -/
theorem jennifer_grooming_time :
  totalGroomingTime 2 20 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_grooming_time_l2311_231153


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l2311_231109

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  total_time : ℕ
  change_time : ℕ
  num_changes : ℕ

/-- Calculates the probability of observing a color change -/
def probability_of_change (cycle : TrafficLightCycle) : ℚ :=
  (cycle.change_time * cycle.num_changes : ℚ) / cycle.total_time

/-- Theorem: The probability of observing a color change in the given traffic light cycle is 2/9 -/
theorem traffic_light_change_probability :
  let cycle : TrafficLightCycle := ⟨90, 5, 4⟩
  probability_of_change cycle = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_change_probability_l2311_231109
