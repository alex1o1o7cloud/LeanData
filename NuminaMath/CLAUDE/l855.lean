import Mathlib

namespace NUMINAMATH_CALUDE_instantaneous_speed_at_t_1_l855_85523

/-- The displacement function for the particle's motion --/
def s (t : ℝ) : ℝ := 2 * t^3

/-- The velocity function (derivative of displacement) --/
def v (t : ℝ) : ℝ := 6 * t^2

theorem instantaneous_speed_at_t_1 :
  v 1 = 6 := by sorry

end NUMINAMATH_CALUDE_instantaneous_speed_at_t_1_l855_85523


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l855_85506

theorem negative_fractions_comparison : -1/2 < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l855_85506


namespace NUMINAMATH_CALUDE_prob_ace_then_king_standard_deck_l855_85532

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_aces : ℕ)
  (num_kings : ℕ)

/-- Calculates the probability of drawing an Ace first and a King second from a standard deck -/
def prob_ace_then_king (d : Deck) : ℚ :=
  (d.num_aces : ℚ) / d.total_cards * (d.num_kings : ℚ) / (d.total_cards - 1)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_aces := 4,
    num_kings := 4 }

theorem prob_ace_then_king_standard_deck :
  prob_ace_then_king standard_deck = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_then_king_standard_deck_l855_85532


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l855_85587

theorem wrong_mark_calculation (n : ℕ) (correct_mark : ℝ) (average_increase : ℝ) : 
  n = 56 → 
  correct_mark = 45 → 
  average_increase = 1/2 → 
  ∃ x : ℝ, x - correct_mark = n * average_increase ∧ x = 73 := by
sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l855_85587


namespace NUMINAMATH_CALUDE_geometric_sum_problem_l855_85558

/-- Sum of a finite geometric series -/
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_problem : 
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 4095/12288 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_problem_l855_85558


namespace NUMINAMATH_CALUDE_oil_price_reduction_l855_85551

/-- Given a 20% reduction in the price of oil, if a housewife can obtain 5 kg more for Rs. 800 after the reduction, then the reduced price per kg is Rs. 32. -/
theorem oil_price_reduction (P : ℝ) (h1 : P > 0) :
  let R := 0.8 * P
  800 / R - 800 / P = 5 →
  R = 32 := by sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l855_85551


namespace NUMINAMATH_CALUDE_solve_square_equation_solve_cubic_equation_l855_85541

-- Part 1
theorem solve_square_equation :
  ∀ x : ℝ, (x - 1)^2 = 9 ↔ x = 4 ∨ x = -2 :=
by sorry

-- Part 2
theorem solve_cubic_equation :
  ∀ x : ℝ, (1/3) * (x + 3)^3 - 9 = 0 ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_solve_square_equation_solve_cubic_equation_l855_85541


namespace NUMINAMATH_CALUDE_light_could_be_green_l855_85536

/-- Represents the state of a traffic light -/
inductive TrafficLightState
| Red
| Green
| Yellow

/-- Represents a traffic light with its cycle durations -/
structure TrafficLight where
  total_cycle : ℕ
  red_duration : ℕ
  green_duration : ℕ
  yellow_duration : ℕ
  cycle_valid : total_cycle = red_duration + green_duration + yellow_duration

/-- Defines the specific traffic light from the problem -/
def intersection_light : TrafficLight :=
  { total_cycle := 60
  , red_duration := 30
  , green_duration := 25
  , yellow_duration := 5
  , cycle_valid := by rfl }

/-- Theorem stating that the traffic light could be green at any random observation -/
theorem light_could_be_green (t : ℕ) : 
  ∃ (s : TrafficLightState), s = TrafficLightState.Green :=
sorry

end NUMINAMATH_CALUDE_light_could_be_green_l855_85536


namespace NUMINAMATH_CALUDE_max_xy_on_line_segment_l855_85588

/-- The maximum value of xy for a point P(x,y) on the line segment AB, where A(3,0) and B(0,4) -/
theorem max_xy_on_line_segment : ∀ x y : ℝ, 
  (x / 3 + y / 4 = 1) → -- Point P(x,y) is on the line segment AB
  (x ≥ 0 ∧ y ≥ 0) →    -- P is between A and B (non-negative coordinates)
  (x ≤ 3 ∧ y ≤ 4) →    -- P is between A and B (upper bounds)
  x * y ≤ 3 :=         -- The maximum value of xy is 3
by
  sorry

#check max_xy_on_line_segment

end NUMINAMATH_CALUDE_max_xy_on_line_segment_l855_85588


namespace NUMINAMATH_CALUDE_integer_between_sqrt3_plus_1_and_sqrt11_l855_85562

theorem integer_between_sqrt3_plus_1_and_sqrt11 :
  ∃! n : ℤ, (Real.sqrt 3 + 1 < n) ∧ (n < Real.sqrt 11) :=
by
  -- We assume the following inequalities as given:
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := sorry
  have h2 : 3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4 := sorry

  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt3_plus_1_and_sqrt11_l855_85562


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l855_85574

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = (1 : ℝ) / Real.sqrt (x - 2)) ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l855_85574


namespace NUMINAMATH_CALUDE_binomial_p_value_l855_85581

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  mean : ℝ
  variance : ℝ
  mean_eq : mean = n * p
  variance_eq : variance = n * p * (1 - p)

/-- Theorem stating the value of p for a binomial random variable with given mean and variance -/
theorem binomial_p_value (ξ : BinomialRV) 
  (h_mean : ξ.mean = 300)
  (h_var : ξ.variance = 200) :
  ξ.p = 1/3 := by
  sorry

#check binomial_p_value

end NUMINAMATH_CALUDE_binomial_p_value_l855_85581


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l855_85590

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (l : Line) (α β : Plane)
  (h1 : subset l α)
  (h2 : perp l β) :
  perpPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l855_85590


namespace NUMINAMATH_CALUDE_imaginary_unit_equation_l855_85546

theorem imaginary_unit_equation : Complex.I ^ 3 - 2 / Complex.I = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_equation_l855_85546


namespace NUMINAMATH_CALUDE_least_number_remainder_l855_85520

theorem least_number_remainder : ∃ (r : ℕ), r > 0 ∧ 386 % 35 = r ∧ 386 % 11 = r := by
  sorry

end NUMINAMATH_CALUDE_least_number_remainder_l855_85520


namespace NUMINAMATH_CALUDE_fifth_pythagorean_triple_l855_85559

/-- Generates the nth Pythagorean triple based on the given pattern -/
def pythagoreanTriple (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := 2 * n + 1
  let b := 2 * n * (n + 1)
  let c := 2 * n * (n + 1) + 1
  (a, b, c)

/-- Checks if a triple of natural numbers forms a Pythagorean triple -/
def isPythagoreanTriple (triple : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := triple
  a * a + b * b = c * c

theorem fifth_pythagorean_triple :
  let triple := pythagoreanTriple 5
  triple = (11, 60, 61) ∧ isPythagoreanTriple triple :=
by sorry

end NUMINAMATH_CALUDE_fifth_pythagorean_triple_l855_85559


namespace NUMINAMATH_CALUDE_fraction_calculation_l855_85508

theorem fraction_calculation : 
  (2 / 5 + 3 / 7) / ((4 / 9) * (1 / 8)) = 522 / 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l855_85508


namespace NUMINAMATH_CALUDE_election_majority_l855_85539

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 6000 → 
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 1200 := by
sorry

end NUMINAMATH_CALUDE_election_majority_l855_85539


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l855_85580

theorem triangle_angle_inequality (A B C : ℝ) (h_triangle : A + B + C = π) (h_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  2 * (Real.sin A / A + Real.sin B / B + Real.sin C / C) ≤
  (1/B + 1/C) * Real.sin A + (1/C + 1/A) * Real.sin B + (1/A + 1/B) * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l855_85580


namespace NUMINAMATH_CALUDE_binomial_distribution_problem_l855_85531

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

variable (ξ : BinomialDistribution)

/-- The expected value of a binomial distribution -/
def expectation (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: For a binomial distribution with E[ξ] = 300 and D[ξ] = 200, p = 1/3 -/
theorem binomial_distribution_problem (ξ : BinomialDistribution) 
  (h2 : expectation ξ = 300) (h3 : variance ξ = 200) : ξ.p = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_binomial_distribution_problem_l855_85531


namespace NUMINAMATH_CALUDE_product_of_roots_l855_85517

theorem product_of_roots : Real.sqrt 16 * (27 ^ (1/3 : ℝ)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l855_85517


namespace NUMINAMATH_CALUDE_document_typing_time_l855_85589

theorem document_typing_time (barbara_speed jim_speed : ℕ) (document_length : ℕ) (jim_time : ℕ) :
  barbara_speed = 172 →
  jim_speed = 100 →
  document_length = 3440 →
  jim_time = 20 →
  ∃ t : ℕ, t < jim_time ∧ t * (barbara_speed + jim_speed) ≥ document_length :=
by sorry

end NUMINAMATH_CALUDE_document_typing_time_l855_85589


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l855_85513

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, x^2 = 1 + 4*y^3*(y + 2) ↔ 
    (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l855_85513


namespace NUMINAMATH_CALUDE_extremum_point_implies_k_range_l855_85563

noncomputable def f (x k : ℝ) : ℝ := (Real.exp x) / (x^2) - k * (2/x + Real.log x)

theorem extremum_point_implies_k_range :
  (∀ x : ℝ, x > 0 → (∀ y : ℝ, y > 0 → f x k = f y k → x = y ∨ x = 2)) →
  k ∈ Set.Iic (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_extremum_point_implies_k_range_l855_85563


namespace NUMINAMATH_CALUDE_import_tax_problem_l855_85579

/-- Calculates the import tax percentage given the total value, tax-free portion, and tax amount -/
def import_tax_percentage (total_value tax_free_portion tax_amount : ℚ) : ℚ :=
  (tax_amount / (total_value - tax_free_portion)) * 100

/-- Proves that the import tax percentage is 7% given the problem conditions -/
theorem import_tax_problem :
  let total_value : ℚ := 2610
  let tax_free_portion : ℚ := 1000
  let tax_amount : ℚ := 1127/10
by
  sorry


end NUMINAMATH_CALUDE_import_tax_problem_l855_85579


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l855_85518

theorem largest_divisor_of_expression (p q : ℤ) 
  (hp : Odd p) (hq : Odd q) (hlt : q < p) : 
  ∃ k : ℤ, p^2 - q^2 + 2*p - 2*q = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l855_85518


namespace NUMINAMATH_CALUDE_unique_pizza_combinations_l855_85556

def number_of_toppings : ℕ := 8
def toppings_per_pizza : ℕ := 5

theorem unique_pizza_combinations : 
  (number_of_toppings.choose toppings_per_pizza) = 56 := by
  sorry

end NUMINAMATH_CALUDE_unique_pizza_combinations_l855_85556


namespace NUMINAMATH_CALUDE_cleaning_time_proof_l855_85505

theorem cleaning_time_proof (total_time : ℝ) (lilly_fraction : ℝ) : 
  total_time = 8 → lilly_fraction = 1/4 → 
  (total_time - lilly_fraction * total_time) * 60 = 360 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_time_proof_l855_85505


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l855_85569

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 0 ∧ x₂ = 3/2) ∧ 
  (∀ x : ℝ, 2*x^2 - 3*x = 0 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l855_85569


namespace NUMINAMATH_CALUDE_fourth_coefficient_equals_five_l855_85585

theorem fourth_coefficient_equals_five (x a₁ a₂ a₃ a₄ a₅ aₙ : ℝ) :
  (∀ x, x^5 = aₙ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₄ = 5 := by
sorry

end NUMINAMATH_CALUDE_fourth_coefficient_equals_five_l855_85585


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l855_85573

theorem arithmetic_sequence_middle_term (y : ℝ) :
  y > 0 ∧ 
  (∃ (d : ℝ), y^2 - 2^2 = d ∧ 5^2 - y^2 = d) →
  y = Real.sqrt 14.5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l855_85573


namespace NUMINAMATH_CALUDE_oak_grove_library_books_l855_85540

theorem oak_grove_library_books :
  let public_library : ℝ := 1986
  let school_libraries : ℝ := 5106
  let community_college_library : ℝ := 3294.5
  let medical_library : ℝ := 1342.25
  let law_library : ℝ := 2785.75
  public_library + school_libraries + community_college_library + medical_library + law_library = 15514.5 := by
  sorry

end NUMINAMATH_CALUDE_oak_grove_library_books_l855_85540


namespace NUMINAMATH_CALUDE_ornamental_rings_ratio_l855_85572

theorem ornamental_rings_ratio (initial_purchase : ℕ) (mother_purchase : ℕ) (sold_after : ℕ) (remaining : ℕ) :
  initial_purchase = 200 →
  mother_purchase = 300 →
  sold_after = 150 →
  remaining = 225 →
  ∃ (original_stock : ℕ),
    initial_purchase + original_stock > 0 ∧
    (1 / 4 : ℚ) * (initial_purchase + original_stock : ℚ) + (mother_purchase : ℚ) - (sold_after : ℚ) = (remaining : ℚ) ∧
    (initial_purchase : ℚ) / (original_stock : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ornamental_rings_ratio_l855_85572


namespace NUMINAMATH_CALUDE_four_different_results_l855_85534

/-- Represents a parenthesized expression of 3^3^3^3 -/
inductive ParenthesizedExpr
| Single : ParenthesizedExpr
| Left : ParenthesizedExpr → ParenthesizedExpr
| Right : ParenthesizedExpr → ParenthesizedExpr
| Both : ParenthesizedExpr → ParenthesizedExpr → ParenthesizedExpr

/-- Evaluates a parenthesized expression to a natural number -/
def evaluate : ParenthesizedExpr → ℕ
| ParenthesizedExpr.Single => 3^3^3^3
| ParenthesizedExpr.Left e => 3^(evaluate e)
| ParenthesizedExpr.Right e => (evaluate e)^3
| ParenthesizedExpr.Both e1 e2 => (evaluate e1)^(evaluate e2)

/-- All possible parenthesized expressions of 3^3^3^3 -/
def allExpressions : List ParenthesizedExpr := [
  ParenthesizedExpr.Single,
  ParenthesizedExpr.Left (ParenthesizedExpr.Left (ParenthesizedExpr.Single)),
  ParenthesizedExpr.Left (ParenthesizedExpr.Right ParenthesizedExpr.Single),
  ParenthesizedExpr.Right (ParenthesizedExpr.Left ParenthesizedExpr.Single),
  ParenthesizedExpr.Right (ParenthesizedExpr.Right ParenthesizedExpr.Single),
  ParenthesizedExpr.Both ParenthesizedExpr.Single ParenthesizedExpr.Single
]

/-- The theorem stating that there are exactly 4 different results -/
theorem four_different_results :
  (allExpressions.map evaluate).toFinset.card = 4 := by sorry

end NUMINAMATH_CALUDE_four_different_results_l855_85534


namespace NUMINAMATH_CALUDE_laptop_price_theorem_l855_85583

/-- The sticker price of the laptop -/
def stickerPrice : ℝ := 900

/-- The price at Store P -/
def storePPrice (price : ℝ) : ℝ := 0.8 * price - 120

/-- The price at Store Q -/
def storeQPrice (price : ℝ) : ℝ := 0.7 * price

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem laptop_price_theorem :
  storeQPrice stickerPrice - storePPrice stickerPrice = 30 := by
  sorry

#check laptop_price_theorem

end NUMINAMATH_CALUDE_laptop_price_theorem_l855_85583


namespace NUMINAMATH_CALUDE_complementary_angles_theorem_l855_85582

theorem complementary_angles_theorem (x : ℝ) : 
  (2 * x + 3 * x = 90) → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_theorem_l855_85582


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l855_85555

theorem tennis_tournament_matches (total_players : ℕ) (bye_players : ℕ) (first_round_players : ℕ) (first_round_matches : ℕ) :
  total_players = 128 →
  bye_players = 36 →
  first_round_players = 92 →
  first_round_matches = 46 →
  first_round_players = 2 * first_round_matches →
  total_players = bye_players + first_round_players →
  (∃ (total_matches : ℕ), total_matches = first_round_matches + (total_players - 1) ∧ total_matches = 127) :=
by sorry

end NUMINAMATH_CALUDE_tennis_tournament_matches_l855_85555


namespace NUMINAMATH_CALUDE_emily_sixth_score_l855_85544

def emily_scores : List ℕ := [91, 94, 88, 90, 101]
def target_mean : ℕ := 95
def num_quizzes : ℕ := 6

theorem emily_sixth_score :
  ∃ (sixth_score : ℕ),
    (emily_scores.sum + sixth_score) / num_quizzes = target_mean ∧
    sixth_score = 106 := by
  sorry

end NUMINAMATH_CALUDE_emily_sixth_score_l855_85544


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l855_85577

theorem trig_expression_equals_one : 
  (Real.tan (30 * π / 180))^2 - (Real.sin (30 * π / 180))^2 = 
  (Real.tan (30 * π / 180))^2 * (Real.sin (30 * π / 180))^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l855_85577


namespace NUMINAMATH_CALUDE_factorial_ratio_squared_l855_85598

theorem factorial_ratio_squared (M : ℕ) : 
  (Nat.factorial (M + 1) : ℚ) / (Nat.factorial (M + 2) : ℚ)^2 = 1 / ((M + 2 : ℚ)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_squared_l855_85598


namespace NUMINAMATH_CALUDE_unique_integer_representation_l855_85525

theorem unique_integer_representation (A m n p : ℕ) : 
  A > 0 ∧ 
  m ≥ n ∧ n ≥ p ∧ p ≥ 1 ∧
  A = (m - 1/n) * (n - 1/p) * (p - 1/m) →
  A = 21 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_representation_l855_85525


namespace NUMINAMATH_CALUDE_maximize_product_l855_85548

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 40) :
  x^6 * y^3 ≤ 24^6 * 16^3 ∧
  (x^6 * y^3 = 24^6 * 16^3 ↔ x = 24 ∧ y = 16) :=
sorry

end NUMINAMATH_CALUDE_maximize_product_l855_85548


namespace NUMINAMATH_CALUDE_subcommittee_count_l855_85545

/-- The number of members in the planning committee -/
def totalMembers : ℕ := 12

/-- The number of professors in the planning committee -/
def professorCount : ℕ := 5

/-- The size of the subcommittee -/
def subcommitteeSize : ℕ := 4

/-- The minimum number of professors required in the subcommittee -/
def minProfessors : ℕ := 2

/-- Calculates the number of valid subcommittees -/
def validSubcommittees : ℕ := sorry

theorem subcommittee_count :
  validSubcommittees = 285 := by sorry

end NUMINAMATH_CALUDE_subcommittee_count_l855_85545


namespace NUMINAMATH_CALUDE_coefficient_of_y_l855_85515

theorem coefficient_of_y (y : ℝ) : 
  let expression := 5 * (y - 6) + 6 * (9 - 3 * y^2 + 7 * y) - 10 * (3 * y - 2)
  ∃ a b c : ℝ, expression = a * y^2 + 17 * y + c :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_y_l855_85515


namespace NUMINAMATH_CALUDE_coin_division_theorem_l855_85561

theorem coin_division_theorem :
  let sum_20 := (20 * 21) / 2
  let sum_20_plus_100 := sum_20 + 100
  (sum_20 % 3 = 0) ∧ (sum_20_plus_100 % 3 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_coin_division_theorem_l855_85561


namespace NUMINAMATH_CALUDE_suresh_completion_time_l855_85514

theorem suresh_completion_time (ashutosh_time : ℝ) (suresh_partial_time : ℝ) (ashutosh_partial_time : ℝ) 
  (h1 : ashutosh_time = 30)
  (h2 : suresh_partial_time = 9)
  (h3 : ashutosh_partial_time = 12)
  : ∃ (suresh_time : ℝ), 
    suresh_partial_time / suresh_time + ashutosh_partial_time / ashutosh_time = 1 ∧ 
    suresh_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_suresh_completion_time_l855_85514


namespace NUMINAMATH_CALUDE_unique_consecutive_set_l855_85599

/-- Represents a set of consecutive positive integers -/
structure ConsecutiveSet where
  start : Nat
  length : Nat

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : Nat :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- A set is valid if it contains at least two integers and sums to 20 -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 2 ∧ sum_consecutive s = 20

theorem unique_consecutive_set : ∃! s : ConsecutiveSet, is_valid_set s :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_set_l855_85599


namespace NUMINAMATH_CALUDE_sum_of_solutions_l855_85564

theorem sum_of_solutions (x : ℝ) : (|3 * x - 9| = 6) → (∃ y : ℝ, (|3 * y - 9| = 6) ∧ x + y = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l855_85564


namespace NUMINAMATH_CALUDE_bakery_items_l855_85529

theorem bakery_items (total : ℕ) (bread_rolls : ℕ) (croissants : ℕ) (bagels : ℕ)
  (h1 : total = 90)
  (h2 : bread_rolls = 49)
  (h3 : croissants = 19)
  (h4 : total = bread_rolls + croissants + bagels) :
  bagels = 22 := by
sorry

end NUMINAMATH_CALUDE_bakery_items_l855_85529


namespace NUMINAMATH_CALUDE_product_expansion_sum_l855_85597

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (2*x^2 - 3*x + 5)*(8 - 3*x) = a*x^3 + b*x^2 + c*x + d) →
  9*a + 3*b + 6*c + d = -173 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l855_85597


namespace NUMINAMATH_CALUDE_count_congruent_integers_l855_85542

theorem count_congruent_integers (n : ℕ) : 
  (Finset.filter (fun x => x > 0 ∧ x < 500 ∧ x % 14 = 9) (Finset.range 500)).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_congruent_integers_l855_85542


namespace NUMINAMATH_CALUDE_right_angled_triangle_set_l855_85547

theorem right_angled_triangle_set : ∃! (a b c : ℝ), 
  ((a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3) ∨
   (a = 1 ∧ b = 2 ∧ c = 3) ∨
   (a = 2 ∧ b = 3 ∧ c = 4) ∨
   (a = Real.sqrt 2 ∧ b = 3 ∧ c = 5)) ∧
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_set_l855_85547


namespace NUMINAMATH_CALUDE_alternating_sum_2023_l855_85526

/-- Calculates the sum of the alternating series from 1 to n -/
def alternatingSum (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

/-- The sum of the series 1-2+3-4+5-6+...-2022+2023 equals 1012 -/
theorem alternating_sum_2023 :
  alternatingSum 2023 = 1012 := by
  sorry

#eval alternatingSum 2023

end NUMINAMATH_CALUDE_alternating_sum_2023_l855_85526


namespace NUMINAMATH_CALUDE_gcd_of_168_56_224_l855_85501

theorem gcd_of_168_56_224 : Nat.gcd 168 (Nat.gcd 56 224) = 56 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_168_56_224_l855_85501


namespace NUMINAMATH_CALUDE_tuesday_to_monday_work_ratio_l855_85530

theorem tuesday_to_monday_work_ratio :
  let monday : ℚ := 3/4
  let wednesday : ℚ := 2/3
  let thursday : ℚ := 5/6
  let friday : ℚ := 75/60
  let total : ℚ := 4
  let tuesday : ℚ := total - (monday + wednesday + thursday + friday)
  tuesday / monday = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_to_monday_work_ratio_l855_85530


namespace NUMINAMATH_CALUDE_scientific_notation_of_11580000_l855_85512

theorem scientific_notation_of_11580000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 11580000 = a * (10 : ℝ) ^ n ∧ a = 1.158 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_11580000_l855_85512


namespace NUMINAMATH_CALUDE_total_apples_collected_l855_85593

def apples_per_day : ℕ := 4
def days : ℕ := 30
def remaining_apples : ℕ := 230

theorem total_apples_collected :
  apples_per_day * days + remaining_apples = 350 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_collected_l855_85593


namespace NUMINAMATH_CALUDE_charms_per_necklace_is_10_l855_85595

/-- The number of charms used to make each necklace -/
def charms_per_necklace : ℕ := sorry

/-- The cost of each charm in dollars -/
def charm_cost : ℕ := 15

/-- The selling price of each necklace in dollars -/
def necklace_price : ℕ := 200

/-- The number of necklaces sold -/
def necklaces_sold : ℕ := 30

/-- The total profit in dollars -/
def total_profit : ℕ := 1500

theorem charms_per_necklace_is_10 :
  charms_per_necklace = 10 ∧
  charm_cost = 15 ∧
  necklace_price = 200 ∧
  necklaces_sold = 30 ∧
  total_profit = 1500 ∧
  necklaces_sold * (necklace_price - charms_per_necklace * charm_cost) = total_profit :=
sorry

end NUMINAMATH_CALUDE_charms_per_necklace_is_10_l855_85595


namespace NUMINAMATH_CALUDE_find_m_l855_85571

theorem find_m (A B : Set ℕ) (m : ℕ) : 
  A = {1, 2, 3} →
  B = {2, m, 4} →
  A ∩ B = {2, 3} →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_find_m_l855_85571


namespace NUMINAMATH_CALUDE_min_white_surface_3x3x3_l855_85568

/-- Represents a cube with unit cubes of different colors --/
structure ColoredCube where
  edge_length : ℕ
  total_units : ℕ
  red_units : ℕ
  white_units : ℕ

/-- Calculates the minimum white surface area fraction for a ColoredCube --/
def min_white_surface_fraction (c : ColoredCube) : ℚ :=
  sorry

/-- Theorem: For a 3x3x3 cube with 21 red and 6 white unit cubes,
    the minimum white surface area fraction is 5/54 --/
theorem min_white_surface_3x3x3 :
  let c : ColoredCube := {
    edge_length := 3,
    total_units := 27,
    red_units := 21,
    white_units := 6
  }
  min_white_surface_fraction c = 5 / 54 := by
  sorry

end NUMINAMATH_CALUDE_min_white_surface_3x3x3_l855_85568


namespace NUMINAMATH_CALUDE_cone_base_circumference_l855_85537

theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 18 * Real.pi ∧ h = 3 ∧ V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 6 * Real.sqrt 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l855_85537


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l855_85596

theorem unknown_blanket_rate (price1 price2 avg_price : ℚ) 
  (count1 count2 count_unknown : ℕ) : 
  price1 = 100 → 
  price2 = 150 → 
  avg_price = 150 → 
  count1 = 4 → 
  count2 = 5 → 
  count_unknown = 2 → 
  (count1 * price1 + count2 * price2 + count_unknown * 
    ((count1 + count2 + count_unknown) * avg_price - count1 * price1 - count2 * price2) / count_unknown) / 
    (count1 + count2 + count_unknown) = avg_price → 
  ((count1 + count2 + count_unknown) * avg_price - count1 * price1 - count2 * price2) / count_unknown = 250 :=
by sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l855_85596


namespace NUMINAMATH_CALUDE_a_4_equals_8_l855_85521

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem a_4_equals_8 (a : ℕ → ℝ) 
    (h1 : a 1 = 1)
    (h2 : ∀ (n : ℕ), a (n + 1) = 2 * a n) : 
  a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_8_l855_85521


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_l855_85538

/-- The surface area of a rectangular prism with given dimensions -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

/-- Theorem: The surface area of a rectangular prism with length 5, width 4, and height 3 is 94 -/
theorem rectangular_prism_surface_area :
  surface_area 5 4 3 = 94 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_l855_85538


namespace NUMINAMATH_CALUDE_log4_of_16_equals_2_l855_85500

-- Define the logarithm function for base 4
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

-- State the theorem
theorem log4_of_16_equals_2 : log4 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log4_of_16_equals_2_l855_85500


namespace NUMINAMATH_CALUDE_drain_time_for_specific_pumps_l855_85553

/-- Represents the time taken to drain a lake with three pumps working together -/
def drain_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating the time taken to drain a lake with three specific pumps -/
theorem drain_time_for_specific_pumps :
  drain_time (1/9) (1/6) (1/12) = 36/13 := by
  sorry

end NUMINAMATH_CALUDE_drain_time_for_specific_pumps_l855_85553


namespace NUMINAMATH_CALUDE_ratio_p_to_r_l855_85516

theorem ratio_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 3 / 5)
  (h2 : r / s = 5 / 4)
  (h3 : s / q = 1 / 3) :
  p / r = 36 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_p_to_r_l855_85516


namespace NUMINAMATH_CALUDE_number_square_problem_l855_85527

theorem number_square_problem : ∃! x : ℝ, x^2 + 64 = (x - 16)^2 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_number_square_problem_l855_85527


namespace NUMINAMATH_CALUDE_sourball_candies_count_l855_85554

/-- The number of sourball candies in the bucket initially -/
def initial_candies : ℕ := 30

/-- The number of candies Nellie can eat before crying -/
def nellie_candies : ℕ := 12

/-- The number of candies Jacob can eat before crying -/
def jacob_candies : ℕ := nellie_candies / 2

/-- The number of candies Lana can eat before crying -/
def lana_candies : ℕ := jacob_candies - 3

/-- The number of candies each person gets after division -/
def remaining_per_person : ℕ := 3

/-- The number of people -/
def num_people : ℕ := 3

theorem sourball_candies_count :
  initial_candies = nellie_candies + jacob_candies + lana_candies + remaining_per_person * num_people :=
by sorry

end NUMINAMATH_CALUDE_sourball_candies_count_l855_85554


namespace NUMINAMATH_CALUDE_four_digit_number_puzzle_l855_85519

theorem four_digit_number_puzzle :
  ∀ (A B x y : ℕ),
    1000 ≤ A ∧ A < 10000 →
    0 ≤ x ∧ x < 10 →
    0 ≤ y ∧ y < 10 →
    B = 100000 * x + 10 * A + y →
    B = 21 * A →
    A = 9091 ∧ B = 190911 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_puzzle_l855_85519


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l855_85560

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k * 120 = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) ∧
  ∀ m : ℤ, m > 120 → ∃ l : ℤ, l * m ≠ (l * (l + 1) * (l + 2) * (l + 3) * (l + 4)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l855_85560


namespace NUMINAMATH_CALUDE_sum_coordinates_of_B_l855_85567

/-- Given a point M which is the midpoint of segment AB, and the coordinates of points M and A,
    prove that the sum of the coordinates of point B is 5. -/
theorem sum_coordinates_of_B (M A B : ℝ × ℝ) : 
  M = (2, 5) →  -- M has coordinates (2, 5)
  A = (6, 3) →  -- A has coordinates (6, 3)
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  B.1 + B.2 = 5 := by  -- The sum of B's coordinates is 5
sorry


end NUMINAMATH_CALUDE_sum_coordinates_of_B_l855_85567


namespace NUMINAMATH_CALUDE_right_triangle_height_properties_l855_85586

/-- Properties of a right-angled triangle with height to hypotenuse --/
theorem right_triangle_height_properties
  (a b c h p q : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (height_divides_hypotenuse : p + q = c)
  (height_forms_similar_triangles : h^2 = a * b) :
  h^2 = p * q ∧ a^2 = p * c ∧ b^2 = q * c ∧ p / q = (a / b)^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_height_properties_l855_85586


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l855_85594

theorem fixed_point_parabola :
  ∀ (k : ℝ), 225 = 9 * (5 : ℝ)^2 + k * 5 - 5 * k := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l855_85594


namespace NUMINAMATH_CALUDE_point_in_intersection_l855_85503

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set A
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + m > 0}

-- Define set B
def B (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - n > 0}

-- Define the complement of B with respect to U
def C_U_B (n : ℝ) : Set (ℝ × ℝ) := U \ B n

-- Define point P
def P : ℝ × ℝ := (2, 3)

-- State the theorem
theorem point_in_intersection (m n : ℝ) :
  P ∈ A m ∩ C_U_B n ↔ m > -1 ∧ n ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_point_in_intersection_l855_85503


namespace NUMINAMATH_CALUDE_divisible_by_45_sum_of_digits_l855_85502

theorem divisible_by_45_sum_of_digits (a b : ℕ) : 
  a < 10 → b < 10 → (60000 + 1000 * a + 780 + b) % 45 = 0 → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_45_sum_of_digits_l855_85502


namespace NUMINAMATH_CALUDE_g_of_five_l855_85511

/-- Given a function g : ℝ → ℝ satisfying 3g(x) + 4g(1 - x) = 6x^2 for all real x, prove that g(5) = -66/7 -/
theorem g_of_five (g : ℝ → ℝ) (h : ∀ x : ℝ, 3 * g x + 4 * g (1 - x) = 6 * x^2) : g 5 = -66/7 := by
  sorry

end NUMINAMATH_CALUDE_g_of_five_l855_85511


namespace NUMINAMATH_CALUDE_fraction_division_multiplication_l855_85584

theorem fraction_division_multiplication : 
  (5 : ℚ) / 6 / (2 / 3) * (4 / 9) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_division_multiplication_l855_85584


namespace NUMINAMATH_CALUDE_train_speed_l855_85507

/-- The speed of a train given its length, the speed of a man running in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_speed (train_length : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_length = 110 →
  man_speed = 4 →
  passing_time = 9 / 3600 →
  (train_length / 1000) / passing_time - man_speed = 40 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l855_85507


namespace NUMINAMATH_CALUDE_pirate_treasure_l855_85575

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_l855_85575


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l855_85578

theorem solution_implies_a_value (a x y : ℝ) : 
  x = 2 → y = 1 → a * x - 3 * y = 1 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l855_85578


namespace NUMINAMATH_CALUDE_fraction_evaluation_l855_85510

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l855_85510


namespace NUMINAMATH_CALUDE_senior_junior_ratio_l855_85550

theorem senior_junior_ratio (S J : ℕ) (k : ℕ+) :
  S = k * J →
  (1 / 8 : ℚ) * S + (3 / 4 : ℚ) * J = (1 / 3 : ℚ) * (S + J) →
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_senior_junior_ratio_l855_85550


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l855_85570

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  -(2 * x^2 + 3 * x) + 2 * (4 * x + x^2) = -10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l855_85570


namespace NUMINAMATH_CALUDE_price_per_game_l855_85535

def playstation_cost : ℝ := 500
def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def games_to_sell : ℕ := 20

theorem price_per_game :
  (playstation_cost - (birthday_money + christmas_money)) / games_to_sell = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_price_per_game_l855_85535


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_e_l855_85533

/-- If f(x) = e^x - ax has an extremum at x = 1, then a = e -/
theorem extremum_implies_a_equals_e (a : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = Real.exp x - a * x) ∧ 
   (∃ ε > 0, ∀ h ≠ 0, |h| < ε → f (1 + h) ≤ f 1)) → 
  a = Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_extremum_implies_a_equals_e_l855_85533


namespace NUMINAMATH_CALUDE_solve_smores_problem_l855_85509

def smores_problem (graham_crackers_per_smore : ℕ) 
                   (total_graham_crackers : ℕ) 
                   (initial_marshmallows : ℕ) 
                   (additional_marshmallows : ℕ) : Prop :=
  let total_smores := total_graham_crackers / graham_crackers_per_smore
  let total_marshmallows := initial_marshmallows + additional_marshmallows
  (total_marshmallows / total_smores = 1)

theorem solve_smores_problem :
  smores_problem 2 48 6 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_smores_problem_l855_85509


namespace NUMINAMATH_CALUDE_log_range_theorem_l855_85592

-- Define the set of valid 'a' values
def validA : Set ℝ := {a | a ∈ (Set.Ioo 2 3) ∪ (Set.Ioo 3 5)}

-- Define the conditions for a meaningful logarithmic expression
def isValidLog (a : ℝ) : Prop :=
  a - 2 > 0 ∧ 5 - a > 0 ∧ a - 2 ≠ 1

-- Theorem statement
theorem log_range_theorem :
  ∀ a : ℝ, isValidLog a ↔ a ∈ validA :=
by sorry

end NUMINAMATH_CALUDE_log_range_theorem_l855_85592


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l855_85566

theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x, ax^2 + b*x + c ≥ 0 ↔ -1 ≤ x ∧ x ≤ 2) :
  a + b = 0 ∧ a + b + c > 0 ∧ c > 0 ∧ b > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l855_85566


namespace NUMINAMATH_CALUDE_alice_expected_games_l855_85522

/-- Represents a tournament with n competitors -/
structure Tournament (n : ℕ) where
  skillLevels : Fin n → ℕ
  distinctSkills : ∀ i j, i ≠ j → skillLevels i ≠ skillLevels j

/-- The expected number of games played by a competitor with a given skill level -/
noncomputable def expectedGames (t : Tournament 21) (skillLevel : ℕ) : ℚ :=
  sorry

/-- Theorem stating the expected number of games for Alice -/
theorem alice_expected_games (t : Tournament 21) (h : t.skillLevels 10 = 11) :
  expectedGames t 11 = 47 / 42 :=
sorry

end NUMINAMATH_CALUDE_alice_expected_games_l855_85522


namespace NUMINAMATH_CALUDE_train_cross_platform_time_l855_85576

def train_length : ℝ := 300
def platform_length : ℝ := 300
def time_cross_pole : ℝ := 18

theorem train_cross_platform_time :
  let train_speed := train_length / time_cross_pole
  let total_distance := train_length + platform_length
  total_distance / train_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_cross_platform_time_l855_85576


namespace NUMINAMATH_CALUDE_reema_loan_interest_l855_85543

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem reema_loan_interest :
  let principal : ℚ := 1500
  let rate : ℚ := 7
  let time : ℚ := rate
  simple_interest principal rate time = 735 := by sorry

end NUMINAMATH_CALUDE_reema_loan_interest_l855_85543


namespace NUMINAMATH_CALUDE_orange_distribution_l855_85557

/-- Given a number of oranges, calories per orange, and calories per person,
    calculate the number of people who can receive an equal share of the total calories. -/
def people_fed (num_oranges : ℕ) (calories_per_orange : ℕ) (calories_per_person : ℕ) : ℕ :=
  (num_oranges * calories_per_orange) / calories_per_person

/-- Prove that with 5 oranges, 80 calories per orange, and 100 calories per person,
    the number of people fed is 4. -/
theorem orange_distribution :
  people_fed 5 80 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l855_85557


namespace NUMINAMATH_CALUDE_towels_used_is_285_towels_used_le_total_towels_l855_85552

/-- Calculates the total number of towels used in a gym over 4 hours -/
def totalTowelsUsed (firstHourGuests : ℕ) : ℕ :=
  let secondHourGuests := firstHourGuests + (firstHourGuests * 20 / 100)
  let thirdHourGuests := secondHourGuests + (secondHourGuests * 25 / 100)
  let fourthHourGuests := thirdHourGuests + (thirdHourGuests * 1 / 3)
  firstHourGuests + secondHourGuests + thirdHourGuests + fourthHourGuests

/-- Theorem stating that the total number of towels used is 285 -/
theorem towels_used_is_285 :
  totalTowelsUsed 50 = 285 := by
  sorry

/-- The number of towels laid out daily -/
def totalTowels : ℕ := 300

/-- Theorem stating that the number of towels used is less than or equal to the total towels -/
theorem towels_used_le_total_towels :
  totalTowelsUsed 50 ≤ totalTowels := by
  sorry

end NUMINAMATH_CALUDE_towels_used_is_285_towels_used_le_total_towels_l855_85552


namespace NUMINAMATH_CALUDE_child_running_speed_l855_85549

/-- The child's running speed in meters per minute -/
def child_speed : ℝ := 74

/-- The sidewalk's speed in meters per minute -/
def sidewalk_speed : ℝ := child_speed - 55

theorem child_running_speed 
  (h1 : (child_speed + sidewalk_speed) * 4 = 372)
  (h2 : (child_speed - sidewalk_speed) * 3 = 165) :
  child_speed = 74 := by sorry

end NUMINAMATH_CALUDE_child_running_speed_l855_85549


namespace NUMINAMATH_CALUDE_gcf_of_16_and_24_l855_85591

theorem gcf_of_16_and_24 : Nat.gcd 16 24 = 8 :=
by
  have h1 : Nat.lcm 16 24 = 48 := by sorry
  sorry

end NUMINAMATH_CALUDE_gcf_of_16_and_24_l855_85591


namespace NUMINAMATH_CALUDE_cos_angle_with_z_axis_l855_85524

/-- Given a point Q in the first octant of 3D space, prove that if the cosine of the angle between OQ
    and the x-axis is 2/5, and the cosine of the angle between OQ and the y-axis is 1/4, then the
    cosine of the angle between OQ and the z-axis is √(311) / 20. -/
theorem cos_angle_with_z_axis (Q : ℝ × ℝ × ℝ) 
    (h_pos : Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0)
    (h_cos_alpha : Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = 2/5)
    (h_cos_beta : Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = 1/4) :
  Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = Real.sqrt 311 / 20 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_with_z_axis_l855_85524


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l855_85565

theorem inequality_and_equality_condition 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) 
  (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) 
  (hb₁ : b₁ > 0) (hb₂ : b₂ > 0) (hb₃ : b₃ > 0) : 
  (((a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂) : ℚ) ^ 2 ≥ 
   4 * ((a₁ * a₂ + a₂ * a₃ + a₃ * a₁) : ℚ) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁)) ∧
  (((a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂) : ℚ) ^ 2 = 
   4 * ((a₁ * a₂ + a₂ * a₃ + a₃ * a₁) : ℚ) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁) ↔ 
   (a₁ : ℚ) / b₁ = (a₂ : ℚ) / b₂ ∧ (a₂ : ℚ) / b₂ = (a₃ : ℚ) / b₃) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l855_85565


namespace NUMINAMATH_CALUDE_inequalities_proof_l855_85504

theorem inequalities_proof (a b c d : ℝ) 
  (h1 : b > a) (h2 : a > 1) (h3 : c < d) (h4 : d < -1) : 
  (1/b < 1/a ∧ 1/a < 1) ∧ 
  (1/c > 1/d ∧ 1/d > -1) ∧ 
  (a*d > b*c) := by
sorry


end NUMINAMATH_CALUDE_inequalities_proof_l855_85504


namespace NUMINAMATH_CALUDE_existence_of_incommensurable_segments_l855_85528

-- Define incommensurability
def incommensurable (x y : ℝ) : Prop :=
  ∀ k : ℚ, k ≠ 0 → x ≠ k * y

-- State the theorem
theorem existence_of_incommensurable_segments :
  ∃ (a b c d : ℝ),
    a + b + c = d ∧
    incommensurable a d ∧
    incommensurable b d ∧
    incommensurable c d :=
by sorry

end NUMINAMATH_CALUDE_existence_of_incommensurable_segments_l855_85528
