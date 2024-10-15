import Mathlib

namespace NUMINAMATH_CALUDE_profit_difference_theorem_l3544_354468

/-- Calculates the difference between profit shares of two partners given investments and one partner's profit share. -/
def profit_share_difference (invest_a invest_b invest_c b_profit : ℕ) : ℕ :=
  let total_invest := invest_a + invest_b + invest_c
  let profit_per_unit := b_profit * total_invest / invest_b
  let a_profit := profit_per_unit * invest_a / total_invest
  let c_profit := profit_per_unit * invest_c / total_invest
  c_profit - a_profit

/-- Theorem stating that given the investments and B's profit share, the difference between A's and C's profit shares is 560. -/
theorem profit_difference_theorem :
  profit_share_difference 8000 10000 12000 1400 = 560 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_theorem_l3544_354468


namespace NUMINAMATH_CALUDE_bc_plus_ce_is_one_third_of_ad_l3544_354454

-- Define the points and lengths
variable (A B C D E : ℝ)
variable (AB AC AE BD CD ED BC CE AD : ℝ)

-- State the conditions
variable (h1 : B < C)
variable (h2 : C < E)
variable (h3 : E < D)
variable (h4 : AB = 3 * BD)
variable (h5 : AC = 7 * CD)
variable (h6 : AE = 5 * ED)
variable (h7 : AD = AB + BD + CD + ED)
variable (h8 : BC = AC - AB)
variable (h9 : CE = AE - AC)

-- State the theorem
theorem bc_plus_ce_is_one_third_of_ad :
  (BC + CE) / AD = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_bc_plus_ce_is_one_third_of_ad_l3544_354454


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_function_range_l3544_354418

/-- Given a quadratic function f(x) = ax² + 2x + c with range [2,+∞), 
    prove that the minimum value of 1/a + 9/c is 4 -/
theorem min_value_of_quadratic_function_range (a c : ℝ) : 
  a > 0 → 
  (∀ x : ℝ, ax^2 + 2*x + c ≥ 2) → 
  (∃ x : ℝ, ax^2 + 2*x + c = 2) → 
  (∀ y : ℝ, 1/a + 9/c ≥ y) → 
  (∃ y : ℝ, 1/a + 9/c = y ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_function_range_l3544_354418


namespace NUMINAMATH_CALUDE_school_population_l3544_354467

theorem school_population (boys : ℕ) (girls : ℕ) : 
  (boys : ℚ) / girls = 8 / 5 → 
  boys = 128 → 
  boys + girls = 208 := by
sorry

end NUMINAMATH_CALUDE_school_population_l3544_354467


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3544_354433

theorem fraction_product_simplification :
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l3544_354433


namespace NUMINAMATH_CALUDE_deductive_reasoning_correctness_l3544_354404

/-- Represents a deductive reasoning process -/
structure DeductiveReasoning where
  premise : Prop
  form : Prop
  conclusion : Prop

/-- Represents the correctness of a component in the reasoning process -/
def isCorrect (p : Prop) : Prop := p

theorem deductive_reasoning_correctness 
  (dr : DeductiveReasoning) 
  (h_premise : isCorrect dr.premise) 
  (h_form : isCorrect dr.form) : 
  isCorrect dr.conclusion :=
sorry

end NUMINAMATH_CALUDE_deductive_reasoning_correctness_l3544_354404


namespace NUMINAMATH_CALUDE_equation_equivalence_l3544_354424

theorem equation_equivalence (a c x y : ℝ) (s t u : ℤ) : 
  (a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1)) →
  ((a^s * x - a^t) * (a^u * y - a^3) = a^5 * c^5) →
  s * t * u = 18 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3544_354424


namespace NUMINAMATH_CALUDE_H_points_infinite_but_not_all_l3544_354405

-- Define the curve C
def C : Set (ℝ × ℝ) := {p | p.1^2 / 4 + p.2^2 = 1}

-- Define the line l
def l : Set (ℝ × ℝ) := {p | p.1 = 4}

-- Define what it means to be an H point
def is_H_point (P : ℝ × ℝ) : Prop :=
  P ∈ C ∧ ∃ (A B : ℝ × ℝ) (k m : ℝ),
    A ∈ C ∧ B ∈ l ∧
    (∀ x y, y = k * x + m ↔ (x, y) ∈ ({P, A, B} : Set (ℝ × ℝ))) ∧
    (dist P A = dist P B ∨ dist P A = dist A B)

-- Define the set of H points
def H_points : Set (ℝ × ℝ) := {P | is_H_point P}

-- The theorem to be proved
theorem H_points_infinite_but_not_all :
  Set.Infinite H_points ∧ H_points ≠ C :=
sorry


end NUMINAMATH_CALUDE_H_points_infinite_but_not_all_l3544_354405


namespace NUMINAMATH_CALUDE_counterexample_37_l3544_354450

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem counterexample_37 : 
  is_prime 37 ∧ ¬(is_prime (37 - 2) ∨ is_prime (37 + 2)) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_37_l3544_354450


namespace NUMINAMATH_CALUDE_march_greatest_drop_l3544_354497

/-- Represents the months of the year --/
inductive Month
| january | february | march | april | may | june | july

/-- The price change for each month --/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.january  => -0.75
  | Month.february => 1.50
  | Month.march    => -3.00
  | Month.april    => 2.50
  | Month.may      => -1.00
  | Month.june     => 0.50
  | Month.july     => -2.50

/-- The set of months considered in the problem --/
def considered_months : List Month :=
  [Month.january, Month.february, Month.march, Month.april, Month.may, Month.june, Month.july]

/-- Predicate to check if a month has a price drop --/
def has_price_drop (m : Month) : Prop :=
  price_change m < 0

/-- The theorem stating that March had the greatest monthly drop in price --/
theorem march_greatest_drop :
  ∀ m ∈ considered_months, has_price_drop m →
    price_change Month.march ≤ price_change m :=
  sorry

end NUMINAMATH_CALUDE_march_greatest_drop_l3544_354497


namespace NUMINAMATH_CALUDE_product_of_numbers_l3544_354457

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 8) (h2 : x^2 + y^2 = 160) : x * y = 48 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3544_354457


namespace NUMINAMATH_CALUDE_bret_caught_12_frogs_l3544_354420

-- Define the number of frogs caught by each person
def alster_frogs : ℕ := 2
def quinn_frogs : ℕ := 2 * alster_frogs
def bret_frogs : ℕ := 3 * quinn_frogs

-- Theorem to prove
theorem bret_caught_12_frogs : bret_frogs = 12 := by
  sorry

end NUMINAMATH_CALUDE_bret_caught_12_frogs_l3544_354420


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3544_354400

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 96 →
  volume = (surface_area / 6) ^ (3/2) →
  volume = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3544_354400


namespace NUMINAMATH_CALUDE_water_balloon_ratio_l3544_354499

theorem water_balloon_ratio : ∀ (anthony_balloons luke_balloons tom_balloons : ℕ),
  anthony_balloons = 44 →
  luke_balloons = anthony_balloons / 4 →
  tom_balloons = 33 →
  (tom_balloons : ℚ) / luke_balloons = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_water_balloon_ratio_l3544_354499


namespace NUMINAMATH_CALUDE_sum_of_odd_numbers_less_than_100_eq_2500_l3544_354471

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def sum_of_odd_numbers_less_than_100 : ℕ := 
  (Finset.range 50).sum (λ i => 2*i + 1)

theorem sum_of_odd_numbers_less_than_100_eq_2500 : 
  sum_of_odd_numbers_less_than_100 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_numbers_less_than_100_eq_2500_l3544_354471


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3544_354488

theorem hyperbola_focal_length (m : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/m = 1) → 
  (∃ c : ℝ, c = 5) →
  m = 16 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3544_354488


namespace NUMINAMATH_CALUDE_survey_respondents_l3544_354416

theorem survey_respondents : 
  ∀ (x y : ℕ), 
    x = 60 → -- Number of people who prefer brand X
    x = 3 * y → -- Ratio of preference for X to Y is 3:1
    x + y = 80 -- Total number of respondents
    := by sorry

end NUMINAMATH_CALUDE_survey_respondents_l3544_354416


namespace NUMINAMATH_CALUDE_daves_ice_cubes_l3544_354422

theorem daves_ice_cubes (original : ℕ) (made : ℕ) (total : ℕ) : 
  made = 7 → total = 9 → original + made = total → original = 2 := by
sorry

end NUMINAMATH_CALUDE_daves_ice_cubes_l3544_354422


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l3544_354410

theorem meaningful_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = 2 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l3544_354410


namespace NUMINAMATH_CALUDE_dihedral_angle_cosine_l3544_354481

/-- Regular triangular pyramid with given properties -/
structure RegularPyramid where
  base_side : ℝ
  lateral_edge : ℝ
  base_side_eq_one : base_side = 1
  lateral_edge_eq_two : lateral_edge = 2

/-- Section that divides the pyramid volume equally -/
structure EqualVolumeSection (p : RegularPyramid) where
  passes_through_AB : Bool
  divides_equally : Bool

/-- Dihedral angle between the section and the base -/
def dihedralAngle (p : RegularPyramid) (s : EqualVolumeSection p) : ℝ := sorry

theorem dihedral_angle_cosine 
  (p : RegularPyramid) 
  (s : EqualVolumeSection p) : 
  Real.cos (dihedralAngle p s) = 2 * Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_cosine_l3544_354481


namespace NUMINAMATH_CALUDE_find_number_l3544_354448

theorem find_number (x : ℝ) : 5 + 2 * (x - 3) = 15 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3544_354448


namespace NUMINAMATH_CALUDE_fruit_cost_prices_l3544_354492

/-- Represents the cost and selling prices of fruits -/
structure FruitPrices where
  appleCost : ℚ
  appleSell : ℚ
  orangeCost : ℚ
  orangeSell : ℚ
  bananaCost : ℚ
  bananaSell : ℚ

/-- Calculates the cost prices of fruits based on selling prices and profit/loss percentages -/
def calculateCostPrices (p : FruitPrices) : Prop :=
  p.appleSell = p.appleCost - (1/6 * p.appleCost) ∧
  p.orangeSell = p.orangeCost + (1/5 * p.orangeCost) ∧
  p.bananaSell = p.bananaCost

/-- Theorem stating the correct cost prices of fruits -/
theorem fruit_cost_prices :
  ∃ (p : FruitPrices),
    p.appleSell = 15 ∧
    p.orangeSell = 20 ∧
    p.bananaSell = 10 ∧
    calculateCostPrices p ∧
    p.appleCost = 18 ∧
    p.orangeCost = 100/6 ∧
    p.bananaCost = 10 :=
  sorry

end NUMINAMATH_CALUDE_fruit_cost_prices_l3544_354492


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3544_354470

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmeticSequence a) 
  (h_sum : a 1 + a 13 = 10) : 
  a 3 + a 5 + a 7 + a 9 + a 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3544_354470


namespace NUMINAMATH_CALUDE_triangle_problem_l3544_354413

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.a * Real.sin t.B - Real.sqrt 3 * t.b * Real.cos t.A = 0)
  (h2 : t.a = Real.sqrt 7)
  (h3 : t.b = 2) :
  t.A = Real.pi / 3 ∧ t.c = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3544_354413


namespace NUMINAMATH_CALUDE_inequality_proof_l3544_354472

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) : 
  (a^b * b^c * c^d * d^a) / (b^d * c^b * d^c * a^d) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3544_354472


namespace NUMINAMATH_CALUDE_third_degree_polynomial_theorem_l3544_354417

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- Property that |g(x)| = 18 for x = 1, 2, 3, 4, 5, 6 -/
def has_absolute_value_18 (g : ThirdDegreePolynomial) : Prop :=
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 6 → |g x| = 18

/-- Main theorem: If g is a third-degree polynomial with |g(x)| = 18 for x = 1, 2, 3, 4, 5, 6, then |g(0)| = 162 -/
theorem third_degree_polynomial_theorem (g : ThirdDegreePolynomial) 
  (h : has_absolute_value_18 g) : |g 0| = 162 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_theorem_l3544_354417


namespace NUMINAMATH_CALUDE_johns_game_percentage_l3544_354478

theorem johns_game_percentage (shots_per_foul : ℕ) (fouls_per_game : ℕ) (total_games : ℕ) (actual_shots : ℕ) :
  shots_per_foul = 2 →
  fouls_per_game = 5 →
  total_games = 20 →
  actual_shots = 112 →
  (actual_shots : ℚ) / ((shots_per_foul * fouls_per_game * total_games) : ℚ) * 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_johns_game_percentage_l3544_354478


namespace NUMINAMATH_CALUDE_present_age_ratio_l3544_354443

/-- Given Suji's present age and the future ratio of ages, find the present ratio of ages --/
theorem present_age_ratio (suji_age : ℕ) (future_ratio_abi : ℕ) (future_ratio_suji : ℕ) :
  suji_age = 24 →
  (future_ratio_abi : ℚ) / future_ratio_suji = 11 / 9 →
  ∃ (abi_age : ℕ),
    (abi_age + 3 : ℚ) / (suji_age + 3) = (future_ratio_abi : ℚ) / future_ratio_suji ∧
    (abi_age : ℚ) / suji_age = 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_present_age_ratio_l3544_354443


namespace NUMINAMATH_CALUDE_max_cabbages_is_256_l3544_354491

structure Region where
  area : ℕ
  sunlight : ℕ
  water : ℕ

def is_suitable (r : Region) : Bool :=
  r.sunlight ≥ 4 ∧ r.water ≤ 16

def count_cabbages (regions : List Region) : ℕ :=
  (regions.filter is_suitable).foldl (fun acc r => acc + r.area) 0

def garden : List Region :=
  [
    ⟨30, 5, 15⟩,
    ⟨25, 6, 12⟩,
    ⟨35, 8, 18⟩,
    ⟨40, 4, 10⟩,
    ⟨20, 7, 14⟩
  ]

theorem max_cabbages_is_256 :
  count_cabbages garden + 181 = 256 :=
by sorry

end NUMINAMATH_CALUDE_max_cabbages_is_256_l3544_354491


namespace NUMINAMATH_CALUDE_contrapositive_example_l3544_354465

theorem contrapositive_example (a b : ℝ) :
  (¬(a = 0 → a * b = 0) ↔ (a * b ≠ 0 → a ≠ 0)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l3544_354465


namespace NUMINAMATH_CALUDE_square_root_sum_l3544_354482

theorem square_root_sum (x : ℝ) (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) :
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_l3544_354482


namespace NUMINAMATH_CALUDE_action_figure_collection_l3544_354414

/-- The problem of calculating the total number of action figures needed for a complete collection. -/
theorem action_figure_collection
  (jerry_has : ℕ)
  (cost_per_figure : ℕ)
  (total_cost_to_finish : ℕ)
  (h1 : jerry_has = 7)
  (h2 : cost_per_figure = 8)
  (h3 : total_cost_to_finish = 72) :
  jerry_has + total_cost_to_finish / cost_per_figure = 16 :=
by sorry

end NUMINAMATH_CALUDE_action_figure_collection_l3544_354414


namespace NUMINAMATH_CALUDE_joans_remaining_books_l3544_354453

/-- Calculates the number of remaining books after a sale. -/
def remaining_books (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

/-- Theorem stating that Joan's remaining books is 7. -/
theorem joans_remaining_books :
  remaining_books 33 26 = 7 := by
  sorry

end NUMINAMATH_CALUDE_joans_remaining_books_l3544_354453


namespace NUMINAMATH_CALUDE_trivia_game_total_points_l3544_354473

/-- Given the points scored by three teams in a trivia game, prove that the total points scored is 15. -/
theorem trivia_game_total_points (team_a team_b team_c : ℕ) 
  (h1 : team_a = 2) 
  (h2 : team_b = 9) 
  (h3 : team_c = 4) : 
  team_a + team_b + team_c = 15 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_total_points_l3544_354473


namespace NUMINAMATH_CALUDE_square_sum_eq_243_l3544_354494

theorem square_sum_eq_243 (x y : ℝ) (h1 : x + 3 * y = 9) (h2 : x * y = -27) :
  x^2 + 9 * y^2 = 243 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_eq_243_l3544_354494


namespace NUMINAMATH_CALUDE_discount_calculation_l3544_354477

/-- The original price of a shirt before discount -/
def original_price : ℚ := 746.68

/-- The discounted price of the shirt -/
def discounted_price : ℚ := 560

/-- The discount rate applied to the shirt -/
def discount_rate : ℚ := 0.25

/-- Theorem stating that the discounted price is equal to the original price minus the discount -/
theorem discount_calculation (original : ℚ) (discount : ℚ) (discounted : ℚ) :
  original = discounted_price ∧ discount = discount_rate ∧ discounted = original * (1 - discount) →
  original = original_price :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l3544_354477


namespace NUMINAMATH_CALUDE_sequence_properties_l3544_354466

/-- Sequence a_n with sum S_n = n^2 + pn -/
def S (n : ℕ) (p : ℝ) : ℝ := n^2 + p * n

/-- Sequence b_n with sum T_n = 3n^2 - 2n -/
def T (n : ℕ) : ℝ := 3 * n^2 - 2 * n

/-- a_n is the difference of consecutive S_n terms -/
def a (n : ℕ) (p : ℝ) : ℝ := S n p - S (n-1) p

/-- b_n is the difference of consecutive T_n terms -/
def b (n : ℕ) : ℝ := T n - T (n-1)

/-- c_n is the sequence formed by odd-indexed terms of b_n -/
def c (n : ℕ) : ℝ := b (2*n - 1)

theorem sequence_properties (p : ℝ) :
  (a 10 p = b 10) → p = 36 ∧ ∀ n, c n = 12 * n - 11 := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3544_354466


namespace NUMINAMATH_CALUDE_estimated_probability_is_two_ninths_l3544_354421

/-- Represents the outcome of a single trial -/
inductive Outcome
| StopOnThird
| Other

/-- Represents the result of a random simulation -/
structure SimulationResult :=
  (trials : Nat)
  (stopsOnThird : Nat)

/-- Calculates the estimated probability -/
def estimateProbability (result : SimulationResult) : Rat :=
  result.stopsOnThird / result.trials

theorem estimated_probability_is_two_ninths 
  (result : SimulationResult)
  (h1 : result.trials = 18)
  (h2 : result.stopsOnThird = 4) :
  estimateProbability result = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_estimated_probability_is_two_ninths_l3544_354421


namespace NUMINAMATH_CALUDE_triangle_cos_2C_l3544_354451

theorem triangle_cos_2C (a b : ℝ) (S_ABC : ℝ) (C : ℝ) :
  a = 8 →
  b = 5 →
  S_ABC = 12 →
  S_ABC = 1/2 * a * b * Real.sin C →
  Real.cos (2 * C) = 7/25 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_cos_2C_l3544_354451


namespace NUMINAMATH_CALUDE_shop_length_calculation_l3544_354493

/-- Given a shop with specified dimensions and rent, calculate its length -/
theorem shop_length_calculation (width : ℝ) (monthly_rent : ℝ) (annual_rent_per_sqft : ℝ) :
  width = 20 →
  monthly_rent = 3600 →
  annual_rent_per_sqft = 120 →
  (monthly_rent * 12) / (width * annual_rent_per_sqft) = 18 :=
by sorry

end NUMINAMATH_CALUDE_shop_length_calculation_l3544_354493


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_512_l3544_354475

theorem sqrt_expression_equals_512 : 
  Real.sqrt ((16^12 + 2^24) / (16^5 + 2^30)) = 512 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_512_l3544_354475


namespace NUMINAMATH_CALUDE_derivative_from_limit_l3544_354441

theorem derivative_from_limit (f : ℝ → ℝ) (x₀ : ℝ) (h : Differentiable ℝ f) :
  (∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |(f (x₀ - 2*Δx) - f x₀) / Δx - 2| < ε) →
  deriv f x₀ = -1 := by sorry

end NUMINAMATH_CALUDE_derivative_from_limit_l3544_354441


namespace NUMINAMATH_CALUDE_complex_power_equality_l3544_354444

theorem complex_power_equality (n : ℕ) (hn : n ≤ 1000) :
  ∀ t : ℝ, (Complex.cos t - Complex.I * Complex.sin t) ^ n = Complex.cos (n * t) - Complex.I * Complex.sin (n * t) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_equality_l3544_354444


namespace NUMINAMATH_CALUDE_xy_value_l3544_354402

theorem xy_value (x y : ℝ) : |x - y + 6| + (y + 8)^2 = 0 → x * y = 112 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3544_354402


namespace NUMINAMATH_CALUDE_total_rainfall_2004_l3544_354435

/-- The average monthly rainfall in Mathborough in 2003 (in mm) -/
def rainfall_2003 : ℝ := 41.5

/-- The increase in average monthly rainfall from 2003 to 2004 (in mm) -/
def rainfall_increase : ℝ := 2

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: The total rainfall in Mathborough in 2004 was 522 mm -/
theorem total_rainfall_2004 : 
  (rainfall_2003 + rainfall_increase) * months_in_year = 522 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_2004_l3544_354435


namespace NUMINAMATH_CALUDE_number_whose_quarter_is_nine_more_l3544_354425

theorem number_whose_quarter_is_nine_more (x : ℚ) : (x / 4 = x + 9) → x = -12 := by
  sorry

end NUMINAMATH_CALUDE_number_whose_quarter_is_nine_more_l3544_354425


namespace NUMINAMATH_CALUDE_min_value_theorem_l3544_354490

/-- Two circles C₁ and C₂ with given equations -/
def C₁ (x y a : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 9 = 0
def C₂ (x y b : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The condition that the circles have only one common tangent line -/
def one_common_tangent (a b : ℝ) : Prop := sorry

theorem min_value_theorem (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : one_common_tangent a b) :
  (∀ x y, C₁ x y a → C₂ x y b → 4/a^2 + 1/b^2 ≥ 4) ∧ 
  (∃ x y, C₁ x y a ∧ C₂ x y b ∧ 4/a^2 + 1/b^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3544_354490


namespace NUMINAMATH_CALUDE_handshakes_in_gathering_l3544_354463

/-- The number of handshakes in a gathering of couples with specific rules -/
theorem handshakes_in_gathering (n : ℕ) (h : n = 6) : 
  (2 * n) * (2 * n - 3) / 2 = 54 := by sorry

end NUMINAMATH_CALUDE_handshakes_in_gathering_l3544_354463


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3544_354434

theorem polynomial_factorization (x y : ℝ) : 
  2 * x^2 * y - 4 * x * y^2 + 2 * y^3 = 2 * y * (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3544_354434


namespace NUMINAMATH_CALUDE_equation_solution_l3544_354474

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3

-- Theorem statement
theorem equation_solution :
  ∃ (x : ℝ), x > 9 ∧ equation x ∧ x = 21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3544_354474


namespace NUMINAMATH_CALUDE_distance_to_school_l3544_354407

theorem distance_to_school (normal_time normal_speed light_time : ℚ) 
  (h1 : normal_time = 20 / 60)
  (h2 : light_time = 10 / 60)
  (h3 : normal_time * normal_speed = light_time * (normal_speed + 15)) :
  normal_time * normal_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_school_l3544_354407


namespace NUMINAMATH_CALUDE_power_multiplication_equality_l3544_354446

theorem power_multiplication_equality : (512 : ℝ)^(2/3) * 8 = 512 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_equality_l3544_354446


namespace NUMINAMATH_CALUDE_pythagorean_theorem_3d_l3544_354452

/-- The Pythagorean theorem extended to a rectangular solid -/
theorem pythagorean_theorem_3d (p q r d : ℝ) 
  (h : d > 0) 
  (h_diagonal : d = Real.sqrt (p^2 + q^2 + r^2)) : 
  p^2 + q^2 + r^2 = d^2 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_3d_l3544_354452


namespace NUMINAMATH_CALUDE_monkey_family_size_l3544_354436

/-- The number of monkeys in a family that collected bananas -/
def number_of_monkeys : ℕ := by sorry

theorem monkey_family_size :
  let total_piles : ℕ := 10
  let piles_type1 : ℕ := 6
  let hands_per_pile_type1 : ℕ := 9
  let bananas_per_hand_type1 : ℕ := 14
  let piles_type2 : ℕ := total_piles - piles_type1
  let hands_per_pile_type2 : ℕ := 12
  let bananas_per_hand_type2 : ℕ := 9
  let bananas_per_monkey : ℕ := 99

  let total_bananas : ℕ := 
    piles_type1 * hands_per_pile_type1 * bananas_per_hand_type1 +
    piles_type2 * hands_per_pile_type2 * bananas_per_hand_type2

  number_of_monkeys = total_bananas / bananas_per_monkey := by sorry

end NUMINAMATH_CALUDE_monkey_family_size_l3544_354436


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l3544_354431

/-- The sum of interior angles of a regular polygon with exterior angles of 20 degrees -/
theorem sum_interior_angles_regular_polygon (n : ℕ) (h : n * 20 = 360) : 
  (n - 2) * 180 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l3544_354431


namespace NUMINAMATH_CALUDE_triangle_inequality_l3544_354498

/-- Given a triangle with side lengths a, b, and c, the expression
    a^2 b(a-b) + b^2 c(b-c) + c^2 a(c-a) is non-negative,
    with equality if and only if the triangle is equilateral. -/
theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3544_354498


namespace NUMINAMATH_CALUDE_production_days_l3544_354406

theorem production_days (n : ℕ) 
  (h1 : (n * 40 + 90) / (n + 1) = 45) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l3544_354406


namespace NUMINAMATH_CALUDE_expected_knowers_value_l3544_354489

/-- The number of scientists at the conference -/
def total_scientists : ℕ := 18

/-- The number of scientists who initially know the news -/
def initial_knowers : ℕ := 10

/-- The probability that an initially unknowing scientist learns the news during the coffee break -/
def prob_learn : ℚ := 10 / 17

/-- The expected number of scientists who know the news after the coffee break -/
def expected_knowers : ℚ := initial_knowers + (total_scientists - initial_knowers) * prob_learn

theorem expected_knowers_value : expected_knowers = 248 / 17 := by sorry

end NUMINAMATH_CALUDE_expected_knowers_value_l3544_354489


namespace NUMINAMATH_CALUDE_evaluate_expression_l3544_354496

theorem evaluate_expression : -(18 / 3 * 8 - 70 + 5 * 7) = -13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3544_354496


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3544_354460

theorem quadratic_inequality_solution_set (c : ℝ) (hc : c > 1) :
  {x : ℝ | x^2 - (c + 1/c)*x + 1 > 0} = {x : ℝ | x < 1/c ∨ x > c} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3544_354460


namespace NUMINAMATH_CALUDE_min_value_problem_l3544_354415

theorem min_value_problem (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hab : a * b = 1/4) :
  (∀ x y : ℝ, 0 < x ∧ x < 1 → 0 < y ∧ y < 1 → x * y = 1/4 →
    1 / (1 - x) + 2 / (1 - y) ≥ 4 + 4 * Real.sqrt 2 / 3) ∧
  (∃ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x * y = 1/4 ∧
    1 / (1 - x) + 2 / (1 - y) = 4 + 4 * Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l3544_354415


namespace NUMINAMATH_CALUDE_bob_monthly_hours_l3544_354423

/-- Calculates the total hours worked in a month given daily hours, workdays per week, and average weeks per month. -/
def total_monthly_hours (daily_hours : ℝ) (workdays_per_week : ℝ) (avg_weeks_per_month : ℝ) : ℝ :=
  daily_hours * workdays_per_week * avg_weeks_per_month

/-- Proves that Bob's total monthly hours are approximately 216.5 -/
theorem bob_monthly_hours :
  let daily_hours : ℝ := 10
  let workdays_per_week : ℝ := 5
  let avg_weeks_per_month : ℝ := 4.33
  abs (total_monthly_hours daily_hours workdays_per_week avg_weeks_per_month - 216.5) < 0.1 := by
  sorry

#eval total_monthly_hours 10 5 4.33

end NUMINAMATH_CALUDE_bob_monthly_hours_l3544_354423


namespace NUMINAMATH_CALUDE_fermat_number_properties_l3544_354456

/-- Fermat number F_n -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Main theorem -/
theorem fermat_number_properties (n : ℕ) (p : ℕ) (h_n : n ≥ 2) (h_p : Nat.Prime p) (h_factor : p ∣ F n) :
  (∃ x : ℤ, x^2 ≡ 2 [ZMOD p]) ∧ p ≡ 1 [ZMOD 2^(n+2)] := by sorry

end NUMINAMATH_CALUDE_fermat_number_properties_l3544_354456


namespace NUMINAMATH_CALUDE_quadratic_sum_l3544_354442

theorem quadratic_sum (x : ℝ) : ∃ (a h k : ℝ),
  (3 * x^2 - 6 * x - 2 = a * (x - h)^2 + k) ∧ (a + h + k = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3544_354442


namespace NUMINAMATH_CALUDE_max_balls_l3544_354483

theorem max_balls (n : ℕ) : 
  (∃ r : ℕ, r ≤ n ∧ 
    (r ≥ 49 ∧ r ≤ 50) ∧ 
    (∀ k : ℕ, k > 0 → (7 * k ≤ r - 49) ∧ (r - 49 < 8 * k)) ∧
    (10 * r ≥ 9 * n)) → 
  n ≤ 210 :=
sorry

end NUMINAMATH_CALUDE_max_balls_l3544_354483


namespace NUMINAMATH_CALUDE_reading_difference_l3544_354449

-- Define the reading rates in pages per hour
def dustin_rate : ℚ := 75
def sam_rate : ℚ := 24

-- Define the time in hours (40 minutes = 2/3 hour)
def reading_time : ℚ := 2/3

-- Define the function to calculate pages read given rate and time
def pages_read (rate : ℚ) (time : ℚ) : ℚ := rate * time

-- Theorem statement
theorem reading_difference :
  pages_read dustin_rate reading_time - pages_read sam_rate reading_time = 34 := by
  sorry

end NUMINAMATH_CALUDE_reading_difference_l3544_354449


namespace NUMINAMATH_CALUDE_a_2016_div_2017_l3544_354486

/-- The sequence a defined by the given recurrence relation -/
def a : ℕ → ℤ
  | 0 => 0
  | 1 => 2
  | (n + 2) => 2 * a (n + 1) + 41 * a n

/-- The theorem stating that the 2016th term of the sequence is divisible by 2017 -/
theorem a_2016_div_2017 : 2017 ∣ a 2016 := by
  sorry

end NUMINAMATH_CALUDE_a_2016_div_2017_l3544_354486


namespace NUMINAMATH_CALUDE_acute_angle_vector_range_l3544_354464

/-- The range of k for acute angle between vectors (2, 1) and (1, k) -/
theorem acute_angle_vector_range :
  ∀ k : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (1, k)
  -- Acute angle condition
  (0 < (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) →
  -- Non-parallel condition
  (a.1 / a.2 ≠ b.1 / b.2) →
  -- Range of k
  (k > -2 ∧ k ≠ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_acute_angle_vector_range_l3544_354464


namespace NUMINAMATH_CALUDE_selection_problem_l3544_354458

theorem selection_problem (n_boys m_boys n_girls m_girls : ℕ) 
  (h1 : n_boys = 5) (h2 : m_boys = 3) (h3 : n_girls = 4) (h4 : m_girls = 2) : 
  (Nat.choose n_boys m_boys) * (Nat.choose n_girls m_girls) = 
  (Nat.choose 5 3) * (Nat.choose 4 2) := by
  sorry

end NUMINAMATH_CALUDE_selection_problem_l3544_354458


namespace NUMINAMATH_CALUDE_more_ones_than_zeros_mod_500_l3544_354411

/-- The number of positive integers less than or equal to 500 whose binary 
    representation contains more 1's than 0's -/
def N : ℕ := sorry

/-- Function to count 1's in the binary representation of a natural number -/
def count_ones (n : ℕ) : ℕ := sorry

/-- Function to count 0's in the binary representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem more_ones_than_zeros_mod_500 :
  N % 500 = 305 :=
sorry

end NUMINAMATH_CALUDE_more_ones_than_zeros_mod_500_l3544_354411


namespace NUMINAMATH_CALUDE_largest_divisible_by_8_l3544_354430

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def last_three_digits (n : ℕ) : ℕ := n % 1000

def number_format (a : ℕ) : ℕ := 365000 + a * 100 + 20

theorem largest_divisible_by_8 :
  ∀ a : ℕ, a ≤ 9 →
    is_divisible_by_8 (number_format 9) ∧
    (is_divisible_by_8 (number_format a) → a ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_by_8_l3544_354430


namespace NUMINAMATH_CALUDE_set_existence_condition_l3544_354447

theorem set_existence_condition (r : ℝ) (hr : 0 < r ∧ r < 1) :
  (∃ S : Set ℝ, 
    (∀ t : ℝ, (t ∈ S ∨ (t + r) ∈ S ∨ (t + 1) ∈ S) ∧
              (t ∉ S ∨ (t + r) ∉ S) ∧ ((t + r) ∉ S ∨ (t + 1) ∉ S) ∧ (t ∉ S ∨ (t + 1) ∉ S)) ∧
    (∀ t : ℝ, (t ∈ S ∨ (t - r) ∈ S ∨ (t - 1) ∈ S) ∧
              (t ∉ S ∨ (t - r) ∉ S) ∧ ((t - r) ∉ S ∨ (t - 1) ∉ S) ∧ (t ∉ S ∨ (t - 1) ∉ S))) ↔
  (¬ ∃ (a b : ℤ), r = (a : ℝ) / (b : ℝ)) ∨
  (∃ (a b : ℤ), r = (a : ℝ) / (b : ℝ) ∧ 3 ∣ (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_set_existence_condition_l3544_354447


namespace NUMINAMATH_CALUDE_not_perfect_square_floor_sqrt_l3544_354484

theorem not_perfect_square_floor_sqrt (A : ℕ) (h : ∀ k : ℕ, k * k ≠ A) :
  ∃ n : ℕ, A = ⌊(n : ℝ) + Real.sqrt n + 1/2⌋ :=
sorry

end NUMINAMATH_CALUDE_not_perfect_square_floor_sqrt_l3544_354484


namespace NUMINAMATH_CALUDE_school_ratio_problem_l3544_354440

theorem school_ratio_problem (S T : ℕ) : 
  S / T = 50 →
  (S + 50) / (T + 5) = 25 →
  T = 3 := by
sorry

end NUMINAMATH_CALUDE_school_ratio_problem_l3544_354440


namespace NUMINAMATH_CALUDE_smithtown_population_ratio_l3544_354432

/-- Represents the population of Smithtown -/
structure Population where
  total : ℝ
  rightHanded : ℝ
  leftHanded : ℝ
  men : ℝ
  women : ℝ
  leftHandedWomen : ℝ

/-- The conditions given in the problem -/
def populationConditions (p : Population) : Prop :=
  p.rightHanded / p.leftHanded = 3 ∧
  p.leftHandedWomen / p.total = 0.2500000000000001 ∧
  p.rightHanded = p.men

/-- The theorem to be proved -/
theorem smithtown_population_ratio
  (p : Population)
  (h : populationConditions p) :
  p.men / p.women = 3 := by
  sorry

end NUMINAMATH_CALUDE_smithtown_population_ratio_l3544_354432


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l3544_354469

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 4 / 3)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 6) :
  w / y = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l3544_354469


namespace NUMINAMATH_CALUDE_min_value_of_function_l3544_354427

theorem min_value_of_function (m n : ℝ) : 
  m > 0 → n > 0 →  -- point in first quadrant
  ∃ (a b : ℝ), (m + a) / 2 + (n + b) / 2 - 2 = 0 →  -- symmetry condition
  2 * a + b + 3 = 0 →  -- (a,b) lies on 2x+y+3=0
  (n - b) / (m - a) = 1 →  -- slope of line of symmetry
  2 * m + n + 3 = 0 →  -- (m,n) lies on 2x+y+3=0
  (1 / m + 8 / n) ≥ 25 / 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3544_354427


namespace NUMINAMATH_CALUDE_expression_value_l3544_354485

theorem expression_value (x y : ℤ) (hx : x = -2) (hy : y = 3) :
  (x + 2*y)^2 - (x + y)*(2*x - y) = 23 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3544_354485


namespace NUMINAMATH_CALUDE_no_integer_solution_a_l3544_354445

theorem no_integer_solution_a (x y : ℤ) : x^2 + y^2 ≠ 2003 := by
  sorry

#check no_integer_solution_a

end NUMINAMATH_CALUDE_no_integer_solution_a_l3544_354445


namespace NUMINAMATH_CALUDE_vector_equation_m_range_l3544_354428

theorem vector_equation_m_range :
  ∀ (m n x : ℝ),
  (∃ x, (n + 2, n - Real.cos x ^ 2) = (2 * m, m + Real.sin x)) →
  (∀ m', (∃ n' x', (n' + 2, n' - Real.cos x' ^ 2) = (2 * m', m' + Real.sin x')) → 
    0 ≤ m' ∧ m' ≤ 4) ∧
  (∃ n₁ x₁, (n₁ + 2, n₁ - Real.cos x₁ ^ 2) = (2 * 0, 0 + Real.sin x₁)) ∧
  (∃ n₂ x₂, (n₂ + 2, n₂ - Real.cos x₂ ^ 2) = (2 * 4, 4 + Real.sin x₂)) :=
by sorry

end NUMINAMATH_CALUDE_vector_equation_m_range_l3544_354428


namespace NUMINAMATH_CALUDE_equation_solution_l3544_354412

theorem equation_solution : ∃! x : ℝ, (1/4 : ℝ)^(4*x + 12) = 16^(x + 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3544_354412


namespace NUMINAMATH_CALUDE_square_difference_equals_one_l3544_354487

theorem square_difference_equals_one (a b : ℝ) (h : a - b = 1) :
  a^2 - b^2 - 2*b = 1 := by sorry

end NUMINAMATH_CALUDE_square_difference_equals_one_l3544_354487


namespace NUMINAMATH_CALUDE_equal_numbers_after_operations_l3544_354459

theorem equal_numbers_after_operations : ∃ (x a b : ℝ), 
  x > 0 ∧ a > 0 ∧ b > 0 ∧
  96 / a = x ∧
  28 - b = x ∧
  20 + b = x ∧
  6 * a = x := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_after_operations_l3544_354459


namespace NUMINAMATH_CALUDE_angle_measure_when_supplement_is_four_times_complement_l3544_354409

theorem angle_measure_when_supplement_is_four_times_complement :
  ∀ x : ℝ,
  (0 < x) →
  (x < 180) →
  (180 - x = 4 * (90 - x)) →
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_when_supplement_is_four_times_complement_l3544_354409


namespace NUMINAMATH_CALUDE_soccer_balls_count_l3544_354438

theorem soccer_balls_count (soccer : ℕ) (baseball : ℕ) (volleyball : ℕ) : 
  baseball = 5 * soccer →
  volleyball = 3 * soccer →
  baseball + volleyball = 160 →
  soccer = 20 :=
by sorry

end NUMINAMATH_CALUDE_soccer_balls_count_l3544_354438


namespace NUMINAMATH_CALUDE_BA_equals_AB_l3544_354480

variable {α : Type*} [CommRing α]

def matrix_eq (A B : Matrix (Fin 2) (Fin 2) α) : Prop :=
  ∀ i j, A i j = B i j

theorem BA_equals_AB (A B : Matrix (Fin 2) (Fin 2) α) 
  (h1 : A + B = A * B)
  (h2 : matrix_eq (A * B) !![5, 2; -2, 4]) :
  matrix_eq (B * A) !![5, 2; -2, 4] := by
  sorry

end NUMINAMATH_CALUDE_BA_equals_AB_l3544_354480


namespace NUMINAMATH_CALUDE_probability_gpa_at_least_3_6_l3544_354476

/-- Grade points for each letter grade -/
def gradePoints (grade : Char) : ℕ :=
  match grade with
  | 'A' => 4
  | 'B' => 3
  | 'C' => 2
  | 'D' => 1
  | _ => 0

/-- Calculate GPA given a list of grades -/
def calculateGPA (grades : List Char) : ℚ :=
  (grades.map gradePoints).sum / 5

/-- Probability of getting an A in English -/
def pEnglishA : ℚ := 1/4

/-- Probability of getting a B in English -/
def pEnglishB : ℚ := 1/2

/-- Probability of getting an A in History -/
def pHistoryA : ℚ := 2/5

/-- Probability of getting a B in History -/
def pHistoryB : ℚ := 1/2

/-- Theorem stating the probability of achieving a GPA of at least 3.6 -/
theorem probability_gpa_at_least_3_6 :
  let p := pEnglishA * pHistoryA + pEnglishA * pHistoryB + pEnglishB * pHistoryA
  p = 17/40 := by sorry

end NUMINAMATH_CALUDE_probability_gpa_at_least_3_6_l3544_354476


namespace NUMINAMATH_CALUDE_mat_weaving_problem_l3544_354426

/-- Given that 4 mat-weaves can weave 4 mats in 4 days, 
    prove that 8 mat-weaves will weave 16 mats in 8 days. -/
theorem mat_weaving_problem (weave_rate : ℕ → ℕ → ℕ → ℕ) :
  weave_rate 4 4 4 = 4 →
  weave_rate 8 16 8 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_mat_weaving_problem_l3544_354426


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3544_354455

def U : Set ℕ := {x : ℕ | (x + 1 : ℚ) / (x - 5 : ℚ) ≤ 0}

def A : Set ℕ := {1, 2, 4}

theorem complement_of_A_in_U : 
  (U \ A) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3544_354455


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l3544_354479

def total_players : ℕ := 15
def quadruplets : ℕ := 4
def starters : ℕ := 6
def max_quadruplets_in_lineup : ℕ := 2

theorem basketball_lineup_combinations : 
  (Nat.choose (total_players - quadruplets) starters) + 
  (quadruplets * Nat.choose (total_players - quadruplets) (starters - 1)) + 
  (Nat.choose quadruplets 2 * Nat.choose (total_players - quadruplets) (starters - 2)) = 4290 :=
sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l3544_354479


namespace NUMINAMATH_CALUDE_one_more_bird_than_storks_l3544_354408

/-- Given a fence with birds and storks, calculate the difference between the number of birds and storks -/
def bird_stork_difference (num_birds : ℕ) (num_storks : ℕ) : ℤ :=
  (num_birds : ℤ) - (num_storks : ℤ)

/-- Theorem: On a fence with 6 birds and 5 storks, there is 1 more bird than storks -/
theorem one_more_bird_than_storks :
  bird_stork_difference 6 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_more_bird_than_storks_l3544_354408


namespace NUMINAMATH_CALUDE_tan_sum_ratio_equals_neg_sqrt_three_over_three_l3544_354495

theorem tan_sum_ratio_equals_neg_sqrt_three_over_three : 
  (Real.tan (10 * π / 180) + Real.tan (20 * π / 180) + Real.tan (150 * π / 180)) / 
  (Real.tan (10 * π / 180) * Real.tan (20 * π / 180)) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_ratio_equals_neg_sqrt_three_over_three_l3544_354495


namespace NUMINAMATH_CALUDE_chord_inscribed_squares_side_difference_l3544_354429

/-- Given a circle with radius r and a chord at distance h from the center,
    prove that the difference in side lengths of two squares inscribed in the segments
    formed by the chord is 8h/5. -/
theorem chord_inscribed_squares_side_difference
  (r h : ℝ) (hr : r > 0) (hh : 0 < h ∧ h < r) :
  ∃ (a b : ℝ),
    (a > 0 ∧ b > 0) ∧
    (a - h)^2 = r^2 - (a^2 / 4) ∧
    (b + h)^2 = r^2 - (b^2 / 4) ∧
    b - a = (8 * h) / 5 :=
sorry

end NUMINAMATH_CALUDE_chord_inscribed_squares_side_difference_l3544_354429


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3544_354462

-- Define the sets M and N
def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x | x < 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3544_354462


namespace NUMINAMATH_CALUDE_brothers_ages_product_l3544_354461

theorem brothers_ages_product (O Y : ℕ) 
  (h1 : O > Y)
  (h2 : O - Y = 12)
  (h3 : O + Y = (O - Y) + 40) : 
  O * Y = 640 := by
  sorry

end NUMINAMATH_CALUDE_brothers_ages_product_l3544_354461


namespace NUMINAMATH_CALUDE_triangle_theorem_l3544_354437

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle)
  (h1 : 3 * (t.b^2 + t.c^2) = 3 * t.a^2 + 2 * t.b * t.c) :
  (∀ (h2 : Real.sin t.B = Real.sqrt 2 * Real.cos t.C),
    Real.tan t.C = Real.sqrt 2) ∧
  (∀ (h3 : t.a = 2)
     (h4 : (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 2 / 2)
     (h5 : t.b > t.c),
    t.b = 3 * Real.sqrt 2 / 2 ∧ t.c = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3544_354437


namespace NUMINAMATH_CALUDE_finite_fun_primes_l3544_354403

/-- A prime p is fun with respect to positive integers a and b if there exists a positive integer n
    satisfying the given conditions. -/
def IsFunPrime (p a b : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ 
    p.Prime ∧
    (p ∣ a^(n.factorial) + b) ∧
    (p ∣ a^((n+1).factorial) + b) ∧
    (p < 2*n^2 + 1)

/-- The set of fun primes for given positive integers a and b is finite. -/
theorem finite_fun_primes (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  {p : ℕ | IsFunPrime p a b}.Finite :=
sorry

end NUMINAMATH_CALUDE_finite_fun_primes_l3544_354403


namespace NUMINAMATH_CALUDE_vector_decomposition_l3544_354419

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![(-15), 5, 6]
def p : Fin 3 → ℝ := ![0, 5, 1]
def q : Fin 3 → ℝ := ![3, 2, (-1)]
def r : Fin 3 → ℝ := ![(-1), 1, 0]

/-- The theorem to be proved -/
theorem vector_decomposition :
  x = (2 : ℝ) • p + (-4 : ℝ) • q + (3 : ℝ) • r := by
  sorry

end NUMINAMATH_CALUDE_vector_decomposition_l3544_354419


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_inequality_l3544_354439

theorem cubic_polynomial_root_inequality (A B C : ℝ) (α β γ : ℂ) 
  (h : ∀ x : ℂ, x^3 + A*x^2 + B*x + C = 0 ↔ x = α ∨ x = β ∨ x = γ) :
  (1 + |A| + |B| + |C|) / (Complex.abs α + Complex.abs β + Complex.abs γ) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_inequality_l3544_354439


namespace NUMINAMATH_CALUDE_total_lives_after_third_level_l3544_354401

-- Define the game parameters
def initial_lives : ℕ := 2
def enemies_defeated : ℕ := 5
def powerups_collected : ℕ := 4
def first_level_penalty : ℕ := 3
def second_level_modifier (x : ℕ) : ℕ := x / 2

-- Define the game rules
def first_level_lives (x : ℕ) : ℕ := initial_lives + 2 * x - first_level_penalty

def second_level_lives (first_level : ℕ) (y : ℕ) : ℕ :=
  first_level + 3 * y - second_level_modifier first_level

def third_level_bonus (x y : ℕ) : ℕ := x + 2 * y - 5

-- The main theorem
theorem total_lives_after_third_level :
  let first_level := first_level_lives enemies_defeated
  let second_level := second_level_lives first_level powerups_collected
  second_level + third_level_bonus enemies_defeated powerups_collected = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_after_third_level_l3544_354401
