import Mathlib

namespace NUMINAMATH_CALUDE_pyramid_top_value_l669_66900

/-- Represents a pyramid structure with four levels -/
structure Pyramid :=
  (bottom_left : ℕ)
  (bottom_right : ℕ)
  (second_left : ℕ)
  (second_right : ℕ)
  (third_left : ℕ)
  (third_right : ℕ)
  (top : ℕ)

/-- Checks if the pyramid satisfies the product rule -/
def is_valid_pyramid (p : Pyramid) : Prop :=
  p.bottom_left = p.second_left * p.third_left ∧
  p.bottom_right = p.second_right * p.third_right ∧
  p.second_left = p.top * p.third_left ∧
  p.second_right = p.top * p.third_right

theorem pyramid_top_value (p : Pyramid) :
  p.bottom_left = 300 ∧ 
  p.bottom_right = 1800 ∧ 
  p.second_left = 6 ∧ 
  p.second_right = 30 ∧
  is_valid_pyramid p →
  p.top = 60 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_top_value_l669_66900


namespace NUMINAMATH_CALUDE_misha_grade_size_l669_66978

/-- The number of students in Misha's grade -/
def num_students : ℕ := 149

/-- Misha's position from the top of the grade -/
def position_from_top : ℕ := 75

/-- Misha's position from the bottom of the grade -/
def position_from_bottom : ℕ := 75

/-- Theorem: Given Misha's positions from top and bottom, prove the number of students in her grade -/
theorem misha_grade_size :
  position_from_top + position_from_bottom - 1 = num_students :=
by sorry

end NUMINAMATH_CALUDE_misha_grade_size_l669_66978


namespace NUMINAMATH_CALUDE_tile_c_in_rectangle_three_l669_66974

/-- Represents the four sides of a tile -/
structure TileSides :=
  (top : Nat)
  (right : Nat)
  (bottom : Nat)
  (left : Nat)

/-- Represents a tile with its sides -/
inductive Tile
| A
| B
| C
| D

/-- Represents the four rectangles -/
inductive Rectangle
| One
| Two
| Three
| Four

/-- Function to get the sides of a tile -/
def getTileSides (t : Tile) : TileSides :=
  match t with
  | Tile.A => ⟨6, 1, 3, 2⟩
  | Tile.B => ⟨3, 6, 2, 0⟩
  | Tile.C => ⟨4, 0, 5, 6⟩
  | Tile.D => ⟨2, 5, 1, 4⟩

/-- Predicate to check if two tiles can be placed adjacent to each other -/
def canBePlacedAdjacent (t1 t2 : Tile) (side : Nat → Nat) : Prop :=
  side (getTileSides t1).right = side (getTileSides t2).left

/-- The main theorem stating that Tile C must be placed in Rectangle 3 -/
theorem tile_c_in_rectangle_three :
  ∃ (placement : Tile → Rectangle),
    placement Tile.C = Rectangle.Three ∧
    (∀ t1 t2 : Tile, t1 ≠ t2 → placement t1 ≠ placement t2) ∧
    (∀ t1 t2 : Tile, 
      (placement t1 = Rectangle.One ∧ placement t2 = Rectangle.Two) ∨
      (placement t1 = Rectangle.Two ∧ placement t2 = Rectangle.Three) ∨
      (placement t1 = Rectangle.Three ∧ placement t2 = Rectangle.Four) →
      canBePlacedAdjacent t1 t2 id) := by
  sorry

end NUMINAMATH_CALUDE_tile_c_in_rectangle_three_l669_66974


namespace NUMINAMATH_CALUDE_prob_not_blue_from_odds_l669_66956

-- Define the odds ratio
def odds_blue : ℚ := 5 / 6

-- Define the probability of not obtaining a blue ball
def prob_not_blue : ℚ := 6 / 11

-- Theorem statement
theorem prob_not_blue_from_odds :
  odds_blue = 5 / 6 → prob_not_blue = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_blue_from_odds_l669_66956


namespace NUMINAMATH_CALUDE_adam_apples_solution_l669_66997

/-- Adam's apple purchases over three days --/
def adam_apples (monday_quantity : ℕ) (tuesday_multiple : ℕ) (wednesday_multiple : ℕ) : Prop :=
  let tuesday_quantity := monday_quantity * tuesday_multiple
  let wednesday_quantity := tuesday_quantity * wednesday_multiple
  monday_quantity + tuesday_quantity + wednesday_quantity = 240

theorem adam_apples_solution :
  ∃ (wednesday_multiple : ℕ),
    adam_apples 15 3 wednesday_multiple ∧ wednesday_multiple = 4 := by
  sorry

end NUMINAMATH_CALUDE_adam_apples_solution_l669_66997


namespace NUMINAMATH_CALUDE_gcd_128_144_256_l669_66915

theorem gcd_128_144_256 : Nat.gcd 128 (Nat.gcd 144 256) = 128 := by sorry

end NUMINAMATH_CALUDE_gcd_128_144_256_l669_66915


namespace NUMINAMATH_CALUDE_fraction_subtraction_l669_66991

theorem fraction_subtraction : 
  (4 : ℚ) / 5 - (1 : ℚ) / 5 = (6 : ℚ) / 10 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l669_66991


namespace NUMINAMATH_CALUDE_figure_to_square_possible_l669_66968

/-- A figure on a grid paper -/
structure GridFigure where
  area : ℕ

/-- Represents a dissection of a figure into parts -/
structure Dissection where
  parts : ℕ

/-- Represents a square shape -/
structure Square where
  side_length : ℕ

/-- A function that checks if a figure can be dissected into parts and formed into a square -/
def can_form_square (figure : GridFigure) (d : Dissection) (s : Square) : Prop :=
  figure.area = s.side_length ^ 2 ∧ d.parts = 3

theorem figure_to_square_possible (figure : GridFigure) (d : Dissection) (s : Square) 
  (h_area : figure.area = 16) (h_parts : d.parts = 3) (h_side : s.side_length = 4) : 
  can_form_square figure d s := by
  sorry

#check figure_to_square_possible

end NUMINAMATH_CALUDE_figure_to_square_possible_l669_66968


namespace NUMINAMATH_CALUDE_chromium_percent_alloy1_l669_66934

-- Define the weights and percentages
def weight_alloy1 : ℝ := 15
def weight_alloy2 : ℝ := 30
def chromium_percent_alloy2 : ℝ := 8
def chromium_percent_new : ℝ := 9.333333333333334

-- Theorem statement
theorem chromium_percent_alloy1 :
  ∃ (x : ℝ),
    x ≥ 0 ∧ x ≤ 100 ∧
    (x / 100 * weight_alloy1 + chromium_percent_alloy2 / 100 * weight_alloy2) / (weight_alloy1 + weight_alloy2) * 100 = chromium_percent_new ∧
    x = 12 :=
by sorry

end NUMINAMATH_CALUDE_chromium_percent_alloy1_l669_66934


namespace NUMINAMATH_CALUDE_basketball_handshakes_l669_66903

theorem basketball_handshakes :
  let team_size : ℕ := 6
  let num_teams : ℕ := 2
  let num_referees : ℕ := 3
  let player_handshakes := team_size * team_size
  let referee_handshakes := (team_size * num_teams) * num_referees
  player_handshakes + referee_handshakes = 72 := by
sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l669_66903


namespace NUMINAMATH_CALUDE_crayon_selection_problem_l669_66924

theorem crayon_selection_problem :
  let n : ℕ := 20  -- Total number of crayons
  let k : ℕ := 6   -- Number of crayons to select
  Nat.choose n k = 38760 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_problem_l669_66924


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l669_66909

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) → 
  a + h + k = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l669_66909


namespace NUMINAMATH_CALUDE_outfit_combinations_l669_66959

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (hats : ℕ) : 
  shirts = 5 → pants = 4 → hats = 2 → shirts * pants * hats = 40 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l669_66959


namespace NUMINAMATH_CALUDE_percentage_calculation_l669_66935

theorem percentage_calculation (p : ℝ) : 
  (p / 100) * 2348 / 4.98 = 528.0642570281125 → 
  ∃ ε > 0, |p - 112| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l669_66935


namespace NUMINAMATH_CALUDE_sam_found_35_seashells_l669_66979

/-- The number of seashells Joan found -/
def joans_seashells : ℕ := 18

/-- The total number of seashells Sam and Joan found together -/
def total_seashells : ℕ := 53

/-- The number of seashells Sam found -/
def sams_seashells : ℕ := total_seashells - joans_seashells

theorem sam_found_35_seashells : sams_seashells = 35 := by
  sorry

end NUMINAMATH_CALUDE_sam_found_35_seashells_l669_66979


namespace NUMINAMATH_CALUDE_negation_of_implication_l669_66953

theorem negation_of_implication (a : ℝ) : 
  ¬(a = -1 → a^2 = 1) ↔ (a ≠ -1 → a^2 ≠ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l669_66953


namespace NUMINAMATH_CALUDE_range_of_2m_plus_n_l669_66998

noncomputable def f (x : ℝ) := |Real.log x / Real.log 3|

theorem range_of_2m_plus_n (m n : ℝ) (h1 : 0 < m) (h2 : m < n) (h3 : f m = f n) :
  ∃ (lower : ℝ), lower = 2 * Real.sqrt 2 ∧
  (∀ x, x ≥ lower ↔ ∃ (m' n' : ℝ), 0 < m' ∧ m' < n' ∧ f m' = f n' ∧ 2 * m' + n' = x) :=
sorry

end NUMINAMATH_CALUDE_range_of_2m_plus_n_l669_66998


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l669_66957

/-- Given an arithmetic sequence {a_n} where a_3 = 1 and a_4 + a_10 = 18, prove that a_1 = -3 -/
theorem arithmetic_sequence_first_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a3 : a 3 = 1) 
  (h_sum : a 4 + a 10 = 18) : 
  a 1 = -3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l669_66957


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l669_66919

theorem quadratic_roots_property (c : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + x₁ + c = 0) → 
  (x₂^2 + x₂ + c = 0) → 
  (x₁^2 * x₂ + x₂^2 * x₁ = 3) → 
  c = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l669_66919


namespace NUMINAMATH_CALUDE_monthly_repayment_l669_66989

theorem monthly_repayment (T M : ℝ) 
  (h1 : T / 2 = 6 * M)
  (h2 : T / 2 - 4 * M = 20) : 
  M = 10 := by sorry

end NUMINAMATH_CALUDE_monthly_repayment_l669_66989


namespace NUMINAMATH_CALUDE_bird_nest_problem_l669_66967

theorem bird_nest_problem (first_bird_initial : Nat) (first_bird_additional : Nat)
                          (second_bird_initial : Nat) (second_bird_additional : Nat)
                          (third_bird_initial : Nat) (third_bird_additional : Nat)
                          (first_bird_carry_capacity : Nat) (tree_drop_fraction : Nat) :
  first_bird_initial = 12 →
  first_bird_additional = 6 →
  second_bird_initial = 15 →
  second_bird_additional = 8 →
  third_bird_initial = 10 →
  third_bird_additional = 4 →
  first_bird_carry_capacity = 3 →
  tree_drop_fraction = 3 →
  (first_bird_initial * first_bird_additional +
   second_bird_initial * second_bird_additional +
   third_bird_initial * third_bird_additional = 232) ∧
  (((first_bird_initial * first_bird_additional) -
    (first_bird_initial * first_bird_additional / tree_drop_fraction)) /
    first_bird_carry_capacity = 16) :=
by sorry

end NUMINAMATH_CALUDE_bird_nest_problem_l669_66967


namespace NUMINAMATH_CALUDE_triangle_height_proof_l669_66926

/-- Given a square with side length s, a rectangle with base s and height h,
    and an isosceles triangle with base s and height h,
    prove that h = 2s/3 when the combined area of the rectangle and triangle
    equals the area of the square. -/
theorem triangle_height_proof (s : ℝ) (h : ℝ) : 
  s > 0 → h > 0 → s * h + (s * h) / 2 = s^2 → h = 2 * s / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_proof_l669_66926


namespace NUMINAMATH_CALUDE_smart_mart_science_kits_l669_66985

/-- The number of puzzles sold by Smart Mart -/
def puzzles_sold : ℕ := 36

/-- The difference between science kits and puzzles sold -/
def difference : ℕ := 9

/-- The number of science kits sold by Smart Mart -/
def science_kits_sold : ℕ := puzzles_sold + difference

/-- Theorem stating that Smart Mart sold 45 science kits -/
theorem smart_mart_science_kits : science_kits_sold = 45 := by
  sorry

end NUMINAMATH_CALUDE_smart_mart_science_kits_l669_66985


namespace NUMINAMATH_CALUDE_function_sum_positive_l669_66901

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x < y → x < 2 → f x < f y)
variable (h2 : ∀ x, f (x + 2) = -f (-x + 2))

-- Define the theorem
theorem function_sum_positive (x₁ x₂ : ℝ) 
  (hx₁ : x₁ < 2) (hx₂ : x₂ > 2) (h : |x₁ - 2| < |x₂ - 2|) :
  f x₁ + f x₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_sum_positive_l669_66901


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l669_66938

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) :
  (a + 2 + 4 / (a - 2)) / (a^3 / (a^2 - 4*a + 4)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l669_66938


namespace NUMINAMATH_CALUDE_abs_rational_nonnegative_l669_66999

theorem abs_rational_nonnegative (a : ℚ) : 0 ≤ |a| := by
  sorry

end NUMINAMATH_CALUDE_abs_rational_nonnegative_l669_66999


namespace NUMINAMATH_CALUDE_instrument_probability_l669_66990

theorem instrument_probability (total : ℕ) (at_least_one : ℚ) (two_or_more : ℕ) : 
  total = 800 →
  at_least_one = 2 / 5 →
  two_or_more = 96 →
  (((at_least_one * total) - two_or_more) / total : ℚ) = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_instrument_probability_l669_66990


namespace NUMINAMATH_CALUDE_size_relationship_l669_66963

theorem size_relationship (x : ℝ) : 
  let a := x^2 + x + Real.sqrt 2
  let b := Real.log 3 / Real.log 10
  let c := Real.exp (-1/2)
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_size_relationship_l669_66963


namespace NUMINAMATH_CALUDE_expected_worth_unfair_coin_l669_66911

/-- An unfair coin with given probabilities and payoffs -/
structure UnfairCoin where
  probHeads : ℚ
  probTails : ℚ
  payoffHeads : ℚ
  payoffTails : ℚ
  prob_sum : probHeads + probTails = 1

/-- The expected value of a flip of the unfair coin -/
def expectedValue (coin : UnfairCoin) : ℚ :=
  coin.probHeads * coin.payoffHeads + coin.probTails * coin.payoffTails

/-- Theorem: The expected worth of a specific unfair coin flip -/
theorem expected_worth_unfair_coin :
  ∃ (coin : UnfairCoin),
    coin.probHeads = 2/3 ∧
    coin.probTails = 1/3 ∧
    coin.payoffHeads = 5 ∧
    coin.payoffTails = -9 ∧
    expectedValue coin = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_unfair_coin_l669_66911


namespace NUMINAMATH_CALUDE_democrat_count_l669_66908

theorem democrat_count (total : ℕ) (difference : ℕ) (h1 : total = 434) (h2 : difference = 30) :
  let democrats := (total - difference) / 2
  democrats = 202 := by
sorry

end NUMINAMATH_CALUDE_democrat_count_l669_66908


namespace NUMINAMATH_CALUDE_true_compound_proposition_l669_66977

-- Define the propositions
def p : Prop := ∀ x : ℝ, x < 0 → x^3 < 0
def q : Prop := ∀ x : ℝ, x > 0 → Real.log x < 0

-- Theorem to prove
theorem true_compound_proposition : (¬p) ∨ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_true_compound_proposition_l669_66977


namespace NUMINAMATH_CALUDE_walking_distance_l669_66976

theorem walking_distance (initial_speed : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) 
  (h1 : initial_speed = 12)
  (h2 : faster_speed = 16)
  (h3 : additional_distance = 20) :
  ∃ (actual_distance : ℝ) (time : ℝ),
    actual_distance = initial_speed * time ∧
    actual_distance + additional_distance = faster_speed * time ∧
    actual_distance = 60 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_l669_66976


namespace NUMINAMATH_CALUDE_line_through_two_points_l669_66987

/-- Given a line x = 6y + 5 passing through points (m, n) and (m + 2, n + p) in a rectangular coordinate system, prove that p = 1/3 -/
theorem line_through_two_points (m n : ℝ) : 
  (m = 6 * n + 5) → (m + 2 = 6 * (n + p) + 5) → p = 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_line_through_two_points_l669_66987


namespace NUMINAMATH_CALUDE_sum_of_integers_l669_66928

theorem sum_of_integers (a b : ℕ+) : a^2 - b^2 = 52 → a * b = 168 → a + b = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l669_66928


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4290_l669_66936

theorem largest_prime_factor_of_4290 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4290 ∧ ∀ q, Nat.Prime q → q ∣ 4290 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4290_l669_66936


namespace NUMINAMATH_CALUDE_exponent_division_l669_66986

theorem exponent_division (a : ℝ) (m n : ℕ) : a ^ m / a ^ n = a ^ (m - n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l669_66986


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l669_66942

theorem cubic_polynomial_root (a b c : ℚ) :
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 2 - Real.sqrt 5) →
  (∀ x y : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ y^3 + a*y^2 + b*y + c = 0 → x + y = 4) →
  (-4 : ℝ)^3 + a*(-4 : ℝ)^2 + b*(-4 : ℝ) + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l669_66942


namespace NUMINAMATH_CALUDE_functional_equation_solution_l669_66937

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x - f y) = f x * f y) : 
  (∀ x : ℚ, f x = 0) ∨ (∀ x : ℚ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l669_66937


namespace NUMINAMATH_CALUDE_asparagus_per_plate_l669_66907

theorem asparagus_per_plate
  (bridgette_guests : ℕ)
  (alex_guests : ℕ)
  (extra_plates : ℕ)
  (total_asparagus : ℕ)
  (h1 : bridgette_guests = 84)
  (h2 : alex_guests = 2 * bridgette_guests / 3)
  (h3 : extra_plates = 10)
  (h4 : total_asparagus = 1200) :
  total_asparagus / (bridgette_guests + alex_guests + extra_plates) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_asparagus_per_plate_l669_66907


namespace NUMINAMATH_CALUDE_daps_equivalent_to_48_dips_l669_66922

/-- Represents the number of units of a currency -/
structure Currency where
  amount : ℚ
  name : String

/-- Defines the exchange rate between two currencies -/
def exchange_rate (a b : Currency) : ℚ := a.amount / b.amount

/-- Given conditions of the problem -/
axiom daps_to_dops : exchange_rate (Currency.mk 5 "daps") (Currency.mk 4 "dops") = 1
axiom dops_to_dips : exchange_rate (Currency.mk 3 "dops") (Currency.mk 8 "dips") = 1

/-- The theorem to be proved -/
theorem daps_equivalent_to_48_dips :
  exchange_rate (Currency.mk 22.5 "daps") (Currency.mk 48 "dips") = 1 := by
  sorry

end NUMINAMATH_CALUDE_daps_equivalent_to_48_dips_l669_66922


namespace NUMINAMATH_CALUDE_tan_inequality_l669_66962

open Real

theorem tan_inequality (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁ ∧ x₁ < π/2) 
  (h₂ : 0 < x₂ ∧ x₂ < π/2) 
  (h₃ : x₁ ≠ x₂) : 
  (1/2) * (tan x₁ + tan x₂) > tan ((x₁ + x₂)/2) := by
  sorry

end NUMINAMATH_CALUDE_tan_inequality_l669_66962


namespace NUMINAMATH_CALUDE_semicircle_area_ratio_l669_66947

theorem semicircle_area_ratio (r : ℝ) (h : r > 0) :
  let semicircle_area := π * (r / Real.sqrt 2)^2 / 2
  let circle_area := π * r^2
  2 * semicircle_area / circle_area = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_semicircle_area_ratio_l669_66947


namespace NUMINAMATH_CALUDE_infinite_integer_solutions_l669_66943

theorem infinite_integer_solutions :
  ∃ (S : Set ℕ+), (Set.Infinite S) ∧
  (∀ n ∈ S, ¬ ∃ m : ℕ, n = m^3) ∧
  (∀ n ∈ S,
    let a : ℝ := (n : ℝ)^(1/3)
    let b : ℝ := 1 / (a - ⌊a⌋)
    let c : ℝ := 1 / (b - ⌊b⌋)
    ∃ r s t : ℤ, (r ≠ 0 ∨ s ≠ 0 ∨ t ≠ 0) ∧ r * a + s * b + t * c = 0) :=
by sorry

end NUMINAMATH_CALUDE_infinite_integer_solutions_l669_66943


namespace NUMINAMATH_CALUDE_edward_initial_amount_l669_66966

def initial_amount (book_price shirt_price shirt_discount meal_price
                    ticket_price ticket_discount amount_left : ℝ) : ℝ :=
  book_price +
  (shirt_price * (1 - shirt_discount)) +
  meal_price +
  (ticket_price - ticket_discount) +
  amount_left

theorem edward_initial_amount :
  initial_amount 9 25 0.2 15 10 2 17 = 69 := by
  sorry

end NUMINAMATH_CALUDE_edward_initial_amount_l669_66966


namespace NUMINAMATH_CALUDE_round_trip_speed_l669_66904

/-- Proves that given the conditions of the round trip, the speed from B to A is 45 miles per hour -/
theorem round_trip_speed (distance : ℝ) (speed_ab : ℝ) (avg_speed : ℝ) (speed_ba : ℝ) : 
  distance = 180 →
  speed_ab = 90 →
  avg_speed = 60 →
  speed_ba = (2 * distance * avg_speed) / (2 * distance - avg_speed * (distance / speed_ab)) →
  speed_ba = 45 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_speed_l669_66904


namespace NUMINAMATH_CALUDE_x_value_proof_l669_66910

theorem x_value_proof (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 36) : x = 28 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l669_66910


namespace NUMINAMATH_CALUDE_two_activities_count_l669_66905

/-- Represents a school club with three activities -/
structure Club where
  total_members : ℕ
  cannot_paint : ℕ
  cannot_sculpt : ℕ
  cannot_draw : ℕ

/-- Calculates the number of members involved in exactly two activities -/
def members_in_two_activities (c : Club) : ℕ :=
  let can_paint := c.total_members - c.cannot_paint
  let can_sculpt := c.total_members - c.cannot_sculpt
  let can_draw := c.total_members - c.cannot_draw
  can_paint + can_sculpt + can_draw - c.total_members

/-- Theorem stating the number of members involved in exactly two activities -/
theorem two_activities_count (c : Club) 
  (h1 : c.total_members = 150)
  (h2 : c.cannot_paint = 55)
  (h3 : c.cannot_sculpt = 90)
  (h4 : c.cannot_draw = 40) :
  members_in_two_activities c = 115 := by
  sorry

#eval members_in_two_activities ⟨150, 55, 90, 40⟩

end NUMINAMATH_CALUDE_two_activities_count_l669_66905


namespace NUMINAMATH_CALUDE_exist_consecutive_lucky_tickets_l669_66948

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a number is a six-digit number -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- A predicate that checks if a number is lucky (sum of digits divisible by 7) -/
def is_lucky (n : ℕ) : Prop := sum_of_digits n % 7 = 0

/-- Theorem stating that there exist two consecutive six-digit numbers that are both lucky -/
theorem exist_consecutive_lucky_tickets : 
  ∃ n : ℕ, is_six_digit n ∧ is_six_digit (n + 1) ∧ is_lucky n ∧ is_lucky (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_exist_consecutive_lucky_tickets_l669_66948


namespace NUMINAMATH_CALUDE_geometric_sequence_product_roots_product_geometric_sequence_roots_product_l669_66975

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∀ i j k l : ℕ, i + j = k + l → a i * a j = a k * a l :=
sorry

theorem roots_product (p q r : ℝ) (x y : ℝ) (hx : p * x^2 + q * x + r = 0) (hy : p * y^2 + q * y + r = 0) :
  x * y = r / p :=
sorry

theorem geometric_sequence_roots_product (a : ℕ → ℝ) :
  geometric_sequence a →
  3 * (a 1)^2 + 7 * (a 1) - 9 = 0 →
  3 * (a 10)^2 + 7 * (a 10) - 9 = 0 →
  a 4 * a 7 = -3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_roots_product_geometric_sequence_roots_product_l669_66975


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l669_66918

-- Equation 1
theorem solve_equation_one (x : ℚ) : 2 * (x - 3) = 1 - 3 * (x + 1) ↔ x = 4 / 5 := by sorry

-- Equation 2
theorem solve_equation_two (x : ℚ) : 3 * x + (x - 1) / 2 = 3 - (x - 1) / 3 ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l669_66918


namespace NUMINAMATH_CALUDE_gcd_176_88_l669_66914

theorem gcd_176_88 : Nat.gcd 176 88 = 88 := by
  sorry

end NUMINAMATH_CALUDE_gcd_176_88_l669_66914


namespace NUMINAMATH_CALUDE_semi_circle_perimeter_after_increase_l669_66993

/-- The perimeter of a semi-circle with radius 7.68 cm is approximately 39.50 cm. -/
theorem semi_circle_perimeter_after_increase : 
  let r : ℝ := 7.68
  let π : ℝ := 3.14159
  let perimeter : ℝ := π * r + 2 * r
  ∃ ε > 0, |perimeter - 39.50| < ε :=
by sorry

end NUMINAMATH_CALUDE_semi_circle_perimeter_after_increase_l669_66993


namespace NUMINAMATH_CALUDE_chromatid_non_separation_can_result_in_XXY_l669_66906

/- Define the basic types and structures -/
inductive Chromosome
| X
| Y

structure Sperm :=
(chromosomes : List Chromosome)

structure Egg :=
(chromosomes : List Chromosome)

structure Offspring :=
(chromosomes : List Chromosome)

/- Define the process of sperm formation with non-separation of chromatids -/
def spermFormationWithNonSeparation : List Sperm :=
[{chromosomes := [Chromosome.X, Chromosome.X]}, {chromosomes := [Chromosome.Y, Chromosome.Y]}]

/- Define a normal egg -/
def normalEgg : Egg :=
{chromosomes := [Chromosome.X]}

/- Define the fertilization process -/
def fertilize (sperm : Sperm) (egg : Egg) : Offspring :=
{chromosomes := sperm.chromosomes ++ egg.chromosomes}

/- The theorem to be proved -/
theorem chromatid_non_separation_can_result_in_XXY :
  ∃ (sperm : Sperm) (egg : Egg),
    sperm ∈ spermFormationWithNonSeparation ∧
    egg = normalEgg ∧
    (fertilize sperm egg).chromosomes = [Chromosome.X, Chromosome.X, Chromosome.Y] :=
sorry

end NUMINAMATH_CALUDE_chromatid_non_separation_can_result_in_XXY_l669_66906


namespace NUMINAMATH_CALUDE_new_building_windows_l669_66944

/-- The number of windows needed for a new building --/
def total_windows (installed : ℕ) (hours_per_window : ℕ) (remaining_hours : ℕ) : ℕ :=
  installed + remaining_hours / hours_per_window

/-- Proof that the total number of windows needed is 14 --/
theorem new_building_windows :
  total_windows 5 4 36 = 14 :=
by sorry

end NUMINAMATH_CALUDE_new_building_windows_l669_66944


namespace NUMINAMATH_CALUDE_pathway_layers_l669_66961

def bricks_in_layer (n : ℕ) : ℕ :=
  if n % 2 = 1 then 4 else 4 * 2^((n / 2) - 1)

def total_bricks (n : ℕ) : ℕ :=
  (List.range n).map (λ i => bricks_in_layer (i + 1)) |> List.sum

theorem pathway_layers : ∃ n : ℕ, n > 0 ∧ total_bricks n = 280 :=
  sorry

end NUMINAMATH_CALUDE_pathway_layers_l669_66961


namespace NUMINAMATH_CALUDE_product_congruence_zero_mod_17_l669_66988

theorem product_congruence_zero_mod_17 : 
  (2357 * 2369 * 2384 * 2391) * (3017 * 3079 * 3082) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_congruence_zero_mod_17_l669_66988


namespace NUMINAMATH_CALUDE_hall_dimension_l669_66983

/-- Represents a square rug with a given side length. -/
structure Rug where
  side : ℝ
  square : side > 0

/-- Represents a square hall containing two rugs. -/
structure Hall where
  small_rug : Rug
  large_rug : Rug
  opposite_overlap : ℝ
  adjacent_overlap : ℝ
  hall_side : ℝ

/-- The theorem stating the conditions and the conclusion about the hall's dimensions. -/
theorem hall_dimension (h : Hall) : 
  h.large_rug.side = 2 * h.small_rug.side ∧ 
  h.opposite_overlap = 4 ∧ 
  h.adjacent_overlap = 14 → 
  h.hall_side = 19 := by
  sorry


end NUMINAMATH_CALUDE_hall_dimension_l669_66983


namespace NUMINAMATH_CALUDE_sector_central_angle_l669_66913

/-- Given a circular sector with area 6 cm² and radius 2 cm, prove its central angle is 3 radians. -/
theorem sector_central_angle (area : ℝ) (radius : ℝ) (h1 : area = 6) (h2 : radius = 2) :
  (2 * area) / (radius ^ 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l669_66913


namespace NUMINAMATH_CALUDE_difference_of_41st_terms_l669_66971

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem difference_of_41st_terms : 
  let C := arithmetic_sequence 50 15
  let D := arithmetic_sequence 50 (-15)
  |C 41 - D 41| = 1200 := by sorry

end NUMINAMATH_CALUDE_difference_of_41st_terms_l669_66971


namespace NUMINAMATH_CALUDE_three_digit_prime_integers_count_l669_66939

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := sorry

-- Define a function to get all single-digit prime numbers
def singleDigitPrimes : List ℕ := sorry

-- Define a function to count the number of three-digit positive integers
-- where the digits are three different prime numbers
def countThreeDigitPrimeIntegers : ℕ := sorry

-- Theorem statement
theorem three_digit_prime_integers_count :
  countThreeDigitPrimeIntegers = 24 := by sorry

end NUMINAMATH_CALUDE_three_digit_prime_integers_count_l669_66939


namespace NUMINAMATH_CALUDE_smoking_rate_estimate_l669_66921

/-- Represents the survey results and conditions -/
structure SurveyData where
  total_students : ℕ
  yes_answers : ℕ
  die_prob : ℚ

/-- Calculates the estimated smoking rate based on survey data -/
def estimate_smoking_rate (data : SurveyData) : ℚ :=
  let estimated_smokers := data.yes_answers / 2
  (estimated_smokers : ℚ) / data.total_students

/-- Theorem stating the estimated smoking rate for the given survey data -/
theorem smoking_rate_estimate (data : SurveyData) 
  (h1 : data.total_students = 300)
  (h2 : data.yes_answers = 80)
  (h3 : data.die_prob = 1/2) :
  ∃ (ε : ℚ), abs (estimate_smoking_rate data - 40/300) < ε ∧ ε < 1/1000 := by
  sorry

end NUMINAMATH_CALUDE_smoking_rate_estimate_l669_66921


namespace NUMINAMATH_CALUDE_valid_B_values_l669_66970

def is_valid_B (B : ℕ) : Prop :=
  B < 10 ∧ (∃ k : ℤ, 40000 + 1110 * B + 2 = 9 * k)

theorem valid_B_values :
  ∀ B : ℕ, is_valid_B B ↔ (B = 1 ∨ B = 4 ∨ B = 7) :=
by sorry

end NUMINAMATH_CALUDE_valid_B_values_l669_66970


namespace NUMINAMATH_CALUDE_smallest_m_for_nth_root_in_T_l669_66925

def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

theorem smallest_m_for_nth_root_in_T : 
  (∀ n : ℕ, n ≥ 12 → ∃ z ∈ T, z ^ n = 1) ∧ 
  (∀ m : ℕ, m < 12 → ∃ n : ℕ, n ≥ m ∧ ∀ z ∈ T, z ^ n ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_nth_root_in_T_l669_66925


namespace NUMINAMATH_CALUDE_correct_proposition_l669_66965

/-- Proposition p: The solution set of ax^2 + ax + 1 > 0 is ℝ, then a ∈ (0,4) -/
def p : Prop := ∀ x : ℝ, (∃ a : ℝ, a * x^2 + a * x + 1 > 0) → (∃ a : ℝ, 0 < a ∧ a < 4)

/-- Proposition q: "x^2 - 2x - 8 > 0" is a necessary but not sufficient condition for "x > 5" -/
def q : Prop := (∀ x : ℝ, x > 5 → x^2 - 2*x - 8 > 0) ∧ (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≤ 5)

/-- The correct proposition is (¬p) ∧ q -/
theorem correct_proposition : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_correct_proposition_l669_66965


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l669_66973

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 3 ∧ 
  (∃ (n : ℕ), 4 * b + 5 = n^2) ∧ 
  (∀ (x : ℕ), x > 3 ∧ x < b → ¬∃ (m : ℕ), 4 * x + 5 = m^2) ∧
  b = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l669_66973


namespace NUMINAMATH_CALUDE_sausage_cutting_theorem_l669_66969

/-- Represents the number of pieces produced when cutting along rings of a single color -/
def PiecesFromSingleColor : ℕ → ℕ := λ n => n + 1

/-- Represents the total number of pieces produced when cutting along rings of multiple colors -/
def TotalPieces (cuts : List ℕ) : ℕ :=
  (cuts.sum) + 1

theorem sausage_cutting_theorem (red yellow green : ℕ) 
  (h_red : PiecesFromSingleColor red = 5)
  (h_yellow : PiecesFromSingleColor yellow = 7)
  (h_green : PiecesFromSingleColor green = 11) :
  TotalPieces [red, yellow, green] = 21 := by
  sorry

#check sausage_cutting_theorem

end NUMINAMATH_CALUDE_sausage_cutting_theorem_l669_66969


namespace NUMINAMATH_CALUDE_total_balloons_l669_66995

theorem total_balloons (joan_balloons melanie_balloons : ℕ) 
  (h1 : joan_balloons = 40)
  (h2 : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l669_66995


namespace NUMINAMATH_CALUDE_total_distance_calculation_l669_66964

/-- Calculates the total distance covered by a man rowing upstream and downstream -/
theorem total_distance_calculation (upstream_speed : ℝ) (upstream_time : ℝ) 
  (downstream_speed : ℝ) (downstream_time : ℝ) : 
  upstream_speed * upstream_time + downstream_speed * downstream_time = 62 :=
by
  -- Proof goes here
  sorry

#check total_distance_calculation 12 2 38 1

end NUMINAMATH_CALUDE_total_distance_calculation_l669_66964


namespace NUMINAMATH_CALUDE_perpendicular_vectors_t_value_l669_66933

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_t_value :
  ∀ t : ℝ, perpendicular (3, 1) (t, -3) → t = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_t_value_l669_66933


namespace NUMINAMATH_CALUDE_sum_lent_problem_l669_66923

/-- Proves that given a sum P lent at 5% per annum simple interest for 8 years,
    if the interest is $360 less than P, then P equals $600. -/
theorem sum_lent_problem (P : ℝ) : 
  (P * 0.05 * 8 = P - 360) → P = 600 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_problem_l669_66923


namespace NUMINAMATH_CALUDE_symmetric_points_product_l669_66945

/-- Two points (x₁, y₁) and (x₂, y₂) are symmetric with respect to the origin if x₁ = -x₂ and y₁ = -y₂ -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

theorem symmetric_points_product (a b : ℝ) :
  symmetric_wrt_origin (a + 2) 2 4 (-b) →
  a * b = -12 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_product_l669_66945


namespace NUMINAMATH_CALUDE_positive_numbers_l669_66932

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (sum_products_positive : b * c + c * a + a * b > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_l669_66932


namespace NUMINAMATH_CALUDE_positive_numbers_equality_l669_66981

theorem positive_numbers_equality (a b : ℝ) : 
  0 < a → 0 < b → a^b = b^a → b = 3*a → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_equality_l669_66981


namespace NUMINAMATH_CALUDE_min_socks_for_15_pairs_l669_66929

/-- Represents the number of socks of each color in the box -/
structure SockBox where
  white : Nat
  red : Nat
  blue : Nat
  green : Nat
  yellow : Nat

/-- Calculates the minimum number of socks needed to guarantee at least n pairs -/
def minSocksForPairs (box : SockBox) (n : Nat) : Nat :=
  (box.white.min n + box.red.min n + box.blue.min n + box.green.min n + box.yellow.min n) * 2 - 1

/-- The theorem stating the minimum number of socks needed for 15 pairs -/
theorem min_socks_for_15_pairs (box : SockBox) 
    (h_white : box.white = 150)
    (h_red : box.red = 120)
    (h_blue : box.blue = 90)
    (h_green : box.green = 60)
    (h_yellow : box.yellow = 30) :
    minSocksForPairs box 15 = 146 := by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_15_pairs_l669_66929


namespace NUMINAMATH_CALUDE_real_estate_commission_l669_66992

/-- Calculate the commission for a real estate agent given the selling price and commission rate -/
def calculate_commission (selling_price : ℝ) (commission_rate : ℝ) : ℝ :=
  selling_price * commission_rate

/-- Theorem stating that the commission for a house sold at $148,000 with a 6% commission rate is $8,880 -/
theorem real_estate_commission :
  let selling_price : ℝ := 148000
  let commission_rate : ℝ := 0.06
  calculate_commission selling_price commission_rate = 8880 := by
  sorry

end NUMINAMATH_CALUDE_real_estate_commission_l669_66992


namespace NUMINAMATH_CALUDE_hotel_expenditure_l669_66955

theorem hotel_expenditure (num_men : ℕ) (standard_cost : ℚ) (extra_cost : ℚ) :
  num_men = 9 →
  standard_cost = 3 →
  extra_cost = 2 →
  (((num_men - 1) * standard_cost + 
    (standard_cost + extra_cost + 
      ((num_men - 1) * standard_cost + (standard_cost + extra_cost)) / num_men)) = 29.25) := by
  sorry

end NUMINAMATH_CALUDE_hotel_expenditure_l669_66955


namespace NUMINAMATH_CALUDE_empty_quadratic_inequality_implies_a_range_l669_66951

theorem empty_quadratic_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 4 ≥ 0) → a ∈ Set.Icc (-4) 4 := by
  sorry

end NUMINAMATH_CALUDE_empty_quadratic_inequality_implies_a_range_l669_66951


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l669_66912

theorem similar_triangles_leg_sum (A₁ A₂ : ℝ) (h : ℝ) (s : ℝ) :
  A₁ = 18 →
  A₂ = 288 →
  h = 9 →
  (A₂ / A₁ = (s / h) ^ 2) →
  s = 4 * Real.sqrt 153 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l669_66912


namespace NUMINAMATH_CALUDE_sales_growth_rate_l669_66972

theorem sales_growth_rate (x : ℝ) : (1 + x)^2 = 1 + 0.44 → x < 0.22 := by
  sorry

end NUMINAMATH_CALUDE_sales_growth_rate_l669_66972


namespace NUMINAMATH_CALUDE_circle_radius_l669_66946

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 34 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (4, 5)

/-- The theorem stating that the radius of the circle is √7 -/
theorem circle_radius :
  ∃ (r : ℝ), r > 0 ∧
  (∀ (x y : ℝ), circle_equation x y ↔ 
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = r^2) ∧
  r = Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l669_66946


namespace NUMINAMATH_CALUDE_beef_pack_weight_l669_66920

/-- Given the conditions of James' beef purchase, prove the weight of each pack. -/
theorem beef_pack_weight (num_packs : ℕ) (price_per_pound : ℚ) (total_paid : ℚ) 
  (h1 : num_packs = 5)
  (h2 : price_per_pound = 5.5)
  (h3 : total_paid = 110) :
  (total_paid / price_per_pound) / num_packs = 4 := by
  sorry

#check beef_pack_weight

end NUMINAMATH_CALUDE_beef_pack_weight_l669_66920


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l669_66917

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l669_66917


namespace NUMINAMATH_CALUDE_infinitely_many_primes_mod_3_eq_2_l669_66950

theorem infinitely_many_primes_mod_3_eq_2 : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 3 = 2} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_mod_3_eq_2_l669_66950


namespace NUMINAMATH_CALUDE_sum_of_exponentials_l669_66960

theorem sum_of_exponentials (x y : ℝ) :
  (3^x + 3^(y+1) = 5 * Real.sqrt 3) →
  (3^(x+1) + 3^y = 3 * Real.sqrt 3) →
  3^x + 3^y = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_exponentials_l669_66960


namespace NUMINAMATH_CALUDE_function_is_linear_l669_66902

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

/-- The main theorem stating that any function satisfying the equation is linear -/
theorem function_is_linear (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x := by
  sorry

end NUMINAMATH_CALUDE_function_is_linear_l669_66902


namespace NUMINAMATH_CALUDE_unique_solution_system_l669_66984

theorem unique_solution_system :
  ∃! (x y z w : ℝ),
    x = z + w + z * w * z ∧
    y = w + x + w * z * x ∧
    z = x + y + x * y * x ∧
    w = y + z + z * y * z :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l669_66984


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_triangles_l669_66954

/-- Represents a trapezoid with given area and bases -/
structure Trapezoid where
  area : ℝ
  base1 : ℝ
  base2 : ℝ

/-- Represents the areas of triangles formed by diagonals in a trapezoid -/
structure DiagonalTriangles where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

/-- 
Given a trapezoid with area 3 and bases 1 and 2, 
the areas of the triangles formed by its diagonals are 1/3, 2/3, 2/3, and 4/3
-/
theorem trapezoid_diagonal_triangles (t : Trapezoid) 
  (h1 : t.area = 3) 
  (h2 : t.base1 = 1) 
  (h3 : t.base2 = 2) : 
  ∃ (d : DiagonalTriangles), 
    d.area1 = 1/3 ∧ 
    d.area2 = 2/3 ∧ 
    d.area3 = 2/3 ∧ 
    d.area4 = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_diagonal_triangles_l669_66954


namespace NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l669_66952

theorem ten_thousandths_place_of_5_32 : 
  ∃ (n : ℕ), (5 : ℚ) / 32 = (n * 10000 + 2) / 100000 ∧ n < 10000 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l669_66952


namespace NUMINAMATH_CALUDE_quadratic_property_l669_66958

/-- Quadratic function -/
def f (c : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + c

theorem quadratic_property (c : ℝ) (x₁ : ℝ) (hc : c < 0) (hx₁ : f c x₁ > 0) :
  f c (x₁ - 2) < 0 ∧ f c (x₁ + 2) < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_property_l669_66958


namespace NUMINAMATH_CALUDE_set_inclusion_iff_m_range_l669_66941

theorem set_inclusion_iff_m_range (m : ℝ) :
  ({x : ℝ | x^2 - 2*x - 3 ≤ 0} ⊆ {x : ℝ | |x - m| > 3}) ↔ 
  (m < -4 ∨ m > 6) :=
sorry

end NUMINAMATH_CALUDE_set_inclusion_iff_m_range_l669_66941


namespace NUMINAMATH_CALUDE_planting_cost_l669_66916

def flower_cost : ℕ := 9
def clay_pot_cost : ℕ := flower_cost + 20
def soil_cost : ℕ := flower_cost - 2

def total_cost : ℕ := flower_cost + clay_pot_cost + soil_cost

theorem planting_cost : total_cost = 45 := by
  sorry

end NUMINAMATH_CALUDE_planting_cost_l669_66916


namespace NUMINAMATH_CALUDE_circle_point_range_l669_66927

/-- Given a point A(0,-3) and a circle C: (x-a)^2 + (y-a+2)^2 = 1,
    if there exists a point M on C such that MA = 2MO,
    then 0 ≤ a ≤ 3 -/
theorem circle_point_range (a : ℝ) :
  (∃ x y : ℝ, (x - a)^2 + (y - a + 2)^2 = 1 ∧
              (x^2 + (y + 3)^2) = 4 * (x^2 + y^2)) →
  0 ≤ a ∧ a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_circle_point_range_l669_66927


namespace NUMINAMATH_CALUDE_cricket_bat_selling_price_l669_66940

-- Define the profit amount
def profit : ℝ := 230

-- Define the profit percentage
def profitPercentage : ℝ := 37.096774193548384

-- Define the selling price
def sellingPrice : ℝ := 850

-- Theorem to prove
theorem cricket_bat_selling_price :
  (profit / (profitPercentage / 100) + profit) = sellingPrice := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_selling_price_l669_66940


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l669_66980

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : is_geometric_sequence a q)
  (h_q_bounds : 0 < q ∧ q < 1/2)
  (h_property : ∀ k : ℕ, k > 0 → ∃ n : ℕ, a k - (a (k+1) + a (k+2)) = a n) :
  q = Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l669_66980


namespace NUMINAMATH_CALUDE_probability_three_black_cards_l669_66931

/-- The probability of drawing three black cards consecutively from a standard deck --/
theorem probability_three_black_cards (total_cards : ℕ) (black_cards : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : black_cards = 26) : 
  (black_cards * (black_cards - 1) * (black_cards - 2)) / 
  (total_cards * (total_cards - 1) * (total_cards - 2)) = 4 / 17 := by
sorry

end NUMINAMATH_CALUDE_probability_three_black_cards_l669_66931


namespace NUMINAMATH_CALUDE_no_real_solutions_for_matrix_equation_l669_66982

theorem no_real_solutions_for_matrix_equation : 
  ¬∃ (x : ℝ), (3 * x * x - 8 = 2 * x^2 - 3 * x - 4) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_matrix_equation_l669_66982


namespace NUMINAMATH_CALUDE_complex_square_l669_66930

theorem complex_square (a b : ℝ) (h : (a : ℂ) + Complex.I = 2 - b * Complex.I) :
  (a + b * Complex.I)^2 = 3 - 4 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_square_l669_66930


namespace NUMINAMATH_CALUDE_total_students_l669_66996

theorem total_students (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 16) 
  (h2 : rank_from_left = 6) : 
  rank_from_right + rank_from_left - 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l669_66996


namespace NUMINAMATH_CALUDE_parallelogram_z_range_l669_66949

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (4, -2)

-- Define the function z
def z (x y : ℝ) : ℝ := 2 * x - 5 * y

-- Theorem statement
theorem parallelogram_z_range :
  ∀ (x y : ℝ), 
  (∃ (t₁ t₂ t₃ : ℝ), 0 ≤ t₁ ∧ 0 ≤ t₂ ∧ 0 ≤ t₃ ∧ t₁ + t₂ + t₃ ≤ 1 ∧
    (x, y) = t₁ • A + t₂ • B + t₃ • C + (1 - t₁ - t₂ - t₃) • (A + C - B)) →
  -14 ≤ z x y ∧ z x y ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_z_range_l669_66949


namespace NUMINAMATH_CALUDE_spatial_pythagorean_quadruplet_l669_66994

theorem spatial_pythagorean_quadruplet (m n p q : ℤ) : 
  let x := 2 * m * p + 2 * n * q
  let y := |2 * m * q - 2 * n * p|
  let z := |m^2 + n^2 - p^2 - q^2|
  let u := m^2 + n^2 + p^2 + q^2
  (x^2 + y^2 + z^2 = u^2) →
  (∀ d : ℤ, d > 1 → ¬(d ∣ x ∧ d ∣ y ∧ d ∣ z ∧ d ∣ u)) →
  (∀ d : ℤ, d > 1 → ¬(d ∣ m ∧ d ∣ n ∧ d ∣ p ∧ d ∣ q)) :=
by sorry

end NUMINAMATH_CALUDE_spatial_pythagorean_quadruplet_l669_66994
