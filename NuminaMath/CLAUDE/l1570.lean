import Mathlib

namespace NUMINAMATH_CALUDE_speed_conversion_proof_l1570_157009

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℚ := 3.6

/-- Given speed in km/h -/
def given_speed_kmph : ℝ := 1.5428571428571427

/-- Speed in m/s as a fraction -/
def speed_mps : ℚ := 3/7

theorem speed_conversion_proof :
  (speed_mps : ℝ) * mps_to_kmph = given_speed_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_proof_l1570_157009


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1570_157097

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 500)
  (h2 : profit_percentage = 25) :
  selling_price / (1 + profit_percentage / 100) = 400 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1570_157097


namespace NUMINAMATH_CALUDE_function_properties_l1570_157067

/-- The function f(x) = ax ln x - x^2 - 2x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x - x^2 - 2 * x

/-- The derivative of f(x) --/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * (1 + Real.log x) - 2 * x - 2

/-- The function g(x) = f(x) + 2x --/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + 2 * x

theorem function_properties (a : ℝ) (x₁ x₂ : ℝ) (h1 : a > 0) :
  (∃ (x : ℝ), f_deriv 4 x = 4 * Real.log 2 - 2 ∧
    ∀ (y : ℝ), f_deriv 4 y ≤ 4 * Real.log 2 - 2) ∧
  (g a x₁ = 0 ∧ g a x₂ = 0 ∧ x₂ / x₁ > Real.exp 1 →
    Real.log a + Real.log (x₁ * x₂) > 3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1570_157067


namespace NUMINAMATH_CALUDE_number_of_elements_in_set_l1570_157098

theorem number_of_elements_in_set (S : ℝ) (n : ℕ) 
  (h1 : (S + 26) / n = 5)
  (h2 : (S + 36) / n = 6) :
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_elements_in_set_l1570_157098


namespace NUMINAMATH_CALUDE_least_difference_l1570_157072

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem least_difference (x y z : ℕ) : 
  x < y → 
  y < z → 
  y - x > 5 → 
  Even x → 
  x % 3 = 0 → 
  Odd y → 
  Odd z → 
  is_prime y → 
  y > 20 → 
  z % 5 = 0 → 
  1 < x → 
  x < 30 → 
  (∀ x' y' z' : ℕ, 
    x' < y' → 
    y' < z' → 
    y' - x' > 5 → 
    Even x' → 
    x' % 3 = 0 → 
    Odd y' → 
    Odd z' → 
    is_prime y' → 
    y' > 20 → 
    z' % 5 = 0 → 
    1 < x' → 
    x' < 30 → 
    z - x ≤ z' - x') → 
  z - x = 19 := by
sorry

end NUMINAMATH_CALUDE_least_difference_l1570_157072


namespace NUMINAMATH_CALUDE_root_difference_of_cubic_l1570_157064

theorem root_difference_of_cubic (x : ℝ → ℝ) :
  (∀ t, 81 * (x t)^3 - 162 * (x t)^2 + 81 * (x t) - 8 = 0) →
  (∃ a d, ∀ t, x t = a + d * t) →
  (∃ t₁ t₂, ∀ t, x t₁ ≤ x t ∧ x t ≤ x t₂) →
  x t₂ - x t₁ = 4 * Real.sqrt 6 / 9 := by
sorry

end NUMINAMATH_CALUDE_root_difference_of_cubic_l1570_157064


namespace NUMINAMATH_CALUDE_sum_of_even_factors_420_l1570_157075

def sumOfEvenFactors (n : ℕ) : ℕ := sorry

theorem sum_of_even_factors_420 :
  sumOfEvenFactors 420 = 1152 := by sorry

end NUMINAMATH_CALUDE_sum_of_even_factors_420_l1570_157075


namespace NUMINAMATH_CALUDE_problem_solution_l1570_157094

theorem problem_solution (x y : ℝ) : 
  x > 0 → 
  y > 0 → 
  x / 100 * y = 5 → 
  y = 2 * x + 10 → 
  x = 14 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1570_157094


namespace NUMINAMATH_CALUDE_expression_simplification_l1570_157010

theorem expression_simplification (x y b c d : ℝ) (h : c * y + d * x ≠ 0) :
  (c * y * (b * x^3 + 3 * b * x^2 * y + 3 * b * x * y^2 + b * y^3) + 
   d * x * (c * x^3 + 3 * c * x^2 * y + 3 * c * x * y^2 + c * y^3)) / 
  (c * y + d * x) = (c * x + y)^3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1570_157010


namespace NUMINAMATH_CALUDE_work_comparison_l1570_157031

/-- Represents the amount of work that can be done by a group of people in a given number of days -/
structure WorkCapacity where
  people : ℕ
  days : ℕ
  work : ℝ

/-- The work capacity is directly proportional to the number of people and days -/
axiom work_proportional {w1 w2 : WorkCapacity} : 
  w1.work / w2.work = (w1.people * w1.days : ℝ) / (w2.people * w2.days)

theorem work_comparison (w1 w2 : WorkCapacity) 
  (h1 : w1.people = 3 ∧ w1.days = 3)
  (h2 : w2.people = 8 ∧ w2.days = 3)
  (h3 : w2.work = 8 * w1.work) :
  w1.work = 3 * w1.work := by
  sorry

end NUMINAMATH_CALUDE_work_comparison_l1570_157031


namespace NUMINAMATH_CALUDE_basketball_team_chances_l1570_157073

/-- The starting percentage for making the basketball team for a 66-inch tall player -/
def starting_percentage : ℝ := 10

/-- The increase in percentage chance per inch above 66 inches -/
def increase_per_inch : ℝ := 10

/-- The height of the player with known chances -/
def known_height : ℝ := 68

/-- The chances of making the team for the player with known height -/
def known_chances : ℝ := 30

/-- The baseline height for the starting percentage -/
def baseline_height : ℝ := 66

theorem basketball_team_chances :
  starting_percentage =
    known_chances - (increase_per_inch * (known_height - baseline_height)) :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_chances_l1570_157073


namespace NUMINAMATH_CALUDE_percentage_of_green_caps_l1570_157004

/-- Calculates the percentage of green bottle caps -/
theorem percentage_of_green_caps 
  (total_caps : ℕ) 
  (red_caps : ℕ) 
  (h1 : total_caps = 125) 
  (h2 : red_caps = 50) 
  (h3 : red_caps ≤ total_caps) : 
  (((total_caps - red_caps) : ℚ) / total_caps) * 100 = 60 := by
  sorry

#check percentage_of_green_caps

end NUMINAMATH_CALUDE_percentage_of_green_caps_l1570_157004


namespace NUMINAMATH_CALUDE_inequality_proof_l1570_157003

theorem inequality_proof (x y z t : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) 
  (ht : 0 < t ∧ t < 1) : 
  Real.sqrt (x^2 + (1-t)^2) + Real.sqrt (y^2 + (1-x)^2) + 
  Real.sqrt (z^2 + (1-y)^2) + Real.sqrt (t^2 + (1-z)^2) < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1570_157003


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l1570_157025

theorem sum_of_A_and_B (A B : ℝ) :
  (∀ x : ℝ, x ≠ 7 → A / (x - 7) + B * (x + 2) = (-4 * x^2 + 16 * x + 28) / (x - 7)) →
  A + B = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l1570_157025


namespace NUMINAMATH_CALUDE_distance_to_origin_l1570_157020

/-- The distance from point P (-2, 4) to the origin (0, 0) is 2√5 -/
theorem distance_to_origin : 
  let P : ℝ × ℝ := (-2, 4)
  let O : ℝ × ℝ := (0, 0)
  Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1570_157020


namespace NUMINAMATH_CALUDE_amount_added_to_doubled_number_l1570_157028

theorem amount_added_to_doubled_number (original : ℝ) (total : ℝ) (h1 : original = 6.0) (h2 : 2 * original + (total - 2 * original) = 17) : 
  total - 2 * original = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_amount_added_to_doubled_number_l1570_157028


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1570_157036

theorem rationalize_denominator (x : ℝ) (h : x^4 = 81) :
  1 / (x + x^(1/4)) = 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1570_157036


namespace NUMINAMATH_CALUDE_bee_count_l1570_157096

/-- The number of bees initially in the hive -/
def initial_bees : ℕ := 16

/-- The number of bees that flew in -/
def new_bees : ℕ := 10

/-- The total number of bees in the hive -/
def total_bees : ℕ := initial_bees + new_bees

theorem bee_count : total_bees = 26 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l1570_157096


namespace NUMINAMATH_CALUDE_total_floor_area_l1570_157053

/-- The total floor area covered by square stone slabs -/
theorem total_floor_area (num_slabs : ℕ) (slab_length : ℝ) : 
  num_slabs = 30 → slab_length = 150 → 
  (num_slabs * (slab_length / 100)^2 : ℝ) = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_total_floor_area_l1570_157053


namespace NUMINAMATH_CALUDE_choose_two_correct_l1570_157078

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that choose_two gives the correct number of combinations -/
theorem choose_two_correct (n : ℕ) : choose_two n = Nat.choose n 2 :=
sorry

end NUMINAMATH_CALUDE_choose_two_correct_l1570_157078


namespace NUMINAMATH_CALUDE_womans_speed_in_still_water_l1570_157013

/-- The speed of a woman rowing a boat in still water, given her downstream performance. -/
theorem womans_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 60) 
  (h2 : distance = 500 / 1000) 
  (h3 : time = 9.99920006399488 / 3600) : 
  ∃ (speed : ℝ), abs (speed - 120.01800180018) < 0.00000000001 := by
  sorry

end NUMINAMATH_CALUDE_womans_speed_in_still_water_l1570_157013


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1570_157047

theorem complex_number_quadrant (z : ℂ) (h : z + z * Complex.I = 2 + 3 * Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1570_157047


namespace NUMINAMATH_CALUDE_cubic_greater_than_quadratic_plus_one_l1570_157032

theorem cubic_greater_than_quadratic_plus_one (x : ℝ) (h : x > 1) : 2 * x^3 > x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_greater_than_quadratic_plus_one_l1570_157032


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l1570_157076

/-- The circle with equation x²+(y-1)²=2 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 2}

/-- The point (1,2) on the circle -/
def Point : ℝ × ℝ := (1, 2)

/-- The proposed tangent line with equation x+y-3=0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 - 3 = 0}

theorem tangent_line_at_point :
  Point ∈ Circle ∧
  Point ∈ TangentLine ∧
  ∀ p ∈ Circle, p ≠ Point → p ∉ TangentLine :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l1570_157076


namespace NUMINAMATH_CALUDE_total_boys_and_girls_l1570_157087

theorem total_boys_and_girls (total_amount : ℕ) (boys : ℕ) (amount_per_boy : ℕ) (amount_per_girl : ℕ) : 
  total_amount = 460 → 
  boys = 33 → 
  amount_per_boy = 12 → 
  amount_per_girl = 8 → 
  ∃ (girls : ℕ), boys + girls = 41 ∧ total_amount = boys * amount_per_boy + girls * amount_per_girl :=
by
  sorry


end NUMINAMATH_CALUDE_total_boys_and_girls_l1570_157087


namespace NUMINAMATH_CALUDE_first_half_speed_l1570_157041

def total_distance : ℝ := 112
def total_time : ℝ := 5
def second_half_speed : ℝ := 24

theorem first_half_speed : 
  ∃ (v : ℝ), 
    v * (total_time - (total_distance / 2) / second_half_speed) = total_distance / 2 ∧ 
    v = 21 := by
  sorry

end NUMINAMATH_CALUDE_first_half_speed_l1570_157041


namespace NUMINAMATH_CALUDE_farm_tax_percentage_l1570_157008

theorem farm_tax_percentage (total_tax collection_tax : ℝ) : 
  total_tax > 0 → 
  collection_tax > 0 → 
  collection_tax ≤ total_tax → 
  (collection_tax / total_tax) * 100 = 12.5 → 
  total_tax = 3840 ∧ collection_tax = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_tax_percentage_l1570_157008


namespace NUMINAMATH_CALUDE_ellipse_right_triangle_distance_l1570_157033

def Ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

def LeftFocus (x y : ℝ) : Prop := x = -Real.sqrt 7 ∧ y = 0
def RightFocus (x y : ℝ) : Prop := x = Real.sqrt 7 ∧ y = 0

def RightTriangle (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0 ∨
  (x₁ - x₂) * (x₃ - x₂) + (y₁ - y₂) * (y₃ - y₂) = 0 ∨
  (x₁ - x₃) * (x₂ - x₃) + (y₁ - y₃) * (y₂ - y₃) = 0

theorem ellipse_right_triangle_distance (x y xf₁ yf₁ xf₂ yf₂ : ℝ) :
  Ellipse x y →
  LeftFocus xf₁ yf₁ →
  RightFocus xf₂ yf₂ →
  RightTriangle x y xf₁ yf₁ xf₂ yf₂ →
  |y| = 9/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_right_triangle_distance_l1570_157033


namespace NUMINAMATH_CALUDE_unique_solution_to_nested_equation_l1570_157021

theorem unique_solution_to_nested_equation : 
  ∃! x : ℝ, x = 2 * (2 * (2 * (2 * (2 * x - 1) - 1) - 1) - 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_nested_equation_l1570_157021


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l1570_157089

/-- Represents the number of male students in the group -/
def num_male : ℕ := 3

/-- Represents the number of female students in the group -/
def num_female : ℕ := 2

/-- Represents the number of students selected for the competition -/
def num_selected : ℕ := 2

/-- Represents the event of selecting at least one female student -/
def at_least_one_female : Set (Fin num_male × Fin num_female) := sorry

/-- Represents the event of selecting all male students -/
def all_male : Set (Fin num_male × Fin num_female) := sorry

/-- Theorem stating that the events are mutually exclusive and complementary -/
theorem events_mutually_exclusive_and_complementary :
  (at_least_one_female ∩ all_male = ∅) ∧
  (at_least_one_female ∪ all_male = Set.univ) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l1570_157089


namespace NUMINAMATH_CALUDE_academy_league_games_l1570_157034

/-- The number of teams in the Academy League -/
def num_teams : ℕ := 8

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- Calculates the total number of games in a season for the Academy League -/
def total_games (n : ℕ) (nc : ℕ) : ℕ :=
  (n * (n - 1)) + (n * nc)

/-- Theorem stating that the total number of games in the Academy League season is 104 -/
theorem academy_league_games :
  total_games num_teams non_conference_games = 104 := by
  sorry

end NUMINAMATH_CALUDE_academy_league_games_l1570_157034


namespace NUMINAMATH_CALUDE_total_dolls_l1570_157085

theorem total_dolls (jazmin_dolls geraldine_dolls : ℕ) 
  (h1 : jazmin_dolls = 1209) 
  (h2 : geraldine_dolls = 2186) : 
  jazmin_dolls + geraldine_dolls = 3395 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_l1570_157085


namespace NUMINAMATH_CALUDE_arithmetic_sequence_300th_term_l1570_157024

/-- 
Given an arithmetic sequence where:
- The first term is 6
- The common difference is 4
Prove that the 300th term is equal to 1202
-/
theorem arithmetic_sequence_300th_term : 
  let a : ℕ → ℕ := λ n => 6 + (n - 1) * 4
  a 300 = 1202 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_300th_term_l1570_157024


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1570_157086

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    ∀ x : ℝ, x^3 - 8*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  p + q = 27 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1570_157086


namespace NUMINAMATH_CALUDE_impossible_all_win_bets_l1570_157002

/-- Represents the outcome of a girl's jump -/
inductive JumpOutcome
  | Success
  | Failure

/-- Represents the three girls -/
inductive Girl
  | First
  | Second
  | Third

/-- The bet condition: one girl's success is equivalent to another girl's failure -/
def betCondition (g1 g2 : Girl) (outcomes : Girl → JumpOutcome) : Prop :=
  outcomes g1 = JumpOutcome.Success ↔ outcomes g2 = JumpOutcome.Failure

/-- The theorem stating it's impossible for all girls to win their bets -/
theorem impossible_all_win_bets :
  ¬∃ (outcomes : Girl → JumpOutcome),
    (betCondition Girl.First Girl.Second outcomes) ∧
    (betCondition Girl.Second Girl.Third outcomes) ∧
    (betCondition Girl.Third Girl.First outcomes) :=
by sorry

end NUMINAMATH_CALUDE_impossible_all_win_bets_l1570_157002


namespace NUMINAMATH_CALUDE_bus_stop_problem_l1570_157019

theorem bus_stop_problem (girls boys : ℕ) : 
  (girls - 15 = 5 * (boys - 45)) →
  (boys = 2 * (girls - 15)) →
  (girls = 40 ∧ boys = 50) :=
by sorry

end NUMINAMATH_CALUDE_bus_stop_problem_l1570_157019


namespace NUMINAMATH_CALUDE_floor_abs_sum_equals_57_l1570_157045

theorem floor_abs_sum_equals_57 : ⌊|(-57.85 : ℝ) + 0.1|⌋ = 57 := by sorry

end NUMINAMATH_CALUDE_floor_abs_sum_equals_57_l1570_157045


namespace NUMINAMATH_CALUDE_insufficient_information_for_unique_solution_l1570_157057

theorem insufficient_information_for_unique_solution :
  ∀ (x y z w : ℕ),
  x + y + z + w = 750 →
  10 * x + 20 * y + 50 * z + 100 * w = 27500 →
  ∃ (y' : ℕ), y ≠ y' ∧
  ∃ (x' z' w' : ℕ),
  x' + y' + z' + w' = 750 ∧
  10 * x' + 20 * y' + 50 * z' + 100 * w' = 27500 :=
by sorry

end NUMINAMATH_CALUDE_insufficient_information_for_unique_solution_l1570_157057


namespace NUMINAMATH_CALUDE_direct_proportion_is_straight_line_direct_proportion_passes_through_origin_l1570_157016

/-- A direct proportion function -/
def direct_proportion (k : ℝ) : ℝ → ℝ := fun x ↦ k * x

/-- The graph of a function -/
def graph (f : ℝ → ℝ) : Set (ℝ × ℝ) := {p | p.2 = f p.1}

theorem direct_proportion_is_straight_line (k : ℝ) :
  ∃ (a b : ℝ), ∀ x y, (x, y) ∈ graph (direct_proportion k) ↔ a * x + b * y = 0 :=
sorry

theorem direct_proportion_passes_through_origin (k : ℝ) :
  (0, 0) ∈ graph (direct_proportion k) :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_is_straight_line_direct_proportion_passes_through_origin_l1570_157016


namespace NUMINAMATH_CALUDE_sin_cos_2alpha_l1570_157093

def fixed_point : ℝ × ℝ := (4, 2)

def is_on_terminal_side (α : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = fixed_point.1 ∧ r * (Real.sin α) = fixed_point.2

theorem sin_cos_2alpha (α : ℝ) (h : is_on_terminal_side α) : 
  Real.sin (2 * α) + Real.cos (2 * α) = 7/5 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_2alpha_l1570_157093


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l1570_157038

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 4 / 5 →  -- ratio of angles is 4:5
  |a - b| = 10 := by  -- positive difference is 10°
sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l1570_157038


namespace NUMINAMATH_CALUDE_probability_is_one_half_l1570_157079

/-- Represents the class of a bus -/
inductive BusClass
| Upper
| Middle
| Lower

/-- Represents a sequence of three buses -/
def BusSequence := (BusClass × BusClass × BusClass)

/-- All possible bus sequences -/
def allSequences : List BusSequence := sorry

/-- Determines if Mr. Li boards an upper-class bus given a sequence -/
def boardsUpperClass (seq : BusSequence) : Bool := sorry

/-- The probability of Mr. Li boarding an upper-class bus -/
def probabilityOfUpperClass : ℚ := sorry

/-- Theorem stating that the probability of boarding an upper-class bus is 1/2 -/
theorem probability_is_one_half : probabilityOfUpperClass = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_half_l1570_157079


namespace NUMINAMATH_CALUDE_more_ones_than_zeros_mod_500_l1570_157080

/-- A function that returns the number of 1's in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in the binary representation of a natural number -/
def binaryDigits (n : ℕ) : ℕ := sorry

/-- The set of positive integers less than or equal to 1000 whose binary representation has more 1's than 0's -/
def M : Finset ℕ := sorry

theorem more_ones_than_zeros_mod_500 : M.card % 500 = 61 := by sorry

end NUMINAMATH_CALUDE_more_ones_than_zeros_mod_500_l1570_157080


namespace NUMINAMATH_CALUDE_little_krish_money_distribution_l1570_157001

-- Define the problem parameters
def initial_amount : ℚ := 200.50
def spent_on_sweets : ℚ := 35.25
def amount_left : ℚ := 114.85
def num_friends : ℕ := 2

-- Define the theorem
theorem little_krish_money_distribution :
  ∃ (amount_per_friend : ℚ),
    amount_per_friend = 25.20 ∧
    initial_amount - spent_on_sweets - (num_friends : ℚ) * amount_per_friend = amount_left :=
by
  sorry


end NUMINAMATH_CALUDE_little_krish_money_distribution_l1570_157001


namespace NUMINAMATH_CALUDE_row_arrangement_counts_l1570_157044

/-- Represents a person in the row -/
inductive Person : Type
| A | B | C | D | E

/-- A row is a permutation of five people -/
def Row := Fin 5 → Person

/-- Checks if A and B are adjacent with B to the right of A in a given row -/
def adjacent_AB (row : Row) : Prop :=
  ∃ i : Fin 4, row i = Person.A ∧ row (i.succ) = Person.B

/-- Checks if A, B, and C are in order from left to right in a given row -/
def ABC_in_order (row : Row) : Prop :=
  ∃ i j k : Fin 5, i < j ∧ j < k ∧ 
    row i = Person.A ∧ row j = Person.B ∧ row k = Person.C

/-- The main theorem to be proved -/
theorem row_arrangement_counts :
  (∃! (s : Finset Row), s.card = 24 ∧ ∀ row ∈ s, adjacent_AB row) ∧
  (∃! (s : Finset Row), s.card = 20 ∧ ∀ row ∈ s, ABC_in_order row) :=
sorry

end NUMINAMATH_CALUDE_row_arrangement_counts_l1570_157044


namespace NUMINAMATH_CALUDE_gravelling_rate_calculation_l1570_157061

/-- Given a rectangular lawn with two intersecting roads, calculate the rate per square meter for gravelling the roads. -/
theorem gravelling_rate_calculation (lawn_length lawn_width road_width total_cost : ℝ) 
  (h1 : lawn_length = 70)
  (h2 : lawn_width = 30)
  (h3 : road_width = 5)
  (h4 : total_cost = 1900) : 
  total_cost / ((lawn_length * road_width) + (lawn_width * road_width) - (road_width * road_width)) = 4 := by
  sorry

#check gravelling_rate_calculation

end NUMINAMATH_CALUDE_gravelling_rate_calculation_l1570_157061


namespace NUMINAMATH_CALUDE_joan_seashells_l1570_157039

def seashells_problem (initial_shells : ℕ) (given_away : ℕ) : ℕ :=
  initial_shells - given_away

theorem joan_seashells : seashells_problem 79 63 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l1570_157039


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l1570_157088

/-- The smallest possible next divisor after 221 for an even 4-digit number -/
theorem smallest_next_divisor_after_221 (n : ℕ) (h1 : 1000 ≤ n) (h2 : n < 10000) 
  (h3 : Even n) (h4 : n % 221 = 0) :
  ∃ (d : ℕ), d ∣ n ∧ d > 221 ∧ (∀ (x : ℕ), x ∣ n → x > 221 → x ≥ d) ∧ d = 442 := by
sorry


end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l1570_157088


namespace NUMINAMATH_CALUDE_gcd_count_for_product_180_l1570_157046

theorem gcd_count_for_product_180 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 180) :
  ∃! (s : Finset ℕ+), (∀ x ∈ s, ∃ (a' b' : ℕ+), Nat.gcd a' b' * Nat.lcm a' b' = 180 ∧ Nat.gcd a' b' = x) ∧ s.card = 8 :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_180_l1570_157046


namespace NUMINAMATH_CALUDE_max_value_polynomial_l1570_157007

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (max : ℝ), ∀ (a b : ℝ), a + b = 5 →
    a^5*b + a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 + a*b^5 ≤ max) ∧
  (x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 ≤ 22884) ∧
  (∃ (x₀ y₀ : ℝ), x₀ + y₀ = 5 ∧
    x₀^5*y₀ + x₀^4*y₀ + x₀^3*y₀ + x₀^2*y₀ + x₀*y₀ + x₀*y₀^2 + x₀*y₀^3 + x₀*y₀^4 + x₀*y₀^5 = 22884) :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l1570_157007


namespace NUMINAMATH_CALUDE_candidate_probability_l1570_157012

/-- Represents the probability space of job candidates -/
structure CandidateSpace where
  /-- Probability of having intermediate or advanced Excel skills -/
  excel_skills : ℝ
  /-- Probability of having intermediate Excel skills -/
  intermediate_excel : ℝ
  /-- Probability of having advanced Excel skills -/
  advanced_excel : ℝ
  /-- Probability of being willing to work night shifts among those with Excel skills -/
  night_shift_willing : ℝ
  /-- Probability of not being willing to work weekends among those willing to work night shifts -/
  weekend_unwilling : ℝ
  /-- Ensure probabilities are valid -/
  excel_skills_valid : excel_skills = intermediate_excel + advanced_excel
  excel_skills_prob : excel_skills = 0.45
  intermediate_excel_prob : intermediate_excel = 0.25
  advanced_excel_prob : advanced_excel = 0.20
  night_shift_willing_prob : night_shift_willing = 0.32
  weekend_unwilling_prob : weekend_unwilling = 0.60

/-- The main theorem to prove -/
theorem candidate_probability (cs : CandidateSpace) :
  cs.excel_skills * cs.night_shift_willing * cs.weekend_unwilling = 0.0864 := by
  sorry

end NUMINAMATH_CALUDE_candidate_probability_l1570_157012


namespace NUMINAMATH_CALUDE_necessary_condition_inequality_l1570_157091

theorem necessary_condition_inequality (a b c : ℝ) (hc : c ≠ 0) :
  (∀ a b c, c ≠ 0 → (a * c^2 > b * c^2 → a > b)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_condition_inequality_l1570_157091


namespace NUMINAMATH_CALUDE_solve_equation_l1570_157042

theorem solve_equation (m x : ℝ) : 
  (m * x + 1 = 2 * (m - x)) ∧ (|x + 2| = 0) → m = -|3/4| :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l1570_157042


namespace NUMINAMATH_CALUDE_books_sold_l1570_157090

/-- Given the initial number of books and the remaining number of books,
    prove that the number of books sold is their difference. -/
theorem books_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) 
    (h1 : initial = 115)
    (h2 : remaining = 37)
    (h3 : sold = initial - remaining) : 
  sold = 78 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l1570_157090


namespace NUMINAMATH_CALUDE_polynomial_derivative_sum_l1570_157056

theorem polynomial_derivative_sum (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_derivative_sum_l1570_157056


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1570_157081

theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : 
  (d^2 / 2 : ℝ) = 50 := by
  sorry

#check square_area_from_diagonal

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1570_157081


namespace NUMINAMATH_CALUDE_equation_solution_l1570_157037

theorem equation_solution : ∃ x : ℝ, 
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ 
  x = -9 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1570_157037


namespace NUMINAMATH_CALUDE_farm_animals_l1570_157000

/-- Given a farm with chickens and buffalos, prove the number of chickens -/
theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) (chicken_legs : ℕ) (buffalo_legs : ℕ) 
  (h_total_animals : total_animals = 13)
  (h_total_legs : total_legs = 44)
  (h_chicken_legs : chicken_legs = 2)
  (h_buffalo_legs : buffalo_legs = 4) :
  ∃ (chickens : ℕ) (buffalos : ℕ),
    chickens + buffalos = total_animals ∧
    chickens * chicken_legs + buffalos * buffalo_legs = total_legs ∧
    chickens = 4 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l1570_157000


namespace NUMINAMATH_CALUDE_students_without_A_l1570_157006

theorem students_without_A (total : ℕ) (lit_A : ℕ) (sci_A : ℕ) (both_A : ℕ) : 
  total - (lit_A + sci_A - both_A) = total - (lit_A + sci_A - both_A) :=
by sorry

#check students_without_A 40 10 18 6

end NUMINAMATH_CALUDE_students_without_A_l1570_157006


namespace NUMINAMATH_CALUDE_max_marbles_for_score_l1570_157026

/-- Represents the size of a marble -/
inductive MarbleSize
| Small
| Medium
| Large

/-- Represents a hole with its score -/
structure Hole :=
  (number : Nat)
  (score : Nat)

/-- Represents the game setup -/
structure GameSetup :=
  (holes : List Hole)
  (maxMarbles : Nat)
  (totalScore : Nat)

/-- Checks if a marble can go through a hole -/
def canGoThrough (size : MarbleSize) (hole : Hole) : Bool :=
  match size with
  | MarbleSize.Small => true
  | MarbleSize.Medium => hole.number ≥ 3
  | MarbleSize.Large => hole.number = 5

/-- Represents a valid game configuration -/
structure GameConfig :=
  (smallMarbles : List Hole)
  (mediumMarbles : List Hole)
  (largeMarbles : List Hole)

/-- Calculates the total score for a game configuration -/
def totalScore (config : GameConfig) : Nat :=
  (config.smallMarbles.map (·.score)).sum +
  (config.mediumMarbles.map (·.score)).sum +
  (config.largeMarbles.map (·.score)).sum

/-- Calculates the total number of marbles used in a game configuration -/
def totalMarbles (config : GameConfig) : Nat :=
  config.smallMarbles.length +
  config.mediumMarbles.length +
  config.largeMarbles.length

/-- The main theorem to prove -/
theorem max_marbles_for_score (setup : GameSetup) :
  (∃ (config : GameConfig),
    totalScore config = setup.totalScore ∧
    totalMarbles config ≤ setup.maxMarbles ∧
    (∀ (other : GameConfig),
      totalScore other = setup.totalScore →
      totalMarbles other ≤ totalMarbles config)) →
  (∃ (maxConfig : GameConfig),
    totalScore maxConfig = setup.totalScore ∧
    totalMarbles maxConfig = 14 ∧
    (∀ (other : GameConfig),
      totalScore other = setup.totalScore →
      totalMarbles other ≤ 14)) :=
by sorry

end NUMINAMATH_CALUDE_max_marbles_for_score_l1570_157026


namespace NUMINAMATH_CALUDE_trigonometric_expression_l1570_157017

theorem trigonometric_expression (α : Real) (m : Real) 
  (h : Real.tan (5 * Real.pi + α) = m) : 
  (Real.sin (α - 3 * Real.pi) + Real.cos (-α)) / (Real.sin α - Real.cos (Real.pi + α)) = (m + 1) / (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_l1570_157017


namespace NUMINAMATH_CALUDE_candidate_a_votes_l1570_157018

def total_votes : ℕ := 560000
def invalid_vote_percentage : ℚ := 15 / 100
def candidate_a_percentage : ℚ := 85 / 100

theorem candidate_a_votes : 
  ⌊(1 - invalid_vote_percentage) * candidate_a_percentage * total_votes⌋ = 404600 := by
  sorry

end NUMINAMATH_CALUDE_candidate_a_votes_l1570_157018


namespace NUMINAMATH_CALUDE_f_minimum_value_a_range_zeros_inequality_l1570_157048

noncomputable section

def f (x : ℝ) := x * Real.log (x + 1)

def g (a x : ℝ) := a * (x + 1 / (x + 1) - 1)

theorem f_minimum_value :
  ∃ (x_min : ℝ), f x_min = 0 ∧ ∀ x, f x ≥ f x_min :=
sorry

theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f x ≤ g a x) ↔ a ≥ 1 :=
sorry

theorem zeros_inequality (b : ℝ) (x₁ x₂ : ℝ) :
  f x₁ = b → f x₂ = b → 2 * |x₁ - x₂| > Real.sqrt (b^2 + 4*b) + 2 * Real.sqrt b - b :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_a_range_zeros_inequality_l1570_157048


namespace NUMINAMATH_CALUDE_intersection_M_N_l1570_157059

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1570_157059


namespace NUMINAMATH_CALUDE_fraction_equality_l1570_157083

theorem fraction_equality : (20 * 2 + 10) / (5 + 3 - 1) = 50 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1570_157083


namespace NUMINAMATH_CALUDE_lending_amount_calculation_l1570_157092

theorem lending_amount_calculation (P : ℝ) 
  (h1 : (P * 0.115 * 3) - (P * 0.10 * 3) = 157.5) : P = 3500 := by
  sorry

end NUMINAMATH_CALUDE_lending_amount_calculation_l1570_157092


namespace NUMINAMATH_CALUDE_dynamic_number_sum_divisible_by_three_l1570_157084

/-- A dynamic number is a four-digit positive integer where each digit is not 0,
    and the two-digit number formed by the tenth and unit places is twice
    the two-digit number formed by the thousandth and hundredth places. -/
def isDynamicNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∀ a b c d : ℕ,
    n = 1000 * a + 100 * b + 10 * c + d →
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    10 * c + d = 2 * (10 * a + b)

theorem dynamic_number_sum_divisible_by_three (a : ℕ) (h : 10 ≤ a ∧ a < 100) :
  ∃ k : ℕ, 102 * a + (200 * a + a) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_dynamic_number_sum_divisible_by_three_l1570_157084


namespace NUMINAMATH_CALUDE_half_percent_to_decimal_l1570_157029

/-- Expresses a percentage as a decimal fraction -/
def percentToDecimal (x : ℚ) : ℚ := x / 100

/-- The problem statement -/
theorem half_percent_to_decimal : percentToDecimal (1 / 2) = 0.005 := by
  sorry

end NUMINAMATH_CALUDE_half_percent_to_decimal_l1570_157029


namespace NUMINAMATH_CALUDE_b_share_is_3315_l1570_157052

/-- Calculates the share of a partner in a partnership based on investments and known share. -/
def calculate_share (investment_a investment_b investment_c share_a : ℚ) : ℚ :=
  (share_a * investment_b) / investment_a

/-- Theorem stating that given the investments and a's share, b's share is 3315. -/
theorem b_share_is_3315 (investment_a investment_b investment_c share_a : ℚ) 
  (h1 : investment_a = 11000)
  (h2 : investment_b = 15000)
  (h3 : investment_c = 23000)
  (h4 : share_a = 2431) :
  calculate_share investment_a investment_b investment_c share_a = 3315 := by
sorry

#eval calculate_share 11000 15000 23000 2431

end NUMINAMATH_CALUDE_b_share_is_3315_l1570_157052


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l1570_157065

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    if a_2 : a_3 = 5 : 2, then S_3 : S_5 = 3 : 2 -/
theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →
  (∀ n, a (n + 1) = a n + (a 2 - a 1)) →
  (a 2 : ℝ) / (a 3) = 5 / 2 →
  (S 3 : ℝ) / (S 5) = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l1570_157065


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1570_157058

theorem quadratic_inequality_solution_set : 
  {x : ℝ | x^2 - 3*x - 18 ≤ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 6} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1570_157058


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1570_157082

/-- The line l with equation y = kx - 3k intersects the circle C with equation x^2 + y^2 - 4x = 0 for any real k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  y = k * x - 3 * k ∧ x^2 + y^2 - 4 * x = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1570_157082


namespace NUMINAMATH_CALUDE_londolozi_lions_growth_l1570_157071

/-- The number of lion cubs born per month in Londolozi -/
def cubs_per_month : ℕ := sorry

/-- The initial number of lions in Londolozi -/
def initial_lions : ℕ := 100

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of lions that die per month -/
def lions_die_per_month : ℕ := 1

/-- The number of lions after one year -/
def lions_after_year : ℕ := 148

theorem londolozi_lions_growth :
  cubs_per_month * months_in_year - lions_die_per_month * months_in_year + initial_lions = lions_after_year ∧
  cubs_per_month = 5 := by sorry

end NUMINAMATH_CALUDE_londolozi_lions_growth_l1570_157071


namespace NUMINAMATH_CALUDE_fox_coins_l1570_157074

def bridge_crossings (initial_coins : ℕ) : ℕ → ℕ
  | 0 => initial_coins + 10
  | n + 1 => (2 * bridge_crossings initial_coins n) - 50

theorem fox_coins (x : ℕ) : x = 37 → bridge_crossings x 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fox_coins_l1570_157074


namespace NUMINAMATH_CALUDE_prob_not_face_card_is_ten_thirteenths_l1570_157054

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of face cards in a standard deck
def face_cards : ℕ := 12

-- Define the probability of not getting a face card
def prob_not_face_card : ℚ := (total_cards - face_cards) / total_cards

-- Theorem statement
theorem prob_not_face_card_is_ten_thirteenths :
  prob_not_face_card = 10 / 13 := by sorry

end NUMINAMATH_CALUDE_prob_not_face_card_is_ten_thirteenths_l1570_157054


namespace NUMINAMATH_CALUDE_binomial_100_100_l1570_157077

theorem binomial_100_100 : Nat.choose 100 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_100_100_l1570_157077


namespace NUMINAMATH_CALUDE_inverse_sum_lower_bound_l1570_157050

theorem inverse_sum_lower_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  1/x + 1/y ≥ 3 + 2*Real.sqrt 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 1 ∧ 1/x₀ + 1/y₀ = 3 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_sum_lower_bound_l1570_157050


namespace NUMINAMATH_CALUDE_steven_peach_apple_difference_l1570_157068

theorem steven_peach_apple_difference :
  ∀ (steven_peaches steven_apples jake_peaches jake_apples : ℕ),
    steven_peaches = 18 →
    steven_apples = 11 →
    jake_peaches = steven_peaches - 8 →
    jake_apples = steven_apples + 10 →
    steven_peaches - steven_apples = 7 := by
  sorry

end NUMINAMATH_CALUDE_steven_peach_apple_difference_l1570_157068


namespace NUMINAMATH_CALUDE_average_shirts_per_person_l1570_157005

/-- Represents the average number of shirts made by each person per day -/
def S : ℕ := sorry

/-- The number of employees -/
def employees : ℕ := 20

/-- The number of hours in a shift -/
def shift_hours : ℕ := 8

/-- The hourly wage in dollars -/
def hourly_wage : ℕ := 12

/-- The bonus per shirt made in dollars -/
def bonus_per_shirt : ℕ := 5

/-- The selling price of a shirt in dollars -/
def shirt_price : ℕ := 35

/-- The daily nonemployee expenses in dollars -/
def nonemployee_expenses : ℕ := 1000

/-- The daily profit in dollars -/
def daily_profit : ℕ := 9080

theorem average_shirts_per_person (S : ℕ) :
  S * (shirt_price * employees - bonus_per_shirt * employees) = 
  daily_profit + nonemployee_expenses + employees * shift_hours * hourly_wage →
  S = 20 := by sorry

end NUMINAMATH_CALUDE_average_shirts_per_person_l1570_157005


namespace NUMINAMATH_CALUDE_sqrt_product_eq_180_l1570_157030

theorem sqrt_product_eq_180 : Real.sqrt 75 * Real.sqrt 48 * (27 ^ (1/3 : ℝ)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_eq_180_l1570_157030


namespace NUMINAMATH_CALUDE_arrangements_part1_arrangements_part2_arrangements_part3_arrangements_part4_arrangements_part5_arrangements_part6_l1570_157035

/- Given: 3 male students and 4 female students -/
def num_male : ℕ := 3
def num_female : ℕ := 4
def total_students : ℕ := num_male + num_female

/- Part 1: Select 5 people and arrange them in a row -/
theorem arrangements_part1 : (Nat.choose total_students 5) * (Nat.factorial 5) = 2520 := by sorry

/- Part 2: Arrange them in two rows, with 3 in the front row and 4 in the back row -/
theorem arrangements_part2 : (Nat.factorial 7) * (Nat.factorial 6) * (Nat.factorial 5) = 5040 := by sorry

/- Part 3: Arrange all of them in a row, with a specific person not standing at the head or tail of the row -/
theorem arrangements_part3 : (Nat.factorial 6) * 5 = 3600 := by sorry

/- Part 4: Arrange all of them in a row, with all female students standing together -/
theorem arrangements_part4 : (Nat.factorial 4) * (Nat.factorial 4) = 576 := by sorry

/- Part 5: Arrange all of them in a row, with male students not standing next to each other -/
theorem arrangements_part5 : (Nat.factorial 4) * (Nat.factorial 5) * (Nat.factorial 3) = 1440 := by sorry

/- Part 6: Arrange all of them in a row, with exactly 3 people between person A and person B -/
theorem arrangements_part6 : (Nat.factorial 5) * 2 * (Nat.factorial 2) = 720 := by sorry

end NUMINAMATH_CALUDE_arrangements_part1_arrangements_part2_arrangements_part3_arrangements_part4_arrangements_part5_arrangements_part6_l1570_157035


namespace NUMINAMATH_CALUDE_greatest_sum_is_correct_l1570_157060

/-- The greatest possible sum of two consecutive integers whose product is less than 500 -/
def greatest_sum : ℕ := 43

/-- Predicate to check if two consecutive integers have a product less than 500 -/
def valid_pair (n : ℕ) : Prop := n * (n + 1) < 500

theorem greatest_sum_is_correct :
  (∀ n : ℕ, valid_pair n → n + (n + 1) ≤ greatest_sum) ∧
  (∃ n : ℕ, valid_pair n ∧ n + (n + 1) = greatest_sum) :=
sorry

end NUMINAMATH_CALUDE_greatest_sum_is_correct_l1570_157060


namespace NUMINAMATH_CALUDE_smaller_cube_weight_l1570_157011

/-- Represents the weight of a cube given its side length -/
def cube_weight (side_length : ℝ) : ℝ := sorry

theorem smaller_cube_weight :
  let small_side : ℝ := 1
  let large_side : ℝ := 2 * small_side
  let large_weight : ℝ := 56
  cube_weight small_side = 7 ∧ 
  cube_weight large_side = large_weight ∧
  cube_weight large_side = 8 * cube_weight small_side :=
by sorry

end NUMINAMATH_CALUDE_smaller_cube_weight_l1570_157011


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_28_l1570_157051

theorem largest_four_digit_congruent_to_17_mod_28 :
  ∃ (n : ℕ), n = 9982 ∧ n < 10000 ∧ n ≡ 17 [MOD 28] ∧
  ∀ (m : ℕ), m < 10000 ∧ m ≡ 17 [MOD 28] → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_28_l1570_157051


namespace NUMINAMATH_CALUDE_plane_parallel_criterion_l1570_157027

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the relation of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem plane_parallel_criterion
  (α β : Plane)
  (h : ∀ l : Line, line_in_plane l α → line_parallel_plane l β) :
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_plane_parallel_criterion_l1570_157027


namespace NUMINAMATH_CALUDE_license_plate_count_l1570_157099

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The total number of characters in the license plate -/
def total_chars : ℕ := 8

/-- The number of digits in the license plate -/
def num_plate_digits : ℕ := 5

/-- The number of letters in the license plate -/
def num_plate_letters : ℕ := 3

/-- The number of possible starting positions for the letter block -/
def letter_block_positions : ℕ := 6

/-- Calculates the total number of distinct license plates -/
def total_license_plates : ℕ :=
  letter_block_positions * num_digits ^ num_plate_digits * num_letters ^ num_plate_letters

theorem license_plate_count :
  total_license_plates = 10545600000 := by sorry

end NUMINAMATH_CALUDE_license_plate_count_l1570_157099


namespace NUMINAMATH_CALUDE_clock_resale_price_l1570_157014

theorem clock_resale_price (original_cost : ℝ) : 
  -- Conditions
  original_cost > 0 → 
  -- Store sold to collector for 20% more than original cost
  let collector_price := 1.2 * original_cost
  -- Store bought back at 50% of collector's price
  let buyback_price := 0.5 * collector_price
  -- Difference between original cost and buyback price is $100
  original_cost - buyback_price = 100 →
  -- Store resold at 80% profit on buyback price
  let final_price := buyback_price + 0.8 * buyback_price
  -- Theorem: The final selling price is $270
  final_price = 270 := by
sorry

end NUMINAMATH_CALUDE_clock_resale_price_l1570_157014


namespace NUMINAMATH_CALUDE_count_squarish_numbers_l1570_157063

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def first_two_digits (n : ℕ) : ℕ := n / 10000

def middle_two_digits (n : ℕ) : ℕ := (n / 100) % 100

def last_two_digits (n : ℕ) : ℕ := n % 100

def has_no_zero_digit (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 6 → (n / 10^d) % 10 ≠ 0

def is_squarish (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧
  is_perfect_square n ∧
  has_no_zero_digit n ∧
  is_perfect_square (first_two_digits n) ∧
  is_perfect_square (middle_two_digits n) ∧
  is_perfect_square (last_two_digits n)

theorem count_squarish_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_squarish n) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_count_squarish_numbers_l1570_157063


namespace NUMINAMATH_CALUDE_four_boxes_volume_l1570_157055

/-- The volume of a cube with edge length a -/
def cube_volume (a : ℝ) : ℝ := a^3

/-- The total volume of n identical cubes with edge length a -/
def total_volume (n : ℕ) (a : ℝ) : ℝ := n * (cube_volume a)

/-- Theorem: The total volume of four cubic boxes, each with an edge length of 5 feet, is 500 cubic feet -/
theorem four_boxes_volume : total_volume 4 5 = 500 := by
  sorry

end NUMINAMATH_CALUDE_four_boxes_volume_l1570_157055


namespace NUMINAMATH_CALUDE_marcus_oven_capacity_l1570_157022

/-- Given that Marcus bakes 7 batches of pies, drops 8 pies, and has 27 pies left,
    prove that he can fit 5 pies in his oven at once. -/
theorem marcus_oven_capacity :
  ∀ x : ℕ,
    (7 * x - 8 = 27) →
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_marcus_oven_capacity_l1570_157022


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1570_157069

theorem trigonometric_simplification (α : ℝ) (h : 2 * Real.sin α ^ 2 * (2 * α) - Real.sin (4 * α) ≠ 0) :
  (1 - Real.cos (2 * α)) * Real.cos (π / 4 + α) / (2 * Real.sin α ^ 2 * (2 * α) - Real.sin (4 * α)) =
  -Real.sqrt 2 / 4 * Real.tan α :=
sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1570_157069


namespace NUMINAMATH_CALUDE_dividing_line_coefficients_l1570_157062

/-- A region formed by nine unit circles tightly packed in the first quadrant -/
def Region : Set (ℝ × ℝ) := sorry

/-- A line with slope 2 that divides the region into two equal-area parts -/
def dividingLine : Set (ℝ × ℝ) := sorry

/-- The coefficients of the line equation ax = by + c -/
def lineCoefficients : ℕ × ℕ × ℕ := sorry

theorem dividing_line_coefficients :
  ∀ (a b c : ℕ),
    lineCoefficients = (a, b, c) →
    dividingLine = {(x, y) | a * x = b * y + c} →
    (∀ (x y : ℝ), (x, y) ∈ dividingLine → y = 2 * x) →
    Nat.gcd a (Nat.gcd b c) = 1 →
    a^2 + b^2 + c^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dividing_line_coefficients_l1570_157062


namespace NUMINAMATH_CALUDE_max_lcm_20_and_others_l1570_157043

theorem max_lcm_20_and_others : 
  let lcm_list := [Nat.lcm 20 2, Nat.lcm 20 4, Nat.lcm 20 6, Nat.lcm 20 8, Nat.lcm 20 10, Nat.lcm 20 12]
  List.maximum lcm_list = some 60 := by sorry

end NUMINAMATH_CALUDE_max_lcm_20_and_others_l1570_157043


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1570_157066

-- Define variables
variable (a b x y : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : 2*a - (4*a + 5*b) + 2*(3*a - 4*b) = 4*a - 13*b := by
  sorry

-- Theorem for the second expression
theorem simplify_expression_2 : 5*x^2 - 2*(3*y^2 - 5*x^2) + (-4*y^2 + 7*x*y) = 15*x^2 - 10*y^2 + 7*x*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1570_157066


namespace NUMINAMATH_CALUDE_crop_planting_arrangement_l1570_157023

theorem crop_planting_arrangement (n : ℕ) (h : n = 10) : 
  (Finset.sum (Finset.range (n - 6)) (λ i => n - i - 6)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_crop_planting_arrangement_l1570_157023


namespace NUMINAMATH_CALUDE_ship_supplies_problem_l1570_157015

theorem ship_supplies_problem (initial_supply : ℝ) 
  (remaining_supply : ℝ) (h1 : initial_supply = 400) 
  (h2 : remaining_supply = 96) : 
  ∃ x : ℝ, x = 2/5 ∧ 
    remaining_supply = (2/5) * (1 - x) * initial_supply :=
by sorry

end NUMINAMATH_CALUDE_ship_supplies_problem_l1570_157015


namespace NUMINAMATH_CALUDE_function_value_problem_l1570_157070

theorem function_value_problem (f : ℝ → ℝ) 
  (h : ∀ x, 2 * f x + f (-x) = 3 * x + 2) : 
  f 2 = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_function_value_problem_l1570_157070


namespace NUMINAMATH_CALUDE_gcd_228_1995_l1570_157095

theorem gcd_228_1995 : Int.gcd 228 1995 = 57 := by sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l1570_157095


namespace NUMINAMATH_CALUDE_opera_ticket_price_increase_l1570_157040

theorem opera_ticket_price_increase (old_price new_price : ℝ) 
  (h1 : old_price = 85)
  (h2 : new_price = 102) : 
  (new_price - old_price) / old_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_opera_ticket_price_increase_l1570_157040


namespace NUMINAMATH_CALUDE_negative_four_is_square_root_of_sixteen_l1570_157049

-- Definition of square root
def is_square_root (x y : ℝ) : Prop := x * x = y

-- Theorem to prove
theorem negative_four_is_square_root_of_sixteen :
  is_square_root (-4) 16 := by
  sorry


end NUMINAMATH_CALUDE_negative_four_is_square_root_of_sixteen_l1570_157049
